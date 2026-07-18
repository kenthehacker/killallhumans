"""Safety-gated runner for AI Grand Prix VQ2 build 3385.

The current qualifier build exposes camera, ``HIGHRES_IMU``, race status,
actuator status, collision messages, and heartbeat, but no pose or usable gate
map.  This runner therefore performs only bounded training stages:

``preflight``
    Receive and validate every required stream.  Sends no arm or flight target.
``sign-id``
    Apply two very small, below-hover roll/pitch rate pulses, then stop/reset.
``hover``
    Level and hold for 2.5 seconds, then stop/reset.
``gate0``
    Approach only the first visible gate and reset immediately when race status
    advances from gate 0 to gate 1.

Every powered stage proves both the race and IMU clocks rolled back after
``SIM_RESET``, calibrates a gyro-only flight estimator during the countdown,
waits past GO, confirms arming on a newer heartbeat, and runs a latched
watchdog at 50 Hz.  It never consumes the placeholder pose fields present in
``TelemetryState`` and never calls attitude-target mode.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from competition.adapter import AttitudeRateCommand, Quaternion
from competition.aigp_mavlink import (
    DEFAULT_MAVLINK_URL,
    AIGPMavlinkAdapter,
    _attitude_error_body_rates,
)
from competition.vq2_vision import VQ2VisionThread
from estimation.imu_attitude import (
    AttitudeEstimate,
    ImuAttitudeConfig,
    ImuAttitudeEstimator,
)
from gate_detection.src.gate_detector import GateDetection
from gate_detection.src.vq2_detector import VQ2GateDetector


logger = logging.getLogger("aigp.vq2")

CONTROL_HZ = 50.0
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ

MAX_HEARTBEAT_AGE_S = 1.5
MAX_IMU_AGE_S = 0.050
MAX_RACE_AGE_S = 0.40
MAX_ACTUATOR_AGE_S = 0.10
MAX_VISION_AGE_S = 0.10
MAX_TARGET_LOSS_S = 0.25

CROSSING_TARGET_LOSS_S = 0.08
CROSSING_STATUS_TIMEOUT_S = 0.40
CROSSING_MIN_AREA_RATIO = 25.0
CROSSING_MIN_WIDTH_PX = 512

MAX_BENIGN_PAD_CONTACTS = 12
MAX_BENIGN_PAD_IMPULSE = 0.05

MAX_ROLL_RAD = math.radians(25.0)
MIN_PITCH_RAD = math.radians(-35.0)
MAX_PITCH_RAD = math.radians(10.0)
MAX_BODY_RATE_RAD_S = 2.0
IMMEDIATE_MAX_BODY_RATE_RAD_S = 3.0
MAX_COMMAND_RATE_RAD_S = 0.25

RESET_RACE_DROP_MS = 500
RESET_IMU_DROP_US = 100_000
RESET_PROOF_TIMEOUT_S = 2.8
RESET_MAX_ATTEMPTS = 4


class SafetyAbort(RuntimeError):
    """A latched no-recovery flight watchdog failure."""


def next_control_deadline(
    previous_deadline_s: float,
    now_s: float,
    period_s: float = CONTROL_PERIOD_S,
) -> float:
    """Pace setpoints without replaying missed ticks after a loop stall.

    A stale absolute schedule can emit back-to-back commands while catching up.
    Keep the nominal grid only when it also leaves a complete period after the
    most recent send; otherwise drop the missed ticks.
    """

    values = (previous_deadline_s, now_s, period_s)
    if not all(math.isfinite(value) for value in values) or period_s <= 0.0:
        raise ValueError("control pacing inputs must be finite and period_s > 0")
    return max(previous_deadline_s + period_s, now_s + period_s)


@dataclass(frozen=True)
class GateTarget:
    frame_id: int
    sim_time_ns: int
    received_monotonic_s: float
    center_x: int
    center_y: int
    bbox: Tuple[int, int, int, int]
    confidence: float

    @property
    def bbox_area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    def age_s(self, now: Optional[float] = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.received_monotonic_s)


@dataclass(frozen=True)
class ResetProof:
    attempt: int
    pre_race_boot_ms: int
    post_race_boot_ms: int
    pre_imu_us: int
    post_imu_us: int
    advancing_race_samples: int
    advancing_imu_samples: int
    countdown_observed: bool


@dataclass(frozen=True)
class StartContext:
    spawn_roll_rad: float
    spawn_pitch_rad: float
    initial_gate_x: int
    initial_gate_y: int
    initial_gate_area: int
    go_boot_ms: int


@dataclass(frozen=True)
class StageResult:
    stage: str
    success: bool
    reason: str
    duration_s: float
    gate_index_before: Optional[int] = None
    gate_index_after: Optional[int] = None
    cleanup_confirmed: bool = False
    details: Optional[Dict[str, Any]] = None


def clock_rolled_back(pre_value: int, current_value: int, margin: int) -> bool:
    """Whether a simulator clock is authoritatively below its prior epoch."""

    return int(current_value) < int(pre_value) - int(margin)


def clock_within_epoch_envelope(
    anchor_value: int,
    current_value: int,
    elapsed_s: float,
    *,
    units_per_second: float,
    slack: int,
) -> bool:
    """Reject delayed packets whose clock is impossible in the proved epoch."""

    maximum = int(anchor_value) + int(max(0.0, elapsed_s) * units_per_second) + int(slack)
    return int(current_value) <= maximum


def select_primary_gate(
    detections: Iterable[GateDetection],
) -> Optional[GateDetection]:
    """Select the largest plausible gate using only pixel geometry.

    Metric distance and detector-provided corners are intentionally ignored:
    their current scale is a placeholder.  Near-square filtering rejects the
    cyan racing line, starting lights, and thin fragments.
    """

    candidates: List[GateDetection] = []
    for detection in detections:
        _x, _y, width, height = detection.bbox
        if width < 20 or height < 20:
            continue
        short = min(width, height)
        long = max(width, height)
        if short <= 0 or long / short > 1.85:
            continue
        if not math.isfinite(detection.confidence) or detection.confidence < 0.10:
            continue
        candidates.append(detection)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.bbox[2] * item.bbox[3])


class GateTargetTracker:
    """Small temporal gate filter for the first bounded vision-only run."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.target: Optional[GateTarget] = None
        self.consecutive = 0
        self._last_frame_id: Optional[int] = None

    def update(
        self,
        detections: Iterable[GateDetection],
        *,
        frame_id: int,
        sim_time_ns: int,
        received_monotonic_s: float,
    ) -> Optional[GateTarget]:
        if self._last_frame_id == int(frame_id):
            return self.target
        self._last_frame_id = int(frame_id)
        selected = select_primary_gate(detections)
        if selected is None:
            self.consecutive = 0
            return None

        candidate = GateTarget(
            frame_id=int(frame_id),
            sim_time_ns=int(sim_time_ns),
            received_monotonic_s=float(received_monotonic_s),
            center_x=int(selected.center_x),
            center_y=int(selected.center_y),
            bbox=tuple(int(value) for value in selected.bbox),
            confidence=float(selected.confidence),
        )
        continuous = False
        if self.target is not None:
            dx = candidate.center_x - self.target.center_x
            dy = candidate.center_y - self.target.center_y
            center_jump = math.hypot(dx, dy)
            prior_area = max(1, self.target.bbox_area)
            area_ratio = candidate.bbox_area / prior_area
            continuous = center_jump <= 100.0 and 0.20 <= area_ratio <= 5.0
            if not continuous:
                # Do not let a one-frame red fragment replace the tracked near
                # gate.  Build 3385 emits exactly this failure at very close
                # range when the frame clips the image boundary.
                self.consecutive = 0
                return None
        self.consecutive = self.consecutive + 1 if continuous else 1
        self.target = candidate
        return candidate

    def fresh(self, max_age_s: float, now: Optional[float] = None) -> bool:
        return self.target is not None and self.target.age_s(now) <= max_age_s


def gate_vertical_reference_px(
    initial_gate_y: float,
    spawn_pitch_rad: float,
    target_pitch_rad: float,
    *,
    focal_length_px: float = 320.0,
) -> float:
    """Expected gate row after changing pitch without translating."""

    delta = float(target_pitch_rad) - float(spawn_pitch_rad)
    delta = max(math.radians(-35.0), min(math.radians(35.0), delta))
    return float(initial_gate_y) + float(focal_length_px) * math.tan(delta)


def gate_control_center_y_px(
    target: GateTarget,
    image_height: int = 360,
    *,
    previous_center_y: Optional[float] = None,
) -> float:
    """Estimate outer-gate center when its lower edge clips the image.

    A head-on VQ2 gate is square.  Once the lower edge leaves the 360 px frame,
    the raw bbox center is biased sharply upward.  While left/right remain
    visible, width is the reliable side length, so infer the missing vertical
    half from it.  Otherwise retain the measured center and let freshness/
    continuity guards stop on ambiguous fragments.
    """

    x, y, width, height = target.bbox
    top_clipped = y <= 2
    bottom_clipped = y + height >= int(image_height) - 2
    horizontally_visible = x > 1 and x + width < 640 - 2
    if top_clipped and bottom_clipped:
        # No vertical edge remains from which to infer the square center.  A
        # fully clipped close-range frame must not create a fictitious jump;
        # hold the last bottom-only estimate for the few frames until pass.
        if previous_center_y is not None and math.isfinite(previous_center_y):
            return float(previous_center_y)
        return float(target.center_y)
    if bottom_clipped and horizontally_visible and width > height:
        return float(y) + 0.5 * float(width)
    return float(target.center_y)


def gate_vertical_thrust(control_y: float, control_y_rate: float) -> float:
    """Pixel-space PD for gate altitude, bounded to the validated envelope."""

    if not math.isfinite(control_y) or not math.isfinite(control_y_rate):
        raise SafetyAbort("non-finite gate vertical control input")
    proportional = 0.040 * max(-1.0, min(1.0, (180.0 - control_y) / 90.0))
    damping = -0.00070 * max(-300.0, min(300.0, control_y_rate))
    return max(0.21, min(0.32, 0.275 + proportional + damping))


def is_close_gate_crossing_candidate(
    target: GateTarget,
    *,
    initial_gate_area: int,
    control_y: float,
) -> bool:
    """Whether target loss may be the aperture expanding beyond the camera.

    This does not infer a pass. It only permits a bounded wait for the next
    authoritative race-status packet after a centered, monotonically expanded
    gate has clipped both vertical image edges.
    """

    if initial_gate_area <= 0 or not math.isfinite(control_y):
        return False
    _x, y, width, height = target.bbox
    return bool(
        target.bbox_area >= CROSSING_MIN_AREA_RATIO * initial_gate_area
        and width >= CROSSING_MIN_WIDTH_PX
        and y <= 2
        and y + height >= 358
        and abs(target.center_x - 320.0) <= 0.15 * width
        and abs(control_y - 180.0) <= 75.0
    )


def crossing_status_decision(
    *,
    baseline_race_boot_ms: int,
    current_race_boot_ms: int,
    active_gate_index: int,
    elapsed_s: float,
    timeout_s: float = CROSSING_STATUS_TIMEOUT_S,
) -> str:
    """Classify the bounded authoritative-status wait after visual commit."""

    if (
        not math.isfinite(elapsed_s)
        or elapsed_s < 0.0
        or not math.isfinite(timeout_s)
        or timeout_s <= 0.0
    ):
        raise ValueError("crossing status timing must be finite with timeout_s > 0")
    if active_gate_index not in (0, 1):
        return "invalid_gate_index"
    if current_race_boot_ms < baseline_race_boot_ms:
        return "race_clock_regressed"
    if current_race_boot_ms == baseline_race_boot_ms:
        return "status_timeout" if elapsed_s >= timeout_s else "waiting"
    if active_gate_index == 1:
        return "passed"
    if active_gate_index == 0:
        return "not_credited"
    raise AssertionError("unreachable gate crossing decision")


def attitude_rate_command(
    estimate: AttitudeEstimate,
    *,
    target_roll_rad: float,
    target_pitch_rad: float,
    thrust: float,
) -> AttitudeRateCommand:
    """Conservative roll/pitch attitude loop with yaw deliberately disabled."""

    desired = Quaternion.from_euler(
        float(target_roll_rad),
        float(target_pitch_rad),
        estimate.yaw,
    )
    roll_rate, pitch_rate, _yaw_rate = _attitude_error_body_rates(
        estimate.orientation,
        desired,
        omega=estimate.body_rates,
        kp=(1.0, 0.5, 0.0),
        kd=(0.4, 0.2, 0.0),
        max_rate=(MAX_COMMAND_RATE_RAD_S,) * 3,
    )
    command = AttitudeRateCommand(
        roll_rate=float(roll_rate),
        pitch_rate=float(pitch_rate),
        yaw_rate=0.0,
        thrust=float(thrust),
    )
    validate_command(command)
    return command


def validate_command(command: AttitudeRateCommand) -> None:
    values = (
        command.roll_rate,
        command.pitch_rate,
        command.yaw_rate,
        command.thrust,
    )
    if not all(math.isfinite(value) for value in values):
        raise SafetyAbort("non-finite command")
    if max(abs(command.roll_rate), abs(command.pitch_rate), abs(command.yaw_rate)) > (
        MAX_COMMAND_RATE_RAD_S + 1e-9
    ):
        raise SafetyAbort("commanded body rate exceeded conservative clamp")
    if not 0.0 <= command.thrust <= 0.35:
        raise SafetyAbort("commanded thrust exceeded conservative VQ2 envelope")


def is_benign_pad_contact(collision: Dict[str, Any]) -> bool:
    """Exact low-energy spawn-pad contact class observed during motor preload."""

    try:
        return (
            collision.get("id") == 1002
            and int(collision.get("threat_level", 99)) <= 1
            and abs(float(collision.get("impulse", math.inf))) <= 0.01
        )
    except (TypeError, ValueError, OverflowError):
        return False


class JsonlRecorder:
    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path).resolve() if path else None
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if str(self.path).endswith(".gz"):
                self._handle = gzip.open(self.path, "wt", encoding="utf-8")
            else:
                self._handle = self.path.open("w", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            return
        row = {"event": event, "wall_time_ns": time.time_ns(), **fields}
        self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class VQ2Runner:
    def __init__(
        self,
        adapter: AIGPMavlinkAdapter,
        vision: VQ2VisionThread,
        *,
        recorder: Optional[JsonlRecorder] = None,
    ) -> None:
        if adapter.enable_vision:
            raise ValueError("VQ2Runner requires adapter vision disabled")
        if adapter.telemetry_mode != "imu" or adapter.fetch_track_on_connect:
            raise ValueError("VQ2Runner requires track-free IMU telemetry mode")
        self.adapter = adapter
        self.vision = vision
        self.detector = VQ2GateDetector()
        self.tracker = GateTargetTracker()
        self.recorder = recorder or JsonlRecorder(None)

        config = ImuAttitudeConfig(
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
        if config.gravity_correction_kp != 0.0 or config.gyro_bias_ki != 0.0:
            raise AssertionError("powered VQ2 estimator must be pure gyro after bootstrap")
        self.estimator = ImuAttitudeEstimator(config)

        self.estimate: Optional[AttitudeEstimate] = None
        self._last_imu_us: Optional[int] = None
        self._last_imu_advance_s = 0.0
        self._last_race_boot_ms: Optional[int] = None
        self._last_race_advance_s = 0.0
        self._last_frame_sim_ns: Optional[int] = None
        self._imu_regressed = False
        self._race_regressed = False
        self._imu_forward_jump = False
        self._race_forward_jump = False
        self._epoch_imu_anchor_us: Optional[int] = None
        self._epoch_race_anchor_ms: Optional[int] = None
        self._epoch_anchor_monotonic_s = 0.0
        self._countdown_observed = False
        self._detection_error: Optional[str] = None
        self._estimator_unhealthy_latched = False
        self._estimator_failure_reason: Optional[str] = None
        self._benign_pad_contact_count = 0
        self._benign_pad_contact_impulse = 0.0
        self._high_rate_samples = 0
        self._abort_latched = False

    def _clear_epoch_state(self) -> None:
        self.estimator.reset()
        self.estimate = None
        self._last_imu_us = None
        self._last_imu_advance_s = 0.0
        self._last_race_boot_ms = None
        self._last_race_advance_s = 0.0
        self._last_frame_sim_ns = None
        self._imu_regressed = False
        self._race_regressed = False
        self._imu_forward_jump = False
        self._race_forward_jump = False
        self._epoch_imu_anchor_us = None
        self._epoch_race_anchor_ms = None
        self._epoch_anchor_monotonic_s = 0.0
        self._countdown_observed = False
        self._detection_error = None
        self._estimator_unhealthy_latched = False
        self._estimator_failure_reason = None
        self._benign_pad_contact_count = 0
        self._benign_pad_contact_impulse = 0.0
        self._high_rate_samples = 0
        self.tracker.reset()

    def _sample(self) -> None:
        now = time.monotonic()
        telemetry = self.adapter.latest_telemetry
        drain_imu = getattr(self.adapter, "drain_imu_samples", None)
        if callable(drain_imu):
            imu_samples = drain_imu()
        else:
            imu = telemetry.imu if telemetry is not None else None
            imu_samples = [imu] if imu is not None else []
        for imu in imu_samples:
            stamp = int(imu.timestamp_us)
            if (
                self._epoch_imu_anchor_us is not None
                and not clock_within_epoch_envelope(
                    self._epoch_imu_anchor_us,
                    stamp,
                    now - self._epoch_anchor_monotonic_s,
                    units_per_second=1_000_000.0,
                    slack=500_000,
                )
            ):
                self._imu_forward_jump = True
            elif self._last_imu_us is None or stamp > self._last_imu_us:
                estimator_was_ready = self.estimator.is_ready
                estimate = self.estimator.update_imu(imu)
                self._last_imu_us = stamp
                self._last_imu_advance_s = now
                if estimate is None and estimator_was_ready:
                    # Transport freshness is not estimator health. Once the
                    # estimator is ready, any rejected newer sample must latch
                    # an abort instead of letting an old estimate look current.
                    self._estimator_unhealthy_latched = True
                    self._estimator_failure_reason = (
                        self.estimator.last_rejection_reason or "sample rejected"
                    )
                elif estimate is not None:
                    self.estimate = estimate
                    if not estimate.healthy:
                        self._estimator_unhealthy_latched = True
                        self._estimator_failure_reason = estimate.reason or "unhealthy estimate"
            elif stamp < self._last_imu_us:
                self._imu_regressed = True

        race = self.adapter.race_status
        if race is not None:
            boot = int(race.sim_boot_time_ms)
            if (
                self._epoch_race_anchor_ms is not None
                and not clock_within_epoch_envelope(
                    self._epoch_race_anchor_ms,
                    boot,
                    now - self._epoch_anchor_monotonic_s,
                    units_per_second=1_000.0,
                    slack=700,
                )
            ):
                self._race_forward_jump = True
            elif self._last_race_boot_ms is None or boot > self._last_race_boot_ms:
                self._last_race_boot_ms = boot
                self._last_race_advance_s = now
                if race.race_start_boot_time_ms < 0 or boot < race.race_start_boot_time_ms:
                    self._countdown_observed = True
            elif boot < self._last_race_boot_ms:
                self._race_regressed = True

        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if snapshot is not None and snapshot.sim_time_ns != self._last_frame_sim_ns:
            self._last_frame_sim_ns = snapshot.sim_time_ns
            try:
                detections = self.detector.detect(snapshot.camera_frame.image)
                self.tracker.update(
                    detections,
                    frame_id=snapshot.frame_id,
                    sim_time_ns=snapshot.sim_time_ns,
                    received_monotonic_s=snapshot.received_monotonic_s,
                )
            except Exception as exc:  # OpenCV errors must fail closed in flight.
                self._detection_error = f"{type(exc).__name__}: {exc}"

    def _stream_failures(
        self,
        *,
        require_estimator: bool,
        require_target: bool,
        require_armed: bool,
    ) -> List[str]:
        now = time.monotonic()
        failures: List[str] = []
        if self.adapter.heartbeat_age_s > MAX_HEARTBEAT_AGE_S:
            failures.append(f"heartbeat stale ({self.adapter.heartbeat_age_s:.3f}s)")
        if self.adapter.imu_age_s > MAX_IMU_AGE_S:
            failures.append(f"IMU receive stale ({self.adapter.imu_age_s:.3f}s)")
        if now - self._last_imu_advance_s > MAX_IMU_AGE_S:
            failures.append("IMU timestamp not advancing")
        if self.adapter.race_status_age_s > MAX_RACE_AGE_S:
            failures.append(f"race status stale ({self.adapter.race_status_age_s:.3f}s)")
        if now - self._last_race_advance_s > MAX_RACE_AGE_S:
            failures.append("race clock not advancing")
        if self.adapter.actuator_age_s > MAX_ACTUATOR_AGE_S:
            failures.append(f"actuator status stale ({self.adapter.actuator_age_s:.3f}s)")
        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if snapshot is None:
            failures.append("camera stale or unavailable")
        if self._detection_error is not None:
            failures.append(f"gate detector failed: {self._detection_error}")
        if self._imu_regressed:
            failures.append("IMU timestamp regressed")
        if self._race_regressed:
            failures.append("race clock regressed")
        if self._imu_forward_jump:
            failures.append("IMU timestamp jumped outside proved reset epoch")
        if self._race_forward_jump:
            failures.append("race clock jumped outside proved reset epoch")
        if self._estimator_unhealthy_latched:
            failures.append(
                "attitude estimator failure latched "
                f"({self._estimator_failure_reason or 'unknown reason'})"
            )
        if require_estimator:
            if not self.estimator.is_ready or self.estimate is None:
                failures.append("attitude estimator not ready")
            elif not self.estimate.healthy:
                failures.append(f"attitude estimator unhealthy ({self.estimate.reason})")
        if require_target:
            if not self.tracker.fresh(MAX_TARGET_LOSS_S, now):
                failures.append("primary gate target lost")
        if require_armed and not self.adapter.is_armed:
            failures.append("vehicle no longer armed")
        return failures

    def _watchdog(
        self,
        *,
        require_target: bool = True,
        allow_benign_pad_contact: bool = False,
        enforce_benign_pad_budget: bool = False,
    ) -> None:
        if self._abort_latched:
            raise SafetyAbort("abort already latched")
        failures = self._stream_failures(
            require_estimator=True,
            require_target=require_target,
            require_armed=True,
        )
        collisions = self.adapter.drain_collisions()
        if collisions:
            harmful = []
            for collision in collisions:
                benign_pad = allow_benign_pad_contact and is_benign_pad_contact(collision)
                if benign_pad:
                    self._benign_pad_contact_count += 1
                    self._benign_pad_contact_impulse += abs(float(collision["impulse"]))
                    self.recorder.emit(
                        "benign_pad_contact",
                        collision=collision,
                        cumulative_count=self._benign_pad_contact_count,
                        cumulative_impulse=self._benign_pad_contact_impulse,
                    )
                else:
                    harmful.append(collision)
            if harmful:
                failures.append(f"collision reported: {harmful!r}")
            if enforce_benign_pad_budget and (
                self._benign_pad_contact_count > MAX_BENIGN_PAD_CONTACTS
                or self._benign_pad_contact_impulse > MAX_BENIGN_PAD_IMPULSE
            ):
                failures.append(
                    "repeated pad contacts exceeded launch budget "
                    f"(count={self._benign_pad_contact_count}, "
                    f"impulse={self._benign_pad_contact_impulse:.3f})"
                )
        if self.estimate is not None:
            roll, pitch, _yaw = self.estimate.orientation.to_euler()
            rates = self.estimate.body_rates
            if not all(math.isfinite(value) for value in (roll, pitch, *rates)):
                failures.append("non-finite estimated state")
            if abs(roll) > MAX_ROLL_RAD:
                failures.append(f"roll limit exceeded ({math.degrees(roll):.1f}deg)")
            if pitch < MIN_PITCH_RAD or pitch > MAX_PITCH_RAD:
                failures.append(f"pitch limit exceeded ({math.degrees(pitch):.1f}deg)")
            peak_rate = max(abs(value) for value in rates)
            if peak_rate > IMMEDIATE_MAX_BODY_RATE_RAD_S:
                failures.append(f"body rate immediate limit exceeded ({peak_rate:.2f}rad/s)")
            self._high_rate_samples = self._high_rate_samples + 1 if (
                peak_rate > MAX_BODY_RATE_RAD_S
            ) else 0
            if self._high_rate_samples >= 2:
                failures.append(f"body rate sustained limit exceeded ({peak_rate:.2f}rad/s)")
        if failures:
            self._abort_latched = True
            raise SafetyAbort("; ".join(failures))

    def _record_tick(
        self,
        stage: str,
        elapsed_s: float,
        command: Optional[AttitudeRateCommand],
    ) -> None:
        race = self.adapter.race_status
        telemetry = self.adapter.latest_telemetry
        imu = telemetry.imu if telemetry is not None else None
        target = self.tracker.target
        estimate = self.estimate
        self.recorder.emit(
            "tick",
            stage=stage,
            elapsed_s=elapsed_s,
            imu_us=self._last_imu_us,
            imu_accel=(list(imu.accel) if imu else None),
            race_boot_ms=(race.sim_boot_time_ms if race else None),
            race_start_ms=(race.race_start_boot_time_ms if race else None),
            gate_index=(race.active_gate_index if race else None),
            rpy=(list(estimate.orientation.to_euler()) if estimate else None),
            body_rates=(list(estimate.body_rates) if estimate else None),
            target=(asdict(target) if target else None),
            command=(asdict(command) if command else None),
        )

    async def preflight(self, timeout_s: float = 10.0) -> Dict[str, Any]:
        """Passively validate feeds, estimator bootstrap, detector, and rate."""

        if not self.vision.is_running:
            self.vision.start()
        self._clear_epoch_state()
        start = time.monotonic()
        initial_frames = self.vision.stats().frames_decoded
        last_log = start
        while time.monotonic() - start < timeout_s:
            self._sample()
            elapsed = time.monotonic() - start
            stats = self.vision.stats()
            fps = (stats.frames_decoded - initial_frames) / max(elapsed, 1e-6)
            failures = self._stream_failures(
                require_estimator=True,
                require_target=True,
                require_armed=False,
            )
            if elapsed >= 1.0 and fps >= 20.0 and self.tracker.consecutive >= 3 and not failures:
                assert self.estimate is not None and self.tracker.target is not None
                roll, pitch, yaw = self.estimate.orientation.to_euler()
                result = {
                    "vision_fps": fps,
                    "vision_frames": stats.frames_decoded - initial_frames,
                    "vision_duplicates": stats.duplicate_datagrams,
                    "imu_us": self._last_imu_us,
                    "attitude_rpy_rad": [roll, pitch, yaw],
                    "gyro_bias_rad_s": list(self.estimator.gyro_bias),
                    "gate_bbox": list(self.tracker.target.bbox),
                    "gate_center": [self.tracker.target.center_x, self.tracker.target.center_y],
                    "gate_confidence": self.tracker.target.confidence,
                    "race_gate_index": self.adapter.race_status.active_gate_index,
                    # Build 3385 can boot Training with this bit already set,
                    # despite zero actuator demand.  Powered stages explicitly
                    # normalize to disarmed after their proved reset.
                    "sim_reports_armed": self.adapter.is_armed,
                }
                self.recorder.emit("preflight_pass", **result)
                logger.info(
                    "Preflight PASS: vision %.1f fps, IMU fresh, attitude "
                    "(roll=%.2fdeg pitch=%.2fdeg), gate bbox=%s",
                    fps,
                    math.degrees(roll),
                    math.degrees(pitch),
                    self.tracker.target.bbox,
                )
                return result
            if time.monotonic() - last_log >= 1.0:
                logger.info(
                    "Preflight: %.1fs, vision %.1ffps, calibration %.0f%%, "
                    "gate streak %d, waiting=%s",
                    elapsed,
                    fps,
                    100.0 * self.estimator.calibration_progress,
                    self.tracker.consecutive,
                    failures[:3],
                )
                last_log = time.monotonic()
            await asyncio.sleep(0.005)
        failures = self._stream_failures(
            require_estimator=True,
            require_target=True,
            require_armed=False,
        )
        raise SafetyAbort(f"preflight timed out: {failures}")

    async def _fresh_reset_baseline(self, timeout_s: float = 2.0) -> Tuple[int, int]:
        deadline = time.monotonic() + timeout_s
        last_race: Optional[int] = None
        last_imu: Optional[int] = None
        race_advances = 0
        imu_advances = 0
        current_pair: Optional[Tuple[int, int]] = None
        while time.monotonic() < deadline:
            telemetry = self.adapter.latest_telemetry
            imu = telemetry.imu if telemetry is not None else None
            race = self.adapter.race_status
            if (
                imu is not None
                and race is not None
                and self.adapter.imu_age_s <= MAX_IMU_AGE_S
                and self.adapter.race_status_age_s <= MAX_RACE_AGE_S
                and race.sim_boot_time_ms >= 800
                and imu.timestamp_us >= 200_000
            ):
                pair = (int(race.sim_boot_time_ms), int(imu.timestamp_us))
                if last_race is not None and pair[0] > last_race:
                    race_advances += 1
                if last_imu is not None and pair[1] > last_imu:
                    imu_advances += 1
                last_race, last_imu = pair
                current_pair = pair
                # These streams have very different cadences (about 4 Hz vs
                # 145 Hz).  Prove each independently; requiring simultaneous
                # advancement makes a healthy baseline impossible.
                if race_advances >= 2 and imu_advances >= 5:
                    return current_pair
            await asyncio.sleep(0.01)
        raise SafetyAbort("could not obtain fresh, advancing pre-reset race/IMU clocks")

    async def _observe_reset_proof(
        self,
        *,
        attempt: int,
        pre_race: int,
        pre_imu: int,
    ) -> Optional[ResetProof]:
        """Observe rollback after a reset that has already been sent."""

        deadline = time.monotonic() + RESET_PROOF_TIMEOUT_S
        race_samples: List[int] = []
        imu_samples: List[int] = []
        countdown_observed = False
        while time.monotonic() < deadline:
            telemetry = self.adapter.latest_telemetry
            imu = telemetry.imu if telemetry is not None else None
            race = self.adapter.race_status
            if race is not None:
                boot = int(race.sim_boot_time_ms)
                if clock_rolled_back(pre_race, boot, RESET_RACE_DROP_MS):
                    if not race_samples or boot > race_samples[-1]:
                        race_samples.append(boot)
                    if race.race_start_boot_time_ms < 0 or boot < race.race_start_boot_time_ms:
                        countdown_observed = True
            if imu is not None:
                stamp = int(imu.timestamp_us)
                if clock_rolled_back(pre_imu, stamp, RESET_IMU_DROP_US):
                    if not imu_samples or stamp > imu_samples[-1]:
                        imu_samples.append(stamp)
            if (
                len(race_samples) >= 2
                and len(imu_samples) >= 5
                and countdown_observed
            ):
                return ResetProof(
                    attempt=attempt,
                    pre_race_boot_ms=pre_race,
                    post_race_boot_ms=race_samples[-1],
                    pre_imu_us=pre_imu,
                    post_imu_us=imu_samples[-1],
                    advancing_race_samples=len(race_samples),
                    advancing_imu_samples=len(imu_samples),
                    countdown_observed=True,
                )
            await asyncio.sleep(0.005)
        return None

    def _accept_reset_proof(self, proof: ResetProof, *, restart_vision: bool) -> None:
        drain_imu = getattr(self.adapter, "drain_imu_samples", None)
        if callable(drain_imu):
            # Proof observation intentionally does not estimate attitude.  Drop
            # that accumulated batch and begin calibration on samples received
            # strictly after the accepted boundary.
            drain_imu()
        self._clear_epoch_state()
        self._epoch_race_anchor_ms = proof.post_race_boot_ms
        self._epoch_imu_anchor_us = proof.post_imu_us
        self._epoch_anchor_monotonic_s = time.monotonic()
        self._countdown_observed = proof.countdown_observed
        self.adapter.drain_collisions()
        if restart_vision:
            self.vision.reset()
            self.vision.start()
        self.recorder.emit("reset_proved", **asdict(proof))
        logger.info(
            "Reset epoch proved on attempt %d: race %d->%dms, IMU %d->%dus",
            proof.attempt,
            proof.pre_race_boot_ms,
            proof.post_race_boot_ms,
            proof.pre_imu_us,
            proof.post_imu_us,
        )

    async def establish_reset_epoch(self, *, restart_vision: bool) -> ResetProof:
        """Send reset and prove both authoritative simulator clocks rolled back."""

        self.vision.stop()
        self.tracker.reset()
        for attempt in range(1, RESET_MAX_ATTEMPTS + 1):
            pre_race, pre_imu = await self._fresh_reset_baseline()
            self.recorder.emit(
                "reset_sent",
                attempt=attempt,
                emergency=False,
                pre_race_boot_ms=pre_race,
                pre_gate_index=(
                    self.adapter.race_status.active_gate_index
                    if self.adapter.race_status is not None
                    else None
                ),
                pre_imu_us=pre_imu,
            )
            await self.adapter.reset()
            proof = await self._observe_reset_proof(
                attempt=attempt,
                pre_race=pre_race,
                pre_imu=pre_imu,
            )
            if proof is not None:
                self._accept_reset_proof(proof, restart_vision=restart_vision)
                return proof
            logger.warning("Reset attempt %d was not authoritatively proved; retrying", attempt)
        raise SafetyAbort("SIM_RESET was not proved after four bounded attempts")

    async def emergency_reset(self) -> Optional[ResetProof]:
        """Always send reset first; proof is best-effort and never a send gate."""

        self.vision.stop()
        self.tracker.reset()
        for attempt in range(1, RESET_MAX_ATTEMPTS + 1):
            telemetry = self.adapter.latest_telemetry
            imu = telemetry.imu if telemetry is not None else None
            race = self.adapter.race_status
            pre_race = int(race.sim_boot_time_ms) if race is not None else None
            pre_imu = int(imu.timestamp_us) if imu is not None else None
            self.recorder.emit(
                "reset_sent",
                attempt=attempt,
                emergency=True,
                pre_race_boot_ms=pre_race,
                pre_gate_index=(int(race.active_gate_index) if race is not None else None),
                pre_imu_us=pre_imu,
            )
            # This send is deliberately unconditional.  Stale/missing streams
            # may prevent proof, but can never prevent the emergency command.
            await self.adapter.reset()
            if pre_race is not None and pre_imu is not None:
                proof = await self._observe_reset_proof(
                    attempt=attempt,
                    pre_race=pre_race,
                    pre_imu=pre_imu,
                )
                if proof is not None:
                    self._accept_reset_proof(proof, restart_vision=False)
                    return proof
            else:
                await asyncio.sleep(0.5)
            logger.warning(
                "Emergency reset attempt %d was sent but not proved; retrying",
                attempt,
            )
        return None

    async def wait_for_go(self, timeout_s: float = 8.0) -> StartContext:
        deadline = time.monotonic() + timeout_s
        go_seen_at: Optional[float] = None
        while time.monotonic() < deadline:
            self._sample()
            race = self.adapter.race_status
            if self.adapter.is_armed:
                raise SafetyAbort("vehicle became armed before runner issued post-GO arm")
            collisions = self.adapter.drain_collisions()
            if collisions:
                raise SafetyAbort(f"collision during countdown: {collisions!r}")
            if (
                self._imu_regressed
                or self._race_regressed
                or self._imu_forward_jump
                or self._race_forward_jump
            ):
                raise SafetyAbort("clock left the proved reset epoch")
            if race is not None and race.active_gate_index != 0:
                raise SafetyAbort(f"fresh race did not start at gate 0 ({race.active_gate_index})")
            if (
                race is not None
                and self._countdown_observed
                and race.race_start_boot_time_ms >= 0
                and race.sim_boot_time_ms >= race.race_start_boot_time_ms + 150
            ):
                go_seen_at = go_seen_at or time.monotonic()
                failures = self._stream_failures(
                    require_estimator=True,
                    require_target=True,
                    require_armed=False,
                )
                if self.tracker.consecutive < 3:
                    failures.append("gate target lacks three-frame confirmation")
                if not self._countdown_observed:
                    failures.append("fresh post-reset countdown was not observed")
                if not failures:
                    assert self.estimate is not None and self.tracker.target is not None
                    roll, pitch, _yaw = self.estimate.orientation.to_euler()
                    if abs(roll) > math.radians(5.0):
                        raise SafetyAbort("pad calibration roll is implausible")
                    if not MIN_PITCH_RAD <= pitch <= MAX_PITCH_RAD:
                        raise SafetyAbort("pad calibration pitch is outside safety envelope")
                    context = StartContext(
                        spawn_roll_rad=roll,
                        spawn_pitch_rad=pitch,
                        initial_gate_x=self.tracker.target.center_x,
                        initial_gate_y=self.tracker.target.center_y,
                        initial_gate_area=self.tracker.target.bbox_area,
                        go_boot_ms=int(race.sim_boot_time_ms),
                    )
                    self.recorder.emit("go_ready", **asdict(context))
                    return context
                if time.monotonic() - go_seen_at > 1.0:
                    raise SafetyAbort(f"GO passed without full readiness: {failures}")
            await asyncio.sleep(0.005)
        raise SafetyAbort("timed out waiting for fresh reset countdown and GO")

    async def arm_confirmed(self, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            token = self.adapter.heartbeat_sequence
            await self.adapter.arm()
            confirm_deadline = min(deadline, time.monotonic() + 0.45)
            while time.monotonic() < confirm_deadline:
                self._sample()
                if self.adapter.heartbeat_sequence > token and self.adapter.is_armed:
                    self.recorder.emit("arm_confirmed", heartbeat_sequence=self.adapter.heartbeat_sequence)
                    logger.info("Arming confirmed by a post-attempt heartbeat")
                    return
                failures = self._stream_failures(
                    require_estimator=True,
                    require_target=True,
                    require_armed=False,
                )
                if failures:
                    raise SafetyAbort(f"stream failure while confirming arm: {failures}")
                await asyncio.sleep(0.01)
        raise SafetyAbort("arming was not confirmed by a newer heartbeat")

    async def normalize_disarmed(self) -> None:
        """Force and confirm a ground-safe state before waiting for GO."""

        if not await self._disarm_confirmed():
            raise SafetyAbort("could not confirm disarmed state after reset")
        if self.adapter.is_armed:
            raise SafetyAbort("simulator still reports armed after disarm confirmation")
        self.recorder.emit("ground_disarmed")

    async def _disarm_confirmed(self, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            token = self.adapter.heartbeat_sequence
            try:
                await self.adapter.disarm()
            except Exception:
                logger.exception("Disarm send failed")
            confirm_deadline = min(deadline, time.monotonic() + 0.25)
            while time.monotonic() < confirm_deadline:
                if self.adapter.heartbeat_sequence > token and not self.adapter.is_armed:
                    self.recorder.emit(
                        "disarm_confirmed",
                        heartbeat_sequence=self.adapter.heartbeat_sequence,
                    )
                    return True
                await asyncio.sleep(0.01)
        return False

    async def safe_cleanup(self) -> bool:
        """Latch command production, cut thrust, confirm disarm, then reset."""

        self._abort_latched = True
        race_before_cleanup = self.adapter.race_status
        gate_index_before_cleanup = (
            int(race_before_cleanup.active_gate_index)
            if race_before_cleanup is not None
            else None
        )
        race_boot_before_cleanup = (
            int(race_before_cleanup.sim_boot_time_ms)
            if race_before_cleanup is not None
            else None
        )
        try:
            if self.adapter.is_armed:
                zero = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
                await self.adapter.send_attitude_rate(zero)
                self.recorder.emit(
                    "zero_thrust_sent",
                    gate_index=gate_index_before_cleanup,
                    race_boot_ms=race_boot_before_cleanup,
                )
        except Exception:
            logger.exception("Could not send the one-shot zero-thrust command")
        # Do not delay the unconditional reset fallback behind a long heartbeat
        # wait when the abort itself may be a transport failure.
        disarmed = await self._disarm_confirmed(timeout_s=0.6)
        if not disarmed:
            logger.error("Disarm was not confirmed before reset fallback")

        reset_proved = False
        try:
            reset_proved = await self.emergency_reset() is not None
        except Exception:
            logger.exception("Emergency SIM_RESET send/proof path failed")

        if not disarmed or self.adapter.is_armed:
            disarmed = await self._disarm_confirmed()
        confirmed = bool(disarmed and reset_proved and not self.adapter.is_armed)
        self.recorder.emit(
            "cleanup_complete",
            disarmed=disarmed,
            reset_proved=reset_proved,
            confirmed=confirmed,
            gate_index_before_cleanup=gate_index_before_cleanup,
            race_boot_before_cleanup=race_boot_before_cleanup,
        )
        if not confirmed:
            logger.critical("UNRESOLVED EMERGENCY: stop/reset state was not fully confirmed")
        return confirmed

    async def _run_sign_id(self) -> Dict[str, Any]:
        assert self.estimate is not None
        start_rpy = self.estimate.orientation.to_euler()
        max_excursion = 0.0
        segments = [
            ("settle", 0.25, (0.0, 0.0, 0.0)),
            ("roll", 0.10, (0.08, 0.0, 0.0)),
            ("neutral", 0.12, (0.0, 0.0, 0.0)),
            ("pitch", 0.10, (0.0, 0.08, 0.0)),
        ]
        responses: Dict[str, List[float]] = {"roll": [], "pitch": []}
        baseline_samples: List[Tuple[float, float]] = []
        flight_start = time.monotonic()
        next_tick = flight_start
        for name, duration, rates in segments:
            segment_start = time.monotonic()
            while time.monotonic() - segment_start < duration:
                segment_elapsed = time.monotonic() - segment_start
                self._sample()
                self._watchdog(allow_benign_pad_contact=True)
                assert self.estimate is not None
                current_rpy = self.estimate.orientation.to_euler()
                max_excursion = max(
                    max_excursion,
                    abs(current_rpy[0] - start_rpy[0]),
                    abs(current_rpy[1] - start_rpy[1]),
                )
                if max_excursion > 0.05:
                    raise SafetyAbort(
                        f"sign-ID attitude excursion too large ({max_excursion:.3f}rad)"
                    )
                command = AttitudeRateCommand(rates[0], rates[1], 0.0, 0.235)
                validate_command(command)
                await self.adapter.send_attitude_rate(command)
                elapsed = time.monotonic() - flight_start
                self._record_tick(f"sign-id/{name}", elapsed, command)
                if name == "settle" and segment_elapsed > 0.15:
                    baseline_samples.append(
                        (self.estimate.body_rates[0], self.estimate.body_rates[1])
                    )
                if name in responses and segment_elapsed > 0.04:
                    axis = 0 if name == "roll" else 1
                    responses[name].append(self.estimate.body_rates[axis])
                next_tick = next_control_deadline(next_tick, time.monotonic())
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
        assert self.estimate is not None
        end_rpy = self.estimate.orientation.to_euler()
        excursion = max(abs(end_rpy[i] - start_rpy[i]) for i in (0, 1))
        raw_means = {
            axis: (statistics.fmean(values) if values else 0.0)
            for axis, values in responses.items()
        }
        baseline = {
            "roll": statistics.fmean(sample[0] for sample in baseline_samples),
            "pitch": statistics.fmean(sample[1] for sample in baseline_samples),
        }
        means = {axis: raw_means[axis] - baseline[axis] for axis in responses}
        bad = [axis for axis, mean in means.items() if mean <= 0.006]
        if bad:
            raise SafetyAbort(f"sign-ID response inconclusive/wrong for {bad}: {means}")
        return {
            "mean_responses_rad_s": means,
            "raw_mean_responses_rad_s": raw_means,
            "baseline_rates_rad_s": baseline,
            "final_attitude_excursion_rad": excursion,
            "max_attitude_excursion_rad": max_excursion,
        }

    async def _run_hover(self, context: StartContext) -> Dict[str, Any]:
        assert self.estimate is not None
        flight_start = time.monotonic()
        next_tick = flight_start
        max_abs_roll = 0.0
        max_abs_rate = 0.0
        while True:
            now = time.monotonic()
            elapsed = now - flight_start
            if elapsed >= 2.5:
                break
            self._sample()
            self._watchdog(
                allow_benign_pad_contact=elapsed < 0.35,
                enforce_benign_pad_budget=True,
            )
            assert self.estimate is not None
            blend = min(1.0, elapsed / 0.8)
            target_pitch = (1.0 - blend) * context.spawn_pitch_rad
            if elapsed < 0.15:
                thrust = 0.26
            elif elapsed < 1.0:
                # Spawn is pitched about 18deg nose-down.  The previous
                # 0.20->0.265 ramp never exceeded weight along world-up, so the
                # drone slid off the launch pad.  0.32 supplies positive climb
                # margin while the attitude loop levels it.
                thrust = 0.32
            else:
                thrust = min(0.29, 0.275 / max(0.95, math.cos(target_pitch)))
            command = attitude_rate_command(
                self.estimate,
                target_roll_rad=0.0,
                target_pitch_rad=target_pitch,
                thrust=thrust,
            )
            await self.adapter.send_attitude_rate(command)
            max_abs_roll = max(max_abs_roll, abs(self.estimate.roll))
            max_abs_rate = max(max_abs_rate, max(abs(v) for v in self.estimate.body_rates))
            self._record_tick("hover", elapsed, command)
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
        return {
            "max_abs_roll_deg": math.degrees(max_abs_roll),
            "max_abs_body_rate_rad_s": max_abs_rate,
            "final_rpy_rad": list(self.estimate.orientation.to_euler()),
        }

    async def _run_gate0(self, context: StartContext) -> Dict[str, Any]:
        flight_start = time.monotonic()
        next_tick = flight_start
        max_gate_area = context.initial_gate_area
        last_target_frame: Optional[int] = None
        last_control_y: Optional[float] = None
        last_target_time: Optional[float] = None
        control_y_rate = 0.0
        crossing_armed = False
        crossing_started_s: Optional[float] = None
        crossing_race_boot_ms: Optional[int] = None
        while True:
            now = time.monotonic()
            elapsed = now - flight_start
            if elapsed >= 5.0:
                raise SafetyAbort("gate-0 wall-time limit reached")
            self._sample()
            race = self.adapter.race_status
            assert race is not None and self.estimate is not None
            target = self.tracker.target
            assert target is not None
            max_gate_area = max(max_gate_area, target.bbox_area)
            if elapsed > 3.5 and max_gate_area < 1.25 * context.initial_gate_area:
                raise SafetyAbort("no visual approach progress toward gate 0")

            blend = min(1.0, elapsed / 0.8)
            target_pitch = (1.0 - blend) * context.spawn_pitch_rad
            normalized_x = (target.center_x - 320.0) / 320.0
            target_roll = max(-0.08, min(0.08, 0.15 * normalized_x))

            control_y = gate_control_center_y_px(
                target,
                previous_center_y=last_control_y,
            )
            if (
                not crossing_armed
                and target.age_s(now) <= CROSSING_TARGET_LOSS_S
                and self.tracker.consecutive >= 3
                and race.active_gate_index == 0
                and is_close_gate_crossing_candidate(
                    target,
                    initial_gate_area=context.initial_gate_area,
                    control_y=control_y,
                )
            ):
                crossing_armed = True
                self.recorder.emit(
                    "crossing_candidate_armed",
                    elapsed_s=elapsed,
                    race_boot_ms=race.sim_boot_time_ms,
                    target=asdict(target),
                    control_y=control_y,
                )

            crossing_confirming = bool(
                crossing_started_s is not None
                or (
                    crossing_armed
                    and target.age_s(now) > CROSSING_TARGET_LOSS_S
                )
            )
            self._watchdog(
                require_target=not (
                    crossing_confirming or race.active_gate_index == 1
                ),
                allow_benign_pad_contact=elapsed < 0.35,
                enforce_benign_pad_budget=True,
            )
            if race.active_gate_index not in (0, 1):
                raise SafetyAbort(f"unexpected gate-index jump to {race.active_gate_index}")
            if not crossing_confirming and race.active_gate_index == 1:
                return {
                    "gate0_passed": True,
                    "race_boot_ms": race.sim_boot_time_ms,
                    "last_gate_race_time": race.last_gate_race_time,
                    "max_gate_area_px": max_gate_area,
                    "crossing_confirmation_used": crossing_started_s is not None,
                }

            if crossing_confirming:
                if crossing_started_s is None:
                    crossing_started_s = now
                    crossing_race_boot_ms = int(race.sim_boot_time_ms)
                    self.recorder.emit(
                        "crossing_confirmation_started",
                        elapsed_s=elapsed,
                        baseline_race_boot_ms=crossing_race_boot_ms,
                        target_age_s=target.age_s(now),
                    )
                assert crossing_race_boot_ms is not None
                decision = crossing_status_decision(
                    baseline_race_boot_ms=crossing_race_boot_ms,
                    current_race_boot_ms=int(race.sim_boot_time_ms),
                    active_gate_index=int(race.active_gate_index),
                    elapsed_s=now - crossing_started_s,
                )
                if decision != "waiting":
                    self.recorder.emit(
                        "crossing_status_decision",
                        decision=decision,
                        baseline_race_boot_ms=crossing_race_boot_ms,
                        current_race_boot_ms=race.sim_boot_time_ms,
                        gate_index=race.active_gate_index,
                    )
                    if decision == "passed":
                        return {
                            "gate0_passed": True,
                            "race_boot_ms": race.sim_boot_time_ms,
                            "last_gate_race_time": race.last_gate_race_time,
                            "max_gate_area_px": max_gate_area,
                            "crossing_confirmation_used": True,
                        }
                    raise SafetyAbort(f"gate-0 crossing {decision.replace('_', ' ')}")
                command = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
                await self.adapter.send_attitude_rate(command)
                self._record_tick("gate0/confirm", elapsed, command)
                next_tick = next_control_deadline(next_tick, time.monotonic())
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            if target.frame_id != last_target_frame:
                if last_control_y is not None and last_target_time is not None:
                    dt_target = target.received_monotonic_s - last_target_time
                    if dt_target > 1e-3:
                        raw_rate = (control_y - last_control_y) / dt_target
                        raw_rate = max(-300.0, min(300.0, raw_rate))
                        control_y_rate = 0.65 * control_y_rate + 0.35 * raw_rate
                last_target_frame = target.frame_id
                last_control_y = control_y
                last_target_time = target.received_monotonic_s
            if elapsed < 0.15:
                thrust = 0.26
            elif elapsed < 0.45:
                thrust = 0.32
            else:
                # Steer the camera ray through the opening center.  Image-rate
                # damping brakes climb before positional error grows near the
                # rapidly approaching gate.
                thrust = gate_vertical_thrust(control_y, control_y_rate)

            # At close range the uncorrected contour center becomes unsafe if
            # the lower gate edge is clipped.  Abort before impact when the
            # square-inferred center is plainly outside the opening corridor.
            if (
                target.bbox_area >= 8 * context.initial_gate_area
                and abs(control_y - 180.0) > 75.0
            ):
                raise SafetyAbort(
                    f"gate-0 close approach outside vertical corridor ({control_y:.1f}px)"
                )
            command = attitude_rate_command(
                self.estimate,
                target_roll_rad=target_roll,
                target_pitch_rad=target_pitch,
                thrust=thrust,
            )
            await self.adapter.send_attitude_rate(command)
            self._record_tick("gate0", elapsed, command)
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def run_powered_stage(self, stage: str) -> StageResult:
        if stage not in {"sign-id", "hover", "gate0"}:
            raise ValueError(f"unsupported powered stage: {stage}")
        started = time.monotonic()
        reason = "unknown"
        success = False
        details: Dict[str, Any] = {}
        gate_before: Optional[int] = None
        gate_after: Optional[int] = None
        cleanup_confirmed = False
        try:
            self._abort_latched = False
            await self.establish_reset_epoch(restart_vision=True)
            await self.normalize_disarmed()
            context = await self.wait_for_go()
            race = self.adapter.race_status
            gate_before = race.active_gate_index if race else None
            await self.arm_confirmed()
            if stage == "sign-id":
                details = await self._run_sign_id()
            elif stage == "hover":
                details = await self._run_hover(context)
            else:
                details = await self._run_gate0(context)
            success = True
            reason = "stage completed"
        except (SafetyAbort, asyncio.CancelledError) as exc:
            reason = str(exc) or type(exc).__name__
            logger.error("%s ABORT: %s", stage, reason)
            self.recorder.emit("stage_abort", stage=stage, reason=reason)
            if isinstance(exc, asyncio.CancelledError):
                raise
        except Exception as exc:
            reason = f"unexpected {type(exc).__name__}: {exc}"
            logger.exception("%s failed unexpectedly", stage)
            self.recorder.emit("stage_abort", stage=stage, reason=reason)
        finally:
            cleanup_confirmed = await self.safe_cleanup()
            race = self.adapter.race_status
            gate_after = race.active_gate_index if race else None
        return StageResult(
            stage=stage,
            success=success and cleanup_confirmed,
            reason=(reason if cleanup_confirmed else f"{reason}; cleanup unconfirmed"),
            duration_s=time.monotonic() - started,
            gate_index_before=gate_before,
            gate_index_after=gate_after,
            cleanup_confirmed=cleanup_confirmed,
            details=details,
        )


async def run_live(stage: str, address: str, record: Optional[str]) -> StageResult:
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
    )
    vision = VQ2VisionThread()
    recorder = JsonlRecorder(record)
    runner = VQ2Runner(adapter, vision, recorder=recorder)
    connected = False
    try:
        await adapter.connect(address)
        connected = True
        preflight = await runner.preflight()
        if stage == "preflight":
            return StageResult(
                stage=stage,
                success=True,
                reason="passive preflight completed; no flight command sent",
                duration_s=0.0,
                gate_index_before=preflight.get("race_gate_index"),
                gate_index_after=preflight.get("race_gate_index"),
                cleanup_confirmed=True,
                details=preflight,
            )
        return await runner.run_powered_stage(stage)
    finally:
        vision.stop()
        if connected:
            await adapter.disconnect()
        recorder.close()


def _default_record_path(stage: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return str(Path("captures") / f"vq2_{stage}_{stamp}.jsonl.gz")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded AIGP VQ2 training runner")
    parser.add_argument(
        "--stage",
        choices=("preflight", "sign-id", "hover", "gate0"),
        default="preflight",
    )
    parser.add_argument("--address", default=DEFAULT_MAVLINK_URL)
    parser.add_argument(
        "--record",
        nargs="?",
        const="auto",
        default=None,
        help="write JSONL capture; omit the value for an automatic gzip path",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    record = _default_record_path(args.stage) if args.record == "auto" else args.record
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = asyncio.run(run_live(args.stage, args.address, record))
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
