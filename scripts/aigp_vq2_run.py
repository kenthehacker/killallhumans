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
``gate0-observe``
    Run the proved gate-0 stage, then hold zero thrust for at most 0.20 seconds
    while recording a three-frame observation of the next gate.

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

if TYPE_CHECKING:
    from aigp_loop.replay import AsyncReplayRecorder


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
GATE0_FLIGHT_TIMEOUT_S = 5.0

POST_GATE_OBSERVATION_TIMEOUT_S = 0.20
POST_GATE_REQUIRED_FRAMES = 3
POST_GATE_MAX_ATTITUDE_DELTA_RAD = math.radians(5.0)
POST_GATE_IMMEDIATE_MAX_BODY_RATE_RAD_S = 1.0
POST_GATE_SUSTAINED_MAX_BODY_RATE_RAD_S = 0.5

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


def _replay_capture_dependencies():
    """Load optional evidence tooling only when private capture is requested."""

    from aigp_loop._util import environment_fingerprint, git_provenance
    from aigp_loop.replay import AsyncReplayRecorder, ReplayBundleWriter

    return (
        AsyncReplayRecorder,
        ReplayBundleWriter,
        environment_fingerprint,
        git_provenance,
    )


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
class GateTransitionProof:
    """Internal authority and timing handoff from gate 0 to observation."""

    pre_gate_race_boot_ms: int
    post_gate_race_boot_ms: int
    flight_started_monotonic_s: float
    crossing_started_monotonic_s: Optional[float]
    pass_confirmed_monotonic_s: float
    next_control_deadline_s: float
    vision_generation: int
    vision_frame_id: int
    vision_sim_time_ns: int
    vision_received_monotonic_s: float
    pass_rpy_rad: Tuple[float, float, float]


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


def post_gate_observation_deadline(
    *,
    pass_confirmed_s: float,
    flight_started_s: float,
    crossing_started_s: Optional[float],
) -> float:
    """Fixed observation deadline nested inside every existing flight bound."""

    values = [float(pass_confirmed_s), float(flight_started_s)]
    if crossing_started_s is not None:
        values.append(float(crossing_started_s))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("post-gate deadline inputs must be finite")
    candidates = [
        float(pass_confirmed_s) + POST_GATE_OBSERVATION_TIMEOUT_S,
        float(flight_started_s) + GATE0_FLIGHT_TIMEOUT_S,
    ]
    if crossing_started_s is not None:
        candidates.append(float(crossing_started_s) + CROSSING_STATUS_TIMEOUT_S)
    return min(candidates)


def is_crossing_residue(
    target: GateTarget | GateDetection,
    *,
    image_width: int = 640,
    image_height: int = 360,
) -> bool:
    """Reject a large clipped remnant of gate 0 during gate-1 reacquisition.

    The predicate is deliberately scoped to the post-pass tracker.  Large
    clipped contours are useful evidence during the gate-0 approach, but they
    must not seed a fresh tracker after race status authoritatively advances.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x, y, width, height = (int(value) for value in target.bbox)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        return False

    width_fraction = width / image_width
    height_fraction = height / image_height
    area_fraction = width_fraction * height_fraction
    left = x < 3
    top = y < 3
    right = x + width > image_width - 3
    bottom = y + height > image_height - 3
    opposing_edges = (top and bottom) or (left and right)
    edge_count = sum((left, top, right, bottom))
    return bool(
        (opposing_edges and area_fraction >= 0.25)
        or (
            edge_count >= 1
            and area_fraction >= 0.60
            and width_fraction >= 0.70
            and height_fraction >= 0.70
        )
        or (width_fraction >= 0.90 and height_fraction >= 0.90)
    )


def _finite_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def gate_detection_summary(
    detection: GateDetection,
    *,
    detector_index: int,
    image_width: int = 640,
    image_height: int = 360,
    reject_crossing_residue: bool = False,
) -> Dict[str, Any]:
    """Return JSON-safe pixel diagnostics for one raw detector result."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x, y, width, height = (int(value) for value in detection.bbox)
    valid_bbox = bool(
        x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= image_width
        and y + height <= image_height
    )
    confidence = _finite_float(detection.confidence)
    raw_corners = getattr(detection, "corners", None)
    corners_px: Optional[List[List[float]]] = None
    try:
        candidate_corners = [
            [_finite_float(point[0]), _finite_float(point[1])]
            for point in raw_corners
        ]
        if (
            len(candidate_corners) == 4
            and all(value is not None for point in candidate_corners for value in point)
        ):
            corners_px = [
                [float(point[0]), float(point[1])] for point in candidate_corners
            ]
    except (TypeError, IndexError):
        corners_px = None
    axis_aspect = (
        max(width, height) / min(width, height)
        if width > 0 and height > 0
        else None
    )
    rejections: List[str] = []
    if not valid_bbox:
        rejections.append("invalid_bbox")
    if width < 20:
        rejections.append("min_width")
    if height < 20:
        rejections.append("min_height")
    if axis_aspect is None or axis_aspect > 1.85:
        rejections.append("axis_aspect_gt_1.85")
    if confidence is None:
        rejections.append("nonfinite_confidence")
    elif confidence < 0.10:
        rejections.append("confidence_below_0.10")
    residue = is_crossing_residue(
        detection,
        image_width=image_width,
        image_height=image_height,
    )
    base_selector_eligible = not rejections
    if residue and reject_crossing_residue:
        rejections.append("crossing_residue")
    return {
        "detector_index": int(detector_index),
        "center_px": [int(detection.center_x), int(detection.center_y)],
        "corners_px": corners_px,
        "bbox_xywh_px": [x, y, width, height],
        "reported_area_px": int(detection.area),
        "bbox_area_px": width * height,
        "bbox_area_fraction": (
            (width * height) / (image_width * image_height) if valid_bbox else None
        ),
        "axis_aspect_ratio": axis_aspect,
        "apparent_size_px": [
            _finite_float(detection.apparent_width_px),
            _finite_float(detection.apparent_height_px),
        ],
        "min_rect_aspect_ratio": _finite_float(detection.aspect_ratio),
        "rotation_deg": _finite_float(detection.rotation_deg),
        "rectangularity": _finite_float(detection.rectangularity),
        "confidence": confidence,
        "method": str(detection.detection_method),
        "edge_touch": {
            "left": valid_bbox and x < 3,
            "top": valid_bbox and y < 3,
            "right": valid_bbox and x + width > image_width - 3,
            "bottom": valid_bbox and y + height > image_height - 3,
        },
        "base_selector_eligible": base_selector_eligible,
        "post_pass_selector_eligible": base_selector_eligible and not residue,
        "selector_eligible": not rejections,
        "selector_rejections": rejections,
        "crossing_residue": residue,
    }


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
    def __init__(
        self,
        path: Optional[str],
        *,
        replay: Optional[AsyncReplayRecorder] = None,
    ) -> None:
        self.path = Path(path).resolve() if path else None
        self.replay = replay
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if str(self.path).endswith(".gz"):
                self._handle = gzip.open(self.path, "wt", encoding="utf-8")
            else:
                self._handle = self.path.open("w", encoding="utf-8")

    @property
    def capture_enabled(self) -> bool:
        return self.replay is not None

    def emit(self, event: str, **fields: Any) -> None:
        if self._handle is not None:
            row = {"event": event, "wall_time_ns": time.time_ns(), **fields}
            self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        if self.replay is not None:
            self.replay.record_event(event, **fields)

    def record_imu(self, imu: Any, estimator: Optional[Dict[str, Any]], now_s: float) -> None:
        if self.replay is not None:
            self.replay.record_imu(
                imu,
                estimator=estimator,
                received_monotonic_s=now_s,
            )

    def record_race(self, race: Any, now_s: float) -> None:
        if self.replay is not None:
            self.replay.record_race(race, received_monotonic_s=now_s)

    def record_command(
        self,
        kind: str,
        command: AttitudeRateCommand,
        *,
        monotonic_s: float,
        frame_token: Optional[Tuple[int, int, int]],
    ) -> None:
        if self.replay is not None:
            self.replay.record_command(
                kind,
                command,
                monotonic_s=monotonic_s,
                frame_token=frame_token,
            )

    def capture_frame(self, image: Any, **fields: Any) -> None:
        if self.replay is not None:
            self.replay.capture_frame(image, **fields)

    def save_png(self, label: str, image: Any) -> Optional[str]:
        """Persist one deferred diagnostic image beside the JSONL capture."""

        if self.path is None:
            return None
        import cv2

        base = self.path
        if base.suffix == ".gz":
            base = base.with_suffix("")
        if base.suffix == ".jsonl":
            base = base.with_suffix("")
        safe_label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(label)
        ).strip("_") or "frame"
        output = base.parent / f"{base.name}_{safe_label}.png"
        if not cv2.imwrite(str(output), image):
            raise OSError(f"OpenCV could not write diagnostic image {output}")
        return str(output.resolve())

    def close(self, *, outcome: Optional[Dict[str, Any]] = None) -> Any:
        handle_error: Optional[BaseException] = None
        handle_traceback = None
        if self._handle is not None:
            handle = self._handle
            self._handle = None
            try:
                handle.close()
            except BaseException as exc:
                handle_error = exc
                handle_traceback = exc.__traceback__
        replay_result = None
        replay_error: Optional[BaseException] = None
        if self.replay is not None:
            expected = None
            if outcome is not None:
                vision_stats = outcome.get("vision_capture_stats")
                if isinstance(vision_stats, dict):
                    expected = vision_stats.get("frames_decoded")
            if handle_error is not None:
                try:
                    self.replay.fail(
                        "legacy JSONL recorder close failed before replay seal: "
                        f"{type(handle_error).__name__}: {handle_error}"
                    )
                except BaseException as exc:
                    replay_error = exc
            try:
                replay_result = self.replay.close(
                    outcome=outcome,
                    expected_decoded_frames=expected,
                )
            except BaseException as exc:
                replay_error = replay_error or exc
        if handle_error is not None:
            if replay_error is not None:
                handle_error.add_note(
                    "Replay cleanup also failed: "
                    f"{type(replay_error).__name__}: {replay_error}"
                )
            raise handle_error.with_traceback(handle_traceback)
        if replay_error is not None:
            raise replay_error
        return replay_result


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
        self._last_frame_identity: Optional[Tuple[int, int]] = None
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
        self._latest_raw_detections: List[GateDetection] = []
        self._latest_accepted_target: Optional[GateTarget] = None
        self._latest_detection_frame_id: Optional[int] = None
        self._latest_detection_frame_sim_ns: Optional[int] = None
        self._latest_detection_generation: Optional[int] = None
        self._latest_detection_received_s: Optional[float] = None
        self._latest_detection_image: Any = None
        self._post_gate_last_frame: Optional[Tuple[Tuple[int, int, int], Any]] = None
        self._vision_diagnostic_logging = False
        self._post_gate_reacquisition = False
        self._last_flight_command: Optional[AttitudeRateCommand] = None
        self._last_flight_command_sent_s: Optional[float] = None
        self._gate0_transition_proof: Optional[GateTransitionProof] = None
        self._deferred_pngs: List[Tuple[str, Any]] = []

    def _replay_estimator_fields(self) -> Optional[Dict[str, Any]]:
        estimate = self.estimate
        if estimate is None:
            return None
        return {
            "timestamp_us": int(estimate.timestamp_us),
            "rpy_rad": list(estimate.orientation.to_euler()),
            "orientation_wxyz": [
                estimate.orientation.w,
                estimate.orientation.x,
                estimate.orientation.y,
                estimate.orientation.z,
            ],
            "body_rates": list(estimate.body_rates),
            "gyro_bias": list(estimate.gyro_bias),
            "healthy": bool(estimate.healthy),
            "reason": estimate.reason,
            "propagated": bool(estimate.propagated),
        }

    def _latest_frame_token(self) -> Optional[Tuple[int, int, int]]:
        if (
            self._latest_detection_generation is None
            or self._latest_detection_frame_id is None
            or self._latest_detection_frame_sim_ns is None
        ):
            return None
        return (
            self._latest_detection_generation,
            self._latest_detection_frame_id,
            self._latest_detection_frame_sim_ns,
        )

    def _clear_epoch_state(self) -> None:
        self.estimator.reset()
        self.estimate = None
        self._last_imu_us = None
        self._last_imu_advance_s = 0.0
        self._last_race_boot_ms = None
        self._last_race_advance_s = 0.0
        self._last_frame_identity = None
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
        self._latest_raw_detections = []
        self._latest_accepted_target = None
        self._latest_detection_frame_id = None
        self._latest_detection_frame_sim_ns = None
        self._latest_detection_generation = None
        self._latest_detection_received_s = None
        self._latest_detection_image = None
        self._vision_diagnostic_logging = False
        self._post_gate_reacquisition = False
        self._last_flight_command = None
        self._last_flight_command_sent_s = None
        self._gate0_transition_proof = None
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
            record_imu = getattr(self.recorder, "record_imu", None)
            if callable(record_imu):
                record_imu(imu, self._replay_estimator_fields(), now)

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
                record_race = getattr(self.recorder, "record_race", None)
                if callable(record_race):
                    record_race(race, now)
            elif boot < self._last_race_boot_ms:
                self._race_regressed = True

        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        frame_identity = (
            None
            if snapshot is None
            else (int(snapshot.generation), int(snapshot.frame_id))
        )
        if snapshot is not None and frame_identity != self._last_frame_identity:
            # The camera source timestamp is an opaque ordering token, not
            # frame identity.  Repeated control polls consume one publication
            # once, while a receiver generation restart can reuse frame IDs.
            self._last_frame_identity = frame_identity
            self._last_frame_sim_ns = snapshot.sim_time_ns
            self._latest_detection_frame_id = int(snapshot.frame_id)
            self._latest_detection_frame_sim_ns = int(snapshot.sim_time_ns)
            self._latest_detection_generation = int(snapshot.generation)
            self._latest_detection_received_s = float(snapshot.received_monotonic_s)
            capture_enabled = bool(
                getattr(self.recorder, "capture_enabled", False)
            )
            detector_started_ns = time.perf_counter_ns() if capture_enabled else None
            detector_latency_ms: Optional[float] = None
            try:
                image = snapshot.camera_frame.image
                self._latest_detection_image = image
                image_height, image_width = image.shape[:2]
                detections = list(self.detector.detect(image))
                if detector_started_ns is not None:
                    detector_latency_ms = (
                        time.perf_counter_ns() - detector_started_ns
                    ) / 1_000_000.0
                self._latest_raw_detections = detections
                tracking_detections = detections
                if self._post_gate_reacquisition:
                    tracking_detections = [
                        detection
                        for detection in detections
                        if not is_crossing_residue(
                            detection,
                            image_width=image_width,
                            image_height=image_height,
                        )
                        and gate_detection_summary(
                            detection,
                            detector_index=0,
                            image_width=image_width,
                            image_height=image_height,
                            reject_crossing_residue=True,
                        )["selector_eligible"]
                    ]
                accepted = self.tracker.update(
                    tracking_detections,
                    frame_id=snapshot.frame_id,
                    sim_time_ns=snapshot.sim_time_ns,
                    received_monotonic_s=snapshot.received_monotonic_s,
                )
                self._latest_accepted_target = accepted
                if self._post_gate_reacquisition:
                    self._post_gate_last_frame = (
                        (
                            int(snapshot.generation),
                            int(snapshot.frame_id),
                            int(snapshot.sim_time_ns),
                        ),
                        image,
                    )
                summaries = (
                    [
                        gate_detection_summary(
                            detection,
                            detector_index=index,
                            image_width=image_width,
                            image_height=image_height,
                            reject_crossing_residue=self._post_gate_reacquisition,
                        )
                        for index, detection in enumerate(detections)
                    ]
                    if self._vision_diagnostic_logging or capture_enabled
                    else []
                )
                if self._vision_diagnostic_logging:
                    selected = select_primary_gate(tracking_detections)
                    selected_index = next(
                        (
                            index
                            for index, detection in enumerate(detections)
                            if detection is selected
                        ),
                        None,
                    )
                    race = self.adapter.race_status
                    estimate = self.estimate
                    self.recorder.emit(
                        "vision_detection_frame",
                        phase=(
                            "gate1_reacquisition"
                            if self._post_gate_reacquisition
                            else "gate0_crossing"
                        ),
                        frame_id=snapshot.frame_id,
                        sim_time_ns=snapshot.sim_time_ns,
                        generation=snapshot.generation,
                        received_monotonic_s=snapshot.received_monotonic_s,
                        receive_age_s=snapshot.age_s(now),
                        image_size_px=[image_width, image_height],
                        race_boot_ms=(race.sim_boot_time_ms if race else None),
                        gate_index=(race.active_gate_index if race else None),
                        detections=summaries,
                        selected_detection_index=selected_index,
                        tracker_streak=self.tracker.consecutive,
                        accepted_target=(asdict(accepted) if accepted else None),
                        tracker_target=(
                            asdict(self.tracker.target) if self.tracker.target else None
                        ),
                        rpy=(
                            list(estimate.orientation.to_euler()) if estimate else None
                        ),
                        body_rates=(list(estimate.body_rates) if estimate else None),
                        last_command=(
                            asdict(self._last_flight_command)
                            if self._last_flight_command
                            else None
                        ),
                    )
                capture_frame = getattr(self.recorder, "capture_frame", None)
                if capture_enabled and callable(capture_frame):
                    current_telemetry = self.adapter.latest_telemetry
                    current_imu = (
                        current_telemetry.imu
                        if current_telemetry is not None
                        else None
                    )
                    current_command = (
                        asdict(self._last_flight_command)
                        if self._last_flight_command is not None
                        else None
                    )
                    capture_frame(
                        image,
                        generation=int(snapshot.generation),
                        frame_id=int(snapshot.frame_id),
                        sim_time_ns=int(snapshot.sim_time_ns),
                        received_monotonic_s=float(snapshot.received_monotonic_s),
                        detector_latency_ms=detector_latency_ms,
                        detections=summaries,
                        tracker={
                            "consecutive": self.tracker.consecutive,
                            "target": asdict(self.tracker.target) if self.tracker.target else None,
                        },
                        imu=current_imu,
                        estimator=self._replay_estimator_fields(),
                        race_status=self.adapter.race_status,
                        generated_command=current_command,
                        sent_command=current_command,
                        phase=(
                            "gate1_reacquisition"
                            if self._post_gate_reacquisition
                            else "gate0_or_preflight"
                        ),
                    )
            except Exception as exc:  # OpenCV errors must fail closed in flight.
                if detector_started_ns is not None:
                    detector_latency_ms = (
                        time.perf_counter_ns() - detector_started_ns
                    ) / 1_000_000.0
                self._latest_raw_detections = []
                self._latest_accepted_target = None
                self._detection_error = f"{type(exc).__name__}: {exc}"
                self.recorder.emit(
                    "frame_processing_error",
                    generation=int(snapshot.generation),
                    frame_id=int(snapshot.frame_id),
                    sim_time_ns=int(snapshot.sim_time_ns),
                    reason=self._detection_error,
                )

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
        count_rate_sample: bool = True,
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
            if count_rate_sample:
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

    async def _send_flight_command(self, command: AttitudeRateCommand) -> None:
        """Send one validated setpoint and remember its completion time."""

        generated_at = time.monotonic()
        frame_token = self._latest_frame_token()
        record_command = getattr(self.recorder, "record_command", None)
        if callable(record_command):
            record_command(
                "generated",
                command,
                monotonic_s=generated_at,
                frame_token=frame_token,
            )
        validate_command(command)
        await self.adapter.send_attitude_rate(command)
        self._last_flight_command = command
        self._last_flight_command_sent_s = time.monotonic()
        if callable(record_command):
            record_command(
                "sent",
                command,
                monotonic_s=self._last_flight_command_sent_s,
                frame_token=frame_token,
            )

    @staticmethod
    def _is_exact_zero_command(command: Optional[AttitudeRateCommand]) -> bool:
        return bool(
            command is not None
            and command.roll_rate == 0.0
            and command.pitch_rate == 0.0
            and command.yaw_rate == 0.0
            and command.thrust == 0.0
        )

    def _defer_snapshot(self, label: str) -> Optional[Dict[str, Any]]:
        """Copy a diagnostic frame in memory; encoding happens after cleanup."""

        if self.recorder.path is None:
            return None
        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if snapshot is None:
            self.recorder.emit("diagnostic_snapshot_unavailable", label=label)
            return None
        image = getattr(snapshot.camera_frame, "image", None)
        if image is None:
            self.recorder.emit("diagnostic_snapshot_unavailable", label=label)
            return None
        # VQ2 snapshots publish a new read-only ndarray per decoded frame.
        # Holding the reference is sufficient; copy/encode only after cleanup.
        self._deferred_pngs.append((str(label), image))
        metadata = {
            "label": str(label),
            "frame_id": int(snapshot.frame_id),
            "sim_time_ns": int(snapshot.sim_time_ns),
            "generation": int(snapshot.generation),
            "received_monotonic_s": float(snapshot.received_monotonic_s),
        }
        return metadata

    def _flush_deferred_snapshots(self) -> Tuple[List[str], List[str]]:
        paths: List[str] = []
        errors: List[str] = []
        pending = self._deferred_pngs
        self._deferred_pngs = []
        for label, image in pending:
            try:
                path = self.recorder.save_png(label, image)
                if path is not None:
                    paths.append(path)
                    self.recorder.emit(
                        "diagnostic_snapshot_saved",
                        label=label,
                        path=path,
                    )
            except Exception as exc:
                message = f"{label}: {type(exc).__name__}: {exc}"
                errors.append(message)
                logger.exception("Could not save deferred diagnostic snapshot %s", label)
                self.recorder.emit(
                    "diagnostic_snapshot_save_failed",
                    label=label,
                    reason=message,
                )
        return paths, errors

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
                zero_is_recent = bool(
                    self._is_exact_zero_command(self._last_flight_command)
                    and self._last_flight_command_sent_s is not None
                    and time.monotonic() - self._last_flight_command_sent_s
                    < CONTROL_PERIOD_S
                )
                if zero_is_recent:
                    self.recorder.emit(
                        "zero_thrust_already_active",
                        gate_index=gate_index_before_cleanup,
                        race_boot_ms=race_boot_before_cleanup,
                    )
                else:
                    cleanup_send_started = time.monotonic()
                    frame_token = self._latest_frame_token()
                    record_command = getattr(self.recorder, "record_command", None)
                    if callable(record_command):
                        record_command(
                            "generated",
                            zero,
                            monotonic_s=cleanup_send_started,
                            frame_token=frame_token,
                        )
                    await self.adapter.send_attitude_rate(zero)
                    self._last_flight_command = zero
                    self._last_flight_command_sent_s = time.monotonic()
                    if callable(record_command):
                        record_command(
                            "sent",
                            zero,
                            monotonic_s=self._last_flight_command_sent_s,
                            frame_token=frame_token,
                        )
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
                await self._send_flight_command(command)
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
            await self._send_flight_command(command)
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

    def _complete_gate0_pass(
        self,
        *,
        race: Any,
        pre_gate_race_boot_ms: int,
        flight_start_s: float,
        crossing_started_s: Optional[float],
        next_tick_s: float,
        max_gate_area: int,
        capture_transition: bool,
    ) -> Dict[str, Any]:
        """Build the sole authoritative gate-0 pass handoff."""

        post_gate_boot_ms = int(race.sim_boot_time_ms)
        if int(race.active_gate_index) != 1:
            raise SafetyAbort("gate-0 pass handoff did not contain gate index 1")
        if post_gate_boot_ms <= int(pre_gate_race_boot_ms):
            raise SafetyAbort(
                "gate-1 race status was not strictly newer than recorded gate 0"
            )
        if self.estimate is None:
            raise SafetyAbort("attitude estimate unavailable at gate-0 pass")
        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if snapshot is None:
            raise SafetyAbort("camera unavailable at authoritative gate-0 pass")

        pass_confirmed_s = time.monotonic()
        paced_deadline = float(next_tick_s)
        if self._last_flight_command_sent_s is not None:
            paced_deadline = max(
                paced_deadline,
                self._last_flight_command_sent_s + CONTROL_PERIOD_S,
            )
        proof = GateTransitionProof(
            pre_gate_race_boot_ms=int(pre_gate_race_boot_ms),
            post_gate_race_boot_ms=post_gate_boot_ms,
            flight_started_monotonic_s=float(flight_start_s),
            crossing_started_monotonic_s=(
                float(crossing_started_s) if crossing_started_s is not None else None
            ),
            pass_confirmed_monotonic_s=pass_confirmed_s,
            next_control_deadline_s=paced_deadline,
            vision_generation=int(snapshot.generation),
            vision_frame_id=int(snapshot.frame_id),
            vision_sim_time_ns=int(snapshot.sim_time_ns),
            vision_received_monotonic_s=float(snapshot.received_monotonic_s),
            pass_rpy_rad=tuple(
                float(value) for value in self.estimate.orientation.to_euler()
            ),
        )
        self._gate0_transition_proof = proof
        if capture_transition:
            self._defer_snapshot("gate1_race_credit")
        result = {
            "gate0_passed": True,
            "gate_transition_proved": True,
            "pre_gate_race_boot_ms": proof.pre_gate_race_boot_ms,
            "race_boot_ms": proof.post_gate_race_boot_ms,
            "last_gate_race_time": race.last_gate_race_time,
            "max_gate_area_px": int(max_gate_area),
            "crossing_confirmation_used": crossing_started_s is not None,
            "crossing_confirmation_elapsed_s": (
                pass_confirmed_s - crossing_started_s
                if crossing_started_s is not None
                else None
            ),
            "flight_elapsed_s": pass_confirmed_s - flight_start_s,
        }
        self.recorder.emit("gate0_pass_proved", **result)
        return result

    async def _run_gate0(
        self,
        context: StartContext,
        *,
        capture_transition: bool = False,
    ) -> Dict[str, Any]:
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
        last_gate0_race_boot_ms: Optional[int] = int(context.go_boot_ms)
        while True:
            now = time.monotonic()
            elapsed = now - flight_start
            if elapsed >= GATE0_FLIGHT_TIMEOUT_S:
                raise SafetyAbort("gate-0 wall-time limit reached")
            self._sample()
            race = self.adapter.race_status
            assert race is not None and self.estimate is not None
            if int(race.active_gate_index) == 0:
                last_gate0_race_boot_ms = int(race.sim_boot_time_ms)
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
                if capture_transition:
                    self._vision_diagnostic_logging = True
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
                if last_gate0_race_boot_ms is None:
                    raise SafetyAbort("gate 1 appeared without a recorded gate-0 packet")
                return self._complete_gate0_pass(
                    race=race,
                    pre_gate_race_boot_ms=last_gate0_race_boot_ms,
                    flight_start_s=flight_start,
                    crossing_started_s=crossing_started_s,
                    next_tick_s=next_tick,
                    max_gate_area=max_gate_area,
                    capture_transition=capture_transition,
                )

            if crossing_confirming:
                if crossing_started_s is None:
                    crossing_started_s = now
                    if last_gate0_race_boot_ms is None:
                        raise SafetyAbort(
                            "crossing confirmation lacks a recorded gate-0 packet"
                        )
                    crossing_race_boot_ms = last_gate0_race_boot_ms
                    if capture_transition:
                        self._vision_diagnostic_logging = True
                        self._defer_snapshot("gate0_visual_loss")
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
                        return self._complete_gate0_pass(
                            race=race,
                            pre_gate_race_boot_ms=crossing_race_boot_ms,
                            flight_start_s=flight_start,
                            crossing_started_s=crossing_started_s,
                            next_tick_s=next_tick,
                            max_gate_area=max_gate_area,
                            capture_transition=capture_transition,
                        )
                    raise SafetyAbort(f"gate-0 crossing {decision.replace('_', ' ')}")
                command = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
                await self._send_flight_command(command)
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
            await self._send_flight_command(command)
            self._record_tick("gate0", elapsed, command)
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def _observe_gate1(
        self,
        gate0_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect a bounded, zero-thrust view after a proved gate-0 pass."""

        proof = self._gate0_transition_proof
        if proof is None or not gate0_details.get("gate_transition_proved"):
            raise SafetyAbort("gate-1 observation lacks an authoritative transition proof")
        if (
            int(gate0_details.get("pre_gate_race_boot_ms", -1))
            != proof.pre_gate_race_boot_ms
            or int(gate0_details.get("race_boot_ms", -1))
            != proof.post_gate_race_boot_ms
            or proof.post_gate_race_boot_ms <= proof.pre_gate_race_boot_ms
        ):
            raise SafetyAbort("gate-1 observation transition proof is inconsistent")

        race = self.adapter.race_status
        if (
            race is None
            or int(race.active_gate_index) != 1
            or int(race.sim_boot_time_ms) < proof.post_gate_race_boot_ms
        ):
            raise SafetyAbort("race status no longer matches the proved gate-1 transition")
        watermark = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if watermark is None:
            raise SafetyAbort("camera unavailable at gate-1 observation handoff")
        if int(watermark.generation) != proof.vision_generation:
            raise SafetyAbort("vision generation changed after gate-0 passage")
        if (
            int(watermark.frame_id) < proof.vision_frame_id
            or int(watermark.sim_time_ns) < proof.vision_sim_time_ns
            or float(watermark.received_monotonic_s)
            < proof.vision_received_monotonic_s
        ):
            raise SafetyAbort("camera snapshot regressed after gate-0 passage")

        # Deliberately skip any frame that existed before the tracker reset.
        # The vision receiver and its generation remain untouched.
        self._last_frame_identity = (
            int(watermark.generation),
            int(watermark.frame_id),
        )
        self._last_frame_sim_ns = int(watermark.sim_time_ns)
        self._latest_detection_frame_id = int(watermark.frame_id)
        self._latest_detection_frame_sim_ns = int(watermark.sim_time_ns)
        self._latest_detection_generation = int(watermark.generation)
        self._latest_detection_received_s = float(watermark.received_monotonic_s)
        self._latest_raw_detections = []
        self.tracker.reset()
        self._latest_accepted_target = None
        self._post_gate_reacquisition = True
        self._vision_diagnostic_logging = True

        hard_deadline = post_gate_observation_deadline(
            pass_confirmed_s=proof.pass_confirmed_monotonic_s,
            flight_started_s=proof.flight_started_monotonic_s,
            crossing_started_s=proof.crossing_started_monotonic_s,
        )
        observation_started_s = time.monotonic()
        if observation_started_s >= hard_deadline:
            self._post_gate_reacquisition = False
            raise SafetyAbort("gate-1 observation has no remaining safety budget")

        next_tick = max(
            proof.next_control_deadline_s,
            (
                self._last_flight_command_sent_s + CONTROL_PERIOD_S
                if self._last_flight_command_sent_s is not None
                else observation_started_s
            ),
        )
        zero = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
        last_processed_token = (
            int(watermark.generation),
            int(watermark.frame_id),
            int(watermark.sim_time_ns),
        )
        qualifying_frames: List[Dict[str, Any]] = []
        strict_high_rate_samples = 0
        self.recorder.emit(
            "post_gate_observation_started",
            pre_gate_race_boot_ms=proof.pre_gate_race_boot_ms,
            post_gate_race_boot_ms=proof.post_gate_race_boot_ms,
            hard_deadline_monotonic_s=hard_deadline,
            budget_s=hard_deadline - observation_started_s,
            watermark={
                "generation": watermark.generation,
                "frame_id": watermark.frame_id,
                "sim_time_ns": watermark.sim_time_ns,
                "received_monotonic_s": watermark.received_monotonic_s,
            },
        )

        try:
            initial_wait = min(next_tick, hard_deadline) - time.monotonic()
            if initial_wait > 0.0:
                await asyncio.sleep(initial_wait)
            while True:
                if time.monotonic() >= hard_deadline:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                self._sample()
                self._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=False,
                    enforce_benign_pad_budget=False,
                )
                now = time.monotonic()
                if now >= hard_deadline:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                race = self.adapter.race_status
                if race is None or int(race.active_gate_index) != 1:
                    gate_index = race.active_gate_index if race is not None else None
                    raise SafetyAbort(
                        f"gate index changed during gate-1 observation ({gate_index})"
                    )
                if int(race.sim_boot_time_ms) < proof.post_gate_race_boot_ms:
                    raise SafetyAbort("race clock regressed below the gate-1 proof")
                assert self.estimate is not None
                roll, pitch, _yaw = self.estimate.orientation.to_euler()
                if (
                    abs(roll - proof.pass_rpy_rad[0])
                    > POST_GATE_MAX_ATTITUDE_DELTA_RAD
                    or abs(pitch - proof.pass_rpy_rad[1])
                    > POST_GATE_MAX_ATTITUDE_DELTA_RAD
                ):
                    raise SafetyAbort("attitude changed over 5deg during zero-thrust observation")
                peak_rate = max(abs(value) for value in self.estimate.body_rates)
                if peak_rate > POST_GATE_IMMEDIATE_MAX_BODY_RATE_RAD_S:
                    raise SafetyAbort(
                        "body rate exceeded 1.0rad/s during gate-1 observation"
                    )
                strict_high_rate_samples = (
                    strict_high_rate_samples + 1
                    if peak_rate > POST_GATE_SUSTAINED_MAX_BODY_RATE_RAD_S
                    else 0
                )
                if strict_high_rate_samples >= 2:
                    raise SafetyAbort(
                        "body rate exceeded 0.5rad/s for two gate-1 observation samples"
                    )

                frame_token: Optional[Tuple[int, int, int]] = None
                if (
                    self._latest_detection_generation is not None
                    and self._latest_detection_frame_id is not None
                    and self._latest_detection_frame_sim_ns is not None
                ):
                    frame_token = (
                        self._latest_detection_generation,
                        self._latest_detection_frame_id,
                        self._latest_detection_frame_sim_ns,
                    )
                if frame_token is not None and frame_token != last_processed_token:
                    generation, frame_id, sim_time_ns = frame_token
                    received_s = self._latest_detection_received_s
                    if generation != int(watermark.generation):
                        raise SafetyAbort("vision generation changed during gate-1 observation")
                    if (
                        frame_id <= int(watermark.frame_id)
                        or sim_time_ns <= int(watermark.sim_time_ns)
                        or received_s is None
                        or received_s <= float(watermark.received_monotonic_s)
                    ):
                        raise SafetyAbort("post-pass camera frame did not advance strictly")
                    last_processed_token = frame_token
                    accepted = self._latest_accepted_target
                    if accepted is None or is_crossing_residue(accepted):
                        self.recorder.emit(
                            "post_gate_candidate_reset",
                            frame_id=frame_id,
                            sim_time_ns=sim_time_ns,
                            reason=(
                                "crossing_residue"
                                if accepted is not None
                                else "no_continuous_candidate"
                            ),
                        )
                        self.tracker.reset()
                        qualifying_frames = []
                    else:
                        record = {
                            "frame_id": accepted.frame_id,
                            "sim_time_ns": accepted.sim_time_ns,
                            "received_monotonic_s": accepted.received_monotonic_s,
                            "center_px": [accepted.center_x, accepted.center_y],
                            "bbox_xywh_px": list(accepted.bbox),
                            "confidence": accepted.confidence,
                            "tracker_streak": self.tracker.consecutive,
                            "rpy_rad": list(self.estimate.orientation.to_euler()),
                            "body_rates_rad_s": list(self.estimate.body_rates),
                        }
                        if self.tracker.consecutive == 1:
                            qualifying_frames = [record]
                        else:
                            qualifying_frames.append(record)
                            qualifying_frames = qualifying_frames[
                                -self.tracker.consecutive :
                            ]
                        self.recorder.emit("post_gate_candidate_frame", **record)
                        candidate_checked_s = time.monotonic()
                        if candidate_checked_s >= hard_deadline:
                            raise SafetyAbort(
                                "gate-1 observation timed out before three frames"
                            )
                        if (
                            self.tracker.consecutive >= POST_GATE_REQUIRED_FRAMES
                            and len(qualifying_frames) >= POST_GATE_REQUIRED_FRAMES
                            and accepted.age_s(candidate_checked_s) <= MAX_VISION_AGE_S
                        ):
                            # Recheck every generic guard immediately before
                            # accepting the observation result.
                            self._watchdog(
                                require_target=False,
                                allow_benign_pad_contact=False,
                                enforce_benign_pad_budget=False,
                            )
                            accepted_at_s = time.monotonic()
                            if accepted_at_s >= hard_deadline:
                                raise SafetyAbort(
                                    "gate-1 observation timed out before three frames"
                                )
                            final_race = self.adapter.race_status
                            if (
                                final_race is None
                                or int(final_race.active_gate_index) != 1
                                or int(final_race.sim_boot_time_ms)
                                < proof.post_gate_race_boot_ms
                            ):
                                raise SafetyAbort(
                                    "race status changed at gate-1 observation acceptance"
                                )
                            result = {
                                "gate1_observed": True,
                                "observation_elapsed_s": (
                                    accepted_at_s - observation_started_s
                                ),
                                "frame_count": POST_GATE_REQUIRED_FRAMES,
                                "frames": qualifying_frames[-POST_GATE_REQUIRED_FRAMES:],
                                "final_gate_bbox": list(accepted.bbox),
                                "final_gate_center": [
                                    accepted.center_x,
                                    accepted.center_y,
                                ],
                                "race_boot_ms": int(final_race.sim_boot_time_ms),
                                "gate_index": int(final_race.active_gate_index),
                            }
                            return result

                # Leave the final setpoint slot for cleanup, preventing a
                # zero-command burst at an odd nested deadline.
                if hard_deadline - time.monotonic() <= CONTROL_PERIOD_S:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                self._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=False,
                    enforce_benign_pad_budget=False,
                    count_rate_sample=False,
                )
                send_checked_s = time.monotonic()
                if hard_deadline - send_checked_s <= CONTROL_PERIOD_S:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                send_race = self.adapter.race_status
                if (
                    send_race is None
                    or int(send_race.active_gate_index) != 1
                    or int(send_race.sim_boot_time_ms)
                    < proof.post_gate_race_boot_ms
                ):
                    raise SafetyAbort(
                        "race status changed before gate-1 observation setpoint"
                    )

                await self._send_flight_command(zero)
                self._record_tick(
                    "gate0-observe/post-pass",
                    send_checked_s - observation_started_s,
                    zero,
                )
                next_tick = next_control_deadline(next_tick, time.monotonic())
                await asyncio.sleep(
                    max(0.0, min(next_tick, hard_deadline) - time.monotonic())
                )
        finally:
            self._post_gate_reacquisition = False

    async def run_powered_stage(self, stage: str) -> StageResult:
        if stage not in {"sign-id", "hover", "gate0", "gate0-observe"}:
            raise ValueError(f"unsupported powered stage: {stage}")
        started = time.monotonic()
        reason = "unknown"
        success = False
        details: Dict[str, Any] = {}
        gate_before: Optional[int] = None
        gate_after: Optional[int] = None
        cleanup_confirmed = False
        try:
            self._deferred_pngs = []
            self._post_gate_last_frame = None
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
            elif stage == "gate0":
                details = await self._run_gate0(context)
            else:
                gate0_details = await self._run_gate0(
                    context,
                    capture_transition=True,
                )
                details = {"gate0": gate0_details}
                try:
                    details["gate1_observation"] = await self._observe_gate1(
                        gate0_details
                    )
                except SafetyAbort as exc:
                    details["gate1_observation"] = {
                        "gate1_observed": False,
                        "reason": str(exc),
                    }
                    raise
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
            post_cleanup_diagnostic_errors: List[str] = []
            if cleanup_confirmed and self._post_gate_last_frame is not None:
                token, image = self._post_gate_last_frame
                observation = details.get("gate1_observation", {})
                if observation.get("gate1_observed"):
                    final_frame = observation.get("frames", [{}])[-1]
                    if (
                        token[1] == final_frame.get("frame_id")
                        and token[2] == final_frame.get("sim_time_ns")
                    ):
                        self._deferred_pngs.append(("gate1_acquired", image))
                        self.recorder.emit("next_gate_reacquired", **observation)
                    else:
                        post_cleanup_diagnostic_errors.append(
                            "acquired-frame PNG token did not match the accepted target"
                        )
                else:
                    self._deferred_pngs.append(
                        ("gate1_observation_terminal", image)
                    )
            if cleanup_confirmed:
                diagnostic_paths, diagnostic_errors = self._flush_deferred_snapshots()
                diagnostic_errors = (
                    post_cleanup_diagnostic_errors + diagnostic_errors
                )
            else:
                self._deferred_pngs = []
                diagnostic_paths = []
                diagnostic_errors = [
                    "diagnostic images not encoded because cleanup was unconfirmed"
                ]
            if diagnostic_paths:
                details["diagnostic_pngs"] = diagnostic_paths
            if diagnostic_errors:
                details["diagnostic_errors"] = diagnostic_errors
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


async def run_live(
    stage: str,
    address: str,
    record: Optional[str],
    *,
    replay_bundle: Optional[str] = None,
    recording_approved: bool = False,
) -> StageResult:
    if type(recording_approved) is not bool:
        raise TypeError("recording_approved must be an exact bool")
    if replay_bundle is not None and recording_approved is not True:
        raise PermissionError(
            "programmatic replay capture requires explicit recording_approved=True"
        )
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
    )
    replay = None
    if replay_bundle is not None:
        (
            async_replay_recorder,
            replay_bundle_writer,
            capture_environment_fingerprint,
            capture_git_provenance,
        ) = _replay_capture_dependencies()
        repo_root = Path(__file__).resolve().parents[1]
        commit_hash, dirty_diff_hash, code_hash = capture_git_provenance(repo_root)
        replay_writer = replay_bundle_writer(
            replay_bundle,
            metadata={
                    "simulator_build": "3385",
                    "stage": stage,
                    "mavlink_address": address,
                    "capture_kind": "private-development-session",
                    "commit_hash": commit_hash,
                    "dirty_diff_hash": dirty_diff_hash,
                    "code_hash": code_hash,
                    "environment_fingerprint": capture_environment_fingerprint(),
                    "runner_evaluator_version": "vq2-runner-capture/1",
                    # Frozen replay-evaluator RNG seed.  This is independent
                    # of simulator randomness and is bound into T1 identity.
                    "seed": 42,
                    "detector": {
                        "class": "VQ2GateDetector",
                        "image_size_px": [640, 360],
                        "min_area": 500,
                        "max_area": 500000,
                        "max_aspect_ratio": 3.0,
                        "min_confidence": 0.10,
                        "hsv_ranges": [
                            [[0, 50, 100], [12, 255, 255]],
                            [[150, 50, 100], [180, 255, 255]],
                        ],
                    },
                    "controller_envelope": {
                        "control_hz": CONTROL_HZ,
                        "max_roll_pitch_command_rate_rad_s": MAX_COMMAND_RATE_RAD_S,
                        "yaw_rate_rad_s": 0.0,
                        "max_thrust": 0.35,
                    },
            },
            repo_root=repo_root,
        )
        try:
            replay = async_replay_recorder(replay_writer)
        except BaseException as exc:
            try:
                replay_writer.abort(
                    "async replay recorder construction failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            except BaseException as cleanup_exc:
                exc.add_note(
                    "Replay writer abort also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
    vision: Optional[VQ2VisionThread] = None
    recorder: Optional[JsonlRecorder] = None
    runner: Optional[VQ2Runner] = None
    connected = False
    result: Optional[StageResult] = None
    failure: Optional[str] = None
    capture_stats = None
    primary_exception: Optional[BaseException] = None
    primary_traceback = None
    cleanup_exceptions: List[BaseException] = []
    try:
        # The replay writer thread already exists at this point.  Every later
        # constructor is therefore inside the same cleanup ownership region.
        vision = VQ2VisionThread(
            on_snapshot=(
                replay.capture_decoded_snapshot if replay is not None else None
            )
        )
        recorder = JsonlRecorder(record, replay=replay)
        runner = VQ2Runner(adapter, vision, recorder=recorder)
        await adapter.connect(address)
        connected = True
        preflight = await runner.preflight()
        if stage == "preflight":
            result = StageResult(
                stage=stage,
                success=True,
                reason="passive preflight completed; no flight command sent",
                duration_s=0.0,
                gate_index_before=preflight.get("race_gate_index"),
                gate_index_after=preflight.get("race_gate_index"),
                cleanup_confirmed=True,
                details=preflight,
            )
        else:
            result = await runner.run_powered_stage(stage)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        primary_exception = exc
        primary_traceback = exc.__traceback__
    finally:
        if vision is not None:
            try:
                vision.stop()
            except BaseException as exc:
                cleanup_exceptions.append(exc)
                if replay is not None:
                    replay.fail(
                        f"vision termination not proved before replay seal: "
                        f"{type(exc).__name__}: {exc}"
                    )
            try:
                vision_capture_stats = asdict(vision.stats())
            except BaseException as exc:
                cleanup_exceptions.append(exc)
                vision_capture_stats = {
                    "stats_error": f"{type(exc).__name__}: {exc}"
                }
                if replay is not None:
                    replay.fail(
                        f"vision stats unavailable: {type(exc).__name__}: {exc}"
                    )
        else:
            vision_capture_stats = {"unavailable": True}
            if replay is not None:
                replay.fail("vision construction failed before capture ownership")
        try:
            if connected:
                await adapter.disconnect()
        except BaseException as exc:
            cleanup_exceptions.append(exc)
        base_outcome = (
            asdict(result)
            if result is not None
            else {"success": False, "failure": failure or "runner did not return"}
        )
        base_outcome["vision_capture_stats"] = vision_capture_stats
        if cleanup_exceptions:
            base_outcome["transport_cleanup_errors"] = [
                f"{type(exc).__name__}: {exc}" for exc in cleanup_exceptions
            ]
        try:
            if recorder is not None:
                capture_stats = recorder.close(outcome=base_outcome)
            elif replay is not None:
                replay.fail("recorder construction failed before capture ownership")
                capture_stats = replay.close(outcome=base_outcome)
        except BaseException as exc:
            cleanup_exceptions.append(exc)
    if primary_exception is not None:
        raise primary_exception.with_traceback(primary_traceback)
    if cleanup_exceptions:
        raise cleanup_exceptions[0]
    assert result is not None
    return replay_capture_result(
        result,
        capture_requested=replay is not None,
        capture_stats=capture_stats,
    )


def replay_capture_result(
    result: StageResult,
    *,
    capture_requested: bool,
    capture_stats: Any,
) -> StageResult:
    """Fail closed when an explicitly requested replay is incomplete."""

    if capture_requested and (capture_stats is None or not capture_stats.complete):
        replay_details = (
            asdict(capture_stats)
            if capture_stats is not None
            else {"complete": False, "reason": "capture stats unavailable"}
        )
        details = dict(result.details or {})
        details["replay_capture"] = replay_details
        result = replace(
            result,
            success=False,
            reason=f"{result.reason}; replay capture incomplete",
            details=details,
        )
    elif capture_requested and capture_stats is not None:
        details = dict(result.details or {})
        details["replay_capture"] = asdict(capture_stats)
        result = replace(result, details=details)
    return result


def _default_record_path(stage: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return str(Path("captures") / f"vq2_{stage}_{stamp}.jsonl.gz")


def _default_replay_path(stage: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return str(Path("captures") / "replays" / f"vq2_{stage}_{stamp}.vq2replay")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded AIGP VQ2 training runner")
    parser.add_argument(
        "--stage",
        choices=("preflight", "sign-id", "hover", "gate0", "gate0-observe"),
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
    parser.add_argument(
        "--replay-bundle",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "write a private decoded-frame replay bundle outside normal Git; "
            "requires --recording-approved"
        ),
    )
    parser.add_argument(
        "--recording-approved",
        action="store_true",
        help="attest that organizer approval/credentials permit this recording",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.replay_bundle is not None and not args.recording_approved:
        parser.error("--replay-bundle requires explicit --recording-approved")
    record = _default_record_path(args.stage) if args.record == "auto" else args.record
    replay_bundle = (
        _default_replay_path(args.stage)
        if args.replay_bundle == "auto"
        else args.replay_bundle
    )
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = asyncio.run(
        run_live(
            args.stage,
            args.address,
            record,
            replay_bundle=replay_bundle,
            recording_approved=args.recording_approved,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
