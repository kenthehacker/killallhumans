"""Generic rolling-graph coordinator for the build-3385 visual course.

This module owns no reset, arm, transport, collision, or cleanup mechanism.
Those remain on :class:`scripts.aigp_vq2_run.VQ2Runner` and are exposed through
the narrow host protocol below.  The coordinator repeats one gate-agnostic
lifecycle until authoritative race status says ``race_finished``:

``approach -> reviewed passage -> crossing latch -> race credit -> promotion``.

Only UDP-camera-derived rolling identities, HIGHRES_IMU attitude, and exact
live race ingress are consumed.  There is no pose, odometry, metric gate
coordinate, or simulator gate-map input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol

from competition.adapter import (
    AttitudeRateCommand,
    RaceActiveBoundaryChangedBeforeWire,
)
from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    AssociationEvidence,
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualTrackRole,
    VisualTrackSample,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    GateGraphError,
    RaceStatusProvenanceBasis,
)
from planning.vq2_visual_approach import (
    RollingVisualApproachServo,
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
    VisualApproachPassageAdmission,
    VisualApproachPassageSafetyUnavailable,
    VisualApproachRefusal,
)
from planning.vq2_visual_recovery import (
    RECOVERY_HISTORY_SAMPLE_COUNT,
    RECOVERY_MAX_START_DELAY_AFTER_CREDIT_S,
    VisualRecoveryRefusal,
    require_recovery_continuation,
    require_transition_recovery_admission,
)
from planning.vq2_visual_servo import (
    MAX_NEXT_GATE_BLEND,
    MAX_VISUAL_TARGET_PITCH_RAD,
    MAX_VISUAL_TARGET_ROLL_RAD,
    MAX_VISUAL_SEGMENT_DURATION_S,
    MAX_VISUAL_THRUST,
    MAX_VISUAL_YAW_RATE_RAD_S,
    MIN_VISUAL_TARGET_PITCH_RAD,
    MIN_VISUAL_THRUST,
    PREPASS_CURRENT_MAX_APPARENT_SCALE,
    VisualServoOutput,
    VisualTarget,
)
from scripts.aigp_vq2_yaw_profile import (
    DEFAULT_YAW_CALIBRATION_PROFILE_PATH,
    YAW_CALIBRATION_PROFILE_ID,
    YAW_CALIBRATION_PROFILE_SCHEMA,
    YAW_CALIBRATION_PROFILE_SHA256,
    YAW_CALIBRATION_SOURCE_COMMIT,
    YAW_CONTROLLER_TO_BODY_SIGN,
    YAW_CONTROLLER_TO_IMAGE_SIGN,
    YAW_CONTROL_HOLD_HORIZON_S,
    YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD,
    YAW_MAX_CALIBRATION_MEASURED_RATE_RAD_S,
    YAW_MAX_COMMAND_RATE_RAD_S,
    YAW_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S,
    YAW_MAX_GYRO_RESPONSE_DELAY_S,
    YAW_OBSERVED_MAX_MEASURED_RATE_RAD_S,
    YAW_CALIBRATION_PLAN_ID,
    YAW_CALIBRATION_PLAN_SHA256,
    load_yaw_calibration_profile,
    yaw_calibration_profile_evidence,
)


VISUAL_COURSE_STAGE = "visual-course"
VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON = (
    "visual receiver advanced beyond the admitted command target"
)
MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S = 0.10
MAX_CONSECUTIVE_VISUAL_PROPOSAL_SUPERSESSIONS = 4
VISUAL_COURSE_YAW_PROFILE_SCHEMA = YAW_CALIBRATION_PROFILE_SCHEMA
INITIAL_PAD_PRELOAD_DURATION_S = 0.15
INITIAL_PAD_PRELOAD_THRUST = 0.26
# Exact normalized form of the build-3385 Gate-0 pixel-space collective that
# already completed Gate 0 under the same launch pitch schedule:
# 0.275 + 0.040 * ((180 - row) / 90) - 0.00070 * row_rate.
GATE0_PROVED_COLLECTIVE_BASE = 0.275
GATE0_PROVED_COLLECTIVE_ERROR_GAIN = 0.080
GATE0_PROVED_COLLECTIVE_RATE_GAIN = 0.126
GATE0_PROVED_COLLECTIVE_MAX_ABS_ERROR = 0.50
GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE = 5.0 / 3.0
GATE0_PROVED_COLLECTIVE_RATE_FILTER_ALPHA = 0.35
GATE0_PROVED_COLLECTIVE_BASIS = "proved-gate0-normalized-collective-v1"
GATE0_PROVED_NEXT_PREVIEW_ERROR_GAIN = 0.080
GATE0_PROVED_NEXT_PREVIEW_MAX_THRUST_DELTA = 0.012
GATE0_PROVED_NEXT_PREVIEW_BASIS = (
    "proved-gate0-reviewed-next-preview-collective-v1"
)
CURRENT_ADVANCE_CROSSING_BASIS = "current-advance-corridor-v1"
RETAINED_ADVANCE_CROSSING_BASIS = (
    "retained-advance-close-alignment-dwell-v1"
)
RETAINED_ADVANCE_WIRE_PROJECTED_CROSSING_BASIS = (
    "retained-advance-wire-projected-close-alignment-dwell-v1"
)
RETAINED_ADJACENT_SCALE_JUMP_CROSSING_BASIS = (
    "retained-adjacent-scale-threshold-crossing-v1"
)
# Attempts 21 and 22 sampled the same clean Gate-0 contour expansion after a
# completed retained-safe dwell.  The previous admitted state projected across
# the unchanged crossing plane before the exact adjacent observation, while
# the tracker rate update itself was intentionally too discontinuous for the
# ordinary retained rule.  Keep this exception narrower than either history:
# it is one-publication-only, association-bounded, and adds no command
# authority.
ADJACENT_SCALE_JUMP_MIN_PREDECESSOR_LOG_SCALE = -0.85
ADJACENT_SCALE_JUMP_MAX_LOG_DELTA = 0.18
ADJACENT_SCALE_JUMP_MAX_OBSERVATION_GAP_S = 0.036
ADJACENT_SCALE_JUMP_MIN_DETECTION_CONFIDENCE = 0.90
ADJACENT_SCALE_JUMP_MIN_ASSOCIATION_CONFIDENCE = 0.75
ADJACENT_SCALE_JUMP_MAX_ASSOCIATION_COST = 0.21
ADJACENT_SCALE_JUMP_MIN_BBOX_IOU = 0.70
ADJACENT_SCALE_JUMP_MAX_CENTER_RESIDUAL_NORM = 0.09
ADJACENT_SCALE_JUMP_MAX_ABS_LOG_WIDTH_CHANGE = 0.29
ADJACENT_SCALE_JUMP_MAX_ABS_LOG_HEIGHT_CHANGE = 0.06
ADJACENT_SCALE_JUMP_MAX_ABS_LOG_AREA_RESIDUAL = 0.26
ADJACENT_SCALE_JUMP_MAX_ABS_CURRENT_HORIZONTAL_RATE_S = 1.50
ADJACENT_SCALE_JUMP_MAX_CURRENT_LOG_SCALE_RATE_S = 3.40
CENSORED_PASSAGE_COAST_BASIS = (
    "latched-clean-attitude-close-censored-passage-v1"
)
CENSORED_PASSAGE_BOTTOM_TRANSITION_BASIS = (
    "latched-clean-attitude-bottom-transition-v1"
)
APPROACH_PREVIEW_REQUALIFICATION_BASIS = (
    "fresh-current-corridor-sealed-next-identity-v1"
)
# Attempt 15's hard rate-only discontinuity occurred at publication 104.
# The same continuously visible next identity first re-earned every ordinary
# current-corridor and next-target predicate at publication 117: 12 newer
# observations, publication delta 13, and 0.437 seconds later.  These bounds
# create no preview command authority; they only bound one fresh-planner
# opportunity to earn the unchanged ordinary admission.
MAX_APPROACH_PREVIEW_REQUALIFICATION_FRESH_FRAMES = 12
MAX_APPROACH_PREVIEW_REQUALIFICATION_PUBLICATION_DELTA = 13
MAX_APPROACH_PREVIEW_REQUALIFICATION_DURATION_S = 0.45
# Pub104->117 took 0.4534 seconds on the compact event wall clock.  Keep one
# additional 50 Hz scheduling interval plus recorder jitter inside this
# separate wall ceiling; the ordinary camera watchdog remains authoritative
# for an actual stalled stream.
MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S = 0.50
_REQUALIFIABLE_APPROACH_PREVIEW_VIOLATIONS = (
    "current_vertical_rate",
    "current_log_scale_rate",
)
_YAW_PROFILE_ISSUER = object()


def _gate0_proved_vertical_collective(
    vertical: float,
    filtered_vertical_rate: float,
) -> float:
    """Return the live-proved Gate-0 collective in normalized image space."""

    vertical = float(vertical)
    filtered_vertical_rate = float(filtered_vertical_rate)
    if not all(
        math.isfinite(value)
        for value in (vertical, filtered_vertical_rate)
    ):
        raise ValueError("Gate-0 collective inputs must be finite")
    bounded_vertical = max(
        -GATE0_PROVED_COLLECTIVE_MAX_ABS_ERROR,
        min(GATE0_PROVED_COLLECTIVE_MAX_ABS_ERROR, vertical),
    )
    bounded_rate = max(
        -GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE,
        min(
            GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE,
            filtered_vertical_rate,
        ),
    )
    requested = (
        GATE0_PROVED_COLLECTIVE_BASE
        - GATE0_PROVED_COLLECTIVE_ERROR_GAIN * bounded_vertical
        - GATE0_PROVED_COLLECTIVE_RATE_GAIN * bounded_rate
    )
    return max(
        MIN_VISUAL_THRUST,
        min(MAX_VISUAL_THRUST, requested),
    )


def _gate0_proved_next_preview_collective_delta(
    *,
    proved_collective: float,
    current_vertical: float,
    next_vertical: float,
    preview_blend: float,
) -> float:
    """Add bounded lift only when preview reinforces the proved correction."""

    proved_collective = float(proved_collective)
    current_vertical = float(current_vertical)
    next_vertical = float(next_vertical)
    preview_blend = float(preview_blend)
    if not all(
        math.isfinite(value)
        for value in (
            proved_collective,
            current_vertical,
            next_vertical,
            preview_blend,
        )
    ):
        raise ValueError("Gate-0 next-preview collective inputs must be finite")
    if not MIN_VISUAL_THRUST <= proved_collective <= MAX_VISUAL_THRUST:
        raise ValueError(
            "Gate-0 next-preview collective base is outside its fixed envelope"
        )
    if not 0.0 <= preview_blend <= MAX_NEXT_GATE_BLEND:
        raise ValueError(
            "Gate-0 next-preview blend is outside its immutable ceiling"
        )
    # Never weaken or reverse the already live-proved current-aperture law.
    # Image-down negative means the current aperture is high and already asks
    # for lift.  A still-higher reviewed next aperture may add lift; every
    # other geometry receives no new collective authority.
    if (
        preview_blend == 0.0
        or current_vertical > 0.0
        or next_vertical >= current_vertical
    ):
        return 0.0
    requested = (
        -GATE0_PROVED_NEXT_PREVIEW_ERROR_GAIN
        * preview_blend
        * (next_vertical - current_vertical)
    )
    return max(
        0.0,
        min(
            requested,
            GATE0_PROVED_NEXT_PREVIEW_MAX_THRUST_DELTA,
            MAX_VISUAL_THRUST - proved_collective,
        ),
    )


def _gate0_proved_collective_with_exact_next_preview(
    *,
    proved_collective: float,
    current_target: Any,
    next_target: Any,
    latched_next_track_id: Optional[str],
    servo_output: Any,
) -> tuple[float, float]:
    """Apply only an exact, identity-latched same-publication preview."""

    blend = float(servo_output.next_gate_blend)
    if blend == 0.0:
        return float(proved_collective), 0.0
    if next_target is None:
        raise ValueError("Gate-0 next-preview target is missing")
    if next_target.frame_token != current_target.frame_token:
        raise ValueError(
            "Gate-0 next-preview targets do not share one exact publication"
        )
    if (
        latched_next_track_id is None
        or latched_next_track_id != next_target.track_id
    ):
        raise ValueError(
            "Gate-0 next-preview identity lacks the persistent coordinator "
            "latch"
        )
    current_reviewed_track_id = getattr(
        servo_output,
        "reviewed_next_track_id",
        None,
    )
    if (
        current_reviewed_track_id is not None
        and current_reviewed_track_id != next_target.track_id
    ):
        raise ValueError(
            "Gate-0 next-preview identity conflicts with current servo review"
        )
    next_vertical = getattr(
        servo_output,
        "next_vertical_error_image_down",
        None,
    )
    if (
        next_vertical is None
        or bool(getattr(current_target, "vertical_censored", False))
        or bool(getattr(next_target, "vertical_censored", False))
    ):
        return float(proved_collective), 0.0
    current_vertical = float(current_target.normalized_y_down)
    next_vertical = float(next_vertical)
    if (
        float(servo_output.vertical_error_image_down) != current_vertical
        or float(next_target.normalized_y_down) != next_vertical
    ):
        raise ValueError(
            "Gate-0 next-preview geometry diverged from reviewed targets"
        )
    delta = _gate0_proved_next_preview_collective_delta(
        proved_collective=proved_collective,
        current_vertical=current_vertical,
        next_vertical=next_vertical,
        preview_blend=blend,
    )
    return float(proved_collective) + delta, delta


@dataclass(slots=True)
class _Gate0ProvedCollectiveState:
    last_token_key: Optional[tuple[str, int, int, int]] = None
    last_received_monotonic_s: Optional[float] = None
    last_vertical: Optional[float] = None
    filtered_vertical_rate: float = 0.0

    def observe(self, target: Any) -> tuple[float, float]:
        """Apply the proved 0.35 filter on exact graph receiver timing."""

        token = target.frame_token
        token_key = (
            str(token.stream_id),
            int(token.generation),
            int(token.frame_id),
            int(token.publication_sequence),
        )
        received = float(target.received_monotonic_s)
        vertical = float(target.normalized_y_down)
        if not math.isfinite(received) or not math.isfinite(vertical):
            raise ValueError("Gate-0 collective observation must be finite")
        if self.last_token_key is not None:
            if (
                token_key[0] != self.last_token_key[0]
                or token_key[1] != self.last_token_key[1]
                or token_key[3] <= self.last_token_key[3]
                or self.last_received_monotonic_s is None
                or self.last_vertical is None
            ):
                raise ValueError(
                    "Gate-0 collective publication did not advance"
                )
            if token_key[2] != self.last_token_key[2]:
                elapsed = received - self.last_received_monotonic_s
                if elapsed > 1e-3:
                    raw_rate = (vertical - self.last_vertical) / elapsed
                    raw_rate = max(
                        -GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE,
                        min(
                            GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE,
                            raw_rate,
                        ),
                    )
                    alpha = GATE0_PROVED_COLLECTIVE_RATE_FILTER_ALPHA
                    self.filtered_vertical_rate = (
                        (1.0 - alpha) * self.filtered_vertical_rate
                        + alpha * raw_rate
                    )
                self.last_received_monotonic_s = received
                self.last_vertical = vertical
        else:
            self.last_received_monotonic_s = received
            self.last_vertical = vertical
        self.last_token_key = token_key
        thrust = _gate0_proved_vertical_collective(
            vertical,
            self.filtered_vertical_rate,
        )
        return thrust, self.filtered_vertical_rate


@dataclass(frozen=True, slots=True)
class _AcceptedVisualCommand:
    command: AttitudeRateCommand
    yaw_soft_stop_zeroed: bool
    observation_monotonic_ns: int
    wire_start_monotonic_ns: int
    target_roll_rad: float
    target_pitch_rad: float
    next_preview_collective_delta: float


@dataclass(frozen=True, slots=True)
class _SupersededVisualProposal:
    expected_camera_token: CameraFrameToken
    receiver_camera_token: CameraFrameToken
    held_previous_command_s: float
    consecutive_count: int


class _PreviewRequalificationWireSlotUnavailable(RuntimeError):
    """A preview candidate has no wire-start instant inside its wall bound."""

    def __init__(self, checked_perf_counter_ns: int) -> None:
        super().__init__(
            "preview requalification has no bounded wire-start slot"
        )
        self.checked_perf_counter_ns = checked_perf_counter_ns


@dataclass(frozen=True, slots=True)
class _CensoredPassageCoastAuthority:
    gate_index: int
    track_id: str
    anchor_camera_token: CameraFrameToken
    target_roll_rad: float
    target_pitch_rad: float
    thrust: float


@dataclass(frozen=True, slots=True)
class _RetainedCrossingWireProjection:
    target: Any
    observation_monotonic_ns: int
    wire_start_monotonic_ns: int
    observation_to_wire_ns: int
    observation_to_wire_s: float
    projected_log_scale: float
    projected_normalized_x: float
    projected_normalized_y_down: float


@dataclass(frozen=True, slots=True)
class _RetainedCrossingCandidate:
    """One accepted retained-safe command sealed for its exact successor."""

    gate_index: int
    track_id: str
    camera_token: CameraFrameToken
    tracker_frame_sequence: int
    target: VisualTarget
    output: VisualServoOutput
    accepted: _AcceptedVisualCommand
    wire_projection: _RetainedCrossingWireProjection
    retained_crossing_dwell_frames: int
    advance_command_count: int


@dataclass(frozen=True, slots=True)
class _AdjacentScaleJumpCrossingProof:
    """Exact evidence for one authority-reducing near-plane contour jump."""

    predecessor: _RetainedCrossingCandidate
    current_wire_projection: _RetainedCrossingWireProjection
    association: AssociationEvidence
    observation_gap_ns: int
    publication_gap_ns: int
    log_scale_delta: float
    predecessor_projected_log_scale: float
    predecessor_projected_normalized_x: float
    predecessor_projected_normalized_y_down: float


@dataclass(frozen=True, slots=True)
class VisualCourseStageLimits:
    """Code-owned bounds for the generic lifecycle."""

    control_period_s: float = 0.02
    course_hard_duration_s: float = 120.0
    segment_hard_duration_s: float = MAX_VISUAL_SEGMENT_DURATION_S
    passage_hard_duration_s: float = MAX_VISUAL_SEGMENT_DURATION_S
    crossing_status_timeout_s: float = 0.40
    censored_passage_coast_max_duration_s: float = 0.30
    censored_passage_coast_max_fresh_frames: int = 8
    post_credit_fresh_frame_timeout_s: float = 0.20
    max_validation_to_wire_delay_s: float = 0.012
    max_command_rate_rad_s: float = 0.25
    max_yaw_rate_rad_s: float = MAX_VISUAL_YAW_RATE_RAD_S
    max_abs_measured_roll_rad: float = 0.18
    min_measured_pitch_rad: float = -0.35
    max_measured_pitch_rad: float = 0.15
    max_abs_measured_body_rate_rad_s: float = 0.50
    max_segment_yaw_excursion_rad: float = (
        YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD
    )
    max_measured_yaw_rate_rad_s: float = 0.50
    min_thrust: float = MIN_VISUAL_THRUST
    max_thrust: float = MAX_VISUAL_THRUST
    crossing_arm_min_log_scale: float = -0.80
    # The last clean retained observations in attempts 10 and 11 were
    # -0.801151 and -0.827087.  Projection may refine command-wire timing,
    # but may never expand authority beyond this independently replayed floor.
    retained_crossing_projection_min_log_scale: float = -0.83
    # Attempt 17's last centered, stable retained observation reached the
    # command wire in 0.0334435 s.  Keep this exact projection below one
    # 30-Hz camera period plus the reviewed transport allowance.
    retained_crossing_max_observation_to_wire_s: float = 0.035
    crossing_arm_min_log_scale_rate_s: float = 0.0
    crossing_arm_min_advance_commands: int = 3
    max_gate_segments: int = 64

    def __post_init__(self) -> None:
        numeric = (
            self.control_period_s,
            self.course_hard_duration_s,
            self.segment_hard_duration_s,
            self.passage_hard_duration_s,
            self.crossing_status_timeout_s,
            self.censored_passage_coast_max_duration_s,
            self.post_credit_fresh_frame_timeout_s,
            self.max_validation_to_wire_delay_s,
            self.max_command_rate_rad_s,
            self.max_yaw_rate_rad_s,
            self.max_abs_measured_roll_rad,
            self.min_measured_pitch_rad,
            self.max_measured_pitch_rad,
            self.max_abs_measured_body_rate_rad_s,
            self.max_segment_yaw_excursion_rad,
            self.max_measured_yaw_rate_rad_s,
            self.min_thrust,
            self.max_thrust,
            self.crossing_arm_min_log_scale,
            self.retained_crossing_projection_min_log_scale,
            self.retained_crossing_max_observation_to_wire_s,
            self.crossing_arm_min_log_scale_rate_s,
        )
        if not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("visual-course limits must be finite")
        if not math.isclose(
            float(self.control_period_s),
            0.02,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("visual-course control period must be exactly 50 Hz")
        if not 10.0 <= self.course_hard_duration_s <= 180.0:
            raise ValueError("visual-course hard duration is outside bounds")
        if not 0.5 <= self.segment_hard_duration_s <= (
            MAX_VISUAL_SEGMENT_DURATION_S
        ):
            raise ValueError("visual-course segment duration is outside bounds")
        if not 0.40 <= self.passage_hard_duration_s <= (
            self.segment_hard_duration_s
        ):
            raise ValueError("visual-course passage duration is outside bounds")
        if not 0.05 <= self.crossing_status_timeout_s <= 0.40:
            raise ValueError("visual-course crossing wait is outside bounds")
        if not 0.20 <= self.censored_passage_coast_max_duration_s <= 0.30:
            raise ValueError(
                "visual-course censored passage coast is outside bounds"
            )
        if not 0.05 <= self.post_credit_fresh_frame_timeout_s <= 0.20:
            raise ValueError("visual-course fresh-frame wait is outside bounds")
        if not 0.0 < self.max_validation_to_wire_delay_s < (
            self.control_period_s
        ):
            raise ValueError("visual-course wire lease is outside one tick")
        if not 0.0 < self.max_command_rate_rad_s <= 0.25:
            raise ValueError("visual-course command-rate bound is invalid")
        if not 0.0 < self.max_yaw_rate_rad_s <= MAX_VISUAL_YAW_RATE_RAD_S:
            raise ValueError("visual-course yaw-rate bound is invalid")
        if not (
            MAX_VISUAL_TARGET_ROLL_RAD
            <= self.max_abs_measured_roll_rad
            <= 0.18
        ):
            raise ValueError("visual-course measured roll bound is invalid")
        if not (
            -0.35
            <= self.min_measured_pitch_rad
            <= MIN_VISUAL_TARGET_PITCH_RAD
        ):
            raise ValueError(
                "visual-course measured minimum pitch bound is invalid"
            )
        if not (
            MAX_VISUAL_TARGET_PITCH_RAD
            <= self.max_measured_pitch_rad
            <= 0.15
        ):
            raise ValueError(
                "visual-course measured maximum pitch bound is invalid"
            )
        if not (
            0.0 < self.max_abs_measured_body_rate_rad_s <= 0.50
        ):
            raise ValueError(
                "visual-course measured body-rate bound is invalid"
            )
        if not 0.01 <= self.max_segment_yaw_excursion_rad <= (
            YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD
        ):
            raise ValueError("visual-course yaw excursion bound is invalid")
        if not 0.0 < self.max_measured_yaw_rate_rad_s <= 0.50:
            raise ValueError("visual-course measured yaw-rate bound is invalid")
        if (
            self.max_measured_yaw_rate_rad_s
            > self.max_abs_measured_body_rate_rad_s
        ):
            raise ValueError(
                "visual-course yaw-rate bound exceeds all-axis rate bound"
            )
        if not (
            MIN_VISUAL_THRUST
            <= self.min_thrust
            <= self.max_thrust
            <= MAX_VISUAL_THRUST
        ):
            raise ValueError("visual-course thrust bounds are invalid")
        if not -1.50 <= self.crossing_arm_min_log_scale <= -0.20:
            raise ValueError("visual-course crossing scale is outside bounds")
        if not (
            -0.83
            <= self.retained_crossing_projection_min_log_scale
            <= self.crossing_arm_min_log_scale
        ):
            raise ValueError(
                "visual-course retained projection scale is outside bounds"
            )
        if not (
            0.032
            <= self.retained_crossing_max_observation_to_wire_s
            <= 0.035
        ):
            raise ValueError(
                "visual-course retained projection timing is outside bounds"
            )
        if not 0.0 <= self.crossing_arm_min_log_scale_rate_s <= 1.0:
            raise ValueError("visual-course crossing scale rate is invalid")
        if (
            type(self.crossing_arm_min_advance_commands) is not int
            or not 2 <= self.crossing_arm_min_advance_commands <= 8
            or type(self.censored_passage_coast_max_fresh_frames) is not int
            or not 4
            <= self.censored_passage_coast_max_fresh_frames
            <= 8
            or type(self.max_gate_segments) is not int
            or not 1 <= self.max_gate_segments <= 128
        ):
            raise ValueError("visual-course discrete bounds are invalid")


DEFAULT_VISUAL_COURSE_LIMITS = VisualCourseStageLimits()


@dataclass(frozen=True, slots=True, init=False)
class VisualCourseYawProfile:
    """Module-issued identity of the exact tracked three-run yaw profile."""

    schema: str
    profile_id: str
    profile_sha256: str
    source_commit: str
    plan_id: str
    plan_sha256: str
    controller_to_body_sign: int
    controller_to_image_sign: int
    max_abs_yaw_rate_command_rad_s: float
    max_gyro_response_delay_s: float
    max_first_image_observation_delay_s: float
    max_attitude_excursion_rad: float
    max_abs_measured_yaw_rate_rad_s: float
    observed_max_abs_measured_yaw_rate_rad_s: float
    control_hold_horizon_s: float

    def __init__(
        self,
        *,
        issuer: object,
        schema: str,
        profile_id: str,
        profile_sha256: str,
        source_commit: str,
        plan_id: str,
        plan_sha256: str,
        controller_to_body_sign: int,
        controller_to_image_sign: int,
        max_abs_yaw_rate_command_rad_s: float,
        max_gyro_response_delay_s: float,
        max_first_image_observation_delay_s: float,
        max_attitude_excursion_rad: float,
        max_abs_measured_yaw_rate_rad_s: float,
        observed_max_abs_measured_yaw_rate_rad_s: float,
        control_hold_horizon_s: float,
    ) -> None:
        if issuer is not _YAW_PROFILE_ISSUER:
            raise TypeError(
                "visual-course yaw profiles must come from the tracked loader"
            )
        if (
            schema != YAW_CALIBRATION_PROFILE_SCHEMA
            or profile_id != YAW_CALIBRATION_PROFILE_ID
            or profile_sha256 != YAW_CALIBRATION_PROFILE_SHA256
            or source_commit != YAW_CALIBRATION_SOURCE_COMMIT
            or plan_id != YAW_CALIBRATION_PLAN_ID
            or plan_sha256 != YAW_CALIBRATION_PLAN_SHA256
        ):
            raise ValueError("visual-course yaw profile identity is not frozen")
        if (
            controller_to_body_sign != YAW_CONTROLLER_TO_BODY_SIGN
            or controller_to_image_sign != YAW_CONTROLLER_TO_IMAGE_SIGN
            or max_abs_yaw_rate_command_rad_s
            != YAW_MAX_COMMAND_RATE_RAD_S
            or max_gyro_response_delay_s
            != YAW_MAX_GYRO_RESPONSE_DELAY_S
            or max_first_image_observation_delay_s
            != YAW_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S
            or max_attitude_excursion_rad
            != YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD
            or max_abs_measured_yaw_rate_rad_s
            != YAW_MAX_CALIBRATION_MEASURED_RATE_RAD_S
            or observed_max_abs_measured_yaw_rate_rad_s
            != YAW_OBSERVED_MAX_MEASURED_RATE_RAD_S
            or control_hold_horizon_s != YAW_CONTROL_HOLD_HORIZON_S
        ):
            raise ValueError("visual-course yaw authority is not frozen")
        for name, value in (
            ("schema", schema),
            ("profile_id", profile_id),
            ("profile_sha256", profile_sha256),
            ("source_commit", source_commit),
            ("plan_id", plan_id),
            ("plan_sha256", plan_sha256),
            ("controller_to_body_sign", controller_to_body_sign),
            ("controller_to_image_sign", controller_to_image_sign),
            (
                "max_abs_yaw_rate_command_rad_s",
                max_abs_yaw_rate_command_rad_s,
            ),
            ("max_gyro_response_delay_s", max_gyro_response_delay_s),
            (
                "max_first_image_observation_delay_s",
                max_first_image_observation_delay_s,
            ),
            ("max_attitude_excursion_rad", max_attitude_excursion_rad),
            (
                "max_abs_measured_yaw_rate_rad_s",
                max_abs_measured_yaw_rate_rad_s,
            ),
            (
                "observed_max_abs_measured_yaw_rate_rad_s",
                observed_max_abs_measured_yaw_rate_rad_s,
            ),
            ("control_hold_horizon_s", control_hold_horizon_s),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def load_tracked(
        cls,
        path: Any = DEFAULT_YAW_CALIBRATION_PROFILE_PATH,
    ) -> "VisualCourseYawProfile":
        """Load and fully validate the sole tracked multi-run authority."""

        profile = load_yaw_calibration_profile(path)
        evidence = yaw_calibration_profile_evidence(profile)
        authority = evidence["authority"]
        return cls(
            issuer=_YAW_PROFILE_ISSUER,
            schema=YAW_CALIBRATION_PROFILE_SCHEMA,
            profile_id=evidence["profile_id"],
            profile_sha256=evidence["sha256"],
            source_commit=evidence["source_commit"],
            plan_id=evidence["plan_id"],
            plan_sha256=evidence["plan_sha256"],
            controller_to_body_sign=authority["controller_to_body_sign"],
            controller_to_image_sign=authority[
                "controller_to_image_sign"
            ],
            max_abs_yaw_rate_command_rad_s=authority[
                "max_abs_yaw_rate_command_rad_s"
            ],
            max_gyro_response_delay_s=authority[
                "max_gyro_response_delay_s"
            ],
            max_first_image_observation_delay_s=authority[
                "max_first_image_observation_delay_s"
            ],
            max_attitude_excursion_rad=authority[
                "max_attitude_excursion_rad"
            ],
            max_abs_measured_yaw_rate_rad_s=authority[
                "max_abs_measured_yaw_rate_rad_s"
            ],
            observed_max_abs_measured_yaw_rate_rad_s=(
                profile["observed_ranges"][
                    "max_abs_measured_yaw_rate_rad_s"
                ]["max"]
            ),
            control_hold_horizon_s=authority["control_hold_horizon_s"],
        )

    def to_evidence(self) -> Dict[str, Any]:
        """Match the strict manifest identity emitted by the profile module."""

        return {
            "profile_id": self.profile_id,
            "sha256": self.profile_sha256,
            "source_commit": self.source_commit,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "authority": {
                "controller_to_body_sign": self.controller_to_body_sign,
                "controller_to_image_sign": self.controller_to_image_sign,
                "max_abs_yaw_rate_command_rad_s": (
                    self.max_abs_yaw_rate_command_rad_s
                ),
                "max_gyro_response_delay_s": (
                    self.max_gyro_response_delay_s
                ),
                "max_first_image_observation_delay_s": (
                    self.max_first_image_observation_delay_s
                ),
                "max_attitude_excursion_rad": (
                    self.max_attitude_excursion_rad
                ),
                "max_abs_measured_yaw_rate_rad_s": (
                    self.max_abs_measured_yaw_rate_rad_s
                ),
                "control_hold_horizon_s": self.control_hold_horizon_s,
            },
        }


@dataclass(frozen=True, slots=True)
class VisualCourseStageRuntime:
    """Runner-owned operations and already-reviewed calibration authority."""

    safety_abort_type: type[BaseException]
    cancelled_error_type: type[BaseException]
    monotonic: Callable[[], float]
    perf_counter_ns: Callable[[], int]
    sleep: Callable[[float], Awaitable[Any]]
    next_control_deadline: Callable[..., float]
    attitude_rate_command: Callable[..., AttitudeRateCommand]
    limit_command_rates: Callable[..., AttitudeRateCommand]
    validate_command: Callable[[AttitudeRateCommand], None]
    yaw_profile: Optional[VisualCourseYawProfile]
    expected_yaw_profile_sha256: Optional[str]
    transition_recovery_admission: Callable[..., Any] = (
        require_transition_recovery_admission
    )
    recovery_continuation_admission: Callable[..., Any] = (
        require_recovery_continuation
    )
    limits: VisualCourseStageLimits = DEFAULT_VISUAL_COURSE_LIMITS
    servo_factory: Callable[..., Any] = RollingVisualApproachServo

    def __post_init__(self) -> None:
        if not isinstance(self.safety_abort_type, type) or not issubclass(
            self.safety_abort_type, BaseException
        ):
            raise TypeError("visual-course safety abort type is invalid")
        if not isinstance(self.cancelled_error_type, type) or not issubclass(
            self.cancelled_error_type, BaseException
        ):
            raise TypeError("visual-course cancellation type is invalid")
        if type(self.limits) is not VisualCourseStageLimits:
            raise TypeError("visual-course limits must be exact")
        if self.yaw_profile is not None and type(
            self.yaw_profile
        ) is not VisualCourseYawProfile:
            raise TypeError("visual-course yaw profile must be exact or None")
        if self.yaw_profile is None:
            if self.expected_yaw_profile_sha256 is not None:
                raise ValueError(
                    "visual-course profile identity exists without authority"
                )
        elif (
            self.expected_yaw_profile_sha256
            != self.yaw_profile.profile_sha256
            or self.expected_yaw_profile_sha256
            != YAW_CALIBRATION_PROFILE_SHA256
        ):
            raise ValueError(
                "visual-course runtime/manifest yaw profile identity differs"
            )
        if self.yaw_profile is not None and (
            self.limits.max_yaw_rate_rad_s
            > self.yaw_profile.max_abs_yaw_rate_command_rad_s
            or self.limits.max_segment_yaw_excursion_rad
            > self.yaw_profile.max_attitude_excursion_rad
            or self.limits.max_measured_yaw_rate_rad_s
            > self.yaw_profile.max_abs_measured_yaw_rate_rad_s
        ):
            raise ValueError(
                "visual-course limits exceed tracked yaw authority"
            )
        for name in (
            "monotonic",
            "perf_counter_ns",
            "sleep",
            "next_control_deadline",
            "attitude_rate_command",
            "limit_command_rates",
            "validate_command",
            "transition_recovery_admission",
            "recovery_continuation_admission",
            "servo_factory",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"visual-course runtime {name} is not callable")


class VisualCourseStageHost(Protocol):
    """Narrow runner surface used by the generic coordinator."""

    _visual_tracking_enabled: bool
    _visual_course_summary: Optional[Dict[str, Any]]
    _last_flight_command_started_ns: Optional[int]
    visual_gate_graph: Any
    visual_tracker: Any
    visual_config: Any
    estimate: Any
    recorder: Any

    def _sample(self) -> None: ...

    def _watchdog(self, **kwargs: Any) -> None: ...

    async def _wait_for_next_flight_command_slot(self) -> float: ...

    async def _send_flight_command(
        self,
        command: AttitudeRateCommand,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]: ...

    def _assert_visual_receiver_token_current(
        self,
        expected_token: CameraFrameToken,
    ) -> CameraFrameToken: ...

    def _visual_race_status_ref(self) -> AuthoritativeRaceStatusRef: ...

    def _visual_camera_token_at_race_credit(
        self,
        race_status: AuthoritativeRaceStatusRef,
    ) -> CameraFrameToken: ...

    def _confirm_visual_transition(
        self,
        *,
        from_gate_index: int,
        to_gate_index: int,
        race_status: AuthoritativeRaceStatusRef,
        promoted_track_id: Optional[str] = None,
    ) -> Any: ...

    def _record_tick(
        self,
        stage: str,
        elapsed_s: float,
        command: Optional[AttitudeRateCommand],
    ) -> None: ...


def _wrapped_delta(value: float, reference: float) -> float:
    return math.atan2(
        math.sin(float(value) - float(reference)),
        math.cos(float(value) - float(reference)),
    )


def _attitude_state(
    host: VisualCourseStageHost,
    abort_type: type[BaseException],
) -> tuple[float, float, float, tuple[float, float, float]]:
    estimate = host.estimate
    try:
        roll, pitch, yaw = estimate.orientation.to_euler()
        rates = tuple(float(value) for value in estimate.body_rates)
    except (AttributeError, TypeError, ValueError) as exc:
        raise abort_type("visual-course attitude state is unavailable") from exc
    values = (float(roll), float(pitch), float(yaw), *rates)
    if len(rates) != 3 or not all(math.isfinite(value) for value in values):
        raise abort_type("visual-course attitude state is non-finite")
    return float(roll), float(pitch), float(yaw), rates


def _assert_course_attitude_state(
    host: VisualCourseStageHost,
    *,
    yaw_reference_rad: float,
    limits: VisualCourseStageLimits,
    yaw_profile: Optional[VisualCourseYawProfile],
    abort_type: type[BaseException],
    phase: str,
) -> tuple[float, tuple[float, float, float], float]:
    """Enforce the measured course envelope independently of camera cadence."""

    roll, pitch, yaw, rates = _attitude_state(host, abort_type)
    excursion = _wrapped_delta(yaw, yaw_reference_rad)
    if (
        abs(roll) > limits.max_abs_measured_roll_rad
        or pitch < limits.min_measured_pitch_rad
        or pitch > limits.max_measured_pitch_rad
        or max(abs(value) for value in rates)
        > limits.max_abs_measured_body_rate_rad_s
    ):
        raise abort_type(
            "visual-course measured attitude/body-rate envelope was exceeded "
            f"during {phase}"
        )
    if (
        abs(excursion) > limits.max_segment_yaw_excursion_rad
        or abs(rates[2]) > limits.max_measured_yaw_rate_rad_s
    ):
        raise abort_type(
            "visual-course segment yaw envelope was exceeded "
            f"during {phase}"
        )
    hold_horizon_s = (
        YAW_CONTROL_HOLD_HORIZON_S
        if yaw_profile is None
        else yaw_profile.control_hold_horizon_s
    )
    cos_pitch = math.cos(pitch)
    if cos_pitch <= 0.0:
        raise abort_type(
            "visual-course Euler yaw rate is singular "
            f"during {phase}"
        )
    euler_yaw_rate = (
        rates[1] * math.sin(roll) + rates[2] * math.cos(roll)
    ) / cos_pitch
    if not math.isfinite(euler_yaw_rate):
        raise abort_type(
            "visual-course Euler yaw rate is non-finite "
            f"during {phase}"
        )
    projected_excursion = excursion + euler_yaw_rate * hold_horizon_s
    if abs(projected_excursion) > limits.max_segment_yaw_excursion_rad:
        raise abort_type(
            "visual-course measured yaw momentum projects outside its "
            f"envelope during {phase}"
        )
    return excursion, rates, euler_yaw_rate


def _limit_calibrated_yaw_request(
    requested_yaw_rate_rad_s: float,
    *,
    excursion_rad: float,
    measured_euler_yaw_rate_rad_s: float,
    limits: VisualCourseStageLimits,
    profile: VisualCourseYawProfile,
    abort_type: type[BaseException],
) -> float:
    """Admit yaw only while a worst observed response fits the hard bound.

    The soft boundary retains one full observed peak-rate response over the
    frozen hold horizon.  Current measured momentum is projected through the
    accepted response delay before allocating any of the remaining soft
    headroom to a new command.  Every nonzero request is charged the full
    observed response; no uncalibrated linear response scaling is assumed.
    """

    values = (
        requested_yaw_rate_rad_s,
        excursion_rad,
        measured_euler_yaw_rate_rad_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise abort_type("visual-course yaw command admission is non-finite")
    requested = float(requested_yaw_rate_rad_s)
    if abs(requested) > limits.max_yaw_rate_rad_s + 1e-12:
        raise abort_type("visual-course servo exceeded yaw-rate authority")
    if abs(requested) > profile.max_abs_yaw_rate_command_rad_s + 1e-12:
        raise abort_type(
            "visual-course yaw request exceeds calibrated rate"
        )
    if requested == 0.0:
        return 0.0

    max_abs_pitch = max(
        abs(limits.min_measured_pitch_rad),
        abs(limits.max_measured_pitch_rad),
    )
    # The calibration measured body-r.  Convert its worst observed response
    # into Euler-yaw authority over the full admitted course pitch envelope.
    # Current q/r momentum is handled independently by the exact Euler
    # kinematics in ``_assert_course_attitude_state`` and below.
    observed_peak_rate = (
        profile.observed_max_abs_measured_yaw_rate_rad_s
        / math.cos(max_abs_pitch)
    )
    response_reserve_rad = (
        observed_peak_rate * profile.control_hold_horizon_s
    )
    hard_boundary_rad = min(
        limits.max_segment_yaw_excursion_rad,
        profile.max_attitude_excursion_rad,
    )
    soft_boundary_rad = hard_boundary_rad - response_reserve_rad
    if (
        not math.isfinite(soft_boundary_rad)
        or soft_boundary_rad <= 0.0
        or response_reserve_rad <= 0.0
    ):
        raise abort_type(
            "visual-course calibrated yaw reserve is structurally invalid"
        )

    body_request = requested * profile.controller_to_body_sign
    direction = math.copysign(1.0, body_request)
    excursion = float(excursion_rad)
    measured_rate = float(measured_euler_yaw_rate_rad_s)
    delayed_excursion = (
        excursion + measured_rate * profile.max_gyro_response_delay_s
    )
    directional_start = max(
        direction * excursion,
        direction * delayed_excursion,
    )
    remaining_soft_headroom = soft_boundary_rad - directional_start
    if remaining_soft_headroom <= 0.0:
        # Exact zero removes outward authority while measured excursion,
        # measured rate, and the longer hard-envelope projection continue to
        # be checked independently on every control tick and immediately
        # before every send.  Do not scale to an uncalibrated nonzero request.
        return 0.0

    return requested


def _retained_crossing_observation_usable(
    target: Any,
    output: Any,
    *,
    tuning: Any,
    limits: VisualCourseStageLimits,
    yaw_soft_stop_zeroed: bool = False,
) -> bool:
    """Admit one close-alignment sample to non-advance crossing dwell."""

    return bool(
        type(yaw_soft_stop_zeroed) is bool
        and not target.clipped
        and not target.center_censored
        and not target.horizontal_geometry_censored
        and not target.vertical_geometry_censored
        and not target.ambiguous
        and abs(float(target.normalized_x)) <= tuning.horizontal_corridor
        and abs(float(target.normalized_y_down)) <= tuning.vertical_corridor
        and abs(float(target.normalized_x_rate_s))
        <= tuning.stable_rate_norm_s
        and abs(float(target.normalized_y_rate_down_s))
        <= tuning.stable_rate_norm_s
        and limits.crossing_arm_min_log_scale_rate_s
        <= float(target.log_scale_rate_s)
        < tuning.brake_scale_rate_s
        and (
            (
                not output.advance_enabled
                and output.brake_reason == "aligning"
            )
            or (
                yaw_soft_stop_zeroed
                and output.advance_enabled
                and output.brake_reason is None
            )
        )
        and not output.yaw_envelope_limited
    )


def _retained_crossing_wire_projection(
    target: Any,
    *,
    observation_monotonic_ns: int,
    wire_start_monotonic_ns: int,
    tuning: Any,
    limits: VisualCourseStageLimits,
    abort_type: type[BaseException],
) -> Optional[_RetainedCrossingWireProjection]:
    """Project retained visual scale and center to its accepted wire instant."""

    if type(wire_start_monotonic_ns) is not int or wire_start_monotonic_ns < 0:
        raise abort_type(
            "visual-course retained crossing lacks exact wire timing"
        )
    if (
        type(observation_monotonic_ns) is not int
        or observation_monotonic_ns < 0
    ):
        raise abort_type(
            "visual-course retained crossing lacks exact observation timing"
        )
    received_s = float(target.received_monotonic_s)
    raw_log_scale = float(target.log_scale)
    log_scale_rate_s = float(target.log_scale_rate_s)
    normalized_x = float(target.normalized_x)
    normalized_y_down = float(target.normalized_y_down)
    normalized_x_rate_s = float(target.normalized_x_rate_s)
    normalized_y_rate_down_s = float(
        target.normalized_y_rate_down_s
    )
    if not all(
        math.isfinite(value)
        for value in (
            received_s,
            raw_log_scale,
            log_scale_rate_s,
            normalized_x,
            normalized_y_down,
            normalized_x_rate_s,
            normalized_y_rate_down_s,
        )
    ) or received_s < 0.0:
        raise abort_type(
            "visual-course retained crossing timing is invalid"
        )
    if round(received_s * 1_000_000_000) != observation_monotonic_ns:
        raise abort_type(
            "visual-course retained crossing observation timing differs "
            "from its target"
        )
    observation_to_wire_ns = (
        wire_start_monotonic_ns - observation_monotonic_ns
    )
    if observation_to_wire_ns < 0:
        raise abort_type(
            "visual-course retained crossing wire predates observation"
        )
    max_projection_ns = round(
        limits.retained_crossing_max_observation_to_wire_s
        * 1_000_000_000
    )
    if (
        observation_to_wire_ns > max_projection_ns
        or log_scale_rate_s <= 0.0
    ):
        return None
    observation_to_wire_s = observation_to_wire_ns / 1_000_000_000.0
    projected_log_scale = (
        raw_log_scale + log_scale_rate_s * observation_to_wire_s
    )
    projected_normalized_x = (
        normalized_x + normalized_x_rate_s * observation_to_wire_s
    )
    projected_normalized_y_down = (
        normalized_y_down
        + normalized_y_rate_down_s * observation_to_wire_s
    )
    if not all(
        math.isfinite(value)
        for value in (
            projected_log_scale,
            projected_normalized_x,
            projected_normalized_y_down,
        )
    ):
        raise abort_type(
            "visual-course retained crossing projection is invalid"
        )
    if (
        abs(projected_normalized_x) > tuning.horizontal_corridor
        or abs(projected_normalized_y_down) > tuning.vertical_corridor
    ):
        return None
    return _RetainedCrossingWireProjection(
        target=target,
        observation_monotonic_ns=observation_monotonic_ns,
        wire_start_monotonic_ns=wire_start_monotonic_ns,
        observation_to_wire_ns=observation_to_wire_ns,
        observation_to_wire_s=observation_to_wire_s,
        projected_log_scale=projected_log_scale,
        projected_normalized_x=projected_normalized_x,
        projected_normalized_y_down=projected_normalized_y_down,
    )


def _retained_crossing_projection_matches_target(
    target: Any,
    projection: Any,
    *,
    tuning: Any,
    limits: VisualCourseStageLimits,
) -> bool:
    """Recompute a private projection before it may prove crossing."""

    if (
        type(projection) is not _RetainedCrossingWireProjection
        or projection.target is not target
    ):
        return False
    observation_ns = round(
        float(target.received_monotonic_s) * 1_000_000_000
    )
    horizon_ns = (
        projection.wire_start_monotonic_ns - observation_ns
    )
    max_horizon_ns = round(
        limits.retained_crossing_max_observation_to_wire_s
        * 1_000_000_000
    )
    if (
        projection.observation_monotonic_ns != observation_ns
        or horizon_ns != projection.observation_to_wire_ns
        or not 0 <= horizon_ns <= max_horizon_ns
        or float(target.log_scale_rate_s) <= 0.0
    ):
        return False
    horizon_s = horizon_ns / 1_000_000_000.0
    expected_log_scale = (
        float(target.log_scale)
        + float(target.log_scale_rate_s) * horizon_s
    )
    expected_x = (
        float(target.normalized_x)
        + float(target.normalized_x_rate_s) * horizon_s
    )
    expected_y = (
        float(target.normalized_y_down)
        + float(target.normalized_y_rate_down_s) * horizon_s
    )
    return bool(
        math.isclose(
            projection.observation_to_wire_s,
            horizon_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            projection.projected_log_scale,
            expected_log_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            projection.projected_normalized_x,
            expected_x,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            projection.projected_normalized_y_down,
            expected_y,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and abs(expected_x) <= tuning.horizontal_corridor
        and abs(expected_y) <= tuning.vertical_corridor
    )


def _current_target_observation_monotonic_ns(
    snapshot: Any,
    target: Any,
    *,
    abort_type: type[BaseException],
) -> int:
    """Bind the proposal to the exact latest receiver observation QPC."""

    token = getattr(snapshot, "latest_camera_token", None)
    track = getattr(snapshot, "current_track", None)
    history = getattr(track, "history", None)
    if (
        type(token) is not CameraFrameToken
        or not _servo_token_matches_camera(target.frame_token, token)
        or type(history) is not tuple
        or not history
    ):
        raise abort_type(
            "visual-course target lacks exact observation provenance"
        )
    latest_sample = history[-1]
    observation_monotonic_ns = getattr(
        latest_sample,
        "observation_monotonic_ns",
        None,
    )
    if (
        getattr(latest_sample, "token", None) != token
        or type(observation_monotonic_ns) is not int
        or observation_monotonic_ns < 0
        or round(
            float(target.received_monotonic_s) * 1_000_000_000
        )
        != observation_monotonic_ns
    ):
        raise abort_type(
            "visual-course target observation provenance is inconsistent"
        )
    return observation_monotonic_ns


def _crossing_anchor_basis(
    target: Any,
    output: Any,
    *,
    passage_admission: Optional[VisualApproachPassageAdmission],
    current_gate_index: int,
    current_track_id: str,
    advance_command_count: int,
    retained_crossing_dwell_frames: int,
    tuning: Any,
    limits: VisualCourseStageLimits,
    retained_wire_projection: Optional[
        _RetainedCrossingWireProjection
    ] = None,
    yaw_soft_stop_zeroed: bool = False,
) -> Optional[str]:
    """Select an exact-frame crossing proof without adding motion authority."""

    if type(yaw_soft_stop_zeroed) is not bool:
        return None
    if (
        advance_command_count
        < limits.crossing_arm_min_advance_commands
        or float(target.log_scale_rate_s)
        < limits.crossing_arm_min_log_scale_rate_s
        or target.clipped
        or target.center_censored
        or target.ambiguous
    ):
        return None
    if (
        not yaw_soft_stop_zeroed
        and float(target.log_scale) >= limits.crossing_arm_min_log_scale
        and output.corridor_frames >= tuning.required_corridor_frames
        and output.advance_enabled
    ):
        return CURRENT_ADVANCE_CROSSING_BASIS
    if (
        type(passage_admission) is VisualApproachPassageAdmission
        and passage_admission.current_gate_index == current_gate_index
        and passage_admission.current_target.track_id == current_track_id
        and target.track_id == current_track_id
        and retained_crossing_dwell_frames
        >= tuning.required_corridor_frames
        and _retained_crossing_observation_usable(
            target,
            output,
            tuning=tuning,
            limits=limits,
            yaw_soft_stop_zeroed=yaw_soft_stop_zeroed,
        )
    ):
        if float(target.log_scale) >= limits.crossing_arm_min_log_scale:
            return RETAINED_ADVANCE_CROSSING_BASIS
        if (
            float(target.log_scale)
            >= limits.retained_crossing_projection_min_log_scale
            and _retained_crossing_projection_matches_target(
                target,
                retained_wire_projection,
                tuning=tuning,
                limits=limits,
            )
            and retained_wire_projection.projected_log_scale
            >= limits.crossing_arm_min_log_scale
        ):
            return RETAINED_ADVANCE_WIRE_PROJECTED_CROSSING_BASIS
    return None


def _race_relation(
    current: AuthoritativeRaceStatusRef,
    previous: AuthoritativeRaceStatusRef,
    abort_type: type[BaseException],
) -> int:
    """Return -1/0/+1 sequence ordering inside one proved live epoch."""

    if (
        type(current) is not AuthoritativeRaceStatusRef
        or type(previous) is not AuthoritativeRaceStatusRef
        or current.provenance_basis is not RaceStatusProvenanceBasis.LIVE_INGRESS
        or previous.provenance_basis is not RaceStatusProvenanceBasis.LIVE_INGRESS
        or current.session_id != previous.session_id
        or current.reset_epoch != previous.reset_epoch
        or current.race_generation != previous.race_generation
        or current.host_clock_id != previous.host_clock_id
    ):
        raise abort_type("visual-course race status crossed its proved epoch")
    assert current.race_status_sequence is not None
    assert previous.race_status_sequence is not None
    if current.race_status_sequence < previous.race_status_sequence:
        return -1
    if current.race_status_sequence > previous.race_status_sequence:
        return 1
    if current != previous:
        raise abort_type("visual-course repeated race sequence changed payload")
    return 0


def _token_strictly_newer(
    current: CameraFrameToken,
    previous: CameraFrameToken,
) -> bool:
    return bool(
        type(current) is CameraFrameToken
        and type(previous) is CameraFrameToken
        and current.stream_id is not None
        and current.stream_id == previous.stream_id
        and current.generation == previous.generation
        and current.publication_sequence is not None
        and previous.publication_sequence is not None
        and current.publication_sequence > previous.publication_sequence
    )


def _tokens_are_adjacent_camera_publications(
    current: CameraFrameToken,
    previous: CameraFrameToken,
) -> bool:
    return bool(
        type(current) is CameraFrameToken
        and type(previous) is CameraFrameToken
        and type(current.stream_id) is str
        and current.stream_id
        and current.stream_id == previous.stream_id
        and type(current.generation) is int
        and current.generation == previous.generation
        and type(current.frame_id) is int
        and type(previous.frame_id) is int
        and current.frame_id == previous.frame_id + 1
        and type(current.publication_sequence) is int
        and type(previous.publication_sequence) is int
        and current.publication_sequence
        == previous.publication_sequence + 1
    )


def _adjacent_scale_jump_crossing_proof(
    snapshot: Any,
    target: Any,
    output: Any,
    accepted: Any,
    *,
    predecessor: Optional[_RetainedCrossingCandidate],
    passage_admission: Optional[VisualApproachPassageAdmission],
    current_gate_index: int,
    current_track_id: str,
    advance_command_count: int,
    next_preview_retired: bool,
    tuning: Any,
    limits: VisualCourseStageLimits,
    abort_type: type[BaseException],
) -> Optional[_AdjacentScaleJumpCrossingProof]:
    """Prove one exact adjacent contour jump without extending authority.

    A retained-safe accepted predecessor must have completed the ordinary
    corridor dwell.  Its already-admitted image rates must project across the
    unchanged crossing plane at the successor observation.  The successor
    itself must then be an accepted, current-only scale-rate brake with clean
    raw and command-wire geometry.  The proof is intentionally consumed by
    this exact successor and never carries across a skipped publication.
    """

    if (
        type(predecessor) is not _RetainedCrossingCandidate
        or type(target) is not VisualTarget
        or type(output) is not VisualServoOutput
        or type(accepted) is not _AcceptedVisualCommand
        or type(passage_admission) is not VisualApproachPassageAdmission
        or type(next_preview_retired) is not bool
        or next_preview_retired is not True
        or type(current_gate_index) is not int
        or type(current_track_id) is not str
        or not current_track_id
        or type(advance_command_count) is not int
    ):
        return None

    token = getattr(snapshot, "latest_camera_token", None)
    track = getattr(snapshot, "current_track", None)
    history = getattr(track, "history", None)
    predecessor_target = predecessor.target
    predecessor_output = predecessor.output
    predecessor_accepted = predecessor.accepted
    predecessor_projection = predecessor.wire_projection
    predecessor_command = predecessor_accepted.command
    command = accepted.command
    predecessor_command_values = (
        getattr(predecessor_command, "roll_rate", math.nan),
        getattr(predecessor_command, "pitch_rate", math.nan),
        getattr(predecessor_command, "yaw_rate", math.nan),
        getattr(predecessor_command, "thrust", math.nan),
        predecessor_accepted.target_roll_rad,
        predecessor_accepted.target_pitch_rad,
        predecessor_accepted.next_preview_collective_delta,
    )
    command_values = (
        getattr(command, "roll_rate", math.nan),
        getattr(command, "pitch_rate", math.nan),
        getattr(command, "yaw_rate", math.nan),
        getattr(command, "thrust", math.nan),
        accepted.target_roll_rad,
        accepted.target_pitch_rad,
        accepted.next_preview_collective_delta,
    )
    if (
        predecessor.gate_index != current_gate_index
        or predecessor.track_id != current_track_id
        or predecessor_target.track_id != current_track_id
        or target.track_id != current_track_id
        or type(predecessor_output) is not VisualServoOutput
        or passage_admission.current_gate_index != current_gate_index
        or passage_admission.current_target.track_id != current_track_id
        or predecessor.advance_command_count
        < limits.crossing_arm_min_advance_commands
        or advance_command_count != predecessor.advance_command_count
        or predecessor.retained_crossing_dwell_frames
        < tuning.required_corridor_frames
        or type(predecessor.tracker_frame_sequence) is not int
        or predecessor.tracker_frame_sequence < 0
        or type(predecessor.camera_token) is not CameraFrameToken
        or not _servo_token_matches_camera(
            predecessor_target.frame_token,
            predecessor.camera_token,
        )
        or type(predecessor_accepted) is not _AcceptedVisualCommand
        or type(predecessor_command) is not AttitudeRateCommand
        or not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in predecessor_command_values
        )
        or not 0.0
        <= float(predecessor_accepted.next_preview_collective_delta)
        <= GATE0_PROVED_NEXT_PREVIEW_MAX_THRUST_DELTA
        or not limits.min_thrust
        <= (
            float(predecessor_command.thrust)
            - float(predecessor_accepted.next_preview_collective_delta)
        )
        <= limits.max_thrust
        or max(
            abs(float(predecessor_command.roll_rate)),
            abs(float(predecessor_command.pitch_rate)),
        )
        > limits.max_command_rate_rad_s + 1e-12
        or float(predecessor_command.yaw_rate) != 0.0
        or not limits.min_thrust
        <= float(predecessor_command.thrust)
        <= limits.max_thrust
        or abs(float(predecessor_accepted.target_roll_rad))
        > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
        or float(predecessor_accepted.target_pitch_rad)
        < MIN_VISUAL_TARGET_PITCH_RAD - 1e-12
        or float(predecessor_accepted.target_pitch_rad)
        > MAX_VISUAL_TARGET_PITCH_RAD + 1e-12
        or not math.isclose(
            float(predecessor_accepted.target_roll_rad),
            float(predecessor_output.target_roll_rad),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(predecessor_accepted.target_pitch_rad),
            float(predecessor_output.target_pitch_rad),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or predecessor_accepted.yaw_soft_stop_zeroed is not True
        or predecessor_output.advance_enabled is not True
        or predecessor_output.brake_reason is not None
        or float(predecessor_output.yaw_rate_rad_s) == 0.0
        or predecessor_output.passage_preview_retired is not False
        or type(predecessor_projection)
        is not _RetainedCrossingWireProjection
        or predecessor_projection.wire_start_monotonic_ns
        != predecessor_accepted.wire_start_monotonic_ns
        or predecessor_projection.observation_monotonic_ns
        != predecessor_accepted.observation_monotonic_ns
        or not _retained_crossing_projection_matches_target(
            predecessor_target,
            predecessor_projection,
            tuning=tuning,
            limits=limits,
        )
        or predecessor_projection.projected_log_scale
        >= limits.crossing_arm_min_log_scale
        or not _retained_crossing_observation_usable(
            predecessor_target,
            predecessor_output,
            tuning=tuning,
            limits=limits,
            yaw_soft_stop_zeroed=(
                predecessor_accepted.yaw_soft_stop_zeroed
            ),
        )
        or not (
            ADJACENT_SCALE_JUMP_MIN_PREDECESSOR_LOG_SCALE
            <= float(predecessor_target.log_scale)
            < limits.crossing_arm_min_log_scale
        )
        or type(token) is not CameraFrameToken
        or not _tokens_are_adjacent_camera_publications(
            token,
            predecessor.camera_token,
        )
        or not _servo_token_matches_camera(target.frame_token, token)
        or getattr(snapshot, "current_gate_index", None)
        != current_gate_index
        or getattr(snapshot, "current_track_id", None)
        != current_track_id
        or getattr(snapshot, "authority_usable", False) is not True
        or getattr(snapshot, "race_finished", False) is not False
        or track is None
        or getattr(track, "track_id", None) != current_track_id
        or getattr(track, "latest_token", None) != token
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "visible", False) is not True
        or getattr(track, "missed_frame_count", 1) != 0
        or getattr(track, "ambiguous", True) is not False
        or getattr(track, "clipping", None) is not FrameEdge.NONE
        or getattr(track, "center_censored", True) is not False
        or type(history) is not tuple
        or len(history) < 2
        or type(getattr(track, "consecutive_frame_count", None)) is not int
        or track.consecutive_frame_count < 2
        or type(getattr(snapshot, "tracker_frame_sequence", None)) is not int
        or type(command) is not AttitudeRateCommand
        or not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in command_values
        )
        or max(abs(float(command.roll_rate)), abs(float(command.pitch_rate)))
        > limits.max_command_rate_rad_s + 1e-12
        or float(command.yaw_rate) != 0.0
        or not limits.min_thrust
        <= float(command.thrust)
        <= limits.max_thrust
        or abs(float(accepted.target_roll_rad))
        > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
        or float(accepted.target_pitch_rad)
        < MIN_VISUAL_TARGET_PITCH_RAD - 1e-12
        or float(accepted.target_pitch_rad)
        > MAX_VISUAL_TARGET_PITCH_RAD + 1e-12
        or not math.isclose(
            float(accepted.target_roll_rad),
            float(output.target_roll_rad),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(accepted.target_pitch_rad),
            float(output.target_pitch_rad),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or accepted.yaw_soft_stop_zeroed is not True
        or float(output.yaw_rate_rad_s) == 0.0
        or output.yaw_envelope_limited is not False
        or output.advance_enabled is not False
        or output.brake_reason != "scale_rate"
        or output.passage_preview_retired is not True
        or float(output.next_gate_blend) != 0.0
        or float(accepted.next_preview_collective_delta) != 0.0
        or target.clipped
        or target.center_censored
        or target.horizontal_geometry_censored
        or target.vertical_geometry_censored
        or target.ambiguous
        or abs(float(target.normalized_x)) > tuning.horizontal_corridor
        or abs(float(target.normalized_y_down)) > tuning.vertical_corridor
        or abs(float(target.normalized_y_rate_down_s))
        > tuning.stable_rate_norm_s
        or abs(float(target.normalized_x_rate_s))
        > ADJACENT_SCALE_JUMP_MAX_ABS_CURRENT_HORIZONTAL_RATE_S
        or float(target.log_scale_rate_s) < tuning.brake_scale_rate_s
        or float(target.log_scale_rate_s)
        > ADJACENT_SCALE_JUMP_MAX_CURRENT_LOG_SCALE_RATE_S
        or float(target.log_scale) < limits.crossing_arm_min_log_scale
    ):
        return None

    previous = history[-2]
    current = history[-1]
    if (
        type(previous) is not VisualTrackSample
        or type(current) is not VisualTrackSample
        or previous.token != predecessor.camera_token
        or current.token != token
        or previous.provenance_basis
        is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        or current.provenance_basis
        is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        or previous.tracker_frame_sequence
        != predecessor.tracker_frame_sequence
        or current.tracker_frame_sequence
        != predecessor.tracker_frame_sequence + 1
        or snapshot.tracker_frame_sequence
        != current.tracker_frame_sequence
        or previous.clipping is not FrameEdge.NONE
        or current.clipping is not FrameEdge.NONE
        or previous.center_censored
        or current.center_censored
        or type(previous.observation_monotonic_ns) is not int
        or type(current.observation_monotonic_ns) is not int
        or type(previous.publication_monotonic_ns) is not int
        or type(current.publication_monotonic_ns) is not int
        or not (
            previous.observation_monotonic_ns
            < previous.publication_monotonic_ns
            <= predecessor_accepted.wire_start_monotonic_ns
        )
        or not (
            current.observation_monotonic_ns
            < current.publication_monotonic_ns
            <= accepted.wire_start_monotonic_ns
        )
        or predecessor_accepted.observation_monotonic_ns
        != previous.observation_monotonic_ns
        or accepted.observation_monotonic_ns
        != current.observation_monotonic_ns
        or round(
            float(predecessor_target.received_monotonic_s)
            * 1_000_000_000
        )
        != previous.observation_monotonic_ns
        or round(float(target.received_monotonic_s) * 1_000_000_000)
        != current.observation_monotonic_ns
        or not all(
            type(center) is tuple
            and len(center) == 2
            and all(
                type(value) in {int, float}
                and math.isfinite(float(value))
                for value in center
            )
            for center in (previous.center_norm, current.center_norm)
        )
        or not math.isclose(
            float(previous.center_norm[0]),
            float(predecessor_target.normalized_x),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(previous.center_norm[1]),
            float(predecessor_target.normalized_y_down),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            math.log(float(previous.apparent_scale)),
            float(predecessor_target.log_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(current.center_norm[0]),
            float(target.normalized_x),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(current.center_norm[1]),
            float(target.normalized_y_down),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            math.log(float(current.apparent_scale)),
            float(target.log_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(getattr(track, "apparent_scale", math.nan)),
            float(current.apparent_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(getattr(track, "center_norm", (math.nan, math.nan))[0]),
            float(current.center_norm[0]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(getattr(track, "center_norm", (math.nan, math.nan))[1]),
            float(current.center_norm[1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(getattr(track, "center_velocity_norm_s", (math.nan,))[0]),
            float(target.normalized_x_rate_s),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(
                getattr(
                    track,
                    "center_velocity_norm_s",
                    (math.nan, math.nan),
                )[1]
            ),
            float(target.normalized_y_rate_down_s),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(getattr(track, "log_scale_rate_s", math.nan)),
            float(target.log_scale_rate_s),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or getattr(track, "consecutive_frame_count", None)
        != target.consecutive_frames
    ):
        return None

    predecessor_association = previous.accepted_association
    association = current.accepted_association
    if (
        type(predecessor_association) is not AssociationEvidence
        or predecessor_association.track_id != current_track_id
        or predecessor_association.current_token != previous.token
        or predecessor_association.detection_source_index
        != previous.source_index
        or predecessor_association.missed_frame_count_before_association
        != 0
        or predecessor_association.ambiguous
        or predecessor_association.track_ambiguous_before_association
        or float(predecessor_association.confidence)
        < ADJACENT_SCALE_JUMP_MIN_ASSOCIATION_CONFIDENCE
        or type(association) is not AssociationEvidence
        or association.track_id != current_track_id
        or association.previous_token != previous.token
        or association.current_token != current.token
        or association.detection_source_index != current.source_index
        or association.missed_frame_count_before_association != 0
        or association.ambiguous
        or association.track_ambiguous_before_association
    ):
        return None

    observation_gap_ns = (
        current.observation_monotonic_ns
        - previous.observation_monotonic_ns
    )
    publication_gap_ns = (
        current.publication_monotonic_ns
        - previous.publication_monotonic_ns
    )
    maximum_gap_ns = round(
        ADJACENT_SCALE_JUMP_MAX_OBSERVATION_GAP_S * 1_000_000_000
    )
    association_values = (
        association.cost,
        association.confidence,
        association.predicted_center_residual_norm,
        association.bbox_iou,
        association.log_width_change,
        association.log_height_change,
        association.log_area_residual,
        association.clipping_continuity,
        association.temporal_consistency,
        previous.confidence,
        previous.association_confidence,
        current.confidence,
        current.association_confidence,
        getattr(track, "confidence", math.nan),
        getattr(track, "association_confidence", math.nan),
        target.confidence,
        target.association_confidence,
    )
    if (
        observation_gap_ns <= 0
        or observation_gap_ns > maximum_gap_ns
        or publication_gap_ns <= 0
        or publication_gap_ns > maximum_gap_ns
        or association.observation_gap_ns != observation_gap_ns
        or association.publication_gap_ns != publication_gap_ns
        or not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in association_values
        )
        or float(association.cost)
        > ADJACENT_SCALE_JUMP_MAX_ASSOCIATION_COST
        or float(association.confidence)
        < ADJACENT_SCALE_JUMP_MIN_ASSOCIATION_CONFIDENCE
        or float(association.bbox_iou)
        < ADJACENT_SCALE_JUMP_MIN_BBOX_IOU
        or float(association.predicted_center_residual_norm)
        > ADJACENT_SCALE_JUMP_MAX_CENTER_RESIDUAL_NORM
        or abs(float(association.log_width_change))
        > ADJACENT_SCALE_JUMP_MAX_ABS_LOG_WIDTH_CHANGE
        or abs(float(association.log_height_change))
        > ADJACENT_SCALE_JUMP_MAX_ABS_LOG_HEIGHT_CHANGE
        or abs(float(association.log_area_residual))
        > ADJACENT_SCALE_JUMP_MAX_ABS_LOG_AREA_RESIDUAL
        or not math.isclose(
            float(association.clipping_continuity),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(association.temporal_consistency),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or min(float(previous.confidence), float(current.confidence))
        < ADJACENT_SCALE_JUMP_MIN_DETECTION_CONFIDENCE
        or min(
            float(previous.association_confidence),
            float(current.association_confidence),
            float(track.association_confidence),
            float(target.association_confidence),
        )
        < ADJACENT_SCALE_JUMP_MIN_ASSOCIATION_CONFIDENCE
        or min(float(track.confidence), float(target.confidence))
        < ADJACENT_SCALE_JUMP_MIN_DETECTION_CONFIDENCE
        or not math.isclose(
            float(current.association_confidence),
            float(association.confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(track.association_confidence),
            float(association.confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(target.association_confidence),
            float(association.confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(target.confidence),
            float(track.confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return None

    gap_s = observation_gap_ns / 1_000_000_000.0
    predecessor_projected_x = (
        float(predecessor_target.normalized_x)
        + float(predecessor_target.normalized_x_rate_s) * gap_s
    )
    predecessor_projected_y = (
        float(predecessor_target.normalized_y_down)
        + float(predecessor_target.normalized_y_rate_down_s) * gap_s
    )
    predecessor_projected_log_scale = (
        float(predecessor_target.log_scale)
        + float(predecessor_target.log_scale_rate_s) * gap_s
    )
    log_scale_delta = (
        float(target.log_scale) - float(predecessor_target.log_scale)
    )
    previous_width = float(previous.bbox_norm[2]) - float(
        previous.bbox_norm[0]
    )
    previous_height = float(previous.bbox_norm[3]) - float(
        previous.bbox_norm[1]
    )
    current_width = float(current.bbox_norm[2]) - float(
        current.bbox_norm[0]
    )
    current_height = float(current.bbox_norm[3]) - float(
        current.bbox_norm[1]
    )
    predicted_center_residual = math.hypot(
        float(current.center_norm[0]) - predecessor_projected_x,
        float(current.center_norm[1]) - predecessor_projected_y,
    )
    expected_log_area_residual = 2.0 * (
        float(target.log_scale) - predecessor_projected_log_scale
    )
    if (
        not all(
            math.isfinite(value)
            for value in (
                predecessor_projected_x,
                predecessor_projected_y,
                predecessor_projected_log_scale,
                log_scale_delta,
                previous_width,
                previous_height,
                current_width,
                current_height,
                predicted_center_residual,
                expected_log_area_residual,
            )
        )
        or min(
            previous_width,
            previous_height,
            current_width,
            current_height,
        )
        <= 0.0
        or not 0.0 < log_scale_delta <= ADJACENT_SCALE_JUMP_MAX_LOG_DELTA
        or predecessor_projected_log_scale
        < limits.crossing_arm_min_log_scale
        or float(target.log_scale) < predecessor_projected_log_scale
        or abs(predecessor_projected_x) > tuning.horizontal_corridor
        or abs(predecessor_projected_y) > tuning.vertical_corridor
        or not math.isclose(
            float(association.predicted_center_residual_norm),
            predicted_center_residual,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(association.log_width_change),
            math.log(current_width / previous_width),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(association.log_height_change),
            math.log(current_height / previous_height),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(association.log_area_residual),
            expected_log_area_residual,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return None

    current_wire_projection = _retained_crossing_wire_projection(
        target,
        observation_monotonic_ns=accepted.observation_monotonic_ns,
        wire_start_monotonic_ns=accepted.wire_start_monotonic_ns,
        tuning=tuning,
        limits=limits,
        abort_type=abort_type,
    )
    if (
        current_wire_projection is None
        or not _retained_crossing_projection_matches_target(
            target,
            current_wire_projection,
            tuning=tuning,
            limits=limits,
        )
        or current_wire_projection.projected_log_scale
        < limits.crossing_arm_min_log_scale
    ):
        return None

    return _AdjacentScaleJumpCrossingProof(
        predecessor=predecessor,
        current_wire_projection=current_wire_projection,
        association=association,
        observation_gap_ns=observation_gap_ns,
        publication_gap_ns=publication_gap_ns,
        log_scale_delta=log_scale_delta,
        predecessor_projected_log_scale=(
            predecessor_projected_log_scale
        ),
        predecessor_projected_normalized_x=predecessor_projected_x,
        predecessor_projected_normalized_y_down=predecessor_projected_y,
    )


def _retained_crossing_supersession_usable(
    target: Any,
    output: Any,
    supersession: _SupersededVisualProposal,
    *,
    last_accepted_camera_token: Optional[CameraFrameToken],
    current_track_id: str,
    retained_crossing_dwell_frames: int,
    tuning: Any,
    limits: VisualCourseStageLimits,
) -> bool:
    """Preserve, but never extend, one completed safe retained dwell."""

    target_token = getattr(target, "frame_token", None)
    maximum_hold_s = (
        limits.control_period_s + limits.max_validation_to_wire_delay_s
    )
    return bool(
        type(supersession) is _SupersededVisualProposal
        and supersession.consecutive_count == 1
        and type(last_accepted_camera_token) is CameraFrameToken
        and _tokens_are_adjacent_camera_publications(
            supersession.expected_camera_token,
            last_accepted_camera_token,
        )
        and _tokens_are_adjacent_camera_publications(
            supersession.receiver_camera_token,
            supersession.expected_camera_token,
        )
        and type(supersession.held_previous_command_s) is float
        and math.isfinite(supersession.held_previous_command_s)
        and 0.0 <= supersession.held_previous_command_s <= maximum_hold_s
        and type(current_track_id) is str
        and current_track_id
        and getattr(target, "track_id", None) == current_track_id
        and getattr(target_token, "stream_id", None)
        == supersession.expected_camera_token.stream_id
        and getattr(target_token, "generation", None)
        == supersession.expected_camera_token.generation
        and getattr(target_token, "frame_id", None)
        == supersession.expected_camera_token.frame_id
        and getattr(target_token, "publication_sequence", None)
        == supersession.expected_camera_token.publication_sequence
        and type(retained_crossing_dwell_frames) is int
        and retained_crossing_dwell_frames
        >= tuning.required_corridor_frames
        and _retained_crossing_observation_usable(
            target,
            output,
            tuning=tuning,
            limits=limits,
            yaw_soft_stop_zeroed=False,
        )
    )


def _censored_passage_visibility_suffix_usable(
    track: Any,
    *,
    previous_visible_token: CameraFrameToken,
    current_token: CameraFrameToken,
    previous_apparent_scale: Optional[float],
    minimum_apparent_scale: float,
) -> bool:
    """Prove one uninterrupted near-plane visibility epoch through history."""

    history = getattr(track, "history", None)
    consecutive_frames = getattr(track, "consecutive_frame_count", None)
    if (
        type(history) is not tuple
        or len(history) < 2
        or type(consecutive_frames) is not int
        or consecutive_frames < 2
        or type(previous_visible_token) is not CameraFrameToken
        or type(current_token) is not CameraFrameToken
        or type(minimum_apparent_scale) not in {int, float}
        or not math.isfinite(float(minimum_apparent_scale))
        or (
            previous_apparent_scale is not None
            and (
                type(previous_apparent_scale) not in {int, float}
                or not math.isfinite(float(previous_apparent_scale))
            )
        )
    ):
        return False

    retained_epoch_length = min(consecutive_frames, len(history))
    visibility_epoch = history[len(history) - retained_epoch_length :]
    if any(type(sample) is not VisualTrackSample for sample in visibility_epoch):
        return False
    previous_indices = [
        index
        for index, sample in enumerate(visibility_epoch)
        if sample.token == previous_visible_token
    ]
    if len(previous_indices) != 1:
        return False
    suffix = visibility_epoch[previous_indices[0] :]
    if len(suffix) < 2 or suffix[-1].token != current_token:
        return False

    previous = suffix[0]
    previous_scale = previous.apparent_scale
    if (
        type(previous_scale) not in {int, float}
        or not math.isfinite(float(previous_scale))
        or (
            previous_apparent_scale is not None
            and not math.isclose(
                float(previous_scale),
                float(previous_apparent_scale),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    ):
        return False

    vertical_edges = FrameEdge.TOP | FrameEdge.BOTTOM
    for sample in suffix[1:]:
        evidence = sample.accepted_association
        sample_scale = sample.apparent_scale
        sample_confidence = sample.confidence
        sample_association_confidence = sample.association_confidence
        evidence_confidence = getattr(evidence, "confidence", None)
        if (
            type(evidence) is not AssociationEvidence
            or type(sample.token) is not CameraFrameToken
            or type(previous.token) is not CameraFrameToken
            or type(sample.tracker_frame_sequence) is not int
            or type(previous.tracker_frame_sequence) is not int
            or type(sample.observation_monotonic_ns) is not int
            or type(previous.observation_monotonic_ns) is not int
            or type(sample.publication_monotonic_ns) is not int
            or type(previous.publication_monotonic_ns) is not int
            or evidence.track_id != getattr(track, "track_id", None)
            or evidence.previous_token != previous.token
            or evidence.current_token != sample.token
            or evidence.detection_source_index != sample.source_index
            or type(evidence.missed_frame_count_before_association) is not int
            or evidence.missed_frame_count_before_association != 0
            or type(evidence.ambiguous) is not bool
            or evidence.ambiguous
            or type(evidence.track_ambiguous_before_association) is not bool
            or evidence.track_ambiguous_before_association
            or type(sample_confidence) not in {int, float}
            or not math.isfinite(float(sample_confidence))
            or not 0.0 <= float(sample_confidence) <= 1.0
            or type(sample_association_confidence) not in {int, float}
            or not math.isfinite(float(sample_association_confidence))
            or not 0.10 <= float(sample_association_confidence) <= 1.0
            or type(evidence_confidence) not in {int, float}
            or not math.isfinite(float(evidence_confidence))
            or not math.isclose(
                float(sample_association_confidence),
                float(evidence_confidence),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or sample.token.stream_id != previous.token.stream_id
            or sample.token.generation != previous.token.generation
            or sample.tracker_frame_sequence
            - previous.tracker_frame_sequence
            != 1
            or sample.token.publication_sequence is None
            or previous.token.publication_sequence is None
            or sample.token.publication_sequence
            - previous.token.publication_sequence
            != 1
            or sample.observation_monotonic_ns
            - previous.observation_monotonic_ns
            != evidence.observation_gap_ns
            or sample.publication_monotonic_ns
            - previous.publication_monotonic_ns
            != evidence.publication_gap_ns
            or type(sample.clipping) is not FrameEdge
            or sample.clipping & vertical_edges != vertical_edges
            or sample.center_censored is not True
            or type(sample_scale) not in {int, float}
            or not math.isfinite(float(sample_scale))
            or float(sample_scale) < minimum_apparent_scale
            or float(sample_scale) < float(previous_scale)
        ):
            return False
        previous = sample
        previous_scale = sample_scale

    latest = suffix[-1]
    track_scale = getattr(track, "apparent_scale", None)
    track_confidence = getattr(track, "confidence", None)
    track_association_confidence = getattr(
        track,
        "association_confidence",
        None,
    )
    return bool(
        latest.clipping == getattr(track, "clipping", None)
        and latest.center_censored
        == getattr(track, "center_censored", None)
        and type(track_confidence) in {int, float}
        and math.isfinite(float(track_confidence))
        and 0.10 <= float(track_confidence) <= 1.0
        and type(track_association_confidence) in {int, float}
        and math.isfinite(float(track_association_confidence))
        and 0.10 <= float(track_association_confidence) <= 1.0
        and math.isclose(
            float(latest.association_confidence),
            float(track_association_confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and type(track_scale) in {int, float}
        and math.isfinite(float(track_scale))
        and math.isclose(
            float(latest.apparent_scale),
            float(track_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _censored_passage_coast_eligible(
    snapshot: Any,
    *,
    current_gate_index: int,
    current_track_id: str,
    crossing_anchor_token: CameraFrameToken,
    authority: _CensoredPassageCoastAuthority,
    previous_visible_token: CameraFrameToken,
    previous_apparent_scale: Optional[float],
    minimum_apparent_scale: float,
) -> bool:
    """Admit only the exact near-plane censor pattern seen before target loss."""

    token = getattr(snapshot, "latest_camera_token", None)
    track = getattr(snapshot, "current_track", None)
    clipping = getattr(track, "clipping", None)
    apparent_scale = getattr(track, "apparent_scale", None)
    vertical_edges = FrameEdge.TOP | FrameEdge.BOTTOM
    return bool(
        type(authority) is _CensoredPassageCoastAuthority
        and authority.gate_index == current_gate_index
        and authority.track_id == current_track_id
        and authority.anchor_camera_token == crossing_anchor_token
        and type(token) is CameraFrameToken
        and type(crossing_anchor_token) is CameraFrameToken
        and _token_strictly_newer(token, crossing_anchor_token)
        and type(previous_visible_token) is CameraFrameToken
        and _token_strictly_newer(token, previous_visible_token)
        and token.publication_sequence
        - previous_visible_token.publication_sequence
        <= 2
        and getattr(snapshot, "current_gate_index", None)
        == current_gate_index
        and getattr(snapshot, "current_track_id", None)
        == current_track_id
        and getattr(snapshot, "authority_usable", False) is True
        and getattr(snapshot, "race_finished", False) is False
        and track is not None
        and getattr(track, "track_id", None) == current_track_id
        and getattr(track, "latest_token", None) == token
        and getattr(track, "role", None) is VisualTrackRole.CURRENT
        and getattr(track, "visible", False) is True
        and getattr(track, "missed_frame_count", 1) == 0
        and getattr(track, "ambiguous", True) is False
        and type(clipping) is FrameEdge
        and clipping & vertical_edges == vertical_edges
        and getattr(track, "center_censored", False) is True
        and type(apparent_scale) in {int, float}
        and math.isfinite(float(apparent_scale))
        and float(apparent_scale) >= minimum_apparent_scale
        and (
            previous_apparent_scale is None
            or float(apparent_scale) >= float(previous_apparent_scale)
        )
        and _censored_passage_visibility_suffix_usable(
            track,
            previous_visible_token=previous_visible_token,
            current_token=token,
            previous_apparent_scale=previous_apparent_scale,
            minimum_apparent_scale=minimum_apparent_scale,
        )
    )


def _bottom_censored_passage_transition_eligible(
    snapshot: Any,
    *,
    current_gate_index: int,
    current_track_id: str,
    crossing_anchor_token: CameraFrameToken,
    authority: _CensoredPassageCoastAuthority,
    previous_visible_token: CameraFrameToken,
    previous_apparent_scale: Optional[float],
    minimum_apparent_scale: float,
    passage_admission: VisualApproachPassageAdmission,
    next_preview_retired: bool,
    tuning: Any,
) -> bool:
    """Admit one exact bottom-edge sample before full vertical censorship."""

    token = getattr(snapshot, "latest_camera_token", None)
    track = getattr(snapshot, "current_track", None)
    history = getattr(track, "history", None)
    if (
        type(authority) is not _CensoredPassageCoastAuthority
        or authority.gate_index != current_gate_index
        or authority.track_id != current_track_id
        or authority.anchor_camera_token != crossing_anchor_token
        or type(token) is not CameraFrameToken
        or type(previous_visible_token) is not CameraFrameToken
        or not (
            previous_visible_token == crossing_anchor_token
            or _token_strictly_newer(
                previous_visible_token,
                crossing_anchor_token,
            )
        )
        or not _tokens_are_adjacent_camera_publications(
            token,
            previous_visible_token,
        )
        or type(passage_admission) is not VisualApproachPassageAdmission
        or passage_admission.current_gate_index != current_gate_index
        or passage_admission.current_target.track_id != current_track_id
        or type(next_preview_retired) is not bool
        or (
            passage_admission.preview_track_id is not None
            and not next_preview_retired
        )
        or getattr(snapshot, "current_gate_index", None)
        != current_gate_index
        or getattr(snapshot, "current_track_id", None)
        != current_track_id
        or getattr(snapshot, "authority_usable", False) is not True
        or getattr(snapshot, "race_finished", False) is not False
        or track is None
        or getattr(track, "track_id", None) != current_track_id
        or getattr(track, "latest_token", None) != token
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "visible", False) is not True
        or getattr(track, "missed_frame_count", 1) != 0
        or getattr(track, "ambiguous", True) is not False
        or getattr(track, "clipping", None) != FrameEdge.BOTTOM
        or getattr(track, "center_censored", False) is not True
        or type(history) is not tuple
        or len(history) < 2
        or type(getattr(track, "consecutive_frame_count", None)) is not int
        or track.consecutive_frame_count < 2
    ):
        return False

    previous = history[-2]
    current = history[-1]
    if (
        type(previous) is not VisualTrackSample
        or type(current) is not VisualTrackSample
        or previous.token != previous_visible_token
        or current.token != token
        or type(previous.tracker_frame_sequence) is not int
        or type(current.tracker_frame_sequence) is not int
        or current.tracker_frame_sequence
        != previous.tracker_frame_sequence + 1
        or getattr(snapshot, "tracker_frame_sequence", None)
        != current.tracker_frame_sequence
        or previous.clipping is not FrameEdge.NONE
        or previous.center_censored
        or current.clipping != FrameEdge.BOTTOM
        or current.center_censored is not True
    ):
        return False

    association = current.accepted_association
    if (
        type(association) is not AssociationEvidence
        or association.track_id != current_track_id
        or association.previous_token != previous.token
        or association.current_token != current.token
        or association.detection_source_index != current.source_index
        or type(association.missed_frame_count_before_association) is not int
        or association.missed_frame_count_before_association != 0
        or type(association.ambiguous) is not bool
        or association.ambiguous
        or type(association.track_ambiguous_before_association) is not bool
        or association.track_ambiguous_before_association
        or type(previous.observation_monotonic_ns) is not int
        or type(current.observation_monotonic_ns) is not int
        or current.observation_monotonic_ns
        - previous.observation_monotonic_ns
        != association.observation_gap_ns
        or type(previous.publication_monotonic_ns) is not int
        or type(current.publication_monotonic_ns) is not int
        or current.publication_monotonic_ns
        - previous.publication_monotonic_ns
        != association.publication_gap_ns
    ):
        return False

    previous_center = previous.center_norm
    previous_scale = previous.apparent_scale
    current_scale = current.apparent_scale
    track_scale = getattr(track, "apparent_scale", None)
    sample_confidence = current.confidence
    sample_association_confidence = current.association_confidence
    association_confidence = association.confidence
    track_confidence = getattr(track, "confidence", None)
    track_association_confidence = getattr(
        track,
        "association_confidence",
        None,
    )
    return bool(
        type(previous_center) is tuple
        and len(previous_center) == 2
        and all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in previous_center
        )
        and abs(float(previous_center[0])) <= tuning.horizontal_corridor
        and abs(float(previous_center[1])) <= tuning.vertical_corridor
        and type(previous_scale) in {int, float}
        and math.isfinite(float(previous_scale))
        and type(previous_apparent_scale) in {int, float}
        and math.isfinite(float(previous_apparent_scale))
        and math.isclose(
            float(previous_scale),
            float(previous_apparent_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and float(previous_scale) > PREPASS_CURRENT_MAX_APPARENT_SCALE
        and float(previous_scale) >= minimum_apparent_scale
        and type(current_scale) in {int, float}
        and math.isfinite(float(current_scale))
        and float(current_scale) > float(previous_scale)
        and float(current_scale) >= minimum_apparent_scale
        and type(track_scale) in {int, float}
        and math.isfinite(float(track_scale))
        and math.isclose(
            float(track_scale),
            float(current_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and all(
            type(value) in {int, float}
            and math.isfinite(float(value))
            and 0.10 <= float(value) <= 1.0
            for value in (
                sample_confidence,
                sample_association_confidence,
                association_confidence,
                track_confidence,
                track_association_confidence,
            )
        )
        and math.isclose(
            float(sample_association_confidence),
            float(association_confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(sample_association_confidence),
            float(track_association_confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _current_snapshot_ready(
    snapshot: Any,
    *,
    gate_index: int,
    track_id: str,
    newer_than: Optional[CameraFrameToken] = None,
) -> bool:
    track = getattr(snapshot, "current_track", None)
    token = getattr(snapshot, "latest_camera_token", None)
    if newer_than is not None and not _token_strictly_newer(token, newer_than):
        return False
    return bool(
        getattr(snapshot, "current_gate_index", None) == gate_index
        and getattr(snapshot, "current_track_id", None) == track_id
        and getattr(snapshot, "authority_usable", False) is True
        and getattr(snapshot, "race_finished", False) is False
        and track is not None
        and getattr(track, "track_id", None) == track_id
        and getattr(track, "visible", False) is True
        and getattr(track, "ambiguous", True) is False
        and getattr(track, "missed_frame_count", 1) == 0
    )


def _servo_token_matches_camera(
    servo_token: Any,
    camera_token: CameraFrameToken,
) -> bool:
    return bool(
        type(camera_token) is CameraFrameToken
        and getattr(servo_token, "stream_id", None)
        == camera_token.stream_id
        and getattr(servo_token, "generation", None)
        == camera_token.generation
        and getattr(servo_token, "frame_id", None)
        == camera_token.frame_id
        and getattr(servo_token, "publication_sequence", None)
        == camera_token.publication_sequence
    )


def _recovery_identity_matches_transition(
    admission: Any,
    transition: Any,
) -> bool:
    visibility_tokens = getattr(
        admission,
        "visibility_epoch_tokens",
        None,
    )
    visibility_sequences = getattr(
        admission,
        "visibility_epoch_tracker_frame_sequences",
        None,
    )
    frame_count = getattr(
        admission,
        "visibility_epoch_frame_count",
        None,
    )
    return bool(
        getattr(admission, "track_id", None)
        == transition.promoted_track_id
        and getattr(admission, "promotion_identity_sha256", None)
        == transition.promoted_history_sha256
        and isinstance(
            getattr(admission, "promotion_identity_basis", None),
            str,
        )
        and type(
            getattr(admission, "cross_gap_identity_claimed", None)
        )
        is bool
        and type(visibility_tokens) is tuple
        and type(visibility_sequences) is tuple
        and type(frame_count) is int
        and frame_count >= RECOVERY_HISTORY_SAMPLE_COUNT
        and len(visibility_tokens) == frame_count
        and len(visibility_sequences) == frame_count
        and all(type(token) is CameraFrameToken for token in visibility_tokens)
        and visibility_tokens[-1]
        == transition.promoted_latest_token_at_promotion
    )


async def _run_visual_course_stage_impl(
    host: VisualCourseStageHost,
    context: Any,
    *,
    runtime: VisualCourseStageRuntime,
) -> Dict[str, Any]:
    """Run the generic visual lifecycle until authoritative race finish.

    ``context`` is retained for the runner stage signature and evidence
    boundary; no pose or metric field is read from it.
    """

    abort_type = runtime.safety_abort_type
    limits = runtime.limits
    host._visual_course_summary = {
        "stage": VISUAL_COURSE_STAGE,
        "success": False,
        "race_finished": False,
        "outcome": "starting",
        "first_causal_blocker": None,
        "initial_gate_index": None,
        "maximum_authoritative_gate_index": None,
        "final_gate_index": None,
        "authoritative_transitions": [],
        "segments": [],
        "visual_navigation_command_count": 0,
        "exact_zero_command_count": 0,
        "passage_authority_enabled": False,
        "passage_next_preview_command_count": 0,
        "yaw_calibration_profile": (
            None
            if runtime.yaw_profile is None
            else runtime.yaw_profile.to_evidence()
        ),
    }
    if not host._visual_tracking_enabled:
        raise abort_type("visual-course tracker was not enabled before reset")
    servo_tuning = getattr(host.visual_config, "servo", None)
    if runtime.yaw_profile is not None and (
        runtime.yaw_profile.controller_to_image_sign != 1
        or runtime.yaw_profile.controller_to_body_sign != 1
        or not math.isfinite(
            float(getattr(servo_tuning, "yaw_error_gain", math.nan))
        )
        or float(servo_tuning.yaw_error_gain) <= 0.0
        or not math.isfinite(
            float(getattr(servo_tuning, "yaw_rate_gain", math.nan))
        )
        or float(servo_tuning.yaw_rate_gain) < 0.0
    ):
        raise abort_type(
            "visual-course yaw calibration is not bound to the production "
            "right-image-error-to-negative-controller-yaw convention"
        )
    host._visual_course_summary["yaw_servo_sign_binding"] = {
        "controller_to_image_sign": (
            None
            if runtime.yaw_profile is None
            else runtime.yaw_profile.controller_to_image_sign
        ),
        "controller_to_body_sign": (
            None
            if runtime.yaw_profile is None
            else runtime.yaw_profile.controller_to_body_sign
        ),
        "right_image_error_to_controller_yaw_sign": -1,
        "yaw_error_gain": float(servo_tuning.yaw_error_gain),
        "yaw_rate_gain": float(servo_tuning.yaw_rate_gain),
    }
    initial = host.visual_gate_graph.latest_snapshot
    if (
        initial is None
        or type(getattr(initial, "current_gate_index", None)) is not int
        or not isinstance(getattr(initial, "current_track_id", None), str)
        or not _current_snapshot_ready(
            initial,
            gate_index=initial.current_gate_index,
            track_id=initial.current_track_id,
        )
    ):
        raise abort_type("visual-course lacks an exact initial current gate")

    course_started_s = float(runtime.monotonic())
    if not math.isfinite(course_started_s) or course_started_s < 0.0:
        raise abort_type("visual-course monotonic clock is invalid")
    course_deadline_s = course_started_s + limits.course_hard_duration_s
    next_tick_s = course_started_s
    current_gate_index = int(initial.current_gate_index)
    current_track_id = str(initial.current_track_id)
    initial_gate_index = current_gate_index
    if initial_gate_index == 0:
        try:
            launch_spawn_pitch_rad = float(context.spawn_pitch_rad)
        except (AttributeError, TypeError, ValueError) as exc:
            raise abort_type(
                "visual-course initial launch lacks its proved spawn pitch"
            ) from exc
        if (
            not math.isfinite(launch_spawn_pitch_rad)
            or launch_spawn_pitch_rad
            < limits.min_measured_pitch_rad
            or launch_spawn_pitch_rad
            > limits.max_measured_pitch_rad
        ):
            raise abort_type(
                "visual-course initial launch spawn pitch is outside the "
                "measured course envelope"
            )
    else:
        launch_spawn_pitch_rad = None
    last_race = host._visual_race_status_ref()
    if (
        last_race.active_gate_index != current_gate_index
        or last_race.race_finished
    ):
        raise abort_type("visual-course initial race authority does not match")

    transitions: list[Dict[str, Any]] = []
    segments: list[Dict[str, Any]] = []
    total_navigation_commands = 0
    total_zero_commands = 0
    max_gate_index = current_gate_index
    latest_authoritative_gate_index = current_gate_index
    launch_collective_state: Optional[
        _Gate0ProvedCollectiveState
    ] = None
    host._visual_course_summary.update(
        {
            "outcome": "running",
            "initial_gate_index": initial_gate_index,
            "maximum_authoritative_gate_index": max_gate_index,
            "final_gate_index": current_gate_index,
            "authoritative_transitions": transitions,
            "segments": segments,
        }
    )

    def refresh_live_summary() -> None:
        host._visual_course_summary.update(
            {
                "maximum_authoritative_gate_index": max_gate_index,
                "final_gate_index": latest_authoritative_gate_index,
                "visual_navigation_command_count": (
                    total_navigation_commands
                ),
                "exact_zero_command_count": total_zero_commands,
                "passage_authority_enabled": any(
                    bool(item.get("passage_authority_enabled"))
                    for item in segments
                ),
                "passage_next_preview_command_count": sum(
                    int(
                        item.get(
                            "passage_next_preview_command_count",
                            0,
                        )
                    )
                    for item in segments
                ),
                "course_elapsed_s": (
                    float(runtime.monotonic()) - course_started_s
                ),
            }
        )

    def initial_pad_contact_authority() -> bool:
        elapsed_s = float(runtime.monotonic()) - course_started_s
        return bool(0.0 <= elapsed_s < 0.35)

    async def pace_tick() -> float:
        nonlocal next_tick_s
        observed = float(runtime.monotonic())
        if observed < next_tick_s:
            await runtime.sleep(next_tick_s - observed)
        ready = float(runtime.monotonic())
        if not math.isfinite(ready) or ready < observed:
            raise abort_type("visual-course pacing clock regressed")
        next_tick_s = float(
            runtime.next_control_deadline(
                next_tick_s,
                ready,
                limits.control_period_s,
            )
        )
        if not math.isfinite(next_tick_s) or next_tick_s < (
            ready + limits.control_period_s - 1e-9
        ):
            raise abort_type("visual-course scheduler attempted catch-up")
        return ready

    async def send_zero(
        stage: str,
        elapsed_s: float,
        *,
        yaw_reference_rad: float,
    ) -> None:
        nonlocal total_zero_commands
        await host._wait_for_next_flight_command_slot()
        pad_contact = initial_pad_contact_authority()
        host._watchdog(
            require_target=False,
            allow_benign_pad_contact=pad_contact,
            enforce_benign_pad_budget=True,
            count_rate_sample=False,
        )
        _assert_course_attitude_state(
            host,
            yaw_reference_rad=yaw_reference_rad,
            limits=limits,
            yaw_profile=runtime.yaw_profile,
            abort_type=abort_type,
            phase=f"{stage} pre-send",
        )
        command = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
        runtime.validate_command(command)
        await host._send_flight_command(command)
        host._record_tick(stage, elapsed_s, command)
        total_zero_commands += 1
        refresh_live_summary()

    def assert_pending_supersession_hold(phase: str) -> None:
        if consecutive_superseded_proposals <= 0:
            return
        now_s = float(runtime.monotonic())
        if not math.isfinite(now_s) or now_s < last_navigation_send_s:
            raise abort_type(
                "visual-course supersession clock regressed"
            )
        if (
            now_s - last_navigation_send_s
            >= MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
        ):
            raise abort_type(
                "visual-course receiver supersession held prior command "
                f"beyond its bound during {phase}"
            )

    async def send_visual(
        *,
        proposal: Any,
        snapshot: Any,
        yaw_reference_rad: float,
        segment_started_s: float,
        stage: str,
        preview_requalification_wire_deadline_ns: Optional[int] = None,
    ) -> _AcceptedVisualCommand | _SupersededVisualProposal:
        nonlocal total_navigation_commands
        nonlocal last_navigation_send_s
        nonlocal consecutive_superseded_proposals

        def drop_superseded_proposal(
            exc: BaseException,
        ) -> _SupersededVisualProposal:
            nonlocal consecutive_superseded_proposals

            expected_token = snapshot.latest_camera_token
            receiver_token = getattr(
                exc,
                "receiver_visual_token",
                None,
            )
            if (
                str(exc)
                != VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                or getattr(exc, "expected_visual_token", None)
                != expected_token
                or not _token_strictly_newer(
                    receiver_token,
                    expected_token,
                )
            ):
                raise exc
            now_s = float(runtime.monotonic())
            if not math.isfinite(now_s) or now_s < last_navigation_send_s:
                raise abort_type(
                    "visual-course supersession clock regressed"
                ) from exc
            consecutive_superseded_proposals += 1
            segment["superseded_proposal_count"] = int(
                segment["superseded_proposal_count"]
            ) + 1
            hold_s = now_s - last_navigation_send_s
            host.recorder.emit(
                "visual_course_proposal_superseded",
                gate_index=current_gate_index,
                stage=stage,
                expected_frame_token=asdict(expected_token),
                receiver_frame_token=asdict(receiver_token),
                consecutive_count=consecutive_superseded_proposals,
                total_count=segment["superseded_proposal_count"],
                held_previous_command_s=hold_s,
            )
            refresh_live_summary()
            if (
                consecutive_superseded_proposals
                > MAX_CONSECUTIVE_VISUAL_PROPOSAL_SUPERSESSIONS
                or hold_s
                >= MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
            ):
                raise abort_type(
                    "visual-course receiver repeatedly superseded command "
                    "authority"
                ) from exc
            return _SupersededVisualProposal(
                expected_camera_token=expected_token,
                receiver_camera_token=receiver_token,
                held_previous_command_s=hold_s,
                consecutive_count=consecutive_superseded_proposals,
            )

        output = proposal.servo_output
        observation_monotonic_ns = (
            _current_target_observation_monotonic_ns(
                snapshot,
                proposal.current_target,
                abort_type=abort_type,
            )
        )
        requested_yaw = float(output.yaw_rate_rad_s)
        if requested_yaw != 0.0:
            profile = runtime.yaw_profile
            if profile is None:
                raise abort_type(
                    "visual-course nonzero yaw lacks calibrated authority"
                )
        target_roll_rad = float(output.target_roll_rad)
        target_pitch_rad = float(output.target_pitch_rad)
        command_thrust = float(output.thrust)
        if (
            not all(
                math.isfinite(value)
                for value in (target_roll_rad, target_pitch_rad)
            )
            or abs(target_roll_rad) > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
            or target_pitch_rad < MIN_VISUAL_TARGET_PITCH_RAD - 1e-12
            or target_pitch_rad > MAX_VISUAL_TARGET_PITCH_RAD + 1e-12
        ):
            raise abort_type(
                "visual-course servo target attitude escaped its fixed "
                "passage envelope"
            )
        launch = segment["launch_bootstrap"]
        launch_evidence: Optional[Dict[str, Any]] = None
        if launch["enabled"]:
            assert launch_spawn_pitch_rad is not None
            launch_elapsed_s = max(
                0.0,
                float(runtime.monotonic()) - course_started_s,
            )
            pitch_blend_s = float(
                host.visual_config.lifecycle.launch_pitch_blend_s
            )
            pitch_blend = min(1.0, launch_elapsed_s / pitch_blend_s)
            target_pitch_rad = (
                (1.0 - pitch_blend) * launch_spawn_pitch_rad
                + pitch_blend * target_pitch_rad
            )
            assert launch_collective_state is not None
            (
                proved_collective,
                proved_filtered_vertical_rate,
            ) = launch_collective_state.observe(
                proposal.current_target
            )
            next_preview_collective_delta = 0.0
            next_preview_collective_track_id: Optional[str] = None
            if launch_elapsed_s < INITIAL_PAD_PRELOAD_DURATION_S:
                command_thrust = INITIAL_PAD_PRELOAD_THRUST
                thrust_phase = "preload"
            elif launch_elapsed_s < float(
                host.visual_config.lifecycle.launch_boost_duration_s
            ):
                command_thrust = float(
                    host.visual_config.lifecycle.launch_boost_thrust
                )
                thrust_phase = "boost"
            else:
                # Preserve the already live-proved Gate-0 vertical collective
                # after the fixed launch boost.  The generic minimum-
                # collective approach output lost vertical authority in the
                # exact attempt-7 history and drove a centered aperture into
                # the top edge.  This remains Gate-0 launch-only and inside
                # the unchanged controller thrust envelope.
                try:
                    (
                        command_thrust,
                        next_preview_collective_delta,
                    ) = _gate0_proved_collective_with_exact_next_preview(
                        proved_collective=proved_collective,
                        current_target=proposal.current_target,
                        next_target=getattr(
                            proposal,
                            "next_target",
                            None,
                        ),
                        latched_next_track_id=getattr(
                            proposal,
                            "latched_next_track_id",
                            None,
                        ),
                        servo_output=output,
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course Gate-0 next-preview collective "
                        f"refused exact authority: {exc}"
                    ) from exc
                if next_preview_collective_delta > 0.0:
                    next_preview_collective_track_id = str(
                        proposal.next_target.track_id
                    )
                thrust_phase = GATE0_PROVED_COLLECTIVE_BASIS
            if (
                target_pitch_rad < limits.min_measured_pitch_rad
                or target_pitch_rad > limits.max_measured_pitch_rad
                or not limits.min_thrust
                <= command_thrust
                <= limits.max_thrust
            ):
                raise abort_type(
                    "visual-course launch bootstrap escaped its fixed "
                    "attitude/thrust envelope"
                )
            launch_evidence = {
                "elapsed_s": launch_elapsed_s,
                "pitch_blend": pitch_blend,
                "target_pitch_rad": target_pitch_rad,
                "thrust": command_thrust,
                "thrust_phase": thrust_phase,
                "current_vertical_error_image_down": float(
                    proposal.current_target.normalized_y_down
                ),
                "current_vertical_rate_down_s": float(
                    proposal.current_target.normalized_y_rate_down_s
                ),
                "proved_filtered_vertical_rate_down_s": (
                    proved_filtered_vertical_rate
                ),
                "next_preview_collective_delta": (
                    next_preview_collective_delta
                ),
                "next_preview_collective_track_id": (
                    next_preview_collective_track_id
                ),
            }

        await host._wait_for_next_flight_command_slot()
        assert_pending_supersession_hold("command-slot wait")
        pad_contact = initial_pad_contact_authority()
        host._watchdog(
            require_target=False,
            allow_benign_pad_contact=pad_contact,
            enforce_benign_pad_budget=True,
            count_rate_sample=False,
        )
        try:
            receiver_token = host._assert_visual_receiver_token_current(
                snapshot.latest_camera_token
            )
        except abort_type as exc:
            return drop_superseded_proposal(exc)
        if receiver_token != snapshot.latest_camera_token:
            raise abort_type("visual-course receiver watermark changed")
        send_race = host._visual_race_status_ref()
        if (
            send_race.race_finished
            or send_race.active_gate_index != current_gate_index
        ):
            raise RaceActiveBoundaryChangedBeforeWire(
                "visual-course race boundary changed before navigation send"
            )
        excursion, rates, euler_yaw_rate = _assert_course_attitude_state(
            host,
            yaw_reference_rad=yaw_reference_rad,
            limits=limits,
            yaw_profile=runtime.yaw_profile,
            abort_type=abort_type,
            phase=f"{stage} pre-send",
        )
        bounded_yaw = requested_yaw
        yaw_soft_stop_zeroed = False
        if requested_yaw != 0.0:
            assert runtime.yaw_profile is not None
            bounded_yaw = _limit_calibrated_yaw_request(
                requested_yaw,
                excursion_rad=excursion,
                measured_euler_yaw_rate_rad_s=euler_yaw_rate,
                limits=limits,
                profile=runtime.yaw_profile,
                abort_type=abort_type,
            )
            yaw_soft_stop_zeroed = bounded_yaw == 0.0
        base = runtime.attitude_rate_command(
            host.estimate,
            target_roll_rad=target_roll_rad,
            target_pitch_rad=target_pitch_rad,
            thrust=command_thrust,
        )
        limited = runtime.limit_command_rates(
            base,
            limits.max_command_rate_rad_s,
        )
        command = AttitudeRateCommand(
            roll_rate=float(limited.roll_rate),
            pitch_rate=float(limited.pitch_rate),
            yaw_rate=bounded_yaw,
            thrust=float(limited.thrust),
        )
        runtime.validate_command(command)
        if (
            max(abs(command.roll_rate), abs(command.pitch_rate))
            > limits.max_command_rate_rad_s + 1e-12
            or abs(command.yaw_rate) > limits.max_yaw_rate_rad_s + 1e-12
            or not limits.min_thrust <= command.thrust <= limits.max_thrust
        ):
            raise abort_type("visual-course command escaped its fixed envelope")
        assert_pending_supersession_hold("pre-wire validation")
        validation_ns = runtime.perf_counter_ns()
        if type(validation_ns) is not int or validation_ns < 0:
            raise abort_type("visual-course wire clock is invalid")
        if preview_requalification_wire_deadline_ns is not None:
            if (
                type(preview_requalification_wire_deadline_ns) is not int
                or preview_requalification_wire_deadline_ns < 0
            ):
                raise abort_type(
                    "visual-course preview requalification wire deadline "
                    "is invalid"
                )
            preview_wire_anchor_ns = (
                preview_requalification_wire_deadline_ns
                - round(
                    MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                    * 1_000_000_000
                )
            )
            if validation_ns < preview_wire_anchor_ns:
                raise abort_type(
                    "visual-course preview requalification wire clock "
                    "regressed"
                )
        not_before_ns = (
            None
            if host._last_flight_command_started_ns is None
            else host._last_flight_command_started_ns
            + round(limits.control_period_s * 1_000_000_000)
        )
        deadline_ns = validation_ns + round(
            limits.max_validation_to_wire_delay_s * 1_000_000_000
        )
        if preview_requalification_wire_deadline_ns is not None:
            deadline_ns = min(
                deadline_ns,
                preview_requalification_wire_deadline_ns,
            )
        if consecutive_superseded_proposals > 0:
            hold_checked_s = float(runtime.monotonic())
            if (
                not math.isfinite(hold_checked_s)
                or hold_checked_s < last_navigation_send_s
            ):
                raise abort_type(
                    "visual-course supersession clock regressed"
                )
            hold_remaining_s = (
                MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
                - (hold_checked_s - last_navigation_send_s)
            )
            if hold_remaining_s <= 0.0:
                raise abort_type(
                    "visual-course receiver supersession held prior command "
                    "beyond its bound during wire deadline admission"
                )
            hold_deadline_ns = validation_ns + math.floor(
                hold_remaining_s * 1_000_000_000
            )
            deadline_ns = min(deadline_ns, hold_deadline_ns)
        if (
            deadline_ns <= validation_ns
            or (
                not_before_ns is not None
                and not_before_ns >= deadline_ns
            )
        ):
            if preview_requalification_wire_deadline_ns is not None:
                raise _PreviewRequalificationWireSlotUnavailable(
                    validation_ns
                )
            raise abort_type(
                "visual-course supersession leaves no bounded wire slot"
            )
        try:
            receipt = await host._send_flight_command(
                command,
                require_wire_receipt=True,
                wire_start_not_before_ns=not_before_ns,
                wire_start_deadline_ns=deadline_ns,
                wire_visual_token=snapshot.latest_camera_token,
                wire_race_gate_index=current_gate_index,
            )
        except abort_type as exc:
            return drop_superseded_proposal(exc)
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(
                receipt.get("visual_receiver_authority"),
                Mapping,
            )
        ):
            raise abort_type("visual-course send lacks visual wire authority")
        visual_wire_authority = receipt["visual_receiver_authority"]
        wire_start_monotonic_ns = visual_wire_authority.get(
            "call_start_monotonic_ns"
        )
        top_level_wire_start_ns = receipt.get("call_start_monotonic_ns")
        wire_frame_token = visual_wire_authority.get("frame_token")
        if (
            visual_wire_authority.get("schema")
            != "aigp-vq2-visual-wire-authority/1"
            or type(wire_start_monotonic_ns) is not int
            or wire_start_monotonic_ns < 0
            or top_level_wire_start_ns != wire_start_monotonic_ns
            or host._last_flight_command_started_ns
            != wire_start_monotonic_ns
            or (
                preview_requalification_wire_deadline_ns is not None
                and wire_start_monotonic_ns
                >= preview_requalification_wire_deadline_ns
            )
            or not isinstance(wire_frame_token, Mapping)
            or dict(wire_frame_token)
            != asdict(snapshot.latest_camera_token)
            or visual_wire_authority.get(
                "publication_pinned_through_transport_return"
            )
            is not True
        ):
            raise abort_type(
                "visual-course send lacks exact visual wire timing"
            )
        if yaw_soft_stop_zeroed:
            segment["yaw_soft_stop_zero_command_count"] = int(
                segment["yaw_soft_stop_zero_command_count"]
            ) + 1
            host.recorder.emit(
                "visual_course_yaw_soft_stop_zeroed",
                gate_index=current_gate_index,
                stage=stage,
                camera_token=asdict(snapshot.latest_camera_token),
                requested_yaw_rate_rad_s=requested_yaw,
                admitted_yaw_rate_rad_s=bounded_yaw,
                yaw_excursion_rad=excursion,
                measured_euler_yaw_rate_rad_s=euler_yaw_rate,
                count=segment["yaw_soft_stop_zero_command_count"],
            )
        if launch_evidence is not None:
            if int(launch["command_count"]) == 0:
                launch["first_target_pitch_rad"] = launch_evidence[
                    "target_pitch_rad"
                ]
                launch["first_thrust"] = launch_evidence["thrust"]
            launch["command_count"] = int(launch["command_count"]) + 1
            launch["last_elapsed_s"] = launch_evidence["elapsed_s"]
            launch["last_pitch_blend"] = launch_evidence["pitch_blend"]
            launch["last_target_pitch_rad"] = launch_evidence[
                "target_pitch_rad"
            ]
            launch["last_thrust"] = launch_evidence["thrust"]
            launch["last_thrust_phase"] = launch_evidence[
                "thrust_phase"
            ]
            launch["last_current_vertical_error_image_down"] = (
                launch_evidence[
                    "current_vertical_error_image_down"
                ]
            )
            launch["last_current_vertical_rate_down_s"] = (
                launch_evidence["current_vertical_rate_down_s"]
            )
            launch["last_proved_filtered_vertical_rate_down_s"] = (
                launch_evidence[
                    "proved_filtered_vertical_rate_down_s"
                ]
            )
            preview_delta = float(
                launch_evidence["next_preview_collective_delta"]
            )
            launch["last_next_preview_collective_delta"] = (
                preview_delta
            )
            launch["max_next_preview_collective_delta"] = max(
                float(launch["max_next_preview_collective_delta"]),
                preview_delta,
            )
            if preview_delta > 0.0:
                launch["next_preview_collective_command_count"] = (
                    int(
                        launch[
                            "next_preview_collective_command_count"
                        ]
                    )
                    + 1
                )
                launch["last_next_preview_collective_track_id"] = (
                    launch_evidence[
                        "next_preview_collective_track_id"
                    ]
                )
        host._record_tick(
            stage,
            float(runtime.monotonic()) - segment_started_s,
            command,
        )
        total_navigation_commands += 1
        last_navigation_send_s = float(runtime.monotonic())
        consecutive_superseded_proposals = 0
        if (
            transitions
            and transitions[-1]["to_gate_index"] == current_gate_index
        ):
            transitions[-1][
                "post_transition_navigation_command_count"
            ] = int(
                transitions[-1][
                    "post_transition_navigation_command_count"
                ]
            ) + 1
        refresh_live_summary()
        return _AcceptedVisualCommand(
            command=command,
            yaw_soft_stop_zeroed=yaw_soft_stop_zeroed,
            observation_monotonic_ns=observation_monotonic_ns,
            wire_start_monotonic_ns=wire_start_monotonic_ns,
            target_roll_rad=target_roll_rad,
            target_pitch_rad=target_pitch_rad,
            next_preview_collective_delta=(
                0.0
                if launch_evidence is None
                else float(
                    launch_evidence[
                        "next_preview_collective_delta"
                    ]
                )
            ),
        )

    async def send_censored_passage_coast(
        *,
        snapshot: Any,
        authority: _CensoredPassageCoastAuthority,
        yaw_reference_rad: float,
        segment_started_s: float,
        stage: str,
    ) -> Optional[AttitudeRateCommand]:
        """Reissue one frozen clean attitude target on an exact censored frame."""

        nonlocal total_navigation_commands
        nonlocal last_navigation_send_s
        nonlocal consecutive_superseded_proposals

        def drop_superseded_coast(
            exc: BaseException,
        ) -> Optional[AttitudeRateCommand]:
            nonlocal consecutive_superseded_proposals

            expected_token = snapshot.latest_camera_token
            receiver_token = getattr(exc, "receiver_visual_token", None)
            if (
                str(exc)
                != VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                or getattr(exc, "expected_visual_token", None)
                != expected_token
                or not _token_strictly_newer(
                    receiver_token,
                    expected_token,
                )
            ):
                raise exc
            now_s = float(runtime.monotonic())
            if not math.isfinite(now_s) or now_s < last_navigation_send_s:
                raise abort_type(
                    "visual-course supersession clock regressed"
                ) from exc
            consecutive_superseded_proposals += 1
            segment["superseded_proposal_count"] = int(
                segment["superseded_proposal_count"]
            ) + 1
            hold_s = now_s - last_navigation_send_s
            host.recorder.emit(
                "visual_course_proposal_superseded",
                gate_index=current_gate_index,
                stage=stage,
                expected_frame_token=asdict(expected_token),
                receiver_frame_token=asdict(receiver_token),
                consecutive_count=consecutive_superseded_proposals,
                total_count=segment["superseded_proposal_count"],
                held_previous_command_s=hold_s,
            )
            refresh_live_summary()
            if (
                consecutive_superseded_proposals
                > MAX_CONSECUTIVE_VISUAL_PROPOSAL_SUPERSESSIONS
                or hold_s
                >= MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
            ):
                raise abort_type(
                    "visual-course receiver repeatedly superseded command "
                    "authority"
                ) from exc
            return None

        values = (
            authority.target_roll_rad,
            authority.target_pitch_rad,
            authority.thrust,
        )
        if (
            authority.gate_index != current_gate_index
            or authority.track_id != current_track_id
            or not all(math.isfinite(value) for value in values)
            or abs(authority.target_roll_rad)
            > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
            or authority.target_pitch_rad
            < MIN_VISUAL_TARGET_PITCH_RAD - 1e-12
            or authority.target_pitch_rad
            > MAX_VISUAL_TARGET_PITCH_RAD + 1e-12
            or not limits.min_thrust
            <= authority.thrust
            <= limits.max_thrust
        ):
            raise abort_type(
                "visual-course censored passage coast authority is invalid"
            )

        await host._wait_for_next_flight_command_slot()
        assert_pending_supersession_hold("censored coast command-slot wait")
        pad_contact = initial_pad_contact_authority()
        host._watchdog(
            require_target=False,
            allow_benign_pad_contact=pad_contact,
            enforce_benign_pad_budget=True,
            count_rate_sample=False,
        )
        try:
            receiver_token = host._assert_visual_receiver_token_current(
                snapshot.latest_camera_token
            )
        except abort_type as exc:
            return drop_superseded_coast(exc)
        if receiver_token != snapshot.latest_camera_token:
            raise abort_type("visual-course receiver watermark changed")
        send_race = host._visual_race_status_ref()
        if (
            send_race.race_finished
            or send_race.active_gate_index != current_gate_index
        ):
            raise RaceActiveBoundaryChangedBeforeWire(
                "visual-course race boundary changed before navigation send"
            )
        if censored_passage_coast_started_s is None:
            raise abort_type(
                "visual-course censored passage coast lacks a start time"
            )
        coast_deadline_s = (
            censored_passage_coast_started_s
            + limits.censored_passage_coast_max_duration_s
        )
        _assert_course_attitude_state(
            host,
            yaw_reference_rad=yaw_reference_rad,
            limits=limits,
            yaw_profile=runtime.yaw_profile,
            abort_type=abort_type,
            phase=f"{stage} pre-send",
        )
        if float(runtime.monotonic()) >= coast_deadline_s:
            raise abort_type(
                "visual-course censored passage coast expired"
            )
        base = runtime.attitude_rate_command(
            host.estimate,
            target_roll_rad=authority.target_roll_rad,
            target_pitch_rad=authority.target_pitch_rad,
            thrust=authority.thrust,
        )
        limited = runtime.limit_command_rates(
            base,
            limits.max_command_rate_rad_s,
        )
        command = AttitudeRateCommand(
            roll_rate=float(limited.roll_rate),
            pitch_rate=float(limited.pitch_rate),
            yaw_rate=0.0,
            thrust=float(limited.thrust),
        )
        runtime.validate_command(command)
        if (
            max(abs(command.roll_rate), abs(command.pitch_rate))
            > limits.max_command_rate_rad_s + 1e-12
            or command.yaw_rate != 0.0
            or command.thrust != authority.thrust
            or not limits.min_thrust <= command.thrust <= limits.max_thrust
        ):
            raise abort_type(
                "visual-course censored passage command escaped its fixed "
                "envelope"
            )
        assert_pending_supersession_hold(
            "censored coast pre-wire validation"
        )
        validation_ns = runtime.perf_counter_ns()
        if type(validation_ns) is not int or validation_ns < 0:
            raise abort_type("visual-course wire clock is invalid")
        not_before_ns = (
            None
            if host._last_flight_command_started_ns is None
            else host._last_flight_command_started_ns
            + round(limits.control_period_s * 1_000_000_000)
        )
        deadline_ns = validation_ns + round(
            limits.max_validation_to_wire_delay_s * 1_000_000_000
        )
        coast_deadline_ns = round(
            coast_deadline_s * 1_000_000_000
        )
        deadline_ns = min(deadline_ns, coast_deadline_ns)
        if consecutive_superseded_proposals > 0:
            hold_checked_s = float(runtime.monotonic())
            if (
                not math.isfinite(hold_checked_s)
                or hold_checked_s < last_navigation_send_s
            ):
                raise abort_type(
                    "visual-course supersession clock regressed"
                )
            hold_remaining_s = (
                MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
                - (hold_checked_s - last_navigation_send_s)
            )
            if hold_remaining_s <= 0.0:
                raise abort_type(
                    "visual-course receiver supersession held prior command "
                    "beyond its bound during wire deadline admission"
                )
            deadline_ns = min(
                deadline_ns,
                validation_ns
                + math.floor(hold_remaining_s * 1_000_000_000),
            )
        if (
            deadline_ns <= validation_ns
            or (
                not_before_ns is not None
                and not_before_ns >= deadline_ns
            )
        ):
            raise abort_type(
                "visual-course supersession leaves no bounded wire slot"
            )
        try:
            receipt = await host._send_flight_command(
                command,
                require_wire_receipt=True,
                wire_start_not_before_ns=not_before_ns,
                wire_start_deadline_ns=deadline_ns,
                wire_visual_token=snapshot.latest_camera_token,
                wire_race_gate_index=current_gate_index,
            )
        except abort_type as exc:
            return drop_superseded_coast(exc)
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(
                receipt.get("visual_receiver_authority"),
                Mapping,
            )
        ):
            raise abort_type(
                "visual-course censored passage send lacks visual wire "
                "authority"
            )
        visual_wire_authority = receipt["visual_receiver_authority"]
        wire_start_monotonic_ns = visual_wire_authority.get(
            "call_start_monotonic_ns"
        )
        wire_frame_token = visual_wire_authority.get("frame_token")
        if (
            visual_wire_authority.get("schema")
            != "aigp-vq2-visual-wire-authority/1"
            or type(wire_start_monotonic_ns) is not int
            or wire_start_monotonic_ns < 0
            or wire_start_monotonic_ns >= coast_deadline_ns
            or receipt.get("call_start_monotonic_ns")
            != wire_start_monotonic_ns
            or host._last_flight_command_started_ns
            != wire_start_monotonic_ns
            or not isinstance(wire_frame_token, Mapping)
            or dict(wire_frame_token)
            != asdict(snapshot.latest_camera_token)
            or visual_wire_authority.get(
                "publication_pinned_through_transport_return"
            )
            is not True
        ):
            raise abort_type(
                "visual-course censored passage send lacks exact visual "
                "wire timing"
            )

        host._record_tick(
            stage,
            float(runtime.monotonic()) - segment_started_s,
            command,
        )
        total_navigation_commands += 1
        last_navigation_send_s = float(runtime.monotonic())
        consecutive_superseded_proposals = 0
        if segment["launch_bootstrap"]["enabled"]:
            launch = segment["launch_bootstrap"]
            launch["command_count"] = int(launch["command_count"]) + 1
            launch["last_elapsed_s"] = (
                float(runtime.monotonic()) - course_started_s
            )
            launch["last_target_pitch_rad"] = authority.target_pitch_rad
            launch["last_thrust"] = authority.thrust
            launch["last_thrust_phase"] = CENSORED_PASSAGE_COAST_BASIS
        if (
            transitions
            and transitions[-1]["to_gate_index"] == current_gate_index
        ):
            transitions[-1][
                "post_transition_navigation_command_count"
            ] = int(
                transitions[-1][
                    "post_transition_navigation_command_count"
                ]
            ) + 1
        host.recorder.emit(
            "visual_course_censored_passage_coast_command",
            gate_index=current_gate_index,
            stage=stage,
            camera_token=asdict(snapshot.latest_camera_token),
            anchor_camera_token=asdict(authority.anchor_camera_token),
            target_roll_rad=authority.target_roll_rad,
            target_pitch_rad=authority.target_pitch_rad,
            thrust=authority.thrust,
            command=asdict(command),
        )
        refresh_live_summary()
        return command

    for segment_number in range(limits.max_gate_segments):
        segment_started_s = float(runtime.monotonic())
        segment_deadline_s = min(
            course_deadline_s,
            segment_started_s + limits.segment_hard_duration_s,
        )
        _roll, _pitch, yaw_reference_rad, _rates = _attitude_state(
            host,
            abort_type,
        )
        launch_enabled = bool(
            segment_number == 0
            and initial_gate_index == 0
            and current_gate_index == initial_gate_index
        )
        launch_collective_state = (
            _Gate0ProvedCollectiveState()
            if launch_enabled
            else None
        )

        def make_planner(
            *,
            next_gate_blend: float,
            required_next_track_id: Optional[str] = None,
        ) -> Any:
            kwargs = {
                "next_gate_blend": next_gate_blend,
                "next_gate_blend_start_log_scale": (
                    host.visual_config.lifecycle
                    .next_gate_blend_start_log_scale
                ),
                "next_gate_blend_full_log_scale": (
                    host.visual_config.lifecycle
                    .next_gate_blend_full_log_scale
                ),
            }
            if required_next_track_id is not None:
                kwargs["required_next_track_id"] = required_next_track_id
            return runtime.servo_factory(
                current_track_id,
                current_gate_index,
                host.visual_config.servo,
                **kwargs,
            )

        planner = make_planner(
            next_gate_blend=(
                host.visual_config.lifecycle.next_gate_blend_max
            ),
        )
        mode = VisualApproachMode.APPROACH
        passage_admission: Optional[VisualApproachPassageAdmission] = None
        passage_started_s: Optional[float] = None
        passage_command_count = 0
        passage_next_preview_command_count = 0
        advance_command_count = 0
        approach_command_count = 0
        next_preview_retired = False
        next_preview_requalification_used = False
        next_preview_requalification: Optional[Dict[str, Any]] = None
        crossing_anchor: Optional[Dict[str, Any]] = None
        crossing_coast_authority: Optional[
            _CensoredPassageCoastAuthority
        ] = None
        last_clean_passage_token: Optional[CameraFrameToken] = None
        last_clean_passage_scale: Optional[float] = None
        censored_passage_coast_started_s: Optional[float] = None
        censored_passage_coast_last_observed_token: Optional[
            CameraFrameToken
        ] = None
        censored_passage_coast_last_observed_scale: Optional[float] = None
        censored_passage_coast_fresh_frame_count = 0
        censored_passage_coast_command_count = 0
        bottom_censored_transition_pending_token: Optional[
            CameraFrameToken
        ] = None
        bottom_censored_transition_used = False
        retained_crossing_dwell_frames = 0
        retained_crossing_supersession: Optional[
            _SupersededVisualProposal
        ] = None
        retained_crossing_candidate: Optional[
            _RetainedCrossingCandidate
        ] = None
        crossing_started_s: Optional[float] = None
        crossing_baseline_race: Optional[AuthoritativeRaceStatusRef] = None
        last_planned_token: Optional[CameraFrameToken] = None
        last_navigation_send_s = segment_started_s
        consecutive_superseded_proposals = 0
        segment = {
            "segment_number": segment_number,
            "gate_index": current_gate_index,
            "current_track_id": current_track_id,
            "approach_command_count": 0,
            "passage_command_count": 0,
            "passage_next_preview_command_count": 0,
            "advance_command_count": 0,
            "superseded_proposal_count": 0,
            "next_preview_withdrawal_count": 0,
            "next_preview_withdrawal": None,
            "next_preview_retired": False,
            "next_preview_requalification_count": 0,
            "next_preview_requalification": None,
            "yaw_soft_stop_zero_command_count": 0,
            "passage_admission_yaw_soft_stop_withheld_count": 0,
            "retained_crossing_dwell_frames": 0,
            "max_retained_crossing_dwell_frames": 0,
            "retained_crossing_supersession_hold_count": 0,
            "retained_crossing_supersession_hold": None,
            "adjacent_scale_jump_crossing_count": 0,
            "crossing_wait_zero_command_count": 0,
            "censored_passage_coast_fresh_frame_count": 0,
            "censored_passage_coast_command_count": 0,
            "censored_passage_coast": None,
            "bottom_censored_transition_count": 0,
            "bottom_censored_transition": None,
            "post_credit_zero_command_count": 0,
            "passage_authority_enabled": False,
            "passage_admission": None,
            "crossing_anchor": None,
            "outcome": "running",
            "launch_bootstrap": {
                "enabled": launch_enabled,
                "preload_duration_s": INITIAL_PAD_PRELOAD_DURATION_S,
                "preload_thrust": INITIAL_PAD_PRELOAD_THRUST,
                "post_boost_collective_basis": (
                    GATE0_PROVED_COLLECTIVE_BASIS
                    if launch_enabled
                    else None
                ),
                "post_boost_collective_base": (
                    GATE0_PROVED_COLLECTIVE_BASE
                    if launch_enabled
                    else None
                ),
                "post_boost_collective_error_gain": (
                    GATE0_PROVED_COLLECTIVE_ERROR_GAIN
                    if launch_enabled
                    else None
                ),
                "post_boost_collective_rate_gain": (
                    GATE0_PROVED_COLLECTIVE_RATE_GAIN
                    if launch_enabled
                    else None
                ),
                "post_boost_collective_max_abs_error": (
                    GATE0_PROVED_COLLECTIVE_MAX_ABS_ERROR
                    if launch_enabled
                    else None
                ),
                "post_boost_collective_max_abs_rate": (
                    GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE
                    if launch_enabled
                    else None
                ),
                "post_boost_collective_rate_filter_alpha": (
                    GATE0_PROVED_COLLECTIVE_RATE_FILTER_ALPHA
                    if launch_enabled
                    else None
                ),
                "post_boost_next_preview_collective_basis": (
                    GATE0_PROVED_NEXT_PREVIEW_BASIS
                    if launch_enabled
                    else None
                ),
                "post_boost_next_preview_collective_error_gain": (
                    GATE0_PROVED_NEXT_PREVIEW_ERROR_GAIN
                    if launch_enabled
                    else None
                ),
                "post_boost_next_preview_collective_max_thrust_delta": (
                    GATE0_PROVED_NEXT_PREVIEW_MAX_THRUST_DELTA
                    if launch_enabled
                    else None
                ),
                "boost_duration_s": float(
                    host.visual_config.lifecycle.launch_boost_duration_s
                ),
                "boost_thrust": float(
                    host.visual_config.lifecycle.launch_boost_thrust
                ),
                "pitch_blend_s": float(
                    host.visual_config.lifecycle.launch_pitch_blend_s
                ),
                "spawn_pitch_rad": (
                    launch_spawn_pitch_rad
                    if segment_number == 0
                    and initial_gate_index == 0
                    else None
                ),
                "command_count": 0,
                "next_preview_collective_command_count": 0,
                "max_next_preview_collective_delta": 0.0,
                "last_next_preview_collective_delta": 0.0,
                "last_next_preview_collective_track_id": None,
            },
        }
        segments.append(segment)
        refresh_live_summary()

        credited_race: Optional[AuthoritativeRaceStatusRef] = None

        def accept_bottom_transition_credit() -> None:
            nonlocal bottom_censored_transition_pending_token

            if bottom_censored_transition_pending_token is None:
                return
            transition_evidence = segment[
                "bottom_censored_transition"
            ]
            coast_evidence = segment["censored_passage_coast"]
            if (
                not isinstance(transition_evidence, dict)
                or not isinstance(coast_evidence, dict)
            ):
                raise abort_type(
                    "visual-course bottom-censored transition lost its "
                    "credit evidence"
                )
            transition_evidence["outcome"] = (
                "authoritative_credit_before_full_censorship"
            )
            coast_evidence["bottom_transition_pending"] = False
            bottom_censored_transition_pending_token = None
            host.recorder.emit(
                "visual_course_bottom_censored_transition_credited",
                gate_index=current_gate_index,
                stage=(
                    f"{VISUAL_COURSE_STAGE}/gate"
                    f"{current_gate_index}/censored-passage"
                ),
                **transition_evidence,
            )

        def accept_no_wire_race_boundary(
            exc: RaceActiveBoundaryChangedBeforeWire,
        ) -> AuthoritativeRaceStatusRef:
            """Consume only proved credit after a navigation send was refused."""

            nonlocal crossing_started_s
            nonlocal last_race

            refused_race = host._visual_race_status_ref()
            relation = _race_relation(
                refused_race,
                last_race,
                abort_type,
            )
            if crossing_anchor is None:
                raise abort_type(
                    "visual-course race boundary changed before wire "
                    "without previously latched passage evidence"
                ) from exc
            if relation <= 0:
                raise abort_type(
                    "visual-course no-wire race boundary lacks newer "
                    "authoritative ingress"
                ) from exc
            if not (
                refused_race.race_finished
                or refused_race.active_gate_index
                == current_gate_index + 1
            ):
                raise abort_type(
                    "visual-course no-wire race boundary is not a "
                    "sequential credit or race finish"
                ) from exc
            last_race = refused_race
            crossing_started_s = (
                crossing_started_s or float(runtime.monotonic())
            )
            accept_bottom_transition_credit()
            return refused_race

        def current_only_replan(
            snapshot: Any,
            *,
            now: float,
            excursion: float,
            required_next_track_id: Optional[str],
        ) -> tuple[Any, Any]:
            """Consume one exact approach publication with no preview."""

            replacement = make_planner(
                next_gate_blend=0.0,
                required_next_track_id=required_next_track_id,
            )
            try:
                replacement_proposal = replacement.observe(
                    snapshot,
                    host.visual_tracker,
                    runtime.perf_counter_ns() / 1_000_000_000.0,
                    now - segment_started_s,
                    excursion,
                    mode=VisualApproachMode.APPROACH,
                    passage_admission=None,
                )
            except VisualApproachRefusal as exc:
                raise abort_type(
                    "visual-course current-only replan failed after preview "
                    f"retirement: {exc}"
                ) from exc
            fallback_admission = replacement_proposal.passage_admission
            identity_only_admission = bool(
                type(fallback_admission)
                is VisualApproachPassageAdmission
                and type(fallback_admission.preview_track_id) is str
                and bool(fallback_admission.preview_track_id)
                and fallback_admission.preview_blend == 0.0
                and (
                    required_next_track_id is None
                    or fallback_admission.preview_track_id
                    == required_next_track_id
                )
            )
            if (
                replacement_proposal.mode
                is not VisualApproachMode.APPROACH
                or replacement_proposal.servo_output.advance_enabled
                or replacement_proposal.servo_output.next_gate_blend != 0.0
                or (
                    fallback_admission is not None
                    and not identity_only_admission
                )
            ):
                raise abort_type(
                    "visual-course preview retirement retained prior blend, "
                    "advance, or unreviewed admission authority"
                )
            return replacement, replacement_proposal

        def record_preview_retirement(
            *,
            reason: str,
            token: CameraFrameToken,
            tracker_frame_sequence: int,
            violation_codes: list[str],
            violation_evidence: list[Dict[str, Any]],
            transient_eligible: bool,
        ) -> None:
            nonlocal next_preview_retired
            nonlocal next_preview_requalification

            if next_preview_retired:
                raise abort_type(
                    "visual-course attempted to retire preview more than once"
                )
            if next_preview_requalification is not None:
                next_preview_requalification.update(
                    {
                        "outcome": "retired",
                        "retirement_reason": reason,
                        "retirement_camera_token": asdict(token),
                    }
                )
                next_preview_requalification = None
            next_preview_retired = True
            withdrawal = {
                "reason": reason,
                "camera_token": asdict(token),
                "tracker_frame_sequence": tracker_frame_sequence,
                "violation_codes": violation_codes,
                "violation_evidence": violation_evidence,
                "transient_eligible": transient_eligible,
            }
            segment["next_preview_withdrawal_count"] = int(
                segment["next_preview_withdrawal_count"]
            ) + 1
            segment["next_preview_withdrawal"] = withdrawal
            segment["next_preview_retired"] = True
            host.recorder.emit(
                "visual_course_next_preview_withdrawn",
                gate_index=current_gate_index,
                stage=(
                    f"{VISUAL_COURSE_STAGE}/gate"
                    f"{current_gate_index}/approach"
                ),
                **withdrawal,
            )

        while credited_race is None:
            now = await pace_tick()
            assert_pending_supersession_hold("paced control tick")
            if now >= course_deadline_s:
                raise abort_type("visual-course hard duration expired")
            if now >= segment_deadline_s:
                raise abort_type(
                    f"visual-course gate {current_gate_index} segment expired"
                )
            if (
                passage_started_s is not None
                and crossing_started_s is None
                and now - passage_started_s > limits.passage_hard_duration_s
            ):
                raise abort_type(
                    f"visual-course gate {current_gate_index} passage expired"
                )

            host._sample()
            pad_contact = initial_pad_contact_authority()
            host._watchdog(
                require_target=False,
                allow_benign_pad_contact=pad_contact,
                enforce_benign_pad_budget=True,
            )
            excursion, _rates, _euler_yaw_rate = (
                _assert_course_attitude_state(
                host,
                yaw_reference_rad=yaw_reference_rad,
                limits=limits,
                yaw_profile=runtime.yaw_profile,
                abort_type=abort_type,
                phase=(
                    f"gate {current_gate_index} "
                    f"{mode.value} control tick"
                ),
                )
            )
            race = host._visual_race_status_ref()
            relation = _race_relation(race, last_race, abort_type)
            if relation < 0:
                raise abort_type("visual-course race ingress regressed")
            if (
                not race.race_finished
                and race.active_gate_index == current_gate_index
            ):
                if relation > 0:
                    last_race = race
                if crossing_started_s is None:
                    crossing_baseline_race = race
            elif (
                race.race_finished
                or race.active_gate_index == current_gate_index + 1
            ):
                if relation <= 0:
                    raise abort_type(
                        "visual-course transition lacks newer race ingress"
                    )
                if crossing_anchor is None:
                    raise abort_type(
                        "visual-course race credit arrived without credible "
                        "passage evidence"
                    )
                accept_bottom_transition_credit()
                credited_race = race
                last_race = race
                crossing_started_s = crossing_started_s or now
                break
            else:
                raise abort_type(
                    "visual-course authoritative gate index jumped "
                    f"{current_gate_index}->{race.active_gate_index}"
                )
            if (
                censored_passage_coast_started_s is not None
                and now
                >= (
                    censored_passage_coast_started_s
                    + limits.censored_passage_coast_max_duration_s
                )
            ):
                raise abort_type(
                    "visual-course censored passage coast expired"
                )

            snapshot = host.visual_gate_graph.latest_snapshot
            token = getattr(snapshot, "latest_camera_token", None)
            if type(token) is not CameraFrameToken:
                raise abort_type("visual-course graph lacks exact camera token")
            if next_preview_requalification is not None:
                control_checked_ns = runtime.perf_counter_ns()
                control_anchor_ns = int(
                    next_preview_requalification[
                        "refusal_control_perf_counter_ns"
                    ]
                )
                if (
                    type(control_checked_ns) is not int
                    or control_checked_ns < control_anchor_ns
                ):
                    raise abort_type(
                        "visual-course preview requalification control clock "
                        "regressed"
                    )
                control_elapsed_ns = (
                    control_checked_ns - control_anchor_ns
                )
                control_elapsed_s = (
                    control_elapsed_ns / 1_000_000_000
                )
                next_preview_requalification["control_elapsed_s"] = (
                    control_elapsed_s
                )
                next_preview_requalification["control_elapsed_ns"] = (
                    control_elapsed_ns
                )
                if (
                    control_elapsed_ns
                    > round(
                        MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                        * 1_000_000_000
                    )
                ):
                    sealed_next_track_id = (
                        next_preview_requalification[
                            "sealed_next_track_id"
                        ]
                    )
                    planner = make_planner(
                        next_gate_blend=0.0,
                        required_next_track_id=sealed_next_track_id,
                    )
                    record_preview_retirement(
                        reason=(
                            "preview_requalification_control_timeout"
                        ),
                        token=token,
                        tracker_frame_sequence=(
                            snapshot.tracker_frame_sequence
                        ),
                        violation_codes=[
                            "requalification_control_duration_s",
                        ],
                        violation_evidence=[
                            {
                                "code": (
                                    "requalification_control_duration_s"
                                ),
                                "observed": control_elapsed_s,
                                "limit": (
                                    MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                                ),
                                "excess": (
                                    control_elapsed_s
                                    - (
                                        MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                                    )
                                ),
                            },
                        ],
                        transient_eligible=False,
                    )
            if last_planned_token is not None and token == last_planned_token:
                continue
            if bottom_censored_transition_pending_token is not None:
                pending_track = getattr(snapshot, "current_track", None)
                vertical_edges = FrameEdge.TOP | FrameEdge.BOTTOM
                if (
                    not _tokens_are_adjacent_camera_publications(
                        token,
                        bottom_censored_transition_pending_token,
                    )
                    or pending_track is None
                    or getattr(pending_track, "track_id", None)
                    != current_track_id
                    or getattr(pending_track, "latest_token", None) != token
                    or getattr(pending_track, "visible", False) is not True
                    or getattr(pending_track, "missed_frame_count", 1) != 0
                    or getattr(pending_track, "ambiguous", True) is not False
                    or type(getattr(pending_track, "clipping", None))
                    is not FrameEdge
                    or pending_track.clipping & vertical_edges
                    != vertical_edges
                    or getattr(
                        pending_track,
                        "center_censored",
                        False,
                    )
                    is not True
                ):
                    raise abort_type(
                        "visual-course bottom-censored transition did not "
                        "become full vertical censorship on its exact "
                        "successor"
                    )

            preview_requalification_wire_candidate = False
            try:
                proposal = planner.observe(
                    snapshot,
                    host.visual_tracker,
                    runtime.perf_counter_ns() / 1_000_000_000.0,
                    now - segment_started_s,
                    excursion,
                    mode=mode,
                    passage_admission=(
                        passage_admission
                        if mode is VisualApproachMode.PASSAGE
                        else None
                    ),
                )
            except VisualApproachPassageSafetyUnavailable as exc:
                if (
                    mode is not VisualApproachMode.APPROACH
                    or next_preview_retired
                ):
                    raise abort_type(
                        "visual-course passage safety failed after preview "
                        "retirement or passage entry"
                    ) from exc
                violation_evidence = [
                    {
                        "code": code,
                        "observed": observed,
                        "limit": limit,
                        "excess": excess,
                    }
                    for code, observed, limit, excess
                    in exc.violation_evidence
                ]
                sealed_next_track_id = exc.latched_next_track_id
                if (
                    type(sealed_next_track_id) is not str
                    or not sealed_next_track_id
                ):
                    raise abort_type(
                        "visual-course passage safety refusal lacks its "
                        "sealed next identity"
                    ) from exc
                refusal_history = getattr(
                    getattr(snapshot, "current_track", None),
                    "history",
                    None,
                )
                refusal_sample = (
                    refusal_history[-1]
                    if type(refusal_history) is tuple
                    and bool(refusal_history)
                    else None
                )
                refusal_observation_ns = getattr(
                    refusal_sample,
                    "observation_monotonic_ns",
                    None,
                )
                if (
                    getattr(refusal_sample, "token", None) != token
                    or type(refusal_observation_ns) is not int
                    or refusal_observation_ns < 0
                    or round(
                        exc.camera_observation_monotonic_s
                        * 1_000_000_000
                    )
                    != refusal_observation_ns
                ):
                    raise abort_type(
                        "visual-course passage safety refusal lacks exact "
                        "camera observation provenance"
                    ) from exc
                requalification_eligible = bool(
                    not next_preview_requalification_used
                    and next_preview_requalification is None
                    and exc.violation_codes
                    == _REQUALIFIABLE_APPROACH_PREVIEW_VIOLATIONS
                )
                if requalification_eligible:
                    refusal_control_perf_counter_ns = (
                        runtime.perf_counter_ns()
                    )
                    if (
                        type(refusal_control_perf_counter_ns) is not int
                        or refusal_control_perf_counter_ns < 0
                    ):
                        raise abort_type(
                            "visual-course preview requalification wire "
                            "deadline clock is invalid"
                        )
                    wire_start_deadline_monotonic_ns = (
                        refusal_control_perf_counter_ns
                        + round(
                            MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                            * 1_000_000_000
                        )
                    )
                    planner = make_planner(
                        next_gate_blend=(
                            host.visual_config.lifecycle
                            .next_gate_blend_max
                        ),
                        required_next_track_id=sealed_next_track_id,
                    )
                    try:
                        proposal = planner.observe(
                            snapshot,
                            host.visual_tracker,
                            (
                                runtime.perf_counter_ns()
                                / 1_000_000_000.0
                            ),
                            now - segment_started_s,
                            excursion,
                            mode=VisualApproachMode.APPROACH,
                            passage_admission=None,
                        )
                    except VisualApproachRefusal as requalify_exc:
                        raise abort_type(
                            "visual-course sealed preview requalification "
                            "could not consume its refusal publication: "
                            f"{requalify_exc}"
                        ) from requalify_exc
                    if (
                        proposal.mode is not VisualApproachMode.APPROACH
                        or proposal.servo_output.advance_enabled
                        or proposal.servo_output.next_gate_blend != 0.0
                        or proposal.passage_admission is not None
                        or getattr(
                            proposal,
                            "latched_next_track_id",
                            None,
                        )
                        is not None
                    ):
                        raise abort_type(
                            "visual-course sealed preview requalification "
                            "retained authority on its refusal publication"
                        )
                    next_preview_requalification_used = True
                    next_preview_requalification = {
                        "basis": (
                            APPROACH_PREVIEW_REQUALIFICATION_BASIS
                        ),
                        "outcome": "pending",
                        "sealed_next_track_id": sealed_next_track_id,
                        "refusal_camera_token": asdict(token),
                        "refusal_tracker_frame_sequence": (
                            snapshot.tracker_frame_sequence
                        ),
                        "refusal_observation_monotonic_s": (
                            exc.camera_observation_monotonic_s
                        ),
                        "refusal_observation_monotonic_ns": (
                            refusal_observation_ns
                        ),
                        "refusal_control_monotonic_s": now,
                        "refusal_control_perf_counter_ns": (
                            refusal_control_perf_counter_ns
                        ),
                        "refusal_violation_codes": list(
                            exc.violation_codes
                        ),
                        "refusal_violation_evidence": (
                            violation_evidence
                        ),
                        "fresh_frame_count": 0,
                        "latest_camera_token": asdict(token),
                        "latest_tracker_frame_sequence": (
                            snapshot.tracker_frame_sequence
                        ),
                        "publication_delta": 0,
                        "elapsed_s": 0.0,
                        "control_elapsed_s": 0.0,
                        "control_elapsed_ns": 0,
                        "max_fresh_frames": (
                            MAX_APPROACH_PREVIEW_REQUALIFICATION_FRESH_FRAMES
                        ),
                        "max_publication_delta": (
                            MAX_APPROACH_PREVIEW_REQUALIFICATION_PUBLICATION_DELTA
                        ),
                        "max_duration_s": (
                            MAX_APPROACH_PREVIEW_REQUALIFICATION_DURATION_S
                        ),
                        "max_control_duration_s": (
                            MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                        ),
                        "wire_start_deadline_monotonic_ns": (
                            wire_start_deadline_monotonic_ns
                        ),
                        "requalified_camera_token": None,
                        "requalified_tracker_frame_sequence": None,
                        "requalified_preview_blend": None,
                        "retirement_reason": None,
                        "retirement_camera_token": None,
                    }
                    segment["next_preview_requalification_count"] = 1
                    segment["next_preview_requalification"] = (
                        next_preview_requalification
                    )
                    host.recorder.emit(
                        "visual_course_next_preview_requalification_started",
                        gate_index=current_gate_index,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/approach"
                        ),
                        **next_preview_requalification,
                    )
                else:
                    required_next_track_id = (
                        next_preview_requalification[
                            "sealed_next_track_id"
                        ]
                        if next_preview_requalification is not None
                        else (
                            sealed_next_track_id
                        )
                    )
                    planner, proposal = current_only_replan(
                        snapshot,
                        now=now,
                        excursion=excursion,
                        required_next_track_id=(
                            required_next_track_id
                        ),
                    )
                    retirement_reason = (
                        "preview_requalification_safety_violation"
                        if next_preview_requalification is not None
                        else "current_passage_safety_violation"
                    )
                    record_preview_retirement(
                        reason=retirement_reason,
                        token=token,
                        tracker_frame_sequence=(
                            snapshot.tracker_frame_sequence
                        ),
                        violation_codes=list(exc.violation_codes),
                        violation_evidence=violation_evidence,
                        transient_eligible=exc.transient_eligible,
                    )
            except (
                VisualApproachCurrentGeometryUnavailable,
                VisualApproachRefusal,
            ) as exc:
                track = getattr(snapshot, "current_track", None)
                previous_visible_token = (
                    censored_passage_coast_last_observed_token
                    or last_clean_passage_token
                )
                censored_coast_eligible = bool(
                    type(exc)
                    is VisualApproachCurrentGeometryUnavailable
                    and mode is VisualApproachMode.PASSAGE
                    and type(passage_admission)
                    is VisualApproachPassageAdmission
                    and crossing_anchor is not None
                    and crossing_coast_authority is not None
                    and previous_visible_token is not None
                    and _censored_passage_coast_eligible(
                        snapshot,
                        current_gate_index=current_gate_index,
                        current_track_id=current_track_id,
                        crossing_anchor_token=(
                            crossing_anchor["camera_token"]
                        ),
                        authority=crossing_coast_authority,
                        previous_visible_token=previous_visible_token,
                        previous_apparent_scale=(
                            censored_passage_coast_last_observed_scale
                            if (
                                censored_passage_coast_last_observed_scale
                                is not None
                            )
                            else last_clean_passage_scale
                        ),
                        minimum_apparent_scale=math.exp(
                            limits.crossing_arm_min_log_scale
                        ),
                    )
                )
                bottom_transition_eligible = bool(
                    type(exc)
                    is VisualApproachCurrentGeometryUnavailable
                    and mode is VisualApproachMode.PASSAGE
                    and type(passage_admission)
                    is VisualApproachPassageAdmission
                    and crossing_anchor is not None
                    and crossing_coast_authority is not None
                    and censored_passage_coast_started_s is None
                    and not bottom_censored_transition_used
                    and bottom_censored_transition_pending_token is None
                    and previous_visible_token is not None
                    and _bottom_censored_passage_transition_eligible(
                        snapshot,
                        current_gate_index=current_gate_index,
                        current_track_id=current_track_id,
                        crossing_anchor_token=(
                            crossing_anchor["camera_token"]
                        ),
                        authority=crossing_coast_authority,
                        previous_visible_token=previous_visible_token,
                        previous_apparent_scale=last_clean_passage_scale,
                        minimum_apparent_scale=math.exp(
                            limits.crossing_arm_min_log_scale
                        ),
                        passage_admission=passage_admission,
                        next_preview_retired=next_preview_retired,
                        tuning=host.visual_config.servo,
                    )
                )
                if bottom_transition_eligible:
                    bottom_censored_transition_used = True
                    bottom_censored_transition_pending_token = token
                    censored_passage_coast_started_s = now
                    censored_passage_coast_last_observed_token = token
                    censored_passage_coast_last_observed_scale = float(
                        track.apparent_scale
                    )
                    censored_passage_coast_fresh_frame_count = 1
                    transition_evidence = {
                        "basis": (
                            CENSORED_PASSAGE_BOTTOM_TRANSITION_BASIS
                        ),
                        "outcome": "awaiting_full_vertical_censorship",
                        "anchor_camera_token": asdict(
                            crossing_coast_authority.anchor_camera_token
                        ),
                        "previous_clean_camera_token": asdict(
                            previous_visible_token
                        ),
                        "bottom_censored_camera_token": asdict(token),
                        "full_vertical_censor_camera_token": None,
                        "previous_apparent_scale": (
                            last_clean_passage_scale
                        ),
                        "bottom_censored_apparent_scale": float(
                            track.apparent_scale
                        ),
                    }
                    segment["bottom_censored_transition_count"] = 1
                    segment["bottom_censored_transition"] = (
                        transition_evidence
                    )
                    segment["censored_passage_coast"] = {
                        "basis": (
                            CENSORED_PASSAGE_BOTTOM_TRANSITION_BASIS
                        ),
                        "anchor_camera_token": asdict(
                            crossing_coast_authority.anchor_camera_token
                        ),
                        "first_censored_camera_token": asdict(token),
                        "last_censored_camera_token": asdict(token),
                        "loss_camera_token": None,
                        "target_roll_rad": (
                            crossing_coast_authority.target_roll_rad
                        ),
                        "target_pitch_rad": (
                            crossing_coast_authority.target_pitch_rad
                        ),
                        "thrust": crossing_coast_authority.thrust,
                        "max_duration_s": (
                            limits.censored_passage_coast_max_duration_s
                        ),
                        "max_fresh_frames": (
                            limits
                            .censored_passage_coast_max_fresh_frames
                        ),
                        "elapsed_s": 0.0,
                        "bottom_transition_pending": True,
                    }
                    segment[
                        "censored_passage_coast_fresh_frame_count"
                    ] = censored_passage_coast_fresh_frame_count
                    last_planned_token = token
                    host.recorder.emit(
                        "visual_course_bottom_censored_transition_started",
                        gate_index=current_gate_index,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/censored-passage"
                        ),
                        **transition_evidence,
                    )
                    refresh_live_summary()
                    try:
                        coast_command = (
                            await send_censored_passage_coast(
                                snapshot=snapshot,
                                authority=crossing_coast_authority,
                                yaw_reference_rad=yaw_reference_rad,
                                segment_started_s=segment_started_s,
                                stage=(
                                    f"{VISUAL_COURSE_STAGE}/gate"
                                    f"{current_gate_index}/"
                                    "censored-passage"
                                ),
                            )
                        )
                    except RaceActiveBoundaryChangedBeforeWire as race_exc:
                        credited_race = accept_no_wire_race_boundary(
                            race_exc
                        )
                        break
                    if coast_command is None:
                        continue
                    censored_passage_coast_command_count += 1
                    passage_command_count += 1
                    segment["passage_command_count"] = (
                        passage_command_count
                    )
                    segment[
                        "censored_passage_coast_command_count"
                    ] = censored_passage_coast_command_count
                    continue
                if censored_coast_eligible:
                    if bottom_censored_transition_pending_token is not None:
                        bottom_censored_transition_pending_token = None
                        transition_evidence = segment[
                            "bottom_censored_transition"
                        ]
                        if not isinstance(transition_evidence, dict):
                            raise abort_type(
                                "visual-course bottom-censored transition "
                                "lost its evidence"
                            )
                        transition_evidence["outcome"] = (
                            "full_vertical_censorship_confirmed"
                        )
                        transition_evidence[
                            "full_vertical_censor_camera_token"
                        ] = asdict(token)
                        coast_evidence = segment[
                            "censored_passage_coast"
                        ]
                        if not isinstance(coast_evidence, dict):
                            raise abort_type(
                                "visual-course bottom-censored transition "
                                "lost its coast evidence"
                            )
                        coast_evidence["bottom_transition_pending"] = False
                        host.recorder.emit(
                            "visual_course_bottom_censored_transition_confirmed",
                            gate_index=current_gate_index,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/censored-passage"
                            ),
                            **transition_evidence,
                        )
                    if censored_passage_coast_started_s is None:
                        censored_passage_coast_started_s = now
                        segment["censored_passage_coast"] = {
                            "basis": CENSORED_PASSAGE_COAST_BASIS,
                            "anchor_camera_token": asdict(
                                crossing_coast_authority
                                .anchor_camera_token
                            ),
                            "first_censored_camera_token": asdict(token),
                            "last_censored_camera_token": None,
                            "loss_camera_token": None,
                            "target_roll_rad": (
                                crossing_coast_authority.target_roll_rad
                            ),
                            "target_pitch_rad": (
                                crossing_coast_authority.target_pitch_rad
                            ),
                            "thrust": crossing_coast_authority.thrust,
                            "max_duration_s": (
                                limits
                                .censored_passage_coast_max_duration_s
                            ),
                            "max_fresh_frames": (
                                limits
                                .censored_passage_coast_max_fresh_frames
                            ),
                            "elapsed_s": 0.0,
                        }
                        host.recorder.emit(
                            "visual_course_censored_passage_coast_started",
                            gate_index=current_gate_index,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/censored-passage"
                            ),
                            **segment["censored_passage_coast"],
                        )
                    coast_elapsed_s = (
                        now - censored_passage_coast_started_s
                    )
                    if (
                        coast_elapsed_s
                        >= limits.censored_passage_coast_max_duration_s
                        or censored_passage_coast_fresh_frame_count
                        >= (
                            limits
                            .censored_passage_coast_max_fresh_frames
                        )
                    ):
                        raise abort_type(
                            "visual-course censored passage coast expired"
                        ) from exc
                    censored_passage_coast_last_observed_token = token
                    censored_passage_coast_last_observed_scale = float(
                        track.apparent_scale
                    )
                    censored_passage_coast_fresh_frame_count += 1
                    segment[
                        "censored_passage_coast_fresh_frame_count"
                    ] = censored_passage_coast_fresh_frame_count
                    segment["censored_passage_coast"].update(
                        {
                            "last_censored_camera_token": asdict(token),
                            "elapsed_s": coast_elapsed_s,
                        }
                    )
                    last_planned_token = token
                    try:
                        coast_command = (
                            await send_censored_passage_coast(
                                snapshot=snapshot,
                                authority=crossing_coast_authority,
                                yaw_reference_rad=yaw_reference_rad,
                                segment_started_s=segment_started_s,
                                stage=(
                                    f"{VISUAL_COURSE_STAGE}/gate"
                                    f"{current_gate_index}/"
                                    "censored-passage"
                                ),
                            )
                        )
                    except RaceActiveBoundaryChangedBeforeWire as race_exc:
                        credited_race = accept_no_wire_race_boundary(
                            race_exc
                        )
                        break
                    if coast_command is None:
                        continue
                    censored_passage_coast_command_count += 1
                    passage_command_count += 1
                    segment["passage_command_count"] = (
                        passage_command_count
                    )
                    segment[
                        "censored_passage_coast_command_count"
                    ] = censored_passage_coast_command_count
                    continue

                if bottom_censored_transition_pending_token is not None:
                    raise abort_type(
                        "visual-course bottom-censored transition lacked "
                        "full vertical censorship on its exact successor"
                    ) from exc
                credible_loss = bool(
                    mode is VisualApproachMode.PASSAGE
                    and crossing_anchor is not None
                    and previous_visible_token is not None
                    and getattr(snapshot, "current_gate_index", None)
                    == current_gate_index
                    and getattr(snapshot, "current_track_id", None)
                    == current_track_id
                    and track is not None
                    and getattr(track, "track_id", None)
                    == current_track_id
                    and getattr(track, "latest_token", None)
                    == previous_visible_token
                    and getattr(track, "role", None)
                    is VisualTrackRole.CURRENT
                    and getattr(track, "ambiguous", True) is False
                    and getattr(track, "visible", True) is False
                    and type(
                        getattr(track, "missed_frame_count", None)
                    )
                    is int
                    and getattr(track, "missed_frame_count", 0) > 0
                    and _token_strictly_newer(
                        token,
                        previous_visible_token,
                    )
                    and token.publication_sequence
                    - previous_visible_token.publication_sequence
                    == track.missed_frame_count
                    and type(getattr(track, "history", None)) is tuple
                    and bool(track.history)
                    and getattr(track.history[-1], "token", None)
                    == previous_visible_token
                )
                if not credible_loss:
                    raise abort_type(
                        "visual-course visual authority refused: "
                        f"{exc}"
                    ) from exc
                crossing_started_s = now
                if crossing_baseline_race is None:
                    raise abort_type(
                        "visual-course crossing lacks a race baseline"
                    )
                if segment["censored_passage_coast"] is not None:
                    segment["censored_passage_coast"][
                        "loss_camera_token"
                    ] = asdict(token)
                break

            if censored_passage_coast_started_s is not None:
                raise abort_type(
                    "visual-course censored passage coast returned to "
                    "uncensored geometry"
                )
            if next_preview_requalification is not None:
                requalification = next_preview_requalification
                sealed_next_track_id = requalification[
                    "sealed_next_track_id"
                ]
                refusal_token = requalification[
                    "refusal_camera_token"
                ]
                same_refusal_publication = bool(
                    token.stream_id == refusal_token["stream_id"]
                    and token.generation == refusal_token["generation"]
                    and token.frame_id == refusal_token["frame_id"]
                    and token.publication_sequence
                    == refusal_token["publication_sequence"]
                )
                if not same_refusal_publication:
                    if (
                        token.stream_id != refusal_token["stream_id"]
                        or token.generation != refusal_token["generation"]
                        or token.publication_sequence
                        <= refusal_token["publication_sequence"]
                        or snapshot.tracker_frame_sequence
                        <= requalification[
                            "refusal_tracker_frame_sequence"
                        ]
                    ):
                        raise abort_type(
                            "visual-course preview requalification crossed "
                            "or replayed its exact camera epoch"
                        )
                    requalification["fresh_frame_count"] = (
                        int(requalification["fresh_frame_count"]) + 1
                    )
                observation_ns = (
                    _current_target_observation_monotonic_ns(
                        snapshot,
                        proposal.current_target,
                        abort_type=abort_type,
                    )
                )
                requalification_elapsed_ns = (
                    observation_ns
                    - int(
                        requalification[
                            "refusal_observation_monotonic_ns"
                        ]
                    )
                )
                requalification_elapsed_s = (
                    requalification_elapsed_ns / 1_000_000_000
                )
                publication_delta = (
                    token.publication_sequence
                    - refusal_token["publication_sequence"]
                )
                if (
                    requalification_elapsed_ns < 0
                    or publication_delta < 0
                ):
                    raise abort_type(
                        "visual-course preview requalification timing "
                        "regressed"
                    )
                requalification.update(
                    {
                        "latest_camera_token": asdict(token),
                        "latest_tracker_frame_sequence": (
                            snapshot.tracker_frame_sequence
                        ),
                        "publication_delta": publication_delta,
                        "elapsed_s": requalification_elapsed_s,
                        "elapsed_ns": requalification_elapsed_ns,
                    }
                )

                proposal_latched_id = getattr(
                    proposal,
                    "latched_next_track_id",
                    None,
                )
                reviewed_next_id = getattr(
                    proposal.servo_output,
                    "reviewed_next_track_id",
                    None,
                )
                proposal_next_target = getattr(
                    proposal,
                    "next_target",
                    None,
                )
                proposal_next_id = getattr(
                    proposal_next_target,
                    "track_id",
                    None,
                )
                proposal_admission = proposal.passage_admission
                admission_preview_id = getattr(
                    proposal_admission,
                    "preview_track_id",
                    None,
                )
                if any(
                    identity not in {None, sealed_next_track_id}
                    for identity in (
                        proposal_latched_id,
                        reviewed_next_id,
                        proposal_next_id,
                        admission_preview_id,
                    )
                ):
                    raise abort_type(
                        "visual-course preview requalification changed its "
                        "sealed next identity"
                    )

                preview_requalification_wire_candidate = bool(
                    proposal.servo_output.next_gate_blend > 0.0
                    and proposal_latched_id == sealed_next_track_id
                    and reviewed_next_id == sealed_next_track_id
                    and proposal_next_id == sealed_next_track_id
                    and type(proposal_admission)
                    is VisualApproachPassageAdmission
                    and proposal_admission.preview_track_id
                    == sealed_next_track_id
                    and proposal_admission.preview_blend
                    == proposal.servo_output.next_gate_blend
                )
                if (
                    proposal.servo_output.next_gate_blend > 0.0
                    and not preview_requalification_wire_candidate
                ):
                    raise abort_type(
                        "visual-course preview requalification proposed "
                        "authority before ordinary same-identity admission"
                    )

                requalification_bounds_exhausted = bool(
                    int(requalification["fresh_frame_count"])
                    > MAX_APPROACH_PREVIEW_REQUALIFICATION_FRESH_FRAMES
                    or publication_delta
                    > (
                        MAX_APPROACH_PREVIEW_REQUALIFICATION_PUBLICATION_DELTA
                    )
                    or requalification_elapsed_ns
                    > round(
                        MAX_APPROACH_PREVIEW_REQUALIFICATION_DURATION_S
                        * 1_000_000_000
                    )
                )
                if requalification_bounds_exhausted:
                    planner, proposal = current_only_replan(
                        snapshot,
                        now=now,
                        excursion=excursion,
                        required_next_track_id=sealed_next_track_id,
                    )
                    bound_evidence = [
                        {
                            "code": "requalification_fresh_frames",
                            "observed": int(
                                requalification["fresh_frame_count"]
                            ),
                            "limit": (
                                MAX_APPROACH_PREVIEW_REQUALIFICATION_FRESH_FRAMES
                            ),
                            "excess": max(
                                0,
                                int(
                                    requalification[
                                        "fresh_frame_count"
                                    ]
                                )
                                - (
                                    MAX_APPROACH_PREVIEW_REQUALIFICATION_FRESH_FRAMES
                                ),
                            ),
                        },
                        {
                            "code": "requalification_publication_delta",
                            "observed": publication_delta,
                            "limit": (
                                MAX_APPROACH_PREVIEW_REQUALIFICATION_PUBLICATION_DELTA
                            ),
                            "excess": max(
                                0,
                                publication_delta
                                - (
                                    MAX_APPROACH_PREVIEW_REQUALIFICATION_PUBLICATION_DELTA
                                ),
                            ),
                        },
                        {
                            "code": "requalification_duration_s",
                            "observed": requalification_elapsed_s,
                            "limit": (
                                MAX_APPROACH_PREVIEW_REQUALIFICATION_DURATION_S
                            ),
                            "excess": max(
                                0.0,
                                requalification_elapsed_s
                                - (
                                    MAX_APPROACH_PREVIEW_REQUALIFICATION_DURATION_S
                                ),
                            ),
                        },
                    ]
                    record_preview_retirement(
                        reason="preview_requalification_bounds_exhausted",
                        token=token,
                        tracker_frame_sequence=(
                            snapshot.tracker_frame_sequence
                        ),
                        violation_codes=[
                            item["code"] for item in bound_evidence
                        ],
                        violation_evidence=bound_evidence,
                        transient_eligible=False,
                    )
                    preview_requalification_wire_candidate = False
            last_planned_token = token
            if (
                proposal.servo_output.passage_preview_retired
                and not segment["next_preview_retired"]
            ):
                if mode is not VisualApproachMode.PASSAGE:
                    raise abort_type(
                        "visual-course passage preview retired outside "
                        "passage mode"
                    )
                retirement_details = (
                    proposal.servo_output
                    .passage_preview_retirement_violations
                )
                if not retirement_details:
                    raise abort_type(
                        "visual-course passage preview retirement lacks "
                        "structured evidence"
                    )
                withdrawal = {
                    "reason": "passage_preview_envelope_retired",
                    "camera_token": asdict(token),
                    "tracker_frame_sequence": (
                        snapshot.tracker_frame_sequence
                    ),
                    "violation_codes": [
                        detail.violation.value
                        for detail in retirement_details
                    ],
                    "violation_evidence": [
                        {
                            "code": detail.violation.value,
                            "observed": detail.observed,
                            "limit": detail.limit,
                            "excess": detail.excess,
                        }
                        for detail in retirement_details
                    ],
                    "transient_eligible": False,
                }
                next_preview_retired = True
                segment["next_preview_withdrawal_count"] = int(
                    segment["next_preview_withdrawal_count"]
                ) + 1
                segment["next_preview_withdrawal"] = withdrawal
                segment["next_preview_retired"] = True
                host.recorder.emit(
                    "visual_course_next_preview_withdrawn",
                    gate_index=current_gate_index,
                    stage=(
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/passage"
                    ),
                    **withdrawal,
                )
                refresh_live_summary()
            if mode is VisualApproachMode.APPROACH:
                if proposal.servo_output.advance_enabled:
                    raise abort_type(
                        "visual-course approach acquired passage authority"
                    )
                try:
                    accepted = await send_visual(
                        proposal=proposal,
                        snapshot=snapshot,
                        yaw_reference_rad=yaw_reference_rad,
                        segment_started_s=segment_started_s,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/approach"
                        ),
                        preview_requalification_wire_deadline_ns=(
                            (
                                int(
                                    next_preview_requalification[
                                        "wire_start_deadline_monotonic_ns"
                                    ]
                                )
                            )
                            if preview_requalification_wire_candidate
                            and next_preview_requalification is not None
                            else None
                        ),
                    )
                except _PreviewRequalificationWireSlotUnavailable as exc:
                    if (
                        not preview_requalification_wire_candidate
                        or next_preview_requalification is None
                    ):
                        raise abort_type(
                            "visual-course preview wire deadline expired "
                            "without pending requalification authority"
                        ) from exc
                    sealed_next_track_id = next_preview_requalification[
                        "sealed_next_track_id"
                    ]
                    planner = make_planner(
                        next_gate_blend=0.0,
                        required_next_track_id=sealed_next_track_id,
                    )
                    record_preview_retirement(
                        reason=(
                            "preview_requalification_wire_deadline_expired"
                        ),
                        token=token,
                        tracker_frame_sequence=(
                            snapshot.tracker_frame_sequence
                        ),
                        violation_codes=[
                            "requalification_control_duration_s",
                        ],
                        violation_evidence=[
                            {
                                "code": (
                                    "requalification_control_duration_s"
                                ),
                                "observed": (
                                    (
                                        exc.checked_perf_counter_ns
                                        - int(
                                            next_preview_requalification[
                                                "refusal_control_perf_counter_ns"
                                            ]
                                        )
                                    )
                                    / 1_000_000_000
                                ),
                                "limit": (
                                    MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                                ),
                                "excess": max(
                                    0.0,
                                    (
                                        exc.checked_perf_counter_ns
                                        - int(
                                            next_preview_requalification[
                                                "refusal_control_perf_counter_ns"
                                            ]
                                        )
                                    )
                                    / 1_000_000_000
                                    - (
                                        MAX_APPROACH_PREVIEW_REQUALIFICATION_CONTROL_DURATION_S
                                    ),
                                ),
                            },
                        ],
                        transient_eligible=False,
                    )
                    # The launch collective and other per-observation proofs
                    # have already consumed this token while preparing the
                    # rejected candidate.  Wait for the next exact camera
                    # publication before planning sealed current-only
                    # authority.
                    continue
                except RaceActiveBoundaryChangedBeforeWire as exc:
                    credited_race = accept_no_wire_race_boundary(exc)
                    break
                if type(accepted) is _SupersededVisualProposal:
                    continue
                if type(accepted) is not _AcceptedVisualCommand:
                    raise abort_type(
                        "visual-course approach command outcome is invalid"
                    )
                approach_command_count += 1
                segment["approach_command_count"] = approach_command_count
                if preview_requalification_wire_candidate:
                    if next_preview_requalification is None:
                        raise abort_type(
                            "visual-course preview requalification candidate "
                            "lost its pending state"
                        )
                    next_preview_requalification.update(
                        {
                            "outcome": "requalified",
                            "requalified_camera_token": asdict(token),
                            "requalified_tracker_frame_sequence": (
                                snapshot.tracker_frame_sequence
                            ),
                            "requalified_preview_blend": (
                                proposal.servo_output.next_gate_blend
                            ),
                        }
                    )
                    host.recorder.emit(
                        "visual_course_next_preview_requalified",
                        gate_index=current_gate_index,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/approach"
                        ),
                        **next_preview_requalification,
                    )
                    next_preview_requalification = None
                if proposal.passage_admission is not None:
                    if next_preview_requalification is not None:
                        launch = segment["launch_bootstrap"]
                        launch["passage_admission_withheld_count"] = (
                            int(
                                launch.get(
                                    "passage_admission_withheld_count",
                                    0,
                                )
                            )
                            + 1
                        )
                        continue
                    if accepted.yaw_soft_stop_zeroed:
                        segment[
                            "passage_admission_yaw_soft_stop_withheld_count"
                        ] = int(
                            segment[
                                "passage_admission_yaw_soft_stop_withheld_count"
                            ]
                        ) + 1
                        continue
                    launch_ready = bool(
                        not segment["launch_bootstrap"]["enabled"]
                        or float(runtime.monotonic()) - course_started_s
                        >= float(
                            host.visual_config.lifecycle
                            .launch_pitch_blend_s
                        )
                    )
                    if launch_ready:
                        passage_admission = proposal.passage_admission
                        mode = VisualApproachMode.PASSAGE
                        passage_started_s = float(runtime.monotonic())
                        segment["passage_authority_enabled"] = True
                        segment["passage_admission"] = asdict(
                            passage_admission
                        )
                    else:
                        launch = segment["launch_bootstrap"]
                        launch["passage_admission_withheld_count"] = (
                            int(
                                launch.get(
                                    "passage_admission_withheld_count",
                                    0,
                                )
                            )
                            + 1
                        )
                continue

            if proposal.mode is not VisualApproachMode.PASSAGE:
                raise abort_type("visual-course passage mode was not retained")
            try:
                accepted = await send_visual(
                    proposal=proposal,
                    snapshot=snapshot,
                    yaw_reference_rad=yaw_reference_rad,
                    segment_started_s=segment_started_s,
                    stage=(
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/passage"
                    ),
                )
            except RaceActiveBoundaryChangedBeforeWire as exc:
                credited_race = accept_no_wire_race_boundary(exc)
                break
            if type(accepted) is _SupersededVisualProposal:
                # A scale-jump proof is exact-successor-only.  Receiver
                # supersession may preserve the ordinary completed dwell
                # through its separate proof, but it can never carry this
                # sealed command candidate across an unaccepted publication.
                retained_crossing_candidate = None
                if (
                    retained_crossing_supersession is None
                    and _retained_crossing_supersession_usable(
                        proposal.current_target,
                        proposal.servo_output,
                        accepted,
                        last_accepted_camera_token=(
                            last_clean_passage_token
                        ),
                        current_track_id=current_track_id,
                        retained_crossing_dwell_frames=(
                            retained_crossing_dwell_frames
                        ),
                        tuning=host.visual_config.servo,
                        limits=limits,
                    )
                ):
                    retained_crossing_supersession = accepted
                    hold_evidence = {
                        "basis": (
                            "completed-retained-dwell-adjacent-"
                            "supersession-v1"
                        ),
                        "outcome": "awaiting_exact_receiver_successor",
                        "previous_accepted_camera_token": asdict(
                            last_clean_passage_token
                        ),
                        "superseded_camera_token": asdict(
                            accepted.expected_camera_token
                        ),
                        "receiver_camera_token": asdict(
                            accepted.receiver_camera_token
                        ),
                        "held_previous_command_s": (
                            accepted.held_previous_command_s
                        ),
                        "retained_crossing_dwell_frames": (
                            retained_crossing_dwell_frames
                        ),
                        "track_id": proposal.current_target.track_id,
                        "normalized_x": (
                            proposal.current_target.normalized_x
                        ),
                        "normalized_y_down": (
                            proposal.current_target.normalized_y_down
                        ),
                        "normalized_x_rate_s": (
                            proposal.current_target.normalized_x_rate_s
                        ),
                        "normalized_y_rate_down_s": (
                            proposal.current_target
                            .normalized_y_rate_down_s
                        ),
                        "log_scale": proposal.current_target.log_scale,
                        "log_scale_rate_s": (
                            proposal.current_target.log_scale_rate_s
                        ),
                    }
                    segment[
                        "retained_crossing_supersession_hold_count"
                    ] = int(
                        segment[
                            "retained_crossing_supersession_hold_count"
                        ]
                    ) + 1
                    segment[
                        "retained_crossing_supersession_hold"
                    ] = hold_evidence
                    host.recorder.emit(
                        "visual_course_retained_crossing_supersession_held",
                        gate_index=current_gate_index,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/passage"
                        ),
                        **hold_evidence,
                    )
                else:
                    retained_crossing_supersession = None
                    retained_crossing_dwell_frames = 0
                segment["retained_crossing_dwell_frames"] = (
                    retained_crossing_dwell_frames
                )
                refresh_live_summary()
                continue
            if type(accepted) is not _AcceptedVisualCommand:
                raise abort_type(
                    "visual-course passage command outcome is invalid"
                )
            adjacent_predecessor = retained_crossing_candidate
            # Consume the opportunity before inspecting this publication.
            # A newly eligible current command may seal a fresh candidate for
            # its own exact successor below.
            retained_crossing_candidate = None
            if retained_crossing_supersession is not None:
                exact_successor = bool(
                    token
                    == retained_crossing_supersession.receiver_camera_token
                )
                hold_evidence = segment[
                    "retained_crossing_supersession_hold"
                ]
                if not exact_successor:
                    retained_crossing_dwell_frames = 0
                if isinstance(hold_evidence, dict):
                    hold_evidence["outcome"] = (
                        "exact_receiver_successor_accepted"
                        if exact_successor
                        else "receiver_successor_changed"
                    )
                    hold_evidence["accepted_successor_camera_token"] = (
                        asdict(token)
                    )
                retained_crossing_supersession = None
            command = accepted.command
            target = proposal.current_target
            last_clean_passage_token = token
            last_clean_passage_scale = math.exp(float(target.log_scale))
            passage_command_count += 1
            segment["passage_command_count"] = passage_command_count
            if proposal.servo_output.next_gate_blend > 0.0:
                passage_next_preview_command_count += 1
                segment[
                    "passage_next_preview_command_count"
                ] = passage_next_preview_command_count
            # The calibrated yaw limiter owns only the yaw channel.  Keep the
            # already sealed, same-identity preview alive so its independently
            # bounded pitch/collective corrections remain fresh.  This frame
            # still cannot count as advance or arm current-advance crossing
            # below, and every later publication must independently re-pass
            # the yaw limiter.
            refresh_live_summary()
            if (
                proposal.servo_output.advance_enabled
                and not accepted.yaw_soft_stop_zeroed
            ):
                advance_command_count += 1
                segment["advance_command_count"] = advance_command_count
            target_observation_monotonic_ns = (
                accepted.observation_monotonic_ns
            )
            retained_wire_projection: Optional[
                _RetainedCrossingWireProjection
            ] = None
            retained_observation_usable = (
                _retained_crossing_observation_usable(
                    target,
                    proposal.servo_output,
                    tuning=host.visual_config.servo,
                    limits=limits,
                    yaw_soft_stop_zeroed=(
                        accepted.yaw_soft_stop_zeroed
                    ),
                )
            )
            if retained_observation_usable:
                retained_crossing_dwell_frames += 1
                if (
                    ADJACENT_SCALE_JUMP_MIN_PREDECESSOR_LOG_SCALE
                    <= float(target.log_scale)
                    < limits.crossing_arm_min_log_scale
                ):
                    retained_wire_projection = (
                        _retained_crossing_wire_projection(
                            target,
                            observation_monotonic_ns=(
                                target_observation_monotonic_ns
                            ),
                            wire_start_monotonic_ns=(
                                accepted.wire_start_monotonic_ns
                            ),
                            tuning=host.visual_config.servo,
                            limits=limits,
                            abort_type=abort_type,
                        )
                    )
            else:
                retained_crossing_dwell_frames = 0
            if (
                retained_observation_usable
                and retained_crossing_dwell_frames
                >= host.visual_config.servo.required_corridor_frames
                and retained_wire_projection is not None
            ):
                retained_crossing_candidate = _RetainedCrossingCandidate(
                    gate_index=current_gate_index,
                    track_id=current_track_id,
                    camera_token=token,
                    tracker_frame_sequence=(
                        snapshot.tracker_frame_sequence
                    ),
                    target=target,
                    output=proposal.servo_output,
                    accepted=accepted,
                    wire_projection=retained_wire_projection,
                    retained_crossing_dwell_frames=(
                        retained_crossing_dwell_frames
                    ),
                    advance_command_count=advance_command_count,
                )
            segment["retained_crossing_dwell_frames"] = (
                retained_crossing_dwell_frames
            )
            segment["max_retained_crossing_dwell_frames"] = max(
                int(segment["max_retained_crossing_dwell_frames"]),
                retained_crossing_dwell_frames,
            )
            crossing_basis = _crossing_anchor_basis(
                target,
                proposal.servo_output,
                passage_admission=passage_admission,
                current_gate_index=current_gate_index,
                current_track_id=current_track_id,
                advance_command_count=advance_command_count,
                retained_crossing_dwell_frames=(
                    retained_crossing_dwell_frames
                ),
                tuning=host.visual_config.servo,
                limits=limits,
                retained_wire_projection=retained_wire_projection,
                yaw_soft_stop_zeroed=accepted.yaw_soft_stop_zeroed,
            )
            adjacent_scale_jump_proof: Optional[
                _AdjacentScaleJumpCrossingProof
            ] = None
            if crossing_basis is None:
                adjacent_scale_jump_proof = (
                    _adjacent_scale_jump_crossing_proof(
                        snapshot,
                        target,
                        proposal.servo_output,
                        accepted,
                        predecessor=adjacent_predecessor,
                        passage_admission=passage_admission,
                        current_gate_index=current_gate_index,
                        current_track_id=current_track_id,
                        advance_command_count=advance_command_count,
                        next_preview_retired=next_preview_retired,
                        tuning=host.visual_config.servo,
                        limits=limits,
                        abort_type=abort_type,
                    )
                )
                if adjacent_scale_jump_proof is not None:
                    crossing_basis = (
                        RETAINED_ADJACENT_SCALE_JUMP_CROSSING_BASIS
                    )
                    retained_wire_projection = (
                        adjacent_scale_jump_proof
                        .current_wire_projection
                    )
            if crossing_basis is not None:
                anchor_retained_crossing_dwell_frames = (
                    retained_crossing_dwell_frames
                    if adjacent_scale_jump_proof is None
                    else (
                        adjacent_scale_jump_proof.predecessor
                        .retained_crossing_dwell_frames
                    )
                )
                adjacent_scale_jump_evidence = None
                if adjacent_scale_jump_proof is not None:
                    proof = adjacent_scale_jump_proof
                    predecessor = proof.predecessor
                    association = proof.association
                    adjacent_scale_jump_evidence = {
                        "basis": (
                            RETAINED_ADJACENT_SCALE_JUMP_CROSSING_BASIS
                        ),
                        "predecessor_camera_token": asdict(
                            predecessor.camera_token
                        ),
                        "predecessor_tracker_frame_sequence": (
                            predecessor.tracker_frame_sequence
                        ),
                        "predecessor_retained_crossing_dwell_frames": (
                            predecessor.retained_crossing_dwell_frames
                        ),
                        "predecessor_log_scale": (
                            predecessor.target.log_scale
                        ),
                        "predecessor_normalized_x": (
                            predecessor.target.normalized_x
                        ),
                        "predecessor_normalized_y_down": (
                            predecessor.target.normalized_y_down
                        ),
                        "predecessor_normalized_x_rate_s": (
                            predecessor.target.normalized_x_rate_s
                        ),
                        "predecessor_normalized_y_rate_down_s": (
                            predecessor.target.normalized_y_rate_down_s
                        ),
                        "predecessor_log_scale_rate_s": (
                            predecessor.target.log_scale_rate_s
                        ),
                        "predecessor_wire_projected_log_scale": (
                            predecessor.wire_projection
                            .projected_log_scale
                        ),
                        "predecessor_to_current_observation_gap_ns": (
                            proof.observation_gap_ns
                        ),
                        "predecessor_to_current_publication_gap_ns": (
                            proof.publication_gap_ns
                        ),
                        "predecessor_projected_log_scale_at_current": (
                            proof.predecessor_projected_log_scale
                        ),
                        "predecessor_projected_normalized_x_at_current": (
                            proof
                            .predecessor_projected_normalized_x
                        ),
                        "predecessor_projected_normalized_y_down_at_current": (
                            proof
                            .predecessor_projected_normalized_y_down
                        ),
                        "current_log_scale_delta": (
                            proof.log_scale_delta
                        ),
                        "association_confidence": (
                            association.confidence
                        ),
                        "association_cost": association.cost,
                        "association_bbox_iou": association.bbox_iou,
                        "association_predicted_center_residual_norm": (
                            association.predicted_center_residual_norm
                        ),
                        "association_log_width_change": (
                            association.log_width_change
                        ),
                        "association_log_height_change": (
                            association.log_height_change
                        ),
                        "association_log_area_residual": (
                            association.log_area_residual
                        ),
                    }
                    segment[
                        "adjacent_scale_jump_crossing_count"
                    ] = int(
                        segment[
                            "adjacent_scale_jump_crossing_count"
                        ]
                    ) + 1
                crossing_coast_thrust = (
                    float(command.thrust)
                    - accepted.next_preview_collective_delta
                )
                if (
                    not math.isfinite(crossing_coast_thrust)
                    or crossing_coast_thrust
                    < limits.min_thrust - 1e-12
                    or crossing_coast_thrust
                    > limits.max_thrust + 1e-12
                ):
                    raise abort_type(
                        "visual-course current-only crossing coast thrust "
                        "escaped its fixed envelope"
                    )
                crossing_anchor = {
                    "basis": crossing_basis,
                    "camera_token": token,
                    "tracker_frame_sequence": (
                        snapshot.tracker_frame_sequence
                    ),
                    "track_id": target.track_id,
                    "log_scale": target.log_scale,
                    "observation_log_scale": target.log_scale,
                    "log_scale_rate_s": target.log_scale_rate_s,
                    "observation_monotonic_ns": (
                        target_observation_monotonic_ns
                    ),
                    "wire_start_monotonic_ns": (
                        accepted.wire_start_monotonic_ns
                    ),
                    "observation_to_wire_s": (
                        None
                        if retained_wire_projection is None
                        else (
                            retained_wire_projection
                            .observation_to_wire_s
                        )
                    ),
                    "observation_to_wire_ns": (
                        None
                        if retained_wire_projection is None
                        else (
                            retained_wire_projection
                            .observation_to_wire_ns
                        )
                    ),
                    "wire_projected_log_scale": (
                        None
                        if retained_wire_projection is None
                        else (
                            retained_wire_projection.projected_log_scale
                        )
                    ),
                    "wire_projected_normalized_x": (
                        None
                        if retained_wire_projection is None
                        else (
                            retained_wire_projection
                            .projected_normalized_x
                        )
                    ),
                    "wire_projected_normalized_y_down": (
                        None
                        if retained_wire_projection is None
                        else (
                            retained_wire_projection
                            .projected_normalized_y_down
                        )
                    ),
                    "corridor_frames": (
                        proposal.servo_output.corridor_frames
                    ),
                    "retained_crossing_dwell_frames": (
                        anchor_retained_crossing_dwell_frames
                    ),
                    "advance_command_count": advance_command_count,
                    "current_advance_enabled": (
                        proposal.servo_output.advance_enabled
                    ),
                    "yaw_soft_stop_zeroed": (
                        accepted.yaw_soft_stop_zeroed
                    ),
                    "normalized_x": target.normalized_x,
                    "normalized_y_down": target.normalized_y_down,
                    "normalized_x_rate_s": target.normalized_x_rate_s,
                    "normalized_y_rate_down_s": (
                        target.normalized_y_rate_down_s
                    ),
                    "command": asdict(command),
                    "next_preview_collective_delta": (
                        accepted.next_preview_collective_delta
                    ),
                    "current_only_crossing_coast_thrust": (
                        crossing_coast_thrust
                    ),
                    "adjacent_scale_jump_evidence": (
                        adjacent_scale_jump_evidence
                    ),
                }
                crossing_coast_authority = (
                    _CensoredPassageCoastAuthority(
                        gate_index=current_gate_index,
                        track_id=current_track_id,
                        anchor_camera_token=token,
                        target_roll_rad=accepted.target_roll_rad,
                        target_pitch_rad=accepted.target_pitch_rad,
                        # Optional next-preview authority retires with the
                        # live preview.  The existing censored-passage coast
                        # remains current-aperture-only and therefore latches
                        # the proved base rather than reintroducing a retired
                        # preview delta after a visibility gap.
                        thrust=crossing_coast_thrust,
                    )
                )
                segment["crossing_anchor"] = {
                    **crossing_anchor,
                    "camera_token": asdict(token),
                }

        if crossing_started_s is None:
            crossing_started_s = float(runtime.monotonic())
        if crossing_baseline_race is None:
            crossing_baseline_race = last_race
        crossing_deadline_s = (
            crossing_started_s + limits.crossing_status_timeout_s
        )
        while credited_race is None:
            now = await pace_tick()
            if now >= min(course_deadline_s, crossing_deadline_s):
                raise abort_type(
                    f"visual-course gate {current_gate_index} credit timed out"
                )
            host._sample()
            pad_contact = initial_pad_contact_authority()
            host._watchdog(
                require_target=False,
                allow_benign_pad_contact=pad_contact,
                enforce_benign_pad_budget=True,
            )
            _assert_course_attitude_state(
                host,
                yaw_reference_rad=yaw_reference_rad,
                limits=limits,
                yaw_profile=runtime.yaw_profile,
                abort_type=abort_type,
                phase=f"gate {current_gate_index} crossing wait",
            )
            race = host._visual_race_status_ref()
            relation = _race_relation(
                race,
                crossing_baseline_race,
                abort_type,
            )
            if relation < 0:
                raise abort_type("visual-course crossing race ingress regressed")
            if relation > 0:
                if (
                    race.race_finished
                    or race.active_gate_index == current_gate_index + 1
                ):
                    credited_race = race
                    last_race = race
                    break
                if race.active_gate_index == current_gate_index:
                    # Race status is slower than the control loop.  A newer
                    # same-gate ingress is an authoritative pending-credit
                    # heartbeat, not a refusal.  Advance the exact baseline
                    # and retain zero authority until credit or timeout.
                    crossing_baseline_race = race
                    last_race = race
                else:
                    raise abort_type(
                        "visual-course crossing produced an invalid gate index"
                    )
            await send_zero(
                (
                    f"{VISUAL_COURSE_STAGE}/gate"
                    f"{current_gate_index}/crossing-zero"
                ),
                now - segment_started_s,
                yaw_reference_rad=yaw_reference_rad,
            )
            segment["crossing_wait_zero_command_count"] = int(
                segment["crossing_wait_zero_command_count"]
            ) + 1

        assert credited_race is not None
        latest_authoritative_gate_index = int(
            credited_race.active_gate_index
        )
        max_gate_index = max(
            max_gate_index,
            latest_authoritative_gate_index,
        )
        refresh_live_summary()
        if credited_race.race_finished:
            _assert_course_attitude_state(
                host,
                yaw_reference_rad=yaw_reference_rad,
                limits=limits,
                yaw_profile=runtime.yaw_profile,
                abort_type=abort_type,
                phase=f"gate {current_gate_index} terminal acceptance",
            )
            finish_token = host._visual_camera_token_at_race_credit(
                credited_race
            )
            try:
                finished_snapshot = (
                    host.visual_gate_graph.confirm_race_finished(
                        host.visual_tracker,
                        race_status=credited_race,
                        camera_token_at_finish=finish_token,
                    )
                )
            except GateGraphError as exc:
                raise abort_type(
                    f"visual-course race-finish proof refused: {exc}"
                ) from exc
            if not getattr(finished_snapshot, "race_finished", False):
                raise abort_type("visual-course race finish did not latch")
            _assert_course_attitude_state(
                host,
                yaw_reference_rad=yaw_reference_rad,
                limits=limits,
                yaw_profile=runtime.yaw_profile,
                abort_type=abort_type,
                phase=f"gate {current_gate_index} terminal return",
            )
            segment["outcome"] = "race_finished"
            refresh_live_summary()
            summary = dict(host._visual_course_summary)
            summary.update(
                {
                    "stage": VISUAL_COURSE_STAGE,
                    "success": True,
                    "race_finished": True,
                    "outcome": "race_finished",
                    "first_causal_blocker": None,
                    "maximum_authoritative_gate_index": max_gate_index,
                    "final_gate_index": latest_authoritative_gate_index,
                }
            )
            host.recorder.emit("visual_course_complete", **summary)
            host._visual_course_summary = dict(summary)
            return summary

        transition_summary: Dict[str, Any] = {
            "from_gate_index": current_gate_index,
            "to_gate_index": int(credited_race.active_gate_index),
            "race_status_sequence": (
                credited_race.race_status_sequence
            ),
            "race_received_monotonic_ns": (
                credited_race.received_monotonic_ns
            ),
            "promotion_confirmed": False,
            "retired_track_id": current_track_id,
            "promoted_track_id": None,
            "pre_transition_navigation_command_count": (
                approach_command_count + passage_command_count
            ),
            "pre_transition_approach_command_count": (
                approach_command_count
            ),
            "pre_transition_passage_command_count": (
                passage_command_count
            ),
            "pre_transition_passage_next_preview_command_count": (
                passage_next_preview_command_count
            ),
            "crossing_wait_zero_command_count": int(
                segment["crossing_wait_zero_command_count"]
            ),
            "post_transition_zero_command_count": 0,
            "post_transition_navigation_command_count": 0,
            "passage_authority_enabled": bool(
                segment["passage_authority_enabled"]
            ),
            "history_length_before_promotion": None,
            "history_length_after_promotion": None,
        }
        transitions.append(transition_summary)
        segment["outcome"] = "authoritative_credit_observed"
        segment["transition"] = [
            current_gate_index,
            int(credited_race.active_gate_index),
        ]
        refresh_live_summary()

        if (
            type(passage_admission) is not VisualApproachPassageAdmission
            or type(passage_admission.preview_track_id) is not str
            or not passage_admission.preview_track_id
        ):
            raise abort_type(
                "visual-course nonterminal transition lacks its reviewed "
                "next-track identity"
            )
        requested_promoted_track_id = passage_admission.preview_track_id
        transition = host._confirm_visual_transition(
            from_gate_index=current_gate_index,
            to_gate_index=current_gate_index + 1,
            race_status=credited_race,
            promoted_track_id=requested_promoted_track_id,
        )
        if (
            requested_promoted_track_id is not None
            and transition.promoted_track_id
            != requested_promoted_track_id
        ):
            raise abort_type(
                "visual-course transition replaced its reviewed "
                "next-track identity"
            )
        if (
            transition.from_gate_index != current_gate_index
            or transition.to_gate_index != current_gate_index + 1
            or transition.retired_track_id != current_track_id
            or transition.promoted_track_id == current_track_id
            or transition.history_length_before_promotion
            != transition.history_length_after_promotion
        ):
            raise abort_type(
                "visual-course transition promotion is incomplete"
            )
        transition_summary.update(
            {
                "promotion_confirmed": True,
                "retired_track_id": transition.retired_track_id,
                "promoted_track_id": transition.promoted_track_id,
                "history_length_before_promotion": (
                transition.history_length_before_promotion
                ),
                "history_length_after_promotion": (
                transition.history_length_after_promotion
                ),
            }
        )
        segment["outcome"] = "transition_confirmed"

        current_gate_index = transition.to_gate_index
        current_track_id = transition.promoted_track_id
        max_gate_index = max(max_gate_index, current_gate_index)
        refresh_live_summary()
        fresh_deadline_s = min(
            course_deadline_s,
            float(runtime.monotonic())
            + limits.post_credit_fresh_frame_timeout_s,
        )
        recovery_started_ns = runtime.perf_counter_ns()
        if type(recovery_started_ns) is not int or recovery_started_ns < 0:
            raise abort_type(
                "visual-course recovery clock is invalid at transition"
            )
        recovery_admission: Any = None
        recovery_admission_kind: Optional[str] = None
        admitted_recovery_token: Optional[CameraFrameToken] = None
        latest_recovery_refusal: Optional[str] = None

        def evaluate_recovery_candidate(snapshot: Any) -> bool:
            nonlocal recovery_admission
            nonlocal recovery_admission_kind
            nonlocal admitted_recovery_token
            nonlocal latest_recovery_refusal

            recovery_admission = None
            recovery_admission_kind = None
            admitted_recovery_token = None
            if not _current_snapshot_ready(
                snapshot,
                gate_index=current_gate_index,
                track_id=current_track_id,
            ):
                latest_recovery_refusal = (
                    "promoted track is not currently visible and unambiguous"
                )
                return False
            token = getattr(snapshot, "latest_camera_token", None)
            if type(token) is not CameraFrameToken:
                raise abort_type(
                    "visual-course recovery snapshot lacks an exact token"
                )
            try:
                promoted_track = host.visual_tracker.track(
                    current_track_id
                )
                if getattr(promoted_track, "latest_token", None) != token:
                    raise abort_type(
                        "visual-course recovery track does not end at the "
                        "ready snapshot token"
                    )
                _roll, measured_pitch, _yaw, _rates = _attitude_state(
                    host,
                    abort_type,
                )
                now_ns = runtime.perf_counter_ns()
                if type(now_ns) is not int or now_ns < recovery_started_ns:
                    raise abort_type(
                        "visual-course recovery clock regressed"
                    )
                if (
                    token
                    == transition.promoted_latest_token_at_promotion
                ):
                    candidate = runtime.transition_recovery_admission(
                        promoted_track,
                        transition,
                        tracker_time_basis_id=(
                            host.visual_tracker.time_basis_id
                        ),
                        measured_pitch_rad=measured_pitch,
                        now_monotonic_ns=now_ns,
                    )
                    history_tokens = getattr(
                        candidate,
                        "history_tokens",
                        None,
                    )
                    transition_race = transition.race_status
                    if (
                        not _recovery_identity_matches_transition(
                            candidate,
                            transition,
                        )
                        or getattr(
                            candidate,
                            "promotion_anchor_token",
                            None,
                        )
                        != token
                        or type(history_tokens) is not tuple
                        or not history_tokens
                        or history_tokens[-1] != token
                        or getattr(
                            candidate,
                            "race_status_sequence",
                            None,
                        )
                        != transition_race.race_status_sequence
                        or getattr(
                            candidate,
                            "race_received_monotonic_ns",
                            None,
                        )
                        != transition_race.received_monotonic_ns
                    ):
                        raise abort_type(
                            "visual-course transition recovery admission is "
                            "not bound to its exact anchor"
                        )
                    kind = "transition_anchor"
                else:
                    candidate = runtime.recovery_continuation_admission(
                        promoted_track,
                        transition,
                        previous_token=(
                            transition.promoted_latest_token_at_promotion
                        ),
                        tracker_time_basis_id=(
                            host.visual_tracker.time_basis_id
                        ),
                        measured_pitch_rad=measured_pitch,
                        recovery_started_monotonic_ns=(
                            recovery_started_ns
                        ),
                        now_monotonic_ns=now_ns,
                    )
                    capture = getattr(candidate, "capture", None)
                    if (
                        not _recovery_identity_matches_transition(
                            candidate,
                            transition,
                        )
                        or getattr(candidate, "previous_token", None)
                        != transition.promoted_latest_token_at_promotion
                        or getattr(candidate, "frame_token", None) != token
                        or not _servo_token_matches_camera(
                            getattr(capture, "frame_token", None),
                            token,
                        )
                    ):
                        raise abort_type(
                            "visual-course recovery continuation is not "
                            "bound to its exact frame"
                        )
                    kind = "exact_next_continuation"
            except VisualRecoveryRefusal as exc:
                latest_recovery_refusal = str(exc)
                return False
            if not _token_strictly_newer(
                token,
                transition.camera_token_at_credit,
            ):
                latest_recovery_refusal = (
                    "recovery admission is not newer than race credit"
                )
                return False
            recovery_admission = candidate
            recovery_admission_kind = kind
            admitted_recovery_token = token
            return True

        while True:
            _assert_course_attitude_state(
                host,
                yaw_reference_rad=yaw_reference_rad,
                limits=limits,
                yaw_profile=runtime.yaw_profile,
                abort_type=abort_type,
                phase=f"gate {current_gate_index} recovery admission",
            )
            snapshot = host.visual_gate_graph.latest_snapshot
            if evaluate_recovery_candidate(snapshot):
                break
            now = await pace_tick()
            if now >= fresh_deadline_s:
                raise abort_type(
                    "visual-course promoted gate lacks completed recovery "
                    "admission and a fresh post-credit camera frame"
                    + (
                        ""
                        if latest_recovery_refusal is None
                        else f": {latest_recovery_refusal}"
                    )
                )
            host._sample()
            pad_contact = initial_pad_contact_authority()
            host._watchdog(
                require_target=False,
                allow_benign_pad_contact=pad_contact,
                enforce_benign_pad_budget=True,
            )
            _assert_course_attitude_state(
                host,
                yaw_reference_rad=yaw_reference_rad,
                limits=limits,
                yaw_profile=runtime.yaw_profile,
                abort_type=abort_type,
                phase=f"gate {current_gate_index} post-credit wait",
            )
            race = host._visual_race_status_ref()
            if (
                race.race_finished
                or race.active_gate_index != current_gate_index
            ):
                raise abort_type(
                    "visual-course race boundary changed during fresh-frame "
                    "handoff"
                )
            snapshot = host.visual_gate_graph.latest_snapshot
            if evaluate_recovery_candidate(snapshot):
                break
            await send_zero(
                (
                    f"{VISUAL_COURSE_STAGE}/gate"
                    f"{current_gate_index}/post-credit-zero"
                ),
                float(runtime.monotonic()) - segment_started_s,
                yaw_reference_rad=yaw_reference_rad,
            )
            segment["post_credit_zero_command_count"] = int(
                segment["post_credit_zero_command_count"]
            ) + 1
            transition_summary["post_transition_zero_command_count"] = int(
                transition_summary["post_transition_zero_command_count"]
            ) + 1
        assert recovery_admission is not None
        assert recovery_admission_kind is not None
        assert admitted_recovery_token is not None
        transition_summary["recovery_admission"] = {
            "admission_kind": recovery_admission_kind,
            "admitted_frame_token": asdict(admitted_recovery_token),
            "track_id": recovery_admission.track_id,
            "promotion_identity_sha256": (
                recovery_admission.promotion_identity_sha256
            ),
            "promotion_identity_basis": (
                recovery_admission.promotion_identity_basis
            ),
            "cross_gap_identity_claimed": (
                recovery_admission.cross_gap_identity_claimed
            ),
            "visibility_epoch_frame_count": (
                recovery_admission.visibility_epoch_frame_count
            ),
            "visibility_epoch_span_s": (
                recovery_admission.visibility_epoch_span_s
            ),
        }

    raise abort_type("visual-course exceeded its gate-segment bound")


async def run_visual_course_stage(
    host: VisualCourseStageHost,
    context: Any,
    *,
    runtime: VisualCourseStageRuntime,
) -> Dict[str, Any]:
    """Run the course while retaining compact evidence on every failure."""

    try:
        return await _run_visual_course_stage_impl(
            host,
            context,
            runtime=runtime,
        )
    except BaseException as exc:
        summary = dict(
            getattr(host, "_visual_course_summary", None) or {}
        )
        summary.update(
            {
                "stage": VISUAL_COURSE_STAGE,
                "success": False,
                "race_finished": False,
                "outcome": "aborted",
                "first_causal_blocker": (
                    str(exc) or type(exc).__name__
                ),
            }
        )
        host._visual_course_summary = summary
        try:
            host.recorder.emit("visual_course_aborted", **summary)
        except BaseException as recorder_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "visual-course abort evidence also failed: "
                    f"{recorder_exc}"
                )
        raise


__all__ = [
    "DEFAULT_VISUAL_COURSE_LIMITS",
    "VISUAL_COURSE_STAGE",
    "VISUAL_COURSE_YAW_PROFILE_SCHEMA",
    "VisualCourseStageHost",
    "VisualCourseStageLimits",
    "VisualCourseStageRuntime",
    "VisualCourseYawProfile",
    "run_visual_course_stage",
]
