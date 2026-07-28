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

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol

from competition.adapter import (
    AttitudeRateCommand,
    RaceActiveBoundaryChangedBeforeWire,
)
from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    VisualInnerApertureGeometry,
    VisualTrackRole,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateReacquisition,
    ConfirmedGateTransition,
    CreditedUnboundGateAdvance,
    DEFAULT_ROLLING_GATE_GRAPH_CONFIG,
    GateGraphError,
    GateReacquisitionPending,
    RaceStatusProvenanceBasis,
)
from planning.vq2_course_lifecycle import (
    CourseLifecycle,
    DYNAMIC_CROSSING_PREDICTION_MAX_HORIZON_S,
    DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
    DYNAMIC_NEAR_PLANE_LATCH_BASIS,
    LatchedMeasurementMode,
    NearPlaneEvidence,
    NearPlaneLatch,
    NearPlaneWireSample,
    PostCreditMeasurementMode,
    advance_dynamic_near_plane_evidence,
    advance_near_plane_evidence,
    classify_post_credit_measurement,
    classify_latched_measurement,
)
from planning.vq2_dynamic_visual_approach import (
    BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ,
    DYNAMIC_CROSSING_COORDINATE_BASIS,
    DynamicVisualCourseSession,
    PostCreditSuccessorSteeringUnavailable,
)
from planning.vq2_visual_approach import (
    RollingVisualApproachServo,
    VisualApproachAdjacentUnavailable,
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
    VisualApproachPassageAdmission,
    VisualApproachRefusal,
)
from planning.vq2_visual_servo import (
    MAX_NEXT_GATE_BLEND,
    MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD,
    MAX_VISUAL_TARGET_PITCH_RAD,
    MAX_VISUAL_TARGET_ROLL_RAD,
    MAX_VISUAL_SEGMENT_DURATION_S,
    MAX_VISUAL_THRUST,
    MAX_VISUAL_YAW_RATE_RAD_S,
    MIN_VISUAL_TARGET_PITCH_RAD,
    MIN_VISUAL_THRUST,
    visual_bearing_yaw_rate,
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
CURRENT_APERTURE_PROVED_COLLECTIVE_BASIS = (
    "imu-derotated-current-center-proved-law-v2"
)
RAW_CURRENT_APERTURE_COLLECTIVE_BASIS = (
    "raw-camera-current-center-proved-law-v1"
)
GATE0_PROVED_NEXT_PREVIEW_ERROR_GAIN = 0.080
GATE0_PROVED_NEXT_PREVIEW_MAX_THRUST_DELTA = 0.012
GATE0_PROVED_NEXT_PREVIEW_BASIS = (
    "proved-gate0-reviewed-next-preview-collective-v1"
)
CENSORED_PASSAGE_COAST_BASIS = (
    "latched-clean-attitude-close-censored-passage-v1"
)
# Once a clean, uncertainty-bounded near-plane state has been latched, a
# tracker-rate outlier is a rejected measurement rather than evidence that the
# separately bounded crossing commitment became unsafe.  All lineage,
# identity, geometry, race, and command leases are still revalidated below.
_LATCHED_RATE_MEASUREMENT_REFUSALS = frozenset(
    {
        "visual target adaptation refused: horizontal target rate is implausible",
        (
            "visual target adaptation refused: image-down vertical target "
            "rate is implausible"
        ),
        "visual target adaptation refused: target scale rate is implausible",
    }
)
APPROACH_TOP_RECOVERY_BASIS = (
    "clean-q-converging-top-censored-approach-v1"
)
APPROACH_INNER_DROPOUT_HOLD_BASIS = (
    "fresh-top-censored-prior-inner-fov-continuity-v1"
)
APPROACH_PROPAGATED_VISIBILITY_GAP_BASIS = (
    "qualified-local-state-clipped-visibility-gap-v2"
)
APPROACH_CURRENT_AMBIGUITY_QUARANTINE_BASIS = (
    "same-current-identity-ambiguity-quarantine-v1"
)
APPROACH_CURRENT_AMBIGUITY_EXACT_RAW_LEASE_BASIS = (
    "same-current-first-ambiguity-exact-raw-top-lease-v1"
)
FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS = (
    "fresh-top-boundary-imu-closure-recovery-v1"
)
NONRAPID_OFF_AXIS_TOP_FOV_PRIORITY_BASIS = (
    "current-top-nonrapid-off-axis-fov-priority-v2"
)
RETAINED_FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS = (
    "retained-fresh-top-boundary-closure-recovery-v1"
)
POST_CREDIT_SUCCESSOR_HANDOFF_RETIREMENT_BASIS = (
    "fresh-authoritative-current-successor-handoff-retirement-v1"
)
FRESH_HORIZONTAL_DIRECT_TOP_FOV_BASIS = (
    "fresh-horizontal-edge-direct-top-fov-steering-v1"
)
FRESH_HORIZONTAL_FOV_CLOSURE_BRAKE_BASIS = (
    "fresh-horizontal-edge-predicted-contact-brake-v1"
)
FRESH_HORIZONTAL_FOV_EDGE_THRESHOLD = 0.90
FRESH_HORIZONTAL_FOV_MAX_EDGE_CONTACT_S = 0.25
APPROACH_INNER_DROPOUT_MAX_DURATION_S = 0.120
APPROACH_TOP_RECOVERY_ENDPOINT_SIGMA = 2.0
APPROACH_TOP_RECOVERY_MIN_INWARD_Q_RATE_S = 0.25
APPROACH_TOP_RECOVERY_MAX_VERTICAL_Q_STD = 0.18
APPROACH_TOP_RECOVERY_MAX_ABS_CAMERA_CENTER_NORM = 0.50
APPROACH_TOP_RECOVERY_THRUST_SLEW_PER_S = 0.15
APPROACH_TOP_RECOVERY_MAX_THRUST_SETTLE_S = 0.20
APPROACH_TOP_RECOVERY_ACTION_DELAY_S = 0.08
# The last dynamic run to earn authoritative Gate 0 -> 1 used these pitch
# reference kinematics during launch.  Reproduce that zero-initial-slope
# trajectory as a pure function of launch time and the current responsive
# destination; it has no prior-command state and is not a wire governor.
LAUNCH_PITCH_REFERENCE_MAX_RATE_RAD_S = 0.60
LAUNCH_PITCH_REFERENCE_ACCEL_RAD_S2 = 2.50
LAUNCH_PITCH_REFERENCE_BASIS = (
    "credited-gate0-stateless-accelerating-reference-v1"
)
# Raw current-gate geometry owns only camera observability.  This leaves 54 px
# above the conservative fitted inner-aperture edge at 640x360; derotated
# geometry remains the sole passage/collective input.
TOP_FOV_SAFE_EDGE_IMAGE_DOWN = -0.70
TOP_FOV_INNER_EDGE_SIGMA = 2.0
TOP_FOV_PITCH_PROTECTION_BASIS = (
    "raw-current-inner-top-pure-pitch-observability-v2"
)
TOP_FOV_INNER_EDGE_BASIS = (
    "raw-current-fitted-inner-aperture-top-2sigma-v1"
)
TOP_FOV_OUTER_EDGE_FALLBACK_BASIS = (
    "raw-current-bbox-top-fallback-v1"
)
TOP_FOV_PROPAGATED_INNER_EDGE_BASIS = (
    "propagated-current-camera-inner-aperture-top-2sigma-v1"
)
TOP_FOV_RETAINED_RAW_STATE_BASIS = (
    "prior-exact-raw-top-imu-propagation-v1"
)
TOP_FOV_EXACT_RAW_ANCHOR_BASIS = (
    "exact-raw-top-fov-anchor-v1"
)
_TOP_FOV_INNER_MODEL_PAIRS = frozenset(
    {
        (
            "vq2-visible-inner-quad-lines-v1",
            "vq2-visible-aperture-diagonal-v1",
        ),
        (
            "vq2-temporally-associated-inner-quad-lines-v1",
            "vq2-temporally-associated-aperture-diagonal-v1",
        ),
    }
)
_YAW_PROFILE_ISSUER = object()


def _post_credit_successor_handoff_required_after_command(
    *,
    required_before: bool,
    measurement_mode: PostCreditMeasurementMode,
    propagated_steering_applied: bool,
) -> bool:
    """Retire the required seam after one current-owned one-edge command."""

    if (
        type(required_before) is not bool
        or type(measurement_mode) is not PostCreditMeasurementMode
        or type(propagated_steering_applied) is not bool
    ):
        raise TypeError("post-credit successor handoff state is invalid")
    return bool(
        required_before
        and not (
            propagated_steering_applied
            and measurement_mode
            is PostCreditMeasurementMode.ONE_EDGE_CENSORED
        )
    )


def _body_to_reference_pitch_rad(
    body_to_reference_wxyz: tuple[float, float, float, float],
) -> float:
    if type(body_to_reference_wxyz) is not tuple or len(
        body_to_reference_wxyz
    ) != 4:
        raise ValueError("body-to-reference quaternion is invalid")
    w, x, y, z = map(float, body_to_reference_wxyz)
    if not all(math.isfinite(value) for value in (w, x, y, z)):
        raise ValueError("body-to-reference quaternion is invalid")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("body-to-reference quaternion is not unit length")
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return math.asin(max(-1.0, min(1.0, sin_pitch)))


def _raw_bbox_top_image_down(
    bbox_norm_ltrb: tuple[float, float, float, float],
) -> float:
    if type(bbox_norm_ltrb) is not tuple or len(bbox_norm_ltrb) != 4:
        raise ValueError("raw current bbox is invalid")
    left, top, right, bottom = map(float, bbox_norm_ltrb)
    if not (
        all(math.isfinite(value) for value in (left, top, right, bottom))
        and
        0.0 <= left < right <= 1.0
        and 0.0 <= top < bottom <= 1.0
    ):
        raise ValueError("raw current bbox is outside the unit image")
    return 2.0 * top - 1.0


@dataclass(frozen=True, slots=True)
class _TopFovRawEdge:
    top_edge_image_down: float
    nominal_top_edge_image_down: float
    top_edge_std_image_down: float
    basis: str
    confidence: float


def _conservative_inner_aperture_top_image_down(
    inner: VisualInnerApertureGeometry,
) -> float:
    """Return the raw fitted aperture top including two-sigma uncertainty."""

    if (
        type(inner) is not VisualInnerApertureGeometry
        or not inner.fitted
        or not inner.complete_visibility
        or inner.clipping != FrameEdge.NONE
    ):
        raise ValueError("complete raw inner-aperture geometry is unavailable")
    assert inner.center_norm is not None
    assert inner.half_size_norm is not None
    assert inner.measurement_std is not None
    center_y = float(inner.center_norm[1])
    half_y = float(inner.half_size_norm[1])
    std_y = float(inner.measurement_std[1])
    std_log_scale = float(inner.measurement_std[2])
    if (
        inner.geometry_model_id,
        inner.covariance_model_id,
    ) not in _TOP_FOV_INNER_MODEL_PAIRS:
        raise ValueError("inner-aperture top model identity is invalid")
    nominal_top = center_y - half_y
    top_std = math.sqrt(
        std_y * std_y + (half_y * std_log_scale) ** 2
    )
    top = nominal_top - TOP_FOV_INNER_EDGE_SIGMA * top_std
    if (
        not all(
            math.isfinite(value)
            for value in (
                center_y,
                half_y,
                std_y,
                std_log_scale,
                nominal_top,
                top_std,
                top,
            )
        )
        or top > 1.0
    ):
        raise ValueError("inner-aperture top geometry is invalid")
    # The uncertainty interval can extend beyond the physical image while the
    # fitted edge remains visible. Clamp only at that physical boundary.
    return max(-1.0, top)


def _top_fov_raw_edge(sample: Any) -> _TopFovRawEdge:
    """Select exact raw inner geometry, with an early outer-support fallback."""

    inner = getattr(sample, "inner_aperture", None)
    if inner is not None and type(inner) is not VisualInnerApertureGeometry:
        raise ValueError("raw inner-aperture geometry has an invalid type")
    if (
        type(inner) is VisualInnerApertureGeometry
        and inner.fitted
        and inner.complete_visibility
        and inner.clipping == FrameEdge.NONE
    ):
        conservative_top = _conservative_inner_aperture_top_image_down(inner)
        assert inner.center_norm is not None
        assert inner.half_size_norm is not None
        assert inner.measurement_std is not None
        half_y = float(inner.half_size_norm[1])
        top_std = math.sqrt(
            float(inner.measurement_std[1]) ** 2
            + (half_y * float(inner.measurement_std[2])) ** 2
        )
        return _TopFovRawEdge(
            top_edge_image_down=conservative_top,
            nominal_top_edge_image_down=(
                float(inner.center_norm[1]) - half_y
            ),
            top_edge_std_image_down=top_std,
            basis=TOP_FOV_INNER_EDGE_BASIS,
            confidence=float(inner.confidence),
        )

    outer_top = _raw_bbox_top_image_down(sample.bbox_norm)
    clipping = getattr(sample, "clipping", None)
    center_censored = getattr(sample, "center_censored", None)
    horizontal_edges = FrameEdge.LEFT | FrameEdge.RIGHT
    horizontal_only = bool(
        clipping is not None
        and clipping != FrameEdge.NONE
        and clipping & ~horizontal_edges == FrameEdge.NONE
    )
    if (
        clipping is None
        or bool(clipping & (FrameEdge.TOP | FrameEdge.BOTTOM))
        or center_censored is not False
        and not horizontal_only
    ):
        raise ValueError(
            "complete raw inner aperture and clean outer fallback are "
            "unavailable"
        )
    confidence = float(sample.confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("raw outer-support confidence is invalid")
    return _TopFovRawEdge(
        top_edge_image_down=outer_top,
        nominal_top_edge_image_down=outer_top,
        top_edge_std_image_down=0.0,
        basis=TOP_FOV_OUTER_EDGE_FALLBACK_BASIS,
        confidence=confidence,
    )


def _top_fov_edge_recovery_rate_down_s(
    *,
    current: _TopFovRawEdge,
    previous: _TopFovRawEdge,
    elapsed_s: float,
) -> Optional[float]:
    """Return the motion of the same-source conservative top edge."""

    elapsed = float(elapsed_s)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("top-FOV edge-rate interval is invalid")
    if current.basis != previous.basis:
        return None
    rate = (
        current.top_edge_image_down
        - previous.top_edge_image_down
    ) / elapsed
    if not math.isfinite(rate):
        raise ValueError("top-FOV edge recovery rate is nonfinite")
    return rate


def _top_fov_nonrotational_angle_rate_rad_s(
    *,
    current_top_edge_image_down: float,
    previous_top_edge_image_down: float,
    vertical_angle_scale_rad: float,
    elapsed_s: float,
    measured_pitch_rate_rad_s: float,
) -> float:
    """Remove calibrated pure-pitch image motion from one raw edge rate."""

    current_top, previous_top, scale, elapsed, pitch_rate = map(
        float,
        (
            current_top_edge_image_down,
            previous_top_edge_image_down,
            vertical_angle_scale_rad,
            elapsed_s,
            measured_pitch_rate_rad_s,
        ),
    )
    if (
        not all(
            math.isfinite(value)
            for value in (
                current_top,
                previous_top,
                scale,
                elapsed,
                pitch_rate,
            )
        )
        or not -1.0 <= current_top <= 1.0
        or not -1.0 <= previous_top <= 1.0
        or scale <= 0.0
        or elapsed <= 0.0
    ):
        raise ValueError("top-FOV nonrotational edge-rate inputs are invalid")
    raw_angle_rate = (
        math.atan(current_top * scale)
        - math.atan(previous_top * scale)
    ) / elapsed
    result = raw_angle_rate + pitch_rate
    if not math.isfinite(result):
        raise ValueError("top-FOV nonrotational edge rate is nonfinite")
    return result


def _project_raw_vertical_edge_for_pitch_reference(
    *,
    edge_image_down: float,
    capture_pitch_rad: float,
    target_pitch_rad: float,
    vertical_angle_scale_rad: float,
) -> float:
    """Project one raw edge using ``alpha_t=alpha_c-(pitch_t-pitch_c)``."""

    edge, capture, target, scale = map(
        float,
        (
            edge_image_down,
            capture_pitch_rad,
            target_pitch_rad,
            vertical_angle_scale_rad,
        ),
    )
    if (
        not all(
            math.isfinite(value)
            for value in (edge, capture, target, scale)
        )
        or not -1.0 <= edge <= 1.0
        or scale <= 0.0
    ):
        raise ValueError("raw edge pitch reprojection inputs are invalid")
    return math.tan(
        math.atan(edge * scale) - (target - capture)
    ) / scale


@dataclass(frozen=True, slots=True)
class _TopFovPitchProposal:
    raw_top_edge_image_down: float
    raw_top_edge_rate_down_s: Optional[float]
    raw_top_edge_nonrotational_angle_rate_rad_s: Optional[float]
    prediction_horizon_s: float
    forecast_top_edge_image_down: float
    capture_pitch_rad: float
    requested_target_pitch_rad: float
    maximum_observable_target_pitch_rad: float
    protected_target_pitch_rad: float
    predicted_requested_top_edge_image_down: float
    predicted_protected_top_edge_image_down: float
    clearance_recovering: bool
    envelope_saturated: bool
    active_before: bool
    active_after: bool
    limited: bool


@dataclass(frozen=True, slots=True)
class _TopFovObservation:
    capture_pitch_rad: float
    raw_top_edge_image_down: float
    raw_nominal_top_edge_image_down: float
    raw_top_edge_std_image_down: float
    raw_top_edge_rate_down_s: Optional[float]
    raw_top_edge_motion_angle_rate_rad_s: Optional[float]
    raw_top_edge_nonrotational_angle_rate_rad_s: Optional[float]
    vertical_angle_scale_rad: float
    pitch_response_delay_s: float
    previous_target_pitch_rad: Optional[float]
    raw_top_edge_basis: str
    raw_top_edge_confidence: float


@dataclass(frozen=True, slots=True)
class _TopFovPropagatedObservation:
    capture_pitch_rad: float
    projected_top_edge_image_down: float
    projected_nominal_top_edge_image_down: float
    projected_top_edge_std_image_down: float
    vertical_angle_scale_rad: float
    prediction_horizon_remaining_s: float
    geometry_basis: str


@dataclass(frozen=True, slots=True)
class _TopFovRetainedRawStateObservation:
    anchor_camera_token: CameraFrameToken
    camera_token: CameraFrameToken
    anchor_capture_pitch_rad: float
    capture_pitch_rad: float
    projected_top_edge_image_down: float
    projected_nominal_top_edge_image_down: float
    projected_uncertainty_growth_rad: float
    raw_top_edge_nonrotational_angle_rate_rad_s: Optional[float]
    vertical_angle_scale_rad: float
    observation_age_s: float
    wall_age_s: float
    prediction_horizon_remaining_s: float
    geometry_basis: str


@dataclass(frozen=True, slots=True)
class _FreshTopCensoredClosureRecovery:
    """Joint pitch/collective allocation for one fresh TOP boundary.

    The raw boundary and IMU state recover only steering/control allocation.
    They do not recreate an aperture, crossing clearance, passage, or race
    authority.
    """

    basis: str
    raw_top_edge_image_down: float
    capture_pitch_rad: float
    body_pitch_rate_rad_s: float
    predicted_pitch_at_response_rad: float
    stable_center_norm: tuple[float, float]
    residual_rate_norm_s: tuple[float, float]
    expansion_rate_s: float
    time_to_contact_s: Optional[float]
    horizontal_aligned: bool
    requested_target_pitch_rad: float
    fov_protected_target_pitch_rad: float
    allocated_target_pitch_rad: float
    requested_thrust: float
    allocated_thrust: float
    fresh_boundary_current_authority: bool
    forward_closure_authorized: bool
    steering_only: bool
    passage_authority: bool
    advance_authority: bool


@dataclass(frozen=True, slots=True)
class _RetainedFreshTopCensoredClosureRecovery:
    """Accepted exact-TOP allocation bridged across one retained-track loss."""

    basis: str
    source_basis: str
    gate_index: int
    track_id: str
    source_camera_token: CameraFrameToken
    current_camera_token: CameraFrameToken
    source_wire_start_monotonic_ns: int
    expires_monotonic_ns: int
    missed_frame_count: int
    requested_target_pitch_rad: float
    retained_target_pitch_floor_rad: float
    allocated_target_pitch_rad: float
    requested_thrust: float
    retained_thrust_floor: float
    allocated_thrust: float
    retained_through_missing_frame: bool
    forward_closure_authorized: bool
    steering_only: bool
    passage_authority: bool
    advance_authority: bool


@dataclass(frozen=True, slots=True)
class _FreshHorizontalFovClosureBrake:
    """Non-forward allocation for one predicted horizontal FOV escape."""

    basis: str
    horizontal_edge: int
    raw_bbox_norm_ltrb: tuple[float, float, float, float]
    raw_outward_edge_image_fraction: float
    raw_half_width_image_fraction: float
    raw_center_velocity_norm_s: float
    log_scale_rate_s: float
    outward_edge_rate_image_fraction_s: float
    predicted_edge_contact_s: float
    edge_threshold_image_fraction: float
    maximum_edge_contact_s: float
    requested_target_pitch_rad: float
    fov_protected_target_pitch_rad: float
    allocated_target_pitch_rad: float
    requested_thrust: float
    allocated_thrust: float
    fresh_current_authority: bool
    forward_closure_authorized: bool
    steering_only: bool
    passage_authority: bool
    advance_authority: bool


@dataclass(frozen=True, slots=True)
class _FreshCurrentTopBoundaryAuthority:
    """Exact authoritative-current publication allowed to withhold closure."""

    gate_index: int
    track_id: str
    camera_token: CameraFrameToken
    tracker_frame_sequence: int
    current: Any
    track: Any
    sample: Any


def _fresh_exact_top_boundary_preempts_propagated_fov(
    clipping: Any,
) -> bool:
    """Select physical TOP closure before still-live propagated aperture."""

    return clipping is FrameEdge.TOP


def _allocate_fresh_horizontal_fov_closure_brake(
    *,
    bbox_norm_ltrb: tuple[float, float, float, float],
    center_velocity_norm_s: tuple[float, float],
    log_scale_rate_s: float,
    clipping: FrameEdge,
    center_censored: bool,
    current_visible: bool,
    current_ambiguous: bool,
    current_missed_count: int,
    current_censored_axes: tuple[bool, bool],
    passage_committed: bool,
    requested_target_pitch_rad: float,
    fov_protected_target_pitch_rad: float,
    requested_thrust: float,
) -> Optional[_FreshHorizontalFovClosureBrake]:
    """Stop closure before a fresh current gate leaves a horizontal edge.

    Tracker center velocity is expressed in normalized ``[-1, 1]`` image
    coordinates, while the raw bbox uses image fractions.  Half the center
    rate plus scale-driven half-width growth therefore predicts each raw
    side's contact with the viewport.  This allocation changes pitch only;
    fresh visual guidance retains roll/yaw steering and no passage authority
    is created.
    """

    if (
        type(bbox_norm_ltrb) is not tuple
        or len(bbox_norm_ltrb) != 4
        or type(center_velocity_norm_s) is not tuple
        or len(center_velocity_norm_s) != 2
        or type(clipping) is not FrameEdge
        or type(center_censored) is not bool
        or type(current_visible) is not bool
        or type(current_ambiguous) is not bool
        or type(current_missed_count) is not int
        or type(current_censored_axes) is not tuple
        or len(current_censored_axes) != 2
        or type(passage_committed) is not bool
    ):
        raise ValueError("fresh horizontal FOV brake structure is invalid")
    left, top, right, bottom = map(float, bbox_norm_ltrb)
    velocity_x, velocity_y = map(float, center_velocity_norm_s)
    expansion = float(log_scale_rate_s)
    requested_pitch = float(requested_target_pitch_rad)
    protected_pitch = float(fov_protected_target_pitch_rad)
    thrust = float(requested_thrust)
    if (
        not all(
            math.isfinite(value)
            for value in (
                left,
                top,
                right,
                bottom,
                velocity_x,
                velocity_y,
                expansion,
                requested_pitch,
                protected_pitch,
                thrust,
            )
        )
        or not 0.0 <= left < right <= 1.0
        or not 0.0 <= top < bottom <= 1.0
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= requested_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= protected_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_THRUST <= thrust <= MAX_VISUAL_THRUST
    ):
        raise ValueError("fresh horizontal FOV brake input is invalid")
    if not (
        clipping is FrameEdge.NONE
        and not center_censored
        and current_visible
        and not current_ambiguous
        and current_missed_count == 0
        and current_censored_axes == (False, False)
        and not passage_committed
    ):
        return None

    half_width = 0.5 * (right - left)
    candidates: list[tuple[float, FrameEdge, float, float]] = []
    side_inputs = (
        (
            FrameEdge.LEFT,
            1.0 - left,
            -0.5 * velocity_x + half_width * expansion,
        ),
        (
            FrameEdge.RIGHT,
            right,
            0.5 * velocity_x + half_width * expansion,
        ),
    )
    for side, outward_edge, outward_rate in side_inputs:
        if (
            outward_edge
            < FRESH_HORIZONTAL_FOV_EDGE_THRESHOLD - 1e-12
            or outward_rate <= 0.0
        ):
            continue
        edge_contact_s = (1.0 - outward_edge) / outward_rate
        if (
            edge_contact_s < -1e-12
            or edge_contact_s
            > FRESH_HORIZONTAL_FOV_MAX_EDGE_CONTACT_S + 1e-12
        ):
            continue
        candidates.append(
            (
                max(0.0, edge_contact_s),
                side,
                outward_edge,
                outward_rate,
            )
        )
    if not candidates:
        return None
    (
        edge_contact_s,
        side,
        outward_edge,
        outward_rate,
    ) = min(candidates, key=lambda candidate: candidate[0])
    allocated_pitch = max(0.0, protected_pitch)
    return _FreshHorizontalFovClosureBrake(
        basis=FRESH_HORIZONTAL_FOV_CLOSURE_BRAKE_BASIS,
        horizontal_edge=int(side),
        raw_bbox_norm_ltrb=(left, top, right, bottom),
        raw_outward_edge_image_fraction=outward_edge,
        raw_half_width_image_fraction=half_width,
        raw_center_velocity_norm_s=velocity_x,
        log_scale_rate_s=expansion,
        outward_edge_rate_image_fraction_s=outward_rate,
        predicted_edge_contact_s=edge_contact_s,
        edge_threshold_image_fraction=FRESH_HORIZONTAL_FOV_EDGE_THRESHOLD,
        maximum_edge_contact_s=FRESH_HORIZONTAL_FOV_MAX_EDGE_CONTACT_S,
        requested_target_pitch_rad=requested_pitch,
        fov_protected_target_pitch_rad=protected_pitch,
        allocated_target_pitch_rad=allocated_pitch,
        requested_thrust=thrust,
        allocated_thrust=thrust,
        fresh_current_authority=True,
        forward_closure_authorized=False,
        steering_only=True,
        passage_authority=False,
        advance_authority=False,
    )


def _fresh_current_top_boundary_authority(
    session: DynamicVisualCourseSession,
    *,
    snapshot: Any,
    current_gate_index: int,
    current_track_id: str,
) -> _FreshCurrentTopBoundaryAuthority:
    """Bind one fresh TOP boundary without consuming stale aperture state."""

    if type(session) is not DynamicVisualCourseSession:
        raise ValueError("fresh current TOP boundary lacks dynamic authority")
    state = session.core.course_state()
    current = getattr(state, "current", None)
    track = getattr(snapshot, "current_track", None)
    token = getattr(snapshot, "latest_camera_token", None)
    history = getattr(track, "history", None)
    sample = (
        None
        if type(history) is not tuple or not history
        else history[-1]
    )
    raw_top: Optional[float] = None
    if sample is not None:
        try:
            raw_top = _raw_bbox_top_image_down(sample.bbox_norm)
        except (AttributeError, TypeError, ValueError):
            raw_top = None
    if (
        type(current_gate_index) is not int
        or current_gate_index < 0
        or type(current_track_id) is not str
        or not current_track_id
        or type(token) is not CameraFrameToken
        or sample is None
        or current is None
        or getattr(snapshot, "current_gate_index", None)
        != current_gate_index
        or getattr(snapshot, "current_track_id", None)
        != current_track_id
        or getattr(snapshot, "authority_usable", None) is not True
        or getattr(snapshot, "withholding_reason", None) is not None
        or getattr(snapshot, "race_finished", None) is not False
        or getattr(state, "current_gate_index", None)
        != current_gate_index
        or getattr(state, "current_track_id", None)
        != current_track_id
        or getattr(current, "track_id", None) != current_track_id
        or getattr(track, "track_id", None) != current_track_id
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "authoritative_gate_index", None)
        != current_gate_index
        or getattr(track, "latest_token", None) != token
        or getattr(sample, "token", None) != token
        or getattr(current, "frame_sequence", None)
        != getattr(sample, "tracker_frame_sequence", None)
        or getattr(current, "stream_generation", None)
        != token.generation
        # The dynamic state may carry a complete tracking-only inner fit from
        # this same publication, so its control geometry remains unclipped
        # while the exact raw outer support below owns the TOP boundary.
        or getattr(current, "clipping", None)
        not in {FrameEdge.NONE, FrameEdge.TOP}
        or getattr(current, "censored_axes", None)
        not in {(False, True), (False, False)}
        or not bool(getattr(current, "visible", False))
        or bool(getattr(current, "ambiguous", True))
        or getattr(current, "missed_count", None) != 0
        or not bool(getattr(track, "visible", False))
        or bool(getattr(track, "ambiguous", True))
        or getattr(track, "missed_frame_count", None) != 0
        or getattr(track, "clipping", None) is not FrameEdge.TOP
        or getattr(sample, "clipping", None) is not FrameEdge.TOP
        or getattr(track, "center_censored", None) is not True
        or getattr(sample, "center_censored", None) is not True
        or raw_top is None
        or raw_top > -1.0 + 1e-12
    ):
        raise ValueError(
            "fresh current TOP boundary differs from authoritative lineage"
        )
    return _FreshCurrentTopBoundaryAuthority(
        gate_index=current_gate_index,
        track_id=current_track_id,
        camera_token=token,
        tracker_frame_sequence=sample.tracker_frame_sequence,
        current=current,
        track=track,
        sample=sample,
    )


def _fresh_post_credit_top_boundary_authority(
    *,
    state: Any,
    decision: Any,
    authority: Mapping[str, Any],
    recovery_snapshot: Any,
    current_gate_index: int,
) -> _FreshCurrentTopBoundaryAuthority:
    """Bind aperture-free recovery to one exact fresh credited boundary."""

    track = getattr(recovery_snapshot, "current_track", None)
    token = getattr(recovery_snapshot, "latest_camera_token", None)
    history = getattr(track, "history", None)
    sample = (
        None
        if type(history) is not tuple or not history
        else history[-1]
    )
    current = getattr(state, "current", None)
    track_id = getattr(state, "current_track_id", None)
    if (
        type(current_gate_index) is not int
        or current_gate_index < 0
        or not isinstance(authority, Mapping)
        or type(token) is not CameraFrameToken
        or sample is None
        or current is None
        or type(track_id) is not str
        or not track_id
        or getattr(state, "current_gate_index", None)
        != current_gate_index
        or getattr(decision, "current_gate_index", None)
        != current_gate_index
        or getattr(decision, "current_track_id", None) != track_id
        or authority.get("reviewed_track_id") != track_id
        or getattr(recovery_snapshot, "current_gate_index", None)
        != current_gate_index
        or getattr(recovery_snapshot, "current_track_id", None)
        != track_id
        or getattr(current, "track_id", None) != track_id
        or getattr(track, "track_id", None) != track_id
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "authoritative_gate_index", None)
        != current_gate_index
        or getattr(track, "latest_token", None) != token
        or getattr(sample, "token", None) != token
        or getattr(current, "frame_sequence", None)
        != getattr(sample, "tracker_frame_sequence", None)
        or getattr(current, "stream_generation", None)
        != token.generation
        or not bool(getattr(current, "visible", False))
        or bool(getattr(current, "ambiguous", True))
        or getattr(current, "missed_count", None) != 0
        or not bool(getattr(track, "visible", False))
        or bool(getattr(track, "ambiguous", True))
        or getattr(track, "missed_frame_count", None) != 0
        or getattr(track, "clipping", None) is not FrameEdge.TOP
        or getattr(sample, "clipping", None) is not FrameEdge.TOP
        or getattr(track, "center_censored", None) is not True
    ):
        raise ValueError(
            "fresh post-credit TOP boundary differs from credited current "
            "lineage"
        )
    return _FreshCurrentTopBoundaryAuthority(
        gate_index=current_gate_index,
        track_id=track_id,
        camera_token=token,
        tracker_frame_sequence=sample.tracker_frame_sequence,
        current=current,
        track=track,
        sample=sample,
    )


def _fresh_post_credit_horizontal_top_fov_pitch_reference(
    session: DynamicVisualCourseSession,
    *,
    state: Any,
    decision: Any,
    authority: Mapping[str, Any],
    recovery_snapshot: Any,
    current_gate_index: int,
    requested_target_pitch_rad: float,
) -> tuple[_TopFovPitchProposal, Mapping[str, Any]]:
    """Protect pitch from a fresh top edge during LEFT/RIGHT-only recovery.

    Horizontal clipping leaves the outer support's top boundary directly
    observable.  This path consumes only the already-authoritative current
    identity and the existing post-credit steering envelope.  It cannot
    create aperture, passage, advance, or cross-gap identity authority.
    """

    track = getattr(recovery_snapshot, "current_track", None)
    token = getattr(recovery_snapshot, "latest_camera_token", None)
    history = getattr(track, "history", None)
    sample = (
        None
        if type(history) is not tuple or not history
        else history[-1]
    )
    current = getattr(state, "current", None)
    track_id = getattr(state, "current_track_id", None)
    clipping = getattr(track, "clipping", None)
    direct_top: Optional[float] = None
    if sample is not None:
        try:
            direct_top = _raw_bbox_top_image_down(sample.bbox_norm)
        except (AttributeError, TypeError, ValueError):
            direct_top = None
    if (
        type(session) is not DynamicVisualCourseSession
        or type(current_gate_index) is not int
        or current_gate_index < 0
        or not isinstance(authority, Mapping)
        or type(token) is not CameraFrameToken
        or sample is None
        or current is None
        or type(track_id) is not str
        or not track_id
        or type(clipping) is not FrameEdge
        or clipping not in {FrameEdge.LEFT, FrameEdge.RIGHT}
        or getattr(state, "current_gate_index", None)
        != current_gate_index
        or getattr(decision, "current_gate_index", None)
        != current_gate_index
        or getattr(decision, "current_track_id", None) != track_id
        or authority.get("reviewed_track_id") != track_id
        or authority.get("steering_track_id") != track_id
        or authority.get("to_gate_index") != current_gate_index
        or authority.get("stream_generation") != token.generation
        or authority.get("steering_available") is not True
        or authority.get("steering_only") is not True
        or authority.get("passage_authority") is not False
        or authority.get("advance_authority") is not False
        or authority.get("vertical_axis_censored") is not False
        or authority.get("current_raw_clipping") != int(clipping)
        or getattr(recovery_snapshot, "current_gate_index", None)
        != current_gate_index
        or getattr(recovery_snapshot, "current_track_id", None)
        != track_id
        or getattr(recovery_snapshot, "authority_usable", None) is not True
        or getattr(recovery_snapshot, "withholding_reason", None) is not None
        or getattr(recovery_snapshot, "race_finished", None) is not False
        or getattr(current, "track_id", None) != track_id
        or getattr(track, "track_id", None) != track_id
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "authoritative_gate_index", None)
        != current_gate_index
        or getattr(track, "latest_token", None) != token
        or getattr(sample, "token", None) != token
        or getattr(current, "frame_sequence", None)
        != getattr(sample, "tracker_frame_sequence", None)
        or getattr(current, "stream_generation", None)
        != token.generation
        or not bool(getattr(current, "visible", False))
        or bool(getattr(current, "ambiguous", True))
        or getattr(current, "missed_count", None) != 0
        or getattr(current, "censored_axes", None) != (True, False)
        or not bool(getattr(track, "visible", False))
        or bool(getattr(track, "ambiguous", True))
        or getattr(track, "missed_frame_count", None) != 0
        or getattr(sample, "clipping", None) != clipping
        or getattr(track, "center_censored", None) is not True
        or getattr(sample, "center_censored", None) is not True
        or direct_top is None
        or not -1.0 < direct_top <= 1.0
    ):
        raise ValueError(
            "fresh post-credit horizontal top boundary differs from "
            "credited current lineage"
        )

    config = session.core.config
    requested_pitch = float(requested_target_pitch_rad)
    retained_ceiling_value = authority.get(
        "retained_pitch_ceiling_rad"
    )
    retained_ceiling = (
        requested_pitch
        if retained_ceiling_value is None
        else float(retained_ceiling_value)
    )
    capture_pitch = _body_to_reference_pitch_rad(
        current.body_to_reference_wxyz
    )
    vertical_scale = float(config.vertical_angle_scale_rad)
    response_delay = float(config.pitch_command_delay_s)
    if (
        not all(
            math.isfinite(value)
            for value in (
                requested_pitch,
                retained_ceiling,
                capture_pitch,
                vertical_scale,
                response_delay,
            )
        )
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= requested_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= retained_ceiling
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or vertical_scale <= 0.0
        or response_delay < 0.0
    ):
        raise ValueError(
            "fresh post-credit horizontal top boundary inputs are invalid"
        )
    proposal = _propose_top_fov_pitch_reference(
        capture_pitch_rad=capture_pitch,
        raw_top_edge_image_down=direct_top,
        raw_top_edge_rate_down_s=None,
        requested_target_pitch_rad=requested_pitch,
        prior_target_pitch_rad=min(requested_pitch, retained_ceiling),
        vertical_angle_scale_rad=vertical_scale,
        # This is a transition from a censored recovery state.  A fresh
        # horizontal-only frame proves its top location but not a temporal
        # recovery trend, so it may tighten and never loosen the prior bound.
        active_before=True,
        raw_top_edge_nonrotational_angle_rate_rad_s=None,
        prediction_horizon_s=response_delay,
    )
    evidence = {
        "basis": FRESH_HORIZONTAL_DIRECT_TOP_FOV_BASIS,
        "gate_index": current_gate_index,
        "track_id": track_id,
        "camera_token": asdict(token),
        "clipping": int(clipping),
        "raw_top_edge_basis": TOP_FOV_OUTER_EDGE_FALLBACK_BASIS,
        "raw_top_edge_image_down": direct_top,
        "safe_top_edge_image_down": TOP_FOV_SAFE_EDGE_IMAGE_DOWN,
        "capture_pitch_rad": capture_pitch,
        "retained_pitch_ceiling_rad": retained_ceiling,
        "source_target_pitch_rad": requested_pitch,
        "propagated_aperture_available": False,
        **asdict(proposal),
        "aperture_authority": False,
        "steering_only": True,
        "passage_authority": False,
        "advance_authority": False,
        "cross_gap_identity_claimed": False,
    }
    return proposal, evidence


def _allocate_fresh_top_censored_closure_recovery(
    *,
    raw_top_edge_image_down: float,
    clipping: FrameEdge,
    center_censored: bool,
    current_visible: bool,
    current_ambiguous: bool,
    current_missed_count: int,
    current_censored_axes: tuple[bool, bool],
    current_aperture_propagated: bool,
    current_aperture_dynamics_qualified: bool,
    passage_committed: bool,
    capture_pitch_rad: float,
    body_pitch_rate_rad_s: float,
    pitch_response_delay_s: float,
    stable_center_norm: tuple[float, float],
    residual_rate_rad_s: tuple[float, float],
    horizontal_angle_scale_rad: float,
    vertical_angle_scale_rad: float,
    off_axis_brake_rad: float,
    expansion_rate_s: float,
    time_to_contact_s: Optional[float],
    requested_target_pitch_rad: float,
    fov_protected_target_pitch_rad: float,
    requested_thrust: float,
    fresh_boundary_current_authority: Optional[
        _FreshCurrentTopBoundaryAuthority
    ] = None,
) -> Optional[_FreshTopCensoredClosureRecovery]:
    """Remove forward authority while a fresh current gate is TOP-censored.

    A saturated raw top edge is a fresh one-sided observation, not a measured
    aperture edge.  The local state and IMU therefore retain vertical/lateral
    steering, while the requested attitude is projected to the non-forward
    half-space and the existing collective remains dedicated to vertical
    recovery.  An exact fresh post-credit current boundary can withhold
    forward authority even before an inner-aperture seed exists; unknown TTC
    never grants closure.  This is a state-dependent reference allocation;
    the final wire governor remains the sole temporal continuity authority.
    """

    boundary_authorized = (
        fresh_boundary_current_authority is not None
    )
    if (
        type(clipping) is not FrameEdge
        or (
            fresh_boundary_current_authority is not None
            and type(fresh_boundary_current_authority)
            is not _FreshCurrentTopBoundaryAuthority
        )
        or type(current_censored_axes) is not tuple
        or len(current_censored_axes) != 2
        or type(stable_center_norm) is not tuple
        or len(stable_center_norm) != 2
        or type(residual_rate_rad_s) is not tuple
        or len(residual_rate_rad_s) != 2
    ):
        raise ValueError("fresh TOP recovery structure is invalid")
    (
        raw_top,
        capture_pitch,
        pitch_rate,
        response_delay,
        stable_x,
        stable_y,
        residual_x,
        residual_y,
        horizontal_scale,
        vertical_scale,
        off_axis,
        expansion,
        requested_pitch,
        fov_pitch,
        thrust,
    ) = map(
        float,
        (
            raw_top_edge_image_down,
            capture_pitch_rad,
            body_pitch_rate_rad_s,
            pitch_response_delay_s,
            *stable_center_norm,
            *residual_rate_rad_s,
            horizontal_angle_scale_rad,
            vertical_angle_scale_rad,
            off_axis_brake_rad,
            expansion_rate_s,
            requested_target_pitch_rad,
            fov_protected_target_pitch_rad,
            requested_thrust,
        ),
    )
    ttc = None if time_to_contact_s is None else float(time_to_contact_s)
    if (
        not all(
            math.isfinite(value)
            for value in (
                raw_top,
                capture_pitch,
                pitch_rate,
                response_delay,
                stable_x,
                stable_y,
                residual_x,
                residual_y,
                horizontal_scale,
                vertical_scale,
                off_axis,
                expansion,
                requested_pitch,
                fov_pitch,
                thrust,
            )
        )
        or ttc is not None and not math.isfinite(ttc)
        or response_delay < 0.0
        or horizontal_scale <= 0.0
        or vertical_scale <= 0.0
        or off_axis <= 0.0
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= requested_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= fov_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_THRUST <= thrust <= MAX_VISUAL_THRUST
    ):
        raise ValueError("fresh TOP recovery input is invalid")
    if not (
        clipping is FrameEdge.TOP
        and center_censored
        and current_visible
        and not current_ambiguous
        and current_missed_count == 0
        and (
            current_censored_axes == (False, True)
            or (
                boundary_authorized
                and current_censored_axes == (False, False)
            )
        )
        and (
            boundary_authorized
            or (
                current_aperture_propagated
                and current_aperture_dynamics_qualified
                and expansion > 0.0
                and ttc is not None
                and ttc > response_delay
                and fov_pitch < requested_pitch - 1e-12
            )
        )
        and not passage_committed
        and raw_top <= -1.0 + 1e-12
        and (
            boundary_authorized
            or stable_y < 0.0
        )
        and requested_pitch >= 0.0
    ):
        return None

    predicted_pitch = capture_pitch + pitch_rate * response_delay
    # The exact fresh boundary removes forward-closure authority and calls
    # for the already-bounded dynamic brake reference.  The final wire
    # governor remains the sole temporal slew authority; projecting this
    # target back to zero previously left only ~0.024 rad/s of pitch response
    # in live TOP saturation and failed to arrest closure.
    allocated_pitch = (
        requested_pitch
        if boundary_authorized
        else min(requested_pitch, max(0.0, predicted_pitch))
    )
    allocated_thrust = max(GATE0_PROVED_COLLECTIVE_BASE, thrust)
    residual_rate_norm = (
        residual_x / horizontal_scale,
        residual_y / vertical_scale,
    )
    horizontal_aligned = bool(
        abs(math.atan(stable_x * horizontal_scale)) < off_axis
    )
    return _FreshTopCensoredClosureRecovery(
        basis=FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS,
        raw_top_edge_image_down=raw_top,
        capture_pitch_rad=capture_pitch,
        body_pitch_rate_rad_s=pitch_rate,
        predicted_pitch_at_response_rad=predicted_pitch,
        stable_center_norm=(stable_x, stable_y),
        residual_rate_norm_s=residual_rate_norm,
        expansion_rate_s=expansion,
        time_to_contact_s=ttc,
        horizontal_aligned=horizontal_aligned,
        requested_target_pitch_rad=requested_pitch,
        fov_protected_target_pitch_rad=fov_pitch,
        allocated_target_pitch_rad=allocated_pitch,
        requested_thrust=thrust,
        allocated_thrust=allocated_thrust,
        fresh_boundary_current_authority=boundary_authorized,
        forward_closure_authorized=False,
        steering_only=True,
        passage_authority=False,
        advance_authority=False,
    )


def _nonrapid_off_axis_top_fov_owns_pitch(
    *,
    mode: VisualApproachMode,
    fov_proposal: _TopFovPitchProposal,
    fresh_top_boundary: _FreshCurrentTopBoundaryAuthority,
    closure_recovery: _FreshTopCensoredClosureRecovery,
    rapid_expansion_rate_s: float,
    rapid_closure_ttc_s: float,
    retained_raw_handoff: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Arbitrate one fresh TOP frame without reviving urgent closure.

    The full positive-pitch brake remains authoritative for aligned, rapid,
    or ordinary contact-time-unknown approach closure.  During an off-axis,
    nonrapid approach, replacing the exact FOV-safe pitch reverses camera
    observability before the horizontal intercept can take effect.

    Post-credit recovery may also consume the already-existing fixed,
    nonrenewing retained-raw FOV lease.  That mode is itself bounded and
    needs a second clean accepted wire before release, so unknown TTC does
    not imply rapid closure when expansion remains below the planner's
    existing rapid threshold.  Successor-propagated and geometry-refusal
    paths never call this policy.
    """

    rapid_expansion, rapid_ttc = map(
        float,
        (rapid_expansion_rate_s, rapid_closure_ttc_s),
    )
    retained = retained_raw_handoff is not None
    if (
        type(mode) is not VisualApproachMode
        or type(fov_proposal) is not _TopFovPitchProposal
        or type(fresh_top_boundary) is not _FreshCurrentTopBoundaryAuthority
        or type(closure_recovery) is not _FreshTopCensoredClosureRecovery
        or retained
        and (
            not isinstance(retained_raw_handoff, Mapping)
            or retained_raw_handoff.get("basis")
            != TOP_FOV_RETAINED_RAW_STATE_BASIS
            or retained_raw_handoff.get("steering_only") is not True
            or retained_raw_handoff.get("passage_authority") is not False
            or retained_raw_handoff.get("advance_authority") is not False
            or type(
                retained_raw_handoff.get(
                    "prediction_horizon_remaining_s"
                )
            )
            not in {int, float}
            or not math.isfinite(
                float(
                    retained_raw_handoff[
                        "prediction_horizon_remaining_s"
                    ]
                )
            )
            or float(
                retained_raw_handoff[
                    "prediction_horizon_remaining_s"
                ]
            )
            <= 0.0
        )
        or not all(
            math.isfinite(value)
            for value in (rapid_expansion, rapid_ttc)
        )
        or rapid_expansion <= 0.0
        or rapid_ttc <= 0.0
    ):
        raise ValueError("TOP pitch arbitration inputs are invalid")

    ttc = closure_recovery.time_to_contact_s
    normal_exact_approach = bool(
        mode is VisualApproachMode.APPROACH
        and not retained
        and ttc is not None
    )
    bounded_post_credit_recovery = bool(
        mode is VisualApproachMode.PROMOTE_REACQUIRE
    )
    return bool(
        (normal_exact_approach or bounded_post_credit_recovery)
        and closure_recovery.fresh_boundary_current_authority
        and not closure_recovery.horizontal_aligned
        and closure_recovery.steering_only
        and not closure_recovery.forward_closure_authorized
        and not closure_recovery.passage_authority
        and not closure_recovery.advance_authority
        and fov_proposal.active_after
        and fov_proposal.limited
        and math.isclose(
            fov_proposal.requested_target_pitch_rad,
            closure_recovery.requested_target_pitch_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            fov_proposal.protected_target_pitch_rad,
            closure_recovery.fov_protected_target_pitch_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and fov_proposal.protected_target_pitch_rad
        < closure_recovery.allocated_target_pitch_rad - 1e-12
        and closure_recovery.expansion_rate_s < rapid_expansion
        and (ttc is None or ttc > rapid_ttc)
    )


def _top_fov_propagated_observation(
    authority: Mapping[str, Any],
) -> _TopFovPropagatedObservation:
    """Recover a conservative current-camera top edge from local gate state."""

    if (
        not isinstance(authority, Mapping)
        or authority.get("basis")
        != "propagated-current-fov-gap-steering-v1"
        or authority.get("steering_only") is not True
        or authority.get("passage_authority") is not False
        or authority.get("advance_authority") is not False
    ):
        raise ValueError(
            "propagated top-FOV authority is not steering-only"
        )
    center = authority.get("camera_center_norm")
    aperture = authority.get("camera_aperture_half_size_norm")
    center_std = authority.get("camera_center_std_norm")
    quaternion = authority.get("body_to_reference_wxyz")
    clipping_value = authority.get("clipping")
    if (
        not isinstance(center, (list, tuple))
        or len(center) != 2
        or not isinstance(aperture, (list, tuple))
        or len(aperture) != 2
        or not isinstance(center_std, (list, tuple))
        or len(center_std) != 2
        or not isinstance(quaternion, (list, tuple))
        or len(quaternion) != 4
        or type(clipping_value) is not int
    ):
        raise ValueError("propagated top-FOV geometry is malformed")
    center_x, center_y = map(float, center)
    aperture_x, aperture_y = map(float, aperture)
    center_std_x, center_std_y = map(float, center_std)
    log_scale_std = float(authority["aperture_log_scale_std"])
    vertical_scale = float(authority["vertical_angle_scale_rad"])
    remaining_horizon_s = float(
        authority["aperture_prediction_horizon_remaining_s"]
    )
    clipping = FrameEdge(clipping_value)
    vertical_edges = FrameEdge.TOP | FrameEdge.BOTTOM
    if (
        not all(
            math.isfinite(value)
            for value in (
                center_x,
                center_y,
                aperture_x,
                aperture_y,
                center_std_x,
                center_std_y,
                log_scale_std,
                vertical_scale,
                remaining_horizon_s,
                *map(float, quaternion),
            )
        )
        or aperture_x <= 0.0
        or aperture_y <= 0.0
        or center_std_x < 0.0
        or center_std_y < 0.0
        or log_scale_std < 0.0
        or vertical_scale <= 0.0
        or remaining_horizon_s <= 0.0
        or not bool(clipping & vertical_edges)
    ):
        raise ValueError("propagated top-FOV geometry is invalid")
    nominal_top = center_y - aperture_y
    top_std = math.sqrt(
        center_std_y * center_std_y
        + (aperture_y * log_scale_std) ** 2
    )
    conservative_top = max(
        -1.0,
        min(
            1.0,
            nominal_top - TOP_FOV_INNER_EDGE_SIGMA * top_std,
        ),
    )
    capture_pitch = _body_to_reference_pitch_rad(tuple(map(float, quaternion)))
    if not all(
        math.isfinite(value)
        for value in (
            nominal_top,
            top_std,
            conservative_top,
            capture_pitch,
        )
    ):
        raise ValueError("propagated top-FOV projection is nonfinite")
    return _TopFovPropagatedObservation(
        capture_pitch_rad=capture_pitch,
        projected_top_edge_image_down=conservative_top,
        projected_nominal_top_edge_image_down=nominal_top,
        projected_top_edge_std_image_down=top_std,
        vertical_angle_scale_rad=vertical_scale,
        prediction_horizon_remaining_s=remaining_horizon_s,
        geometry_basis=TOP_FOV_PROPAGATED_INNER_EDGE_BASIS,
    )


@dataclass(frozen=True, slots=True)
class _ApproachInnerDropoutAuthority:
    track_id: str
    anchor_camera_token: CameraFrameToken
    anchor_observation_monotonic_ns: int
    anchor_wire_start_monotonic_ns: int
    last_camera_token: CameraFrameToken
    age_s: float
    maximum_age_s: float
    maximum_target_pitch_rad: float


def _derive_approach_inner_dropout_authority(
    *,
    snapshot: Any,
    expected_gate_index: int,
    expected_track_id: str,
    maximum_age_s: float,
    now_monotonic_ns: int,
    fov_summary: Mapping[str, Any],
    existing: Optional[_ApproachInnerDropoutAuthority] = None,
) -> Optional[_ApproachInnerDropoutAuthority]:
    """Bound a vertical-support refusal to the last exact inner authority."""

    maximum_age = float(maximum_age_s)
    track = getattr(snapshot, "current_track", None)
    token = getattr(snapshot, "latest_camera_token", None)
    history = getattr(track, "history", None)
    if (
        type(expected_gate_index) is not int
        or expected_gate_index < 0
        or type(expected_track_id) is not str
        or not expected_track_id
        or not math.isfinite(maximum_age)
        or maximum_age <= 0.0
        or maximum_age
        > APPROACH_INNER_DROPOUT_MAX_DURATION_S + 1e-12
        or type(now_monotonic_ns) is not int
        or now_monotonic_ns < 0
        or not isinstance(fov_summary, Mapping)
        or getattr(snapshot, "current_gate_index", None)
        != expected_gate_index
        or getattr(snapshot, "current_track_id", None)
        != expected_track_id
        or getattr(snapshot, "authority_usable", False) is not True
        or track is None
        or getattr(track, "track_id", None) != expected_track_id
        or getattr(track, "latest_token", None) != token
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "visible", False) is not True
        or getattr(track, "ambiguous", True) is not False
        or getattr(track, "missed_frame_count", None) != 0
        or getattr(track, "clipping", None)
        not in {FrameEdge.TOP, FrameEdge.BOTTOM}
        or getattr(track, "center_censored", None) is not True
        or type(token) is not CameraFrameToken
        or type(history) is not tuple
        or len(history) < 2
    ):
        return None
    current_sample = history[-1]
    if (
        current_sample.token != token
        or type(current_sample.observation_monotonic_ns) is not int
        or current_sample.observation_monotonic_ns < 0
    ):
        return None
    current_inner = current_sample.inner_aperture
    if (
        type(current_inner) is VisualInnerApertureGeometry
        and current_inner.fitted
        and current_inner.clipping == FrameEdge.NONE
        and current_inner.complete_visibility
    ):
        return None

    if existing is None:
        anchor_token_value = fov_summary.get(
            "last_inner_camera_token"
        )
        anchor_index = next(
            (
                index
                for index, sample in enumerate(history[:-1])
                if type(getattr(sample, "token", None))
                is CameraFrameToken
                and isinstance(anchor_token_value, Mapping)
                and asdict(sample.token) == dict(anchor_token_value)
            ),
            None,
        )
        if anchor_index is None:
            return None
        anchor_sample = history[anchor_index]
        anchor_inner = anchor_sample.inner_aperture
        if (
            type(anchor_sample.token) is not CameraFrameToken
            or anchor_sample.token.stream_id != token.stream_id
            or anchor_sample.token.generation != token.generation
            or not _token_strictly_newer(token, anchor_sample.token)
            or type(anchor_sample.observation_monotonic_ns) is not int
            or anchor_sample.observation_monotonic_ns < 0
            or type(anchor_inner) is not VisualInnerApertureGeometry
            or not anchor_inner.fitted
            or anchor_inner.clipping != FrameEdge.NONE
            or not anchor_inner.complete_visibility
            or fov_summary.get("last_inner_active") is not True
            or fov_summary.get("last_inner_track_id")
            != expected_track_id
            or fov_summary.get("last_inner_raw_top_edge_basis")
            != TOP_FOV_INNER_EDGE_BASIS
            or type(
                fov_summary.get(
                    "last_inner_wire_start_monotonic_ns"
                )
            )
            is not int
        ):
            return None
        previous_sample = anchor_sample
        for dropout_sample in history[anchor_index + 1 :]:
            dropout_inner = dropout_sample.inner_aperture
            if (
                type(dropout_sample.token) is not CameraFrameToken
                or dropout_sample.token.stream_id != token.stream_id
                or dropout_sample.token.generation != token.generation
                or not _token_strictly_newer(
                    dropout_sample.token,
                    previous_sample.token,
                )
                or type(
                    dropout_sample.observation_monotonic_ns
                )
                is not int
                or dropout_sample.observation_monotonic_ns
                <= previous_sample.observation_monotonic_ns
                or dropout_sample.clipping
                not in {
                    FrameEdge.NONE,
                    FrameEdge.TOP,
                    FrameEdge.BOTTOM,
                }
                or (
                    dropout_sample.clipping == FrameEdge.NONE
                    and dropout_sample.center_censored
                )
                or (
                    dropout_sample.clipping != FrameEdge.NONE
                    and not dropout_sample.center_censored
                )
                or (
                    type(dropout_inner)
                    is VisualInnerApertureGeometry
                    and dropout_inner.fitted
                    and dropout_inner.clipping == FrameEdge.NONE
                    and dropout_inner.complete_visibility
                )
            ):
                return None
            previous_sample = dropout_sample
        maximum_target_pitch = float(
            fov_summary.get(
                "last_inner_protected_target_pitch_rad",
                math.nan,
            )
        )
        anchor_token = anchor_sample.token
        anchor_observation_ns = anchor_sample.observation_monotonic_ns
        anchor_wire_start_ns = int(
            fov_summary["last_inner_wire_start_monotonic_ns"]
        )
    else:
        if (
            type(existing) is not _ApproachInnerDropoutAuthority
            or existing.track_id != expected_track_id
            or existing.anchor_camera_token.stream_id != token.stream_id
            or existing.anchor_camera_token.generation != token.generation
            or not _token_strictly_newer(token, existing.last_camera_token)
            or history[-2].token != existing.last_camera_token
            or abs(existing.maximum_age_s - maximum_age) > 1e-12
        ):
            return None
        maximum_target_pitch = existing.maximum_target_pitch_rad
        anchor_token = existing.anchor_camera_token
        anchor_observation_ns = (
            existing.anchor_observation_monotonic_ns
        )
        anchor_wire_start_ns = (
            existing.anchor_wire_start_monotonic_ns
        )
    observation_age_s = (
        current_sample.observation_monotonic_ns - anchor_observation_ns
    ) / 1_000_000_000.0
    wall_age_s = (
        now_monotonic_ns - anchor_wire_start_ns
    ) / 1_000_000_000.0
    age_s = max(observation_age_s, wall_age_s)
    if (
        not MIN_VISUAL_TARGET_PITCH_RAD
        <= maximum_target_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not math.isfinite(observation_age_s)
        or observation_age_s <= 0.0
        or observation_age_s > maximum_age
        or not math.isfinite(wall_age_s)
        or wall_age_s < 0.0
        or wall_age_s > maximum_age
        or not math.isfinite(age_s)
    ):
        return None
    return _ApproachInnerDropoutAuthority(
        track_id=expected_track_id,
        anchor_camera_token=anchor_token,
        anchor_observation_monotonic_ns=anchor_observation_ns,
        anchor_wire_start_monotonic_ns=anchor_wire_start_ns,
        last_camera_token=token,
        age_s=age_s,
        maximum_age_s=maximum_age,
        maximum_target_pitch_rad=maximum_target_pitch,
    )


def _propose_top_fov_pitch_reference(
    *,
    capture_pitch_rad: float,
    raw_top_edge_image_down: float,
    raw_top_edge_rate_down_s: Optional[float],
    requested_target_pitch_rad: float,
    prior_target_pitch_rad: float,
    vertical_angle_scale_rad: float,
    active_before: bool,
    raw_top_edge_nonrotational_angle_rate_rad_s: Optional[float] = None,
    prediction_horizon_s: float = 0.0,
) -> _TopFovPitchProposal:
    """Hold nose-up authority until raw top-edge clearance recovers."""

    capture, raw_top, requested, prior, scale = map(
        float,
        (
            capture_pitch_rad,
            raw_top_edge_image_down,
            requested_target_pitch_rad,
            prior_target_pitch_rad,
            vertical_angle_scale_rad,
        ),
    )
    rate = (
        None
        if raw_top_edge_rate_down_s is None
        else float(raw_top_edge_rate_down_s)
    )
    nonrotational_rate = (
        None
        if raw_top_edge_nonrotational_angle_rate_rad_s is None
        else float(raw_top_edge_nonrotational_angle_rate_rad_s)
    )
    horizon = float(prediction_horizon_s)
    if (
        not all(
            math.isfinite(value)
            for value in (
                capture,
                raw_top,
                requested,
                prior,
                scale,
            )
        )
        or rate is not None
        and not math.isfinite(rate)
        or nonrotational_rate is not None
        and not math.isfinite(nonrotational_rate)
        or not math.isfinite(horizon)
        or horizon < 0.0
        or not -1.0 <= raw_top <= 1.0
        or scale <= 0.0
        or type(active_before) is not bool
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= requested
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= prior
        <= MAX_VISUAL_TARGET_PITCH_RAD
    ):
        raise ValueError("top-FOV pitch guidance inputs are invalid")
    current_top_angle = math.atan(raw_top * scale)
    forecast_top_angle = (
        current_top_angle
        + min(0.0, nonrotational_rate or 0.0) * horizon
    )
    forecast_top_angle = max(
        math.atan(-scale),
        min(math.atan(scale), forecast_top_angle),
    )
    forecast_top = math.tan(forecast_top_angle) / scale
    maximum_observable = (
        capture
        + math.atan(forecast_top * scale)
        - math.atan(TOP_FOV_SAFE_EDGE_IMAGE_DOWN * scale)
    )
    maximum_observable = min(
        MAX_VISUAL_TARGET_PITCH_RAD,
        maximum_observable,
    )
    if not math.isfinite(maximum_observable):
        raise ValueError("top-FOV pitch ceiling is nonfinite")
    envelope_saturated = (
        maximum_observable < MIN_VISUAL_TARGET_PITCH_RAD
    )
    bounded_ceiling = max(
        MIN_VISUAL_TARGET_PITCH_RAD,
        maximum_observable,
    )
    recovery_rate = (
        nonrotational_rate
        if nonrotational_rate is not None
        else rate
    )
    recovering = bool(
        recovery_rate is not None and recovery_rate > 0.0
    )
    exceeds = requested > maximum_observable + 1e-12
    active_after = active_before
    protected = requested
    if active_before:
        if recovering and not exceeds:
            active_after = False
        else:
            protected = min(
                requested,
                bounded_ceiling,
                bounded_ceiling if recovering else prior,
            )
            active_after = True
    elif exceeds:
        protected = min(
            requested,
            bounded_ceiling,
            bounded_ceiling if recovering else capture,
        )
        active_after = True
    predicted_requested = _project_raw_vertical_edge_for_pitch_reference(
        edge_image_down=forecast_top,
        capture_pitch_rad=capture,
        target_pitch_rad=requested,
        vertical_angle_scale_rad=scale,
    )
    predicted_protected = _project_raw_vertical_edge_for_pitch_reference(
        edge_image_down=forecast_top,
        capture_pitch_rad=capture,
        target_pitch_rad=protected,
        vertical_angle_scale_rad=scale,
    )
    if (
        not MIN_VISUAL_TARGET_PITCH_RAD
        <= protected
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or active_after
        and not envelope_saturated
        and predicted_protected
        < TOP_FOV_SAFE_EDGE_IMAGE_DOWN - 1e-9
    ):
        raise ValueError("top-FOV pitch guidance escaped its geometry")
    return _TopFovPitchProposal(
        raw_top_edge_image_down=raw_top,
        raw_top_edge_rate_down_s=rate,
        raw_top_edge_nonrotational_angle_rate_rad_s=nonrotational_rate,
        prediction_horizon_s=horizon,
        forecast_top_edge_image_down=forecast_top,
        capture_pitch_rad=capture,
        requested_target_pitch_rad=requested,
        maximum_observable_target_pitch_rad=maximum_observable,
        protected_target_pitch_rad=protected,
        predicted_requested_top_edge_image_down=predicted_requested,
        predicted_protected_top_edge_image_down=predicted_protected,
        clearance_recovering=recovering,
        envelope_saturated=envelope_saturated,
        active_before=active_before,
        active_after=active_after,
        limited=protected < requested - 1e-12,
    )


def _propose_propagated_top_fov_pitch_reference(
    authority: Mapping[str, Any],
    *,
    requested_target_pitch_rad: float,
    prior_target_pitch_rad: Optional[float],
) -> tuple[_TopFovPropagatedObservation, _TopFovPitchProposal]:
    """Keep the current raw-FOV constraint active through a predicted gap."""

    observation = _top_fov_propagated_observation(authority)
    prior = (
        observation.capture_pitch_rad
        if prior_target_pitch_rad is None
        else float(prior_target_pitch_rad)
    )
    proposal = _propose_top_fov_pitch_reference(
        capture_pitch_rad=observation.capture_pitch_rad,
        raw_top_edge_image_down=(
            observation.projected_top_edge_image_down
        ),
        raw_top_edge_rate_down_s=None,
        requested_target_pitch_rad=requested_target_pitch_rad,
        prior_target_pitch_rad=prior,
        vertical_angle_scale_rad=observation.vertical_angle_scale_rad,
        # A censored local-state projection cannot establish raw clearance
        # recovery.  Keep the state-dependent ceiling engaged until a raw
        # observation proves recovery or near-plane passage owns transition.
        active_before=True,
        raw_top_edge_nonrotational_angle_rate_rad_s=None,
        prediction_horizon_s=0.0,
    )
    return observation, proposal


def _retain_post_credit_top_fov_pitch_reference(
    authority: Mapping[str, Any],
    fov_summary: Mapping[str, Any],
) -> Optional[tuple[float, Mapping[str, Any]]]:
    """Retain an accepted raw-FOV ceiling through a blind successor gap."""

    if not isinstance(authority, Mapping) or not isinstance(
        fov_summary,
        Mapping,
    ):
        raise ValueError("retained post-credit TOP-FOV inputs are invalid")
    if fov_summary.get("active") is not True:
        return None
    retained_track_id = fov_summary.get("last_track_id")
    retained_pitch = fov_summary.get(
        "last_protected_target_pitch_rad"
    )
    requested_pitch = float(authority["target_pitch_rad"])
    if (
        retained_track_id != authority.get("reviewed_track_id")
        or type(retained_pitch) not in {int, float}
        or not math.isfinite(float(retained_pitch))
        or not math.isfinite(requested_pitch)
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= float(retained_pitch)
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= requested_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
    ):
        raise ValueError(
            "retained post-credit TOP-FOV authority is invalid"
        )
    protected_pitch = min(requested_pitch, float(retained_pitch))
    return protected_pitch, {
        "basis": TOP_FOV_PITCH_PROTECTION_BASIS,
        "track_id": retained_track_id,
        "safe_top_edge_image_down": TOP_FOV_SAFE_EDGE_IMAGE_DOWN,
        "requested_target_pitch_rad": requested_pitch,
        "protected_target_pitch_rad": protected_pitch,
        "retained_through_missing_frame": True,
        "active_before": True,
        "active_after": True,
        "limited": protected_pitch < requested_pitch - 1e-12,
        "steering_only": True,
        "passage_authority": False,
        "advance_authority": False,
    }


def _retain_fresh_top_censored_closure_recovery(
    *,
    authority: Mapping[str, Any],
    fov_summary: Mapping[str, Any],
    recovery_snapshot: Any,
    current_gate_index: int,
    now_monotonic_ns: int,
    requested_target_pitch_rad: float,
    requested_thrust: float,
) -> Optional[_RetainedFreshTopCensoredClosureRecovery]:
    """Keep an accepted exact-TOP brake through its retained-current gap.

    The fresh boundary itself is not extrapolated.  This only retains the
    already accepted non-forward pitch/collective allocation while the exact
    authoritative current identity is invisible, its last track publication
    remains the source boundary, and the existing post-credit steering lease
    remains live.  A new association, ambiguity, identity change, or lease
    expiry removes this authority.
    """

    if not isinstance(authority, Mapping) or not isinstance(
        fov_summary,
        Mapping,
    ):
        raise ValueError("retained fresh TOP recovery inputs are invalid")
    source = fov_summary.get("last_exact_top_closure_recovery")
    if source is None:
        return None
    if not isinstance(source, Mapping):
        raise ValueError("retained fresh TOP recovery source is invalid")

    source_token_fields = source.get("camera_token")
    if not isinstance(source_token_fields, Mapping):
        raise ValueError("retained fresh TOP recovery token is invalid")
    try:
        source_token = CameraFrameToken(**dict(source_token_fields))
        current_token = recovery_snapshot.latest_camera_token
        track = recovery_snapshot.current_track
        source_wire_ns = int(source["source_wire_start_monotonic_ns"])
        expires_ns = int(source["expires_monotonic_ns"])
        pitch_floor = float(source["target_pitch_floor_rad"])
        thrust_floor = float(source["thrust_floor"])
        requested_pitch = float(requested_target_pitch_rad)
        thrust = float(requested_thrust)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "retained fresh TOP recovery source is malformed"
        ) from exc

    authority_expires_ns = authority.get("expires_monotonic_ns")
    missed_count = getattr(track, "missed_frame_count", None)
    retained_current_loss = bool(
        getattr(recovery_snapshot, "authority_usable", None) is False
        and getattr(recovery_snapshot, "withholding_reason", None)
        == "current_track_not_visible"
        and getattr(track, "latest_token", None) == source_token
        and getattr(track, "visible", None) is False
        and getattr(track, "ambiguous", None) is False
        and type(missed_count) is int
        and missed_count > 0
    )
    if now_monotonic_ns >= expires_ns or not retained_current_loss:
        return None
    if (
        source.get("basis")
        != RETAINED_FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS
        or source.get("source_basis")
        != FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS
        or source.get("gate_index") != current_gate_index
        or source.get("track_id") != authority.get("reviewed_track_id")
        or source.get("steering_only") is not True
        or source.get("passage_authority") is not False
        or source.get("advance_authority") is not False
        or type(current_gate_index) is not int
        or current_gate_index < 0
        or type(now_monotonic_ns) is not int
        or now_monotonic_ns < 0
        or source_wire_ns < 0
        or expires_ns <= source_wire_ns
        or type(authority_expires_ns) is not int
        or authority_expires_ns != expires_ns
        or authority.get("to_gate_index") != current_gate_index
        or authority.get("stream_generation") != source_token.generation
        or authority.get("steering_available") is not True
        or authority.get("steering_only") is not True
        or authority.get("passage_authority") is not False
        or authority.get("advance_authority") is not False
        or getattr(recovery_snapshot, "current_gate_index", None)
        != current_gate_index
        or getattr(recovery_snapshot, "current_track_id", None)
        != source.get("track_id")
        or getattr(recovery_snapshot, "race_finished", None) is not False
        or type(current_token) is not CameraFrameToken
        or not _token_strictly_newer(current_token, source_token)
        or getattr(track, "track_id", None) != source.get("track_id")
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "authoritative_gate_index", None)
        != current_gate_index
        or getattr(track, "clipping", None) is not FrameEdge.TOP
        or getattr(track, "center_censored", None) is not True
        or not all(
            math.isfinite(value)
            for value in (
                pitch_floor,
                thrust_floor,
                requested_pitch,
                thrust,
            )
        )
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= pitch_floor
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= requested_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not MIN_VISUAL_THRUST <= thrust_floor <= MAX_VISUAL_THRUST
        or not MIN_VISUAL_THRUST <= thrust <= MAX_VISUAL_THRUST
    ):
        raise ValueError("retained fresh TOP recovery authority is invalid")

    allocated_pitch = max(requested_pitch, pitch_floor)
    allocated_thrust = max(thrust, thrust_floor)
    return _RetainedFreshTopCensoredClosureRecovery(
        basis=RETAINED_FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS,
        source_basis=FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS,
        gate_index=current_gate_index,
        track_id=str(source["track_id"]),
        source_camera_token=source_token,
        current_camera_token=current_token,
        source_wire_start_monotonic_ns=source_wire_ns,
        expires_monotonic_ns=expires_ns,
        missed_frame_count=missed_count,
        requested_target_pitch_rad=requested_pitch,
        retained_target_pitch_floor_rad=pitch_floor,
        allocated_target_pitch_rad=allocated_pitch,
        requested_thrust=thrust,
        retained_thrust_floor=thrust_floor,
        allocated_thrust=allocated_thrust,
        retained_through_missing_frame=True,
        forward_closure_authorized=False,
        steering_only=True,
        passage_authority=False,
        advance_authority=False,
    )


def _propose_retained_raw_top_fov_pitch_reference(
    session: DynamicVisualCourseSession,
    target_track: Any,
    camera_token: CameraFrameToken,
    *,
    fov_summary: Mapping[str, Any],
    now_monotonic_ns: int,
    requested_target_pitch_rad: float,
) -> tuple[
    _TopFovRetainedRawStateObservation,
    _TopFovPitchProposal,
    Mapping[str, Any],
]:
    """Propagate one exact raw top edge through bounded vertical clipping.

    The immutable anchor is corrected into the current camera with measured
    pitch.  Only adverse residual edge motion is extrapolated, and the
    retained edge grows more conservative with the dynamic-state rate and
    capture-timing uncertainty.  This state supplies observability steering
    only; it cannot establish passage or gate advance.
    """

    if (
        type(camera_token) is not CameraFrameToken
        or type(now_monotonic_ns) is not int
        or now_monotonic_ns < 0
        or not isinstance(fov_summary, Mapping)
    ):
        raise ValueError("retained top-FOV state inputs are invalid")
    anchor = fov_summary.get("exact_raw_anchor")
    if (
        not isinstance(anchor, Mapping)
        or anchor.get("basis") != TOP_FOV_EXACT_RAW_ANCHOR_BASIS
        or anchor.get("steering_only") is not True
        or anchor.get("passage_authority") is not False
        or anchor.get("advance_authority") is not False
        or fov_summary.get("active") is not True
    ):
        raise ValueError("retained top-FOV state lacks an active raw anchor")

    course = session.core.course_state()
    current = course.current
    config = session.core.config
    track_id = getattr(target_track, "track_id", None)
    history = getattr(target_track, "history", None)
    current_sample = None if not history else history[-1]
    vertical_edges = FrameEdge.TOP | FrameEdge.BOTTOM
    if (
        type(track_id) is not str
        or not track_id
        or course.current_track_id != track_id
        or current.track_id != track_id
        or tuple(config.camera_to_body_wxyz)
        != BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ
        or type(history) is not tuple
        or len(history) < 2
        or current_sample is None
        or current_sample.token != camera_token
        or getattr(target_track, "latest_token", None) != camera_token
        or getattr(target_track, "role", None)
        is not VisualTrackRole.CURRENT
        or getattr(target_track, "visible", False) is not True
        or getattr(target_track, "ambiguous", True) is not False
        or getattr(target_track, "missed_frame_count", None) != 0
        or not bool(getattr(target_track, "clipping", FrameEdge.NONE) & vertical_edges)
        or getattr(target_track, "center_censored", None) is not True
        or current.frame_sequence
        != current_sample.tracker_frame_sequence
        or current.stream_generation != camera_token.generation
        or current.clipping != current_sample.clipping
        or current.clipping != getattr(target_track, "clipping", None)
        or not current.visible
        or current.ambiguous
    ):
        raise ValueError(
            "retained top-FOV state lacks exact vertically clipped lineage"
        )

    anchor_token_value = anchor.get("camera_token")
    anchor_index = next(
        (
            index
            for index, sample in enumerate(history[:-1])
            if isinstance(anchor_token_value, Mapping)
            and asdict(sample.token) == dict(anchor_token_value)
        ),
        None,
    )
    if anchor_index is None:
        raise ValueError("retained top-FOV anchor is outside track history")
    anchor_sample = history[anchor_index]
    try:
        anchor_edge = _top_fov_raw_edge(anchor_sample)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "retained top-FOV anchor lacks exact clean raw geometry"
        ) from exc
    anchor_observation_ns = anchor.get("observation_monotonic_ns")
    anchor_wire_start_ns = anchor.get("wire_start_monotonic_ns")
    anchor_capture_pitch = float(
        anchor.get("capture_pitch_rad", math.nan)
    )
    anchor_top = float(anchor.get("raw_top_edge_image_down", math.nan))
    anchor_nominal_top = float(
        anchor.get("raw_nominal_top_edge_image_down", math.nan)
    )
    anchor_top_std = float(
        anchor.get("raw_top_edge_std_image_down", math.nan)
    )
    prior_protected_pitch = float(
        anchor.get("protected_target_pitch_rad", math.nan)
    )
    nonrotational_rate_value = anchor.get(
        "raw_top_edge_nonrotational_angle_rate_rad_s"
    )
    nonrotational_rate = (
        None
        if nonrotational_rate_value is None
        else float(nonrotational_rate_value)
    )
    # This is uncertainty-growing local gate state, not a retained command.
    # Keep it on the existing bounded current-state prediction horizon rather
    # than expiring it with the much shorter one-frame command-dropout hold.
    maximum_age_s = float(
        config.post_credit_current_prediction_max_horizon_s
    )
    if (
        type(anchor_observation_ns) is not int
        or anchor_observation_ns < 0
        or type(anchor_wire_start_ns) is not int
        or anchor_wire_start_ns < 0
        or anchor.get("track_id") != track_id
        or anchor.get("raw_top_edge_basis") != anchor_edge.basis
        or not all(
            math.isfinite(value)
            for value in (
                anchor_capture_pitch,
                anchor_top,
                anchor_nominal_top,
                anchor_top_std,
                prior_protected_pitch,
                maximum_age_s,
            )
        )
        or nonrotational_rate is not None
        and not math.isfinite(nonrotational_rate)
        or maximum_age_s <= 0.0
        or not math.isclose(
            anchor_top,
            anchor_edge.top_edge_image_down,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            anchor_nominal_top,
            anchor_edge.nominal_top_edge_image_down,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            anchor_top_std,
            anchor_edge.top_edge_std_image_down,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= prior_protected_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
    ):
        raise ValueError("retained top-FOV anchor evidence differs")

    previous_sample = anchor_sample
    for clipped_sample in history[anchor_index + 1 :]:
        if (
            type(clipped_sample.token) is not CameraFrameToken
            or not _token_strictly_newer(
                clipped_sample.token,
                previous_sample.token,
            )
            or type(clipped_sample.observation_monotonic_ns) is not int
            or clipped_sample.observation_monotonic_ns
            <= previous_sample.observation_monotonic_ns
            or not bool(clipped_sample.clipping & vertical_edges)
            or clipped_sample.center_censored is not True
        ):
            raise ValueError(
                "retained top-FOV clipped history is discontinuous"
            )
        try:
            _top_fov_raw_edge(clipped_sample)
        except (AttributeError, TypeError, ValueError):
            pass
        else:
            raise ValueError(
                "retained top-FOV state bypasses recoverable raw geometry"
            )
        previous_sample = clipped_sample

    current_observation_ns = current_sample.observation_monotonic_ns
    observation_age_s = (
        current_observation_ns - anchor_observation_ns
    ) / 1_000_000_000.0
    wall_age_s = (
        now_monotonic_ns - anchor_wire_start_ns
    ) / 1_000_000_000.0
    state_age_s = max(observation_age_s, wall_age_s)
    if (
        not math.isfinite(observation_age_s)
        or observation_age_s <= 0.0
        or observation_age_s > maximum_age_s
        or not math.isfinite(wall_age_s)
        or wall_age_s < 0.0
        or wall_age_s > maximum_age_s
        or not math.isfinite(state_age_s)
    ):
        raise ValueError("retained top-FOV state horizon expired")

    capture_pitch = _body_to_reference_pitch_rad(
        current.body_to_reference_wxyz
    )
    vertical_scale = float(config.vertical_angle_scale_rad)
    rate_std = float(current.rate_std_rad_s[1])
    pitch_rate = float(current.body_rates_rad_s[1])
    timing_uncertainty_s = float(current.capture_timing_uncertainty_s)
    process_noise_rate = float(config.process_noise_bearing_rad_s)
    if (
        not all(
            math.isfinite(value)
            for value in (
                capture_pitch,
                vertical_scale,
                rate_std,
                pitch_rate,
                timing_uncertainty_s,
                process_noise_rate,
            )
        )
        or vertical_scale <= 0.0
        or rate_std < 0.0
        or timing_uncertainty_s < 0.0
        or process_noise_rate < 0.0
    ):
        raise ValueError("retained top-FOV dynamic uncertainty is invalid")
    adverse_rate = min(0.0, nonrotational_rate or 0.0)
    pitch_delta = capture_pitch - anchor_capture_pitch
    uncertainty_growth_rad = TOP_FOV_INNER_EDGE_SIGMA * math.sqrt(
        (rate_std * observation_age_s) ** 2
        + (abs(pitch_rate) * timing_uncertainty_s) ** 2
        + (process_noise_rate * observation_age_s) ** 2
    )
    angle_min = math.atan(-vertical_scale)
    angle_max = math.atan(vertical_scale)
    projected_nominal_angle = (
        math.atan(anchor_nominal_top * vertical_scale)
        + adverse_rate * observation_age_s
        - pitch_delta
    )
    projected_top_angle = (
        math.atan(anchor_top * vertical_scale)
        + adverse_rate * observation_age_s
        - pitch_delta
        - uncertainty_growth_rad
    )
    projected_nominal_angle = max(
        angle_min,
        min(angle_max, projected_nominal_angle),
    )
    projected_top_angle = max(
        angle_min,
        min(angle_max, projected_top_angle),
    )
    projected_nominal_top = (
        math.tan(projected_nominal_angle) / vertical_scale
    )
    projected_top = math.tan(projected_top_angle) / vertical_scale
    remaining_horizon_s = maximum_age_s - state_age_s
    if (
        not all(
            math.isfinite(value)
            for value in (
                projected_nominal_top,
                projected_top,
                uncertainty_growth_rad,
                remaining_horizon_s,
            )
        )
        or remaining_horizon_s <= 0.0
    ):
        raise ValueError("retained top-FOV projection is invalid")

    observation = _TopFovRetainedRawStateObservation(
        anchor_camera_token=anchor_sample.token,
        camera_token=camera_token,
        anchor_capture_pitch_rad=anchor_capture_pitch,
        capture_pitch_rad=capture_pitch,
        projected_top_edge_image_down=projected_top,
        projected_nominal_top_edge_image_down=projected_nominal_top,
        projected_uncertainty_growth_rad=uncertainty_growth_rad,
        raw_top_edge_nonrotational_angle_rate_rad_s=(
            nonrotational_rate
        ),
        vertical_angle_scale_rad=vertical_scale,
        observation_age_s=observation_age_s,
        wall_age_s=wall_age_s,
        prediction_horizon_remaining_s=remaining_horizon_s,
        geometry_basis=TOP_FOV_RETAINED_RAW_STATE_BASIS,
    )
    proposal = _propose_top_fov_pitch_reference(
        capture_pitch_rad=capture_pitch,
        raw_top_edge_image_down=projected_top,
        raw_top_edge_rate_down_s=None,
        requested_target_pitch_rad=requested_target_pitch_rad,
        prior_target_pitch_rad=prior_protected_pitch,
        vertical_angle_scale_rad=vertical_scale,
        # A censored state projection cannot establish recovery.
        active_before=True,
        raw_top_edge_nonrotational_angle_rate_rad_s=(
            nonrotational_rate
        ),
        prediction_horizon_s=float(config.pitch_command_delay_s),
    )
    evidence = {
        "basis": TOP_FOV_RETAINED_RAW_STATE_BASIS,
        "gate_index": course.current_gate_index,
        "track_id": track_id,
        "anchor_camera_token": asdict(anchor_sample.token),
        "camera_token": asdict(camera_token),
        "anchor_observation_monotonic_ns": anchor_observation_ns,
        "anchor_wire_start_monotonic_ns": anchor_wire_start_ns,
        "authority_monotonic_ns": now_monotonic_ns,
        "maximum_age_s": maximum_age_s,
        "observation_age_s": observation_age_s,
        "wall_age_s": wall_age_s,
        "prediction_horizon_remaining_s": remaining_horizon_s,
        "steering_only": True,
        "passage_authority": False,
        "advance_authority": False,
    }
    return observation, proposal, evidence


def _top_fov_observation(
    session: DynamicVisualCourseSession,
    target_track: Any,
    camera_token: CameraFrameToken,
) -> _TopFovObservation:
    """Return exact raw-edge/capture-pitch guidance inputs."""

    course = session.core.course_state()
    current = course.current
    config = session.core.config
    track_id = getattr(target_track, "track_id", None)
    history = getattr(target_track, "history", None)
    if (
        course.current_track_id != track_id
        or current.track_id != track_id
        or tuple(config.camera_to_body_wxyz)
        != BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ
        or type(history) is not tuple
        or not history
    ):
        raise ValueError("top-FOV authority is not the calibrated current")
    sample = history[-1]
    if (
        sample.token != camera_token
        or sample.token != getattr(target_track, "latest_token", None)
        or current.frame_sequence != sample.tracker_frame_sequence
        or getattr(target_track, "clipping", None) != sample.clipping
        or getattr(target_track, "center_censored", None)
        != sample.center_censored
    ):
        raise ValueError("top-FOV authority lacks exact current raw geometry")
    edge = _top_fov_raw_edge(sample)
    top = edge.top_edge_image_down
    observed_ns = sample.observation_monotonic_ns
    if type(observed_ns) is not int or observed_ns < 0:
        raise ValueError("top-FOV observation clock is invalid")
    top_rate: Optional[float] = None
    top_motion_angle_rate: Optional[float] = None
    nonrotational_angle_rate: Optional[float] = None
    if len(history) >= 2:
        previous = history[-2]
        previous_ns = previous.observation_monotonic_ns
        if (
            type(previous_ns) is not int
            or previous_ns < 0
            or previous_ns >= observed_ns
        ):
            raise ValueError("top-FOV bbox history clock did not advance")
        try:
            previous_edge = _top_fov_raw_edge(previous)
        except (AttributeError, TypeError, ValueError):
            previous_edge = None
        if previous_edge is not None and previous_edge.basis == edge.basis:
            elapsed_s = (
                (observed_ns - previous_ns) / 1_000_000_000.0
            )
            top_rate = _top_fov_edge_recovery_rate_down_s(
                current=edge,
                previous=previous_edge,
                elapsed_s=elapsed_s,
            )
            vertical_scale = float(config.vertical_angle_scale_rad)
            top_motion_angle_rate = (
                math.atan(
                    edge.nominal_top_edge_image_down * vertical_scale
                )
                - math.atan(
                    previous_edge.nominal_top_edge_image_down
                    * vertical_scale
                )
            ) / elapsed_s
            body_rates = current.body_rates_rad_s
            if type(body_rates) is not tuple or len(body_rates) != 3:
                raise ValueError("top-FOV measured body rates are invalid")
            pitch_rate = float(body_rates[1])
            nonrotational_angle_rate = (
                _top_fov_nonrotational_angle_rate_rad_s(
                    current_top_edge_image_down=(
                        edge.nominal_top_edge_image_down
                    ),
                    previous_top_edge_image_down=(
                        previous_edge.nominal_top_edge_image_down
                    ),
                    vertical_angle_scale_rad=vertical_scale,
                    elapsed_s=elapsed_s,
                    measured_pitch_rate_rad_s=pitch_rate,
                )
            )
            if not all(
                math.isfinite(value)
                for value in (
                    top_motion_angle_rate,
                    pitch_rate,
                    nonrotational_angle_rate,
                )
            ):
                raise ValueError("top-FOV edge motion is nonfinite")
    previous_target = (
        None
        if course.last_applied_command is None
        else course.last_applied_command.target_pitch_rad
    )
    return _TopFovObservation(
        capture_pitch_rad=_body_to_reference_pitch_rad(
            current.body_to_reference_wxyz
        ),
        raw_top_edge_image_down=top,
        raw_nominal_top_edge_image_down=(
            edge.nominal_top_edge_image_down
        ),
        raw_top_edge_std_image_down=edge.top_edge_std_image_down,
        raw_top_edge_rate_down_s=top_rate,
        raw_top_edge_motion_angle_rate_rad_s=top_motion_angle_rate,
        raw_top_edge_nonrotational_angle_rate_rad_s=(
            nonrotational_angle_rate
        ),
        vertical_angle_scale_rad=float(config.vertical_angle_scale_rad),
        pitch_response_delay_s=float(config.pitch_command_delay_s),
        previous_target_pitch_rad=previous_target,
        raw_top_edge_basis=edge.basis,
        raw_top_edge_confidence=edge.confidence,
    )


def _allocate_launch_pitch_target(
    *,
    spawn_pitch_rad: float,
    responsive_target_pitch_rad: float,
    launch_elapsed_s: float,
) -> tuple[float, float]:
    """Generate the initial launch attitude reference from measured spawn.

    The destination remains the current responsive visual demand on every
    tick.  Distance traveled along the reference is determined only by launch
    elapsed time, with zero initial slope and the live-credited bounded
    acceleration/rate profile.  There is no prior-command or slew state; the
    final wire governor remains the sole command-continuity authority.
    """

    values = (
        float(spawn_pitch_rad),
        float(responsive_target_pitch_rad),
        float(launch_elapsed_s),
    )
    if (
        not all(math.isfinite(value) for value in values)
        or values[2] < 0.0
    ):
        raise ValueError("launch pitch allocation inputs are invalid")
    distance = abs(values[1] - values[0])
    if distance <= 1e-12:
        return values[1], 1.0
    acceleration = LAUNCH_PITCH_REFERENCE_ACCEL_RAD_S2
    maximum_rate = LAUNCH_PITCH_REFERENCE_MAX_RATE_RAD_S
    acceleration_duration_s = maximum_rate / acceleration
    acceleration_distance_rad = (
        0.5
        * acceleration
        * acceleration_duration_s
        * acceleration_duration_s
    )
    elapsed_s = values[2]
    if elapsed_s <= acceleration_duration_s:
        traveled_rad = 0.5 * acceleration * elapsed_s * elapsed_s
    else:
        traveled_rad = (
            acceleration_distance_rad
            + maximum_rate
            * (elapsed_s - acceleration_duration_s)
        )
    blend = min(1.0, traveled_rad / distance)
    return (
        (1.0 - blend) * values[0] + blend * values[1],
        blend,
    )


def _pitch_response_authority(
    *,
    allocated_target_pitch_rad: float,
    intercept_response_authority: float,
) -> float:
    """Allocate static attitude-loop response from the applied reference.

    A launch destination that has not yet been allocated cannot silently
    increase the inner attitude-loop gain.  The reference itself remains
    responsive, while the final wire governor owns command continuity.
    """

    target = float(allocated_target_pitch_rad)
    intercept = float(intercept_response_authority)
    if (
        not math.isfinite(target)
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= target
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or not math.isfinite(intercept)
        or not 0.0 <= intercept <= 1.0
    ):
        raise ValueError("pitch response authority inputs are invalid")
    return max(
        intercept,
        max(
            0.0,
            min(1.0, target / MAX_VISUAL_TARGET_PITCH_RAD),
        ),
    )


def _roll_yaw_transport_rate_rad_s(
    *,
    target_roll_rad: float,
    target_pitch_rad: float,
    bounded_yaw_rate_rad_s: float,
) -> float:
    """Keep a bank reference fixed while its yaw chart rotates.

    For fixed roll/pitch Euler references, a desired body-yaw rate requires
    ``p = -r*tan(pitch)/cos(roll)``.  The visual attitude loop previously
    treated the corresponding body-roll transport as roll motion to damp,
    leaving a persistent bank error during the simultaneous Gate-1 turn.
    Allocate the instantaneous desired-reference term here; the existing
    static command clamp and final wire governor remain the only
    envelope/continuity authorities.
    """

    roll = float(target_roll_rad)
    pitch = float(target_pitch_rad)
    yaw_rate = float(bounded_yaw_rate_rad_s)
    if (
        not all(math.isfinite(value) for value in (roll, pitch, yaw_rate))
        or abs(roll) > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or abs(yaw_rate) > MAX_VISUAL_YAW_RATE_RAD_S + 1e-12
    ):
        raise ValueError("roll/yaw transport inputs are invalid")
    roll_cosine = math.cos(roll)
    if not math.isfinite(roll_cosine) or roll_cosine <= 0.0:
        raise ValueError("roll/yaw transport chart is singular")
    transport = -yaw_rate * math.tan(pitch) / roll_cosine
    if not math.isfinite(transport):
        raise ValueError("roll/yaw transport is non-finite")
    return transport


def _allocate_roll_yaw_transport(
    command: AttitudeRateCommand,
    *,
    target_roll_rad: float,
    target_pitch_rad: float,
    bounded_yaw_rate_rad_s: float,
) -> tuple[AttitudeRateCommand, float]:
    """Add desired-reference roll transport without changing other axes."""

    if type(command) is not AttitudeRateCommand:
        raise TypeError("roll/yaw transport requires an exact command")
    transport = _roll_yaw_transport_rate_rad_s(
        target_roll_rad=target_roll_rad,
        target_pitch_rad=target_pitch_rad,
        bounded_yaw_rate_rad_s=bounded_yaw_rate_rad_s,
    )
    allocated = AttitudeRateCommand(
        roll_rate=float(command.roll_rate) + transport,
        pitch_rate=float(command.pitch_rate),
        yaw_rate=float(command.yaw_rate),
        thrust=float(command.thrust),
    )
    if not all(
        math.isfinite(float(value))
        for value in (
            allocated.roll_rate,
            allocated.pitch_rate,
            allocated.yaw_rate,
            allocated.thrust,
        )
    ):
        raise ValueError("roll/yaw transport command is non-finite")
    return allocated, transport


def _allocate_launch_collective(
    *,
    launch_elapsed_s: float,
    post_preload_thrust: float,
    configured_boost_duration_s: float,
    configured_boost_thrust: float,
    dynamic_collective_owns_post_preload: bool,
) -> tuple[float, str]:
    """Keep the fixed legacy boost out of dynamic aperture feedback."""

    values = (
        float(launch_elapsed_s),
        float(post_preload_thrust),
        float(configured_boost_duration_s),
        float(configured_boost_thrust),
    )
    if (
        not all(math.isfinite(value) for value in values)
        or values[0] < 0.0
        or values[2] < INITIAL_PAD_PRELOAD_DURATION_S
        or not MIN_VISUAL_THRUST <= values[1] <= MAX_VISUAL_THRUST
        or not MIN_VISUAL_THRUST <= values[3] <= MAX_VISUAL_THRUST
        or type(dynamic_collective_owns_post_preload) is not bool
    ):
        raise ValueError("launch collective allocation inputs are invalid")
    if values[0] < INITIAL_PAD_PRELOAD_DURATION_S:
        return INITIAL_PAD_PRELOAD_THRUST, "preload"
    if dynamic_collective_owns_post_preload:
        # In 8319198e the restored current-aperture loop requested about
        # 0.278-0.280 and then lower thrust, but this layer discarded it for
        # a fixed 0.320 boost.  Hand authority to the generic proved loop as
        # soon as the established pad preload ends.
        return values[1], "proved-current-aperture"
    if values[0] < values[2]:
        return values[3], "boost"
    return values[1], "generic-visual-servo"


def _gate0_proved_vertical_collective(
    vertical: float,
    filtered_vertical_rate: float,
) -> float:
    """Return the Gate-0-proved law for any current course aperture."""

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
class _CurrentApertureProvedCollectiveState:
    """Exact-frame collective state bound to one authoritative aperture.

    The gains remain the flight-proved Gate-0 law.  The state itself is
    gate-agnostic and is recreated at each authoritative gate transition.
    A censored vertical axis cannot synthesize a new collective request, so
    it retains the last request derived from an observable current aperture.
    """

    track_id: Optional[str] = None
    last_token_key: Optional[tuple[str, int, int, int]] = None
    last_received_monotonic_s: Optional[float] = None
    last_vertical: Optional[float] = None
    last_observable_frame_id: Optional[int] = None
    last_observable_thrust: Optional[float] = None
    filtered_vertical_rate: float = 0.0
    last_control_basis: str = RAW_CURRENT_APERTURE_COLLECTIVE_BASIS
    last_observation_vertical_censored: bool = False
    last_hold_reason: Optional[str] = None

    def hold(self, *, reason: str) -> tuple[float, float]:
        """Retain current-aperture collective without inventing geometry."""

        if reason not in {
            "vertical_censored",
            "current_aperture_dropout",
        }:
            raise ValueError(
                "current-aperture collective hold reason is invalid"
            )
        self.last_hold_reason = reason
        self.last_observation_vertical_censored = (
            reason == "vertical_censored"
        )
        thrust = (
            GATE0_PROVED_COLLECTIVE_BASE
            if self.last_observable_thrust is None
            else self.last_observable_thrust
        )
        return thrust, self.filtered_vertical_rate

    def observe(
        self,
        target: Any,
        *,
        control_vertical_error_image_down: Optional[float] = None,
        control_vertical_rate_down_s: Optional[float] = None,
        control_basis: str = RAW_CURRENT_APERTURE_COLLECTIVE_BASIS,
    ) -> tuple[float, float]:
        """Apply the proved law to raw or already-filtered derotated state."""

        track_id = getattr(target, "track_id", None)
        if type(track_id) is not str or not track_id:
            raise ValueError(
                "current-aperture collective target identity is invalid"
            )
        if self.track_id is None:
            self.track_id = track_id
        elif self.track_id != track_id:
            raise ValueError(
                "current-aperture collective target identity changed"
            )
        token = target.frame_token
        token_key = (
            str(token.stream_id),
            int(token.generation),
            int(token.frame_id),
            int(token.publication_sequence),
        )
        received = float(target.received_monotonic_s)
        raw_vertical = float(target.normalized_y_down)
        supplied_dynamic_state = bool(
            control_vertical_error_image_down is not None
            or control_vertical_rate_down_s is not None
        )
        if supplied_dynamic_state and (
            control_vertical_error_image_down is None
            or control_vertical_rate_down_s is None
        ):
            raise ValueError(
                "current-aperture derotated collective state is incomplete"
            )
        vertical = (
            raw_vertical
            if control_vertical_error_image_down is None
            else float(control_vertical_error_image_down)
        )
        direct_rate = (
            None
            if control_vertical_rate_down_s is None
            else float(control_vertical_rate_down_s)
        )
        if (
            not math.isfinite(received)
            or not math.isfinite(raw_vertical)
            or not math.isfinite(vertical)
            or (
                direct_rate is not None
                and not math.isfinite(direct_rate)
            )
            or type(control_basis) is not str
            or not control_basis
        ):
            raise ValueError(
                "current-aperture collective observation must be finite"
            )
        vertical_censored_value = getattr(
            target,
            "vertical_geometry_censored",
            None,
        )
        if vertical_censored_value is None:
            vertical_censored_value = bool(
                getattr(target, "vertical_censored", False)
                or (
                    not getattr(target, "horizontal_censored", False)
                    and (
                        getattr(target, "clipped", False)
                        or getattr(target, "center_censored", False)
                    )
                )
            )
        if type(vertical_censored_value) is not bool:
            raise ValueError(
                "current-aperture vertical censorship is invalid"
            )
        vertical_censored = vertical_censored_value
        self.last_hold_reason = None
        if self.last_token_key is not None:
            if (
                token_key[0] != self.last_token_key[0]
                or token_key[1] != self.last_token_key[1]
                or token_key[3] <= self.last_token_key[3]
            ):
                raise ValueError(
                    "current-aperture collective publication did not advance"
                )
        if not vertical_censored:
            if direct_rate is not None:
                self.filtered_vertical_rate = max(
                    -GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE,
                    min(
                        GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE,
                        direct_rate,
                    ),
                )
                self.last_received_monotonic_s = received
                self.last_vertical = vertical
                self.last_observable_frame_id = token_key[2]
                self.last_control_basis = control_basis
            elif self.last_received_monotonic_s is None:
                self.last_received_monotonic_s = received
                self.last_vertical = vertical
                self.last_observable_frame_id = token_key[2]
                self.last_control_basis = control_basis
            elif (
                self.last_vertical is not None
                and self.last_observable_frame_id is not None
                and token_key[2] != self.last_observable_frame_id
            ):
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
                self.last_observable_frame_id = token_key[2]
                self.last_control_basis = control_basis
        self.last_token_key = token_key
        self.last_observation_vertical_censored = vertical_censored
        if vertical_censored:
            return self.hold(
                reason="vertical_censored",
            )
        thrust = _gate0_proved_vertical_collective(
            vertical,
            self.filtered_vertical_rate,
        )
        self.last_observable_thrust = thrust
        return thrust, self.filtered_vertical_rate


@dataclass(frozen=True, slots=True)
class _CurrentApertureCollectiveProposal:
    requested_thrust: float
    unconstrained_requested_thrust: float
    filtered_vertical_rate_down_s: float
    control_vertical_error_image_down: Optional[float]
    control_basis: str
    vertical_censored: bool
    current_aperture_dropout: bool
    held_last_observable_collective: bool
    noncommitted_support_floor_applied: bool
    subsupport_collective_authorized: bool


def _dynamic_current_aperture_collective_inputs(
    dynamic_decision: Any,
    current_dynamic: Any,
    *,
    vertical_angle_scale_rad: float,
) -> tuple[float, float, str]:
    """Return translation-only collective P/D inputs in image units.

    Aperture-relative q-rate deliberately includes aperture expansion.  That
    is the correct crossing-geometry state, but it is not vertical vehicle
    motion and must not unload collective while the vehicle is still moving
    toward an aperture edge.
    """

    passage_error = getattr(
        dynamic_decision,
        "passage_error_norm",
        None,
    )
    if passage_error is None:
        passage_error = dynamic_decision.current_center_norm
    residual_rate_rad_s = current_dynamic.residual_translational_rate_rad_s
    vertical_angle_scale_rad = float(vertical_angle_scale_rad)
    if (
        not isinstance(passage_error, tuple)
        or len(passage_error) != 2
        or not isinstance(residual_rate_rad_s, tuple)
        or len(residual_rate_rad_s) != 2
        or not math.isfinite(vertical_angle_scale_rad)
        or vertical_angle_scale_rad <= 0.0
    ):
        raise ValueError(
            "dynamic current-aperture collective state is invalid"
        )
    vertical_error = float(passage_error[1])
    vertical_rate = (
        float(residual_rate_rad_s[1]) / vertical_angle_scale_rad
    )
    basis = CURRENT_APERTURE_PROVED_COLLECTIVE_BASIS
    if not all(
        math.isfinite(value)
        for value in (vertical_error, vertical_rate)
    ):
        raise ValueError(
            "dynamic current-aperture collective inputs must be finite"
        )
    return vertical_error, vertical_rate, basis


def _propose_current_aperture_collective(
    state: _CurrentApertureProvedCollectiveState,
    target: Any,
    *,
    authoritative_current_track_id: str,
    control_vertical_error_image_down: Optional[float] = None,
    control_vertical_rate_down_s: Optional[float] = None,
    control_basis: str = RAW_CURRENT_APERTURE_COLLECTIVE_BASIS,
    current_aperture_observable: bool = True,
    subsupport_collective_authorized: bool = False,
) -> _CurrentApertureCollectiveProposal:
    """Allocate collective only from the authoritative current aperture.

    Residual image-rate damping may request less than the proved support
    collective only after exact near-plane evidence has committed the
    crossing.  This is a state-dependent authority boundary, not a temporal
    command governor.
    """

    if (
        type(authoritative_current_track_id) is not str
        or not authoritative_current_track_id
        or state.track_id != authoritative_current_track_id
        or type(current_aperture_observable) is not bool
        or type(subsupport_collective_authorized) is not bool
    ):
        raise ValueError(
            "current-aperture collective authority is invalid"
        )
    if (
        target.track_id == authoritative_current_track_id
        and current_aperture_observable
    ):
        thrust, filtered_rate = state.observe(
            target,
            control_vertical_error_image_down=(
                control_vertical_error_image_down
            ),
            control_vertical_rate_down_s=(
                control_vertical_rate_down_s
            ),
            control_basis=control_basis,
        )
    else:
        # A clean adjacent target may own pre-credit heading during a bounded
        # current-aperture dropout, but it cannot take collective authority
        # before authoritative promotion.
        thrust, filtered_rate = state.hold(
            reason="current_aperture_dropout",
        )
    unconstrained_thrust = float(thrust)
    support_floor_applied = bool(
        not subsupport_collective_authorized
        and unconstrained_thrust < GATE0_PROVED_COLLECTIVE_BASE
    )
    if support_floor_applied:
        thrust = GATE0_PROVED_COLLECTIVE_BASE
    held = bool(
        state.last_hold_reason is not None
        and state.last_observable_thrust is not None
    )
    return _CurrentApertureCollectiveProposal(
        requested_thrust=thrust,
        unconstrained_requested_thrust=unconstrained_thrust,
        filtered_vertical_rate_down_s=filtered_rate,
        control_vertical_error_image_down=state.last_vertical,
        control_basis=state.last_control_basis,
        vertical_censored=(
            state.last_hold_reason == "vertical_censored"
        ),
        current_aperture_dropout=(
            state.last_hold_reason == "current_aperture_dropout"
        ),
        held_last_observable_collective=held,
        noncommitted_support_floor_applied=support_floor_applied,
        subsupport_collective_authorized=(
            subsupport_collective_authorized
        ),
    )


@dataclass(frozen=True, slots=True)
class _AcceptedVisualCommand:
    command: AttitudeRateCommand
    yaw_soft_stop_zeroed: bool
    observation_monotonic_ns: int
    publication_monotonic_ns: int
    wire_start_monotonic_ns: int
    wire_return_monotonic_ns: int
    wire_camera_token: CameraFrameToken
    wire_race_gate_index: int
    publication_pinned_through_transport_return: bool
    target_roll_rad: float
    target_pitch_rad: float
    next_preview_collective_delta: float
    dynamic_evidence: Optional[Dict[str, Any]]


def _refresh_committed_successor_steering(
    authority: _CensoredPassageCoastAuthority,
    accepted: _AcceptedVisualCommand,
    *,
    gate_index: int,
    current_track_id: str,
    reviewed_successor_track_id: str,
) -> _CensoredPassageCoastAuthority:
    """Carry bounded successor roll/yaw into the next crossing command.

    The current-gate clearance proof, pitch, thrust, identity, and passage
    authority remain sealed.  Exact-lineage dynamic evidence may update only
    roll and yaw steering, which still passes through the final wire governor
    on the next control tick.
    """

    if (
        type(authority) is not _CensoredPassageCoastAuthority
        or type(accepted) is not _AcceptedVisualCommand
        or type(gate_index) is not int
        or gate_index < 0
        or type(current_track_id) is not str
        or not current_track_id
        or type(reviewed_successor_track_id) is not str
        or not reviewed_successor_track_id
        or authority.gate_index != gate_index
        or authority.track_id != current_track_id
    ):
        raise ValueError(
            "committed successor steering refresh identity is invalid"
        )
    evidence = accepted.dynamic_evidence
    if (
        evidence is None
        or evidence.get("passage_committed") is not True
    ):
        return authority
    if (
        evidence.get("schema") != "aigp-vq2-dynamic-command/1"
        or evidence.get("gate_index") != gate_index
        or evidence.get("current_track_id") != current_track_id
        or evidence.get("successor_track_id")
        != reviewed_successor_track_id
    ):
        raise ValueError(
            "committed successor steering lacks exact lineage"
        )

    def admitted_axis(
        authority_name: str,
        value_name: str,
        *,
        lower: float,
        upper: float,
    ) -> Optional[float]:
        axis_authority = evidence.get(authority_name)
        value = evidence.get(value_name)
        if axis_authority == 0.0 and value is None:
            return None
        if (
            type(axis_authority) not in {int, float}
            or float(axis_authority) != 1.0
            or type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not lower - 1e-12 <= float(value) <= upper + 1e-12
        ):
            raise ValueError(
                f"committed successor {value_name} escaped its authority"
            )
        return float(value)

    committed_roll = admitted_axis(
        "committed_successor_roll_authority",
        "committed_successor_target_roll_rad",
        lower=-MAX_VISUAL_TARGET_ROLL_RAD,
        upper=MAX_VISUAL_TARGET_ROLL_RAD,
    )
    admitted_axis(
        "committed_successor_pitch_authority",
        "committed_successor_target_pitch_rad",
        lower=MIN_VISUAL_TARGET_PITCH_RAD,
        upper=MAX_VISUAL_TARGET_PITCH_RAD,
    )
    committed_yaw = admitted_axis(
        "committed_successor_yaw_authority",
        "committed_successor_yaw_rate_rad_s",
        lower=-MAX_VISUAL_YAW_RATE_RAD_S,
        upper=MAX_VISUAL_YAW_RATE_RAD_S,
    )
    if committed_roll is None or committed_yaw is None:
        return authority
    return replace(
        authority,
        target_roll_rad=committed_roll,
        yaw_rate_rad_s=(
            authority.yaw_rate_rad_s
            if accepted.yaw_soft_stop_zeroed
            else committed_yaw
        ),
    )


def _finalize_crossing_command_at_passage_admission(
    authority: _CensoredPassageCoastAuthority,
    accepted: _AcceptedVisualCommand,
    *,
    gate_index: int,
    current_track_id: str,
    reviewed_successor_track_id: str,
) -> _CensoredPassageCoastAuthority:
    """Seal an accepted successor-roll reference exactly once.

    A near-plane latch may precede the clean passage-admission frame that
    finishes the current-gate geometry proof.  Its command is therefore only
    the bounded fallback if clipping begins immediately.  Once admission
    accepts an exact bounded successor-roll reference under current-gate
    crossing reserve, that roll becomes part of the immutable crossing
    command.  Pitch, yaw, thrust, anchor, and all later propagated successor
    updates remain unchanged.
    """

    evidence = accepted.dynamic_evidence
    if (
        type(authority) is not _CensoredPassageCoastAuthority
        or type(accepted) is not _AcceptedVisualCommand
        or type(gate_index) is not int
        or gate_index < 0
        or type(current_track_id) is not str
        or not current_track_id
        or type(reviewed_successor_track_id) is not str
        or not reviewed_successor_track_id
        or authority.gate_index != gate_index
        or authority.track_id != current_track_id
        or accepted.wire_race_gate_index != gate_index
        or accepted.publication_pinned_through_transport_return is not True
        or accepted.yaw_soft_stop_zeroed
        or accepted.observation_monotonic_ns
        > accepted.publication_monotonic_ns
        or accepted.publication_monotonic_ns
        > accepted.wire_start_monotonic_ns
        or accepted.wire_start_monotonic_ns
        > accepted.wire_return_monotonic_ns
        or not isinstance(evidence, Mapping)
        or evidence.get("schema") != "aigp-vq2-dynamic-command/1"
        or evidence.get("gate_index") != gate_index
        or evidence.get("current_track_id") != current_track_id
    ):
        raise ValueError(
            "passage-admission crossing command evidence is invalid"
        )

    roll_authority = evidence.get(
        "precommit_successor_roll_authority"
    )
    roll_reference = evidence.get(
        "precommit_successor_target_roll_rad"
    )
    if roll_authority in {None, 0.0} and roll_reference is None:
        return authority
    if (
        evidence.get("passage_committed") is not False
        or evidence.get("successor_track_id")
        != reviewed_successor_track_id
        or type(roll_authority) not in {int, float}
        or not math.isfinite(float(roll_authority))
        or not 0.0 < float(roll_authority) <= 1.0
        or type(roll_reference) not in {int, float}
        or not math.isfinite(float(roll_reference))
        or abs(float(roll_reference))
        > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
        or not math.isclose(
            float(roll_reference),
            float(accepted.target_roll_rad),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "passage-admission successor roll evidence is invalid"
        )

    return replace(
        authority,
        target_roll_rad=float(roll_reference),
    )


def _dynamic_near_plane_wire_sample(
    accepted: _AcceptedVisualCommand,
    *,
    gate_index: int,
    track_id: str,
    target: Any,
    clipping: FrameEdge,
) -> Optional[NearPlaneWireSample]:
    """Adapt one accepted dynamic decision into derotated crossing evidence."""

    evidence = accepted.dynamic_evidence
    if evidence is None:
        return None
    if (
        evidence.get("gate_index") is None
        and evidence.get("current_track_id") is None
        and evidence.get("dynamic_command_count") == 0
    ):
        return None
    if (
        evidence.get("schema") != "aigp-vq2-dynamic-command/1"
        or evidence.get("gate_index") != gate_index
        or evidence.get("current_track_id") != track_id
    ):
        raise ValueError("dynamic near-plane evidence identity is invalid")
    if evidence.get("time_to_contact_s") is None:
        # Warm-up decisions cannot define either a crossing window or the
        # post-governor contact budget required to consume it.
        return None
    propagated_aperture = evidence.get(
        "current_aperture_propagated",
        False,
    )
    if type(propagated_aperture) is not bool:
        raise ValueError(
            "dynamic near-plane aperture provenance is invalid"
        )
    propagated_dynamics_qualified = evidence.get(
        "current_aperture_dynamics_qualified",
        False,
    )
    if type(propagated_dynamics_qualified) is not bool:
        raise ValueError(
            "dynamic near-plane aperture dynamics provenance is invalid"
        )
    if propagated_aperture and not propagated_dynamics_qualified:
        return None

    def scalar(
        name: str,
        *,
        minimum: Optional[float] = None,
        minimum_inclusive: bool = True,
        maximum: Optional[float] = None,
    ) -> float:
        value = evidence.get(name)
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or (
                minimum is not None
                and (
                    float(value) < minimum
                    if minimum_inclusive
                    else float(value) <= minimum
                )
            )
            or (
                maximum is not None
                and float(value) > maximum
            )
        ):
            raise ValueError(
                f"dynamic near-plane evidence {name} is invalid"
            )
        return float(value)

    def pair(
        name: str,
        *,
        minimum: Optional[float] = None,
        minimum_inclusive: bool = True,
    ) -> tuple[float, float]:
        value = evidence.get(name)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                type(item) not in {int, float}
                or not math.isfinite(float(item))
                or (
                    minimum is not None
                    and (
                        float(item) < minimum
                        if minimum_inclusive
                        else float(item) <= minimum
                    )
                )
                for item in value
            )
        ):
            raise ValueError(
                f"dynamic near-plane evidence {name} is invalid"
            )
        return float(value[0]), float(value[1])

    passage_error = pair("passage_error_norm")
    bearing_std = pair("current_bearing_std_norm")
    residual_rate = pair("residual_translation_rate_norm_s")
    current_crossing_q = pair("current_crossing_error_q")
    crossing_q_rate = pair("crossing_rate_q_s")
    crossing_prediction_horizon_s = scalar(
        "crossing_prediction_horizon_s",
        minimum=0.0,
        maximum=DYNAMIC_CROSSING_PREDICTION_MAX_HORIZON_S,
    )
    time_to_contact_s = scalar(
        "time_to_contact_s",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if crossing_prediction_horizon_s + 1e-9 < time_to_contact_s:
        # A max-horizon-capped projection does not reach its own estimated
        # gate plane.  It may steer the approach, but cannot mint a crossing
        # commitment whose bounded command lease would expire before contact.
        return None
    propagated_horizon_remaining_s = (
        scalar(
            "current_aperture_prediction_horizon_remaining_s",
            minimum=0.0,
            maximum=DYNAMIC_CROSSING_PREDICTION_MAX_HORIZON_S,
        )
        if propagated_aperture
        else None
    )
    if propagated_horizon_remaining_s == 0.0:
        # The core reports a clamped zero when a previously qualified local
        # aperture reaches its fixed, non-sliding prediction deadline.  That
        # is valid expired state, but it owns no new passage evidence.
        return None
    if (
        propagated_horizon_remaining_s is not None
        and crossing_prediction_horizon_s
        > propagated_horizon_remaining_s + 1e-9
    ):
        # An otherwise valid local prediction whose bounded lease cannot
        # reach the crossing plane simply owns no passage evidence.  It is
        # not malformed navigation state and must not abort the approach.
        return None
    predicted_crossing_error = pair(
        "predicted_crossing_error_norm",
    )
    predicted_crossing_std = pair(
        "predicted_crossing_std_norm",
        minimum=0.0,
    )
    crossing_allowance = pair(
        "crossing_allowance_norm",
        minimum=0.0,
    )
    crossing_swept_occupancy = pair(
        "crossing_swept_occupancy_norm",
        minimum=0.0,
    )
    reported_crossing_clearance = pair(
        "predicted_crossing_clearance_norm",
    )
    terminal_crossing_occupancy = pair(
        "terminal_crossing_occupancy_norm",
        minimum=0.0,
    )
    reported_terminal_clearance = pair(
        "terminal_crossing_clearance_norm",
    )
    post_governor_contact_budget_s = scalar(
        "post_governor_contact_budget_s",
    )
    if (
        evidence.get("crossing_coordinate_basis")
        != DYNAMIC_CROSSING_COORDINATE_BASIS
    ):
        raise ValueError(
            "dynamic near-plane crossing coordinate basis is invalid"
        )
    recomputed_crossing_clearance = tuple(
        crossing_allowance[axis]
        - crossing_swept_occupancy[axis]
        for axis in range(2)
    )
    if any(
        not math.isclose(
            reported_crossing_clearance[axis],
            recomputed_crossing_clearance[axis],
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for axis in range(2)
    ):
        raise ValueError(
            "dynamic near-plane evidence predicted crossing clearance "
            "is inconsistent"
        )
    recomputed_terminal_occupancy = tuple(
        abs(predicted_crossing_error[axis])
        + 2.0 * predicted_crossing_std[axis]
        for axis in range(2)
    )
    recomputed_terminal_clearance = tuple(
        crossing_allowance[axis]
        - recomputed_terminal_occupancy[axis]
        for axis in range(2)
    )
    if any(
        not math.isclose(
            terminal_crossing_occupancy[axis],
            recomputed_terminal_occupancy[axis],
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            reported_terminal_clearance[axis],
            recomputed_terminal_clearance[axis],
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for axis in range(2)
    ):
        raise ValueError(
            "dynamic near-plane terminal crossing window is inconsistent"
        )
    current_censored = evidence.get("current_censored_axes")
    if (
        not isinstance(current_censored, (list, tuple))
        or len(current_censored) != 2
        or any(type(value) is not bool for value in current_censored)
    ):
        raise ValueError(
            "dynamic near-plane censorship evidence is invalid"
        )
    bearing_rate_qualified = evidence.get(
        "current_bearing_rate_qualified"
    )
    scale_rate_qualified = evidence.get(
        "current_scale_rate_qualified"
    )
    if (
        not isinstance(bearing_rate_qualified, (list, tuple))
        or len(bearing_rate_qualified) != 2
        or any(
            type(value) is not bool
            for value in bearing_rate_qualified
        )
        or type(scale_rate_qualified) is not bool
    ):
        raise ValueError(
            "dynamic near-plane rate qualification is invalid"
        )
    return NearPlaneWireSample(
        gate_index=gate_index,
        track_id=track_id,
        camera_token=accepted.wire_camera_token,
        wire_camera_token=accepted.wire_camera_token,
        observation_monotonic_ns=accepted.observation_monotonic_ns,
        publication_monotonic_ns=accepted.publication_monotonic_ns,
        wire_start_monotonic_ns=accepted.wire_start_monotonic_ns,
        wire_return_monotonic_ns=accepted.wire_return_monotonic_ns,
        wire_race_gate_index=accepted.wire_race_gate_index,
        publication_pinned_through_transport_return=(
            accepted.publication_pinned_through_transport_return
        ),
        normalized_x=passage_error[0],
        normalized_y_down=passage_error[1],
        normalized_x_rate_s=residual_rate[0],
        normalized_y_rate_down_s=residual_rate[1],
        log_scale=float(evidence["current_log_scale"]),
        log_scale_rate_s=float(evidence["expansion_rate_s"]),
        confidence=float(evidence["current_confidence"]),
        association_confidence=float(target.association_confidence),
        clipping=clipping,
        center_censored=bool(target.center_censored),
        ambiguous=bool(
            evidence["current_ambiguous"]
            or evidence["dropout_held"]
            or not evidence["current_visible"]
            or (
                not propagated_aperture
                and (
                    not all(bearing_rate_qualified)
                    or not scale_rate_qualified
                )
            )
        ),
        command_roll_rate=accepted.command.roll_rate,
        command_pitch_rate=accepted.command.pitch_rate,
        command_yaw_rate=accepted.command.yaw_rate,
        command_thrust=accepted.command.thrust,
        geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
        normalized_x_std=bearing_std[0],
        normalized_y_std=bearing_std[1],
        log_scale_std=float(evidence["current_log_scale_std"]),
        crossing_prediction_horizon_s=crossing_prediction_horizon_s,
        predicted_crossing_x_norm=predicted_crossing_error[0],
        predicted_crossing_y_down_norm=predicted_crossing_error[1],
        predicted_crossing_x_std_norm=predicted_crossing_std[0],
        predicted_crossing_y_std_norm=predicted_crossing_std[1],
        crossing_allowance_x_norm=crossing_allowance[0],
        crossing_allowance_y_norm=crossing_allowance[1],
        crossing_swept_x_occupancy_norm=(
            crossing_swept_occupancy[0]
        ),
        crossing_swept_y_occupancy_norm=(
            crossing_swept_occupancy[1]
        ),
        current_crossing_x_q=current_crossing_q[0],
        current_crossing_y_q=current_crossing_q[1],
        crossing_x_q_rate_s=crossing_q_rate[0],
        crossing_y_q_rate_s=crossing_q_rate[1],
        post_governor_contact_budget_s=(
            post_governor_contact_budget_s
        ),
        propagated_state_horizon_remaining_s=(
            propagated_horizon_remaining_s
        ),
        propagated_state_dynamics_qualified=bool(
            propagated_aperture
            and propagated_dynamics_qualified
        ),
    )


@dataclass(frozen=True, slots=True)
class _SupersededVisualProposal:
    expected_camera_token: CameraFrameToken
    receiver_camera_token: CameraFrameToken
    held_previous_command_s: float
    consecutive_count: int


@dataclass(frozen=True, slots=True)
class _CensoredPassageCoastAuthority:
    gate_index: int
    track_id: str
    anchor_camera_token: CameraFrameToken
    target_roll_rad: float
    target_pitch_rad: float
    yaw_rate_rad_s: float
    requested_thrust: float


@dataclass(frozen=True, slots=True)
class _ApproachPropagatedVisibilityGapAuthority:
    """State-propagated steering authority after exact current visual loss."""

    command: _CensoredPassageCoastAuthority
    missed_frame_count: int
    remaining_horizon_s: float
    evidence: Mapping[str, Any]


def _approach_propagated_visibility_gap_command_deadline_s(
    authority: _ApproachPropagatedVisibilityGapAuthority,
    *,
    now_s: float,
    control_period_s: float,
) -> float:
    """Use the core's fixed local-state lease as the sole gap deadline."""

    remaining_horizon_s = authority.remaining_horizon_s
    if (
        not math.isfinite(now_s)
        or not math.isfinite(control_period_s)
        or control_period_s <= 0.0
        or not math.isfinite(remaining_horizon_s)
        or remaining_horizon_s <= control_period_s
    ):
        raise ValueError(
            "approach visibility gap exhausted its local-state horizon"
        )
    deadline_s = now_s + remaining_horizon_s
    if not math.isfinite(deadline_s) or deadline_s <= now_s:
        raise ValueError(
            "approach visibility gap deadline is invalid"
        )
    return deadline_s


def _approach_propagated_visibility_gap_authority(
    evidence: Mapping[str, Any],
    *,
    snapshot: Any,
    gate_index: int,
    track_id: str,
    fov_summary: Mapping[str, Any],
) -> _ApproachPropagatedVisibilityGapAuthority:
    """Bind state guidance to the last FOV-protected visible publication."""

    token = getattr(snapshot, "latest_camera_token", None)
    track = getattr(snapshot, "current_track", None)
    last_visible_token = getattr(track, "latest_token", None)
    command = evidence.get("command") if isinstance(evidence, Mapping) else None
    last_visible = evidence.get("last_visible_camera_token")
    last_handoff = fov_summary.get("last_propagated_state_handoff")
    last_retained_raw_handoff = fov_summary.get(
        "last_retained_raw_state_handoff"
    )
    missed_count = evidence.get("missed_frame_count")
    last_visible_clipping = evidence.get("last_visible_clipping")
    remaining_horizon_s = evidence.get(
        "steering_prediction_horizon_remaining_s"
    )
    protected_pitch = fov_summary.get(
        "last_protected_target_pitch_rad"
    )
    known_clipping_edges = int(
        FrameEdge.LEFT
        | FrameEdge.TOP
        | FrameEdge.RIGHT
        | FrameEdge.BOTTOM
    )
    vertical_clipping_edges = int(FrameEdge.TOP | FrameEdge.BOTTOM)
    last_fov_token = fov_summary.get("last_camera_token")
    direct_outer_token_lineage = bool(
        isinstance(last_fov_token, Mapping)
        and isinstance(last_visible, Mapping)
        and last_fov_token.get("stream_id")
        == last_visible.get("stream_id")
        and last_fov_token.get("generation")
        == last_visible.get("generation")
        and type(last_fov_token.get("publication_sequence")) is int
        and type(last_visible.get("publication_sequence")) is int
        and 0
        <= (
            int(last_visible["publication_sequence"])
            - int(last_fov_token["publication_sequence"])
        )
        <= 1
    )
    propagated_fov_lineage = bool(
        isinstance(last_handoff, Mapping)
        and last_handoff.get("basis")
        == "propagated-current-fov-gap-steering-v1"
        and isinstance(last_visible, Mapping)
        and last_handoff.get("camera_token") == dict(last_visible)
    )
    propagated_superseded_fov_lineage = bool(
        isinstance(last_handoff, Mapping)
        and last_handoff.get("basis")
        == "propagated-current-fov-gap-steering-v1"
        and last_handoff.get("gate_index") == gate_index
        and last_handoff.get("track_id") == track_id
        and last_handoff.get("steering_only") is True
        and last_handoff.get("passage_authority") is False
        and last_handoff.get("advance_authority") is False
        and isinstance(last_visible, Mapping)
        and isinstance(last_handoff.get("camera_token"), Mapping)
        and last_handoff["camera_token"].get("stream_id")
        == last_visible.get("stream_id")
        and last_handoff["camera_token"].get("generation")
        == last_visible.get("generation")
        and type(
            last_handoff["camera_token"].get("publication_sequence")
        )
        is int
        and type(last_visible.get("publication_sequence")) is int
        and (
            int(last_visible["publication_sequence"])
            - int(
                last_handoff["camera_token"]["publication_sequence"]
            )
        )
        == 1
        and last_fov_token == dict(last_handoff["camera_token"])
        and type(last_visible_clipping) is int
        and last_visible_clipping != 0
    )
    retained_raw_fov_lineage = bool(
        isinstance(last_retained_raw_handoff, Mapping)
        and last_retained_raw_handoff.get("basis")
        == TOP_FOV_RETAINED_RAW_STATE_BASIS
        and last_retained_raw_handoff.get("steering_only") is True
        and last_retained_raw_handoff.get("passage_authority") is False
        and last_retained_raw_handoff.get("advance_authority") is False
        and isinstance(last_visible, Mapping)
        and last_retained_raw_handoff.get("camera_token")
        == dict(last_visible)
    )
    direct_inner_fov_lineage = bool(
        fov_summary.get("last_raw_top_edge_basis")
        == TOP_FOV_INNER_EDGE_BASIS
        and fov_summary.get("last_inner_raw_top_edge_basis")
        == TOP_FOV_INNER_EDGE_BASIS
        and fov_summary.get("last_inner_track_id") == track_id
        and fov_summary.get("last_inner_camera_token")
        == (
            None
            if not isinstance(last_visible, Mapping)
            else dict(last_visible)
        )
        and fov_summary.get("last_inner_active") is True
    )
    direct_outer_fov_lineage = bool(
        type(last_visible_clipping) is int
        and last_visible_clipping != 0
        and last_visible_clipping & vertical_clipping_edges == 0
        and fov_summary.get("last_raw_top_edge_basis")
        == TOP_FOV_OUTER_EDGE_FALLBACK_BASIS
        # A visible proposal can be superseded by the first missing
        # publication before its raw FOV summary reaches the wire.  For a
        # horizontal-only loss, retain the last accepted outer-edge pitch
        # authority across exactly that one-publication race.
        and direct_outer_token_lineage
    )
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("basis")
        != "propagated-current-visibility-gap-guidance-v2"
        or evidence.get("steering_only") is not True
        or evidence.get("passage_authority") is not False
        or evidence.get("advance_authority") is not False
        or type(token) is not CameraFrameToken
        or evidence.get("camera_token") != asdict(token)
        or evidence.get("gate_index") != gate_index
        or evidence.get("track_id") != track_id
        or track is None
        or getattr(track, "track_id", None) != track_id
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or getattr(track, "visible", True) is not False
        or getattr(track, "ambiguous", True) is not False
        or type(missed_count) is not int
        or missed_count <= 0
        or getattr(track, "missed_frame_count", None) != missed_count
        or type(last_visible_token) is not CameraFrameToken
        or not isinstance(last_visible, Mapping)
        or dict(last_visible) != asdict(last_visible_token)
        or type(last_visible_clipping) is not int
        or last_visible_clipping == 0
        or last_visible_clipping & ~known_clipping_edges
        or int(getattr(track, "clipping", FrameEdge.NONE))
        != last_visible_clipping
        # A horizontal-only clipped loss does not need an actively limiting
        # top-FOV envelope.  Its exact outer-edge lineage and the already
        # finite protected pitch are sufficient; vertical clipping still
        # requires active inner/propagated FOV ownership.
        or (
            fov_summary.get("active") is not True
            and not direct_outer_fov_lineage
        )
        or fov_summary.get("last_track_id") != track_id
        or (
            fov_summary.get("last_camera_token") != dict(last_visible)
            and not direct_outer_fov_lineage
            and not propagated_superseded_fov_lineage
        )
        or not (
            propagated_fov_lineage
            or propagated_superseded_fov_lineage
            or retained_raw_fov_lineage
            or direct_inner_fov_lineage
            or direct_outer_fov_lineage
        )
        or not isinstance(command, Mapping)
        or type(remaining_horizon_s) not in {int, float}
        or not math.isfinite(float(remaining_horizon_s))
        or float(remaining_horizon_s) <= 0.0
        or type(protected_pitch) not in {int, float}
        or not math.isfinite(float(protected_pitch))
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= float(protected_pitch)
        <= MAX_VISUAL_TARGET_PITCH_RAD
    ):
        raise ValueError(
            "approach visibility gap lacks exact propagated/FOV authority"
        )
    try:
        target_roll = float(command["target_roll_rad"])
        requested_pitch = float(command["target_pitch_rad"])
        yaw_rate = float(command["yaw_rate_rad_s"])
        thrust = float(command["thrust"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "approach visibility gap command is malformed"
        ) from exc
    target_pitch = min(requested_pitch, float(protected_pitch))
    if (
        not all(
            math.isfinite(value)
            for value in (
                target_roll,
                requested_pitch,
                target_pitch,
                yaw_rate,
                thrust,
            )
        )
        or abs(target_roll) > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= target_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or abs(yaw_rate) > MAX_VISUAL_YAW_RATE_RAD_S + 1e-12
        or not MIN_VISUAL_THRUST <= thrust <= MAX_VISUAL_THRUST
    ):
        raise ValueError(
            "approach visibility gap command escaped its envelope"
        )
    return _ApproachPropagatedVisibilityGapAuthority(
        command=_CensoredPassageCoastAuthority(
            gate_index=gate_index,
            track_id=track_id,
            anchor_camera_token=last_visible_token,
            target_roll_rad=target_roll,
            target_pitch_rad=target_pitch,
            yaw_rate_rad_s=yaw_rate,
            requested_thrust=thrust,
        ),
        missed_frame_count=missed_count,
        remaining_horizon_s=float(remaining_horizon_s),
        evidence=dict(evidence),
    )


@dataclass(frozen=True, slots=True)
class _ApproachCurrentAmbiguityQuarantineAuthority:
    """Prior accepted command retained without consuming ambiguous geometry."""

    command: _CensoredPassageCoastAuthority
    clean_camera_token: CameraFrameToken
    first_ambiguous_camera_token: CameraFrameToken
    latest_ambiguous_camera_token: CameraFrameToken
    anchor_wire_start_monotonic_ns: int
    source_wire_start_monotonic_ns: int
    expires_monotonic_ns: int
    raw_top_handoff: Mapping[str, Any]


def _first_ambiguity_exact_raw_top_handoff(
    *,
    token: CameraFrameToken,
    gate_index: int,
    track_id: str,
    now_monotonic_ns: int,
    maximum_age_s: float,
    fov_summary: Mapping[str, Any],
    hold: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Freeze the immediately preceding accepted exact raw-TOP anchor.

    The ordinary retained-raw projection is produced only after a clean
    vertically clipped publication has already followed its exact anchor.
    The first ambiguous publication has no such intermediate frame.  Admit
    that one structural seam directly from the preceding accepted exact
    anchor, with the caller-proved command-continuity horizon and no renewal,
    passage, or advance authority.
    """

    anchor = fov_summary.get("exact_raw_anchor")
    last_token_fields = fov_summary.get("last_camera_token")
    if (
        type(token) is not CameraFrameToken
        or not isinstance(anchor, Mapping)
        or not isinstance(last_token_fields, Mapping)
        or not isinstance(hold, Mapping)
    ):
        raise ValueError(
            "approach ambiguity quarantine lacks retained raw TOP authority"
        )
    try:
        anchor_token = CameraFrameToken(
            **dict(anchor["camera_token"])
        )
        last_token = CameraFrameToken(**dict(last_token_fields))
        anchor_wire_ns = int(anchor["wire_start_monotonic_ns"])
        source_wire_ns = int(hold["source_wire_start_monotonic_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "approach ambiguity quarantine exact raw TOP anchor is malformed"
        ) from exc

    maximum_age = float(maximum_age_s)
    expires_ns = anchor_wire_ns + round(
        maximum_age * 1_000_000_000.0
    )
    remaining_s = (
        expires_ns - anchor_wire_ns
    ) / 1_000_000_000.0
    if (
        anchor.get("basis") != TOP_FOV_EXACT_RAW_ANCHOR_BASIS
        or anchor.get("gate_index") != gate_index
        or anchor.get("track_id") != track_id
        or anchor.get("active") is not True
        or anchor.get("steering_only") is not True
        or anchor.get("passage_authority") is not False
        or anchor.get("advance_authority") is not False
        or fov_summary.get("active") is not True
        or fov_summary.get("last_track_id") != track_id
        or last_token != anchor_token
        or not _token_strictly_newer(token, anchor_token)
        or token.stream_id != anchor_token.stream_id
        or token.generation != anchor_token.generation
        or token.publication_sequence
        - anchor_token.publication_sequence
        != 1
        or anchor_wire_ns < 0
        or source_wire_ns != anchor_wire_ns
        or fov_summary.get("last_wire_start_monotonic_ns")
        != source_wire_ns
        or not math.isfinite(maximum_age)
        or maximum_age <= 0.0
        or maximum_age > 0.12 + 1e-12
        or now_monotonic_ns < source_wire_ns
        or now_monotonic_ns >= expires_ns
        or not math.isfinite(remaining_s)
        or remaining_s <= 0.0
    ):
        raise ValueError(
            "approach ambiguity quarantine lacks exact fixed raw TOP lease"
        )
    return {
        "basis": (
            APPROACH_CURRENT_AMBIGUITY_EXACT_RAW_LEASE_BASIS
        ),
        "gate_index": gate_index,
        "track_id": track_id,
        "anchor_camera_token": asdict(anchor_token),
        "camera_token": asdict(anchor_token),
        "anchor_wire_start_monotonic_ns": anchor_wire_ns,
        "authority_monotonic_ns": anchor_wire_ns,
        "maximum_age_s": maximum_age,
        "prediction_horizon_remaining_s": remaining_s,
        "steering_only": True,
        "passage_authority": False,
        "advance_authority": False,
    }


def _approach_current_ambiguity_quarantine_authority(
    *,
    snapshot: Any,
    gate_index: int,
    track_id: str,
    now_monotonic_ns: int,
    maximum_hold_age_s: float,
    fov_summary: Mapping[str, Any],
    hold: Optional[Mapping[str, Any]],
    existing: Optional[_ApproachCurrentAmbiguityQuarantineAuthority],
) -> _ApproachCurrentAmbiguityQuarantineAuthority:
    """Quarantine one same-identity ambiguous TOP publication.

    The ambiguous publication is admitted only as a fresh wire watermark.  Its
    center, scale, rates, and aperture never enter the dynamic estimator.  The
    command and absolute deadline come from the immediately preceding accepted
    unambiguous raw-TOP handoff, and subsequent ambiguous publications cannot
    renew either.
    """

    if (
        type(gate_index) is not int
        or gate_index < 0
        or type(track_id) is not str
        or not track_id
        or type(now_monotonic_ns) is not int
        or now_monotonic_ns < 0
        or type(maximum_hold_age_s) not in {int, float}
        or not math.isfinite(float(maximum_hold_age_s))
        or not 0.0 < float(maximum_hold_age_s) <= 0.12 + 1e-12
        or not isinstance(fov_summary, Mapping)
        or (
            existing is not None
            and type(existing)
            is not _ApproachCurrentAmbiguityQuarantineAuthority
        )
    ):
        raise ValueError("approach ambiguity quarantine inputs are invalid")

    token = getattr(snapshot, "latest_camera_token", None)
    track = getattr(snapshot, "current_track", None)
    history = getattr(track, "history", None)
    sample = history[-1] if type(history) is tuple and history else None
    missed_count = getattr(track, "missed_frame_count", None)
    provisional_ids = getattr(snapshot, "provisional_track_ids", None)
    if (
        type(token) is not CameraFrameToken
        or getattr(snapshot, "current_gate_index", None) != gate_index
        or getattr(snapshot, "current_track_id", None) != track_id
        or getattr(snapshot, "authority_usable", True) is not False
        or getattr(snapshot, "withholding_reason", None)
        != "current_track_ambiguous"
        or getattr(snapshot, "race_finished", True) is not False
        or getattr(snapshot, "next_selection_ambiguous", True) is not False
        or type(provisional_ids) is not tuple
        or track is None
        or getattr(track, "track_id", None) != track_id
        or getattr(track, "role", None) is not VisualTrackRole.AMBIGUOUS
        or getattr(track, "authoritative_gate_index", None) != gate_index
        or getattr(track, "visible", False) is not True
        or getattr(track, "ambiguous", False) is not True
        or type(missed_count) is not int
        or missed_count != 0
        or getattr(track, "latest_token", None) != token
        or getattr(track, "clipping", None) is not FrameEdge.TOP
        or getattr(track, "center_censored", False) is not True
        or sample is None
        or getattr(sample, "token", None) != token
        or getattr(sample, "clipping", None) is not FrameEdge.TOP
        or getattr(sample, "center_censored", False) is not True
    ):
        raise ValueError(
            "approach ambiguity quarantine lacks exact current TOP identity"
        )

    raw_handoff = fov_summary.get("last_retained_raw_state_handoff")
    if not isinstance(raw_handoff, Mapping):
        if existing is not None:
            raw_handoff = existing.raw_top_handoff
        elif isinstance(hold, Mapping):
            raw_handoff = _first_ambiguity_exact_raw_top_handoff(
                token=token,
                gate_index=gate_index,
                track_id=track_id,
                now_monotonic_ns=now_monotonic_ns,
                maximum_age_s=float(maximum_hold_age_s),
                fov_summary=fov_summary,
                hold=hold,
            )
        else:
            raise ValueError(
                "approach ambiguity quarantine lacks retained raw TOP "
                "authority"
            )

    if existing is not None:
        if (
            hold is not None
            or dict(raw_handoff) != dict(existing.raw_top_handoff)
            or existing.command.gate_index != gate_index
            or existing.command.track_id != track_id
            or existing.clean_camera_token
            != existing.command.anchor_camera_token
            or not _token_strictly_newer(
                token,
                existing.latest_ambiguous_camera_token,
            )
            or token.stream_id != existing.clean_camera_token.stream_id
            or token.generation != existing.clean_camera_token.generation
            or now_monotonic_ns >= existing.expires_monotonic_ns
        ):
            raise ValueError(
                "approach ambiguity quarantine continuity is invalid or "
                "expired"
            )
        return replace(
            existing,
            latest_ambiguous_camera_token=token,
        )

    if not isinstance(hold, Mapping):
        raise ValueError(
            "approach ambiguity quarantine lacks an accepted command"
        )
    raw_token_fields = raw_handoff.get("camera_token")
    raw_anchor_token_fields = raw_handoff.get("anchor_camera_token")
    exact_raw_anchor = fov_summary.get("exact_raw_anchor")
    if (
        not isinstance(raw_token_fields, Mapping)
        or not isinstance(raw_anchor_token_fields, Mapping)
        or not isinstance(exact_raw_anchor, Mapping)
    ):
        raise ValueError(
            "approach ambiguity quarantine raw TOP lineage is malformed"
        )
    try:
        clean_token = CameraFrameToken(**dict(raw_token_fields))
        raw_anchor_token = CameraFrameToken(
            **dict(raw_anchor_token_fields)
        )
        anchor_wire_ns = int(
            raw_handoff["anchor_wire_start_monotonic_ns"]
        )
        authority_ns = int(raw_handoff["authority_monotonic_ns"])
        maximum_age_s = float(raw_handoff["maximum_age_s"])
        stated_remaining_s = float(
            raw_handoff["prediction_horizon_remaining_s"]
        )
        source_wire_ns = int(hold["source_wire_start_monotonic_ns"])
        target_roll = float(hold["target_roll_rad"])
        target_pitch = float(hold["target_pitch_rad"])
        yaw_rate = float(hold["yaw_rate_rad_s"])
        thrust = float(hold["thrust"])
        protected_pitch = float(
            fov_summary["last_protected_target_pitch_rad"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "approach ambiguity quarantine authority is malformed"
        ) from exc

    expires_ns = anchor_wire_ns + round(
        maximum_age_s * 1_000_000_000.0
    )
    expected_remaining_s = (
        expires_ns - authority_ns
    ) / 1_000_000_000.0
    clean_publication = clean_token.publication_sequence
    ambiguous_publication = token.publication_sequence
    retained_projection_handoff = bool(
        raw_handoff.get("basis")
        == TOP_FOV_RETAINED_RAW_STATE_BASIS
    )
    direct_exact_handoff = bool(
        raw_handoff.get("basis")
        == APPROACH_CURRENT_AMBIGUITY_EXACT_RAW_LEASE_BASIS
    )
    if (
        not (retained_projection_handoff or direct_exact_handoff)
        or raw_handoff.get("gate_index") != gate_index
        or raw_handoff.get("track_id") != track_id
        or raw_handoff.get("steering_only") is not True
        or raw_handoff.get("passage_authority") is not False
        or raw_handoff.get("advance_authority") is not False
        or fov_summary.get("active") is not True
        or fov_summary.get("last_track_id") != track_id
        or fov_summary.get("last_camera_token") != dict(raw_token_fields)
        or fov_summary.get("last_wire_start_monotonic_ns")
        != source_wire_ns
        or exact_raw_anchor.get("basis")
        != TOP_FOV_EXACT_RAW_ANCHOR_BASIS
        or exact_raw_anchor.get("gate_index") != gate_index
        or exact_raw_anchor.get("track_id") != track_id
        or exact_raw_anchor.get("camera_token")
        != dict(raw_anchor_token_fields)
        or exact_raw_anchor.get("wire_start_monotonic_ns")
        != anchor_wire_ns
        or exact_raw_anchor.get("steering_only") is not True
        or exact_raw_anchor.get("passage_authority") is not False
        or exact_raw_anchor.get("advance_authority") is not False
        or (
            retained_projection_handoff
            and not _token_strictly_newer(
                clean_token,
                raw_anchor_token,
            )
        )
        or (
            direct_exact_handoff
            and clean_token != raw_anchor_token
        )
        or not _token_strictly_newer(token, clean_token)
        or clean_token.stream_id != token.stream_id
        or clean_token.generation != token.generation
        or type(clean_publication) is not int
        or type(ambiguous_publication) is not int
        or ambiguous_publication - clean_publication != 1
        or anchor_wire_ns < 0
        or authority_ns < anchor_wire_ns
        or source_wire_ns < authority_ns
        or source_wire_ns >= expires_ns
        or not math.isfinite(maximum_age_s)
        or maximum_age_s <= 0.0
        or maximum_age_s > 0.60 + 1e-12
        or not math.isfinite(stated_remaining_s)
        or not math.isclose(
            stated_remaining_s,
            expected_remaining_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or authority_ns > now_monotonic_ns
        or now_monotonic_ns >= expires_ns
        or not all(
            math.isfinite(value)
            for value in (
                target_roll,
                target_pitch,
                yaw_rate,
                thrust,
                protected_pitch,
            )
        )
        or abs(target_roll) > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= target_pitch
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or abs(yaw_rate) > MAX_VISUAL_YAW_RATE_RAD_S + 1e-12
        or not MIN_VISUAL_THRUST <= thrust <= MAX_VISUAL_THRUST
        or not math.isclose(
            target_pitch,
            protected_pitch,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "approach ambiguity quarantine lacks exact fixed raw TOP lease"
        )

    return _ApproachCurrentAmbiguityQuarantineAuthority(
        command=_CensoredPassageCoastAuthority(
            gate_index=gate_index,
            track_id=track_id,
            anchor_camera_token=clean_token,
            target_roll_rad=target_roll,
            target_pitch_rad=target_pitch,
            yaw_rate_rad_s=yaw_rate,
            requested_thrust=thrust,
        ),
        clean_camera_token=clean_token,
        first_ambiguous_camera_token=token,
        latest_ambiguous_camera_token=token,
        anchor_wire_start_monotonic_ns=anchor_wire_ns,
        source_wire_start_monotonic_ns=source_wire_ns,
        expires_monotonic_ns=expires_ns,
        raw_top_handoff=dict(raw_handoff),
    )


@dataclass(frozen=True, slots=True)
class _ApproachTopRecoveryAuthority:
    """Clean exact-wire authority for a bounded TOP-only approach hold.

    This is deliberately not crossing authority.  It only preserves the last
    clean current-gate command while a single censored vertical observation
    arrives after both aperture-relative and raw image motion have turned
    away from TOP.  Passage evidence cannot be advanced from this authority.
    """

    command: _CensoredPassageCoastAuthority
    anchor_wire_start_monotonic_ns: int
    current_vertical_q: float
    vertical_q_rate_s: float
    predicted_vertical_q: float
    predicted_vertical_q_std: float
    vertical_allowance_q: float
    vertical_endpoint_occupancy_q: float
    time_to_contact_s: float
    raw_vertical_rate_down_s: float
    thrust_settle_s: float
    post_settle_contact_budget_s: float


def _derive_approach_top_recovery_authority(
    accepted: _AcceptedVisualCommand,
    *,
    gate_index: int,
    track_id: str,
    raw_vertical_rate_down_s: float,
    requested_thrust: float,
    minimum_brake_pitch_rad: float,
    maximum_recovery_duration_s: float,
) -> Optional[_ApproachTopRecoveryAuthority]:
    """Admit only the clean c25-class state that is already moving inward."""

    evidence = accepted.dynamic_evidence
    if evidence is None:
        return None
    if not isinstance(evidence, Mapping):
        raise ValueError("approach TOP recovery evidence is not a mapping")
    if (
        evidence.get("gate_index") is None
        and evidence.get("current_track_id") is None
        and evidence.get("dynamic_command_count") == 0
    ):
        return None
    if evidence.get("time_to_contact_s") is None:
        # A structurally valid warm-up decision cannot estimate closure until
        # its scale-rate filter is qualified.  It is not recovery authority.
        return None

    def scalar(name: str) -> float:
        value = evidence.get(name)
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"approach TOP recovery evidence {name} is invalid"
            )
        return float(value)

    def pair(name: str) -> tuple[float, float]:
        value = evidence.get(name)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                type(item) not in {int, float}
                or not math.isfinite(float(item))
                for item in value
            )
        ):
            raise ValueError(
                f"approach TOP recovery evidence {name} is invalid"
            )
        return float(value[0]), float(value[1])

    values = (
        float(raw_vertical_rate_down_s),
        float(requested_thrust),
        float(minimum_brake_pitch_rad),
        float(maximum_recovery_duration_s),
    )
    if (
        not all(math.isfinite(value) for value in values)
        or not MIN_VISUAL_THRUST <= values[1] <= MAX_VISUAL_THRUST
        or not MIN_VISUAL_TARGET_PITCH_RAD
        <= values[2]
        <= MAX_VISUAL_TARGET_PITCH_RAD
        or values[3] <= 0.0
    ):
        raise ValueError("approach TOP recovery inputs are invalid")
    if (
        evidence.get("schema") != "aigp-vq2-dynamic-command/1"
        or evidence.get("gate_index") != gate_index
        or evidence.get("current_track_id") != track_id
        or evidence.get("crossing_coordinate_basis")
        != DYNAMIC_CROSSING_COORDINATE_BASIS
    ):
        raise ValueError("approach TOP recovery identity is invalid")

    current_q = pair("current_crossing_error_q")
    q_rate = pair("crossing_rate_q_s")
    predicted_q = pair("predicted_crossing_error_norm")
    predicted_std = pair("predicted_crossing_std_norm")
    allowance = pair("crossing_allowance_norm")
    camera_center = pair("camera_current_center_norm")
    time_to_contact_s = scalar("time_to_contact_s")
    successor_yaw = scalar("successor_yaw_contribution_rad")
    expansion_rate_s = scalar("expansion_rate_s")
    qualified = evidence.get("current_bearing_rate_qualified")
    censored = evidence.get("current_censored_axes")
    if (
        not isinstance(qualified, (list, tuple))
        or len(qualified) != 2
        or any(type(value) is not bool for value in qualified)
        or not isinstance(censored, (list, tuple))
        or len(censored) != 2
        or any(type(value) is not bool for value in censored)
        or type(evidence.get("current_scale_rate_qualified")) is not bool
        or type(evidence.get("current_visible")) is not bool
        or type(evidence.get("current_ambiguous")) is not bool
        or type(evidence.get("braking")) is not bool
        or type(evidence.get("passage_scale_ready")) is not bool
    ):
        raise ValueError(
            "approach TOP recovery qualification evidence is invalid"
        )

    vertical_endpoint_occupancy = (
        abs(predicted_q[1])
        + APPROACH_TOP_RECOVERY_ENDPOINT_SIGMA * predicted_std[1]
    )
    thrust_settle_s = (
        abs(float(accepted.command.thrust) - requested_thrust)
        / APPROACH_TOP_RECOVERY_THRUST_SLEW_PER_S
    )
    post_settle_contact_budget_s = (
        time_to_contact_s
        - APPROACH_TOP_RECOVERY_ACTION_DELAY_S
        - thrust_settle_s
    )
    eligible = bool(
        evidence["current_visible"]
        and not evidence["current_ambiguous"]
        and not any(censored)
        and all(qualified)
        and evidence["current_scale_rate_qualified"]
        and evidence["braking"]
        and evidence.get("brake_reason")
        == "vertical_alignment_unsettled"
        and not evidence["passage_scale_ready"]
        and current_q[1] < 0.0
        and q_rate[1]
        >= APPROACH_TOP_RECOVERY_MIN_INWARD_Q_RATE_S
        and current_q[1] * q_rate[1] < 0.0
        and predicted_std[1]
        <= APPROACH_TOP_RECOVERY_MAX_VERTICAL_Q_STD
        and allowance[1] > 0.0
        and vertical_endpoint_occupancy <= allowance[1]
        and time_to_contact_s > maximum_recovery_duration_s
        and thrust_settle_s
        <= APPROACH_TOP_RECOVERY_MAX_THRUST_SETTLE_S
        and post_settle_contact_budget_s
        >= maximum_recovery_duration_s
        and expansion_rate_s > 0.0
        and raw_vertical_rate_down_s >= 0.0
        and abs(camera_center[0])
        <= APPROACH_TOP_RECOVERY_MAX_ABS_CAMERA_CENTER_NORM
        and abs(camera_center[1])
        <= APPROACH_TOP_RECOVERY_MAX_ABS_CAMERA_CENTER_NORM
        and abs(successor_yaw) <= 1e-9
        and accepted.target_pitch_rad
        >= minimum_brake_pitch_rad - 1e-12
    )
    if not eligible:
        return None
    if (
        any(value < 0.0 for value in predicted_std)
        or any(value < 0.0 for value in allowance)
        or type(accepted.wire_start_monotonic_ns) is not int
        or accepted.wire_start_monotonic_ns < 0
    ):
        raise ValueError(
            "approach TOP recovery uncertainty/timing is invalid"
        )
    return _ApproachTopRecoveryAuthority(
        command=_CensoredPassageCoastAuthority(
            gate_index=gate_index,
            track_id=track_id,
            anchor_camera_token=accepted.wire_camera_token,
            target_roll_rad=accepted.target_roll_rad,
            target_pitch_rad=accepted.target_pitch_rad,
            yaw_rate_rad_s=float(accepted.command.yaw_rate),
            requested_thrust=requested_thrust,
        ),
        anchor_wire_start_monotonic_ns=(
            accepted.wire_start_monotonic_ns
        ),
        current_vertical_q=current_q[1],
        vertical_q_rate_s=q_rate[1],
        predicted_vertical_q=predicted_q[1],
        predicted_vertical_q_std=predicted_std[1],
        vertical_allowance_q=allowance[1],
        vertical_endpoint_occupancy_q=vertical_endpoint_occupancy,
        time_to_contact_s=time_to_contact_s,
        raw_vertical_rate_down_s=raw_vertical_rate_down_s,
        thrust_settle_s=thrust_settle_s,
        post_settle_contact_budget_s=post_settle_contact_budget_s,
    )


@dataclass(frozen=True, slots=True)
class _ConfirmedCourseHandoff:
    """Common command boundary for retained promotion or fresh reacquisition."""

    from_gate_index: int
    to_gate_index: int
    retired_track_id: str
    promoted_track_id: str
    race_status: AuthoritativeRaceStatusRef
    camera_token_at_credit: CameraFrameToken
    promoted_history_sha256: str
    history_length_before_promotion: int
    history_length_after_promotion: int
    promotion_identity_basis: str
    cross_gap_identity_claimed: bool


@dataclass(frozen=True, slots=True)
class _PendingPostCreditRecovery:
    """Carry one admitted promoted publication into its command segment."""

    from_gate_index: int
    to_gate_index: int
    track_id: str
    camera_token_at_credit: CameraFrameToken
    admitted_camera_token: CameraFrameToken
    deadline_s: float
    successor_steering_available: bool


@dataclass(frozen=True, slots=True)
class VisualCourseStageLimits:
    """Code-owned bounds for the generic lifecycle."""

    control_period_s: float = 0.02
    course_hard_duration_s: float = 120.0
    segment_hard_duration_s: float = MAX_VISUAL_SEGMENT_DURATION_S
    passage_hard_duration_s: float = MAX_VISUAL_SEGMENT_DURATION_S
    # Three nominal 4 Hz race packets cover the observed censor-to-credit
    # interval without changing command envelopes, freshness, or segment time.
    crossing_status_timeout_s: float = 0.75
    censored_passage_coast_max_duration_s: float = 0.30
    censored_passage_coast_max_fresh_frames: int = 8
    approach_top_recovery_max_duration_s: float = 0.12
    approach_top_recovery_max_fresh_frames: int = 3
    post_credit_fresh_frame_timeout_s: float = 0.20
    max_validation_to_wire_delay_s: float = 0.012
    max_command_rate_rad_s: float = 0.25
    max_yaw_rate_rad_s: float = MAX_VISUAL_YAW_RATE_RAD_S
    max_abs_measured_roll_rad: float = 0.18
    min_measured_pitch_rad: float = -0.35
    max_measured_pitch_rad: float = 0.15
    max_abs_measured_body_rate_rad_s: float = 0.50
    max_segment_yaw_excursion_rad: float = (
        MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD
    )
    max_measured_yaw_rate_rad_s: float = 0.50
    min_thrust: float = MIN_VISUAL_THRUST
    max_thrust: float = MAX_VISUAL_THRUST
    crossing_arm_min_log_scale: float = -0.80
    max_gate_segments: int = 64

    def __post_init__(self) -> None:
        numeric = (
            self.control_period_s,
            self.course_hard_duration_s,
            self.segment_hard_duration_s,
            self.passage_hard_duration_s,
            self.crossing_status_timeout_s,
            self.censored_passage_coast_max_duration_s,
            self.approach_top_recovery_max_duration_s,
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
        if not 0.05 <= self.crossing_status_timeout_s <= 0.75:
            raise ValueError("visual-course crossing wait is outside bounds")
        if not 0.20 <= self.censored_passage_coast_max_duration_s <= 0.30:
            raise ValueError(
                "visual-course censored passage coast is outside bounds"
            )
        if not 0.08 <= self.approach_top_recovery_max_duration_s <= 0.12:
            raise ValueError(
                "visual-course approach TOP recovery is outside bounds"
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
        if not 0.0 < self.max_abs_measured_roll_rad <= 0.18:
            raise ValueError(
                "visual-course measured roll diagnostic threshold is invalid"
            )
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
            MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD
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
        if (
            type(self.censored_passage_coast_max_fresh_frames) is not int
            or not 4
            <= self.censored_passage_coast_max_fresh_frames
            <= 8
            or type(self.approach_top_recovery_max_fresh_frames) is not int
            or not 2
            <= self.approach_top_recovery_max_fresh_frames
            <= 3
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
        """Load and validate the tracked sign-plus-capability authority."""

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
                profile["capability"]["max_abs_body_rate_rad_s"]
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
    limits: VisualCourseStageLimits = DEFAULT_VISUAL_COURSE_LIMITS
    servo_factory: Callable[..., Any] = RollingVisualApproachServo
    dynamic_controller: Optional[Any] = None

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
            "servo_factory",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"visual-course runtime {name} is not callable")
        if self.dynamic_controller is not None:
            for name in (
                "govern_wire_command",
                "record_wire_acceptance",
                "continuity_hold_authority",
                "evidence_summary",
            ):
                if not callable(
                    getattr(self.dynamic_controller, name, None)
                ):
                    raise TypeError(
                        "visual-course dynamic controller lacks "
                        f"{name}"
                    )


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

    def _sample_control_ingress(self) -> None: ...

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

    def _confirm_visual_course_advance(
        self,
        *,
        from_gate_index: int,
        to_gate_index: int,
        race_status: AuthoritativeRaceStatusRef,
        reviewed_track_id: str,
    ) -> Any: ...

    def _try_visual_reacquired_current(self) -> Any: ...

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


def _perf_counter_deadline_from_monotonic(
    *,
    deadline_monotonic_s: float,
    now_monotonic_s: float,
    validation_perf_counter_ns: int,
) -> int:
    """Translate a monotonic deadline by duration, never by clock epoch."""

    remaining_s = max(
        0.0,
        float(deadline_monotonic_s) - float(now_monotonic_s),
    )
    return int(validation_perf_counter_ns) + math.floor(
        remaining_s * 1_000_000_000
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
    peak_body_rate = max(abs(value) for value in rates)
    if peak_body_rate > limits.max_abs_measured_body_rate_rad_s:
        host.recorder.emit(
            "visual_course_measured_body_rate_corridor_exceeded",
            phase=phase,
            disposition="diagnostic_only",
            threshold_rad_s=limits.max_abs_measured_body_rate_rad_s,
            peak_abs_body_rate_rad_s=peak_body_rate,
            measured_body_rates_rad_s=list(rates),
            measured_attitude_rpy_rad=[roll, pitch, yaw],
        )
    if abs(roll) > limits.max_abs_measured_roll_rad:
        host.recorder.emit(
            "visual_course_measured_roll_corridor_exceeded",
            phase=phase,
            disposition="diagnostic_only",
            threshold_rad=limits.max_abs_measured_roll_rad,
            measured_roll_rad=roll,
            measured_pitch_rad=pitch,
            measured_body_rates_rad_s=list(rates),
        )
    if (
        pitch < limits.min_measured_pitch_rad
        or pitch > limits.max_measured_pitch_rad
    ):
        raise abort_type(
            "visual-course measured attitude envelope was exceeded "
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
    # The tracked profile identifies yaw sign, command rate, response delay,
    # and measured free-flight rate.  Its calibration attitude excursion is
    # not a course-heading limit.  Course heading remains independently
    # bounded by the visual controller's immutable per-segment envelope.
    hard_boundary_rad = limits.max_segment_yaw_excursion_rad
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


def _target_observation_monotonic_ns(
    snapshot: Any,
    target: Any,
    track: Any,
    *,
    abort_type: type[BaseException],
) -> int:
    """Bind the proposal to the exact latest receiver observation QPC."""

    token = getattr(snapshot, "latest_camera_token", None)
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


def _current_target_observation_monotonic_ns(
    snapshot: Any,
    target: Any,
    *,
    abort_type: type[BaseException],
) -> int:
    """Compatibility wrapper for an authoritative current-track proposal."""

    return _target_observation_monotonic_ns(
        snapshot,
        target,
        getattr(snapshot, "current_track", None),
        abort_type=abort_type,
    )


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


def _latest_latched_observation_token(
    *tokens: Optional[CameraFrameToken],
) -> Optional[CameraFrameToken]:
    """Return the latest compatible tracker observation consumed by guidance.

    A proposal replaced before its wire slot still advances visual-state
    lineage, but it never supplies command or passage authority.  Returning
    ``None`` for crossed camera epochs keeps the subsequent latch classifier
    fail-closed.
    """

    latest: Optional[CameraFrameToken] = None
    for token in tokens:
        if token is None:
            continue
        if type(token) is not CameraFrameToken:
            return None
        if latest is None or token == latest:
            latest = token
            continue
        if _token_strictly_newer(token, latest):
            latest = token
            continue
        if not _token_strictly_newer(latest, token):
            return None
    return latest


def _classify_latched_snapshot(
    latch: NearPlaneLatch,
    *,
    previous_camera_token: CameraFrameToken,
    camera_token: CameraFrameToken,
    snapshot: Any,
    current_gate_index: int,
    min_track_confidence: float,
    min_association_confidence: float,
) -> LatchedMeasurementMode:
    """Classify one graph snapshot against an existing near-plane latch."""

    track = getattr(snapshot, "current_track", None)
    center = getattr(track, "center_norm", None)
    velocity = getattr(track, "center_velocity_norm_s", None)
    return classify_latched_measurement(
        latch,
        previous_camera_token=previous_camera_token,
        camera_token=camera_token,
        current_gate_index=getattr(
            snapshot,
            "current_gate_index",
            None,
        ),
        current_track_id=getattr(snapshot, "current_track_id", None),
        track_latest_camera_token=getattr(track, "latest_token", None),
        track_role=getattr(track, "role", None),
        track_authoritative_gate_index=getattr(
            track,
            "authoritative_gate_index",
            current_gate_index,
        ),
        visible=bool(getattr(track, "visible", False)),
        missed_frame_count=getattr(track, "missed_frame_count", -1),
        ambiguous=bool(getattr(track, "ambiguous", True)),
        clipping=getattr(track, "clipping", FrameEdge.NONE),
        center_censored=bool(
            getattr(track, "center_censored", False)
        ),
        normalized_x=(
            None
            if type(center) is not tuple or len(center) != 2
            else center[0]
        ),
        normalized_y_down=(
            None
            if type(center) is not tuple or len(center) != 2
            else center[1]
        ),
        normalized_x_rate_s=(
            None
            if type(velocity) is not tuple or len(velocity) != 2
            else velocity[0]
        ),
        normalized_y_rate_down_s=(
            None
            if type(velocity) is not tuple or len(velocity) != 2
            else velocity[1]
        ),
        apparent_scale=getattr(track, "apparent_scale", None),
        confidence=getattr(track, "confidence", None),
        association_confidence=getattr(
            track,
            "association_confidence",
            None,
        ),
        min_track_confidence=min_track_confidence,
        min_association_confidence=min_association_confidence,
        race_finished=bool(getattr(snapshot, "race_finished", False)),
    )


def _current_snapshot_ready(
    snapshot: Any,
    *,
    gate_index: int,
    track_id: str,
    newer_than: Optional[CameraFrameToken] = None,
    observed_after_ns: Optional[int] = None,
    allow_one_edge_censored: bool = False,
) -> bool:
    track = getattr(snapshot, "current_track", None)
    token = getattr(snapshot, "latest_camera_token", None)
    if newer_than is not None and not _token_strictly_newer(token, newer_than):
        return False
    if observed_after_ns is not None:
        history = getattr(track, "history", None)
        if (
            type(observed_after_ns) is not int
            or observed_after_ns < 0
            or type(history) is not tuple
            or not history
        ):
            return False
        latest = history[-1]
        if (
            getattr(latest, "token", None) != token
            or type(
                getattr(latest, "observation_monotonic_ns", None)
            )
            is not int
            or type(
                getattr(latest, "publication_monotonic_ns", None)
            )
            is not int
            or latest.observation_monotonic_ns <= observed_after_ns
            or latest.publication_monotonic_ns <= observed_after_ns
            or latest.publication_monotonic_ns
            < latest.observation_monotonic_ns
        ):
            return False
    clipping = getattr(track, "clipping", None)
    center_censored = getattr(track, "center_censored", True)
    center = getattr(track, "center_norm", None)
    velocity = getattr(track, "center_velocity_norm_s", None)
    measurement_ready = bool(
        (
            clipping == FrameEdge.NONE
            and center_censored is False
        )
        or (
            allow_one_edge_censored
            and clipping
            in {
                FrameEdge.LEFT,
                FrameEdge.TOP,
                FrameEdge.RIGHT,
                FrameEdge.BOTTOM,
            }
            and type(center) is tuple
            and len(center) == 2
            and type(velocity) is tuple
            and len(velocity) == 2
            and (
                (
                    clipping in {FrameEdge.TOP, FrameEdge.BOTTOM}
                    and type(center[0]) in {int, float}
                    and math.isfinite(float(center[0]))
                    and abs(float(center[0])) <= 1.0
                    and type(velocity[0]) in {int, float}
                    and math.isfinite(float(velocity[0]))
                )
                or (
                    clipping in {FrameEdge.LEFT, FrameEdge.RIGHT}
                    and type(center[1]) in {int, float}
                    and math.isfinite(float(center[1]))
                    and abs(float(center[1])) <= 1.0
                    and type(velocity[1]) in {int, float}
                    and math.isfinite(float(velocity[1]))
                )
            )
        )
    )
    return bool(
        type(token) is CameraFrameToken
        and getattr(snapshot, "current_gate_index", None) == gate_index
        and getattr(snapshot, "current_track_id", None) == track_id
        and getattr(snapshot, "authority_usable", False) is True
        and getattr(snapshot, "race_finished", False) is False
        and track is not None
        and getattr(track, "track_id", None) == track_id
        and getattr(track, "latest_token", None) == token
        and getattr(track, "role", None) is VisualTrackRole.CURRENT
        and getattr(
            track,
            "authoritative_gate_index",
            gate_index,
        )
        == gate_index
        and getattr(track, "visible", False) is True
        and getattr(track, "ambiguous", True) is False
        and getattr(track, "missed_frame_count", 1) == 0
        and measurement_ready
    )


def _dynamic_current_steering_correction_ready(
    snapshot: Any,
    *,
    track_id: str,
) -> bool:
    """Require exact image geometry before ending predicted steering.

    Complete inner geometry can correct both steering and the local aperture
    model.  Clean or one-edge outer support can correct only its observable
    center axis; it cannot create aperture, passage, or race authority.
    """

    track = getattr(snapshot, "current_track", None)
    token = getattr(snapshot, "latest_camera_token", None)
    history = getattr(track, "history", None)
    if (
        type(track_id) is not str
        or not track_id
        or type(token) is not CameraFrameToken
        or track is None
        or getattr(track, "track_id", None) != track_id
        or getattr(track, "latest_token", None) != token
        or getattr(track, "visible", False) is not True
        or getattr(track, "ambiguous", True) is not False
        or getattr(track, "missed_frame_count", 1) != 0
        or getattr(track, "role", None) is not VisualTrackRole.CURRENT
        or type(history) is not tuple
        or not history
    ):
        return False
    sample = history[-1]
    inner = getattr(sample, "inner_aperture", None)
    complete_inner = bool(
        type(inner) is VisualInnerApertureGeometry
        and inner.fitted
        and inner.complete_visibility
        and inner.clipping == FrameEdge.NONE
    )
    clipping = getattr(track, "clipping", None)
    center = getattr(track, "center_norm", None)
    velocity = getattr(track, "center_velocity_norm_s", None)
    outer_axis_ready = bool(
        type(center) is tuple
        and len(center) == 2
        and type(velocity) is tuple
        and len(velocity) == 2
        and (
            (
                clipping == FrameEdge.NONE
                and getattr(track, "center_censored", True) is False
                and all(
                    type(value) in {int, float}
                    and math.isfinite(float(value))
                    for value in (*center, *velocity)
                )
            )
            or (
                clipping in {FrameEdge.TOP, FrameEdge.BOTTOM}
                and type(center[0]) in {int, float}
                and math.isfinite(float(center[0]))
                and type(velocity[0]) in {int, float}
                and math.isfinite(float(velocity[0]))
            )
            or (
                clipping in {FrameEdge.LEFT, FrameEdge.RIGHT}
                and type(center[1]) in {int, float}
                and math.isfinite(float(center[1]))
                and type(velocity[1]) in {int, float}
                and math.isfinite(float(velocity[1]))
            )
        )
    )
    return bool(
        getattr(sample, "token", None) == token
        and (complete_inner or outer_axis_ready)
    )


def _unlatched_atomic_credit_successor_evidence(
    snapshot: Any,
    *,
    current_gate_index: int,
    current_track_id: str,
) -> Optional[Mapping[str, Any]]:
    """Select only one exact graph-vetted identity after atomic race credit.

    This does not create passage or advance authority.  The newer live race
    status owns credit; vision supplies only the adjacent identity needed to
    continue the already-authoritative course lifecycle.
    """

    token = getattr(snapshot, "latest_camera_token", None)
    current = getattr(snapshot, "current_track", None)
    candidates = getattr(snapshot, "next_candidates", ())
    if (
        type(token) is not CameraFrameToken
        or getattr(snapshot, "current_gate_index", None)
        != current_gate_index
        or getattr(snapshot, "current_track_id", None) != current_track_id
        or getattr(snapshot, "race_finished", False) is not False
        or getattr(snapshot, "authority_usable", True) is not False
        or getattr(snapshot, "withholding_reason", None)
        != "current_track_not_visible"
        or getattr(snapshot, "next_selection_ambiguous", True) is not False
        or getattr(snapshot, "provisional_track_ids", ()) != ()
        or current is None
        or getattr(current, "track_id", None) != current_track_id
        or getattr(current, "role", None) is not VisualTrackRole.CURRENT
        or getattr(current, "authoritative_gate_index", None)
        != current_gate_index
        or getattr(current, "visible", True) is not False
        or getattr(current, "ambiguous", True) is not False
        or type(getattr(current, "missed_frame_count", None)) is not int
        or getattr(current, "missed_frame_count", 0) <= 0
        or getattr(current, "clipping", FrameEdge.NONE) == FrameEdge.NONE
        or type(candidates) is not tuple
        or len(candidates) != 1
    ):
        return None
    current_token = getattr(current, "latest_token", None)
    candidate = candidates[0]
    candidate_id = getattr(candidate, "track_id", None)
    if (
        type(current_token) is not CameraFrameToken
        or not _token_strictly_newer(token, current_token)
        or getattr(candidate, "promotable", False) is not True
        or getattr(candidate, "latest_token", None) != token
        or type(candidate_id) is not str
        or not candidate_id
        or candidate_id == current_track_id
        or getattr(candidate, "center_censored", True) is not False
    ):
        return None
    return {
        "basis": "authoritative-credit-clipped-current-successor-identity-v1",
        "camera_token": asdict(token),
        "retired_track_id": current_track_id,
        "reviewed_track_id": candidate_id,
        "current_missed_frame_count": current.missed_frame_count,
        "current_clipping": int(current.clipping),
        "candidate_stable_frame_count": getattr(
            candidate,
            "stable_frame_count",
            None,
        ),
        "candidate_confidence": getattr(candidate, "confidence", None),
        "candidate_association_confidence": getattr(
            candidate,
            "association_confidence",
            None,
        ),
        "passage_authority": False,
        "advance_authority": False,
        "cross_gap_identity_claimed": False,
    }


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
    current_aperture_collective_state: Optional[
        _CurrentApertureProvedCollectiveState
    ] = None
    pending_post_credit_recovery: Optional[
        _PendingPostCreditRecovery
    ] = None
    pending_post_credit_planner: Optional[Any] = None
    pending_post_credit_yaw_reference_rad: Optional[float] = None
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
                "dynamic_controller": (
                    None
                    if runtime.dynamic_controller is None
                    else dict(
                        runtime.dynamic_controller.evidence_summary()
                    )
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
        nonlocal last_command_send_s
        nonlocal consecutive_superseded_proposals
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
        receipt = await host._send_flight_command(
            command,
            require_wire_receipt=True,
        )
        if not isinstance(receipt, Mapping):
            raise abort_type(
                f"{stage} zero command lacks exact outbound receipt"
        )
        host._record_tick(stage, elapsed_s, command)
        total_zero_commands += 1
        last_command_send_s = float(runtime.monotonic())
        consecutive_superseded_proposals = 0
        refresh_live_summary()

    async def send_continuity_hold(
        stage: str,
        elapsed_s: float,
        *,
        yaw_reference_rad: float,
        successor_steering: bool = False,
        require_successor_steering: bool = False,
        command_deadline_s: Optional[float] = None,
        recovery_measurement_mode: Optional[
            PostCreditMeasurementMode
        ] = None,
        recovery_snapshot: Any = None,
    ) -> bool:
        """Bridge one bounded handoff gap with retained dynamic authority."""

        nonlocal last_command_send_s
        nonlocal consecutive_superseded_proposals
        nonlocal total_navigation_commands

        dynamic_controller = runtime.dynamic_controller
        profile = runtime.yaw_profile
        if dynamic_controller is None or profile is None:
            raise abort_type(
                "visual-course continuity hold lacks dynamic/yaw authority"
            )
        if (
            type(successor_steering) is not bool
            or type(require_successor_steering) is not bool
            or require_successor_steering
            and not successor_steering
        ):
            raise abort_type(
                "visual-course successor-steering selection is invalid"
            )
        if (
            recovery_measurement_mode is None
            and recovery_snapshot is not None
            or recovery_measurement_mode is not None
            and (
                type(recovery_measurement_mode)
                is not PostCreditMeasurementMode
                or recovery_measurement_mode
                not in {
                    PostCreditMeasurementMode.ONE_EDGE_CENSORED,
                    PostCreditMeasurementMode.REACQUIRE,
                }
                or recovery_snapshot is None
                or not require_successor_steering
            )
        ):
            raise abort_type(
                "visual-course successor recovery selection is invalid"
            )
        if (
            command_deadline_s is not None
            and (
                type(command_deadline_s) not in {int, float}
                or not math.isfinite(float(command_deadline_s))
            )
        ):
            raise abort_type(
                "visual-course continuity-hold deadline is invalid"
            )
        await host._wait_for_next_flight_command_slot()
        if (
            command_deadline_s is not None
            and float(runtime.monotonic())
            >= float(command_deadline_s)
        ):
            raise abort_type(
                "visual-course continuity-hold authority expired before wire"
            )
        pad_contact = initial_pad_contact_authority()
        host._watchdog(
            require_target=False,
            allow_benign_pad_contact=pad_contact,
            enforce_benign_pad_budget=True,
            count_rate_sample=False,
        )
        excursion, _rates, euler_yaw_rate = _assert_course_attitude_state(
            host,
            yaw_reference_rad=yaw_reference_rad,
            limits=limits,
            yaw_profile=profile,
            abort_type=abort_type,
            phase=f"{stage} pre-send",
        )
        proposal_ns = runtime.perf_counter_ns()
        if type(proposal_ns) is not int or proposal_ns < 0:
            raise abort_type(
                "visual-course continuity-hold clock is invalid"
            )
        actual_successor_steering = successor_steering
        try:
            authority = (
                dynamic_controller.post_credit_successor_steering_authority(
                    now_monotonic_ns=proposal_ns,
                )
                if actual_successor_steering
                and type(dynamic_controller)
                is DynamicVisualCourseSession
                else dynamic_controller.continuity_hold_authority(
                    now_monotonic_ns=proposal_ns,
                    maximum_age_s=profile.control_hold_horizon_s,
                )
            )
        except PostCreditSuccessorSteeringUnavailable as exc:
            if require_successor_steering:
                raise abort_type(
                    "visual-course required propagated successor state "
                    f"expired: {exc}"
                ) from exc
            actual_successor_steering = False
            try:
                authority = dynamic_controller.continuity_hold_authority(
                    now_monotonic_ns=proposal_ns,
                    maximum_age_s=profile.control_hold_horizon_s,
                )
            except (TypeError, ValueError) as exc:
                raise abort_type(
                    "visual-course continuity hold expired after "
                    f"successor steering ended: {exc}"
                ) from exc
        except (TypeError, ValueError) as exc:
            raise abort_type(
                "visual-course "
                + (
                    "post-credit successor steering"
                    if actual_successor_steering
                    else "continuity hold"
                )
                + f" expired: {exc}"
            ) from exc
        if not isinstance(authority, Mapping):
            raise abort_type(
                "visual-course continuity hold authority is invalid"
            )
        target_roll_rad = float(authority["target_roll_rad"])
        target_pitch_rad = float(authority["target_pitch_rad"])
        requested_yaw = float(authority["yaw_rate_rad_s"])
        thrust = float(authority["thrust"])
        top_fov_handoff: Optional[Mapping[str, Any]] = None
        top_fov_observation: Optional[
            _TopFovPropagatedObservation
        ] = None
        top_fov_proposal: Optional[_TopFovPitchProposal] = None
        top_fov_guidance: Optional[Mapping[str, Any]] = None
        top_censored_closure_recovery: Optional[
            _FreshTopCensoredClosureRecovery
        ] = None
        retained_top_censored_closure_recovery: Optional[
            _RetainedFreshTopCensoredClosureRecovery
        ] = None
        if (
            recovery_measurement_mode
            is PostCreditMeasurementMode.ONE_EDGE_CENSORED
        ):
            if (
                type(dynamic_controller)
                is not DynamicVisualCourseSession
                or not actual_successor_steering
            ):
                raise abort_type(
                    "visual-course one-edge recovery lacks dynamic "
                    "successor authority"
                )
            try:
                state = dynamic_controller.core.course_state()
                decision = dynamic_controller.guide(
                    current_track_id=state.current_track_id,
                    successor_track_id=state.successor_track_id,
                    monotonic_ns=proposal_ns,
                )
                if (
                    decision is None
                    or decision.current_gate_index
                    != current_gate_index
                    or decision.current_track_id
                    != authority.get("reviewed_track_id")
                ):
                    raise ValueError(
                        "one-edge FOV decision lost credited current ownership"
                    )
                fov_summary = segment["top_fov_pitch_protection"]
                recovery_clipping = getattr(
                    recovery_snapshot.current_track,
                    "clipping",
                    None,
                )
                try:
                    if _fresh_exact_top_boundary_preempts_propagated_fov(
                        recovery_clipping
                    ):
                        # A fresh exact TOP boundary is physical closure
                        # evidence.  Brake on that publication before a
                        # still-live propagated aperture can keep selecting
                        # increasingly forward pitch.  The exact boundary
                        # guard below remains the fail-closed admission.
                        raise ValueError(
                            "propagated FOV gap lacks a clean propagated "
                            "aperture"
                        )
                    top_fov_handoff = (
                        dynamic_controller
                        .propagated_current_fov_gap_authority(
                            track=recovery_snapshot.current_track,
                            camera_token=(
                                recovery_snapshot.latest_camera_token
                            ),
                            now_monotonic_ns=proposal_ns,
                            allow_tracking_only_inner_raw_clipping=True,
                        )
                    )
                except ValueError as propagated_fov_error:
                    if str(propagated_fov_error) != (
                        "propagated FOV gap lacks a clean propagated aperture"
                    ):
                        raise
                    if recovery_clipping in {
                        FrameEdge.LEFT,
                        FrameEdge.RIGHT,
                    }:
                        (
                            top_fov_proposal,
                            top_fov_guidance,
                        ) = (
                            _fresh_post_credit_horizontal_top_fov_pitch_reference(
                                dynamic_controller,
                                state=state,
                                decision=decision,
                                authority=authority,
                                recovery_snapshot=recovery_snapshot,
                                current_gate_index=current_gate_index,
                                requested_target_pitch_rad=(
                                    target_pitch_rad
                                ),
                            )
                        )
                        target_pitch_rad = (
                            top_fov_proposal.protected_target_pitch_rad
                        )
                    else:
                        recovery_boundary = (
                            _fresh_post_credit_top_boundary_authority(
                                state=state,
                                decision=decision,
                                authority=authority,
                                recovery_snapshot=recovery_snapshot,
                                current_gate_index=current_gate_index,
                            )
                        )
                        recovery_current = recovery_boundary.current
                        recovery_track = recovery_boundary.track
                        recovery_sample = recovery_boundary.sample
                        recovery_config = dynamic_controller.core.config
                        top_censored_closure_recovery = (
                            _allocate_fresh_top_censored_closure_recovery(
                                raw_top_edge_image_down=(
                                    _raw_bbox_top_image_down(
                                        recovery_sample.bbox_norm
                                    )
                                ),
                                clipping=getattr(
                                    recovery_track,
                                    "clipping",
                                    FrameEdge.NONE,
                                ),
                                center_censored=bool(
                                    getattr(
                                        recovery_track,
                                        "center_censored",
                                        False,
                                    )
                                ),
                                current_visible=bool(
                                    recovery_current.visible
                                ),
                                current_ambiguous=bool(
                                    recovery_current.ambiguous
                                ),
                                current_missed_count=int(
                                    recovery_current.missed_count
                                ),
                                current_censored_axes=(
                                    recovery_current.censored_axes
                                ),
                                current_aperture_propagated=bool(
                                    recovery_current.aperture_propagated
                                ),
                                current_aperture_dynamics_qualified=bool(
                                    recovery_current
                                    .aperture_dynamics_qualified
                                ),
                                passage_committed=False,
                                capture_pitch_rad=(
                                    _body_to_reference_pitch_rad(
                                        recovery_current
                                        .body_to_reference_wxyz
                                    )
                                ),
                                body_pitch_rate_rad_s=float(
                                    recovery_current.body_rates_rad_s[1]
                                ),
                                pitch_response_delay_s=float(
                                    recovery_config.pitch_command_delay_s
                                ),
                                stable_center_norm=(
                                    decision.current_center_norm
                                ),
                                residual_rate_rad_s=(
                                    recovery_current
                                    .residual_translational_rate_rad_s
                                ),
                                horizontal_angle_scale_rad=float(
                                    recovery_config
                                    .horizontal_angle_scale_rad
                                ),
                                vertical_angle_scale_rad=float(
                                    recovery_config.vertical_angle_scale_rad
                                ),
                                off_axis_brake_rad=float(
                                    recovery_config.off_axis_brake_rad
                                ),
                                expansion_rate_s=float(
                                    recovery_current.expansion_rate_s
                                ),
                                time_to_contact_s=(
                                    recovery_current.time_to_contact_s
                                ),
                                requested_target_pitch_rad=float(
                                    recovery_config.brake_pitch_rad
                                ),
                                fov_protected_target_pitch_rad=(
                                    target_pitch_rad
                                ),
                                requested_thrust=float(limits.max_thrust),
                                fresh_boundary_current_authority=(
                                    recovery_boundary
                                ),
                            )
                        )
                        if top_censored_closure_recovery is None:
                            raise ValueError(
                                "fresh post-credit TOP boundary lacks bounded "
                                "closure-recovery authority"
                            ) from propagated_fov_error
                        target_pitch_rad = (
                            top_censored_closure_recovery
                            .allocated_target_pitch_rad
                        )
                        thrust = (
                            top_censored_closure_recovery.allocated_thrust
                        )
                        top_fov_guidance = {
                            **asdict(top_censored_closure_recovery),
                            "gate_index": current_gate_index,
                            "track_id": authority["reviewed_track_id"],
                            "camera_token": asdict(
                                recovery_boundary.camera_token
                            ),
                            "expires_monotonic_ns": authority[
                                "expires_monotonic_ns"
                            ],
                            "source_target_pitch_rad": float(
                                authority["target_pitch_rad"]
                            ),
                            "source_thrust": float(authority["thrust"]),
                            "propagated_aperture_available": False,
                            "steering_only": True,
                            "passage_authority": False,
                            "advance_authority": False,
                        }
                else:
                    if not isinstance(top_fov_handoff, Mapping):
                        raise ValueError(
                            "one-edge recovery lacks local FOV state"
                        )
                    if bool(
                        FrameEdge(int(top_fov_handoff["clipping"]))
                        & FrameEdge.TOP
                    ):
                        (
                            top_fov_observation,
                            top_fov_proposal,
                        ) = _propose_propagated_top_fov_pitch_reference(
                            top_fov_handoff,
                            requested_target_pitch_rad=target_pitch_rad,
                            prior_target_pitch_rad=(
                                fov_summary[
                                    "last_protected_target_pitch_rad"
                                ]
                            ),
                        )
                        target_pitch_rad = (
                            top_fov_proposal.protected_target_pitch_rad
                        )
                        top_fov_guidance = {
                            "basis": TOP_FOV_PITCH_PROTECTION_BASIS,
                            "track_id": authority["reviewed_track_id"],
                            "safe_top_edge_image_down": (
                                TOP_FOV_SAFE_EDGE_IMAGE_DOWN
                            ),
                            "geometry_basis": (
                                top_fov_observation.geometry_basis
                            ),
                            "projected_nominal_top_edge_image_down": (
                                top_fov_observation
                                .projected_nominal_top_edge_image_down
                            ),
                            "projected_top_edge_std_image_down": (
                                top_fov_observation
                                .projected_top_edge_std_image_down
                            ),
                            "prediction_horizon_remaining_s": (
                                top_fov_observation
                                .prediction_horizon_remaining_s
                            ),
                            **asdict(top_fov_proposal),
                            "steering_only": True,
                            "passage_authority": False,
                            "advance_authority": False,
                        }
                    else:
                        retained = (
                            _retain_post_credit_top_fov_pitch_reference(
                                authority,
                                fov_summary,
                            )
                        )
                        if retained is not None:
                            target_pitch_rad, top_fov_guidance = retained
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise abort_type(
                    "visual-course post-credit TOP-FOV pitch guidance "
                    f"refused: {exc}"
                ) from exc
        elif (
            recovery_measurement_mode
            is PostCreditMeasurementMode.REACQUIRE
        ):
            fov_summary = segment["top_fov_pitch_protection"]
            try:
                retained_top_censored_closure_recovery = (
                    _retain_fresh_top_censored_closure_recovery(
                        authority=authority,
                        fov_summary=fov_summary,
                        recovery_snapshot=recovery_snapshot,
                        current_gate_index=current_gate_index,
                        now_monotonic_ns=proposal_ns,
                        requested_target_pitch_rad=target_pitch_rad,
                        requested_thrust=thrust,
                    )
                )
                retained = None
                if retained_top_censored_closure_recovery is None:
                    retained = _retain_post_credit_top_fov_pitch_reference(
                        authority,
                        fov_summary,
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise abort_type(
                    "visual-course retained post-credit TOP-FOV "
                    f"authority refused: {exc}"
                ) from exc
            if retained_top_censored_closure_recovery is not None:
                target_pitch_rad = (
                    retained_top_censored_closure_recovery
                    .allocated_target_pitch_rad
                )
                thrust = (
                    retained_top_censored_closure_recovery.allocated_thrust
                )
                top_fov_guidance = asdict(
                    retained_top_censored_closure_recovery
                )
            elif retained is not None:
                target_pitch_rad, top_fov_guidance = retained
        bounded_yaw = requested_yaw
        if requested_yaw != 0.0:
            bounded_yaw = _limit_calibrated_yaw_request(
                requested_yaw,
                excursion_rad=excursion,
                measured_euler_yaw_rate_rad_s=euler_yaw_rate,
                limits=limits,
                profile=profile,
                abort_type=abort_type,
            )
        base = runtime.attitude_rate_command(
            host.estimate,
            target_roll_rad=target_roll_rad,
            target_pitch_rad=target_pitch_rad,
            thrust=thrust,
        )
        base, roll_yaw_transport_rate = _allocate_roll_yaw_transport(
            base,
            target_roll_rad=target_roll_rad,
            target_pitch_rad=target_pitch_rad,
            bounded_yaw_rate_rad_s=bounded_yaw,
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
        yaw_safety_override = bool(
            requested_yaw != 0.0 and bounded_yaw == 0.0
        )
        try:
            command = dynamic_controller.govern_wire_command(
                command,
                proposal_monotonic_ns=proposal_ns,
                launch_thrust_override=not actual_successor_steering,
                yaw_safety_override=yaw_safety_override,
            )
        except (TypeError, ValueError) as exc:
            raise abort_type(
                f"visual-course continuity governor refused: {exc}"
            ) from exc
        runtime.validate_command(command)
        if (
            max(abs(command.roll_rate), abs(command.pitch_rate))
            > limits.max_command_rate_rad_s + 1e-12
            or abs(command.yaw_rate)
            > limits.max_yaw_rate_rad_s + 1e-12
            or not limits.min_thrust <= command.thrust <= limits.max_thrust
            or (
                not actual_successor_steering
                and command.thrust != thrust
            )
        ):
            raise abort_type(
                "visual-course continuity hold escaped its envelope"
            )
        receipt = await host._send_flight_command(
            command,
            require_wire_receipt=True,
        )
        call_start = (
            receipt.get("call_start_monotonic_ns")
            if isinstance(receipt, Mapping)
            else None
        )
        if (
            type(call_start) is not int
            or call_start < proposal_ns
            or host._last_flight_command_started_ns != call_start
        ):
            raise abort_type(
                "visual-course continuity hold lacks exact wire timing"
            )
        try:
            dynamic_evidence = (
                dynamic_controller.record_wire_acceptance(
                    target_roll_rad=target_roll_rad,
                    target_pitch_rad=target_pitch_rad,
                    yaw_rate_rad_s=float(command.yaw_rate),
                    thrust=float(command.thrust),
                    wire_command=command,
                    wire_start_monotonic_ns=call_start,
                    requested_thrust=thrust,
                    thrust_slew_override=not actual_successor_steering,
                    yaw_slew_override=yaw_safety_override,
                )
            )
        except (TypeError, ValueError) as exc:
            raise abort_type(
                "visual-course continuity hold could not commit: "
                f"{exc}"
            ) from exc
        if not isinstance(dynamic_evidence, Mapping):
            raise abort_type(
                "visual-course continuity hold evidence is invalid"
            )
        accepted_dynamic_evidence = dict(dynamic_evidence)
        accepted_dynamic_evidence["roll_yaw_transport_rate_rad_s"] = (
            roll_yaw_transport_rate
        )
        if top_fov_guidance is not None:
            accepted_dynamic_evidence["top_fov_pitch_guidance"] = dict(
                top_fov_guidance
            )
        host.recorder.emit(
            "visual_course_dynamic_command",
            **accepted_dynamic_evidence,
        )
        if actual_successor_steering:
            steering_evidence = {
                key: (
                    asdict(value)
                    if type(value) is AttitudeRateCommand
                    else value
                )
                for key, value in authority.items()
            }
            if top_fov_guidance is not None:
                steering_evidence[
                    "target_pitch_rad_before_top_fov"
                ] = float(authority["target_pitch_rad"])
                steering_evidence["target_pitch_rad"] = target_pitch_rad
                steering_evidence["top_fov_pitch_guidance"] = dict(
                    top_fov_guidance
                )
                fov_summary = segment["top_fov_pitch_protection"]
                if top_censored_closure_recovery is not None:
                    fov_summary["last_exact_top_closure_recovery"] = {
                        "basis": (
                            RETAINED_FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS
                        ),
                        "source_basis": (
                            FRESH_TOP_CENSORED_CLOSURE_RECOVERY_BASIS
                        ),
                        "gate_index": current_gate_index,
                        "track_id": authority["reviewed_track_id"],
                        "camera_token": dict(
                            top_fov_guidance["camera_token"]
                        ),
                        "source_wire_start_monotonic_ns": call_start,
                        "expires_monotonic_ns": int(
                            authority["expires_monotonic_ns"]
                        ),
                        "target_pitch_floor_rad": target_pitch_rad,
                        "thrust_floor": thrust,
                        "steering_only": True,
                        "passage_authority": False,
                        "advance_authority": False,
                    }
                elif retained_top_censored_closure_recovery is not None:
                    fov_summary[
                        "retained_exact_top_closure_recovery_command_count"
                    ] = int(
                        fov_summary[
                            "retained_exact_top_closure_recovery_command_count"
                        ]
                    ) + 1
                    fov_summary[
                        "last_retained_exact_top_closure_recovery"
                    ] = asdict(
                        retained_top_censored_closure_recovery
                    )
                elif recovery_measurement_mode is not None:
                    fov_summary["last_exact_top_closure_recovery"] = None
                if (
                    top_fov_proposal is not None
                    and top_fov_handoff is not None
                ):
                    assert top_fov_handoff is not None
                    assert top_fov_observation is not None
                    if top_fov_proposal.limited:
                        fov_summary["limited_command_count"] += 1
                    fov_summary[
                        "propagated_state_handoff_command_count"
                    ] = int(
                        fov_summary[
                            "propagated_state_handoff_command_count"
                        ]
                    ) + 1
                    fov_summary["last_propagated_state_handoff"] = dict(
                        top_fov_handoff
                    )
                    fov_summary.update(
                        {
                            "last_track_id": authority[
                                "reviewed_track_id"
                            ],
                            "last_camera_token": dict(
                                top_fov_handoff["camera_token"]
                            ),
                            "last_wire_start_monotonic_ns": call_start,
                            "last_forecast_top_edge_image_down": (
                                top_fov_proposal
                                .forecast_top_edge_image_down
                            ),
                            "last_protected_target_pitch_rad": (
                                top_fov_proposal
                                .protected_target_pitch_rad
                            ),
                            "active": top_fov_proposal.active_after,
                        }
                    )
                    host.recorder.emit(
                        "visual_course_dynamic_fov_gap_handoff",
                        gate_index=current_gate_index,
                        stage=stage,
                        camera_token=dict(
                            top_fov_handoff["camera_token"]
                        ),
                        authority=dict(top_fov_handoff),
                        pitch_guidance=dict(top_fov_guidance),
                        command=asdict(command),
                    )
                elif top_fov_proposal is not None:
                    if top_fov_proposal.limited:
                        fov_summary["limited_command_count"] += 1
                    fov_summary.update(
                        {
                            "last_track_id": authority[
                                "reviewed_track_id"
                            ],
                            "last_camera_token": dict(
                                top_fov_guidance["camera_token"]
                            ),
                            "last_wire_start_monotonic_ns": call_start,
                            "last_forecast_top_edge_image_down": (
                                top_fov_proposal
                                .forecast_top_edge_image_down
                            ),
                            "last_protected_target_pitch_rad": (
                                top_fov_proposal
                                .protected_target_pitch_rad
                            ),
                            "active": top_fov_proposal.active_after,
                        }
                    )
                else:
                    fov_summary.update(
                        {
                            "last_wire_start_monotonic_ns": call_start,
                            "last_protected_target_pitch_rad": (
                                target_pitch_rad
                            ),
                        }
                    )
            elif recovery_measurement_mode is not None:
                segment["top_fov_pitch_protection"][
                    "last_exact_top_closure_recovery"
                ] = None
            host.recorder.emit(
                "visual_course_dynamic_post_credit_successor_steering",
                gate_index=current_gate_index,
                stage=stage,
                wire_start_monotonic_ns=call_start,
                command=asdict(command),
                authority=steering_evidence,
            )
            total_navigation_commands += 1
        else:
            host.recorder.emit(
                "visual_course_dynamic_handoff_hold",
                gate_index=current_gate_index,
                stage=stage,
                source_wire_start_monotonic_ns=authority[
                    "source_wire_start_monotonic_ns"
                ],
                wire_start_monotonic_ns=call_start,
                command=asdict(command),
            )
        host._record_tick(stage, elapsed_s, command)
        last_command_send_s = float(runtime.monotonic())
        consecutive_superseded_proposals = 0
        segment["dynamic_controller"] = dict(
            dynamic_controller.evidence_summary()
        )
        refresh_live_summary()
        return actual_successor_steering

    async def send_visual(
        *,
        proposal: Any,
        snapshot: Any,
        yaw_reference_rad: float,
        segment_started_s: float,
        stage: str,
        target_track: Any = None,
        apply_launch_bootstrap: bool = True,
        command_deadline_s: Optional[float] = None,
        refresh_ingress_after_slot: bool = False,
        intercept_response_authority: float = 0.0,
        top_fov_transition_owned: bool = False,
        horizontal_fov_closure_brake_enabled: bool = False,
        committed_crossing_authority: Optional[
            _CensoredPassageCoastAuthority
        ] = None,
    ) -> _AcceptedVisualCommand | _SupersededVisualProposal:
        nonlocal total_navigation_commands
        nonlocal last_command_send_s
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
            if not math.isfinite(now_s) or now_s < last_command_send_s:
                raise abort_type(
                    "visual-course supersession clock regressed"
                ) from exc
            consecutive_superseded_proposals += 1
            segment["superseded_proposal_count"] = int(
                segment["superseded_proposal_count"]
            ) + 1
            hold_s = now_s - last_command_send_s
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
            return _SupersededVisualProposal(
                expected_camera_token=expected_token,
                receiver_camera_token=receiver_token,
                held_previous_command_s=hold_s,
                consecutive_count=consecutive_superseded_proposals,
            )

        if type(apply_launch_bootstrap) is not bool:
            raise abort_type(
                "visual-course launch-bootstrap selection is invalid"
            )
        if type(top_fov_transition_owned) is not bool:
            raise abort_type(
                "visual-course top-FOV transition ownership is invalid"
            )
        if type(horizontal_fov_closure_brake_enabled) is not bool:
            raise abort_type(
                "visual-course horizontal-FOV brake selection is invalid"
            )
        if (
            committed_crossing_authority is not None
            and (
                type(committed_crossing_authority)
                is not _CensoredPassageCoastAuthority
                or committed_crossing_authority.gate_index
                != current_gate_index
                or committed_crossing_authority.track_id
                != current_track_id
                or not top_fov_transition_owned
            )
        ):
            raise abort_type(
                "visual-course committed crossing reference is invalid"
            )
        if (
            command_deadline_s is not None
            and (
                type(command_deadline_s) not in {int, float}
                or not math.isfinite(float(command_deadline_s))
            )
        ):
            raise abort_type(
                "visual-course command deadline is invalid"
            )
        output = proposal.servo_output
        target_track = (
            getattr(snapshot, "current_track", None)
            if target_track is None
            else target_track
        )
        observation_monotonic_ns = (
            _target_observation_monotonic_ns(
                snapshot,
                proposal.current_target,
                target_track,
                abort_type=abort_type,
            )
        )
        current_history = getattr(
            target_track,
            "history",
            None,
        )
        publication_monotonic_ns = (
            None
            if type(current_history) is not tuple or not current_history
            else getattr(
                current_history[-1],
                "publication_monotonic_ns",
                None,
            )
        )
        if (
            type(publication_monotonic_ns) is not int
            or publication_monotonic_ns < observation_monotonic_ns
        ):
            raise abort_type(
                "visual-course target lacks exact publication provenance"
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
        proved_collective: Optional[float] = None
        proved_filtered_vertical_rate: Optional[float] = None
        collective_evidence: Optional[Dict[str, Any]] = None
        collective_proposal: Optional[
            _CurrentApertureCollectiveProposal
        ] = None
        if runtime.dynamic_controller is not None:
            assert current_aperture_collective_state is not None
            control_vertical_error_image_down: Optional[float] = None
            control_vertical_rate_down_s: Optional[float] = None
            control_basis = RAW_CURRENT_APERTURE_COLLECTIVE_BASIS
            current_aperture_observable = True
            dynamic_decision = runtime.dynamic_controller.last_decision
            if dynamic_decision is not None:
                dynamic_course = (
                    runtime.dynamic_controller.core.course_state()
                )
                if (
                    dynamic_decision.current_track_id
                    != current_track_id
                    or dynamic_course.current_track_id
                    != current_track_id
                ):
                    raise abort_type(
                        "current-aperture collective dynamic identity changed"
                    )
                current_dynamic = dynamic_course.current
                current_aperture_observable = bool(
                    getattr(current_dynamic, "visible", True)
                    and not getattr(
                        current_dynamic,
                        "ambiguous",
                        False,
                    )
                    and not any(
                        getattr(
                            current_dynamic,
                            "censored_axes",
                            (False, False),
                        )
                    )
                )
                if current_aperture_observable:
                    (
                        control_vertical_error_image_down,
                        control_vertical_rate_down_s,
                        control_basis,
                    ) = _dynamic_current_aperture_collective_inputs(
                        dynamic_decision,
                        current_dynamic,
                        vertical_angle_scale_rad=(
                            runtime.dynamic_controller.core.config
                            .vertical_angle_scale_rad
                        ),
                    )
            collective_proposal = (
                _propose_current_aperture_collective(
                    current_aperture_collective_state,
                    proposal.current_target,
                    authoritative_current_track_id=current_track_id,
                    control_vertical_error_image_down=(
                        control_vertical_error_image_down
                    ),
                    control_vertical_rate_down_s=(
                        control_vertical_rate_down_s
                    ),
                    control_basis=control_basis,
                    current_aperture_observable=(
                        current_aperture_observable
                    ),
                    subsupport_collective_authorized=(
                        top_fov_transition_owned
                    ),
                )
            )
            proved_collective = (
                collective_proposal.requested_thrust
            )
            proved_filtered_vertical_rate = (
                collective_proposal.filtered_vertical_rate_down_s
            )
        elif launch["enabled"] and apply_launch_bootstrap:
            assert current_aperture_collective_state is not None
            (
                proved_collective,
                proved_filtered_vertical_rate,
            ) = current_aperture_collective_state.observe(
                proposal.current_target
            )
        if runtime.dynamic_controller is not None:
            assert proved_collective is not None
            assert proved_filtered_vertical_rate is not None
            assert collective_proposal is not None
            # Dynamic pitch owns forward closure.  Current-aperture image-down
            # error and its exact-frame filtered rate exclusively own
            # collective after any launch-only boost.
            command_thrust = proved_collective
            collective_evidence = {
                "basis": collective_proposal.control_basis,
                "gate_index": current_gate_index,
                "authority_track_id": current_track_id,
                "observation_track_id": (
                    proposal.current_target.track_id
                ),
                "current_vertical_error_image_down": (
                    None
                    if collective_proposal.current_aperture_dropout
                    else float(
                        proposal.current_target.normalized_y_down
                    )
                ),
                "current_vertical_rate_down_s": (
                    None
                    if collective_proposal.current_aperture_dropout
                    else float(
                        proposal.current_target
                        .normalized_y_rate_down_s
                    )
                ),
                "proved_filtered_vertical_rate_down_s": (
                    proved_filtered_vertical_rate
                ),
                "control_vertical_error_image_down": (
                    collective_proposal
                    .control_vertical_error_image_down
                ),
                "control_vertical_rate_down_s": (
                    proved_filtered_vertical_rate
                ),
                "requested_thrust": proved_collective,
                "unconstrained_requested_thrust": (
                    collective_proposal
                    .unconstrained_requested_thrust
                ),
                "noncommitted_support_floor_applied": (
                    collective_proposal
                    .noncommitted_support_floor_applied
                ),
                "subsupport_collective_authorized": (
                    collective_proposal
                    .subsupport_collective_authorized
                ),
                "vertical_censored": (
                    collective_proposal.vertical_censored
                ),
                "current_aperture_dropout": (
                    collective_proposal.current_aperture_dropout
                ),
                "held_last_observable_collective": (
                    collective_proposal
                    .held_last_observable_collective
                ),
            }
        launch_evidence: Optional[Dict[str, Any]] = None
        if launch["enabled"] and apply_launch_bootstrap:
            assert launch_spawn_pitch_rad is not None
            launch_elapsed_s = max(
                0.0,
                float(runtime.monotonic()) - course_started_s,
            )
            target_pitch_rad, pitch_blend = (
                _allocate_launch_pitch_target(
                    spawn_pitch_rad=launch_spawn_pitch_rad,
                    responsive_target_pitch_rad=target_pitch_rad,
                    launch_elapsed_s=launch_elapsed_s,
                )
            )
            assert proved_collective is not None
            assert proved_filtered_vertical_rate is not None
            next_preview_collective_delta = 0.0
            next_preview_collective_track_id: Optional[str] = None
            command_thrust, thrust_phase = _allocate_launch_collective(
                launch_elapsed_s=launch_elapsed_s,
                post_preload_thrust=command_thrust,
                configured_boost_duration_s=float(
                    host.visual_config.lifecycle.launch_boost_duration_s
                ),
                configured_boost_thrust=float(
                    host.visual_config.lifecycle.launch_boost_thrust
                ),
                dynamic_collective_owns_post_preload=bool(
                    runtime.dynamic_controller is not None
                ),
            )
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
        top_fov_proposal: Optional[_TopFovPitchProposal] = None
        top_fov_observation: Optional[_TopFovObservation] = None
        top_fov_propagated_handoff: Optional[Mapping[str, Any]] = None
        top_fov_propagated_observation: Optional[
            _TopFovPropagatedObservation
        ] = None
        top_fov_propagated_proposal: Optional[
            _TopFovPitchProposal
        ] = None
        top_fov_retained_raw_handoff: Optional[Mapping[str, Any]] = None
        top_fov_retained_raw_observation: Optional[
            _TopFovRetainedRawStateObservation
        ] = None
        top_fov_retained_raw_proposal: Optional[
            _TopFovPitchProposal
        ] = None
        top_censored_closure_recovery: Optional[
            _FreshTopCensoredClosureRecovery
        ] = None
        top_censored_pitch_arbitration: Optional[Dict[str, Any]] = None
        horizontal_fov_closure_brake: Optional[
            _FreshHorizontalFovClosureBrake
        ] = None
        requested_pitch_before_top_fov = target_pitch_rad
        top_fov_track_id: Optional[str] = None
        dynamic_controller = runtime.dynamic_controller
        if type(dynamic_controller) is DynamicVisualCourseSession:
            top_fov_track_id = getattr(target_track, "track_id", None)
            if (
                type(top_fov_track_id) is not str
                or not top_fov_track_id
            ):
                raise abort_type(
                    "visual-course top-FOV target identity is invalid"
                )
            fov_summary = segment["top_fov_pitch_protection"]
            if not top_fov_transition_owned:
                try:
                    top_fov_observation = _top_fov_observation(
                        dynamic_controller,
                        target_track,
                        snapshot.latest_camera_token,
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    handoff_ns = runtime.perf_counter_ns()
                    if type(handoff_ns) is not int or handoff_ns < 0:
                        raise abort_type(
                            "visual-course propagated FOV-handoff clock is "
                            "invalid"
                        ) from exc
                    try:
                        top_fov_propagated_handoff = (
                            dynamic_controller
                            .propagated_current_fov_gap_authority(
                                track=target_track,
                                camera_token=(
                                    snapshot.latest_camera_token
                                ),
                                now_monotonic_ns=handoff_ns,
                            )
                        )
                        if not isinstance(
                            top_fov_propagated_handoff,
                            Mapping,
                        ):
                            raise ValueError(
                                "propagated FOV-handoff evidence is invalid"
                            )
                        (
                            top_fov_propagated_observation,
                            top_fov_propagated_proposal,
                        ) = _propose_propagated_top_fov_pitch_reference(
                            top_fov_propagated_handoff,
                            requested_target_pitch_rad=target_pitch_rad,
                            prior_target_pitch_rad=(
                                fov_summary[
                                    "last_protected_target_pitch_rad"
                                ]
                            ),
                        )
                        target_pitch_rad = (
                            top_fov_propagated_proposal
                            .protected_target_pitch_rad
                        )
                    except (
                        AttributeError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ):
                        try:
                            (
                                top_fov_retained_raw_observation,
                                top_fov_retained_raw_proposal,
                                top_fov_retained_raw_handoff,
                            ) = (
                                _propose_retained_raw_top_fov_pitch_reference(
                                    dynamic_controller,
                                    target_track,
                                    snapshot.latest_camera_token,
                                    fov_summary=fov_summary,
                                    now_monotonic_ns=handoff_ns,
                                    requested_target_pitch_rad=(
                                        target_pitch_rad
                                    ),
                                )
                            )
                        except (
                            AttributeError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ):
                            raise abort_type(
                                "visual-course top-FOV pitch guidance "
                                f"refused: {exc}"
                            ) from exc
                        target_pitch_rad = (
                            top_fov_retained_raw_proposal
                            .protected_target_pitch_rad
                        )
                    if launch_evidence is not None:
                        launch_evidence[
                            "target_pitch_rad_before_top_fov"
                        ] = launch_evidence["target_pitch_rad"]
                        launch_evidence["target_pitch_rad"] = (
                            target_pitch_rad
                        )
                if top_fov_observation is not None:
                    try:
                        prior_target_pitch_rad = (
                            fov_summary["last_protected_target_pitch_rad"]
                            if fov_summary[
                                "last_protected_target_pitch_rad"
                            ]
                            is not None
                            else (
                                top_fov_observation
                                .previous_target_pitch_rad
                                if (
                                    top_fov_observation
                                    .previous_target_pitch_rad
                                    is not None
                                )
                                else top_fov_observation.capture_pitch_rad
                            )
                        )
                        top_fov_proposal = (
                            _propose_top_fov_pitch_reference(
                                capture_pitch_rad=(
                                    top_fov_observation.capture_pitch_rad
                                ),
                                raw_top_edge_image_down=(
                                    top_fov_observation
                                    .raw_top_edge_image_down
                                ),
                                raw_top_edge_rate_down_s=(
                                    top_fov_observation
                                    .raw_top_edge_rate_down_s
                                ),
                                requested_target_pitch_rad=(
                                    target_pitch_rad
                                ),
                                prior_target_pitch_rad=(
                                    prior_target_pitch_rad
                                ),
                                vertical_angle_scale_rad=(
                                    top_fov_observation
                                    .vertical_angle_scale_rad
                                ),
                                active_before=bool(
                                    fov_summary["active"]
                                ),
                                raw_top_edge_nonrotational_angle_rate_rad_s=(
                                    top_fov_observation
                                    .raw_top_edge_nonrotational_angle_rate_rad_s
                                ),
                                prediction_horizon_s=(
                                    top_fov_observation
                                    .pitch_response_delay_s
                                ),
                            )
                        )
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise abort_type(
                            "visual-course top-FOV pitch guidance refused: "
                            f"{exc}"
                        ) from exc
                    target_pitch_rad = (
                        top_fov_proposal.protected_target_pitch_rad
                    )
                    if launch_evidence is not None:
                        launch_evidence[
                            "target_pitch_rad_before_top_fov"
                        ] = launch_evidence["target_pitch_rad"]
                        launch_evidence["target_pitch_rad"] = (
                            target_pitch_rad
                        )
            fov_reference = (
                top_fov_proposal
                or top_fov_propagated_proposal
                or top_fov_retained_raw_proposal
            )
            if (
                fov_reference is not None
                and committed_crossing_authority is None
                and getattr(target_track, "clipping", None)
                is FrameEdge.TOP
                and getattr(target_track, "center_censored", None)
                is True
            ):
                try:
                    recovery_course = (
                        dynamic_controller.core.course_state()
                    )
                    recovery_current = recovery_course.current
                    recovery_decision = dynamic_controller.last_decision
                    recovery_history = getattr(
                        target_track,
                        "history",
                        None,
                    )
                    recovery_sample = (
                        None
                        if type(recovery_history) is not tuple
                        or not recovery_history
                        else recovery_history[-1]
                    )
                    if (
                        recovery_decision is None
                        or recovery_sample is None
                        or recovery_course.current_gate_index
                        != current_gate_index
                        or recovery_course.current_track_id
                        != current_track_id
                        or recovery_current.track_id
                        != current_track_id
                        or getattr(target_track, "track_id", None)
                        != current_track_id
                        or recovery_sample.token
                        != snapshot.latest_camera_token
                        or recovery_current.frame_sequence
                        != recovery_sample.tracker_frame_sequence
                        or recovery_current.stream_generation
                        != snapshot.latest_camera_token.generation
                    ):
                        raise ValueError(
                            "fresh TOP recovery differs from current lineage"
                        )
                    recovery_config = dynamic_controller.core.config
                    fresh_top_boundary = (
                        _fresh_current_top_boundary_authority(
                            dynamic_controller,
                            snapshot=snapshot,
                            current_gate_index=current_gate_index,
                            current_track_id=current_track_id,
                        )
                    )
                    top_censored_closure_recovery = (
                        _allocate_fresh_top_censored_closure_recovery(
                            raw_top_edge_image_down=(
                                _raw_bbox_top_image_down(
                                    recovery_sample.bbox_norm
                                )
                            ),
                            clipping=getattr(
                                target_track,
                                "clipping",
                                FrameEdge.NONE,
                            ),
                            center_censored=bool(
                                getattr(
                                    target_track,
                                    "center_censored",
                                    False,
                                )
                            ),
                            current_visible=bool(
                                recovery_current.visible
                            ),
                            current_ambiguous=bool(
                                recovery_current.ambiguous
                            ),
                            current_missed_count=int(
                                recovery_current.missed_count
                            ),
                            current_censored_axes=(
                                recovery_current.censored_axes
                            ),
                            current_aperture_propagated=bool(
                                recovery_current.aperture_propagated
                            ),
                            current_aperture_dynamics_qualified=bool(
                                recovery_current
                                .aperture_dynamics_qualified
                            ),
                            passage_committed=bool(
                                proposal.mode
                                is VisualApproachMode.PASSAGE
                            ),
                            capture_pitch_rad=(
                                _body_to_reference_pitch_rad(
                                    recovery_current
                                    .body_to_reference_wxyz
                                )
                            ),
                            body_pitch_rate_rad_s=float(
                                recovery_current.body_rates_rad_s[1]
                            ),
                            pitch_response_delay_s=float(
                                recovery_config.pitch_command_delay_s
                            ),
                            stable_center_norm=(
                                recovery_decision.current_center_norm
                            ),
                            residual_rate_rad_s=(
                                recovery_current
                                .residual_translational_rate_rad_s
                            ),
                            horizontal_angle_scale_rad=float(
                                recovery_config
                                .horizontal_angle_scale_rad
                            ),
                            vertical_angle_scale_rad=float(
                                recovery_config.vertical_angle_scale_rad
                            ),
                            off_axis_brake_rad=float(
                                recovery_config.off_axis_brake_rad
                            ),
                            expansion_rate_s=float(
                                recovery_current.expansion_rate_s
                            ),
                            time_to_contact_s=(
                                recovery_current.time_to_contact_s
                            ),
                            requested_target_pitch_rad=(
                                requested_pitch_before_top_fov
                            ),
                            fov_protected_target_pitch_rad=(
                                fov_reference
                                .protected_target_pitch_rad
                            ),
                            requested_thrust=command_thrust,
                            fresh_boundary_current_authority=(
                                fresh_top_boundary
                            ),
                        )
                    )
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise abort_type(
                        "visual-course fresh TOP-censored recovery "
                        f"refused: {exc}"
                    ) from exc
                if top_censored_closure_recovery is not None:
                    pitch_priority_proposal = (
                        top_fov_proposal
                        if top_fov_proposal is not None
                        else top_fov_retained_raw_proposal
                    )
                    pitch_priority_retained_handoff = (
                        None
                        if top_fov_proposal is not None
                        else top_fov_retained_raw_handoff
                    )
                    fov_owns_pitch = (
                        pitch_priority_proposal is not None
                        and _nonrapid_off_axis_top_fov_owns_pitch(
                            mode=proposal.mode,
                            fov_proposal=pitch_priority_proposal,
                            fresh_top_boundary=fresh_top_boundary,
                            closure_recovery=(
                                top_censored_closure_recovery
                            ),
                            rapid_expansion_rate_s=float(
                                recovery_config.rapid_expansion_rate_s
                            ),
                            rapid_closure_ttc_s=float(
                                recovery_config.successor_lookahead_ttc_s
                            ),
                            retained_raw_handoff=(
                                pitch_priority_retained_handoff
                            ),
                        )
                    )
                    if fov_owns_pitch:
                        assert pitch_priority_proposal is not None
                        top_censored_pitch_arbitration = {
                            "basis": (
                                NONRAPID_OFF_AXIS_TOP_FOV_PRIORITY_BASIS
                            ),
                            "lifecycle_mode": proposal.mode.value,
                            "fov_authority_kind": (
                                "exact_same_publication"
                                if pitch_priority_retained_handoff is None
                                else "fixed_retained_raw_lease"
                            ),
                            "closure_recovery": asdict(
                                top_censored_closure_recovery
                            ),
                            "rapid_expansion_rate_s": float(
                                recovery_config.rapid_expansion_rate_s
                            ),
                            "rapid_closure_ttc_s": float(
                                recovery_config.successor_lookahead_ttc_s
                            ),
                            "selected_target_pitch_rad": (
                                pitch_priority_proposal
                                .protected_target_pitch_rad
                            ),
                            "retained_raw_handoff": (
                                None
                                if pitch_priority_retained_handoff is None
                                else dict(
                                    pitch_priority_retained_handoff
                                )
                            ),
                            "steering_only": True,
                            "passage_authority": False,
                            "advance_authority": False,
                        }
                        top_censored_closure_recovery = None
                    else:
                        target_pitch_rad = (
                            top_censored_closure_recovery
                            .allocated_target_pitch_rad
                        )
                        command_thrust = (
                            top_censored_closure_recovery.allocated_thrust
                        )
                        if launch_evidence is not None:
                            launch_evidence["target_pitch_rad"] = (
                                target_pitch_rad
                            )
                            launch_evidence["thrust"] = command_thrust
            if (
                horizontal_fov_closure_brake_enabled
                and committed_crossing_authority is None
                and not top_fov_transition_owned
                and getattr(target_track, "clipping", None)
                is FrameEdge.NONE
                and getattr(target_track, "center_censored", None)
                is False
                and bool(getattr(target_track, "visible", False))
                and not bool(getattr(target_track, "ambiguous", True))
                and getattr(target_track, "missed_frame_count", None) == 0
            ):
                try:
                    horizontal_history = getattr(
                        target_track,
                        "history",
                        None,
                    )
                    horizontal_sample = (
                        None
                        if type(horizontal_history) is not tuple
                        or not horizontal_history
                        else horizontal_history[-1]
                    )
                    if (
                        top_fov_observation is None
                        or top_fov_proposal is None
                        or horizontal_sample is None
                    ):
                        raise ValueError(
                            "fresh horizontal-FOV brake lacks the admitted "
                            "raw current"
                        )
                    # _top_fov_observation has already proved this exact
                    # sample/token against the dynamic authoritative CURRENT.
                    # Reuse that admission instead of maintaining a second,
                    # subtly different lineage contract here.
                    horizontal_fov_closure_brake = (
                        _allocate_fresh_horizontal_fov_closure_brake(
                            bbox_norm_ltrb=horizontal_sample.bbox_norm,
                            center_velocity_norm_s=(
                                target_track.center_velocity_norm_s
                            ),
                            log_scale_rate_s=(
                                target_track.log_scale_rate_s
                            ),
                            clipping=target_track.clipping,
                            center_censored=(
                                target_track.center_censored
                            ),
                            current_visible=target_track.visible,
                            current_ambiguous=(
                                target_track.ambiguous
                            ),
                            current_missed_count=(
                                target_track.missed_frame_count
                            ),
                            current_censored_axes=(
                                bool(
                                    proposal.current_target
                                    .horizontal_geometry_censored
                                ),
                                bool(
                                    proposal.current_target
                                    .vertical_geometry_censored
                                ),
                            ),
                            passage_committed=bool(
                                proposal.mode
                                is VisualApproachMode.PASSAGE
                            ),
                            requested_target_pitch_rad=(
                                requested_pitch_before_top_fov
                            ),
                            fov_protected_target_pitch_rad=(
                                target_pitch_rad
                            ),
                            requested_thrust=command_thrust,
                        )
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course fresh horizontal-FOV closure brake "
                        f"refused: {exc}"
                    ) from exc
                if horizontal_fov_closure_brake is not None:
                    target_pitch_rad = (
                        horizontal_fov_closure_brake
                        .allocated_target_pitch_rad
                    )
                    if launch_evidence is not None:
                        launch_evidence["target_pitch_rad"] = (
                            target_pitch_rad
                        )
        if committed_crossing_authority is not None:
            # Observations, local-state propagation, and diagnostics continue
            # updating above, but the latch-sealed current-gate coast owns the
            # applied reference until authoritative race credit.  The hard
            # yaw soft-stop and final wire governor remain downstream.
            target_roll_rad = float(
                committed_crossing_authority.target_roll_rad
            )
            target_pitch_rad = float(
                committed_crossing_authority.target_pitch_rad
            )
            requested_yaw = float(
                committed_crossing_authority.yaw_rate_rad_s
            )
            command_thrust = float(
                committed_crossing_authority.requested_thrust
            )
            if (
                not all(
                    math.isfinite(value)
                    for value in (
                        target_roll_rad,
                        target_pitch_rad,
                        requested_yaw,
                        command_thrust,
                    )
                )
                or abs(target_roll_rad)
                > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
                or not MIN_VISUAL_TARGET_PITCH_RAD - 1e-12
                <= target_pitch_rad
                <= MAX_VISUAL_TARGET_PITCH_RAD + 1e-12
                or abs(requested_yaw)
                > MAX_VISUAL_YAW_RATE_RAD_S + 1e-12
                or not limits.min_thrust
                <= command_thrust
                <= limits.max_thrust
            ):
                raise abort_type(
                    "visual-course committed crossing reference escaped "
                    "its fixed envelope"
                )
            if requested_yaw != 0.0 and runtime.yaw_profile is None:
                raise abort_type(
                    "visual-course committed crossing yaw lacks calibrated "
                    "authority"
                )
        # Allocate the static attitude-loop response from the reference that
        # will actually be applied this tick.  Before the old inner governor
        # was removed, its governed output implicitly provided this behavior;
        # using the unallocated +0.12 destination here doubled early launch q.
        continuous_pitch_response_authority = (
            _pitch_response_authority(
                allocated_target_pitch_rad=target_pitch_rad,
                intercept_response_authority=(
                    intercept_response_authority
                ),
            )
        )
        if collective_evidence is not None:
            collective_evidence[
                "allocated_thrust_before_wire_governor"
            ] = command_thrust

        await host._wait_for_next_flight_command_slot()
        if refresh_ingress_after_slot:
            # The first post-credit iteration deliberately reuses its
            # already-admitted graph snapshot and therefore skipped the
            # outer sample.  Consume IMU/race/actuator ingress that arrived
            # during the slot wait before the pre-wire watchdog without
            # replacing the admitted graph.  A newer receiver camera
            # publication remains a no-wire supersession through the exact
            # token check below.
            host._sample_control_ingress()
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
        if (
            command_deadline_s is not None
            and float(runtime.monotonic())
            >= float(command_deadline_s)
        ):
            raise abort_type("visual-course command deadline expired")
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
            intercept_response_authority=(
                continuous_pitch_response_authority
            ),
        )
        base, roll_yaw_transport_rate = _allocate_roll_yaw_transport(
            base,
            target_roll_rad=target_roll_rad,
            target_pitch_rad=target_pitch_rad,
            bounded_yaw_rate_rad_s=bounded_yaw,
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
        dynamic_controller = runtime.dynamic_controller
        accepted_dynamic_evidence: Optional[Dict[str, Any]] = None
        if dynamic_controller is not None:
            governor_proposal_ns = runtime.perf_counter_ns()
            if (
                type(governor_proposal_ns) is not int
                or governor_proposal_ns < 0
            ):
                raise abort_type(
                    "visual-course dynamic governor clock is invalid"
                )
            launch_thrust_override = bool(
                launch_evidence is not None
                and launch_evidence["thrust_phase"] in {"preload", "boost"}
            )
            try:
                command = dynamic_controller.govern_wire_command(
                    command,
                    proposal_monotonic_ns=governor_proposal_ns,
                    launch_thrust_override=launch_thrust_override,
                    yaw_safety_override=yaw_soft_stop_zeroed,
                )
            except (TypeError, ValueError) as exc:
                raise abort_type(
                    f"visual-course dynamic wire governor refused: {exc}"
                ) from exc
            if type(command) is not AttitudeRateCommand:
                raise abort_type(
                    "visual-course dynamic wire governor returned an "
                    "invalid command"
                )
        runtime.validate_command(command)
        if (
            max(abs(command.roll_rate), abs(command.pitch_rate))
            > limits.max_command_rate_rad_s + 1e-12
            or abs(command.yaw_rate) > limits.max_yaw_rate_rad_s + 1e-12
            or not limits.min_thrust <= command.thrust <= limits.max_thrust
        ):
            raise abort_type("visual-course command escaped its fixed envelope")
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
        if command_deadline_s is not None:
            deadline_now_s = float(runtime.monotonic())
            deadline_ns = min(
                deadline_ns,
                _perf_counter_deadline_from_monotonic(
                    deadline_monotonic_s=float(command_deadline_s),
                    now_monotonic_s=deadline_now_s,
                    validation_perf_counter_ns=validation_ns,
                ),
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
        wire_return_monotonic_ns = visual_wire_authority.get(
            "transport_return_monotonic_ns"
        )
        top_level_wire_start_ns = receipt.get("call_start_monotonic_ns")
        wire_frame_token = visual_wire_authority.get("frame_token")
        if (
            visual_wire_authority.get("schema")
            != "aigp-vq2-visual-wire-authority/1"
            or type(wire_start_monotonic_ns) is not int
            or wire_start_monotonic_ns < 0
            or type(wire_return_monotonic_ns) is not int
            or wire_return_monotonic_ns < wire_start_monotonic_ns
            or top_level_wire_start_ns != wire_start_monotonic_ns
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
        if dynamic_controller is not None:
            try:
                dynamic_evidence = (
                    dynamic_controller.record_wire_acceptance(
                        target_roll_rad=target_roll_rad,
                        target_pitch_rad=target_pitch_rad,
                        yaw_rate_rad_s=float(command.yaw_rate),
                        thrust=float(command.thrust),
                        wire_command=command,
                        wire_start_monotonic_ns=(
                            wire_start_monotonic_ns
                        ),
                        requested_thrust=command_thrust,
                        thrust_slew_override=launch_thrust_override,
                        yaw_slew_override=yaw_soft_stop_zeroed,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise abort_type(
                    "visual-course dynamic controller could not commit "
                    f"the accepted wire command: {exc}"
                ) from exc
            if not isinstance(dynamic_evidence, Mapping):
                raise abort_type(
                    "visual-course dynamic command evidence is invalid"
                )
            accepted_dynamic_evidence = dict(dynamic_evidence)
            accepted_dynamic_evidence["roll_yaw_transport_rate_rad_s"] = (
                roll_yaw_transport_rate
            )
            if (
                type(dynamic_controller)
                is DynamicVisualCourseSession
                and top_fov_track_id is not None
            ):
                fov_summary = segment["top_fov_pitch_protection"]
                if top_fov_proposal is not None:
                    assert top_fov_observation is not None
                    if top_fov_proposal.limited:
                        fov_summary["limited_command_count"] += 1
                    fov_summary.update(
                        {
                            "last_track_id": top_fov_track_id,
                            "last_camera_token": asdict(
                                snapshot.latest_camera_token
                            ),
                            "last_wire_start_monotonic_ns": (
                                wire_start_monotonic_ns
                            ),
                            "last_raw_top_edge_image_down": (
                                top_fov_proposal.raw_top_edge_image_down
                            ),
                            "last_raw_top_edge_basis": (
                                top_fov_observation.raw_top_edge_basis
                            ),
                            "last_raw_top_edge_confidence": (
                                top_fov_observation
                                .raw_top_edge_confidence
                            ),
                            "last_raw_nominal_top_edge_image_down": (
                                top_fov_observation
                                .raw_nominal_top_edge_image_down
                            ),
                            "last_raw_top_edge_std_image_down": (
                                top_fov_observation
                                .raw_top_edge_std_image_down
                            ),
                            "last_forecast_top_edge_image_down": (
                                top_fov_proposal
                                .forecast_top_edge_image_down
                            ),
                            "last_raw_top_edge_nonrotational_angle_rate_rad_s": (
                                top_fov_proposal
                                .raw_top_edge_nonrotational_angle_rate_rad_s
                            ),
                            "last_protected_target_pitch_rad": (
                                top_fov_proposal.protected_target_pitch_rad
                            ),
                            "active": top_fov_proposal.active_after,
                        }
                    )
                    fov_summary["exact_raw_anchor"] = {
                        "basis": TOP_FOV_EXACT_RAW_ANCHOR_BASIS,
                        "gate_index": current_gate_index,
                        "track_id": top_fov_track_id,
                        "camera_token": asdict(
                            snapshot.latest_camera_token
                        ),
                        "observation_monotonic_ns": (
                            observation_monotonic_ns
                        ),
                        "wire_start_monotonic_ns": (
                            wire_start_monotonic_ns
                        ),
                        "capture_pitch_rad": (
                            top_fov_observation.capture_pitch_rad
                        ),
                        "raw_top_edge_image_down": (
                            top_fov_proposal.raw_top_edge_image_down
                        ),
                        "raw_nominal_top_edge_image_down": (
                            top_fov_observation
                            .raw_nominal_top_edge_image_down
                        ),
                        "raw_top_edge_std_image_down": (
                            top_fov_observation
                            .raw_top_edge_std_image_down
                        ),
                        "raw_top_edge_basis": (
                            top_fov_observation.raw_top_edge_basis
                        ),
                        "raw_top_edge_nonrotational_angle_rate_rad_s": (
                            top_fov_proposal
                            .raw_top_edge_nonrotational_angle_rate_rad_s
                        ),
                        "protected_target_pitch_rad": (
                            top_fov_proposal.protected_target_pitch_rad
                        ),
                        "active": top_fov_proposal.active_after,
                        "steering_only": True,
                        "passage_authority": False,
                        "advance_authority": False,
                    }
                    if (
                        top_fov_observation.raw_top_edge_basis
                        == TOP_FOV_INNER_EDGE_BASIS
                    ):
                        fov_summary.update(
                            {
                                "last_inner_track_id": (
                                    top_fov_track_id
                                ),
                                "last_inner_camera_token": asdict(
                                    snapshot.latest_camera_token
                                ),
                                "last_inner_wire_start_monotonic_ns": (
                                    wire_start_monotonic_ns
                                ),
                                "last_inner_raw_top_edge_basis": (
                                    TOP_FOV_INNER_EDGE_BASIS
                                ),
                                "last_inner_protected_target_pitch_rad": (
                                    top_fov_proposal
                                    .protected_target_pitch_rad
                                ),
                                "last_inner_active": (
                                    top_fov_proposal.active_after
                                ),
                            }
                        )
                    accepted_dynamic_evidence["top_fov_pitch_guidance"] = {
                        "basis": TOP_FOV_PITCH_PROTECTION_BASIS,
                        "track_id": top_fov_track_id,
                        "safe_top_edge_image_down": (
                            TOP_FOV_SAFE_EDGE_IMAGE_DOWN
                        ),
                        "raw_top_edge_basis": (
                            top_fov_observation.raw_top_edge_basis
                        ),
                        "raw_top_edge_confidence": (
                            top_fov_observation.raw_top_edge_confidence
                        ),
                        "raw_nominal_top_edge_image_down": (
                            top_fov_observation
                            .raw_nominal_top_edge_image_down
                        ),
                        "raw_top_edge_std_image_down": (
                            top_fov_observation
                            .raw_top_edge_std_image_down
                        ),
                        "raw_top_edge_motion_angle_rate_rad_s": (
                            top_fov_observation
                            .raw_top_edge_motion_angle_rate_rad_s
                        ),
                        **asdict(top_fov_proposal),
                    }
                elif (
                    top_fov_propagated_handoff is not None
                    and top_fov_propagated_observation is not None
                    and top_fov_propagated_proposal is not None
                ):
                    propagated_handoff = dict(
                        top_fov_propagated_handoff
                    )
                    if top_fov_propagated_proposal.limited:
                        fov_summary["limited_command_count"] += 1
                    fov_summary[
                        "propagated_state_handoff_command_count"
                    ] = int(
                        fov_summary[
                            "propagated_state_handoff_command_count"
                        ]
                    ) + 1
                    fov_summary[
                        "last_propagated_state_handoff"
                    ] = propagated_handoff
                    fov_summary.update(
                        {
                            "last_track_id": top_fov_track_id,
                            "last_camera_token": asdict(
                                snapshot.latest_camera_token
                            ),
                            "last_wire_start_monotonic_ns": (
                                wire_start_monotonic_ns
                            ),
                            "last_forecast_top_edge_image_down": (
                                top_fov_propagated_proposal
                                .forecast_top_edge_image_down
                            ),
                            "last_protected_target_pitch_rad": (
                                top_fov_propagated_proposal
                                .protected_target_pitch_rad
                            ),
                            "active": (
                                top_fov_propagated_proposal.active_after
                            ),
                        }
                    )
                    accepted_dynamic_evidence[
                        "top_fov_propagated_state_handoff"
                    ] = propagated_handoff
                    accepted_dynamic_evidence["top_fov_pitch_guidance"] = {
                        "basis": TOP_FOV_PITCH_PROTECTION_BASIS,
                        "track_id": top_fov_track_id,
                        "safe_top_edge_image_down": (
                            TOP_FOV_SAFE_EDGE_IMAGE_DOWN
                        ),
                        "geometry_basis": (
                            top_fov_propagated_observation.geometry_basis
                        ),
                        "projected_nominal_top_edge_image_down": (
                            top_fov_propagated_observation
                            .projected_nominal_top_edge_image_down
                        ),
                        "projected_top_edge_std_image_down": (
                            top_fov_propagated_observation
                            .projected_top_edge_std_image_down
                        ),
                        "prediction_horizon_remaining_s": (
                            top_fov_propagated_observation
                            .prediction_horizon_remaining_s
                        ),
                        **asdict(top_fov_propagated_proposal),
                    }
                    host.recorder.emit(
                        "visual_course_dynamic_fov_gap_handoff",
                        gate_index=current_gate_index,
                        stage=stage,
                        camera_token=asdict(
                            snapshot.latest_camera_token
                        ),
                        authority=propagated_handoff,
                        pitch_guidance=accepted_dynamic_evidence[
                            "top_fov_pitch_guidance"
                        ],
                        command=asdict(command),
                    )
                elif (
                    top_fov_retained_raw_handoff is not None
                    and top_fov_retained_raw_observation is not None
                    and top_fov_retained_raw_proposal is not None
                ):
                    retained_handoff = dict(
                        top_fov_retained_raw_handoff
                    )
                    if top_fov_retained_raw_proposal.limited:
                        fov_summary["limited_command_count"] += 1
                    fov_summary[
                        "retained_raw_state_handoff_command_count"
                    ] = int(
                        fov_summary[
                            "retained_raw_state_handoff_command_count"
                        ]
                    ) + 1
                    fov_summary[
                        "last_retained_raw_state_handoff"
                    ] = retained_handoff
                    fov_summary.update(
                        {
                            "last_track_id": top_fov_track_id,
                            "last_camera_token": asdict(
                                snapshot.latest_camera_token
                            ),
                            "last_wire_start_monotonic_ns": (
                                wire_start_monotonic_ns
                            ),
                            "last_forecast_top_edge_image_down": (
                                top_fov_retained_raw_proposal
                                .forecast_top_edge_image_down
                            ),
                            "last_protected_target_pitch_rad": (
                                top_fov_retained_raw_proposal
                                .protected_target_pitch_rad
                            ),
                            "active": (
                                top_fov_retained_raw_proposal.active_after
                            ),
                        }
                    )
                    accepted_dynamic_evidence[
                        "top_fov_retained_raw_state_handoff"
                    ] = retained_handoff
                    accepted_dynamic_evidence["top_fov_pitch_guidance"] = {
                        "basis": TOP_FOV_PITCH_PROTECTION_BASIS,
                        "track_id": top_fov_track_id,
                        "safe_top_edge_image_down": (
                            TOP_FOV_SAFE_EDGE_IMAGE_DOWN
                        ),
                        **asdict(top_fov_retained_raw_observation),
                        **asdict(top_fov_retained_raw_proposal),
                    }
                    host.recorder.emit(
                        "visual_course_dynamic_fov_gap_handoff",
                        gate_index=current_gate_index,
                        stage=stage,
                        camera_token=asdict(
                            snapshot.latest_camera_token
                        ),
                        authority=retained_handoff,
                        pitch_guidance=accepted_dynamic_evidence[
                            "top_fov_pitch_guidance"
                        ],
                        command=asdict(command),
                    )
                if top_censored_pitch_arbitration is not None:
                    accepted_dynamic_evidence[
                        "top_censored_pitch_arbitration"
                    ] = dict(top_censored_pitch_arbitration)
                    host.recorder.emit(
                        "visual_course_top_censored_pitch_arbitration",
                        gate_index=current_gate_index,
                        stage=stage,
                        camera_token=asdict(
                            snapshot.latest_camera_token
                        ),
                        arbitration=dict(
                            top_censored_pitch_arbitration
                        ),
                        command=asdict(command),
                    )
                if top_censored_closure_recovery is not None:
                    recovery_evidence = asdict(
                        top_censored_closure_recovery
                    )
                    accepted_dynamic_evidence[
                        "fresh_top_censored_closure_recovery"
                    ] = recovery_evidence
                    pitch_guidance = accepted_dynamic_evidence.get(
                        "top_fov_pitch_guidance"
                    )
                    if isinstance(pitch_guidance, dict):
                        pitch_guidance[
                            "superseded_by_fresh_boundary_recovery"
                        ] = True
                        pitch_guidance[
                            "applied_target_pitch_rad"
                        ] = target_pitch_rad
                    fov_summary[
                        "last_protected_target_pitch_rad"
                    ] = target_pitch_rad
                    host.recorder.emit(
                        "visual_course_fresh_top_censored_recovery",
                        gate_index=current_gate_index,
                        stage=stage,
                        camera_token=asdict(
                            snapshot.latest_camera_token
                        ),
                        allocation=recovery_evidence,
                        command=asdict(command),
                    )
                if horizontal_fov_closure_brake is not None:
                    brake_evidence = asdict(
                        horizontal_fov_closure_brake
                    )
                    accepted_dynamic_evidence[
                        "fresh_horizontal_fov_closure_brake"
                    ] = brake_evidence
                    pitch_guidance = accepted_dynamic_evidence.get(
                        "top_fov_pitch_guidance"
                    )
                    if isinstance(pitch_guidance, dict):
                        pitch_guidance[
                            "superseded_by_horizontal_fov_closure_brake"
                        ] = True
                        pitch_guidance[
                            "applied_target_pitch_rad"
                        ] = target_pitch_rad
                    fov_summary[
                        "last_protected_target_pitch_rad"
                    ] = target_pitch_rad
                    segment[
                        "horizontal_fov_closure_brake_command_count"
                    ] = (
                        int(
                            segment[
                                "horizontal_fov_closure_brake_command_count"
                            ]
                        )
                        + 1
                    )
                    segment[
                        "last_horizontal_fov_closure_brake"
                    ] = brake_evidence
                    host.recorder.emit(
                        "visual_course_fresh_horizontal_fov_closure_brake",
                        gate_index=current_gate_index,
                        stage=stage,
                        camera_token=asdict(
                            snapshot.latest_camera_token
                        ),
                        allocation=brake_evidence,
                        command=asdict(command),
                    )
            host.recorder.emit(
                "visual_course_dynamic_command",
                **accepted_dynamic_evidence,
            )
            segment["dynamic_controller"] = dict(
                dynamic_controller.evidence_summary()
            )
        if collective_evidence is not None:
            collective_evidence["wire_thrust"] = float(command.thrust)
            host.recorder.emit(
                "visual_course_current_aperture_collective",
                **collective_evidence,
            )
            collective_summary = segment[
                "current_aperture_collective"
            ]
            collective_summary["command_count"] = (
                int(collective_summary["command_count"]) + 1
            )
            if collective_evidence[
                "held_last_observable_collective"
            ]:
                collective_summary["held_command_count"] = (
                    int(collective_summary["held_command_count"]) + 1
                )
            elif not collective_evidence["vertical_censored"]:
                collective_summary["observable_command_count"] = (
                    int(
                        collective_summary[
                            "observable_command_count"
                        ]
                    )
                    + 1
                )
            for summary_name, evidence_name in (
                (
                    "last_current_vertical_error_image_down",
                    "current_vertical_error_image_down",
                ),
                (
                    "last_current_vertical_rate_down_s",
                    "current_vertical_rate_down_s",
                ),
                (
                    "last_filtered_vertical_rate_down_s",
                    "proved_filtered_vertical_rate_down_s",
                ),
                (
                    "last_control_vertical_error_image_down",
                    "control_vertical_error_image_down",
                ),
                (
                    "last_control_vertical_rate_down_s",
                    "control_vertical_rate_down_s",
                ),
                ("last_control_basis", "basis"),
                ("last_requested_thrust", "requested_thrust"),
                (
                    "last_allocated_thrust_before_wire_governor",
                    "allocated_thrust_before_wire_governor",
                ),
                ("last_wire_thrust", "wire_thrust"),
            ):
                collective_summary[summary_name] = (
                    collective_evidence[evidence_name]
                )
        host._record_tick(
            stage,
            float(runtime.monotonic()) - segment_started_s,
            command,
        )
        total_navigation_commands += 1
        last_command_send_s = float(runtime.monotonic())
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
            publication_monotonic_ns=publication_monotonic_ns,
            wire_start_monotonic_ns=wire_start_monotonic_ns,
            wire_return_monotonic_ns=wire_return_monotonic_ns,
            wire_camera_token=snapshot.latest_camera_token,
            wire_race_gate_index=current_gate_index,
            publication_pinned_through_transport_return=True,
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
            dynamic_evidence=accepted_dynamic_evidence,
        )

    async def send_censored_passage_coast(
        *,
        snapshot: Any,
        authority: _CensoredPassageCoastAuthority,
        yaw_reference_rad: float,
        segment_started_s: float,
        stage: str,
        command_deadline_s: float,
        count_as_navigation: bool = True,
        hold_basis: str = CENSORED_PASSAGE_COAST_BASIS,
    ) -> Optional[AttitudeRateCommand]:
        """Send a bounded attitude/thrust hold on one exact fresh frame."""

        nonlocal total_navigation_commands
        nonlocal last_command_send_s
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
            if not math.isfinite(now_s) or now_s < last_command_send_s:
                raise abort_type(
                    "visual-course supersession clock regressed"
                ) from exc
            consecutive_superseded_proposals += 1
            segment["superseded_proposal_count"] = int(
                segment["superseded_proposal_count"]
            ) + 1
            hold_s = now_s - last_command_send_s
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
            return None

        values = (
            authority.target_roll_rad,
            authority.target_pitch_rad,
            authority.yaw_rate_rad_s,
            authority.requested_thrust,
            command_deadline_s,
        )
        if (
            hold_basis
            not in {
                CENSORED_PASSAGE_COAST_BASIS,
                APPROACH_TOP_RECOVERY_BASIS,
                APPROACH_INNER_DROPOUT_HOLD_BASIS,
                APPROACH_PROPAGATED_VISIBILITY_GAP_BASIS,
                APPROACH_CURRENT_AMBIGUITY_QUARANTINE_BASIS,
            }
            or authority.gate_index != current_gate_index
            or authority.track_id != current_track_id
            or not all(math.isfinite(value) for value in values)
            or abs(authority.target_roll_rad)
            > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
            or authority.target_pitch_rad
            < MIN_VISUAL_TARGET_PITCH_RAD - 1e-12
            or authority.target_pitch_rad
            > MAX_VISUAL_TARGET_PITCH_RAD + 1e-12
            or abs(authority.yaw_rate_rad_s)
            > limits.max_yaw_rate_rad_s + 1e-12
            or not limits.min_thrust
            <= authority.requested_thrust
            <= limits.max_thrust
            or (
                authority.yaw_rate_rad_s != 0.0
                and runtime.yaw_profile is None
            )
        ):
            raise abort_type(
                "visual-course censored passage coast authority is invalid"
            )

        await host._wait_for_next_flight_command_slot()
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
        coast_deadline_s = float(command_deadline_s)
        excursion, _rates, euler_yaw_rate = _assert_course_attitude_state(
            host,
            yaw_reference_rad=yaw_reference_rad,
            limits=limits,
            yaw_profile=runtime.yaw_profile,
            abort_type=abort_type,
            phase=f"{stage} pre-send",
        )
        bounded_yaw = authority.yaw_rate_rad_s
        if bounded_yaw != 0.0:
            assert runtime.yaw_profile is not None
            bounded_yaw = _limit_calibrated_yaw_request(
                bounded_yaw,
                excursion_rad=excursion,
                measured_euler_yaw_rate_rad_s=euler_yaw_rate,
                limits=limits,
                profile=runtime.yaw_profile,
                abort_type=abort_type,
            )
        if float(runtime.monotonic()) >= coast_deadline_s:
            raise abort_type(
                "visual-course censored passage coast expired"
            )
        base = runtime.attitude_rate_command(
            host.estimate,
            target_roll_rad=authority.target_roll_rad,
            target_pitch_rad=authority.target_pitch_rad,
            thrust=authority.requested_thrust,
        )
        base, roll_yaw_transport_rate = _allocate_roll_yaw_transport(
            base,
            target_roll_rad=authority.target_roll_rad,
            target_pitch_rad=authority.target_pitch_rad,
            bounded_yaw_rate_rad_s=bounded_yaw,
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
        dynamic_controller = runtime.dynamic_controller
        if dynamic_controller is not None:
            governor_proposal_ns = runtime.perf_counter_ns()
            if (
                type(governor_proposal_ns) is not int
                or governor_proposal_ns < 0
            ):
                raise abort_type(
                    "visual-course dynamic governor clock is invalid"
                )
            yaw_safety_override = bool(
                authority.yaw_rate_rad_s != 0.0
                and bounded_yaw == 0.0
            )
            try:
                command = dynamic_controller.govern_wire_command(
                    command,
                    proposal_monotonic_ns=governor_proposal_ns,
                    # Continue the ordinary bounded thrust ramp toward the
                    # last clean current-aperture request.  Censorship removes
                    # new geometry, not the retained collective objective.
                    launch_thrust_override=False,
                    yaw_safety_override=yaw_safety_override,
                )
            except (TypeError, ValueError) as exc:
                raise abort_type(
                    "visual-course dynamic coast governor refused: "
                    f"{exc}"
                ) from exc
        runtime.validate_command(command)
        if (
            max(abs(command.roll_rate), abs(command.pitch_rate))
            > limits.max_command_rate_rad_s + 1e-12
            or abs(command.yaw_rate)
            > limits.max_yaw_rate_rad_s + 1e-12
            or not limits.min_thrust <= command.thrust <= limits.max_thrust
        ):
            raise abort_type(
                "visual-course censored passage command escaped its fixed "
                "envelope"
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
        coast_deadline_ns = _perf_counter_deadline_from_monotonic(
            deadline_monotonic_s=coast_deadline_s,
            now_monotonic_s=float(runtime.monotonic()),
            validation_perf_counter_ns=validation_ns,
        )
        deadline_ns = min(deadline_ns, coast_deadline_ns)
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
        if dynamic_controller is not None:
            try:
                dynamic_evidence = (
                    dynamic_controller.record_wire_acceptance(
                        target_roll_rad=authority.target_roll_rad,
                        target_pitch_rad=authority.target_pitch_rad,
                        yaw_rate_rad_s=float(command.yaw_rate),
                        thrust=float(command.thrust),
                        wire_command=command,
                        wire_start_monotonic_ns=(
                            wire_start_monotonic_ns
                        ),
                        thrust_slew_override=False,
                        yaw_slew_override=yaw_safety_override,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise abort_type(
                    "visual-course dynamic controller could not commit "
                    f"the coast command: {exc}"
                ) from exc
            if not isinstance(dynamic_evidence, Mapping):
                raise abort_type(
                    "visual-course dynamic coast evidence is invalid"
                )
            accepted_dynamic_evidence = dict(dynamic_evidence)
            accepted_dynamic_evidence["roll_yaw_transport_rate_rad_s"] = (
                roll_yaw_transport_rate
            )
            host.recorder.emit(
                "visual_course_dynamic_command",
                **accepted_dynamic_evidence,
            )
            segment["dynamic_controller"] = dict(
                dynamic_controller.evidence_summary()
            )

        host._record_tick(
            stage,
            float(runtime.monotonic()) - segment_started_s,
            command,
        )
        if count_as_navigation:
            total_navigation_commands += 1
        last_command_send_s = float(runtime.monotonic())
        consecutive_superseded_proposals = 0
        if count_as_navigation and segment["launch_bootstrap"]["enabled"]:
            launch = segment["launch_bootstrap"]
            launch["command_count"] = int(launch["command_count"]) + 1
            launch["last_elapsed_s"] = (
                float(runtime.monotonic()) - course_started_s
            )
            launch["last_target_pitch_rad"] = authority.target_pitch_rad
            launch["last_thrust"] = command.thrust
            launch["last_thrust_phase"] = hold_basis
        if (
            count_as_navigation
            and
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
        command_event = (
            "visual_course_approach_top_recovery_command"
            if hold_basis == APPROACH_TOP_RECOVERY_BASIS
            else (
                "visual_course_approach_inner_dropout_hold_command"
                if hold_basis
                == APPROACH_INNER_DROPOUT_HOLD_BASIS
                else (
                    "visual_course_approach_propagated_visibility_gap_command"
                    if hold_basis
                    == APPROACH_PROPAGATED_VISIBILITY_GAP_BASIS
                    else (
                        "visual_course_approach_current_ambiguity_"
                        "quarantine_command"
                        if hold_basis
                        == APPROACH_CURRENT_AMBIGUITY_QUARANTINE_BASIS
                        else "visual_course_censored_passage_coast_command"
                    )
                )
            )
        )
        host.recorder.emit(
            command_event,
            gate_index=current_gate_index,
            stage=stage,
            basis=hold_basis,
            camera_token=asdict(snapshot.latest_camera_token),
            anchor_camera_token=asdict(authority.anchor_camera_token),
            target_roll_rad=authority.target_roll_rad,
            target_pitch_rad=authority.target_pitch_rad,
            requested_yaw_rate_rad_s=authority.yaw_rate_rad_s,
            requested_thrust=authority.requested_thrust,
            wire_thrust=command.thrust,
            counted_as_navigation=count_as_navigation,
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
        _roll, _pitch, observed_yaw_reference_rad, _rates = _attitude_state(
            host,
            abort_type,
        )
        launch_enabled = bool(
            segment_number == 0
            and initial_gate_index == 0
            and current_gate_index == initial_gate_index
        )
        post_credit_recovery = pending_post_credit_recovery
        pending_post_credit_recovery = None
        carried_post_credit_planner = pending_post_credit_planner
        pending_post_credit_planner = None
        carried_post_credit_yaw_reference_rad = (
            pending_post_credit_yaw_reference_rad
        )
        pending_post_credit_yaw_reference_rad = None
        if post_credit_recovery is not None and (
            segment_number <= 0
            or post_credit_recovery.to_gate_index
            != current_gate_index
            or post_credit_recovery.track_id != current_track_id
            or post_credit_recovery.from_gate_index
            != current_gate_index - 1
            or not _token_strictly_newer(
                post_credit_recovery.admitted_camera_token,
                post_credit_recovery.camera_token_at_credit,
            )
            or not math.isfinite(post_credit_recovery.deadline_s)
            or post_credit_recovery.deadline_s <= segment_started_s
            or type(post_credit_recovery.successor_steering_available)
            is not bool
        ):
            raise abort_type(
                "visual-course pending post-credit recovery is invalid "
                "or expired"
            )
        if (
            (carried_post_credit_planner is None)
            != (carried_post_credit_yaw_reference_rad is None)
            or (
                carried_post_credit_planner is not None
                and (
                    post_credit_recovery is None
                    or not math.isfinite(
                        carried_post_credit_yaw_reference_rad
                    )
                )
            )
        ):
            raise abort_type(
                "visual-course carried planner/yaw authority is invalid"
            )
        yaw_reference_rad = (
            observed_yaw_reference_rad
            if carried_post_credit_yaw_reference_rad is None
            else carried_post_credit_yaw_reference_rad
        )
        current_aperture_collective_state = (
            _CurrentApertureProvedCollectiveState(
                track_id=current_track_id
            )
            if launch_enabled or runtime.dynamic_controller is not None
            else None
        )

        def retained_current_aperture_collective(
            fallback_wire_thrust: float,
            *,
            subsupport_collective_authorized: bool,
        ) -> float:
            """Return the last clean aperture request for bounded coast."""

            fallback = float(fallback_wire_thrust)
            if type(subsupport_collective_authorized) is not bool:
                raise abort_type(
                    "visual-course retained collective authority is invalid"
                )
            if runtime.dynamic_controller is None:
                return fallback
            state = current_aperture_collective_state
            if state is None or state.last_observable_thrust is None:
                raise abort_type(
                    "visual-course dynamic crossing lacks a retained "
                    "current-aperture collective"
                )
            requested = float(state.last_observable_thrust)
            if (
                not math.isfinite(requested)
                or requested < limits.min_thrust - 1e-12
                or requested > limits.max_thrust + 1e-12
            ):
                raise abort_type(
                    "visual-course retained current-aperture collective "
                    "escaped its fixed envelope"
                )
            if not subsupport_collective_authorized:
                requested = max(
                    GATE0_PROVED_COLLECTIVE_BASE,
                    requested,
                )
            return requested

        def make_planner(
            *,
            track_id: str,
            gate_index: int,
            next_gate_blend: float,
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
            return runtime.servo_factory(
                track_id,
                gate_index,
                host.visual_config.servo,
                **kwargs,
            )

        planner = (
            carried_post_credit_planner
            if carried_post_credit_planner is not None
            else make_planner(
                track_id=current_track_id,
                gate_index=current_gate_index,
                next_gate_blend=(
                    host.visual_config.lifecycle.next_gate_blend_max
                ),
            )
        )
        mode = (
            VisualApproachMode.PROMOTE_REACQUIRE
            if post_credit_recovery is not None
            else VisualApproachMode.APPROACH
        )
        lifecycle = (
            CourseLifecycle.PROMOTE_REACQUIRE
            if post_credit_recovery is not None
            else CourseLifecycle.APPROACH
        )
        recovery_deadline_s = (
            None
            if post_credit_recovery is None
            else post_credit_recovery.deadline_s
        )
        recovery_previous_camera_token = (
            None
            if post_credit_recovery is None
            else post_credit_recovery.camera_token_at_credit
        )
        recovery_last_track_token = (
            None
            if post_credit_recovery is None
            else post_credit_recovery.admitted_camera_token
        )
        recovery_reuse_graph_snapshot = (
            post_credit_recovery is not None
        )
        recovery_refresh_receiver_snapshot = False
        post_credit_successor_handoff_required = bool(
            post_credit_recovery is not None
            and post_credit_recovery.successor_steering_available
        )
        recovery_first_clean_wire_token: Optional[
            CameraFrameToken
        ] = None
        passage_admission: Optional[VisualApproachPassageAdmission] = None
        passage_started_s: Optional[float] = None
        passage_command_count = 0
        passage_next_preview_command_count = 0
        last_passage_successor_horizontal: Optional[float] = None
        last_passage_successor_horizontal_rate: Optional[float] = None
        advance_command_count = 0
        approach_command_count = 0
        crossing_anchor: Optional[Dict[str, Any]] = None
        near_plane_evidence = NearPlaneEvidence()
        near_plane_latch: Optional[NearPlaneLatch] = None
        crossing_coast_authority: Optional[
            _CensoredPassageCoastAuthority
        ] = None
        crossing_reviewed_track_id: Optional[str] = None
        crossing_successor_identity_sealed = False
        crossing_commitment_deadline_s: Optional[float] = None
        crossing_predicted_contact_deadline_s: Optional[float] = None
        last_clean_passage_token: Optional[CameraFrameToken] = None
        censored_passage_coast_started_s: Optional[float] = None
        censored_passage_coast_last_observed_token: Optional[
            CameraFrameToken
        ] = None
        censored_passage_coast_fresh_frame_count = 0
        censored_passage_coast_command_count = 0
        approach_top_recovery_authority: Optional[
            _ApproachTopRecoveryAuthority
        ] = None
        approach_top_recovery_started_s: Optional[float] = None
        approach_top_recovery_last_token: Optional[
            CameraFrameToken
        ] = None
        approach_top_recovery_fresh_frame_count = 0
        approach_top_recovery_command_count = 0
        approach_inner_dropout_authority: Optional[
            _ApproachInnerDropoutAuthority
        ] = None
        approach_inner_dropout_hold_command_count = 0
        approach_propagated_visibility_gap_started_s: Optional[float] = None
        approach_propagated_visibility_gap_fresh_frame_count = 0
        approach_propagated_visibility_gap_command_count = 0
        approach_current_ambiguity_quarantine: Optional[
            _ApproachCurrentAmbiguityQuarantineAuthority
        ] = None
        approach_current_ambiguity_quarantine_command_count = 0
        crossing_wait_coast_command_count = 0
        crossing_wait_adjacent_command_count = 0
        credit_wait_adjacent_planner: Optional[Any] = None
        credit_wait_adjacent_track_id: Optional[str] = None
        credit_wait_reviewed_track_id: Optional[str] = None
        crossing_started_s: Optional[float] = None
        crossing_baseline_race: Optional[AuthoritativeRaceStatusRef] = None
        last_planned_token: Optional[CameraFrameToken] = None
        last_command_send_s = segment_started_s
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
            "yaw_soft_stop_zero_command_count": 0,
            "passage_admission_yaw_soft_stop_withheld_count": 0,
            "committed_successor_steering_refresh_count": 0,
            "last_committed_successor_steering_refresh": None,
            "crossing_wait_zero_command_count": 0,
            "crossing_wait_coast_command_count": 0,
            "crossing_wait_adjacent_command_count": 0,
            "crossing_wait_adjacent_track_id": None,
            "censored_passage_coast_fresh_frame_count": 0,
            "censored_passage_coast_command_count": 0,
            "censored_passage_coast": None,
            "approach_top_recovery_fresh_frame_count": 0,
            "approach_top_recovery_command_count": 0,
            "approach_top_recovery": None,
            "approach_inner_dropout_hold_command_count": 0,
            "approach_inner_dropout_hold": None,
            "approach_propagated_visibility_gap_fresh_frame_count": 0,
            "approach_propagated_visibility_gap_command_count": 0,
            "approach_propagated_visibility_gap": None,
            "approach_current_ambiguity_quarantine_command_count": 0,
            "approach_current_ambiguity_quarantine": None,
            "horizontal_fov_closure_brake_command_count": 0,
            "last_horizontal_fov_closure_brake": None,
            "authoritative_credit_reconciliation": None,
            "post_credit_zero_command_count": 0,
            "post_credit_hold_command_count": 0,
            "post_credit_successor_steering_command_count": 0,
            "recovery_navigation_command_count": 0,
            "recovery_clean_command_count": 0,
            "recovery_one_edge_command_count": 0,
            "recovery_propagated_state_command_count": 0,
            "recovery_zero_command_count": 0,
            "recovery_support_command_count": 0,
            "post_credit_successor_handoff_required": (
                post_credit_successor_handoff_required
            ),
            "post_credit_successor_handoff_retirement": None,
            "passage_authority_enabled": False,
            "passage_admission": None,
            "passage_command_seal": None,
            "lifecycle": lifecycle.value,
            "near_plane_evidence_frame_count": 0,
            "near_plane_latch": None,
            "near_plane_reviewed_track_id": None,
            "near_plane_successor_identity_sealed": False,
            "near_plane_measurement_mode": None,
            "crossing_anchor": None,
            "outcome": "running",
            "top_fov_pitch_protection": {
                "enabled": (
                    type(runtime.dynamic_controller)
                    is DynamicVisualCourseSession
                ),
                "basis": (
                    TOP_FOV_PITCH_PROTECTION_BASIS
                    if (
                        type(runtime.dynamic_controller)
                        is DynamicVisualCourseSession
                    )
                    else None
                ),
                "safe_top_edge_image_down": (
                    TOP_FOV_SAFE_EDGE_IMAGE_DOWN
                    if (
                        type(runtime.dynamic_controller)
                        is DynamicVisualCourseSession
                    )
                    else None
                ),
                "limited_command_count": 0,
                "propagated_state_handoff_command_count": 0,
                "last_propagated_state_handoff": None,
                "retained_raw_state_handoff_command_count": 0,
                "last_retained_raw_state_handoff": None,
                "retained_exact_top_closure_recovery_command_count": 0,
                "last_retained_exact_top_closure_recovery": None,
                "last_exact_top_closure_recovery": None,
                "exact_raw_anchor": None,
                "last_track_id": None,
                "last_camera_token": None,
                "last_wire_start_monotonic_ns": None,
                "last_inner_track_id": None,
                "last_inner_camera_token": None,
                "last_inner_wire_start_monotonic_ns": None,
                "last_inner_raw_top_edge_basis": None,
                "last_inner_protected_target_pitch_rad": None,
                "last_inner_active": False,
                "last_raw_top_edge_image_down": None,
                "last_raw_top_edge_basis": None,
                "last_raw_top_edge_confidence": None,
                "last_raw_nominal_top_edge_image_down": None,
                "last_raw_top_edge_std_image_down": None,
                "last_forecast_top_edge_image_down": None,
                "last_raw_top_edge_nonrotational_angle_rate_rad_s": None,
                "last_protected_target_pitch_rad": None,
                "active": False,
            },
            "current_aperture_collective": {
                "enabled": runtime.dynamic_controller is not None,
                "basis": (
                    CURRENT_APERTURE_PROVED_COLLECTIVE_BASIS
                    if runtime.dynamic_controller is not None
                    else None
                ),
                "base": (
                    GATE0_PROVED_COLLECTIVE_BASE
                    if runtime.dynamic_controller is not None
                    else None
                ),
                "error_gain": (
                    GATE0_PROVED_COLLECTIVE_ERROR_GAIN
                    if runtime.dynamic_controller is not None
                    else None
                ),
                "rate_gain": (
                    GATE0_PROVED_COLLECTIVE_RATE_GAIN
                    if runtime.dynamic_controller is not None
                    else None
                ),
                "rate_filter_alpha": (
                    GATE0_PROVED_COLLECTIVE_RATE_FILTER_ALPHA
                    if runtime.dynamic_controller is not None
                    else None
                ),
                "command_count": 0,
                "observable_command_count": 0,
                "held_command_count": 0,
                "last_current_vertical_error_image_down": None,
                "last_current_vertical_rate_down_s": None,
                "last_filtered_vertical_rate_down_s": None,
                "last_control_vertical_error_image_down": None,
                "last_control_vertical_rate_down_s": None,
                "last_control_basis": None,
                "last_requested_thrust": None,
                "last_allocated_thrust_before_wire_governor": None,
                "last_wire_thrust": None,
            },
            "launch_bootstrap": {
                "enabled": launch_enabled,
                "preload_duration_s": INITIAL_PAD_PRELOAD_DURATION_S,
                "preload_thrust": INITIAL_PAD_PRELOAD_THRUST,
                "post_boost_collective_basis": (
                    "generic-visual-servo"
                    if launch_enabled
                    else None
                ),
                "post_boost_collective_base": None,
                "post_boost_collective_error_gain": None,
                "post_boost_collective_rate_gain": None,
                "post_boost_collective_max_abs_error": None,
                "post_boost_collective_max_abs_rate": None,
                "post_boost_collective_rate_filter_alpha": None,
                "post_boost_next_preview_collective_basis": None,
                "post_boost_next_preview_collective_error_gain": None,
                "post_boost_next_preview_collective_max_thrust_delta": None,
                "boost_duration_s": float(
                    host.visual_config.lifecycle.launch_boost_duration_s
                ),
                "boost_thrust": float(
                    host.visual_config.lifecycle.launch_boost_thrust
                ),
                "pitch_blend_s": float(
                    host.visual_config.lifecycle.launch_pitch_blend_s
                ),
                "pitch_reference_basis": (
                    LAUNCH_PITCH_REFERENCE_BASIS
                ),
                "pitch_reference_max_rate_rad_s": (
                    LAUNCH_PITCH_REFERENCE_MAX_RATE_RAD_S
                ),
                "pitch_reference_accel_rad_s2": (
                    LAUNCH_PITCH_REFERENCE_ACCEL_RAD_S2
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

        def finish_from_authoritative_status(
            race_status: AuthoritativeRaceStatusRef,
            *,
            phase: str,
        ) -> Dict[str, Any]:
            nonlocal last_race
            nonlocal latest_authoritative_gate_index
            nonlocal max_gate_index

            if not race_status.race_finished:
                raise abort_type(
                    "visual-course terminal helper lacks race_finished"
                )
            _assert_course_attitude_state(
                host,
                yaw_reference_rad=yaw_reference_rad,
                limits=limits,
                yaw_profile=runtime.yaw_profile,
                abort_type=abort_type,
                phase=phase,
            )
            finish_token = host._visual_camera_token_at_race_credit(
                race_status
            )
            try:
                finished_snapshot = (
                    host.visual_gate_graph.confirm_race_finished(
                        host.visual_tracker,
                        race_status=race_status,
                        camera_token_at_finish=finish_token,
                    )
                )
            except GateGraphError as exc:
                raise abort_type(
                    f"visual-course race-finish proof refused: {exc}"
                ) from exc
            if not getattr(finished_snapshot, "race_finished", False):
                raise abort_type("visual-course race finish did not latch")
            last_race = race_status
            latest_authoritative_gate_index = int(
                race_status.active_gate_index
            )
            max_gate_index = max(
                max_gate_index,
                latest_authoritative_gate_index,
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
                    "final_gate_index": (
                        latest_authoritative_gate_index
                    ),
                }
            )
            host.recorder.emit("visual_course_complete", **summary)
            host._visual_course_summary = dict(summary)
            return summary

        credited_race: Optional[AuthoritativeRaceStatusRef] = None

        def accept_no_wire_race_boundary(
            exc: RaceActiveBoundaryChangedBeforeWire,
            *,
            allow_unlatched_graph_reconciliation: bool = False,
        ) -> AuthoritativeRaceStatusRef:
            """Consume only proved credit after a navigation send was refused."""

            nonlocal crossing_started_s
            nonlocal credit_wait_reviewed_track_id
            nonlocal last_race

            refused_race = host._visual_race_status_ref()
            relation = _race_relation(
                refused_race,
                last_race,
                abort_type,
            )
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
            if crossing_anchor is None and not refused_race.race_finished:
                reconciliation = (
                    _unlatched_atomic_credit_successor_evidence(
                        snapshot,
                        current_gate_index=current_gate_index,
                        current_track_id=current_track_id,
                    )
                    if allow_unlatched_graph_reconciliation
                    else None
                )
                if reconciliation is None:
                    raise abort_type(
                        "visual-course race boundary changed before wire "
                        "without previously latched passage evidence"
                    ) from exc
                credit_wait_reviewed_track_id = str(
                    reconciliation["reviewed_track_id"]
                )
                segment["authoritative_credit_reconciliation"] = {
                    **dict(reconciliation),
                    "race_status_sequence": (
                        refused_race.race_status_sequence
                    ),
                    "race_received_monotonic_ns": (
                        refused_race.received_monotonic_ns
                    ),
                }
            last_race = refused_race
            crossing_started_s = (
                crossing_started_s or float(runtime.monotonic())
            )
            return refused_race

        while credited_race is None:
            reuse_recovery_graph = bool(recovery_reuse_graph_snapshot)
            refresh_recovery_receiver = bool(
                recovery_refresh_receiver_snapshot
            )
            if reuse_recovery_graph or refresh_recovery_receiver:
                recovery_reuse_graph_snapshot = False
                recovery_refresh_receiver_snapshot = False
                now = float(runtime.monotonic())
                if not math.isfinite(now):
                    raise abort_type(
                        "visual-course recovery clock is invalid"
                    )
            else:
                now = await pace_tick()
            if now >= course_deadline_s:
                raise abort_type("visual-course hard duration expired")
            if now >= segment_deadline_s:
                raise abort_type(
                    f"visual-course gate {current_gate_index} segment expired"
                )
            if (
                lifecycle is CourseLifecycle.PROMOTE_REACQUIRE
                and (
                    recovery_deadline_s is None
                    or now >= recovery_deadline_s
                )
            ):
                raise abort_type(
                    f"visual-course gate {current_gate_index} "
                    "post-credit recovery timed out"
                )
            if (
                passage_started_s is not None
                and crossing_started_s is None
                and now - passage_started_s > limits.passage_hard_duration_s
            ):
                raise abort_type(
                    f"visual-course gate {current_gate_index} passage expired"
                )

            if not reuse_recovery_graph:
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
            elif race.race_finished:
                if relation <= 0:
                    raise abort_type(
                        "visual-course transition lacks newer race ingress"
                    )
                return finish_from_authoritative_status(
                    race,
                    phase=(
                        f"gate {current_gate_index} delayed terminal "
                        "acceptance"
                    ),
                )
            elif race.active_gate_index == current_gate_index + 1:
                if relation <= 0:
                    raise abort_type(
                        "visual-course transition lacks newer race ingress"
                    )
                if crossing_anchor is None:
                    credit_snapshot = (
                        host.visual_gate_graph.latest_snapshot
                    )
                    reconciliation = (
                        _unlatched_atomic_credit_successor_evidence(
                            credit_snapshot,
                            current_gate_index=current_gate_index,
                            current_track_id=current_track_id,
                        )
                    )
                    if reconciliation is None:
                        raise abort_type(
                            "visual-course race credit arrived without "
                            "credible passage evidence"
                        )
                    credit_wait_reviewed_track_id = str(
                        reconciliation["reviewed_track_id"]
                    )
                    segment["authoritative_credit_reconciliation"] = {
                        **dict(reconciliation),
                        "race_status_sequence": (
                            race.race_status_sequence
                        ),
                        "race_received_monotonic_ns": (
                            race.received_monotonic_ns
                        ),
                    }
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
                    crossing_commitment_deadline_s
                    if crossing_commitment_deadline_s is not None
                    else (
                        censored_passage_coast_started_s
                        + limits.censored_passage_coast_max_duration_s
                    )
                )
            ):
                crossing_started_s = crossing_started_s or now
                if crossing_baseline_race is None:
                    raise abort_type(
                        "visual-course bounded coast lacks a race baseline"
                    )
                break

            snapshot = host.visual_gate_graph.latest_snapshot
            token = getattr(snapshot, "latest_camera_token", None)
            if type(token) is not CameraFrameToken:
                raise abort_type("visual-course graph lacks exact camera token")
            if last_planned_token is not None and token == last_planned_token:
                if refresh_recovery_receiver:
                    raise abort_type(
                        "visual-course receiver replacement did not publish "
                        "a newer tracker/graph snapshot"
                    )
                continue
            recovery_measurement_mode: Optional[
                PostCreditMeasurementMode
            ] = None
            if lifecycle is CourseLifecycle.PROMOTE_REACQUIRE:
                assert post_credit_recovery is not None
                if (
                    (
                        reuse_recovery_graph
                        or refresh_recovery_receiver
                    )
                    and token
                    != post_credit_recovery.admitted_camera_token
                ):
                    if not _token_strictly_newer(
                        token,
                        post_credit_recovery.admitted_camera_token,
                    ):
                        raise abort_type(
                            "visual-course recovery lost its admitted "
                            "camera publication"
                        )
                assert recovery_previous_camera_token is not None
                assert recovery_last_track_token is not None
                recovery_measurement_mode = (
                    classify_post_credit_measurement(
                        snapshot,
                        gate_index=current_gate_index,
                        track_id=current_track_id,
                        previous_camera_token=(
                            recovery_previous_camera_token
                        ),
                        last_track_token=recovery_last_track_token,
                    )
                )
                if (
                    recovery_measurement_mode
                    is PostCreditMeasurementMode.UNSAFE
                ):
                    raise abort_type(
                        "visual-course post-credit recovery measurement "
                        "became unsafe"
                )
                recovery_previous_camera_token = token
                recovery_last_track_token = (
                    snapshot.current_track.latest_token
                )
                if (
                    type(runtime.dynamic_controller)
                    is DynamicVisualCourseSession
                    and post_credit_successor_handoff_required
                    and recovery_measurement_mode
                    in {
                        PostCreditMeasurementMode.ONE_EDGE_CENSORED,
                        PostCreditMeasurementMode.REACQUIRE,
                    }
                ):
                    dynamic_controller = runtime.dynamic_controller
                    assert recovery_deadline_s is not None
                    last_planned_token = token
                    try:
                        dynamic_controller.stage_snapshot(
                            snapshot,
                            host.visual_tracker,
                            expected_gate_index=current_gate_index,
                            expected_current_track_id=current_track_id,
                            adjacent_precredit=False,
                        )
                        propagated_steering = await send_continuity_hold(
                            (
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/"
                                "recovery-propagated-state"
                            ),
                            float(runtime.monotonic()) - segment_started_s,
                            yaw_reference_rad=yaw_reference_rad,
                            successor_steering=True,
                            require_successor_steering=True,
                            command_deadline_s=min(
                                course_deadline_s,
                                segment_deadline_s,
                            ),
                            recovery_measurement_mode=(
                                recovery_measurement_mode
                            ),
                            recovery_snapshot=snapshot,
                        )
                    except RaceActiveBoundaryChangedBeforeWire as exc:
                        raise abort_type(
                            "visual-course race boundary changed during "
                            "post-credit propagated recovery"
                        ) from exc
                    except (TypeError, ValueError) as exc:
                        raise abort_type(
                            "visual-course post-credit local-state "
                            f"propagation refused: {exc}"
                        ) from exc
                    if not propagated_steering:
                        raise abort_type(
                            "visual-course post-credit local-state steering "
                            "was not applied"
                        )
                    segment[
                        "recovery_propagated_state_command_count"
                    ] = int(
                        segment[
                            "recovery_propagated_state_command_count"
                        ]
                    ) + 1
                    segment["recovery_navigation_command_count"] = int(
                        segment["recovery_navigation_command_count"]
                    ) + 1
                    if (
                        transitions
                        and transitions[-1]["to_gate_index"]
                        == current_gate_index
                    ):
                        transitions[-1][
                            "post_transition_navigation_command_count"
                        ] = int(
                            transitions[-1][
                                "post_transition_navigation_command_count"
                            ]
                        ) + 1
                    recovery_deadline_s = min(
                        course_deadline_s,
                        segment_deadline_s,
                        float(runtime.monotonic())
                        + limits.post_credit_fresh_frame_timeout_s,
                    )
                    if (
                        recovery_measurement_mode
                        is PostCreditMeasurementMode.ONE_EDGE_CENSORED
                    ):
                        segment["recovery_one_edge_command_count"] = int(
                            segment["recovery_one_edge_command_count"]
                        ) + 1
                        # The race/graph handoff is required only until one
                        # exact fresh authoritative current publication has
                        # driven an accepted steering-only command.  Keep the
                        # lifecycle in PROMOTE_REACQUIRE, but let subsequent
                        # one-edge publications use the normal current-owned
                        # planner and let missing publications use its bounded
                        # recovery support.  Passage and advance remain
                        # unavailable until the existing two-clean-wire rule.
                        post_credit_successor_handoff_required = (
                            _post_credit_successor_handoff_required_after_command(
                                required_before=(
                                    post_credit_successor_handoff_required
                                ),
                                measurement_mode=recovery_measurement_mode,
                                propagated_steering_applied=(
                                    propagated_steering
                                ),
                            )
                        )
                        if post_credit_successor_handoff_required:
                            raise abort_type(
                                "visual-course accepted one-edge successor "
                                "handoff did not retire"
                            )
                        retirement = {
                            "basis": (
                                POST_CREDIT_SUCCESSOR_HANDOFF_RETIREMENT_BASIS
                            ),
                            "gate_index": current_gate_index,
                            "track_id": current_track_id,
                            "source_camera_token": asdict(token),
                            "measurement_mode": (
                                recovery_measurement_mode.value
                            ),
                            "accepted_propagated_command_count": int(
                                segment[
                                    "recovery_propagated_state_command_count"
                                ]
                            ),
                            "steering_only": True,
                            "passage_authority": False,
                            "advance_authority": False,
                        }
                        segment[
                            "post_credit_successor_handoff_required"
                        ] = False
                        segment[
                            "post_credit_successor_handoff_retirement"
                        ] = retirement
                        host.recorder.emit(
                            "visual_course_post_credit_successor_handoff_retired",
                            **retirement,
                        )
                        refresh_live_summary()
                    continue
                if (
                    recovery_measurement_mode
                    is PostCreditMeasurementMode.REACQUIRE
                ):
                    assert recovery_deadline_s is not None
                    servo_tuning = host.visual_config.servo
                    last_planned_token = token
                    try:
                        support_command = (
                            await send_censored_passage_coast(
                                snapshot=snapshot,
                                authority=_CensoredPassageCoastAuthority(
                                    gate_index=current_gate_index,
                                    track_id=current_track_id,
                                    anchor_camera_token=(
                                        post_credit_recovery
                                        .admitted_camera_token
                                    ),
                                    target_roll_rad=0.0,
                                    target_pitch_rad=float(
                                        servo_tuning.brake_pitch_rad
                                    ),
                                    yaw_rate_rad_s=0.0,
                                    requested_thrust=float(
                                        servo_tuning.brake_thrust
                                    ),
                                ),
                                yaw_reference_rad=yaw_reference_rad,
                                segment_started_s=segment_started_s,
                                stage=(
                                    f"{VISUAL_COURSE_STAGE}/gate"
                                    f"{current_gate_index}/"
                                    "recovery-support"
                                ),
                                command_deadline_s=recovery_deadline_s,
                                count_as_navigation=False,
                            )
                        )
                    except RaceActiveBoundaryChangedBeforeWire as exc:
                        raise abort_type(
                            "visual-course race boundary changed during "
                            "post-credit support"
                        ) from exc
                    if support_command is None:
                        recovery_refresh_receiver_snapshot = True
                        continue
                    segment["recovery_support_command_count"] = int(
                        segment["recovery_support_command_count"]
                    ) + 1
                    continue
            current_ambiguity_eligible = bool(
                mode is VisualApproachMode.APPROACH
                and lifecycle is CourseLifecycle.APPROACH
                and type(runtime.dynamic_controller)
                is DynamicVisualCourseSession
                and passage_admission is None
                and near_plane_latch is None
                and crossing_anchor is None
                and getattr(snapshot, "current_gate_index", None)
                == current_gate_index
                and getattr(snapshot, "current_track_id", None)
                == current_track_id
                and getattr(snapshot, "authority_usable", True) is False
                and getattr(snapshot, "withholding_reason", None)
                == "current_track_ambiguous"
                and getattr(snapshot, "race_finished", True) is False
            )
            if current_ambiguity_eligible:
                dynamic_controller = runtime.dynamic_controller
                assert type(dynamic_controller) is DynamicVisualCourseSession
                ambiguity_proposal_ns = runtime.perf_counter_ns()
                if (
                    type(ambiguity_proposal_ns) is not int
                    or ambiguity_proposal_ns < 0
                ):
                    raise abort_type(
                        "visual-course current-ambiguity quarantine QPC "
                        "clock is invalid"
                    )
                try:
                    ambiguity_hold = (
                        dynamic_controller.continuity_hold_authority(
                            now_monotonic_ns=ambiguity_proposal_ns,
                            maximum_age_s=(
                                runtime.yaw_profile.control_hold_horizon_s
                            ),
                        )
                        if (
                            approach_current_ambiguity_quarantine is None
                        )
                        else None
                    )
                    approach_current_ambiguity_quarantine = (
                        _approach_current_ambiguity_quarantine_authority(
                            snapshot=snapshot,
                            gate_index=current_gate_index,
                            track_id=current_track_id,
                            now_monotonic_ns=ambiguity_proposal_ns,
                            maximum_hold_age_s=(
                                runtime.yaw_profile
                                .control_hold_horizon_s
                            ),
                            fov_summary=segment[
                                "top_fov_pitch_protection"
                            ],
                            hold=ambiguity_hold,
                            existing=(
                                approach_current_ambiguity_quarantine
                            ),
                        )
                    )
                except (TypeError, ValueError) as ambiguity_exc:
                    raise abort_type(
                        "visual-course current-ambiguity quarantine refused: "
                        f"{ambiguity_exc}"
                    ) from ambiguity_exc
                ambiguity_authority = (
                    approach_current_ambiguity_quarantine
                )
                remaining_ambiguity_horizon_s = (
                    ambiguity_authority.expires_monotonic_ns
                    - ambiguity_proposal_ns
                ) / 1_000_000_000.0
                if (
                    not math.isfinite(remaining_ambiguity_horizon_s)
                    or remaining_ambiguity_horizon_s <= 0.0
                ):
                    raise abort_type(
                        "visual-course current-ambiguity quarantine "
                        "expired"
                    )
                quarantine_summary = segment[
                    "approach_current_ambiguity_quarantine"
                ]
                if quarantine_summary is None:
                    quarantine_summary = {
                        "basis": (
                            APPROACH_CURRENT_AMBIGUITY_QUARANTINE_BASIS
                        ),
                        "anchor_camera_token": asdict(
                            ambiguity_authority.clean_camera_token
                        ),
                        "first_ambiguous_camera_token": asdict(
                            ambiguity_authority
                            .first_ambiguous_camera_token
                        ),
                        "latest_ambiguous_camera_token": None,
                        "reacquired_camera_token": None,
                        "anchor_wire_start_monotonic_ns": (
                            ambiguity_authority
                            .anchor_wire_start_monotonic_ns
                        ),
                        "source_wire_start_monotonic_ns": (
                            ambiguity_authority
                            .source_wire_start_monotonic_ns
                        ),
                        "expires_monotonic_ns": (
                            ambiguity_authority.expires_monotonic_ns
                        ),
                        "initial_remaining_horizon_s": (
                            remaining_ambiguity_horizon_s
                        ),
                        "ambiguous_geometry_consumed": False,
                        "lease_renewable": False,
                        "steering_only": True,
                        "passage_authority": False,
                        "advance_authority": False,
                        "outcome": "quarantining",
                    }
                    segment[
                        "approach_current_ambiguity_quarantine"
                    ] = quarantine_summary
                assert isinstance(quarantine_summary, dict)
                quarantine_summary.update(
                    {
                        "latest_ambiguous_camera_token": asdict(token),
                        "remaining_horizon_s": (
                            remaining_ambiguity_horizon_s
                        ),
                    }
                )
                last_planned_token = token
                try:
                    ambiguity_command = (
                        await send_censored_passage_coast(
                            snapshot=snapshot,
                            authority=ambiguity_authority.command,
                            yaw_reference_rad=yaw_reference_rad,
                            segment_started_s=segment_started_s,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/"
                                "approach-current-ambiguity-quarantine"
                            ),
                            command_deadline_s=(
                                now + remaining_ambiguity_horizon_s
                            ),
                            hold_basis=(
                                APPROACH_CURRENT_AMBIGUITY_QUARANTINE_BASIS
                            ),
                        )
                    )
                except RaceActiveBoundaryChangedBeforeWire as race_exc:
                    credited_race = accept_no_wire_race_boundary(
                        race_exc,
                    )
                    break
                if ambiguity_command is None:
                    continue
                approach_current_ambiguity_quarantine_command_count += 1
                approach_command_count += 1
                segment[
                    "approach_current_ambiguity_quarantine_command_count"
                ] = (
                    approach_current_ambiguity_quarantine_command_count
                )
                segment["approach_command_count"] = approach_command_count
                continue
            if approach_current_ambiguity_quarantine is not None:
                reacquired_track = getattr(
                    snapshot,
                    "current_track",
                    None,
                )
                clean_same_current_reacquisition = bool(
                    mode is VisualApproachMode.APPROACH
                    and lifecycle is CourseLifecycle.APPROACH
                    and getattr(snapshot, "current_gate_index", None)
                    == current_gate_index
                    and getattr(snapshot, "current_track_id", None)
                    == current_track_id
                    and getattr(snapshot, "authority_usable", False)
                    is True
                    and getattr(snapshot, "withholding_reason", None)
                    is None
                    and getattr(snapshot, "race_finished", True) is False
                    and reacquired_track is not None
                    and getattr(reacquired_track, "track_id", None)
                    == current_track_id
                    and getattr(reacquired_track, "role", None)
                    is VisualTrackRole.CURRENT
                    and getattr(reacquired_track, "visible", False)
                    is True
                    and getattr(reacquired_track, "ambiguous", True)
                    is False
                    and getattr(reacquired_track, "missed_frame_count", -1)
                    == 0
                    and getattr(reacquired_track, "latest_token", None)
                    == token
                    and _token_strictly_newer(
                        token,
                        approach_current_ambiguity_quarantine
                        .latest_ambiguous_camera_token,
                    )
                )
                if not clean_same_current_reacquisition:
                    raise abort_type(
                        "visual-course current-ambiguity quarantine ended "
                        "without clean same-current reacquisition"
                    )
                quarantine_summary = segment[
                    "approach_current_ambiguity_quarantine"
                ]
                if not isinstance(quarantine_summary, dict):
                    raise abort_type(
                        "visual-course current-ambiguity quarantine summary "
                        "is invalid"
                    )
                quarantine_summary.update(
                    {
                        "reacquired_camera_token": asdict(token),
                        "outcome": "reacquired",
                    }
                )
                host.recorder.emit(
                    "visual_course_approach_current_ambiguity_reacquired",
                    gate_index=current_gate_index,
                    track_id=current_track_id,
                    camera_token=asdict(token),
                    quarantined_command_count=(
                        approach_current_ambiguity_quarantine_command_count
                    ),
                    ambiguous_geometry_consumed=False,
                    lease_renewed=False,
                )
                approach_current_ambiguity_quarantine = None
            passage_forward_closure_authorized = bool(
                not segment["launch_bootstrap"]["enabled"]
                or now - course_started_s
                >= float(
                    host.visual_config.lifecycle.launch_pitch_blend_s
                )
            )
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
                    passage_forward_closure_authorized=(
                        passage_forward_closure_authorized
                    ),
                )
            except (
                VisualApproachCurrentGeometryUnavailable,
                VisualApproachRefusal,
            ) as exc:
                visibility_gap_eligible = bool(
                    type(exc) is VisualApproachRefusal
                    and str(exc)
                    == (
                        "gate graph withheld authoritative "
                        "current-gate identity"
                    )
                    and mode is VisualApproachMode.APPROACH
                    and lifecycle is CourseLifecycle.APPROACH
                    and type(runtime.dynamic_controller)
                    is DynamicVisualCourseSession
                    and passage_admission is None
                    and near_plane_latch is None
                    and crossing_anchor is None
                    and getattr(snapshot, "current_gate_index", None)
                    == current_gate_index
                    and getattr(snapshot, "current_track_id", None)
                    == current_track_id
                    and getattr(snapshot, "authority_usable", True)
                    is False
                    and getattr(snapshot, "withholding_reason", None)
                    == "current_track_not_visible"
                    and getattr(snapshot, "race_finished", True) is False
                )
                if visibility_gap_eligible:
                    dynamic_controller = runtime.dynamic_controller
                    assert type(dynamic_controller) is DynamicVisualCourseSession
                    gap_proposal_ns = runtime.perf_counter_ns()
                    if (
                        type(gap_proposal_ns) is not int
                        or gap_proposal_ns < 0
                    ):
                        raise abort_type(
                            "visual-course propagated visibility-gap QPC "
                            "clock is invalid"
                        ) from exc
                    try:
                        gap_evidence = (
                            dynamic_controller
                            .propagated_current_visibility_gap_authority(
                                track=snapshot.current_track,
                                camera_token=token,
                                now_monotonic_ns=gap_proposal_ns,
                            )
                        )
                        gap_authority = (
                            _approach_propagated_visibility_gap_authority(
                                gap_evidence,
                                snapshot=snapshot,
                                gate_index=current_gate_index,
                                track_id=current_track_id,
                                fov_summary=segment[
                                    "top_fov_pitch_protection"
                                ],
                            )
                        )
                    except (
                        AttributeError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as gap_exc:
                        raise abort_type(
                            "visual-course propagated visibility-gap "
                            f"guidance refused: {gap_exc}"
                        ) from gap_exc
                    if (
                        approach_propagated_visibility_gap_started_s
                        is None
                    ):
                        approach_propagated_visibility_gap_started_s = now
                        approach_propagated_visibility_gap_fresh_frame_count = 0
                        segment[
                            "approach_propagated_visibility_gap"
                        ] = {
                            "basis": (
                                APPROACH_PROPAGATED_VISIBILITY_GAP_BASIS
                            ),
                            "anchor_camera_token": asdict(
                                gap_authority.command.anchor_camera_token
                            ),
                            "first_missing_camera_token": asdict(token),
                            "last_missing_camera_token": None,
                            "reacquired_camera_token": None,
                            "initial_state_horizon_s": (
                                gap_authority.remaining_horizon_s
                            ),
                            "state_deadline_basis": (
                                gap_authority.evidence.get(
                                    "steering_prediction_deadline_basis"
                                )
                            ),
                            "outcome": "propagating",
                        }
                    gap_elapsed_s = (
                        now
                        - approach_propagated_visibility_gap_started_s
                    )
                    try:
                        command_deadline_s = (
                            _approach_propagated_visibility_gap_command_deadline_s(
                                gap_authority,
                                now_s=now,
                                control_period_s=limits.control_period_s,
                            )
                        )
                    except ValueError:
                        assert (
                            segment[
                                "approach_propagated_visibility_gap"
                            ]
                            is not None
                        )
                        segment[
                            "approach_propagated_visibility_gap"
                        ]["outcome"] = "state_horizon_expired"
                        raise abort_type(
                            "visual-course propagated visibility-gap "
                            "horizon expired"
                        ) from exc
                    approach_propagated_visibility_gap_fresh_frame_count += 1
                    segment[
                        "approach_propagated_visibility_gap_fresh_frame_count"
                    ] = (
                        approach_propagated_visibility_gap_fresh_frame_count
                    )
                    assert (
                        segment["approach_propagated_visibility_gap"]
                        is not None
                    )
                    segment["approach_propagated_visibility_gap"].update(
                        {
                            "last_missing_camera_token": asdict(token),
                            "missed_frame_count": (
                                gap_authority.missed_frame_count
                            ),
                            "elapsed_s": gap_elapsed_s,
                            "remaining_state_horizon_s": (
                                gap_authority.remaining_horizon_s
                            ),
                        }
                    )
                    last_planned_token = token
                    try:
                        gap_command = await send_censored_passage_coast(
                            snapshot=snapshot,
                            authority=gap_authority.command,
                            yaw_reference_rad=yaw_reference_rad,
                            segment_started_s=segment_started_s,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/"
                                "approach-propagated-visibility-gap"
                            ),
                            command_deadline_s=command_deadline_s,
                            hold_basis=(
                                APPROACH_PROPAGATED_VISIBILITY_GAP_BASIS
                            ),
                        )
                    except RaceActiveBoundaryChangedBeforeWire as race_exc:
                        credited_race = accept_no_wire_race_boundary(
                            race_exc,
                            allow_unlatched_graph_reconciliation=True,
                        )
                        break
                    if gap_command is None:
                        continue
                    approach_propagated_visibility_gap_command_count += 1
                    approach_command_count += 1
                    segment["approach_command_count"] = (
                        approach_command_count
                    )
                    segment[
                        "approach_propagated_visibility_gap_command_count"
                    ] = (
                        approach_propagated_visibility_gap_command_count
                    )
                    continue
                fresh_top_track = getattr(
                    snapshot,
                    "current_track",
                    None,
                )
                fresh_top_boundary_eligible = bool(
                    type(exc)
                    is VisualApproachCurrentGeometryUnavailable
                    and mode is VisualApproachMode.APPROACH
                    and lifecycle is CourseLifecycle.APPROACH
                    and type(runtime.dynamic_controller)
                    is DynamicVisualCourseSession
                    and passage_admission is None
                    and near_plane_latch is None
                    and crossing_anchor is None
                    and getattr(snapshot, "current_gate_index", None)
                    == current_gate_index
                    and getattr(snapshot, "current_track_id", None)
                    == current_track_id
                    and getattr(snapshot, "authority_usable", None)
                    is True
                    and getattr(snapshot, "withholding_reason", None)
                    is None
                    and getattr(snapshot, "race_finished", None) is False
                    and getattr(fresh_top_track, "clipping", None)
                    is FrameEdge.TOP
                )
                if fresh_top_boundary_eligible:
                    dynamic_controller = runtime.dynamic_controller
                    assert type(dynamic_controller) is DynamicVisualCourseSession
                    recovery_proposal_ns = runtime.perf_counter_ns()
                    if (
                        type(recovery_proposal_ns) is not int
                        or recovery_proposal_ns < 0
                    ):
                        raise abort_type(
                            "visual-course fresh TOP-boundary brake QPC "
                            "clock is invalid"
                        ) from exc
                    try:
                        boundary = _fresh_current_top_boundary_authority(
                            dynamic_controller,
                            snapshot=snapshot,
                            current_gate_index=current_gate_index,
                            current_track_id=current_track_id,
                        )
                        hold = (
                            dynamic_controller.continuity_hold_authority(
                                now_monotonic_ns=recovery_proposal_ns,
                                maximum_age_s=(
                                    runtime.yaw_profile
                                    .control_hold_horizon_s
                                ),
                            )
                        )
                        if not isinstance(hold, Mapping):
                            raise ValueError(
                                "fresh TOP-boundary brake lacks an accepted "
                                "command"
                            )
                        recovery_config = dynamic_controller.core.config
                        recovery_current = boundary.current
                        recovery_center = tuple(
                            map(float, boundary.track.center_norm)
                        )
                        recovery = (
                            _allocate_fresh_top_censored_closure_recovery(
                                raw_top_edge_image_down=(
                                    _raw_bbox_top_image_down(
                                        boundary.sample.bbox_norm
                                    )
                                ),
                                clipping=boundary.track.clipping,
                                center_censored=True,
                                current_visible=True,
                                current_ambiguous=False,
                                current_missed_count=0,
                                current_censored_axes=(
                                    recovery_current.censored_axes
                                ),
                                # The expired aperture is deliberately not
                                # consumed by this exact fresh-boundary path.
                                current_aperture_propagated=False,
                                current_aperture_dynamics_qualified=False,
                                passage_committed=False,
                                capture_pitch_rad=(
                                    _body_to_reference_pitch_rad(
                                        recovery_current
                                        .body_to_reference_wxyz
                                    )
                                ),
                                body_pitch_rate_rad_s=float(
                                    recovery_current.body_rates_rad_s[1]
                                ),
                                pitch_response_delay_s=float(
                                    recovery_config.pitch_command_delay_s
                                ),
                                stable_center_norm=recovery_center,
                                residual_rate_rad_s=(0.0, 0.0),
                                horizontal_angle_scale_rad=float(
                                    recovery_config
                                    .horizontal_angle_scale_rad
                                ),
                                vertical_angle_scale_rad=float(
                                    recovery_config
                                    .vertical_angle_scale_rad
                                ),
                                off_axis_brake_rad=float(
                                    recovery_config.off_axis_brake_rad
                                ),
                                expansion_rate_s=0.0,
                                time_to_contact_s=None,
                                requested_target_pitch_rad=float(
                                    recovery_config.brake_pitch_rad
                                ),
                                fov_protected_target_pitch_rad=float(
                                    hold["target_pitch_rad"]
                                ),
                                requested_thrust=float(
                                    limits.max_thrust
                                ),
                                fresh_boundary_current_authority=boundary,
                            )
                        )
                        if (
                            recovery is None
                            or recovery.allocated_target_pitch_rad
                            != recovery.requested_target_pitch_rad
                            or recovery.forward_closure_authorized
                            or not recovery.steering_only
                            or recovery.passage_authority
                            or recovery.advance_authority
                        ):
                            raise ValueError(
                                "fresh TOP-boundary brake lacks bounded "
                                "steering-only authority"
                            )
                    except (
                        AttributeError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as recovery_exc:
                        raise abort_type(
                            "visual-course fresh TOP-boundary brake "
                            f"refused: {recovery_exc}"
                        ) from recovery_exc
                    if approach_top_recovery_started_s is None:
                        approach_top_recovery_started_s = now
                        approach_top_recovery_fresh_frame_count = 0
                        approach_top_recovery_last_token = None
                        segment["approach_top_recovery"] = {
                            "basis": recovery.basis,
                            "authority_basis": (
                                "fresh-authoritative-current-top-boundary-v1"
                            ),
                            "anchor_camera_token": asdict(token),
                            "first_censored_camera_token": asdict(token),
                            "last_censored_camera_token": None,
                            "clean_reacquired_camera_token": None,
                            "outcome": "braking",
                            "target_roll_rad": float(
                                hold["target_roll_rad"]
                            ),
                            "requested_brake_target_pitch_rad": (
                                recovery.requested_target_pitch_rad
                            ),
                            "applied_brake_target_pitch_rad": (
                                recovery.allocated_target_pitch_rad
                            ),
                            "source_fov_target_pitch_rad": (
                                recovery.fov_protected_target_pitch_rad
                            ),
                            "requested_thrust": (
                                recovery.allocated_thrust
                            ),
                            "steering_only": True,
                            "passage_authority": False,
                            "advance_authority": False,
                            "cross_gap_identity_claimed": False,
                            "max_duration_s": (
                                limits
                                .approach_top_recovery_max_duration_s
                            ),
                            "max_fresh_frames": (
                                limits
                                .approach_top_recovery_max_fresh_frames
                            ),
                            "elapsed_s": 0.0,
                        }
                        near_plane_evidence = NearPlaneEvidence()
                        approach_top_recovery_authority = None
                        segment["near_plane_evidence_frame_count"] = 0
                        host.recorder.emit(
                            "visual_course_approach_top_recovery_started",
                            gate_index=current_gate_index,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/"
                                "approach-top-boundary-brake"
                            ),
                            **segment["approach_top_recovery"],
                        )
                    recovery_elapsed_s = (
                        now - approach_top_recovery_started_s
                    )
                    if (
                        recovery_elapsed_s
                        >= limits.approach_top_recovery_max_duration_s
                        or approach_top_recovery_fresh_frame_count
                        >= limits.approach_top_recovery_max_fresh_frames
                    ):
                        assert segment["approach_top_recovery"] is not None
                        segment["approach_top_recovery"]["outcome"] = (
                            "bounded_brake_expired"
                        )
                        raise abort_type(
                            "visual-course bounded fresh TOP-boundary "
                            "brake expired"
                        ) from exc
                    if (
                        approach_top_recovery_last_token is not None
                        and not _token_strictly_newer(
                            token,
                            approach_top_recovery_last_token,
                        )
                    ):
                        raise abort_type(
                            "visual-course fresh TOP-boundary brake did "
                            "not advance"
                        ) from exc
                    approach_top_recovery_last_token = token
                    approach_top_recovery_fresh_frame_count += 1
                    segment[
                        "approach_top_recovery_fresh_frame_count"
                    ] = approach_top_recovery_fresh_frame_count
                    assert segment["approach_top_recovery"] is not None
                    segment["approach_top_recovery"].update(
                        {
                            "last_censored_camera_token": asdict(token),
                            "elapsed_s": recovery_elapsed_s,
                            "requested_brake_target_pitch_rad": (
                                recovery.requested_target_pitch_rad
                            ),
                            "applied_brake_target_pitch_rad": (
                                recovery.allocated_target_pitch_rad
                            ),
                        }
                    )
                    last_planned_token = token
                    brake_authority = _CensoredPassageCoastAuthority(
                        gate_index=current_gate_index,
                        track_id=current_track_id,
                        anchor_camera_token=token,
                        target_roll_rad=float(hold["target_roll_rad"]),
                        target_pitch_rad=(
                            recovery.allocated_target_pitch_rad
                        ),
                        yaw_rate_rad_s=float(hold["yaw_rate_rad_s"]),
                        requested_thrust=recovery.allocated_thrust,
                    )
                    try:
                        brake_command = await send_censored_passage_coast(
                            snapshot=snapshot,
                            authority=brake_authority,
                            yaw_reference_rad=yaw_reference_rad,
                            segment_started_s=segment_started_s,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/"
                                "approach-top-boundary-brake"
                            ),
                            command_deadline_s=min(
                                course_deadline_s,
                                segment_deadline_s,
                                approach_top_recovery_started_s
                                + (
                                    limits
                                    .approach_top_recovery_max_duration_s
                                ),
                            ),
                            hold_basis=APPROACH_TOP_RECOVERY_BASIS,
                        )
                    except RaceActiveBoundaryChangedBeforeWire as race_exc:
                        credited_race = accept_no_wire_race_boundary(
                            race_exc
                        )
                        break
                    if brake_command is None:
                        continue
                    approach_top_recovery_command_count += 1
                    approach_command_count += 1
                    segment["approach_command_count"] = (
                        approach_command_count
                    )
                    segment[
                        "approach_top_recovery_command_count"
                    ] = approach_top_recovery_command_count
                    host.recorder.emit(
                        "visual_course_fresh_top_boundary_brake_applied",
                        gate_index=current_gate_index,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/"
                            "approach-top-boundary-brake"
                        ),
                        camera_token=asdict(token),
                        requested_brake_target_pitch_rad=(
                            recovery.requested_target_pitch_rad
                        ),
                        applied_brake_target_pitch_rad=(
                            recovery.allocated_target_pitch_rad
                        ),
                        requested_thrust=recovery.allocated_thrust,
                        steering_only=True,
                        passage_authority=False,
                        advance_authority=False,
                        cross_gap_identity_claimed=False,
                        command=asdict(brake_command),
                    )
                    continue
                dropout_authority: Optional[
                    _ApproachInnerDropoutAuthority
                ] = None
                if (
                    type(exc)
                    is VisualApproachCurrentGeometryUnavailable
                    and mode is VisualApproachMode.APPROACH
                    and lifecycle is CourseLifecycle.APPROACH
                    and type(runtime.dynamic_controller)
                    is DynamicVisualCourseSession
                    and passage_admission is None
                    and near_plane_latch is None
                    and crossing_anchor is None
                ):
                    dropout_proposal_ns = runtime.perf_counter_ns()
                    if (
                        type(dropout_proposal_ns) is not int
                        or dropout_proposal_ns < 0
                    ):
                        raise abort_type(
                            "visual-course approach inner-dropout QPC "
                            "clock is invalid"
                        ) from exc
                    dropout_authority = (
                        _derive_approach_inner_dropout_authority(
                            snapshot=snapshot,
                            expected_gate_index=current_gate_index,
                            expected_track_id=current_track_id,
                            maximum_age_s=(
                                runtime.dynamic_controller.core.config
                                .dropout_hold_s
                            ),
                            now_monotonic_ns=dropout_proposal_ns,
                            fov_summary=segment[
                                "top_fov_pitch_protection"
                            ],
                            existing=approach_inner_dropout_authority,
                        )
                    )
                if dropout_authority is not None:
                    dynamic_controller = runtime.dynamic_controller
                    assert type(dynamic_controller) is DynamicVisualCourseSession
                    try:
                        hold = dynamic_controller.continuity_hold_authority(
                            now_monotonic_ns=dropout_proposal_ns,
                            maximum_age_s=dropout_authority.maximum_age_s,
                        )
                    except (TypeError, ValueError) as hold_exc:
                        raise abort_type(
                            "visual-course approach inner-dropout hold "
                            f"expired: {hold_exc}"
                        ) from hold_exc
                    if (
                        not isinstance(hold, Mapping)
                        or float(hold["target_pitch_rad"])
                        > (
                            dropout_authority
                            .maximum_target_pitch_rad
                            + 1e-12
                        )
                        or type(
                            hold.get("source_wire_start_monotonic_ns")
                        )
                        is not int
                        or int(
                            hold["source_wire_start_monotonic_ns"]
                        )
                        < (
                            dropout_authority
                            .anchor_wire_start_monotonic_ns
                        )
                    ):
                        raise abort_type(
                            "visual-course approach inner-dropout hold "
                            "escaped its last FOV-protected pitch"
                        ) from exc
                    near_plane_evidence = NearPlaneEvidence()
                    approach_top_recovery_authority = None
                    segment["near_plane_evidence_frame_count"] = 0
                    last_planned_token = token
                    approach_inner_dropout_authority = (
                        dropout_authority
                    )
                    deadline_s = now + max(
                        0.0,
                        dropout_authority.maximum_age_s
                        - dropout_authority.age_s,
                    )
                    hold_authority = _CensoredPassageCoastAuthority(
                        gate_index=current_gate_index,
                        track_id=current_track_id,
                        anchor_camera_token=(
                            dropout_authority.anchor_camera_token
                        ),
                        target_roll_rad=float(hold["target_roll_rad"]),
                        target_pitch_rad=float(
                            hold["target_pitch_rad"]
                        ),
                        yaw_rate_rad_s=float(hold["yaw_rate_rad_s"]),
                        requested_thrust=float(hold["thrust"]),
                    )
                    hold_command = await send_censored_passage_coast(
                        snapshot=snapshot,
                        authority=hold_authority,
                        yaw_reference_rad=yaw_reference_rad,
                        segment_started_s=segment_started_s,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/"
                            "approach-inner-dropout-hold"
                        ),
                        command_deadline_s=deadline_s,
                        hold_basis=(
                            APPROACH_INNER_DROPOUT_HOLD_BASIS
                        ),
                    )
                    if hold_command is None:
                        continue
                    approach_inner_dropout_hold_command_count += 1
                    approach_command_count += 1
                    segment["approach_command_count"] = (
                        approach_command_count
                    )
                    segment[
                        "approach_inner_dropout_hold_command_count"
                    ] = approach_inner_dropout_hold_command_count
                    hold_summary = segment[
                        "approach_inner_dropout_hold"
                    ]
                    if hold_summary is None:
                        hold_summary = {
                            "basis": (
                                APPROACH_INNER_DROPOUT_HOLD_BASIS
                            ),
                            "anchor_camera_token": asdict(
                                dropout_authority.anchor_camera_token
                            ),
                            "first_dropout_camera_token": asdict(token),
                            "last_dropout_camera_token": None,
                            "reacquired_camera_token": None,
                            "maximum_age_s": (
                                dropout_authority.maximum_age_s
                            ),
                            "maximum_target_pitch_rad": (
                                dropout_authority
                                .maximum_target_pitch_rad
                            ),
                            "outcome": "holding",
                        }
                        segment["approach_inner_dropout_hold"] = (
                            hold_summary
                        )
                    hold_summary.update(
                        {
                            "last_dropout_camera_token": asdict(token),
                            "age_s": dropout_authority.age_s,
                        }
                    )
                    continue
                recovery_authority = approach_top_recovery_authority
                recovery_track = getattr(snapshot, "current_track", None)
                recovery_velocity = getattr(
                    recovery_track,
                    "center_velocity_norm_s",
                    None,
                )
                recovery_confidence = getattr(
                    recovery_track,
                    "confidence",
                    None,
                )
                recovery_association = getattr(
                    recovery_track,
                    "association_confidence",
                    None,
                )
                approach_top_recovery_eligible = bool(
                    type(exc)
                    is VisualApproachCurrentGeometryUnavailable
                    and mode is VisualApproachMode.APPROACH
                    and lifecycle is CourseLifecycle.APPROACH
                    and runtime.dynamic_controller is not None
                    and passage_admission is None
                    and near_plane_latch is None
                    and crossing_anchor is None
                    and recovery_authority is not None
                    and recovery_track is not None
                    and getattr(recovery_track, "clipping", None)
                    == FrameEdge.TOP
                    and isinstance(recovery_velocity, tuple)
                    and len(recovery_velocity) == 2
                    and type(recovery_velocity[1]) in {int, float}
                    and math.isfinite(float(recovery_velocity[1]))
                    and float(recovery_velocity[1]) >= 0.0
                    and type(recovery_confidence) in {int, float}
                    and math.isfinite(float(recovery_confidence))
                    and float(recovery_confidence)
                    >= (
                        DEFAULT_ROLLING_GATE_GRAPH_CONFIG
                        .min_track_confidence
                    )
                    and type(recovery_association) in {int, float}
                    and math.isfinite(float(recovery_association))
                    and float(recovery_association)
                    >= (
                        DEFAULT_ROLLING_GATE_GRAPH_CONFIG
                        .min_association_confidence
                    )
                    and _current_snapshot_ready(
                        snapshot,
                        gate_index=current_gate_index,
                        track_id=current_track_id,
                        newer_than=(
                            recovery_authority
                            .command.anchor_camera_token
                        ),
                        allow_one_edge_censored=True,
                    )
                    and (
                        approach_top_recovery_last_token is None
                        or _token_strictly_newer(
                            token,
                            approach_top_recovery_last_token,
                        )
                    )
                )
                if approach_top_recovery_eligible:
                    assert recovery_authority is not None
                    recovery_proposal_ns = runtime.perf_counter_ns()
                    if (
                        type(recovery_proposal_ns) is not int
                        or recovery_proposal_ns
                        < (
                            recovery_authority
                            .anchor_wire_start_monotonic_ns
                        )
                    ):
                        raise abort_type(
                            "visual-course approach TOP recovery QPC "
                            "clock regressed"
                        ) from exc
                    anchor_age_s = (
                        recovery_proposal_ns
                        - (
                            recovery_authority
                            .anchor_wire_start_monotonic_ns
                        )
                    ) / 1_000_000_000.0
                    remaining_contact_s = (
                        recovery_authority.time_to_contact_s
                        - anchor_age_s
                    )
                    contact_deadline_s = now + remaining_contact_s
                    if (
                        not math.isfinite(anchor_age_s)
                        or (
                            approach_top_recovery_started_s is None
                            and anchor_age_s
                            > (
                                limits
                                .approach_top_recovery_max_duration_s
                            )
                        )
                        or not math.isfinite(remaining_contact_s)
                        or remaining_contact_s <= limits.control_period_s
                        or not math.isfinite(contact_deadline_s)
                        or contact_deadline_s
                        <= now + limits.control_period_s
                    ):
                        raise abort_type(
                            "visual-course approach TOP recovery reached "
                            "its clean-anchor contact horizon"
                        ) from exc
                    if approach_top_recovery_started_s is None:
                        approach_top_recovery_started_s = now
                        approach_top_recovery_fresh_frame_count = 0
                        approach_top_recovery_last_token = None
                        segment["approach_top_recovery"] = {
                            "basis": APPROACH_TOP_RECOVERY_BASIS,
                            "anchor_camera_token": asdict(
                                recovery_authority
                                .command.anchor_camera_token
                            ),
                            "first_censored_camera_token": asdict(token),
                            "last_censored_camera_token": None,
                            "clean_reacquired_camera_token": None,
                            "outcome": "holding",
                            "target_roll_rad": (
                                recovery_authority
                                .command.target_roll_rad
                            ),
                            "target_pitch_rad": (
                                recovery_authority
                                .command.target_pitch_rad
                            ),
                            "requested_thrust": (
                                recovery_authority
                                .command.requested_thrust
                            ),
                            "current_vertical_q": (
                                recovery_authority.current_vertical_q
                            ),
                            "vertical_q_rate_s": (
                                recovery_authority.vertical_q_rate_s
                            ),
                            "predicted_vertical_q": (
                                recovery_authority.predicted_vertical_q
                            ),
                            "predicted_vertical_q_std": (
                                recovery_authority
                                .predicted_vertical_q_std
                            ),
                            "vertical_endpoint_occupancy_q": (
                                recovery_authority
                                .vertical_endpoint_occupancy_q
                            ),
                            "vertical_allowance_q": (
                                recovery_authority
                                .vertical_allowance_q
                            ),
                            "raw_vertical_rate_down_s": (
                                recovery_authority
                                .raw_vertical_rate_down_s
                            ),
                            "time_to_contact_s": (
                                recovery_authority.time_to_contact_s
                            ),
                            "thrust_settle_s": (
                                recovery_authority.thrust_settle_s
                            ),
                            "post_settle_contact_budget_s": (
                                recovery_authority
                                .post_settle_contact_budget_s
                            ),
                            "max_duration_s": (
                                limits
                                .approach_top_recovery_max_duration_s
                            ),
                            "max_fresh_frames": (
                                limits
                                .approach_top_recovery_max_fresh_frames
                            ),
                            "elapsed_s": 0.0,
                        }
                        near_plane_evidence = NearPlaneEvidence()
                        segment["near_plane_evidence_frame_count"] = 0
                        host.recorder.emit(
                            "visual_course_approach_top_recovery_started",
                            gate_index=current_gate_index,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/"
                                "approach-top-recovery"
                            ),
                            **segment["approach_top_recovery"],
                        )
                    recovery_elapsed_s = (
                        now - approach_top_recovery_started_s
                    )
                    if (
                        recovery_elapsed_s
                        >= (
                            limits
                            .approach_top_recovery_max_duration_s
                        )
                        or approach_top_recovery_fresh_frame_count
                        >= (
                            limits
                            .approach_top_recovery_max_fresh_frames
                        )
                    ):
                        assert segment["approach_top_recovery"] is not None
                        segment["approach_top_recovery"]["outcome"] = (
                            "bounded_hold_expired"
                        )
                        raise abort_type(
                            "visual-course bounded approach TOP recovery "
                            "expired"
                        ) from exc
                    approach_top_recovery_last_token = token
                    approach_top_recovery_fresh_frame_count += 1
                    segment[
                        "approach_top_recovery_fresh_frame_count"
                    ] = approach_top_recovery_fresh_frame_count
                    assert segment["approach_top_recovery"] is not None
                    segment["approach_top_recovery"].update(
                        {
                            "last_censored_camera_token": asdict(token),
                            "elapsed_s": recovery_elapsed_s,
                        }
                    )
                    last_planned_token = token
                    command_deadline_s = min(
                        approach_top_recovery_started_s
                        + (
                            limits
                            .approach_top_recovery_max_duration_s
                        ),
                        contact_deadline_s,
                    )
                    try:
                        recovery_command = (
                            await send_censored_passage_coast(
                                snapshot=snapshot,
                                authority=(
                                    recovery_authority.command
                                ),
                                yaw_reference_rad=yaw_reference_rad,
                                segment_started_s=segment_started_s,
                                stage=(
                                    f"{VISUAL_COURSE_STAGE}/gate"
                                    f"{current_gate_index}/"
                                    "approach-top-recovery"
                                ),
                                command_deadline_s=(
                                    command_deadline_s
                                ),
                                hold_basis=(
                                    APPROACH_TOP_RECOVERY_BASIS
                                ),
                            )
                        )
                    except RaceActiveBoundaryChangedBeforeWire as race_exc:
                        credited_race = accept_no_wire_race_boundary(
                            race_exc
                        )
                        break
                    if recovery_command is None:
                        continue
                    approach_top_recovery_command_count += 1
                    approach_command_count += 1
                    segment["approach_command_count"] = (
                        approach_command_count
                    )
                    segment[
                        "approach_top_recovery_command_count"
                    ] = approach_top_recovery_command_count
                    continue

                previous_visible_token = (
                    _latest_latched_observation_token(
                        censored_passage_coast_last_observed_token,
                        last_planned_token,
                        last_clean_passage_token,
                    )
                )
                measurement_mode: Optional[
                    LatchedMeasurementMode
                ] = None
                if (
                    near_plane_latch is not None
                    and previous_visible_token is not None
                ):
                    graph_config = getattr(
                        host.visual_gate_graph,
                        "config",
                        DEFAULT_ROLLING_GATE_GRAPH_CONFIG,
                    )
                    measurement_mode = _classify_latched_snapshot(
                        near_plane_latch,
                        previous_camera_token=previous_visible_token,
                        camera_token=token,
                        snapshot=snapshot,
                        current_gate_index=current_gate_index,
                        min_track_confidence=(
                            graph_config.min_track_confidence
                        ),
                        min_association_confidence=(
                            graph_config.min_association_confidence
                        ),
                    )
                    if (
                        measurement_mode
                        is LatchedMeasurementMode.CREDIT_WAIT
                    ):
                        if (
                            crossing_anchor is None
                            or crossing_coast_authority is None
                        ):
                            raise abort_type(
                                "visual-course credit wait lacks an atomic "
                                "crossing commitment"
                            ) from exc
                        lifecycle = CourseLifecycle.CREDIT_WAIT
                        segment["lifecycle"] = lifecycle.value
                        segment["near_plane_measurement_mode"] = (
                            measurement_mode.value
                        )
                        crossing_started_s = (
                            crossing_started_s or now
                        )
                        if crossing_baseline_race is None:
                            raise abort_type(
                                "visual-course credit wait lacks a race "
                                "baseline"
                            ) from exc
                        if (
                            segment["censored_passage_coast"]
                            is not None
                        ):
                            segment["censored_passage_coast"][
                                "loss_camera_token"
                            ] = asdict(token)
                        last_planned_token = token
                        break
                    if (
                        measurement_mode
                        is LatchedMeasurementMode.UNSAFE
                    ):
                        raise abort_type(
                            "visual-course latched near-plane measurement "
                            "became unsafe"
                        ) from exc
                rate_measurement_refused = bool(
                    type(exc) is VisualApproachRefusal
                    and str(exc) in _LATCHED_RATE_MEASUREMENT_REFUSALS
                )
                censored_coast_eligible = bool(
                    (
                        type(exc)
                        is VisualApproachCurrentGeometryUnavailable
                        or rate_measurement_refused
                    )
                    and crossing_anchor is not None
                    and crossing_coast_authority is not None
                    and previous_visible_token is not None
                    and measurement_mode
                    is LatchedMeasurementMode.COAST
                    and (
                        (
                            mode is VisualApproachMode.PASSAGE
                            and type(passage_admission)
                            is VisualApproachPassageAdmission
                        )
                        or (
                            near_plane_latch is not None
                            and near_plane_latch.basis
                            == DYNAMIC_NEAR_PLANE_LATCH_BASIS
                            and lifecycle
                            is CourseLifecycle.NEAR_PLANE_LATCHED
                        )
                    )
                )
                if censored_coast_eligible:
                    segment["near_plane_measurement_mode"] = (
                        LatchedMeasurementMode.COAST.value
                    )
                    if censored_passage_coast_started_s is None:
                        censored_passage_coast_started_s = now
                        dynamic_coast_deadline_s = (
                            None
                            if crossing_commitment_deadline_s is None
                            else crossing_commitment_deadline_s
                        )
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
                            "requested_thrust": (
                                crossing_coast_authority.requested_thrust
                            ),
                            "max_duration_s": (
                                (
                                    dynamic_coast_deadline_s - now
                                )
                                if dynamic_coast_deadline_s is not None
                                else (
                                    limits
                                    .censored_passage_coast_max_duration_s
                                )
                            ),
                            "max_fresh_frames": (
                                None
                                if dynamic_coast_deadline_s is not None
                                else (
                                    limits
                                    .censored_passage_coast_max_fresh_frames
                                )
                            ),
                            "commitment_deadline_monotonic_s": (
                                crossing_commitment_deadline_s
                            ),
                            "measurement_refusal": (
                                str(exc)
                                if rate_measurement_refused
                                else None
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
                        (
                            crossing_commitment_deadline_s is not None
                            and now
                            >= crossing_commitment_deadline_s
                        )
                        or (
                            crossing_commitment_deadline_s is None
                            and (
                                coast_elapsed_s
                                >= (
                                    limits
                                    .censored_passage_coast_max_duration_s
                                )
                                or censored_passage_coast_fresh_frame_count
                                >= (
                                    limits
                                    .censored_passage_coast_max_fresh_frames
                                )
                            )
                        )
                    ):
                        crossing_started_s = now
                        if crossing_baseline_race is None:
                            raise abort_type(
                                "visual-course bounded coast lacks a race "
                                "baseline"
                            ) from exc
                        break
                    censored_passage_coast_last_observed_token = token
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
                                command_deadline_s=(
                                    (
                                        crossing_commitment_deadline_s
                                    )
                                    if (
                                        crossing_commitment_deadline_s
                                        is not None
                                    )
                                    else (
                                        censored_passage_coast_started_s
                                        + limits
                                        .censored_passage_coast_max_duration_s
                                    )
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

                track = getattr(snapshot, "current_track", None)
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
                    >= track.missed_frame_count
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
                crossing_started_s = crossing_started_s or now
                if crossing_baseline_race is None:
                    raise abort_type(
                        "visual-course crossing lacks a race baseline"
                    )
                if segment["censored_passage_coast"] is not None:
                    segment["censored_passage_coast"][
                        "loss_camera_token"
                    ] = asdict(token)
                break

            if (
                approach_propagated_visibility_gap_started_s
                is not None
            ):
                gap_summary = segment[
                    "approach_propagated_visibility_gap"
                ]
                if isinstance(gap_summary, dict):
                    gap_summary.update(
                        {
                            "reacquired_camera_token": asdict(token),
                            "outcome": "reacquired",
                        }
                    )
                approach_propagated_visibility_gap_started_s = None
                approach_propagated_visibility_gap_fresh_frame_count = 0
            if approach_inner_dropout_authority is not None:
                hold_summary = segment[
                    "approach_inner_dropout_hold"
                ]
                if isinstance(hold_summary, dict):
                    hold_summary.update(
                        {
                            "reacquired_camera_token": asdict(token),
                            "outcome": "reacquired",
                        }
                    )
                approach_inner_dropout_authority = None
            if censored_passage_coast_started_s is not None:
                if (
                    crossing_commitment_deadline_s is None
                    or crossing_coast_authority is None
                ):
                    raise abort_type(
                        "visual-course censored passage coast returned to "
                        "uncensored geometry"
                    )
                if now >= crossing_commitment_deadline_s:
                    crossing_started_s = crossing_started_s or now
                    if crossing_baseline_race is None:
                        raise abort_type(
                            "visual-course bounded coast lacks a race "
                            "baseline"
                        )
                    break
                censored_passage_coast_last_observed_token = token
                censored_passage_coast_fresh_frame_count += 1
                segment[
                    "censored_passage_coast_fresh_frame_count"
                ] = censored_passage_coast_fresh_frame_count
                segment["near_plane_measurement_mode"] = (
                    LatchedMeasurementMode.COAST.value
                )
                segment["censored_passage_coast"].update(
                    {
                        "last_censored_camera_token": asdict(token),
                        "elapsed_s": (
                            now - censored_passage_coast_started_s
                        ),
                        "geometry_reacquired": True,
                    }
                )
                last_planned_token = token
                try:
                    coast_command = await send_censored_passage_coast(
                        snapshot=snapshot,
                        authority=crossing_coast_authority,
                        yaw_reference_rad=yaw_reference_rad,
                        segment_started_s=segment_started_s,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/censored-passage-"
                            "reacquired"
                        ),
                        command_deadline_s=(
                            crossing_commitment_deadline_s
                        ),
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
                segment["passage_command_count"] = passage_command_count
                segment[
                    "censored_passage_coast_command_count"
                ] = censored_passage_coast_command_count
                continue
            if (
                mode is VisualApproachMode.PASSAGE
                and not passage_forward_closure_authorized
            ):
                inhibited_advance = (
                    proposal.servo_output.advance_enabled
                )
                inhibited_target_pitch = (
                    proposal.servo_output.target_pitch_rad
                )
                if (
                    type(inhibited_advance) is not bool
                    or inhibited_advance
                    or type(inhibited_target_pitch)
                    not in {int, float}
                    or not math.isfinite(
                        float(inhibited_target_pitch)
                    )
                    or float(inhibited_target_pitch) < -1e-12
                ):
                    raise abort_type(
                        "visual-course passage escaped its launch "
                        "forward-closure inhibit"
                    )
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
            if mode is VisualApproachMode.PROMOTE_REACQUIRE:
                if (
                    lifecycle is not CourseLifecycle.PROMOTE_REACQUIRE
                    or recovery_measurement_mode
                    not in {
                        PostCreditMeasurementMode.CLEAN,
                        PostCreditMeasurementMode.ONE_EDGE_CENSORED,
                    }
                    or proposal.mode
                    is not VisualApproachMode.PROMOTE_REACQUIRE
                    or proposal.passage_admission is not None
                    or proposal.servo_output.advance_enabled
                    or proposal.servo_output.next_gate_blend != 0.0
                ):
                    raise abort_type(
                        "visual-course recovery proposal escaped its "
                        "current-only no-advance authority"
                    )
                recovery_stage_suffix = (
                    "recovery-clean"
                    if recovery_measurement_mode
                    is PostCreditMeasurementMode.CLEAN
                    else "recovery-one-edge"
                )
                try:
                    accepted = await send_visual(
                        proposal=proposal,
                        snapshot=snapshot,
                        yaw_reference_rad=yaw_reference_rad,
                        segment_started_s=segment_started_s,
                        refresh_ingress_after_slot=reuse_recovery_graph,
                        # Faster braking response belongs to the generic
                        # successor-intercept lifecycle. Current-aperture
                        # approach, passage, and crossing retain their proved
                        # baseline roll/pitch loop.
                        intercept_response_authority=1.0,
                        # Promotion changes race ownership, not the promoted
                        # current gate's raw camera observability.  Apply the
                        # same gate-generic FOV constraint during recovery.
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/"
                            f"{recovery_stage_suffix}"
                        ),
                    )
                except RaceActiveBoundaryChangedBeforeWire as exc:
                    raise abort_type(
                        "visual-course race boundary changed during "
                        "post-credit recovery"
                    ) from exc
                if type(accepted) is _SupersededVisualProposal:
                    recovery_refresh_receiver_snapshot = True
                    continue
                if type(accepted) is not _AcceptedVisualCommand:
                    raise abort_type(
                        "visual-course recovery command outcome is invalid"
                    )
                segment["recovery_navigation_command_count"] = int(
                    segment["recovery_navigation_command_count"]
                ) + 1
                admission_evidence = transitions[-1].get(
                    "recovery_admission"
                )
                if (
                    not isinstance(admission_evidence, dict)
                    or "wire_frame_token" not in admission_evidence
                ):
                    raise abort_type(
                        "visual-course recovery wire lacks its exact "
                        "candidate evidence"
                    )
                if admission_evidence["wire_frame_token"] is None:
                    if (
                        admission_evidence.get("wire_start_monotonic_ns")
                        is not None
                        or admission_evidence.get(
                            "wire_return_monotonic_ns"
                        )
                        is not None
                    ):
                        raise abort_type(
                            "visual-course recovery wire lacks its exact "
                            "candidate evidence"
                        )
                    admission_evidence.update(
                        {
                            "wire_frame_token": asdict(
                                accepted.wire_camera_token
                            ),
                            "wire_start_monotonic_ns": (
                                accepted.wire_start_monotonic_ns
                            ),
                            "wire_return_monotonic_ns": (
                                accepted.wire_return_monotonic_ns
                            ),
                        }
                    )
                recovery_deadline_s = min(
                    course_deadline_s,
                    segment_deadline_s,
                    float(runtime.monotonic())
                    + limits.post_credit_fresh_frame_timeout_s,
                )
                if (
                    recovery_measurement_mode
                    is PostCreditMeasurementMode.CLEAN
                ):
                    segment["recovery_clean_command_count"] = int(
                        segment["recovery_clean_command_count"]
                    ) + 1
                    if recovery_first_clean_wire_token is None:
                        recovery_first_clean_wire_token = (
                            accepted.wire_camera_token
                        )
                else:
                    segment["recovery_one_edge_command_count"] = int(
                        segment["recovery_one_edge_command_count"]
                    ) + 1
                recovery_completed = bool(
                    recovery_measurement_mode
                    is PostCreditMeasurementMode.CLEAN
                    and recovery_first_clean_wire_token is not None
                    and _token_strictly_newer(
                        accepted.wire_camera_token,
                        recovery_first_clean_wire_token,
                    )
                    and not accepted.yaw_soft_stop_zeroed
                )
                if recovery_completed:
                    dynamic_recovery_release = None
                    if (
                        type(runtime.dynamic_controller)
                        is DynamicVisualCourseSession
                    ):
                        try:
                            dynamic_recovery_release = (
                                runtime.dynamic_controller
                                .complete_post_credit_recovery(
                                    camera_token=(
                                        accepted.wire_camera_token
                                    )
                                )
                            )
                        except (TypeError, ValueError) as exc:
                            raise abort_type(
                                "visual-course dynamic post-credit recovery "
                                f"release refused: {exc}"
                            ) from exc
                        if not isinstance(
                            dynamic_recovery_release,
                            Mapping,
                        ):
                            raise abort_type(
                                "visual-course dynamic post-credit recovery "
                                "release evidence is invalid"
                            )
                    lifecycle = CourseLifecycle.APPROACH
                    mode = VisualApproachMode.APPROACH
                    segment["lifecycle"] = lifecycle.value
                    host.recorder.emit(
                        "visual_course_recovery_completed",
                        gate_index=current_gate_index,
                        camera_token=asdict(
                            accepted.wire_camera_token
                        ),
                        clean_command_count=(
                            segment["recovery_clean_command_count"]
                        ),
                        dynamic_recovery_release=(
                            None
                            if dynamic_recovery_release is None
                            else dict(dynamic_recovery_release)
                        ),
                    )
                refresh_live_summary()
                continue
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
                        horizontal_fov_closure_brake_enabled=True,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/approach"
                        ),
                    )
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
                if approach_top_recovery_started_s is not None:
                    current_track = getattr(
                        snapshot,
                        "current_track",
                        None,
                    )
                    if (
                        current_track is None
                        or getattr(current_track, "clipping", None)
                        != FrameEdge.NONE
                        or getattr(
                            current_track,
                            "center_censored",
                            True,
                        )
                        is not False
                    ):
                        raise abort_type(
                            "visual-course approach TOP recovery did not "
                            "return through clean geometry"
                        )
                    assert segment["approach_top_recovery"] is not None
                    segment["approach_top_recovery"].update(
                        {
                            "clean_reacquired_camera_token": asdict(
                                accepted.wire_camera_token
                            ),
                            "outcome": "clean_geometry_reacquired",
                        }
                    )
                    host.recorder.emit(
                        "visual_course_approach_top_recovery_completed",
                        gate_index=current_gate_index,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/approach"
                        ),
                        **segment["approach_top_recovery"],
                    )
                    approach_top_recovery_started_s = None
                try:
                    approach_top_recovery_authority = (
                        _derive_approach_top_recovery_authority(
                            accepted,
                            gate_index=current_gate_index,
                            track_id=current_track_id,
                            raw_vertical_rate_down_s=float(
                                proposal.current_target
                                .normalized_y_rate_down_s
                            ),
                            requested_thrust=(
                                retained_current_aperture_collective(
                                    accepted.command.thrust,
                                    subsupport_collective_authorized=False,
                                )
                            ),
                            minimum_brake_pitch_rad=float(
                                host.visual_config.servo
                                .brake_pitch_rad
                            ),
                            maximum_recovery_duration_s=(
                                limits
                                .approach_top_recovery_max_duration_s
                            ),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course approach TOP recovery evidence is "
                        f"invalid: {exc}"
                    ) from exc
                if accepted.dynamic_evidence is not None:
                    current_track = getattr(
                        snapshot,
                        "current_track",
                        None,
                    )
                    clipping = getattr(
                        current_track,
                        "clipping",
                        None,
                    )
                    if type(clipping) is not FrameEdge:
                        raise abort_type(
                            "visual-course dynamic crossing evidence lacks "
                            "exact clipping state"
                        )
                    try:
                        dynamic_sample = (
                            _dynamic_near_plane_wire_sample(
                                accepted,
                                gate_index=current_gate_index,
                                track_id=current_track_id,
                                target=proposal.current_target,
                                clipping=clipping,
                            )
                        )
                        if dynamic_sample is None:
                            near_plane_evidence = NearPlaneEvidence()
                            candidate_latch = None
                        else:
                            (
                                near_plane_evidence,
                                candidate_latch,
                            ) = advance_dynamic_near_plane_evidence(
                                near_plane_evidence,
                                dynamic_sample,
                                # Dynamic rate/uncertainty qualification
                                # already contains a multi-frame estimator
                                # history.  One exact accepted terminal-safe
                                # wire sample can therefore seal the local
                                # crossing commitment before FOV clipping.
                                required_corridor_frames=1,
                                crossing_min_log_scale=(
                                    limits.crossing_arm_min_log_scale
                                ),
                                horizontal_corridor=(
                                    host.visual_config.servo
                                    .horizontal_corridor
                                ),
                                vertical_corridor=(
                                    host.visual_config.servo
                                    .vertical_corridor
                                ),
                                minimum_post_governor_contact_budget_s=(
                                    runtime.dynamic_controller.core.config
                                    .terminal_min_post_governor_contact_budget_s
                                ),
                                min_track_confidence=(
                                    DEFAULT_ROLLING_GATE_GRAPH_CONFIG
                                    .min_track_confidence
                                ),
                                min_association_confidence=(
                                    DEFAULT_ROLLING_GATE_GRAPH_CONFIG
                                    .min_association_confidence
                                ),
                            )
                    except (TypeError, ValueError) as exc:
                        raise abort_type(
                            "visual-course dynamic crossing evidence is "
                            f"invalid: {exc}"
                        ) from exc
                    segment["near_plane_evidence_frame_count"] = len(
                        near_plane_evidence.samples
                    )
                    if (
                        candidate_latch is not None
                        and not accepted.yaw_soft_stop_zeroed
                        and near_plane_latch is None
                    ):
                        if (
                            candidate_latch.basis
                            != DYNAMIC_NEAR_PLANE_LATCH_BASIS
                        ):
                            raise abort_type(
                                "visual-course dynamic crossing candidate "
                                "has the wrong authority basis"
                            )
                        near_plane_latch = candidate_latch
                        command = accepted.command
                        coast_thrust = (
                            float(command.thrust)
                            - accepted.next_preview_collective_delta
                        )
                        if not (
                            limits.min_thrust - 1e-12
                            <= coast_thrust
                            <= limits.max_thrust + 1e-12
                        ):
                            raise abort_type(
                                "visual-course dynamic crossing coast thrust "
                                "escaped its fixed envelope"
                            )
                        crossing_coast_authority = (
                            _CensoredPassageCoastAuthority(
                                gate_index=current_gate_index,
                                track_id=current_track_id,
                                anchor_camera_token=(
                                    near_plane_latch.anchor_camera_token
                                ),
                                target_roll_rad=accepted.target_roll_rad,
                                target_pitch_rad=max(
                                    accepted.target_pitch_rad,
                                    float(
                                        host.visual_config.servo
                                        .brake_pitch_rad
                                    ),
                                ),
                                yaw_rate_rad_s=command.yaw_rate,
                                requested_thrust=(
                                    retained_current_aperture_collective(
                                        coast_thrust,
                                        subsupport_collective_authorized=True,
                                    )
                                ),
                            )
                        )
                        proposal_reviewed_track_id = getattr(
                            proposal,
                            "latched_next_track_id",
                            None,
                        )
                        if (
                            proposal_reviewed_track_id is not None
                            and (
                                type(proposal_reviewed_track_id) is not str
                                or not proposal_reviewed_track_id
                            )
                        ):
                            raise abort_type(
                                "visual-course dynamic crossing retained "
                                "successor identity is invalid"
                            )
                        dynamic_successor_track_id = (
                            accepted.dynamic_evidence.get(
                                "successor_track_id"
                            )
                        )
                        if (
                            dynamic_successor_track_id is not None
                            and (
                                type(dynamic_successor_track_id) is not str
                                or not dynamic_successor_track_id
                            )
                        ):
                            raise abort_type(
                                "visual-course dynamic crossing successor "
                                "identity is invalid"
                            )
                        if (
                            proposal_reviewed_track_id is not None
                            and dynamic_successor_track_id is not None
                            and proposal_reviewed_track_id
                            != dynamic_successor_track_id
                        ):
                            raise abort_type(
                                "visual-course dynamic crossing successor "
                                "identities diverged"
                            )
                        retained_dynamic_successor_id: Optional[str] = None
                        if (
                            type(runtime.dynamic_controller)
                            is DynamicVisualCourseSession
                            and dynamic_successor_track_id is not None
                            and runtime.dynamic_controller.core
                            .retains_successor_lineage(
                                dynamic_successor_track_id,
                                accepted.wire_return_monotonic_ns,
                            )
                        ):
                            retained_dynamic_successor_id = (
                                dynamic_successor_track_id
                            )
                        committed_reviewed_track_id = (
                            proposal_reviewed_track_id
                            if proposal_reviewed_track_id is not None
                            else retained_dynamic_successor_id
                        )
                        if (
                            crossing_reviewed_track_id is not None
                            and committed_reviewed_track_id is not None
                            and crossing_reviewed_track_id
                            != committed_reviewed_track_id
                        ):
                            raise abort_type(
                                "visual-course dynamic crossing changed its "
                                "reviewed successor identity"
                            )
                        if committed_reviewed_track_id is not None:
                            crossing_reviewed_track_id = (
                                committed_reviewed_track_id
                            )
                            crossing_successor_identity_sealed = True
                        anchor = near_plane_latch.anchor_sample
                        assert (
                            anchor.crossing_prediction_horizon_s
                            is not None
                        )
                        # A physical commitment reaches its own predicted
                        # crossing plus the already-bounded near-plane ingress
                        # allowance, but never outlives a propagated aperture
                        # lease.  This is a state-evidence lease, not a second
                        # command slew governor.
                        contact_plus_ingress_horizon_s = (
                            float(
                                anchor.crossing_prediction_horizon_s
                            )
                            + min(
                                limits
                                .censored_passage_coast_max_duration_s,
                                limits.crossing_status_timeout_s,
                            )
                        )
                        commitment_horizon_s = (
                            min(
                                DYNAMIC_CROSSING_PREDICTION_MAX_HORIZON_S,
                                contact_plus_ingress_horizon_s,
                            )
                            if (
                                anchor
                                .propagated_state_horizon_remaining_s
                                is None
                            )
                            else min(
                                contact_plus_ingress_horizon_s,
                                float(
                                    anchor
                                    .propagated_state_horizon_remaining_s
                                ),
                            )
                        )
                        commitment_deadline_perf_counter_ns = (
                            accepted.wire_start_monotonic_ns
                            + round(
                                commitment_horizon_s
                                * 1_000_000_000.0
                            )
                        )
                        predicted_contact_perf_counter_ns = (
                            accepted.wire_start_monotonic_ns
                            + round(
                                float(
                                    anchor.crossing_prediction_horizon_s
                                )
                                * 1_000_000_000.0
                            )
                        )
                        deadline_validation_perf_counter_ns = (
                            runtime.perf_counter_ns()
                        )
                        deadline_validation_monotonic_s = float(
                            runtime.monotonic()
                        )
                        commitment_remaining_s = (
                            commitment_deadline_perf_counter_ns
                            - deadline_validation_perf_counter_ns
                        ) / 1_000_000_000.0
                        predicted_contact_remaining_s = (
                            predicted_contact_perf_counter_ns
                            - deadline_validation_perf_counter_ns
                        ) / 1_000_000_000.0
                        if commitment_remaining_s <= 0.0:
                            raise abort_type(
                                "visual-course dynamic crossing commitment "
                                "has no remaining causal horizon"
                            )
                        crossing_commitment_deadline_s = (
                            deadline_validation_monotonic_s
                            + commitment_remaining_s
                        )
                        crossing_predicted_contact_deadline_s = (
                            deadline_validation_monotonic_s
                            + max(0.0, predicted_contact_remaining_s)
                        )
                        crossing_anchor = {
                            "basis": near_plane_latch.basis,
                            "camera_token": (
                                near_plane_latch.anchor_camera_token
                            ),
                            "track_id": near_plane_latch.track_id,
                            "gate_index": near_plane_latch.gate_index,
                            "accepted_wire_frame_count": len(
                                near_plane_latch.evidence.samples
                            ),
                            "advance_command_count": advance_command_count,
                            "reviewed_successor_track_id": (
                                crossing_reviewed_track_id
                            ),
                            "log_scale": anchor.log_scale,
                            "log_scale_rate_s": anchor.log_scale_rate_s,
                            "normalized_x": anchor.normalized_x,
                            "normalized_y_down": (
                                anchor.normalized_y_down
                            ),
                            "normalized_x_rate_s": (
                                anchor.normalized_x_rate_s
                            ),
                            "normalized_y_rate_down_s": (
                                anchor.normalized_y_rate_down_s
                            ),
                            "normalized_x_std": anchor.normalized_x_std,
                            "normalized_y_std": anchor.normalized_y_std,
                            "log_scale_std": anchor.log_scale_std,
                            "crossing_prediction_horizon_s": (
                                anchor.crossing_prediction_horizon_s
                            ),
                            "predicted_crossing_error_norm": [
                                anchor.predicted_crossing_x_norm,
                                anchor.predicted_crossing_y_down_norm,
                            ],
                            "predicted_crossing_std_norm": [
                                anchor.predicted_crossing_x_std_norm,
                                anchor.predicted_crossing_y_std_norm,
                            ],
                            "crossing_allowance_norm": [
                                anchor.crossing_allowance_x_norm,
                                anchor.crossing_allowance_y_norm,
                            ],
                            "crossing_swept_occupancy_norm": [
                                anchor.crossing_swept_x_occupancy_norm,
                                anchor.crossing_swept_y_occupancy_norm,
                            ],
                            "terminal_crossing_occupancy_norm": [
                                (
                                    abs(
                                        anchor
                                        .predicted_crossing_x_norm
                                    )
                                    + 2.0
                                    * anchor
                                    .predicted_crossing_x_std_norm
                                ),
                                (
                                    abs(
                                        anchor
                                        .predicted_crossing_y_down_norm
                                    )
                                    + 2.0
                                    * anchor
                                    .predicted_crossing_y_std_norm
                                ),
                            ],
                            "predicted_crossing_clearance_norm": [
                                (
                                    anchor.crossing_allowance_x_norm
                                    - abs(
                                        anchor.predicted_crossing_x_norm
                                    )
                                    - 2.0
                                    * anchor
                                    .predicted_crossing_x_std_norm
                                ),
                                (
                                    anchor.crossing_allowance_y_norm
                                    - abs(
                                        anchor.predicted_crossing_y_down_norm
                                    )
                                    - 2.0
                                    * anchor.predicted_crossing_y_std_norm
                                ),
                            ],
                            "post_governor_contact_budget_s": (
                                anchor.post_governor_contact_budget_s
                            ),
                            "propagated_state_horizon_remaining_s": (
                                anchor
                                .propagated_state_horizon_remaining_s
                            ),
                            "commitment_horizon_s": (
                                commitment_horizon_s
                            ),
                            "predicted_contact_deadline_perf_counter_ns": (
                                predicted_contact_perf_counter_ns
                            ),
                            "commitment_deadline_perf_counter_ns": (
                                commitment_deadline_perf_counter_ns
                            ),
                            "commitment_deadline_monotonic_s": (
                                crossing_commitment_deadline_s
                            ),
                            "command": asdict(command),
                            "current_only_crossing_coast_thrust": (
                                coast_thrust
                            ),
                        }
                        last_clean_passage_token = (
                            near_plane_latch.anchor_camera_token
                        )
                        lifecycle = CourseLifecycle.NEAR_PLANE_LATCHED
                        passage_started_s = (
                            passage_started_s
                            or float(runtime.monotonic())
                        )
                        segment["passage_authority_enabled"] = True
                        segment["lifecycle"] = lifecycle.value
                        segment["near_plane_reviewed_track_id"] = (
                            crossing_reviewed_track_id
                        )
                        segment[
                            "near_plane_successor_identity_sealed"
                        ] = crossing_successor_identity_sealed
                        segment["near_plane_latch"] = {
                            **crossing_anchor,
                            "camera_token": asdict(
                                near_plane_latch.anchor_camera_token
                            ),
                        }
                        segment["crossing_anchor"] = dict(
                            segment["near_plane_latch"]
                        )
                        host.recorder.emit(
                            "visual_course_near_plane_latched",
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/approach"
                            ),
                            **segment["near_plane_latch"],
                        )
                if proposal.passage_admission is not None:
                    if accepted.yaw_soft_stop_zeroed:
                        segment[
                            "passage_admission_yaw_soft_stop_withheld_count"
                        ] = int(
                            segment[
                                "passage_admission_yaw_soft_stop_withheld_count"
                            ]
                        ) + 1
                        continue
                    if (
                        accepted.dynamic_evidence is not None
                        and (
                            near_plane_latch is None
                            or near_plane_latch.basis
                            != DYNAMIC_NEAR_PLANE_LATCH_BASIS
                        )
                    ):
                        continue
                    passage_admission = proposal.passage_admission
                    mode = VisualApproachMode.PASSAGE
                    lifecycle = (
                        CourseLifecycle.NEAR_PLANE_LATCHED
                        if near_plane_latch is not None
                        else CourseLifecycle.PASSAGE_ARMED
                    )
                    passage_started_s = float(runtime.monotonic())
                    segment["passage_authority_enabled"] = True
                    segment["lifecycle"] = lifecycle.value
                    segment["passage_admission"] = asdict(
                        passage_admission
                    )
                    if (
                        passage_admission.preview_track_id is not None
                    ):
                        if not (
                            crossing_reviewed_track_id is None
                            or crossing_reviewed_track_id
                            == passage_admission.preview_track_id
                        ):
                            raise abort_type(
                                "visual-course passage admission changed its "
                                "reviewed successor identity"
                            )
                        crossing_reviewed_track_id = (
                            passage_admission.preview_track_id
                        )
                        if (
                            near_plane_latch is not None
                            and near_plane_latch.basis
                            == DYNAMIC_NEAR_PLANE_LATCH_BASIS
                        ):
                            crossing_successor_identity_sealed = True
                        segment["near_plane_reviewed_track_id"] = (
                            crossing_reviewed_track_id
                        )
                        segment[
                            "near_plane_successor_identity_sealed"
                        ] = crossing_successor_identity_sealed
                    if (
                        accepted.dynamic_evidence is not None
                        and crossing_reviewed_track_id is not None
                    ):
                        if (
                            crossing_coast_authority is None
                            or not crossing_successor_identity_sealed
                        ):
                            raise abort_type(
                                "visual-course passage admission lacks its "
                                "sealed successor crossing command"
                            )
                        prior_crossing_command = (
                            crossing_coast_authority
                        )
                        try:
                            crossing_coast_authority = (
                                _finalize_crossing_command_at_passage_admission(
                                    prior_crossing_command,
                                    accepted,
                                    gate_index=current_gate_index,
                                    current_track_id=current_track_id,
                                    reviewed_successor_track_id=(
                                        crossing_reviewed_track_id
                                    ),
                                )
                            )
                        except (TypeError, ValueError) as exc:
                            raise abort_type(
                                "visual-course passage-admission command "
                                f"seal is invalid: {exc}"
                            ) from exc
                        segment["passage_command_seal"] = {
                            "basis": (
                                "accepted-passage-admission-successor-roll-v1"
                            ),
                            "camera_token": asdict(
                                accepted.wire_camera_token
                            ),
                            "source_wire_start_monotonic_ns": (
                                accepted.wire_start_monotonic_ns
                            ),
                            "previous_command": asdict(
                                prior_crossing_command
                            ),
                            "command": asdict(
                                crossing_coast_authority
                            ),
                        }
                        host.recorder.emit(
                            "visual_course_passage_command_sealed",
                            gate_index=current_gate_index,
                            stage=(
                                f"{VISUAL_COURSE_STAGE}/gate"
                                f"{current_gate_index}/approach"
                            ),
                            **segment["passage_command_seal"],
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
                    top_fov_transition_owned=bool(
                        near_plane_latch is not None
                        or crossing_anchor is not None
                    ),
                    committed_crossing_authority=(
                        crossing_coast_authority
                    ),
                    stage=(
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/passage"
                    ),
                )
            except RaceActiveBoundaryChangedBeforeWire as exc:
                credited_race = accept_no_wire_race_boundary(exc)
                break
            if type(accepted) is _SupersededVisualProposal:
                refresh_live_summary()
                continue
            if type(accepted) is not _AcceptedVisualCommand:
                raise abort_type(
                    "visual-course passage command outcome is invalid"
                )
            command = accepted.command
            target = proposal.current_target
            last_clean_passage_token = token
            passage_command_count += 1
            segment["passage_command_count"] = passage_command_count
            if proposal.servo_output.next_gate_blend > 0.0:
                passage_next_preview_command_count += 1
                next_target = proposal.next_target
                if (
                    next_target is None
                    or proposal.servo_output.next_horizontal_error is None
                ):
                    raise abort_type(
                        "visual-course accepted preview lacks successor "
                        "geometry"
                    )
                last_passage_successor_horizontal = float(
                    proposal.servo_output.next_horizontal_error
                )
                last_passage_successor_horizontal_rate = float(
                    next_target.normalized_x_rate_s
                )
                segment[
                    "passage_next_preview_command_count"
                ] = passage_next_preview_command_count
            # The calibrated yaw limiter owns only the yaw channel.  Every
            # publication must independently re-pass it; advance remains a
            # diagnostic count and never supplies near-plane authority.
            refresh_live_summary()
            if (
                proposal.servo_output.advance_enabled
                and not accepted.yaw_soft_stop_zeroed
            ):
                advance_command_count += 1
                segment["advance_command_count"] = advance_command_count
            if (
                near_plane_latch is not None
                and near_plane_latch.basis
                == DYNAMIC_NEAR_PLANE_LATCH_BASIS
                and crossing_coast_authority is not None
                and crossing_anchor is not None
                and crossing_successor_identity_sealed
                and crossing_reviewed_track_id is not None
            ):
                prior_crossing_targets = (
                    crossing_coast_authority.target_roll_rad,
                    crossing_coast_authority.target_pitch_rad,
                    crossing_coast_authority.yaw_rate_rad_s,
                )
                try:
                    refreshed_crossing_authority = (
                        _refresh_committed_successor_steering(
                            crossing_coast_authority,
                            accepted,
                            gate_index=current_gate_index,
                            current_track_id=current_track_id,
                            reviewed_successor_track_id=(
                                crossing_reviewed_track_id
                            ),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course committed successor steering refresh "
                        f"refused: {exc}"
                    ) from exc
                if refreshed_crossing_authority != crossing_coast_authority:
                    crossing_coast_authority = (
                        refreshed_crossing_authority
                    )
                    refresh_evidence = {
                        "basis": (
                            "bounded-committed-successor-crossing-steering-v1"
                        ),
                        "camera_token": asdict(
                            accepted.wire_camera_token
                        ),
                        "reviewed_successor_track_id": (
                            crossing_reviewed_track_id
                        ),
                        "previous_target_attitude_yaw": list(
                            prior_crossing_targets
                        ),
                        "requested_target_attitude_yaw": [
                            crossing_coast_authority.target_roll_rad,
                            crossing_coast_authority.target_pitch_rad,
                            crossing_coast_authority.yaw_rate_rad_s,
                        ],
                        "accepted_wire_body_rates": [
                            accepted.command.roll_rate,
                            accepted.command.pitch_rate,
                            accepted.command.yaw_rate,
                        ],
                        "steering_only": True,
                        "passage_authority": False,
                        "advance_authority": False,
                    }
                    segment[
                        "committed_successor_steering_refresh_count"
                    ] = int(
                        segment[
                            "committed_successor_steering_refresh_count"
                        ]
                    ) + 1
                    segment[
                        "last_committed_successor_steering_refresh"
                    ] = refresh_evidence
                    host.recorder.emit(
                        "visual_course_committed_successor_steering_refreshed",
                        gate_index=current_gate_index,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/passage"
                        ),
                        **refresh_evidence,
                    )
            if near_plane_latch is None:
                track = getattr(snapshot, "current_track", None)
                clipping = getattr(track, "clipping", None)
                if type(clipping) is not FrameEdge:
                    raise abort_type(
                        "visual-course near-plane evidence lacks exact "
                        "clipping state"
                    )
                wire_sample = NearPlaneWireSample(
                    gate_index=current_gate_index,
                    track_id=target.track_id,
                    camera_token=token,
                    wire_camera_token=accepted.wire_camera_token,
                    observation_monotonic_ns=(
                        accepted.observation_monotonic_ns
                    ),
                    publication_monotonic_ns=(
                        accepted.publication_monotonic_ns
                    ),
                    wire_start_monotonic_ns=(
                        accepted.wire_start_monotonic_ns
                    ),
                    wire_return_monotonic_ns=(
                        accepted.wire_return_monotonic_ns
                    ),
                    wire_race_gate_index=(
                        accepted.wire_race_gate_index
                    ),
                    publication_pinned_through_transport_return=(
                        accepted
                        .publication_pinned_through_transport_return
                    ),
                    normalized_x=target.normalized_x,
                    normalized_y_down=target.normalized_y_down,
                    normalized_x_rate_s=target.normalized_x_rate_s,
                    normalized_y_rate_down_s=(
                        target.normalized_y_rate_down_s
                    ),
                    log_scale=target.log_scale,
                    log_scale_rate_s=target.log_scale_rate_s,
                    confidence=target.confidence,
                    association_confidence=(
                        target.association_confidence
                    ),
                    clipping=clipping,
                    center_censored=target.center_censored,
                    ambiguous=target.ambiguous,
                    command_roll_rate=command.roll_rate,
                    command_pitch_rate=command.pitch_rate,
                    command_yaw_rate=command.yaw_rate,
                    command_thrust=command.thrust,
                )
                graph_config = getattr(
                    host.visual_gate_graph,
                    "config",
                    DEFAULT_ROLLING_GATE_GRAPH_CONFIG,
                )
                near_plane_evidence, candidate_latch = (
                    advance_near_plane_evidence(
                        near_plane_evidence,
                        wire_sample,
                        required_corridor_frames=(
                            host.visual_config.servo
                            .required_corridor_frames
                        ),
                        crossing_min_log_scale=(
                            limits.crossing_arm_min_log_scale
                        ),
                        min_track_confidence=(
                            graph_config.min_track_confidence
                        ),
                        min_association_confidence=(
                            graph_config.min_association_confidence
                        ),
                    )
                )
                segment["near_plane_evidence_frame_count"] = len(
                    near_plane_evidence.samples
                )
                if candidate_latch is not None:
                    near_plane_latch = candidate_latch
                    lifecycle = CourseLifecycle.NEAR_PLANE_LATCHED
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
                            "visual-course near-plane coast thrust escaped "
                            "its fixed envelope"
                        )
                    servo_tuning = host.visual_config.servo
                    coast_brake_rate_span = float(
                        servo_tuning.brake_scale_rate_s
                        - servo_tuning.stable_scale_rate_s
                    )
                    if coast_brake_rate_span <= 0.0:
                        raise abort_type(
                            "visual-course closure brake configuration is "
                            "invalid"
                        )
                    coast_brake_authority = max(
                        0.0,
                        min(
                            1.0,
                            (
                                float(target.log_scale_rate_s)
                                - float(servo_tuning.stable_scale_rate_s)
                            )
                            / coast_brake_rate_span,
                        ),
                    )
                    crossing_coast_target_pitch = max(
                        accepted.target_pitch_rad,
                        float(servo_tuning.brake_pitch_rad)
                        + coast_brake_authority
                        * (
                            MAX_VISUAL_TARGET_PITCH_RAD
                            - float(servo_tuning.brake_pitch_rad)
                        ),
                    )
                    crossing_successor_yaw_rate = 0.0
                    if (
                        last_passage_successor_horizontal is not None
                        and last_passage_successor_horizontal_rate
                        is not None
                    ):
                        crossing_successor_yaw_rate = (
                            visual_bearing_yaw_rate(
                                last_passage_successor_horizontal,
                                last_passage_successor_horizontal_rate,
                                servo_tuning,
                            )
                        )
                    crossing_coast_authority = (
                        _CensoredPassageCoastAuthority(
                            gate_index=current_gate_index,
                            track_id=current_track_id,
                            anchor_camera_token=(
                                candidate_latch.anchor_camera_token
                            ),
                            target_roll_rad=accepted.target_roll_rad,
                            target_pitch_rad=crossing_coast_target_pitch,
                            yaw_rate_rad_s=crossing_successor_yaw_rate,
                            requested_thrust=(
                                retained_current_aperture_collective(
                                    crossing_coast_thrust,
                                    subsupport_collective_authorized=True,
                                )
                            ),
                        )
                    )
                    crossing_anchor = {
                        "basis": candidate_latch.basis,
                        "camera_token": (
                            candidate_latch.anchor_camera_token
                        ),
                        "track_id": candidate_latch.track_id,
                        "gate_index": candidate_latch.gate_index,
                        "accepted_wire_frame_count": len(
                            candidate_latch.evidence.samples
                        ),
                        "advance_command_count": (
                            advance_command_count
                        ),
                        "log_scale": target.log_scale,
                        "log_scale_rate_s": (
                            target.log_scale_rate_s
                        ),
                        "normalized_x": target.normalized_x,
                        "normalized_y_down": (
                            target.normalized_y_down
                        ),
                        "normalized_x_rate_s": (
                            target.normalized_x_rate_s
                        ),
                        "normalized_y_rate_down_s": (
                            target.normalized_y_rate_down_s
                        ),
                        "command": asdict(command),
                        "current_only_crossing_coast_thrust": (
                            crossing_coast_thrust
                        ),
                    }
                    segment["lifecycle"] = lifecycle.value
                    segment["near_plane_latch"] = {
                        **crossing_anchor,
                        "camera_token": asdict(
                            candidate_latch.anchor_camera_token
                        ),
                    }
                    # Preserve the compact legacy key for downstream evidence
                    # readers while making its generic physical basis explicit.
                    segment["crossing_anchor"] = dict(
                        segment["near_plane_latch"]
                    )
                    host.recorder.emit(
                        "visual_course_near_plane_latched",
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/passage"
                        ),
                        **segment["near_plane_latch"],
                    )
            continue
        if crossing_started_s is None:
            crossing_started_s = float(runtime.monotonic())
        if crossing_baseline_race is None:
            crossing_baseline_race = last_race
        crossing_deadline_s = (
            crossing_started_s + limits.crossing_status_timeout_s
        )
        if crossing_predicted_contact_deadline_s is not None:
            # Race ingress is an independent 4 Hz authority.  Its bounded
            # polling window starts no earlier than predicted physical
            # contact; the command/state lease below may expire first.
            crossing_deadline_s = (
                max(
                    crossing_started_s,
                    crossing_predicted_contact_deadline_s,
                )
                + limits.crossing_status_timeout_s
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
                    # and retain the bounded latched coast until credit or
                    # timeout.
                    crossing_baseline_race = race
                    last_race = race
                else:
                    raise abort_type(
                        "visual-course crossing produced an invalid gate index"
                    )
            if (
                crossing_coast_authority is None
                or near_plane_latch is None
                or crossing_anchor is None
            ):
                raise abort_type(
                    "visual-course credit wait lacks latched coast authority"
                )
            snapshot = host.visual_gate_graph.latest_snapshot
            token = getattr(snapshot, "latest_camera_token", None)
            if type(token) is not CameraFrameToken:
                raise abort_type(
                    "visual-course credit wait lacks a fresh camera token"
                )
            if last_planned_token is None:
                raise abort_type(
                    "visual-course credit wait lacks prior camera lineage"
                )
            if token == last_planned_token:
                continue
            if not _token_strictly_newer(token, last_planned_token):
                raise abort_type(
                    "visual-course credit wait camera lineage regressed"
                )
            graph_config = getattr(
                host.visual_gate_graph,
                "config",
                DEFAULT_ROLLING_GATE_GRAPH_CONFIG,
            )
            measurement_mode = _classify_latched_snapshot(
                near_plane_latch,
                previous_camera_token=last_planned_token,
                camera_token=token,
                snapshot=snapshot,
                current_gate_index=current_gate_index,
                min_track_confidence=(
                    graph_config.min_track_confidence
                ),
                min_association_confidence=(
                    graph_config.min_association_confidence
                ),
            )
            segment["near_plane_measurement_mode"] = (
                measurement_mode.value
            )
            if measurement_mode is LatchedMeasurementMode.UNSAFE:
                raise abort_type(
                    "visual-course credit-wait measurement became unsafe"
                )
            adjacent_candidates = tuple(
                candidate
                for candidate in getattr(snapshot, "next_candidates", ())
                if (
                    getattr(candidate, "promotable", False) is True
                    or (
                        measurement_mode
                        is LatchedMeasurementMode.CREDIT_WAIT
                        and getattr(candidate, "relationship", None) is None
                        and getattr(candidate, "stable_frame_count", 0)
                        >= graph_config.min_next_candidate_frames
                        and getattr(candidate, "confidence", -1.0)
                        >= graph_config.min_track_confidence
                        and getattr(
                            candidate,
                            "association_confidence",
                            -1.0,
                        )
                        >= graph_config.min_association_confidence
                    )
                )
            )
            # Across the physical gate-plane occlusion, a clean successor can
            # lack only the simultaneous-image relationship needed for
            # promotion.  That specific gap may receive fresh no-advance
            # guidance.  Its image-track ID may differ from the pre-clipping
            # preview: uniqueness, stability, confidence, exact publication,
            # and no provisional contender own this steering-only rebind.
            # Low-confidence or contended relationship failures do not gain
            # command authority.
            admitted_adjacent_candidate: Optional[Any] = None
            if (
                getattr(
                    snapshot,
                    "next_selection_ambiguous",
                    True,
                )
                is False
                and not getattr(snapshot, "provisional_track_ids", ())
                and len(adjacent_candidates) == 1
                and adjacent_candidates[0].latest_token == token
                and type(adjacent_candidates[0].track_id) is str
                and adjacent_candidates[0].track_id
                and crossing_coast_authority is not None
                and type(passage_admission)
                is VisualApproachPassageAdmission
            ):
                admitted_adjacent_candidate = adjacent_candidates[0]
            if (
                admitted_adjacent_candidate is not None
                and (
                    credit_wait_adjacent_planner is None
                    or credit_wait_adjacent_track_id
                    != admitted_adjacent_candidate.track_id
                )
            ):
                credit_wait_adjacent_track_id = (
                    admitted_adjacent_candidate.track_id
                )
                credit_wait_adjacent_planner = make_planner(
                    track_id=credit_wait_adjacent_track_id,
                    gate_index=current_gate_index + 1,
                    next_gate_blend=(
                        host.visual_config.lifecycle
                        .next_gate_blend_max
                    ),
                )
                if not callable(
                    getattr(
                        credit_wait_adjacent_planner,
                        "observe_promotable_adjacent",
                        None,
                    )
                ):
                    raise abort_type(
                        "visual-course adjacent planner lacks bounded "
                        "recenter authority"
                    )

            adjacent_proposal: Optional[Any] = None
            adjacent_track: Optional[Any] = None
            adjacent_yaw_reference_rad = yaw_reference_rad
            if (
                admitted_adjacent_candidate is not None
                and credit_wait_adjacent_planner is not None
                and credit_wait_adjacent_track_id
                == admitted_adjacent_candidate.track_id
            ):
                if crossing_wait_adjacent_command_count == 0:
                    (
                        _roll,
                        _pitch,
                        adjacent_yaw_reference_rad,
                        _rates,
                    ) = _attitude_state(host, abort_type)
                adjacent_excursion, _rates, _euler_yaw_rate = (
                    _assert_course_attitude_state(
                        host,
                        yaw_reference_rad=adjacent_yaw_reference_rad,
                        limits=limits,
                        yaw_profile=runtime.yaw_profile,
                        abort_type=abort_type,
                        phase=(
                            f"gate {current_gate_index} credit-wait "
                            "adjacent recenter"
                        ),
                    )
                )
                try:
                    assert credit_wait_adjacent_track_id is not None
                    adjacent_track = host.visual_tracker.track(
                        credit_wait_adjacent_track_id
                    )
                    adjacent_proposal = (
                        credit_wait_adjacent_planner
                        .observe_promotable_adjacent(
                            snapshot,
                            host.visual_tracker,
                            runtime.perf_counter_ns()
                            / 1_000_000_000.0,
                            now - segment_started_s,
                            adjacent_excursion,
                        )
                    )
                except (KeyError, VisualApproachAdjacentUnavailable):
                    adjacent_track = None
                    adjacent_proposal = None
            if adjacent_proposal is not None:
                assert credit_wait_adjacent_track_id is not None
                assert adjacent_track is not None
                assert crossing_coast_authority is not None
                credit_wait_reviewed_track_id = (
                    credit_wait_adjacent_track_id
                )
                segment["crossing_wait_adjacent_track_id"] = (
                    credit_wait_reviewed_track_id
                )
                adjacent_output = adjacent_proposal.servo_output
                adjacent_roll = float(
                    adjacent_output.target_roll_rad
                )
                adjacent_yaw = float(
                    adjacent_output.yaw_rate_rad_s
                )
                if (
                    not math.isfinite(adjacent_roll)
                    or abs(adjacent_roll)
                    > MAX_VISUAL_TARGET_ROLL_RAD + 1e-12
                    or not math.isfinite(adjacent_yaw)
                    or abs(adjacent_yaw)
                    > MAX_VISUAL_YAW_RATE_RAD_S + 1e-12
                ):
                    raise abort_type(
                        "visual-course adjacent successor steering escaped "
                        "its fixed envelope"
                    )
                adjacent_crossing_authority = replace(
                    crossing_coast_authority,
                    target_roll_rad=adjacent_roll,
                    yaw_rate_rad_s=adjacent_yaw,
                )
                try:
                    accepted_adjacent = await send_visual(
                        proposal=adjacent_proposal,
                        snapshot=snapshot,
                        target_track=adjacent_track,
                        apply_launch_bootstrap=False,
                        # The exact graph-vetted successor owns roll/yaw
                        # steering only.  The current gate retains its sealed
                        # pitch, thrust, passage proof, and race ownership.
                        top_fov_transition_owned=True,
                        committed_crossing_authority=(
                            adjacent_crossing_authority
                        ),
                        command_deadline_s=min(
                            course_deadline_s,
                            crossing_deadline_s,
                        ),
                        yaw_reference_rad=adjacent_yaw_reference_rad,
                        segment_started_s=segment_started_s,
                        stage=(
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{current_gate_index}/credit-wait-adjacent"
                        ),
                    )
                except RaceActiveBoundaryChangedBeforeWire as race_exc:
                    credited_race = accept_no_wire_race_boundary(
                        race_exc
                    )
                    break
                if type(accepted_adjacent) is _SupersededVisualProposal:
                    continue
                if type(accepted_adjacent) is not _AcceptedVisualCommand:
                    raise abort_type(
                        "visual-course adjacent command outcome is invalid"
                    )
                yaw_reference_rad = adjacent_yaw_reference_rad
                last_planned_token = token
                crossing_wait_adjacent_command_count += 1
                segment["crossing_wait_adjacent_command_count"] = (
                    crossing_wait_adjacent_command_count
                )
                host.recorder.emit(
                    "visual_course_committed_successor_steering_applied",
                    gate_index=current_gate_index,
                    stage=(
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/credit-wait-adjacent"
                    ),
                    camera_token=asdict(token),
                    successor_track_id=credit_wait_reviewed_track_id,
                    sealed_crossing_authority=asdict(
                        crossing_coast_authority
                    ),
                    steering_crossing_authority=asdict(
                        adjacent_crossing_authority
                    ),
                    wire_command=asdict(accepted_adjacent.command),
                    steering_only=True,
                    passage_authority=False,
                    promotion_authority=False,
                    advance_authority=False,
                )
                continue

            if (
                crossing_commitment_deadline_s is not None
                and now >= crossing_commitment_deadline_s
            ):
                # The expired current-gate state no longer owns geometry.
                # A fresh, unique, exact-lineage successor above may still
                # provide steering-only control while authoritative race
                # ingress catches up.  Exact zero remains the fallback when
                # that independently bounded authority is unavailable; it
                # never fabricates passage or promotion.
                last_planned_token = token
                await send_zero(
                    (
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/credit-wait-zero"
                    ),
                    now - segment_started_s,
                    yaw_reference_rad=yaw_reference_rad,
                )
                segment["crossing_wait_zero_command_count"] = int(
                    segment["crossing_wait_zero_command_count"]
                ) + 1
                continue

            try:
                coast_command = await send_censored_passage_coast(
                    snapshot=snapshot,
                    authority=crossing_coast_authority,
                    yaw_reference_rad=yaw_reference_rad,
                    segment_started_s=segment_started_s,
                    stage=(
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/credit-wait"
                    ),
                    command_deadline_s=min(
                        course_deadline_s,
                        crossing_deadline_s,
                    ),
                )
            except RaceActiveBoundaryChangedBeforeWire as race_exc:
                credited_race = accept_no_wire_race_boundary(race_exc)
                break
            if coast_command is None:
                continue
            last_planned_token = token
            crossing_wait_coast_command_count += 1
            segment["crossing_wait_coast_command_count"] = (
                crossing_wait_coast_command_count
            )

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
            return finish_from_authoritative_status(
                credited_race,
                phase=f"gate {current_gate_index} terminal acceptance",
            )

        lifecycle = CourseLifecycle.PROMOTE_REACQUIRE
        segment["lifecycle"] = lifecycle.value
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
                approach_command_count
                + passage_command_count
                + crossing_wait_coast_command_count
                + crossing_wait_adjacent_command_count
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
            "crossing_wait_coast_command_count": int(
                segment["crossing_wait_coast_command_count"]
            ),
            "crossing_wait_adjacent_command_count": int(
                segment["crossing_wait_adjacent_command_count"]
            ),
            "crossing_wait_adjacent_track_id": (
                credit_wait_reviewed_track_id
            ),
            "post_transition_zero_command_count": 0,
            "post_transition_hold_command_count": 0,
            "post_transition_successor_steering_command_count": 0,
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

        admission_reviewed_track_id = (
            passage_admission.preview_track_id
            if (
                type(passage_admission)
                is VisualApproachPassageAdmission
                and type(passage_admission.preview_track_id) is str
                and passage_admission.preview_track_id
            )
            else None
        )
        requested_promoted_track_id = (
            credit_wait_reviewed_track_id
            or admission_reviewed_track_id
            or crossing_reviewed_track_id
        )
        if (
            type(requested_promoted_track_id) is not str
            or not requested_promoted_track_id
        ):
            raise abort_type(
                "visual-course nonterminal transition lacks its reviewed "
                "next-track identity"
            )
        # A sole graph-vetted adjacent observed during the physical
        # gate-plane gap is the freshest local successor identity.  It may
        # replace the pre-clipping preview for the promotion request, but only
        # after authoritative race credit below; its pre-credit proposal has
        # steering-only, no-advance authority.
        fresh_deadline_s = min(
            course_deadline_s,
            float(runtime.monotonic())
            + limits.post_credit_fresh_frame_timeout_s,
        )
        course_handoff: _ConfirmedCourseHandoff
        advance_outcome = host._confirm_visual_course_advance(
            from_gate_index=current_gate_index,
            to_gate_index=current_gate_index + 1,
            race_status=credited_race,
            reviewed_track_id=requested_promoted_track_id,
        )
        post_credit_successor_steering_active = False
        post_credit_dynamic_handoff_active = False
        if type(advance_outcome) is CreditedUnboundGateAdvance:
            unbound_advance = advance_outcome
            if (
                unbound_advance.from_gate_index
                != current_gate_index
                or unbound_advance.to_gate_index
                != current_gate_index + 1
                or unbound_advance.retired_track_id
                != current_track_id
                or unbound_advance.reviewed_track_id
                != requested_promoted_track_id
                or unbound_advance.race_status != credited_race
            ):
                raise abort_type(
                    "visual-course credited-unbound transition is incomplete"
                )
            dynamic_controller = runtime.dynamic_controller
            if type(dynamic_controller) is DynamicVisualCourseSession:
                activation_ns = runtime.perf_counter_ns()
                if type(activation_ns) is not int or activation_ns < 0:
                    raise abort_type(
                        "visual-course successor-steering activation clock "
                        "is invalid"
                    )
                try:
                    steering_activation = (
                        dynamic_controller
                        .activate_post_credit_successor_steering(
                            credited_race,
                            from_gate_index=current_gate_index,
                            reviewed_track_id=(
                                requested_promoted_track_id
                            ),
                            activation_monotonic_ns=activation_ns,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course could not activate credited successor "
                        f"steering: {exc}"
                    ) from exc
                if not isinstance(steering_activation, Mapping):
                    raise abort_type(
                        "visual-course successor-steering activation evidence "
                        "is invalid"
                    )
                post_credit_dynamic_handoff_active = True
                steering_available = steering_activation.get(
                    "steering_available"
                )
                if (
                    type(steering_available) is not bool
                    or steering_activation.get("passage_authority")
                    is not False
                    or steering_activation.get("advance_authority")
                    is not False
                ):
                    raise abort_type(
                        "visual-course successor handoff evidence grants "
                        "invalid authority"
                    )
                post_credit_successor_steering_active = steering_available
                transition_summary["successor_handoff_activation"] = dict(
                    steering_activation
                )
                if post_credit_successor_steering_active:
                    transition_summary[
                        "successor_steering_activation"
                    ] = dict(steering_activation)
                host.recorder.emit(
                    (
                        "visual_course_dynamic_successor_steering_activated"
                        if post_credit_successor_steering_active
                        else (
                            "visual_course_dynamic_successor_steering_"
                            "unavailable"
                        )
                    ),
                    **dict(steering_activation),
                )
            transition_summary.update(
                {
                    "promotion_mode": "credited_unbound",
                    "reviewed_track_id": requested_promoted_track_id,
                    "credit_consumed_without_visual_current": True,
                }
            )
            segment["outcome"] = "credited_unbound"
            refresh_live_summary()
            latest_reacquisition_refusal: Optional[str] = None
            reacquisition: Optional[ConfirmedGateReacquisition] = None

            while reacquisition is None:
                _assert_course_attitude_state(
                    host,
                    yaw_reference_rad=yaw_reference_rad,
                    limits=limits,
                    yaw_profile=runtime.yaw_profile,
                    abort_type=abort_type,
                    phase=(
                        f"gate {unbound_advance.to_gate_index} "
                        "credited-unbound reacquisition"
                    ),
                )
                candidate = host._try_visual_reacquired_current()
                if type(candidate) is ConfirmedGateReacquisition:
                    reacquisition = candidate
                    break
                if type(candidate) is not GateReacquisitionPending:
                    raise abort_type(
                        "visual-course reacquisition returned an invalid "
                        "lifecycle outcome"
                    )
                latest_reacquisition_refusal = candidate.reason

                now = await pace_tick()
                if now >= fresh_deadline_s:
                    raise abort_type(
                        "visual-course credited gate lacks a bounded fresh "
                        "visual reacquisition"
                        + (
                            ""
                            if latest_reacquisition_refusal is None
                            else f": {latest_reacquisition_refusal}"
                        )
                    )
                host._sample()
                pad_contact = initial_pad_contact_authority()
                host._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=pad_contact,
                    enforce_benign_pad_budget=True,
                )
                race = host._visual_race_status_ref()
                if race.race_finished:
                    return finish_from_authoritative_status(
                        race,
                        phase=(
                            f"gate {unbound_advance.to_gate_index} "
                            "credited-unbound terminal acceptance"
                        ),
                    )
                if (
                    race.active_gate_index
                    != unbound_advance.to_gate_index
                ):
                    raise abort_type(
                        "visual-course race boundary changed during "
                        "credited-unbound reacquisition"
                    )
                if runtime.dynamic_controller is not None:
                    post_credit_successor_steering_active = (
                        await send_continuity_hold(
                        (
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{unbound_advance.to_gate_index}/"
                            "credited-unbound-hold"
                        ),
                        float(runtime.monotonic()) - segment_started_s,
                        yaw_reference_rad=yaw_reference_rad,
                        successor_steering=(
                            post_credit_successor_steering_active
                        ),
                        )
                    )
                    segment["post_credit_hold_command_count"] = int(
                        segment["post_credit_hold_command_count"]
                    ) + 1
                    transition_summary[
                        "post_transition_hold_command_count"
                    ] = int(
                        transition_summary[
                            "post_transition_hold_command_count"
                        ]
                    ) + 1
                    if post_credit_successor_steering_active:
                        segment[
                            "post_credit_successor_steering_command_count"
                        ] = int(
                            segment[
                                "post_credit_successor_steering_command_count"
                            ]
                        ) + 1
                        transition_summary[
                            "post_transition_successor_steering_command_count"
                        ] = int(
                            transition_summary[
                                "post_transition_successor_steering_command_count"
                            ]
                        ) + 1
                        transition_summary[
                            "post_transition_navigation_command_count"
                        ] = int(
                            transition_summary[
                                "post_transition_navigation_command_count"
                            ]
                        ) + 1
                else:
                    await send_zero(
                        (
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{unbound_advance.to_gate_index}/"
                            "credited-unbound-zero"
                        ),
                        float(runtime.monotonic()) - segment_started_s,
                        yaw_reference_rad=yaw_reference_rad,
                    )
                    segment["post_credit_zero_command_count"] = int(
                        segment["post_credit_zero_command_count"]
                    ) + 1
                    transition_summary[
                        "post_transition_zero_command_count"
                    ] = int(
                        transition_summary[
                            "post_transition_zero_command_count"
                        ]
                    ) + 1

            assert reacquisition is not None
            if (
                reacquisition.credited_advance != unbound_advance
                or reacquisition.gate_index
                != unbound_advance.to_gate_index
                or reacquisition.reacquired_track_id == current_track_id
                or reacquisition.cross_gap_identity_claimed
                or reacquisition.history_length_at_binding <= 0
            ):
                raise abort_type(
                    "visual-course fresh reacquisition proof is incomplete"
                )
            if (
                post_credit_dynamic_handoff_active
                and reacquisition.reacquired_track_id
                != requested_promoted_track_id
            ):
                dynamic_controller = runtime.dynamic_controller
                assert (
                    type(dynamic_controller)
                    is DynamicVisualCourseSession
                )
                try:
                    dynamic_rebind = (
                        dynamic_controller.rebind_confirmed_reacquisition(
                            reacquisition,
                            host.visual_tracker,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course could not rebind graph-proven dynamic "
                        f"reacquisition: {exc}"
                    ) from exc
                if not isinstance(dynamic_rebind, Mapping):
                    raise abort_type(
                        "visual-course dynamic reacquisition evidence is "
                        "invalid"
                    )
                rebound_steering_available = dynamic_rebind.get(
                    "steering_available"
                )
                if type(rebound_steering_available) is not bool:
                    raise abort_type(
                        "visual-course dynamic reacquisition steering "
                        "availability is invalid"
                    )
                if (
                    dynamic_rebind.get("steering_only")
                    is not rebound_steering_available
                    or dynamic_rebind.get("passage_authority") is not False
                    or dynamic_rebind.get("advance_authority") is not False
                ):
                    raise abort_type(
                        "visual-course dynamic reacquisition exceeded "
                        "steering-only authority"
                    )
                post_credit_successor_steering_active = (
                    rebound_steering_available
                )
                post_credit_dynamic_handoff_active = (
                    rebound_steering_available
                )
                transition_summary["successor_steering_rebind"] = dict(
                    dynamic_rebind
                )
                host.recorder.emit(
                    "visual_course_dynamic_successor_rebound",
                    **dict(dynamic_rebind),
                )
            course_handoff = _ConfirmedCourseHandoff(
                from_gate_index=unbound_advance.from_gate_index,
                to_gate_index=unbound_advance.to_gate_index,
                retired_track_id=unbound_advance.retired_track_id,
                promoted_track_id=reacquisition.reacquired_track_id,
                race_status=unbound_advance.race_status,
                camera_token_at_credit=(
                    unbound_advance.camera_token_at_credit
                ),
                promoted_history_sha256=reacquisition.history_sha256,
                history_length_before_promotion=(
                    reacquisition.history_length_at_binding
                ),
                history_length_after_promotion=(
                    reacquisition.history_length_at_binding
                ),
                promotion_identity_basis=(
                    "rolling-graph-retained-reviewed-fresh-rebind-v1"
                    if reacquisition.reacquired_track_id
                    == requested_promoted_track_id
                    else "rolling-graph-fresh-cross-id-reacquisition-v1"
                ),
                cross_gap_identity_claimed=False,
            )
            transition_summary.update(
                {
                    "promotion_mode": "fresh_reacquisition",
                    "reacquisition_camera_token": asdict(
                        reacquisition.camera_token_at_binding
                    ),
                    "reacquisition_identity_basis": (
                        reacquisition.identity_basis
                    ),
                }
            )
        elif type(advance_outcome) is ConfirmedGateTransition:
            retained_transition = advance_outcome
            if (
                retained_transition.promoted_track_id
                != requested_promoted_track_id
                or retained_transition.from_gate_index
                != current_gate_index
                or retained_transition.to_gate_index
                != current_gate_index + 1
                or retained_transition.retired_track_id
                != current_track_id
                or retained_transition.promoted_track_id
                == current_track_id
                or retained_transition.history_length_before_promotion
                != retained_transition.history_length_after_promotion
            ):
                raise abort_type(
                    "visual-course retained transition promotion is incomplete"
                )
            dynamic_controller = runtime.dynamic_controller
            if type(dynamic_controller) is DynamicVisualCourseSession:
                activation_ns = runtime.perf_counter_ns()
                if type(activation_ns) is not int or activation_ns < 0:
                    raise abort_type(
                        "visual-course successor-steering activation clock "
                        "is invalid"
                    )
                try:
                    steering_activation = (
                        dynamic_controller
                        .activate_post_credit_successor_steering(
                            credited_race,
                            from_gate_index=current_gate_index,
                            reviewed_track_id=(
                                requested_promoted_track_id
                            ),
                            activation_monotonic_ns=activation_ns,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course could not activate credited successor "
                        f"steering: {exc}"
                    ) from exc
                if not isinstance(steering_activation, Mapping):
                    raise abort_type(
                        "visual-course successor-steering activation evidence "
                        "is invalid"
                    )
                post_credit_dynamic_handoff_active = True
                steering_available = steering_activation.get(
                    "steering_available"
                )
                if (
                    type(steering_available) is not bool
                    or steering_activation.get("passage_authority")
                    is not False
                    or steering_activation.get("advance_authority")
                    is not False
                ):
                    raise abort_type(
                        "visual-course successor handoff evidence grants "
                        "invalid authority"
                    )
                post_credit_successor_steering_active = steering_available
                transition_summary["successor_handoff_activation"] = dict(
                    steering_activation
                )
                if post_credit_successor_steering_active:
                    transition_summary[
                        "successor_steering_activation"
                    ] = dict(steering_activation)
                host.recorder.emit(
                    (
                        "visual_course_dynamic_successor_steering_activated"
                        if post_credit_successor_steering_active
                        else (
                            "visual_course_dynamic_successor_steering_"
                            "unavailable"
                        )
                    ),
                    **dict(steering_activation),
                )
            course_handoff = _ConfirmedCourseHandoff(
                from_gate_index=retained_transition.from_gate_index,
                to_gate_index=retained_transition.to_gate_index,
                retired_track_id=retained_transition.retired_track_id,
                promoted_track_id=retained_transition.promoted_track_id,
                race_status=retained_transition.race_status,
                camera_token_at_credit=(
                    retained_transition.camera_token_at_credit
                ),
                promoted_history_sha256=(
                    retained_transition.promoted_history_sha256
                ),
                history_length_before_promotion=(
                    retained_transition.history_length_before_promotion
                ),
                history_length_after_promotion=(
                    retained_transition.history_length_after_promotion
                ),
                promotion_identity_basis=(
                    "rolling-graph-retained-reviewed-identity-v1"
                ),
                cross_gap_identity_claimed=False,
            )
            transition_summary.update(
                {
                    "promotion_mode": "retained_reviewed_identity",
                    "reviewed_track_id": requested_promoted_track_id,
                    "credit_consumed_without_visual_current": False,
                }
            )
        else:
            raise abort_type(
                "visual-course advance returned an invalid lifecycle outcome"
            )
        transition_summary.update(
            {
                "promotion_confirmed": True,
                "retired_track_id": course_handoff.retired_track_id,
                "promoted_track_id": course_handoff.promoted_track_id,
                "history_length_before_promotion": (
                    course_handoff.history_length_before_promotion
                ),
                "history_length_after_promotion": (
                    course_handoff.history_length_after_promotion
                ),
            }
        )
        segment["outcome"] = "transition_confirmed"

        current_gate_index = course_handoff.to_gate_index
        current_track_id = course_handoff.promoted_track_id
        max_gate_index = max(max_gate_index, current_gate_index)
        refresh_live_summary()
        admitted_recovery_token: Optional[CameraFrameToken] = None
        latest_recovery_refusal: Optional[str] = None

        def evaluate_recovery_candidate(snapshot: Any) -> bool:
            nonlocal admitted_recovery_token
            nonlocal latest_recovery_refusal

            admitted_recovery_token = None
            graph_current_ready = _current_snapshot_ready(
                snapshot,
                gate_index=current_gate_index,
                track_id=current_track_id,
                newer_than=course_handoff.camera_token_at_credit,
                observed_after_ns=(
                    course_handoff.race_status.received_monotonic_ns
                ),
                allow_one_edge_censored=True,
            )
            dynamic_steering_ready = bool(
                type(runtime.dynamic_controller)
                is not DynamicVisualCourseSession
                or _dynamic_current_steering_correction_ready(
                    snapshot,
                    track_id=current_track_id,
                )
            )
            if graph_current_ready and dynamic_steering_ready:
                admitted_recovery_token = snapshot.latest_camera_token
                latest_recovery_refusal = None
                return True
            latest_recovery_refusal = (
                "promoted dynamic current lacks a strictly newer "
                "image-axis steering correction"
                if graph_current_ready and not dynamic_steering_ready
                else (
                    "promoted current lacks a strictly newer observable, "
                    "visible, unambiguous frame"
                )
            )
            return False

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
            if race.race_finished:
                return finish_from_authoritative_status(
                    race,
                    phase=(
                        f"gate {current_gate_index} post-credit "
                        "terminal acceptance"
                    ),
                )
            if race.active_gate_index != current_gate_index:
                raise abort_type(
                    "visual-course race boundary changed during fresh-frame "
                    "handoff"
                )
            snapshot = host.visual_gate_graph.latest_snapshot
            if evaluate_recovery_candidate(snapshot):
                break
            if runtime.dynamic_controller is not None:
                post_credit_successor_steering_active = (
                    await send_continuity_hold(
                    (
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/post-credit-hold"
                    ),
                    float(runtime.monotonic()) - segment_started_s,
                    yaw_reference_rad=yaw_reference_rad,
                    successor_steering=(
                        post_credit_successor_steering_active
                    ),
                    )
                )
                segment["post_credit_hold_command_count"] = int(
                    segment["post_credit_hold_command_count"]
                ) + 1
                transition_summary[
                    "post_transition_hold_command_count"
                ] = int(
                    transition_summary[
                        "post_transition_hold_command_count"
                    ]
                ) + 1
                if post_credit_successor_steering_active:
                    segment[
                        "post_credit_successor_steering_command_count"
                    ] = int(
                        segment[
                            "post_credit_successor_steering_command_count"
                        ]
                    ) + 1
                    transition_summary[
                        "post_transition_successor_steering_command_count"
                    ] = int(
                        transition_summary[
                            "post_transition_successor_steering_command_count"
                        ]
                    ) + 1
                    transition_summary[
                        "post_transition_navigation_command_count"
                    ] = int(
                        transition_summary[
                            "post_transition_navigation_command_count"
                        ]
                    ) + 1
            else:
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
                transition_summary[
                    "post_transition_zero_command_count"
                ] = int(
                    transition_summary[
                        "post_transition_zero_command_count"
                    ]
                ) + 1
        assert admitted_recovery_token is not None
        transition_summary["recovery_admission"] = {
            "admission_kind": "fresh_promoted_current",
            "admitted_frame_token": asdict(admitted_recovery_token),
            "wire_frame_token": None,
            "wire_start_monotonic_ns": None,
            "wire_return_monotonic_ns": None,
            "track_id": current_track_id,
            "promotion_identity_sha256": (
                course_handoff.promoted_history_sha256
            ),
            "promotion_identity_basis": (
                course_handoff.promotion_identity_basis
            ),
            "cross_gap_identity_claimed": (
                course_handoff.cross_gap_identity_claimed
            ),
        }
        pending_post_credit_recovery = _PendingPostCreditRecovery(
            from_gate_index=course_handoff.from_gate_index,
            to_gate_index=course_handoff.to_gate_index,
            track_id=course_handoff.promoted_track_id,
            camera_token_at_credit=(
                course_handoff.camera_token_at_credit
            ),
            admitted_camera_token=admitted_recovery_token,
            deadline_s=fresh_deadline_s,
            successor_steering_available=(
                post_credit_successor_steering_active
            ),
        )
        carry_adjacent_planner = bool(
            crossing_wait_adjacent_command_count > 0
            and credit_wait_adjacent_track_id
            == course_handoff.promoted_track_id
        )
        pending_post_credit_planner = (
            credit_wait_adjacent_planner
            if carry_adjacent_planner
            else None
        )
        pending_post_credit_yaw_reference_rad = (
            yaw_reference_rad
            if carry_adjacent_planner
            else None
        )

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
