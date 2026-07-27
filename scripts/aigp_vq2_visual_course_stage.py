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
APPROACH_TOP_RECOVERY_BASIS = (
    "clean-q-converging-top-censored-approach-v1"
)
APPROACH_INNER_DROPOUT_HOLD_BASIS = (
    "fresh-top-censored-prior-inner-fov-continuity-v1"
)
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
_YAW_PROFILE_ISSUER = object()


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
        inner.geometry_model_id
        != "vq2-visible-inner-quad-lines-v1"
        or inner.covariance_model_id
        != "vq2-visible-aperture-diagonal-v1"
    ):
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
    if (
        getattr(sample, "clipping", None) != FrameEdge.NONE
        or getattr(sample, "center_censored", None) is not False
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
    """Return a same-source two-sigma lower bound on top-edge recovery."""

    elapsed = float(elapsed_s)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("top-FOV edge-rate interval is invalid")
    if current.basis != previous.basis:
        return None
    combined_edge_std = math.sqrt(
        current.top_edge_std_image_down**2
        + previous.top_edge_std_image_down**2
    )
    rate = (
        current.nominal_top_edge_image_down
        - previous.nominal_top_edge_image_down
        - TOP_FOV_INNER_EDGE_SIGMA * combined_edge_std
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
    """Bound a TOP-only fit dropout to the last exact inner/FOV authority."""

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
        or getattr(track, "clipping", None) != FrameEdge.TOP
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
        anchor_sample = history[-2]
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
            or fov_summary.get("active") is not True
            or fov_summary.get("last_track_id") != expected_track_id
            or fov_summary.get("last_raw_top_edge_basis")
            != TOP_FOV_INNER_EDGE_BASIS
            or fov_summary.get("last_camera_token")
            != asdict(anchor_sample.token)
            or type(
                fov_summary.get("last_wire_start_monotonic_ns")
            )
            is not int
        ):
            return None
        maximum_target_pitch = float(
            fov_summary.get("last_protected_target_pitch_rad", math.nan)
        )
        anchor_token = anchor_sample.token
        anchor_observation_ns = anchor_sample.observation_monotonic_ns
        anchor_wire_start_ns = int(
            fov_summary["last_wire_start_monotonic_ns"]
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
    recovering = bool(rate is not None and rate > 0.0)
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
                math.atan(edge.top_edge_image_down * vertical_scale)
                - math.atan(
                    previous_edge.top_edge_image_down * vertical_scale
                )
            ) / elapsed_s
            body_rates = current.body_rates_rad_s
            if type(body_rates) is not tuple or len(body_rates) != 3:
                raise ValueError("top-FOV measured body rates are invalid")
            pitch_rate = float(body_rates[1])
            nonrotational_angle_rate = (
                _top_fov_nonrotational_angle_rate_rad_s(
                    current_top_edge_image_down=edge.top_edge_image_down,
                    previous_top_edge_image_down=(
                        previous_edge.top_edge_image_down
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
    filtered_vertical_rate_down_s: float
    control_vertical_error_image_down: Optional[float]
    control_basis: str
    vertical_censored: bool
    current_aperture_dropout: bool
    held_last_observable_collective: bool


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
) -> _CurrentApertureCollectiveProposal:
    """Allocate collective only from the authoritative current aperture."""

    if (
        type(authoritative_current_track_id) is not str
        or not authoritative_current_track_id
        or state.track_id != authoritative_current_track_id
    ):
        raise ValueError(
            "current-aperture collective authority is invalid"
        )
    if target.track_id == authoritative_current_track_id:
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
    held = bool(
        state.last_hold_reason is not None
        and state.last_observable_thrust is not None
    )
    return _CurrentApertureCollectiveProposal(
        requested_thrust=thrust,
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
            or not all(bearing_rate_qualified)
            or not scale_rate_qualified
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
    if (
        abs(roll) > limits.max_abs_measured_roll_rad
        or pitch < limits.min_measured_pitch_rad
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
    ) -> None:
        """Hold the last dynamic target across one bounded handoff gap."""

        nonlocal last_command_send_s
        nonlocal consecutive_superseded_proposals

        dynamic_controller = runtime.dynamic_controller
        profile = runtime.yaw_profile
        if dynamic_controller is None or profile is None:
            raise abort_type(
                "visual-course continuity hold lacks dynamic/yaw authority"
            )
        await host._wait_for_next_flight_command_slot()
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
        try:
            authority = dynamic_controller.continuity_hold_authority(
                now_monotonic_ns=proposal_ns,
                maximum_age_s=profile.control_hold_horizon_s,
            )
        except (TypeError, ValueError) as exc:
            raise abort_type(
                f"visual-course continuity hold expired: {exc}"
            ) from exc
        if not isinstance(authority, Mapping):
            raise abort_type(
                "visual-course continuity hold authority is invalid"
            )
        target_roll_rad = float(authority["target_roll_rad"])
        target_pitch_rad = float(authority["target_pitch_rad"])
        requested_yaw = float(authority["yaw_rate_rad_s"])
        thrust = float(authority["thrust"])
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
                launch_thrust_override=True,
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
            or command.thrust != thrust
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
                    thrust_slew_override=True,
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
        host.recorder.emit(
            "visual_course_dynamic_command",
            **dict(dynamic_evidence),
        )
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
                    prior_target_pitch_rad = (
                        fov_summary["last_protected_target_pitch_rad"]
                        if fov_summary[
                            "last_protected_target_pitch_rad"
                        ]
                        is not None
                        else (
                            top_fov_observation.previous_target_pitch_rad
                            if (
                                top_fov_observation
                                .previous_target_pitch_rad
                                is not None
                            )
                            else top_fov_observation.capture_pitch_rad
                        )
                    )
                    top_fov_proposal = _propose_top_fov_pitch_reference(
                        capture_pitch_rad=(
                            top_fov_observation.capture_pitch_rad
                        ),
                        raw_top_edge_image_down=(
                            top_fov_observation.raw_top_edge_image_down
                        ),
                        raw_top_edge_rate_down_s=(
                            top_fov_observation
                            .raw_top_edge_rate_down_s
                        ),
                        requested_target_pitch_rad=target_pitch_rad,
                        prior_target_pitch_rad=prior_target_pitch_rad,
                        vertical_angle_scale_rad=(
                            top_fov_observation.vertical_angle_scale_rad
                        ),
                        active_before=bool(fov_summary["active"]),
                        raw_top_edge_nonrotational_angle_rate_rad_s=(
                            top_fov_observation
                            .raw_top_edge_nonrotational_angle_rate_rad_s
                        ),
                        prediction_horizon_s=(
                            top_fov_observation.pitch_response_delay_s
                        ),
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise abort_type(
                        "visual-course top-FOV pitch guidance refused: "
                        f"{exc}"
                    ) from exc
                target_pitch_rad = top_fov_proposal.protected_target_pitch_rad
                if launch_evidence is not None:
                    launch_evidence["target_pitch_rad_before_top_fov"] = (
                        launch_evidence["target_pitch_rad"]
                    )
                    launch_evidence["target_pitch_rad"] = target_pitch_rad
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
            host.recorder.emit(
                "visual_course_dynamic_command",
                **dict(dynamic_evidence),
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
                else "visual_course_censored_passage_coast_command"
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
        ) -> float:
            """Return the last clean aperture request for bounded coast."""

            fallback = float(fallback_wire_thrust)
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
        crossing_wait_coast_command_count = 0
        crossing_wait_adjacent_command_count = 0
        credit_wait_adjacent_planner: Optional[Any] = None
        credit_wait_adjacent_track_id: Optional[str] = None
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
            "crossing_wait_zero_command_count": 0,
            "crossing_wait_coast_command_count": 0,
            "crossing_wait_adjacent_command_count": 0,
            "censored_passage_coast_fresh_frame_count": 0,
            "censored_passage_coast_command_count": 0,
            "censored_passage_coast": None,
            "approach_top_recovery_fresh_frame_count": 0,
            "approach_top_recovery_command_count": 0,
            "approach_top_recovery": None,
            "approach_inner_dropout_hold_command_count": 0,
            "approach_inner_dropout_hold": None,
            "post_credit_zero_command_count": 0,
            "post_credit_hold_command_count": 0,
            "recovery_navigation_command_count": 0,
            "recovery_clean_command_count": 0,
            "recovery_one_edge_command_count": 0,
            "recovery_zero_command_count": 0,
            "recovery_support_command_count": 0,
            "passage_authority_enabled": False,
            "passage_admission": None,
            "lifecycle": lifecycle.value,
            "near_plane_evidence_frame_count": 0,
            "near_plane_latch": None,
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
                "last_track_id": None,
                "last_camera_token": None,
                "last_wire_start_monotonic_ns": None,
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
                    raise abort_type(
                        "visual-course race credit arrived without credible "
                        "passage evidence"
                    )
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
                        or abs(
                            float(hold["target_pitch_rad"])
                            - (
                                dropout_authority
                                .maximum_target_pitch_rad
                            )
                        )
                        > 1e-12
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
                    censored_passage_coast_last_observed_token
                    or last_clean_passage_token
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
                censored_coast_eligible = bool(
                    type(exc)
                    is VisualApproachCurrentGeometryUnavailable
                    and mode is VisualApproachMode.PASSAGE
                    and type(passage_admission)
                    is VisualApproachPassageAdmission
                    and crossing_anchor is not None
                    and crossing_coast_authority is not None
                    and previous_visible_token is not None
                    and measurement_mode
                    is LatchedMeasurementMode.COAST
                )
                if censored_coast_eligible:
                    segment["near_plane_measurement_mode"] = (
                        LatchedMeasurementMode.COAST.value
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
                            "requested_thrust": (
                                crossing_coast_authority.requested_thrust
                            ),
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
                                    censored_passage_coast_started_s
                                    + limits
                                    .censored_passage_coast_max_duration_s
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
                raise abort_type(
                    "visual-course censored passage coast returned to "
                    "uncensored geometry"
                )
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
                        # Authoritative promotion has already retired the
                        # predecessor aperture.  The bounded recovery
                        # lifecycle owns observability until the promoted
                        # current returns through clean geometry.
                        top_fov_transition_owned=True,
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
                    and proposal.servo_output.corridor_frames
                    >= (
                        host.visual_config.servo
                        .required_corridor_frames
                    )
                    and proposal.servo_output.brake_reason == "aligning"
                    and not accepted.yaw_soft_stop_zeroed
                )
                if recovery_completed:
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
                                    accepted.command.thrust
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
                                required_corridor_frames=(
                                    host.visual_config.servo
                                    .required_corridor_frames
                                ),
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
                    if candidate_latch is not None:
                        near_plane_latch = candidate_latch
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
                        near_plane_latch is not None
                        and near_plane_latch.basis
                        == DYNAMIC_NEAR_PLANE_LATCH_BASIS
                    ):
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
                                        coast_thrust
                                    )
                                ),
                            )
                        )
                        anchor = near_plane_latch.anchor_sample
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
                            "advance_command_count": (
                                advance_command_count
                            ),
                            "log_scale": anchor.log_scale,
                            "log_scale_rate_s": (
                                anchor.log_scale_rate_s
                            ),
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
                            "normalized_x_std": (
                                anchor.normalized_x_std
                            ),
                            "normalized_y_std": (
                                anchor.normalized_y_std
                            ),
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
                                    * anchor.predicted_crossing_x_std_norm
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
                                anchor
                                .post_governor_contact_budget_s
                            ),
                            "command": asdict(command),
                            "current_only_crossing_coast_thrust": (
                                coast_thrust
                            ),
                        }
                        last_clean_passage_token = (
                            near_plane_latch.anchor_camera_token
                        )
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
                                    crossing_coast_thrust
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
            # guidance; low-confidence or contended relationship failures do
            # not gain command authority.
            if (
                credit_wait_adjacent_planner is None
                and getattr(
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
            ):
                credit_wait_adjacent_track_id = (
                    adjacent_candidates[0].track_id
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
            if credit_wait_adjacent_planner is not None:
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
                try:
                    accepted_adjacent = await send_visual(
                        proposal=adjacent_proposal,
                        snapshot=snapshot,
                        target_track=adjacent_track,
                        apply_launch_bootstrap=False,
                        # Once the near plane is latched and the old aperture
                        # has disappeared, apply the already flight-proved
                        # pitch response to fresh successor braking.  This
                        # moves closure control before race-packet latency
                        # without changing roll, yaw, thrust, or any bound.
                        intercept_response_authority=1.0,
                        top_fov_transition_owned=True,
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
            "post_transition_zero_command_count": 0,
            "post_transition_hold_command_count": 0,
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
                    await send_continuity_hold(
                        (
                            f"{VISUAL_COURSE_STAGE}/gate"
                            f"{unbound_advance.to_gate_index}/"
                            "credited-unbound-hold"
                        ),
                        float(runtime.monotonic()) - segment_started_s,
                        yaw_reference_rad=yaw_reference_rad,
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
            if _current_snapshot_ready(
                snapshot,
                gate_index=current_gate_index,
                track_id=current_track_id,
                newer_than=course_handoff.camera_token_at_credit,
                observed_after_ns=(
                    course_handoff.race_status.received_monotonic_ns
                ),
                allow_one_edge_censored=True,
            ):
                admitted_recovery_token = snapshot.latest_camera_token
                latest_recovery_refusal = None
                return True
            latest_recovery_refusal = (
                "promoted current lacks a strictly newer observable, visible, "
                "unambiguous frame"
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
                await send_continuity_hold(
                    (
                        f"{VISUAL_COURSE_STAGE}/gate"
                        f"{current_gate_index}/post-credit-hold"
                    ),
                    float(runtime.monotonic()) - segment_started_s,
                    yaw_reference_rad=yaw_reference_rad,
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
