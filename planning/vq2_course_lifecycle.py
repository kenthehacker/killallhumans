"""Pure gate-generic lifecycle evidence for the build-3385 visual course.

The production inputs represented here are image-space tracker facts and an
exact accepted command-wire publication.  This module does not consume pose,
odometry, metric gate geometry, or simulator gate maps.  It also does not send
commands or declare passage: only authoritative race status may do that.

Near-plane evidence intentionally counts distinct accepted camera
publications, not controller-mode or command-count history.  Once latched,
clipping is an axis-censoring measurement mode.  A censored axis contributes
no center or rate evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import CameraFrameToken, VisualTrackRole
from planning.vq2_visual_servo import (
    PREPASS_CURRENT_MAX_ABS_X_NORM,
    PREPASS_CURRENT_MAX_ABS_Y_NORM,
    PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S,
    PREPASS_CURRENT_MIN_LOG_SCALE_RATE_S,
    PREPASS_CURRENT_PROJECTION_HORIZON_S,
)


NEAR_PLANE_LATCH_BASIS = "centered-expanding-accepted-wire-history-v1"
DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS = (
    "imu-derotated-aperture-quotient-swept-envelope-v2"
)
DYNAMIC_NEAR_PLANE_LATCH_BASIS = (
    "imu-derotated-aperture-quotient-accepted-wire-history-v2"
)
RAW_NEAR_PLANE_GEOMETRY_BASIS = "raw-image-current-aperture-v1"
DYNAMIC_CROSSING_PREDICTION_MAX_HORIZON_S = 1.20


class CourseLifecycle(str, Enum):
    """One gate-agnostic course lifecycle, repeated until race finish."""

    APPROACH = "approach"
    PASSAGE_ARMED = "passage_armed"
    NEAR_PLANE_LATCHED = "near_plane_latched"
    CREDIT_WAIT = "credit_wait"
    PROMOTE_REACQUIRE = "promote_reacquire"


class LatchedMeasurementMode(str, Enum):
    """Permitted response to a fresh observation after a near-plane latch."""

    COAST = "coast"
    CREDIT_WAIT = "credit_wait"
    UNSAFE = "unsafe"


class PostCreditMeasurementMode(str, Enum):
    """Permitted response while a promoted gate is being reacquired."""

    CLEAN = "clean"
    ONE_EDGE_CENSORED = "one_edge_censored"
    REACQUIRE = "reacquire"
    UNSAFE = "unsafe"


def classify_post_credit_measurement(
    snapshot: object,
    *,
    gate_index: int,
    track_id: str,
    previous_camera_token: CameraFrameToken,
    last_track_token: CameraFrameToken,
) -> PostCreditMeasurementMode:
    """Classify one newer promoted-current camera publication."""

    token = getattr(snapshot, "latest_camera_token", None)
    track = getattr(snapshot, "current_track", None)
    clipping = getattr(track, "clipping", None)
    ambiguous = getattr(track, "ambiguous", None)
    track_role = getattr(track, "role", None)
    if (
        type(gate_index) is not int
        or gate_index < 0
        or type(track_id) is not str
        or not track_id
        or not _token_strictly_newer(token, previous_camera_token)
        or getattr(snapshot, "current_gate_index", None) != gate_index
        or getattr(snapshot, "current_track_id", None) != track_id
        or track is None
        or getattr(track, "track_id", None) != track_id
        or type(ambiguous) is not bool
        or type(clipping) is not FrameEdge
    ):
        return PostCreditMeasurementMode.UNSAFE

    if not getattr(track, "visible", False):
        latest_track_token = getattr(track, "latest_token", None)
        retained_current_role = bool(
            (track_role is VisualTrackRole.CURRENT and not ambiguous)
            or (
                track_role is VisualTrackRole.AMBIGUOUS
                and ambiguous
            )
        )
        retained_loss = bool(
            getattr(track, "missed_frame_count", 0) > 0
            and retained_current_role
            and getattr(track, "authoritative_gate_index", None) == gate_index
            and _token_not_older(latest_track_token, last_track_token)
            and _token_not_older(token, latest_track_token)
        )
        return (
            PostCreditMeasurementMode.REACQUIRE
            if retained_loss
            else PostCreditMeasurementMode.UNSAFE
        )

    if ambiguous or track_role is VisualTrackRole.AMBIGUOUS:
        return PostCreditMeasurementMode.UNSAFE
    if (
        getattr(track, "missed_frame_count", -1) != 0
        or getattr(track, "latest_token", None) != token
        or track_role is not VisualTrackRole.CURRENT
        or getattr(track, "authoritative_gate_index", None) != gate_index
        or getattr(snapshot, "authority_usable", False) is not True
    ):
        return PostCreditMeasurementMode.UNSAFE
    if clipping == FrameEdge.NONE and not getattr(
        track,
        "center_censored",
        True,
    ):
        return PostCreditMeasurementMode.CLEAN
    if clipping in {
        FrameEdge.LEFT,
        FrameEdge.TOP,
        FrameEdge.RIGHT,
        FrameEdge.BOTTOM,
    }:
        center = getattr(track, "center_norm", None)
        velocity = getattr(track, "center_velocity_norm_s", None)
        observable_axis = (
            0
            if clipping in {FrameEdge.TOP, FrameEdge.BOTTOM}
            else 1
        )
        if (
            type(center) is not tuple
            or len(center) != 2
            or type(velocity) is not tuple
            or len(velocity) != 2
            or not _finite(center[observable_axis])
            or abs(float(center[observable_axis])) > 1.0
            or not _finite(velocity[observable_axis])
        ):
            # A censored dimension owns no geometry.  The remaining dimension
            # must still be an actual finite image coordinate; +/-1 is the
            # normalized image boundary, not a navigation tuning threshold.
            return PostCreditMeasurementMode.UNSAFE
        return PostCreditMeasurementMode.ONE_EDGE_CENSORED
    return PostCreditMeasurementMode.REACQUIRE


def _finite(value: object) -> bool:
    return bool(
        type(value) in {int, float} and math.isfinite(float(value))
    )


def _live_token(token: object) -> bool:
    return bool(
        type(token) is CameraFrameToken
        and type(token.stream_id) is str
        and bool(token.stream_id)
        and type(token.publication_sequence) is int
        and token.publication_sequence > 0
    )


def _same_camera_epoch(
    current: CameraFrameToken,
    previous: CameraFrameToken,
) -> bool:
    return bool(
        _live_token(current)
        and _live_token(previous)
        and current.stream_id == previous.stream_id
        and current.generation == previous.generation
    )


def _token_strictly_newer(
    current: CameraFrameToken,
    previous: CameraFrameToken,
) -> bool:
    return bool(
        _same_camera_epoch(current, previous)
        and current.publication_sequence > previous.publication_sequence
    )


def _token_not_older(
    current: CameraFrameToken,
    previous: CameraFrameToken,
) -> bool:
    return bool(
        _same_camera_epoch(current, previous)
        and current.publication_sequence >= previous.publication_sequence
    )


def _confidence_threshold(value: object, name: str) -> float:
    if not _finite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and inside [0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class NearPlaneWireSample:
    """One fresh tracker observation proved accepted at the command wire.

    The command fields are the final bounded wire command.  No unrecorded
    ``VisualServoOutput`` state, controller phase, or advance classification is
    required.
    """

    gate_index: int
    track_id: str
    camera_token: CameraFrameToken
    wire_camera_token: CameraFrameToken
    observation_monotonic_ns: int
    publication_monotonic_ns: int
    wire_start_monotonic_ns: int
    wire_return_monotonic_ns: int
    wire_race_gate_index: int
    publication_pinned_through_transport_return: bool
    normalized_x: float
    normalized_y_down: float
    normalized_x_rate_s: float
    normalized_y_rate_down_s: float
    log_scale: float
    log_scale_rate_s: float
    confidence: float
    association_confidence: float
    clipping: FrameEdge
    center_censored: bool
    ambiguous: bool
    command_roll_rate: float
    command_pitch_rate: float
    command_yaw_rate: float
    command_thrust: float
    geometry_basis: str = RAW_NEAR_PLANE_GEOMETRY_BASIS
    normalized_x_std: float = 0.0
    normalized_y_std: float = 0.0
    log_scale_std: float = 0.0
    crossing_prediction_horizon_s: Optional[float] = None
    predicted_crossing_x_norm: Optional[float] = None
    predicted_crossing_y_down_norm: Optional[float] = None
    predicted_crossing_x_std_norm: Optional[float] = None
    predicted_crossing_y_std_norm: Optional[float] = None
    crossing_allowance_x_norm: Optional[float] = None
    crossing_allowance_y_norm: Optional[float] = None
    crossing_swept_x_occupancy_norm: Optional[float] = None
    crossing_swept_y_occupancy_norm: Optional[float] = None
    current_crossing_x_q: Optional[float] = None
    current_crossing_y_q: Optional[float] = None
    crossing_x_q_rate_s: Optional[float] = None
    crossing_y_q_rate_s: Optional[float] = None
    post_governor_contact_budget_s: Optional[float] = None

    def __post_init__(self) -> None:
        if type(self.gate_index) is not int or self.gate_index < 0:
            raise ValueError("near-plane sample gate index is invalid")
        if type(self.track_id) is not str or not self.track_id:
            raise ValueError("near-plane sample track id is invalid")
        if not _live_token(self.camera_token) or not _live_token(
            self.wire_camera_token
        ):
            raise ValueError("near-plane sample requires exact live tokens")
        if self.wire_camera_token != self.camera_token:
            raise ValueError(
                "near-plane observation and accepted wire tokens differ"
            )
        for name in (
            "observation_monotonic_ns",
            "publication_monotonic_ns",
            "wire_start_monotonic_ns",
            "wire_return_monotonic_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"near-plane sample {name} is invalid")
        if not (
            self.observation_monotonic_ns
            <= self.publication_monotonic_ns
            <= self.wire_start_monotonic_ns
            <= self.wire_return_monotonic_ns
        ):
            raise ValueError("near-plane sample timing order is invalid")
        if (
            type(self.wire_race_gate_index) is not int
            or self.wire_race_gate_index != self.gate_index
        ):
            raise ValueError(
                "near-plane wire race authority differs from its gate"
            )
        if self.publication_pinned_through_transport_return is not True:
            raise ValueError(
                "near-plane wire did not pin its camera publication"
            )
        numeric = (
            self.normalized_x,
            self.normalized_y_down,
            self.normalized_x_rate_s,
            self.normalized_y_rate_down_s,
            self.log_scale,
            self.log_scale_rate_s,
            self.confidence,
            self.association_confidence,
            self.command_roll_rate,
            self.command_pitch_rate,
            self.command_yaw_rate,
            self.command_thrust,
        )
        if not all(_finite(value) for value in numeric):
            raise ValueError("near-plane sample fields must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("near-plane track confidence is outside [0, 1]")
        if not 0.0 <= float(self.association_confidence) <= 1.0:
            raise ValueError(
                "near-plane association confidence is outside [0, 1]"
            )
        if type(self.clipping) is not FrameEdge:
            raise TypeError("near-plane clipping must be an exact FrameEdge")
        if (
            type(self.center_censored) is not bool
            or type(self.ambiguous) is not bool
        ):
            raise TypeError(
                "near-plane censoring and ambiguity flags must be exact"
            )
        if self.geometry_basis not in {
            RAW_NEAR_PLANE_GEOMETRY_BASIS,
            DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
        }:
            raise ValueError("near-plane sample geometry basis is invalid")
        for name in (
            "normalized_x_std",
            "normalized_y_std",
            "log_scale_std",
        ):
            value = getattr(self, name)
            if not _finite(value) or float(value) < 0.0:
                raise ValueError(
                    f"near-plane sample {name} must be finite and nonnegative"
                )
        crossing_fields = (
            self.crossing_prediction_horizon_s,
            self.predicted_crossing_x_norm,
            self.predicted_crossing_y_down_norm,
            self.predicted_crossing_x_std_norm,
            self.predicted_crossing_y_std_norm,
            self.crossing_allowance_x_norm,
            self.crossing_allowance_y_norm,
            self.crossing_swept_x_occupancy_norm,
            self.crossing_swept_y_occupancy_norm,
            self.current_crossing_x_q,
            self.current_crossing_y_q,
            self.crossing_x_q_rate_s,
            self.crossing_y_q_rate_s,
            self.post_governor_contact_budget_s,
        )
        if self.geometry_basis == DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS:
            if any(value is None for value in crossing_fields):
                raise ValueError(
                    "dynamic near-plane sample lacks crossing prediction"
                )
            assert self.crossing_prediction_horizon_s is not None
            if not _finite(self.crossing_prediction_horizon_s) or not (
                0.0
                <= float(self.crossing_prediction_horizon_s)
                <= DYNAMIC_CROSSING_PREDICTION_MAX_HORIZON_S
            ):
                raise ValueError(
                    "dynamic near-plane crossing horizon is outside bounds"
                )
            for name in (
                "predicted_crossing_x_norm",
                "predicted_crossing_y_down_norm",
                "current_crossing_x_q",
                "current_crossing_y_q",
                "crossing_x_q_rate_s",
                "crossing_y_q_rate_s",
                "post_governor_contact_budget_s",
            ):
                if not _finite(getattr(self, name)):
                    raise ValueError(
                        f"dynamic near-plane sample {name} must be finite"
                    )
            for name in (
                "predicted_crossing_x_std_norm",
                "predicted_crossing_y_std_norm",
                "crossing_allowance_x_norm",
                "crossing_allowance_y_norm",
                "crossing_swept_x_occupancy_norm",
                "crossing_swept_y_occupancy_norm",
            ):
                value = getattr(self, name)
                if not _finite(value) or float(value) < 0.0:
                    raise ValueError(
                        f"dynamic near-plane sample {name} must be finite "
                        "and nonnegative"
                    )
        elif any(value is not None for value in crossing_fields):
            raise ValueError(
                "raw near-plane sample cannot carry dynamic crossing prediction"
            )

    @property
    def apparent_scale(self) -> float:
        return math.exp(float(self.log_scale))

    @property
    def command(self) -> tuple[float, float, float, float]:
        """Return the exact final wire command in roll/pitch/yaw/thrust order."""

        return (
            float(self.command_roll_rate),
            float(self.command_pitch_rate),
            float(self.command_yaw_rate),
            float(self.command_thrust),
        )


@dataclass(frozen=True, slots=True)
class NearPlaneEvidence:
    """A homogeneous, strictly advancing expansion history."""

    samples: tuple[NearPlaneWireSample, ...] = ()
    last_observed_sample: Optional[NearPlaneWireSample] = None

    def __post_init__(self) -> None:
        if type(self.samples) is not tuple:
            raise TypeError("near-plane evidence samples must be an exact tuple")
        if any(type(sample) is not NearPlaneWireSample for sample in self.samples):
            raise TypeError(
                "near-plane evidence must contain exact wire samples"
            )
        if (
            self.last_observed_sample is not None
            and type(self.last_observed_sample) is not NearPlaneWireSample
        ):
            raise TypeError(
                "near-plane continuity must contain an exact wire sample"
            )
        if not self.samples:
            if self.last_observed_sample is not None:
                raise ValueError(
                    "near-plane continuity lacks qualified evidence"
                )
            return
        first = self.samples[0]
        previous = first
        if float(first.log_scale_rate_s) <= 0.0:
            raise ValueError("near-plane evidence must be positively expanding")
        for sample in self.samples[1:]:
            if (
                sample.gate_index != first.gate_index
                or sample.track_id != first.track_id
                or sample.geometry_basis != first.geometry_basis
                or not _same_camera_epoch(sample.camera_token, first.camera_token)
            ):
                raise ValueError(
                    "near-plane evidence crossed its gate, track, or epoch"
                )
            if (
                not _token_strictly_newer(
                    sample.camera_token,
                    previous.camera_token,
                )
                or sample.observation_monotonic_ns
                <= previous.observation_monotonic_ns
                or sample.publication_monotonic_ns
                <= previous.publication_monotonic_ns
                or sample.wire_start_monotonic_ns
                <= previous.wire_start_monotonic_ns
                or sample.wire_return_monotonic_ns
                <= previous.wire_return_monotonic_ns
            ):
                raise ValueError(
                    "near-plane evidence did not strictly advance"
                )
            if (
                float(sample.log_scale_rate_s) <= 0.0
                or float(sample.log_scale) <= float(previous.log_scale)
            ):
                raise ValueError(
                    "near-plane evidence lost its expansion trend"
                )
            previous = sample
        observed = self.last_observed_sample
        if observed is None or observed == previous:
            return
        if (
            observed.gate_index != first.gate_index
            or observed.track_id != first.track_id
            or observed.geometry_basis != first.geometry_basis
            or not _same_camera_epoch(
                observed.camera_token,
                first.camera_token,
            )
        ):
            raise ValueError(
                "near-plane continuity crossed its gate, track, or epoch"
            )
        if (
            not _token_strictly_newer(
                observed.camera_token,
                previous.camera_token,
            )
            or observed.observation_monotonic_ns
            <= previous.observation_monotonic_ns
            or observed.publication_monotonic_ns
            <= previous.publication_monotonic_ns
            or observed.wire_start_monotonic_ns
            <= previous.wire_start_monotonic_ns
            or observed.wire_return_monotonic_ns
            <= previous.wire_return_monotonic_ns
        ):
            raise ValueError(
                "near-plane continuity did not strictly advance"
            )
        if (
            float(observed.log_scale_rate_s) <= 0.0
            or float(observed.log_scale) <= float(previous.log_scale)
        ):
            raise ValueError(
                "near-plane continuity lost its expansion trend"
            )

    @property
    def gate_index(self) -> Optional[int]:
        return None if not self.samples else self.samples[0].gate_index

    @property
    def track_id(self) -> Optional[str]:
        return None if not self.samples else self.samples[0].track_id

    @property
    def camera_epoch(self) -> Optional[tuple[str, int]]:
        if not self.samples:
            return None
        token = self.samples[0].camera_token
        assert token.stream_id is not None
        return token.stream_id, token.generation


@dataclass(frozen=True, slots=True)
class NearPlaneLatch:
    """Immutable multi-frame authority preceding predictable censorship."""

    evidence: NearPlaneEvidence
    anchor_sample: NearPlaneWireSample
    required_corridor_frames: int
    crossing_min_log_scale: float
    basis: str = NEAR_PLANE_LATCH_BASIS

    def __post_init__(self) -> None:
        if type(self.evidence) is not NearPlaneEvidence:
            raise TypeError("near-plane latch evidence must be exact")
        if type(self.anchor_sample) is not NearPlaneWireSample:
            raise TypeError("near-plane latch anchor must be an exact sample")
        if (
            type(self.required_corridor_frames) is not int
            or self.required_corridor_frames <= 0
            or len(self.evidence.samples) != self.required_corridor_frames
        ):
            raise ValueError(
                "near-plane latch does not contain its required frame history"
            )
        if not _finite(self.crossing_min_log_scale):
            raise ValueError("near-plane close-scale bound must be finite")
        final_qualified = self.evidence.samples[-1]
        if (
            self.evidence.last_observed_sample != self.anchor_sample
            or self.anchor_sample.gate_index != final_qualified.gate_index
            or self.anchor_sample.track_id != final_qualified.track_id
            or not _same_camera_epoch(
                self.anchor_sample.camera_token,
                final_qualified.camera_token,
            )
            or (
                self.anchor_sample != final_qualified
                and (
                    not _token_strictly_newer(
                        self.anchor_sample.camera_token,
                        final_qualified.camera_token,
                    )
                    or self.anchor_sample.observation_monotonic_ns
                    <= final_qualified.observation_monotonic_ns
                    or self.anchor_sample.publication_monotonic_ns
                    <= final_qualified.publication_monotonic_ns
                    or self.anchor_sample.wire_start_monotonic_ns
                    <= final_qualified.wire_start_monotonic_ns
                    or self.anchor_sample.wire_return_monotonic_ns
                    <= final_qualified.wire_return_monotonic_ns
                    or float(self.anchor_sample.log_scale_rate_s) <= 0.0
                    or float(self.anchor_sample.log_scale)
                    <= float(final_qualified.log_scale)
                )
            )
        ):
            raise ValueError("near-plane latch anchor is discontinuous")
        if (
            float(self.anchor_sample.log_scale)
            < float(self.crossing_min_log_scale)
        ):
            raise ValueError("near-plane latch has not reached close scale")
        expected_basis = (
            DYNAMIC_NEAR_PLANE_LATCH_BASIS
            if final_qualified.geometry_basis
            == DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS
            else NEAR_PLANE_LATCH_BASIS
        )
        if self.basis != expected_basis:
            raise ValueError("near-plane latch basis is invalid")

    @property
    def lifecycle(self) -> CourseLifecycle:
        return CourseLifecycle.NEAR_PLANE_LATCHED

    @property
    def gate_index(self) -> int:
        assert self.evidence.gate_index is not None
        return self.evidence.gate_index

    @property
    def track_id(self) -> str:
        assert self.evidence.track_id is not None
        return self.evidence.track_id

    @property
    def anchor_camera_token(self) -> CameraFrameToken:
        return self.anchor_sample.camera_token

    @property
    def accepted_command(self) -> tuple[float, float, float, float]:
        return self.anchor_sample.command


def _observable_axis_unsafe(
    *,
    value: float,
    rate: float,
    maximum_abs_value: float,
) -> bool:
    """Bound the observed center now and at the fixed control horizon."""

    return bool(
        abs(value) > maximum_abs_value
        or abs(
            value + rate * PREPASS_CURRENT_PROJECTION_HORIZON_S
        )
        > maximum_abs_value
    )


def _wire_sample_hard_safe(
    sample: NearPlaneWireSample,
    *,
    min_track_confidence: float,
    min_association_confidence: float,
) -> bool:
    x = float(sample.normalized_x)
    y = float(sample.normalized_y_down)
    return bool(
        sample.clipping == FrameEdge.NONE
        and not sample.center_censored
        and not sample.ambiguous
        and float(sample.confidence) >= min_track_confidence
        and float(sample.association_confidence)
        >= min_association_confidence
        and abs(x) <= PREPASS_CURRENT_MAX_ABS_X_NORM
        and abs(y) <= PREPASS_CURRENT_MAX_ABS_Y_NORM
    )


def _wire_sample_usable(
    sample: NearPlaneWireSample,
    *,
    min_track_confidence: float,
    min_association_confidence: float,
) -> bool:
    x = float(sample.normalized_x)
    y = float(sample.normalized_y_down)
    x_rate = float(sample.normalized_x_rate_s)
    y_rate = float(sample.normalized_y_rate_down_s)
    scale_rate = float(sample.log_scale_rate_s)
    return bool(
        _wire_sample_hard_safe(
            sample,
            min_track_confidence=min_track_confidence,
            min_association_confidence=min_association_confidence,
        )
        and not _observable_axis_unsafe(
            value=x,
            rate=x_rate,
            maximum_abs_value=PREPASS_CURRENT_MAX_ABS_X_NORM,
        )
        and not _observable_axis_unsafe(
            value=y,
            rate=y_rate,
            maximum_abs_value=PREPASS_CURRENT_MAX_ABS_Y_NORM,
        )
        and PREPASS_CURRENT_MIN_LOG_SCALE_RATE_S <= scale_rate
        and scale_rate <= PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
        and scale_rate > 0.0
    )


def advance_dynamic_near_plane_evidence(
    evidence: NearPlaneEvidence,
    sample: NearPlaneWireSample,
    *,
    required_corridor_frames: int,
    crossing_min_log_scale: float,
    horizontal_corridor: float,
    vertical_corridor: float,
    minimum_post_governor_contact_budget_s: float,
    min_track_confidence: float,
    min_association_confidence: float,
) -> tuple[NearPlaneEvidence, Optional[NearPlaneLatch]]:
    """Advance exact-wire passage evidence in IMU-derotated coordinates."""

    if type(evidence) is not NearPlaneEvidence:
        raise TypeError("dynamic near-plane reducer evidence must be exact")
    if type(sample) is not NearPlaneWireSample:
        raise TypeError("dynamic near-plane reducer sample must be exact")
    if sample.geometry_basis != DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS:
        raise ValueError("dynamic near-plane sample has the wrong geometry basis")
    if (
        type(required_corridor_frames) is not int
        or required_corridor_frames <= 0
    ):
        raise ValueError("required corridor frames must be positive")
    for name, value in (
        ("crossing minimum log scale", crossing_min_log_scale),
        ("horizontal corridor", horizontal_corridor),
        ("vertical corridor", vertical_corridor),
        (
            "minimum post-governor contact budget",
            minimum_post_governor_contact_budget_s,
        ),
    ):
        if not _finite(value):
            raise ValueError(f"{name} must be finite")
    if (
        float(horizontal_corridor) <= 0.0
        or float(vertical_corridor) <= 0.0
        or float(minimum_post_governor_contact_budget_s) <= 0.0
    ):
        raise ValueError(
            "dynamic passage corridors and contact budget must be positive"
        )
    track_floor = _confidence_threshold(
        min_track_confidence,
        "minimum track confidence",
    )
    association_floor = _confidence_threshold(
        min_association_confidence,
        "minimum association confidence",
    )

    previous = (
        evidence.last_observed_sample
        if evidence.last_observed_sample is not None
        else (evidence.samples[-1] if evidence.samples else None)
    )
    if previous is not None:
        same_lineage = bool(
            sample.gate_index == previous.gate_index
            and sample.track_id == previous.track_id
            and sample.geometry_basis == previous.geometry_basis
            and _same_camera_epoch(
                sample.camera_token,
                previous.camera_token,
            )
        )
        strictly_advancing = bool(
            _token_strictly_newer(
                sample.camera_token,
                previous.camera_token,
            )
            and sample.observation_monotonic_ns
            > previous.observation_monotonic_ns
            and sample.publication_monotonic_ns
            > previous.publication_monotonic_ns
            and sample.wire_start_monotonic_ns
            > previous.wire_start_monotonic_ns
            and sample.wire_return_monotonic_ns
            > previous.wire_return_monotonic_ns
        )
        if not same_lineage or not strictly_advancing:
            return NearPlaneEvidence(), None

    scale_lower_bound = float(sample.log_scale) - 2.0 * float(
        sample.log_scale_std
    )
    assert sample.predicted_crossing_x_norm is not None
    assert sample.predicted_crossing_y_down_norm is not None
    assert sample.predicted_crossing_x_std_norm is not None
    assert sample.predicted_crossing_y_std_norm is not None
    assert sample.crossing_allowance_x_norm is not None
    assert sample.crossing_allowance_y_norm is not None
    assert sample.crossing_swept_x_occupancy_norm is not None
    assert sample.crossing_swept_y_occupancy_norm is not None
    assert sample.current_crossing_x_q is not None
    assert sample.current_crossing_y_q is not None
    assert sample.crossing_x_q_rate_s is not None
    assert sample.crossing_y_q_rate_s is not None
    assert sample.post_governor_contact_budget_s is not None
    full_sweep_clearance = (
        float(sample.crossing_allowance_x_norm)
        - float(sample.crossing_swept_x_occupancy_norm),
        float(sample.crossing_allowance_y_norm)
        - float(sample.crossing_swept_y_occupancy_norm),
    )
    terminal_occupancy = (
        abs(float(sample.predicted_crossing_x_norm))
        + 2.0 * float(sample.predicted_crossing_x_std_norm),
        abs(float(sample.predicted_crossing_y_down_norm))
        + 2.0 * float(sample.predicted_crossing_y_std_norm),
    )
    terminal_clearance = (
        float(sample.crossing_allowance_x_norm)
        - terminal_occupancy[0],
        float(sample.crossing_allowance_y_norm)
        - terminal_occupancy[1],
    )
    current_q = (
        float(sample.current_crossing_x_q),
        float(sample.current_crossing_y_q),
    )
    q_rate = (
        float(sample.crossing_x_q_rate_s),
        float(sample.crossing_y_q_rate_s),
    )
    qualified = bool(
        sample.clipping == FrameEdge.NONE
        and not sample.center_censored
        and not sample.ambiguous
        and float(sample.confidence) >= track_floor
        and float(sample.association_confidence) >= association_floor
        # A zero allowance is a valid, early passage-point result, but it
        # leaves no physical aperture authority and therefore cannot qualify.
        and float(sample.crossing_allowance_x_norm) > 0.0
        and float(sample.crossing_allowance_y_norm) > 0.0
        and terminal_clearance[0] >= 0.0
        and terminal_clearance[1] >= 0.0
        and all(
            full_sweep_clearance[axis] >= 0.0
            or current_q[axis] * q_rate[axis] < 0.0
            for axis in range(2)
        )
        and float(sample.log_scale_rate_s) > 0.0
    )
    if not qualified:
        return NearPlaneEvidence(), None

    if (
        previous is None
        or float(sample.log_scale) <= float(previous.log_scale)
    ):
        retained = (sample,)
    else:
        retained = evidence.samples + (sample,)
    if len(retained) > required_corridor_frames:
        retained = retained[-required_corridor_frames:]
    advanced = NearPlaneEvidence(
        samples=retained,
        last_observed_sample=sample,
    )
    if (
        len(retained) < required_corridor_frames
        or scale_lower_bound < float(crossing_min_log_scale)
        # Contact budget describes the command that can be acted on now.  A
        # historical sample's older wire/request gap cannot invalidate an
        # otherwise continuous terminal-geometry corridor.
        or float(sample.post_governor_contact_budget_s)
        < float(minimum_post_governor_contact_budget_s)
    ):
        return advanced, None
    return advanced, NearPlaneLatch(
        evidence=advanced,
        anchor_sample=sample,
        required_corridor_frames=required_corridor_frames,
        crossing_min_log_scale=float(crossing_min_log_scale),
        basis=DYNAMIC_NEAR_PLANE_LATCH_BASIS,
    )


def advance_near_plane_evidence(
    evidence: NearPlaneEvidence,
    sample: NearPlaneWireSample,
    *,
    required_corridor_frames: int,
    crossing_min_log_scale: float,
    min_track_confidence: float,
    min_association_confidence: float,
) -> tuple[NearPlaneEvidence, Optional[NearPlaneLatch]]:
    """Advance physical near-plane evidence without command-count authority.

    Publication gaps are permitted.  A repeated or regressed publication never
    contributes evidence, while a fresh sample after a broken expansion trend
    starts a new candidate history.
    """

    if type(evidence) is not NearPlaneEvidence:
        raise TypeError("near-plane reducer evidence must be exact")
    if type(sample) is not NearPlaneWireSample:
        raise TypeError("near-plane reducer sample must be exact")
    if (
        type(required_corridor_frames) is not int
        or required_corridor_frames <= 0
    ):
        raise ValueError("required corridor frames must be positive")
    if not _finite(crossing_min_log_scale):
        raise ValueError("crossing minimum log scale must be finite")
    track_floor = _confidence_threshold(
        min_track_confidence,
        "minimum track confidence",
    )
    association_floor = _confidence_threshold(
        min_association_confidence,
        "minimum association confidence",
    )
    previous: Optional[NearPlaneWireSample] = None
    if evidence.samples:
        previous = (
            evidence.last_observed_sample
            or evidence.samples[-1]
        )
        same_lineage = bool(
            sample.gate_index == previous.gate_index
            and sample.track_id == previous.track_id
            and sample.geometry_basis == previous.geometry_basis
            and _same_camera_epoch(
                sample.camera_token,
                previous.camera_token,
            )
        )
        strictly_advancing = bool(
            _token_strictly_newer(
                sample.camera_token,
                previous.camera_token,
            )
            and sample.observation_monotonic_ns
            > previous.observation_monotonic_ns
            and sample.publication_monotonic_ns
            > previous.publication_monotonic_ns
            and sample.wire_start_monotonic_ns
            > previous.wire_start_monotonic_ns
            and sample.wire_return_monotonic_ns
            > previous.wire_return_monotonic_ns
        )
        if not same_lineage or not strictly_advancing:
            return NearPlaneEvidence(), None

    hard_safe = _wire_sample_hard_safe(
        sample,
        min_track_confidence=track_floor,
        min_association_confidence=association_floor,
    )
    if not hard_safe:
        return NearPlaneEvidence(), None

    usable = _wire_sample_usable(
        sample,
        min_track_confidence=track_floor,
        min_association_confidence=association_floor,
    )
    if not usable:
        if (
            previous is not None
            and float(sample.log_scale_rate_s) > 0.0
            and float(sample.log_scale) > float(previous.log_scale)
        ):
            # A high-rate or projected-center transient is not itself latch
            # evidence, but it also does not erase prior accepted samples
            # while the same hard-safe aperture continues to expand.  This
            # permits a later qualified 30 Hz publication to complete the
            # multi-frame latch without pretending the transient was clean.
            bridged = NearPlaneEvidence(
                samples=evidence.samples,
                last_observed_sample=sample,
            )
            if (
                len(bridged.samples) >= required_corridor_frames
                and float(sample.log_scale)
                >= float(crossing_min_log_scale)
            ):
                return bridged, NearPlaneLatch(
                    evidence=bridged,
                    anchor_sample=sample,
                    required_corridor_frames=required_corridor_frames,
                    crossing_min_log_scale=float(
                        crossing_min_log_scale
                    ),
                )
            return bridged, None
        return NearPlaneEvidence(), None

    retained: tuple[NearPlaneWireSample, ...]
    if previous is None:
        retained = (sample,)
    else:
        if float(sample.log_scale) <= float(previous.log_scale):
            retained = (sample,)
        else:
            retained = evidence.samples + (sample,)

    if len(retained) > required_corridor_frames:
        retained = retained[-required_corridor_frames:]
    advanced = NearPlaneEvidence(
        samples=retained,
        last_observed_sample=sample,
    )
    if (
        len(advanced.samples) < required_corridor_frames
        or float(advanced.samples[-1].log_scale)
        < float(crossing_min_log_scale)
    ):
        return advanced, None
    latch = NearPlaneLatch(
        evidence=advanced,
        anchor_sample=sample,
        required_corridor_frames=required_corridor_frames,
        crossing_min_log_scale=float(crossing_min_log_scale),
    )
    return advanced, latch


def classify_latched_measurement(
    latch: NearPlaneLatch,
    *,
    previous_camera_token: CameraFrameToken,
    camera_token: CameraFrameToken,
    current_gate_index: Optional[int],
    current_track_id: Optional[str],
    track_latest_camera_token: Optional[CameraFrameToken],
    track_role: Optional[VisualTrackRole],
    track_authoritative_gate_index: Optional[int],
    visible: bool,
    missed_frame_count: int,
    ambiguous: bool,
    clipping: FrameEdge,
    center_censored: bool,
    normalized_x: Optional[float],
    normalized_y_down: Optional[float],
    normalized_x_rate_s: Optional[float],
    normalized_y_rate_down_s: Optional[float],
    apparent_scale: Optional[float],
    confidence: Optional[float],
    association_confidence: Optional[float],
    min_track_confidence: float,
    min_association_confidence: float,
    race_finished: bool = False,
) -> LatchedMeasurementMode:
    """Classify one fresh post-latch tracker publication.

    The function creates no command authority.  ``COAST`` only says that the
    caller may use its separately bounded command lease.  ``CREDIT_WAIT`` means
    the measurement is unavailable and therefore permits only the caller's
    separately bounded latched coast or cut-and-authoritative-credit path.
    """

    if type(latch) is not NearPlaneLatch:
        raise TypeError("latched measurement requires an exact latch")
    track_floor = _confidence_threshold(
        min_track_confidence,
        "minimum track confidence",
    )
    association_floor = _confidence_threshold(
        min_association_confidence,
        "minimum association confidence",
    )
    if (
        type(visible) is not bool
        or type(ambiguous) is not bool
        or type(center_censored) is not bool
        or type(race_finished) is not bool
        or type(missed_frame_count) is not int
        or missed_frame_count < 0
        or type(clipping) is not FrameEdge
    ):
        raise TypeError("latched measurement fields have invalid types")

    anchor = latch.anchor_camera_token
    lineage_usable = bool(
        _live_token(previous_camera_token)
        and _live_token(camera_token)
        and _token_not_older(previous_camera_token, anchor)
        and _token_strictly_newer(camera_token, previous_camera_token)
    )
    if (
        not lineage_usable
        or race_finished
        or current_gate_index != latch.gate_index
        or current_track_id != latch.track_id
    ):
        return LatchedMeasurementMode.UNSAFE
    if ambiguous or track_role is VisualTrackRole.AMBIGUOUS:
        return LatchedMeasurementMode.UNSAFE

    if not visible:
        if (
            missed_frame_count <= 0
            or track_role not in {None, VisualTrackRole.CURRENT}
            or track_authoritative_gate_index not in {
                None,
                latch.gate_index,
            }
            or (
                track_latest_camera_token is not None
                and (
                    not _same_camera_epoch(
                        track_latest_camera_token,
                        anchor,
                    )
                    or not _token_not_older(
                        previous_camera_token,
                        track_latest_camera_token,
                    )
                )
            )
        ):
            return LatchedMeasurementMode.UNSAFE
        return LatchedMeasurementMode.CREDIT_WAIT

    if (
        missed_frame_count != 0
        or track_role is not VisualTrackRole.CURRENT
        or track_authoritative_gate_index != latch.gate_index
        or track_latest_camera_token != camera_token
    ):
        return LatchedMeasurementMode.UNSAFE

    numeric = (
        normalized_x,
        normalized_y_down,
        normalized_x_rate_s,
        normalized_y_rate_down_s,
        apparent_scale,
        confidence,
        association_confidence,
    )
    if not all(_finite(value) for value in numeric):
        return LatchedMeasurementMode.CREDIT_WAIT
    assert normalized_x is not None
    assert normalized_y_down is not None
    assert normalized_x_rate_s is not None
    assert normalized_y_rate_down_s is not None
    assert apparent_scale is not None
    assert confidence is not None
    assert association_confidence is not None
    if (
        float(apparent_scale) <= 0.0
        or not 0.0 <= float(confidence) <= 1.0
        or not 0.0 <= float(association_confidence) <= 1.0
    ):
        return LatchedMeasurementMode.CREDIT_WAIT
    if (
        float(confidence) < track_floor
        or float(association_confidence) < association_floor
        or float(apparent_scale)
        < latch.evidence.samples[-1].apparent_scale
    ):
        return LatchedMeasurementMode.CREDIT_WAIT

    horizontal_censored = bool(
        clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
    )
    vertical_censored = bool(
        clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
    )
    if center_censored and not (
        horizontal_censored or vertical_censored
    ):
        horizontal_censored = True
        vertical_censored = True
    if horizontal_censored and vertical_censored:
        return LatchedMeasurementMode.CREDIT_WAIT
    if latch.basis == DYNAMIC_NEAR_PLANE_LATCH_BASIS:
        if (
            not horizontal_censored
            and abs(float(normalized_x)) > 1.0
        ):
            return LatchedMeasurementMode.UNSAFE
        if (
            not vertical_censored
            and abs(float(normalized_y_down)) > 1.0
        ):
            return LatchedMeasurementMode.UNSAFE
        return LatchedMeasurementMode.COAST

    if (
        not horizontal_censored
        and abs(float(normalized_x))
        > PREPASS_CURRENT_MAX_ABS_X_NORM
    ):
        return LatchedMeasurementMode.UNSAFE
    if (
        not vertical_censored
        and abs(float(normalized_y_down))
        > PREPASS_CURRENT_MAX_ABS_Y_NORM
    ):
        return LatchedMeasurementMode.UNSAFE
    return LatchedMeasurementMode.COAST


__all__ = [
    "CourseLifecycle",
    "DYNAMIC_CROSSING_PREDICTION_MAX_HORIZON_S",
    "DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS",
    "DYNAMIC_NEAR_PLANE_LATCH_BASIS",
    "LatchedMeasurementMode",
    "NEAR_PLANE_LATCH_BASIS",
    "NearPlaneEvidence",
    "NearPlaneLatch",
    "NearPlaneWireSample",
    "PostCreditMeasurementMode",
    "RAW_NEAR_PLANE_GEOMETRY_BASIS",
    "advance_dynamic_near_plane_evidence",
    "advance_near_plane_evidence",
    "classify_post_credit_measurement",
    "classify_latched_measurement",
]
