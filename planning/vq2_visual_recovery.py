"""Fail-closed admission for short post-promotion visual recovery.

The ordinary visual-alignment entry gate remains authoritative.  This module
only decides whether one already-promoted track has enough exact receiver
history and image margin for a short no-advance response after race credit.
Promotion may freeze exactly one clean target observation after credit; its
pre-credit prefix remains the transition identity proof.  This module derives
no metric pose or world-frame map.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    AssociationEvidence,
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualTrack,
    VisualTrackRole,
    VisualTrackSample,
    visual_track_history_sha256,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateTransition,
    RaceStatusProvenanceBasis,
)
from planning.vq2_visual_alignment import (
    POST_PROMOTION_ENTRY_MAX_ABS_X_NORM,
    POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM,
    POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD,
    VisualAlignmentCaptureAdmission,
    require_visual_alignment_capture_entry,
)
from planning.vq2_visual_servo import VisualTarget


RECOVERY_HISTORY_SAMPLE_COUNT = 4
RECOVERY_MIN_HISTORY_SPAN_S = 0.090
RECOVERY_MIN_ASSOCIATION_CONFIDENCE = 0.90
RECOVERY_MIN_DETECTION_CONFIDENCE = 0.65
RECOVERY_PRETRANSITION_VISIBILITY_SAMPLE_COUNT = 5
RECOVERY_PRE_GAP_HISTORY_SAMPLE_COUNT = 3
RECOVERY_MIN_PRE_GAP_HISTORY_SPAN_S = 0.060
RECOVERY_MIN_PRE_GAP_DETECTION_CONFIDENCE = 0.64
RECOVERY_MIN_PRE_GAP_ASSOCIATION_CONFIDENCE = 0.90
RECOVERY_MIN_REACQUISITION_DETECTION_CONFIDENCE = 0.70
RECOVERY_MIN_REACQUISITION_ASSOCIATION_CONFIDENCE = 0.65
RECOVERY_TRACKER_MAX_ASSIGNMENT_COST = 0.82
RECOVERY_MAX_REACQUISITION_MISSED_FRAMES = 11
RECOVERY_MAX_REACQUISITION_PUBLICATION_DELTA = 12
RECOVERY_MAX_REACQUISITION_UNOBSERVED_PUBLICATIONS = 1
RECOVERY_MAX_REACQUISITION_GAP_S = 0.410
RECOVERY_MAX_REACQUISITION_ASSOCIATION_COST = 0.29
RECOVERY_MIN_REACQUISITION_BBOX_IOU = 0.55
RECOVERY_MIN_REACQUISITION_DIRECT_BBOX_IOU = 0.30
RECOVERY_MAX_REACQUISITION_CENTER_RESIDUAL_NORM = 0.065
RECOVERY_MAX_REACQUISITION_ABS_LOG_WIDTH_CHANGE = 0.32
RECOVERY_MAX_REACQUISITION_ABS_LOG_HEIGHT_CHANGE = 0.22
# The exact build-3385 delayed-credit bridge measured 0.1461438556.  This
# narrow ceiling still composes with independent overlap, per-dimension,
# residual, motion, timing, confidence, and clipping bounds below.
RECOVERY_MAX_REACQUISITION_ABS_LOG_AREA_RESIDUAL = 0.15
RECOVERY_MAX_REACQUISITION_CENTER_RATE_NORM_S = 0.25
RECOVERY_MAX_REACQUISITION_LOG_SCALE_RATE_S = 0.70
RECOVERY_MIN_FRAME_DT_S = 0.020
RECOVERY_MAX_FRAME_DT_S = 0.050
RECOVERY_MAX_RECEIVER_PIPELINE_LATENCY_S = 0.010
RECOVERY_MAX_ANCHOR_CREDIT_AGE_S = 0.020
RECOVERY_MAX_START_DELAY_AFTER_CREDIT_S = 0.060
RECOVERY_MAX_CONTINUATION_AGE_S = 0.060
RECOVERY_MAX_POSTCREDIT_PROMOTION_SAMPLES = 1
RECOVERY_MAX_ABS_X_NORM = 0.60
RECOVERY_MAX_ABS_Y_NORM = 0.68
RECOVERY_MAX_FILTERED_CENTER_RATE_NORM_S = 0.40
RECOVERY_MAX_FILTERED_LOG_SCALE_RATE_S = 1.00
RECOVERY_MAX_RAW_CENTER_RATE_NORM_S = 0.50
RECOVERY_MAX_RAW_LOG_SCALE_RATE_S = 1.10
RECOVERY_MAX_RAW_LOG_DIMENSION_RATE_S = 1.30
RECOVERY_MIN_PROJECTION_HORIZON_S = 0.080
RECOVERY_MAX_VALIDATION_TO_WIRE_DELAY_S = 0.005
RECOVERY_COMMAND_RESPONSE_HORIZON_S = 0.045
RECOVERY_MAX_PROJECTION_HORIZON_S = 0.140
# Image centers use [-1, 1], while bboxes use [0, 1].
RECOVERY_CENTER_PADDING_X_NORM = 4.0 / 640.0
RECOVERY_CENTER_PADDING_Y_NORM = 4.0 / 360.0
RECOVERY_BBOX_PADDING_X_NORM = 2.0 / 640.0
RECOVERY_BBOX_PADDING_Y_NORM = 2.0 / 360.0
RECOVERY_MIN_PROJECTED_EDGE_MARGIN_X_NORM = 6.0 / 640.0
RECOVERY_MIN_PROJECTED_EDGE_MARGIN_Y_NORM = 6.0 / 360.0
RECOVERY_MAX_PROJECTED_WIDTH_NORM = 0.22
RECOVERY_MAX_PROJECTED_HEIGHT_NORM = 0.30
RECOVERY_MAX_PROJECTED_AREA_NORM = 0.08
RECOVERY_MAX_PROJECTED_APPARENT_SCALE = 0.23
# Recovery is already limited to |y| <= 0.68 and independently projects the
# entire bbox against a six-pixel edge margin.  Permit only this immutable
# sub-pixel extension beyond the ordinary-entry center bound while a
# no-advance corrective command is in flight.  It admits the exact build-3385
# token 173 at the 80 ms horizon with about 0.004 normalized margin; any extra
# observation age remains bounded by that margin, while faster outward motion
# or bbox edge loss still refuses.
RECOVERY_MAX_PROJECTED_ABS_Y_NORM = 0.715
RECOVERY_HARD_DURATION_S = 0.16
RECOVERY_MAX_FRESH_FRAMES = 5
RECOVERY_MAX_COMMANDS = 8
RECOVERY_MAX_COMMAND_RATE_RAD_S = 0.12
RECOVERY_MAX_YAW_RATE_RAD_S = 0.08
RECOVERY_MAX_THRUST = 0.285
RECOVERY_REQUIRED_STRICT_ENTRY_FRAMES = 2
_PROMOTION_HISTORY_AUTHORITY_ISSUER = object()


class VisualRecoveryRefusal(ValueError):
    """The transition anchor cannot safely support recovery authority."""


class _PromotionHistorySeal:
    """Module-issued immutable record bound to one validated history."""

    __slots__ = (
        "track_id",
        "history_length",
        "history_sha256",
        "history",
        "transition",
        "_frozen",
    )

    def __init__(
        self,
        *,
        issuer: object,
        track_id: str,
        history_length: int,
        history_sha256: str,
        history: tuple[VisualTrackSample, ...],
        transition: ConfirmedGateTransition,
    ) -> None:
        if issuer is not _PROMOTION_HISTORY_AUTHORITY_ISSUER:
            raise TypeError("promotion history seals are module-issued")
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "history_length", history_length)
        object.__setattr__(self, "history_sha256", history_sha256)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_frozen", False):
            raise TypeError("promotion history seals are immutable")
        object.__setattr__(self, _name, _value)


@dataclass(frozen=True, slots=True)
class PromotionHistoryAuthority:
    """Validator-issued immutable promotion prefix for wire-time rechecks."""

    track_id: str
    history_length: int
    history_sha256: str
    history: tuple[VisualTrackSample, ...]
    _validator_seal: _PromotionHistorySeal


@dataclass(frozen=True, slots=True)
class ReacquisitionBridgeAdmission:
    """Exact identity evidence across one bounded visibility interruption."""

    predecessor_token: CameraFrameToken
    reacquisition_token: CameraFrameToken
    missed_frame_count: int
    tracker_frame_delta: int
    publication_delta: int
    unobserved_publication_count: int
    observation_gap_s: float
    publication_gap_s: float
    association_confidence: float
    association_cost: float
    predicted_center_residual_norm: float
    bbox_iou: float
    direct_bbox_iou: float
    log_width_change: float
    log_height_change: float
    log_area_residual: float
    average_horizontal_rate_norm_s: float
    average_vertical_rate_norm_s: float
    average_log_scale_rate_s: float


@dataclass(frozen=True, slots=True)
class TransitionRecoveryAdmission:
    """Exact image-history and projection facts for one recovery anchor."""

    track_id: str
    credit_prefix_token: CameraFrameToken
    promotion_anchor_token: CameraFrameToken
    history_tokens: tuple[CameraFrameToken, ...]
    race_status_sequence: int
    race_received_monotonic_ns: int
    credit_prefix_publication_monotonic_ns: int
    credit_prefix_age_s: float
    promotion_anchor_observation_monotonic_ns: int
    promotion_anchor_publication_monotonic_ns: int
    promotion_anchor_publication_delta_from_credit_s: float
    recovery_start_delay_s: float
    observation_age_s: float
    projection_horizon_s: float
    stable_history_span_s: float
    min_history_detection_confidence: float
    min_history_association_confidence: float
    promotion_identity_sha256: str
    reacquisition_bridge: ReacquisitionBridgeAdmission | None
    horizontal_error: float
    vertical_error_image_down: float
    filtered_horizontal_rate_s: float
    filtered_vertical_rate_down_s: float
    filtered_log_scale_rate_s: float
    max_raw_horizontal_rate_s: float
    max_raw_vertical_rate_down_s: float
    max_raw_log_scale_rate_s: float
    max_raw_log_width_rate_s: float
    max_raw_log_height_rate_s: float
    max_positive_raw_log_scale_rate_s: float
    projected_abs_horizontal_error: float
    projected_abs_vertical_error_image_down: float
    projected_bbox_norm_ltrb: tuple[float, float, float, float]
    projected_width_norm: float
    projected_height_norm: float
    projected_area_norm: float
    projected_apparent_scale: float
    measured_pitch_rad: float


@dataclass(frozen=True, slots=True)
class RecoveryContinuationAdmission:
    """Fresh post-credit facts retaining no-advance recovery authority."""

    track_id: str
    previous_token: CameraFrameToken
    frame_token: CameraFrameToken
    observation_age_s: float
    recovery_elapsed_s: float
    projection_horizon_s: float
    stable_history_span_s: float
    min_history_detection_confidence: float
    min_history_association_confidence: float
    promotion_identity_sha256: str
    reacquisition_bridge: ReacquisitionBridgeAdmission | None
    capture: VisualAlignmentCaptureAdmission
    max_raw_horizontal_rate_s: float
    max_raw_vertical_rate_down_s: float
    max_raw_log_scale_rate_s: float
    max_raw_log_width_rate_s: float
    max_raw_log_height_rate_s: float
    projected_abs_horizontal_error: float
    projected_abs_vertical_error_image_down: float
    projected_bbox_norm_ltrb: tuple[float, float, float, float]
    projected_width_norm: float
    projected_height_norm: float
    projected_area_norm: float
    projected_apparent_scale: float


@dataclass(frozen=True, slots=True)
class _HistoryProjection:
    tokens: tuple[CameraFrameToken, ...]
    latest_observation_monotonic_ns: int
    latest_publication_monotonic_ns: int
    stable_history_span_s: float
    min_detection_confidence: float
    min_association_confidence: float
    max_raw_horizontal_rate_s: float
    max_raw_vertical_rate_down_s: float
    max_raw_log_scale_rate_s: float
    max_raw_log_width_rate_s: float
    max_raw_log_height_rate_s: float
    max_positive_raw_log_scale_rate_s: float
    projected_abs_horizontal_error: float
    projected_abs_vertical_error_image_down: float
    projected_bbox_norm_ltrb: tuple[float, float, float, float]
    projected_width_norm: float
    projected_height_norm: float
    projected_area_norm: float
    projected_apparent_scale: float


def _finite(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise VisualRecoveryRefusal(
            f"post-promotion recovery {label} must be finite"
        )
    return float(value)


def _finite_pitch(value: object) -> float:
    return _finite(value, "measured pitch")


def _require_live_transition_authority(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
    *,
    tracker_time_basis_id: str,
) -> tuple[int, int]:
    if type(track) is not VisualTrack:
        raise VisualRecoveryRefusal(
            "post-promotion recovery requires an exact VisualTrack"
        )
    if type(transition) is not ConfirmedGateTransition:
        raise VisualRecoveryRefusal(
            "post-promotion recovery requires an exact transition"
        )
    if type(tracker_time_basis_id) is not str or not tracker_time_basis_id:
        raise VisualRecoveryRefusal(
            "post-promotion recovery requires a camera time basis"
        )
    race = transition.race_status
    if (
        type(race) is not AuthoritativeRaceStatusRef
        or type(transition.from_gate_index) is not int
        or transition.from_gate_index < 0
        or type(transition.to_gate_index) is not int
        or transition.to_gate_index < 0
        or type(transition.promoted_track_id) is not str
        or not transition.promoted_track_id
        or type(transition.retired_track_id) is not str
        or not transition.retired_track_id
        or type(transition.camera_token_at_credit) is not CameraFrameToken
        or type(transition.promoted_first_token) is not CameraFrameToken
        or type(transition.promoted_latest_token_before_credit)
        is not CameraFrameToken
        or type(transition.promoted_latest_token_at_promotion)
        is not CameraFrameToken
        or type(transition.pretransition_frame_tokens) is not tuple
        or not transition.pretransition_frame_tokens
        or any(
            type(token) is not CameraFrameToken
            for token in transition.pretransition_frame_tokens
        )
        or type(transition.promoted_history_length_at_credit) is not int
        or type(transition.history_length_before_promotion) is not int
        or type(transition.history_length_after_promotion) is not int
        or type(transition.promoted_history_sha256) is not str
        or len(transition.promoted_history_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in transition.promoted_history_sha256
        )
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery transition structure is invalid"
        )
    if (
        type(track.history) is not tuple
        or not track.history
        or any(type(sample) is not VisualTrackSample for sample in track.history)
        or type(track.first_token) is not CameraFrameToken
        or type(track.latest_token) is not CameraFrameToken
        or type(track.missed_frame_count) is not int
        or type(track.consecutive_frame_count) is not int
        or type(track.total_observation_count) is not int
        or not track.visible
        or track.missed_frame_count != 0
        or track.consecutive_frame_count < RECOVERY_HISTORY_SAMPLE_COUNT
        or track.total_observation_count != len(track.history)
        or track.consecutive_frame_count > track.total_observation_count
        or track.ambiguous
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery track structure lacks current authority"
        )
    if (
        race.provenance_basis is not RaceStatusProvenanceBasis.LIVE_INGRESS
        or race.received_monotonic_ns is None
        or race.race_status_sequence is None
        or race.host_clock_id != tracker_time_basis_id
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery requires matched live race provenance"
        )
    if (
        transition.to_gate_index != transition.from_gate_index + 1
        or race.active_gate_index != transition.to_gate_index
        or race.race_finished
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery requires one unfinished adjacent transition"
        )
    if (
        track.track_id != transition.promoted_track_id
        or track.role is not VisualTrackRole.CURRENT
        or track.authoritative_gate_index != transition.to_gate_index
        or track.authority_race_status_sequence != race.race_status_sequence
        or track.authority_race_status_boot_ms != race.race_status_boot_ms
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery track authority disagrees with transition"
        )
    credit_length = transition.promoted_history_length_at_credit
    promotion_length = transition.history_length_after_promotion
    if (
        transition.promoted_first_token != track.first_token
        or transition.history_length_before_promotion != promotion_length
        or credit_length <= 0
        or credit_length > promotion_length
        or promotion_length <= 0
        or promotion_length - credit_length
        > RECOVERY_MAX_POSTCREDIT_PROMOTION_SAMPLES
        or len(track.history) < promotion_length
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery transition identity is inconsistent"
        )
    promotion_history = track.history[:promotion_length]
    credit_history = promotion_history[:credit_length]
    credit_token = transition.promoted_latest_token_before_credit
    promotion_token = transition.promoted_latest_token_at_promotion
    camera_token = transition.camera_token_at_credit
    if (
        promotion_history[0].token != transition.promoted_first_token
        or credit_history[-1].token != credit_token
        or promotion_history[-1].token != promotion_token
        or not transition.pretransition_frame_tokens
        or len(transition.pretransition_frame_tokens) > credit_length
        or tuple(
            sample.token
            for sample in credit_history[
                -len(transition.pretransition_frame_tokens):
            ]
        )
        != transition.pretransition_frame_tokens
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery history is not bound to the transition"
        )
    credit_sequence = credit_token.publication_sequence
    camera_sequence = camera_token.publication_sequence
    promotion_sequence = promotion_token.publication_sequence
    if (
        credit_token.stream_id != camera_token.stream_id
        or credit_token.generation != camera_token.generation
        or promotion_token.stream_id != credit_token.stream_id
        or promotion_token.generation != credit_token.generation
        or type(credit_sequence) is not int
        or type(camera_sequence) is not int
        or type(promotion_sequence) is not int
        or credit_sequence <= 0
        or camera_sequence < credit_sequence
        or promotion_sequence < credit_sequence
        or (
            camera_sequence == credit_sequence
            and camera_token != credit_token
        )
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery credit boundary is inconsistent"
        )
    if any(
        type(sample.publication_monotonic_ns) is not int
        or type(sample.observation_monotonic_ns) is not int
        or sample.observation_monotonic_ns < 0
        or sample.publication_monotonic_ns
        < sample.observation_monotonic_ns
        or sample.publication_monotonic_ns > race.received_monotonic_ns
        for sample in credit_history
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery credit prefix is not pre-credit"
        )
    postcredit_history = promotion_history[credit_length:]
    if postcredit_history:
        postcredit = postcredit_history[0]
        precredit = credit_history[-1]
        if (
            camera_sequence >= promotion_sequence
            or type(postcredit.observation_monotonic_ns) is not int
            or postcredit.observation_monotonic_ns
            <= race.received_monotonic_ns
            or type(postcredit.publication_monotonic_ns) is not int
            or postcredit.publication_monotonic_ns
            <= race.received_monotonic_ns
        ):
            raise VisualRecoveryRefusal(
                "post-promotion recovery promotion suffix is not post-credit"
            )
        _sample_geometry(
            postcredit,
            stream_id=credit_token.stream_id,
            generation=credit_token.generation,
        )
        _require_accepted_association(
            track,
            precredit,
            postcredit,
            expected_missed_frames=0,
            min_association_confidence=RECOVERY_MIN_ASSOCIATION_CONFIDENCE,
        )
    return int(race.received_monotonic_ns), int(race.race_status_sequence)


def _require_exact_transition_anchor(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
) -> None:
    if (
        track.latest_token != transition.promoted_latest_token_at_promotion
        or len(track.history) != transition.history_length_after_promotion
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery did not preserve the exact promotion anchor"
        )


def _require_promotion_history_digest(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
    *,
    authority: PromotionHistoryAuthority | None = None,
) -> None:
    promotion_length = transition.history_length_after_promotion
    if authority is not None:
        if type(authority) is not PromotionHistoryAuthority:
            raise VisualRecoveryRefusal(
                "post-promotion recovery prevalidated history changed"
            )
        seal = authority._validator_seal
        if (
            type(seal) is not _PromotionHistorySeal
            or authority.track_id != track.track_id
            or authority.history_length != promotion_length
            or authority.history_sha256
            != transition.promoted_history_sha256
            or type(authority.history) is not tuple
            or len(authority.history) != promotion_length
            or any(
                type(sample) is not VisualTrackSample
                for sample in authority.history
            )
            or authority.track_id != seal.track_id
            or authority.history_length != seal.history_length
            or authority.history_sha256 != seal.history_sha256
            or authority.history != seal.history
            or track.history[:promotion_length] != seal.history
            or transition != seal.transition
        ):
            raise VisualRecoveryRefusal(
                "post-promotion recovery prevalidated history changed"
            )
        return
    try:
        promotion_identity_sha256 = visual_track_history_sha256(
            track.history[:promotion_length]
        )
    except (TypeError, ValueError) as exc:
        raise VisualRecoveryRefusal(
            "post-promotion recovery history digest is invalid"
        ) from exc
    if promotion_identity_sha256 != transition.promoted_history_sha256:
        raise VisualRecoveryRefusal(
            "post-promotion recovery history digest changed after promotion"
        )


def require_promotion_history_authority(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
    *,
    tracker_time_basis_id: str,
) -> PromotionHistoryAuthority:
    """Verify the promotion digest once, outside the wire-critical lease."""

    _require_live_transition_authority(
        track,
        transition,
        tracker_time_basis_id=tracker_time_basis_id,
    )
    _require_promotion_history_digest(track, transition)
    promotion_length = transition.history_length_after_promotion
    history = track.history[:promotion_length]
    seal = _PromotionHistorySeal(
        issuer=_PROMOTION_HISTORY_AUTHORITY_ISSUER,
        track_id=track.track_id,
        history_length=promotion_length,
        history_sha256=transition.promoted_history_sha256,
        history=history,
        transition=transition,
    )
    return PromotionHistoryAuthority(
        track_id=track.track_id,
        history_length=promotion_length,
        history_sha256=transition.promoted_history_sha256,
        history=history,
        _validator_seal=seal,
    )


def _sample_geometry(
    sample: VisualTrackSample,
    *,
    stream_id: str,
    generation: int,
    min_detection_confidence: float = RECOVERY_MIN_DETECTION_CONFIDENCE,
    min_association_confidence: float = (
        RECOVERY_MIN_ASSOCIATION_CONFIDENCE
    ),
) -> tuple[float, float, float, float]:
    if type(sample) is not VisualTrackSample:
        raise VisualRecoveryRefusal(
            "post-promotion recovery history contains a non-sample value"
        )
    if (
        type(sample.token) is not CameraFrameToken
        or sample.token.stream_id != stream_id
        or sample.token.generation != generation
        or type(sample.token.publication_sequence) is not int
        or sample.token.publication_sequence <= 0
        or sample.provenance_basis
        is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        or type(sample.tracker_frame_sequence) is not int
        or sample.tracker_frame_sequence < 0
        or type(sample.observation_monotonic_ns) is not int
        or sample.observation_monotonic_ns < 0
        or type(sample.publication_monotonic_ns) is not int
        or sample.publication_monotonic_ns < sample.observation_monotonic_ns
        or type(sample.clipping) is not FrameEdge
        or sample.clipping != FrameEdge.NONE
        or type(sample.center_censored) is not bool
        or sample.center_censored
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery history lacks clean live provenance"
        )
    pipeline_latency_s = (
        sample.publication_monotonic_ns - sample.observation_monotonic_ns
    ) / 1_000_000_000.0
    if pipeline_latency_s > RECOVERY_MAX_RECEIVER_PIPELINE_LATENCY_S:
        raise VisualRecoveryRefusal(
            "post-promotion recovery receiver pipeline latency is excessive"
        )
    confidence = _finite(sample.confidence, "sample confidence")
    association = _finite(
        sample.association_confidence,
        "sample association confidence",
    )
    if (
        not min_detection_confidence <= confidence <= 1.0
        or not min_association_confidence <= association <= 1.0
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery history confidence is insufficient"
        )
    if type(sample.center_norm) is not tuple or len(sample.center_norm) != 2:
        raise VisualRecoveryRefusal(
            "post-promotion recovery sample center is invalid"
        )
    center_x = _finite(sample.center_norm[0], "sample horizontal center")
    center_y = _finite(sample.center_norm[1], "sample vertical center")
    if abs(center_x) > 1.0 or abs(center_y) > 1.0:
        raise VisualRecoveryRefusal(
            "post-promotion recovery sample center is outside the image"
        )
    if type(sample.bbox_norm) is not tuple or len(sample.bbox_norm) != 4:
        raise VisualRecoveryRefusal(
            "post-promotion recovery sample bbox is invalid"
        )
    left, top, right, bottom = (
        _finite(value, "sample bbox coordinate") for value in sample.bbox_norm
    )
    if not (
        0.0 <= left < right <= 1.0
        and 0.0 <= top < bottom <= 1.0
        and left <= 0.5 * (center_x + 1.0) <= right
        and top <= 0.5 * (center_y + 1.0) <= bottom
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery sample bbox is incoherent"
        )
    scale = _finite(sample.apparent_scale, "sample apparent scale")
    if scale <= 0.0:
        raise VisualRecoveryRefusal(
            "post-promotion recovery sample scale is invalid"
        )
    return left, top, right, bottom


def _require_accepted_association(
    track: VisualTrack,
    previous: VisualTrackSample,
    sample: VisualTrackSample,
    *,
    expected_missed_frames: int,
    min_association_confidence: float,
    max_unobserved_publications: int = 0,
) -> AssociationEvidence:
    evidence = sample.accepted_association
    if type(evidence) is not AssociationEvidence:
        raise VisualRecoveryRefusal(
            "post-promotion recovery association provenance is absent"
        )
    if (
        type(expected_missed_frames) is not int
        or expected_missed_frames < 0
        or type(max_unobserved_publications) is not int
        or max_unobserved_publications < 0
        or type(evidence.missed_frame_count_before_association) is not int
        or evidence.missed_frame_count_before_association
        != expected_missed_frames
        or type(evidence.observation_gap_ns) is not int
        or evidence.observation_gap_ns <= 0
        or type(evidence.publication_gap_ns) is not int
        or evidence.publication_gap_ns <= 0
        or type(evidence.ambiguous) is not bool
        or evidence.ambiguous
        or type(evidence.track_ambiguous_before_association) is not bool
        or evidence.track_ambiguous_before_association
        or evidence.track_id != track.track_id
        or evidence.previous_token != previous.token
        or evidence.current_token != sample.token
        or evidence.detection_source_index != sample.source_index
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery association provenance is inconsistent"
        )
    if (
        type(previous.tracker_frame_sequence) is not int
        or type(sample.tracker_frame_sequence) is not int
        or type(previous.token.publication_sequence) is not int
        or type(sample.token.publication_sequence) is not int
        or type(previous.observation_monotonic_ns) is not int
        or type(sample.observation_monotonic_ns) is not int
        or type(previous.publication_monotonic_ns) is not int
        or type(sample.publication_monotonic_ns) is not int
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery association timing is invalid"
        )
    expected_delta = expected_missed_frames + 1
    tracker_delta = (
        sample.tracker_frame_sequence - previous.tracker_frame_sequence
    )
    publication_delta = (
        sample.token.publication_sequence
        - previous.token.publication_sequence
    )
    observation_gap_ns = (
        sample.observation_monotonic_ns
        - previous.observation_monotonic_ns
    )
    publication_gap_ns = (
        sample.publication_monotonic_ns
        - previous.publication_monotonic_ns
    )
    if (
        sample.token.stream_id != previous.token.stream_id
        or sample.token.generation != previous.token.generation
        or tracker_delta != expected_delta
        or publication_delta < tracker_delta
        or publication_delta - tracker_delta
        > max_unobserved_publications
        or observation_gap_ns != evidence.observation_gap_ns
        or publication_gap_ns != evidence.publication_gap_ns
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery association gap is inconsistent"
        )

    confidence = _finite(
        evidence.confidence,
        "association evidence confidence",
    )
    sample_confidence = _finite(
        sample.association_confidence,
        "sample association confidence",
    )
    cost = _finite(evidence.cost, "association evidence cost")
    residual = _finite(
        evidence.predicted_center_residual_norm,
        "association center residual",
    )
    bbox_iou = _finite(evidence.bbox_iou, "association bbox overlap")
    log_width = _finite(
        evidence.log_width_change,
        "association log-width change",
    )
    log_height = _finite(
        evidence.log_height_change,
        "association log-height change",
    )
    log_area = _finite(
        evidence.log_area_residual,
        "association log-area residual",
    )
    clipping_continuity = _finite(
        evidence.clipping_continuity,
        "association clipping continuity",
    )
    temporal_consistency = _finite(
        evidence.temporal_consistency,
        "association temporal consistency",
    )
    if evidence.appearance_distance is not None:
        appearance_distance = _finite(
            evidence.appearance_distance,
            "association appearance distance",
        )
        if appearance_distance < 0.0:
            raise VisualRecoveryRefusal(
                "post-promotion recovery association appearance is invalid"
            )
    if (
        not min_association_confidence <= confidence <= 1.0
        or not math.isclose(
            confidence,
            sample_confidence,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or cost < 0.0
        or not math.isclose(
            confidence,
            max(
                0.0,
                min(
                    1.0,
                    1.0 - cost / RECOVERY_TRACKER_MAX_ASSIGNMENT_COST,
                ),
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or residual < 0.0
        or not 0.0 <= bbox_iou <= 1.0
        or not 0.0 <= clipping_continuity <= 1.0
        or not math.isclose(
            temporal_consistency,
            1.0 / expected_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not all(
            math.isfinite(value)
            for value in (log_width, log_height, log_area)
        )
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery association authority is insufficient"
        )
    return evidence


def _require_clean_history_segment(
    track: VisualTrack,
    *,
    start_index: int,
    sample_count: int,
    min_detection_confidence: float,
    min_association_confidence: float,
    min_span_s: float,
) -> tuple[VisualTrackSample, ...]:
    if (
        type(start_index) is not int
        or type(sample_count) is not int
        or sample_count <= 0
        or start_index < 1
        or start_index + sample_count > len(track.history)
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery clean history is unavailable"
        )
    samples = track.history[start_index : start_index + sample_count]
    token = samples[-1].token
    if type(token) is not CameraFrameToken:
        raise VisualRecoveryRefusal(
            "post-promotion recovery clean history lacks a token"
        )
    span_s = (
        samples[-1].observation_monotonic_ns
        - samples[0].observation_monotonic_ns
    ) / 1_000_000_000.0
    if span_s < min_span_s:
        raise VisualRecoveryRefusal(
            "post-promotion recovery clean history span is insufficient"
        )
    for offset, sample in enumerate(samples):
        previous = track.history[start_index + offset - 1]
        _sample_geometry(
            sample,
            stream_id=token.stream_id,
            generation=token.generation,
            min_detection_confidence=min_detection_confidence,
            min_association_confidence=min_association_confidence,
        )
        _require_accepted_association(
            track,
            previous,
            sample,
            expected_missed_frames=0,
            min_association_confidence=min_association_confidence,
        )
    return samples


def _require_promotion_identity_bridge(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
) -> ReacquisitionBridgeAdmission | None:
    credit_length = transition.promoted_history_length_at_credit
    promotion_length = transition.history_length_after_promotion
    pretransition_tokens = transition.pretransition_frame_tokens
    if (
        len(pretransition_tokens)
        < RECOVERY_PRETRANSITION_VISIBILITY_SAMPLE_COUNT
        or promotion_length > len(track.history)
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery transition visibility proof is invalid"
        )
    transition_epoch_start = credit_length - len(pretransition_tokens)
    current_epoch_start = len(track.history) - track.consecutive_frame_count
    if transition_epoch_start != current_epoch_start:
        raise VisualRecoveryRefusal(
            "post-promotion recovery visibility epoch is inconsistent"
        )
    bridge_index = (
        credit_length - RECOVERY_PRETRANSITION_VISIBILITY_SAMPLE_COUNT
        if current_epoch_start == 0
        else current_epoch_start
    )
    if bridge_index < RECOVERY_PRE_GAP_HISTORY_SAMPLE_COUNT + 1:
        raise VisualRecoveryRefusal(
            "post-promotion recovery lacks established pre-gap identity"
        )
    stable_tail_start = credit_length - RECOVERY_HISTORY_SAMPLE_COUNT
    if stable_tail_start <= bridge_index:
        raise VisualRecoveryRefusal(
            "post-promotion recovery stable epoch is insufficient"
        )

    _require_clean_history_segment(
        track,
        start_index=bridge_index - RECOVERY_PRE_GAP_HISTORY_SAMPLE_COUNT,
        sample_count=RECOVERY_PRE_GAP_HISTORY_SAMPLE_COUNT,
        min_detection_confidence=(
            RECOVERY_MIN_PRE_GAP_DETECTION_CONFIDENCE
        ),
        min_association_confidence=(
            RECOVERY_MIN_PRE_GAP_ASSOCIATION_CONFIDENCE
        ),
        min_span_s=RECOVERY_MIN_PRE_GAP_HISTORY_SPAN_S,
    )
    _require_clean_history_segment(
        track,
        start_index=stable_tail_start,
        sample_count=RECOVERY_HISTORY_SAMPLE_COUNT,
        min_detection_confidence=RECOVERY_MIN_DETECTION_CONFIDENCE,
        min_association_confidence=RECOVERY_MIN_ASSOCIATION_CONFIDENCE,
        min_span_s=RECOVERY_MIN_HISTORY_SPAN_S,
    )
    if current_epoch_start > 0:
        _require_clean_history_segment(
            track,
            start_index=bridge_index + 1,
            sample_count=promotion_length - bridge_index - 1,
            min_detection_confidence=RECOVERY_MIN_DETECTION_CONFIDENCE,
            min_association_confidence=(
                RECOVERY_MIN_ASSOCIATION_CONFIDENCE
            ),
            min_span_s=RECOVERY_MIN_HISTORY_SPAN_S,
        )

    predecessor = track.history[bridge_index - 1]
    bridge = track.history[bridge_index]
    bridge_evidence = bridge.accepted_association
    if type(bridge_evidence) is not AssociationEvidence:
        raise VisualRecoveryRefusal(
            "post-promotion recovery reacquisition bridge is absent"
        )
    missed_frames = bridge_evidence.missed_frame_count_before_association
    if current_epoch_start == 0:
        if missed_frames != 0:
            raise VisualRecoveryRefusal(
                "post-promotion recovery continuous identity is inconsistent"
            )
        _sample_geometry(
            bridge,
            stream_id=bridge.token.stream_id,
            generation=bridge.token.generation,
        )
        _require_accepted_association(
            track,
            predecessor,
            bridge,
            expected_missed_frames=0,
            min_association_confidence=(
                RECOVERY_MIN_ASSOCIATION_CONFIDENCE
            ),
        )
        return None

    if (
        type(missed_frames) is not int
        or not 1
        <= missed_frames
        <= RECOVERY_MAX_REACQUISITION_MISSED_FRAMES
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery reacquisition gap is outside bounds"
        )
    _sample_geometry(
        bridge,
        stream_id=bridge.token.stream_id,
        generation=bridge.token.generation,
        min_detection_confidence=(
            RECOVERY_MIN_REACQUISITION_DETECTION_CONFIDENCE
        ),
        min_association_confidence=(
            RECOVERY_MIN_REACQUISITION_ASSOCIATION_CONFIDENCE
        ),
    )
    evidence = _require_accepted_association(
        track,
        predecessor,
        bridge,
        expected_missed_frames=missed_frames,
        min_association_confidence=(
            RECOVERY_MIN_REACQUISITION_ASSOCIATION_CONFIDENCE
        ),
        max_unobserved_publications=(
            RECOVERY_MAX_REACQUISITION_UNOBSERVED_PUBLICATIONS
        ),
    )
    tracker_delta = (
        bridge.tracker_frame_sequence - predecessor.tracker_frame_sequence
    )
    publication_delta = (
        bridge.token.publication_sequence
        - predecessor.token.publication_sequence
    )
    unobserved_publication_count = publication_delta - tracker_delta
    observation_gap_s = evidence.observation_gap_ns / 1_000_000_000.0
    assert evidence.publication_gap_ns is not None
    publication_gap_s = evidence.publication_gap_ns / 1_000_000_000.0
    if (
        publication_delta > RECOVERY_MAX_REACQUISITION_PUBLICATION_DELTA
        or unobserved_publication_count < 0
        or unobserved_publication_count
        > RECOVERY_MAX_REACQUISITION_UNOBSERVED_PUBLICATIONS
        or observation_gap_s > RECOVERY_MAX_REACQUISITION_GAP_S
        or publication_gap_s > RECOVERY_MAX_REACQUISITION_GAP_S
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery reacquisition timing is unsafe"
        )
    cost = float(evidence.cost)
    residual = float(evidence.predicted_center_residual_norm)
    bbox_iou = float(evidence.bbox_iou)
    log_width = float(evidence.log_width_change)
    log_height = float(evidence.log_height_change)
    log_area = float(evidence.log_area_residual)
    if (
        cost > RECOVERY_MAX_REACQUISITION_ASSOCIATION_COST
        or bbox_iou < RECOVERY_MIN_REACQUISITION_BBOX_IOU
        or residual
        > RECOVERY_MAX_REACQUISITION_CENTER_RESIDUAL_NORM
        or abs(log_width)
        > RECOVERY_MAX_REACQUISITION_ABS_LOG_WIDTH_CHANGE
        or abs(log_height)
        > RECOVERY_MAX_REACQUISITION_ABS_LOG_HEIGHT_CHANGE
        or abs(log_area)
        > RECOVERY_MAX_REACQUISITION_ABS_LOG_AREA_RESIDUAL
        or not math.isclose(
            float(evidence.clipping_continuity),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery reacquisition geometry is unsafe"
        )
    predecessor_width = (
        predecessor.bbox_norm[2] - predecessor.bbox_norm[0]
    )
    predecessor_height = (
        predecessor.bbox_norm[3] - predecessor.bbox_norm[1]
    )
    bridge_width = bridge.bbox_norm[2] - bridge.bbox_norm[0]
    bridge_height = bridge.bbox_norm[3] - bridge.bbox_norm[1]
    intersection_left = max(predecessor.bbox_norm[0], bridge.bbox_norm[0])
    intersection_top = max(predecessor.bbox_norm[1], bridge.bbox_norm[1])
    intersection_right = min(predecessor.bbox_norm[2], bridge.bbox_norm[2])
    intersection_bottom = min(
        predecessor.bbox_norm[3],
        bridge.bbox_norm[3],
    )
    intersection_area = max(
        0.0,
        intersection_right - intersection_left,
    ) * max(0.0, intersection_bottom - intersection_top)
    predecessor_area = predecessor_width * predecessor_height
    bridge_area = bridge_width * bridge_height
    direct_bbox_iou = intersection_area / (
        predecessor_area + bridge_area - intersection_area
    )
    if (
        predecessor_width <= 0.0
        or predecessor_height <= 0.0
        or bridge_width <= 0.0
        or bridge_height <= 0.0
        or direct_bbox_iou < RECOVERY_MIN_REACQUISITION_DIRECT_BBOX_IOU
        or not math.isclose(
            log_width,
            math.log(bridge_width / predecessor_width),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            log_height,
            math.log(bridge_height / predecessor_height),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery reacquisition contour is inconsistent"
        )
    horizontal_rate = (
        bridge.center_norm[0] - predecessor.center_norm[0]
    ) / observation_gap_s
    vertical_rate = (
        bridge.center_norm[1] - predecessor.center_norm[1]
    ) / observation_gap_s
    log_scale_rate = math.log(
        bridge.apparent_scale / predecessor.apparent_scale
    ) / observation_gap_s
    if (
        abs(horizontal_rate)
        > RECOVERY_MAX_REACQUISITION_CENTER_RATE_NORM_S
        or abs(vertical_rate)
        > RECOVERY_MAX_REACQUISITION_CENTER_RATE_NORM_S
        or abs(log_scale_rate)
        > RECOVERY_MAX_REACQUISITION_LOG_SCALE_RATE_S
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery reacquisition motion is unsafe"
        )
    return ReacquisitionBridgeAdmission(
        predecessor_token=predecessor.token,
        reacquisition_token=bridge.token,
        missed_frame_count=missed_frames,
        tracker_frame_delta=tracker_delta,
        publication_delta=publication_delta,
        unobserved_publication_count=unobserved_publication_count,
        observation_gap_s=observation_gap_s,
        publication_gap_s=publication_gap_s,
        association_confidence=float(evidence.confidence),
        association_cost=cost,
        predicted_center_residual_norm=residual,
        bbox_iou=bbox_iou,
        direct_bbox_iou=direct_bbox_iou,
        log_width_change=log_width,
        log_height_change=log_height,
        log_area_residual=log_area,
        average_horizontal_rate_norm_s=horizontal_rate,
        average_vertical_rate_norm_s=vertical_rate,
        average_log_scale_rate_s=log_scale_rate,
    )


def _projection_horizon_s(
    *,
    latest_observation_monotonic_ns: int,
    now_monotonic_ns: int,
) -> tuple[float, float]:
    if now_monotonic_ns < latest_observation_monotonic_ns:
        raise VisualRecoveryRefusal(
            "post-promotion recovery observation is future-dated"
        )
    observation_age_s = (
        now_monotonic_ns - latest_observation_monotonic_ns
    ) / 1_000_000_000.0
    horizon_s = max(
        RECOVERY_MIN_PROJECTION_HORIZON_S,
        observation_age_s + RECOVERY_COMMAND_RESPONSE_HORIZON_S,
    )
    if horizon_s > RECOVERY_MAX_PROJECTION_HORIZON_S:
        raise VisualRecoveryRefusal(
            "post-promotion recovery projection horizon is exhausted"
        )
    return observation_age_s, horizon_s


def _validate_recent_history_and_project(
    track: VisualTrack,
    *,
    projection_horizon_s: float,
) -> _HistoryProjection:
    if len(track.history) < RECOVERY_HISTORY_SAMPLE_COUNT:
        raise VisualRecoveryRefusal(
            "post-promotion recovery lacks exact track history"
        )
    samples = track.history[-RECOVERY_HISTORY_SAMPLE_COUNT:]
    if samples[-1].token != track.latest_token:
        raise VisualRecoveryRefusal(
            "post-promotion recovery history does not end at the latest token"
        )
    token = track.latest_token
    if type(token.stream_id) is not str or not token.stream_id:
        raise VisualRecoveryRefusal(
            "post-promotion recovery latest token lacks a camera stream"
        )
    bboxes = tuple(
        _sample_geometry(
            sample,
            stream_id=token.stream_id,
            generation=token.generation,
        )
        for sample in samples
    )
    history_span_s = (
        samples[-1].observation_monotonic_ns
        - samples[0].observation_monotonic_ns
    ) / 1_000_000_000.0
    if history_span_s < RECOVERY_MIN_HISTORY_SPAN_S:
        raise VisualRecoveryRefusal(
            "post-promotion recovery stable history span is insufficient"
        )
    if type(track.center_norm) is not tuple or len(track.center_norm) != 2:
        raise VisualRecoveryRefusal(
            "post-promotion recovery filtered center state is invalid"
        )
    horizontal = _finite(track.center_norm[0], "horizontal error")
    vertical = _finite(track.center_norm[1], "vertical error")
    if abs(horizontal) > RECOVERY_MAX_ABS_X_NORM:
        raise VisualRecoveryRefusal(
            "post-promotion recovery horizontal position is unsafe"
        )
    if abs(vertical) > RECOVERY_MAX_ABS_Y_NORM:
        raise VisualRecoveryRefusal(
            "post-promotion recovery vertical position is unsafe"
        )
    raw_x_rates: list[float] = []
    raw_y_rates: list[float] = []
    raw_scale_rates: list[float] = []
    raw_log_width_rates: list[float] = []
    raw_log_height_rates: list[float] = []
    left_rates: list[float] = []
    top_rates: list[float] = []
    right_rates: list[float] = []
    bottom_rates: list[float] = []
    for index, (previous, sample) in enumerate(
        zip(samples, samples[1:]),
        start=1,
    ):
        previous_sequence = previous.token.publication_sequence
        sample_sequence = sample.token.publication_sequence
        assert previous_sequence is not None
        assert sample_sequence is not None
        if (
            sample.tracker_frame_sequence
            != previous.tracker_frame_sequence + 1
            or sample_sequence != previous_sequence + 1
            or sample.observation_monotonic_ns
            <= previous.observation_monotonic_ns
            or sample.publication_monotonic_ns
            <= previous.publication_monotonic_ns
        ):
            raise VisualRecoveryRefusal(
                "post-promotion recovery history is not contiguous"
            )
        dt_s = (
            sample.observation_monotonic_ns
            - previous.observation_monotonic_ns
        ) / 1_000_000_000.0
        if not RECOVERY_MIN_FRAME_DT_S <= dt_s <= RECOVERY_MAX_FRAME_DT_S:
            raise VisualRecoveryRefusal(
                "post-promotion recovery history timing is outside bounds"
            )
        previous_bbox = bboxes[index - 1]
        bbox = bboxes[index]
        previous_width = previous_bbox[2] - previous_bbox[0]
        previous_height = previous_bbox[3] - previous_bbox[1]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        raw_x_rates.append(
            (sample.center_norm[0] - previous.center_norm[0]) / dt_s
        )
        raw_y_rates.append(
            (sample.center_norm[1] - previous.center_norm[1]) / dt_s
        )
        raw_scale_rates.append(
            math.log(sample.apparent_scale / previous.apparent_scale) / dt_s
        )
        raw_log_width_rates.append(math.log(width / previous_width) / dt_s)
        raw_log_height_rates.append(math.log(height / previous_height) / dt_s)
        left_rates.append((bbox[0] - previous_bbox[0]) / dt_s)
        top_rates.append((bbox[1] - previous_bbox[1]) / dt_s)
        right_rates.append((bbox[2] - previous_bbox[2]) / dt_s)
        bottom_rates.append((bbox[3] - previous_bbox[3]) / dt_s)

    rate_groups = (
        raw_x_rates,
        raw_y_rates,
        raw_scale_rates,
        raw_log_width_rates,
        raw_log_height_rates,
        left_rates,
        top_rates,
        right_rates,
        bottom_rates,
    )
    if not all(
        math.isfinite(value)
        for group in rate_groups
        for value in group
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery raw history rate is non-finite"
        )
    max_raw_x = max(abs(value) for value in raw_x_rates)
    max_raw_y = max(abs(value) for value in raw_y_rates)
    max_raw_scale = max(abs(value) for value in raw_scale_rates)
    max_raw_log_width = max(abs(value) for value in raw_log_width_rates)
    max_raw_log_height = max(abs(value) for value in raw_log_height_rates)
    max_positive_scale = max(0.0, *raw_scale_rates)
    if (
        max_raw_x > RECOVERY_MAX_RAW_CENTER_RATE_NORM_S
        or max_raw_y > RECOVERY_MAX_RAW_CENTER_RATE_NORM_S
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery raw center motion is unsafe"
        )
    if max_raw_scale > RECOVERY_MAX_RAW_LOG_SCALE_RATE_S:
        raise VisualRecoveryRefusal(
            "post-promotion recovery raw scale motion is unsafe"
        )
    if (
        max_raw_log_width > RECOVERY_MAX_RAW_LOG_DIMENSION_RATE_S
        or max_raw_log_height > RECOVERY_MAX_RAW_LOG_DIMENSION_RATE_S
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery contour deformation is unsafe"
        )
    history_start_index = len(track.history) - RECOVERY_HISTORY_SAMPLE_COUNT
    if history_start_index < 1:
        raise VisualRecoveryRefusal(
            "post-promotion recovery stable history lacks identity provenance"
        )
    for offset, sample in enumerate(samples):
        _require_accepted_association(
            track,
            track.history[history_start_index + offset - 1],
            sample,
            expected_missed_frames=0,
            min_association_confidence=(
                RECOVERY_MIN_ASSOCIATION_CONFIDENCE
            ),
        )

    if (
        type(track.center_velocity_norm_s) is not tuple
        or len(track.center_velocity_norm_s) != 2
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery filtered center state is invalid"
        )
    filtered_x_rate = _finite(
        track.center_velocity_norm_s[0],
        "filtered horizontal rate",
    )
    filtered_y_rate = _finite(
        track.center_velocity_norm_s[1],
        "filtered vertical rate",
    )
    filtered_scale_rate = _finite(
        track.log_scale_rate_s,
        "filtered scale rate",
    )
    latest_apparent_scale = _finite(
        track.apparent_scale,
        "latest apparent scale",
    )
    if latest_apparent_scale <= 0.0:
        raise VisualRecoveryRefusal(
            "post-promotion recovery latest apparent scale is invalid"
        )
    projected_abs_x = (
        abs(horizontal)
        + max(max_raw_x, abs(filtered_x_rate)) * projection_horizon_s
        + RECOVERY_CENTER_PADDING_X_NORM
    )
    projected_abs_y = (
        abs(vertical)
        + max(max_raw_y, abs(filtered_y_rate)) * projection_horizon_s
        + RECOVERY_CENTER_PADDING_Y_NORM
    )
    if projected_abs_x > POST_PROMOTION_ENTRY_MAX_ABS_X_NORM:
        raise VisualRecoveryRefusal(
            "post-promotion recovery horizontal projection is unsafe"
        )
    if projected_abs_y > RECOVERY_MAX_PROJECTED_ABS_Y_NORM:
        raise VisualRecoveryRefusal(
            "post-promotion recovery vertical projection is unsafe"
        )

    latest_bbox = bboxes[-1]
    projected_bbox = (
        latest_bbox[0]
        - max(0.0, *(-value for value in left_rates))
        * projection_horizon_s
        - RECOVERY_BBOX_PADDING_X_NORM,
        latest_bbox[1]
        - max(0.0, *(-value for value in top_rates))
        * projection_horizon_s
        - RECOVERY_BBOX_PADDING_Y_NORM,
        latest_bbox[2]
        + max(0.0, *right_rates) * projection_horizon_s
        + RECOVERY_BBOX_PADDING_X_NORM,
        latest_bbox[3]
        + max(0.0, *bottom_rates) * projection_horizon_s
        + RECOVERY_BBOX_PADDING_Y_NORM,
    )
    projected_width = projected_bbox[2] - projected_bbox[0]
    projected_height = projected_bbox[3] - projected_bbox[1]
    projected_area = projected_width * projected_height
    projected_apparent_scale = latest_apparent_scale * math.exp(
        max(0.0, max_positive_scale, filtered_scale_rate)
        * projection_horizon_s
    )
    if (
        projected_width >= RECOVERY_MAX_PROJECTED_WIDTH_NORM
        or projected_height >= RECOVERY_MAX_PROJECTED_HEIGHT_NORM
        or projected_area >= RECOVERY_MAX_PROJECTED_AREA_NORM
        or projected_apparent_scale
        >= RECOVERY_MAX_PROJECTED_APPARENT_SCALE
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery projected scale is unsafe"
        )
    if (
        projected_bbox[0] < RECOVERY_MIN_PROJECTED_EDGE_MARGIN_X_NORM
        or projected_bbox[1] < RECOVERY_MIN_PROJECTED_EDGE_MARGIN_Y_NORM
        or projected_bbox[2]
        > 1.0 - RECOVERY_MIN_PROJECTED_EDGE_MARGIN_X_NORM
        or projected_bbox[3]
        > 1.0 - RECOVERY_MIN_PROJECTED_EDGE_MARGIN_Y_NORM
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery projected bbox lacks edge margin"
        )
    return _HistoryProjection(
        tokens=tuple(sample.token for sample in samples),
        latest_observation_monotonic_ns=samples[-1].observation_monotonic_ns,
        latest_publication_monotonic_ns=int(
            samples[-1].publication_monotonic_ns
        ),
        stable_history_span_s=history_span_s,
        min_detection_confidence=min(
            float(sample.confidence) for sample in samples
        ),
        min_association_confidence=min(
            float(sample.association_confidence) for sample in samples
        ),
        max_raw_horizontal_rate_s=max_raw_x,
        max_raw_vertical_rate_down_s=max_raw_y,
        max_raw_log_scale_rate_s=max_raw_scale,
        max_raw_log_width_rate_s=max_raw_log_width,
        max_raw_log_height_rate_s=max_raw_log_height,
        max_positive_raw_log_scale_rate_s=max_positive_scale,
        projected_abs_horizontal_error=projected_abs_x,
        projected_abs_vertical_error_image_down=projected_abs_y,
        projected_bbox_norm_ltrb=projected_bbox,
        projected_width_norm=projected_width,
        projected_height_norm=projected_height,
        projected_area_norm=projected_area,
        projected_apparent_scale=projected_apparent_scale,
    )


def require_transition_recovery_admission(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
    *,
    tracker_time_basis_id: str,
    measured_pitch_rad: float,
    now_monotonic_ns: int,
    promotion_history_authority: PromotionHistoryAuthority | None = None,
) -> TransitionRecoveryAdmission:
    """Admit one transition-anchor-derived, no-advance recovery response."""

    if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
        raise VisualRecoveryRefusal(
            "post-promotion recovery requires an exact monotonic time"
        )
    measured_pitch = _finite_pitch(measured_pitch_rad)
    race_ns, race_sequence = _require_live_transition_authority(
        track,
        transition,
        tracker_time_basis_id=tracker_time_basis_id,
    )
    _require_exact_transition_anchor(track, transition)
    reacquisition_bridge = _require_promotion_identity_bridge(
        track,
        transition,
    )
    if now_monotonic_ns < race_ns:
        raise VisualRecoveryRefusal(
            "post-promotion recovery time precedes authoritative credit"
        )
    start_delay_s = (now_monotonic_ns - race_ns) / 1_000_000_000.0
    if start_delay_s > RECOVERY_MAX_START_DELAY_AFTER_CREDIT_S:
        raise VisualRecoveryRefusal(
            "post-promotion recovery start exceeded its credit lease"
        )
    try:
        target = VisualTarget.from_visual_track(
            track,
            expected_gate_index=transition.to_gate_index,
        )
    except (TypeError, ValueError) as exc:
        raise VisualRecoveryRefusal(
            f"post-promotion recovery target is invalid: {exc}"
        ) from exc
    if (
        target.ambiguous
        or target.clipped
        or target.center_censored
        or target.horizontal_geometry_censored
        or target.vertical_geometry_censored
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery target is ambiguous or censored"
        )
    if measured_pitch < POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD:
        raise VisualRecoveryRefusal(
            "post-promotion recovery pitch cannot reach braking attitude"
        )
    horizontal = float(target.normalized_x)
    vertical = float(target.normalized_y_down)
    if (
        abs(horizontal) > RECOVERY_MAX_ABS_X_NORM
        or abs(horizontal) > POST_PROMOTION_ENTRY_MAX_ABS_X_NORM
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery horizontal position is unsafe"
        )
    if (
        abs(vertical) > RECOVERY_MAX_ABS_Y_NORM
        or abs(vertical) > POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery vertical position is unsafe"
        )
    filtered_x_rate = float(target.normalized_x_rate_s)
    filtered_y_rate = float(target.normalized_y_rate_down_s)
    filtered_scale_rate = float(target.log_scale_rate_s)
    if (
        abs(filtered_x_rate) > RECOVERY_MAX_FILTERED_CENTER_RATE_NORM_S
        or abs(filtered_y_rate) > RECOVERY_MAX_FILTERED_CENTER_RATE_NORM_S
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery filtered center motion is unsafe"
        )
    if (
        abs(filtered_scale_rate)
        > RECOVERY_MAX_FILTERED_LOG_SCALE_RATE_S
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery filtered scale motion is unsafe"
        )

    credit_sample = track.history[
        transition.promoted_history_length_at_credit - 1
    ]
    latest = track.history[-1]
    if (
        type(credit_sample.publication_monotonic_ns) is not int
        or type(credit_sample.observation_monotonic_ns) is not int
        or type(latest.observation_monotonic_ns) is not int
        or latest.observation_monotonic_ns < 0
        or type(latest.publication_monotonic_ns) is not int
        or latest.publication_monotonic_ns > now_monotonic_ns
    ):
        raise VisualRecoveryRefusal(
            "post-promotion recovery anchor lacks receiver timing"
        )
    credit_age_s = (
        race_ns - credit_sample.publication_monotonic_ns
    ) / 1_000_000_000.0
    if not 0.0 <= credit_age_s <= RECOVERY_MAX_ANCHOR_CREDIT_AGE_S:
        raise VisualRecoveryRefusal(
            "post-promotion recovery anchor is stale at race credit"
        )
    observation_age_s, projection_horizon_s = _projection_horizon_s(
        latest_observation_monotonic_ns=latest.observation_monotonic_ns,
        now_monotonic_ns=now_monotonic_ns,
    )
    projection = _validate_recent_history_and_project(
        track,
        projection_horizon_s=projection_horizon_s,
    )
    _require_promotion_history_digest(
        track,
        transition,
        authority=promotion_history_authority,
    )
    return TransitionRecoveryAdmission(
        track_id=track.track_id,
        credit_prefix_token=(
            transition.promoted_latest_token_before_credit
        ),
        promotion_anchor_token=(
            transition.promoted_latest_token_at_promotion
        ),
        history_tokens=projection.tokens,
        race_status_sequence=race_sequence,
        race_received_monotonic_ns=race_ns,
        credit_prefix_publication_monotonic_ns=(
            credit_sample.publication_monotonic_ns
        ),
        credit_prefix_age_s=credit_age_s,
        promotion_anchor_observation_monotonic_ns=(
            projection.latest_observation_monotonic_ns
        ),
        promotion_anchor_publication_monotonic_ns=(
            projection.latest_publication_monotonic_ns
        ),
        promotion_anchor_publication_delta_from_credit_s=(
            projection.latest_publication_monotonic_ns - race_ns
        )
        / 1_000_000_000.0,
        recovery_start_delay_s=start_delay_s,
        observation_age_s=observation_age_s,
        projection_horizon_s=projection_horizon_s,
        stable_history_span_s=projection.stable_history_span_s,
        min_history_detection_confidence=(
            projection.min_detection_confidence
        ),
        min_history_association_confidence=(
            projection.min_association_confidence
        ),
        promotion_identity_sha256=transition.promoted_history_sha256,
        reacquisition_bridge=reacquisition_bridge,
        horizontal_error=horizontal,
        vertical_error_image_down=vertical,
        filtered_horizontal_rate_s=filtered_x_rate,
        filtered_vertical_rate_down_s=filtered_y_rate,
        filtered_log_scale_rate_s=filtered_scale_rate,
        max_raw_horizontal_rate_s=(
            projection.max_raw_horizontal_rate_s
        ),
        max_raw_vertical_rate_down_s=(
            projection.max_raw_vertical_rate_down_s
        ),
        max_raw_log_scale_rate_s=projection.max_raw_log_scale_rate_s,
        max_raw_log_width_rate_s=projection.max_raw_log_width_rate_s,
        max_raw_log_height_rate_s=projection.max_raw_log_height_rate_s,
        max_positive_raw_log_scale_rate_s=(
            projection.max_positive_raw_log_scale_rate_s
        ),
        projected_abs_horizontal_error=(
            projection.projected_abs_horizontal_error
        ),
        projected_abs_vertical_error_image_down=(
            projection.projected_abs_vertical_error_image_down
        ),
        projected_bbox_norm_ltrb=projection.projected_bbox_norm_ltrb,
        projected_width_norm=projection.projected_width_norm,
        projected_height_norm=projection.projected_height_norm,
        projected_area_norm=projection.projected_area_norm,
        projected_apparent_scale=projection.projected_apparent_scale,
        measured_pitch_rad=measured_pitch,
    )


def require_recovery_continuation(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
    *,
    previous_token: CameraFrameToken,
    tracker_time_basis_id: str,
    measured_pitch_rad: float,
    recovery_started_monotonic_ns: int,
    now_monotonic_ns: int,
    promotion_history_authority: PromotionHistoryAuthority | None = None,
) -> RecoveryContinuationAdmission:
    """Revalidate one exactly-next post-credit recovery publication."""

    if type(previous_token) is not CameraFrameToken:
        raise VisualRecoveryRefusal(
            "recovery continuation requires an exact previous token"
        )
    if (
        type(recovery_started_monotonic_ns) is not int
        or recovery_started_monotonic_ns < 0
        or type(now_monotonic_ns) is not int
        or now_monotonic_ns < recovery_started_monotonic_ns
    ):
        raise VisualRecoveryRefusal(
            "recovery continuation requires a coherent recovery clock"
        )
    race_ns, _race_sequence = _require_live_transition_authority(
        track,
        transition,
        tracker_time_basis_id=tracker_time_basis_id,
    )
    reacquisition_bridge = _require_promotion_identity_bridge(
        track,
        transition,
    )
    if (
        recovery_started_monotonic_ns < race_ns
        or (
            recovery_started_monotonic_ns - race_ns
        ) / 1_000_000_000.0
        > RECOVERY_MAX_START_DELAY_AFTER_CREDIT_S
    ):
        raise VisualRecoveryRefusal(
            "recovery continuation start is outside the credit lease"
        )
    recovery_elapsed_s = (
        now_monotonic_ns - recovery_started_monotonic_ns
    ) / 1_000_000_000.0
    if recovery_elapsed_s > RECOVERY_HARD_DURATION_S:
        raise VisualRecoveryRefusal(
            "recovery continuation exceeded its hard duration"
        )
    if (
        not track.visible
        or track.missed_frame_count != 0
        or track.ambiguous
    ):
        raise VisualRecoveryRefusal(
            "recovery continuation lost promoted-track authority"
        )
    token = track.latest_token
    previous_sequence = previous_token.publication_sequence
    token_sequence = token.publication_sequence
    if (
        token.stream_id != previous_token.stream_id
        or token.generation != previous_token.generation
        or type(previous_sequence) is not int
        or type(token_sequence) is not int
        or token_sequence != previous_sequence + 1
        or len(track.history) < 2
        or track.history[-2].token != previous_token
    ):
        raise VisualRecoveryRefusal(
            "recovery continuation publication did not advance exactly"
        )
    latest = track.history[-1]
    if (
        type(latest.observation_monotonic_ns) is not int
        or latest.observation_monotonic_ns <= race_ns
        or type(latest.publication_monotonic_ns) is not int
        or latest.publication_monotonic_ns <= race_ns
    ):
        raise VisualRecoveryRefusal(
            "recovery continuation observation is not post-credit"
        )
    try:
        target = VisualTarget.from_visual_track(
            track,
            expected_gate_index=transition.to_gate_index,
        )
        capture = require_visual_alignment_capture_entry(
            target,
            measured_pitch_rad=measured_pitch_rad,
        )
    except (TypeError, ValueError) as exc:
        raise VisualRecoveryRefusal(
            f"recovery continuation predictive admission failed: {exc}"
        ) from exc
    observation_age_s, projection_horizon_s = _projection_horizon_s(
        latest_observation_monotonic_ns=latest.observation_monotonic_ns,
        now_monotonic_ns=now_monotonic_ns,
    )
    if observation_age_s > RECOVERY_MAX_CONTINUATION_AGE_S:
        raise VisualRecoveryRefusal(
            "recovery continuation frame is stale"
        )
    projection = _validate_recent_history_and_project(
        track,
        projection_horizon_s=projection_horizon_s,
    )
    _require_promotion_history_digest(
        track,
        transition,
        authority=promotion_history_authority,
    )
    return RecoveryContinuationAdmission(
        track_id=track.track_id,
        previous_token=previous_token,
        frame_token=token,
        observation_age_s=observation_age_s,
        recovery_elapsed_s=recovery_elapsed_s,
        projection_horizon_s=projection_horizon_s,
        stable_history_span_s=projection.stable_history_span_s,
        min_history_detection_confidence=(
            projection.min_detection_confidence
        ),
        min_history_association_confidence=(
            projection.min_association_confidence
        ),
        promotion_identity_sha256=transition.promoted_history_sha256,
        reacquisition_bridge=reacquisition_bridge,
        capture=capture,
        max_raw_horizontal_rate_s=(
            projection.max_raw_horizontal_rate_s
        ),
        max_raw_vertical_rate_down_s=(
            projection.max_raw_vertical_rate_down_s
        ),
        max_raw_log_scale_rate_s=projection.max_raw_log_scale_rate_s,
        max_raw_log_width_rate_s=projection.max_raw_log_width_rate_s,
        max_raw_log_height_rate_s=projection.max_raw_log_height_rate_s,
        projected_abs_horizontal_error=(
            projection.projected_abs_horizontal_error
        ),
        projected_abs_vertical_error_image_down=(
            projection.projected_abs_vertical_error_image_down
        ),
        projected_bbox_norm_ltrb=projection.projected_bbox_norm_ltrb,
        projected_width_norm=projection.projected_width_norm,
        projected_height_norm=projection.projected_height_norm,
        projected_area_norm=projection.projected_area_norm,
        projected_apparent_scale=projection.projected_apparent_scale,
    )


__all__ = [
    "RECOVERY_COMMAND_RESPONSE_HORIZON_S",
    "RECOVERY_HARD_DURATION_S",
    "RECOVERY_HISTORY_SAMPLE_COUNT",
    "RECOVERY_MAX_ABS_X_NORM",
    "RECOVERY_MAX_ABS_Y_NORM",
    "RECOVERY_MAX_COMMAND_RATE_RAD_S",
    "RECOVERY_MAX_COMMANDS",
    "RECOVERY_MAX_CONTINUATION_AGE_S",
    "RECOVERY_MAX_FRESH_FRAMES",
    "RECOVERY_MAX_POSTCREDIT_PROMOTION_SAMPLES",
    "RECOVERY_MAX_PROJECTED_ABS_Y_NORM",
    "RECOVERY_MAX_PROJECTION_HORIZON_S",
    "RECOVERY_MAX_START_DELAY_AFTER_CREDIT_S",
    "RECOVERY_MAX_THRUST",
    "RECOVERY_MAX_VALIDATION_TO_WIRE_DELAY_S",
    "RECOVERY_MAX_YAW_RATE_RAD_S",
    "RECOVERY_MIN_PROJECTED_EDGE_MARGIN_X_NORM",
    "RECOVERY_MIN_PROJECTED_EDGE_MARGIN_Y_NORM",
    "RECOVERY_MIN_HISTORY_SPAN_S",
    "RECOVERY_MIN_REACQUISITION_ASSOCIATION_CONFIDENCE",
    "RECOVERY_PRE_GAP_HISTORY_SAMPLE_COUNT",
    "RECOVERY_PRETRANSITION_VISIBILITY_SAMPLE_COUNT",
    "RECOVERY_MIN_PROJECTION_HORIZON_S",
    "RECOVERY_REQUIRED_STRICT_ENTRY_FRAMES",
    "PromotionHistoryAuthority",
    "ReacquisitionBridgeAdmission",
    "RecoveryContinuationAdmission",
    "TransitionRecoveryAdmission",
    "VisualRecoveryRefusal",
    "require_promotion_history_authority",
    "require_recovery_continuation",
    "require_transition_recovery_admission",
]
