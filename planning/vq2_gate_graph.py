"""Rolling camera-derived gate graph for VQ2 build 3385.

This module labels visual tracks only when a caller supplies authoritative race
status.  It never constructs a global metric map: graph edges contain normalized
image bearing/elevation, apparent scale, rates, confidence, and exact source
provenance.  Race credit confirms a transition after the fact; it does not
prevent pre-transition detection or erase the promoted track's history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualTrack,
    VisualTrackRole,
    VisualTrackSample,
    visual_track_history_sha256,
)

_MAX_PROMOTION_MISSED_CAMERA_PUBLICATIONS = 2
_ADJACENT_HANDOFF_MAX_GAP_NS = 100_000_000
_ADJACENT_HANDOFF_MAX_CREDIT_AGE_NS = 100_000_000
_ADJACENT_HANDOFF_MAX_CONFIRMATION_PUBLICATIONS = 12
_ADJACENT_HANDOFF_MIN_CURRENT_SCALE = 0.95
_ADJACENT_HANDOFF_MAX_NEXT_SCALE_RATIO = 0.50
_ADJACENT_HANDOFF_BBOX_EDGE_TOLERANCE = 0.01
_ADJACENT_HANDOFF_CONFIDENCE_FACTOR = 0.50
_ALL_FRAME_EDGES = (
    FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
)


class RaceStatusProvenanceBasis(str, Enum):
    """Authoritative live ingress or exact legacy capture ordering."""

    LIVE_INGRESS = "live_ingress"
    LEGACY_CAPTURE = "legacy_capture"


class GateGraphError(ValueError):
    """A supplied graph or race transition is not safe to accept."""


class AmbiguousGatePromotionError(GateGraphError):
    """Race credit cannot uniquely identify a pretracked next gate."""


class RequestedGatePromotionUnavailableError(GateGraphError):
    """The caller's reviewed next-track identity is no longer promotable."""


class GateReacquisitionNotReadyError(GateGraphError):
    """A credited-unbound gate has no compatible command-safe successor yet."""


class AmbiguousGateReacquisitionError(GateReacquisitionNotReadyError):
    """More than one compatible successor prevents fail-closed reacquisition."""


class GateGraphPhase(str, Enum):
    """Explicit visual-authority phase for the rolling graph."""

    INITIAL_UNBOUND = "initial_unbound"
    CURRENT_BOUND = "current_bound"
    CREDITED_UNBOUND = "credited_unbound"
    RACE_FINISHED = "race_finished"


@dataclass(frozen=True, slots=True)
class GateReacquisitionPending:
    """A soft navigation outcome with no visual command authority."""

    reason: str
    ambiguous: bool

    def __post_init__(self) -> None:
        if type(self.reason) is not str or not self.reason:
            raise TypeError("pending reacquisition reason must be non-empty")
        if type(self.ambiguous) is not bool:
            raise TypeError("pending ambiguity flag must be an exact bool")


class GateRelationshipBasis(str, Enum):
    """Exact observation basis for one rolling current-to-next relationship."""

    SIMULTANEOUS_IMAGE = "simultaneous_image"
    ADJACENT_PUBLICATION_HANDOFF = "adjacent_publication_handoff"


@dataclass(frozen=True, slots=True)
class AuthoritativeRaceStatusRef:
    """Caller-proved race status, preserving the evidence fields that exist.

    Live status requires ingress generation/sequence and a host-monotonic
    receipt time.  Historical July 18 capture rows did not record those fields,
    so ``LEGACY_CAPTURE`` instead requires exact event order, boot time, and wall
    event token.  The two bases cannot be relabelled as each other.
    """

    provenance_basis: RaceStatusProvenanceBasis
    session_id: str
    reset_epoch: int
    race_status_boot_ms: int
    active_gate_index: int
    race_finished: bool
    race_generation: Optional[int] = None
    race_status_sequence: Optional[int] = None
    received_monotonic_ns: Optional[int] = None
    host_clock_id: Optional[str] = None
    legacy_event_order: Optional[int] = None
    event_wall_time_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if type(self.provenance_basis) is not RaceStatusProvenanceBasis:
            raise TypeError(
                "provenance_basis must be an exact RaceStatusProvenanceBasis"
            )
        if type(self.session_id) is not str or not self.session_id:
            raise TypeError("session_id must be a non-empty exact string")
        for name in ("reset_epoch", "race_status_boot_ms", "active_gate_index"):
            _nonnegative_int(getattr(self, name), name)
        if type(self.race_finished) is not bool:
            raise TypeError("race_finished must be an exact bool")
        if self.provenance_basis is RaceStatusProvenanceBasis.LIVE_INGRESS:
            for name in (
                "race_generation",
                "race_status_sequence",
                "received_monotonic_ns",
            ):
                value = getattr(self, name)
                if value is None:
                    raise ValueError(f"live race status requires {name}")
                _nonnegative_int(value, name)
            if type(self.host_clock_id) is not str or not self.host_clock_id:
                raise TypeError("live race status requires a host_clock_id")
            if self.legacy_event_order is not None or self.event_wall_time_ns is not None:
                raise ValueError("live race status cannot carry legacy event fields")
        else:
            if (
                self.race_generation is not None
                or self.race_status_sequence is not None
                or self.received_monotonic_ns is not None
                or self.host_clock_id is not None
            ):
                raise ValueError("legacy race status cannot invent live ingress fields")
            if self.legacy_event_order is None or self.event_wall_time_ns is None:
                raise ValueError(
                    "legacy race status requires event order and wall-time token"
                )
            _nonnegative_int(self.legacy_event_order, "legacy_event_order")
            _nonnegative_int(self.event_wall_time_ns, "event_wall_time_ns")

    @classmethod
    def live(
        cls,
        *,
        session_id: str,
        reset_epoch: int,
        race_generation: int,
        race_status_sequence: int,
        race_status_boot_ms: int,
        active_gate_index: int,
        received_monotonic_ns: int,
        host_clock_id: str,
        race_finished: bool = False,
    ) -> "AuthoritativeRaceStatusRef":
        return cls(
            provenance_basis=RaceStatusProvenanceBasis.LIVE_INGRESS,
            session_id=session_id,
            reset_epoch=reset_epoch,
            race_generation=race_generation,
            race_status_sequence=race_status_sequence,
            race_status_boot_ms=race_status_boot_ms,
            active_gate_index=active_gate_index,
            received_monotonic_ns=received_monotonic_ns,
            host_clock_id=host_clock_id,
            race_finished=race_finished,
        )

    @classmethod
    def legacy_capture(
        cls,
        *,
        session_id: str,
        reset_epoch: int,
        legacy_event_order: int,
        event_wall_time_ns: int,
        race_status_boot_ms: int,
        active_gate_index: int,
        race_finished: bool = False,
    ) -> "AuthoritativeRaceStatusRef":
        return cls(
            provenance_basis=RaceStatusProvenanceBasis.LEGACY_CAPTURE,
            session_id=session_id,
            reset_epoch=reset_epoch,
            legacy_event_order=legacy_event_order,
            event_wall_time_ns=event_wall_time_ns,
            race_status_boot_ms=race_status_boot_ms,
            active_gate_index=active_gate_index,
            race_finished=race_finished,
        )


@dataclass(frozen=True, slots=True)
class RollingGateGraphConfig:
    """Conservative visual-identity gates; no field authorizes a command."""

    min_current_binding_frames: int = 3
    min_next_candidate_frames: int = 3
    min_track_confidence: float = 0.20
    min_association_confidence: float = 0.10
    min_relationship_confidence: float = 0.12
    candidate_selection_margin: float = 0.08
    relationship_history_limit: int = 256

    def __post_init__(self) -> None:
        for name in (
            "min_current_binding_frames",
            "min_next_candidate_frames",
            "relationship_history_limit",
        ):
            _positive_int(getattr(self, name), name)
        for name in (
            "min_track_confidence",
            "min_association_confidence",
            "min_relationship_confidence",
            "candidate_selection_margin",
        ):
            _finite(getattr(self, name), name, minimum=0.0, maximum=1.0)


@dataclass(frozen=True, slots=True)
class ObservedGateRelationship:
    """One rolling current-to-candidate relationship in normalized image space."""

    current_track_id: str
    next_track_id: str
    basis: GateRelationshipBasis
    current_anchor_token: CameraFrameToken
    next_anchor_token: CameraFrameToken
    anchor_publication_delta: int
    anchor_time_gap_ns: int
    first_token: CameraFrameToken
    latest_token: CameraFrameToken
    observation_count: int
    simultaneous_observation_count: int
    sequential_observation_count: int
    latest_tracker_frame_sequence: int
    current_bearing_norm: float
    current_elevation_norm: float
    next_bearing_norm: float
    next_elevation_norm: float
    relative_bearing_norm: float
    relative_elevation_norm: float
    current_apparent_scale: float
    next_apparent_scale: float
    current_log_scale_rate_s: float
    next_log_scale_rate_s: float
    observation_confidence: float
    current_center_censored: bool
    next_center_censored: bool
    fresh: bool
    contended: bool

    @property
    def geometry_degraded(self) -> bool:
        return self.current_center_censored or self.next_center_censored

    @property
    def relative_geometry_usable(self) -> bool:
        """Whether the relative image geometry came from an uncensored joint view."""

        return (
            self.basis is GateRelationshipBasis.SIMULTANEOUS_IMAGE
            and self.simultaneous_observation_count > 0
            and not self.geometry_degraded
        )


@dataclass(frozen=True, slots=True)
class NextGateCandidate:
    track_id: str
    score: float
    stable_frame_count: int
    first_token: CameraFrameToken
    latest_token: CameraFrameToken
    bearing_norm: float
    elevation_norm: float
    bearing_rate_norm_s: float
    elevation_rate_norm_s: float
    apparent_scale: float
    log_scale_rate_s: float
    confidence: float
    association_confidence: float
    center_censored: bool
    promotable: bool
    relationship: Optional[ObservedGateRelationship]


@dataclass(frozen=True, slots=True)
class ConfirmedGateTransition:
    """Authoritative race credit attached to an already-existing visual track."""

    from_gate_index: int
    to_gate_index: int
    retired_track_id: str
    promoted_track_id: str
    race_status: AuthoritativeRaceStatusRef
    camera_token_at_credit: CameraFrameToken
    promoted_first_token: CameraFrameToken
    promoted_latest_token_before_credit: CameraFrameToken
    promoted_history_length_at_credit: int
    promoted_latest_token_at_promotion: CameraFrameToken
    pretransition_frame_tokens: tuple[CameraFrameToken, ...]
    history_length_before_promotion: int
    history_length_after_promotion: int
    promoted_history_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.from_gate_index, "from_gate_index")
        _nonnegative_int(self.to_gate_index, "to_gate_index")
        if self.to_gate_index != self.from_gate_index + 1:
            raise ValueError("confirmed transition must advance exactly one gate")
        for name in ("retired_track_id", "promoted_track_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty exact string")
        if self.retired_track_id == self.promoted_track_id:
            raise ValueError("retired and promoted track identities must differ")
        if type(self.race_status) is not AuthoritativeRaceStatusRef:
            raise TypeError(
                "race_status must be an exact AuthoritativeRaceStatusRef"
            )
        if (
            self.race_status.race_finished
            or self.race_status.active_gate_index != self.to_gate_index
        ):
            raise ValueError("race status does not prove this gate transition")
        for name in (
            "camera_token_at_credit",
            "promoted_first_token",
            "promoted_latest_token_before_credit",
            "promoted_latest_token_at_promotion",
        ):
            if type(getattr(self, name)) is not CameraFrameToken:
                raise TypeError(f"{name} must be an exact CameraFrameToken")
        if (
            type(self.pretransition_frame_tokens) is not tuple
            or not self.pretransition_frame_tokens
            or any(
                type(token) is not CameraFrameToken
                for token in self.pretransition_frame_tokens
            )
        ):
            raise TypeError(
                "pretransition_frame_tokens must be a non-empty exact tuple "
                "of CameraFrameToken values"
            )
        for name in (
            "promoted_history_length_at_credit",
            "history_length_before_promotion",
            "history_length_after_promotion",
        ):
            _positive_int(getattr(self, name), name)
        if (
            self.history_length_before_promotion
            != self.history_length_after_promotion
        ):
            raise ValueError("promotion must preserve the complete track history")
        if (
            self.promoted_history_length_at_credit
            > self.history_length_before_promotion
        ):
            raise ValueError("credit prefix cannot exceed promotion history")
        if (
            len(self.pretransition_frame_tokens)
            > self.promoted_history_length_at_credit
        ):
            raise ValueError("pretransition tail cannot exceed the credit prefix")
        if (
            self.pretransition_frame_tokens[-1]
            != self.promoted_latest_token_before_credit
        ):
            raise ValueError(
                "pretransition tail must end at the target's credit boundary"
            )
        if not _token_precedes_or_equals(
            self.promoted_first_token,
            self.promoted_latest_token_before_credit,
        ):
            raise ValueError("promoted credit boundary predates its first token")
        if not _token_precedes_or_equals(
            self.promoted_latest_token_before_credit,
            self.camera_token_at_credit,
        ):
            raise ValueError("target credit prefix postdates the camera watermark")
        if not _token_precedes_or_equals(
            self.promoted_latest_token_before_credit,
            self.promoted_latest_token_at_promotion,
        ):
            raise ValueError("promotion token predates the target credit prefix")
        if (
            self.promoted_history_length_at_credit
            == self.history_length_before_promotion
            and self.promoted_latest_token_at_promotion
            != self.promoted_latest_token_before_credit
        ):
            raise ValueError(
                "zero-delay promotion token must equal the credit-prefix token"
            )
        if (
            self.promoted_history_length_at_credit
            < self.history_length_before_promotion
            and not _token_strictly_precedes(
                self.camera_token_at_credit,
                self.promoted_latest_token_at_promotion,
            )
        ):
            raise ValueError(
                "post-credit promotion token must follow the camera watermark"
            )
        for predecessor, successor in zip(
            self.pretransition_frame_tokens,
            self.pretransition_frame_tokens[1:],
        ):
            if not _token_strictly_precedes(predecessor, successor):
                raise ValueError(
                    "pretransition frame tokens must strictly advance"
                )
        if (
            type(self.promoted_history_sha256) is not str
            or len(self.promoted_history_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.promoted_history_sha256
            )
        ):
            raise TypeError(
                "promoted_history_sha256 must be a lowercase SHA-256 hex string"
            )


@dataclass(frozen=True, slots=True)
class CreditedUnboundGateAdvance:
    """Authoritative gate credit recorded without asserting a successor ID.

    The reviewed pre-gap identity is retained as evidence, but the graph is
    deliberately left without a command-authoritative current track.
    """

    from_gate_index: int
    to_gate_index: int
    retired_track_id: str
    reviewed_track_id: str
    race_status: AuthoritativeRaceStatusRef
    camera_token_at_credit: CameraFrameToken
    reviewed_first_token: CameraFrameToken
    reviewed_latest_token_before_credit: CameraFrameToken
    reviewed_history_length_at_credit: int
    reviewed_history_length_at_advance: int
    reviewed_history_sha256: str
    alternative_reacquisition_track_ids_at_credit: tuple[str, ...] = ()

    @property
    def promoted_track_id(self) -> None:
        """No visual identity was promoted at this authoritative boundary."""

        return None

    def __post_init__(self) -> None:
        _nonnegative_int(self.from_gate_index, "from_gate_index")
        _nonnegative_int(self.to_gate_index, "to_gate_index")
        if self.to_gate_index != self.from_gate_index + 1:
            raise ValueError("credited unbound advance must move exactly one gate")
        for name in ("retired_track_id", "reviewed_track_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty exact string")
        if self.retired_track_id == self.reviewed_track_id:
            raise ValueError("reviewed successor must differ from retired current")
        if type(self.race_status) is not AuthoritativeRaceStatusRef:
            raise TypeError(
                "race_status must be an exact AuthoritativeRaceStatusRef"
            )
        if (
            self.race_status.race_finished
            or self.race_status.active_gate_index != self.to_gate_index
        ):
            raise ValueError("race status does not prove this unbound advance")
        for name in (
            "camera_token_at_credit",
            "reviewed_first_token",
            "reviewed_latest_token_before_credit",
        ):
            if type(getattr(self, name)) is not CameraFrameToken:
                raise TypeError(f"{name} must be an exact CameraFrameToken")
        for name in (
            "reviewed_history_length_at_credit",
            "reviewed_history_length_at_advance",
        ):
            _positive_int(getattr(self, name), name)
        if (
            self.reviewed_history_length_at_credit
            > self.reviewed_history_length_at_advance
        ):
            raise ValueError("reviewed credit prefix exceeds retained history")
        if not _token_precedes_or_equals(
            self.reviewed_first_token,
            self.reviewed_latest_token_before_credit,
        ):
            raise ValueError("reviewed credit history is reversed")
        if not _token_precedes_or_equals(
            self.reviewed_latest_token_before_credit,
            self.camera_token_at_credit,
        ):
            raise ValueError("reviewed credit history postdates camera watermark")
        _history_digest(self.reviewed_history_sha256, "reviewed_history_sha256")
        alternatives = self.alternative_reacquisition_track_ids_at_credit
        if (
            type(alternatives) is not tuple
            or any(type(track_id) is not str or not track_id for track_id in alternatives)
            or len(set(alternatives)) != len(alternatives)
            or self.retired_track_id in alternatives
            or self.reviewed_track_id in alternatives
        ):
            raise TypeError(
                "alternative reacquisition identities must be a unique exact "
                "track-id tuple excluding the crossed and reviewed identities"
            )


@dataclass(frozen=True, slots=True)
class ConfirmedGateReacquisition:
    """A locally stable successor bound after an authoritative unbound advance."""

    credited_advance: CreditedUnboundGateAdvance
    gate_index: int
    reacquired_track_id: str
    camera_token_at_binding: CameraFrameToken
    reacquired_first_token: CameraFrameToken
    stable_frame_tokens: tuple[CameraFrameToken, ...]
    history_length_at_binding: int
    history_sha256: str
    cross_gap_identity_claimed: bool = False

    @property
    def from_gate_index(self) -> int:
        return self.credited_advance.from_gate_index

    @property
    def to_gate_index(self) -> int:
        return self.credited_advance.to_gate_index

    @property
    def retired_track_id(self) -> str:
        return self.credited_advance.retired_track_id

    @property
    def promoted_track_id(self) -> str:
        return self.reacquired_track_id

    @property
    def identity_basis(self) -> str:
        if (
            self.reacquired_track_id
            == self.credited_advance.reviewed_track_id
        ):
            return "retained-reviewed-local-track"
        return "fresh-unique-local-track"

    def __post_init__(self) -> None:
        if type(self.credited_advance) is not CreditedUnboundGateAdvance:
            raise TypeError(
                "credited_advance must be an exact CreditedUnboundGateAdvance"
            )
        _nonnegative_int(self.gate_index, "gate_index")
        if self.gate_index != self.credited_advance.to_gate_index:
            raise ValueError("reacquisition gate index differs from credited gate")
        if type(self.reacquired_track_id) is not str or not self.reacquired_track_id:
            raise TypeError("reacquired_track_id must be a non-empty exact string")
        if self.reacquired_track_id == self.credited_advance.retired_track_id:
            raise ValueError("reacquisition cannot restore the crossed gate")
        # A fresh local track does not claim identity through the physical
        # passage gap.  Its authority comes from the credited gate plus the
        # strictly post-credit, stable binding evidence checked below, so
        # pre-credit promotability is evidence rather than a prerequisite.
        for name in ("camera_token_at_binding", "reacquired_first_token"):
            if type(getattr(self, name)) is not CameraFrameToken:
                raise TypeError(f"{name} must be an exact CameraFrameToken")
        if (
            type(self.stable_frame_tokens) is not tuple
            or not self.stable_frame_tokens
            or any(
                type(token) is not CameraFrameToken
                for token in self.stable_frame_tokens
            )
        ):
            raise TypeError(
                "stable_frame_tokens must be a non-empty exact token tuple"
            )
        if self.stable_frame_tokens[-1] != self.camera_token_at_binding:
            raise ValueError("stable reacquisition tail must end at binding token")
        for predecessor, successor in zip(
            self.stable_frame_tokens,
            self.stable_frame_tokens[1:],
        ):
            if not _token_strictly_precedes(predecessor, successor):
                raise ValueError("stable reacquisition tokens must strictly advance")
        if not _token_precedes_or_equals(
            self.reacquired_first_token,
            self.stable_frame_tokens[0],
        ):
            raise ValueError("stable reacquisition tail predates track creation")
        _positive_int(self.history_length_at_binding, "history_length_at_binding")
        if len(self.stable_frame_tokens) > self.history_length_at_binding:
            raise ValueError("stable tail exceeds reacquired local history")
        _history_digest(self.history_sha256, "history_sha256")
        if type(self.cross_gap_identity_claimed) is not bool:
            raise TypeError("cross_gap_identity_claimed must be an exact bool")
        if self.cross_gap_identity_claimed:
            raise ValueError("fresh reacquisition cannot claim cross-gap identity")


@dataclass(frozen=True, slots=True)
class SameGateRebindSearch:
    """Exact start boundary for replacing one lost current visual identity.

    This is not a gate transition and does not claim that a later local track
    is the same physical contour as the lost tracker identity.  It only freezes
    the point after which a wholly new identity may become eligible to carry
    the unchanged authoritative race-gate index.
    """

    gate_index: int
    lost_track_id: str
    race_status_at_start: AuthoritativeRaceStatusRef
    camera_token_at_start: CameraFrameToken
    tracker_frame_sequence_at_start: int
    lost_track_latest_token: CameraFrameToken
    lost_history_length_at_start: int
    lost_history_sha256_at_start: str
    excluded_track_ids_at_start: tuple[str, ...]
    required_stable_frames: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.gate_index, "gate_index")
        if type(self.lost_track_id) is not str or not self.lost_track_id:
            raise TypeError("lost_track_id must be a non-empty exact string")
        if type(self.race_status_at_start) is not AuthoritativeRaceStatusRef:
            raise TypeError(
                "race_status_at_start must be an exact AuthoritativeRaceStatusRef"
            )
        if (
            self.race_status_at_start.race_finished
            or self.race_status_at_start.active_gate_index != self.gate_index
        ):
            raise ValueError("search race status does not prove the unchanged gate")
        for name in ("camera_token_at_start", "lost_track_latest_token"):
            if type(getattr(self, name)) is not CameraFrameToken:
                raise TypeError(f"{name} must be an exact CameraFrameToken")
        _nonnegative_int(
            self.tracker_frame_sequence_at_start,
            "tracker_frame_sequence_at_start",
        )
        _positive_int(
            self.lost_history_length_at_start,
            "lost_history_length_at_start",
        )
        _history_digest(
            self.lost_history_sha256_at_start,
            "lost_history_sha256_at_start",
        )
        if not _token_strictly_precedes(
            self.lost_track_latest_token,
            self.camera_token_at_start,
        ):
            raise ValueError(
                "lost current must be invisible before the rebind search starts"
            )
        excluded = self.excluded_track_ids_at_start
        if (
            type(excluded) is not tuple
            or not excluded
            or any(type(track_id) is not str or not track_id for track_id in excluded)
            or tuple(sorted(excluded)) != excluded
            or len(set(excluded)) != len(excluded)
            or self.lost_track_id not in excluded
        ):
            raise TypeError(
                "excluded_track_ids_at_start must be a sorted unique exact "
                "track-id tuple containing the lost current"
            )
        _positive_int(self.required_stable_frames, "required_stable_frames")
        if self.required_stable_frames < 3:
            raise ValueError("same-gate rebind requires at least three stable frames")


@dataclass(frozen=True, slots=True)
class ConfirmedSameGateRebind:
    """One new local identity bound to an unchanged authoritative race gate."""

    search: SameGateRebindSearch
    race_status_at_binding: AuthoritativeRaceStatusRef
    gate_index: int
    retired_track_id: str
    rebound_track_id: str
    camera_token_at_binding: CameraFrameToken
    rebound_first_token: CameraFrameToken
    stable_frame_tokens: tuple[CameraFrameToken, ...]
    history_length_at_binding: int
    history_sha256: str
    cross_gap_identity_claimed: bool = False

    @property
    def current_track_id(self) -> str:
        return self.rebound_track_id

    @property
    def identity_basis(self) -> str:
        return "fresh-unique-same-gate-local-track"

    def __post_init__(self) -> None:
        if type(self.search) is not SameGateRebindSearch:
            raise TypeError("search must be an exact SameGateRebindSearch")
        if type(self.race_status_at_binding) is not AuthoritativeRaceStatusRef:
            raise TypeError(
                "race_status_at_binding must be an exact AuthoritativeRaceStatusRef"
            )
        _nonnegative_int(self.gate_index, "gate_index")
        if (
            self.gate_index != self.search.gate_index
            or self.race_status_at_binding.race_finished
            or self.race_status_at_binding.active_gate_index != self.gate_index
        ):
            raise ValueError("same-gate rebind changed the authoritative race gate")
        for name in ("retired_track_id", "rebound_track_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty exact string")
        if (
            self.retired_track_id != self.search.lost_track_id
            or self.rebound_track_id == self.retired_track_id
            or self.rebound_track_id in self.search.excluded_track_ids_at_start
        ):
            raise ValueError(
                "same-gate rebind must replace the lost current with a new identity"
            )
        for name in ("camera_token_at_binding", "rebound_first_token"):
            if type(getattr(self, name)) is not CameraFrameToken:
                raise TypeError(f"{name} must be an exact CameraFrameToken")
        if not _token_strictly_precedes(
            self.search.camera_token_at_start,
            self.rebound_first_token,
        ):
            raise ValueError("rebound identity was not first observed after search start")
        if (
            type(self.stable_frame_tokens) is not tuple
            or len(self.stable_frame_tokens) < self.search.required_stable_frames
            or any(
                type(token) is not CameraFrameToken
                for token in self.stable_frame_tokens
            )
        ):
            raise TypeError(
                "stable_frame_tokens must carry the required exact token tail"
            )
        if self.stable_frame_tokens[-1] != self.camera_token_at_binding:
            raise ValueError("stable same-gate tail must end at the binding token")
        if not _token_precedes_or_equals(
            self.rebound_first_token,
            self.stable_frame_tokens[0],
        ):
            raise ValueError("stable same-gate tail predates the rebound identity")
        for predecessor, successor in zip(
            self.stable_frame_tokens,
            self.stable_frame_tokens[1:],
        ):
            if not _token_strictly_precedes(predecessor, successor):
                raise ValueError("stable same-gate tokens must strictly advance")
        _positive_int(self.history_length_at_binding, "history_length_at_binding")
        if len(self.stable_frame_tokens) > self.history_length_at_binding:
            raise ValueError("stable tail exceeds rebound local history")
        _history_digest(self.history_sha256, "history_sha256")
        if type(self.cross_gap_identity_claimed) is not bool:
            raise TypeError("cross_gap_identity_claimed must be an exact bool")
        if self.cross_gap_identity_claimed:
            raise ValueError("same-gate rebind cannot claim cross-gap identity")


@dataclass(frozen=True, slots=True)
class GateGraphSnapshot:
    tracker_frame_sequence: int
    latest_camera_token: CameraFrameToken
    current_track_id: Optional[str]
    current_gate_index: Optional[int]
    current_track: Optional[VisualTrack]
    next_candidates: tuple[NextGateCandidate, ...]
    provisional_track_ids: tuple[str, ...]
    relationships: tuple[ObservedGateRelationship, ...]
    confirmed_transitions: tuple[
        ConfirmedGateTransition
        | ConfirmedGateReacquisition
        | CreditedUnboundGateAdvance,
        ...,
    ]
    next_selection_ambiguous: bool
    authority_usable: bool
    withholding_reason: Optional[str]
    race_finished: bool
    latest_race_status: Optional[AuthoritativeRaceStatusRef]
    phase: GateGraphPhase = GateGraphPhase.INITIAL_UNBOUND
    pending_unbound_advance: Optional[CreditedUnboundGateAdvance] = None


@dataclass(slots=True)
class _RelationshipState:
    basis: GateRelationshipBasis
    current_anchor_token: CameraFrameToken
    next_anchor_token: CameraFrameToken
    anchor_publication_delta: int
    anchor_time_gap_ns: int
    first_token: CameraFrameToken
    latest_token: CameraFrameToken
    observation_count: int
    simultaneous_observation_count: int
    sequential_observation_count: int
    latest_tracker_frame_sequence: int
    current_bearing_norm: float
    current_elevation_norm: float
    next_bearing_norm: float
    next_elevation_norm: float
    current_apparent_scale: float
    next_apparent_scale: float
    current_log_scale_rate_s: float
    next_log_scale_rate_s: float
    observation_confidence: float
    current_center_censored: bool
    next_center_censored: bool
    fresh: bool
    contended: bool


@dataclass(frozen=True, slots=True)
class _PretrackedCandidateEvidence:
    stable_frame_count: int
    missed_camera_publications: int
    confidence: float
    association_confidence: float


@dataclass(frozen=True, slots=True)
class _PromotionCreditBoundary:
    """Exact target-history split at the authoritative race receipt."""

    credit_prefix: tuple[VisualTrackSample, ...]
    pretransition_tail: tuple[VisualTrackSample, ...]
    post_credit_suffix: tuple[VisualTrackSample, ...]


class RollingVisualGateGraph:
    """Maintain current/next visual relationships and proof-bound promotion."""

    def __init__(self, config: Optional[RollingGateGraphConfig] = None) -> None:
        if config is None:
            config = DEFAULT_ROLLING_GATE_GRAPH_CONFIG
        if type(config) is not RollingGateGraphConfig:
            raise TypeError("config must be an exact RollingGateGraphConfig")
        self.config = config
        self._phase = GateGraphPhase.INITIAL_UNBOUND
        self._current_track_id: Optional[str] = None
        self._current_gate_index: Optional[int] = None
        self._pending_unbound_advance: Optional[
            CreditedUnboundGateAdvance
        ] = None
        self._relationships: dict[tuple[str, str], _RelationshipState] = {}
        self._transitions: list[
            ConfirmedGateTransition
            | ConfirmedGateReacquisition
            | CreditedUnboundGateAdvance
        ] = []
        self._last_race_status: Optional[AuthoritativeRaceStatusRef] = None
        self._last_relationship_frame_sequence: Optional[int] = None
        self._race_finished = False
        self._latest_snapshot: Optional[GateGraphSnapshot] = None

    @property
    def latest_snapshot(self) -> Optional[GateGraphSnapshot]:
        return self._latest_snapshot

    def bind_initial_current(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        track_id: str,
        race_status: AuthoritativeRaceStatusRef,
    ) -> GateGraphSnapshot:
        """Bind the proven bootstrap gate to an authoritative race index.

        The caller remains responsible for the bootstrap association (Gate 0
        uses the already-proved approach behavior).  This method requires a
        stable, unambiguous visible track and records the supplied race proof.
        """

        self._tracker(tracker)
        self._race_ref(race_status)
        if (
            self._phase is not GateGraphPhase.INITIAL_UNBOUND
            or self._current_track_id is not None
            or self._current_gate_index is not None
            or self._pending_unbound_advance is not None
        ):
            raise GateGraphError("initial current gate is already bound")
        if race_status.race_finished:
            raise GateGraphError("cannot bind a current gate after race finish")
        track = tracker.track(track_id)
        if (
            not track.visible
            or track.ambiguous
            or track.consecutive_frame_count < self.config.min_current_binding_frames
        ):
            raise GateGraphError("initial current track is not stable and unambiguous")
        tracker.assign_role(track_id, VisualTrackRole.CURRENT)
        tracker.confirm_authoritative_gate(
            track_id,
            gate_index=race_status.active_gate_index,
            race_status_sequence=race_status.race_status_sequence,
            race_status_boot_ms=race_status.race_status_boot_ms,
        )
        self._current_track_id = track_id
        self._current_gate_index = race_status.active_gate_index
        self._last_race_status = race_status
        self._phase = GateGraphPhase.CURRENT_BOUND
        return self.observe(tracker)

    def observe(self, tracker: MultiTargetVisualTracker) -> GateGraphSnapshot:
        """Update roles and rolling image relationships from the latest frame."""

        self._tracker(tracker)
        update = tracker.latest_update
        if update is None:
            raise GateGraphError("tracker has no camera update")
        tracks = {track.track_id: track for track in update.tracks}
        if self._current_track_id is not None:
            if self._current_track_id not in tracks:
                raise GateGraphError("current visual identity disappeared from tracker")
            current = tracks[self._current_track_id]
            if current.role is not VisualTrackRole.RETIRED:
                tracker.assign_role(self._current_track_id, VisualTrackRole.CURRENT)
            for track in tracks.values():
                if (
                    track.track_id == self._current_track_id
                    or track.role is VisualTrackRole.RETIRED
                ):
                    continue
                stable = (
                    track.visible
                    and not track.ambiguous
                    and track.consecutive_frame_count
                    >= self.config.min_next_candidate_frames
                    and track.confidence >= self.config.min_track_confidence
                    and track.association_confidence
                    >= self.config.min_association_confidence
                )
                tracker.assign_role(
                    track.track_id,
                    VisualTrackRole.NEXT if stable else VisualTrackRole.UNKNOWN,
                )
            update = tracker.latest_update
            assert update is not None
            tracks = {track.track_id: track for track in update.tracks}

        if (
            self._current_track_id is not None
            and update.tracker_frame_sequence != self._last_relationship_frame_sequence
        ):
            current = tracks[self._current_track_id]
            visible_noncurrent = tuple(
                track
                for track in tracks.values()
                if (
                    track.track_id != current.track_id
                    and track.visible
                    and track.role is not VisualTrackRole.RETIRED
                )
            )
            handoff_contenders = tuple(
                track
                for track in tracks.values()
                if (
                    track.track_id != current.track_id
                    and track.role is not VisualTrackRole.RETIRED
                    and (
                        track.visible
                        or _is_recent_handoff_contender(
                            track,
                            current_tracker_frame_sequence=(
                                update.tracker_frame_sequence
                            ),
                        )
                    )
                )
            )
            sequential_seed_track_id: Optional[str] = None
            if (
                len(visible_noncurrent) == 1
                and len(handoff_contenders) == 1
                and handoff_contenders[0].track_id
                == visible_noncurrent[0].track_id
                and self._eligible_adjacent_handoff_seed(
                    current,
                    visible_noncurrent[0],
                    update.tracker_frame_sequence,
                    update.token,
                )
            ):
                sequential_seed_track_id = visible_noncurrent[0].track_id
            for track in visible_noncurrent:
                self._update_relationship(
                    current,
                    track,
                    update.tracker_frame_sequence,
                    update.token,
                    allow_adjacent_handoff_seed=(
                        track.track_id == sequential_seed_track_id
                    ),
                )
            self._mark_adjacent_handoff_contention(
                current,
                visible_noncurrent,
            )
            self._last_relationship_frame_sequence = update.tracker_frame_sequence

        snapshot = self._snapshot(update.tracker_frame_sequence, update.token, tracks)
        self._latest_snapshot = snapshot
        return snapshot

    def confirm_transition(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        race_status: AuthoritativeRaceStatusRef,
        camera_token_at_credit: CameraFrameToken,
        promoted_track_id: Optional[str] = None,
    ) -> ConfirmedGateTransition:
        """Promote a stable pre-credit next track without resetting its history."""

        self._tracker(tracker)
        self._race_ref(race_status)
        if type(camera_token_at_credit) is not CameraFrameToken:
            raise TypeError("camera_token_at_credit must be an exact CameraFrameToken")
        if self._race_finished:
            raise GateGraphError("cannot promote a gate after race finish")
        if (
            self._phase is not GateGraphPhase.CURRENT_BOUND
            or self._pending_unbound_advance is not None
            or self._current_track_id is None
            or self._current_gate_index is None
        ):
            raise GateGraphError("current gate is not bound")
        if race_status.race_finished:
            raise GateGraphError(
                "race_finished is terminal; use confirm_race_finished"
            )
        self._validate_race_advance(race_status)
        if race_status.active_gate_index != self._current_gate_index + 1:
            raise GateGraphError("authoritative gate index did not advance by one")
        if not tracker.has_processed_token(camera_token_at_credit):
            raise GateGraphError("credit references an unprocessed camera token")
        self._validate_camera_precedes_race(
            tracker,
            camera_token_at_credit,
            race_status,
        )
        snapshot = self.observe(tracker)
        promotable = tuple(
            candidate for candidate in snapshot.next_candidates if candidate.promotable
        )
        if snapshot.next_selection_ambiguous:
            raise AmbiguousGatePromotionError(
                "multiple next-gate tracks have indistinguishable authority"
            )
        if not promotable:
            if promoted_track_id is not None:
                raise RequestedGatePromotionUnavailableError(
                    "no stable pretracked next gate is promotable; "
                    "requested promotion track is unavailable"
                )
            raise GateGraphError("no stable pretracked next gate is promotable")
        selected: Optional[NextGateCandidate] = None
        if promoted_track_id is not None:
            selected = next(
                (
                    candidate
                    for candidate in promotable
                    if candidate.track_id == promoted_track_id
                ),
                None,
            )
            if selected is None:
                raise RequestedGatePromotionUnavailableError(
                    "requested promotion track is not promotable"
                )
        else:
            selected = promotable[0]
        competing = tuple(
            candidate
            for candidate in promotable
            if candidate.track_id != selected.track_id
        )
        if competing and (
            selected.score - competing[0].score
            < self.config.candidate_selection_margin
        ):
            raise AmbiguousGatePromotionError(
                "multiple next-gate tracks have indistinguishable authority"
            )

        promoted_before = tracker.track(selected.track_id)
        credit_boundary = _promotion_credit_boundary(
            promoted_before,
            camera_token_at_credit,
            race_status,
        )
        pretransition_samples = credit_boundary.pretransition_tail
        if len(pretransition_samples) < self.config.min_next_candidate_frames:
            raise GateGraphError(
                "promotion lacks three consecutive fresh pre-transition frames"
            )
        _validate_adjacent_handoff_credit_freshness(
            selected,
            pretransition_samples,
            race_status,
        )
        if any(sample.association_confidence < self.config.min_association_confidence
               for sample in pretransition_samples[-self.config.min_next_candidate_frames :]):
            raise GateGraphError("pre-transition association confidence is insufficient")

        retired_track_id = self._current_track_id
        tracker.retire_track(retired_track_id)
        tracker.assign_role(selected.track_id, VisualTrackRole.CURRENT)
        tracker.confirm_authoritative_gate(
            selected.track_id,
            gate_index=race_status.active_gate_index,
            race_status_sequence=race_status.race_status_sequence,
            race_status_boot_ms=race_status.race_status_boot_ms,
        )
        promoted_after = tracker.track(selected.track_id)
        if promoted_after.first_token != promoted_before.first_token:
            raise RuntimeError("promotion changed the pretracked identity")
        if promoted_after.history != promoted_before.history:
            raise RuntimeError("promotion reset or rewrote visual history")

        transition = ConfirmedGateTransition(
            from_gate_index=self._current_gate_index,
            to_gate_index=race_status.active_gate_index,
            retired_track_id=retired_track_id,
            promoted_track_id=selected.track_id,
            race_status=race_status,
            camera_token_at_credit=camera_token_at_credit,
            promoted_first_token=promoted_before.first_token,
            promoted_latest_token_before_credit=(
                credit_boundary.credit_prefix[-1].token
            ),
            promoted_history_length_at_credit=len(
                credit_boundary.credit_prefix
            ),
            promoted_latest_token_at_promotion=promoted_before.latest_token,
            pretransition_frame_tokens=tuple(
                sample.token for sample in pretransition_samples
            ),
            history_length_before_promotion=len(promoted_before.history),
            history_length_after_promotion=len(promoted_after.history),
            promoted_history_sha256=visual_track_history_sha256(
                promoted_before.history
            ),
        )
        _validate_transition_history(
            transition,
            promoted_before.history,
            credit_boundary,
        )
        self._transitions.append(transition)
        if len(self._transitions) > self.config.relationship_history_limit:
            del self._transitions[
                : len(self._transitions) - self.config.relationship_history_limit
            ]
        self._current_track_id = selected.track_id
        self._current_gate_index = race_status.active_gate_index
        self._last_race_status = race_status
        self._phase = GateGraphPhase.CURRENT_BOUND
        self.observe(tracker)
        return transition

    def confirm_unbound_advance(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        race_status: AuthoritativeRaceStatusRef,
        camera_token_at_credit: CameraFrameToken,
        reviewed_track_id: str,
    ) -> CreditedUnboundGateAdvance:
        """Consume exact race credit while withholding successor authority.

        This path is for a reviewed pre-gap next identity that cannot be
        safely promoted at credit.  It retires the crossed current gate but
        never relabels another track or claims identity across the visual gap.
        """

        self._tracker(tracker)
        self._race_ref(race_status)
        if type(camera_token_at_credit) is not CameraFrameToken:
            raise TypeError(
                "camera_token_at_credit must be an exact CameraFrameToken"
            )
        if type(reviewed_track_id) is not str or not reviewed_track_id:
            raise TypeError("reviewed_track_id must be a non-empty exact string")
        if self._race_finished:
            raise GateGraphError("cannot advance an unbound gate after race finish")
        if (
            self._phase is not GateGraphPhase.CURRENT_BOUND
            or self._pending_unbound_advance is not None
            or self._current_track_id is None
            or self._current_gate_index is None
        ):
            raise GateGraphError("current gate is not bound")
        if reviewed_track_id == self._current_track_id:
            raise GateGraphError("reviewed successor equals the current gate")
        if race_status.race_finished:
            raise GateGraphError(
                "race_finished is terminal; use confirm_race_finished"
            )
        self._validate_race_advance(race_status)
        if race_status.active_gate_index != self._current_gate_index + 1:
            raise GateGraphError("authoritative gate index did not advance by one")
        if not tracker.has_processed_token(camera_token_at_credit):
            raise GateGraphError("credit references an unprocessed camera token")
        self._validate_camera_precedes_race(
            tracker,
            camera_token_at_credit,
            race_status,
        )
        try:
            reviewed_track = tracker.track(reviewed_track_id)
        except KeyError as exc:
            raise GateGraphError(
                "reviewed successor is absent from tracker history"
            ) from exc
        snapshot = self.observe(tracker)
        reviewed_candidate = next(
            (
                candidate
                for candidate in snapshot.next_candidates
                if candidate.track_id == reviewed_track_id
            ),
            None,
        )
        if (
            reviewed_candidate is not None
            and reviewed_candidate.promotable
            and not snapshot.next_selection_ambiguous
        ):
            raise GateGraphError(
                "reviewed successor remains directly promotable"
            )
        reviewed_boundary = _promotion_credit_boundary(
            reviewed_track,
            camera_token_at_credit,
            race_status,
        )
        retired_track_id = self._current_track_id
        advance = CreditedUnboundGateAdvance(
            from_gate_index=self._current_gate_index,
            to_gate_index=race_status.active_gate_index,
            retired_track_id=retired_track_id,
            reviewed_track_id=reviewed_track_id,
            race_status=race_status,
            camera_token_at_credit=camera_token_at_credit,
            reviewed_first_token=reviewed_track.first_token,
            reviewed_latest_token_before_credit=(
                reviewed_boundary.credit_prefix[-1].token
            ),
            reviewed_history_length_at_credit=len(
                reviewed_boundary.credit_prefix
            ),
            reviewed_history_length_at_advance=len(reviewed_track.history),
            reviewed_history_sha256=visual_track_history_sha256(
                reviewed_track.history
            ),
            alternative_reacquisition_track_ids_at_credit=tuple(
                sorted(
                    candidate.track_id
                    for candidate in snapshot.next_candidates
                    if (
                        candidate.promotable
                        and candidate.track_id != reviewed_track_id
                    )
                )
            ),
        )

        # Every operation that can reject caller evidence has completed.
        # Retiring one known track is the only tracker mutation in this commit.
        tracker.retire_track(retired_track_id)
        self._current_track_id = None
        self._current_gate_index = race_status.active_gate_index
        self._last_race_status = race_status
        self._pending_unbound_advance = advance
        self._phase = GateGraphPhase.CREDITED_UNBOUND
        self.observe(tracker)
        return advance

    def confirm_reviewed_advance(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        race_status: AuthoritativeRaceStatusRef,
        camera_token_at_credit: CameraFrameToken,
        reviewed_track_id: str,
    ) -> ConfirmedGateTransition | CreditedUnboundGateAdvance:
        """Atomically consume credit as retained promotion or explicit unbound."""

        try:
            return self.confirm_transition(
                tracker,
                race_status=race_status,
                camera_token_at_credit=camera_token_at_credit,
                promoted_track_id=reviewed_track_id,
            )
        except (
            RequestedGatePromotionUnavailableError,
            AmbiguousGatePromotionError,
        ):
            return self.confirm_unbound_advance(
                tracker,
                race_status=race_status,
                camera_token_at_credit=camera_token_at_credit,
                reviewed_track_id=reviewed_track_id,
            )

    def try_confirm_reacquired_current(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        credited_advance: CreditedUnboundGateAdvance,
        camera_token_at_binding: CameraFrameToken,
    ) -> ConfirmedGateReacquisition | GateReacquisitionPending:
        """Return soft readiness explicitly; reserve exceptions for hard faults."""

        try:
            return self.confirm_reacquired_current(
                tracker,
                credited_advance=credited_advance,
                camera_token_at_binding=camera_token_at_binding,
            )
        except AmbiguousGateReacquisitionError as exc:
            return GateReacquisitionPending(
                reason=str(exc),
                ambiguous=True,
            )
        except GateReacquisitionNotReadyError as exc:
            return GateReacquisitionPending(
                reason=str(exc),
                ambiguous=False,
            )

    def confirm_reacquired_current(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        credited_advance: CreditedUnboundGateAdvance,
        camera_token_at_binding: CameraFrameToken,
    ) -> ConfirmedGateReacquisition:
        """Bind one unique clean successor on a strictly post-credit frame."""

        self._tracker(tracker)
        if type(credited_advance) is not CreditedUnboundGateAdvance:
            raise TypeError(
                "credited_advance must be an exact CreditedUnboundGateAdvance"
            )
        if type(camera_token_at_binding) is not CameraFrameToken:
            raise TypeError(
                "camera_token_at_binding must be an exact CameraFrameToken"
            )
        if (
            self._phase is not GateGraphPhase.CREDITED_UNBOUND
            or self._pending_unbound_advance is None
            or credited_advance != self._pending_unbound_advance
            or self._current_track_id is not None
            or self._current_gate_index != credited_advance.to_gate_index
            or self._last_race_status != credited_advance.race_status
        ):
            raise GateGraphError("credited-unbound advance is not pending")
        update = tracker.latest_update
        if (
            update is None
            or update.token != camera_token_at_binding
            or not tracker.has_processed_token(camera_token_at_binding)
        ):
            raise GateGraphError(
                "reacquisition binding token is not the latest processed frame"
            )
        if not _token_strictly_precedes(
            credited_advance.camera_token_at_credit,
            camera_token_at_binding,
        ):
            raise GateReacquisitionNotReadyError(
                "reacquisition frame does not strictly follow race credit"
            )
        race_received_ns = credited_advance.race_status.received_monotonic_ns
        if (
            credited_advance.race_status.provenance_basis
            is not RaceStatusProvenanceBasis.LIVE_INGRESS
            or race_received_ns is None
            or update.provenance_basis
            is not FrameProvenanceBasis.RECEIVER_TIMING_V1
            or tracker.time_basis_id
            != credited_advance.race_status.host_clock_id
            or update.observation_monotonic_ns <= race_received_ns
            or update.publish_monotonic_ns is None
            or update.publish_monotonic_ns <= race_received_ns
            or update.publish_monotonic_ns < update.observation_monotonic_ns
        ):
            raise GateReacquisitionNotReadyError(
                "reacquisition frame is not exact fresh post-credit evidence"
            )

        snapshot = self.observe(tracker)
        if (
            snapshot.latest_camera_token != camera_token_at_binding
            or snapshot.tracker_frame_sequence
            != update.tracker_frame_sequence
        ):
            raise GateGraphError(
                "reacquisition graph does not match the binding frame"
            )
        local_candidates: list[
            tuple[VisualTrack, tuple[VisualTrackSample, ...]]
        ] = []
        for track in update.tracks:
            stable_tail = _reacquisition_observable_tail(
                track,
                camera_token_at_binding=camera_token_at_binding,
                required_frames=self.config.min_current_binding_frames,
                min_track_confidence=self.config.min_track_confidence,
                min_association_confidence=(
                    self.config.min_association_confidence
                ),
            )
            if (
                stable_tail is not None
                and track.track_id != credited_advance.retired_track_id
            ):
                local_candidates.append((track, stable_tail))
        if not local_candidates:
            raise GateReacquisitionNotReadyError(
                "no unique observable local successor is ready for reacquisition"
            )
        if len(local_candidates) > 1:
            raise AmbiguousGateReacquisitionError(
                "post-credit local successor selection is ambiguous"
            )
        selected, stable_tail = local_candidates[0]
        if (
            not selected.visible
            or selected.role is VisualTrackRole.RETIRED
            or selected.authoritative_gate_index is not None
            or selected.track_id == credited_advance.retired_track_id
            or selected.ambiguous
        ):
            raise GateReacquisitionNotReadyError(
                "rolling-graph successor is not a visible unbound local target"
            )
        latest_sample = stable_tail[-1]
        if (
            latest_sample.observation_monotonic_ns <= race_received_ns
            or latest_sample.publication_monotonic_ns is None
            or latest_sample.publication_monotonic_ns <= race_received_ns
            or latest_sample.publication_monotonic_ns
            < latest_sample.observation_monotonic_ns
        ):
            raise GateReacquisitionNotReadyError(
                "compatible successor latest sample is not strictly post-credit"
            )
        history_before = selected.history
        reacquisition = ConfirmedGateReacquisition(
            credited_advance=credited_advance,
            gate_index=credited_advance.to_gate_index,
            reacquired_track_id=selected.track_id,
            camera_token_at_binding=camera_token_at_binding,
            reacquired_first_token=selected.first_token,
            stable_frame_tokens=tuple(
                sample.token for sample in stable_tail
            ),
            history_length_at_binding=len(history_before),
            history_sha256=visual_track_history_sha256(history_before),
            cross_gap_identity_claimed=False,
        )

        # Selection, freshness, and proof construction are complete before
        # the bounded role/authority commit.
        tracker.assign_role(selected.track_id, VisualTrackRole.CURRENT)
        tracker.confirm_authoritative_gate(
            selected.track_id,
            gate_index=credited_advance.to_gate_index,
            race_status_sequence=(
                credited_advance.race_status.race_status_sequence
            ),
            race_status_boot_ms=(
                credited_advance.race_status.race_status_boot_ms
            ),
        )
        bound = tracker.track(selected.track_id)
        if bound.history != history_before:
            raise RuntimeError("reacquisition reset or rewrote local history")
        self._transitions.append(reacquisition)
        if len(self._transitions) > self.config.relationship_history_limit:
            del self._transitions[
                : len(self._transitions) - self.config.relationship_history_limit
            ]
        self._current_track_id = selected.track_id
        self._current_gate_index = credited_advance.to_gate_index
        self._pending_unbound_advance = None
        self._phase = GateGraphPhase.CURRENT_BOUND
        self.observe(tracker)
        return reacquisition

    def begin_same_gate_rebind_search(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        race_status: AuthoritativeRaceStatusRef,
        camera_token_at_start: CameraFrameToken,
    ) -> SameGateRebindSearch:
        """Freeze a lost-current boundary without changing visual authority."""

        self._tracker(tracker)
        self._race_ref(race_status)
        if type(camera_token_at_start) is not CameraFrameToken:
            raise TypeError(
                "camera_token_at_start must be an exact CameraFrameToken"
            )
        if (
            self._phase is not GateGraphPhase.CURRENT_BOUND
            or self._pending_unbound_advance is not None
            or self._current_track_id is None
            or self._current_gate_index is None
            or self._race_finished
        ):
            raise GateGraphError(
                "same-gate rebind search requires one bound current gate"
            )
        update = tracker.latest_update
        if (
            update is None
            or update.token != camera_token_at_start
            or not tracker.has_processed_token(camera_token_at_start)
        ):
            raise GateGraphError(
                "same-gate search token is not the latest processed frame"
            )
        snapshot = self.observe(tracker)
        current = snapshot.current_track
        if (
            snapshot.current_track_id != self._current_track_id
            or snapshot.current_gate_index != self._current_gate_index
            or current is None
            or current.role is not VisualTrackRole.CURRENT
            or current.authoritative_gate_index != self._current_gate_index
        ):
            raise GateGraphError(
                "same-gate rebind search lacks the bound current identity"
            )
        if current.visible:
            raise GateGraphError(
                "same-gate rebind search requires an invisible current track"
            )
        if current.ambiguous:
            raise GateGraphError(
                "same-gate rebind search cannot start from ambiguous current state"
            )
        self._validate_unchanged_race_status(
            race_status,
            expected_gate_index=self._current_gate_index,
        )
        self._last_race_status = race_status
        self.observe(tracker)
        return SameGateRebindSearch(
            gate_index=self._current_gate_index,
            lost_track_id=self._current_track_id,
            race_status_at_start=race_status,
            camera_token_at_start=camera_token_at_start,
            tracker_frame_sequence_at_start=update.tracker_frame_sequence,
            lost_track_latest_token=current.latest_token,
            lost_history_length_at_start=len(current.history),
            lost_history_sha256_at_start=visual_track_history_sha256(
                current.history
            ),
            excluded_track_ids_at_start=tuple(
                sorted(track.track_id for track in update.tracks)
            ),
            required_stable_frames=max(
                3,
                self.config.min_current_binding_frames,
            ),
        )

    def try_confirm_same_gate_rebind(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        search: SameGateRebindSearch,
        race_status: AuthoritativeRaceStatusRef,
        camera_token_at_binding: CameraFrameToken,
    ) -> ConfirmedSameGateRebind | GateReacquisitionPending:
        """Return soft uniqueness/readiness outcomes for active search."""

        try:
            return self.confirm_same_gate_rebind(
                tracker,
                search=search,
                race_status=race_status,
                camera_token_at_binding=camera_token_at_binding,
            )
        except AmbiguousGateReacquisitionError as exc:
            return GateReacquisitionPending(
                reason=str(exc),
                ambiguous=True,
            )
        except GateReacquisitionNotReadyError as exc:
            return GateReacquisitionPending(
                reason=str(exc),
                ambiguous=False,
            )

    def confirm_same_gate_rebind(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        search: SameGateRebindSearch,
        race_status: AuthoritativeRaceStatusRef,
        camera_token_at_binding: CameraFrameToken,
    ) -> ConfirmedSameGateRebind:
        """Replace one lost CURRENT with one unique post-search local track."""

        self._tracker(tracker)
        if type(search) is not SameGateRebindSearch:
            raise TypeError("search must be an exact SameGateRebindSearch")
        self._race_ref(race_status)
        if type(camera_token_at_binding) is not CameraFrameToken:
            raise TypeError(
                "camera_token_at_binding must be an exact CameraFrameToken"
            )
        if (
            self._phase is not GateGraphPhase.CURRENT_BOUND
            or self._pending_unbound_advance is not None
            or self._current_track_id != search.lost_track_id
            or self._current_gate_index != search.gate_index
            or self._last_race_status != search.race_status_at_start
            or self._race_finished
        ):
            raise GateGraphError(
                "same-gate rebind search no longer matches graph authority"
            )
        update = tracker.latest_update
        if (
            update is None
            or update.token != camera_token_at_binding
            or not tracker.has_processed_token(camera_token_at_binding)
        ):
            raise GateGraphError(
                "same-gate binding token is not the latest processed frame"
            )
        if not _token_strictly_precedes(
            search.camera_token_at_start,
            camera_token_at_binding,
        ):
            raise GateReacquisitionNotReadyError(
                "same-gate binding frame does not follow search start"
            )
        self._validate_unchanged_race_status(
            race_status,
            expected_gate_index=search.gate_index,
        )

        snapshot = self.observe(tracker)
        lost = snapshot.current_track
        if (
            snapshot.latest_camera_token != camera_token_at_binding
            or snapshot.tracker_frame_sequence != update.tracker_frame_sequence
        ):
            raise GateGraphError(
                "same-gate graph does not match the binding frame"
            )
        if (
            lost is None
            or lost.track_id != search.lost_track_id
            or lost.role is not VisualTrackRole.CURRENT
            or lost.authoritative_gate_index != search.gate_index
        ):
            raise GateGraphError(
                "same-gate rebind lost its original current authority"
            )
        if lost.visible:
            raise GateReacquisitionNotReadyError(
                "original current became visible; same-gate rebind is unnecessary"
            )
        if (
            lost.latest_token != search.lost_track_latest_token
            or len(lost.history) != search.lost_history_length_at_start
            or visual_track_history_sha256(lost.history)
            != search.lost_history_sha256_at_start
        ):
            raise GateReacquisitionNotReadyError(
                "original current changed after same-gate search start"
            )

        local_candidates: list[
            tuple[VisualTrack, tuple[VisualTrackSample, ...]]
        ] = []
        departure_side_rejected = False
        for track in update.tracks:
            stable_tail = _same_gate_rebind_observable_tail(
                track,
                search=search,
                camera_token_at_binding=camera_token_at_binding,
                min_track_confidence=self.config.min_track_confidence,
                min_association_confidence=(
                    self.config.min_association_confidence
                ),
            )
            if stable_tail is not None:
                if not _same_gate_rebind_matches_departure_half_plane(
                    lost,
                    stable_tail,
                ):
                    departure_side_rejected = True
                    continue
                local_candidates.append((track, stable_tail))
        if not local_candidates:
            raise GateReacquisitionNotReadyError(
                (
                    "fresh post-search same-gate candidate contradicts "
                    "lost-current departure side"
                    if departure_side_rejected
                    else (
                        "no unique fresh post-search same-gate candidate "
                        "is ready"
                    )
                )
            )
        if len(local_candidates) > 1:
            raise AmbiguousGateReacquisitionError(
                "post-search same-gate candidate selection is ambiguous"
            )
        selected, stable_tail = local_candidates[0]
        history_before = selected.history
        rebind = ConfirmedSameGateRebind(
            search=search,
            race_status_at_binding=race_status,
            gate_index=search.gate_index,
            retired_track_id=search.lost_track_id,
            rebound_track_id=selected.track_id,
            camera_token_at_binding=camera_token_at_binding,
            rebound_first_token=selected.first_token,
            stable_frame_tokens=tuple(
                sample.token for sample in stable_tail
            ),
            history_length_at_binding=len(history_before),
            history_sha256=visual_track_history_sha256(history_before),
            cross_gap_identity_claimed=False,
        )

        # All readiness, uniqueness, race-gate, and immutable-history checks
        # complete before this bounded role/authority replacement.
        tracker.retire_track(search.lost_track_id)
        tracker.assign_role(selected.track_id, VisualTrackRole.CURRENT)
        tracker.confirm_authoritative_gate(
            selected.track_id,
            gate_index=search.gate_index,
            race_status_sequence=race_status.race_status_sequence,
            race_status_boot_ms=race_status.race_status_boot_ms,
        )
        bound = tracker.track(selected.track_id)
        if bound.history != history_before:
            raise RuntimeError("same-gate rebind reset or rewrote local history")
        self._current_track_id = selected.track_id
        self._current_gate_index = search.gate_index
        self._last_race_status = race_status
        self._phase = GateGraphPhase.CURRENT_BOUND
        self.observe(tracker)
        return rebind

    def confirm_race_finished(
        self,
        tracker: MultiTargetVisualTracker,
        *,
        race_status: AuthoritativeRaceStatusRef,
        camera_token_at_finish: CameraFrameToken,
    ) -> GateGraphSnapshot:
        """Latch terminal completion from authoritative race status only."""

        self._tracker(tracker)
        self._race_ref(race_status)
        if not race_status.race_finished:
            raise GateGraphError("race status does not assert race_finished")
        if self._race_finished:
            raise GateGraphError("race_finished was already confirmed")
        if self._phase is GateGraphPhase.INITIAL_UNBOUND:
            raise GateGraphError("race finish lacks an authoritative gate baseline")
        self._validate_race_advance(race_status, allow_same_gate=True)
        if not tracker.has_processed_token(camera_token_at_finish):
            raise GateGraphError("finish references an unprocessed camera token")
        self._validate_camera_precedes_race(
            tracker,
            camera_token_at_finish,
            race_status,
        )
        if self._phase is GateGraphPhase.CREDITED_UNBOUND:
            assert self._pending_unbound_advance is not None
            self._transitions.append(self._pending_unbound_advance)
            if len(self._transitions) > self.config.relationship_history_limit:
                del self._transitions[
                    : len(self._transitions)
                    - self.config.relationship_history_limit
                ]
            self._pending_unbound_advance = None
        self._race_finished = True
        self._last_race_status = race_status
        self._phase = GateGraphPhase.RACE_FINISHED
        return self.observe(tracker)

    def _update_relationship(
        self,
        current: VisualTrack,
        candidate: VisualTrack,
        frame_sequence: int,
        token: CameraFrameToken,
        *,
        allow_adjacent_handoff_seed: bool,
    ) -> None:
        key = (current.track_id, candidate.track_id)
        previous = self._relationships.get(key)
        adjacent_anchors = None
        if (
            previous is not None
            and previous.basis
            is GateRelationshipBasis.ADJACENT_PUBLICATION_HANDOFF
            and not current.visible
            and not _adjacent_handoff_step_is_contiguous(
                previous,
                candidate,
                frame_sequence=frame_sequence,
                token=token,
            )
        ):
            previous.contended = True
            return
        if not current.visible and previous is None:
            if not allow_adjacent_handoff_seed:
                # No joint image observation or tightly bounded sequential
                # crossing evidence exists from which to claim a relationship.
                return
            adjacent_anchors = _adjacent_handoff_anchors(
                current,
                candidate,
                frame_sequence=frame_sequence,
                token=token,
                config=self.config,
            )
            if adjacent_anchors is None:
                return
        count = 1 if previous is None else previous.observation_count + 1
        if current.visible:
            simultaneous_count = (
                1
                if previous is None
                else previous.simultaneous_observation_count + 1
            )
            confidence = math.sqrt(
                max(0.0, current.confidence) * max(0.0, candidate.confidence)
            ) * min(
                current.association_confidence,
                candidate.association_confidence,
            )
            if current.center_censored or candidate.center_censored:
                confidence *= 0.75
            confidence *= min(
                1.0,
                simultaneous_count / self.config.min_next_candidate_frames,
            )
            current_bearing = current.bearing_norm
            current_elevation = current.elevation_norm
            current_scale = current.apparent_scale
            current_scale_rate = current.log_scale_rate_s
            current_censored = current.center_censored
            basis = (
                GateRelationshipBasis.SIMULTANEOUS_IMAGE
                if previous is None
                else previous.basis
            )
            current_anchor_token = (
                token if previous is None else previous.current_anchor_token
            )
            next_anchor_token = (
                token if previous is None else previous.next_anchor_token
            )
            anchor_publication_delta = (
                0 if previous is None else previous.anchor_publication_delta
            )
            anchor_time_gap_ns = (
                0 if previous is None else previous.anchor_time_gap_ns
            )
            sequential_count = (
                0 if previous is None else previous.sequential_observation_count
            )
            contended = (
                False
                if previous is None
                else (
                    previous.contended
                    or previous.basis
                    is GateRelationshipBasis.ADJACENT_PUBLICATION_HANDOFF
                )
            )
        elif previous is None:
            assert adjacent_anchors is not None
            current_sample, candidate_sample, anchor_time_gap_ns = adjacent_anchors
            simultaneous_count = 0
            sequential_count = 1
            confidence = _adjacent_handoff_confidence(
                current_sample,
                candidate_sample,
            )
            current_bearing = current_sample.bearing_norm
            current_elevation = current_sample.elevation_norm
            current_scale = current_sample.apparent_scale
            current_scale_rate = current.log_scale_rate_s
            current_censored = current_sample.center_censored
            basis = GateRelationshipBasis.ADJACENT_PUBLICATION_HANDOFF
            current_anchor_token = current_sample.token
            next_anchor_token = candidate_sample.token
            anchor_publication_delta = 1
            contended = False
        else:
            # The next track may keep accumulating exact pre-credit history
            # after the crossed/current contour disappears.  Retain the last
            # jointly observed relationship confidence and mark it non-fresh;
            # never synthesize new current geometry from the stale support.
            simultaneous_count = previous.simultaneous_observation_count
            sequential_count = (
                previous.sequential_observation_count
                + (
                    1
                    if previous.basis
                    is GateRelationshipBasis.ADJACENT_PUBLICATION_HANDOFF
                    else 0
                )
            )
            confidence = previous.observation_confidence
            current_bearing = previous.current_bearing_norm
            current_elevation = previous.current_elevation_norm
            current_scale = previous.current_apparent_scale
            current_scale_rate = previous.current_log_scale_rate_s
            current_censored = previous.current_center_censored
            basis = previous.basis
            current_anchor_token = previous.current_anchor_token
            next_anchor_token = previous.next_anchor_token
            anchor_publication_delta = previous.anchor_publication_delta
            anchor_time_gap_ns = previous.anchor_time_gap_ns
            contended = previous.contended or candidate.ambiguous
        state = _RelationshipState(
            basis=basis,
            current_anchor_token=current_anchor_token,
            next_anchor_token=next_anchor_token,
            anchor_publication_delta=anchor_publication_delta,
            anchor_time_gap_ns=anchor_time_gap_ns,
            first_token=token if previous is None else previous.first_token,
            latest_token=token,
            observation_count=count,
            simultaneous_observation_count=simultaneous_count,
            sequential_observation_count=sequential_count,
            latest_tracker_frame_sequence=frame_sequence,
            current_bearing_norm=current_bearing,
            current_elevation_norm=current_elevation,
            next_bearing_norm=candidate.bearing_norm,
            next_elevation_norm=candidate.elevation_norm,
            current_apparent_scale=current_scale,
            next_apparent_scale=candidate.apparent_scale,
            current_log_scale_rate_s=current_scale_rate,
            next_log_scale_rate_s=candidate.log_scale_rate_s,
            observation_confidence=confidence,
            current_center_censored=current_censored,
            next_center_censored=candidate.center_censored,
            fresh=current.visible and candidate.visible,
            contended=contended,
        )
        self._relationships[key] = state
        if len(self._relationships) > self.config.relationship_history_limit:
            oldest_key = min(
                self._relationships,
                key=lambda item: (
                    self._relationships[item].latest_tracker_frame_sequence,
                    item,
                ),
            )
            del self._relationships[oldest_key]

    def _eligible_adjacent_handoff_seed(
        self,
        current: VisualTrack,
        candidate: VisualTrack,
        frame_sequence: int,
        token: CameraFrameToken,
    ) -> bool:
        return _adjacent_handoff_anchors(
            current,
            candidate,
            frame_sequence=frame_sequence,
            token=token,
            config=self.config,
        ) is not None

    def _mark_adjacent_handoff_contention(
        self,
        current: VisualTrack,
        visible_noncurrent: tuple[VisualTrack, ...],
    ) -> None:
        visible_ids = {track.track_id for track in visible_noncurrent}
        for key, state in self._relationships.items():
            if (
                key[0] != current.track_id
                or state.basis
                is not GateRelationshipBasis.ADJACENT_PUBLICATION_HANDOFF
            ):
                continue
            if (
                current.visible
                or any(track_id != key[1] for track_id in visible_ids)
                or any(
                    track.track_id == key[1] and track.ambiguous
                    for track in visible_noncurrent
                )
            ):
                state.contended = True

    def _snapshot(
        self,
        frame_sequence: int,
        token: CameraFrameToken,
        tracks: dict[str, VisualTrack],
    ) -> GateGraphSnapshot:
        current = (
            None
            if self._current_track_id is None
            else tracks.get(self._current_track_id)
        )
        relationships = (
            ()
            if self._pending_unbound_advance is not None
            else tuple(
                self._relationship(key, state)
                for key, state in sorted(self._relationships.items())
                if (
                    self._current_track_id is None
                    or key[0] == self._current_track_id
                )
            )
        )
        relationship_by_next = {
            relationship.next_track_id: relationship
            for relationship in relationships
        }
        candidates: list[NextGateCandidate] = []
        provisional: list[str] = []
        for track in sorted(tracks.values(), key=lambda item: item.track_id):
            if (
                track.track_id == self._current_track_id
                or track.role is VisualTrackRole.RETIRED
            ):
                continue
            relation = relationship_by_next.get(track.track_id)
            pretracked = _pretracked_candidate_evidence(
                track,
                current_tracker_frame_sequence=frame_sequence,
                stability_target=self.config.min_next_candidate_frames,
            )
            if pretracked is not None and (
                track.role is VisualTrackRole.NEXT or not track.visible
            ):
                score = _candidate_score(
                    track,
                    relation,
                    self.config.min_next_candidate_frames,
                    stable_frame_count=pretracked.stable_frame_count,
                    confidence=pretracked.confidence,
                    association_confidence=pretracked.association_confidence,
                )
                promotable = (
                    pretracked.stable_frame_count
                    >= self.config.min_next_candidate_frames
                    and pretracked.confidence >= self.config.min_track_confidence
                    and pretracked.association_confidence
                    >= self.config.min_association_confidence
                    and relation is not None
                    and relation.observation_count
                    >= self.config.min_next_candidate_frames
                    and relation.observation_confidence
                    >= self.config.min_relationship_confidence
                    and not relation.contended
                    and (
                        relation.basis
                        is GateRelationshipBasis.SIMULTANEOUS_IMAGE
                        or (
                            pretracked.missed_camera_publications == 0
                            and relation.sequential_observation_count
                            >= self.config.min_next_candidate_frames
                            and relation.latest_tracker_frame_sequence
                            == frame_sequence
                            and _adjacent_handoff_within_confirmation_horizon(
                                current,
                                track,
                                relation,
                                frame_sequence=frame_sequence,
                            )
                        )
                    )
                )
                candidates.append(
                    NextGateCandidate(
                        track_id=track.track_id,
                        score=score,
                        stable_frame_count=pretracked.stable_frame_count,
                        first_token=track.first_token,
                        latest_token=track.latest_token,
                        bearing_norm=track.bearing_norm,
                        elevation_norm=track.elevation_norm,
                        bearing_rate_norm_s=track.bearing_rate_norm_s,
                        elevation_rate_norm_s=track.elevation_rate_norm_s,
                        apparent_scale=track.apparent_scale,
                        log_scale_rate_s=track.log_scale_rate_s,
                        confidence=pretracked.confidence,
                        association_confidence=pretracked.association_confidence,
                        center_censored=track.center_censored,
                        promotable=promotable,
                        relationship=relation,
                    )
                )
            elif track.visible:
                provisional.append(track.track_id)
        candidates.sort(key=lambda item: (-item.score, item.track_id))
        ambiguous_next = (
            len(candidates) > 1
            and candidates[0].score - candidates[1].score
            < self.config.candidate_selection_margin
        )
        reason: Optional[str] = None
        usable = True
        if self._race_finished:
            usable = False
            reason = "race_finished"
        elif self._phase is GateGraphPhase.CREDITED_UNBOUND:
            usable = False
            reason = "credited_gate_unbound"
        elif current is None:
            usable = False
            reason = "current_gate_unbound"
        elif current.role is VisualTrackRole.RETIRED:
            usable = False
            reason = "current_track_retired"
        elif current.ambiguous:
            usable = False
            reason = "current_track_ambiguous"
        elif not current.visible:
            usable = False
            reason = "current_track_not_visible"
        return GateGraphSnapshot(
            tracker_frame_sequence=frame_sequence,
            latest_camera_token=token,
            current_track_id=self._current_track_id,
            current_gate_index=self._current_gate_index,
            current_track=current,
            next_candidates=tuple(candidates),
            provisional_track_ids=tuple(provisional),
            relationships=relationships,
            confirmed_transitions=tuple(self._transitions),
            next_selection_ambiguous=ambiguous_next,
            authority_usable=usable,
            withholding_reason=reason,
            race_finished=self._race_finished,
            latest_race_status=self._last_race_status,
            phase=self._phase,
            pending_unbound_advance=self._pending_unbound_advance,
        )

    @staticmethod
    def _relationship(
        key: tuple[str, str],
        state: _RelationshipState,
    ) -> ObservedGateRelationship:
        return ObservedGateRelationship(
            current_track_id=key[0],
            next_track_id=key[1],
            basis=state.basis,
            current_anchor_token=state.current_anchor_token,
            next_anchor_token=state.next_anchor_token,
            anchor_publication_delta=state.anchor_publication_delta,
            anchor_time_gap_ns=state.anchor_time_gap_ns,
            first_token=state.first_token,
            latest_token=state.latest_token,
            observation_count=state.observation_count,
            simultaneous_observation_count=state.simultaneous_observation_count,
            sequential_observation_count=state.sequential_observation_count,
            latest_tracker_frame_sequence=state.latest_tracker_frame_sequence,
            current_bearing_norm=state.current_bearing_norm,
            current_elevation_norm=state.current_elevation_norm,
            next_bearing_norm=state.next_bearing_norm,
            next_elevation_norm=state.next_elevation_norm,
            relative_bearing_norm=(
                state.next_bearing_norm - state.current_bearing_norm
            ),
            relative_elevation_norm=(
                state.next_elevation_norm - state.current_elevation_norm
            ),
            current_apparent_scale=state.current_apparent_scale,
            next_apparent_scale=state.next_apparent_scale,
            current_log_scale_rate_s=state.current_log_scale_rate_s,
            next_log_scale_rate_s=state.next_log_scale_rate_s,
            observation_confidence=state.observation_confidence,
            current_center_censored=state.current_center_censored,
            next_center_censored=state.next_center_censored,
            fresh=state.fresh,
            contended=state.contended,
        )

    def _validate_unchanged_race_status(
        self,
        current: AuthoritativeRaceStatusRef,
        *,
        expected_gate_index: int,
    ) -> None:
        """Accept the graph baseline or one newer heartbeat at the same gate."""

        previous = self._last_race_status
        if previous is None:
            raise GateGraphError(
                "same-gate rebind lacks an authoritative race baseline"
            )
        if current.race_finished:
            raise GateGraphError("race finished during same-gate rebind")
        if current.active_gate_index != expected_gate_index:
            raise GateGraphError(
                "authoritative race gate changed during same-gate rebind"
            )
        if current == previous:
            return
        self._validate_race_advance(current, allow_same_gate=True)
        if current.active_gate_index != previous.active_gate_index:
            raise GateGraphError(
                "authoritative race gate changed during same-gate rebind"
            )

    def _validate_race_advance(
        self,
        current: AuthoritativeRaceStatusRef,
        *,
        allow_same_gate: bool = False,
    ) -> None:
        previous = self._last_race_status
        if previous is None:
            raise GateGraphError("race transition lacks a baseline status")
        if (
            current.session_id != previous.session_id
            or current.reset_epoch != previous.reset_epoch
            or current.provenance_basis is not previous.provenance_basis
        ):
            raise GateGraphError("race status crossed its proved epoch")
        if current.race_status_boot_ms <= previous.race_status_boot_ms:
            raise GateGraphError("race boot time did not strictly advance")
        if current.provenance_basis is RaceStatusProvenanceBasis.LIVE_INGRESS:
            if current.race_generation != previous.race_generation:
                raise GateGraphError("race ingress generation changed")
            if current.host_clock_id != previous.host_clock_id:
                raise GateGraphError("race ingress host clock changed")
            assert current.race_status_sequence is not None
            assert previous.race_status_sequence is not None
            assert current.received_monotonic_ns is not None
            assert previous.received_monotonic_ns is not None
            if current.race_status_sequence <= previous.race_status_sequence:
                raise GateGraphError("race ingress sequence did not advance")
            if current.received_monotonic_ns <= previous.received_monotonic_ns:
                raise GateGraphError("race ingress receipt time did not advance")
        else:
            assert current.legacy_event_order is not None
            assert previous.legacy_event_order is not None
            assert current.event_wall_time_ns is not None
            assert previous.event_wall_time_ns is not None
            if current.legacy_event_order <= previous.legacy_event_order:
                raise GateGraphError("legacy race event order did not advance")
            if current.event_wall_time_ns <= previous.event_wall_time_ns:
                raise GateGraphError("legacy race wall-time token did not advance")
        if not allow_same_gate and current.active_gate_index == previous.active_gate_index:
            raise GateGraphError("race gate index did not advance")
        if current.active_gate_index < previous.active_gate_index:
            raise GateGraphError("race gate index regressed")

    @staticmethod
    def _validate_camera_precedes_race(
        tracker: MultiTargetVisualTracker,
        token: CameraFrameToken,
        race_status: AuthoritativeRaceStatusRef,
    ) -> None:
        frame_basis = tracker.frame_provenance_basis(token)
        if race_status.provenance_basis is RaceStatusProvenanceBasis.LIVE_INGRESS:
            if frame_basis is not FrameProvenanceBasis.RECEIVER_TIMING_V1:
                raise GateGraphError("live race ingress cannot authorize legacy camera timing")
            assert race_status.received_monotonic_ns is not None
            if tracker.time_basis_id != race_status.host_clock_id:
                raise GateGraphError("camera and race ingress host clocks differ")
            publish_monotonic_ns = tracker.frame_publish_time_ns(token)
            if publish_monotonic_ns is None:
                raise GateGraphError("live frame lacks publication time")
            if publish_monotonic_ns > race_status.received_monotonic_ns:
                raise GateGraphError("camera publication postdates race credit ingress")
        elif frame_basis is not FrameProvenanceBasis.LEGACY_CAPTURE:
            raise GateGraphError("legacy race capture cannot authorize live frame timing")

    @staticmethod
    def _tracker(tracker: MultiTargetVisualTracker) -> None:
        if type(tracker) is not MultiTargetVisualTracker:
            raise TypeError("tracker must be an exact MultiTargetVisualTracker")

    @staticmethod
    def _race_ref(race_status: AuthoritativeRaceStatusRef) -> None:
        if type(race_status) is not AuthoritativeRaceStatusRef:
            raise TypeError("race_status must be an exact AuthoritativeRaceStatusRef")


def _promotion_credit_boundary(
    track: VisualTrack,
    anchor: CameraFrameToken,
    race_status: AuthoritativeRaceStatusRef,
) -> _PromotionCreditBoundary:
    """Split one immutable track history at the authoritative credit boundary.

    ``anchor`` is the global camera watermark and therefore need not be a
    target observation.  For live ingress, the target prefix is maximal by
    host-monotonic publication time.  Already-processed observations may
    follow race receipt; they remain in the tracker's bounded full-history
    promotion snapshot but can never be relabelled as pre-credit evidence.
    A frame observed before receipt but published just after it is neutral
    boundary history: it is neither pre-credit evidence nor post-credit
    command authority.
    """

    history = track.history
    if not history:
        raise GateGraphError("promotion track has no visual history")
    is_live = (
        race_status.provenance_basis is RaceStatusProvenanceBasis.LIVE_INGRESS
    )
    received_ns = race_status.received_monotonic_ns
    eligibility: list[bool] = []
    for sample in history:
        token_precedes_credit = _token_precedes_or_equals(sample.token, anchor)
        if is_live:
            assert received_ns is not None
            if (
                sample.provenance_basis
                is not FrameProvenanceBasis.RECEIVER_TIMING_V1
            ):
                raise GateGraphError(
                    "live promotion history contains legacy provenance"
                )
            publish_ns = sample.publication_monotonic_ns
            if publish_ns is None:
                raise GateGraphError(
                    "live promotion history lacks publication provenance"
                )
            publication_precedes_credit = publish_ns <= received_ns
            if token_precedes_credit != publication_precedes_credit:
                raise GateGraphError(
                    "target history disagrees with the camera credit watermark"
                )
            eligibility.append(publication_precedes_credit)
        else:
            if (
                sample.provenance_basis
                is not FrameProvenanceBasis.LEGACY_CAPTURE
                or sample.publication_monotonic_ns is not None
            ):
                raise GateGraphError(
                    "legacy promotion history invented live publication timing"
                )
            eligibility.append(token_precedes_credit)

    first_ineligible = next(
        (index for index, eligible in enumerate(eligibility) if not eligible),
        len(history),
    )
    if any(eligibility[first_ineligible:]):
        raise GateGraphError("target credit samples do not form an exact prefix")
    credit_prefix = history[:first_ineligible]
    post_credit_suffix = history[first_ineligible:]
    if not credit_prefix:
        raise GateGraphError("promotion has no target observation before credit")
    if not is_live and post_credit_suffix:
        raise GateGraphError(
            "legacy capture cannot prove post-credit promotion samples"
        )
    if is_live:
        assert received_ns is not None
        observed_strictly_post_credit = False
        for sample in post_credit_suffix:
            assert sample.publication_monotonic_ns is not None
            if sample.publication_monotonic_ns <= received_ns:
                raise GateGraphError(
                    "promotion suffix publication does not strictly postdate "
                    "race receipt"
                )
            if not _token_strictly_precedes(anchor, sample.token):
                raise GateGraphError(
                    "post-credit target sample does not follow the camera "
                    "watermark"
                )
            if sample.observation_monotonic_ns <= received_ns:
                if observed_strictly_post_credit:
                    raise GateGraphError(
                        "neutral credit-boundary samples are not a prefix"
                    )
                continue
            observed_strictly_post_credit = True

    tail = [credit_prefix[-1]]
    for sample in reversed(credit_prefix[:-1]):
        if sample.tracker_frame_sequence != tail[-1].tracker_frame_sequence - 1:
            break
        tail.append(sample)
    tail.reverse()
    return _PromotionCreditBoundary(
        credit_prefix=credit_prefix,
        pretransition_tail=tuple(tail),
        post_credit_suffix=post_credit_suffix,
    )


def _validate_transition_history(
    transition: ConfirmedGateTransition,
    history: tuple[VisualTrackSample, ...],
    boundary: _PromotionCreditBoundary,
) -> None:
    """Fail closed if construction fields drift from the frozen full history."""

    if (
        transition.history_length_before_promotion != len(history)
        or transition.history_length_after_promotion != len(history)
        or transition.promoted_history_sha256
        != visual_track_history_sha256(history)
        or transition.promoted_first_token != history[0].token
        or transition.promoted_latest_token_at_promotion != history[-1].token
        or transition.promoted_history_length_at_credit
        != len(boundary.credit_prefix)
        or transition.promoted_latest_token_before_credit
        != boundary.credit_prefix[-1].token
        or transition.pretransition_frame_tokens
        != tuple(sample.token for sample in boundary.pretransition_tail)
        or history[: transition.promoted_history_length_at_credit]
        != boundary.credit_prefix
        or history[transition.promoted_history_length_at_credit :]
        != boundary.post_credit_suffix
    ):
        raise RuntimeError(
            "confirmed transition fields disagree with frozen visual history"
        )


def _adjacent_handoff_anchors(
    current: VisualTrack,
    candidate: VisualTrack,
    *,
    frame_sequence: int,
    token: CameraFrameToken,
    config: RollingGateGraphConfig,
) -> Optional[tuple[VisualTrackSample, VisualTrackSample, int]]:
    """Return exact sequential anchors only for the observed crossing boundary.

    This is deliberately narrower than ordinary visual association.  It can
    seed a graph edge between two different identities, but can never merge
    them or label the successor with a gate index.  Race status remains the
    sole authority for that later promotion.
    """

    if (
        current.visible
        or current.role is not VisualTrackRole.CURRENT
        or current.authoritative_gate_index is None
        or current.ambiguous
        or current.missed_frame_count != 1
        or not current.history
        or not candidate.visible
        or candidate.role is VisualTrackRole.RETIRED
        or candidate.authoritative_gate_index is not None
        or candidate.ambiguous
        or candidate.missed_frame_count != 0
        or candidate.consecutive_frame_count != 1
        or candidate.total_observation_count != 1
        or len(candidate.history) != 1
        or candidate.first_token != token
        or candidate.latest_token != token
    ):
        return None
    current_sample = current.history[-1]
    candidate_sample = candidate.history[0]
    if (
        current_sample.tracker_frame_sequence != frame_sequence - 1
        or candidate_sample.tracker_frame_sequence != frame_sequence
        or current_sample.provenance_basis
        is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        or candidate_sample.provenance_basis
        is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        or current_sample.confidence < config.min_track_confidence
        or candidate_sample.confidence < config.min_track_confidence
        or current_sample.association_confidence
        < config.min_association_confidence
        or candidate_sample.association_confidence
        < config.min_association_confidence
        or not _is_aperture_filling_crossing_sample(current_sample)
        or candidate_sample.center_censored
        or candidate_sample.apparent_scale
        > (
            current_sample.apparent_scale
            * _ADJACENT_HANDOFF_MAX_NEXT_SCALE_RATIO
        )
        or not _tokens_are_adjacent_publications(
            current_sample.token,
            candidate_sample.token,
        )
    ):
        return None
    current_publish_ns = current_sample.publication_monotonic_ns
    candidate_publish_ns = candidate_sample.publication_monotonic_ns
    if current_publish_ns is None or candidate_publish_ns is None:
        return None
    gap_ns = candidate_publish_ns - current_publish_ns
    if gap_ns <= 0 or gap_ns > _ADJACENT_HANDOFF_MAX_GAP_NS:
        return None
    return current_sample, candidate_sample, gap_ns


def _is_recent_handoff_contender(
    track: VisualTrack,
    *,
    current_tracker_frame_sequence: int,
) -> bool:
    """Keep recent identities in contention even when their match is ambiguous."""

    if not track.history:
        return False
    latest_sequence = track.history[-1].tracker_frame_sequence
    missed_publications = current_tracker_frame_sequence - latest_sequence
    return (
        0 <= missed_publications <= _MAX_PROMOTION_MISSED_CAMERA_PUBLICATIONS
        and missed_publications == track.missed_frame_count
    )


def _is_aperture_filling_crossing_sample(sample: VisualTrackSample) -> bool:
    left, top, right, bottom = sample.bbox_norm
    return (
        sample.center_censored
        and sample.clipping & _ALL_FRAME_EDGES == _ALL_FRAME_EDGES
        and sample.apparent_scale >= _ADJACENT_HANDOFF_MIN_CURRENT_SCALE
        and left <= _ADJACENT_HANDOFF_BBOX_EDGE_TOLERANCE
        and top <= _ADJACENT_HANDOFF_BBOX_EDGE_TOLERANCE
        and right >= 1.0 - _ADJACENT_HANDOFF_BBOX_EDGE_TOLERANCE
        and bottom >= 1.0 - _ADJACENT_HANDOFF_BBOX_EDGE_TOLERANCE
    )


def _tokens_are_adjacent_publications(
    predecessor: CameraFrameToken,
    successor: CameraFrameToken,
) -> bool:
    return (
        predecessor.stream_id is not None
        and predecessor.stream_id == successor.stream_id
        and predecessor.generation == successor.generation
        and predecessor.publication_sequence is not None
        and successor.publication_sequence
        == predecessor.publication_sequence + 1
    )


def _adjacent_handoff_confidence(
    current_sample: VisualTrackSample,
    candidate_sample: VisualTrackSample,
) -> float:
    confidence = math.sqrt(
        max(0.0, current_sample.confidence)
        * max(0.0, candidate_sample.confidence)
    ) * min(
        current_sample.association_confidence,
        candidate_sample.association_confidence,
    )
    if current_sample.center_censored or candidate_sample.center_censored:
        confidence *= 0.75
    return confidence * _ADJACENT_HANDOFF_CONFIDENCE_FACTOR


def _adjacent_handoff_step_is_contiguous(
    previous: _RelationshipState,
    candidate: VisualTrack,
    *,
    frame_sequence: int,
    token: CameraFrameToken,
) -> bool:
    """Require every successor observation to preserve live publication order."""

    if (
        candidate.ambiguous
        or not candidate.visible
        or candidate.latest_token != token
        or len(candidate.history) < 2
        or previous.latest_tracker_frame_sequence != frame_sequence - 1
        or not _tokens_are_adjacent_publications(previous.latest_token, token)
    ):
        return False
    predecessor_sample = candidate.history[-2]
    successor_sample = candidate.history[-1]
    if (
        predecessor_sample.token != previous.latest_token
        or predecessor_sample.tracker_frame_sequence != frame_sequence - 1
        or successor_sample.token != token
        or successor_sample.tracker_frame_sequence != frame_sequence
        or predecessor_sample.provenance_basis
        is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        or successor_sample.provenance_basis
        is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        or predecessor_sample.publication_monotonic_ns is None
        or successor_sample.publication_monotonic_ns is None
    ):
        return False
    gap_ns = (
        successor_sample.publication_monotonic_ns
        - predecessor_sample.publication_monotonic_ns
    )
    return 0 < gap_ns <= _ADJACENT_HANDOFF_MAX_GAP_NS


def _adjacent_handoff_within_confirmation_horizon(
    current: Optional[VisualTrack],
    candidate: VisualTrack,
    relationship: ObservedGateRelationship,
    *,
    frame_sequence: int,
) -> bool:
    """Keep the sequential exception inside the predecessor's live lease."""

    if (
        current is None
        or current.role is not VisualTrackRole.CURRENT
        or current.visible
        or current.ambiguous
        or current.missed_frame_count < 1
        or current.missed_frame_count
        > _ADJACENT_HANDOFF_MAX_CONFIRMATION_PUBLICATIONS
        or not current.history
        or not candidate.visible
        or candidate.ambiguous
        or candidate.latest_token != relationship.latest_token
        or relationship.contended
    ):
        return False
    current_anchor = current.history[-1]
    if (
        current_anchor.token != relationship.current_anchor_token
        or frame_sequence - current_anchor.tracker_frame_sequence
        != current.missed_frame_count
        or frame_sequence - current_anchor.tracker_frame_sequence
        > _ADJACENT_HANDOFF_MAX_CONFIRMATION_PUBLICATIONS
    ):
        return False
    anchor_publication = relationship.current_anchor_token.publication_sequence
    latest_publication = relationship.latest_token.publication_sequence
    return (
        anchor_publication is not None
        and latest_publication is not None
        and 1
        <= latest_publication - anchor_publication
        <= _ADJACENT_HANDOFF_MAX_CONFIRMATION_PUBLICATIONS
    )


def _validate_adjacent_handoff_credit_freshness(
    candidate: NextGateCandidate,
    pretransition_samples: tuple,
    race_status: AuthoritativeRaceStatusRef,
) -> None:
    relationship = candidate.relationship
    if (
        relationship is None
        or relationship.basis
        is not GateRelationshipBasis.ADJACENT_PUBLICATION_HANDOFF
    ):
        return
    if race_status.provenance_basis is not RaceStatusProvenanceBasis.LIVE_INGRESS:
        raise GateGraphError(
            "adjacent handoff promotion requires live race ingress"
        )
    assert race_status.received_monotonic_ns is not None
    latest_publish_ns = pretransition_samples[-1].publication_monotonic_ns
    if latest_publish_ns is None:
        raise GateGraphError(
            "adjacent handoff promotion lacks live publication time"
        )
    credit_age_ns = race_status.received_monotonic_ns - latest_publish_ns
    if (
        credit_age_ns < 0
        or credit_age_ns > _ADJACENT_HANDOFF_MAX_CREDIT_AGE_NS
    ):
        raise GateGraphError(
            "adjacent handoff candidate is stale at race credit"
        )


def _pretracked_candidate_evidence(
    track: VisualTrack,
    *,
    current_tracker_frame_sequence: int,
    stability_target: int,
) -> Optional[_PretrackedCandidateEvidence]:
    """Return exact last-observation proof within the bounded miss grace.

    A tracker miss intentionally clears its live association confidence and
    consecutive count.  Promotion may nevertheless use the immediately
    preceding exact observation streak for at most two subsequently processed
    camera publications.  The grace cannot create proof: ambiguity, a broken
    observation streak, insufficient confidence, or any wider publication gap
    returns no candidate.
    """

    if track.ambiguous or not track.history:
        return None
    latest_sample = track.history[-1]
    missed_publications = (
        current_tracker_frame_sequence - latest_sample.tracker_frame_sequence
    )
    if (
        missed_publications < 0
        or missed_publications > _MAX_PROMOTION_MISSED_CAMERA_PUBLICATIONS
        or missed_publications != track.missed_frame_count
    ):
        return None
    tail_count = 1
    expected_sequence = latest_sample.tracker_frame_sequence - 1
    for sample in reversed(track.history[:-1]):
        if sample.tracker_frame_sequence != expected_sequence:
            break
        tail_count += 1
        expected_sequence -= 1
    if tail_count < stability_target:
        return None
    return _PretrackedCandidateEvidence(
        stable_frame_count=tail_count,
        missed_camera_publications=missed_publications,
        confidence=latest_sample.confidence,
        association_confidence=latest_sample.association_confidence,
    )


def _reacquisition_observable_tail(
    track: VisualTrack,
    *,
    camera_token_at_binding: CameraFrameToken,
    required_frames: int,
    min_track_confidence: float,
    min_association_confidence: float,
) -> Optional[tuple[VisualTrackSample, ...]]:
    """Return a clean or single-edge local tail eligible for reacquisition."""

    if (
        track.role is VisualTrackRole.RETIRED
        or track.authoritative_gate_index is not None
        or not track.visible
        or track.ambiguous
        or track.missed_frame_count != 0
        or track.clipping
        not in {
            FrameEdge.NONE,
            FrameEdge.LEFT,
            FrameEdge.TOP,
            FrameEdge.RIGHT,
            FrameEdge.BOTTOM,
        }
        or (
            track.clipping == FrameEdge.NONE
            and track.center_censored
        )
        or track.latest_token != camera_token_at_binding
        or track.consecutive_frame_count < required_frames
        or track.confidence < min_track_confidence
        or track.association_confidence < min_association_confidence
        or len(track.history) < required_frames
    ):
        return None
    if track.clipping != FrameEdge.NONE:
        observable_axis = (
            0
            if track.clipping in {FrameEdge.TOP, FrameEdge.BOTTOM}
            else 1
        )
        if (
            not math.isfinite(float(track.center_norm[observable_axis]))
            or abs(float(track.center_norm[observable_axis])) > 1.0
            or not math.isfinite(
                float(track.center_velocity_norm_s[observable_axis])
            )
        ):
            return None
    tail = track.history[-required_frames:]
    if tail[-1].token != camera_token_at_binding:
        return None
    if any(
        sample.clipping not in {FrameEdge.NONE, track.clipping}
        or (
            sample.clipping == FrameEdge.NONE
            and sample.center_censored
        )
        or sample.confidence < min_track_confidence
        or sample.association_confidence < min_association_confidence
        or (
            sample.accepted_association is not None
            and sample.accepted_association.ambiguous
        )
        for sample in tail
    ):
        return None
    for predecessor, successor in zip(tail, tail[1:]):
        if (
            successor.tracker_frame_sequence
            != predecessor.tracker_frame_sequence + 1
            or not _token_strictly_precedes(
                predecessor.token,
                successor.token,
            )
        ):
            return None
    return tail


def _same_gate_rebind_observable_tail(
    track: VisualTrack,
    *,
    search: SameGateRebindSearch,
    camera_token_at_binding: CameraFrameToken,
    min_track_confidence: float,
    min_association_confidence: float,
) -> Optional[tuple[VisualTrackSample, ...]]:
    """Return one wholly post-search, contiguous local identity tail."""

    required_frames = search.required_stable_frames
    if (
        track.track_id in search.excluded_track_ids_at_start
        or track.role is VisualTrackRole.RETIRED
        or track.authoritative_gate_index is not None
        or not track.visible
        or track.ambiguous
        or track.missed_frame_count != 0
        or track.clipping
        not in {
            FrameEdge.NONE,
            FrameEdge.LEFT,
            FrameEdge.TOP,
            FrameEdge.RIGHT,
            FrameEdge.BOTTOM,
        }
        or (
            track.clipping == FrameEdge.NONE
            and track.center_censored
        )
        or track.latest_token != camera_token_at_binding
        or not _token_strictly_precedes(
            search.camera_token_at_start,
            track.first_token,
        )
        or track.consecutive_frame_count < required_frames
        or track.confidence < min_track_confidence
        or track.association_confidence < min_association_confidence
        or len(track.history) < required_frames
        or track.history[0].token != track.first_token
    ):
        return None
    tail = track.history[-required_frames:]
    if tail[-1].token != camera_token_at_binding:
        return None
    expected_provenance = (
        FrameProvenanceBasis.RECEIVER_TIMING_V1
        if (
            search.race_status_at_start.provenance_basis
            is RaceStatusProvenanceBasis.LIVE_INGRESS
        )
        else FrameProvenanceBasis.LEGACY_CAPTURE
    )
    if any(
        sample.provenance_basis is not expected_provenance
        or not _token_strictly_precedes(
            search.camera_token_at_start,
            sample.token,
        )
        or sample.confidence < min_track_confidence
        or sample.association_confidence < min_association_confidence
        or (
            sample.accepted_association is not None
            and sample.accepted_association.ambiguous
        )
        for sample in tail
    ):
        return None
    for predecessor, successor in zip(tail, tail[1:]):
        if (
            successor.tracker_frame_sequence
            != predecessor.tracker_frame_sequence + 1
            or not _token_strictly_precedes(
                predecessor.token,
                successor.token,
            )
        ):
            return None
    return tail


def _same_gate_rebind_matches_departure_half_plane(
    lost: VisualTrack,
    stable_tail: tuple[VisualTrackSample, ...],
) -> bool:
    """Reject a new identity that appears across the optical axis.

    A wholly off-axis current can disappear and later acquire a new local
    tracker identity, but an opposite-side contour is not a defensible
    same-gate continuation.  This is a fixed geometric boundary rather than
    a timer: when the lost current spans the optical axis, no side constraint
    is imposed.
    """

    if (
        type(lost) is not VisualTrack
        or not lost.history
        or type(stable_tail) is not tuple
        or not stable_tail
        or any(type(sample) is not VisualTrackSample for sample in stable_tail)
    ):
        raise TypeError(
            "same-gate departure-side continuity inputs are invalid"
        )

    last_visible = lost.history[-1]
    has_left = bool(last_visible.clipping & FrameEdge.LEFT)
    has_right = bool(last_visible.clipping & FrameEdge.RIGHT)
    departure_side = 0
    if has_left != has_right:
        departure_side = -1 if has_left else 1
    else:
        left, _top, right, _bottom = last_visible.bbox_norm
        if right < 0.5:
            departure_side = -1
        elif left > 0.5:
            departure_side = 1

    if departure_side == 0:
        return True
    return all(
        departure_side * float(sample.center_norm[0]) >= 0.0
        for sample in stable_tail
    )


def _token_precedes_or_equals(
    sample: CameraFrameToken,
    anchor: CameraFrameToken,
) -> bool:
    if (
        sample.generation != anchor.generation
        or sample.stream_id != anchor.stream_id
        or (
            (sample.publication_sequence is None)
            != (anchor.publication_sequence is None)
        )
    ):
        return False
    if (
        sample.publication_sequence is not None
        and anchor.publication_sequence is not None
    ):
        return sample.publication_sequence <= anchor.publication_sequence
    # Legacy frame IDs are exact identity and the replay batch validates strict
    # receipt/source ordering separately.  UInt32 wrap is not crossed inside a
    # bounded captured transition excerpt.
    return sample.frame_id <= anchor.frame_id


def _token_strictly_precedes(
    sample: CameraFrameToken,
    anchor: CameraFrameToken,
) -> bool:
    return (
        sample != anchor
        and _token_precedes_or_equals(sample, anchor)
    )


def _candidate_score(
    track: VisualTrack,
    relationship: Optional[ObservedGateRelationship],
    stability_target: int,
    *,
    stable_frame_count: int,
    confidence: float,
    association_confidence: float,
) -> float:
    stability = min(1.0, stable_frame_count / stability_target)
    relationship_confidence = (
        0.0 if relationship is None else relationship.observation_confidence
    )
    relationship_stability = (
        0.0
        if relationship is None
        else min(1.0, relationship.observation_count / stability_target)
    )
    censor_factor = 0.85 if track.center_censored else 1.0
    # Deliberately excludes apparent area/scale: size is evidence, not semantic
    # next-gate identity.
    return censor_factor * (
        0.30 * stability
        + 0.25 * confidence
        + 0.20 * association_confidence
        + 0.15 * relationship_confidence
        + 0.10 * relationship_stability
    )


def _finite(
    value: object,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise TypeError(f"{label} must be finite numeric data")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _history_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be a lowercase SHA-256 hex string")
    return value


DEFAULT_ROLLING_GATE_GRAPH_CONFIG = RollingGateGraphConfig()
