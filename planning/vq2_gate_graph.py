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
    pretransition_frame_tokens: tuple[CameraFrameToken, ...]
    history_length_before_promotion: int
    history_length_after_promotion: int
    promoted_history_sha256: str


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
    confirmed_transitions: tuple[ConfirmedGateTransition, ...]
    next_selection_ambiguous: bool
    authority_usable: bool
    withholding_reason: Optional[str]
    race_finished: bool
    latest_race_status: Optional[AuthoritativeRaceStatusRef]


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


class RollingVisualGateGraph:
    """Maintain current/next visual relationships and proof-bound promotion."""

    def __init__(self, config: Optional[RollingGateGraphConfig] = None) -> None:
        if config is None:
            config = DEFAULT_ROLLING_GATE_GRAPH_CONFIG
        if type(config) is not RollingGateGraphConfig:
            raise TypeError("config must be an exact RollingGateGraphConfig")
        self.config = config
        self._current_track_id: Optional[str] = None
        self._current_gate_index: Optional[int] = None
        self._relationships: dict[tuple[str, str], _RelationshipState] = {}
        self._transitions: list[ConfirmedGateTransition] = []
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
        if self._current_track_id is not None:
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
        if self._current_track_id is None or self._current_gate_index is None:
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
                raise GateGraphError("requested promotion track is not promotable")
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
        pretransition_samples = _pretransition_tail(
            promoted_before,
            camera_token_at_credit,
            race_status,
        )
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
            promoted_latest_token_before_credit=promoted_before.latest_token,
            pretransition_frame_tokens=tuple(
                sample.token for sample in pretransition_samples
            ),
            history_length_before_promotion=len(promoted_before.history),
            history_length_after_promotion=len(promoted_after.history),
            promoted_history_sha256=visual_track_history_sha256(
                promoted_before.history
            ),
        )
        self._transitions.append(transition)
        if len(self._transitions) > self.config.relationship_history_limit:
            del self._transitions[
                : len(self._transitions) - self.config.relationship_history_limit
            ]
        self._current_track_id = selected.track_id
        self._current_gate_index = race_status.active_gate_index
        self._last_race_status = race_status
        self.observe(tracker)
        return transition

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
        self._validate_race_advance(race_status, allow_same_gate=True)
        if not tracker.has_processed_token(camera_token_at_finish):
            raise GateGraphError("finish references an unprocessed camera token")
        self._validate_camera_precedes_race(
            tracker,
            camera_token_at_finish,
            race_status,
        )
        self._race_finished = True
        self._last_race_status = race_status
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
        relationships = tuple(
            self._relationship(key, state)
            for key, state in sorted(self._relationships.items())
            if self._current_track_id is None or key[0] == self._current_track_id
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


def _pretransition_tail(
    track: VisualTrack,
    anchor: CameraFrameToken,
    race_status: AuthoritativeRaceStatusRef,
) -> tuple:
    anchor_sample = next(
        (sample for sample in track.history if sample.token == anchor),
        None,
    )
    if anchor_sample is None:
        # The candidate need not be detected on the exact credit-anchor frame;
        # retain its latest earlier observation as long as provenance is exact.
        anchor_sequence = max(
            (
                sample.tracker_frame_sequence
                for sample in track.history
                if _token_precedes_or_equals(sample.token, anchor)
            ),
            default=-1,
        )
    else:
        anchor_sequence = anchor_sample.tracker_frame_sequence
    eligible = [
        sample
        for sample in track.history
        if sample.tracker_frame_sequence <= anchor_sequence
    ]
    if race_status.provenance_basis is RaceStatusProvenanceBasis.LIVE_INGRESS:
        assert race_status.received_monotonic_ns is not None
        eligible = [
            sample
            for sample in eligible
            if sample.publication_monotonic_ns is not None
            and sample.publication_monotonic_ns <= race_status.received_monotonic_ns
        ]
    if not eligible:
        return ()
    tail = [eligible[-1]]
    for sample in reversed(eligible[:-1]):
        if sample.tracker_frame_sequence != tail[-1].tracker_frame_sequence - 1:
            break
        tail.append(sample)
    tail.reverse()
    return tuple(tail)


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


def _token_precedes_or_equals(
    sample: CameraFrameToken,
    anchor: CameraFrameToken,
) -> bool:
    if sample.generation != anchor.generation:
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


DEFAULT_ROLLING_GATE_GRAPH_CONFIG = RollingGateGraphConfig()
