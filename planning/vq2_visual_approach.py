"""Pure Gate-0 visual-approach adapter for the rolling VQ2 gate graph.

The adapter is deliberately narrower than the live runner.  It binds one
authoritative current identity for one segment, admits at most one exact
same-publication next-gate target, and delegates image-space control to
``ImageVisualServo`` with forward advance disabled.  It owns no transport,
race transition, reset, watchdog, collision, or cleanup authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualTrack,
    VisualTrackRole,
    VisualTrackerUpdate,
)
from planning.vq2_gate_graph import (
    GateGraphSnapshot,
    GateRelationshipBasis,
    NextGateCandidate,
    RaceStatusProvenanceBasis,
)
from planning.vq2_visual_servo import (
    ImageVisualServo,
    MAX_NEXT_GATE_BLEND,
    MAX_VISUAL_OBSERVATION_AGE_S,
    VisualServoOutput,
    VisualServoRefusal,
    VisualServoTuning,
    VisualTarget,
)


_QPC_TIME_BASIS_ID = "host-perf-counter"
_FUTURE_TOLERANCE_S = 1e-6
_REQUIRED_NEXT_FRAMES = 3


class VisualApproachRefusal(ValueError):
    """The graph/tracker pair cannot safely authorize an approach proposal."""


class VisualApproachCurrentGeometryUnavailable(VisualApproachRefusal):
    """The optional visual blend must withdraw from a clipped current gate."""


@dataclass(frozen=True, slots=True)
class VisualApproachProposal:
    """One exact-publication, no-advance visual-approach proposal."""

    current_target: VisualTarget
    next_target: Optional[VisualTarget]
    servo_output: VisualServoOutput
    candidate_track_ids: tuple[str, ...]
    provisional_track_ids: tuple[str, ...]
    withholding_reason: Optional[str]
    relationship_basis: Optional[GateRelationshipBasis]
    latched_next_track_id: Optional[str]


class RollingVisualApproachServo:
    """Keep current identity fixed while cautiously blending one next gate."""

    def __init__(
        self,
        expected_current_track_id: str,
        expected_gate_index: int,
        tuning: Optional[VisualServoTuning] = None,
    ) -> None:
        if (
            type(expected_current_track_id) is not str
            or not expected_current_track_id
            or len(expected_current_track_id) > 128
        ):
            raise VisualApproachRefusal(
                "expected current track id must be a bounded string"
            )
        if type(expected_gate_index) is not int or expected_gate_index < 0:
            raise VisualApproachRefusal(
                "expected gate index must be a nonnegative exact integer"
            )
        if tuning is not None and type(tuning) is not VisualServoTuning:
            raise VisualApproachRefusal(
                "tuning must be an exact VisualServoTuning or None"
            )
        self.expected_current_track_id = expected_current_track_id
        self.expected_gate_index = expected_gate_index
        self._servo = ImageVisualServo(tuning)
        self._last_camera_token: Optional[CameraFrameToken] = None
        self._last_tracker_frame_sequence: Optional[int] = None
        self._latched_next_track_id: Optional[str] = None

    @property
    def latched_next_track_id(self) -> Optional[str]:
        return self._latched_next_track_id

    def reset_segment(self) -> None:
        """Clear publication and next-identity state for an explicit new segment."""

        self._servo.reset_segment()
        self._last_camera_token = None
        self._last_tracker_frame_sequence = None
        self._latched_next_track_id = None

    def observe(
        self,
        snapshot: GateGraphSnapshot,
        tracker: MultiTargetVisualTracker,
        now_monotonic_s: float,
        segment_elapsed_s: float,
        segment_yaw_excursion_rad: float,
    ) -> VisualApproachProposal:
        """Produce one current-only or current/next no-advance servo proposal.

        ``now_monotonic_s`` must be QueryPerformanceCounter seconds, matching
        the receiver's ``host-perf-counter`` final-packet observations.
        """

        if type(snapshot) is not GateGraphSnapshot:
            raise TypeError("snapshot must be an exact GateGraphSnapshot")
        if type(tracker) is not MultiTargetVisualTracker:
            raise TypeError("tracker must be an exact MultiTargetVisualTracker")
        for name, value in (
            ("now_monotonic_s", now_monotonic_s),
            ("segment_elapsed_s", segment_elapsed_s),
            ("segment_yaw_excursion_rad", segment_yaw_excursion_rad),
        ):
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise VisualApproachRefusal(f"{name} must be finite")

        update = tracker.latest_update
        if update is None:
            raise VisualApproachRefusal("tracker has no exact camera update")
        self._validate_snapshot(snapshot, tracker)
        self._validate_publication_advance(snapshot)

        current = tracker.track(self.expected_current_track_id)
        self._validate_current(snapshot, update, current)
        current_target = self._target(
            current,
            now_monotonic_s=float(now_monotonic_s),
            require_current_authority=True,
        )

        candidate_ids = tuple(
            candidate.track_id for candidate in snapshot.next_candidates
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise VisualApproachRefusal(
                "gate graph repeated a next-candidate identity"
            )
        provisional_ids = tuple(snapshot.provisional_track_ids)
        if len(provisional_ids) != len(set(provisional_ids)):
            raise VisualApproachRefusal(
                "gate graph repeated a provisional identity"
            )

        visible_ambiguous = tuple(
            track_id
            for track_id in update.ambiguous_track_ids
            if track_id != self.expected_current_track_id
            and tracker.track(track_id).visible
        )
        if snapshot.next_selection_ambiguous or visible_ambiguous:
            raise VisualApproachRefusal(
                "next-gate visual identity is ambiguous"
            )

        visible_stable = self._visible_stable_candidates(
            snapshot,
            tracker,
        )
        eligible = tuple(
            (candidate, track)
            for candidate, track in visible_stable
            if self._candidate_is_blend_eligible(
                candidate,
                track,
                snapshot,
            )
        )
        if provisional_ids:
            # A new one-frame contour is not yet evidence that the stable
            # incumbent identity changed.  It is nevertheless unresolved
            # geometry, so remove all next-gate blend authority for this exact
            # publication.  If it matures into a competing stable candidate,
            # the checks below refuse instead of silently keeping the latch.
            eligible = ()
        if len(visible_stable) > 1 or len(eligible) > 1:
            raise VisualApproachRefusal(
                "competing stable next-gate identities are present"
            )

        next_target: Optional[VisualTarget] = None
        relationship_basis: Optional[GateRelationshipBasis] = None
        requested_blend = 0.0
        withholding_reason: Optional[str]
        eligible_id: Optional[str] = None
        if eligible:
            candidate, next_track = eligible[0]
            eligible_id = candidate.track_id
            if (
                self._latched_next_track_id is not None
                and eligible_id != self._latched_next_track_id
            ):
                raise VisualApproachRefusal(
                    "next-gate identity changed after blend latch"
                )
            next_target = self._target(
                next_track,
                now_monotonic_s=float(now_monotonic_s),
                require_current_authority=False,
            )
            if next_target.frame_token != current_target.frame_token:
                raise VisualApproachRefusal(
                    "current and next targets do not share one exact publication"
                )
            assert candidate.relationship is not None
            relationship_basis = candidate.relationship.basis
            requested_blend = MAX_NEXT_GATE_BLEND
            withholding_reason = None
        else:
            different_visible = tuple(
                candidate.track_id
                for candidate, track in visible_stable
                if (
                    self._latched_next_track_id is not None
                    and candidate.track_id != self._latched_next_track_id
                    and track.visible
                )
            )
            if different_visible:
                raise VisualApproachRefusal(
                    "a different stable next-gate identity replaced the latch"
                )
            if provisional_ids:
                withholding_reason = "provisional_next_identity_unresolved"
            elif self._latched_next_track_id is not None:
                withholding_reason = "latched_next_track_unavailable"
            elif candidate_ids:
                withholding_reason = "no_visible_promotable_next_candidate"
            else:
                withholding_reason = "no_next_candidate"

        try:
            output = self._servo.step(
                current_target,
                now_monotonic_s=float(now_monotonic_s),
                segment_elapsed_s=float(segment_elapsed_s),
                segment_yaw_excursion_rad=float(
                    segment_yaw_excursion_rad
                ),
                next_target=next_target,
                requested_next_blend=requested_blend,
                allow_advance=False,
            )
        except VisualServoRefusal as exc:
            raise VisualApproachRefusal(
                f"image visual servo refused approach authority: {exc}"
            ) from exc
        if output.advance_enabled:
            raise VisualApproachRefusal(
                "visual approach escaped its no-advance envelope"
            )
        if output.next_gate_blend not in {0.0, MAX_NEXT_GATE_BLEND}:
            raise VisualApproachRefusal(
                "visual approach produced an unexpected blend magnitude"
            )
        if output.next_gate_blend > 0.0:
            if eligible_id is None or next_target is None:
                raise VisualApproachRefusal(
                    "visual approach blended without an exact next target"
                )
            if self._latched_next_track_id is None:
                self._latched_next_track_id = eligible_id
            elif self._latched_next_track_id != eligible_id:
                raise VisualApproachRefusal(
                    "visual approach changed its latched next identity"
                )
            withholding_reason = None
        elif next_target is not None:
            withholding_reason = "current_corridor_not_ready"

        self._last_camera_token = snapshot.latest_camera_token
        self._last_tracker_frame_sequence = snapshot.tracker_frame_sequence
        return VisualApproachProposal(
            current_target=current_target,
            next_target=next_target,
            servo_output=output,
            candidate_track_ids=candidate_ids,
            provisional_track_ids=provisional_ids,
            withholding_reason=withholding_reason,
            relationship_basis=relationship_basis,
            latched_next_track_id=self._latched_next_track_id,
        )

    def _validate_snapshot(
        self,
        snapshot: GateGraphSnapshot,
        tracker: MultiTargetVisualTracker,
    ) -> None:
        update = tracker.latest_update
        assert update is not None
        if (
            tracker.time_basis_id != _QPC_TIME_BASIS_ID
            or update.provenance_basis
            is not FrameProvenanceBasis.RECEIVER_TIMING_V1
        ):
            raise VisualApproachRefusal(
                "visual approach requires exact QPC receiver provenance"
            )
        if (
            snapshot.latest_camera_token != update.token
            or snapshot.tracker_frame_sequence
            != update.tracker_frame_sequence
        ):
            raise VisualApproachRefusal(
                "gate graph and tracker do not share one exact publication"
            )
        if (
            snapshot.current_track_id != self.expected_current_track_id
            or snapshot.current_gate_index != self.expected_gate_index
            or snapshot.current_track is None
            or snapshot.race_finished
            or not snapshot.authority_usable
            or snapshot.withholding_reason is not None
        ):
            raise VisualApproachRefusal(
                "gate graph withheld authoritative current-gate identity"
            )
        race = snapshot.latest_race_status
        if (
            race is None
            or race.provenance_basis
            is not RaceStatusProvenanceBasis.LIVE_INGRESS
            or race.race_finished
            or race.active_gate_index != self.expected_gate_index
            or race.host_clock_id != _QPC_TIME_BASIS_ID
        ):
            raise VisualApproachRefusal(
                "current gate lacks matching live race/QPC authority"
            )

    def _validate_publication_advance(
        self,
        snapshot: GateGraphSnapshot,
    ) -> None:
        token = snapshot.latest_camera_token
        if (
            token.stream_id is None
            or token.publication_sequence is None
        ):
            raise VisualApproachRefusal(
                "visual approach requires exact publication provenance"
            )
        previous = self._last_camera_token
        previous_sequence = self._last_tracker_frame_sequence
        if previous is None:
            return
        if (
            token.stream_id != previous.stream_id
            or token.generation != previous.generation
            or token.publication_sequence
            <= previous.publication_sequence
            or previous_sequence is None
            or snapshot.tracker_frame_sequence <= previous_sequence
        ):
            raise VisualApproachRefusal(
                "visual approach publication order did not strictly advance"
            )

    def _validate_current(
        self,
        snapshot: GateGraphSnapshot,
        update: VisualTrackerUpdate,
        current: VisualTrack,
    ) -> None:
        update_track = update.track(self.expected_current_track_id)
        if (
            snapshot.current_track != current
            or update_track != current
            or current.track_id != self.expected_current_track_id
            or current.latest_token != snapshot.latest_camera_token
            or current.role is not VisualTrackRole.CURRENT
            or current.authoritative_gate_index != self.expected_gate_index
            or not current.visible
            or current.missed_frame_count != 0
            or current.ambiguous
            or current.consecutive_frame_count < 1
        ):
            raise VisualApproachRefusal(
                "authoritative current identity is not exact and visible"
            )
        if (
            current.clipping != FrameEdge.NONE
            or current.center_censored
        ):
            raise VisualApproachCurrentGeometryUnavailable(
                "authoritative current aperture is clipped or censored"
            )

    def _target(
        self,
        track: VisualTrack,
        *,
        now_monotonic_s: float,
        require_current_authority: bool,
    ) -> VisualTarget:
        if not track.history:
            raise VisualApproachRefusal("visual track lacks sample history")
        sample = track.history[-1]
        if (
            sample.provenance_basis
            is not FrameProvenanceBasis.RECEIVER_TIMING_V1
            or sample.token != track.latest_token
            or sample.publication_monotonic_ns is None
            or sample.publication_monotonic_ns
            < sample.observation_monotonic_ns
        ):
            raise VisualApproachRefusal(
                "visual track lacks coherent QPC publication provenance"
            )
        age_s = (
            float(now_monotonic_s)
            - sample.observation_monotonic_ns / 1_000_000_000.0
        )
        if (
            age_s < -_FUTURE_TOLERANCE_S
            or age_s > MAX_VISUAL_OBSERVATION_AGE_S
        ):
            raise VisualApproachRefusal(
                "visual track is stale or future-dated in QPC time"
            )
        try:
            return VisualTarget.from_visual_track(
                track,
                require_current_authority=require_current_authority,
                expected_gate_index=(
                    self.expected_gate_index
                    if require_current_authority
                    else None
                ),
            )
        except VisualServoRefusal as exc:
            raise VisualApproachRefusal(
                f"visual target adaptation refused: {exc}"
            ) from exc

    @staticmethod
    def _visible_stable_candidates(
        snapshot: GateGraphSnapshot,
        tracker: MultiTargetVisualTracker,
    ) -> tuple[tuple[NextGateCandidate, VisualTrack], ...]:
        result: list[tuple[NextGateCandidate, VisualTrack]] = []
        for candidate in snapshot.next_candidates:
            try:
                track = tracker.track(candidate.track_id)
            except KeyError as exc:
                raise VisualApproachRefusal(
                    "gate graph candidate is absent from tracker"
                ) from exc
            if (
                track.visible
                and candidate.stable_frame_count >= _REQUIRED_NEXT_FRAMES
            ):
                result.append((candidate, track))
        return tuple(result)

    @staticmethod
    def _candidate_is_blend_eligible(
        candidate: NextGateCandidate,
        track: VisualTrack,
        snapshot: GateGraphSnapshot,
    ) -> bool:
        relation = candidate.relationship
        return bool(
            candidate.promotable
            and candidate.stable_frame_count >= _REQUIRED_NEXT_FRAMES
            and candidate.latest_token == snapshot.latest_camera_token
            and track.role is VisualTrackRole.NEXT
            and track.authoritative_gate_index is None
            and track.visible
            and track.missed_frame_count == 0
            and not track.ambiguous
            and track.consecutive_frame_count >= _REQUIRED_NEXT_FRAMES
            and track.latest_token == snapshot.latest_camera_token
            and relation is not None
            and relation in snapshot.relationships
            and relation.current_track_id == snapshot.current_track_id
            and relation.next_track_id == candidate.track_id
            and relation.latest_token == snapshot.latest_camera_token
            and relation.latest_tracker_frame_sequence
            == snapshot.tracker_frame_sequence
            and relation.fresh
            and not relation.contended
        )


__all__ = [
    "RollingVisualApproachServo",
    "VisualApproachCurrentGeometryUnavailable",
    "VisualApproachProposal",
    "VisualApproachRefusal",
]
