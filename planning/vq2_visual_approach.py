"""Pure visual approach/passage adapter for the rolling VQ2 gate graph.

The adapter is deliberately narrower than the live runner.  It binds one
authoritative current identity for one segment, admits at most one exact
same-publication next-gate target, and delegates image-space control to
``ImageVisualServo``.  Approach mode cannot advance.  Passage mode is a
one-way, bounded segment transition that requires an exact admission from a
prior accepted approach publication.  It may retain only that admission's
latched next identity under the immutable passage-preview envelope while the
current aperture independently owns advance authority.  It owns no transport,
race transition, reset, watchdog, collision, or cleanup authority.  In
particular, its yaw proposal is not yaw calibration or transport authority;
build-3385 integration must retain the separate calibrated-yaw safety boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
    DEFAULT_ROLLING_GATE_GRAPH_CONFIG,
    GateGraphSnapshot,
    GateRelationshipBasis,
    NextGateCandidate,
    RaceStatusProvenanceBasis,
)
from planning.vq2_visual_servo import (
    ImageVisualServo,
    MAX_NEXT_GATE_BLEND,
    MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM,
    MAX_VISUAL_OBSERVATION_AGE_S,
    PREPASS_CURRENT_MAX_ABS_Y_NORM,
    VisualServoOutput,
    VisualServoPassageSafetyUnavailable,
    VisualServoRefusal,
    VisualServoTuning,
    VisualTarget,
)


_QPC_TIME_BASIS_ID = "host-perf-counter"
_FUTURE_TOLERANCE_S = 1e-6
_REQUIRED_NEXT_FRAMES = 3
# A passage-corridor refusal removes authority for its exact publication, but
# one short predictive excursion must not destroy a proved same-identity latch.
# The latest exact build-3385 handoff left only the projected vertical bound
# for publications 117 and 118, then re-entered every unchanged bound at 119.
# A third consecutive refused fresh publication retires the lease.
MAX_PASSAGE_SUSPENSION_FRESH_FRAMES = 2
# The same replay contains three brief predictive-excursion epochs totaling
# four refused fresh publications before the close-range contour merger.
# These whole-segment limits prevent alternating safe/unsafe frames from
# manufacturing an unbounded lease.
MAX_PASSAGE_SUSPENSION_TOTAL_FRESH_FRAMES = 4
MAX_PASSAGE_SUSPENSION_EPOCHS = 3
MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S = 0.12
MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S = 0.20
VISUAL_PASSAGE_ADMISSION_BASIS = "tight-current-corridor-dwell-v1"


class VisualApproachRefusal(ValueError):
    """The graph/tracker pair cannot safely authorize an approach proposal."""


class VisualApproachAdjacentUnavailable(VisualApproachRefusal):
    """One publication lacks optional, graph-vetted adjacent authority."""


class VisualApproachCurrentGeometryUnavailable(VisualApproachRefusal):
    """The optional visual blend must withdraw from a clipped current gate."""


class VisualApproachPassageSafetyUnavailable(VisualApproachRefusal):
    """One exact publication cannot safely continue a latched blend."""

    def __init__(
        self,
        message: str,
        *,
        violation_codes: tuple[str, ...],
        violation_evidence: tuple[tuple[str, float, float, float], ...],
        camera_observation_monotonic_s: float,
        latched_next_track_id: Optional[str] = None,
    ) -> None:
        if (
            type(violation_codes) is not tuple
            or not violation_codes
            or any(
                type(code) is not str or not code
                for code in violation_codes
            )
            or len(violation_codes) != len(set(violation_codes))
            or type(violation_evidence) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 4
                or type(item[0]) is not str
                or any(
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                    for value in item[1:]
                )
                for item in violation_evidence
            )
            or tuple(item[0] for item in violation_evidence)
            != violation_codes
            or type(camera_observation_monotonic_s) not in {int, float}
            or not math.isfinite(float(camera_observation_monotonic_s))
            or float(camera_observation_monotonic_s) < 0.0
            or (
                latched_next_track_id is not None
                and (
                    type(latched_next_track_id) is not str
                    or not latched_next_track_id
                    or len(latched_next_track_id) > 128
                )
            )
        ):
            raise ValueError(
                "passage refusal requires structured immutable evidence"
            )
        self.violation_codes = violation_codes
        self.violation_evidence = violation_evidence
        self.camera_observation_monotonic_s = float(
            camera_observation_monotonic_s
        )
        self.latched_next_track_id = latched_next_track_id
        transient_evidence = violation_evidence[0]
        self.transient_eligible = bool(
            violation_codes == ("current_projected_vertical",)
            and transient_evidence[1]
            < -PREPASS_CURRENT_MAX_ABS_Y_NORM
            and transient_evidence[2]
            == -PREPASS_CURRENT_MAX_ABS_Y_NORM
            and math.isclose(
                transient_evidence[2] - transient_evidence[1],
                transient_evidence[3],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and 0.0 < float(transient_evidence[3])
            <= MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM
        )
        super().__init__(message)

    @classmethod
    def from_servo_refusal(
        cls,
        refusal: VisualServoPassageSafetyUnavailable,
        *,
        camera_observation_monotonic_s: float,
        latched_next_track_id: Optional[str] = None,
    ) -> "VisualApproachPassageSafetyUnavailable":
        if type(refusal) is not VisualServoPassageSafetyUnavailable:
            raise TypeError(
                "passage wrapper requires an exact visual-servo refusal"
            )
        return cls(
            f"image visual servo retired passage authority: {refusal}",
            violation_codes=tuple(
                violation.value for violation in refusal.violations
            ),
            violation_evidence=tuple(
                (
                    detail.violation.value,
                    detail.observed,
                    detail.limit,
                    detail.excess,
                )
                for detail in refusal.details
            ),
            camera_observation_monotonic_s=(
                camera_observation_monotonic_s
            ),
            latched_next_track_id=latched_next_track_id,
        )


@dataclass(frozen=True, slots=True)
class VisualApproachPassageLeaseState:
    """Exact-token state for a short, zero-authority passage suspension."""

    camera_token: CameraFrameToken
    passage_safe: bool
    blend_active: bool
    recovered: bool
    resumed: bool
    suspension_streak: int
    total_suspended_fresh_frames: int
    suspension_epoch_count: int
    suspension_epoch_duration_s: float
    total_suspension_duration_s: float
    resume_count: int
    retirement_required: bool
    retirement_reason: Optional[str]


class VisualApproachPassageLease:
    """Bound retries after a fresh-frame passage refusal.

    This lease never creates a command.  The runner clears its visual proposal
    on every refused publication and may resume only after the ordinary
    approach path accepts a newer exact token with the same latched identity.
    """

    def __init__(self) -> None:
        self._last_token: Optional[CameraFrameToken] = None
        self._suspension_streak = 0
        self._total_suspended_fresh_frames = 0
        self._suspension_epoch_count = 0
        self._suspension_epoch_started_s: Optional[float] = None
        self._completed_suspension_duration_s = 0.0
        self._last_observation_monotonic_s: Optional[float] = None
        self._corridor_recovery_reported = False
        self._resume_count = 0
        self._retired = False
        self._retirement_reason: Optional[str] = None

    def observe(
        self,
        camera_token: CameraFrameToken,
        *,
        observation_monotonic_s: float,
        passage_safe: bool,
        blend_active: bool,
    ) -> VisualApproachPassageLeaseState:
        """Consume one newer exact camera token and update the bounded lease."""

        if type(camera_token) is not CameraFrameToken:
            raise VisualApproachRefusal(
                "passage suspension requires an exact camera token"
            )
        if camera_token.live_identity_tuple is None:
            raise VisualApproachRefusal(
                "passage suspension requires live publication provenance"
            )
        if (
            type(observation_monotonic_s) not in {int, float}
            or not math.isfinite(float(observation_monotonic_s))
            or float(observation_monotonic_s) < 0.0
        ):
            raise VisualApproachRefusal(
                "passage suspension observation time must be finite"
            )
        if type(passage_safe) is not bool or type(blend_active) is not bool:
            raise VisualApproachRefusal(
                "passage suspension flags must be exact booleans"
            )
        if blend_active and not passage_safe:
            raise VisualApproachRefusal(
                "an unsafe passage publication cannot retain blend authority"
            )
        if self._retired and blend_active:
            raise VisualApproachRefusal(
                "a retired passage suspension lease cannot reactivate blend "
                "authority"
            )
        previous = self._last_token
        if previous is not None and (
            camera_token.stream_id != previous.stream_id
            or camera_token.generation != previous.generation
            or camera_token.publication_sequence
            <= previous.publication_sequence
        ):
            raise VisualApproachRefusal(
                "passage suspension camera token did not strictly advance"
            )
        observation_s = float(observation_monotonic_s)
        if (
            self._last_observation_monotonic_s is not None
            and observation_s <= self._last_observation_monotonic_s
        ):
            raise VisualApproachRefusal(
                "passage suspension observation time did not strictly advance"
            )
        self._last_token = camera_token
        self._last_observation_monotonic_s = observation_s

        recovered = False
        resumed = False
        epoch_duration_s = (
            0.0
            if self._suspension_epoch_started_s is None
            else observation_s - self._suspension_epoch_started_s
        )
        total_duration_s = (
            self._completed_suspension_duration_s + epoch_duration_s
        )
        if self._retired:
            retirement_required = True
            retirement_reason = self._retirement_reason
        elif passage_safe:
            pending_suspension = self._suspension_streak > 0
            recovered = bool(
                pending_suspension
                and (
                    blend_active
                    or not self._corridor_recovery_reported
                )
            )
            if (
                pending_suspension
                and epoch_duration_s
                > MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S
            ):
                retirement_reason = "suspension_epoch_duration_exhausted"
            elif (
                pending_suspension
                and total_duration_s
                > MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S
            ):
                retirement_reason = "total_suspension_duration_exhausted"
            else:
                retirement_reason = None
            retirement_required = retirement_reason is not None
            if retirement_required:
                self._retired = True
                self._retirement_reason = retirement_reason
            elif pending_suspension and blend_active:
                resumed = True
                self._resume_count += 1
                self._completed_suspension_duration_s = total_duration_s
                self._suspension_streak = 0
                self._suspension_epoch_started_s = None
                self._corridor_recovery_reported = False
            elif pending_suspension:
                self._corridor_recovery_reported = True
            else:
                self._suspension_streak = 0
                self._suspension_epoch_started_s = None
                self._corridor_recovery_reported = False
        else:
            if self._suspension_streak == 0:
                self._suspension_epoch_count += 1
                self._suspension_epoch_started_s = observation_s
                self._corridor_recovery_reported = False
                epoch_duration_s = 0.0
                total_duration_s = (
                    self._completed_suspension_duration_s
                )
            self._suspension_streak += 1
            self._total_suspended_fresh_frames += 1
            retirement_reason = None
            if (
                self._suspension_streak
                > MAX_PASSAGE_SUSPENSION_FRESH_FRAMES
            ):
                retirement_reason = "consecutive_fresh_frames_exhausted"
            elif (
                self._total_suspended_fresh_frames
                > MAX_PASSAGE_SUSPENSION_TOTAL_FRESH_FRAMES
            ):
                retirement_reason = "total_fresh_frames_exhausted"
            elif (
                self._suspension_epoch_count
                > MAX_PASSAGE_SUSPENSION_EPOCHS
            ):
                retirement_reason = "suspension_epochs_exhausted"
            elif (
                epoch_duration_s
                > MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S
            ):
                retirement_reason = "suspension_epoch_duration_exhausted"
            elif (
                total_duration_s
                > MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S
            ):
                retirement_reason = "total_suspension_duration_exhausted"
            retirement_required = retirement_reason is not None
            if retirement_required:
                self._retired = True
                self._retirement_reason = retirement_reason

        return VisualApproachPassageLeaseState(
            camera_token=camera_token,
            passage_safe=passage_safe,
            blend_active=blend_active,
            recovered=recovered,
            resumed=resumed,
            suspension_streak=self._suspension_streak,
            total_suspended_fresh_frames=(
                self._total_suspended_fresh_frames
            ),
            suspension_epoch_count=self._suspension_epoch_count,
            suspension_epoch_duration_s=epoch_duration_s,
            total_suspension_duration_s=total_duration_s,
            resume_count=self._resume_count,
            retirement_required=retirement_required,
            retirement_reason=retirement_reason,
        )


class VisualApproachMode(Enum):
    """Mutually exclusive command-authority modes for one visual segment."""

    APPROACH = "approach"
    PASSAGE = "passage"
    ADJACENT_RECENTER = "adjacent_recenter"
    PROMOTE_REACQUIRE = "promote_reacquire"


@dataclass(frozen=True, slots=True)
class VisualApproachPassageAdmission:
    """Exact prior-publication evidence admitting bounded passage mode."""

    basis: str
    current_gate_index: int
    current_target: VisualTarget
    camera_token: CameraFrameToken
    tracker_frame_sequence: int
    corridor_frames: int
    preview_track_id: Optional[str]
    preview_blend: float


@dataclass(frozen=True, slots=True)
class VisualApproachProposal:
    """One exact-publication visual approach or bounded passage proposal."""

    current_target: VisualTarget
    next_target: Optional[VisualTarget]
    servo_output: VisualServoOutput
    candidate_track_ids: tuple[str, ...]
    provisional_track_ids: tuple[str, ...]
    withholding_reason: Optional[str]
    relationship_basis: Optional[GateRelationshipBasis]
    latched_next_track_id: Optional[str]
    mode: VisualApproachMode = VisualApproachMode.APPROACH
    passage_admission: Optional[VisualApproachPassageAdmission] = None


class RollingVisualApproachServo:
    """Keep current identity fixed through approach and bounded passage."""

    def __init__(
        self,
        expected_current_track_id: str,
        expected_gate_index: int,
        tuning: Optional[VisualServoTuning] = None,
        *,
        next_gate_blend: float,
        next_gate_blend_start_log_scale: Optional[float] = None,
        next_gate_blend_full_log_scale: Optional[float] = None,
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
        if (
            type(next_gate_blend) not in {int, float}
            or not math.isfinite(float(next_gate_blend))
            or not (
                0.0
                <= float(next_gate_blend)
                <= MAX_NEXT_GATE_BLEND
            )
        ):
            raise VisualApproachRefusal(
                "next_gate_blend must stay inside its immutable ceiling"
            )
        ramp_values = (
            next_gate_blend_start_log_scale,
            next_gate_blend_full_log_scale,
        )
        if (ramp_values[0] is None) != (ramp_values[1] is None):
            raise VisualApproachRefusal(
                "next_gate_blend scale ramp must provide both boundaries"
            )
        if ramp_values[0] is not None:
            if (
                any(
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                    for value in ramp_values
                )
                or not -3.0 <= float(ramp_values[0]) <= -1.0
                or not -1.0 <= float(ramp_values[1]) <= -0.20
                or float(ramp_values[0]) >= float(ramp_values[1])
            ):
                raise VisualApproachRefusal(
                    "next_gate_blend scale ramp is outside its fixed bounds"
                )
        self.expected_current_track_id = expected_current_track_id
        self.expected_gate_index = expected_gate_index
        self.next_gate_blend = float(next_gate_blend)
        self.next_gate_blend_start_log_scale = (
            None
            if next_gate_blend_start_log_scale is None
            else float(next_gate_blend_start_log_scale)
        )
        self.next_gate_blend_full_log_scale = (
            None
            if next_gate_blend_full_log_scale is None
            else float(next_gate_blend_full_log_scale)
        )
        self._servo = ImageVisualServo(tuning)
        self._last_camera_token: Optional[CameraFrameToken] = None
        self._last_tracker_frame_sequence: Optional[int] = None
        self._latched_next_track_id: Optional[str] = None
        self._pending_passage_admission: Optional[
            VisualApproachPassageAdmission
        ] = None
        self._active_passage_admission: Optional[
            VisualApproachPassageAdmission
        ] = None

    @property
    def latched_next_track_id(self) -> Optional[str]:
        return self._latched_next_track_id

    def reset_segment(self) -> None:
        """Clear publication and next-identity state for an explicit new segment."""

        self._servo.reset_segment()
        self._last_camera_token = None
        self._last_tracker_frame_sequence = None
        self._latched_next_track_id = None
        self._pending_passage_admission = None
        self._active_passage_admission = None

    def retire_passage_preview(self, expected_track_id: str) -> None:
        """Permanently withdraw an active passage's optional preview."""

        admission = self._active_passage_admission
        if (
            type(expected_track_id) is not str
            or not expected_track_id
            or type(admission) is not VisualApproachPassageAdmission
            or admission.preview_track_id != expected_track_id
            or self._latched_next_track_id != expected_track_id
        ):
            raise VisualApproachRefusal(
                "passage preview retirement identity is inconsistent"
            )
        try:
            self._servo.retire_advance_passage_preview()
        except VisualServoRefusal as exc:
            raise VisualApproachRefusal(
                f"visual servo refused passage preview retirement: {exc}"
            ) from exc

    def _requested_next_gate_blend(
        self,
        current_log_scale: float,
    ) -> float:
        """Ramp preview authority with the current aperture's apparent scale."""

        if (
            self.next_gate_blend_start_log_scale is None
            or self.next_gate_blend_full_log_scale is None
        ):
            return self.next_gate_blend
        if (
            type(current_log_scale) not in {int, float}
            or not math.isfinite(float(current_log_scale))
        ):
            raise VisualApproachRefusal(
                "current gate log scale is invalid for next-gate blending"
            )
        fraction = (
            float(current_log_scale)
            - self.next_gate_blend_start_log_scale
        ) / (
            self.next_gate_blend_full_log_scale
            - self.next_gate_blend_start_log_scale
        )
        return self.next_gate_blend * max(0.0, min(1.0, fraction))

    def observe_promotable_adjacent(
        self,
        snapshot: GateGraphSnapshot,
        tracker: MultiTargetVisualTracker,
        now_monotonic_s: float,
        segment_elapsed_s: float,
        segment_yaw_excursion_rad: float,
    ) -> VisualApproachProposal:
        """Recenter on one graph-vetted successor without passage authority.

        Promotion may require a simultaneous-image relationship that is
        unobservable across the gate plane.  This path instead requires the
        graph's sole clean, stable NEXT role and never advances or promotes
        it; authoritative race credit remains the only passage authority.
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
            if type(value) not in {int, float} or not math.isfinite(
                float(value)
            ):
                raise VisualApproachRefusal(f"{name} must be finite")

        update = tracker.latest_update
        race = snapshot.latest_race_status
        if update is None:
            raise VisualApproachAdjacentUnavailable(
                "credit-wait adjacent tracker publication is unavailable"
            )
        if (
            tracker.time_basis_id != _QPC_TIME_BASIS_ID
            or update.provenance_basis
            is not FrameProvenanceBasis.RECEIVER_TIMING_V1
            or race is None
            or race.provenance_basis
            is not RaceStatusProvenanceBasis.LIVE_INGRESS
            or race.host_clock_id != _QPC_TIME_BASIS_ID
        ):
            raise VisualApproachRefusal(
                "credit-wait adjacent provenance authority is invalid"
            )
        graph_config = DEFAULT_ROLLING_GATE_GRAPH_CONFIG
        candidates = tuple(
            candidate
            for candidate in snapshot.next_candidates
            if (
                candidate.promotable
                or (
                    candidate.relationship is None
                    and candidate.stable_frame_count
                    >= graph_config.min_next_candidate_frames
                    and candidate.confidence
                    >= graph_config.min_track_confidence
                    and candidate.association_confidence
                    >= graph_config.min_association_confidence
                )
            )
        )
        if (
            snapshot.latest_camera_token != update.token
            or snapshot.tracker_frame_sequence
            != update.tracker_frame_sequence
            or snapshot.race_finished
            or snapshot.current_gate_index
            != self.expected_gate_index - 1
            or snapshot.current_track_id is None
            or snapshot.current_track_id
            == self.expected_current_track_id
            or snapshot.next_selection_ambiguous
            or snapshot.provisional_track_ids
            or len(candidates) != 1
            or candidates[0].track_id
            != self.expected_current_track_id
            or race.race_finished
            or race.active_gate_index
            != self.expected_gate_index - 1
        ):
            raise VisualApproachAdjacentUnavailable(
                "credit-wait adjacent authority is unavailable"
            )

        candidate = candidates[0]
        try:
            track = tracker.track(candidate.track_id)
        except KeyError as exc:
            raise VisualApproachAdjacentUnavailable(
                "credit-wait adjacent candidate is absent from tracker"
            ) from exc
        if (
            candidate.stable_frame_count < _REQUIRED_NEXT_FRAMES
            or candidate.latest_token != snapshot.latest_camera_token
            or track.role is not VisualTrackRole.NEXT
            or track.authoritative_gate_index is not None
            or not track.visible
            or track.missed_frame_count != 0
            or track.ambiguous
            or track.consecutive_frame_count < _REQUIRED_NEXT_FRAMES
            or track.latest_token != snapshot.latest_camera_token
            or track.clipping is not FrameEdge.NONE
            or track.center_censored
        ):
            raise VisualApproachAdjacentUnavailable(
                "credit-wait adjacent candidate is not clean and stable"
            )

        self._validate_publication_advance(snapshot)
        target = self._target(
            track,
            now_monotonic_s=float(now_monotonic_s),
            require_current_authority=False,
        )
        try:
            output = self._servo.step(
                target,
                now_monotonic_s=float(now_monotonic_s),
                segment_elapsed_s=float(segment_elapsed_s),
                segment_yaw_excursion_rad=float(
                    segment_yaw_excursion_rad
                ),
                requested_next_blend=0.0,
                allow_advance=False,
                allow_passage_safe_next_blend=False,
            )
        except VisualServoRefusal as exc:
            raise VisualApproachRefusal(
                "credit-wait adjacent servo refused authority: "
                f"{exc}"
            ) from exc
        if (
            output.advance_enabled
            or output.next_gate_blend != 0.0
            or output.reviewed_next_track_id is not None
        ):
            raise VisualApproachRefusal(
                "credit-wait adjacent proposal escaped no-advance authority"
            )

        self._last_camera_token = snapshot.latest_camera_token
        self._last_tracker_frame_sequence = (
            snapshot.tracker_frame_sequence
        )
        return VisualApproachProposal(
            current_target=target,
            next_target=None,
            servo_output=output,
            candidate_track_ids=(candidate.track_id,),
            provisional_track_ids=(),
            withholding_reason=None,
            relationship_basis=(
                None
                if candidate.relationship is None
                else candidate.relationship.basis
            ),
            latched_next_track_id=None,
            mode=VisualApproachMode.ADJACENT_RECENTER,
            passage_admission=None,
        )

    def observe(
        self,
        snapshot: GateGraphSnapshot,
        tracker: MultiTargetVisualTracker,
        now_monotonic_s: float,
        segment_elapsed_s: float,
        segment_yaw_excursion_rad: float,
        *,
        mode: VisualApproachMode = VisualApproachMode.APPROACH,
        passage_admission: Optional[
            VisualApproachPassageAdmission
        ] = None,
        passage_forward_closure_authorized: bool = True,
    ) -> VisualApproachProposal:
        """Produce one current-only or current/next visual-servo proposal.

        ``now_monotonic_s`` must be QueryPerformanceCounter seconds, matching
        the receiver's ``host-perf-counter`` final-packet observations.

        ``APPROACH`` preserves the historical no-advance semantics and may
        cautiously blend an exact next target.  ``PROMOTE_REACQUIRE`` is
        current-only, never advances, and may consume exactly one censored
        frame edge after the course coordinator has established its bounded
        post-credit authority.  ``PASSAGE`` requires the module-issued
        admission from the latest safe approach dwell and may retain only its
        exact latched preview identity while current-gate advance remains
        independently safety-gated.  It cannot transition back to approach
        without an explicit segment reset.  Forward-closure authorization may
        inhibit advance without erasing that sealed passage lifecycle.
        """

        if type(snapshot) is not GateGraphSnapshot:
            raise TypeError("snapshot must be an exact GateGraphSnapshot")
        if type(tracker) is not MultiTargetVisualTracker:
            raise TypeError("tracker must be an exact MultiTargetVisualTracker")
        if type(passage_forward_closure_authorized) is not bool:
            raise TypeError(
                "passage_forward_closure_authorized must be an exact bool"
            )
        self._validate_mode_request(mode, passage_admission)
        starting_passage = bool(
            mode is VisualApproachMode.PASSAGE
            and self._active_passage_admission is None
        )
        if mode in {
            VisualApproachMode.APPROACH,
            VisualApproachMode.PROMOTE_REACQUIRE,
        }:
            # A newer attempted approach publication invalidates an older
            # corridor admission even when a later check refuses that frame.
            self._pending_passage_admission = None
        elif starting_passage:
            # The exact admission is single-use at passage entry.  Only the
            # explicitly restored near-plane censorship case below may retry.
            self._pending_passage_admission = None
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
        try:
            self._validate_current(snapshot, update, current, mode=mode)
        except VisualApproachCurrentGeometryUnavailable:
            if starting_passage:
                # The coordinator independently classifies and bounds expected
                # near-plane censorship.  Restore only the already-validated
                # exact admission so a repeated censored coast or later clean
                # publication resumes the same sealed successor identity.
                assert passage_admission is not None
                self._pending_passage_admission = passage_admission
            raise
        current_target = self._target(
            current,
            now_monotonic_s=float(now_monotonic_s),
            require_current_authority=True,
        )

        if mode is VisualApproachMode.PROMOTE_REACQUIRE:
            candidate_ids = ()
            provisional_ids = ()
            next_identity_ambiguous = False
            visible_stable = ()
            eligible = ()
        else:
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
            next_identity_ambiguous = bool(
                snapshot.next_selection_ambiguous or visible_ambiguous
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
        if provisional_ids or next_identity_ambiguous:
            # A new one-frame contour is not yet evidence that the stable
            # incumbent identity changed, and ambiguous geometry is not a
            # review of any exact identity.  Both remove only optional
            # next-gate authority for this publication.  The independently
            # valid current-aperture proposal remains available.
            eligible = ()
        competing_next_identities = bool(
            len(visible_stable) > 1 or len(eligible) > 1
        )
        if next_identity_ambiguous or competing_next_identities:
            # Next-only uncertainty cannot veto current-gate navigation in
            # either mode.  Passage still consumes only its sealed reviewed
            # identity below, while approach can earn a new admission only
            # after a later fresh, unambiguous review.
            eligible = ()
        if mode is VisualApproachMode.PASSAGE:
            assert passage_admission is not None
            sealed_preview_id = passage_admission.preview_track_id
            eligible = tuple(
                (candidate, track)
                for candidate, track in eligible
                if (
                    sealed_preview_id is not None
                    and candidate.track_id == sealed_preview_id
                    and candidate.track_id == self._latched_next_track_id
                )
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
                eligible = ()
                eligible_id = None
            if not eligible:
                withholding_reason = (
                    "passage_next_identity_withheld"
                    if mode is VisualApproachMode.PASSAGE
                    else "latched_next_identity_conflict"
                )
            else:
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
                requested_blend = (
                    self._requested_next_gate_blend(
                        current_target.log_scale
                    )
                )
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
            if provisional_ids:
                withholding_reason = "provisional_next_identity_unresolved"
            elif next_identity_ambiguous or competing_next_identities:
                withholding_reason = (
                    "passage_next_identity_withheld"
                    if mode is VisualApproachMode.PASSAGE
                    else "next_identity_unresolved"
                )
            elif different_visible:
                withholding_reason = (
                    "passage_next_identity_withheld"
                    if mode is VisualApproachMode.PASSAGE
                    else "latched_next_identity_conflict"
                )
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
                allow_advance=bool(
                    mode is VisualApproachMode.PASSAGE
                    and passage_forward_closure_authorized
                ),
                allow_passage_safe_next_blend=(
                    mode is VisualApproachMode.APPROACH
                    or (
                        passage_admission is not None
                        and passage_admission.preview_track_id is not None
                    )
                ),
            )
        except VisualServoPassageSafetyUnavailable as exc:
            raise VisualApproachPassageSafetyUnavailable.from_servo_refusal(
                exc,
                camera_observation_monotonic_s=(
                    current_target.received_monotonic_s
                ),
                latched_next_track_id=self._latched_next_track_id,
            ) from exc
        except VisualServoRefusal as exc:
            raise VisualApproachRefusal(
                f"image visual servo refused {mode.value} authority: {exc}"
            ) from exc
        if (
            mode is not VisualApproachMode.PASSAGE
            and output.advance_enabled
        ):
            raise VisualApproachRefusal(
                f"visual {mode.value} escaped its no-advance envelope"
            )
        if (
            mode is VisualApproachMode.APPROACH
            and not 0.0 <= output.next_gate_blend <= requested_blend
        ):
            raise VisualApproachRefusal(
                "visual approach blend exceeded its requested authority"
            )
        if (
            mode is VisualApproachMode.PASSAGE
            and not 0.0 <= output.next_gate_blend <= requested_blend
        ):
            raise VisualApproachRefusal(
                "passage next-preview blend exceeded its requested authority"
            )
        # Rapid expansion can continuously taper the reported translation
        # blend to zero while the servo retains the sealed successor's
        # corridor-bounded heading for yaw/bank and brakes forward closure.
        # The next-error fields preserve that exact same-frame observation;
        # they are not an independent grant of advance authority.
        reviewed_next_track_id = output.reviewed_next_track_id
        if reviewed_next_track_id is not None:
            if (
                eligible_id != reviewed_next_track_id
                or next_target is None
                or next_target.track_id != reviewed_next_track_id
            ):
                raise VisualApproachRefusal(
                    "visual servo reviewed an ineligible next-track identity"
                )
            if self._latched_next_track_id is None:
                self._latched_next_track_id = reviewed_next_track_id
            elif self._latched_next_track_id != reviewed_next_track_id:
                raise VisualApproachRefusal(
                    "visual servo changed its reviewed next-track identity"
                )
        servo_latched_next_track_id = self._servo.latched_next_track_id
        if (
            servo_latched_next_track_id
            != self._latched_next_track_id
        ):
            raise VisualApproachRefusal(
                "coordinator and visual servo next-track latches diverged"
            )
        if output.next_gate_blend > 0.0:
            if eligible_id is None or next_target is None:
                raise VisualApproachRefusal(
                    "visual approach blended without an exact next target"
                )
            if self._latched_next_track_id != eligible_id:
                raise VisualApproachRefusal(
                    "visual approach changed its latched next identity"
                )
            withholding_reason = None
        elif next_target is not None:
            withholding_reason = (
                "passage_next_preview_safety_withheld"
                if mode is VisualApproachMode.PASSAGE
                else "current_passage_corridor_not_ready"
            )

        proposal_admission: Optional[
            VisualApproachPassageAdmission
        ]
        if mode is VisualApproachMode.APPROACH:
            proposal_admission = self._passage_admission_from_approach(
                snapshot,
                current_target,
                next_target,
                output,
            )
            self._pending_passage_admission = proposal_admission
        elif mode is VisualApproachMode.PASSAGE:
            assert passage_admission is not None
            if starting_passage:
                self._active_passage_admission = passage_admission
            proposal_admission = self._active_passage_admission
        else:
            proposal_admission = None

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
            mode=mode,
            passage_admission=proposal_admission,
        )

    def _validate_mode_request(
        self,
        mode: VisualApproachMode,
        passage_admission: Optional[
            VisualApproachPassageAdmission
        ],
    ) -> None:
        if type(mode) is not VisualApproachMode:
            raise VisualApproachRefusal(
                "visual approach mode must be an exact VisualApproachMode"
            )
        if mode in {
            VisualApproachMode.APPROACH,
            VisualApproachMode.PROMOTE_REACQUIRE,
        }:
            if passage_admission is not None:
                raise VisualApproachRefusal(
                    f"{mode.value} mode cannot consume passage admission "
                    "evidence"
                )
            if self._active_passage_admission is not None:
                raise VisualApproachRefusal(
                    "an active passage segment cannot return to a "
                    "non-passage mode"
                )
            return
        if type(passage_admission) is not VisualApproachPassageAdmission:
            raise VisualApproachRefusal(
                "passage mode requires exact reviewed admission evidence"
            )
        expected = (
            self._active_passage_admission
            or self._pending_passage_admission
        )
        if expected is None or passage_admission != expected:
            raise VisualApproachRefusal(
                "passage admission does not match this segment's latest "
                "reviewed evidence"
            )
        preview_id = passage_admission.preview_track_id
        preview_blend = passage_admission.preview_blend
        if (
            type(preview_blend) not in {int, float}
            or not math.isfinite(float(preview_blend))
            or not 0.0 <= float(preview_blend) <= self.next_gate_blend
            or (
                preview_id is None
                and float(preview_blend) != 0.0
            )
            or (
                preview_id is not None
                and (
                    type(preview_id) is not str
                    or not preview_id
                    or len(preview_id) > 128
                    or preview_id != self._latched_next_track_id
                )
            )
        ):
            raise VisualApproachRefusal(
                "passage admission preview authority is inconsistent"
            )
        if self._active_passage_admission is None and (
            self._last_camera_token != passage_admission.camera_token
            or self._last_tracker_frame_sequence
            != passage_admission.tracker_frame_sequence
            or passage_admission.current_gate_index
            != self.expected_gate_index
            or passage_admission.current_target.track_id
            != self.expected_current_track_id
            or passage_admission.corridor_frames
            < self._servo.tuning.required_corridor_frames
            or passage_admission.basis
            != VISUAL_PASSAGE_ADMISSION_BASIS
        ):
            raise VisualApproachRefusal(
                "passage admission is stale or inconsistent with the segment"
            )

    def _passage_admission_from_approach(
        self,
        snapshot: GateGraphSnapshot,
        current_target: VisualTarget,
        next_target: Optional[VisualTarget],
        output: VisualServoOutput,
    ) -> Optional[VisualApproachPassageAdmission]:
        if (
            output.corridor_frames
            < self._servo.tuning.required_corridor_frames
            or output.brake_reason != "aligning"
            or output.yaw_envelope_limited
        ):
            return None
        reviewed_preview_id = output.reviewed_next_track_id
        if (
            self._latched_next_track_id is not None
            and reviewed_preview_id is None
        ):
            # Once an exact next identity is latched, a later frame with no
            # fresh same-identity review cannot overwrite the pending passage
            # evidence with an identity-less admission.  With no latch, the
            # caller may still consume current-only admission for a terminal
            # gate that has no adjacent visual target.
            return None
        if (
            reviewed_preview_id is not None
            and (
                next_target is None
                or next_target.track_id != reviewed_preview_id
                or self._latched_next_track_id != reviewed_preview_id
            )
        ):
            raise VisualApproachRefusal(
                "passage admission next-track review is inconsistent"
            )
        return VisualApproachPassageAdmission(
            basis=VISUAL_PASSAGE_ADMISSION_BASIS,
            current_gate_index=self.expected_gate_index,
            current_target=current_target,
            camera_token=snapshot.latest_camera_token,
            tracker_frame_sequence=snapshot.tracker_frame_sequence,
            corridor_frames=output.corridor_frames,
            preview_track_id=(
                reviewed_preview_id
            ),
            preview_blend=output.next_gate_blend,
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
        *,
        mode: VisualApproachMode,
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
        recovery_one_edge = bool(
            mode is VisualApproachMode.PROMOTE_REACQUIRE
            and current.clipping
            in {
                FrameEdge.LEFT,
                FrameEdge.TOP,
                FrameEdge.RIGHT,
                FrameEdge.BOTTOM,
            }
        )
        if (
            current.clipping != FrameEdge.NONE
            or current.center_censored
        ) and not recovery_one_edge:
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
    "MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S",
    "MAX_PASSAGE_SUSPENSION_EPOCHS",
    "MAX_PASSAGE_SUSPENSION_FRESH_FRAMES",
    "MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S",
    "MAX_PASSAGE_SUSPENSION_TOTAL_FRESH_FRAMES",
    "RollingVisualApproachServo",
    "VISUAL_PASSAGE_ADMISSION_BASIS",
    "VisualApproachAdjacentUnavailable",
    "VisualApproachCurrentGeometryUnavailable",
    "VisualApproachMode",
    "VisualApproachPassageAdmission",
    "VisualApproachPassageLease",
    "VisualApproachPassageLeaseState",
    "VisualApproachPassageSafetyUnavailable",
    "VisualApproachProposal",
    "VisualApproachRefusal",
]
