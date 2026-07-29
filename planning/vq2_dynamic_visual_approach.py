"""Live adapter for the build-3385 dynamic image-space course controller.

The pure estimator and guidance model lives in :mod:`planning.vq2_dynamic_course`.
This module adapts exact rolling-graph publications to that model while retaining
the existing graph identity, race-credit, wire-authority, watchdog, and cleanup
boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping, Optional

from competition.adapter import AttitudeRateCommand
from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    MultiTargetVisualTracker,
    VisualInnerApertureGeometry,
    VisualTrack,
    VisualTrackRole,
    visual_track_history_sha256,
)
from planning.vq2_dynamic_course import (
    AppliedCommandSample,
    DynamicCourseConfig,
    DynamicCourseCore,
    DynamicCourseError,
    GateObservation,
    GuidanceDecision,
    ImuAttitudeSample,
    MAX_TARGET_PITCH_RAD,
    MAX_TARGET_ROLL_RAD,
    MAX_THRUST,
    MAX_YAW_RATE_RAD_S,
    MIN_TARGET_PITCH_RAD,
    MIN_THRUST,
    SUPPORT_THRUST,
    successor_camera_pitch_reference,
    successor_roll_reference,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateReacquisition,
    ConfirmedGateTransition,
    ConfirmedSameGateRebind,
    RaceStatusProvenanceBasis,
)
from planning.vq2_visual_approach import (
    RollingVisualApproachServo,
    VISUAL_PASSAGE_ADMISSION_BASIS,
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
    VisualApproachPassageAdmission,
    VisualApproachRefusal,
)
from planning.vq2_visual_servo import (
    MAX_VISUAL_TARGET_COORDINATE_NORM,
    VisualServoOutput,
    VisualServoRefusal,
    VisualServoTuning,
    VisualTarget,
)


DYNAMIC_CONTROLLER_FAMILY = "aigp-vq2-dynamic-image-course/1"
DYNAMIC_TIMING_BASIS = (
    "receiver-final-packet-proxy-minus-identified-camera-delay"
)
DYNAMIC_CROSSING_COORDINATE_BASIS = (
    "aperture-relative-q-swept-vehicle-envelope-v1"
)
_HOST_CLOCK_ID = "host-perf-counter"
BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ = (0.0, 1.0, 0.0, 0.0)


class PostCreditSuccessorSteeringUnavailable(DynamicCourseError):
    """Optional predicted steering ended while race-owned handoff remains."""


class PropagatedCurrentVisibilityGapUnavailable(DynamicCourseError):
    """No stale local steering state remains; fresh search must take over."""


def _predicted_successor_pitch_reference(
    *,
    camera_center_y_norm: float,
    camera_center_rate_y_norm_s: float,
    vertical_angle_scale_rad: float,
    pitch_command_delay_s: float,
    maximum_lead_rad: float,
    baseline_pitch_rad: float,
) -> tuple[float, float, float, float]:
    """Convert the predicted camera elevation into a bounded pitch reference."""

    return successor_camera_pitch_reference(
        camera_center_y_norm=camera_center_y_norm,
        camera_center_rate_y_norm_s=camera_center_rate_y_norm_s,
        vertical_angle_scale_rad=vertical_angle_scale_rad,
        pitch_command_delay_s=pitch_command_delay_s,
        maximum_lead_rad=maximum_lead_rad,
        baseline_pitch_rad=baseline_pitch_rad,
    )


def _complete_current_inner_geometry(
    track: VisualTrack,
) -> VisualInnerApertureGeometry | None:
    """Return the co-timed complete aperture fit used by dynamic steering."""

    if type(track) is not VisualTrack or not track.history:
        return None
    sample = track.history[-1]
    inner = sample.inner_aperture
    if (
        sample.token != track.latest_token
        or inner is None
        or type(inner) is not VisualInnerApertureGeometry
        or not inner.fitted
        or inner.clipping != FrameEdge.NONE
        or not inner.complete_visibility
    ):
        return None
    return inner


def production_dynamic_course_config() -> DynamicCourseConfig:
    """Return the conservative first-flight identified configuration.

    The cohort cleanly identifies the body/wire roll sign but not a trustworthy
    roll-to-image acceleration magnitude.  Production therefore uses that sign
    for a modest intercept request while leaving model-based roll prediction at
    zero until an isolated characterization supplies a gain.
    """

    return replace(
        DynamicCourseConfig(),
        # The decoded camera chart is (forward, pixel-right, pixel-down).
        # Paired build-3385 pitch and yaw response identifies the effective
        # camera-to-body map as Rx(pi): forward/right/down -> forward/left/up.
        # This is a proper rotation at the camera boundary, while the estimator
        # quaternion remains the active body-FRD-to-NED orientation.
        camera_to_body_wxyz=(
            BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ
        ),
        # Rx(pi) maps camera/image right to body-FRD left.  Positive body roll
        # accelerates toward body right, so a positive stable gate bearing
        # requires negative bank.  With the heading preview and crossing
        # command held fixed, positive bank left the Gate-1 residual moving
        # outward after the measured attitude had settled.
        roll_guidance_sign=-1.0,
        roll_gain=0.18,
        lateral_rate_gain=0.045,
        roll_to_lateral_bearing_accel=0.0,
    )


@dataclass(frozen=True, slots=True)
class _StagedContext:
    expected_gate_index: int
    expected_current_track_id: str
    adjacent_precredit: bool
    passage_committed: bool
    camera_token: CameraFrameToken
    tracker_frame_sequence: int
    current_raw_clipping: FrameEdge


@dataclass(frozen=True, slots=True)
class _PostCreditSuccessorSteering:
    """Exact race-owned handoff with optional bounded steering authority."""

    authority_basis: str
    race_status: AuthoritativeRaceStatusRef
    from_gate_index: int
    to_gate_index: int
    reviewed_track_id: str
    stream_generation: int
    last_measurement_monotonic_ns: int
    last_correction_monotonic_ns: int
    activation_monotonic_ns: int
    expires_monotonic_ns: int
    prediction_horizon_s: float
    steering_available: bool
    steering_unavailable_reason: str | None
    promotion_count: int
    vertical_target_pitch_ceiling_rad: float | None


@dataclass(frozen=True, slots=True)
class _PreCreditSuccessorRollReference:
    """One exact graph-vetted successor target awaiting race promotion."""

    from_gate_index: int
    to_gate_index: int
    track_id: str
    stream_generation: int
    target_roll_rad: float
    authority_monotonic_ns: int
    accepted_wire_start_monotonic_ns: int | None


@dataclass(frozen=True, slots=True)
class _PostCreditRollReferenceHandoff:
    """One bounded successor reference retained across race-owned promotion."""

    authority_basis: str
    to_gate_index: int
    track_id: str
    stream_generation: int
    promotion_count: int
    retained_target_roll_rad: float
    source_authority_monotonic_ns: int
    source_wire_start_monotonic_ns: int
    expires_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class _PendingPostCreditRollReference:
    """One steering-only target awaiting exact accepted-wire binding."""

    to_gate_index: int
    track_id: str
    stream_generation: int
    promotion_count: int
    target_roll_rad: float
    authority_monotonic_ns: int
    expires_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class _SameGateSteeringAnchor:
    """Last exact visible same-gate reference accepted by the wire."""

    gate_index: int
    track_id: str
    stream_generation: int
    camera_token: CameraFrameToken
    wire_start_monotonic_ns: int
    decision: GuidanceDecision


@dataclass(frozen=True, slots=True)
class _WireGovernorConfig:
    max_roll_pitch_rate_rad_s: float = 0.25
    max_roll_slew_rad_s2: float = 2.0
    # Pitch needs the same bounded response authority as roll during an
    # off-axis brake.  The final body-rate clamp remains +/-0.25 rad/s.
    max_pitch_slew_rad_s2: float = 2.0
    max_yaw_slew_rad_s2: float = 0.75
    max_thrust_slew_s: float = 0.15
    max_roll_accel_rad_s3: float = 20.0
    max_pitch_accel_rad_s3: float = 20.0
    max_yaw_accel_rad_s3: float = 6.0
    max_thrust_accel_s2: float = 1.5
    max_step_s: float = 0.10


class _WireCommandGovernor:
    """Final body-rate/thrust continuity after the inner attitude loop."""

    def __init__(self) -> None:
        self.config = _WireGovernorConfig()
        self._last_ns: Optional[int] = None
        self._last: Optional[AttitudeRateCommand] = None
        self._slews = (0.0, 0.0, 0.0, 0.0)

    @property
    def last_command(self) -> Optional[AttitudeRateCommand]:
        return self._last

    @staticmethod
    def _values(command: AttitudeRateCommand) -> tuple[float, float, float, float]:
        return (
            float(command.roll_rate),
            float(command.pitch_rate),
            float(command.yaw_rate),
            float(command.thrust),
        )

    def preview(
        self,
        command: AttitudeRateCommand,
        monotonic_ns: int,
        *,
        launch_thrust_override: bool,
        yaw_safety_override: bool,
    ) -> AttitudeRateCommand:
        if type(monotonic_ns) is not int or monotonic_ns < 0:
            raise DynamicCourseError("wire-governor time is invalid")
        if self._last_ns is not None and monotonic_ns <= self._last_ns:
            raise DynamicCourseError("wire-governor preview did not advance")
        if self._last is None:
            return command
        assert self._last_ns is not None
        dt = min(
            (monotonic_ns - self._last_ns) / 1_000_000_000.0,
            self.config.max_step_s,
        )
        if dt <= 0.0:
            raise DynamicCourseError("wire-governor step is empty")
        previous = self._values(self._last)
        targets = self._values(command)
        maximum_slews = (
            self.config.max_roll_slew_rad_s2,
            self.config.max_pitch_slew_rad_s2,
            self.config.max_yaw_slew_rad_s2,
            self.config.max_thrust_slew_s,
        )
        maximum_accelerations = (
            self.config.max_roll_accel_rad_s3,
            self.config.max_pitch_accel_rad_s3,
            self.config.max_yaw_accel_rad_s3,
            self.config.max_thrust_accel_s2,
        )
        bounds = (
            (
                -self.config.max_roll_pitch_rate_rad_s,
                self.config.max_roll_pitch_rate_rad_s,
            ),
            (
                -self.config.max_roll_pitch_rate_rad_s,
                self.config.max_roll_pitch_rate_rad_s,
            ),
            (-MAX_YAW_RATE_RAD_S, MAX_YAW_RATE_RAD_S),
            (MIN_THRUST, MAX_THRUST),
        )
        values: list[float] = []
        for axis, (
            old,
            target,
            maximum_slew,
            maximum_accel,
            bound,
        ) in enumerate(
            zip(
                previous,
                targets,
                maximum_slews,
                maximum_accelerations,
                bounds,
            )
        ):
            if (axis == 3 and launch_thrust_override) or (
                axis == 2 and yaw_safety_override
            ):
                values.append(target)
                continue
            desired_slew = max(
                -maximum_slew,
                min(maximum_slew, (target - old) / dt),
            )
            previous_slew = self._slews[axis]
            slew = previous_slew + max(
                -maximum_accel * dt,
                min(
                    maximum_accel * dt,
                    desired_slew - previous_slew,
                ),
            )
            value = old + slew * dt
            if (target - old) * (target - value) <= 0.0:
                value = target
            if axis == 0 and old * value < 0.0:
                value = 0.0
            value = max(bound[0], min(bound[1], value))
            values.append(value)
        return AttitudeRateCommand(
            roll_rate=values[0],
            pitch_rate=values[1],
            yaw_rate=values[2],
            thrust=values[3],
        )

    def commit(
        self,
        command: AttitudeRateCommand,
        monotonic_ns: int,
        *,
        discontinuity: bool = False,
        discontinuity_axes: tuple[int, ...] = (),
    ) -> None:
        if type(monotonic_ns) is not int or monotonic_ns < 0:
            raise DynamicCourseError("wire-governor commit time is invalid")
        if self._last_ns is not None and monotonic_ns <= self._last_ns:
            raise DynamicCourseError("wire-governor commit did not advance")
        if any(
            type(axis) is not int or axis < 0 or axis > 3
            for axis in discontinuity_axes
        ):
            raise DynamicCourseError(
                "wire-governor discontinuity axis is invalid"
            )
        if len(set(discontinuity_axes)) != len(discontinuity_axes):
            raise DynamicCourseError(
                "wire-governor discontinuity axes must be unique"
            )
        if self._last is None or discontinuity:
            slews = (0.0, 0.0, 0.0, 0.0)
        else:
            assert self._last_ns is not None
            dt = (monotonic_ns - self._last_ns) / 1_000_000_000.0
            previous = self._values(self._last)
            current = self._values(command)
            slews = tuple(
                (current[index] - previous[index]) / dt
                for index in range(4)
            )
            if discontinuity_axes:
                slews = tuple(
                    0.0 if index in discontinuity_axes else slew
                    for index, slew in enumerate(slews)
                )
        self._last = command
        self._last_ns = monotonic_ns
        self._slews = slews


class DynamicVisualCourseSession:
    """One estimator/guidance/governor lifecycle shared across every gate."""

    def __init__(
        self,
        config: Optional[DynamicCourseConfig] = None,
    ) -> None:
        self.core = DynamicCourseCore(
            config or production_dynamic_course_config()
        )
        self._last_frame_by_track: dict[str, int] = {}
        self._staged: Optional[_StagedContext] = None
        self._last_decision: Optional[GuidanceDecision] = None
        self._wire_governor = _WireCommandGovernor()
        self._applied_command_count = 0
        self._dynamic_command_count = 0
        self._last_applied_sample: Optional[
            AppliedCommandSample
        ] = None
        self._roll_reversal_count = 0
        self._last_nonzero_roll_sign = 0
        self._post_credit_successor_steering: Optional[
            _PostCreditSuccessorSteering
        ] = None
        self._post_credit_roll_reference_handoff: Optional[
            _PostCreditRollReferenceHandoff
        ] = None
        self._pending_post_credit_roll_reference: Optional[
            _PendingPostCreditRollReference
        ] = None
        self._precredit_successor_roll_reference: Optional[
            _PreCreditSuccessorRollReference
        ] = None
        self._pending_precredit_successor_roll_reference: Optional[
            _PreCreditSuccessorRollReference
        ] = None
        self._same_gate_steering_anchor: Optional[
            _SameGateSteeringAnchor
        ] = None

    @property
    def has_applied_command(self) -> bool:
        return self._applied_command_count > 0

    @property
    def last_decision(self) -> Optional[GuidanceDecision]:
        return self._last_decision

    @property
    def post_credit_successor_steering_active(self) -> bool:
        lease = self._post_credit_successor_steering
        return bool(lease is not None and lease.steering_available)

    @property
    def post_credit_roll_reference_handoff_active(self) -> bool:
        return self._post_credit_roll_reference_handoff is not None

    def record_imu(self, sample: ImuAttitudeSample) -> None:
        self.core.record_imu(sample)

    @staticmethod
    def _track_observation(
        track: VisualTrack,
        *,
        tracker_frame_sequence: int,
        observation_monotonic_ns: int,
        stream_generation: int,
    ) -> GateObservation:
        if track.visible:
            if not track.history:
                raise DynamicCourseError(
                    "visible graph track lacks exact observation history"
                )
            sample = track.history[-1]
            if (
                sample.tracker_frame_sequence != tracker_frame_sequence
                or sample.observation_monotonic_ns
                != observation_monotonic_ns
                or sample.token != track.latest_token
            ):
                raise DynamicCourseError(
                    "graph track and tracker observation are not co-timed"
                )
            inner = sample.inner_aperture
            inner_tracking_usable = bool(
                inner is not None
                and inner.fitted
                and inner.clipping == FrameEdge.NONE
                and inner.complete_visibility
            )
            confidence = min(
                float(track.confidence),
                float(track.association_confidence),
                (
                    float(inner.confidence)
                    if inner_tracking_usable and inner is not None
                    else 0.0
                ),
            )
            if inner_tracking_usable:
                assert inner is not None
                assert inner.center_norm is not None
                assert inner.half_size_norm is not None
                assert inner.log_scale is not None
                assert inner.measurement_std is not None
                center_norm = inner.center_norm
                log_scale = inner.log_scale
                # Degraded complete fits may stabilize bearing/scale, but only
                # nominal fits can create q or passage-clearance authority.
                aperture = (
                    inner.half_size_norm
                    if inner.passage_usable
                    else None
                )
                clipping = inner.clipping
                center_censored = False
                ambiguous = bool(track.ambiguous)
                measurement_std = inner.measurement_std
            else:
                # Outer support may correct only the image axes that are not
                # clipped.  It owns no aperture scale, q, TTC, clearance, or
                # passage authority; those remain on the last clean inner
                # seed.  This still lets a freshly reacquired current gate
                # supply confidence-weighted steering before its inner
                # opening becomes measurable.
                center_norm = track.center_norm
                log_scale = math.log(float(track.apparent_scale))
                aperture = None
                clipping = track.clipping
                center_censored = bool(
                    track.center_censored
                    and track.clipping == FrameEdge.NONE
                )
                # Missing inner geometry censors control measurements, but it
                # is not itself evidence that the graph-retained track
                # identity is ambiguous.  The bounded local state may bridge
                # near-plane clipping without consuming this outer center or
                # scale.
                ambiguous = bool(track.ambiguous)
                confidence = min(
                    float(track.confidence),
                    float(track.association_confidence),
                )
                measurement_std = (0.05, 0.06, 0.12)
            return GateObservation(
                track_id=track.track_id,
                frame_sequence=tracker_frame_sequence,
                observation_monotonic_ns=observation_monotonic_ns,
                center_norm=(
                    float(center_norm[0]),
                    float(center_norm[1]),
                ),
                log_scale=float(log_scale),
                aperture_half_size_norm=aperture,
                clipping=clipping,
                center_censored=center_censored,
                visible=True,
                ambiguous=ambiguous,
                confidence=confidence,
                measurement_std=measurement_std,
                inner_scale_measurement_usable=inner_tracking_usable,
                timing_basis=DYNAMIC_TIMING_BASIS,
                timing_uncertainty_s=0.020,
                stream_generation=stream_generation,
                host_clock_id=_HOST_CLOCK_ID,
            )
        return GateObservation(
            track_id=track.track_id,
            frame_sequence=tracker_frame_sequence,
            observation_monotonic_ns=observation_monotonic_ns,
            center_norm=None,
            log_scale=None,
            aperture_half_size_norm=None,
            clipping=FrameEdge.NONE,
            center_censored=False,
            visible=False,
            ambiguous=bool(track.ambiguous),
            confidence=0.0,
            inner_scale_measurement_usable=False,
            timing_basis=DYNAMIC_TIMING_BASIS,
            timing_uncertainty_s=0.020,
            stream_generation=stream_generation,
            host_clock_id=_HOST_CLOCK_ID,
        )

    def stage_snapshot(
        self,
        snapshot: Any,
        tracker: MultiTargetVisualTracker,
        *,
        expected_gate_index: int,
        expected_current_track_id: str,
        adjacent_precredit: bool,
        passage_committed: bool = False,
    ) -> None:
        if type(passage_committed) is not bool:
            raise DynamicCourseError(
                "dynamic staged passage commitment is invalid"
            )
        if adjacent_precredit and passage_committed:
            raise DynamicCourseError(
                "adjacent steering cannot own passage commitment"
            )
        update = tracker.latest_update
        if update is None:
            raise VisualApproachRefusal(
                "dynamic course lacks an exact tracker publication"
            )
        token = snapshot.latest_camera_token
        if (
            token != update.token
            or snapshot.tracker_frame_sequence
            != update.tracker_frame_sequence
            or type(token) is not CameraFrameToken
        ):
            raise VisualApproachRefusal(
                "dynamic course graph/tracker publication differs"
            )
        track_ids = {expected_current_track_id}
        track_ids.update(
            candidate.track_id for candidate in snapshot.next_candidates
        )
        try:
            course_state = self.core.course_state()
        except DynamicCourseError:
            course_state = None
        if course_state is not None:
            track_ids.add(course_state.current_track_id)
            if course_state.successor_track_id is not None:
                track_ids.add(course_state.successor_track_id)
        known_dynamic_track_ids = {
            state.track_id for state in self.core.track_states
        }
        current_raw_clipping = FrameEdge.NONE
        for track_id in sorted(track_ids):
            try:
                track = tracker.track(track_id)
            except KeyError:
                continue
            if track_id == expected_current_track_id:
                current_raw_clipping = track.clipping
            if (
                not track.visible
                and track_id not in known_dynamic_track_ids
            ):
                if track_id == expected_current_track_id:
                    raise VisualApproachRefusal(
                        "dynamic current cannot initialize from an "
                        "invisible track"
                    )
                # A candidate that appeared only in receiver publications
                # skipped by this controller has no dynamic state and no
                # steering authority. Ignore it until a visible publication
                # can initialize it; this is normal tracker churn.
                continue
            if self._last_frame_by_track.get(track_id) == (
                update.tracker_frame_sequence
            ):
                continue
            try:
                self.core.observe_track(
                    self._track_observation(
                        track,
                        tracker_frame_sequence=(
                            update.tracker_frame_sequence
                        ),
                        observation_monotonic_ns=(
                            update.observation_monotonic_ns
                        ),
                        stream_generation=token.generation,
                    )
                )
            except DynamicCourseError as exc:
                raise VisualApproachRefusal(
                    f"dynamic track estimator refused {track_id}: {exc}"
                ) from exc
            self._last_frame_by_track[track_id] = (
                update.tracker_frame_sequence
            )
        self._staged = _StagedContext(
            expected_gate_index=expected_gate_index,
            expected_current_track_id=expected_current_track_id,
            adjacent_precredit=adjacent_precredit,
            passage_committed=passage_committed,
            camera_token=token,
            tracker_frame_sequence=update.tracker_frame_sequence,
            current_raw_clipping=current_raw_clipping,
        )

    @staticmethod
    def _post_credit_lease_evidence(
        lease: _PostCreditSuccessorSteering,
    ) -> Mapping[str, Any]:
        return {
            "basis": (
                lease.authority_basis
                if lease.steering_available
                else "authoritative-post-credit-expired-successor-handoff-v1"
            ),
            "from_gate_index": lease.from_gate_index,
            "to_gate_index": lease.to_gate_index,
            "reviewed_track_id": lease.reviewed_track_id,
            "steering_track_id": lease.reviewed_track_id,
            "stream_generation": lease.stream_generation,
            "race_status_sequence": (
                lease.race_status.race_status_sequence
            ),
            "credit_received_monotonic_ns": (
                lease.race_status.received_monotonic_ns
            ),
            "last_measurement_monotonic_ns": (
                lease.last_measurement_monotonic_ns
            ),
            "last_correction_monotonic_ns": (
                lease.last_correction_monotonic_ns
            ),
            "activation_monotonic_ns": lease.activation_monotonic_ns,
            "expires_monotonic_ns": lease.expires_monotonic_ns,
            "prediction_horizon_s": lease.prediction_horizon_s,
            "promotion_count": lease.promotion_count,
            "steering_available": lease.steering_available,
            "steering_unavailable_reason": (
                lease.steering_unavailable_reason
            ),
            "steering_only": lease.steering_available,
            "passage_authority": False,
            "advance_authority": False,
            "vertical_target_pitch_ceiling_rad": (
                lease.vertical_target_pitch_ceiling_rad
            ),
        }

    def activate_confirmed_transition_steering(
        self,
        transition: ConfirmedGateTransition,
        tracker: MultiTargetVisualTracker,
        *,
        activation_monotonic_ns: int,
    ) -> Mapping[str, Any]:
        """Synchronize dynamic roles from one exact graph-owned transition.

        Race status and the rolling graph own gate identity.  The dynamic
        successor slot is only local steering state, so a stale slot must not
        veto a graph-confirmed transition.  This method verifies the exact
        promoted tracker publication, adopts that track as the local
        successor when necessary, and then uses the normal authoritative
        promotion path.
        """

        if type(transition) is not ConfirmedGateTransition:
            raise DynamicCourseError(
                "dynamic transition activation requires exact graph proof"
            )
        if type(tracker) is not MultiTargetVisualTracker:
            raise DynamicCourseError(
                "dynamic transition activation requires the exact tracker"
            )
        update = tracker.latest_update
        if (
            update is None
            or update.token
            != transition.promoted_latest_token_at_promotion
        ):
            raise DynamicCourseError(
                "dynamic transition activation lacks its exact publication"
            )
        try:
            track = tracker.track(transition.promoted_track_id)
        except KeyError as exc:
            raise DynamicCourseError(
                "dynamic transition promoted track is absent"
            ) from exc
        if (
            track.latest_token
            != transition.promoted_latest_token_at_promotion
            or track.first_token != transition.promoted_first_token
            or len(track.history)
            != transition.history_length_after_promotion
            or visual_track_history_sha256(track.history)
            != transition.promoted_history_sha256
            or not track.visible
            or track.ambiguous
            or track.role is not VisualTrackRole.CURRENT
            or track.authoritative_gate_index
            != transition.to_gate_index
            or track.authority_race_status_sequence
            != transition.race_status.race_status_sequence
        ):
            raise DynamicCourseError(
                "dynamic transition differs from graph-owned promoted state"
            )

        state = self.core.course_state()
        if (
            state.current_gate_index != transition.from_gate_index
            or state.current_track_id != transition.retired_track_id
            or state.current_track_id == transition.promoted_track_id
        ):
            raise DynamicCourseError(
                "dynamic transition predecessor ownership differs"
            )
        known_track_ids = {
            dynamic.track_id for dynamic in self.core.track_states
        }
        if (
            self._last_frame_by_track.get(track.track_id)
            != update.tracker_frame_sequence
        ):
            self.core.observe_track(
                self._track_observation(
                    track,
                    tracker_frame_sequence=update.tracker_frame_sequence,
                    observation_monotonic_ns=(
                        update.observation_monotonic_ns
                    ),
                    stream_generation=update.token.generation,
                )
            )
            self._last_frame_by_track[track.track_id] = (
                update.tracker_frame_sequence
            )
        elif track.track_id not in known_track_ids:
            raise DynamicCourseError(
                "dynamic transition publication was consumed without state"
            )

        previous_successor_track_id = state.successor_track_id
        successor_reconciled = bool(
            previous_successor_track_id
            != transition.promoted_track_id
        )
        if successor_reconciled:
            if previous_successor_track_id is not None:
                self.core.handoff_graph_vetted_successor_state(
                    predecessor_track_id=previous_successor_track_id,
                    replacement_track_id=transition.promoted_track_id,
                )
            state = self.core.bind(
                current_gate_index=state.current_gate_index,
                current_track_id=state.current_track_id,
                successor_track_id=transition.promoted_track_id,
            )
        successor = state.successor
        if (
            successor is None
            or successor.track_id != transition.promoted_track_id
            or successor.stream_generation != update.token.generation
        ):
            raise DynamicCourseError(
                "dynamic transition could not adopt graph successor"
            )

        evidence = dict(
            self.activate_post_credit_successor_steering(
                transition.race_status,
                from_gate_index=transition.from_gate_index,
                reviewed_track_id=transition.promoted_track_id,
                activation_monotonic_ns=activation_monotonic_ns,
            )
        )
        evidence.update(
            {
                "graph_transition_reconciled": successor_reconciled,
                "previous_successor_track_id": (
                    previous_successor_track_id
                ),
                "graph_transition_history_sha256": (
                    transition.promoted_history_sha256
                ),
            }
        )
        return evidence

    def activate_post_credit_successor_steering(
        self,
        race_status: AuthoritativeRaceStatusRef,
        *,
        from_gate_index: int,
        reviewed_track_id: str,
        activation_monotonic_ns: int,
    ) -> Mapping[str, Any]:
        """Promote a reviewed successor and expose steering only while valid.

        The authoritative race status supplies gate ownership.  This method
        changes only the dynamic controller's current/successor roles; rolling
        graph promotion remains exclusively owned by the graph coordinator.
        An expired prediction still records the race-owned handoff needed for
        an exact graph-proven reacquisition, but grants no command authority.
        """

        if type(race_status) is not AuthoritativeRaceStatusRef:
            raise DynamicCourseError(
                "post-credit steering requires exact race status"
            )
        if (
            race_status.provenance_basis
            is not RaceStatusProvenanceBasis.LIVE_INGRESS
            or race_status.race_finished
            or race_status.host_clock_id != _HOST_CLOCK_ID
            or type(race_status.race_status_sequence) is not int
            or type(race_status.received_monotonic_ns) is not int
        ):
            raise DynamicCourseError(
                "post-credit steering requires live nonterminal race credit"
            )
        if type(from_gate_index) is not int or from_gate_index < 0:
            raise DynamicCourseError(
                "post-credit steering source gate is invalid"
            )
        if type(reviewed_track_id) is not str or not reviewed_track_id:
            raise DynamicCourseError(
                "post-credit steering reviewed identity is invalid"
            )
        if (
            type(activation_monotonic_ns) is not int
            or activation_monotonic_ns < 0
        ):
            raise DynamicCourseError(
                "post-credit steering activation clock is invalid"
            )
        to_gate_index = from_gate_index + 1
        if race_status.active_gate_index != to_gate_index:
            raise DynamicCourseError(
                "post-credit steering race credit is not sequential"
            )

        active = self._post_credit_successor_steering
        if active is not None:
            if (
                active.race_status == race_status
                and active.from_gate_index == from_gate_index
                and active.reviewed_track_id == reviewed_track_id
                and active.activation_monotonic_ns
                == activation_monotonic_ns
            ):
                state = self.core.course_state()
                if (
                    state.current_gate_index != active.to_gate_index
                    or state.current_track_id != active.reviewed_track_id
                    or state.promotion_count != active.promotion_count
                ):
                    raise DynamicCourseError(
                        "post-credit steering lease lost dynamic ownership"
                    )
                return self._post_credit_lease_evidence(active)
            raise DynamicCourseError(
                "a different post-credit steering lease is already active"
            )

        state = self.core.course_state()
        successor = state.successor
        if (
            state.current_gate_index != from_gate_index
            or state.current_track_id == reviewed_track_id
            or state.successor_track_id != reviewed_track_id
            or successor is None
            or successor.track_id != reviewed_track_id
            or successor.stream_generation
            != state.current.stream_generation
        ):
            raise DynamicCourseError(
                "post-credit steering lacks the reviewed bound successor"
            )
        received_ns = race_status.received_monotonic_ns
        assert type(received_ns) is int
        last_measurement_ns = successor.last_measurement_monotonic_ns
        horizon_ns = round(
            self.core.config.successor_prediction_max_horizon_s
            * 1_000_000_000.0
        )
        expires_ns = min(
            received_ns + horizon_ns,
            last_measurement_ns + horizon_ns,
        )
        if (
            activation_monotonic_ns < received_ns
            or activation_monotonic_ns
            < successor.state_monotonic_ns
        ):
            raise DynamicCourseError(
                "post-credit activation is outside its causal bounded horizon"
            )
        steering_available = bool(
            successor.sample_count >= 4
            and not successor.ambiguous
            and activation_monotonic_ns <= expires_ns
        )
        steering_unavailable_reason = (
            None
            if steering_available
            else (
                "ambiguous_reviewed_state"
                if successor.ambiguous
                else (
                    "insufficient_reviewed_state"
                    if successor.sample_count < 4
                    else "expired_prediction"
                )
            )
        )
        applied = self._last_applied_sample
        if (
            applied is None
            or self._wire_governor.last_command is None
            or applied.monotonic_ns > activation_monotonic_ns
        ):
            raise DynamicCourseError(
                "post-credit steering lacks causal accepted-command memory"
            )
        promoted = self.core.promote_authoritative(
            from_gate_index=from_gate_index,
            to_gate_index=to_gate_index,
            promoted_track_id=reviewed_track_id,
            next_successor_track_id=None,
            monotonic_ns=activation_monotonic_ns,
        )
        lease = _PostCreditSuccessorSteering(
            authority_basis=(
                "authoritative-post-credit-propagated-successor-steering-v1"
            ),
            race_status=race_status,
            from_gate_index=from_gate_index,
            to_gate_index=to_gate_index,
            reviewed_track_id=reviewed_track_id,
            stream_generation=successor.stream_generation,
            last_measurement_monotonic_ns=last_measurement_ns,
            last_correction_monotonic_ns=last_measurement_ns,
            activation_monotonic_ns=activation_monotonic_ns,
            expires_monotonic_ns=expires_ns,
            prediction_horizon_s=(
                self.core.config.successor_prediction_max_horizon_s
            ),
            steering_available=steering_available,
            steering_unavailable_reason=steering_unavailable_reason,
            promotion_count=promoted.promotion_count,
            vertical_target_pitch_ceiling_rad=None,
        )
        self._post_credit_successor_steering = lease
        self._pending_post_credit_roll_reference = None
        self._pending_precredit_successor_roll_reference = None
        self._precredit_successor_roll_reference = None
        # The committed pre-credit target ends at authoritative promotion.
        # Promoted-gate roll is recomputed proportionally from its own fresh
        # geometry; no accepted predecessor bank may become a temporal latch.
        self._post_credit_roll_reference_handoff = None
        self._staged = None
        self._last_decision = None
        return self._post_credit_lease_evidence(lease)

    def _refresh_post_credit_successor_state(
        self,
        lease: _PostCreditSuccessorSteering,
    ) -> _PostCreditSuccessorSteering:
        """Correct observable axes without renewing fully censored geometry."""

        state = self.core.course_state()
        current = state.current
        if (
            state.current_gate_index != lease.to_gate_index
            or state.current_track_id != lease.reviewed_track_id
            or current.track_id != lease.reviewed_track_id
            or current.stream_generation != lease.stream_generation
            or state.promotion_count != lease.promotion_count
        ):
            raise DynamicCourseError(
                "post-credit successor steering lost dynamic ownership"
            )
        if (
            current.last_measurement_monotonic_ns
            < lease.last_measurement_monotonic_ns
        ):
            raise DynamicCourseError(
                "post-credit successor measurement clock regressed"
            )
        if (
            current.last_measurement_monotonic_ns
            == lease.last_measurement_monotonic_ns
        ):
            return lease

        staged = self._staged
        if (
            staged is None
            or staged.adjacent_precredit
            or staged.expected_gate_index != lease.to_gate_index
            or staged.expected_current_track_id
            != lease.reviewed_track_id
            or staged.camera_token.generation != lease.stream_generation
            or current.frame_sequence != staged.tracker_frame_sequence
            or current.state_monotonic_ns
            != current.last_measurement_monotonic_ns
            or not current.visible
            or current.ambiguous
            or current.missed_count != 0
        ):
            raise DynamicCourseError(
                "post-credit successor correction lacks exact current state"
            )

        correction_available = not all(current.censored_axes)
        correction_ns = lease.last_correction_monotonic_ns
        expires_ns = lease.expires_monotonic_ns
        if (
            correction_available
            and current.state_monotonic_ns <= expires_ns
        ):
            correction_ns = current.state_monotonic_ns
            prediction_horizon_s = (
                self.core.config
                .post_credit_current_prediction_max_horizon_s
            )
            expires_ns = correction_ns + round(
                prediction_horizon_s
                * 1_000_000_000.0
            )
        else:
            prediction_horizon_s = lease.prediction_horizon_s
        refreshed = replace(
            lease,
            last_measurement_monotonic_ns=(
                current.last_measurement_monotonic_ns
            ),
            last_correction_monotonic_ns=correction_ns,
            expires_monotonic_ns=expires_ns,
            prediction_horizon_s=prediction_horizon_s,
        )
        self._post_credit_successor_steering = refreshed
        return refreshed

    def post_credit_successor_steering_authority(
        self,
        *,
        now_monotonic_ns: int,
    ) -> Mapping[str, Any]:
        """Return bounded steering-only authority while successor vision gaps."""

        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise DynamicCourseError(
                "post-credit steering clock is invalid"
            )
        lease = self._post_credit_successor_steering
        if lease is None:
            raise DynamicCourseError(
                "post-credit successor steering is not active"
            )
        if not lease.steering_available:
            raise PostCreditSuccessorSteeringUnavailable(
                "post-credit successor handoff has no steering authority"
            )
        if now_monotonic_ns < lease.activation_monotonic_ns:
            raise DynamicCourseError(
                "post-credit successor steering precedes activation"
            )
        lease = self._refresh_post_credit_successor_state(lease)
        if now_monotonic_ns > lease.expires_monotonic_ns:
            self._post_credit_successor_steering = replace(
                lease,
                steering_available=False,
                steering_unavailable_reason="expired_prediction",
            )
            raise PostCreditSuccessorSteeringUnavailable(
                "post-credit successor steering expired"
            )
        state = self.core.course_state()
        if (
            state.current_gate_index != lease.to_gate_index
            or state.current_track_id != lease.reviewed_track_id
            or state.current.stream_generation != lease.stream_generation
            or state.promotion_count != lease.promotion_count
        ):
            raise DynamicCourseError(
                "post-credit successor steering lost dynamic ownership"
            )
        prediction = self.core.predict_track_steering(
            lease.reviewed_track_id,
            now_monotonic_ns,
        )
        if (
            prediction.stream_generation != lease.stream_generation
            or prediction.last_measurement_monotonic_ns
            != lease.last_measurement_monotonic_ns
        ):
            raise DynamicCourseError(
                "post-credit successor steering prediction differs from "
                "the credited state"
            )
        if (
            prediction.measurement_age_s
            > lease.prediction_horizon_s + 1e-12
        ):
            self._post_credit_successor_steering = replace(
                lease,
                steering_available=False,
                steering_unavailable_reason="expired_prediction",
            )
            raise PostCreditSuccessorSteeringUnavailable(
                "post-credit successor steering prediction expired"
            )
        if any(
            value
            > self.core.config.successor_prediction_max_extrapolation_rad
            + 1e-12
            for value in prediction.bearing_std_rad
        ):
            self._post_credit_successor_steering = replace(
                lease,
                steering_available=False,
                steering_unavailable_reason=(
                    "excessive_prediction_uncertainty"
                ),
            )
            raise PostCreditSuccessorSteeringUnavailable(
                "post-credit successor steering uncertainty expired"
            )

        targets = self._successor_steering_targets(prediction)
        unconstrained_target_roll = float(targets["target_roll_rad"])
        yaw_rate = float(targets["yaw_rate_rad_s"])
        # After credit, fresh successor geometry may bend the trajectory as
        # well as point the camera.  Never retain the old pre-credit bank.
        # When the exact promoted state is moving outward while calibrated yaw
        # is already saturated, use the full bounded lateral intercept on this
        # publication instead of spending the short visibility window ramping
        # a small proportional request.  No target is latched: an inward or
        # degraded next publication releases immediately.
        self._pending_post_credit_roll_reference = None
        self._post_credit_roll_reference_handoff = None
        target_roll = unconstrained_target_roll
        outward_full_bank_applied = False
        exact_fresh_promoted_current = bool(
            state.current.visible
            and state.current.missed_count == 0
            and state.current.state_monotonic_ns
            == state.current.last_measurement_monotonic_ns
            and not state.current.ambiguous
            and not state.current.censored_axes[0]
            and state.current.bearing_rate_qualified[0]
            and prediction.last_measurement_monotonic_ns
            == state.current.last_measurement_monotonic_ns
            and abs(yaw_rate) >= 0.90 * MAX_YAW_RATE_RAD_S
        )
        if exact_fresh_promoted_current:
            (
                target_roll,
                outward_full_bank_applied,
            ) = successor_roll_reference(
                stable_bearing_rad=prediction.stable_bearing_rad[0],
                stable_bearing_rate_rad_s=(
                    prediction.stable_bearing_rate_rad_s[0]
                ),
                bearing_std_rad=prediction.bearing_std_rad[0],
                roll_guidance_sign=self.core.config.roll_guidance_sign,
                roll_gain=self.core.config.roll_gain,
                lateral_rate_gain=self.core.config.lateral_rate_gain,
                off_axis_brake_rad=self.core.config.off_axis_brake_rad,
                maximum_bearing_std_rad=(
                    self.core.config
                    .successor_prediction_max_extrapolation_rad
                ),
                full_bank_when_outward=True,
            )
        target_pitch = float(targets["target_pitch_rad"])
        camera_elevation = float(
            targets["camera_elevation_error_rad"]
        )
        camera_elevation_rate = float(
            targets["camera_elevation_rate_rad_s"]
        )
        pitch_delay_lead = float(targets["pitch_delay_lead_rad"])
        staged = self._staged
        staged_raw_clipping = FrameEdge.NONE
        if (
            staged is not None
            and staged.expected_gate_index == state.current_gate_index
            and staged.expected_current_track_id == state.current_track_id
            and staged.camera_token.generation
            == state.current.stream_generation
            and staged.tracker_frame_sequence
            == state.current.frame_sequence
        ):
            staged_raw_clipping = staged.current_raw_clipping
        vertical_axis_censored = bool(
            not state.current.visible
            or state.current.censored_axes[1]
            # A complete tracking-only inner fit may correctly preserve
            # steering while the outer gate support remains at a frame edge.
            # Raw camera geometry still owns FOV protection, so that degraded
            # fit cannot release the retained pitch ceiling.
            or staged_raw_clipping
            & (FrameEdge.TOP | FrameEdge.BOTTOM)
        )
        retained_pitch_ceiling = (
            lease.vertical_target_pitch_ceiling_rad
        )
        if vertical_axis_censored:
            if retained_pitch_ceiling is None:
                retained_pitch_ceiling = target_pitch
            target_pitch = min(target_pitch, retained_pitch_ceiling)
            retained_pitch_ceiling = min(
                retained_pitch_ceiling,
                target_pitch,
            )
        else:
            retained_pitch_ceiling = target_pitch
        if (
            lease.vertical_target_pitch_ceiling_rad
            != retained_pitch_ceiling
        ):
            lease = replace(
                lease,
                vertical_target_pitch_ceiling_rad=(
                    retained_pitch_ceiling
                ),
            )
            self._post_credit_successor_steering = lease
        values = (
            target_roll,
            target_pitch,
            yaw_rate,
            SUPPORT_THRUST,
            *prediction.stable_bearing_rad,
            *prediction.stable_bearing_rate_rad_s,
            *prediction.camera_center_norm,
            *prediction.camera_center_rate_norm_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise DynamicCourseError(
                "post-credit successor steering produced non-finite authority"
            )
        applied = self._last_applied_sample
        wire_command = self._wire_governor.last_command
        if (
            applied is None
            or wire_command is None
            or applied.monotonic_ns > now_monotonic_ns
        ):
            raise DynamicCourseError(
                "post-credit steering lacks accepted wire continuity"
            )
        evidence = dict(self._post_credit_lease_evidence(lease))
        evidence.update(
            {
                "target_roll_rad": target_roll,
                "unconstrained_target_roll_rad": unconstrained_target_roll,
                "retained_roll_reference_applied": False,
                "outward_full_bank_applied": (
                    outward_full_bank_applied
                ),
                "target_pitch_rad": target_pitch,
                "yaw_rate_rad_s": yaw_rate,
                "thrust": SUPPORT_THRUST,
                "wire_command": wire_command,
                "source_wire_start_monotonic_ns": (
                    applied.monotonic_ns
                ),
                "authority_monotonic_ns": now_monotonic_ns,
                "measurement_age_s": prediction.measurement_age_s,
                "horizon_remaining_s": (
                    lease.expires_monotonic_ns - now_monotonic_ns
                )
                / 1_000_000_000.0,
                "stable_bearing_rad": list(
                    prediction.stable_bearing_rad
                ),
                "stable_bearing_rate_rad_s": list(
                    prediction.stable_bearing_rate_rad_s
                ),
                "camera_center_norm": list(
                    prediction.camera_center_norm
                ),
                "camera_center_rate_norm_s": list(
                    prediction.camera_center_rate_norm_s
                ),
                "camera_elevation_error_rad": camera_elevation,
                "camera_elevation_rate_rad_s": camera_elevation_rate,
                "pitch_delay_lead_rad": pitch_delay_lead,
                "vertical_axis_censored": vertical_axis_censored,
                "current_raw_clipping": int(staged_raw_clipping),
                "retained_pitch_ceiling_rad": retained_pitch_ceiling,
                "bearing_std_rad": list(
                    prediction.bearing_std_rad
                ),
                "body_rates_rad_s": list(
                    prediction.body_rates_rad_s
                ),
            }
        )
        # Fresh promoted-gate roll is recomputed every frame.  It must never
        # become a retained handoff reference.
        self._pending_post_credit_roll_reference = None
        return evidence

    def _retain_fresh_reacquisition_roll_reference(
        self,
        *,
        lease: _PostCreditSuccessorSteering,
        target_roll_rad: float,
        stable_bearing_rad: float,
        now_monotonic_ns: int,
    ) -> float:
        """Keep accepted bank while exact fresh geometry needs that correction.

        The reference is an already accepted gate-relative attitude target,
        not propagated predecessor geometry and not a temporal command
        governor.  Only an exact fresh post-credit lease may consume it; the
        final wire governor remains the sole command-continuity authority.
        """

        handoff = self._post_credit_roll_reference_handoff
        if handoff is None:
            return target_roll_rad
        state = self.core.course_state()
        current = state.current
        retained_roll = float(handoff.retained_target_roll_rad)
        guidance_sign = float(self.core.config.roll_guidance_sign)
        lineage_matches = bool(
            handoff.to_gate_index == lease.to_gate_index
            and handoff.track_id == lease.reviewed_track_id
            and handoff.stream_generation == lease.stream_generation
            and handoff.promotion_count == lease.promotion_count
            and state.current_gate_index == handoff.to_gate_index
            and state.current_track_id == handoff.track_id
            and current.track_id == handoff.track_id
            and current.stream_generation == handoff.stream_generation
            and state.promotion_count == handoff.promotion_count
        )
        values = (
            target_roll_rad,
            retained_roll,
            stable_bearing_rad,
            guidance_sign,
        )
        bounded_state = bool(
            lineage_matches
            and all(math.isfinite(value) for value in values)
            and 0.0 < abs(retained_roll) <= MAX_TARGET_ROLL_RAD
            and current.visible
            and not current.ambiguous
            and not current.censored_axes[0]
            and current.bearing_std_rad[0]
            <= (
                self.core.config.successor_prediction_max_extrapolation_rad
            )
            + 1e-12
            and abs(guidance_sign) > 1e-12
        )
        direction = 1.0 if retained_roll > 0.0 else -1.0
        same_corrective_demand = direction * target_roll_rad > 1e-12
        error_still_requires_correction = bool(
            direction * guidance_sign * stable_bearing_rad > 1e-12
        )
        demand_caught_up = bool(
            same_corrective_demand
            and abs(target_roll_rad) + 1e-12 >= abs(retained_roll)
        )
        if (
            not bounded_state
            or not same_corrective_demand
            or not error_still_requires_correction
            or demand_caught_up
        ):
            self._post_credit_roll_reference_handoff = None
            return target_roll_rad
        constrained = math.copysign(
            max(abs(target_roll_rad), abs(retained_roll)),
            retained_roll,
        )
        if (
            not math.isfinite(constrained)
            or abs(constrained) > MAX_TARGET_ROLL_RAD + 1e-12
        ):
            raise DynamicCourseError(
                "fresh reacquisition roll reference left the bounded envelope"
            )
        return constrained

    def complete_post_credit_recovery(
        self,
        *,
        camera_token: CameraFrameToken,
    ) -> Mapping[str, Any]:
        """Release predecessor steering after exact observable current control."""

        if type(camera_token) is not CameraFrameToken:
            raise DynamicCourseError(
                "post-credit recovery completion requires an exact token"
            )
        lease = self._post_credit_successor_steering
        staged = self._staged
        state = self.core.course_state()
        current = state.current
        if (
            lease is None
            or staged is None
            or staged.adjacent_precredit
            or staged.camera_token != camera_token
            or staged.expected_gate_index != lease.to_gate_index
            or staged.expected_current_track_id
            != lease.reviewed_track_id
            or state.current_gate_index != lease.to_gate_index
            or state.current_track_id != lease.reviewed_track_id
            or current.track_id != lease.reviewed_track_id
            or current.stream_generation != camera_token.generation
            or current.frame_sequence != staged.tracker_frame_sequence
            or not current.visible
            or current.ambiguous
            or current.missed_count != 0
            or all(current.censored_axes)
        ):
            raise DynamicCourseError(
                "post-credit recovery completion lacks fresh observable "
                "current state"
            )
        measurement_mode = (
            "one_axis_censored"
            if any(current.censored_axes)
            else "clean"
        )
        evidence = dict(self._post_credit_lease_evidence(lease))
        evidence.update(
            {
                "basis": "fresh-current-post-credit-recovery-release-v2",
                "camera_token": asdict(camera_token),
                "measurement_mode": measurement_mode,
                "current_censored_axes": list(current.censored_axes),
                "steering_only": False,
                "passage_authority": False,
                "advance_authority": False,
            }
        )
        self._post_credit_successor_steering = None
        return evidence

    def rebind_confirmed_reacquisition(
        self,
        reacquisition: ConfirmedGateReacquisition,
        tracker: MultiTargetVisualTracker,
    ) -> Mapping[str, Any]:
        """Bind a graph-proven fresh post-credit identity into dynamic state."""

        if type(reacquisition) is not ConfirmedGateReacquisition:
            raise DynamicCourseError(
                "dynamic fresh rebind requires exact graph proof"
            )
        if type(tracker) is not MultiTargetVisualTracker:
            raise DynamicCourseError(
                "dynamic fresh rebind requires the exact tracker"
            )
        lease = self._post_credit_successor_steering
        if lease is None:
            raise DynamicCourseError(
                "dynamic fresh rebind lacks an active credited lease"
            )
        advance = reacquisition.credited_advance
        if (
            reacquisition.cross_gap_identity_claimed
            or advance.race_status != lease.race_status
            or advance.from_gate_index != lease.from_gate_index
            or advance.to_gate_index != lease.to_gate_index
            or advance.reviewed_track_id != lease.reviewed_track_id
            or reacquisition.gate_index != lease.to_gate_index
            or reacquisition.reacquired_track_id
            == lease.reviewed_track_id
        ):
            raise DynamicCourseError(
                "dynamic fresh rebind proof differs from credited ownership"
            )
        update = tracker.latest_update
        if (
            update is None
            or update.token != reacquisition.camera_token_at_binding
        ):
            raise DynamicCourseError(
                "dynamic fresh rebind lacks the exact binding publication"
            )
        try:
            track = tracker.track(reacquisition.reacquired_track_id)
        except KeyError as exc:
            raise DynamicCourseError(
                "dynamic fresh rebind track is absent"
            ) from exc
        if (
            not track.history
            or
            track.latest_token != reacquisition.camera_token_at_binding
            or track.first_token != reacquisition.reacquired_first_token
            or not track.visible
            or track.ambiguous
            or track.role is not VisualTrackRole.CURRENT
            or track.authoritative_gate_index != lease.to_gate_index
            or track.authority_race_status_sequence
            != lease.race_status.race_status_sequence
        ):
            raise DynamicCourseError(
                "dynamic fresh rebind track lacks graph current authority"
            )
        binding_publication_ns = (
            track.history[-1].publication_monotonic_ns
        )
        if (
            type(binding_publication_ns) is not int
            or type(update.publish_monotonic_ns) is not int
            or binding_publication_ns != update.publish_monotonic_ns
            or binding_publication_ns < update.observation_monotonic_ns
        ):
            raise DynamicCourseError(
                "dynamic fresh rebind publication clock is invalid"
            )
        known_track_ids = {
            state.track_id for state in self.core.track_states
        }
        if (
            self._last_frame_by_track.get(track.track_id)
            != update.tracker_frame_sequence
        ):
            self.core.observe_track(
                self._track_observation(
                    track,
                    tracker_frame_sequence=update.tracker_frame_sequence,
                    observation_monotonic_ns=(
                        update.observation_monotonic_ns
                    ),
                    stream_generation=update.token.generation,
                )
            )
            self._last_frame_by_track[track.track_id] = (
                update.tracker_frame_sequence
            )
        elif track.track_id not in known_track_ids:
            raise DynamicCourseError(
                "dynamic fresh rebind publication was consumed without state"
            )
        rebound = self.core.bind(
            current_gate_index=lease.to_gate_index,
            current_track_id=track.track_id,
            successor_track_id=None,
        )
        current = rebound.current
        activation_monotonic_ns = max(
            lease.activation_monotonic_ns,
            binding_publication_ns,
        )
        expires_monotonic_ns = (
            current.state_monotonic_ns
            + round(
                self.core.config
                .post_credit_current_prediction_max_horizon_s
                * 1_000_000_000.0
            )
        )
        fresh_state_available = bool(
            current.track_id == track.track_id
            and current.stream_generation == update.token.generation
            and current.frame_sequence == update.tracker_frame_sequence
            and current.last_measurement_monotonic_ns
            == current.state_monotonic_ns
            and current.visible
            and not current.ambiguous
            and current.missed_count == 0
            # A fresh exact rebind needs at least one observable image axis
            # for steering, not complete geometry.  The censored axis remains
            # on bounded prediction and this lease can never create passage
            # or gate-advance authority; clean geometry is still required to
            # complete post-credit recovery below.
            and not all(current.censored_axes)
            and all(
                value
                <= (
                    self.core.config
                    .successor_prediction_max_extrapolation_rad
                )
                + 1e-12
                for value in current.bearing_std_rad
            )
            and activation_monotonic_ns <= expires_monotonic_ns
        )
        rebound_lease = replace(
            lease,
            authority_basis=(
                "authoritative-post-credit-fresh-reacquisition-steering-v1"
            ),
            reviewed_track_id=track.track_id,
            stream_generation=update.token.generation,
            last_measurement_monotonic_ns=(
                current.last_measurement_monotonic_ns
            ),
            last_correction_monotonic_ns=current.state_monotonic_ns,
            activation_monotonic_ns=activation_monotonic_ns,
            expires_monotonic_ns=expires_monotonic_ns,
            prediction_horizon_s=(
                self.core.config
                .post_credit_current_prediction_max_horizon_s
            ),
            steering_available=fresh_state_available,
            steering_unavailable_reason=(
                None
                if fresh_state_available
                else "fresh_reacquisition_state_unavailable"
            ),
            promotion_count=rebound.promotion_count,
            vertical_target_pitch_ceiling_rad=None,
        )
        self._post_credit_successor_steering = rebound_lease
        roll_handoff = self._post_credit_roll_reference_handoff
        if roll_handoff is not None:
            retained_roll = float(roll_handoff.retained_target_roll_rad)
            fresh_bearing = float(current.bearing_rad[0])
            guidance_sign = float(self.core.config.roll_guidance_sign)
            fresh_horizontal_reference = bool(
                fresh_state_available
                and not current.censored_axes[0]
                and all(
                    math.isfinite(value)
                    for value in (
                        retained_roll,
                        fresh_bearing,
                        guidance_sign,
                    )
                )
                and 0.0 < abs(retained_roll) <= MAX_TARGET_ROLL_RAD
                and abs(guidance_sign) > 1e-12
                and abs(fresh_bearing)
                >= self.core.config.off_axis_brake_rad - 1e-12
                and retained_roll * guidance_sign * fresh_bearing > 1e-12
            )
            if (
                roll_handoff.to_gate_index == lease.to_gate_index
                and roll_handoff.track_id == lease.reviewed_track_id
                and roll_handoff.stream_generation
                == lease.stream_generation
                and roll_handoff.promotion_count
                == lease.promotion_count
                and fresh_horizontal_reference
            ):
                self._post_credit_roll_reference_handoff = replace(
                    roll_handoff,
                    authority_basis=(
                        "graph-proven-fresh-reacquisition-roll-reference-v1"
                    ),
                    track_id=track.track_id,
                    stream_generation=update.token.generation,
                    promotion_count=rebound.promotion_count,
                    expires_monotonic_ns=expires_monotonic_ns,
                )
            else:
                self._post_credit_roll_reference_handoff = None
        self._staged = None
        self._last_decision = None
        evidence = {
            "basis": "graph-proven-fresh-post-credit-dynamic-rebind-v1",
            "from_gate_index": lease.from_gate_index,
            "to_gate_index": lease.to_gate_index,
            "reviewed_track_id": lease.reviewed_track_id,
            "reacquired_track_id": track.track_id,
            "binding_camera_token": asdict(
                reacquisition.camera_token_at_binding
            ),
            "stream_generation": update.token.generation,
            "promotion_count": rebound.promotion_count,
            "cross_gap_identity_claimed": False,
            "steering_available": fresh_state_available,
            "steering_only": fresh_state_available,
            "passage_authority": False,
            "advance_authority": False,
            "recovery_steering": dict(
                self._post_credit_lease_evidence(rebound_lease)
            ),
        }
        return evidence

    def rebind_confirmed_same_gate(
        self,
        rebind: ConfirmedSameGateRebind,
        tracker: MultiTargetVisualTracker,
    ) -> Mapping[str, Any]:
        """Replace a lost dynamic CURRENT without advancing the race gate."""

        if type(rebind) is not ConfirmedSameGateRebind:
            raise DynamicCourseError(
                "dynamic same-gate rebind requires exact graph proof"
            )
        if type(tracker) is not MultiTargetVisualTracker:
            raise DynamicCourseError(
                "dynamic same-gate rebind requires the exact tracker"
            )
        update = tracker.latest_update
        if (
            update is None
            or update.token != rebind.camera_token_at_binding
        ):
            raise DynamicCourseError(
                "dynamic same-gate rebind lacks the exact binding publication"
            )
        try:
            track = tracker.track(rebind.rebound_track_id)
        except KeyError as exc:
            raise DynamicCourseError(
                "dynamic same-gate rebound track is absent"
            ) from exc
        if (
            rebind.cross_gap_identity_claimed
            or rebind.gate_index != rebind.search.gate_index
            or rebind.retired_track_id != rebind.search.lost_track_id
            or rebind.rebound_track_id == rebind.retired_track_id
            or not track.history
            or track.latest_token != rebind.camera_token_at_binding
            or track.first_token != rebind.rebound_first_token
            or len(track.history) != rebind.history_length_at_binding
            or not track.visible
            or track.ambiguous
            or track.role is not VisualTrackRole.CURRENT
            or track.authoritative_gate_index != rebind.gate_index
            or track.authority_race_status_sequence
            != rebind.race_status_at_binding.race_status_sequence
        ):
            raise DynamicCourseError(
                "dynamic same-gate rebound lacks graph current authority"
            )
        state_before = self.core.course_state()
        if (
            state_before.current_gate_index != rebind.gate_index
            or state_before.current_track_id != rebind.retired_track_id
        ):
            raise DynamicCourseError(
                "dynamic same-gate rebind differs from current ownership"
            )
        known_track_ids = {
            state.track_id for state in self.core.track_states
        }
        if (
            self._last_frame_by_track.get(track.track_id)
            != update.tracker_frame_sequence
        ):
            self.core.observe_track(
                self._track_observation(
                    track,
                    tracker_frame_sequence=update.tracker_frame_sequence,
                    observation_monotonic_ns=(
                        update.observation_monotonic_ns
                    ),
                    stream_generation=update.token.generation,
                )
            )
            self._last_frame_by_track[track.track_id] = (
                update.tracker_frame_sequence
            )
        elif track.track_id not in known_track_ids:
            raise DynamicCourseError(
                "dynamic same-gate binding publication was consumed "
                "without state"
            )
        rebound_state = self.core.bind(
            current_gate_index=rebind.gate_index,
            current_track_id=track.track_id,
            successor_track_id=None,
        )
        current = rebound_state.current
        fresh_state_available = bool(
            current.track_id == track.track_id
            and current.stream_generation == update.token.generation
            and current.frame_sequence == update.tracker_frame_sequence
            and current.last_measurement_monotonic_ns
            == current.state_monotonic_ns
            and current.visible
            and not current.ambiguous
            and current.missed_count == 0
            and not all(current.censored_axes)
        )
        if (
            not fresh_state_available
            or rebound_state.promotion_count
            != state_before.promotion_count
        ):
            raise DynamicCourseError(
                "dynamic same-gate rebound lacks fresh steering state"
            )

        # No predecessor command, prediction lease, passage state, or
        # successor slot survives an identity replacement. The wire governor
        # remains continuous; the exact binding frame is planned normally.
        self._staged = None
        self._last_decision = None
        self._post_credit_successor_steering = None
        self._post_credit_roll_reference_handoff = None
        self._pending_post_credit_roll_reference = None
        self._precredit_successor_roll_reference = None
        self._pending_precredit_successor_roll_reference = None
        self._same_gate_steering_anchor = None
        return {
            "basis": "graph-proven-fresh-same-gate-dynamic-rebind-v1",
            "gate_index": rebind.gate_index,
            "retired_track_id": rebind.retired_track_id,
            "rebound_track_id": rebind.rebound_track_id,
            "current_track_id": rebind.rebound_track_id,
            "binding_camera_token": asdict(
                rebind.camera_token_at_binding
            ),
            "stream_generation": update.token.generation,
            "promotion_count": rebound_state.promotion_count,
            "cross_gap_identity_claimed": False,
            "fresh_state_available": True,
            "same_gate_current_authority": True,
            "passage_authority": False,
            "advance_authority": False,
        }

    def _prepare_roles(
        self,
        *,
        current_track_id: str,
        successor_track_id: Optional[str],
        monotonic_ns: int,
    ) -> None:
        staged = self._staged
        if staged is None:
            raise DynamicCourseError(
                "dynamic guidance lacks a staged graph publication"
            )
        if staged.expected_current_track_id != current_track_id:
            raise DynamicCourseError(
                "dynamic staged current identity changed"
            )
        try:
            state = self.core.course_state()
        except DynamicCourseError:
            self.core.bind(
                current_gate_index=staged.expected_gate_index,
                current_track_id=current_track_id,
                successor_track_id=successor_track_id,
            )
            return
        if staged.adjacent_precredit:
            if not (
                staged.expected_gate_index == state.current_gate_index + 1
                and state.successor_track_id == current_track_id
            ):
                raise DynamicCourseError(
                    "adjacent guidance does not preserve the tracked successor"
                )
            return
        if staged.expected_gate_index == state.current_gate_index:
            if current_track_id != state.current_track_id:
                raise DynamicCourseError(
                    "same-gate dynamic current identity changed"
                )
            active = self._post_credit_successor_steering
            if active is not None:
                if (
                    active.to_gate_index
                    != staged.expected_gate_index
                    or active.reviewed_track_id != current_track_id
                    or active.promotion_count != state.promotion_count
                ):
                    raise DynamicCourseError(
                        "same-gate recovery differs from post-credit ownership"
                    )
            retained_successor = (
                successor_track_id
                if successor_track_id is not None
                else state.successor_track_id
            )
            self.core.bind(
                current_gate_index=state.current_gate_index,
                current_track_id=current_track_id,
                successor_track_id=retained_successor,
            )
            return
        if staged.expected_gate_index == state.current_gate_index + 1:
            if state.successor_track_id != current_track_id:
                raise DynamicCourseError(
                    "authoritative promotion lost dynamic successor lineage"
                )
            self.core.promote_authoritative(
                from_gate_index=state.current_gate_index,
                to_gate_index=staged.expected_gate_index,
                promoted_track_id=current_track_id,
                next_successor_track_id=successor_track_id,
                monotonic_ns=monotonic_ns,
            )
            return
        raise DynamicCourseError(
            "dynamic gate lifecycle is non-sequential"
        )

    def _successor_steering_targets(
        self,
        prediction: Any,
    ) -> Mapping[str, float]:
        """Map one bounded local successor prediction to steering targets."""

        target_roll = self.core.config.roll_guidance_sign * (
            self.core.config.roll_gain
            * prediction.stable_bearing_rad[0]
            + self.core.config.lateral_rate_gain
            * prediction.stable_bearing_rate_rad_s[0]
        )
        target_roll = min(
            MAX_TARGET_ROLL_RAD,
            max(-MAX_TARGET_ROLL_RAD, target_roll),
        )
        (
            target_pitch,
            camera_elevation,
            camera_elevation_rate,
            pitch_delay_lead,
        ) = _predicted_successor_pitch_reference(
            camera_center_y_norm=prediction.camera_center_norm[1],
            camera_center_rate_y_norm_s=(
                prediction.camera_center_rate_norm_s[1]
            ),
            vertical_angle_scale_rad=(
                self.core.config.vertical_angle_scale_rad
            ),
            pitch_command_delay_s=self.core.config.pitch_command_delay_s,
            maximum_lead_rad=(
                self.core.config.successor_prediction_max_extrapolation_rad
            ),
            baseline_pitch_rad=self.core.config.brake_pitch_rad,
        )
        camera_heading = math.atan(
            prediction.camera_center_norm[0]
            * self.core.config.horizontal_angle_scale_rad
        )
        yaw_rate = min(
            MAX_YAW_RATE_RAD_S,
            max(
                -MAX_YAW_RATE_RAD_S,
                -self.core.config.yaw_gain * camera_heading,
            ),
        )
        values = (
            target_roll,
            target_pitch,
            yaw_rate,
            SUPPORT_THRUST,
            camera_elevation,
            camera_elevation_rate,
            pitch_delay_lead,
        )
        if not all(math.isfinite(value) for value in values):
            raise DynamicCourseError(
                "successor steering produced non-finite authority"
            )
        return {
            "target_roll_rad": target_roll,
            "target_pitch_rad": target_pitch,
            "yaw_rate_rad_s": yaw_rate,
            "thrust": SUPPORT_THRUST,
            "camera_elevation_error_rad": camera_elevation,
            "camera_elevation_rate_rad_s": camera_elevation_rate,
            "pitch_delay_lead_rad": pitch_delay_lead,
        }

    def adjacent_precredit_successor_steering_authority(
        self,
        *,
        track_id: str,
        now_monotonic_ns: int,
    ) -> Mapping[str, Any]:
        """Steer toward one graph-vetted successor without promoting it."""

        if type(track_id) is not str or not track_id:
            raise DynamicCourseError(
                "precredit successor steering track is invalid"
            )
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise DynamicCourseError(
                "precredit successor steering clock is invalid"
            )
        staged = self._staged
        state = self.core.course_state()
        if (
            staged is None
            or not staged.adjacent_precredit
            or staged.expected_current_track_id != track_id
            or staged.expected_gate_index != state.current_gate_index + 1
            or state.current_track_id == track_id
            or not self.has_applied_command
        ):
            raise DynamicCourseError(
                "precredit successor steering lacks staged graph authority"
            )

        # RollingVisualApproachServo invokes this only after validating the
        # exact latest graph publication as one clean, stable, unambiguous
        # NEXT candidate.  Replace the stale pre-clipping image identity in
        # the local successor slot, but retain current gate ownership.
        if state.successor_track_id != track_id:
            if state.successor_track_id is not None:
                self.core.handoff_graph_vetted_successor_state(
                    predecessor_track_id=state.successor_track_id,
                    replacement_track_id=track_id,
                )
            state = self.core.bind(
                current_gate_index=state.current_gate_index,
                current_track_id=state.current_track_id,
                successor_track_id=track_id,
            )
        successor = state.successor
        if (
            successor is None
            or successor.track_id != track_id
            or successor.stream_generation
            != staged.camera_token.generation
            or successor.frame_sequence != staged.tracker_frame_sequence
            or successor.state_monotonic_ns
            != successor.last_measurement_monotonic_ns
            or not successor.visible
            or successor.ambiguous
            or successor.missed_count != 0
            or any(successor.censored_axes)
        ):
            raise DynamicCourseError(
                "precredit successor steering lacks exact clean state"
            )
        prediction = self.core.predict_track_steering(
            track_id,
            now_monotonic_ns,
        )
        if (
            prediction.stream_generation
            != successor.stream_generation
            or prediction.last_measurement_monotonic_ns
            != successor.last_measurement_monotonic_ns
            or prediction.measurement_age_s
            > self.core.config.successor_prediction_max_horizon_s
            + 1e-12
            or any(
                value
                > (
                    self.core.config
                    .successor_prediction_max_extrapolation_rad
                )
                + 1e-12
                for value in prediction.bearing_std_rad
            )
        ):
            raise DynamicCourseError(
                "precredit successor steering prediction is unavailable"
            )
        targets = dict(self._successor_steering_targets(prediction))
        fresh_horizontal_rate = (
            prediction.stable_bearing_rate_rad_s[0]
            if successor.bearing_rate_qualified[0]
            else 0.0
        )
        (
            proportional_roll,
            _unused_full_bank,
        ) = successor_roll_reference(
            stable_bearing_rad=prediction.stable_bearing_rad[0],
            stable_bearing_rate_rad_s=fresh_horizontal_rate,
            bearing_std_rad=prediction.bearing_std_rad[0],
            roll_guidance_sign=self.core.config.roll_guidance_sign,
            roll_gain=self.core.config.roll_gain,
            lateral_rate_gain=self.core.config.lateral_rate_gain,
            off_axis_brake_rad=self.core.config.off_axis_brake_rad,
            maximum_bearing_std_rad=(
                self.core.config
                .successor_prediction_max_extrapolation_rad
            ),
            full_bank_when_outward=False,
        )
        (
            target_roll,
            outward_full_bank,
        ) = successor_roll_reference(
            stable_bearing_rad=prediction.stable_bearing_rad[0],
            stable_bearing_rate_rad_s=fresh_horizontal_rate,
            bearing_std_rad=prediction.bearing_std_rad[0],
            roll_guidance_sign=self.core.config.roll_guidance_sign,
            roll_gain=self.core.config.roll_gain,
            lateral_rate_gain=self.core.config.lateral_rate_gain,
            off_axis_brake_rad=self.core.config.off_axis_brake_rad,
            maximum_bearing_std_rad=(
                self.core.config
                .successor_prediction_max_extrapolation_rad
            ),
            full_bank_when_outward=(
                successor.bearing_rate_qualified[0]
            ),
        )
        targets["target_roll_rad"] = target_roll
        # Every adjacent proposal is derived from this exact staged graph
        # publication. Accepted predecessor targets are deliberately ignored,
        # and promotion cannot retain this pre-credit bank.
        self._pending_precredit_successor_roll_reference = None
        self._precredit_successor_roll_reference = None
        self._last_decision = None
        targets.update(
            {
                "basis": (
                    "graph-vetted-precredit-successor-steering-v1"
                ),
                "from_gate_index": state.current_gate_index,
                "to_gate_index": staged.expected_gate_index,
                "steering_track_id": track_id,
                "stream_generation": successor.stream_generation,
                "tracker_frame_sequence": successor.frame_sequence,
                "last_measurement_monotonic_ns": (
                    successor.last_measurement_monotonic_ns
                ),
                "authority_monotonic_ns": now_monotonic_ns,
                "measurement_age_s": prediction.measurement_age_s,
                "unconstrained_target_roll_rad": proportional_roll,
                "retained_roll_reference_applied": False,
                "outward_full_bank_applied": outward_full_bank,
                "stable_bearing_rad": list(
                    prediction.stable_bearing_rad
                ),
                "stable_bearing_rate_rad_s": list(
                    prediction.stable_bearing_rate_rad_s
                ),
                "camera_center_norm": list(
                    prediction.camera_center_norm
                ),
                "camera_center_rate_norm_s": list(
                    prediction.camera_center_rate_norm_s
                ),
                "bearing_std_rad": list(prediction.bearing_std_rad),
                "steering_only": True,
                "passage_authority": False,
                "advance_authority": False,
            }
        )
        return targets

    def guide(
        self,
        *,
        current_track_id: str,
        successor_track_id: Optional[str],
        monotonic_ns: int,
    ) -> Optional[GuidanceDecision]:
        self._prepare_roles(
            current_track_id=current_track_id,
            successor_track_id=successor_track_id,
            monotonic_ns=monotonic_ns,
        )
        if not self.has_applied_command:
            self._last_decision = None
            return None
        staged = self._staged
        if staged is None:
            raise DynamicCourseError(
                "dynamic guidance lacks a staged graph publication"
            )
        decision = self.core.guide(
            monotonic_ns,
            passage_committed=staged.passage_committed,
        )
        decision = self._apply_post_credit_rebound_roll_reference(
            decision
        )
        decision = self._apply_post_credit_roll_reference_handoff(
            decision
        )
        self._last_decision = decision
        return decision

    def _apply_post_credit_rebound_roll_reference(
        self,
        decision: GuidanceDecision,
    ) -> GuidanceDecision:
        """Consume fresh rebound steering until an accepted reference exists.

        A graph-proven cross-ID reacquisition creates a bounded steering-only
        lease, but normal guidance previously ignored it and the lifecycle
        discarded it after two clean frames.  Admit only its horizontal
        attitude reference while the qualified promoted-gate residual is
        moving outward.  Exact wire acceptance then arms the existing
        geometry-released handoff; no temporal command continuity is added.
        """

        lease = self._post_credit_successor_steering
        if (
            lease is None
            or not lease.steering_available
            or self._post_credit_roll_reference_handoff is not None
        ):
            return decision
        state = self.core.course_state()
        current = state.current
        lineage_matches = bool(
            decision.current_gate_index == lease.to_gate_index
            and decision.current_track_id == lease.reviewed_track_id
            and not decision.passage_committed
            and state.current_gate_index == lease.to_gate_index
            and state.current_track_id == lease.reviewed_track_id
            and current.track_id == lease.reviewed_track_id
            and current.stream_generation == lease.stream_generation
            and state.promotion_count == lease.promotion_count
        )
        if not lineage_matches:
            self._pending_post_credit_roll_reference = None
            return decision
        try:
            authority = self.post_credit_successor_steering_authority(
                now_monotonic_ns=decision.monotonic_ns,
            )
        except PostCreditSuccessorSteeringUnavailable:
            self._pending_post_credit_roll_reference = None
            return decision

        normal_roll = float(decision.command.target_roll_rad)
        rebound_roll = float(authority["target_roll_rad"])
        stable_error = float(decision.current_center_norm[0])
        residual_rate = float(
            current.residual_translational_rate_rad_s[0]
        )
        guidance_sign = float(self.core.config.roll_guidance_sign)
        values = (
            normal_roll,
            rebound_roll,
            stable_error,
            residual_rate,
            guidance_sign,
        )
        bounded = bool(
            all(math.isfinite(value) for value in values)
            and abs(rebound_roll) <= MAX_TARGET_ROLL_RAD + 1e-12
            and current.visible
            and not current.ambiguous
            and not current.censored_axes[0]
            and current.bearing_rate_qualified[0]
            and current.bearing_std_rad[0]
            <= (
                self.core.config
                .successor_prediction_max_extrapolation_rad
            )
            + 1e-12
            and abs(guidance_sign) > 1e-12
        )
        outward = stable_error * residual_rate > 1e-12
        corrective = (
            rebound_roll * guidance_sign * stable_error > 1e-12
        )
        compatible = normal_roll * rebound_roll >= -1e-12
        stronger = abs(rebound_roll) > abs(normal_roll) + 1e-12
        equal_reference = math.isclose(
            rebound_roll,
            normal_roll,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if not (
            bounded
            and outward
            and corrective
            and compatible
            and (stronger or equal_reference)
        ):
            self._pending_post_credit_roll_reference = None
            return decision
        if equal_reference:
            # The ordinary law already requests the exact bounded rebound,
            # so leave this proposal unchanged while preserving the pending
            # reference for accepted-wire admission.  A later transient rate
            # reversal must not erase the useful saturated attitude target.
            return decision
        return replace(
            decision,
            command=replace(
                decision.command,
                target_roll_rad=rebound_roll,
            ),
        )

    def _apply_post_credit_roll_reference_handoff(
        self,
        decision: GuidanceDecision,
    ) -> GuidanceDecision:
        """Prevent an outward promotion from unwinding helpful successor bank.

        This is a gate-relative reference constraint, not a command slew.  It
        retains one already accepted, bounded successor attitude target while
        the promoted current error still requires the same correction.  A
        short unqualified-rate gap retains that reference only through the
        reviewed successor horizon.  A qualified recovering-rate sample
        releases it immediately.  Once fresh promoted-current geometry proves
        that the lateral error is already closing, retaining a larger
        pre-credit bank would fight the current controller and create an
        avoidable overshoot.
        """

        handoff = self._post_credit_roll_reference_handoff
        if handoff is None:
            return decision
        if decision.monotonic_ns > handoff.expires_monotonic_ns:
            self._post_credit_roll_reference_handoff = None
            return decision
        state = self.core.course_state()
        current = state.current
        normal_roll = float(decision.command.target_roll_rad)
        retained_roll = float(handoff.retained_target_roll_rad)
        residual_rate = float(
            current.residual_translational_rate_rad_s[0]
        )
        stable_error = float(decision.current_center_norm[0])
        stable_bearing_rad = math.atan(
            stable_error
            * self.core.config.horizontal_angle_scale_rad
        )
        guidance_sign = float(self.core.config.roll_guidance_sign)
        lineage_matches = bool(
            decision.current_gate_index == handoff.to_gate_index
            and decision.current_track_id == handoff.track_id
            and state.current_gate_index == handoff.to_gate_index
            and state.current_track_id == handoff.track_id
            and current.track_id == handoff.track_id
            and current.stream_generation == handoff.stream_generation
            and state.promotion_count == handoff.promotion_count
        )
        values = (
            normal_roll,
            retained_roll,
            residual_rate,
            stable_error,
            stable_bearing_rad,
            guidance_sign,
        )
        bounded_state = bool(
            lineage_matches
            and all(math.isfinite(value) for value in values)
            and 0.0 < abs(retained_roll) <= MAX_TARGET_ROLL_RAD
            and not current.ambiguous
            and current.bearing_std_rad[0]
            <= (
                self.core.config
                .successor_prediction_max_extrapolation_rad
            )
            + 1e-12
            and abs(guidance_sign) > 1e-12
        )
        direction = 1.0 if retained_roll > 0.0 else -1.0
        same_corrective_demand = bool(
            direction * normal_roll > 1e-12
        )
        error_still_requires_correction = bool(
            direction * guidance_sign * stable_error > 1e-12
        )
        qualified_recovery = bool(
            current.bearing_rate_qualified[0]
            and direction * guidance_sign * residual_rate <= 1e-12
        )
        if (
            not bounded_state
            or not same_corrective_demand
            or not error_still_requires_correction
            or qualified_recovery
        ):
            self._post_credit_roll_reference_handoff = None
            return decision
        constrained = math.copysign(
            max(abs(normal_roll), abs(retained_roll)),
            retained_roll,
        )
        if (
            not math.isfinite(constrained)
            or abs(constrained) > MAX_TARGET_ROLL_RAD + 1e-12
        ):
            raise DynamicCourseError(
                "post-credit roll reference left the bounded envelope"
            )
        return replace(
            decision,
            command=replace(
                decision.command,
                target_roll_rad=constrained,
            ),
        )

    def propagated_current_fov_gap_authority(
        self,
        *,
        track: VisualTrack,
        camera_token: CameraFrameToken,
        now_monotonic_ns: int,
        allow_tracking_only_inner_raw_clipping: bool = False,
    ) -> Mapping[str, Any]:
        """Prove exact, bounded steering ownership of a clipped FOV gap.

        This is a read-only authority check over the already-staged tracker
        publication, rolling local state, and latest guidance decision.  It
        cannot create passage or authoritative gate-advance evidence.
        """

        if type(track) is not VisualTrack:
            raise DynamicCourseError(
                "propagated FOV gap requires an exact visual track"
            )
        if type(camera_token) is not CameraFrameToken:
            raise DynamicCourseError(
                "propagated FOV gap requires an exact camera token"
            )
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise DynamicCourseError(
                "propagated FOV gap clock is invalid"
            )
        if type(allow_tracking_only_inner_raw_clipping) is not bool:
            raise DynamicCourseError(
                "propagated FOV gap clipping selection is invalid"
            )
        if (
            tuple(self.core.config.camera_to_body_wxyz)
            != BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ
        ):
            raise DynamicCourseError(
                "propagated FOV gap lacks the calibrated camera boundary"
            )
        staged = self._staged
        if staged is None or not track.history:
            raise DynamicCourseError(
                "propagated FOV gap lacks exact staged lineage"
            )
        sample = track.history[-1]
        if (
            staged.expected_current_track_id != track.track_id
            or staged.camera_token != camera_token
            or staged.tracker_frame_sequence
            != sample.tracker_frame_sequence
            or track.latest_token != camera_token
            or sample.token != camera_token
            or type(sample.publication_monotonic_ns) is not int
            or now_monotonic_ns < sample.observation_monotonic_ns
            or now_monotonic_ns < sample.publication_monotonic_ns
        ):
            raise DynamicCourseError(
                "propagated FOV gap tracker publication differs"
            )

        try:
            course = self.core.course_state()
        except DynamicCourseError as exc:
            raise DynamicCourseError(
                "propagated FOV gap lacks current dynamic state"
            ) from exc
        current = course.current
        decision = self._last_decision
        if (
            decision is None
            or decision.monotonic_ns > now_monotonic_ns
            or decision.current_gate_index != course.current_gate_index
            or decision.current_track_id != course.current_track_id
            or decision.current_track_id != track.track_id
            or staged.expected_gate_index != course.current_gate_index
            or current.track_id != track.track_id
            or current.frame_sequence != sample.tracker_frame_sequence
            or current.stream_generation != camera_token.generation
        ):
            raise DynamicCourseError(
                "propagated FOV gap decision and current state differ"
            )

        state_aperture = current.aperture_half_size_norm
        decision_aperture = decision.current_aperture_half_size_norm
        seed_ns = current.aperture_seed_monotonic_ns
        deadline_ns = current.aperture_prediction_deadline_monotonic_ns
        if (
            state_aperture is None
            or decision_aperture is None
            or seed_ns is None
            or deadline_ns is None
            or not current.aperture_propagated
            or not decision.current_aperture_propagated
            or seed_ns > current.state_monotonic_ns
            or deadline_ns <= seed_ns
            or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in (*state_aperture, *decision_aperture)
            )
        ):
            raise DynamicCourseError(
                "propagated FOV gap lacks a clean propagated aperture"
            )

        # The state aperture is retained in the last clean camera orientation,
        # while the decision aperture is reprojected into the current track's
        # fixed passage orientation.  They describe the same gate but are not
        # numerically identical under pitch/roll motion.  Reproject the local
        # state once more into the current camera orientation for raw-FOV
        # observability; validate each representation independently rather
        # than using cross-coordinate tuple equality as authority.
        camera_center, camera_aperture = self.core._decision_geometry(
            current.track_id,
            current.state_monotonic_ns,
        )
        config = self.core.config
        horizontal_scale = float(config.horizontal_angle_scale_rad)
        vertical_scale = float(config.vertical_angle_scale_rad)
        camera_center_std_norm = (
            float(current.bearing_std_rad[0]) / horizontal_scale,
            float(current.bearing_std_rad[1]) / vertical_scale,
        )
        quaternion = current.body_to_reference_wxyz
        if (
            camera_aperture is None
            or type(quaternion) is not tuple
            or len(quaternion) != 4
            or not all(
                math.isfinite(float(value))
                for value in (
                    *camera_center,
                    *camera_aperture,
                    *camera_center_std_norm,
                    current.log_scale_std,
                    *quaternion,
                    horizontal_scale,
                    vertical_scale,
                )
            )
            or any(float(value) <= 0.0 for value in camera_aperture)
            or any(float(value) < 0.0 for value in camera_center_std_norm)
            or float(current.log_scale_std) < 0.0
            or horizontal_scale <= 0.0
            or vertical_scale <= 0.0
        ):
            raise DynamicCourseError(
                "propagated FOV gap camera projection is invalid"
            )

        expected_decision_age_s = max(
            0.0,
            (decision.monotonic_ns - seed_ns) / 1_000_000_000.0,
        )
        expected_decision_horizon_s = max(
            0.0,
            (deadline_ns - decision.monotonic_ns) / 1_000_000_000.0,
        )
        if (
            not math.isfinite(
                decision.current_aperture_prediction_age_s
            )
            or not math.isfinite(
                decision
                .current_aperture_prediction_horizon_remaining_s
            )
            or not math.isclose(
                decision.current_aperture_prediction_age_s,
                expected_decision_age_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                decision
                .current_aperture_prediction_horizon_remaining_s,
                expected_decision_horizon_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise DynamicCourseError(
                "propagated FOV gap decision horizon differs"
            )

        if (
            not current.visible
            or current.ambiguous
            or not track.visible
            or track.ambiguous
        ):
            raise DynamicCourseError(
                "propagated FOV gap current track is not unambiguous"
            )
        raw_clipping = (
            staged.current_raw_clipping
            if allow_tracking_only_inner_raw_clipping
            else current.clipping
        )
        if (
            raw_clipping != track.clipping
            or raw_clipping != sample.clipping
            or raw_clipping == FrameEdge.NONE
        ):
            raise DynamicCourseError(
                "propagated FOV gap publication is not clipped"
            )

        remaining_horizon_s = (
            deadline_ns - now_monotonic_ns
        ) / 1_000_000_000.0
        if (
            not math.isfinite(remaining_horizon_s)
            or remaining_horizon_s <= 0.0
        ):
            raise DynamicCourseError(
                "propagated FOV gap aperture prediction expired"
            )
        return {
            "basis": "propagated-current-fov-gap-steering-v1",
            "gate_index": course.current_gate_index,
            "track_id": track.track_id,
            "camera_token": asdict(camera_token),
            "tracker_frame_sequence": sample.tracker_frame_sequence,
            "publication_monotonic_ns": sample.publication_monotonic_ns,
            "authority_monotonic_ns": now_monotonic_ns,
            "stream_generation": current.stream_generation,
            "aperture_seed_monotonic_ns": seed_ns,
            "aperture_prediction_deadline_monotonic_ns": deadline_ns,
            "aperture_prediction_horizon_remaining_s": (
                remaining_horizon_s
            ),
            "aperture_half_size_norm": list(decision_aperture),
            "state_aperture_half_size_norm": list(state_aperture),
            "camera_center_norm": list(camera_center),
            "camera_aperture_half_size_norm": list(camera_aperture),
            "camera_center_std_norm": list(camera_center_std_norm),
            "aperture_log_scale_std": float(current.log_scale_std),
            "body_to_reference_wxyz": list(quaternion),
            "horizontal_angle_scale_rad": horizontal_scale,
            "vertical_angle_scale_rad": vertical_scale,
            "clipping": int(raw_clipping),
            "terminal_crossing_clearance_norm": list(
                decision.terminal_crossing_clearance_norm
            ),
            "steering_only": True,
            "passage_authority": False,
            "advance_authority": False,
        }

    def propagated_current_visibility_gap_authority(
        self,
        *,
        track: VisualTrack,
        camera_token: CameraFrameToken,
        now_monotonic_ns: int,
    ) -> Mapping[str, Any]:
        """Continue accepted same-gate steering across exact fresh misses.

        The rolling graph keeps authoritative identity while independently
        withholding visual-measurement authority.  Each exact newer camera
        publication proves the stream is live, so a clipped-edge miss may
        continue the last bounded same-gate command that actually reached the
        wire.  No bearing, aperture, scale, TTC, passage, race, or advance
        geometry is renewed through blindness.
        """

        if type(track) is not VisualTrack:
            raise DynamicCourseError(
                "propagated visibility gap requires an exact visual track"
            )
        if type(camera_token) is not CameraFrameToken:
            raise DynamicCourseError(
                "propagated visibility gap requires an exact camera token"
            )
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise DynamicCourseError(
                "propagated visibility gap clock is invalid"
            )
        staged = self._staged
        if staged is None or not track.history:
            raise DynamicCourseError(
                "propagated visibility gap lacks exact staged lineage"
            )
        sample = track.history[-1]
        missed_count = track.missed_frame_count
        if (
            staged.expected_current_track_id != track.track_id
            or staged.camera_token != camera_token
            or staged.tracker_frame_sequence <= sample.tracker_frame_sequence
            or staged.adjacent_precredit
            or track.role is not VisualTrackRole.CURRENT
            or track.authoritative_gate_index != staged.expected_gate_index
            or track.visible
            or track.ambiguous
            or type(missed_count) is not int
            or missed_count <= 0
            or sample.token != track.latest_token
            or sample.clipping == FrameEdge.NONE
            or track.clipping != sample.clipping
            or type(sample.observation_monotonic_ns) is not int
            or sample.observation_monotonic_ns < 0
            or camera_token.stream_id is None
            or camera_token.stream_id != sample.token.stream_id
            or camera_token.generation != sample.token.generation
            or camera_token.publication_sequence is None
            or sample.token.publication_sequence is None
            or camera_token.publication_sequence
            <= sample.token.publication_sequence
            or camera_token.publication_sequence
            - sample.token.publication_sequence
            < missed_count
            or staged.tracker_frame_sequence
            - sample.tracker_frame_sequence
            < missed_count
        ):
            raise DynamicCourseError(
                "propagated visibility gap is not an exact clipped miss"
            )

        state = self.core.course_state()
        current = state.current
        expected_last_measurement_ns = (
            sample.observation_monotonic_ns
            - round(
                self.core.config.camera_delay_s
                * 1_000_000_000.0
            )
        )
        anchor = self._same_gate_steering_anchor
        bearing_prediction_seed_ns = current.last_measurement_monotonic_ns
        # Aperture/scale authority still expires from its measured seed.  It
        # must never own passage or clearance after that deadline.  A fresh
        # blank publication proves only that the receiver is live; it cannot
        # renew geometry or the accepted command.  Pin steering expiry to the
        # last visible command that actually reached the wire.
        aperture_steering_deadline_ns = (
            current.aperture_prediction_deadline_monotonic_ns
            if (
                current.aperture_propagated
                and current.aperture_half_size_norm is not None
                and current.aperture_seed_monotonic_ns is not None
                and current
                .aperture_prediction_deadline_monotonic_ns
                is not None
            )
            else None
        )
        fixed_visibility_gap_horizon_s = (
            self.core.config.post_credit_current_prediction_max_horizon_s
        )
        steering_deadline_basis = (
            "accepted-wire-same-gate-steering-anchor-v2"
        )
        if (
            state.current_gate_index != staged.expected_gate_index
            or state.current_track_id != track.track_id
            or current.track_id != track.track_id
            or current.stream_generation != camera_token.generation
            or current.frame_sequence != staged.tracker_frame_sequence
            or current.last_measurement_monotonic_ns
            != expected_last_measurement_ns
            or current.visible
            or current.ambiguous
            or current.missed_count != missed_count
            or anchor is None
            or anchor.gate_index != state.current_gate_index
            or anchor.track_id != state.current_track_id
            or anchor.stream_generation != current.stream_generation
            or anchor.camera_token != sample.token
            or anchor.camera_token.stream_id != camera_token.stream_id
            or anchor.wire_start_monotonic_ns
            < current.last_measurement_monotonic_ns
            or anchor.wire_start_monotonic_ns > now_monotonic_ns
        ):
            raise PropagatedCurrentVisibilityGapUnavailable(
                "propagated visibility gap lacks fresh local steering state"
            )

        steering_deadline_ns = anchor.wire_start_monotonic_ns + round(
            fixed_visibility_gap_horizon_s * 1_000_000_000.0
        )
        decision = anchor.decision
        command = replace(
            decision.command,
            # Never carry forward/negative pitch through blindness.  This
            # branch is reacquisition steering, not approach closure.
            target_pitch_rad=max(
                0.0,
                float(decision.command.target_pitch_rad),
            ),
        )
        remaining_horizon_s = (
            steering_deadline_ns - now_monotonic_ns
        ) / 1_000_000_000.0
        # The anchor owns only bounded attitude/yaw/thrust steering.  Never
        # carry aperture, scale, TTC, passage, or advance geometry through
        # blindness.
        decision_aperture = None
        maximum_bearing_std_rad = (
            self.core.config.max_abs_bearing_rad
        )
        if (
            remaining_horizon_s <= 0.0
            or any(
                value > maximum_bearing_std_rad + 1e-12
                for value in decision.current_bearing_std_rad
            )
        ):
            # Expired local geometry is a normal navigation state.  The stage
            # will retire propagation and continue with fresh-publication
            # yaw-only search; it must not abort the course.
            raise PropagatedCurrentVisibilityGapUnavailable(
                "propagated visibility gap exhausted local steering state"
            )
        if (
            decision.current_gate_index != state.current_gate_index
            or decision.current_track_id != state.current_track_id
            or not all(
                math.isfinite(float(value))
                for value in (
                    *decision.current_center_norm,
                    *decision.current_bearing_std_rad,
                    command.target_roll_rad,
                    command.target_pitch_rad,
                    command.yaw_rate_rad_s,
                    command.thrust,
                )
            )
            or (
                decision_aperture is not None
                and not all(
                    math.isfinite(float(value)) and float(value) > 0.0
                    for value in decision_aperture
                )
            )
        ):
            raise DynamicCourseError(
                "propagated visibility gap guidance is invalid"
            )
        return {
            "basis": "propagated-current-visibility-gap-guidance-v2",
            "gate_index": state.current_gate_index,
            "track_id": state.current_track_id,
            "camera_token": asdict(camera_token),
            "last_visible_camera_token": asdict(sample.token),
            "steering_anchor_camera_token": asdict(
                anchor.camera_token
            ),
            "steering_anchor_wire_start_monotonic_ns": (
                anchor.wire_start_monotonic_ns
            ),
            "tracker_frame_sequence": staged.tracker_frame_sequence,
            "last_visible_tracker_frame_sequence": (
                sample.tracker_frame_sequence
            ),
            "missed_frame_count": missed_count,
            "last_visible_clipping": int(sample.clipping),
            "guidance_monotonic_ns": decision.monotonic_ns,
            "last_measurement_monotonic_ns": (
                current.last_measurement_monotonic_ns
            ),
            "steering_prediction_deadline_monotonic_ns": (
                steering_deadline_ns
            ),
            "steering_prediction_deadline_basis": (
                steering_deadline_basis
            ),
            "fallback_steering_deadline_monotonic_ns": (
                steering_deadline_ns
            ),
            "bearing_prediction_seed_monotonic_ns": (
                bearing_prediction_seed_ns
            ),
            "bearing_prediction_deadline_monotonic_ns": (
                steering_deadline_ns
            ),
            "aperture_prediction_deadline_monotonic_ns": (
                aperture_steering_deadline_ns
            ),
            "steering_prediction_horizon_remaining_s": (
                remaining_horizon_s
            ),
            "current_center_norm": list(decision.current_center_norm),
            "current_aperture_half_size_norm": (
                None
                if decision_aperture is None
                else list(decision_aperture)
            ),
            "current_aperture_propagated": (
                False
            ),
            "current_aperture_dynamics_qualified": (
                False
            ),
            "current_bearing_std_rad": list(
                decision.current_bearing_std_rad
            ),
            "command": asdict(command),
            "steering_only": True,
            "passage_authority": False,
            "advance_authority": False,
        }

    def govern_wire_command(
        self,
        command: AttitudeRateCommand,
        *,
        proposal_monotonic_ns: int,
        launch_thrust_override: bool,
        yaw_safety_override: bool,
    ) -> AttitudeRateCommand:
        return self._wire_governor.preview(
            command,
            proposal_monotonic_ns,
            launch_thrust_override=launch_thrust_override,
            yaw_safety_override=yaw_safety_override,
        )

    def record_wire_acceptance(
        self,
        *,
        target_roll_rad: float,
        target_pitch_rad: float,
        yaw_rate_rad_s: float,
        thrust: float,
        wire_command: AttitudeRateCommand,
        wire_start_monotonic_ns: int,
        requested_thrust: Optional[float] = None,
        thrust_slew_override: bool = False,
        yaw_slew_override: bool = False,
        same_gate_steering_anchor_authorized: bool = True,
    ) -> Mapping[str, Any]:
        requested_wire_thrust = (
            float(thrust)
            if requested_thrust is None
            else float(requested_thrust)
        )
        if type(same_gate_steering_anchor_authorized) is not bool:
            raise TypeError(
                "same-gate steering-anchor authority must be an exact bool"
            )
        if (
            not math.isfinite(requested_wire_thrust)
            or requested_wire_thrust < MIN_THRUST
            or requested_wire_thrust > MAX_THRUST
        ):
            raise DynamicCourseError(
                "requested wire thrust is outside the production envelope"
            )
        discontinuity_axes = (
            ((2,) if yaw_slew_override else ())
            + ((3,) if thrust_slew_override else ())
        )
        applied_sample = AppliedCommandSample(
            monotonic_ns=wire_start_monotonic_ns,
            target_roll_rad=target_roll_rad,
            target_pitch_rad=target_pitch_rad,
            yaw_rate_rad_s=yaw_rate_rad_s,
            thrust=thrust,
            roll_rate_rad_s=float(wire_command.roll_rate),
            pitch_rate_rad_s=float(wire_command.pitch_rate),
            host_clock_id=_HOST_CLOCK_ID,
        )
        self.core.record_applied_command(applied_sample)
        self._wire_governor.commit(
            wire_command,
            wire_start_monotonic_ns,
            discontinuity_axes=discontinuity_axes,
        )
        self._last_applied_sample = applied_sample
        staged = self._staged
        decision = self._last_decision
        if staged is not None and decision is not None:
            state = self.core.course_state()
            current = state.current
            exact_visible_same_gate = bool(
                not staged.adjacent_precredit
                and decision.current_gate_index == state.current_gate_index
                and decision.current_track_id == state.current_track_id
                and staged.expected_gate_index == state.current_gate_index
                and staged.expected_current_track_id == state.current_track_id
                and staged.camera_token.generation
                == current.stream_generation
                and staged.tracker_frame_sequence == current.frame_sequence
                and current.state_monotonic_ns
                == current.last_measurement_monotonic_ns
                and current.visible
                and not current.ambiguous
                and current.missed_count == 0
                and not decision.passage_committed
                and decision.monotonic_ns <= wire_start_monotonic_ns
            )
            if exact_visible_same_gate and same_gate_steering_anchor_authorized:
                accepted_anchor_decision = replace(
                    decision,
                    command=replace(
                        decision.command,
                        target_roll_rad=float(target_roll_rad),
                        target_pitch_rad=float(target_pitch_rad),
                        yaw_rate_rad_s=float(yaw_rate_rad_s),
                        thrust=float(thrust),
                    ),
                )
                self._same_gate_steering_anchor = (
                    _SameGateSteeringAnchor(
                        gate_index=state.current_gate_index,
                        track_id=state.current_track_id,
                        stream_generation=current.stream_generation,
                        camera_token=staged.camera_token,
                        wire_start_monotonic_ns=wire_start_monotonic_ns,
                        decision=accepted_anchor_decision,
                    )
                )
        pending_roll = self._pending_precredit_successor_roll_reference
        if pending_roll is not None:
            self._pending_precredit_successor_roll_reference = None
            if (
                pending_roll.authority_monotonic_ns
                <= wire_start_monotonic_ns
                and math.isclose(
                    pending_roll.target_roll_rad,
                    applied_sample.target_roll_rad,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                self._precredit_successor_roll_reference = replace(
                    pending_roll,
                    accepted_wire_start_monotonic_ns=(
                        wire_start_monotonic_ns
                    ),
                )
        pending_post_credit = self._pending_post_credit_roll_reference
        if pending_post_credit is not None:
            lease = self._post_credit_successor_steering
            state = self.core.course_state()
            accepted_post_credit_reference = bool(
                lease is not None
                and lease.steering_available
                and lease.to_gate_index
                == pending_post_credit.to_gate_index
                and lease.reviewed_track_id
                == pending_post_credit.track_id
                and lease.stream_generation
                == pending_post_credit.stream_generation
                and lease.promotion_count
                == pending_post_credit.promotion_count
                and state.current_gate_index
                == pending_post_credit.to_gate_index
                and state.current_track_id
                == pending_post_credit.track_id
                and state.current.stream_generation
                == pending_post_credit.stream_generation
                and pending_post_credit.authority_monotonic_ns
                <= wire_start_monotonic_ns
                <= pending_post_credit.expires_monotonic_ns
                and math.isclose(
                    pending_post_credit.target_roll_rad,
                    applied_sample.target_roll_rad,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            self._pending_post_credit_roll_reference = None
            if (
                accepted_post_credit_reference
                and self._post_credit_roll_reference_handoff is None
            ):
                self._post_credit_roll_reference_handoff = (
                    _PostCreditRollReferenceHandoff(
                        authority_basis=(
                            "accepted-post-credit-successor-roll-reference-v1"
                        ),
                        to_gate_index=pending_post_credit.to_gate_index,
                        track_id=pending_post_credit.track_id,
                        stream_generation=(
                            pending_post_credit.stream_generation
                        ),
                        promotion_count=pending_post_credit.promotion_count,
                        retained_target_roll_rad=(
                            pending_post_credit.target_roll_rad
                        ),
                        source_authority_monotonic_ns=(
                            pending_post_credit.authority_monotonic_ns
                        ),
                        source_wire_start_monotonic_ns=(
                            wire_start_monotonic_ns
                        ),
                        expires_monotonic_ns=(
                            pending_post_credit.expires_monotonic_ns
                        ),
                    )
                )
        self._applied_command_count += 1
        decision = self._last_decision
        if decision is not None:
            self._dynamic_command_count += 1
        roll = float(wire_command.roll_rate)
        sign = 1 if roll > 0.01 else (-1 if roll < -0.01 else 0)
        if (
            sign != 0
            and self._last_nonzero_roll_sign != 0
            and sign != self._last_nonzero_roll_sign
        ):
            self._roll_reversal_count += 1
        if sign != 0:
            self._last_nonzero_roll_sign = sign
        evidence: dict[str, Any] = {
            "schema": "aigp-vq2-dynamic-command/1",
            "controller_family": DYNAMIC_CONTROLLER_FAMILY,
            "applied_command_count": self._applied_command_count,
            "dynamic_command_count": self._dynamic_command_count,
            "roll_reversal_count": self._roll_reversal_count,
            "wire_start_monotonic_ns": wire_start_monotonic_ns,
            "requested_thrust_before_wire_governor": (
                requested_wire_thrust
            ),
            "target_attitude_yaw_thrust": [
                target_roll_rad,
                target_pitch_rad,
                yaw_rate_rad_s,
                thrust,
            ],
            "wire_command": asdict(wire_command),
        }
        if decision is not None:
            course = self.core.course_state()
            roll_handoff = self._post_credit_roll_reference_handoff
            thrust_settle_s = (
                abs(
                    float(wire_command.thrust)
                    - requested_wire_thrust
                )
                / self._wire_governor.config.max_thrust_slew_s
            )
            post_governor_contact_budget_s = (
                None
                if decision.current_time_to_contact_s is None
                else (
                    decision.current_time_to_contact_s
                    - self.core.config.thrust_command_delay_s
                    - thrust_settle_s
                )
            )
            evidence.update(
                {
                    "gate_index": decision.current_gate_index,
                    "current_track_id": decision.current_track_id,
                    "successor_track_id": decision.successor_track_id,
                    "unconstrained_target_roll_rad": (
                        decision.proposed_command.target_roll_rad
                    ),
                    "post_credit_roll_reference_handoff": (
                        None
                        if roll_handoff is None
                        else {
                            "basis": roll_handoff.authority_basis,
                            "to_gate_index": (
                                roll_handoff.to_gate_index
                            ),
                            "track_id": roll_handoff.track_id,
                            "stream_generation": (
                                roll_handoff.stream_generation
                            ),
                            "promotion_count": (
                                roll_handoff.promotion_count
                            ),
                            "retained_target_roll_rad": (
                                roll_handoff.retained_target_roll_rad
                            ),
                            "source_authority_monotonic_ns": (
                                roll_handoff
                                .source_authority_monotonic_ns
                            ),
                            "source_wire_start_monotonic_ns": (
                                roll_handoff
                                .source_wire_start_monotonic_ns
                            ),
                            "expires_monotonic_ns": (
                                roll_handoff.expires_monotonic_ns
                            ),
                            "steering_only": True,
                            "passage_authority": False,
                            "advance_authority": False,
                        }
                    ),
                    "current_center_norm": list(
                        decision.current_center_norm
                    ),
                    "camera_current_center_norm": list(
                        decision.camera_current_center_norm
                    ),
                    "current_aperture_half_size_norm": (
                        None
                        if decision.current_aperture_half_size_norm is None
                        else list(
                            decision.current_aperture_half_size_norm
                        )
                    ),
                    "current_aperture_propagated": (
                        decision.current_aperture_propagated
                    ),
                    "current_aperture_dynamics_qualified": (
                        decision.current_aperture_dynamics_qualified
                    ),
                    "current_aperture_prediction_age_s": (
                        decision.current_aperture_prediction_age_s
                    ),
                    "current_aperture_prediction_horizon_remaining_s": (
                        decision
                        .current_aperture_prediction_horizon_remaining_s
                    ),
                    "passage_point_norm": list(
                        decision.passage_point_norm
                    ),
                    "successor_passage_authority": (
                        decision.successor_passage_authority
                    ),
                    "centered_crossing_clearance_norm": list(
                        decision.centered_crossing_clearance_norm
                    ),
                    "successor_clearance_dwell_s": (
                        decision.successor_clearance_dwell_s
                    ),
                    "successor_clearance_authority": (
                        decision.successor_clearance_authority
                    ),
                    "passage_error_norm": list(
                        decision.passage_error_norm
                    ),
                    "aperture_margin_norm": list(
                        decision.aperture_margin_norm
                    ),
                    "crossing_prediction_horizon_s": (
                        decision.crossing_prediction_horizon_s
                    ),
                    "crossing_coordinate_basis": (
                        DYNAMIC_CROSSING_COORDINATE_BASIS
                    ),
                    "current_crossing_error_q": list(
                        decision.current_crossing_error_q
                    ),
                    "crossing_rate_q_s": list(
                        decision.crossing_rate_q_s
                    ),
                    "predicted_crossing_error_norm": list(
                        decision.predicted_crossing_error_norm
                    ),
                    "predicted_crossing_std_norm": list(
                        decision.predicted_crossing_std_norm
                    ),
                    "crossing_allowance_norm": list(
                        decision.crossing_allowance_norm
                    ),
                    "crossing_swept_occupancy_norm": list(
                        decision.crossing_swept_occupancy_norm
                    ),
                    "predicted_crossing_clearance_norm": list(
                        decision.predicted_crossing_clearance_norm
                    ),
                    "terminal_crossing_occupancy_norm": list(
                        decision.terminal_crossing_occupancy_norm
                    ),
                    "terminal_crossing_clearance_norm": list(
                        decision.terminal_crossing_clearance_norm
                    ),
                    "post_governor_thrust_settle_s": thrust_settle_s,
                    "post_governor_contact_budget_s": (
                        post_governor_contact_budget_s
                    ),
                    "current_bearing_std_rad": list(
                        decision.current_bearing_std_rad
                    ),
                    "current_bearing_std_norm": [
                        decision.current_bearing_std_rad[0]
                        / self.core.config.horizontal_angle_scale_rad,
                        decision.current_bearing_std_rad[1]
                        / self.core.config.vertical_angle_scale_rad,
                    ],
                    "successor_weight": decision.successor_weight,
                    "successor_bearing_std_rad": (
                        None
                        if decision.successor_bearing_std_rad is None
                        else list(decision.successor_bearing_std_rad)
                    ),
                    "predicted_successor_bearing_rad": (
                        None
                        if decision.predicted_successor_bearing_rad is None
                        else list(
                            decision.predicted_successor_bearing_rad
                        )
                    ),
                    "measured_successor_bearing_rad": (
                        None
                        if decision.measured_successor_bearing_rad is None
                        else list(
                            decision.measured_successor_bearing_rad
                        )
                    ),
                    "successor_rate_rad_s": (
                        None
                        if decision.successor_rate_rad_s is None
                        else list(decision.successor_rate_rad_s)
                    ),
                    "successor_prediction_horizon_s": (
                        decision.successor_prediction_horizon_s
                    ),
                    "successor_prediction_confidence": (
                        decision.successor_prediction_confidence
                    ),
                    "current_yaw_release": (
                        decision.current_yaw_release
                    ),
                    "passage_yaw_authority": (
                        decision.passage_yaw_authority
                    ),
                    "successor_yaw_contribution_rad": (
                        decision.successor_yaw_contribution_rad
                    ),
                    "passage_committed": decision.passage_committed,
                    "precommit_successor_roll_authority": (
                        decision.precommit_successor_roll_authority
                    ),
                    "precommit_successor_target_roll_rad": (
                        decision.precommit_successor_target_roll_rad
                    ),
                    "precommit_successor_yaw_authority": (
                        decision.precommit_successor_yaw_authority
                    ),
                    "precommit_successor_yaw_rate_rad_s": (
                        decision.precommit_successor_yaw_rate_rad_s
                    ),
                    "precommit_successor_yaw_heading_delta_rad": (
                        decision
                        .precommit_successor_yaw_heading_delta_rad
                    ),
                    "precommit_successor_yaw_contribution_rad": (
                        decision
                        .precommit_successor_yaw_contribution_rad
                    ),
                    "precommit_current_horizontal_fov_clearance_norm": (
                        decision
                        .precommit_current_horizontal_fov_clearance_norm
                    ),
                    "committed_successor_roll_authority": (
                        decision.committed_successor_roll_authority
                    ),
                    "committed_successor_target_roll_rad": (
                        decision.committed_successor_target_roll_rad
                    ),
                    "committed_successor_pitch_authority": (
                        decision.committed_successor_pitch_authority
                    ),
                    "committed_successor_target_pitch_rad": (
                        decision.committed_successor_target_pitch_rad
                    ),
                    "committed_successor_yaw_authority": (
                        decision.committed_successor_yaw_authority
                    ),
                    "committed_successor_yaw_rate_rad_s": (
                        decision.committed_successor_yaw_rate_rad_s
                    ),
                    "committed_successor_camera_center_norm": (
                        None
                        if (
                            decision
                            .committed_successor_camera_center_norm
                            is None
                        )
                        else list(
                            decision
                            .committed_successor_camera_center_norm
                        )
                    ),
                    "committed_successor_camera_center_rate_norm_s": (
                        None
                        if (
                            decision
                            .committed_successor_camera_center_rate_norm_s
                            is None
                        )
                        else list(
                            decision
                            .committed_successor_camera_center_rate_norm_s
                        )
                    ),
                    "successor_transition_held": (
                        decision.successor_transition_held
                    ),
                    "time_to_contact_s": (
                        decision.current_time_to_contact_s
                    ),
                    "braking": decision.braking,
                    "brake_reason": decision.brake_reason,
                    "dropout_held": decision.dropout_held,
                    "residual_translation_rate_rad_s": list(
                        course.current
                        .residual_translational_rate_rad_s
                    ),
                    "residual_translation_rate_norm_s": [
                        course.current.residual_translational_rate_rad_s[0]
                        / self.core.config.horizontal_angle_scale_rad,
                        course.current.residual_translational_rate_rad_s[1]
                        / self.core.config.vertical_angle_scale_rad,
                    ],
                    "current_bearing_rate_qualified": list(
                        course.current.bearing_rate_qualified
                    ),
                    "expansion_rate_s": (
                        course.current.expansion_rate_s
                    ),
                    "current_scale_rate_qualified": (
                        course.current.scale_rate_qualified
                    ),
                    "current_log_scale": course.current.log_scale,
                    "current_log_scale_std": course.current.log_scale_std,
                    "passage_scale_lower_bound": (
                        course.current.log_scale
                        - 2.0 * course.current.log_scale_std
                    ),
                    "passage_scale_ready": bool(
                        course.current.visible
                        and not course.current.ambiguous
                        and not any(course.current.censored_axes)
                        and not course.current.aperture_propagated
                        and all(
                            course.current.bearing_rate_qualified
                        )
                        and course.current.scale_rate_qualified
                        and decision.current_time_to_contact_s is not None
                        and decision.crossing_prediction_horizon_s
                        >= decision.current_time_to_contact_s - 1e-9
                        and all(
                            allowance > 0.0
                            for allowance in (
                                decision.crossing_allowance_norm
                            )
                        )
                        and all(
                            clearance >= 0.0
                            for clearance in (
                                decision.terminal_crossing_clearance_norm
                            )
                        )
                        and (
                            post_governor_contact_budget_s is not None
                            and post_governor_contact_budget_s
                            >= self.core.config
                            .terminal_min_post_governor_contact_budget_s
                        )
                    ),
                    "current_visible": course.current.visible,
                    "current_ambiguous": course.current.ambiguous,
                    "current_censored_axes": list(
                        course.current.censored_axes
                    ),
                    "current_confidence": course.current.confidence,
                    "promotion_count": course.promotion_count,
                }
            )
        return evidence

    def continuity_hold_authority(
        self,
        *,
        now_monotonic_ns: int,
        maximum_age_s: float,
    ) -> Mapping[str, Any]:
        sample = self._last_applied_sample
        command = self._wire_governor.last_command
        if (
            sample is None
            or command is None
            or type(now_monotonic_ns) is not int
            or now_monotonic_ns < sample.monotonic_ns
            or not math.isfinite(float(maximum_age_s))
            or float(maximum_age_s) <= 0.0
            or now_monotonic_ns - sample.monotonic_ns
            > round(float(maximum_age_s) * 1_000_000_000.0)
        ):
            raise DynamicCourseError(
                "dynamic continuity hold lacks a fresh applied command"
            )
        return {
            "target_roll_rad": sample.target_roll_rad,
            "target_pitch_rad": sample.target_pitch_rad,
            "yaw_rate_rad_s": sample.yaw_rate_rad_s,
            "thrust": sample.thrust,
            "wire_command": command,
            "source_wire_start_monotonic_ns": sample.monotonic_ns,
        }

    def evidence_summary(self) -> Mapping[str, Any]:
        return {
            "schema": "aigp-vq2-dynamic-controller-summary/1",
            "controller_family": DYNAMIC_CONTROLLER_FAMILY,
            "applied_command_count": self._applied_command_count,
            "dynamic_command_count": self._dynamic_command_count,
            "roll_reversal_count": self._roll_reversal_count,
            "track_count": len(self.core.track_states),
            "promotion_count": (
                0
                if not self.has_applied_command
                else self.core.course_state().promotion_count
            ),
        }


class _DynamicImageServo:
    """Image-servo-shaped view of one dynamic course session."""

    def __init__(
        self,
        session: DynamicVisualCourseSession,
        expected_current_track_id: str,
        expected_gate_index: int,
        tuning: VisualServoTuning,
    ) -> None:
        self.session = session
        self.expected_current_track_id = expected_current_track_id
        self.expected_gate_index = expected_gate_index
        self.tuning = tuning
        self.reset_segment()

    def reset_segment(self) -> None:
        self._corridor_frames = 0
        self._latched_next_track_id: Optional[str] = None
        self._passage_preview_retired = False
        self._last_abs_error: Optional[tuple[float, float]] = None

    @property
    def corridor_frames(self) -> int:
        return self._corridor_frames

    @property
    def latched_next_track_id(self) -> Optional[str]:
        return self._latched_next_track_id

    def retire_advance_passage_preview(self) -> None:
        if self._latched_next_track_id is None:
            raise VisualServoRefusal(
                "dynamic passage preview lacks a latched successor"
            )
        self._passage_preview_retired = True

    def step(
        self,
        current: VisualTarget,
        *,
        now_monotonic_s: float,
        segment_elapsed_s: float,
        segment_yaw_excursion_rad: float,
        next_target: Optional[VisualTarget] = None,
        requested_next_blend: float = 0.0,
        allow_advance: bool = True,
        allow_passage_safe_next_blend: bool = False,
    ) -> VisualServoOutput:
        del segment_elapsed_s
        del segment_yaw_excursion_rad
        if current.track_id != self.expected_current_track_id:
            raise VisualServoRefusal(
                "dynamic servo current identity changed"
            )
        now_ns = round(float(now_monotonic_s) * 1_000_000_000.0)
        staged = self.session._staged
        if staged is not None and staged.adjacent_precredit:
            try:
                authority = (
                    self.session
                    .adjacent_precredit_successor_steering_authority(
                        track_id=current.track_id,
                        now_monotonic_ns=now_ns,
                    )
                )
            except DynamicCourseError as exc:
                raise VisualServoRefusal(
                    "dynamic precredit successor steering refused: "
                    f"{exc}"
                ) from exc
            current_abs = (
                abs(float(current.normalized_x)),
                abs(float(current.normalized_y_down)),
            )
            horizontal_delta = (
                None
                if self._last_abs_error is None
                else current_abs[0] - self._last_abs_error[0]
            )
            vertical_delta = (
                None
                if self._last_abs_error is None
                else current_abs[1] - self._last_abs_error[1]
            )
            self._last_abs_error = current_abs
            self._corridor_frames = 0
            return VisualServoOutput(
                target_roll_rad=float(authority["target_roll_rad"]),
                target_pitch_rad=float(authority["target_pitch_rad"]),
                yaw_rate_rad_s=float(authority["yaw_rate_rad_s"]),
                thrust=float(authority["thrust"]),
                corridor_frames=0,
                advance_enabled=False,
                next_gate_blend=0.0,
                horizontal_error=float(current.normalized_x),
                vertical_error_image_down=float(
                    current.normalized_y_down
                ),
                effective_horizontal_error=float(
                    current.normalized_x
                ),
                effective_vertical_error_image_down=float(
                    current.normalized_y_down
                ),
                effective_horizontal_rate_s=float(
                    current.normalized_x_rate_s
                ),
                effective_vertical_rate_down_s=float(
                    current.normalized_y_rate_down_s
                ),
                next_horizontal_error=None,
                next_vertical_error_image_down=None,
                horizontal_abs_error_delta=horizontal_delta,
                vertical_abs_error_delta=vertical_delta,
                brake_reason="adjacent_recenter",
                yaw_envelope_limited=False,
                reviewed_next_track_id=None,
                passage_preview_retired=False,
                passage_preview_retirement_violations=(),
            )
        successor_track_id = (
            None if next_target is None else next_target.track_id
        )
        try:
            decision = self.session.guide(
                current_track_id=current.track_id,
                successor_track_id=successor_track_id,
                monotonic_ns=now_ns,
            )
        except DynamicCourseError as exc:
            raise VisualServoRefusal(
                f"dynamic course guidance refused: {exc}"
            ) from exc

        if decision is None:
            target_roll = 0.0
            target_pitch = 0.0
            yaw_rate = 0.0
            thrust = 0.275
            passage_error = (
                float(current.normalized_x),
                float(current.normalized_y_down),
            )
            effective_rate = (0.0, 0.0)
            braking = True
            dynamic_brake_reason: Optional[str] = "continuity_seed"
            predicted_next: Optional[tuple[float, float]] = None
            predicted_crossing_clearance = (
                -math.inf,
                -math.inf,
            )
            terminal_crossing_clearance = (
                -math.inf,
                -math.inf,
            )
            crossing_allowance = (0.0, 0.0)
        else:
            target_roll = decision.command.target_roll_rad
            target_pitch = decision.command.target_pitch_rad
            yaw_rate = decision.command.yaw_rate_rad_s
            thrust = decision.command.thrust
            passage_error = decision.passage_error_norm
            course_state = self.session.core.course_state()
            current_dynamic = course_state.current
            effective_rate = (
                current_dynamic.residual_translational_rate_rad_s[0]
                / self.session.core.config.horizontal_angle_scale_rad,
                current_dynamic.residual_translational_rate_rad_s[1]
                / self.session.core.config.vertical_angle_scale_rad,
            )
            braking = decision.braking
            dynamic_brake_reason = decision.brake_reason
            predicted_next = decision.predicted_successor_bearing_rad
            predicted_crossing_clearance = (
                decision.predicted_crossing_clearance_norm
            )
            terminal_crossing_clearance = (
                decision.terminal_crossing_clearance_norm
            )
            crossing_allowance = decision.crossing_allowance_norm

        terminal_history_qualified = bool(
            decision is not None
            and current_dynamic.visible
            and not current_dynamic.ambiguous
            and not any(current_dynamic.censored_axes)
            and not current_dynamic.aperture_propagated
            and all(current_dynamic.bearing_rate_qualified)
            and current_dynamic.scale_rate_qualified
            and decision.current_time_to_contact_s is not None
            and decision.crossing_prediction_horizon_s
            >= decision.current_time_to_contact_s - 1e-9
            and decision.current_time_to_contact_s
            >= (
                self.session.core.config.thrust_command_delay_s
                + self.session.core.config
                .terminal_min_post_governor_contact_budget_s
            )
            and crossing_allowance[0] > 0.0
            and crossing_allowance[1] > 0.0
            and terminal_crossing_clearance[0] >= 0.0
            and terminal_crossing_clearance[1] >= 0.0
            # A full-sweep-unsafe axis may enter the terminal window only
            # while its qualified q motion is carrying it inward.  Axes whose
            # complete approach sweep is already safe need no exception.
            and all(
                predicted_crossing_clearance[axis] >= 0.0
                or (
                    decision.current_crossing_error_q[axis]
                    * decision.crossing_rate_q_s[axis]
                    < 0.0
                )
                for axis in range(2)
            )
            # Negative image-down motion heads toward TOP.  Positive motion,
            # including the +0.337/s 1cab recovery, must not be rejected by a
            # symmetric settled-rate test once q is terminal-safe.
            and float(current.normalized_y_rate_down_s)
            >= -self.session.core.config.vertical_settled_rate_norm_s
        )
        propagated_commitment = bool(
            decision is not None
            and current_dynamic.aperture_propagated
            and current_dynamic.aperture_dynamics_qualified
            and decision.current_aperture_dynamics_qualified
            and decision.current_aperture_half_size_norm is not None
            and decision.current_aperture_prediction_horizon_remaining_s > 0.0
            and current_dynamic.visible
            and not current_dynamic.ambiguous
            and decision.current_time_to_contact_s is not None
            and decision.current_time_to_contact_s
            <= decision.current_aperture_prediction_horizon_remaining_s
            + 1e-9
            and decision.crossing_prediction_horizon_s
            >= decision.current_time_to_contact_s - 1e-9
            and decision.current_time_to_contact_s
            >= (
                self.session.core.config.thrust_command_delay_s
                + self.session.core.config
                .terminal_min_post_governor_contact_budget_s
            )
            and all(
                allowance > 0.0
                for allowance in decision.crossing_allowance_norm
            )
            and all(
                clearance >= 0.0
                for clearance in decision.terminal_crossing_clearance_norm
            )
            and all(
                decision.predicted_crossing_clearance_norm[axis] >= 0.0
                or (
                    decision.current_crossing_error_q[axis]
                    * decision.crossing_rate_q_s[axis]
                    < 0.0
                )
                for axis in range(2)
            )
            and current_dynamic.expansion_rate_s > 0.0
        )
        passage_plane_ready = bool(
            terminal_history_qualified or propagated_commitment
        )
        within_corridor = bool(
            terminal_history_qualified or propagated_commitment
        )
        self._corridor_frames = (
            max(
                self._corridor_frames + 1,
                self.tuning.required_corridor_frames,
            )
            if terminal_history_qualified
            else (
                max(
                    self._corridor_frames,
                    self.tuning.required_corridor_frames,
                )
                if propagated_commitment
                else 0
            )
        )
        if next_target is not None and requested_next_blend > 0.0:
            if self._latched_next_track_id is None:
                self._latched_next_track_id = next_target.track_id
            elif self._latched_next_track_id != next_target.track_id:
                raise VisualServoRefusal(
                    "dynamic servo successor identity changed"
                )
        reviewed_next_track_id = (
            None
            if next_target is None or requested_next_blend <= 0.0
            else next_target.track_id
        )
        next_blend = (
            0.0
            if (
                next_target is None
                or self._passage_preview_retired
                or not allow_passage_safe_next_blend
            )
            else float(requested_next_blend)
        )
        advance_enabled = bool(
            allow_advance
            and self._corridor_frames
            >= self.tuning.required_corridor_frames
        )
        current_abs = (
            abs(float(passage_error[0])),
            abs(float(passage_error[1])),
        )
        horizontal_delta = (
            None
            if self._last_abs_error is None
            else current_abs[0] - self._last_abs_error[0]
        )
        vertical_delta = (
            None
            if self._last_abs_error is None
            else current_abs[1] - self._last_abs_error[1]
        )
        self._last_abs_error = current_abs
        next_horizontal = (
            None
            if predicted_next is None
            else predicted_next[0]
            / self.session.core.config.horizontal_angle_scale_rad
        )
        next_vertical = (
            None
            if predicted_next is None
            else predicted_next[1]
            / self.session.core.config.vertical_angle_scale_rad
        )
        return VisualServoOutput(
            target_roll_rad=target_roll,
            target_pitch_rad=target_pitch,
            yaw_rate_rad_s=yaw_rate,
            thrust=thrust,
            corridor_frames=self._corridor_frames,
            advance_enabled=advance_enabled,
            next_gate_blend=next_blend,
            horizontal_error=float(current.normalized_x),
            vertical_error_image_down=float(
                current.normalized_y_down
            ),
            effective_horizontal_error=float(passage_error[0]),
            effective_vertical_error_image_down=float(
                passage_error[1]
            ),
            effective_horizontal_rate_s=float(effective_rate[0]),
            effective_vertical_rate_down_s=float(effective_rate[1]),
            next_horizontal_error=next_horizontal,
            next_vertical_error_image_down=next_vertical,
            horizontal_abs_error_delta=horizontal_delta,
            vertical_abs_error_delta=vertical_delta,
            brake_reason=(
                "aligning"
                if within_corridor
                else (
                    "dynamic_plane_not_ready"
                    if not passage_plane_ready and not braking
                    else dynamic_brake_reason or "dynamic_intercept"
                )
            ),
            yaw_envelope_limited=False,
            reviewed_next_track_id=reviewed_next_track_id,
            passage_preview_retired=self._passage_preview_retired,
            passage_preview_retirement_violations=(),
        )


class DynamicRollingVisualApproachServo(RollingVisualApproachServo):
    """Graph-authority shell with dynamic estimation and two-gate guidance."""

    def __init__(
        self,
        expected_current_track_id: str,
        expected_gate_index: int,
        tuning: Optional[VisualServoTuning] = None,
        *,
        next_gate_blend: float,
        next_gate_blend_start_log_scale: Optional[float] = None,
        next_gate_blend_full_log_scale: Optional[float] = None,
        session: DynamicVisualCourseSession,
    ) -> None:
        super().__init__(
            expected_current_track_id,
            expected_gate_index,
            tuning,
            next_gate_blend=next_gate_blend,
            next_gate_blend_start_log_scale=(
                next_gate_blend_start_log_scale
            ),
            next_gate_blend_full_log_scale=(
                next_gate_blend_full_log_scale
            ),
        )
        if type(session) is not DynamicVisualCourseSession:
            raise VisualApproachRefusal(
                "dynamic planner requires one exact shared session"
            )
        self._dynamic_session = session
        self._servo = _DynamicImageServo(
            session,
            expected_current_track_id,
            expected_gate_index,
            self._servo.tuning,
        )

    def _validate_current(
        self,
        snapshot: Any,
        update: Any,
        current: VisualTrack,
        *,
        mode: VisualApproachMode,
    ) -> None:
        """Keep graph identity hard while letting the inner aperture steer."""

        try:
            super()._validate_current(
                snapshot,
                update,
                current,
                mode=mode,
            )
        except VisualApproachCurrentGeometryUnavailable:
            # Dynamic stage_snapshot has already admitted this exact tracker
            # publication.  Outer red support may touch an image edge while
            # the complete fitted opening remains clean and co-timed.
            if (
                mode
                not in {
                    VisualApproachMode.APPROACH,
                    VisualApproachMode.PASSAGE,
                }
                or (
                    _complete_current_inner_geometry(current) is None
                    and not self._has_exact_propagated_current_state(
                        current
                    )
                    and not (
                        mode is VisualApproachMode.APPROACH
                        and self._has_exact_degraded_current_steering_state(
                            current
                        )
                    )
                )
            ):
                raise

    def _has_exact_degraded_current_steering_state(
        self,
        track: VisualTrack,
    ) -> bool:
        """Admit only the image axis left observable by clipped outer support."""

        if not track.history:
            return False
        sample = track.history[-1]
        try:
            state = self._dynamic_session.core.course_state().current
        except DynamicCourseError:
            return False
        censored_axes = (
            bool(track.clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)),
            bool(track.clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)),
        )
        return bool(
            _complete_current_inner_geometry(track) is None
            and track.visible
            and not track.ambiguous
            and track.clipping != FrameEdge.NONE
            and any(censored_axes)
            and not all(censored_axes)
            and sample.token == track.latest_token
            and state.track_id == track.track_id
            and state.stream_generation == track.latest_token.generation
            and state.frame_sequence == sample.tracker_frame_sequence
            and state.state_monotonic_ns
            == state.last_measurement_monotonic_ns
            and state.visible
            and not state.ambiguous
            and state.clipping == track.clipping
            and state.clipping == sample.clipping
            and state.censored_axes == censored_axes
            and state.raw_center_norm == track.center_norm
            and state.raw_log_scale is None
            and state.aperture_half_size_norm is None
        )

    def _has_exact_propagated_current_state(
        self,
        track: VisualTrack,
    ) -> bool:
        try:
            state = self._dynamic_session.core.course_state().current
        except DynamicCourseError:
            return False
        return bool(
            state.track_id == track.track_id
            and track.visible
            and not track.ambiguous
            and state.frame_sequence
            == track.history[-1].tracker_frame_sequence
            and state.aperture_half_size_norm is not None
            and state.aperture_propagated
            and state.aperture_seed_monotonic_ns is not None
            and state.aperture_prediction_deadline_monotonic_ns is not None
            and state.state_monotonic_ns
            <= state.aperture_prediction_deadline_monotonic_ns
        )

    def _target(
        self,
        track: VisualTrack,
        *,
        now_monotonic_s: float,
        require_current_authority: bool,
    ) -> VisualTarget:
        target = super()._target(
            track,
            now_monotonic_s=now_monotonic_s,
            require_current_authority=require_current_authority,
        )
        if not require_current_authority:
            return target
        inner = _complete_current_inner_geometry(track)
        propagated = self._has_exact_propagated_current_state(track)
        if (
            not propagated
            and (
                inner is None
                or (
                track.clipping == FrameEdge.NONE
                and not track.center_censored
                )
            )
        ):
            return target
        try:
            state = self._dynamic_session.core.course_state().current
        except DynamicCourseError:
            # The first exact publication seeds the graph shell before
            # _DynamicImageServo.step binds course roles.
            return target
        if (
            state.track_id != track.track_id
            or state.frame_sequence != track.history[-1].tracker_frame_sequence
            or (
                not propagated
                and (
                    inner is None
                    or state.raw_center_norm != inner.center_norm
                    or state.raw_log_scale != inner.log_scale
                    or any(state.censored_axes)
                )
            )
        ):
            raise VisualApproachRefusal(
                "dynamic current target differs from complete inner geometry"
            )
        scale = self._dynamic_session.core.config
        if propagated:
            now_ns = round(float(now_monotonic_s) * 1_000_000_000.0)
            if (
                state.aperture_prediction_deadline_monotonic_ns is None
                or now_ns
                > state.aperture_prediction_deadline_monotonic_ns
            ):
                # A state may be valid at camera capture and expire before the
                # later control decision.  The expired aperture loses all
                # scale, passage, and clearance authority, but it must not
                # erase a fresh outer-support steering measurement.  A clean
                # center can steer directly; a one-axis clip can steer from
                # its measured axis plus the bounded edge direction.  The
                # dynamic core independently withholds the expired aperture
                # from passage/advance decisions.
                horizontal_censored = bool(
                    track.clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
                )
                vertical_censored = bool(
                    track.clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
                )
                if (
                    (
                        track.clipping == FrameEdge.NONE
                        and not track.center_censored
                    )
                    or horizontal_censored != vertical_censored
                ):
                    return target
                raise VisualApproachCurrentGeometryUnavailable(
                    "propagated current-gate state expired"
                )
            camera_center, _ = (
                self._dynamic_session.core._decision_geometry(
                    track.track_id,
                    now_ns,
                )
            )
            prediction_span_ns = (
                state.aperture_prediction_deadline_monotonic_ns
                - state.aperture_seed_monotonic_ns
            )
            assert prediction_span_ns > 0
            remaining_fraction = max(
                0.0,
                min(
                    1.0,
                    (
                        state.aperture_prediction_deadline_monotonic_ns
                        - now_ns
                    )
                    / prediction_span_ns,
                ),
            )
            # A one-axis frame-edge clip censors only that image coordinate.
            # Keep the fresh measured coordinate on the still-observable axis
            # inside the legacy VisualTarget's raw-image domain; the bounded
            # local prediction owns only the censored axis.  Fully observable
            # propagated inner-state corrections and dual-axis clipping keep
            # the paired decision-time reprojection used by the dynamic core.
            one_axis_censored = (
                state.censored_axes[0] != state.censored_axes[1]
            )
            measured_shell_center = (
                target.normalized_x,
                target.normalized_y_down,
            )
            shell_center = tuple(
                (
                    camera_center[axis]
                    if (
                        not one_axis_censored
                        or state.censored_axes[axis]
                    )
                    else measured_shell_center[axis]
                )
                for axis in range(2)
            )
            if not all(
                math.isfinite(float(value))
                for value in shell_center
            ):
                raise VisualApproachRefusal(
                    "dynamic current target projection is nonfinite"
                )
            # The dynamic core retains the unsaturated off-frame projection.
            # The legacy VisualTarget is only a bounded servo-shaped shell:
            # clamp every propagated coordinate at its representable
            # boundary while preserving direction.  A degraded fitted inner
            # can make the state formally uncensored without pulling the
            # command-time camera reprojection back on frame.  This does not
            # create fresh geometry or change passage/advance authority.
            bounded_shell_center = tuple(
                max(
                    -MAX_VISUAL_TARGET_COORDINATE_NORM,
                    min(
                        MAX_VISUAL_TARGET_COORDINATE_NORM,
                        float(shell_center[axis]),
                    ),
                )
                for axis in range(2)
            )
            return replace(
                target,
                normalized_x=bounded_shell_center[0],
                normalized_y_down=bounded_shell_center[1],
                normalized_x_rate_s=(
                    state.residual_translational_rate_rad_s[0]
                    / scale.horizontal_angle_scale_rad
                ),
                normalized_y_rate_down_s=(
                    state.residual_translational_rate_rad_s[1]
                    / scale.vertical_angle_scale_rad
                ),
                log_scale=float(state.log_scale),
                confidence=min(
                    float(track.confidence),
                    float(track.association_confidence),
                )
                * remaining_fraction,
            )
        assert inner is not None
        assert inner.center_norm is not None
        assert inner.log_scale is not None
        return replace(
            target,
            normalized_x=float(inner.center_norm[0]),
            normalized_y_down=float(inner.center_norm[1]),
            normalized_x_rate_s=(
                state.residual_translational_rate_rad_s[0]
                / scale.horizontal_angle_scale_rad
            ),
            normalized_y_rate_down_s=(
                state.residual_translational_rate_rad_s[1]
                / scale.vertical_angle_scale_rad
            ),
            # Preserve direct clean-aperture targeting.  A degraded complete
            # fit may only correct the core's bounded filtered scale; it still
            # cannot mint aperture, clearance, passage, or race authority.
            log_scale=float(
                inner.log_scale
                if inner.passage_usable
                else state.log_scale
            ),
            confidence=min(
                float(track.confidence),
                float(track.association_confidence),
                float(inner.confidence),
            ),
            clipped=False,
            center_censored=False,
            horizontal_censored=False,
            vertical_censored=False,
        )

    def _passage_admission_from_approach(
        self,
        snapshot: Any,
        current_target: VisualTarget,
        next_target: Optional[VisualTarget],
        output: VisualServoOutput,
    ) -> Optional[VisualApproachPassageAdmission]:
        admission = super()._passage_admission_from_approach(
            snapshot,
            current_target,
            next_target,
            output,
        )
        if admission is not None:
            return admission
        retained_id = self._latched_next_track_id
        decision = self._dynamic_session.last_decision
        if (
            next_target is not None
            or retained_id is None
            or decision is None
            or decision.successor_track_id != retained_id
            or output.corridor_frames
            < self._servo.tuning.required_corridor_frames
            or output.brake_reason != "aligning"
            or output.yaw_envelope_limited
            or decision.current_time_to_contact_s is None
            or decision.crossing_prediction_horizon_s <= 0.0
            or any(
                allowance <= 0.0
                for allowance in decision.crossing_allowance_norm
            )
            or any(
                clearance < 0.0
                for clearance in decision.terminal_crossing_clearance_norm
            )
            or snapshot.next_selection_ambiguous
            or snapshot.provisional_track_ids
            or any(
                candidate.track_id != retained_id
                for candidate in snapshot.next_candidates
            )
        ):
            return None
        state = self._dynamic_session.core.course_state()
        successor = state.successor
        if (
            successor is None
            or successor.visible
            or successor.missed_count <= 0
            or not self._dynamic_session.core.retains_successor_lineage(
                retained_id,
                decision.monotonic_ns,
            )
        ):
            return None
        return VisualApproachPassageAdmission(
            basis=VISUAL_PASSAGE_ADMISSION_BASIS,
            current_gate_index=self.expected_gate_index,
            current_target=current_target,
            camera_token=snapshot.latest_camera_token,
            tracker_frame_sequence=snapshot.tracker_frame_sequence,
            corridor_frames=output.corridor_frames,
            preview_track_id=retained_id,
            # Occluded successor geometry has no command authority.  The exact
            # lineage is sealed for post-credit promotion only.
            preview_blend=0.0,
        )

    def observe(
        self,
        snapshot: Any,
        tracker: MultiTargetVisualTracker,
        now_monotonic_s: float,
        segment_elapsed_s: float,
        segment_yaw_excursion_rad: float,
        *,
        mode: VisualApproachMode = VisualApproachMode.APPROACH,
        passage_admission: Any = None,
        passage_forward_closure_authorized: bool = True,
    ) -> Any:
        self._dynamic_session.stage_snapshot(
            snapshot,
            tracker,
            expected_gate_index=self.expected_gate_index,
            expected_current_track_id=self.expected_current_track_id,
            adjacent_precredit=False,
            passage_committed=(mode is VisualApproachMode.PASSAGE),
        )
        return super().observe(
            snapshot,
            tracker,
            now_monotonic_s,
            segment_elapsed_s,
            segment_yaw_excursion_rad,
            mode=mode,
            passage_admission=passage_admission,
            passage_forward_closure_authorized=(
                passage_forward_closure_authorized
            ),
        )

    def observe_promotable_adjacent(
        self,
        snapshot: Any,
        tracker: MultiTargetVisualTracker,
        now_monotonic_s: float,
        segment_elapsed_s: float,
        segment_yaw_excursion_rad: float,
    ) -> Any:
        self._dynamic_session.stage_snapshot(
            snapshot,
            tracker,
            expected_gate_index=self.expected_gate_index,
            expected_current_track_id=self.expected_current_track_id,
            adjacent_precredit=True,
            passage_committed=False,
        )
        return super().observe_promotable_adjacent(
            snapshot,
            tracker,
            now_monotonic_s,
            segment_elapsed_s,
            segment_yaw_excursion_rad,
        )


__all__ = [
    "BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ",
    "DYNAMIC_CONTROLLER_FAMILY",
    "DYNAMIC_CROSSING_COORDINATE_BASIS",
    "DYNAMIC_TIMING_BASIS",
    "DynamicRollingVisualApproachServo",
    "DynamicVisualCourseSession",
    "PostCreditSuccessorSteeringUnavailable",
    "PropagatedCurrentVisibilityGapUnavailable",
    "production_dynamic_course_config",
]
