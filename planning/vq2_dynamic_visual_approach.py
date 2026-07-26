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
    VisualTrack,
)
from planning.vq2_dynamic_course import (
    AppliedCommandSample,
    DynamicCourseConfig,
    DynamicCourseCore,
    DynamicCourseError,
    GateObservation,
    GuidanceDecision,
    ImuAttitudeSample,
)
from planning.vq2_visual_approach import (
    RollingVisualApproachServo,
    VisualApproachMode,
    VisualApproachRefusal,
)
from planning.vq2_visual_servo import (
    VisualServoOutput,
    VisualServoRefusal,
    VisualServoTuning,
    VisualTarget,
)


DYNAMIC_CONTROLLER_FAMILY = "aigp-vq2-dynamic-image-course/1"
DYNAMIC_TIMING_BASIS = (
    "receiver-final-packet-proxy-minus-identified-camera-delay"
)
_HOST_CLOCK_ID = "host-perf-counter"


def production_dynamic_course_config() -> DynamicCourseConfig:
    """Return the conservative first-flight identified configuration.

    The cohort cleanly identifies the body/wire roll sign but not a trustworthy
    roll-to-image acceleration magnitude.  Production therefore uses that sign
    for a modest intercept request while leaving model-based roll prediction at
    zero until an isolated characterization supplies a gain.
    """

    return replace(
        DynamicCourseConfig(),
        roll_guidance_sign=1.0,
        roll_gain=0.18,
        lateral_rate_gain=0.045,
        roll_to_lateral_bearing_accel=0.0,
    )


@dataclass(frozen=True, slots=True)
class _StagedContext:
    expected_gate_index: int
    expected_current_track_id: str
    adjacent_precredit: bool
    camera_token: CameraFrameToken
    tracker_frame_sequence: int


@dataclass(frozen=True, slots=True)
class _WireGovernorConfig:
    max_roll_slew_rad_s2: float = 2.0
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
        values: list[float] = []
        for axis, (old, target, maximum_slew, maximum_accel) in enumerate(
            zip(
                previous,
                targets,
                maximum_slews,
                maximum_accelerations,
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

    @property
    def has_applied_command(self) -> bool:
        return self._applied_command_count > 0

    @property
    def last_decision(self) -> Optional[GuidanceDecision]:
        return self._last_decision

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
            left, top, right, bottom = sample.bbox_norm
            aperture = (right - left, bottom - top)
            confidence = min(
                float(track.confidence),
                float(track.association_confidence),
            )
            measurement_std = (
                0.012 + 0.035 * (1.0 - confidence),
                0.014 + 0.040 * (1.0 - confidence),
                0.035 + 0.080 * (1.0 - confidence),
            )
            return GateObservation(
                track_id=track.track_id,
                frame_sequence=tracker_frame_sequence,
                observation_monotonic_ns=observation_monotonic_ns,
                center_norm=(
                    float(track.center_norm[0]),
                    float(track.center_norm[1]),
                ),
                log_scale=math.log(float(track.apparent_scale)),
                aperture_half_size_norm=aperture,
                clipping=track.clipping,
                center_censored=bool(track.center_censored),
                visible=True,
                ambiguous=bool(track.ambiguous),
                confidence=confidence,
                measurement_std=measurement_std,
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
    ) -> None:
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
        for track_id in sorted(track_ids):
            try:
                track = tracker.track(track_id)
            except KeyError:
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
            camera_token=token,
            tracker_frame_sequence=update.tracker_frame_sequence,
        )

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
        decision = self.core.guide(monotonic_ns)
        self._last_decision = decision
        return decision

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
        thrust_slew_override: bool = False,
        yaw_slew_override: bool = False,
    ) -> Mapping[str, Any]:
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
        self.core.record_applied_command(
            applied_sample,
            governor_discontinuity_axes=discontinuity_axes,
        )
        self._wire_governor.commit(
            wire_command,
            wire_start_monotonic_ns,
            discontinuity_axes=discontinuity_axes,
        )
        self._last_applied_sample = applied_sample
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
            evidence.update(
                {
                    "gate_index": decision.current_gate_index,
                    "current_track_id": decision.current_track_id,
                    "successor_track_id": decision.successor_track_id,
                    "current_center_norm": list(
                        decision.current_center_norm
                    ),
                    "passage_point_norm": list(
                        decision.passage_point_norm
                    ),
                    "passage_error_norm": list(
                        decision.passage_error_norm
                    ),
                    "aperture_margin_norm": list(
                        decision.aperture_margin_norm
                    ),
                    "successor_weight": decision.successor_weight,
                    "predicted_successor_bearing_rad": (
                        None
                        if decision.predicted_successor_bearing_rad is None
                        else list(
                            decision.predicted_successor_bearing_rad
                        )
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
                    "expansion_rate_s": (
                        course.current.expansion_rate_s
                    ),
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
            bearing_std_norm = (1.0, 1.0)
        else:
            target_roll = decision.command.target_roll_rad
            target_pitch = decision.command.target_pitch_rad
            yaw_rate = decision.command.yaw_rate_rad_s
            thrust = decision.command.thrust
            passage_error = decision.passage_error_norm
            course_state = self.session.core.course_state()
            effective_rate = (
                course_state.current.residual_translational_rate_rad_s[0]
                / self.session.core.config.horizontal_angle_scale_rad,
                course_state.current.residual_translational_rate_rad_s[1]
                / self.session.core.config.vertical_angle_scale_rad,
            )
            braking = decision.braking
            dynamic_brake_reason = decision.brake_reason
            predicted_next = decision.predicted_successor_bearing_rad
            bearing_std_norm = (
                decision.current_bearing_std_rad[0]
                / self.session.core.config.horizontal_angle_scale_rad,
                decision.current_bearing_std_rad[1]
                / self.session.core.config.vertical_angle_scale_rad,
            )

        within_corridor = bool(
            decision is not None
            and not braking
            and abs(passage_error[0]) + 2.0 * bearing_std_norm[0]
            <= self.tuning.horizontal_corridor
            and abs(passage_error[1]) + 2.0 * bearing_std_norm[1]
            <= self.tuning.vertical_corridor
        )
        self._corridor_frames = (
            self._corridor_frames + 1 if within_corridor else 0
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
            and not braking
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
                else dynamic_brake_reason or "dynamic_intercept"
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
        )
        return super().observe_promotable_adjacent(
            snapshot,
            tracker,
            now_monotonic_s,
            segment_elapsed_s,
            segment_yaw_excursion_rad,
        )


__all__ = [
    "DYNAMIC_CONTROLLER_FAMILY",
    "DYNAMIC_TIMING_BASIS",
    "DynamicRollingVisualApproachServo",
    "DynamicVisualCourseSession",
    "production_dynamic_course_config",
]
