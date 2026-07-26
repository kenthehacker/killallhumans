from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import math
from types import SimpleNamespace

import pytest

from competition.adapter import (
    AttitudeRateCommand,
    RaceActiveBoundaryChangedBeforeWire,
)
from competition.aigp_messages import RaceStatus
from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    AssociationEvidence,
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualTrackRole,
    VisualTrackSample,
)
from planning.vq2_gate_graph import AuthoritativeRaceStatusRef
from planning.vq2_visual_approach import (
    VisualApproachMode,
    VisualApproachPassageAdmission,
    VisualApproachPassageSafetyUnavailable,
    VisualApproachRefusal,
)
from planning.vq2_visual_servo import (
    ImageVisualServo,
    PassageSafetyViolation,
    PassageSafetyViolationDetail,
    ServoFrameToken,
    VisualServoOutput,
    VisualTarget,
)
from scripts.aigp_vq2_visual_config import default_visual_config
from scripts import aigp_vq2_run as vq2_run
from scripts import aigp_vq2_visual_course_stage as course_stage
from scripts.aigp_vq2_visual_course_stage import (
    VISUAL_COURSE_YAW_PROFILE_SCHEMA,
    VisualCourseStageLimits,
    VisualCourseStageRuntime,
    VisualCourseYawProfile,
    run_visual_course_stage,
)
from scripts.aigp_vq2_yaw_profile import (
    YAW_CALIBRATION_PLAN_ID,
    YAW_CALIBRATION_PLAN_SHA256,
)
from scripts.aigp_vq2_yaw_profile import (
    YAW_CALIBRATION_PROFILE_ID,
    YAW_CALIBRATION_PROFILE_SHA256,
    YAW_CALIBRATION_SOURCE_COMMIT,
    load_yaw_calibration_profile,
    yaw_calibration_profile_evidence,
)


class SafetyAbort(RuntimeError):
    pass


class _Orientation:
    def to_euler(self):
        return 0.0, 0.0, 0.0


class _Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event, **payload):
        self.events.append((event, payload))


def _token(sequence: int) -> CameraFrameToken:
    return CameraFrameToken(
        generation=7,
        frame_id=100 + sequence,
        publication_sequence=sequence,
        stream_id="camera-live",
    )


def _history_sample(
    track_id: str,
    token: CameraFrameToken,
    *,
    previous_token: CameraFrameToken | None = None,
    missed_frames_before_association: int = 0,
    apparent_scale: float = math.exp(-0.50),
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
) -> VisualTrackSample:
    publication_sequence = token.publication_sequence
    assert publication_sequence is not None
    # The deterministic host starts at 0.20 s with publication 10 already
    # available, then publishes before the following 50 Hz wire slot.  Keep
    # the synthetic receiver timestamps on that same clock instead of placing
    # each publication one control period in the future.
    publication_monotonic_ns = (
        publication_sequence - 1
    ) * 20_000_000
    accepted_association = None
    if previous_token is not None:
        previous_publication_sequence = previous_token.publication_sequence
        assert previous_publication_sequence is not None
        gap_ns = (
            publication_sequence - previous_publication_sequence
        ) * 20_000_000
        accepted_association = AssociationEvidence(
            track_id=track_id,
            previous_token=previous_token,
            current_token=token,
            detection_source_index=0,
            cost=0.10,
            confidence=0.80,
            predicted_center_residual_norm=0.01,
            bbox_iou=0.90,
            log_width_change=0.0,
            log_height_change=0.0,
            log_area_residual=0.0,
            clipping_continuity=1.0,
            temporal_consistency=(
                1.0 / (missed_frames_before_association + 1)
            ),
            appearance_distance=0.0,
            ambiguous=False,
            missed_frame_count_before_association=(
                missed_frames_before_association
            ),
            observation_gap_ns=gap_ns,
            publication_gap_ns=gap_ns,
            track_ambiguous_before_association=False,
        )
    return VisualTrackSample(
        tracker_frame_sequence=publication_sequence,
        token=token,
        observation_monotonic_ns=publication_monotonic_ns,
        publication_monotonic_ns=publication_monotonic_ns,
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        camera_source_time_ns=publication_sequence * 20_000_000,
        source_index=0,
        center_norm=(0.04, 0.03),
        bbox_norm=(-0.20, -0.20, 0.20, 0.20),
        apparent_scale=apparent_scale,
        confidence=0.90,
        clipping=clipping,
        center_censored=center_censored,
        association_confidence=(
            0.90 if accepted_association is None else 0.80
        ),
        accepted_association=accepted_association,
    )


def _track(
    track_id: str,
    *,
    visible: bool = True,
    token: CameraFrameToken,
):
    publication_sequence = token.publication_sequence
    assert publication_sequence is not None
    if publication_sequence == 1:
        history = (_history_sample(track_id, token),)
    else:
        previous_token = _token(publication_sequence - 1)
        history = (
            _history_sample(track_id, previous_token),
            _history_sample(
                track_id,
                token,
                previous_token=previous_token,
            ),
        )
    return SimpleNamespace(
        track_id=track_id,
        latest_token=token,
        visible=visible,
        ambiguous=False,
        missed_frame_count=0 if visible else 1,
        consecutive_frame_count=len(history),
        confidence=0.90,
        association_confidence=0.80,
        clipping=FrameEdge.NONE,
        center_censored=False,
        apparent_scale=math.exp(-0.50),
        role=VisualTrackRole.CURRENT,
        history=history,
    )


def _snapshot(
    gate_index: int,
    track_id: str,
    sequence: int,
    *,
    visible: bool = True,
    race_finished: bool = False,
):
    token = _token(sequence)
    track_token = token if visible else _token(sequence - 1)
    return SimpleNamespace(
        tracker_frame_sequence=sequence,
        latest_camera_token=token,
        current_gate_index=gate_index,
        current_track_id=track_id,
        current_track=_track(
            track_id,
            visible=visible,
            token=track_token,
        ),
        authority_usable=visible and not race_finished,
        race_finished=race_finished,
    )


class _Graph:
    def __init__(self, snapshot):
        self.latest_snapshot = snapshot
        self.finish_calls = []

    def confirm_race_finished(
        self,
        tracker,
        *,
        race_status,
        camera_token_at_finish,
    ):
        self.finish_calls.append(
            (tracker, race_status, camera_token_at_finish)
        )
        current = self.latest_snapshot
        self.latest_snapshot = _snapshot(
            race_status.active_gate_index,
            current.current_track_id,
            current.latest_camera_token.publication_sequence,
            visible=False,
            race_finished=True,
        )
        return self.latest_snapshot


class _Tracker:
    time_basis_id = "host-perf-counter"

    def __init__(self, host):
        self.host = host

    def track(self, track_id):
        assert track_id == self.host.current_track_id
        return self.host.visual_gate_graph.latest_snapshot.current_track


def _target(snapshot, track_id):
    token = snapshot.latest_camera_token
    return VisualTarget(
        track_id=track_id,
        frame_token=ServoFrameToken(
            stream_id=token.stream_id,
            generation=token.generation,
            frame_id=token.frame_id,
            publication_sequence=token.publication_sequence,
        ),
        received_monotonic_s=(token.publication_sequence - 1) * 0.02,
        normalized_x=0.04,
        normalized_y_down=0.03,
        normalized_x_rate_s=0.0,
        normalized_y_rate_down_s=0.0,
        log_scale=-0.50,
        log_scale_rate_s=0.20,
        confidence=0.9,
        association_confidence=0.8,
        consecutive_frames=5,
    )


class _Servo:
    """Small deterministic stand-in for the already-focused planner tests."""

    def __init__(
        self,
        expected_current_track_id,
        expected_gate_index,
        tuning,
        *,
        next_gate_blend,
        next_gate_blend_start_log_scale=None,
        next_gate_blend_full_log_scale=None,
        required_next_track_id=None,
        yaw_rate=0.02,
        passage_advances=True,
        preview_track_id=None,
        passage_preview_blend=0.0,
        passage_preview_retire_once=False,
        calls=None,
    ):
        self.track_id = expected_current_track_id
        self.gate_index = expected_gate_index
        self.tuning = tuning
        self.next_gate_blend = next_gate_blend
        self.next_gate_blend_start_log_scale = (
            next_gate_blend_start_log_scale
        )
        self.next_gate_blend_full_log_scale = (
            next_gate_blend_full_log_scale
        )
        self.required_next_track_id = required_next_track_id
        self.yaw_rate = yaw_rate
        self.passage_advances = passage_advances
        self.preview_track_id = (
            preview_track_id
            if preview_track_id is not None
            else f"track-{expected_gate_index + 1}"
        )
        self.passage_preview_blend = passage_preview_blend
        self.passage_preview_retire_once = (
            passage_preview_retire_once
        )
        self.passage_preview_retirement_emitted = False
        self.passage_preview_retired = False
        self.generic_passage_sample_count = 0
        self.calls = calls if calls is not None else []

    def retire_passage_preview(self, expected_track_id):
        assert expected_track_id == self.preview_track_id
        self.passage_preview_retired = True

    def observe(
        self,
        snapshot,
        tracker,
        now_monotonic_s,
        segment_elapsed_s,
        segment_yaw_excursion_rad,
        *,
        mode,
        passage_admission,
    ):
        del tracker, now_monotonic_s, segment_elapsed_s
        del segment_yaw_excursion_rad
        self.calls.append((self.gate_index, mode, passage_admission))
        if not snapshot.current_track.visible:
            raise VisualApproachRefusal("current identity is no longer visible")
        target = _target(snapshot, self.track_id)
        if mode is VisualApproachMode.PASSAGE:
            self.generic_passage_sample_count += 1
            target = replace(
                target,
                log_scale=(
                    -0.90
                    + 0.06 * self.generic_passage_sample_count
                ),
            )
        advance = bool(
            mode is VisualApproachMode.PASSAGE
            and self.passage_advances
        )
        retirement_details = ()
        if (
            mode is VisualApproachMode.PASSAGE
            and self.passage_preview_retire_once
            and not self.passage_preview_retirement_emitted
        ):
            self.passage_preview_retirement_emitted = True
            self.passage_preview_retired = True
            retirement_details = (
                PassageSafetyViolationDetail(
                    violation=(
                        PassageSafetyViolation.CURRENT_APPARENT_SCALE
                    ),
                    observed=0.56,
                    limit=0.55,
                    excess=0.01,
                ),
            )
        active_preview_blend = (
            self.passage_preview_blend
            if not self.passage_preview_retired
            else 0.0
        )
        output = VisualServoOutput(
            target_roll_rad=0.0,
            target_pitch_rad=-0.105 if advance else 0.0,
            yaw_rate_rad_s=self.yaw_rate,
            thrust=0.295 if advance else 0.21,
            corridor_frames=5,
            advance_enabled=advance,
            next_gate_blend=(
                active_preview_blend
                if mode is VisualApproachMode.PASSAGE
                else 0.0
            ),
            horizontal_error=0.04,
            vertical_error_image_down=0.03,
            effective_horizontal_error=0.04,
            effective_vertical_error_image_down=0.03,
            effective_horizontal_rate_s=0.0,
            effective_vertical_rate_down_s=0.0,
            next_horizontal_error=(
                0.30
                if (
                    mode is VisualApproachMode.PASSAGE
                    and active_preview_blend > 0.0
                )
                else None
            ),
            next_vertical_error_image_down=(
                -0.20
                if (
                    mode is VisualApproachMode.PASSAGE
                    and active_preview_blend > 0.0
                )
                else None
            ),
            horizontal_abs_error_delta=0.0,
            vertical_abs_error_delta=0.0,
            brake_reason=None if advance else "aligning",
            yaw_envelope_limited=False,
            passage_preview_retired=self.passage_preview_retired,
            passage_preview_retirement_violations=(
                retirement_details
            ),
        )
        admission = passage_admission
        if mode is VisualApproachMode.APPROACH:
            admission = VisualApproachPassageAdmission(
                basis="tight-current-corridor-dwell-v1",
                current_gate_index=self.gate_index,
                current_target=target,
                camera_token=snapshot.latest_camera_token,
                tracker_frame_sequence=snapshot.tracker_frame_sequence,
                corridor_frames=5,
                preview_track_id=self.preview_track_id,
                preview_blend=(
                    self.next_gate_blend
                    if self.preview_track_id is not None
                    else 0.0
                ),
            )
        return SimpleNamespace(
            current_target=target,
            servo_output=output,
            passage_admission=admission,
            mode=mode,
        )


class _Host:
    def __init__(
        self,
        *,
        initial_gate=3,
        finish_gate=4,
        fresh_after_samples=2,
        lose_before_credit=False,
        credit_on_approach=False,
        disable_credit=False,
    ):
        self._visual_tracking_enabled = True
        self._last_flight_command_started_ns = None
        self.clock = 0.20
        self.sequence = 10
        self.visual_config = default_visual_config()
        self.visual_tracker = _Tracker(self)
        self.current_gate = initial_gate
        self.current_track_id = f"track-{initial_gate}"
        self.visual_gate_graph = _Graph(
            _snapshot(initial_gate, self.current_track_id, self.sequence)
        )
        self.estimate = SimpleNamespace(
            orientation=_Orientation(),
            body_rates=(0.0, 0.0, 0.0),
        )
        self.recorder = _Recorder()
        self.finish_gate = finish_gate
        self.fresh_after_samples = fresh_after_samples
        self.lose_before_credit = lose_before_credit
        self.credit_on_approach = credit_on_approach
        self.disable_credit = disable_credit
        self.passage_counts = {}
        self.after_promotion_samples = None
        self.crossing_zero_count = 0
        self.commands = []
        self.ticks = []
        self.watchdogs = 0
        self.credit_token = None
        self.confirmed_race_statuses = []
        self.promotion_tokens = []
        self.requested_promotion_track_ids = []
        self.race = AuthoritativeRaceStatusRef.live(
            session_id="test-session",
            reset_epoch=2,
            race_generation=9,
            race_status_sequence=1,
            race_status_boot_ms=1000,
            active_gate_index=initial_gate,
            received_monotonic_ns=100_000_000,
            host_clock_id="host-perf-counter",
        )

    def _advance_race(self):
        if self.disable_credit or self.race.race_finished:
            return
        self.credit_token = self.visual_gate_graph.latest_snapshot.latest_camera_token
        finished = self.current_gate == self.finish_gate
        next_gate = self.current_gate if finished else self.current_gate + 1
        self.race = AuthoritativeRaceStatusRef.live(
            session_id=self.race.session_id,
            reset_epoch=self.race.reset_epoch,
            race_generation=self.race.race_generation,
            race_status_sequence=self.race.race_status_sequence + 1,
            race_status_boot_ms=self.race.race_status_boot_ms + 250,
            active_gate_index=next_gate,
            received_monotonic_ns=max(
                self.race.received_monotonic_ns + 1,
                round(self.clock * 1_000_000_000),
            ),
            host_clock_id=self.race.host_clock_id,
            race_finished=finished,
        )

    def _sample(self):
        self.sequence += 1
        visible = True
        if self.after_promotion_samples is not None:
            self.after_promotion_samples += 1
            visible = (
                self.after_promotion_samples >= self.fresh_after_samples
            )
        if (
            self.lose_before_credit
            and self.passage_counts.get(self.current_gate, 0) >= 3
            and self.race.active_gate_index == self.current_gate
        ):
            visible = False
        self.visual_gate_graph.latest_snapshot = _snapshot(
            self.current_gate,
            self.current_track_id,
            self.sequence,
            visible=visible,
        )

    def _watchdog(self, **kwargs):
        assert kwargs["enforce_benign_pad_budget"] is True
        assert kwargs["allow_benign_pad_contact"] is (
            self.clock < 0.55
        )
        self.watchdogs += 1

    async def _wait_for_next_flight_command_slot(self):
        if self._last_flight_command_started_ns is not None:
            self.clock = max(
                self.clock,
                self._last_flight_command_started_ns / 1e9 + 0.02,
            )
        return self.clock

    async def _send_flight_command(self, command, **kwargs):
        self.commands.append((command, dict(kwargs), self.current_gate))
        self._last_flight_command_started_ns = round(self.clock * 1e9)
        passage_wire_command = bool(
            command.thrust == 0.295
            or (
                self.current_gate == 0
                and command.pitch_rate == -0.105
                and command.thrust > 0.0
            )
        )
        if passage_wire_command:
            self.passage_counts[self.current_gate] = (
                self.passage_counts.get(self.current_gate, 0) + 1
            )
            if (
                self.passage_counts[self.current_gate] >= 3
                and not self.lose_before_credit
            ):
                self._advance_race()
        elif command.thrust == 0.21 and self.credit_on_approach:
            self._advance_race()
        elif (
            command.roll_rate
            == command.pitch_rate
            == command.yaw_rate
            == command.thrust
            == 0.0
        ):
            self.crossing_zero_count += 1
            if self.lose_before_credit and self.crossing_zero_count >= 2:
                self._advance_race()
        if kwargs.get("require_wire_receipt"):
            token = kwargs["wire_visual_token"]
            return {
                "call_start_monotonic_ns": (
                    self._last_flight_command_started_ns
                ),
                "visual_receiver_authority": {
                    "schema": "aigp-vq2-visual-wire-authority/1",
                    "frame_token": asdict(token),
                    "call_start_monotonic_ns": (
                        self._last_flight_command_started_ns
                    ),
                    "transport_return_monotonic_ns": (
                        self._last_flight_command_started_ns
                    ),
                    "publication_pinned_through_transport_return": True,
                }
            }
        return None

    def _assert_visual_receiver_token_current(self, expected_token):
        assert (
            expected_token
            == self.visual_gate_graph.latest_snapshot.latest_camera_token
        )
        return expected_token

    def _visual_race_status_ref(self):
        return self.race

    def _visual_camera_token_at_race_credit(self, race_status):
        assert race_status == self.race
        return self.credit_token

    def _confirm_visual_transition(
        self,
        *,
        from_gate_index,
        to_gate_index,
        race_status,
        promoted_track_id=None,
    ):
        assert from_gate_index == self.current_gate
        assert to_gate_index == self.current_gate + 1
        assert race_status is self.race
        self.confirmed_race_statuses.append(race_status)
        self.requested_promotion_track_ids.append(promoted_track_id)
        retired = self.current_track_id
        promoted = promoted_track_id or f"track-{to_gate_index}"
        promotion_token = (
            self.visual_gate_graph.latest_snapshot.latest_camera_token
        )
        self.promotion_tokens.append(promotion_token)
        transition = SimpleNamespace(
            from_gate_index=from_gate_index,
            to_gate_index=to_gate_index,
            retired_track_id=retired,
            promoted_track_id=promoted,
            history_length_before_promotion=17,
            history_length_after_promotion=17,
            promoted_history_sha256="a" * 64,
            camera_token_at_credit=self.credit_token,
            promoted_latest_token_at_promotion=promotion_token,
            race_status=race_status,
        )
        self.current_gate = to_gate_index
        self.current_track_id = promoted
        self.after_promotion_samples = 0
        self.visual_gate_graph.latest_snapshot = _snapshot(
            self.current_gate,
            self.current_track_id,
            promotion_token.publication_sequence,
            visible=False,
        )
        return transition

    def _record_tick(self, stage, elapsed_s, command):
        self.ticks.append((stage, elapsed_s, command))


def _yaw_profile():
    return VisualCourseYawProfile.load_tracked()


def _context():
    return SimpleNamespace(spawn_pitch_rad=-0.31)


def _runtime(host, *, yaw_profile=True, servo_options=None, limits=None):
    calls = []
    options = dict(servo_options or {})

    def servo_factory(*args, **kwargs):
        return _Servo(*args, **kwargs, calls=calls, **options)

    async def sleep(delay):
        assert delay >= 0.0
        host.clock += delay

    def next_deadline(previous, now, period):
        return max(previous + period, now + period)

    def attitude_rate(_estimate, *, target_roll_rad, target_pitch_rad, thrust):
        return AttitudeRateCommand(
            target_roll_rad,
            target_pitch_rad,
            0.0,
            thrust,
        )

    def limit(command, maximum):
        return AttitudeRateCommand(
            max(-maximum, min(maximum, command.roll_rate)),
            max(-maximum, min(maximum, command.pitch_rate)),
            0.0,
            command.thrust,
        )

    def validate(command):
        assert max(
            abs(command.roll_rate),
            abs(command.pitch_rate),
            abs(command.yaw_rate),
        ) <= 0.25
        assert 0.0 <= command.thrust <= 0.35

    profile = _yaw_profile() if yaw_profile else None
    runtime = VisualCourseStageRuntime(
        safety_abort_type=SafetyAbort,
        cancelled_error_type=asyncio.CancelledError,
        monotonic=lambda: host.clock,
        perf_counter_ns=lambda: round(host.clock * 1e9),
        sleep=sleep,
        next_control_deadline=next_deadline,
        attitude_rate_command=attitude_rate,
        limit_command_rates=limit,
        validate_command=validate,
        yaw_profile=profile,
        expected_yaw_profile_sha256=(
            None if profile is None else profile.profile_sha256
        ),
        limits=limits or VisualCourseStageLimits(),
        servo_factory=servo_factory,
    )
    return runtime, calls


def test_generic_course_repeats_lifecycle_from_nonzero_gate_until_finish():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["outcome"] == "race_finished"
    assert result["first_causal_blocker"] is None
    assert result["passage_authority_enabled"] is True
    assert result["initial_gate_index"] == 3
    assert result["maximum_authoritative_gate_index"] == 4
    assert result["final_gate_index"] == 4
    assert [
        (item["from_gate_index"], item["to_gate_index"])
        for item in result["authoritative_transitions"]
    ] == [(3, 4)]
    assert [item["gate_index"] for item in result["segments"]] == [3, 4]
    assert all(
        item["passage_authority_enabled"] is True
        for item in result["segments"]
    )
    assert all(
        item["advance_command_count"] >= 3
        for item in result["segments"]
    )
    assert all(
        calls[index][1] is VisualApproachMode.APPROACH
        and calls[index + 1][1] is VisualApproachMode.PASSAGE
        for index in (0, 4)
    )
    transition = result["authoritative_transitions"][0]
    assert transition["promotion_confirmed"] is True
    assert transition["pre_transition_approach_command_count"] == 1
    assert transition["pre_transition_passage_command_count"] == 3
    assert transition["post_transition_navigation_command_count"] == 4
    assert transition["passage_authority_enabled"] is True
    assert transition["history_length_before_promotion"] == (
        transition["history_length_after_promotion"]
    )
    assert host.visual_gate_graph.finish_calls
    assert host.recorder.events[-1][0] == "visual_course_complete"


def test_transition_carries_exact_reviewed_passage_preview_identity():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"preview_track_id": "reviewed-track-4"},
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert host.requested_promotion_track_ids == ["reviewed-track-4"]
    transition = result["authoritative_transitions"][0]
    assert transition["promoted_track_id"] == "reviewed-track-4"


def test_passage_preview_wire_count_is_compact_and_transition_scoped():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"passage_preview_blend": 0.20},
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment_count = sum(
        segment["passage_next_preview_command_count"]
        for segment in result["segments"]
    )
    assert segment_count > 0
    assert result["passage_next_preview_command_count"] == segment_count
    transition = result["authoritative_transitions"][0]
    assert (
        transition[
            "pre_transition_passage_next_preview_command_count"
        ]
        > 0
    )


def test_passage_preview_hard_retirement_is_recorded_and_nonfatal():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"passage_preview_retire_once": True},
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["passage_next_preview_command_count"] == 0
    for segment in result["segments"]:
        assert segment["next_preview_retired"] is True
        assert segment["next_preview_withdrawal_count"] == 1
        withdrawal = segment["next_preview_withdrawal"]
        assert withdrawal["reason"] == (
            "passage_preview_envelope_retired"
        )
        assert withdrawal["violation_codes"] == [
            "current_apparent_scale"
        ]
        assert segment["passage_command_count"] > 0
    events = [
        payload
        for event, payload in host.recorder.events
        if (
            event == "visual_course_next_preview_withdrawn"
            and payload["reason"]
            == "passage_preview_envelope_retired"
        )
    ]
    assert len(events) == len(result["segments"])


def test_nonterminal_transition_refuses_without_reviewed_preview_identity():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"preview_track_id": ""},
    )

    with pytest.raises(
        SafetyAbort,
        match="nonterminal transition lacks its reviewed next-track identity",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.requested_promotion_track_ids == []
    transition = host._visual_course_summary[
        "authoritative_transitions"
    ][0]
    assert transition["promotion_confirmed"] is False


def test_crossing_loss_latches_only_after_credible_passage_and_sends_zeros():
    host = _Host(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["race_finished"] is True
    assert result["segments"][0]["crossing_anchor"] is not None
    assert (
        result["segments"][0]["crossing_wait_zero_command_count"]
        == 2
    )
    crossing = [
        command
        for command, _kwargs, _gate in host.commands
        if command.thrust == 0.0
    ]
    assert len(crossing) == 2
    assert all(
        command.roll_rate
        == command.pitch_rate
        == command.yaw_rate
        == command.thrust
        == 0.0
        for command in crossing
    )


def test_latched_crossing_without_authoritative_credit_times_out_bounded():
    host = _Host(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="gate 6 credit timed out"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["near_plane_latch"] is not None
    assert 1 <= segment["crossing_wait_zero_command_count"] <= 20
    assert host.race.active_gate_index == 6
    assert host.race.race_finished is False


@pytest.mark.parametrize("case", ("stale", "ambiguous", "off_center"))
def test_latched_unsafe_target_is_refused_without_a_wire_send(case):
    class UnsafePostLatchHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.unsafe_token = None

        def _sample(self):
            super()._sample()
            summary = getattr(self, "_visual_course_summary", {})
            segments = summary.get("segments", ())
            if (
                not segments
                or segments[-1].get("near_plane_latch") is None
            ):
                return
            snapshot = self.visual_gate_graph.latest_snapshot
            track = snapshot.current_track
            track.center_norm = (0.04, 0.03)
            track.center_velocity_norm_s = (0.0, 0.0)
            track.authoritative_gate_index = self.current_gate
            if case == "stale":
                track.latest_token = _token(self.sequence - 1)
            elif case == "ambiguous":
                track.ambiguous = True
                track.role = VisualTrackRole.AMBIGUOUS
            else:
                track.center_norm = (0.80, 0.03)
            self.unsafe_token = snapshot.latest_camera_token

    class UnsafeRefusingServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            track = snapshot.current_track
            center = getattr(track, "center_norm", (0.0, 0.0))
            if (
                track.latest_token != snapshot.latest_camera_token
                or track.ambiguous
                or abs(center[0]) > 0.50
            ):
                raise VisualApproachRefusal(
                    "unsafe current target in deterministic fixture"
                )
            return super().observe(snapshot, *args, **kwargs)

    host = UnsafePostLatchHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: UnsafeRefusingServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="latched near-plane measurement became unsafe",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.unsafe_token is not None
    sent_tokens = {
        kwargs.get("wire_visual_token")
        for _command, kwargs, _gate in host.commands
    }
    assert host.unsafe_token not in sent_tokens


@pytest.mark.parametrize(
    "case",
    (
        "schema",
        "nested_start",
        "top_start",
        "host_start",
        "frame_token",
        "not_pinned",
    ),
)
def test_visual_send_rejects_unbound_wire_timing(case):
    class UnboundReceiptHost(_Host):
        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(command, **kwargs)
            if receipt is None:
                return receipt
            authority = receipt["visual_receiver_authority"]
            if case == "schema":
                authority["schema"] = "other"
            elif case == "nested_start":
                authority["call_start_monotonic_ns"] = True
            elif case == "top_start":
                receipt["call_start_monotonic_ns"] += 1
            elif case == "host_start":
                self._last_flight_command_started_ns += 1
            elif case == "frame_token":
                authority["frame_token"] = asdict(
                    _token(
                        kwargs[
                            "wire_visual_token"
                        ].publication_sequence
                        + 1
                    )
                )
            else:
                authority[
                    "publication_pinned_through_transport_return"
                ] = False
            return receipt

    host = UnboundReceiptHost(initial_gate=3, finish_gate=3)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="lacks exact visual wire timing",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )


@pytest.mark.parametrize("case", ("token", "observation"))
def test_current_target_rejects_inconsistent_observation_provenance(case):
    snapshot = _snapshot(0, "track-0", 10)
    target = _target(snapshot, "track-0")
    sample = snapshot.current_track.history[-1]
    if case == "token":
        snapshot.current_track.history = (
            replace(sample, token=_token(11)),
        )
    else:
        snapshot.current_track.history = (
            replace(sample, observation_monotonic_ns=200_000_001),
        )

    with pytest.raises(SafetyAbort, match="provenance is inconsistent"):
        course_stage._current_target_observation_monotonic_ns(
            snapshot,
            target,
            abort_type=SafetyAbort,
        )


def test_crossing_wait_accepts_newer_same_gate_status_before_credit():
    class SameGateHeartbeatHost(_Host):
        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(
                command,
                **kwargs,
            )
            if (
                command.roll_rate
                == command.pitch_rate
                == command.yaw_rate
                == command.thrust
                == 0.0
                and self.crossing_zero_count == 1
            ):
                self.race = AuthoritativeRaceStatusRef.live(
                    session_id=self.race.session_id,
                    reset_epoch=self.race.reset_epoch,
                    race_generation=self.race.race_generation,
                    race_status_sequence=(
                        self.race.race_status_sequence + 1
                    ),
                    race_status_boot_ms=(
                        self.race.race_status_boot_ms + 100
                    ),
                    active_gate_index=self.current_gate,
                    received_monotonic_ns=max(
                        self.race.received_monotonic_ns + 1,
                        round(self.clock * 1_000_000_000),
                    ),
                    host_clock_id=self.race.host_clock_id,
                )
            return receipt

    host = SameGateHeartbeatHost(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["segments"][0][
        "crossing_wait_zero_command_count"
    ] == 2


def test_atomic_no_wire_credit_uses_latched_passage_and_finishes():
    class AtomicCreditHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.refused_attempts = 0
            self.post_credit_navigation_attempts = 0
            self.commands_at_credit = None

        async def _send_flight_command(self, command, **kwargs):
            if self.race.race_finished and command.thrust > 0.0:
                self.post_credit_navigation_attempts += 1
            if (
                command.thrust == 0.295
                and self.passage_counts.get(self.current_gate, 0) >= 3
                and not self.race.race_finished
            ):
                self.refused_attempts += 1
                self.disable_credit = False
                try:
                    self._advance_race()
                finally:
                    self.disable_credit = True
                self.commands_at_credit = len(self.commands)
                raise RaceActiveBoundaryChangedBeforeWire(
                    "synthetic atomic race credit before wire"
                )
            return await super()._send_flight_command(command, **kwargs)

    host = AtomicCreditHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["segments"][0]["crossing_anchor"] is not None
    assert result["segments"][0]["passage_command_count"] == 3
    assert host.refused_attempts == 1
    assert host.commands_at_credit == len(host.commands)
    assert host.post_credit_navigation_attempts == 0


@pytest.mark.parametrize(
    ("vertical", "vertical_rate", "expected"),
    [
        (-0.08333333333333333, -0.23, 0.3106466666666667),
        (-1.0, 0.5, 0.252),
        (1.0, -0.5, 0.298),
        (-1.0, -1.0, 0.32),
        (1.0, 1.0, 0.21),
    ],
)
def test_gate0_proved_vertical_collective_is_exact_and_bounded(
    vertical,
    vertical_rate,
    expected,
):
    assert course_stage._gate0_proved_vertical_collective(
        vertical,
        vertical_rate,
    ) == pytest.approx(expected)


def test_gate0_proved_vertical_collective_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="must be finite"):
        course_stage._gate0_proved_vertical_collective(
            float("nan"),
            0.0,
        )


def test_gate0_proved_collective_recreates_new_frame_rate_filter():
    state = course_stage._Gate0ProvedCollectiveState()

    def observed(sequence, received, vertical):
        target = _target(
            _snapshot(0, "track-0", sequence),
            "track-0",
        )
        return replace(
            target,
            received_monotonic_s=received,
            normalized_y_down=vertical,
        )

    first = observed(1, 1.0, 0.0)
    second = observed(2, 1.1, -0.02)
    first_thrust, first_rate = state.observe(first)
    second_thrust, second_rate = state.observe(second)
    duplicate_thrust, duplicate_rate = state.observe(
        replace(
            second,
            frame_token=replace(
                second.frame_token,
                publication_sequence=3,
            ),
            received_monotonic_s=1.15,
        )
    )
    third_thrust, third_rate = state.observe(
        observed(4, 1.2, -0.03)
    )

    assert first_rate == 0.0
    assert first_thrust == pytest.approx(0.275)
    assert second_rate == pytest.approx(-0.07)
    assert second_thrust == pytest.approx(0.28542)
    assert duplicate_rate == pytest.approx(-0.07)
    assert duplicate_thrust == pytest.approx(0.28542)
    assert third_rate == pytest.approx(-0.0805)
    assert third_thrust == pytest.approx(0.287543)


def test_initial_gate_uses_hashed_launch_bootstrap_only_once():
    host = _Host(
        initial_gate=0,
        finish_gate=1,
        fresh_after_samples=1,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    initial, later = result["segments"]
    launch = initial["launch_bootstrap"]
    assert launch["enabled"] is True
    assert launch["preload_duration_s"] == 0.15
    assert launch["preload_thrust"] == 0.26
    assert launch["boost_duration_s"] == (
        host.visual_config.lifecycle.launch_boost_duration_s
    )
    assert launch["boost_thrust"] == (
        host.visual_config.lifecycle.launch_boost_thrust
    )
    assert launch["post_boost_collective_basis"] == (
        course_stage.GATE0_PROVED_COLLECTIVE_BASIS
    )
    assert launch["post_boost_collective_base"] == 0.275
    assert launch["post_boost_collective_error_gain"] == 0.080
    assert launch["post_boost_collective_rate_gain"] == 0.126
    assert launch["post_boost_collective_max_abs_error"] == 0.50
    assert launch["post_boost_collective_max_abs_rate"] == pytest.approx(
        5.0 / 3.0
    )
    assert launch["post_boost_collective_rate_filter_alpha"] == 0.35
    assert launch["post_boost_next_preview_collective_basis"] == (
        course_stage.GATE0_PROVED_NEXT_PREVIEW_BASIS
    )
    assert launch[
        "post_boost_next_preview_collective_error_gain"
    ] == 0.080
    assert launch[
        "post_boost_next_preview_collective_max_thrust_delta"
    ] == 0.012
    assert launch["pitch_blend_s"] == (
        host.visual_config.lifecycle.launch_pitch_blend_s
    )
    assert launch["passage_admission_withheld_count"] > 0
    assert later["launch_bootstrap"]["enabled"] is False

    gate0_commands = [
        (command, kwargs)
        for command, kwargs, gate_index in host.commands
        if gate_index == 0
    ]
    assert launch["first_target_pitch_rad"] == pytest.approx(-0.31)
    assert gate0_commands[0][0].pitch_rate == pytest.approx(-0.25)
    assert gate0_commands[0][0].thrust == 0.26
    assert any(
        command.thrust
        == host.visual_config.lifecycle.launch_boost_thrust
        for command, _kwargs in gate0_commands
    )
    assert launch["last_thrust_phase"] == (
        course_stage.GATE0_PROVED_COLLECTIVE_BASIS
    )
    assert launch["last_thrust"] == pytest.approx(0.2726)
    assert launch["last_current_vertical_error_image_down"] == 0.03
    assert launch["last_current_vertical_rate_down_s"] == 0.0
    assert launch["last_proved_filtered_vertical_rate_down_s"] == 0.0
    assert launch["next_preview_collective_command_count"] == 0
    assert launch["max_next_preview_collective_delta"] == 0.0
    assert launch["last_next_preview_collective_delta"] == 0.0
    assert launch["last_next_preview_collective_track_id"] is None
    assert any(
        command.thrust == pytest.approx(0.2726)
        for command, _kwargs in gate0_commands
    )
    assert all(
        kwargs["wire_race_gate_index"] == 0
        for _command, kwargs in gate0_commands
        if kwargs.get("require_wire_receipt")
    )
    later_navigation = [
        command
        for command, _kwargs, gate_index in host.commands
        if gate_index == 1 and command.thrust > 0.0
    ]
    assert later_navigation
    assert later_navigation[0].thrust == 0.21
    assert later["launch_bootstrap"][
        "post_boost_collective_basis"
    ] is None
    assert later["launch_bootstrap"][
        "post_boost_next_preview_collective_basis"
    ] is None
    gate0_passage = [
        command
        for stage, _elapsed, command in host.ticks
        if stage == "visual-course/gate0/passage"
    ]
    gate1_passage = [
        command
        for stage, _elapsed, command in host.ticks
        if stage == "visual-course/gate1/passage"
    ]
    assert gate0_passage
    assert all(
        command.thrust == pytest.approx(0.2726)
        for command in gate0_passage
    )
    assert gate1_passage
    assert all(command.thrust == 0.295 for command in gate1_passage)


def test_course_wires_the_hashed_next_preview_scale_ramp_to_every_segment():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(host)
    factory_calls = []

    def capture_factory(*args, **kwargs):
        factory_calls.append(dict(kwargs))
        return _Servo(*args, **kwargs)

    runtime = replace(runtime, servo_factory=capture_factory)
    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert len(factory_calls) == 2
    assert all(
        call["next_gate_blend"]
        == host.visual_config.lifecycle.next_gate_blend_max
        and call["next_gate_blend_start_log_scale"]
        == host.visual_config.lifecycle.next_gate_blend_start_log_scale
        and call["next_gate_blend_full_log_scale"]
        == host.visual_config.lifecycle.next_gate_blend_full_log_scale
        for call in factory_calls
    )


def test_passage_safety_refusal_after_entry_remains_fatal():
    host = _Host(initial_gate=3, finish_gate=3, disable_credit=True)

    class PassageRefusingServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            if kwargs["mode"] is VisualApproachMode.PASSAGE:
                raise VisualApproachPassageSafetyUnavailable(
                    "passage authority left its corridor",
                    violation_codes=("current_vertical_rate",),
                    violation_evidence=(
                        (
                            "current_vertical_rate",
                            0.61,
                            0.60,
                            0.01,
                        ),
                    ),
                    camera_observation_monotonic_s=(
                        snapshot.latest_camera_token.publication_sequence
                        * 0.02
                    ),
                )
            return super().observe(snapshot, *args, **kwargs)

    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: PassageRefusingServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="after preview retirement or passage entry",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )


def test_newer_receiver_publication_drops_unsent_proposal_and_replans():
    class SupersedingHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.superseded_expected = None
            self.superseded_receiver = None

        async def _send_flight_command(self, command, **kwargs):
            if (
                self.superseded_expected is None
                and kwargs.get("wire_visual_token") is not None
            ):
                expected = kwargs["wire_visual_token"]
                receiver = _token(expected.publication_sequence + 1)
                self.superseded_expected = expected
                self.superseded_receiver = receiver
                self.sequence = receiver.publication_sequence
                self.clock = max(
                    self.clock,
                    receiver.publication_sequence * 0.02,
                )
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
            return await super()._send_flight_command(command, **kwargs)

    host = SupersedingHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    segment = result["segments"][0]
    assert segment["superseded_proposal_count"] == 1
    assert segment["launch_bootstrap"]["command_count"] == (
        result["visual_navigation_command_count"]
    )
    sent_tokens = [
        kwargs["wire_visual_token"]
        for _command, kwargs, _gate_index in host.commands
        if kwargs.get("wire_visual_token") is not None
    ]
    assert host.superseded_expected not in sent_tokens
    assert all(
        token.publication_sequence
        > host.superseded_expected.publication_sequence
        for token in sent_tokens
    )
    events = [
        payload
        for event, payload in host.recorder.events
        if event == "visual_course_proposal_superseded"
    ]
    assert len(events) == 1
    assert events[0]["expected_frame_token"] == {
        "generation": host.superseded_expected.generation,
        "frame_id": host.superseded_expected.frame_id,
        "publication_sequence": (
            host.superseded_expected.publication_sequence
        ),
        "stream_id": host.superseded_expected.stream_id,
    }
    assert events[0]["receiver_frame_token"] == {
        "generation": host.superseded_receiver.generation,
        "frame_id": host.superseded_receiver.frame_id,
        "publication_sequence": (
            host.superseded_receiver.publication_sequence
        ),
        "stream_id": host.superseded_receiver.stream_id,
    }


def test_precheck_publication_supersession_replans_before_wire():
    class PrecheckSupersedingHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.superseded_expected = None

        def _assert_visual_receiver_token_current(self, expected_token):
            if self.superseded_expected is None:
                receiver = _token(
                    expected_token.publication_sequence + 1
                )
                self.superseded_expected = expected_token
                self.sequence = receiver.publication_sequence
                self.clock = max(
                    self.clock,
                    receiver.publication_sequence * 0.02,
                )
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected_token
                exc.receiver_visual_token = receiver
                raise exc
            return super()._assert_visual_receiver_token_current(
                expected_token
            )

    host = PrecheckSupersedingHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["segments"][0]["superseded_proposal_count"] == 1
    sent_tokens = [
        kwargs["wire_visual_token"]
        for _command, kwargs, _gate_index in host.commands
        if kwargs.get("wire_visual_token") is not None
    ]
    assert host.superseded_expected not in sent_tokens
    assert result["segments"][0]["launch_bootstrap"][
        "command_count"
    ] == result["visual_navigation_command_count"]


def test_repeated_receiver_supersession_aborts_with_no_stale_send():
    class AlwaysSupersedingHost(_Host):
        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if expected is not None:
                receiver = _token(expected.publication_sequence + 1)
                self.sequence = receiver.publication_sequence
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
            return await super()._send_flight_command(command, **kwargs)

    host = AlwaysSupersedingHost(
        initial_gate=0,
        finish_gate=0,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="repeatedly superseded command authority",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []
    segment = host._visual_course_summary["segments"][0]
    assert segment["superseded_proposal_count"] == (
        course_stage.MAX_CONSECUTIVE_VISUAL_PROPOSAL_SUPERSESSIONS + 1
    )
    assert segment["launch_bootstrap"]["command_count"] == 0


def test_single_supersession_with_stuck_graph_hits_hold_bound():
    class StuckGraphHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.superseded = False

        def _sample(self):
            if not self.superseded:
                super()._sample()

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if expected is not None and not self.superseded:
                self.superseded = True
                receiver = _token(expected.publication_sequence + 1)
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
            return await super()._send_flight_command(command, **kwargs)

    host = StuckGraphHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="paced control tick",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []
    segment = host._visual_course_summary["segments"][0]
    assert segment["superseded_proposal_count"] == 1
    assert segment["launch_bootstrap"]["command_count"] == 0


def test_supersession_then_delayed_wire_slot_hits_hold_bound():
    class DelayedSlotHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.superseded = False
            self.delayed = False

        def _sample(self):
            if not self.superseded:
                super()._sample()

        async def _wait_for_next_flight_command_slot(self):
            ready = await super()._wait_for_next_flight_command_slot()
            if self.superseded and not self.delayed:
                self.clock += (
                    course_stage
                    .MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
                    + 0.001
                )
                self.delayed = True
            return ready

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if expected is not None and not self.superseded:
                self.superseded = True
                receiver = _token(expected.publication_sequence + 1)
                self.sequence = receiver.publication_sequence
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
            return await super()._send_flight_command(command, **kwargs)

    host = DelayedSlotHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="command-slot wait",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []
    segment = host._visual_course_summary["segments"][0]
    assert segment["superseded_proposal_count"] == 1
    assert segment["launch_bootstrap"]["command_count"] == 0


def test_supersession_then_post_slot_validation_delay_hits_hold_bound():
    class PostSlotDelayHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.receiver_checks = 0

        def _assert_visual_receiver_token_current(self, expected_token):
            self.receiver_checks += 1
            if self.receiver_checks == 1:
                receiver = _token(
                    expected_token.publication_sequence + 1
                )
                self.sequence = receiver.publication_sequence
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected_token
                exc.receiver_visual_token = receiver
                raise exc
            if self.receiver_checks == 2:
                self.clock = (
                    0.20
                    + course_stage
                    .MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
                    + 0.001
                )
            return super()._assert_visual_receiver_token_current(
                expected_token
            )

    host = PostSlotDelayHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="pre-wire validation",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []
    segment = host._visual_course_summary["segments"][0]
    assert segment["superseded_proposal_count"] == 1
    assert segment["launch_bootstrap"]["command_count"] == 0


def test_supersession_clips_replacement_wire_deadline_to_hold_bound():
    class NearHoldDeadlineHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.receiver_checks = 0

        def _assert_visual_receiver_token_current(self, expected_token):
            self.receiver_checks += 1
            if self.receiver_checks == 1:
                receiver = _token(
                    expected_token.publication_sequence + 1
                )
                self.sequence = receiver.publication_sequence
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected_token
                exc.receiver_visual_token = receiver
                raise exc
            if self.receiver_checks == 2:
                self.clock = (
                    0.20
                    + course_stage
                    .MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
                    - 0.005
                )
            return super()._assert_visual_receiver_token_current(
                expected_token
            )

    host = NearHoldDeadlineHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    first_command_kwargs = next(
        kwargs
        for _command, kwargs, _gate_index in host.commands
        if kwargs.get("wire_visual_token") is not None
    )
    hold_deadline_ns = round(
        (
            0.20
            + course_stage
            .MAX_VISUAL_PROPOSAL_SUPERSESSION_HOLD_S
        )
        * 1_000_000_000
    )
    assert (
        first_command_kwargs["wire_start_deadline_ns"]
        <= hold_deadline_ns
    )
    assert (
        first_command_kwargs["wire_start_deadline_ns"]
        < round((0.20 + 0.095 + 0.012) * 1_000_000_000)
    )


def test_promotion_failure_retains_observed_authoritative_credit_and_counts():
    class PromotionFailureHost(_Host):
        def _confirm_visual_transition(self, **_kwargs):
            raise SafetyAbort("synthetic promotion failure")

    host = PromotionFailureHost(
        initial_gate=3,
        finish_gate=4,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="synthetic promotion failure"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    summary = host._visual_course_summary
    assert summary["maximum_authoritative_gate_index"] == 4
    assert summary["final_gate_index"] == 4
    assert summary["first_causal_blocker"] == (
        "synthetic promotion failure"
    )
    assert len(summary["authoritative_transitions"]) == 1
    transition = summary["authoritative_transitions"][0]
    assert transition["from_gate_index"] == 3
    assert transition["to_gate_index"] == 4
    assert transition["promotion_confirmed"] is False
    assert transition["pre_transition_navigation_command_count"] == 4
    assert transition["post_transition_navigation_command_count"] == 0


def test_nonzero_yaw_requires_the_accepted_calibration_profile():
    host = _Host(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host, yaw_profile=False)

    with pytest.raises(
        SafetyAbort,
        match="nonzero yaw lacks calibrated authority",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []


def test_authoritative_transition_without_passage_evidence_fails_closed():
    host = _Host(
        initial_gate=8,
        finish_gate=9,
        credit_on_approach=True,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="without credible passage evidence",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )


def test_passage_timer_is_tight_and_cannot_run_without_advance():
    host = _Host(initial_gate=2, finish_gate=2, disable_credit=True)
    limits = replace(
        VisualCourseStageLimits(),
        passage_hard_duration_s=0.40,
    )
    runtime, _calls = _runtime(
        host,
        servo_options={"passage_advances": False, "yaw_rate": 0.0},
        limits=limits,
    )

    with pytest.raises(SafetyAbort, match="passage expired"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert all(
        command.thrust != 0.295
        for command, _kwargs, _gate in host.commands
    )


def test_promoted_gate_fresh_frame_opportunity_has_a_hard_timeout():
    host = _Host(
        initial_gate=0,
        finish_gate=1,
        fresh_after_samples=999,
    )
    limits = replace(
        VisualCourseStageLimits(),
        post_credit_fresh_frame_timeout_s=0.05,
    )
    runtime, _calls = _runtime(host, limits=limits)

    with pytest.raises(SafetyAbort, match="fresh post-credit"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    zeros = [
        command
        for command, _kwargs, _gate in host.commands
        if command.thrust == 0.0
    ]
    assert 1 <= len(zeros) <= 3


def test_yaw_profile_loads_only_the_exact_tracked_multi_run_authority():
    profile = _yaw_profile()

    assert profile.schema == VISUAL_COURSE_YAW_PROFILE_SCHEMA
    assert profile.profile_id == YAW_CALIBRATION_PROFILE_ID
    assert profile.profile_sha256 == YAW_CALIBRATION_PROFILE_SHA256
    assert profile.source_commit == YAW_CALIBRATION_SOURCE_COMMIT
    assert profile.plan_id == YAW_CALIBRATION_PLAN_ID
    assert profile.plan_sha256 == YAW_CALIBRATION_PLAN_SHA256
    assert profile.max_attitude_excursion_rad == 0.05
    assert profile.max_abs_measured_yaw_rate_rad_s == 0.5
    with pytest.raises(TypeError, match="tracked loader"):
        VisualCourseYawProfile(
            issuer=object(),
            **{
                field: getattr(profile, field)
                for field in profile.__dataclass_fields__
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("control_period_s", 0.019, "exactly 50 Hz"),
        ("passage_hard_duration_s", 8.01, "passage duration"),
        ("crossing_status_timeout_s", 0.41, "crossing wait"),
        (
            "censored_passage_coast_max_duration_s",
            0.301,
            "censored passage coast",
        ),
        (
            "censored_passage_coast_max_fresh_frames",
            9,
            "discrete bounds",
        ),
        ("post_credit_fresh_frame_timeout_s", 0.21, "fresh-frame"),
    ),
)
def test_course_limits_refuse_widened_safety_envelopes(
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        replace(VisualCourseStageLimits(), **{field: value})


class _SettableOrientation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def to_euler(self):
        return self.roll, self.pitch, self.yaw


def _set_attitude(
    host,
    *,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    rates=(0.0, 0.0, 0.0),
):
    host.estimate = SimpleNamespace(
        orientation=_SettableOrientation(roll, pitch, yaw),
        body_rates=rates,
    )


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_yaw_at_hard_excursion_zeroes_outward_but_allows_inward(direction):
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    excursion = direction * limits.max_segment_yaw_excursion_rad

    outward = course_stage._limit_calibrated_yaw_request(
        direction * 0.02,
        excursion_rad=excursion,
        measured_euler_yaw_rate_rad_s=0.0,
        limits=limits,
        profile=profile,
        abort_type=SafetyAbort,
    )
    assert outward == 0.0

    admitted = course_stage._limit_calibrated_yaw_request(
        -direction * 0.02,
        excursion_rad=excursion,
        measured_euler_yaw_rate_rad_s=0.0,
        limits=limits,
        profile=profile,
        abort_type=SafetyAbort,
    )
    assert admitted == -direction * 0.02


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_yaw_delayed_projection_at_soft_boundary_zeroes_outward(direction):
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    response_reserve = (
        profile.observed_max_abs_measured_yaw_rate_rad_s
        * profile.control_hold_horizon_s
    )
    soft_boundary = (
        limits.max_segment_yaw_excursion_rad - response_reserve
    )
    excursion = direction * 0.005
    rate_to_boundary = direction * (
        (soft_boundary - abs(excursion)) / profile.max_gyro_response_delay_s
        + 1e-9
    )

    admitted = course_stage._limit_calibrated_yaw_request(
        direction * 0.02,
        excursion_rad=excursion,
        measured_euler_yaw_rate_rad_s=rate_to_boundary,
        limits=limits,
        profile=profile,
        abort_type=SafetyAbort,
    )
    assert admitted == 0.0


def test_course_wires_exact_zero_at_yaw_soft_stop_and_keeps_hard_guards():
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    response_reserve = (
        profile.observed_max_abs_measured_yaw_rate_rad_s
        / math.cos(
            max(
                abs(limits.min_measured_pitch_rad),
                abs(limits.max_measured_pitch_rad),
            )
        )
        * profile.control_hold_horizon_s
    )
    soft_boundary = (
        limits.max_segment_yaw_excursion_rad - response_reserve
    )
    excursion = 0.005
    rate_to_soft_stop = (
        (soft_boundary - excursion) / profile.max_gyro_response_delay_s
        + 1e-6
    )

    class SoftStopHost(_Host):
        sample_count = 0

        def _sample(self):
            super()._sample()
            self.sample_count += 1
            if self.sample_count == 1:
                _set_attitude(
                    self,
                    yaw=excursion,
                    rates=(0.0, 0.0, rate_to_soft_stop),
                )
            else:
                _set_attitude(self)

    host = SoftStopHost(initial_gate=3, finish_gate=3)
    runtime, calls = _runtime(host, limits=limits)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    assert result["success"] is True
    assert segment["yaw_soft_stop_zero_command_count"] == 1
    assert (
        segment["passage_admission_yaw_soft_stop_withheld_count"]
        == 1
    )
    assert calls[0][1] is VisualApproachMode.APPROACH
    assert calls[1][1] is VisualApproachMode.APPROACH
    assert calls[2][1] is VisualApproachMode.PASSAGE
    navigation = [
        command
        for command, kwargs, _gate in host.commands
        if kwargs.get("require_wire_receipt")
    ]
    assert navigation
    assert navigation[0].yaw_rate == 0.0
    assert all(command.yaw_rate == 0.02 for command in navigation[1:])
    events = [
        payload
        for event, payload in host.recorder.events
        if event == "visual_course_yaw_soft_stop_zeroed"
    ]
    assert len(events) == 1
    assert events[0]["requested_yaw_rate_rad_s"] == pytest.approx(0.02)
    assert events[0]["admitted_yaw_rate_rad_s"] == 0.0


@pytest.mark.parametrize(
    ("excursion", "yaw_rate"),
    (
        (0.024, 0.22243007003911772),
        (0.001, -0.50),
    ),
)
def test_measured_yaw_momentum_projects_outside_hard_boundary(
    excursion,
    yaw_rate,
):
    host = _Host()
    _set_attitude(host, yaw=excursion, rates=(0.0, 0.0, yaw_rate))

    with pytest.raises(SafetyAbort, match="momentum projects outside"):
        course_stage._assert_course_attitude_state(
            host,
            yaw_reference_rad=0.0,
            limits=VisualCourseStageLimits(),
            yaw_profile=_yaw_profile(),
            abort_type=SafetyAbort,
            phase="regression",
        )


def test_cross_axis_body_rates_use_exact_euler_yaw_kinematics():
    host = _Host()
    _set_attitude(
        host,
        roll=0.18,
        pitch=-0.35,
        yaw=0.02,
        rates=(0.0, -0.5, -0.5),
    )

    with pytest.raises(SafetyAbort, match="momentum projects outside"):
        course_stage._assert_course_attitude_state(
            host,
            yaw_reference_rad=0.0,
            limits=VisualCourseStageLimits(),
            yaw_profile=_yaw_profile(),
            abort_type=SafetyAbort,
            phase="cross-axis-regression",
        )


@pytest.mark.parametrize(
    ("roll", "pitch", "rates"),
    (
        (0.180001, 0.0, (0.0, 0.0, 0.0)),
        (0.0, -0.350001, (0.0, 0.0, 0.0)),
        (0.0, 0.150001, (0.0, 0.0, 0.0)),
        (0.0, 0.0, (0.500001, 0.0, 0.0)),
        (0.0, 0.0, (0.0, -0.500001, 0.0)),
    ),
)
def test_measured_roll_pitch_and_all_axis_rate_envelopes_fail_closed(
    roll,
    pitch,
    rates,
):
    host = _Host()
    _set_attitude(host, roll=roll, pitch=pitch, rates=rates)

    with pytest.raises(SafetyAbort, match="attitude/body-rate envelope"):
        course_stage._assert_course_attitude_state(
            host,
            yaw_reference_rad=0.0,
            limits=VisualCourseStageLimits(),
            yaw_profile=_yaw_profile(),
            abort_type=SafetyAbort,
            phase="regression",
        )


def test_repeated_same_sign_yaw_requests_zero_before_hard_boundary():
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    excursion = 0.0
    admitted = 0

    while True:
        request = course_stage._limit_calibrated_yaw_request(
            0.02,
            excursion_rad=excursion,
            measured_euler_yaw_rate_rad_s=0.0,
            limits=limits,
            profile=profile,
            abort_type=SafetyAbort,
        )
        if request == 0.0:
            break
        admitted += 1
        excursion += (
            profile.observed_max_abs_measured_yaw_rate_rad_s
            * limits.control_period_s
        )
        assert admitted < 20

    assert admitted > 0
    assert excursion < limits.max_segment_yaw_excursion_rad


def test_duplicate_camera_frame_cannot_hide_new_unsafe_attitude():
    class DuplicateUnsafeHost(_Host):
        sample_count = 0

        def _sample(self):
            self.sample_count += 1
            if self.sample_count == 2:
                _set_attitude(self, yaw=0.050001)
                return
            super()._sample()

    host = DuplicateUnsafeHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="yaw envelope"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert len(host.commands) == 1
    assert host._visual_course_summary["success"] is False


def test_crossing_zero_wait_checks_attitude_before_sending_zero():
    class CrossingUnsafeHost(_Host):
        crossing_loss_observed = False

        def _sample(self):
            if self.crossing_loss_observed:
                super()._sample()
                _set_attitude(self, roll=0.180001)
                return
            super()._sample()
            if (
                self.passage_counts.get(self.current_gate, 0) >= 3
                and not self.visual_gate_graph.latest_snapshot.current_track.visible
            ):
                self.crossing_loss_observed = True

    host = CrossingUnsafeHost(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="attitude/body-rate envelope",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert len(host.commands) == 4
    assert not any(command.thrust == 0.0 for command, _kwargs, _gate in host.commands)


def test_post_credit_wait_checks_attitude_before_sending_zero():
    class RecoveryUnsafeHost(_Host):
        def _sample(self):
            super()._sample()
            if (
                self.after_promotion_samples is not None
                and self.after_promotion_samples > 0
            ):
                _set_attitude(self, pitch=0.150001)

    host = RecoveryUnsafeHost(
        initial_gate=1,
        finish_gate=2,
        fresh_after_samples=999,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="attitude/body-rate envelope",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert [
        command
        for command, _kwargs, gate_index in host.commands
        if gate_index == 2
    ] == []


def test_command_slot_wait_cannot_invalidate_attitude_guard():
    class SlotUnsafeHost(_Host):
        async def _wait_for_next_flight_command_slot(self):
            ready = await super()._wait_for_next_flight_command_slot()
            _set_attitude(self, rates=(0.0, 0.500001, 0.0))
            return ready

    host = SlotUnsafeHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="attitude/body-rate envelope",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []


def test_terminal_race_finish_checks_attitude_before_success_return():
    class TerminalUnsafeHost(_Host):
        def _visual_race_status_ref(self):
            if self.race.race_finished:
                _set_attitude(self, yaw=-0.050001)
            return self.race

    host = TerminalUnsafeHost(initial_gate=4, finish_gate=4)
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="yaw envelope"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert len(host.commands) == 4
    assert host._visual_course_summary["success"] is False
    assert all(
        event != "visual_course_complete"
        for event, _payload in host.recorder.events
    )


def test_production_servo_sign_matches_frozen_image_sign_authority():
    profile = _yaw_profile()
    snapshot = _snapshot(0, "track-0", 10)
    target = _target(snapshot, "track-0")
    servo = ImageVisualServo(default_visual_config().servo)

    output = servo.step(
        target,
        now_monotonic_s=target.received_monotonic_s,
        segment_elapsed_s=0.0,
        segment_yaw_excursion_rad=0.0,
        allow_advance=False,
    )

    assert profile.controller_to_image_sign == 1
    assert target.normalized_x > 0.0
    assert output.yaw_rate_rad_s < 0.0
    assert (
        output.yaw_rate_rad_s
        * target.normalized_x
        * profile.controller_to_image_sign
        < 0.0
    )


class _RunnerAdapter:
    enable_vision = False
    telemetry_mode = "imu"
    fetch_track_on_connect = False

    def __init__(self):
        self.race_status = RaceStatus(1000, 0, -1, 0, -1)


class _RunnerVision:
    pass


@pytest.mark.parametrize(
    ("race_finished", "ordered_transitions", "cleanup_confirmed", "success"),
    (
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ),
)
def test_runner_course_boundary_requires_finish_chain_and_cleanup(
    monkeypatch,
    race_finished,
    ordered_transitions,
    cleanup_confirmed,
    success,
):
    adapter = _RunnerAdapter()
    profile = load_yaw_calibration_profile()
    profile_evidence = yaw_calibration_profile_evidence(profile)
    runner = vq2_run.VQ2Runner(
        adapter,
        _RunnerVision(),
        yaw_calibration_profile=profile,
        yaw_calibration_profile_evidence=profile_evidence,
    )
    context = vq2_run.StartContext(
        0.0,
        -0.31,
        320,
        180,
        6_400,
        1_000,
    )

    async def no_result(*_args, **_kwargs):
        return None

    async def wait_for_go():
        adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
        return context

    async def run_course(_context):
        assert _context is context
        adapter.race_status = RaceStatus(
            1500,
            0,
            50_000_000 if race_finished else -1,
            2,
            500,
        )
        pairs = (
            [(0, 1), (1, 2)]
            if ordered_transitions
            else [(0, 2)]
        )
        graph_transitions = tuple(
            SimpleNamespace(
                from_gate_index=from_index,
                to_gate_index=to_index,
                retired_track_id=f"track-{from_index}",
                promoted_track_id=f"track-{to_index}",
            )
            for from_index, to_index in pairs
        )
        finish_ref = AuthoritativeRaceStatusRef.live(
            session_id="course-boundary-test",
            reset_epoch=1,
            race_generation=3,
            race_status_sequence=9,
            race_status_boot_ms=1500,
            active_gate_index=2,
            received_monotonic_ns=20_000_000,
            host_clock_id="host-perf-counter",
            race_finished=race_finished,
        )
        runner.visual_gate_graph = SimpleNamespace(
            latest_snapshot=SimpleNamespace(
                confirmed_transitions=graph_transitions,
                race_finished=race_finished,
                current_gate_index=2,
                latest_race_status=finish_ref,
            )
        )
        runner._visual_transition = graph_transitions[-1]
        summary = {
            "stage": vq2_run.VISUAL_COURSE_STAGE,
            "success": True,
            "race_finished": race_finished,
            "initial_gate_index": 0,
            "maximum_authoritative_gate_index": 2,
            "final_gate_index": 2,
            "authoritative_transitions": [
                {
                    "from_gate_index": item.from_gate_index,
                    "to_gate_index": item.to_gate_index,
                }
                for item in graph_transitions
            ],
            "segments": [
                {
                    "gate_index": 0,
                    "passage_authority_enabled": True,
                }
            ],
            "visual_navigation_command_count": 8,
            "exact_zero_command_count": 3,
            "yaw_calibration_profile": profile_evidence,
        }
        runner._visual_course_summary = dict(summary)
        return summary

    async def cleanup():
        return cleanup_confirmed

    monkeypatch.setattr(runner, "establish_reset_epoch", no_result)
    monkeypatch.setattr(runner, "normalize_disarmed", no_result)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(
        runner,
        "_bind_initial_visual_gate",
        lambda _context: None,
    )
    monkeypatch.setattr(runner, "arm_confirmed", no_result)
    monkeypatch.setattr(runner, "_run_visual_course", run_course)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(
        runner.run_powered_stage(
            vq2_run.VISUAL_COURSE_STAGE,
            write_diagnostic_pngs=False,
        )
    )

    assert result.success is success
    assert result.cleanup_confirmed is cleanup_confirmed
    assert result.details["authoritative_cleanup_entry"][
        "transitions"
    ] == [list(pair) for pair in (
        [(0, 1), (1, 2)]
        if ordered_transitions
        else [(0, 2)]
    )]
    assert result.details["visual_course"]["cleanup_confirmed"] is (
        cleanup_confirmed
    )
    if not race_finished or not ordered_transitions:
        assert "race_finished/ordered rolling-graph proof" in result.reason
    if not cleanup_confirmed:
        assert "cleanup unconfirmed" in result.reason
