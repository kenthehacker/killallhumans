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
    VisualTrack,
    VisualTrackRole,
    VisualTrackSample,
)
from planning.vq2_gate_graph import AuthoritativeRaceStatusRef
from planning.vq2_visual_approach import (
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
    VisualApproachPassageAdmission,
    VisualApproachPassageSafetyUnavailable,
    VisualApproachRefusal,
)
from planning.vq2_visual_recovery import VisualRecoveryRefusal
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
        observation_monotonic_ns=publication_sequence * 20_000_000,
        publication_monotonic_ns=publication_sequence * 20_000_000,
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
        received_monotonic_s=token.publication_sequence * 0.02,
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

    def recovery_admission(track, transition, **kwargs):
        assert track.track_id == transition.promoted_track_id
        assert kwargs["tracker_time_basis_id"] == "host-perf-counter"
        token = transition.promoted_latest_token_at_promotion
        visibility_tokens = tuple(
            _token(sequence)
            for sequence in range(
                token.publication_sequence - 4,
                token.publication_sequence + 1,
            )
        )
        return SimpleNamespace(
            track_id=track.track_id,
            promotion_anchor_token=token,
            history_tokens=visibility_tokens,
            race_status_sequence=(
                transition.race_status.race_status_sequence
            ),
            race_received_monotonic_ns=(
                transition.race_status.received_monotonic_ns
            ),
            promotion_identity_sha256=transition.promoted_history_sha256,
            promotion_identity_basis="complete_current_visibility_epoch_v1",
            cross_gap_identity_claimed=False,
            visibility_epoch_frame_count=5,
            visibility_epoch_span_s=0.12,
            visibility_epoch_tokens=visibility_tokens,
            visibility_epoch_tracker_frame_sequences=tuple(
                range(5)
            ),
        )

    def recovery_continuation(track, transition, **kwargs):
        assert track.track_id == transition.promoted_track_id
        assert kwargs["tracker_time_basis_id"] == "host-perf-counter"
        previous = kwargs["previous_token"]
        token = track.latest_token
        if (
            token.publication_sequence
            != previous.publication_sequence + 1
        ):
            raise VisualRecoveryRefusal(
                "recovery continuation publication did not advance exactly"
            )
        visibility_tokens = tuple(
            _token(sequence)
            for sequence in range(
                previous.publication_sequence - 4,
                previous.publication_sequence + 1,
            )
        )
        return SimpleNamespace(
            track_id=track.track_id,
            previous_token=previous,
            frame_token=token,
            capture=SimpleNamespace(
                frame_token=ServoFrameToken(
                    stream_id=token.stream_id,
                    generation=token.generation,
                    frame_id=token.frame_id,
                    publication_sequence=token.publication_sequence,
                )
            ),
            promotion_identity_sha256=transition.promoted_history_sha256,
            promotion_identity_basis="complete_current_visibility_epoch_v1",
            cross_gap_identity_claimed=False,
            visibility_epoch_frame_count=5,
            visibility_epoch_span_s=0.12,
            visibility_epoch_tokens=visibility_tokens,
            visibility_epoch_tracker_frame_sequences=tuple(range(5)),
        )

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
        transition_recovery_admission=recovery_admission,
        recovery_continuation_admission=recovery_continuation,
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


def _attempt8_close_alignment_crossing_values():
    snapshot = _snapshot(0, "track-0", 157)
    target = replace(
        _target(snapshot, "track-0"),
        normalized_x=0.009375,
        normalized_y_down=-0.0444444444444444,
        normalized_x_rate_s=0.0501729262260095,
        normalized_y_rate_down_s=0.218327071392714,
        log_scale=-0.7898068361159876,
        log_scale_rate_s=1.57677074528819,
    )
    output = VisualServoOutput(
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=-0.00457,
        thrust=0.21,
        corridor_frames=0,
        advance_enabled=False,
        next_gate_blend=0.0,
        horizontal_error=target.normalized_x,
        vertical_error_image_down=target.normalized_y_down,
        effective_horizontal_error=target.normalized_x,
        effective_vertical_error_image_down=target.normalized_y_down,
        effective_horizontal_rate_s=target.normalized_x_rate_s,
        effective_vertical_rate_down_s=target.normalized_y_rate_down_s,
        next_horizontal_error=None,
        next_vertical_error_image_down=None,
        horizontal_abs_error_delta=0.003125,
        vertical_abs_error_delta=-0.0111111111111112,
        brake_reason="aligning",
        yaw_envelope_limited=False,
    )
    admission = VisualApproachPassageAdmission(
        basis="tight-current-corridor-dwell-v1",
        current_gate_index=0,
        current_target=replace(target, log_scale=-1.6),
        camera_token=snapshot.latest_camera_token,
        tracker_frame_sequence=121,
        corridor_frames=25,
        preview_track_id="track-1",
        preview_blend=0.032425,
    )
    return target, output, admission


def test_attempt8_close_alignment_uses_retained_advance_crossing_proof():
    target, output, admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()

    assert course_stage._retained_crossing_observation_usable(
        target,
        output,
        tuning=tuning,
        limits=limits,
    )
    assert course_stage._crossing_anchor_basis(
        target,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=3,
        tuning=tuning,
        limits=limits,
    ) == course_stage.RETAINED_ADVANCE_CROSSING_BASIS


def test_attempt10_wire_projection_admits_only_the_last_safe_frame():
    target, output, admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    publication_158 = replace(
        target,
        received_monotonic_s=117777179027800 / 1_000_000_000.0,
        normalized_x=0.009374999999999911,
        normalized_y_down=-0.06666666666666665,
        normalized_x_rate_s=0.0580976279904446,
        normalized_y_rate_down_s=0.14205879374503383,
        log_scale=-0.8583646516890313,
        log_scale_rate_s=1.2940521542445622,
    )
    publication_159 = replace(
        target,
        received_monotonic_s=117777213551000 / 1_000_000_000.0,
        normalized_x=0.009374999999999911,
        normalized_y_down=-0.05555555555555558,
        normalized_x_rate_s=0.026143932595700067,
        normalized_y_rate_down_s=0.24094107666176479,
        log_scale=-0.8011509877843259,
        log_scale_rate_s=1.493812414455355,
    )
    projection_158 = course_stage._retained_crossing_wire_projection(
        publication_158,
        observation_monotonic_ns=117777179027800,
        wire_start_monotonic_ns=117777203077700,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )
    projection_159 = course_stage._retained_crossing_wire_projection(
        publication_159,
        observation_monotonic_ns=117777213551000,
        wire_start_monotonic_ns=117777234757300,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )

    assert limits.crossing_arm_min_log_scale == -0.80
    assert projection_158 is not None
    assert projection_159 is not None
    assert projection_158.observation_to_wire_s == pytest.approx(
        0.0240499
    )
    assert projection_158.projected_log_scale == pytest.approx(
        -0.827242826784665
    )
    assert projection_159.observation_to_wire_s == pytest.approx(
        0.0212063
    )
    assert projection_159.projected_log_scale == pytest.approx(
        -0.7694727535796613
    )
    assert course_stage._crossing_anchor_basis(
        publication_158,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=9,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection_158,
    ) is None
    assert course_stage._crossing_anchor_basis(
        publication_159,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=10,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection_159,
    ) == (
        course_stage.RETAINED_ADVANCE_WIRE_PROJECTED_CROSSING_BASIS
    )


def test_attempt11_wire_projection_admits_only_the_last_safe_frame():
    target, output, admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    publication_156 = replace(
        target,
        received_monotonic_s=118787645850200 / 1_000_000_000.0,
        normalized_x=0.009374999999999911,
        normalized_y_down=-0.061111111111111116,
        normalized_x_rate_s=0.09642996990836059,
        normalized_y_rate_down_s=0.15786849612897946,
        log_scale=-0.8783520348010684,
        log_scale_rate_s=1.2578718503651323,
    )
    publication_157 = replace(
        target,
        received_monotonic_s=118787681367200 / 1_000_000_000.0,
        normalized_x=0.009374999999999911,
        normalized_y_down=-0.05555555555555558,
        normalized_x_rate_s=0.04339348645876226,
        normalized_y_rate_down_s=0.15707161289555363,
        log_scale=-0.8270869031377459,
        log_scale_rate_s=1.3599106890803176,
    )
    projection_156 = course_stage._retained_crossing_wire_projection(
        publication_156,
        observation_monotonic_ns=118787645850200,
        wire_start_monotonic_ns=118787671633500,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )
    projection_157 = course_stage._retained_crossing_wire_projection(
        publication_157,
        observation_monotonic_ns=118787681367200,
        wire_start_monotonic_ns=118787703677900,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )

    assert projection_156 is not None
    assert projection_157 is not None
    assert projection_156.projected_log_scale == pytest.approx(
        -0.8459199475215491
    )
    assert projection_157.projected_log_scale == pytest.approx(
        -0.7967463437268817
    )
    assert course_stage._crossing_anchor_basis(
        publication_156,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=8,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection_156,
    ) is None
    assert course_stage._crossing_anchor_basis(
        publication_157,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=9,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection_157,
    ) == (
        course_stage.RETAINED_ADVANCE_WIRE_PROJECTED_CROSSING_BASIS
    )


def test_attempt17_yaw_soft_stop_retains_only_exact_projected_crossing():
    target, output, admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    target = replace(
        target,
        received_monotonic_s=136_468.312,
        normalized_x=-0.006249999999999978,
        normalized_y_down=-0.050000000000000044,
        normalized_x_rate_s=0.09978301265856465,
        normalized_y_rate_down_s=0.14650777115619515,
        log_scale=-0.8127695326567331,
        log_scale_rate_s=1.4305436485446084,
    )
    output = replace(
        output,
        advance_enabled=True,
        brake_reason=None,
    )
    projection = course_stage._retained_crossing_wire_projection(
        target,
        observation_monotonic_ns=136_468_312_000_000,
        wire_start_monotonic_ns=136_468_345_443_500,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )

    assert limits.retained_crossing_max_observation_to_wire_s == 0.035
    assert projection is not None
    assert projection.observation_to_wire_ns == 33_443_500
    assert projection.projected_log_scale == pytest.approx(
        -0.7649271461466315
    )
    assert projection.projected_normalized_x == pytest.approx(
        -0.0029129068161532707
    )
    assert projection.projected_normalized_y_down == pytest.approx(
        -0.04510026735533783
    )
    assert not course_stage._retained_crossing_observation_usable(
        target,
        output,
        tuning=tuning,
        limits=limits,
    )
    assert course_stage._retained_crossing_observation_usable(
        target,
        output,
        tuning=tuning,
        limits=limits,
        yaw_soft_stop_zeroed=True,
    )
    assert course_stage._crossing_anchor_basis(
        target,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=25,
        retained_crossing_dwell_frames=3,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection,
    ) is None
    assert course_stage._crossing_anchor_basis(
        target,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=25,
        retained_crossing_dwell_frames=3,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection,
        yaw_soft_stop_zeroed=True,
    ) == (
        course_stage.RETAINED_ADVANCE_WIRE_PROJECTED_CROSSING_BASIS
    )


def test_retained_projection_is_narrowed_by_raw_and_projected_scales():
    target, output, admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    target = replace(
        target,
        received_monotonic_s=1.0,
        log_scale=limits.retained_crossing_projection_min_log_scale,
        log_scale_rate_s=1.0,
    )
    projection = course_stage._retained_crossing_wire_projection(
        target,
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=1_030_000_000,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )
    assert projection is not None
    assert projection.projected_log_scale == pytest.approx(-0.80)
    assert course_stage._crossing_anchor_basis(
        target,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=3,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection,
    ) == (
        course_stage.RETAINED_ADVANCE_WIRE_PROJECTED_CROSSING_BASIS
    )

    below_floor = replace(
        target,
        log_scale=(
            limits.retained_crossing_projection_min_log_scale
            - 0.000001
        ),
        log_scale_rate_s=1.9,
    )
    below_floor_projection = (
        course_stage._retained_crossing_wire_projection(
            below_floor,
            observation_monotonic_ns=1_000_000_000,
            wire_start_monotonic_ns=1_030_000_000,
            tuning=tuning,
            limits=limits,
            abort_type=SafetyAbort,
        )
    )
    assert below_floor_projection is not None
    assert below_floor_projection.projected_log_scale > -0.80
    assert course_stage._crossing_anchor_basis(
        below_floor,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=3,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=below_floor_projection,
    ) is None

    projected_below = replace(
        target,
        log_scale_rate_s=(0.03 - 0.000001) / 0.03,
    )
    projected_below_proof = (
        course_stage._retained_crossing_wire_projection(
            projected_below,
            observation_monotonic_ns=1_000_000_000,
            wire_start_monotonic_ns=1_030_000_000,
            tuning=tuning,
            limits=limits,
            abort_type=SafetyAbort,
        )
    )
    assert projected_below_proof is not None
    assert projected_below_proof.projected_log_scale == pytest.approx(
        -0.800001
    )
    assert course_stage._crossing_anchor_basis(
        projected_below,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=24,
        retained_crossing_dwell_frames=3,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projected_below_proof,
    ) is None


def test_retained_projection_must_match_and_recompute_for_target():
    target, output, admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    target = replace(
        target,
        received_monotonic_s=1.0,
        log_scale=-0.81,
        log_scale_rate_s=1.0,
    )
    projection = course_stage._retained_crossing_wire_projection(
        target,
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=1_020_000_000,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )
    assert projection is not None

    for mismatched_target, mismatched_projection in (
        (replace(target, normalized_x=0.01), projection),
            (
                target,
                replace(projection, projected_log_scale=-0.78),
            ),
        (
            target,
            replace(projection, observation_to_wire_ns=19_000_000),
        ),
    ):
        assert course_stage._crossing_anchor_basis(
            mismatched_target,
            output,
            passage_admission=admission,
            current_gate_index=0,
            current_track_id="track-0",
            advance_command_count=24,
            retained_crossing_dwell_frames=3,
            tuning=tuning,
            limits=limits,
            retained_wire_projection=mismatched_projection,
        ) is None


def test_retained_wire_projection_is_bounded_and_fail_closed():
    target, _output, _admission = (
        _attempt8_close_alignment_crossing_values()
    )
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    target = replace(
        target,
        received_monotonic_s=1.0,
        log_scale=-0.81,
        log_scale_rate_s=1.0,
    )

    max_horizon_ns = round(
        limits.retained_crossing_max_observation_to_wire_s
        * 1_000_000_000
    )
    at_bound = course_stage._retained_crossing_wire_projection(
        target,
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=1_000_000_000 + max_horizon_ns,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )
    assert at_bound is not None
    assert at_bound.projected_log_scale == pytest.approx(-0.775)
    assert course_stage._retained_crossing_wire_projection(
        target,
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=(
            1_000_000_000 + max_horizon_ns + 1
        ),
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    ) is None
    assert course_stage._retained_crossing_wire_projection(
        replace(target, log_scale_rate_s=0.0),
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=1_020_000_000,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    ) is None
    assert course_stage._retained_crossing_wire_projection(
        replace(target, log_scale_rate_s=-0.1),
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=1_020_000_000,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    ) is None
    assert course_stage._retained_crossing_wire_projection(
        replace(
            target,
            normalized_x=tuning.horizontal_corridor,
            normalized_x_rate_s=tuning.stable_rate_norm_s,
        ),
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=1_032_000_000,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    ) is None


@pytest.mark.parametrize("wire_start", (None, True, -1))
def test_retained_wire_projection_rejects_invalid_wire_timing(wire_start):
    target, _output, _admission = (
        _attempt8_close_alignment_crossing_values()
    )

    with pytest.raises(SafetyAbort, match="exact wire timing"):
        course_stage._retained_crossing_wire_projection(
            target,
            observation_monotonic_ns=round(
                target.received_monotonic_s * 1_000_000_000
            ),
            wire_start_monotonic_ns=wire_start,
            tuning=default_visual_config().servo,
            limits=VisualCourseStageLimits(),
            abort_type=SafetyAbort,
        )


def test_retained_wire_projection_rejects_wire_before_observation():
    target, _output, _admission = (
        _attempt8_close_alignment_crossing_values()
    )
    target = replace(target, received_monotonic_s=1.0)

    with pytest.raises(SafetyAbort, match="predates observation"):
        course_stage._retained_crossing_wire_projection(
            target,
            observation_monotonic_ns=1_000_000_000,
            wire_start_monotonic_ns=999_999_999,
            tuning=default_visual_config().servo,
            limits=VisualCourseStageLimits(),
            abort_type=SafetyAbort,
        )


@pytest.mark.parametrize(
    "case",
    (
        "too_few_advance",
        "too_little_dwell",
        "missing_admission",
        "admission_gate_mismatch",
        "track_mismatch",
        "not_close",
        "retreating",
        "scale_brake",
        "horizontal_position",
        "vertical_position",
        "horizontal_rate",
        "vertical_rate",
        "clipped",
        "center_censored",
        "horizontal_censored",
        "vertical_censored",
        "ambiguous",
        "wrong_brake_reason",
        "planner_yaw_stop",
    ),
)
def test_retained_advance_crossing_proof_fails_closed(case):
    target, output, admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    advance_count = 24
    dwell = 3

    if case == "too_few_advance":
        advance_count = 2
    elif case == "too_little_dwell":
        dwell = 2
    elif case == "missing_admission":
        admission = None
    elif case == "admission_gate_mismatch":
        admission = replace(admission, current_gate_index=1)
    elif case == "track_mismatch":
        target = replace(target, track_id="other-track")
    elif case == "not_close":
        target = replace(
            target,
            log_scale=limits.crossing_arm_min_log_scale - 0.000001,
        )
    elif case == "retreating":
        target = replace(target, log_scale_rate_s=-0.000001)
    elif case == "scale_brake":
        target = replace(
            target,
            log_scale_rate_s=tuning.brake_scale_rate_s,
        )
    elif case == "horizontal_position":
        target = replace(
            target,
            normalized_x=tuning.horizontal_corridor + 0.000001,
        )
    elif case == "vertical_position":
        target = replace(
            target,
            normalized_y_down=tuning.vertical_corridor + 0.000001,
        )
    elif case == "horizontal_rate":
        target = replace(
            target,
            normalized_x_rate_s=tuning.stable_rate_norm_s + 0.000001,
        )
    elif case == "vertical_rate":
        target = replace(
            target,
            normalized_y_rate_down_s=(
                tuning.stable_rate_norm_s + 0.000001
            ),
        )
    elif case == "clipped":
        target = replace(target, clipped=True)
    elif case == "center_censored":
        target = replace(target, center_censored=True)
    elif case == "horizontal_censored":
        target = replace(target, horizontal_censored=True)
    elif case == "vertical_censored":
        target = replace(target, vertical_censored=True)
    elif case == "ambiguous":
        target = replace(target, ambiguous=True)
    elif case == "wrong_brake_reason":
        output = replace(output, brake_reason="scale_rate")
    elif case == "planner_yaw_stop":
        output = replace(
            output,
            brake_reason="segment_yaw_outward_soft_stop",
            yaw_envelope_limited=True,
        )

    assert course_stage._crossing_anchor_basis(
        target,
        output,
        passage_admission=admission,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=advance_count,
        retained_crossing_dwell_frames=dwell,
        tuning=tuning,
        limits=limits,
    ) is None


def test_current_advance_crossing_proof_is_preserved():
    target, output, _admission = _attempt8_close_alignment_crossing_values()
    tuning = default_visual_config().servo
    limits = VisualCourseStageLimits()
    output = replace(
        output,
        corridor_frames=tuning.required_corridor_frames,
        advance_enabled=True,
        brake_reason=None,
    )

    not_raw_close = replace(
        target,
        received_monotonic_s=1.0,
        log_scale=limits.crossing_arm_min_log_scale - 0.01,
        log_scale_rate_s=1.0,
    )
    projection = course_stage._retained_crossing_wire_projection(
        not_raw_close,
        observation_monotonic_ns=1_000_000_000,
        wire_start_monotonic_ns=1_020_000_000,
        tuning=tuning,
        limits=limits,
        abort_type=SafetyAbort,
    )
    assert projection is not None
    assert course_stage._crossing_anchor_basis(
        not_raw_close,
        output,
        passage_admission=None,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=3,
        retained_crossing_dwell_frames=0,
        tuning=tuning,
        limits=limits,
        retained_wire_projection=projection,
    ) is None
    assert course_stage._crossing_anchor_basis(
        replace(
            target,
            log_scale=limits.crossing_arm_min_log_scale,
        ),
        output,
        passage_admission=None,
        current_gate_index=0,
        current_track_id="track-0",
        advance_command_count=3,
        retained_crossing_dwell_frames=0,
        tuning=tuning,
        limits=limits,
    ) == course_stage.CURRENT_ADVANCE_CROSSING_BASIS


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


def test_retained_crossing_dwell_latches_only_after_accepted_wire_commands():
    class RetainedCrossingServo(_Servo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passage_observations = 0

        def observe(self, snapshot, *args, **kwargs):
            if snapshot.current_track.center_censored:
                raise VisualApproachCurrentGeometryUnavailable(
                    "authoritative current aperture is clipped or censored"
                )
            proposal = super().observe(snapshot, *args, **kwargs)
            if kwargs["mode"] is not VisualApproachMode.PASSAGE:
                return proposal
            self.passage_observations += 1
            if self.passage_observations <= 3:
                return SimpleNamespace(
                    current_target=replace(
                        proposal.current_target,
                        log_scale=-1.0,
                        log_scale_rate_s=0.2,
                    ),
                    servo_output=proposal.servo_output,
                    passage_admission=proposal.passage_admission,
                    mode=proposal.mode,
                )
            target, output, _admission = (
                _attempt8_close_alignment_crossing_values()
            )
            target = replace(
                target,
                track_id=proposal.current_target.track_id,
                frame_token=proposal.current_target.frame_token,
                received_monotonic_s=(
                    proposal.current_target.received_monotonic_s
                ),
                log_scale=-0.81,
                log_scale_rate_s=1.0,
            )
            return SimpleNamespace(
                current_target=target,
                servo_output=replace(output, yaw_rate_rad_s=0.0),
                passage_admission=proposal.passage_admission,
                mode=proposal.mode,
            )

    class RetainedLossHost(_Host):
        def _sample(self):
            super()._sample()
            navigation_count = sum(
                bool(kwargs.get("require_wire_receipt"))
                for _command, kwargs, _gate in self.commands
            )
            if (
                navigation_count >= 14
                and not self.race.race_finished
            ):
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                    visible=False,
                )
            elif (
                navigation_count >= 7
                and not self.race.race_finished
            ):
                coast_index = navigation_count - 7
                scales = (
                    0.802340,
                    0.853302,
                    0.912586,
                    0.984251,
                    1.0,
                    1.0,
                    1.0,
                )
                _set_censored_current(
                    self,
                    apparent_scale=scales[coast_index],
                    previous_apparent_scale=math.exp(-0.81),
                    clipping=(
                        FrameEdge(15)
                        if coast_index >= 4
                        else FrameEdge.TOP | FrameEdge.BOTTOM
                    ),
                )

        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(command, **kwargs)
            token = kwargs.get("wire_visual_token")
            if receipt is not None and token is not None:
                wire_start_ns = (
                    token.publication_sequence * 20_000_000
                    + 20_000_000
                )
                self._last_flight_command_started_ns = wire_start_ns
                receipt["call_start_monotonic_ns"] = wire_start_ns
                receipt["visual_receiver_authority"][
                    "call_start_monotonic_ns"
                ] = wire_start_ns
            if (
                command.roll_rate
                == command.pitch_rate
                == command.yaw_rate
                == command.thrust
                == 0.0
                and self.crossing_zero_count >= 2
            ):
                self.disable_credit = False
                self._advance_race()
            return receipt

    host = RetainedLossHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: RetainedCrossingServo(
            *args,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    anchor = result["segments"][0]["crossing_anchor"]
    assert result["race_finished"] is True
    assert anchor["basis"] == (
        course_stage
        .RETAINED_ADVANCE_WIRE_PROJECTED_CROSSING_BASIS
    )
    assert anchor["advance_command_count"] == 3
    assert anchor["observation_to_wire_ns"] == 20_000_000
    assert anchor["wire_projected_log_scale"] == pytest.approx(-0.79)
    assert anchor["wire_projected_normalized_x"] == pytest.approx(
        0.0103784585245201
    )
    assert anchor["wire_projected_normalized_y_down"] == pytest.approx(
        -0.0400779026492578
    )
    assert anchor["current_advance_enabled"] is False
    assert anchor["yaw_soft_stop_zeroed"] is False
    assert anchor["retained_crossing_dwell_frames"] == 3
    assert anchor["camera_token"]["publication_sequence"] == 17
    segment = result["segments"][0]
    assert segment["censored_passage_coast_fresh_frame_count"] == 7
    assert segment["censored_passage_coast_command_count"] == 7
    assert segment["passage_command_count"] == 13
    coast = segment["censored_passage_coast"]
    assert coast["basis"] == course_stage.CENSORED_PASSAGE_COAST_BASIS
    assert coast["anchor_camera_token"]["publication_sequence"] == 17
    assert coast["first_censored_camera_token"][
        "publication_sequence"
    ] == 18
    assert coast["last_censored_camera_token"][
        "publication_sequence"
    ] == 24
    assert coast["loss_camera_token"]["publication_sequence"] == 25
    assert segment["crossing_wait_zero_command_count"] == 2
    coast_commands = [
        (command, kwargs["wire_visual_token"])
        for command, kwargs, _gate in host.commands
        if kwargs.get("wire_visual_token") is not None
        and 18 <= kwargs["wire_visual_token"].publication_sequence <= 24
    ]
    assert [
        token.publication_sequence for _command, token in coast_commands
    ] == list(range(18, 25))
    assert all(
        command.yaw_rate == 0.0 and command.thrust > 0.0
        for command, _token_value in coast_commands
    )
    first_zero_index = next(
        index
        for index, (command, _kwargs, _gate) in enumerate(host.commands)
        if command.thrust == 0.0
    )
    assert first_zero_index > max(
        index
        for index, (_command, kwargs, _gate) in enumerate(host.commands)
        if kwargs.get("wire_visual_token") is not None
    )


class _CensoredPassageServo(_Servo):
    def observe(self, snapshot, *args, **kwargs):
        if snapshot.current_track.center_censored:
            raise VisualApproachCurrentGeometryUnavailable(
                "authoritative current aperture is clipped or censored"
            )
        return super().observe(snapshot, *args, **kwargs)


def _set_censored_current(
    host,
    *,
    apparent_scale,
    clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
    previous_apparent_scale=None,
):
    clipped = _snapshot(
        host.current_gate,
        host.current_track_id,
        host.sequence,
    )
    clipped.current_track.clipping = clipping
    clipped.current_track.center_censored = True
    clipped.current_track.apparent_scale = apparent_scale
    history = clipped.current_track.history
    previous_censored_scale = getattr(
        host,
        "_test_previous_censored_scale",
        previous_apparent_scale,
    )
    if previous_censored_scale is not None:
        history = (
            replace(
                history[0],
                apparent_scale=previous_censored_scale,
                clipping=clipping,
                center_censored=True,
            ),
            *history[1:],
        )
    clipped.current_track.history = (
        *history[:-1],
        replace(
            history[-1],
            apparent_scale=apparent_scale,
            clipping=clipped.current_track.clipping,
            center_censored=True,
        ),
    )
    host._test_previous_censored_scale = apparent_scale
    host.visual_gate_graph.latest_snapshot = clipped


def _set_bottom_transition_successor(
    host,
    *,
    apparent_scale,
    clipping,
    center_censored=True,
):
    snapshot = _snapshot(
        host.current_gate,
        host.current_track_id,
        host.sequence,
    )
    history = snapshot.current_track.history
    prior_scale = getattr(host, "_test_previous_censored_scale", 0.80)
    previous = replace(
        history[0],
        apparent_scale=prior_scale,
        clipping=FrameEdge.BOTTOM,
        center_censored=True,
    )
    current = replace(
        history[-1],
        apparent_scale=apparent_scale,
        clipping=clipping,
        center_censored=center_censored,
    )
    snapshot.current_track.history = (previous, current)
    snapshot.current_track.clipping = clipping
    snapshot.current_track.center_censored = center_censored
    snapshot.current_track.apparent_scale = apparent_scale
    host._test_previous_censored_scale = apparent_scale
    host.visual_gate_graph.latest_snapshot = snapshot


def test_bottom_transition_requires_full_censor_then_accepts_credit():
    class BottomThenFullCreditHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.bottom_command = None
            self.full_command = None
            self.anchor_seen = False

        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if not segments or segments[0]["crossing_anchor"] is None:
                return
            if not self.anchor_seen:
                self.anchor_seen = True
                return
            transition = segments[0]["bottom_censored_transition"]
            if transition is None:
                _set_censored_current(
                    self,
                    apparent_scale=0.80,
                    clipping=FrameEdge.BOTTOM,
                )
            elif (
                transition["outcome"]
                == "awaiting_full_vertical_censorship"
            ):
                _set_bottom_transition_successor(
                    self,
                    apparent_scale=0.82,
                    clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
                )

        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(command, **kwargs)
            token = kwargs.get("wire_visual_token")
            snapshot = self.visual_gate_graph.latest_snapshot
            if token is not None and snapshot.current_track.center_censored:
                if snapshot.current_track.clipping == FrameEdge.BOTTOM:
                    self.bottom_command = command
                elif (
                    snapshot.current_track.clipping
                    & (FrameEdge.TOP | FrameEdge.BOTTOM)
                    == (FrameEdge.TOP | FrameEdge.BOTTOM)
                ):
                    self.full_command = command
                    self.disable_credit = False
                    self._advance_race()
            return receipt

    host = BottomThenFullCreditHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            passage_preview_retire_once=True,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    transition = segment["bottom_censored_transition"]
    assert result["race_finished"] is True
    assert segment["next_preview_retired"] is True
    assert segment["bottom_censored_transition_count"] == 1
    assert transition["outcome"] == "full_vertical_censorship_confirmed"
    assert transition["bottom_censored_camera_token"][
        "publication_sequence"
    ] + 1 == transition["full_vertical_censor_camera_token"][
        "publication_sequence"
    ]
    assert segment["censored_passage_coast_fresh_frame_count"] == 2
    assert segment["censored_passage_coast_command_count"] == 2
    assert segment["advance_command_count"] == (
        segment["crossing_anchor"]["advance_command_count"]
    )
    assert segment["censored_passage_coast"]["bottom_transition_pending"] is False
    assert host.bottom_command is not None
    assert host.full_command is not None
    assert host.bottom_command.yaw_rate == 0.0
    assert host.full_command.yaw_rate == 0.0
    assert host.bottom_command.thrust == pytest.approx(
        host.full_command.thrust
    )
    assert segment["crossing_wait_zero_command_count"] == 0


@pytest.mark.parametrize("credit_timing", ("before_wire", "after_wire"))
def test_bottom_transition_accepts_authoritative_credit(credit_timing):
    class BottomCreditHost(_Host):
        anchor_seen = False
        credited = False
        bottom_command = None

        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if not segments or segments[0]["crossing_anchor"] is None:
                return
            if not self.anchor_seen:
                self.anchor_seen = True
                return
            if segments[0]["bottom_censored_transition"] is None:
                _set_censored_current(
                    self,
                    apparent_scale=0.80,
                    clipping=FrameEdge.BOTTOM,
                )

        def _assert_visual_receiver_token_current(self, expected_token):
            token = super()._assert_visual_receiver_token_current(
                expected_token
            )
            snapshot = self.visual_gate_graph.latest_snapshot
            if (
                snapshot.current_track.clipping == FrameEdge.BOTTOM
                and not self.credited
                and credit_timing == "before_wire"
            ):
                self.credited = True
                self.disable_credit = False
                self._advance_race()
            return token

        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(command, **kwargs)
            snapshot = self.visual_gate_graph.latest_snapshot
            if (
                snapshot.current_track.clipping == FrameEdge.BOTTOM
                and not self.credited
                and credit_timing == "after_wire"
            ):
                self.bottom_command = command
                self.credited = True
                self.disable_credit = False
                self._advance_race()
            return receipt

    host = BottomCreditHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            passage_preview_retire_once=True,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    transition = segment["bottom_censored_transition"]
    assert result["race_finished"] is True
    assert host.credited is True
    assert segment["bottom_censored_transition_count"] == 1
    assert transition["outcome"] == (
        "authoritative_credit_before_full_censorship"
    )
    assert segment["censored_passage_coast_fresh_frame_count"] == 1
    expected_coast_commands = 0 if credit_timing == "before_wire" else 1
    assert (
        segment["censored_passage_coast_command_count"]
        == expected_coast_commands
    )
    assert segment["advance_command_count"] == (
        segment["crossing_anchor"]["advance_command_count"]
    )
    assert segment["censored_passage_coast"]["bottom_transition_pending"] is False
    bottom_token = transition["bottom_censored_camera_token"][
        "publication_sequence"
    ]
    bottom_wire_commands = [
        command
        for command, kwargs, _gate in host.commands
        if kwargs.get("wire_visual_token") is not None
        and kwargs["wire_visual_token"].publication_sequence == bottom_token
    ]
    assert len(bottom_wire_commands) == expected_coast_commands
    if credit_timing == "after_wire":
        assert host.bottom_command is bottom_wire_commands[0]
        assert host.bottom_command.yaw_rate == 0.0


def test_bottom_transition_supersession_can_land_only_on_full_censor():
    class SupersededBottomToFullHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.anchor_seen = False
            self.superseded = False
            self.reuse_receiver_snapshot = False
            self.full_command = None

        def _sample(self):
            if self.reuse_receiver_snapshot:
                self.reuse_receiver_snapshot = False
                return
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if not segments or segments[0]["crossing_anchor"] is None:
                return
            if not self.anchor_seen:
                self.anchor_seen = True
                return
            if segments[0]["bottom_censored_transition"] is None:
                _set_censored_current(
                    self,
                    apparent_scale=0.80,
                    clipping=FrameEdge.BOTTOM,
                )

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            snapshot = self.visual_gate_graph.latest_snapshot
            if (
                expected is not None
                and snapshot.current_track.clipping == FrameEdge.BOTTOM
                and not self.superseded
            ):
                receiver = _token(expected.publication_sequence + 1)
                self.superseded = True
                self.sequence = receiver.publication_sequence
                _set_bottom_transition_successor(
                    self,
                    apparent_scale=0.82,
                    clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
                )
                self.reuse_receiver_snapshot = True
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
            receipt = await super()._send_flight_command(command, **kwargs)
            if (
                expected is not None
                and snapshot.current_track.clipping
                & (FrameEdge.TOP | FrameEdge.BOTTOM)
                == (FrameEdge.TOP | FrameEdge.BOTTOM)
            ):
                self.full_command = command
                self.disable_credit = False
                self._advance_race()
            return receipt

    host = SupersededBottomToFullHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            passage_preview_retire_once=True,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    assert result["race_finished"] is True
    assert host.superseded is True
    assert segment["superseded_proposal_count"] == 1
    assert segment["bottom_censored_transition"]["outcome"] == (
        "full_vertical_censorship_confirmed"
    )
    assert segment["censored_passage_coast_fresh_frame_count"] == 2
    assert segment["censored_passage_coast_command_count"] == 1
    assert host.full_command is not None
    assert host.full_command.yaw_rate == 0.0


@pytest.mark.parametrize(
    "successor",
    (
        "bottom",
        "top",
        "horizontal",
        "uncensored",
        "loss",
        "scale_regression",
    ),
)
def test_bottom_transition_refuses_any_non_full_successor(successor):
    class InvalidBottomSuccessorHost(_Host):
        anchor_seen = False

        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if not segments or segments[0]["crossing_anchor"] is None:
                return
            if not self.anchor_seen:
                self.anchor_seen = True
                return
            transition = segments[0]["bottom_censored_transition"]
            if transition is None:
                _set_censored_current(
                    self,
                    apparent_scale=0.80,
                    clipping=FrameEdge.BOTTOM,
                )
                return
            if (
                transition["outcome"]
                != "awaiting_full_vertical_censorship"
            ):
                return
            if successor == "bottom":
                _set_bottom_transition_successor(
                    self,
                    apparent_scale=0.82,
                    clipping=FrameEdge.BOTTOM,
                )
            elif successor == "top":
                _set_bottom_transition_successor(
                    self,
                    apparent_scale=0.82,
                    clipping=FrameEdge.TOP,
                )
            elif successor == "horizontal":
                _set_bottom_transition_successor(
                    self,
                    apparent_scale=0.82,
                    clipping=FrameEdge.BOTTOM | FrameEdge.LEFT,
                )
            elif successor == "uncensored":
                return
            elif successor == "loss":
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                    visible=False,
                )
            elif successor == "scale_regression":
                _set_bottom_transition_successor(
                    self,
                    apparent_scale=0.79,
                    clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
                )

    host = InvalidBottomSuccessorHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            passage_preview_retire_once=True,
            **kwargs,
        ),
    )

    with pytest.raises(SafetyAbort, match="bottom-censored transition"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["bottom_censored_transition_count"] == 1
    assert segment["censored_passage_coast_fresh_frame_count"] == 1
    assert segment["censored_passage_coast_command_count"] == 1
    assert segment["crossing_wait_zero_command_count"] == 0


def test_visible_censored_passage_coast_expires_before_a_ninth_frame():
    class NeverLostHost(_Host):
        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if segments and segments[0]["crossing_anchor"] is not None:
                coast_count = segments[0][
                    "censored_passage_coast_command_count"
                ]
                _set_censored_current(
                    self,
                    apparent_scale=0.80 + 0.01 * coast_count,
                )

    host = NeverLostHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="censored passage coast expired",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["censored_passage_coast_fresh_frame_count"] == 8
    assert segment["censored_passage_coast_command_count"] == 8
    assert segment["crossing_wait_zero_command_count"] == 0
    assert all(command.thrust > 0.0 for command, _kwargs, _gate in host.commands)


def test_censored_passage_coast_deadline_applies_during_duplicate_frames():
    class FrozenCensoredHost(_Host):
        def _sample(self):
            segments = self._visual_course_summary.get("segments", ())
            if (
                segments
                and segments[0][
                    "censored_passage_coast_command_count"
                ]
                >= 1
            ):
                return
            super()._sample()
            if segments and segments[0]["crossing_anchor"] is not None:
                _set_censored_current(self, apparent_scale=0.80)

    host = FrozenCensoredHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="censored passage coast expired",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["censored_passage_coast_fresh_frame_count"] == 1
    assert segment["censored_passage_coast_command_count"] == 1
    assert segment["crossing_wait_zero_command_count"] == 0


def test_censored_passage_coast_cannot_wait_past_deadline_before_wire():
    class DelayedCoastSlotHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.delayed_coast_slot = False
            self.commands_before_delay = None

        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if segments and segments[0]["crossing_anchor"] is not None:
                _set_censored_current(self, apparent_scale=0.80)

        async def _wait_for_next_flight_command_slot(self):
            ready = await super()._wait_for_next_flight_command_slot()
            snapshot = self.visual_gate_graph.latest_snapshot
            if (
                snapshot.current_track.center_censored
                and not self.delayed_coast_slot
            ):
                self.commands_before_delay = len(self.commands)
                self.clock += (
                    course_stage.DEFAULT_VISUAL_COURSE_LIMITS
                    .censored_passage_coast_max_duration_s
                    + 0.001
                )
                self.delayed_coast_slot = True
            return ready

    host = DelayedCoastSlotHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="censored passage coast expired",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.delayed_coast_slot is True
    assert host.commands_before_delay is not None
    assert len(host.commands) == host.commands_before_delay
    segment = host._visual_course_summary["segments"][0]
    assert segment["censored_passage_coast_fresh_frame_count"] == 1
    assert segment["censored_passage_coast_command_count"] == 0
    assert segment["crossing_wait_zero_command_count"] == 0


def test_first_censored_coast_frame_cannot_regress_from_last_clean_scale():
    class RegressingScaleHost(_Host):
        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if segments and segments[0]["crossing_anchor"] is not None:
                _set_censored_current(self, apparent_scale=0.60)

    host = RegressingScaleHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(SafetyAbort, match="visual authority refused"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["censored_passage_coast_fresh_frame_count"] == 0
    assert segment["censored_passage_coast_command_count"] == 0
    assert segment["crossing_wait_zero_command_count"] == 0


def test_gate0_coast_recomputes_rates_and_accepts_credit_without_post_credit_send():
    class CreditDuringCoastHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.coast_command = None

        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if segments and segments[0]["crossing_anchor"] is not None:
                _set_attitude(self, pitch=0.05)
                _set_censored_current(self, apparent_scale=0.80)

        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(command, **kwargs)
            snapshot = self.visual_gate_graph.latest_snapshot
            if (
                kwargs.get("wire_visual_token") is not None
                and snapshot.current_track.center_censored
                and not self.race.race_finished
            ):
                self.coast_command = command
                self.disable_credit = False
                self._advance_race()
            return receipt

    host = CreditDuringCoastHost(
        initial_gate=0,
        finish_gate=0,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)

    def attitude_rate(
        estimate,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
    ):
        roll, pitch, _yaw = estimate.orientation.to_euler()
        return AttitudeRateCommand(
            target_roll_rad - roll,
            target_pitch_rad - pitch,
            0.0,
            thrust,
        )

    runtime = replace(
        runtime,
        attitude_rate_command=attitude_rate,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    anchor = segment["crossing_anchor"]
    coast = segment["censored_passage_coast"]
    assert result["race_finished"] is True
    assert segment["censored_passage_coast_command_count"] == 1
    assert segment["crossing_wait_zero_command_count"] == 0
    assert host.coast_command is not None
    assert host.coast_command.pitch_rate == pytest.approx(-0.155)
    assert host.coast_command.pitch_rate != pytest.approx(
        anchor["command"]["pitch_rate"]
    )
    assert host.coast_command.yaw_rate == 0.0
    assert coast["thrust"] == pytest.approx(
        anchor["command"]["thrust"]
    )
    assert coast["thrust"] != pytest.approx(
        host.visual_config.servo.advance_thrust
    )
    commands_at_credit = len(host.commands)
    assert commands_at_credit == result["visual_navigation_command_count"]
    assert segment["post_credit_zero_command_count"] == 0


def test_censored_coast_accepts_atomic_credit_before_wire():
    class CreditAtCoastPrecheckHost(_Host):
        credited = False

        def _sample(self):
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if segments and segments[0]["crossing_anchor"] is not None:
                _set_censored_current(self, apparent_scale=0.80)

        def _assert_visual_receiver_token_current(self, expected_token):
            token = super()._assert_visual_receiver_token_current(
                expected_token
            )
            if (
                self.visual_gate_graph.latest_snapshot
                .current_track.center_censored
                and not self.credited
            ):
                self.credited = True
                self.disable_credit = False
                self._advance_race()
            return token

    host = CreditAtCoastPrecheckHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    assert result["race_finished"] is True
    assert host.credited is True
    assert segment["censored_passage_coast_fresh_frame_count"] == 1
    assert segment["censored_passage_coast_command_count"] == 0
    assert segment["crossing_wait_zero_command_count"] == 0
    censored_token = segment["censored_passage_coast"][
        "first_censored_camera_token"
    ]["publication_sequence"]
    assert all(
        kwargs.get("wire_visual_token") is None
        or kwargs["wire_visual_token"].publication_sequence
        != censored_token
        for _command, kwargs, _gate in host.commands
    )


def test_censored_coast_supersession_preserves_observed_loss_lineage():
    class SupersededLossHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.superseded = False
            self.expected = None
            self.receiver = None

        def _sample(self):
            if self.superseded:
                return
            super()._sample()
            segments = self._visual_course_summary.get("segments", ())
            if segments and segments[0]["crossing_anchor"] is not None:
                _set_censored_current(self, apparent_scale=0.80)

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if (
                expected is not None
                and self.visual_gate_graph.latest_snapshot
                .current_track.center_censored
                and not self.superseded
            ):
                receiver = _token(expected.publication_sequence + 1)
                self.superseded = True
                self.expected = expected
                self.receiver = receiver
                self.sequence = receiver.publication_sequence
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                    visible=False,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
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
                and self.crossing_zero_count >= 2
            ):
                self.disable_credit = False
                self._advance_race()
            return receipt

    host = SupersededLossHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _CensoredPassageServo(
            *args,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    assert result["race_finished"] is True
    assert host.superseded is True
    assert segment["superseded_proposal_count"] == 1
    assert segment["censored_passage_coast_fresh_frame_count"] == 1
    assert segment["censored_passage_coast_command_count"] == 0
    assert segment["censored_passage_coast"][
        "last_censored_camera_token"
    ] == asdict(host.expected)
    assert segment["censored_passage_coast"][
        "loss_camera_token"
    ] == asdict(host.receiver)
    assert segment["crossing_wait_zero_command_count"] == 2


@pytest.mark.parametrize(
    "mutation",
    (
        "top_only",
        "wrong_role",
        "ambiguous",
        "visible_missed",
        "publication_gap",
        "scale_regression",
        "not_center_censored",
        "low_association_confidence",
        "low_track_confidence",
    ),
)
def test_censored_passage_coast_admission_fails_closed(mutation):
    snapshot = _snapshot(4, "track-4", 21)
    snapshot.current_track.clipping = (
        FrameEdge.TOP | FrameEdge.BOTTOM
    )
    snapshot.current_track.center_censored = True
    snapshot.current_track.apparent_scale = 0.80
    history = snapshot.current_track.history
    snapshot.current_track.history = (
        *history[:-1],
        replace(
            history[-1],
            apparent_scale=0.80,
            clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
            center_censored=True,
        ),
    )
    previous = _token(20)
    anchor = _token(19)
    authority = course_stage._CensoredPassageCoastAuthority(
        gate_index=4,
        track_id="track-4",
        anchor_camera_token=anchor,
        target_roll_rad=0.0,
        target_pitch_rad=-0.105,
        thrust=0.295,
    )

    if mutation == "top_only":
        snapshot.current_track.clipping = FrameEdge.TOP
    elif mutation == "wrong_role":
        snapshot.current_track.role = VisualTrackRole.NEXT
    elif mutation == "ambiguous":
        snapshot.current_track.ambiguous = True
    elif mutation == "visible_missed":
        snapshot.current_track.missed_frame_count = 1
    elif mutation == "publication_gap":
        previous = _token(18)
    elif mutation == "scale_regression":
        snapshot.current_track.apparent_scale = 0.75
    elif mutation == "not_center_censored":
        snapshot.current_track.center_censored = False
    elif mutation == "low_association_confidence":
        latest = snapshot.current_track.history[-1]
        assert latest.accepted_association is not None
        snapshot.current_track.association_confidence = 0.09
        snapshot.current_track.history = (
            *snapshot.current_track.history[:-1],
            replace(
                latest,
                association_confidence=0.09,
                accepted_association=replace(
                    latest.accepted_association,
                    confidence=0.09,
                ),
            ),
        )
    elif mutation == "low_track_confidence":
        snapshot.current_track.confidence = 0.09

    assert not course_stage._censored_passage_coast_eligible(
        snapshot,
        current_gate_index=4,
        current_track_id="track-4",
        crossing_anchor_token=anchor,
        authority=authority,
        previous_visible_token=previous,
        previous_apparent_scale=(
            0.80 if mutation == "scale_regression" else None
        ),
        minimum_apparent_scale=math.exp(-0.80),
    )


def test_censored_passage_coast_accepts_retained_clean_epoch_suffix():
    snapshot = _snapshot(4, "track-4", 21)
    clipping = FrameEdge.TOP | FrameEdge.BOTTOM
    snapshot.current_track.clipping = clipping
    snapshot.current_track.center_censored = True
    snapshot.current_track.apparent_scale = 0.80
    snapshot.current_track.consecutive_frame_count = 300
    history = snapshot.current_track.history
    snapshot.current_track.history = (
        *history[:-1],
        replace(
            history[-1],
            apparent_scale=0.80,
            clipping=clipping,
            center_censored=True,
        ),
    )
    anchor = _token(19)
    authority = course_stage._CensoredPassageCoastAuthority(
        gate_index=4,
        track_id="track-4",
        anchor_camera_token=anchor,
        target_roll_rad=0.0,
        target_pitch_rad=-0.105,
        thrust=0.295,
    )

    assert course_stage._censored_passage_coast_eligible(
        snapshot,
        current_gate_index=4,
        current_track_id="track-4",
        crossing_anchor_token=anchor,
        authority=authority,
        previous_visible_token=_token(20),
        previous_apparent_scale=math.exp(-0.50),
        minimum_apparent_scale=math.exp(-0.80),
    )


def _attempt18_bottom_transition_values():
    track_id = "track-0"
    anchor = _token(157)
    previous_token = _token(163)
    current_token = _token(164)
    previous_sample = replace(
        _history_sample(
            track_id,
            previous_token,
            apparent_scale=0.706124005519908,
        ),
        center_norm=(0.00625, 0.016666666666666607),
    )
    current_sample = _history_sample(
        track_id,
        current_token,
        previous_token=previous_token,
        apparent_scale=0.7767118191253388,
        clipping=FrameEdge.BOTTOM,
        center_censored=True,
    )
    track = VisualTrack(
        track_id=track_id,
        first_token=previous_token,
        latest_token=current_token,
        center_norm=current_sample.center_norm,
        bbox_norm=current_sample.bbox_norm,
        apparent_scale=current_sample.apparent_scale,
        center_velocity_norm_s=(0.0, 0.0),
        log_scale_rate_s=0.0,
        confidence=0.90,
        association_confidence=0.80,
        consecutive_frame_count=160,
        total_observation_count=160,
        missed_frame_count=0,
        clipping=FrameEdge.BOTTOM,
        center_censored=True,
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=0,
        authority_race_status_sequence=10,
        authority_race_status_boot_ms=2_000,
        ambiguous=False,
        visible=True,
        history=(previous_sample, current_sample),
    )
    snapshot = SimpleNamespace(
        tracker_frame_sequence=164,
        latest_camera_token=current_token,
        current_gate_index=0,
        current_track_id=track_id,
        current_track=track,
        authority_usable=True,
        race_finished=False,
    )
    target = _target(_snapshot(0, track_id, 157), track_id)
    admission = VisualApproachPassageAdmission(
        basis="tight-current-corridor-dwell-v1",
        current_gate_index=0,
        current_target=target,
        camera_token=anchor,
        tracker_frame_sequence=157,
        corridor_frames=26,
        preview_track_id="track-1",
        preview_blend=0.25,
    )
    authority = course_stage._CensoredPassageCoastAuthority(
        gate_index=0,
        track_id=track_id,
        anchor_camera_token=anchor,
        target_roll_rad=0.0,
        target_pitch_rad=-0.105,
        thrust=0.295,
    )
    return snapshot, admission, authority, previous_token


def test_attempt18_bottom_transition_is_one_exact_pending_frame():
    snapshot, admission, authority, previous = (
        _attempt18_bottom_transition_values()
    )

    assert course_stage._bottom_censored_passage_transition_eligible(
        snapshot,
        current_gate_index=0,
        current_track_id="track-0",
        crossing_anchor_token=authority.anchor_camera_token,
        authority=authority,
        previous_visible_token=previous,
        previous_apparent_scale=0.706124005519908,
        minimum_apparent_scale=math.exp(-0.80),
        passage_admission=admission,
        next_preview_retired=True,
        tuning=default_visual_config().servo,
    )
    assert not course_stage._censored_passage_coast_eligible(
        snapshot,
        current_gate_index=0,
        current_track_id="track-0",
        crossing_anchor_token=authority.anchor_camera_token,
        authority=authority,
        previous_visible_token=previous,
        previous_apparent_scale=0.706124005519908,
        minimum_apparent_scale=math.exp(-0.80),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "preview_active",
        "top_only",
        "both_edges",
        "horizontal_edge",
        "nonadjacent",
        "prior_not_near",
        "scale_not_growing",
        "prior_clipped",
        "prior_center_outside",
        "missed_association",
        "ambiguous",
        "wrong_identity",
    ),
)
def test_bottom_transition_admission_fails_closed(mutation):
    snapshot, admission, authority, previous = (
        _attempt18_bottom_transition_values()
    )
    previous_scale = 0.706124005519908
    next_preview_retired = True
    track = snapshot.current_track
    history = track.history

    if mutation == "preview_active":
        next_preview_retired = False
    elif mutation == "top_only":
        track = replace(
            track,
            clipping=FrameEdge.TOP,
            history=(
                history[0],
                replace(history[1], clipping=FrameEdge.TOP),
            ),
        )
    elif mutation == "both_edges":
        clipping = FrameEdge.TOP | FrameEdge.BOTTOM
        track = replace(
            track,
            clipping=clipping,
            history=(
                history[0],
                replace(history[1], clipping=clipping),
            ),
        )
    elif mutation == "horizontal_edge":
        clipping = FrameEdge.BOTTOM | FrameEdge.LEFT
        track = replace(
            track,
            clipping=clipping,
            history=(
                history[0],
                replace(history[1], clipping=clipping),
            ),
        )
    elif mutation == "nonadjacent":
        previous = _token(162)
    elif mutation == "prior_not_near":
        previous_scale = 0.55
        track = replace(
            track,
            history=(
                replace(history[0], apparent_scale=0.55),
                history[1],
            ),
        )
    elif mutation == "scale_not_growing":
        track = replace(
            track,
            apparent_scale=previous_scale,
            history=(
                history[0],
                replace(history[1], apparent_scale=previous_scale),
            ),
        )
    elif mutation == "prior_clipped":
        track = replace(
            track,
            history=(
                replace(history[0], clipping=FrameEdge.BOTTOM),
                history[1],
            ),
        )
    elif mutation == "prior_center_outside":
        track = replace(
            track,
            history=(
                replace(history[0], center_norm=(0.160001, 0.0)),
                history[1],
            ),
        )
    elif mutation == "missed_association":
        association = history[1].accepted_association
        assert association is not None
        track = replace(
            track,
            history=(
                history[0],
                replace(
                    history[1],
                    accepted_association=replace(
                        association,
                        missed_frame_count_before_association=1,
                    ),
                ),
            ),
        )
    elif mutation == "ambiguous":
        track = replace(track, ambiguous=True)
    elif mutation == "wrong_identity":
        snapshot.current_track_id = "other-track"
    snapshot.current_track = track

    assert not course_stage._bottom_censored_passage_transition_eligible(
        snapshot,
        current_gate_index=0,
        current_track_id="track-0",
        crossing_anchor_token=authority.anchor_camera_token,
        authority=authority,
        previous_visible_token=previous,
        previous_apparent_scale=previous_scale,
        minimum_apparent_scale=math.exp(-0.80),
        passage_admission=admission,
        next_preview_retired=next_preview_retired,
        tuning=default_visual_config().servo,
    )


def test_censored_passage_coast_rejects_same_id_reacquisition():
    track_id = "track-4"
    previous = _token(19)
    current = _token(21)
    clipping = FrameEdge.TOP | FrameEdge.BOTTOM
    previous_sample = _history_sample(
        track_id,
        previous,
        apparent_scale=0.78,
    )
    reacquired_sample = _history_sample(
        track_id,
        current,
        previous_token=previous,
        missed_frames_before_association=1,
        apparent_scale=0.80,
        clipping=clipping,
        center_censored=True,
    )
    track = VisualTrack(
        track_id=track_id,
        first_token=previous,
        latest_token=current,
        center_norm=reacquired_sample.center_norm,
        bbox_norm=reacquired_sample.bbox_norm,
        apparent_scale=reacquired_sample.apparent_scale,
        center_velocity_norm_s=(0.0, 0.0),
        log_scale_rate_s=0.0,
        confidence=0.90,
        association_confidence=0.80,
        consecutive_frame_count=1,
        total_observation_count=2,
        missed_frame_count=0,
        clipping=clipping,
        center_censored=True,
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=4,
        authority_race_status_sequence=10,
        authority_race_status_boot_ms=2_000,
        ambiguous=False,
        visible=True,
        history=(previous_sample, reacquired_sample),
    )
    snapshot = SimpleNamespace(
        tracker_frame_sequence=21,
        latest_camera_token=current,
        current_gate_index=4,
        current_track_id=track_id,
        current_track=track,
        authority_usable=True,
        race_finished=False,
    )
    anchor = _token(18)
    authority = course_stage._CensoredPassageCoastAuthority(
        gate_index=4,
        track_id=track_id,
        anchor_camera_token=anchor,
        target_roll_rad=0.0,
        target_pitch_rad=-0.105,
        thrust=0.295,
    )

    assert (
        reacquired_sample.accepted_association
        .missed_frame_count_before_association
        == 1
    )
    assert not course_stage._censored_passage_coast_eligible(
        snapshot,
        current_gate_index=4,
        current_track_id=track_id,
        crossing_anchor_token=anchor,
        authority=authority,
        previous_visible_token=previous,
        previous_apparent_scale=0.78,
        minimum_apparent_scale=math.exp(-0.80),
    )
    snapshot.current_track = replace(
        track,
        consecutive_frame_count=2,
    )
    assert not course_stage._censored_passage_coast_eligible(
        snapshot,
        current_gate_index=4,
        current_track_id=track_id,
        crossing_anchor_token=anchor,
        authority=authority,
        previous_visible_token=previous,
        previous_apparent_scale=0.78,
        minimum_apparent_scale=math.exp(-0.80),
    )


@pytest.mark.parametrize(
    "case",
    (
        "nonadjacent_previous",
        "nonadjacent_receiver",
        "late_hold",
        "repeated_supersession",
        "incomplete_dwell",
        "track_mismatch",
        "token_mismatch",
        "unsafe_geometry",
        "advance_enabled",
    ),
)
def test_retained_crossing_supersession_is_exact_and_fail_closed(case):
    previous = _token(155)
    expected = _token(156)
    receiver = _token(157)
    snapshot = _snapshot(0, "track-0", 156)
    target, output, _admission = (
        _attempt8_close_alignment_crossing_values()
    )
    target = replace(
        target,
        track_id="track-0",
        frame_token=_target(snapshot, "track-0").frame_token,
        received_monotonic_s=3.12,
        normalized_x=-0.009375,
        normalized_y_down=-0.05555555555555558,
        normalized_x_rate_s=0.06504730414345428,
        normalized_y_rate_down_s=0.12021386622797031,
        log_scale=-0.8341951068701171,
        log_scale_rate_s=1.3715516000267416,
    )
    supersession = course_stage._SupersededVisualProposal(
        expected_camera_token=expected,
        receiver_camera_token=receiver,
        held_previous_command_s=0.031,
        consecutive_count=1,
    )
    dwell = 8
    current_track_id = "track-0"

    if case == "nonadjacent_previous":
        previous = _token(154)
    elif case == "nonadjacent_receiver":
        supersession = replace(
            supersession,
            receiver_camera_token=_token(158),
        )
    elif case == "late_hold":
        supersession = replace(
            supersession,
            held_previous_command_s=0.032000001,
        )
    elif case == "repeated_supersession":
        supersession = replace(supersession, consecutive_count=2)
    elif case == "incomplete_dwell":
        dwell = 2
    elif case == "track_mismatch":
        current_track_id = "other-track"
    elif case == "token_mismatch":
        target = replace(
            target,
            frame_token=_target(
                _snapshot(0, "track-0", 155),
                "track-0",
            ).frame_token,
        )
    elif case == "unsafe_geometry":
        target = replace(target, normalized_x=0.160001)
    elif case == "advance_enabled":
        output = replace(
            output,
            advance_enabled=True,
            brake_reason=None,
        )

    assert not course_stage._retained_crossing_supersession_usable(
        target,
        output,
        supersession,
        last_accepted_camera_token=previous,
        current_track_id=current_track_id,
        retained_crossing_dwell_frames=dwell,
        tuning=default_visual_config().servo,
        limits=VisualCourseStageLimits(),
    )


def test_attempt18_safe_adjacent_supersession_preserves_completed_dwell():
    previous = _token(155)
    expected = _token(156)
    receiver = _token(157)
    snapshot = _snapshot(0, "track-0", 156)
    target, output, _admission = (
        _attempt8_close_alignment_crossing_values()
    )
    target = replace(
        target,
        frame_token=_target(snapshot, "track-0").frame_token,
        received_monotonic_s=3.12,
        normalized_x=-0.009375,
        normalized_y_down=-0.05555555555555558,
        normalized_x_rate_s=0.06504730414345428,
        normalized_y_rate_down_s=0.12021386622797031,
        log_scale=-0.8341951068701171,
        log_scale_rate_s=1.3715516000267416,
    )
    supersession = course_stage._SupersededVisualProposal(
        expected_camera_token=expected,
        receiver_camera_token=receiver,
        held_previous_command_s=0.031,
        consecutive_count=1,
    )

    assert course_stage._retained_crossing_supersession_usable(
        target,
        output,
        supersession,
        last_accepted_camera_token=previous,
        current_track_id="track-0",
        retained_crossing_dwell_frames=8,
        tuning=default_visual_config().servo,
        limits=VisualCourseStageLimits(),
    )
    assert course_stage._retained_crossing_supersession_usable(
        target,
        output,
        replace(supersession, held_previous_command_s=0.032),
        last_accepted_camera_token=previous,
        current_track_id="track-0",
        retained_crossing_dwell_frames=8,
        tuning=default_visual_config().servo,
        limits=VisualCourseStageLimits(),
    )


def test_safe_adjacent_supersession_does_not_count_and_next_wire_anchors():
    class SafeSupersededDwellServo(_Servo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passage_observations = 0

        def observe(self, snapshot, *args, **kwargs):
            proposal = super().observe(snapshot, *args, **kwargs)
            if kwargs["mode"] is not VisualApproachMode.PASSAGE:
                return proposal
            self.passage_observations += 1
            target = replace(
                proposal.current_target,
                log_scale=-1.0,
                log_scale_rate_s=0.2,
            )
            output = proposal.servo_output
            if self.passage_observations >= 4:
                retained_target, retained_output, _admission = (
                    _attempt8_close_alignment_crossing_values()
                )
                target = replace(
                    retained_target,
                    track_id=proposal.current_target.track_id,
                    frame_token=proposal.current_target.frame_token,
                    received_monotonic_s=(
                        proposal.current_target.received_monotonic_s
                    ),
                    log_scale=(
                        -0.8341951068701171
                        if self.passage_observations == 7
                        else -1.0
                    ),
                )
                output = replace(
                    retained_output,
                    yaw_rate_rad_s=0.0,
                )
                if self.passage_observations >= 8:
                    target = replace(target, log_scale=-0.785089819945014)
            return SimpleNamespace(
                current_target=target,
                servo_output=output,
                passage_admission=proposal.passage_admission,
                mode=proposal.mode,
            )

    class SafeSupersededDwellHost(_Host):
        superseded = False
        reuse_receiver_snapshot = False

        def _sample(self):
            if self.reuse_receiver_snapshot:
                self.reuse_receiver_snapshot = False
            else:
                super()._sample()
            navigation_count = sum(
                bool(kwargs.get("require_wire_receipt"))
                for _command, kwargs, _gate in self.commands
            )
            segments = getattr(
                self,
                "_visual_course_summary",
                {},
            ).get("segments", [])
            if (
                navigation_count >= 8
                and segments
                and segments[0]["crossing_anchor"] is not None
                and not self.race.race_finished
            ):
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                    visible=False,
                )

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if (
                not self.superseded
                and expected is not None
                and expected.publication_sequence == 18
            ):
                self.superseded = True
                receiver = _token(19)
                self.sequence = receiver.publication_sequence
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                self.reuse_receiver_snapshot = True
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
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
                and self.crossing_zero_count >= 2
            ):
                self.disable_credit = False
                self._advance_race()
            return receipt

    host = SafeSupersededDwellHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: SafeSupersededDwellServo(
            *args,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    assert host.superseded is True
    assert segment["superseded_proposal_count"] == 1
    assert segment["passage_command_count"] == 7
    assert segment["advance_command_count"] == 3
    assert segment["retained_crossing_supersession_hold_count"] == 1
    hold = segment["retained_crossing_supersession_hold"]
    assert hold["retained_crossing_dwell_frames"] == 3
    assert hold["superseded_camera_token"]["publication_sequence"] == 18
    assert hold["receiver_camera_token"]["publication_sequence"] == 19
    assert hold["outcome"] == "exact_receiver_successor_accepted"
    assert segment["crossing_anchor"]["basis"] == (
        course_stage.RETAINED_ADVANCE_CROSSING_BASIS
    )
    assert segment["crossing_anchor"]["camera_token"][
        "publication_sequence"
    ] == 19
    assert segment["crossing_anchor"][
        "retained_crossing_dwell_frames"
    ] == 4
    sent_tokens = {
        kwargs["wire_visual_token"].publication_sequence
        for _command, kwargs, _gate in host.commands
        if kwargs.get("require_wire_receipt")
    }
    assert 18 not in sent_tokens
    assert 19 in sent_tokens
    events = [
        payload
        for event, payload in host.recorder.events
        if event
        == "visual_course_retained_crossing_supersession_held"
    ]
    assert len(events) == 1


def test_superseded_passage_proposal_breaks_retained_crossing_dwell():
    class SupersededDwellServo(_Servo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passage_observations = 0

        def observe(self, snapshot, *args, **kwargs):
            proposal = super().observe(snapshot, *args, **kwargs)
            if kwargs["mode"] is not VisualApproachMode.PASSAGE:
                return proposal
            self.passage_observations += 1
            if self.passage_observations <= 3:
                return SimpleNamespace(
                    current_target=replace(
                        proposal.current_target,
                        log_scale=-1.0,
                        log_scale_rate_s=0.2,
                    ),
                    servo_output=proposal.servo_output,
                    passage_admission=proposal.passage_admission,
                    mode=proposal.mode,
                )
            target, output, _admission = (
                _attempt8_close_alignment_crossing_values()
            )
            target = replace(
                target,
                track_id=proposal.current_target.track_id,
                frame_token=proposal.current_target.frame_token,
                received_monotonic_s=(
                    proposal.current_target.received_monotonic_s
                ),
            )
            if self.passage_observations == 6:
                target = replace(target, normalized_x=0.50)
                output = replace(
                    output,
                    horizontal_error=0.50,
                    effective_horizontal_error=0.50,
                )
            return SimpleNamespace(
                current_target=target,
                servo_output=replace(output, yaw_rate_rad_s=0.0),
                passage_admission=proposal.passage_admission,
                mode=proposal.mode,
            )

    class SupersededDwellHost(_Host):
        superseded = False

        def _sample(self):
            super()._sample()
            navigation_count = sum(
                bool(kwargs.get("require_wire_receipt"))
                for _command, kwargs, _gate in self.commands
            )
            if navigation_count >= 9 and not self.race.race_finished:
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                    visible=False,
                )

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if (
                not self.superseded
                and expected is not None
                and expected.publication_sequence == 17
            ):
                self.superseded = True
                receiver = _token(18)
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
            receipt = await super()._send_flight_command(command, **kwargs)
            if (
                command.roll_rate
                == command.pitch_rate
                == command.yaw_rate
                == command.thrust
                == 0.0
                and self.crossing_zero_count >= 2
            ):
                self.disable_credit = False
                self._advance_race()
            return receipt

    host = SupersededDwellHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: SupersededDwellServo(
            *args,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    assert host.superseded is True
    assert segment["superseded_proposal_count"] == 1
    assert segment["crossing_anchor"]["basis"] == (
        course_stage.RETAINED_ADVANCE_CROSSING_BASIS
    )
    assert segment["crossing_anchor"]["camera_token"][
        "publication_sequence"
    ] == 21
    assert segment["crossing_anchor"][
        "retained_crossing_dwell_frames"
    ] == 3


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


@pytest.mark.parametrize(
    (
        "proved_collective",
        "current_vertical",
        "next_vertical",
        "preview_blend",
        "expected",
    ),
    (
        (
            0.3171871091857116,
            -0.16666666666666663,
            -0.28888888888888886,
            0.03983091428968513,
            0.00038945782861025464,
        ),
        (
            0.2528945018839622,
            -0.07222222222222219,
            -0.3666666666666667,
            0.1810863001315085,
            0.004265588403097756,
        ),
        (
            0.25971237467284425,
            -0.0444444444444444,
            -0.4277777777777778,
            0.2719750825987748,
            0.00834056919969576,
        ),
    ),
)
def test_attempt20_gate0_next_preview_collective_replay_is_exact(
    proved_collective,
    current_vertical,
    next_vertical,
    preview_blend,
    expected,
):
    delta = course_stage._gate0_proved_next_preview_collective_delta(
        proved_collective=proved_collective,
        current_vertical=current_vertical,
        next_vertical=next_vertical,
        preview_blend=preview_blend,
    )

    assert delta == pytest.approx(expected)
    assert proved_collective + delta <= 0.32


def test_gate0_next_preview_collective_is_capped_by_delta_and_headroom():
    assert course_stage._gate0_proved_next_preview_collective_delta(
        proved_collective=0.275,
        current_vertical=-0.10,
        next_vertical=-1.0,
        preview_blend=0.35,
    ) == pytest.approx(0.012)
    assert course_stage._gate0_proved_next_preview_collective_delta(
        proved_collective=0.319,
        current_vertical=-0.10,
        next_vertical=-1.0,
        preview_blend=0.35,
    ) == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("current_vertical", "next_vertical", "preview_blend"),
    (
        (0.01, -0.40, 0.35),
        (-0.10, -0.05, 0.35),
        (-0.10, -0.40, 0.0),
    ),
)
def test_gate0_next_preview_collective_never_opposes_current_aperture(
    current_vertical,
    next_vertical,
    preview_blend,
):
    assert course_stage._gate0_proved_next_preview_collective_delta(
        proved_collective=0.275,
        current_vertical=current_vertical,
        next_vertical=next_vertical,
        preview_blend=preview_blend,
    ) == 0.0


def test_gate0_next_preview_collective_requires_exact_latched_publication():
    current = replace(
        _target(_snapshot(0, "track-0", 126), "track-0"),
        normalized_y_down=-0.16666666666666663,
    )
    next_target = replace(
        current,
        track_id="track-1",
        normalized_y_down=-0.28888888888888886,
    )
    output = SimpleNamespace(
        next_gate_blend=0.03983091428968513,
        # Retained passage publications need not repeat the one-shot servo
        # review marker; the coordinator latch remains authoritative.
        reviewed_next_track_id=None,
        vertical_error_image_down=current.normalized_y_down,
        next_vertical_error_image_down=(
            next_target.normalized_y_down
        ),
        yaw_envelope_limited=True,
    )

    thrust, delta = (
        course_stage._gate0_proved_collective_with_exact_next_preview(
            proved_collective=0.3171871091857116,
            current_target=current,
            next_target=next_target,
            latched_next_track_id="track-1",
            servo_output=output,
        )
    )

    assert delta == pytest.approx(0.00038945782861025464)
    assert thrust == pytest.approx(0.31757656701432185)

    with pytest.raises(ValueError, match="one exact publication"):
        course_stage._gate0_proved_collective_with_exact_next_preview(
            proved_collective=0.3171871091857116,
            current_target=current,
            next_target=replace(
                next_target,
                frame_token=replace(
                    next_target.frame_token,
                    publication_sequence=127,
                ),
            ),
            latched_next_track_id="track-1",
            servo_output=output,
        )

    with pytest.raises(ValueError, match="persistent coordinator latch"):
        course_stage._gate0_proved_collective_with_exact_next_preview(
            proved_collective=0.3171871091857116,
            current_target=current,
            next_target=next_target,
            latched_next_track_id=None,
            servo_output=output,
        )


def test_gate0_next_preview_collective_is_zero_after_preview_retirement():
    current = replace(
        _target(_snapshot(0, "track-0", 159), "track-0"),
        normalized_y_down=-0.03,
    )
    output = SimpleNamespace(next_gate_blend=0.0)

    thrust, delta = (
        course_stage._gate0_proved_collective_with_exact_next_preview(
            proved_collective=0.265,
            current_target=current,
            next_target=None,
            latched_next_track_id=None,
            servo_output=output,
        )
    )

    assert thrust == 0.265
    assert delta == 0.0


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


def test_initial_gate_applies_only_exact_latched_next_preview_collective():
    class ExactPreviewServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            proposal = super().observe(snapshot, *args, **kwargs)
            current = replace(
                proposal.current_target,
                normalized_y_down=-0.10,
            )
            next_target = replace(
                current,
                track_id=self.preview_track_id,
                normalized_y_down=-0.40,
            )
            output = replace(
                proposal.servo_output,
                next_gate_blend=0.35,
                vertical_error_image_down=current.normalized_y_down,
                effective_vertical_error_image_down=-0.1525,
                next_vertical_error_image_down=(
                    next_target.normalized_y_down
                ),
                reviewed_next_track_id=self.preview_track_id,
            )
            admission = proposal.passage_admission
            if admission is not None:
                admission = replace(
                    admission,
                    current_target=current,
                    preview_track_id=self.preview_track_id,
                    preview_blend=0.35,
                )
            return SimpleNamespace(
                current_target=current,
                next_target=next_target,
                servo_output=output,
                candidate_track_ids=(self.preview_track_id,),
                provisional_track_ids=(),
                withholding_reason=None,
                relationship_basis=None,
                latched_next_track_id=self.preview_track_id,
                passage_admission=admission,
                mode=proposal.mode,
            )

    host = _Host(
        initial_gate=0,
        finish_gate=1,
        fresh_after_samples=1,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: ExactPreviewServo(
            *args,
            yaw_rate=0.0,
            **kwargs,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    launch = result["segments"][0]["launch_bootstrap"]
    assert launch["next_preview_collective_command_count"] > 0
    assert launch["max_next_preview_collective_delta"] == pytest.approx(
        0.0084
    )
    assert launch["last_next_preview_collective_delta"] == pytest.approx(
        0.0084
    )
    assert launch["last_next_preview_collective_track_id"] == "track-1"
    assert launch["last_thrust"] == pytest.approx(0.2914)
    crossing = result["segments"][0]["crossing_anchor"]
    assert crossing["command"]["thrust"] == pytest.approx(0.2914)
    assert crossing["next_preview_collective_delta"] == pytest.approx(
        0.0084
    )
    assert crossing["current_only_crossing_coast_thrust"] == (
        pytest.approx(0.283)
    )


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


@pytest.mark.parametrize(
    "wire_scenario",
    (
        "accepted",
        "superseded",
        "wall_before",
        "wall_equal",
        "wall_over",
        "duplicate_timeout",
        "second_refusal",
        "refusal_superseded",
        "observation_equal",
        "observation_over",
        "lying_receipt",
        "qpc_wall_equal_frozen_monotonic",
        "qpc_timeout_frozen_monotonic",
    ),
)
def test_rate_only_refusal_requalifies_sealed_preview_at_exact_bounds(
    wire_scenario,
):
    class RequalificationHost(_Host):
        skip_one_publication = False
        supersede_requalification_token = None
        requalification_superseded = False
        requalification_refusal_control_s = None
        wall_slot_delayed = False
        preview_candidate_send_calls = 0
        preview_candidate_call_start_ns = None
        refusal_token = None
        refusal_frame_superseded = False
        refusal_observation_ns = None
        qpc_override_ns = None
        qpc_override_set_monotonic_s = None

        def _sample(self):
            preview_withdrawn = any(
                event == "visual_course_next_preview_withdrawn"
                for event, _payload in self.recorder.events
            )
            if preview_withdrawn:
                self.qpc_override_ns = None
            if (
                wire_scenario == "qpc_timeout_frozen_monotonic"
                and self.requalification_refusal_control_s is not None
                and not preview_withdrawn
            ):
                requalification = self._visual_course_summary[
                    "segments"
                ][0]["next_preview_requalification"]
                self.qpc_override_ns = (
                    requalification[
                        "refusal_control_perf_counter_ns"
                    ]
                    + 500_000_001
                )
                self.qpc_override_set_monotonic_s = self.clock
                return
            if (
                wire_scenario == "duplicate_timeout"
                and self.requalification_refusal_control_s is not None
                and not preview_withdrawn
            ):
                return
            super()._sample()
            if self.skip_one_publication:
                self.skip_one_publication = False
                self.sequence += 1
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
            if (
                wire_scenario
                in {"observation_equal", "observation_over"}
                and self.refusal_token is not None
            ):
                snapshot = self.visual_gate_graph.latest_snapshot
                publication_delta = (
                    snapshot.latest_camera_token.publication_sequence
                    - self.refusal_token.publication_sequence
                )
                assert self.refusal_observation_ns is not None
                total_observation_delta_ns = (
                    450_000_000
                    + (1 if wire_scenario == "observation_over" else 0)
                )
                observation_ns = (
                    self.refusal_observation_ns
                    + math.floor(
                        publication_delta
                        * total_observation_delta_ns
                        / 13
                    )
                )
                history = snapshot.current_track.history
                snapshot.current_track.history = (
                    *history[:-1],
                    replace(
                        history[-1],
                        observation_monotonic_ns=observation_ns,
                    ),
                )

        async def _wait_for_next_flight_command_slot(self):
            ready = await super()._wait_for_next_flight_command_slot()
            if (
                wire_scenario
                in {
                    "wall_before",
                    "wall_equal",
                    "wall_over",
                    "qpc_wall_equal_frozen_monotonic",
                }
                and self.supersede_requalification_token
                == self.visual_gate_graph.latest_snapshot.latest_camera_token
                and not self.wall_slot_delayed
            ):
                requalification = self._visual_course_summary[
                    "segments"
                ][0]["next_preview_requalification"]
                deadline_ns = requalification[
                    "wire_start_deadline_monotonic_ns"
                ]
                if wire_scenario == "qpc_wall_equal_frozen_monotonic":
                    self.qpc_override_ns = deadline_ns
                    self.qpc_override_set_monotonic_s = self.clock
                else:
                    self.clock = (
                        deadline_ns
                        + {
                            "wall_before": -1,
                            "wall_equal": 0,
                            "wall_over": 1,
                        }[wire_scenario]
                    ) / 1_000_000_000
                self.wall_slot_delayed = True
            return ready

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if (
                wire_scenario == "refusal_superseded"
                and expected == self.refusal_token
                and not self.refusal_frame_superseded
            ):
                self.refusal_frame_superseded = True
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
            if (
                expected is not None
                and expected == self.supersede_requalification_token
            ):
                self.preview_candidate_send_calls += 1
                self.preview_candidate_call_start_ns = round(
                    self.clock * 1_000_000_000
                )
                assert (
                    self.preview_candidate_call_start_ns
                    < kwargs["wire_start_deadline_ns"]
                )
            if (
                wire_scenario == "superseded"
                and expected == self.supersede_requalification_token
                and not self.requalification_superseded
            ):
                self.requalification_superseded = True
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
            receipt = await super()._send_flight_command(
                command,
                **kwargs,
            )
            if (
                wire_scenario == "lying_receipt"
                and expected == self.supersede_requalification_token
            ):
                deadline_ns = self._visual_course_summary["segments"][0][
                    "next_preview_requalification"
                ]["wire_start_deadline_monotonic_ns"]
                self._last_flight_command_started_ns = deadline_ns
                receipt["call_start_monotonic_ns"] = deadline_ns
                receipt["visual_receiver_authority"][
                    "call_start_monotonic_ns"
                ] = deadline_ns
            return receipt

    host = RequalificationHost(
        initial_gate=0,
        finish_gate=0,
        fresh_after_samples=1,
    )
    runtime, _calls = _runtime(host)
    factory_blends = []
    factory_required_ids = []
    refusal_tokens = []
    requalification_tokens = []

    class RefusingPreviewServo(_Servo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observe_count = 0

        def observe(self, snapshot, *args, **kwargs):
            self.observe_count += 1
            if self.observe_count == 8:
                refusal_tokens.append(snapshot.latest_camera_token)
                host.refusal_token = snapshot.latest_camera_token
                host.refusal_observation_ns = (
                    snapshot.current_track.history[-1]
                    .observation_monotonic_ns
                )
                host.requalification_refusal_control_s = host.clock
                raise VisualApproachPassageSafetyUnavailable(
                    "exact attempt-6 current passage discontinuity",
                    violation_codes=(
                        "current_vertical_rate",
                        "current_log_scale_rate",
                    ),
                    violation_evidence=(
                        (
                            "current_vertical_rate",
                            0.9084642237935744,
                            0.60,
                            0.3084642237935744,
                        ),
                        (
                            "current_log_scale_rate",
                            -2.6499873864506065,
                            -1.50,
                            1.1499873864506065,
                        ),
                    ),
                    camera_observation_monotonic_s=(
                        snapshot.latest_camera_token.publication_sequence
                        * 0.02
                    ),
                    latched_next_track_id=self.preview_track_id,
                )
            return super().observe(snapshot, *args, **kwargs)

    class ObservationProvenanceServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            proposal = super().observe(snapshot, *args, **kwargs)
            observation_ns = (
                snapshot.current_track.history[-1]
                .observation_monotonic_ns
            )
            target = replace(
                proposal.current_target,
                received_monotonic_s=(
                    observation_ns / 1_000_000_000
                ),
            )
            proposal.current_target = target
            if proposal.passage_admission is not None:
                proposal.passage_admission = replace(
                    proposal.passage_admission,
                    current_target=target,
                )
            return proposal

    class SealedRequalificationServo(ObservationProvenanceServo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observe_count = 0
            assert self.required_next_track_id == self.preview_track_id

        def observe(self, snapshot, *args, **kwargs):
            self.observe_count += 1
            if (
                wire_scenario == "second_refusal"
                and self.observe_count == 2
            ):
                raise VisualApproachPassageSafetyUnavailable(
                    "second sealed rate-only discontinuity",
                    violation_codes=(
                        "current_vertical_rate",
                        "current_log_scale_rate",
                    ),
                    violation_evidence=(
                        (
                            "current_vertical_rate",
                            0.61,
                            0.60,
                            0.01,
                        ),
                        (
                            "current_log_scale_rate",
                            -1.51,
                            -1.50,
                            0.01,
                        ),
                    ),
                    camera_observation_monotonic_s=(
                        snapshot.latest_camera_token.publication_sequence
                        * 0.02
                    ),
                    latched_next_track_id=self.preview_track_id,
                )
            proposal = super().observe(snapshot, *args, **kwargs)
            if self.observe_count == 1:
                requalification_tokens.append(
                    snapshot.latest_camera_token
                )
            if (
                kwargs["mode"] is VisualApproachMode.APPROACH
                and self.observe_count <= 12
            ):
                return SimpleNamespace(
                    current_target=proposal.current_target,
                    next_target=None,
                    servo_output=replace(
                        proposal.servo_output,
                        next_gate_blend=0.0,
                        reviewed_next_track_id=None,
                    ),
                    candidate_track_ids=(self.preview_track_id,),
                    provisional_track_ids=(),
                    withholding_reason=(
                        "current_passage_corridor_not_ready"
                    ),
                    relationship_basis=None,
                    latched_next_track_id=None,
                    passage_admission=None,
                    mode=proposal.mode,
                )
            if kwargs["mode"] is VisualApproachMode.APPROACH:
                preview_blend = 0.05
                if (
                    wire_scenario
                    in {
                        "superseded",
                        "wall_before",
                        "wall_equal",
                        "wall_over",
                        "lying_receipt",
                        "qpc_wall_equal_frozen_monotonic",
                    }
                    and self.observe_count == 13
                ):
                    host.supersede_requalification_token = (
                        snapshot.latest_camera_token
                    )
                admission = VisualApproachPassageAdmission(
                    basis="tight-current-corridor-dwell-v1",
                    current_gate_index=self.gate_index,
                    current_target=proposal.current_target,
                    camera_token=snapshot.latest_camera_token,
                    tracker_frame_sequence=(
                        snapshot.tracker_frame_sequence
                    ),
                    corridor_frames=5,
                    preview_track_id=self.preview_track_id,
                    preview_blend=preview_blend,
                )
                return SimpleNamespace(
                    current_target=proposal.current_target,
                    next_target=replace(
                        proposal.current_target,
                        track_id=self.preview_track_id,
                        normalized_y_down=-0.20,
                    ),
                    servo_output=replace(
                        proposal.servo_output,
                        next_gate_blend=preview_blend,
                        next_horizontal_error=0.30,
                        next_vertical_error_image_down=-0.20,
                        reviewed_next_track_id=self.preview_track_id,
                    ),
                    candidate_track_ids=(self.preview_track_id,),
                    provisional_track_ids=(),
                    withholding_reason=None,
                    relationship_basis=None,
                    latched_next_track_id=self.preview_track_id,
                    passage_admission=admission,
                    mode=proposal.mode,
                )
            return proposal

    def factory(*args, **kwargs):
        factory_blends.append(kwargs["next_gate_blend"])
        factory_required_ids.append(
            kwargs.get("required_next_track_id")
        )
        servo_type = (
            RefusingPreviewServo
            if len(factory_blends) == 1
            else (
                SealedRequalificationServo
                if len(factory_blends) == 2
                else ObservationProvenanceServo
            )
        )
        if (
            len(factory_blends) == 2
            and wire_scenario != "refusal_superseded"
        ):
            host.skip_one_publication = True
        return servo_type(*args, **kwargs)

    runtime = replace(runtime, servo_factory=factory)
    if wire_scenario in {
        "qpc_wall_equal_frozen_monotonic",
        "qpc_timeout_frozen_monotonic",
    }:
        runtime = replace(
            runtime,
            perf_counter_ns=lambda: (
                host.qpc_override_ns
                if host.qpc_override_ns is not None
                else round(host.clock * 1_000_000_000)
            ),
        )
    if wire_scenario == "lying_receipt":
        with pytest.raises(
            SafetyAbort,
            match="lacks exact visual wire timing",
        ):
            asyncio.run(
                run_visual_course_stage(
                    host,
                    _context(),
                    runtime=runtime,
                )
            )
        assert host.preview_candidate_send_calls == 1
        assert not any(
            event == "visual_course_next_preview_requalified"
            for event, _payload in host.recorder.events
        )
        assert (
            host._visual_course_summary["segments"][0][
                "next_preview_requalification"
            ]["outcome"]
            == "pending"
        )
        return
    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    expected_factory_blends = [
        host.visual_config.lifecycle.next_gate_blend_max,
        host.visual_config.lifecycle.next_gate_blend_max,
    ]
    expected_factory_required_ids = [None, "track-1"]
    preview_retired = wire_scenario in {
        "superseded",
        "wall_equal",
        "wall_over",
        "duplicate_timeout",
        "second_refusal",
        "observation_over",
        "qpc_wall_equal_frozen_monotonic",
        "qpc_timeout_frozen_monotonic",
    }
    if preview_retired:
        expected_factory_blends.append(0.0)
        expected_factory_required_ids.append("track-1")
    assert factory_blends == expected_factory_blends
    assert factory_required_ids == expected_factory_required_ids
    assert refusal_tokens == requalification_tokens
    segment = result["segments"][0]
    assert segment["passage_admission"]["preview_track_id"] == "track-1"
    requalification = segment["next_preview_requalification"]
    assert requalification["sealed_next_track_id"] == "track-1"
    assert requalification["refusal_violation_codes"] == [
        "current_vertical_rate",
        "current_log_scale_rate",
    ]
    assert requalification["max_fresh_frames"] == 12
    assert requalification["max_publication_delta"] == 13
    assert requalification["max_duration_s"] == 0.45
    assert requalification["max_control_duration_s"] == 0.50
    if preview_retired:
        if wire_scenario == "superseded":
            assert requalification["fresh_frame_count"] > 12
            assert requalification["publication_delta"] > 13
            expected_retirement_reason = (
                "preview_requalification_bounds_exhausted"
            )
        elif wire_scenario in {
            "wall_equal",
            "wall_over",
            "qpc_wall_equal_frozen_monotonic",
        }:
            assert requalification["fresh_frame_count"] == 12
            assert requalification["publication_delta"] == 13
            expected_retirement_reason = (
                "preview_requalification_wire_deadline_expired"
            )
        elif wire_scenario == "duplicate_timeout":
            assert requalification["fresh_frame_count"] == 0
            assert requalification["publication_delta"] == 0
            assert requalification["control_elapsed_s"] > 0.50
            expected_retirement_reason = (
                "preview_requalification_control_timeout"
            )
        elif wire_scenario == "qpc_timeout_frozen_monotonic":
            assert requalification["fresh_frame_count"] == 0
            assert requalification["publication_delta"] == 0
            assert requalification["control_elapsed_ns"] == 500_000_001
            assert (
                host.qpc_override_set_monotonic_s
                - host.requalification_refusal_control_s
                < 0.50
            )
            expected_retirement_reason = (
                "preview_requalification_control_timeout"
            )
        elif wire_scenario == "second_refusal":
            assert requalification["fresh_frame_count"] == 0
            assert requalification["publication_delta"] == 0
            expected_retirement_reason = (
                "preview_requalification_safety_violation"
            )
        elif wire_scenario == "observation_over":
            assert requalification["fresh_frame_count"] == 12
            assert requalification["publication_delta"] == 13
            assert requalification["elapsed_s"] == pytest.approx(
                0.450000001
            )
            expected_retirement_reason = (
                "preview_requalification_bounds_exhausted"
            )
        else:
            raise AssertionError(
                f"unexpected retired scenario {wire_scenario}"
            )
        assert segment["passage_admission"]["preview_blend"] == 0.0
        assert segment["next_preview_retired"] is True
        assert segment["next_preview_withdrawal_count"] == 1
        assert segment["next_preview_withdrawal"]["reason"] == (
            expected_retirement_reason
        )
        assert requalification["outcome"] == "retired"
        assert requalification["retirement_reason"] == (
            expected_retirement_reason
        )
        assert host.requalification_superseded is (
            wire_scenario == "superseded"
        )
        assert (segment["superseded_proposal_count"] >= 1) is (
            wire_scenario == "superseded"
        )
        if wire_scenario in {
            "wall_equal",
            "wall_over",
            "qpc_wall_equal_frozen_monotonic",
        }:
            assert host.preview_candidate_send_calls == 0
        if wire_scenario == "qpc_wall_equal_frozen_monotonic":
            assert (
                host.qpc_override_set_monotonic_s
                - host.requalification_refusal_control_s
                < 0.50
            )
    else:
        assert requalification["fresh_frame_count"] == 12
        assert requalification["publication_delta"] == 13
        if wire_scenario == "observation_equal":
            assert requalification["elapsed_s"] == pytest.approx(0.45)
        assert segment["passage_admission"]["preview_blend"] == 0.05
        assert segment["next_preview_retired"] is False
        assert segment["next_preview_withdrawal_count"] == 0
        assert segment["next_preview_withdrawal"] is None
        assert requalification["outcome"] == "requalified"
        assert requalification["requalified_preview_blend"] == 0.05
        assert host.requalification_superseded is False
        candidate_send = next(
            kwargs
            for _command, kwargs, _gate_index in host.commands
            if kwargs.get("wire_visual_token")
            == host.supersede_requalification_token
        ) if wire_scenario == "wall_before" else None
        if candidate_send is not None:
            stored_deadline_ns = requalification[
                "wire_start_deadline_monotonic_ns"
            ]
            candidate_wire_start_ns = stored_deadline_ns - 1
            assert candidate_wire_start_ns < stored_deadline_ns
            assert host.preview_candidate_send_calls == 1
            assert (
                host.preview_candidate_call_start_ns
                == candidate_wire_start_ns
            )
            assert (
                candidate_send["wire_start_deadline_ns"]
                == stored_deadline_ns
            )
    assert segment["passage_authority_enabled"] is True
    assert segment["advance_command_count"] >= 3
    refusal_sends = [
        command
        for command, kwargs, _gate_index in host.commands
        if kwargs.get("wire_visual_token") == refusal_tokens[0]
    ]
    if wire_scenario == "refusal_superseded":
        assert refusal_sends == []
        assert host.refusal_frame_superseded is True
        assert segment["superseded_proposal_count"] >= 1
    else:
        assert len(refusal_sends) == 1
        assert refusal_sends[0].thrust in {
            0.26,
            host.visual_config.lifecycle.launch_boost_thrust,
        }
        assert refusal_sends[0].thrust != (
            host.visual_config.servo.brake_thrust
        )
    navigation_send_count = sum(
        bool(kwargs.get("require_wire_receipt"))
        for _command, kwargs, _gate_index in host.commands
    )
    assert segment["launch_bootstrap"]["command_count"] == (
        navigation_send_count
    )
    assert any(
        event == "visual_course_next_preview_requalified"
        for event, _payload in host.recorder.events
    ) is (not preview_retired)
    assert any(
        event == "visual_course_next_preview_withdrawn"
        for event, _payload in host.recorder.events
    ) is preview_retired


def test_rate_only_refusal_without_latched_identity_aborts_unsealed_replan():
    host = _Host(initial_gate=0, finish_gate=0)

    class MissingIdentityRefusalServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            raise VisualApproachPassageSafetyUnavailable(
                "structured refusal omitted its identity",
                violation_codes=(
                    "current_vertical_rate",
                    "current_log_scale_rate",
                ),
                violation_evidence=(
                    (
                        "current_vertical_rate",
                        1.0,
                        0.60,
                        0.40,
                    ),
                    (
                        "current_log_scale_rate",
                        -3.0,
                        -1.50,
                        1.50,
                    ),
                ),
                camera_observation_monotonic_s=(
                    snapshot.latest_camera_token.publication_sequence
                    * 0.02
                ),
            )

    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: (
            MissingIdentityRefusalServo(*args, **kwargs)
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="lacks its sealed next identity",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )


def test_preview_requalification_expiry_retires_to_same_identity_only():
    host = _Host(initial_gate=0, finish_gate=0, fresh_after_samples=1)
    factory_blends = []
    factory_required_ids = []

    class RefusingPreviewServo(_Servo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observe_count = 0

        def observe(self, snapshot, *args, **kwargs):
            self.observe_count += 1
            if self.observe_count == 8:
                raise VisualApproachPassageSafetyUnavailable(
                    "exact rate-only discontinuity",
                    violation_codes=(
                        "current_vertical_rate",
                        "current_log_scale_rate",
                    ),
                    violation_evidence=(
                        (
                            "current_vertical_rate",
                            1.0,
                            0.60,
                            0.40,
                        ),
                        (
                            "current_log_scale_rate",
                            -3.0,
                            -1.50,
                            1.50,
                        ),
                    ),
                    camera_observation_monotonic_s=(
                        snapshot.latest_camera_token.publication_sequence
                        * 0.02
                    ),
                    latched_next_track_id=self.preview_track_id,
                )
            return super().observe(snapshot, *args, **kwargs)

    class NeverRequalifiedServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            proposal = super().observe(snapshot, *args, **kwargs)
            return SimpleNamespace(
                current_target=proposal.current_target,
                next_target=None,
                servo_output=replace(
                    proposal.servo_output,
                    next_gate_blend=0.0,
                    reviewed_next_track_id=None,
                ),
                candidate_track_ids=(self.preview_track_id,),
                provisional_track_ids=(),
                withholding_reason=(
                    "current_passage_corridor_not_ready"
                ),
                relationship_basis=None,
                latched_next_track_id=None,
                passage_admission=None,
                mode=proposal.mode,
            )

    def factory(*args, **kwargs):
        factory_blends.append(kwargs["next_gate_blend"])
        factory_required_ids.append(
            kwargs.get("required_next_track_id")
        )
        servo_type = (
            RefusingPreviewServo
            if len(factory_blends) == 1
            else (
                NeverRequalifiedServo
                if len(factory_blends) == 2
                else _Servo
            )
        )
        return servo_type(*args, **kwargs)

    runtime, _calls = _runtime(host)
    runtime = replace(runtime, servo_factory=factory)
    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert factory_blends == [
        host.visual_config.lifecycle.next_gate_blend_max,
        host.visual_config.lifecycle.next_gate_blend_max,
        0.0,
    ]
    assert factory_required_ids == [None, "track-1", "track-1"]
    segment = result["segments"][0]
    assert segment["next_preview_retired"] is True
    assert segment["next_preview_withdrawal_count"] == 1
    assert segment["next_preview_withdrawal"]["reason"] == (
        "preview_requalification_bounds_exhausted"
    )
    requalification = segment["next_preview_requalification"]
    assert requalification["outcome"] == "retired"
    assert requalification["retirement_reason"] == (
        "preview_requalification_bounds_exhausted"
    )
    assert requalification["fresh_frame_count"] == 13
    assert requalification["publication_delta"] == 13
    assert segment["passage_admission"]["preview_track_id"] == "track-1"
    assert segment["passage_admission"]["preview_blend"] == 0.0


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


def test_exact_next_post_credit_frame_is_admitted_without_immediate_abort():
    host = _Host(initial_gate=1, finish_gate=2, fresh_after_samples=1)
    runtime, _calls = _runtime(host)
    continuation_admission = runtime.recovery_continuation_admission
    navigation_counts_at_admission = []

    def audited_continuation_admission(*args, **kwargs):
        navigation_counts_at_admission.append(
            sum(
                command.thrust > 0.0
                for command, _wire, gate_index in host.commands
                if gate_index == 2
            )
        )
        return continuation_admission(*args, **kwargs)

    runtime = replace(
        runtime,
        recovery_continuation_admission=(
            audited_continuation_admission
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    transition = result["authoritative_transitions"][0]
    assert navigation_counts_at_admission == [0]
    assert transition["post_transition_zero_command_count"] == 0
    assert transition["recovery_admission"]["admission_kind"] == (
        "exact_next_continuation"
    )
    assert transition["recovery_admission"]["admitted_frame_token"][
        "publication_sequence"
    ] == host.promotion_tokens[0].publication_sequence + 1


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


def test_post_credit_miss_then_reacquisition_never_reuses_stale_admission():
    host = _Host(initial_gate=1, finish_gate=2, fresh_after_samples=3)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="fresh post-credit camera frame",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host._visual_course_summary["success"] is False
    post_transition_commands = [
        command
        for command, _kwargs, gate_index in host.commands
        if gate_index == 2
    ]
    assert post_transition_commands
    assert all(
        command.roll_rate
        == command.pitch_rate
        == command.yaw_rate
        == command.thrust
        == 0.0
        for command in post_transition_commands
    )


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
        (
            "retained_crossing_projection_min_log_scale",
            -0.830001,
            "retained projection scale",
        ),
        (
            "retained_crossing_max_observation_to_wire_s",
            0.035001,
            "retained projection timing",
        ),
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


def test_passage_yaw_soft_stop_accrues_only_retained_crossing_dwell():
    profile = _yaw_profile()
    limits = replace(
        VisualCourseStageLimits(),
        crossing_arm_min_advance_commands=2,
    )
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
    excursion = 0.005
    rate_to_soft_stop = (
        (
            limits.max_segment_yaw_excursion_rad
            - response_reserve
            - excursion
        )
        / profile.max_gyro_response_delay_s
        + 1e-6
    )

    class CloseThirdAdvanceServo(_Servo):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passage_observations = 0

        def observe(self, snapshot, *args, **kwargs):
            proposal = super().observe(snapshot, *args, **kwargs)
            if kwargs["mode"] is VisualApproachMode.PASSAGE:
                self.passage_observations += 1
                if self.passage_observations <= 2:
                    return SimpleNamespace(
                        current_target=replace(
                            proposal.current_target,
                            log_scale=-1.0,
                        ),
                        servo_output=proposal.servo_output,
                        passage_admission=proposal.passage_admission,
                        mode=proposal.mode,
                    )
            return proposal

    class SoftStopThenLossHost(_Host):
        sample_count = 0

        def _sample(self):
            super()._sample()
            self.sample_count += 1
            if self.sample_count in {4, 5, 6}:
                _set_attitude(
                    self,
                    yaw=excursion,
                    rates=(0.0, 0.0, rate_to_soft_stop),
                )
            else:
                _set_attitude(self)
            if self.sample_count >= 7:
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                    visible=False,
                )

    host = SoftStopThenLossHost(
        initial_gate=3,
        finish_gate=3,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host, limits=limits)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: CloseThirdAdvanceServo(
            *args,
            passage_preview_blend=0.20,
            **kwargs,
        ),
    )

    with pytest.raises(SafetyAbort, match="gate 3 credit timed out"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["passage_command_count"] == 5
    assert segment["advance_command_count"] == 2
    assert segment["yaw_soft_stop_zero_command_count"] == 3
    assert segment["retained_crossing_dwell_frames"] == 3
    assert segment["max_retained_crossing_dwell_frames"] == 3
    assert segment["crossing_anchor"]["basis"] == (
        course_stage.RETAINED_ADVANCE_CROSSING_BASIS
    )
    assert segment["crossing_anchor"]["current_advance_enabled"] is True
    assert segment["crossing_anchor"]["yaw_soft_stop_zeroed"] is True
    assert segment["passage_next_preview_command_count"] == 5
    assert segment["passage_admission"]["preview_track_id"] == "track-4"
    assert segment["next_preview_retired"] is False
    assert segment["next_preview_withdrawal_count"] == 0
    assert segment["next_preview_withdrawal"] is None
    navigation = [
        command
        for command, kwargs, _gate in host.commands
        if kwargs.get("require_wire_receipt")
    ]
    assert [command.yaw_rate for command in navigation[-3:]] == [
        0.0,
        0.0,
        0.0,
    ]
    assert [command.pitch_rate for command in navigation[-3:]] == [
        navigation[-4].pitch_rate,
        navigation[-4].pitch_rate,
        navigation[-4].pitch_rate,
    ]
    assert [command.thrust for command in navigation[-3:]] == [
        0.295,
        0.295,
        0.295,
    ]


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
