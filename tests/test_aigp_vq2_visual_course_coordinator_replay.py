"""Coordinator replay of compact build-3385 logged-state timing modes.

This exercises the real visual-course coordinator with synthetic host
boundaries shaped by the compact tracker/graph/IMU/race/wire facts.  It is not
JPEG replay, detector replay, or full receiver replay.  In particular, the
camera scheduler below models receiver publication/replacement at about
30 Hz while the production coordinator retains its exact 50 Hz dropped-tick
control policy.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import VisualTrackRole
from planning.vq2_visual_approach import (
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
)
from scripts import aigp_vq2_visual_course_stage as course_stage
from scripts.aigp_vq2_visual_course_stage import (
    VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON,
    VisualCourseStageLimits,
    run_visual_course_stage,
)
from tests.test_aigp_vq2_visual_course import (
    SafetyAbort,
    _Host as _CoordinatorHost,
    _Servo as _CoordinatorServo,
    _context,
    _runtime,
    _snapshot,
    _token,
)


_CAMERA_PERIOD_S = 1.0 / 30.0
_CONTROL_PERIOD_S = 0.02
_RACE_STATUS_PERIOD_S = 0.25


class _CadencedCoordinatorServo(_CoordinatorServo):
    """Keep planning deterministic while exposing real coordinator refusals."""

    def observe(self, snapshot, *args, **kwargs):
        track = snapshot.current_track
        one_edge_recovery = bool(
            kwargs.get("mode") is VisualApproachMode.PROMOTE_REACQUIRE
            and track.clipping
            in {
                FrameEdge.LEFT,
                FrameEdge.TOP,
                FrameEdge.RIGHT,
                FrameEdge.BOTTOM,
            }
        )
        if (
            not track.visible
            or (
                track.clipping != FrameEdge.NONE
                and not one_edge_recovery
            )
            or (
                track.center_censored
                and not one_edge_recovery
            )
        ):
            raise VisualApproachCurrentGeometryUnavailable(
                "logged near-plane geometry is censored or unavailable"
            )
        proposal = super().observe(snapshot, *args, **kwargs)
        latest = track.history[-1]
        target = replace(
            proposal.current_target,
            received_monotonic_s=(
                latest.observation_monotonic_ns / 1_000_000_000.0
            ),
            normalized_x=track.center_norm[0],
            normalized_y_down=track.center_norm[1],
            normalized_x_rate_s=track.center_velocity_norm_s[0],
            normalized_y_rate_down_s=track.center_velocity_norm_s[1],
            confidence=track.confidence,
            association_confidence=track.association_confidence,
            clipped=track.clipping != FrameEdge.NONE,
            center_censored=track.center_censored,
            horizontal_censored=bool(
                track.clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
            ),
            vertical_censored=bool(
                track.clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
            ),
            ambiguous=track.ambiguous,
        )
        proposal.current_target = target
        if proposal.passage_admission is not None:
            proposal.passage_admission = replace(
                proposal.passage_admission,
                current_target=target,
                camera_token=snapshot.latest_camera_token,
                tracker_frame_sequence=snapshot.tracker_frame_sequence,
            )
        return proposal


class _CadencedCoordinatorHost(_CoordinatorHost):
    """Latest-frame host with 30 Hz camera and delayed 4 Hz race ingress."""

    def __init__(
        self,
        *,
        credit_policy: str,
        unsafe_after_latch: bool = False,
        finish_gate: int = 2,
        credit_delay_s: float = _RACE_STATUS_PERIOD_S,
    ) -> None:
        super().__init__(
            initial_gate=1,
            finish_gate=finish_gate,
            disable_credit=True,
        )
        self.credit_policy = credit_policy
        self.credit_delay_s = credit_delay_s
        self.unsafe_after_latch = unsafe_after_latch
        self._next_camera_s = self.clock + _CAMERA_PERIOD_S
        self._track_histories: dict[str, list[SimpleNamespace]] = {}
        self._censor_step: dict[int, int] = {}
        self.anchor_times: dict[int, float] = {}
        self.credit_due_times: dict[int, float] = {}
        self.credit_publish_times: dict[int, float] = {}
        self.credit_tokens_by_gate = {}
        self.camera_publications: list[dict[str, object]] = []
        self.control_sample_times: list[float] = []
        self.navigation_wires: list[dict[str, object]] = []
        self._overrun_injected = False
        self._install_snapshot(
            token=_token(self.sequence),
            publication_s=self.clock,
            state="clean",
        )

    def _install_snapshot(
        self,
        *,
        token,
        publication_s: float,
        state: str,
    ) -> None:
        history = self._track_histories.setdefault(
            self.current_track_id,
            [],
        )
        visible = state != "lost"
        ambiguous = False
        clipping = FrameEdge.NONE
        center_censored = False
        center = (0.04, 0.03)
        velocity = (0.0, 0.0)
        apparent_scale = math.exp(-0.50)
        if state == "bottom":
            clipping = FrameEdge.BOTTOM
            center_censored = True
            center = (
                0.25 if self.unsafe_after_latch else 0.0,
                0.02,
            )
            apparent_scale = 0.78
        elif state == "top_bottom":
            clipping = FrameEdge.TOP | FrameEdge.BOTTOM
            center_censored = True
            center = (0.0, 0.0)
            apparent_scale = 0.84
        elif state == "lost":
            clipping = FrameEdge.TOP | FrameEdge.BOTTOM
            center_censored = True
            center = None
            velocity = None
            apparent_scale = 0.84

        if visible:
            observation_ns = round(
                (publication_s - 0.001) * 1_000_000_000
            )
            sample = SimpleNamespace(
                token=token,
                observation_monotonic_ns=observation_ns,
                publication_monotonic_ns=round(
                    publication_s * 1_000_000_000
                ),
            )
            history.append(sample)
            del history[:-8]
            latest_token = token
            missed_frame_count = 0
        else:
            assert history
            latest_token = history[-1].token
            missed_frame_count = max(
                1,
                token.publication_sequence
                - latest_token.publication_sequence,
            )

        track = SimpleNamespace(
            track_id=self.current_track_id,
            latest_token=latest_token,
            visible=visible,
            ambiguous=ambiguous,
            missed_frame_count=missed_frame_count,
            consecutive_frame_count=len(history),
            confidence=0.94,
            association_confidence=0.88,
            clipping=clipping,
            center_censored=center_censored,
            apparent_scale=apparent_scale,
            role=VisualTrackRole.CURRENT,
            authoritative_gate_index=self.current_gate,
            history=tuple(history),
            center_norm=center,
            center_velocity_norm_s=velocity,
        )
        self.visual_gate_graph.latest_snapshot = SimpleNamespace(
            tracker_frame_sequence=token.publication_sequence,
            latest_camera_token=token,
            current_gate_index=self.current_gate,
            current_track_id=self.current_track_id,
            current_track=track,
            authority_usable=visible and not ambiguous,
            race_finished=False,
        )
        self.camera_publications.append(
            {
                "gate_index": self.current_gate,
                "sequence": token.publication_sequence,
                "publication_s": publication_s,
                "observed_s": self.clock,
                "state": state,
                "clipping": clipping,
                "visible": visible,
            }
        )

    def _publish_due_camera(self) -> None:
        if self.clock + 1e-12 < self._next_camera_s:
            return
        due_count = (
            math.floor(
                (self.clock - self._next_camera_s)
                / _CAMERA_PERIOD_S
            )
            + 1
        )
        publication_s = (
            self._next_camera_s
            + (due_count - 1) * _CAMERA_PERIOD_S
        )
        self._next_camera_s += due_count * _CAMERA_PERIOD_S
        self.sequence += due_count

        state = "clean"
        if self.current_gate in self._censor_step:
            step = self._censor_step[self.current_gate]
            state = (
                "bottom"
                if step == 0
                else "top_bottom"
                if step == 1
                else "lost"
            )
            self._censor_step[self.current_gate] = step + 1
        self._install_snapshot(
            token=_token(self.sequence),
            publication_s=publication_s,
            state=state,
        )

    def _arm_new_near_plane_latches(self) -> None:
        latched_gate_indexes = {
            int(payload["gate_index"])
            for event, payload in self.recorder.events
            if event == "visual_course_near_plane_latched"
        }
        for gate_index in latched_gate_indexes - self.anchor_times.keys():
            accepted_wires = [
                wire
                for wire in self.navigation_wires
                if wire["gate_index"] == gate_index
            ]
            assert accepted_wires
            anchor_s = float(accepted_wires[-1]["wire_start_s"])
            self.anchor_times[gate_index] = anchor_s
            self.credit_due_times[gate_index] = (
                anchor_s + self.credit_delay_s
            )
            self._censor_step[gate_index] = 0
            if not self._overrun_injected:
                # One receiver/control overrun forces the production
                # scheduler to drop missed 50 Hz ticks and the latest-frame
                # receiver to replace at least one 30 Hz publication.
                self.clock += 0.075
                self._overrun_injected = True

    def _publish_due_credit(self) -> None:
        gate_index = self.current_gate
        due_s = self.credit_due_times.get(gate_index)
        if (
            self.credit_policy != "delayed"
            or due_s is None
            or self.clock + 1e-12 < due_s
        ):
            return
        self.disable_credit = False
        try:
            super()._advance_race()
        finally:
            self.disable_credit = True
        self.credit_publish_times[gate_index] = self.clock
        self.credit_tokens_by_gate[gate_index] = self.credit_token
        del self.credit_due_times[gate_index]

    def _sample(self) -> None:
        self.control_sample_times.append(self.clock)
        self._arm_new_near_plane_latches()
        self._publish_due_camera()
        self._publish_due_credit()

    async def _send_flight_command(self, command, **kwargs):
        gate_index = self.current_gate
        wire_start_s = self.clock
        receipt = await super()._send_flight_command(command, **kwargs)
        wire_token = kwargs.get("wire_visual_token")
        if wire_token is not None:
            self.navigation_wires.append(
                {
                    "gate_index": gate_index,
                    "wire_start_s": wire_start_s,
                    "token": wire_token,
                    "command": command,
                }
            )
        return receipt

    def _confirm_visual_transition(self, **kwargs):
        transition = super()._confirm_visual_transition(**kwargs)
        self._track_histories.setdefault(transition.promoted_track_id, [])
        return transition


def _cadenced_runtime(host, *, limits=None, yaw_rate=0.0):
    runtime, _calls = _runtime(host, limits=limits)
    servo_calls = []

    def servo_factory(*args, **kwargs):
        return _CadencedCoordinatorServo(
            *args,
            **kwargs,
            calls=servo_calls,
            yaw_rate=yaw_rate,
        )

    return replace(runtime, servo_factory=servo_factory)


def _state_suffix(host, gate_index):
    return [
        publication["state"]
        for publication in host.camera_publications
        if publication["gate_index"] == gate_index
        and publication["state"] != "clean"
    ][:3]


def test_coordinator_replay_crosses_censor_gap_promotes_and_commands_next_gate():
    host = _CadencedCoordinatorHost(
        credit_policy="delayed",
        finish_gate=2,
    )

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_cadenced_runtime(host),
        )
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert [
        (transition["from_gate_index"], transition["to_gate_index"])
        for transition in result["authoritative_transitions"]
    ] == [(1, 2)]
    assert _state_suffix(host, 1) == ["bottom", "top_bottom", "lost"]
    assert _state_suffix(host, 2) == ["bottom", "top_bottom", "lost"]

    first_segment = result["segments"][0]
    transition = result["authoritative_transitions"][0]
    assert first_segment["near_plane_latch"] is not None
    assert first_segment["near_plane_measurement_mode"] == "credit_wait"
    assert first_segment["lifecycle"] == "promote_reacquire"
    assert first_segment["censored_passage_coast_command_count"] >= 2
    assert first_segment["crossing_wait_zero_command_count"] == 0
    assert first_segment["crossing_wait_coast_command_count"] >= 1
    assert transition["post_transition_navigation_command_count"] >= 1

    first_next_navigation = next(
        wire
        for wire in host.navigation_wires
        if wire["gate_index"] == 2
    )
    credit_token = host.credit_tokens_by_gate[1]
    assert (
        first_next_navigation["token"].publication_sequence
        > credit_token.publication_sequence
    )
    assert (
        first_next_navigation["wire_start_s"]
        >= host.credit_publish_times[1]
    )
    assert transition["recovery_admission"]["admission_kind"] == (
        "fresh_promoted_current"
    )
    assert (
        transition["recovery_admission"]["admitted_frame_token"][
            "publication_sequence"
        ]
        == first_next_navigation["token"].publication_sequence
    )

    for gate_index in (1, 2):
        delay_s = (
            host.credit_publish_times[gate_index]
            - host.anchor_times[gate_index]
        )
        assert _RACE_STATUS_PERIOD_S <= delay_s < 0.29

    publication_pairs = zip(
        host.camera_publications,
        host.camera_publications[1:],
    )
    observed_replacement_gap = False
    for previous, current in publication_pairs:
        sequence_gap = current["sequence"] - previous["sequence"]
        time_gap = current["publication_s"] - previous["publication_s"]
        assert time_gap / sequence_gap == pytest.approx(
            _CAMERA_PERIOD_S,
            abs=1e-12,
        )
        observed_replacement_gap |= sequence_gap > 1
    assert observed_replacement_gap

    control_gaps = [
        current - previous
        for previous, current in zip(
            host.control_sample_times,
            host.control_sample_times[1:],
        )
    ]
    assert all(gap >= _CONTROL_PERIOD_S - 1e-12 for gap in control_gaps)
    assert any(gap >= 0.06 for gap in control_gaps)


def test_coordinator_replay_no_credit_times_out_on_bounded_hold():
    limits = replace(
        VisualCourseStageLimits(),
        crossing_status_timeout_s=0.08,
    )
    host = _CadencedCoordinatorHost(
        credit_policy="none",
        finish_gate=1,
    )

    with pytest.raises(SafetyAbort, match="credit timed out"):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_cadenced_runtime(host, limits=limits),
            )
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["near_plane_latch"] is not None
    assert segment["near_plane_measurement_mode"] == "credit_wait"
    assert segment["crossing_wait_zero_command_count"] == 0
    assert segment["crossing_wait_coast_command_count"] >= 1
    assert host.requested_promotion_track_ids == []
    assert host.credit_publish_times == {}


def test_coordinator_replay_off_center_censor_aborts_before_coast_or_credit():
    host = _CadencedCoordinatorHost(
        credit_policy="none",
        unsafe_after_latch=True,
        finish_gate=1,
    )

    with pytest.raises(
        SafetyAbort,
        match="latched near-plane measurement became unsafe",
    ):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_cadenced_runtime(host),
            )
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["near_plane_latch"] is not None
    assert segment["censored_passage_coast_command_count"] == 0
    assert segment["crossing_wait_zero_command_count"] == 0
    assert segment["crossing_wait_coast_command_count"] == 0
    assert host.requested_promotion_track_ids == []


_ATTEMPT5_POST_CREDIT_ROWS = {
    180: {
        "center": (0.59375, -0.6555555555555556),
        "velocity": (0.27821149946603263, -0.5777751631733091),
        "scale": 0.1957335738815506,
        "confidence": 0.7872491939937416,
        "association": 0.9297907186380104,
        "clipping": FrameEdge.NONE,
        "center_censored": False,
    },
    181: {
        "center": (0.6031249999999999, -0.6722222222222223),
        "velocity": (0.27110915873146013, -0.5194014615999842),
        "scale": 0.2019866607256804,
        "confidence": 0.7945289760068612,
        "association": 0.9345003622370058,
        "clipping": FrameEdge.NONE,
        "center_censored": False,
    },
    182: {
        "center": (0.6125, -0.6944444444444444),
        "velocity": (0.3144518037296715, -0.6899147935434264),
        "scale": 0.20823956223008577,
        "confidence": 0.8015915392030875,
        "association": 0.9297269854038298,
        "clipping": FrameEdge.NONE,
        "center_censored": False,
    },
    183: {
        "center": (0.621875, -0.7222222222222222),
        "velocity": (0.2885910808222251, -0.7462772693726882),
        "scale": 0.21556208824579728,
        "confidence": 0.8131901926413894,
        "association": 0.8605656550353479,
        "clipping": FrameEdge.TOP,
        "center_censored": True,
    },
}


class _Attempt5RecoveryHost(_CoordinatorHost):
    """Coordinator fact replay with exact logged post-credit tracker rows.

    Planner proposals, IMU, and stage boundaries remain deterministic mocks;
    this is not JPEG, detector, or full tracker replay.
    """

    def __init__(self) -> None:
        super().__init__(
            initial_gate=3,
            finish_gate=99,
            fresh_after_samples=1,
        )
        self.clock = 3.50
        self.sequence = 175
        self.visual_gate_graph.latest_snapshot = _snapshot(
            self.current_gate,
            self.current_track_id,
            self.sequence,
        )

    def _watchdog(self, **kwargs):
        assert kwargs["enforce_benign_pad_budget"] is True
        assert type(kwargs["allow_benign_pad_contact"]) is bool
        self.watchdogs += 1

    def _sample(self) -> None:
        # Credit is received during the accepted publication-179 wire.  Keep
        # that exact camera watermark until promotion consumes the boundary.
        if (
            self.after_promotion_samples is None
            and not self.race.race_finished
            and self.race.active_gate_index == self.current_gate + 1
        ):
            return
        super()._sample()
        if self.after_promotion_samples is None:
            return
        snapshot = self.visual_gate_graph.latest_snapshot
        track = snapshot.current_track
        row = _ATTEMPT5_POST_CREDIT_ROWS.get(self.sequence)
        if row is not None:
            track.center_norm = row["center"]
            track.center_velocity_norm_s = row["velocity"]
            track.apparent_scale = row["scale"]
            track.confidence = row["confidence"]
            track.association_confidence = row["association"]
            track.clipping = row["clipping"]
            track.center_censored = row["center_censored"]
            snapshot.authority_usable = True
            return
        if self.sequence >= 184:
            track.visible = False
            track.missed_frame_count = 1
            track.latest_token = _token(self.sequence - 1)
            snapshot.authority_usable = False


class _Attempt5RecoveryServo(_CoordinatorServo):
    """Expose exact target facts while retaining deterministic wire control."""

    def observe(self, snapshot, *args, **kwargs):
        proposal = super().observe(snapshot, *args, **kwargs)
        if kwargs["mode"] is not VisualApproachMode.PROMOTE_REACQUIRE:
            return proposal
        track = snapshot.current_track
        latest = track.history[-1]
        clipping = track.clipping
        target = replace(
            proposal.current_target,
            received_monotonic_s=(
                latest.observation_monotonic_ns / 1_000_000_000.0
            ),
            normalized_x=track.center_norm[0],
            normalized_y_down=track.center_norm[1],
            normalized_x_rate_s=track.center_velocity_norm_s[0],
            normalized_y_rate_down_s=track.center_velocity_norm_s[1],
            log_scale=math.log(track.apparent_scale),
            confidence=track.confidence,
            association_confidence=track.association_confidence,
            clipped=clipping != FrameEdge.NONE,
            center_censored=track.center_censored,
            horizontal_censored=bool(
                clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
            ),
            vertical_censored=bool(
                clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
            ),
        )
        one_edge = clipping != FrameEdge.NONE
        if not one_edge:
            self._last_vertical_observable_thrust = 0.29
        retained_thrust = getattr(
            self,
            "_last_vertical_observable_thrust",
            proposal.servo_output.thrust,
        )
        proposal.current_target = target
        proposal.passage_admission = None
        proposal.servo_output = replace(
            proposal.servo_output,
            target_pitch_rad=0.035 if one_edge else 0.08,
            yaw_rate_rad_s=-0.08,
            thrust=retained_thrust,
            corridor_frames=0,
            advance_enabled=False,
            next_gate_blend=0.0,
            brake_reason=(
                "target_edge_or_clipping" if one_edge else "aligning"
            ),
        )
        return proposal


def _attempt5_runtime(host):
    limits = replace(
        VisualCourseStageLimits(),
        post_credit_fresh_frame_timeout_s=0.05,
    )
    runtime, _calls = _runtime(host, limits=limits)
    servo_calls = []

    def servo_factory(*args, **kwargs):
        return _Attempt5RecoveryServo(
            *args,
            **kwargs,
            calls=servo_calls,
            yaw_rate=0.0,
        )

    return replace(runtime, servo_factory=servo_factory)


def test_attempt5_clean_reacquisition_releases_recovery_before_clipping():
    """Two clean accepted rows restore ordinary current-gate ownership."""

    host = _Attempt5RecoveryHost()

    with pytest.raises(
        SafetyAbort,
        match="current identity is no longer visible",
    ):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_attempt5_runtime(host),
            )
        )

    recovery_wires = [
        (kwargs["wire_visual_token"].publication_sequence, stage)
        for (
            (_command, kwargs, gate_index),
            (stage, _elapsed_s, _recorded_command),
        ) in zip(host.commands, host.ticks)
        if (
            gate_index == 4
            and kwargs.get("wire_visual_token") is not None
            and "/recovery-" in stage
        )
    ]
    assert [sequence for sequence, _stage in recovery_wires] == [180, 181]
    assert all(stage.endswith("recovery-clean") for _, stage in recovery_wires)
    assert any(
        gate_index == 4 and stage.endswith("/approach")
        for (
            (_command, _kwargs, gate_index),
            (stage, _elapsed_s, _recorded_command),
        ) in zip(host.commands, host.ticks)
    )

    transition = host._visual_course_summary[
        "authoritative_transitions"
    ][0]
    assert transition["from_gate_index"] == 3
    assert transition["to_gate_index"] == 4
    assert transition["recovery_admission"]["admitted_frame_token"][
        "publication_sequence"
    ] == 180
    assert transition["recovery_admission"]["wire_frame_token"][
        "publication_sequence"
    ] == 180
    recovery = host._visual_course_summary["segments"][1]
    assert recovery["lifecycle"] != "promote_reacquire"
    assert recovery["recovery_clean_command_count"] == 2
    assert recovery["recovery_one_edge_command_count"] == 0
    assert recovery["recovery_support_command_count"] == 0
    completed = [
        fields
        for name, fields in host.recorder.events
        if name == "visual_course_recovery_completed"
    ]
    assert len(completed) == 1
    assert completed[0]["clean_command_count"] == 2


class _Attempt5OutsideImageHost(_Attempt5RecoveryHost):
    """Counterfactual unsafe mutation before clean recovery is complete."""

    def _sample(self):
        super()._sample()
        if self.after_promotion_samples is not None and self.sequence == 181:
            track = self.visual_gate_graph.latest_snapshot.current_track
            track.center_norm = (
                1.01,
                -0.6722222222222223,
            )
            track.clipping = FrameEdge.TOP
            track.center_censored = True


def test_recovery_outside_observable_axis_aborts_before_second_clean_wire():
    host = _Attempt5OutsideImageHost()

    with pytest.raises(
        SafetyAbort,
        match="post-credit recovery measurement became unsafe",
    ):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_attempt5_runtime(host),
            )
        )

    assert [
        kwargs["wire_visual_token"].publication_sequence
        for _command, kwargs, gate_index in host.commands
        if gate_index == 4 and kwargs.get("wire_visual_token") is not None
    ] == [180]


class _RecoverySlotReplacementHost(_CadencedCoordinatorHost):
    """Publish one receiver frame during the admitted recovery wire wait."""

    def __init__(self) -> None:
        super().__init__(
            credit_policy="delayed",
            finish_gate=2,
        )
        self.recovery_slot_replacement_injected = False
        self.recovery_replacement_sequence = None
        self.recovery_replacement_received_s = None
        self.recovery_refresh_sample_s = None
        self.receiver_camera_token = (
            self.visual_gate_graph.latest_snapshot.latest_camera_token
        )

    async def _wait_for_next_flight_command_slot(self):
        ready = await super()._wait_for_next_flight_command_slot()
        if (
            not self.recovery_slot_replacement_injected
            and self.current_gate == 2
            and self.after_promotion_samples is not None
            and self.visual_gate_graph.latest_snapshot.current_track.visible
        ):
            self.clock += 0.021
            self._next_camera_s = min(
                self._next_camera_s,
                self.clock,
            )
            self.receiver_camera_token = _token(self.sequence + 1)
            self.recovery_slot_replacement_injected = True
            self.recovery_replacement_received_s = self.clock
            self.recovery_replacement_sequence = (
                self.receiver_camera_token.publication_sequence
            )
        return ready

    def _sample(self):
        graph_token = (
            self.visual_gate_graph.latest_snapshot.latest_camera_token
        )
        if self.receiver_camera_token != graph_token:
            self.recovery_refresh_sample_s = self.clock
        super()._sample()
        self.receiver_camera_token = (
            self.visual_gate_graph.latest_snapshot.latest_camera_token
        )

    def _assert_visual_receiver_token_current(self, expected_token):
        receiver_token = self.receiver_camera_token
        if receiver_token != expected_token:
            exc = SafetyAbort(
                VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
            )
            exc.expected_visual_token = expected_token
            exc.receiver_visual_token = receiver_token
            raise exc
        return receiver_token


def test_recovery_replans_when_receiver_replaces_candidate_in_wire_slot():
    host = _RecoverySlotReplacementHost()

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_cadenced_runtime(host),
        )
    )

    assert result["race_finished"] is True
    assert host.recovery_slot_replacement_injected
    assert host.recovery_refresh_sample_s == pytest.approx(
        host.recovery_replacement_received_s,
        abs=1e-12,
    )
    transition = result["authoritative_transitions"][0]
    admission = transition["recovery_admission"]
    admitted_sequence = admission["admitted_frame_token"][
        "publication_sequence"
    ]
    wire_sequence = admission["wire_frame_token"][
        "publication_sequence"
    ]
    assert wire_sequence == host.recovery_replacement_sequence
    assert wire_sequence > admitted_sequence
    assert all(
        wire["token"].publication_sequence != admitted_sequence
        for wire in host.navigation_wires
        if wire["gate_index"] == 2
    )
    recovery_segment = result["segments"][1]
    assert recovery_segment["superseded_proposal_count"] >= 1
    assert recovery_segment["recovery_clean_command_count"] >= 2


class _RecoveryClippedReplacementHost(_RecoverySlotReplacementHost):
    """Keep replacement publications TOP-censored until the first wire."""

    def __init__(self) -> None:
        super().__init__()
        self.one_edge_wire_sent = False

    def _censor_recovery_current(self):
        if (
            self.one_edge_wire_sent
            or not self.recovery_slot_replacement_injected
            or self.current_gate != 2
            or self.after_promotion_samples is None
        ):
            return
        snapshot = self.visual_gate_graph.latest_snapshot
        snapshot.current_track.clipping = FrameEdge.TOP
        snapshot.current_track.center_censored = True

    async def _send_flight_command(self, command, **kwargs):
        receipt = await super()._send_flight_command(command, **kwargs)
        if (
            self.current_gate == 2
            and kwargs.get("wire_visual_token") is not None
        ):
            self.one_edge_wire_sent = True
        return receipt

    def _sample(self):
        super()._sample()
        self._censor_recovery_current()

    async def _wait_for_next_flight_command_slot(self):
        ready = await super()._wait_for_next_flight_command_slot()
        self._censor_recovery_current()
        return ready


def test_recovery_one_edge_commands_before_a_clean_wire():
    host = _RecoveryClippedReplacementHost()

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_cadenced_runtime(host, yaw_rate=-0.08),
        )
    )

    assert result["race_finished"] is True
    assert host.recovery_slot_replacement_injected
    recovery_wires = [
        wire
        for wire in host.navigation_wires
        if wire["gate_index"] == 2
    ]
    assert recovery_wires
    assert recovery_wires[0]["command"].yaw_rate != 0.0
    recovery = host._visual_course_summary["segments"][1]
    assert recovery["recovery_navigation_command_count"] == (
        recovery["recovery_one_edge_command_count"]
        + recovery["recovery_clean_command_count"]
    )
    assert recovery["recovery_one_edge_command_count"] >= 1
    assert recovery["recovery_clean_command_count"] >= 1
    transition = host._visual_course_summary[
        "authoritative_transitions"
    ][0]
    assert transition["recovery_admission"]["wire_frame_token"] == {
        "stream_id": recovery_wires[0]["token"].stream_id,
        "generation": recovery_wires[0]["token"].generation,
        "frame_id": recovery_wires[0]["token"].frame_id,
        "publication_sequence": (
            recovery_wires[0]["token"].publication_sequence
        ),
    }


class _C25DynamicCore:
    def __init__(self, track_id):
        self.config = SimpleNamespace(
            vertical_angle_scale_rad=1.0,
            terminal_min_post_governor_contact_budget_s=0.12,
        )
        self._state = SimpleNamespace(
            current_track_id=track_id,
            current=SimpleNamespace(
                residual_translational_rate_rad_s=(
                    0.121279229347492,
                    0.20443111726250962,
                )
            ),
        )

    def course_state(self):
        return self._state


class _C25DynamicController:
    """Exact terminal c25 q/collective facts with a deterministic governor."""

    def __init__(self, track_id, gate_index):
        self.track_id = track_id
        self.gate_index = gate_index
        self.core = _C25DynamicCore(track_id)
        self.last_decision = SimpleNamespace(
            current_track_id=track_id,
            current_center_norm=(
                0.052115845809853005,
                -1.0958578515774178,
            ),
        )
        self.govern_count = 0
        self.accepted_count = 0

    def govern_wire_command(
        self,
        command,
        *,
        proposal_monotonic_ns,
        launch_thrust_override,
        yaw_safety_override,
    ):
        assert type(proposal_monotonic_ns) is int
        assert launch_thrust_override is False
        assert yaw_safety_override is False
        governed_thrust = (
            0.30799241399874683
            if self.govern_count == 0
            else 0.304
        )
        self.govern_count += 1
        return replace(command, thrust=governed_thrust)

    def record_wire_acceptance(
        self,
        *,
        target_roll_rad,
        target_pitch_rad,
        yaw_rate_rad_s,
        thrust,
        wire_command,
        wire_start_monotonic_ns,
        requested_thrust=None,
        thrust_slew_override=False,
        yaw_slew_override=False,
    ):
        effective_requested_thrust = (
            thrust if requested_thrust is None else requested_thrust
        )
        assert 0.21 <= effective_requested_thrust <= 0.32
        assert thrust_slew_override is False
        assert yaw_slew_override is False
        self.accepted_count += 1
        return {
            "schema": "aigp-vq2-dynamic-command/1",
            "controller_family": "aigp-vq2-dynamic-image-course/1",
            "applied_command_count": self.accepted_count,
            "dynamic_command_count": self.accepted_count,
            "roll_reversal_count": 0,
            "wire_start_monotonic_ns": wire_start_monotonic_ns,
            "target_attitude_yaw_thrust": [
                target_roll_rad,
                target_pitch_rad,
                yaw_rate_rad_s,
                thrust,
            ],
            "wire_command": {
                "roll_rate": wire_command.roll_rate,
                "pitch_rate": wire_command.pitch_rate,
                "yaw_rate": wire_command.yaw_rate,
                "thrust": wire_command.thrust,
            },
            "gate_index": self.gate_index,
            "current_track_id": self.track_id,
            "successor_track_id": "track-4",
            "current_center_norm": [
                0.052115845809853005,
                -1.0958578515774178,
            ],
            "camera_current_center_norm": [
                0.0599040959869629,
                -0.3384720638086744,
            ],
            "passage_point_norm": [
                0.2553373231601403,
                -0.23547313447361046,
            ],
            "successor_passage_authority": 0.9171578072505727,
            "passage_error_norm": [
                0.3074531689699933,
                -1.3313309860510283,
            ],
            "aperture_margin_norm": [
                0.045211055792231,
                0.4389479208946809,
            ],
            "crossing_prediction_horizon_s": 0.7509888689491439,
            "crossing_coordinate_basis": (
                course_stage.DYNAMIC_CROSSING_COORDINATE_BASIS
            ),
            "current_crossing_error_q": [
                0.7872345285230035,
                -1.7416199837792337,
            ],
            "crossing_rate_q_s": [
                0.1328463672794427,
                2.1763541677795883,
            ],
            "predicted_crossing_error_norm": [
                0.8870006716301947,
                -0.10720222888568531,
            ],
            "predicted_crossing_std_norm": [
                0.09250658369983762,
                0.16712723618103104,
            ],
            "crossing_allowance_norm": [0.50, 0.45],
            "crossing_swept_occupancy_norm": [
                1.7258056275019444,
                2.2968614309187667,
            ],
            "predicted_crossing_clearance_norm": [
                -1.2258056275019444,
                -1.8468614309187668,
            ],
            "terminal_crossing_occupancy_norm": [
                1.07201383902987,
                0.4414567012477474,
            ],
            "terminal_crossing_clearance_norm": [
                -0.57201383902987,
                0.00854329875225257,
            ],
            "post_governor_contact_budget_s": (
                0.7509888689491439
                - 0.08
                - abs(
                    wire_command.thrust
                    - effective_requested_thrust
                )
                / 0.15
            ),
            "current_bearing_std_norm": [
                0.018934613171084462,
                0.024523885374696446,
            ],
            "residual_translation_rate_norm_s": [
                0.121279229347492,
                0.20443111726250962,
            ],
            "current_censored_axes": [False, False],
            "current_bearing_rate_qualified": [True, True],
            "current_scale_rate_qualified": True,
            "current_log_scale": -0.7155249885531791,
            "current_log_scale_std": 0.052547919807168456,
            "expansion_rate_s": 1.3315776589329433,
            "current_confidence": 0.749706062324875,
            "current_ambiguous": False,
            "current_visible": True,
            "dropout_held": False,
            "time_to_contact_s": 0.7509888689491439,
            "braking": True,
            "brake_reason": "vertical_alignment_unsettled",
            "passage_scale_ready": False,
            "successor_yaw_contribution_rad": 0.0,
        }

    def evidence_summary(self):
        return {
            "schema": "aigp-vq2-dynamic-controller-summary/1",
            "controller_family": "aigp-vq2-dynamic-image-course/1",
            "applied_command_count": self.accepted_count,
            "dynamic_command_count": self.accepted_count,
            "roll_reversal_count": 0,
            "track_count": 2,
            "promotion_count": 0,
        }

    def continuity_hold_authority(self, **_kwargs):
        raise AssertionError("c25 replay must not enter a handoff hold")


class _C25ApproachTopHost(_CoordinatorHost):
    def __init__(self):
        super().__init__(
            initial_gate=3,
            finish_gate=99,
            disable_credit=True,
        )

    def _sample(self):
        super()._sample()
        snapshot = self.visual_gate_graph.latest_snapshot
        track = snapshot.current_track
        recovery_sent = any(
            stage.endswith("approach-top-recovery")
            for stage, _elapsed_s, _command in self.ticks
        )
        if not self.ticks:
            track.center_norm = (
                0.0599040959869629,
                -0.3388888888888889,
            )
            track.center_velocity_norm_s = (
                0.121279229347492,
                0.6039185423722361,
            )
            track.apparent_scale = math.exp(-0.7155249885531791)
            return
        track.center_norm = (0.103125, -0.2944444444444444)
        track.center_velocity_norm_s = (
            0.6216978916774404,
            0.6637248571542275,
        )
        track.apparent_scale = 0.590227
        track.clipping = (
            FrameEdge.TOP | FrameEdge.BOTTOM
            if recovery_sent
            else FrameEdge.TOP
        )
        track.center_censored = True


class _C25ApproachTopServo(_CoordinatorServo):
    def observe(self, snapshot, *args, **kwargs):
        track = snapshot.current_track
        if track.clipping != FrameEdge.NONE:
            raise VisualApproachCurrentGeometryUnavailable(
                "authoritative current aperture is clipped or censored"
            )
        proposal = super().observe(snapshot, *args, **kwargs)
        latest = track.history[-1]
        proposal.current_target = replace(
            proposal.current_target,
            received_monotonic_s=(
                latest.observation_monotonic_ns / 1_000_000_000.0
            ),
            normalized_x=track.center_norm[0],
            normalized_y_down=track.center_norm[1],
            normalized_x_rate_s=track.center_velocity_norm_s[0],
            normalized_y_rate_down_s=track.center_velocity_norm_s[1],
            log_scale=math.log(track.apparent_scale),
            log_scale_rate_s=1.3315776589329433,
        )
        proposal.servo_output = replace(
            proposal.servo_output,
            target_roll_rad=0.08288152550333001,
            target_pitch_rad=0.12,
            yaw_rate_rad_s=-0.010391620866495196,
            thrust=0.275,
            corridor_frames=0,
            advance_enabled=False,
            next_gate_blend=0.0,
            brake_reason="vertical_alignment_unsettled",
        )
        proposal.passage_admission = None
        return proposal


def _c25_approach_top_runtime(host):
    runtime, _calls = _runtime(host)
    dynamic = _C25DynamicController(
        host.current_track_id,
        host.current_gate,
    )
    servo_calls = []

    def servo_factory(*args, **kwargs):
        return _C25ApproachTopServo(
            *args,
            **kwargs,
            calls=servo_calls,
        )

    return replace(
        runtime,
        servo_factory=servo_factory,
        dynamic_controller=dynamic,
    )


def test_c25_top_only_approach_recovery_is_bounded_and_never_latches():
    host = _C25ApproachTopHost()

    with pytest.raises(
        SafetyAbort,
        match="authoritative current aperture is clipped or censored",
    ):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_c25_approach_top_runtime(host),
            )
        )

    recovery_wires = [
        command
        for (
            (command, _kwargs, _gate_index),
            (stage, _elapsed_s, _recorded_command),
        ) in zip(host.commands, host.ticks)
        if stage.endswith("approach-top-recovery")
    ]
    assert len(recovery_wires) == 1
    assert recovery_wires[0].thrust == pytest.approx(0.304)
    assert recovery_wires[0].thrust < host.commands[0][0].thrust
    assert recovery_wires[0].thrust > 0.2892416792249238

    segment = host._visual_course_summary["segments"][0]
    assert segment["lifecycle"] == "approach"
    assert segment["passage_authority_enabled"] is False
    assert segment["near_plane_latch"] is None
    assert segment["crossing_anchor"] is None
    assert segment["approach_top_recovery_command_count"] == 1
    assert segment["approach_top_recovery_fresh_frame_count"] == 1
    recovery = segment["approach_top_recovery"]
    assert recovery["basis"] == course_stage.APPROACH_TOP_RECOVERY_BASIS
    assert recovery["vertical_endpoint_occupancy_q"] < (
        recovery["vertical_allowance_q"]
    )
    assert recovery["requested_thrust"] == pytest.approx(
        0.2892416792249238
    )
    assert not any(
        event == "visual_course_near_plane_latched"
        for event, _payload in host.recorder.events
    )


class _AdmissionlessCadencedServo(_CadencedCoordinatorServo):
    """Expose a safe dynamic crossing without planner admission."""

    def observe(self, *args, **kwargs):
        proposal = super().observe(*args, **kwargs)
        proposal.passage_admission = None
        return proposal


class _AtomicCrossingDynamicController(_C25DynamicController):
    """Emit one internally consistent accepted-wire crossing candidate."""

    def __init__(self, track_id, gate_index, *, safe_clearance):
        super().__init__(track_id, gate_index)
        self.safe_clearance = safe_clearance

    def record_wire_acceptance(self, **kwargs):
        evidence = super().record_wire_acceptance(**kwargs)
        if self.safe_clearance:
            evidence.update(
                {
                    "predicted_crossing_error_norm": [0.10, 0.10],
                    "predicted_crossing_std_norm": [0.04, 0.04],
                    "crossing_swept_occupancy_norm": [0.18, 0.18],
                    "predicted_crossing_clearance_norm": [0.32, 0.27],
                    "terminal_crossing_occupancy_norm": [0.18, 0.18],
                    "terminal_crossing_clearance_norm": [0.32, 0.27],
                    "post_governor_contact_budget_s": 0.40,
                    "brake_reason": "aligning",
                }
            )
        return evidence


def _atomic_crossing_runtime(host, *, safe_clearance, limits=None):
    runtime = _cadenced_runtime(host, limits=limits)
    servo_calls = []

    def servo_factory(*args, **kwargs):
        return _AdmissionlessCadencedServo(
            *args,
            **kwargs,
            calls=servo_calls,
            yaw_rate=0.0,
        )

    return replace(
        runtime,
        servo_factory=servo_factory,
        dynamic_controller=_AtomicCrossingDynamicController(
            host.current_track_id,
            host.current_gate,
            safe_clearance=safe_clearance,
        ),
    )


def test_dynamic_latch_without_admission_atomically_coasts_to_finish():
    host = _CadencedCoordinatorHost(
        credit_policy="delayed",
        finish_gate=1,
        # Later than the accepted command/state lease: committed crossing
        # reaches predicted contact, then exact zero preserves a separate
        # bounded authoritative-status ingress window.
        credit_delay_s=1.15,
    )

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_atomic_crossing_runtime(
                host,
                safe_clearance=True,
            ),
        )
    )

    assert result["race_finished"] is True
    assert _state_suffix(host, 1) == ["bottom", "top_bottom", "lost"]
    segment = result["segments"][0]
    assert segment["passage_admission"] is None
    assert segment["near_plane_latch"] is not None
    assert segment["crossing_anchor"] == segment["near_plane_latch"]
    assert segment["passage_authority_enabled"] is True
    assert segment["near_plane_latch"]["commitment_horizon_s"] < 1.15
    assert segment["censored_passage_coast_command_count"] >= 2
    assert segment["crossing_wait_coast_command_count"] >= 1
    assert segment["crossing_wait_zero_command_count"] >= 1
    assert [
        event
        for event, _payload in host.recorder.events
        if event == "visual_course_near_plane_latched"
    ] == ["visual_course_near_plane_latched"]


def test_dynamic_negative_clearance_without_admission_never_latches():
    limits = replace(
        VisualCourseStageLimits(),
        segment_hard_duration_s=0.50,
        passage_hard_duration_s=0.40,
    )
    host = _CadencedCoordinatorHost(
        credit_policy="none",
        finish_gate=1,
    )

    with pytest.raises(
        SafetyAbort,
        match="visual-course gate 1 segment expired",
    ):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_atomic_crossing_runtime(
                    host,
                    safe_clearance=False,
                    limits=limits,
                ),
            )
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["passage_admission"] is None
    assert segment["passage_authority_enabled"] is False
    assert segment["near_plane_latch"] is None
    assert segment["crossing_anchor"] is None
    assert not any(
        event == "visual_course_near_plane_latched"
        for event, _payload in host.recorder.events
    )
