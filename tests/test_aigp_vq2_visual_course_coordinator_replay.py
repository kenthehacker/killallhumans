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
)
from scripts.aigp_vq2_visual_course_stage import (
    VisualCourseStageLimits,
    run_visual_course_stage,
)
from tests.test_aigp_vq2_visual_course import (
    SafetyAbort,
    _Host as _CoordinatorHost,
    _Servo as _CoordinatorServo,
    _context,
    _runtime,
    _token,
)


_CAMERA_PERIOD_S = 1.0 / 30.0
_CONTROL_PERIOD_S = 0.02
_RACE_STATUS_PERIOD_S = 0.25


class _CadencedCoordinatorServo(_CoordinatorServo):
    """Keep planning deterministic while exposing real coordinator refusals."""

    def observe(self, snapshot, *args, **kwargs):
        track = snapshot.current_track
        if (
            not track.visible
            or track.clipping != FrameEdge.NONE
            or track.center_censored
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
    ) -> None:
        super().__init__(
            initial_gate=1,
            finish_gate=finish_gate,
            disable_credit=True,
        )
        self.credit_policy = credit_policy
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
                anchor_s + _RACE_STATUS_PERIOD_S
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


def _cadenced_runtime(host, *, limits=None):
    runtime, _calls = _runtime(host, limits=limits)
    servo_calls = []

    def servo_factory(*args, **kwargs):
        return _CadencedCoordinatorServo(
            *args,
            **kwargs,
            calls=servo_calls,
            yaw_rate=0.0,
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
    assert first_segment["crossing_wait_zero_command_count"] >= 1
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


def test_coordinator_replay_no_credit_times_out_on_zero_authority():
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
    assert segment["crossing_wait_zero_command_count"] >= 1
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
    assert host.requested_promotion_track_ids == []
