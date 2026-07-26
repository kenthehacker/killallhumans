"""Coordinator regressions for credited-unbound visual reacquisition.

These tests replay explicit tracker/graph/IMU/race/wire boundary facts through
the real visual-course coordinator.  They do not replay JPEGs, the detector,
the camera receiver, or FlightSim dynamics.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    VisualTrackRole,
)
from planning.vq2_gate_graph import (
    ConfirmedGateReacquisition,
    CreditedUnboundGateAdvance,
    GateReacquisitionPending,
)
from scripts.aigp_vq2_visual_course_stage import (
    VisualCourseStageLimits,
    run_visual_course_stage,
)
from tests.test_aigp_vq2_visual_course import (
    SafetyAbort,
    _Host as _CoordinatorHost,
    _assert_course_zero_receipts,
    _context,
    _history_sample,
    _runtime,
    _token,
)


def _unbound_snapshot(gate_index: int, sequence: int) -> SimpleNamespace:
    """Expose a credited gate without inventing command-authoritative vision."""

    return SimpleNamespace(
        tracker_frame_sequence=sequence,
        latest_camera_token=_token(sequence),
        current_gate_index=gate_index,
        current_track_id=None,
        current_track=None,
        authority_usable=False,
        race_finished=False,
    )


def _clean_current_snapshot(
    gate_index: int,
    track_id: str,
    sequence: int,
    *,
    clipping: FrameEdge = FrameEdge.NONE,
) -> SimpleNamespace:
    """Build a stable local tail with a clean or one-edge latest sample."""

    tokens = tuple(_token(sequence - offset) for offset in (2, 1, 0))
    history = (
        _history_sample(track_id, tokens[0]),
        _history_sample(
            track_id,
            tokens[1],
            previous_token=tokens[0],
        ),
        _history_sample(
            track_id,
            tokens[2],
            previous_token=tokens[1],
            clipping=clipping,
            center_censored=clipping != FrameEdge.NONE,
        ),
    )
    track = SimpleNamespace(
        track_id=track_id,
        latest_token=tokens[-1],
        visible=True,
        ambiguous=False,
        missed_frame_count=0,
        consecutive_frame_count=len(history),
        confidence=0.90,
        association_confidence=0.80,
        clipping=clipping,
        center_censored=clipping != FrameEdge.NONE,
        center_norm=history[-1].center_norm,
        center_velocity_norm_s=(0.20, -0.20),
        apparent_scale=history[-1].apparent_scale,
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=gate_index,
        history=history,
    )
    return SimpleNamespace(
        tracker_frame_sequence=sequence,
        latest_camera_token=tokens[-1],
        current_gate_index=gate_index,
        current_track_id=track_id,
        current_track=track,
        authority_usable=True,
        race_finished=False,
    )


class _CreditedUnboundCoordinatorHost(_CoordinatorHost):
    """Fake host for exact credited-unbound coordinator state transitions."""

    def __init__(
        self,
        *,
        initial_gate: int,
        finish_gate: int,
        reacquire_after_samples: int | None = 2,
        ambiguous_reacquisition: bool = False,
        same_reviewed_identity: bool = False,
        finish_during_unbound_after_samples: int | None = None,
        binding_clipping: FrameEdge = FrameEdge.NONE,
    ) -> None:
        super().__init__(
            initial_gate=initial_gate,
            finish_gate=finish_gate,
        )
        self.reacquire_after_samples = reacquire_after_samples
        self.ambiguous_reacquisition = ambiguous_reacquisition
        self.same_reviewed_identity = same_reviewed_identity
        self.finish_during_unbound_after_samples = (
            finish_during_unbound_after_samples
        )
        self.binding_clipping = binding_clipping
        self.unbound_advance: CreditedUnboundGateAdvance | None = None
        self.reacquisition: ConfirmedGateReacquisition | None = None
        self.unbound_sample_count = 0
        self.reacquisition_poll_count = 0
        self.binding_snapshot: SimpleNamespace | None = None

    def _sample(self) -> None:
        if self.unbound_advance is None or self.reacquisition is not None:
            super()._sample()
            return
        self.sequence += 1
        self.unbound_sample_count += 1
        self.visual_gate_graph.latest_snapshot = _unbound_snapshot(
            self.current_gate,
            self.sequence,
        )
        if (
            self.finish_during_unbound_after_samples is not None
            and self.unbound_sample_count
            >= self.finish_during_unbound_after_samples
            and not self.race.race_finished
        ):
            self.credit_token = (
                self.visual_gate_graph.latest_snapshot.latest_camera_token
            )
            self.race = replace(
                self.race,
                race_status_sequence=self.race.race_status_sequence + 1,
                race_status_boot_ms=self.race.race_status_boot_ms + 250,
                received_monotonic_ns=max(
                    self.race.received_monotonic_ns + 1,
                    round(self.clock * 1_000_000_000),
                ),
                race_finished=True,
            )

    def _confirm_visual_course_advance(
        self,
        *,
        from_gate_index,
        to_gate_index,
        race_status,
        reviewed_track_id,
    ) -> CreditedUnboundGateAdvance:
        assert self.unbound_advance is None
        assert from_gate_index == self.current_gate
        assert to_gate_index == self.current_gate + 1
        assert race_status is self.race
        assert reviewed_track_id == f"track-{to_gate_index}"
        self.requested_promotion_track_ids.append(reviewed_track_id)
        assert self.credit_token is not None
        credit_sequence = self.credit_token.publication_sequence
        assert credit_sequence is not None
        advance = CreditedUnboundGateAdvance(
            from_gate_index=from_gate_index,
            to_gate_index=to_gate_index,
            retired_track_id=self.current_track_id,
            reviewed_track_id=reviewed_track_id,
            race_status=race_status,
            camera_token_at_credit=self.credit_token,
            reviewed_first_token=_token(credit_sequence - 2),
            reviewed_latest_token_before_credit=self.credit_token,
            reviewed_history_length_at_credit=3,
            reviewed_history_length_at_advance=3,
            reviewed_history_sha256="b" * 64,
            alternative_reacquisition_track_ids_at_credit=(
                ()
                if self.same_reviewed_identity
                else (f"track-{to_gate_index}-reacquired",)
            ),
        )
        self.unbound_advance = advance
        self.current_gate = to_gate_index
        self.current_track_id = f"unbound-gate-{to_gate_index}"
        self.visual_gate_graph.latest_snapshot = _unbound_snapshot(
            to_gate_index,
            self.sequence,
        )
        return advance

    def _try_visual_reacquired_current(
        self,
    ) -> ConfirmedGateReacquisition | GateReacquisitionPending:
        assert self.unbound_advance is not None
        self.reacquisition_poll_count += 1
        if self.ambiguous_reacquisition:
            return GateReacquisitionPending(
                reason=(
                    "two locally stable successor candidates are ambiguous"
                ),
                ambiguous=True,
            )
        if (
            self.reacquire_after_samples is None
            or self.unbound_sample_count < self.reacquire_after_samples
        ):
            return GateReacquisitionPending(
                reason=(
                    "no unique clean locally stable successor is ready"
                ),
                ambiguous=False,
            )

        reacquired_track_id = (
            self.unbound_advance.reviewed_track_id
            if self.same_reviewed_identity
            else f"track-{self.current_gate}-reacquired"
        )
        snapshot = _clean_current_snapshot(
            self.current_gate,
            reacquired_track_id,
            self.sequence,
            clipping=self.binding_clipping,
        )
        stable_tokens = tuple(
            sample.token for sample in snapshot.current_track.history
        )
        credit_token = self.unbound_advance.camera_token_at_credit
        credit_sequence = credit_token.publication_sequence
        first_sequence = stable_tokens[0].publication_sequence
        assert credit_sequence is not None
        assert first_sequence is not None
        reacquisition = ConfirmedGateReacquisition(
            credited_advance=self.unbound_advance,
            gate_index=self.current_gate,
            reacquired_track_id=reacquired_track_id,
            camera_token_at_binding=snapshot.latest_camera_token,
            reacquired_first_token=stable_tokens[0],
            stable_frame_tokens=stable_tokens,
            history_length_at_binding=len(stable_tokens),
            history_sha256="c" * 64,
            cross_gap_identity_claimed=False,
        )
        self.reacquisition = reacquisition
        self.current_track_id = reacquired_track_id
        self.after_promotion_samples = None
        self.binding_snapshot = snapshot
        self.visual_gate_graph.latest_snapshot = snapshot
        return reacquisition


def _navigation_commands(host, gate_index):
    return [
        (index, command, kwargs)
        for index, (command, kwargs, command_gate) in enumerate(host.commands)
        if command_gate == gate_index
        and kwargs.get("wire_visual_token") is not None
    ]


def _assert_successful_reacquisition(
    host: _CreditedUnboundCoordinatorHost,
    result,
    *,
    from_gate_index: int,
) -> None:
    to_gate_index = from_gate_index + 1
    assert result["success"] is True
    assert result["race_finished"] is True
    assert type(host.unbound_advance) is CreditedUnboundGateAdvance
    assert type(host.reacquisition) is ConfirmedGateReacquisition
    assert host.reacquisition.cross_gap_identity_claimed is False

    transition = result["authoritative_transitions"][0]
    assert (
        transition["from_gate_index"],
        transition["to_gate_index"],
    ) == (from_gate_index, to_gate_index)
    assert transition["post_transition_zero_command_count"] >= 1
    assert transition["post_transition_navigation_command_count"] >= 1

    navigation = _navigation_commands(host, to_gate_index)
    assert navigation
    first_navigation_index, _command, first_kwargs = navigation[0]
    phase_receipts = _assert_course_zero_receipts(host)
    credited_unbound_indices = phase_receipts[
        "credited-unbound-zero"
    ]
    assert credited_unbound_indices
    assert all(
        command_index < first_navigation_index
        for command_index in credited_unbound_indices
    )
    first_token = first_kwargs["wire_visual_token"]
    credit = host.unbound_advance
    assert credit is not None
    assert (
        first_token.publication_sequence
        > credit.camera_token_at_credit.publication_sequence
    )
    first_observation_ns = (
        first_token.publication_sequence - 1
    ) * 20_000_000
    assert (
        first_observation_ns
        > credit.race_status.received_monotonic_ns
    )
    assert any(
        index < first_navigation_index
        and command_gate == to_gate_index
        and command.roll_rate
        == command.pitch_rate
        == command.yaw_rate
        == command.thrust
        == 0.0
        for index, (command, _kwargs, command_gate) in enumerate(
            host.commands
        )
    )


def test_credited_unbound_gate0_waits_zero_then_commands_fresh_gate1():
    host = _CreditedUnboundCoordinatorHost(
        initial_gate=0,
        finish_gate=1,
    )

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_runtime(host)[0],
        )
    )

    _assert_successful_reacquisition(
        host,
        result,
        from_gate_index=0,
    )


def test_credited_unbound_one_edge_successor_commands_observable_axis():
    host = _CreditedUnboundCoordinatorHost(
        initial_gate=0,
        finish_gate=1,
        binding_clipping=FrameEdge.TOP,
    )

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_runtime(host)[0],
        )
    )

    _assert_successful_reacquisition(
        host,
        result,
        from_gate_index=0,
    )
    gate1 = next(
        segment for segment in result["segments"]
        if segment["gate_index"] == 1
    )
    assert gate1["recovery_one_edge_command_count"] >= 1
    first_command = _navigation_commands(host, 1)[0][1]
    assert first_command.yaw_rate != 0.0


def test_credited_unbound_same_reviewed_identity_rebinds_without_seam():
    host = _CreditedUnboundCoordinatorHost(
        initial_gate=0,
        finish_gate=1,
        same_reviewed_identity=True,
    )

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_runtime(host)[0],
        )
    )

    _assert_successful_reacquisition(
        host,
        result,
        from_gate_index=0,
    )
    transition = result["authoritative_transitions"][0]
    assert transition["reacquisition_identity_basis"] == (
        "retained-reviewed-local-track"
    )
    assert transition["recovery_admission"]["promotion_identity_basis"] == (
        "rolling-graph-retained-reviewed-fresh-rebind-v1"
    )


def test_credited_unbound_accepts_separate_delayed_race_finished_packet():
    host = _CreditedUnboundCoordinatorHost(
        initial_gate=0,
        finish_gate=2,
        reacquire_after_samples=None,
        finish_during_unbound_after_samples=2,
    )

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_runtime(host)[0],
        )
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert type(host.unbound_advance) is CreditedUnboundGateAdvance
    assert host.reacquisition is None
    assert _navigation_commands(host, 1) == []
    assert host.visual_gate_graph.finish_calls


def test_credited_unbound_reacquisition_timeout_sends_no_navigation():
    host = _CreditedUnboundCoordinatorHost(
        initial_gate=0,
        finish_gate=1,
        reacquire_after_samples=None,
    )
    limits = replace(
        VisualCourseStageLimits(),
        post_credit_fresh_frame_timeout_s=0.05,
    )

    with pytest.raises(SafetyAbort, match="reacqui|fresh post-credit"):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_runtime(host, limits=limits)[0],
            )
        )

    assert type(host.unbound_advance) is CreditedUnboundGateAdvance
    assert host.reacquisition is None
    assert _navigation_commands(host, 1) == []
    assert any(
        gate_index == 1
        and command.roll_rate
        == command.pitch_rate
        == command.yaw_rate
        == command.thrust
        == 0.0
        for command, _kwargs, gate_index in host.commands
    )


def test_credited_unbound_ambiguous_successors_never_command():
    host = _CreditedUnboundCoordinatorHost(
        initial_gate=0,
        finish_gate=1,
        ambiguous_reacquisition=True,
    )

    with pytest.raises(SafetyAbort, match="[Aa]mbiguous"):
        asyncio.run(
            run_visual_course_stage(
                host,
                _context(),
                runtime=_runtime(host)[0],
            )
        )

    assert type(host.unbound_advance) is CreditedUnboundGateAdvance
    assert host.reacquisition is None
    assert _navigation_commands(host, 1) == []


def test_credited_unbound_lifecycle_is_not_gate0_specific():
    host = _CreditedUnboundCoordinatorHost(
        initial_gate=6,
        finish_gate=7,
    )

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_runtime(host)[0],
        )
    )

    _assert_successful_reacquisition(
        host,
        result,
        from_gate_index=6,
    )
