"""Tests for gate sequencer (gate_sequencing/sequencer.py)."""

import math

import numpy as np
import pytest

from gate_sequencing.sequencer import (
    GateSequencer,
    GateSpec,
    RaceState,
    SequencerConfig,
)


def _make_gate(gate_id: str, position, yaw=0.0, idx=0, **kwargs) -> GateSpec:
    return GateSpec(
        gate_id=gate_id,
        position=position,
        yaw=yaw,
        sequence_index=idx,
        **kwargs,
    )


def _make_course() -> list:
    """Simple 3-gate straight course along X axis, gates facing +X."""
    return [
        _make_gate("G1", (5, 0, 0), yaw=0.0, idx=0),
        _make_gate("G2", (10, 0, 0), yaw=0.0, idx=1),
        _make_gate("G3", (15, 0, 0), yaw=0.0, idx=2),
    ]


# ── Gate pass-through detection ──────────────────────────────────────────


class TestPassThroughDetection:
    def test_fly_through_gate_plane(self):
        """Flying straight through a gate should detect pass-through."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        # Approach from behind
        seq.update((3, 0, 0))
        # Fly through
        result = seq.update((7, 0, 0))
        assert result is not None
        assert result.gate_id == "G1"

    def test_no_detection_when_missing_gate(self):
        """Flying past gate but outside the opening should not trigger."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2)]
        seq = GateSequencer(gates)
        seq.start()

        # Approach far off to the side
        seq.update((3, 10, 0))
        result = seq.update((7, 10, 0))
        assert result is None

    def test_fly_parallel_no_crossing(self):
        """Moving parallel to gate plane should not trigger."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        seq.update((5, -2, 0))
        result = seq.update((5, 2, 0))
        # Moving along the gate plane, not crossing it
        assert result is None

    def test_crossing_in_opening(self):
        """Crossing through the center of the opening triggers detection."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        cfg = SequencerConfig(pass_through_margin=1.0)
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        seq.update((4.9, 0, 0))
        result = seq.update((5.1, 0, 0))
        assert result is not None


# ── Sequence progression ─────────────────────────────────────────────────


class TestSequenceProgression:
    def test_gates_must_be_passed_in_order(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        # Try to pass gate 2 first (skip gate 1) — should not register
        seq.update((9, 0, 0))
        result = seq.update((11, 0, 0))
        assert result is None  # G2 is not the current target yet
        assert seq.gates_passed == 0

    def test_sequential_gate_passing(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        # Pass gate 1
        seq.update((4, 0, 0))
        r1 = seq.update((6, 0, 0))
        assert r1 is not None
        assert r1.gate_id == "G1"
        assert seq.gates_passed == 1

        # Pass gate 2
        seq.update((9, 0, 0))
        r2 = seq.update((11, 0, 0))
        assert r2 is not None
        assert r2.gate_id == "G2"
        assert seq.gates_passed == 2

        # Pass gate 3
        seq.update((14, 0, 0))
        r3 = seq.update((16, 0, 0))
        assert r3 is not None
        assert r3.gate_id == "G3"
        assert seq.gates_passed == 3

    def test_progress_fraction(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        assert seq.progress == pytest.approx(0.0)

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.progress == pytest.approx(1 / 3)

    def test_current_gate_advances(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        assert seq.current_gate.gate_id == "G1"

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.current_gate.gate_id == "G2"

    def test_next_gate_property(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        assert seq.next_gate.gate_id == "G2"

    def test_gates_sorted_by_sequence_index(self):
        # Provide gates out of order
        gates = [
            _make_gate("G3", (15, 0, 0), idx=2),
            _make_gate("G1", (5, 0, 0), idx=0),
            _make_gate("G2", (10, 0, 0), idx=1),
        ]
        seq = GateSequencer(gates)
        assert seq.current_gate.gate_id == "G1"


# ── Recovery state transitions ───────────────────────────────────────────


class TestRecoveryTransitions:
    def test_initial_state_waiting(self):
        seq = GateSequencer(_make_course())
        assert seq.state == RaceState.WAITING

    def test_start_transitions_to_racing(self):
        seq = GateSequencer(_make_course())
        seq.start()
        assert seq.state == RaceState.RACING

    def test_off_track_triggers_recovery(self):
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Move very far from the expected path
        seq.update((0, 0, 0))
        seq.update((0, 100, 0))  # way off track
        assert seq.state == RaceState.RECOVERY

    def test_recovery_provides_target(self):
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        seq.update((0, 0, 0))
        seq.update((0, 100, 0))

        target = seq.get_recovery_target()
        assert target is not None
        # Should point toward current gate
        assert target == gates[0].position

    def test_passing_gate_exits_recovery(self):
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Go off track
        seq.update((0, 0, 0))
        seq.update((0, 100, 0))
        assert seq.state == RaceState.RECOVERY

        # Come back and pass through gate 1
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.state == RaceState.RACING

    def test_detection_dropout_triggers_slow_down(self):
        cfg = SequencerConfig(detection_dropout_frames=5)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        for i in range(10):
            seq.update((0, 0, 0), gate_detected=False)

        assert seq.should_slow_down()

    def test_detection_recovery_clears_slow_down(self):
        cfg = SequencerConfig(detection_dropout_frames=5)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Lose detection
        for _ in range(10):
            seq.update((0, 0, 0), gate_detected=False)
        assert seq.should_slow_down()

        # Regain detection
        seq.update((0, 0, 0), gate_detected=True)
        # In RACING state with detection → should not slow down
        if seq.state == RaceState.RACING:
            assert not seq.should_slow_down()


# ── Completion detection ─────────────────────────────────────────────────


class TestCompletion:
    def test_all_gates_passed_is_complete(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        # Pass all 3 gates
        for gx in [5, 10, 15]:
            seq.update((gx - 1, 0, 0))
            seq.update((gx + 1, 0, 0))

        assert seq.is_complete
        assert seq.state == RaceState.COMPLETED
        assert seq.gates_passed == 3

    def test_update_after_completion_returns_none(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.is_complete

        result = seq.update((100, 0, 0))
        assert result is None

    def test_waiting_state_returns_none(self):
        seq = GateSequencer(_make_course())
        # Not started
        result = seq.update((4, 0, 0))
        assert result is None

    def test_current_gate_none_after_completion(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.current_gate is None


# ── Reset ────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.gates_passed == 1

        seq.reset()
        assert seq.gates_passed == 0
        assert seq.state == RaceState.WAITING
        assert not seq.is_complete
        assert seq.current_gate.gate_id == "G1"
