"""Crash + miss + iff-highlighted tests for sim_pybullet.sequencer.

The sim_pybullet sequencer is the live race-state machine the runner
drives. It carries the same crash/miss/last_event contract as the
platform-agnostic gate_sequencing.sequencer so the DynamicReplanner can
consume either one. These tests validate that surface."""

from __future__ import annotations

import pytest

from simulation.model_types import Gate, GateConfig, Pose3D
from sim_pybullet.sequencer import GateSequencer


def _gate(gate_id: str, x: float, idx: int,
          interior_w: float = 1.2, interior_h: float = 1.2,
          border_w: float = 0.15) -> Gate:
    return Gate(
        gate_id=gate_id,
        config=GateConfig(
            interior_width_m=interior_w,
            interior_height_m=interior_h,
            border_width_m=border_w,
        ),
        pose=Pose3D(x=x, y=0.0, z=0.0, yaw=0.0),
        sequence_index=idx,
    )


def _course():
    return [_gate("G1", 5, 0), _gate("G2", 10, 1), _gate("G3", 15, 2)]


# All tests in this file use pass_through_margin=1.0 so the crash zone
# (between the bare opening and the outer frame) is non-empty. The
# runner's production sequencer uses the default 1.5 margin for
# imprecise-flight tolerance — covered separately by the runner
# integration tests.
def _seq(gates):
    return GateSequencer(gates, pass_through_margin=1.0)


# ── Crash detection ──────────────────────────────────────────────────────


class TestCrashDetection:
    def test_strut_hit_classified_as_crash(self):
        seq = _seq([_gate("G1", 5, 0)])
        # Lateral 0.7m → outside opening (half=0.6) but inside outer frame (half=0.75)
        seq.update((4.5, 0.7, 0))
        passed = seq.update((5.5, 0.7, 0))
        assert passed is None
        assert seq.crashed_gate_ids == ["G1"]
        assert seq.last_event == "crash"

    def test_complete_miss_classified_as_miss(self):
        seq = _seq([_gate("G1", 5, 0)])
        # Lateral 5m → outside outer frame
        seq.update((4.5, 5.0, 0))
        seq.update((5.5, 5.0, 0))
        assert seq.missed_gate_ids == ["G1"]
        assert seq.crashed_gate_ids == []
        assert seq.last_event == "miss"

    def test_clean_pass_no_crash_no_miss(self):
        seq = _seq([_gate("G1", 5, 0)])
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.crashed_gate_ids == []
        assert seq.missed_gate_ids == []
        assert seq.last_event == "pass"
        assert seq.gates_passed == 1

    def test_mark_collision_records_authoritatively(self):
        seq = _seq(_course())
        seq.update((1, 0, 0))
        seq.update((2, 0, 0))
        seq.mark_collision("G1", position=(2.0, 0.0, 0.0))
        assert seq.crashed_gate_ids == ["G1"]
        assert seq.last_event == "crash"
        gid, pt = seq.last_crash
        assert gid == "G1"
        assert pt == (2.0, 0.0, 0.0)

    def test_mark_collision_unknown_gate_raises(self):
        seq = _seq(_course())
        with pytest.raises(ValueError):
            seq.mark_collision("G99")

    def test_reset_clears_crash_state(self):
        seq = _seq([_gate("G1", 5, 0)])
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))
        assert seq.crashed_gate_ids == ["G1"]
        seq.reset()
        assert seq.crashed_gate_ids == []
        assert seq.missed_gate_ids == []
        assert seq.last_event is None


# ── iff highlighted ──────────────────────────────────────────────────────


class TestPassIfAndOnlyIfHighlighted:
    """Pass credit must be issued if AND only if the gate the drone
    crossed was the highlighted (current target) gate at the time of
    crossing. This mirrors the gate_sequencing-level contract for the
    PyBullet sequencer."""

    def test_pass_credited_for_highlighted_gate(self):
        seq = _seq(_course())
        assert seq.current_gate.gate_id == "G1"  # G1 is highlighted
        seq.update((4, 0, 0))
        passed = seq.update((6, 0, 0))
        assert passed is not None and passed.gate_id == "G1"

    def test_no_credit_for_non_highlighted_gate(self):
        """Drone strays past G2 while G1 is highlighted — must not credit."""
        seq = _seq(_course())
        seq.update((9, 0, 0))
        passed = seq.update((11, 0, 0))   # crosses G2's plane in opening
        assert passed is None
        assert seq.gates_passed == 0
        assert "G2" not in seq.passed_gate_ids
        assert seq.current_gate.gate_id == "G1"

    def test_late_passed_gate_does_not_recredit(self):
        seq = _seq(_course())
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))   # G1 passed
        assert seq.gates_passed == 1
        # Loop back past G1 — it's no longer highlighted
        seq.update((4, 0, 0))
        # G1 is not the current target — no credit on the recross
        assert seq.gates_passed == 1
        assert seq.current_gate.gate_id == "G2"

    def test_in_order_walk_iff_highlighted(self):
        seq = _seq(_course())
        observed = []
        for x in range(0, 22):
            highlighted_before = (
                seq.current_gate.gate_id if seq.current_gate else None
            )
            r = seq.update((float(x), 0, 0))
            if r is not None:
                # Pass must match what was highlighted just before.
                assert r.gate_id == highlighted_before
                observed.append(r.gate_id)
        assert observed == ["G1", "G2", "G3"]


# ── Crash does not advance, miss does not advance ────────────────────────


class TestNonPassDoesNotAdvance:
    def test_crash_keeps_target_highlighted(self):
        seq = _seq(_course())
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))   # crash
        assert seq.current_gate.gate_id == "G1"   # still highlighted
        assert seq.gates_passed == 0

    def test_miss_keeps_target_highlighted(self):
        seq = _seq(_course())
        seq.update((4.5, 5.0, 0))
        seq.update((5.5, 5.0, 0))   # miss
        assert seq.current_gate.gate_id == "G1"
        assert seq.gates_passed == 0
