"""Tests for planning.dynamic_replanner.DynamicReplanner."""

from __future__ import annotations

import pytest

from gate_sequencing.sequencer import (
    GateSequencer,
    GateSpec,
    SequencerConfig,
)
from planning.dynamic_replanner import (
    DynamicReplanner,
    ReplanConfig,
    ReplanTrigger,
)


def _course():
    return [
        GateSpec(gate_id="G1", position=(5, 0, 0), sequence_index=0,
                 interior_width=1.2, interior_height=1.2, border_width=0.15),
        GateSpec(gate_id="G2", position=(10, 0, 0), sequence_index=1,
                 interior_width=1.2, interior_height=1.2, border_width=0.15),
        GateSpec(gate_id="G3", position=(15, 0, 0), sequence_index=2,
                 interior_width=1.2, interior_height=1.2, border_width=0.15),
    ]


# ── Trigger detection ────────────────────────────────────────────────────


class TestTriggers:
    def test_no_trigger_on_clean_run(self):
        seq = GateSequencer(_course())
        seq.start()
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))  # clean pass

        rep = DynamicReplanner()
        t = rep.evaluate(sim_time=1.0, sequencer=seq, lateral_error=0.1)
        assert not t.triggered
        assert t.reasons == []

    def test_gate_collision_triggers(self):
        seq = GateSequencer(_course())
        seq.start()
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))  # crash

        rep = DynamicReplanner()
        t = rep.evaluate(sim_time=1.0, sequencer=seq, lateral_error=0.0)
        assert t.gate_collision
        assert t.crashed_gate_id == "G1"
        assert t.triggered

    def test_missed_gate_triggers(self):
        seq = GateSequencer(_course())
        seq.start()
        # Cross plane far outside outer frame → miss
        seq.update((4.5, 5.0, 0))
        seq.update((5.5, 5.0, 0))

        rep = DynamicReplanner()
        t = rep.evaluate(sim_time=1.0, sequencer=seq, lateral_error=0.0)
        assert t.missed_gate
        assert t.triggered

    def test_off_track_state_triggers(self):
        seq = GateSequencer(_course(),
                            config=SequencerConfig(off_track_distance=5.0))
        seq.start()
        seq.update((0, 0, 0))
        seq.update((0, 100, 0))  # way off, sequencer enters RECOVERY

        rep = DynamicReplanner()
        t = rep.evaluate(sim_time=1.0, sequencer=seq, lateral_error=0.0)
        assert t.off_track
        assert t.triggered

    def test_sustained_lateral_error_triggers_after_threshold(self):
        seq = GateSequencer(_course())
        seq.start()
        seq.update((0, 0, 0))

        rep = DynamicReplanner(ReplanConfig(
            lateral_error_threshold_m=1.5,
            sustained_frames=10,
            cooldown_seconds=0.0,
        ))

        # 9 frames of high error — not yet triggered
        for i in range(9):
            t = rep.evaluate(sim_time=i * 0.01, sequencer=seq,
                             lateral_error=2.0)
        assert not t.triggered

        # 10th frame — sustained
        t = rep.evaluate(sim_time=0.1, sequencer=seq, lateral_error=2.0)
        assert t.sustained_lateral_error
        assert t.triggered

    def test_lateral_error_counter_resets_on_low_error(self):
        rep = DynamicReplanner(ReplanConfig(
            lateral_error_threshold_m=1.5,
            sustained_frames=5,
            cooldown_seconds=0.0,
        ))
        seq = GateSequencer(_course())
        seq.start()
        seq.update((0, 0, 0))

        for i in range(4):
            rep.evaluate(sim_time=i * 0.01, sequencer=seq, lateral_error=2.0)
        # One frame of low error — counter resets
        rep.evaluate(sim_time=0.04, sequencer=seq, lateral_error=0.1)
        # Now four more high-error frames — should NOT trigger yet
        for i in range(4):
            t = rep.evaluate(sim_time=0.05 + i * 0.01,
                             sequencer=seq, lateral_error=2.0)
            assert not t.sustained_lateral_error


# ── Cooldown ─────────────────────────────────────────────────────────────


class TestCooldown:
    def test_within_cooldown_should_not_replan(self):
        rep = DynamicReplanner(ReplanConfig(cooldown_seconds=0.5))
        # Force a trigger record
        trig = ReplanTrigger(gate_collision=True, crashed_gate_id="G1")
        rep.mark_replanned(sim_time=10.0, trigger=trig)
        # 0.4s later — still in cooldown
        assert not rep.should_replan(trig, sim_time=10.4)

    def test_after_cooldown_should_replan(self):
        rep = DynamicReplanner(ReplanConfig(cooldown_seconds=0.5))
        rep.mark_replanned(
            sim_time=10.0,
            trigger=ReplanTrigger(gate_collision=True, crashed_gate_id="G1"),
        )
        trig = ReplanTrigger(gate_collision=True, crashed_gate_id="G1")
        assert rep.should_replan(trig, sim_time=10.6)

    def test_no_trigger_no_replan_even_after_cooldown(self):
        rep = DynamicReplanner(ReplanConfig(cooldown_seconds=0.5))
        # Cooldown is open from the start (last_replan_time = -inf)
        idle = ReplanTrigger()
        assert not idle.triggered
        assert not rep.should_replan(idle, sim_time=100.0)


# ── Waypoint construction ────────────────────────────────────────────────


class TestWaypoints:
    def test_waypoints_start_at_drone_and_include_remaining_gates(self):
        seq = GateSequencer(_course())
        seq.start()
        rep = DynamicReplanner()

        wps = rep.waypoints_for_replan(
            drone_position=(2.5, 1.0, 0.5), sequencer=seq,
        )
        # 1 (drone) + 3 (gates) = 4 waypoints
        assert len(wps) == 4
        assert wps[0] == (2.5, 1.0, 0.5)
        assert wps[1] == (5.0, 0.0, 0.0)
        assert wps[2] == (10.0, 0.0, 0.0)
        assert wps[3] == (15.0, 0.0, 0.0)

    def test_waypoints_skip_already_passed_gates(self):
        seq = GateSequencer(_course())
        seq.start()
        # Pass G1
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.gates_passed == 1

        rep = DynamicReplanner()
        wps = rep.waypoints_for_replan(
            drone_position=(6.5, 0.0, 0.0), sequencer=seq,
        )
        # drone + remaining (G2, G3)
        assert len(wps) == 3
        assert wps[0] == (6.5, 0.0, 0.0)
        assert wps[1] == (10.0, 0.0, 0.0)
        assert wps[2] == (15.0, 0.0, 0.0)


# ── Counters & reset ─────────────────────────────────────────────────────


class TestCounters:
    def test_replan_count_increments(self):
        rep = DynamicReplanner()
        assert rep.replan_count == 0
        trig = ReplanTrigger(gate_collision=True, crashed_gate_id="G1")
        rep.mark_replanned(sim_time=1.0, trigger=trig)
        rep.mark_replanned(sim_time=2.0, trigger=trig)
        assert rep.replan_count == 2

    def test_reset_clears_state(self):
        rep = DynamicReplanner()
        rep.mark_replanned(
            sim_time=5.0,
            trigger=ReplanTrigger(gate_collision=True, crashed_gate_id="G1"),
        )
        rep.reset()
        assert rep.replan_count == 0
        assert rep.last_trigger is None

    def test_replan_clears_lateral_error_counter(self):
        """After a replan, the racing line is fresh — old lateral error
        counter is no longer meaningful."""
        rep = DynamicReplanner(ReplanConfig(
            lateral_error_threshold_m=1.0,
            sustained_frames=5,
            cooldown_seconds=0.0,
        ))
        seq = GateSequencer(_course())
        seq.start()
        seq.update((0, 0, 0))

        # Build up lateral error
        for i in range(4):
            rep.evaluate(sim_time=i * 0.01, sequencer=seq,
                         lateral_error=2.0)
        # Replan
        rep.mark_replanned(
            sim_time=0.04,
            trigger=ReplanTrigger(sustained_lateral_error=True),
        )
        # One more high-error frame — must NOT fire again since counter reset
        t = rep.evaluate(sim_time=0.05, sequencer=seq, lateral_error=2.0)
        assert not t.sustained_lateral_error
