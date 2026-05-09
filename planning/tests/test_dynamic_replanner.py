"""Tests for planning.dynamic_replanner.DynamicReplanner."""

from __future__ import annotations

import math

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


# ── P1-3 set-of-IDs for crash/miss detection ─────────────────────────────


class TestSetOfIdsDetection:
    """Crash/miss detection by set-difference. A sequencer.reset() that
    isn't paired with replanner.reset() must NOT swallow the next real
    crash — the set-based detector self-resyncs as the sequencer's
    authoritative set shrinks."""

    def test_unpaired_sequencer_reset_does_not_swallow_next_crash(self):
        """The realistic scenario: a tick-based caller (runner, RL rollout)
        calls evaluate() every tick, so the empty crash list immediately
        post-reset is observed. With the count-delta detector,
        `n_crashes (0) > _last_seen (1)` is False so the baseline never
        rolls back; the next real crash silently no-ops. Set-of-IDs auto-
        resyncs by shrinking _seen as the sequencer's list shrinks."""
        seq = GateSequencer(_course())
        seq.start()
        rep = DynamicReplanner()

        # First crash → replanner sees it.
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))
        t = rep.evaluate(sim_time=1.0, sequencer=seq, lateral_error=0.0)
        assert t.gate_collision

        # Sequencer reset *without* paired replanner reset; one tick of
        # evaluate to observe the empty baseline.
        seq.reset()
        seq.start()
        t = rep.evaluate(sim_time=2.0, sequencer=seq, lateral_error=0.0)
        assert not t.gate_collision  # baseline silently rolled back

        # Reproduce the crash — a stale count-delta baseline would refuse
        # to fire here.
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))
        t = rep.evaluate(sim_time=3.0, sequencer=seq, lateral_error=0.0)
        assert t.gate_collision, (
            "set-of-IDs detector failed to re-detect after sequencer reset"
        )

    def test_repeat_crash_on_same_gate_does_not_refire(self):
        """Same-ID crash within one race counts once. P1-11 handles
        escalation (skip-after-N-failures) separately."""
        seq = GateSequencer(_course())
        seq.start()
        rep = DynamicReplanner()

        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))
        t1 = rep.evaluate(sim_time=1.0, sequencer=seq, lateral_error=0.0)
        assert t1.gate_collision

        # Force a second mark_collision on the same gate (e.g. via
        # mark_collision-style external call) — the sequencer just appends.
        seq.mark_collision("G1", position=(5.0, 0.7, 0))
        t2 = rep.evaluate(sim_time=2.0, sequencer=seq, lateral_error=0.0)
        # Same gate → same set → no new trigger.
        assert not t2.gate_collision


# ── P1-8 NaN/non-finite guards ───────────────────────────────────────────


class TestNonFiniteGuards:
    def test_nan_sim_time_does_not_force_replan(self):
        """`sim_time - last_replan_time < cooldown` returns False per
        IEEE-754 when sim_time is NaN, falling through to True. Must reject."""
        rep = DynamicReplanner(ReplanConfig(cooldown_seconds=0.5))
        rep.mark_replanned(
            sim_time=10.0,
            trigger=ReplanTrigger(gate_collision=True, crashed_gate_id="G1"),
        )
        trig = ReplanTrigger(gate_collision=True, crashed_gate_id="G1")
        assert not rep.should_replan(trig, sim_time=float("nan"))
        assert not rep.should_replan(trig, sim_time=float("inf"))
        assert not rep.should_replan(trig, sim_time=float("-inf"))

    def test_nan_sim_time_does_not_poison_last_replan_time(self):
        """A successful mark_replanned with NaN would write NaN, then any
        subsequent comparison would underflow → cooldown disabled forever."""
        rep = DynamicReplanner(ReplanConfig(cooldown_seconds=0.5))
        # Spec rule 7: refuse non-finite writes — mark_replanned must no-op.
        trig = ReplanTrigger(gate_collision=True, crashed_gate_id="G1")
        rep.mark_replanned(sim_time=float("nan"), trigger=trig)
        # _last_replan_time should still be -inf so the next finite call
        # is gated only by the cooldown window from a real replan time.
        assert math.isfinite(rep._last_replan_time) or rep._last_replan_time == -math.inf
        # And replan_count must NOT have ticked.
        assert rep.replan_count == 0

    def test_nan_lateral_error_skips_counter_update(self):
        """A NaN lateral_error must not increment _consecutive_high_lateral
        nor reset it — silent corruption either way."""
        rep = DynamicReplanner(ReplanConfig(
            lateral_error_threshold_m=1.0,
            sustained_frames=5,
            cooldown_seconds=0.0,
        ))
        seq = GateSequencer(_course())
        seq.start()
        seq.update((0, 0, 0))

        # Build up real high-lateral counter to 4.
        for i in range(4):
            rep.evaluate(sim_time=i * 0.01, sequencer=seq, lateral_error=2.0)
        # NaN tick — must NOT reset counter (silent disable) and must NOT
        # increment to crossing the threshold.
        rep.evaluate(sim_time=0.04, sequencer=seq, lateral_error=float("nan"))
        # One more real high-lateral tick — should bring us to 5 and fire.
        t = rep.evaluate(sim_time=0.05, sequencer=seq, lateral_error=2.0)
        assert t.sustained_lateral_error, (
            "NaN tick either reset the counter or skipped the increment"
        )


# ── P1-1 edge-trigger ────────────────────────────────────────────────────


class TestEdgeTriggers:
    """off_track and sustained_lateral_error are level signals — without
    edge-triggering, the trigger record reads True every tick the
    perturbation persists, which downstream callers can mistake for new
    events. After P1-1 they fire once on the rising edge."""

    def _seq_in_recovery(self):
        seq = GateSequencer(
            _course(), config=SequencerConfig(off_track_distance=5.0),
        )
        seq.start()
        seq.update((0, 0, 0))
        seq.update((0, 100, 0))  # RECOVERY
        return seq

    def test_off_track_fires_once_on_rising_edge(self):
        seq = self._seq_in_recovery()
        rep = DynamicReplanner()

        t1 = rep.evaluate(sim_time=0.0, sequencer=seq, lateral_error=0.0)
        assert t1.off_track  # rising edge

        # Sequencer is still in RECOVERY for the next 5 ticks — level
        # signal is still True but edge has already fired.
        for i in range(1, 6):
            t = rep.evaluate(
                sim_time=i * 0.01, sequencer=seq, lateral_error=0.0,
            )
            assert not t.off_track, f"off_track refired on tick {i}"

    def test_off_track_rearms_after_clear(self):
        """After leaving RECOVERY (passing a gate) and re-entering, off_track
        fires again. The sequencer only exits RECOVERY via a gate pass."""
        rep = DynamicReplanner()

        seq = self._seq_in_recovery()
        t = rep.evaluate(sim_time=0.0, sequencer=seq, lateral_error=0.0)
        assert t.off_track

        # Pass G1: drone crosses gate plane through the opening. Sequencer
        # transitions back to RACING.
        seq.update((4.5, 0, 0))
        seq.update((5.5, 0, 0))
        t = rep.evaluate(sim_time=0.01, sequencer=seq, lateral_error=0.0)
        assert not t.off_track

        # Knock far off the line again — sequencer re-enters RECOVERY.
        seq.update((6, 100, 0))
        t = rep.evaluate(sim_time=0.02, sequencer=seq, lateral_error=0.0)
        assert t.off_track, "off_track failed to re-arm after clearing"

    def test_sustained_lateral_fires_once_on_threshold(self):
        rep = DynamicReplanner(ReplanConfig(
            lateral_error_threshold_m=1.0,
            sustained_frames=5,
            cooldown_seconds=0.0,
        ))
        seq = GateSequencer(_course())
        seq.start()
        seq.update((0, 0, 0))

        # Frames 1-4: counter ramps but threshold not crossed
        for i in range(4):
            t = rep.evaluate(
                sim_time=i * 0.01, sequencer=seq, lateral_error=2.0,
            )
            assert not t.sustained_lateral_error

        # Frame 5: rising edge — fire once
        t = rep.evaluate(sim_time=0.04, sequencer=seq, lateral_error=2.0)
        assert t.sustained_lateral_error

        # Frames 6-10: still sustained, but already armed — don't refire
        for i in range(5, 10):
            t = rep.evaluate(
                sim_time=i * 0.01, sequencer=seq, lateral_error=2.0,
            )
            assert not t.sustained_lateral_error, (
                f"sustained_lateral_error refired on tick {i}"
            )

    def test_replan_does_not_storm_when_off_track_persists(self):
        """The motivating scenario: drone enters RECOVERY, replan fires,
        controller hasn't recovered yet, sequencer still reports RECOVERY
        on the next tick. Without edge-trigger the field stays True and
        fires again the moment cooldown expires."""
        seq = self._seq_in_recovery()
        rep = DynamicReplanner(ReplanConfig(cooldown_seconds=0.0))

        t = rep.evaluate(sim_time=0.0, sequencer=seq, lateral_error=0.0)
        assert t.off_track
        rep.mark_replanned(sim_time=0.0, trigger=t)

        # 10 ticks later — sequencer still in RECOVERY, but the latch
        # remembers it has fired.
        for i in range(1, 11):
            t = rep.evaluate(
                sim_time=i * 0.01, sequencer=seq, lateral_error=0.0,
            )
            assert not t.off_track

    def test_reset_clears_edge_latches(self):
        """After reset, the next True level should fire again as a new edge."""
        seq = self._seq_in_recovery()
        rep = DynamicReplanner()

        t = rep.evaluate(sim_time=0.0, sequencer=seq, lateral_error=0.0)
        assert t.off_track

        rep.reset()
        # Same sequencer state — level is still True. After reset, the
        # latch is cleared, so this counts as a new rising edge.
        t = rep.evaluate(sim_time=0.0, sequencer=seq, lateral_error=0.0)
        assert t.off_track, "edge latch survived reset"


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
