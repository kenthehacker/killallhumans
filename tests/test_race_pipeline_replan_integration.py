"""Integration tests for RacePipeline's dynamic-replan wiring (P1-14).

Mirrors sim_pybullet/tests/test_runner_replan_integration.py: bypass the
heavy RacePipeline.__init__ (which spins up DroneEKF, Phase1GateDetector,
RacingLineOptimizer, TrajectoryOptimizer, etc.) using __new__ + a hand-
stuffed object graph. Validates that a sequencer crash surface event
triggers a trajectory rebuild via the replanner, the same way the sim
runner rebuilds its racing line.
"""

from __future__ import annotations

import math
import threading
from typing import List, Tuple

import pytest

from gate_sequencing.sequencer import (
    GateSequencer, GateSpec, SequencerConfig,
)

from planning.dynamic_replanner import DynamicReplanner, ReplanConfig
from race_pipeline import PipelineConfig, RacePipeline, _ref_override_position


def test_pipeline_tracker_preclamps_lateral_acceleration_to_tilt_envelope():
    pipe = RacePipeline(interface=None, config=PipelineConfig())
    tracker = pipe.tracker.config

    assert tracker.max_lateral_accel == pytest.approx(
        tracker.gravity * math.tan(tracker.max_tilt_rad)
    )


def test_recovery_reference_discards_incompatible_derivative_hints():
    from planning.trajectory_optimizer import TrajectoryPoint

    original = TrajectoryPoint(
        time=1.25,
        position=(10.0, 20.0, 30.0),
        velocity=(4.0, 5.0, 6.0),
        acceleration=(7.0, 8.0, 9.0),
        jerk=(1.0, 2.0, 3.0),
        yaw=0.75,
        yaw_rate=0.25,
        ff_acceleration=(3.0, 2.0, 1.0),
    )

    recovery = _ref_override_position(original, (1.0, 2.0, 3.0))

    assert recovery.position == (1.0, 2.0, 3.0)
    assert recovery.velocity == (0.0, 0.0, 0.0)
    assert recovery.acceleration == (0.0, 0.0, 0.0)
    assert recovery.jerk == (0.0, 0.0, 0.0)
    assert recovery.time == original.time
    assert recovery.yaw == original.yaw
    assert recovery.yaw_rate == original.yaw_rate
    assert recovery.ff_acceleration == (0.0, 0.0, 0.0)
    assert original.position == (10.0, 20.0, 30.0)


def _course() -> List[GateSpec]:
    return [
        GateSpec(gate_id="G1", position=(5, 0, 1.5), sequence_index=0),
        GateSpec(gate_id="G2", position=(10, 0, 1.5), sequence_index=1),
        GateSpec(gate_id="G3", position=(15, 0, 1.5), sequence_index=2),
    ]


class _FakeTrajectoryPoint:
    """Minimal stand-in for trajectory_optimizer.TrajectoryPoint."""

    def __init__(
        self,
        position: Tuple[float, float, float],
        time: float = 0.0,
    ):
        self.position = position
        self.time = time


class _FakeTrajectory:
    """Minimal stand-in for RaceTrajectory used by the pipeline."""

    def __init__(self, total_time: float = 5.0):
        self.total_time = total_time
        self.points = [_FakeTrajectoryPoint((0, 0, 1.5), 0.0)]

    def find_closest_forward(self, position, t, search_window_s=2.0):
        return _FakeTrajectoryPoint(position, time=t)

    def sample(self, t):
        return _FakeTrajectoryPoint((0, 0, 1.5), time=t)


def _make_pipeline_stub(gates: List[GateSpec]) -> RacePipeline:
    """Construct a RacePipeline with just enough surface to drive
    _maybe_replan (and the build/rebuild trajectory path)."""
    pipe = RacePipeline.__new__(RacePipeline)

    # Synchronous replan keeps these unit tests deterministic; the async path
    # has its own dedicated test below.
    pipe.config = PipelineConfig(async_replan=False)
    pipe.sequencer = GateSequencer(
        gates, config=SequencerConfig(pass_through_margin=1.0),
    )
    pipe.sequencer.start()
    pipe.replanner = DynamicReplanner(ReplanConfig(cooldown_seconds=0.0))
    pipe.trajectory = _FakeTrajectory()
    pipe._gate_specs = list(gates)
    pipe._replan_count = 0
    pipe._last_replan_reasons = []
    pipe._last_lateral_err = 0.0
    pipe._ref_progress_time = 0.0
    pipe._pending_trigger = None
    pipe._rebuild_lock = threading.Lock()
    pipe._rebuild_in_flight = False
    pipe._rebuild_result = None

    # Stub _build_trajectory_from so we don't need scipy / RacingLineOptimizer.
    pipe._build_trajectory_calls: List[Tuple] = []

    def _fake_build(start_pos, start_vel, remaining):
        pipe._build_trajectory_calls.append((start_pos, start_vel, list(remaining)))
        pipe.trajectory = _FakeTrajectory()

    pipe._build_trajectory_from = _fake_build  # type: ignore[assignment]
    return pipe


# ── Replanner triggers a trajectory rebuild on a crash ────────────────────


class TestRacePipelineReplanIntegration:
    def test_crash_triggers_trajectory_rebuild(self):
        gates = _course()
        pipe = _make_pipeline_stub(gates)

        # Drone hits G1 — physics layer marks the crash.
        pipe.sequencer.mark_collision("G1", position=(5.0, 0.7, 1.5))
        pipe._maybe_replan(
            sim_time=1.0, position=(4.5, 0.5, 1.5), velocity=(0, 0, 0),
        )

        assert pipe._replan_count == 1
        assert "gate_collision" in pipe._last_replan_reasons
        assert len(pipe._build_trajectory_calls) == 1
        # Rebuild starts from the drone's position with the remaining
        # gates (G1 onwards, since G1 wasn't passed).
        start_pos, start_vel, remaining = pipe._build_trajectory_calls[0]
        assert start_pos == (4.5, 0.5, 1.5)
        assert start_vel == (0, 0, 0)
        assert [g.gate_id for g in remaining] == ["G1", "G2", "G3"]

    def test_clean_run_does_not_replan(self):
        gates = _course()
        pipe = _make_pipeline_stub(gates)

        pipe._last_lateral_err = 0.05
        pipe._maybe_replan(
            sim_time=1.0, position=(1.0, 0.0, 1.5), velocity=(0, 0, 0),
        )
        assert pipe._replan_count == 0
        assert len(pipe._build_trajectory_calls) == 0

    def test_replan_resets_reference_anchor(self):
        gates = _course()
        pipe = _make_pipeline_stub(gates)
        pipe._ref_progress_time = 1.7  # was tracking partway through

        pipe.sequencer.mark_collision("G1", position=(5.0, 0.7, 1.5))
        pipe._maybe_replan(
            sim_time=1.0, position=(4.5, 0.5, 1.5), velocity=(0, 0, 0),
        )

        # New trajectory → reset the closest-forward anchor.
        assert pipe._ref_progress_time == 0.0

    def test_replan_after_one_gate_passed_uses_remaining(self):
        gates = _course()
        pipe = _make_pipeline_stub(gates)

        # Pass G1 cleanly.
        pipe.sequencer.update((4.0, 0.0, 1.5))
        pipe.sequencer.update((6.0, 0.0, 1.5))
        assert pipe.sequencer.gates_passed == 1

        # Crash G2.
        pipe.sequencer.mark_collision("G2", position=(10.0, 0.7, 1.5))
        pipe._maybe_replan(
            sim_time=2.0, position=(9.5, 0.5, 1.5), velocity=(0, 0, 0),
        )

        assert pipe._replan_count == 1
        _, _, remaining = pipe._build_trajectory_calls[-1]
        # G1 is passed; rebuild should target G2 onwards.
        assert [g.gate_id for g in remaining] == ["G2", "G3"]

    def test_cooldown_blocks_consecutive_replans(self):
        gates = _course()
        pipe = _make_pipeline_stub(gates)
        # Restore the production cooldown for this case.
        pipe.replanner = DynamicReplanner(ReplanConfig(cooldown_seconds=0.5))

        pipe.sequencer.mark_collision("G1", position=(5.0, 0.7, 1.5))
        pipe._maybe_replan(
            sim_time=1.0, position=(4.5, 0.5, 1.5), velocity=(0, 0, 0),
        )
        assert pipe._replan_count == 1

        # 0.4 s later — still in cooldown. New crash on G2 should NOT
        # trigger a second replan.
        pipe.sequencer.mark_collision("G2", position=(10.0, 0.7, 1.5))
        pipe._maybe_replan(
            sim_time=1.4, position=(4.5, 0.5, 1.5), velocity=(0, 0, 0),
        )
        assert pipe._replan_count == 1

    def test_cooldown_deferred_crash_is_served_after_cooldown(self):
        """Regression for audit Blocker 7: a crash whose rising edge lands
        inside the cooldown window must NOT be lost — it should fire a
        replan the moment the cooldown expires, rather than being consumed
        by evaluate()'s fire-once edge detection."""
        gates = _course()
        pipe = _make_pipeline_stub(gates)
        pipe.replanner = DynamicReplanner(ReplanConfig(cooldown_seconds=0.5))

        # First replan at t=1.0 starts the cooldown.
        pipe.sequencer.mark_collision("G1", position=(5.0, 0.7, 1.5))
        pipe._maybe_replan(
            sim_time=1.0, position=(4.5, 0.5, 1.5), velocity=(0, 0, 0),
        )
        assert pipe._replan_count == 1

        # New crash on G2 at t=1.2 — inside cooldown. evaluate() reports the
        # edge exactly once and would normally consume it; the pipeline must
        # remember it instead.
        pipe.sequencer.mark_collision("G2", position=(10.0, 0.7, 1.5))
        pipe._maybe_replan(
            sim_time=1.2, position=(9.5, 0.5, 1.5), velocity=(0, 0, 0),
        )
        assert pipe._replan_count == 1  # still cooling down
        assert pipe._pending_trigger is not None
        assert pipe._pending_trigger.gate_collision

        # t=1.6 — cooldown expired, no NEW edge from evaluate() (G2 already
        # seen), but the deferred trigger must now be served.
        pipe._maybe_replan(
            sim_time=1.6, position=(9.5, 0.5, 1.5), velocity=(0, 0, 0),
        )
        assert pipe._replan_count == 2, "deferred crash was lost"
        assert pipe._pending_trigger is None

    def test_async_replan_is_nonblocking_and_swaps_when_ready(self):
        """Audit Blocker 9 fix: with async_replan the control loop must NOT
        block on the rebuild — it keeps the current trajectory until the
        background worker finishes, then swaps atomically on a later tick."""
        gates = _course()
        pipe = _make_pipeline_stub(gates)
        pipe.config = PipelineConfig(async_replan=True)

        started = threading.Event()
        release = threading.Event()
        new_traj = _FakeTrajectory()

        def _slow_compute(pos, vel, remaining):
            started.set()
            release.wait(2.0)  # hold the "optimisation" open
            return new_traj

        pipe._compute_trajectory = _slow_compute  # type: ignore[assignment]
        old_traj = pipe.trajectory

        pipe.sequencer.mark_collision("G1", position=(5.0, 0.7, 1.5))
        pipe._maybe_replan(sim_time=1.0, position=(4.5, 0.5, 1.5), velocity=(0, 0, 0))

        # Returned promptly without blocking: rebuild in flight, trajectory
        # unchanged, no replan counted yet.
        assert pipe._rebuild_in_flight is True
        assert pipe.trajectory is old_traj
        assert pipe._replan_count == 0
        assert started.wait(2.0), "worker never started"

        # A tick while the rebuild is still in flight must not start a second
        # rebuild nor swap anything.
        pipe._maybe_replan(sim_time=1.1, position=(4.6, 0.5, 1.5), velocity=(0, 0, 0))
        assert pipe.trajectory is old_traj
        assert pipe._replan_count == 0

        # Let the worker finish and publish, then the next tick lands it.
        release.set()
        pipe._rebuild_thread.join(2.0)
        pipe._maybe_replan(sim_time=1.5, position=(4.6, 0.5, 1.5), velocity=(0, 0, 0))
        assert pipe.trajectory is new_traj
        assert pipe._replan_count == 1
        assert pipe._rebuild_in_flight is False
        assert pipe._ref_progress_time == 0.0

    def test_remaining_empty_skips_rebuild(self):
        # Edge case: all gates already passed (race effectively complete);
        # _maybe_replan must not call _build_trajectory_from with [].
        gates = _course()
        pipe = _make_pipeline_stub(gates)

        # Mark all gates as passed.
        for g in gates:
            pipe.sequencer._passed.append(g.gate_id)
        pipe.sequencer._current_idx = len(gates)
        pipe.sequencer.mark_collision = lambda *a, **kw: None  # silence

        # Force a trigger via NaN-safe fake replanner state by feeding a
        # sustained lateral error.
        pipe._last_lateral_err = 100.0
        cfg = ReplanConfig(
            cooldown_seconds=0.0,
            lateral_error_threshold_m=1.0,
            sustained_frames=1,
        )
        pipe.replanner = DynamicReplanner(cfg)

        pipe._maybe_replan(
            sim_time=1.0, position=(20.0, 0.0, 1.5), velocity=(0, 0, 0),
        )
        # No remaining gates → no rebuild attempted.
        assert len(pipe._build_trajectory_calls) == 0
