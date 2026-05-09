"""Integration tests for RacePipeline's dynamic-replan wiring (P1-14).

Mirrors sim_pybullet/tests/test_runner_replan_integration.py: bypass the
heavy RacePipeline.__init__ (which spins up DroneEKF, Phase1GateDetector,
RacingLineOptimizer, TrajectoryOptimizer, etc.) using __new__ + a hand-
stuffed object graph. Validates that a sequencer crash surface event
triggers a trajectory rebuild via the replanner, the same way the sim
runner rebuilds its racing line.
"""

from __future__ import annotations

from typing import List, Tuple

from gate_sequencing.sequencer import (
    GateSequencer, GateSpec, SequencerConfig,
)
from planning.dynamic_replanner import DynamicReplanner, ReplanConfig
from race_pipeline import RacePipeline


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
