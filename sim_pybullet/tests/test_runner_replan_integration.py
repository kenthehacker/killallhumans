"""Integration tests for RaceRunner gate-contact + dynamic-replan wiring.

These tests bypass the heavy RaceRunner.__init__ (which spins up
PyBullet, gym-pybullet-drones, file logging, etc.) using __new__ + a
hand-stuffed object graph. The point is to validate the wire-up of
env.gate_contact() -> sequencer.mark_collision() -> DynamicReplanner ->
RacingLine rebuild, without needing a live physics client."""

from __future__ import annotations

import sys
import types
from typing import List, Optional, Tuple
from unittest.mock import MagicMock


def _install_fake_pybullet():
    """Stub out pybullet so RacingLine + runner imports don't fault."""
    fake = types.ModuleType("pybullet")
    fake.getContactPoints = lambda *a, **kw: []
    fake.addUserDebugLine = lambda *a, **kw: 0
    fake.removeUserDebugItem = lambda *a, **kw: None
    sys.modules["pybullet"] = fake
    return fake


_install_fake_pybullet()


from flight_control.types import DroneState, TargetState  # noqa: E402
from simulation.model_types import Gate, GateConfig, Pose3D  # noqa: E402
from sim_pybullet.runner import RaceRunner, RacingLine  # noqa: E402
from sim_pybullet.sequencer import GateSequencer  # noqa: E402
from planning.dynamic_replanner import DynamicReplanner, ReplanConfig  # noqa: E402


def _gate(gate_id: str, x: float, idx: int) -> Gate:
    return Gate(
        gate_id=gate_id,
        config=GateConfig(
            interior_width_m=1.2,
            interior_height_m=1.2,
            border_width_m=0.15,
        ),
        pose=Pose3D(x=x, y=0.0, z=1.5, yaw=0.0),
        sequence_index=idx,
    )


def _make_runner_stub(gates: List[Gate], contact_returns: List[Optional[str]]):
    """Construct a RaceRunner with just the surface needed for _maybe_replan
    + the gate-contact polling block in the main loop."""
    runner = RaceRunner.__new__(RaceRunner)
    # Use the same default margin (1.5) the runner uses in production —
    # tests rely on mark_collision (physics-driven) rather than geometric
    # detection for crash setup, so the margin doesn't matter for these
    # cases. Keeping it at the production default ensures the test
    # exercises the same code path as the live runner.
    runner.sequencer = GateSequencer(gates)
    runner._racing_line = RacingLine(
        [
            __import__("numpy").array([0, 0, 1.5]),
            *[__import__("numpy").array([g.pose.x, g.pose.y, g.pose.z])
              for g in gates],
        ]
    )
    runner._replanner = DynamicReplanner(ReplanConfig(cooldown_seconds=0.0))
    runner._replan_count = 0
    runner._last_replan_reasons = []
    runner._last_contact_gate_id = None
    runner._last_lateral_err = 0.0
    runner._prev_target_z = None
    runner._draw_racing_lines = lambda: None  # no-op for tests

    # Replace env with a mock that returns scripted gate_contact() values.
    env = MagicMock()
    env.gate_contact = MagicMock(side_effect=list(contact_returns))
    env.dim_gate = MagicMock()
    env.highlight_gate = MagicMock()
    runner.env = env
    return runner


def _simulate_tick(runner, drone_pos):
    """Replicate the contact-polling block from RaceRunner.run() for one tick.

    Mirrors the wiring under test verbatim — see runner.py's main loop."""
    contact_gate_id = runner.env.gate_contact()
    known_ids = {g.gate_id for g in runner.sequencer.all_gates}
    if (contact_gate_id is not None
            and contact_gate_id in known_ids
            and contact_gate_id != runner._last_contact_gate_id):
        runner.sequencer.mark_collision(
            contact_gate_id, position=drone_pos
        )
    runner._last_contact_gate_id = contact_gate_id
    runner.sequencer.update(drone_pos)


# ── Gate-contact → mark_collision ─────────────────────────────────────────


class TestContactToMarkCollision:
    def test_gate_contact_marks_crash_once(self):
        """A multi-tick sustained PyBullet contact should mark exactly one crash."""
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1)]
        # 5 ticks where env.gate_contact() reports 'G1', then clears.
        runner = _make_runner_stub(
            gates,
            contact_returns=["G1", "G1", "G1", None, None],
        )
        for _ in range(5):
            _simulate_tick(runner, (4.5, 0.5, 1.5))
        assert runner.sequencer.crashed_gate_ids == ["G1"]
        assert runner.crashed_into_gate == "G1"

    def test_gate_contact_marks_each_distinct_hit(self):
        """Two distinct contacts (broken by a clean tick) should mark twice."""
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1)]
        runner = _make_runner_stub(
            gates,
            contact_returns=["G1", None, "G1", None],
        )
        for _ in range(4):
            _simulate_tick(runner, (4.5, 0.5, 1.5))
        assert runner.sequencer.crashed_gate_ids == ["G1", "G1"]

    def test_unknown_gate_id_silently_ignored(self):
        gates = [_gate("G1", 5, 0)]
        runner = _make_runner_stub(
            gates, contact_returns=["GHOST", None],
        )
        for _ in range(2):
            _simulate_tick(runner, (4.5, 0, 1.5))
        assert runner.sequencer.crashed_gate_ids == []

    def test_no_contact_no_crash(self):
        gates = [_gate("G1", 5, 0)]
        runner = _make_runner_stub(gates, contact_returns=[None, None, None])
        for _ in range(3):
            _simulate_tick(runner, (1, 0, 1.5))
        assert runner.sequencer.crashed_gate_ids == []
        assert runner.crashed_into_gate is None


# ── Replanner integration ─────────────────────────────────────────────────


class TestReplanIntegration:
    def test_crash_triggers_replan(self):
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1), _gate("G3", 15, 2)]
        runner = _make_runner_stub(gates, contact_returns=["G1", None])
        # tick 1: crash registered
        drone_state = DroneState(
            position=(4.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0,
        )
        _simulate_tick(runner, drone_state.position)
        assert runner.sequencer.crashed_gate_ids == ["G1"]

        old_line = runner._racing_line
        runner._maybe_replan(sim_time=1.0, drone_state=drone_state)
        assert runner._replan_count == 1
        # Line was rebuilt
        assert runner._racing_line is not old_line
        assert "gate_collision" in runner._last_replan_reasons

    def test_clean_run_does_not_replan(self):
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1)]
        runner = _make_runner_stub(gates, contact_returns=[None, None, None])
        for _ in range(3):
            _simulate_tick(runner, (1, 0, 1.5))
        runner._last_lateral_err = 0.05
        drone_state = DroneState(position=(1, 0, 1.5), velocity=(0, 0, 0), yaw=0.0)
        runner._maybe_replan(sim_time=1.0, drone_state=drone_state)
        assert runner._replan_count == 0

    def test_replanned_line_starts_at_drone_position(self):
        """Replan must rebuild from current drone state, not from initial start."""
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1)]
        runner = _make_runner_stub(gates, contact_returns=["G1"])
        drone_state = DroneState(
            position=(4.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0,
        )
        _simulate_tick(runner, drone_state.position)
        runner._maybe_replan(sim_time=1.0, drone_state=drone_state)
        # The first waypoint of the new line should be the drone position
        first_pt = runner._racing_line.points[0]
        assert abs(first_pt[0] - 4.5) < 0.5
        assert abs(first_pt[1] - 0.5) < 0.5
        assert abs(first_pt[2] - 1.5) < 0.5

    def test_replan_count_persists_across_multiple_triggers(self):
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1), _gate("G3", 15, 2)]
        runner = _make_runner_stub(
            gates, contact_returns=["G1", None, "G1", None],
        )
        drone_state = DroneState(
            position=(4.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0,
        )

        _simulate_tick(runner, drone_state.position)
        runner._maybe_replan(sim_time=1.0, drone_state=drone_state)
        assert runner._replan_count == 1

        _simulate_tick(runner, drone_state.position)
        runner._maybe_replan(sim_time=2.0, drone_state=drone_state)

        _simulate_tick(runner, drone_state.position)
        runner._maybe_replan(sim_time=3.0, drone_state=drone_state)
        assert runner._replan_count == 2  # second contact (broken by None tick)
