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
from sim_pybullet._gate_to_spec import to_spec as _gate_to_spec  # noqa: E402
from gate_sequencing.sequencer import (  # noqa: E402
    GateSequencer, SequencerConfig,
)
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
    # Mirrors RaceRunner.__init__: platform-agnostic GateSequencer fed
    # GateSpec adapters with the production sim_pybullet pass margin
    # (1.5) plus decoupled crash_margin (1.0) so geometric crash
    # detection is non-empty. Tests here exercise the same code path
    # as the live runner.
    runner.sequencer = GateSequencer(
        [_gate_to_spec(g) for g in gates],
        config=SequencerConfig(pass_through_margin=1.5, crash_margin=1.0),
    )
    runner.sequencer.start()
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
    runner._replan_gates_baseline = 0
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

    def test_gate_contact_marks_distinct_gates(self):
        """The runner's `_last_contact_gate_id` dedupe + the sequencer's
        P1-5 idempotency together: each distinct gate id contacted
        registers exactly one crash entry. Same-gate re-hits collapse
        (P1-5); a different gate is a new event."""
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1)]
        runner = _make_runner_stub(
            gates,
            contact_returns=["G1", None, "G2", None],
        )
        for _ in range(4):
            _simulate_tick(runner, (4.5, 0.5, 1.5))
        assert runner.sequencer.crashed_gate_ids == ["G1", "G2"]

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
        # Distinct gate-id contacts each fire their own trigger. Set-of-ID
        # detection (P1-3) suppresses repeat crashes on the same gate, so
        # this test uses a different gate for the second crash. Repeat-
        # same-gate escalation is owned by P1-11 (skip-after-N-failures).
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1), _gate("G3", 15, 2)]
        runner = _make_runner_stub(
            gates, contact_returns=["G1", None, "G2", None],
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
        assert runner._replan_count == 2  # G2 is a fresh trigger


# ── P0-1: replan baseline snapshot + local-index targeting ────────────────


class TestReplanTargeting:
    """The replanned line starts fresh; min_arc indexing must reset per replan."""

    def test_replan_baseline_snapshot_tracks_gates_passed(self):
        # Drone has passed G1 already; now crashes G2.
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1), _gate("G3", 15, 2)]
        runner = _make_runner_stub(gates, contact_returns=["G2"])
        runner.sequencer._passed.append("G1")
        runner.sequencer._current_idx = 1

        assert runner._replan_gates_baseline == 0  # initial

        drone_state = DroneState(
            position=(9.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0,
        )
        _simulate_tick(runner, drone_state.position)
        runner._maybe_replan(sim_time=1.0, drone_state=drone_state)

        assert runner._replan_gates_baseline == 1  # G1 was already passed

    def test_replan_targets_next_gate_not_final(self):
        """Spec test: replanned line waypoint 1 must be G1 (next un-passed)."""
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1), _gate("G3", 15, 2)]
        runner = _make_runner_stub(gates, contact_returns=["G1"])
        state = DroneState(position=(4.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0)
        _simulate_tick(runner, state.position)
        runner._maybe_replan(sim_time=1.0, drone_state=state)
        # Replanned line points[60] sits at the seg-1 endpoint = G1 (~x=5),
        # not jumped past to G3 (x=15).
        assert tuple(runner._racing_line.points[60])[0] < 7.0

    def test_reset_restores_initial_racing_line(self):
        """P0-2: after a replan rebuilds the line mid-flight, _reset() must
        restore the line from the original start_position so 'r' brings the
        drone back to a clean state, not chasing a stale post-replan stub."""
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1)]
        runner = _make_runner_stub(gates, contact_returns=["G1"])
        runner.env.race_config = types.SimpleNamespace(
            start_position=(0.0, 0.0, 1.5),
            gates=gates,
        )
        runner.env.reset = MagicMock()

        # Force a mid-race replan so the racing line is overwritten.
        state = DroneState(position=(4.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0)
        _simulate_tick(runner, state.position)
        runner._maybe_replan(sim_time=1.0, drone_state=state)
        # Sanity: replanned line starts at drone, not at origin.
        assert abs(runner._racing_line.points[0][0] - 4.5) < 0.5

        # Press 'r' equivalent.
        runner._reset()

        first_pt = runner._racing_line.points[0]
        assert abs(first_pt[0] - 0.0) < 0.01
        assert abs(first_pt[1] - 0.0) < 0.01
        assert abs(first_pt[2] - 1.5) < 0.01
        # Slew + lateral + contact + baseline state must also be cleared.
        assert runner._prev_target_z is None
        assert runner._last_lateral_err == 0.0
        assert runner._last_contact_gate_id is None
        assert runner._replan_gates_baseline == 0

    def test_replan_seeds_prev_target_z_to_drone_altitude(self):
        """P1-9: post-replan, the slew limiter must stay armed. Nulling
        _prev_target_z disables it for one tick — the worst time, since
        the next-gate altitude can be 2m+ off the drone's current z."""
        gates = [_gate("G1", 5, 0), _gate("G2", 10, 1)]
        runner = _make_runner_stub(gates, contact_returns=["G1"])
        # Simulate a meaningful drone altitude pre-replan.
        runner._prev_target_z = 1.0  # tracking from prior tick
        drone_state = DroneState(
            position=(4.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0,
        )
        _simulate_tick(runner, drone_state.position)
        runner._maybe_replan(sim_time=1.0, drone_state=drone_state)

        # Slew limiter must be ARMED (not None) and seeded to the drone's
        # current z, not stale from before the replan.
        assert runner._prev_target_z is not None, (
            "P1-9: _prev_target_z nulled — slew limiter disabled post-replan"
        )
        assert abs(runner._prev_target_z - 1.5) < 1e-6, (
            f"P1-9: _prev_target_z={runner._prev_target_z}, expected 1.5 "
            f"(drone z at replan time)"
        )

    def test_post_replan_min_arc_uses_local_index(self):
        """Pre-fix: gates_passed=2 → waypoint_arc(2) on rebuilt 4-wp line clamps
        near G5, controller jumps past G3 and G4 to the final gate. With the
        fix, local_idx = gates_passed - baseline = 0 → search from drone."""
        gates = [
            _gate("G1", 5, 0), _gate("G2", 10, 1), _gate("G3", 15, 2),
            _gate("G4", 20, 3), _gate("G5", 25, 4),
        ]
        runner = _make_runner_stub(gates, contact_returns=["G3"])
        runner.env.drone.config.ctrl_freq = 240
        # Pretend drone already passed G1 + G2.
        runner.sequencer._passed.extend(["G1", "G2"])
        runner.sequencer._current_idx = 2

        drone_state = DroneState(
            position=(14.5, 0.5, 1.5), velocity=(0, 0, 0), yaw=0.0,
        )
        _simulate_tick(runner, drone_state.position)
        runner._maybe_replan(sim_time=1.0, drone_state=drone_state)

        # New line: [drone(14.5), G3(15), G4(20), G5(25)] — 3 segments.
        target = runner._target_from_sim_metadata(drone_state)
        # Drone is at x=14.5. With the fix (local_idx=0, min_arc=0), the
        # spline search starts at drone and lookahead lands near G3 (x≈15).
        # Without the fix, min_arc = waypoint_arc(2) on a 3-seg line clamps
        # past G3, so the target jumps somewhere into seg 1/2 (x>17).
        assert target.position[0] < 16.0, (
            f"target x={target.position[0]:.2f} jumped past G3 — local_idx fix missing"
        )
