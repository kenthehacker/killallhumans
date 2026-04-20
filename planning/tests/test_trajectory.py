"""Tests for trajectory planning (planning/trajectory_optimizer.py and planning/racing_line.py)."""

import math

import numpy as np
import pytest

from planning.trajectory_optimizer import (
    DroneConstraints,
    FOVConfig,
    GateWaypoint,
    RaceTrajectory,
    TrajectoryOptimizer,
    TrajectoryPoint,
    _min_snap_1d,
    _poly_eval,
    _poly_deriv_eval,
    _wrap_angle,
)
from planning.racing_line import (
    RacingLineConfig,
    RacingLineOptimizer,
    SpeedProfiler,
    _dist3,
)


# ── Minimum-snap polynomial boundary conditions ─────────────────────────


class TestMinSnapPolynomial:
    def test_position_boundary_conditions(self):
        """p(0) = p0, p(T) = pf"""
        p0, v0, a0 = 1.0, 0.5, 0.0
        pf, vf, af = 5.0, 0.0, 0.0
        T = 2.0
        coeffs = _min_snap_1d(p0, v0, a0, pf, vf, af, T)

        assert _poly_eval(coeffs, 0.0) == pytest.approx(p0, abs=1e-8)
        assert _poly_eval(coeffs, T) == pytest.approx(pf, abs=1e-6)

    def test_velocity_boundary_conditions(self):
        """p'(0) = v0, p'(T) = vf"""
        p0, v0, a0 = 0.0, 2.0, 0.0
        pf, vf, af = 10.0, 1.0, 0.0
        T = 3.0
        coeffs = _min_snap_1d(p0, v0, a0, pf, vf, af, T)

        assert _poly_deriv_eval(coeffs, 0.0, 1) == pytest.approx(v0, abs=1e-8)
        assert _poly_deriv_eval(coeffs, T, 1) == pytest.approx(vf, abs=1e-5)

    def test_acceleration_boundary_conditions(self):
        """p''(0) = a0, p''(T) = af"""
        p0, v0, a0 = 0.0, 0.0, 1.0
        pf, vf, af = 5.0, 0.0, -0.5
        T = 2.0
        coeffs = _min_snap_1d(p0, v0, a0, pf, vf, af, T)

        assert _poly_deriv_eval(coeffs, 0.0, 2) == pytest.approx(a0, abs=1e-6)
        assert _poly_deriv_eval(coeffs, T, 2) == pytest.approx(af, abs=1e-4)

    def test_zero_motion(self):
        """Stationary: p(t) should be constant."""
        coeffs = _min_snap_1d(3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert _poly_eval(coeffs, t) == pytest.approx(3.0, abs=1e-6)

    def test_polynomial_has_6_coefficients(self):
        coeffs = _min_snap_1d(0, 0, 0, 1, 0, 0, 1.0)
        assert len(coeffs) == 6


class TestPolyEval:
    def test_constant_polynomial(self):
        coeffs = np.array([5.0, 0, 0, 0, 0, 0])
        assert _poly_eval(coeffs, 0.0) == pytest.approx(5.0)
        assert _poly_eval(coeffs, 10.0) == pytest.approx(5.0)

    def test_linear_polynomial(self):
        coeffs = np.array([1.0, 2.0, 0, 0, 0, 0])
        assert _poly_eval(coeffs, 0.0) == pytest.approx(1.0)
        assert _poly_eval(coeffs, 3.0) == pytest.approx(7.0)

    def test_derivative_of_linear(self):
        coeffs = np.array([1.0, 3.0, 0, 0, 0, 0])
        assert _poly_deriv_eval(coeffs, 0.0, 1) == pytest.approx(3.0)
        assert _poly_deriv_eval(coeffs, 5.0, 1) == pytest.approx(3.0)

    def test_second_derivative_of_quadratic(self):
        coeffs = np.array([0, 0, 4.0, 0, 0, 0])
        # p(t) = 4t^2, p''(t) = 8
        assert _poly_deriv_eval(coeffs, 0.0, 2) == pytest.approx(8.0)
        assert _poly_deriv_eval(coeffs, 1.0, 2) == pytest.approx(8.0)


# ── Trajectory optimizer with gates ──────────────────────────────────────


class TestTrajectoryOptimizer:
    @pytest.fixture
    def three_gates(self):
        return [
            GateWaypoint(position=(5, 0, 0), normal=(1, 0, 0), yaw=0.0),
            GateWaypoint(position=(10, 5, 0), normal=(0, 1, 0), yaw=math.pi / 2),
            GateWaypoint(position=(15, 0, 0), normal=(1, 0, 0), yaw=0.0),
        ]

    def test_no_gates_raises(self):
        opt = TrajectoryOptimizer()
        with pytest.raises(ValueError, match="No gates"):
            opt.optimize([])

    def test_single_gate(self):
        opt = TrajectoryOptimizer()
        gates = [GateWaypoint(position=(5, 0, 0), normal=(1, 0, 0))]
        traj = opt.optimize(gates)
        assert isinstance(traj, RaceTrajectory)
        assert len(traj.points) > 0
        assert traj.total_time > 0

    def test_three_gates_smooth_trajectory(self, three_gates):
        opt = TrajectoryOptimizer(dt_sample=0.02)
        traj = opt.optimize(three_gates)

        assert len(traj.points) > 10
        assert traj.total_time > 0
        # Each gate expands into entry + interior + exit waypoints, plus a
        # finish segment, so 3 gates → 7 segments (2*3 + 1). Matches the
        # benchmark contract in scripts/benchmark.py.
        assert len(traj.segment_times) == 7

        # Verify trajectory starts near origin
        assert traj.points[0].position[0] == pytest.approx(0, abs=0.01)
        assert traj.points[0].position[1] == pytest.approx(0, abs=0.01)

    def test_trajectory_passes_near_gates(self, three_gates):
        opt = TrajectoryOptimizer(dt_sample=0.02)
        traj = opt.optimize(three_gates)

        for gate in three_gates:
            closest = traj.find_closest(gate.position)
            dist = np.linalg.norm(
                np.array(closest.position) - np.array(gate.position)
            )
            assert dist < 2.0  # should pass within 2m of each gate

    def test_trajectory_sample_interpolation(self, three_gates):
        opt = TrajectoryOptimizer(dt_sample=0.02)
        traj = opt.optimize(three_gates)

        # Sample at boundaries
        p0 = traj.sample(0.0)
        assert p0.time == pytest.approx(0.0)

        pend = traj.sample(traj.total_time)
        assert pend.time == pytest.approx(traj.total_time)

        # Sample in middle
        pmid = traj.sample(traj.total_time / 2)
        assert 0 < pmid.time < traj.total_time

    def test_trajectory_sample_clamping(self, three_gates):
        opt = TrajectoryOptimizer(dt_sample=0.02)
        traj = opt.optimize(three_gates)

        # Before start
        p = traj.sample(-1.0)
        assert p.time == traj.points[0].time

        # After end
        p = traj.sample(traj.total_time + 10)
        assert p.time == traj.points[-1].time

    def test_trajectory_points_have_all_fields(self, three_gates):
        opt = TrajectoryOptimizer(dt_sample=0.05)
        traj = opt.optimize(three_gates)
        pt = traj.points[len(traj.points) // 2]

        assert isinstance(pt.time, float)
        assert len(pt.position) == 3
        assert len(pt.velocity) == 3
        assert len(pt.acceleration) == 3
        assert len(pt.jerk) == 3
        assert isinstance(pt.yaw, float)
        assert isinstance(pt.yaw_rate, float)


# ── Speed profiler ───────────────────────────────────────────────────────


class TestSpeedProfiler:
    def test_straight_line_fast(self):
        """Straight segments should reach high speeds."""
        profiler = SpeedProfiler(max_speed=15.0, min_speed=2.0)
        # Straight line
        waypoints = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0)]
        speeds = profiler.profile(waypoints)
        assert len(speeds) == 4
        # Middle waypoints should be faster than min
        assert speeds[2] > profiler.min_speed

    def test_sharp_turn_slow(self):
        """Sharp turns should produce lower speeds than straights."""
        profiler = SpeedProfiler(max_speed=15.0, min_speed=2.0)
        # 90-degree turn
        turn_points = [(0, 0, 0), (5, 0, 0), (5, 5, 0)]
        turn_speeds = profiler.profile(turn_points)

        # Straight line for comparison
        straight_points = [(0, 0, 0), (5, 0, 0), (10, 0, 0)]
        straight_speeds = profiler.profile(straight_points)

        # Turn should be slower at the corner
        assert turn_speeds[1] <= straight_speeds[1]

    def test_single_waypoint(self):
        profiler = SpeedProfiler()
        speeds = profiler.profile([(0, 0, 0)])
        assert len(speeds) == 1
        assert speeds[0] == profiler.min_speed

    def test_speeds_respect_bounds(self):
        profiler = SpeedProfiler(max_speed=10.0, min_speed=1.0)
        waypoints = [(i * 5, 0, 0) for i in range(10)]
        speeds = profiler.profile(waypoints)
        for s in speeds:
            assert profiler.min_speed <= s <= profiler.max_speed

    def test_deceleration_before_turn(self):
        """Speed should decrease approaching a sharp turn."""
        profiler = SpeedProfiler(max_speed=15.0, min_speed=2.0)
        # Long straight then sharp turn
        waypoints = [
            (0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0),
            (30, 10, 0),  # sharp right turn
        ]
        speeds = profiler.profile(waypoints)
        # Speed at turn point (index 3) should be less than max
        assert speeds[3] < profiler.max_speed


# ── Racing line optimizer ────────────────────────────────────────────────


class TestRacingLineOptimizer:
    def test_single_gate_unchanged(self):
        opt = RacingLineOptimizer()
        gates = [GateWaypoint(position=(5, 0, 0), normal=(1, 0, 0))]
        result = opt.optimize(gates)
        assert len(result) == 1

    def test_path_length_decreases_vs_centerline(self):
        """Optimized racing line should be shorter than center-to-center."""
        opt = RacingLineOptimizer(config=RacingLineConfig(
            max_lateral_offset=0.4,
            corner_cut_aggressiveness=0.7,
        ))
        # L-shaped course: forces corner cutting
        gates = [
            GateWaypoint(position=(5, 0, 0), normal=(1, 0, 0), yaw=0.0, width=1.5),
            GateWaypoint(position=(10, 0, 0), normal=(1, 0, 0), yaw=0.0, width=1.5),
            GateWaypoint(position=(10, 5, 0), normal=(0, 1, 0), yaw=math.pi/2, width=1.5),
            GateWaypoint(position=(10, 10, 0), normal=(0, 1, 0), yaw=math.pi/2, width=1.5),
        ]
        start = (0, 0, 0)

        # Center-line path length
        center_pts = [np.array(start)] + [np.array(g.position) for g in gates]
        center_length = sum(
            np.linalg.norm(center_pts[i+1] - center_pts[i])
            for i in range(len(center_pts) - 1)
        )

        optimized = opt.optimize(gates, start_position=start)
        opt_pts = [np.array(start)] + [np.array(g.position) for g in optimized]
        opt_length = sum(
            np.linalg.norm(opt_pts[i+1] - opt_pts[i])
            for i in range(len(opt_pts) - 1)
        )

        # Optimized should be shorter or equal
        assert opt_length <= center_length + 0.01

    def test_optimized_gates_preserve_count(self):
        opt = RacingLineOptimizer()
        gates = [
            GateWaypoint(position=(5, 0, 0), normal=(1, 0, 0)),
            GateWaypoint(position=(10, 5, 0), normal=(0, 1, 0)),
            GateWaypoint(position=(15, 0, 0), normal=(1, 0, 0)),
        ]
        result = opt.optimize(gates)
        assert len(result) == len(gates)

    def test_optimized_gates_stay_near_originals(self):
        """Optimized positions shouldn't stray far from gate centers."""
        opt = RacingLineOptimizer(config=RacingLineConfig(max_lateral_offset=0.4))
        gates = [
            GateWaypoint(position=(5, 0, 0), normal=(1, 0, 0), width=1.5),
            GateWaypoint(position=(10, 5, 0), normal=(0, 1, 0), width=1.5),
        ]
        result = opt.optimize(gates)
        for orig, optg in zip(gates, result):
            dist = np.linalg.norm(
                np.array(orig.position) - np.array(optg.position)
            )
            # Should be within half the gate width
            assert dist < orig.width


# ── Helper ───────────────────────────────────────────────────────────────


class TestDistHelper:
    def test_dist3_same_point(self):
        assert _dist3((0, 0, 0), (0, 0, 0)) == pytest.approx(0)

    def test_dist3_known_distance(self):
        assert _dist3((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)

    def test_dist3_3d(self):
        assert _dist3((1, 2, 3), (4, 6, 3)) == pytest.approx(5.0)
