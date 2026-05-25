"""
Tests for the plan validator (iter-004 Phase 1).

Verifies:
  - a clean straight-line trajectory through ordered gates PASSES
  - a trajectory that crosses gate-2 before gate-1 is DQ'd
  - a trajectory that grazes a gate's strut is CRASHED
  - an empty or zero-time trajectory fails cleanly with a diagnostic
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pytest

from gate_sequencing.sequencer import GateSpec
from planning.plan_validator import validate_trajectory, ValidationResult


# ---------------------------------------------------------------------------
# Duck-typed trajectory stub — avoids RaceTrajectory's full surface
# ---------------------------------------------------------------------------

@dataclass
class _StubRef:
    position: Tuple[float, float, float]


@dataclass
class _StubTrajectory:
    """Sampled at evenly-spaced times along a piecewise-linear path."""
    waypoints: List[Tuple[float, float, float]]   # (t, x, y, z) or just positions
    times: List[float]                             # monotone in [0, total_time]

    @property
    def total_time(self) -> float:
        return float(self.times[-1])

    def sample(self, t: float) -> _StubRef:
        if t <= self.times[0]:
            return _StubRef(self.waypoints[0])
        if t >= self.times[-1]:
            return _StubRef(self.waypoints[-1])
        # piecewise linear
        for i in range(1, len(self.times)):
            if t <= self.times[i]:
                t0, t1 = self.times[i - 1], self.times[i]
                p0, p1 = self.waypoints[i - 1], self.waypoints[i]
                alpha = (t - t0) / (t1 - t0)
                return _StubRef((
                    p0[0] + alpha * (p1[0] - p0[0]),
                    p0[1] + alpha * (p1[1] - p0[1]),
                    p0[2] + alpha * (p1[2] - p0[2]),
                ))
        return _StubRef(self.waypoints[-1])


def _make_line_gates(n: int = 3, spacing: float = 5.0) -> List[GateSpec]:
    return [
        GateSpec(
            gate_id=f"g{i+1}",
            position=(spacing * (i + 1), 0.0, 2.0),
            yaw=0.0,
            sequence_index=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Clean run
# ---------------------------------------------------------------------------

def test_clean_in_order_trajectory_passes():
    """A trajectory that flies cleanly through three gates in order."""
    gates = _make_line_gates(3, spacing=5.0)
    traj = _StubTrajectory(
        waypoints=[
            (0.0, 0.0, 2.0),
            (5.0, 0.0, 2.0),
            (10.0, 0.0, 2.0),
            (15.0, 0.0, 2.0),
            (20.0, 0.0, 2.0),
        ],
        times=[0.0, 1.0, 2.0, 3.0, 4.0],
    )
    result = validate_trajectory(traj, gates, dt=0.02)
    assert result.ok, result.reason
    assert result.gates_passed == 3
    assert not result.disqualified
    assert not result.crashed


# ---------------------------------------------------------------------------
# Out-of-order DQ detection (the overfitting failure mode)
# ---------------------------------------------------------------------------

def test_trajectory_crossing_future_gate_first_is_dq():
    """If the trajectory's first segment crosses gate-2's plane before
    gate-1 is credited, validator must flag the DQ."""
    gates = _make_line_gates(3, spacing=5.0)
    # Trajectory jumps directly to past-gate-2: from (0,0,-2) it shoots
    # to (11, 0, -2) in one second, crossing both g1 (x=5) and g2 (x=10)
    # in the same segment without crediting g1 first.
    # Wait — actually if the segment crosses g1's opening AND g2's opening
    # in one tick, the multi-gate-per-tick fix from iter-002 credits both.
    # To trigger DQ, the trajectory needs to cross g2 BEFORE crossing g1.
    # We do this by sampling such that the first sample skips past g1
    # entirely and the second hits g2's opening from beyond.
    traj = _StubTrajectory(
        waypoints=[
            (0.0, 0.0, 2.0),
            (8.0, 5.0, 2.0),         # away from g1 (at y=5)
            (10.0, 0.0, 2.0),        # back onto g2's centerline
            (15.0, 0.0, 2.0),
        ],
        times=[0.0, 1.0, 2.0, 3.0],
    )
    # The piecewise-linear path from (8,5) to (10,0) crosses x=10 at
    # y≈0 — inside g2's opening — while g1 was never crossed (we went
    # AROUND g1 at y=5).
    result = validate_trajectory(traj, gates, dt=0.02)
    assert not result.ok
    assert result.disqualified
    assert result.dq_reason is not None
    assert "g2" in result.dq_reason
    assert result.first_failure_time_s is not None
    assert 0 < result.first_failure_time_s < 3.0


# ---------------------------------------------------------------------------
# Crash detection (strut hit)
# ---------------------------------------------------------------------------

def test_trajectory_grazing_strut_is_crash():
    """A trajectory that grazes the gate frame strut is a CRASH, not DQ."""
    # AIGP defaults: 1.5m opening, 0.6m border → opening half = 0.75, outer half = 1.35.
    # Trajectory crosses g1's plane at y=1.0 — in the strut annulus.
    gates = _make_line_gates(1)
    traj = _StubTrajectory(
        waypoints=[
            (0.0, 1.0, 2.0),
            (5.0, 1.0, 2.0),
            (10.0, 1.0, 2.0),
        ],
        times=[0.0, 1.0, 2.0],
    )
    result = validate_trajectory(traj, gates, dt=0.02)
    assert not result.ok
    assert result.crashed
    assert result.last_crash_gate == "g1"


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_empty_trajectory_fails_cleanly():
    gates = _make_line_gates(3)
    traj = _StubTrajectory(waypoints=[(0.0, 0.0, 0.0)], times=[0.0])
    result = validate_trajectory(traj, gates, dt=0.02)
    assert not result.ok
    assert "empty" in result.reason.lower() or "zero" in result.reason.lower()


def test_no_gates_fails_cleanly():
    traj = _StubTrajectory(
        waypoints=[(0.0, 0.0, 2.0), (5.0, 0.0, 2.0)],
        times=[0.0, 1.0],
    )
    result = validate_trajectory(traj, [], dt=0.02)
    assert not result.ok
    assert "gates" in result.reason.lower() or "zero" in result.reason.lower()


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_has_all_diagnostic_fields():
    gates = _make_line_gates(2)
    traj = _StubTrajectory(
        waypoints=[(0.0, 0.0, 2.0), (5.0, 0.0, 2.0), (10.0, 0.0, 2.0)],
        times=[0.0, 1.0, 2.0],
    )
    result = validate_trajectory(traj, gates, dt=0.02)
    assert isinstance(result, ValidationResult)
    for field in (
        "ok", "reason", "gates_passed", "total_gates", "crashed",
        "disqualified", "dq_reason", "last_crash_gate", "samples_evaluated",
    ):
        assert hasattr(result, field)


# ---------------------------------------------------------------------------
# Iter-006 F5 (Opus MAJOR): airspace bounds
# ---------------------------------------------------------------------------

def test_trajectory_below_ground_is_flagged_as_crash_ground():
    """Validator must match the bench's ground crash semantics
    (z < 0.05 = crash). The bench is z-up convention here."""
    gates = _make_line_gates(1)
    traj = _StubTrajectory(
        waypoints=[
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),     # below ground threshold (0.05)
            (5.0, 0.0, 2.0),    # well below ground
        ],
        times=[0.0, 1.0, 2.0],
    )
    result = validate_trajectory(
        traj, gates, dt=0.02,
        ground_z_threshold=0.05, ceiling_z_threshold=20.0,
    )
    assert not result.ok
    assert "ground" in result.reason.lower()
    assert result.first_failure_time_s is not None


def test_trajectory_above_ceiling_is_flagged_as_crash_ceiling():
    """Symmetrical check: z > 20.0 means the drone exited the airspace
    upward."""
    gates = _make_line_gates(1)
    traj = _StubTrajectory(
        waypoints=[
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 25.0),    # above ceiling (20.0)
            (5.0, 0.0, 30.0),
        ],
        times=[0.0, 1.0, 2.0],
    )
    result = validate_trajectory(
        traj, gates, dt=0.02,
        ground_z_threshold=0.05, ceiling_z_threshold=20.0,
    )
    assert not result.ok
    assert "ceiling" in result.reason.lower()


def test_trajectory_within_airspace_no_flag():
    """A trajectory that stays within bounds shouldn't trigger airspace check."""
    gates = _make_line_gates(2)
    # Stay at z=2 throughout (between 0.05 ground and 20.0 ceiling)
    traj = _StubTrajectory(
        waypoints=[
            (0.0, 0.0, 2.0), (5.0, 0.0, 2.0), (10.0, 0.0, 2.0),
        ],
        times=[0.0, 1.0, 2.0],
    )
    # Note: gates here are at z=-2 from _make_line_gates so the path
    # doesn't pass them. Result will be incomplete but NOT airspace.
    result = validate_trajectory(traj, gates, dt=0.02)
    if not result.ok:
        assert "ground" not in result.reason.lower()
        assert "ceiling" not in result.reason.lower()
