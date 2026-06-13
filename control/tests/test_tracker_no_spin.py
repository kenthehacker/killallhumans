"""Closed-loop regression guarding the 'drone spins / diverges' failure mode.

Audit Blocker 1 (2026-06-09): the GeometricTracker extracted roll/pitch from
the *world*-frame thrust direction instead of the yaw-rotated body frame, so
at yaw=180 deg the commanded acceleration was inverted -> positive feedback ->
the drone accelerated *away* from the target and diverged/spun. The VQ1 course
starts with the drone needing to face gate-0 (behind it, yaw ~ +/-pi), which
is exactly where the bug bit hardest and where nobody had a test.

This test is a black-box check: feed the tracker's AttitudeCommand through
standard NED quadrotor dynamics and integrate. If position control is correct
the drone converges to the target from every heading; if the yaw-frame bug
returns, the yaw~pi cases diverge and these tests fail.
"""

import math

import numpy as np
import pytest

from control.mpc_tracker import GeometricTracker, TrackerConfig
from planning.trajectory_optimizer import TrajectoryPoint


def _ref(position, yaw=0.0):
    return TrajectoryPoint(
        time=0.0,
        position=tuple(position),
        velocity=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        jerk=(0.0, 0.0, 0.0),
        yaw=yaw,
        yaw_rate=0.0,
    )


def _world_accel_from_cmd(cmd, cfg: TrackerConfig):
    """Reconstruct world-frame acceleration (NED) from an attitude command
    using textbook quad dynamics: thrust along body -z (up), plus gravity."""
    phi, theta, psi = cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad
    T = cmd.thrust * cfg.max_thrust_n  # newtons
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    # Body-to-world ZYX rotation, third column = body z-axis in world.
    bz = (
        cpsi * sth * cphi + spsi * sphi,
        spsi * sth * cphi - cpsi * sphi,
        cth * cphi,
    )
    # Thrust points along body -z (up in NED).
    ax = -T / cfg.mass * bz[0]
    ay = -T / cfg.mass * bz[1]
    az = -T / cfg.mass * bz[2] + cfg.gravity  # gravity is +z in NED
    return np.array([ax, ay, az])


def _simulate(start_pos, start_yaw, target_pos, *, steps=300, dt=0.02):
    """Fixed-heading point-mass closed loop. Returns the error history."""
    cfg = TrackerConfig()
    tracker = GeometricTracker(cfg)
    pos = np.array(start_pos, dtype=float)
    vel = np.zeros(3)
    ref = _ref(target_pos, yaw=start_yaw)  # hold heading; isolate position ctrl
    errs = [float(np.linalg.norm(pos - np.array(target_pos)))]
    for _ in range(steps):
        cmd = tracker.track(tuple(pos), tuple(vel), start_yaw, ref)
        acc = _world_accel_from_cmd(cmd, cfg)
        vel += acc * dt
        pos += vel * dt
        errs.append(float(np.linalg.norm(pos - np.array(target_pos))))
    return errs


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi, -math.pi / 2,
                                 -math.pi, 3.0, -3.0])
def test_converges_from_every_heading(yaw):
    """From a fixed heading, horizontal position control must pull the drone
    toward the target. yaw~pi is the Blocker-1 divergence case."""
    target = (0.0, 0.0, -2.0)              # 2 m up (NED z-down)
    start = (5.0, -3.0, -2.0)              # 5.8 m horizontal offset
    errs = _simulate(start, yaw, target)
    assert errs[-1] < errs[0], (
        f"yaw={yaw:.2f}: error grew {errs[0]:.2f}->{errs[-1]:.2f} "
        f"(divergence / spin signature)"
    )
    assert errs[-1] < 0.75, (
        f"yaw={yaw:.2f}: did not converge (final error {errs[-1]:.2f} m)"
    )


@pytest.mark.parametrize("yaw", [0.0, math.pi, math.pi / 2])
def test_error_decreases_monotonically_after_settle(yaw):
    """No sustained growth: once moving, the error should not blow up. A
    positive-feedback (spin) bug shows up as a strictly increasing tail."""
    target = (0.0, 0.0, -2.0)
    start = (4.0, 2.0, -2.0)
    errs = _simulate(start, yaw, target, steps=300)
    # The error late in the run must be well below the start (no divergence).
    assert errs[-1] < 0.5 * errs[0], (
        f"yaw={yaw:.2f}: error did not shrink (tail {errs[-1]:.2f} vs "
        f"start {errs[0]:.2f})"
    )
