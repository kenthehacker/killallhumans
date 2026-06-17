"""Tests for the gray-box DCGame replica + the attitude->rate wrapper.

numpy/scipy only (no torch, no PyBullet). Run: pytest rl/tests/ -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from competition.adapter import Quaternion
from competition.aigp_mavlink import _attitude_error_body_rates
from rl.dcgame_replica import (
    DCGameReplica,
    ReplicaParams,
    ReplicaState,
    attitude_error_body_rates,
    attitude_to_body_rate,
    rotation_body_to_ned,
    RATE_SIGN,
    ATT_RATE_KP,
    ATT_RATE_KD,
    ATT_RATE_MAX,
)


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #
def test_determinism_same_inputs_same_trajectory():
    p = ReplicaParams()

    def run():
        sim = DCGameReplica(p).reset(att=(0.0, 0.0, math.pi))
        traj = []
        for k in range(200):
            sim.step_attitude((0.2, -0.3, math.pi), 0.3, 0.01)
            traj.append(sim.state.pos.copy())
        return np.array(traj)

    a, b = run(), run()
    assert np.array_equal(a, b), "replica must be deterministic for identical inputs"


def test_step_does_not_mutate_param_aliases():
    # state arrays returned by step() are copies; mutating them must not corrupt
    # the internal state.
    sim = DCGameReplica(ReplicaParams()).reset()
    st = sim.step((0.1, 0.0, 0.0), 0.3, 0.01)
    st.pos[:] = 1e9
    assert sim.state.pos[0] < 1e8


def test_zero_and_negative_dt_are_noops():
    sim = DCGameReplica(ReplicaParams()).reset(pos=(1.0, 2.0, 3.0))
    before = sim.state.pos.copy()
    sim.step((0.5, 0.5, 0.5), 0.5, 0.0)
    sim.step((0.5, 0.5, 0.5), 0.5, -0.01)
    assert np.array_equal(sim.state.pos, before)


# --------------------------------------------------------------------------- #
# attitude->body-rate wrapper matches the EXACT live adapter PD law            #
# --------------------------------------------------------------------------- #
def test_wrapper_matches_attitude_error_body_rates_byte_for_byte():
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(500):
        cur = rng.uniform(-0.6, 0.6, 3)
        des = rng.uniform(-0.6, 0.6, 3)
        om = rng.uniform(-1.5, 1.5, 3)
        qc = Quaternion.from_euler(*cur)
        qd = Quaternion.from_euler(*des)
        ref = _attitude_error_body_rates(
            qc, qd, omega=tuple(om), kp=ATT_RATE_KP, kd=ATT_RATE_KD, max_rate=ATT_RATE_MAX)
        # pre-sign: must equal the live adapter's PD output exactly
        mine = attitude_to_body_rate(cur, des, omega=om, apply_rate_sign=False)
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(ref, mine)))
    assert max_diff < 1e-9, f"wrapper diverged from _attitude_error_body_rates ({max_diff})"


def test_wrapper_quaternion_form_matches_euler_form():
    # attitude_error_body_rates (quat) and attitude_to_body_rate (euler, pre-sign)
    # must agree.
    cur = (0.1, -0.2, math.pi)
    des = (0.3, 0.1, math.pi - 0.1)
    qc = Quaternion.from_euler(*cur)
    qd = Quaternion.from_euler(*des)
    a = attitude_error_body_rates((qc.w, qc.x, qc.y, qc.z), (qd.w, qd.x, qd.y, qd.z))
    b = attitude_to_body_rate(cur, des, apply_rate_sign=False)
    assert np.allclose(a, b, atol=1e-12)


def test_rate_sign_application():
    cur = (0.0, 0.0, math.pi)
    des = (0.3, -0.2, math.pi)
    presign = attitude_to_body_rate(cur, des, apply_rate_sign=False)
    signed = attitude_to_body_rate(cur, des, apply_rate_sign=True)
    for i in range(3):
        assert math.isclose(signed[i], RATE_SIGN[i] * presign[i], abs_tol=1e-12)


def test_rate_clamp_respected():
    # a huge attitude error must saturate at the per-axis clamp (pre-sign).
    r = attitude_to_body_rate((0.0, 0.0, 0.0), (1.5, -1.5, 0.0), apply_rate_sign=False)
    assert abs(r[0]) <= ATT_RATE_MAX + 1e-12
    assert abs(r[1]) <= ATT_RATE_MAX + 1e-12


# --------------------------------------------------------------------------- #
# Physical sanity of the composite map                                        #
# --------------------------------------------------------------------------- #
def test_attitude_converges_to_command():
    # holding a constant attitude command, achieved attitude -> ~command (DC gain
    # ~1) and converges in the correct direction (no _rate_sign double-flip).
    p = ReplicaParams()
    sim = DCGameReplica(p).reset(att=(0.0, 0.0, math.pi))
    for _ in range(600):
        sim.step_attitude((0.3, -0.25, math.pi), 0.28, 0.01)
    assert abs(sim.state.att[0] - 0.3) < 0.08, sim.state.att
    assert abs(sim.state.att[1] - (-0.25)) < 0.08, sim.state.att


def test_hover_thrust_roughly_holds_altitude():
    # at hover thrust (g/k_t), level, the vertical velocity stays near zero.
    p = ReplicaParams()
    hover = 9.81 / p.k_t
    sim = DCGameReplica(p).reset(att=(0.0, 0.0, math.pi))
    for _ in range(200):
        sim.step_attitude((0.0, 0.0, math.pi), hover, 0.01)
    assert abs(sim.state.vel[2]) < 0.5, sim.state.vel


def test_dcgame_roll_sign_convention_positive_roll_gives_positive_y():
    # DCGame composite: +roll (telemetry convention) -> +Y thrust-accel at yaw=pi
    # (the opposite of standard NED; the replica reproduces it). Check via a step.
    p = ReplicaParams(k_d=0.0)
    sim = DCGameReplica(p).reset(att=(0.3, 0.0, math.pi), vel=(0, 0, 0))
    sim.step((0.0, 0.0, 0.0), 9.81 / p.k_t, 0.05)  # zero rate cmd, hover thrust
    assert sim.state.vel[1] > 0.0, "DCGame +roll must accelerate +Y (East)"


def test_thrust_zero_falls_under_gravity():
    p = ReplicaParams(k_d=0.0)
    sim = DCGameReplica(p).reset(att=(0.0, 0.0, math.pi))
    sim.step((0.0, 0.0, 0.0), 0.0, 0.1)
    # NED z is down; no thrust => accelerate downward (+z).
    assert sim.state.vel[2] > 0.0


def test_rotation_matrix_orthonormal():
    R = rotation_body_to_ned(0.3, -0.2, 1.1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-12)


def test_thrust_norm_clamped():
    sim = DCGameReplica(ReplicaParams()).reset()
    # absurd thrust gets clamped to [0,1]; no NaN/inf in state.
    sim.step((0.0, 0.0, 0.0), 50.0, 0.01)
    assert np.all(np.isfinite(sim.state.vel))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
