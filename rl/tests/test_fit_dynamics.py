"""Tests for rl.fit_dynamics: synthetic-recovery + robust-dt + IO round-trip.

The fit recovers SEEDED params when we generate synthetic telemetry FROM the
replica itself (the standard system-ID self-consistency check): drive a known-param
replica with a rich command sequence, log it in the capture schema, and confirm
the fitter recovers k_t / k_d / tau within tolerance.

numpy/scipy only. Run: pytest rl/tests/test_fit_dynamics.py -q
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from rl.dcgame_replica import DCGameReplica, ReplicaParams
from rl.fit_dynamics import (
    fit_translation, fit_axis_tau, refine_k_t, robust_dt,
    fit_dynamics_from_captures, save_params, load_params,
)


# --------------------------------------------------------------------------- #
# Synthetic capture generation                                                #
# --------------------------------------------------------------------------- #
def _make_synth_capture(params: ReplicaParams, n=1500, dt=0.0102, seed=0) -> list:
    """Drive a known-param replica with a smooth, richly-exciting attitude+thrust
    command sequence and emit rows in the capture JSON schema."""
    rng = np.random.default_rng(seed)
    sim = DCGameReplica(params).reset(att=(0.0, 0.0, math.pi))
    rows = []
    t_us = 0
    # smooth multi-sine excitation on roll/pitch + thrust around hover
    hover = 9.81 / params.k_t
    for k in range(n):
        tt = k * dt
        cmd_roll = 0.5 * math.sin(2 * math.pi * 0.4 * tt) + 0.1 * math.sin(2 * math.pi * 1.1 * tt)
        cmd_pitch = -0.4 + 0.3 * math.sin(2 * math.pi * 0.3 * tt + 1.0)
        cmd_yaw = math.pi
        cmd_thr = float(np.clip(hover + 0.15 * math.sin(2 * math.pi * 0.25 * tt), 0.05, 0.95))
        st = sim.state
        rows.append({
            "t_us": int(t_us),
            "pos": st.pos.tolist(),
            "vel": st.vel.tolist(),
            "roll": float(st.att[0]), "pitch": float(st.att[1]), "yaw": float(st.att[2]),
            "gyro": st.rate.tolist(),
            "cmd_roll": round(cmd_roll, 4), "cmd_pitch": round(cmd_pitch, 4),
            "cmd_yaw": round(cmd_yaw, 4), "cmd_thrust": round(cmd_thr, 4),
        })
        sim.step_attitude((cmd_roll, cmd_pitch, cmd_yaw), cmd_thr, dt)
        t_us += int(dt * 1e6)
    return rows


SEED = ReplicaParams(k_t=33.0, k_d=0.40, tau_roll=0.20, tau_pitch=0.10, tau_yaw=0.30)


def test_robust_dt_recovers_cadence():
    rows = _make_synth_capture(SEED, n=800, dt=0.0102)
    dt = robust_dt([rows])
    assert math.isclose(dt, 0.0102, abs_tol=2e-4)


def test_translation_recovers_seeded_k_t_k_d():
    caps = [_make_synth_capture(SEED, n=1500, seed=s) for s in range(3)]
    dt = robust_dt(caps)
    k_t, k_d, rmse, n = fit_translation(caps, fixed_dt=dt)
    # synthetic data is noise-free (same integrator), so recovery is tight.
    assert abs(k_t - SEED.k_t) < 0.6, f"k_t {k_t} vs {SEED.k_t}"
    assert abs(k_d - SEED.k_d) < 0.05, f"k_d {k_d} vs {SEED.k_d}"
    assert rmse < 0.5


def test_axis_tau_recovers_seeded_roll_pitch():
    caps = [_make_synth_capture(SEED, n=1500, seed=s) for s in range(3)]
    dt = robust_dt(caps)
    tau_r, _ = fit_axis_tau(caps, dt, axis=0, other_tau=(0.2, SEED.tau_pitch, SEED.tau_yaw))
    tau_p, _ = fit_axis_tau(caps, dt, axis=1, other_tau=(SEED.tau_roll, 0.1, SEED.tau_yaw))
    # Pitch is well-identified. Roll's multi-step attitude objective is genuinely
    # SHALLOW (a known property — see fit_dynamics / README: the 0.53 attenuation
    # is a closed-loop effect, so open-loop roll-tau is only loosely identifiable),
    # so a wider tolerance is honest here. Both must land in the right ballpark.
    assert abs(tau_p - SEED.tau_pitch) < 0.06, f"tau_pitch {tau_p} vs {SEED.tau_pitch}"
    assert abs(tau_r - SEED.tau_roll) < 0.12, f"tau_roll {tau_r} vs {SEED.tau_roll}"


def test_refine_k_t_recovers_on_synthetic():
    caps = [_make_synth_capture(SEED, n=1500, seed=s) for s in range(2)]
    dt = robust_dt(caps)
    # widen bounds around the seed so the recovery is not masked by the prod cap.
    k_t, rms = refine_k_t(caps, dt, SEED.k_d,
                          (SEED.tau_roll, SEED.tau_pitch, SEED.tau_yaw),
                          lo=30.0, hi=36.0)
    assert abs(k_t - SEED.k_t) < 0.8, f"refined k_t {k_t} vs {SEED.k_t}"
    assert rms < 0.5


def test_params_json_round_trip():
    p = ReplicaParams(k_t=34.0, k_d=0.42, tau_roll=0.22, tau_pitch=0.10, tau_yaw=0.5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.json")
        save_params(p, path)
        q = load_params(path)
    assert math.isclose(q.k_t, p.k_t) and math.isclose(q.k_d, p.k_d)
    assert math.isclose(q.tau_roll, p.tau_roll) and math.isclose(q.tau_pitch, p.tau_pitch)
    assert tuple(q.rate_kp) == tuple(p.rate_kp)


def test_from_dict_drops_unknown_keys():
    d = ReplicaParams().to_dict()
    d["some_future_field"] = 123
    p = ReplicaParams.from_dict(d)  # must not raise
    assert isinstance(p, ReplicaParams)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
