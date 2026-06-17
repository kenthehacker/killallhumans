"""Tests for rl.validate_fidelity + rl.champion_control.

Confirms the fidelity validator RUNS and that, with the fitted params, the
champion-in-replica reproduces the headline ground-truth (completion, lap,
descent, roll attenuation) and the held-out open-loop prediction is bounded.

numpy/scipy only. Run: pytest rl/tests/test_validate_fidelity.py -q
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np
import pytest

from rl.dcgame_replica import DCGameReplica, ReplicaParams
from rl.champion_control import ChampionConfig, ChampionDriver, Gate, _gate_normal
from rl.fit_dynamics import DEFAULT_CAPTURE_GLOB, DEFAULT_PARAMS_PATH, load_capture
from rl.validate_fidelity import (
    GATE_MAP_NED, GATE_YAW, SPAWN_POS, SPAWN_YAW,
    TARGET_LAP_S, TARGET_DESCENT_MPS, TARGET_ROLL_ATTEN,
    run_closed_loop, open_loop_prediction, ClosedLoopResult, OpenLoopResult,
)

_HAVE_PARAMS = os.path.exists(DEFAULT_PARAMS_PATH)
_HAVE_CAPS = len(glob.glob(DEFAULT_CAPTURE_GLOB)) > 0


def _params():
    from rl.fit_dynamics import load_params
    return load_params(DEFAULT_PARAMS_PATH)


# --------------------------------------------------------------------------- #
# Champion driver basics                                                      #
# --------------------------------------------------------------------------- #
def test_gate_normal_points_minus_x_at_yaw_pi():
    n = _gate_normal(math.pi, 0.0)
    assert math.isclose(n[0], -1.0, abs_tol=1e-9)
    assert abs(n[1]) < 1e-9 and abs(n[2]) < 1e-9


def test_champion_driver_bakes_aim_z():
    gates = [Gate(p, GATE_YAW, 0.0, i) for i, p in enumerate(GATE_MAP_NED)]
    drv = ChampionDriver(gates, ChampionConfig(aim_z_offset=-0.85))
    # baked gate z = raw z + (-0.85)
    assert math.isclose(drv.gates[0].position[2], GATE_MAP_NED[0][2] - 0.85, abs_tol=1e-9)


def test_champion_driver_emits_finite_commands():
    gates = [Gate(p, GATE_YAW, 0.0, i) for i, p in enumerate(GATE_MAP_NED)]
    drv = ChampionDriver(gates, ChampionConfig())
    sim = DCGameReplica(ReplicaParams()).reset(pos=SPAWN_POS, att=(0, 0, SPAWN_YAW))
    for _ in range(50):
        st = sim.state
        att = drv.step(st.pos, st.vel, st.att[2], 0.01)
        assert all(math.isfinite(x) for x in
                   (att.roll_rad, att.pitch_rad, att.yaw_rad, att.thrust))
        sim.step_attitude((att.roll_rad, att.pitch_rad, att.yaw_rad), att.thrust, 0.01)


# --------------------------------------------------------------------------- #
# Validator runs (smoke) — works even with default params (no captures needed) #
# --------------------------------------------------------------------------- #
def test_run_closed_loop_returns_result():
    cl = run_closed_loop(_params() if _HAVE_PARAMS else ReplicaParams(), dt=0.01,
                         max_t=40.0)
    assert isinstance(cl, ClosedLoopResult)
    assert cl.gates_passed >= 0
    # roll attenuation must be a finite ratio whenever the drone banked
    if cl.roll_attenuation is not None:
        assert 0.0 < cl.roll_attenuation < 2.0


@pytest.mark.skipif(not _HAVE_CAPS, reason="captures not present")
def test_open_loop_prediction_bounded():
    rows = load_capture(sorted(glob.glob(DEFAULT_CAPTURE_GLOB))[-1])
    ol = open_loop_prediction(_params() if _HAVE_PARAMS else ReplicaParams(), rows)
    assert isinstance(ol, OpenLoopResult)
    # 0.5 s horizon prediction must be well under a metre for a faithful map.
    assert ol.pos_rms[0] < 1.0, f"0.5s pos RMS too high: {ol.pos_rms[0]}"
    assert ol.att_rms[0] < 0.5


# --------------------------------------------------------------------------- #
# THE GATE — with the fitted params, the champion reproduces the ground truth  #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_PARAMS, reason="fitted params not present")
def test_fidelity_gate_reproduces_champion():
    cl = run_closed_loop(_params(), dt=0.01, max_t=40.0)
    # completion
    assert cl.completed, f"champion did not complete the course: {cl.gates_passed}/6"
    # lap within 2 s of 16.2
    assert cl.lap_g0_g5 is not None and abs(cl.lap_g0_g5 - TARGET_LAP_S) <= 2.0, \
        f"lap {cl.lap_g0_g5} vs {TARGET_LAP_S}"
    # descent within 0.6 m/s of 2.27
    assert cl.descent_g0_g3 is not None and abs(cl.descent_g0_g3 - TARGET_DESCENT_MPS) <= 0.6, \
        f"descent {cl.descent_g0_g3} vs {TARGET_DESCENT_MPS}"
    # the 0.53 roll attenuation emerges within 0.15
    assert cl.roll_attenuation is not None and abs(cl.roll_attenuation - TARGET_ROLL_ATTEN) <= 0.15, \
        f"attenuation {cl.roll_attenuation} vs {TARGET_ROLL_ATTEN}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
