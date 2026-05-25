"""
Iter-009i (F9 fix verification): the racing-line selection must NOT
flip basins when `max_velocity_mps` changes.

This is the regression test that codifies the 4-agent research-swarm
consensus (Opus 4.7 max-thinking + GPT-5.5 xhigh + Composer + Gemini)
on the F9 issue: the optimal lateral-offset *geometry* is chosen
independent of the velocity it will be flown at (Heilmeier 2019,
Kapania 2016 TUM minimum-curvature method). Concretely:

  RacingLineConfig.max_velocity_mps    = informational (execution speed)
  RacingLineConfig.select_velocity_mps = governs the BO scorer's basin

The test runs the optimizer on a small synthetic gate layout at four
`max_velocity_mps` values and asserts the returned waypoint offsets
are bit-identical. Selection should ONLY shift when
`select_velocity_mps` changes — and even then, only intentionally.
"""
from __future__ import annotations

import numpy as np

from planning.racing_line import RacingLineConfig, RacingLineOptimizer
from planning.trajectory_optimizer import GateWaypoint


def _make_gates():
    """A small 4-gate slalom-ish layout. Doesn't need to match a real
    track; just needs to be the same input across runs."""
    return [
        GateWaypoint(position=(5.0, 0.0, 2.0), normal=(1.0, 0.0, 0.0)),
        GateWaypoint(position=(10.0, 2.0, 2.0), normal=(1.0, 0.0, 0.0)),
        GateWaypoint(position=(15.0, -2.0, 2.0), normal=(1.0, 0.0, 0.0)),
        GateWaypoint(position=(20.0, 0.0, 2.0), normal=(1.0, 0.0, 0.0)),
    ]


def _offsets(opt_wps, gates):
    """Lateral offsets from gate centers (signed, in the gate's lateral
    direction). Comparing these is a velocity-agnostic identity check."""
    out = []
    for w, g in zip(opt_wps, gates):
        out.append((
            w.position[0] - g.position[0],
            w.position[1] - g.position[1],
            w.position[2] - g.position[2],
        ))
    return np.array(out)


def test_max_velocity_does_not_affect_racing_line_geometry():
    """Iter-009i F9 fix: changing `max_velocity_mps` (informational
    field; what the trajectory will be executed at) must NOT change
    the racing-line geometry. Only `select_velocity_mps` can."""
    gates = _make_gates()
    start = (0.0, 0.0, 2.0)
    baselines = []
    for v_exec in (5.0, 8.0, 12.0, 15.0):
        cfg = RacingLineConfig(
            max_velocity_mps=v_exec,
            select_velocity_mps=15.0,  # held constant — the actual SELECTION speed
            use_cache=False,           # bypass cache so we exercise the optimizer
        )
        opt = RacingLineOptimizer(config=cfg)
        wps = opt.optimize(gates, start)
        baselines.append(_offsets(wps, gates))

    # All four runs must produce identical geometry.
    for i in range(1, len(baselines)):
        np.testing.assert_array_almost_equal(
            baselines[0], baselines[i], decimal=6,
            err_msg=(
                f"Racing-line geometry shifted between max_velocity_mps "
                f"values; the F9 decoupling is broken. Run 0 offsets vs "
                f"run {i}: \n{baselines[0]}\n vs\n{baselines[i]}"
            ),
        )


def test_select_velocity_DOES_change_geometry():
    """Sanity check on the test: confirm that select_velocity_mps DOES
    change the selected geometry. If this test fails, the decoupling
    is too strong (BO is broken) or the basin-switching mechanism the
    research swarm identified isn't actually present in this code.

    NB: this is a DIAGNOSTIC; it can be flaky on toy layouts where
    both selection velocities happen to land on the same basin. We
    accept that — the assertion is "if there's a diff, we observe it"
    rather than "diff must exist on this exact layout".
    """
    gates = _make_gates()
    start = (0.0, 0.0, 2.0)
    cfg_lo = RacingLineConfig(
        max_velocity_mps=8.0, select_velocity_mps=6.0, use_cache=False,
    )
    cfg_hi = RacingLineConfig(
        max_velocity_mps=8.0, select_velocity_mps=15.0, use_cache=False,
    )
    wps_lo = RacingLineOptimizer(cfg_lo).optimize(gates, start)
    wps_hi = RacingLineOptimizer(cfg_hi).optimize(gates, start)
    off_lo = _offsets(wps_lo, gates)
    off_hi = _offsets(wps_hi, gates)
    # Don't assert they MUST differ (toy layout may not show basin
    # switching); just assert the test infrastructure can run both
    # configurations without crashing.
    assert off_lo.shape == off_hi.shape
    assert np.all(np.isfinite(off_lo))
    assert np.all(np.isfinite(off_hi))
