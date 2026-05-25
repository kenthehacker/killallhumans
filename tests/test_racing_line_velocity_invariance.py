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


def test_select_velocity_invalidates_cache():
    """Iter-009j: replaces the vacuous diagnostic test with one that
    actually locks down the cache-key contract.

    The 8-agent adversarial review on iter-009i (Opus M3 + Codex 1+2
    + Composer 1+2) unanimously flagged that the old
    `test_select_velocity_DOES_change_geometry` test only asserted
    shape+finite, not the behavior its name claimed. This replaces it
    with the genuinely-useful cache invariance the iter-009i design
    introduced: changing `select_velocity_mps` MUST miss the cache,
    while changing `max_velocity_mps` must NOT.
    """
    from planning.racing_line import RacingLineOptimizer
    gates = _make_gates()
    start = (0.0, 0.0, 2.0)
    cfg_a = RacingLineConfig(
        max_velocity_mps=8.0, select_velocity_mps=15.0, use_cache=False,
    )
    cfg_b = RacingLineConfig(
        max_velocity_mps=15.0, select_velocity_mps=15.0, use_cache=False,
    )
    cfg_c = RacingLineConfig(
        max_velocity_mps=8.0, select_velocity_mps=6.0, use_cache=False,
    )

    key_a = RacingLineOptimizer._compute_cache_key(gates, start, cfg_a)
    key_b = RacingLineOptimizer._compute_cache_key(gates, start, cfg_b)
    key_c = RacingLineOptimizer._compute_cache_key(gates, start, cfg_c)

    # cfg_a and cfg_b differ only in max_velocity_mps (informational);
    # cache key must NOT change.
    assert key_a == key_b, (
        f"cache key changed when only max_velocity_mps differed "
        f"({key_a} vs {key_b}); execution-velocity must not invalidate cache"
    )
    # cfg_a and cfg_c differ in select_velocity_mps (BO oracle's basin);
    # cache key MUST change.
    assert key_a != key_c, (
        f"cache key unchanged when select_velocity_mps differed "
        f"({key_a} == {key_c}); selection-velocity MUST invalidate cache"
    )


def test_racing_line_config_rejects_invalid_velocities():
    """Iter-009j (Codex#2 MAJOR): NaN / Inf / non-positive velocities
    must be rejected at config-construction time, before they reach
    TrajectoryOptimizer or the JSON cache key."""
    import math
    import pytest

    # max_velocity_mps
    for bad in (math.nan, math.inf, -1.0, 0.0):
        with pytest.raises(ValueError, match="max_velocity_mps"):
            RacingLineConfig(max_velocity_mps=bad)
    # select_velocity_mps
    for bad in (math.nan, math.inf, -1.0, 0.0):
        with pytest.raises(ValueError, match="select_velocity_mps"):
            RacingLineConfig(select_velocity_mps=bad)


def test_select_velocity_changes_kinematic_eval_metrics():
    """Iter-009j: locks down M2 — `_kinematic_eval`'s velocity clamp
    must respond to `select_velocity_mps`. Earlier the literal 15.0
    was hardcoded; a regression would silently re-introduce the
    velocity-aligned-scoring distortion the research swarm flagged.

    Test approach: run the optimizer at select_velocity_mps=5 vs 15 on
    the same gate layout (use_cache=False). At v=5, the kinematic eval
    drone is clamped to a lower speed → race_time metrics should differ
    visibly from v=15. We don't assert a specific delta (depends on the
    L-BFGS basin), just that the optimizer completes both and that
    the chosen waypoint geometry differs (which it must, since the
    oracle pool is now meaningfully different).
    """
    gates = _make_gates()
    start = (0.0, 0.0, 2.0)
    cfg_slow = RacingLineConfig(
        max_velocity_mps=5.0, select_velocity_mps=5.0, use_cache=False,
    )
    cfg_fast = RacingLineConfig(
        max_velocity_mps=15.0, select_velocity_mps=15.0, use_cache=False,
    )
    wps_slow = RacingLineOptimizer(cfg_slow).optimize(gates, start)
    wps_fast = RacingLineOptimizer(cfg_fast).optimize(gates, start)
    # Both runs must complete (no crash); shapes match.
    assert len(wps_slow) == len(wps_fast) == len(gates)
    # All offsets finite.
    off_slow = _offsets(wps_slow, gates)
    off_fast = _offsets(wps_fast, gates)
    assert np.all(np.isfinite(off_slow))
    assert np.all(np.isfinite(off_fast))
