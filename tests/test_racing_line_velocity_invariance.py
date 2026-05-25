"""
Iter-009i + iter-009j + iter-009k racing-line velocity decoupling tests.

The 4-agent research swarm (Opus 4.7, GPT-5.5, Composer 2, Gemini 3.1)
unanimously diagnosed F9 as min-snap segment-time velocity coupling that
causes the BO scorer to flip basins under different `max_velocity`. The
fix (Heilmeier 2019, Kapania 2016 TUM minimum-curvature method) is to
decouple the SELECTION velocity from the EXECUTION velocity.

Iter-009i added `RacingLineConfig.select_velocity_mps` and kept the old
`max_velocity_mps` field as informational. The 7-agent adversarial
review of iter-009i flagged that informational field as a semantic API
trap (4/7 reviewers), so iter-009k REMOVED it. The only remaining knob
is `select_velocity_mps`, which controls both the BO oracle's
TrajectoryOptimizer cap and (via iter-009j) the `_kinematic_eval`
clamp.

Tests below codify the invariants:
  1. The cache key must change on `select_velocity_mps` change.
  2. NaN / Inf / non-positive velocities must be rejected at config
     construction.
  3. Sweeping `select_velocity_mps` across reasonable values returns
     well-formed trajectories (no crashes, all offsets finite).
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
    """Raw world-frame deltas from gate centres. Identity check across
    runs — not gate-frame projection (the synthetic gates here all face
    +X so it coincides). See iter_009i_composer_2.md issue 3 for the
    distinction; on rotated/yawed gates this would need gate-frame
    projection to be the "lateral offset" the review prose implied."""
    out = []
    for w, g in zip(opt_wps, gates):
        out.append((
            w.position[0] - g.position[0],
            w.position[1] - g.position[1],
            w.position[2] - g.position[2],
        ))
    return np.array(out)


def test_select_velocity_invalidates_cache():
    """Locks the cache-key contract: changing `select_velocity_mps`
    MUST miss the cache, since the BO oracle's basin choice depends on
    it. Cache version bump (iter-009i) splits stale entries safely.
    """
    gates = _make_gates()
    start = (0.0, 0.0, 2.0)
    cfg_a = RacingLineConfig(select_velocity_mps=15.0, use_cache=False)
    cfg_b = RacingLineConfig(select_velocity_mps=6.0, use_cache=False)

    key_a = RacingLineOptimizer._compute_cache_key(gates, start, cfg_a)
    key_b = RacingLineOptimizer._compute_cache_key(gates, start, cfg_b)

    assert key_a != key_b, (
        f"cache key unchanged when select_velocity_mps differed "
        f"({key_a} == {key_b}); selection-velocity MUST invalidate cache"
    )


def test_racing_line_config_rejects_invalid_velocities():
    """Iter-009j (Codex#2 MAJOR): NaN / Inf / non-positive velocities
    must be rejected at config-construction time, before they reach
    TrajectoryOptimizer or the JSON cache key."""
    import math
    import pytest

    for bad in (math.nan, math.inf, -1.0, 0.0):
        with pytest.raises(ValueError, match="select_velocity_mps"):
            RacingLineConfig(select_velocity_mps=bad)


def test_select_velocity_sweep_produces_well_formed_geometry():
    """Sweeping select_velocity_mps across {5, 8, 12, 15} must return
    well-formed geometry every time: same length, all finite offsets,
    no crash. Replaces the iter-009i max_velocity_mps invariance test
    (which lost meaning once max_velocity_mps was removed in iter-009k)
    with a positive coverage of the only remaining velocity knob.
    """
    gates = _make_gates()
    start = (0.0, 0.0, 2.0)
    for v in (5.0, 8.0, 12.0, 15.0):
        cfg = RacingLineConfig(select_velocity_mps=v, use_cache=False)
        wps = RacingLineOptimizer(cfg).optimize(gates, start)
        assert len(wps) == len(gates), (
            f"got {len(wps)} waypoints for {len(gates)} gates at "
            f"select_velocity_mps={v}"
        )
        off = _offsets(wps, gates)
        assert np.all(np.isfinite(off)), (
            f"non-finite offset at select_velocity_mps={v}: {off}"
        )
