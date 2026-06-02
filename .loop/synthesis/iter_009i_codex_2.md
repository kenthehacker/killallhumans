# Second Codex Adversarial Review: iter-009i F9 Fix

Commit reviewed: `b926734` (`iter-009i: F9 fix - path-velocity decoupling for racing-line selection`).

## Findings

### Major: `select_velocity_mps` is not validated before it controls both cache identity and trajectory construction

`RacingLineConfig.select_velocity_mps` is now the actual BO oracle speed, but it accepts any float. `_compute_cache_key()` rounds it with `round(config.select_velocity_mps, 2)` and `json.dumps()` will serialize non-finite values as `NaN`, `Infinity`, or `-Infinity`. `_select_by_sim()` then passes the raw value into `DroneConstraints(max_velocity=...)`.

Consequences:

- Fractional values below the 0.01 bucket collide in the cache even though the optimizer uses the unrounded speed. Example: `15.004` keys as `15.0`, so a later run can reuse geometry selected at a different oracle speed.
- `NaN` and infinities produce stable but non-standard cache keys and then poison trajectory generation. Candidate exceptions are swallowed as `(999,999,999)`, so a bad velocity can silently fall back to L-BFGS objective selection and cache that geometry under a nonsensical oracle-speed key.
- Zero or negative values are also not rejected. They can make time allocation invalid or force the same fallback path.

Recommendation: enforce `math.isfinite(select_velocity_mps) and select_velocity_mps > 0` at config use time, and either key by the exact canonical float representation or consciously quantize after documenting that oracle speeds are bucketed. Add cache-specific tests for fractional collision and non-finite rejection; the current invariance test bypasses cache.

### Moderate: cache writes remain globally shared and non-atomic

The cache is one process-wide JSON file containing one entry. Including `select_velocity_mps` in the key avoids cross-speed reuse when the file is intact, but concurrent optimizers can still race on `_save_cache()`: writers truncate and rewrite the same file without a lock or atomic rename.

This is mostly determinism/performance risk rather than a direct wrong-key read, because `_load_cache()` checks the key and treats JSON parse failures as misses. Still, two benchmarks or agent loops running different tracks can thrash the cache, observe partial JSON, recompute unnecessarily, and leave whichever run writes last as the only cached entry. If the cache is now part of the determinism story, it should use atomic write-and-rename at minimum; a multi-entry `{cache_key: entry}` layout plus file locking would be better.

### Moderate: the explicit `select_velocity_mps=15.0` bench wiring is synthetic-only

The synthetic benchmark now constructs `RacingLineConfig(max_velocity_mps=max_velocity, select_velocity_mps=15.0)`. The PyBullet benchmark and `visual_demo.py` still construct `RacingLineConfig` only from `race_config.racing_line_overrides`, so they rely on the dataclass default `select_velocity_mps=15.0` rather than an explicit pin. That is behaviorally equivalent today, but it means "bench constructs with both" is not true for all benchmark paths.

I probed `aigp_default` through `run_synthetic_benchmark(..., config=aigp_default_with_use_cache_false)` and the fixed 15 m/s selector does still complete: 6/6 gates, no crash/DQ, plan validation OK, avg tracking error about `0.205m`, max `0.618m`, trajectory time about `7.93s`. So this patch appears safe for `aigp_default` in the synthetic path.

The broader design question remains: 15 m/s is validated as the legacy race_01-class selector, not proven as the right reference for every future course. If `select_velocity_mps` is intentionally a geometry oracle reference, tracks should be able to override it explicitly and the matrix should include at least race_01, aigp_default, and a tight/slalom course with cache disabled.

### Minor: no current non-test caller passes `max_velocity_mps` expecting BO behavior, but the config API silently accepts that obsolete expectation

Repository search shows no non-test caller currently passes `max_velocity_mps` into `RacingLineConfig` expecting it to affect `_select_by_sim`, except the synthetic benchmark where it is deliberately informational. Existing JSON configs also do not put `max_velocity_mps` under the `racing_line` section.

However, `_dataclass_from_overrides(RacingLineConfig, race_config.racing_line_overrides)` will still accept a future/old `"racing_line": {"max_velocity_mps": ...}` override and silently ignore it for selection. That is a compatibility footgun. If this semantic split is final, consider a warning, rename, or config validation that rejects `racing_line.max_velocity_mps` unless paired with explicit `select_velocity_mps`.

### Minor: `test_select_velocity_DOES_change_geometry` is not testing what its name says

The test is framed as "DOES change geometry", but it deliberately only asserts both outputs have the same shape and finite values. It would pass even if `select_velocity_mps` were accidentally unused by `_select_by_sim`, which is exactly the class of regression this new field needs to guard against.

On the current toy layout, 6.0 vs 15.0 produces only a tiny vertical-offset difference (`max_abs_diff ~= 1.28e-4`) while lateral offsets are saturated identically. That is weak evidence for "selection speed governs basin choice." Either rename this as a smoke test, or replace it with a fixture based on the real F9/aigp_default basin-switching layout and assert a meaningful geometry or selected-index difference. As written, the diagnostic framing hides a possible bug rather than protecting the design.

## Bottom Line

The core decoupling is plausible and the synthetic `aigp_default` probe is green, but I would not call this fully hardened. The biggest gap is cache/config hygiene around `select_velocity_mps`: non-finite and bucket-colliding values can produce silent fallback geometry and cache aliases. The second gap is test intent: invariance of `max_velocity_mps` is covered, but positive evidence that `select_velocity_mps` actually controls selection is not.
