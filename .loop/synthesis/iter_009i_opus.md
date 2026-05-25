# Adversarial review: iter-009i F9 fix (`b926734`) — Opus 4.7

## TL;DR

A **clean refactor** introducing `RacingLineConfig.select_velocity_mps` as the BO oracle's reference speed, demoting `max_velocity_mps` to "informational." The change is **behaviorally a no-op** on every shipping track. The commit's framing ("F9 fix", "No more F9 basin-switching regression", "4-agent consensus") overstates both the scope and the research adoption. No BLOCKERs. MAJORs concentrate on honesty/contract drift, oracle inconsistency, vacuous test, and the new field as a footgun. Cache invalidates via SHA change but version was not bumped.

---

## Context: what actually changed

Pre-iter-009i: bench constructed `RacingLineOptimizer()` (defaults). `RacingLineConfig.max_velocity_mps=15.0`. `_select_by_sim` built oracle trajectories with `DroneConstraints(max_velocity=15.0)`.

Post-iter-009i: bench constructs `RacingLineConfig(max_velocity_mps=<auto>, select_velocity_mps=15.0)`. `_select_by_sim` reads `select_velocity_mps=15.0`. Same scalar, same code path, **identical selected racing line on every shipping track** (race_01, aigp_default, slalom, figure8, grand_tour, straight_hairpin, vertical_cliff). The "race_01 still 12/12 at 17.2s; matrix 6/7 held" verification is static because nothing moved.

The change is real but narrow: it removes the latent footgun that re-attempting the iter-009b deferred wire-up (`RacingLineConfig(max_velocity_mps=<auto>)`) would have re-triggered iter-006 F9 on aigp_default. The regression the commit claims to **fix** was already prevented by iter-009b's **deferral**. iter-009i prevents accidental re-introduction, which is different from fixing.

## BLOCKER

None. The implementation is correct, tests pass, and the field rename has no observable side effects on shipping tracks. I considered three candidates and demoted all to MAJOR:

- Cache key change without version bump (downgraded: deterministic regen, single-entry cache that is always transient).
- PyBullet bench path doesn't explicitly set `select_velocity_mps` (downgraded: default 15.0 matches synthetic's explicit 15.0).
- Overclaimed commit message (downgraded: process/honesty issue, not a code defect).

## MAJOR

### M1. Commit message and code comments overstate the research consensus

The commit text claims "4-agent research swarm consensus (Opus 4.7 max-thinking, GPT-5.5 xhigh, Composer 2, Gemini 3.1 pro) ... unanimous diagnosis ... Fix (Heilmeier 2019 / Kapania 2016 / TUM path-velocity decoupling)". Reading the four `.loop/research/f9_*.md` notes:

- **Opus (f9_opus_4_7.md)**: recommends T1 (convex minimum-curvature QP replacing L-BFGS) as the principled fix, with **T2 (plan-validator feasibility filter + velocity-bucketed cache, `version: 2`)** as the iter-010 patch. Neither is implemented here.
- **GPT-5.5 (f9_gpt55.md)**: recommends multi-fidelity scoring (`{auto, legacy}` candidates), a feasibility phase, and **adding `max_velocity_mps` to the cache key** (the opposite of what shipped).
- **Composer (f9_composer.md)**: lists "freeze the oracle velocity to 15.0" as one option, paired with **fixing `_kinematic_eval`'s hard-coded `max_speed = 15.0`** and **dropping `race_time` from selection or computing it after path shape is fixed**.
- **Gemini (f9_gemini.md)**: recommends `evaluation_velocity = max(self.config.max_velocity_mps, 15.0)` — i.e. *raise* the floor, don't add a separate field.

What unanimously got endorsed: the **diagnosis** (BO is velocity-coupled, min-snap segment times scale with max_velocity, this causes basin shifts). What shipped: Composer's "freeze the oracle" sub-option in isolation, without the secondary fixes (`_kinematic_eval` consistency, race_time drop, feasibility filter). Only ~1-of-4 agents endorsed this specific implementation slice; the others recommended substantially more. Several comments in `racing_line.py` and `benchmark.py` repeat "4-agent research swarm consensus" verbatim — this should be tightened to "consistent with the diagnosis; implements Composer's primary recommendation."

### M2. `_kinematic_eval` still has `max_speed = 15.0` hardcoded — drift trap for the new field

`planning/racing_line.py:551` clamps the BO oracle's virtual drone at `max_speed = 15.0` regardless of `select_velocity_mps`. Today the defaults coincide, so the inconsistency is dormant. The moment anyone overrides `select_velocity_mps` (e.g., a future tight-geometry track sets it to 8.0 to avoid the legacy basin), the trajectory is built at 8.0 m/s but the eval drone is allowed to race ahead to 15.0 m/s, producing distorted `(avg_err, worst_err, race_time)` metrics that no longer reflect what the real drone will do. Composer flagged this as a required secondary bugfix; iter-009i didn't address it. The new `test_select_velocity_DOES_change_geometry` test happens to exercise this exact inconsistent state (`select_velocity_mps=6.0`) but asserts nothing about the metrics.

### M3. `test_select_velocity_DOES_change_geometry` is vacuous and mislabeled

The test name asserts behavior ("DOES change geometry") that the test body never checks. The actual assertions:

```python
assert off_lo.shape == off_hi.shape
assert np.all(np.isfinite(off_lo))
assert np.all(np.isfinite(off_hi))
```

This is a smoke test (no crash, no NaN), not a "DOES change" check. It would pass if `select_velocity_mps` were dead code, if the optimizer always returned zeros, or if a future refactor accidentally wired `_select_by_sim` to read `max_velocity_mps` again. The docstring acknowledges the laxity ("DIAGNOSTIC ... can be flaky on toy layouts") but the test name doesn't. Result: a CI signal that **looks like** positive coverage of the new field's effect, but encodes none. Rename or replace with a fixture from the actual aigp_default basin-switching geometry that asserts a numeric difference (or an `assert not np.allclose(off_lo, off_hi)` with explicit `pytest.xfail` if the toy doesn't show it).

### M4. `max_velocity_mps` as "informational" is a footgun, not a feature

After iter-009i, `RacingLineConfig.max_velocity_mps` is set by exactly one caller (the synthetic bench, line 361) and read by **no downstream consumer in the entire repo** (verified via grep — `_select_by_sim` reads `select_velocity_mps`; no other code touches the field). The field's documented purpose ("execution-speed hint for downstream tooling") has no downstream tooling. This duplicates the iter-006 "dead code BLOCKER for honesty contract" pattern (Opus iter-006 F3): a config field that looks consequential but isn't read by anything is misleading.

Additional risks specific to this field:

1. **Semantic trap (codex_2, composer_3 flagged this independently).** A future contributor reads `RacingLineConfig(max_velocity_mps=6.0)` and reasonably believes the racing-line optimizer is now optimizing for 6 m/s — when it isn't. The fix is "read the 18-line docstring" rather than "make the type system / API self-documenting."
2. **`_dataclass_from_overrides` accepts it silently.** A future track JSON with `"racing_line": {"max_velocity_mps": 8.0}` will set the field with zero effect, with no warning. The PyBullet bench (`benchmark.py:835`) and `visual_demo.py:363` both use this code path.

Either remove the field from `RacingLineConfig` entirely (push it into `PlannerConfig` / `DroneConstraints` where it has real meaning), rename it (`execution_velocity_hint_mps`), or add a runtime check that warns when `max_velocity_mps != select_velocity_mps`.

### M5. PyBullet / visual_demo / RacePipeline never explicitly pin `select_velocity_mps`

The synthetic bench explicitly writes `select_velocity_mps=15.0`. PyBullet (`benchmark.py:856`), visual_demo (`scripts/visual_demo.py:370`), and `race_pipeline.py:240` all inherit the field via `_dataclass_from_overrides` defaults (or `RacingLineOptimizer()` with no config). Today the dataclass default is 15.0, so behavior coincides. If anyone bumps the default to anything else (or a track overrides it under `racing_line:`), the synthetic-vs-PyBullet honesty contract breaks silently: synthetic stays at 15.0, the others drift to the default. One bench-level invariance test asserting "all four entry points resolve to the same effective oracle velocity for the same track JSON" would lock the contract; today there's nothing.

## MINOR

### Mi1. Cache version not bumped despite key schema change

`_compute_cache_key` now includes `select_velocity_mps`. Pre-iter-009i caches were keyed without it (confirmed via `git show b926734^:planning/racing_line.py`). Every existing cache entry will SHA-mismatch on first cold run and be regenerated. `_load_cache` still checks `version != 1`. Bumping to `version: 2` would invalidate stale entries explicitly (and provide a clean migration story), avoid the "silent miss → regenerate → overwrite" thrash on first cold start, and document the schema change in the file itself. Cost: one line.

### Mi2. Shipped `planning/racing_line_cache.json` is from a 2-gate test, not race_01

Working-tree `M` status; the file has 4 offsets (= 2 gates × 2 axes) and was last written 2026-05-25T02:53:24, during the iter-009i verification runs. The cache is single-entry, so the race_01 12-gate offsets that the commit relies on for "still 12/12" are **not** in the shipped file — the next cold race_01 run regenerates them. Deterministic, so behavior survives, but the committed cache state doesn't match the "race_01 verified 12/12" claim. Either commit the race_01 cache as the deterministic seed, or `.gitignore` the cache.

### Mi3. Commit text misattributes the iter-006 mechanism to iter-009

The benchmark.py block at lines 348-357 says "This is the conceptually-correct decoupling that the iter-009 attempt got wrong: it had been coupling SELECTION and EXECUTION through the same velocity." iter-009b (the most recent iter-009) didn't couple them — it explicitly **deferred** the wire-up. iter-006 F9 was the attempt that coupled them. Confuses the timeline for future readers.

### Mi4. Velocity sweep `(5, 8, 12, 15)` never probes `max_velocity_mps > select_velocity_mps`

The invariance test asserts geometry is identical for `max_velocity_mps ∈ {5, 8, 12, 15}` with `select_velocity_mps=15.0`. It never probes the half-space where execution speed exceeds selection reference (e.g. 18 vs 15) — which is allowed by the API. Unlikely to bite, but uncovered.

### Mi5. Function-local `from planning.racing_line import RacingLineConfig` in `benchmark.py:358`

`RacingLineOptimizer` and `SpeedProfiler` are already imported at line 263. Hoist `RacingLineConfig` to the same import for consistency.

---

**Recommend**: ship iter-009i (it's a safe refactor), but follow with iter-010 covering M2 (`_kinematic_eval` clamp threaded from `select_velocity_mps`), M3 (replace vacuous test with aigp_default fixture and a numeric assertion), M4 (rename `max_velocity_mps` or add `__post_init__` warning when it's set without matching `select_velocity_mps`), and Mi1 (cache version bump). M1 is documentation only — soften the consensus language in the commit message and the in-file docstrings.
