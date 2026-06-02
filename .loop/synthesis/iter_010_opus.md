# Adversarial review: iter-010 drone-spec unification (`f1505ee`) — Opus 4.7

## TL;DR

A clean refactor that centralises drone constants into `competition/drone_spec.py` and drops `DroneConstraints.max_acceleration` 20→15. **No BLOCKER** — matrix gates hold and no callsite explicitly overrides the new default. **Four MAJORs** concentrate on (1) the omitted cache-version bump, (2) the hypothesis–data mismatch (aigp_default tracking *increased*), (3) a likely L-BFGS-semantics confusion that drove the 20→15 choice, and (4) the still-incomplete unification (benchmark/ILC/tracker continue to inline duplicates). Several MINORs around naming, frozen-vs-mutable asymmetry, dead imports, and missing regression coverage.

---

## Context

Pre-iter-010: `DroneConstraints.max_acceleration = 20.0` (iter-005 deliberate relaxation; comment: *"rough estimate overestimates actual accel at segment boundaries"*). `scripts/benchmark.py:486` vector-clamps actual accel at 15.0. The commit's hypothesis: trajectories time-allocated under 20 m/s² command accels the bench clamps to 15, producing feedforward mismatch and inflated tracking error.

Post-iter-010: a new `competition/drone_spec.py` module exposes `DEFAULT_MAX_ACCEL_MPS2 = 15.0` and a frozen `DroneSpec` dataclass. `DroneConstraints` re-imports nine field defaults from drone_spec. The `max_acceleration` default drops 20→15; the other eight field values are numerically unchanged.

---

## BLOCKER

None. The implementation is correct, every shipping test passes, every `DroneConstraints(...)` callsite in the tree only overrides `max_velocity` (verified — no leftover `max_acceleration=20.0` literals anywhere in production code). The 0.205→0.233 m tracking degradation on aigp_default is well inside the 0.40 m matrix gate. Demoted candidates:

- *Hypothesis–data mismatch on aigp_default* → MAJOR (degradation is real but bounded; matrix gate protects).
- *Cache invalidation skipped* → MAJOR (cache is gitignored / transient; effect is on-host reproducibility, not CI correctness).

## MAJOR

### M1. Cache-version bump from the Opus plan was NOT performed

`next_iter_opus.md` step 3 explicitly required: *"Cache invalidation (mirror iter-009l): bump racing_line_cache.json schema version since trajectory shapes change with max_acceleration=15."* The shipping commit leaves `RacingLineOptimizer._save_cache` at `version: 2` and the cache_key composition (`racing_line.py:174-185`) at the iter-009i fields. The key DOES NOT include `max_acceleration`.

Consequence: `_select_by_sim` builds candidate trajectories via `DroneConstraints(max_velocity=self.config.select_velocity_mps)` — now defaulting `max_acceleration=15.0`. A cache hit from iter-009l (selected under 20) returns offsets that were optimal for the OLD dynamics. On this checkout, `planning/racing_line_cache.json` is timestamped 20:43 — one minute before commit `f1505ee` at 20:44. Whether the "race_01 12/12 at 17.2s, 0.089m tracking (no change from iter-009l baseline)" claim holds against a *clean* cache is therefore untested by the commit's verification.

Worse, this is a structural F9-class regression risk: future runs with a primed iter-009l cache see different offsets from CI runs with cold cache, and neither reflects what a fresh competition run would produce. The fix is one-line — bump `version: 2 → 3` and add `_DRONE_MAX_ACCEL` to the cache_key — and it was on the plan. Skipping it without comment is the single biggest defect in this commit.

### M2. The fix's predicted outcome is contradicted by the data

The commit explicitly claims: *"A polynomial trajectory time-allocated under 20 m/s² commands accelerations the bench then clamps to 15 → feedforward mismatch and inflated tracking error."* If correct, lowering `max_acceleration` from 20 to 15 should *reduce* tracking error on tracks where the saturation regime was active.

Measured outcome (commit body): aigp_default tracking 0.205 → 0.233 m, a **+14%** increase. race_01 is unchanged. The simplest explanation consistent with both observations is that **the bench's vector clamp on the polynomial accel never actually engaged on these tracks** — peak polynomial accel was already below 15 m/s². If so, the "feedforward mismatch" hypothesis is unsupported, and the 20→15 change merely tightens an unrelated soft penalty in `_optimize_time_allocation` (M3 below), making the trajectory slower for no compensating tracking benefit.

The commit acknowledges the regression ("still under the 0.40m threshold; trade-off is now physical-accurate") but does not reconcile it with the hypothesis. Either the hypothesis is partially wrong, or the saturation regime needs to be empirically demonstrated (instrument `benchmark.py:486` to log clamp-active fraction before declaring victory on "alignment").

### M3. The 20.0 default was a deliberate L-BFGS estimator margin, not a careless lie

`_optimize_time_allocation` (line 1641-1648) penalizes:

```
accel = abs(v2 - v1) / max(times[i], 0.01)
if accel > self.constraints.max_acceleration:
    penalty += (accel - self.constraints.max_acceleration) ** 2 * 20
```

This `accel` is the **segment-boundary scalar finite-difference of segment-averaged speeds** — not the peak polynomial accel that the bench clamps. The iter-005 commit log and the in-line comment that iter-010 deleted (`"# (~2g) — relaxed: rough estimate overestimates actual accel at segment boundaries"`) state explicitly that this estimator is **conservative by ~50%** vs the realised polynomial accel. Setting `max_acceleration = 20.0` was a calibrated allowance to bring the soft-penalty bound into rough alignment with the polynomial's actual ~10–13 m/s² peaks under the bench's 15 m/s² clamp.

Dropping the soft-bound to 15 conflates two different quantities (the rough segment-boundary estimator vs the instantaneous polynomial-accel vs the bench's saturator). The L-BFGS now penalizes segment configurations where the *averaged* estimator merely exceeds 15, even though the *polynomial* peak — and hence what the bench actually executes — is still well under saturation. Result: segment times stretch unnecessarily; race time drifts up; tracking error can even worsen because trajectories now spend more time at boundary-condition curvature mismatches (consistent with the aigp_default observation in M2).

Recommended re-think: if the goal is "L-BFGS bound = bench saturation", the right fix is to replace the L-BFGS estimator with a peak-polynomial-accel measure (which `_topp_retime` already computes via `a_centripetal` and `a_longitudinal`) and keep the bound at the bench's 15 m/s². Just lowering the bound without correcting the estimator inherits the iter-005 problem in the opposite direction.

### M4. "Single source of truth" is aspirational — three sites still inline the same numbers

Despite the import unification, three independent code paths still hardcode the bench envelope:

- `scripts/benchmark.py:486-489`: `max_accel=15.0`, `max_speed=15.0`, `drag=0.5`, `yaw_rate_max=4.0`.
- `planning/trajectory_optimizer.py:308-310` (`compute_ilc_offset_table`): `max_accel=15.0`, `max_speed=15.0`, `drag=0.5`.
- `control/mpc_tracker.py:80`: `TrackerConfig.max_thrust_n=20.0`.

The commit body acknowledges these as "deferred to iter-011+", but the *claim* (commit subject: "single source of truth") is therefore inaccurate. The change minimises the cross-module lie, it does not eliminate it. Worse, the new `drone_spec.py` constants give the *appearance* of authority — a future maintainer who edits `DEFAULT_MAX_ACCEL_MPS2 = 12.0` will reasonably assume the bench will respect it; the bench will silently continue at 15.0 and produce a new mismatch.

The Opus plan also called for a `tests/test_drone_spec_provenance.py` grep-negative regression test that fails if any other module hardcodes a constant present in `drone_spec.py`. This test was NOT added, so there is currently no automated barrier to the lie growing back.

---

## MINOR

- **m1 (naming/semantic)**: `competition/drone_spec.py` documents the synthetic-bench drone (per its own docstring: *"this is the SYNTHETIC BENCH'S drone, not the AIGP competition drone"*) but lives next to `aigp_geometry.py` (which IS spec-derived AIGP truth). A future maintainer will reasonably mis-read this. Move to `core/drone_spec.py` or rename `synthetic_drone_spec.py`.
- **m2 (frozen-vs-mutable)**: `DroneSpec(frozen=True)` but `DroneConstraints` is not. `dc = DroneConstraints(); dc.max_acceleration = 99.0` silently works and breaks the SSOT claim at instance level. Freeze `DroneConstraints` or document why one is frozen and the other isn't.
- **m3 (dead import)**: `drone_spec.py:34` imports `field` from `dataclasses` but never uses it.
- **m4 (silent default change)**: `DroneConstraints(max_velocity=...)` is constructed in 9+ call sites (`race_pipeline.py:256`, `racing_line.py:418`, `visual_demo.py:395`, `smoke_test.py:125,231`, `helix_offset_search.py:83`, `benchmark.py:131,369,860`). Each now silently picks up 15.0 instead of 20.0 for max_acceleration. The behavior change is real on every code path; the commit body's "verification" doesn't decompose which call site contributed which delta.
- **m5 (missing regression pin)**: No test asserts `DroneSpec.max_acceleration_mps2 == 15.0` or `DroneConstraints().max_acceleration == 15.0`. Matrix gate is the only protection.
- **m6 (third accel limit in tree)**: `flight_control/types.py:24` defines a per-axis `max_acceleration = (5.0, 5.0, 3.0)` for a separate MPC config. Not touched by iter-010. If reachable from any production path, the tree has THREE acceleration-limit sources (DroneSpec, DroneConstraints, flight_control/MPCConfig). Worth a follow-up audit.
- **m7 (commit narrative overstates)**: "matches scripts/benchmark.py:486's saturation clamp" is true at the numeric level, but the L-BFGS bound and the bench clamp are *different physical quantities* (M3). The change unifies *labels*, not *semantics*.

— Opus 4.7 (~1450 words)
