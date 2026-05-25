# iter-009 F9 regression — research note (Composer)

## What broke

`planning/auto_velocity.py::derive_safe_max_velocity` lowers `DroneConstraints.max_velocity` on tight courses (centripetal limit). Wiring that same value into `RacingLineOptimizer` so `_select_by_sim` scores candidates with `TrajectoryOptimizer(..., max_velocity= v_auto)` changed which lateral-offset basin won for `aigp_default`. The full `scripts/benchmark.py` pipeline then crashed at gate 1. The bench was reverted at `ef344b1`: `RacingLineOptimizer()` stays on legacy defaults (including `max_velocity_mps=15.0` for the BO oracle), while trajectory generation still uses `max_velocity` from JSON override or auto-derive.

`sim_pybullet/configs/aigp_default.json` has no `max_velocity_mps`, so the executor uses auto-derived speed (~5–6 m/s class for this geometry), but line selection remains tuned as if the oracle ran at 15 m/s.

## Mechanism (code-grounded)

1. **Multi-start L-BFGS** produces several local minima in offset space (`all_results` in `optimize()`).
2. **`_select_by_sim`** is the “BO” oracle: for each candidate it builds a min-snap trajectory with `TrajectoryOptimizer(constraints=DroneConstraints(max_velocity=self.config.max_velocity_mps))`, then scores **(avg tracking error, worst gate error, race time)** with COP-style min–max normalization across the pool and weights `_W_AVG_ERR`, `_W_WORST`, `_W_TIME`.
3. **Min-snap trajectories are velocity-coupled**: changing `max_velocity` alters segment times and shape of feasible accelerations, so the same offset vector is not the same space–time curve. The composite score’s relative ranking across candidates shifts; a line that is “good enough” at 15 m/s can lose to another basin at 6 m/s.
4. **Oracle inconsistency (independent of the revert):** `_kinematic_eval` still clamps the lightweight tracker at `max_speed = 15.0` regardless of `RacingLineConfig.max_velocity_mps`. So when the trajectory is generated with a low cap, the selection sim can still allow the virtual vehicle to track faster than the planned reference envelope implies. That skews error and time proxies relative to the real bench, where `max_velocity` is aligned. Fixing this is necessary for a velocity-aligned scorer but **not sufficient** to remove basin switching.

## Published approaches robust to velocity changes (top 2)

### 1) Path–velocity decomposition (spatial path, then 1-D speed on arc length)

**Idea:** Choose a **purely geometric** reference (positions in the corridor, optionally heading) using objectives that do not depend on the eventual `v_max`—typically integrated curvature, path length, or clearance—then compute a **velocity profile** along arc length subject to \(v^2 \kappa \le a_{\mathrm{lat}}\) and actuator limits (classical **minimum-time along a fixed path**, Bobrow–Dubowsky–Gibson-style time scaling, and standard industrial manipulator trajectory generation).

**Why it is robust:** Changing the global speed cap mostly rescales the time parameterization (until a curvature bind is hit); the **lateral offsets** do not need to be re-optimized when `v_max` changes unless you deliberately co-design path and time for joint time optimality.

**Relation to F9:** Your racing-line stage optimizes geometry but the oracle mixes in **race_time from a velocity-dependent trajectory**, which reintroduces the coupling the decomposition avoids.

### 2) Corridor-constrained geometric racing line (elastic band / quadratic curvature penalty on a lattice or spline)

**Idea:** In autonomous racing literature and practice (elastic bands, Voronoi field + smoothing, “minimum curvature” splines through gates), the spatial path is the solution of a **geometric** variational problem in a free-space tube. A separate **time-parameterization** or MPC layer handles speed.

**Why it is robust:** The spatial argmin moves slowly with `v_max` because `v_max` is not in the geometric cost; it enters only when assigning \(ds/dt\).

**Relation to F9:** Your L-BFGS objective already mixes `speed_weight * path_length` with curvature (`racing_line.py` proxy stage). The instability comes from the **second, sim-based stage** re-ranking candidates under a different dynamical scaling than production.

---

## Concrete code change (proposal only — not implemented)

**Primary (architectural, matches literature):** **Decouple racing-line selection from execution speed.**

- Keep `derive_safe_max_velocity` for `TrajectoryOptimizer` in `benchmark.py` (and PyBullet) as today.
- For `RacingLineOptimizer._select_by_sim`, **freeze the oracle velocity used for ranking** to a constant that matches how `planning/racing_line_cache.json` was produced (e.g. 15.0), *or* explicitly use a **geometry-only** score for basin choice: L-BFGS proxy cost, integrated curvature along the polyline through offset gate frames, plus a gate-margin penalty—**drop `race_time` from selection** or compute time only after fixing path shape.

- Alternatively (same family): **two-pass** — optimize offsets with `max_velocity_mps = V_ref` fixed; after selection, call `TrajectoryOptimizer` once with true `max_velocity` for the run. Document that `V_ref` is a “selection reference speed,” not the flight cap.

**Secondary (correctness bugfix, still proposal):** Thread `self.config.max_velocity_mps` into `_kinematic_eval` (replace hardcoded `max_speed = 15.0`) so the kinematic oracle’s velocity clamp matches the trajectory generator. Pass the value as an argument from `_select_by_sim` to avoid static-method signature churn. This aligns the inner sim with the outer bench when you eventually re-enable velocity-aware selection.

**Cache hygiene:** If any velocity-aware selection ships, extend `_compute_cache_key` to include `max_velocity_mps` (and possibly a discretized bucket) so stale JSON lines cannot load offsets optimized under a different scorer.

## Risk

| Change | Risk |
|--------|------|
| Geometry-only selection | May pick lines that are suboptimal for **minimum lap time** at high `v_max` on easy tracks; need A/B on `race_01` and matrix. |
| Fixed `V_ref` oracle + true `v_max` execution | Mismatch remains by design; could under-cut corners relative to a joint TOGT-style plan (Qin 2024 class methods) when the drone is actually fast. |
| `_kinematic_eval` clamp fix alone | Low risk, improves honesty of the scorer, but **does not** fix multi-modality; basin ranks can still flip. |
| Cache key + velocity | Invalidates `racing_line_cache.json`; deterministic replays change until caches regenerate. |

## Bottom line

F9 is not mysterious: **a sim-oracle BO stage that ranks geometry using velocity-coupled trajectories is expected to change basins when `max_velocity` changes.** The literature-standard mitigation is **path–velocity decomposition** (plus corridor geometry), optionally layered with a **minimum-curvature / elastic** spatial solve. The smallest honest patch is scorer velocity consistency in `_kinematic_eval`; the robust product fix is **stop using execution `v_max` inside racing-line basin selection**, or score with a fixed reference `V_ref` and accept the semantic split until a true joint planner lands.
