# iter-009 F9 — Velocity-Sensitive Racing-Line Basin Switching (Opus 4.7)

## Diagnosis

`RacingLineOptimizer._select_by_sim` is a velocity-coupled, **safety-blind**
scalarizer. Three things conspire to make low-velocity scoring pick a crash-
prone line on `aigp_default`:

1. **Trajectory shape is velocity-coupled.** `TrajectoryOptimizer.optimize`
   is min-snap with time-allocation tied to `DroneConstraints.max_velocity`.
   At v=15 m/s the segment times for aigp_default are short → polynomial
   "tightens" along the chord. At v≈6 m/s (auto-derived for that geometry)
   segment times triple → the polynomial gains lateral freedom and bulges
   further from gate centers, so the same offsets produce a different
   space-time curve. Lines that were marginal at v=15 become "good"
   geometrically at v=6 and vice-versa.

2. **Scorer's race_time term collapses.** `_kinematic_eval` runs a fixed-
   gain PD tracker against the new trajectory. At v=6, the controller
   trivially tracks every candidate (`tracking_errors` small for all 10
   starts + 3 interpolants), so `norm_avg` and `norm_worst` lose
   discriminating power. The `_W_TIME=0.3` term then dominates → the BO
   tie-breaks toward the shortest **path**, i.e. the most aggressive
   corner-cutter (cached value `[0.6, -0.6, 0.6, -0.6, ...]` is already at
   `max_lateral_offset`). At v=15 the tracker's overshoot under high
   commanded accel penalizes those same extreme offsets, so a milder line
   wins.

3. **No safety/clearance term.** The scorer never checks gate-strut
   clearance or replays the sequencer. A line that exits gate-1 at
   x≈0.36 m off-center has 0.24 m to the strut (gate half-width 0.6 m).
   At v=6 the polynomial entry into gate-2 bends back hard, and the
   first-segment exit from gate-1 *under tracker overshoot* clips the
   strut. The BO is happy — tracking error to the polynomial is small;
   the *polynomial itself* is the unsafe object. This is the
   gate-1 crash signature.

Cache is keyed by gate geometry + `(max_lateral_offset, smoothness_weight,
speed_weight, corner_cut_aggressiveness)` — explicitly **not** by
`max_velocity_mps`. So a velocity-aware bench wire-up doesn't even
invalidate the prior cache, it just decides which basin gets persisted on
the *next* cold start.

## Mechanism summary (one line)

The scorer is a **scalarized, single-fidelity, velocity-coupled, safety-
blind** picker over a multi-modal landscape. Of those four adjectives, the
binding ones for F9 are *velocity-coupled* (changes the ranking) and
*safety-blind* (lets the new ranking pick a crash).

## Top 2 research-backed techniques

### T1 (primary). Heilmeier 2019 minimum-curvature QP — velocity-agnostic line selection by construction

**Citation:** Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp, Lohmann,
*"Minimum curvature trajectory planning and control for an autonomous race
car,"* Vehicle System Dynamics 58(10) 1497–1527, 2020 (TUM/Roborace).
Extended by Kapania, Subosits, Gerdes 2016 *"A sequential two-step
algorithm for fast generation of vehicle racing trajectories."*

Replace the multi-start L-BFGS + sim-oracle stack with the canonical
F1/Roborace formulation. Each gate contributes one lateral offset
α_i ∈ [-1, +1] within the gate opening (our `_apply_offsets` parameter).
Heilmeier's key trick: the path can be written as a piecewise-linear
function of {α_i}, so curvature κ_i at the i-th waypoint is a *quadratic*
function of {α_{i-1}, α_i, α_{i+1}}, giving the convex QP

    minimize  α^T (P_curv + ε · P_len) α + q^T α      s.t. -1 ≤ α_i ≤ 1

where P_curv encodes ∑κ_i² (banded, PSD), P_len encodes path length
(positive), and ε ∈ [0, 1] hybridizes minimum-curvature (slow corners) with
shortest-path (fast straights). Heilmeier proves this is **convex** — one
unique global minimum, no L-BFGS multi-start, no basin selection problem.

**Why this is velocity-agnostic.** Heilmeier 2019 §3.2 shows the minimum-
curvature line maximizes the centripetal-limited speed profile
v(s) = √(a_lat_max / κ(s)) everywhere on the track. The optimal line is
defined *purely* by the geometric centripetal constraint — exactly the same
constraint `planning/auto_velocity.py:derive_safe_max_velocity` uses.
Solving Heilmeier and using the auto-derived `v_max` *both* fall out of
the same physics; they cannot disagree.

**Why this kills the F9 regression.** A unique global minimum cannot
"switch basin" when a velocity parameter is tweaked, because there are no
basins to switch between. The cache becomes velocity-trivially safe (one
entry per gate geometry; valid for any `v_max`).

**Cost:** S/M. ~150 lines of Python replacing `_select_by_sim`. The
P_curv matrix has the Bristow & Alleyne 2007 banded structure that scipy's
`scipy.sparse.linalg.spsolve` or `osqp` handles in <10 ms for 12 gates.

### T2 (defensive). Add a feasibility filter before scoring — independent of T1

Even if we keep the sim oracle, fix the *safety blindness* first. Before
`_W_AVG_ERR · norm_avg + ...` scoring, replay `GateSequencer` against the
trajectory samples (this is exactly `planning/plan_validator.py`) and
discard any candidate whose plan crashes, DQs, or has gate clearance
<0.15 m. Citation: Krinner et al. *"MPCC++: MPC with Recursive Feasibility
Guarantees,"* RSS 2024 — terminal-set feasibility filter before cost
minimization. T-MPC (de Groot, T-RO 2024) Theorem 2 — *guaranteed
fallback* if no candidate satisfies the safety constraint.

Concretely: the L-BFGS candidate that wins at v=6 on aigp_default would be
*rejected* by the plan validator (because it crashes at gate-1), forcing
the picker to keep a safe (if slower) line. This is a one-week patch with
no algorithm rewrite.

## Concrete code change

**File:** `planning/racing_line.py`. **Recommended path:** ship T2 in
iter-010 (safety filter, 1-day patch), land T1 in iter-011 (Heilmeier QP,
1-week swap-out). Both end with the bench safely using
`RacingLineConfig(max_velocity_mps=max_velocity)` and racing_line_cache.json
being velocity-invariant.

T2 patch (iter-010):
1. In `_select_by_sim`, between lines 396 and 405, after `trajectory = traj_opt.optimize(...)`, call `validate_trajectory(trajectory, gate_specs, dt=0.02)` from `planning.plan_validator` and skip the candidate if `not pv.ok or pv.crashed or pv.disqualified`. Pass `gate_specs` down through `_select_by_sim` (currently only `gates: List[GateWaypoint]` — add a converter or take both).
2. If all candidates fail, fall back to the L-BFGS-objective minimum (T-MPC Theorem 2 fallback). Today this already happens via the `if not valid: ...` branch — extend it to cover the feasibility-filter case.
3. Update `_compute_cache_key` to include `round(config.max_velocity_mps, 1)` so the cache fragments across velocity buckets. Bump cache `version` to 2; existing cache file invalidates cleanly.

**Tests:**
- `python3 scripts/benchmark.py --mode synthetic --config sim_pybullet/configs/aigp_default.json` must pass at the auto-derived velocity (~6 m/s).
- `python3 scripts/benchmark_matrix.py` must still hit 6/7 sim_pass.
- New unit test in `tests/test_racing_line_velocity_invariance.py`: build the aigp_default gate layout, call `RacingLineOptimizer(RacingLineConfig(max_velocity_mps=v, use_cache=False)).optimize(...)` for v ∈ {5, 8, 12, 15}, assert all four resulting offset vectors are *feasible* (plan_validator OK) — not necessarily identical, just safe.
- Re-enable the deferred wire-up in `scripts/benchmark.py:357` once green.

## Risks

- **T2 alone may over-prune.** On slalom-class tracks with tight margins, all 13 candidates could fail the plan validator at v=6, forcing the L-BFGS-objective fallback (which is exactly the iter-005 magic-number behaviour). Mitigation: log the fallback count in the bench output and add it as a matrix-gate metric.
- **T1 changes race_01's basin.** race_01's hand-tuned ILC section schedule (and the 22.5 s sim_time ceiling from commit `aa5aea1`) are tied to the *current* basin. Heilmeier may pick a structurally different line — likely faster, but the ILC schedule, sequencer margins, and per-gate ceilings would need re-tuning. Mitigation: gate T1 behind a `racing_line_overrides.use_min_curvature: true` flag and migrate one track at a time.
- **Convex ≠ best.** Heilmeier minimum-curvature is *optimal under the centripetal constraint only* — it ignores jerk, FOV, perception. For VQ1 (perfect state, no FOV) this is fine; for VQ2+ (camera-based perception) we'll want the TOGT (Qin 2024) gate-polygon constraint layered on top. T1 doesn't preclude that.
