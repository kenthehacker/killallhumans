# F9 Research: Velocity-Sensitive Racing Line Selection

## Diagnosis

The F9 failure is real, but the root problem is not simply "pass the derived velocity into `RacingLineConfig`." The current racing-line optimizer is a multi-start L-BFGS proxy optimizer followed by a lightweight sim selector. That selector changes the objective landscape when `DroneConstraints.max_velocity` changes, so it can pick a different local basin even when the geometric course is unchanged. The attempted benchmark wire-up in `ef344b1` correctly identified this: scoring at the lower auto-derived speed selected a worse line for `aigp_default` and crashed near gate 1.

There are two concrete code-level hazards:

1. The cache is not velocity-aware. `_compute_cache_key()` includes lateral/smoothness/speed/corner config, but not `max_velocity_mps`. A run scored at 15.0 m/s, 12.28 m/s, or 6.0 m/s hashes to the same key. That makes cache behavior alias distinct dynamic regimes and can either hide or resurrect the F9 regression.

2. The selector is a fragile scalarization. It min-max normalizes avg error, worst-gate error, and time over a tiny candidate pool, then applies fixed weights. At lower speeds, time spread compresses and small/noisy sim-error differences dominate. Because candidate generation is still geometric L-BFGS, not a velocity-conditioned trajectory optimizer, the scorer can flip between near-equal basins without any robustness check for pass-through feasibility or crash margin.

The deeper mismatch: `auto_velocity.py` produces one global speed cap from worst triplet curvature, while real racing-line methods usually choose a line from geometry/curvature and then compute a local speed profile along that line. A global cap is useful as a safety ceiling, but using it as the optimization fidelity for line selection couples two decisions that should be only weakly coupled.

## Top Techniques

1. Decouple minimum-curvature line selection from velocity profiling.

Heilmeier et al., "Minimum curvature trajectory planning and control for an autonomous race car" (Vehicle System Dynamics, 2020), use a QP to compute a minimum-curvature raceline, then compute the velocity profile with a forward-backward solver under longitudinal/lateral acceleration limits. Jain and Morari, "Computing the racing line using Bayesian optimization" (CDC 2020), also note that minimum-curvature paths are close to minimum-time paths because lower curvature permits higher cornering speed for a fixed lateral acceleration limit.

For this repo, that argues for choosing offsets primarily by curvature, clearance, and path smoothness, then letting `TrajectoryOptimizer` and a local speed/ILC profile handle the speed. Do not let a single global `max_velocity_mps` flip the racing-line basin unless the line is invalid at that speed.

2. If keeping the sim scorer, make it robust multi-scenario / multi-objective, not single-fidelity scalar BO.

Multi-fidelity BO is designed for cases where cheap approximate objectives guide expensive target evaluations. Kandasamy et al., "Multi-Fidelity Bayesian Optimisation with Continuous Approximations" (ICML 2017), formalize choosing among fidelities while targeting the high-fidelity optimum. For multi-objective expensive evaluation, Knowles' ParEGO (IEEE TEC 2006) uses randomized scalarizations over a Pareto surface, and Daulton et al.'s qNEHVI (NeurIPS 2021) improves robustness with noisy objectives.

Here, "fidelity" can be scorer velocity / sim resolution: evaluate candidate offsets at `v_auto` and `v_legacy=15`, optionally with coarse `dt=0.02` first and final validation at benchmark dt. Select by feasibility first: no plan-validation crash/DQ, sufficient gate clearance, finite trajectory. Then rank by robust aggregate such as worst-case normalized tracking error plus median time, or Pareto rank with a deterministic tie-breaker by curvature. This directly targets the observed "good at high velocity, bad at low velocity" split.

## Concrete Code Change Suggestion

Do not flip `scripts/benchmark.py` directly to `RacingLineConfig(max_velocity_mps=max_velocity)` yet. First make racing-line selection velocity-safe:

- Add `max_velocity_mps`, selector version, and `_kinematic_eval` physics constants to `_compute_cache_key()`. Bump cache `version` to 2 and consider storing a map of `{cache_key: entry}` instead of one global entry, so race_01 and aigp_default do not overwrite each other.
- Add a candidate feasibility phase in `_select_by_sim()`: for each candidate, build trajectories at both `self.config.max_velocity_mps` and `15.0` (or `{auto, legacy}` supplied by config), run a cheap plan validator / gate-clearance check, and discard candidates that fail either scenario.
- Change scoring from current fixed min-max scalarization to robust constrained scoring: choose the feasible candidate minimizing `max(norm_avg_err_auto, norm_avg_err_legacy) + 0.5*max(norm_worst_auto, norm_worst_legacy) + 0.2*median(norm_time)`. Tie-break by the original curvature objective and then by zero/late-apex seed order for determinism.
- After that passes the matrix, wire synthetic benchmark as `RacingLineOptimizer(RacingLineConfig(max_velocity_mps=max_velocity))`; mirror the same config path in PyBullet/visual demo through `racing_line_overrides`.

The more principled follow-up is to split `RacingLineOptimizer` into `LineSelector` and `SpeedProfiler`: line offsets minimize curvature/clearance independent of global speed; local speeds are computed from curvature with `v(s)=sqrt(a_lat_max / kappa(s))`, capped by track/drone max, then smoothed with a forward-backward acceleration pass.

## Risk

Velocity-aware caching alone is necessary but not sufficient; it prevents aliasing but can faithfully cache a bad low-speed basin. Multi-scenario scoring adds runtime, although the candidate pool is small enough for coarse evals. Decoupling line and velocity is safer architecturally, but it changes optimization behavior across all tracks and needs matrix tests plus explicit `aigp_default` regression coverage for "same gates, different scorer velocities choose crash-free offsets."
