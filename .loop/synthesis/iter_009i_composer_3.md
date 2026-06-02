# Adversarial review: iter-009i F9 fix (b926734)

## Scope

Commit `b926734` decouples racing-line **geometry selection** from **execution** `max_velocity` by introducing `RacingLineConfig.select_velocity_mps` (wired to `TrajectoryOptimizer` inside `_select_by_sim`) while treating `max_velocity_mps` as caller metadata. `scripts/benchmark.py` passes auto-derived `max_velocity` into the former field and locks `select_velocity_mps=15.0`. Cache keys now split on `select_velocity_mps` only.

## Is path–velocity decoupling principled here?

**Partially, with an important caveat.**

What the change actually decouples is not a full path–velocity decomposition in the control-theoretic sense (spatial path \( \gamma(s) \) then time law \( s(t) \) under true dynamics). It decouples **two different uses of a scalar cap** that were accidentally fused: (1) the min-snap segment-time scaling inside `TrajectoryOptimizer` when the BO oracle builds candidate trajectories for scoring, and (2) the velocity cap used later when the benchmark generates the flown reference. Those two caps need not be identical for *geometric* gate-offset selection; tying them made the multi-start landscape velocity-sensitive and produced basin switching at lower auto-derived speeds (F9).

That is a legitimate *engineering* decoupling: fix a **reference parameterization** for the inner oracle so its local minima stay in a basin that empirically works, then solve timing/feasibility downstream at the real cap. It aligns with hierarchical ideas (coarse planner at a nominal speed, refine later)—but the nominal speed is chosen for **stability of the discrete candidate pool**, not from first principles of curvature-limited flight on the final path.

**Caveat:** The docstring cites Heilmeier / Kapania–style “geometry first, velocity profile second.” In this codebase, “second” is not a rigorous speed-on-path solve that proves the chosen offsets remain optimal (or even safe) under the execution envelope; it is “run the existing pipeline with a different `DroneConstraints.max_velocity`.” If a line is marginally aggressive under the 15 m/s scorer but the real profile is much slower, you usually gain margin—but the oracle’s **tracking error and race-time proxies** are still computed from trajectories timed at `select_velocity_mps`, not at execution speed. So the multi-objective score is **not** a consistent preview of the run you will actually fly; it is a **stabilized** preview. Principled for regression closure; not a certificate of velocity-constrained optimality.

## Tech debt: `max_velocity_mps` as “informational”

**Yes—meaningful debt, mostly naming and coupling hygiene.**

1. **Semantic trap.** `max_velocity_mps` reads like the knob that bounds the optimizer the class is named for. After this commit, it does nothing inside `RacingLineOptimizer` except sit in config for callers to read. Anyone importing `RacingLineConfig` without reading the long docstring can easily reintroduce F9 by “fixing” the obvious bug of wiring execution speed into the wrong field—or assume cache entries are keyed by execution speed when they are not.

2. **Split sources of truth.** Benchmark hardcodes `select_velocity_mps=15.0` while defaults also say 15.0. Drift happens when one changes and the other does not. A single named constant (module-level `RACING_LINE_SELECTION_V_REF`) would reduce that class of debt.

3. **Kinematic oracle still hard-coded.** `_kinematic_eval` clamps `max_speed = 15.0` regardless of `select_velocity_mps` or `max_velocity_mps`. Today defaults align at 15, so the inconsistency is latent. If `select_velocity_mps` ever diverges from 15, the scorer’s trajectory generator and the eval clamp disagree—exactly the class of “oracle skew” the research notes flagged as secondary risk.

4. **Cache semantics.** Omitting execution `max_velocity` from the cache key is correct *given* the new contract (geometry invariant to execution cap). It also encodes the contract in persistent JSON: you cannot infer “this offset was validated at \(v_\mathrm{exec}\)” from the key alone. That is fine if the contract holds; it is brittle if a future change reintroduces execution-speed-dependent feasibility into selection without revisiting caching.

## Verdict

The fix is **sound as a targeted stabilization** of a velocity-coupled BO oracle and is backed by a clear regression test (`max_velocity_mps` sweeps do not move geometry). It is **not** a fully principled minimum-time gate pass in the coupled sense; it trades physical co-scoring for a fixed reference basin. Keeping `max_velocity_mps` as informational **does create interface and documentation debt** unless follow-ups rename (`execution_max_velocity_mps` / `selection_reference_velocity_mps`), centralize constants, and optionally thread `select_velocity_mps` into `_kinematic_eval` so the inner sim cannot silently desynchronize.
