# Adversarial review: iter-010 (`f1505ee`) — `aigp_default` tracking error 0.205 → 0.233 m

## Verdict (hostile)

The change is **architecturally correct** (one envelope for bench, auto-velocity, and `DroneConstraints`), but the **observed rise in mean tracking error is not, by itself, proof** of either “physical-accuracy improvement” or “actionable regression.” It is consistent with **(a)** removing a planner–oracle inconsistency that silently warped *which* trajectory and *how* time was allocated, **and** **(b)** a **metric regression** on this placeholder track under fixed tracker gains and stochastic EKF noise. Both can be true: a more honest model can move a scalar surrogate the wrong way for one course.

## What actually changed

`DroneConstraints.max_acceleration` default **20 → 15** (`planning/trajectory_optimizer.py`), sourced from `competition/drone_spec.py` to match `scripts/benchmark.py`’s kinematic clamp (`max_accel = 15.0`). The commit narrative: polynomials time-allocated under 20 m/s² demanded accelerations the bench then saturated at 15 → feedforward / timing stress → inflated error.

That causal chain is **directionally plausible** but **not sufficient** to predict the sign of **mean closest-point tracking error** after the change. Tracking error here is dominated by **integrated motion vs. spatial reference**, not by “named mismatch removed therefore number drops.”

## Behavioral risk: three mechanisms the commit under-weights

**1. Selection oracle was internally inconsistent pre-010**

`RacingLineOptimizer._select_by_sim` builds `TrajectoryOptimizer(constraints=DroneConstraints(max_velocity=select_velocity_mps))` and never overrides `max_acceleration`. Until iter-010, those candidate trajectories used **20 m/s²** in the optimizer, but `_kinematic_eval` always defaulted **`max_accel_mps2=15.0`**. So the BO selector **ranked lines using a physics model that did not match the generator’s constraint set**. Aligning defaults changes **both** the candidate trajectories **and** the relative ranking in the normalized COP score — not a pure “execution got easier” knob.

**2. “Bench clamps so error spikes” oversimplifies the metric**

The synthetic loop applies `GeometricTracker` then clamps **realized** acceleration (`scripts/benchmark.py`). Reference **position/velocity** still come from the polynomial. Tightening the optimizer’s accel budget **lengthens segment times** and reshapes snap; mean error can rise if the drone spends **more simulated time** in hard-to-track regions, if gate timing stresses the sequencer differently, or if **EKF odometry noise** (`np.random.normal` on position/velocity) integrates over more steps at similar dt.

**3. `aigp_default` is a weak discriminator**

It is a **placeholder** geometry with looser matrix semantics than production tracks. A ~**14%** relative bump (0.205 → 0.233 m) on a single config, single seed class, is **noisy** and far below the matrix’s **0.40 m** tracking gate — good for CI safety, weak for claiming “physical truth beat regression.”

## Is this “physical accuracy” or “regression”?

| Interpretation | Supports | Contradicts / caveats |
|----------------|----------|------------------------|
| **Physical-accuracy / honesty win** | Removes documented 15 vs 20 cross-module lie; optimizer and bench now share the same cap; `race_01` unchanged in the commit report — suggests no broad meltdown. | Honesty does **not** imply this scalar **must** improve; it can worsen if the old inconsistency accidentally selected a line that scored well under the buggy oracle. |
| **Regression on `aigp_default` surrogate** | Mean error rose; any gate-timing or clearance margin change is unproven from the headline number alone. | Still passes gates/thresholds per commit; may be acceptable product trade if competition stack differs anyway (`drone_spec` explicitly disclaims AIGP airframe). |

**Bottom line:** treat the delta as **unresolved** until decomposed. Do **not** let “still under 0.40 m” substitute for answering *why* the mean moved.

## How to verify (concrete, falsifiable)

1. **Saturation duty cycle** — Log per-step `‖a_des‖` before clamp and indicator `‖a‖ ≥ max_accel − ε` for parent vs `f1505ee` on `aigp_default`. If post-010 saturation **drops materially** while mean error rises, the old “clamp inflates error” story is incomplete and **controller / geometry / dwell-time** effects dominate.

2. **Trajectory invariants** — Dump `trajectory.total_time`, peak `‖a(t)‖`, peak `‖v(t)‖`, and per-gate mean error from `per_gate_errors`. Check whether **time-in-turn** increased. If total_time rose ~proportionally to error, suspect **longer exposure** not “worse controller.”

3. **Isolate planner vs selector** — **A:** Freeze racing-line offsets from pre-010 cache, recompute **only** min-snap timing with `max_acceleration ∈ {15,20}`. **B:** Freeze offsets, vary nothing but random seed **N=30** runs; confidence interval on mean error. (A) separates time-allocation from basin change; (B) quantifies EKF noise.

4. **Oracle consistency check** — Temporarily pass `max_accel_mps2=traj.constraints.max_acceleration` into `_kinematic_eval` at call sites (even if redundant post-010) and add a unit test that fails if they diverge again. Prevents silent reintroduction of the **20-gen / 15-score** split.

5. **Higher-fidelity shadow** — If available, replay the same reference in PyBullet with recorded thrust/accel limits; kinematic bench remains a **proxy**, not ground truth for “VQ1 physical accuracy.”

## Closing stab

The commit message’s “trade-off is now physical-accurate” is **marketing-grade**. The defensible claim is: **we removed a known envelope inconsistency; matrix stayed green; one placeholder metric moved against us and needs attribution, not dismissal.** Ship the constants consolidation — but gate **iter-011** on evidence from (1)–(3), not on the comfort of sub-threshold noise on `aigp_default`.
