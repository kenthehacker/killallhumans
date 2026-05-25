# Adversarial review: iter-010 (`f1505ee`) — DroneSpec partial adoption

## Verdict

iter-010 correctly fixes the **optimizer-vs-bench acceleration lie** (`DroneConstraints.max_acceleration` 20→15, sourced from `competition/drone_spec.py`). The behavioral change is real and measurable (commit notes a modest `aigp_default` tracking regression that remains inside matrix gates). The main weakness is **narrative overreach**: the codebase does **not** yet have a single mechanical source of truth across the stack—only the trajectory optimizer’s defaults were rewired. Deferred benchmark and tracker wiring is honest scoping, but it leaves **documentation debt** and a **false sense of closure** that iter-011 can accidentally amplify.

## Deferred items (benchmark.py / mpc_tracker.py)

**`scripts/benchmark.py`** still hardcodes `max_accel = 15.0` and `max_speed = 15.0` in the kinematic loop (~486–487). Values numerically match `DroneSpec` today, so runtime is consistent *by accident*. Risk: the next maintainer edits `DEFAULT_MAX_ACCEL_MPS2` assuming the bench follows, and the synthetic harness silently diverges until matrix drift shows up—or worse, unit tests that mock only one side pass while integration lies.

**`control/mpc_tracker.py` `TrackerConfig`** still duplicates `mass`, `gravity`, `max_thrust_n`, tilt/body-rate caps as literals. The geometric tracker’s thrust mapping and Lee-style attitude solution depend on `mass` and `max_thrust_n`; the bench comment in `drone_spec.py` already admits thrust is **not** what the kinematic loop saturates—so tracker vs bench is a second “envelope story.” iter-010 did not unify that story; it only aligned **lateral accel** for min-snap time allocation.

**Stale prose bug (high signal):** `planning/auto_velocity.py` still documents that `DroneConstraints.max_acceleration` is `*20.0*` relative to the bench’s 15—that is **post-iter-010 false**. Same class of issue: `planning/trajectory_optimizer.py` header comment claims “bench, optimizer, tracker, and auto_velocity all see the same numbers”—**the tracker does not import `drone_spec`**. These mismatches are exactly how the last ghost mismatch reproduced itself: comments became law while code forked.

## Sequencing risks (gate order / pass logic)

The gate sequencer (`gate_sequencing/sequencer.py`) reasons on **geometry and discrete tick segments** (plane crossings, opening vs outer frame, out-of-order DQ). It does not read acceleration limits directly.

**Indirect risk (non-zero, second-order):** Changing polynomial time allocation (15 m/s² budget vs 20) shifts segment durations and cornering speeds. That changes **when** the position trace crosses gate planes relative to the control timestep, which can affect edge cases such as **multi-gate-per-tick crediting** and strut-hit classification—paths that are already adversarially tested but tuned at the current dynamics. This is unlikely to be a primary regression vector compared to tracker/plant mismatch, but it is not theoretically zero.

**Pipeline coupling:** `race_pipeline.py` builds `TrajectoryOptimizer(constraints=DroneConstraints(max_velocity=self.config.max_speed))`, relying on defaults for acceleration—now 15 via imports. `GeometricTracker(TrackerConfig())` is still a **separate, non-parameterized envelope**. So “sequencing” at the autonomy level is really **time-parameterization vs execution-layer limits**; any future iter that raises `PipelineConfig.max_speed` without matching tracker thrust/tilt headroom could strand the drone short of gates (misses) without the sequencer being “wrong.”

## Could iter-011 break what iter-010 left half-done?

**Yes, in several plausible ways:**

1. **Refactor hazard:** Wiring `DroneSpec` into `TrackerConfig` often tempts constructors to take a `DroneSpec` instance. If `race_pipeline` continues to instantiate `TrackerConfig()` with literals while tests construct alternate specs, you get **dual construction paths** and flaky parity between PyBullet harness vs kinematic bench.

2. **Import / init cycles:** `drone_spec.py` is intentionally minimal today. If iter-011 pulls bench helpers into `drone_spec` for “provenance,” avoid importing `benchmark` from `drone_spec` (easy foot-gun).

3. **Constant migration without comment sync:** If iter-011 replaces literals but does not fix `auto_velocity.py` and the `trajectory_optimizer.py` banner comment, the project re-enters the **documented lie** state—harder to catch than a failing test because reviewers read comments first.

4. **ML residual interaction (commit already defers ML):** `TrackerConfig` includes optional learned residuals. If iter-011 changes default mass/thrust sourcing, a previously trained `.npz` could become **physically inconsistent** with the tracker’s nominal model unless training data is regenerated—silent quality loss, not necessarily a crash.

5. **“SSoT” strictness:** Freezing `DroneSpec` as `frozen=True` is good for immutability; iter-011 might need **profiles** (bench vs AIGP SITL). A naive single global object could force awkward `if env == …` at call sites and reintroduce forks.

## What iter-010 did well (for balance)

Centralizing numerics with provenance in `drone_spec.py` is the right artifact. Dropping the optimizer’s 20 m/s² default removes the worst feedforward mismatch class for min-snap. Deferring benchmark/tracker wiring is a reasonable cut line **if** iter-011 is scheduled immediately to (a) import shared constants, (b) delete duplicated literals, and (c) sweep stale comments that still mention 20 m/s² or claim full-stack unification.

## iter-011 acceptance criteria (minimal)

- `benchmark.py` kinematic clamps read from `competition.drone_spec` (or a tiny `bench_kinematics.py` re-export to keep `drone_spec` free of sim imports).
- `TrackerConfig` defaults sourced from the same constants; any `GeometricTracker` mass/thrust mismatch vs trajectory **documented** or explicitly parameterized in one call site (`race_pipeline`).
- Grep CI or pre-commit: fail if `20.0` reappears next to “max_acceleration” / “max_accel” without an approved exception token.
- Fix the stale `auto_velocity.py` and `trajectory_optimizer.py` comments in the **same** PR as the wiring—do not leave another half-truth iteration.

---

*Word count target: <1000; adversarial focus: deferred wiring, doc drift, iter-011 coupling risks, indirect sequencer sensitivity.*
