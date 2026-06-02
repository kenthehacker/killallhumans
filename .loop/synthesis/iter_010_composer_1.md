# Adversarial review — iter-010 (`f1505ee`)

**Scope:** `competition/drone_spec.py` (new), `planning/trajectory_optimizer.py` (`DroneConstraints` defaults + `max_acceleration` 20→15), research notes in `.loop/research/next_iter_*.md`.

**Verdict:** The core diagnosis (polynomial time allocation assumed 20 m/s² while the synthetic kinematic loop clamps to 15 m/s²) is sound and the default change is directionally correct. The implementation and messaging overclaim “single source of truth” and leave several consistency and maintenance hazards.

---

## 1. False “SSoT” — duplicates and stale prose remain

- **`scripts/benchmark.py`** still hardcodes `max_accel = 15.0`, `max_speed = 15.0`, drag, yaw rate, mass, etc. The commit message explicitly defers wiring; that is honest in the message but **contradicts** `trajectory_optimizer.py` comments claiming the bench, tracker, and `auto_velocity` “all see the same numbers.” They do not: only `DroneConstraints` defaults were rewired.
- **`planning/auto_velocity.py`** still defines `DEFAULT_DRONE_MAX_ACCEL: float = 15.0` locally and its module docstring still states that `DroneConstraints.max_acceleration` is **20.0** — that is now **wrong** and will mislead the next engineer or static audit.
- **`control/mpc_tracker.py`** `TrackerConfig` still uses inline defaults (e.g. `max_thrust_n: float = 20.0`); no import from `drone_spec`.

**Risk:** The repo now has *three* authority patterns (spec module, benchmark literals, planning literals) while prose asserts one.

---

## 2. Racing-line cache / BO oracle — silent skew risk

`RacingLineOptimizer` constructs `TrajectoryOptimizer(constraints=DroneConstraints(max_velocity=select_velocity_mps))`, so **`max_acceleration` follows the dataclass default**. Pre-iter-010 that meant **20** inside the BO scoring loop; post-commit it means **15**, i.e. candidate trajectories and composite scores change for the same gate layout.

The JSON cache (`planning/racing_line_cache.json`) is **keyed on geometry + `select_velocity_mps` + weights**, not on optimizer acceleration budget or cache “physics generation.” **`f1505ee` does not bump cache schema version nor appear to regenerate the committed cache** in the diff.

**Failure mode:** A cache hit reuses offsets that were **ranked under a 20 m/s² inner optimizer** while other paths now plan under **15 m/s²** — or CI/dev machines with cold cache diverge from machines with warm/old cache. Opus’s own pre-commit plan called for a cache bump; shipping without it weakens the “bounded risk / matrix catches it” story.

---

## 3. `DroneSpec` surface area vs actual use

`DroneSpec` includes `linear_drag_per_mass` and `yaw_rate_max_rad_s`, but **`DroneConstraints` does not consume them** (and nothing else imports `DroneSpec` in this commit). Unused `dataclasses.field` import in `drone_spec.py` is a small hygiene smell.

**Risk:** Future edits may assume `DroneSpec` is authoritative for drag/yaw while the bench still owns those literals — repeating the 15-vs-20 class of drift under a nicer type.

---

## 4. Architecture / packaging

`planning/trajectory_optimizer.py` now imports **`competition.drone_spec`**. That inverts a common layering expectation (core planner → competition adapter). It may be acceptable in this monolith but is brittle if packaging, minimal installs, or future “planning-only” tests ever exclude `competition/`.

Composer’s own survey suggested `planning/drone_limits.py`; putting bench-empirical caps next to the optimizer avoids the dependency direction issue.

---

## 5. Documentation precision

- **`DEFAULT_MAX_ACCEL_MPS2` docstring** labels it “Maximum **lateral** acceleration”; the benchmark clamps **vector** acceleration magnitude (`accel_mag > max_accel`). Wording should match implementation to avoid wrong centripetal reasoning later.
- **Line-number citations** (`benchmark.py:486`, etc.) will rot on the next edit; prefer symbol names or grep anchors.

---

## 6. What was done well

- Clear provenance narrative in `drone_spec.py` (bench vs placeholder vs competition reality).
- `frozen=True` on `DroneSpec` is appropriate for a value object.
- Aligning **default** `max_acceleration` with the binding sim clamp fixes a real feedforward / time-allocation inconsistency for all call sites that omit explicit accel.

---

## Recommended follow-ups (not blocking merge, but blocking “SSoT” claims)

1. Update **`auto_velocity.py`** comments to reference `drone_spec` (or import constants) and remove the obsolete “20.0” claim.
2. Either **bump `racing_line` cache version** and regenerate `racing_line_cache.json`, or add **`max_acceleration` (or a schema hash)** into `_compute_cache_key`.
3. Tone down **`trajectory_optimizer.py`** banner comment to “optimizer defaults sourced from `drone_spec`; bench/tracker still duplicated until iter-011.”
4. Remove unused **`field`** import; add a **tiny test** that `DroneConstraints().max_acceleration == DEFAULT_MAX_ACCEL_MPS2` to prevent silent regression.
