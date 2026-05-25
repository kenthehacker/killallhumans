# Adversarial review: iter-010 (f1505ee) — test coverage gaps

**Scope:** Commit `f1505ee` introduces `competition/drone_spec.py` and rewires `planning/trajectory_optimizer.py::DroneConstraints` defaults to import from that module (notably `max_acceleration` 20→15). This review asks whether *tests* back the “single source of truth” narrative.

## Question 1: Is there a test that no other module redefines drone constants?

**Answer: No.** Nothing in `tests/`, `**/tests/`, or `scripts/smoke_test.py` performs a repo-wide invariant such as “only `drone_spec` may define `DEFAULT_MAX_ACCEL_MPS2`” or “benchmark saturation literals must equal `drone_spec`.”

**Evidence:**

- A repository search for `drone_spec` in `*.py` hits **only** `planning/trajectory_optimizer.py` (import + comments). No test file imports or references `competition.drone_spec`.
- Large literals remain elsewhere by design (iter-010 explicitly deferred wiring `DroneSpec` into `scripts/benchmark.py`) and by omission: `planning/auto_velocity.py` still defines `DEFAULT_DRONE_MAX_ACCEL = 15.0` and `DEFAULT_DRONE_MAX_SPEED_MPS = 15.0` independently; `planning/racing_line.py` still hard-codes `15.0` in multiple defaults; `planning/trajectory_optimizer.py` itself still uses **inline** `max_accel = 15.0` / `max_speed = 15.0` inside the ILC helper (~L308–309), parallel to `DEFAULT_LINEAR_DRAG_PER_MASS`-style duplication rather than `drone_spec`.
- `scripts/helix_offset_search.py` clamps with raw `15.0` literals.

So iter-010 improves *one* default path (`DroneConstraints` field defaults) but does not establish a test-enforced boundary against “another module redefines the same physics ceiling.” Matrix / integration tests may *observe* aggregate tracking degradation if literals diverge, but they do not localize the failure to “constant shadowing” nor fail fast on import.

**Adversarial angle:** The commit message closes charter item “no cross-module magic numbers,” yet the codebase still contains multiple authoritative-looking `15.0` sources. Without a guard test, the regression class iter-010 fixed (planner assumes 20, bench delivers 15) can reappear if someone reverts `trajectory_optimizer` imports or duplicates a new `20.0` in another planner path.

**Gap closure ideas (not implemented here):** (a) `ast`/grep-based test that fails if `max_accel = <float>` appears in `benchmark.py` with a value ≠ `drone_spec.DEFAULT_MAX_ACCEL_MPS2`; (b) a small allowlist file of modules permitted to mention numeric literals, everything else must import from `drone_spec`; (c) property test: `DroneConstraints()` defaults == `DroneSpec()` fields for overlapping semantics.

---

## Question 2: Is there a test that `DroneConstraints.max_acceleration` cannot drift from `drone_spec.DEFAULT_MAX_ACCEL_MPS2`?

**Answer: No.** `planning/tests/test_trajectory.py` imports `DroneConstraints` for optimizer scenarios but **does not** assert default field values against `DEFAULT_MAX_ACCEL_MPS2` (or against `dataclasses.fields(DroneConstraints)` defaults resolving to the imported aliases).

**What would catch drift:**

- `assert DroneConstraints().max_acceleration == DEFAULT_MAX_ACCEL_MPS2`
- or `inspect.signature` / `fields()` comparison to the imported `_DRONE_MAX_ACCEL` binding

None exist. A developer could change `DroneConstraints.max_acceleration` to a literal `18.0` “for tuning” while leaving `drone_spec` at `15.0`; unit tests focused on polynomial math would likely still pass until a slower / threshold-sensitive benchmark run.

**Secondary smell:** `planning/auto_velocity.py` header comments still claim `DroneConstraints.max_acceleration` is `*20.0*` relative to the bench — stale relative to iter-010. No test fails on documentation drift.

---

## Summary verdict

| Invariant | Covered by tests? |
|-----------|-------------------|
| No duplicate / conflicting drone envelope literals across modules | **No** |
| `DroneConstraints.max_acceleration` default tracks `drone_spec.DEFAULT_MAX_ACCEL_MPS2` | **No** |

Iter-010 is a **reasonable architectural fix** with **weak test safety nets**. The highest-ROI follow-up is a tiny `competition/tests/test_drone_spec_contract.py` (or similar) that locks the dataclass default wiring and optionally cross-checks `scripts/benchmark.py`’s kinematic saturation constants against `drone_spec` once benchmark wiring lands (per iter-010 deferral notes).
