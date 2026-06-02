# iter-009i (b926734) — Composer-2 adversarial review

**Scope:** F9 path–velocity decoupling (`select_velocity_mps` vs informational `max_velocity_mps`), `scripts/benchmark.py` synthetic wiring, and `tests/test_racing_line_velocity_invariance.py`. Goal: issues a typical first-pass “Composer-1” style review tends to miss (import placement, comment typos, obvious coupling).

## 1. Tests document behavior they do not enforce

`test_select_velocity_DOES_change_geometry` is named like a **positive behavioral claim** (“select velocity changes geometry”). The body only checks tensor shape and finiteness. That is not a weaker assertion—it is a **different contract**: the test cannot fail when the decoupling knob is ineffective on the toy course. Automated readers (coverage dashboards, future refactors, `pytest -k` selectors) will treat the name as a guarantee that is not actually encoded.

Rename to something non-causal (`test_racing_line_runs_for_distinct_select_velocities`) or split: (a) parametrized smoke, (b) optional strict branch behind an explicit golden fixture when offsets diverge.

## 2. “Bit-identical” narrative vs `assert_array_almost_equal`

The commit message and module-level test prose emphasize **identical** geometry across execution velocities. The assertion uses `decimal=6`. That is fine engineering (L-BFGS + float noise), but it contradicts the marketing language. A reviewer auditing “did we really lock bit-identical basins?” can answer **no**—we locked ~1e-6 agreement in offset space. Tighten prose everywhere, or switch to an exact integer test on quantized offsets if the pipeline allows.

## 3. `_offsets` is not what its docstring claims

The helper says it returns signed lateral offsets in the gate’s lateral direction. It returns **raw world-frame deltas** `(wx−cx, wy−cy, wz−cz)`. For the synthetic gates (normal along +X) this happens to align with intuition, but it is **not** a projection onto the gate opening’s lateral axis. If a regression ever perturbs along-track components, this “geometry identity” could still pass while true lateral clearance changed. A gate-frame projection (or comparing only the two coordinates spanning the opening plane) would match the stated invariant.

## 4. Falsification gap: toy layout vs. the failing track class

F9 was a **specific course + bench stack** interaction (auto-derived low execution speed, BO basin at gate 1). The regression test uses a hand-built four-gate zigzag with parallel normals. It proves **“under these seeds and this optimizer, varying `max_velocity_mps` does not move the returned waypoint deltas at 1e-6.”** It does not prove **“aigp_default can no longer pick the sharp basin.”** That gap matters for epistemic honesty: the fix is architecturally right, the test is a convenient proxy—not a minimized repro of the incident.

A stronger (still unit-level) addition: load the smallest frozen gate list that once reproduced basin flip, or assert monotonicity of a scalar summary statistic across velocities on `race_01` centers without spinning PyBullet.

## 5. Cache path is untested for the new key semantics

Both tests force `use_cache=False`. The behavioral change explicitly moves **`select_velocity_mps` into `_compute_cache_key`** and removes execution speed from the key. There is no test that:

- two runs with identical gates, same `select_velocity`, different `max_velocity`, `use_cache=True` **hit the same cache entry**; or  
- changing `select_velocity_mps` **misses** the old cache.

Without that, a future edit could silently reintroduce `max_velocity_mps` into the key (or drop `select_velocity_mps`) and unit tests would stay green.

## 6. Asymmetric benchmark wiring (synthetic vs PyBullet)

`run_synthetic_benchmark` now constructs `RacingLineConfig(max_velocity_mps=…, select_velocity_mps=15.0)` explicitly. `run_pybullet_benchmark` still builds `racing_line_cfg` purely from `racing_line_overrides` defaults merge. Today `select_velocity_mps` defaults to 15.0, so behavior matches **by accident**. If the default ever shifts, only the synthetic path is self-documenting. One line in a bench-level test asserting both entry points resolve to the same effective pair would lock intent.

## 7. Velocity sampling asymmetry

The invariance sweep uses `(5, 8, 12, 15)` m/s. It never probes **`max_velocity_mps > select_velocity_mps`** (e.g. 18 vs selector 15). The informational field is allowed to exceed the selection reference; nothing in the test says the implementation must remain stable in that half-space. Unlikely to bite soon, but it is an uncovered quadrant of the config space.

## 8. Split-brain kinematic oracle (deeper than “wire max_v through”)

`_select_by_sim` builds trajectories with `DroneConstraints(max_velocity=self.config.select_velocity_mps)`. `_kinematic_eval` still hard-codes `max_speed = 15.0` for the PD follower. When `select_velocity_mps` is lowered in an experiment, the reference trajectory slows but the evaluator’s vehicle can still be clamped as if it were a 15 m/s platform—**metric distortion**, not basin flip. The new tests never exercise `select_velocity_mps ≠ 15` with quantitative expectations, so this incoherence can grow unnoticed.

## 9. Naming debt: `max_velocity_mps` as “informational”

The field name still reads like a control parameter. Downstream humans will keep wiring it into optimizers “because the name says max velocity.” The docstring mitigates that; **tests do not**. Consider a follow-up rename (`execution_velocity_hint_mps`) or a runtime assert in debug builds when `max_velocity_mps` is read inside selection paths. Composer-1 often stops at “comments are long”; the hazard is **semantic API drift over months**.

---

**Bottom line:** The production fix in `racing_line.py` is coherent and matches the swarm diagnosis. The test file is a good first guard but optimizes for a narrow signal (numeric equality on a toy) while **mislabeling** one test and **skipping** cache, kinematic-oracle coherence, and real-track falsification—areas a surface-level review rarely prioritizes.
