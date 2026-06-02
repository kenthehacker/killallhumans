# Next iteration — Composer survey (branch `aigp-vq1-loop`, HEAD iter-009l)

## Repo snapshot (2026-05-24)

- **Branch / delta**: `aigp-vq1-loop`, ~36 commits ahead of `origin/main`. Recent arc: iter-006..009 — auto-velocity, plan validator, matrix regression gate (≥6/7), iter-009i–l **F9 closed** (path–velocity decoupling for racing-line *selection*; cache version / dedupe).
- **Matrix**: `tests/test_benchmark_matrix.py` locks **6/7** synthetic tracks with tracking `<0.40 m` and per-track `sim_time` ceilings; **`figure8` is explicitly excluded** — coplanar gates 1/5, plan validator flags illegal bare trajectory (`out_of_order:gate-5`), not a “slow down” problem.
- **Deferred from earlier loop state**: vision UDP receiver landed in iter-001; **A12 `mavlink_bridge` wiring** still the main integration gap. MLP residual **ships off-by-default**; multi-track train + enable was always deferred.

---

## Candidate options (impact / effort / risk)

Scores are qualitative (1–5): **impact** on VQ1 readiness & honesty, **effort** (engineering time), **risk** (regressions / unknowns). Higher impact and lower effort+risk ⇒ better ROI.

| Candidate | Impact | Effort | Risk | ROI |
|-----------|--------|--------|------|-----|
| **A. Dynamics envelope audit — unify 15 vs 20 m/s² (and thread through planner / bench / kinematic oracle)** | **4** — removes a **documented cross-module lie**: `planning/auto_velocity.py` states centripetal / safe-speed uses **15** because the synthetic bench clamps at **15**, while `DroneConstraints.max_acceleration` defaults to **20** in `planning/trajectory_optimizer.py`. Trajectories can be time-allocated under **2g** assumptions the bench never executes, inflating feedforward mismatch and gate-timing stress. Aligning defaults closes that gap before any further velocity or MLP work. | **2** — mostly constants + constructor call sites + tests; optional small `drone_limits` module. Cache invalidation already has precedent (iter-009l). | **3** — will shift segment times and possibly racing-line pool scores; **must** re-run `benchmark_matrix` and watch race_01 `sim_time` gate (22.5s). Not expected to magically fix figure8 (topology). | **High** |
| B. Tracker MLP — multi-track train + flip `use_residual` | 3 | 4 | **5** — charter already flagged race_01 overfitting; MLP without strict multi-track holdout can **undo** iter-006–009 generalisation. | Low–Med |
| C. Vision pipeline → `mavlink_bridge` wiring (A12) | **5** for real **competition** I/O; **2** for current **PyBullet matrix** (bench does not exercise JPEG/UDP path). | 3 | 2 | Med — do soon for submission, not the best *next* step if the goal is matrix / planner honesty. |
| D. SITL calibration validation | 4 for hardware fidelity | **5** — needs SITL stack, sensors, repeatable harness; reviews consistently **out of scope** until sim honest. | 3 | Low short-term ROI |
| E. Figure8 coplanar gates (SFC / corridor / TOGT-style gate solids) | **5** — only path to honest **7/7** | **5** — research-grade planner change | **4** — easy to break other tracks | Low ROI **next**; correct epic for a dedicated milestone |

---

## Pick ONE: **A — Dynamics envelope single source of truth (15 vs 20 m/s²)**

**Rationale:** F9 fixed *velocity* coupling between racing-line selection and execution. The **acceleration** mismatch is the same *class* of bug (planner assumes dynamics the evaluation stack does not have). It is already spelled out in-repo; fixing it is **smaller blast radius** than figure8 or MLP, **unblocks honest tuning** of ILC / feedforward / auto-velocity, and aligns with the charter’s “no phantom capacity” spirit without weakening gate-order rules.

---

## Concrete plan (iter-010 scope)

1. **Single authority**  
   - Introduce `planning/drone_limits.py` (or equivalent) exporting e.g. `DEFAULT_MAX_ACCEL_MPS2 = 15.0` and optionally `DEFAULT_MAX_VELOCITY_MPS = 15.0`, with a one-paragraph docstring citing: synthetic bench clamp in `scripts/benchmark.py`, centripetal derivation in `planning/auto_velocity.py`, and competition mass/thrust headroom (`TrackerConfig.max_thrust_n`) as separate concerns.

2. **Wire defaults**  
   - Set `DroneConstraints.max_acceleration` default from that constant (replacing **20.0**).  
   - Audit call sites that construct `DroneConstraints(...)` without explicit accel (e.g. `race_pipeline.py`, `visual_demo.py`, `benchmark.py` PyBullet path) so none reintroduce 20 by accident.  
   - Keep `max_thrust` / tilt limits unchanged — this iteration is **accel envelope only**, not a thrust rebalance.

3. **Kinematic oracle alignment**  
   - Confirm `racing_line.py` `_kinematic_eval` / `SpeedProfiler` paths use the same numeric source for `max_accel_mps2` as the bench (iter-009 already threaded velocity more carefully; accel should match).

4. **Caches & determinism**  
   - Bump racing-line cache key version if stored trajectories depend on accel (per iter-009l pattern) or force regen in dev; ensure CI does not silently reuse stale JSON.

5. **Verification (must all pass)**  
   - `python3 scripts/benchmark.py --mode unit`  
   - `pytest tests/test_benchmark_matrix.py tests/test_auto_velocity.py tests/test_racing_line_velocity_invariance.py`  
   - Full matrix: `python3 -m scripts.benchmark_matrix` (or project-standard duration). **Exit criteria:** ≥6/7 non–figure8 unchanged or improved; race_01 still satisfies iter-009e/009g gates (no new DQ / crash class). Expect modest **lap-time drift**; if race_01 slows past 22.5s, tune is *downstream* of honesty (document, do not paper over by reintroducing 20).

6. **Explicit non-goals (this iter)**  
   - No sequencer / DQ relaxation.  
   - No figure8 geometry edits.  
   - No MLP enable.  
   - Vision/SITL can be scheduled as parallel track once sim envelope is truthful.

---

## One-line executive summary

**Ship a single-source drone accel limit (15 m/s²) through `DroneConstraints`, the bench, and any kinematic scorers — closing the documented 15-vs-20 planner/bench gap with bounded effort before touching MLP, vision, or coplanar gate topology.**
