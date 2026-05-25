# Next-Iteration Survey — Opus 4.7 (HEAD `00cb1d4`, branch `aigp-vq1-loop`, ~36 ahead)

## Snapshot (re-verified just now via `benchmark_matrix --duration 25`)

6/7 production passing (race_01 12/12 in 17.2s, 0.089m avg tracking; aigp_default placeholder 6/6; grand_tour 14/14; slalom 8/8; straight_hairpin 6/6; vertical_cliff 4/4). **figure8 1/8, crash_gate:gate-5** — fixture is *physically degenerate*: gate-1 (5,0,2.0) and gate-5 (5,0,2.2) are stacked on the same vertical line with same yaw, so any tangent trajectory through gate-1 grazes gate-5's plane.

---

## Candidate evaluation

| # | Candidate | Impact | Effort | Risk | ROI |
|-|-|-|-|-|-|
| 1 | Train + enable Tracker MLP residual | Med | M | **Med** | Med |
| 2 | Vision pipeline → `RacePipeline._step()` | Low (now) / High (post-DCL) | M | Low | Med (deferred) |
| 3 | SITL calibration end-to-end validation | Low (no DCL) | M | Low | Low (now) |
| 4 | **Drone-spec / magic-number unification** | **High** | **S–M** | **Low** | **High** |
| 5 | figure8 SFC corridor / TOGT fix | Low | L | Med-High | Low |
| 6a | Synth-bench drone proxy → AIGP geometry | Med | S | Med | rolled into #4 |
| 6b | ML residual eval gate via existing matrix | Med | S | Low | precondition for #1 |

### 1. Tracker residual MLP — train + flip `use_residual=True`
- **Impact (Med)**: race_01 already at 0.089m avg / ~0.30m max tracking. Headroom is real but bounded.
- **Effort (M)**: rollout collector + feedback-error-learning regressor + per-track holdout + npz writer (~2 days).
- **Risk (Med)**: hard ±0.05 rad / ±5% thrust clamp limits worst-case but not average-case regression. Decisive: training against a stack with **two coexisting acceleration limits (15 m/s² in bench vs 20 m/s² in optimizer)** means the residual learns to compensate for *that ghost*, not real plant mismatch. #4 is upstream.

### 2. Vision pipeline → RacePipeline integration
- `competition/vision_udp.py` ships and IS wired into `mavlink_bridge.py:51,96,114,143,198`. Real gap: **`race_pipeline.py` doesn't import either** — it consumes cv2-mocked or pybullet_adapter snapshots. There is no `RacePipeline + MAVLinkBridge` glue.
- No DCL binary on this worktree → wire-up is faith-based code.
- **Verdict**: queue for the iter immediately following DCL binary release.

### 3. SITL calibration end-to-end validation
- `calibration.py` is a least-squares fit with strong unit tests (`test_calibration.py` covers the iter-001b BLOCKER sign error). "End-to-end" without a DCL binary = building a synthetic UDP feeder with seeded thrust/drag, replaying it, confirming recovery — *almost identical* to the existing unit test, just over a socket. Marginal information gain.
- **Verdict**: defer until DCL connection.

### 4. Drone-spec / magic-number unification ⭐
- **The bug**: `planning/auto_velocity.py` documents `DEFAULT_DRONE_MAX_ACCEL = 15.0` because the synthetic bench saturates there (`scripts/benchmark.py:486`). **`planning/trajectory_optimizer.py:34` defaults `max_acceleration = 20.0`.** The optimizer time-allocates segments under a 2g budget the bench cannot execute. Whenever a polynomial requests >15 m/s² the bench clamps and tracking error spikes. iter-009 fixed the 15-vs-20 m/s velocity coupling for racing-line *selection*; the same class of bug is open on the *acceleration* axis.
- Other duplicates: `mass=1.0`, `max_thrust_n=20.0`, `gravity=9.81` hardcoded in `benchmark.py:159,463`; `drag=0.5`, `yaw_rate_max=4.0` in `benchmark.py:488,489`; `TrackerConfig.max_thrust_n=20.0` in `mpc_tracker.py:80`. **No module models the 280×280×160mm AIGP chassis** — charter item 5 (drone-sim mismatch) unaddressed.
- **Impact (High)**: closes charter items 2 (no magic numbers), 3 (architecture suspect), 5 (drone-sim mismatch acknowledged) in one shot.
- **Effort (S–M)**: ~1 new file + ~10 imports + ~5 test updates.
- **Risk (Low)**: matrix gate catches regressions on the 6 passing tracks. Lowering `max_acceleration` 20→15 will lengthen segment times slightly; race_01 has a 22.5s ceiling (iter-009g) which it currently clears at 17.2s — ~30% headroom.
- **Why #1**: documented cross-module lie; *precondition* for honest ML training, calibration, and planner tuning. `next_iter_composer.md` independently picked the same option.

### 5. figure8 coplanar-gates fix
- Real fix is SFC corridors / TOGT (iter-003 research swarm's unanimous pick). L effort, replaces the QP, weeks of regression hardening.
- Cheap fix: rewrite the fixture so gates 1/5 and 3/7 don't share xy. But the user has tagged figure8 "known-unsolvable" and it's excluded from the matrix gate. Competition won't have stacked gates.
- **Verdict**: dedicated milestone after spec/calibration/ML pillars are honest.

### 6. Other items I spot
- **6a — synthetic-bench drone proxy**: bench models a Crazyflie-class point mass (1.0 kg). AIGP drone is 280×280×160mm. Roll into #4 (consume `DroneSpec`).
- **6b — ML eval gate**: before flipping `use_residual=True`, add CI assertion that residual beats baseline on ≥4/6 production tracks AND doesn't regress race_01 by >10%. Cheap insurance against iter-005 overfit pattern. Folds into iter-011.
- **PyBullet vs synthetic divergence**: bench has TWO drone proxies (kinematic loop + PyBullet harness) with different mass/thrust assumptions. Unify under `DroneSpec` in #4.
- **MLP scaffolding**: `tests/test_tracker_residual.py:158-188` proves byte-identical baseline behaviour — solid. Activation gate is the missing piece.

---

## Recommendation: **iter-010 = Drone-Spec Unification + Magic-Number Audit**

### Concrete plan (one iter, ~1 day)

1. **Canonical authority** — `competition/drone_spec.py`:
   ```
   AIGP_DRONE_MASS_KG: float            # placeholder until calibration
   AIGP_DRONE_MAX_THRUST_N: float       # placeholder
   AIGP_DRONE_MAX_ACCEL_MPS2: float = 15.0   # binding bench clamp
   AIGP_DRONE_MAX_VELOCITY_MPS: float = 15.0 # matches saturation
   AIGP_DRONE_MAX_TILT_RAD: float = 0.85
   AIGP_DRONE_MAX_BODY_RATE_RAD_S: float = 6.0
   AIGP_DRONE_LINEAR_DRAG_PER_MASS: float = 0.5  # synth-bench proxy; flagged
   AIGP_DRONE_YAW_RATE_MAX_RAD_S: float = 4.0
   ```
   Each value carries a docstring tagging provenance: spec-derived (chassis from `aigp_geometry.py`), bench-empirical (max_accel), or placeholder pending calibration (mass, thrust). `DroneSpec` dataclass with `__post_init__` warning when AIGP-vs-PyBullet provenance is mixed.

2. **Rewire defaults** (no behaviour change on passing tracks):
   - `planning/trajectory_optimizer.py:33-34` → import from spec; **`max_acceleration` 20 → 15**.
   - `control/mpc_tracker.py:78-80` → mass/thrust/max_tilt/max_body_rate from spec.
   - `scripts/benchmark.py:159,463,486-489` → spec-driven; delete duplicates.
   - `planning/auto_velocity.py:44,58` → re-export from spec.
   - `planning/plan_validator.py:58` → ceiling from spec / aigp_geometry.

3. **Cache invalidation** (mirror iter-009l): bump `racing_line_cache.json` schema version since trajectory shapes change with `max_acceleration=15`. Force regen.

4. **Verification gates** (all must pass):
   - `pytest -q` whole tree green (expect ~5 numeric updates in `test_trajectory_optimizer.py`, `test_auto_velocity.py`).
   - `python -m scripts.benchmark_matrix --duration 25` → 6/7 still passing, race_01 sim_time ≤ 22.5s.
   - `python scripts/benchmark.py --mode unit` → all 9 unit tests green.
   - New `tests/test_drone_spec_provenance.py`: grep-negative-test asserting no module redefines a constant that lives in `drone_spec.py`.

5. **Non-goals (explicit)**: no SFC; no MLP enable; no figure8; no MAVLink wiring. Each is its own future iter.

### Why this and not the ML training the user emphasised

ML training is charter pillar #4 but its value is bounded by I/O consistency with the underlying plant. Training a residual on a stack with two coexisting acceleration ceilings (15 vs 20 m/s²) yields a model that compensates for the discrepancy itself — a ghost that disappears once #4 lands, leaving a worse tracker. **iter-010 = unify; iter-011 = train + matrix-gate the residual** with a clean `DroneSpec` as input feature scaling. That ordering protects the user's ML ask without paying twice.

— Opus 4.7
