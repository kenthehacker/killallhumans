# Sim Stack Unification — Opus 4.7 Recommendation (iter-026)

## TL;DR — top recommendation

**Switch the PyBullet backend in both `benchmark --mode sim` AND `scripts/visual_demo.py` from `GPDDrone` (CF2X, 27 g / 0.6 N) to `sim_pybullet/drone.QuadrotorDrone` (1 kg / 20 N) — which already matches the kinematic matrix bench *and* `TrackerConfig` defaults. Freeze the existing values in `competition/drone_spec.py` as "AIGP proxy v1" with explicit provenance. Keep the kinematic matrix as a fast oracle (do NOT replace with PyBullet). PyBullet `QuadrotorDrone` becomes the truth gate.**

This is the **smallest** intervention that closes the matrix-vs-demo divergence: the kinematic bench and `QuadrotorDrone` are *already* the same 1 kg / 20 N envelope (iter-021 preserved this), so swapping `GPDDrone → QuadrotorDrone` deletes the impedance mismatch in one move. CF2X stays available behind an explicit `backend="cf2x"` flag for legacy comparison.

### Why this beats sibling proposals

- **gpt-55-xhigh** (`sim_stack_unification_gpt55.md`): concur on direction; **disagree on changing the numbers** to 0.8 kg / 24 N / arm 0.14 m in this iter. Today matrix + `QuadrotorDrone` + `TrackerConfig` all coincide at 1 / 20; changing the envelope invalidates iter-038 gains, iter-009i racing-line basin, race_01 ILC schedule, and the auto-velocity ceiling. Land structure first; rebaseline numbers when SITL data lands.
- **composer-25** (`sim_stack_unification_composer.md`): correct that physics numbers shouldn't be conflated across plant profiles, but underweights the present problem (visual_demo silently flies the wrong-scale plant with no parity gate). Adopt composer's Phase A (thread `target_acc` into bench) AS PART OF the swap, not as a substitute.
- **Whole-stack PyBullet matrix**: kinematic is ~100× faster (5 s/track vs 30+ s). The AI iteration loop runs `benchmark_matrix.py` constantly. Trade speed for fidelity at the *truth* tier, not the *smoke* tier.

---

## Q1 — visual_demo backend swap

**Yes — swap to `QuadrotorDrone`, keep PyBullet for rendering.** The API gap:

| Concern | Current (`GPDDrone`) | After swap (`QuadrotorDrone`) |
|---|---|---|
| Step API | `step(target_pos, vel, yaw, target_acc)` | `apply_command(throttle, roll, pitch, yaw_rate)` |
| Inner loop | DSLPIDControl (CF2X-tuned, 48 Hz) | Attitude PD `kp=12, kd=4` on multibody |
| Camera | `get_camera_image`, `project_points_to_fpv` | Already implemented (`drone.py:394–485`) |
| Tilt cap | 35° (in DSLPIDControl) | 0.35 rad in `DroneConfig` (iter-021 conservative) |

The bridge: a thin **`QuadrotorDrone.step_reference(target_pos, target_vel, target_yaw, target_acc=(0,0,0))`** method (~25 lines) that wraps the reference in a `TrajectoryPoint`, runs `GeometricTracker(TrackerConfig()).track(...)` (the same tracker the matrix bench uses), and maps `AttitudeCommand` → normalized `apply_command(thrust, roll/max_tilt, pitch/max_tilt, yaw_rate_norm)`.

Net effect: `visual_demo`, `benchmark --mode sim`, and the live MAVLink path use the **same tracker → command chain**. Only the *plant* differs. That is the unification we actually need.

---

## Q2 — should the matrix ALSO use PyBullet?

**No.** Keep kinematic as a fast oracle:

- 100× speed advantage matters for the AI-driven iteration loop in `CLAUDE.md`.
- Kinematic checks *autonomy logic* (planner produces a legal trajectory, sequencer credits gates, EKF converges) — rigid-body fidelity not required for that.
- **Two-tier model**: `kinematic_oracle` (smoke) + `pybullet_aigp_quad` (truth). `--strict` treats PyBullet failure as binding. Kinematic-passes-but-PyBullet-fails is a real bug; PyBullet-passes-but-kinematic-fails is acceptable (kinematic is overly pessimistic).

Rename the `synthetic_sim` JSON key → `kinematic_oracle` so consumers can't mistake it for race truth.

---

## Q3 — freeze drone_spec values?

**Yes — freeze the CURRENT values with widened scope, not new guessed values.**

`drone_spec.py` today already declares 1 kg / 20 N / 15 m/s² / 15 m/s with provenance. Three minimal changes:

1. **Re-frame the docstring**: "synthetic-bench drone envelope" → "**AIGP proxy v1** — chassis dimensions spec-backed (VADR-TS-002 §3.6); mass/thrust/drag inferred, pending SITL calibration via `competition/calibration.py`". Numbers unchanged; contract widens from "kinematic-bench only" to "kinematic-bench + PyBullet `QuadrotorDrone` + `TrackerConfig`".
2. **Add chassis dimensions** (spec-derived, not inferred): `DEFAULT_BODY_SIZE_M = (0.28, 0.28, 0.16)` and `DEFAULT_ARM_LENGTH_M = 0.14`, citing `aigp_geometry.AIGP_DRONE_*` for provenance.
3. **Route `sim_pybullet/drone.py:DroneConfig` defaults through `drone_spec`** (mass, max_thrust, gravity, arm_length, body_size) via `dataclasses.field(default_factory=...)`, mirroring iter-013's `TrackerConfig` pattern. Keep attitude PD gains and tilt limits inline — those are plant-local tuning, not envelope properties (iter-021 split is correct on that axis).

**Resist gpt55's 0.8 kg / 24 N proposal in iter-026.** Cost of changing the number is re-tuning everything baked at 1 / 20. Land the *contract* first; rebaseline the *envelope* as a follow-up once SITL telemetry exists.

---

## Q4 — regression-test story

1. **Extend `test_drone_spec_contract.py`** — assert `DroneConfig().mass_kg == DEFAULT_MASS_KG`, `max_thrust_n == DEFAULT_MAX_THRUST_N`, `arm_length_m == DEFAULT_ARM_LENGTH_M`, `body_size == DEFAULT_BODY_SIZE_M`. Pins `QuadrotorDrone` to SSOT.

2. **NEW `test_sim_stack_parity.py`** — the binding gate. Run `run_sim_benchmark(race_01, duration=20)` and `VisualDemo(race_01, no_render=True, max_time=20).run()`. Assert: `drone_backend == "aigp_quad"` for both, `gates_passed_bench == gates_passed_demo`, both `>= 12`, `|Δavg_track_err| / max(...) < 0.15`. **Would have caught the present 12/12 vs 4/12 split.**

3. **NEW `test_backend_selection.py`** — smoke `DroneRaceEnv(backend="aigp_quad"|"cf2x")`. Assert `aigp_quad` is default; CF2X is opt-in.

4. **CI guard in `benchmark.py`** — make `overall_passed = (unit AND synthetic AND sim)` when PyBullet available. Today `overall_passed=True` can co-occur with PyBullet skipped — that's exactly how 12/12 synthetic + 4/12 demo slipped through. Default `--strict` in CI.

5. **`CLAUDE.md`** — iteration loop runs `--mode full --strict`; promote PyBullet to truth gate.

---

## Q5 — concrete code-change list (in commit order)

1. **`competition/drone_spec.py`** — add `DEFAULT_BODY_SIZE_M`, `DEFAULT_ARM_LENGTH_M`, `DEFAULT_LINEAR_DAMPING=0.3`, `DEFAULT_ANGULAR_DAMPING=0.8` (`drone.py:197–202` values); rename docstring to "AIGP proxy v1".
2. **`sim_pybullet/drone.py:DroneConfig`** — source `mass_kg`, `max_thrust_n`, `gravity`, `arm_length_m`, `body_size` from `drone_spec` via `default_factory`. Keep attitude PD gains and tilt limits inline.
3. **`sim_pybullet/drone.py:QuadrotorDrone`** — add `step_reference(...)`, `get_sim_time()`, `step_count`. Lazily construct `self._tracker = GeometricTracker(TrackerConfig())`.
4. **`sim_pybullet/env.py`** — add `backend: Literal["aigp_quad","cf2x"] = "aigp_quad"` kwarg. Branch: `aigp_quad` → `p.connect` + `loadURDF("plane.urdf")` + `QuadrotorDrone`; `cf2x` → `GPDDrone` (legacy). Gate creation already client-agnostic.
5. **`scripts/visual_demo.py`** — drop the `TrackerConfig(mass=0.027, max_thrust_n=0.6)` override (line 417–420) + the CF2X `PLAN_MAX_SPEED=4.0`/`cmd_max_speed_mps=4.0` ceiling. Replace `env.drone.step(...)` (line 661–664) with `env.drone.step_reference(...)`.
6. **`scripts/benchmark.py:run_sim_benchmark`** — change `env.drone.step(...)` (line 1055) to `env.drone.step_reference(..., target_acc=ref.acceleration)` (composer's Phase A). Add `"drone_backend"`/`"drone_spec"` keys to the result dict.
7. **`sim_pybullet/configs/race_01.json`** — drop `planner.plan_max_speed_mps=4.0` + `planner.cmd_max_speed_mps=4.0` (CF2X residue). Re-run `benchmark_matrix.py`; if 12/12 holds, leave ILC alone.
8. **`competition/pybullet_adapter.py`** — audit `send_attitude` (line 187–195) post-unification; backend dual-support already there.
9. **Tests** — extend `test_drone_spec_contract.py`; add `test_sim_stack_parity.py`, `test_backend_selection.py`.
10. **`CLAUDE.md`** — promote PyBullet to truth tier; update aspirational metrics to PyBullet baseline.

Steps 1–4 land the SSOT; 5–6 swap consumers; 7 drops CF2X residue; 8 audits adapter; 9 locks contract; 10 documents. Each step is independently revertable.

---

## Q6 — risks + mitigation

| Risk | Sev | Mitigation |
|---|---|---|
| **race_01 12/12 regresses** when CF2X 4 m/s ceiling drops + new plant takes over. | HIGH | EXPECTED. Current 12/12 / 13.7 s passed on a permissive kinematic bench; new baseline is the *honest* number. Snapshot `regression_baseline_post_iter_026.json`. Re-tune ILC + auto-velocity if needed. |
| **PyBullet rendering / camera dependencies** survive? | LOW | `QuadrotorDrone` already implements `get_camera_image`, `project_points_to_fpv`, `get_spectator_image` (`drone.py:394–485`). HUD only needs `get_state` + image methods. |
| **gym-pybullet-drones uninstall** breaks CI? | LOW | DON'T uninstall. `backend="cf2x"` stays for legacy. Dependency remains in `requirements.txt`. CI runs `aigp_quad` by default. |
| **`QuadrotorDrone` PD poorly tuned** (max_tilt=0.35 ≠ TrackerConfig 0.85). | MED | iter-021 picked conservative cap on purpose. Run one `max_tilt=0.85` sweep; if stable, route through `drone_spec`; otherwise document the saturation. |
| **1 kg / 20 N still wrong vs real AIGP** (likely 0.6–0.9 kg / 30–40 N). | MED-LONG | Land structural unification first. Schedule `competition/aigp_dynamics.py` re-baseline as iter-027+ once SITL calibration exists. "v1" naming keeps future swap honest. |
| **Parity test flakiness** from PyBullet non-determinism. | LOW | Seed `np.random.seed(42)` in both paths. ±15% tracking-error tolerance; *strict* `gates_passed` equality (discrete; jitter doesn't reach it on a healthy stack). |
| **Composer's worry**: forcing one mass/thrust across kinematic + GPD. | N/A | Not doing this. CF2X envelope untouched. Only `aigp_quad` is unified with the kinematic bench — already coincident at 1 / 20. |
| **Matrix vs PyBullet metric divergence** is unsettling. | LOW | DON'T tune matrix to match PyBullet. Two fidelity layers = two valid signals. CI fails only on PyBullet failure. |
| **`PipelineConfig.max_speed=8.0`** (race_pipeline.py:95) hardcoded; conflicts with new auto-derived velocities. | MED | Out of scope for iter-026 (MAVLink path). Follow-up; add cross-reference comment. |
| **visual_demo ILC + speed profile assumed CF2X.** | MED | race_01 ILC tuned at v=15 originally (iter-009 fractional format), then reapplied at v=4 for CF2X. Reverting to ~v=10–15 should reactivate original tuning. If not, re-run `helix_offset_search.py`. |

---

## Bottom line

The cleanest path **removes the most code** and **adds the most parity**. Promoting `QuadrotorDrone` to the default PyBullet backend reuses what's already in the repo *at the right scale*, makes `visual_demo` and `benchmark --mode sim` run the **same code path** end-to-end, freezes `drone_spec.py` as the SSOT with an honest "AIGP proxy v1" label, and keeps the kinematic matrix as the fast oracle it should be. CF2X stays opt-in for legacy; SITL-calibrated AIGP dynamics become a follow-up swap (iter-027+), not a rewrite blocking iter-026.

`tests/test_sim_stack_parity.py` is the single most important deliverable: it converts "matrix passes / demo crashes" from invisible drift into a visible CI failure. Ship it first, watch it fail on `main`, then unwind with the changes above until it goes green.
