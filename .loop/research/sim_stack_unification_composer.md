# Sim stack unification — matrix bench vs visual_demo vs AIGP-class drone

**Audience:** orchestrator / future implementers  
**Scope:** `scripts/benchmark.py`, `scripts/visual_demo.py`, `sim_pybullet/drone.py` (`QuadrotorDrone`), `sim_pybullet/gpd_drone.py` (`GPDDrone`), `competition/drone_spec.py`, `race_pipeline.py`, plus `competition/aigp_geometry.py` for spec geometry.

---

## Executive summary

The **cleanest** unification is **not** collapsing mass/thrust/accel into one global constant set. The repo already encodes the right idea in comments (iter-021 `DroneConfig`, iter-023 `visual_demo` `TrackerConfig`): **one logical “autonomy stack”** (planner → references → sequencing → EKF) with **explicitly named plant profiles** for (1) fast deterministic CI, (2) CF2X PyBullet fidelity, and (3) future competition-calibrated dynamics. Unify **interfaces, config keys, and reference generation**; keep **physics parameters** profile-specific until SITL fills `competition`-level calibration.

---

## Current state (as of this review)

| Layer | Synthetic (`benchmark` kinematic) | PyBullet (`benchmark` sim + `visual_demo`) |
|--------|-----------------------------------|---------------------------------------------|
| **Plant** | No rigid body: PD-style accel/vel clamps, noise-injected odometry | `DroneRaceEnv` → **only** `GPDDrone` (gym-pybullet-drones CF2X, ~27 g) |
| **Low-level command** | `GeometricTracker` → synthetic integrator | `GPDDrone.step(pos, vel, yaw[, target_acc])` — **not** the same code path as synthetic |
| **Mass/thrust SSOT** | `competition.drone_spec` (1 kg, 20 N, 15 m/s² accel cap — *bench proxy*, documented) | Real mass from DSLPIDControl; `visual_demo` uses **different** `TrackerConfig(mass=0.027, max_thrust_n=0.6)` for HUD-only geometric tracker |
| **Trajectory** | Shared pattern: `RacingLineOptimizer` + `TrajectoryOptimizer` + `derive_safe_max_velocity` | Same; **but** PyBullet bench calls `step` **without** `target_acc`; `visual_demo` feeds trajectory **acceleration feedforward** into `target_acc` |
| **`QuadrotorDrone`** | Unused in default env | Legacy 1 kg / 20 N custom multibody; **deliberately** conservative attitude caps vs `drone_spec` / tracker — documented split |
| **`race_pipeline.py`** | N/A (orchestrates competition adapter path) | Same stack conceptually; `PipelineConfig` defaults (e.g. `max_speed=8`) are **policy** choices partly decoupled from bench |

**AIGP geometry** (`aigp_geometry.py`): chassis 280×280×160 mm and camera intrinsics are **spec-backed**; mass/thrust are **explicitly unspecified** — any “AIGP-class” dynamics numbers remain placeholders until calibration.

---

## What “unification” should mean

1. **Single contract for “what the autonomy layer needs from the sim”**  
   Position, velocity, yaw, IMU-ish rates, camera image, sim time, optional contact/gate hit — already converging; formalize as a small protocol (typed dict or `Protocol`) shared by `DroneRaceEnv`, future adapters, and tests.

2. **Single contract for “what the plant accepts”**  
   Today: `(target_pos, target_vel, target_yaw[, target_acc])`. **Unify** PyBullet bench with `visual_demo` by threading **`target_acc`** (and any future yaw-rate) through `run_sim_benchmark` so feedforward behavior matches the human demo and reduces synthetic↔PyBullet **control** skew.

3. **Named profiles, not merged constants**  
   Introduce something like `PlantProfile` / `SimulationProfile`:  
   - `KINEMATIC_BENCH` — reads `drone_spec` for accel/vel/drag; no PyBullet.  
   - `GPD_CF2X` — current `GPDDrone`; optional JSON overrides only for **camera FOV / timing**, not for pretending the CF2 is 1 kg.  
   - `AIGP_CALIBRATED` (future) — populated from SITL or vendor tables; may swap `GPDDrone` for a new URDF or restored `QuadrotorDrone` tuned to measured inertia/thrust curve.

4. **Keep `drone_spec` as “planner + synthetic tracker envelope”**  
   Rename in docs (not necessarily code) to **BenchDynamicsEnvelope** to avoid the false implication that it describes the CF2X or the final AIGP drone. When real numbers arrive, add **`competition/aigp_dynamics.py`** (or extend `calibration` hook mentioned in `drone_spec` docstring) and **migrate defaults** behind a version flag so CI can pin the old envelope for regression matrices.

5. **`race_pipeline` alignment**  
   Ensure `PipelineConfig` limits (`max_speed`, camera pitch, gate size) **read the same JSON keys** as `RaceConfig` / track files where applicable; avoid duplicate magic numbers beyond what is already partially fixed (iter-007 `max_velocity_mps`, iter-010 `drone_spec`).

---

## Implementation plan (ordered, minimal blast radius)

**Phase A — Behavioral parity (low risk)**  
- Add `target_acc` to `run_sim_benchmark` using the same finite-difference or analytic sample as `visual_demo` (from `TrajectoryPoint` acceleration fields).  
- Factor shared “trajectory sample + progress clock + gate-seek fallback” into a small module used by **both** `benchmark` sim and `visual_demo` to stop silent divergence.  
- Normalize `SequencerConfig` construction: one helper that merges `race_config.sequencer_overrides` + track defaults so `visual_demo` vs `benchmark` cannot drift on `proximity_pass_distance` / `pass_through_margin`.

**Phase B — Structural clarity (medium risk)**  
- Add `sim_pybullet/plant_profile.py` (or `competition/sim_profiles.py`) with frozen dataclasses: `KinematicBenchParams` (wraps `DroneSpec`), `GPDPlantParams` (ctrl/pybullet freq, tilt limits), `AIGPGeometryRef` (import from `aigp_geometry` only).  
- `DroneRaceEnv.__init__` accepts optional `plant: Literal["gpd"] | ...` for forward extension; default remains GPD.  
- Document `QuadrotorDrone` as **optional dev harness** or schedule deprecation once AIGP plant exists.

**Phase C — Competition truth (high risk, blocked on data)**  
- When mass/thrust/inertia are known: add `AIGPDynamicsSpec`, wire `TrackerConfig` defaults for **MAVLink / pipeline** from it, keep **matrix** `KINEMATIC_BENCH` frozen unless intentionally rebaselined.  
- Optionally retune `DroneConstraints` max velocity/accel from measured capability instead of conservative proxies.

---

## Risks

| Risk | Why it hurts | Mitigation |
|------|----------------|------------|
| **Forcing one mass/thrust** across kinematic + GPD | Breaks either CI thresholds (synthetic tuned for 1 kg / 15 m/s²) or CF2 stability (27 g, 48 Hz inner loop) | Profiles + frozen regression baselines; never alias `DEFAULT_MASS_KG` to CF2 mass |
| **`target_acc` in bench changes metrics** | Gate timing and tracking error distributions shift | Treat as **bugfix parity** with demo; snapshot new `state/regression_baseline_*.json` after review |
| **gym-pybullet-drones fragility** | Headless CI, version pins, DSL tuning | Keep kinematic mode mandatory for “no GPU / no GPD” agents; strict mode already exists (`--strict`) |
| **Over-unifying `DroneConfig` with `drone_spec`** | Iter-021 note: attitude saturations differ on purpose | Only share **mass, gravity, max thrust** if a profile explicitly declares compatibility; keep inner-loop PD caps plant-local |
| **Pipeline vs bench speed mismatch** | `PipelineConfig.max_speed` vs track `plan_max_speed_mps` caused historical drift | Single JSON schema section `planner` / `cmd` already started — extend lint or unit test “config self-consistency” |
| **AIGP spec vs dev tracks** | `race_01` gate sizes ≠ VADR default | Keep track-level overrides; `aigp_geometry` remains spec-only constants |

---

## Bottom line

Unify **data flow and naming**, not **physics numbers**. Short term: **shared trajectory + feedforward + sequencer wiring** between `benchmark` PyBullet and `visual_demo`. Medium term: **`PlantProfile` + clearer `drone_spec` semantics**. Long term: **`AIGPDynamicsSpec`** from measurement, with the kinematic matrix kept as a **stable, fast oracle** for autonomy logic — not as a literal model of the 280 mm airframe.
