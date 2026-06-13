# Iter 001 — Synthesised Action Plan

Inputs:
- `.loop/iter_001_plan/opus-47-max-thinking.md` (10 actions, deepest test design)
- `.loop/iter_001_plan/gpt-55-xhigh.md` (11 actions, strong test-first DI emphasis)
- `.loop/iter_001_plan/composer-25.md` (14 actions, best file:line specificity)

## Convergence (all 3 agents agreed)
Across plans A1, A2, A3, A4 (where their numbering varies) the three agents
independently arrived at the same shape for these items:

| Concern | Resolution |
|---|---|
| I-1 sequencer in-order | New `RaceState.DISQUALIFIED` + scan unpassed future gates for plane crossings INSIDE the opening; gate behind `SequencerConfig.enforce_in_order: bool = True` (default True). |
| I-2 synthetic crash terminal | Synthetic bench calls the sequencer's existing P1-6 crash classification, then breaks on `seq.last_crash` (mirroring PyBullet path at `scripts/benchmark.py:746-750`). |
| I-3 magic numbers | Curvature-derived section partition with **global-ILC fallback** when curvature peaks < threshold; race_01 keeps the old per-step schedule via an explicit `ilc_section_overrides` block. |
| I-4 ILC hyper-params | Move `convergence_threshold`, `momentum_gamma`, `alpha`, `max_iterations` to `config/ilc_defaults.yaml`. |
| I-5 gate geometry | New `competition/aigp_geometry.py` constants module; `GateSpec`, `GateGeometry`, `GateWaypoint`, `PipelineConfig` all default from it. |
| I-6 drone mismatch | Stub `competition/calibration.py` with synthetic-data unit test only; real SITL validation deferred. |
| I-7 runner collapse | Composer's compromise: ship a **new** thin entry `scripts/run_pipeline_pybullet.py` that uses `RacePipeline` via the existing `PyBulletAdapter`, but do NOT touch `sim_pybullet/runner.py`. Collapse to iter ≥ 002. |
| I-8 camera shape | Defaults → 640×360, fx=fy=320, cx=320, cy=180, **+20° upward pitch** wired through `CameraIntrinsics.body_R_camera` and consumed by `gate_pose_to_drone_position`. |
| I-9 vision UDP | New `competition/vision_udp.py` async datagram receiver on :5600 with 24-byte LE header, frame-id reassembly, 100ms timeout, JPEG decode via `cv2.imdecode` on `asyncio.to_thread`. |
| I-10 ML pick | **DECISION: tracker residual** (see below). |
| I-11 tests-first | Adversarial test suite lands FIRST, fails on main, turns green as fixes land. |

## ML pick — RESOLVED: learned tracker residual (Opus's pick)
Three plans, three different ML options:
- Opus: tracker residual (small MLP, hard-clamped)
- GPT-5.5: dynamics ridge regression (sim-to-real from MAVLink telemetry)
- Composer: corner regressor (MobileNetV2-0.25 → PnP)

Pick **tracker residual** because:
1. **We can actually train it this iter.** The bench already emits a controller
   trace (`scripts/benchmark.py:561-568`); the trainer reads it.
2. **Dynamics regression needs MAVLink SITL telemetry** — we don't have the
   DCL binary in-hand. Stubbed via A12 (`competition/calibration.py`) for
   iter ≥ 002.
3. **Corner regressor needs PyBullet FPV imagery + matching AIGP visuals.**
   AIGP imagery won't match PyBullet's; the model would overfit. Defer until
   we have a DCL frame corpus.
4. **Safety story is provable.** Hard `±0.05 rad / ±0.05 thrust` clamp +
   `use_residual=False` default means a corrupted weight file cannot regress
   anything. Composer + GPT-5.5's picks have weaker safety stories.
5. Research backing: NGTC (Pries 2025), "Leveling the Playing Field"
   (Kunapuli 2025) both validate residual feedforward on geometric
   trackers; clamp pattern from "Safe-RL with hard projection" (Berkeley 2024).

## Incompatibilities — RESOLVED
1. **Geometry default + race_01 reproducibility**:
   - Defaults move to AIGP (1.5 m). race_01.json keeps its 1.2 m geometry
     via explicit `gate_defaults` override (already present at
     `sim_pybullet/configs/race_01.json:7`). NEW: ship
     `sim_pybullet/configs/aigp_default.json` as a 6-gate AIGP-geometry
     placeholder (positions scaled from race_01 by 1.5/1.2 = 1.25×) marked
     `"placeholder": true` until the real VQ1 course drops.
2. **ILC partition fallback**:
   - Curvature-peak partition with **single global ILC** as the fallback
     when curvature peaks below threshold. (Reject the equal-time-quartiles
     fallback — too clever, would still be "magic 4 sections".)
3. **Future-gate plane crossing OUTSIDE opening**:
   - Existing `miss` semantics preserved (harmless). Only an opening-inside
     crossing of a non-current gate triggers DQ.
4. **Runner collapse**:
   - Composer's parallel-entry scaffold wins. No edits to `runner.py` this
     iter. Add `scripts/run_pipeline_pybullet.py` + smoke test.
5. **MAVLink command-path tests** (GPT-5.5 only):
   - Adopt — small (S) effort, locks in both `SET_POSITION_TARGET_LOCAL_NED`
     and `SET_ATTITUDE_TARGET` paths against MAVSDK regression.

## Final action list (ordered, dependency-aware)
TDD discipline: tests for each subsystem land before that subsystem's fix.

| # | Action | Resolves | Effort | Test gate |
|---|---|---|---|---|
| **A1** | `tests/test_sequencer_adversarial.py` — out-of-order DQ, U-turn DQ, far-plane benign, AIGP geometry default | I-1, I-5, I-11 | S | itself |
| **A2** | `tests/test_benchmark_adversarial.py` — geometric crash terminates, ground-not-only-crash, no-magic-constants grep | I-2, I-3, I-11 | M | itself |
| **A3** | `tests/test_camera_geometry.py` — AIGP intrinsics defaults, +20° tilt projection sanity, PnP accuracy under tilt | I-8, I-11 | S | itself |
| **A4** | `competition/aigp_geometry.py` (new) — `AIGP_GATE_INTERIOR_M = 1.5`, `AIGP_GATE_BORDER_M = 0.6`, `AIGP_GATE_DEPTH_M = 0.26`, `AIGP_DRONE_FOOTPRINT_M = 0.28`, `AIGP_CAM_*` consts incl. `+20°` tilt | I-5, I-8 | S | A1/4, A3/8 |
| **A5** | `gate_sequencing/sequencer.py` — `RaceState.DISQUALIFIED`, `enforce_in_order`, scan-future-gates loop, `is_disqualified` / `dq_reason` props, `GateSpec` defaults from A4 | I-1, I-5 | M | A1/1-3 |
| **A6** | `estimation/gate_pnp.py` + `competition/adapter.py` + `race_pipeline.py` — `CameraIntrinsics` defaults from A4, `body_R_camera` for 20° tilt threaded through `gate_pose_to_drone_position`, `CameraFrame` default height 360 | I-8 | M | A3/8-9 |
| **A7** | `scripts/benchmark.py` synthetic bench — call sequencer + observe `last_crash` + `is_disqualified`; break on either; `overall_passed` requires neither | I-2 | S | A2/5-6 |
| **A8** | `tests/test_ilc_sections.py` — curvature peaks → monotonic boundaries; smooth track → global-ILC fallback | I-3, I-11 | S | itself |
| **A9** | `planning/ilc_sections.py` (new) + `scripts/benchmark.py:310-355` refactor — `derive_section_boundaries`; `config/ilc_defaults.yaml`; race_01 override block | I-3, I-4 | M | A2/7, A8 |
| **A10** | `tests/test_vision_udp.py` — chunk reassembly in order, out of order, partial-timeout, duplicate frame-id | I-9, I-11 | S | itself |
| **A11** | `competition/vision_udp.py` (new) + wire into `competition/mavlink_bridge.py` | I-9 | M | A10 |
| **A12** | `tests/test_mavlink_bridge_commands.py` — `SET_POSITION_TARGET_LOCAL_NED` + `SET_ATTITUDE_TARGET` + heartbeat cadence (GPT-5.5 A9) | I-9, I-11 | S | itself |
| **A13** | `competition/calibration.py` (new, stub) + `tests/test_calibration_fit_synthetic.py` (synthetic-data unit test only) | I-6 | S | itself |
| **A14** | `tests/test_tracker_residual.py` — clamp behaviour, off-by-default byte-identity | I-10, I-11 | S | itself |
| **A15** | `control/learned_residual.py` (new) — 10→64→3 MLP, numpy-only forward; `TrackerConfig` extension; `scripts/train_tracker_residual.py`; `control/residual_weights.npz` | I-10 | M | A14 + holdout regression |
| **A16** | `scripts/run_pipeline_pybullet.py` (new) — thin entry binding `PyBulletAdapter` to `RacePipeline.run()`; `tests/test_pipeline_pybullet_smoke.py` | I-7 (partial) | S | smoke test |
| **A17** | `scripts/benchmark_matrix.py` (new) + `.loop/state/regression_baseline.json` — multi-track suite (race_01 + figure8 + slalom + grand_tour + aigp_default) | I-4 | S | itself |
| **A18** | `sim_pybullet/configs/aigp_default.json` (new, placeholder=true) | I-5 | S | A1/4 |

## What NOT to do this iter
- Don't gut `sim_pybullet/runner.py` — full collapse is iter ≥ 002.
- Don't swap Phase1 detector for the YOLOv8n-pose ONNX (TII-Aerial-trained, AIGP mismatch).
- Don't migrate the geometric tracker to MPCC++ (controller redesign).
- Don't run SITL calibration end-to-end (no DCL binary in-hand).
- Don't tighten the global `THRESHOLDS` in `scripts/benchmark.py:47-56` — bench must be honest before thresholds can be tightened.

## Open questions left for the implementer
1. The `aigp_default.json` placeholder gates: scaled-from-race_01 (1.25×) or 6 hand-placed reps? — **decided: scaled, marked placeholder**.
2. Per-section ILC `(alpha, max_corr, cutoff_hz, vel_scale)` defaults for the curvature-derived sections: pull current race_01 values as the "high-curvature" default, neutral values for "low-curvature"? — **decided: yes**.
3. Tracker residual training corpus: race_01 + figure8 + slalom (train) / grand_tour + aigp_default (eval)? — **decided: yes; eval gate is ≥ 3% avg-err drop on holdout**.

## Cross-iteration invariants
- Every fix must clear its own adversarial test BEFORE moving on.
- `git commit` granularity: one commit per (A1 batch, A2 batch, …) so a regression can be bisected.
- After all A1-A18 land, the iter 001 review round (8-agent adversarial) reads
  the diff and looks for what we missed.
