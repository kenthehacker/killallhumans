# Iter 001 — Adversarial Review Synthesis

7 substantive reviews returned (1 Opus 4.7 max-thinking + 2 GPT-5.5 extra-high + 4 Composer 2.5). 1 Composer task crashed at start (composer-25-1, duration 2.7s — model error, not output issue).

## BLOCKERs (apply NOW or iter-002 immediately)

### B1. Calibration regression solves the wrong equation; test is tautological [Opus F1 — singleton with high confidence]
- File: `competition/calibration.py:93,104` + `tests/test_calibration.py:36`
- The code sets `y = a + g` and `X = [-u, -v]`, solving `a + g = -k_t·u - k_d·v_z`, which is off by `+2g` from the correct NED physics `a_z = g - k_t·u - k_d·v_z`.
- Recovered `k_t` is sign-flipped (negative ≈ -22 at hover). The unit test's `_synth_samples` uses `a = -k_t·u - k_d·v - g` — matching the code's wrong equation — so the bug is invisible from test pass.
- **Apply this iter**: change `y = self.gravity - a` and `X = [u, v]`; flip the test's synth formula; add positivity assertion on recovered `thrust_per_mass`.

### B2. RacePipeline keeps flying after sequencer DQ or crash [5/7 reviews — universal BLOCKER]
- File: `race_pipeline.py:326-328` (only checks `is_complete`) + `RaceSession.should_stop` lambda at line 263
- Today: an out-of-order DQ or gate-frame crash silently leaves the pipeline running; the drone keeps tracking the next reference point.
- **Apply this iter**: add `is_disqualified` and `last_crash is not None` checks to the early-return block AND to `should_stop`.

### B3. CameraFrame dataclass default still 640×480 [Opus F4 + 4 other reviews]
- File: `competition/adapter.py:145-146`
- Iter-001 A6 fixed `CameraIntrinsics` and `PipelineConfig` defaults but left the actual transport dataclass at 480.
- **Apply this iter**: change default to 360.

### B4. Vision UDP receiver not wired into `mavlink_bridge.get_camera_frame()` [6/7 reviews]
- File: `competition/mavlink_bridge.py:163-167`
- The reassembler is unit-tested but unreachable from the live flight path; bridge still returns `None`.
- **Defer to iter-002**: needs MAVSDK and JPEG-decode integration tests; bigger surface than fits this turn.

### B5. Camera +20° tilt stored but never applied in PnP world transform [4/7 reviews BLOCKER, Opus F13 NIT]
- File: `estimation/gate_pnp.py::gate_pose_to_drone_position` + `race_pipeline.py:115-119`
- `PipelineConfig.camera_pitch_offset_rad` is computed but never threaded to `CameraIntrinsics.pitch_offset_rad`, and `gate_pose_to_drone_position` doesn't consume `pitch_offset_rad`.
- **Defer to iter-002**: requires careful rotation-frame audit; horizon test only verifies the trigonometric formula, not the code path.

## MAJORs (iter-002 scope)
- **Multi-gate-per-tick crediting** drops the second pass (Opus F2 + gpt-55-1 F6). DQ scan starts at `_current_idx + 1` AFTER credit, so a single segment that crosses two openings in one tick credits one and silently DQs (or mis-attributes) the other.
- **Future-gate strut hits silently ignored** (Opus F3 + composer-25-3 F3). DQ check uses opening; crash check only on current gate. Annulus `[opening, outer_frame]` on future gates is neither.
- **Vision reassembly trusts `jpeg_size` without validating sum of chunk sizes** (7/7 reviews). Risk: corrupt JPEG silently passed to `cv2.imdecode`, which returns None and is dropped.
- **8-minute VQ1 run cap not enforced** (5/7 reviews). `AIGP_VQ1_MAX_RUN_DURATION_S` is defined; `RaceState.TIMED_OUT` is dead code.
- **`enforce_in_order=True` may false-DQ legitimate replanner recovery** (3/7 reviews). After missing a gate, drone may legitimately re-attempt; current code DQs on the first plane crossing of any future gate the recovery path encounters.
- **`pass_through_margin=1.5` leaks into DQ opening check** (Opus F14 + composer-25-4 F10). For race_01 geometry (interior 1.2, border 0.18), opening_half × margin = 0.9 m > outer_half = 0.78 m — false-positive DQ region.
- **AIGP intrinsics fx=fy=320 vs 90° VFoV contradiction** (composer-25-4 F8). With h=360, VFoV from `2·atan(180/320) = 58.7°`, not the spec's 90°. Either fx/fy or VFoV is wrong; the spec PDF is the authority — likely the spec means HFoV.
- **ILC absolute threshold 1e-6 misclassifies bimodal trajectories** (Opus F5 + composer-25-3 F8 + composer-25-4 F11 + composer-25-5 F8). A track with long low-curvature stretches and one sharp turn quantile-collapses to "all low."
- **PnPEstimator standalone defaults still 1.2 m / 640×480** (gpt-55-1 F4 + composer-25-2 F10 + composer-25-3 F10 + composer-25-4 F12 + composer-25-5 F11). The pipeline wires the new defaults but a bare `GatePnPEstimator()` constructor still emits legacy geometry.

## MINORs (iter-002 or later)
- ILC `_pt_to_step` integer truncation drops short sections (Opus F6 + others)
- `aigp_default.json` is topology-simplified, NOT a true 1.25× race_01 (Opus F7)
- `vel_scale=0.5` high default is the median of race_01's tunings (Opus F8)
- Bench overloads `crashed=True` for both crash and DQ (composer-25-2 F7 + composer-25-3 F6)
- Calibration accepts degenerate zero-thrust fits without rank check (composer-25-2 F9 + composer-25-3 F12 + composer-25-4 F13 + composer-25-5 F10)
- `_delivered_ids` cap allows late dup frames after ~64 frames (Opus F10 + composer-25-4 F16 + composer-25-5 F14)
- Existing sequencer test docstrings drift from new DQ semantics (Opus F12)

## NITs
- `SimplePositionTracker` silently ignores `use_residual` (Opus F11 + multi-review consensus)
- `_delivered_ids` uses O(N) list (Opus F10)
- AIGP drone footprint constants defined but unused in planning/collision (composer-25-4 F14)

## Things iter-001 got right (cross-validated by 4+ reviews)
- Sequencer in-order DQ semantics for the single-skip case
- AIGP geometry centralisation in `competition/aigp_geometry.py`
- Residual MLP clamp composition (residual-clamp THEN physical-clamp)
- Vision UDP reassembler **unit** tests (15 tests, OoO, dup, GC, eviction)
- ILC JSON config / race_01 override mechanism (the magic-number cleanup)

## Decision: this-turn patch scope (iter-001b)
Apply B1, B2, B3 NOW (BLOCKERs with small surface). Defer B4 (vision wiring), B5 (camera tilt in PnP), and all MAJORs to iter-002 with clear traceback per finding.
