# iter-031: ML residual — infrastructure shipped, beats-baseline goal NOT reached

## Result

The yaw-corrected FEL target with all five iter-027 defect fixes
(per-track weights, normalization, Adam, sin/cos yaw features,
closed-loop early-stop) trained for 121 epochs and still produced
hard-fail checkpoints in every closed-loop evaluation:

| Track | base | trained | diff | sim_passed (trained) |
|---|---:|---:|---:|---|
| race_01 | 0.0887 | 0.0853 | **+3.83%** | ✓ |
| aigp_default | 0.2338 | 0.2604 | −11.41% | ✓ |
| figure8 | 0.4397 | 0.4482 | −1.94% | ✓ |
| grand_tour | 0.0794 | 0.0860 | −8.32% | ✓ |
| slalom | 0.1588 | 0.1747 | −10.00% | ✗ **BROKE COMPLETION** |
| straight_hairpin | 0.0701 | 0.0753 | −7.54% | ✓ |
| vertical_cliff | 0.0550 | 0.0847 | −54.01% | ✓ |

Best closed-loop score: **−1e6** (hard fail) every checkpoint.
Score = −1e6 means at least one of: track lost sim_passed, OR a track
regressed > 1%. Every checkpoint hit one of those.

## Root-cause analysis

The kinematic bench is **structurally hostile to a residual MLP**:

1. **PD is already well-tuned.** kp_xy=7, kd_xy=5.5, drag-compensating
   feedforward; matrix race_01 12/12 at ~17 s with avg err 0.09 m. The
   residual has nowhere to add value because the system is already
   near-optimal under PD.
2. **No model mismatch to compensate for.** Real-world drone control
   residuals capture rotor mismatch, drag asymmetry, gyro bias. The
   kinematic bench is a closed-form `accel - drag*vel; clamp;
   integrate`. The PD's model IS the bench's model. There's no
   systematic bias for the residual to predict.
3. **FEL target is redundant with PD.** `target_droll = −kp_xy·ep_y/g`
   is a linearisation of the same map PD uses. Any nonzero output
   either re-derives PD (no value) or perturbs PD (negative value).
4. **BC oracle target is out-of-range.** Inverting the bench step
   demands corrections of 10-80 m/s² per step — far above the
   safety-clamped residual ceiling of g·0.05 ≈ 0.5 m/s². >90% of BC
   oracle targets saturated at the clamp.
5. **Tracks have different optimal residuals.** vertical_cliff needs
   −Δthrust on descent; race_01 needs +Δroll into right turns. A
   single MLP can't learn to be inactive on one track and active on
   another without explicit "track context" features (which would
   leak course-specific info — charter violation).

## What iter-031 SHIPPED

Despite the residual not beating baseline, all the **infrastructure**
is correct and production-grade:

- `control/learned_residual.py`: 12-D `build_input_features` (adds
  `sin(yaw)`, `cos(yaw)`); v2 `save_feature_trace` / `load_feature_trace`;
  optional `feat_mean`/`feat_std` input normalization stored in npz;
  optional `output_clamp * tanh(raw/output_clamp)` bounded output
  activation (smooth, in-envelope by construction).
- `control/mpc_tracker.py`: v2 13-tuple trace fields (`pos`, `vel`,
  `yaw_des`, `ref_pos`, `ref_vel`, `ref_accel`, `accel_des_baseline`)
  give the trainer everything it needs for yaw-aware targets. Wide
  `except (FileNotFoundError, OSError, KeyError, ValueError)` net
  around weights load, with `n_inputs` schema check — `from_npz` can
  never crash `GeometricTracker.__init__`. Auto-resolve
  `<repo>/control/residual_weights.npz` when `use_residual=True` and
  no explicit path.
- `scripts/collect_residual_dataset.py`: 6 tracks (added aigp_default,
  excluded only figure8); `track_id` tag required by trainer.
- `scripts/train_tracker_residual.py`: numpy Adam, cosine LR schedule
  3e-3 → 1e-4, stratified train/val 80/20 per-track, per-sample
  weighting (equal-gradient + curvature boost capped 2×), yaw-corrected
  FEL target (rotates `pos_err` into body frame before linearisation),
  **HARD-FAIL closed-loop early-stop**: every 20 epochs runs the
  matrix; score is −1e6 if ANY track loses `sim_passed` or regresses
  > 1% relative. Acceptance-test-aligned scoring.
- `tests/test_residual_matrix_gain.py`: acceptance gate. Skipped iff
  weights absent (no false-positives). Asserts ≥5/6 improve + no
  > 1% regression + figure8 sim_passed.

## Why this matters even though baseline still wins

When (a) we move to PyBullet as the truth tier (task #13), or (b) we
ship to the real AIGP-class drone (post-SITL calibration), there WILL
be model mismatch the residual can learn. The infrastructure ready and
trainable on collected real-flight data is the iter-031 deliverable.

For the kinematic bench, baseline PD remains the best available
controller. `TrackerConfig.use_residual = False` stays as default;
auto-resolve picks up trained weights only when:
1. `control/residual_weights.npz` is committed (currently gitignored).
2. `tests/test_residual_matrix_gain.py` passes (currently SKIP).

Both gates must turn green together before the default flips.

## Per brick-wall rule

The training stack is fully wired. Iter-031 does NOT continue grinding
the matrix because the bench is structurally hostile. iter-031's loop
budget allocates the remaining iters to:
- iter-032: planner peak-accel projection (task #10) — concrete, the
  three-planner spawn already converged.
- iter-033+: visual_demo / NED↔ENU tracker refactor (task #13) — the
  real path to using the AIGP-proxy drone in visualization.

## Adversarial review fixes applied this iter

From the 3-reviewer round (Opus, Composer, Codex):
- **Auto-resolve catches all load failures**, not just FileNotFoundError
  (Opus m1, Composer #1, Codex MAJOR). Wide `except` + `n_inputs`
  schema check.
- **Trainer's CL scoring uses acceptance-test-identical thresholds**:
  ≥1% improvement, >1% regress = hard fail, broken sim_passed = hard
  fail. No more "best CL ≠ acceptance pass" mismatch (Opus M2,
  Composer #3, Codex MAJOR).
- **Acceptance test uses 1% relative threshold** instead of 1e-4 m
  absolute (Opus M1, Codex MAJOR — at sim noise floor).
- **`track_id` required, no silent fallback** (Composer #2).
- **Stale "10-D" comments fixed** in 5 files (all reviewers NIT).
- **`residual_weights.tmp.npz` + meta JSON gitignored** (Opus m5/m6).
- **Test for zero_init fallback uses explicit nonexistent path** so
  auto-resolve doesn't accidentally pick up a trained weights file.

Adversarial-review-residual:
- Curvature proxy is acceleration magnitude, not geometric κ
  (all 3 reviewers MINOR) — acknowledged; per-sample weighting
  is a soft signal and matrix scoring is the real gate.
- Stratified split edge for 1-sample tracks (Opus m4, Codex MINOR) —
  not exercised in current 6-track collector. Pinned by training data
  size; not a ship blocker.
