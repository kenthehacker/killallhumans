# iter-031: make the tracker residual actually help

## Two-agent synthesis (Opus max-thinking + Composer 2.5)

Both agents converged on the key diagnosis and the fix shape.

### Root cause (FEL is structurally redundant)

`scripts/train_tracker_residual.py:65-85` derives targets via a static
hover linearisation: `target_droll = −k·ep_y/g`. This is the same
linear map the PD already applies via `kp_xy·ep` in
`control/mpc_tracker.py:195-199`. The MLP learns a 0.7% gain bump that
fights the PD on every track.

Additional defects (all five flagged by both agents):
1. **Yaw-frame error.** Trainer uses raw world `pos_err`; inference
   rotates by `yaw_des` (`mpc_tracker.py:316-326`). Tracks with
   `yaw ≠ 0` get sign-wrong targets.
2. **Sample-count bias.** Longer tracks dominate MSE; race_01 ~17 s
   vs slalom ~8 s. Matches the iter-027 outcome: race_01 −3.3%,
   slalom +7.8%.
3. **Clamp ceiling on signal.** Clipping at ±0.05 rad squashes most
   non-zero targets to half the clamp; model learns noise-saturated
   values.
4. **Validation measures wrong thing.** Best-by-val-loss ≠
   best-by-closed-loop tracking error.
5. **Training set gap.** `aigp_default` excluded — the placeholder
   goal track is never represented.

### Resolved design (pure BC oracle, no FEL hybrid)

| Choice | Decision | Source |
|---|---|---|
| Target | **One-step BC oracle** from inverting the bench drag-clamp step | Both (Opus called BC "core", Composer suggested 10% FEL hybrid — REJECTED, mixing in failing signal) |
| Frame | Project oracle ∆a back through inverse small-angle map (rotate by `−yaw_des`) | Both |
| Features | **12-D** (10-D + `sin(yaw)`, `cos(yaw)`) | Composer (Opus 16-D adds redundant `vel_ref` and `speed` derivable from existing) |
| Hidden | **64** | Composer (Opus 128 is overkill for 12→3) |
| Activation | tanh | Both |
| Output | unchanged `(∆roll, ∆pitch, ∆thrust)` clipped ±0.05 | both (preserves injection contract — constraint #5) |
| Normalization | **store `feat_mean`/`feat_std` in npz, apply in forward** | Composer (Opus assumed unscaled; Adam fails on heterogeneous scales) |
| Optimizer | Numpy Adam (β=0.9/0.999) | Both |
| LR schedule | cosine 3e-3 → 1e-4 over 500 epochs | Opus (Composer same) |
| Batch | 256 | Both |
| Split | **Stratified 80/20 per-track** | Opus (Composer LOTO is 6× cost; we have closed-loop matrix as the real gate) |
| Per-track weight | `w = (N_total/N_tracks) / n_track` + mild curvature boost (capped 2×) | Composer (Opus's adaptive boost is harder to reason about) |
| Training set | **6 tracks** (skip ONLY figure8) | Composer (Opus skipped both figure8 + aigp_default) |
| Early stop | **Closed-loop matrix every 25 epochs**: keep iff ≥4/6 improved + figure8 8/8 holds | Opus's killer idea |
| Seeds | 3 (0,1,2), ship best by closed-loop | Both |

### Acceptance gate (binding)

New test `tests/test_residual_matrix_gain.py`:
- A/B matrix with `use_residual={False, True}`
- Pass iff ≥5/6 tracks improve (err drops below baseline by ≥1e-4 m)
  AND no track regresses >1% AND figure8 sim_passed under residual ON.

### Default-on rollout

1. `TrackerConfig.use_residual: bool = True` (flip default at `mpc_tracker.py:109`)
2. `GeometricTracker.__init__` auto-resolves
   `<repo>/control/residual_weights.npz`; falls back to `zero_init` on
   FileNotFoundError (byte-identical to baseline).
3. Weights committed to repo.

### Files

ADD: `control/residual_weights.npz`, `control/residual_weights_meta.json`,
`tests/test_residual_matrix_gain.py`.

EDIT: `control/learned_residual.py` (12-D features, v2 schema,
optional normalization), `control/mpc_tracker.py` (trace v2 fields,
auto-resolve, default-on), `scripts/collect_residual_dataset.py`
(remove aigp_default skip, tag track_id), `scripts/train_tracker_residual.py`
(BC oracle target, Adam, per-track weights, closed-loop early-stop,
normalization stats), `tests/test_tracker_residual.py` (12-D contract).

Skip: `.gitignore` — weights/ → control/residual_weights.npz needs to be
ADDED, not excluded.

### Commit order

1. Schema upgrade (no behavior change; tests still green).
2. Trainer rewrite (still off by default).
3. Train, commit weights + meta JSON.
4. Acceptance test passes locally.
5. Flip default-on; full pytest green.
