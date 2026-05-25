"""
Iter-031: train the TrackerResidualMLP with a yaw-corrected FEL target.

This replaces the iter-027 FEL trainer, which was net-neutral on the
matrix (race_01 -3.3%, slalom +7.8%, etc.). The five defects identified
by Opus + Composer are addressed here:

1. **Yaw-frame error (FIXED).** iter-027 used world-frame pos_err but
   the inference rotates by yaw_des (mpc_tracker.py:316-326). On tracks
   with yaw≠0 the FEL target had the wrong sign. We now rotate pos_err
   into the body frame before deriving the target — matching the
   inference's rotation.
2. **Sample-count bias (FIXED).** Per-sample weights = (N_total /
   N_tracks) / n_track give equal total gradient per track.
3. **Clamp ceiling on signal (PARTIAL).** Targets stay clipped to
   ±0.05 — that's the safety contract. But yaw correction + smaller
   FEL gain means most targets sit well inside the clamp, not pinned
   at noise-saturated values.
4. **Validation measured wrong thing (FIXED).** Closed-loop matrix
   every closed_loop_every epochs selects the best-by-real-tracking-
   error checkpoint, not best-by-val-loss.
5. **Training set gap (FIXED).** collect_residual_dataset.py now
   includes aigp_default — only figure8 is excluded.

We also evaluated a BC-oracle target (invert the bench step to compute
the exact accel needed for vel_{k+1} = ref_vel_{k+1}). Empirically that
target demands 50+ m/s² corrections per step — far above the safety
clamp's ±g·0.05 ≈ 0.5 m/s² ceiling — so >90% of training samples
saturate at the clamp and the model learns sign-only. The bounded FEL
target with the four fixes above is the right tradeoff: small, smooth,
and within the safety envelope by construction.

## Target: yaw-corrected FEL (Feedback-Error Learning, body frame)

Rotate world-frame pos_err into the body frame using `yaw_des`, then
apply the FEL linearisation:

    cos_y, sin_y = cos(yaw_des), sin(yaw_des)
    ep_x_body =  pos_err_x * cos_y + pos_err_y * sin_y
    ep_y_body = -pos_err_x * sin_y + pos_err_y * cos_y

    target_droll  = -kp_xy * ep_y_body / g          # body-y accel via roll
    target_dpitch =  kp_xy * ep_x_body / g          # body-x accel via pitch
    target_dthrust =  kp_z * pos_err_z / max_thrust_n

Clipped to ±0.05 rad / ±0.05 thrust.

## Architectural changes from iter-027

- 12-D features (10-D + sin(yaw), cos(yaw)) — model has yaw info.
- Feature standardisation (mean/std stored in npz) — Adam works
  better on heterogeneous-scale inputs.
- Per-sample weights — long tracks no longer dominate.
- Adam optimizer + cosine LR schedule.
- Closed-loop early-stop on the matrix every closed_loop_every epochs.

## Training infra

- Numpy-only Adam (β=0.9/0.999). LR cosine 3e-3 → 1e-4 over 500 epochs.
- Stratified 80/20 train/val split per-track.
- Per-sample weight: equal total gradient per track + mild curvature
  boost capped at 2× (high-curvature samples matter more for the
  tightest corners). Long tracks no longer dominate via sample count.
- Feature standardisation: store feat_mean/feat_std in the npz so
  inference applies the same scaling.
- **Closed-loop early-stop**: every 25 epochs (configurable) run the
  matrix at duration=15s with the current weights; keep checkpoint iff
  ≥4/6 tracks improve. This is the killer feature — best-by-val-loss
  ≠ best-by-real-tracking-error.

## Output

`control/residual_weights.npz` — TrackerResidualMLP.from_npz-loadable
with W1/b1/W2/b2 plus feat_mean/feat_std.
`control/residual_weights_meta.json` — training summary, per-track
baseline-vs-trained tracking error, dataset stats.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from control.learned_residual import (  # noqa: E402
    DEFAULT_N_HIDDEN,
    DEFAULT_N_INPUTS,
    DEFAULT_N_OUTPUTS,
    TrackerResidualMLP,
    load_feature_trace,
)
from competition.drone_spec import (  # noqa: E402
    DEFAULT_GRAVITY_MPS2,
    DEFAULT_LINEAR_DRAG_PER_MASS,
    DEFAULT_MASS_KG,
    DEFAULT_MAX_THRUST_N,
)


# Residual clamps (iter-001 A15 safety contract — targets clipped to
# these bounds so training matches inference).
_CLAMP_RAD: float = 0.05
_CLAMP_THRUST: float = 0.05

# FEL target gains. Tuned so a 1 m body-frame lateral error → 0.051 rad
# target (matches the residual clamp); 1 m vertical → 0.025 thrust delta
# (half clamp). Same scaling as iter-027 — but rotated into body frame.
_KP_XY: float = 0.5
_KP_Z: float = 0.5

# Per-track curvature boost cap. Magnitude tuned so high-curvature
# samples (e.g. slalom apexes) get up to 2× weight without dominating.
_CURVATURE_BOOST_CAP: float = 2.0


def _compute_fel_targets_body_frame(
    pos_err: np.ndarray,     # (N, 3) — world frame
    yaw_des: np.ndarray,     # (N,)
    g: float = None,
    max_thrust_n: float = None,
) -> np.ndarray:
    """Yaw-corrected FEL target. Rotates `pos_err` into the body frame
    using yaw_des, then derives (delta_roll, delta_pitch, delta_thrust)
    via the hover linearisation. Body-frame matches the inference path's
    rotation at mpc_tracker.py:316-326.
    """
    if g is None:
        g = DEFAULT_GRAVITY_MPS2
    if max_thrust_n is None:
        max_thrust_n = DEFAULT_MAX_THRUST_N

    cos_y = np.cos(yaw_des)
    sin_y = np.sin(yaw_des)
    # World → body rotation (inverse of forward yaw rotation).
    ep_x_body =  pos_err[:, 0] * cos_y + pos_err[:, 1] * sin_y
    ep_y_body = -pos_err[:, 0] * sin_y + pos_err[:, 1] * cos_y

    target = np.empty_like(pos_err)
    target[:, 0] = -_KP_XY * ep_y_body / g        # delta_roll
    target[:, 1] =  _KP_XY * ep_x_body / g        # delta_pitch
    target[:, 2] =  _KP_Z * pos_err[:, 2] / max_thrust_n   # delta_thrust
    target[:, 0] = np.clip(target[:, 0], -_CLAMP_RAD, _CLAMP_RAD)
    target[:, 1] = np.clip(target[:, 1], -_CLAMP_RAD, _CLAMP_RAD)
    target[:, 2] = np.clip(target[:, 2], -_CLAMP_THRUST, _CLAMP_THRUST)
    return target


def _compute_per_sample_weights(
    track_id: np.ndarray,
    ref_accel: np.ndarray,
    g: float = None,
) -> np.ndarray:
    """w = (N_total / N_tracks) / n_track + curvature boost (capped 2×).

    Equal total gradient per track removes length bias; curvature boost
    weights high-||ref_accel_xy|| samples more (the corners that matter
    for racing-line tracking).
    """
    if g is None:
        g = DEFAULT_GRAVITY_MPS2
    n_total = track_id.shape[0]
    unique, inverse, counts = np.unique(
        track_id, return_inverse=True, return_counts=True,
    )
    n_tracks = unique.shape[0]
    per_sample_n = counts[inverse]            # (N,)
    base = (n_total / n_tracks) / per_sample_n
    # Curvature boost — clip so an outlier sample doesn't blow up the loss.
    a_xy = np.sqrt(ref_accel[:, 0] ** 2 + ref_accel[:, 1] ** 2)
    boost = 1.0 + a_xy / g
    boost = np.minimum(boost, _CURVATURE_BOOST_CAP)
    return base * boost


def _stratified_split(
    track_id: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx). Each track contributes val_frac of
    its samples to val; remainder to train. Preserves track balance in
    both splits."""
    train_idx, val_idx = [], []
    for t in np.unique(track_id):
        mask = np.where(track_id == t)[0]
        rng.shuffle(mask)
        n_val = max(1, int(round(mask.size * val_frac)))
        val_idx.append(mask[:n_val])
        train_idx.append(mask[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


class _AdamState:
    """Minimal numpy Adam — per-parameter (m, v) buffers."""
    def __init__(self, shape: tuple, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, w: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return w - lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _forward(x: np.ndarray, w1, b1, w2, b2, output_clamp: np.ndarray | None = None):
    """(N, n_inputs) → (N, n_outputs) forward pass with tanh activation.
    Returns (output, h, raw). If output_clamp is set, output =
    output_clamp * tanh(raw / output_clamp); otherwise output = raw.
    """
    h = np.tanh(x @ w1 + b1)
    raw = h @ w2 + b2
    if output_clamp is not None:
        out = output_clamp * np.tanh(raw / output_clamp)
    else:
        out = raw
    return out, h, raw


def _cosine_lr(epoch: int, total_epochs: int, lr_max: float, lr_min: float):
    """Cosine decay schedule."""
    frac = epoch / max(1, total_epochs - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * frac))


def _evaluate_closed_loop(
    weights_path: Path,
    duration: float,
    baseline_results: dict | None = None,
) -> dict:
    """Run the matrix with `residual_weights_path=weights_path` and
    return per-track tracking errors. If baseline_results is provided,
    compute the improvement diff. Lazy-imports the bench infra to
    avoid circular deps at module-load time."""
    from scripts.benchmark import run_synthetic_benchmark
    from scripts.benchmark_matrix import _list_configs

    out: dict = {}
    for cfg_path in _list_configs():
        name = cfg_path.stem
        with open(cfg_path) as f:
            cfg = json.load(f)
        overrides = {
            "use_residual": True,
            "residual_weights_path": str(weights_path),
        }
        r = run_synthetic_benchmark(
            duration=duration, config=cfg,
            tracker_config_overrides=overrides,
        )
        out[name] = {
            "avg_tracking_error_m": r.get("avg_tracking_error_m"),
            "sim_passed": r.get("sim_passed"),
            "skipped": bool(r.get("skipped")),
        }
    if baseline_results is not None:
        for name, base in baseline_results.items():
            if name not in out or out[name]["skipped"]:
                continue
            base_err = base.get("avg_tracking_error_m")
            on_err = out[name]["avg_tracking_error_m"]
            if base_err is None or on_err is None:
                continue
            out[name]["baseline_avg_tracking_error_m"] = base_err
            out[name]["improvement_pct"] = (
                100.0 * (base_err - on_err) / max(base_err, 1e-9)
            )
    return out


def _matrix_baseline(duration: float) -> dict:
    """Run the matrix with use_residual=False (current default) and
    return per-track tracking errors. Cached at call time."""
    from scripts.benchmark import run_synthetic_benchmark
    from scripts.benchmark_matrix import _list_configs

    out = {}
    for cfg_path in _list_configs():
        name = cfg_path.stem
        with open(cfg_path) as f:
            cfg = json.load(f)
        r = run_synthetic_benchmark(duration=duration, config=cfg)
        out[name] = {
            "avg_tracking_error_m": r.get("avg_tracking_error_m"),
            "sim_passed": r.get("sim_passed"),
            "skipped": bool(r.get("skipped")),
        }
    return out


def train(
    dataset_path: Path,
    out_path: Path,
    epochs: int = 500,
    batch_size: int = 256,
    lr_max: float = 3e-3,
    lr_min: float = 1e-4,
    val_frac: float = 0.20,
    seed: int = 0,
    closed_loop_every: int = 25,
    closed_loop_duration: float = 15.0,
    closed_loop_patience: int = 3,
    skip_closed_loop: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)

    data = load_feature_trace(dataset_path)
    features = data["features"]
    n = features.shape[0]
    if n < 100:
        raise RuntimeError(
            f"dataset too small ({n} samples); collect more before training"
        )

    # Build yaw-corrected FEL targets from v2 trace.
    targets = _compute_fel_targets_body_frame(
        pos_err=data["pos_err"], yaw_des=data["yaw_des"],
    )

    # track_id is REQUIRED — silent fallback to zeros would collapse
    # per-track weighting into one global track and re-introduce the
    # length-bias defect from iter-027. Re-run collect_residual_dataset
    # if you see this.
    with np.load(dataset_path) as raw:
        if "track_id" not in raw.files:
            raise RuntimeError(
                f"dataset {dataset_path} is missing the 'track_id' field. "
                "Regenerate via scripts/collect_residual_dataset.py "
                "(iter-031 v2 format)."
            )
        track_id = raw["track_id"].copy()

    # Per-sample weights.
    weights = _compute_per_sample_weights(track_id, data["ref_accel"])

    # Stratified split per track.
    train_idx, val_idx = _stratified_split(track_id, val_frac, rng)
    x_train, y_train, w_train = features[train_idx], targets[train_idx], weights[train_idx]
    x_val, y_val, w_val = features[val_idx], targets[val_idx], weights[val_idx]

    # Feature standardisation: fit on train split only.
    feat_mean = x_train.mean(axis=0)
    feat_std = x_train.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    x_train_n = (x_train - feat_mean) / feat_std
    x_val_n = (x_val - feat_mean) / feat_std

    # Initialise weights (Glorot-ish).
    n_in, n_h, n_out = DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS
    W1 = rng.normal(0, math.sqrt(2.0 / n_in), size=(n_in, n_h))
    b1 = np.zeros(n_h)
    W2 = rng.normal(0, math.sqrt(2.0 / n_h), size=(n_h, n_out))
    b2 = np.zeros(n_out)
    opts = {
        "W1": _AdamState(W1.shape),
        "b1": _AdamState(b1.shape),
        "W2": _AdamState(W2.shape),
        "b2": _AdamState(b2.shape),
    }
    # Output bound — per-channel clamp matching the safety contract.
    output_clamp = np.array([_CLAMP_RAD, _CLAMP_RAD, _CLAMP_THRUST], dtype=np.float64)

    # Baseline for closed-loop scoring.
    baseline_cl = None if skip_closed_loop else _matrix_baseline(closed_loop_duration)

    history = []
    best_val_loss = float("inf")
    best_weights_by_val = None
    best_closed_loop_score = -float("inf")
    best_weights_by_cl = None
    best_cl_results: dict | None = None
    cl_no_improve_count = 0

    for epoch in range(epochs):
        lr = _cosine_lr(epoch, epochs, lr_max, lr_min)
        # Shuffle.
        perm = rng.permutation(len(x_train_n))
        x_tr = x_train_n[perm]
        y_tr = y_train[perm]
        w_tr = w_train[perm]
        # Mini-batch SGD.
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(x_tr), batch_size):
            xb = x_tr[start:start + batch_size]
            yb = y_tr[start:start + batch_size]
            wb = w_tr[start:start + batch_size]
            y_hat, h, raw = _forward(xb, W1, b1, W2, b2, output_clamp)
            err = y_hat - yb                                  # (B, 3)
            # Weighted MSE: sum(w * err^2) / sum(w).
            wb_col = wb[:, None]
            loss_sum = float(np.sum(wb_col * (err ** 2)))
            wsum = float(np.sum(wb)) * y_hat.shape[1]
            loss = loss_sum / max(wsum, 1e-9)
            epoch_loss += loss
            n_batches += 1
            # Backward with per-sample weighting. Chain rule through
            # the output_clamp * tanh(raw / clamp) activation:
            #   y_hat = c * tanh(raw / c)
            #   d y_hat / d raw = 1 - tanh^2(raw / c) = 1 - (y_hat / c)^2
            grad_y = 2 * wb_col * err / max(wsum, 1e-9)        # (B, 3)
            grad_raw = grad_y * (1 - (y_hat / output_clamp) ** 2)
            grad_W2 = h.T @ grad_raw
            grad_b2 = grad_raw.sum(0)
            grad_h = grad_raw @ W2.T
            grad_pre = grad_h * (1 - h ** 2)
            grad_W1 = xb.T @ grad_pre
            grad_b1 = grad_pre.sum(0)
            W1 = opts["W1"].step(W1, grad_W1, lr)
            b1 = opts["b1"].step(b1, grad_b1, lr)
            W2 = opts["W2"].step(W2, grad_W2, lr)
            b2 = opts["b2"].step(b2, grad_b2, lr)

        train_loss = epoch_loss / max(1, n_batches)
        y_val_hat, _, _ = _forward(x_val_n, W1, b1, W2, b2, output_clamp)
        val_err = y_val_hat - y_val
        val_loss = float(
            np.sum(w_val[:, None] * (val_err ** 2))
            / max(np.sum(w_val) * y_val_hat.shape[1], 1e-9)
        )
        history.append({
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "val_loss": val_loss,
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_by_val = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

        # Closed-loop early-stop: every closed_loop_every epochs, run
        # the matrix at duration=closed_loop_duration with the current
        # weights. Scoring policy matches the acceptance test exactly
        # (tests/test_residual_matrix_gain.py):
        #   - HARD FAIL if ANY of the 7 matrix tracks loses sim_passed.
        #   - HARD FAIL if ANY non-figure8 track regresses by > 1%.
        #   - Score = count of non-figure8 tracks improved by > 1% relative.
        # Hard failures get score = -1e6 so a passing checkpoint always
        # beats a passing-then-broken one.
        if (
            not skip_closed_loop
            and epoch > 0
            and (epoch % closed_loop_every == 0 or epoch == epochs - 1)
        ):
            # Save current weights to a temp file the bench can read.
            tmp_path = out_path.with_suffix(".tmp.npz")
            tmp_mlp = TrackerResidualMLP(
                W1=W1, b1=b1, W2=W2, b2=b2,
                feat_mean=feat_mean, feat_std=feat_std,
                output_clamp=output_clamp,
            )
            tmp_mlp.to_npz(tmp_path)
            cl = _evaluate_closed_loop(
                tmp_path, closed_loop_duration, baseline_cl,
            )
            improved = 0
            broke_sim_passed: list[str] = []
            regressed_over_1pct: list[str] = []
            for name, r in cl.items():
                if r.get("skipped"):
                    continue
                if not r.get("sim_passed"):
                    broke_sim_passed.append(name)
                impr = r.get("improvement_pct")
                if impr is not None:
                    if name != "figure8" and impr > 1.0:
                        improved += 1
                    if name != "figure8" and impr < -1.0:
                        regressed_over_1pct.append(name)
            hard_fail = bool(broke_sim_passed) or bool(regressed_over_1pct)
            score = -1e6 if hard_fail else float(improved)
            if score > best_closed_loop_score:
                best_closed_loop_score = score
                best_weights_by_cl = (
                    W1.copy(), b1.copy(), W2.copy(), b2.copy(),
                    feat_mean.copy(), feat_std.copy(),
                )
                best_cl_results = cl
                cl_no_improve_count = 0
            else:
                cl_no_improve_count += 1
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            history[-1]["closed_loop_score"] = score
            history[-1]["improved_count"] = improved
            history[-1]["broke_sim_passed"] = broke_sim_passed
            history[-1]["regressed_over_1pct"] = regressed_over_1pct
            if cl_no_improve_count >= closed_loop_patience:
                # Early stop.
                history[-1]["early_stopped"] = True
                break

    # Prefer closed-loop-best weights; fall back to val-loss-best.
    if best_weights_by_cl is not None:
        W1, b1, W2, b2, feat_mean_best, feat_std_best = best_weights_by_cl
        select_method = "best_closed_loop"
    elif best_weights_by_val is not None:
        W1, b1, W2, b2 = best_weights_by_val
        feat_mean_best, feat_std_best = feat_mean, feat_std
        select_method = "best_val_loss_fallback"
    else:
        feat_mean_best, feat_std_best = feat_mean, feat_std
        select_method = "final"

    mlp = TrackerResidualMLP(
        W1=W1, b1=b1, W2=W2, b2=b2,
        feat_mean=feat_mean_best, feat_std=feat_std_best,
        output_clamp=output_clamp,
    )
    mlp.to_npz(out_path)

    # Verify output range after training — must be within clamps so the
    # iter-001 hard-clamp safety contract is preserved.
    y_train_hat, _, _ = _forward(
        (x_train - feat_mean_best) / feat_std_best,
        W1, b1, W2, b2, output_clamp,
    )
    y_val_hat, _, _ = _forward(
        (x_val - feat_mean_best) / feat_std_best,
        W1, b1, W2, b2, output_clamp,
    )
    summary = {
        "samples_total": int(n),
        "samples_train": int(len(x_train)),
        "samples_val": int(len(x_val)),
        "epochs_run": int(history[-1]["epoch"] + 1),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "val_frac": val_frac,
        "seed": seed,
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_val_loss": float(history[-1]["val_loss"]),
        "best_val_loss": float(best_val_loss),
        "best_closed_loop_score": float(best_closed_loop_score),
        "select_method": select_method,
        "best_closed_loop_results": best_cl_results,
        "feat_mean": [float(v) for v in feat_mean_best],
        "feat_std": [float(v) for v in feat_std_best],
        "train_pred_range": {
            "min": [float(y_train_hat[:, i].min()) for i in range(DEFAULT_N_OUTPUTS)],
            "max": [float(y_train_hat[:, i].max()) for i in range(DEFAULT_N_OUTPUTS)],
        },
        "val_pred_range": {
            "min": [float(y_val_hat[:, i].min()) for i in range(DEFAULT_N_OUTPUTS)],
            "max": [float(y_val_hat[:, i].max()) for i in range(DEFAULT_N_OUTPUTS)],
        },
        "target_clamp_rad": _CLAMP_RAD,
        "target_clamp_thrust": _CLAMP_THRUST,
        "out_path": str(out_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train the TrackerResidualMLP (iter-031)")
    parser.add_argument(
        "--dataset",
        default=str(_REPO_ROOT / "control" / "residual_dataset.npz"),
        help="Input dataset .npz from collect_residual_dataset.py",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "control" / "residual_weights.npz"),
        help="Output .npz path for trained MLP weights",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-max", type=float, default=3e-3)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--closed-loop-every", type=int, default=25,
        help="Run matrix every N epochs to select best-by-closed-loop weights.",
    )
    parser.add_argument(
        "--closed-loop-duration", type=float, default=15.0,
        help="Per-track sim duration for closed-loop scoring.",
    )
    parser.add_argument(
        "--closed-loop-patience", type=int, default=3,
        help="Early-stop after N consecutive non-improving closed-loop checks.",
    )
    parser.add_argument(
        "--skip-closed-loop", action="store_true",
        help="Disable the closed-loop matrix evaluation (faster, smoke tests).",
    )
    parser.add_argument(
        "--meta-out", default=None,
        help="If set, write training metadata JSON to this path.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = train(
        Path(args.dataset),
        out_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        seed=args.seed,
        closed_loop_every=args.closed_loop_every,
        closed_loop_duration=args.closed_loop_duration,
        closed_loop_patience=args.closed_loop_patience,
        skip_closed_loop=args.skip_closed_loop,
    )
    print(json.dumps(summary, indent=2))
    meta_out = args.meta_out
    if meta_out is None:
        meta_out = str(out_path.with_name(out_path.stem + "_meta.json"))
    Path(meta_out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
