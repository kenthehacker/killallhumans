"""
Iter-027: train the TrackerResidualMLP from a collected dataset.

Approach: Feedback-Error Learning (Romero 2025 "On Your Own"). The
target residual at each sample is what additional (roll, pitch, thrust)
correction would have driven the tracker's position error toward zero.

Target derivation (linearised around hover):
  - delta_roll target  = -k_xy * pos_err_y / g
      (positive roll commands +y body lateral accel via gravity tilt)
  - delta_pitch target = +k_xy * pos_err_x / g
      (positive pitch commands +x body fwd accel via gravity tilt)
  - delta_thrust target = +k_z * pos_err_z / max_thrust_n
      (positive thrust delta commands +z accel directly)

Sign / scaling: chose +k_xy=0.5 so a 1 m lateral error → 0.05 rad target
(matches the iter-001 residual clamp ±0.05 rad). 1 m vertical error →
~0.025 thrust delta (half the clamp). Clip targets to the iter-001
clamps so the model can't learn an unsafe response.

Training: numpy-only gradient descent on MSE loss. 200 epochs at lr=0.01
with batch SGD (batch=128). Validates by computing avg residual norm
on a held-out 10% split.

Output: control/residual_weights.npz, loadable via
`TrackerResidualMLP.from_npz`.
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
    DEFAULT_MAX_THRUST_N,
)


# Target derivation gains.
_KP_XY: float = 0.5   # lateral error → roll/pitch target
_KP_Z: float = 0.5    # vertical error → thrust delta target

# Residual clamps (iter-001 A15 safety contract — model can't be trained
# to produce values outside these bounds either).
_CLAMP_RAD: float = 0.05
_CLAMP_THRUST: float = 0.05


def derive_targets(pos_err: np.ndarray) -> np.ndarray:
    """Convert (N, 3) position errors into (N, 3) (delta_roll, delta_pitch,
    delta_thrust) targets. Linearised around hover; clipped to the
    residual clamps so the model has a chance of fitting bounded outputs.

    pos_err is in world frame (z-up convention used by the synthetic
    bench). Negative z error = below reference → need MORE thrust.
    """
    g = DEFAULT_GRAVITY_MPS2
    target = np.empty_like(pos_err)
    # roll commands lateral y-accel via tilt: delta_roll ≈ pos_err_y / g
    target[:, 0] = -_KP_XY * pos_err[:, 1] / g
    # pitch commands fwd x-accel: delta_pitch ≈ pos_err_x / g
    target[:, 1] = _KP_XY * pos_err[:, 0] / g
    # thrust delta normalized by max_thrust_n
    target[:, 2] = _KP_Z * pos_err[:, 2] / DEFAULT_MAX_THRUST_N
    # Clip to the iter-001 clamps so training matches the inference clamp.
    target[:, 0] = np.clip(target[:, 0], -_CLAMP_RAD, _CLAMP_RAD)
    target[:, 1] = np.clip(target[:, 1], -_CLAMP_RAD, _CLAMP_RAD)
    target[:, 2] = np.clip(target[:, 2], -_CLAMP_THRUST, _CLAMP_THRUST)
    return target


def _forward(x: np.ndarray, w1, b1, w2, b2):
    """(N, 10) → (N, 3) forward pass with tanh activation."""
    h = np.tanh(x @ w1 + b1)
    return h @ w2 + b2, h


def _normalize(x: np.ndarray):
    """Feature standardisation. Returns (x_norm, mean, std)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def train(
    dataset_path: Path,
    out_path: Path,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    val_frac: float = 0.10,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)

    data = load_feature_trace(dataset_path)
    features = data["features"]
    pos_err = data["pos_err"]
    n = features.shape[0]
    if n < 100:
        raise RuntimeError(
            f"dataset too small ({n} samples); collect more before training"
        )

    targets = derive_targets(pos_err)

    # Shuffle and split.
    perm = rng.permutation(n)
    features = features[perm]
    targets = targets[perm]
    n_val = int(n * val_frac)
    x_train, y_train = features[n_val:], targets[n_val:]
    x_val, y_val = features[:n_val], targets[:n_val]

    # NB: do NOT normalize features — the MLP at inference time consumes
    # raw 10-dim features. Building a normalizer into the network is a
    # future improvement; for now we let the W1 weights absorb the scale.

    # Initialise weights — small random.
    W1 = rng.normal(0, 0.1, size=(DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN))
    b1 = np.zeros(DEFAULT_N_HIDDEN)
    W2 = rng.normal(0, 0.1, size=(DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS))
    b2 = np.zeros(DEFAULT_N_OUTPUTS)

    history = []
    best_val_loss = float("inf")
    best_weights = None
    for epoch in range(epochs):
        # Shuffle training set each epoch.
        idx = rng.permutation(len(x_train))
        x_tr = x_train[idx]
        y_tr = y_train[idx]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(x_tr), batch_size):
            xb = x_tr[start:start + batch_size]
            yb = y_tr[start:start + batch_size]
            # Forward
            y_hat, h = _forward(xb, W1, b1, W2, b2)
            err = y_hat - yb           # (B, 3)
            loss = float(np.mean(err ** 2))
            epoch_loss += loss
            n_batches += 1
            # Backward (manual; MSE over batch).
            bsize = xb.shape[0]
            grad_y = 2 * err / bsize     # (B, 3)
            grad_W2 = h.T @ grad_y       # (H, 3)
            grad_b2 = grad_y.sum(0)      # (3,)
            grad_h = grad_y @ W2.T       # (B, H)
            grad_pre = grad_h * (1 - h ** 2)  # tanh derivative
            grad_W1 = xb.T @ grad_pre    # (10, H)
            grad_b1 = grad_pre.sum(0)    # (H,)
            # SGD update.
            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2

        train_loss = epoch_loss / max(1, n_batches)
        # Validation pass.
        y_val_hat, _ = _forward(x_val, W1, b1, W2, b2)
        val_loss = float(np.mean((y_val_hat - y_val) ** 2))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    if best_weights is None:
        best_weights = (W1, b1, W2, b2)
    W1, b1, W2, b2 = best_weights

    # Verify output range AFTER training — must be within clamps so the
    # iter-001 hard-clamp safety contract is preserved.
    y_train_hat, _ = _forward(x_train, W1, b1, W2, b2)
    y_val_hat, _ = _forward(x_val, W1, b1, W2, b2)
    train_pred_range = {
        "min": [float(y_train_hat[:, i].min()) for i in range(DEFAULT_N_OUTPUTS)],
        "max": [float(y_train_hat[:, i].max()) for i in range(DEFAULT_N_OUTPUTS)],
    }
    val_pred_range = {
        "min": [float(y_val_hat[:, i].min()) for i in range(DEFAULT_N_OUTPUTS)],
        "max": [float(y_val_hat[:, i].max()) for i in range(DEFAULT_N_OUTPUTS)],
    }

    mlp = TrackerResidualMLP(W1=W1, b1=b1, W2=W2, b2=b2)
    mlp.to_npz(out_path)

    summary = {
        "samples_total": int(n),
        "samples_train": int(len(x_train)),
        "samples_val": int(len(x_val)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_val_loss": float(history[-1]["val_loss"]),
        "best_val_loss": float(best_val_loss),
        "train_pred_range": train_pred_range,
        "val_pred_range": val_pred_range,
        "target_clamp_rad": _CLAMP_RAD,
        "target_clamp_thrust": _CLAMP_THRUST,
        "out_path": str(out_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train the TrackerResidualMLP")
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = train(
        Path(args.dataset),
        out_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
