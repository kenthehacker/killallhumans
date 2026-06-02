"""
Lightweight learned tracker residual (iter-031, supersedes iter-001 A15).

A 12 → 64 → 3 MLP with tanh activation, numpy-only forward pass.
Inputs (12 features, order pinned): pos_err_xyz, vel_err_xyz,
ref_accel_xyz, thrust_normalised, sin(yaw_des), cos(yaw_des).
Outputs are bounded residual adjustments to (roll, pitch, thrust).
The GeometricTracker applies the output AFTER hard-clamping it to
±0.05 rad / ±0.05 thrust, so even a corrupted weight file cannot push
commands beyond ~2.9° / 5% thrust.

Design choices:
- numpy-only (no torch import at inference time): keeps the control loop
  under the 100 Hz / <1 ms budget and avoids a runtime dep tax.
- Off-by-default: `TrackerConfig.use_residual=False`. When True but no
  weights path is given, falls back to zero-init weights → identical to
  baseline behaviour (additive safety net on top of the off-switch).
- Hard clamps at the consumer (mpc_tracker), not inside the model. This
  keeps the model itself stateless and easy to validate independently.

Research backing:
- "On Your Own" (Romero 2025): residual feedforward stacked on classical
  geometric tracker.
- "Leveling the Playing Field" (Kunapuli 2025): feedforward is the most
  impactful single fix on geometric controllers.
- NGTC (Pries 2025): small learned residuals beat hand-tuned gains by 10-30%.
- Safe-RL with hard projection (Berkeley 2024): output clamp at the
  consumer is the cheapest safety guarantee for residual-style ML.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Default architecture. Inputs come from the GeometricTracker:
#   pos_err_xyz (3) + vel_err_xyz (3) + ref_accel_xyz (3) + thrust_norm (1)
#   + sin(yaw_des) + cos(yaw_des) = 12 features (iter-031).
# Output: (delta_roll, delta_pitch, delta_thrust).
#
# Iter-031 raised n_inputs 10 → 12. The trainer's BC-oracle target rotates
# by yaw_des; without sin/cos features the MLP can't represent the
# heading-dependent correction. iter-027 weights (10-D) are not loadable
# under the new schema — they were /tmp scratch and not committed.
DEFAULT_N_INPUTS: int = 12
DEFAULT_N_HIDDEN: int = 64
DEFAULT_N_OUTPUTS: int = 3


@dataclass
class TrackerResidualMLP:
    """A tiny MLP with explicit weights — easy to construct, test, ship."""
    W1: np.ndarray   # (n_inputs, n_hidden)
    b1: np.ndarray   # (n_hidden,)
    W2: np.ndarray   # (n_hidden, n_outputs)
    b2: np.ndarray   # (n_outputs,)
    # Iter-031: optional input normalization stored alongside weights.
    # When set, forward standardises x via (x - feat_mean) / feat_std.
    # `None` → identity (zero_init, untrained, or backward-compat).
    feat_mean: Optional[np.ndarray] = None
    feat_std: Optional[np.ndarray] = None
    # Iter-031: optional output activation. When `output_clamp` is set,
    # the raw output is squashed via `output_clamp * tanh(raw / output_clamp)`,
    # producing a smooth bounded prediction within ±output_clamp by
    # construction. The runtime clamp at the consumer still applies on
    # top (defence in depth). zero_init's output remains zero.
    # Shape: (n_outputs,) — per-channel clamp.
    output_clamp: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Force float64 — keeps deterministic across platforms.
        self.W1 = np.asarray(self.W1, dtype=np.float64)
        self.b1 = np.asarray(self.b1, dtype=np.float64)
        self.W2 = np.asarray(self.W2, dtype=np.float64)
        self.b2 = np.asarray(self.b2, dtype=np.float64)
        if self.W1.ndim != 2:
            raise ValueError(f"W1 must be 2D, got shape {self.W1.shape}")
        if self.W2.ndim != 2:
            raise ValueError(f"W2 must be 2D, got shape {self.W2.shape}")
        if self.W1.shape[1] != self.b1.shape[0]:
            raise ValueError(
                f"W1 hidden dim {self.W1.shape[1]} != b1 dim {self.b1.shape[0]}"
            )
        if self.W2.shape[0] != self.b1.shape[0]:
            raise ValueError(
                f"W2 input dim {self.W2.shape[0]} != hidden dim {self.b1.shape[0]}"
            )
        if self.W2.shape[1] != self.b2.shape[0]:
            raise ValueError(
                f"W2 output dim {self.W2.shape[1]} != b2 dim {self.b2.shape[0]}"
            )
        if self.feat_mean is not None or self.feat_std is not None:
            if self.feat_mean is None or self.feat_std is None:
                raise ValueError("feat_mean and feat_std must both be set or both None")
            self.feat_mean = np.asarray(self.feat_mean, dtype=np.float64).reshape(-1)
            self.feat_std = np.asarray(self.feat_std, dtype=np.float64).reshape(-1)
            if self.feat_mean.shape != (self.n_inputs,):
                raise ValueError(
                    f"feat_mean shape {self.feat_mean.shape} != ({self.n_inputs},)"
                )
            if self.feat_std.shape != (self.n_inputs,):
                raise ValueError(
                    f"feat_std shape {self.feat_std.shape} != ({self.n_inputs},)"
                )
        if self.output_clamp is not None:
            self.output_clamp = np.asarray(self.output_clamp, dtype=np.float64).reshape(-1)
            if self.output_clamp.shape != (self.n_outputs,):
                raise ValueError(
                    f"output_clamp shape {self.output_clamp.shape} != ({self.n_outputs},)"
                )
            if not np.all(self.output_clamp > 0):
                raise ValueError("output_clamp values must be > 0")

    @property
    def n_inputs(self) -> int:
        return int(self.W1.shape[0])

    @property
    def n_hidden(self) -> int:
        return int(self.W1.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.W2.shape[1])

    @classmethod
    def zero_init(
        cls,
        n_inputs: int = DEFAULT_N_INPUTS,
        n_hidden: int = DEFAULT_N_HIDDEN,
        n_outputs: int = DEFAULT_N_OUTPUTS,
    ) -> "TrackerResidualMLP":
        """Construct with all-zero weights. Forward pass returns the zero
        vector — useful as a safety baseline when no trained weights exist."""
        return cls(
            W1=np.zeros((n_inputs, n_hidden), dtype=np.float64),
            b1=np.zeros(n_hidden, dtype=np.float64),
            W2=np.zeros((n_hidden, n_outputs), dtype=np.float64),
            b2=np.zeros(n_outputs, dtype=np.float64),
        )

    @classmethod
    def random_init(
        cls,
        seed: int = 0,
        n_inputs: int = DEFAULT_N_INPUTS,
        n_hidden: int = DEFAULT_N_HIDDEN,
        n_outputs: int = DEFAULT_N_OUTPUTS,
        scale: float = 0.05,
    ) -> "TrackerResidualMLP":
        """Glorot-ish init. Used by the trainer; not by inference paths."""
        rng = np.random.default_rng(seed)
        return cls(
            W1=rng.normal(0.0, scale, size=(n_inputs, n_hidden)),
            b1=np.zeros(n_hidden, dtype=np.float64),
            W2=rng.normal(0.0, scale, size=(n_hidden, n_outputs)),
            b2=np.zeros(n_outputs, dtype=np.float64),
        )

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "TrackerResidualMLP":
        """Load weights from an .npz file. Keys: W1, b1, W2, b2, and
        optionally feat_mean/feat_std (iter-031 normalization) and
        output_clamp (iter-031 output bound)."""
        with np.load(str(path)) as data:
            keys = set(data.files)
            feat_mean = data["feat_mean"] if "feat_mean" in keys else None
            feat_std = data["feat_std"] if "feat_std" in keys else None
            output_clamp = data["output_clamp"] if "output_clamp" in keys else None
            return cls(
                W1=data["W1"], b1=data["b1"],
                W2=data["W2"], b2=data["b2"],
                feat_mean=feat_mean, feat_std=feat_std,
                output_clamp=output_clamp,
            )

    def to_npz(self, path: Union[str, Path]) -> None:
        """Save weights to an .npz file. Includes feat_mean/feat_std iff
        normalization is set, and output_clamp iff bounded output is on."""
        payload = {
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
        }
        if self.feat_mean is not None:
            payload["feat_mean"] = self.feat_mean
            payload["feat_std"] = self.feat_std
        if self.output_clamp is not None:
            payload["output_clamp"] = self.output_clamp
        np.savez(str(path), **payload)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Single-sample forward pass. Returns (n_outputs,).

        If feat_mean/feat_std are set, the input is standardised before
        the first matmul. The standardisation parameters are SAVED with
        the weights, so the same scaling is applied at inference and at
        validation time.

        If output_clamp is set, the raw output is squashed via
        `output_clamp * tanh(raw / output_clamp)`, producing a smooth
        bounded prediction within ±output_clamp.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_inputs:
            raise ValueError(
                f"input dim {x.shape[0]} != model n_inputs {self.n_inputs}"
            )
        if self.feat_mean is not None:
            x = (x - self.feat_mean) / self.feat_std
        h = np.tanh(x @ self.W1 + self.b1)
        raw = h @ self.W2 + self.b2
        if self.output_clamp is not None:
            return self.output_clamp * np.tanh(raw / self.output_clamp)
        return raw


def save_feature_trace(trace, path: Union[str, Path]) -> None:
    """Persist `GeometricTracker.feature_trace` to .npz (v2 schema, iter-031).

    Each trace entry is a 13-tuple:
      (features_12d, roll_nom, pitch_nom, thrust_nom, pos_err_xyz,
       vel_err_xyz, pos_xyz, vel_xyz, yaw_des,
       ref_pos_xyz, ref_vel_xyz, ref_accel_xyz, accel_des_baseline_xyz)

    v2 fields beyond v1 (logged at the tracker so the trainer can compute
    a one-step BC oracle target without re-running the bench):
      - pos, vel: drone state at this step (world frame).
      - yaw_des: reference yaw used by the rotation in mpc_tracker.py:316.
      - ref_pos, ref_vel, ref_accel: trajectory sample at this step.
      - accel_des_baseline: PD output BEFORE the residual was added.

    File schema (v2):
      features              (N, 12)   float64
      roll_nom              (N,)      float64
      pitch_nom             (N,)      float64
      thrust_nom            (N,)      float64
      pos_err               (N, 3)    float64
      vel_err               (N, 3)    float64
      pos                   (N, 3)    float64
      vel                   (N, 3)    float64
      yaw_des               (N,)      float64
      ref_pos               (N, 3)    float64
      ref_vel               (N, 3)    float64
      ref_accel             (N, 3)    float64
      accel_des_baseline    (N, 3)    float64
      version               scalar    int (==2)

    track_id is added by the collector at concat time (separate field).
    """
    if not trace:
        raise ValueError("trace is empty; nothing to save")
    n = len(trace)
    features = np.empty((n, DEFAULT_N_INPUTS), dtype=np.float64)
    roll = np.empty(n, dtype=np.float64)
    pitch = np.empty(n, dtype=np.float64)
    thrust = np.empty(n, dtype=np.float64)
    pos_err = np.empty((n, 3), dtype=np.float64)
    vel_err = np.empty((n, 3), dtype=np.float64)
    pos = np.empty((n, 3), dtype=np.float64)
    vel = np.empty((n, 3), dtype=np.float64)
    yaw_des = np.empty(n, dtype=np.float64)
    ref_pos = np.empty((n, 3), dtype=np.float64)
    ref_vel = np.empty((n, 3), dtype=np.float64)
    ref_accel = np.empty((n, 3), dtype=np.float64)
    accel_des_baseline = np.empty((n, 3), dtype=np.float64)
    for i, entry in enumerate(trace):
        if len(entry) != 13:
            raise ValueError(
                f"trace entry {i} has {len(entry)} fields; v2 expects 13"
            )
        (feats, r, p, t, pe, ve, po, ve_act, yd, rp, rv, ra, adb) = entry
        features[i] = feats
        roll[i] = r
        pitch[i] = p
        thrust[i] = t
        pos_err[i] = pe
        vel_err[i] = ve
        pos[i] = po
        vel[i] = ve_act
        yaw_des[i] = yd
        ref_pos[i] = rp
        ref_vel[i] = rv
        ref_accel[i] = ra
        accel_des_baseline[i] = adb
    np.savez(
        str(path),
        features=features,
        roll_nom=roll,
        pitch_nom=pitch,
        thrust_nom=thrust,
        pos_err=pos_err,
        vel_err=vel_err,
        pos=pos,
        vel=vel,
        yaw_des=yaw_des,
        ref_pos=ref_pos,
        ref_vel=ref_vel,
        ref_accel=ref_accel,
        accel_des_baseline=accel_des_baseline,
        version=np.array(2, dtype=np.int64),
    )


def load_feature_trace(path: Union[str, Path]) -> dict:
    """Inverse of `save_feature_trace`. Returns a dict with the v2 keys.
    Raises ValueError if the version isn't 2."""
    with np.load(str(path)) as data:
        version = int(data["version"])
        if version != 2:
            raise ValueError(
                f"feature trace version {version} unsupported; expected 2"
            )
        return {
            "features": data["features"].copy(),
            "roll_nom": data["roll_nom"].copy(),
            "pitch_nom": data["pitch_nom"].copy(),
            "thrust_nom": data["thrust_nom"].copy(),
            "pos_err": data["pos_err"].copy(),
            "vel_err": data["vel_err"].copy(),
            "pos": data["pos"].copy(),
            "vel": data["vel"].copy(),
            "yaw_des": data["yaw_des"].copy(),
            "ref_pos": data["ref_pos"].copy(),
            "ref_vel": data["ref_vel"].copy(),
            "ref_accel": data["ref_accel"].copy(),
            "accel_des_baseline": data["accel_des_baseline"].copy(),
        }


def build_input_features(
    pos_err: np.ndarray,
    vel_err: np.ndarray,
    ref_accel: np.ndarray,
    thrust_normalized: float,
    yaw_des: float = 0.0,
) -> np.ndarray:
    """Pack the canonical 12-dim feature vector for the tracker residual.

    Order is locked: (ep_x, ep_y, ep_z, ev_x, ev_y, ev_z,
                      ref_ax, ref_ay, ref_az, thrust_norm,
                      sin_yaw, cos_yaw).
    Changing this order breaks every saved weight file — don't.

    yaw_des defaults to 0.0 so existing call sites that don't pass it
    yet still produce a well-defined vector. New call sites SHOULD pass
    the actual yaw_des (rotation reference used by the controller's
    small-angle map), since the BC-oracle target rotates by yaw_des and
    the model needs sin/cos features to represent that.
    """
    pe = np.asarray(pos_err, dtype=np.float64).reshape(-1)
    ve = np.asarray(vel_err, dtype=np.float64).reshape(-1)
    ra = np.asarray(ref_accel, dtype=np.float64).reshape(-1)
    if pe.size != 3 or ve.size != 3 or ra.size != 3:
        raise ValueError(
            "pos_err / vel_err / ref_accel must each be length 3; "
            f"got {pe.size}, {ve.size}, {ra.size}"
        )
    y = float(yaw_des)
    return np.concatenate([
        pe, ve, ra,
        np.array([float(thrust_normalized), np.sin(y), np.cos(y)]),
    ])
