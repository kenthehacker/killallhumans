"""
Lightweight learned tracker residual (iter-001 A15).

A 10 → 64 → 3 MLP with tanh activation, numpy-only forward pass. Inputs are
position error, velocity error, reference acceleration, and the
gravity-compensated thrust (10 features). Outputs are bounded residual
adjustments to (roll, pitch, thrust). The GeometricTracker applies the
output AFTER hard-clamping it to ±0.05 rad / ±0.05 thrust, so even a
corrupted weight file cannot push commands beyond ~2.9° / 5% thrust.

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
from typing import Union

import numpy as np

# Default architecture. Inputs come from the GeometricTracker:
#   pos_err_xyz (3) + vel_err_xyz (3) + ref_accel_xyz (3) + thrust_norm (1)
# = 10 features. Output: (delta_roll, delta_pitch, delta_thrust).
DEFAULT_N_INPUTS: int = 10
DEFAULT_N_HIDDEN: int = 64
DEFAULT_N_OUTPUTS: int = 3


@dataclass
class TrackerResidualMLP:
    """A tiny MLP with explicit weights — easy to construct, test, ship."""
    W1: np.ndarray   # (n_inputs, n_hidden)
    b1: np.ndarray   # (n_hidden,)
    W2: np.ndarray   # (n_hidden, n_outputs)
    b2: np.ndarray   # (n_outputs,)

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
        """Load weights from an .npz file with keys W1, b1, W2, b2."""
        with np.load(str(path)) as data:
            return cls(
                W1=data["W1"], b1=data["b1"],
                W2=data["W2"], b2=data["b2"],
            )

    def to_npz(self, path: Union[str, Path]) -> None:
        """Save weights to an .npz file."""
        np.savez(
            str(path),
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Single-sample forward pass. Returns (n_outputs,)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_inputs:
            raise ValueError(
                f"input dim {x.shape[0]} != model n_inputs {self.n_inputs}"
            )
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def build_input_features(
    pos_err: np.ndarray,
    vel_err: np.ndarray,
    ref_accel: np.ndarray,
    thrust_normalized: float,
) -> np.ndarray:
    """Pack the canonical 10-dim feature vector for the tracker residual.

    Order is locked: (ep_x, ep_y, ep_z, ev_x, ev_y, ev_z,
                      ref_ax, ref_ay, ref_az, thrust_norm).
    Changing this order breaks every saved weight file — don't.
    """
    pe = np.asarray(pos_err, dtype=np.float64).reshape(-1)
    ve = np.asarray(vel_err, dtype=np.float64).reshape(-1)
    ra = np.asarray(ref_accel, dtype=np.float64).reshape(-1)
    if pe.size != 3 or ve.size != 3 or ra.size != 3:
        raise ValueError(
            "pos_err / vel_err / ref_accel must each be length 3; "
            f"got {pe.size}, {ve.size}, {ra.size}"
        )
    return np.concatenate([pe, ve, ra, np.array([float(thrust_normalized)])])
