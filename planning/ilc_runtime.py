"""
Iterative Learning Control (ILC) runtime loader.

Research_topics_2.md C3 (PyBullet-native ILC, Schoellig 2012; Bristow &
Alleyne 2007; Freeman 2025): learn the systematic cross-track offset
between the planned trajectory and the actual PyBullet CF2X tracking
response off-line, then inject it as a cached acceleration feedforward
at runtime via ``GPDDrone.step(target_acc=...)``.

This module owns the *runtime* half of that loop: loading a pre-computed
offset table from JSON and interpolating it at arbitrary query times so
the trajectory optimizer can populate ``TrajectoryPoint.ff_acceleration``
with state-specific, Q-filter-smoothed values.

The offline calibrator (which runs visual_demo headless 5-8 times with
progressive offset updates and Butterworth Q-filtering) will land in a
future iteration. Until then this module is the empty-table-safe
infrastructure ``PlannerConfig.ilc_table_path`` flips on.

JSON schema (v1)
----------------
    {
        "schema_version": 1,
        "generated_at": "2026-04-20T00:00:00Z",
        "race_config_hash": "sha256-of-race_01.json",
        "n_iterations": 6,
        "q_filter": {"kind": "butterworth_lowpass", "order": 2, "cutoff_hz": 2.0},
        "samples": [
            {"t": 0.00, "ff_acc": [0.00, 0.00, 0.00]},
            {"t": 0.01, "ff_acc": [0.12, -0.03, 0.01]},
            ...
        ]
    }

Times are trajectory-relative seconds; ``ff_acc`` is a (x, y, z) tuple
in m/s² in the world frame (the same frame ``GPDDrone.step`` expects).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


# The JSON layout is versioned so the offline calibrator can evolve
# (e.g., add per-axis gains) without silently breaking the runtime.
SUPPORTED_SCHEMA_VERSIONS = (1,)


@dataclass
class ILCTable:
    """Time-indexed acceleration feedforward table loaded from JSON.

    Instances are cheap to query: ``get_ff_acceleration(t)`` runs a
    binary search over a cached numpy array of timestamps, so the hot
    loop cost is one ``searchsorted`` + 3 linear interpolations — not
    measurable against the ~60 µs control tick in practice.
    """

    times: np.ndarray          # shape (N,), strictly increasing
    ff_accelerations: np.ndarray  # shape (N, 3), world-frame m/s²
    total_time: float
    metadata: dict

    @classmethod
    def load_from_json(cls, path: str) -> "ILCTable":
        with open(path) as f:
            data = json.load(f)

        version = int(data.get("schema_version", 1))
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"ILC JSON at {path}: schema_version={version} not in "
                f"{SUPPORTED_SCHEMA_VERSIONS}"
            )

        samples = data.get("samples", [])
        if not samples:
            raise ValueError(f"ILC JSON at {path}: samples is empty")

        times = np.asarray([float(s["t"]) for s in samples], dtype=float)
        accels = np.asarray(
            [list(s["ff_acc"]) for s in samples], dtype=float
        )

        if times.ndim != 1 or accels.ndim != 2 or accels.shape[1] != 3:
            raise ValueError(
                f"ILC JSON at {path}: expected times shape (N,), "
                f"ff_acc shape (N, 3); got {times.shape} / {accels.shape}"
            )

        if not np.all(np.diff(times) > 0):
            raise ValueError(
                f"ILC JSON at {path}: times must be strictly increasing"
            )

        metadata = {
            k: v for k, v in data.items() if k not in {"samples"}
        }

        return cls(
            times=times,
            ff_accelerations=accels,
            total_time=float(times[-1]),
            metadata=metadata,
        )

    def get_ff_acceleration(self, t: float) -> Tuple[float, float, float]:
        """Linear-interpolated feedforward acceleration at query time ``t``.

        Clamps to endpoint values outside the calibrated range (standard
        ILC practice: don't extrapolate a learned bias past its support).
        """
        if t <= self.times[0]:
            a = self.ff_accelerations[0]
            return (float(a[0]), float(a[1]), float(a[2]))
        if t >= self.times[-1]:
            a = self.ff_accelerations[-1]
            return (float(a[0]), float(a[1]), float(a[2]))

        idx = int(np.searchsorted(self.times, t, side="right"))
        t0 = self.times[idx - 1]
        t1 = self.times[idx]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        a0 = self.ff_accelerations[idx - 1]
        a1 = self.ff_accelerations[idx]
        lerp = a0 + alpha * (a1 - a0)
        return (float(lerp[0]), float(lerp[1]), float(lerp[2]))


def try_load_ilc_table(path: str) -> "ILCTable | None":
    """Load an ILC table, or return ``None`` if the path is empty/missing.

    Callers flip the feature on by setting ``PlannerConfig.ilc_table_path``
    to a JSON path; this helper keeps the path-empty case silent so the
    default PlannerConfig (which points at the empty string) is a safe
    no-op rather than a crash.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ILC table path does not exist: {path}")
    return ILCTable.load_from_json(str(p))
