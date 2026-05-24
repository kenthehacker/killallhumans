"""
Drone calibration harness — sim-to-AIGP (iter-001 A13, stub).

The AIGP competition drone is a 280 mm × 280 mm × 160 mm quad whose mass,
thrust ceiling, and drag are NOT specified in VADR-TS-002. The simulator
we use locally (PyBullet, ~Crazyflie CF2X dynamics) is a poor match.

This module fits thrust-per-mass and drag-per-mass ratios from MAVLink
telemetry samples so the controller / trajectory planner can be auto-
configured at race start. The identifier is intentionally tiny: a single
linear least-squares solve. No ML, no PyTorch, no warm-up time.

End-to-end validation against the DCL binary is deferred to iter ≥ 002
(blocker: no DCL binary on this worktree). The unit test here uses
synthetic samples to verify the identifier recovers seeded parameters.

Usage (deferred to iter ≥ 002):
    samples = collect_during_calibration_run(bridge, duration_s=10.0)
    cal = DroneCalibrator().identify_thrust_drag_ratios(samples)
    DroneCalibrator.write_calibration_json(cal, "drone_calibration.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np


# Gravity in NED — Z points DOWN, so gravity is +9.81 along +Z.
_G_MPS2_NED: float = 9.81


@dataclass
class CalibrationSample:
    """One telemetry tick used in the fit.

    Fields are world-frame NED. `thrust_normalized` is the offboard
    thrust command in [0, 1]; `velocity_z_world` is the drone's NED z
    velocity (positive = downward); `accel_z_world` is the measured z
    acceleration (positive = downward).
    """
    thrust_normalized: float
    velocity_z_world: float
    accel_z_world: float


@dataclass
class CalibrationResult:
    """Fitted parameters. Stored ratios let us delay knowing mass alone."""
    thrust_per_mass: float    # max_thrust_n / mass — units 1/s² when input is normalized to 1.0
    drag_per_mass: float      # drag_coefficient / mass — units 1/s
    n_samples: int
    rmse: float
    # Optional: if the caller has an independent estimate of mass, derived values follow.
    mass_kg: Optional[float] = None
    max_thrust_n: Optional[float] = None
    drag_coefficient: Optional[float] = None


class DroneCalibrator:
    """Fit thrust/drag ratios from MAVLink telemetry."""

    def __init__(self, gravity: float = _G_MPS2_NED):
        self.gravity = float(gravity)

    def identify_thrust_drag_ratios(
        self,
        samples: Iterable[CalibrationSample],
        assumed_mass_kg: Optional[float] = None,
    ) -> CalibrationResult:
        """Solve `a_z_world + g = k_t · u − k_d · v_z_world` via least squares.

        Where:
            k_t = max_thrust_n / mass      (acceleration produced per unit normalized thrust)
            k_d = drag_coefficient / mass  (drag deceleration per unit downward velocity)

        Identifies the two RATIOS only — separating mass from thrust
        requires an independent measurement (e.g., a known reference
        acceleration at zero thrust). If `assumed_mass_kg` is provided,
        we attribute the ratios to physical values for downstream use.
        """
        sample_list: List[CalibrationSample] = list(samples)
        n = len(sample_list)
        if n < 2:
            raise ValueError(f"need ≥ 2 calibration samples; got {n}")

        u = np.array([s.thrust_normalized for s in sample_list], dtype=np.float64)
        v = np.array([s.velocity_z_world for s in sample_list], dtype=np.float64)
        a = np.array([s.accel_z_world for s in sample_list], dtype=np.float64)
        y = a + self.gravity
        # X @ [k_t, k_d] = y  →  least squares.
        # Sign: a_z = k_t*u (downward thrust contribution... wait, in NED
        # thrust opposes gravity so thrust pushes UP = negative z. The
        # spec says HIGHRES_IMU is body NED with z-down, so a-down is
        # positive; for a hovering drone the IMU reads ~0 z-accel because
        # thrust balances gravity. The equation `a + g = k_t*u - k_d*v`
        # assumes positive thrust input produces NEGATIVE z (upward),
        # so we use -u in the design matrix. Equivalently we negate k_t
        # by flipping the sign here — keep the convention consistent with
        # the controller, which treats thrust as upward-positive.
        X = np.column_stack([-u, -v])
        coeffs, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        k_t, k_d = float(coeffs[0]), float(coeffs[1])
        y_hat = X @ coeffs
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

        result = CalibrationResult(
            thrust_per_mass=k_t,
            drag_per_mass=k_d,
            n_samples=n,
            rmse=rmse,
        )
        if assumed_mass_kg is not None:
            m = float(assumed_mass_kg)
            result.mass_kg = m
            result.max_thrust_n = k_t * m
            result.drag_coefficient = k_d * m
        return result

    @staticmethod
    def write_calibration_json(
        result: CalibrationResult, path: Union[str, Path],
    ) -> None:
        """Persist the fit so the controller / planner can consume it."""
        payload = {
            "thrust_per_mass_1_per_s2": result.thrust_per_mass,
            "drag_per_mass_1_per_s": result.drag_per_mass,
            "n_samples": result.n_samples,
            "rmse_mps2": result.rmse,
        }
        if result.mass_kg is not None:
            payload["mass_kg"] = result.mass_kg
            payload["max_thrust_n"] = result.max_thrust_n
            payload["drag_coefficient"] = result.drag_coefficient
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def read_calibration_json(
        path: Union[str, Path],
    ) -> CalibrationResult:
        """Reverse of write_calibration_json."""
        with open(path) as f:
            d = json.load(f)
        return CalibrationResult(
            thrust_per_mass=float(d["thrust_per_mass_1_per_s2"]),
            drag_per_mass=float(d["drag_per_mass_1_per_s"]),
            n_samples=int(d["n_samples"]),
            rmse=float(d["rmse_mps2"]),
            mass_kg=d.get("mass_kg"),
            max_thrust_n=d.get("max_thrust_n"),
            drag_coefficient=d.get("drag_coefficient"),
        )
