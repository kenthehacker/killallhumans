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
        # Physics in NED (z-down, gravity = +g along +z):
        #   F_z = m·g − k_t_phys·u·m − k_d_phys·v_z·m   (thrust opposes gravity)
        #   a_z = g − k_t·u − k_d·v_z         where k_t = max_thrust/mass, k_d = drag/mass
        # Rearranging for the regression:
        #   g − a_z = k_t·u + k_d·v_z
        # Iter-001 review (Opus F1, BLOCKER) caught the earlier code's sign
        # error: it solved `a + g = -k_t·u - k_d·v_z` (off by 2g from the
        # physics), and the unit test generated synth data matching the
        # wrong equation, so the bug was invisible.
        y = self.gravity - a
        X = np.column_stack([u, v])
        coeffs, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        k_t, k_d = float(coeffs[0]), float(coeffs[1])
        y_hat = X @ coeffs
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

        # Positivity sanity check — a sign-flipped fit usually means the
        # caller swapped thrust convention or the bug above regressed.
        # k_d may legitimately be ≤ 0 for a perfect drag-free model, so
        # only guard k_t.
        if k_t <= 0.0:
            raise ValueError(
                f"recovered thrust_per_mass {k_t!r} is non-positive — likely a "
                "sign / convention mismatch in the sample feeder; expected "
                "samples in NED (gravity = +9.81 along +z) with positive "
                "thrust input producing UPWARD (negative-z) acceleration"
            )

        # Iter-002 review M4 (6/7 reviews MAJOR): a sign-flipped feeder
        # absorbs the missing intercept into k_t (large positive value)
        # without tripping the positivity guard. RMSE/|y| catches it —
        # a fit that aligns with the physics has RMSE small compared to
        # the typical |g − a_z|. A wrong-physics fit has RMSE ~ g/√3
        # (residual range −g to +g with uniform u), giving ratio ~0.18
        # vs ~0.005 for the right physics. Threshold 0.10 picks up gross
        # misfits with comfortable margin on real telemetry noise (which
        # rarely exceeds 0.5 m/s² RMSE on accel readings, well under 0.10
        # of typical |y| ≈ 10 m/s²).
        y_scale = float(np.mean(np.abs(y))) or 1e-3
        if rmse / y_scale > 0.10:
            raise ValueError(
                f"calibration fit is gross (rmse={rmse:.3f} m/s², "
                f"mean|y|={y_scale:.3f} m/s²) — likely a sign / convention "
                "mismatch in the sample feeder. Check that thrust_normalized "
                "is in [0,1], velocity_z_world is NED (+z down), and "
                "accel_z_world is the IMU's z-axis reading in the same NED frame."
            )

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
