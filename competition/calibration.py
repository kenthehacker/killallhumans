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
        """Reverse of write_calibration_json with basic schema validation.

        iter-003 (4/7 reviews MINOR): the previous version accepted any
        JSON object with the right keys, including physically impossible
        values (negative thrust, NaN). Now: required keys must be present,
        all numeric fields must be finite, and thrust_per_mass must be
        positive (drag may legitimately be ~0 for a frictionless model).
        """
        import math as _math

        required = (
            "thrust_per_mass_1_per_s2", "drag_per_mass_1_per_s",
            "n_samples", "rmse_mps2",
        )
        with open(path) as f:
            d = json.load(f)
        for k in required:
            if k not in d:
                raise ValueError(f"calibration JSON missing required key {k!r}")

        k_t = float(d["thrust_per_mass_1_per_s2"])
        k_d = float(d["drag_per_mass_1_per_s"])
        rmse = float(d["rmse_mps2"])
        n = int(d["n_samples"])
        if not (_math.isfinite(k_t) and _math.isfinite(k_d) and _math.isfinite(rmse)):
            raise ValueError(
                f"calibration JSON contains non-finite numerics: "
                f"k_t={k_t!r}, k_d={k_d!r}, rmse={rmse!r}"
            )
        if k_t <= 0.0:
            raise ValueError(
                f"calibration JSON has non-positive thrust_per_mass "
                f"({k_t!r}) — physical thrust must be upward-positive"
            )
        if n < 1:
            raise ValueError(f"calibration JSON has n_samples<1 ({n!r})")

        mass = d.get("mass_kg")
        max_thrust = d.get("max_thrust_n")
        drag = d.get("drag_coefficient")
        for label, val in (("mass_kg", mass),
                          ("max_thrust_n", max_thrust),
                          ("drag_coefficient", drag)):
            if val is not None and not _math.isfinite(float(val)):
                raise ValueError(f"calibration JSON has non-finite {label}: {val!r}")
        if mass is not None and float(mass) <= 0.0:
            raise ValueError(f"calibration JSON has non-positive mass_kg: {mass!r}")
        if max_thrust is not None and float(max_thrust) <= 0.0:
            raise ValueError(f"calibration JSON has non-positive max_thrust_n: {max_thrust!r}")

        return CalibrationResult(
            thrust_per_mass=k_t,
            drag_per_mass=k_d,
            n_samples=n,
            rmse=rmse,
            mass_kg=float(mass) if mass is not None else None,
            max_thrust_n=float(max_thrust) if max_thrust is not None else None,
            drag_coefficient=float(drag) if drag is not None else None,
        )


class OnlineDroneCalibrator:
    """Recursive (online) thrust/drag identifier — roadmap item #1.

    Streaming counterpart to `DroneCalibrator.identify_thrust_drag_ratios`.
    Ingests `CalibrationSample`-style ticks ONE AT A TIME and maintains
    running estimates of the SAME two ratios the batch solver fits, using
    the EXACT same NED physics / regression:

        g − a_z = k_t · u + k_d · v_z          (regressors φ = [u, v_z])

    where k_t = max_thrust / mass and k_d = drag / mass. (The batch method
    names the second term `+ k_d·v_z`; this estimator is byte-for-byte the
    same model, so an `OnlineDroneCalibrator` fed the batch's samples
    converges to the batch lstsq solution.)

    Algorithm: recursive least squares (RLS) with an exponential forgetting
    factor λ (Ljung, *System Identification* 1999; the canonical adaptive
    estimator). Per tick, given regressor φ and target y:

        e = y − φᵀ θ                              (a-priori error)
        K = P φ / (λ + φᵀ P φ)                     (Kalman-style gain)
        θ ← θ + K e
        P ← (P − K φᵀ P) / λ                       (forget old data)

    λ = 1 → ordinary growing-memory least squares (no forgetting). λ < 1
    weights recent ticks more, so the estimate TRACKS slow drift (battery
    sag, prop wear, payload change) at the cost of more variance — the
    classic forgetting-factor bias/variance knob. Default 0.99 ≈ a ~100-
    sample effective window, matching the batch test's sample counts.

    Cost per tick is a fixed handful of 2×2 / 2-vector ops — no allocation
    growth, numpy-only, comfortably inside the <1 ms / >100 Hz control-loop
    budget, so a live MAVLink loop can call `update()` every telemetry tick
    (see `update_from_telemetry`). This estimator does NOT require the DCL
    binary; end-to-end DCL validation stays deferred to iter ≥ 002 exactly
    as the batch path documents. Validated here only against synthetic
    streams (see tests/test_calibration.py).

    Why RLS over LMS here: the deep-research report (2026-06-16) notes
    Smeur et al. prefer LMS for the *INDI control-effectiveness* problem
    because a finite-window estimator "forgets" outside its window — but
    that is a design rationale for online-G, not a theorem, and the report
    explicitly allows "variable-forgetting RLS." For a 2-parameter, well-
    conditioned thrust/drag fit RLS converges faster from few samples and
    its forgetting factor gives the drift-tracking the report asks for.
    """

    def __init__(
        self,
        gravity: float = _G_MPS2_NED,
        forgetting_factor: float = 0.99,
        k_t_init: float = 0.0,
        k_d_init: float = 0.0,
        covariance_init: float = 1.0e3,
        max_covariance: float = 1.0e6,
        min_k_t: float = 1.0e-6,
    ):
        """
        Args:
            gravity: NED gravity magnitude (+z down). Matches the batch path.
            forgetting_factor: λ in (0, 1]. 1.0 = growing-memory LS (no
                forgetting); <1 tracks drift. Default 0.99 (~100-sample
                effective memory). Must be in (0, 1].
            k_t_init, k_d_init: initial parameter guesses. 0.0 is a neutral
                start (the large covariance lets the first few ticks move
                the estimate freely); pass a `drone_spec`-derived prior to
                warm-start.
            covariance_init: diagonal value of the initial covariance P₀.
                Large (1e3) ⇒ low confidence in the init ⇒ fast initial
                adaptation. Small ⇒ trust the init / adapt slowly.
            max_covariance: upper bound on tr(P), the COVARIANCE-WINDUP guard.
                With a forgetting factor (λ<1) and POORLY-EXCITED data (e.g. a
                near-constant thrust on a smooth racing line — the exact
                under-excitation risk the report flags), the `/λ` step keeps
                inflating P every tick because no new information arrives,
                and P eventually overflows → NaN gain. Capping tr(P) (rescale
                P when tr(P) > max_covariance) is the standard RLS safeguard;
                it bounds the adaptation gain so a quiet stream can't blow the
                estimator up, while leaving normally-excited runs untouched
                (tr(P) settles well below the cap). Must exceed covariance_init.
            min_k_t: positivity floor for the reported thrust_per_mass.
                Mirrors the batch path's "guard k_t only" positivity idea
                (drag k_d may legitimately be ~0 for a drag-free model, so
                it is NOT floored). The INTERNAL θ is left untouched so the
                RLS recursion stays numerically consistent; only the value
                surfaced in the `CalibrationResult` is clamped.
        """
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError(
                f"forgetting_factor must be in (0, 1]; got {forgetting_factor!r}"
            )
        if covariance_init <= 0.0:
            raise ValueError(
                f"covariance_init must be positive; got {covariance_init!r}"
            )
        if max_covariance <= covariance_init:
            raise ValueError(
                f"max_covariance ({max_covariance!r}) must exceed "
                f"covariance_init ({covariance_init!r})"
            )
        self.gravity = float(gravity)
        self.lam = float(forgetting_factor)
        self.max_covariance = float(max_covariance)
        self.min_k_t = float(min_k_t)
        # θ = [k_t, k_d]ᵀ. P = parameter covariance (2×2).
        self._theta = np.array([float(k_t_init), float(k_d_init)], dtype=np.float64)
        self._P = np.eye(2, dtype=np.float64) * float(covariance_init)
        # Running diagnostics. _sse / _sy2 accumulate the FORGETTING-WEIGHTED
        # a-priori squared error and target energy so `rmse` reflects the
        # same effective window as the parameter estimate (a plain mean would
        # be dominated by stale ticks once thousands have streamed by).
        self.n_samples: int = 0
        self._sse: float = 0.0   # Σ λ^(n-i) e_i²   (a-priori prediction error)
        self._sw: float = 0.0    # Σ λ^(n-i)        (effective sample count)

    @property
    def thrust_per_mass(self) -> float:
        """Current k_t estimate, floored at `min_k_t` (positivity guard)."""
        return max(self.min_k_t, float(self._theta[0]))

    @property
    def drag_per_mass(self) -> float:
        """Current k_d estimate (NOT floored — drag may be ~0)."""
        return float(self._theta[1])

    @property
    def rmse(self) -> float:
        """Forgetting-weighted RMS of the a-priori prediction error (m/s²).

        This is the error BEFORE each tick's update, so it reflects how well
        the current model predicts incoming data — the online analogue of the
        batch fit's residual RMSE. ~0 until the first update.
        """
        if self._sw <= 0.0:
            return 0.0
        return float(np.sqrt(self._sse / self._sw))

    def update(self, sample: CalibrationSample) -> CalibrationResult:
        """Ingest one telemetry tick; return the updated running estimate.

        Performs a single RLS step on the batch model
        `g − a_z = k_t·u + k_d·v_z` and returns a `CalibrationResult`
        snapshot (same dataclass the batch path returns) so callers can use
        the two paths interchangeably. The returned `thrust_per_mass` is
        positivity-guarded; `n_samples` is the cumulative tick count and
        `rmse` is the forgetting-weighted a-priori error.

        Cheap and allocation-light — safe to call every control tick.
        """
        u = float(sample.thrust_normalized)
        v = float(sample.velocity_z_world)
        a = float(sample.accel_z_world)
        phi = np.array([u, v], dtype=np.float64)
        y = self.gravity - a

        # A-priori error (prediction BEFORE this tick updates θ).
        e = y - float(phi @ self._theta)

        # RLS gain. Denominator λ + φᵀPφ > 0 for P ≻ 0, λ > 0.
        Pphi = self._P @ phi
        denom = self.lam + float(phi @ Pphi)
        K = Pphi / denom

        # Parameter + covariance update.
        self._theta = self._theta + K * e
        # P ← (P − K φᵀ P)/λ. Symmetrise to fight float drift over long runs
        # (a classic RLS numerical safeguard — keeps P a valid covariance).
        self._P = (self._P - np.outer(K, Pphi)) / self.lam
        self._P = 0.5 * (self._P + self._P.T)

        # Covariance-windup guard: under low excitation the `/λ` step inflates
        # P without bound (nothing new to learn) until it overflows to NaN.
        # Cap tr(P) by rescaling — bounds the gain so a quiet/constant stream
        # can't blow the estimator up. No-op on normally-excited data.
        trace = float(self._P[0, 0] + self._P[1, 1])
        if trace > self.max_covariance:
            self._P *= self.max_covariance / trace

        # Forgetting-weighted error/energy accumulators for `rmse`.
        self._sse = self.lam * self._sse + e * e
        self._sw = self.lam * self._sw + 1.0
        self.n_samples += 1

        return self.result()

    def update_from_telemetry(
        self,
        thrust_normalized: float,
        velocity_z_world: float,
        accel_z_world: float,
    ) -> CalibrationResult:
        """Convenience hook for a live MAVLink loop.

        A real offboard loop holds the last commanded normalized thrust and
        a `TelemetryState` (NED velocity + IMU accel, both z-down). It can
        feed this estimator every tick WITHOUT building a `CalibrationSample`:

            cal = OnlineDroneCalibrator(k_t_init=ds.DEFAULT_MAX_THRUST_N / ds.DEFAULT_MASS_KG)
            ...
            telem = bridge.latest_telemetry          # TelemetryState (NED)
            res = cal.update_from_telemetry(
                thrust_normalized=last_cmd_thrust,    # the [0,1] you sent
                velocity_z_world=telem.velocity_ned[2],
                accel_z_world=telem.imu.accel[2],     # IMU z (NED, +down)
            )
            tracker.config.drag_ff_coeff = res.drag_per_mass   # feed roadmap #1 FF

        Wiring this into `competition/aigp_mavlink.py`'s telemetry callback is
        left for on-sim work (it needs the DCL binary to validate end-to-end
        and a decision on excitation gating — a smooth racing line may under-
        excite k_d, the same excitation caveat the report flags for online-G).
        """
        return self.update(
            CalibrationSample(
                thrust_normalized=float(thrust_normalized),
                velocity_z_world=float(velocity_z_world),
                accel_z_world=float(accel_z_world),
            )
        )

    def result(self) -> CalibrationResult:
        """Snapshot the current estimate as a `CalibrationResult` (no update)."""
        return CalibrationResult(
            thrust_per_mass=self.thrust_per_mass,
            drag_per_mass=self.drag_per_mass,
            n_samples=self.n_samples,
            rmse=self.rmse,
        )
