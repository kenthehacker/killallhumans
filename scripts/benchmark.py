#!/usr/bin/env python3
"""
Headless benchmark — runs the full pipeline and outputs structured JSON metrics.

Designed for autonomous AI agent iteration: run → parse JSON → identify issues → fix → repeat.

Modes:
  --mode unit       Run unit tests only (no external dependency)
  --mode synthetic  Run synthetic simulation (pure Python, no PyBullet)
  --mode sim        Run full PyBullet simulation headless
  --mode full       Run unit + synthetic + sim (default)

Output:
  Prints a JSON object to stdout with all metrics. Human-readable summary to stderr.
  Exit code 0 = all thresholds met, 1 = at least one failure.

Usage:
    python3 scripts/benchmark.py                          # full benchmark
    python3 scripts/benchmark.py --mode unit              # unit tests only
    python3 scripts/benchmark.py --mode synthetic         # synthetic sim (no PyBullet)
    python3 scripts/benchmark.py --mode sim --duration 30 # PyBullet sim
    python3 scripts/benchmark.py --json-only              # suppress stderr summary
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
_SCRIPT_MODULE_STARTED = time.perf_counter()
import traceback
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Quality thresholds — an AI agent should aim to improve these
# ---------------------------------------------------------------------------
THRESHOLDS = MappingProxyType({
    "unit_tests_pass_rate": 1.0,          # 100% unit tests must pass
    # The current evaluator measures error to the scheduled/commanded
    # reference, not
    # the globally nearest path point. A fresh seven-track calibration pins
    # 1.0 m average / 3.5 m instantaneous while the separate nearest-path
    # diagnostics stay below 0.60 m on completed production tracks.
    "max_avg_tracking_error_m": 1.0,
    "max_max_tracking_error_m": 3.5,
    "max_ekf_uncertainty_m": 0.5,         # aspirational target (tightened from 1.0)
    "min_loop_hz": 100,                   # minimum control loop frequency
    "min_gate_pass_rate": 1.0,            # Phase 1: require full gate completion (was 0.8)
    "max_total_time_s": 30.0,             # must finish within 30s
    "no_crash": True,                     # must not crash
})

EVALUATOR_VERSION = "synthetic-v4-exact-state-time"
PYBULLET_EVALUATOR_VERSION = "pybullet-v3-plane-crossing"
BENCHMARK_RESULT_SCHEMA = "benchmark-result-v2"
COMPARISON_SERIES = "prepared-benchmark-v4-exact-state-time"


def _strict_json_object(payload: str, *, context: str) -> Dict[str, Any]:
    """Decode configuration JSON without duplicate keys or numeric extensions."""

    def unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context}: duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{context}: non-standard JSON numeric constant: {value}")

    decoded = json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise TypeError(f"{context}: root must be a JSON object")
    return decoded


def _read_strict_json_object(path: Path, *, context: str) -> Dict[str, Any]:
    return _strict_json_object(path.read_text(encoding="utf-8"), context=context)


def _json_numpy_default(value: Any) -> Any:
    """Convert NumPy trace values without permitting non-finite JSON."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _restore_tracker_feature_arrays(result: Dict[str, Any]) -> Dict[str, Any]:
    """Restore the public trace feature-vector type after JSON normalization.

    Artifact payloads deliberately encode arrays as lists.  The in-memory
    benchmark API, however, has historically exposed the first field of every
    tracker trace entry as a NumPy vector.  Restore that field on both cold and
    cache-hit paths so callers see one stable contract.
    """

    trace = result.get("tracker_feature_trace")
    if not isinstance(trace, list) or not trace:
        return result
    from control.learned_residual import DEFAULT_N_INPUTS

    for entry in trace:
        if not isinstance(entry, list) or not entry:
            continue
        try:
            features = np.asarray(entry[0], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if features.shape == (DEFAULT_N_INPUTS,) and np.all(np.isfinite(features)):
            entry[0] = features
    return result


def _finite_real_array(value: Any, shape: Tuple[int, ...]) -> bool:
    """Return whether an untrusted JSON value is a finite real array."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        array.shape == shape
        and (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        )
        and np.all(np.isfinite(array))
    )


def _finite_real_scalar(value: Any, *, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return False
    resolved = float(value)
    return math.isfinite(resolved) and (not nonnegative or resolved >= 0.0)


def _exact_finite_float(
    name: str,
    value: Any,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Normalize a public numeric argument without bool/string coercion."""

    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and resolved <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _exact_nonnegative_int(name: str, value: Any) -> int:
    """Normalize a public integer argument without bool/float coercion."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _exact_bool(name: str, value: Any) -> bool:
    """Reject truthy strings/integers at public evaluator boundaries."""

    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return value


def _exact_mapping(name: str, value: Any) -> Dict[str, Any]:
    """Copy a configuration mapping while rejecting ambiguous key/value shapes."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    invalid_keys = [key for key in value if type(key) is not str]
    if invalid_keys:
        raise TypeError(f"{name} keys must be exact strings")
    return dict(value)


def _normalize_ilc_global(value: Any) -> Dict[str, Any]:
    """Resolve ILC hyperparameters without JSON truthiness/numeric coercion."""

    resolved = _exact_mapping("ILC global config", value)
    allowed = {
        "alpha",
        "max_iterations",
        "smoothing_sigma",
        "max_correction_m",
        "convergence_threshold",
        "filter_cutoff_hz",
        "momentum_gamma",
        "blend_steps",
    }
    unknown = sorted(set(resolved) - allowed)
    missing = sorted(allowed - set(resolved))
    if unknown:
        raise ValueError("unknown ILC global key(s): " + ", ".join(unknown))
    if missing:
        raise ValueError("missing ILC global key(s): " + ", ".join(missing))

    normalized = {
        name: _exact_finite_float(f"ILC global.{name}", resolved[name])
        for name in (
            "alpha",
            "smoothing_sigma",
            "max_correction_m",
            "convergence_threshold",
            "filter_cutoff_hz",
            "momentum_gamma",
        )
    }
    normalized["max_iterations"] = _exact_nonnegative_int(
        "ILC global.max_iterations", resolved["max_iterations"]
    )
    normalized["blend_steps"] = _exact_nonnegative_int(
        "ILC global.blend_steps", resolved["blend_steps"]
    )
    if not 0.0 < normalized["alpha"] <= 1.0:
        raise ValueError("ILC global.alpha must be within (0, 1]")
    if normalized["max_iterations"] <= 0:
        raise ValueError("ILC global.max_iterations must be strictly positive")
    if normalized["smoothing_sigma"] <= 0.0:
        raise ValueError("ILC global.smoothing_sigma must be strictly positive")
    if normalized["max_correction_m"] < 0.0:
        raise ValueError("ILC global.max_correction_m must be non-negative")
    if normalized["convergence_threshold"] < 0.0:
        raise ValueError("ILC global.convergence_threshold must be non-negative")
    if normalized["filter_cutoff_hz"] <= 0.0:
        raise ValueError("ILC global.filter_cutoff_hz must be strictly positive")
    if not 0.0 <= normalized["momentum_gamma"] < 1.0:
        raise ValueError("ILC global.momentum_gamma must be within [0, 1)")
    return normalized


def _normalize_ilc_section_overrides(
    value: Any, *, override_format: Any, total_steps: int
) -> Optional[List[Tuple[Any, ...]]]:
    """Validate and resolve optional fractional/step ILC section tuples."""

    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError("ILC section_overrides must be a non-empty list or null")
    if type(override_format) is not str or override_format not in {
        "auto",
        "fractions",
        "steps",
    }:
        raise ValueError(
            "ILC section_overrides_format must be 'auto', 'fractions', or 'steps'"
        )

    raw_sections: List[List[Any]] = []
    for index, section in enumerate(value):
        if not isinstance(section, (list, tuple)) or not 3 <= len(section) <= 7:
            raise ValueError(f"ILC section[{index}] must contain 3 to 7 values")
        raw_sections.append(list(section))

    endpoint_values = [
        _exact_finite_float(f"ILC section[{index}].{name}", section[position])
        for index, section in enumerate(raw_sections)
        for position, name in ((0, "start"), (1, "end"))
    ]
    fractional = override_format == "fractions" or (
        override_format == "auto"
        and max(endpoint_values[0::2]) < 2.0
        and max(endpoint_values[1::2]) <= 1.0 + 1e-6
    )

    normalized: List[Tuple[Any, ...]] = []
    for index, section in enumerate(raw_sections):
        raw_start, raw_end = section[:2]
        if fractional:
            start_fraction = _exact_finite_float(
                f"ILC section[{index}].start", raw_start, nonnegative=True
            )
            end_fraction = _exact_finite_float(
                f"ILC section[{index}].end", raw_end, strictly_positive=True
            )
            if start_fraction >= end_fraction or end_fraction > 1.0:
                raise ValueError(
                    f"ILC section[{index}] fractional bounds must satisfy "
                    "0 <= start < end <= 1"
                )
            start = int(start_fraction * total_steps)
            end = int(end_fraction * total_steps)
        else:
            start = _exact_nonnegative_int(
                f"ILC section[{index}].start", raw_start
            )
            end = _exact_nonnegative_int(f"ILC section[{index}].end", raw_end)
            if start >= end or end > total_steps:
                raise ValueError(
                    f"ILC section[{index}] step bounds must satisfy "
                    f"0 <= start < end <= {total_steps}"
                )
        if start >= end:
            raise ValueError(
                f"ILC section[{index}] collapses to an empty resolved step range"
            )

        alpha = _exact_finite_float(f"ILC section[{index}].alpha", section[2])
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"ILC section[{index}].alpha must be within (0, 1]")
        parameters: List[Any] = [start, end, alpha]
        if len(section) > 3:
            maximum = _exact_finite_float(
                f"ILC section[{index}].max_correction_m",
                section[3],
                nonnegative=True,
            )
            parameters.append(maximum)
        if len(section) > 4:
            cutoff = _exact_finite_float(
                f"ILC section[{index}].filter_cutoff_hz",
                section[4],
                strictly_positive=True,
            )
            parameters.append(cutoff)
        if len(section) > 5:
            velocity_scale = _exact_finite_float(
                f"ILC section[{index}].velocity_scale",
                section[5],
                nonnegative=True,
            )
            parameters.append(velocity_scale)
        if len(section) > 6:
            momentum = _exact_finite_float(
                f"ILC section[{index}].momentum_gamma", section[6]
            )
            if not 0.0 <= momentum < 1.0:
                raise ValueError(
                    f"ILC section[{index}].momentum_gamma must be within [0, 1)"
                )
            parameters.append(momentum)
        normalized.append(tuple(parameters))
    return normalized


def _valid_tracker_feature_trace(trace: Any, total_steps: int) -> bool:
    """Semantically validate an optional cached v2 tracker trace."""

    if not isinstance(trace, list):
        return False
    if not trace:
        return True
    if len(trace) != total_steps:
        return False
    from control.learned_residual import DEFAULT_N_INPUTS

    for entry in trace:
        if not isinstance(entry, (list, tuple)) or len(entry) != 13:
            return False
        if not _finite_real_array(entry[0], (DEFAULT_N_INPUTS,)):
            return False
        if any(not _finite_real_scalar(entry[index]) for index in (1, 2, 3, 8)):
            return False
        if any(
            not _finite_real_array(entry[index], (3,))
            for index in (4, 5, 6, 7, 9, 10, 11, 12)
        ):
            return False
    return True


def _valid_position_trace(trace: Any, total_steps: int, sim_time_s: float) -> bool:
    """Semantically validate an optional cached post-integration state trace."""

    if trace is None:
        return True
    if not isinstance(trace, list):
        return False
    if not trace:
        return True
    if len(trace) != total_steps:
        return False
    previous_time = -math.inf
    for entry in trace:
        if not isinstance(entry, dict) or set(entry) != {
            "t",
            "pos",
            "vel",
            "yaw",
            "tracking_err_m",
        }:
            return False
        if (
            not _finite_real_scalar(entry["t"], nonnegative=True)
            or float(entry["t"]) <= previous_time
            or not _finite_real_array(entry["pos"], (3,))
            or not _finite_real_array(entry["vel"], (3,))
            or not _finite_real_scalar(entry["yaw"])
            or not _finite_real_scalar(entry["tracking_err_m"], nonnegative=True)
        ):
            return False
        previous_time = float(entry["t"])
    return math.isclose(previous_time, sim_time_s, rel_tol=0.0, abs_tol=1e-12)


FUNCTION_TIMING_SCOPE = (
    "wall time inside the benchmark function call; Python interpreter and "
    "module-import startup before the call are excluded"
)
PHASE_TIMING_NOTES = {
    "cache_lookup": (
        "artifact reads, checksum verification, decoding, and semantic validation; "
        "RacingLine lock acquisition remains in racing_line and other lock waits "
        "remain in startup"
    ),
    "build_phases": (
        "trajectory, plan_validation, and ilc are cache-miss work only; racing_line "
        "also includes key-lock/materialization overhead around a cache hit"
    ),
    "startup": (
        "measured residual for configuration resolution, imports, lock waits not "
        "classified above, provenance, and result assembly"
    ),
}


def _finalize_phase_timings(
    phases: Dict[str, float], total_started: float
) -> float:
    """Close a mutually-exclusive phase ledger against measured call wall time."""

    total_wall = time.perf_counter() - total_started
    classified = sum(
        float(value)
        for name, value in phases.items()
        if name not in {"startup", "total_wall"}
    )
    measured_startup = float(phases.get("startup", 0.0))
    tolerance = max(1e-9, total_wall * 1e-9)
    if classified + measured_startup > total_wall + tolerance:
        raise AssertionError(
            "benchmark phase timers overlap: "
            f"classified={classified + measured_startup:.9f}s total={total_wall:.9f}s"
        )
    phases["startup"] = measured_startup + max(
        0.0, total_wall - classified - measured_startup
    )
    phases["total_wall"] = total_wall
    phase_sum = sum(
        float(value) for name, value in phases.items() if name != "total_wall"
    )
    assert total_wall + tolerance >= phase_sum
    return total_wall


def _dataclass_from_overrides(cls, overrides: Mapping[str, Any]):
    """Construct a config dataclass from exact, finite resolved values."""
    if not isinstance(overrides, Mapping):
        raise TypeError(f"{cls.__name__} overrides must be a mapping")
    defaults = cls()
    if not overrides:
        return defaults
    invalid_keys = [key for key in overrides if type(key) is not str]
    if invalid_keys:
        raise TypeError(f"{cls.__name__} override keys must be exact strings")
    fields = {field.name: field for field in dataclasses.fields(cls)}
    valid_fields = set(fields)
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} override key(s): {', '.join(unknown)}; "
            f"valid keys: {', '.join(sorted(valid_fields))}"
        )
    normalized: Dict[str, Any] = {}
    for name, value in overrides.items():
        default = getattr(defaults, name)
        if type(default) is bool:
            normalized[name] = _exact_bool(f"{cls.__name__}.{name}", value)
        elif type(default) is int:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{cls.__name__}.{name} must be an integer")
            normalized[name] = int(value)
        elif type(default) is float:
            normalized[name] = _exact_finite_float(
                f"{cls.__name__}.{name}", value
            )
        elif type(default) is str:
            if type(value) is not str:
                raise TypeError(f"{cls.__name__}.{name} must be a string")
            normalized[name] = value
        elif default is None:
            if value is None:
                normalized[name] = None
            else:
                annotation = str(fields[name].type).lower()
                if "float" in annotation:
                    normalized[name] = _exact_finite_float(
                        f"{cls.__name__}.{name}", value
                    )
                elif "str" in annotation:
                    if type(value) is not str:
                        raise TypeError(f"{cls.__name__}.{name} must be a string or None")
                    normalized[name] = value
                else:
                    raise TypeError(
                        f"unsupported optional config field {cls.__name__}.{name}"
                    )
        else:
            raise TypeError(
                f"unsupported config field type for {cls.__name__}.{name}"
            )
    resolved = cls(**normalized)
    _validate_resolved_config_dataclass(resolved)
    return resolved


def _validate_resolved_config_dataclass(config: Any) -> None:
    """Enforce only hard physical/algorithmic invariants at the benchmark seam."""

    name = type(config).__name__

    def require_positive(*fields: str) -> None:
        for field in fields:
            if float(getattr(config, field)) <= 0.0:
                raise ValueError(f"{name}.{field} must be strictly positive")

    def require_nonnegative(*fields: str) -> None:
        for field in fields:
            if float(getattr(config, field)) < 0.0:
                raise ValueError(f"{name}.{field} must be non-negative")

    if name == "DroneConstraints":
        require_positive(
            "max_velocity",
            "max_acceleration",
            "max_jerk",
            "max_tilt_angle",
            "max_thrust",
            "max_body_rate",
            "mass",
            "gravity",
        )
        if config.max_tilt_angle > math.pi / 2.0:
            raise ValueError("DroneConstraints.max_tilt_angle must not exceed pi/2")
    elif name == "PlannerConfig":
        require_nonnegative(
            "entry_exit_offset_m",
            "lookahead_s",
            "accel_ff_gain",
            "accel_ff_clamp_ms2",
        )
        require_positive(
            "helix_entry_inflate",
            "helix_interior_inflate",
            "plan_max_speed_mps",
            "cmd_max_speed_mps",
            "search_window_s",
            "accel_ff_cutoff_hz",
        )
        for field in (
            "max_compression_sturn",
            "max_compression_protected",
            "max_compression_helix",
            "max_compression_easy",
        ):
            value = float(getattr(config, field))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"PlannerConfig.{field} must be within (0, 1]")
    elif name == "SequencerConfig":
        require_positive(
            "pass_through_margin",
            "crash_margin",
            "off_track_distance",
            "max_approach_angle",
        )
        require_nonnegative("proximity_pass_distance", "detection_dropout_frames")
        if config.max_approach_angle > math.pi:
            raise ValueError("SequencerConfig.max_approach_angle must not exceed pi")
        if not 0.0 < config.recovery_speed_factor <= 1.0:
            raise ValueError(
                "SequencerConfig.recovery_speed_factor must be within (0, 1]"
            )
    elif name == "TrackerConfig":
        require_positive(
            "max_tilt_rad", "max_body_rate", "mass", "gravity", "max_thrust_n"
        )
        require_nonnegative(
            "drag_coefficient",
            "drag_ff_coeff",
            "residual_clamp_rad",
            "residual_thrust_clamp",
        )
        if config.max_lateral_accel is not None and config.max_lateral_accel <= 0.0:
            raise ValueError("TrackerConfig.max_lateral_accel must be positive or None")
        if config.sim_roll_sign not in (-1.0, 1.0):
            raise ValueError("TrackerConfig.sim_roll_sign must be exactly -1 or +1")
        if config.max_tilt_rad > math.pi / 2.0:
            raise ValueError("TrackerConfig.max_tilt_rad must not exceed pi/2")
        if not (
            0.0
            <= config.min_thrust_normalized
            < config.max_thrust_normalized
            <= 1.0
        ):
            raise ValueError(
                "TrackerConfig normalized thrust bounds must satisfy "
                "0 <= min < max <= 1"
            )


def _threshold_snapshot(
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve and validate an immutable-by-convention evaluator policy."""

    if overrides is not None and not isinstance(overrides, Mapping):
        raise TypeError("threshold overrides must be a mapping or None")
    resolved = dict(THRESHOLDS)
    if overrides:
        unknown = sorted(set(overrides) - set(resolved))
        if unknown:
            raise ValueError("unknown threshold key(s): " + ", ".join(unknown))
        resolved.update(dict(overrides))
    if type(resolved["no_crash"]) is not bool:
        raise TypeError("threshold no_crash must be a bool")
    for name in (
        "unit_tests_pass_rate",
        "min_gate_pass_rate",
    ):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"threshold {name} must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"threshold {name} must be within [0, 1]")
        resolved[name] = float(value)
    for name in (
        "max_avg_tracking_error_m",
        "max_max_tracking_error_m",
        "max_ekf_uncertainty_m",
        "min_loop_hz",
        "max_total_time_s",
    ):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"threshold {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"threshold {name} must be finite and non-negative")
        resolved[name] = float(value)
    return resolved


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_single_test(name: str, fn) -> Dict[str, Any]:
    """Run a single unit test and return structured result."""
    t0 = time.perf_counter()
    try:
        fn()
        return {"name": name, "passed": True, "time_ms": (time.perf_counter() - t0) * 1000}
    except Exception as e:
        return {
            "name": name, "passed": False,
            "error": str(e), "time_ms": (time.perf_counter() - t0) * 1000,
        }


def run_unit_tests() -> Dict[str, Any]:
    """Run all pipeline unit tests and return structured results."""
    from competition.adapter import AttitudeCommand, Quaternion, TelemetryState
    from estimation.ekf import DroneEKF, EKFConfig
    from estimation.state_predictor import StatePredictor
    from estimation.gate_tracker import GateTracker
    from gate_sequencing.sequencer import GateSequencer, GateSpec
    from planning.trajectory_optimizer import (
        DroneConstraints, GateWaypoint, TrajectoryOptimizer, TrajectoryPoint,
    )
    from planning.racing_line import RacingLineOptimizer, SpeedProfiler
    from control.mpc_tracker import GeometricTracker, SimplePositionTracker, TrackerConfig

    tests = []

    # --- Quaternion roundtrip ---
    def _quat():
        for r, p, y in [(0, 0, 0), (0.1, 0.2, 0.3), (-0.5, 0.3, 1.0), (0, 0, math.pi)]:
            q = Quaternion.from_euler(r, p, y)
            r2, p2, y2 = q.to_euler()
            assert abs(r - r2) < 1e-5, f"Roll {r} != {r2}"
            assert abs(p - p2) < 1e-5, f"Pitch {p} != {p2}"
    tests.append(("quaternion_roundtrip", _quat))

    # --- EKF convergence ---
    def _ekf():
        ekf = DroneEKF(EKFConfig(position_noise_std=0.01, velocity_noise_std=0.05))
        ekf.initialize((1.5, 2.5, -2.5), (0, 0, 0), timestamp_s=0.0)
        true_pos, true_vel = (1.0, 2.0, -3.0), (0.5, -0.3, 0.0)
        for i in range(100):
            ekf.predict((0, 0, -9.81), (0, 0, 0), i * 0.01)
            ekf.update_odometry(true_pos, true_vel)
        pos_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ekf.position, true_pos)))
        vel_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ekf.velocity, true_vel)))
        assert pos_err < 0.5, f"pos_err={pos_err:.4f}"
        assert vel_err < 0.5, f"vel_err={vel_err:.4f}"
    tests.append(("ekf_convergence", _ekf))

    # --- Trajectory generation ---
    def _traj():
        wps = [
            GateWaypoint(position=(5, 0, -2), normal=(1, 0, 0), yaw=0),
            GateWaypoint(position=(10, 5, -3), normal=(0, 1, 0), yaw=math.pi / 2),
            GateWaypoint(position=(15, 0, -2), normal=(-1, 0, 0), yaw=math.pi),
        ]
        traj = TrajectoryOptimizer(
            DroneConstraints(max_velocity=10.0), dt_sample=0.05
        ).optimize(wps, start_position=(0, 0, -2))
        assert traj.total_time > 0, "total_time must be positive"
        assert len(traj.points) > 10, f"too few points: {len(traj.points)}"
        assert len(traj.segment_times) == 7, f"expected 7 segments (3 gates × 2 entry/exit + finish), got {len(traj.segment_times)}"
    tests.append(("trajectory_generation", _traj))

    # --- Racing line ---
    def _rl():
        wps = [
            GateWaypoint(position=(5, 0, -2), normal=(1, 0, 0), yaw=0),
            GateWaypoint(position=(10, 5, -2), normal=(0, 1, 0), yaw=math.pi / 2),
        ]
        from planning.racing_line import RacingLineConfig
        out = RacingLineOptimizer(RacingLineConfig(use_cache=False)).optimize(wps, (0, 0, -2))
        assert len(out) == 2
    tests.append(("racing_line", _rl))

    # --- Speed profiler ---
    def _sp():
        # Iter-018: max_speed sourced from drone_spec; min_speed is a
        # planner-policy choice (not a drone-envelope property) so it
        # stays inline.
        from competition.drone_spec import DEFAULT_MAX_VELOCITY_MPS
        pts = [(0, 0, -2), (10, 0, -2), (20, 0, -2), (20, 10, -2), (20, 20, -2)]
        speeds = SpeedProfiler(
            max_speed=DEFAULT_MAX_VELOCITY_MPS, min_speed=2.0,
        ).profile(pts)
        assert len(speeds) == 5
        assert all(2.0 <= s <= DEFAULT_MAX_VELOCITY_MPS for s in speeds), (
            f"speeds out of range: {speeds}"
        )
    tests.append(("speed_profiler", _sp))

    # --- Geometric tracker (tight hover test — Phase 1 requirement) ---
    def _gt():
        # Iter-020: explicit overrides are now redundant since iter-013
        # routed TrackerConfig defaults through competition.drone_spec.
        # mass=1.0, gravity=9.81, max_thrust_n=20.0 are the spec defaults.
        tr = GeometricTracker(TrackerConfig())
        ref = TrajectoryPoint(0, (0, 0, -2), (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, 0)
        cmd = tr.track((0, 0, -2), (0, 0, 0), 0.0, ref)
        assert abs(cmd.roll_rad) < 0.01, f"hover roll={cmd.roll_rad:.4f} (must be <0.01)"
        assert abs(cmd.pitch_rad) < 0.01, f"hover pitch={cmd.pitch_rad:.4f} (must be <0.01)"
        assert 0.01 < cmd.thrust < 0.99, f"thrust={cmd.thrust}"
    tests.append(("geometric_tracker", _gt))

    # --- Gate sequencer ---
    def _gs():
        gs = GateSequencer([
            GateSpec("g1", position=(5, 0, -2), yaw=0, sequence_index=0),
            GateSpec("g2", position=(10, 0, -2), yaw=0, sequence_index=1),
        ])
        gs.start()
        assert gs.update((4, 0, -2)) is None
        p = gs.update((6, 0, -2))
        assert p is not None and p.gate_id == "g1"
        assert gs.update((9, 0, -2)) is None
        p2 = gs.update((11, 0, -2))
        assert p2 is not None and p2.gate_id == "g2"
        assert gs.is_complete
    tests.append(("gate_sequencer", _gs))

    # --- Gate tracker ---
    def _gtr():
        tracker = GateTracker()
        for frame in range(20):
            cx = 320 + frame * 5
            tracker.predict()
            tracker.update([("gate_1", (cx, 240, 80, 80), 0.9)])
        gates = tracker.get_tracked_gates()
        assert len(gates) >= 1, "no confirmed tracks"
        g = tracker.get_gate("gate_1")
        assert g is not None
        assert g.hits == 20
        # Coast test
        for _ in range(5):
            tracker.predict()
            tracker.update([])
        g_c = tracker.get_gate("gate_1")
        assert g_c is not None, "track should survive 5 frames coast"
        pred = tracker.get_predicted_bbox("gate_1")
        assert pred is not None
        assert pred[0] > 320 + 19 * 5, "prediction should extrapolate forward"
    tests.append(("gate_tracker", _gtr))

    # --- State predictor ---
    def _pred():
        pr = StatePredictor()
        pp, pv, po = pr.predict((0, 0, -5), (3, 0, 0), (0, 0, 0), (0, 0, 0), dt_override=0.1)
        assert abs(pp[0] - 0.3) < 0.05, f"predicted x={pp[0]}"
    tests.append(("state_predictor", _pred))

    # Run all
    results = [_run_single_test(name, fn) for name, fn in tests]
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total_ms = sum(r["time_ms"] for r in results)

    return {
        "tests": results,
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0,
        "total_time_ms": total_ms,
    }


# ---------------------------------------------------------------------------
# Prepared synthetic benchmark (content-addressed v2 evaluator)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PreparedCourse:
    """Deterministic planning output reusable across controller rollouts."""

    track: Dict[str, Any]
    gate_specs: List[Any]
    gate_waypoints: List[Any]
    sequencer_config: Any
    optimized_waypoints: List[Any]
    start_position: np.ndarray
    max_velocity: float
    drone_constraints: Any
    trajectory: Any
    plan_validation: Dict[str, Any]
    ilc_offsets: Optional[np.ndarray]
    ilc_velocity_offsets: Optional[np.ndarray]
    dt: float
    resolved_planning_config: Dict[str, Any]
    config_hash: str
    artifact_key: str
    artifact_keys: Dict[str, str]
    cache_states: Dict[str, str]
    phase_timings_s: Dict[str, float]
    dependency_fingerprint: Dict[str, Any]


def _git_provenance() -> Dict[str, Any]:
    """Return one stable commit/worktree snapshot or fail closed on drift."""

    def _git(*args: str) -> bytes:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "could not capture benchmark provenance: git "
                + " ".join(args)
                + f" exited {completed.returncode}"
            )
        return completed.stdout

    # Reuse the promotion loop's length-prefixed, no-follow reader. It rejects
    # symlinks, non-canonical paths, repository escapes, and non-regular files
    # instead of accidentally hashing content outside the worktree.
    from aigp_loop._util import _untracked_files_digest

    excluded_prefixes = (".cache/", ".loop/", ".research_loop/")
    secret_names = {".env", ".env.local", "credentials.json", "secrets.json"}

    def capture() -> tuple[bytes, bytes, bytes, str, List[str]]:
        commit_bytes = _git("rev-parse", "HEAD").strip()
        if len(commit_bytes) not in {40, 64} or any(
            byte not in b"0123456789abcdefABCDEF" for byte in commit_bytes
        ):
            raise RuntimeError("could not capture an exact Git commit identity")
        status = _git("status", "--porcelain=v1", "--untracked-files=all")
        diff = _git(
            "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD"
        )
        raw_names = _git("ls-files", "--others", "--exclude-standard", "-z")
        try:
            decoded_names = sorted(
                raw.decode("utf-8").replace("\\", "/")
                for raw in raw_names.split(b"\0")
                if raw
            )
        except UnicodeDecodeError as error:
            raise ValueError("untracked Git file names must be UTF-8") from error
        excluded: List[str] = []
        included: List[str] = []
        for name in decoded_names:
            if name.startswith(excluded_prefixes) or Path(name).name.lower() in secret_names:
                excluded.append(name)
            else:
                included.append(name)
        untracked_digest = _untracked_files_digest(_REPO.resolve(), included)
        return commit_bytes, status, diff, untracked_digest.hex(), excluded

    # A single series of Git/file reads can straddle an editor write. Require
    # two identical complete captures. This also makes worker/orchestrator
    # comparisons meaningful instead of hashing a state that never existed.
    previous = capture()
    for _ in range(3):
        current = capture()
        if current == previous:
            break
        previous = current
    else:
        raise RuntimeError("worktree changed while capturing benchmark provenance")

    commit_bytes, status, diff, untracked_hash, excluded_untracked = current
    commit = commit_bytes.decode("ascii")
    tracked_hash = hashlib.sha256(diff).hexdigest()
    digest = hashlib.sha256(
        status + b"\0" + bytes.fromhex(tracked_hash) + bytes.fromhex(untracked_hash)
    ).hexdigest()
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_diff_hash": digest,
        "tracked_diff_hash": tracked_hash,
        "untracked_content_hash": untracked_hash,
        "excluded_untracked_paths": excluded_untracked,
    }


def _trajectory_to_arrays(trajectory: Any) -> Dict[str, np.ndarray]:
    points = list(trajectory.points)
    point_times = np.asarray([point.time for point in points], dtype=np.float64)
    if len(point_times) == 0 or np.any(np.diff(point_times) < 0):
        raise ValueError("trajectory points must have nondecreasing timestamps")
    # The planner emits both sides of each segment boundary at the same
    # timestamp. RaceTrajectory.sample() selects the later point there, so
    # retain the last duplicate. The stored canonical artifact is strictly
    # increasing and cold execution is reconstructed from this same form.
    keep = np.concatenate((np.diff(point_times) > 0, np.asarray([True])))
    points = [point for point, retain in zip(points, keep) if bool(retain)]
    return {
        "time": np.asarray([point.time for point in points], dtype=np.float64),
        "position": np.asarray([point.position for point in points], dtype=np.float64),
        "velocity": np.asarray([point.velocity for point in points], dtype=np.float64),
        "acceleration": np.asarray([point.acceleration for point in points], dtype=np.float64),
        "jerk": np.asarray([point.jerk for point in points], dtype=np.float64),
        "yaw": np.asarray([point.yaw for point in points], dtype=np.float64),
        "yaw_rate": np.asarray([point.yaw_rate for point in points], dtype=np.float64),
        "ff_acceleration": np.asarray(
            [point.ff_acceleration for point in points], dtype=np.float64
        ),
        "total_time": np.asarray([trajectory.total_time], dtype=np.float64),
        "segment_times": np.asarray(trajectory.segment_times, dtype=np.float64),
        "gate_position": np.asarray(
            [gate.position for gate in trajectory.gate_waypoints], dtype=np.float64
        ),
        "gate_normal": np.asarray(
            [gate.normal for gate in trajectory.gate_waypoints], dtype=np.float64
        ),
        "gate_width": np.asarray(
            [gate.width for gate in trajectory.gate_waypoints], dtype=np.float64
        ),
        "gate_height": np.asarray(
            [gate.height for gate in trajectory.gate_waypoints], dtype=np.float64
        ),
        "gate_yaw": np.asarray(
            [gate.yaw for gate in trajectory.gate_waypoints], dtype=np.float64
        ),
    }


def _trajectory_from_arrays(
    arrays: Mapping[str, np.ndarray], expected_gate_count: Optional[int] = None
) -> Any:
    from planning.trajectory_optimizer import GateWaypoint, RaceTrajectory, TrajectoryPoint

    count = len(arrays["time"])
    vector_names = ("position", "velocity", "acceleration", "jerk", "ff_acceleration")
    if count == 0 or any(np.asarray(arrays[name]).shape != (count, 3) for name in vector_names):
        raise ValueError("invalid cached trajectory point arrays")
    if any(len(arrays[name]) != count for name in ("yaw", "yaw_rate")):
        raise ValueError("invalid cached trajectory scalar arrays")
    numeric_names = (*vector_names, "time", "yaw", "yaw_rate", "segment_times")
    if any(not np.all(np.isfinite(np.asarray(arrays[name]))) for name in numeric_names):
        raise ValueError("cached trajectory contains non-finite values")
    times = np.asarray(arrays["time"], dtype=np.float64)
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("cached trajectory times are not strictly increasing")
    points = [
        TrajectoryPoint(
            time=float(arrays["time"][index]),
            position=tuple(float(v) for v in arrays["position"][index]),
            velocity=tuple(float(v) for v in arrays["velocity"][index]),
            acceleration=tuple(float(v) for v in arrays["acceleration"][index]),
            jerk=tuple(float(v) for v in arrays["jerk"][index]),
            yaw=float(arrays["yaw"][index]),
            yaw_rate=float(arrays["yaw_rate"][index]),
            ff_acceleration=tuple(
                float(v) for v in arrays["ff_acceleration"][index]
            ),
        )
        for index in range(count)
    ]
    gate_count = len(arrays["gate_width"])
    if expected_gate_count is not None and gate_count != expected_gate_count:
        raise ValueError("cached trajectory gate count differs from prepared course")
    if any(
        len(arrays[name]) != gate_count
        for name in ("gate_position", "gate_normal", "gate_height", "gate_yaw")
    ):
        raise ValueError("invalid cached trajectory gate arrays")
    if np.asarray(arrays["gate_position"]).shape != (gate_count, 3):
        raise ValueError("invalid cached gate positions")
    if np.asarray(arrays["gate_normal"]).shape != (gate_count, 3):
        raise ValueError("invalid cached gate normals")
    gate_numeric_names = (
        "gate_position", "gate_normal", "gate_width", "gate_height", "gate_yaw"
    )
    if any(
        not np.all(np.isfinite(np.asarray(arrays[name], dtype=np.float64)))
        for name in gate_numeric_names
    ):
        raise ValueError("cached trajectory gates contain non-finite values")
    if np.any(np.asarray(arrays["gate_width"]) <= 0) or np.any(
        np.asarray(arrays["gate_height"]) <= 0
    ):
        raise ValueError("cached trajectory gate dimensions must be positive")
    if np.any(np.linalg.norm(np.asarray(arrays["gate_normal"]), axis=1) <= 1e-12):
        raise ValueError("cached trajectory gate normals must be nonzero")
    gates = [
        GateWaypoint(
            position=tuple(float(v) for v in arrays["gate_position"][index]),
            normal=tuple(float(v) for v in arrays["gate_normal"][index]),
            width=float(arrays["gate_width"][index]),
            height=float(arrays["gate_height"][index]),
            yaw=float(arrays["gate_yaw"][index]),
        )
        for index in range(gate_count)
    ]
    total_time_array = np.asarray(arrays["total_time"]).reshape(-1)
    if len(total_time_array) != 1 or not np.isfinite(total_time_array[0]):
        raise ValueError("invalid cached trajectory total time")
    if total_time_array[0] <= 0 or times[-1] > total_time_array[0] + 1e-9:
        raise ValueError("cached trajectory total time is inconsistent")
    return RaceTrajectory(
        points=points,
        total_time=float(total_time_array[0]),
        segment_times=[float(value) for value in arrays["segment_times"]],
        gate_waypoints=gates,
    )


def _gate_inputs(track: Mapping[str, Any]) -> Tuple[List[Any], List[Any], np.ndarray]:
    from gate_sequencing.sequencer import GateSpec
    from planning.trajectory_optimizer import GateWaypoint

    gate_defaults = track.get("gate_defaults", {})
    if not isinstance(gate_defaults, Mapping):
        raise TypeError("track gate_defaults must be a mapping")
    default_width = _exact_finite_float(
        "gate_defaults.interior_width_m",
        gate_defaults.get("interior_width_m", 1.2),
        strictly_positive=True,
    )
    default_height = _exact_finite_float(
        "gate_defaults.interior_height_m",
        gate_defaults.get("interior_height_m", 1.2),
        strictly_positive=True,
    )
    default_border = gate_defaults.get("border_width_m")
    if default_border is not None:
        default_border = _exact_finite_float(
            "gate_defaults.border_width_m", default_border, nonnegative=True
        )
    default_depth = gate_defaults.get("depth_m")
    if default_depth is not None:
        default_depth = _exact_finite_float(
            "gate_defaults.depth_m", default_depth, strictly_positive=True
        )
    gates_data = track.get("gates", [])
    if not isinstance(gates_data, list):
        raise TypeError("track gates must be a list")
    gate_specs: List[Any] = []
    gate_waypoints: List[Any] = []
    gate_ids: set[str] = set()
    sequence_indices: set[int] = set()
    for gate_number, gate_data in enumerate(gates_data):
        if not isinstance(gate_data, Mapping):
            raise TypeError(f"gate[{gate_number}] must be a mapping")
        pose = gate_data.get("pose", {})
        gate_config = gate_data.get("config", {})
        if not isinstance(pose, Mapping):
            raise TypeError(f"gate[{gate_number}].pose must be a mapping")
        if not isinstance(gate_config, Mapping):
            raise TypeError(f"gate[{gate_number}].config must be a mapping")
        gate_id = gate_data.get("id")
        if type(gate_id) is not str or not gate_id:
            raise TypeError(f"gate[{gate_number}].id must be a non-empty string")
        if gate_id in gate_ids:
            raise ValueError(f"duplicate gate id: {gate_id}")
        gate_ids.add(gate_id)
        sequence_index = gate_data.get("sequence_index", gate_number)
        if isinstance(sequence_index, bool) or not isinstance(
            sequence_index, (int, np.integer)
        ):
            raise TypeError(
                f"gate[{gate_number}].sequence_index must be an integer"
            )
        sequence_index = int(sequence_index)
        if sequence_index < 0:
            raise ValueError(
                f"gate[{gate_number}].sequence_index must be non-negative"
            )
        if sequence_index in sequence_indices:
            raise ValueError(f"duplicate gate sequence_index: {sequence_index}")
        sequence_indices.add(sequence_index)
        x = _exact_finite_float(f"gate[{gate_number}].pose.x", pose.get("x", 0))
        y = _exact_finite_float(f"gate[{gate_number}].pose.y", pose.get("y", 0))
        z = _exact_finite_float(f"gate[{gate_number}].pose.z", pose.get("z", 1.5))
        yaw = _exact_finite_float(
            f"gate[{gate_number}].pose.yaw", pose.get("yaw", 0)
        )
        pitch = _exact_finite_float(
            f"gate[{gate_number}].pose.pitch", pose.get("pitch", 0)
        )
        roll = _exact_finite_float(
            f"gate[{gate_number}].pose.roll", pose.get("roll", 0)
        )
        width = _exact_finite_float(
            f"gate[{gate_number}].config.interior_width_m",
            gate_config.get("interior_width_m", default_width),
            strictly_positive=True,
        )
        height = _exact_finite_float(
            f"gate[{gate_number}].config.interior_height_m",
            gate_config.get("interior_height_m", default_height),
            strictly_positive=True,
        )
        border = gate_config.get("border_width_m", default_border)
        if border is not None:
            border = _exact_finite_float(
                f"gate[{gate_number}].config.border_width_m",
                border,
                nonnegative=True,
            )
        depth = gate_config.get("depth_m", default_depth)
        if depth is not None:
            depth = _exact_finite_float(
                f"gate[{gate_number}].config.depth_m",
                depth,
                strictly_positive=True,
            )
        spec_kwargs = {
            "gate_id": gate_id,
            "position": (x, y, z),
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "interior_width": width,
            "interior_height": height,
            "sequence_index": sequence_index,
        }
        if border is not None:
            spec_kwargs["border_width"] = border
        if depth is not None:
            spec_kwargs["depth"] = depth
        gate_specs.append(GateSpec(**spec_kwargs))
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        gate_waypoints.append(
            GateWaypoint(
                position=(x, y, z),
                normal=(cy * cp, sy * cp, sp),
                width=width,
                height=height,
                yaw=yaw,
            )
        )
    start = track.get("start", {})
    if not isinstance(start, Mapping):
        raise TypeError("track start must be a mapping")
    raw_start_position = start.get("position", [0.0, 0.0, 1.5])
    if not isinstance(raw_start_position, (list, tuple)) or len(raw_start_position) != 3:
        raise ValueError("track start.position must contain three finite numbers")
    start_position = np.asarray(
        [
            _exact_finite_float(f"track start.position[{index}]", value)
            for index, value in enumerate(raw_start_position)
        ],
        dtype=np.float64,
    )
    if not gate_specs:
        raise ValueError("track must define at least one gate")
    return gate_specs, gate_waypoints, start_position


def _validation_payload(validation: Any) -> Dict[str, Any]:
    return dataclasses.asdict(validation) if dataclasses.is_dataclass(validation) else dict(validation)


def _valid_validation_payload(payload: Any, expected_total_gates: int) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {
        "ok",
        "reason",
        "gates_passed",
        "total_gates",
        "crashed",
        "disqualified",
        "dq_reason",
        "last_crash_gate",
        "samples_evaluated",
        "first_failure_time_s",
    }
    if not required.issubset(payload):
        return False
    if type(payload["ok"]) is not bool:
        return False
    if type(payload["crashed"]) is not bool or type(payload["disqualified"]) is not bool:
        return False
    if not isinstance(payload["reason"], str):
        return False
    if type(payload["total_gates"]) is not int or payload["total_gates"] != expected_total_gates:
        return False
    if type(payload["gates_passed"]) is not int or not (
        0 <= payload["gates_passed"] <= expected_total_gates
    ):
        return False
    if type(payload["samples_evaluated"]) is not int or payload["samples_evaluated"] < 0:
        return False
    for name in ("dq_reason", "last_crash_gate"):
        if payload[name] is not None and not isinstance(payload[name], str):
            return False
    failure_time = payload["first_failure_time_s"]
    if failure_time is not None and (
        isinstance(failure_time, bool)
        or not isinstance(failure_time, (int, float))
        or not math.isfinite(float(failure_time))
    ):
        return False
    if payload["ok"] and (
        payload["gates_passed"] != expected_total_gates
        or payload["crashed"]
        or payload["disqualified"]
        or payload["dq_reason"] is not None
        or payload["last_crash_gate"] is not None
        or failure_time is not None
    ):
        return False
    if payload["crashed"] and payload["last_crash_gate"] is None:
        return False
    if payload["disqualified"] and payload["dq_reason"] is None:
        return False
    return True


def _decode_ilc_arrays(
    arrays: Optional[Mapping[str, np.ndarray]],
    *,
    expected_steps: int,
    max_position_offset_m: float,
    dt: float,
) -> Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    if arrays is None:
        return None
    try:
        present_array = np.asarray(arrays["present"]).reshape(-1)
        if present_array.shape != (1,) or int(present_array[0]) not in (0, 1):
            return None
        position = np.asarray(arrays["position_offsets"], dtype=np.float64)
        velocity = np.asarray(arrays["velocity_offsets"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if position.ndim != 2 or position.shape[1:] != (3,):
        return None
    if velocity.shape != position.shape:
        return None
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        return None
    present = bool(int(present_array[0]))
    if present and len(position) != expected_steps:
        return None
    if not present and len(position) != 0:
        return None
    if present:
        position_norms = np.linalg.norm(position, axis=1)
        velocity_norms = np.linalg.norm(velocity, axis=1)
        # The learner clips every per-iteration correction.  Account for all
        # configured iterations/momentum before decode, then reject an artifact
        # that exceeds that derived envelope.  The derivative cannot exceed a
        # two-sided jump across the complete position envelope in one timestep.
        if np.any(position_norms > max_position_offset_m + 1e-9):
            return None
        max_velocity_offset_mps = 2.0 * max_position_offset_m / dt
        if np.any(velocity_norms > max_velocity_offset_mps + 1e-9):
            return None
    return (position, velocity) if present else (None, None)


def _valid_cached_benchmark_result(
    payload: Any,
    *,
    expected_seed: Optional[int] = None,
    expected_thresholds: Optional[Mapping[str, Any]] = None,
    expected_gate_ids: Optional[Sequence[str]] = None,
    expected_dt: Optional[float] = None,
    expected_controller_config: Optional[Mapping[str, Any]] = None,
    expected_record_position_trace: Optional[bool] = None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != BENCHMARK_RESULT_SCHEMA:
        return False
    if payload.get("evaluator_version") != EVALUATOR_VERSION:
        return False
    boolean_fields = (
        "available",
        "skipped",
        "crashed",
        "disqualified",
        "complete",
        "sim_passed",
        "safety_passed",
        "validity_passed",
    )
    if any(type(payload.get(field)) is not bool for field in boolean_fields):
        return False
    required = {
        "sim_type",
        "seed",
        "thresholds",
        "termination_reason",
        "gate_pass_times",
        "gates_passed",
        "total_gates",
        "gate_pass_rate",
        "dq_reason",
        "last_crash_gate",
        "threshold_failures",
        "plan_validation",
        "completion",
        "failure_summary",
        "sim_time_s",
        "trajectory_time_s",
        "trajectory_points",
        "rollout_wall_time_s",
        "rollout_materialization_wall_time_s",
        "dt",
        "avg_tracking_error_m",
        "max_tracking_error_m",
        "p50_tracking_error_m",
        "p95_tracking_error_m",
        "avg_nearest_path_error_m",
        "max_nearest_path_error_m",
        "ekf_uncertainty_m",
        "avg_loop_hz",
        "total_steps",
        "per_gate_avg_error",
        "controller_trace_summary",
        "tracker_feature_trace",
        "position_trace",
        "resolved_controller_config",
        "_simulate_phase_timings_s",
    }
    if not required.issubset(payload):
        return False
    if payload["sim_type"] != "synthetic_kinematic":
        return False
    if type(payload["seed"]) is not int or (
        expected_seed is not None and payload["seed"] != expected_seed
    ):
        return False
    try:
        cached_thresholds = _threshold_snapshot(payload["thresholds"])
    except (TypeError, ValueError):
        return False
    if dict(payload["thresholds"]) != cached_thresholds:
        return False
    if expected_thresholds is not None and cached_thresholds != _threshold_snapshot(
        expected_thresholds
    ):
        return False
    if not isinstance(payload["termination_reason"], str) or not payload[
        "termination_reason"
    ]:
        return False
    if payload["available"] is not True or payload["skipped"] is not False:
        return False
    gates_passed = payload["gates_passed"]
    total_gates = payload["total_gates"]
    if type(gates_passed) is not int or type(total_gates) is not int:
        return False
    if total_gates <= 0 or not 0 <= gates_passed <= total_gates:
        return False
    gate_rate = payload["gate_pass_rate"]
    if (
        isinstance(gate_rate, bool)
        or not isinstance(gate_rate, (int, float))
        or not math.isfinite(float(gate_rate))
        or not 0.0 <= float(gate_rate) <= 1.0
        or not math.isclose(
            float(gate_rate), gates_passed / total_gates, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        return False
    if payload["complete"] is not (gates_passed == total_gates):
        return False
    if payload["complete"] is not (payload["termination_reason"] == "race_complete"):
        return False
    failures = payload["threshold_failures"]
    if not isinstance(failures, list) or any(
        not isinstance(failure, str) for failure in failures
    ):
        return False
    completion = payload["completion"]
    if not isinstance(completion, dict) or set(completion) != {
        "complete",
        "gates_passed",
        "total_gates",
    }:
        return False
    if (
        type(completion["complete"]) is not bool
        or type(completion["gates_passed"]) is not int
        or type(completion["total_gates"]) is not int
        or completion["complete"] is not payload["complete"]
        or completion["gates_passed"] != gates_passed
        or completion["total_gates"] != total_gates
    ):
        return False
    validation = payload["plan_validation"]
    if not _valid_validation_payload(validation, total_gates):
        return False
    if payload["validity_passed"] is not (validation["ok"] is True):
        return False
    expected_safety = not payload["crashed"] and not payload["disqualified"]
    if payload["safety_passed"] is not expected_safety:
        return False
    if payload["dq_reason"] is not None and not isinstance(payload["dq_reason"], str):
        return False
    if payload["last_crash_gate"] is not None and not isinstance(
        payload["last_crash_gate"], str
    ):
        return False
    if payload["disqualified"] is not (payload["dq_reason"] is not None):
        return False
    termination_reason = payload["termination_reason"]
    termination_is_crash = termination_reason.startswith("crash_")
    termination_is_gate_crash = termination_reason.startswith("crash_gate:")
    termination_is_disqualification = termination_reason.startswith("disqualified:")
    if payload["crashed"] is not termination_is_crash:
        return False
    if (payload["last_crash_gate"] is not None) is not termination_is_gate_crash:
        return False
    if payload["disqualified"] is not termination_is_disqualification:
        return False
    failure_summary = payload["failure_summary"]
    if not isinstance(failure_summary, dict) or not {
        "stdout_tail",
        "stderr_tail",
        "exception",
        "threshold_failures",
    }.issubset(failure_summary):
        return False
    if (
        not isinstance(failure_summary["stdout_tail"], str)
        or not isinstance(failure_summary["stderr_tail"], str)
        or failure_summary["exception"] is not None
        or failure_summary["threshold_failures"] != failures
    ):
        return False
    expected_pass = (
        not failures
        and expected_safety
        and validation["ok"] is True
        and payload["complete"] is True
    )
    if payload["sim_passed"] is not expected_pass:
        return False
    numeric_fields = (
        "sim_time_s",
        "trajectory_time_s",
        "rollout_wall_time_s",
        "rollout_materialization_wall_time_s",
        "dt",
        "avg_tracking_error_m",
        "max_tracking_error_m",
        "p50_tracking_error_m",
        "p95_tracking_error_m",
        "avg_nearest_path_error_m",
        "max_nearest_path_error_m",
        "ekf_uncertainty_m",
        "avg_loop_hz",
    )
    if any(
        isinstance(payload[name], bool)
        or not isinstance(payload[name], (int, float))
        or not math.isfinite(float(payload[name]))
        or float(payload[name]) < 0.0
        for name in numeric_fields
    ):
        return False
    if float(payload["dt"]) <= 0.0 or float(payload["trajectory_time_s"]) <= 0.0:
        return False
    if expected_dt is not None and not math.isclose(
        float(payload["dt"]), float(expected_dt), rel_tol=0.0, abs_tol=1e-15
    ):
        return False
    if not math.isclose(
        float(payload["rollout_wall_time_s"]),
        float(payload["rollout_materialization_wall_time_s"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        return False
    if not (
        float(payload["avg_tracking_error_m"])
        <= float(payload["max_tracking_error_m"]) + 1e-12
        and float(payload["p50_tracking_error_m"])
        <= float(payload["p95_tracking_error_m"]) + 1e-12
        and float(payload["p95_tracking_error_m"])
        <= float(payload["max_tracking_error_m"]) + 1e-12
        and float(payload["avg_nearest_path_error_m"])
        <= float(payload["max_nearest_path_error_m"]) + 1e-12
    ):
        return False
    if type(payload["trajectory_points"]) is not int or payload["trajectory_points"] <= 0:
        return False
    if type(payload["total_steps"]) is not int or payload["total_steps"] < 0:
        return False
    gate_pass_times = payload["gate_pass_times"]
    if type(gate_pass_times) is not list or len(gate_pass_times) != gates_passed:
        return False
    observed_gate_ids: List[str] = []
    previous_pass_time = -math.inf
    sim_time = float(payload["sim_time_s"])
    for gate_pass in gate_pass_times:
        if type(gate_pass) is not dict or set(gate_pass) != {"gate_id", "time_s"}:
            return False
        gate_id = gate_pass["gate_id"]
        pass_time = gate_pass["time_s"]
        if type(gate_id) is not str or not gate_id:
            return False
        if (
            type(pass_time) not in {int, float}
            or not math.isfinite(pass_time)
            or pass_time < 0.0
            or pass_time <= previous_pass_time
            or pass_time > sim_time + 1e-12
        ):
            return False
        observed_gate_ids.append(gate_id)
        previous_pass_time = float(pass_time)
    if len(set(observed_gate_ids)) != len(observed_gate_ids):
        return False
    if expected_gate_ids is not None:
        if (
            any(type(gate_id) is not str or not gate_id for gate_id in expected_gate_ids)
            or observed_gate_ids != list(expected_gate_ids)[:gates_passed]
        ):
            return False
    per_gate = payload["per_gate_avg_error"]
    if not isinstance(per_gate, dict) or any(
        type(gate_id) is not str
        or not gate_id
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for gate_id, value in per_gate.items()
    ):
        return False
    if expected_gate_ids is not None and not set(per_gate).issubset(expected_gate_ids):
        return False
    controller_summary = payload["controller_trace_summary"]
    expected_summary_fields = {
        "samples",
        "avg_roll_rad",
        "avg_pitch_rad",
        "avg_thrust",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "accel_clamp_active_frac",
        "speed_clamp_active_frac",
        "max_accel_mag_pre_clamp",
    }
    if payload["total_steps"] == 0:
        if controller_summary != {}:
            return False
    elif not isinstance(controller_summary, dict) or set(controller_summary) != expected_summary_fields:
        return False
    else:
        if (
            type(controller_summary["samples"]) is not int
            or controller_summary["samples"] != payload["total_steps"]
        ):
            return False
        for name in expected_summary_fields - {"samples"}:
            value = controller_summary[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return False
        for name in (
            "max_abs_roll_rad",
            "max_abs_pitch_rad",
            "max_accel_mag_pre_clamp",
        ):
            if float(controller_summary[name]) < 0.0:
                return False
        for name in ("accel_clamp_active_frac", "speed_clamp_active_frac"):
            if not 0.0 <= float(controller_summary[name]) <= 1.0:
                return False
    resolved_controller = payload["resolved_controller_config"]
    if not isinstance(resolved_controller, dict):
        return False
    if (
        expected_controller_config is not None
        and resolved_controller != dict(expected_controller_config)
    ):
        return False
    if not _valid_tracker_feature_trace(
        payload["tracker_feature_trace"], payload["total_steps"]
    ):
        return False
    if expected_controller_config is not None:
        trace_features = expected_controller_config.get("trace_features")
        if type(trace_features) is not bool:
            return False
        tracker_trace = payload["tracker_feature_trace"]
        if trace_features:
            if len(tracker_trace) != payload["total_steps"]:
                return False
        elif tracker_trace != []:
            return False
    if not _valid_position_trace(
        payload["position_trace"],
        payload["total_steps"],
        float(payload["sim_time_s"]),
    ):
        return False
    if expected_record_position_trace is not None:
        if type(expected_record_position_trace) is not bool:
            return False
        if expected_record_position_trace:
            if not isinstance(payload["position_trace"], list) or len(
                payload["position_trace"]
            ) != payload["total_steps"]:
                return False
        elif payload["position_trace"] is not None:
            return False
    if payload["sim_passed"] and (
        float(payload["avg_tracking_error_m"])
        > cached_thresholds["max_avg_tracking_error_m"]
        or float(payload["max_tracking_error_m"])
        > cached_thresholds["max_max_tracking_error_m"]
        or float(payload["ekf_uncertainty_m"])
        > cached_thresholds["max_ekf_uncertainty_m"]
        or float(payload["avg_loop_hz"]) < cached_thresholds["min_loop_hz"]
        or float(payload["gate_pass_rate"])
        < cached_thresholds["min_gate_pass_rate"]
        or float(payload["sim_time_s"]) > cached_thresholds["max_total_time_s"]
        or (cached_thresholds["no_crash"] and payload["crashed"])
    ):
        return False
    phases = payload["_simulate_phase_timings_s"]
    if not isinstance(phases, dict) or set(phases) != {"rollout", "metrics"}:
        return False
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
        for value in phases.values()
    ):
        return False
    return math.isclose(
        float(phases["rollout"]),
        float(payload["rollout_materialization_wall_time_s"]),
        rel_tol=0.0,
        abs_tol=0.0,
    )


def _benchmark_result_source_files(
    controller_resolved: Mapping[str, Any],
) -> List[Path]:
    """All Python/data sources directly executed during synthetic rollout."""

    sources = [
        Path(__file__),
        _REPO / "control" / "mpc_tracker.py",
        _REPO / "competition" / "adapter.py",
        _REPO / "competition" / "drone_spec.py",
        _REPO / "competition" / "aigp_geometry.py",
        _REPO / "estimation" / "ekf.py",
        _REPO / "gate_sequencing" / "sequencer.py",
    ]
    if controller_resolved.get("use_residual"):
        sources.append(_REPO / "control" / "learned_residual.py")
        residual_path = controller_resolved.get("residual_weights_path")
        if residual_path:
            sources.append(Path(residual_path))
        else:
            default_residual = _REPO / "control" / "residual_weights.npz"
            if default_residual.exists():
                sources.append(default_residual)
    return sources


def _trajectory_source_files(planner_config: Any) -> List[Path]:
    """Code/data sources that can change a materialized trajectory."""

    sources = [
        Path(__file__),
        _REPO / "planning" / "trajectory_optimizer.py",
        _REPO / "planning" / "ilc_runtime.py",
        _REPO / "competition" / "drone_spec.py",
    ]
    table_path = planner_config.ilc_table_path
    if table_path:
        # ``ILCTable`` reads this payload during optimization.  Hash the
        # resolved file bytes, not merely the configured path, so an updated
        # table at the same path cannot alias an older trajectory artifact.
        sources.append(Path(table_path).resolve())
    return sources


def _validation_source_files() -> List[Path]:
    """Code that computes, serializes, and validates cached plan evidence."""

    return [
        Path(__file__),
        _REPO / "planning" / "plan_validator.py",
        _REPO / "gate_sequencing" / "sequencer.py",
    ]


def _ilc_source_files() -> List[Path]:
    """Code/data that computes, encodes, and validates cached ILC tables."""

    return [
        Path(__file__),
        _REPO / "planning" / "trajectory_optimizer.py",
        _REPO / "planning" / "ilc_sections.py",
        _REPO / "config" / "ilc_defaults.json",
    ]


def prepare_course(
    track: Mapping[str, Any],
    planning_config: Optional[Mapping[str, Any]] = None,
    *,
    dt: float = 0.01,
    cache_root: Optional[os.PathLike[str] | str] = None,
) -> PreparedCourse:
    """Resolve and cache deterministic racing-line, trajectory, validation, and ILC work."""

    dt = _exact_finite_float("dt", dt, strictly_positive=True)
    if not isinstance(track, Mapping):
        raise TypeError("track must be a mapping")
    if planning_config is not None and not isinstance(planning_config, Mapping):
        raise TypeError("planning_config must be a mapping or None")
    from planning.artifact_cache import (
        ArtifactStore,
        artifact_key,
        dependency_fingerprint,
        sha256_json,
    )
    from gate_sequencing.sequencer import SequencerConfig
    from planning.auto_velocity import derive_safe_max_velocity
    from planning.ilc_sections import derive_section_boundaries, load_ilc_config
    from planning.plan_validator import validate_trajectory
    from planning.racing_line import RacingLineConfig, RacingLineOptimizer
    from planning.trajectory_optimizer import (
        DroneConstraints,
        FOVConfig,
        PlannerConfig,
        TrajectoryOptimizer,
        compute_ilc_offset_table,
    )

    planning = _exact_mapping(
        "planning_config", {} if planning_config is None else planning_config
    )
    allowed_planning_keys = {
        "racing_line", "planner", "drone", "ilc", "sequencer"
    }
    unknown_planning = sorted(set(planning) - allowed_planning_keys)
    if unknown_planning:
        raise ValueError(
            "unknown planning_config key(s): " + ", ".join(unknown_planning)
        )
    # A JSON round trip gives the prepared object an immutable-by-convention
    # copy and normalizes tuples before config hashing.
    track_data = json.loads(json.dumps(track, allow_nan=False))
    gate_specs, gate_waypoints, start_position = _gate_inputs(track_data)
    environment = dependency_fingerprint()
    store = ArtifactStore(cache_root)
    timings = {
        "cache_lookup": 0.0,
        "racing_line": 0.0,
        "trajectory": 0.0,
        "plan_validation": 0.0,
        "ilc": 0.0,
    }
    states: Dict[str, str] = {}

    if "max_velocity_mps" in track_data:
        max_velocity = _exact_finite_float(
            "track.max_velocity_mps",
            track_data["max_velocity_mps"],
            strictly_positive=True,
        )
    else:
        max_velocity = float(derive_safe_max_velocity(gate_specs))

    track_sequencer_overrides = _exact_mapping(
        "track.sequencer",
        track_data.get("sequencer", track_data.get("sequencer_overrides", {})),
    )
    planning_sequencer_overrides = _exact_mapping(
        "planning_config.sequencer", planning.get("sequencer", {})
    )
    sequencer_overrides = {
        "pass_through_margin": 1.0,
        **track_sequencer_overrides,
        **planning_sequencer_overrides,
        # The current evaluator requires an observed plane crossing.  Legacy
        # proximity credit could complete a gate up to 0.5 m before its plane
        # and is not authoritative competition completion evidence.
        "proximity_pass_distance": 0.0,
    }
    sequencer_config = _dataclass_from_overrides(
        SequencerConfig, sequencer_overrides
    )

    racing_overrides = _exact_mapping(
        "track.racing_line",
        track_data.get(
            "racing_line", track_data.get("racing_line_overrides", {})
        ),
    )
    racing_overrides.update(
        _exact_mapping(
            "planning_config.racing_line", planning.get("racing_line", {})
        )
    )
    if cache_root is not None:
        racing_overrides["cache_root"] = str(Path(cache_root).resolve())
    racing_config = _dataclass_from_overrides(RacingLineConfig, racing_overrides)

    planner_overrides = _exact_mapping(
        "track.planner",
        track_data.get("planner", track_data.get("planner_overrides", {})),
    )
    planner_overrides.update(
        _exact_mapping("planning_config.planner", planning.get("planner", {}))
    )
    planner_config_resolved = _dataclass_from_overrides(PlannerConfig, planner_overrides)

    drone_overrides = {
        "max_velocity": max_velocity,
        **_exact_mapping("planning_config.drone", planning.get("drone", {})),
    }
    drone_constraints = _dataclass_from_overrides(DroneConstraints, drone_overrides)
    fov_config = FOVConfig()

    phase_started = time.perf_counter()
    racing_optimizer = RacingLineOptimizer(racing_config)
    optimized_waypoints = racing_optimizer.optimize(
        gate_waypoints, tuple(float(value) for value in start_position)
    )
    racing_elapsed = time.perf_counter() - phase_started
    timings["cache_lookup"] += racing_optimizer.last_cache_lookup_s
    timings["racing_line"] = max(
        0.0, racing_elapsed - racing_optimizer.last_cache_lookup_s
    )
    racing_key = racing_optimizer.last_artifact_key or artifact_key(
        "racing-lines-disabled",
        {
            "gates": gate_waypoints,
            "start": start_position,
            "config": {
                key: value
                for key, value in dataclasses.asdict(racing_config).items()
                if key != "cache_root"
            },
        },
        schema_version="racing-line-disabled-v1",
        source_files=[_REPO / "planning" / "racing_line.py"],
        environment=environment,
    )
    states["racing_line"] = "hit" if racing_optimizer.last_cache_hit else "miss"

    trajectory_inputs = {
        "racing_line_key": racing_key,
        "optimized_waypoints": [dataclasses.asdict(gate) for gate in optimized_waypoints],
        "start_position": start_position,
        "start_velocity": [0.0, 0.0, 0.0],
        "constraints": dataclasses.asdict(drone_constraints),
        "planner": dataclasses.asdict(planner_config_resolved),
        "fov": dataclasses.asdict(fov_config),
        "sample_dt": 0.02,
    }
    trajectory_key = artifact_key(
        "trajectories",
        trajectory_inputs,
        schema_version="trajectory-v2",
        source_files=_trajectory_source_files(planner_config_resolved),
        environment=environment,
    )

    lookup_started = time.perf_counter()
    trajectory_arrays = store.load_npz("trajectories", trajectory_key)
    timings["cache_lookup"] += time.perf_counter() - lookup_started
    trajectory = None
    # TrajectoryOptimizer appends one virtual finish gate beyond the scored
    # course. It is part of interpolation/FOV metadata but not GateSequencer.
    expected_trajectory_gate_count = len(optimized_waypoints) + 1
    if trajectory_arrays is not None:
        try:
            trajectory = _trajectory_from_arrays(
                trajectory_arrays,
                expected_gate_count=expected_trajectory_gate_count,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            trajectory = None
    trajectory_build_time = 0.0
    if trajectory is None:
        with store.lock("trajectories", trajectory_key):
            lookup_started = time.perf_counter()
            trajectory_arrays = store.load_npz("trajectories", trajectory_key)
            timings["cache_lookup"] += time.perf_counter() - lookup_started
            if trajectory_arrays is not None:
                try:
                    trajectory = _trajectory_from_arrays(
                        trajectory_arrays,
                        expected_gate_count=expected_trajectory_gate_count,
                    )
                except (KeyError, TypeError, ValueError, IndexError):
                    trajectory = None
            if trajectory is None:
                build_started = time.perf_counter()
                trajectory_optimizer = TrajectoryOptimizer(
                    constraints=drone_constraints,
                    dt_sample=0.02,
                    fov_config=fov_config,
                    planner_config=planner_config_resolved,
                )
                trajectory = trajectory_optimizer.optimize(
                    optimized_waypoints,
                    tuple(float(value) for value in start_position),
                    (0, 0, 0),
                )
                canonical_arrays = _trajectory_to_arrays(trajectory)
                store.save_npz(
                    "trajectories", trajectory_key, canonical_arrays
                )
                # Use exactly the representation future cache hits decode so
                # cold and warm no-cache rollouts cannot diverge at duplicate
                # segment-boundary timestamps.
                trajectory = _trajectory_from_arrays(
                    canonical_arrays,
                    expected_gate_count=expected_trajectory_gate_count,
                )
                trajectory_build_time += time.perf_counter() - build_started
        states["trajectory"] = "miss"
    else:
        states["trajectory"] = "hit"
    timings["trajectory"] = trajectory_build_time

    validation_inputs = {
        "trajectory_key": trajectory_key,
        "gates": [dataclasses.asdict(gate) for gate in gate_specs],
        "dt": dt,
        "sequencer": dataclasses.asdict(sequencer_config),
        "ground_z_threshold": 0.05,
        "ceiling_z_threshold": 20.0,
    }
    validation_key = artifact_key(
        "plan-validation",
        validation_inputs,
        schema_version="plan-validation-v2",
        source_files=_validation_source_files(),
        environment=environment,
    )
    lookup_started = time.perf_counter()
    validation_payload = store.load_json("plan-validation", validation_key)
    timings["cache_lookup"] += time.perf_counter() - lookup_started
    validation_build_time = 0.0
    if not _valid_validation_payload(validation_payload, len(gate_specs)):
        with store.lock("plan-validation", validation_key):
            lookup_started = time.perf_counter()
            validation_payload = store.load_json("plan-validation", validation_key)
            timings["cache_lookup"] += time.perf_counter() - lookup_started
            if not _valid_validation_payload(validation_payload, len(gate_specs)):
                build_started = time.perf_counter()
                validation_payload = _validation_payload(
                    validate_trajectory(
                        trajectory,
                        gate_specs,
                        dt=dt,
                        sequencer_config=sequencer_config,
                    )
                )
                store.save_json("plan-validation", validation_key, validation_payload)
                validation_build_time += time.perf_counter() - build_started
        states["plan_validation"] = "miss"
    else:
        states["plan_validation"] = "hit"
    timings["plan_validation"] = validation_build_time

    ilc_defaults = load_ilc_config()
    ilc_options = _exact_mapping("planning_config.ilc", planning.get("ilc", {}))
    unknown_ilc = sorted(
        set(ilc_options) - {"global", "section_overrides", "section_overrides_format"}
    )
    if unknown_ilc:
        raise ValueError("unknown ILC planning key(s): " + ", ".join(unknown_ilc))
    track_ilc_global = _exact_mapping(
        "track.ilc_global_overrides", track_data.get("ilc_global_overrides", {})
    )
    planning_ilc_global = _exact_mapping(
        "planning_config.ilc.global", ilc_options.get("global", {})
    )
    ilc_global = _normalize_ilc_global({
        **_exact_mapping("ILC defaults.global", ilc_defaults["global"]),
        **track_ilc_global,
        **planning_ilc_global,
    })
    total_steps = int(trajectory.total_time / dt) + 50
    raw_section_overrides = ilc_options.get(
        "section_overrides", track_data.get("ilc_section_overrides")
    )
    override_format = ilc_options.get(
        "section_overrides_format",
        track_data.get("ilc_section_overrides_format", "auto"),
    )
    section_overrides = _normalize_ilc_section_overrides(
        raw_section_overrides,
        override_format=override_format,
        total_steps=total_steps,
    )
    if section_overrides is not None:
        section_boundaries = section_overrides
    else:
        section_boundaries = _normalize_ilc_section_overrides(
            [list(section) for section in derive_section_boundaries(
                trajectory, dt, config=ilc_defaults
            )],
            override_format="steps",
            total_steps=total_steps,
        )
        if section_boundaries is None:  # pragma: no cover - normalized nonempty input
            raise RuntimeError("derived ILC sections unexpectedly resolved to null")

    correction_caps = [float(ilc_global["max_correction_m"])]
    learning_rates = [float(ilc_global["alpha"])]
    momentum_rates = [float(ilc_global["momentum_gamma"])]
    for section in section_boundaries:
        learning_rates.append(float(section[2]))
        if len(section) > 3:
            correction_caps.append(float(section[3]))
        if len(section) > 6:
            momentum_rates.append(float(section[6]))
    max_iterations = int(ilc_global["max_iterations"])
    max_position_offset_m = (
        max(correction_caps)
        * max_iterations
        * max(1.0, max(learning_rates))
        * (1.0 + max(0.0, max(momentum_rates)) * max_iterations)
    )

    ilc_inputs = {
        "trajectory_key": trajectory_key,
        "start_position": start_position,
        "dt": dt,
        "global": ilc_global,
        "sections": section_boundaries,
    }
    ilc_key = artifact_key(
        "ilc",
        ilc_inputs,
        schema_version="ilc-offsets-v2",
        source_files=_ilc_source_files(),
        environment=environment,
    )
    lookup_started = time.perf_counter()
    ilc_arrays = store.load_npz("ilc", ilc_key)
    timings["cache_lookup"] += time.perf_counter() - lookup_started
    ilc_decode_policy = {
        "expected_steps": total_steps,
        "max_position_offset_m": max_position_offset_m,
        "dt": dt,
    }
    decoded_ilc = _decode_ilc_arrays(ilc_arrays, **ilc_decode_policy)
    ilc_build_time = 0.0
    if decoded_ilc is None:
        with store.lock("ilc", ilc_key):
            lookup_started = time.perf_counter()
            ilc_arrays = store.load_npz("ilc", ilc_key)
            timings["cache_lookup"] += time.perf_counter() - lookup_started
            decoded_ilc = _decode_ilc_arrays(ilc_arrays, **ilc_decode_policy)
            if decoded_ilc is None:
                build_started = time.perf_counter()
                computed = compute_ilc_offset_table(
                    trajectory,
                    tuple(float(value) for value in start_position),
                    alpha=ilc_global["alpha"],
                    max_iterations=ilc_global["max_iterations"],
                    smoothing_sigma=ilc_global["smoothing_sigma"],
                    max_correction_m=ilc_global["max_correction_m"],
                    convergence_threshold=ilc_global["convergence_threshold"],
                    dt=dt,
                    section_boundaries=section_boundaries,
                    blend_steps=ilc_global["blend_steps"],
                    filter_cutoff_hz=ilc_global["filter_cutoff_hz"],
                    momentum_gamma=ilc_global["momentum_gamma"],
                )
                if computed is None:
                    position_offsets = np.empty((0, 3), dtype=np.float64)
                    velocity_offsets = np.empty((0, 3), dtype=np.float64)
                    present = np.asarray([0], dtype=np.uint8)
                else:
                    position_offsets, velocity_offsets = computed
                    present = np.asarray([1], dtype=np.uint8)
                ilc_arrays = {
                    "present": present,
                    "position_offsets": np.asarray(position_offsets, dtype=np.float64),
                    "velocity_offsets": np.asarray(velocity_offsets, dtype=np.float64),
                }
                store.save_npz("ilc", ilc_key, ilc_arrays)
                decoded_ilc = _decode_ilc_arrays(ilc_arrays, **ilc_decode_policy)
                ilc_build_time += time.perf_counter() - build_started
        states["ilc"] = "miss"
    else:
        states["ilc"] = "hit"
    timings["ilc"] = ilc_build_time
    if decoded_ilc is None:  # defensive: a freshly encoded artifact must validate
        raise RuntimeError(f"failed to build a valid ILC artifact {ilc_key}")
    ilc_offsets, ilc_velocity_offsets = decoded_ilc

    resolved_planning = {
        "racing_line": {
            key: value
            for key, value in dataclasses.asdict(racing_config).items()
            if key != "cache_root"
        },
        "planner": dataclasses.asdict(planner_config_resolved),
        "drone": dataclasses.asdict(drone_constraints),
        "fov": dataclasses.asdict(fov_config),
        "sequencer": dataclasses.asdict(sequencer_config),
        "ilc": {"global": ilc_global, "sections": section_boundaries},
        "dt": dt,
        "trajectory_sample_dt": 0.02,
    }
    config_hash = sha256_json({"track": track_data, "planning": resolved_planning})
    prepared_key = artifact_key(
        "prepared-courses",
        {
            "racing_line": racing_key,
            "trajectory": trajectory_key,
            "plan_validation": validation_key,
            "ilc": ilc_key,
            "gate_geometry": [dataclasses.asdict(gate) for gate in gate_specs],
            "start": start_position,
        },
        schema_version="prepared-course-v2",
        source_files=[Path(__file__)],
        environment=environment,
    )
    return PreparedCourse(
        track=track_data,
        gate_specs=gate_specs,
        gate_waypoints=gate_waypoints,
        sequencer_config=sequencer_config,
        optimized_waypoints=optimized_waypoints,
        start_position=start_position,
        max_velocity=float(drone_constraints.max_velocity),
        drone_constraints=drone_constraints,
        trajectory=trajectory,
        plan_validation=dict(validation_payload),
        ilc_offsets=ilc_offsets,
        ilc_velocity_offsets=ilc_velocity_offsets,
        dt=dt,
        resolved_planning_config=resolved_planning,
        config_hash=config_hash,
        artifact_key=prepared_key,
        artifact_keys={
            "racing_line": racing_key,
            "trajectory": trajectory_key,
            "plan_validation": validation_key,
            "ilc": ilc_key,
        },
        cache_states=states,
        phase_timings_s=timings,
        dependency_fingerprint=environment,
    )


def _resolved_tracker_config(
    track: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]],
    drone_constraints: Optional[Any] = None,
) -> Any:
    from competition.drone_spec import DEFAULT_LINEAR_DRAG_PER_MASS
    from control.mpc_tracker import TrackerConfig

    tracker_values: Dict[str, Any] = {
        "kp_xy": 7.0,
        "kd_xy": 5.5,
        "kp_z": 8.0,
        "kd_z": 5.0,
        "feedforward_accel": 0.50,
        "velocity_feedforward": 0.0,
        # The synthetic plant below applies the spec's known linear drag.
        # Compensate 90% of that modeled forcing while retaining a 10%
        # passive-damping/model-error margin. Exact 100% cancellation reduced
        # RMS error but overtracked figure8 into gate 2; this conservative
        # model-based margin completed every prepared course in the matrix.
        # TrackerConfig's production default remains OFF, so this does not
        # silently enable an uncalibrated feedforward term on FlightSim.
        "use_drag_ff": True,
        "drag_ff_coeff": 0.9 * DEFAULT_LINEAR_DRAG_PER_MASS,
    }
    if drone_constraints is not None:
        tracker_values.update(
            {
                "max_tilt_rad": drone_constraints.max_tilt_angle,
                "max_body_rate": drone_constraints.max_body_rate,
                "mass": drone_constraints.mass,
                "gravity": drone_constraints.gravity,
                "max_thrust_n": drone_constraints.max_thrust,
            }
        )
    tracker_values.update(track.get("tracker_overrides", {}))
    tracker_values.update(dict(overrides or {}))
    return _dataclass_from_overrides(TrackerConfig, tracker_values)


def simulate(
    prepared: PreparedCourse,
    controller_config: Optional[Mapping[str, Any]] = None,
    seed: int = 42,
    *,
    duration: float = 30.0,
    dt: Optional[float] = None,
    record_position_trace: bool = False,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Roll out one controller against an already prepared deterministic course."""

    if dt is None:
        dt = prepared.dt
    dt = _exact_finite_float("dt", dt, strictly_positive=True)
    if not math.isclose(dt, prepared.dt, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"simulate dt={dt} differs from prepared dt={prepared.dt}; "
            "prepare a new course so validation and ILC stay aligned"
        )
    duration = _exact_finite_float("duration", duration, nonnegative=True)
    seed = _exact_nonnegative_int("seed", seed)
    record_position_trace = _exact_bool(
        "record_position_trace", record_position_trace
    )
    if controller_config is not None and not isinstance(controller_config, Mapping):
        raise TypeError("controller_config must be a mapping or None")
    threshold_values = _threshold_snapshot(thresholds)

    from competition.aigp_geometry import AIGP_VQ1_MAX_RUN_DURATION_S
    from competition.drone_spec import (
        DEFAULT_LINEAR_DRAG_PER_MASS,
        DEFAULT_YAW_RATE_MAX_RAD_S,
    )
    from control.mpc_tracker import GeometricTracker
    from estimation.ekf import DroneEKF, EKFConfig
    from gate_sequencing.sequencer import GateSequencer
    from planning.trajectory_optimizer import TrajectoryPoint

    rng = np.random.default_rng(seed)
    sequence = GateSequencer(
        prepared.gate_specs,
        prepared.sequencer_config,
    )
    sequence.start()
    estimator = DroneEKF(EKFConfig())
    estimator.initialize(tuple(prepared.start_position), (0, 0, 0), timestamp_s=0.0)
    tracker_config = _resolved_tracker_config(
        prepared.track, controller_config, prepared.drone_constraints
    )
    tracker = GeometricTracker(tracker_config)
    trajectory = prepared.trajectory

    position = prepared.start_position.copy()
    velocity = np.zeros(3)
    yaw = 0.0
    max_acceleration = float(prepared.drone_constraints.max_acceleration)
    max_speed = float(prepared.drone_constraints.max_velocity)
    drag = DEFAULT_LINEAR_DRAG_PER_MASS
    yaw_rate_max = DEFAULT_YAW_RATE_MAX_RAD_S
    feedforward_lookahead_s = 0.05

    tracking_errors: List[float] = []
    nearest_path_errors: List[float] = []
    loop_times: List[float] = []
    gate_pass_times: List[Dict[str, Any]] = []
    per_gate_errors: Dict[str, List[float]] = {}
    controller_trace: List[Dict[str, Any]] = []
    position_trace: Optional[List[Dict[str, Any]]] = [] if record_position_trace else None
    crashed = False
    termination_reason = "time_limit"
    number_steps = int(duration / dt)
    final_simulation_time = 0.0
    rollout_started = time.perf_counter()

    for step in range(number_steps):
        loop_started = time.perf_counter()
        simulation_time = step * dt
        # The state inspected by sequencing and terminal checks is exactly at
        # this pre-integration timestamp. If a terminal fires, do not report a
        # fictitious extra dt that was never integrated.
        final_simulation_time = simulation_time
        estimator.predict(
            (0, 0, -float(prepared.drone_constraints.gravity)),
            (0, 0, 0),
            simulation_time,
        )
        noisy_position = tuple(
            value + rng.normal(0, 0.005) for value in position
        )
        noisy_velocity = tuple(
            value + rng.normal(0, 0.01) for value in velocity
        )
        estimator.update_odometry(noisy_position, noisy_velocity)

        passed = sequence.update(tuple(position))
        if passed:
            gate_pass_times.append(
                {"gate_id": passed.gate_id, "time_s": simulation_time}
            )
        if simulation_time > AIGP_VQ1_MAX_RUN_DURATION_S:
            sequence.mark_timed_out(
                f"vq1_max_run_duration_exceeded:{simulation_time:.1f}s"
            )
            termination_reason = f"timed_out:{sequence.timeout_reason}"
            break
        if sequence.last_crash is not None:
            crashed = True
            termination_reason = f"crash_gate:{sequence.last_crash[0]}"
            break
        if sequence.is_disqualified:
            termination_reason = f"disqualified:{sequence.dq_reason}"
            break
        if position[2] < 0.05:
            crashed = True
            termination_reason = "crash_ground"
            break
        if position[2] > 20.0:
            crashed = True
            termination_reason = "crash_ceiling"
            break
        if sequence.is_complete:
            termination_reason = "race_complete"
            break

        reference = trajectory.sample(simulation_time)
        target_position = np.asarray(reference.position, dtype=float)
        if prepared.ilc_offsets is not None and step < len(prepared.ilc_offsets):
            target_position = target_position + prepared.ilc_offsets[step]
        target_velocity = np.asarray(reference.velocity, dtype=float)
        if (
            prepared.ilc_velocity_offsets is not None
            and step < len(prepared.ilc_velocity_offsets)
        ):
            target_velocity = target_velocity + prepared.ilc_velocity_offsets[step]
        target_yaw = reference.yaw

        if simulation_time > trajectory.total_time and not sequence.is_complete:
            gate = sequence.current_gate
            if gate:
                gate_position = np.asarray(gate.position, dtype=float)
                displacement = gate_position - position
                distance = float(np.linalg.norm(displacement))
                if distance > 0.1:
                    target_position = gate_position
                    target_velocity = displacement / distance * min(distance * 2, 5.0)
                    target_yaw = float(math.atan2(displacement[1], displacement[0]))

        fallback = simulation_time > trajectory.total_time and not sequence.is_complete
        if not fallback and feedforward_lookahead_s > 0:
            ahead = trajectory.sample(simulation_time + feedforward_lookahead_s)
            feedforward_acceleration = ahead.acceleration
            feedforward_jerk = ahead.jerk
        else:
            feedforward_acceleration = (
                (0, 0, 0) if fallback else reference.acceleration
            )
            feedforward_jerk = (0, 0, 0) if fallback else reference.jerk
        reference_point = TrajectoryPoint(
            time=simulation_time,
            position=tuple(target_position),
            velocity=tuple(target_velocity),
            acceleration=feedforward_acceleration,
            jerk=feedforward_jerk,
            yaw=target_yaw,
            yaw_rate=0.0 if fallback else reference.yaw_rate,
        )
        command = tracker.track(
            tuple(position), tuple(velocity), yaw, reference_point
        )
        controller_trace.append(
            {
                "t": simulation_time,
                "roll": command.roll_rad,
                "pitch": command.pitch_rad,
                "thrust": command.thrust,
            }
        )
        desired_acceleration = tracker.last_desired_acceleration
        acceleration = (
            np.asarray(desired_acceleration, dtype=float) - drag * velocity
            if desired_acceleration is not None
            else -drag * velocity
        )
        magnitude_before_clamp = float(np.linalg.norm(acceleration))
        acceleration_clamped = magnitude_before_clamp > max_acceleration
        if acceleration_clamped:
            acceleration = acceleration / magnitude_before_clamp * max_acceleration
        controller_trace[-1]["accel_mag_pre_clamp"] = magnitude_before_clamp
        controller_trace[-1]["accel_clamp_active"] = acceleration_clamped

        velocity = velocity + acceleration * dt
        speed_before_clamp = float(np.linalg.norm(velocity))
        speed_clamped = speed_before_clamp > max_speed
        if speed_clamped:
            velocity = velocity / speed_before_clamp * max_speed
        controller_trace[-1]["speed_pre_clamp"] = speed_before_clamp
        controller_trace[-1]["speed_clamp_active"] = speed_clamped
        position = position + velocity * dt
        post_step_time = (step + 1) * dt
        final_simulation_time = post_step_time

        yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        yaw += float(
            np.clip(yaw_error * 3.0, -yaw_rate_max * dt, yaw_rate_max * dt)
        )
        # The current evaluator scores the reference actually commanded at
        # this step.
        # The historical globally-nearest-path metric flattered schedule lag
        # and could jump branches at self-crossings. Keep it only as a clearly
        # named geometric diagnostic.
        tracking_error = float(np.linalg.norm(position - target_position))
        closest = trajectory.find_closest(tuple(position))
        nearest_path_error = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(position, closest.position))
        )
        tracking_errors.append(tracking_error)
        nearest_path_errors.append(nearest_path_error)
        if position_trace is not None:
            position_trace.append(
                {
                    # Position/velocity/yaw are the post-integration state.
                    # Controller trace entries remain at the pre-step command
                    # timestamp above.
                    "t": post_step_time,
                    "pos": [float(value) for value in position],
                    "vel": [float(value) for value in velocity],
                    "yaw": float(yaw),
                    "tracking_err_m": float(tracking_error),
                }
            )
        current_gate = sequence.current_gate
        if current_gate:
            per_gate_errors.setdefault(current_gate.gate_id, []).append(
                tracking_error
            )
        loop_times.append(time.perf_counter() - loop_started)

    rollout_wall = time.perf_counter() - rollout_started
    metrics_started = time.perf_counter()
    average_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
    maximum_error = float(np.max(tracking_errors)) if tracking_errors else 0.0
    p50_error = float(np.percentile(tracking_errors, 50)) if tracking_errors else 0.0
    p95_error = float(np.percentile(tracking_errors, 95)) if tracking_errors else 0.0
    average_nearest_path_error = (
        float(np.mean(nearest_path_errors)) if nearest_path_errors else 0.0
    )
    maximum_nearest_path_error = (
        float(np.max(nearest_path_errors)) if nearest_path_errors else 0.0
    )
    average_hz = 1.0 / np.mean(loop_times) if loop_times else 0.0
    validation = prepared.plan_validation
    trace_summary = {
        "samples": len(controller_trace),
        "avg_roll_rad": float(np.mean([item["roll"] for item in controller_trace])),
        "avg_pitch_rad": float(np.mean([item["pitch"] for item in controller_trace])),
        "avg_thrust": float(np.mean([item["thrust"] for item in controller_trace])),
        "max_abs_roll_rad": float(
            np.max([abs(item["roll"]) for item in controller_trace])
        ),
        "max_abs_pitch_rad": float(
            np.max([abs(item["pitch"]) for item in controller_trace])
        ),
        "accel_clamp_active_frac": float(
            np.mean([item["accel_clamp_active"] for item in controller_trace])
        ),
        "speed_clamp_active_frac": float(
            np.mean([item["speed_clamp_active"] for item in controller_trace])
        ),
        "max_accel_mag_pre_clamp": float(
            np.max([item["accel_mag_pre_clamp"] for item in controller_trace])
        ),
    } if controller_trace else {}
    result: Dict[str, Any] = {
        "evaluator_version": EVALUATOR_VERSION,
        "schema_version": BENCHMARK_RESULT_SCHEMA,
        "available": True,
        "skipped": False,
        "sim_type": "synthetic_kinematic",
        "trajectory_time_s": trajectory.total_time,
        "trajectory_points": len(trajectory.points),
        "sim_time_s": final_simulation_time,
        "rollout_wall_time_s": rollout_wall,
        "rollout_materialization_wall_time_s": rollout_wall,
        "dt": dt,
        "seed": int(seed),
        "termination_reason": termination_reason,
        "crashed": crashed,
        "disqualified": bool(sequence.is_disqualified),
        "dq_reason": sequence.dq_reason,
        "last_crash_gate": sequence.last_crash[0] if sequence.last_crash else None,
        "plan_validation": dict(validation),
        "gates_passed": sequence.gates_passed,
        "total_gates": sequence.total_gates,
        "gate_pass_rate": (
            sequence.gates_passed / sequence.total_gates
            if sequence.total_gates > 0
            else 0.0
        ),
        "complete": sequence.is_complete,
        "gate_pass_times": gate_pass_times,
        "avg_tracking_error_m": average_error,
        "max_tracking_error_m": maximum_error,
        "p50_tracking_error_m": p50_error,
        "p95_tracking_error_m": p95_error,
        "avg_nearest_path_error_m": average_nearest_path_error,
        "max_nearest_path_error_m": maximum_nearest_path_error,
        "ekf_uncertainty_m": float(estimator.position_uncertainty),
        "avg_loop_hz": float(average_hz),
        "total_steps": len(loop_times),
        "per_gate_avg_error": {
            gate_id: float(np.mean(errors))
            for gate_id, errors in per_gate_errors.items()
        },
        "controller_trace_summary": trace_summary,
        "tracker_feature_trace": (
            list(tracker.feature_trace)
            if getattr(tracker, "feature_trace", None)
            else []
        ),
        "position_trace": position_trace,
        "resolved_controller_config": dataclasses.asdict(tracker_config),
        "thresholds": threshold_values,
    }
    failures: List[str] = []
    if crashed:
        failures.append(f"drone crashed ({termination_reason})")
    if sequence.is_disqualified:
        failures.append(f"drone disqualified ({sequence.dq_reason})")
    if validation.get("ok") is not True:
        failures.append(
            "plan_validation failed "
            f"({validation.get('reason', 'missing/invalid validation result')})"
        )
    if not sequence.is_complete:
        failures.append(
            f"race incomplete ({sequence.gates_passed}/{sequence.total_gates} gates)"
        )
    if average_error > threshold_values["max_avg_tracking_error_m"]:
        failures.append(
            f"avg_tracking_error {average_error:.2f}m > "
            f"{threshold_values['max_avg_tracking_error_m']}m"
        )
    if maximum_error > threshold_values["max_max_tracking_error_m"]:
        failures.append(
            f"max_tracking_error {maximum_error:.2f}m > "
            f"{threshold_values['max_max_tracking_error_m']}m"
        )
    if float(estimator.position_uncertainty) > threshold_values["max_ekf_uncertainty_m"]:
        failures.append(
            f"ekf_uncertainty {estimator.position_uncertainty:.3f}m > "
            f"{threshold_values['max_ekf_uncertainty_m']}m"
        )
    if average_hz < threshold_values["min_loop_hz"]:
        failures.append(
            f"loop_hz {average_hz:.0f} < {threshold_values['min_loop_hz']}"
        )
    gate_rate = (
        sequence.gates_passed / sequence.total_gates
        if sequence.total_gates > 0
        else 0.0
    )
    if gate_rate < threshold_values["min_gate_pass_rate"]:
        failures.append(
            f"gate_pass_rate {gate_rate:.0%} < "
            f"{threshold_values['min_gate_pass_rate']:.0%}"
        )
    if final_simulation_time > threshold_values["max_total_time_s"]:
        failures.append(
            f"race_time {final_simulation_time:.1f}s > "
            f"{threshold_values['max_total_time_s']}s"
        )
    result["threshold_failures"] = failures
    result["sim_passed"] = (
        len(failures) == 0
        and not crashed
        and not sequence.is_disqualified
        and validation.get("ok") is True
        and sequence.is_complete
    )
    result["safety_passed"] = not crashed and not sequence.is_disqualified
    result["validity_passed"] = validation.get("ok") is True
    result["completion"] = {
        "complete": sequence.is_complete,
        "gates_passed": sequence.gates_passed,
        "total_gates": sequence.total_gates,
    }
    result["failure_summary"] = {
        "stdout_tail": "",
        "stderr_tail": "",
        "exception": None,
        "threshold_failures": list(failures),
    }
    result["_simulate_phase_timings_s"] = {
        "rollout": rollout_wall,
        "metrics": time.perf_counter() - metrics_started,
    }
    # Normalize tuples/numpy scalar subclasses so cold and JSON-cache-hit
    # results have identical value types.
    normalized = json.loads(
        json.dumps(result, allow_nan=False, default=_json_numpy_default)
    )
    return _restore_tracker_feature_arrays(normalized)


def run_synthetic_benchmark(
    duration: float = 30.0,
    dt: float = 0.01,
    config: Optional[Dict[str, Any]] = None,
    tracker_config_overrides: Optional[Dict[str, Any]] = None,
    record_position_trace: bool = False,
    *,
    planning_config: Optional[Mapping[str, Any]] = None,
    cache_root: Optional[os.PathLike[str] | str] = None,
    seed: int = 42,
    use_result_cache: bool = True,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare then simulate a synthetic course with honest end-to-end metadata."""

    from planning.artifact_cache import (
        ArtifactStore,
        artifact_key,
        dependency_fingerprint,
        sha256_json,
    )

    total_started = time.perf_counter()
    duration = _exact_finite_float("duration", duration, nonnegative=True)
    dt = _exact_finite_float("dt", dt, strictly_positive=True)
    seed = _exact_nonnegative_int("seed", seed)
    record_position_trace = _exact_bool(
        "record_position_trace", record_position_trace
    )
    use_result_cache = _exact_bool("use_result_cache", use_result_cache)
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or None")
    if planning_config is not None and not isinstance(planning_config, Mapping):
        raise TypeError("planning_config must be a mapping or None")
    if tracker_config_overrides is not None and not isinstance(
        tracker_config_overrides, Mapping
    ):
        raise TypeError("tracker_config_overrides must be a mapping or None")
    threshold_values = _threshold_snapshot(thresholds)
    config_started = time.perf_counter()
    if config is None:
        config_path = _REPO / "sim_pybullet" / "configs" / "race_01.json"
        try:
            config_data = _read_strict_json_object(
                config_path, context="synthetic track config"
            )
        except FileNotFoundError:
            config_load_time = time.perf_counter() - config_started
            runtime_config = {
                "duration": duration,
                "dt": dt,
                "seed": seed,
                "record_position_trace": record_position_trace,
                "thresholds": threshold_values,
            }
            resolved_configuration = {
                "track": {"config_path": str(config_path), "status": "missing"},
                "planning": planning_config,
                "controller": tracker_config_overrides,
                "runtime": runtime_config,
            }
            failure = f"Config not found: {config_path}"
            phases = {
                "startup": 0.0,
                "config_load": config_load_time,
                "cache_lookup": 0.0,
                "racing_line": 0.0,
                "trajectory": 0.0,
                "plan_validation": 0.0,
                "ilc": 0.0,
                "rollout": 0.0,
                "metrics": 0.0,
                "total_wall": 0.0,
            }
            result = {
                "available": False,
                "skipped": True,
                "skip_reason": failure,
                "sim_passed": False,
                "evaluator_version": EVALUATOR_VERSION,
                "schema_version": BENCHMARK_RESULT_SCHEMA,
                "comparison_series": COMPARISON_SERIES,
                "seed": int(seed),
                "thresholds": threshold_values,
                "resolved_configuration": resolved_configuration,
                "config_hash": sha256_json(resolved_configuration),
                "dependency_fingerprint": dependency_fingerprint(),
                "cache_hit_or_miss": "not_applicable",
                "cache": {"status": "not_applicable"},
                "phase_timings_s": phases,
                "timing_scope": FUNCTION_TIMING_SCOPE,
                "phase_timing_notes": dict(PHASE_TIMING_NOTES),
                "threshold_failures": [failure],
                "safety_passed": False,
                "validity_passed": False,
                "completion": {
                    "complete": False,
                    "gates_passed": 0,
                    "total_gates": 0,
                },
                "failure_summary": {
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "exception": failure,
                    "threshold_failures": [failure],
                },
                "code_provenance": _git_provenance(),
            }
            result["wall_time_s"] = _finalize_phase_timings(
                phases, total_started
            )
            result["timing_consistency"] = {
                "mutually_exclusive_phase_sum_s": sum(
                    float(value)
                    for name, value in phases.items()
                    if name != "total_wall"
                ),
                "total_covers_phases": True,
            }
            return result
    else:
        config_data = json.loads(json.dumps(config, allow_nan=False))
    config_load_time = time.perf_counter() - config_started

    prepared = prepare_course(
        config_data, planning_config, dt=dt, cache_root=cache_root
    )
    tracker_config = _resolved_tracker_config(
        config_data, tracker_config_overrides, prepared.drone_constraints
    )
    controller_resolved = dataclasses.asdict(tracker_config)
    runtime_config = {
        "duration": duration,
        "dt": dt,
        "seed": seed,
        "record_position_trace": record_position_trace,
        "thresholds": threshold_values,
    }
    resolved_configuration = {
        "track": config_data,
        "planning": prepared.resolved_planning_config,
        "controller": controller_resolved,
        "runtime": runtime_config,
    }
    resolved_config_hash = sha256_json(resolved_configuration)
    result_sources = _benchmark_result_source_files(controller_resolved)
    result_key = artifact_key(
        "benchmark-results",
        {
            "prepared_course": prepared.artifact_key,
            "controller": controller_resolved,
            "runtime": runtime_config,
            "evaluator_version": EVALUATOR_VERSION,
        },
        schema_version=BENCHMARK_RESULT_SCHEMA,
        source_files=result_sources,
        environment=prepared.dependency_fingerprint,
    )
    store = ArtifactStore(cache_root)
    result_cache_lookup = 0.0
    cached_result = None
    if use_result_cache:
        lookup_started = time.perf_counter()
        cached_result = store.load_json("benchmark-results", result_key)
        result_cache_lookup += time.perf_counter() - lookup_started

    simulated = False
    if not _valid_cached_benchmark_result(
        cached_result,
        expected_seed=int(seed),
        expected_thresholds=threshold_values,
        expected_gate_ids=[
            gate.gate_id
            for gate in sorted(
                prepared.gate_specs, key=lambda gate: gate.sequence_index
            )
        ],
        expected_dt=dt,
        expected_controller_config=controller_resolved,
        expected_record_position_trace=record_position_trace,
    ):
        if use_result_cache:
            with store.lock("benchmark-results", result_key):
                lookup_started = time.perf_counter()
                cached_result = store.load_json("benchmark-results", result_key)
                result_cache_lookup += time.perf_counter() - lookup_started
                if not _valid_cached_benchmark_result(
                    cached_result,
                    expected_seed=int(seed),
                    expected_thresholds=threshold_values,
                    expected_gate_ids=[
                        gate.gate_id
                        for gate in sorted(
                            prepared.gate_specs,
                            key=lambda gate: gate.sequence_index,
                        )
                    ],
                    expected_dt=dt,
                    expected_controller_config=controller_resolved,
                    expected_record_position_trace=record_position_trace,
                ):
                    cached_result = simulate(
                        prepared,
                        tracker_config_overrides,
                        seed,
                        duration=duration,
                        dt=dt,
                        record_position_trace=record_position_trace,
                        thresholds=threshold_values,
                    )
                    store.save_json("benchmark-results", result_key, cached_result)
                    simulated = True
        else:
            cached_result = simulate(
                prepared,
                tracker_config_overrides,
                seed,
                duration=duration,
                dt=dt,
                record_position_trace=record_position_trace,
                thresholds=threshold_values,
            )
            simulated = True

    result = dict(cached_result)
    simulation_timings = result.pop("_simulate_phase_timings_s", {})
    result["rollout_wall_time_s"] = (
        float(result["rollout_materialization_wall_time_s"])
        if simulated
        else 0.0
    )
    phases = {
        "startup": 0.0,
        "config_load": config_load_time,
        "cache_lookup": prepared.phase_timings_s.get("cache_lookup", 0.0)
        + result_cache_lookup,
        "racing_line": prepared.phase_timings_s.get("racing_line", 0.0),
        "trajectory": prepared.phase_timings_s.get("trajectory", 0.0),
        "plan_validation": prepared.phase_timings_s.get("plan_validation", 0.0),
        "ilc": prepared.phase_timings_s.get("ilc", 0.0),
        "rollout": (
            float(simulation_timings.get("rollout", 0.0)) if simulated else 0.0
        ),
        "metrics": (
            float(simulation_timings.get("metrics", 0.0)) if simulated else 0.0
        ),
        "total_wall": 0.0,
    }
    cache_states = dict(prepared.cache_states)
    cache_states["benchmark_result"] = "miss" if simulated else "hit"
    provenance = _git_provenance()
    result.update(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evaluator_version": EVALUATOR_VERSION,
            "schema_version": BENCHMARK_RESULT_SCHEMA,
            "comparison_series": COMPARISON_SERIES,
            "phase_timings_s": phases,
            "timing_scope": FUNCTION_TIMING_SCOPE,
            "phase_timing_notes": dict(PHASE_TIMING_NOTES),
            "wall_time_s": 0.0,
            "cache_hit_or_miss": "miss" if simulated else "hit",
            "cache": cache_states,
            "rollout_executed": simulated,
            "result_cache_enabled": bool(use_result_cache),
            "resolved_configuration": resolved_configuration,
            "config_hash": resolved_config_hash,
            "planning_config_hash": prepared.config_hash,
            "artifact_hashes": {
                **prepared.artifact_keys,
                "prepared_course": prepared.artifact_key,
                "benchmark_result": result_key,
            },
            "dependency_fingerprint": prepared.dependency_fingerprint,
            "code_provenance": provenance,
            "seed": int(seed),
            "thresholds": threshold_values,
        }
    )
    total_wall = _finalize_phase_timings(phases, total_started)
    result["wall_time_s"] = total_wall
    result["timing_consistency"] = {
        "mutually_exclusive_phase_sum_s": sum(
            float(value)
            for name, value in phases.items()
            if name != "total_wall"
        ),
        "total_covers_phases": True,
    }
    return _restore_tracker_feature_arrays(result)


# ---------------------------------------------------------------------------
# PyBullet simulation benchmark
# ---------------------------------------------------------------------------


def _positive_sim_duration(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("PyBullet duration must be a number, not bool/string")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("PyBullet duration must be finite and strictly positive")
    return resolved


def _observed_sim_delta(previous: Optional[float], current: Any) -> float:
    """Return elapsed simulator time and fail closed on a stalled/reversed clock."""

    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise TypeError("simulator time must be numeric")
    now = float(current)
    if not math.isfinite(now) or now < 0.0:
        raise ValueError("simulator time must be finite and non-negative")
    if previous is None:
        return 0.0
    delta = now - float(previous)
    if not math.isfinite(delta) or delta <= 0.0:
        raise RuntimeError(
            f"simulator clock did not advance monotonically: previous={previous}, now={now}"
        )
    return delta


def _physics_dependency_fingerprint() -> Dict[str, Any]:
    """Extend the numeric fingerprint with the packages that define physics."""

    import importlib.metadata
    import importlib.util

    from planning.artifact_cache import dependency_fingerprint, source_digest

    packages: Dict[str, Any] = {}
    for distribution, module_name in (
        ("pybullet", "pybullet"),
        ("gym-pybullet-drones", "gym_pybullet_drones"),
    ):
        entry: Dict[str, Any] = {
            "distribution": distribution,
            "version": None,
            "module_origin_sha256": None,
            "vcs_commit": None,
        }
        try:
            dist = importlib.metadata.distribution(distribution)
            entry["version"] = dist.version
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                direct_payload = json.loads(direct_url)
                vcs_info = direct_payload.get("vcs_info", {})
                commit = vcs_info.get("commit_id")
                entry["vcs_commit"] = commit if isinstance(commit, str) else None
        except (importlib.metadata.PackageNotFoundError, ValueError, json.JSONDecodeError):
            pass
        try:
            spec = importlib.util.find_spec(module_name)
            origin = Path(spec.origin).resolve() if spec and spec.origin else None
            if origin is not None and origin.is_file():
                entry["module_origin_sha256"] = hashlib.sha256(
                    origin.read_bytes()
                ).hexdigest()
        except (ImportError, ModuleNotFoundError, OSError, ValueError):
            pass
        packages[module_name] = entry

    fingerprint = dependency_fingerprint()
    fingerprint["physics_dependencies"] = packages
    fingerprint["physics_adapter_source_digest"] = source_digest(
        [
            _REPO / "sim_pybullet" / "env.py",
            _REPO / "sim_pybullet" / "gpd_drone.py",
            _REPO / "sim_pybullet" / "gate_models.py",
        ]
    )
    return fingerprint


def _pybullet_threshold_failures(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> List[str]:
    failures: List[str] = []
    for name in ("crashed", "disqualified", "complete"):
        if type(metrics.get(name)) is not bool:
            failures.append(f"{name} evidence is not an exact bool")
    if metrics.get("crashed") is True:
        failures.append(f"drone crashed ({metrics['termination_reason']})")
    if metrics.get("disqualified") is True:
        failures.append(f"drone disqualified ({metrics.get('dq_reason')})")
    validation = metrics.get("plan_validation")
    if not isinstance(validation, Mapping) or validation.get("ok") is not True:
        reason = validation.get("reason", "missing") if isinstance(validation, Mapping) else "missing"
        failures.append(f"plan_validation failed ({reason})")
    numeric: Dict[str, Optional[float]] = {}
    for name in (
        "avg_tracking_error_m",
        "max_tracking_error_m",
        "ekf_uncertainty_m",
        "avg_loop_hz",
        "gate_pass_rate",
        "sim_time_s",
    ):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            failures.append(f"{name} must be a finite non-negative number")
            numeric[name] = None
        else:
            numeric[name] = float(value)
    gate_rate = numeric["gate_pass_rate"]
    if gate_rate is not None and gate_rate > 1.0:
        failures.append("gate_pass_rate must be within [0, 1]")
    for name in ("gates_passed", "total_gates"):
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            failures.append(f"{name} must be a non-negative exact integer")
    if (
        type(metrics.get("gates_passed")) is int
        and type(metrics.get("total_gates")) is int
        and metrics["gates_passed"] > metrics["total_gates"]
    ):
        failures.append("gates_passed exceeds total_gates")
    counts_valid = (
        type(metrics.get("gates_passed")) is int
        and type(metrics.get("total_gates")) is int
        and metrics["total_gates"] > 0
        and 0 <= metrics["gates_passed"] <= metrics["total_gates"]
    )
    if counts_valid and gate_rate is not None:
        expected_rate = metrics["gates_passed"] / metrics["total_gates"]
        if not math.isclose(gate_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
            failures.append("gate_pass_rate contradicts gate counts")
    if metrics.get("complete") is True and (
        not counts_valid or metrics["gates_passed"] != metrics["total_gates"]
    ):
        failures.append("complete=true contradicts gate counts")

    avg_error = numeric["avg_tracking_error_m"]
    max_error = numeric["max_tracking_error_m"]
    uncertainty = numeric["ekf_uncertainty_m"]
    loop_hz = numeric["avg_loop_hz"]
    sim_time = numeric["sim_time_s"]
    if avg_error is not None and avg_error > thresholds["max_avg_tracking_error_m"]:
        failures.append(
            f"avg_tracking_error {avg_error:.2f}m > {thresholds['max_avg_tracking_error_m']}m"
        )
    if max_error is not None and max_error > thresholds["max_max_tracking_error_m"]:
        failures.append(
            f"max_tracking_error {max_error:.2f}m > {thresholds['max_max_tracking_error_m']}m"
        )
    if uncertainty is not None and uncertainty > thresholds["max_ekf_uncertainty_m"]:
        failures.append(
            f"ekf_uncertainty {uncertainty:.3f}m > {thresholds['max_ekf_uncertainty_m']}m"
        )
    if loop_hz is not None and loop_hz < thresholds["min_loop_hz"]:
        failures.append(f"loop_hz {loop_hz:.0f} < {thresholds['min_loop_hz']}")
    if gate_rate is not None and gate_rate < thresholds["min_gate_pass_rate"]:
        failures.append(
            f"gate_pass_rate {gate_rate:.0%} < {thresholds['min_gate_pass_rate']:.0%}"
        )
    if metrics.get("complete") is not True:
        failures.append(
            f"race incomplete ({metrics.get('gates_passed')}/{metrics.get('total_gates')} gates)"
        )
    if sim_time is not None and sim_time > thresholds["max_total_time_s"]:
        failures.append(
            f"race_time {sim_time:.3f}s > {thresholds['max_total_time_s']}s"
        )
    return failures

def run_sim_benchmark(
    config_path: str,
    duration: float,
    *,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full pipeline against PyBullet headless. Returns structured metrics."""
    from planning.artifact_cache import sha256_json

    total_started = time.perf_counter()
    duration = _positive_sim_duration(duration)
    threshold_values = _threshold_snapshot(thresholds)
    phases = {
        "startup": 0.0,
        "config_load": 0.0,
        "cache_lookup": 0.0,
        "racing_line": 0.0,
        "trajectory": 0.0,
        "plan_validation": 0.0,
        "ilc": 0.0,
        "rollout": 0.0,
        "metrics": 0.0,
        "total_wall": 0.0,
    }
    try:
        raw_config = _read_strict_json_object(
            Path(config_path), context="PyBullet track config"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raw_config = {"config_path": str(config_path)}
    resolved_configuration = {
        "track": raw_config,
        "duration": duration,
        "thresholds": threshold_values,
    }
    result: Dict[str, Any] = {
        "available": False,
        "skipped": False,
        "evaluator_version": PYBULLET_EVALUATOR_VERSION,
        "schema_version": BENCHMARK_RESULT_SCHEMA,
        "comparison_series": "pybullet-v3",
        "seed": None,
        "thresholds": threshold_values,
        "resolved_configuration": resolved_configuration,
        "config_hash": sha256_json(resolved_configuration),
        "dependency_fingerprint": _physics_dependency_fingerprint(),
        "cache_hit_or_miss": "not_applicable",
        "cache": {
            "status": "not_applicable",
            "reason": "PyBullet evaluator does not use prepared artifacts",
        },
        "timing_scope": FUNCTION_TIMING_SCOPE,
        "phase_timing_notes": dict(PHASE_TIMING_NOTES),
    }
    phases["startup"] = time.perf_counter() - total_started

    def _finish() -> Dict[str, Any]:
        failures = result.setdefault("threshold_failures", [])
        result.setdefault("crashed", False)
        result.setdefault("disqualified", False)
        result.setdefault("complete", False)
        result.setdefault(
            "plan_validation",
            {
                "ok": False,
                "reason": "plan validation was not executed",
                "gates_passed": 0,
                "total_gates": 0,
            },
        )
        executed = result.get("available") is True and result.get("skipped") is False
        result["safety_passed"] = (
            executed
            and result["crashed"] is False
            and result["disqualified"] is False
        )
        result["validity_passed"] = (
            executed
            and isinstance(result.get("plan_validation"), Mapping)
            and result["plan_validation"].get("ok") is True
            and result["disqualified"] is False
        )
        result["completion"] = {
            "complete": result.get("complete") is True,
            "gates_passed": (
                result.get("gates_passed")
                if type(result.get("gates_passed")) is int
                else 0
            ),
            "total_gates": (
                result.get("total_gates")
                if type(result.get("total_gates")) is int
                else 0
            ),
        }
        if result.get("skipped") is True:
            skip_failure = (
                "PyBullet skipped; full/sim mode requires an executed evaluator"
            )
            if skip_failure not in failures:
                failures.append(skip_failure)
            result["sim_passed"] = False
        else:
            result["sim_passed"] = (
                result.get("sim_passed") is True
                and executed
                and result["safety_passed"] is True
                and result["validity_passed"] is True
                and result["completion"]["complete"] is True
            )
        result["failure_summary"] = {
            "stdout_tail": "",
            "stderr_tail": "",
            "exception": result.get("skip_reason"),
            "threshold_failures": list(failures),
        }
        result["code_provenance"] = _git_provenance()
        total_wall = _finalize_phase_timings(phases, total_started)
        result["phase_timings_s"] = phases
        result["wall_time_s"] = total_wall
        result["timing_consistency"] = {
            "mutually_exclusive_phase_sum_s": sum(
                float(value)
                for name, value in phases.items()
                if name != "total_wall"
            ),
            "total_covers_phases": True,
        }
        return result

    try:
        from sim_pybullet.env import DroneRaceEnv
    except ImportError as e:
        result["skipped"] = True
        result["skip_reason"] = f"PyBullet not available: {e}"
        return _finish()

    config_started = time.perf_counter()
    try:
        race_config = DroneRaceEnv.load_config(config_path)
    except Exception as e:
        phases["config_load"] = time.perf_counter() - config_started
        result["skipped"] = True
        result["skip_reason"] = f"Cannot load config: {e}"
        return _finish()
    phases["config_load"] = time.perf_counter() - config_started

    env = None
    env_closed = False
    primary_error: Optional[BaseException] = None
    try:
        # Pipeline setup
        from competition.aigp_geometry import AIGP_VQ1_MAX_RUN_DURATION_S
        from estimation.ekf import DroneEKF, EKFConfig
        from gate_sequencing.sequencer import GateSequencer, GateSpec, SequencerConfig
        from planning.plan_validator import validate_trajectory
        from planning.trajectory_optimizer import (
            DroneConstraints,
            GateWaypoint,
            PlannerConfig,
            TrajectoryOptimizer,
        )
        from planning.racing_line import RacingLineConfig, RacingLineOptimizer
        from sim_pybullet.gpd_drone import GPDDroneConfig

        start_pos = race_config.start_position

        def _to_specs(gates):
            return [
                GateSpec(
                    gate_id=g.gate_id,
                    position=(g.pose.x, g.pose.y, g.pose.z),
                    yaw=g.pose.yaw, pitch=g.pose.pitch, roll=g.pose.roll,
                    interior_width=g.config.interior_width_m,
                    interior_height=g.config.interior_height_m,
                    border_width=g.config.border_width_m,
                    depth=g.config.depth_m,
                    sequence_index=(
                        g.sequence_index if g.sequence_index is not None else index
                    ),
                ) for index, g in enumerate(gates)
            ]

        def _to_waypoints(gates):
            out = []
            for g in gates:
                cy, sy = math.cos(g.pose.yaw), math.sin(g.pose.yaw)
                cp, sp = math.cos(g.pose.pitch), math.sin(g.pose.pitch)
                out.append(GateWaypoint(
                    position=(g.pose.x, g.pose.y, g.pose.z),
                    normal=(cy * cp, sy * cp, sp),
                    width=g.config.interior_width_m,
                    height=g.config.interior_height_m,
                    yaw=g.pose.yaw,
                ))
            return out

        gate_specs = _to_specs(race_config.gates)
        gate_waypoints = _to_waypoints(race_config.gates)

        seq_cfg = _dataclass_from_overrides(
            SequencerConfig,
            {
                **race_config.sequencer_overrides,
                "proximity_pass_distance": 0.0,
            },
        )
        seq = GateSequencer(gate_specs, config=seq_cfg)
        seq.start()

        ekf_cfg = EKFConfig()
        ekf = DroneEKF(ekf_cfg)
        ekf.initialize(start_pos, (0, 0, 0), timestamp_s=0.0)

        # Trajectory
        racing_line_cfg = _dataclass_from_overrides(
            RacingLineConfig, race_config.racing_line_overrides
        )
        planner_cfg = _dataclass_from_overrides(
            PlannerConfig, race_config.planner_overrides
        )

        # iter-007 (3-way BLOCKER fix to iter-005b's dead code): RaceConfig
        # now actually has the max_velocity_mps field, so this getattr does
        # what iter-005b claimed. Fallback chain matches the synthetic bench:
        #   1. explicit `max_velocity_mps` in track JSON
        #   2. legacy `planner_overrides.plan_max_speed_mps`
        #   3. auto-derive from gate geometry
        from planning.auto_velocity import derive_safe_max_velocity
        explicit_max_v = race_config.max_velocity_mps
        if explicit_max_v is not None:
            pybullet_max_v = float(explicit_max_v)
        elif race_config.planner_overrides.get("plan_max_speed_mps") is not None:
            pybullet_max_v = float(planner_cfg.plan_max_speed_mps)
        else:
            pybullet_max_v = derive_safe_max_velocity(gate_specs)

        racing_started = time.perf_counter()
        rl_opt = RacingLineOptimizer(config=racing_line_cfg)
        opt_wps = rl_opt.optimize(gate_waypoints, start_pos)
        racing_elapsed = time.perf_counter() - racing_started
        phases["cache_lookup"] += rl_opt.last_cache_lookup_s
        phases["racing_line"] = max(
            0.0, racing_elapsed - rl_opt.last_cache_lookup_s
        )
        drone_constraints = _dataclass_from_overrides(
            DroneConstraints, {"max_velocity": pybullet_max_v}
        )
        trajectory_started = time.perf_counter()
        traj_opt = TrajectoryOptimizer(
            constraints=drone_constraints,
            dt_sample=0.02,
            planner_config=planner_cfg,
        )
        trajectory = traj_opt.optimize(opt_wps, start_pos, (0, 0, 0))
        phases["trajectory"] = time.perf_counter() - trajectory_started

        result["trajectory_time_s"] = trajectory.total_time
        result["trajectory_points"] = len(trajectory.points)

        validation_started = time.perf_counter()
        validation = _validation_payload(
            validate_trajectory(
                trajectory,
                gate_specs,
                dt=0.01,
                sequencer_config=seq_cfg,
            )
        )
        phases["plan_validation"] = time.perf_counter() - validation_started
        result["plan_validation"] = validation

        gpd_config = GPDDroneConfig()
        velocity_source = (
            "track.max_velocity_mps"
            if explicit_max_v is not None
            else (
                "planner.plan_max_speed_mps"
                if race_config.planner_overrides.get("plan_max_speed_mps") is not None
                else "derived_from_gate_geometry"
            )
        )
        resolved_configuration = {
            "raw_track": raw_config,
            "race_config": dataclasses.asdict(race_config),
            "duration": duration,
            "thresholds": threshold_values,
            "gate_specs": [dataclasses.asdict(gate) for gate in gate_specs],
            "sequencer": dataclasses.asdict(seq_cfg),
            "ekf": dataclasses.asdict(ekf_cfg),
            "racing_line": {
                key: value
                for key, value in dataclasses.asdict(racing_line_cfg).items()
                if key != "cache_root"
            },
            "planner": dataclasses.asdict(planner_cfg),
            "drone_constraints": dataclasses.asdict(drone_constraints),
            "derived_max_velocity": {
                "value_mps": float(pybullet_max_v),
                "source": velocity_source,
            },
            "trajectory_sample_dt": 0.02,
            "plan_validation_dt": 0.01,
            "progress_clock": {
                "source": "observed_env_sim_time_delta",
                "max_lag_m": 1.5,
            },
            "physics_backend": {
                "drone_config": dataclasses.asdict(gpd_config),
                "drone_model": "CF2X",
                "physics": "PYB",
                "gui": False,
                "record": False,
                "control_dt_s": 1.0 / gpd_config.ctrl_freq,
                "physics_dt_s": 1.0 / gpd_config.pyb_freq,
            },
        }
        result["resolved_configuration"] = resolved_configuration
        result["config_hash"] = sha256_json(resolved_configuration)

        if validation.get("ok") is not True:
            result.update(
                {
                    "termination_reason": "invalid_plan",
                    "gates_passed": 0,
                    "total_gates": len(gate_specs),
                    "gate_pass_rate": 0.0,
                    "complete": False,
                    "threshold_failures": [
                        f"plan_validation failed ({validation.get('reason', 'unknown')})"
                    ],
                    "sim_passed": False,
                }
            )
            return _finish()

        try:
            env = DroneRaceEnv(
                race_config=race_config,
                drone_config=gpd_config,
                gui=False,
            )
        except Exception as e:
            result["skipped"] = True
            result["skip_reason"] = f"Cannot create sim: {e}"
            return _finish()
        result["available"] = True

        # Run loop
        tracking_errors = []
        nearest_path_errors = []
        loop_times = []
        gate_pass_times = []
        per_gate_errors = {}
        wall_start = time.perf_counter()
        crashed = False
        termination_reason = "time_limit"

        # Progress clock: advances only when drone is close to its current
        # reference. Replaces wall-clock sampling so a stalled / bumped drone
        # doesn't have its plan fly away from it.
        progress_t = 0.0
        progress_max_lag_m = 1.5  # hold reference if drone is more than this far away
        previous_sim_time: Optional[float] = None
        final_sim_time = 0.0

        while True:
            t0 = time.perf_counter()
            sim_time = env.get_sim_time()
            sim_delta = _observed_sim_delta(previous_sim_time, sim_time)
            final_sim_time = float(sim_time)

            if sim_time > duration:
                break

            sd = env.drone.get_state()
            pos, vel, yaw = sd["position"], sd["velocity"], sd["yaw"]

            # EKF
            gyro = sd.get("angular_velocity", (0, 0, 0))
            ekf.predict((0, 0, -9.81), gyro, sim_time)
            ekf.update_odometry(pos, vel)

            # Sequencing
            passed = seq.update(pos)
            if passed:
                gate_pass_times.append({"gate_id": passed.gate_id, "time_s": sim_time})

            # Iter-003 M6 mirror: enforce VQ1 8-minute cap on the PyBullet
            # bench too. The synthetic bench already has this check; mirror
            # it here so both platforms honour the competition rule.
            if sim_time > AIGP_VQ1_MAX_RUN_DURATION_S:
                seq.mark_timed_out(
                    f"vq1_max_run_duration_exceeded:{sim_time:.1f}s"
                )
                termination_reason = f"timed_out:{seq.timeout_reason}"
                break

            # iter-001 A7 + iter-002 (composer-25 F6/F7): DQ is terminal on
            # the PyBullet path too — but it's NOT a crash. Frame-strut
            # crashes still flow primarily through `env.gate_contact()`
            # (the contact manifold is authoritative); the sequencer's
            # geometric crash classification is a secondary signal.
            if seq.is_disqualified:
                # Not setting crashed=True — surfaced via the result dict's
                # disqualified field.
                termination_reason = f"disqualified:{seq.dq_reason}"
                break

            if pos[2] < 0.05:
                crashed = True
                termination_reason = "crash_ground"
                break

            # Iter-008 F12 (Opus, platform-drift MINOR): synthetic bench has
            # a ceiling check; PyBullet now matches. Without this the same
            # trajectory would terminate at z>20 in the synthetic bench but
            # silently fly out of the airspace in the PyBullet bench.
            if pos[2] > 20.0:
                crashed = True
                termination_reason = "crash_ceiling"
                break

            # Gate-contact crash detection: any contact point against an
            # un-passed gate counts as a crash. Passing through the gate
            # opening triggers the sequencer first (handled above), so a
            # contact remaining here means we've hit a frame strut.
            hit_gate = env.gate_contact()
            if hit_gate is not None:
                seq.mark_collision(hit_gate)
                crashed = True
                termination_reason = f"crash_gate:{hit_gate}"
                break

            # If env didn't report a contact but the geometric sequencer's
            # P1-6 branch flagged a frame strike (e.g. sub-frame proximity not
            # quite touching), trust it.
            if seq.last_crash is not None:
                crashed = True
                termination_reason = f"crash_gate:{seq.last_crash[0]}"
                break

            # Safety and rules terminals above win when completion occurs on the
            # same update; only a clean final-gate step is race_complete.
            if seq.is_complete:
                termination_reason = "race_complete"
                break

            # Progress-clock advance: only if drone is keeping up with the
            # reference. If we're lagging, hold the reference and let the
            # tracker pull us back to it.
            ref_now = trajectory.sample(progress_t)
            lag = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, ref_now.position)))
            if lag < progress_max_lag_m and progress_t < trajectory.total_time:
                progress_t = min(progress_t + sim_delta, trajectory.total_time)

            # Trajectory tracking (sampled by progress clock, not wall clock)
            ref = trajectory.sample(progress_t)
            target_pos = ref.position
            target_vel = ref.velocity
            target_yaw = ref.yaw

            # Gate-seeking fallback (always-armed): if we've drifted off the
            # plan AND there's still an un-passed gate, seek straight at it.
            gate = seq.current_gate
            if gate is not None and lag >= progress_max_lag_m:
                gp = np.array(gate.position)
                dp = np.array(pos)
                d = gp - dp
                dist = float(np.linalg.norm(d))
                if dist > 0.1:
                    target_pos = tuple(gp)
                    target_vel = tuple(d / dist * min(dist * 2, 5.0))
                    target_yaw = float(math.atan2(d[1], d[0]))

            env.drone.step(target_pos, target_vel, target_yaw)

            post_step_position = env.drone.get_state()["position"]
            err = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(post_step_position, target_pos))
            )
            closest = trajectory.find_closest(post_step_position)
            nearest_error = math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(post_step_position, closest.position)
                )
            )
            tracking_errors.append(err)
            nearest_path_errors.append(nearest_error)
            loop_times.append(time.perf_counter() - t0)
            previous_sim_time = float(sim_time)

            # Per-gate tracking error
            cur = seq.current_gate
            if cur:
                gid = cur.gate_id
                per_gate_errors.setdefault(gid, []).append(err)

        wall_elapsed = time.perf_counter() - wall_start
        phases["rollout"] = wall_elapsed
        metrics_started = time.perf_counter()
        env.close()
        env_closed = True

        avg_err = float(np.mean(tracking_errors)) if tracking_errors else 0
        max_err = float(np.max(tracking_errors)) if tracking_errors else 0
        p50_err = float(np.percentile(tracking_errors, 50)) if tracking_errors else 0
        p95_err = float(np.percentile(tracking_errors, 95)) if tracking_errors else 0
        avg_nearest_err = (
            float(np.mean(nearest_path_errors)) if nearest_path_errors else 0.0
        )
        max_nearest_err = (
            float(np.max(nearest_path_errors)) if nearest_path_errors else 0.0
        )
        avg_hz = 1.0 / np.mean(loop_times) if loop_times else 0

        result.update({
            "sim_time_s": final_sim_time,
            "wall_time_s": wall_elapsed,
            "termination_reason": termination_reason,
            "crashed": crashed,
            # iter-001 A7: same honesty surface as the synthetic bench.
            "disqualified": bool(seq.is_disqualified),
            "dq_reason": seq.dq_reason,
            "last_crash_gate": seq.last_crash[0] if seq.last_crash else None,
            "gates_passed": seq.gates_passed,
            "total_gates": seq.total_gates,
            "gate_pass_rate": seq.gates_passed / seq.total_gates if seq.total_gates > 0 else 0,
            "complete": seq.is_complete,
            "gate_pass_times": gate_pass_times,
            "avg_tracking_error_m": avg_err,
            "max_tracking_error_m": max_err,
            "p50_tracking_error_m": p50_err,
            "p95_tracking_error_m": p95_err,
            "avg_nearest_path_error_m": avg_nearest_err,
            "max_nearest_path_error_m": max_nearest_err,
            "ekf_uncertainty_m": float(ekf.position_uncertainty),
            "avg_loop_hz": float(avg_hz),
            "total_steps": len(loop_times),
            "per_gate_avg_error": {
                gid: float(np.mean(errs)) for gid, errs in per_gate_errors.items()
            },
        })

        failures = _pybullet_threshold_failures(result, threshold_values)
        result["threshold_failures"] = failures
        result["sim_passed"] = (
            len(failures) == 0
            and result.get("available") is True
            and crashed is False
            and seq.is_disqualified is False
            and validation.get("ok") is True
            and seq.is_complete is True
        )
        phases["metrics"] = time.perf_counter() - metrics_started

        return _finish()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if env is not None and not env_closed:
            try:
                env.close()
            except Exception:
                if primary_error is None:
                    raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _failed_benchmark_envelope(
    error: BaseException,
    *,
    resolved_configuration: Mapping[str, Any],
    evaluator_version: str,
    comparison_series: str,
    seed: Optional[int],
    total_started: float,
) -> Dict[str, Any]:
    """Return the same auditable surface when an evaluator raises."""

    from planning.artifact_cache import dependency_fingerprint, sha256_json

    failure = f"{type(error).__name__}: {error}"
    phases = {
        "startup": 0.0,
        "config_load": 0.0,
        "cache_lookup": 0.0,
        "racing_line": 0.0,
        "trajectory": 0.0,
        "plan_validation": 0.0,
        "ilc": 0.0,
        "rollout": 0.0,
        "metrics": 0.0,
        "total_wall": 0.0,
    }
    result: Dict[str, Any] = {
        "available": False,
        "skipped": False,
        "sim_passed": False,
        "threshold_failures": [f"benchmark failed: {failure}"],
        "failure_summary": {
            "stdout_tail": "",
            "stderr_tail": "",
            "exception": failure,
            "threshold_failures": [f"benchmark failed: {failure}"],
        },
        "safety_passed": False,
        "validity_passed": False,
        "completion": {"complete": False, "gates_passed": 0, "total_gates": 0},
        "evaluator_version": evaluator_version,
        "schema_version": BENCHMARK_RESULT_SCHEMA,
        "comparison_series": comparison_series,
        "resolved_configuration": dict(resolved_configuration),
        "config_hash": sha256_json(resolved_configuration),
        "dependency_fingerprint": (
            _physics_dependency_fingerprint()
            if evaluator_version == PYBULLET_EVALUATOR_VERSION
            else dependency_fingerprint()
        ),
        "seed": seed,
        "thresholds": dict(
            resolved_configuration.get("thresholds", _threshold_snapshot())
        ),
        "cache_hit_or_miss": "not_applicable",
        "cache": {"status": "not_applicable", "reason": "evaluator raised"},
        "code_provenance": _git_provenance(),
        "phase_timings_s": phases,
        "timing_scope": FUNCTION_TIMING_SCOPE,
        "phase_timing_notes": dict(PHASE_TIMING_NOTES),
    }
    result["wall_time_s"] = _finalize_phase_timings(phases, total_started)
    result["timing_consistency"] = {
        "mutually_exclusive_phase_sum_s": sum(
            float(value)
            for name, value in phases.items()
            if name != "total_wall"
        ),
        "total_covers_phases": True,
    }
    return result


def main():
    main_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="AI Grand Prix — Headless Benchmark")
    parser.add_argument("--mode", choices=["unit", "synthetic", "sim", "full"], default="full")
    parser.add_argument("--config", type=str,
                        default=str(_REPO / "sim_pybullet" / "configs" / "race_01.json"))
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Sim-time seconds (default 30, matches max_total_time_s threshold)")
    parser.add_argument("--json-only", action="store_true",
                        help="Only output JSON to stdout, suppress stderr summary")
    parser.add_argument("--strict", action="store_true",
                        help="Deprecated compatibility flag; sim/full skips are always failures")
    parser.add_argument("--completion-threshold", type=float, default=None,
                        help="Override min_gate_pass_rate (0.0-1.0)")
    args = parser.parse_args()

    try:
        if args.mode in ("sim", "full"):
            args.duration = _positive_sim_duration(args.duration)
        else:
            args.duration = _exact_finite_float(
                "duration", args.duration, nonnegative=True
            )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    threshold_overrides = (
        {"min_gate_pass_rate": args.completion_threshold}
        if args.completion_threshold is not None
        else None
    )
    try:
        threshold_values = _threshold_snapshot(threshold_overrides)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    from planning.artifact_cache import dependency_fingerprint, sha256_json

    selected_path = Path(args.config).resolve()
    try:
        selected_config = _read_strict_json_object(
            selected_path, context="selected track config"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"cannot load selected track config: {error}")
    resolved_selected_config = {
        "selected_config_path": str(selected_path),
        "track": selected_config,
        "duration": args.duration,
        "thresholds": threshold_values,
    }
    evaluators: Dict[str, str] = {}
    if args.mode in ("unit", "full"):
        evaluators["unit"] = "embedded-unit-v1"
    if args.mode in ("synthetic", "full"):
        evaluators["synthetic"] = EVALUATOR_VERSION
    if args.mode in ("sim", "full"):
        evaluators["pybullet"] = PYBULLET_EVALUATOR_VERSION

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "thresholds": threshold_values,
        "schema_version": BENCHMARK_RESULT_SCHEMA,
        "evaluator_version": (
            next(iter(evaluators.values()))
            if len(evaluators) == 1
            else "multiple; see evaluators"
        ),
        "evaluators": evaluators,
        "comparison_series": (
            COMPARISON_SERIES
            if args.mode == "synthetic"
            else "multiple-or-not-applicable; see evaluator results"
        ),
        "code_provenance": _git_provenance(),
        "resolved_configuration": resolved_selected_config,
        "config_hash": sha256_json(resolved_selected_config),
        "dependency_fingerprint": dependency_fingerprint(),
        "seed": 42 if args.mode in ("synthetic", "full") else None,
        "seed_status": (
            "fixed synthetic seed"
            if args.mode in ("synthetic", "full")
            else "not_applicable"
        ),
        "cache_hit_or_miss": "not_applicable",
        "cache": {"status": "not_applicable"},
        "safety_passed": None,
        "validity_passed": None,
        "completion": {
            "status": "not_applicable",
            "complete": None,
            "gates_passed": None,
            "total_gates": None,
        },
        "failure_summary": {
            "stdout_tail": "",
            "stderr_tail": "",
            "exception": None,
            "threshold_failures": [],
            "result_references": [],
        },
        "timing_scope": (
            "main() dispatch through report assembly; excludes interpreter/module "
            "imports before main and final JSON serialization/output"
        ),
    }

    overall_pass = True

    # Unit tests
    if args.mode in ("unit", "full"):
        unit = run_unit_tests()
        report["unit_tests"] = unit
        if unit["pass_rate"] < threshold_values["unit_tests_pass_rate"]:
            overall_pass = False

    # Synthetic simulation (always available)
    if args.mode in ("synthetic", "full"):
        synthetic_started = time.perf_counter()
        try:
            synth = run_synthetic_benchmark(
                duration=args.duration,
                config=selected_config,
                thresholds=threshold_values,
            )
        except Exception as error:
            synth = _failed_benchmark_envelope(
                error,
                resolved_configuration=resolved_selected_config,
                evaluator_version=EVALUATOR_VERSION,
                comparison_series=COMPARISON_SERIES,
                seed=42,
                total_started=synthetic_started,
            )
        report["synthetic_sim"] = synth
        if synth.get("skipped", False) or not synth.get("sim_passed", False):
            overall_pass = False

    # PyBullet simulation
    if args.mode in ("sim", "full"):
        sim_started = time.perf_counter()
        try:
            sim = run_sim_benchmark(
                args.config, args.duration, thresholds=threshold_values
            )
        except Exception as error:
            sim = _failed_benchmark_envelope(
                error,
                resolved_configuration=resolved_selected_config,
                evaluator_version=PYBULLET_EVALUATOR_VERSION,
                comparison_series="pybullet-v3",
                seed=None,
                total_started=sim_started,
            )
        report["simulation"] = sim
        if sim.get("skipped", False):
            overall_pass = False
            skip_failure = (
                "PyBullet skipped; full/sim mode requires an executed evaluator"
            )
            failures = sim.setdefault("threshold_failures", [])
            if skip_failure not in failures:
                failures.append(skip_failure)
            sim["sim_passed"] = False
            sim.setdefault("safety_passed", False)
            sim.setdefault("validity_passed", False)
            sim.setdefault(
                "completion",
                {"complete": False, "gates_passed": 0, "total_gates": 0},
            )
            summary = sim.setdefault(
                "failure_summary",
                {
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "exception": sim.get("skip_reason"),
                },
            )
            summary["threshold_failures"] = list(failures)
        elif not sim.get("skipped", False) and not sim.get("sim_passed", False):
            overall_pass = False

    flight_results = [
        report[key]
        for key in ("synthetic_sim", "simulation")
        if key in report
    ]
    top_failures: List[str] = []
    if "unit_tests" in report:
        for test_result in report["unit_tests"].get("tests", []):
            if not test_result.get("passed", False):
                top_failures.append(
                    f"unit test failed: {test_result.get('name', 'unknown')}"
                )
    for key in ("synthetic_sim", "simulation"):
        if key not in report:
            continue
        nested = report[key]
        report["failure_summary"]["result_references"].append(
            f"{key}.failure_summary"
        )
        top_failures.extend(nested.get("threshold_failures", []))
        nested_exception = nested.get("failure_summary", {}).get("exception")
        if report["failure_summary"]["exception"] is None and nested_exception:
            report["failure_summary"]["exception"] = nested_exception

    if flight_results:
        report["safety_passed"] = all(
            result.get("safety_passed", result.get("sim_passed", False)) is True
            for result in flight_results
        )
        report["validity_passed"] = all(
            result.get("validity_passed", result.get("sim_passed", False)) is True
            for result in flight_results
        )
        completions = [
            result.get(
                "completion",
                {
                    "complete": result.get("complete", result.get("sim_passed", False)),
                    "gates_passed": result.get("gates_passed", 0),
                    "total_gates": result.get("total_gates", 0),
                },
            )
            for result in flight_results
        ]
        report["completion"] = {
            "status": "evaluated",
            "complete": all(item.get("complete") is True for item in completions),
            "gates_passed": sum(int(item.get("gates_passed", 0)) for item in completions),
            "total_gates": sum(int(item.get("total_gates", 0)) for item in completions),
            "result_references": [
                f"{key}.completion"
                for key in ("synthetic_sim", "simulation")
                if key in report
            ],
        }
    if "synthetic_sim" in report:
        report["cache_hit_or_miss"] = report["synthetic_sim"].get(
            "cache_hit_or_miss", "not_applicable"
        )
        report["cache"] = {"result_reference": "synthetic_sim.cache"}

    report["failure_summary"]["threshold_failures"] = top_failures
    report["overall_passed"] = overall_pass
    dispatch_wall = time.perf_counter() - main_started
    process_startup = (
        max(0.0, main_started - _SCRIPT_MODULE_STARTED)
        if __name__ == "__main__"
        else 0.0
    )
    report["function_dispatch_wall_time_s"] = dispatch_wall
    report["total_wall_time_s"] = process_startup + dispatch_wall
    report["phase_timings_s"] = {
        "startup": process_startup,
        "benchmark_dispatch": dispatch_wall,
        "total_wall": report["total_wall_time_s"],
    }
    report["timing_consistency"] = {
        "mutually_exclusive_phase_sum_s": process_startup + dispatch_wall,
        "total_covers_phases": True,
    }

    # Output JSON to stdout
    print(json.dumps(report, indent=2, allow_nan=False))

    # Human-readable summary to stderr
    if not args.json_only:
        _print_summary(report, file=sys.stderr)

    return 0 if overall_pass else 1


def _print_summary(report: Dict[str, Any], file=sys.stderr):
    p = lambda *a, **kw: print(*a, **kw, file=file)
    p(f"\n{'='*60}")
    p("AI Grand Prix — Benchmark Summary")
    p(f"{'='*60}")

    if "unit_tests" in report:
        u = report["unit_tests"]
        p(f"\nUnit Tests: {u['passed']}/{u['total']} passed ({u['total_time_ms']:.0f}ms)")
        for t in u["tests"]:
            status = "PASS" if t["passed"] else "FAIL"
            line = f"  [{status}] {t['name']} ({t['time_ms']:.1f}ms)"
            if not t["passed"]:
                line += f" — {t.get('error', 'unknown')}"
            p(line)

    for key, label in [("synthetic_sim", "Synthetic Sim"), ("simulation", "PyBullet Sim")]:
        if key not in report:
            continue
        s = report[key]
        if s.get("skipped"):
            p(f"\n{label}: SKIPPED — {s.get('skip_reason', 'unknown')}")
        elif s.get("available"):
            p(f"\n{label}:")
            p(f"  Gates: {s['gates_passed']}/{s['total_gates']} ({s['gate_pass_rate']:.0%})")
            p(f"  Sim time: {s.get('sim_time_s', 0):.1f}s  Wall: {s.get('wall_time_s', 0):.1f}s")
            p(f"  Tracking: avg={s['avg_tracking_error_m']:.2f}m  "
              f"p95={s.get('p95_tracking_error_m', 0):.2f}m  max={s['max_tracking_error_m']:.2f}m")
            p(f"  EKF uncertainty: {s['ekf_uncertainty_m']:.3f}m")
            p(f"  Loop: {s['avg_loop_hz']:.0f} Hz ({s['total_steps']} steps)")
            p(f"  Termination: {s['termination_reason']}")
            if s.get("gate_pass_times"):
                for gpt in s["gate_pass_times"]:
                    p(f"    {gpt['gate_id']} at {gpt['time_s']:.2f}s")
            if s["threshold_failures"]:
                p(f"  Failures:")
                for f_ in s["threshold_failures"]:
                    p(f"    - {f_}")
            else:
                p(f"  All thresholds met!")

    status = "PASS" if report["overall_passed"] else "FAIL"
    p(f"\n{'='*60}")
    p(f"Overall: {status}")
    p(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
