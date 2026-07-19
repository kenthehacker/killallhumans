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
- Leakage-safe grouped holdout: complete sessions (or complete tracks for
  historical datasets) are assigned wholly to train or validation.
- Per-sample weight: equal total gradient per track + mild curvature
  boost capped at 2× (high-curvature samples matter more for the
  tightest corners). Long tracks no longer dominate via sample count.
- Feature standardisation: store feat_mean/feat_std in the npz so
  inference applies the same scaling.
- **Closed-loop early-stop**: prepare the matrix once, then evaluate every
  25 epochs (configurable). Completion mode uses a race-capable automatic
  horizon; a short fixed horizon must be explicitly labeled ``prefix``.
- Atomic checkpoints contain the current/best models, Adam buffers, RNG,
  epoch, history, grouped split, baseline evidence, and selection state.

## Output

`control/residual_weights.npz` — TrackerResidualMLP.from_npz-loadable
with W1/b1/W2/b2 plus feat_mean/feat_std.
`control/residual_weights_meta.json` — training summary, per-track
baseline-vs-trained tracking error, dataset stats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

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

_CHECKPOINT_VERSION = 1
_AUTO_COMPLETION_MIN_DURATION_S = 45.0
_AUTO_COMPLETION_GRACE_S = 15.0


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


def _grouped_split(
    group_id: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a leakage-safe split whose groups never cross the boundary.

    A residual trace is sampled at 100 Hz, so randomly assigning adjacent
    rows makes validation nearly a copy of training.  ``group_id`` is a
    complete flight/session when available and otherwise a complete track.
    """
    groups = np.unique(np.asarray(group_id))
    if groups.size < 2:
        raise RuntimeError(
            "grouped validation requires at least two complete sessions/tracks"
        )
    if not math.isfinite(val_frac) or not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be strictly between 0 and 1")
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    n_val_groups = max(1, min(groups.size - 1, int(round(groups.size * val_frac))))
    val_groups = shuffled[:n_val_groups]
    val_mask = np.isin(group_id, val_groups)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if train_idx.size == 0 or val_idx.size == 0:  # defensive, bounds above imply not
        raise RuntimeError("grouped split produced an empty train or validation set")
    return train_idx, val_idx


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace an NPZ without exposing a partial checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_save_model(model: TrackerResidualMLP, path: Path) -> None:
    """Write a runtime model through the model's canonical serializer atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    temp_path = Path(raw_temp)
    try:
        model.to_npz(temp_path)
        with temp_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _optimizer_payload(opts: Mapping[str, "_AdamState"]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, state in opts.items():
        payload[f"optimizer_{name}_m"] = state.m
        payload[f"optimizer_{name}_v"] = state.v
        payload[f"optimizer_{name}_t"] = np.asarray(state.t, dtype=np.int64)
        payload[f"optimizer_{name}_beta1"] = np.asarray(state.beta1, dtype=np.float64)
        payload[f"optimizer_{name}_beta2"] = np.asarray(state.beta2, dtype=np.float64)
        payload[f"optimizer_{name}_eps"] = np.asarray(state.eps, dtype=np.float64)
    return payload


def _checkpoint_exact_int(
    checkpoint: Mapping[str, np.ndarray],
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Decode one exact integer scalar from an untrusted resume archive."""

    try:
        value = np.asarray(checkpoint[name])
    except KeyError as exc:
        raise RuntimeError(f"checkpoint is missing exact integer {name}") from exc
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise RuntimeError(f"checkpoint field {name} must be an exact integer scalar")
    decoded = int(value.item())
    if minimum is not None and decoded < minimum:
        raise RuntimeError(f"checkpoint field {name} is below its valid range")
    if maximum is not None and decoded > maximum:
        raise RuntimeError(f"checkpoint field {name} is above its valid range")
    return decoded


def _restore_optimizer(
    checkpoint: Mapping[str, np.ndarray],
    parameters: Mapping[str, np.ndarray],
) -> dict[str, "_AdamState"]:
    restored: dict[str, _AdamState] = {}
    for name, parameter in parameters.items():
        float_scalar_names = (
            f"optimizer_{name}_beta1",
            f"optimizer_{name}_beta2",
            f"optimizer_{name}_eps",
        )
        if any(np.asarray(checkpoint[key]).shape != () for key in float_scalar_names):
            raise RuntimeError(f"checkpoint optimizer scalars are malformed for {name}")
        beta1 = float(checkpoint[f"optimizer_{name}_beta1"])
        beta2 = float(checkpoint[f"optimizer_{name}_beta2"])
        eps = float(checkpoint[f"optimizer_{name}_eps"])
        step = _checkpoint_exact_int(
            checkpoint, f"optimizer_{name}_t", minimum=0
        )
        if (
            not math.isfinite(beta1)
            or not math.isfinite(beta2)
            or not math.isfinite(eps)
            or not 0.0 <= beta1 < 1.0
            or not 0.0 <= beta2 < 1.0
            or eps <= 0.0
            or step < 0
        ):
            raise RuntimeError(f"checkpoint optimizer values are invalid for {name}")
        state = _AdamState(
            parameter.shape,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        state.m = np.asarray(checkpoint[f"optimizer_{name}_m"], dtype=np.float64).copy()
        state.v = np.asarray(checkpoint[f"optimizer_{name}_v"], dtype=np.float64).copy()
        state.t = step
        if (
            state.m.shape != parameter.shape
            or state.v.shape != parameter.shape
            or not np.all(np.isfinite(state.m))
            or not np.all(np.isfinite(state.v))
        ):
            raise RuntimeError(f"checkpoint optimizer shape mismatch for {name}")
        restored[name] = state
    return restored


def _load_training_checkpoint(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"invalid training checkpoint {path}: {exc}") from exc
    try:
        checkpoint_version = _checkpoint_exact_int(
            payload, "checkpoint_version", minimum=0
        )
    except RuntimeError as exc:
        raise RuntimeError(f"invalid training checkpoint {path}: {exc}") from exc
    if checkpoint_version != _CHECKPOINT_VERSION:
        raise RuntimeError(
            f"unsupported training checkpoint version in {path}; "
            f"expected {_CHECKPOINT_VERSION}"
        )
    return payload


def _resolve_evaluation_duration(
    prepared: Any,
    requested_duration: float | None,
    mode: str,
) -> float:
    if mode not in {"completion", "prefix"}:
        raise ValueError("closed_loop_mode must be 'completion' or 'prefix'")
    if requested_duration is not None:
        duration = float(requested_duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("closed_loop_duration must be finite and positive")
        return duration
    if mode == "prefix":
        raise ValueError(
            "prefix scoring requires an explicit --closed-loop-duration"
        )
    trajectory_time = float(prepared.trajectory.total_time)
    return max(
        _AUTO_COMPLETION_MIN_DURATION_S,
        trajectory_time + _AUTO_COMPLETION_GRACE_S,
    )


def _prepare_matrix(
    *, cache_root: Path | str | None = None,
) -> dict[str, Any]:
    """Prepare each deterministic course once for all training evaluations."""
    from scripts.benchmark import prepare_course
    from scripts.benchmark_matrix import _list_configs

    prepared: dict[str, Any] = {}
    for config_path in _list_configs():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prepared[config_path.stem] = prepare_course(
            config, cache_root=cache_root
        )
    return prepared


def _closed_loop_evaluator_identity() -> dict[str, str]:
    """Bind resumable baselines to every source used by rollout semantics."""

    from planning.artifact_cache import source_digest
    from scripts.benchmark import (
        EVALUATOR_VERSION,
        _benchmark_result_source_files,
    )

    sources = _benchmark_result_source_files({"use_residual": False})
    sources.append(_REPO_ROOT / "control" / "learned_residual.py")
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "source_digest": source_digest(sources),
    }


def _completion_contract_valid(result: Mapping[str, Any]) -> bool:
    """Validate exact, mutually coherent rollout completion evidence."""

    bool_fields = (
        "sim_passed",
        "safety_passed",
        "validity_passed",
        "complete",
        "crashed",
        "disqualified",
        "skipped",
    )
    if any(type(result.get(name)) is not bool for name in bool_fields):
        return False

    gates_passed = result.get("gates_passed")
    total_gates = result.get("total_gates")
    if (
        type(gates_passed) is not int
        or type(total_gates) is not int
        or total_gates <= 0
        or not 0 <= gates_passed <= total_gates
    ):
        return False

    completion = result.get("completion")
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
        or completion["complete"] is not result["complete"]
        or completion["gates_passed"] != gates_passed
        or completion["total_gates"] != total_gates
        or result["complete"] is not (gates_passed == total_gates)
    ):
        return False

    termination_reason = result.get("termination_reason")
    if type(termination_reason) is not str or not termination_reason:
        return False
    if result["complete"] is not (termination_reason == "race_complete"):
        return False

    expected_safety = not result["crashed"] and not result["disqualified"]
    if result["safety_passed"] is not expected_safety:
        return False
    if result["sim_passed"] and not (
        result["safety_passed"]
        and result["validity_passed"]
        and result["complete"]
        and not result["crashed"]
        and not result["disqualified"]
        and not result["skipped"]
    ):
        return False
    return True


def _exact_rollout_summary(
    result: Mapping[str, Any], *, mode: str, duration_s: float
) -> dict[str, Any]:
    """Preserve exact evidence types; malformed evaluator output fails closed."""

    bool_fields = (
        "sim_passed",
        "safety_passed",
        "validity_passed",
        "complete",
        "crashed",
        "disqualified",
        "skipped",
    )
    evidence_valid = all(type(result.get(name)) is bool for name in bool_fields)
    gates_passed = result.get("gates_passed")
    total_gates = result.get("total_gates")
    if (
        type(gates_passed) is not int
        or type(total_gates) is not int
        or total_gates <= 0
        or not 0 <= gates_passed <= total_gates
    ):
        evidence_valid = False
        gates_passed = -1
        total_gates = -1
    error = result.get("avg_tracking_error_m")
    if (
        type(error) not in {int, float}
        or not math.isfinite(error)
        or error < 0.0
    ):
        evidence_valid = False
        error = None
    termination_reason = result.get("termination_reason")
    if type(termination_reason) is not str or not termination_reason:
        evidence_valid = False
        termination_reason = None
    if not _completion_contract_valid(result):
        evidence_valid = False
    completion = result.get("completion")
    if isinstance(completion, dict):
        completion_summary = {
            "complete": completion.get("complete"),
            "gates_passed": completion.get("gates_passed"),
            "total_gates": completion.get("total_gates"),
        }
    else:
        completion_summary = None
    return {
        "evaluation_mode": mode,
        "duration_s": duration_s,
        "avg_tracking_error_m": error,
        "sim_passed": result.get("sim_passed") is True,
        "safety_passed": result.get("safety_passed") is True,
        "validity_passed": result.get("validity_passed") is True,
        "complete": result.get("complete") is True,
        "crashed": result.get("crashed") if type(result.get("crashed")) is bool else None,
        "disqualified": (
            result.get("disqualified")
            if type(result.get("disqualified")) is bool
            else None
        ),
        "gates_passed": gates_passed,
        "total_gates": total_gates,
        "completion": completion_summary,
        "termination_reason": termination_reason,
        # Missing/coerced skip evidence is unsafe, so represent it as skipped.
        "skipped": result.get("skipped") is not False,
        "evidence_valid": evidence_valid,
    }


def _evaluate_closed_loop(
    weights_path: Path,
    prepared_courses: Mapping[str, Any],
    duration: float | None,
    mode: str,
    baseline_results: dict | None = None,
    *,
    seed: int = 42,
) -> dict:
    """Evaluate one model without rebuilding deterministic planning.

    ``completion`` is the promotion/acceptance mode and requires a completed
    race. ``prefix`` is an explicitly diagnostic fixed-horizon comparison;
    its result is never represented as completion evidence.
    """
    from scripts.benchmark import simulate

    out: dict = {}
    for name, prepared in prepared_courses.items():
        track_duration = _resolve_evaluation_duration(prepared, duration, mode)
        overrides = {
            "use_residual": True,
            "residual_weights_path": str(weights_path),
        }
        result = simulate(
            prepared,
            controller_config=overrides,
            seed=seed,
            duration=track_duration,
            thresholds={"max_total_time_s": track_duration},
        )
        out[name] = _exact_rollout_summary(
            result, mode=mode, duration_s=track_duration
        )
    if baseline_results is not None:
        for name, base in baseline_results.items():
            if name not in out or out[name]["skipped"]:
                continue
            base_err = base.get("avg_tracking_error_m")
            on_err = out[name]["avg_tracking_error_m"]
            if base_err is None or on_err is None:
                continue
            out[name]["baseline_avg_tracking_error_m"] = base_err
            baseline_gates = base.get("gates_passed")
            if type(baseline_gates) is not int or baseline_gates < 0:
                out[name]["evidence_valid"] = False
                continue
            out[name]["baseline_gates_passed"] = baseline_gates
            out[name]["improvement_pct"] = (
                100.0 * (base_err - on_err) / max(base_err, 1e-9)
            )
    return out


def _matrix_baseline(
    prepared_courses: Mapping[str, Any],
    duration: float | None,
    mode: str,
    *,
    seed: int = 42,
) -> dict:
    """Evaluate the baseline once against already-prepared courses."""
    from scripts.benchmark import simulate

    out: dict[str, dict[str, Any]] = {}
    for name, prepared in prepared_courses.items():
        track_duration = _resolve_evaluation_duration(prepared, duration, mode)
        result = simulate(
            prepared,
            controller_config={"use_residual": False},
            seed=seed,
            duration=track_duration,
            thresholds={"max_total_time_s": track_duration},
        )
        out[name] = _exact_rollout_summary(
            result, mode=mode, duration_s=track_duration
        )
    return out


def _score_closed_loop(
    results: Mapping[str, Mapping[str, Any]],
    mode: str,
    *,
    expected_tracks: set[str] | None = None,
) -> dict:
    """Apply hard gates before the tracking-improvement objective."""
    improved = 0
    hard_failures: list[str] = []
    regressions: list[str] = []
    if expected_tracks is not None:
        for name in sorted(expected_tracks - set(results)):
            hard_failures.append(f"{name}:missing_result")
        for name in sorted(set(results) - expected_tracks):
            hard_failures.append(f"{name}:unexpected_result")
    for name, result in results.items():
        if (
            result.get("evidence_valid") is not True
            or not _completion_contract_valid(result)
        ):
            hard_failures.append(f"{name}:malformed_evidence")
            continue
        if result.get("skipped") is not False:
            hard_failures.append(f"{name}:skipped")
            continue
        if result.get("safety_passed") is not True:
            hard_failures.append(f"{name}:safety")
        if result.get("validity_passed") is not True:
            hard_failures.append(f"{name}:validity")
        if result.get("crashed") is not False:
            hard_failures.append(f"{name}:crash")
        if result.get("disqualified") is not False:
            hard_failures.append(f"{name}:disqualification")
        if mode == "completion":
            if (
                result.get("complete") is not True
                or result.get("sim_passed") is not True
                or result.get("termination_reason") != "race_complete"
            ):
                hard_failures.append(f"{name}:incomplete_or_failed")
        else:
            if result.get("termination_reason") not in {"time_limit", "race_complete"}:
                hard_failures.append(f"{name}:invalid_prefix_termination")
            baseline_gates = result.get("baseline_gates_passed")
            if type(baseline_gates) is not int or baseline_gates < 0:
                hard_failures.append(f"{name}:invalid_baseline_progress")
            elif result["gates_passed"] < baseline_gates:
                hard_failures.append(f"{name}:prefix_progress_regressed")
        improvement = result.get("improvement_pct")
        if name != "figure8" and (
            type(improvement) not in {int, float}
            or not math.isfinite(improvement)
        ):
            hard_failures.append(f"{name}:missing_tracking_comparison")
        elif improvement is not None and name != "figure8":
            if improvement > 1.0:
                improved += 1
            if improvement < -1.0:
                regressions.append(name)
    hard_failures.extend(f"{name}:tracking_regression" for name in regressions)
    return {
        "score": -1e6 if hard_failures else float(improved),
        "improved_count": improved,
        "hard_failures": hard_failures,
        "regressed_over_1pct": regressions,
    }


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
    closed_loop_duration: float | None = None,
    closed_loop_patience: int = 3,
    closed_loop_mode: str = "completion",
    skip_closed_loop: bool = False,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    restart: bool = False,
    cache_root: Path | str | None = None,
    evaluation_seed: int = 42,
) -> dict:
    """Train with grouped validation and an atomic, resumable state file.

    The deterministic courses are prepared once and shared by the baseline
    and all candidate evaluations.  A checkpoint is written after every SGD
    epoch and, critically, after every completed closed-loop matrix.
    """
    integer_options = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "closed_loop_every": closed_loop_every,
        "closed_loop_patience": closed_loop_patience,
        "evaluation_seed": evaluation_seed,
    }
    for name, value in integer_options.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact integer")
    if seed < 0 or evaluation_seed < 0:
        raise ValueError("seed and evaluation_seed must be non-negative")
    boolean_options = {
        "skip_closed_loop": skip_closed_loop,
        "resume": resume,
        "restart": restart,
    }
    for name, value in boolean_options.items():
        if type(value) is not bool:
            raise TypeError(f"{name} must be an exact bool")
    for name, value in {"lr_max": lr_max, "lr_min": lr_min}.items():
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if lr_min > lr_max:
        raise ValueError("lr_min must not exceed lr_max")
    if (
        type(val_frac) not in {int, float}
        or not math.isfinite(val_frac)
        or not 0.0 < val_frac < 1.0
    ):
        raise ValueError("val_frac must be strictly between 0 and 1")
    if closed_loop_duration is not None and (
        type(closed_loop_duration) not in {int, float}
        or not math.isfinite(closed_loop_duration)
        or closed_loop_duration <= 0.0
    ):
        raise ValueError("closed_loop_duration must be finite and positive")
    dataset_path = Path(dataset_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else out_path.with_name(out_path.stem + "_checkpoint.npz")
    )
    if checkpoint_path.resolve() == out_path.resolve():
        raise ValueError("checkpoint_path and out_path must be different files")
    if resume and restart:
        raise ValueError("resume and restart are mutually exclusive")
    if checkpoint_path.exists() and not resume and not restart:
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint_path}; use --resume to "
            "continue it or --restart to explicitly begin a new experiment"
        )
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if closed_loop_every < 1:
        raise ValueError("closed_loop_every must be at least 1")
    if closed_loop_patience < 1:
        raise ValueError("closed_loop_patience must be at least 1")
    if closed_loop_mode not in {"completion", "prefix"}:
        raise ValueError("closed_loop_mode must be 'completion' or 'prefix'")
    if (
        not skip_closed_loop
        and closed_loop_mode == "prefix"
        and closed_loop_duration is None
    ):
        raise ValueError("prefix scoring requires an explicit closed_loop_duration")

    rng = np.random.default_rng(seed)
    data = load_feature_trace(dataset_path)
    features = data["features"]
    n = features.shape[0]
    if n < 100:
        raise RuntimeError(
            f"dataset too small ({n} samples); collect more before training"
        )
    targets = _compute_fel_targets_body_frame(
        pos_err=data["pos_err"], yaw_des=data["yaw_des"],
    )

    names_by_group: dict[int, str] = {}
    with np.load(dataset_path, allow_pickle=False) as raw:
        if "track_id" not in raw.files:
            raise RuntimeError(
                f"dataset {dataset_path} is missing the 'track_id' field. "
                "Regenerate via scripts/collect_residual_dataset.py."
            )
        track_id = raw["track_id"].copy()
        if "session_id" in raw.files:
            group_id = raw["session_id"].copy()
            group_kind = "session"
            names_key = "session_names"
        else:
            group_id = track_id.copy()
            group_kind = "track"
            names_key = "track_names"
        if names_key in raw.files:
            try:
                names = raw[names_key]
                if names.dtype.kind in {"U", "S"}:
                    names_by_group = {
                        index: str(value)
                        for index, value in enumerate(names.tolist())
                    }
            except ValueError:
                # Old files used object dtype. Never enable pickle solely for
                # labels; numeric whole-track grouping remains safe.
                names_by_group = {}
    if track_id.shape != (n,) or group_id.shape != (n,):
        raise RuntimeError("track_id/session_id length does not match feature rows")
    if not np.issubdtype(track_id.dtype, np.integer) or not np.issubdtype(
        group_id.dtype, np.integer
    ):
        raise RuntimeError("track_id/session_id must contain integer identifiers")

    prepared_courses = (
        {} if skip_closed_loop else _prepare_matrix(cache_root=cache_root)
    )
    prepared_artifacts = {
        name: prepared.artifact_key for name, prepared in prepared_courses.items()
    }
    training_signature = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "trainer_source_sha256": _sha256_file(Path(__file__)),
        "learned_residual_source_sha256": _sha256_file(
            _REPO_ROOT / "control" / "learned_residual.py"
        ),
        "drone_spec_source_sha256": _sha256_file(
            _REPO_ROOT / "competition" / "drone_spec.py"
        ),
        "python_version": sys.version,
        "python_cache_tag": sys.implementation.cache_tag,
        "numpy_version": np.__version__,
        "dataset_sha256": _sha256_file(dataset_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "val_frac": val_frac,
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "closed_loop_every": closed_loop_every,
        "closed_loop_duration": closed_loop_duration,
        "closed_loop_patience": closed_loop_patience,
        "closed_loop_mode": closed_loop_mode,
        "skip_closed_loop": skip_closed_loop,
        "group_kind": group_kind,
        "model_shape": [
            DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS,
        ],
        "prepared_artifacts": prepared_artifacts,
        "closed_loop_evaluator": (
            None if skip_closed_loop else _closed_loop_evaluator_identity()
        ),
        "evaluation_threshold_policy": (
            "max_total_time_s=resolved_evaluation_duration_s"
        ),
    }
    signature_json = _json_text(training_signature)
    output_clamp = np.array(
        [_CLAMP_RAD, _CLAMP_RAD, _CLAMP_THRUST], dtype=np.float64
    )

    resumed_from_checkpoint = False
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = _load_training_checkpoint(checkpoint_path)
        if str(checkpoint["training_signature_json"].item()) != signature_json:
            raise RuntimeError(
                "checkpoint configuration/dataset mismatch; start a new "
                "checkpoint rather than mutating a resumed experiment"
            )
        raw_train_idx = np.asarray(checkpoint["train_idx"])
        raw_val_idx = np.asarray(checkpoint["val_idx"])
        if not np.issubdtype(raw_train_idx.dtype, np.integer) or not np.issubdtype(
            raw_val_idx.dtype, np.integer
        ):
            raise RuntimeError("checkpoint grouped split indices must be exact integers")
        train_idx = np.asarray(raw_train_idx, dtype=np.int64)
        val_idx = np.asarray(raw_val_idx, dtype=np.int64)
        all_indices = np.concatenate([train_idx, val_idx])
        if (
            all_indices.size != n
            or np.any(all_indices < 0)
            or np.any(all_indices >= n)
            or np.unique(all_indices).size != n
            or np.intersect1d(group_id[train_idx], group_id[val_idx]).size
        ):
            raise RuntimeError(
                "checkpoint contains an invalid or leaking grouped split"
            )
        feat_mean = np.asarray(checkpoint["feat_mean"], dtype=np.float64)
        feat_std = np.asarray(checkpoint["feat_std"], dtype=np.float64)
        W1 = np.asarray(checkpoint["current_W1"], dtype=np.float64)
        b1 = np.asarray(checkpoint["current_b1"], dtype=np.float64)
        W2 = np.asarray(checkpoint["current_W2"], dtype=np.float64)
        b2 = np.asarray(checkpoint["current_b2"], dtype=np.float64)
        expected_shapes = {
            "current_W1": (DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN),
            "current_b1": (DEFAULT_N_HIDDEN,),
            "current_W2": (DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS),
            "current_b2": (DEFAULT_N_OUTPUTS,),
            "feat_mean": (DEFAULT_N_INPUTS,),
            "feat_std": (DEFAULT_N_INPUTS,),
            "output_clamp": (DEFAULT_N_OUTPUTS,),
        }
        checkpoint_arrays = {
            "current_W1": W1,
            "current_b1": b1,
            "current_W2": W2,
            "current_b2": b2,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "output_clamp": np.asarray(
                checkpoint["output_clamp"], dtype=np.float64
            ),
        }
        if any(
            value.shape != expected_shapes[name]
            or not np.all(np.isfinite(value))
            for name, value in checkpoint_arrays.items()
        ):
            raise RuntimeError("checkpoint model arrays are malformed or non-finite")
        if np.any(feat_std <= 0.0) or not np.array_equal(
            checkpoint_arrays["output_clamp"], output_clamp
        ):
            raise RuntimeError("checkpoint normalization or output clamp is invalid")
        opts = _restore_optimizer(
            checkpoint, {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        )
        rng.bit_generator.state = json.loads(
            str(checkpoint["rng_state_json"].item())
        )
        history = json.loads(str(checkpoint["history_json"].item()))
        best_val_loss = float(checkpoint["best_val_loss"])
        best_weights_by_val = None
        best_val_present = _checkpoint_exact_int(
            checkpoint, "best_val_present", minimum=0, maximum=1
        )
        if bool(best_val_present):
            best_weights_by_val = tuple(
                np.asarray(checkpoint[f"best_val_{name}"], dtype=np.float64)
                for name in ("W1", "b1", "W2", "b2")
            )
            if any(
                value.shape != expected_shapes[f"current_{name}"]
                or not np.all(np.isfinite(value))
                for name, value in zip(
                    ("W1", "b1", "W2", "b2"), best_weights_by_val
                )
            ) or not math.isfinite(best_val_loss):
                raise RuntimeError("checkpoint best-validation state is invalid")
        best_closed_loop_score = float(checkpoint["best_closed_loop_score"])
        best_weights_by_cl = None
        best_cl_present = _checkpoint_exact_int(
            checkpoint, "best_cl_present", minimum=0, maximum=1
        )
        if bool(best_cl_present):
            best_weights_by_cl = tuple(
                np.asarray(checkpoint[f"best_cl_{name}"], dtype=np.float64)
                for name in (
                    "W1", "b1", "W2", "b2", "feat_mean", "feat_std",
                )
            )
            best_cl_shapes = (
                expected_shapes["current_W1"],
                expected_shapes["current_b1"],
                expected_shapes["current_W2"],
                expected_shapes["current_b2"],
                expected_shapes["feat_mean"],
                expected_shapes["feat_std"],
            )
            if (
                any(
                    value.shape != shape or not np.all(np.isfinite(value))
                    for value, shape in zip(best_weights_by_cl, best_cl_shapes)
                )
                or np.any(best_weights_by_cl[-1] <= 0.0)
                or not math.isfinite(best_closed_loop_score)
            ):
                raise RuntimeError("checkpoint best-closed-loop state is invalid")
        best_cl_results = json.loads(
            str(checkpoint["best_cl_results_json"].item())
        )
        baseline_cl = json.loads(str(checkpoint["baseline_cl_json"].item()))
        cl_no_improve_count = _checkpoint_exact_int(
            checkpoint, "cl_no_improve_count", minimum=0
        )
        evaluation_count = _checkpoint_exact_int(
            checkpoint, "evaluation_count", minimum=0
        )
        epoch_completed = _checkpoint_exact_int(
            checkpoint, "epoch_completed", minimum=-1, maximum=epochs - 1
        )
        start_epoch = epoch_completed + 1
        training_complete = bool(
            _checkpoint_exact_int(
                checkpoint, "training_complete", minimum=0, maximum=1
            )
        )
        resumed_from_checkpoint = True
    else:
        train_idx, val_idx = _grouped_split(group_id, val_frac, rng)
        x_train_initial = features[train_idx]
        feat_mean = x_train_initial.mean(axis=0)
        feat_std = x_train_initial.std(axis=0)
        feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
        n_in, n_h, n_out = (
            DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS,
        )
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
        history: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        best_weights_by_val = None
        best_closed_loop_score = -float("inf")
        best_weights_by_cl = None
        best_cl_results: dict | None = None
        cl_no_improve_count = 0
        evaluation_count = 0
        start_epoch = 0
        training_complete = False
        baseline_cl = None

    x_train = features[train_idx]
    y_train = targets[train_idx]
    w_train = _compute_per_sample_weights(
        track_id[train_idx], data["ref_accel"][train_idx]
    )
    x_val = features[val_idx]
    y_val = targets[val_idx]
    w_val = _compute_per_sample_weights(
        track_id[val_idx], data["ref_accel"][val_idx]
    )
    computed_mean = x_train.mean(axis=0)
    raw_std = x_train.std(axis=0)
    computed_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    if not np.array_equal(feat_mean, computed_mean) or not np.array_equal(
        feat_std, computed_std
    ):
        raise RuntimeError(
            "checkpoint feature normalization does not match its grouped split"
        )
    x_train_n = (x_train - feat_mean) / feat_std
    x_val_n = (x_val - feat_mean) / feat_std

    def save_checkpoint(epoch_completed: int, *, complete: bool) -> None:
        payload: dict[str, Any] = {
            "checkpoint_version": np.asarray(
                _CHECKPOINT_VERSION, dtype=np.int64
            ),
            "training_signature_json": np.asarray(signature_json),
            "epoch_completed": np.asarray(epoch_completed, dtype=np.int64),
            "training_complete": np.asarray(int(complete), dtype=np.uint8),
            "history_json": np.asarray(_json_text(history)),
            "rng_state_json": np.asarray(
                _json_text(rng.bit_generator.state)
            ),
            "train_idx": np.asarray(train_idx, dtype=np.int64),
            "val_idx": np.asarray(val_idx, dtype=np.int64),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "output_clamp": output_clamp,
            "current_W1": W1,
            "current_b1": b1,
            "current_W2": W2,
            "current_b2": b2,
            "best_val_loss": np.asarray(best_val_loss, dtype=np.float64),
            "best_val_present": np.asarray(
                int(best_weights_by_val is not None), dtype=np.uint8
            ),
            "best_closed_loop_score": np.asarray(
                best_closed_loop_score, dtype=np.float64
            ),
            "best_cl_present": np.asarray(
                int(best_weights_by_cl is not None), dtype=np.uint8
            ),
            "best_cl_results_json": np.asarray(
                _json_text(best_cl_results)
            ),
            "baseline_cl_json": np.asarray(_json_text(baseline_cl)),
            "cl_no_improve_count": np.asarray(
                cl_no_improve_count, dtype=np.int64
            ),
            "evaluation_count": np.asarray(
                evaluation_count, dtype=np.int64
            ),
        }
        payload.update(_optimizer_payload(opts))
        if best_weights_by_val is not None:
            for name, value in zip(
                ("W1", "b1", "W2", "b2"), best_weights_by_val
            ):
                payload[f"best_val_{name}"] = value
        if best_weights_by_cl is not None:
            for name, value in zip(
                ("W1", "b1", "W2", "b2", "feat_mean", "feat_std"),
                best_weights_by_cl,
            ):
                payload[f"best_cl_{name}"] = value
        _atomic_save_npz(checkpoint_path, payload)

    if not skip_closed_loop and baseline_cl is None:
        baseline_cl = _matrix_baseline(
            prepared_courses,
            closed_loop_duration,
            closed_loop_mode,
            seed=evaluation_seed,
        )
        evaluation_count += 1
        # Persist the completed expensive matrix before enforcing its horizon.
        save_checkpoint(-1, complete=False)

    if not skip_closed_loop:
        baseline_failures: list[str] = []
        missing_baselines = set(prepared_courses) - set(baseline_cl or {})
        baseline_failures.extend(
            f"{name}:missing_result" for name in sorted(missing_baselines)
        )
        for name, result in (baseline_cl or {}).items():
            if (
                result.get("evidence_valid") is not True
                or not _completion_contract_valid(result)
                or result.get("skipped") is not False
                or result.get("safety_passed") is not True
                or result.get("validity_passed") is not True
                or result.get("crashed") is not False
                or result.get("disqualified") is not False
            ):
                baseline_failures.append(name)
            elif closed_loop_mode == "completion" and (
                result.get("complete") is not True
                or result.get("sim_passed") is not True
                or result.get("termination_reason") != "race_complete"
            ):
                baseline_failures.append(name)
            elif (
                closed_loop_mode == "prefix"
                and result.get("termination_reason")
                not in {"time_limit", "race_complete"}
            ):
                baseline_failures.append(name)
        if baseline_failures:
            raise RuntimeError(
                "closed-loop baseline failed the "
                f"{closed_loop_mode} contract on {baseline_failures}; "
                "increase the completion horizon or explicitly use prefix scoring"
            )
    elif not resume:
        save_checkpoint(-1, complete=False)

    epochs_to_run = range(start_epoch, epochs) if not training_complete else ()
    for epoch in epochs_to_run:
        lr = _cosine_lr(epoch, epochs, lr_max, lr_min)
        permutation = rng.permutation(len(x_train_n))
        x_shuffled = x_train_n[permutation]
        y_shuffled = y_train[permutation]
        w_shuffled = w_train[permutation]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(x_shuffled), batch_size):
            xb = x_shuffled[start:start + batch_size]
            yb = y_shuffled[start:start + batch_size]
            wb = w_shuffled[start:start + batch_size]
            y_hat, hidden, _ = _forward(
                xb, W1, b1, W2, b2, output_clamp
            )
            error = y_hat - yb
            wb_column = wb[:, None]
            weight_sum = float(np.sum(wb)) * y_hat.shape[1]
            loss = float(np.sum(wb_column * error ** 2)) / max(
                weight_sum, 1e-9
            )
            epoch_loss += loss
            n_batches += 1
            grad_y = 2 * wb_column * error / max(weight_sum, 1e-9)
            grad_raw = grad_y * (1 - (y_hat / output_clamp) ** 2)
            grad_W2 = hidden.T @ grad_raw
            grad_b2 = grad_raw.sum(0)
            grad_hidden = grad_raw @ W2.T
            grad_pre = grad_hidden * (1 - hidden ** 2)
            grad_W1 = xb.T @ grad_pre
            grad_b1 = grad_pre.sum(0)
            W1 = opts["W1"].step(W1, grad_W1, lr)
            b1 = opts["b1"].step(b1, grad_b1, lr)
            W2 = opts["W2"].step(W2, grad_W2, lr)
            b2 = opts["b2"].step(b2, grad_b2, lr)

        train_loss = epoch_loss / max(1, n_batches)
        y_val_hat, _, _ = _forward(
            x_val_n, W1, b1, W2, b2, output_clamp
        )
        val_error = y_val_hat - y_val
        val_loss = float(
            np.sum(w_val[:, None] * val_error ** 2)
            / max(np.sum(w_val) * y_val_hat.shape[1], 1e-9)
        )
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_by_val = (
                W1.copy(), b1.copy(), W2.copy(), b2.copy(),
            )

        if not skip_closed_loop and (
            (epoch + 1) % closed_loop_every == 0
            or epoch == epochs - 1
        ):
            fd, raw_eval_path = tempfile.mkstemp(
                prefix=f".{out_path.stem}.eval-",
                suffix=".npz",
                dir=str(out_path.parent),
            )
            os.close(fd)
            eval_path = Path(raw_eval_path)
            candidate = TrackerResidualMLP(
                W1=W1,
                b1=b1,
                W2=W2,
                b2=b2,
                feat_mean=feat_mean,
                feat_std=feat_std,
                output_clamp=output_clamp,
            )
            try:
                _atomic_save_model(candidate, eval_path)
                closed_loop = _evaluate_closed_loop(
                    eval_path,
                    prepared_courses,
                    closed_loop_duration,
                    closed_loop_mode,
                    baseline_cl,
                    seed=evaluation_seed,
                )
            finally:
                eval_path.unlink(missing_ok=True)
            evaluation_count += 1
            scored = _score_closed_loop(
                closed_loop,
                closed_loop_mode,
                expected_tracks=set(prepared_courses),
            )
            score = float(scored["score"])
            if not scored["hard_failures"] and score > best_closed_loop_score:
                best_closed_loop_score = score
                best_weights_by_cl = (
                    W1.copy(), b1.copy(), W2.copy(), b2.copy(),
                    feat_mean.copy(), feat_std.copy(),
                )
                best_cl_results = closed_loop
                cl_no_improve_count = 0
            else:
                cl_no_improve_count += 1
            history[-1].update(
                {
                    "closed_loop_score": score,
                    "closed_loop_mode": closed_loop_mode,
                    "improved_count": scored["improved_count"],
                    "hard_failures": scored["hard_failures"],
                    "regressed_over_1pct": scored["regressed_over_1pct"],
                }
            )
            if cl_no_improve_count >= closed_loop_patience:
                history[-1]["early_stopped"] = True
                training_complete = True
        if epoch == epochs - 1:
            training_complete = True
        # If an evaluation is interrupted, the prior checkpoint replays this
        # epoch with the exact optimizer and RNG state. A completed evaluation
        # is always persisted here before another epoch starts.
        save_checkpoint(epoch, complete=training_complete)
        if training_complete and history[-1].get("early_stopped"):
            break

    if not skip_closed_loop and best_weights_by_cl is None:
        raise RuntimeError(
            "no residual candidate passed every closed-loop safety, validity, "
            "completion, progress, and tracking-regression gate; the resumable "
            "checkpoint was preserved but runtime weights were not published"
        )
    if best_weights_by_cl is not None:
        W1, b1, W2, b2, feat_mean_best, feat_std_best = best_weights_by_cl
        select_method = "best_closed_loop"
    elif best_weights_by_val is not None:
        W1, b1, W2, b2 = best_weights_by_val
        feat_mean_best, feat_std_best = feat_mean, feat_std
        select_method = "best_val_loss_unvalidated_fallback"
    else:  # defensive for a malformed complete checkpoint
        feat_mean_best, feat_std_best = feat_mean, feat_std
        select_method = "final"

    model = TrackerResidualMLP(
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        feat_mean=feat_mean_best,
        feat_std=feat_std_best,
        output_clamp=output_clamp,
    )
    _atomic_save_model(model, out_path)

    train_prediction, _, _ = _forward(
        (x_train - feat_mean_best) / feat_std_best,
        W1, b1, W2, b2, output_clamp,
    )
    val_prediction, _, _ = _forward(
        (x_val - feat_mean_best) / feat_std_best,
        W1, b1, W2, b2, output_clamp,
    )
    train_groups = np.unique(group_id[train_idx])
    val_groups = np.unique(group_id[val_idx])
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
        "best_closed_loop_score": (
            float(best_closed_loop_score)
            if math.isfinite(best_closed_loop_score)
            else None
        ),
        "select_method": select_method,
        "best_closed_loop_results": best_cl_results,
        "closed_loop_hard_gates_passed": best_weights_by_cl is not None,
        "history": history,
        "dataset_sha256": training_signature["dataset_sha256"],
        "trainer_source_sha256": training_signature["trainer_source_sha256"],
        "closed_loop_evaluator": training_signature["closed_loop_evaluator"],
        "evaluation_threshold_policy": training_signature[
            "evaluation_threshold_policy"
        ],
        "environment": {
            "python_version": training_signature["python_version"],
            "python_cache_tag": training_signature["python_cache_tag"],
            "numpy_version": training_signature["numpy_version"],
        },
        "split": {
            "kind": group_kind,
            "train_group_ids": [int(value) for value in train_groups],
            "validation_group_ids": [int(value) for value in val_groups],
            "train_group_names": [
                names_by_group.get(int(value), str(int(value)))
                for value in train_groups
            ],
            "validation_group_names": [
                names_by_group.get(int(value), str(int(value)))
                for value in val_groups
            ],
        },
        "closed_loop": {
            "enabled": not skip_closed_loop,
            "mode": closed_loop_mode,
            "requested_duration_s": closed_loop_duration,
            "evaluation_seed": evaluation_seed,
            "evaluations_completed": evaluation_count,
            "prepared_artifacts": prepared_artifacts,
            "baseline_results": baseline_cl,
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_sha256": _sha256_file(out_path),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "feat_mean": [float(value) for value in feat_mean_best],
        "feat_std": [float(value) for value in feat_std_best],
        "train_pred_range": {
            "min": [
                float(train_prediction[:, index].min())
                for index in range(DEFAULT_N_OUTPUTS)
            ],
            "max": [
                float(train_prediction[:, index].max())
                for index in range(DEFAULT_N_OUTPUTS)
            ],
        },
        "val_pred_range": {
            "min": [
                float(val_prediction[:, index].min())
                for index in range(DEFAULT_N_OUTPUTS)
            ],
            "max": [
                float(val_prediction[:, index].max())
                for index in range(DEFAULT_N_OUTPUTS)
            ],
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
        "--closed-loop-duration", type=float, default=None,
        help=(
            "Per-track horizon. Completion mode defaults to max(45s, "
            "trajectory time + 15s); prefix mode requires an explicit value."
        ),
    )
    parser.add_argument(
        "--closed-loop-mode", choices=("completion", "prefix"),
        default="completion",
        help=(
            "Completion is promotion evidence; prefix is a separately labeled "
            "fixed-horizon diagnostic and cannot claim race completion."
        ),
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
        "--checkpoint", default=None,
        help="Atomic resumable checkpoint path (default: beside --out).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume exactly from --checkpoint; config/dataset drift fails closed.",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Explicitly replace an existing checkpoint with a new experiment.",
    )
    parser.add_argument(
        "--cache-root", default=None,
        help="Prepared-artifact cache root (default: AIGP_CACHE_ROOT/.cache).",
    )
    parser.add_argument("--evaluation-seed", type=int, default=42)
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
        closed_loop_mode=args.closed_loop_mode,
        skip_closed_loop=args.skip_closed_loop,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        resume=args.resume,
        restart=args.restart,
        cache_root=args.cache_root,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps(summary, indent=2))
    meta_out = args.meta_out
    if meta_out is None:
        meta_out = str(out_path.with_name(out_path.stem + "_meta.json"))
    _atomic_write_json(Path(meta_out), summary)


if __name__ == "__main__":
    main()
