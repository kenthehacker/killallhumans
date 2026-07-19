"""Collect a grouped, provenance-bearing tracker-residual dataset.

Each non-excluded synthetic track is prepared once through the benchmark's
content-addressed planning API and rolled out once with feature tracing.  The
default corpus accepts only safe, valid, completed sessions.  A deliberately
truncated corpus requires ``--allow-prefix`` and remains labeled as such.

The output retains the learned-residual v2 feature schema and adds dataset
schema-v4 fields: integer ``track_id``/``session_id``, non-pickle string name
tables, and a JSON manifest with source/config/evaluator and numerical-
environment provenance. Publication is atomic.
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

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Every current deterministic course completes under evaluator v4. Do not
# silently omit a topology from grouped training data; an unsafe/incomplete
# course fails the entire publication below.
_SKIP: set[str] = set()
_AUTO_COMPLETION_MIN_DURATION_S = 45.0
_AUTO_COMPLETION_GRACE_S = 15.0
_DATASET_SCHEMA_VERSION = 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npz(path: Path, payload: dict) -> None:
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


def _collection_duration(prepared, requested: float | None) -> float:
    if requested is not None:
        value = float(requested)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("duration must be finite and positive")
        return value
    return max(
        _AUTO_COMPLETION_MIN_DURATION_S,
        float(prepared.trajectory.total_time) + _AUTO_COMPLETION_GRACE_S,
    )


def _completion_contract_valid(result: dict) -> bool:
    """Validate exact completion evidence before publishing any trace."""

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
    if result["skipped"] is not False:
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
    ):
        return False
    return True


def _accepted_collection_outcome(result: dict, *, allow_prefix: bool) -> bool:
    """Require exact, mutually coherent safety and terminal evidence."""

    if (
        not _completion_contract_valid(result)
        or result.get("safety_passed") is not True
        or result.get("validity_passed") is not True
        or result.get("crashed") is not False
        or result.get("disqualified") is not False
    ):
        return False
    if result.get("complete") is True:
        return result.get("termination_reason") == "race_complete"
    return bool(
        allow_prefix
        and result.get("complete") is False
        and result.get("termination_reason") == "time_limit"
    )


def collect(
    out_path: Path,
    duration: float | None = None,
    *,
    seed: int = 42,
    cache_root: Path | str | None = None,
    allow_prefix: bool = False,
) -> dict:
    """Collect one safe complete-session trace per non-skipped track.

    ``allow_prefix`` permits only safe/valid time-limit prefixes. It never
    permits a crash, DQ, invalid plan, or other terminal failure.
    """
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if type(allow_prefix) is not bool:
        raise TypeError("allow_prefix must be an exact bool")
    if duration is not None and (
        type(duration) not in {int, float}
        or not math.isfinite(duration)
        or duration <= 0.0
    ):
        raise ValueError("duration must be finite and positive")
    from control.learned_residual import save_feature_trace
    from planning.artifact_cache import dependency_fingerprint
    from scripts.benchmark import EVALUATOR_VERSION, prepare_course, simulate
    from scripts.benchmark_matrix import _list_configs

    out_path = Path(out_path)
    combined: list = []
    per_track: list[dict] = []
    track_names: list[str] = []
    track_ids: list[int] = []
    session_names: list[str] = []
    session_ids: list[int] = []
    rejected: list[str] = []
    track_config_sha256: dict[str, str] = {}

    for config_path in _list_configs():
        name = config_path.stem
        if name in _SKIP:
            per_track.append({"track": name, "skipped": True})
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        track_config_sha256[name] = _sha256_file(config_path)
        prepared = prepare_course(config, cache_root=cache_root)
        track_duration = _collection_duration(prepared, duration)
        result = simulate(
            prepared,
            controller_config={"trace_features": True, "use_residual": False},
            seed=seed,
            duration=track_duration,
        )
        # The current prepared evaluator guarantees these explicit fields.
        # Missing evidence is not equivalent to a pass.
        safety_passed = result.get("safety_passed") is True
        validity_passed = result.get("validity_passed") is True
        complete = result.get("complete") is True
        accepted = _accepted_collection_outcome(
            result, allow_prefix=allow_prefix
        )
        if not accepted:
            rejected.append(name)
            per_track.append(
                {
                    "track": name,
                    "skipped": True,
                    "skip_reason": (
                        "unsafe_or_invalid"
                        if not safety_passed or not validity_passed
                        else "race_incomplete"
                    ),
                    "termination_reason": result.get("termination_reason"),
                    "gates_passed": result.get("gates_passed"),
                    "total_gates": result.get("total_gates"),
                }
            )
            continue

        trace = result.get("tracker_feature_trace") or []
        if not trace:
            rejected.append(name)
            per_track.append(
                {"track": name, "skipped": True, "skip_reason": "empty_trace"}
            )
            continue
        track_id = len(track_names)
        session_id = len(session_names)
        track_names.append(name)
        session_names.append(f"{name}:seed-{seed}")
        track_ids.extend([track_id] * len(trace))
        session_ids.extend([session_id] * len(trace))
        combined.extend(trace)
        per_track.append(
            {
                "track": name,
                "samples": len(trace),
                "sim_passed": result.get("sim_passed"),
                "sim_time_s": result.get("sim_time_s"),
                "evaluation_duration_s": track_duration,
                "avg_tracking_error_m": result.get("avg_tracking_error_m"),
                "complete": complete,
                "accepted_as": "complete" if complete else "prefix",
                "prepared_artifact_key": prepared.artifact_key,
                "prepared_cache_states": prepared.cache_states,
            }
        )

    if rejected:
        raise RuntimeError(
            "refusing to write a partial/unsafe residual dataset; rejected "
            f"tracks: {rejected}. Increase --duration or use --allow-prefix "
            "only for an explicitly diagnostic corpus."
        )
    if not combined:
        raise RuntimeError("no traces collected; every track returned an empty trace")

    # Use the canonical trace serializer on a temporary file, then add grouped
    # provenance fields and atomically publish one immutable dataset.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_trace = tempfile.mkstemp(
        prefix=f".{out_path.name}.trace-",
        suffix=".npz",
        dir=str(out_path.parent),
    )
    os.close(fd)
    trace_path = Path(raw_trace)
    try:
        save_feature_trace(combined, trace_path)
        with np.load(trace_path, allow_pickle=False) as archive:
            payload = {key: archive[key].copy() for key in archive.files}
    finally:
        trace_path.unlink(missing_ok=True)

    payload["track_id"] = np.asarray(track_ids, dtype=np.int64)
    payload["track_names"] = np.asarray(track_names, dtype=np.str_)
    payload["session_id"] = np.asarray(session_ids, dtype=np.int64)
    payload["session_names"] = np.asarray(session_names, dtype=np.str_)
    payload["dataset_schema_version"] = np.asarray(
        _DATASET_SCHEMA_VERSION, dtype=np.int64
    )
    manifest = {
        "schema_version": _DATASET_SCHEMA_VERSION,
        "collection_mode": "prefix" if allow_prefix else "completion",
        "seed": seed,
        "requested_duration_s": duration,
        "collector_source_sha256": _sha256_file(Path(__file__)),
        "evaluator_version": EVALUATOR_VERSION,
        "dependency_fingerprint": dependency_fingerprint(),
        "track_config_sha256": track_config_sha256,
        "tracks": per_track,
    }
    payload["dataset_manifest_json"] = np.asarray(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    _atomic_save_npz(out_path, payload)

    return {
        "total_samples": len(combined),
        "tracks": per_track,
        "track_names": track_names,
        "session_names": session_names,
        "duration_per_track_s": duration,
        "collection_mode": "prefix" if allow_prefix else "completion",
        "dataset_schema_version": _DATASET_SCHEMA_VERSION,
        "dataset_sha256": _sha256_file(out_path),
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect grouped tracker-residual synthetic sessions",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "control" / "residual_dataset.npz"),
        help="Output .npz path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Per-track horizon. Default is max(45s, trajectory time + 15s) "
            "so nominal sessions can complete."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument(
        "--allow-prefix",
        action="store_true",
        help=(
            "Allow safe/valid time-limit prefixes and label the dataset as "
            "diagnostic; never accepts crash/DQ/invalid traces."
        ),
    )
    args = parser.parse_args()
    summary = collect(
        Path(args.out),
        duration=args.duration,
        seed=args.seed,
        cache_root=args.cache_root,
        allow_prefix=args.allow_prefix,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
