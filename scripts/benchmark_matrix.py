#!/usr/bin/env python3
"""
Multi-track regression matrix (iter-003 A17).

Runs the synthetic benchmark against every track config in
`sim_pybullet/configs/` and produces a JSON summary. The goal is to
catch the failure mode the user explicitly called out: "if we optimize
for just a single drone racing course then it might be overfit."

A change that improves race_01 but regresses figure8 (or any other
track) by >25% on `gate_pass_rate`, or introduces a new `crashed=True`
or `disqualified=True` on a track that previously passed, is a
regression and should not ship.

Usage:
    python -m scripts.benchmark_matrix              # JSON to stdout
    python -m scripts.benchmark_matrix --human      # ASCII table to stderr
    python -m scripts.benchmark_matrix --configs race_01,aigp_default
    python -m scripts.benchmark_matrix --duration 15  # per-track sim time

Exit code:
    0 — every requested track executes, validates, completes, and passes
    1 — at least one track regressed
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import multiprocessing
import os
import stat
import sys
import time
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Mapping

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_THREAD_ENV_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
)


def _list_configs() -> List[Path]:
    """All JSON track configs in sim_pybullet/configs/, sorted by name."""
    cfg_dir = _REPO / "sim_pybullet" / "configs"
    return sorted(cfg_dir.glob("*.json"))


def _strict_config_loads(payload: str) -> Dict[str, Any]:
    """Decode one track config without duplicate keys or JSON extensions."""

    def unique_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in track config: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    decoded = json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise TypeError("track config root must be a JSON object")
    return decoded


def _load_config(path: Path) -> Dict[str, Any]:
    return _strict_config_loads(path.read_text(encoding="utf-8"))


def _captured_source_digest(path: Path, payload: bytes) -> str:
    """Match ``source_digest([path])`` using already-captured bytes."""

    resolved = path.resolve()
    try:
        label = resolved.relative_to(_REPO).as_posix()
    except ValueError:
        label = resolved.name
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")
    return digest.hexdigest()


def _stable_config_bytes(path: Path) -> tuple[Path, bytes]:
    """Read a regular config twice from one descriptor and reject path races."""

    requested = Path(path)
    requested_stat = requested.lstat()
    if stat.S_ISLNK(requested_stat.st_mode):
        raise ValueError("track config must not be a symbolic link")
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("track config must be a regular file")
        first = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            first.extend(chunk)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            second.extend(chunk)
        after = os.fstat(descriptor)
        named_after = resolved.lstat()
        requested_after = requested.lstat()
        stable_signature = lambda value: (
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            first != second
            or stable_signature(opened) != stable_signature(middle)
            or stable_signature(opened) != stable_signature(after)
            or not os.path.samestat(opened, middle)
            or not os.path.samestat(opened, after)
            or not os.path.samestat(opened, named_after)
            or not os.path.samestat(requested_stat, requested_after)
            or not os.path.samestat(opened, requested_after)
        ):
            raise RuntimeError("track config changed while it was being captured")
        return resolved, bytes(first)
    finally:
        os.close(descriptor)


def _snapshot_config(path: Path) -> Dict[str, Any]:
    """Capture the exact bytes and parsed payload workers will execute."""

    resolved = path.resolve()
    try:
        resolved, payload = _stable_config_bytes(path)
        data = _strict_config_loads(payload.decode("utf-8"))
    except Exception as error:
        return {
            "path": str(resolved),
            "name": path.stem,
            "data": None,
            "content_sha256": None,
            "source_digest": None,
            "error": f"load failed: {type(error).__name__}: {error}",
        }
    return {
        "path": str(resolved),
        "name": path.stem,
        "data": data,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "source_digest": _captured_source_digest(resolved, payload),
        "error": None,
    }


def _limit_worker_threads() -> None:
    """Prevent BLAS/OpenCV oversubscription inside each track process."""

    for variable in _THREAD_ENV_VARIABLES[:-2]:
        os.environ[variable] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass


@contextlib.contextmanager
def worker_numeric_environment():
    """Temporarily apply the exact numeric environment used by matrix workers.

    Preparation and measured execution must derive identical artifact keys.
    The context restores both environment variables and OpenCV's process-local
    thread setting, including when preparation or execution raises.
    """

    previous_environment = {
        variable: os.environ.get(variable) for variable in _THREAD_ENV_VARIABLES
    }
    cv2_module = None
    previous_cv2_threads = None
    try:
        try:
            import cv2 as cv2_module

            previous_cv2_threads = cv2_module.getNumThreads()
        except ImportError:
            cv2_module = None
        _limit_worker_threads()
        from planning.artifact_cache import dependency_fingerprint

        yield dependency_fingerprint()
    finally:
        for variable, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        if cv2_module is not None and previous_cv2_threads is not None:
            cv2_module.setNumThreads(previous_cv2_threads)


def _resolved_worker_dependency_fingerprint(
    orchestrator_fingerprint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve the numeric identity workers must report after thread capping.

    Derive this without temporarily mutating the orchestrator environment:
    importing NumPy before restoring those variables could permanently select
    a different BLAS runtime policy in the parent process.
    """

    worker_fingerprint = dict(orchestrator_fingerprint)
    numeric_environment = dict(
        orchestrator_fingerprint.get("numeric_thread_environment", {})
    )
    for variable in _THREAD_ENV_VARIABLES[:-2]:
        numeric_environment[variable] = "1"
    numeric_environment["OMP_DYNAMIC"] = "FALSE"
    numeric_environment["MKL_DYNAMIC"] = "FALSE"
    worker_fingerprint["numeric_thread_environment"] = numeric_environment
    return worker_fingerprint


def _run_tracks_in_caller_with_restored_numeric_state(tasks: List[tuple]) -> List[tuple]:
    """Sequential fallback that cannot poison the caller's numeric identity."""

    with worker_numeric_environment():
        return [_run_track_worker_with_identity(task) for task in tasks]


def _run_track_worker(task: tuple) -> tuple:
    """Top-level/pickle-safe worker for one independent synthetic track."""

    (
        path_text,
        captured_config,
        capture_error,
        duration,
        dt,
        cache_root,
        record_position_trace,
        use_result_cache,
    ) = task
    config_path = Path(path_text)
    name = config_path.stem
    if capture_error is not None:
        return name, None, None, capture_error
    data = captured_config
    if not isinstance(data, dict):
        return name, None, None, "load failed: captured config is not an object"
    try:
        from scripts.benchmark import run_synthetic_benchmark

        result = run_synthetic_benchmark(
            duration=duration,
            dt=dt,
            config=data,
            cache_root=cache_root,
            record_position_trace=record_position_trace,
            use_result_cache=use_result_cache,
        )
        return name, data, result, None
    except Exception as error:
        return (
            name,
            data,
            None,
            f"bench raised: {type(error).__name__}: {error}",
        )


def _run_track_worker_with_identity(task: tuple) -> tuple:
    """Execute one track bracketed by exact code and numeric identities."""

    worker_start_provenance = None
    worker_end_provenance = None
    try:
        from scripts import benchmark as benchmark_module

        worker_start_provenance = benchmark_module._git_provenance()
    except Exception as provenance_error:
        name = Path(task[0]).stem
        return (
            name,
            task[1] if len(task) > 1 else None,
            None,
            "worker start provenance failed: "
            f"{type(provenance_error).__name__}: {provenance_error}",
            None,
            None,
            None,
        )

    name, data, result, error = _run_track_worker(task)
    try:
        from planning.artifact_cache import dependency_fingerprint

        worker_fingerprint = dependency_fingerprint()
    except Exception as fingerprint_error:
        worker_fingerprint = None
        if error is None:
            result = None
            error = (
                "worker dependency fingerprint failed: "
                f"{type(fingerprint_error).__name__}: {fingerprint_error}"
            )
    try:
        worker_end_provenance = benchmark_module._git_provenance()
    except Exception as provenance_error:
        if error is None:
            result = None
            error = (
                "worker end provenance failed: "
                f"{type(provenance_error).__name__}: {provenance_error}"
            )
    return (
        name,
        data,
        result,
        error,
        worker_fingerprint,
        worker_start_provenance,
        worker_end_provenance,
    )


def _completion_contract_valid(track: Dict[str, Any]) -> bool:
    """Require exact, coherent completion evidence from one worker result."""

    bool_fields = (
        "complete",
        "sim_passed",
        "safety_passed",
        "validity_passed",
        "crashed",
        "disqualified",
    )
    if any(type(track.get(name)) is not bool for name in bool_fields):
        return False

    gates_passed = track.get("gates_passed")
    total_gates = track.get("total_gates")
    if (
        type(gates_passed) is not int
        or type(total_gates) is not int
        or total_gates <= 0
        or not 0 <= gates_passed <= total_gates
    ):
        return False
    completion = track.get("completion")
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
        or completion["complete"] is not track["complete"]
        or completion["gates_passed"] != gates_passed
        or completion["total_gates"] != total_gates
        or track["complete"] is not (gates_passed == total_gates)
    ):
        return False

    termination_reason = track.get("termination_reason")
    if type(termination_reason) is not str or not termination_reason:
        return False
    if track["complete"] is not (termination_reason == "race_complete"):
        return False
    expected_safety = not track["crashed"] and not track["disqualified"]
    if track["safety_passed"] is not expected_safety:
        return False
    if track["sim_passed"] and not (
        track["safety_passed"]
        and track["validity_passed"]
        and track["complete"]
        and not track["crashed"]
        and not track["disqualified"]
    ):
        return False
    return True


def run_matrix(
    configs: List[Path],
    duration: float = 30.0,
    dt: float = 0.01,
    *,
    max_workers: int | None = None,
    cache_root: str | Path | None = None,
    record_position_trace: bool = False,
    include_results: bool = False,
    use_result_cache: bool = True,
    _in_process_test_worker: bool = False,
) -> Dict[str, Any]:
    """Run independent tracks in capped worker processes and summarize them."""

    total_started = time.perf_counter()
    from scripts.benchmark import _exact_finite_float

    duration = _exact_finite_float("duration", duration, nonnegative=True)
    dt = _exact_finite_float("dt", dt, strictly_positive=True)
    for name, value in (
        ("record_position_trace", record_position_trace),
        ("include_results", include_results),
        ("use_result_cache", use_result_cache),
        ("_in_process_test_worker", _in_process_test_worker),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be an exact bool")
    if max_workers is not None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, Integral):
            raise TypeError("max_workers must be an exact integer")
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
    resolved_paths = [str(path.resolve()).casefold() for path in configs]
    normalized_stems = [path.stem.casefold() for path in configs]
    duplicate_paths = sorted(
        {value for value in resolved_paths if resolved_paths.count(value) > 1}
    )
    duplicate_stems = sorted(
        {value for value in normalized_stems if normalized_stems.count(value) > 1}
    )
    if duplicate_paths or duplicate_stems:
        details = []
        if duplicate_paths:
            details.append("duplicate paths: " + ", ".join(duplicate_paths))
        if duplicate_stems:
            details.append("duplicate stems: " + ", ".join(duplicate_stems))
        raise ValueError("matrix configs must be uniquely attributable; " + "; ".join(details))
    effective_workers = min(max_workers or 4, max(1, len(configs)))

    from planning.artifact_cache import ArtifactStore, dependency_fingerprint, sha256_json
    from scripts.benchmark import (
        BENCHMARK_RESULT_SCHEMA,
        COMPARISON_SERIES,
        EVALUATOR_VERSION,
        _git_provenance,
        _threshold_snapshot,
    )

    threshold_values = _threshold_snapshot()
    orchestrator_fingerprint = dependency_fingerprint()
    start_code_provenance = _git_provenance()
    config_snapshots = [_snapshot_config(path) for path in configs]
    worker_fingerprint = (
        _resolved_worker_dependency_fingerprint(orchestrator_fingerprint)
        if configs
        else None
    )
    resolved_configuration = {
        "tracks": [
            {
                "name": snapshot["name"],
                "path": snapshot["path"],
                "source_digest": snapshot["source_digest"],
                "content_sha256": snapshot["content_sha256"],
                "parsed_config_hash": (
                    sha256_json(snapshot["data"])
                    if snapshot["data"] is not None
                    else None
                ),
                "load_error": snapshot["error"],
            }
            for snapshot in config_snapshots
        ],
        "runtime": {
            "duration": duration,
            "dt": dt,
            "workers": effective_workers if configs else 0,
            "record_position_trace": record_position_trace,
            "include_results": include_results,
            "use_result_cache": use_result_cache,
            "thresholds": threshold_values,
            "worker_dependency_fingerprint": worker_fingerprint,
            "worker_start_method": (
                "in_process_test_only"
                if _in_process_test_worker and configs
                else ("spawn" if configs else "not_applicable")
            ),
        },
        # Bind the exact clean/dirty source snapshot expected of every worker.
        # A shared worktree edit during a matrix invalidates the whole evidence
        # set even when every numerical threshold happened to pass.
        "code_provenance": start_code_provenance,
        "orchestrator_code_provenance_expected": start_code_provenance,
    }

    matrix: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluator_version": EVALUATOR_VERSION,
        "schema_version": BENCHMARK_RESULT_SCHEMA,
        "comparison_series": COMPARISON_SERIES,
        "resolved_configuration": resolved_configuration,
        "config_hash": sha256_json(resolved_configuration),
        # Backward-compatible singular field identifies the environment that
        # performed numerical work.  The richer mapping preserves the parent
        # identity too and is populated with exact worker observations below.
        "dependency_fingerprint": (
            worker_fingerprint if worker_fingerprint is not None
            else orchestrator_fingerprint
        ),
        "dependency_fingerprints": {
            "orchestrator": orchestrator_fingerprint,
            "worker_expected": worker_fingerprint,
            "workers_observed": [],
        },
        "worker_environment_verified": None,
        "code_provenance": start_code_provenance,
        "code_provenance_observations": [],
        "worker_code_provenance_observations": [],
        "orchestrator_code_provenance_observations": {
            "before_matrix": start_code_provenance,
            "after_aggregation": None,
        },
        "code_provenance_verified": None,
        "seed": 42,
        "thresholds": threshold_values,
        "duration_s": duration,
        "dt": dt,
        "tracks": {},
        "all_passed": True,
        "regressions": [],
        "worker_processes": (
            effective_workers if configs and not _in_process_test_worker else 0
        ),
        "worker_slots": effective_workers if configs else 0,
        "worker_threads_capped": bool(configs) and not _in_process_test_worker,
        "worker_execution": (
            "caller_test_seam"
            if _in_process_test_worker and configs
            else ("isolated_processes_spawn" if configs else "not_run")
        ),
        "promotion_evidence_eligible": not _in_process_test_worker,
        "cache_hit_or_miss": "not_applicable" if not configs else "pending",
        "cache": {
            "root": str(ArtifactStore(cache_root).root),
            "track_result_references": [],
        },
        "safety_passed": None,
        "validity_passed": None,
        "completion": {
            "complete": None,
            "gates_passed": 0,
            "total_gates": 0,
        },
        "failure_summary": {
            "stdout_tail": "",
            "stderr_tail": "",
            "exception": None,
            "threshold_failures": [],
            "track_result_references": [],
        },
        "timing_scope": (
            "run_matrix() orchestration wall time; worker phase timers execute "
            "in parallel and are available in included per-track results"
        ),
    }
    if not configs:
        matrix["all_passed"] = False
        matrix["regressions"].append(
            "no track configs supplied; promotion evidence is missing"
        )
    if _in_process_test_worker:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise RuntimeError("in-process matrix worker is restricted to pytest")
        matrix["all_passed"] = False
        matrix["regressions"].append(
            "in-process test worker is not eligible promotion evidence"
        )

    tasks = [
        (
            snapshot["path"],
            snapshot["data"],
            snapshot["error"],
            duration,
            dt,
            str(Path(cache_root).resolve()) if cache_root is not None else None,
            record_position_trace,
            use_result_cache,
        )
        for snapshot in config_snapshots
    ]
    if not tasks:
        completed_tracks = []
    elif _in_process_test_worker:
        completed_tracks = _run_tracks_in_caller_with_restored_numeric_state(tasks)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_limit_worker_threads,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            completed_tracks = list(
                executor.map(_run_track_worker_with_identity, tasks)
            )

    worker_observations = [
        {"track": item[0], "fingerprint": item[4]}
        for item in completed_tracks
    ]
    matrix["dependency_fingerprints"]["workers_observed"] = worker_observations
    code_provenance_observations = [
        {
            "track": item[0],
            "code_provenance": (
                item[2].get("code_provenance")
                if isinstance(item[2], dict)
                else None
            ),
        }
        for item in completed_tracks
    ]
    matrix["code_provenance_observations"] = code_provenance_observations
    worker_code_provenance_observations = [
        {
            "track": item[0],
            # Five-field tuples are retained as a compatibility seam for
            # bounded test doubles; production workers always return both
            # explicit bracketing observations.
            "before_evaluator": (
                item[5]
                if len(item) > 5
                else (
                    item[2].get("code_provenance")
                    if isinstance(item[2], dict)
                    else None
                )
            ),
            "after_evaluator": (
                item[6]
                if len(item) > 6
                else (
                    item[2].get("code_provenance")
                    if isinstance(item[2], dict)
                    else None
                )
            ),
        }
        for item in completed_tracks
    ]
    matrix["worker_code_provenance_observations"] = (
        worker_code_provenance_observations
    )
    if configs:
        worker_environment_verified = (
            len(worker_observations) == len(configs)
            and all(
                observation["fingerprint"] == worker_fingerprint
                for observation in worker_observations
            )
        )
        matrix["worker_environment_verified"] = worker_environment_verified
        if worker_environment_verified:
            matrix["dependency_fingerprint"] = worker_observations[0]["fingerprint"]
        else:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                "worker dependency fingerprints are missing, inconsistent, "
                "or differ from the environment bound into config_hash"
            )
        code_provenance_verified = (
            len(code_provenance_observations) == len(configs)
            and len(worker_code_provenance_observations) == len(configs)
            and all(
                observation["code_provenance"] == start_code_provenance
                for observation in code_provenance_observations
            )
            and all(
                observation["before_evaluator"] == start_code_provenance
                and observation["after_evaluator"] == start_code_provenance
                for observation in worker_code_provenance_observations
            )
        )
        matrix["code_provenance_verified"] = code_provenance_verified
        if not code_provenance_verified:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                "worker code provenance is missing or differs from the source "
                "snapshot bound into config_hash"
            )
    completed_by_name = {item[0]: item for item in completed_tracks}
    if include_results:
        matrix["results"] = {}

    for cfg_path in configs:
        name = cfg_path.stem
        completed = completed_by_name[name]
        _, data, result, error, _worker_fingerprint = completed[:5]
        if error is not None:
            matrix["tracks"][name] = {"error": error}
            matrix["all_passed"] = False
            matrix["regressions"].append(f"{name}: {error}")
            matrix["failure_summary"]["threshold_failures"].append(
                f"{name}: {error}"
            )
            if matrix["failure_summary"]["exception"] is None:
                matrix["failure_summary"]["exception"] = error
            continue
        if include_results:
            matrix["results"][name] = result

        result_resolved = (
            result.get("resolved_configuration")
            if isinstance(result, dict)
            else None
        )
        config_snapshot_verified = bool(
            isinstance(result_resolved, dict)
            and result_resolved.get("track") == data
            and result.get("config_hash") == sha256_json(result_resolved)
        )

        # Iter-017: surface the clamp-engagement metrics added in
        # iter-016 so matrix consumers can track planner-vs-bench
        # over-commanding as a first-class signal.
        cts = result.get("controller_trace_summary") or {}
        placeholder_value = data.get("placeholder", False)
        placeholder_valid = type(placeholder_value) is bool
        track_summary = {
            "gates_passed": result.get("gates_passed", 0),
            "total_gates": result.get("total_gates", 0),
            "gate_pass_rate": result.get("gate_pass_rate", 0.0),
            "complete": result.get("complete", False),
            "crashed": result.get("crashed", False),
            "disqualified": result.get("disqualified", False),
            "termination_reason": result.get("termination_reason", "unknown"),
            "sim_time_s": result.get("sim_time_s", 0.0),
            "avg_tracking_error_m": result.get("avg_tracking_error_m", 0.0),
            "max_tracking_error_m": result.get("max_tracking_error_m", 0.0),
            "p95_tracking_error_m": result.get("p95_tracking_error_m", 0.0),
            "avg_nearest_path_error_m": result.get(
                "avg_nearest_path_error_m", 0.0
            ),
            "max_nearest_path_error_m": result.get(
                "max_nearest_path_error_m", 0.0
            ),
            "sim_passed": result.get("sim_passed", False),
            "safety_passed": result.get("safety_passed", False),
            "validity_passed": result.get("validity_passed", False),
            "completion": result.get("completion"),
            "plan_validation": result.get("plan_validation"),  # iter-004 Phase 1
            "evaluator_version": result.get("evaluator_version"),
            "schema_version": result.get("schema_version"),
            "config_hash": result.get("config_hash"),
            "config_snapshot_verified": config_snapshot_verified,
            "artifact_hashes": result.get("artifact_hashes", {}),
            "cache_hit_or_miss": result.get(
                "cache_hit_or_miss", "not_applicable"
            ),
            "prepared_cache_states": {
                layer: result.get("cache", {}).get(layer)
                for layer in (
                    "racing_line",
                    "trajectory",
                    "plan_validation",
                    "ilc",
                )
            },
            "rollout_executed": result.get("rollout_executed"),
            "result_cache_enabled": result.get("result_cache_enabled"),
            "threshold_failures": result.get("threshold_failures", []),
            "is_placeholder": placeholder_value if placeholder_valid else False,
            "accel_clamp_active_frac": float(cts.get("accel_clamp_active_frac", 0.0)),
            "max_accel_mag_pre_clamp": float(cts.get("max_accel_mag_pre_clamp", 0.0)),
        }
        matrix["tracks"][name] = track_summary
        matrix["cache"]["track_result_references"].append(
            f"tracks.{name}.cache_hit_or_miss"
        )
        matrix["failure_summary"]["track_result_references"].append(
            f"tracks.{name}"
        )

        if not config_snapshot_verified:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: worker result config does not match the immutable "
                "track snapshot bound into the matrix config_hash"
            )

        if not placeholder_valid:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: placeholder must be an exact JSON boolean"
            )

        # Safety, plan validity, and exact completion are hard gates for every
        # course.  A placeholder label may relax only separately reported
        # performance diagnostics; it can never turn an incomplete rollout or
        # a failed evaluator into promotion evidence.
        if track_summary["crashed"]:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: crashed ({track_summary['termination_reason']})"
            )
        if track_summary["disqualified"]:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: disqualified ({track_summary['termination_reason']})"
            )
        if track_summary["safety_passed"] is not True:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: safety_passed={track_summary['safety_passed']!r}"
            )
        if track_summary["validity_passed"] is not True:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: validity_passed={track_summary['validity_passed']!r}"
            )
        completion_valid = _completion_contract_valid(track_summary)
        if not completion_valid:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: completion evidence missing, malformed, or contradictory"
            )
        pv = track_summary.get("plan_validation")
        if not isinstance(pv, dict):
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: plan_validation missing or malformed — bench didn't emit a mapping"
            )
        elif pv.get("ok") is not True:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: plan_validation.ok={pv.get('ok')!r} — "
                f"{pv.get('reason', 'unknown')}"
            )

        if track_summary["complete"] is not True:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: completion is not explicitly true"
            )
        if track_summary["sim_passed"] is not True:
            matrix["all_passed"] = False
            matrix["regressions"].append(
                f"{name}: sim_passed={track_summary['sim_passed']!r}"
            )
        if not track_summary["is_placeholder"]:
            if track_summary["gate_pass_rate"] < 0.75:
                matrix["all_passed"] = False
                matrix["regressions"].append(
                    f"{name}: gate_pass_rate {track_summary['gate_pass_rate']:.0%} < 75%"
                )

    successful_tracks = [
        track for track in matrix["tracks"].values() if "error" not in track
    ]
    evidence_complete = len(successful_tracks) == len(configs)
    if successful_tracks:
        matrix["safety_passed"] = evidence_complete and all(
            track.get("safety_passed") is True for track in successful_tracks
        )
        matrix["validity_passed"] = evidence_complete and all(
            track.get("validity_passed") is True for track in successful_tracks
        )
        matrix["completion"] = {
            "complete": evidence_complete and all(
                _completion_contract_valid(track)
                and track.get("complete") is True
                for track in successful_tracks
            ),
            "gates_passed": sum(
                int(track.get("gates_passed", 0)) for track in successful_tracks
            ),
            "total_gates": sum(
                int(track.get("total_gates", 0)) for track in successful_tracks
            ),
            "track_result_references": [
                f"tracks.{name}.completion"
                for name, track in matrix["tracks"].items()
                if "error" not in track
            ],
            "requested_tracks": len(configs),
            "evaluated_tracks": len(successful_tracks),
            "evidence_complete": evidence_complete,
        }
        cache_states = {
            track.get("cache_hit_or_miss", "not_applicable")
            for track in successful_tracks
        }
        if "miss" in cache_states:
            matrix["cache_hit_or_miss"] = "miss"
        elif cache_states == {"hit"}:
            matrix["cache_hit_or_miss"] = "hit"
        else:
            matrix["cache_hit_or_miss"] = "mixed-or-not-applicable"
    elif configs:
        matrix["cache_hit_or_miss"] = "unavailable"
        matrix["safety_passed"] = False
        matrix["validity_passed"] = False
        matrix["completion"] = {
            "complete": False,
            "gates_passed": 0,
            "total_gates": 0,
            "track_result_references": [],
            "requested_tracks": len(configs),
            "evaluated_tracks": 0,
            "evidence_complete": False,
        }

    end_code_provenance = _git_provenance()
    matrix["orchestrator_code_provenance_observations"][
        "after_aggregation"
    ] = end_code_provenance
    orchestrator_code_provenance_verified = (
        end_code_provenance == start_code_provenance
    )
    matrix["orchestrator_code_provenance_verified"] = (
        orchestrator_code_provenance_verified
    )
    if configs:
        matrix["code_provenance_verified"] = bool(
            matrix["code_provenance_verified"]
            and orchestrator_code_provenance_verified
        )
    if not orchestrator_code_provenance_verified:
        matrix["all_passed"] = False
        matrix["regressions"].append(
            "orchestrator code provenance changed while the matrix was running"
        )
    matrix["failure_summary"]["threshold_failures"] = list(
        matrix["regressions"]
    )
    total_wall = time.perf_counter() - total_started
    matrix["wall_time_s"] = total_wall
    matrix["phase_timings_s"] = {
        "matrix_orchestration": total_wall,
        "total_wall": total_wall,
    }
    matrix["timing_consistency"] = {
        "mutually_exclusive_phase_sum_s": total_wall,
        "total_covers_phases": True,
    }
    return matrix


def _print_human(matrix: Dict[str, Any], file=sys.stderr) -> None:
    p = lambda *a, **kw: print(*a, **kw, file=file)
    p(f"\n{'=' * 96}")
    p("AI Grand Prix — Multi-Track Regression Matrix")
    p(f"{'=' * 96}")
    p(f"Duration per track: {matrix['duration_s']}s    dt: {matrix['dt']}s\n")
    # Iter-019: widened table to show tracking error, sim_time, and the
    # iter-016/017 clamp engagement metric. Hidden in JSON mode (machine
    # consumers still read full track_summary dicts).
    fmt = "{:<20} {:>6} {:>5} {:>6} {:>4} {:>4} {:>6} {:>6} {:>7} {:>9} {:<14}"
    p(fmt.format(
        "track", "gates", "%", "time_s", "crash", "DQ",
        "err_m", "sat_a%", "peak_a", "term", "tag",
    ))
    p("-" * 96)
    for name, t in matrix["tracks"].items():
        if "error" in t:
            p(fmt.format(name, "-", "-", "-", "-", "-", "-", "-", "-", "-",
                         f"ERR: {t['error'][:13]}"))
            continue
        tag = "placeholder" if t["is_placeholder"] else "production"
        p(fmt.format(
            name,
            f"{t['gates_passed']}/{t['total_gates']}",
            f"{int(t['gate_pass_rate']*100)}",
            f"{t['sim_time_s']:.1f}",
            "Y" if t["crashed"] else "N",
            "Y" if t["disqualified"] else "N",
            f"{t['avg_tracking_error_m']:.3f}",
            f"{int(t.get('accel_clamp_active_frac', 0.0) * 100)}",
            f"{t.get('max_accel_mag_pre_clamp', 0.0):.1f}",
            t["termination_reason"][:9],
            tag,
        ))
    p("\n" + "=" * 96)
    if matrix["all_passed"]:
        p("Overall: PASS")
    else:
        p(f"Overall: FAIL ({len(matrix['regressions'])} regression(s))")
        for r in matrix["regressions"]:
            p(f"  - {r}")
    p("=" * 96)


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic bench across all track configs",
    )
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated config names (without .json); default: all in sim_pybullet/configs/",
    )
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Per-track sim duration in seconds (default 30)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Independent track worker processes (default 4; BLAS/OpenCV threads capped at 1)",
    )
    parser.add_argument("--human", action="store_true",
                        help="Also print ASCII table to stderr")
    parser.add_argument("--json-only", action="store_true",
                        help="Suppress the ASCII table (overrides --human)")
    args = parser.parse_args()

    if args.configs:
        wanted = set(args.configs.split(","))
        all_paths = _list_configs()
        paths = [p for p in all_paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            print(f"ERROR: unknown configs: {sorted(missing)}", file=sys.stderr)
            return 2
    else:
        paths = _list_configs()

    if not paths:
        print("ERROR: no track configs found", file=sys.stderr)
        return 2

    matrix = run_matrix(
        paths, duration=args.duration, dt=args.dt, max_workers=args.workers
    )
    print(json.dumps(matrix, indent=2, allow_nan=False))
    if args.human and not args.json_only:
        _print_human(matrix)

    return 0 if matrix["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
