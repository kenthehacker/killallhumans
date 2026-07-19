"""Trusted, fail-closed promotion evidence for synthetic non-live tiers.

This adapter deliberately does not execute a powered simulator.  It validates
the exact structured output of ``scripts.benchmark_matrix`` and translates
that evidence into the promotion ladder's hard gates and lexicographic quality
vector.  Cleanup and stale-stream gates are only true when the input proves the
pure synthetic evaluator domain, where those powered-flight hazards are
vacuously absent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ._util import (
    json_hash,
    read_secure_regular_file,
    secure_directory,
    secure_relative_regular_file,
    sha256_bytes,
    strict_json_loads,
)
from .promotion import HardGates, QualityVector, Tier


NONLIVE_EVIDENCE_SCHEMA = "aigp-nonlive-promotion-evidence/1"
FULL_TRACK_SET = (
    "aigp_default",
    "figure8",
    "grand_tour",
    "race_01",
    "slalom",
    "straight_hairpin",
    "vertical_cliff",
)
DOMAIN_TRACK_SET = ("grand_tour", "slalom", "vertical_cliff")
TRACK_CONFIG_FILES = {
    f"sim_pybullet/configs/{name}.json"
    for name in (
        "aigp_default",
        "figure8",
        "grand_tour",
        "race_01",
        "slalom",
        "straight_hairpin",
        "vertical_cliff",
    )
}
CORE_EVALUATOR_FILES = {
    # Importing any ``aigp_loop`` submodule executes this package bootstrap,
    # whose transitive local imports must be pinned before evidence code runs.
    "aigp_loop/__init__.py",
    "aigp_loop/_util.py",
    "aigp_loop/evidence.py",
    "aigp_loop/ledger.py",
    "aigp_loop/nonlive.py",
    "aigp_loop/promotion.py",
    "aigp_loop/scheduler.py",
    "planning/__init__.py",
    "planning/artifact_cache.py",
    "scripts/aigp_nonlive.py",
    "scripts/benchmark.py",
    "scripts/benchmark_matrix.py",
} | TRACK_CONFIG_FILES
_NUMERIC_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
}


def _exact_number(value: Any, *, minimum: Optional[float] = None) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("promotion metric must be an exact finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError("promotion metric is below its valid range")
    return result


def _exact_sha_map(value: Any) -> Dict[str, str]:
    if type(value) is not dict:
        raise ValueError("artifact hashes must be an exact object")
    result: Dict[str, str] = {}
    for name, digest in value.items():
        if (
            type(name) is not str
            or not name
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("artifact hashes must contain named SHA-256 digests")
        result[name] = digest
    return result


def _validate_dependency_fingerprint(value: Any, *, label: str) -> Dict[str, Any]:
    expected_fields = {
        "python",
        "implementation",
        "platform",
        "machine",
        "dependencies",
        "numpy_build",
        "numeric_thread_environment",
    }
    if (
        type(value) is not dict
        or set(value) != expected_fields
        or any(
            type(value[name]) is not str or not value[name]
            for name in ("python", "implementation", "platform", "machine")
        )
        or type(value["dependencies"]) is not dict
        or set(value["dependencies"]) != {"numpy", "scipy"}
        or any(
            dependency is not None
            and (type(dependency) is not str or not dependency)
            for dependency in value["dependencies"].values()
        )
        or type(value["numpy_build"]) is not dict
        or type(value["numeric_thread_environment"]) is not dict
        or not value["numeric_thread_environment"]
        or any(
            type(name) is not str
            or not name
            or (setting is not None and type(setting) is not str)
            for name, setting in value["numeric_thread_environment"].items()
        )
    ):
        raise ValueError(f"matrix {label} dependency fingerprint is missing")
    try:
        json_hash(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"matrix {label} dependency fingerprint is not canonical JSON"
        ) from exc
    return dict(value)


def _validate_code_provenance(value: Any, *, label: str) -> Dict[str, Any]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "commit",
            "dirty",
            "dirty_diff_hash",
            "tracked_diff_hash",
            "untracked_content_hash",
            "excluded_untracked_paths",
        }
        or type(value["commit"]) is not str
        or not value["commit"]
        or type(value["dirty"]) is not bool
        or type(value["excluded_untracked_paths"]) is not list
        or any(type(path) is not str for path in value["excluded_untracked_paths"])
    ):
        raise ValueError(f"matrix {label} code provenance is malformed")
    _exact_sha_map(
        {
            name: value[name]
            for name in (
                "dirty_diff_hash",
                "tracked_diff_hash",
                "untracked_content_hash",
            )
        }
    )
    return dict(value)


def evaluator_source_hashes(
    repository: Path | str, trusted_manifest: Optional[Path | str] = None
) -> Dict[str, str]:
    """Hash the evaluator boundary that must be pinned by the scheduler."""

    root = Path(repository).resolve()
    if trusted_manifest is None:
        relative = (
            "aigp_loop/nonlive.py",
            "scripts/aigp_nonlive.py",
            "scripts/benchmark.py",
            "scripts/benchmark_matrix.py",
        )
        return {
            name: sha256_bytes(
                read_secure_regular_file(
                    secure_relative_regular_file(root, name)
                )
            )
            for name in relative
        }
    manifest_path = Path(trusted_manifest)
    if manifest_path.is_absolute():
        try:
            manifest_relative = manifest_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("trusted evaluator manifest escapes repository") from exc
    else:
        manifest_relative = manifest_path
    manifest_path = secure_relative_regular_file(root, manifest_relative)
    manifest_payload = read_secure_regular_file(manifest_path)
    try:
        manifest = strict_json_loads(manifest_payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("trusted evaluator manifest must be UTF-8") from exc
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "files"}
        or manifest.get("schema") != "aigp-trusted-evaluator-files/1"
        or type(manifest.get("files")) is not dict
        or not manifest["files"]
    ):
        raise ValueError("trusted evaluator manifest has an invalid exact schema")
    result = _exact_sha_map(manifest["files"])
    missing_core = CORE_EVALUATOR_FILES - set(result)
    if missing_core:
        raise ValueError(
            "trusted evaluator manifest omits required core files: "
            + ", ".join(sorted(missing_core))
        )
    for name, expected in result.items():
        try:
            target = secure_relative_regular_file(root, name)
        except ValueError as exc:
            raise ValueError(f"trusted evaluator file mismatch: {name}") from exc
        if sha256_bytes(read_secure_regular_file(target)) != expected:
            raise ValueError(f"trusted evaluator file mismatch: {name}")
    config_root = secure_directory(root / "sim_pybullet" / "configs")
    observed_configs = {
        path.relative_to(root).as_posix() for path in config_root.iterdir()
    }
    if observed_configs != TRACK_CONFIG_FILES:
        raise ValueError("track config inventory differs from reviewed exact set")
    relative_manifest = manifest_path.relative_to(root).as_posix()
    result[relative_manifest] = sha256_bytes(manifest_payload)
    return result


def _validate_track(name: str, summary: Any, result: Any) -> Dict[str, Any]:
    if type(summary) is not dict or type(result) is not dict:
        raise ValueError(f"{name}: missing exact summary/full result evidence")
    required_bools = (
        "complete",
        "crashed",
        "disqualified",
        "sim_passed",
        "safety_passed",
        "validity_passed",
    )
    if any(type(summary.get(field)) is not bool for field in required_bools):
        raise ValueError(f"{name}: hard-gate fields must be exact booleans")
    if result.get("sim_type") != "synthetic_kinematic":
        raise ValueError(f"{name}: evaluator did not prove synthetic-only execution")
    if result.get("available") is not True or result.get("skipped") is not False:
        raise ValueError(f"{name}: synthetic evaluation was absent or skipped")
    for field in required_bools:
        if result.get(field) is not summary[field]:
            raise ValueError(f"{name}: summary/full result contradiction for {field}")
    gates_passed = summary.get("gates_passed")
    total_gates = summary.get("total_gates")
    if (
        type(gates_passed) is not int
        or type(total_gates) is not int
        or total_gates <= 0
        or not 0 <= gates_passed <= total_gates
    ):
        raise ValueError(f"{name}: gate counts are malformed")
    if summary["complete"] is not (gates_passed == total_gates):
        raise ValueError(f"{name}: completion contradicts exact gate counts")
    gate_rate = _exact_number(summary.get("gate_pass_rate"), minimum=0.0)
    if gate_rate > 1.0 or not math.isclose(
        gate_rate, gates_passed / total_gates, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{name}: gate pass rate contradicts counts")
    completion = summary.get("completion")
    if (
        type(completion) is not dict
        or type(completion.get("complete")) is not bool
        or type(completion.get("gates_passed")) is not int
        or type(completion.get("total_gates")) is not int
        or completion["complete"] is not summary["complete"]
        or completion["gates_passed"] != gates_passed
        or completion["total_gates"] != total_gates
    ):
        raise ValueError(f"{name}: completion evidence is malformed or contradictory")
    validation = summary.get("plan_validation")
    if (
        type(validation) is not dict
        or type(validation.get("ok")) is not bool
        or validation.get("ok") is not summary["validity_passed"]
        or validation.get("gates_passed") != gates_passed
        or validation.get("total_gates") != total_gates
        or validation.get("crashed") is not summary["crashed"]
        or validation.get("disqualified") is not summary["disqualified"]
    ):
        raise ValueError(f"{name}: ordered gate-sequence evidence is malformed")
    p95_tracking = _exact_number(summary.get("p95_tracking_error_m"), minimum=0.0)
    max_tracking = _exact_number(summary.get("max_tracking_error_m"), minimum=0.0)
    sim_time = _exact_number(summary.get("sim_time_s"), minimum=0.0)
    cache_state = summary.get("cache_hit_or_miss")
    if cache_state not in {"hit", "miss", "not_applicable"}:
        raise ValueError(f"{name}: cache evidence is missing or malformed")
    prepared_cache_states = summary.get("prepared_cache_states")
    if (
        type(prepared_cache_states) is not dict
        or set(prepared_cache_states)
        != {"racing_line", "trajectory", "plan_validation", "ilc"}
        or any(state not in {"hit", "miss"} for state in prepared_cache_states.values())
    ):
        raise ValueError(f"{name}: prepared-cache evidence is missing or malformed")
    if type(summary.get("rollout_executed")) is not bool or type(
        summary.get("result_cache_enabled")
    ) is not bool:
        raise ValueError(f"{name}: rollout execution evidence is missing")
    result_cache_states = result.get("cache")
    if (
        type(result_cache_states) is not dict
        or set(result_cache_states)
        != {
            "racing_line",
            "trajectory",
            "plan_validation",
            "ilc",
            "benchmark_result",
        }
        or any(state not in {"hit", "miss"} for state in result_cache_states.values())
        or {
            layer: result_cache_states[layer] for layer in prepared_cache_states
        }
        != prepared_cache_states
        or result_cache_states["benchmark_result"] != cache_state
        or summary["rollout_executed"] is not (cache_state == "miss")
    ):
        raise ValueError(f"{name}: cache/rollout evidence is contradictory")
    source_artifacts = _exact_sha_map(summary.get("artifact_hashes", {}))
    compared_fields = (
        "gates_passed",
        "total_gates",
        "gate_pass_rate",
        "completion",
        "plan_validation",
        "p95_tracking_error_m",
        "max_tracking_error_m",
        "sim_time_s",
        "artifact_hashes",
        "cache_hit_or_miss",
        "rollout_executed",
        "result_cache_enabled",
    )
    if any(result.get(field) != summary.get(field) for field in compared_fields):
        raise ValueError(f"{name}: summary/full result evidence is contradictory")
    if summary["sim_passed"] is True and not (
        summary["complete"]
        and summary["safety_passed"]
        and summary["validity_passed"]
        and not summary["crashed"]
        and not summary["disqualified"]
    ):
        raise ValueError(f"{name}: sim_passed contradicts hard-gate evidence")
    return {
        "complete": summary["complete"],
        "crashed": summary["crashed"],
        "disqualified": summary["disqualified"],
        "sim_passed": summary["sim_passed"],
        "safety_passed": summary["safety_passed"],
        "validity_passed": summary["validity_passed"],
        "gates_passed": gates_passed,
        "total_gates": total_gates,
        "gate_pass_rate": gate_rate,
        "p95_tracking_error_m": p95_tracking,
        "max_tracking_error_m": max_tracking,
        "sim_time_s": sim_time,
        "artifact_hashes": source_artifacts,
        "cache_hit_or_miss": cache_state,
        "prepared_cache_states": dict(prepared_cache_states),
        "rollout_executed": summary["rollout_executed"],
        "result_cache_enabled": summary["result_cache_enabled"],
    }


def adapt_matrix_evidence(
    matrix: Mapping[str, Any],
    *,
    tier: Tier,
    expected_tracks: Sequence[str],
    source_hashes: Mapping[str, str],
    full_nonlive_suite: Optional[Mapping[str, Any]] = None,
    cache_preparation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a matrix report and derive non-live promotion evidence."""

    if tier not in {
        Tier.T2_WARM_SIM,
        Tier.T3_DOMAIN_TRACKS,
        Tier.T4_FULL_NON_LIVE,
    }:
        raise ValueError("matrix promotion evidence is only valid for T2-T4")
    if type(matrix) is not dict:
        raise ValueError("matrix evidence must be an exact object")
    expected = tuple(expected_tracks)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or any(type(name) is not str or not name for name in expected)
    ):
        raise ValueError("expected tracks must be unique non-empty strings")
    if tier is Tier.T2_WARM_SIM and expected != ("race_01",):
        raise ValueError("T2 is exactly the one-track race_01 warm evaluation")
    if tier is Tier.T3_DOMAIN_TRACKS and tuple(sorted(expected)) != DOMAIN_TRACK_SET:
        raise ValueError("T3 requires the exact reviewed domain-track subset")
    if tier is Tier.T4_FULL_NON_LIVE and tuple(sorted(expected)) != FULL_TRACK_SET:
        raise ValueError("T4 requires the exact seven-track matrix")
    if tier is not Tier.T4_FULL_NON_LIVE and full_nonlive_suite is not None:
        raise ValueError("full-suite evidence belongs only to T4")
    if tier is not Tier.T2_WARM_SIM and cache_preparation is not None:
        raise ValueError("cache-preparation evidence belongs only to T2")
    if tier is Tier.T2_WARM_SIM:
        if (
            type(cache_preparation) is not dict
            or set(cache_preparation)
            != {
                "schema",
                "preparation_result_sha256",
                "dependency_fingerprint_sha256",
                "cache_hit_or_miss",
            }
            or cache_preparation.get("schema") != "aigp-cache-preparation/3"
            or cache_preparation.get("cache_hit_or_miss")
            not in {"hit", "miss"}
        ):
            raise ValueError("T2 requires exact cache-preparation evidence")
        _exact_sha_map(
            {
                "preparation_result": cache_preparation.get(
                    "preparation_result_sha256"
                ),
                "dependency_fingerprint": cache_preparation.get(
                    "dependency_fingerprint_sha256"
                ),
            }
        )
    if tier is Tier.T4_FULL_NON_LIVE:
        if (
            type(full_nonlive_suite) is not dict
            or set(full_nonlive_suite) != {
                "schema",
                "passed",
                "returncode",
                "pytest_args",
                "output_sha256",
                "output_tail",
            }
            or full_nonlive_suite.get("schema") != "aigp-nonlive-pytest/1"
            or type(full_nonlive_suite.get("passed")) is not bool
            or type(full_nonlive_suite.get("returncode")) is not int
            or full_nonlive_suite["returncode"] < 0
            or full_nonlive_suite.get("pytest_args")
            != [
                "-q",
                "-p",
                "pytest_timeout",
                "-p",
                "no:cacheprovider",
                "-o",
                "required_plugins=",
                "-c",
                "pyproject.toml",
                "-m",
                "not live",
                "--timeout=300",
                "tests",
                "competition/tests",
                "control/tests",
                "estimation/tests",
                "flight_control/tests",
                "gate_detection/tests",
                "gate_sequencing/tests",
                "planning/tests",
                "sim_pybullet/tests",
                "simulation/tests",
            ]
            or full_nonlive_suite["passed"]
            is not (full_nonlive_suite["returncode"] == 0)
            or type(full_nonlive_suite.get("output_tail")) is not str
            or len(full_nonlive_suite["output_tail"]) > 32_000
        ):
            raise ValueError("T4 requires exact full non-live pytest evidence")
        _exact_sha_map({"output": full_nonlive_suite.get("output_sha256")})

    tracks = matrix.get("tracks")
    results = matrix.get("results")
    if type(tracks) is not dict or type(results) is not dict:
        raise ValueError("matrix must include exact summaries and full results")
    if set(tracks) != set(expected) or set(results) != set(expected):
        raise ValueError("matrix track set does not match the frozen tier identity")
    if type(matrix.get("all_passed")) is not bool:
        raise ValueError("matrix all_passed must be an exact bool")
    validated = {
        name: _validate_track(name, tracks[name], results[name]) for name in expected
    }
    matrix_cache_state = matrix.get("cache_hit_or_miss")
    observed_cache_states = {track["cache_hit_or_miss"] for track in validated.values()}
    expected_matrix_cache_state = (
        "miss"
        if "miss" in observed_cache_states
        else "hit"
        if observed_cache_states == {"hit"}
        else "mixed-or-not-applicable"
    )
    if matrix_cache_state != expected_matrix_cache_state:
        raise ValueError("matrix cache evidence contradicts per-track evidence")
    if tier is Tier.T2_WARM_SIM and any(
        track["cache_hit_or_miss"] != "miss"
        or track["result_cache_enabled"] is not False
        or track["rollout_executed"] is not True
        or any(
            state != "hit" for state in track["prepared_cache_states"].values()
        )
        for track in validated.values()
    ):
        raise ValueError(
            "T2 requires warm prepared layers and a newly executed measured rollout"
        )
    source_hash_map = _exact_sha_map(dict(source_hashes))
    missing_core = CORE_EVALUATOR_FILES - set(source_hash_map)
    if missing_core:
        raise ValueError(
            "promotion evidence omits required core evaluator files: "
            + ", ".join(sorted(missing_core))
        )
    matrix_config_hash = matrix.get("config_hash")
    if (
        type(matrix_config_hash) is not str
        or len(matrix_config_hash) != 64
        or any(character not in "0123456789abcdef" for character in matrix_config_hash)
    ):
        raise ValueError("matrix config_hash must be a SHA-256 digest")
    dependency_fingerprint = _validate_dependency_fingerprint(
        matrix.get("dependency_fingerprint"), label="worker"
    )
    dependency_fingerprint_hash = json_hash(dependency_fingerprint)
    fingerprints = matrix.get("dependency_fingerprints")
    if type(fingerprints) is not dict or set(fingerprints) != {
        "orchestrator",
        "worker_expected",
        "workers_observed",
    }:
        raise ValueError("matrix dependency fingerprint provenance is missing")
    orchestrator_fingerprint = _validate_dependency_fingerprint(
        fingerprints["orchestrator"], label="orchestrator"
    )
    expected_worker_fingerprint = _validate_dependency_fingerprint(
        fingerprints["worker_expected"], label="expected worker"
    )
    observations = fingerprints["workers_observed"]
    if (
        matrix.get("worker_environment_verified") is not True
        or expected_worker_fingerprint != dependency_fingerprint
        or type(observations) is not list
        or len(observations) != len(expected)
        or any(
            type(observation) is not dict
            or set(observation) != {"track", "fingerprint"}
            or type(observation["track"]) is not str
            for observation in observations
        )
        or {observation["track"] for observation in observations} != set(expected)
        or any(
            _validate_dependency_fingerprint(
                observation["fingerprint"], label="observed worker"
            )
            != dependency_fingerprint
            for observation in observations
        )
        or dependency_fingerprint["numeric_thread_environment"]
        != _NUMERIC_THREAD_ENVIRONMENT
    ):
        raise ValueError("matrix worker dependency fingerprints are contradictory")
    resolved_configuration = matrix.get("resolved_configuration")
    if (
        type(resolved_configuration) is not dict
        or type(resolved_configuration.get("runtime")) is not dict
        or resolved_configuration["runtime"].get("worker_dependency_fingerprint")
        != dependency_fingerprint
    ):
        raise ValueError("matrix resolved configuration omits worker identity")
    from planning.artifact_cache import sha256_json

    if sha256_json(resolved_configuration) != matrix_config_hash:
        raise ValueError("matrix config_hash contradicts resolved configuration")
    if (
        tier is Tier.T2_WARM_SIM
        and cache_preparation["dependency_fingerprint_sha256"]
        != dependency_fingerprint_hash
    ):
        raise ValueError("T2 preparation and measured worker identities differ")
    code_provenance = _validate_code_provenance(
        matrix.get("code_provenance"), label="orchestrator"
    )
    provenance_observations = matrix.get("code_provenance_observations")
    if (
        matrix.get("code_provenance_verified") is not True
        or resolved_configuration.get("code_provenance") != code_provenance
        or type(provenance_observations) is not list
        or len(provenance_observations) != len(expected)
        or any(
            type(observation) is not dict
            or set(observation) != {"track", "code_provenance"}
            or type(observation["track"]) is not str
            for observation in provenance_observations
        )
        or {observation["track"] for observation in provenance_observations}
        != set(expected)
        or any(
            _validate_code_provenance(
                observation["code_provenance"], label="worker"
            )
            != code_provenance
            for observation in provenance_observations
        )
    ):
        raise ValueError("matrix code provenance observations are contradictory")
    seed = matrix.get("seed")
    if type(seed) is not int:
        raise ValueError("matrix seed must be an exact integer")
    matrix_valid = (
        matrix["all_passed"] is True
        and all(track["sim_passed"] is True for track in validated.values())
        and (
            tier is not Tier.T4_FULL_NON_LIVE
            or full_nonlive_suite["passed"] is True
        )
    )
    gates = HardGates(
        valid=matrix_valid,
        completed=all(track["complete"] is True for track in validated.values()),
        correct_gate_sequence=all(
            track["validity_passed"] is True for track in validated.values()
        ),
        cleanup_confirmed=True,
        no_collision=all(track["crashed"] is False for track in validated.values()),
        no_disqualification=all(
            track["disqualified"] is False for track in validated.values()
        ),
        no_stale_stream_flight=True,
    )
    quality = QualityVector(
        completion_reliability=min(
            track["gate_pass_rate"] for track in validated.values()
        ),
        centering_margin=-max(
            track["p95_tracking_error_m"] for track in validated.values()
        ),
        stability_margin=-max(
            track["max_tracking_error_m"] for track in validated.values()
        ),
        race_time_s=sum(track["sim_time_s"] for track in validated.values()),
    )
    hard_gates = {
        "valid": gates.valid,
        "completed": gates.completed,
        "correct_gate_sequence": gates.correct_gate_sequence,
        "cleanup_confirmed": gates.cleanup_confirmed,
        "no_collision": gates.no_collision,
        "no_disqualification": gates.no_disqualification,
        "no_stale_stream_flight": gates.no_stale_stream_flight,
    }
    quality_mapping = {
        "completion_reliability": quality.completion_reliability,
        "centering_margin": quality.centering_margin,
        "stability_margin": quality.stability_margin,
        "race_time_s": quality.race_time_s,
    }
    evaluator_identity = {
        "adapter_schema": NONLIVE_EVIDENCE_SCHEMA,
        "benchmark_evaluator_version": matrix.get("evaluator_version"),
        "benchmark_schema_version": matrix.get("schema_version"),
        "benchmark_comparison_series": matrix.get("comparison_series"),
        "source_sha256": source_hash_map,
        "orchestrator_dependency_fingerprint_sha256": json_hash(
            orchestrator_fingerprint
        ),
        "worker_dependency_fingerprint_sha256": dependency_fingerprint_hash,
        "code_provenance_sha256": json_hash(code_provenance),
    }
    if any(
        type(evaluator_identity[name]) is not str
        or not evaluator_identity[name]
        for name in (
            "benchmark_evaluator_version",
            "benchmark_schema_version",
            "benchmark_comparison_series",
        )
    ):
        raise ValueError("matrix evaluator identity is missing")
    artifacts = {
        "matrix_result": json_hash(matrix),
        "evaluator_identity": json_hash(evaluator_identity),
    }
    if cache_preparation is not None:
        artifacts["cache_preparation_result"] = cache_preparation[
            "preparation_result_sha256"
        ]
    for track_name, track in validated.items():
        for artifact_name, digest in track["artifact_hashes"].items():
            artifacts[f"track.{track_name}.{artifact_name}"] = digest
    if full_nonlive_suite is not None:
        artifacts["full_nonlive_pytest_output"] = full_nonlive_suite["output_sha256"]
    evaluation_config_hash = json_hash(
        {
            "schema": "aigp-nonlive-evaluation-config/1",
            "tier": int(tier),
            "tracks": list(expected),
            "matrix_config_sha256": matrix_config_hash,
            "cache_requirement": "measured_hit" if tier is Tier.T2_WARM_SIM else None,
            "full_nonlive_pytest_args": (
                full_nonlive_suite["pytest_args"]
                if full_nonlive_suite is not None
                else None
            ),
        }
    )
    evaluation_input_hash = json_hash(
        {
            "evaluation_config_sha256": evaluation_config_hash,
            "evaluator_identity_sha256": artifacts["evaluator_identity"],
            "dependency_fingerprint_sha256": dependency_fingerprint_hash,
        }
    )
    evaluation_result_hash = json_hash(
        {
            "matrix_result_sha256": artifacts["matrix_result"],
            "cache_preparation_result_sha256": artifacts.get(
                "cache_preparation_result"
            ),
            "full_nonlive_pytest_output_sha256": (
                full_nonlive_suite["output_sha256"]
                if full_nonlive_suite is not None
                else None
            ),
            "hard_gates": hard_gates,
            "quality": quality_mapping,
        }
    )
    return {
        "schema": NONLIVE_EVIDENCE_SCHEMA,
        "tier": int(tier),
        "track_identity": list(expected),
        "evaluator_identity": evaluator_identity,
        "domain_provenance": {
            "execution": "deterministic_synthetic_kinematic_nonpowered",
            "powered_resources_used": False,
            "cleanup_gate_semantics": "vacuously_true_only_after_synthetic_domain_proof",
            "stale_stream_gate_semantics": "vacuously_true_only_after_synthetic_domain_proof",
            "centering_proxy": "negative_worst_p95_tracking_error_m",
            "stability_proxy": "negative_worst_max_tracking_error_m",
        },
        "safety_and_completion_metrics": hard_gates,
        "promotion": {"hard_gates": hard_gates, "quality": quality_mapping},
        "tracks": validated,
        "cache_preparation": dict(cache_preparation) if cache_preparation else None,
        "full_nonlive_suite": dict(full_nonlive_suite) if full_nonlive_suite else None,
        "artifact_hashes": artifacts,
        "evaluation_input_hash": evaluation_input_hash,
        "evaluation_result_hash": evaluation_result_hash,
        "evaluation_config_sha256": evaluation_config_hash,
        "evaluator_version": (
            "aigp-nonlive/1:" + artifacts["evaluator_identity"]
        ),
        "repetitions": 1,
        "seed": seed,
    }
