"""Read-only provenance/smoke audit for the historical YOLO pose run.

This is intentionally not a training wrapper. The source dataset and the
pipeline named by the historical architecture document are absent, so starting
or resuming training would create a false impression of reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REQUIRED_PIPELINE_FILES = (
    "extract_frames.py",
    "train.py",
    "validate.py",
    "export.py",
)
REQUIRED_TRAINING_PACKAGES = ("torch", "ultralytics", "onnxruntime")
CAMPAIGN_CONTRACT_NAME = "campaign_contract.json"
CAMPAIGN_CONTRACT_SCHEMA_VERSION = 1
_CAMPAIGN_PIPELINE_PATHS = {
    f"gate_detection/training/{name}" for name in REQUIRED_PIPELINE_FILES
}
_CHECKPOINT_STATE_FIELDS = {
    "optimizer",
    "scheduler",
    "scaler",
    "rng",
    "epoch",
    "history",
    "split_manifest_hash",
    "best_candidate",
}
_RESUME_DRIFT_FIELDS = {"dataset", "config", "code"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_loads(text: str) -> Any:
    """Decode evidence JSON without duplicate keys or non-finite numbers."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number: {value}")
        return parsed

    return json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def _yaml_scalars(path: Path) -> dict[str, Any]:
    """Read the simple top-level scalars needed from Ultralytics args.yaml."""
    values: dict[str, Any] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([A-Za-z0-9_/-]+):\s*(.*?)\s*$", line)
        if not match:
            continue
        key, raw = match.groups()
        if raw in {"null", "None", "~", ""}:
            value: Any = None
        elif raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw.strip("'\"")
        values[key] = value
    return values


def _results_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "rows": 0}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"available": True, "rows": 0, "valid": False}
    pose_key = "metrics/mAP50-95(P)"
    parsed: list[dict[str, float]] = []
    for row in rows:
        try:
            numeric = {
                "epoch": float(row["epoch"]),
                "time": float(row["time"]),
                "pose_map50_95": float(row[pose_key]),
            }
        except (KeyError, TypeError, ValueError):
            return {
                "available": True,
                "rows": len(rows),
                "valid": False,
                "failure": "missing_or_non_numeric_required_column",
            }
        if not all(math.isfinite(value) for value in numeric.values()):
            return {
                "available": True,
                "rows": len(rows),
                "valid": False,
                "failure": "non_finite_metric",
            }
        parsed.append(numeric)
    by_epoch = {int(row["epoch"]): row for row in parsed}
    best = max(parsed, key=lambda row: row["pose_map50_95"])
    return {
        "available": True,
        "valid": True,
        "rows": len(rows),
        "last_epoch": int(parsed[-1]["epoch"]),
        "elapsed_seconds": parsed[-1]["time"],
        "pose_map50_95": {
            "epoch_10": by_epoch.get(10, {}).get("pose_map50_95"),
            "epoch_25": by_epoch.get(25, {}).get("pose_map50_95"),
            "final": parsed[-1]["pose_map50_95"],
            "best": best["pose_map50_95"],
            "best_epoch": int(best["epoch"]),
        },
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in REQUIRED_TRAINING_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _expected_package_versions(repo_root: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    lock = repo_root / "requirements" / "optional-training.txt"
    if not lock.exists():
        return expected
    for line in lock.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-r ")) or "==" not in stripped:
            continue
        name, version = stripped.split("==", 1)
        expected[name.strip().lower()] = version.strip()
    return expected


def _dataset_tree_digest(root: Path) -> tuple[str, int, int]:
    """Hash a dataset tree using stable relative paths, sizes, and bytes."""

    if not root.is_dir():
        raise ValueError("dataset root is not a directory")
    digest = hashlib.sha256(b"aigp-dataset-tree-v1\0")
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"dataset symlink is not content-stable: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported dataset entry: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        file_count += 1
        total_bytes += size
    if file_count == 0:
        raise ValueError("dataset tree is empty")
    return digest.hexdigest(), file_count, total_bytes


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _manifest_schema_contract_valid(manifest: Any) -> bool:
    root_keys = {
        "schema_version",
        "dataset_id",
        "created_at",
        "provenance",
        "content_sha256",
        "label_schema",
        "groups",
        "split",
    }
    if not isinstance(manifest, dict) or set(manifest) != root_keys:
        return False
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        return False
    if not _nonempty_string(manifest["dataset_id"]):
        return False
    if not _nonempty_string(manifest["created_at"]):
        return False
    try:
        timestamp = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    except ValueError:
        return False
    if timestamp.tzinfo is None:
        return False
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "source", "license_or_permission", "acquisition_notes"
    }:
        return False
    if not all(_nonempty_string(value) for value in provenance.values()):
        return False
    content_hash = manifest["content_sha256"]
    if not isinstance(content_hash, str) or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
        return False
    label_schema = manifest["label_schema"]
    if not isinstance(label_schema, dict) or set(label_schema) != {
        "task", "keypoint_order"
    }:
        return False
    keypoints = label_schema["keypoint_order"]
    if (
        label_schema["task"] != "pose"
        or not isinstance(keypoints, list)
        or len(keypoints) < 4
        or not all(_nonempty_string(value) for value in keypoints)
        or len(set(keypoints)) != len(keypoints)
    ):
        return False
    groups = manifest["groups"]
    if not isinstance(groups, list) or len(groups) < 2:
        return False
    group_ids: list[str] = []
    session_by_group: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, dict) or set(group) != {
            "group_id", "session_or_track", "sample_count"
        }:
            return False
        if not _nonempty_string(group["group_id"]) or not _nonempty_string(
            group["session_or_track"]
        ):
            return False
        if type(group["sample_count"]) is not int or group["sample_count"] < 1:
            return False
        group_ids.append(group["group_id"])
        session_by_group[group["group_id"]] = group["session_or_track"]
    if len(set(group_ids)) != len(group_ids):
        return False
    split = manifest["split"]
    if not isinstance(split, dict) or set(split) != {
        "method", "train_groups", "validation_groups", "test_groups"
    }:
        return False
    if split["method"] != "grouped_by_session_or_track":
        return False
    partitions: list[set[str]] = []
    for name in ("train_groups", "validation_groups", "test_groups"):
        values = split[name]
        if (
            not isinstance(values, list)
            or not values
            or not all(_nonempty_string(value) for value in values)
            or len(set(values)) != len(values)
        ):
            return False
        partitions.append(set(values))
    train, validation, test = partitions
    session_partitions = [
        {session_by_group[group_id] for group_id in partition}
        for partition in partitions
    ]
    train_sessions, validation_sessions, test_sessions = session_partitions
    return (
        not (train & validation or train & test or validation & test)
        and not (
            train_sessions & validation_sessions
            or train_sessions & test_sessions
            or validation_sessions & test_sessions
        )
        and train | validation | test == set(group_ids)
    )


def _dataset_manifest_status(
    path: Path, dataset_root: Path | None = None
) -> dict[str, Any]:
    dataset_root = dataset_root or path.parent / "dataset"
    status: dict[str, Any] = {
        "available": path.exists(),
        "schema_valid": False,
        "content_digest_verified": False,
        "valid": False,
        "dataset_root": str(dataset_root),
    }
    if not path.exists():
        return status
    try:
        manifest = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["failure"] = f"{type(exc).__name__}: {exc}"
        return status
    schema_valid = _manifest_schema_contract_valid(manifest)
    status["schema_valid"] = schema_valid
    if not schema_valid:
        status["failure"] = "manifest does not satisfy grouped provenance contract"
        return status
    try:
        computed, file_count, total_bytes = _dataset_tree_digest(dataset_root)
    except (OSError, ValueError) as exc:
        status["failure"] = f"dataset content unavailable: {exc}"
        return status
    status["computed_content_sha256"] = computed
    status["dataset_file_count"] = file_count
    status["dataset_size_bytes"] = total_bytes
    status["content_digest_verified"] = computed == manifest["content_sha256"]
    status["valid"] = status["content_digest_verified"] is True
    if not status["content_digest_verified"]:
        status["failure"] = "dataset content digest does not match manifest"
    return status


def _artifact_reference_contract_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        return False
    path = value["path"]
    digest = value["sha256"]
    if not _nonempty_string(path) or "\\" in path:
        return False
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or path != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return False
    return isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None


def _verified_artifact_path(
    reference: dict[str, str], repo_root: Path
) -> Path:
    """Resolve and hash one repo-relative, regular, no-symlink artifact."""

    relative = PurePosixPath(reference["path"])
    candidate = repo_root.joinpath(*relative.parts)
    cursor = repo_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {reference['path']}")
    if not candidate.is_file():
        raise ValueError(f"artifact is absent or not a regular file: {reference['path']}")
    if _sha256(candidate) != reference["sha256"]:
        raise ValueError(f"artifact digest mismatch: {reference['path']}")
    return candidate


def _transitive_lock_valid(path: Path) -> tuple[bool, str | None]:
    """Require a self-contained exact pip lock with SHA-256 wheel hashes."""

    try:
        physical_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return False, f"lock unreadable: {exc}"
    logical_lines: list[str] = []
    pending = ""
    for physical in physical_lines:
        stripped = physical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        return False, "lock ends with an unterminated continuation"
    if not logical_lines:
        return False, "lock contains no resolved distributions"

    requirement_pattern = re.compile(
        r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s;]+)"
        r"(?P<hashes>(?:\s+--hash=sha256:[0-9a-f]{64})+)$"
    )
    packages: set[str] = set()
    for line in logical_lines:
        match = requirement_pattern.fullmatch(line)
        if match is None:
            return False, f"lock entry is not an exact hashed requirement: {line}"
        packages.add(re.sub(r"[-_.]+", "-", match.group("name")).lower())
    required = {
        re.sub(r"[-_.]+", "-", name).lower()
        for name in REQUIRED_TRAINING_PACKAGES
    }
    missing = sorted(required - packages)
    if missing:
        return False, "lock omits required training packages: " + ",".join(missing)
    return True, None


def _campaign_contract_structure_valid(contract: Any) -> bool:
    root_keys = {
        "schema_version",
        "contract_id",
        "created_at",
        "dataset_manifest_sha256",
        "pipeline",
        "budgets",
        "checkpointing",
        "evaluation",
        "runtime",
    }
    if not isinstance(contract, dict) or set(contract) != root_keys:
        return False
    if type(contract["schema_version"]) is not int or contract[
        "schema_version"
    ] != CAMPAIGN_CONTRACT_SCHEMA_VERSION:
        return False
    if not _nonempty_string(contract["contract_id"]) or not _nonempty_string(
        contract["created_at"]
    ):
        return False
    try:
        created_at = datetime.fromisoformat(
            contract["created_at"].replace("Z", "+00:00")
        )
    except ValueError:
        return False
    if created_at.tzinfo is None:
        return False
    if not isinstance(contract["dataset_manifest_sha256"], str) or re.fullmatch(
        r"[0-9a-f]{64}", contract["dataset_manifest_sha256"]
    ) is None:
        return False

    pipeline = contract["pipeline"]
    if not isinstance(pipeline, dict) or set(pipeline) != {
        "source_files",
        "resolved_config",
        "transitive_lock",
        "lock_kind",
    }:
        return False
    sources = pipeline["source_files"]
    if (
        not isinstance(sources, list)
        or len(sources) != len(REQUIRED_PIPELINE_FILES)
        or not all(_artifact_reference_contract_valid(value) for value in sources)
        or {value["path"] for value in sources} != _CAMPAIGN_PIPELINE_PATHS
        or pipeline["lock_kind"] != "fully_transitive_exact_with_sha256_hashes"
        or not _artifact_reference_contract_valid(pipeline["resolved_config"])
        or not _artifact_reference_contract_valid(pipeline["transitive_lock"])
    ):
        return False

    budgets = contract["budgets"]
    if not isinstance(budgets, dict) or set(budgets) != {
        "smoke_epochs", "successive_halving_epochs", "evidence"
    }:
        return False
    smoke_epochs = budgets["smoke_epochs"]
    halving = budgets["successive_halving_epochs"]
    if (
        type(smoke_epochs) is not int
        or smoke_epochs not in {1, 2}
        or not isinstance(halving, list)
        or len(halving) < 2
        or any(type(value) is not int or value < 1 for value in halving)
        or halving[0] != smoke_epochs
        or any(first >= second for first, second in zip(halving, halving[1:]))
        or not _artifact_reference_contract_valid(budgets["evidence"])
    ):
        return False

    checkpointing = contract["checkpointing"]
    if not isinstance(checkpointing, dict) or set(checkpointing) != {
        "atomic_periodic",
        "state_fields",
        "resume_rejects_drift",
        "evidence",
    }:
        return False
    if (
        checkpointing["atomic_periodic"] is not True
        or not isinstance(checkpointing["state_fields"], list)
        or set(checkpointing["state_fields"]) != _CHECKPOINT_STATE_FIELDS
        or len(checkpointing["state_fields"]) != len(_CHECKPOINT_STATE_FIELDS)
        or not isinstance(checkpointing["resume_rejects_drift"], list)
        or set(checkpointing["resume_rejects_drift"]) != _RESUME_DRIFT_FIELDS
        or len(checkpointing["resume_rejects_drift"]) != len(_RESUME_DRIFT_FIELDS)
        or not _artifact_reference_contract_valid(checkpointing["evidence"])
    ):
        return False

    evaluation = contract["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "split_method",
        "grouped_holdout",
        "classical_baseline",
        "onnx_comparison",
        "tensorrt_comparison",
    }:
        return False
    if evaluation["split_method"] != "grouped_by_session_or_track" or not all(
        _artifact_reference_contract_valid(evaluation[name])
        for name in (
            "grouped_holdout",
            "classical_baseline",
            "onnx_comparison",
            "tensorrt_comparison",
        )
    ):
        return False

    runtime = contract["runtime"]
    return (
        isinstance(runtime, dict)
        and set(runtime) == {"integration", "replay_evidence"}
        and _artifact_reference_contract_valid(runtime["integration"])
        and _artifact_reference_contract_valid(runtime["replay_evidence"])
    )


def _campaign_contract_status(
    path: Path,
    repo_root: Path,
    dataset_manifest: Path,
    runtime_references: list[str],
) -> dict[str, Any]:
    """Verify the exact, content-bound contract required before training."""

    status: dict[str, Any] = {
        "path": str(path),
        "available": path.exists(),
        "schema_valid": False,
        "artifacts_verified": False,
        "valid": False,
        "verified_artifacts": [],
    }
    if not path.exists():
        status["failure"] = "campaign contract is absent"
        return status
    try:
        contract = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["failure"] = f"{type(exc).__name__}: {exc}"
        return status
    if not _campaign_contract_structure_valid(contract):
        status["failure"] = "campaign contract does not satisfy the exact schema"
        return status
    status["schema_valid"] = True
    status["contract_id"] = contract["contract_id"]
    status["created_at"] = contract["created_at"]

    try:
        if not dataset_manifest.is_file() or _sha256(dataset_manifest) != contract[
            "dataset_manifest_sha256"
        ]:
            raise ValueError("campaign contract dataset-manifest digest mismatch")
        references: list[tuple[str, dict[str, str]]] = []
        references.extend(
            (f"pipeline.source_files[{index}]", reference)
            for index, reference in enumerate(contract["pipeline"]["source_files"])
        )
        references.extend(
            (
                ("pipeline.resolved_config", contract["pipeline"]["resolved_config"]),
                ("pipeline.transitive_lock", contract["pipeline"]["transitive_lock"]),
                ("budgets.evidence", contract["budgets"]["evidence"]),
                ("checkpointing.evidence", contract["checkpointing"]["evidence"]),
                ("evaluation.grouped_holdout", contract["evaluation"]["grouped_holdout"]),
                ("evaluation.classical_baseline", contract["evaluation"]["classical_baseline"]),
                ("evaluation.onnx_comparison", contract["evaluation"]["onnx_comparison"]),
                ("evaluation.tensorrt_comparison", contract["evaluation"]["tensorrt_comparison"]),
                ("runtime.integration", contract["runtime"]["integration"]),
                ("runtime.replay_evidence", contract["runtime"]["replay_evidence"]),
            )
        )
        verified: list[dict[str, str]] = []
        verified_paths: dict[str, Path] = {}
        for label, reference in references:
            verified_paths[label] = _verified_artifact_path(reference, repo_root)
            verified.append(
                {"gate": label, "path": reference["path"], "sha256": reference["sha256"]}
            )
        lock_valid, lock_failure = _transitive_lock_valid(
            verified_paths["pipeline.transitive_lock"]
        )
        if not lock_valid:
            raise ValueError(lock_failure or "transitive lock is invalid")
        integration_path = contract["runtime"]["integration"]["path"]
        if integration_path not in runtime_references:
            raise ValueError(
                "runtime integration artifact is not detected in the VQ2 runtime"
            )
    except (OSError, ValueError) as exc:
        status["failure"] = str(exc)
        return status

    status["verified_artifacts"] = verified
    status["artifacts_verified"] = True
    status["valid"] = True
    return status


def _runtime_weight_references(repo_root: Path) -> list[str]:
    needles = ("gate_pose_v1", "weights/best.pt", "weights\\best.pt")
    references: list[str] = []
    candidates: list[Path] = []
    for root_name in ("competition", "control", "gate_detection"):
        root = repo_root / root_name
        if root.exists():
            candidates.extend(root.rglob("*.py"))
    runner = repo_root / "scripts" / "aigp_vq2_run.py"
    if runner.exists():
        candidates.append(runner)
    for path in sorted(set(candidates)):
        if "training" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(needle in text for needle in needles):
            references.append(path.relative_to(repo_root).as_posix())
    return references


def _smoke_archive(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "valid_zip": False}
    if not zipfile.is_zipfile(path):
        return {"available": True, "valid_zip": False}
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            members = len(archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        return {
            "available": True,
            "valid_zip": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "valid_zip": bad_member is None,
        "members": members,
        "bad_member": bad_member,
    }


def audit_experiment(
    run_dir: Path,
    repo_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Return evidence and blockers without importing Torch or loading pickle."""
    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root).resolve()
    training_root = repo_root / "gate_detection" / "training"
    args = _yaml_scalars(run_dir / "args.yaml")
    configured_data = args.get("data")
    configured_path = Path(str(configured_data)) if configured_data else None
    configured_absolute = bool(
        configured_data
        and (
            configured_path.is_absolute()
            or PurePosixPath(str(configured_data)).is_absolute()
            or PureWindowsPath(str(configured_data)).is_absolute()
        )
    )
    if configured_path is None:
        configured_resolved_path = None
    elif configured_path.is_absolute():
        configured_resolved_path = configured_path.resolve()
    elif configured_absolute:
        # A foreign absolute path (for example the historical POSIX path on a
        # Windows audit host) has no meaningful native resolution.
        configured_resolved_path = None
    else:
        configured_resolved_path = (run_dir / configured_path).resolve()
    local_dataset = training_root / "dataset"
    dataset_manifest = training_root / "dataset_manifest.json"
    manifest_status = _dataset_manifest_status(dataset_manifest)
    pipeline = {
        name: (training_root / name).exists() for name in REQUIRED_PIPELINE_FILES
    }
    weights: dict[str, dict[str, Any]] = {}
    for name in ("best.pt", "last.pt"):
        path = run_dir / "weights" / name
        weights[name] = {
            "available": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256(path) if path.exists() else None,
        }
        if smoke:
            weights[name]["archive_smoke"] = _smoke_archive(path)
    hashes = [entry["sha256"] for entry in weights.values() if entry["sha256"]]
    results = _results_summary(run_dir / "results.csv")
    packages = _package_versions()
    expected_packages = _expected_package_versions(repo_root)
    runtime_references = _runtime_weight_references(repo_root)

    basic_blockers: list[str] = []
    missing_pipeline = [name for name, exists in pipeline.items() if not exists]
    if missing_pipeline:
        basic_blockers.append("missing_pipeline:" + ",".join(missing_pipeline))
    if not local_dataset.exists():
        basic_blockers.append("dataset_absent")
    if not manifest_status["available"]:
        basic_blockers.append("dataset_manifest_absent")
    elif not manifest_status["schema_valid"]:
        basic_blockers.append("dataset_manifest_invalid")
    elif manifest_status["content_digest_verified"] is not True:
        basic_blockers.append("dataset_content_digest_unverified")
    configured_exists = bool(
        configured_resolved_path and configured_resolved_path.exists()
    )
    expected_dataset_targets = {
        local_dataset.resolve(),
        (local_dataset / "data.yaml").resolve(),
    }
    configured_matches_verified_dataset = bool(
        configured_resolved_path
        and configured_resolved_path in expected_dataset_targets
    )
    if not configured_exists:
        basic_blockers.append("configured_dataset_path_unavailable")
    elif not configured_matches_verified_dataset:
        basic_blockers.append("configured_dataset_not_content_verified")
    missing_packages = [name for name, version in packages.items() if version is None]
    if missing_packages:
        basic_blockers.append(
            "training_dependencies_unavailable:" + ",".join(missing_packages)
        )
    version_mismatches = [
        name
        for name, expected in expected_packages.items()
        if packages.get(name) is not None and packages.get(name) != expected
    ]
    if version_mismatches:
        basic_blockers.append(
            "training_dependency_version_mismatch:" + ",".join(version_mismatches)
        )
    if not results.get("valid"):
        basic_blockers.append("historical_results_invalid_or_absent")
    if smoke and any(
        not entry.get("archive_smoke", {}).get("valid_zip")
        for entry in weights.values()
    ):
        basic_blockers.append("weight_archive_smoke_failed")

    campaign_contract_path = training_root / CAMPAIGN_CONTRACT_NAME
    campaign_contract = _campaign_contract_status(
        campaign_contract_path,
        repo_root,
        dataset_manifest,
        runtime_references,
    )
    blockers = list(basic_blockers)
    if not campaign_contract["available"]:
        blockers.append("campaign_contract_absent")
    elif not campaign_contract["schema_valid"]:
        blockers.append("campaign_contract_invalid")
    elif campaign_contract["valid"] is not True:
        blockers.append("campaign_contract_evidence_unverified")
    basic_prerequisites_present = not basic_blockers
    ready_for_training = (
        basic_prerequisites_present and campaign_contract["valid"] is True
    )
    last_archive_valid = (
        smoke
        and weights["last.pt"].get("archive_smoke", {}).get("valid_zip") is True
    )

    return {
        "schema_version": 2,
        "audit_kind": "historical_yolo_pose_read_only",
        "run_dir": str(run_dir),
        "basic_prerequisites_present": basic_prerequisites_present,
        "ready_for_training": ready_for_training,
        "resume_ready": (
            ready_for_training
            and bool(weights["last.pt"]["available"])
            and last_archive_valid
        ),
        "basic_blockers": basic_blockers,
        "blockers": blockers,
        "historical_args": {
            key: args.get(key)
            for key in ("task", "model", "data", "epochs", "device", "seed", "resume")
        },
        "configured_dataset_path": {
            "value": str(configured_data) if configured_data is not None else None,
            "absolute": configured_absolute,
            "resolved_value": (
                str(configured_resolved_path)
                if configured_resolved_path is not None
                else None
            ),
            "exists_on_this_host": configured_exists,
            "matches_verified_dataset": configured_matches_verified_dataset,
        },
        "local_dataset": {
            "path": str(local_dataset),
            "available": local_dataset.exists(),
            "manifest_path": str(dataset_manifest),
            "manifest": manifest_status,
        },
        "pipeline_files": pipeline,
        "package_versions": packages,
        "expected_package_versions": expected_packages,
        "weights": weights,
        "weights_byte_identical": len(hashes) == 2 and len(set(hashes)) == 1,
        "results": results,
        "runtime_weight_references": runtime_references,
        "campaign_contract": campaign_contract,
        "smoke_requested": smoke,
        "safety_note": (
            "No weight was deserialized and no training/export command was run."
        ),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Read-only audit of historical YOLO pose artifacts",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            repo_root
            / "gate_detection"
            / "training"
            / "runs"
            / "gate_pose_v1"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CRC-check checkpoint archives; never loads model pickle or runs inference.",
    )
    parser.add_argument(
        "--strict-ready",
        action="store_true",
        help="Exit 2 until basic prerequisites and the exact campaign contract verify.",
    )
    args = parser.parse_args()
    report = audit_experiment(args.run_dir, repo_root, smoke=args.smoke)
    print(json.dumps(report, indent=2))
    return 2 if args.strict_ready and not report["ready_for_training"] else 0


if __name__ == "__main__":
    sys.exit(main())
