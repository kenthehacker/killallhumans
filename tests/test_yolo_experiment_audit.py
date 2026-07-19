from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import audit_yolo_experiment as yolo_audit
from scripts.audit_yolo_experiment import (
    _campaign_contract_status,
    _dataset_manifest_status,
    _dataset_tree_digest,
    audit_experiment,
)


_REPO = Path(__file__).resolve().parent.parent
_RUN = _REPO / "gate_detection" / "training" / "runs" / "gate_pose_v1"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(repo: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _file_sha256(repo / relative)}


def _write_basic_candidate(repo: Path, monkeypatch) -> tuple[Path, Path]:
    training = repo / "gate_detection" / "training"
    dataset = training / "dataset"
    run = training / "runs" / "candidate"
    dataset.mkdir(parents=True)
    run.mkdir(parents=True)
    (dataset / "data.yaml").write_text("path: .\n", encoding="utf-8")
    (dataset / "sample.bin").write_bytes(b"sample")
    content_sha256, _, _ = _dataset_tree_digest(dataset)
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "created_at": "2026-07-18T00:00:00Z",
        "provenance": {
            "source": "fixture",
            "license_or_permission": "fixture permission",
            "acquisition_notes": "fixture only",
        },
        "content_sha256": content_sha256,
        "label_schema": {"task": "pose", "keypoint_order": ["a", "b", "c", "d"]},
        "groups": [
            {"group_id": "a", "session_or_track": "a", "sample_count": 1},
            {"group_id": "b", "session_or_track": "b", "sample_count": 1},
            {"group_id": "c", "session_or_track": "c", "sample_count": 1},
        ],
        "split": {
            "method": "grouped_by_session_or_track",
            "train_groups": ["a"],
            "validation_groups": ["b"],
            "test_groups": ["c"],
        },
    }
    (training / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for name in yolo_audit.REQUIRED_PIPELINE_FILES:
        (training / name).write_text(f"# {name} fixture\n", encoding="utf-8")
    (run / "args.yaml").write_text(
        "data: ../../dataset/data.yaml\nresume: false\n", encoding="utf-8"
    )
    (run / "results.csv").write_text(
        "epoch,time,metrics/mAP50-95(P)\n0,1.0,0.1\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        yolo_audit,
        "_package_versions",
        lambda: {
            name: "fixture" for name in yolo_audit.REQUIRED_TRAINING_PACKAGES
        },
    )
    return training, run


def _write_campaign_contract(repo: Path, training: Path) -> Path:
    evidence = training / "evidence"
    evidence.mkdir()
    evidence_files = {
        "resolved_config": "gate_detection/training/evidence/resolved-config.json",
        "budgets": "gate_detection/training/evidence/budgets.json",
        "checkpoint": "gate_detection/training/evidence/checkpoint-contract.json",
        "holdout": "gate_detection/training/evidence/grouped-holdout.json",
        "classical": "gate_detection/training/evidence/classical-baseline.json",
        "onnx": "gate_detection/training/evidence/onnx-comparison.json",
        "tensorrt": "gate_detection/training/evidence/tensorrt-comparison.json",
        "replay": "gate_detection/training/evidence/vq2-replay.json",
    }
    for label, relative in evidence_files.items():
        (repo / relative).write_text(
            json.dumps({"fixture_evidence": label}), encoding="utf-8"
        )
    runtime_relative = "gate_detection/yolo_pose_runtime.py"
    (repo / runtime_relative).write_text(
        "WEIGHTS = 'gate_pose_v1/weights/best.pt'\n", encoding="utf-8"
    )
    lock_relative = "requirements/optional-training.full.lock.txt"
    lock = repo / lock_relative
    lock.parent.mkdir(exist_ok=True)
    lock.write_text(
        "\n".join(
            f"{name}==1.0 --hash=sha256:{index:064x}"
            for index, name in enumerate(
                yolo_audit.REQUIRED_TRAINING_PACKAGES, start=1
            )
        )
        + "\n",
        encoding="utf-8",
    )
    contract = {
        "schema_version": 1,
        "contract_id": "fixture-campaign",
        "created_at": "2026-07-18T00:00:00Z",
        "dataset_manifest_sha256": _file_sha256(
            training / "dataset_manifest.json"
        ),
        "pipeline": {
            "source_files": [
                _artifact(repo, f"gate_detection/training/{name}")
                for name in yolo_audit.REQUIRED_PIPELINE_FILES
            ],
            "resolved_config": _artifact(repo, evidence_files["resolved_config"]),
            "transitive_lock": _artifact(repo, lock_relative),
            "lock_kind": "fully_transitive_exact_with_sha256_hashes",
        },
        "budgets": {
            "smoke_epochs": 2,
            "successive_halving_epochs": [2, 10, 25],
            "evidence": _artifact(repo, evidence_files["budgets"]),
        },
        "checkpointing": {
            "atomic_periodic": True,
            "state_fields": [
                "optimizer",
                "scheduler",
                "scaler",
                "rng",
                "epoch",
                "history",
                "split_manifest_hash",
                "best_candidate",
            ],
            "resume_rejects_drift": ["dataset", "config", "code"],
            "evidence": _artifact(repo, evidence_files["checkpoint"]),
        },
        "evaluation": {
            "split_method": "grouped_by_session_or_track",
            "grouped_holdout": _artifact(repo, evidence_files["holdout"]),
            "classical_baseline": _artifact(repo, evidence_files["classical"]),
            "onnx_comparison": _artifact(repo, evidence_files["onnx"]),
            "tensorrt_comparison": _artifact(repo, evidence_files["tensorrt"]),
        },
        "runtime": {
            "integration": _artifact(repo, runtime_relative),
            "replay_evidence": _artifact(repo, evidence_files["replay"]),
        },
    }
    path = training / yolo_audit.CAMPAIGN_CONTRACT_NAME
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def test_historical_yolo_run_is_audited_without_claiming_reproducibility() -> None:
    before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (_RUN / "weights").glob("*.pt")
    }
    report = audit_experiment(_RUN, _REPO, smoke=True)
    after = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (_RUN / "weights").glob("*.pt")
    }

    assert before == after
    assert report["ready_for_training"] is False
    assert report["resume_ready"] is False
    assert report["basic_prerequisites_present"] is False
    assert "campaign_contract_absent" in report["blockers"]
    assert report["campaign_contract"]["valid"] is False
    assert "dataset_absent" in report["blockers"]
    assert "dataset_manifest_absent" in report["blockers"]
    assert any(item.startswith("missing_pipeline:") for item in report["blockers"])
    assert report["configured_dataset_path"] == {
        "value": (
            "/Users/kenichimatsuo/Projects/killallhumans/"
            "gate_detection/training/dataset/data.yaml"
        ),
        "absolute": True,
        "resolved_value": None,
        "exists_on_this_host": False,
        "matches_verified_dataset": False,
    }
    assert report["results"]["last_epoch"] == 100
    assert report["results"]["elapsed_seconds"] == 62629.7
    assert report["results"]["pose_map50_95"]["epoch_10"] is not None
    assert report["results"]["pose_map50_95"]["epoch_25"] is not None
    # The handoff suspected duplicate bytes because sizes match. Hashing proves
    # the two historical checkpoints are actually distinct archives.
    assert report["weights_byte_identical"] is False
    for weight in report["weights"].values():
        assert weight["archive_smoke"]["valid_zip"] is True
    assert report["runtime_weight_references"] == []
    assert "No weight was deserialized" in report["safety_note"]


def test_dataset_manifest_schema_requires_grouped_holdouts_and_provenance() -> None:
    schema_path = (
        _REPO
        / "gate_detection"
        / "training"
        / "dataset-manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert {
        "provenance", "content_sha256", "groups", "split",
    } <= set(schema["required"])
    split = schema["properties"]["split"]
    assert split["properties"]["method"]["const"] == (
        "grouped_by_session_or_track"
    )
    assert {"train_groups", "validation_groups", "test_groups"} <= set(
        split["required"]
    )


def test_campaign_contract_schema_covers_every_readiness_gate() -> None:
    schema_path = (
        _REPO
        / "gate_detection"
        / "training"
        / "campaign-contract.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert {
        "dataset_manifest_sha256",
        "pipeline",
        "budgets",
        "checkpointing",
        "evaluation",
        "runtime",
    } <= set(schema["required"])
    assert schema["properties"]["pipeline"]["properties"]["lock_kind"][
        "const"
    ] == "fully_transitive_exact_with_sha256_hashes"
    assert schema["properties"]["evaluation"]["properties"]["split_method"][
        "const"
    ] == "grouped_by_session_or_track"
    assert {
        "grouped_holdout",
        "classical_baseline",
        "onnx_comparison",
        "tensorrt_comparison",
    } <= set(schema["properties"]["evaluation"]["required"])
    assert {"integration", "replay_evidence"} <= set(
        schema["properties"]["runtime"]["required"]
    )


def test_campaign_contract_verifies_all_artifacts_and_detects_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    training, _ = _write_basic_candidate(repo, monkeypatch)
    contract_path = _write_campaign_contract(repo, training)
    runtime_reference = "gate_detection/yolo_pose_runtime.py"

    status = _campaign_contract_status(
        contract_path,
        repo,
        training / "dataset_manifest.json",
        [runtime_reference],
    )
    assert status["schema_valid"] is True
    assert status["artifacts_verified"] is True
    assert status["valid"] is True
    assert len(status["verified_artifacts"]) == 14

    (training / "train.py").write_text("# tampered\n", encoding="utf-8")
    status = _campaign_contract_status(
        contract_path,
        repo,
        training / "dataset_manifest.json",
        [runtime_reference],
    )
    assert status["schema_valid"] is True
    assert status["artifacts_verified"] is False
    assert status["valid"] is False
    assert "digest mismatch" in status["failure"]


def test_basic_yolo_scaffold_without_campaign_contract_is_never_ready(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    _, run = _write_basic_candidate(repo, monkeypatch)

    report = audit_experiment(run, repo)

    assert report["basic_prerequisites_present"] is True
    assert report["basic_blockers"] == []
    assert report["ready_for_training"] is False
    assert report["resume_ready"] is False
    assert report["blockers"] == ["campaign_contract_absent"]


def test_exact_content_bound_campaign_contract_can_unlock_training_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    training, run = _write_basic_candidate(repo, monkeypatch)
    _write_campaign_contract(repo, training)

    report = audit_experiment(run, repo)

    assert report["basic_prerequisites_present"] is True
    assert report["campaign_contract"]["valid"] is True
    assert report["ready_for_training"] is True
    # No resumable checkpoint exists in this fixture, so training readiness
    # cannot be misreported as checkpoint-resume readiness.
    assert report["resume_ready"] is False
    assert report["blockers"] == []


def test_dataset_manifest_audit_rejects_group_leakage(tmp_path: Path) -> None:
    path = tmp_path / "dataset_manifest.json"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "labels.txt").write_text("fixture labels\n", encoding="utf-8")
    content_sha256, _, _ = _dataset_tree_digest(dataset)
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "created_at": "2026-07-18T00:00:00Z",
        "provenance": {
            "source": "fixture",
            "license_or_permission": "fixture permission",
            "acquisition_notes": "fixture only",
        },
        "content_sha256": content_sha256,
        "label_schema": {"task": "pose", "keypoint_order": ["a", "b", "c", "d"]},
        "groups": [
            {"group_id": "a", "session_or_track": "a", "sample_count": 1},
            {"group_id": "b", "session_or_track": "b", "sample_count": 1},
            {"group_id": "c", "session_or_track": "c", "sample_count": 1},
        ],
        "split": {
            "method": "grouped_by_session_or_track",
            "train_groups": ["a"],
            "validation_groups": ["b"],
            "test_groups": ["c"],
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _dataset_manifest_status(path)["valid"] is True
    manifest["split"]["validation_groups"] = ["a"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _dataset_manifest_status(path)["valid"] is False


def test_dataset_manifest_schema_only_is_not_training_ready(tmp_path: Path) -> None:
    path = tmp_path / "dataset_manifest.json"
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "created_at": "2026-07-18T00:00:00Z",
        "provenance": {
            "source": "fixture",
            "license_or_permission": "fixture permission",
            "acquisition_notes": "fixture only",
        },
        "content_sha256": "a" * 64,
        "label_schema": {"task": "pose", "keypoint_order": ["a", "b", "c", "d"]},
        "groups": [
            {"group_id": "a", "session_or_track": "a", "sample_count": 1},
            {"group_id": "b", "session_or_track": "b", "sample_count": 1},
            {"group_id": "c", "session_or_track": "c", "sample_count": 1},
        ],
        "split": {
            "method": "grouped_by_session_or_track",
            "train_groups": ["a"],
            "validation_groups": ["b"],
            "test_groups": ["c"],
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")

    status = _dataset_manifest_status(path)
    assert status["schema_valid"] is True
    assert status["content_digest_verified"] is False
    assert status["valid"] is False

    manifest["unexpected"] = "old partial checker accepted unknown fields"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    status = _dataset_manifest_status(path)
    assert status["schema_valid"] is False
    assert status["valid"] is False


def test_dataset_manifest_detects_content_tampering(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    sample = dataset / "sample.bin"
    sample.write_bytes(b"first")
    content_sha256, _, _ = _dataset_tree_digest(dataset)
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "created_at": "2026-07-18T00:00:00+00:00",
        "provenance": {
            "source": "fixture",
            "license_or_permission": "fixture permission",
            "acquisition_notes": "fixture only",
        },
        "content_sha256": content_sha256,
        "label_schema": {"task": "pose", "keypoint_order": ["a", "b", "c", "d"]},
        "groups": [
            {"group_id": "a", "session_or_track": "a", "sample_count": 1},
            {"group_id": "b", "session_or_track": "b", "sample_count": 1},
            {"group_id": "c", "session_or_track": "c", "sample_count": 1},
        ],
        "split": {
            "method": "grouped_by_session_or_track",
            "train_groups": ["a"],
            "validation_groups": ["b"],
            "test_groups": ["c"],
        },
    }
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _dataset_manifest_status(path)["content_digest_verified"] is True

    sample.write_bytes(b"tampered")
    status = _dataset_manifest_status(path)
    assert status["schema_valid"] is True
    assert status["content_digest_verified"] is False
    assert status["valid"] is False


def test_dataset_manifest_rejects_same_session_across_holdouts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sample.bin").write_bytes(b"sample")
    content_sha256, _, _ = _dataset_tree_digest(dataset)
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "created_at": "2026-07-18T00:00:00Z",
        "provenance": {
            "source": "fixture",
            "license_or_permission": "fixture permission",
            "acquisition_notes": "fixture only",
        },
        "content_sha256": content_sha256,
        "label_schema": {"task": "pose", "keypoint_order": ["a", "b", "c", "d"]},
        "groups": [
            {"group_id": "a1", "session_or_track": "same-session", "sample_count": 1},
            {"group_id": "a2", "session_or_track": "same-session", "sample_count": 1},
            {"group_id": "c", "session_or_track": "other-session", "sample_count": 1},
        ],
        "split": {
            "method": "grouped_by_session_or_track",
            "train_groups": ["a1"],
            "validation_groups": ["a2"],
            "test_groups": ["c"],
        },
    }
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    status = _dataset_manifest_status(path)
    assert status["schema_valid"] is False
    assert status["valid"] is False


def test_audit_rejects_existing_but_unverified_configured_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    training = repo / "gate_detection" / "training"
    dataset = training / "dataset"
    run = training / "runs" / "candidate"
    dataset.mkdir(parents=True)
    run.mkdir(parents=True)
    (dataset / "data.yaml").write_text("path: .\n", encoding="utf-8")
    content_sha256, _, _ = _dataset_tree_digest(dataset)
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "created_at": "2026-07-18T00:00:00Z",
        "provenance": {
            "source": "fixture",
            "license_or_permission": "fixture permission",
            "acquisition_notes": "fixture only",
        },
        "content_sha256": content_sha256,
        "label_schema": {"task": "pose", "keypoint_order": ["a", "b", "c", "d"]},
        "groups": [
            {"group_id": "a", "session_or_track": "a", "sample_count": 1},
            {"group_id": "b", "session_or_track": "b", "sample_count": 1},
            {"group_id": "c", "session_or_track": "c", "sample_count": 1},
        ],
        "split": {
            "method": "grouped_by_session_or_track",
            "train_groups": ["a"],
            "validation_groups": ["b"],
            "test_groups": ["c"],
        },
    }
    (training / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for name in yolo_audit.REQUIRED_PIPELINE_FILES:
        (training / name).write_text("# fixture\n", encoding="utf-8")
    other = tmp_path / "other" / "data.yaml"
    other.parent.mkdir()
    other.write_text("path: unrelated\n", encoding="utf-8")
    (run / "args.yaml").write_text(
        f"data: {other.as_posix()}\nresume: false\n", encoding="utf-8"
    )
    (run / "results.csv").write_text(
        "epoch,time,metrics/mAP50-95(P)\n0,1.0,0.1\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        yolo_audit,
        "_package_versions",
        lambda: {name: "fixture" for name in yolo_audit.REQUIRED_TRAINING_PACKAGES},
    )

    report = audit_experiment(run, repo)
    assert "configured_dataset_not_content_verified" in report["blockers"]
    assert report["configured_dataset_path"]["exists_on_this_host"] is True
    assert report["configured_dataset_path"]["matches_verified_dataset"] is False
    assert report["ready_for_training"] is False
