from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control.learned_residual import (
    DEFAULT_N_INPUTS,
    TrackerResidualMLP,
)
from scripts import collect_residual_dataset as collector
from scripts import train_tracker_residual as trainer


_MISSING = object()


def _write_dataset(path: Path, samples: int = 160) -> None:
    rng = np.random.default_rng(123)
    session_id = np.repeat(np.arange(4, dtype=np.int64), samples // 4)
    track_id = np.repeat(np.arange(2, dtype=np.int64), samples // 2)
    features = rng.normal(size=(samples, DEFAULT_N_INPUTS))
    payload = {
        "features": features,
        "roll_nom": np.zeros(samples),
        "pitch_nom": np.zeros(samples),
        "thrust_nom": np.ones(samples),
        "pos_err": rng.normal(scale=0.2, size=(samples, 3)),
        "vel_err": rng.normal(scale=0.1, size=(samples, 3)),
        "pos": rng.normal(size=(samples, 3)),
        "vel": rng.normal(size=(samples, 3)),
        "yaw_des": rng.normal(scale=0.2, size=samples),
        "ref_pos": rng.normal(size=(samples, 3)),
        "ref_vel": rng.normal(size=(samples, 3)),
        "ref_accel": rng.normal(scale=0.3, size=(samples, 3)),
        "accel_des_baseline": rng.normal(scale=0.3, size=(samples, 3)),
        "version": np.asarray(2, dtype=np.int64),
        "track_id": track_id,
        "track_names": np.asarray(["track-a", "track-b"], dtype=np.str_),
        "session_id": session_id,
        "session_names": np.asarray(
            ["session-a", "session-b", "session-c", "session-d"],
            dtype=np.str_,
        ),
    }
    np.savez_compressed(path, **payload)


def _good_matrix(error: float = 1.0, improvement: float | None = None) -> dict:
    row = {
        "evaluation_mode": "completion",
        "evidence_valid": True,
        "duration_s": 45.0,
        "avg_tracking_error_m": error,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "crashed": False,
        "disqualified": False,
        "complete": True,
        "gates_passed": 4,
        "total_gates": 4,
        "completion": {
            "complete": True,
            "gates_passed": 4,
            "total_gates": 4,
        },
        "termination_reason": "race_complete",
        "skipped": False,
    }
    if improvement is not None:
        row.update(
            {
                "baseline_avg_tracking_error_m": 1.0,
                "baseline_gates_passed": 4,
                "improvement_pct": improvement,
            }
        )
    return {"track-a": row}


def test_grouped_split_never_leaks_adjacent_session_samples() -> None:
    groups = np.repeat(np.arange(5), 20)
    train_idx, val_idx = trainer._grouped_split(
        groups, 0.20, np.random.default_rng(7)
    )
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == set(range(5))
    for group in range(5):
        indices = set(np.flatnonzero(groups == group))
        assert indices <= set(train_idx) or indices <= set(val_idx)


@pytest.mark.parametrize(
    "mutation",
    [
        "nonfinite_features",
        "nonfinite_position_error",
        "wrong_feature_width",
        "wrong_reference_shape",
        "wrong_scalar_series_length",
    ],
)
def test_trainer_rejects_nonfinite_or_wrong_shape_dataset(
    tmp_path: Path, mutation: str
) -> None:
    dataset = tmp_path / "invalid-dataset.npz"
    _write_dataset(dataset)
    with np.load(dataset, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    if mutation == "nonfinite_features":
        payload["features"][0, 0] = np.nan
    elif mutation == "nonfinite_position_error":
        payload["pos_err"][0, 0] = np.inf
    elif mutation == "wrong_feature_width":
        payload["features"] = payload["features"][:, :-1]
    elif mutation == "wrong_reference_shape":
        payload["ref_accel"] = payload["ref_accel"][:, :2]
    elif mutation == "wrong_scalar_series_length":
        payload["yaw_des"] = payload["yaw_des"][:-1]
    np.savez_compressed(dataset, **payload)

    output = tmp_path / "must-not-exist.npz"
    checkpoint = tmp_path / "must-not-exist-checkpoint.npz"
    with pytest.raises(ValueError, match="finite|shape"):
        trainer.train(
            dataset,
            output,
            epochs=1,
            skip_closed_loop=True,
            checkpoint_path=checkpoint,
        )
    assert not output.exists()
    assert not checkpoint.exists()


def test_completion_horizon_and_prefix_score_are_explicit() -> None:
    prepared = SimpleNamespace(
        trajectory=SimpleNamespace(total_time=37.0)
    )
    assert trainer._resolve_evaluation_duration(
        prepared, None, "completion"
    ) == 52.0
    with pytest.raises(ValueError, match="prefix scoring requires"):
        trainer._resolve_evaluation_duration(prepared, None, "prefix")

    incomplete = {
        "track-a": {
            "skipped": False,
            "evidence_valid": True,
            "safety_passed": True,
            "validity_passed": True,
            "crashed": False,
            "disqualified": False,
            "complete": False,
            "sim_passed": False,
            "termination_reason": "time_limit",
            "gates_passed": 2,
            "total_gates": 4,
            "completion": {
                "complete": False,
                "gates_passed": 2,
                "total_gates": 4,
            },
            "baseline_gates_passed": 2,
            "improvement_pct": 2.0,
        }
    }
    assert trainer._score_closed_loop(incomplete, "completion")["score"] == -1e6
    assert trainer._score_closed_loop(incomplete, "prefix")["score"] == 1.0


def test_rollout_summary_rejects_coerced_evaluator_evidence() -> None:
    malformed = {
        "avg_tracking_error_m": "0.1",
        "sim_passed": "true",
        "safety_passed": 1,
        "validity_passed": True,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "skipped": False,
        "gates_passed": "4",
        "total_gates": 4,
        "termination_reason": "race_complete",
    }
    summary = trainer._exact_rollout_summary(
        malformed, mode="completion", duration_s=45.0
    )
    assert summary["evidence_valid"] is False
    assert summary["sim_passed"] is False
    assert trainer._score_closed_loop(
        {"track-a": {**summary, "improvement_pct": 2.0}}, "completion"
    )["score"] == -1e6


@pytest.mark.parametrize(
    "mutation",
    [
        "top_level_count_mismatch",
        "nested_complete_mismatch",
        "nested_count_mismatch",
        "sim_passed_without_safety",
        "sim_passed_without_completion",
    ],
)
def test_rollout_summary_rejects_contradictory_completion_contract(
    mutation: str,
) -> None:
    result = {
        "avg_tracking_error_m": 0.1,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "skipped": False,
        "gates_passed": 4,
        "total_gates": 4,
        "completion": {
            "complete": True,
            "gates_passed": 4,
            "total_gates": 4,
        },
        "termination_reason": "race_complete",
    }
    if mutation == "top_level_count_mismatch":
        result["gates_passed"] = 3
        result["completion"]["gates_passed"] = 3
    elif mutation == "nested_complete_mismatch":
        result["completion"]["complete"] = False
    elif mutation == "nested_count_mismatch":
        result["completion"]["gates_passed"] = 3
    elif mutation == "sim_passed_without_safety":
        result["safety_passed"] = False
    elif mutation == "sim_passed_without_completion":
        result.update(
            complete=False,
            gates_passed=3,
            termination_reason="time_limit",
        )
        result["completion"] = {
            "complete": False,
            "gates_passed": 3,
            "total_gates": 4,
        }

    summary = trainer._exact_rollout_summary(
        result, mode="completion", duration_s=45.0
    )
    assert summary["evidence_valid"] is False
    assert trainer._score_closed_loop(
        {"track-a": {**summary, "improvement_pct": 2.0}}, "completion"
    )["score"] == -1e6


def test_completion_evaluator_aligns_time_threshold_to_resolved_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = {
        "long-track": SimpleNamespace(trajectory=SimpleNamespace(total_time=37.0))
    }
    observed: dict = {}

    def simulate(_prepared, **kwargs):
        observed.update(kwargs)
        return {
            "avg_tracking_error_m": 0.1,
            "sim_passed": True,
            "safety_passed": True,
            "validity_passed": True,
            "complete": True,
            "crashed": False,
            "disqualified": False,
            "skipped": False,
            "gates_passed": 4,
            "total_gates": 4,
            "completion": {
                "complete": True,
                "gates_passed": 4,
                "total_gates": 4,
            },
            "termination_reason": "race_complete",
        }

    import scripts.benchmark as benchmark

    monkeypatch.setattr(benchmark, "simulate", simulate)
    result = trainer._matrix_baseline(prepared, None, "completion")
    assert observed["duration"] == 52.0
    assert observed["thresholds"] == {"max_total_time_s": 52.0}
    assert result["long-track"]["evidence_valid"] is True
    assert result["long-track"]["sim_passed"] is True


def test_checkpoint_contains_full_training_state_and_prepared_course_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    prepared = {"track-a": SimpleNamespace(artifact_key="prepared-1")}
    seen: list[object] = []
    monkeypatch.setattr(trainer, "_prepare_matrix", lambda **_: prepared)

    def baseline(courses, *args, **kwargs):
        seen.append(courses)
        return _good_matrix()

    def evaluate(_weights, courses, *args, **kwargs):
        seen.append(courses)
        return _good_matrix(error=0.98, improvement=2.0)

    monkeypatch.setattr(trainer, "_matrix_baseline", baseline)
    monkeypatch.setattr(trainer, "_evaluate_closed_loop", evaluate)
    checkpoint = tmp_path / "state.npz"
    summary = trainer.train(
        dataset,
        tmp_path / "weights.npz",
        epochs=2,
        batch_size=32,
        closed_loop_every=1,
        checkpoint_path=checkpoint,
    )
    assert seen == [prepared, prepared, prepared]
    assert summary["closed_loop"]["evaluations_completed"] == 3
    assert set(summary["split"]["train_group_ids"]).isdisjoint(
        summary["split"]["validation_group_ids"]
    )
    with np.load(checkpoint, allow_pickle=False) as state:
        required = {
            "current_W1", "current_b1", "current_W2", "current_b2",
            "optimizer_W1_m", "optimizer_W1_v", "optimizer_W1_t",
            "rng_state_json", "epoch_completed", "history_json",
            "train_idx", "val_idx", "best_val_present", "best_cl_present",
            "best_cl_results_json", "baseline_cl_json",
            "cl_no_improve_count", "training_complete",
        }
        assert required <= set(state.files)
        assert int(state["epoch_completed"]) == 1
        assert int(state["evaluation_count"]) == 3
        assert int(state["training_complete"]) == 1


def test_interrupted_training_resumes_bit_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    full_out = tmp_path / "full.npz"
    full_summary = trainer.train(
        dataset,
        full_out,
        epochs=4,
        batch_size=31,
        skip_closed_loop=True,
        checkpoint_path=tmp_path / "full-state.npz",
    )

    interrupted_checkpoint = tmp_path / "resumed-state.npz"
    real_save = trainer._atomic_save_npz

    class SimulatedInterruption(RuntimeError):
        pass

    def interrupt_after_epoch_one(path, payload):
        real_save(path, payload)
        if (
            Path(path) == interrupted_checkpoint
            and int(payload["epoch_completed"]) == 1
        ):
            raise SimulatedInterruption("power loss after durable checkpoint")

    monkeypatch.setattr(trainer, "_atomic_save_npz", interrupt_after_epoch_one)
    with pytest.raises(SimulatedInterruption):
        trainer.train(
            dataset,
            tmp_path / "partial.npz",
            epochs=4,
            batch_size=31,
            skip_closed_loop=True,
            checkpoint_path=interrupted_checkpoint,
        )
    monkeypatch.setattr(trainer, "_atomic_save_npz", real_save)
    resumed_out = tmp_path / "resumed.npz"
    resumed_summary = trainer.train(
        dataset,
        resumed_out,
        epochs=4,
        batch_size=31,
        skip_closed_loop=True,
        checkpoint_path=interrupted_checkpoint,
        resume=True,
    )

    full_model = TrackerResidualMLP.from_npz(full_out)
    resumed_model = TrackerResidualMLP.from_npz(resumed_out)
    for name in ("W1", "b1", "W2", "b2", "feat_mean", "feat_std"):
        np.testing.assert_array_equal(
            getattr(full_model, name), getattr(resumed_model, name)
        )
    assert resumed_summary["history"] == full_summary["history"]
    assert resumed_summary["resumed_from_checkpoint"] is True


def test_corrupt_resume_checkpoint_fails_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    checkpoint = tmp_path / "corrupt.npz"
    checkpoint.write_bytes(b"not an npz")
    with pytest.raises(RuntimeError, match="invalid training checkpoint"):
        trainer.train(
            dataset,
            tmp_path / "weights.npz",
            epochs=1,
            skip_closed_loop=True,
            checkpoint_path=checkpoint,
            resume=True,
        )


def test_failed_completion_baseline_is_checkpointed_and_not_repeated_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    prepared = {"track-a": SimpleNamespace(artifact_key="prepared-1")}
    calls = 0
    monkeypatch.setattr(trainer, "_prepare_matrix", lambda **_: prepared)

    def incomplete(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = _good_matrix()["track-a"]
        result.update(
            complete=False,
            sim_passed=False,
            gates_passed=3,
            completion={
                "complete": False,
                "gates_passed": 3,
                "total_gates": 4,
            },
            termination_reason="time_limit",
        )
        return {"track-a": result}

    monkeypatch.setattr(trainer, "_matrix_baseline", incomplete)
    checkpoint = tmp_path / "failed-baseline.npz"
    kwargs = dict(
        epochs=1,
        skip_closed_loop=False,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(RuntimeError, match="completion contract"):
        trainer.train(dataset, tmp_path / "first.npz", **kwargs)
    assert checkpoint.exists()
    with pytest.raises(RuntimeError, match="completion contract"):
        trainer.train(
            dataset, tmp_path / "second.npz", resume=True, **kwargs
        )
    assert calls == 1


def test_resume_rejects_closed_loop_evaluator_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    prepared = {"track-a": SimpleNamespace(artifact_key="prepared-1")}
    monkeypatch.setattr(trainer, "_prepare_matrix", lambda **_: prepared)
    monkeypatch.setattr(
        trainer,
        "_closed_loop_evaluator_identity",
        lambda: {"evaluator_version": "v2", "source_digest": "source-a"},
    )
    incomplete = _good_matrix()
    incomplete["track-a"].update(
        complete=False,
        sim_passed=False,
        gates_passed=3,
        completion={
            "complete": False,
            "gates_passed": 3,
            "total_gates": 4,
        },
        termination_reason="time_limit",
    )
    monkeypatch.setattr(
        trainer, "_matrix_baseline", lambda *a, **k: incomplete
    )
    checkpoint = tmp_path / "source-drift.npz"
    kwargs = dict(
        epochs=1,
        skip_closed_loop=False,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(RuntimeError, match="completion contract"):
        trainer.train(dataset, tmp_path / "first.npz", **kwargs)

    monkeypatch.setattr(
        trainer,
        "_closed_loop_evaluator_identity",
        lambda: {"evaluator_version": "v2", "source_digest": "source-b"},
    )
    with pytest.raises(RuntimeError, match="configuration/dataset mismatch"):
        trainer.train(
            dataset, tmp_path / "resumed.npz", resume=True, **kwargs
        )


def test_hard_failed_closed_loop_candidate_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    prepared = {"track-a": SimpleNamespace(artifact_key="prepared-1")}
    monkeypatch.setattr(trainer, "_prepare_matrix", lambda **_: prepared)
    monkeypatch.setattr(
        trainer, "_matrix_baseline", lambda *a, **k: _good_matrix()
    )
    failed = _good_matrix(error=0.5, improvement=50.0)
    failed["track-a"].update(
        sim_passed=False,
        safety_passed=False,
        crashed=True,
        complete=False,
        gates_passed=3,
        completion={
            "complete": False,
            "gates_passed": 3,
            "total_gates": 4,
        },
        termination_reason="crash_ground",
    )
    monkeypatch.setattr(
        trainer, "_evaluate_closed_loop", lambda *a, **k: failed
    )
    weights = tmp_path / "weights.npz"
    checkpoint = tmp_path / "state.npz"
    with pytest.raises(RuntimeError, match="no residual candidate passed"):
        trainer.train(
            dataset,
            weights,
            epochs=1,
            closed_loop_every=1,
            checkpoint_path=checkpoint,
        )
    assert checkpoint.exists()
    assert not weights.exists()
    with np.load(checkpoint, allow_pickle=False) as state:
        history = json.loads(str(state["history_json"].item()))
        assert history[-1]["closed_loop_score"] == -1e6
        assert int(state["best_cl_present"].item()) == 0


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    (
        ("epochs", True, TypeError),
        ("batch_size", 1.5, TypeError),
        ("seed", False, TypeError),
        ("seed", -1, ValueError),
        ("evaluation_seed", 1.5, TypeError),
        ("evaluation_seed", -1, ValueError),
        ("skip_closed_loop", 1, TypeError),
        ("lr_max", float("nan"), ValueError),
        ("lr_min", 0.0, ValueError),
        ("val_frac", float("inf"), ValueError),
        ("closed_loop_duration", -1.0, ValueError),
    ),
)
def test_training_configuration_rejects_coerced_or_nonfinite_values(
    tmp_path: Path, keyword: str, value: object, error: type[Exception]
) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    kwargs = {"epochs": 1, "skip_closed_loop": True, keyword: value}
    with pytest.raises(error):
        trainer.train(dataset, tmp_path / "weights.npz", **kwargs)


def test_resume_rejects_integrity_valid_nonfinite_model_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    checkpoint = tmp_path / "state.npz"
    trainer.train(
        dataset,
        tmp_path / "initial.npz",
        epochs=1,
        skip_closed_loop=True,
        checkpoint_path=checkpoint,
    )
    with np.load(checkpoint, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["current_W1"][0, 0] = np.nan
    trainer._atomic_save_npz(checkpoint, payload)

    with pytest.raises(RuntimeError, match="malformed or non-finite"):
        trainer.train(
            dataset,
            tmp_path / "resumed.npz",
            epochs=1,
            skip_closed_loop=True,
            checkpoint_path=checkpoint,
            resume=True,
        )


def test_resume_rejects_coerced_checkpoint_integer_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset)
    checkpoint = tmp_path / "state.npz"
    trainer.train(
        dataset,
        tmp_path / "initial.npz",
        epochs=1,
        skip_closed_loop=True,
        checkpoint_path=checkpoint,
    )
    with np.load(checkpoint, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["best_val_present"] = np.asarray(1.0, dtype=np.float64)
    trainer._atomic_save_npz(checkpoint, payload)

    with pytest.raises(RuntimeError, match="exact integer scalar"):
        trainer.train(
            dataset,
            tmp_path / "resumed.npz",
            epochs=1,
            skip_closed_loop=True,
            checkpoint_path=checkpoint,
            resume=True,
        )


def _trace_entry() -> tuple:
    vector = np.zeros(3, dtype=np.float64)
    return (
        np.zeros(DEFAULT_N_INPUTS, dtype=np.float64),
        0.0,
        0.0,
        1.0,
        vector.copy(),
        vector.copy(),
        vector.copy(),
        vector.copy(),
        0.0,
        vector.copy(),
        vector.copy(),
        vector.copy(),
        vector.copy(),
    )


def test_collector_uses_prepared_api_and_writes_non_pickle_group_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "track-a.json"
    config_path.write_text(json.dumps({"gates": []}), encoding="utf-8")
    prepared = SimpleNamespace(
        artifact_key="prepared-1",
        cache_states={"trajectory": "hit"},
        trajectory=SimpleNamespace(total_time=30.0),
    )
    calls: list[tuple] = []

    import scripts.benchmark as benchmark
    import scripts.benchmark_matrix as matrix

    def prepare(config, **kwargs):
        calls.append(("prepare", config, kwargs))
        return prepared

    def simulate(course, **kwargs):
        calls.append(("simulate", course, kwargs))
        return {
            "tracker_feature_trace": [_trace_entry() for _ in range(3)],
            "safety_passed": True,
            "validity_passed": True,
            "crashed": False,
            "disqualified": False,
            "complete": True,
            "sim_passed": True,
            "skipped": False,
            "gates_passed": 1,
            "total_gates": 1,
            "completion": {
                "complete": True,
                "gates_passed": 1,
                "total_gates": 1,
            },
            "termination_reason": "race_complete",
            "sim_time_s": 31.0,
            "avg_tracking_error_m": 0.1,
        }

    monkeypatch.setattr(matrix, "_list_configs", lambda: [config_path])
    monkeypatch.setattr(benchmark, "prepare_course", prepare)
    monkeypatch.setattr(benchmark, "simulate", simulate)
    monkeypatch.setattr(collector, "_SKIP", set())
    output = tmp_path / "dataset.npz"
    summary = collector.collect(output)

    assert [call[0] for call in calls] == ["prepare", "simulate"]
    assert calls[1][2]["duration"] == 45.0
    assert summary["dataset_schema_version"] == 4
    with np.load(output, allow_pickle=False) as archive:
        assert archive["track_names"].dtype.kind == "U"
        assert archive["session_names"].dtype.kind == "U"
        np.testing.assert_array_equal(archive["track_id"], [0, 0, 0])
        np.testing.assert_array_equal(archive["session_id"], [0, 0, 0])
        manifest = json.loads(str(archive["dataset_manifest_json"]))
        assert manifest["collection_mode"] == "completion"
        assert manifest["evaluator_version"]
        assert len(manifest["collector_source_sha256"]) == 64
        assert len(manifest["track_config_sha256"]["track-a"]) == 64
        assert manifest["dependency_fingerprint"]["dependencies"]


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    (
        ("seed", True, TypeError),
        ("seed", -1, ValueError),
        ("allow_prefix", 1, TypeError),
        ("duration", float("nan"), ValueError),
        ("duration", False, ValueError),
    ),
)
def test_collector_rejects_coerced_or_nonfinite_configuration(
    tmp_path: Path, keyword: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        collector.collect(tmp_path / "dataset.npz", **{keyword: value})


def test_collector_never_publishes_unsafe_trace_even_in_prefix_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "track-a.json"
    config_path.write_text(json.dumps({"gates": []}), encoding="utf-8")
    prepared = SimpleNamespace(
        artifact_key="prepared-1",
        cache_states={},
        trajectory=SimpleNamespace(total_time=10.0),
    )
    import scripts.benchmark as benchmark
    import scripts.benchmark_matrix as matrix

    monkeypatch.setattr(matrix, "_list_configs", lambda: [config_path])
    monkeypatch.setattr(benchmark, "prepare_course", lambda *a, **k: prepared)
    monkeypatch.setattr(
        benchmark,
        "simulate",
        lambda *a, **k: {
            "tracker_feature_trace": [_trace_entry()],
            "safety_passed": False,
            "validity_passed": True,
            "crashed": True,
            "disqualified": False,
            "complete": False,
            "sim_passed": False,
            "skipped": False,
            "gates_passed": 0,
            "total_gates": 1,
            "completion": {
                "complete": False,
                "gates_passed": 0,
                "total_gates": 1,
            },
            "termination_reason": "crash_ground",
        },
    )
    monkeypatch.setattr(collector, "_SKIP", set())
    output = tmp_path / "must-not-exist.npz"
    with pytest.raises(RuntimeError, match="partial/unsafe"):
        collector.collect(output, duration=1.0, allow_prefix=True)
    assert not output.exists()


@pytest.mark.parametrize(
    ("complete_value", "termination_reason"),
    [
        ("true", "race_complete"),
        (1, "race_complete"),
        (False, "race_complete"),
        ("false", "time_limit"),
        (0, "time_limit"),
        (None, "time_limit"),
        (_MISSING, "time_limit"),
    ],
)
def test_collector_rejects_coerced_or_contradictory_completion_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_value,
    termination_reason: str,
) -> None:
    config_path = tmp_path / "track-a.json"
    config_path.write_text(json.dumps({"gates": []}), encoding="utf-8")
    prepared = SimpleNamespace(
        artifact_key="prepared-1",
        cache_states={},
        trajectory=SimpleNamespace(total_time=10.0),
    )
    import scripts.benchmark as benchmark
    import scripts.benchmark_matrix as matrix

    monkeypatch.setattr(matrix, "_list_configs", lambda: [config_path])
    monkeypatch.setattr(benchmark, "prepare_course", lambda *a, **k: prepared)
    def simulate(*args, **kwargs):
        result = {
            "tracker_feature_trace": [_trace_entry()],
            "safety_passed": True,
            "validity_passed": True,
            "crashed": False,
            "disqualified": False,
            "sim_passed": False,
            "skipped": False,
            "gates_passed": 0,
            "total_gates": 1,
            "completion": {
                "complete": False,
                "gates_passed": 0,
                "total_gates": 1,
            },
            "termination_reason": termination_reason,
        }
        if complete_value is not _MISSING:
            result["complete"] = complete_value
        return result

    monkeypatch.setattr(benchmark, "simulate", simulate)
    monkeypatch.setattr(collector, "_SKIP", set())
    output = tmp_path / "must-not-exist.npz"
    with pytest.raises(RuntimeError, match="partial/unsafe"):
        collector.collect(output, duration=1.0, allow_prefix=True)
    assert not output.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"crashed": True},
        {"disqualified": True},
        {"crashed": None},
        {"disqualified": "false"},
        {"termination_reason": "time_limit"},
    ],
)
def test_collector_rejects_contradictory_safety_or_terminal_evidence(
    overrides,
) -> None:
    result = {
        "safety_passed": True,
        "validity_passed": True,
        "crashed": False,
        "disqualified": False,
        "complete": True,
        "sim_passed": True,
        "skipped": False,
        "gates_passed": 1,
        "total_gates": 1,
        "completion": {
            "complete": True,
            "gates_passed": 1,
            "total_gates": 1,
        },
        "termination_reason": "race_complete",
    }
    result.update(overrides)
    assert not collector._accepted_collection_outcome(result, allow_prefix=True)


def test_collector_accepts_only_literal_safe_time_limit_prefix() -> None:
    result = {
        "safety_passed": True,
        "validity_passed": True,
        "crashed": False,
        "disqualified": False,
        "complete": False,
        "sim_passed": False,
        "skipped": False,
        "gates_passed": 0,
        "total_gates": 1,
        "completion": {
            "complete": False,
            "gates_passed": 0,
            "total_gates": 1,
        },
        "termination_reason": "time_limit",
    }
    assert collector._accepted_collection_outcome(result, allow_prefix=True)
    assert not collector._accepted_collection_outcome(result, allow_prefix=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "top_level_count_mismatch",
        "nested_complete_mismatch",
        "nested_count_mismatch",
        "sim_passed_without_safety",
        "sim_passed_without_completion",
    ],
)
def test_collector_rejects_contradictory_completion_contract(
    mutation: str,
) -> None:
    result = {
        "safety_passed": True,
        "validity_passed": True,
        "crashed": False,
        "disqualified": False,
        "complete": True,
        "sim_passed": True,
        "skipped": False,
        "gates_passed": 4,
        "total_gates": 4,
        "completion": {
            "complete": True,
            "gates_passed": 4,
            "total_gates": 4,
        },
        "termination_reason": "race_complete",
    }
    if mutation == "top_level_count_mismatch":
        result["gates_passed"] = 3
        result["completion"]["gates_passed"] = 3
    elif mutation == "nested_complete_mismatch":
        result["completion"]["complete"] = False
    elif mutation == "nested_count_mismatch":
        result["completion"]["gates_passed"] = 3
    elif mutation == "sim_passed_without_safety":
        result["safety_passed"] = False
    elif mutation == "sim_passed_without_completion":
        result.update(
            complete=False,
            gates_passed=3,
            termination_reason="time_limit",
        )
        result["completion"] = {
            "complete": False,
            "gates_passed": 3,
            "total_gates": 4,
        }

    assert not collector._accepted_collection_outcome(
        result, allow_prefix=True
    )
