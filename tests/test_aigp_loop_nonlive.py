from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from aigp_loop._util import sha256_file
from aigp_loop._util import json_hash
from aigp_loop.nonlive import (
    CORE_EVALUATOR_FILES,
    FULL_TRACK_SET,
    adapt_matrix_evidence,
    evaluator_source_hashes,
)
from aigp_loop.promotion import Tier
from planning.artifact_cache import dependency_fingerprint
from planning.artifact_cache import sha256_json
from scripts.aigp_nonlive import _FULL_SUITE_PYTEST_ARGS


def _worker_fingerprint() -> dict:
    result = copy.deepcopy(dependency_fingerprint())
    result["numeric_thread_environment"] = {
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
    return result


def _track(name: str) -> tuple[dict, dict]:
    validation = {
        "ok": True,
        "reason": "complete",
        "gates_passed": 4,
        "total_gates": 4,
        "crashed": False,
        "disqualified": False,
    }
    summary = {
        "gates_passed": 4,
        "total_gates": 4,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 4, "total_gates": 4},
        "plan_validation": validation,
        "p95_tracking_error_m": 0.2,
        "max_tracking_error_m": 0.3,
        "sim_time_s": 10.0,
        "artifact_hashes": {"prepared": "a" * 64},
        "cache_hit_or_miss": "miss",
        "prepared_cache_states": {
            "racing_line": "hit",
            "trajectory": "hit",
            "plan_validation": "hit",
            "ilc": "hit",
        },
        "rollout_executed": True,
        "result_cache_enabled": False,
    }
    result = {
        **{
            key: copy.deepcopy(value)
            for key, value in summary.items()
            if key != "prepared_cache_states"
        },
        "sim_type": "synthetic_kinematic",
        "available": True,
        "skipped": False,
        "cache": {
            "racing_line": "hit",
            "trajectory": "hit",
            "plan_validation": "hit",
            "ilc": "hit",
            "benchmark_result": "miss",
        },
    }
    return summary, result


def _matrix(names=("race_01",)) -> dict:
    pairs = {name: _track(name) for name in names}
    orchestrator_fingerprint = dependency_fingerprint()
    worker_fingerprint = _worker_fingerprint()
    provenance = {
        "commit": "deadbeef",
        "dirty": False,
        "dirty_diff_hash": "1" * 64,
        "tracked_diff_hash": "2" * 64,
        "untracked_content_hash": "3" * 64,
        "excluded_untracked_paths": [],
    }
    resolved_configuration = {
        "runtime": {"worker_dependency_fingerprint": worker_fingerprint},
        "code_provenance": provenance,
    }
    return {
        "evaluator_version": "synthetic-v2",
        "schema_version": "benchmark-v2",
        "comparison_series": "comparison-v2",
        "all_passed": True,
        "config_hash": sha256_json(resolved_configuration),
        "resolved_configuration": resolved_configuration,
        "dependency_fingerprint": worker_fingerprint,
        "dependency_fingerprints": {
            "orchestrator": orchestrator_fingerprint,
            "worker_expected": worker_fingerprint,
            "workers_observed": [
                {"track": name, "fingerprint": worker_fingerprint}
                for name in names
            ],
        },
        "worker_environment_verified": True,
        "code_provenance": provenance,
        "code_provenance_observations": [
            {"track": name, "code_provenance": provenance} for name in names
        ],
        "code_provenance_verified": True,
        "seed": 42,
        "cache_hit_or_miss": "miss",
        "tracks": {name: pair[0] for name, pair in pairs.items()},
        "results": {name: pair[1] for name, pair in pairs.items()},
    }


def _preparation() -> dict:
    return {
        "schema": "aigp-cache-preparation/3",
        "preparation_result_sha256": "e" * 64,
        "dependency_fingerprint_sha256": json_hash(_worker_fingerprint()),
        "cache_hit_or_miss": "miss",
    }


def test_t2_adapter_derives_hard_gates_and_quality_from_exact_track_evidence():
    result = adapt_matrix_evidence(
        _matrix(),
        tier=Tier.T2_WARM_SIM,
        expected_tracks=("race_01",),
        source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
        cache_preparation=_preparation(),
    )
    assert all(result["promotion"]["hard_gates"].values())
    assert result["promotion"]["quality"] == {
        "completion_reliability": 1.0,
        "centering_margin": -0.2,
        "stability_margin": -0.3,
        "race_time_s": 10.0,
    }
    assert result["domain_provenance"]["powered_resources_used"] is False


def test_adapter_itself_rejects_evidence_that_omits_core_evaluator_sources():
    with pytest.raises(ValueError, match="required core evaluator"):
        adapt_matrix_evidence(
            _matrix(),
            tier=Tier.T2_WARM_SIM,
            expected_tracks=("race_01",),
            source_hashes={"scripts/benchmark.py": "b" * 64},
            cache_preparation=_preparation(),
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value["tracks"]["race_01"].update(crashed="false"), "booleans"),
        (lambda value: value["results"]["race_01"].update(skipped=True), "skipped"),
        (lambda value: value["tracks"]["race_01"].update(gate_pass_rate=0.5), "contradicts"),
        (lambda value: value["results"]["race_01"].update(sim_type="powered"), "synthetic"),
    ],
)
def test_adapter_rejects_coercion_contradiction_and_unproved_domain(mutation, message):
    matrix = _matrix()
    mutation(matrix)
    with pytest.raises(ValueError, match=message):
        adapt_matrix_evidence(
            matrix,
            tier=Tier.T2_WARM_SIM,
            expected_tracks=("race_01",),
            source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
            cache_preparation=_preparation(),
        )


def test_t4_requires_exact_seven_tracks_and_passing_full_nonlive_suite():
    suite = {
        "schema": "aigp-nonlive-pytest/1",
        "passed": True,
        "returncode": 0,
        "pytest_args": list(_FULL_SUITE_PYTEST_ARGS),
        "output_sha256": "c" * 64,
        "output_tail": "301 passed",
    }
    result = adapt_matrix_evidence(
        _matrix(FULL_TRACK_SET),
        tier=Tier.T4_FULL_NON_LIVE,
        expected_tracks=FULL_TRACK_SET,
        source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
        full_nonlive_suite=suite,
    )
    assert result["promotion"]["hard_gates"]["valid"] is True
    failed = dict(suite, passed=False, returncode=1)
    result = adapt_matrix_evidence(
        _matrix(FULL_TRACK_SET),
        tier=Tier.T4_FULL_NON_LIVE,
        expected_tracks=FULL_TRACK_SET,
        source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
        full_nonlive_suite=failed,
    )
    assert result["promotion"]["hard_gates"]["valid"] is False


def test_adapter_accepts_and_hashes_real_dependency_fingerprint_mapping():
    matrix = _matrix()
    evidence = adapt_matrix_evidence(
        matrix,
        tier=Tier.T2_WARM_SIM,
        expected_tracks=("race_01",),
        source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
        cache_preparation=_preparation(),
    )
    assert len(evidence["evaluation_input_hash"]) == 64


@pytest.mark.parametrize("location", ["prepared", "rollout", "final_cache"])
def test_t2_rejects_a_cold_cached_or_unexecuted_measured_run(location):
    matrix = _matrix()
    if location == "prepared":
        matrix["tracks"]["race_01"]["prepared_cache_states"]["trajectory"] = "miss"
        matrix["results"]["race_01"]["cache"]["trajectory"] = "miss"
    elif location == "rollout":
        matrix["tracks"]["race_01"]["rollout_executed"] = False
        matrix["results"]["race_01"]["rollout_executed"] = False
    else:
        matrix["cache_hit_or_miss"] = "hit"
        matrix["tracks"]["race_01"]["cache_hit_or_miss"] = "hit"
        matrix["tracks"]["race_01"]["rollout_executed"] = False
        matrix["tracks"]["race_01"]["result_cache_enabled"] = True
        matrix["results"]["race_01"]["cache_hit_or_miss"] = "hit"
        matrix["results"]["race_01"]["rollout_executed"] = False
        matrix["results"]["race_01"]["result_cache_enabled"] = True
        matrix["results"]["race_01"]["cache"]["benchmark_result"] = "hit"
    with pytest.raises(ValueError, match="cache|rollout|prepared"):
        adapt_matrix_evidence(
            matrix,
            tier=Tier.T2_WARM_SIM,
            expected_tracks=("race_01",),
            source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
            cache_preparation=_preparation(),
        )


@pytest.mark.parametrize("field", ["gates_passed", "completion", "plan_validation"])
def test_adapter_rejects_summary_full_result_contradictions(field):
    matrix = _matrix()
    if field == "gates_passed":
        matrix["results"]["race_01"][field] = 3
    elif field == "completion":
        matrix["results"]["race_01"][field] = {
            "complete": False,
            "gates_passed": 4,
            "total_gates": 4,
        }
    else:
        matrix["results"]["race_01"][field] = {
            **matrix["results"]["race_01"][field],
            "ok": False,
        }
    with pytest.raises(ValueError, match="contradict"):
        adapt_matrix_evidence(
            matrix,
            tier=Tier.T2_WARM_SIM,
            expected_tracks=("race_01",),
            source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
            cache_preparation=_preparation(),
        )


def test_adapter_rejects_complete_false_when_all_gates_passed():
    matrix = _matrix()
    for side in (matrix["tracks"]["race_01"], matrix["results"]["race_01"]):
        side["complete"] = False
        side["sim_passed"] = False
        side["completion"]["complete"] = False
    matrix["all_passed"] = False
    with pytest.raises(ValueError, match="completion contradicts"):
        adapt_matrix_evidence(
            matrix,
            tier=Tier.T2_WARM_SIM,
            expected_tracks=("race_01",),
            source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
            cache_preparation=_preparation(),
        )


def test_t4_suite_passed_must_equal_zero_returncode():
    suite = {
        "schema": "aigp-nonlive-pytest/1",
        "passed": True,
        "returncode": 1,
        "pytest_args": list(_FULL_SUITE_PYTEST_ARGS),
        "output_sha256": "c" * 64,
        "output_tail": "failure",
    }
    with pytest.raises(ValueError, match="full non-live pytest"):
        adapt_matrix_evidence(
            _matrix(FULL_TRACK_SET),
            tier=Tier.T4_FULL_NON_LIVE,
            expected_tracks=FULL_TRACK_SET,
            source_hashes={name: "b" * 64 for name in CORE_EVALUATOR_FILES},
            full_nonlive_suite=suite,
        )


def test_trusted_manifest_requires_all_core_evaluator_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for relative in CORE_EVALUATOR_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
    manifest_path = root / "trusted.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "aigp-trusted-evaluator-files/1",
                "files": {"README.md": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required core"):
        evaluator_source_hashes(root, manifest_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "aigp-trusted-evaluator-files/1",
                "files": {
                    relative: sha256_file(root / relative)
                    for relative in CORE_EVALUATOR_FILES
                },
            }
        ),
        encoding="utf-8",
    )
    hashes = evaluator_source_hashes(root, manifest_path)
    assert CORE_EVALUATOR_FILES <= set(hashes)


def test_trusted_evaluator_binds_exact_track_config_inventory(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    expected = {}
    for relative in CORE_EVALUATOR_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
        expected[relative] = sha256_file(target)
    manifest = root / "trusted.json"
    manifest.write_text(
        json.dumps(
            {"schema": "aigp-trusted-evaluator-files/1", "files": expected}
        ),
        encoding="utf-8",
    )
    extra = root / "sim_pybullet" / "configs" / "unexpected.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="config inventory"):
        evaluator_source_hashes(root, manifest)
    extra.unlink()
    (root / "sim_pybullet" / "configs" / "race_01.json").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="file mismatch"):
        evaluator_source_hashes(root, manifest)


@pytest.mark.parametrize("link_kind", ["file", "parent"])
def test_trusted_evaluator_rejects_symlink_or_reparse_components(tmp_path, link_kind):
    root = tmp_path / "repo"
    root.mkdir()
    expected = {}
    for relative in CORE_EVALUATOR_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
        expected[relative] = sha256_file(target)

    try:
        if link_kind == "file":
            victim = root / "aigp_loop" / "nonlive.py"
            outside = tmp_path / "outside_nonlive.py"
            outside.write_text("aigp_loop/nonlive.py", encoding="utf-8")
            victim.unlink()
            victim.symlink_to(outside)
        else:
            victim = root / "scripts"
            outside = tmp_path / "outside_scripts"
            outside.mkdir()
            for name in ("aigp_nonlive.py", "benchmark.py", "benchmark_matrix.py"):
                (victim / name).unlink()
                (outside / name).write_text(f"scripts/{name}", encoding="utf-8")
            victim.rmdir()
            victim.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")

    manifest_path = root / "trusted.json"
    manifest_path.write_text(
        json.dumps(
            {"schema": "aigp-trusted-evaluator-files/1", "files": expected}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatch|symlink|reparse"):
        evaluator_source_hashes(root, manifest_path)


def test_nonlive_bootstrap_rejects_unmanifested_scripts_initializer(
    tmp_path, monkeypatch
):
    import scripts.aigp_nonlive as bootstrap

    root = tmp_path / "candidate"
    hashes = {}
    for reviewed in bootstrap._STARTUP_CORE_FILES:
        target = root / reviewed
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(reviewed, encoding="utf-8")
        hashes[reviewed] = sha256_file(target)
    manifest = root / "trusted.json"
    manifest.write_text(
        json.dumps(
            {"schema": "aigp-trusted-evaluator-files/1", "files": hashes}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bootstrap, "_REPO", root.resolve())
    bootstrap._verify_startup_boundary("trusted.json")
    (root / "scripts" / "__init__.py").write_text(
        "raise RuntimeError('shadow')\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="import-boundary collision"):
        bootstrap._verify_startup_boundary("trusted.json")


@pytest.mark.parametrize(
    "relative",
    [
        "aigp_loop/__pycache__/nonlive.cpython-312.pyc",
        "aigp_loop/__PYCACHE__/nonlive.cpython-312.PYC",
    ],
)
def test_t2_startup_rejects_adjacent_executable_bytecode_before_import(
    tmp_path, monkeypatch, relative
):
    import scripts.aigp_nonlive as bootstrap

    root = tmp_path / "candidate"
    hashes = {}
    for reviewed in bootstrap._STARTUP_CORE_FILES:
        target = root / reviewed
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(reviewed, encoding="utf-8")
        hashes[reviewed] = sha256_file(target)
    manifest = root / "trusted.json"
    manifest.write_text(
        json.dumps(
            {"schema": "aigp-trusted-evaluator-files/1", "files": hashes}
        ),
        encoding="utf-8",
    )
    bytecode = root / relative
    bytecode.parent.mkdir(parents=True, exist_ok=True)
    bytecode.write_bytes(b"executable")
    monkeypatch.setattr(bootstrap, "_REPO", root.resolve())
    with pytest.raises(ValueError, match="executable bytecode"):
        bootstrap._verify_startup_boundary("trusted.json")


def test_startup_rejects_same_name_package_shadowing_pinned_module(
    tmp_path, monkeypatch
):
    import scripts.aigp_nonlive as bootstrap

    root = tmp_path / "candidate"
    hashes = {}
    for relative in bootstrap._STARTUP_CORE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
        hashes[relative] = sha256_file(target)
    manifest = root / "trusted.json"
    manifest.write_text(
        json.dumps(
            {"schema": "aigp-trusted-evaluator-files/1", "files": hashes}
        ),
        encoding="utf-8",
    )
    shadow = root / "aigp_loop" / "nonlive" / "__init__.py"
    shadow.parent.mkdir()
    shadow.write_text("raise RuntimeError('shadow executed')\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "_REPO", root.resolve())
    with pytest.raises(ValueError, match="import-boundary collision"):
        bootstrap._verify_startup_boundary("trusted.json")


@pytest.mark.parametrize(
    "relative",
    [
        "aigp_loop/__init__.pyd",
        "aigp_loop/__INIT__.PYD",
        "scripts/__init__.pyd",
        "scripts.pyd",
        "scripts.PYD",
        "scripts/benchmark.cp312-win_amd64.pyd",
    ],
)
def test_startup_rejects_native_extension_import_alternatives(
    tmp_path, monkeypatch, relative
):
    import scripts.aigp_nonlive as bootstrap

    root = tmp_path / "candidate"
    hashes = {}
    for reviewed in bootstrap._STARTUP_CORE_FILES:
        target = root / reviewed
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(reviewed, encoding="utf-8")
        hashes[reviewed] = sha256_file(target)
    manifest = root / "trusted.json"
    manifest.write_text(
        json.dumps(
            {"schema": "aigp-trusted-evaluator-files/1", "files": hashes}
        ),
        encoding="utf-8",
    )
    collision = root / relative
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_bytes(b"unreviewed native extension")
    monkeypatch.setattr(bootstrap, "_REPO", root.resolve())
    with pytest.raises(ValueError, match="import-boundary collision"):
        bootstrap._verify_startup_boundary("trusted.json")


def test_full_suite_streams_multi_megabyte_output_with_a_bounded_tail(
    tmp_path, monkeypatch
):
    import aigp_loop.scheduler as scheduler_module
    import scripts.aigp_nonlive as bootstrap

    attacker_tests = tmp_path / "tests"
    attacker_tests.mkdir()
    (attacker_tests / "test_injected.py").write_text(
        "raise RuntimeError('must not collect')\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only ../untrusted")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")
    monkeypatch.setenv("pytest_debug", "inherited")
    monkeypatch.setenv("AIGP_TEST_SECRET_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")

    class FakeProcess:
        pid = 123

        def __init__(self):
            self.stdout = io.BytesIO(b"o" * 8_000_000)
            self.stderr = io.BytesIO(b"e" * 8_000_000)

        def wait(self, timeout=None):
            return 1

    captured = {}

    class FakeContainment:
        def __init__(self):
            captured["containment_created"] = True

        def attach_and_resume(self, process):
            captured["attached"] = process

        def terminate_and_prove(self, process):
            captured["cleaned"] = process

        def close(self):
            captured["containment_closed"] = True

    def launch(*args, **kwargs):
        captured["command"] = args[0]
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return FakeProcess()

    monkeypatch.setattr(bootstrap.subprocess, "Popen", launch)
    monkeypatch.setattr(
        scheduler_module, "_WindowsJobContainment", FakeContainment
    )
    monkeypatch.setattr(
        scheduler_module.TrialScheduler,
        "_terminate_process_tree",
        staticmethod(lambda process: captured.update(cleaned=process)),
    )
    evidence = bootstrap._full_suite(1.0)
    assert evidence["returncode"] == 1
    assert evidence["passed"] is False
    assert len(evidence["output_tail"]) <= 32_000
    assert len(evidence["output_sha256"]) == 64
    assert tuple(evidence["pytest_args"]) == bootstrap._FULL_SUITE_PYTEST_ARGS
    assert captured["env"]
    assert captured["command"]
    assert captured["cwd"] == str(bootstrap._REPO)
    assert captured["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"]["PYTHONPYCACHEPREFIX"]
    assert "pycache_prefix=" in " ".join(captured["command"])
    assert "PYTEST_ADDOPTS" not in captured["env"]
    assert "PYTEST_PLUGINS" not in captured["env"]
    assert "pytest_debug" not in captured["env"]
    assert "AIGP_TEST_SECRET_TOKEN" not in captured["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]
    assert captured["env"]["AIGP_CACHE_ROOT"]
    assert captured["env"]["AIGP_TRIAL_OFFLINE"] == "1"
    if "PATH" in os.environ:
        assert captured["env"]["PATH"] == os.environ["PATH"]
    assert captured["cleaned"] is not None


def test_posix_delegated_full_suite_timeout_fails_for_outer_group_cleanup(
    tmp_path, monkeypatch
):
    import scripts.aigp_nonlive as bootstrap

    class FakeProcess:
        pid = 456

        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")
            self.killed = False
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise bootstrap.subprocess.TimeoutExpired("pytest", timeout)
            return -9

        def kill(self):
            self.killed = True

    captured = {}
    process = FakeProcess()

    def launch(*args, **kwargs):
        captured["kwargs"] = kwargs
        return process

    fake_environment = dict(os.environ)
    fake_environment["AIGP_TRIAL_ID"] = "trial-under-scheduler"
    monkeypatch.setattr(
        bootstrap,
        "os",
        SimpleNamespace(
            name="posix",
            environ=fake_environment,
            getpgrp=lambda: 9001,
            getpid=lambda: 9001,
        ),
    )
    monkeypatch.setattr(bootstrap.subprocess, "Popen", launch)

    with pytest.raises(RuntimeError, match="outer scheduler process-group cleanup"):
        bootstrap._run_full_suite(0.01, tmp_path / "pycache", tmp_path / "cache")

    assert process.killed is True
    assert process.wait_calls == 2
    assert "start_new_session" not in captured["kwargs"]
    assert "creationflags" not in captured["kwargs"]


def test_numeric_caps_are_applied_before_repository_numeric_imports(monkeypatch):
    import inspect
    import os
    import scripts.aigp_nonlive as bootstrap

    for name in bootstrap._NUMERIC_STARTUP_ENVIRONMENT:
        monkeypatch.setenv(name, "host-dependent")
    bootstrap._apply_numeric_startup_environment()
    assert {
        name: os.environ[name]
        for name in bootstrap._NUMERIC_STARTUP_ENVIRONMENT
    } == bootstrap._NUMERIC_STARTUP_ENVIRONMENT
    source = inspect.getsource(bootstrap.main)
    assert source.index("_apply_numeric_startup_environment()") < source.index(
        "from scripts.benchmark import prepare_course"
    )


def test_standalone_startup_snapshot_rejects_metadata_drift(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    import scripts.aigp_nonlive as bootstrap

    target = tmp_path / "trusted.py"
    target.write_bytes(b"reviewed")
    real_fstat = os.fstat
    calls = 0

    def drift(descriptor):
        nonlocal calls
        info = real_fstat(descriptor)
        calls += 1
        if calls != 2:
            return info
        return SimpleNamespace(
            st_mode=info.st_mode,
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_size=info.st_size + 1,
            st_mtime_ns=info.st_mtime_ns,
            st_ctime_ns=info.st_ctime_ns,
            st_mtime=info.st_mtime,
            st_ctime=info.st_ctime,
        )

    monkeypatch.setattr(os, "fstat", drift)
    with pytest.raises(ValueError, match="mutated while being read"):
        bootstrap._read_stable_regular_file(target, maximum_bytes=1024)


def test_t4_inventory_rejects_unmanifested_committed_test_or_discovery_input(
    tmp_path, monkeypatch
):
    import scripts.aigp_nonlive as bootstrap

    root = tmp_path / "candidate"
    root.mkdir()
    trusted = {}
    for relative in ("conftest.py", "pyproject.toml"):
        target = root / relative
        target.write_text("# reviewed\n", encoding="utf-8")
        trusted[relative] = "a" * 64
    for relative_root in bootstrap._T4_TEST_ROOTS:
        test_root = root / relative_root
        test_root.mkdir(parents=True)
        target = test_root / "test_reviewed.py"
        target.write_text("def test_ok():\n    pass\n", encoding="utf-8")
        trusted[target.relative_to(root).as_posix()] = "b" * 64
    monkeypatch.setattr(bootstrap, "_REPO", root.resolve())
    bootstrap._audit_t4_test_boundary(trusted)
    cache = root / "control" / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "test_reviewed.cpython-313-pytest.pyc").write_bytes(b"generated")
    (root / "control" / "tests" / ".pytest_cache").mkdir()
    with pytest.raises(ValueError, match="executable bytecode"):
        bootstrap._audit_t4_test_boundary(trusted)
    (cache / "test_reviewed.cpython-313-pytest.pyc").unlink()
    cache.rmdir()
    bootstrap._audit_t4_test_boundary(trusted)
    unreviewed_asset = root / "control" / "tests" / "fixture.bin"
    unreviewed_asset.write_bytes(b"real evidence input")
    with pytest.raises(ValueError, match="inventory differs"):
        bootstrap._audit_t4_test_boundary(trusted)
    unreviewed_asset.unlink()
    injected = root / "control" / "tests" / "test_unreviewed.py"
    injected.write_text(
        "raise RuntimeError('must never collect')\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inventory differs"):
        bootstrap._audit_t4_test_boundary(trusted)
    injected.unlink()
    (root / "sitecustomize.py").write_text(
        "raise RuntimeError('must never start')\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="startup discovery"):
        bootstrap._audit_t4_test_boundary(trusted)


def test_full_suite_pytest_args_run_with_autoload_disabled_and_no_cache(tmp_path):
    import os
    import subprocess
    import sys
    import scripts.aigp_nonlive as bootstrap

    test_file = tmp_path / "test_t4_bootstrap.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    options = list(bootstrap._FULL_SUITE_PYTEST_ARGS)
    config_index = options.index("-c")
    del options[config_index : config_index + 2]
    options = [
        item for item in options if item not in bootstrap._T4_TEST_ROOTS
    ]
    run = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            *options,
            str(test_file),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout
    assert not (tmp_path / ".pytest_cache").exists()
