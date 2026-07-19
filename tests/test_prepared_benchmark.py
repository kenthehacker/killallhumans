from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from planning.artifact_cache import (
    ArtifactStore,
    artifact_key,
    dependency_fingerprint,
    sha256_json,
)
from scripts import benchmark
from scripts import benchmark_matrix


def _one_gate_course(*, proximity: float = 1.0, planner_entry: float = 0.4):
    return {
        "gate_defaults": {
            "interior_width_m": 1.5,
            "interior_height_m": 1.5,
            "border_width_m": 0.2,
        },
        "sequencer": {"proximity_pass_distance": proximity},
        "planner": {"entry_exit_offset_m": planner_entry},
        "gates": [
            {
                "id": "gate-1",
                "sequence_index": 1,
                "pose": {"x": 2.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            }
        ],
        "start": {"position": [0.0, 0.0, 1.5]},
    }


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("seed", True, TypeError),
        ("seed", 1.5, TypeError),
        ("seed", -1, ValueError),
        ("duration", "1.0", TypeError),
        ("dt", True, TypeError),
        ("record_position_trace", 1, TypeError),
        ("use_result_cache", "false", TypeError),
        ("config", [], TypeError),
        ("planning_config", [], TypeError),
        ("tracker_config_overrides", [], TypeError),
        ("thresholds", [], TypeError),
    ],
)
def test_synthetic_public_arguments_fail_before_planning(
    monkeypatch, keyword, value, error
):
    monkeypatch.setattr(
        benchmark,
        "prepare_course",
        lambda *args, **kwargs: pytest.fail("invalid input reached planning"),
    )
    kwargs = {"config": _one_gate_course(), keyword: value}
    with pytest.raises(error):
        benchmark.run_synthetic_benchmark(**kwargs)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("seed", True, TypeError),
        ("seed", 2.0, TypeError),
        ("seed", -1, ValueError),
        ("record_position_trace", 1, TypeError),
        ("controller_config", [], TypeError),
        ("thresholds", [], TypeError),
    ],
)
def test_simulate_rejects_coerced_public_arguments(keyword, value, error):
    prepared = SimpleNamespace(dt=0.02)
    with pytest.raises(error):
        benchmark.simulate(prepared, duration=0.0, **{keyword: value})


def test_prepare_course_requires_mapping_inputs():
    with pytest.raises(TypeError, match="track must be a mapping"):
        benchmark.prepare_course([])
    with pytest.raises(TypeError, match="planning_config must be a mapping"):
        benchmark.prepare_course(_one_gate_course(), planning_config=[])


def test_pytest_default_cache_root_is_session_isolated(_isolated_artifact_cache):
    assert ArtifactStore().root == Path(_isolated_artifact_cache).resolve()
    assert ArtifactStore().root != (Path(__file__).resolve().parents[1] / ".cache")


def test_artifact_store_json_corruption_fails_closed(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("example", {"value": 1}, schema_version="test-v1")
    path = store.save_json("example", key, {"answer": 42})
    assert store.load_json("example", key) == {"answer": 42}

    path.write_bytes(b'{"partial":')
    assert store.load_json("example", key) is None
    with store.lock("example", key):
        store.save_json("example", key, {"answer": 43})
    assert store.load_json("example", key) == {"answer": 43}
    assert not list(path.parent.glob("*.partial"))


def test_artifact_store_rejects_ambiguous_duplicate_json_keys(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("duplicate", {"value": 1}, schema_version="test-v1")
    path = store.save_json("duplicate", key, {"value": 1})
    encoded = path.read_text(encoding="utf-8")
    encoded = encoded.replace(
        '"namespace":"duplicate"',
        '"namespace":"untrusted","namespace":"duplicate"',
        1,
    )
    path.write_text(encoded, encoding="utf-8")
    assert store.load_json("duplicate", key) is None


def test_artifact_lock_is_a_persistent_reusable_os_rendezvous(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("lease", {"value": 1}, schema_version="test-v1")
    lock_path = store.path("lease", key, ".lock")

    with store.lock("lease", key):
        assert lock_path.exists()
        assert lock_path.stat().st_size >= 1
    sentinel = lock_path.read_bytes()

    with store.lock("lease", key, timeout_s=0.5):
        assert lock_path.exists()
    assert lock_path.read_bytes() == sentinel


def test_artifact_lock_os_lease_releases_after_abrupt_process_exit(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("crash", {"value": 1}, schema_version="test-v1")
    lock_path = store.path("crash", key, ".lock")
    marker = tmp_path / "lease-held"
    script = r"""
import os
import sys
from pathlib import Path

from planning.artifact_cache import ArtifactStore

root, key, marker = sys.argv[1:]
with ArtifactStore(root).lock("crash", key):
    Path(marker).write_text("held", encoding="utf-8")
    os._exit(73)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), key, str(marker)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )

    assert completed.returncode == 73
    assert marker.read_text(encoding="utf-8") == "held"
    assert lock_path.exists()
    started = time.monotonic()
    with store.lock("crash", key, timeout_s=0.5):
        assert lock_path.exists()
    assert time.monotonic() - started < 0.5
    assert lock_path.exists()


def test_live_artifact_owner_cannot_be_stolen(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("contended", {"value": 1}, schema_version="test-v1")
    lock_path = store.path("contended", key, ".lock")

    with store.lock("contended", key):
        assert lock_path.exists()
        with pytest.raises(TimeoutError, match="artifact lock"):
            with store.lock("contended", key, timeout_s=0.05):
                raise AssertionError("a contender stole the live inode")

    assert lock_path.exists()


def test_rendezvous_contents_and_age_never_claim_ownership(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("legacy", {"value": 1}, schema_version="test-v1")
    lock_path = store.path("legacy", key, ".lock")
    lock_path.parent.mkdir(parents=True)
    snapshot = b"{legacy-or-malformed-owner-metadata"
    lock_path.write_bytes(snapshot)
    old = time.time() - 86_400.0
    os.utime(lock_path, (old, old))

    with store.lock("legacy", key, timeout_s=0.5, stale_after_s=0.01):
        assert lock_path.exists()
    assert lock_path.read_bytes() == snapshot


def test_distinct_artifact_keys_do_not_serialize(tmp_path):
    store = ArtifactStore(tmp_path)
    first = artifact_key("distinct", {"value": 1}, schema_version="test-v1")
    second = artifact_key("distinct", {"value": 2}, schema_version="test-v1")

    with store.lock("distinct", first):
        with store.lock("distinct", second, timeout_s=0.05):
            pass


def test_artifact_store_npz_detects_tampering(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("arrays", {"shape": [2, 3]}, schema_version="test-v1")
    path = store.save_npz(
        "arrays", key, {"values": np.arange(6, dtype=np.float64).reshape(2, 3)}
    )
    loaded = store.load_npz("arrays", key)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["values"], np.arange(6).reshape(2, 3))

    path.write_bytes(path.read_bytes()[:20])
    assert store.load_npz("arrays", key) is None


def test_artifact_store_npz_rejects_nonobject_metadata_and_reserved_name(tmp_path):
    store = ArtifactStore(tmp_path)
    key = artifact_key("arrays", {"malformed": True}, schema_version="test-v1")
    path = store.path("arrays", key, ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=np.arange(3),
        __metadata__=np.frombuffer(b"[]", dtype=np.uint8),
    )

    assert store.load_npz("arrays", key) is None
    with pytest.raises(ValueError, match="reserved"):
        store.save_npz("arrays", key, {"__metadata__": np.arange(1)})

    for ambiguous_name in ("foo.npy", "foo/bar", r"foo\bar", "1foo", ""):
        with pytest.raises(ValueError, match="unambiguous ASCII identifiers"):
            store.save_npz("arrays", key, {ambiguous_name: np.arange(1)})


def test_benchmark_provenance_rejects_untracked_symlink_escape(
    tmp_path, monkeypatch
):
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tests"], cwd=repo, check=True
    )
    (repo / "tracked.txt").write_text("tracked", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

    external = tmp_path / "external.txt"
    external.write_text("must-not-be-read", encoding="utf-8")
    try:
        (repo / "linked.txt").symlink_to(external)
    except OSError:
        pytest.skip("creating a file symlink is unavailable on this Windows host")
    monkeypatch.setattr(benchmark, "_REPO", repo)

    with pytest.raises(ValueError, match="symlink"):
        benchmark._git_provenance()


def test_benchmark_provenance_rejects_a_never_stable_worktree(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    diffs = iter((b"diff-1", b"diff-2", b"diff-3", b"diff-4"))

    def fake_git(argv, **kwargs):
        operation = tuple(argv[1:])
        if operation == ("rev-parse", "HEAD"):
            stdout = b"a" * 40 + b"\n"
        elif operation[:1] == ("status",):
            stdout = b" M changing.py\n"
        elif operation[:1] == ("diff",):
            stdout = next(diffs)
        elif operation[:1] == ("ls-files",):
            stdout = b""
        else:  # pragma: no cover - the exact Git surface is part of the test
            raise AssertionError(operation)
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(benchmark, "_REPO", tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_git)
    with pytest.raises(RuntimeError, match="worktree changed"):
        benchmark._git_provenance()


@pytest.mark.parametrize(
    "failed_operation",
    [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain=v1", "--untracked-files=all"),
        ("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD"),
        ("ls-files", "--others", "--exclude-standard", "-z"),
    ],
    ids=("commit", "status", "tracked-diff", "untracked-list"),
)
def test_benchmark_provenance_fails_closed_when_any_git_read_fails(
    tmp_path, monkeypatch, failed_operation
):
    from types import SimpleNamespace

    outputs = {
        ("rev-parse", "HEAD"): b"a" * 40 + b"\n",
        ("status", "--porcelain=v1", "--untracked-files=all"): b"",
        (
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
        ): b"",
        ("ls-files", "--others", "--exclude-standard", "-z"): b"",
    }

    def fake_git(argv, **kwargs):
        operation = tuple(argv[1:])
        assert operation in outputs
        return SimpleNamespace(
            returncode=7 if operation == failed_operation else 0,
            stdout=outputs[operation],
        )

    monkeypatch.setattr(benchmark, "_REPO", tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_git)
    with pytest.raises(RuntimeError, match="could not capture benchmark provenance"):
        benchmark._git_provenance()


def test_benchmark_provenance_rejects_noncanonical_commit_identity(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    def fake_git(argv, **kwargs):
        operation = tuple(argv[1:])
        stdout = b"a\n" if operation == ("rev-parse", "HEAD") else b""
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(benchmark, "_REPO", tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_git)
    with pytest.raises(RuntimeError, match="exact Git commit identity"):
        benchmark._git_provenance()


def test_artifact_key_changes_with_schema_config_and_environment():
    base = artifact_key(
        "layer", {"gain": 1.0}, schema_version="v1", environment={"numpy": "1"}
    )
    assert base != artifact_key(
        "layer", {"gain": 1.1}, schema_version="v1", environment={"numpy": "1"}
    )
    assert base != artifact_key(
        "layer", {"gain": 1.0}, schema_version="v2", environment={"numpy": "1"}
    )
    assert base != artifact_key(
        "layer", {"gain": 1.0}, schema_version="v1", environment={"numpy": "2"}
    )


def test_artifact_key_rejects_stringification_key_aliases():
    with pytest.raises(TypeError, match="exact string keys"):
        artifact_key(
            "ambiguous",
            {1: "integer key", "1": "string key"},
            schema_version="v1",
        )


def test_numeric_thread_environment_invalidates_artifacts(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    first_environment = dependency_fingerprint()
    first = artifact_key(
        "numeric", {}, schema_version="v1", environment=first_environment
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    second_environment = dependency_fingerprint()
    second = artifact_key(
        "numeric", {}, schema_version="v1", environment=second_environment
    )
    assert first_environment["numpy_build"]
    assert first_environment["numeric_thread_environment"]["OMP_NUM_THREADS"] == "1"
    assert first != second


def test_source_content_change_invalidates_rollout_artifact(tmp_path):
    source = tmp_path / "sequencer.py"
    source.write_text("ALGORITHM = 1\n", encoding="utf-8")
    first = artifact_key(
        "benchmark-results", {}, schema_version="v2", source_files=[source]
    )
    source.write_text("ALGORITHM = 2\n", encoding="utf-8")
    second = artifact_key(
        "benchmark-results", {}, schema_version="v2", source_files=[source]
    )
    assert first != second
    rollout_sources = benchmark._benchmark_result_source_files({"use_residual": False})
    repo = Path(benchmark.__file__).resolve().parents[1]
    assert repo / "gate_sequencing" / "sequencer.py" in rollout_sources
    assert repo / "competition" / "adapter.py" in rollout_sources


def test_same_path_ilc_table_content_and_loader_are_trajectory_key_sources(tmp_path):
    from planning.trajectory_optimizer import PlannerConfig

    table = tmp_path / "ilc-table.json"
    table.write_text('{"schema_version":1,"samples":[]}', encoding="utf-8")
    planner = PlannerConfig(ilc_table_path=str(table))
    sources = benchmark._trajectory_source_files(planner)
    repo = Path(benchmark.__file__).resolve().parents[1]
    assert repo / "planning" / "ilc_runtime.py" in sources
    assert table.resolve() in sources
    first = artifact_key(
        "trajectories", {}, schema_version="trajectory-test", source_files=sources
    )
    table.write_text(
        '{"schema_version":1,"samples":[{"t":0,"ff_acc":[0,0,0]}]}',
        encoding="utf-8",
    )
    second = artifact_key(
        "trajectories", {}, schema_version="trajectory-test", source_files=sources
    )
    assert first != second


def test_unknown_dataclass_override_is_rejected():
    from planning.racing_line import RacingLineConfig

    with pytest.raises(ValueError, match="unknown RacingLineConfig override key"):
        benchmark._dataclass_from_overrides(RacingLineConfig, {"max_lateral_ofset": 0.2})


@pytest.mark.parametrize("config", [{}, 0, False])
def test_racing_line_optimizer_rejects_falsey_wrong_type_config(config):
    from planning.racing_line import RacingLineOptimizer

    with pytest.raises(TypeError, match="RacingLineConfig"):
        RacingLineOptimizer(config)


@pytest.mark.parametrize(
    ("config_class", "overrides", "error"),
    [
        pytest.param(
            "DroneConstraints", {"max_velocity": True}, TypeError, id="drone-bool"
        ),
        pytest.param(
            "DroneConstraints",
            {"max_tilt_angle": math.pi},
            ValueError,
            id="drone-tilt-range",
        ),
        pytest.param(
            "PlannerConfig", {"lookahead_s": "0.3"}, TypeError, id="planner-string"
        ),
        pytest.param(
            "PlannerConfig",
            {"max_compression_easy": 0.0},
            ValueError,
            id="planner-compression-range",
        ),
        pytest.param(
            "SequencerConfig",
            {"detection_dropout_frames": 1.0},
            TypeError,
            id="sequencer-float-int",
        ),
        pytest.param(
            "SequencerConfig",
            {"recovery_speed_factor": 1.1},
            ValueError,
            id="sequencer-recovery-range",
        ),
        pytest.param(
            "TrackerConfig", {"use_drag_ff": 1}, TypeError, id="tracker-int-bool"
        ),
        pytest.param(
            "TrackerConfig", {"sim_roll_sign": 0}, ValueError, id="tracker-roll-sign"
        ),
        pytest.param(
            "TrackerConfig",
            {"min_thrust_normalized": 0.95},
            ValueError,
            id="tracker-thrust-order",
        ),
    ],
)
def test_resolved_dataclass_configs_reject_coercion_and_invalid_ranges(
    config_class, overrides, error
):
    from control.mpc_tracker import TrackerConfig
    from gate_sequencing.sequencer import SequencerConfig
    from planning.trajectory_optimizer import DroneConstraints, PlannerConfig

    classes = {
        "DroneConstraints": DroneConstraints,
        "PlannerConfig": PlannerConfig,
        "SequencerConfig": SequencerConfig,
        "TrackerConfig": TrackerConfig,
    }
    with pytest.raises(error):
        benchmark._dataclass_from_overrides(classes[config_class], overrides)


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        ({"max_lateral_offset": float("nan")}, ValueError),
        ({"max_vertical_offset": -0.1}, ValueError),
        ({"corner_cut_aggressiveness": 1.1}, ValueError),
        ({"speed_weight": 0.0, "smoothness_weight": 0.0}, ValueError),
        ({"lookahead_gates": True}, ValueError),
        ({"use_cache": 1}, TypeError),
        ({"cache_root": Path("cache")}, TypeError),
    ),
)
def test_racing_line_config_rejects_coerced_or_invalid_policy(overrides, error):
    from planning.racing_line import RacingLineConfig

    with pytest.raises(error):
        RacingLineConfig(**overrides)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        pytest.param(True, TypeError, id="bool"),
        pytest.param("0.01", TypeError, id="string"),
        pytest.param(0.0, ValueError, id="zero"),
        pytest.param(-0.01, ValueError, id="negative"),
        pytest.param(math.nan, ValueError, id="nan"),
        pytest.param(math.inf, ValueError, id="infinite"),
    ],
)
def test_prepare_course_dt_is_exact_finite_and_positive(tmp_path, value, error):
    with pytest.raises(error, match="dt"):
        benchmark.prepare_course(_one_gate_course(), dt=value, cache_root=tmp_path)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        ("4.0", TypeError),
        (0.0, ValueError),
        (math.nan, ValueError),
    ],
)
def test_prepare_course_max_velocity_is_exact_finite_and_positive(
    tmp_path, value, error
):
    course = _one_gate_course()
    course["max_velocity_mps"] = value
    with pytest.raises(error):
        benchmark.prepare_course(course, cache_root=tmp_path)


def test_gate_inputs_preserve_3d_geometry_and_reject_duplicate_identity():
    course = _one_gate_course()
    gate = course["gates"][0]
    gate["pose"].update({"pitch": 0.2, "roll": -0.3})
    gate["config"] = {"depth_m": 0.4, "border_width_m": 0.25}
    specs, waypoints, start = benchmark._gate_inputs(course)

    assert specs[0].pitch == pytest.approx(0.2)
    assert specs[0].roll == pytest.approx(-0.3)
    assert specs[0].depth == pytest.approx(0.4)
    assert specs[0].border_width == pytest.approx(0.25)
    assert np.linalg.norm(waypoints[0].normal) == pytest.approx(1.0)
    np.testing.assert_array_equal(start, [0.0, 0.0, 1.5])

    duplicate = _one_gate_course()
    duplicate["gates"].append(dict(duplicate["gates"][0]))
    with pytest.raises(ValueError, match="duplicate gate id"):
        benchmark._gate_inputs(duplicate)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("id", True, TypeError),
        ("sequence_index", False, TypeError),
        ("pose.x", "2.0", TypeError),
        ("config.interior_width_m", 0.0, ValueError),
    ],
)
def test_gate_inputs_reject_coerced_or_nonphysical_geometry(field, value, error):
    course = _one_gate_course()
    gate = course["gates"][0]
    if field == "id":
        gate["id"] = value
    elif field == "sequence_index":
        gate["sequence_index"] = value
    elif field == "pose.x":
        gate["pose"]["x"] = value
    else:
        gate.setdefault("config", {})["interior_width_m"] = value
    with pytest.raises(error):
        benchmark._gate_inputs(course)


@pytest.mark.parametrize("nested", ["planner", "racing_line", "sequencer"])
def test_prepare_course_rejects_nonmapping_track_overrides(tmp_path, nested):
    course = _one_gate_course()
    course[nested] = []
    with pytest.raises(TypeError, match=nested):
        benchmark.prepare_course(course, cache_root=tmp_path)


def _valid_ilc_global():
    return {
        "alpha": 0.4,
        "max_iterations": 5,
        "smoothing_sigma": 10.0,
        "max_correction_m": 0.15,
        "convergence_threshold": 0.002,
        "filter_cutoff_hz": 0.35,
        "momentum_gamma": 0.0,
        "blend_steps": 50,
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("alpha", "0.4", TypeError),
        ("alpha", 1.1, ValueError),
        ("max_iterations", True, TypeError),
        ("max_iterations", 0, ValueError),
        ("filter_cutoff_hz", 0.0, ValueError),
        ("momentum_gamma", 1.0, ValueError),
    ],
)
def test_ilc_global_config_rejects_coercion_and_invalid_ranges(field, value, error):
    config = _valid_ilc_global()
    config[field] = value
    with pytest.raises(error):
        benchmark._normalize_ilc_global(config)


@pytest.mark.parametrize(
    ("sections", "section_format", "error"),
    [
        ([['0', 1.0, 0.3]], "fractions", TypeError),
        ([[0.0, 1.1, 0.3]], "fractions", ValueError),
        ([[0.0, 1.0, True]], "fractions", TypeError),
        ([[0.0, 1.0, 0.3, -0.1]], "fractions", ValueError),
        ([[0.0, 1.0, 0.3, 0.1, 0.0]], "fractions", ValueError),
        ([[0.0, 10, 0.3]], "steps", TypeError),
        ([[0, 101, 0.3]], "steps", ValueError),
    ],
)
def test_ilc_section_config_rejects_coercion_and_invalid_ranges(
    sections, section_format, error
):
    with pytest.raises(error):
        benchmark._normalize_ilc_section_overrides(
            sections, override_format=section_format, total_steps=100
        )


def test_tracked_fractional_ilc_sections_resolve_to_exact_step_ranges():
    track = json.loads(
        (Path(benchmark.__file__).resolve().parents[1]
         / "sim_pybullet" / "configs" / "race_01.json").read_text(encoding="utf-8")
    )
    sections = benchmark._normalize_ilc_section_overrides(
        track["ilc_section_overrides"],
        override_format=track["ilc_section_overrides_format"],
        total_steps=1000,
    )

    assert sections is not None
    assert sections[0][:2] == (0, 146)
    assert sections[-1][:2] == (540, 1000)


def test_simulate_numeric_boundaries_reject_bool_string_and_nonfinite(tmp_path):
    prepared = benchmark.prepare_course(
        _one_gate_course(), dt=0.02, cache_root=tmp_path
    )
    invalid_dt = [
        (True, TypeError),
        ("0.02", TypeError),
        (0.0, ValueError),
        (-0.02, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
    ]
    for value, error in invalid_dt:
        with pytest.raises(error, match="dt"):
            benchmark.simulate(prepared, dt=value, duration=0.0)

    invalid_duration = [
        (True, TypeError),
        ("1", TypeError),
        (-1.0, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
    ]
    for value, error in invalid_duration:
        with pytest.raises(error, match="duration"):
            benchmark.simulate(prepared, dt=0.02, duration=value)

    for value, error in ((True, TypeError), (1.5, TypeError), (-1, ValueError)):
        with pytest.raises(error, match="seed"):
            benchmark.simulate(prepared, dt=0.02, duration=0.0, seed=value)


def test_synthetic_tracker_uses_conservative_modeled_drag_compensation():
    from competition.drone_spec import DEFAULT_LINEAR_DRAG_PER_MASS
    from planning.trajectory_optimizer import DroneConstraints

    resolved = benchmark._resolved_tracker_config({}, None, DroneConstraints())
    assert resolved.use_drag_ff is True
    assert resolved.drag_ff_coeff == pytest.approx(
        0.9 * DEFAULT_LINEAR_DRAG_PER_MASS
    )

    explicit = benchmark._resolved_tracker_config(
        {}, {"use_drag_ff": False, "drag_ff_coeff": 0.0}, DroneConstraints()
    )
    assert explicit.use_drag_ff is False
    assert explicit.drag_ff_coeff == 0.0


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        pytest.param("duration", True, TypeError, id="duration-bool"),
        pytest.param("duration", "1", TypeError, id="duration-string"),
        pytest.param("duration", -1.0, ValueError, id="duration-negative"),
        pytest.param("duration", math.nan, ValueError, id="duration-nan"),
        pytest.param("dt", False, TypeError, id="dt-bool"),
        pytest.param("dt", "0.01", TypeError, id="dt-string"),
        pytest.param("dt", 0.0, ValueError, id="dt-zero"),
        pytest.param("dt", math.inf, ValueError, id="dt-infinite"),
        pytest.param("max_workers", True, TypeError, id="workers-bool"),
        pytest.param("max_workers", 1.5, TypeError, id="workers-float"),
        pytest.param("max_workers", "1", TypeError, id="workers-string"),
        pytest.param("max_workers", 0, ValueError, id="workers-zero"),
        pytest.param("record_position_trace", 1, TypeError, id="trace-int"),
        pytest.param("include_results", "yes", TypeError, id="results-string"),
        pytest.param("use_result_cache", 1, TypeError, id="cache-int"),
    ],
)
def test_matrix_numeric_boundaries_are_exact(keyword, value, error):
    with pytest.raises(error, match=keyword):
        benchmark_matrix.run_matrix([], **{keyword: value})


def test_racing_line_semantic_corruption_is_replaced(tmp_path, monkeypatch):
    import planning.racing_line as racing_line
    from planning.trajectory_optimizer import GateWaypoint

    gates = [
        GateWaypoint((2.0, 0.0, 1.5), (1.0, 0.0, 0.0)),
        GateWaypoint((4.0, 0.0, 1.5), (1.0, 0.0, 0.0)),
    ]
    config = racing_line.RacingLineConfig(cache_root=str(tmp_path))
    optimizer = racing_line.RacingLineOptimizer(config)
    key = optimizer._compute_cache_key(gates, (0.0, 0.0, 1.5), config)
    ArtifactStore(tmp_path).save_json(
        "racing-lines",
        key,
        {"schema": racing_line.RACING_LINE_CACHE_SCHEMA, "offsets": [0.0]},
    )
    monkeypatch.setattr(
        racing_line,
        "minimize",
        lambda objective, x0, **kwargs: SimpleNamespace(
            x=np.zeros_like(x0), fun=float(objective(np.zeros_like(x0)))
        ),
    )
    monkeypatch.setattr(
        racing_line.RacingLineOptimizer,
        "_select_by_sim",
        lambda self, gates, results, start: 0,
    )

    optimizer.optimize(gates, (0.0, 0.0, 1.5))
    rebuilt = optimizer._load_cache(key, str(tmp_path), expected_size=4)
    assert rebuilt is not None
    assert rebuilt.shape == (4,)

    # Integrity-valid JSON is still untrusted: offsets outside the optimizer
    # bounds must be rejected and rebuilt under the same content address.
    ArtifactStore(tmp_path).save_json(
        "racing-lines",
        key,
        {
            "schema": racing_line.RACING_LINE_CACHE_SCHEMA,
            "offsets": [1000.0, -1000.0, 1.0, -1.0],
        },
    )
    second_optimizer = racing_line.RacingLineOptimizer(config)
    second_optimizer.optimize(gates, (0.0, 0.0, 1.5))
    assert second_optimizer.last_cache_hit is False
    bounded = second_optimizer._load_cache(
        key,
        str(tmp_path),
        expected_size=4,
        expected_lateral_bound=config.max_lateral_offset,
        expected_vertical_bound=config.max_vertical_offset,
    )
    assert bounded is not None
    assert np.all(np.abs(bounded[:2]) <= config.max_lateral_offset)
    assert np.all(np.abs(bounded[2:]) <= config.max_vertical_offset)


def test_racing_line_uses_same_key_winner_published_during_cold_compute(
    tmp_path, monkeypatch
):
    import planning.racing_line as racing_line
    from planning.trajectory_optimizer import GateWaypoint

    gates = [
        GateWaypoint((2.0, 0.0, 1.5), (1.0, 0.0, 0.0)),
        GateWaypoint((4.0, 0.0, 1.5), (1.0, 0.0, 0.0)),
    ]
    config = racing_line.RacingLineConfig(cache_root=str(tmp_path))
    optimizer = racing_line.RacingLineOptimizer(config)
    computed = np.array([0.4, 0.4, 0.0, 0.0])
    published = np.array([-0.3, -0.2, 0.0, 0.0])
    loads = iter((None, published))
    monkeypatch.setattr(optimizer, "_load_cache", lambda *args, **kwargs: next(loads))
    monkeypatch.setattr(
        racing_line,
        "minimize",
        lambda objective, x0, **kwargs: SimpleNamespace(x=computed.copy(), fun=0.0),
    )
    monkeypatch.setattr(optimizer, "_select_by_sim", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        optimizer,
        "_save_cache",
        lambda *args, **kwargs: pytest.fail("published winner must not be overwritten"),
    )

    optimized = optimizer.optimize(gates, (0.0, 0.0, 1.5))
    expected_positions = optimizer._apply_offsets(gates, published)
    np.testing.assert_allclose(
        np.asarray([gate.position for gate in optimized]),
        np.asarray(expected_positions),
        rtol=0.0,
        atol=0.0,
    )


def test_same_key_cold_racing_line_is_materialized_only_once(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import planning.racing_line as racing_line
    from planning.trajectory_optimizer import GateWaypoint

    gates = [
        GateWaypoint((2.0, 0.0, 1.5), (1.0, 0.0, 0.0)),
        GateWaypoint((4.0, 0.0, 1.5), (1.0, 0.0, 0.0)),
    ]
    config = racing_line.RacingLineConfig(cache_root=str(tmp_path))
    solve_entered = threading.Event()
    allow_publish = threading.Event()
    count_lock = threading.Lock()
    solve_count = 0

    def fake_uncached(self, candidate_gates, start_position):
        nonlocal solve_count
        assert self.config.use_cache is False
        with count_lock:
            solve_count += 1
        solve_entered.set()
        assert allow_publish.wait(timeout=2.0)
        offsets = np.zeros(len(candidate_gates) * 2, dtype=np.float64)
        self._last_selected_offsets = offsets
        self._last_candidate_count = 1
        self._last_selected_idx = 0
        return self._gates_with_offsets(candidate_gates, offsets)

    monkeypatch.setattr(
        racing_line.RacingLineOptimizer, "_optimize_impl", fake_uncached
    )
    first = racing_line.RacingLineOptimizer(config)
    second = racing_line.RacingLineOptimizer(config)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first.optimize, gates, (0.0, 0.0, 1.5))
        assert solve_entered.wait(timeout=2.0)
        second_future = executor.submit(second.optimize, gates, (0.0, 0.0, 1.5))
        allow_publish.set()
        first_result = first_future.result(timeout=3.0)
        second_result = second_future.result(timeout=3.0)

    assert solve_count == 1
    assert sorted((first.last_cache_hit, second.last_cache_hit)) == [False, True]
    np.testing.assert_allclose(
        np.asarray([gate.position for gate in first_result]),
        np.asarray([gate.position for gate in second_result]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.benchmark
def test_prepared_course_keys_include_config_but_not_cache_location(tmp_path):
    course = _one_gate_course()
    first = benchmark.prepare_course(course, cache_root=tmp_path / "one", dt=0.02)
    second = benchmark.prepare_course(course, cache_root=tmp_path / "two", dt=0.02)
    changed = benchmark.prepare_course(
        _one_gate_course(planner_entry=0.55),
        cache_root=tmp_path / "three",
        dt=0.02,
    )

    assert first.config_hash == second.config_hash
    assert first.artifact_key == second.artifact_key
    assert first.artifact_keys["trajectory"] != changed.artifact_keys["trajectory"]


@pytest.mark.benchmark
def test_evaluator_forces_plane_crossing_and_tracks_legacy_request_in_config(tmp_path):
    first = benchmark.prepare_course(
        _one_gate_course(proximity=0.25), cache_root=tmp_path, dt=0.02
    )
    second = benchmark.prepare_course(
        _one_gate_course(proximity=1.25), cache_root=tmp_path, dt=0.02
    )

    assert first.sequencer_config.proximity_pass_distance == 0.0
    assert second.sequencer_config.proximity_pass_distance == 0.0
    # The effective validation policy is identical, so deterministic artifacts
    # are reusable.  Provenance still records the differing raw track request.
    assert first.artifact_keys["plan_validation"] == second.artifact_keys["plan_validation"]
    assert first.config_hash != second.config_hash


@pytest.mark.benchmark
def test_semantically_invalid_validation_ilc_and_result_rebuild(tmp_path):
    course = _one_gate_course()
    initial = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    store = ArtifactStore(tmp_path)
    trajectory_arrays = store.load_npz(
        "trajectories", initial["artifact_hashes"]["trajectory"]
    )
    assert trajectory_arrays is not None
    trajectory_arrays["gate_width"] = np.asarray([-1.0], dtype=np.float64)
    store.save_npz(
        "trajectories",
        initial["artifact_hashes"]["trajectory"],
        trajectory_arrays,
    )
    contradictory_validation = store.load_json(
        "plan-validation", initial["artifact_hashes"]["plan_validation"]
    )
    assert contradictory_validation is not None
    contradictory_validation.update(
        {
            "ok": True,
            "gates_passed": contradictory_validation["total_gates"],
            "crashed": True,
            "last_crash_gate": "gate-1",
        }
    )
    store.save_json(
        "plan-validation",
        initial["artifact_hashes"]["plan_validation"],
        contradictory_validation,
    )
    store.save_npz(
        "ilc",
        initial["artifact_hashes"]["ilc"],
        {
            "present": np.asarray([1], dtype=np.uint8),
            # Integrity-valid and shape-valid, but semantically truncated.
            "position_offsets": np.asarray([[0.01, 0.02, 0.03]]),
            "velocity_offsets": np.asarray([[0.01, 0.02, 0.03]]),
        },
    )
    contradictory_result = store.load_json(
        "benchmark-results", initial["artifact_hashes"]["benchmark_result"]
    )
    assert contradictory_result is not None
    contradictory_result.update(
        {
            "sim_passed": True,
            "crashed": True,
            "safety_passed": True,
            "gate_pass_rate": 1.5,
        }
    )
    store.save_json(
        "benchmark-results",
        initial["artifact_hashes"]["benchmark_result"],
        contradictory_result,
    )

    rebuilt = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    assert rebuilt["cache"]["trajectory"] == "miss"
    assert rebuilt["cache"]["plan_validation"] == "miss"
    assert rebuilt["cache"]["ilc"] == "miss"
    assert rebuilt["cache"]["benchmark_result"] == "miss"
    assert benchmark._valid_cached_benchmark_result(
        store.load_json(
            "benchmark-results", rebuilt["artifact_hashes"]["benchmark_result"]
        )
    )


def test_cached_fabricated_pass_cannot_violate_threshold_metrics(tmp_path):
    course = _one_gate_course()
    initial = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    store = ArtifactStore(tmp_path)
    key = initial["artifact_hashes"]["benchmark_result"]
    fabricated = store.load_json("benchmark-results", key)
    assert fabricated is not None
    assert fabricated["plan_validation"]["ok"] is True
    total_gates = fabricated["total_gates"]
    fabricated.update(
        {
            "termination_reason": "race_complete",
            "gates_passed": total_gates,
            "gate_pass_rate": 1.0,
            "complete": True,
            "crashed": False,
            "disqualified": False,
            "dq_reason": None,
            "last_crash_gate": None,
            "sim_passed": True,
            "safety_passed": True,
            "validity_passed": True,
            "avg_tracking_error_m": 100.0,
            "max_tracking_error_m": 100.0,
            "threshold_failures": [],
            "completion": {
                "complete": True,
                "gates_passed": total_gates,
                "total_gates": total_gates,
            },
        }
    )
    fabricated["failure_summary"]["threshold_failures"] = []
    store.save_json("benchmark-results", key, fabricated)

    rebuilt = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    assert rebuilt["cache"]["benchmark_result"] == "miss"
    assert rebuilt["sim_passed"] is False
    assert rebuilt["avg_tracking_error_m"] < 100.0


def test_cached_result_rejects_terminal_and_trace_contradictions(tmp_path):
    initial = benchmark.run_synthetic_benchmark(
        duration=0.1,
        dt=0.02,
        config=_one_gate_course(),
        cache_root=tmp_path,
    )
    store = ArtifactStore(tmp_path)
    payload = store.load_json(
        "benchmark-results", initial["artifact_hashes"]["benchmark_result"]
    )
    assert benchmark._valid_cached_benchmark_result(payload)

    terminal = json.loads(json.dumps(payload))
    terminal["termination_reason"] = "race_complete"
    assert terminal["complete"] is False
    assert not benchmark._valid_cached_benchmark_result(terminal)

    tracker_trace = json.loads(json.dumps(payload))
    tracker_trace["tracker_feature_trace"] = [
        "malformed"
    ] * tracker_trace["total_steps"]
    assert not benchmark._valid_cached_benchmark_result(tracker_trace)

    position_trace = json.loads(json.dumps(payload))
    position_trace["position_trace"] = [{}] * position_trace["total_steps"]
    assert not benchmark._valid_cached_benchmark_result(position_trace)

    crash_identity = json.loads(json.dumps(payload))
    crash_identity["last_crash_gate"] = "gate-1"
    assert not benchmark._valid_cached_benchmark_result(crash_identity)

    missing_pass_evidence = json.loads(json.dumps(payload))
    missing_pass_evidence.pop("gate_pass_times")
    assert not benchmark._valid_cached_benchmark_result(missing_pass_evidence)


def test_requested_cached_traces_cannot_be_silently_dropped(tmp_path):
    course = _one_gate_course()
    kwargs = {
        "duration": 0.1,
        "dt": 0.02,
        "config": course,
        "cache_root": tmp_path,
        "record_position_trace": True,
        "tracker_config_overrides": {"trace_features": True},
    }
    initial = benchmark.run_synthetic_benchmark(**kwargs)
    assert initial["total_steps"] > 0
    assert len(initial["position_trace"]) == initial["total_steps"]
    assert len(initial["tracker_feature_trace"]) == initial["total_steps"]
    store = ArtifactStore(tmp_path)
    key = initial["artifact_hashes"]["benchmark_result"]
    corrupted = store.load_json("benchmark-results", key)
    assert corrupted is not None
    corrupted["position_trace"] = []
    corrupted["tracker_feature_trace"] = []
    store.save_json("benchmark-results", key, corrupted)

    rebuilt = benchmark.run_synthetic_benchmark(**kwargs)

    assert rebuilt["cache"]["benchmark_result"] == "miss"
    assert rebuilt["rollout_executed"] is True
    assert len(rebuilt["position_trace"]) == rebuilt["total_steps"]
    assert len(rebuilt["tracker_feature_trace"]) == rebuilt["total_steps"]


def test_cached_result_requires_exact_ordered_gate_pass_evidence(tmp_path):
    course = _one_gate_course()
    course["gates"].append(
        {
            "id": "gate-2",
            "sequence_index": 2,
            "pose": {"x": 4.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        }
    )
    result = benchmark.run_synthetic_benchmark(
        duration=5.0,
        dt=0.02,
        config=course,
        cache_root=tmp_path,
    )
    store = ArtifactStore(tmp_path)
    payload = store.load_json(
        "benchmark-results", result["artifact_hashes"]["benchmark_result"]
    )
    expected_ids = ["gate-1", "gate-2"]
    assert benchmark._valid_cached_benchmark_result(
        payload, expected_gate_ids=expected_ids
    )

    wrong_order = json.loads(json.dumps(payload))
    wrong_order["gate_pass_times"][0]["gate_id"] = "gate-2"
    assert not benchmark._valid_cached_benchmark_result(
        wrong_order, expected_gate_ids=expected_ids
    )

    duplicate_time = json.loads(json.dumps(payload))
    if len(duplicate_time["gate_pass_times"]) >= 2:
        duplicate_time["gate_pass_times"][1]["time_s"] = duplicate_time[
            "gate_pass_times"
        ][0]["time_s"]
        assert not benchmark._valid_cached_benchmark_result(
            duplicate_time, expected_gate_ids=expected_ids
        )


@pytest.mark.benchmark
def test_simulate_uses_local_seeded_rng_and_resolved_drone_limits(tmp_path):
    course = _one_gate_course()
    normal = benchmark.prepare_course(course, cache_root=tmp_path / "normal", dt=0.02)
    constrained = benchmark.prepare_course(
        course,
        planning_config={"drone": {"max_acceleration": 0.01}},
        cache_root=tmp_path / "constrained",
        dt=0.02,
    )
    first = benchmark.simulate(
        normal, seed=17, duration=0.2, record_position_trace=True
    )
    second = benchmark.simulate(
        normal, seed=17, duration=0.2, record_position_trace=True
    )
    assert first["position_trace"] == second["position_trace"]
    assert first["gate_pass_times"] == second["gate_pass_times"]

    np.random.seed(1234)
    expected_global_sample = np.random.random()
    np.random.seed(1234)
    benchmark.simulate(normal, seed=99, duration=0.02)
    assert np.random.random() == expected_global_sample

    slow = benchmark.simulate(
        constrained, seed=17, duration=0.2, record_position_trace=True
    )
    normal_peak_speed = max(
        np.linalg.norm(sample["vel"]) for sample in first["position_trace"]
    )
    constrained_peak_speed = max(
        np.linalg.norm(sample["vel"]) for sample in slow["position_trace"]
    )
    assert constrained.drone_constraints.max_acceleration == 0.01
    assert constrained_peak_speed < normal_peak_speed * 0.1


@pytest.mark.benchmark
def test_synthetic_state_and_terminal_timestamps_match_integration_time(tmp_path):
    normal = benchmark.prepare_course(
        _one_gate_course(), cache_root=tmp_path / "normal-time", dt=0.02
    )
    timed = benchmark.simulate(
        normal,
        seed=17,
        duration=0.04,
        dt=0.02,
        record_position_trace=True,
    )
    assert [sample["t"] for sample in timed["position_trace"]] == [0.02, 0.04]
    assert timed["controller_trace_summary"]["samples"] == 2
    assert timed["sim_time_s"] == 0.04

    ground_course = _one_gate_course()
    ground_course["start"] = {"position": [0.0, 0.0, 0.0]}
    grounded = benchmark.prepare_course(
        ground_course, cache_root=tmp_path / "ground-time", dt=0.02
    )
    terminal = benchmark.simulate(
        grounded,
        seed=17,
        duration=0.04,
        dt=0.02,
        record_position_trace=True,
    )
    assert terminal["termination_reason"] == "crash_ground"
    assert terminal["sim_time_s"] == 0.0
    assert terminal["position_trace"] == []
    assert terminal["controller_trace_summary"] == {}


@pytest.mark.benchmark
def test_cold_warm_metrics_match_and_corrupt_trajectory_rebuilds(tmp_path):
    course = _one_gate_course()
    cold = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    warm = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    # Compare the complete deterministic result/evidence payload.  Only cache
    # mechanics, wall-clock observations, and their derived timing fields are
    # expected to differ between a cold rollout and a result-cache hit.
    execution_metadata = {
        "cache",
        "cache_hit_or_miss",
        "phase_timings_s",
        "rollout_executed",
        "rollout_wall_time_s",
        "timestamp",
        "timing_consistency",
        "wall_time_s",
    }

    def deterministic_payload(result):
        return {
            key: value
            for key, value in result.items()
            if key not in execution_metadata
        }

    assert deterministic_payload(cold) == deterministic_payload(warm)
    assert cold["cache_hit_or_miss"] == "miss"
    assert warm["cache_hit_or_miss"] == "hit"
    assert cold["rollout_wall_time_s"] == cold["rollout_materialization_wall_time_s"]
    assert warm["rollout_wall_time_s"] == 0.0
    assert warm["rollout_materialization_wall_time_s"] == cold[
        "rollout_materialization_wall_time_s"
    ]
    assert warm["evaluator_version"] == "synthetic-v4-exact-state-time"
    assert warm["comparison_series"] == "prepared-benchmark-v4-exact-state-time"
    assert warm["cache"]["trajectory"] == "hit"
    assert set(warm["phase_timings_s"]) == {
        "startup",
        "config_load",
        "cache_lookup",
        "racing_line",
        "trajectory",
        "plan_validation",
        "ilc",
        "rollout",
        "metrics",
        "total_wall",
    }
    phase_sum = sum(
        value
        for name, value in warm["phase_timings_s"].items()
        if name != "total_wall"
    )
    assert warm["phase_timings_s"]["total_wall"] + 1e-9 >= phase_sum
    assert warm["timing_consistency"]["total_covers_phases"] is True
    assert "interpreter" in warm["timing_scope"]
    assert "checksum verification" in warm["phase_timing_notes"]["cache_lookup"]
    assert warm["validity_passed"] is True
    assert warm["sim_passed"] is False  # valid plan, deliberately truncated rollout
    assert any("race incomplete" in item for item in warm["threshold_failures"])

    trajectory_path = (
        tmp_path / "trajectories" / f"{cold['artifact_hashes']['trajectory']}.npz"
    )
    trajectory_path.write_bytes(b"partial npz")
    # The result entry may still be reusable, but preparation must fail closed
    # and reconstruct its corrupted deterministic dependency.
    rebuilt = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    assert rebuilt["cache"]["trajectory"] == "miss"
    assert ArtifactStore(tmp_path).load_npz(
        "trajectories", cold["artifact_hashes"]["trajectory"]
    ) is not None


@pytest.mark.benchmark
def test_warm_result_uses_current_provenance_not_artifact_producer(tmp_path, monkeypatch):
    producer = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    consumer = {**producer, "dirty": True, "dirty_diff_hash": "e" * 64}
    observations = iter((producer, consumer))
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: next(observations))
    course = _one_gate_course()

    cold = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    warm = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )

    assert cold["cache_hit_or_miss"] == "miss"
    assert warm["cache_hit_or_miss"] == "hit"
    assert cold["code_provenance"] == producer
    assert warm["code_provenance"] == consumer


def test_lower_prepared_cache_layers_bind_benchmark_encoding_source():
    benchmark_source = Path(benchmark.__file__).resolve()
    from planning.trajectory_optimizer import PlannerConfig

    assert benchmark_source in {
        path.resolve() for path in benchmark._trajectory_source_files(PlannerConfig())
    }
    assert benchmark_source in {
        path.resolve() for path in benchmark._validation_source_files()
    }
    assert benchmark_source in {
        path.resolve() for path in benchmark._ilc_source_files()
    }


@pytest.mark.benchmark
def test_cached_result_semantic_contract_rejects_missing_or_malformed_metrics(tmp_path):
    course = _one_gate_course()
    result = benchmark.run_synthetic_benchmark(
        duration=0.1, dt=0.02, config=course, cache_root=tmp_path
    )
    store = ArtifactStore(tmp_path)
    key = result["artifact_hashes"]["benchmark_result"]
    baseline = store.load_json("benchmark-results", key)
    assert baseline is not None

    mutations = (
        lambda payload: payload.pop("trajectory_points"),
        lambda payload: payload.__setitem__("rollout_wall_time_s", "stale"),
        lambda payload: payload.__setitem__("dt", 0.0),
        lambda payload: payload.__setitem__("p95_tracking_error_m", -1.0),
        lambda payload: payload.__setitem__("per_gate_avg_error", []),
        lambda payload: payload.__setitem__("controller_trace_summary", {}),
        lambda payload: payload.__setitem__("resolved_controller_config", {}),
    )
    expected_gate_ids = [
        gate.gate_id
        for gate in sorted(
            benchmark.prepare_course(course, dt=0.02, cache_root=tmp_path).gate_specs,
            key=lambda gate: gate.sequence_index,
        )
    ]
    expected_controller = result["resolved_configuration"]["controller"]
    for mutate in mutations:
        poisoned = json.loads(json.dumps(baseline))
        mutate(poisoned)
        assert not benchmark._valid_cached_benchmark_result(
            poisoned,
            expected_seed=42,
            expected_thresholds=result["thresholds"],
            expected_gate_ids=expected_gate_ids,
            expected_dt=0.02,
            expected_controller_config=expected_controller,
        )


def test_threshold_policy_changes_invalidate_result_cache(tmp_path):
    course = _one_gate_course()
    lenient = benchmark.run_synthetic_benchmark(
        duration=0.1,
        dt=0.02,
        config=course,
        cache_root=tmp_path,
        thresholds={"min_gate_pass_rate": 0.0},
    )
    strict = benchmark.run_synthetic_benchmark(
        duration=0.1,
        dt=0.02,
        config=course,
        cache_root=tmp_path,
        thresholds={"min_gate_pass_rate": 1.0},
    )
    strict_warm = benchmark.run_synthetic_benchmark(
        duration=0.1,
        dt=0.02,
        config=course,
        cache_root=tmp_path,
        thresholds={"min_gate_pass_rate": 1.0},
    )

    assert lenient["artifact_hashes"]["benchmark_result"] != strict[
        "artifact_hashes"
    ]["benchmark_result"]
    assert strict["cache_hit_or_miss"] == "miss"
    assert strict_warm["cache_hit_or_miss"] == "hit"
    assert strict["resolved_configuration"]["runtime"]["thresholds"][
        "min_gate_pass_rate"
    ] == 1.0


def test_same_step_safety_violation_wins_over_final_gate(tmp_path, monkeypatch):
    from gate_sequencing.sequencer import GateSequencer

    prepared = benchmark.prepare_course(
        _one_gate_course(), dt=0.02, cache_root=tmp_path
    )
    prepared.start_position = np.asarray([0.0, 0.0, 0.0], dtype=float)

    def complete_on_update(self, position, *args, **kwargs):
        gate = self._gates[0]
        self._passed.append(gate)
        self._current_idx = len(self._gates)
        return gate

    monkeypatch.setattr(GateSequencer, "update", complete_on_update)
    result = benchmark.simulate(prepared, duration=0.02, dt=0.02)

    assert result["complete"] is True
    assert result["crashed"] is True
    assert result["termination_reason"] == "crash_ground"
    assert result["safety_passed"] is False
    assert result["sim_passed"] is False


def test_matrix_rejects_sim_failure_despite_high_completion(tmp_path, monkeypatch):
    config_path = tmp_path / "production.json"
    config_path.write_text("{}", encoding="utf-8")
    stable_provenance = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: stable_provenance)
    fake_result = {
        "gates_passed": 4,
        "total_gates": 4,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 2.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": False,
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
        "code_provenance": stable_provenance,
    }

    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker",
        lambda task: ("production", {}, fake_result, None),
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_DYNAMIC", raising=False)
    numeric_environment_before = {
        name: os.environ.get(name)
        for name in benchmark_matrix._THREAD_ENV_VARIABLES
    }
    try:
        import cv2

        cv2_threads_before = cv2.getNumThreads()
    except ImportError:
        cv2 = None
        cv2_threads_before = None
    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )
    assert result["all_passed"] is False
    assert any("sim_passed=False" in item for item in result["regressions"])
    fingerprints = result["dependency_fingerprints"]
    assert fingerprints["orchestrator"]["numeric_thread_environment"][
        "OMP_NUM_THREADS"
    ] == "7"
    assert fingerprints["worker_expected"]["numeric_thread_environment"][
        "OMP_NUM_THREADS"
    ] == "1"
    assert fingerprints["worker_expected"]["numeric_thread_environment"][
        "MKL_DYNAMIC"
    ] == "FALSE"
    assert fingerprints["workers_observed"] == [
        {
            "track": "production",
            "fingerprint": fingerprints["worker_expected"],
        }
    ]
    assert result["worker_environment_verified"] is True
    assert result["code_provenance_verified"] is True
    assert result["dependency_fingerprint"] == fingerprints["worker_expected"]
    assert result["resolved_configuration"]["runtime"][
        "worker_dependency_fingerprint"
    ] == fingerprints["worker_expected"]
    assert result["config_hash"] == sha256_json(result["resolved_configuration"])
    assert {
        name: os.environ.get(name)
        for name in benchmark_matrix._THREAD_ENV_VARIABLES
    } == numeric_environment_before
    if cv2 is not None:
        assert cv2.getNumThreads() == cv2_threads_before


def test_matrix_rejects_inconsistent_worker_dependency_fingerprints(
    tmp_path, monkeypatch
):
    config_paths = []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        config_paths.append(path)
    stable_provenance = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: stable_provenance)

    fake_result = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
        "code_provenance": stable_provenance,
    }

    def inconsistent_worker(task):
        name = Path(task[0]).stem
        fingerprint = dependency_fingerprint()
        if name == "second":
            fingerprint = dict(fingerprint)
            numeric_environment = dict(
                fingerprint["numeric_thread_environment"]
            )
            numeric_environment["OMP_NUM_THREADS"] = "2"
            fingerprint["numeric_thread_environment"] = numeric_environment
        return name, {}, fake_result, None, fingerprint

    monkeypatch.setattr(
        benchmark_matrix, "_run_track_worker_with_identity", inconsistent_worker
    )
    result = benchmark_matrix.run_matrix(
        config_paths,
        duration=1.0,
        max_workers=1,
        cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["worker_environment_verified"] is False
    assert result["all_passed"] is False
    assert any(
        "worker dependency fingerprints" in item
        for item in result["regressions"]
    )
    assert len(result["dependency_fingerprints"]["workers_observed"]) == 2


def test_matrix_rejects_worker_code_provenance_drift(tmp_path, monkeypatch):
    config_path = tmp_path / "production.json"
    config_path.write_text("{}", encoding="utf-8")
    start_provenance = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    drifted_provenance = {
        **start_provenance,
        "dirty": True,
        "dirty_diff_hash": "e" * 64,
    }
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: start_provenance)
    fake_result = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
        "code_provenance": drifted_provenance,
    }

    def drifted_worker(task):
        return (
            Path(task[0]).stem,
            {},
            fake_result,
            None,
            dependency_fingerprint(),
        )

    monkeypatch.setattr(
        benchmark_matrix, "_run_track_worker_with_identity", drifted_worker
    )
    result = benchmark_matrix.run_matrix(
        [config_path],
        duration=1.0,
        max_workers=1,
        cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["code_provenance"] == start_provenance
    assert result["resolved_configuration"]["code_provenance"] == start_provenance
    assert result["config_hash"] == sha256_json(result["resolved_configuration"])
    assert result["code_provenance_verified"] is False
    assert result["all_passed"] is False
    assert any("worker code provenance" in item for item in result["regressions"])
    assert result["code_provenance_observations"] == [
        {"track": "production", "code_provenance": drifted_provenance}
    ]


def test_matrix_rejects_transient_worker_provenance_drift(tmp_path, monkeypatch):
    config_path = tmp_path / "production.json"
    config_path.write_text("{}", encoding="utf-8")
    stable = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    transient = {**stable, "dirty": True, "dirty_diff_hash": "e" * 64}
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: stable)
    resolved = {"track": {}}
    fake_result = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
        "resolved_configuration": resolved,
        "config_hash": sha256_json(resolved),
        "code_provenance": stable,
    }
    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker_with_identity",
        lambda task: (
            "production",
            {},
            fake_result,
            None,
            dependency_fingerprint(),
            transient,
            stable,
        ),
    )
    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["code_provenance_verified"] is False
    assert result["all_passed"] is False
    assert result["worker_code_provenance_observations"] == [
        {
            "track": "production",
            "before_evaluator": transient,
            "after_evaluator": stable,
        }
    ]
    assert any("worker code provenance" in item for item in result["regressions"])


def test_matrix_rejects_orchestrator_provenance_drift_during_aggregation(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "production.json"
    config_path.write_text("{}", encoding="utf-8")
    stable = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    drifted = {**stable, "dirty": True, "dirty_diff_hash": "e" * 64}
    provenance = iter((stable, drifted))
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: next(provenance))
    resolved = {"track": {}}
    fake_result = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
        "resolved_configuration": resolved,
        "config_hash": sha256_json(resolved),
        "code_provenance": stable,
    }
    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker_with_identity",
        lambda task: (
            "production",
            {},
            fake_result,
            None,
            dependency_fingerprint(),
            stable,
            stable,
        ),
    )

    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert result["code_provenance_verified"] is False
    assert result["orchestrator_code_provenance_verified"] is False
    assert result["orchestrator_code_provenance_observations"] == {
        "before_matrix": stable,
        "after_aggregation": drifted,
    }
    assert any("orchestrator code provenance" in item for item in result["regressions"])


def test_matrix_executes_immutable_parent_config_snapshot(tmp_path, monkeypatch):
    config_path = tmp_path / "production.json"
    original = {"placeholder": False, "sentinel": "captured"}
    replacement = {"placeholder": True, "sentinel": "changed-after-capture"}
    config_path.write_text(json.dumps(original), encoding="utf-8")
    stable_provenance = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: stable_provenance)
    observed = {}

    def worker(task):
        captured = task[1]
        config_path.write_text(json.dumps(replacement), encoding="utf-8")
        observed["captured"] = captured
        resolved = {"track": captured}
        result = {
            "gates_passed": 1,
            "total_gates": 1,
            "gate_pass_rate": 1.0,
            "complete": True,
            "crashed": False,
            "disqualified": False,
            "termination_reason": "race_complete",
            "sim_time_s": 1.0,
            "avg_tracking_error_m": 0.1,
            "max_tracking_error_m": 0.2,
            "sim_passed": True,
            "safety_passed": True,
            "validity_passed": True,
            "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
            "plan_validation": {"ok": True, "reason": "ok"},
            "controller_trace_summary": {},
            "resolved_configuration": resolved,
            "config_hash": sha256_json(resolved),
            "code_provenance": stable_provenance,
        }
        return "production", captured, result, None

    monkeypatch.setattr(benchmark_matrix, "_run_track_worker", worker)
    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert observed["captured"] == original
    assert json.loads(config_path.read_text(encoding="utf-8")) == replacement
    assert result["tracks"]["production"]["config_snapshot_verified"] is True
    bound = result["resolved_configuration"]["tracks"][0]
    assert bound["parsed_config_hash"] == sha256_json(original)
    assert bound["content_sha256"] == hashlib.sha256(
        json.dumps(original).encode("utf-8")
    ).hexdigest()


def test_matrix_config_snapshot_rejects_content_change_during_read(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "changing.json"
    config_path.write_bytes(b'{"sentinel":1}')
    replacement = b'{"sentinel":2}'
    real_read = benchmark_matrix.os.read
    reached_first_eof = False

    def mutating_read(descriptor, size):
        nonlocal reached_first_eof
        chunk = real_read(descriptor, size)
        if chunk == b"" and not reached_first_eof:
            reached_first_eof = True
            config_path.write_bytes(replacement)
        return chunk

    monkeypatch.setattr(benchmark_matrix.os, "read", mutating_read)
    snapshot = benchmark_matrix._snapshot_config(config_path)

    assert snapshot["data"] is None
    assert "changed while it was being captured" in snapshot["error"]


@pytest.mark.parametrize(
    "payload",
    (
        '{"placeholder": false, "placeholder": true}',
        '{"value": NaN}',
        '[{"not": "an object"}]',
    ),
)
def test_matrix_rejects_ambiguous_or_nonstandard_config_json(
    tmp_path, monkeypatch, payload
):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(payload, encoding="utf-8")
    stable_provenance = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: stable_provenance)

    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert "load failed" in result["tracks"]["invalid"]["error"]
    assert result["resolved_configuration"]["tracks"][0]["load_error"]


def test_matrix_rejects_nonboolean_placeholder_label(tmp_path, monkeypatch):
    config_path = tmp_path / "production.json"
    data = {"placeholder": "false"}
    config_path.write_text(json.dumps(data), encoding="utf-8")
    stable = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: stable)
    resolved = {"track": data}
    fake_result = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
        "resolved_configuration": resolved,
        "config_hash": sha256_json(resolved),
        "code_provenance": stable,
    }
    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker_with_identity",
        lambda task: (
            "production",
            data,
            fake_result,
            None,
            dependency_fingerprint(),
            stable,
            stable,
        ),
    )

    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert result["tracks"]["production"]["is_placeholder"] is False
    assert any("placeholder must be" in item for item in result["regressions"])


@pytest.mark.parametrize(
    ("crashed", "disqualified", "expected"),
    ((True, False, "crashed"), (False, True, "disqualified")),
)
def test_matrix_placeholder_cannot_relax_safety_hard_gates(
    tmp_path, monkeypatch, crashed, disqualified, expected
):
    config_path = tmp_path / "placeholder.json"
    config_path.write_text('{"placeholder": true}', encoding="utf-8")
    fake_result = {
        "gates_passed": 2,
        "total_gates": 4,
        "gate_pass_rate": 0.5,
        "complete": False,
        "crashed": crashed,
        "disqualified": disqualified,
        "termination_reason": expected,
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": False,
        "safety_passed": False,
        "validity_passed": not disqualified,
        "completion": {"complete": False, "gates_passed": 2, "total_gates": 4},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
    }
    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker",
        lambda task: ("placeholder", {"placeholder": True}, fake_result, None),
    )

    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert result["safety_passed"] is False
    assert any(expected in item for item in result["regressions"])


def test_matrix_placeholder_cannot_relax_completion_or_sim_pass(tmp_path, monkeypatch):
    config_path = tmp_path / "placeholder.json"
    config_path.write_text('{"placeholder": true}', encoding="utf-8")
    fake_result = {
        "gates_passed": 3,
        "total_gates": 4,
        "gate_pass_rate": 0.75,
        "complete": False,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "time_limit",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": False,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": False, "gates_passed": 3, "total_gates": 4},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
    }
    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker",
        lambda task: ("placeholder", {"placeholder": True}, fake_result, None),
    )

    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert result["completion"]["complete"] is False
    assert any("completion is not explicitly true" in item for item in result["regressions"])
    assert any("sim_passed=False" in item for item in result["regressions"])


@pytest.mark.parametrize(
    "mutation",
    [
        "matching_zero_of_twelve_complete",
        "zero_total",
        "passed_over_total",
        "nested_mismatch",
        "termination_mismatch",
        "sim_passed_without_safety",
        "sim_passed_without_completion",
    ],
)
def test_matrix_rejects_internally_contradictory_completion_contract(
    tmp_path, monkeypatch, mutation
):
    config_path = tmp_path / "production.json"
    config_path.write_text("{}", encoding="utf-8")
    fake_result = {
        "gates_passed": 4,
        "total_gates": 4,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {
            "complete": True,
            "gates_passed": 4,
            "total_gates": 4,
        },
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
    }
    if mutation == "matching_zero_of_twelve_complete":
        fake_result.update(gates_passed=0, total_gates=12, gate_pass_rate=0.0)
        fake_result["completion"] = {
            "complete": True,
            "gates_passed": 0,
            "total_gates": 12,
        }
    elif mutation == "zero_total":
        fake_result.update(gates_passed=0, total_gates=0, gate_pass_rate=0.0)
        fake_result["completion"] = {
            "complete": True,
            "gates_passed": 0,
            "total_gates": 0,
        }
    elif mutation == "passed_over_total":
        fake_result.update(gates_passed=5, total_gates=4)
        fake_result["completion"] = {
            "complete": True,
            "gates_passed": 5,
            "total_gates": 4,
        }
    elif mutation == "nested_mismatch":
        fake_result["completion"]["gates_passed"] = 3
    elif mutation == "termination_mismatch":
        fake_result["termination_reason"] = "time_limit"
    elif mutation == "sim_passed_without_safety":
        fake_result["safety_passed"] = False
    elif mutation == "sim_passed_without_completion":
        fake_result.update(
            complete=False,
            gates_passed=3,
            gate_pass_rate=0.75,
            termination_reason="time_limit",
        )
        fake_result["completion"] = {
            "complete": False,
            "gates_passed": 3,
            "total_gates": 4,
        }

    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker",
        lambda task: ("production", {}, fake_result, None),
    )
    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert result["completion"]["complete"] is False
    assert any(
        "completion evidence missing, malformed, or contradictory" in item
        for item in result["regressions"]
    )


def test_matrix_top_level_hard_gates_fail_when_any_track_errors(tmp_path, monkeypatch):
    good_path = tmp_path / "good.json"
    bad_path = tmp_path / "bad.json"
    good_path.write_text("{}", encoding="utf-8")
    bad_path.write_text("{}", encoding="utf-8")
    good = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
    }

    def worker(task):
        name = Path(task[0]).stem
        if name == "bad":
            return name, {}, None, "bench raised: RuntimeError: broken"
        return name, {}, good, None

    monkeypatch.setattr(benchmark_matrix, "_run_track_worker", worker)
    result = benchmark_matrix.run_matrix(
        [good_path, bad_path],
        duration=1.0,
        max_workers=1,
        cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert result["safety_passed"] is False
    assert result["validity_passed"] is False
    assert result["completion"]["complete"] is False
    assert result["completion"]["evidence_complete"] is False
    assert result["completion"]["evaluated_tracks"] == 1
    assert result["completion"]["requested_tracks"] == 2


@pytest.mark.parametrize(
    ("missing_field", "expected"),
    (
        ("safety_passed", "safety_passed=False"),
        ("validity_passed", "validity_passed=False"),
        ("completion", "completion evidence"),
    ),
)
def test_matrix_requires_explicit_hard_gate_evidence(
    tmp_path, monkeypatch, missing_field, expected
):
    config_path = tmp_path / "production.json"
    config_path.write_text("{}", encoding="utf-8")
    fake_result = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
    }
    del fake_result[missing_field]
    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker",
        lambda task: ("production", {}, fake_result, None),
    )

    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache",
        _in_process_test_worker=True,
    )

    assert result["all_passed"] is False
    assert any(expected in item for item in result["regressions"])


@pytest.mark.benchmark
def test_matrix_single_track_production_path_uses_spawn_executor(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "production.json"
    config_path.write_text("{}", encoding="utf-8")
    stable = {
        "commit": "a" * 40,
        "dirty": False,
        "dirty_diff_hash": "b" * 64,
        "tracked_diff_hash": "c" * 64,
        "untracked_content_hash": "d" * 64,
        "excluded_untracked_paths": [],
    }
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: stable)
    resolved = {"track": {}}
    fake_result = {
        "gates_passed": 1,
        "total_gates": 1,
        "gate_pass_rate": 1.0,
        "complete": True,
        "crashed": False,
        "disqualified": False,
        "termination_reason": "race_complete",
        "sim_time_s": 1.0,
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "sim_passed": True,
        "safety_passed": True,
        "validity_passed": True,
        "completion": {"complete": True, "gates_passed": 1, "total_gates": 1},
        "plan_validation": {"ok": True, "reason": "ok"},
        "controller_trace_summary": {},
        "resolved_configuration": resolved,
        "config_hash": sha256_json(resolved),
        "code_provenance": stable,
    }
    expected_worker_fingerprint = (
        benchmark_matrix._resolved_worker_dependency_fingerprint(
            dependency_fingerprint()
        )
    )
    monkeypatch.setattr(
        benchmark_matrix,
        "_run_track_worker_with_identity",
        lambda task: (
            "production",
            {},
            fake_result,
            None,
            expected_worker_fingerprint,
            stable,
            stable,
        ),
    )
    observed = {}

    class FakeSpawnExecutor:
        def __init__(self, *, max_workers, initializer, mp_context):
            observed["max_workers"] = max_workers
            observed["start_method"] = mp_context.get_start_method()
            observed["initializer"] = initializer.__name__

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, tasks):
            return map(function, tasks)

    monkeypatch.setattr(
        benchmark_matrix.concurrent.futures,
        "ProcessPoolExecutor",
        FakeSpawnExecutor,
    )

    result = benchmark_matrix.run_matrix(
        [config_path], duration=1.0, max_workers=1, cache_root=tmp_path / "cache"
    )

    assert observed == {
        "max_workers": 1,
        "start_method": "spawn",
        "initializer": "_limit_worker_threads",
    }
    assert result["worker_processes"] == 1
    assert result["worker_execution"] == "isolated_processes_spawn"
    assert result["promotion_evidence_eligible"] is True
    assert result["all_passed"] is True


@pytest.mark.benchmark
def test_matrix_runs_distinct_tracks_in_capped_processes(tmp_path):
    config_paths = []
    for index, gate_x in enumerate((2.0, 3.0), start=1):
        course = _one_gate_course()
        course["gates"][0]["pose"]["x"] = gate_x
        path = tmp_path / f"track-{index}.json"
        path.write_text(json.dumps(course), encoding="utf-8")
        config_paths.append(path)

    result = benchmark_matrix.run_matrix(
        config_paths,
        duration=0.1,
        dt=0.02,
        max_workers=2,
        cache_root=tmp_path / "artifacts",
    )
    assert result["worker_processes"] == 2
    assert result["worker_threads_capped"] is True
    assert result["worker_execution"] == "isolated_processes_spawn"
    assert result["resolved_configuration"]["runtime"]["worker_start_method"] == "spawn"
    assert set(result["tracks"]) == {"track-1", "track-2"}
    assert all("error" not in track for track in result["tracks"].values())


def test_synthetic_cli_wires_selected_config(tmp_path, monkeypatch, capsys):
    selected = _one_gate_course(proximity=0.321)
    config_path = tmp_path / "selected.json"
    config_path.write_text(json.dumps(selected), encoding="utf-8")
    observed = {}

    def fake_synthetic(*, duration, config, thresholds):
        observed["config"] = config
        observed["thresholds"] = thresholds
        return {"skipped": False, "sim_passed": True}

    monkeypatch.setattr(benchmark, "run_synthetic_benchmark", fake_synthetic)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--mode",
            "synthetic",
            "--config",
            str(config_path),
            "--completion-threshold",
            "0.25",
            "--json-only",
        ],
    )
    assert benchmark.main() == 0
    assert observed["config"] == selected
    assert observed["thresholds"]["min_gate_pass_rate"] == 0.25
    assert benchmark.THRESHOLDS["min_gate_pass_rate"] == 1.0
    assert json.loads(capsys.readouterr().out)["overall_passed"] is True


def test_cli_rejects_out_of_range_completion_threshold(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--mode",
            "unit",
            "--completion-threshold",
            "1.01",
            "--json-only",
        ],
    )
    with pytest.raises(SystemExit) as raised:
        benchmark.main()
    assert raised.value.code == 2
    assert "within [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mode", "duration", "message"),
    [
        ("synthetic", "nan", "duration must be finite"),
        ("synthetic", "-1", "duration must be non-negative"),
        ("sim", "0", "strictly positive"),
        ("full", "inf", "finite and strictly positive"),
    ],
)
def test_cli_rejects_nonfinite_or_out_of_range_duration(
    tmp_path, monkeypatch, capsys, mode, duration, message
):
    config_path = tmp_path / "selected.json"
    config_path.write_text(json.dumps(_one_gate_course()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--mode",
            mode,
            "--config",
            str(config_path),
            "--duration",
            duration,
            "--json-only",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.main()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"gates": [], "gates": []}', "duplicate JSON key"),
        ('{"max_velocity_mps": NaN}', "non-standard JSON numeric constant"),
        ('[{"not": "an object"}]', "root must be a JSON object"),
    ],
)
def test_cli_rejects_ambiguous_or_nonstandard_track_json(
    tmp_path, monkeypatch, capsys, payload, message
):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--mode",
            "unit",
            "--config",
            str(config_path),
            "--json-only",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.main()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_full_mode_fails_closed_when_pybullet_is_skipped(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "selected.json"
    config_path.write_text(json.dumps(_one_gate_course()), encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "run_unit_tests",
        lambda: {"pass_rate": 1.0, "passed": 1, "total": 1, "total_time_ms": 0, "tests": []},
    )
    monkeypatch.setattr(
        benchmark,
        "run_synthetic_benchmark",
        lambda **kwargs: {"skipped": False, "sim_passed": True},
    )
    monkeypatch.setattr(
        benchmark,
        "run_sim_benchmark",
        lambda *args, **kwargs: {"skipped": True, "skip_reason": "not installed"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--mode",
            "full",
            "--config",
            str(config_path),
            "--json-only",
        ],
    )
    assert benchmark.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["overall_passed"] is False
    assert "requires an executed evaluator" in report["simulation"]["threshold_failures"][0]


def test_pybullet_skip_has_complete_failure_envelope(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sim_pybullet.env", None)
    result = benchmark.run_sim_benchmark(str(tmp_path / "missing.json"), 0.1)

    assert result["skipped"] is True
    assert result["evaluator_version"] == "pybullet-v3-plane-crossing"
    assert result["comparison_series"] == "pybullet-v3"
    assert result["sim_passed"] is False
    assert result["safety_passed"] is False
    assert result["validity_passed"] is False
    assert result["completion"]["complete"] is False
    assert result["cache_hit_or_miss"] == "not_applicable"
    assert result["code_provenance"]["commit"]
    assert result["dependency_fingerprint"]
    assert set(result["dependency_fingerprint"]["physics_dependencies"]) == {
        "pybullet",
        "gym_pybullet_drones",
    }
    assert result["resolved_configuration"]
    assert result["config_hash"]
    assert result["phase_timings_s"]["total_wall"] >= 0.0
    assert result["timing_consistency"]["total_covers_phases"] is True
    assert (
        "PyBullet skipped; full/sim mode requires an executed evaluator"
        in result["failure_summary"]["threshold_failures"]
    )


@pytest.mark.parametrize("duration", [True, "30", 0, -1, float("nan"), float("inf")])
def test_pybullet_duration_is_exact_finite_and_positive(duration):
    expected = TypeError if isinstance(duration, (bool, str)) else ValueError
    with pytest.raises(expected):
        benchmark.run_sim_benchmark("unused.json", duration)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda course: course.update(max_velocity_mps=True),
        lambda course: course["gates"][0]["pose"].update(x="2.0"),
        lambda course: course["gate_defaults"].update(interior_width_m=0.0),
        lambda course: course["start"].update(position=[0.0, 1.0]),
    ],
)
def test_pybullet_config_loader_rejects_coerced_or_nonphysical_values(
    tmp_path, mutation
):
    from sim_pybullet.env import DroneRaceEnv

    course = _one_gate_course()
    mutation(course)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(course), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        DroneRaceEnv.load_config(str(path))


def test_pybullet_config_loader_preserves_gate_safety_geometry(tmp_path):
    from sim_pybullet.env import DroneRaceEnv

    course = _one_gate_course()
    course["gates"][0]["config"] = {
        "border_width_m": 0.25,
        "depth_m": 0.4,
    }
    path = tmp_path / "valid.json"
    path.write_text(json.dumps(course), encoding="utf-8")

    loaded = DroneRaceEnv.load_config(str(path))

    assert loaded.gates[0].config.border_width_m == pytest.approx(0.25)
    assert loaded.gates[0].config.depth_m == pytest.approx(0.4)
    assert loaded.gates[0].sequence_index == 1


def test_pybullet_progress_clock_uses_observed_control_time_not_race_timestep():
    # The historical RaceConfig timestep is ~1/240 s while one GPD control
    # step advances ~1/48 s.  The progress clock must use the latter observed
    # delta and must fail closed if that clock stalls or reverses.
    assert benchmark._observed_sim_delta(None, 0.0) == 0.0
    assert benchmark._observed_sim_delta(1.0, 1.0 + 1.0 / 48.0) == pytest.approx(
        1.0 / 48.0
    )
    with pytest.raises(RuntimeError, match="did not advance"):
        benchmark._observed_sim_delta(1.0, 1.0)
    with pytest.raises(RuntimeError, match="did not advance"):
        benchmark._observed_sim_delta(1.0, 0.9)


def test_pybullet_thresholds_include_plan_validity_and_total_time():
    metrics = {
        "crashed": False,
        "disqualified": False,
        "dq_reason": None,
        "termination_reason": "race_complete",
        "plan_validation": {"ok": False, "reason": "missed gate"},
        "avg_tracking_error_m": 0.1,
        "max_tracking_error_m": 0.2,
        "ekf_uncertainty_m": 0.01,
        "avg_loop_hz": 1000.0,
        "gate_pass_rate": 1.0,
        "gates_passed": 1,
        "total_gates": 1,
        "complete": True,
        "sim_time_s": 30.01,
    }
    failures = benchmark._pybullet_threshold_failures(
        metrics, benchmark._threshold_snapshot()
    )
    assert any("plan_validation failed" in failure for failure in failures)
    assert any("race_time" in failure for failure in failures)


def test_pybullet_thresholds_fail_closed_on_nonfinite_metrics():
    metrics = {
        "crashed": False,
        "disqualified": False,
        "dq_reason": None,
        "termination_reason": "race_complete",
        "plan_validation": {"ok": True, "reason": "ok"},
        "avg_tracking_error_m": float("nan"),
        "max_tracking_error_m": float("inf"),
        "ekf_uncertainty_m": -1.0,
        "avg_loop_hz": 1000.0,
        "gate_pass_rate": 1.0,
        "gates_passed": 1,
        "total_gates": 1,
        "complete": True,
        "sim_time_s": 1.0,
    }
    failures = benchmark._pybullet_threshold_failures(
        metrics, benchmark._threshold_snapshot()
    )
    assert any("avg_tracking_error_m must be" in failure for failure in failures)
    assert any("max_tracking_error_m must be" in failure for failure in failures)
    assert any("ekf_uncertainty_m must be" in failure for failure in failures)


def test_sim_cli_wraps_runtime_exception_in_honest_envelope(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "selected.json"
    config_path.write_text(json.dumps(_one_gate_course()), encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "run_sim_benchmark",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sim exploded")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--mode",
            "sim",
            "--config",
            str(config_path),
            "--json-only",
        ],
    )

    assert benchmark.main() == 1
    result = json.loads(capsys.readouterr().out)["simulation"]
    assert result["sim_passed"] is False
    assert result["safety_passed"] is False
    assert result["validity_passed"] is False
    assert result["failure_summary"]["exception"] == "RuntimeError: sim exploded"
    assert result["code_provenance"]["commit"]
    assert result["phase_timings_s"]["total_wall"] >= 0.0


def test_pybullet_environment_closes_when_pipeline_raises(tmp_path, monkeypatch):
    from sim_pybullet.env import DroneRaceEnv as ConfigLoader
    from planning import plan_validator

    closed = []
    config_path = tmp_path / "selected.json"
    config_path.write_text(json.dumps(_one_gate_course()), encoding="utf-8")
    race_config = ConfigLoader.load_config(str(config_path))

    class FakeDroneRaceEnv:
        @staticmethod
        def load_config(path):
            return race_config

        def __init__(self, *, race_config, drone_config, gui):
            self.race_config = race_config

        def get_sim_time(self):
            raise RuntimeError("rollout exploded")

        def close(self):
            closed.append(True)

    fake_module = ModuleType("sim_pybullet.env")
    fake_module.DroneRaceEnv = FakeDroneRaceEnv
    monkeypatch.setitem(sys.modules, "sim_pybullet.env", fake_module)
    monkeypatch.setattr(
        plan_validator,
        "validate_trajectory",
        lambda *args, **kwargs: {
            "ok": True,
            "reason": "ok",
            "gates_passed": 1,
            "total_gates": 1,
            "crashed": False,
            "disqualified": False,
            "dq_reason": None,
            "last_crash_gate": None,
            "samples_evaluated": 1,
            "first_failure_time_s": None,
        },
    )

    with pytest.raises(RuntimeError, match="rollout exploded"):
        benchmark.run_sim_benchmark(str(config_path), 0.1)
    assert closed == [True]


def test_pybullet_environment_retries_close_after_first_close_failure(
    tmp_path, monkeypatch
):
    from sim_pybullet.env import DroneRaceEnv as ConfigLoader
    from planning import plan_validator

    close_calls = []
    config_path = tmp_path / "selected.json"
    config_path.write_text(json.dumps(_one_gate_course()), encoding="utf-8")
    race_config = ConfigLoader.load_config(str(config_path))

    class FakeDrone:
        def __init__(self):
            self.state = {
                "position": race_config.start_position,
                "velocity": (0.0, 0.0, 0.0),
                "yaw": 0.0,
                "angular_velocity": (0.0, 0.0, 0.0),
            }

        def get_state(self):
            return dict(self.state)

        def step(self, target_pos, target_vel, target_yaw):
            return None

    class FakeDroneRaceEnv:
        @staticmethod
        def load_config(path):
            return race_config

        def __init__(self, *, race_config, drone_config, gui):
            self.race_config = race_config
            self.drone = FakeDrone()
            self.time_calls = 0

        def get_sim_time(self):
            self.time_calls += 1
            return 0.0 if self.time_calls == 1 else 1.0

        def gate_contact(self):
            return None

        def close(self):
            close_calls.append(len(close_calls) + 1)
            if len(close_calls) == 1:
                raise RuntimeError("first close failed")

    fake_module = ModuleType("sim_pybullet.env")
    fake_module.DroneRaceEnv = FakeDroneRaceEnv
    monkeypatch.setitem(sys.modules, "sim_pybullet.env", fake_module)
    monkeypatch.setattr(
        plan_validator,
        "validate_trajectory",
        lambda *args, **kwargs: {
            "ok": True,
            "reason": "ok",
            "gates_passed": 1,
            "total_gates": 1,
            "crashed": False,
            "disqualified": False,
            "dq_reason": None,
            "last_crash_gate": None,
            "samples_evaluated": 1,
            "first_failure_time_s": None,
        },
    )

    with pytest.raises(RuntimeError, match="first close failed"):
        benchmark.run_sim_benchmark(str(config_path), 0.1)
    assert close_calls == [1, 2]
