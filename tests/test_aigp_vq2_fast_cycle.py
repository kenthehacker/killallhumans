"""Regression tests for the compact, noninteractive powered-cycle front door."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import aigp_vq2_fast_cycle as fast_cycle
from scripts import aigp_vq2_controller_config as controller_config
from scripts import aigp_vq2_visual_config as visual_config
from scripts import aigp_vq2_yaw_calibration as yaw_calibration
from scripts import aigp_vq2_yaw_profile as yaw_profile


UTC = datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc)


def test_calibration_manifest_binds_exact_yaw_plan():
    expected = {
        "plan_id": yaw_calibration.YAW_CALIBRATION_PLAN_ID,
        "sha256": yaw_calibration.YAW_CALIBRATION_PLAN_SHA256,
        "tick_count": yaw_calibration.YAW_CALIBRATION_TICK_COUNT,
        "control_period_ns": (
            yaw_calibration.YAW_CALIBRATION_CONTROL_PERIOD_NS
        ),
    }
    assert fast_cycle._excitation_plan_identity(
        "calibration-excite"
    ) == expected
    assert fast_cycle._excitation_plan_identity("sign-id") == expected
    assert fast_cycle._excitation_plan_identity("hover") is None
    assert (
        "scripts/aigp_vq2_yaw_calibration.py"
        in fast_cycle._RUNTIME_SOURCE_PATHS
    )


def test_visual_course_manifest_binds_reviewed_yaw_profile():
    identity = fast_cycle._yaw_profile_identity("visual-course")

    assert identity == {
        "profile_id": yaw_profile.YAW_CALIBRATION_PROFILE_ID,
        "sha256": yaw_profile.YAW_CALIBRATION_PROFILE_SHA256,
        "source_commit": yaw_profile.YAW_CALIBRATION_SOURCE_COMMIT,
        "plan_id": yaw_profile.YAW_CALIBRATION_PLAN_ID,
        "plan_sha256": yaw_profile.YAW_CALIBRATION_PLAN_SHA256,
        "authority": yaw_profile.load_yaw_calibration_profile()[
            "authority"
        ],
    }
    assert fast_cycle._yaw_profile_identity("visual-align") is None
    assert (
        "scripts/aigp_vq2_visual_course_stage.py"
        in fast_cycle._RUNTIME_SOURCE_PATHS
    )
    assert (
        "planning/vq2_course_lifecycle.py"
        in fast_cycle._RUNTIME_SOURCE_PATHS
    )
    assert (
        "scripts/aigp_vq2_yaw_profile.py"
        in fast_cycle._RUNTIME_SOURCE_PATHS
    )
    assert (
        "config/aigp_vq2_yaw_calibration_build3385.json"
        in fast_cycle._RUNTIME_SOURCE_PATHS
    )


@contextmanager
def _fake_lease(path, *, initial_phase):
    assert initial_phase == "fast_cycle"
    Path(path).write_text('{"phase":"acquired"}\n', encoding="utf-8")
    yield


@dataclass(frozen=True)
class _StageResult:
    stage: str
    success: bool
    reason: str
    duration_s: float
    gate_index_before: int | None
    gate_index_after: int | None
    cleanup_confirmed: bool
    details: dict[str, object]
    controller: dict[str, object] | None = None


def test_compact_manifest_has_no_interactive_or_bulk_freeze_inputs(tmp_path):
    manifest = fast_cycle.build_manifest(
        stage="calibration-excite",
        run_id="run-1",
        created_at=UTC,
        repo_root=Path(r"C:\candidate"),
        run_directory=tmp_path,
        git_snapshot={
            "head_commit": "a" * 40,
            "head_tree": "b" * 40,
            "worktree_state": "dirty",
            "status_sha256": "c" * 64,
            "tracked_diff_sha256": "d" * 64,
        },
        runtime_sources=[
            {"path": "scripts/aigp_vq2_run.py", "size_bytes": 1, "sha256": "e" * 64}
        ],
        target_config={"path": "target.json", "size_bytes": 2, "sha256": "f" * 64},
        development_lock={"path": "lock.txt", "size_bytes": 3, "sha256": "1" * 64},
        excitation_plan=fast_cycle._excitation_plan_identity(
            "calibration-excite"
        ),
    )

    assert manifest["schema"] == fast_cycle.MANIFEST_SCHEMA
    assert manifest["authorization"] == {
        "basis": "caller_asserted_existing_scoped_user_authorization",
        "scope": {"stage": "calibration-excite", "attempt_limit": 1},
        "interactive_confirmation": False,
        "expires": "process_exit",
    }
    assert manifest["execution"] == {
        "stage": "calibration-excite",
        "address": fast_cycle.DEFAULT_ADDRESS,
        "separate_passive_preflight": False,
        "preflight_healthy_dwell_s": 0.0,
        "manual_training_challenge": False,
        "screenshot_capture": False,
        "full_environment_or_import_inventory": False,
        "live_lease_mutex": fast_cycle.LIVE_LEASE_MUTEX_NAME,
    }
    assert set(manifest["candidate"]) == {
        "worktree",
        "head_commit",
        "head_tree",
        "worktree_state",
        "status_sha256",
        "tracked_diff_sha256",
        "runtime_sources",
    }
    assert manifest["simulator"]["mode_basis"] == (
        "configured_session_not_machine_readable"
    )
    assert manifest["runtime"]["isolation_flags"] == ["-E", "-s", "-B"]
    assert "live_freeze" not in json.dumps(manifest)
    assert "attestation" not in json.dumps(manifest)


def test_full_lap_is_quarantined_from_fast_powered_stages():
    assert "full-lap" not in fast_cycle.FAST_POWERED_STAGES
    with pytest.raises(SystemExit):
        fast_cycle.build_argument_parser().parse_args(["full-lap"])


def test_gate1_recenter_is_admitted_as_the_bounded_position_only_stage():
    assert "gate1-recenter" in fast_cycle.FAST_POWERED_STAGES
    arguments = fast_cycle.build_argument_parser().parse_args(
        ["gate1-recenter"]
    )
    assert arguments.stage == "gate1-recenter"


@pytest.mark.parametrize(
    ("requested_stage", "expected_replay_name"),
    [
        ("visual-shadow", "shadow.vq2replay"),
        ("visual-align", "alignment.vq2replay"),
        ("visual-course", None),
    ],
)
def test_visual_stage_is_admitted_with_stage_scoped_compact_evidence(
    tmp_path,
    monkeypatch,
    requested_stage,
    expected_replay_name,
):
    git_snapshot = {
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "worktree_state": "clean",
        "status_sha256": "c" * 64,
        "tracked_diff_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        fast_cycle,
        "_git_snapshot",
        lambda _root: dict(git_snapshot),
    )
    calls = []

    async def run_live(stage, address, record, **kwargs):
        calls.append((stage, address, record, kwargs))
        Path(record).write_bytes(b"visual navigation trace")
        effective = visual_config.validate_visual_config(
            kwargs["controller_config"]
        )
        evidence = fast_cycle._controller_evidence(
            effective,
            candidate_commit=kwargs["candidate_commit"],
        )
        return _StageResult(
            stage=stage,
            success=True,
            reason="stage completed",
            duration_s=4.0,
            gate_index_before=0,
            gate_index_after=0,
            cleanup_confirmed=True,
            details={"authoritative_transition": [0, 1]},
            controller=evidence,
        )

    code, result = fast_cycle._execute_fast_cycle(
        requested_stage,
        evidence_root=tmp_path,
        now=lambda: UTC,
        load_runner=lambda: SimpleNamespace(run_live=run_live),
        lease_factory=_fake_lease,
    )

    assert code == 0
    assert result["success"] is True
    assert len(calls) == 1
    stage, address, record, kwargs = calls[0]
    assert stage == requested_stage
    assert address == fast_cycle.DEFAULT_ADDRESS
    effective = visual_config.validate_visual_config(
        kwargs["controller_config"]
    )
    assert result["controller"]["controller_family"] == (
        visual_config.VISUAL_CONTROLLER_FAMILY
    )
    assert result["controller"]["config_sha256"] == (
        effective.effective_config_sha256
    )
    assert kwargs["expected_controller_config_sha256"] == (
        effective.effective_config_sha256
    )
    assert kwargs["expected_yaw_calibration_profile_sha256"] == (
        yaw_profile.YAW_CALIBRATION_PROFILE_SHA256
        if requested_stage == "visual-course"
        else None
    )
    assert kwargs["recording_approved"] is (
        requested_stage in fast_cycle.VISUAL_REPLAY_STAGES
    )
    assert kwargs["preflight_before_powered_stage"] is False
    assert kwargs["write_diagnostic_pngs"] is False
    replay_path = (
        None
        if kwargs["replay_bundle"] is None
        else Path(kwargs["replay_bundle"])
    )
    if expected_replay_name is None:
        assert replay_path is None
    else:
        assert replay_path is not None
        assert replay_path.parent == Path(record).parent
        assert replay_path.name == expected_replay_name
    manifest = json.loads(
        (Path(record).parent / "run-manifest.json").read_text()
    )
    assert manifest["evidence"]["replay_bundle"] == (
        None if replay_path is None else str(replay_path)
    )
    assert manifest["candidate"]["worktree_state"] == "clean"
    assert manifest["inputs"]["yaw_calibration_profile"] == (
        fast_cycle._yaw_profile_identity(requested_stage)
    )


@pytest.mark.parametrize(
    "requested_stage",
    ["visual-shadow", "visual-align", "visual-course"],
)
def test_visual_stage_refuses_dirty_candidate_before_live_contact(
    tmp_path,
    monkeypatch,
    requested_stage,
):
    monkeypatch.setattr(
        fast_cycle,
        "_git_snapshot",
        lambda _root: {
            "head_commit": "a" * 40,
            "head_tree": "b" * 40,
            "worktree_state": "dirty",
            "status_sha256": "c" * 64,
            "tracked_diff_sha256": "d" * 64,
        },
    )
    called = False

    async def run_live(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("dirty visual candidate must not contact live runner")

    with pytest.raises(
        fast_cycle.FastCycleError,
        match="require a clean exact commit",
    ):
        fast_cycle._execute_fast_cycle(
            requested_stage,
            evidence_root=tmp_path,
            now=lambda: UTC,
            load_runner=lambda: SimpleNamespace(run_live=run_live),
            lease_factory=_fake_lease,
        )
    assert called is False


@pytest.mark.parametrize(
    "requested_stage",
    ["sign-id", "calibration-excite"],
)
def test_fast_cycle_runs_once_without_separate_preflight_or_prompt(
    tmp_path,
    requested_stage,
):
    calls = []

    async def run_live(stage, address, record, **kwargs):
        calls.append((stage, address, record, kwargs))
        Path(record).write_bytes(b"compact trace")
        evidence = fast_cycle._controller_evidence(
            controller_config.validate_controller_config(
                kwargs["controller_config"]
            ),
            candidate_commit=kwargs["candidate_commit"],
        )
        return _StageResult(
            stage=stage,
            success=True,
            reason="stage completed",
            duration_s=5.0,
            gate_index_before=0,
            gate_index_after=0,
            cleanup_confirmed=True,
            details={
                "ticks_sent": yaw_calibration.YAW_CALIBRATION_TICK_COUNT
            },
            controller=evidence,
        )

    code, result = fast_cycle._execute_fast_cycle(
        requested_stage,
        evidence_root=tmp_path,
        now=lambda: UTC,
        load_runner=lambda: SimpleNamespace(run_live=run_live),
        lease_factory=_fake_lease,
    )

    assert code == 0
    assert result["success"] is True
    assert len(calls) == 1
    stage, address, record, kwargs = calls[0]
    assert stage == requested_stage
    assert address == fast_cycle.DEFAULT_ADDRESS
    assert kwargs["preflight_before_powered_stage"] is False
    assert kwargs["write_diagnostic_pngs"] is False
    assert kwargs["run_manifest_sha256"] == result["run_manifest_sha256"]
    effective = controller_config.validate_controller_config(
        kwargs["controller_config"]
    )
    assert kwargs["candidate_commit"] == result["controller"]["git_commit"]
    assert (
        kwargs["expected_controller_config_sha256"]
        == effective.effective_config_sha256
        == result["controller"]["config_sha256"]
    )
    run_directory = Path(record).parent
    manifest = json.loads((run_directory / "run-manifest.json").read_text())
    terminal = json.loads((run_directory / "result.json").read_text())
    assert manifest["run_id"] == terminal["run_id"] == result["run_id"]
    assert manifest["authorization"]["scope"] == {
        "stage": requested_stage,
        "attempt_limit": 1,
    }
    assert manifest["execution"]["stage"] == requested_stage
    assert manifest["controller"] == result["controller"]
    assert terminal["controller"] == result["controller"]
    assert result["runner_result"]["controller"] == result["controller"]
    assert (
        result["controller"]["effective_parameters"]
        == {
            key: value
            for key, value in effective.to_effective_mapping().items()
            if key not in {"schema", "controller_family"}
        }
    )
    assert manifest["inputs"]["excitation_plan"] is not None
    assert (run_directory / "session.jsonl.gz").read_bytes() == b"compact trace"
    assert result["trace"]["sha256"] == fast_cycle.hashlib.sha256(
        b"compact trace"
    ).hexdigest()
    assert (run_directory / "live-lease.json").is_file()
    assert not list(run_directory.glob("*.png"))


def test_runtime_source_drift_refuses_before_live_contact(tmp_path, monkeypatch):
    snapshots = [
        [{"path": "runner.py", "size_bytes": 1, "sha256": "a" * 64}],
        [{"path": "runner.py", "size_bytes": 2, "sha256": "b" * 64}],
    ]
    monkeypatch.setattr(
        fast_cycle,
        "_runtime_source_identities",
        lambda _root: snapshots.pop(0),
    )
    monkeypatch.setattr(
        fast_cycle,
        "_git_snapshot",
        lambda _root: {
            "head_commit": "a" * 40,
            "head_tree": "b" * 40,
            "worktree_state": "clean",
            "status_sha256": "c" * 64,
            "tracked_diff_sha256": "d" * 64,
        },
    )
    called = False

    async def run_live(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("live runner must not be called")

    code, result = fast_cycle._execute_fast_cycle(
        "hover",
        evidence_root=tmp_path,
        now=lambda: UTC,
        load_runner=lambda: SimpleNamespace(run_live=run_live),
        lease_factory=_fake_lease,
    )

    assert code == 2
    assert result["success"] is False
    assert "runtime sources changed" in result["reason"]
    assert called is False


def test_fast_cycle_uses_the_canonical_live_lease_factory(tmp_path):
    observed = []

    @contextmanager
    def refusing_lease(path, *, initial_phase):
        observed.append((Path(path), initial_phase))
        raise fast_cycle.LiveLeaseError("busy canonical lease")
        yield

    with pytest.raises(fast_cycle.LiveLeaseError, match="busy canonical lease"):
        fast_cycle._execute_fast_cycle(
            "hover",
            evidence_root=tmp_path,
            lease_factory=refusing_lease,
        )

    assert len(observed) == 1
    assert observed[0][0].name == "live-lease.json"
    assert observed[0][1] == "fast_cycle"


def test_execution_boundary_enforces_python_isolation_before_creating_a_run(
    tmp_path,
    monkeypatch,
):
    def reject():
        raise fast_cycle.FastCycleError("isolation missing")

    monkeypatch.setattr(fast_cycle, "_require_isolated_runtime", reject)
    with pytest.raises(fast_cycle.FastCycleError, match="isolation missing"):
        fast_cycle.execute_fast_cycle("hover", evidence_root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_production_wrapper_pins_real_runner_and_canonical_lease(monkeypatch):
    observed = {}
    monkeypatch.setattr(fast_cycle, "_require_isolated_runtime", lambda: None)

    def implementation(stage, **kwargs):
        observed.update({"stage": stage, **kwargs})
        return 0, {"success": True}

    monkeypatch.setattr(fast_cycle, "_execute_fast_cycle", implementation)

    code, _result = fast_cycle.execute_fast_cycle("hover")

    assert code == 0
    assert observed["stage"] == "hover"
    assert observed["address"] == fast_cycle.DEFAULT_ADDRESS
    assert observed["load_runner"] is None
    assert observed["lease_factory"] is fast_cycle.live_simulator_lease


def test_lease_release_failure_still_publishes_failed_terminal_result(tmp_path):
    class FailingReleaseLease:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            self.path.write_text('{"phase":"acquired"}\n', encoding="utf-8")
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            raise fast_cycle.LiveLeaseError("release failed")

    def lease_factory(path, *, initial_phase):
        assert initial_phase == "fast_cycle"
        return FailingReleaseLease(path)

    async def run_live(stage, address, record, **_kwargs):
        del address
        Path(record).write_bytes(b"trace")
        return _StageResult(stage, True, "stage completed", 1.0, 0, 0, True, {})

    code, result = fast_cycle._execute_fast_cycle(
        "hover",
        evidence_root=tmp_path,
        now=lambda: UTC,
        load_runner=lambda: SimpleNamespace(run_live=run_live),
        lease_factory=lease_factory,
    )

    assert code == 2
    assert result["success"] is False
    assert "live lease cleanup failed" in result["reason"]
    result_paths = list(tmp_path.glob("*/result.json"))
    assert len(result_paths) == 1
    assert json.loads(result_paths[0].read_text())["success"] is False


def test_evidence_root_inside_worktree_is_rejected():
    repo_root = Path(fast_cycle.__file__).resolve().parents[1]
    with pytest.raises(fast_cycle.FastCycleError, match="outside the Git worktree"):
        fast_cycle._execute_fast_cycle(
            "hover",
            evidence_root=repo_root / ".artifacts/fast-flight",
            lease_factory=_fake_lease,
        )


def test_controller_config_cli_and_loader_require_one_complete_document(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text(
        json.dumps(controller_config.default_controller_config_mapping()),
        encoding="utf-8",
    )
    parsed = fast_cycle.build_argument_parser().parse_args(
        ["gate1-recenter", "--controller-config", str(path)]
    )
    assert parsed.controller_config == path
    assert (
        fast_cycle._load_controller_config(path)
        == controller_config.default_controller_config()
    )

    path.write_text('{"schema":"incomplete"}', encoding="utf-8")
    with pytest.raises(
        fast_cycle.FastCycleError,
        match="controller configuration refused",
    ):
        fast_cycle._load_controller_config(path)


def test_custom_controller_is_rejected_for_unrelated_stage_before_lease(tmp_path):
    document = controller_config.default_controller_config_mapping()
    document["yaw_control"]["gate1_error_gain"] = -0.08
    document["yaw_control"]["command_rate_cap_rad_s"] = 0.08
    lease_called = False

    @contextmanager
    def forbidden_lease(*_args, **_kwargs):
        nonlocal lease_called
        lease_called = True
        yield

    with pytest.raises(
        fast_cycle.FastCycleError,
        match="only for gate1-recenter",
    ):
        fast_cycle._execute_fast_cycle(
            "hover",
            evidence_root=tmp_path,
            lease_factory=forbidden_lease,
            controller_config=document,
        )
    assert lease_called is False
    assert list(tmp_path.iterdir()) == []
