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


UTC = datetime(2026, 7, 22, 20, 0, tzinfo=timezone.utc)


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
        excitation_plan={
            "plan_id": "plan",
            "sha256": "2" * 64,
            "tick_count": 245,
            "control_period_ns": 20_000_000,
        },
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


def test_gate1_recenter_remains_offline_until_replay_prerequisite_is_accepted():
    assert "gate1-recenter" not in fast_cycle.FAST_POWERED_STAGES
    with pytest.raises(SystemExit):
        fast_cycle.build_argument_parser().parse_args(["gate1-recenter"])


@pytest.mark.parametrize("requested_stage", ["calibration-excite"])
def test_fast_cycle_runs_once_without_separate_preflight_or_prompt(
    tmp_path,
    requested_stage,
):
    calls = []

    async def run_live(stage, address, record, **kwargs):
        calls.append((stage, address, record, kwargs))
        Path(record).write_bytes(b"compact trace")
        return _StageResult(
            stage=stage,
            success=True,
            reason="stage completed",
            duration_s=5.0,
            gate_index_before=0,
            gate_index_after=0,
            cleanup_confirmed=True,
            details={"ticks_sent": 245},
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
    assert kwargs == {
        "preflight_before_powered_stage": False,
        "write_diagnostic_pngs": False,
        "run_manifest_sha256": result["run_manifest_sha256"],
    }
    run_directory = Path(record).parent
    manifest = json.loads((run_directory / "run-manifest.json").read_text())
    terminal = json.loads((run_directory / "result.json").read_text())
    assert manifest["run_id"] == terminal["run_id"] == result["run_id"]
    assert manifest["authorization"]["scope"] == {
        "stage": requested_stage,
        "attempt_limit": 1,
    }
    assert manifest["execution"]["stage"] == requested_stage
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
