import json
import os
import sys
from pathlib import Path

import pytest

import scripts.aigp_vq2_passive_probe as probe


def _successful_stage_result():
    return {
        "stage": "preflight",
        "success": True,
        "cleanup_confirmed": True,
        "details": {
            "requested_healthy_dwell_s": probe.HEALTHY_DWELL_S,
            "healthy_dwell_s": probe.HEALTHY_DWELL_S + 0.01,
            "mavlink_outbound_audit": {"disallowed_count": 0},
            "mavlink_ingress_stats": {"dropped": 0},
            "replay_capture": {
                "complete": True,
                "dropped": 0,
                "decoded_frames_dropped": 0,
                "writer_errors": 0,
            },
        },
    }


def test_runner_command_is_capture_loaded_preflight_only(tmp_path):
    command = probe._runner_command(
        Path(sys.executable),
        tmp_path / "capture.jsonl.gz",
        tmp_path / "capture.vq2replay",
    )

    assert command[command.index("--stage") + 1] == "preflight"
    assert "--recording-approved" in command
    assert command[command.index("--preflight-healthy-dwell-s") + 1] == "5.0"
    assert not any(
        forbidden in command
        for forbidden in ("sign-id", "hover", "gate0", "gate0-observe")
    )


def test_stage_result_requires_dwell_capture_ingress_and_zero_outbound(tmp_path):
    path = tmp_path / "stdout.json"
    value = _successful_stage_result()
    path.write_text(json.dumps(value), encoding="utf-8")
    assert probe._parse_stage_result(path) == value

    mutations = [
        ("mavlink_outbound_audit", "disallowed_count", 1, "outbound"),
        ("mavlink_outbound_audit", "disallowed_count", False, "outbound"),
        ("mavlink_ingress_stats", "dropped", 1, "overflowed"),
        ("replay_capture", "decoded_frames_dropped", 1, "not complete"),
    ]
    for section, field, replacement, message in mutations:
        changed = _successful_stage_result()
        changed["details"][section][field] = replacement
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(probe.PassiveProbeError, match=message):
            probe._parse_stage_result(path)

    short = _successful_stage_result()
    short["details"]["healthy_dwell_s"] = 4.999
    path.write_text(json.dumps(short), encoding="utf-8")
    with pytest.raises(probe.PassiveProbeError, match="requested dwell"):
        probe._parse_stage_result(path)


def test_stage_result_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "stdout.json"
    path.write_text('{"stage":"preflight","stage":"preflight"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        probe._parse_stage_result(path)


def test_process_row_validation_is_exact():
    valid = {
        "pid": 12,
        "parent_pid": 1,
        "path": r"C:\FlightSim.exe",
        "command_line": r"C:\FlightSim.exe -arg",
        "creation_time": "2026-07-20T12:00:00.0000000Z",
        "session_id": 1,
        "cpu_total_ns": 10,
        "working_set_bytes": 20,
        "responding": True,
        "main_window_handle": 30,
        "main_window_title": "FlightSim",
    }
    assert probe._validated_process_row(valid) == valid

    with pytest.raises(probe.PassiveProbeError, match="pid is invalid"):
        probe._validated_process_row({**valid, "pid": True})
    with pytest.raises(probe.PassiveProbeError, match="missing or unknown"):
        probe._validated_process_row({**valid, "extra": 1})


def test_process_identity_cannot_change_within_probe():
    initial = {
        "launcher": {
            "pid": 1,
            "creation_time": "first",
            "path": "launcher",
            "session_id": 2,
        },
        "payload": {
            "pid": 3,
            "creation_time": "first",
            "path": "payload",
            "session_id": 2,
        },
        "launcher_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
    }
    probe._require_same_process_identity(initial, initial)

    changed = {**initial, "payload": {**initial["payload"], "pid": 4}}
    with pytest.raises(probe.PassiveProbeError, match="identity changed"):
        probe._require_same_process_identity(initial, changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("metadata", "commit_hash"),
        ("dataset", "analysis identity"),
        ("code", "analysis identity"),
        ("camera_minimum", "camera_observations_at_least_140"),
        ("imu_minimum", "highres_imu_arrivals_at_least_600"),
    ],
)
def test_capture_binding_rejects_provenance_or_live_minimum_drift(
    mutation, message
):
    git_state = {
        "commit_hash": "a" * 40,
        "dirty_diff_hash": "b" * 64,
        "code_hash": "c" * 64,
    }
    manifest = {
        "metadata": {
            "simulator_build": "3385",
            "simulator_mode": "Training",
            "simulator_mode_basis": "operator-attested-2026-07-20",
            "stage": "preflight",
            "preflight_healthy_dwell_s": 5.0,
            "mavlink_address": "udpin:127.0.0.1:14550",
            **git_state,
        }
    }
    verification = {"dataset_hash": "d" * 64}
    checks = {
        "generic_passive_timing_valid": True,
        "capture_complete": True,
        "five_second_healthy_dwell": True,
        "camera_capture_shortfalls_zero": True,
        "camera_observations_at_least_140": True,
        "highres_imu_arrivals_at_least_600": True,
        "disallowed_outbound_zero": True,
        "ingress_queue_drops_zero": True,
        "ingress_queue_capacity_proved": True,
    }
    analysis = {
        "dataset_hash": "d" * 64,
        "code_hash": "c" * 64,
        "acceptance_checks": checks,
    }
    if mutation == "metadata":
        manifest["metadata"]["commit_hash"] = "f" * 40
    elif mutation == "dataset":
        analysis["dataset_hash"] = "e" * 64
    elif mutation == "code":
        analysis["code_hash"] = "e" * 64
    elif mutation == "camera_minimum":
        checks["camera_observations_at_least_140"] = False
    else:
        checks["highres_imu_arrivals_at_least_600"] = False

    with pytest.raises(probe.PassiveProbeError, match=message):
        probe._require_capture_binding(
            manifest=manifest,
            git_state=git_state,
            verification=verification,
            analysis=analysis,
        )


@pytest.mark.parametrize(
    "mutation",
    ["active", "missing_release", "wrong_wrapper", "wrong_child"],
)
def test_released_lease_must_bind_exact_wrapper_and_child(mutation):
    evidence = {
        "phase": "released",
        "released_wall_time_ns": 10,
        "wrapper_pid": os.getpid(),
        "child_pid": 9876,
    }
    if mutation == "active":
        evidence["phase"] = "postcheck"
    elif mutation == "missing_release":
        evidence["released_wall_time_ns"] = None
    elif mutation == "wrong_wrapper":
        evidence["wrapper_pid"] += 1
    else:
        evidence["child_pid"] += 1

    with pytest.raises(probe.PassiveProbeError, match="released cleanly"):
        probe._require_released_lease(evidence, child_pid=9876)


@pytest.mark.parametrize(
    "window",
    [
        {"valid": False, "visible": False, "minimized": False},
        {"valid": True, "visible": False, "minimized": False},
        {"valid": True, "visible": True, "minimized": True},
    ],
)
def test_process_snapshot_rejects_invalid_hidden_or_minimized_window(
    tmp_path, monkeypatch, window
):
    launcher = tmp_path / "FlightSim.exe"
    payload = tmp_path / "DCGame-Win64-Shipping.exe"
    launcher.write_bytes(b"launcher")
    payload.write_bytes(b"payload")
    monkeypatch.setattr(probe, "LAUNCHER_PATH", launcher)
    monkeypatch.setattr(probe, "PAYLOAD_PATH", payload)
    monkeypatch.setattr(
        probe,
        "_sha256_file",
        lambda path: (
            probe.LAUNCHER_SHA256 if path == launcher else probe.PAYLOAD_SHA256
        ),
    )
    common = {
        "command_line": "sim",
        "creation_time": "2026-07-20T12:00:00Z",
        "session_id": 1,
        "cpu_total_ns": 1,
        "working_set_bytes": 1,
        "responding": True,
        "main_window_title": "FlightSim",
    }
    monkeypatch.setattr(
        probe,
        "_powershell_process_rows",
        lambda: [
            {
                **common,
                "pid": 10,
                "parent_pid": 1,
                "path": str(launcher),
                "main_window_handle": 0,
            },
            {
                **common,
                "pid": 20,
                "parent_pid": 10,
                "path": str(payload),
                "main_window_handle": 30,
            },
        ],
    )
    monkeypatch.setattr(
        probe,
        "_window_context",
        lambda _handle: {
            **window,
            "foreground": False,
            "foreground_window_handle": 0,
        },
    )

    with pytest.raises(probe.PassiveProbeError, match="valid, visible"):
        probe._validated_process_snapshot()


@pytest.mark.parametrize("manifest", [b"{malformed", b'{"complete":true}\n'])
def test_capture_invalidation_preserves_existing_forensic_bytes(tmp_path, manifest):
    root = tmp_path / "session"
    bundle = root / "preflight.vq2replay"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_bytes(manifest)
    (bundle / "records.jsonl").write_bytes(b"original-records\n")

    probe._permanently_invalidate_capture(root, bundle, "rejected")

    assert (bundle / "manifest.json").read_bytes() == manifest
    assert (bundle / "records.jsonl").read_bytes() == b"original-records\n"
    assert (bundle / "capture-invalid.json").is_file()
    assert (root / "capture-invalid.json").is_file()


def test_probe_rejects_relative_output_before_creating_it(monkeypatch):
    monkeypatch.setattr(
        probe,
        "private_path_guard",
        lambda *_args, **_kwargs: pytest.fail("guard should not be reached"),
    )

    with pytest.raises(probe.PassiveProbeError, match="must be absolute"):
        probe.run_probe(
            output_dir=Path("relative-passive-probe-output"),
            python=Path(sys.executable),
            expected_commit="a" * 40,
            recording_approved=True,
            training_mode_attested=True,
        )


class _FakeLease:
    def __init__(self, *, fail_child_heartbeat=False, event_log=None):
        self.active = False
        self.fail_child_heartbeat = fail_child_heartbeat
        self.phases = []
        self.acquire_calls = 0
        self.release_calls = 0
        self.event_log = event_log

    def acquire(self):
        self.acquire_calls += 1
        self.active = True
        if self.event_log is not None:
            self.event_log.append("acquire")
        return self

    def release(self):
        self.release_calls += 1
        if self.event_log is not None:
            self.event_log.append("release")
        self.active = False

    def heartbeat(self, *, phase, child_pid=None):
        self.phases.append((phase, child_pid, self.active))
        if phase == "child_running" and self.fail_child_heartbeat:
            raise probe.PassiveProbeError("heartbeat publication failed")


class _CompletedChild:
    pid = 9876
    returncode = 0

    def poll(self):
        return self.returncode


class _RunningChild:
    pid = 9876

    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        assert timeout == 2.0
        return self.returncode

    def kill(self):
        self.returncode = -9


class _UnkillableChild:
    pid = 9876
    returncode = None

    def __init__(self):
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminate_calls += 1
        raise OSError("terminate denied")

    def wait(self, timeout=None):
        raise probe.subprocess.TimeoutExpired("probe", timeout)

    def kill(self):
        self.kill_calls += 1
        raise OSError("kill denied")


def _install_common_probe_fakes(monkeypatch, lease, evidence_root):
    monkeypatch.setattr(probe, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(probe, "live_simulator_lease", lambda *_a, **_k: lease)
    monkeypatch.setattr(probe, "_qpc_frequency_hz", lambda: 10_000_000)
    monkeypatch.setattr(
        probe,
        "_require_clean_expected_commit",
        lambda commit: {
            "commit_hash": commit,
            "dirty_diff_hash": "b" * 64,
            "code_hash": "c" * 64,
        },
    )
    monkeypatch.setattr(
        probe,
        "_validated_process_snapshot",
        lambda: {"launcher": {"pid": 1}, "payload": {"pid": 2}},
    )
    monkeypatch.setattr(
        probe,
        "_require_receive_ports_free",
        lambda: {"14550": True, "5600": True},
    )
    monkeypatch.setattr(
        probe,
        "load_live_lease_evidence",
        lambda _path: {
            "schema": "aigp-vq2-live-lease-evidence/1",
            "phase": "released",
            "released_wall_time_ns": 10,
            "wrapper_pid": os.getpid(),
            "child_pid": 9876,
        },
    )
    monkeypatch.setattr(probe, "_sha256_file", lambda _path: "d" * 64)

    class Reader:
        def __init__(self, _path):
            self.manifest = {
                "metadata": {
                    "simulator_build": "3385",
                    "simulator_mode": "Training",
                    "simulator_mode_basis": "operator-attested-2026-07-20",
                    "stage": "preflight",
                    "preflight_healthy_dwell_s": 5.0,
                    "mavlink_address": "udpin:127.0.0.1:14550",
                    "commit_hash": "a" * 40,
                    "dirty_diff_hash": "b" * 64,
                    "code_hash": "c" * 64,
                }
            }

        def verify(self, *, verify_frames):
            assert verify_frames is True
            return {"dataset_hash": "e" * 64, "records": 1000}

    monkeypatch.setattr(probe, "ReplayBundleReader", Reader)
    monkeypatch.setattr(
        probe,
        "analyze_bundle",
        lambda _path: {
            "dataset_hash": "e" * 64,
            "code_hash": "c" * 64,
            "acceptance_checks": {
                "generic_passive_timing_valid": True,
                "capture_complete": True,
                "five_second_healthy_dwell": True,
                "camera_capture_shortfalls_zero": True,
                "camera_observations_at_least_140": True,
                "highres_imu_arrivals_at_least_600": True,
                "disallowed_outbound_zero": True,
                "ingress_queue_drops_zero": True,
                "ingress_queue_capacity_proved": True,
            },
        },
    )


def test_successful_probe_holds_lease_through_child_and_postcheck(
    tmp_path, monkeypatch
):
    lease = _FakeLease()
    _install_common_probe_fakes(monkeypatch, lease, tmp_path)
    postcheck_observations = []

    def ports_after():
        postcheck_observations.append(lease.active)
        return {"14550": True, "5600": True}

    monkeypatch.setattr(probe, "_receive_port_state", ports_after)

    def popen(command, **kwargs):
        assert lease.active
        assert command[command.index("--stage") + 1] == "preflight"
        kwargs["stdout"].write(json.dumps(_successful_stage_result()))
        kwargs["stdout"].flush()
        return _CompletedChild()

    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    output = tmp_path / "accepted-session"
    result = probe.run_probe(
        output_dir=output,
        python=Path(sys.executable),
        expected_commit="a" * 40,
        recording_approved=True,
        training_mode_attested=True,
    )

    assert result["success"] is True
    assert result["failure"] is None
    assert postcheck_observations == [True]
    assert lease.active is False
    assert lease.acquire_calls == 1
    assert lease.release_calls == 1
    assert [phase for phase, _pid, _active in lease.phases] == [
        "starting_child",
        "child_running",
        "postcheck",
    ]
    persisted = json.loads((output / "probe-context.json").read_text("utf-8"))
    assert persisted["success"] is True
    assert persisted["finished_at"] is not None


def test_body_failure_stops_child_and_checks_ports_before_releasing_lease(
    tmp_path, monkeypatch
):
    lease = _FakeLease(fail_child_heartbeat=True)
    _install_common_probe_fakes(monkeypatch, lease, tmp_path)
    child = _RunningChild()
    postcheck_observations = []

    def ports_after():
        postcheck_observations.append((lease.active, child.poll()))
        return {"14550": True, "5600": True}

    monkeypatch.setattr(probe, "_receive_port_state", ports_after)
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_a, **_k: child)
    output = tmp_path / "failed-session"

    with pytest.raises(probe.PassiveProbeError, match="heartbeat publication"):
        probe.run_probe(
            output_dir=output,
            python=Path(sys.executable),
            expected_commit="a" * 40,
            recording_approved=True,
            training_mode_attested=True,
        )

    assert child.terminated is True
    assert postcheck_observations == [(True, -15)]
    assert lease.active is False
    assert lease.release_calls == 1
    persisted = json.loads((output / "probe-context.json").read_text("utf-8"))
    assert persisted["success"] is False
    assert "heartbeat publication failed" in persisted["failure"]
    assert persisted["runner_exit_code"] == -15
    assert (output / "capture-invalid.json").is_file()
    assert (output / "preflight.vq2replay" / "capture-invalid.json").is_file()


def test_final_leased_observation_order_is_process_git_port_then_release(
    tmp_path, monkeypatch
):
    events = []
    lease = _FakeLease(event_log=events)
    _install_common_probe_fakes(monkeypatch, lease, tmp_path)

    monkeypatch.setattr(
        probe,
        "_validated_process_snapshot",
        lambda: (
            events.append("process")
            or {"launcher": {"pid": 1}, "payload": {"pid": 2}}
        ),
    )

    def git_state(commit):
        events.append("git")
        return {
            "commit_hash": commit,
            "dirty_diff_hash": "b" * 64,
            "code_hash": "c" * 64,
        }

    monkeypatch.setattr(probe, "_require_clean_expected_commit", git_state)
    monkeypatch.setattr(
        probe,
        "_receive_port_state",
        lambda: events.append("port") or {"14550": True, "5600": True},
    )

    def popen(_command, **kwargs):
        kwargs["stdout"].write(json.dumps(_successful_stage_result()))
        kwargs["stdout"].flush()
        return _CompletedChild()

    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    probe.run_probe(
        output_dir=tmp_path / "ordered-session",
        python=Path(sys.executable),
        expected_commit="a" * 40,
        recording_approved=True,
        training_mode_attested=True,
    )

    assert events[-4:] == ["process", "git", "port", "release"]


def test_unproved_child_exit_poison_retains_lease(tmp_path, monkeypatch):
    lease = _FakeLease()
    _install_common_probe_fakes(monkeypatch, lease, tmp_path)
    child = _UnkillableChild()
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(probe, "RUNNER_TIMEOUT_S", 0.0)
    monkeypatch.setattr(
        probe, "_receive_port_state", lambda: {"14550": True, "5600": True}
    )
    output = tmp_path / "poisoned-session"

    with pytest.raises(probe.PassiveProbeError, match="exceeded timeout"):
        probe.run_probe(
            output_dir=output,
            python=Path(sys.executable),
            expected_commit="a" * 40,
            recording_approved=True,
            training_mode_attested=True,
        )

    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert lease.acquire_calls == 1
    assert lease.release_calls == 0
    assert lease.active is True
    assert (tmp_path / probe.PROBE_POISON_FILENAME).is_file()
    context = json.loads((output / "probe-context.json").read_text("utf-8"))
    assert context["lease_release_permitted"] is False
