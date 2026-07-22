from __future__ import annotations

import copy
import hashlib
import ntpath
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from scripts import aigp_vq2_powered_attempt as contract
from scripts import aigp_vq2_powered_calibration_probe as probe


H = "a" * 64
H2 = "b" * 64
H3 = "c" * 64
PROBE_HASH = "e" * 64
IMPORT_AUDIT_HASH = "1" * 64
COMMIT = "d" * 40
TREE = "f" * 40
UTC = "2026-07-20T12:34:56.123456Z"
LIVE_WORKTREE = (
    r"C:\Users\John\aigp-worktrees"
    r"\wt-package2-f03-powered-calibration-attempt-live"
)
PYTHON = r"C:\Users\John\killallhumans\.venv\Scripts\python.exe"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def _identity(path: str, digest: str = H) -> dict[str, object]:
    return {"path": path, "sha256": digest}


def _implementation_inventory() -> dict[str, object]:
    return {
        "schema": "aigp-vq2-powered-implementation-inventory/1",
        "commit": COMMIT,
        "tree": TREE,
        "entries": [
            {
                "path": "scripts/aigp_vq2_powered_calibration_probe.py",
                "size_bytes": 123,
                "sha256": PROBE_HASH,
            },
            {
                "path": "scripts/aigp_vq2_powered_import_audit.py",
                "size_bytes": 456,
                "sha256": IMPORT_AUDIT_HASH,
            },
        ],
    }


def _environment_inventory() -> dict[str, object]:
    return {
        "schema": "aigp-vq2-powered-environment-inventory/1",
        "created_at_utc": UTC,
        "variables": [
            {
                "name": "PYTHONDONTWRITEBYTECODE",
                "defined": True,
                "value_sha256": hashlib.sha256(b"1").hexdigest(),
            },
            {
                "name": "PYTHONNOUSERSITE",
                "defined": True,
                "value_sha256": hashlib.sha256(b"1").hexdigest(),
            },
        ],
    }


def _import_inventory() -> dict[str, object]:
    return {
        "schema": "aigp-vq2-powered-import-inventory/1",
        "python_sha256": H,
        "seeds": [
            "scripts.aigp_vq2_powered_attempt",
            "scripts.aigp_vq2_powered_calibration_analysis",
            "scripts.aigp_vq2_powered_calibration_probe",
            "scripts.aigp_vq2_powered_cleanup",
            "scripts.aigp_vq2_powered_runtime",
            "scripts.aigp_vq2_run",
        ],
        "entries": [
            {
                "module": "scripts.aigp_vq2_powered_calibration_probe",
                "origin": LIVE_WORKTREE
                + r"\scripts\aigp_vq2_powered_calibration_probe.py",
                "size_bytes": 123,
                "sha256": PROBE_HASH,
                "root_class": "candidate",
                "namespace_roots": [],
            }
        ],
    }


def _freeze() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    implementation = _implementation_inventory()
    environment = _environment_inventory()
    imports = _import_inventory()
    implementation_path = contract.EVIDENCE_ROOT + r"\implementation.json"
    environment_path = contract.EVIDENCE_ROOT + r"\environment.json"
    imports_path = contract.EVIDENCE_ROOT + r"\imports.json"
    launcher_script = LIVE_WORKTREE + r"\scripts\launch_sim.ps1"
    launcher = r"C:\Users\John\AIGP\AIGP_3385\FlightSim.exe"
    payload = (
        r"C:\Users\John\AIGP\AIGP_3385\FlightSim\Binaries\Win64"
        r"\FlightSim-Win64-Shipping.exe"
    )
    semantic = {name: implementation[name] for name in ("commit", "tree", "entries")}
    value = {
        "schema": "aigp-vq2-powered-calibration-live-freeze/1",
        "task_id": contract.TASK_ID,
        "freeze_id": "vq2-package2-f03-powered-calibration-attempt-f03-a01-live-freeze",
        "candidate": {
            "commit": COMMIT,
            "code_sha256": contract.canonical_object_sha256(semantic),
            "live_worktree": LIVE_WORKTREE,
            "detached_head_required": True,
            "clean_tracked_untracked_ignored_required": True,
            "implementation_inventory": _identity(
                implementation_path, contract.canonical_file_sha256(implementation)
            ),
        },
        "session": {
            "session_id": contract.SESSION_ID,
            "attempt_id": contract.ATTEMPT_ID,
            "attempt_limit": 1,
            "split": "discovery_fit",
        },
        "inputs": {
            "target_config": {
                "schema": "aigp-vq2-sim-calibration-collection-config/1",
                **_identity(contract.EVIDENCE_ROOT + r"\target.json"),
            },
            "capture_authorization": {
                "schema": "aigp-vq2-simulation-capture-authorization/1",
                **_identity(contract.EVIDENCE_ROOT + r"\authorization.json", H2),
            },
            "excitation_plan": {
                "schema": contract.EXCITATION_PLAN_SCHEMA,
                "plan_id": contract.EXCITATION_PLAN_ID,
                "path": contract.EVIDENCE_ROOT + r"\plan.json",
                "sha256": contract.EXCITATION_PLAN_SHA256,
            },
        },
        "runtime": {
            "python": {
                "path": PYTHON,
                "implementation": "CPython",
                "version": "3.12.2",
                "sha256": H,
            },
            "powershell": {
                "path": POWERSHELL,
                "product_version": "5.1.0",
                "sha256": H2,
            },
            "development_test_lock": _identity(
                LIVE_WORKTREE + r"\requirements\development-test.lock.txt", H3
            ),
            "environment_inventory": _identity(
                environment_path, contract.canonical_file_sha256(environment)
            ),
            "import_inventory": _identity(
                imports_path, contract.canonical_file_sha256(imports)
            ),
        },
        "simulator": {
            "build": 3385,
            "mode": "Training",
            "launcher_script": _identity(launcher_script, H),
            "launcher": _identity(launcher, H2),
            "payload": _identity(payload, H3),
            "topology": "one_launcher_parent_retained_one_payload_child",
            "mode_evidence": "post_topology_local_interactive_attestation",
        },
        "transport": {
            "mavlink_bind": {
                "host": "127.0.0.1",
                "port": 14550,
                "socket_policy": "ipv4-exclusive-address-use",
            },
            "camera_bind": {
                "host": "0.0.0.0",
                "port": 5600,
                "socket_policy": "ipv4-exclusive-address-use",
            },
            "peer_policy": "freeze_first_valid_build3385_source",
            "allowed_outbound_categories": [
                "arm",
                "attitude_target",
                "disarm",
                "gcs_heartbeat",
                "sim_reset",
                "timesync",
            ],
            "unknown_category_policy": "invalidate",
        },
        "execution": {
            "wrapper_cwd": LIVE_WORKTREE,
            "security_environment": {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "forbidden_defined": ["PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"],
            },
            "launcher_cwd": LIVE_WORKTREE,
            "launcher_argv": [
                POWERSHELL,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                launcher_script,
                "-SimulatorPath",
                launcher,
                "-TaskName",
                "AIGP-P2-F03-A01-Launch",
                "-StartupTimeoutSeconds",
                "25",
            ],
            "launcher_environment_sha256": contract.environment_variables_sha256(
                environment
            ),
            "child_cwd": LIVE_WORKTREE,
            "cleanup_cwd": LIVE_WORKTREE,
        },
        "paths": contract.frozen_paths(),
        "deadline_durations_ns": dict(contract.DEADLINE_DURATIONS_NS),
    }
    documents = {
        implementation_path: implementation,
        environment_path: environment,
        imports_path: imports,
    }
    return value, documents


def _path_proof(path: str, kind: str = "file") -> probe.PathProof:
    return probe.PathProof(
        path=path,
        final_path=path,
        kind=kind,
        volume_id="volume-1",
    )


class FakeOffline:
    def __init__(self, tmp_path: Path, *, freeze: dict[str, object], documents):
        self.tmp_path = tmp_path
        self.freeze = freeze
        self.documents = dict(documents)
        self.documents[freeze["paths"]["live_freeze"]] = freeze
        self.calls: list[str] = []
        self.events: list[str] | None = None
        self.identity_hashes: dict[str, str] = {}
        for section in (freeze["inputs"], freeze["runtime"], freeze["simulator"]):
            for value in section.values():
                if isinstance(value, dict) and "path" in value and "sha256" in value:
                    self.identity_hashes[value["path"]] = value["sha256"]
        self.identity_hashes[freeze["candidate"]["implementation_inventory"]["path"]] = (
            freeze["candidate"]["implementation_inventory"]["sha256"]
        )
        self.identity_hashes[
            LIVE_WORKTREE + r"\scripts\aigp_vq2_powered_calibration_probe.py"
        ] = PROBE_HASH
        self.identity_hashes[
            LIVE_WORKTREE + r"\scripts\aigp_vq2_powered_import_audit.py"
        ] = IMPORT_AUDIT_HASH

    def read_stable_json(self, path: str) -> probe.StableJsonProof:
        self.calls.append(f"json:{path}")
        if self.events is not None and path == self.freeze["paths"]["live_freeze"]:
            self.events.append("offline.admit")
        value = copy.deepcopy(self.documents[path])
        raw = contract.canonical_json_file_bytes(value)
        return probe.StableJsonProof(
            identity=probe.FileIdentityProof(
                path=_path_proof(path),
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            ),
            raw_bytes=raw,
            value=value,
        )

    def observe_file_identity(
        self, path: str, *, hash_kind: str
    ) -> probe.FileIdentityProof:
        self.calls.append(f"identity:{path}")
        plan_path = self.freeze["inputs"]["excitation_plan"]["path"]
        expected_kind = "canonical_object" if path == plan_path else "file_bytes"
        assert hash_kind == expected_kind
        return probe.FileIdentityProof(
            path=_path_proof(path),
            size_bytes=123,
            sha256=self.identity_hashes[path],
            hash_kind=expected_kind,
        )

    def current_working_directory(self) -> probe.PathProof:
        self.calls.append("cwd")
        return _path_proof(LIVE_WORKTREE, "directory")

    def module_origin(self, module_name: str) -> probe.PathProof:
        self.calls.append(f"module:{module_name}")
        return _path_proof(
            LIVE_WORKTREE + r"\scripts\aigp_vq2_powered_calibration_probe.py"
        )

    def git_worktree(self, path: str) -> probe.GitWorktreeProof:
        self.calls.append("git")
        return probe.GitWorktreeProof(
            worktree_path=path,
            head_commit=COMMIT,
            head_tree=TREE,
            detached_head=True,
            tracked_clean=True,
            untracked_clean=True,
            ignored_clean=True,
        )

    def security_environment(self):
        self.calls.append("environment-security")
        return {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHOME": None,
            "PYTHONPATH": None,
            "PYTHONSTARTUP": None,
        }

    def rederive_implementation_inventory(self, frozen_inventory):
        self.calls.append("implementation-rederive")
        return copy.deepcopy(frozen_inventory)

    def rederive_environment_inventory(self, frozen_inventory):
        self.calls.append("environment-rederive")
        value = copy.deepcopy(frozen_inventory)
        value["created_at_utc"] = "2026-07-20T12:35:00.000000Z"
        return value

    def rederive_import_inventory(
        self, frozen_inventory, eager_modules, *, environment_inventory
    ):
        self.calls.append("import-rederive")
        assert tuple(eager_modules) == probe.POWERED_EAGER_IMPORT_MODULES
        assert environment_inventory["variables"] == _environment_inventory()[
            "variables"
        ]
        return probe.ImportRevalidation(
            inventory=copy.deepcopy(frozen_inventory),
            origins_reverified=True,
            user_site_on_sys_path=False,
        )


def _admission(tmp_path: Path):
    freeze, documents = _freeze()
    service = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    freeze_hash = contract.canonical_file_sha256(freeze)
    arguments = probe.ProbeArguments(
        live_freeze=freeze["paths"]["live_freeze"],
        live_freeze_sha256=freeze_hash,
        expected_commit=COMMIT,
    )
    return freeze, service, probe.admit_offline(arguments, service)


def test_native_environment_entry_parser_rejects_hidden_drive_state():
    environment: dict[str, str] = {}
    with pytest.raises(ValueError, match="not spawn-safe"):
        probe._append_native_environment_entry(
            environment,
            r"=C:=C:\hidden-drive-state",
        )
    assert environment == {}
    probe._append_native_environment_entry(
        environment,
        r"=C:=C:\hidden-drive-state",
        allow_drive_state=True,
    )
    assert environment == {"=C:": r"C:\hidden-drive-state"}

    probe._append_native_environment_entry(environment, r"Path=C:\Windows")
    assert environment == {
        "=C:": r"C:\hidden-drive-state",
        "PATH": r"C:\Windows",
    }
    with pytest.raises(ValueError, match="not spawn-safe"):
        probe._append_native_environment_entry(environment, r"PATH=C:\Other")


class FakeSecure:
    def __init__(self, tmp_path: Path, freeze: dict[str, object]):
        self.physical_root = tmp_path / "evidence"
        self.physical_root.mkdir()
        self.freeze = freeze
        self.failures: dict[str, str] = {}
        self.file_calls: list[str] = []
        self.directory_calls: list[str] = []
        self.open_directory_calls: list[str] = []
        self.inspect_calls = 0
        self.events: list[str] | None = None
        self.clock: StepClock | None = None
        self.advance_after_write_ns: dict[str, int] = {}
        self.advance_after_phase_end: tuple[str, int] | None = None
        self.bad_dacl = False
        self.physical(ntpath.dirname(freeze["paths"]["split_registry"])).mkdir(
            parents=True
        )

    def physical(self, logical: str) -> Path:
        relative = ntpath.relpath(logical, self.freeze["paths"]["evidence_root"])
        if relative == ".":
            return self.physical_root
        return self.physical_root.joinpath(*relative.split("\\"))

    def inspect_attempt_root(self, paths):
        self.inspect_calls += 1
        return probe.AttemptRootSnapshot(
            evidence_root=paths["evidence_root"],
            live_poison_present=self.physical(paths["live_poison"]).exists(),
            target_attempt_directory_present=self.physical(paths["attempt_dir"]).exists(),
            target_attempt_envelope_present=self.physical(paths["attempt_envelope"]).exists(),
        )

    def open_private_directory(self, path: str, *, parent_path: str):
        self.open_directory_calls.append(path)
        target = self.physical(path)
        if not target.is_dir():
            raise probe.BoundaryCreateNewError(path, state="absent")
        return probe.SecureDirectoryReceipt(
            path=path,
            final_path=path,
            parent_final_path=parent_path,
            volume_id="volume-1",
            parent_volume_id="volume-1",
            owner_id="sid-current",
            current_user_id="sid-current",
            created_new=False,
            owner_is_current_user=True,
            current_user_only_dacl=True,
            dacl_applied_at_create=True,
            non_reparse=True,
            ancestors_non_reparse=True,
            retained_handle=True,
        )

    def create_private_directory_create_new(self, path: str, *, parent_path: str):
        self.directory_calls.append(path)
        if self.events is not None:
            self.events.append(f"secure.mkdir:{ntpath.basename(path)}")
        target = self.physical(path)
        target.mkdir(parents=False, exist_ok=False)
        return probe.SecureDirectoryReceipt(
            path=path,
            final_path=path,
            parent_final_path=parent_path,
            volume_id="volume-1",
            parent_volume_id="volume-1",
            owner_id="sid-current",
            current_user_id="sid-current",
            created_new=True,
            owner_is_current_user=True,
            current_user_only_dacl=not self.bad_dacl,
            dacl_applied_at_create=True,
            non_reparse=True,
            ancestors_non_reparse=True,
            retained_handle=True,
        )

    def create_new_file(self, path, payload, *, parent, deadline_monotonic_ns):
        self.file_calls.append(path)
        if self.events is not None:
            self.events.append(f"secure.write:{ntpath.basename(path)}")
        target = self.physical(path)
        state = self.failures.get(path)
        if state == "partial":
            with target.open("xb") as handle:
                partial = payload[: max(1, len(payload) // 2)]
                handle.write(partial)
                handle.flush()
                os.fsync(handle.fileno())
            raise probe.BoundaryCreateNewError(
                path,
                state="partial",
                observed_sha256=hashlib.sha256(partial).hexdigest(),
            )
        if state == "absent":
            raise probe.BoundaryCreateNewError(path, state="absent")
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        observed = target.read_bytes()
        advance = self.advance_after_write_ns.get(path, 0)
        if self.advance_after_phase_end is not None:
            phase, phase_advance = self.advance_after_phase_end
            if (
                b'"event":"phase_end"' in payload
                and f'"phase":"{phase}"'.encode() in payload
            ):
                advance += phase_advance
        if advance:
            assert self.clock is not None
            self.clock.advance(advance)
        return probe.CreateNewFileReceipt(
            path=path,
            final_path=path,
            parent_final_path=parent.final_path,
            volume_id="volume-1",
            parent_volume_id="volume-1",
            owner_id="sid-current",
            current_user_id="sid-current",
            size_bytes=len(observed),
            sha256=hashlib.sha256(observed).hexdigest(),
            completed_monotonic_ns=min(1_000, deadline_monotonic_ns - 1),
            created_new=True,
            regular_file=True,
            owner_is_current_user=True,
            current_user_only_dacl=True,
            dacl_applied_at_create=True,
            non_reparse=True,
            ancestors_non_reparse=True,
            flushed=True,
            readback_verified=True,
        )


class FakeClock:
    def __init__(self, values):
        self.values = iter(values)

    def now_ns(self):
        return next(self.values)

    def query_performance_frequency_hz(self):
        return 10_000_000


class StepClock:
    def __init__(self, start: int = 100, step: int = 1_000_000):
        self.value = start - step
        self.step = step

    def now_ns(self):
        self.value += self.step
        return self.value

    def query_performance_frequency_hz(self):
        return 10_000_000

    def advance(self, nanoseconds: int) -> None:
        self.value += nanoseconds


class RecordingValidators:
    """Keep orchestration tests about ordering; contract tests cover schemas."""

    def __init__(self):
        self.calls: list[str] = []

    def _accept(self, name: str, value: Any):
        self.calls.append(name)
        assert isinstance(value, dict)
        return copy.deepcopy(value)

    def process_proof(self, value):
        return self._accept("process_proof", value)

    def training_attestation(self, value, *, process_proof):
        assert process_proof["phase"] == "prechild"
        return self._accept("training_attestation", value)

    def process_authority(self, value, *, attempt, argv):
        assert attempt["context"]["attempt_id"] == contract.ATTEMPT_ID
        assert argv
        return self._accept("process_authority", value)

    def lease_final(self, value):
        return self._accept("lease_final", value)

    def bundle_verification(self, value):
        return self._accept("bundle_verification", value)

    def capture_seal(self, value):
        return self._accept("capture_seal", value)

    def analysis_report(self, value):
        return self._accept("analysis_report", value)

    def split_claim(self, value):
        return self._accept("split_claim", value)

    def split_registry(self, value, *, split_claim):
        assert split_claim["kind"] == "claim"
        return self._accept("split_registry", value)

    def complete_terminal(self, value, *, lifecycle):
        assert lifecycle["schema"] == "aigp-vq2-powered-wrapper-lifecycle/1"
        return self._accept("complete_terminal", value)

    def live_poison(self, value):
        return self._accept("live_poison", value)

    def invalid_terminal(self, value):
        return self._accept("invalid_terminal", value)


def _sealed_child_artifacts() -> dict[str, object]:
    return {
        "legacy_record": "closed",
        "legacy_record_sha256": H,
        "replay_bundle": "sealed",
        "replay_dataset_hash": H,
        "replay_manifest_sha256": H2,
        "replay_records_sha256": H3,
    }


class FakeLive:
    """One deterministic injected implementation for wrapper-only tests."""

    def __init__(self):
        self.events: list[str] = []
        self.failures: dict[str, Exception] = {}
        self.clock: StepClock | None = None
        self.random_index = 0
        self.launcher_heartbeat_advances: tuple[int, ...] = ()
        self.launcher_silent_advance = 0
        self.acquire_silent_advance = 0
        self.launcher_heartbeat_period_ns: int | None = None
        self.fallback_gate_proved = True
        self.final_ports_proved = True
        self.final_topology_proved = True
        self.child_outcome = probe.ChildSupervisionOutcome(
            cleanup_proved=True,
            collection_valid=True,
            artifact_state_patch=_sealed_child_artifacts(),
        )
        self.fallback_outcome = probe.FallbackSupervisionOutcome(
            cleanup_proved=True
        )
        self.acquired_qpc_frequency_hz: int | None = None
        self.heartbeat_records: list[tuple[str, int, int]] = []
        self.owner_secret_reference: bytearray | None = None
        self.capability_frame_references: list[bytearray] = []
        self.release_callback_attempts = 0
        self.complete_context: dict[str, Any] | None = None
        self.invalid_context: dict[str, Any] | None = None
        self.poison_contexts: list[dict[str, Any]] = []
        self.postrelease_phase_deadlines: dict[str, dict[str, Any]] = {}
        self.tree_empty_handles: set[str] = set()

    def _call(self, name: str) -> None:
        self.events.append(name)
        failure = self.failures.get(name)
        if failure is not None:
            raise failure

    def utc_now(self):
        self._call("host.utc_now")
        return UTC

    def host_boot_id_sha256(self):
        self._call("host.boot_id")
        return H3

    def token_bytes(self, size):
        self._call("csprng.token_bytes")
        value = bytes((76 + self.random_index,)) * size
        self.random_index += 1
        return value

    def current_wrapper_identity(self):
        self._call("process.wrapper_identity")
        return _process()

    def retain_and_reprove(self, identity):
        self._call("process.retain_wrapper")
        return {"retained": copy.deepcopy(identity)}

    def prove_prechild_identity(
        self,
        retained_wrapper,
        *,
        topology_proof,
        deadline_monotonic_ns,
        heartbeat,
    ):
        self._call("process.prechild_identity")
        assert retained_wrapper["retained"] == _process()
        assert topology_proof["phase"] == "prechild"
        return {"proved": True}

    def prove_child_tree_exit(
        self, child, *, deadline_monotonic_ns, heartbeat
    ):
        self._call("process.child_tree_exit")
        self.tree_empty_handles.add(child)
        return {"proved": True}

    def prove_final_process_state(self, *, deadline_monotonic_ns, heartbeat):
        self._call("process.final")
        return {"phase": "postchild", "proof": "exact"}

    def allocate_attempt_handles(self, wrapper_process):
        self._call("spawn.allocate_handles")
        assert wrapper_process == _process()
        return probe.AttemptHandleSet(41, 42, 43, 44)

    def seal_spawn_environment(self, *, deadline_monotonic_ns):
        self._call("spawn.seal_environment")
        assert self.clock is not None
        assert self.clock.value < deadline_monotonic_ns

    def launch_and_wait(
        self, *, freeze, deadline_monotonic_ns, heartbeat
    ):
        self._call("launcher.wait")
        assert freeze["simulator"]["build"] == 3385
        assert self.clock is not None
        self.launcher_heartbeat_period_ns = heartbeat.period_ns
        for advance in self.launcher_heartbeat_advances:
            self.clock.advance(advance)
            heartbeat()
        self.clock.advance(self.launcher_silent_advance)
        return {"disposition": "accepted"}

    def prove_topology(
        self, *, launch_result, deadline_monotonic_ns, heartbeat
    ):
        self._call("topology.prechild")
        assert launch_result["disposition"] == "accepted"
        return {"phase": "prechild", "proof": "exact"}

    def prove_unchanged(
        self, *, launch_result, deadline_monotonic_ns, heartbeat
    ):
        self._call("topology.unchanged")
        return {
            "proved": self.final_topology_proved,
            "topology": "unchanged",
            "responsive": "yes",
            "scheduled_task": "absent",
        }

    def attest_training(
        self, *, topology_proof, deadline_monotonic_ns, heartbeat
    ):
        self._call("training.attest")
        assert topology_proof["phase"] == "prechild"
        return {"mode": "Training"}

    def prove_prechild_free(self, *, deadline_monotonic_ns, heartbeat):
        self._call("ports.prechild_free")
        return {"proved": True}

    def prove_child_owners(
        self, child, *, deadline_monotonic_ns, heartbeat
    ):
        self._call("ports.child_owners")
        return {"proved": True}

    def prove_fallback_gate(self, *, deadline_monotonic_ns, heartbeat):
        self._call("ports.fallback_gate")
        return {"proved": self.fallback_gate_proved}

    def prove_final_free(self, *, deadline_monotonic_ns, heartbeat):
        self._call("ports.final_free")
        return {
            "proved": self.final_ports_proved,
            "ports": "free",
            "transport": "closed",
        }

    def acquire(
        self,
        *,
        owner_secret,
        qpc_frequency_hz,
        deadline_monotonic_ns,
    ):
        self._call("lease.acquire")
        assert len(owner_secret) == 32
        self.owner_secret_reference = owner_secret
        self.acquired_qpc_frequency_hz = qpc_frequency_hz
        assert self.clock is not None
        self.clock.advance(self.acquire_silent_advance)
        return {"lease": "retained"}

    def heartbeat(self, lease, *, phase, deadline_monotonic_ns):
        self._call(f"lease.heartbeat:{phase}")
        assert lease["lease"] == "retained"
        assert self.clock is not None
        self.heartbeat_records.append(
            (phase, deadline_monotonic_ns, self.clock.value)
        )

    def release_and_verify(
        self, lease, *, deadline_monotonic_ns, heartbeat
    ):
        self._call("lease.release")
        assert self.clock is not None
        self.release_callback_attempts += 1
        heartbeat()
        return probe.LeaseReleaseOutcome(
            kernel_released=True,
            released_monotonic_ns=self.clock.value,
            final_index={"state": "released"},
        )

    def spawn_powered_child_blocked(
        self, *, argv, handles, deadline_monotonic_ns, heartbeat
    ):
        self._call("spawn.child_blocked")
        identity = {**_process(), "pid": 124, "creation_filetime_100ns": 457}
        return probe.BlockedProcess(
            handle="child-handle",
            identity=identity,
            authority={"role": "powered_child"},
        )

    def release_child_capability(
        self, child, *, frame, deadline_monotonic_ns, heartbeat
    ):
        self._call("spawn.release_child")
        assert child == "child-handle" and len(frame) > 32
        self.capability_frame_references.append(frame)

    def spawn_cleanup_fallback_blocked(
        self, *, argv, handles, deadline_monotonic_ns, heartbeat
    ):
        self._call("spawn.fallback_blocked")
        identity = {**_process(), "pid": 125, "creation_filetime_100ns": 458}
        return probe.BlockedProcess(
            handle="fallback-handle",
            identity=identity,
            authority={"role": "cleanup_fallback"},
        )

    def release_cleanup_capability(
        self, child, *, frame, deadline_monotonic_ns, heartbeat
    ):
        self._call("spawn.release_fallback")
        assert child == "fallback-handle" and len(frame) > 32
        self.capability_frame_references.append(frame)

    def abort_blocked_process(
        self, child, *, deadline_monotonic_ns, heartbeat
    ):
        self._call(f"spawn.abort:{child}")
        self.tree_empty_handles.add(child)

    def close_attempt_handles(self, handles, *, deadline_monotonic_ns):
        self._call("spawn.close_attempt_handles")

    def close_process_handle(self, child, *, deadline_monotonic_ns):
        self._call(f"spawn.close_process_handle:{child}")
        if child not in self.tree_empty_handles:
            raise probe.OrchestrationPhaseError(
                "process_residue", "fake process tree is not proved empty"
            )

    def close_retained_wrapper(self, retained_wrapper, *, deadline_monotonic_ns):
        self._call("process.close_retained_wrapper")

    def supervise_powered_child(
        self, child, *, deadline_monotonic_ns, heartbeat
    ):
        self._call("supervision.child")
        return self.child_outcome

    def supervise_cleanup_fallback(
        self, child, *, deadline_monotonic_ns, heartbeat
    ):
        self._call("supervision.fallback")
        self.tree_empty_handles.add(child)
        return self.fallback_outcome

    def verify_bundle(self, *, phase_deadline):
        self._call("post.verify_bundle")
        self.postrelease_phase_deadlines["bundle_verify"] = dict(phase_deadline)
        return {"kind": "bundle_verification"}

    def build_capture_seal(self, *, phase_deadline):
        self._call("post.capture_seal")
        self.postrelease_phase_deadlines["capture_seal"] = dict(phase_deadline)
        return {"kind": "capture_seal"}

    def analyze_capture(self, *, phase_deadline):
        self._call("post.analysis")
        self.postrelease_phase_deadlines["analysis"] = dict(phase_deadline)
        return {"kind": "analysis"}

    def publish_split(self, *, analysis, phase_deadline):
        self._call("post.split")
        self.postrelease_phase_deadlines["split_publish"] = dict(phase_deadline)
        assert analysis["kind"] == "analysis"
        return probe.SplitPublications(
            claim={"kind": "claim"},
            registry={"kind": "registry"},
            report={"kind": "report"},
        )

    def build_complete_terminal(self, *, context):
        self._call("post.complete_terminal")
        self.complete_context = dict(context)
        return {"kind": "complete"}

    def build_live_poison(self, *, context):
        self._call("post.live_poison")
        self.poison_contexts.append(dict(context))
        return {"kind": "poison"}

    def build_invalid_terminal(self, *, context):
        self._call("post.invalid_terminal")
        self.invalid_context = dict(context)
        return {"kind": "invalid"}


def _live_services(fake: FakeLive) -> probe.LiveOrchestrationServices:
    return probe.LiveOrchestrationServices(
        host=fake,
        csprng=fake,
        launcher=fake,
        topology=fake,
        training=fake,
        process=fake,
        ports=fake,
        lease=fake,
        spawn=fake,
        supervision=fake,
        postrelease=fake,
    )


def _run_orchestration(tmp_path: Path, configure=None):
    freeze, documents = _freeze()
    offline = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    secure = FakeSecure(tmp_path, freeze)
    live = FakeLive()
    clock = StepClock()
    live.clock = clock
    secure.clock = clock
    validators = RecordingValidators()
    offline.events = live.events
    secure.events = live.events
    if configure is not None:
        configure(live, secure, freeze)
    orchestrator = probe.ProbeOrchestrator(
        offline=offline,
        secure=secure,
        clock=clock,
        live=_live_services(live),
        validators=validators,
    )
    result = orchestrator.run(
        probe.ProbeArguments(
            freeze["paths"]["live_freeze"],
            contract.canonical_file_sha256(freeze),
            COMMIT,
        )
    )
    return freeze, offline, secure, live, validators, result


def _process() -> dict[str, object]:
    return {
        "pid": 123,
        "creation_filetime_100ns": 456,
        "windows_session_id": 1,
        "image_path": PYTHON,
        "image_sha256": H,
        "argv_sha256": H2,
    }


def _material(admission: probe.OfflineAdmission) -> probe.AttemptMaterial:
    values = iter((b"L" * 32, b"C" * 32, b"F" * 32))
    return probe.build_attempt_material(
        admission=admission,
        wrapper_process=_process(),
        host_boot_id_sha256=H3,
        qpc_frequency_hz=10_000_000,
        handles=probe.AttemptHandleSet(41, 42, 43, 44),
        created_at_utc=UTC,
        wrapper_started_monotonic_ns=100,
        offline_precheck_completed_monotonic_ns=110,
        attempt_publish_started_monotonic_ns=120,
        random_bytes=lambda size: next(values),
    )


def _safe_cleanup() -> dict[str, object]:
    return {
        "child_exit": "not_created",
        "fallback": "not_eligible",
        "ports": "not_opened",
        "lease": "not_acquired",
        "processes": "not_created",
        "transport": "not_opened",
        "scheduled_task": "not_created",
        "simulator_topology": "not_launched",
        "simulator_responsive": "not_launched",
    }


def _safe_artifacts() -> dict[str, object]:
    return {
        "legacy_record": "absent",
        "legacy_record_sha256": None,
        "replay_bundle": "absent",
        "replay_dataset_hash": None,
        "replay_manifest_sha256": None,
        "replay_records_sha256": None,
        "bundle_verification": "absent",
        "bundle_verification_sha256": None,
        "capture_seal": "absent",
        "capture_seal_sha256": None,
        "split_claim": "absent",
        "split_claim_sha256": None,
        "split_registry": "absent",
        "split_registry_sha256": None,
        "analysis_report": "absent",
        "analysis_report_sha256": None,
        "wrapper_lifecycle": "absent",
        "wrapper_lifecycle_sha256": None,
        "attempt_complete": "absent",
        "attempt_complete_partial_sha256": None,
        "terminal_publication": "invalid_record",
        "forensic_bytes_preserved": True,
    }


def _complete_cleanup() -> dict[str, object]:
    return {
        "child_exit": "proved",
        "fallback": "not_required",
        "ports": "free",
        "lease": "released",
        "processes": "exited",
        "transport": "closed",
        "scheduled_task": "absent",
        "simulator_topology": "unchanged",
        "simulator_responsive": "yes",
    }


def _complete_artifacts() -> dict[str, object]:
    value = _safe_artifacts()
    value.update(
        {
            "legacy_record": "closed",
            "legacy_record_sha256": H,
            "replay_bundle": "sealed",
            "replay_dataset_hash": H,
            "replay_manifest_sha256": H,
            "replay_records_sha256": H,
            "bundle_verification": "valid",
            "bundle_verification_sha256": H,
            "capture_seal": "valid",
            "capture_seal_sha256": H,
            "split_claim": "valid",
            "split_claim_sha256": H,
            "split_registry": "valid",
            "split_registry_sha256": H,
            "analysis_report": "valid",
            "analysis_report_sha256": H,
            "wrapper_lifecycle": "valid",
            "wrapper_lifecycle_sha256": H,
        }
    )
    return value


def test_import_is_inert_and_cli_is_exact(monkeypatch):
    assert probe.POWERED_EAGER_IMPORT_MODULES == tuple(
        sorted(probe.POWERED_EAGER_IMPORT_MODULES, key=lambda item: item.encode())
    )
    assert "socket" not in probe.__dict__
    assert "subprocess" not in probe.__dict__
    parser = probe.build_argument_parser()
    assert parser.allow_abbrev is False
    argv = [
        "--live-freeze",
        contract.frozen_paths()["live_freeze"],
        "--live-freeze-sha256",
        H,
        "--expected-commit",
        COMMIT,
    ]
    assert probe.parse_arguments(argv).expected_commit == COMMIT
    with pytest.raises(SystemExit):
        probe.parse_arguments(argv + ["--stage", "calibration-excite"])
    with pytest.raises(SystemExit):
        probe.parse_arguments(argv[:-2])
    abbreviated = argv.copy()
    abbreviated[0] = "--live-f"
    with pytest.raises(SystemExit):
        probe.parse_arguments(abbreviated)


@pytest.mark.skipif(os.name != "nt", reason="powered production module is Win32-only")
def test_real_dash_m_probe_has_production_main_identity_before_cli_refusal(
    monkeypatch,
):
    import ctypes
    import struct
    import winreg

    result = subprocess.run(
        [sys.executable, "-E", "-s", "-B", "-m", probe.PROBE_MODULE],
        cwd=Path.cwd(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    stderr = result.stderr.decode(errors="replace")
    assert result.returncode == 2
    assert "--live-freeze" in stderr
    assert "--live-freeze-sha256" in stderr
    assert "--expected-commit" in stderr
    assert "exact -m production module" not in stderr

    alias_check = subprocess.run(
        [
            sys.executable,
            "-E",
            "-s",
            "-B",
            "-c",
            (
                "import ntpath,runpy,sys\n"
                f"name={probe.PROBE_MODULE!r}\n"
                "try:\n"
                "    runpy.run_module(name,run_name='__main__',alter_sys=True)\n"
                "except SystemExit as exc:\n"
                "    assert exc.code == 2\n"
                "else:\n"
                "    raise AssertionError('production module did not stop at CLI')\n"
                "execution=sys.modules[name]\n"
                "spec=execution.__spec__\n"
                "assert execution.__name__ == '__main__'\n"
                "assert spec.name == name\n"
                "assert execution.__file__ == spec.origin\n"
                "assert ntpath.normpath(spec.loader.get_filename(name)) == "
                "ntpath.normpath(spec.origin)\n"
            ),
        ],
        cwd=Path.cwd(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert alias_check.returncode == 0, alias_check.stderr.decode(errors="replace")

    boundary = object.__new__(probe.WindowsProductionLiveBoundary)
    native_boot_id = boundary.host_boot_id_sha256()
    assert native_boot_id == boundary.host_boot_id_sha256()
    assert len(native_boot_id) == 64
    assert set(native_boot_id) <= set("0123456789abcdef")

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"SOFTWARE\Microsoft\Cryptography",
        0,
        winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
    ) as key:
        machine_guid, _value_type = winreg.QueryValueEx(key, "MachineGuid")

    class FakeNtQuery:
        argtypes = None
        restype = None

        def __init__(self):
            self.status = 0
            self.returned_length = 48
            self.boot_filetime = 0x123456789ABCDEF
            self.calls = []

        def __call__(self, information_class, buffer, length, returned):
            self.calls.append((information_class, length))
            ctypes.memmove(
                buffer,
                struct.pack("<Q", self.boot_filetime),
                ctypes.sizeof(ctypes.c_uint64),
            )
            returned._obj.value = self.returned_length
            return self.status

    query = FakeNtQuery()
    fake_ntdll = type("FakeNtdll", (), {"NtQuerySystemInformation": query})()
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, **_kwargs: fake_ntdll
        if name == "ntdll"
        else pytest.fail(f"unexpected WinDLL request: {name}"),
    )
    expected_boot_id = hashlib.sha256(
        machine_guid.upper().encode("utf-8")
        + b"\x00"
        + struct.pack("<Q", query.boot_filetime)
    ).hexdigest()
    assert boundary.host_boot_id_sha256() == expected_boot_id
    assert query.calls == [(3, 48)]
    assert boundary._SYSTEM_TIME_OF_DAY_INFORMATION_CLASS == 3
    assert boundary._SYSTEM_TIME_OF_DAY_INFORMATION_SIZE == 48

    for status, returned_length in (
        (1, 48),
        (-1073741820, 48),
        (0, 8),
        (0, 47),
        (0, 49),
        (0, 64),
    ):
        query.status = status
        query.returned_length = returned_length
        with pytest.raises(
            probe.OrchestrationPhaseError, match="host boot FILETIME query failed"
        ) as exc_info:
            boundary.host_boot_id_sha256()
        assert exc_info.value.reason_code == "internal_error"

    query.status = 0
    query.returned_length = 48
    query.boot_filetime = 0
    with pytest.raises(
        probe.OrchestrationPhaseError, match="host boot FILETIME is invalid"
    ):
        boundary.host_boot_id_sha256()

    monkeypatch.setitem(sys.modules, "__main__", probe)
    monkeypatch.setitem(sys.modules, probe.PROBE_MODULE, probe)
    with pytest.raises(probe.OfflineAdmissionError, match="populated before binding"):
        probe._bind_production_main_module_alias()


def test_offline_admission_revalidates_every_semantic_and_origin(tmp_path):
    freeze, service, admission = _admission(tmp_path)
    assert admission.live_freeze == freeze
    assert "implementation-rederive" in service.calls
    assert "environment-rederive" in service.calls
    assert "import-rederive" in service.calls
    assert service.calls.index("cwd") < service.calls.index("import-rederive")
    assert service.calls.index("environment-rederive") < service.calls.index(
        "import-rederive"
    )
    expected_module = LIVE_WORKTREE + r"\scripts\aigp_vq2_powered_calibration_probe.py"
    assert f"identity:{expected_module}" in service.calls


def test_offline_admission_cross_binds_frozen_environment_to_launcher(tmp_path):
    freeze, documents = _freeze()
    freeze["execution"]["launcher_environment_sha256"] = H
    service = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    arguments = probe.ProbeArguments(
        freeze["paths"]["live_freeze"],
        contract.canonical_file_sha256(freeze),
        COMMIT,
    )
    with pytest.raises(
        probe.OfflineAdmissionError,
        match=r"execution\.launcher_environment_sha256",
    ):
        probe.admit_offline(arguments, service)


@pytest.mark.parametrize("fault", ["dirty", "environment", "imports", "canonical"])
def test_offline_admission_fails_closed_on_identity_drift(tmp_path, fault):
    freeze, documents = _freeze()
    service = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    if fault == "dirty":
        original = service.git_worktree

        def dirty(path):
            value = original(path)
            return probe.GitWorktreeProof(**{**value.__dict__, "ignored_clean": False})

        service.git_worktree = dirty
    elif fault == "environment":
        service.security_environment = lambda: {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHOME": "defined",
            "PYTHONPATH": None,
            "PYTHONSTARTUP": None,
        }
    elif fault == "imports":
        original_import = service.rederive_import_inventory

        def bad_import(frozen, eager, *, environment_inventory):
            value = original_import(
                frozen,
                eager,
                environment_inventory=environment_inventory,
            )
            return probe.ImportRevalidation(
                value.inventory, True, False, ("unexpected.module",), ()
            )

        service.rederive_import_inventory = bad_import
    else:
        original_json = service.read_stable_json

        def noncanonical(path):
            value = original_json(path)
            if path == freeze["paths"]["live_freeze"]:
                raw = value.raw_bytes[:-1] + b" \n"
                return probe.StableJsonProof(
                    identity=probe.FileIdentityProof(
                        path=value.identity.path,
                        size_bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    ),
                    raw_bytes=raw,
                    value=value.value,
                )
            return value

        service.read_stable_json = noncanonical
    arguments = probe.ProbeArguments(
        freeze["paths"]["live_freeze"],
        contract.canonical_file_sha256(freeze),
        COMMIT,
    )
    with pytest.raises(probe.OfflineAdmissionError):
        probe.admit_offline(arguments, service)


def test_attempt_gate_rejects_poison_existing_and_any_prior_attempt():
    freeze, _ = _freeze()
    clean = probe.AttemptRootSnapshot(contract.EVIDENCE_ROOT, False, False, False)
    probe.validate_attempt_gate(freeze, clean)
    with pytest.raises(probe.AttemptGateError, match="poison"):
        probe.validate_attempt_gate(
            freeze,
            probe.AttemptRootSnapshot(contract.EVIDENCE_ROOT, True, False, False),
        )
    with pytest.raises(probe.AttemptGateError, match="already exists"):
        probe.validate_attempt_gate(
            freeze,
            probe.AttemptRootSnapshot(contract.EVIDENCE_ROOT, False, True, False),
        )
    with pytest.raises(probe.AttemptGateError, match="already consumed"):
        probe.validate_attempt_gate(
            freeze,
            probe.AttemptRootSnapshot(
                contract.EVIDENCE_ROOT,
                False,
                False,
                False,
                (probe.PriorAttemptObservation("F03-A00", 1, 1),),
            ),
        )
    with pytest.raises(probe.AttemptGateError, match="sole terminal"):
        probe.validate_attempt_gate(
            freeze,
            probe.AttemptRootSnapshot(
                contract.EVIDENCE_ROOT,
                False,
                False,
                False,
                (probe.PriorAttemptObservation("F03-A00", 2, 1),),
            ),
        )


def test_qpc_deadlines_capability_hashes_and_exact_argv(tmp_path):
    freeze, _, admission = _admission(tmp_path)
    material = _material(admission)
    checked = contract.validate_attempt(material.envelope, live_freeze=freeze)
    assert checked["context_sha256"] == material.context_sha256
    assert material.absolute_deadlines.live_contact_deadline_monotonic_ns == 300_000_000_100
    assert material.absolute_deadlines.total_deadline_monotonic_ns == 390_000_000_100
    assert material.child_argv[6:10] == (
        "--stage",
        "calibration-excite",
        "--powered-attempt-envelope",
        freeze["paths"]["attempt_envelope"],
    )
    assert material.cleanup_argv[5] == "scripts.aigp_vq2_powered_cleanup"
    payload = contract.canonical_json_file_bytes(material.envelope)
    assert b"L" * 32 not in payload
    assert b"C" * 32 not in payload
    assert b"F" * 32 not in payload
    assert repr(material.capabilities) == "CapabilitySecrets(<redacted>)"
    child_frame = material.capabilities.consume_frame("child")
    cleanup_frame = material.capabilities.consume_frame("cleanup")
    assert len(child_frame) == 36
    assert contract.decode_capability_frame(cleanup_frame) == b"F" * 32
    assert material.capabilities.is_zeroized("child")
    assert material.capabilities.is_zeroized("cleanup")
    with pytest.raises(probe.PoweredCalibrationProbeError, match="one-use"):
        material.capabilities.consume_frame("child")
    child_frame[:] = b"\x00" * len(child_frame)
    cleanup_frame[:] = b"\x00" * len(cleanup_frame)
    material.capabilities.zeroize_all()
    assert material.capabilities.is_zeroized("lease_owner")
    assert "query_performance_frequency_hz" in probe.TRANCHE2_INTEGRATION_METHODS["clock"]


def test_capabilities_require_three_independent_exact_csprng_values():
    with pytest.raises(probe.PoweredCalibrationProbeError, match="independent"):
        probe.generate_capability_secrets(lambda size: b"x" * 32)
    with pytest.raises(probe.PoweredCalibrationProbeError, match="exactly 32"):
        probe.generate_capability_secrets(lambda size: b"x" * 31)


def test_secure_attempt_directory_and_attempt_envelope_are_create_new(tmp_path):
    freeze, _, admission = _admission(tmp_path)
    secure = FakeSecure(tmp_path, freeze)
    workspace = probe.AttemptWorkspace.consume(secure, freeze)
    assert workspace.directory.current_user_only_dacl is True
    receipt = workspace.publish_attempt(_material(admission))
    assert secure.physical(receipt.path).read_bytes() == contract.canonical_json_file_bytes(
        _material(admission).envelope
    )
    with pytest.raises(probe.PublicationError, match="already attempted"):
        workspace.publish_attempt(_material(admission))
    with pytest.raises(probe.AttemptGateError):
        probe.AttemptWorkspace.consume(secure, freeze)


def test_secure_directory_requires_creation_time_dacl_owner_reparse_and_volume(tmp_path):
    freeze, _ = _freeze()
    secure = FakeSecure(tmp_path, freeze)
    secure.bad_dacl = True
    with pytest.raises(probe.SecureBoundaryError, match="invariants"):
        probe.AttemptWorkspace.consume(secure, freeze)
    assert secure.physical(freeze["paths"]["attempt_dir"]).is_dir()


@pytest.mark.skipif(os.name != "nt", reason="native protected-DACL boundary")
def test_windows_secure_create_new_applies_private_acl_at_creation(tmp_path):
    service = probe.WindowsSecureCreateNew()
    root = str(tmp_path / "private")
    root_handle = service._create_directory_native(root)
    service._close_handle(root_handle)
    try:
        root_receipt = service.open_private_directory(
            root, parent_path=str(tmp_path)
        )
        attempt_path = root + "\\" + contract.ATTEMPT_ID
        attempt_receipt = service.create_private_directory_create_new(
            attempt_path, parent_path=root
        )
        payload = contract.canonical_json_file_bytes({"proof": "native"})
        target = attempt_path + r"\proof.json"
        receipt = service.create_new_file(
            target,
            payload,
            parent=attempt_receipt,
            deadline_monotonic_ns=service._clock() + 2_000_000_000,
        )
        assert root_receipt.current_user_only_dacl is True
        assert attempt_receipt.dacl_applied_at_create is True
        assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
        assert Path(target).read_bytes() == payload
        with pytest.raises(probe.BoundaryCreateNewError) as duplicate:
            service.create_new_file(
                target,
                payload,
                parent=attempt_receipt,
                deadline_monotonic_ns=service._clock() + 2_000_000_000,
            )
        assert duplicate.value.state == "unknown"
    finally:
        service.close()


@pytest.mark.skipif(os.name != "nt", reason="native protected-DACL boundary")
def test_windows_secure_boundary_rejects_inherited_public_acl(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    service = probe.WindowsSecureCreateNew()
    try:
        with pytest.raises(probe.SecureBoundaryError):
            service.open_private_directory(
                str(public), parent_path=str(tmp_path)
            )
    finally:
        service.close()


def test_partial_publication_is_preserved_and_globally_never_retried(tmp_path):
    freeze, _ = _freeze()
    secure = FakeSecure(tmp_path, freeze)
    workspace = probe.AttemptWorkspace.consume(secure, freeze)
    target = freeze["paths"]["capture_seal"]
    secure.failures[target] = "partial"
    with pytest.raises(probe.PartialPublicationError) as caught:
        workspace.attempt_publisher.publish(
            target,
            {"forensic": "payload"},
            deadline_monotonic_ns=10_000,
        )
    partial = secure.physical(target)
    assert partial.exists() and partial.read_bytes()
    assert caught.value.retry_allowed is False
    first_call_count = len(secure.file_calls)
    with pytest.raises(probe.PublicationError, match="poisoned"):
        workspace.attempt_publisher.publish(
            freeze["paths"]["capture_seal"],
            {"forensic": "replacement"},
            deadline_monotonic_ns=10_000,
        )
    with pytest.raises(probe.PublicationError, match="poisoned"):
        workspace.attempt_publisher.publish(
            freeze["paths"]["analysis_report"],
            {"different": "target"},
            deadline_monotonic_ns=10_000,
        )
    assert len(secure.file_calls) == first_call_count
    assert partial.exists()
    recovery = workspace.recovery_publisher_for(workspace.directory)
    invalid_receipt = recovery.publish(
        freeze["paths"]["attempt_invalid"],
        {"recovery": "immutable-invalid"},
        deadline_monotonic_ns=10_000,
    )
    assert secure.physical(invalid_receipt.path).exists()
    with pytest.raises(probe.PublicationError, match="already attempted"):
        recovery.publish(
            target,
            {"forensic": "retry"},
            deadline_monotonic_ns=10_000,
        )


def test_wrapper_ledger_predecessors_artifacts_deadlines_and_lifecycle(tmp_path):
    freeze, _, admission = _admission(tmp_path)
    secure = FakeSecure(tmp_path, freeze)
    workspace = probe.AttemptWorkspace.consume(secure, freeze)
    material = _material(admission)
    attempt_receipt = workspace.publish_attempt(material)
    ledger_dir = workspace.create_subdirectory("wrapper_ledger_directory")
    ledger = probe.WrapperLedger(
        publisher=workspace.publisher_for(ledger_dir),
        lifecycle_publisher=workspace.attempt_publisher,
        ledger_directory=freeze["paths"]["wrapper_ledger_directory"],
        lifecycle_path=freeze["paths"]["wrapper_lifecycle"],
        timeline=probe.WrapperTimeline(material.absolute_deadlines),
        clock=FakeClock((200, 300, 400, 500, 600, 700)),
    )
    first = ledger.record_attempt_publish_end(
        attempt_publish_deadline=material.attempt_publish_deadline,
        observed_monotonic_ns=attempt_receipt.completed_monotonic_ns,
    )
    lease_deadline = ledger.start_phase("lease_release_and_verify")
    assert lease_deadline["parent_deadline_monotonic_ns"] == 300_000_000_100
    lease_receipt = workspace.attempt_publisher.publish(
        freeze["paths"]["lease_final"],
        {"lease": "released"},
        deadline_monotonic_ns=lease_deadline["deadline_monotonic_ns"],
    )
    ledger.end_phase(
        outcome="completed", artifacts=[probe.artifact_ref("lease_final", lease_receipt)]
    )
    assert ledger.events[1]["predecessor_sha256"] == first.sha256
    bundle_deadline = ledger.start_phase("bundle_verify")
    assert bundle_deadline["parent_deadline_monotonic_ns"] == min(
        material.absolute_deadlines.total_deadline_monotonic_ns,
        300 + contract.DEADLINE_DURATIONS_NS["postrelease_total"],
    )
    bundle_receipt = workspace.attempt_publisher.publish(
        freeze["paths"]["bundle_verification"],
        {"bundle": "verified"},
        deadline_monotonic_ns=bundle_deadline["deadline_monotonic_ns"],
    )
    ledger.end_phase(
        outcome="completed",
        artifacts=[probe.artifact_ref("bundle_verification", bundle_receipt)],
    )
    ledger.start_phase("terminal_ready")
    ledger.end_phase(outcome="completed")
    lifecycle_receipt = ledger.finalize_lifecycle()
    lifecycle = contract.parse_canonical_json_bytes(
        secure.physical(lifecycle_receipt.path).read_bytes(), file_form=True
    )
    assert contract.validate_wrapper_lifecycle(
        lifecycle, ledger_events=ledger.events
    ) == lifecycle
    assert lifecycle["records"][0]["path"].endswith(r"\event-000000.json")
    assert lifecycle["final_record_sha256"] == ledger.receipts[-1].sha256


def test_wrapper_ledger_refuses_deadline_refresh_and_bad_artifacts(tmp_path):
    freeze, _, admission = _admission(tmp_path)
    secure = FakeSecure(tmp_path, freeze)
    workspace = probe.AttemptWorkspace.consume(secure, freeze)
    material = _material(admission)
    workspace.publish_attempt(material)
    ledger_dir = workspace.create_subdirectory("wrapper_ledger_directory")
    ledger = probe.WrapperLedger(
        publisher=workspace.publisher_for(ledger_dir),
        lifecycle_publisher=workspace.attempt_publisher,
        ledger_directory=freeze["paths"]["wrapper_ledger_directory"],
        lifecycle_path=freeze["paths"]["wrapper_lifecycle"],
        timeline=probe.WrapperTimeline(material.absolute_deadlines),
        clock=FakeClock((200, 300)),
    )
    ledger.record_attempt_publish_end(
        attempt_publish_deadline=material.attempt_publish_deadline,
        observed_monotonic_ns=1_000,
    )
    ledger.start_phase("lease_acquire")
    with pytest.raises(contract.PoweredAttemptContractError, match="artifact set"):
        ledger.end_phase(
            outcome="completed",
            artifacts=[
                {
                    "name": "unexpected",
                    "path": freeze["paths"]["capture_seal"],
                    "size_bytes": 1,
                    "sha256": H,
                }
            ],
        )
    timeline = probe.WrapperTimeline(material.absolute_deadlines)
    timeline.note_lease_release(200)
    with pytest.raises(probe.PoweredCalibrationProbeError, match="refreshed"):
        timeline.note_lease_release(201)


def test_fallback_state_machine_is_single_use_and_cleanup_driven():
    eligible = probe.FallbackFacts(
        child_created=True,
        child_tree_exit="proved",
        child_cleanup="invalid",
        ports="free",
        simulator_topology="unchanged",
        cleanup_capability="available",
        fallback_already_attempted=False,
        wrapper_alive=True,
    )
    assert probe.decide_fallback(eligible).spawn is True
    assert probe.decide_fallback(
        probe.FallbackFacts(**{**eligible.__dict__, "child_cleanup": "valid"})
    ).status == "not_required"
    assert probe.decide_fallback(
        probe.FallbackFacts(**{**eligible.__dict__, "ports": "owned"})
    ).status == "not_eligible"
    assert probe.decide_fallback(
        probe.FallbackFacts(**{**eligible.__dict__, "fallback_already_attempted": True})
    ).retry_allowed is False


def test_poison_and_terminal_decisions_use_frozen_truth_table():
    cleanup = _safe_cleanup()
    artifacts = _safe_artifacts()
    assert probe.derive_poison_required(
        cleanup_state=cleanup,
        artifact_state=artifacts,
        reason_codes=["internal_error"],
        attempt_envelope_state="absent",
    ) is False
    complete = probe.decide_terminal(
        completion_ready=True,
        fallback_used=False,
        cleanup_state=_complete_cleanup(),
        artifact_state=_complete_artifacts(),
        reason_codes=[],
        attempt_envelope_state="valid",
    )
    assert complete == probe.TerminalDecision("complete", False, ())
    partial = copy.deepcopy(artifacts)
    partial["bundle_verification"] = "partial"
    failed = probe.decide_terminal(
        completion_ready=False,
        fallback_used=False,
        cleanup_state=cleanup,
        artifact_state=partial,
        reason_codes=["artifact_mismatch"],
        attempt_envelope_state="valid",
        publication_poisoned=True,
    )
    assert failed.terminal == "invalid"
    assert failed.poison_required is True
    assert "terminal_write_failed" in failed.reason_codes
    assert failed.retry_allowed is False


def test_orchestrator_is_offline_first_and_requires_live_services(tmp_path):
    freeze, documents = _freeze()
    offline = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    secure = FakeSecure(tmp_path, freeze)
    clock = FakeClock((100,))
    orchestrator = probe.ProbeOrchestrator(
        offline=offline, secure=secure, clock=clock
    )
    arguments = probe.ProbeArguments(
        freeze["paths"]["live_freeze"],
        contract.canonical_file_sha256(freeze),
        COMMIT,
    )
    with pytest.raises(probe.LiveIntegrationUnavailable):
        orchestrator.run(arguments)
    assert not secure.physical(freeze["paths"]["attempt_dir"]).exists()


def _production_cli(freeze):
    return [
        "--live-freeze",
        freeze["paths"]["live_freeze"],
        "--live-freeze-sha256",
        contract.canonical_file_sha256(freeze),
        "--expected-commit",
        COMMIT,
    ]


def _production_composition_fakes(
    freeze,
    *,
    failure=None,
    result_status="complete",
):
    events = []
    state = {}
    clock = StepClock()
    arguments = probe.ProbeArguments(
        freeze["paths"]["live_freeze"],
        contract.canonical_file_sha256(freeze),
        COMMIT,
    )
    admitted = type(
        "Admitted",
        (),
        {"live_freeze": freeze, "arguments": arguments},
    )()
    foundation = type("Foundation", (), {"offline": admitted})()

    class Offline:
        def close(self):
            events.append("offline.close")

    class Secure:
        def close(self):
            events.append("secure.close")

    class Boundary(FakeLive):
        def __init__(self, *, supplied_freeze, supplied_secure, supplied_clock):
            super().__init__()
            self.supplied_freeze = supplied_freeze
            self.supplied_secure = supplied_secure
            self.supplied_clock = supplied_clock

        def supervision_snapshot(self):
            return {"inert": True}

        def close(self):
            events.append("boundary.close")
            if failure == "close":
                raise probe.SecureBoundaryError("injected boundary close failure")

    offline = Offline()
    secure = Secure()
    postrelease = FakeLive()

    class Orchestrator:
        def __init__(self, *, offline, secure, clock):
            events.append("orchestrator.construct")
            assert offline is state["offline"]
            assert secure is state["secure"]
            assert clock is state["clock"]

        def admit(self, supplied_arguments):
            events.append("orchestrator.admit")
            assert supplied_arguments == arguments
            if failure == "admit":
                raise probe.OfflineAdmissionError("injected admission failure")
            return foundation

        def execute_admitted(self, supplied_arguments, supplied, *, live):
            events.append("orchestrator.execute_admitted")
            assert supplied_arguments == arguments
            assert supplied is foundation
            state["services"] = live
            if failure == "execute":
                raise RuntimeError("injected execution failure")
            return probe.OrchestrationResult(
                status=result_status,
                attempt_consumed=True,
                fallback_used=False,
                reason_codes=() if result_status == "complete" else ("internal_error",),
                terminal_receipt=None,
                poison_receipt=None,
                lifecycle_receipt=None,
                ledger_events=(),
                live_kernel_released=True,
                live_release_proved=True,
                no_live_after_release=True,
            )

        def run(self, supplied_arguments):
            raise AssertionError("production composition must use two-phase admission")

    def make_clock():
        events.append("clock.construct")
        state["clock"] = clock
        return clock

    def make_offline():
        events.append("offline.construct")
        state["offline"] = offline
        return offline

    def make_secure(supplied_clock):
        events.append("secure.construct")
        assert supplied_clock is clock
        state["secure"] = secure
        return secure

    def make_boundary(*, freeze, secure, clock):
        events.append("boundary.construct")
        assert freeze is admitted.live_freeze
        assert secure is state["secure"]
        assert clock is state["clock"]
        if failure == "boundary":
            raise RuntimeError("injected boundary construction failure")
        boundary = Boundary(
            supplied_freeze=freeze,
            supplied_secure=secure,
            supplied_clock=clock,
        )
        state["boundary"] = boundary
        return boundary

    def make_postrelease(*, inputs, clock, boundary):
        events.append("postrelease.construct")
        state["postrelease_inputs"] = inputs
        assert clock is state["clock"]
        assert boundary is state["boundary"]
        if failure == "postrelease":
            raise RuntimeError("injected postrelease construction failure")
        state["postrelease"] = postrelease
        return postrelease

    factories = probe._ProductionFactories(
        clock=make_clock,
        offline=make_offline,
        secure=make_secure,
        orchestrator=Orchestrator,
        boundary=make_boundary,
        postrelease=make_postrelease,
    )
    return factories, events, state


def test_production_main_is_two_phase_exactly_wired_and_deterministically_closed(
    capsys,
):
    freeze, _documents = _freeze()
    factories, events, state = _production_composition_fakes(freeze)
    assert probe.main(
        _production_cli(freeze),
        _production_factories=factories,
    ) == 0
    assert capsys.readouterr().err == ""
    assert events == [
        "clock.construct",
        "offline.construct",
        "secure.construct",
        "orchestrator.construct",
        "orchestrator.admit",
        "boundary.construct",
        "postrelease.construct",
        "orchestrator.execute_admitted",
        "boundary.close",
        "secure.close",
        "offline.close",
    ]
    boundary = state["boundary"]
    services = state["services"]
    assert all(
        getattr(services, name) is boundary
        for name in (
            "host",
            "csprng",
            "launcher",
            "topology",
            "training",
            "process",
            "ports",
            "lease",
            "spawn",
            "supervision",
        )
    )
    assert services.postrelease is state["postrelease"]
    inputs = state["postrelease_inputs"]
    assert inputs.paths == freeze["paths"]
    assert inputs.live_freeze_path == freeze["paths"]["live_freeze"]
    assert inputs.implementation_inventory_path == freeze["candidate"][
        "implementation_inventory"
    ]["path"]
    assert inputs.environment_inventory_path == freeze["runtime"][
        "environment_inventory"
    ]["path"]
    assert inputs.import_inventory_path == freeze["runtime"]["import_inventory"][
        "path"
    ]
    assert inputs.supervision_snapshot.__self__ is boundary


@pytest.mark.parametrize(
    ("failure", "expected_tail", "boundary_constructed"),
    [
        ("admit", ["secure.close", "offline.close"], False),
        ("boundary", ["secure.close", "offline.close"], False),
        ("postrelease", ["boundary.close", "secure.close", "offline.close"], True),
        ("execute", ["boundary.close", "secure.close", "offline.close"], True),
        ("close", ["boundary.close", "secure.close", "offline.close"], True),
    ],
)
def test_production_main_failure_paths_close_in_exact_order(
    failure,
    expected_tail,
    boundary_constructed,
    capsys,
):
    freeze, _documents = _freeze()
    factories, events, state = _production_composition_fakes(
        freeze,
        failure=failure,
    )
    assert probe.main(
        _production_cli(freeze),
        _production_factories=factories,
    ) == 2
    assert events[-len(expected_tail) :] == expected_tail
    assert ("boundary" in state) is boundary_constructed
    assert "powered calibration probe refused:" in capsys.readouterr().err


def test_production_main_noncomplete_result_closes_and_returns_refusal():
    freeze, _documents = _freeze()
    factories, events, _state = _production_composition_fakes(
        freeze,
        result_status="invalid",
    )
    assert probe.main(
        _production_cli(freeze),
        _production_factories=factories,
    ) == 2
    assert events[-3:] == ["boundary.close", "secure.close", "offline.close"]


def test_malformed_production_cli_constructs_no_provider():
    freeze, _documents = _freeze()
    factories, events, _state = _production_composition_fakes(freeze)
    with pytest.raises(SystemExit):
        probe.main(["--live-freeze"], _production_factories=factories)
    assert events == []


def test_orchestrator_carries_exact_qpc_frequency_into_foundation(tmp_path):
    freeze, documents = _freeze()
    offline = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    secure = FakeSecure(tmp_path, freeze)
    admitted = probe.ProbeOrchestrator(
        offline=offline,
        secure=secure,
        clock=FakeClock((12345,)),
    ).admit(
        probe.ProbeArguments(
            freeze["paths"]["live_freeze"],
            contract.canonical_file_sha256(freeze),
            COMMIT,
        )
    )
    assert admitted.wrapper_started_monotonic_ns == 12345
    assert admitted.qpc_frequency_hz == 10_000_000


def _phase_events(result, phase: str):
    return [event for event in result.ledger_events if event["phase"] == phase]


def _interrupt_ledger_write(secure: FakeSecure, freeze, sequence: int) -> None:
    target = freeze["paths"]["wrapper_ledger_directory"] + (
        f"\\event-{sequence:06d}.json"
    )
    create_new_file = secure.create_new_file

    def interrupted(path, payload, *, parent, deadline_monotonic_ns):
        if path == target:
            secure.file_calls.append(path)
            if secure.events is not None:
                secure.events.append(f"secure.write:{ntpath.basename(path)}")
            raise KeyboardInterrupt()
        return create_new_file(
            path,
            payload,
            parent=parent,
            deadline_monotonic_ns=deadline_monotonic_ns,
        )

    secure.create_new_file = interrupted


def test_single_attempt_wrapper_completes_exact_sequence_and_goes_offline(tmp_path):
    freeze, offline, secure, live, validators, result = _run_orchestration(tmp_path)

    assert result.status == "complete"
    assert result.attempt_consumed is True
    assert result.fallback_used is False
    assert result.reason_codes == ()
    assert result.live_release_proved is True
    assert result.no_live_after_release is True
    assert result.terminal_receipt.path == freeze["paths"]["attempt_complete"]
    assert result.poison_receipt is None

    expected_phases = [
        "attempt_publish",
        "lease_acquire",
        "launcher_return",
        "topology_and_training_attestation",
        "prechild_identity_and_ports",
        "child_spawn",
        "child_supervision",
        "child_exit_proof",
        "postcheck_identity_process_ports",
        "lease_release_and_verify",
        "bundle_verify",
        "capture_seal",
        "analysis",
        "split_publish",
        "terminal_ready",
    ]
    assert [event["phase"] for event in result.ledger_events] == [
        "attempt_publish",
        *[phase for phase in expected_phases[1:] for _ in range(2)],
    ]
    assert result.ledger_events[0]["event"] == "phase_end"
    for phase in expected_phases[1:]:
        start, end = _phase_events(result, phase)
        assert (start["event"], end["event"]) == ("phase_start", "phase_end")
        assert end["outcome"] == "completed"
        assert end["reason_code"] is None
        expected_deadline = probe.derive_phase_deadline(
            phase,
            started_monotonic_ns=start["observed_monotonic_ns"],
            parent_deadline_monotonic_ns=start[
                "parent_deadline_monotonic_ns"
            ],
        )
        for name in (
            "duration_ns",
            "parent_deadline_monotonic_ns",
            "deadline_monotonic_ns",
        ):
            assert start[name] == expected_deadline[name]
            assert end[name] == start[name]

    for phase in ("bundle_verify", "capture_seal", "analysis", "split_publish"):
        start, _end = _phase_events(result, phase)
        assert live.postrelease_phase_deadlines[phase] == {
            "phase": phase,
            "started_monotonic_ns": start["observed_monotonic_ns"],
            "duration_ns": start["duration_ns"],
            "parent_deadline_monotonic_ns": start[
                "parent_deadline_monotonic_ns"
            ],
            "deadline_monotonic_ns": start["deadline_monotonic_ns"],
        }

    terminal_start, _terminal_end = _phase_events(result, "terminal_ready")
    assert live.complete_context["phase"] == "terminal_ready"
    assert live.complete_context["publication_timing"] == {
        "phase": "terminal_publish",
        "started_monotonic_ns": terminal_start["observed_monotonic_ns"],
        "duration_ns": terminal_start["duration_ns"],
        "parent_deadline_monotonic_ns": terminal_start[
            "parent_deadline_monotonic_ns"
        ],
        "deadline_monotonic_ns": terminal_start["deadline_monotonic_ns"],
        "prepared_monotonic_ns": live.complete_context[
            "completed_monotonic_ns"
        ],
    }

    artifacts_by_phase = {
        event["phase"]: [item["name"] for item in event["artifacts"]]
        for event in result.ledger_events
        if event["event"] == "phase_end"
    }
    assert artifacts_by_phase["lease_release_and_verify"] == ["lease_final"]
    assert artifacts_by_phase["bundle_verify"] == ["bundle_verification"]
    assert artifacts_by_phase["capture_seal"] == ["capture_seal"]
    assert artifacts_by_phase["split_publish"] == [
        "analysis_report",
        "split_claim",
        "split_registry",
    ]

    assert secure.directory_calls.count(freeze["paths"]["attempt_dir"]) == 1
    assert secure.file_calls.count(freeze["paths"]["attempt_envelope"]) == 1
    assert live.events.count("spawn.seal_environment") == 1
    assert live.events.index("spawn.seal_environment") + 1 == live.events.index(
        f"secure.mkdir:{contract.ATTEMPT_ID}"
    )
    assert offline.calls.count("git") == 2
    assert live.acquired_qpc_frequency_hz == 10_000_000
    assert live.owner_secret_reference is not None
    assert not any(live.owner_secret_reference)
    assert live.capability_frame_references
    assert all(not any(frame) for frame in live.capability_frame_references)
    assert any(event.startswith("lease.heartbeat:") for event in live.events)
    assert live.events.index("secure.write:child-authority.json") < live.events.index(
        "spawn.release_child"
    )
    offline_indices = [
        index for index, event in enumerate(live.events) if event == "offline.admit"
    ]
    assert len(offline_indices) == 2
    assert live.events.index("secure.write:child-authority.json") < offline_indices[1]
    assert offline_indices[1] < live.events.index("spawn.release_child")
    assert live.events.index("secure.write:process-final-proof.json") < live.events.index(
        "lease.release"
    )
    release_index = live.events.index("lease.release")
    assert release_index < live.events.index("post.verify_bundle")
    assert live.release_callback_attempts == 1
    assert not any(
        event.startswith("lease.heartbeat:") for event in live.events[release_index + 1 :]
    )
    assert [
        event
        for event in live.events
        if event
        in {
            "secure.write:split-claim.json",
            "secure.write:registry-000001.json",
            "secure.write:analysis.json",
        }
    ] == [
        "secure.write:split-claim.json",
        "secure.write:registry-000001.json",
        "secure.write:analysis.json",
    ]
    assert "complete_terminal" in validators.calls
    assert live.events.count("spawn.close_process_handle:child-handle") == 1
    assert live.events.count("spawn.close_attempt_handles") == 1
    assert live.events.count("process.close_retained_wrapper") == 1
    close_index = live.events.index("spawn.close_attempt_handles")
    assert close_index < live.events.index("post.verify_bundle")


def test_lost_child_spawn_phase_end_still_drains_before_recorded_release(tmp_path):
    def configure(live, secure, freeze):
        # event 10 is child_spawn phase_end.  BaseException bypasses the fake
        # publisher's poison latch while still making the wrapper ledger
        # unusable, so the recovery lane can demonstrate recorded release.
        _interrupt_ledger_write(secure, freeze, 10)

    freeze, _, secure, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert result.live_release_proved is True
    assert live.events.count("spawn.release_child") == 1
    assert live.events.count("supervision.child") == 1
    assert live.events.count("process.child_tree_exit") == 1
    assert live.events.count("spawn.close_process_handle:child-handle") == 1
    assert live.events.count("lease.release") == 1
    assert secure.file_calls.count(freeze["paths"]["lease_final"]) == 1
    assert live.events.index("spawn.release_child") < live.events.index(
        "supervision.child"
    )
    assert live.events.index("supervision.child") < live.events.index(
        "process.child_tree_exit"
    )
    assert live.events.index("process.child_tree_exit") < live.events.index(
        "process.final"
    )
    assert live.events.index("process.final") < live.events.index("lease.release")
    assert not _phase_events(result, "child_supervision")
    assert not _phase_events(result, "child_exit_proof")
    assert not _phase_events(result, "postcheck_identity_process_ports")


def test_unproved_child_tree_blocks_lease_release(tmp_path):
    def configure(live, secure, freeze):
        live.failures["process.child_tree_exit"] = probe.OrchestrationPhaseError(
            "process_residue", "injected missing whole-tree proof"
        )

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "process_residue" in result.reason_codes
    assert live.events.count("process.child_tree_exit") == 1
    assert live.events.count("lease.release") == 0
    assert "process.final" not in live.events
    assert result.live_kernel_released is False
    assert result.live_release_proved is False


def test_failed_final_postcheck_is_retried_unledgered_and_blocks_release(tmp_path):
    def configure(live, secure, freeze):
        live.failures["ports.final_free"] = probe.OrchestrationPhaseError(
            "port_residue", "injected final-port ambiguity"
        )

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "port_residue" in result.reason_codes
    assert live.events.count("process.final") == 2
    assert live.events.count("ports.final_free") == 2
    assert live.events.count("lease.release") == 0
    assert result.live_release_proved is False


def test_ledger_loss_after_launcher_before_child_needs_full_final_gate(tmp_path):
    def configure(live, secure, freeze):
        # event 6 is topology_and_training_attestation phase_end: launcher
        # authority has already been invoked, but no child exists yet.
        _interrupt_ledger_write(secure, freeze, 6)
        live.final_topology_proved = False

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "spawn.child_blocked" not in live.events
    assert live.events.count("process.final") == 1
    assert live.events.count("ports.final_free") == 1
    assert live.events.count("topology.unchanged") == 1
    assert live.events.count("lease.release") == 0
    assert result.live_release_proved is False


def test_late_successful_acquire_uses_no_launcher_release_recovery(tmp_path):
    def configure(live, secure, freeze):
        live.acquire_silent_advance = (
            contract.DEADLINE_DURATIONS_NS["lease_acquire"] + 1
        )

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "deadline_expired" in result.reason_codes
    assert "launcher.wait" not in live.events
    assert "process.final" not in live.events
    assert live.events.count("lease.release") == 1
    assert result.live_release_proved is True
    assert live.invalid_context["cleanup_state"] == {
        "child_exit": "not_created",
        "fallback": "not_eligible",
        "ports": "not_opened",
        "lease": "released",
        "processes": "not_created",
        "transport": "not_opened",
        "scheduled_task": "not_created",
        "simulator_topology": "not_launched",
        "simulator_responsive": "not_launched",
    }


def test_lost_acquire_phase_end_still_releases_known_owned_lease(tmp_path):
    def configure(live, secure, freeze):
        # event 2 is lease_acquire phase_end, after acquire returned ownership.
        _interrupt_ledger_write(secure, freeze, 2)

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "launcher.wait" not in live.events
    assert "process.final" not in live.events
    assert live.events.count("lease.release") == 1
    assert result.live_kernel_released is True
    assert result.live_release_proved is True


def test_timeout_is_terminal_invalid_and_poisoned_without_fallback(tmp_path):
    def configure(live, secure, freeze):
        live.failures["supervision.child"] = probe.OrchestrationPhaseError(
            "deadline_expired", "injected child supervision deadline"
        )
        live.fallback_gate_proved = False

    freeze, _, secure, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "deadline_expired" in result.reason_codes
    assert result.fallback_used is False
    assert result.poison_receipt.path == freeze["paths"]["live_poison"]
    assert result.terminal_receipt.path == freeze["paths"]["attempt_invalid"]
    assert "spawn.fallback_blocked" not in live.events
    failed = _phase_events(result, "child_supervision")[-1]
    assert failed["outcome"] == "failed"
    assert failed["reason_code"] == "deadline_expired"
    assert secure.file_calls.count(freeze["paths"]["live_poison"]) == 1


def test_wrapper_death_forbids_fallback_and_requires_poison(tmp_path):
    def configure(live, secure, freeze):
        live.child_outcome = probe.ChildSupervisionOutcome(
            cleanup_proved=True,
            collection_valid=True,
            wrapper_death=True,
            artifact_state_patch=_sealed_child_artifacts(),
        )

    freeze, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "wrapper_death" in result.reason_codes
    assert result.fallback_used is False
    assert result.poison_receipt.path == freeze["paths"]["live_poison"]
    assert "spawn.fallback_blocked" not in live.events
    assert live.invalid_context["wrapper_alive"] is False


def test_cleanup_fallback_runs_once_after_proved_gate_and_invalidates_attempt(tmp_path):
    def configure(live, secure, freeze):
        live.child_outcome = probe.ChildSupervisionOutcome(
            cleanup_proved=False,
            collection_valid=True,
            artifact_state_patch=_sealed_child_artifacts(),
        )

    freeze, offline, secure, live, _, result = _run_orchestration(
        tmp_path, configure
    )
    assert result.status == "invalid"
    assert result.fallback_used is True
    assert "cleanup_unconfirmed" in result.reason_codes
    assert result.poison_receipt is None
    assert secure.file_calls.count(freeze["paths"]["cleanup_authority"]) == 1
    assert live.events.count("spawn.fallback_blocked") == 1
    assert live.events.count("spawn.release_fallback") == 1
    assert live.events.count("supervision.fallback") == 1
    assert offline.calls.count("git") == 3
    assert live.events.index("secure.write:cleanup-authority.json") < live.events.index(
        "spawn.release_fallback"
    )
    assert live.invalid_context["cleanup_state"]["fallback"] == "proved"


def test_failed_cleanup_fallback_poison_latches_attempt(tmp_path):
    def configure(live, secure, freeze):
        live.child_outcome = probe.ChildSupervisionOutcome(
            cleanup_proved=False,
            collection_valid=True,
            artifact_state_patch=_sealed_child_artifacts(),
        )
        live.fallback_outcome = probe.FallbackSupervisionOutcome(
            cleanup_proved=False,
            reason_codes=("cleanup_unconfirmed",),
        )

    freeze, _, secure, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert result.fallback_used is True
    assert result.poison_receipt.path == freeze["paths"]["live_poison"]
    assert live.invalid_context["cleanup_state"]["fallback"] == "failed"
    assert secure.file_calls.count(freeze["paths"]["live_poison"]) == 1
    failed = _phase_events(result, "fallback_supervision")[-1]
    assert (failed["outcome"], failed["reason_code"]) == (
        "failed",
        "cleanup_unconfirmed",
    )


@pytest.mark.parametrize("target_key", ["split_registry", "attempt_complete"])
def test_partial_publication_is_preserved_poisoned_and_never_retried(
    tmp_path, target_key
):
    def configure(live, secure, freeze):
        secure.failures[freeze["paths"][target_key]] = "partial"

    freeze, _, secure, live, _, result = _run_orchestration(tmp_path, configure)
    target = freeze["paths"][target_key]
    assert result.status == "invalid"
    assert result.poison_receipt.path == freeze["paths"]["live_poison"]
    assert result.terminal_receipt.path == freeze["paths"]["attempt_invalid"]
    assert secure.file_calls.count(target) == 1
    partial_bytes = secure.physical(target).read_bytes()
    assert partial_bytes
    state = live.invalid_context["artifact_state"]
    if target_key == "attempt_complete":
        assert state["attempt_complete"] == "partial"
        assert state["attempt_complete_partial_sha256"] == hashlib.sha256(
            partial_bytes
        ).hexdigest()
    else:
        assert state["split_claim"] == "valid"
        assert state["split_registry"] == "partial"
        assert state["split_registry_sha256"] is None
    assert "terminal_write_failed" in result.reason_codes or "artifact_mismatch" in result.reason_codes
    assert live.events.count("spawn.close_attempt_handles") == 1
    assert live.events.count("process.close_retained_wrapper") == 1
    assert live.events.index("spawn.close_attempt_handles") < live.events.index(
        f"secure.write:{ntpath.basename(target)}"
    )


def test_postrelease_failure_never_reenters_live_services(tmp_path):
    def configure(live, secure, freeze):
        live.failures["post.verify_bundle"] = probe.OrchestrationPhaseError(
            "artifact_mismatch", "injected post-release verification failure"
        )

    freeze, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert result.live_release_proved is True
    assert result.no_live_after_release is True
    release_index = live.events.index("lease.release")
    tail = live.events[release_index + 1 :]
    forbidden_prefixes = (
        "launcher.",
        "topology.",
        "training.",
        "process.",
        "ports.",
        "spawn.",
        "supervision.",
        "lease.acquire",
        "lease.heartbeat",
        "lease.release",
    )
    assert not any(
        event.startswith(forbidden_prefixes) and ".close" not in event
        for event in tail
    )
    assert "post.verify_bundle" in tail
    assert result.terminal_receipt.path == freeze["paths"]["attempt_invalid"]


def test_long_launcher_wait_uses_periodic_fixed_deadline_heartbeats(tmp_path):
    def configure(live, secure, freeze):
        live.launcher_heartbeat_advances = (
            900_000_000,
            900_000_000,
            900_000_000,
        )

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "complete"
    launcher_start = _phase_events(result, "launcher_return")[0]
    launcher_heartbeats = [
        item for item in live.heartbeat_records if item[0] == "launcher_return"
    ]
    assert len(launcher_heartbeats) >= 5
    assert live.launcher_heartbeat_period_ns == 1_000_000_000
    assert launcher_heartbeats[-1][2] - launcher_heartbeats[0][2] > 1_500_000_000
    assert {item[1] for item in launcher_heartbeats} == {
        launcher_start["deadline_monotonic_ns"]
    }
    assert launcher_start["deadline_monotonic_ns"] == min(
        launcher_start["observed_monotonic_ns"]
        + contract.DEADLINE_DURATIONS_NS["launcher_return"],
        launcher_start["parent_deadline_monotonic_ns"],
    )


def test_pre_release_heartbeat_gap_invalidates_but_still_releases_once(tmp_path):
    def configure(live, secure, freeze):
        secure.advance_after_phase_end = (
            "postcheck_identity_process_ports",
            1_600_000_000,
        )

    freeze, _, secure, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "lease_unverifiable" in result.reason_codes
    assert result.live_kernel_released is True
    assert result.live_release_proved is True
    assert result.no_live_after_release is True
    assert live.events.count("lease.release") == 1
    assert secure.file_calls.count(freeze["paths"]["lease_final"]) == 1
    release_end = _phase_events(result, "lease_release_and_verify")[-1]
    assert (release_end["outcome"], release_end["reason_code"]) == (
        "failed",
        "lease_unverifiable",
    )
    assert release_end["artifacts"] == []
    assert live.invalid_context["cleanup_state"]["lease"] == "released"


def test_uncertain_release_is_single_use_and_latches_no_more_live_contact(tmp_path):
    def configure(live, secure, freeze):
        live.failures["lease.release"] = RuntimeError("injected release uncertainty")

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "lease_release_unconfirmed" in result.reason_codes
    assert result.live_kernel_released is False
    assert result.live_release_proved is False
    assert result.no_live_after_release is True
    assert live.events.count("lease.release") == 1
    release_index = live.events.index("lease.release")
    assert not any(
        event.startswith(
            (
                "launcher.",
                "topology.",
                "training.",
                "process.final",
                "ports.",
                "supervision.",
                "spawn.child",
                "spawn.fallback",
                "spawn.release",
                "lease.heartbeat",
            )
        )
        for event in live.events[release_index + 1 :]
    )


def test_prepublication_failure_zeroizes_all_secrets_and_closes_handles(
    tmp_path, monkeypatch
):
    captured: list[probe.CapabilitySecrets] = []
    original = probe.generate_capability_secrets

    def capture(random_bytes):
        secrets = original(random_bytes)
        captured.append(secrets)
        return secrets

    monkeypatch.setattr(probe, "generate_capability_secrets", capture)

    def configure(live, secure, freeze):
        secure.failures[freeze["paths"]["attempt_envelope"]] = "partial"

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert len(captured) == 1
    assert all(
        captured[0].is_zeroized(role)
        for role in ("lease_owner", "child", "cleanup")
    )
    assert live.owner_secret_reference is None
    assert live.events.count("spawn.close_attempt_handles") == 1
    assert live.events.count("process.close_retained_wrapper") == 1


def test_final_environment_seal_failure_is_unconsumed_and_leaves_no_poison(
    tmp_path, monkeypatch
):
    captured: list[probe.CapabilitySecrets] = []
    original = probe.generate_capability_secrets

    def capture(random_bytes):
        secrets = original(random_bytes)
        captured.append(secrets)
        return secrets

    monkeypatch.setattr(probe, "generate_capability_secrets", capture)
    freeze, documents = _freeze()
    offline = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    secure = FakeSecure(tmp_path, freeze)
    live = FakeLive()
    clock = StepClock()
    live.clock = clock
    secure.clock = clock
    offline.events = live.events
    secure.events = live.events
    live.failures["spawn.seal_environment"] = probe.OrchestrationPhaseError(
        "build_or_candidate_changed",
        f"native environment drifted expected_sha256={H} observed_sha256={H2}",
    )
    orchestrator = probe.ProbeOrchestrator(
        offline=offline,
        secure=secure,
        clock=clock,
        live=_live_services(live),
        validators=RecordingValidators(),
    )

    with pytest.raises(
        probe.OrchestrationPhaseError,
        match="expected_sha256=[a-f0-9]{64} observed_sha256=[a-f0-9]{64}",
    ):
        orchestrator.run(
            probe.ProbeArguments(
                freeze["paths"]["live_freeze"],
                contract.canonical_file_sha256(freeze),
                COMMIT,
            )
        )

    assert len(captured) == 1
    assert all(
        captured[0].is_zeroized(role)
        for role in ("lease_owner", "child", "cleanup")
    )
    assert secure.directory_calls == []
    assert secure.file_calls == []
    assert not secure.physical(freeze["paths"]["attempt_dir"]).exists()
    assert not secure.physical(freeze["paths"]["live_poison"]).exists()
    assert live.events.count("spawn.seal_environment") == 1
    assert live.events.count("spawn.close_attempt_handles") == 1
    assert live.events.count("process.close_retained_wrapper") == 1
    assert not any(
        event.startswith(("lease.", "launcher.", "ports.", "spawn.child", "spawn.fallback"))
        for event in live.events
    )


def test_import_audit_environment_mutation_fails_before_attempt_or_live_contact(
    tmp_path,
):
    freeze, documents = _freeze()
    offline = FakeOffline(tmp_path, freeze=freeze, documents=documents)

    def reject_mutation(_frozen, _eager, *, environment_inventory):
        assert environment_inventory["variables"] == _environment_inventory()[
            "variables"
        ]
        raise probe.OfflineAdmissionError(
            f"parent native environment changed; expected_sha256={H}; "
            f"observed_sha256={H2}"
        )

    offline.rederive_import_inventory = reject_mutation
    secure = FakeSecure(tmp_path, freeze)
    live = FakeLive()
    offline.events = live.events
    secure.events = live.events
    orchestrator = probe.ProbeOrchestrator(
        offline=offline,
        secure=secure,
        clock=StepClock(),
        live=_live_services(live),
        validators=RecordingValidators(),
    )

    with pytest.raises(
        probe.OfflineAdmissionError,
        match="parent native environment changed",
    ):
        orchestrator.run(
            probe.ProbeArguments(
                freeze["paths"]["live_freeze"],
                contract.canonical_file_sha256(freeze),
                COMMIT,
            )
        )

    assert secure.inspect_calls == 0
    assert secure.directory_calls == []
    assert secure.file_calls == []
    assert not secure.physical(freeze["paths"]["attempt_dir"]).exists()
    assert not secure.physical(freeze["paths"]["live_poison"]).exists()
    assert live.events == ["offline.admit"]


def test_spawn_failure_closes_prepared_owners_before_invalid_terminal(tmp_path):
    def configure(live, secure, freeze):
        live.failures["spawn.child_blocked"] = probe.OrchestrationPhaseError(
            "child_spawn_failed", "injected blocked-spawn failure"
        )

    _, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "child_spawn_failed" in result.reason_codes
    assert live.events.count("spawn.close_attempt_handles") == 1
    assert live.events.count("process.close_retained_wrapper") == 1
    assert not any(
        event.startswith("spawn.close_process_handle") for event in live.events
    )
    assert live.events.index("spawn.close_attempt_handles") < live.events.index(
        "post.invalid_terminal"
    )


def test_semantic_ambiguity_blocks_before_attempt_root_or_live_contact(
    tmp_path, monkeypatch
):
    from scripts import aigp_vq2_powered_calibration_analysis as analysis

    monkeypatch.setattr(
        analysis,
        "semantic_ambiguities",
        lambda: ("injected unresolved cooked-PAK semantic",),
    )
    freeze, documents = _freeze()
    offline = FakeOffline(tmp_path, freeze=freeze, documents=documents)
    secure = FakeSecure(tmp_path, freeze)
    live = FakeLive()
    orchestrator = probe.ProbeOrchestrator(
        offline=offline,
        secure=secure,
        clock=StepClock(),
        live=_live_services(live),
        validators=RecordingValidators(),
    )
    with pytest.raises(probe.OfflineAdmissionError, match="semantic ambiguities"):
        orchestrator.run(
            probe.ProbeArguments(
                freeze["paths"]["live_freeze"],
                contract.canonical_file_sha256(freeze),
                COMMIT,
            )
        )
    assert secure.inspect_calls == 0
    assert secure.directory_calls == []
    assert secure.file_calls == []
    assert not secure.physical(freeze["paths"]["attempt_dir"]).exists()
    assert live.events == []


@pytest.mark.skipif(os.name != "nt", reason="production admission is Win32-only")
class TestWindowsProductionOfflineAdmission:
    @staticmethod
    def _begin(service, seconds=10.0, heartbeat=None):
        service.begin_bounded_admission(
            deadline_monotonic_ns=time.perf_counter_ns() + int(seconds * 1e9),
            monotonic_ns=time.perf_counter_ns,
            heartbeat=heartbeat,
        )

    @staticmethod
    def _native_environment_value(kernel, name):
        size = int(kernel.GetEnvironmentVariableW(name, None, 0))
        if size == 0:
            return None
        import ctypes

        buffer = ctypes.create_unicode_buffer(size)
        observed = int(kernel.GetEnvironmentVariableW(name, buffer, size))
        assert observed == size - 1
        return buffer.value

    def test_isolated_initial_import_audit_is_complete_and_deterministic(self):
        assert tuple(probe._POWERED_RUNTIME_IMPORT_PROVIDERS) == (
            contract.RUNTIME_IMPORT_MODULES
        )
        command = [
            sys.executable,
            "-E",
            "-s",
            "-B",
            "-m",
            probe.IMPORT_AUDIT_MODULE,
        ]
        observed = [
            subprocess.run(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            for _index in range(2)
        ]
        for result in observed:
            assert result.returncode == 0, result.stderr.decode(errors="replace")
            assert result.stderr == b""
        assert observed[0].stdout == observed[1].stdout
        inventory = contract.parse_and_validate_powered_record(observed[0].stdout)
        assert inventory["seeds"] == list(probe.POWERED_IMPORT_SEED_MODULES)
        entries = {entry["module"]: entry for entry in inventory["entries"]}
        assert len(entries) == len(inventory["entries"])
        assert probe.IMPORT_AUDIT_MODULE not in entries
        assert entries["__main__"]["root_class"] == "candidate"
        assert entries["__main__"]["origin"] == str(
            (Path.cwd() / "scripts" / "aigp_vq2_powered_calibration_probe.py").resolve()
        )
        assert entries[probe.PROBE_MODULE]["origin"] == entries["__main__"]["origin"]
        for name in (
            "typing.io",
            "typing.re",
            "cv2.utils.fs",
            "cv2.utils.logging",
            "cv2.utils.nested",
        ):
            assert entries[name]["root_class"] == "runtime"
            assert entries[name]["sha256"] is not None
            assert entries[name]["origin"] is not None
            assert entries[name]["size_bytes"] > 0
        assert entries["typing.io"]["origin"].endswith(r"\Lib\typing.py")
        assert entries["typing.re"]["origin"] == entries["typing.io"]["origin"]
        assert entries["cv2.utils.fs"]["origin"].endswith(r"\cv2\cv2.pyd")
        assert entries["cv2.utils.logging"]["origin"] == entries["cv2.utils.fs"][
            "origin"
        ]
        assert entries["cv2.utils.nested"]["origin"] == entries["cv2.utils.fs"][
            "origin"
        ]
        assert {
            "builtin",
            "candidate",
            "frozen",
            "namespace",
            "runtime",
            "stdlib",
            "venv",
        }.issubset({entry["root_class"] for entry in inventory["entries"]})

    def test_initial_import_snapshot_rejects_same_name_object_replacement(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        service = probe.WindowsProductionOfflineAdmission()
        self._begin(service)
        replacement_name = "aigp_snapshot_replacement_probe"
        initial_value = SimpleNamespace()
        monkeypatch.setitem(sys.modules, replacement_name, initial_value)
        monkeypatch.setitem(
            sys.modules,
            "__main__",
            SimpleNamespace(__spec__=SimpleNamespace(name=probe.IMPORT_AUDIT_MODULE)),
        )
        for name in probe.POWERED_IMPORT_SEED_MODULES:
            current = sys.modules.get(name)
            if current is None or getattr(getattr(current, "__spec__", None), "name", None) != name:
                monkeypatch.setitem(
                    sys.modules,
                    name,
                    SimpleNamespace(__spec__=SimpleNamespace(name=name)),
                )
        monkeypatch.setattr(
            service,
            "_bounded_import",
            lambda _importlib, name: SimpleNamespace(name=name),
        )
        monkeypatch.setattr(
            service,
            "current_working_directory",
            lambda: probe.PathProof(
                LIVE_WORKTREE,
                LIVE_WORKTREE,
                "directory",
                "test-volume",
            ),
        )
        monkeypatch.setattr(
            service,
            "_runtime_roots",
            lambda _candidate, _sysconfig: (r"C:\venv", r"C:\stdlib"),
        )
        replaced = False

        def entry(module_name, _module, **_roots):
            nonlocal replaced
            if not replaced:
                sys.modules[replacement_name] = SimpleNamespace()
                replaced = True
            return {
                "module": module_name,
                "origin": None,
                "size_bytes": None,
                "sha256": None,
                "root_class": "builtin",
                "namespace_roots": [],
            }

        monkeypatch.setattr(service, "_initial_import_entry", entry)
        try:
            with pytest.raises(probe.OfflineAdmissionError, match="sys.modules changed"):
                service.derive_initial_import_inventory(
                    probe.POWERED_IMPORT_SEED_MODULES,
                    probe.POWERED_EAGER_IMPORT_MODULES,
                    audit_module=probe.IMPORT_AUDIT_MODULE,
                )
        finally:
            service.end_bounded_admission(succeeded=False)

    def test_isolated_import_audit_revalidates_in_the_live_main_shape(self):
        inventory_process = subprocess.run(
            [
                sys.executable,
                "-E",
                "-s",
                "-B",
                "-m",
                probe.IMPORT_AUDIT_MODULE,
            ],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert inventory_process.returncode == 0, inventory_process.stderr.decode(
            errors="replace"
        )
        child_code = """
import os
import runpy
import sys
from scripts import aigp_vq2_powered_attempt as contract
from scripts import aigp_vq2_powered_calibration_probe as probe
from scripts import aigp_vq2_powered_runtime as runtime
inventory = contract.parse_canonical_json_bytes(sys.stdin.buffer.read(), file_form=True)
sys.path[0] = os.getcwd()
sys.modules['__main__'] = sys.modules[probe.PROBE_MODULE]
clock = runtime.WindowsQpcProvider()
service = probe.WindowsProductionOfflineAdmission()
native_environment = {name.upper(): value for name, value in os.environ.items()}
service._read_native_environment_block = lambda: dict(native_environment)
service.begin_bounded_admission(
    deadline_monotonic_ns=clock.now_ns() + 10_000_000_000,
    monotonic_ns=clock.now_ns,
    heartbeat=None,
)
succeeded = False
try:
    environment = service.rederive_environment_inventory({
        'schema': 'aigp-vq2-powered-environment-inventory/1',
        'created_at_utc': '2026-07-21T00:00:00.000000Z',
        'variables': [],
    })
    missing_before = [
        entry['module'] for entry in inventory['entries']
        if sys.modules.get(entry['module']) is None
    ]
    audit = service.rederive_import_inventory(
        inventory,
        probe.POWERED_EAGER_IMPORT_MODULES,
        environment_inventory=environment,
    )
    succeeded = True
finally:
    try:
        service.end_bounded_admission(succeeded=succeeded)
    finally:
        service.close()
result = {
    'semantic_equal': {
        key: audit.inventory[key] for key in ('python_sha256', 'seeds', 'entries')
    } == {
        key: inventory[key] for key in ('python_sha256', 'seeds', 'entries')
    },
    'origins_reverified': audit.origins_reverified,
    'user_site_on_sys_path': audit.user_site_on_sys_path,
    'unexpected': list(audit.unexpected_candidate_or_venv_modules),
    'unclassified': list(audit.unclassified_origins),
    'parent_missing_preserved': [
        entry['module'] for entry in inventory['entries']
        if sys.modules.get(entry['module']) is None
    ] == missing_before,
}
sys.stdout.buffer.write(contract.canonical_json_file_bytes(result))
"""
        revalidation = subprocess.run(
            [sys.executable, "-E", "-s", "-B", "-c", child_code],
            cwd=Path.cwd(),
            input=inventory_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert revalidation.returncode == 0, revalidation.stderr.decode(
            errors="replace"
        )
        assert contract.parse_canonical_json_bytes(
            revalidation.stdout, file_form=True
        ) == {
            "semantic_equal": True,
            "origins_reverified": True,
            "user_site_on_sys_path": False,
            "unexpected": [],
            "unclassified": [],
            "parent_missing_preserved": True,
        }

    def test_production_revalidation_isolated_from_parent_modules_and_path(
        self, monkeypatch
    ):
        inventory_process = subprocess.run(
            [
                sys.executable,
                "-E",
                "-s",
                "-B",
                "-m",
                probe.IMPORT_AUDIT_MODULE,
            ],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert inventory_process.returncode == 0, inventory_process.stderr.decode(
            errors="replace"
        )
        inventory = contract.parse_canonical_json_bytes(
            inventory_process.stdout, file_form=True
        )
        service = probe.WindowsProductionOfflineAdmission()
        native_environment = {
            name.upper(): value for name, value in os.environ.items()
        }
        monkeypatch.setattr(
            service,
            "_read_native_environment_block",
            lambda: dict(native_environment),
        )
        self._begin(service, seconds=15.0)
        succeeded = False
        try:
            environment = service.rederive_environment_inventory(
                {
                    "schema": "aigp-vq2-powered-environment-inventory/1",
                    "created_at_utc": UTC,
                    "variables": [],
                }
            )
            native_before = service._native_environment()
            modules_before = tuple(sys.modules.items())
            audit = service.rederive_import_inventory(
                inventory,
                probe.POWERED_EAGER_IMPORT_MODULES,
                environment_inventory=environment,
            )
            assert audit.inventory == inventory
            assert audit.origins_reverified is True
            assert audit.user_site_on_sys_path is False
            assert audit.unexpected_candidate_or_venv_modules == ()
            assert audit.unclassified_origins == ()
            assert service._native_environment() == native_before
            modules_after = tuple(sys.modules.items())
            assert len(modules_after) == len(modules_before)
            assert all(
                after_name == before_name and after_module is before_module
                for (before_name, before_module), (after_name, after_module) in zip(
                    modules_before, modules_after, strict=True
                )
            )
            succeeded = True
        finally:
            service.end_bounded_admission(succeeded=succeeded)
            service.close()

    def test_production_import_audit_rejects_parent_native_environment_mutation(
        self, monkeypatch
    ):
        inventory_process = subprocess.run(
            [
                sys.executable,
                "-E",
                "-s",
                "-B",
                "-m",
                probe.IMPORT_AUDIT_MODULE,
            ],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert inventory_process.returncode == 0, inventory_process.stderr.decode(
            errors="replace"
        )
        inventory = contract.parse_canonical_json_bytes(
            inventory_process.stdout, file_form=True
        )
        service = probe.WindowsProductionOfflineAdmission()
        native_environment = {
            name.upper(): value for name, value in os.environ.items()
        }
        monkeypatch.setattr(
            service,
            "_read_native_environment_block",
            lambda: dict(native_environment),
        )
        sentinel_name = "AIGP_NATIVE_MUTATION_SENTINEL"
        sentinel_value = "must-not-leak-from-diagnostic"

        def mutate_parent_environment(argv, **_kwargs):
            native_environment[sentinel_name] = sentinel_value
            return subprocess.CompletedProcess(
                list(argv),
                0,
                contract.canonical_json_file_bytes(inventory),
                b"",
            )

        monkeypatch.setattr(service, "_run_process", mutate_parent_environment)
        self._begin(service)
        try:
            environment = service.rederive_environment_inventory(
                {
                    "schema": "aigp-vq2-powered-environment-inventory/1",
                    "created_at_utc": UTC,
                    "variables": [],
                }
            )
            with pytest.raises(
                probe.OfflineAdmissionError,
                match=(
                    "parent native environment changed across isolated import audit; "
                    "expected_sha256=[a-f0-9]{64}; observed_sha256=[a-f0-9]{64}"
                ),
            ) as failure:
                service.rederive_import_inventory(
                    inventory,
                    probe.POWERED_EAGER_IMPORT_MODULES,
                    environment_inventory=environment,
                )
            assert sentinel_name not in str(failure.value)
            assert sentinel_value not in str(failure.value)
        finally:
            service.end_bounded_admission(succeeded=False)

    def test_production_import_audit_rejects_parent_module_graph_mutation(
        self, monkeypatch
    ):
        inventory_process = subprocess.run(
            [
                sys.executable,
                "-E",
                "-s",
                "-B",
                "-m",
                probe.IMPORT_AUDIT_MODULE,
            ],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert inventory_process.returncode == 0, inventory_process.stderr.decode(
            errors="replace"
        )
        inventory = contract.parse_canonical_json_bytes(
            inventory_process.stdout, file_form=True
        )
        service = probe.WindowsProductionOfflineAdmission()
        native_environment = {
            name.upper(): value for name, value in os.environ.items()
        }
        monkeypatch.setattr(
            service,
            "_read_native_environment_block",
            lambda: dict(native_environment),
        )
        sentinel_name = "aigp_parent_module_mutation_sentinel"

        def mutate_parent_modules(argv, **_kwargs):
            sys.modules[sentinel_name] = object()
            return subprocess.CompletedProcess(
                list(argv),
                0,
                contract.canonical_json_file_bytes(inventory),
                b"",
            )

        monkeypatch.setattr(service, "_run_process", mutate_parent_modules)
        self._begin(service)
        try:
            environment = service.rederive_environment_inventory(
                {
                    "schema": "aigp-vq2-powered-environment-inventory/1",
                    "created_at_utc": UTC,
                    "variables": [],
                }
            )
            with pytest.raises(
                probe.OfflineAdmissionError,
                match="parent import graph changed across isolated import audit",
            ):
                service.rederive_import_inventory(
                    inventory,
                    probe.POWERED_EAGER_IMPORT_MODULES,
                    environment_inventory=environment,
                )
        finally:
            sys.modules.pop(sentinel_name, None)
            service.end_bounded_admission(succeeded=False)

    def test_native_environment_snapshot_detects_non_python_drift(
        self, monkeypatch
    ):
        import ctypes

        service = probe.WindowsProductionOfflineAdmission()
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetEnvironmentVariableW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
        ]
        kernel.GetEnvironmentVariableW.restype = ctypes.c_uint
        kernel.SetEnvironmentVariableW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        kernel.SetEnvironmentVariableW.restype = ctypes.c_int
        name = "PYTHONSTARTUP"
        previous = self._native_environment_value(kernel, name)
        assert kernel.SetEnvironmentVariableW(name, "native-only-review")
        self._begin(service)
        try:
            assert os.environ.get(name) != "native-only-review"
            assert self._native_environment_value(kernel, name) == "native-only-review"
            complete_with_drive_state = {
                key.upper(): value for key, value in os.environ.items()
            }
            complete_with_drive_state[name] = "native-only-review"
            complete_with_drive_state["=C:"] = r"C:\native-drive-state"
            monkeypatch.setattr(
                service,
                "_read_native_environment_block",
                lambda: dict(complete_with_drive_state),
            )
            assert service.security_environment()[name] == "native-only-review"
            with pytest.raises(
                probe.OfflineAdmissionError,
                match="not a canonical spawn mapping",
            ):
                service.rederive_environment_inventory(
                    {
                        "schema": "aigp-vq2-powered-environment-inventory/1",
                        "created_at_utc": UTC,
                        "variables": [],
                    }
                )
        finally:
            service.end_bounded_admission(succeeded=True)
            service.close()
            assert kernel.SetEnvironmentVariableW(name, previous)

    def test_path_proof_retains_real_handle_until_deterministic_close(self, tmp_path):
        source = tmp_path / "identity.json"
        destination = tmp_path / "replacement.json"
        source.write_bytes(b"{}\n")
        path = str(source.resolve())
        service = probe.WindowsProductionOfflineAdmission()
        self._begin(service)
        proof = service.observe_file_identity(path, hash_kind="file_bytes")
        service.end_bounded_admission(succeeded=True)
        assert proof.path.final_path == path
        assert proof.path.retained_handle is True
        assert proof.path.volume_id.startswith("volume-")
        assert path in service._retained
        with pytest.raises(PermissionError):
            os.replace(path, destination)
        service.close()
        os.replace(path, destination)
        assert destination.read_bytes() == b"{}\n"

    def test_path_proof_rejects_hard_link_alias(self, tmp_path):
        system_service = probe.WindowsProductionOfflineAdmission()
        assert system_service._expected_file_link_count(POWERSHELL) == 2
        assert system_service._expected_file_link_count(POWERSHELL.upper()) == 1
        assert system_service._expected_file_link_count(
            r"C:\Windows\WinSxS\powershell.exe"
        ) == 1
        self._begin(system_service)
        system_succeeded = False
        try:
            system_proof = system_service.observe_file_identity(
                POWERSHELL, hash_kind="file_bytes"
            )
            assert system_proof.path.final_path == POWERSHELL
            assert system_proof.path.retained_handle is True
            assert system_proof.size_bytes > 0
            assert len(system_proof.sha256) == 64
            assert int(
                system_service._retained[POWERSHELL][1].nNumberOfLinks
            ) == 2
            launch = subprocess.run(
                [
                    POWERSHELL,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "exit 0",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            assert launch.returncode == 0, launch.stderr.decode(errors="replace")
            system_succeeded = True
        finally:
            system_service.end_bounded_admission(succeeded=system_succeeded)
            system_service.close()

        source = tmp_path / "identity.bin"
        alias = tmp_path / "identity-alias.bin"
        source.write_bytes(b"same bytes")
        try:
            os.link(source, alias)
        except OSError as exc:  # pragma: no cover - unusual non-NTFS temp volume
            pytest.skip(f"hard links unavailable: {exc}")
        service = probe.WindowsProductionOfflineAdmission()
        self._begin(service)
        try:
            with pytest.raises(probe.OfflineAdmissionError, match="hard-link"):
                service.observe_file_identity(
                    str(source.resolve()), hash_kind="file_bytes"
                )
        finally:
            service.end_bounded_admission(succeeded=False)
            service.close()

    def test_base_interpreter_dll_is_stdlib_and_mixed_namespace_is_detected(
        self, monkeypatch
    ):
        import _socket

        candidate = r"C:\candidate"
        venv = r"C:\venv"
        stdlib = r"C:\Python312"
        assert probe.WindowsProductionOfflineAdmission._classify_root(
            r"C:\Python312\DLLs\_socket.pyd",
            candidate_root=candidate,
            venv_root=venv,
            stdlib_root=stdlib,
        ) == "stdlib"
        actual = os.path.abspath(_socket.__spec__.origin)
        assert probe.WindowsProductionOfflineAdmission._classify_root(
            actual,
            candidate_root=str(Path.cwd().resolve()),
            venv_root=os.path.abspath(sys.prefix),
            stdlib_root=os.path.abspath(sys.base_prefix),
        ) == "stdlib"

        service = probe.WindowsProductionOfflineAdmission()
        monkeypatch.setattr(
            service,
            "_path_proof",
            lambda path, *, directory: probe.PathProof(
                path, path, "directory", "test-volume"
            ),
        )
        roots, classes = service._namespace_roots(
            [r"C:\candidate\pkg", r"C:\venv\pkg"],
            candidate_root=candidate,
            venv_root=venv,
            stdlib_root=stdlib,
        )
        assert roots == [r"C:\candidate\pkg", r"C:\venv\pkg"]
        assert classes == {"candidate", "venv"}
        service.close()

    def test_runtime_roots_reject_an_extra_sys_path(self, monkeypatch):
        import sysconfig

        candidate = str(Path.cwd().resolve())
        base = os.path.abspath(sys.base_prefix)
        venv = os.path.abspath(sys.prefix)
        paths = sysconfig.get_paths()
        expected = []
        for value in (
            candidate,
            ntpath.join(base, f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
            ntpath.join(base, "DLLs"),
            paths.get("stdlib"),
            base,
            venv,
            paths.get("purelib"),
            paths.get("platlib"),
        ):
            if value:
                value = os.path.abspath(value)
                if value not in expected:
                    expected.append(value)
        monkeypatch.setattr(sys, "path", expected.copy())
        service = probe.WindowsProductionOfflineAdmission()
        self._begin(service)
        try:
            assert service._runtime_roots(candidate, sysconfig) == (venv, base)
            sys.path.append(str(Path(candidate).parent.resolve()))
            with pytest.raises(probe.OfflineAdmissionError, match="alternate import root"):
                service._runtime_roots(candidate, sysconfig)
        finally:
            service.end_bounded_admission(succeeded=True)
            service.close()

    def test_exact_invocation_value_gate_accepts_only_frozen_flags_and_tail(self):
        tail = ["-E", "-s", "-B", "-m", probe.PROBE_MODULE, "--x", "value"]
        probe.WindowsProductionOfflineAdmission._validate_invocation_values(
            implementation="cpython",
            version=(3, 12, 2),
            ignore_environment=1,
            no_user_site=1,
            dont_write_bytecode=1,
            observed_argv=[r"C:\Python312\python.exe", *tail],
            expected_tail=tail,
        )
        with pytest.raises(probe.OfflineAdmissionError, match="-E -s -B"):
            probe.WindowsProductionOfflineAdmission._validate_invocation_values(
                implementation="cpython",
                version=(3, 12, 2),
                ignore_environment=0,
                no_user_site=1,
                dont_write_bytecode=1,
                observed_argv=["python", *tail],
                expected_tail=tail,
            )
        with pytest.raises(probe.OfflineAdmissionError, match="sole frozen argv"):
            probe.WindowsProductionOfflineAdmission._validate_invocation_values(
                implementation="cpython",
                version=(3, 12, 2),
                ignore_environment=1,
                no_user_site=1,
                dont_write_bytecode=1,
                observed_argv=["python", *tail, "--extra"],
                expected_tail=tail,
            )

    def test_git_proof_is_exact_root_read_only_and_blob_bound(
        self, tmp_path, monkeypatch
    ):
        repository = tmp_path / "repo"
        repository.mkdir()

        def git(*arguments, input_bytes=None):
            return subprocess.run(
                ["git", *arguments],
                cwd=repository,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        git("init", "-q")
        git("config", "user.name", "AIGP Offline Test")
        git("config", "user.email", "aigp-offline@example.invalid")
        (repository / "sample.txt").write_bytes(b"sample\n")
        git("add", "sample.txt")
        git("commit", "-qm", "fixture")
        repository_path = str(repository.resolve())
        index = repository / ".git" / "index"
        index_before = index.stat().st_mtime_ns
        service = probe.WindowsProductionOfflineAdmission()
        self._begin(service)
        try:
            proof = service.git_worktree(repository_path)
            assert proof.worktree_path == repository_path
            assert proof.tracked_clean and proof.untracked_clean and proof.ignored_clean
            assert proof.common_dir_outside_worktree is False
            assert index.stat().st_mtime_ns == index_before
            nested = repository / "nested"
            nested.mkdir()
            with pytest.raises(probe.OfflineAdmissionError, match="different worktree root"):
                service.git_worktree(str(nested.resolve()))
            monkeypatch.chdir(repository)
            inventory = service.rederive_implementation_inventory(
                {
                    "schema": "aigp-vq2-powered-implementation-inventory/1",
                    "commit": "0" * 40,
                    "tree": "0" * 40,
                    "entries": [],
                }
            )
            assert [entry["path"] for entry in inventory["entries"]] == ["sample.txt"]
            assert inventory["entries"][0]["sha256"] == hashlib.sha256(
                b"sample\n"
            ).hexdigest()
        finally:
            service.end_bounded_admission(succeeded=True)
            service.close()

    def test_bounded_process_polls_heartbeats_and_hard_expires(
        self, monkeypatch
    ):
        service = probe.WindowsProductionOfflineAdmission()
        heartbeats = []
        self._begin(service, seconds=2.5, heartbeat=lambda: heartbeats.append(time.perf_counter_ns()))
        environment = {
            name: value
            for name, value in service._native_environment().items()
            if not name.startswith("=")
        }
        result = service._run_process(
            [sys.executable, "-c", "import time; time.sleep(1.15)"],
            cwd=str(Path.cwd().resolve()),
            input_bytes=None,
            environment=environment,
            stdout_limit=1024,
        )
        assert result.returncode == 0
        assert heartbeats
        service.end_bounded_admission(succeeded=True)
        service.close()

        expired = probe.WindowsProductionOfflineAdmission()
        spawned = []
        real_popen = expired._subprocess.Popen

        def capture_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        monkeypatch.setattr(expired._subprocess, "Popen", capture_popen)
        started = time.perf_counter()
        self._begin(expired, seconds=0.15)
        try:
            with pytest.raises(probe.OfflineAdmissionError, match="deadline"):
                expired._run_process(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    cwd=str(Path.cwd().resolve()),
                    input_bytes=b"x" * expired._MAX_GIT_INPUT_BYTES,
                    environment=environment,
                    stdout_limit=1024,
                )
        finally:
            expired.end_bounded_admission(succeeded=False)
        assert time.perf_counter() - started < 1.0
        assert len(spawned) == 1 and spawned[0].poll() is not None
        assert not any(
            thread.name.startswith("aigp-offline-") and thread.is_alive()
            for thread in __import__("threading").enumerate()
        )

    @pytest.mark.parametrize(
        ("stream_name", "payload_size", "stdout_limit"),
        [
            ("stdout", 2_000_000, 1_024),
            ("stderr", 2_000_000, 1_024),
        ],
    )
    def test_bounded_process_hard_caps_running_child_output(
        self,
        monkeypatch,
        stream_name,
        payload_size,
        stdout_limit,
    ):
        service = probe.WindowsProductionOfflineAdmission()
        environment = {
            name.upper(): value for name, value in os.environ.items()
        }
        spawned = []
        real_popen = service._subprocess.Popen

        def capture_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        monkeypatch.setattr(service._subprocess, "Popen", capture_popen)
        child_code = (
            "import sys,time; "
            f"stream=sys.{stream_name}.buffer; "
            f"stream.write(b'x'*{payload_size}); stream.flush(); time.sleep(5)"
        )
        started = time.perf_counter()
        self._begin(service, seconds=3.0)
        try:
            with pytest.raises(
                probe.OfflineAdmissionError,
                match="output exceeded its limit",
            ) as failure:
                service._run_process(
                    [sys.executable, "-c", child_code],
                    cwd=str(Path.cwd().resolve()),
                    input_bytes=None,
                    environment=environment,
                    stdout_limit=stdout_limit,
                )
        finally:
            service.end_bounded_admission(succeeded=False)
        assert time.perf_counter() - started < 1.5
        assert len(spawned) == 1 and spawned[0].poll() is not None
        assert str(failure.value) == (
            "bounded identity process output exceeded its limit"
        )
        assert not any(
            thread.name.startswith("aigp-offline-") and thread.is_alive()
            for thread in __import__("threading").enumerate()
        )

    def test_bounded_process_accepts_output_exactly_at_stdout_limit(self):
        service = probe.WindowsProductionOfflineAdmission()
        environment = {
            name.upper(): value for name, value in os.environ.items()
        }
        payload = b"exact-limit"
        self._begin(service)
        try:
            result = service._run_process(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'exact-limit')",
                ],
                cwd=str(Path.cwd().resolve()),
                input_bytes=None,
                environment=environment,
                stdout_limit=len(payload),
            )
            assert result.returncode == 0
            assert result.stdout == payload
            assert result.stderr == b""
        finally:
            service.end_bounded_admission(succeeded=True)
            service.close()

    def test_bounded_process_thread_start_failure_reaps_child(
        self, monkeypatch
    ):
        service = probe.WindowsProductionOfflineAdmission()
        environment = {
            name.upper(): value for name, value in os.environ.items()
        }
        spawned = []
        real_popen = service._subprocess.Popen

        def capture_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        class RefusedThread:
            def __init__(self, *_args, **_kwargs):
                pass

            def start(self):
                raise RuntimeError("injected thread-start refusal")

        monkeypatch.setattr(service._subprocess, "Popen", capture_popen)
        monkeypatch.setattr(service._threading, "Thread", RefusedThread)
        self._begin(service)
        try:
            with pytest.raises(RuntimeError, match="thread-start refusal"):
                service._run_process(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    cwd=str(Path.cwd().resolve()),
                    input_bytes=None,
                    environment=environment,
                    stdout_limit=1_024,
                )
        finally:
            service.end_bounded_admission(succeeded=False)
        assert len(spawned) == 1 and spawned[0].poll() is not None

    def test_production_provider_construction_and_import_are_inert(self, monkeypatch):
        def unexpected_spawn(*_args, **_kwargs):
            raise AssertionError("provider construction spawned a process")

        monkeypatch.setattr(subprocess, "Popen", unexpected_spawn)
        service = probe.WindowsProductionOfflineAdmission()
        assert service._retained == {}
        assert service._deadline_monotonic_ns is None
        service.close()
        monkeypatch.undo()

        completed = subprocess.run(
            [
                sys.executable,
                "-E",
                "-s",
                "-B",
                "-c",
                "import scripts.aigp_vq2_powered_calibration_probe as p; "
                "assert p.WindowsProductionOfflineAdmission",
            ],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def _production_prechild_proof() -> dict[str, object]:
    wrapper = _process()
    launcher = {**wrapper, "pid": 200, "creation_filetime_100ns": 500}
    payload = {**wrapper, "pid": 201, "creation_filetime_100ns": 501}

    def owner(observed: int) -> dict[str, object]:
        return {
            "observed_monotonic_ns": observed,
            "ipv4_14550": [],
            "ipv6_14550": [],
            "ipv4_5600": [],
            "ipv6_5600": [],
        }

    value = {
        "schema": "aigp-vq2-simulator-process-proof/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "phase": "prechild",
        "observed_at_utc": UTC,
        "observed_monotonic_ns": 100,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "wrapper_process": wrapper,
        "launch": {
            "disposition": "absent_before_launcher_current_after",
            "observed_before_launch_monotonic_ns": 1,
            "launcher_return_monotonic_ns": 2,
            "launcher_exit_code": 0,
            "prelaunch_launcher_process": None,
            "prelaunch_payload_process": None,
        },
        "launcher_process": launcher,
        "payload_process": payload,
        "window": {
            "hwnd": 10,
            "owner_pid": payload["pid"],
            "visible": True,
            "unminimized": True,
            "responsive": True,
        },
        "build": 3385,
        "topology": "one_launcher_parent_retained_one_payload_child",
        "scheduled_task": {
            "name": "AIGP-P2-F03-A01-Launch",
            "observations": [
                {
                    "phase": phase,
                    "observed_monotonic_ns": index + 3,
                    "query_exit_code": 1,
                    "absent": True,
                }
                for index, phase in enumerate(
                    ("before_launch", "after_launcher_return", "before_child")
                )
            ],
        },
        "ports": {
            "owner_table_observations": [owner(10), owner(20)],
            "active_owner_observations": [],
            "exclusive_probes": [
                {
                    "host": "127.0.0.1",
                    "port": 14550,
                    "started_monotonic_ns": 30,
                    "ended_monotonic_ns": 31,
                    "result": "bound_and_closed",
                },
                {
                    "host": "0.0.0.0",
                    "port": 5600,
                    "started_monotonic_ns": 32,
                    "ended_monotonic_ns": 33,
                    "result": "bound_and_closed",
                },
            ],
            "status": "free",
        },
        "responsive": True,
    }
    return contract.validate_simulator_process_proof(value)


def _bare_production_boundary(freeze, *, clock=None):
    boundary = object.__new__(probe.WindowsProductionLiveBoundary)
    boundary.freeze = contract.validate_live_freeze(freeze)
    boundary.clock = clock or StepClock(start=1_000, step=1)
    boundary.secure = None
    boundary.runtime = None
    boundary.process_operations = None
    boundary.udp_operations = None
    boundary.wrapper_identity = _process()
    boundary.wrapper_argv = (PYTHON, "-E", "-s", "-B", "-m", probe.PROBE_MODULE)
    boundary.retained_wrapper = None
    boundary.child_pipe = None
    boundary.cleanup_pipe = None
    boundary.child_parent = None
    boundary.cleanup_parent = None
    boundary.handle_set = None
    boundary.child = None
    boundary.fallback = None
    boundary.tree_proofs = {}
    boundary.launch_result = None
    boundary.simulator_handles = {}
    boundary.prechild_proof = None
    boundary.postchild_proof = None
    boundary.training_attestation = None
    boundary.attempt_envelope = None
    boundary.attempt_envelope_sha256 = None
    boundary.active_owner_observations = []
    boundary.final_ports_contract = None
    boundary.lease_store = None
    boundary.powered_lease = None
    boundary.last_release_index = None
    boundary._spawn_anchors = {}
    boundary._output_paths = {}
    boundary.process_authorities = {}
    boundary.process_results = {}
    boundary.cleanup_certificates = {}
    boundary.stable_file_proofs = {}
    boundary._stable_file_handles = {}
    boundary._stable_file_payloads = {}
    boundary._sealed_spawn_environment = None
    boundary._sealed_spawn_environment_sha256 = None
    boundary._closed_process_ids = set()
    boundary._lease_release_attempted = False
    boundary._closed = False
    return boundary


def test_production_spawn_environment_seal_is_single_use_and_defensive(
    monkeypatch,
):
    freeze, _documents = _freeze()
    native_environment = {
        "SYSTEMROOT": r"C:\Windows",
        "NATIVE_ONLY": "native-value",
    }
    freeze["execution"]["launcher_environment_sha256"] = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
            native_environment
        )
    )
    boundary = _bare_production_boundary(freeze)
    native_reads = 0

    def read_native():
        nonlocal native_reads
        native_reads += 1
        return native_environment

    monkeypatch.setattr(boundary, "_native_environment_for_spawn", read_native)
    boundary.seal_spawn_environment(deadline_monotonic_ns=1_000_000)
    assert native_reads == 1
    first = boundary._sealed_spawn_environment_copy()
    second = boundary._sealed_spawn_environment_copy()
    assert first == second == native_environment
    assert first is not second and first is not native_environment
    first["NATIVE_ONLY"] = "caller-mutation"
    assert boundary._sealed_spawn_environment_copy() == native_environment
    with pytest.raises(probe.OrchestrationPhaseError, match="single-use"):
        boundary.seal_spawn_environment(deadline_monotonic_ns=1_000_000)
    assert native_reads == 1


def test_production_spawn_environment_drift_diagnostic_is_hash_only(monkeypatch):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    observed = {"SECRET_ENVIRONMENT_VALUE": "must-not-leak"}
    observed_sha256 = probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
        observed
    )
    expected_sha256 = freeze["execution"]["launcher_environment_sha256"]
    monkeypatch.setattr(
        boundary,
        "_native_environment_for_spawn",
        lambda: observed,
    )

    with pytest.raises(probe.OrchestrationPhaseError) as failure:
        boundary.seal_spawn_environment(deadline_monotonic_ns=1_000_000)

    detail = str(failure.value)
    assert f"expected_sha256={expected_sha256}" in detail
    assert f"observed_sha256={observed_sha256}" in detail
    assert "SECRET_ENVIRONMENT_VALUE" not in detail
    assert "must-not-leak" not in detail
    assert boundary._sealed_spawn_environment is None


def test_production_spawn_environment_seal_rejects_hidden_drive_state(monkeypatch):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    monkeypatch.setattr(
        boundary,
        "_native_environment_for_spawn",
        lambda: {"=C:": r"C:\hidden-drive-state", "SYSTEMROOT": r"C:\Windows"},
    )

    with pytest.raises(
        probe.OrchestrationPhaseError,
        match="native spawn environment is not canonical",
    ):
        boundary.seal_spawn_environment(deadline_monotonic_ns=1_000_000)

    assert boundary._sealed_spawn_environment is None
    assert boundary._spawn_environment_seal_attempted is True


@pytest.mark.parametrize(
    "system_root",
    [None, "", r"relative\Windows", r"\Windows", r"C:\Windows\..\evil"],
)
def test_production_spawn_environment_seal_requires_canonical_systemroot(
    monkeypatch, system_root
):
    freeze, _documents = _freeze()
    native_environment = {"NATIVE_ONLY": "native-value"}
    if system_root is not None:
        native_environment["SYSTEMROOT"] = system_root
    freeze["execution"]["launcher_environment_sha256"] = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
            native_environment
        )
    )
    boundary = _bare_production_boundary(freeze)
    monkeypatch.setattr(
        boundary,
        "_native_environment_for_spawn",
        lambda: dict(native_environment),
    )

    with pytest.raises(
        probe.OrchestrationPhaseError,
        match="lacks a canonical SYSTEMROOT",
    ):
        boundary.seal_spawn_environment(deadline_monotonic_ns=1_000_000)

    assert boundary._sealed_spawn_environment is None
    assert boundary._spawn_environment_seal_attempted is True


def test_scheduled_task_query_uses_only_the_sealed_environment(monkeypatch):
    freeze, _documents = _freeze()
    sealed = {
        "SYSTEMROOT": r"C:\FrozenWindows",
        "NATIVE_ONLY": "sealed-value",
    }
    freeze["execution"]["launcher_environment_sha256"] = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(sealed)
    )
    boundary = _bare_production_boundary(freeze)
    boundary._sealed_spawn_environment = sealed
    boundary._sealed_spawn_environment_sha256 = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(sealed)
    )
    monkeypatch.setenv("SYSTEMROOT", r"C:\MutableWindows")
    monkeypatch.setenv("OS_ONLY", "must-not-be-forwarded")
    captured: dict[str, Any] = {}

    def popen(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(boundary, "_wait_subprocess", lambda *_args, **_kwargs: 1)
    observation = boundary._query_task_absent(
        "before_launch",
        deadline_monotonic_ns=1_000_000,
        heartbeat=probe.HeartbeatPump(
            "launcher_return", 1_000_000, 1_000, lambda: None
        ),
    )

    assert observation["absent"] is True
    assert captured["argv"] == [
        r"C:\FrozenWindows\System32\schtasks.exe",
        "/Query",
        "/TN",
        "AIGP-P2-F03-A01-Launch",
    ]
    assert captured["env"] == sealed
    assert captured["env"] is not sealed
    assert "OS_ONLY" not in captured["env"]


@pytest.mark.parametrize(
    "system_root",
    [None, "", r"relative\Windows", r"\Windows", r"C:\Windows\..\evil"],
)
def test_scheduled_task_query_requires_canonical_sealed_systemroot(
    monkeypatch, system_root
):
    freeze, _documents = _freeze()
    sealed = {"NATIVE_ONLY": "sealed-value"}
    if system_root is not None:
        sealed["SYSTEMROOT"] = system_root
    freeze["execution"]["launcher_environment_sha256"] = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(sealed)
    )
    boundary = _bare_production_boundary(freeze)
    boundary._sealed_spawn_environment = sealed
    boundary._sealed_spawn_environment_sha256 = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(sealed)
    )
    popen_called = False

    def popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("scheduled-task query must not spawn")

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(
        probe.OrchestrationPhaseError,
        match="lacks a canonical SYSTEMROOT",
    ):
        boundary._query_task_absent(
            "before_launch",
            deadline_monotonic_ns=1_000_000,
            heartbeat=probe.HeartbeatPump(
                "launcher_return", 1_000_000, 1_000, lambda: None
            ),
        )
    assert popen_called is False


def test_production_launcher_passes_exact_native_environment_not_os_environ(
    monkeypatch,
):
    freeze, _documents = _freeze()
    native_environment = {"NATIVE_ONLY": "native-value"}
    freeze["execution"]["launcher_environment_sha256"] = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
            native_environment
        )
    )
    boundary = _bare_production_boundary(freeze)
    boundary._sealed_spawn_environment = native_environment
    boundary._sealed_spawn_environment_sha256 = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
            native_environment
        )
    )
    monkeypatch.setenv("NATIVE_ONLY", "os-environ-value")
    monkeypatch.setenv("OS_ONLY", "must-not-be-forwarded")
    monkeypatch.setattr(
        boundary,
        "_query_task_absent",
        lambda phase, **_kwargs: {
            "phase": phase,
            "observed_monotonic_ns": 1,
            "query_exit_code": 1,
            "absent": True,
        },
    )
    monkeypatch.setattr(
        boundary,
        "_enumerate_simulator",
        lambda: {"launcher": None, "payload": None},
    )
    monkeypatch.setattr(boundary, "_wait_subprocess", lambda *_args, **_kwargs: 0)
    captured: dict[str, Any] = {}

    def popen(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(subprocess, "Popen", popen)
    result = boundary.launch_and_wait(
        freeze=freeze,
        deadline_monotonic_ns=1_000_000,
        heartbeat=probe.HeartbeatPump("launcher_return", 1_000_000, 1_000, lambda: None),
    )
    assert result["launch"]["disposition"] == "absent_before_launcher_current_after"
    assert captured["env"] is not native_environment
    assert captured["env"] == {"NATIVE_ONLY": "native-value"}
    assert "OS_ONLY" not in captured["env"]


def test_production_launcher_refuses_native_only_environment_drift_before_popen(
    monkeypatch,
):
    freeze, _documents = _freeze()
    frozen_environment = {"NATIVE_ONLY": "frozen-value"}
    freeze["execution"]["launcher_environment_sha256"] = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
            frozen_environment
        )
    )
    boundary = _bare_production_boundary(freeze)
    boundary._sealed_spawn_environment = {"NATIVE_ONLY": "native-drift"}
    boundary._sealed_spawn_environment_sha256 = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
            boundary._sealed_spawn_environment
        )
    )
    monkeypatch.setenv("NATIVE_ONLY", "frozen-value")
    popen_called = False

    def popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("launcher Popen must not run after environment drift")

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(
        probe.OrchestrationPhaseError,
        match="sealed spawn environment identity changed",
    ):
        boundary.launch_and_wait(
            freeze=freeze,
            deadline_monotonic_ns=1_000_000,
            heartbeat=probe.HeartbeatPump(
                "launcher_return", 1_000_000, 1_000, lambda: None
            ),
        )
    assert popen_called is False


class _ProductionRetained:
    def __init__(self, handle, identity, calls):
        self.handle_value = handle
        self.identity = copy.deepcopy(identity)
        self.calls = calls
        self.closed = False

    def alive(self):
        return not self.closed

    def reprove(self):
        return copy.deepcopy(self.identity)

    def close(self):
        self.calls.append(("close-retained", self.handle_value))
        self.closed = True


class _ProductionPipe:
    def __init__(self, read_handle, write_handle, calls):
        self.read_handle = read_handle
        self.write_handle = write_handle
        self.calls = calls
        self.released = False
        self.read_closed = False
        self.write_closed = False

    def abort(self):
        self.calls.append(("abort-pipe", self.read_handle, self.write_handle))
        self.read_closed = True
        self.write_closed = True


class _Containment:
    def __init__(self, handle):
        self.handle = handle

    def to_primitive(self):
        return {
            "handle_value": self.handle,
            "assigned_before_capability_release": True,
            "breakaway_allowed": False,
            "silent_breakaway_allowed": False,
            "kill_on_close": False,
            "process_in_job": True,
        }


class _ProductionSpawned:
    def __init__(self, identity, pipe, job_handle, calls):
        self.identity = copy.deepcopy(identity)
        self.capability_pipe = pipe
        self.containment = _Containment(job_handle)
        self.calls = calls
        self.released_secret = None
        self.closed_with = None

    def release_capability(self, secret, **kwargs):
        self.released_secret = bytes(secret)
        self.capability_pipe.released = True
        self.capability_pipe.write_closed = True
        self.calls.append(("release-capability", len(self.released_secret)))

    def close_retained_handles(self, *, tree_exit_proof):
        self.closed_with = tree_exit_proof
        self.calls.append(("close-spawned", self.identity["pid"]))
        return ("process", "job")


def test_production_training_attestation_binds_exact_challenge_and_proof(monkeypatch):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    process_proof = _production_prechild_proof()
    boundary.prechild_proof = process_proof
    observed: dict[str, str] = {}

    def response(*, challenge, deadline_monotonic_ns, heartbeat):
        observed["challenge"] = challenge
        return f"TRAINING {challenge}"

    monkeypatch.setattr(boundary, "_read_training_response", response)
    monkeypatch.setattr("secrets.token_hex", lambda size: "0123456789abcdef" * 2)
    result = boundary.attest_training(
        topology_proof=process_proof,
        deadline_monotonic_ns=10_000,
        heartbeat=probe.HeartbeatPump("training", 10_000, 1_000, lambda: None),
    )
    assert result["mode"] == "Training"
    assert result["challenge_sha256"] == hashlib.sha256(
        observed["challenge"].encode("ascii")
    ).hexdigest()
    assert result["simulator_process_proof_sha256"] == contract.canonical_file_sha256(
        process_proof
    )
    assert "0123456789abcdef" not in repr(result)


def test_production_training_attestation_rejects_nonexact_response(monkeypatch):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    boundary.prechild_proof = _production_prechild_proof()
    monkeypatch.setattr("secrets.token_hex", lambda size: "a" * 32)
    monkeypatch.setattr(
        boundary,
        "_read_training_response",
        lambda **kwargs: "TRAINING " + "A" * 32,
    )
    with pytest.raises(probe.OrchestrationPhaseError) as failure:
        boundary.attest_training(
            topology_proof=boundary.prechild_proof,
            deadline_monotonic_ns=10_000,
            heartbeat=probe.HeartbeatPump("training", 10_000, 1_000, lambda: None),
        )
    assert failure.value.reason_code == "training_unattested"


def test_production_attempt_handle_allocation_is_distinct_and_closes(monkeypatch):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    calls = []
    boundary.retained_wrapper = _ProductionRetained(70, _process(), calls)

    class Runtime:
        next_pipe = iter(((41, 51), (43, 53)))
        next_parent = iter((42, 44))

        @classmethod
        def create_capability_pipe(cls, *, operations):
            return _ProductionPipe(*next(cls.next_pipe), calls)

        @classmethod
        def retain_process(cls, pid, argv, *, inheritable, operations):
            assert inheritable is True
            return _ProductionRetained(next(cls.next_parent), _process(), calls)

    boundary.runtime = Runtime
    handles = boundary.allocate_attempt_handles(_process())
    assert handles == probe.AttemptHandleSet(41, 42, 43, 44)
    assert len(set(handles.__dict__.values())) == 4
    boundary.close_attempt_handles(handles, deadline_monotonic_ns=10_000)
    assert ("abort-pipe", 41, 51) in calls
    assert ("abort-pipe", 43, 53) in calls
    assert ("close-retained", 42) in calls
    assert ("close-retained", 44) in calls


def test_production_acquire_binds_raw_secret_and_defers_final_index(
    tmp_path, monkeypatch
):
    from scripts import aigp_live_lease

    freeze, offline, admission = _admission(tmp_path)
    material = _material(admission)
    boundary = _bare_production_boundary(freeze)
    boundary.attempt_envelope = copy.deepcopy(material.envelope)
    boundary.attempt_envelope_sha256 = contract.canonical_file_sha256(
        material.envelope
    )
    monkeypatch.setattr(
        boundary, "_load_attempt_envelope", lambda: copy.deepcopy(material.envelope)
    )
    parent = probe.SecureDirectoryReceipt(
        freeze["paths"]["attempt_dir"],
        freeze["paths"]["attempt_dir"],
        freeze["paths"]["evidence_root"],
        "volume-1",
        "volume-1",
        "sid",
        "sid",
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    monkeypatch.setattr(boundary, "_attempt_directory_receipt", lambda: parent)

    class Secure:
        def create_private_directory_create_new(self, path, *, parent_path):
            return probe.SecureDirectoryReceipt(
                path,
                path,
                parent_path,
                "volume-1",
                "volume-1",
                "sid",
                "sid",
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            )

    captured = {}

    class Store:
        def __init__(self, directory, index, **kwargs):
            captured.update(kwargs)
            self.records = []
            self.record_hashes = []

    class Lease:
        def __init__(self, store, **kwargs):
            self.store = store
            self.kwargs = kwargs
            self.is_active = False
            self.release_calls = 0
            captured["lease"] = self

        def acquire(self):
            self.is_active = True
            self.store.records.append({"event": "acquired"})
            self.store.record_hashes.append(H)
            boundary.clock.advance(2_000_000)
            return self

        def release(self):
            self.release_calls += 1
            self.is_active = False

    boundary.secure = Secure()
    monkeypatch.setattr(aigp_live_lease, "PoweredLeaseLedgerStore", Store)
    monkeypatch.setattr(aigp_live_lease, "PoweredLiveSimulatorLease", Lease)
    lease = boundary.acquire(
        owner_secret=bytearray(b"L" * 32),
        qpc_frequency_hz=10_000_000,
        deadline_monotonic_ns=1_000_000,
    )
    assert lease.is_active is True
    assert captured["publish_final_index"] is False
    assert captured["attempt_context_sha256"] == material.context_sha256
    assert captured["wrapper_process"] == _process()
    assert boundary.clock.value > 1_000_000
    assert captured["lease"].release_calls == 0
    assert captured["lease"].is_active is True
    with pytest.raises(probe.OrchestrationPhaseError):
        boundary.acquire(
            owner_secret=bytearray(b"X" * 32),
            qpc_frequency_hz=10_000_000,
            deadline_monotonic_ns=1_000_000,
        )


def test_production_spawn_binds_lease_row_and_releases_only_secret(
    tmp_path, monkeypatch
):
    freeze, offline, admission = _admission(tmp_path)
    material = _material(admission)
    boundary = _bare_production_boundary(freeze)
    boundary.attempt_envelope = copy.deepcopy(material.envelope)
    boundary.attempt_envelope_sha256 = contract.canonical_file_sha256(
        material.envelope
    )
    boundary.handle_set = probe.AttemptHandleSet(41, 42, 43, 44)
    calls = []
    boundary.child_pipe = _ProductionPipe(41, 61, calls)
    boundary.cleanup_pipe = _ProductionPipe(43, 63, calls)
    boundary.child_parent = _ProductionRetained(42, _process(), calls)
    boundary.cleanup_parent = _ProductionRetained(44, _process(), calls)
    boundary.prechild_proof = _production_prechild_proof()
    boundary.training_attestation = {
        "schema": "test-training"
    }
    monkeypatch.setattr(
        boundary,
        "_authority_context",
        lambda: (copy.deepcopy(material.envelope), H, H2),
    )
    monkeypatch.setattr(
        boundary, "_load_attempt_envelope", lambda: copy.deepcopy(material.envelope)
    )
    sealed_environment = {"A": "B"}
    freeze_environment_sha256 = (
        probe.WindowsProductionLiveBoundary._spawn_environment_sha256(
            sealed_environment
        )
    )
    boundary.freeze["execution"]["launcher_environment_sha256"] = (
        freeze_environment_sha256
    )
    boundary._sealed_spawn_environment = sealed_environment
    boundary._sealed_spawn_environment_sha256 = freeze_environment_sha256
    parent = probe.SecureDirectoryReceipt(
        freeze["paths"]["attempt_dir"],
        freeze["paths"]["attempt_dir"],
        freeze["paths"]["evidence_root"],
        "volume-1",
        "volume-1",
        "sid",
        "sid",
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    monkeypatch.setattr(boundary, "_attempt_directory_receipt", lambda: parent)

    class Secure:
        next_handle = iter((51, 52))

        def create_inheritable_output_file(self, path, **kwargs):
            calls.append(("create-output", path))
            return next(self.next_handle)

    class Operations:
        def close_handle(self, handle):
            calls.append(("close-standard", handle))

    child_identity = {
        **_process(),
        "pid": 124,
        "creation_filetime_100ns": 457,
        "argv_sha256": contract.canonical_object_sha256(list(material.child_argv)),
    }
    spawned = _ProductionSpawned(child_identity, boundary.child_pipe, 80, calls)

    class Runtime:
        @staticmethod
        def spawn_blocked_child(argv, **kwargs):
            calls.append(("spawn", tuple(kwargs["inherited_handles"]) if "inherited_handles" in kwargs else None))
            assert kwargs["environment"] == sealed_environment
            assert kwargs["environment"] is not sealed_environment
            assert kwargs["capability_pipe"] is boundary.child_pipe
            assert kwargs["parent_process"] is boundary.child_parent
            assert kwargs["stdin_handle"] == 53
            assert kwargs["stdout_handle"] == 51
            assert kwargs["stderr_handle"] == 52
            return spawned

    class Store:
        def __init__(self):
            self.records = [{"event": "heartbeat", "child_process": None, "cleanup_process": None}]
            self.record_hashes = [H3]

    class Lease:
        is_active = True

        def bind_child_process(self, identity):
            self.child = copy.deepcopy(identity)

        def publish_phase(self, phase):
            assert phase == "child_spawn"
            row = {
                "event": "phase",
                "phase": phase,
                "owner_role": "wrapper",
                "owner_process": _process(),
                "wrapper_process": _process(),
                "attempt_envelope_sha256": boundary.attempt_envelope_sha256,
                "attempt_context_sha256": material.context_sha256,
                "child_process": copy.deepcopy(self.child),
                "cleanup_process": None,
            }
            store.records.append(row)
            store.record_hashes.append("9" * 64)
            return row

    store = Store()
    boundary.secure = Secure()
    boundary.process_operations = Operations()
    boundary.runtime = Runtime()
    boundary.lease_store = store
    boundary.powered_lease = Lease()
    null_descriptor = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(
        boundary,
        "_open_readonly_inheritable_nul",
        lambda: (null_descriptor, 53),
    )
    result = boundary.spawn_powered_child_blocked(
        argv=material.child_argv,
        handles=boundary.handle_set,
        deadline_monotonic_ns=1_000_000,
        heartbeat=probe.HeartbeatPump("child_spawn", 1_000_000, 1_000, lambda: None),
    )
    assert result.handle is spawned
    assert result.authority["lease_record_sha256"] == "9" * 64
    assert result.authority["process"] == child_identity
    assert store.records[-1]["child_process"] == child_identity
    assert ("close-standard", 51) in calls and ("close-standard", 52) in calls

    frame = bytearray(contract.encode_capability_frame(b"C" * 32))
    boundary.release_child_capability(
        spawned,
        frame=frame,
        deadline_monotonic_ns=1_000_000,
        heartbeat=probe.HeartbeatPump("child_spawn", 1_000_000, 1_000, lambda: None),
    )
    assert spawned.released_secret == b"C" * 32
    assert len(spawned.released_secret) == 32


def test_production_forced_tree_termination_is_recorded_but_never_proved_cleanup():
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    child = object()
    boundary.child = child

    class Proof:
        def __init__(self, state):
            self.state = state
            self.termination_is_cleanup_proof = False

        def to_primitive(self):
            return {
                "state": self.state,
                "termination_is_cleanup_proof": False,
            }

    class Runtime:
        @staticmethod
        def wait_job_process_tree_exit(*args, **kwargs):
            return Proof("residue")

        @staticmethod
        def terminate_job_process_tree_residue(*args, **kwargs):
            return Proof("terminated_residue")

    boundary.runtime = Runtime()
    with pytest.raises(probe.OrchestrationPhaseError) as failure:
        boundary.prove_child_tree_exit(
            child,
            deadline_monotonic_ns=10_000,
            heartbeat=probe.HeartbeatPump("child_exit", 10_000, 1_000, lambda: None),
        )
    assert failure.value.reason_code == "process_residue"
    assert boundary.tree_proofs[id(child)].state == "terminated_residue"
    assert boundary.tree_proofs[id(child)].termination_is_cleanup_proof is False


def test_production_child_cleanup_accepts_reached_prefix_then_cleanup_only():
    anchor = 1_000
    authority = {
        "absolute_deadlines": {
            "anchor": anchor,
            "total": anchor + 110_000_000_000,
            "prepower": anchor + 52_000_000_000,
            "powered": anchor + 57_000_000_000,
            "cleanup": anchor + 72_000_000_000,
            "replay_close": anchor + 107_000_000_000,
            "exit": anchor + 110_000_000_000,
        }
    }

    def row(phase, started, parent):
        return {
            "phase": phase,
            "started_monotonic_ns": started,
            "parent_deadline_monotonic_ns": parent,
        }

    rows = [
        row("connect", anchor + 1, authority["absolute_deadlines"]["prepower"]),
        row("preflight", anchor + 2, authority["absolute_deadlines"]["prepower"]),
        row("cleanup", anchor + 3, authority["absolute_deadlines"]["cleanup"]),
    ]
    assert probe.WindowsProductionLiveBoundary._phase_deadlines_bind_authority(
        rows,
        role="powered_child",
        authority=authority,
        completed_result=True,
        certificate=True,
    )
    rows[1] = row(
        "reset_epoch", anchor + 2, authority["absolute_deadlines"]["prepower"]
    )
    assert not probe.WindowsProductionLiveBoundary._phase_deadlines_bind_authority(
        rows,
        role="powered_child",
        authority=authority,
        completed_result=True,
        certificate=True,
    )


def test_production_takeover_lease_binding_requires_exact_owner_process():
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    child_identity = {**_process(), "pid": 321, "creation_filetime_100ns": 654}
    child = type("Child", (), {"identity": child_identity})()
    boundary.wrapper_identity = _process()
    record = {
        "owner_role": "powered-child-parent-death",
        "owner_process": copy.deepcopy(child_identity),
        "wrapper_process": _process(),
        "child_process": copy.deepcopy(child_identity),
        "cleanup_process": None,
    }
    boundary.lease_store = type(
        "Store", (), {"records": [record], "record_hashes": [H]}
    )()
    certificate = {
        "parent_state": {
            "mode": "signaled_takeover",
            "wrapper_process": _process(),
            "takeover_lease_record_sha256": H,
        },
        "lease": {
            "owner_role": "powered-child-parent-death",
            "generation": 0,
            "record_sha256": H,
            "authority_valid": True,
        },
    }
    assert boundary._lease_certificate_binding(
        certificate, role="powered_child", child=child
    ) == (True, True)
    record["owner_process"] = _process()
    assert boundary._lease_certificate_binding(
        certificate, role="powered_child", child=child
    ) == (False, True)


def test_production_outbound_audit_must_byte_count_certificate_receipts():
    receipts = [
        {
            "schema": "aigp-vq2-attitude-target-outbound/1",
            "outcome": "returned",
        },
        {
            "schema": "aigp-vq2-nonattitude-outbound/1",
            "category": "disarm",
            "outcome": "returned",
        },
    ]
    audit = {
        "timesync": 0,
        "gcs_heartbeat": 0,
        "sim_reset": 0,
        "arm": 0,
        "disarm": 1,
        "attitude_target": 1,
        "position_target": 0,
        "other_command": 0,
        "receipt_count": 2,
        "receipt_returned": 2,
        "receipt_raised": 0,
        "receipt_dropped": 0,
        "receipt_buffered": 0,
    }
    assert probe.WindowsProductionLiveBoundary._audit_binds_certificate_receipts(
        audit, {"outbound_receipts": receipts}
    )
    audit["receipt_count"] = 1
    assert not probe.WindowsProductionLiveBoundary._audit_binds_certificate_receipts(
        audit, {"outbound_receipts": receipts}
    )


def test_production_fallback_collection_reason_does_not_negate_cleanup(monkeypatch):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    child = object()
    boundary.fallback = child

    class Proof:
        state = "exited"

        @staticmethod
        def to_primitive():
            return {"state": "exited"}

    monkeypatch.setattr(boundary, "_poll_root_exit", lambda *args, **kwargs: ("exited", 0))
    monkeypatch.setattr(
        boundary,
        "_collect_supervision_evidence",
        lambda *args, **kwargs: (
            True,
            False,
            False,
            ("unexpected_outbound",),
            {},
        ),
    )
    monkeypatch.setattr(boundary, "_wait_job_tree", lambda *args, **kwargs: Proof())
    outcome = boundary.supervise_cleanup_fallback(
        child,
        deadline_monotonic_ns=2_000_000_000,
        heartbeat=probe.HeartbeatPump(
            "fallback", 2_000_000_000, 1_000_000_000, lambda: None
        ),
    )
    assert outcome.cleanup_proved is True
    assert outcome.reason_codes == ("unexpected_outbound",)
    assert boundary.tree_proofs[id(child)].state == "exited"


def test_production_fallback_timeout_forces_residue_but_never_proves_cleanup(
    monkeypatch,
):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    child = object()
    boundary.fallback = child

    class Proof:
        def __init__(self, state):
            self.state = state
            self.termination_is_cleanup_proof = False

        def to_primitive(self):
            return {
                "state": self.state,
                "termination_is_cleanup_proof": False,
            }

    monkeypatch.setattr(boundary, "_poll_root_exit", lambda *args, **kwargs: ("timeout", None))
    monkeypatch.setattr(boundary, "_wait_job_tree", lambda *args, **kwargs: Proof("residue"))
    monkeypatch.setattr(
        boundary,
        "_terminate_job_tree",
        lambda *args, **kwargs: Proof("terminated_residue"),
    )
    outcome = boundary.supervise_cleanup_fallback(
        child,
        deadline_monotonic_ns=2_000_000_000,
        heartbeat=probe.HeartbeatPump(
            "fallback", 2_000_000_000, 1_000_000_000, lambda: None
        ),
    )
    assert outcome.cleanup_proved is False
    assert outcome.reason_codes == ("process_residue",)
    assert boundary.tree_proofs[id(child)].state == "terminated_residue"
    assert boundary.tree_proofs[id(child)].termination_is_cleanup_proof is False


def test_production_retained_reader_rejects_hardlinks_and_closes_handle(monkeypatch):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    calls = []

    class TimeValue:
        dwHighDateTime = 0
        dwLowDateTime = 0

    class Info:
        dwFileAttributes = 0
        dwVolumeSerialNumber = 9
        nFileSizeHigh = 0
        nFileSizeLow = 0
        nNumberOfLinks = 2
        nFileIndexHigh = 0
        nFileIndexLow = 4
        ftCreationTime = TimeValue()
        ftLastWriteTime = TimeValue()

    class Kernel:
        @staticmethod
        def CreateFileW(*args):
            return 77

    class Secure:
        _kernel = Kernel()
        _GENERIC_READ = 1
        _FILE_READ_ATTRIBUTES = 2
        _READ_CONTROL = 4
        _FILE_SHARE_READ = 1
        _OPEN_EXISTING = 3
        _FILE_FLAG_OPEN_REPARSE_POINT = 8
        _INVALID_HANDLE = -1
        _ctypes = type("Ctypes", (), {"get_last_error": staticmethod(lambda: 0)})()

        @staticmethod
        def _file_information(handle, *, directory):
            return Info()

        @staticmethod
        def _final_path(handle):
            return freeze["paths"]["runner_stdout"]

        @staticmethod
        def _verify_private_acl(handle):
            return "sid", True

        @staticmethod
        def _normcase(path):
            return ntpath.normcase(path)

        @staticmethod
        def _volume_id(info):
            return "volume-1"

        @staticmethod
        def _close_handle(handle):
            calls.append(handle)

    boundary.secure = Secure()
    parent = probe.SecureDirectoryReceipt(
        freeze["paths"]["attempt_dir"],
        freeze["paths"]["attempt_dir"],
        freeze["paths"]["evidence_root"],
        "volume-1",
        "volume-1",
        "sid",
        "sid",
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    monkeypatch.setattr(boundary, "_attempt_directory_receipt", lambda: parent)
    with pytest.raises(probe.OrchestrationPhaseError) as failure:
        boundary._read_retained_complete_file(
            freeze["paths"]["runner_stdout"],
            maximum_bytes=1024,
            deadline_monotonic_ns=10_000,
            heartbeat=probe.HeartbeatPump("read", 10_000, 1_000, lambda: None),
            label="runner stdout",
        )
    assert failure.value.reason_code == "artifact_mismatch"
    assert calls == [77]


def test_production_boundary_close_releases_retained_file_handles():
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    calls = []

    class Secure:
        def _close_handle(self, handle):
            calls.append(("stable", handle))

        def close(self):
            calls.append(("secure", None))

    boundary.secure = Secure()
    boundary._stable_file_handles = {"one": 71, "two": 72}
    boundary._stable_file_payloads = {"one": b"1", "two": b"2"}
    boundary.simulator_handles = {
        "launcher": _ProductionRetained(81, _process(), calls),
        "payload": _ProductionRetained(82, _process(), calls),
    }
    boundary.retained_wrapper = _ProductionRetained(83, _process(), calls)
    boundary.powered_lease = type("Lease", (), {"is_active": False})()
    boundary.close()
    assert calls[:2] == [("stable", 72), ("stable", 71)]
    assert boundary._stable_file_handles == {}
    assert ("secure", None) in calls
    assert boundary._closed is True


def test_production_boundary_close_does_not_reclose_orchestrator_process_handle():
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(freeze)
    calls = []
    spawned = _ProductionSpawned(
        {**_process(), "pid": 222, "creation_filetime_100ns": 333},
        _ProductionPipe(41, 51, calls),
        80,
        calls,
    )

    class Proof:
        state = "exited"

    class Secure:
        @staticmethod
        def close():
            calls.append(("secure", None))

    boundary.child = spawned
    boundary.tree_proofs[id(spawned)] = Proof()
    boundary.secure = Secure()
    boundary.powered_lease = type("Lease", (), {"is_active": False})()
    boundary.close_process_handle(spawned, deadline_monotonic_ns=10_000)
    boundary.close()
    assert calls.count(("close-spawned", 222)) == 1
    assert id(spawned) in boundary._closed_process_ids


def test_production_boundary_close_forces_unproved_tree_and_keeps_noncleanup_label(
    monkeypatch,
):
    freeze, _documents = _freeze()
    boundary = _bare_production_boundary(
        freeze, clock=StepClock(start=1_000, step=1)
    )
    calls = []
    spawned = _ProductionSpawned(
        {**_process(), "pid": 223, "creation_filetime_100ns": 334},
        _ProductionPipe(41, 51, calls),
        81,
        calls,
    )

    class Proof:
        def __init__(self, state):
            self.state = state
            self.termination_is_cleanup_proof = False

        def to_primitive(self):
            return {
                "state": self.state,
                "termination_is_cleanup_proof": False,
            }

    class Lease:
        is_active = True

        @staticmethod
        def heartbeat(*, phase):
            calls.append(("lease-heartbeat", phase))

    class Secure:
        @staticmethod
        def close():
            calls.append(("secure", None))

    def terminate(*args, **kwargs):
        kwargs["heartbeat"]()
        return Proof("terminated_residue")

    boundary.child = spawned
    boundary.powered_lease = Lease()
    boundary.secure = Secure()
    boundary.attempt_envelope = {
        "context": {
            "wrapper_absolute_deadlines": {
                "total_deadline_monotonic_ns": 10_000,
            }
        }
    }
    monkeypatch.setattr(
        boundary, "_wait_job_tree", lambda *args, **kwargs: Proof("residue")
    )
    monkeypatch.setattr(boundary, "_terminate_job_tree", terminate)
    with pytest.raises(probe.SecureBoundaryError) as failure:
        boundary.close()
    assert "missing_tree_exit_proof" in str(failure.value)
    assert "forced_termination_noncleanup" in str(failure.value)
    assert boundary.tree_proofs[id(spawned)].state == "terminated_residue"
    assert (
        boundary.tree_proofs[id(spawned)].termination_is_cleanup_proof is False
    )
    assert spawned.closed_with is boundary.tree_proofs[id(spawned)]
    assert ("lease-heartbeat", "child_exit_proof") in calls
    assert id(spawned) in boundary._closed_process_ids


def test_fallback_collection_reason_is_preserved_with_proved_cleanup(tmp_path):
    def configure(live, secure, freeze):
        live.child_outcome = probe.ChildSupervisionOutcome(
            cleanup_proved=False,
            collection_valid=True,
            artifact_state_patch=_sealed_child_artifacts(),
        )
        live.fallback_outcome = probe.FallbackSupervisionOutcome(
            cleanup_proved=True,
            reason_codes=("unexpected_outbound",),
        )

    _freeze_value, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "unexpected_outbound" in result.reason_codes
    assert live.invalid_context["cleanup_state"]["fallback"] == "proved"


def test_wrapper_phase_sanitizes_keyboard_interrupt_and_runs_release_recovery(tmp_path):
    def configure(live, secure, freeze):
        live.failures["topology.prechild"] = KeyboardInterrupt()

    freeze, _, _, live, _, result = _run_orchestration(tmp_path, configure)
    assert result.status == "invalid"
    assert "internal_error" in result.reason_codes
    assert result.live_kernel_released is True
    assert result.live_release_proved is True
    assert live.invalid_context["cleanup_state"]["lease"] == "released"
    assert live.events.count("lease.release") == 1
