from __future__ import annotations

import copy

import pytest

from scripts import aigp_vq2_powered_attempt as contract


H = "a" * 64
H2 = "b" * 64
H3 = "c" * 64
COMMIT = "d" * 40
UTC = "2026-07-20T12:34:56.123456Z"
LIVE_WORKTREE = r"C:\Users\John\aigp-worktrees\wt-package2-powered-calibration-live"
PYTHON = r"C:\Users\John\killallhumans\.venv\Scripts\python.exe"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def identity(name: str = "artifact") -> dict[str, object]:
    return {
        "path": contract.EVIDENCE_ROOT + rf"\{name}.json",
        "sha256": H,
    }


def artifact(name: str, *, digest: str = H) -> dict[str, object]:
    return {
        "name": name,
        "path": contract.EVIDENCE_ROOT + rf"\artifact-{name.replace('/', '-')}",
        "size_bytes": 1,
        "sha256": digest,
    }


def process(pid: int = 10) -> dict[str, object]:
    return {
        "pid": pid,
        "creation_filetime_100ns": 1000 + pid,
        "windows_session_id": 1,
        "image_path": PYTHON,
        "image_sha256": H,
        "argv_sha256": H2,
    }


def timing(
    phase: str,
    *,
    start: int = 10,
    duration: int | None = None,
    parent: int | None = None,
    prepared: int = 20,
) -> dict[str, object]:
    if duration is None:
        duration = contract.DEADLINE_DURATIONS_NS.get(phase, 100)
    if parent is None:
        parent = start + duration + 100
    return {
        "phase": phase,
        "started_monotonic_ns": start,
        "duration_ns": duration,
        "parent_deadline_monotonic_ns": parent,
        "deadline_monotonic_ns": min(start + duration, parent),
        "prepared_monotonic_ns": prepared,
    }


def phase_deadline(
    phase: str,
    *,
    start: int = 10,
    duration: int | None = None,
    parent: int | None = None,
) -> dict[str, object]:
    value = timing(phase, start=start, duration=duration, parent=parent)
    value.pop("prepared_monotonic_ns")
    return value


def live_freeze() -> dict[str, object]:
    implementation = identity("implementation-inventory")
    launcher_script = {"path": LIVE_WORKTREE + r"\scripts\launch_sim.ps1", "sha256": H}
    launcher = {"path": r"C:\Users\John\AIGP\AIGP_3385\FlightSim.exe", "sha256": H}
    payload = {
        "path": r"C:\Users\John\AIGP\AIGP_3385\FlightSim\Binaries\Win64\FlightSim-Win64-Shipping.exe",
        "sha256": H,
    }
    return {
        "schema": "aigp-vq2-powered-calibration-live-freeze/1",
        "task_id": contract.TASK_ID,
        "freeze_id": (
            "vq2-package2-powered-calibration-f00-a01-live-freeze-recovery-02"
        ),
        "candidate": {
            "commit": COMMIT,
            "code_sha256": contract.canonical_object_sha256(implementation),
            "live_worktree": LIVE_WORKTREE,
            "detached_head_required": True,
            "clean_tracked_untracked_ignored_required": True,
            "implementation_inventory": implementation,
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
                **identity("target"),
            },
            "capture_authorization": {
                "schema": "aigp-vq2-simulation-capture-authorization/1",
                **identity("authorization"),
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
                "sha256": H,
            },
            "development_test_lock": identity("development-lock"),
            "environment_inventory": identity("environment-inventory"),
            "import_inventory": identity("import-inventory"),
        },
        "simulator": {
            "build": 3385,
            "mode": "Training",
            "launcher_script": launcher_script,
            "launcher": launcher,
            "payload": payload,
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
                launcher_script["path"],
                "-SimulatorPath",
                launcher["path"],
                "-TaskName",
                "AIGP-P2-F00-A01-Launch",
                "-StartupTimeoutSeconds",
                "25",
            ],
            "launcher_environment_sha256": H,
            "child_cwd": LIVE_WORKTREE,
            "cleanup_cwd": LIVE_WORKTREE,
        },
        "paths": contract.frozen_paths(),
        "deadline_durations_ns": dict(contract.DEADLINE_DURATIONS_NS),
    }


def attempt() -> dict[str, object]:
    freeze = live_freeze()
    started = 100
    live_deadline = started + 300_000_000_000
    frozen_paths = contract.frozen_paths()
    wrapper = process(10)
    child_argv = [
        PYTHON, "-E", "-s", "-B", "-m", "scripts.aigp_vq2_run",
        "--stage", "calibration-excite", "--powered-attempt-envelope", frozen_paths["attempt_envelope"],
        "--wrapper-process", f"{wrapper['pid']}:{wrapper['creation_filetime_100ns']}",
        "--powered-process-authority", frozen_paths["child_authority"],
        "--attempt-capability-handle", "41", "--parent-liveness-handle", "42",
        "--record", frozen_paths["legacy_record"], "--replay-bundle", frozen_paths["replay_bundle"],
        "--cleanup-certificate", frozen_paths["child_cleanup_certificate"], "--recording-approved",
    ]
    cleanup_argv = [
        PYTHON, "-E", "-s", "-B", "-m", "scripts.aigp_vq2_powered_cleanup",
        "--powered-attempt-envelope", frozen_paths["attempt_envelope"],
        "--wrapper-process", f"{wrapper['pid']}:{wrapper['creation_filetime_100ns']}",
        "--powered-process-authority", frozen_paths["cleanup_authority"],
        "--cleanup-capability-handle", "43", "--parent-liveness-handle", "44",
        "--cleanup-certificate", frozen_paths["fallback_cleanup_certificate"],
    ]
    context = {
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "created_at_utc": UTC,
        "host": {
            "host_clock_id": contract.HOST_CLOCK_ID,
            "host_boot_id_sha256": H,
            "qpc_frequency_hz": 10_000_000,
        },
        "live_freeze": {
            "path": contract.frozen_paths()["live_freeze"],
            "sha256": contract.canonical_file_sha256(freeze),
        },
        "candidate_commit": COMMIT,
        "target_config": {
            "path": freeze["inputs"]["target_config"]["path"],
            "sha256": freeze["inputs"]["target_config"]["sha256"],
        },
        "capture_authorization": {
            "path": freeze["inputs"]["capture_authorization"]["path"],
            "sha256": freeze["inputs"]["capture_authorization"]["sha256"],
        },
        "excitation_plan": {
            "path": freeze["inputs"]["excitation_plan"]["path"],
            "sha256": freeze["inputs"]["excitation_plan"]["sha256"],
            "plan_id": contract.EXCITATION_PLAN_ID,
        },
        "wrapper_process": wrapper,
        "paths": frozen_paths,
        "child_argv": child_argv,
        "cleanup_argv": cleanup_argv,
        "deadline_durations_ns": dict(contract.DEADLINE_DURATIONS_NS),
        "wrapper_absolute_deadlines": {
            "started_monotonic_ns": started,
            "live_contact_deadline_monotonic_ns": live_deadline,
            "total_deadline_monotonic_ns": started + 390_000_000_000,
        },
        "prepublication_timing": {
            "wrapper_started_monotonic_ns": started,
            "offline_precheck": {
                **phase_deadline(
                    "offline_precheck",
                    start=started,
                    duration=10_000_000_000,
                    parent=live_deadline,
                ),
                "completed_monotonic_ns": started + 1,
                "outcome": "completed",
            },
            "attempt_publish": phase_deadline(
                "attempt_publish",
                start=started + 2,
                duration=2_000_000_000,
                parent=live_deadline,
            ),
        },
    }
    return {
        "schema": "aigp-vq2-powered-calibration-attempt/1",
        "context": context,
        "context_sha256": contract.canonical_object_sha256(context),
        "capabilities": {
            "algorithm": "sha256-domain-separated-context-v1",
            "lease_owner_sha256": H,
            "child_sha256": H2,
            "cleanup_sha256": H3,
        },
    }


def ingress(message_type: str, *, source: int | None = None, generation: int = 1, sequence: int = 1) -> dict[str, object]:
    unit = {
        "HEARTBEAT": None,
        "RACE_STATUS": "ms",
        "HIGHRES_IMU": "us",
        "ACTUATOR_OUTPUT_STATUS": "us",
    }[message_type]
    return {
        "schema": "aigp-vq2-mavlink-ingress/1",
        "stream_id": "vq2-mavlink-udp-14550",
        "generation": generation,
        "sequence": sequence,
        "message_type": message_type,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "received_monotonic_ns": 1000 + sequence,
        "source_time_value": source,
        "source_time_unit": unit,
    }


def heartbeat(
    *,
    generation: int = 1,
    sequence: int = 1,
    received: int | None = None,
    base_mode: int = 0,
) -> dict[str, object]:
    value = ingress("HEARTBEAT", generation=generation, sequence=sequence)
    if received is not None:
        value["received_monotonic_ns"] = received
    return {
        "schema": "aigp-vq2-received-heartbeat/1",
        "ingress": value,
        "heartbeat": {"base_mode": base_mode, "custom_mode": 0},
    }


def race(
    *,
    value: int = 20,
    generation: int = 1,
    sequence: int = 2,
    received: int | None = None,
) -> dict[str, object]:
    value_ingress = ingress(
        "RACE_STATUS",
        source=value,
        generation=generation,
        sequence=sequence,
    )
    if received is not None:
        value_ingress["received_monotonic_ns"] = received
    return {
        "schema": "aigp-vq2-received-race-status/1",
        "ingress": value_ingress,
        "race_status": {
            "sim_boot_time_ms": value,
            "race_start_boot_time_ms": 0,
            "race_finish_time_ns": 0,
            "active_gate_index": 0,
            "last_gate_race_time": 0,
        },
    }


def imu(
    *,
    value: int = 30,
    generation: int = 1,
    sequence: int = 3,
    received: int | None = None,
) -> dict[str, object]:
    value_ingress = ingress(
        "HIGHRES_IMU",
        source=value,
        generation=generation,
        sequence=sequence,
    )
    if received is not None:
        value_ingress["received_monotonic_ns"] = received
    return {
        "schema": "aigp-vq2-received-imu/1",
        "ingress": value_ingress,
        "imu": {
            "timestamp_us": value,
            "accel": [0.0, 0.0, 9.8],
            "gyro": [0.0, 0.0, 0.0],
            "mag": None,
        },
    }


def actuator(*, value: int = 40, sequence: int = 4) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-received-actuator-output-status/1",
        "ingress": ingress("ACTUATOR_OUTPUT_STATUS", source=value, sequence=sequence),
        "actuator_output_status": {
            "time_usec": value,
            "active": 0,
            "actuator": [0.0] * 32,
        },
    }


def frame_timing() -> dict[str, object]:
    return {
        "schema": "aigp-vq2-frame-timing/1",
        "identity": {
            "schema": "aigp-vq2-frame-identity/1",
            "stream_id": "vq2-camera",
            "generation": 1,
            "frame_id": 5,
        },
        "camera_source_time_ns": 50,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "publication_sequence": 1,
        "first_unique_packet_monotonic_ns": 100,
        "final_unique_packet_monotonic_ns": 101,
        "reassembly_complete_monotonic_ns": 102,
        "decode_start_monotonic_ns": 103,
        "decode_end_monotonic_ns": 104,
        "publish_monotonic_ns": 105,
    }


def watchdogs(*, cleanup: bool = False) -> dict[str, object]:
    if cleanup:
        return {
            "checked_monotonic_ns": 100,
            "heartbeat_age_ns": None,
            "imu_age_ns": None,
            "imu_advance_age_ns": None,
            "race_age_ns": None,
            "race_advance_age_ns": None,
            "actuator_age_ns": None,
            "vision_age_ns": None,
            "estimator_healthy": None,
            "target_consecutive": None,
            "target_center_px": None,
            "target_bbox_px": None,
            "target_bbox_area_px": None,
            "initial_target_bbox_area_px": None,
            "roll_excursion_rad": None,
            "pitch_excursion_rad": None,
            "collision_count": None,
            "gate_index": None,
            "result": "cleanup_authorized",
            "failure_codes": [],
        }
    return {
        "checked_monotonic_ns": 100,
        "heartbeat_age_ns": 1,
        "imu_age_ns": 1,
        "imu_advance_age_ns": 1,
        "race_age_ns": 1,
        "race_advance_age_ns": 1,
        "actuator_age_ns": 1,
        "vision_age_ns": 1,
        "estimator_healthy": True,
        "target_consecutive": 3,
        "target_center_px": [320.0, 180.0],
        "target_bbox_px": [300.0, 160.0, 40.0, 40.0],
        "target_bbox_area_px": 1600.0,
        "initial_target_bbox_area_px": 1600.0,
        "roll_excursion_rad": 0.0,
        "pitch_excursion_rad": 0.0,
        "collision_count": 0,
        "gate_index": 0,
        "result": "pass",
        "failure_codes": [],
    }


def generated(*, event_sequence: int = 1) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-calibration-command-generated/1",
        "attempt_id": contract.ATTEMPT_ID,
        "session_id": contract.SESSION_ID,
        "candidate_commit": COMMIT,
        "attempt_context_sha256": H,
        "event_sequence": event_sequence,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "generated_monotonic_ns": 200,
        "reset_epoch": {
            "ingress_generation": 1,
            "race_anchor_boot_ms": 10,
            "imu_anchor_usec": 10,
        },
        "plan": {
            "plan_id": contract.EXCITATION_PLAN_ID,
            "sha256": contract.EXCITATION_PLAN_SHA256,
        },
        "scope": "excitation",
        "command_id": "excitation/000",
        "absolute_tick": 0,
        "segment_id": "dwell-0",
        "slot": {
            "release_monotonic_ns": 1_000_000_000,
            "end_monotonic_ns": 1_020_000_000,
            "powered_expiry_monotonic_ns": 6_000_000_000,
        },
        "command": contract.excitation_command_for_tick(0),
        "source": {
            "frame": {
                "stream_id": "vq2-camera",
                "generation": 1,
                "frame_id": 5,
                "sim_time_ns": 50,
                "timing": frame_timing(),
                "width": 640,
                "height": 360,
            },
            "imu": imu(),
            "race": race(),
            "heartbeat": heartbeat(),
            "actuator": actuator(),
        },
        "watchdogs": watchdogs(),
    }


def attitude_receipt(
    *,
    outcome: str = "returned",
    sequence: int = 1,
    generation: int = 1,
    call_start: int = 201,
    thrust: float = 0.235,
) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-attitude-target-outbound/1",
        "stream_id": "vq2-mavlink-udp-14550",
        "reset_generation": generation,
        "outbound_sequence": sequence,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "call_start_monotonic_ns": call_start,
        "call_end_monotonic_ns": call_start + 1,
        "api": "send_attitude_rate",
        "outcome": outcome,
        "error_type": None if outcome == "returned" else "RuntimeError",
        "wire": {
            "time_boot_ms": 1,
            "target_system": 1,
            "target_component": 1,
            "type_mask": 128,
            "q_wxyz": [1.0, 0.0, 0.0, 0.0],
            "body_rates_rad_s": [0.0, 0.0, 0.0],
            "thrust": thrust,
        },
    }


def nonattitude_receipt(
    category: str,
    *,
    outcome: str = "returned",
    sequence: int = 1,
    generation: int = 1,
    call_start: int = 201,
) -> dict[str, object]:
    if category in {"arm", "disarm", "sim_reset"}:
        api = "command_long_send"
        wire = {
            "target_system": 42,
            "target_component": 99,
            "command": 31_000 if category == "sim_reset" else 400,
            "confirmation": 0,
            "params": [
                1.0 if category == "arm" else 0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    elif category == "timesync":
        api = "timesync_send"
        wire = {"tc1": 0, "ts1": 1_721_500_000_000_000_000}
    elif category == "gcs_heartbeat":
        api = "heartbeat_send"
        wire = {
            "type": 6,
            "autopilot": 8,
            "base_mode": 0,
            "custom_mode": 0,
            "system_status": 4,
        }
    else:
        raise AssertionError(f"unsupported test category {category!r}")
    return {
        "schema": "aigp-vq2-nonattitude-outbound/1",
        "stream_id": "vq2-mavlink-udp-14550",
        "reset_generation": generation,
        "outbound_sequence": sequence,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "call_start_monotonic_ns": call_start,
        "call_end_monotonic_ns": call_start + 1,
        "category": category,
        "api": api,
        "outcome": outcome,
        "error_type": None if outcome == "returned" else "RuntimeError",
        "wire": wire,
    }


def sent(source: dict[str, object]) -> dict[str, object]:
    value = {
        key: copy.deepcopy(item)
        for key, item in source.items()
        if key not in {"schema", "generated_monotonic_ns", "event_sequence"}
    }
    value.update(
        {
            "schema": "aigp-vq2-calibration-command-sent/1",
            "event_sequence": 2,
            "sent_monotonic_ns": 203,
            "generated_event_sequence": source["event_sequence"],
            "generation_sha256": contract.canonical_object_sha256(source),
            "transport": {
                "receipt": attitude_receipt(),
                "audit_count_before": 0,
                "audit_count_after": 1,
            },
        }
    )
    return value


def terminal_cleanup() -> dict[str, object]:
    return {
        "child_certificate_sha256": H,
        "fallback_used": False,
        "fallback_certificate_sha256": None,
        "child_exit": "proved",
        "fallback": "not_required",
        "processes": "exited",
        "transport": "closed",
        "ports": "free",
        "lease": "released",
        "simulator_topology": "unchanged",
        "simulator_responsive": "yes",
        "scheduled_task": "absent",
    }


def failed_cleanup_certificate() -> dict[str, object]:
    return {
        "schema": "aigp-vq2-powered-cleanup-certificate/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "producer_role": "powered_child",
        "cleanup_epoch": "child-cleanup-0",
        "authority": {
            "process_authority": identity("child-authority"),
            "attempt_context_sha256": H,
            "attempt_envelope_sha256": H,
            "producer": process(20),
        },
        "trigger": "stage_abort",
        "started_monotonic_ns": 1,
        "deadline_monotonic_ns": 100,
        "completed_monotonic_ns": 2,
        "parent_state": {
            "mode": "live_delegation",
            "wrapper_process": process(10),
            "observed_monotonic_ns": 1,
            "takeover_completed_monotonic_ns": None,
            "takeover_lease_record_sha256": None,
        },
        "lease": {
            "owner_role": "wrapper",
            "generation": 0,
            "record_sha256": H,
            "authority_valid": False,
        },
        "phase_deadlines": [],
        "endpoints": {
            "mavlink": {
                "state": "not_opened",
                "bind": None,
                "frozen_peer": None,
                "rejected_source_count": 0,
            },
            "camera": {
                "state": "not_opened",
                "bind": None,
                "frozen_peer": None,
                "rejected_source_count": 0,
            },
        },
        "outbound_receipts": [],
        "zero_command": {
            "state": "not_required",
            "required": False,
            "requested": None,
            "generated": None,
            "terminal": None,
            "outbound_receipt": None,
        },
        "disarm": {
            "state": "not_attempted",
            "request_monotonic_ns": None,
            "receipt": None,
            "heartbeat_before": None,
            "heartbeat_after": None,
            "newer_confirmed": False,
        },
        "reset": {
            "state": "not_attempted",
            "request_monotonic_ns": None,
            "receipt": None,
            "boundary": None,
            "baseline": None,
            "clean_epoch": None,
            "advancing_race": [],
            "advancing_imu": [],
            "rollback_and_advance_confirmed": False,
        },
        "collisions": {"observations": [], "invalidating_occurrence_count": 0},
        "final_state": {
            "state": "unobserved",
            "heartbeat": None,
            "disarmed": None,
            "reset_epoch": None,
            "last_race": None,
            "last_imu": None,
        },
        "transport": {
            "production_guard_latched": False,
            "cleanup_guard_closed": False,
            "vision_closed": False,
            "mavlink_socket_closed": False,
            "receiver_joined": False,
            "announcer_joined": False,
            "owned_handles_closed": False,
        },
        "outcome": "failed",
        "failure_codes": ["authority_invalid"],
        "collection_invalidating_codes": ["camera_missing"],
    }


def cleanup_reset_boundary(old_generation: int, *, boundary_monotonic_ns: int = 650) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-calibration-reset-boundary/1",
        "old_generation": old_generation,
        "new_generation": old_generation + 1,
        "boundary_monotonic_ns": boundary_monotonic_ns,
        "observations": [],
        "collisions": [],
        "ingress_stats": {
            "generation": old_generation,
            "next_sequence": 14,
            "highres_imu_received": 4,
            "heartbeat_received": 5,
            "race_status_received": 3,
            "actuator_received": 2,
            "dropped": 0,
            "high_watermark": 4,
            "imu_capacity": 512,
            "other_capacity": 512,
            "imu_dropped": 0,
            "other_dropped": 0,
            "imu_high_watermark": 1,
            "other_high_watermark": 3,
            "buffered_imu": 0,
            "buffered_other": 0,
        },
        "collision_stats": {
            "generation": old_generation,
            "handled": 0,
            "dropped": 0,
            "high_watermark": 0,
            "capacity": 512,
            "buffered": 0,
        },
    }


def cleanup_endpoint(*, camera: bool, owner: dict[str, object]) -> dict[str, object]:
    host, port = ("0.0.0.0", 5600) if camera else ("127.0.0.1", 14550)
    return {
        "state": "closed_with_peer",
        "bind": {
            "role": "camera" if camera else "mavlink",
            "family": "AF_INET",
            "requested": {"host": host, "port": port},
            "actual": {"host": host, "port": port},
            "socket_policy": "ipv4-exclusive-address-use",
            "owner_process": owner,
        },
        "frozen_peer": {"host": "127.0.0.1", "port": 40_000 + (1 if camera else 0)},
        "rejected_source_count": 0,
    }


def cleanup_zero_evidence(receipt: dict[str, object]) -> dict[str, object]:
    command = {
        "roll_rate_rad_s": 0.0,
        "pitch_rate_rad_s": 0.0,
        "yaw_rate_rad_s": 0.0,
        "thrust": 0.0,
    }
    common = {
        "attempt_id": contract.ATTEMPT_ID,
        "session_id": contract.SESSION_ID,
        "candidate_commit": COMMIT,
        "attempt_context_sha256": H,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "reset_epoch": None,
        "plan": None,
        "scope": "cleanup_zero",
        "command_id": "cleanup/zero/0",
        "absolute_tick": None,
        "segment_id": None,
        "slot": None,
        "command": command,
        "source": {
            "frame": None,
            "imu": None,
            "race": None,
            "heartbeat": None,
            "actuator": None,
        },
        "watchdogs": watchdogs(cleanup=True),
    }
    generated = {
        **copy.deepcopy(common),
        "schema": "aigp-vq2-calibration-command-generated/1",
        "event_sequence": 1,
        "generated_monotonic_ns": receipt["call_start_monotonic_ns"],
    }
    terminal = {
        **copy.deepcopy(common),
        "schema": "aigp-vq2-calibration-command-sent/1",
        "event_sequence": 2,
        "sent_monotonic_ns": receipt["call_end_monotonic_ns"],
        "generated_event_sequence": 1,
        "generation_sha256": contract.canonical_object_sha256(generated),
        "transport": {
            "receipt": copy.deepcopy(receipt),
            "audit_count_before": 0,
            "audit_count_after": 1,
        },
    }
    return {
        "state": "returned",
        "required": True,
        "requested": command,
        "generated": generated,
        "terminal": terminal,
        "outbound_receipt": copy.deepcopy(receipt),
    }


def proved_cleanup_certificate(role: str) -> dict[str, object]:
    assert role in {"powered_child", "cleanup_fallback"}
    old_generation = 2 if role == "powered_child" else 0
    new_generation = old_generation + 1
    producer = process(20 if role == "powered_child" else 30)
    zero_receipt = attitude_receipt(
        sequence=0,
        generation=old_generation,
        call_start=100,
        thrust=0.0,
    )
    disarm_receipt = nonattitude_receipt(
        "disarm",
        sequence=1,
        generation=old_generation,
        call_start=310,
    )
    reset_receipt = nonattitude_receipt(
        "sim_reset",
        sequence=2,
        generation=new_generation,
        call_start=700,
    )
    epoch = {
        "ingress_generation": new_generation,
        "race_anchor_boot_ms": 100,
        "imu_anchor_usec": 100_000,
    }
    advancing_race = [
        race(value=101, generation=new_generation, sequence=3, received=830),
        race(value=102, generation=new_generation, sequence=5, received=850),
    ]
    advancing_imu = [
        imu(value=100_001, generation=new_generation, sequence=4, received=840),
        imu(value=100_002, generation=new_generation, sequence=6, received=860),
    ]
    camera = (
        cleanup_endpoint(camera=True, owner=producer)
        if role == "powered_child"
        else None
    )
    return {
        "schema": "aigp-vq2-powered-cleanup-certificate/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "producer_role": role,
        "cleanup_epoch": (
            "child-cleanup-0" if role == "powered_child" else "fallback-cleanup-0"
        ),
        "authority": {
            "process_authority": identity(
                "child-authority" if role == "powered_child" else "cleanup-authority"
            ),
            "attempt_context_sha256": H,
            "attempt_envelope_sha256": H,
            "producer": producer,
        },
        "trigger": "normal_completion" if role == "powered_child" else "wrapper_fallback",
        "started_monotonic_ns": 1,
        "deadline_monotonic_ns": 20_000,
        "completed_monotonic_ns": 10_000,
        "parent_state": {
            "mode": "live_delegation",
            "wrapper_process": process(10),
            "observed_monotonic_ns": 1,
            "takeover_completed_monotonic_ns": None,
            "takeover_lease_record_sha256": None,
        },
        "lease": {
            "owner_role": "wrapper",
            "generation": 0,
            "record_sha256": H,
            "authority_valid": True,
        },
        "phase_deadlines": [],
        "endpoints": {
            "mavlink": cleanup_endpoint(camera=False, owner=producer),
            "camera": camera,
        },
        "outbound_receipts": [zero_receipt, disarm_receipt, reset_receipt],
        "zero_command": cleanup_zero_evidence(zero_receipt),
        "disarm": {
            "state": "confirmed",
            "request_monotonic_ns": 300,
            "receipt": disarm_receipt,
            "heartbeat_before": heartbeat(
                generation=old_generation,
                sequence=10,
                received=200,
                base_mode=128,
            ),
            "heartbeat_after": heartbeat(
                generation=old_generation,
                sequence=11,
                received=400,
            ),
            "newer_confirmed": True,
        },
        "reset": {
            "state": "confirmed",
            "request_monotonic_ns": 600,
            "receipt": reset_receipt,
            "boundary": cleanup_reset_boundary(old_generation),
            "baseline": {
                "race": race(
                    value=1_000,
                    generation=old_generation,
                    sequence=12,
                    received=500,
                ),
                "imu": imu(
                    value=500_000,
                    generation=old_generation,
                    sequence=13,
                    received=510,
                ),
            },
            "clean_epoch": epoch,
            "advancing_race": advancing_race,
            "advancing_imu": advancing_imu,
            "rollback_and_advance_confirmed": True,
        },
        "collisions": {"observations": [], "invalidating_occurrence_count": 0},
        "final_state": {
            "state": "confirmed",
            "heartbeat": heartbeat(
                generation=new_generation,
                sequence=2,
                received=820,
            ),
            "disarmed": True,
            "reset_epoch": copy.deepcopy(epoch),
            "last_race": copy.deepcopy(advancing_race[-1]),
            "last_imu": copy.deepcopy(advancing_imu[-1]),
        },
        "transport": {
            "production_guard_latched": True,
            "cleanup_guard_closed": True,
            "vision_closed": True,
            "mavlink_socket_closed": True,
            "receiver_joined": True,
            "announcer_joined": True,
            "owned_handles_closed": True,
        },
        "outcome": "proved",
        "failure_codes": [],
        "collection_invalidating_codes": [],
    }


def mutate_cleanup_lineage_seam(certificate: dict[str, object], seam: str) -> None:
    disarm = certificate["disarm"]
    reset = certificate["reset"]
    final = certificate["final_state"]

    def mutate_receipt(category: str, generation: int) -> None:
        target = disarm if category == "disarm" else reset
        target["receipt"]["reset_generation"] = generation
        for receipt in certificate["outbound_receipts"]:
            if receipt.get("category") == category:
                receipt["reset_generation"] = generation
                return
        raise AssertionError(f"missing {category} receipt")

    if seam == "disarm_before_generation":
        disarm["heartbeat_before"]["ingress"]["generation"] += 1
    elif seam == "disarm_before_after_request":
        disarm["heartbeat_before"]["ingress"]["received_monotonic_ns"] = (
            disarm["request_monotonic_ns"] + 1
        )
    elif seam == "disarm_after_generation":
        disarm["heartbeat_after"]["ingress"]["generation"] += 1
    elif seam == "disarm_after_reset_request":
        disarm["heartbeat_after"]["ingress"]["received_monotonic_ns"] = (
            reset["request_monotonic_ns"] + 1
        )
    elif seam == "disarm_confirmation_before_receipt_start":
        disarm["heartbeat_after"]["ingress"]["received_monotonic_ns"] = (
            disarm["receipt"]["call_start_monotonic_ns"] - 1
        )
    elif seam == "disarm_receipt_ends_after_reset_request":
        value = reset["request_monotonic_ns"] + 1
        disarm["receipt"]["call_end_monotonic_ns"] = value
        for receipt in certificate["outbound_receipts"]:
            if receipt.get("category") == "disarm":
                receipt["call_end_monotonic_ns"] = value
                break
    elif seam == "disarm_receipt_generation":
        mutate_receipt(
            "disarm",
            disarm["receipt"]["reset_generation"] + 1,
        )
    elif seam == "reset_baseline_race_generation":
        reset["baseline"]["race"]["ingress"]["generation"] += 1
    elif seam == "reset_baseline_imu_generation":
        reset["baseline"]["imu"]["ingress"]["generation"] += 1
    elif seam == "reset_receipt_generation":
        mutate_receipt("sim_reset", reset["boundary"]["old_generation"])
    elif seam == "boundary_generation_pair":
        reset["boundary"]["old_generation"] += 10
        reset["boundary"]["new_generation"] += 10
        reset["boundary"]["ingress_stats"]["generation"] += 10
        reset["boundary"]["collision_stats"]["generation"] += 10
    elif seam == "boundary_before_request":
        reset["boundary"]["boundary_monotonic_ns"] = (
            reset["request_monotonic_ns"] - 1
        )
    elif seam == "boundary_after_receipt":
        reset["boundary"]["boundary_monotonic_ns"] = (
            reset["receipt"]["call_end_monotonic_ns"] + 1
        )
    elif seam == "boundary_foreign_stream":
        observation = heartbeat(
            generation=reset["boundary"]["old_generation"],
            sequence=9,
            received=190,
        )
        observation["ingress"]["stream_id"] = "foreign-stream"
        reset["boundary"]["observations"] = [observation]
    elif seam == "boundary_observation_after_boundary":
        reset["boundary"]["observations"] = [
            heartbeat(
                generation=reset["boundary"]["old_generation"],
                sequence=9,
                received=reset["boundary"]["boundary_monotonic_ns"] + 1,
            )
        ]
    elif seam == "boundary_collision_omitted_from_cleanup":
        reset["boundary"]["collisions"] = [
            {
                "schema": "aigp-vq2-runner-collision-observation/1",
                "reset_generation": reset["boundary"]["old_generation"],
                "observation_sequence": 0,
                "host_clock_id": contract.HOST_CLOCK_ID,
                "observed_monotonic_ns": reset["boundary"][
                    "boundary_monotonic_ns"
                ],
                "phase": "cleanup",
                "disposition": "reset_boundary_discard",
                "boundary": "runner_drain_not_receiver_receipt",
                "collision": {"id": 1, "threat_level": 1, "impulse": 1.0},
            }
        ]
    elif seam == "baseline_race_after_boundary":
        reset["baseline"]["race"]["ingress"]["received_monotonic_ns"] = (
            reset["boundary"]["boundary_monotonic_ns"] + 1
        )
    elif seam == "baseline_imu_after_boundary":
        reset["baseline"]["imu"]["ingress"]["received_monotonic_ns"] = (
            reset["boundary"]["boundary_monotonic_ns"] + 1
        )
    elif seam == "clean_epoch_generation":
        reset["clean_epoch"]["ingress_generation"] += 1
    elif seam == "reset_foreign_stream_island":
        reset["receipt"]["stream_id"] = "foreign-stream"
        for receipt in certificate["outbound_receipts"]:
            if receipt.get("category") == "sim_reset":
                receipt["stream_id"] = "foreign-stream"
        for name in ("race", "imu"):
            reset["baseline"][name]["ingress"]["stream_id"] = "foreign-stream"
        for name in ("advancing_race", "advancing_imu"):
            for observation in reset[name]:
                observation["ingress"]["stream_id"] = "foreign-stream"
        for name in ("heartbeat", "last_race", "last_imu"):
            final[name]["ingress"]["stream_id"] = "foreign-stream"
    elif seam in {"zero_foreign_stream", "zero_foreign_generation"}:
        zero = certificate["zero_command"]
        field = "stream_id" if seam == "zero_foreign_stream" else "reset_generation"
        value = "foreign-stream" if field == "stream_id" else 999
        zero["outbound_receipt"][field] = value
        zero["terminal"]["transport"]["receipt"][field] = value
        for receipt in certificate["outbound_receipts"]:
            if receipt.get("schema") == "aigp-vq2-attitude-target-outbound/1":
                receipt[field] = value
                break
    elif seam == "clean_epoch_race_rollback":
        value = reset["baseline"]["race"]["race_status"]["sim_boot_time_ms"]
        reset["clean_epoch"]["race_anchor_boot_ms"] = value
        final["reset_epoch"]["race_anchor_boot_ms"] = value
    elif seam == "clean_epoch_imu_rollback":
        value = reset["baseline"]["imu"]["imu"]["timestamp_us"]
        reset["clean_epoch"]["imu_anchor_usec"] = value
        final["reset_epoch"]["imu_anchor_usec"] = value
    elif seam == "advancing_race_generation":
        reset["advancing_race"][0]["ingress"]["generation"] += 1
    elif seam == "advancing_race_before_boundary":
        reset["advancing_race"][0]["ingress"]["received_monotonic_ns"] = (
            reset["request_monotonic_ns"]
            + (reset["boundary"]["boundary_monotonic_ns"] - reset["request_monotonic_ns"]) // 2
        )
    elif seam == "reset_receipt_starts_after_proof":
        value = max(
            observation["ingress"]["received_monotonic_ns"]
            for name in ("advancing_race", "advancing_imu")
            for observation in reset[name]
        ) + 1
        reset["receipt"]["call_start_monotonic_ns"] = value
        reset["receipt"]["call_end_monotonic_ns"] = value + 1
        for receipt in certificate["outbound_receipts"]:
            if receipt.get("category") == "sim_reset":
                receipt["call_start_monotonic_ns"] = value
                receipt["call_end_monotonic_ns"] = value + 1
                break
    elif seam == "reset_receipt_ends_after_completion":
        value = certificate["completed_monotonic_ns"] + 1
        reset["receipt"]["call_end_monotonic_ns"] = value
        for receipt in certificate["outbound_receipts"]:
            if receipt.get("category") == "sim_reset":
                receipt["call_end_monotonic_ns"] = value
                break
    elif seam == "advancing_imu_generation":
        reset["advancing_imu"][0]["ingress"]["generation"] += 1
    elif seam == "advancing_imu_before_boundary":
        reset["advancing_imu"][0]["ingress"]["received_monotonic_ns"] = (
            reset["request_monotonic_ns"]
            + (reset["boundary"]["boundary_monotonic_ns"] - reset["request_monotonic_ns"]) // 2
        )
    elif seam == "old_epoch_cross_evidence_occurrence":
        reset["baseline"]["race"]["ingress"]["sequence"] = disarm[
            "heartbeat_after"
        ]["ingress"]["sequence"]
    elif seam == "advancing_race_occurrence_order":
        reset["advancing_race"][1]["ingress"]["sequence"] = reset[
            "advancing_race"
        ][0]["ingress"]["sequence"]
    elif seam == "advancing_imu_occurrence_order":
        reset["advancing_imu"][1]["ingress"]["sequence"] = reset[
            "advancing_imu"
        ][0]["ingress"]["sequence"]
    elif seam == "advancing_cross_type_occurrence":
        reset["advancing_imu"][0]["ingress"]["sequence"] = reset[
            "advancing_race"
        ][0]["ingress"]["sequence"]
    elif seam == "advancing_host_receipt_order":
        reset["advancing_race"][1]["ingress"]["received_monotonic_ns"] = (
            reset["advancing_race"][0]["ingress"]["received_monotonic_ns"] - 1
        )
    elif seam == "advancing_race_source_order":
        value = reset["advancing_race"][0]["race_status"]["sim_boot_time_ms"]
        reset["advancing_race"][1]["race_status"]["sim_boot_time_ms"] = value
        reset["advancing_race"][1]["ingress"]["source_time_value"] = value
    elif seam == "advancing_imu_source_order":
        value = reset["advancing_imu"][0]["imu"]["timestamp_us"]
        reset["advancing_imu"][1]["imu"]["timestamp_us"] = value
        reset["advancing_imu"][1]["ingress"]["source_time_value"] = value
    elif seam == "final_epoch_generation":
        final["reset_epoch"]["ingress_generation"] += 1
    elif seam == "final_epoch_anchor":
        final["reset_epoch"]["race_anchor_boot_ms"] += 1
    elif seam == "final_heartbeat_generation":
        final["heartbeat"]["ingress"]["generation"] += 1
    elif seam == "final_heartbeat_before_boundary":
        final["heartbeat"]["ingress"]["received_monotonic_ns"] = (
            reset["request_monotonic_ns"]
            + (reset["boundary"]["boundary_monotonic_ns"] - reset["request_monotonic_ns"]) // 2
        )
    elif seam == "final_heartbeat_before_reset_receipt":
        final["heartbeat"]["ingress"]["received_monotonic_ns"] = (
            reset["receipt"]["call_start_monotonic_ns"] - 1
        )
    elif seam == "final_race_generation":
        final["last_race"]["ingress"]["generation"] += 1
    elif seam == "final_imu_generation":
        final["last_imu"]["ingress"]["generation"] += 1
    elif seam == "final_heartbeat_armed":
        final["heartbeat"]["heartbeat"]["base_mode"] = 128
    elif seam == "final_same_token_different_payload":
        final["last_race"]["race_status"]["sim_boot_time_ms"] += 1
        final["last_race"]["ingress"]["source_time_value"] += 1
    elif seam == "final_race_precedes_proof":
        final["last_race"] = copy.deepcopy(reset["advancing_race"][0])
    elif seam == "final_imu_precedes_proof":
        final["last_imu"] = copy.deepcopy(reset["advancing_imu"][0])
    else:
        raise AssertionError(f"unknown cleanup lineage seam {seam!r}")


def shift_reset_generation_island(certificate: dict[str, object]) -> None:
    reset = certificate["reset"]
    final = certificate["final_state"]
    delta = 10
    reset["boundary"]["old_generation"] += delta
    reset["boundary"]["new_generation"] += delta
    reset["boundary"]["ingress_stats"]["generation"] += delta
    reset["boundary"]["collision_stats"]["generation"] += delta
    for name in ("race", "imu"):
        reset["baseline"][name]["ingress"]["generation"] += delta
    new_generation = reset["boundary"]["new_generation"]
    reset["receipt"]["reset_generation"] = new_generation
    for receipt in certificate["outbound_receipts"]:
        if receipt.get("category") == "sim_reset":
            receipt["reset_generation"] = new_generation
    reset["clean_epoch"]["ingress_generation"] = new_generation
    for name in ("advancing_race", "advancing_imu"):
        for observation in reset[name]:
            observation["ingress"]["generation"] = new_generation
    final["reset_epoch"]["ingress_generation"] = new_generation
    for name in ("heartbeat", "last_race", "last_imu"):
        final[name]["ingress"]["generation"] = new_generation


def invalid_artifact_state(*, lifecycle: str = "absent") -> dict[str, object]:
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
        "wrapper_lifecycle": lifecycle,
        "wrapper_lifecycle_sha256": H if lifecycle == "valid" else None,
        "attempt_complete": "absent",
        "attempt_complete_partial_sha256": None,
        "terminal_publication": "invalid_record",
        "forensic_bytes_preserved": True,
    }


NO_CONTACT = {
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


def invalid_record(*, unsafe: bool = False) -> dict[str, object]:
    cleanup = dict(NO_CONTACT)
    if unsafe:
        cleanup["ports"] = "unproved"
    return {
        "schema": "aigp-vq2-powered-calibration-attempt-invalid/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "invalidated_at_utc": UTC,
        "invalidated_monotonic_ns": 20,
        "publication_timing": timing("invalid_terminal_publish", prepared=20),
        "phase": "offline_precheck",
        "reason_codes": ["internal_error"],
        "reason_detail": "sanitized",
        "identity": {
            "attempt_envelope_state": "absent",
            "live_freeze_sha256": H,
            "attempt_context_sha256": None,
            "attempt_envelope_sha256": None,
            "candidate_commit": COMMIT,
            "target_config_sha256": H,
            "capture_authorization_sha256": H,
            "excitation_plan_sha256": contract.EXCITATION_PLAN_SHA256,
        },
        "artifact_state": invalid_artifact_state(),
        "cleanup_state": cleanup,
        "poison": {
            "required": unsafe,
            "path": contract.frozen_paths()["live_poison"],
            "sha256": None,
        },
    }


def test_canonical_json_contract_rejects_duplicates_bom_nonfinite_and_noncanonical():
    value = {"z": "é", "a": [1, True, None]}
    assert contract.canonical_json_bytes(value) == '{"a":[1,true,null],"z":"é"}'.encode()
    assert contract.canonical_json_file_bytes(value).endswith(b"\n")
    assert contract.parse_canonical_json_bytes(contract.canonical_json_file_bytes(value), file_form=True) == value
    for payload in (
        b'{"a":1,"a":2}',
        b"\xef\xbb\xbf{}",
        b'{"x":NaN}',
        b'{ "a":1}',
    ):
        with pytest.raises(contract.PoweredAttemptContractError):
            contract.parse_canonical_json_bytes(payload)


def test_plan_hash_derivation_tick_lookup_and_immutability_are_exact():
    plan = contract.frozen_excitation_plan()
    assert contract.canonical_object_sha256(plan) == contract.EXCITATION_PLAN_SHA256
    assert contract.canonical_file_sha256(plan) == (
        "ecaf1912a495cb91ed96fed8b61fc2ff8caa7828534fe2b7c142acf0984e500d"
    )
    assert contract.canonical_file_sha256(plan) != contract.EXCITATION_PLAN_SHA256
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_excitation_plan(
            plan,
            expected_sha256=contract.canonical_file_sha256(plan),
        )
    assert contract.validate_excitation_plan(plan) == plan
    ticks = list(contract.iter_excitation_ticks())
    assert len(ticks) == 245
    assert ticks[0]["release_offset_ns"] == 0
    assert ticks[30]["segment_id"] == "roll-positive"
    assert ticks[-1]["end_offset_ns"] == 4_900_000_000
    assert ticks[-1]["powered_expiry_offset_ns"] == 5_000_000_000
    with pytest.raises(TypeError):
        contract.FROZEN_EXCITATION_PLAN["tick_count"] = 1
    broken = copy.deepcopy(plan)
    broken["segments"][1]["first_tick"] = 31
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_excitation_plan(broken)
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.excitation_tick(True)


def test_paths_deadlines_and_capability_frames_fail_closed():
    assert contract.validate_absolute_windows_path(contract.EVIDENCE_ROOT) == contract.EVIDENCE_ROOT
    for bad in ("relative.json", r"c:\lower.json", r"C:\root\..\escape.json", r"C:\root\alias. "):
        with pytest.raises(contract.PoweredAttemptContractError):
            contract.validate_absolute_windows_path(bad)
    valid = phase_deadline("x")
    assert contract.validate_phase_deadline(valid) == valid
    valid["deadline_monotonic_ns"] += 1
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_phase_deadline(valid)
    secret = bytes(range(32))
    frame = contract.encode_capability_frame(secret)
    assert contract.decode_capability_frame(frame) == secret
    assert contract.derive_capability_sha256("aigp-vq2-powered-child/1", H, secret) == contract.derive_capability_sha256("aigp-vq2-powered-child/1", H, secret)
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.decode_capability_frame(frame + b"x")


def test_live_freeze_attempt_and_process_authority_cross_bind():
    freeze = live_freeze()
    assert freeze["freeze_id"] == (
        "vq2-package2-powered-calibration-f00-a01-live-freeze-recovery-02"
    )
    assert freeze["paths"]["live_freeze"] == (
        contract.EVIDENCE_ROOT + r"\live-freeze-F00-A01-recovery-02.json"
    )
    assert contract.validate_live_freeze(freeze) == freeze

    for stale_id in (
        "vq2-package2-powered-calibration-f00-a01-live-freeze",
        "vq2-package2-powered-calibration-f00-a01-live-freeze-recovery-01",
    ):
        predecessor_id = copy.deepcopy(freeze)
        predecessor_id["freeze_id"] = stale_id
        with pytest.raises(contract.PoweredAttemptContractError, match="freeze_id"):
            contract.validate_live_freeze(predecessor_id)

    for stale_path in (
        contract.EVIDENCE_ROOT + r"\live-freeze-F00-A01.json",
        contract.EVIDENCE_ROOT + r"\live-freeze-F00-A01-recovery-01.json",
    ):
        predecessor_path = copy.deepcopy(freeze)
        predecessor_path["paths"]["live_freeze"] = stale_path
        with pytest.raises(
            contract.PoweredAttemptContractError,
            match=r"paths\.live_freeze",
        ):
            contract.validate_live_freeze(predecessor_path)

    envelope = attempt()
    assert contract.validate_attempt(envelope, live_freeze=freeze) == envelope
    anchor = 1000
    child_argv_hash = contract.canonical_object_sha256(envelope["context"]["child_argv"])
    child_process = process(20)
    child_process["argv_sha256"] = child_argv_hash
    authority = {
        "schema": "aigp-vq2-powered-process-authority/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "role": "powered_child",
        "created_at_utc": UTC,
        "created_monotonic_ns": 500,
        "attempt_envelope_sha256": contract.canonical_file_sha256(envelope),
        "attempt_context_sha256": envelope["context_sha256"],
        "live_freeze_sha256": envelope["context"]["live_freeze"]["sha256"],
        "wrapper_process": envelope["context"]["wrapper_process"],
        "process": child_process,
        "parent_handle": {
            "value": 42,
            "process": envelope["context"]["wrapper_process"],
            "access": "synchronize_query_limited_information",
            "inherited": True,
        },
        "capability_sha256": envelope["capabilities"]["child_sha256"],
        "lease_record_sha256": H,
        "training_attestation_sha256": H,
        "simulator_process_proof_sha256": H,
        "argv_sha256": child_argv_hash,
        "job": {
            "handle_value": 100,
            "assigned_before_capability_release": True,
            "breakaway_allowed": False,
            "silent_breakaway_allowed": False,
            "kill_on_close": False,
            "process_in_job": True,
        },
        "absolute_deadlines": {
            "anchor": anchor,
            "total": anchor + 110_000_000_000,
            "prepower": anchor + 52_000_000_000,
            "powered": anchor + 57_000_000_000,
            "cleanup": anchor + 72_000_000_000,
            "replay_close": anchor + 107_000_000_000,
            "exit": anchor + 110_000_000_000,
        },
    }
    assert contract.validate_process_authority(authority, attempt=envelope) == authority
    authority["job"]["kill_on_close"] = True
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_process_authority(authority)


def test_receive_outbound_and_rich_command_records_are_strict_and_linked():
    for item in (heartbeat(), race(), imu(), actuator()):
        assert contract.validate_powered_record(item) == item
    receipt = attitude_receipt()
    assert contract.validate_attitude_target_outbound(receipt) == receipt
    source = generated()
    assert contract.validate_command_generated(source) == source
    terminal = sent(source)
    assert contract.validate_command_sent(terminal, generated=source) == terminal
    disposition = {
        "schema": "aigp-vq2-calibration-tick-disposition/1",
        "attempt_id": contract.ATTEMPT_ID,
        "session_id": contract.SESSION_ID,
        "attempt_context_sha256": H,
        "plan_id": contract.EXCITATION_PLAN_ID,
        "plan_sha256": contract.EXCITATION_PLAN_SHA256,
        "event_sequence": 3,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "recorded_monotonic_ns": 204,
        "absolute_tick": 0,
        "segment_id": "dwell-0",
        "slot": source["slot"],
        "disposition": "sent",
        "generated_event_sequence": 1,
        "terminal_event_sequence": 2,
        "reason_code": None,
    }
    assert contract.validate_tick_disposition(disposition) == disposition
    terminal["command"]["thrust"] = 0.1
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_command_sent(terminal, generated=source)
    bad_actuator = actuator()
    bad_actuator["actuator_output_status"]["actuator"][0] = float("nan")
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_received_actuator_output_status(bad_actuator)
    bad_frame_keys = generated()
    bad_frame_keys["source"]["frame"]["decoded_width"] = bad_frame_keys["source"][
        "frame"
    ].pop("width")
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_command_generated(bad_frame_keys)


@pytest.mark.parametrize(
    "category",
    ["arm", "disarm", "sim_reset", "timesync", "gcs_heartbeat"],
)
def test_nonattitude_outbound_requires_exact_production_payload(category):
    receipt = nonattitude_receipt(category)

    assert contract.validate_nonattitude_outbound(receipt) == receipt
    assert contract.validate_powered_record(receipt) == receipt


@pytest.mark.parametrize(
    ("category", "field_path", "bad_value"),
    [
        ("arm", ("wire", "command"), 401),
        ("arm", ("wire", "confirmation"), 1),
        ("arm", ("wire", "params", 0), 0.0),
        ("arm", ("wire", "params", 6), 0.01),
        ("disarm", ("wire", "command"), 31_000),
        ("disarm", ("wire", "params", 0), 1.0),
        ("sim_reset", ("wire", "command"), 400),
        ("sim_reset", ("wire", "params", 3), 1.0),
        ("sim_reset", ("wire", "params", 4), -0.0),
        ("timesync", ("wire", "tc1"), 1),
        ("timesync", ("wire", "ts1"), 0),
        ("gcs_heartbeat", ("wire", "type"), 5),
        ("gcs_heartbeat", ("wire", "autopilot"), 7),
        ("gcs_heartbeat", ("wire", "base_mode"), 1),
        ("gcs_heartbeat", ("wire", "custom_mode"), 1),
        ("gcs_heartbeat", ("wire", "system_status"), 3),
    ],
)
def test_nonattitude_outbound_rejects_payload_mutations(
    category,
    field_path,
    bad_value,
):
    receipt = nonattitude_receipt(category)
    target = receipt
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = bad_value

    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_nonattitude_outbound(receipt)


def test_command_long_outbound_keeps_valid_target_ids_variable():
    receipt = nonattitude_receipt("sim_reset")
    receipt["wire"]["target_system"] = 255
    receipt["wire"]["target_component"] = 7

    assert contract.validate_nonattitude_outbound(receipt) == receipt

    receipt["wire"]["target_system"] = 256
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_nonattitude_outbound(receipt)


def test_wrapper_lifecycle_binds_exact_ledger_file_hash():
    event = {
        "schema": "aigp-vq2-powered-wrapper-event/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "event_sequence": 0,
        "predecessor_sha256": None,
        "event": "phase_end",
        "phase": "attempt_publish",
        "observed_monotonic_ns": 100,
        "duration_ns": contract.DEADLINE_DURATIONS_NS["attempt_publish"],
        "parent_deadline_monotonic_ns": 10_000_000_000,
        "deadline_monotonic_ns": 2_000_000_100,
        "outcome": "completed",
        "reason_code": None,
        "artifacts": [],
    }
    checked = contract.validate_wrapper_event(event)
    event_hash = contract.canonical_file_sha256(checked)
    lifecycle = {
        "schema": "aigp-vq2-powered-wrapper-lifecycle/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "records": [
            {
                "event_sequence": 0,
                "path": contract.frozen_paths()["wrapper_ledger_directory"] + r"\event-000000.json",
                "sha256": event_hash,
                "event": "phase_end",
                "phase": "attempt_publish",
                "observed_monotonic_ns": 100,
                "outcome": "completed",
                "reason_code": None,
                "artifacts": [],
            }
        ],
        "final_sequence": 0,
        "final_record_sha256": event_hash,
        "live_contact_deadline_monotonic_ns": 300_000_000_100,
        "total_deadline_monotonic_ns": 390_000_000_100,
    }
    assert contract.validate_wrapper_lifecycle(lifecycle, ledger_events=[event]) == lifecycle
    lifecycle["records"][0]["path"] = contract.EVIDENCE_ROOT + r"\wrong.json"
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_wrapper_lifecycle(lifecycle)


def test_cleanup_certificate_and_process_result_share_exact_prefix_and_hash():
    certificate = failed_cleanup_certificate()
    assert contract.validate_cleanup_certificate(certificate) == certificate
    result = {
        "schema": "aigp-vq2-powered-process-result/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "producer_role": "powered_child",
        "process_authority_sha256": H,
        "started_monotonic_ns": 1,
        "completed_monotonic_ns": 3,
        "outcome": "failed",
        "reason_codes": ["cleanup_unconfirmed"],
        "phase_deadlines": [],
        "cleanup_certificate": {
            "path": contract.frozen_paths()["child_cleanup_certificate"],
            "state": "published",
            "sha256": contract.canonical_file_sha256(certificate),
        },
        "outbound_audit": {name: 0 for name in (
            "timesync", "gcs_heartbeat", "sim_reset", "arm", "disarm",
            "attitude_target", "position_target", "other_command", "receipt_count",
            "receipt_returned", "receipt_raised", "receipt_dropped", "receipt_buffered",
        )},
        "artifacts": {
            "legacy_record": {
                "path": contract.frozen_paths()["legacy_record"],
                "state": "absent",
                "sha256": None,
            },
            "replay_bundle": {
                "path": contract.frozen_paths()["replay_bundle"],
                "state": "absent",
                "dataset_hash": None,
                "manifest_sha256": None,
                "records_sha256": None,
            },
        },
    }
    assert contract.validate_process_result(result, cleanup_certificate=certificate) == result
    result["cleanup_certificate"]["sha256"] = H2
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_process_result(result, cleanup_certificate=certificate)


def test_reset_boundary_rejects_facts_observed_after_atomic_boundary():
    boundary = cleanup_reset_boundary(0)
    boundary["observations"] = [
        heartbeat(
            generation=0,
            sequence=0,
            received=boundary["boundary_monotonic_ns"] + 1,
        )
    ]
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_reset_boundary(boundary)

    boundary = cleanup_reset_boundary(0)
    boundary["collisions"] = [
        {
            "schema": "aigp-vq2-runner-collision-observation/1",
            "reset_generation": 0,
            "observation_sequence": 0,
            "host_clock_id": contract.HOST_CLOCK_ID,
            "observed_monotonic_ns": boundary["boundary_monotonic_ns"] + 1,
            "phase": "cleanup",
            "disposition": "reset_boundary_discard",
            "boundary": "runner_drain_not_receiver_receipt",
            "collision": {"id": 1, "threat_level": 1, "impulse": 1.0},
        }
    ]
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_reset_boundary(boundary)


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
def test_proved_cleanup_certificate_cross_binds_exact_generation_lineage(role):
    certificate = proved_cleanup_certificate(role)

    assert contract.validate_cleanup_certificate(certificate) == certificate


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
def test_cleanup_certificate_rejects_internally_consistent_wrong_generation_island(role):
    certificate = proved_cleanup_certificate(role)
    shift_reset_generation_island(certificate)

    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
@pytest.mark.parametrize(
    "seam",
    [
        "disarm_before_generation",
        "disarm_before_after_request",
        "disarm_after_generation",
        "disarm_after_reset_request",
        "disarm_confirmation_before_receipt_start",
        "disarm_receipt_ends_after_reset_request",
        "disarm_receipt_generation",
        "reset_baseline_race_generation",
        "reset_baseline_imu_generation",
        "reset_receipt_generation",
        "boundary_generation_pair",
        "boundary_before_request",
        "boundary_after_receipt",
        "boundary_foreign_stream",
        "boundary_observation_after_boundary",
        "boundary_collision_omitted_from_cleanup",
        "baseline_race_after_boundary",
        "baseline_imu_after_boundary",
        "clean_epoch_generation",
        "reset_foreign_stream_island",
        "zero_foreign_stream",
        "zero_foreign_generation",
        "clean_epoch_race_rollback",
        "clean_epoch_imu_rollback",
        "advancing_race_generation",
        "advancing_race_before_boundary",
        "reset_receipt_starts_after_proof",
        "reset_receipt_ends_after_completion",
        "advancing_imu_generation",
        "advancing_imu_before_boundary",
        "old_epoch_cross_evidence_occurrence",
        "advancing_race_occurrence_order",
        "advancing_imu_occurrence_order",
        "advancing_cross_type_occurrence",
        "advancing_host_receipt_order",
        "advancing_race_source_order",
        "advancing_imu_source_order",
        "final_epoch_generation",
        "final_epoch_anchor",
        "final_heartbeat_generation",
        "final_heartbeat_before_boundary",
        "final_heartbeat_before_reset_receipt",
        "final_race_generation",
        "final_imu_generation",
        "final_heartbeat_armed",
        "final_same_token_different_payload",
        "final_race_precedes_proof",
        "final_imu_precedes_proof",
    ],
)
def test_cleanup_certificate_rejects_each_lineage_seam_mutation(role, seam):
    certificate = proved_cleanup_certificate(role)
    mutate_cleanup_lineage_seam(certificate, seam)

    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
def test_cleanup_certificate_rejects_internally_consistent_proof_after_window(role):
    certificate = proved_cleanup_certificate(role)
    reset = certificate["reset"]
    final = certificate["final_state"]
    delta = 30_000
    reset["request_monotonic_ns"] += delta
    reset["boundary"]["boundary_monotonic_ns"] += delta
    for name in ("race", "imu"):
        reset["baseline"][name]["ingress"]["received_monotonic_ns"] += delta
    reset["receipt"]["call_start_monotonic_ns"] += delta
    reset["receipt"]["call_end_monotonic_ns"] += delta
    for receipt in certificate["outbound_receipts"]:
        if receipt.get("category") == "sim_reset":
            receipt["call_start_monotonic_ns"] = reset["receipt"][
                "call_start_monotonic_ns"
            ]
            receipt["call_end_monotonic_ns"] = reset["receipt"][
                "call_end_monotonic_ns"
            ]
            break
    for name in ("advancing_race", "advancing_imu"):
        for observation in reset[name]:
            observation["ingress"]["received_monotonic_ns"] += delta
    for name in ("heartbeat", "last_race", "last_imu"):
        final[name]["ingress"]["received_monotonic_ns"] += delta

    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
def test_cleanup_failed_disarm_receipt_cannot_precede_request(role):
    certificate = proved_cleanup_certificate(role)
    disarm = certificate["disarm"]
    receipt = disarm["receipt"]
    receipt["outcome"] = "raised"
    receipt["error_type"] = "RuntimeError"
    disarm.update(
        {
            "state": "request_failed",
            "heartbeat_after": None,
            "newer_confirmed": False,
        }
    )
    certificate["outcome"] = "failed"
    certificate["failure_codes"] = ["disarm_failed"]
    assert contract.validate_cleanup_certificate(certificate) == certificate

    receipt["call_start_monotonic_ns"] = disarm["request_monotonic_ns"] - 1
    receipt["call_end_monotonic_ns"] = disarm["request_monotonic_ns"]
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
@pytest.mark.parametrize(
    "case",
    [
        "zero_next_sequence_and_counts",
        "ingress_high_water_above_capacity",
        "buffered_without_boundary_observation",
        "handled_collisions_without_evidence",
    ],
)
def test_cleanup_certificate_rejects_impossible_boundary_diagnostics(role, case):
    certificate = proved_cleanup_certificate(role)
    boundary = certificate["reset"]["boundary"]
    ingress = boundary["ingress_stats"]
    collision = boundary["collision_stats"]
    if case == "zero_next_sequence_and_counts":
        ingress["next_sequence"] = 0
        for name in (
            "highres_imu_received",
            "heartbeat_received",
            "race_status_received",
            "actuator_received",
        ):
            ingress[name] = 0
        ingress["high_watermark"] = 0
        ingress["imu_high_watermark"] = 0
        ingress["other_high_watermark"] = 0
    elif case == "ingress_high_water_above_capacity":
        ingress["imu_capacity"] = 1
        ingress["imu_high_watermark"] = 2
    elif case == "buffered_without_boundary_observation":
        ingress["buffered_other"] = 1
    elif case == "handled_collisions_without_evidence":
        collision["handled"] = 99
    else:
        raise AssertionError(case)

    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)


def add_cleanup_collision_evidence(certificate):
    boundary = certificate["reset"]["boundary"]
    generation = boundary["old_generation"]

    def observation(sequence, observed, disposition):
        return {
            "schema": "aigp-vq2-runner-collision-observation/1",
            "reset_generation": generation,
            "observation_sequence": sequence,
            "host_clock_id": contract.HOST_CLOCK_ID,
            "observed_monotonic_ns": observed,
            "phase": "cleanup",
            "disposition": disposition,
            "boundary": "runner_drain_not_receiver_receipt",
            "collision": {
                "id": sequence + 1,
                "threat_level": 1,
                "impulse": 1.0,
            },
        }

    earlier = observation(0, boundary["boundary_monotonic_ns"] - 50, "observed")
    at_boundary = observation(
        1,
        boundary["boundary_monotonic_ns"],
        "reset_boundary_discard",
    )
    boundary["collisions"] = [copy.deepcopy(at_boundary)]
    boundary["collision_stats"].update(
        {"handled": 2, "dropped": 0, "high_watermark": 1, "buffered": 1}
    )
    certificate["collisions"] = {
        "observations": [earlier, at_boundary],
        "invalidating_occurrence_count": 2,
    }
    certificate["collection_invalidating_codes"] = ["collision_observed"]


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
def test_cleanup_certificate_accepts_complete_ordered_collision_accounting(role):
    certificate = proved_cleanup_certificate(role)
    add_cleanup_collision_evidence(certificate)

    assert contract.validate_cleanup_certificate(certificate) == certificate


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
@pytest.mark.parametrize(
    "case",
    ["reversed", "duplicate", "handled_mismatch", "buffered_mismatch"],
)
def test_cleanup_certificate_rejects_inconsistent_complete_collision_array(role, case):
    certificate = proved_cleanup_certificate(role)
    add_cleanup_collision_evidence(certificate)
    rows = certificate["collisions"]["observations"]
    stats = certificate["reset"]["boundary"]["collision_stats"]
    if case == "reversed":
        rows.reverse()
    elif case == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
        certificate["collisions"]["invalidating_occurrence_count"] += 1
    elif case == "handled_mismatch":
        stats["handled"] += 1
    elif case == "buffered_mismatch":
        stats["buffered"] = 0
    else:
        raise AssertionError(case)

    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)


@pytest.mark.parametrize("role", ["powered_child", "cleanup_fallback"])
def test_cleanup_partial_final_state_preserves_one_truthful_failed_consistency(role):
    certificate = proved_cleanup_certificate(role)
    certificate["outcome"] = "failed"
    certificate["failure_codes"] = ["final_state_unproved"]
    certificate["final_state"]["state"] = "partial"

    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)

    certificate["final_state"]["heartbeat"]["heartbeat"]["base_mode"] = 128
    certificate["final_state"]["disarmed"] = False

    assert contract.validate_cleanup_certificate(certificate) == certificate

    certificate["final_state"]["heartbeat"]["heartbeat"]["base_mode"] = 0
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(certificate)


def test_child_final_sources_may_advance_beyond_reset_proof_but_fallback_is_exact():
    child = proved_cleanup_certificate("powered_child")
    fallback = proved_cleanup_certificate("cleanup_fallback")
    for certificate in (child, fallback):
        last_race = certificate["final_state"]["last_race"]
        last_race["ingress"]["sequence"] += 2
        last_race["ingress"]["received_monotonic_ns"] += 100
        last_race["ingress"]["source_time_value"] += 1
        last_race["race_status"]["sim_boot_time_ms"] += 1

    assert contract.validate_cleanup_certificate(child) == child
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_cleanup_certificate(fallback)


def test_invalid_record_derives_safe_and_unsafe_poison_predicates():
    safe = invalid_record()
    unsafe = invalid_record(unsafe=True)
    assert contract.validate_attempt_invalid(safe) == safe
    assert contract.validate_attempt_invalid(unsafe) == unsafe
    unsafe["poison"]["required"] = False
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_attempt_invalid(unsafe)


def test_bundle_verification_split_and_complete_terminal_shapes():
    verify = {
        "schema": "aigp-vq2-replay-bundle-verification/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "verified_at_utc": UTC,
        "verified_monotonic_ns": 20,
        "timing": timing("bundle_verify", prepared=20),
        "identity": {
            "candidate_commit": COMMIT,
            "live_freeze_sha256": H,
            "attempt_context_sha256": H,
            "attempt_envelope_sha256": H,
            "child_authority_sha256": H,
            "child_process_result_sha256": H,
            "child_cleanup_certificate_sha256": H,
            "lease_final_sha256": H,
        },
        "bundle": {
            "path": contract.frozen_paths()["replay_bundle"],
            "dataset_hash": H,
            "manifest": artifact("replay_manifest"),
            "records": artifact("replay_records"),
            "frames": [],
        },
        "checks": {name: True for name in contract._BUNDLE_CHECKS},
        "valid": True,
    }
    assert contract.validate_bundle_verification(verify) == verify

    # The name binds decoded content while ArtifactRef.sha256 binds the
    # complete .npy bytes. Those hashes are independent by design.
    distinct_hashes = copy.deepcopy(verify)
    distinct_hashes["bundle"]["frames"] = [
        artifact(f"replay_frame/{H2}", digest=H3)
    ]
    assert contract.validate_bundle_verification(distinct_hashes) == distinct_hashes

    malformed_name = copy.deepcopy(distinct_hashes)
    malformed_name["bundle"]["frames"][0]["name"] = "replay_frame/B" + "b" * 63
    with pytest.raises(
        contract.PoweredAttemptContractError,
        match="canonical lowercase",
    ):
        contract.validate_bundle_verification(malformed_name)
    claim = {
        "schema": "aigp-vq2-package2-run-split-claim/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "claimed_at_utc": UTC,
        "claimed_monotonic_ns": 20,
        "timing": timing("split_publish", prepared=20),
        "run_id": "F00-A01/reset-epoch-1/excitation-1",
        "assigned_split": "discovery_fit",
        "identity": {
            "attempt_context_sha256": H,
            "attempt_envelope_sha256": H,
            "capture_seal_sha256": H,
            "excitation_plan_id": contract.EXCITATION_PLAN_ID,
            "excitation_plan_sha256": contract.EXCITATION_PLAN_SHA256,
        },
        "reset_epochs": [{"ingress_generation": 1, "race_anchor_boot_ms": 1, "imu_anchor_usec": 1}],
        "run_artifacts": sorted(
            [artifact(name) for name in (
                "bundle_verification", "child_cleanup_certificate", "legacy_record",
                "replay_manifest", "replay_records", f"replay_frame/{H}",
                "runner_stdout", "runner_stderr",
            )],
            key=lambda item: item["name"].encode(),
        ),
        "decoded_content_sha256": [H],
        "derivative_sha256": [],
        "collision_policy": "f00_fixed_future_whole_run_discovery_fit_or_global_exclusion",
    }
    assert contract.validate_split_claim(claim) == claim
    registry = {
        "schema": "aigp-vq2-package2-split-registry/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "published_at_utc": UTC,
        "published_monotonic_ns": 20,
        "timing": timing("split_publish", prepared=20),
        "registry_id": "vq2-package2-calibration",
        "revision": 1,
        "previous_registry_sha256": None,
        "claims": [{
            "claim_path": contract.frozen_paths()["split_claim"],
            "claim_sha256": contract.canonical_file_sha256(claim),
            "session_id": contract.SESSION_ID,
            "attempt_id": contract.ATTEMPT_ID,
            "run_id": "F00-A01/reset-epoch-1/excitation-1",
            "assigned_split": "discovery_fit",
            "activation": "requires_matching_attempt_complete",
        }],
        "content_groups": [{
            "decoded_sha256": H,
            "run_ids": ["F00-A01/reset-epoch-1/excitation-1"],
            "assigned_split": "discovery_fit",
            "disposition": "assigned",
            "activation": "requires_matching_attempt_complete",
        }],
    }
    assert contract.validate_split_registry(registry, split_claim=claim) == registry
    complete = {
        "schema": "aigp-vq2-powered-calibration-attempt-complete/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "completed_at_utc": UTC,
        "completed_monotonic_ns": 20,
        "deadline_monotonic_ns": 5_000_000_010,
        "publication_timing": timing("terminal_publish", prepared=20),
        "identity": {
            "candidate_commit": COMMIT,
            "code_sha256": H,
            "live_freeze_sha256": H,
            "attempt_context_sha256": H,
            "attempt_envelope_sha256": H,
            "target_config_sha256": H,
            "capture_authorization_sha256": H,
            "excitation_plan_id": contract.EXCITATION_PLAN_ID,
            "excitation_plan_sha256": contract.EXCITATION_PLAN_SHA256,
            "wrapper_lifecycle_sha256": H,
        },
        "artifact_hashes": {
            name: None if name in {
                "cleanup_authority_sha256", "fallback_cleanup_certificate_sha256",
                "cleanup_stdout_sha256", "cleanup_stderr_sha256",
            } else H
            for name in contract._COMPLETE_ARTIFACT_HASH_KEYS
        },
        "cleanup": terminal_cleanup(),
    }
    assert contract.validate_attempt_complete(complete) == complete
    complete["completed_monotonic_ns"] = True
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_attempt_complete(complete)


def test_unknown_missing_and_bool_as_int_are_rejected_by_dispatch():
    item = heartbeat()
    item["unknown"] = 1
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_powered_record(item)
    item = heartbeat()
    del item["heartbeat"]["custom_mode"]
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_received_heartbeat(item)


def test_environment_and_import_inventories_are_exact_sorted_and_dispatched():
    environment = {
        "schema": "aigp-vq2-powered-environment-inventory/1",
        "created_at_utc": UTC,
        "variables": [
            {"name": "PATH", "defined": True, "value_sha256": H},
            {"name": "TEMP", "defined": True, "value_sha256": H2},
        ],
    }
    assert contract.validate_environment_inventory(environment) == environment
    assert contract.validate_powered_record(environment) == environment

    wrong_case = copy.deepcopy(environment)
    wrong_case["variables"][0]["name"] = "Path"
    with pytest.raises(contract.PoweredAttemptContractError, match="uppercase"):
        contract.validate_environment_inventory(wrong_case)
    wrong_order = copy.deepcopy(environment)
    wrong_order["variables"].reverse()
    with pytest.raises(contract.PoweredAttemptContractError, match="sorted"):
        contract.validate_environment_inventory(wrong_order)

    imports = {
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
                "module": "_frozen_importlib",
                "origin": None,
                "size_bytes": None,
                "sha256": None,
                "root_class": "frozen",
                "namespace_roots": [],
            },
            {
                "module": "cv2.utils.fs",
                "origin": PYTHON,
                "size_bytes": 1,
                "sha256": H3,
                "root_class": "runtime",
                "namespace_roots": [],
            },
            {
                "module": "scripts",
                "origin": None,
                "size_bytes": None,
                "sha256": None,
                "root_class": "namespace",
                "namespace_roots": [LIVE_WORKTREE + r"\scripts"],
            },
            {
                "module": "scripts.aigp_vq2_powered_attempt",
                "origin": LIVE_WORKTREE
                + r"\scripts\aigp_vq2_powered_attempt.py",
                "size_bytes": 1,
                "sha256": H2,
                "root_class": "candidate",
                "namespace_roots": [],
            },
        ],
    }
    assert contract.validate_import_inventory(imports) == imports
    assert contract.validate_powered_record(imports) == imports

    mixed_shape = copy.deepcopy(imports)
    mixed_shape["entries"][0]["origin"] = PYTHON
    with pytest.raises(contract.PoweredAttemptContractError, match="must be null"):
        contract.validate_import_inventory(mixed_shape)
    missing_runtime_descriptor = copy.deepcopy(imports)
    missing_runtime_descriptor["entries"][1]["sha256"] = None
    with pytest.raises(contract.PoweredAttemptContractError, match="exact string"):
        contract.validate_import_inventory(missing_runtime_descriptor)
    unapproved_runtime = copy.deepcopy(imports)
    unapproved_runtime["entries"][1]["module"] = "cv2.utils.unknown"
    with pytest.raises(contract.PoweredAttemptContractError, match="not an allowed"):
        contract.validate_import_inventory(unapproved_runtime)
    duplicate = copy.deepcopy(imports)
    duplicate["entries"].append(copy.deepcopy(duplicate["entries"][-1]))
    with pytest.raises(contract.PoweredAttemptContractError, match="unique"):
        contract.validate_import_inventory(duplicate)


def test_environment_inventory_semantics_can_be_cross_bound_into_live_freeze():
    environment = {
        "schema": "aigp-vq2-powered-environment-inventory/1",
        "created_at_utc": UTC,
        "variables": [
            {"name": "PATH", "defined": True, "value_sha256": H},
            {"name": "TEMP", "defined": True, "value_sha256": H2},
        ],
    }
    expected_semantic_sha256 = contract.canonical_object_sha256(
        {"variables": environment["variables"]}
    )
    assert contract.environment_variables_sha256(environment) == expected_semantic_sha256

    provenance_only_change = copy.deepcopy(environment)
    provenance_only_change["created_at_utc"] = "2026-07-20T12:34:57.123456Z"
    assert (
        contract.environment_variables_sha256(provenance_only_change)
        == expected_semantic_sha256
    )
    assert contract.canonical_file_sha256(provenance_only_change) != contract.canonical_file_sha256(
        environment
    )

    freeze = live_freeze()
    freeze["runtime"]["environment_inventory"]["sha256"] = (
        contract.canonical_file_sha256(environment)
    )
    freeze["execution"]["launcher_environment_sha256"] = expected_semantic_sha256
    assert (
        contract.validate_live_freeze(
            freeze,
            environment_inventory=environment,
        )
        == freeze
    )

    wrong_file_binding = copy.deepcopy(freeze)
    wrong_file_binding["runtime"]["environment_inventory"]["sha256"] = H3
    with pytest.raises(
        contract.PoweredAttemptContractError,
        match=r"runtime\.environment_inventory\.sha256",
    ):
        contract.validate_live_freeze(
            wrong_file_binding,
            environment_inventory=environment,
        )

    wrong_semantic_binding = copy.deepcopy(freeze)
    wrong_semantic_binding["execution"]["launcher_environment_sha256"] = H3
    with pytest.raises(
        contract.PoweredAttemptContractError,
        match=r"execution\.launcher_environment_sha256",
    ):
        contract.validate_live_freeze(
            wrong_semantic_binding,
            environment_inventory=environment,
        )


def test_import_inventory_file_and_python_are_cross_bound_into_live_freeze():
    imports = {
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
        "entries": [],
    }
    freeze = live_freeze()
    freeze["runtime"]["import_inventory"]["sha256"] = (
        contract.canonical_file_sha256(imports)
    )
    assert contract.validate_live_freeze(freeze, import_inventory=imports) == freeze

    wrong_file_binding = copy.deepcopy(freeze)
    wrong_file_binding["runtime"]["import_inventory"]["sha256"] = H2
    with pytest.raises(
        contract.PoweredAttemptContractError,
        match=r"runtime\.import_inventory\.sha256",
    ):
        contract.validate_live_freeze(
            wrong_file_binding,
            import_inventory=imports,
        )

    wrong_python_binding = copy.deepcopy(imports)
    wrong_python_binding["python_sha256"] = H2
    wrong_python_freeze = copy.deepcopy(freeze)
    wrong_python_freeze["runtime"]["import_inventory"]["sha256"] = (
        contract.canonical_file_sha256(wrong_python_binding)
    )
    with pytest.raises(
        contract.PoweredAttemptContractError,
        match="inventory Python",
    ):
        contract.validate_live_freeze(
            wrong_python_freeze,
            import_inventory=wrong_python_binding,
        )


def test_implementation_inventory_can_be_cross_bound_into_live_freeze():
    inventory = {
        "schema": "aigp-vq2-powered-implementation-inventory/1",
        "commit": COMMIT,
        "tree": "e" * 40,
        "entries": [
            {"path": "scripts/a.py", "size_bytes": 1, "sha256": H},
            {"path": "tests/test_a.py", "size_bytes": 2, "sha256": H2},
        ],
    }
    freeze = live_freeze()
    freeze["candidate"]["implementation_inventory"]["sha256"] = contract.canonical_file_sha256(inventory)
    freeze["candidate"]["code_sha256"] = contract.canonical_object_sha256(
        {name: inventory[name] for name in ("commit", "tree", "entries")}
    )
    assert contract.validate_live_freeze(freeze, implementation_inventory=inventory) == freeze
    inventory["entries"].reverse()
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_live_freeze(freeze, implementation_inventory=inventory)


def test_simulator_process_proof_and_training_attestation_bind():
    launcher = process(30)
    payload = process(40)

    def owner(observed: int) -> dict[str, object]:
        return {
            "observed_monotonic_ns": observed,
            "ipv4_14550": [],
            "ipv6_14550": [],
            "ipv4_5600": [],
            "ipv6_5600": [],
        }

    proof = {
        "schema": "aigp-vq2-simulator-process-proof/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "phase": "prechild",
        "observed_at_utc": UTC,
        "observed_monotonic_ns": 100,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "wrapper_process": process(10),
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
            "hwnd": 1,
            "owner_pid": payload["pid"],
            "visible": True,
            "unminimized": True,
            "responsive": True,
        },
        "build": 3385,
        "topology": "one_launcher_parent_retained_one_payload_child",
        "scheduled_task": {
            "name": "AIGP-P2-F00-A01-Launch",
            "observations": [
                {"phase": phase, "observed_monotonic_ns": index + 3, "query_exit_code": 1, "absent": True}
                for index, phase in enumerate(("before_launch", "after_launcher_return", "before_child"))
            ],
        },
        "ports": {
            "owner_table_observations": [owner(10), owner(20)],
            "active_owner_observations": [],
            "exclusive_probes": [
                {"host": "127.0.0.1", "port": 14550, "started_monotonic_ns": 30, "ended_monotonic_ns": 31, "result": "bound_and_closed"},
                {"host": "0.0.0.0", "port": 5600, "started_monotonic_ns": 32, "ended_monotonic_ns": 33, "result": "bound_and_closed"},
            ],
            "status": "free",
        },
        "responsive": True,
    }
    assert contract.validate_simulator_process_proof(proof) == proof
    attestation = {
        "schema": "aigp-vq2-training-mode-attestation/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "attested_at_utc": UTC,
        "attested_monotonic_ns": 101,
        "host_clock_id": contract.HOST_CLOCK_ID,
        "mode": "Training",
        "method": "post_topology_visual_training_check_challenge",
        "challenge_sha256": H,
        "wrapper_process": proof["wrapper_process"],
        "simulator_process_proof_sha256": contract.canonical_file_sha256(proof),
    }
    assert contract.validate_training_attestation(attestation, process_proof=proof) == attestation
    proof["window"]["responsive"] = False
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_simulator_process_proof(proof)


def test_live_poison_shape_has_no_automatic_clear_or_ambiguous_states():
    poison = {
        "schema": "aigp-vq2-powered-calibration-live-poison/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "created_at_utc": UTC,
        "created_monotonic_ns": 20,
        "publication_timing": timing("poison_publish", prepared=20),
        "phase": "child_supervision",
        "reason_codes": ["cleanup_unconfirmed"],
        "attempt_context_sha256": H,
        "attempt_envelope_sha256": H,
        "wrapper_process": process(10),
        "child_process": process(20),
        "cleanup_process": None,
        "lease_state": {
            "phase": "retained",
            "owner_token_sha256": H,
            "release_proved": False,
        },
        "port_state": {"mavlink_14550": "owned", "camera_5600": "unproved"},
        "process_state": "unproved",
        "transport_state": "open",
        "scheduled_task_state": "unproved",
        "publication_state": {
            "bundle_verification": "absent",
            "capture_seal": "absent",
            "claim": "absent",
            "registry": "absent",
            "report": "absent",
            "wrapper_lifecycle": "partial",
            "attempt_complete": "absent",
            "terminal": "missing",
        },
        "simulator_state": {"topology": "unproved", "responsive": "unproved"},
        "required_action": "new_reviewed_recovery_task_no_automatic_clear",
    }
    assert contract.validate_live_poison(poison) == poison
    poison["required_action"] = "retry"
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_live_poison(poison)


def test_capture_seal_and_valid_acquisition_report_keep_calibration_uncomputed():
    required_names = {
        "live_freeze", "implementation_inventory", "environment_inventory", "import_inventory",
        "attempt_envelope", "training_attestation", "process_prechild", "process_postchild",
        "child_authority", "child_cleanup_certificate", "lease_final", "bundle_verification",
        "runner_stdout", "runner_stderr", "legacy_record", "replay_manifest", "replay_records",
    }
    seal = {
        "schema": "aigp-vq2-powered-calibration-capture-seal/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "sealed_at_utc": UTC,
        "timing": timing("capture_seal"),
        "identity": {
            "candidate_commit": COMMIT,
            "code_sha256": H,
            "live_freeze_sha256": H,
            "attempt_context_sha256": H,
            "attempt_envelope_sha256": H,
            "target_config_sha256": H,
            "capture_authorization_sha256": H,
            "excitation_plan_id": contract.EXCITATION_PLAN_ID,
            "excitation_plan_sha256": contract.EXCITATION_PLAN_SHA256,
            "training_attestation_sha256": H,
            "simulator_process_proof_sha256": H,
            "simulator_final_process_proof_sha256": H,
            "child_authority_sha256": H,
            "cleanup_authority_sha256": None,
            "lease_final_sha256": H,
            "bundle_verification_sha256": H,
        },
        "artifacts": sorted([artifact(name) for name in required_names], key=lambda item: item["name"].encode()),
        "capture_stats": {name: 0 for name in contract._CAPTURE_STATS},
        "outbound_audit": {name: 0 for name in (
            "timesync", "gcs_heartbeat", "sim_reset", "arm", "disarm",
            "attitude_target", "position_target", "other_command", "receipt_count",
            "receipt_returned", "receipt_raised", "receipt_dropped", "receipt_buffered",
        )},
        "cleanup": terminal_cleanup(),
    }
    assert contract.validate_capture_seal(seal) == seal

    report = {
        "schema": "aigp-vq2-powered-calibration-acquisition-report/1",
        "task_id": contract.TASK_ID,
        "session_id": contract.SESSION_ID,
        "attempt_id": contract.ATTEMPT_ID,
        "generated_at_utc": UTC,
        "timing": timing("split_publish"),
        "collection_valid": True,
        "invalid_reasons": [],
        "reference_scope": {
            "conditional_on_nominal_gate_config": True,
            "geometry_status": "nominal_unverified_for_build_3385_training",
            "target_config_sha256": H,
        },
        "identity": {
            "candidate_commit": COMMIT,
            "live_freeze_sha256": H,
            "attempt_context_sha256": H,
            "attempt_envelope_sha256": H,
            "target_config_sha256": H,
            "capture_authorization_sha256": H,
            "excitation_plan_id": contract.EXCITATION_PLAN_ID,
            "excitation_plan_sha256": contract.EXCITATION_PLAN_SHA256,
            "training_attestation_sha256": H,
            "simulator_process_proof_sha256": H,
            "simulator_final_process_proof_sha256": H,
            "child_authority_sha256": H,
            "cleanup_authority_sha256": None,
            "lease_final_sha256": H,
            "bundle_verification_sha256": H,
        },
        "input_artifacts": {
            "capture_seal_sha256": H,
            "bundle_dataset_hash": H,
            "bundle_verification_sha256": H,
            "bundle_manifest_sha256": H,
            "bundle_records_sha256": H,
            "legacy_record_sha256": H,
            "lease_final_sha256": H,
            "runner_stdout_sha256": H,
            "runner_stderr_sha256": H,
            "child_cleanup_certificate_sha256": H,
            "fallback_cleanup_certificate_sha256": None,
        },
        "checks": {name: True for name in contract._REPORT_CHECKS},
        "counts": {
            **{name: 0 for name in contract._REPORT_COUNTS},
            "ticks_skipped_before_generation": 245,
        },
        "command_accounting": {
            "attitude_target_audit_delta": 0,
            "generated_count": 0,
            "sent_count": 0,
            "not_sent_count": 0,
            "unmatched_generation_count": 0,
            "unmatched_sent_count": 0,
            "failed_or_uncertain_count": 0,
            "envelope_violation_count": 0,
            "payload_mismatch_count": 0,
            "all_reconciled": True,
        },
        "excitation_accounting": {
            "plan_id": contract.EXCITATION_PLAN_ID,
            "plan_sha256": contract.EXCITATION_PLAN_SHA256,
            "tick_count": 245,
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "planned_ticks": segment["last_tick"] - segment["first_tick"] + 1,
                    "generated": 0,
                    "sent": 0,
                    "skipped": segment["last_tick"] - segment["first_tick"] + 1,
                }
                for segment in contract.frozen_excitation_plan()["segments"]
            ],
            "first_release_monotonic_ns": 100,
            "last_slot_end_monotonic_ns": 4_900_000_100,
            "powered_expiry_monotonic_ns": 5_000_000_100,
        },
        "descriptive_support": {
            "target_observation_count": 1,
            "target_center_x_px_min": 320.0,
            "target_center_x_px_max": 320.0,
            "target_center_y_px_min": 180.0,
            "target_center_y_px_max": 180.0,
            "target_bbox_area_px_min": 100.0,
            "target_bbox_area_px_max": 100.0,
            "gyro_x_rad_s_min": 0.0,
            "gyro_x_rad_s_max": 0.0,
            "gyro_y_rad_s_min": 0.0,
            "gyro_y_rad_s_max": 0.0,
            "gyro_z_rad_s_min": 0.0,
            "gyro_z_rad_s_max": 0.0,
            "roll_reversal_count": 0,
            "pitch_reversal_count": 0,
            "semantics": "descriptive_only_no_acceptance_threshold",
        },
        "calibration_status": {
            name: "uncomputed"
            for name in (
                "intrinsics", "distortion", "camera_to_body_rotation",
                "camera_imu_time_model", "rank", "covariance", "empirical_limits",
            )
        },
        "unmeasured": [
            "absolute_host_phase", "accepted_calibration_coefficients",
            "command_to_actuator_response", "empirical_limits",
            "encode_queue_component_delays", "package2_acceptance",
            "render_exposure_delay",
        ],
        "split": {
            "assigned_split": "discovery_fit",
            "claim_path": contract.frozen_paths()["split_claim"],
            "claim_sha256": H,
            "registry_path": contract.frozen_paths()["split_registry"],
            "registry_sha256": H,
            "activation": "requires_matching_attempt_complete",
        },
    }
    assert contract.validate_acquisition_report(report) == report
    report["calibration_status"]["rank"] = "computed"
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_acquisition_report(report)
    item = heartbeat()
    item["heartbeat"]["base_mode"] = True
    with pytest.raises(contract.PoweredAttemptContractError):
        contract.validate_received_heartbeat(item)
