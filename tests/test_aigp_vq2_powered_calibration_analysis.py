from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import aigp_vq2_powered_attempt as attempt
from scripts import aigp_vq2_powered_calibration_analysis as analysis
from scripts.aigp_live_lease import (
    LIVE_LEASE_MUTEX_NAME,
    validate_powered_live_lease_index,
)


CONTENT_SHA = "1" * 64
FRAME_FILE_SHA = "2" * 64
CONTEXT_SHA = "3" * 64
COMMIT = "4" * 40
UTC = "2026-07-20T12:34:56.123456Z"


def _ingress(message_type: str, sequence: int, source: int | None) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-mavlink-ingress/1",
        "stream_id": "vq2-mavlink-udp-14550",
        "generation": 1,
        "sequence": sequence,
        "message_type": message_type,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "received_monotonic_ns": 1_000 + sequence,
        "source_time_value": source,
        "source_time_unit": {
            "HEARTBEAT": None,
            "RACE_STATUS": "ms",
            "HIGHRES_IMU": "us",
            "ACTUATOR_OUTPUT_STATUS": "us",
        }[message_type],
    }


def _received_values() -> tuple[dict[str, object], ...]:
    heartbeat = {
        "schema": "aigp-vq2-received-heartbeat/1",
        "ingress": _ingress("HEARTBEAT", 0, None),
        "heartbeat": {"base_mode": 128, "custom_mode": 0},
    }
    race = {
        "schema": "aigp-vq2-received-race-status/1",
        "ingress": _ingress("RACE_STATUS", 1, 20),
        "race_status": {
            "sim_boot_time_ms": 20,
            "race_start_boot_time_ms": 0,
            "race_finish_time_ns": 0,
            "active_gate_index": 0,
            "last_gate_race_time": 0,
        },
    }
    imu = {
        "schema": "aigp-vq2-received-imu/1",
        "ingress": _ingress("HIGHRES_IMU", 2, 30),
        "imu": {
            "timestamp_us": 30,
            "accel": [0.0, 0.0, 9.8],
            "gyro": [0.1, -0.2, 0.3],
            "mag": None,
        },
    }
    actuator = {
        "schema": "aigp-vq2-received-actuator-output-status/1",
        "ingress": _ingress("ACTUATOR_OUTPUT_STATUS", 3, 40),
        "actuator_output_status": {
            "time_usec": 40,
            "active": 0,
            "actuator": [0.0] * 32,
        },
    }
    return heartbeat, race, imu, actuator


def _frame_timing() -> dict[str, object]:
    return {
        "schema": "aigp-vq2-frame-timing/1",
        "identity": {
            "schema": "aigp-vq2-frame-identity/1",
            "stream_id": "vq2-camera",
            "generation": 1,
            "frame_id": 5,
        },
        "camera_source_time_ns": 50,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "publication_sequence": 0,
        "first_unique_packet_monotonic_ns": 100,
        "final_unique_packet_monotonic_ns": 101,
        "reassembly_complete_monotonic_ns": 102,
        "decode_start_monotonic_ns": 103,
        "decode_end_monotonic_ns": 104,
        "publish_monotonic_ns": 105,
    }


def _watchdogs(tick: int) -> dict[str, object]:
    return {
        "checked_monotonic_ns": 1_000_000_000 + tick * 20_000_000,
        "heartbeat_age_ns": 1,
        "imu_age_ns": 1,
        "imu_advance_age_ns": 1,
        "race_age_ns": 1,
        "race_advance_age_ns": 1,
        "actuator_age_ns": 1,
        "vision_age_ns": 1,
        "estimator_healthy": True,
        "target_consecutive": 3,
        "target_center_px": [300.0 + tick % 41, 170.0 + tick % 21],
        "target_bbox_px": [280.0, 150.0, 40.0, 40.0],
        "target_bbox_area_px": 1_600.0 + tick,
        "initial_target_bbox_area_px": 1_600.0,
        "roll_excursion_rad": 0.0,
        "pitch_excursion_rad": 0.0,
        "collision_count": 0,
        "gate_index": 0,
        "result": "pass",
        "failure_codes": [],
    }


def _cleanup_watchdogs(checked: int) -> dict[str, object]:
    return {
        "checked_monotonic_ns": checked,
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


def _source() -> dict[str, object]:
    heartbeat, race, imu, actuator = _received_values()
    timing = _frame_timing()
    return {
        "frame": {
            "stream_id": "vq2-camera",
            "generation": 1,
            "frame_id": 5,
            "sim_time_ns": 50,
            "timing": timing,
            "width": 640,
            "height": 360,
        },
        "imu": imu,
        "race": race,
        "heartbeat": heartbeat,
        "actuator": actuator,
    }


def _receipt(command: dict[str, object], sequence: int, when: int) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-attitude-target-outbound/1",
        "stream_id": "vq2-mavlink-udp-14550",
        "reset_generation": 1,
        "outbound_sequence": sequence,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "call_start_monotonic_ns": when,
        "call_end_monotonic_ns": when + 1,
        "api": "send_attitude_rate",
        "outcome": "returned",
        "error_type": None,
        "wire": {
            "time_boot_ms": sequence,
            "target_system": 1,
            "target_component": 1,
            "type_mask": 128,
            "q_wxyz": [1.0, 0.0, 0.0, 0.0],
            "body_rates_rad_s": [
                -command["roll_rate_rad_s"],
                -command["pitch_rate_rad_s"],
                -command["yaw_rate_rad_s"],
            ],
            "thrust": command["thrust"],
        },
    }


def _nonattitude_receipt(
    category: str,
    sequence: int,
    when: int,
    *,
    reset_generation: int = 1,
) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-nonattitude-outbound/1",
        "stream_id": "vq2-mavlink-udp-14550",
        "reset_generation": reset_generation,
        "outbound_sequence": sequence,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "call_start_monotonic_ns": when,
        "call_end_monotonic_ns": when + 1,
        "category": category,
        "api": "command_long_send",
        "outcome": "returned",
        "error_type": None,
        "wire": {
            "target_system": 1,
            "target_component": 1,
            "command": 400 if category == "disarm" else 31_000,
            "confirmation": 0,
            "params": [0.0] * 7,
        },
    }


def _event(name: str, observation: dict[str, object], **extra: object) -> dict[str, object]:
    return {"type": "event", "event": name, "observation": observation, **extra}


_CHILD_PHASE_DURATION_KEYS = {
    "connect": "child_connect",
    "preflight": "child_preflight",
    "reset_epoch": "child_reset_epoch",
    "normalize_disarmed": "child_normalize_disarmed",
    "countdown_go": "child_countdown_go",
    "arm": "child_arm",
    "powered_stage": "powered_stage",
    "cleanup": "child_cleanup",
    "replay_close": "child_replay_close",
}


def _phase_event(
    phase: str, event_sequence: int, started_monotonic_ns: int
) -> dict[str, object]:
    duration = attempt.DEADLINE_DURATIONS_NS[_CHILD_PHASE_DURATION_KEYS[phase]]
    parent = {
        "connect": 52_000_000_000,
        "preflight": 52_000_000_000,
        "reset_epoch": 52_000_000_000,
        "normalize_disarmed": 52_000_000_000,
        "countdown_go": 52_000_000_000,
        "arm": 52_000_000_000,
        "powered_stage": 57_000_000_000,
        "cleanup": 72_000_000_000,
        "replay_close": 107_000_000_000,
    }[phase]
    return _event(
        "calibration_phase_deadline",
        {
            "schema": "aigp-vq2-phase-deadline/1",
            "attempt_id": attempt.ATTEMPT_ID,
            "producer_role": "powered_child",
            "phase": phase,
            "event_sequence": event_sequence,
            "started_monotonic_ns": started_monotonic_ns,
            "duration_ns": duration,
            "parent_deadline_monotonic_ns": parent,
            "deadline_monotonic_ns": min(started_monotonic_ns + duration, parent),
        },
    )


def _replay_reset_boundary(
    old_generation: int,
    new_generation: int,
    boundary_monotonic_ns: int,
    *,
    received: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    counts = {
        "HIGHRES_IMU": 0,
        "HEARTBEAT": 0,
        "RACE_STATUS": 0,
        "ACTUATOR_OUTPUT_STATUS": 0,
    }
    for value in received:
        counts[value["ingress"]["message_type"]] += 1
    return {
        "schema": "aigp-vq2-calibration-reset-boundary/1",
        "old_generation": old_generation,
        "new_generation": new_generation,
        "boundary_monotonic_ns": boundary_monotonic_ns,
        "observations": [],
        "collisions": [],
        "ingress_stats": {
            "generation": old_generation,
            "next_sequence": len(received),
            "highres_imu_received": counts["HIGHRES_IMU"],
            "heartbeat_received": counts["HEARTBEAT"],
            "race_status_received": counts["RACE_STATUS"],
            "actuator_received": counts["ACTUATOR_OUTPUT_STATUS"],
            "dropped": 0,
            "high_watermark": len(received),
            "imu_capacity": 4096,
            "other_capacity": 4096,
            "imu_dropped": 0,
            "other_dropped": 0,
            "imu_high_watermark": counts["HIGHRES_IMU"],
            "other_high_watermark": len(received) - counts["HIGHRES_IMU"],
            "buffered_imu": 0,
            "buffered_other": 0,
        },
        "collision_stats": {
            "generation": old_generation,
            "handled": 0,
            "dropped": 0,
            "high_watermark": 0,
            "capacity": 128,
            "buffered": 0,
        },
    }


def _resource_stats(records: list[dict[str, object]]) -> dict[str, object]:
    boundary_rows = [
        row["observation"]
        for row in records
        if row.get("event") == "calibration_reset_boundary"
    ]
    final_generation = (
        boundary_rows[-1]["new_generation"] if boundary_rows else 1
    )
    received_names = {
        "received_heartbeat",
        "received_race_status",
        "received_actuator_output_status",
        "received_imu",
    }
    final_received = [
        row["observation"]
        for row in records
        if row.get("event") in received_names
        and row["observation"]["ingress"]["generation"] == final_generation
    ]
    received_counts = {
        message_type: sum(
            row["ingress"]["message_type"] == message_type
            for row in final_received
        )
        for message_type in (
            "HIGHRES_IMU",
            "HEARTBEAT",
            "RACE_STATUS",
            "ACTUATOR_OUTPUT_STATUS",
        )
    }
    final_collisions = [
        row["observation"]
        for row in records
        if row.get("event") == "runner_collision_observation"
        and row["observation"]["reset_generation"] == final_generation
    ]
    outbound_rows = [
        row["observation"]
        for row in records
        if row.get("event") in {"attitude_target_outbound", "nonattitude_outbound"}
    ]
    outbound_count = len(outbound_rows)
    return {
        "schema": "aigp-vq2-powered-capture-resource-stats/1",
        "recorder": {
            "constructed": True,
            "enqueued": len(records),
            "written": len(records),
            "dropped": 0,
            "duplicate_frame_tokens": 0,
            "writer_errors": 0,
            "queue_high_watermark": 4,
            "decoded_frames_enqueued": 1,
            "decoded_frames_written": 1,
            "decoded_frames_dropped": 0,
            "complete": False,
            "failure_latched": False,
        },
        "vision": {
            "constructed": True,
            "datagrams_received": 100,
            "unique_datagrams": 10,
            "duplicate_datagrams": 90,
            "malformed_datagrams": 0,
            "frames_reassembled": 1,
            "frames_decoded": 1,
            "decode_failures": 0,
            "out_of_order_frame_drops": 0,
            "reset_generation_drops": 0,
            "processing_errors": 0,
            "socket_errors": 0,
            "snapshot_callback_errors": 0,
            "resets": 1,
            "remembered_chunk_keys": 10,
            "timing_ledger_entries": 1,
            "timing_ledger_high_watermark": 1,
            "timing_ledger_capacity": 256,
            "timing_overflow_latched": False,
            "receiver_buffered_partial_frames": 0,
            "receiver_buffer_high_watermark": 1,
            "receiver_buffer_capacity": 16,
            "capture_snapshot_queue_entries": 0,
            "capture_snapshot_queue_high_watermark": 1,
            "capture_snapshot_queue_capacity": 256,
            "capture_snapshot_queue_dropped": 0,
            "capture_snapshot_queue_enabled": True,
            "receiver_dropped_partial_frames": 0,
            "receiver_duplicate_chunks": 90,
            "receiver_dropped_late_packets": 0,
        },
        "ingress": {
            "constructed": True,
            "generation": final_generation,
            "next_sequence": len(final_received),
            "highres_imu_received": received_counts["HIGHRES_IMU"],
            "heartbeat_received": received_counts["HEARTBEAT"],
            "race_status_received": received_counts["RACE_STATUS"],
            "actuator_received": received_counts["ACTUATOR_OUTPUT_STATUS"],
            "dropped": 0,
            "high_watermark": len(final_received),
            "imu_capacity": 4096,
            "other_capacity": 4096,
            "imu_dropped": 0,
            "other_dropped": 0,
            "imu_high_watermark": received_counts["HIGHRES_IMU"],
            "other_high_watermark": len(final_received)
            - received_counts["HIGHRES_IMU"],
            "buffered_imu": 0,
            "buffered_other": 0,
        },
        "collision": {
            "constructed": True,
            "generation": final_generation,
            "handled": len(final_collisions),
            "dropped": 0,
            "high_watermark": len(final_collisions),
            "capacity": 128,
            "buffered": 0,
        },
        "outbound_receipts": {
            "constructed": True,
            "generation": final_generation,
            "next_sequence": outbound_count,
            "returned": sum(row["outcome"] == "returned" for row in outbound_rows),
            "raised": sum(row["outcome"] == "raised" for row in outbound_rows),
            "dropped": 0,
            "high_watermark": min(outbound_count, 4096),
            "capacity": 4096,
            "buffered": 0,
        },
        "snapshot_capture": {
            "constructed": True,
            "observed_frames": 1,
            "dimensions_admitted": True,
            "failure_latched": False,
        },
    }


def _seal_snapshot(records: list[dict[str, object]]) -> dict[str, object]:
    for sequence, row in enumerate(records):
        row.update(
            {
                "schema": analysis.REPLAY_RECORD_SCHEMA,
                "session_id": attempt.SESSION_ID,
                "sequence": sequence,
                "capture_wall_time_ns": 10_000 + sequence,
            }
        )
    records_bytes = b"".join(attempt.canonical_json_file_bytes(row) for row in records)
    records_sha = hashlib.sha256(records_bytes).hexdigest()
    manifest = {
        "schema": analysis.REPLAY_BUNDLE_SCHEMA,
        "record_schema": analysis.REPLAY_RECORD_SCHEMA,
        "session_id": attempt.SESSION_ID,
        "started_at": UTC,
        "finished_at": UTC,
        "complete": True,
        "private": True,
        "recording_notice": analysis.RECORDING_NOTICE,
        "metadata": {"stage": "calibration-excite"},
        "record_count": len(records),
        "frame_record_count": 0,
        "decoded_frame_record_count": 1,
        "unique_frame_blob_count": 1,
        "integrity": {
            "records_sha256": records_sha,
            "frame_blob_file_sha256": {CONTENT_SHA: FRAME_FILE_SHA},
        },
        "dataset_hash": "0" * 64,
        "outcome": {
            "status": "complete",
            "powered_capture_resource_stats": _resource_stats(records),
        },
    }
    manifest["dataset_hash"] = attempt.canonical_object_sha256(
        {
            "schema": analysis.REPLAY_BUNDLE_SCHEMA,
            "session_id": attempt.SESSION_ID,
            "started_at": UTC,
            "finished_at": UTC,
            "metadata": manifest["metadata"],
            "outcome": manifest["outcome"],
            "records_sha256": records_sha,
            "frame_blob_file_sha256": {CONTENT_SHA: FRAME_FILE_SHA},
        }
    )
    return {
        "manifest": manifest,
        "records": records,
        "manifest_bytes": attempt.canonical_json_file_bytes(manifest),
        "records_bytes": records_bytes,
        "frame_blob_file_sha256": {CONTENT_SHA: FRAME_FILE_SHA},
    }


def _valid_replay() -> dict[str, object]:
    heartbeat, race, imu, actuator = _received_values()
    timing = _frame_timing()
    event_sequence = 0
    records: list[dict[str, object]] = []
    for phase, started in (
        ("connect", 100_000_000),
        ("preflight", 200_000_000),
        ("reset_epoch", 400_000_000),
    ):
        records.append(_phase_event(phase, event_sequence, started))
        event_sequence += 1
    records.append(
        _event(
            "calibration_reset_boundary",
            _replay_reset_boundary(0, 1, 500_000_000),
        )
    )
    records.append(
        _event(
            "nonattitude_outbound",
            _nonattitude_receipt("sim_reset", 0, 500_000_001),
        )
    )
    for phase, started in (
        ("normalize_disarmed", 500_000_100),
        ("countdown_go", 700_000_000),
        ("arm", 800_000_000),
    ):
        records.append(_phase_event(phase, event_sequence, started))
        event_sequence += 1

    decoded_index = len(records)
    records.extend(
        [
            {
                "type": "decoded_frame",
                "generation": 1,
                "frame_id": 5,
                "sim_time_ns": 50,
                "received_monotonic_s": 0.0001,
                "frame_blob": f"frames/{CONTENT_SHA}.npy",
                "frame_hash": CONTENT_SHA,
                "image_shape": [360, 640, 3],
                "image_dtype": "|u1",
            },
        _event(
            "camera_frame_timing",
            timing,
                linked_decoded_frame_record_sequence=decoded_index,
        ),
        _event(
            "camera_frame_timing_observation",
            {
                "schema": "aigp-vq2-camera-frame-timing-observation/1",
                "frame_timing": timing,
                "consume_monotonic_ns": 106,
                "work_start_monotonic_ns": 106,
                "detection_start_monotonic_ns": 106,
                "detection_end_monotonic_ns": 107,
                "tracking_start_monotonic_ns": 107,
                "tracking_end_monotonic_ns": 108,
                "work_end_monotonic_ns": 109,
            },
        ),
        _event(
            "decoded_dimensions_admission",
            {
                "schema": "aigp-vq2-decoded-dimensions-admission/1",
                "config_sha256": "a" * 64,
                "expected": {"width": 640, "height": 360},
                "observed": {"width": 640, "height": 360},
                "first_frame_timing": timing,
                "admitted_monotonic_ns": 110,
                "status": "admitted",
            },
        ),
        ]
    )
    received = (
        ("received_heartbeat", heartbeat),
        ("received_race_status", race),
        ("received_actuator_output_status", actuator),
    )
    for event_name, observation in received:
        records.append(_event("mavlink_ingress", observation["ingress"]))
        records.append(_event(event_name, observation))
    records.append(
        {
            "type": "imu",
            "received_monotonic_s": 0.0002,
            "imu": imu["imu"],
            "estimator": None,
        }
    )
    imu_core_index = len(records) - 1
    records.append(_event("mavlink_ingress", imu["ingress"]))
    records.append(
        _event("received_imu", imu, linked_imu_record_sequence=imu_core_index)
    )
    records.append(
        {
            "type": "race_status",
            "received_monotonic_s": 0.0003,
            "race_status": race["race_status"],
        }
    )

    records.append(_phase_event("powered_stage", event_sequence, 900_000_000))
    event_sequence += 1

    anchor = 1_000_000_000
    for tick in range(245):
        plan_tick = attempt.excitation_tick(tick, anchor_monotonic_ns=anchor)
        command = copy.deepcopy(plan_tick["command"])
        generated = {
            "schema": "aigp-vq2-calibration-command-generated/1",
            "attempt_id": attempt.ATTEMPT_ID,
            "session_id": attempt.SESSION_ID,
            "candidate_commit": COMMIT,
            "attempt_context_sha256": CONTEXT_SHA,
            "event_sequence": event_sequence,
            "host_clock_id": attempt.HOST_CLOCK_ID,
            "generated_monotonic_ns": plan_tick["release_monotonic_ns"],
            "reset_epoch": {
                "ingress_generation": 1,
                "race_anchor_boot_ms": 10,
                "imu_anchor_usec": 10,
            },
            "plan": {
                "plan_id": attempt.EXCITATION_PLAN_ID,
                "sha256": attempt.EXCITATION_PLAN_SHA256,
            },
            "scope": "excitation",
            "command_id": f"excitation/{tick:03d}",
            "absolute_tick": tick,
            "segment_id": plan_tick["segment_id"],
            "slot": {
                "release_monotonic_ns": plan_tick["release_monotonic_ns"],
                "end_monotonic_ns": plan_tick["end_monotonic_ns"],
                "powered_expiry_monotonic_ns": plan_tick[
                    "powered_expiry_monotonic_ns"
                ],
            },
            "command": command,
            "source": _source(),
            "watchdogs": _watchdogs(tick),
        }
        records.append(_event("calibration_command_generated", generated))
        records.append(
            {
                "type": "command",
                "kind": "generated",
                "monotonic_s": generated["generated_monotonic_ns"] / 1_000_000_000,
                "frame_token": [1, 5, 50],
                "command": {
                    "roll_rate": command["roll_rate_rad_s"],
                    "pitch_rate": command["pitch_rate_rad_s"],
                    "yaw_rate": command["yaw_rate_rad_s"],
                    "thrust": command["thrust"],
                },
            }
        )
        receipt = _receipt(
            command, tick + 1, plan_tick["release_monotonic_ns"] + 1
        )
        records.append(_event("attitude_target_outbound", receipt))
        sent = {
            key: copy.deepcopy(value)
            for key, value in generated.items()
            if key not in {"schema", "event_sequence", "generated_monotonic_ns"}
        }
        sent.update(
            {
                "schema": "aigp-vq2-calibration-command-sent/1",
                "event_sequence": event_sequence + 1,
                "sent_monotonic_ns": plan_tick["release_monotonic_ns"] + 2,
                "generated_event_sequence": event_sequence,
                "generation_sha256": attempt.canonical_object_sha256(generated),
                "transport": {
                    "receipt": receipt,
                    "audit_count_before": tick,
                    "audit_count_after": tick + 1,
                },
            }
        )
        records.append(_event("calibration_command_sent", sent))
        records.append(
            {
                "type": "command",
                "kind": "sent",
                "monotonic_s": sent["sent_monotonic_ns"] / 1_000_000_000,
                "frame_token": [1, 5, 50],
                "command": {
                    "roll_rate": command["roll_rate_rad_s"],
                    "pitch_rate": command["pitch_rate_rad_s"],
                    "yaw_rate": command["yaw_rate_rad_s"],
                    "thrust": command["thrust"],
                },
            }
        )
        records.append(
            _event(
                "calibration_tick_disposition",
                {
                    "schema": "aigp-vq2-calibration-tick-disposition/1",
                    "attempt_id": attempt.ATTEMPT_ID,
                    "session_id": attempt.SESSION_ID,
                    "attempt_context_sha256": CONTEXT_SHA,
                    "plan_id": attempt.EXCITATION_PLAN_ID,
                    "plan_sha256": attempt.EXCITATION_PLAN_SHA256,
                    "event_sequence": event_sequence + 2,
                    "host_clock_id": attempt.HOST_CLOCK_ID,
                    "recorded_monotonic_ns": plan_tick["release_monotonic_ns"] + 3,
                    "absolute_tick": tick,
                    "segment_id": plan_tick["segment_id"],
                    "slot": generated["slot"],
                    "disposition": "sent",
                    "generated_event_sequence": event_sequence,
                    "terminal_event_sequence": event_sequence + 1,
                    "reason_code": None,
                },
            )
        )
        event_sequence += 3

    cleanup_time = anchor + 5_000_000_000
    records.append(
        _phase_event("cleanup", event_sequence, cleanup_time - 100)
    )
    event_sequence += 1
    zero = {
        "roll_rate_rad_s": 0.0,
        "pitch_rate_rad_s": 0.0,
        "yaw_rate_rad_s": 0.0,
        "thrust": 0.0,
    }
    cleanup_generated = {
        "schema": "aigp-vq2-calibration-command-generated/1",
        "attempt_id": attempt.ATTEMPT_ID,
        "session_id": attempt.SESSION_ID,
        "candidate_commit": COMMIT,
        "attempt_context_sha256": CONTEXT_SHA,
        "event_sequence": event_sequence,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "generated_monotonic_ns": cleanup_time,
        "reset_epoch": None,
        "plan": None,
        "scope": "cleanup_zero",
        "command_id": "cleanup/zero/0",
        "absolute_tick": None,
        "segment_id": None,
        "slot": None,
        "command": zero,
        "source": {
            "frame": None,
            "imu": None,
            "race": None,
            "heartbeat": None,
            "actuator": None,
        },
        "watchdogs": _cleanup_watchdogs(cleanup_time),
    }
    records.append(_event("calibration_command_generated", cleanup_generated))
    records.append(
        {
            "type": "command",
            "kind": "generated",
            "monotonic_s": cleanup_time / 1_000_000_000,
            "frame_token": None,
            "command": {"roll_rate": 0.0, "pitch_rate": 0.0, "yaw_rate": 0.0, "thrust": 0.0},
        }
    )
    cleanup_receipt = _receipt(zero, 246, cleanup_time + 1)
    records.append(_event("attitude_target_outbound", cleanup_receipt))
    cleanup_sent = {
        key: copy.deepcopy(value)
        for key, value in cleanup_generated.items()
        if key not in {"schema", "event_sequence", "generated_monotonic_ns"}
    }
    cleanup_sent.update(
        {
            "schema": "aigp-vq2-calibration-command-sent/1",
            "event_sequence": event_sequence + 1,
            "sent_monotonic_ns": cleanup_time + 2,
            "generated_event_sequence": event_sequence,
            "generation_sha256": attempt.canonical_object_sha256(cleanup_generated),
            "transport": {
                "receipt": cleanup_receipt,
                "audit_count_before": 245,
                "audit_count_after": 246,
            },
        }
    )
    records.append(_event("calibration_command_sent", cleanup_sent))
    records.append(
        {
            "type": "command",
            "kind": "sent",
            "monotonic_s": (cleanup_time + 2) / 1_000_000_000,
            "frame_token": None,
            "command": {"roll_rate": 0.0, "pitch_rate": 0.0, "yaw_rate": 0.0, "thrust": 0.0},
        }
    )
    records.append(
        _event(
            "nonattitude_outbound",
            _nonattitude_receipt("disarm", 247, cleanup_time + 3),
        )
    )
    records.append(
        _event(
            "calibration_reset_boundary",
            _replay_reset_boundary(
                1,
                2,
                cleanup_time + 20,
                received=(heartbeat, race, imu, actuator),
            ),
        )
    )
    records.append(
        _event(
            "nonattitude_outbound",
            _nonattitude_receipt(
                "sim_reset",
                248,
                cleanup_time + 21,
                reset_generation=2,
            ),
        )
    )
    records.append(
        _phase_event("replay_close", event_sequence + 2, cleanup_time + 100)
    )
    return _seal_snapshot(records)


@pytest.fixture(scope="module")
def replay() -> dict[str, object]:
    return _valid_replay()


def _reconcile(snapshot: dict[str, object]) -> dict[str, object]:
    return analysis.reconcile_calibration_records(
        snapshot["manifest"],
        snapshot["records"],
        manifest_bytes=snapshot["manifest_bytes"],
        records_bytes=snapshot["records_bytes"],
        frame_blob_file_sha256=snapshot["frame_blob_file_sha256"],
    )


def test_reconciles_all_245_ticks_and_only_frozen_descriptive_support(replay):
    result = _reconcile(replay)
    assert result["valid"] is True
    assert result["checks"]["resource_stats_zero"] is True
    assert result["checks"]["resource_counts_bound"] is True
    assert result["checks"]["reset_boundaries_exact"] is True
    assert result["counts"]["capture_drops"] == 0
    assert result["counts"]["ticks_sent"] == 245
    assert result["counts"]["generated_commands"] == 246
    assert result["command_accounting"]["all_reconciled"] is True
    assert result["descriptive_support"] == {
        "target_observation_count": 245,
        "target_center_x_px_min": 300.0,
        "target_center_x_px_max": 340.0,
        "target_center_y_px_min": 170.0,
        "target_center_y_px_max": 190.0,
        "target_bbox_area_px_min": 1600.0,
        "target_bbox_area_px_max": 1844.0,
        "gyro_x_rad_s_min": 0.1,
        "gyro_x_rad_s_max": 0.1,
        "gyro_y_rad_s_min": -0.2,
        "gyro_y_rad_s_max": -0.2,
        "gyro_z_rad_s_min": 0.3,
        "gyro_z_rad_s_max": 0.3,
        "roll_reversal_count": 4,
        "pitch_reversal_count": 3,
        "semantics": "descriptive_only_no_acceptance_threshold",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "zero",
        "missing_prepower",
        "missing_cleanup",
        "extra",
        "reordered",
        "before_cleanup_command",
    ),
)
def test_nominal_reset_boundaries_are_exact_phase_and_command_occurrences(
    replay, mutation
):
    records = copy.deepcopy(replay["records"])
    boundary_indices = [
        index
        for index, row in enumerate(records)
        if row.get("event") == "calibration_reset_boundary"
    ]
    assert len(boundary_indices) == 2

    if mutation == "zero":
        records = [
            row
            for row in records
            if row.get("event") != "calibration_reset_boundary"
        ]
    elif mutation == "missing_prepower":
        del records[boundary_indices[0]]
    elif mutation == "missing_cleanup":
        del records[boundary_indices[1]]
    elif mutation == "extra":
        replay_close_index = next(
            index
            for index, row in enumerate(records)
            if row.get("event") == "calibration_phase_deadline"
            and row["observation"]["phase"] == "replay_close"
        )
        records.insert(
            replay_close_index,
            _event(
                "calibration_reset_boundary",
                _replay_reset_boundary(2, 3, 6_000_000_050),
            ),
        )
    elif mutation == "reordered":
        first = records[boundary_indices[0]]["observation"]
        second = records[boundary_indices[1]]["observation"]
        records[boundary_indices[0]]["observation"] = second
        records[boundary_indices[1]]["observation"] = first
    else:
        cleanup_boundary = records.pop(boundary_indices[1])
        cleanup_phase_index = next(
            index
            for index, row in enumerate(records)
            if row.get("event") == "calibration_phase_deadline"
            and row["observation"]["phase"] == "cleanup"
        )
        records.insert(cleanup_phase_index + 1, cleanup_boundary)

    decoded_index = next(
        index for index, row in enumerate(records) if row.get("type") == "decoded_frame"
    )
    imu_index = next(
        index for index, row in enumerate(records) if row.get("type") == "imu"
    )
    for row in records:
        if row.get("event") == "camera_frame_timing":
            row["linked_decoded_frame_record_sequence"] = decoded_index
        elif row.get("event") == "received_imu":
            row["linked_imu_record_sequence"] = imu_index

    result = _reconcile(_seal_snapshot(records))
    assert result["valid"] is False
    assert result["checks"]["reset_boundaries_exact"] is False
    assert "capture_incomplete" in result["invalid_reasons"]


def test_resource_stats_are_raw_strict_and_never_manufactured(replay):
    raw = replay["manifest"]["outcome"]["powered_capture_resource_stats"]
    evidence = analysis.derive_powered_capture_resource_evidence(raw)
    assert evidence["resource_stats_zero"] is True
    # Build 3385 duplicate camera chunks are diagnostic, not capture loss.
    assert evidence["raw"]["vision"]["duplicate_datagrams"] == 90
    assert evidence["raw"]["vision"]["receiver_duplicate_chunks"] == 90

    dropped = copy.deepcopy(raw)
    dropped["ingress"]["dropped"] = 1
    dropped["ingress"]["imu_dropped"] = 1
    evidence = analysis.derive_powered_capture_resource_evidence(dropped)
    assert evidence["resource_stats_zero"] is False
    assert evidence["loss"]["ingress_drops"] == 1
    assert evidence["loss"]["observation_queue_drops"] == 1

    forged_closed = copy.deepcopy(raw)
    forged_closed["recorder"]["complete"] = True
    assert (
        analysis.derive_powered_capture_resource_evidence(forged_closed)[
            "resource_stats_zero"
        ]
        is False
    )

    missing = copy.deepcopy(raw)
    missing["recorder"].pop("writer_errors")
    with pytest.raises(analysis.PoweredCalibrationAnalysisError, match="fields must be exact"):
        analysis.validate_powered_capture_resource_stats(missing)

    absent = copy.deepcopy(replay)
    absent["manifest"]["outcome"].pop("powered_capture_resource_stats")
    absent["manifest"]["dataset_hash"] = attempt.canonical_object_sha256(
        {
            "schema": analysis.REPLAY_BUNDLE_SCHEMA,
            "session_id": absent["manifest"]["session_id"],
            "started_at": absent["manifest"]["started_at"],
            "finished_at": absent["manifest"]["finished_at"],
            "metadata": absent["manifest"]["metadata"],
            "outcome": absent["manifest"]["outcome"],
            "records_sha256": absent["manifest"]["integrity"][
                "records_sha256"
            ],
            "frame_blob_file_sha256": absent["manifest"]["integrity"][
                "frame_blob_file_sha256"
            ],
        }
    )
    absent["manifest_bytes"] = attempt.canonical_json_file_bytes(absent["manifest"])
    with pytest.raises(analysis.PoweredCalibrationAnalysisError, match="exact object"):
        _reconcile(absent)


def test_core_wire_and_tick_mismatches_fail_closed(replay):
    broken = copy.deepcopy(replay["records"])
    core = next(row for row in broken if row.get("type") == "command")
    core["command"]["roll_rate"] = 0.5
    result = _reconcile(_seal_snapshot(broken))
    assert result["valid"] is False
    assert result["checks"]["command_pairs_exact"] is False
    wrong_wire = copy.deepcopy(replay["records"])
    sent_event = next(
        row for row in wrong_wire if row.get("event") == "calibration_command_sent"
    )
    sent_event["observation"]["transport"]["receipt"]["wire"][
        "body_rates_rad_s"
    ][0] = 0.5
    outbound = next(
        row for row in wrong_wire if row.get("event") == "attitude_target_outbound"
    )
    outbound["observation"] = copy.deepcopy(
        sent_event["observation"]["transport"]["receipt"]
    )
    result = _reconcile(_seal_snapshot(wrong_wire))
    assert result["valid"] is False
    assert result["checks"]["command_pairs_exact"] is False

    missing = copy.deepcopy(replay["records"])
    missing.pop(
        next(
            index
            for index, row in enumerate(missing)
            if row.get("event") == "calibration_tick_disposition"
            and row["observation"]["absolute_tick"] == 100
        )
    )
    with pytest.raises(analysis.PoweredCalibrationAnalysisError, match="contiguous"):
        _reconcile(_seal_snapshot(missing))

    outside_slot = copy.deepcopy(replay["records"])
    generated_event = next(
        row
        for row in outside_slot
        if row.get("event") == "calibration_command_generated"
        and row["observation"]["scope"] == "excitation"
    )
    generated_event["observation"]["generated_monotonic_ns"] = (
        generated_event["observation"]["slot"]["end_monotonic_ns"]
    )
    generated_sequence = generated_event["observation"]["event_sequence"]
    core_generated = next(
        row
        for row in outside_slot
        if row.get("type") == "command"
        and row.get("kind") == "generated"
    )
    core_generated["monotonic_s"] = (
        generated_event["observation"]["generated_monotonic_ns"]
        / 1_000_000_000
    )
    sent_event = next(
        row
        for row in outside_slot
        if row.get("event") == "calibration_command_sent"
        and row["observation"]["generated_event_sequence"] == generated_sequence
    )
    sent_event["observation"]["generation_sha256"] = (
        attempt.canonical_object_sha256(generated_event["observation"])
    )
    sent_event["observation"]["sent_monotonic_ns"] = (
        generated_event["observation"]["generated_monotonic_ns"] + 1
    )
    first_core_sent = next(
        row
        for row in outside_slot
        if row.get("type") == "command" and row.get("kind") == "sent"
    )
    first_core_sent["monotonic_s"] = (
        sent_event["observation"]["sent_monotonic_ns"] / 1_000_000_000
    )
    result = _reconcile(_seal_snapshot(outside_slot))
    assert result["valid"] is False
    assert result["checks"]["command_pairs_exact"] is False

    call_started_at_end = copy.deepcopy(replay["records"])
    sent_event = next(
        row
        for row in call_started_at_end
        if row.get("event") == "calibration_command_sent"
        and row["observation"]["scope"] == "excitation"
    )
    receipt = sent_event["observation"]["transport"]["receipt"]
    receipt["call_start_monotonic_ns"] = sent_event["observation"]["slot"][
        "end_monotonic_ns"
    ]
    receipt["call_end_monotonic_ns"] = receipt["call_start_monotonic_ns"]
    sent_event["observation"]["sent_monotonic_ns"] = receipt[
        "call_end_monotonic_ns"
    ]
    outbound_event = next(
        row
        for row in call_started_at_end
        if row.get("event") == "attitude_target_outbound"
        and row["observation"]["outbound_sequence"]
        == receipt["outbound_sequence"]
    )
    outbound_event["observation"] = copy.deepcopy(receipt)
    core_sent = next(
        row
        for row in call_started_at_end
        if row.get("type") == "command" and row.get("kind") == "sent"
    )
    core_sent["monotonic_s"] = (
        sent_event["observation"]["sent_monotonic_ns"] / 1_000_000_000
    )
    result = _reconcile(_seal_snapshot(call_started_at_end))
    assert result["valid"] is False
    assert result["checks"]["command_pairs_exact"] is False


def test_outbound_sequence_is_attempt_global_across_reset_generations(replay):
    broken = copy.deepcopy(replay["records"])
    tail = [
        row for row in broken if row.get("event") == "nonattitude_outbound"
    ][-2:]
    for sequence, row in enumerate(tail):
        row["observation"]["reset_generation"] = 2
        row["observation"]["outbound_sequence"] = sequence
    result = _reconcile(_seal_snapshot(broken))
    assert result["valid"] is False
    assert result["checks"]["outbound_allowlist_exact"] is False


def test_outbound_resource_counters_and_generations_bind_serialized_receipts(replay):
    raw = replay["manifest"]["outcome"]["powered_capture_resource_stats"]
    exact = analysis._outbound_resource_evidence(
        replay["records"], final_resource_stats=raw
    )
    assert exact["exact"] is True

    # A plausible returned count cannot conceal missing attempt-global
    # sequence occurrences.
    forged = copy.deepcopy(replay)
    forged_manifest = forged["manifest"]
    forged_manifest["outcome"]["powered_capture_resource_stats"][
        "outbound_receipts"
    ]["next_sequence"] = 255
    forged_manifest["dataset_hash"] = attempt.canonical_object_sha256(
        {
            "schema": analysis.REPLAY_BUNDLE_SCHEMA,
            "session_id": forged_manifest["session_id"],
            "started_at": forged_manifest["started_at"],
            "finished_at": forged_manifest["finished_at"],
            "metadata": forged_manifest["metadata"],
            "outcome": forged_manifest["outcome"],
            "records_sha256": forged_manifest["integrity"]["records_sha256"],
            "frame_blob_file_sha256": forged_manifest["integrity"][
                "frame_blob_file_sha256"
            ],
        }
    )
    forged["manifest_bytes"] = attempt.canonical_json_file_bytes(forged_manifest)
    result = _reconcile(forged)
    assert result["valid"] is False
    assert result["checks"]["resource_counts_bound"] is False
    assert result["checks"]["outbound_allowlist_exact"] is False

    nested_only = copy.deepcopy(replay["records"])
    nested_sent = next(
        row
        for row in nested_only
        if row.get("event") == "calibration_command_sent"
        and row["observation"]["scope"] == "excitation"
    )
    nested_receipt = copy.deepcopy(
        nested_sent["observation"]["transport"]["receipt"]
    )
    nested_receipt["reset_generation"] = 2
    nested_sent["observation"]["transport"]["receipt"] = nested_receipt
    nested_snapshot = _seal_snapshot(nested_only)
    nested_resource = nested_snapshot["manifest"]["outcome"][
        "powered_capture_resource_stats"
    ]
    nested_evidence = analysis._outbound_resource_evidence(
        nested_snapshot["records"], final_resource_stats=nested_resource
    )
    assert nested_evidence["command_receipts_exact"] is False
    assert nested_evidence["exact"] is False
    result = _reconcile(nested_snapshot)
    assert result["valid"] is False
    assert result["checks"]["outbound_allowlist_exact"] is False

    # Keep the event receipt and command terminal byte-identical while moving
    # the call into the wrong reset generation.  Both reset-ledger and command
    # epoch binding must still reject it.
    wrong_command_generation = copy.deepcopy(replay["records"])
    sent = next(
        row
        for row in wrong_command_generation
        if row.get("event") == "calibration_command_sent"
        and row["observation"]["scope"] == "excitation"
    )
    receipt = sent["observation"]["transport"]["receipt"]
    receipt["reset_generation"] = 2
    outbound = next(
        row
        for row in wrong_command_generation
        if row.get("event") == "attitude_target_outbound"
        and row["observation"]["outbound_sequence"]
        == receipt["outbound_sequence"]
    )
    outbound["observation"] = copy.deepcopy(receipt)
    result = _reconcile(_seal_snapshot(wrong_command_generation))
    assert result["valid"] is False
    assert result["checks"]["outbound_allowlist_exact"] is False

    wrong_reset_generation = copy.deepcopy(replay["records"])
    reset_receipt = next(
        row
        for row in reversed(wrong_reset_generation)
        if row.get("event") == "nonattitude_outbound"
        and row["observation"].get("category") == "sim_reset"
    )
    reset_receipt["observation"]["reset_generation"] = 1
    result = _reconcile(_seal_snapshot(wrong_reset_generation))
    assert result["valid"] is False
    assert result["checks"]["outbound_allowlist_exact"] is False


def test_reset_boundary_generation_arithmetic_is_exact():
    heartbeat = copy.deepcopy(_received_values()[0])
    heartbeat["ingress"]["generation"] = 0
    heartbeat["ingress"]["received_monotonic_ns"] = 1
    collision = {
        "schema": "aigp-vq2-runner-collision-observation/1",
        "reset_generation": 0,
        "observation_sequence": 0,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "observed_monotonic_ns": 3,
        "phase": "prepower_reset",
        "disposition": "reset_boundary_discard",
        "boundary": "runner_drain_not_receiver_receipt",
        "collision": {"id": 1, "threat_level": 1, "impulse": 1.0},
    }
    boundary = {
        "schema": "aigp-vq2-calibration-reset-boundary/1",
        "old_generation": 0,
        "new_generation": 1,
        "boundary_monotonic_ns": 3,
        "observations": [copy.deepcopy(heartbeat)],
        "collisions": [copy.deepcopy(collision)],
        "ingress_stats": {
            "generation": 0,
            "next_sequence": 1,
            "highres_imu_received": 0,
            "heartbeat_received": 1,
            "race_status_received": 0,
            "actuator_received": 0,
            "dropped": 0,
            "high_watermark": 1,
            "imu_capacity": 4096,
            "other_capacity": 4096,
            "imu_dropped": 0,
            "other_dropped": 0,
            "imu_high_watermark": 0,
            "other_high_watermark": 1,
            "buffered_imu": 0,
            "buffered_other": 1,
        },
        "collision_stats": {
            "generation": 0,
            "handled": 1,
            "dropped": 0,
            "high_watermark": 1,
            "capacity": 128,
            "buffered": 1,
        },
    }
    records = [
        _event("received_heartbeat", copy.deepcopy(heartbeat)),
        _event("runner_collision_observation", copy.deepcopy(collision)),
        _event("calibration_reset_boundary", copy.deepcopy(boundary)),
    ]
    final_stats = {
        "ingress": {
            **copy.deepcopy(boundary["ingress_stats"]),
            "generation": 1,
            "next_sequence": 0,
            "heartbeat_received": 0,
            "high_watermark": 0,
            "other_high_watermark": 0,
            "buffered_other": 0,
        },
        "collision": {
            **copy.deepcopy(boundary["collision_stats"]),
            "generation": 1,
            "handled": 0,
            "high_watermark": 0,
            "buffered": 0,
        },
    }
    exact = analysis._reset_boundary_resource_evidence(
        records, final_resource_stats=final_stats
    )
    assert exact["accounting_exact"] is True
    assert exact["resource_stats_zero"] is True

    invalid_boundaries = []
    wrong_next = copy.deepcopy(records)
    wrong_next[-1]["observation"]["ingress_stats"]["next_sequence"] = 2
    invalid_boundaries.append(wrong_next)
    wrong_type_count = copy.deepcopy(records)
    wrong_type_count[-1]["observation"]["ingress_stats"]["heartbeat_received"] = 0
    invalid_boundaries.append(wrong_type_count)
    wrong_ingress_sequence = copy.deepcopy(records)
    wrong_ingress_sequence[0]["observation"]["ingress"]["sequence"] = 1
    wrong_ingress_sequence[-1]["observation"]["observations"][0]["ingress"][
        "sequence"
    ] = 1
    invalid_boundaries.append(wrong_ingress_sequence)
    wrong_handled = copy.deepcopy(records)
    wrong_handled[-1]["observation"]["collision_stats"]["handled"] = 2
    wrong_collision_sequence = copy.deepcopy(records)
    wrong_collision_sequence[1]["observation"]["observation_sequence"] = 1
    wrong_collision_sequence[-1]["observation"]["collisions"][0][
        "observation_sequence"
    ] = 1
    invalid_boundaries.append(wrong_collision_sequence)
    for mutated_records in invalid_boundaries:
        with pytest.raises(attempt.PoweredAttemptContractError):
            analysis._reset_boundary_resource_evidence(
                mutated_records, final_resource_stats=final_stats
            )

    result = analysis._reset_boundary_resource_evidence(
        wrong_handled, final_resource_stats=final_stats
    )
    assert result["accounting_exact"] is False
    assert result["resource_stats_zero"] is False

    wrong_final_generation = copy.deepcopy(final_stats)
    wrong_final_generation["ingress"]["generation"] = 2
    wrong_final_generation["collision"]["generation"] = 2
    result = analysis._reset_boundary_resource_evidence(
        records, final_resource_stats=wrong_final_generation
    )
    assert result["accounting_exact"] is False
    assert result["resource_stats_zero"] is False


def test_reset_boundary_drop_counters_survive_generation_reset(replay):
    broken = copy.deepcopy(replay["records"])
    broken.insert(
        0,
        _event(
            "calibration_reset_boundary",
            {
                "schema": "aigp-vq2-calibration-reset-boundary/1",
                "old_generation": 0,
                "new_generation": 1,
                "boundary_monotonic_ns": 1,
                "observations": [],
                "collisions": [],
                "ingress_stats": {
                    "generation": 0,
                    "next_sequence": 2,
                    "highres_imu_received": 2,
                    "heartbeat_received": 0,
                    "race_status_received": 0,
                    "actuator_received": 0,
                    "dropped": 1,
                    "high_watermark": 1,
                    "imu_capacity": 1,
                    "other_capacity": 4096,
                    "imu_dropped": 1,
                    "other_dropped": 0,
                    "imu_high_watermark": 1,
                    "other_high_watermark": 0,
                    "buffered_imu": 0,
                    "buffered_other": 0,
                },
                "collision_stats": {
                    "generation": 0,
                    "handled": 0,
                    "dropped": 0,
                    "high_watermark": 0,
                    "capacity": 128,
                    "buffered": 0,
                },
            },
        ),
    )
    result = _reconcile(_seal_snapshot(broken))
    assert result["valid"] is False
    assert result["checks"]["resource_stats_zero"] is False
    assert result["counts"]["ingress_drops"] == 1
    assert result["capture_counts"]["observation_queue_drops"] == 1


def test_replay_bytes_and_rich_core_schema_are_strict(replay):
    with pytest.raises(analysis.PoweredCalibrationAnalysisError, match="canonical"):
        analysis.validate_replay_snapshot(
            replay["manifest"],
            replay["records"],
            manifest_bytes=replay["manifest_bytes"],
            records_bytes=b" " + replay["records_bytes"],
            frame_blob_file_sha256=replay["frame_blob_file_sha256"],
        )
    widened = copy.deepcopy(replay["records"])
    decoded = next(row for row in widened if row.get("type") == "decoded_frame")
    decoded["forbidden"] = True
    with pytest.raises(analysis.PoweredCalibrationAnalysisError, match="fields must be exact"):
        analysis.validate_replay_snapshot(
            _seal_snapshot(widened)["manifest"],
            widened,
        )


def _collision_run(run_id: str, split: str, decoded: list[str], derivative=None):
    return {
        "run_id": run_id,
        "assigned_split": split,
        "decoded_content_sha256": decoded,
        "derivative_sha256": [] if derivative is None else derivative,
    }


def test_global_collision_is_transitive_and_never_moves_individual_frames():
    f00 = "F00-A01/reset-epoch-1/excitation-1"
    f02 = "F02-A01/reset-epoch-1/excitation-1"
    h1, h2, h3, d1 = (character * 64 for character in "abcd")
    runs = [
        _collision_run(f00, "discovery_fit", [h1], [d1]),
        _collision_run("H01/run", "held_out", [h1, h2]),
        _collision_run("L01/run", "limit", [h2, h3]),
    ]
    joined = analysis.plan_global_content_assignment(
        runs, anchor_run_id=f00, allow_anchor_whole_run_join=True
    )
    assert {row["effective_split"] for row in joined["runs"]} == {"discovery_fit"}
    assert {row["disposition"] for row in joined["runs"]} == {"whole_run_join"}
    assert all(
        row["effective_split"] == "discovery_fit"
        for row in joined["content_groups"]
    )
    assert analysis.validate_global_content_assignment(
        joined,
        source_runs=runs,
        anchor_run_id=f00,
        allow_anchor_whole_run_join=True,
    ) == joined

    excluded = analysis.plan_global_content_assignment(
        runs, anchor_run_id=f00, allow_anchor_whole_run_join=False
    )
    assert {row["effective_split"] for row in excluded["runs"]} == {None}
    assert {row["disposition"] for row in excluded["content_groups"]} == {
        "globally_excluded"
    }
    partial = copy.deepcopy(excluded)
    partial["runs"][1]["effective_split"] = "discovery_fit"
    with pytest.raises(analysis.PoweredCalibrationAnalysisError):
        analysis.validate_global_content_assignment(
            partial,
            source_runs=runs,
            anchor_run_id=f00,
            allow_anchor_whole_run_join=False,
        )
    with pytest.raises(
        analysis.PoweredCalibrationAnalysisError,
        match="F02 is immutable discovery_fit",
    ):
        analysis.plan_global_content_assignment(
            [_collision_run(f02, "held_out", [h1])],
            anchor_run_id=f02,
            allow_anchor_whole_run_join=True,
        )


def test_create_new_publication_prevalidates_roundtrips_and_never_overwrites(
    tmp_path, monkeypatch
):
    root = tmp_path / "evidence"
    root.mkdir()
    first, second = root / "claim.json", root / "registry.json"
    result = analysis.publish_create_new_json_sequence(
        ((first, {"z": 1, "a": "x"}), (second, {"ok": True})),
        publication_root=root,
    )
    assert [Path(item["path"]).name for item in result] == ["claim.json", "registry.json"]
    assert first.read_bytes() == b'{"a":"x","z":1}\n'
    assert second.read_bytes() == b'{"ok":true}\n'
    with pytest.raises(analysis.PoweredCalibrationAnalysisError, match="already exists"):
        analysis.publish_create_new_json_sequence(
            ((first, {"different": True}),), publication_root=root
        )
    assert first.read_bytes() == b'{"a":"x","z":1}\n'

    third = root / "third.json"
    with pytest.raises(analysis.PoweredCalibrationAnalysisError):
        analysis.publish_create_new_json_sequence(
            ((third, {"ok": float("nan")}),), publication_root=root
        )
    assert not third.exists()

    partial = root / "partial.json"

    def fail_write(*_args, **_kwargs):
        raise OSError("injected")

    monkeypatch.setattr(analysis.os, "write", fail_write)
    with pytest.raises(analysis.PartialPublicationError) as caught:
        analysis.publish_create_new_json_sequence(
            ((partial, {"ok": True}),), publication_root=root
        )
    assert caught.value.published == (str(partial),)
    assert partial.exists()


def test_on_disk_bundle_loader_verifies_content_and_rejects_linked_frame(
    tmp_path, monkeypatch
):
    import numpy as np

    from aigp_loop.replay import ReplayBundleWriter

    bundle = tmp_path / "sealed.vq2replay"
    writer = ReplayBundleWriter(
        bundle,
        session_id=attempt.SESSION_ID,
        metadata={"test": "offline-analysis"},
        require_private=False,
    )
    writer.capture_decoded_frame(
        np.zeros((360, 640, 3), dtype=np.uint8),
        generation=1,
        frame_id=1,
        sim_time_ns=10,
        received_monotonic_s=1.0,
    )
    writer.close(outcome={"status": "complete"})
    loaded = analysis.load_and_verify_sealed_bundle(bundle)
    assert loaded["manifest"]["dataset_hash"]
    assert len(loaded["records"]) == 1
    assert loaded["records_bytes"].endswith(b"\n")

    digest = next(iter(loaded["frame_blob_file_sha256"]))
    frame = bundle / "frames" / f"{digest}.npy"
    outside = tmp_path / "original.npy"
    frame.replace(outside)
    try:
        os.symlink(outside, frame)
    except OSError:
        # Some Windows service accounts lack SeCreateSymbolicLinkPrivilege.
        # Exercise the same fail-closed stable-identity boundary
        # deterministically instead of silently removing this safety test.
        outside.replace(frame)
        real_identity = analysis._sha256_regular_file

        def reject_linked_frame(path, *, maximum_bytes, label):
            if Path(path) == frame:
                raise analysis.PoweredCalibrationAnalysisError(
                    f"{label}: simulated reparse identity rejection"
                )
            return real_identity(path, maximum_bytes=maximum_bytes, label=label)

        monkeypatch.setattr(analysis, "_sha256_regular_file", reject_linked_frame)
        with pytest.raises(analysis.PoweredCalibrationAnalysisError):
            analysis.load_and_verify_sealed_bundle(bundle)
        return
    with pytest.raises(analysis.PoweredCalibrationAnalysisError):
        analysis.load_and_verify_sealed_bundle(bundle)
    frame.unlink()
    outside.replace(frame)

    manifest = bundle / "manifest.json"
    outside_manifest = tmp_path / "original-manifest.json"
    manifest.replace(outside_manifest)
    os.symlink(outside_manifest, manifest)
    with pytest.raises(analysis.PoweredCalibrationAnalysisError):
        analysis.load_and_verify_sealed_bundle(bundle)


def _artifact(name: str, digest: str = "6" * 64) -> dict[str, object]:
    return {
        "name": name,
        "path": attempt.EVIDENCE_ROOT
        + rf"\analysis-test-{name.replace('/', '-')}",
        "size_bytes": 1,
        "sha256": digest,
    }


def _prospective_safety(summary: dict[str, object]) -> dict[str, object]:
    names = {
        "bundle_verification",
        "child_cleanup_certificate",
        "legacy_record",
        "replay_manifest",
        "replay_records",
        "runner_stdout",
        "runner_stderr",
        f"replay_frame/{CONTENT_SHA}",
    }
    artifacts = {name: _artifact(name) for name in names}
    identity = {
        "candidate_commit": COMMIT,
        "code_sha256": "7" * 64,
        "live_freeze_sha256": "8" * 64,
        "attempt_context_sha256": CONTEXT_SHA,
        "attempt_envelope_sha256": "9" * 64,
        "target_config_sha256": "a" * 64,
        "capture_authorization_sha256": "b" * 64,
        "excitation_plan_id": attempt.EXCITATION_PLAN_ID,
        "excitation_plan_sha256": attempt.EXCITATION_PLAN_SHA256,
        "training_attestation_sha256": "c" * 64,
        "simulator_process_proof_sha256": "d" * 64,
        "simulator_final_process_proof_sha256": "e" * 64,
        "child_authority_sha256": "f" * 64,
        "cleanup_authority_sha256": None,
        "lease_final_sha256": "0" * 64,
        "bundle_verification_sha256": artifacts["bundle_verification"]["sha256"],
    }
    return {
        "valid": True,
        "invalid_reasons": [],
        "checks": {name: True for name in analysis._REPORT_CHECKS},
        "hashes": {
            "capture_seal": "1" * 64,
            "bundle_verification": artifacts["bundle_verification"]["sha256"],
            "lease_final": identity["lease_final_sha256"],
            "child_cleanup_certificate": artifacts[
                "child_cleanup_certificate"
            ]["sha256"],
        },
        "artifacts": artifacts,
        "capture_seal": {"identity": identity},
    }


def _attempt_test_fixtures():
    source = Path(__file__).with_name("test_aigp_vq2_powered_attempt.py")
    spec = importlib.util.spec_from_file_location("_aigp_attempt_test_fixtures", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MemoryPostReleaseReader:
    def __init__(self):
        self._payloads = {}
        self._artifacts = {}
        self.calls = []

    def put_bytes(self, path, payload):
        payload = bytes(payload)
        artifact = analysis.StablePostReleaseArtifact(
            path=path,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        self._payloads[path] = payload
        self._artifacts[path] = artifact
        return artifact

    def put_json(self, path, value):
        return self.put_bytes(path, attempt.canonical_json_file_bytes(value))

    def put_identity(self, path, *, size_bytes, sha256):
        artifact = analysis.StablePostReleaseArtifact(path, size_bytes, sha256)
        self._artifacts[path] = artifact
        return artifact

    def identify(self, path, *, maximum_bytes):
        self.calls.append(("identify", path, maximum_bytes))
        artifact = self._artifacts[path]
        if artifact.size_bytes > maximum_bytes:
            raise ValueError("test artifact exceeds bound")
        return artifact

    def read(self, path, *, maximum_bytes):
        self.calls.append(("read", path, maximum_bytes))
        artifact = self._artifacts[path]
        if artifact.size_bytes > maximum_bytes:
            raise ValueError("test artifact exceeds bound")
        return analysis.StablePostReleasePayload(artifact, self._payloads[path])


def _process_proof(fixtures, *, phase: str) -> dict[str, object]:
    launcher = fixtures.process(30)
    payload = fixtures.process(40)

    def owner(observed: int) -> dict[str, object]:
        return {
            "observed_monotonic_ns": observed,
            "ipv4_14550": [],
            "ipv6_14550": [],
            "ipv4_5600": [],
            "ipv6_5600": [],
        }

    scheduled_phases = ["before_launch", "after_launcher_return", "before_child"]
    owner_times = [10, 20]
    if phase == "postchild":
        scheduled_phases.append("after_child_or_fallback")
        owner_times.append(30)
    active = []
    if phase == "postchild":
        active = [
            {
                "observed_monotonic_ns": 50 + index,
                "port": port,
                "role": "powered_child",
                "pid": fixtures.process(20)["pid"],
                "creation_filetime_100ns": fixtures.process(20)[
                    "creation_filetime_100ns"
                ],
            }
            for index, port in enumerate((14550, 5600))
        ]
    return {
        "schema": "aigp-vq2-simulator-process-proof/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "phase": phase,
        "observed_at_utc": UTC,
        "observed_monotonic_ns": 100 if phase == "prechild" else 200,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "wrapper_process": fixtures.process(10),
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
            "name": "AIGP-P2-F02-A01-Launch",
            "observations": [
                {
                    "phase": name,
                    "observed_monotonic_ns": index + 3,
                    "query_exit_code": 1,
                    "absent": True,
                }
                for index, name in enumerate(scheduled_phases)
            ],
        },
        "ports": {
            "owner_table_observations": [owner(value) for value in owner_times],
            "active_owner_observations": active,
            "exclusive_probes": [
                {
                    "host": "127.0.0.1",
                    "port": 14550,
                    "started_monotonic_ns": 40,
                    "ended_monotonic_ns": 41,
                    "result": "bound_and_closed",
                },
                {
                    "host": "0.0.0.0",
                    "port": 5600,
                    "started_monotonic_ns": 42,
                    "ended_monotonic_ns": 43,
                    "result": "bound_and_closed",
                },
            ],
            "status": "free",
        },
        "responsive": True,
    }


def _received_for_cleanup(
    fixtures, kind: str, *, generation: int, sequence: int, source: int, received: int
):
    if kind == "heartbeat":
        value = fixtures.heartbeat(sequence=sequence, base_mode=0)
    elif kind == "race":
        value = fixtures.race(value=source, sequence=sequence)
    else:
        value = fixtures.imu(value=source, sequence=sequence)
    value["ingress"]["generation"] = generation
    value["ingress"]["received_monotonic_ns"] = received
    return value


def _reset_boundary() -> dict[str, object]:
    boundary = _replay_reset_boundary(
        1,
        2,
        6_000_000_020,
    )
    boundary["ingress_stats"].update(
        {
            "next_sequence": 14,
            "highres_imu_received": 4,
            "heartbeat_received": 5,
            "race_status_received": 3,
            "actuator_received": 2,
            "high_watermark": 4,
            "imu_high_watermark": 1,
            "other_high_watermark": 3,
        }
    )
    return boundary


def _complete_safety_fixture(replay_snapshot: dict[str, object]):
    fixtures = _attempt_test_fixtures()
    freeze = fixtures.live_freeze()
    envelope = fixtures.attempt()
    context_sha = envelope["context_sha256"]
    candidate_commit = freeze["candidate"]["commit"]

    rows = copy.deepcopy(replay_snapshot["records"])
    generated_by_sequence = {}
    for row in rows:
        if row.get("event") == "calibration_command_generated":
            observation = row["observation"]
            observation["candidate_commit"] = candidate_commit
            observation["attempt_context_sha256"] = context_sha
            generated_by_sequence[observation["event_sequence"]] = observation
    for row in rows:
        if row.get("event") == "calibration_command_sent":
            observation = row["observation"]
            generated = generated_by_sequence[observation["generated_event_sequence"]]
            observation["candidate_commit"] = candidate_commit
            observation["attempt_context_sha256"] = context_sha
            observation["generation_sha256"] = attempt.canonical_object_sha256(generated)
        elif row.get("event") == "calibration_tick_disposition":
            row["observation"]["attempt_context_sha256"] = context_sha
    snapshot = _seal_snapshot(rows)
    summary = _reconcile(snapshot)

    prechild = _process_proof(fixtures, phase="prechild")
    postchild = _process_proof(fixtures, phase="postchild")
    training = {
        "schema": "aigp-vq2-training-mode-attestation/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "attested_at_utc": UTC,
        "attested_monotonic_ns": 201,
        "host_clock_id": attempt.HOST_CLOCK_ID,
        "mode": "Training",
        "method": "post_topology_visual_training_check_challenge",
        "challenge_sha256": "7" * 64,
        "wrapper_process": prechild["wrapper_process"],
        "simulator_process_proof_sha256": attempt.canonical_file_sha256(prechild),
    }
    argv_sha = attempt.canonical_object_sha256(envelope["context"]["child_argv"])
    child_process = fixtures.process(20)
    child_process["argv_sha256"] = argv_sha
    lease_rows = []
    lease_hashes = []

    def append_lease_row(
        event,
        phase,
        observed,
        *,
        child=None,
        release_proved=False,
    ):
        generation = len(lease_rows)
        row = {
            "schema": "aigp-vq2-live-lease-evidence/2",
            "mutex_name": LIVE_LEASE_MUTEX_NAME,
            "attempt_id": attempt.ATTEMPT_ID,
            "attempt_envelope_sha256": attempt.canonical_file_sha256(envelope),
            "attempt_context_sha256": context_sha,
            "generation": generation,
            "predecessor_sha256": None if generation == 0 else lease_hashes[-1],
            "event": event,
            "abandoned": False,
            "owner_role": "wrapper",
            "owner_token_sha256": envelope["capabilities"]["lease_owner_sha256"],
            "wrapper_process": envelope["context"]["wrapper_process"],
            "owner_process": envelope["context"]["wrapper_process"],
            "child_process": child,
            "cleanup_process": None,
            "host_clock_id": attempt.HOST_CLOCK_ID,
            "qpc_frequency_hz": envelope["context"]["host"]["qpc_frequency_hz"],
            "observed_monotonic_ns": observed,
            "phase": phase,
            "orphaned_pending": None,
            "release_proved": release_proved,
        }
        lease_rows.append(row)
        lease_hashes.append(attempt.canonical_file_sha256(row))

    append_lease_row("acquired", "lease_acquire", 100)
    append_lease_row("phase", "child_spawn", 200, child=child_process)
    for second, phase in (
        (1, "child_spawn"),
        (2, "child_supervision"),
        (3, "child_supervision"),
        (4, "child_supervision"),
        (5, "child_supervision"),
        (6, "child_cleanup"),
        (7, "postcheck_identity_process_ports"),
    ):
        append_lease_row(
            "heartbeat",
            phase,
            second * 1_000_000_000 + 100,
            child=child_process,
        )
    append_lease_row(
        "release_intent",
        "lease_release_and_verify",
        7_000_000_200,
        child=child_process,
    )
    append_lease_row(
        "released",
        "lease_release_and_verify",
        7_000_000_201,
        child=child_process,
        release_proved=True,
    )
    lease_record_sha = lease_hashes[1]
    cleanup_lease_generation = 7
    cleanup_lease_sha = lease_hashes[cleanup_lease_generation]
    anchor = 1_000
    authority = {
        "schema": "aigp-vq2-powered-process-authority/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "role": "powered_child",
        "created_at_utc": UTC,
        "created_monotonic_ns": 500,
        "attempt_envelope_sha256": attempt.canonical_file_sha256(envelope),
        "attempt_context_sha256": context_sha,
        "live_freeze_sha256": attempt.canonical_file_sha256(freeze),
        "wrapper_process": envelope["context"]["wrapper_process"],
        "process": child_process,
        "parent_handle": {
            "value": 42,
            "process": envelope["context"]["wrapper_process"],
            "access": "synchronize_query_limited_information",
            "inherited": True,
        },
        "capability_sha256": envelope["capabilities"]["child_sha256"],
        "lease_record_sha256": lease_record_sha,
        "training_attestation_sha256": attempt.canonical_file_sha256(training),
        "simulator_process_proof_sha256": attempt.canonical_file_sha256(prechild),
        "argv_sha256": argv_sha,
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
    authority_sha = attempt.canonical_file_sha256(authority)
    cleanup_time = 6_000_000_000
    heartbeat_before = _received_for_cleanup(
        fixtures,
        "heartbeat",
        generation=1,
        sequence=10,
        source=0,
        received=cleanup_time + 1,
    )
    heartbeat_before["heartbeat"]["base_mode"] = 128
    heartbeat_after = _received_for_cleanup(
        fixtures,
        "heartbeat",
        generation=1,
        sequence=11,
        source=0,
        received=cleanup_time + 10,
    )
    baseline_race = _received_for_cleanup(
        fixtures,
        "race",
        generation=1,
        sequence=12,
        source=50,
        received=cleanup_time + 11,
    )
    baseline_imu = _received_for_cleanup(
        fixtures,
        "imu",
        generation=1,
        sequence=13,
        source=50,
        received=cleanup_time + 12,
    )
    race_one = _received_for_cleanup(
        fixtures, "race", generation=2, sequence=0, source=60, received=cleanup_time + 30
    )
    imu_one = _received_for_cleanup(
        fixtures, "imu", generation=2, sequence=1, source=60, received=cleanup_time + 31
    )
    race_two = _received_for_cleanup(
        fixtures, "race", generation=2, sequence=2, source=70, received=cleanup_time + 32
    )
    imu_two = _received_for_cleanup(
        fixtures, "imu", generation=2, sequence=3, source=70, received=cleanup_time + 33
    )
    final_heartbeat = _received_for_cleanup(
        fixtures,
        "heartbeat",
        generation=2,
        sequence=4,
        source=0,
        received=cleanup_time + 34,
    )
    receipts = summary["outbound_receipts"]
    disarm_receipt = next(item for item in receipts if item.get("category") == "disarm")
    reset_receipt = next(
        item for item in reversed(receipts) if item.get("category") == "sim_reset"
    )
    reset_epoch = {
        "ingress_generation": 2,
        "race_anchor_boot_ms": 5,
        "imu_anchor_usec": 5,
    }
    bind_base = {
        "family": "AF_INET",
        "socket_policy": "ipv4-exclusive-address-use",
        "owner_process": child_process,
    }
    certificate = {
        "schema": "aigp-vq2-powered-cleanup-certificate/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "producer_role": "powered_child",
        "cleanup_epoch": "child-cleanup-0",
        "authority": {
            "process_authority": {
                "path": attempt.frozen_paths()["child_authority"],
                "sha256": authority_sha,
            },
            "attempt_context_sha256": context_sha,
            "attempt_envelope_sha256": attempt.canonical_file_sha256(envelope),
            "producer": child_process,
        },
        "trigger": "normal_completion",
        "started_monotonic_ns": cleanup_time - 1,
        "deadline_monotonic_ns": cleanup_time + 1_000_000_000,
        "completed_monotonic_ns": cleanup_time + 100,
        "parent_state": {
            "mode": "live_delegation",
            "wrapper_process": envelope["context"]["wrapper_process"],
            "observed_monotonic_ns": 1,
            "takeover_completed_monotonic_ns": None,
            "takeover_lease_record_sha256": None,
        },
        "lease": {
            "owner_role": "wrapper",
            "generation": cleanup_lease_generation,
            "record_sha256": cleanup_lease_sha,
            "authority_valid": True,
        },
        "phase_deadlines": [],
        "endpoints": {
            "mavlink": {
                "state": "closed_with_peer",
                "bind": {
                    **bind_base,
                    "role": "mavlink",
                    "requested": {"host": "127.0.0.1", "port": 14550},
                    "actual": {"host": "127.0.0.1", "port": 14550},
                },
                "frozen_peer": {"host": "127.0.0.1", "port": 14551},
                "rejected_source_count": 0,
            },
            "camera": {
                "state": "closed_with_peer",
                "bind": {
                    **bind_base,
                    "role": "camera",
                    "requested": {"host": "0.0.0.0", "port": 5600},
                    "actual": {"host": "0.0.0.0", "port": 5600},
                },
                "frozen_peer": {"host": "127.0.0.1", "port": 5601},
                "rejected_source_count": 0,
            },
        },
        "outbound_receipts": receipts,
        "zero_command": {
            "state": "returned",
            "required": True,
            "requested": summary["cleanup_generated"][0]["command"],
            "generated": summary["cleanup_generated"][0],
            "terminal": summary["cleanup_sent"][0],
            "outbound_receipt": summary["cleanup_sent"][0]["transport"]["receipt"],
        },
        "disarm": {
            "state": "confirmed",
            "request_monotonic_ns": disarm_receipt["call_start_monotonic_ns"],
            "receipt": disarm_receipt,
            "heartbeat_before": heartbeat_before,
            "heartbeat_after": heartbeat_after,
            "newer_confirmed": True,
        },
        "reset": {
            "state": "confirmed",
            "request_monotonic_ns": cleanup_time + 20,
            "receipt": reset_receipt,
            "boundary": _reset_boundary(),
            "baseline": {"race": baseline_race, "imu": baseline_imu},
            "clean_epoch": reset_epoch,
            "advancing_race": [race_one, race_two],
            "advancing_imu": [imu_one, imu_two],
            "rollback_and_advance_confirmed": True,
        },
        "collisions": {"observations": [], "invalidating_occurrence_count": 0},
        "final_state": {
            "state": "confirmed",
            "heartbeat": final_heartbeat,
            "disarmed": True,
            "reset_epoch": reset_epoch,
            "last_race": race_two,
            "last_imu": imu_two,
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
    certificate_sha = attempt.canonical_file_sha256(certificate)
    process_result = {
        "schema": "aigp-vq2-powered-process-result/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "producer_role": "powered_child",
        "process_authority_sha256": authority_sha,
        "started_monotonic_ns": 500,
        "completed_monotonic_ns": cleanup_time + 200,
        "outcome": "completed",
        "reason_codes": [],
        "phase_deadlines": [],
        "cleanup_certificate": {
            "path": attempt.frozen_paths()["child_cleanup_certificate"],
            "state": "published",
            "sha256": certificate_sha,
        },
        "outbound_audit": summary["outbound_audit"],
        "artifacts": {
            "legacy_record": {
                "path": attempt.frozen_paths()["legacy_record"],
                "state": "closed",
                "sha256": "9" * 64,
            },
            "replay_bundle": {
                "path": attempt.frozen_paths()["replay_bundle"],
                "state": "closed",
                "dataset_hash": summary["manifest"]["dataset_hash"],
                "manifest_sha256": summary["bundle_artifacts"]["manifest_sha256"],
                "records_sha256": summary["bundle_artifacts"]["records_sha256"],
            },
        },
    }
    lease = {
        "schema": "aigp-vq2-live-lease-ledger/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "attempt_envelope_sha256": attempt.canonical_file_sha256(envelope),
        "records": [
            {
                "generation": generation,
                "path": attempt.frozen_paths()["lease_directory"]
                + rf"\generation-{generation:06d}.json",
                "sha256": lease_hashes[generation],
                "event": row["event"],
            }
            for generation, row in enumerate(lease_rows)
        ],
        "orphaned_pending_files": [],
        "final_generation": len(lease_rows) - 1,
        "final_record_sha256": lease_hashes[-1],
        "release_proved": True,
    }
    lease_rows_by_path = {
        entry["path"]: copy.deepcopy(lease_rows[entry["generation"]])
        for entry in lease["records"]
    }
    manifest_ref = {
        "name": "replay_manifest",
        "path": attempt.frozen_paths()["replay_bundle"] + r"\manifest.json",
        "size_bytes": summary["bundle_artifacts"]["manifest_size_bytes"],
        "sha256": summary["bundle_artifacts"]["manifest_sha256"],
    }
    records_ref = {
        "name": "replay_records",
        "path": attempt.frozen_paths()["replay_bundle"] + r"\records.jsonl",
        "size_bytes": summary["bundle_artifacts"]["records_size_bytes"],
        "sha256": summary["bundle_artifacts"]["records_sha256"],
    }
    frame_ref = {
        "name": f"replay_frame/{CONTENT_SHA}",
        "path": attempt.frozen_paths()["replay_bundle"]
        + rf"\frames\{CONTENT_SHA}.npy",
        "size_bytes": 1,
        "sha256": FRAME_FILE_SHA,
    }
    verification = {
        "schema": "aigp-vq2-replay-bundle-verification/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "verified_at_utc": UTC,
        "verified_monotonic_ns": 20,
        "timing": fixtures.timing("bundle_verify", prepared=20),
        "identity": {
            "candidate_commit": candidate_commit,
            "live_freeze_sha256": attempt.canonical_file_sha256(freeze),
            "attempt_context_sha256": context_sha,
            "attempt_envelope_sha256": attempt.canonical_file_sha256(envelope),
            "child_authority_sha256": authority_sha,
            "child_process_result_sha256": attempt.canonical_file_sha256(
                process_result
            ),
            "child_cleanup_certificate_sha256": certificate_sha,
            "lease_final_sha256": attempt.canonical_file_sha256(lease),
        },
        "bundle": {
            "path": attempt.frozen_paths()["replay_bundle"],
            "dataset_hash": summary["manifest"]["dataset_hash"],
            "manifest": manifest_ref,
            "records": records_ref,
            "frames": [frame_ref],
        },
        "checks": {name: True for name in attempt._BUNDLE_CHECKS},
        "valid": True,
    }
    digest_by_name = {
        "live_freeze": attempt.canonical_file_sha256(freeze),
        "implementation_inventory": freeze["candidate"]["implementation_inventory"][
            "sha256"
        ],
        "environment_inventory": freeze["runtime"]["environment_inventory"]["sha256"],
        "import_inventory": freeze["runtime"]["import_inventory"]["sha256"],
        "attempt_envelope": attempt.canonical_file_sha256(envelope),
        "training_attestation": attempt.canonical_file_sha256(training),
        "process_prechild": attempt.canonical_file_sha256(prechild),
        "process_postchild": attempt.canonical_file_sha256(postchild),
        "child_authority": authority_sha,
        "child_cleanup_certificate": certificate_sha,
        "lease_final": attempt.canonical_file_sha256(lease),
        "bundle_verification": attempt.canonical_file_sha256(verification),
        "runner_stdout": attempt.canonical_file_sha256(process_result),
        "runner_stderr": "a" * 64,
        "legacy_record": "9" * 64,
    }
    artifacts = [
        _artifact(name, digest)
        for name, digest in digest_by_name.items()
    ] + [manifest_ref, records_ref, frame_ref]
    artifacts.sort(key=lambda item: item["name"].encode())
    stats = {name: 0 for name in attempt._CAPTURE_STATS}
    stats.update(summary["capture_counts"])
    cleanup = {
        "child_certificate_sha256": certificate_sha,
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
    seal = {
        "schema": "aigp-vq2-powered-calibration-capture-seal/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "sealed_at_utc": UTC,
        "timing": fixtures.timing("capture_seal"),
        "identity": {
            "candidate_commit": candidate_commit,
            "code_sha256": freeze["candidate"]["code_sha256"],
            "live_freeze_sha256": attempt.canonical_file_sha256(freeze),
            "attempt_context_sha256": context_sha,
            "attempt_envelope_sha256": attempt.canonical_file_sha256(envelope),
            "target_config_sha256": envelope["context"]["target_config"]["sha256"],
            "capture_authorization_sha256": envelope["context"][
                "capture_authorization"
            ]["sha256"],
            "excitation_plan_id": attempt.EXCITATION_PLAN_ID,
            "excitation_plan_sha256": attempt.EXCITATION_PLAN_SHA256,
            "training_attestation_sha256": attempt.canonical_file_sha256(training),
            "simulator_process_proof_sha256": attempt.canonical_file_sha256(prechild),
            "simulator_final_process_proof_sha256": attempt.canonical_file_sha256(
                postchild
            ),
            "child_authority_sha256": authority_sha,
            "cleanup_authority_sha256": None,
            "lease_final_sha256": attempt.canonical_file_sha256(lease),
            "bundle_verification_sha256": attempt.canonical_file_sha256(verification),
        },
        "artifacts": artifacts,
        "capture_stats": stats,
        "outbound_audit": summary["outbound_audit"],
        "cleanup": cleanup,
    }
    return summary, {
        "live_freeze": freeze,
        "attempt_envelope": envelope,
        "bundle_verification": verification,
        "capture_seal": seal,
        "child_authority": authority,
        "child_process_result": process_result,
        "child_cleanup_certificate": certificate,
        "lease_final": lease,
        "training_attestation": training,
        "process_prechild": prechild,
        "process_postchild": postchild,
    }, lease_rows_by_path


def _install_in_memory_lease(monkeypatch, lease_rows):
    monkeypatch.setattr(
        analysis,
        "validate_powered_live_lease_ledger",
        validate_powered_live_lease_index,
    )
    monkeypatch.setattr(
        analysis,
        "load_powered_live_lease_record",
        lambda path: copy.deepcopy(lease_rows[str(path)]),
    )


def test_production_postrelease_service_runs_ordered_in_memory_pipeline(
    replay, monkeypatch
):
    fixtures = _attempt_test_fixtures()
    _summary, evidence, lease_rows = _complete_safety_fixture(replay)
    _install_in_memory_lease(monkeypatch, lease_rows)

    implementation = {
        "schema": "aigp-vq2-powered-implementation-inventory/1",
        "commit": fixtures.COMMIT,
        "tree": "e" * 40,
        "entries": [
            {"path": "scripts/a.py", "size_bytes": 1, "sha256": "a" * 64},
            {"path": "tests/test_a.py", "size_bytes": 2, "sha256": "b" * 64},
        ],
    }
    environment = {
        "schema": "aigp-vq2-powered-environment-inventory/1",
        "created_at_utc": UTC,
        "variables": [
            {"name": "PATH", "defined": True, "value_sha256": "a" * 64},
            {"name": "TEMP", "defined": True, "value_sha256": "b" * 64},
        ],
    }
    imports = {
        "schema": "aigp-vq2-powered-import-inventory/1",
        "python_sha256": "a" * 64,
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
                "module": "scripts",
                "origin": None,
                "size_bytes": None,
                "sha256": None,
                "root_class": "namespace",
                "namespace_roots": [fixtures.LIVE_WORKTREE + r"\scripts"],
            },
            {
                "module": "scripts.aigp_vq2_powered_attempt",
                "origin": fixtures.LIVE_WORKTREE
                + r"\scripts\aigp_vq2_powered_attempt.py",
                "size_bytes": 1,
                "sha256": "b" * 64,
                "root_class": "candidate",
                "namespace_roots": [],
            },
        ],
    }
    freeze = copy.deepcopy(evidence["live_freeze"])
    implementation_path = freeze["candidate"]["implementation_inventory"]["path"]
    environment_path = freeze["runtime"]["environment_inventory"]["path"]
    import_path = freeze["runtime"]["import_inventory"]["path"]
    freeze["candidate"]["implementation_inventory"]["sha256"] = (
        attempt.canonical_file_sha256(implementation)
    )
    freeze["candidate"]["code_sha256"] = attempt.canonical_object_sha256(
        {name: implementation[name] for name in ("commit", "tree", "entries")}
    )
    freeze["runtime"]["environment_inventory"]["sha256"] = (
        attempt.canonical_file_sha256(environment)
    )
    freeze["runtime"]["import_inventory"]["sha256"] = (
        attempt.canonical_file_sha256(imports)
    )
    assert attempt.validate_live_freeze(
        freeze, implementation_inventory=implementation
    ) == freeze
    evidence["live_freeze"] = freeze

    rows = copy.deepcopy(replay["records"])
    generated_by_sequence = {}
    envelope = evidence["attempt_envelope"]
    for row in rows:
        if row.get("event") == "calibration_command_generated":
            observation = row["observation"]
            observation["candidate_commit"] = freeze["candidate"]["commit"]
            observation["attempt_context_sha256"] = envelope["context_sha256"]
            generated_by_sequence[observation["event_sequence"]] = observation
    for row in rows:
        if row.get("event") == "calibration_command_sent":
            observation = row["observation"]
            generated = generated_by_sequence[observation["generated_event_sequence"]]
            observation["candidate_commit"] = freeze["candidate"]["commit"]
            observation["attempt_context_sha256"] = envelope["context_sha256"]
            observation["generation_sha256"] = attempt.canonical_object_sha256(
                generated
            )
        elif row.get("event") == "calibration_tick_disposition":
            row["observation"]["attempt_context_sha256"] = envelope["context_sha256"]
    snapshot = _seal_snapshot(rows)
    resource_stats = snapshot["manifest"]["outcome"][
        "powered_capture_resource_stats"
    ]
    resource_stats["vision"]["timing_ledger_entries"] = 0
    snapshot["manifest"]["outcome"] = {
        "powered_stage_completed": True,
        "cleanup_certificate_outcome": "proved",
        "reason_codes": [],
        "vision_capture_stats": {
            name: value
            for name, value in resource_stats["vision"].items()
            if name != "constructed"
        },
        "powered_capture_resource_stats": resource_stats,
    }
    snapshot["manifest"]["dataset_hash"] = attempt.canonical_object_sha256(
        {
            "schema": analysis.REPLAY_BUNDLE_SCHEMA,
            "session_id": snapshot["manifest"]["session_id"],
            "started_at": snapshot["manifest"]["started_at"],
            "finished_at": snapshot["manifest"]["finished_at"],
            "metadata": snapshot["manifest"]["metadata"],
            "outcome": snapshot["manifest"]["outcome"],
            "records_sha256": snapshot["manifest"]["integrity"]["records_sha256"],
            "frame_blob_file_sha256": snapshot["manifest"]["integrity"][
                "frame_blob_file_sha256"
            ],
        }
    )
    snapshot["manifest_bytes"] = attempt.canonical_json_file_bytes(
        snapshot["manifest"]
    )
    process_result = copy.deepcopy(evidence["child_process_result"])
    process_result["artifacts"]["replay_bundle"].update(
        {
            "dataset_hash": snapshot["manifest"]["dataset_hash"],
            "manifest_sha256": hashlib.sha256(
                snapshot["manifest_bytes"]
            ).hexdigest(),
            "records_sha256": hashlib.sha256(snapshot["records_bytes"]).hexdigest(),
        }
    )
    assert attempt.validate_process_result(
        process_result, cleanup_certificate=evidence["child_cleanup_certificate"]
    ) == process_result
    evidence["child_process_result"] = process_result

    def passthrough(value, *args, **kwargs):
        return copy.deepcopy(value)

    # The fixture's original envelope/authority bind its original live-freeze
    # hash.  These two test-only seams isolate post-release ordering from the
    # already-covered live construction validators.
    monkeypatch.setattr(attempt, "validate_attempt", passthrough)
    monkeypatch.setattr(attempt, "validate_process_authority", passthrough)

    paths = attempt.frozen_paths()
    reader = _MemoryPostReleaseReader()
    for path, value in (
        (implementation_path, implementation),
        (paths["live_freeze"], freeze),
        (environment_path, environment),
        (import_path, imports),
        (paths["attempt_envelope"], envelope),
        (paths["child_authority"], evidence["child_authority"]),
        (paths["child_cleanup_certificate"], evidence["child_cleanup_certificate"]),
        (paths["runner_stdout"], process_result),
        (paths["process_proof"], evidence["process_prechild"]),
        (paths["process_final_proof"], evidence["process_postchild"]),
        (paths["training_attestation"], evidence["training_attestation"]),
        (paths["lease_final"], evidence["lease_final"]),
    ):
        reader.put_json(path, value)
    reader.put_identity(paths["runner_stderr"], size_bytes=0, sha256="a" * 64)
    reader.put_identity(paths["legacy_record"], size_bytes=1, sha256="9" * 64)
    manifest_path = paths["replay_bundle"] + r"\manifest.json"
    records_path = paths["replay_bundle"] + r"\records.jsonl"
    frame_path = paths["replay_bundle"] + rf"\frames\{CONTENT_SHA}.npy"
    reader.put_bytes(manifest_path, snapshot["manifest_bytes"])
    reader.put_bytes(records_path, snapshot["records_bytes"])
    reader.put_identity(frame_path, size_bytes=1, sha256=FRAME_FILE_SHA)

    def stable_proof(path):
        artifact = reader._artifacts[path]
        return {
            "path": path,
            "final_path": path,
            "volume_id": "test-volume",
            "file_id": path,
            "file_state": "retained-stable",
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "regular_file": True,
            "non_reparse": True,
            "hardlink_count_one": True,
            "owner_is_current_user": True,
            "current_user_only_dacl": True,
            "stable_before_after": True,
            "readback_twice_equal": True,
            "retained_handle": True,
        }

    supervision = {
        "stable_files": {
            path: stable_proof(path)
            for path in (
                paths["child_authority"],
                paths["child_cleanup_certificate"],
                paths["runner_stdout"],
                paths["runner_stderr"],
            )
        },
        "process_results": {"child": copy.deepcopy(process_result)},
        "cleanup_certificates": {
            "child": copy.deepcopy(evidence["child_cleanup_certificate"])
        },
        "tree_exit": {
            "child": {
                "state": "exited",
                "root_process": copy.deepcopy(evidence["child_authority"]["process"]),
                "observations": [{"root_signaled": True, "active_pids": []}],
                "natural_exit_proved": True,
                "termination_attempted": False,
                "termination_returned": None,
                "termination_is_cleanup_proof": False,
            }
        },
    }
    claim = {"kind": "claim"}
    registry = {"kind": "registry"}
    report = {"kind": "report"}
    prospective = {
        "claim": claim,
        "registry": registry,
        "report": report,
        "hashes": {
            "claim": attempt.canonical_file_sha256(claim),
            "registry": attempt.canonical_file_sha256(registry),
            "report": attempt.canonical_file_sha256(report),
        },
    }
    inputs = analysis.PostReleaseInputs(
        live_freeze_path=paths["live_freeze"],
        implementation_inventory_path=implementation_path,
        environment_inventory_path=environment_path,
        import_inventory_path=import_path,
        paths=paths,
        supervision_snapshot=lambda: copy.deepcopy(supervision),
    )
    loader_snapshot = {**snapshot, "bundle_path": paths["replay_bundle"]}
    service = analysis.ProductionPostReleaseService(
        inputs,
        now_ns=lambda: 20,
        utc_now=lambda: UTC,
        reader=reader,
        bundle_loader=lambda path: copy.deepcopy(loader_snapshot),
        reconciler=lambda *args, **kwargs: {"valid": True, "source": "replay"},
        safety_validator=lambda *args, **kwargs: {"valid": True, "source": "safety"},
        prospective_builder=lambda *args, **kwargs: copy.deepcopy(prospective),
        prospective_validator=lambda value: copy.deepcopy(value),
        split_publications_factory=lambda **values: SimpleNamespace(**values),
    )
    assert reader.calls == []
    with pytest.raises(analysis.PostReleaseServiceError, match="out of order"):
        service.build_capture_seal(
            phase_deadline=fixtures.phase_deadline("capture_seal")
        )
    assert reader.calls == []

    verification = service.verify_bundle(
        phase_deadline=fixtures.phase_deadline("bundle_verify")
    )
    verification_artifact = reader.put_json(
        paths["bundle_verification"], verification
    )
    seal = service.build_capture_seal(
        phase_deadline=fixtures.phase_deadline("capture_seal")
    )
    seal_artifact = reader.put_json(paths["capture_seal"], seal)
    analyzed = service.analyze_capture(
        phase_deadline=fixtures.phase_deadline("analysis")
    )
    split = service.publish_split(
        analysis=analyzed,
        phase_deadline=fixtures.phase_deadline("split_publish"),
    )
    assert split.claim == claim
    assert split.registry == registry
    assert split.report == report

    lifecycle_event = {
        "schema": "aigp-vq2-powered-wrapper-event/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "event_sequence": 0,
        "predecessor_sha256": None,
        "event": "phase_end",
        "phase": "attempt_publish",
        "observed_monotonic_ns": 100,
        "duration_ns": attempt.DEADLINE_DURATIONS_NS["attempt_publish"],
        "parent_deadline_monotonic_ns": 10_000_000_000,
        "deadline_monotonic_ns": 2_000_000_100,
        "outcome": "completed",
        "reason_code": None,
        "artifacts": [],
    }
    lifecycle_hash = attempt.canonical_file_sha256(lifecycle_event)
    lifecycle = {
        "schema": "aigp-vq2-powered-wrapper-lifecycle/1",
        "task_id": attempt.TASK_ID,
        "session_id": attempt.SESSION_ID,
        "attempt_id": attempt.ATTEMPT_ID,
        "records": [
            {
                "event_sequence": 0,
                "path": paths["wrapper_ledger_directory"] + r"\event-000000.json",
                "sha256": lifecycle_hash,
                "event": "phase_end",
                "phase": "attempt_publish",
                "observed_monotonic_ns": 100,
                "outcome": "completed",
                "reason_code": None,
                "artifacts": [],
            }
        ],
        "final_sequence": 0,
        "final_record_sha256": lifecycle_hash,
        "live_contact_deadline_monotonic_ns": 300_000_000_100,
        "total_deadline_monotonic_ns": 390_000_000_100,
    }
    attempt.validate_wrapper_lifecycle(lifecycle, ledger_events=[lifecycle_event])

    def receipt(path, value, artifact=None):
        payload = attempt.canonical_json_file_bytes(value)
        return SimpleNamespace(
            path=path,
            size_bytes=len(payload) if artifact is None else artifact.size_bytes,
            sha256=(
                hashlib.sha256(payload).hexdigest()
                if artifact is None
                else artifact.sha256
            ),
        )

    complete = service.build_complete_terminal(
        context={
            "phase": "terminal_ready",
            "utc": UTC,
            "reason_codes": (),
            "fallback_used": False,
            "lease_release_proved": True,
            "wrapper_alive": True,
            "cleanup_state": {
                "child_exit": "proved",
                "fallback": "not_required",
                "ports": "free",
                "lease": "released",
                "processes": "exited",
                "transport": "closed",
                "scheduled_task": "absent",
                "simulator_topology": "unchanged",
                "simulator_responsive": "yes",
            },
            "bundle": verification,
            "seal": seal,
            "split": split,
            "lifecycle": lifecycle,
            "artifacts": {
                "bundle_verification": receipt(
                    paths["bundle_verification"], verification, verification_artifact
                ),
                "capture_seal": receipt(paths["capture_seal"], seal, seal_artifact),
                "split_claim": receipt(paths["split_claim"], claim),
                "split_registry": receipt(paths["split_registry"], registry),
                "analysis_report": receipt(paths["analysis_report"], report),
            },
            "publication_timing": fixtures.timing(
                "terminal_publish", prepared=20
            ),
            "completed_monotonic_ns": 20,
        }
    )
    assert complete["schema"] == "aigp-vq2-powered-calibration-attempt-complete/1"
    assert complete["publication_timing"]["phase"] == "terminal_publish"
    assert complete["artifact_hashes"]["capture_seal_sha256"] == (
        seal_artifact.sha256
    )


def test_safety_evidence_cross_binds_cleanup_process_lease_and_bundle(
    replay, monkeypatch
):
    summary, evidence, lease_rows = _complete_safety_fixture(replay)
    _install_in_memory_lease(monkeypatch, lease_rows)
    result = analysis.validate_safety_evidence(summary, **evidence)
    assert result["valid"] is True
    assert all(result["checks"].values())

    tampered = copy.deepcopy(evidence)
    tampered["capture_seal"]["artifacts"] = [
        {
            **item,
            "sha256": "f" * 64,
        }
        if item["name"] == "replay_records"
        else item
        for item in tampered["capture_seal"]["artifacts"]
    ]
    invalid = analysis.validate_safety_evidence(summary, **tampered)
    assert invalid["valid"] is False
    assert invalid["checks"]["identity_bound"] is False


def test_safety_lease_reader_rejects_release_after_excessive_heartbeat_gap(
    replay, monkeypatch
):
    _summary, evidence, lease_rows = _complete_safety_fixture(replay)
    lease = copy.deepcopy(evidence["lease_final"])
    rows = [copy.deepcopy(lease_rows[item["path"]]) for item in lease["records"]]
    release_intent_generation = next(
        index for index, row in enumerate(rows) if row["event"] == "release_intent"
    )
    rows[release_intent_generation]["observed_monotonic_ns"] = 9_000_000_000
    rows[release_intent_generation + 1]["observed_monotonic_ns"] = 9_000_000_001
    for generation in range(release_intent_generation, len(rows)):
        rows[generation]["predecessor_sha256"] = (
            None
            if generation == 0
            else attempt.canonical_file_sha256(rows[generation - 1])
        )
        lease["records"][generation]["sha256"] = attempt.canonical_file_sha256(
            rows[generation]
        )
    lease["final_record_sha256"] = lease["records"][-1]["sha256"]
    rebound = {
        entry["path"]: rows[entry["generation"]] for entry in lease["records"]
    }
    _install_in_memory_lease(monkeypatch, rebound)

    with pytest.raises(analysis.PoweredCalibrationAnalysisError, match="maximum gap"):
        analysis._load_acceptance_lease_records(
            lease,
            envelope=evidence["attempt_envelope"],
            authority=evidence["child_authority"],
            certificate=evidence["child_cleanup_certificate"],
        )


def test_safety_lease_reader_rejects_orphaned_final_index(replay, monkeypatch):
    _summary, evidence, lease_rows = _complete_safety_fixture(replay)
    lease = copy.deepcopy(evidence["lease_final"])
    lease["orphaned_pending_files"] = [
        {
            "path": attempt.frozen_paths()["lease_directory"]
            + r"\pending-generation-000011-wrapper.json",
            "size_bytes": 1,
            "sha256": "e" * 64,
            "owner_role": "wrapper",
        }
    ]
    _install_in_memory_lease(monkeypatch, lease_rows)

    with pytest.raises(
        analysis.PoweredCalibrationAnalysisError,
        match="successful wrapper lineage",
    ):
        analysis._load_acceptance_lease_records(
            lease,
            envelope=evidence["attempt_envelope"],
            authority=evidence["child_authority"],
            certificate=evidence["child_cleanup_certificate"],
        )


def test_prospective_claim_registry_report_are_non_circular_and_uncomputed(replay, tmp_path):
    summary = _reconcile(replay)
    safety = _prospective_safety(summary)
    deadline = {
        "phase": "split_publish",
        "started_monotonic_ns": 100,
        "duration_ns": attempt.DEADLINE_DURATIONS_NS["split_publish"],
        "parent_deadline_monotonic_ns": 6_000_000_000,
        "deadline_monotonic_ns": 5_000_000_100,
    }
    prospective = analysis.build_prospective_publications(
        summary,
        safety,
        split_phase_deadline=deadline,
        claim_path=attempt.frozen_paths()["split_claim"],
        registry_path=attempt.frozen_paths()["split_registry"],
        claimed_at_utc=UTC,
        claimed_monotonic_ns=101,
        published_at_utc=UTC,
        published_monotonic_ns=102,
        generated_at_utc=UTC,
        report_prepared_monotonic_ns=103,
    )
    assert analysis.validate_prospective_publications(prospective) == prospective
    report = prospective["report"]
    assert set(report["calibration_status"].values()) == {"uncomputed"}
    assert report["split"]["claim_sha256"] == prospective["hashes"]["claim"]
    assert report["split"]["registry_sha256"] == prospective["hashes"]["registry"]
    claim_bytes = attempt.canonical_json_bytes(prospective["claim"])
    assert prospective["hashes"]["registry"].encode() not in claim_bytes
    assert prospective["hashes"]["report"].encode() not in claim_bytes

    tampered = copy.deepcopy(prospective)
    tampered["report"]["calibration_status"]["rank"] = "computed"
    with pytest.raises(analysis.PoweredCalibrationAnalysisError):
        analysis.validate_prospective_publications(tampered)

    failed = copy.deepcopy(summary)
    failed["valid"] = False
    root = tmp_path / "none"
    root.mkdir()
    with pytest.raises(analysis.PoweredCalibrationAnalysisError):
        analysis.build_prospective_publications(
            failed,
            safety,
            split_phase_deadline=deadline,
            claim_path=attempt.frozen_paths()["split_claim"],
            registry_path=attempt.frozen_paths()["split_registry"],
            claimed_at_utc=UTC,
            claimed_monotonic_ns=101,
            published_at_utc=UTC,
            published_monotonic_ns=102,
            generated_at_utc=UTC,
            report_prepared_monotonic_ns=103,
        )
    assert list(root.iterdir()) == []


def test_semantic_clarifications_are_resolved_and_module_has_no_fit_surface():
    assert analysis.semantic_ambiguities() == analysis.SEMANTIC_AMBIGUITIES
    assert analysis.semantic_ambiguities() == ()
    exported = set(analysis.__all__)
    assert not any(
        term in name.lower()
        for name in exported
        for term in ("fit", "jacobian", "svd", "covariance")
    )
