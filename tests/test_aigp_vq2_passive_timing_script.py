"""Offline tests for the passive VQ2 timing-bundle analyzer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aigp_loop._util import canonical_json, json_hash, sha256_file
from aigp_loop.replay import ReplayBundleWriter
from competition.adapter import IMUData
from competition.vq2_capture import MavlinkIngressV1, ReceivedIMUSampleV1
from competition.vq2_contracts import FrameIdentityV1, FrameTimingV1
from competition.vq2_passive_timing import CameraFrameTimingObservationV1
from scripts.aigp_vq2_passive_timing import analyze_bundle, main


def _timing(index: int) -> FrameTimingV1:
    base = 1_000_000_000 + index * 33_333_333
    return FrameTimingV1(
        identity=FrameIdentityV1(
            stream_id="vq2-camera-udp-5600",
            generation=2,
            frame_id=100 + index,
        ),
        camera_source_time_ns=5_000_000_000 + index * 33_333_333,
        host_clock_id="host-perf-counter",
        publication_sequence=10 + index,
        first_unique_packet_monotonic_ns=base,
        final_unique_packet_monotonic_ns=base + 100,
        reassembly_complete_monotonic_ns=base + 300,
        decode_start_monotonic_ns=base + 300,
        decode_end_monotonic_ns=base + 600,
        publish_monotonic_ns=base + 1_000,
    )


def _observation(timing: FrameTimingV1) -> CameraFrameTimingObservationV1:
    consume = timing.publish_monotonic_ns + 500
    return CameraFrameTimingObservationV1(
        frame_timing=timing,
        consume_monotonic_ns=consume,
        work_start_monotonic_ns=consume,
        detection_start_monotonic_ns=consume,
        detection_end_monotonic_ns=consume + 2_000,
        tracking_start_monotonic_ns=consume + 2_000,
        tracking_end_monotonic_ns=consume + 3_000,
        work_end_monotonic_ns=consume + 4_000,
    )


def _ingress(
    sequence: int,
    message_type: str,
    *,
    source_time_value: int | None,
    source_time_unit: str | None,
) -> MavlinkIngressV1:
    return MavlinkIngressV1(
        stream_id="vq2-mavlink-udp-14550",
        generation=3,
        sequence=sequence,
        message_type=message_type,
        host_clock_id="host-perf-counter",
        received_monotonic_ns=2_000_000_000 + sequence * 10_000_000,
        source_time_value=source_time_value,
        source_time_unit=source_time_unit,
    )


def _received_imu(sequence: int, source_us: int) -> ReceivedIMUSampleV1:
    return ReceivedIMUSampleV1(
        ingress=_ingress(
            sequence,
            "HIGHRES_IMU",
            source_time_value=source_us,
            source_time_unit="us",
        ),
        imu=IMUData(
            timestamp_us=source_us,
            accel=(123.456, -234.567, 345.678),
            gyro=(4.321, -5.432, 6.543),
            mag=None,
        ),
    )


def _metadata() -> dict:
    return {
        "simulator_build": "3385",
        "simulator_mode": "Training",
        "simulator_mode_basis": "operator-attested-2026-07-20",
        "stage": "preflight",
        "preflight_healthy_dwell_s": 5.0,
        "capture_kind": "private-development-session",
        "commit_hash": "1" * 40,
        "dirty_diff_hash": "2" * 64,
        "code_hash": "3" * 64,
        "timing_evidence_schemas": [
            "aigp-vq2-mavlink-ingress/1",
            "aigp-vq2-received-imu/1",
            "aigp-vq2-camera-frame-timing-observation/1",
        ],
    }


def _vision_stats(frame_count: int) -> dict:
    return {
        "datagrams_received": 300,
        "unique_datagrams": 20,
        "duplicate_datagrams": 280,
        "malformed_datagrams": 0,
        "frames_reassembled": frame_count,
        "frames_decoded": frame_count,
        "decode_failures": 0,
        "out_of_order_frame_drops": 0,
        "reset_generation_drops": 0,
        "processing_errors": 0,
        "socket_errors": 0,
        "snapshot_callback_errors": 0,
        "resets": 1,
        "remembered_chunk_keys": 20,
        "timing_ledger_entries": 0,
        "timing_ledger_high_watermark": 20,
        "timing_ledger_capacity": 4096,
        "timing_overflow_latched": False,
        "receiver_buffered_partial_frames": 0,
        "receiver_buffer_high_watermark": 2,
        "receiver_buffer_capacity": 8,
        "receiver_dropped_partial_frames": 0,
        "receiver_duplicate_chunks": 0,
        "receiver_dropped_late_packets": 0,
    }


def _outcome(frame_count: int) -> dict:
    return {
        "stage": "preflight",
        "success": True,
        "reason": "passive preflight completed",
        "duration_s": 5.0,
        "gate_index_before": 0,
        "gate_index_after": 0,
        "cleanup_confirmed": True,
        "details": {
            "mavlink_ingress_stats": {
                "generation": 3,
                "next_sequence": 5,
                "highres_imu_received": 2,
                "heartbeat_received": 1,
                "race_status_received": 1,
                "actuator_received": 1,
                "dropped": 0,
                "high_watermark": 3,
                "imu_capacity": 4096,
                "other_capacity": 4096,
                "imu_dropped": 0,
                "other_dropped": 0,
                "imu_high_watermark": 2,
                "other_high_watermark": 3,
                "buffered_imu": 0,
                "buffered_other": 0,
            },
            "mavlink_outbound_audit": {
                "timesync": 1,
                "gcs_heartbeat": 1,
                "sim_reset": 0,
                "arm": 0,
                "disarm": 0,
                "attitude_target": 0,
                "position_target": 0,
                "other_command": 0,
                "disallowed_count": 0,
            },
        },
        "vision_capture_stats": _vision_stats(frame_count),
    }


def _bundle(tmp_path: Path, name: str = "passive.vq2replay") -> Path:
    path = tmp_path / name
    writer = ReplayBundleWriter(
        path,
        session_id="synthetic-passive-session",
        metadata=_metadata(),
        require_private=False,
    )

    writer.record_mavlink_ingress(
        _ingress(0, "HEARTBEAT", source_time_value=None, source_time_unit=None)
    )
    first_imu = _received_imu(1, 1_000_000)
    writer.record_imu(
        first_imu.imu,
        received_monotonic_s=2.01,
        received_sample=first_imu,
    )
    writer.record_mavlink_ingress(
        _ingress(2, "RACE_STATUS", source_time_value=5_000, source_time_unit="ms")
    )
    writer.record_mavlink_ingress(
        _ingress(
            3,
            "ACTUATOR_OUTPUT_STATUS",
            source_time_value=1_100_000,
            source_time_unit="us",
        )
    )
    second_imu = _received_imu(4, 1_200_000)
    writer.record_imu(
        second_imu.imu,
        received_monotonic_s=2.04,
        received_sample=second_imu,
    )

    image = np.full((4, 6, 3), 17, dtype=np.uint8)
    for index in range(2):
        timing = _timing(index)
        writer.capture_decoded_frame(
            image,
            generation=timing.identity.generation,
            frame_id=timing.identity.frame_id,
            sim_time_ns=timing.camera_source_time_ns,
            received_monotonic_s=timing.final_unique_packet_monotonic_ns / 1e9,
            frame_timing=timing,
        )
        writer.record_event(
            "camera_frame_timing_observation",
            observation=_observation(timing).to_primitive(),
        )
        writer.capture_frame(
            image,
            generation=timing.identity.generation,
            frame_id=timing.identity.frame_id,
            sim_time_ns=timing.camera_source_time_ns,
            received_monotonic_s=timing.final_unique_packet_monotonic_ns / 1e9,
            detector_latency_ms=0.002,
            detections=[],
            tracker=None,
            imu=None,
            estimator=None,
            race_status=None,
            generated_command=None,
            sent_command=None,
            phase="preflight",
        )
    writer.close(outcome=_outcome(2))
    return path


def _recompute_dataset_hash(manifest: dict) -> None:
    manifest["dataset_hash"] = json_hash(
        {
            "schema": manifest["schema"],
            "session_id": manifest["session_id"],
            "started_at": manifest["started_at"],
            "finished_at": manifest["finished_at"],
            "metadata": manifest["metadata"],
            "outcome": manifest["outcome"],
            "records_sha256": manifest["integrity"]["records_sha256"],
            "frame_blob_file_sha256": manifest["integrity"][
                "frame_blob_file_sha256"
            ],
        }
    )


def _mutate_manifest(bundle: Path, mutate) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    _recompute_dataset_hash(manifest)
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def _mutate_records(bundle: Path, mutate) -> None:
    records_path = bundle / "records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    mutate(rows)
    old_to_new = {row["sequence"]: index for index, row in enumerate(rows)}
    for index, row in enumerate(rows):
        row["sequence"] = index
        for link in (
            "linked_decoded_frame_record_sequence",
            "linked_imu_record_sequence",
        ):
            if link in row:
                row[link] = old_to_new[row[link]]
    records_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count"] = len(rows)
    manifest["frame_record_count"] = sum(row["type"] == "frame" for row in rows)
    manifest["decoded_frame_record_count"] = sum(
        row["type"] == "decoded_frame" for row in rows
    )
    manifest["integrity"]["records_sha256"] = sha256_file(records_path)
    _recompute_dataset_hash(manifest)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def test_analyzer_fully_verifies_and_emits_deterministic_aggregate_only(tmp_path):
    bundle = _bundle(tmp_path)

    first = analyze_bundle(bundle)
    second = analyze_bundle(bundle)

    assert first == second
    assert first["schema"] == "aigp-vq2-passive-timing-analysis/1"
    assert first["dataset_hash"] == json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )["dataset_hash"]
    assert first["code_hash"] == "3" * 64
    assert first["simulator"] == {
        "build": "3385",
        "mode": "Training",
        "mode_basis": "operator-attested-2026-07-20",
    }
    assert first["capture_load"]["label"] == "replay-capture-loaded"
    assert first["counts"]["decoded_frames"] == 2
    assert first["counts"]["processed_frames"] == 2
    assert first["counts"]["camera_frame_timing_events"] == 2
    assert first["counts"]["camera_frame_timing_observations"] == 2
    assert first["counts"]["mavlink_ingress_total"] == 5
    assert first["counts"]["received_imu"] == 2
    assert first["counts"]["mavlink_by_message_type"] == {
        "ACTUATOR_OUTPUT_STATUS": 1,
        "HEARTBEAT": 1,
        "HIGHRES_IMU": 2,
        "RACE_STATUS": 1,
    }
    assert first["rates_hz"]["camera_publication"] == pytest.approx(30.0)
    assert first["rates_hz"]["camera_consumption"] == pytest.approx(30.0)
    assert first["rates_hz"]["mavlink_ingress"] == pytest.approx(100.0)
    assert first["distributions_ns"]["decode"] == {
        "count": 2,
        "p50_ns": 300.0,
        "p95_ns": 300.0,
        "p99_ns": 300.0,
        "maximum_ns": 300,
    }
    assert first["acceptance_checks"]["generic_passive_timing_valid"] is True
    assert first["acceptance_checks"]["camera_observations_at_least_140"] is False
    assert first["acceptance_checks"]["highres_imu_arrivals_at_least_600"] is False
    assert "control_scheduler_deadlines" in first["unmeasured_items"]
    assert "command_send_timing" in first["unmeasured_items"]
    assert "command_to_actuator_causal_response" in first["unmeasured_items"]
    assert "command_to_gyro_causal_response" in first["unmeasured_items"]
    assert "simulator_wall_ratio" in first["unmeasured_items"]
    assert "training_mode_machine_detection" in first["unmeasured_items"]

    encoded = json.dumps(first, sort_keys=True)
    assert "123.456" not in encoded
    assert "234.567" not in encoded
    assert "345.678" not in encoded
    forbidden_keys = {"accel", "gyro", "mag", "image", "frame_blob", "frame_hash"}

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(first)


def test_cli_prints_one_json_document_and_nothing_else(tmp_path, capsys):
    bundle = _bundle(tmp_path)

    assert main([str(bundle)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == analyze_bundle(bundle)

    script = Path(__file__).resolve().parents[1] / "scripts" / "aigp_vq2_passive_timing.py"
    direct = subprocess.run(
        [sys.executable, str(script), str(bundle)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert direct.returncode == 0, direct.stderr
    assert direct.stderr == ""
    assert len(direct.stdout.splitlines()) == 1
    assert json.loads(direct.stdout) == analyze_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("simulator_build", "3384", "simulator_build"),
        ("simulator_mode", "Race", "simulator_mode"),
        ("simulator_mode_basis", "window-title", "simulator_mode_basis"),
        ("stage", "hover", "stage"),
        ("capture_kind", "timing-only", "capture_kind"),
        ("preflight_healthy_dwell_s", 0.0, "healthy_dwell"),
        ("code_hash", None, "code_hash"),
        ("timing_evidence_schemas", [], "schemas"),
    ],
)
def test_analyzer_rejects_missing_or_inexact_capture_metadata(
    tmp_path, field, value, match
):
    bundle = _bundle(tmp_path)
    _mutate_manifest(bundle, lambda manifest: manifest["metadata"].__setitem__(field, value))

    with pytest.raises((TypeError, ValueError), match=match):
        analyze_bundle(bundle)


def test_analyzer_invokes_full_blob_verification(tmp_path):
    bundle = _bundle(tmp_path)
    blob = next((bundle / "frames").glob("*.npy"))
    blob.write_bytes(blob.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="frame blob file hash mismatch"):
        analyze_bundle(bundle)


def test_analyzer_rejects_missing_or_nonexact_camera_observation(tmp_path):
    missing = _bundle(tmp_path, "missing.vq2replay")

    def remove_one(rows):
        for index, row in enumerate(rows):
            if row.get("event") == "camera_frame_timing_observation":
                rows.pop(index)
                return
        raise AssertionError("fixture has no timing observation")

    _mutate_records(missing, remove_one)
    with pytest.raises(ValueError, match="not one-to-one"):
        analyze_bundle(missing)

    unknown = _bundle(tmp_path, "unknown.vq2replay")

    def add_unknown(rows):
        row = next(
            item
            for item in rows
            if item.get("event") == "camera_frame_timing_observation"
        )
        row["unknown"] = True

    _mutate_records(unknown, add_unknown)
    with pytest.raises(ValueError, match="fields must be exact"):
        analyze_bundle(unknown)


def test_analyzer_rejects_camera_identity_or_timing_mismatch(tmp_path):
    bundle = _bundle(tmp_path)

    def mutate(rows):
        row = next(
            item
            for item in rows
            if item.get("event") == "camera_frame_timing_observation"
        )
        row["observation"]["frame_timing"]["publication_sequence"] += 50

    _mutate_records(bundle, mutate)
    with pytest.raises(ValueError, match="differs from receiver timing"):
        analyze_bundle(bundle)


def test_analyzer_requires_exact_qpc_camera_stream_and_no_embedded_commands(tmp_path):
    wrong_clock = _bundle(tmp_path, "wrong-clock.vq2replay")

    def replace_camera_clock(rows):
        for row in rows:
            if row.get("event") in {
                "camera_frame_timing",
                "camera_frame_timing_observation",
            }:
                timing = (
                    row["observation"]
                    if row["event"] == "camera_frame_timing"
                    else row["observation"]["frame_timing"]
                )
                timing["host_clock_id"] = "coarse-clock"

    _mutate_records(wrong_clock, replace_camera_clock)
    with pytest.raises(ValueError, match="unexpected stream or host clock"):
        analyze_bundle(wrong_clock)

    embedded_command = _bundle(tmp_path, "embedded-command.vq2replay")

    def add_command(rows):
        row = next(item for item in rows if item["type"] == "frame")
        row["generated_command"] = {
            "roll_rate": 0.0,
            "pitch_rate": 0.0,
            "yaw_rate": 0.0,
            "thrust": 0.0,
        }

    _mutate_records(embedded_command, add_command)
    with pytest.raises(ValueError, match="embedded frame commands"):
        analyze_bundle(embedded_command)


def test_analyzer_rejects_ingress_duplicate_gap_or_queue_drop(tmp_path):
    gap = _bundle(tmp_path, "gap.vq2replay")

    def add_gap(rows):
        row = next(
            item
            for item in rows
            if item.get("event") == "mavlink_ingress"
            and item["observation"]["message_type"] == "RACE_STATUS"
        )
        row["observation"]["sequence"] = 9

    _mutate_records(gap, add_gap)
    with pytest.raises(ValueError, match="duplicates or gaps"):
        analyze_bundle(gap)

    dropped = _bundle(tmp_path, "dropped.vq2replay")

    def set_drop(manifest):
        stats = manifest["outcome"]["details"]["mavlink_ingress_stats"]
        stats["dropped"] = 1
        stats["other_dropped"] = 1

    _mutate_manifest(dropped, set_drop)
    with pytest.raises(ValueError, match="queue dropped"):
        analyze_bundle(dropped)

    over_capacity = _bundle(tmp_path, "over-capacity.vq2replay")
    _mutate_manifest(
        over_capacity,
        lambda manifest: manifest["outcome"]["details"][
            "mavlink_ingress_stats"
        ].__setitem__("other_high_watermark", 4097),
    )
    with pytest.raises(ValueError, match="exceeds capacity"):
        analyze_bundle(over_capacity)


def test_analyzer_rejects_nonzero_or_inconsistent_outbound_audit(tmp_path):
    nonzero = _bundle(tmp_path, "nonzero.vq2replay")

    def set_nonzero(manifest):
        audit = manifest["outcome"]["details"]["mavlink_outbound_audit"]
        audit["arm"] = 1
        audit["disallowed_count"] = 1

    _mutate_manifest(nonzero, set_nonzero)
    with pytest.raises(ValueError, match="disallowed outbound"):
        analyze_bundle(nonzero)

    inconsistent = _bundle(tmp_path, "inconsistent.vq2replay")
    _mutate_manifest(
        inconsistent,
        lambda manifest: manifest["outcome"]["details"][
            "mavlink_outbound_audit"
        ].__setitem__("disallowed_count", 1),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        analyze_bundle(inconsistent)


def test_analyzer_rejects_incomplete_capture_counts_or_timing_overflow(tmp_path):
    count_mismatch = _bundle(tmp_path, "count-mismatch.vq2replay")
    _mutate_manifest(
        count_mismatch,
        lambda manifest: manifest["outcome"]["vision_capture_stats"].__setitem__(
            "frames_decoded", 3
        ),
    )
    with pytest.raises(ValueError, match="counts disagree"):
        analyze_bundle(count_mismatch)

    overflow = _bundle(tmp_path, "overflow.vq2replay")
    _mutate_manifest(
        overflow,
        lambda manifest: manifest["outcome"]["vision_capture_stats"].__setitem__(
            "timing_overflow_latched", True
        ),
    )
    with pytest.raises(ValueError, match="ledger overflowed"):
        analyze_bundle(overflow)

    capture_error = _bundle(tmp_path, "capture-error.vq2replay")
    _mutate_manifest(
        capture_error,
        lambda manifest: manifest["outcome"]["vision_capture_stats"].__setitem__(
            "receiver_dropped_partial_frames", 1
        ),
    )
    with pytest.raises(ValueError, match="failure/drop counters"):
        analyze_bundle(capture_error)

    missing_diagnostic = _bundle(tmp_path, "missing-diagnostic.vq2replay")
    _mutate_manifest(
        missing_diagnostic,
        lambda manifest: manifest["outcome"]["vision_capture_stats"].pop(
            "receiver_dropped_partial_frames"
        ),
    )
    with pytest.raises(ValueError, match="fields must be exact"):
        analyze_bundle(missing_diagnostic)
