from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from competition.adapter import IMUData
from competition.vq2_capture import MavlinkIngressV1, ReceivedIMUSampleV1
from competition.vq2_contracts import FrameIdentityV1, FrameTimingV1
from aigp_loop.replay import (
    AsyncReplayRecorder,
    ReplayBundleReader,
    ReplayBundleWriter,
    IsolatedReplayProcessor,
    evaluate_score_policy,
    evaluation_evidence_hash,
    evaluation_input_hash,
    evaluation_result_hash,
    grouped_session_split,
    process_frames,
    score_bundle,
    score_corpus,
    score_records,
)
from aigp_loop.promotion import replay_promotion_policy_failures
from scripts.aigp_replay import main as replay_main


def _image(value=0):
    return np.full((4, 6, 3), value, dtype=np.uint8)


def _frame_timing(*, generation=0, frame_id=1, sim_time_ns=10, base_ns=100):
    return FrameTimingV1(
        identity=FrameIdentityV1(
            "vq2-camera-udp-5600", generation, frame_id
        ),
        camera_source_time_ns=sim_time_ns,
        host_clock_id="host-perf-counter",
        publication_sequence=frame_id,
        first_unique_packet_monotonic_ns=base_ns,
        final_unique_packet_monotonic_ns=base_ns + 1,
        reassembly_complete_monotonic_ns=base_ns + 2,
        decode_start_monotonic_ns=base_ns + 3,
        decode_end_monotonic_ns=base_ns + 4,
        publish_monotonic_ns=base_ns + 5,
    )


def _received_imu(*, sequence=0, source_us=500, received_ns=90):
    ingress = MavlinkIngressV1(
        stream_id="vq2-mavlink-udp-14550",
        generation=1,
        sequence=sequence,
        message_type="HIGHRES_IMU",
        host_clock_id="host-perf-counter",
        received_monotonic_ns=received_ns,
        source_time_value=source_us,
        source_time_unit="us",
    )
    return ReceivedIMUSampleV1(
        ingress=ingress,
        imu=IMUData(
            timestamp_us=source_us,
            accel=(1.0, 2.0, -9.0),
            gyro=(0.1, 0.2, 0.3),
        ),
    )


def _patch_wrapper_attestation(monkeypatch, replay_module, wrapper, payload):
    real_run = replay_module.subprocess.run
    expected = [str(wrapper.resolve()), "--attest"]

    def run(argv, *args, **kwargs):
        if list(argv) == expected:
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(replay_module.subprocess, "run", run)


def _commit_test_repository(root: Path) -> None:
    subprocess.run(
        ["git", "init", "-q"], cwd=root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "add", "-A"], cwd=root, check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AIGP Test",
            "-c",
            "user.email=aigp-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "candidate",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _rewrite_records_and_reseal(bundle, mutate):
    from aigp_loop._util import json_hash, sha256_file

    records_path = bundle / "records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["records_sha256"] = sha256_file(records_path)
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
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _recompute_manifest_dataset_hash(manifest):
    from aigp_loop._util import json_hash

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


def test_bundle_deduplicates_decoded_pixels_and_verifies_all_hashes(tmp_path):
    path = tmp_path / "session.vq2replay"
    writer = ReplayBundleWriter(path, session_id="session-1", require_private=False)
    writer.capture_decoded_frame(
        _image(3), generation=0, frame_id=1, sim_time_ns=10, received_monotonic_s=1.0
    )
    writer.capture_frame(
        _image(3),
        generation=0,
        frame_id=1,
        sim_time_ns=10,
        received_monotonic_s=1.0,
        detector_latency_ms=0.5,
        detections=[],
        tracker=None,
        imu=None,
        estimator=None,
        race_status=None,
        generated_command=None,
        sent_command=None,
    )
    dataset_hash = writer.close()
    reader = ReplayBundleReader(path)
    verified = reader.verify()
    assert verified == {
        "dataset_hash": dataset_hash,
        "records": 2,
        "frames": 1,
        "decoded_frames": 1,
        "unique_frame_blobs": 1,
    }
    assert len(list((path / "frames").glob("*.npy"))) == 1


def test_bundle_additively_preserves_exact_camera_and_imu_ingress(tmp_path):
    path = tmp_path / "timed-session.vq2replay"
    writer = ReplayBundleWriter(path, require_private=False)
    received_imu = _received_imu()
    imu_sequence = writer.record_imu(
        received_imu.imu,
        estimator={"healthy": True},
        received_monotonic_s=1.0,
        received_sample=received_imu,
    )
    timing = _frame_timing()
    frame_sequence = writer.capture_decoded_frame(
        _image(4),
        generation=0,
        frame_id=1,
        sim_time_ns=10,
        received_monotonic_s=1.1,
        frame_timing=timing,
    )
    writer.record_mavlink_ingress(
        MavlinkIngressV1(
            stream_id="vq2-mavlink-udp-14550",
            generation=1,
            sequence=1,
            message_type="HEARTBEAT",
            host_clock_id="host-perf-counter",
            received_monotonic_ns=95,
            source_time_value=None,
            source_time_unit=None,
        )
    )
    writer.close()

    reader = ReplayBundleReader(path)
    _summary, records = reader.verify_and_read()
    assert [record["type"] for record in records] == [
        "imu",
        "event",
        "decoded_frame",
        "event",
        "event",
    ]
    received_event = records[1]
    assert received_event["event"] == "received_imu"
    assert received_event["linked_imu_record_sequence"] == imu_sequence
    assert ReceivedIMUSampleV1.from_primitive(
        received_event["observation"]
    ) == received_imu
    camera_event = records[3]
    assert camera_event["event"] == "camera_frame_timing"
    assert camera_event["linked_decoded_frame_record_sequence"] == frame_sequence
    assert FrameTimingV1.from_primitive(camera_event["observation"]) == timing
    assert records[4]["event"] == "mavlink_ingress"


def test_replay_v1_additively_preserves_powered_calibration_rich_events(tmp_path):
    from scripts.aigp_vq2_powered_attempt import (
        ATTEMPT_ID,
        DEADLINE_DURATIONS_NS,
        validate_phase_deadline_event,
    )

    duration_ns = DEADLINE_DURATIONS_NS["child_connect"]
    observation = {
        "schema": "aigp-vq2-phase-deadline/1",
        "attempt_id": ATTEMPT_ID,
        "producer_role": "powered_child",
        "phase": "connect",
        "event_sequence": 0,
        "started_monotonic_ns": 100,
        "duration_ns": duration_ns,
        "parent_deadline_monotonic_ns": 100 + duration_ns + 1,
        "deadline_monotonic_ns": 100 + duration_ns,
    }
    validate_phase_deadline_event(observation)

    bundle = tmp_path / "powered-rich-event.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.record_event(
        "calibration_phase_deadline",
        observation=observation,
    )
    writer.close()

    summary, records = ReplayBundleReader(bundle).verify_and_read()
    assert summary["records"] == 1
    assert records[0]["schema"] == "aigp-vq2-replay-record/1"
    assert records[0]["type"] == "event"
    assert records[0]["event"] == "calibration_phase_deadline"
    assert records[0]["observation"] == observation
    validate_phase_deadline_event(records[0]["observation"])


def test_replay_v1_rejects_powered_rich_fields_inserted_into_frozen_core_rows(
    tmp_path,
):
    bundle = tmp_path / "powered-rich-field-in-core-row.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.capture_decoded_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=10,
        received_monotonic_s=1.0,
    )
    writer.close()

    def insert_rich_field(rows):
        decoded = next(row for row in rows if row["type"] == "decoded_frame")
        decoded["calibration_tick_disposition"] = {
            "schema": "aigp-vq2-calibration-tick-disposition/1"
        }

    _rewrite_records_and_reseal(bundle, insert_rich_field)
    with pytest.raises(ValueError, match="missing/unknown"):
        ReplayBundleReader(bundle).verify_and_read()


def test_bundle_rejects_mismatched_exact_timing_bindings(tmp_path):
    writer = ReplayBundleWriter(
        tmp_path / "mismatch.vq2replay", require_private=False
    )
    received_imu = _received_imu()
    different = IMUData(
        timestamp_us=500,
        accel=(9.0, 9.0, 9.0),
        gyro=(0.1, 0.2, 0.3),
    )
    with pytest.raises(ValueError, match="differs from the core IMU"):
        writer.record_imu(different, received_sample=received_imu)
    with pytest.raises(ValueError, match="timing identity"):
        writer.capture_decoded_frame(
            _image(),
            generation=0,
            frame_id=2,
            sim_time_ns=10,
            received_monotonic_s=1.0,
            frame_timing=_frame_timing(frame_id=1),
        )
    writer.abort("expected binding rejection")


def test_swallowed_exact_capture_validation_cannot_seal_complete(tmp_path):
    received_imu = _received_imu()

    imu_path = tmp_path / "swallowed-imu-rejection.vq2replay"
    writer = ReplayBundleWriter(imu_path, require_private=False)
    with pytest.raises(ValueError, match="differs from the core IMU"):
        writer.record_imu(
            IMUData(
                timestamp_us=500,
                accel=(9.0, 9.0, 9.0),
                gyro=(0.1, 0.2, 0.3),
            ),
            received_sample=received_imu,
        )
    with pytest.raises(RuntimeError, match="cannot seal replay bundle"):
        writer.close()
    assert json.loads((imu_path / "manifest.json").read_text())["complete"] is False

    ingress_path = tmp_path / "swallowed-ingress-rejection.vq2replay"
    writer = ReplayBundleWriter(ingress_path, require_private=False)
    invalid_ingress = received_imu.ingress.to_primitive()
    invalid_ingress["sequence"] = True
    with pytest.raises((TypeError, ValueError)):
        writer.record_mavlink_ingress(invalid_ingress)
    with pytest.raises(RuntimeError, match="cannot seal replay bundle"):
        writer.close()
    assert (
        json.loads((ingress_path / "manifest.json").read_text())["complete"]
        is False
    )

    frame_path = tmp_path / "swallowed-frame-rejection.vq2replay"
    writer = ReplayBundleWriter(frame_path, require_private=False)
    with pytest.raises(ValueError, match="timing identity"):
        writer.capture_decoded_frame(
            _image(),
            generation=0,
            frame_id=2,
            sim_time_ns=10,
            received_monotonic_s=1.0,
            frame_timing=_frame_timing(frame_id=1),
        )
    with pytest.raises(RuntimeError, match="cannot seal replay bundle"):
        writer.close()
    assert (
        json.loads((frame_path / "manifest.json").read_text())["complete"]
        is False
    )


def test_bundle_verify_detects_blob_and_manifest_dataset_corruption(tmp_path):
    first = tmp_path / "blob.vq2replay"
    writer = ReplayBundleWriter(first, require_private=False)
    writer.capture_decoded_frame(
        _image(1), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.close()
    blob = next((first / "frames").glob("*.npy"))
    blob.write_bytes(blob.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="frame blob file hash mismatch"):
        ReplayBundleReader(first).verify(verify_frames=False)

    second = tmp_path / "manifest.vq2replay"
    writer = ReplayBundleWriter(second, require_private=False)
    writer.close()
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        ReplayBundleReader(second).verify(verify_frames=False)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda manifest: manifest.__setitem__("complete", "true"), "exact bool"),
        (lambda manifest: manifest.__setitem__("record_count", "0"), "exact non-negative"),
        (lambda manifest: manifest.__setitem__("unknown", 1), "missing or unknown"),
    ],
)
def test_reader_rejects_coerced_or_unknown_manifest_fields(tmp_path, mutation, match):
    bundle = tmp_path / f"manifest-{match.replace(' ', '-')}.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.close()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ReplayBundleReader(bundle)


def test_verify_rejects_symlinked_frame_blob_even_without_pixel_decode(tmp_path):
    bundle = tmp_path / "symlink.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.capture_decoded_frame(
        _image(1), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.close()
    blob = next((bundle / "frames").glob("*.npy"))
    external = tmp_path / "external.npy"
    blob.replace(external)
    try:
        blob.symlink_to(external)
    except OSError:
        pytest.skip("creating a file symlink is unavailable on this Windows host")
    with pytest.raises(ValueError, match="symlink|reparse"):
        ReplayBundleReader(bundle).verify(verify_frames=False)


def test_verified_records_are_the_same_snapshot_whose_hash_was_checked(tmp_path):
    bundle = tmp_path / "record-snapshot.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="snapshot", require_private=False)
    writer.record_event("original", value=1)
    writer.close()
    reader = ReplayBundleReader(bundle)
    _summary, records = reader.verify_and_read(verify_frames=False)

    (bundle / "records.jsonl").write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-record/1",
                "session_id": "snapshot",
                "sequence": 0,
                "type": "event",
                "capture_wall_time_ns": 0,
                "event": "replacement",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert records[0]["event"] == "original"
    with pytest.raises(ValueError, match="records.jsonl hash mismatch"):
        ReplayBundleReader(bundle).verify(verify_frames=False)


def test_frame_replacement_after_verification_fails_at_point_of_use(tmp_path):
    bundle = tmp_path / "frame-snapshot.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="snapshot", require_private=False)
    writer.capture_decoded_frame(
        _image(7), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.close()
    reader = ReplayBundleReader(bundle)
    _summary, records = reader.verify_and_read(verify_frames=True)
    assert not hasattr(reader, "_verified_blob_bytes")
    blob = next((bundle / "frames").glob("*.npy"))
    blob.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="frame blob file hash mismatch"):
        reader.load_frame(records[0])
    with pytest.raises(ValueError, match="frame blob file hash mismatch"):
        ReplayBundleReader(bundle).verify(verify_frames=False)


def test_score_binds_the_exact_annotation_snapshot_it_parsed(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_bytes

    bundle = tmp_path / "annotation-snapshot.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="snapshot", require_private=False)
    writer.capture_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        detector_latency_ms=1.0,
        detections=[],
        tracker=None,
        imu=None,
        estimator=None,
        race_status=None,
        generated_command=None,
        sent_command=None,
    )
    writer.close()
    annotations = tmp_path / "annotations.jsonl"
    original = (
        json.dumps(
            {
                "schema": "aigp-vq2-replay-annotation/1",
                "session_id": "snapshot",
                "generation": 0,
                "frame_id": 1,
                "gates": [],
            }
        )
        + "\n"
    ).encode("utf-8")
    annotations.write_bytes(original)
    parse_snapshot = replay_module.load_annotations_bytes

    def replace_after_snapshot(payload):
        annotations.write_text("replacement\n", encoding="utf-8")
        return parse_snapshot(payload)

    monkeypatch.setattr(replay_module, "load_annotations_bytes", replace_after_snapshot)
    score = replay_module.score_bundle(bundle, annotations_path=annotations)
    assert score["annotations_sha256"] == sha256_bytes(original)
    assert score["labeled_frames"] == 1


def test_reader_rejects_boolean_sequence_even_when_equal_to_zero(tmp_path):
    bundle = tmp_path / "bool-sequence.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.record_event("event")
    writer.close()
    _rewrite_records_and_reseal(
        bundle, lambda rows: rows[0].__setitem__("sequence", False)
    )
    with pytest.raises(ValueError, match="non-contiguous sequence"):
        ReplayBundleReader(bundle).verify()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.__setitem__("type", "invented"), "unknown replay record"),
        (lambda row: row.__setitem__("event", "bad event name"), "name is invalid"),
        (lambda row: row.__setitem__("frame_blob", "alias"), "reserved semantic"),
    ],
)
def test_reader_rejects_unknown_malformed_or_aliasing_record_schemas(
    tmp_path, mutation, match
):
    bundle = tmp_path / "bad-record-schema.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.record_event("valid_event")
    writer.close()
    _rewrite_records_and_reseal(bundle, lambda rows: mutation(rows[0]))
    with pytest.raises(ValueError, match=match):
        list(ReplayBundleReader(bundle).records())


def test_reader_rejects_extra_fields_on_core_record(tmp_path):
    bundle = tmp_path / "extra-core-field.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.record_imu({"accel": [0, 0, 0]})
    writer.close()
    _rewrite_records_and_reseal(
        bundle, lambda rows: rows[0].__setitem__("unexpected", 1)
    )
    with pytest.raises(ValueError, match="missing/unknown"):
        list(ReplayBundleReader(bundle).records())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("generation", True, "frame token"),
        ("image_shape", [5, 6, 3], "contradicts"),
        ("image_dtype", "<u2", "HxWx3 uint8"),
    ],
)
def test_reader_rejects_malformed_frame_semantics(tmp_path, field, value, match):
    bundle = tmp_path / f"bad-frame-{field}.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.capture_decoded_frame(
        _image(), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.close()
    _rewrite_records_and_reseal(
        bundle, lambda rows: rows[0].__setitem__(field, value)
    )
    with pytest.raises(ValueError, match=match):
        ReplayBundleReader(bundle).verify()


def test_reader_rejects_duplicate_tokens_and_unreferenced_manifest_blobs(tmp_path):
    duplicate = tmp_path / "duplicate-token.vq2replay"
    writer = ReplayBundleWriter(duplicate, require_private=False)
    for frame_id, value in ((1, 1), (2, 2)):
        writer.capture_decoded_frame(
            _image(value),
            generation=0,
            frame_id=frame_id,
            sim_time_ns=frame_id,
            received_monotonic_s=frame_id * 0.1,
        )
    writer.close()
    _rewrite_records_and_reseal(
        duplicate,
        lambda rows: rows[1].update(frame_id=1, sim_time_ns=1),
    )
    with pytest.raises(ValueError, match="duplicate frame token"):
        ReplayBundleReader(duplicate).verify()

    unused = tmp_path / "unused-blob.vq2replay"
    writer = ReplayBundleWriter(unused, require_private=False)
    writer.capture_decoded_frame(
        _image(), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.close()
    from aigp_loop._util import json_hash, sha256_file

    blob = next((unused / "frames").glob("*.npy"))
    extra_digest = "f" * 64
    extra = unused / "frames" / f"{extra_digest}.npy"
    extra.write_bytes(blob.read_bytes())
    manifest_path = unused / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["frame_blob_file_sha256"][extra_digest] = sha256_file(extra)
    manifest["unique_frame_blob_count"] += 1
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
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="counts exceed|exactly referenced"):
        ReplayBundleReader(unused).verify(verify_frames=False)


@pytest.mark.parametrize(
    "reserved",
    ["schema", "session_id", "sequence", "type", "capture_wall_time_ns"],
)
def test_writer_rejects_reserved_record_envelope_overrides(tmp_path, reserved):
    writer = ReplayBundleWriter(
        tmp_path / f"reserved-{reserved}.vq2replay", require_private=False
    )
    with pytest.raises(ValueError, match="trusted envelope"):
        writer.append("event", **{reserved: "attacker"})
    writer.abort("test complete")


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", True),
        ("generation", "0"),
        ("frame_id", -1),
        ("sim_time_ns", 1.0),
    ],
)
def test_writer_rejects_coerced_or_negative_frame_tokens(tmp_path, field, value):
    writer = ReplayBundleWriter(
        tmp_path / f"token-{field}-{value}.vq2replay", require_private=False
    )
    fields = {
        "generation": 0,
        "frame_id": 1,
        "sim_time_ns": 1,
        "received_monotonic_s": 1.0,
    }
    fields[field] = value
    with pytest.raises(ValueError, match="exact integers"):
        writer.capture_decoded_frame(_image(), **fields)
    writer.abort("test complete")


def test_verify_recomputes_declared_pixel_hash_not_only_blob_file_hash(tmp_path):
    from aigp_loop._util import json_hash, sha256_file

    bundle = tmp_path / "semantic.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.capture_decoded_frame(
        _image(1), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.close()
    blob = next((bundle / "frames").glob("*.npy"))
    with blob.open("wb") as handle:
        np.save(handle, _image(9), allow_pickle=False)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["frame_blob_file_sha256"][blob.stem] = sha256_file(blob)
    manifest["dataset_hash"] = json_hash(
        {
            "schema": manifest["schema"],
            "session_id": manifest["session_id"],
            "started_at": manifest["started_at"],
            "finished_at": manifest["finished_at"],
            "metadata": manifest["metadata"],
            "outcome": manifest["outcome"],
            "records_sha256": manifest["integrity"]["records_sha256"],
            "frame_blob_file_sha256": manifest["integrity"]["frame_blob_file_sha256"],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="decoded frame content hash mismatch"):
        ReplayBundleReader(bundle).verify(verify_frames=True)


def test_reader_rejects_huge_npy_shape_before_numpy_allocation(tmp_path, monkeypatch):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_file

    bundle = tmp_path / "huge-npy-shape.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.capture_decoded_frame(
        _image(), generation=0, frame_id=1, sim_time_ns=1,
        received_monotonic_s=1.0,
    )
    writer.close()
    blob = next((bundle / "frames").glob("*.npy"))
    hostile = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        hostile,
        {
            "descr": np.dtype(np.uint8).str,
            "fortran_order": False,
            "shape": (1_000_000_000, 1, 3),
        },
    )
    blob.write_bytes(hostile.getvalue())
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["frame_blob_file_sha256"][blob.stem] = sha256_file(blob)
    _recompute_manifest_dataset_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        replay_module.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("np.load must not see an unbounded header")
        ),
    )
    with pytest.raises(ValueError, match="decoded shape exceeds"):
        ReplayBundleReader(bundle).verify()


def test_reader_enforces_records_blob_and_manifest_count_limits(tmp_path, monkeypatch):
    import aigp_loop.replay as replay_module

    bundle = tmp_path / "bounded.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.capture_decoded_frame(
        _image(), generation=0, frame_id=1, sim_time_ns=1,
        received_monotonic_s=1.0,
    )
    writer.close()
    records_size = (bundle / "records.jsonl").stat().st_size
    monkeypatch.setattr(replay_module, "MAX_REPLAY_RECORDS_BYTES", records_size - 1)
    with pytest.raises(ValueError, match="records.jsonl exceeds"):
        ReplayBundleReader(bundle).verify()

    monkeypatch.setattr(replay_module, "MAX_REPLAY_RECORDS_BYTES", 128 * 1024 * 1024)
    blob = next((bundle / "frames").glob("*.npy"))
    monkeypatch.setattr(
        replay_module, "MAX_REPLAY_FRAME_BLOB_BYTES", blob.stat().st_size - 1
    )
    with pytest.raises(ValueError, match="frame blob exceeds"):
        ReplayBundleReader(bundle).verify()

    monkeypatch.setattr(replay_module, "MAX_REPLAY_FRAME_BLOB_BYTES", 32 * 1024 * 1024)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count"] = replay_module.MAX_REPLAY_RECORD_COUNT + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="counts exceed"):
        ReplayBundleReader(bundle)


def test_dataset_identity_binds_outcome_and_finish_provenance(tmp_path):
    bundle = tmp_path / "outcome-identity.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    writer.record_event("started")
    writer.close(outcome={"completed": True})
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outcome"] = {"completed": False}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        ReplayBundleReader(bundle).verify(verify_frames=False)


def test_writer_rejects_ambiguous_annotation_frame_identity(tmp_path):
    writer = ReplayBundleWriter(
        tmp_path / "ambiguous-label.vq2replay", require_private=False
    )
    writer.capture_decoded_frame(
        _image(), generation=0, frame_id=7, sim_time_ns=1,
        received_monotonic_s=1.0,
    )
    with pytest.raises(ValueError, match="unique for labels"):
        writer.capture_decoded_frame(
            _image(1), generation=0, frame_id=7, sim_time_ns=2,
            received_monotonic_s=2.0,
        )
    writer.abort("expected ambiguous label rejection")


def test_bundle_path_inside_git_must_be_ignored(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    with pytest.raises(ValueError, match="not Git-ignored"):
        ReplayBundleWriter(repo / "private" / "session", repo_root=repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    with pytest.raises(ValueError, match="not Git-ignored"):
        ReplayBundleWriter(repo / "other" / "session")
    (repo / ".gitignore").write_text("private/\n", encoding="utf-8")
    writer = ReplayBundleWriter(
        repo / "private" / "allowed", repo_root=repo, session_id="allowed"
    )
    writer.close()


def test_writer_rejects_symlink_or_reparse_parent_before_creating_bundle(tmp_path):
    outside = tmp_path / "public"
    outside.mkdir()
    linked_parent = tmp_path / "captures"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable on this host: {exc}")
    target = linked_parent / "session.vq2replay"
    with pytest.raises(ValueError, match="symlink|reparse"):
        ReplayBundleWriter(target, require_private=False)
    assert not (outside / "session.vq2replay").exists()


def _frame(sequence, frame_id, now, gate, detections, tracker):
    return {
        "type": "frame",
        "sequence": sequence,
        "generation": 0,
        "frame_id": frame_id,
        "sim_time_ns": int(now * 1e9),
        "received_monotonic_s": now,
        "detector_latency_ms": float(frame_id),
        "detections": detections,
        "tracker": tracker,
        "race_status": {"active_gate_index": gate},
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.0, 0.0, 0.0],
            "body_rates": [0.0, 0.0, 0.0],
        },
    }


def _gate(center, *, corners=None, gate_index=None):
    result = {
        "center_px": list(center),
        "corners_px": corners
        or [[center[0] - 1, center[1] - 1], [center[0] + 1, center[1] - 1],
            [center[0] + 1, center[1] + 1], [center[0] - 1, center[1] + 1]],
        "selector_eligible": True,
    }
    if gate_index is not None:
        result["gate_index"] = gate_index
    return result


def _command(thrust):
    return {"roll_rate": 0.1, "pitch_rate": -0.1, "yaw_rate": 0.0, "thrust": thrust}


def _unsafe_processor(_image, _record):
    return {
        "detections": [
            {"center_px": [500, 500], "selector_eligible": True}
        ],
        "tracker": None,
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.0, 0.0, 0.0],
            "body_rates": [0.0, 0.0, 0.0],
        },
        "generated_command": _command(0.2),
    }


def _noop_processor(_image, _record):
    return {}


def _copy_through_processor(_image, context):
    return {"detections": context["detections"], "tracker": context["tracker"]}


def _matching_processor(_image, _context):
    return {
        "detections": [{"center_px": [2, 2], "selector_eligible": True}],
        "tracker": {"target": {"center_px": [2, 2]}},
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.2, -0.1, 0.3],
            "body_rates": [0.0, 0.0, 0.0],
        },
        "generated_command": _command(0.2),
    }


def _matching_corners_processor(_image, _context):
    return {
        "detections": [
            {
                "center_px": [2, 2],
                "corners_px": [[1, 1], [3, 1], [3, 3], [1, 3]],
                "selector_eligible": True,
            }
        ],
        "tracker": {"target": {"center_px": [2, 2]}},
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.2, -0.1, 0.3],
            "body_rates": [0.0, 0.0, 0.0],
        },
        "generated_command": _command(0.2),
    }


def _constant_zero_behavior_processor(_image, _context):
    return {
        "detections": [
            {
                "center_px": [2, 2],
                "corners_px": [[1, 1], [3, 1], [3, 3], [1, 3]],
                "selector_eligible": True,
            }
        ],
        "tracker": {"target": {"center_px": [2, 2]}},
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.0, 0.0, 0.0],
            "body_rates": [0.0, 0.0, 0.0],
        },
        "generated_command": {
            "roll_rate": 0.0,
            "pitch_rate": 0.0,
            "yaw_rate": 0.0,
            "thrust": 0.0,
        },
    }


def _slow_processor_claiming_zero(_image, _context):
    time.sleep(0.02)
    return {
        "detections": [],
        "tracker": None,
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.0, 0.0, 0.0],
            "body_rates": [0.0, 0.0, 0.0],
        },
        "generated_command": _command(0.2),
    }


def _coercing_processor(_image, _context):
    return {
        "detections": [
            {"center_px": ["2", 2], "selector_eligible": True}
        ],
        "tracker": None,
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.0, 0.0, 0.0],
            "body_rates": [0.0, 0.0, 0.0],
        },
        "generated_command": _command(0.2),
    }


def _seeded_random_processor(_image, _context):
    center = float(np.random.random())
    return {
        "detections": [{"center_px": [center, center], "selector_eligible": True}],
        "tracker": {"target": {"center_px": [center, center]}},
        "estimator": {
            "healthy": True,
            "rpy_rad": [0.0, 0.0, 0.0],
            "body_rates": [0.0, 0.0, 0.0],
        },
        "generated_command": _command(0.2),
    }


_PRIVACY_CONTEXTS = []


def _privacy_probe_processor(_image, context):
    serialized = json.dumps(dict(context), default=list)
    for forbidden in (
        "annotation",
        "expected_command",
        "generated_command",
        "sent_command",
        "accepted_target",
        "tracker_target",
        "selected_detection_index",
        "last_command",
        "bbox_xywh_px",
        "confidence",
    ):
        assert f'"{forbidden}"' not in serialized
    _PRIVACY_CONTEXTS.append(context)
    return _matching_processor(_image, context)


def test_score_covers_perception_stability_reacquisition_latency_and_command_pairing():
    frames = [
        _frame(0, 1, 0.0, 0, [_gate((10, 10)), _gate((100, 100))], {"target": {"center_x": 10, "center_y": 10}}),
        _frame(1, 2, 0.1, 0, [], None),
        _frame(2, 3, 0.2, 1, [], None),
        # Detector center jumps, but the retained tracker target moves only 2px.
        _frame(3, 4, 0.3, 1, [_gate((70, 70))], {"target": {"center_x": 12, "center_y": 10}}),
    ]
    generated_a = {"type": "command", "sequence": 4, "kind": "generated", "frame_token": [0, 1, 0], "command": _command(0.2)}
    generated_b = {"type": "command", "sequence": 5, "kind": "generated", "frame_token": [0, 4, 300], "command": _command(0.3)}
    sent_b = {"type": "command", "sequence": 6, "kind": "sent", "frame_token": [0, 4, 300], "command": _command(0.3)}
    sent_a = {"type": "command", "sequence": 7, "kind": "sent", "frame_token": [0, 1, 0], "command": _command(0.2)}
    unsafe = {"type": "command", "sequence": 8, "kind": "generated", "frame_token": [9, 9, 9], "command": _command(0.36)}
    labels = {
        ("s", 0, 1): {"gates": [_gate((10, 10), gate_index=0)], "active_gate_index": 0, "estimator_rpy_rad": [0, 0, 0]},
        ("s", 0, 2): {"gates": [_gate((11, 10), gate_index=0)], "active_gate_index": 0},
        ("s", 0, 3): {"gates": [_gate((12, 10), gate_index=1)], "active_gate_index": 1},
        ("s", 0, 4): {"gates": [_gate((70, 70), gate_index=1)], "active_gate_index": 1},
    }
    score = score_records(
        [*frames, generated_a, generated_b, sent_b, sent_a, unsafe],
        session_id="s",
        annotations=labels,
    )
    perception = score["perception"]
    assert perception["gate_recall"] == 0.5
    assert perception["false_positives"] == 1
    assert perception["center_error_px_mean"] == 0.0
    assert perception["corner_error_px_mean"] == 0.0
    # Miss streaks are scoped to one active-gate epoch; pre/post-transition
    # misses are independent outages.
    assert perception["longest_consecutive_missed_frames"] == 1
    assert perception["temporal_center_step_px_p95"] is None
    assert perception["post_gate_reacquisition_latency_ms"] == pytest.approx([100.0])
    assert perception["detector_latency_ms_p50"] == 2.5
    assert perception["transition_count"] == 1
    assert perception["reacquired_count"] == 1
    assert perception["unreacquired_count"] == 0
    commands = score["open_loop_commands"]["recorded_stream"]
    assert commands["generated_sent_max_abs_error"] == 0.0
    assert commands["generated_without_matching_send"] == 1
    assert commands["sent_without_matching_generation"] == 0
    assert commands["envelope_violation_count"] == 1  # thrust 0.36 > VQ2 0.35


def test_reacquisition_requires_new_gate_truth_match_not_stale_tracker_target():
    frames = [
        _frame(0, 1, 0.0, 0, [_gate((10, 10))], {"target": {"center_px": [10, 10]}}),
        _frame(1, 2, 0.1, 1, [], {"target": {"center_px": [10, 10]}}),
        # The old gate remains visible and matches, while the new active gate
        # does not.  This must not count as active-target reacquisition.
        _frame(2, 3, 0.2, 1, [_gate((10, 10))], {"target": {"center_px": [10, 10]}}),
    ]
    labels = {
        ("s", 0, 1): {"gates": [_gate((10, 10), gate_index=0)], "active_gate_index": 0},
        ("s", 0, 2): {"gates": [_gate((100, 100), gate_index=1)], "active_gate_index": 1},
        ("s", 0, 3): {
            "gates": [
                _gate((10, 10), gate_index=0),
                _gate((100, 100), gate_index=1),
            ],
            "active_gate_index": 1,
        },
    }
    perception = score_records(frames, session_id="s", annotations=labels)[
        "perception"
    ]
    assert perception["transition_count"] == 1
    assert perception["reacquired_count"] == 0
    assert perception["unreacquired_count"] == 1


def test_temporal_stability_and_miss_streaks_do_not_cross_target_epochs():
    frames = [
        _frame(0, 1, 0.0, 0, [], {"target": {"center_px": [10, 10]}}),
        _frame(1, 2, 0.1, 1, [], {"target": {"center_px": [100, 100]}}),
        _frame(2, 3, 0.2, 1, [], {"target": {"center_px": [101, 100]}}),
    ]
    labels = {
        ("s", 0, 1): {"gates": [_gate((10, 10), gate_index=0)], "active_gate_index": 0},
        ("s", 0, 2): {"gates": [_gate((100, 100), gate_index=1)], "active_gate_index": 1},
        ("s", 0, 3): {"gates": [], "active_gate_index": 1},
    }
    perception = score_records(frames, session_id="s", annotations=labels)[
        "perception"
    ]
    assert perception["temporal_center_step_px_p95"] == pytest.approx(1.0)
    assert perception["longest_consecutive_missed_frames"] == 1


def test_scoring_uses_publication_sequence_when_sim_time_regresses():
    first = _frame(0, 1, 0.0, 0, [_gate((10, 10))], None)
    transition = _frame(1, 2, 0.1, 1, [], None)
    reacquired = _frame(2, 3, 0.2, 1, [_gate((20, 20))], None)
    first["sim_time_ns"] = 300
    transition["sim_time_ns"] = 100
    reacquired["sim_time_ns"] = 200
    labels = {
        ("s", 0, 1): {
            "gates": [_gate((10, 10), gate_index=0)],
            "active_gate_index": 0,
        },
        ("s", 0, 2): {
            "gates": [_gate((20, 20), gate_index=1)],
            "active_gate_index": 1,
        },
        ("s", 0, 3): {
            "gates": [_gate((20, 20), gate_index=1)],
            "active_gate_index": 1,
        },
    }
    perception = score_records(
        [reacquired, first, transition], session_id="s", annotations=labels
    )["perception"]
    assert perception["transition_count"] == 1
    assert perception["reacquired_count"] == 1
    assert perception["unreacquired_count"] == 0
    assert perception["post_gate_reacquisition_latency_ms"] == pytest.approx(
        [100.0]
    )


def test_command_pairing_uses_publication_sequence_not_input_container_order():
    generated_a = {
        "type": "command",
        "sequence": 0,
        "kind": "generated",
        "frame_token": [0, 1, 1],
        "command": _command(0.1),
    }
    generated_b = {
        "type": "command",
        "sequence": 1,
        "kind": "generated",
        "frame_token": [0, 1, 1],
        "command": _command(0.2),
    }
    sent_a = {
        "type": "command",
        "sequence": 2,
        "kind": "sent",
        "frame_token": [0, 1, 1],
        "command": _command(0.1),
    }
    sent_b = {
        "type": "command",
        "sequence": 3,
        "kind": "sent",
        "frame_token": [0, 1, 1],
        "command": _command(0.2),
    }
    score = score_records(
        [sent_b, sent_a, generated_a, generated_b], session_id="s"
    )
    assert score["open_loop_commands"]["recorded_stream"][
        "generated_sent_max_abs_error"
    ] == 0.0


@pytest.mark.parametrize("sequence", [True, "0", -1])
def test_scoring_rejects_nonexact_record_sequence(sequence):
    with pytest.raises(ValueError, match="record sequence"):
        score_records(
            [{"type": "command", "sequence": sequence}], session_id="s"
        )


def test_scoring_rejects_duplicate_record_sequence():
    with pytest.raises(ValueError, match="sequences must be unique"):
        score_records(
            [
                {"type": "command", "sequence": 0},
                {"type": "command", "sequence": 0},
            ],
            session_id="s",
        )


def test_estimator_and_command_metrics_use_global_component_rmse():
    first = _frame(0, 1, 0.0, 0, [], None)
    second = _frame(1, 2, 0.1, 0, [], None)
    first["estimator"]["rpy_rad"] = [1.0, 0.0, 0.0]
    second["estimator"]["rpy_rad"] = [3.0, 0.0, 0.0]
    first["generated_command"] = {
        "roll_rate": 1.0,
        "pitch_rate": 0.0,
        "yaw_rate": 0.0,
        "thrust": 0.0,
    }
    second["generated_command"] = {
        "roll_rate": 3.0,
        "pitch_rate": 0.0,
        "yaw_rate": 0.0,
        "thrust": 0.0,
    }
    labels = {
        ("s", 0, frame_id): {
            "gates": [],
            "estimator_rpy_rad": [0.0, 0.0, 0.0],
            "expected_command": {
                "roll_rate": 0.0,
                "pitch_rate": 0.0,
                "yaw_rate": 0.0,
                "thrust": 0.0,
            },
        }
        for frame_id in (1, 2)
    }
    score = score_records([first, second], session_id="s", annotations=labels)
    assert score["estimator"]["rpy_rmse_rad"] == pytest.approx(
        np.sqrt(10.0 / 6.0)
    )
    assert score["estimator"]["rpy_mean_frame_rmse_rad"] == pytest.approx(
        (1.0 + 3.0) / (2.0 * np.sqrt(3.0))
    )
    replay_commands = score["open_loop_commands"]["replay_frames"]
    assert replay_commands["expected_command_rmse"] == pytest.approx(
        np.sqrt(10.0 / 8.0)
    )
    assert replay_commands["expected_command_mean_frame_rmse"] == pytest.approx(1.0)


def test_weak_passing_policy_is_not_promotion_capable():
    weak = {"schema": "aigp-vq2-replay-policy/1", "metrics": {"frames": {"min": 1}}}
    policy_result = evaluate_score_policy({"frames": 1}, weak)
    assert policy_result["passed"] is True
    evidence = {
        "schema": "aigp-vq2-replay-score/1",
        "policy": policy_result,
    }
    failures = replay_promotion_policy_failures(evidence)
    assert failures
    assert any("mandatory replay" in failure for failure in failures)


def test_transition_at_session_end_is_an_explicit_unreacquired_miss():
    records = [
        _frame(0, 1, 0.0, 0, [_gate((10, 10))], None),
        _frame(1, 2, 0.1, 1, [], None),
    ]
    score = score_records(records, session_id="s")
    assert score["perception"]["transition_count"] == 1
    assert score["perception"]["reacquired_count"] == 0
    assert score["perception"]["unreacquired_count"] == 1


def test_gate_assignment_maximizes_cardinality_before_distance():
    frame = _frame(
        0,
        1,
        0.0,
        0,
        [_gate((1, 0)), _gate((-2, 0))],
        None,
    )
    labels = {
        ("s", 0, 1): {
            "gates": [_gate((0, 0)), _gate((5, 0))],
        }
    }
    score = score_records(
        [frame], session_id="s", annotations=labels, max_center_error_px=5.0
    )
    assert score["perception"]["gate_matches"] == 2
    assert score["perception"]["gate_recall"] == 1.0


def test_policy_fails_closed_on_missing_labels_and_unsafe_replay_frame_command():
    frame = _frame(0, 1, 0.0, 0, [], None)
    frame["generated_command"] = _command(0.5)
    frame["sent_command"] = _command(0.2)
    score = score_records([frame], session_id="s")
    policy = {
        "schema": "aigp-vq2-replay-policy/1",
        "metrics": {
            "labeled_frames": {"min": 1},
            "perception.gate_recall": {"min": 0.9},
            "perception.unreacquired_count": {"max": 0},
            "open_loop_commands.replay_frames.envelope_violation_count": {"max": 0},
            "open_loop_commands.replay_frames.generated_sent_max_abs_error": {"max": 0.01},
        },
    }
    result = evaluate_score_policy(score, policy)
    assert not result["passed"]
    reasons = {item["metric"]: item["reason"] for item in result["violations"]}
    assert reasons["labeled_frames"] == "below_min"
    assert reasons["perception.gate_recall"] == "missing_or_nonfinite"
    assert reasons["open_loop_commands.replay_frames.envelope_violation_count"] == "above_max"
    assert reasons["open_loop_commands.replay_frames.generated_sent_max_abs_error"] == "above_max"


def test_default_policy_fails_closed_without_full_behavior_annotation_coverage():
    frame = _frame(0, 1, 0.0, 0, [_gate((2, 2))], None)
    frame["generated_command"] = _command(0.2)
    score = score_records(
        [frame],
        session_id="s",
        annotations={
            ("s", 0, 1): {"gates": [{"center_px": [2, 2]}]},
        },
    )
    policy = json.loads(
        (Path(__file__).parents[1] / "config" / "vq2_replay_policy.example.json").read_text(
            encoding="utf-8"
        )
    )

    result = evaluate_score_policy(score, policy)

    assert result["passed"] is False
    violations = {item["metric"] for item in result["violations"]}
    assert "estimator.rpy_labeled_frames" in violations
    assert (
        "open_loop_commands.replay_frames.expected_command_labeled_frames"
        in violations
    )


@pytest.mark.parametrize(
    "token,match",
    [
        (("other-session", 0, 1), "wrong-session"),
        (("s", 0, 2), "orphan"),
    ],
)
def test_score_rejects_unconsumed_annotation_tokens(token, match):
    with pytest.raises(ValueError, match=match):
        score_records(
            [_frame(0, 1, 0.0, 0, [], None)],
            session_id="s",
            annotations={token: {"gates": []}},
        )


def test_estimator_missing_or_malformed_output_is_explicitly_invalid():
    missing = _frame(0, 1, 0.0, 0, [], None)
    missing["estimator"] = None
    malformed = _frame(1, 2, 0.1, 0, [], None)
    malformed["estimator"] = {
        "healthy": "false",
        "rpy_rad": [],
        "body_rates": [],
    }
    score = score_records([missing, malformed], session_id="s")
    assert score["estimator"]["missing_frame_estimates"] == 1
    assert score["estimator"]["present_frame_estimates"] == 1
    assert score["estimator"]["invalid_frame_estimates"] == 1
    assert score["estimator"]["healthy_frame_estimates"] == 0


@pytest.mark.parametrize(
    "estimator",
    [
        {"healthy": True, "rpy_rad": ["0", 0, 0], "body_rates": [0, 0, 0]},
        {"healthy": True, "rpy_rad": [0, 0, 0], "body_rates": [0, 0, 0], "extra": 1},
        {"healthy": True, "rpy_rad": [0, 0, 0]},
        {"healthy": True, "rpy_rad": [False, 0, 0], "body_rates": [0, 0, 0]},
    ],
)
def test_estimator_schema_rejects_strings_bools_unknown_and_missing(estimator):
    frame = _frame(0, 1, 0.0, 0, [], None)
    frame["estimator"] = estimator
    score = score_records([frame], session_id="s")
    assert score["estimator"]["invalid_frame_estimates"] == 1
    assert score["estimator"]["healthy_frame_estimates"] == 0


def test_versioned_annotations_and_combined_evidence_hash(tmp_path):
    from aigp_loop.replay import ANNOTATION_SCHEMA, load_annotations

    annotations = tmp_path / "labels.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "schema": ANNOTATION_SCHEMA,
                "session_id": "s",
                "generation": 0,
                "frame_id": 1,
                "gates": [
                    {
                        "center_px": [10, 10],
                        "corners_px": [[9, 9], [11, 9], [11, 11], [9, 11]],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_annotations(annotations)
    assert ("s", 0, 1) in loaded
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"session_id":"s","frame_id":1,"gates":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="annotation"):
        load_annotations(bad)
    score = {
        "dataset_hash": "d" * 64,
        "annotations_sha256": "a" * 64,
        "processor": "package:function-v1",
        "processor_code_sha256": "c" * 64,
        "score_payload_hash": "e" * 64,
    }
    policy_result = {"policy_hash": "b" * 64}
    assert len(evaluation_evidence_hash(score, policy_result)) == 64


def test_deterministic_input_hash_is_separate_from_timing_bearing_result_hash():
    score = {
        "dataset_hash": "d" * 64,
        "annotations_sha256": "a" * 64,
        "processor": "package:function-v1",
        "processor_code_sha256": "c" * 64,
        "evaluator_config_sha256": "f" * 64,
        "evaluator_source_sha256": "e" * 64,
        "isolation_wrapper_sha256": "0" * 64,
        "perception": {"detector_latency_ms_p95": 1.0},
    }
    policy = {"policy_hash": "b" * 64, "passed": True}
    first_input = evaluation_input_hash(score, policy)
    first_result = evaluation_result_hash(score, policy)
    score["perception"]["detector_latency_ms_p95"] = 2.0
    assert evaluation_input_hash(score, policy) == first_input
    assert evaluation_result_hash(score, policy) != first_result


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(extra=True),
        lambda row: row.update(session_id=1),
        lambda row: row.update(session_id=""),
        lambda row: row.update(generation=True),
        lambda row: row.update(generation="0"),
        lambda row: row.update(frame_id=-1),
        lambda row: row["gates"][0].update(extra=1),
        lambda row: row["gates"][0].update(center_px=[True, 2]),
        lambda row: row["gates"][0].update(center_px=["1", 2]),
        lambda row: row["gates"][0].update(corners_px=[[0, 0]] * 3),
        lambda row: row.update(
            expected_command={
                "roll_rate": True,
                "pitch_rate": 0,
                "yaw_rate": 0,
                "thrust": 0,
            }
        ),
        lambda row: row.update(estimator_rpy_rad=[0, float("inf"), 0]),
    ],
)
def test_annotation_schema_rejects_unknown_coerced_and_nonfinite_values(tmp_path, mutation):
    from aigp_loop.replay import ANNOTATION_SCHEMA, load_annotations

    row = {
        "schema": ANNOTATION_SCHEMA,
        "session_id": "s",
        "generation": 0,
        "frame_id": 1,
        "gates": [{"center_px": [10, 10]}],
    }
    mutation(row)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        load_annotations(path)


def test_annotation_bytes_lines_and_count_are_bounded(tmp_path, monkeypatch):
    import aigp_loop.replay as replay_module

    path = tmp_path / "oversized-labels.jsonl"
    path.write_bytes(b"x" * 65)
    monkeypatch.setattr(replay_module, "MAX_REPLAY_ANNOTATIONS_BYTES", 64)
    with pytest.raises(ValueError, match="annotations exceed"):
        replay_module.load_annotations(path)

    monkeypatch.setattr(replay_module, "MAX_REPLAY_ANNOTATIONS_BYTES", 1024)
    monkeypatch.setattr(replay_module, "MAX_REPLAY_ANNOTATION_LINE_BYTES", 16)
    with pytest.raises(ValueError, match="annotation line exceeds"):
        replay_module.load_annotations_bytes(b"{\"long\":\"xxxxxxxxxxxxxxxx\"}\n")

    monkeypatch.setattr(replay_module, "MAX_REPLAY_ANNOTATION_LINE_BYTES", 1024)
    monkeypatch.setattr(replay_module, "MAX_REPLAY_ANNOTATION_COUNT", 1)
    row = {
        "schema": replay_module.ANNOTATION_SCHEMA,
        "session_id": "s",
        "generation": 0,
        "frame_id": 1,
        "gates": [],
    }
    second = {**row, "frame_id": 2}
    payload = (json.dumps(row) + "\n" + json.dumps(second) + "\n").encode()
    with pytest.raises(ValueError, match="annotation count exceeds"):
        replay_module.load_annotations_bytes(payload)


@pytest.mark.parametrize(
    "policy",
    [
        {"schema": "aigp-vq2-replay-policy/1", "metrics": {"x": {"min": 0}}, "extra": 1},
        {"schema": "aigp-vq2-replay-policy/1", "metrics": []},
        {"schema": "aigp-vq2-replay-policy/1", "metrics": {"": {"min": 0}}},
        {"schema": "aigp-vq2-replay-policy/1", "metrics": {"x..y": {"min": 0}}},
        {"schema": "aigp-vq2-replay-policy/1", "metrics": {"x": {"min": True}}},
        {"schema": "aigp-vq2-replay-policy/1", "metrics": {"x": {"max": "1"}}},
        {"schema": "aigp-vq2-replay-policy/1", "metrics": {"x": {"min": 0, "extra": 1}}},
    ],
)
def test_policy_schema_rejects_unknown_and_coerced_values(policy):
    with pytest.raises(ValueError):
        evaluate_score_policy({"x": 1}, policy)


def test_local_processor_detection_is_scored_and_policy_gated(tmp_path):
    bundle = tmp_path / "processor.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="processor", require_private=False)
    writer.capture_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        detector_latency_ms=0.1,
        detections=[],
        tracker=None,
        imu=None,
        estimator=None,
        race_status={"active_gate_index": 0},
        generated_command=_command(0.2),
        sent_command=_command(0.2),
    )
    writer.close()
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-annotation/1",
                "session_id": "processor",
                "generation": 0,
                "frame_id": 1,
                "gates": [{"center_px": [2, 2]}],
            }
        ),
        encoding="utf-8",
    )
    score = score_bundle(
        bundle,
        annotations_path=labels,
        processor_spec=f"{__name__}:_unsafe_processor",
    )
    assert score["perception"]["gate_recall"] == 0.0
    result = evaluate_score_policy(
        score,
        {
            "schema": "aigp-vq2-replay-policy/1",
            "metrics": {
                "perception.gate_recall": {"min": 1.0}
            },
        },
    )
    assert not result["passed"]


def test_empty_processor_cannot_inherit_recorded_baseline_outputs(tmp_path):
    bundle = tmp_path / "noop.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="noop", require_private=False)
    writer.capture_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        detector_latency_ms=0.1,
        detections=[_gate((2, 2))],
        tracker={"target": {"center_x": 2, "center_y": 2}},
        imu=None,
        estimator={"healthy": True, "rpy_rad": [0, 0, 0], "body_rates": [0, 0, 0]},
        race_status={"active_gate_index": 0},
        generated_command=_command(0.2),
        sent_command=_command(0.2),
    )
    writer.close()
    with pytest.raises(TypeError, match="exactly detections"):
        score_bundle(bundle, processor_spec=f"{__name__}:_noop_processor")

    with pytest.raises(KeyError, match="detections"):
        score_bundle(bundle, processor_spec=f"{__name__}:_copy_through_processor")


def test_candidate_processor_scores_decoded_only_frames_and_full_stack_wall_latency(tmp_path):
    bundle = tmp_path / "decoded-only.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="decoded", require_private=False)
    writer.capture_decoded_frame(
        _image(),
        generation=0,
        frame_id=7,
        sim_time_ns=70,
        received_monotonic_s=1.0,
    )
    writer.close()
    labels = tmp_path / "decoded-labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-annotation/1",
                "session_id": "decoded",
                "generation": 0,
                "frame_id": 7,
                "gates": [{"center_px": [2, 2]}],
            }
        ),
        encoding="utf-8",
    )
    score = score_bundle(
        bundle,
        annotations_path=labels,
        processor_spec=f"{__name__}:_matching_processor",
    )
    assert score["frames"] == 1
    assert score["perception"]["gate_recall"] == 1.0

    slow = score_bundle(
        bundle, processor_spec=f"{__name__}:_slow_processor_claiming_zero"
    )
    assert slow["perception"]["detector_latency_ms_p50"] is None
    assert slow["perception"]["full_stack_latency_ms_p50"] >= 15.0
    with pytest.raises(ValueError, match="detection has an invalid"):
        score_bundle(bundle, processor_spec=f"{__name__}:_coercing_processor")


def test_multi_session_corpus_applies_each_policy_and_combines_input_identity(tmp_path):
    sessions = []
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-policy/1",
                "metrics": {
                    "labeled_frames": {"min": 2},
                    "perception.gate_recall": {"min": 1.0},
                    "perception.center_error_px_p95": {"max": 0.0},
                    "perception.temporal_center_step_px_p95": {"max": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )
    for index in range(2):
        session_id = f"golden-{index}"
        bundle = tmp_path / f"{session_id}.vq2replay"
        writer = ReplayBundleWriter(
            bundle, session_id=session_id, require_private=False
        )
        for frame_id in (1, 2):
            writer.capture_decoded_frame(
                _image(index),
                generation=0,
                frame_id=frame_id,
                sim_time_ns=frame_id,
                received_monotonic_s=float(frame_id),
            )
        writer.close()
        annotations = tmp_path / f"{session_id}.jsonl"
        annotations.write_text(
            "\n".join(
                json.dumps(
                    {
                        "schema": "aigp-vq2-replay-annotation/1",
                        "session_id": session_id,
                        "generation": 0,
                        "frame_id": frame_id,
                        "gates": [{"center_px": [2, 2]}],
                    }
                )
                for frame_id in (1, 2)
            )
            + "\n",
            encoding="utf-8",
        )
        sessions.append(
            {
                "session_id": session_id,
                "bundle": bundle.name,
                "annotations": annotations.name,
                "policy": policy.name,
            }
        )
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            {"schema": "aigp-vq2-replay-corpus/1", "sessions": sessions}
        ),
        encoding="utf-8",
    )
    score = score_corpus(
        manifest, processor_spec=f"{__name__}:_matching_processor"
    )
    assert score["policy"]["passed"] is True
    assert score["aggregate"]["session_count"] == 2
    assert score["aggregate"]["gate_recall"] == 1.0
    assert len(score["sessions"]) == 2
    assert all(session["policy"]["passed"] for session in score["sessions"])
    assert score["evaluation_input_hash"] == score["evaluation_evidence_hash"]
    assert score["evaluation_result_hash"] != score["evaluation_input_hash"]


def test_corpus_manifest_and_policy_files_are_bounded_before_scoring(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module

    oversized_manifest = tmp_path / "oversized-corpus.json"
    oversized_manifest.write_bytes(b"x" * 65)
    monkeypatch.setattr(replay_module, "MAX_REPLAY_CORPUS_MANIFEST_BYTES", 64)
    with pytest.raises(ValueError, match="corpus manifest exceeds"):
        score_corpus(oversized_manifest, processor_spec="candidate:process")

    monkeypatch.setattr(
        replay_module, "MAX_REPLAY_CORPUS_MANIFEST_BYTES", 1024 * 1024
    )
    monkeypatch.setattr(replay_module, "MAX_REPLAY_POLICY_BYTES", 64)
    bundle = tmp_path / "placeholder-bundle"
    bundle.mkdir()
    annotations = tmp_path / "labels.jsonl"
    annotations.write_text("", encoding="utf-8")
    policy = tmp_path / "oversized-policy.json"
    policy.write_bytes(b"x" * 65)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-corpus/1",
                "sessions": [
                    {
                        "session_id": "s",
                        "bundle": bundle.name,
                        "annotations": annotations.name,
                        "policy": policy.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corpus policy exceeds"):
        score_corpus(manifest, processor_spec="candidate:process")


def test_full_stack_example_policy_passes_end_to_end(tmp_path):
    bundle = tmp_path / "example-policy.vq2replay"
    writer = ReplayBundleWriter(
        bundle, session_id="example-policy", require_private=False
    )
    for frame_id, gate_index in ((1, 0), (2, 1), (3, 1)):
        writer.capture_frame(
            _image(),
            generation=0,
            frame_id=frame_id,
            sim_time_ns=frame_id,
            received_monotonic_s=frame_id * 0.1,
            detector_latency_ms=0.1,
            detections=[],
            tracker=None,
            imu=None,
            estimator=None,
            race_status={"active_gate_index": gate_index},
            generated_command=None,
            sent_command=None,
        )
    writer.close()
    labels = tmp_path / "example-policy.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": "aigp-vq2-replay-annotation/1",
                    "session_id": "example-policy",
                    "generation": 0,
                    "frame_id": frame_id,
                        "gates": [
                            {
                                "center_px": [2, 2],
                                "corners_px": [[1, 1], [3, 1], [3, 3], [1, 3]],
                                "gate_index": gate_index,
                            }
                        ],
                        "active_gate_index": gate_index,
                        "estimator_rpy_rad": [0.2, -0.1, 0.3],
                        "expected_estimator_healthy": True,
                        "expected_command": _command(0.2),
                }
            )
                    for frame_id, gate_index in ((1, 0), (2, 1), (3, 1))
        )
        + "\n",
        encoding="utf-8",
    )
    score = score_bundle(
        bundle,
        annotations_path=labels,
        processor_spec=f"{__name__}:_matching_corners_processor",
    )
    policy = json.loads(
        (Path(__file__).parents[1] / "config" / "vq2_replay_policy.example.json").read_text(
            encoding="utf-8"
        )
    )
    policy_result = evaluate_score_policy(score, policy)
    assert policy_result["passed"] is True, policy_result["violations"]
    score["policy"] = policy_result
    assert replay_promotion_policy_failures(score) == ()


def test_policy_rejects_constant_healthy_estimator_and_zero_command(tmp_path):
    bundle = tmp_path / "constant-zero-policy.vq2replay"
    writer = ReplayBundleWriter(
        bundle, session_id="constant-zero-policy", require_private=False
    )
    writer.capture_decoded_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
    )
    writer.close()
    labels = tmp_path / "constant-zero-policy.jsonl"
    labels.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-annotation/1",
                "session_id": "constant-zero-policy",
                "generation": 0,
                "frame_id": 1,
                "gates": [
                    {
                        "center_px": [2, 2],
                        "corners_px": [[1, 1], [3, 1], [3, 3], [1, 3]],
                        "gate_index": 0,
                    }
                ],
                "active_gate_index": 0,
                "estimator_rpy_rad": [0.2, -0.1, 0.3],
                "expected_estimator_healthy": True,
                "expected_command": _command(0.2),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    score = score_bundle(
        bundle,
        annotations_path=labels,
        processor_spec=f"{__name__}:_constant_zero_behavior_processor",
    )
    policy = json.loads(
        (Path(__file__).parents[1] / "config" / "vq2_replay_policy.example.json").read_text(
            encoding="utf-8"
        )
    )

    result = evaluate_score_policy(score, policy)

    assert result["passed"] is False
    violations = {item["metric"] for item in result["violations"]}
    assert "estimator.rpy_rmse_rad" in violations
    assert "estimator.health_mismatch_count" not in violations
    assert "open_loop_commands.replay_frames.expected_command_rmse" in violations


def test_ordered_processor_receives_sanitized_sensor_events_and_owns_estimator_command(tmp_path):
    _PRIVACY_CONTEXTS.clear()
    bundle = tmp_path / "ordered.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="ordered", require_private=False)
    writer.record_imu(
        {"accel": [0, 0, 0]},
        estimator={
            "healthy": False,
            "rpy_rad": [9, 9, 9],
            "body_rates": [9, 9, 9],
        },
    )
    writer.record_command("generated", _command(0.35))
    writer.capture_decoded_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
    )
    writer.close()
    score = score_bundle(
        bundle, processor_spec=f"{__name__}:_privacy_probe_processor"
    )
    assert len(_PRIVACY_CONTEXTS) == 1
    events = _PRIVACY_CONTEXTS[0]["sensor_events"]
    assert _PRIVACY_CONTEXTS[0]["schema"] == "aigp-vq2-full-stack-context/1"
    assert len(events) == 1
    assert "imu" in events[0] and "estimator" not in events[0]
    assert score["estimator"]["healthy_frame_estimates"] == 1
    assert score["open_loop_commands"]["replay_frames"]["generated_count"] == 1


def test_candidate_context_rejects_derived_aliases_from_arbitrary_events(tmp_path):
    _PRIVACY_CONTEXTS.clear()
    bundle = tmp_path / "malicious-aliases.vq2replay"
    writer = ReplayBundleWriter(
        bundle, session_id="malicious-aliases", require_private=False
    )
    derived_aliases = {
        "accepted_target": {"center_px": [2, 2]},
        "tracker_target": {"bbox_xywh_px": [1, 1, 2, 2]},
        "selected_detection_index": 0,
        "last_command": _command(0.2),
        "target": {"center_px": [2, 2], "confidence": 1.0},
        "center_px": [2, 2],
        "bbox_xywh_px": [1, 1, 2, 2],
        "confidence": 1.0,
        "rpy": [1, 2, 3],
        "body_rates": [4, 5, 6],
        "gate_reacquisition_latency_ms": 0.0,
    }
    writer.record_event("diagnostic_tick", **derived_aliases)
    writer.record_imu(
        {
            "accel": [0, 0, 0],
            "body_rates": [4, 5, 6],
            "target": {"center_px": [2, 2]},
        },
        estimator={"rpy": [1, 2, 3]},
        received_monotonic_s=0.1,
    )
    writer.record_race(
        {"active_gate_index": 1, **derived_aliases},
        received_monotonic_s=0.2,
    )
    writer.capture_decoded_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=0.3,
    )
    writer.close()

    score_bundle(bundle, processor_spec=f"{__name__}:_privacy_probe_processor")

    assert len(_PRIVACY_CONTEXTS) == 1
    context = _PRIVACY_CONTEXTS[0]
    assert "phase" not in context
    assert context["imu"] == {"accel": (0, 0, 0)}
    assert context["race_status"] == {"active_gate_index": 1}
    assert [event["type"] for event in context["sensor_events"]] == [
        "imu",
        "race_status",
    ]
    serialized = json.dumps(dict(context), default=list)
    for alias in derived_aliases:
        assert f'"{alias}"' not in serialized


def test_matching_control_state_does_not_leak_backward_to_decoded_sequence(tmp_path):
    _PRIVACY_CONTEXTS.clear()
    bundle = tmp_path / "causal.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="causal", require_private=False)
    writer.capture_decoded_frame(
        _image(), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.capture_frame(
        _image(),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.1,
        detector_latency_ms=0.1,
        detections=[],
        tracker=None,
        imu={"accel": [1, 2, 3]},
        estimator=None,
        race_status={"active_gate_index": 4},
        generated_command=None,
        sent_command=None,
        phase="race",
    )
    writer.close()
    score_bundle(bundle, processor_spec=f"{__name__}:_privacy_probe_processor")
    context = _PRIVACY_CONTEXTS[0]
    assert context["decoded_sequence"] == 0
    assert "evaluation_sequence" not in context
    assert context["race_status"] is None
    assert "phase" not in context


def test_processor_seed_is_applied_before_each_replay_and_changes_output(tmp_path):
    def run(seed):
        bundle = tmp_path / f"seed-{seed}.vq2replay"
        writer = ReplayBundleWriter(
            bundle,
            session_id=f"seed-{seed}",
            metadata={"seed": seed},
            require_private=False,
        )
        writer.capture_decoded_frame(
            _image(), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
        )
        writer.close()
        labels = tmp_path / f"seed-{seed}.jsonl"
        labels.write_text(
            json.dumps(
                {
                    "schema": "aigp-vq2-replay-annotation/1",
                    "session_id": f"seed-{seed}",
                    "generation": 0,
                    "frame_id": 1,
                    "gates": [{"center_px": [0.0, 0.0]}],
                }
            ),
            encoding="utf-8",
        )
        return score_bundle(
            bundle,
            annotations_path=labels,
            processor_spec=f"{__name__}:_seeded_random_processor",
        )["perception"]["center_error_px_mean"]

    first = run(7)
    repeat_bundle = tmp_path / "seed-repeat.vq2replay"
    writer = ReplayBundleWriter(
        repeat_bundle,
        session_id="seed-repeat",
        metadata={"seed": 7},
        require_private=False,
    )
    writer.capture_decoded_frame(
        _image(), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.close()
    labels = tmp_path / "seed-repeat.jsonl"
    labels.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-annotation/1",
                "session_id": "seed-repeat",
                "generation": 0,
                "frame_id": 1,
                "gates": [{"center_px": [0.0, 0.0]}],
            }
        ),
        encoding="utf-8",
    )
    repeated = score_bundle(
        repeat_bundle,
        annotations_path=labels,
        processor_spec=f"{__name__}:_seeded_random_processor",
    )["perception"]["center_error_px_mean"]
    assert repeated == first
    assert run(8) != first


def test_candidate_isolation_fails_closed_on_untrusted_wrapper_attestation(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_file

    wrapper = tmp_path / "wrapper.exe"
    wrapper.write_bytes(b"not-an-executable")
    _patch_wrapper_attestation(
        monkeypatch,
        replay_module,
        wrapper,
        {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "allowed",
            "filesystem": "all-files",
            "non_interactive": True,
        },
    )
    with pytest.raises(RuntimeError, match="did not attest"):
        IsolatedReplayProcessor(
            f"{__name__}:_matching_processor", wrapper, sha256_file(wrapper)
        )


def test_candidate_isolation_rejects_wrapper_replacement_after_attestation(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_file

    wrapper = tmp_path / "replaceable-wrapper.exe"
    wrapper.write_bytes(b"reviewed-wrapper")
    expected_hash = sha256_file(wrapper)

    def attest(argv, **_kwargs):
        assert argv == [str(wrapper.resolve()), "--attest"]
        wrapper.write_bytes(b"replacement-wrapper")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "aigp-replay-isolation-attestation/1",
                    "network": "denied",
                    "filesystem": "readonly-worktree-only",
                    "non_interactive": True,
                    "process_tree_containment": "kill-on-wrapper-exit",
                    "host_process_access": "denied",
                }
            ),
        )

    monkeypatch.setattr(replay_module.subprocess, "run", attest)
    monkeypatch.setattr(
        replay_module, "_candidate_worktree_root", lambda _value: Path.cwd().resolve()
    )
    monkeypatch.setattr(
        replay_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replaced wrapper must never launch")
        ),
    )
    with pytest.raises(RuntimeError, match="changed after attestation"):
        IsolatedReplayProcessor(
            f"{__name__}:_matching_processor", wrapper, expected_hash
        )


def test_candidate_isolation_rejects_symlinked_wrapper_path(tmp_path):
    from aigp_loop._util import sha256_file

    target = tmp_path / "real-wrapper.exe"
    target.write_bytes(b"reviewed-wrapper")
    wrapper = tmp_path / "linked-wrapper.exe"
    try:
        wrapper.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")
    with pytest.raises(ValueError, match="missing or mismatched"):
        IsolatedReplayProcessor(
            f"{__name__}:_matching_processor", wrapper, sha256_file(target)
        )


def test_promotion_eligible_processor_rejects_ignored_local_module(tmp_path):
    import aigp_loop.replay as replay_module

    root = tmp_path / "candidate"
    root.mkdir()
    (root / ".gitignore").write_text("ignored_processor.py\n", encoding="utf-8")
    (root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _commit_test_repository(root)
    (root / "ignored_processor.py").write_text(
        "def run(image, context):\n    return {}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exact pristine checkout"):
        replay_module._processor_code_hash(
            "ignored_processor:run", root, require_pristine=True
        )


def test_processor_provenance_rejects_namespace_package_parent(tmp_path):
    import aigp_loop.replay as replay_module

    root = tmp_path / "candidate"
    package = root / "namespace_parent"
    package.mkdir(parents=True)
    (package / "processor.py").write_text(
        "def run(image, context):\n    return {}\n", encoding="utf-8"
    )
    _commit_test_repository(root)

    with pytest.raises(ValueError, match="secure local package parents"):
        replay_module._processor_code_hash(
            "namespace_parent.processor:run", root, require_pristine=True
        )


def test_promotion_eligible_processor_rejects_symlinked_parent_package(tmp_path):
    import aigp_loop.replay as replay_module

    root = tmp_path / "candidate"
    package = root / "candidate_package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    (package / "processor.py").write_text(
        "def run(image, context):\n    return {}\n", encoding="utf-8"
    )
    _commit_test_repository(root)
    external = tmp_path / "external-package"
    package.rename(external)
    try:
        package.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        external.rename(package)
        pytest.skip(f"directory symlinks unavailable on this host: {exc}")

    with pytest.raises(ValueError, match="symlink|reparse"):
        replay_module._processor_code_hash(
            "candidate_package.processor:run", root, require_pristine=True
        )


def test_isolated_worker_receives_no_inherited_secret_or_python_hooks(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_file

    wrapper = tmp_path / "wrapper.exe"
    wrapper.write_bytes(b"reviewed-wrapper")
    _patch_wrapper_attestation(
        monkeypatch,
        replay_module,
        wrapper,
        {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
        },
    )
    monkeypatch.setattr(
        replay_module, "_candidate_worktree_root", lambda _value: Path.cwd().resolve()
    )
    monkeypatch.setenv("AIGP_REPLAY_TEST_SECRET", "must-not-leak")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "malicious-hooks"))
    captured = {"environment": {}}

    def refuse_launch(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["environment"].update(kwargs["env"])
        raise OSError("stop after environment capture")

    monkeypatch.setattr(replay_module.subprocess, "Popen", refuse_launch)
    with pytest.raises(OSError, match="environment capture"):
        IsolatedReplayProcessor(
            f"{__name__}:_matching_processor", wrapper, sha256_file(wrapper), seed=-1
        )

    environment = captured["environment"]
    folded = {name.casefold() for name in environment}
    assert "aigp_replay_test_secret" not in folded
    assert "pythonpath" not in folded
    assert "pythonhome" not in folded
    assert environment["AIGP_TRIAL_OFFLINE"] == "1"
    assert environment["AIGP_REPLAY_CANDIDATE"] == "1"
    assert environment["AIGP_REPLAY_SEED"] == "-1"
    assert environment["PYTHONHASHSEED"] == str(0xFFFFFFFF)
    argv = captured["argv"]
    assert argv[:4] == [str(wrapper.resolve()), "--", sys.executable, "-I"]
    worker = Path(argv[4])
    assert worker.is_absolute()
    assert worker == Path.cwd().resolve() / "scripts" / "aigp_replay_worker.py"
    assert "-m" not in argv
    assert argv[-1] == "tests/test_aigp_loop_replay.py"


def test_real_worker_refreezes_context_after_json_transport(tmp_path):
    root = Path(__file__).resolve().parents[1]
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    worker = candidate_root / "scripts" / "aigp_replay_worker.py"
    worker.parent.mkdir()
    worker.write_text(
        (root / "scripts" / "aigp_replay_worker.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    candidate = candidate_root / "immutable_candidate.py"
    candidate.write_text(
        """
def process(image, context):
    try:
        context[\"new\"] = 1
    except TypeError:
        pass
    else:
        raise AssertionError(\"top-level context is mutable\")
    try:
        context[\"nested\"][\"new\"] = 1
    except TypeError:
        pass
    else:
        raise AssertionError(\"nested context is mutable\")
    try:
        context[\"items\"].append(2)
    except AttributeError:
        pass
    else:
        raise AssertionError(\"context list stayed mutable\")
    try:
        image[0, 0, 0] = 255
    except ValueError:
        pass
    else:
        raise AssertionError(\"image stayed mutable\")
    return {\"immutable\": True}
""".lstrip(),
        encoding="utf-8",
    )
    image_buffer = io.BytesIO()
    np.save(image_buffer, _image(), allow_pickle=False)
    request = {
        "schema": "aigp-replay-worker-request/1",
        "request_id": 0,
        "image_npy_base64": base64.b64encode(image_buffer.getvalue()).decode(
            "ascii"
        ),
        "context": {"nested": {"value": 1}, "items": [1]},
    }
    malicious = tmp_path / "malicious-site"
    (malicious / "scripts").mkdir(parents=True)
    (malicious / "scripts" / "__init__.py").write_text(
        "raise AssertionError('site scripts package was imported')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(malicious)

    run = subprocess.run(
        [
            sys.executable,
            "-I",
            str(worker),
            "immutable_candidate:process",
            "0",
            "immutable_candidate.py",
        ],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=candidate_root,
        env=environment,
        timeout=15.0,
        shell=False,
    )

    assert run.returncode == 0, run.stderr
    response = json.loads(run.stdout)
    assert response["result"] == {"immutable": True}

    decoy = candidate_root / "decoy.py"
    decoy.write_text("VALUE = 1\n", encoding="utf-8")
    origin_mismatch = subprocess.run(
        [
            sys.executable,
            "-I",
            str(worker),
            "immutable_candidate:process",
            "0",
            "decoy.py",
        ],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=candidate_root,
        env=environment,
        timeout=15.0,
        shell=False,
    )
    assert origin_mismatch.returncode == 4
    assert origin_mismatch.stdout == ""


def test_score_cli_refuses_candidate_without_os_isolation_wrapper(tmp_path):
    bundle = tmp_path / "isolation-required.vq2replay"
    ReplayBundleWriter(bundle, require_private=False).close()
    with pytest.raises(SystemExit):
        replay_main(
            [
                "score",
                str(bundle),
                "--processor",
                f"{__name__}:_matching_processor",
            ]
        )


@pytest.mark.parametrize(
    "relative",
    [
        "aigp_loop/replay/__init__.py",
        "aigp_loop/replay.PYD",
        "aigp_loop/__INIT__.PYD",
        "aigp_loop/__init__.PYC",
        "aigp_loop.PYD",
    ],
)
def test_replay_host_rejects_import_boundary_alternatives(tmp_path, relative):
    import scripts.aigp_replay as replay_script

    root = tmp_path / "trusted-host"
    for reviewed in replay_script._REPLAY_HOST_FILES:
        target = root / reviewed
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(reviewed, encoding="utf-8")
    collision = root / relative
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_bytes(b"untrusted import alternative")
    with pytest.raises(ValueError, match="import-boundary collision"):
        replay_script._reject_replay_host_import_collisions(root)


def test_replay_host_self_verifies_stable_manifest_sources(tmp_path, monkeypatch):
    import scripts.aigp_replay as replay_script

    root = tmp_path / "trusted-host"
    hashes = {}
    for reviewed in replay_script._REPLAY_HOST_FILES:
        target = root / reviewed
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(reviewed, encoding="utf-8")
        hashes[reviewed] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = root / replay_script._TRUSTED_MANIFEST_PATH
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"schema": "aigp-trusted-evaluator-files/1", "files": hashes}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_script, "_REPO", root.resolve())
    replay_script._verify_replay_host_manifest(
        replay_script._TRUSTED_MANIFEST_PATH
    )
    (root / "aigp_loop" / "replay.py").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_script._verify_replay_host_manifest(
            replay_script._TRUSTED_MANIFEST_PATH
        )

    with pytest.raises(ValueError, match="must be"):
        replay_script._verify_replay_host_manifest(str(manifest.resolve()))


def test_replay_host_uses_external_fresh_bytecode_prefix():
    import scripts.aigp_replay as replay_script

    prefix = replay_script._activate_fresh_pycache_prefix(replay_script._REPO)
    assert prefix.is_dir()
    assert replay_script._PYCACHE_PREFIX == prefix
    assert replay_script._REPO != prefix
    assert replay_script._REPO not in prefix.parents


def test_replay_host_rejects_untracked_casefolded_bytecode_without_git(tmp_path):
    import scripts.aigp_replay as replay_script

    root = tmp_path / "trusted-host"
    (root / "scripts").mkdir(parents=True)
    cache = root / "aigp_loop" / "__PYCACHE__"
    cache.mkdir(parents=True)
    (cache / "replay.CPYTHON-312.PYC").write_bytes(b"untracked bytecode")
    with pytest.raises(ValueError, match="contains bytecode"):
        replay_script._reject_replay_host_bytecode(root)


def test_replay_host_parses_effective_argv_before_evidence_bootstrap(
    tmp_path, monkeypatch
):
    import scripts.aigp_replay as replay_script

    observed = {}

    def stop_after_bootstrap(manifest, *, promotion_evidence):
        observed.update(
            manifest=manifest, promotion_evidence=promotion_evidence
        )
        raise RuntimeError("bootstrap reached")

    monkeypatch.setattr(
        replay_script, "_bootstrap_trusted_replay_host", stop_after_bootstrap
    )
    with pytest.raises(RuntimeError, match="bootstrap reached"):
        replay_script.main(
            [
                "score",
                str(tmp_path / "bundle"),
                "--processor=candidate:run",
                "--isolation-wrapper=C:/trusted/wrapper.exe",
                f"--isolation-wrapper-sha256={'0' * 64}",
                f"--candidate-worktree={tmp_path}",
                "--trusted-manifest=config/promotion_trusted_files.json",
            ]
        )
    assert observed == {
        "manifest": replay_script._TRUSTED_MANIFEST_PATH,
        "promotion_evidence": True,
    }


def test_replay_script_import_does_not_preimport_repository_package():
    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import runpy,sys; "
                "assert 'aigp_loop' not in sys.modules; "
                "runpy.run_path(sys.argv[1], run_name='replay_bootstrap_probe'); "
                "assert 'aigp_loop' not in sys.modules"
            ),
            str(root / "scripts" / "aigp_replay.py"),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=15.0,
        shell=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_replay_host_rejects_abbreviated_or_noncanonical_manifest_before_bootstrap(
    tmp_path, monkeypatch
):
    import scripts.aigp_replay as replay_script

    called = False

    def bootstrap(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        replay_script, "_bootstrap_trusted_replay_host", bootstrap
    )
    base = [
        "score",
        str(tmp_path / "bundle"),
        "--processor",
        "candidate:run",
        "--isolation-wrapper",
        "C:/trusted/wrapper.exe",
        "--isolation-wrapper-sha256",
        "0" * 64,
        "--candidate-worktree",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        replay_script.main([*base, "--trusted-manif", "config/promotion_trusted_files.json"])
    with pytest.raises(SystemExit):
        replay_script.main([*base, "--trusted-manifest", "config/alternate.json"])
    assert called is False


def test_evaluator_identity_exactly_binds_loaded_trusted_host_sources(tmp_path):
    import scripts.aigp_replay as replay_script

    bundle = tmp_path / "worker-provenance.vq2replay"
    ReplayBundleWriter(bundle, require_private=False).close()
    score = score_bundle(bundle)
    assert set(score["evaluator_identity"]["sources_sha256"]) == set(
        replay_script._REPLAY_HOST_FILES
    )
    assert "scripts/aigp_replay_worker.py" not in replay_script._REPLAY_HOST_FILES


def test_score_cli_policy_returns_nonzero_on_missing_required_evidence(tmp_path, capsys):
    bundle = tmp_path / "cli.vq2replay"
    ReplayBundleWriter(bundle, require_private=False).close()
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-policy/1",
                "metrics": {"labeled_frames": {"min": 1}},
            }
        ),
        encoding="utf-8",
    )
    assert replay_main(["score", str(bundle), "--policy", str(policy)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["policy"]["passed"] is False


def test_score_cli_parses_and_hashes_one_exact_policy_snapshot(
    tmp_path, capsys, monkeypatch
):
    import scripts.aigp_replay as replay_script
    from aigp_loop._util import sha256_bytes

    bundle = tmp_path / "policy-snapshot.vq2replay"
    ReplayBundleWriter(bundle, require_private=False).close()
    policy = tmp_path / "policy.json"
    first = json.dumps(
        {
            "schema": "aigp-vq2-replay-policy/1",
            "metrics": {"labeled_frames": {"min": 0}},
        }
    ).encode("utf-8")
    replacement = json.dumps(
        {
            "schema": "aigp-vq2-replay-policy/1",
            "metrics": {"labeled_frames": {"min": 999}},
        }
    ).encode("utf-8")
    policy.write_bytes(first)
    real_read = replay_script.read_secure_regular_file

    def replace_after_snapshot(path, **kwargs):
        payload = real_read(path, **kwargs)
        policy.write_bytes(replacement)
        return payload

    monkeypatch.setattr(
        replay_script, "read_secure_regular_file", replace_after_snapshot
    )
    replay_script.main(["score", str(bundle), "--policy", str(policy)])
    output = json.loads(capsys.readouterr().out)
    assert output["policy"]["constraints"] == {"labeled_frames": {"min": 0}}
    assert output["policy"]["policy_file_sha256"] == sha256_bytes(first)


def test_score_cli_policy_ceiling_survives_small_to_oversized_path_swap(
    tmp_path, monkeypatch
):
    import aigp_loop._util as util
    import scripts.aigp_replay as replay_script

    bundle = tmp_path / "policy-resource-swap.vq2replay"
    ReplayBundleWriter(bundle, require_private=False).close()
    policy = tmp_path / "policy.json"
    replacement = tmp_path / "oversized-policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "aigp-vq2-replay-policy/1",
                "metrics": {"labeled_frames": {"min": 0}},
            }
        ),
        encoding="utf-8",
    )
    replacement.write_bytes(b"x" * (replay_script.MAX_REPLAY_POLICY_BYTES + 1))
    real_secure = util.secure_regular_file
    swapped = False

    def replace_policy_after_path_check(path):
        nonlocal swapped
        checked = real_secure(path)
        if not swapped and checked == policy.resolve():
            replacement.replace(policy)
            swapped = True
        return checked

    monkeypatch.setattr(util, "secure_regular_file", replace_policy_after_path_check)
    with pytest.raises(ValueError, match="policy exceeds resource limit"):
        replay_script.main(["score", str(bundle), "--policy", str(policy)])
    assert swapped is True


class _BlockingWriter:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.aborted = None
        self.invalidated = None
        self.thread_names = []

    def capture_decoded_frame(self, *_args, **_kwargs):
        self.thread_names.append(threading.current_thread().name)
        self.started.set()
        self.release.wait(2.0)

    def record_event(self, *_args, **_kwargs):
        pass

    def close(self, **_kwargs):
        return "hash"

    def abort(self, reason):
        self.aborted = reason

    def mark_invalid(self, reason):
        self.invalidated = reason


def test_async_capture_never_writes_on_caller_and_timeout_is_bounded_fail_closed():
    writer = _BlockingWriter()
    recorder = AsyncReplayRecorder(writer, max_queue_records=1)
    snapshot = SimpleNamespace(
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        camera_frame=SimpleNamespace(image=_image()),
    )
    assert recorder.capture_decoded_snapshot(snapshot)
    assert writer.started.wait(1.0)
    assert recorder.record_event("queued")
    assert not recorder.record_event("overflow")
    started = time.monotonic()
    stats = recorder.close(expected_decoded_frames=1, timeout_s=0.05)
    assert time.monotonic() - started < 0.3
    assert not stats.complete
    assert "finalization timeout" in stats.failure_reason
    assert stats.dropped == 1
    assert writer.thread_names == ["aigp-replay-writer"]
    writer.release.set()
    deadline = time.monotonic() + 1.0
    while writer.aborted is None and writer.invalidated is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.aborted is not None or writer.invalidated is not None


def test_async_enqueue_snapshots_mutable_telemetry_before_writer_release():
    class Writer:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.received = None

        def record_imu(self, imu, **kwargs):
            self.started.set()
            assert self.release.wait(1.0)
            self.received = (imu, kwargs)

        def close(self, **_kwargs):
            return "dataset"

        def abort(self, _reason):
            raise AssertionError("snapshot test must close losslessly")

        def mark_invalid(self, _reason):
            raise AssertionError("snapshot test must not invalidate")

    writer = Writer()
    recorder = AsyncReplayRecorder(writer)
    imu = {"accel": [1.0, 2.0, 3.0]}
    estimator = {"healthy": True, "rpy_rad": [0.0, 0.1, 0.2]}
    assert recorder.record_imu(imu, estimator=estimator)
    assert writer.started.wait(1.0)
    imu["accel"][0] = 99.0
    estimator["rpy_rad"][1] = 99.0
    writer.release.set()
    stats = recorder.close()
    assert stats.complete is True
    assert writer.received[0] == {"accel": [1.0, 2.0, 3.0]}
    assert writer.received[1]["estimator"] == {
        "healthy": True,
        "rpy_rad": [0.0, 0.1, 0.2],
    }


def test_async_enqueue_copies_readonly_view_with_writable_base_alias():
    class Writer:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.received = None

        def capture_decoded_frame(self, image, **_kwargs):
            self.started.set()
            assert self.release.wait(1.0)
            self.received = np.array(image, copy=True)

        def close(self, **_kwargs):
            return "dataset"

        def abort(self, _reason):
            raise AssertionError("alias snapshot must close losslessly")

        def mark_invalid(self, _reason):
            raise AssertionError("alias snapshot must not invalidate")

    writer = Writer()
    recorder = AsyncReplayRecorder(writer)
    writable_base = _image(1)
    readonly_view = writable_base.view()
    readonly_view.setflags(write=False)
    snapshot = SimpleNamespace(
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        camera_frame=SimpleNamespace(image=readonly_view),
    )
    assert recorder.capture_decoded_snapshot(snapshot)
    assert writer.started.wait(1.0)
    writable_base[:] = 99
    writer.release.set()
    stats = recorder.close(expected_decoded_frames=1)
    assert stats.complete is True
    assert np.all(writer.received == 1)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_async_queue_size_requires_positive_exact_integer(tmp_path, value):
    writer = ReplayBundleWriter(
        tmp_path / f"queue-{value}.vq2replay", require_private=False
    )
    with pytest.raises(ValueError, match="exact integer"):
        AsyncReplayRecorder(writer, max_queue_records=value)
    writer.abort("constructor rejected")


@pytest.mark.parametrize("field,value", [("generation", True), ("frame_id", "1"), ("sim_time_ns", -1)])
def test_async_decoded_callback_rejects_invalid_tokens_without_raising(tmp_path, field, value):
    bundle = tmp_path / f"callback-{field}.vq2replay"
    recorder = AsyncReplayRecorder(ReplayBundleWriter(bundle, require_private=False))
    values = {"generation": 0, "frame_id": 1, "sim_time_ns": 1}
    values[field] = value
    snapshot = SimpleNamespace(
        **values,
        received_monotonic_s=1.0,
        camera_frame=SimpleNamespace(image=_image()),
    )
    assert recorder.capture_decoded_snapshot(snapshot) is False
    stats = recorder.close()
    assert stats.complete is False
    assert "invalid exact" in stats.failure_reason


def test_late_async_mutations_are_idempotently_rejected_after_complete_seal(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module

    bundle = tmp_path / "late-idempotent.vq2replay"
    recorder = AsyncReplayRecorder(ReplayBundleWriter(bundle, require_private=False))
    first = recorder.close(expected_decoded_frames=0)
    assert first.complete is True
    assert recorder.fail("too late") is False
    assert recorder.record_event("too-late") is False
    assert recorder.record_event("invalid-too-late", bad=object()) is False
    monkeypatch.setattr(
        replay_module.np,
        "array",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed recorder must not copy a late writable image")
        ),
    )
    assert recorder.capture_frame(
        _image(), generation=0, frame_id=1, sim_time_ns=1,
        received_monotonic_s=1.0,
    ) is False
    second = recorder.close(expected_decoded_frames=99)
    assert second == first
    ReplayBundleReader(bundle).verify()


def test_decoded_callback_count_mismatch_seals_bundle_incomplete(tmp_path):
    bundle = tmp_path / "incomplete.vq2replay"
    recorder = AsyncReplayRecorder(
        ReplayBundleWriter(bundle, require_private=False), max_queue_records=4
    )
    snapshot = SimpleNamespace(
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        camera_frame=SimpleNamespace(image=_image()),
    )
    assert recorder.capture_decoded_snapshot(snapshot)
    stats = recorder.close(expected_decoded_frames=2)
    assert not stats.complete
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        ReplayBundleReader(bundle)


def test_enqueue_and_close_are_linearized_and_exact_counts_are_required(monkeypatch):
    class Writer:
        def __init__(self):
            self.events = []

        def record_event(self, event, **_fields):
            self.events.append(event)

        def close(self, **_kwargs):
            return "dataset"

        def abort(self, _reason):
            raise AssertionError("a lossless close must not abort")

    writer = Writer()
    recorder = AsyncReplayRecorder(writer, max_queue_records=2)
    entered_put = threading.Event()
    release_put = threading.Event()
    original_put = recorder._queue.put_nowait

    def delayed_put(item):
        entered_put.set()
        assert release_put.wait(1.0)
        original_put(item)

    monkeypatch.setattr(recorder._queue, "put_nowait", delayed_put)
    enqueue_thread = threading.Thread(
        target=lambda: recorder.record_event("before-close")
    )
    enqueue_thread.start()
    assert entered_put.wait(1.0)
    close_result = []
    close_thread = threading.Thread(target=lambda: close_result.append(recorder.close()))
    close_thread.start()
    release_put.set()
    enqueue_thread.join(1.0)
    close_thread.join(1.0)
    assert not enqueue_thread.is_alive()
    assert not close_thread.is_alive()
    stats = close_result[0]
    assert stats.complete
    assert stats.enqueued == stats.written == 1
    assert writer.events == ["before-close"]


def test_finalization_timeout_permanently_invalidates_late_complete_manifest(tmp_path, monkeypatch):
    bundle = tmp_path / "slow-finalize.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    original_close = writer.close
    entered = threading.Event()
    release = threading.Event()

    def delayed_close(**kwargs):
        entered.set()
        assert release.wait(2.0)
        return original_close(**kwargs)

    monkeypatch.setattr(writer, "close", delayed_close)
    recorder = AsyncReplayRecorder(writer)
    close_result = []

    def close_recorder():
        close_result.append(recorder.close(timeout_s=0.05))

    thread = threading.Thread(target=close_recorder)
    thread.start()
    assert entered.wait(1.0)
    thread.join(0.5)
    assert not thread.is_alive()
    assert close_result[0].complete is False
    with pytest.raises(ValueError, match="invalidated"):
        ReplayBundleReader(bundle, require_complete=False)
    release.set()
    deadline = time.monotonic() + 2.0
    while recorder._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not recorder._thread.is_alive()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False


def test_command_scoring_rejects_bool_string_and_unknown_field_coercion():
    records = []
    for sequence, command in enumerate(
        (
            {"roll_rate": "0.1", "pitch_rate": 0, "yaw_rate": 0, "thrust": 0.2},
            {"roll_rate": False, "pitch_rate": 0, "yaw_rate": 0, "thrust": 0.2},
            {"roll_rate": 0.1, "pitch_rate": 0, "yaw_rate": 0, "thrust": 0.2, "extra": 1},
        )
    ):
        records.append(
            {
                "type": "command",
                "sequence": sequence,
                "kind": "generated",
                "command": command,
            }
        )
    score = score_records(records, session_id="s")
    assert score["open_loop_commands"]["recorded_stream"]["invalid_count"] == 3


def test_all_zero_command_is_syntactically_valid_but_explicitly_counted():
    zero = {"roll_rate": 0.0, "pitch_rate": 0.0, "yaw_rate": 0.0, "thrust": 0.0}
    score = score_records(
        [{"type": "command", "sequence": 0, "kind": "sent", "command": zero}],
        session_id="s",
    )
    commands = score["open_loop_commands"]["recorded_stream"]
    assert commands["invalid_count"] == 0
    assert commands["zero_command_count"] == 1


def test_grouped_split_never_splits_frames_within_one_session():
    split = grouped_session_split([("flight-a", "a" * 64), ("flight-b", "b" * 64)])
    assert set(split) == {"flight-a", "flight-b"}
    assert set(split.values()) == {"train", "validation"}


@pytest.mark.parametrize(
    "sessions,match",
    [
        ([("", "a" * 64)], "session id"),
        ([(1, "a" * 64)], "session id"),
        ([("flight", "A" * 64)], "dataset hash"),
        ([("flight", "short")], "dataset hash"),
        ([("flight", "a" * 64), ("flight", "b" * 64)], "duplicate session"),
    ],
)
def test_grouped_split_rejects_ambiguous_identity(sessions, match):
    with pytest.raises(ValueError, match=match):
        grouped_session_split(sessions)


@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": float("nan")},
        {"bad": object()},
        {"bad": np.asarray([float("inf")])},
    ],
)
def test_writer_rejects_noncanonical_metadata_before_creating_bundle(
    tmp_path, metadata
):
    bundle = tmp_path / "invalid-metadata.vq2replay"
    with pytest.raises((TypeError, ValueError)):
        ReplayBundleWriter(bundle, metadata=metadata, require_private=False)
    assert not bundle.exists()


def test_writer_rejects_oversized_initial_manifest_without_leaving_bundle(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module

    bundle = tmp_path / "oversized-initial-manifest.vq2replay"
    monkeypatch.setattr(replay_module, "MAX_REPLAY_MANIFEST_BYTES", 1024)
    monkeypatch.setattr(replay_module, "_MANIFEST_FINALIZATION_RESERVE_BYTES", 128)
    with pytest.raises(ValueError, match="resource limit"):
        ReplayBundleWriter(
            bundle,
            metadata={"oversized": "x" * 4096},
            require_private=False,
        )
    assert not bundle.exists()


def test_writer_rolls_back_bundle_leaf_when_records_open_fails(
    tmp_path, monkeypatch
):
    bundle = tmp_path / "records-open-failure.vq2replay"
    real_open = Path.open

    def fail_records_open(path, *args, **kwargs):
        if path == bundle / "records.jsonl":
            raise OSError("injected records open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_records_open)
    with pytest.raises(OSError, match="injected records open failure"):
        ReplayBundleWriter(bundle, require_private=False)
    assert not bundle.exists()


@pytest.mark.parametrize("failure_point", ["flush", "hash"])
def test_writer_precommit_failure_closes_and_publishes_incomplete(
    tmp_path, monkeypatch, failure_point
):
    import aigp_loop.replay as replay_module

    bundle = tmp_path / f"precommit-{failure_point}.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    records_handle = writer._records

    def injected_failure(*_args, **_kwargs):
        raise OSError(f"injected {failure_point} failure")

    if failure_point == "flush":
        monkeypatch.setattr(writer, "flush", injected_failure)
    else:
        monkeypatch.setattr(replay_module, "sha256_file", injected_failure)
    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        writer.close()

    assert writer.closed is True
    assert records_handle.closed is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert f"injected {failure_point} failure" in manifest["abort_reason"]


def test_writer_oversized_final_manifest_remains_durably_incomplete(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module

    bundle = tmp_path / "oversized-final-manifest.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    monkeypatch.setattr(replay_module, "MAX_REPLAY_MANIFEST_BYTES", 2048)
    with pytest.raises(ValueError, match="manifest exceeds replay resource limit"):
        writer.close(outcome={"oversized": "x" * 16_384})

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert "manifest exceeds replay resource limit" in manifest["abort_reason"]
    with pytest.raises(ValueError, match="incomplete"):
        ReplayBundleReader(bundle)


def test_caught_serialization_failure_can_never_seal_complete_bundle(tmp_path):
    bundle = tmp_path / "latched-write-failure.vq2replay"
    writer = ReplayBundleWriter(bundle, require_private=False)
    with pytest.raises(TypeError, match="unsupported JSON evidence"):
        writer.record_event("bad", value=object())
    with pytest.raises(RuntimeError, match="cannot seal"):
        writer.close()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False


@pytest.mark.parametrize("outcome", [{"bad": float("nan")}, {"bad": object()}])
def test_async_close_invalid_outcome_is_latched_before_worker_can_seal(
    tmp_path, outcome
):
    bundle = tmp_path / "invalid-outcome.vq2replay"
    recorder = AsyncReplayRecorder(
        ReplayBundleWriter(bundle, require_private=False)
    )
    stats = recorder.close(outcome=outcome)
    assert stats.complete is False
    assert "invalid close outcome" in stats.failure_reason
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False


@pytest.mark.parametrize("maximum", [True, False, float("nan"), 0.0])
def test_scoring_rejects_coerced_or_invalid_matching_threshold(maximum, tmp_path):
    with pytest.raises(ValueError, match="max_center_error_px"):
        score_records([], session_id="session", max_center_error_px=maximum)
    with pytest.raises(ValueError, match="max_center_error_px"):
        score_corpus(
            tmp_path / "does-not-exist.json",
            processor_spec="candidate:run",
            max_center_error_px=maximum,
        )


@pytest.mark.parametrize("session_id", ["", "   ", 1, True])
def test_scoring_requires_exact_nonempty_session_identity(session_id):
    with pytest.raises(ValueError, match="session_id"):
        score_records([], session_id=session_id)


def test_boolean_latency_is_not_counted_as_numeric_evidence():
    score = score_records(
        [
            {
                "type": "frame",
                "sequence": 0,
                "generation": 0,
                "frame_id": 1,
                "sim_time_ns": 1,
                "detector_latency_ms": True,
                "full_stack_latency_ms": False,
                "detections": [],
            }
        ],
        session_id="session",
    )
    assert score["perception"]["detector_latency_ms_p50"] is None
    assert score["perception"]["full_stack_latency_ms_p50"] is None


def test_process_frames_consumes_source_records_once_not_once_per_frame(tmp_path):
    bundle = tmp_path / "linear-replay.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="linear", require_private=False)
    for frame_id in range(40):
        writer.record_event("tick", frame_id=frame_id)
        writer.capture_decoded_frame(
            _image(frame_id),
            generation=0,
            frame_id=frame_id,
            sim_time_ns=frame_id,
            received_monotonic_s=float(frame_id),
        )
    writer.close()
    reader = ReplayBundleReader(bundle)

    class CountingRecords:
        def __init__(self, rows):
            self.rows = rows
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.rows)

    source = CountingRecords(list(reader.records()))
    output = process_frames(
        reader,
        source,
        f"{__name__}:_matching_processor",
        processor_callable=_matching_processor,
    )
    assert sum(row.get("type") == "frame" for row in output) == 40
    assert source.iterations == 1


def test_delayed_processed_record_never_reorders_decoded_candidate_feed(tmp_path):
    bundle = tmp_path / "decoded-order.vq2replay"
    writer = ReplayBundleWriter(bundle, session_id="ordered", require_private=False)
    writer.capture_decoded_frame(
        _image(1), generation=0, frame_id=1, sim_time_ns=1, received_monotonic_s=1.0
    )
    writer.capture_decoded_frame(
        _image(2), generation=0, frame_id=2, sim_time_ns=2, received_monotonic_s=2.0
    )
    writer.capture_frame(
        _image(1),
        generation=0,
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        detector_latency_ms=1.0,
        detections=[],
        tracker=None,
        imu=None,
        estimator=None,
        race_status=None,
        generated_command=None,
        sent_command=None,
    )
    writer.close()
    reader = ReplayBundleReader(bundle)
    _summary, records = reader.verify_and_read()
    seen = []

    def observe(image, context):
        seen.append(
            (
                context["frame_id"],
                context["decoded_sequence"],
                "evaluation_sequence" in context,
            )
        )
        return _matching_processor(image, context)

    result = process_frames(
        reader,
        records,
        f"{__name__}:_matching_processor",
        processor_callable=observe,
    )
    assert seen == [(1, 0, False), (2, 1, False)]
    assert [row["frame_id"] for row in result if row["type"] == "frame"] == [1, 2]


def test_isolated_transport_drains_stderr_and_times_out_hung_response(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_file

    wrapper = tmp_path / "wrapper.exe"
    wrapper.write_bytes(b"reviewed-wrapper")
    stopped = threading.Event()
    stderr_drained = threading.Event()

    class FakeInput:
        closed = False

        def write(self, value):
            return len(value)

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class BlockingOutput:
        def readline(self, _limit=-1):
            stopped.wait(1.0)
            return ""

    class FloodingError:
        def __init__(self):
            self.remaining = 2 * 1024 * 1024

        def read(self, size):
            if self.remaining:
                count = min(size, self.remaining)
                self.remaining -= count
                if self.remaining == 0:
                    stderr_drained.set()
                return "x" * count
            stopped.wait(1.0)
            return ""

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.stdin = FakeInput()
            self.stdout = BlockingOutput()
            self.stderr = FloodingError()
            self.returncode = None

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9
            stopped.set()

        def wait(self, timeout=None):
            if not stopped.wait(timeout):
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

    fake = FakeProcess()
    _patch_wrapper_attestation(
        monkeypatch,
        replay_module,
        wrapper,
        {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
        },
    )
    monkeypatch.setattr(
        replay_module, "_candidate_worktree_root", lambda _value: Path.cwd().resolve()
    )
    monkeypatch.setattr(replay_module.subprocess, "Popen", lambda *_a, **_k: fake)
    processor = IsolatedReplayProcessor(
        f"{__name__}:_matching_processor",
        wrapper,
        sha256_file(wrapper),
        response_timeout_s=0.03,
    )
    monkeypatch.setattr(processor, "_terminate_worker_tree", fake.kill)
    assert stderr_drained.wait(1.0)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        processor(_image(), {})
    assert time.monotonic() - started < 0.5
    assert len(processor._stderr_snapshot()) == 32_768
    processor.close()


def test_isolated_transport_deadline_includes_blocked_request_write(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_file

    wrapper = tmp_path / "wrapper.exe"
    wrapper.write_bytes(b"reviewed-wrapper")
    stopped = threading.Event()
    write_started = threading.Event()

    class BlockingInput:
        closed = False

        def write(self, _value):
            write_started.set()
            stopped.wait(2.0)
            return 0

        def flush(self):
            return None

        def close(self):
            self.closed = True
            stopped.set()

    class BlockingOutput:
        def readline(self, _limit=-1):
            stopped.wait(2.0)
            return ""

    class EmptyError:
        def read(self, _size):
            return ""

    class FakeProcess:
        pid = 12346

        def __init__(self):
            self.stdin = BlockingInput()
            self.stdout = BlockingOutput()
            self.stderr = EmptyError()
            self.returncode = None

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9
            stopped.set()

        def wait(self, timeout=None):
            if not stopped.wait(timeout):
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

    fake = FakeProcess()
    _patch_wrapper_attestation(
        monkeypatch,
        replay_module,
        wrapper,
        {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
        },
    )
    monkeypatch.setattr(
        replay_module, "_candidate_worktree_root", lambda _value: Path.cwd().resolve()
    )
    monkeypatch.setattr(replay_module.subprocess, "Popen", lambda *_a, **_k: fake)
    processor = IsolatedReplayProcessor(
        f"{__name__}:_matching_processor",
        wrapper,
        sha256_file(wrapper),
        response_timeout_s=0.03,
    )
    monkeypatch.setattr(processor, "_terminate_worker_tree", fake.kill)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="request deadline exceeded"):
        processor(np.zeros((720, 1280, 3), dtype=np.uint8), {})
    assert write_started.is_set()
    assert time.monotonic() - started < 0.5
    assert fake.returncode == -9
    processor.close()


def test_isolation_wrapper_that_can_exit_before_descendants_is_rejected(
    tmp_path, monkeypatch
):
    import aigp_loop.replay as replay_module
    from aigp_loop._util import sha256_file

    wrapper = tmp_path / "parent-only-wrapper.exe"
    wrapper.write_bytes(b"parent-can-exit")
    _patch_wrapper_attestation(
        monkeypatch,
        replay_module,
        wrapper,
        {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
        },
    )
    monkeypatch.setattr(
        replay_module, "_candidate_worktree_root", lambda _value: Path.cwd().resolve()
    )
    monkeypatch.setattr(
        replay_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("uncontained wrapper must never launch")
        ),
    )
    with pytest.raises(RuntimeError, match="did not attest"):
        IsolatedReplayProcessor(
            f"{__name__}:_matching_processor", wrapper, sha256_file(wrapper)
        )
