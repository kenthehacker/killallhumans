"""Verify and summarize one capture-loaded passive VQ2 timing bundle.

This entry point is deliberately offline.  It opens no simulator process or
socket and has no reset, arm, command, or transport authority.  Successful
output contains aggregate counts and timing only; decoded pixels and raw
sensor values are verified by the replay reader but are never returned.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigp_loop.replay import ReplayBundleReader
from competition.vq2_capture import (
    HIGHRES_IMU_MESSAGE_TYPE,
    HOST_PERF_COUNTER_CLOCK_ID,
    MavlinkIngressV1,
    ReceivedIMUSampleV1,
    SUPPORTED_MAVLINK_MESSAGE_TYPES,
)
from competition.vq2_contracts import FrameTimingV1
from competition.vq2_passive_timing import (
    CameraFrameTimingObservationV1,
    summarize_camera_frame_timing_observations,
)


ANALYSIS_SCHEMA = "aigp-vq2-passive-timing-analysis/1"
TRAINING_MODE_BASIS = "operator-attested-2026-07-20"
CAPTURE_LOAD_LABEL = "replay-capture-loaded"
CAMERA_STREAM_ID = "vq2-camera-udp-5600"
MAVLINK_STREAM_ID = "vq2-mavlink-udp-14550"

_RECORD_ENVELOPE_FIELDS = frozenset(
    {"schema", "session_id", "sequence", "type", "capture_wall_time_ns"}
)
_CAMERA_TIMING_EVENT_FIELDS = frozenset(
    {"event", "observation", "linked_decoded_frame_record_sequence"}
)
_CAMERA_OBSERVATION_EVENT_FIELDS = frozenset({"event", "observation"})
_RECEIVED_IMU_EVENT_FIELDS = frozenset(
    {"event", "observation", "linked_imu_record_sequence"}
)
_MAVLINK_INGRESS_EVENT_FIELDS = frozenset({"event", "observation"})

_TIMING_EVIDENCE_SCHEMAS = (
    MavlinkIngressV1.SCHEMA,
    ReceivedIMUSampleV1.SCHEMA,
    CameraFrameTimingObservationV1.SCHEMA,
)

_INGRESS_STATS_FIELDS = frozenset(
    {
        "generation",
        "next_sequence",
        "highres_imu_received",
        "heartbeat_received",
        "race_status_received",
        "actuator_received",
        "dropped",
        "high_watermark",
        "imu_capacity",
        "other_capacity",
        "imu_dropped",
        "other_dropped",
        "imu_high_watermark",
        "other_high_watermark",
        "buffered_imu",
        "buffered_other",
    }
)
_INGRESS_STAT_COUNT_BY_MESSAGE_TYPE = {
    "HIGHRES_IMU": "highres_imu_received",
    "HEARTBEAT": "heartbeat_received",
    "RACE_STATUS": "race_status_received",
    "ACTUATOR_OUTPUT_STATUS": "actuator_received",
}

_OUTBOUND_AUDIT_FIELDS = frozenset(
    {
        "timesync",
        "gcs_heartbeat",
        "sim_reset",
        "arm",
        "disarm",
        "attitude_target",
        "position_target",
        "other_command",
        "disallowed_count",
    }
)
_DISALLOWED_OUTBOUND_FIELDS = (
    "sim_reset",
    "arm",
    "disarm",
    "attitude_target",
    "position_target",
    "other_command",
)

_VISION_INTEGER_DIAGNOSTICS = (
    "datagrams_received",
    "unique_datagrams",
    "duplicate_datagrams",
    "malformed_datagrams",
    "frames_reassembled",
    "frames_decoded",
    "decode_failures",
    "out_of_order_frame_drops",
    "reset_generation_drops",
    "processing_errors",
    "socket_errors",
    "snapshot_callback_errors",
    "resets",
    "remembered_chunk_keys",
    "timing_ledger_entries",
    "timing_ledger_high_watermark",
    "timing_ledger_capacity",
    "receiver_buffered_partial_frames",
    "receiver_buffer_high_watermark",
    "receiver_buffer_capacity",
    "capture_snapshot_queue_entries",
    "capture_snapshot_queue_high_watermark",
    "capture_snapshot_queue_capacity",
    "capture_snapshot_queue_dropped",
    "receiver_dropped_partial_frames",
    "receiver_duplicate_chunks",
    "receiver_dropped_late_packets",
)
_VISION_DIAGNOSTIC_FIELDS = frozenset(_VISION_INTEGER_DIAGNOSTICS) | frozenset(
    {"timing_overflow_latched", "capture_snapshot_queue_enabled"}
)
_VISION_CAPTURE_FAILURE_COUNTERS = (
    "malformed_datagrams",
    "decode_failures",
    "out_of_order_frame_drops",
    "reset_generation_drops",
    "processing_errors",
    "socket_errors",
    "snapshot_callback_errors",
    "capture_snapshot_queue_dropped",
    "receiver_dropped_partial_frames",
)

_ADDITIONAL_UNMEASURED_ITEMS = (
    "simulator_wall_ratio",
    "training_mode_machine_detection",
    "no_capture_production_latency",
    "host_process_load_context",
    "graphics_focus_context",
    "post_probe_port_release",
)


def _exact_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact object")
    return value


def _exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> dict[str, Any]:
    row = _exact_object(value, label)
    actual = set(row)
    if actual != expected:
        raise ValueError(
            f"{label} fields must be exact; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return row


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _event_row(
    record: Mapping[str, Any],
    *,
    event: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    row = _exact_fields(
        record,
        _RECORD_ENVELOPE_FIELDS | fields,
        f"{event} replay event",
    )
    if row["type"] != "event" or row["event"] != event:
        raise ValueError(f"invalid {event} replay event identity")
    return row


def _frame_token(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        _exact_nonnegative_int(record.get("generation"), "frame generation"),
        _exact_nonnegative_int(record.get("frame_id"), "frame id"),
        _exact_nonnegative_int(record.get("sim_time_ns"), "camera source token"),
    )


def _timing_token(timing: FrameTimingV1) -> tuple[int, int, int]:
    return (
        timing.identity.generation,
        timing.identity.frame_id,
        timing.camera_source_time_ns,
    )


def _rate_hz(points_ns: Sequence[int]) -> Optional[float]:
    if len(points_ns) < 2:
        return None
    span_ns = points_ns[-1] - points_ns[0]
    if span_ns <= 0:
        return None
    rate = (len(points_ns) - 1) * 1_000_000_000.0 / span_ns
    if not math.isfinite(rate):
        raise ValueError("derived timing rate is not finite")
    return rate


def _metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _exact_object(manifest.get("metadata"), "replay metadata")
    required = {
        "simulator_build": "3385",
        "simulator_mode": "Training",
        "simulator_mode_basis": TRAINING_MODE_BASIS,
        "stage": "preflight",
        "capture_kind": "private-development-session",
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"replay metadata {name} must be exact {expected!r}")
    _sha256(metadata.get("code_hash"), "replay metadata code_hash")
    dwell = metadata.get("preflight_healthy_dwell_s")
    if (
        type(dwell) not in {int, float}
        or not math.isfinite(dwell)
        or float(dwell) != 5.0
    ):
        raise ValueError("replay metadata preflight_healthy_dwell_s must be exact 5.0")
    schemas = metadata.get("timing_evidence_schemas")
    if type(schemas) is not list or tuple(schemas) != _TIMING_EVIDENCE_SCHEMAS:
        raise ValueError("replay metadata timing evidence schemas are incomplete")
    return metadata


def _outcome(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    outcome = _exact_object(manifest.get("outcome"), "replay outcome")
    if outcome.get("stage") != "preflight":
        raise ValueError("replay outcome must be for the preflight stage")
    if outcome.get("success") is not True:
        raise ValueError("replay outcome success must be exact true")
    if outcome.get("cleanup_confirmed") is not True:
        raise ValueError("replay outcome cleanup_confirmed must be exact true")
    if outcome.get("transport_cleanup_errors"):
        raise ValueError("replay outcome contains transport cleanup errors")
    details = _exact_object(outcome.get("details"), "replay outcome details")
    return outcome, details


def _vision_diagnostics(outcome: Mapping[str, Any]) -> dict[str, Any]:
    stats = _exact_fields(
        outcome.get("vision_capture_stats"),
        _VISION_DIAGNOSTIC_FIELDS,
        "vision_capture_stats",
    )
    result: dict[str, Any] = {}
    for name in _VISION_INTEGER_DIAGNOSTICS:
        result[name] = _exact_nonnegative_int(
            stats[name], f"vision_capture_stats.{name}"
        )
    if type(stats["timing_overflow_latched"]) is not bool:
        raise TypeError("timing_overflow_latched must be an exact bool")
    if stats["timing_overflow_latched"] is not False:
        raise ValueError("vision timing ledger overflowed")
    result["timing_overflow_latched"] = False
    if stats["capture_snapshot_queue_enabled"] is not True:
        raise ValueError("passive capture snapshot queue was not enabled")
    result["capture_snapshot_queue_enabled"] = True
    capacity = result["timing_ledger_capacity"]
    if capacity < 1:
        raise ValueError("vision timing ledger capacity must be positive")
    if (
        result["timing_ledger_entries"] > capacity
        or result["timing_ledger_high_watermark"] > capacity
    ):
        raise ValueError("vision timing ledger diagnostics exceed capacity")
    capture_capacity = result["capture_snapshot_queue_capacity"]
    if capture_capacity < 1:
        raise ValueError("vision capture snapshot queue capacity must be positive")
    if (
        result["capture_snapshot_queue_entries"] != 0
        or result["capture_snapshot_queue_high_watermark"] > capture_capacity
    ):
        raise ValueError("vision capture snapshot queue was not completely drained")
    failed = [name for name in _VISION_CAPTURE_FAILURE_COUNTERS if result[name]]
    if failed:
        raise ValueError(
            "vision capture has nonzero failure/drop counters: "
            + ", ".join(failed)
        )
    return result


def _ingress_diagnostics(
    details: Mapping[str, Any], ingress: Sequence[MavlinkIngressV1]
) -> tuple[dict[str, int], dict[str, int]]:
    stats = _exact_fields(
        details.get("mavlink_ingress_stats"),
        _INGRESS_STATS_FIELDS,
        "mavlink_ingress_stats",
    )
    checked = {
        name: _exact_nonnegative_int(value, f"mavlink_ingress_stats.{name}")
        for name, value in stats.items()
    }
    if checked["dropped"] != checked["imu_dropped"] + checked["other_dropped"]:
        raise ValueError("MAVLink ingress queue drop counters are inconsistent")
    if checked["dropped"] != 0:
        raise ValueError("MAVLink ingress queue dropped observations")
    if checked["imu_capacity"] < 1 or checked["other_capacity"] < 1:
        raise ValueError("MAVLink ingress queue capacities must be positive")
    if (
        checked["imu_high_watermark"] > checked["imu_capacity"]
        or checked["other_high_watermark"] > checked["other_capacity"]
        or checked["high_watermark"]
        > checked["imu_capacity"] + checked["other_capacity"]
    ):
        raise ValueError("MAVLink ingress queue high watermark exceeds capacity")
    if checked["high_watermark"] < max(
        checked["imu_high_watermark"], checked["other_high_watermark"]
    ):
        raise ValueError("MAVLink ingress aggregate high watermark is inconsistent")
    if checked["buffered_imu"] != 0 or checked["buffered_other"] != 0:
        raise ValueError("MAVLink ingress queues were not completely drained")
    if checked["next_sequence"] != len(ingress):
        raise ValueError("MAVLink ingress next_sequence contradicts captured evidence")
    if ingress and checked["generation"] != ingress[0].generation:
        raise ValueError("MAVLink ingress generation contradicts captured evidence")
    if checked["high_watermark"] > checked["next_sequence"]:
        raise ValueError("MAVLink ingress high watermark is impossible")

    observed_counts = {
        message_type: sum(item.message_type == message_type for item in ingress)
        for message_type in sorted(SUPPORTED_MAVLINK_MESSAGE_TYPES)
    }
    for message_type, stats_name in _INGRESS_STAT_COUNT_BY_MESSAGE_TYPE.items():
        if checked[stats_name] != observed_counts[message_type]:
            raise ValueError(
                f"MAVLink ingress {message_type} count contradicts captured evidence"
            )
    if sum(observed_counts.values()) != checked["next_sequence"]:
        raise ValueError("MAVLink ingress type counts contradict next_sequence")
    return checked, observed_counts


def _outbound_diagnostics(details: Mapping[str, Any]) -> dict[str, int]:
    audit = _exact_fields(
        details.get("mavlink_outbound_audit"),
        _OUTBOUND_AUDIT_FIELDS,
        "mavlink_outbound_audit",
    )
    checked = {
        name: _exact_nonnegative_int(value, f"mavlink_outbound_audit.{name}")
        for name, value in audit.items()
    }
    recomputed = sum(checked[name] for name in _DISALLOWED_OUTBOUND_FIELDS)
    if checked["disallowed_count"] != recomputed:
        raise ValueError("MAVLink outbound disallowed_count is inconsistent")
    if recomputed != 0:
        raise ValueError("passive replay contains disallowed outbound activity")
    return checked


def analyze_bundle(bundle: Path | str) -> dict[str, Any]:
    """Fully verify and aggregate one passive replay bundle."""

    reader = ReplayBundleReader(bundle, require_complete=True)
    verified, records = reader.verify_and_read(verify_frames=True)
    manifest = reader.manifest
    metadata = _metadata(manifest)
    outcome, details = _outcome(manifest)
    vision = _vision_diagnostics(outcome)

    decoded_records = [row for row in records if row["type"] == "decoded_frame"]
    processed_records = [row for row in records if row["type"] == "frame"]
    imu_records = {row["sequence"]: row for row in records if row["type"] == "imu"}
    command_records = [row for row in records if row["type"] == "command"]
    if command_records:
        raise ValueError("passive replay must not contain command records")
    if any(
        row["generated_command"] is not None or row["sent_command"] is not None
        for row in processed_records
    ):
        raise ValueError("passive replay must not contain embedded frame commands")

    decoded_by_sequence = {row["sequence"]: row for row in decoded_records}
    decoded_tokens = {_frame_token(row) for row in decoded_records}
    processed_tokens = {_frame_token(row) for row in processed_records}

    frame_timing_by_token: dict[tuple[int, int, int], FrameTimingV1] = {}
    observations_by_token: dict[
        tuple[int, int, int], CameraFrameTimingObservationV1
    ] = {}
    linked_imu_sequences: set[int] = set()
    ingress: list[MavlinkIngressV1] = []
    received_imu_count = 0

    for record in records:
        if record["type"] != "event":
            continue
        event = record["event"]
        if event == "camera_frame_timing":
            row = _event_row(
                record,
                event=event,
                fields=_CAMERA_TIMING_EVENT_FIELDS,
            )
            timing = FrameTimingV1.from_primitive(row["observation"])
            linked = _exact_nonnegative_int(
                row["linked_decoded_frame_record_sequence"],
                "linked decoded-frame record sequence",
            )
            decoded = decoded_by_sequence.get(linked)
            if decoded is None:
                raise ValueError("camera timing event does not link a decoded frame")
            token = _timing_token(timing)
            if token != _frame_token(decoded):
                raise ValueError("camera timing identity differs from decoded frame")
            if token in frame_timing_by_token:
                raise ValueError("duplicate camera timing event")
            frame_timing_by_token[token] = timing
        elif event == "camera_frame_timing_observation":
            row = _event_row(
                record,
                event=event,
                fields=_CAMERA_OBSERVATION_EVENT_FIELDS,
            )
            observation = CameraFrameTimingObservationV1.from_primitive(
                row["observation"]
            )
            token = _timing_token(observation.frame_timing)
            if token in observations_by_token:
                raise ValueError("duplicate camera timing observation")
            observations_by_token[token] = observation
        elif event == "received_imu":
            row = _event_row(
                record,
                event=event,
                fields=_RECEIVED_IMU_EVENT_FIELDS,
            )
            received = ReceivedIMUSampleV1.from_primitive(row["observation"])
            linked = _exact_nonnegative_int(
                row["linked_imu_record_sequence"],
                "linked IMU record sequence",
            )
            core = imu_records.get(linked)
            if core is None:
                raise ValueError("received IMU event does not link a core IMU record")
            if linked in linked_imu_sequences:
                raise ValueError("duplicate received IMU link")
            if core["imu"] != received.to_primitive()["imu"]:
                raise ValueError("received IMU event differs from its core IMU record")
            linked_imu_sequences.add(linked)
            ingress.append(received.ingress)
            received_imu_count += 1
        elif event == "mavlink_ingress":
            row = _event_row(
                record,
                event=event,
                fields=_MAVLINK_INGRESS_EVENT_FIELDS,
            )
            arrival = MavlinkIngressV1.from_primitive(row["observation"])
            if arrival.message_type == HIGHRES_IMU_MESSAGE_TYPE:
                raise ValueError(
                    "HIGHRES_IMU ingress must use the received_imu evidence event"
                )
            ingress.append(arrival)

    if linked_imu_sequences != set(imu_records):
        raise ValueError("core IMU records and exact received IMU events are not one-to-one")
    if set(frame_timing_by_token) != decoded_tokens:
        raise ValueError("decoded frames and exact camera timing events are not one-to-one")
    if set(observations_by_token) != decoded_tokens:
        raise ValueError("decoded frames and passive timing observations are not one-to-one")
    if processed_tokens != decoded_tokens:
        raise ValueError("decoded and processed frame identities are not one-to-one")
    for token, observation in observations_by_token.items():
        if observation.frame_timing != frame_timing_by_token[token]:
            raise ValueError("passive observation timing differs from receiver timing")

    observations = tuple(
        observations_by_token[_frame_token(row)] for row in processed_records
    )
    decoded_count = len(decoded_records)
    processed_count = len(processed_records)
    if decoded_count < 2:
        raise ValueError("passive timing analysis requires at least two decoded frames")
    if (
        len(frame_timing_by_token) != decoded_count
        or len(observations) != processed_count
        or processed_count != manifest["frame_record_count"]
        or decoded_count != manifest["decoded_frame_record_count"]
        or vision["frames_decoded"] != decoded_count
        or vision["frames_reassembled"] != decoded_count
        or processed_count != decoded_count
    ):
        raise ValueError("camera capture, timing, and processed-frame counts disagree")

    ordered_ingress = sorted(ingress, key=lambda item: item.sequence)
    if not ordered_ingress:
        raise ValueError("passive timing replay contains no exact MAVLink ingress")
    first_ingress = ordered_ingress[0]
    if first_ingress.sequence != 0:
        raise ValueError("MAVLink ingress sequence must begin at zero")
    if any(
        item.stream_id != MAVLINK_STREAM_ID
        or item.generation != first_ingress.generation
        or item.host_clock_id != HOST_PERF_COUNTER_CLOCK_ID
        for item in ordered_ingress
    ):
        raise ValueError("MAVLink ingress mixes stream, generation, or host clock")
    if [item.sequence for item in ordered_ingress] != list(
        range(len(ordered_ingress))
    ):
        raise ValueError("MAVLink ingress sequence has duplicates or gaps")
    if any(
        later.received_monotonic_ns < earlier.received_monotonic_ns
        for earlier, later in zip(ordered_ingress, ordered_ingress[1:])
    ):
        raise ValueError("MAVLink ingress receive time regresses")

    ingress_stats, ingress_counts = _ingress_diagnostics(details, ordered_ingress)
    outbound_audit = _outbound_diagnostics(details)

    summary = summarize_camera_frame_timing_observations(
        observations,
        receiver_frames_reassembled=vision["frames_reassembled"],
        receiver_frames_decoded=vision["frames_decoded"],
        receiver_queue_high_watermark=vision["receiver_buffer_high_watermark"],
        receiver_queue_capacity=vision["receiver_buffer_capacity"],
    )
    if (
        summary.host_clock_id != HOST_PERF_COUNTER_CLOCK_ID
        or summary.stream_id != CAMERA_STREAM_ID
    ):
        raise ValueError("camera timing uses an unexpected stream or host clock")
    distributions = {
        name: {
            "count": distribution.count,
            "p50_ns": distribution.p50_ns,
            "p95_ns": distribution.p95_ns,
            "p99_ns": distribution.p99_ns,
            "maximum_ns": distribution.maximum_ns,
        }
        for name, distribution in summary.distributions
    }

    ingress_points = [item.received_monotonic_ns for item in ordered_ingress]
    message_points = {
        message_type: [
            item.received_monotonic_ns
            for item in ordered_ingress
            if item.message_type == message_type
        ]
        for message_type in sorted(SUPPORTED_MAVLINK_MESSAGE_TYPES)
    }
    unmeasured = list(summary.unmeasured_items)
    unmeasured.extend(
        item for item in _ADDITIONAL_UNMEASURED_ITEMS if item not in unmeasured
    )

    result = {
        "schema": ANALYSIS_SCHEMA,
        "dataset_hash": _sha256(verified["dataset_hash"], "dataset_hash"),
        "code_hash": _sha256(metadata["code_hash"], "code_hash"),
        "simulator": {
            "build": "3385",
            "mode": "Training",
            "mode_basis": TRAINING_MODE_BASIS,
        },
        "capture_load": {
            "label": CAPTURE_LOAD_LABEL,
            "capture_kind": metadata["capture_kind"],
            "preflight_healthy_dwell_s": 5.0,
            "full_bundle_verified": True,
            "frame_blob_decode_verified": True,
        },
        "counts": {
            "replay_records": verified["records"],
            "unique_frame_blobs": verified["unique_frame_blobs"],
            "decoded_frames": decoded_count,
            "processed_frames": processed_count,
            "camera_frame_timing_events": len(frame_timing_by_token),
            "camera_frame_timing_observations": len(observations),
            "mavlink_ingress_total": len(ordered_ingress),
            "received_imu": received_imu_count,
            "mavlink_by_message_type": ingress_counts,
            "command_records": 0,
        },
        "rates_hz": {
            "camera_publication": _rate_hz(
                [item.frame_timing.publish_monotonic_ns for item in observations]
            ),
            "camera_consumption": _rate_hz(
                [item.consume_monotonic_ns for item in observations]
            ),
            "mavlink_ingress": _rate_hz(ingress_points),
            "mavlink_by_message_type": {
                message_type: _rate_hz(points)
                for message_type, points in message_points.items()
            },
        },
        "distributions_ns": distributions,
        "camera_integrity": {
            "host_clock_id": summary.host_clock_id,
            "stream_id": summary.stream_id,
            "generation": summary.generation,
            "publication_sequence_gap_events": (
                summary.publication_sequence_gap_events
            ),
            "missing_publications": summary.missing_publications,
            "work_over_20ms_count": summary.work_over_control_period_count,
            "receiver_decode_shortfall": summary.receiver_decode_shortfall,
            "receiver_observation_shortfall": (
                summary.receiver_observation_shortfall
            ),
        },
        "vision_receiver_diagnostics": vision,
        "mavlink_ingress_diagnostics": ingress_stats,
        "mavlink_outbound_audit": outbound_audit,
        "acceptance_checks": {
            "generic_passive_timing_valid": True,
            "capture_complete": True,
            "five_second_healthy_dwell": True,
            "camera_capture_shortfalls_zero": True,
            "camera_observations_at_least_140": len(observations) >= 140,
            "highres_imu_arrivals_at_least_600": received_imu_count >= 600,
            "disallowed_outbound_zero": True,
            "ingress_queue_drops_zero": True,
            "ingress_queue_capacity_proved": True,
        },
        "unmeasured_items": unmeasured,
    }
    # Guard the public function as well as the CLI against accidental
    # non-standard floats in future aggregate fields.
    json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify and aggregate one passive VQ2 replay timing bundle."
    )
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze_bundle(args.bundle)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
