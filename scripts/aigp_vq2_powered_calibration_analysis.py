"""Pure offline analysis for the build-3385 powered-calibration pilot.

The module verifies already-sealed evidence and computes only the descriptive
support frozen by the I0 contract.  It deliberately contains no fitting,
Jacobian, SVD, covariance, rank, calibration, simulator, or network behavior.
Filesystem writes are available only through the explicit caller-directed
create-new publication function at the bottom of the module.

The I0 clarification prints the exact command source-frame key set and states
that command observations use the plan's semantic object SHA-256, not a
newline-terminated file-byte SHA-256. Identity fields are compared byte-for-
byte with their already-reviewed inputs rather than being reinterpreted here.
"""

from __future__ import annotations

import hashlib
import math
import ntpath
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from competition.vq2_contracts import FrameTimingV1
from competition.vq2_passive_timing import CameraFrameTimingObservationV1
from scripts import aigp_vq2_powered_attempt as attempt_contract
from scripts.aigp_live_lease import (
    load_powered_live_lease_record,
    validate_powered_live_lease_ledger,
    validate_powered_live_lease_record,
)


class PoweredCalibrationAnalysisError(ValueError):
    """Raised when offline evidence cannot satisfy the frozen analysis contract."""


class PartialPublicationError(PoweredCalibrationAnalysisError):
    """A create-new publication failed after at least one immutable write."""

    def __init__(self, message: str, *, published: Sequence[str]) -> None:
        super().__init__(message)
        self.published = tuple(published)


SEMANTIC_AMBIGUITIES: tuple[str, ...] = ()

REPLAY_BUNDLE_SCHEMA = "aigp-vq2-replay/1"
REPLAY_RECORD_SCHEMA = "aigp-vq2-replay-record/1"
RECORDING_NOTICE = (
    "Private competition-development artifact. Do not publish, commit, or "
    "broadcast without organizer approval."
)

_SHA256 = frozenset("0123456789abcdef")
_BASE_RECORD_KEYS = {
    "schema",
    "session_id",
    "sequence",
    "type",
    "capture_wall_time_ns",
}
_CORE_KEYS = {
    "imu": {"received_monotonic_s", "imu", "estimator"},
    "race_status": {"received_monotonic_s", "race_status"},
    "command": {"kind", "monotonic_s", "frame_token", "command"},
    "decoded_frame": {
        "generation",
        "frame_id",
        "sim_time_ns",
        "received_monotonic_s",
        "frame_blob",
        "frame_hash",
        "image_shape",
        "image_dtype",
    },
    "frame": {
        "generation",
        "frame_id",
        "sim_time_ns",
        "received_monotonic_s",
        "frame_blob",
        "frame_hash",
        "image_shape",
        "image_dtype",
        "detector_latency_ms",
        "detections",
        "tracker",
        "imu",
        "estimator",
        "race_status",
        "generated_command",
        "sent_command",
        "phase",
    },
}
_EVENT_SCHEMA = {
    "received_heartbeat": "aigp-vq2-received-heartbeat/1",
    "received_race_status": "aigp-vq2-received-race-status/1",
    "received_actuator_output_status": "aigp-vq2-received-actuator-output-status/1",
    "received_imu": "aigp-vq2-received-imu/1",
    "runner_collision_observation": "aigp-vq2-runner-collision-observation/1",
    "decoded_dimensions_admission": "aigp-vq2-decoded-dimensions-admission/1",
    "attitude_target_outbound": "aigp-vq2-attitude-target-outbound/1",
    "nonattitude_outbound": "aigp-vq2-nonattitude-outbound/1",
    "calibration_phase_deadline": "aigp-vq2-phase-deadline/1",
    "calibration_command_generated": "aigp-vq2-calibration-command-generated/1",
    "calibration_command_sent": "aigp-vq2-calibration-command-sent/1",
    "calibration_command_not_sent": "aigp-vq2-calibration-command-not-sent/1",
    "calibration_tick_disposition": "aigp-vq2-calibration-tick-disposition/1",
    "calibration_reset_boundary": "aigp-vq2-calibration-reset-boundary/1",
}
_SEQUENCED_OBSERVATION_EVENTS = frozenset(
    {
        "calibration_phase_deadline",
        "calibration_command_generated",
        "calibration_command_sent",
        "calibration_command_not_sent",
        "calibration_tick_disposition",
    }
)
_RESERVED_EVENT_FIELDS = frozenset(
    {
        "record_type",
        "record_schema",
        "dataset_hash",
        "integrity",
        "manifest",
        "frame_blob",
        "frame_hash",
        "image_shape",
        "image_dtype",
    }
)


def _fail(path: str, message: str) -> None:
    raise PoweredCalibrationAnalysisError(f"{path}: {message}")


def _object(value: Any, keys: Iterable[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an exact object")
    expected = frozenset(keys)
    actual = frozenset(value)
    if actual != expected:
        _fail(
            path,
            f"fields must be exact; missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}",
        )
    return value


def _array(value: Any, path: str, *, length: int | None = None) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be an exact array")
    if length is not None and len(value) != length:
        _fail(path, f"must contain exactly {length} items")
    return value


def _int(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        _fail(path, "must be an exact integer")
    if value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be an exact boolean")
    return value


def _number(value: Any, path: str) -> float | int:
    if type(value) not in {int, float} or not math.isfinite(value):
        _fail(path, "must be a finite JSON number and not bool")
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        _fail(path, "must be a nonempty exact string")
    return value


def _hash(value: Any, path: str) -> str:
    item = _string(value, path)
    if len(item) != 64 or any(character not in _SHA256 for character in item):
        _fail(path, "must be a canonical lowercase SHA-256")
    return item


def _clone(value: Any) -> Any:
    return attempt_contract.defensive_copy(value)


def _canonical_key(value: Any) -> bytes:
    return attempt_contract.canonical_json_bytes(value)


_RESOURCE_RECORDER_FIELDS = (
    "enqueued",
    "written",
    "dropped",
    "duplicate_frame_tokens",
    "writer_errors",
    "queue_high_watermark",
    "decoded_frames_enqueued",
    "decoded_frames_written",
    "decoded_frames_dropped",
)
_RESOURCE_VISION_FIELDS = (
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
_RESOURCE_INGRESS_FIELDS = (
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
)
_RESOURCE_COLLISION_FIELDS = (
    "generation",
    "handled",
    "dropped",
    "high_watermark",
    "capacity",
    "buffered",
)
_RESOURCE_OUTBOUND_FIELDS = (
    "generation",
    "next_sequence",
    "returned",
    "raised",
    "dropped",
    "high_watermark",
    "capacity",
    "buffered",
)

_NOMINAL_POWERED_CHILD_PHASES = (
    "connect",
    "preflight",
    "reset_epoch",
    "normalize_disarmed",
    "countdown_go",
    "arm",
    "powered_stage",
    "cleanup",
    "replay_close",
)


def _resource_counter_group(
    value: Any,
    fields: Sequence[str],
    path: str,
    *,
    boolean_fields: Sequence[str] = (),
) -> dict[str, Any]:
    booleans = frozenset(boolean_fields)
    row = _object(value, {"constructed", *fields, *booleans}, path)
    _bool(row["constructed"], f"{path}.constructed")
    for name in fields:
        _int(row[name], f"{path}.{name}")
    for name in booleans:
        _bool(row[name], f"{path}.{name}")
    return row


def validate_powered_capture_resource_stats(
    value: Any,
    path: str = "$manifest.outcome.powered_capture_resource_stats",
) -> dict[str, Any]:
    """Validate the raw child snapshot sealed immediately before replay close.

    This object preserves counters rather than a producer-authored success
    boolean.  Packet duplication is deliberately retained as a diagnostic and
    is not treated as loss; build 3385 is known to duplicate camera chunks.
    """

    row = _object(
        value,
        {
            "schema",
            "recorder",
            "vision",
            "ingress",
            "collision",
            "outbound_receipts",
            "snapshot_capture",
        },
        path,
    )
    if row["schema"] != "aigp-vq2-powered-capture-resource-stats/1":
        _fail(f"{path}.schema", "unsupported powered resource-stats schema")
    recorder = _resource_counter_group(
        row["recorder"],
        _RESOURCE_RECORDER_FIELDS,
        f"{path}.recorder",
        boolean_fields=("complete", "failure_latched"),
    )
    vision = _resource_counter_group(
        row["vision"],
        _RESOURCE_VISION_FIELDS,
        f"{path}.vision",
        boolean_fields=(
            "timing_overflow_latched",
            "capture_snapshot_queue_enabled",
        ),
    )
    ingress = _resource_counter_group(
        row["ingress"], _RESOURCE_INGRESS_FIELDS, f"{path}.ingress"
    )
    collision = _resource_counter_group(
        row["collision"], _RESOURCE_COLLISION_FIELDS, f"{path}.collision"
    )
    outbound = _resource_counter_group(
        row["outbound_receipts"],
        _RESOURCE_OUTBOUND_FIELDS,
        f"{path}.outbound_receipts",
    )
    snapshot = _object(
        row["snapshot_capture"],
        {
            "constructed",
            "observed_frames",
            "dimensions_admitted",
            "failure_latched",
        },
        f"{path}.snapshot_capture",
    )
    _bool(snapshot["constructed"], f"{path}.snapshot_capture.constructed")
    _int(snapshot["observed_frames"], f"{path}.snapshot_capture.observed_frames")
    _bool(
        snapshot["dimensions_admitted"],
        f"{path}.snapshot_capture.dimensions_admitted",
    )
    _bool(snapshot["failure_latched"], f"{path}.snapshot_capture.failure_latched")
    return _clone(
        {
            "schema": row["schema"],
            "recorder": recorder,
            "vision": vision,
            "ingress": ingress,
            "collision": collision,
            "outbound_receipts": outbound,
            "snapshot_capture": snapshot,
        }
    )


def derive_powered_capture_resource_evidence(value: Any) -> dict[str, Any]:
    """Derive frozen capture-loss counters and the zero-resource predicate."""

    row = validate_powered_capture_resource_stats(value)
    recorder = row["recorder"]
    vision = row["vision"]
    ingress = row["ingress"]
    collision = row["collision"]
    outbound = row["outbound_receipts"]
    snapshot = row["snapshot_capture"]

    capture_drops = (
        recorder["dropped"]
        + recorder["duplicate_frame_tokens"]
        + vision["malformed_datagrams"]
        + vision["processing_errors"]
        + vision["socket_errors"]
        + vision["snapshot_callback_errors"]
        + vision["capture_snapshot_queue_dropped"]
    )
    decoded_frame_drops = (
        recorder["decoded_frames_dropped"]
        + vision["decode_failures"]
        + vision["out_of_order_frame_drops"]
        + vision["reset_generation_drops"]
        + vision["receiver_dropped_partial_frames"]
        + vision["receiver_dropped_late_packets"]
    )
    observation_queue_drops = ingress["imu_dropped"] + ingress["other_dropped"]
    queue_overflows = (
        recorder["dropped"]
        + vision["capture_snapshot_queue_dropped"]
        + int(vision["timing_overflow_latched"])
        + ingress["dropped"]
        + collision["dropped"]
        + outbound["dropped"]
    )
    loss = {
        "capture_drops": capture_drops,
        "decoded_frame_drops": decoded_frame_drops,
        "writer_queue_drops": recorder["dropped"],
        "writer_errors": recorder["writer_errors"],
        "ingress_drops": ingress["dropped"],
        "observation_queue_drops": observation_queue_drops,
        "collision_queue_drops": collision["dropped"],
        "outbound_trace_drops": outbound["dropped"],
        "queue_overflows": queue_overflows,
    }
    constructed = all(
        group["constructed"] is True
        for group in (recorder, vision, ingress, collision, outbound, snapshot)
    )
    terminal_buffers_empty = (
        vision["receiver_buffered_partial_frames"] == 0
        and vision["capture_snapshot_queue_entries"] == 0
        and ingress["buffered_imu"] == 0
        and ingress["buffered_other"] == 0
        and collision["buffered"] == 0
        and outbound["buffered"] == 0
    )
    producer_state_valid = (
        recorder["complete"] is False
        and recorder["failure_latched"] is False
        and snapshot["observed_frames"] > 0
        and snapshot["dimensions_admitted"] is True
        and snapshot["failure_latched"] is False
        and vision["capture_snapshot_queue_enabled"] is True
        and outbound["raised"] == 0
    )
    return {
        "raw": row,
        "loss": loss,
        "constructed": constructed,
        "terminal_buffers_empty": terminal_buffers_empty,
        "producer_state_valid": producer_state_valid,
        "resource_stats_zero": (
            constructed
            and terminal_buffers_empty
            and producer_state_valid
            and all(value == 0 for value in loss.values())
        ),
    }


def _required_reset_boundary_occurrences_exact(
    records: Sequence[Mapping[str, Any]],
    *,
    boundaries: Sequence[tuple[int, Mapping[str, Any]]],
    final_resource_stats: Mapping[str, Any],
) -> bool:
    """Bind the two nominal reset transitions to phase and command lineage."""

    if len(boundaries) != 2:
        return False
    (prepower_index, prepower), (cleanup_index, cleanup) = boundaries
    if [
        (prepower["old_generation"], prepower["new_generation"]),
        (cleanup["old_generation"], cleanup["new_generation"]),
    ] != [(0, 1), (1, 2)]:
        return False

    phase_rows = [
        (record_index, row["observation"])
        for record_index, row in enumerate(records)
        if row.get("type") == "event"
        and row.get("event") == "calibration_phase_deadline"
    ]
    if [row["phase"] for _, row in phase_rows] != list(
        _NOMINAL_POWERED_CHILD_PHASES
    ):
        return False
    phase_by_name = {row["phase"]: (record_index, row) for record_index, row in phase_rows}
    reset_phase_index, reset_phase = phase_by_name["reset_epoch"]
    normalize_index, _ = phase_by_name["normalize_disarmed"]
    powered_index, _ = phase_by_name["powered_stage"]
    cleanup_phase_index, cleanup_phase = phase_by_name["cleanup"]
    replay_close_index, _ = phase_by_name["replay_close"]
    if not (
        reset_phase_index < prepower_index < normalize_index
        and powered_index < cleanup_phase_index < cleanup_index < replay_close_index
    ):
        return False
    if not (
        reset_phase["started_monotonic_ns"]
        <= prepower["boundary_monotonic_ns"]
        < reset_phase["deadline_monotonic_ns"]
        and cleanup_phase["started_monotonic_ns"]
        <= cleanup["boundary_monotonic_ns"]
        < cleanup_phase["deadline_monotonic_ns"]
    ):
        return False

    generated_rows = [
        (record_index, row["observation"])
        for record_index, row in enumerate(records)
        if row.get("type") == "event"
        and row.get("event") == "calibration_command_generated"
    ]
    excitation = [
        (record_index, row)
        for record_index, row in generated_rows
        if row["scope"] == "excitation"
    ]
    cleanup_generated = [
        (record_index, row)
        for record_index, row in generated_rows
        if row["scope"] == "cleanup_zero"
    ]
    if len(excitation) != 245 or len(cleanup_generated) != 1:
        return False
    excitation_generation = prepower["new_generation"]
    if not all(
        prepower_index < record_index < cleanup_phase_index
        and row["reset_epoch"] is not None
        and row["reset_epoch"]["ingress_generation"] == excitation_generation
        for record_index, row in excitation
    ):
        return False
    cleanup_generated_index, cleanup_generated_row = cleanup_generated[0]
    if not cleanup_phase_index < cleanup_generated_index < cleanup_index:
        return False

    cleanup_terminal_indices = [
        record_index
        for record_index, row in enumerate(records)
        if row.get("type") == "event"
        and row.get("event")
        in {"calibration_command_sent", "calibration_command_not_sent"}
        and row["observation"]["generated_event_sequence"]
        == cleanup_generated_row["event_sequence"]
    ]
    if len(cleanup_terminal_indices) != 1 or not (
        cleanup_generated_index < cleanup_terminal_indices[0] < cleanup_index
    ):
        return False

    return (
        cleanup["old_generation"] == excitation_generation
        and final_resource_stats["ingress"]["generation"]
        == cleanup["new_generation"]
        and final_resource_stats["collision"]["generation"]
        == cleanup["new_generation"]
    )


def _reset_boundary_resource_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    final_resource_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Conserve ingress/collision occurrences across every reset generation."""

    boundary_rows = [
        (record_index, row["observation"])
        for record_index, row in enumerate(records)
        if row.get("type") == "event"
        and row.get("event") == "calibration_reset_boundary"
    ]
    checked_boundary_rows = [
        (
            record_index,
            attempt_contract.validate_reset_boundary(
                value, f"$reset_boundaries[{index}]"
            ),
        )
        for index, (record_index, value) in enumerate(boundary_rows)
    ]
    received_event_names = {
        "received_heartbeat",
        "received_race_status",
        "received_actuator_output_status",
        "received_imu",
    }
    received_by_generation: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    collisions_by_generation: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    for record_index, row in enumerate(records):
        if row.get("type") != "event":
            continue
        event = row.get("event")
        observation = row.get("observation")
        if event in received_event_names and isinstance(observation, Mapping):
            generation = observation["ingress"]["generation"]
            received_by_generation.setdefault(generation, []).append(
                (record_index, observation)
            )
        elif event == "runner_collision_observation" and isinstance(
            observation, Mapping
        ):
            generation = observation["reset_generation"]
            collisions_by_generation.setdefault(generation, []).append(
                (record_index, observation)
            )

    loss = {
        "capture_drops": 0,
        "decoded_frame_drops": 0,
        "writer_queue_drops": 0,
        "writer_errors": 0,
        "ingress_drops": 0,
        "observation_queue_drops": 0,
        "collision_queue_drops": 0,
        "outbound_trace_drops": 0,
        "queue_overflows": 0,
    }
    accounting_exact = True
    prior_new: int | None = None
    completed_generations: set[int] = set()

    def generation_accounting_exact(
        generation: int,
        ingress: Mapping[str, Any],
        collision: Mapping[str, Any],
        *,
        boundary_record_index: int | None,
        buffered_observations: Sequence[Mapping[str, Any]],
        buffered_collisions: Sequence[Mapping[str, Any]],
    ) -> bool:
        received_entries = received_by_generation.get(generation, [])
        collision_entries = collisions_by_generation.get(generation, [])
        received = sorted(
            (value for _, value in received_entries),
            key=lambda value: value["ingress"]["sequence"],
        )
        serialized_collisions = sorted(
            (value for _, value in collision_entries),
            key=lambda value: value["observation_sequence"],
        )
        ingress_sequences = [value["ingress"]["sequence"] for value in received]
        collision_sequences = [
            value["observation_sequence"] for value in serialized_collisions
        ]
        observed_counts = {
            message_type: sum(
                value["ingress"]["message_type"] == message_type
                for value in received
            )
            for message_type in (
                "HIGHRES_IMU",
                "HEARTBEAT",
                "RACE_STATUS",
                "ACTUATOR_OUTPUT_STATUS",
            )
        }
        ingress_counts_exact = (
            ingress["next_sequence"] == len(received)
            and ingress_sequences == list(range(ingress["next_sequence"]))
            and ingress["highres_imu_received"] == observed_counts["HIGHRES_IMU"]
            and ingress["heartbeat_received"] == observed_counts["HEARTBEAT"]
            and ingress["race_status_received"] == observed_counts["RACE_STATUS"]
            and ingress["actuator_received"]
            == observed_counts["ACTUATOR_OUTPUT_STATUS"]
            and sum(observed_counts.values()) == ingress["next_sequence"]
        )
        ingress_capacity_exact = (
            ingress["imu_capacity"] > 0
            and ingress["other_capacity"] > 0
            and ingress["imu_high_watermark"] <= ingress["imu_capacity"]
            and ingress["other_high_watermark"] <= ingress["other_capacity"]
            and ingress["high_watermark"]
            <= ingress["imu_capacity"] + ingress["other_capacity"]
            and ingress["high_watermark"]
            >= max(
                ingress["imu_high_watermark"], ingress["other_high_watermark"]
            )
            and ingress["high_watermark"] <= ingress["next_sequence"]
        )
        collision_counts_exact = (
            collision_sequences == list(range(len(serialized_collisions)))
            and collision["handled"]
            == len(serialized_collisions) + collision["dropped"]
            and collision["capacity"] > 0
            and collision["high_watermark"] <= collision["capacity"]
            and collision["buffered"] <= len(serialized_collisions)
        )
        buffers_exact = (
            ingress["buffered_imu"]
            == sum(
                value["schema"] == "aigp-vq2-received-imu/1"
                for value in buffered_observations
            )
            and ingress["buffered_other"]
            == sum(
                value["schema"] != "aigp-vq2-received-imu/1"
                for value in buffered_observations
            )
            and collision["buffered"] == len(buffered_collisions)
            and list(buffered_observations)
            == received[-len(buffered_observations) :]
            if buffered_observations
            else ingress["buffered_imu"] == 0
            and ingress["buffered_other"] == 0
            and collision["buffered"] == len(buffered_collisions)
        )
        if buffered_collisions:
            buffers_exact = buffers_exact and list(buffered_collisions) == (
                serialized_collisions[-len(buffered_collisions) :]
            )
        if boundary_record_index is not None:
            buffers_exact = buffers_exact and all(
                record_index < boundary_record_index
                for record_index, _ in received_entries + collision_entries
            )
        return (
            ingress["generation"] == generation
            and collision["generation"] == generation
            and ingress_counts_exact
            and ingress_capacity_exact
            and collision_counts_exact
            and buffers_exact
        )

    for record_index, checked in checked_boundary_rows:
        if prior_new is not None and checked["old_generation"] != prior_new:
            accounting_exact = False
        prior_new = checked["new_generation"]
        completed_generations.add(checked["old_generation"])
        ingress = checked["ingress_stats"]
        collision = checked["collision_stats"]
        if not generation_accounting_exact(
            checked["old_generation"],
            ingress,
            collision,
            boundary_record_index=record_index,
            buffered_observations=checked["observations"],
            buffered_collisions=checked["collisions"],
        ):
            accounting_exact = False
        loss["ingress_drops"] += ingress["dropped"]
        loss["observation_queue_drops"] += (
            ingress["imu_dropped"] + ingress["other_dropped"]
        )
        loss["collision_queue_drops"] += collision["dropped"]
        loss["queue_overflows"] += ingress["dropped"] + collision["dropped"]

    raw = final_resource_stats
    final_ingress = raw["ingress"]
    final_collision = raw["collision"]
    final_generation = final_ingress["generation"]
    if final_collision["generation"] != final_generation:
        accounting_exact = False
    if prior_new is not None and final_generation != prior_new:
        accounting_exact = False
    if not generation_accounting_exact(
        final_generation,
        final_ingress,
        final_collision,
        boundary_record_index=None,
        buffered_observations=(),
        buffered_collisions=(),
    ):
        accounting_exact = False
    allowed_generations = completed_generations | {final_generation}
    if not (
        set(received_by_generation) <= allowed_generations
        and set(collisions_by_generation) <= allowed_generations
    ):
        accounting_exact = False
    return {
        "count": len(boundary_rows),
        "loss": loss,
        "accounting_exact": accounting_exact,
        "required_occurrences_exact": _required_reset_boundary_occurrences_exact(
            records,
            boundaries=checked_boundary_rows,
            final_resource_stats=final_resource_stats,
        ),
        "resource_stats_zero": accounting_exact
        and all(value == 0 for value in loss.values()),
    }


def semantic_ambiguities() -> tuple[str, ...]:
    """Return unresolved semantic review questions, if any."""

    return SEMANTIC_AMBIGUITIES


def _validate_ingress_primitive(value: Any, path: str) -> dict[str, Any]:
    row = _object(
        value,
        {
            "schema",
            "stream_id",
            "generation",
            "sequence",
            "message_type",
            "host_clock_id",
            "received_monotonic_ns",
            "source_time_value",
            "source_time_unit",
        },
        path,
    )
    if row["schema"] != "aigp-vq2-mavlink-ingress/1":
        _fail(f"{path}.schema", "unsupported ingress schema")
    _string(row["stream_id"], f"{path}.stream_id")
    _int(row["generation"], f"{path}.generation")
    _int(row["sequence"], f"{path}.sequence")
    message_type = _string(row["message_type"], f"{path}.message_type")
    if message_type not in {
        "HEARTBEAT",
        "RACE_STATUS",
        "HIGHRES_IMU",
        "ACTUATOR_OUTPUT_STATUS",
    }:
        _fail(f"{path}.message_type", "unsupported calibration ingress type")
    if row["host_clock_id"] != attempt_contract.HOST_CLOCK_ID:
        _fail(f"{path}.host_clock_id", "must use host-perf-counter")
    _int(row["received_monotonic_ns"], f"{path}.received_monotonic_ns")
    expected_unit = {
        "HEARTBEAT": None,
        "RACE_STATUS": "ms",
        "HIGHRES_IMU": "us",
        "ACTUATOR_OUTPUT_STATUS": "us",
    }[message_type]
    if expected_unit is None:
        if row["source_time_value"] is not None or row["source_time_unit"] is not None:
            _fail(path, "heartbeat ingress source-time fields must be null")
    else:
        _int(row["source_time_value"], f"{path}.source_time_value")
        if row["source_time_unit"] != expected_unit:
            _fail(f"{path}.source_time_unit", f"must equal {expected_unit!r}")
    return row


def _validate_frame_metadata(row: Mapping[str, Any], path: str) -> tuple[int, int, int]:
    generation = _int(row["generation"], f"{path}.generation")
    frame_id = _int(row["frame_id"], f"{path}.frame_id")
    sim_time_ns = _int(row["sim_time_ns"], f"{path}.sim_time_ns")
    _number(row["received_monotonic_s"], f"{path}.received_monotonic_s")
    shape = _array(row["image_shape"], f"{path}.image_shape", length=3)
    for index, component in enumerate(shape):
        _int(component, f"{path}.image_shape[{index}]", minimum=1)
    if shape[2] != 3 or row["image_dtype"] != "|u1":
        _fail(path, "frame must declare HxWx3 |u1")
    digest = _hash(row["frame_hash"], f"{path}.frame_hash")
    if row["frame_blob"] != f"frames/{digest}.npy":
        _fail(f"{path}.frame_blob", "must be the content-addressed frame path")
    return generation, frame_id, sim_time_ns


def _validate_core_record(row: dict[str, Any], path: str) -> None:
    record_type = row["type"]
    expected_fields = _CORE_KEYS.get(record_type)
    if expected_fields is None:
        _fail(f"{path}.type", f"unsupported core record type {record_type!r}")
    _object(row, _BASE_RECORD_KEYS | expected_fields, path)
    if record_type == "imu":
        if row["received_monotonic_s"] is not None:
            _number(row["received_monotonic_s"], f"{path}.received_monotonic_s")
        if type(row["imu"]) is not dict or row["estimator"] is not None and type(row["estimator"]) is not dict:
            _fail(path, "core IMU payload/estimator shape is invalid")
    elif record_type == "race_status":
        if row["received_monotonic_s"] is not None:
            _number(row["received_monotonic_s"], f"{path}.received_monotonic_s")
        if type(row["race_status"]) is not dict:
            _fail(f"{path}.race_status", "must be an exact object")
    elif record_type == "command":
        if row["kind"] not in {"generated", "sent"}:
            _fail(f"{path}.kind", "must be generated or sent")
        if row["monotonic_s"] is not None:
            _number(row["monotonic_s"], f"{path}.monotonic_s")
        if row["frame_token"] is not None:
            for index, item in enumerate(_array(row["frame_token"], f"{path}.frame_token", length=3)):
                _int(item, f"{path}.frame_token[{index}]")
        command = _object(row["command"], {"roll_rate", "pitch_rate", "yaw_rate", "thrust"}, f"{path}.command")
        for name in command:
            _number(command[name], f"{path}.command.{name}")
    else:
        _validate_frame_metadata(row, path)
        if record_type == "frame":
            if row["detector_latency_ms"] is not None:
                _number(row["detector_latency_ms"], f"{path}.detector_latency_ms")
            if type(row["detections"]) is not list or type(row["tracker"]) is not dict:
                _fail(path, "legacy frame detector/tracker payload is invalid")


def _validate_event_record(row: dict[str, Any], path: str) -> None:
    event = _string(row.get("event"), f"{path}.event")
    if set(row) & _RESERVED_EVENT_FIELDS:
        _fail(path, "event uses a reserved replay semantic field")
    if event in _EVENT_SCHEMA:
        expected = _BASE_RECORD_KEYS | {"event", "observation"}
        if event == "received_imu":
            expected |= {"linked_imu_record_sequence"}
        _object(row, expected, path)
        observation = attempt_contract.validate_powered_record(
            row["observation"], expected_schema=_EVENT_SCHEMA[event]
        )
        if event == "received_imu":
            link = _int(row["linked_imu_record_sequence"], f"{path}.linked_imu_record_sequence")
            if link >= row["sequence"]:
                _fail(f"{path}.linked_imu_record_sequence", "must name a prior core IMU row")
        return
    if event == "mavlink_ingress":
        _object(row, _BASE_RECORD_KEYS | {"event", "observation"}, path)
        _validate_ingress_primitive(row["observation"], f"{path}.observation")
        return
    if event == "camera_frame_timing":
        _object(
            row,
            _BASE_RECORD_KEYS
            | {"event", "observation", "linked_decoded_frame_record_sequence"},
            path,
        )
        try:
            FrameTimingV1.from_primitive(row["observation"])
        except (TypeError, ValueError) as exc:
            raise PoweredCalibrationAnalysisError(
                f"{path}.observation: invalid FrameTimingV1: {exc}"
            ) from exc
        link = _int(
            row["linked_decoded_frame_record_sequence"],
            f"{path}.linked_decoded_frame_record_sequence",
        )
        if link >= row["sequence"]:
            _fail(
                f"{path}.linked_decoded_frame_record_sequence",
                "must name a prior decoded-frame row",
            )
        return
    if event == "camera_frame_timing_observation":
        _object(row, _BASE_RECORD_KEYS | {"event", "observation"}, path)
        try:
            CameraFrameTimingObservationV1.from_primitive(row["observation"])
        except (TypeError, ValueError) as exc:
            raise PoweredCalibrationAnalysisError(
                f"{path}.observation: invalid camera consume timing: {exc}"
            ) from exc
        return
    # Legacy diagnostics are permitted, but they cannot masquerade as a frozen
    # calibration/receive/outbound semantic channel.
    if event.startswith(
        (
            "calibration_",
            "received_",
            "runner_collision_",
            "decoded_dimensions_",
            "attitude_target_",
            "nonattitude_",
            "camera_frame_timing",
            "mavlink_ingress",
        )
    ):
        _fail(f"{path}.event", "unknown reserved-prefix event")
    if not (_BASE_RECORD_KEYS | {"event"}) <= set(row):
        _fail(path, "legacy event is missing its replay envelope")


def validate_replay_snapshot(
    manifest: Any,
    records: Any,
    *,
    manifest_bytes: bytes | None = None,
    records_bytes: bytes | None = None,
    frame_blob_file_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a complete replay `/1` semantic snapshot without opening files."""

    manifest_path = "$manifest"
    base_keys = {
        "schema",
        "record_schema",
        "session_id",
        "started_at",
        "finished_at",
        "complete",
        "private",
        "recording_notice",
        "metadata",
        "record_count",
        "frame_record_count",
        "decoded_frame_record_count",
        "unique_frame_blob_count",
        "integrity",
        "dataset_hash",
        "outcome",
    }
    manifest_row = _object(manifest, base_keys, manifest_path)
    if manifest_row["schema"] != REPLAY_BUNDLE_SCHEMA or manifest_row["record_schema"] != REPLAY_RECORD_SCHEMA:
        _fail(manifest_path, "unsupported replay schema")
    session_id = _string(manifest_row["session_id"], f"{manifest_path}.session_id")
    _string(manifest_row["started_at"], f"{manifest_path}.started_at")
    _string(manifest_row["finished_at"], f"{manifest_path}.finished_at")
    if manifest_row["complete"] is not True or manifest_row["private"] is not True:
        _fail(manifest_path, "bundle must be complete and private")
    if manifest_row["recording_notice"] != RECORDING_NOTICE:
        _fail(f"{manifest_path}.recording_notice", "does not match frozen notice")
    if type(manifest_row["metadata"]) is not dict or type(manifest_row["outcome"]) is not dict:
        _fail(manifest_path, "metadata and outcome must be exact objects")
    counts = {}
    for name in (
        "record_count",
        "frame_record_count",
        "decoded_frame_record_count",
        "unique_frame_blob_count",
    ):
        counts[name] = _int(manifest_row[name], f"{manifest_path}.{name}")
    integrity = _object(
        manifest_row["integrity"],
        {"records_sha256", "frame_blob_file_sha256"},
        f"{manifest_path}.integrity",
    )
    records_digest = _hash(
        integrity["records_sha256"], f"{manifest_path}.integrity.records_sha256"
    )
    blob_map = integrity["frame_blob_file_sha256"]
    if type(blob_map) is not dict:
        _fail(f"{manifest_path}.integrity.frame_blob_file_sha256", "must be an exact object")
    for digest, file_digest in blob_map.items():
        _hash(digest, f"{manifest_path}.integrity.frame_blob_file_sha256.key")
        _hash(file_digest, f"{manifest_path}.integrity.frame_blob_file_sha256.{digest}")
    if len(blob_map) != counts["unique_frame_blob_count"]:
        _fail(manifest_path, "unique frame-blob count does not match integrity map")
    _hash(manifest_row["dataset_hash"], f"{manifest_path}.dataset_hash")
    expected_dataset_hash = attempt_contract.canonical_object_sha256(
        {
            "schema": REPLAY_BUNDLE_SCHEMA,
            "session_id": session_id,
            "started_at": manifest_row["started_at"],
            "finished_at": manifest_row["finished_at"],
            "metadata": manifest_row["metadata"],
            "outcome": manifest_row["outcome"],
            "records_sha256": records_digest,
            "frame_blob_file_sha256": blob_map,
        }
    )
    if manifest_row["dataset_hash"] != expected_dataset_hash:
        _fail(f"{manifest_path}.dataset_hash", "does not match the frozen dataset-hash algorithm")
    if manifest_bytes is not None:
        if bytes(manifest_bytes) != attempt_contract.canonical_json_file_bytes(manifest_row):
            _fail("$manifest_bytes", "manifest bytes are not exact canonical file bytes")

    rows = _array(records, "$records")
    if len(rows) != counts["record_count"]:
        _fail("$records", "record count does not match manifest")
    core_counts = {"frame": 0, "decoded_frame": 0}
    for index, row in enumerate(rows):
        path = f"$records[{index}]"
        if type(row) is not dict:
            _fail(path, "must be an exact object")
        if row.get("schema") != REPLAY_RECORD_SCHEMA:
            _fail(f"{path}.schema", "unsupported replay record schema")
        if row.get("session_id") != session_id:
            _fail(f"{path}.session_id", "does not match manifest")
        if type(row.get("sequence")) is not int or row["sequence"] != index:
            _fail(f"{path}.sequence", "must be contiguous from zero")
        if type(row.get("capture_wall_time_ns")) is not int or row["capture_wall_time_ns"] < 0:
            _fail(f"{path}.capture_wall_time_ns", "must be a nonnegative exact integer")
        record_type = row.get("type")
        if record_type == "event":
            _validate_event_record(row, path)
        else:
            _validate_core_record(row, path)
            if record_type in core_counts:
                core_counts[record_type] += 1
    if core_counts["frame"] != counts["frame_record_count"] or core_counts["decoded_frame"] != counts["decoded_frame_record_count"]:
        _fail("$records", "frame record counts do not match manifest")
    referenced = {
        row["frame_hash"]
        for row in rows
        if row.get("type") in {"frame", "decoded_frame"}
    }
    if referenced != set(blob_map):
        _fail("$records", "referenced frame blobs do not exactly match manifest")
    if records_bytes is not None:
        raw = bytes(records_bytes)
        expected = b"".join(attempt_contract.canonical_json_file_bytes(row) for row in rows)
        if raw != expected:
            _fail("$records_bytes", "records JSONL bytes are not exact canonical rows")
        if hashlib.sha256(raw).hexdigest() != records_digest:
            _fail("$records_bytes", "records SHA-256 does not match manifest")
    if frame_blob_file_sha256 is not None and dict(frame_blob_file_sha256) != blob_map:
        _fail("$frame_blob_file_sha256", "verified blob set/hash map does not match manifest")
    return {"manifest": _clone(manifest_row), "records": _clone(rows)}


def _event_rows(records: Sequence[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in records if row.get("type") == "event" and row.get("event") == name]


def _expected_core_command(observation: Mapping[str, Any]) -> dict[str, Any]:
    command = observation["command"]
    return {
        "roll_rate": command["roll_rate_rad_s"],
        "pitch_rate": command["pitch_rate_rad_s"],
        "yaw_rate": command["yaw_rate_rad_s"],
        "thrust": command["thrust"],
    }


def _expected_frame_token(observation: Mapping[str, Any]) -> list[int] | None:
    frame = observation["source"]["frame"]
    if frame is None:
        return None
    return [frame["generation"], frame["frame_id"], frame["sim_time_ns"]]


def _core_command_matches(
    core: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    kind: str,
    monotonic_key: str,
) -> bool:
    return (
        core["kind"] == kind
        and core["command"] == _expected_core_command(observation)
        and core["frame_token"] == _expected_frame_token(observation)
        and core["monotonic_s"] == observation[monotonic_key] / 1_000_000_000
    )


def _sent_wire_matches_command(observation: Mapping[str, Any]) -> bool:
    """Bind a sent record to the build-3385 IMU-mode post-sign wire payload."""

    receipt = observation["transport"]["receipt"]
    command = observation["command"]
    wire = receipt["wire"]
    return (
        receipt["api"] == "send_attitude_rate"
        and receipt["outcome"] == "returned"
        and wire["type_mask"] == 128
        and wire["q_wxyz"] == [1.0, 0.0, 0.0, 0.0]
        and wire["body_rates_rad_s"]
        == [
            -command["roll_rate_rad_s"],
            -command["pitch_rate_rad_s"],
            -command["yaw_rate_rad_s"],
        ]
        and wire["thrust"] == command["thrust"]
    )


def _sign_reversal_count(values: Sequence[float | int]) -> int:
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    return sum(current != previous for previous, current in zip(signs, signs[1:]))


def _extrema(values: Sequence[float | int], path: str) -> tuple[float | int, float | int]:
    if not values:
        _fail(path, "cannot compute descriptive extrema from an empty sequence")
    for index, value in enumerate(values):
        _number(value, f"{path}[{index}]")
    return min(values), max(values)


def _canonical_multiset(values: Sequence[Any]) -> dict[bytes, int]:
    result: dict[bytes, int] = {}
    for value in values:
        key = _canonical_key(value)
        result[key] = result.get(key, 0) + 1
    return result


def _outbound_resource_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    final_resource_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the final adapter counters to every serialized outbound call.

    Outbound sequence numbers are attempt-global while reset generations
    advance at each persisted reset boundary.  The boundary event is recorded
    synchronously after the adapter advances generations and before the reset
    packet is attempted, so replay order provides the exact generation that
    every later receipt must carry.
    """

    entries = [
        (record_index, row["observation"])
        for record_index, row in enumerate(records)
        if row.get("type") == "event"
        and row.get("event")
        in {"attitude_target_outbound", "nonattitude_outbound"}
    ]
    boundary_entries = [
        (record_index, row["observation"])
        for record_index, row in enumerate(records)
        if row.get("type") == "event"
        and row.get("event") == "calibration_reset_boundary"
    ]
    receipts = [value for _, value in entries]
    raw = final_resource_stats["outbound_receipts"]
    returned = sum(value["outcome"] == "returned" for value in receipts)
    raised = sum(value["outcome"] == "raised" for value in receipts)
    sequences = [value["outbound_sequence"] for value in receipts]
    counters_exact = (
        raw["constructed"] is True
        and raw["next_sequence"] == len(receipts)
        and raw["returned"] == returned
        and raw["raised"] == raised
        and raw["dropped"] == 0
        and raw["buffered"] == 0
        and sequences == list(range(raw["next_sequence"]))
        and raw["capacity"] > 0
        and raw["high_watermark"]
        >= (1 if raw["next_sequence"] else 0)
        and raw["high_watermark"] <= raw["capacity"]
        and raw["high_watermark"] <= raw["next_sequence"]
    )

    # Replay order is the generation ledger.  A boundary advances the current
    # generation before its SIM_RESET receipt can exist; all other receipts
    # retain the most recently established generation.
    generation_exact = True
    current_generation = (
        boundary_entries[0][1]["old_generation"]
        if boundary_entries
        else raw["generation"]
    )
    for row in records:
        if row.get("type") != "event":
            continue
        if row.get("event") == "calibration_reset_boundary":
            boundary = row["observation"]
            if boundary["old_generation"] != current_generation:
                generation_exact = False
            current_generation = boundary["new_generation"]
        elif row.get("event") in {
            "attitude_target_outbound",
            "nonattitude_outbound",
        }:
            if row["observation"]["reset_generation"] != current_generation:
                generation_exact = False
    if (
        raw["generation"] != current_generation
        or raw["generation"] != final_resource_stats["ingress"]["generation"]
        or raw["generation"] != final_resource_stats["collision"]["generation"]
    ):
        generation_exact = False

    # Each persisted boundary must have exactly one reset call in its own
    # post-boundary generation and before the following boundary.
    sim_resets = [
        (record_index, value)
        for record_index, value in entries
        if value.get("schema") == "aigp-vq2-nonattitude-outbound/1"
        and value.get("category") == "sim_reset"
    ]
    reset_receipts_exact = len(sim_resets) == len(boundary_entries)
    if reset_receipts_exact:
        for index, ((boundary_index, boundary), (receipt_index, receipt)) in enumerate(
            zip(boundary_entries, sim_resets)
        ):
            next_boundary_index = (
                boundary_entries[index + 1][0]
                if index + 1 < len(boundary_entries)
                else len(records)
            )
            if not (
                boundary_index < receipt_index < next_boundary_index
                and receipt["reset_generation"] == boundary["new_generation"]
                and receipt["call_start_monotonic_ns"]
                >= boundary["boundary_monotonic_ns"]
            ):
                reset_receipts_exact = False

    attitude_receipts = [
        value
        for value in receipts
        if value.get("schema") == "aigp-vq2-attitude-target-outbound/1"
    ]
    sent_commands = [
        row["observation"]
        for row in records
        if row.get("type") == "event"
        and row.get("event") == "calibration_command_sent"
    ]
    terminal_receipts = [
        value["transport"]["receipt"] for value in sent_commands
    ]
    command_receipts_exact = _canonical_multiset(
        attitude_receipts
    ) == _canonical_multiset(terminal_receipts) and all(
        value["scope"] != "excitation"
        or value["transport"]["receipt"]["reset_generation"]
        == value["reset_epoch"]["ingress_generation"]
        for value in sent_commands
    )

    return {
        "count": len(receipts),
        "counters_exact": counters_exact,
        "generation_exact": generation_exact,
        "reset_receipts_exact": reset_receipts_exact,
        "command_receipts_exact": command_receipts_exact,
        "exact": (
            counters_exact
            and generation_exact
            and reset_receipts_exact
            and command_receipts_exact
        ),
    }


def reconcile_calibration_records(
    manifest: Any,
    records: Any,
    *,
    manifest_bytes: bytes | None = None,
    records_bytes: bytes | None = None,
    frame_blob_file_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate and reconcile the complete powered-calibration replay snapshot.

    Structurally invalid evidence raises.  Semantically complete but unusable
    capture (for example a recorded skipped tick) returns false checks and
    invalidation reasons; callers must not build or publish prospective records
    from such a result.
    """

    snapshot = validate_replay_snapshot(
        manifest,
        records,
        manifest_bytes=manifest_bytes,
        records_bytes=records_bytes,
        frame_blob_file_sha256=frame_blob_file_sha256,
    )
    manifest_row = snapshot["manifest"]
    rows: list[dict[str, Any]] = snapshot["records"]
    resource_evidence = derive_powered_capture_resource_evidence(
        manifest_row["outcome"].get("powered_capture_resource_stats")
    )
    boundary_resource_evidence = _reset_boundary_resource_evidence(
        rows, final_resource_stats=resource_evidence["raw"]
    )
    outbound_resource_evidence = _outbound_resource_evidence(
        rows, final_resource_stats=resource_evidence["raw"]
    )
    aggregate_loss = {
        name: resource_evidence["loss"][name]
        + boundary_resource_evidence["loss"][name]
        for name in resource_evidence["loss"]
    }

    sequenced = [
        row["observation"]
        for row in rows
        if row.get("type") == "event" and row.get("event") in _SEQUENCED_OBSERVATION_EVENTS
    ]
    for expected, observation in enumerate(sequenced):
        if observation["event_sequence"] != expected:
            _fail(
                "$records",
                "phase/command/tick observation event_sequence must be contiguous in replay order",
            )

    generated = [
        row["observation"]
        for row in _event_rows(rows, "calibration_command_generated")
    ]
    generated_by_sequence: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(generated):
        checked = attempt_contract.validate_command_generated(
            item, f"$generated[{index}]"
        )
        sequence = checked["event_sequence"]
        if sequence in generated_by_sequence:
            _fail(f"$generated[{index}].event_sequence", "duplicate generated sequence")
        generated_by_sequence[sequence] = checked

    sent = [row["observation"] for row in _event_rows(rows, "calibration_command_sent")]
    sent_by_sequence: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(sent):
        generated_item = generated_by_sequence.get(item.get("generated_event_sequence"))
        if generated_item is None:
            _fail(f"$sent[{index}].generated_event_sequence", "does not name a generated event")
        checked = attempt_contract.validate_command_sent(
            item, f"$sent[{index}]", generated=generated_item
        )
        sent_by_sequence[checked["event_sequence"]] = checked

    not_sent = [
        row["observation"]
        for row in _event_rows(rows, "calibration_command_not_sent")
    ]
    not_sent_by_sequence: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(not_sent):
        generated_item = generated_by_sequence.get(item.get("generated_event_sequence"))
        if generated_item is None:
            _fail(
                f"$not_sent[{index}].generated_event_sequence",
                "does not name a generated event",
            )
        checked = attempt_contract.validate_command_not_sent(
            item, f"$not_sent[{index}]", generated=generated_item
        )
        not_sent_by_sequence[checked["event_sequence"]] = checked

    dispositions = [
        row["observation"]
        for row in _event_rows(rows, "calibration_tick_disposition")
    ]
    tick_rows: dict[int, dict[str, Any]] = {}
    tick_link_errors = 0
    for index, item in enumerate(dispositions):
        checked = attempt_contract.validate_tick_disposition(
            item, f"$tick_dispositions[{index}]"
        )
        tick = checked["absolute_tick"]
        if tick in tick_rows:
            _fail(f"$tick_dispositions[{index}].absolute_tick", "duplicate tick")
        tick_rows[tick] = checked
        if checked["disposition"] == "sent":
            generated_item = generated_by_sequence.get(checked["generated_event_sequence"])
            terminal = sent_by_sequence.get(checked["terminal_event_sequence"])
            if (
                generated_item is None
                or terminal is None
                or terminal["generated_event_sequence"] != generated_item["event_sequence"]
                or generated_item["absolute_tick"] != tick
            ):
                tick_link_errors += 1
        elif checked["disposition"] == "skipped_after_generation":
            generated_item = generated_by_sequence.get(checked["generated_event_sequence"])
            terminal = not_sent_by_sequence.get(checked["terminal_event_sequence"])
            if (
                generated_item is None
                or terminal is None
                or terminal["generated_event_sequence"] != generated_item["event_sequence"]
                or terminal["outcome"]["reason_code"] != checked["reason_code"]
                or generated_item["absolute_tick"] != tick
            ):
                tick_link_errors += 1
    ticks_exact = sorted(tick_rows) == list(range(245)) and tick_link_errors == 0

    core_generated = [
        row for row in rows if row.get("type") == "command" and row["kind"] == "generated"
    ]
    core_sent = [
        row for row in rows if row.get("type") == "command" and row["kind"] == "sent"
    ]
    unmatched_generation = abs(len(core_generated) - len(generated))
    unmatched_sent = abs(len(core_sent) - len(sent))
    payload_mismatch = 0
    for core, rich in zip(core_generated, generated):
        if not _core_command_matches(
            core, rich, kind="generated", monotonic_key="generated_monotonic_ns"
        ):
            payload_mismatch += 1
    for core, rich in zip(core_sent, sent):
        if not _core_command_matches(
            core, rich, kind="sent", monotonic_key="sent_monotonic_ns"
        ):
            payload_mismatch += 1

    # Eligibility is a half-open host-QPC occurrence, not merely a matching
    # tick label.  Generated evidence must be created inside its slot and the
    # adapter receipt must prove that the actual API call started there.  A
    # call may return after the slot boundary, so completion is not used as a
    # substitute for the call-start occurrence.
    slot_timing_exact = True
    for item in generated:
        if item["scope"] != "excitation":
            continue
        slot = item["slot"]
        generated_ns = item["generated_monotonic_ns"]
        if not (
            slot["release_monotonic_ns"]
            <= generated_ns
            < slot["end_monotonic_ns"]
            <= slot["powered_expiry_monotonic_ns"]
        ):
            slot_timing_exact = False
    for item in sent:
        if item["scope"] != "excitation":
            continue
        slot = item["slot"]
        receipt = item["transport"]["receipt"]
        call_start = receipt["call_start_monotonic_ns"]
        call_end = receipt["call_end_monotonic_ns"]
        generated_item = generated_by_sequence.get(
            item["generated_event_sequence"]
        )
        if (
            generated_item is None
            or not (
                generated_item["generated_monotonic_ns"]
                <= call_start
                <= call_end
                <= item["sent_monotonic_ns"]
            )
            or not (
                slot["release_monotonic_ns"]
                <= call_start
                < slot["end_monotonic_ns"]
                <= slot["powered_expiry_monotonic_ns"]
            )
        ):
            slot_timing_exact = False

    rich_outbound_rows = [
        row["observation"]
        for row in rows
        if row.get("type") == "event"
        and row.get("event") in {"attitude_target_outbound", "nonattitude_outbound"}
    ]
    outbound_generations = [item["reset_generation"] for item in rich_outbound_rows]
    outbound_sequences = [item["outbound_sequence"] for item in rich_outbound_rows]
    outbound_contiguous = (
        outbound_generations == sorted(outbound_generations)
        and outbound_sequences == list(range(len(outbound_sequences)))
    )
    outcomes_returned = all(row["outcome"] == "returned" for row in rich_outbound_rows)
    attitude_rows = [
        row for row in rich_outbound_rows if row["schema"] == "aigp-vq2-attitude-target-outbound/1"
    ]
    sent_receipts = [item["transport"]["receipt"] for item in sent]
    receipt_exact = _canonical_multiset(attitude_rows) == _canonical_multiset(sent_receipts)
    sent_generations_exact = all(
        item["scope"] != "excitation"
        or item["transport"]["receipt"]["reset_generation"]
        == item["reset_epoch"]["ingress_generation"]
        for item in sent
    )
    audit_boundaries = sorted(
        (
            item["transport"]["audit_count_before"],
            item["transport"]["audit_count_after"],
        )
        for item in sent
    )
    audit_contiguous = audit_boundaries == [
        (index, index + 1) for index in range(len(audit_boundaries))
    ]

    received_event_names = {
        "received_heartbeat",
        "received_race_status",
        "received_actuator_output_status",
        "received_imu",
    }
    received_rows = [
        row
        for row in rows
        if row.get("type") == "event" and row.get("event") in received_event_names
    ]
    received_observations = [row["observation"] for row in received_rows]
    received_by_schema: dict[str, dict[bytes, dict[str, Any]]] = {}
    ingress_keys: set[bytes] = set()
    for row in received_rows:
        observation = row["observation"]
        schema_map = received_by_schema.setdefault(observation["schema"], {})
        key = _canonical_key(observation)
        if key in schema_map:
            _fail("$records", "duplicate immutable received observation")
        schema_map[key] = observation
        ingress_key = _canonical_key(observation["ingress"])
        if ingress_key in ingress_keys:
            _fail("$records", "one ingress occurrence was duplicated across payload events")
        ingress_keys.add(ingress_key)

    mavlink_ingress = [
        row["observation"] for row in _event_rows(rows, "mavlink_ingress")
    ]
    ingress_lineage = _canonical_multiset(mavlink_ingress) == _canonical_multiset(
        [item["ingress"] for item in received_observations]
    )

    received_imu_rows = _event_rows(rows, "received_imu")
    core_imu_rows = [row for row in rows if row.get("type") == "imu"]
    linked_imu_sequences: list[int] = []
    imu_linkage = True
    for row in received_imu_rows:
        link = row["linked_imu_record_sequence"]
        linked_imu_sequences.append(link)
        if (
            link >= len(rows)
            or rows[link].get("type") != "imu"
            or rows[link]["imu"] != row["observation"]["imu"]
        ):
            imu_linkage = False
    if (
        len(linked_imu_sequences) != len(set(linked_imu_sequences))
        or set(linked_imu_sequences) != {row["sequence"] for row in core_imu_rows}
        or _canonical_multiset([row["imu"] for row in core_imu_rows])
        != _canonical_multiset(
            [row["observation"]["imu"] for row in received_imu_rows]
        )
    ):
        imu_linkage = False

    race_payloads = [row["race_status"] for row in rows if row.get("type") == "race_status"]
    received_race_payloads = [
        row["observation"]["race_status"]
        for row in _event_rows(rows, "received_race_status")
    ]
    race_core_lineage = _canonical_multiset(race_payloads) == _canonical_multiset(
        received_race_payloads
    )

    decoded_frames = [row for row in rows if row.get("type") == "decoded_frame"]
    decoded_by_sequence = {row["sequence"]: row for row in decoded_frames}
    decoded_by_token = {
        (row["generation"], row["frame_id"], row["sim_time_ns"]): row
        for row in decoded_frames
    }
    if len(decoded_by_token) != len(decoded_frames):
        _fail("$records", "decoded-frame tokens must be unique")
    camera_timing_rows = _event_rows(rows, "camera_frame_timing")
    timing_by_token: dict[tuple[int, int, int], dict[str, Any]] = {}
    camera_links_exact = len(camera_timing_rows) == len(decoded_frames)
    for row in camera_timing_rows:
        link = row["linked_decoded_frame_record_sequence"]
        decoded = decoded_by_sequence.get(link)
        timing = row["observation"]
        identity = timing["identity"]
        token = (
            identity["generation"],
            identity["frame_id"],
            timing["camera_source_time_ns"],
        )
        if (
            decoded is None
            or token
            != (decoded["generation"], decoded["frame_id"], decoded["sim_time_ns"])
            or identity["stream_id"] in {"", None}
            or token in timing_by_token
        ):
            camera_links_exact = False
        timing_by_token[token] = timing
    consume_rows = [
        row["observation"]
        for row in _event_rows(rows, "camera_frame_timing_observation")
    ]
    consume_tokens: list[tuple[int, int, int]] = []
    camera_consumes_exact = len(consume_rows) == len(camera_timing_rows)
    for observation in consume_rows:
        consume_timing = observation["frame_timing"]
        identity = consume_timing["identity"]
        token = (
            identity["generation"],
            identity["frame_id"],
            consume_timing["camera_source_time_ns"],
        )
        consume_tokens.append(token)
        if timing_by_token.get(token) != consume_timing:
            camera_consumes_exact = False
    if len(consume_tokens) != len(set(consume_tokens)):
        camera_consumes_exact = False
    all_frame_rows = [
        row for row in rows if row.get("type") in {"frame", "decoded_frame"}
    ]
    stable_dimensions = bool(decoded_frames) and all(
        row["image_shape"] == [360, 640, 3] and row["image_dtype"] == "|u1"
        for row in all_frame_rows
    )
    admission_rows = _event_rows(rows, "decoded_dimensions_admission")
    if len(admission_rows) != 1 or (
        admission_rows
        and admission_rows[0]["observation"]["first_frame_timing"]
        not in timing_by_token.values()
    ):
        stable_dimensions = False

    source_lineage = True
    target_x: list[float | int] = []
    target_y: list[float | int] = []
    target_area: list[float | int] = []
    gyro_x: list[float | int] = []
    gyro_y: list[float | int] = []
    gyro_z: list[float | int] = []
    excitation_generated = [item for item in generated if item["scope"] == "excitation"]
    cleanup_generated = [item for item in generated if item["scope"] == "cleanup_zero"]
    for item in excitation_generated:
        source = item["source"]
        expected_sources = (
            ("imu", "aigp-vq2-received-imu/1"),
            ("race", "aigp-vq2-received-race-status/1"),
            ("heartbeat", "aigp-vq2-received-heartbeat/1"),
            ("actuator", "aigp-vq2-received-actuator-output-status/1"),
        )
        for name, schema in expected_sources:
            if _canonical_key(source[name]) not in received_by_schema.get(schema, {}):
                source_lineage = False
        frame = source["frame"]
        token = (frame["generation"], frame["frame_id"], frame["sim_time_ns"])
        decoded = decoded_by_token.get(token)
        if (
            decoded is None
            or decoded["image_shape"] != [frame["height"], frame["width"], 3]
            or timing_by_token.get(token) != frame["timing"]
        ):
            source_lineage = False
        watchdogs = item["watchdogs"]
        target_x.append(watchdogs["target_center_px"][0])
        target_y.append(watchdogs["target_center_px"][1])
        target_area.append(watchdogs["target_bbox_area_px"])
        gyro = source["imu"]["imu"]["gyro"]
        gyro_x.append(gyro[0])
        gyro_y.append(gyro[1])
        gyro_z.append(gyro[2])

    tick_disposition_counts = {
        name: sum(item["disposition"] == name for item in dispositions)
        for name in ("sent", "skipped_before_generation", "skipped_after_generation")
    }
    all_ticks_sent = ticks_exact and tick_disposition_counts["sent"] == 245
    no_failed_send = not not_sent and outcomes_returned
    command_pairs_exact = (
        unmatched_generation == 0
        and unmatched_sent == 0
        and payload_mismatch == 0
        and slot_timing_exact
        and all(_sent_wire_matches_command(item) for item in sent)
        and receipt_exact
        and audit_contiguous
    )

    collision_rows = _event_rows(rows, "runner_collision_observation")
    plan_exact = all(
        item["plan"]
        == {
            "plan_id": attempt_contract.EXCITATION_PLAN_ID,
            "sha256": attempt_contract.EXCITATION_PLAN_SHA256,
        }
        for item in excitation_generated
    )
    watchdogs_passed = all(
        item["watchdogs"]["result"] == "pass"
        and item["watchdogs"]["failure_codes"] == []
        for item in excitation_generated
    )

    reset_epochs = sorted(
        {_canonical_key(item["reset_epoch"]): item["reset_epoch"] for item in excitation_generated}.values(),
        key=_canonical_key,
    )
    segment_accounting: list[dict[str, Any]] = []
    for planned in attempt_contract.frozen_excitation_plan()["segments"]:
        segment_id = planned["segment_id"]
        planned_ticks = planned["last_tick"] - planned["first_tick"] + 1
        segment_generated = sum(
            item["segment_id"] == segment_id for item in excitation_generated
        )
        segment_sent = sum(
            item["scope"] == "excitation" and item["segment_id"] == segment_id
            for item in sent
        )
        segment_skipped = sum(
            item["segment_id"] == segment_id
            and item["disposition"] != "sent"
            for item in dispositions
        )
        segment_accounting.append(
            {
                "segment_id": segment_id,
                "planned_ticks": planned_ticks,
                "generated": segment_generated,
                "sent": segment_sent,
                "skipped": segment_skipped,
            }
        )

    slots = [item["slot"] for item in excitation_generated]
    first_release = min((slot["release_monotonic_ns"] for slot in slots), default=0)
    last_end = max((slot["end_monotonic_ns"] for slot in slots), default=0)
    expiries = {slot["powered_expiry_monotonic_ns"] for slot in slots}
    expiry = next(iter(expiries)) if len(expiries) == 1 else 0

    if not excitation_generated:
        _fail("$records", "powered calibration replay contains no excitation commands")

    descriptive = {
        "target_observation_count": len(target_x),
        "target_center_x_px_min": _extrema(target_x, "$target_x")[0],
        "target_center_x_px_max": _extrema(target_x, "$target_x")[1],
        "target_center_y_px_min": _extrema(target_y, "$target_y")[0],
        "target_center_y_px_max": _extrema(target_y, "$target_y")[1],
        "target_bbox_area_px_min": _extrema(target_area, "$target_area")[0],
        "target_bbox_area_px_max": _extrema(target_area, "$target_area")[1],
        "gyro_x_rad_s_min": _extrema(gyro_x, "$gyro_x")[0],
        "gyro_x_rad_s_max": _extrema(gyro_x, "$gyro_x")[1],
        "gyro_y_rad_s_min": _extrema(gyro_y, "$gyro_y")[0],
        "gyro_y_rad_s_max": _extrema(gyro_y, "$gyro_y")[1],
        "gyro_z_rad_s_min": _extrema(gyro_z, "$gyro_z")[0],
        "gyro_z_rad_s_max": _extrema(gyro_z, "$gyro_z")[1],
        "roll_reversal_count": _sign_reversal_count(
            [
                attempt_contract.excitation_command_for_tick(tick)[
                    "roll_rate_rad_s"
                ]
                for tick in sorted(tick_rows)
            ]
        ),
        "pitch_reversal_count": _sign_reversal_count(
            [
                attempt_contract.excitation_command_for_tick(tick)[
                    "pitch_rate_rad_s"
                ]
                for tick in sorted(tick_rows)
            ]
        ),
        "semantics": "descriptive_only_no_acceptance_threshold",
    }

    checks = {
        "bundle_complete": manifest_bytes is not None and records_bytes is not None,
        "frame_hashes_valid": frame_blob_file_sha256 is not None,
        "resource_stats_zero": (
            resource_evidence["resource_stats_zero"]
            and boundary_resource_evidence["resource_stats_zero"]
        ),
        "resource_counts_bound": (
            resource_evidence["raw"]["recorder"]["enqueued"] == len(rows)
            and resource_evidence["raw"]["recorder"][
                "decoded_frames_enqueued"
            ]
            == len(decoded_frames)
            and resource_evidence["raw"]["snapshot_capture"][
                "observed_frames"
            ]
            == len(decoded_frames)
            and resource_evidence["raw"]["vision"]["frames_decoded"]
            == len(decoded_frames)
            and resource_evidence["raw"]["outbound_receipts"]["returned"]
            == len(rich_outbound_rows)
            and outbound_resource_evidence["counters_exact"]
        ),
        "decoded_dimensions_640x360_stable": stable_dimensions,
        "camera_lineage_complete": (
            camera_links_exact and camera_consumes_exact and source_lineage
        ),
        "imu_lineage_complete": imu_linkage and source_lineage,
        "race_heartbeat_actuator_collision_lineage_complete": (
            ingress_lineage and race_core_lineage and source_lineage and not collision_rows
        ),
        "outbound_allowlist_exact": (
            outbound_contiguous
            and outcomes_returned
            and receipt_exact
            and sent_generations_exact
            and outbound_resource_evidence["exact"]
        ),
        "command_pairs_exact": command_pairs_exact,
        "ticks_0_through_244_accounted": ticks_exact,
        "plan_exact": plan_exact and len(excitation_generated) == 245,
        "watchdogs_passed": watchdogs_passed and all_ticks_sent and no_failed_send,
    }
    checks["reset_boundaries_exact"] = boundary_resource_evidence[
        "required_occurrences_exact"
    ]
    invalid_reasons: set[str] = set()
    if not checks["command_pairs_exact"] or not checks["ticks_0_through_244_accounted"]:
        invalid_reasons.add("command_reconciliation_failed")
    if not checks["watchdogs_passed"]:
        invalid_reasons.add("watchdog_failed")
    if not all(
        checks[name]
        for name in (
            "frame_hashes_valid",
            "decoded_dimensions_640x360_stable",
            "camera_lineage_complete",
            "imu_lineage_complete",
            "race_heartbeat_actuator_collision_lineage_complete",
            "resource_stats_zero",
            "resource_counts_bound",
            "reset_boundaries_exact",
        )
    ):
        invalid_reasons.add("capture_incomplete")
    if not checks["outbound_allowlist_exact"]:
        invalid_reasons.add("unexpected_outbound")

    decoded_hashes = sorted(
        manifest_row["integrity"]["frame_blob_file_sha256"],
        key=lambda item: item.encode("utf-8"),
    )
    counts = {
        "decoded_frames": len(decoded_frames),
        "unique_decoded_hashes": len(decoded_hashes),
        "camera_timing_records": len(camera_timing_rows),
        "imu_records": sum(row.get("type") == "imu" for row in rows),
        "mavlink_ingress_records": len(mavlink_ingress),
        "race_records": len(_event_rows(rows, "received_race_status")),
        "heartbeat_records": len(_event_rows(rows, "received_heartbeat")),
        "actuator_records": len(_event_rows(rows, "received_actuator_output_status")),
        "collision_records": len(collision_rows),
        "generated_commands": len(generated),
        "sent_commands": len(sent),
        "not_sent_commands": len(not_sent),
        "ticks_sent": tick_disposition_counts["sent"],
        "ticks_skipped_before_generation": tick_disposition_counts["skipped_before_generation"],
        "ticks_skipped_after_generation": tick_disposition_counts["skipped_after_generation"],
        "capture_drops": aggregate_loss["capture_drops"],
        "decoded_frame_drops": aggregate_loss["decoded_frame_drops"],
        "writer_errors": aggregate_loss["writer_errors"],
        "ingress_drops": aggregate_loss["ingress_drops"],
        "queue_overflows": aggregate_loss["queue_overflows"],
        "send_failed_or_uncertain": sum(
            item["outcome"]["kind"] == "send_failed_or_uncertain"
            for item in not_sent
        ),
    }
    outbound_category_counts = {
        "timesync": 0,
        "gcs_heartbeat": 0,
        "sim_reset": 0,
        "arm": 0,
        "disarm": 0,
        "attitude_target": len(attitude_rows),
        "position_target": 0,
        "other_command": 0,
        "receipt_count": len(rich_outbound_rows),
        "receipt_returned": sum(item["outcome"] == "returned" for item in rich_outbound_rows),
        "receipt_raised": sum(item["outcome"] == "raised" for item in rich_outbound_rows),
        "receipt_dropped": 0,
        "receipt_buffered": 0,
    }
    for item in rich_outbound_rows:
        if item["schema"] == "aigp-vq2-nonattitude-outbound/1":
            outbound_category_counts[item["category"]] += 1
    result = {
        "valid": not invalid_reasons and all(checks.values()),
        "invalid_reasons": sorted(invalid_reasons, key=lambda item: item.encode("utf-8")),
        "checks": checks,
        "counts": counts,
        "capture_counts": {
            "record_count": len(rows),
            "decoded_frames": len(decoded_frames),
            "frame_blobs": len(decoded_hashes),
            "camera_timing_records": len(camera_timing_rows),
            "imu_records": counts["imu_records"],
            "mavlink_ingress_records": counts["mavlink_ingress_records"],
            "race_records": counts["race_records"],
            "heartbeat_records": counts["heartbeat_records"],
            "actuator_records": counts["actuator_records"],
            "collision_records": counts["collision_records"],
            "generated_commands": counts["generated_commands"],
            "sent_commands": counts["sent_commands"],
            "not_sent_commands": counts["not_sent_commands"],
            "tick_dispositions": len(dispositions),
            **aggregate_loss,
        },
        "command_accounting": {
            "attitude_target_audit_delta": len(attitude_rows),
            "generated_count": len(generated),
            "sent_count": len(sent),
            "not_sent_count": len(not_sent),
            "unmatched_generation_count": unmatched_generation,
            "unmatched_sent_count": unmatched_sent,
            "failed_or_uncertain_count": counts["send_failed_or_uncertain"],
            "envelope_violation_count": tick_link_errors,
            "payload_mismatch_count": payload_mismatch,
            "all_reconciled": command_pairs_exact and tick_link_errors == 0,
        },
        "excitation_accounting": {
            "plan_id": attempt_contract.EXCITATION_PLAN_ID,
            "plan_sha256": attempt_contract.EXCITATION_PLAN_SHA256,
            "tick_count": 245,
            "segments": segment_accounting,
            "first_release_monotonic_ns": first_release,
            "last_slot_end_monotonic_ns": last_end,
            "powered_expiry_monotonic_ns": expiry,
        },
        "descriptive_support": descriptive,
        "reset_epochs": _clone(reset_epochs),
        "decoded_content_sha256": decoded_hashes,
        "cleanup_generated": _clone(cleanup_generated),
        "cleanup_sent": _clone([item for item in sent if item["scope"] == "cleanup_zero"]),
        "outbound_receipts": _clone(rich_outbound_rows),
        "outbound_audit": outbound_category_counts,
        "command_identity": {
            "candidate_commits": sorted(
                {item["candidate_commit"] for item in (*generated, *sent, *not_sent)},
                key=lambda item: item.encode("utf-8"),
            ),
            "attempt_context_sha256": sorted(
                {
                    item["attempt_context_sha256"]
                    for item in (*generated, *sent, *not_sent, *dispositions)
                },
                key=lambda item: item.encode("utf-8"),
            ),
        },
        "dimensions_admission": (
            _clone(admission_rows[0]["observation"])
            if len(admission_rows) == 1
            else None
        ),
        "bundle_artifacts": {
            "manifest_sha256": (
                hashlib.sha256(bytes(manifest_bytes)).hexdigest()
                if manifest_bytes is not None
                else None
            ),
            "manifest_size_bytes": (
                len(bytes(manifest_bytes)) if manifest_bytes is not None else None
            ),
            "records_sha256": (
                hashlib.sha256(bytes(records_bytes)).hexdigest()
                if records_bytes is not None
                else None
            ),
            "records_size_bytes": (
                len(bytes(records_bytes)) if records_bytes is not None else None
            ),
            "frame_blob_file_sha256": (
                dict(frame_blob_file_sha256)
                if frame_blob_file_sha256 is not None
                else None
            ),
        },
        "manifest": manifest_row,
    }
    return _clone(result)


def _sha256_regular_file(path: Path, *, maximum_bytes: int, label: str) -> str:
    """Hash one stable bounded file/ancestry through the shared OS contract."""

    try:
        from scripts.aigp_vq2_powered_runtime import stable_file_identity

        identity = stable_file_identity(
            str(path), max_bytes=maximum_bytes
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise PoweredCalibrationAnalysisError(
            f"{label}: stable regular-file identity failed: {exc}"
        ) from exc
    return identity.sha256


def load_and_verify_sealed_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Read and verify an explicitly named replay bundle, without writing.

    No evidence-root or session default exists here.  The caller must supply an
    absolute bundle directory.  The frozen replay reader performs its complete
    `/1` verification (including decoded content hashes); this function then
    re-reads and cross-checks the exact bytes handed to the pure reconciler.
    """

    path = Path(bundle_path)
    if not path.is_absolute():
        _fail("$bundle_path", "must be caller-supplied and absolute")
    try:
        from aigp_loop import replay as replay_contract

        reader = replay_contract.ReplayBundleReader(path, require_complete=True)
        _reader_summary, records = reader.verify_and_read(verify_frames=True)
    except (OSError, TypeError, ValueError) as exc:
        raise PoweredCalibrationAnalysisError(
            f"$bundle_path: sealed replay verification failed: {exc}"
        ) from exc

    manifest_path = path / "manifest.json"
    records_path = path / "records.jsonl"
    manifest_limit = int(replay_contract.MAX_REPLAY_MANIFEST_BYTES)
    records_limit = int(replay_contract.MAX_REPLAY_RECORDS_BYTES)
    manifest_digest_before = _sha256_regular_file(
        manifest_path, maximum_bytes=manifest_limit, label="$bundle.manifest"
    )
    records_digest_before = _sha256_regular_file(
        records_path, maximum_bytes=records_limit, label="$bundle.records"
    )
    try:
        manifest_bytes = manifest_path.read_bytes()
        records_bytes = records_path.read_bytes()
    except OSError as exc:
        raise PoweredCalibrationAnalysisError(
            "$bundle_path: canonical replay files could not be re-read"
        ) from exc
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != manifest_digest_before
        or hashlib.sha256(records_bytes).hexdigest() != records_digest_before
        or _sha256_regular_file(
            manifest_path,
            maximum_bytes=manifest_limit,
            label="$bundle.manifest.readback",
        )
        != manifest_digest_before
        or _sha256_regular_file(
            records_path,
            maximum_bytes=records_limit,
            label="$bundle.records.readback",
        )
        != records_digest_before
    ):
        _fail("$bundle_path", "manifest or records identity changed during readback")
    manifest = attempt_contract.parse_canonical_json_bytes(
        manifest_bytes, file_form=True
    )
    if manifest != reader.manifest:
        _fail("$bundle.manifest", "changed after replay-reader verification")

    blob_hashes: dict[str, str] = {}
    for digest, expected in manifest["integrity"]["frame_blob_file_sha256"].items():
        actual = _sha256_regular_file(
            path / "frames" / f"{digest}.npy",
            maximum_bytes=int(replay_contract.MAX_REPLAY_FRAME_BLOB_BYTES),
            label=f"$bundle.frames.{digest}",
        )
        if actual != expected:
            _fail(f"$bundle.frames.{digest}", "changed after replay verification")
        blob_hashes[digest] = actual

    snapshot = validate_replay_snapshot(
        manifest,
        records,
        manifest_bytes=manifest_bytes,
        records_bytes=records_bytes,
        frame_blob_file_sha256=blob_hashes,
    )
    return {
        **snapshot,
        "manifest_bytes": manifest_bytes,
        "records_bytes": records_bytes,
        "frame_blob_file_sha256": blob_hashes,
        "bundle_path": str(path),
    }


_REPORT_CHECKS = (
    "identity_bound",
    "build3385_training_attested",
    "bundle_complete",
    "frame_hashes_valid",
    "decoded_dimensions_640x360_stable",
    "camera_lineage_complete",
    "imu_lineage_complete",
    "race_heartbeat_actuator_collision_lineage_complete",
    "capture_loss_zero",
    "ingress_loss_zero",
    "outbound_allowlist_exact",
    "command_pairs_exact",
    "ticks_0_through_244_accounted",
    "plan_exact",
    "watchdogs_passed",
    "cleanup_confirmed",
    "fallback_not_used",
    "child_process_tree_exited",
    "ports_released",
    "lease_released",
    "simulator_topology_unchanged",
    "simulator_responsive",
    "scheduled_task_absent",
    "exclusive_binds_and_peers_exact",
    "collection_invalidating_codes_empty",
    "conditional_on_nominal_gate_config",
    "no_fit_or_rank_inspection",
)


def _validated(label: str, validator: Any, value: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return validator(value, **kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PoweredCalibrationAnalysisError(f"{label}: {exc}") from exc


def _artifact_index(seal: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in seal["artifacts"]}


def _file_hash(value: Any) -> str:
    return attempt_contract.canonical_file_sha256(value)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
        os.path.abspath(os.fspath(right))
    )


def _load_acceptance_lease_records(
    lease_final: Any,
    *,
    envelope: Mapping[str, Any],
    authority: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the immutable lease files, cadence, and success lineage.

    The final index is intentionally only a compact list of file identities.
    Acceptance therefore must read the referenced rows rather than trust index
    event labels or a producer's release boolean.  The lower-level validator
    proves canonical bytes, directory closure, transitions, and cadence; the
    checks below additionally bind those rows to this exact attempt, child, and
    successful no-takeover collection.
    """

    lease = _validated(
        "$lease_final", validate_powered_live_lease_ledger, lease_final
    )
    expected_directory = Path(envelope["context"]["paths"]["lease_directory"])
    records: list[dict[str, Any]] = []
    previous_hash: str | None = None
    last_heartbeat_ns: int | None = None
    max_gap_ns = envelope["context"]["deadline_durations_ns"][
        "lease_heartbeat_max_gap"
    ]
    first: dict[str, Any] | None = None
    child_binding: dict[str, Any] | None = None
    cleanup_binding: dict[str, Any] | None = None

    for entry in lease["records"]:
        generation = entry["generation"]
        path = Path(entry["path"])
        if not _same_path(path.parent, expected_directory):
            raise PoweredCalibrationAnalysisError(
                "$lease_final: indexed generation escaped the frozen lease directory"
            )
        if path.name != f"generation-{generation:06d}.json":
            raise PoweredCalibrationAnalysisError(
                "$lease_final: indexed generation filename is not canonical"
            )
        try:
            loaded = load_powered_live_lease_record(path)
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            raise PoweredCalibrationAnalysisError(
                f"$lease_final.records[{generation}]: {exc}"
            ) from exc
        row = _validated(
            f"$lease_final.records[{generation}]",
            validate_powered_live_lease_record,
            loaded,
        )
        digest = _file_hash(row)
        if (
            digest != entry["sha256"]
            or row["generation"] != generation
            or row["event"] != entry["event"]
            or row["predecessor_sha256"] != previous_hash
        ):
            raise PoweredCalibrationAnalysisError(
                "$lease_final: indexed generation identity or predecessor mismatched"
            )
        if first is None:
            first = row
        else:
            previous = records[-1]
            for name in (
                "mutex_name",
                "attempt_id",
                "attempt_envelope_sha256",
                "attempt_context_sha256",
                "wrapper_process",
                "host_clock_id",
                "qpc_frequency_hz",
            ):
                if row[name] != first[name]:
                    raise PoweredCalibrationAnalysisError(
                        f"$lease_final: immutable lease field {name} changed"
                    )
            if row["observed_monotonic_ns"] < previous["observed_monotonic_ns"]:
                raise PoweredCalibrationAnalysisError(
                    "$lease_final: lease occurrence time regressed"
                )
            if previous["event"] == "released":
                raise PoweredCalibrationAnalysisError(
                    "$lease_final: lease chain continued after release"
                )
            if row["event"] == "takeover":
                if (
                    previous["owner_role"] != "wrapper"
                    or previous["abandoned"]
                    or row["owner_token_sha256"]
                    == previous["owner_token_sha256"]
                ):
                    raise PoweredCalibrationAnalysisError(
                        "$lease_final: takeover ownership transition is invalid"
                    )
            elif (
                row["owner_role"] != previous["owner_role"]
                or row["owner_token_sha256"]
                != previous["owner_token_sha256"]
            ):
                raise PoweredCalibrationAnalysisError(
                    "$lease_final: owner changed without takeover"
                )
            if row["event"] == "released" and previous["event"] != "release_intent":
                raise PoweredCalibrationAnalysisError(
                    "$lease_final: released row lacks release_intent predecessor"
                )

        if child_binding is not None and row["child_process"] != child_binding:
            raise PoweredCalibrationAnalysisError(
                "$lease_final: child process binding changed"
            )
        if cleanup_binding is not None and row["cleanup_process"] != cleanup_binding:
            raise PoweredCalibrationAnalysisError(
                "$lease_final: cleanup process binding changed"
            )
        child_binding = row["child_process"] or child_binding
        cleanup_binding = row["cleanup_process"] or cleanup_binding

        observed = row["observed_monotonic_ns"]
        event = row["event"]
        if event == "acquired":
            last_heartbeat_ns = observed
        elif event in {"heartbeat", "takeover"}:
            if (
                last_heartbeat_ns is None
                or observed - last_heartbeat_ns > max_gap_ns
            ):
                raise PoweredCalibrationAnalysisError(
                    "$lease_final: lease heartbeat maximum gap was exceeded"
                )
            last_heartbeat_ns = observed
        elif event == "release_intent" and (
            last_heartbeat_ns is None
            or observed - last_heartbeat_ns > max_gap_ns
        ):
            raise PoweredCalibrationAnalysisError(
                "$lease_final: release intent exceeded the heartbeat maximum gap"
            )

        records.append(row)
        previous_hash = digest

    if first is None or not records:
        raise PoweredCalibrationAnalysisError("$lease_final: lease ledger is empty")
    context = envelope["context"]
    if (
        lease["task_id"] != context["task_id"]
        or lease["session_id"] != context["session_id"]
        or lease["attempt_id"] != context["attempt_id"]
        or lease["attempt_envelope_sha256"] != _file_hash(envelope)
        or lease["orphaned_pending_files"] != []
        or first["event"] != "acquired"
        or first["owner_role"] != "wrapper"
        or first["owner_token_sha256"]
        != envelope["capabilities"]["lease_owner_sha256"]
        or first["wrapper_process"] != context["wrapper_process"]
        or first["attempt_context_sha256"] != envelope["context_sha256"]
        or first["qpc_frequency_hz"] != context["host"]["qpc_frequency_hz"]
        or records[-1]["event"] != "released"
        or records[-1]["release_proved"] is not True
        or any(row["abandoned"] or row["owner_role"] != "wrapper" for row in records)
    ):
        raise PoweredCalibrationAnalysisError(
            "$lease_final: final lease is not the exact successful wrapper lineage"
        )

    by_hash = {entry["sha256"]: entry for entry in lease["records"]}
    authority_entry = by_hash.get(authority["lease_record_sha256"])
    certificate_entry = by_hash.get(certificate["lease"]["record_sha256"])
    if authority_entry is None or certificate_entry is None:
        raise PoweredCalibrationAnalysisError(
            "$lease_final: authority or cleanup lease row is absent"
        )
    authority_row = records[authority_entry["generation"]]
    certificate_row = records[certificate_entry["generation"]]
    if (
        authority_row["child_process"] != authority["process"]
        or authority_row["owner_role"] != "wrapper"
        or authority_row["event"] in {"release_intent", "released"}
        or certificate_entry["generation"] != certificate["lease"]["generation"]
        or certificate_entry["generation"] < authority_entry["generation"]
        or certificate_row["child_process"] != authority["process"]
        or certificate_row["owner_role"] != certificate["lease"]["owner_role"]
        or certificate["lease"]["authority_valid"] is not True
    ):
        raise PoweredCalibrationAnalysisError(
            "$lease_final: delegated child/cleanup lease lineage mismatched"
        )
    return lease, records


def validate_safety_evidence(
    reconciliation: Any,
    *,
    live_freeze: Any,
    attempt_envelope: Any,
    bundle_verification: Any,
    capture_seal: Any,
    child_authority: Any,
    child_process_result: Any,
    child_cleanup_certificate: Any,
    lease_final: Any,
    training_attestation: Any,
    process_prechild: Any,
    process_postchild: Any,
) -> dict[str, Any]:
    """Cross-bind all non-live safety evidence needed for publication.

    Schema-invalid inputs raise immediately.  Complete, schema-valid evidence
    with a semantic mismatch returns false checks and is therefore ineligible
    for prospective construction or publication.
    """

    summary = _object(
        reconciliation,
        {
            "valid",
            "invalid_reasons",
            "checks",
            "counts",
            "capture_counts",
            "command_accounting",
            "excitation_accounting",
            "descriptive_support",
            "reset_epochs",
            "decoded_content_sha256",
            "cleanup_generated",
            "cleanup_sent",
            "outbound_receipts",
            "outbound_audit",
            "command_identity",
            "dimensions_admission",
            "bundle_artifacts",
            "manifest",
        },
        "$reconciliation",
    )
    if type(summary["valid"]) is not bool or type(summary["checks"]) is not dict:
        _fail("$reconciliation", "valid/checks have invalid types")

    freeze = _validated(
        "$live_freeze", attempt_contract.validate_live_freeze, live_freeze
    )
    envelope = _validated(
        "$attempt_envelope",
        attempt_contract.validate_attempt,
        attempt_envelope,
        live_freeze=freeze,
    )
    verification = _validated(
        "$bundle_verification",
        attempt_contract.validate_bundle_verification,
        bundle_verification,
    )
    seal = _validated(
        "$capture_seal", attempt_contract.validate_capture_seal, capture_seal
    )
    authority = _validated(
        "$child_authority",
        attempt_contract.validate_process_authority,
        child_authority,
        attempt=envelope,
        argv=envelope["context"]["child_argv"],
    )
    certificate = _validated(
        "$child_cleanup_certificate",
        attempt_contract.validate_cleanup_certificate,
        child_cleanup_certificate,
    )
    process_result = _validated(
        "$child_process_result",
        attempt_contract.validate_process_result,
        child_process_result,
        cleanup_certificate=certificate,
    )
    prechild = _validated(
        "$process_prechild",
        attempt_contract.validate_simulator_process_proof,
        process_prechild,
    )
    postchild = _validated(
        "$process_postchild",
        attempt_contract.validate_simulator_process_proof,
        process_postchild,
    )
    attestation = _validated(
        "$training_attestation",
        attempt_contract.validate_training_attestation,
        training_attestation,
        process_proof=prechild,
    )
    lease, lease_records = _load_acceptance_lease_records(
        lease_final,
        envelope=envelope,
        authority=authority,
        certificate=certificate,
    )

    hashes = {
        "live_freeze": _file_hash(freeze),
        "attempt_envelope": _file_hash(envelope),
        "bundle_verification": _file_hash(verification),
        "capture_seal": _file_hash(seal),
        "child_authority": _file_hash(authority),
        "child_process_result": _file_hash(process_result),
        "child_cleanup_certificate": _file_hash(certificate),
        "lease_final": _file_hash(lease),
        "training_attestation": _file_hash(attestation),
        "process_prechild": _file_hash(prechild),
        "process_postchild": _file_hash(postchild),
    }
    artifacts = _artifact_index(seal)

    def artifact_hash(name: str) -> str | None:
        item = artifacts.get(name)
        return None if item is None else item["sha256"]

    manifest = summary["manifest"]
    bundle_files = summary["bundle_artifacts"]
    if type(bundle_files) is not dict:
        _fail("$reconciliation.bundle_artifacts", "must be an exact object")
    verified_blob_map = bundle_files.get("frame_blob_file_sha256")
    verification_frames = {
        item["name"].removeprefix("replay_frame/"): item["sha256"]
        for item in verification["bundle"]["frames"]
        if item["name"].startswith("replay_frame/")
    }
    expected_blob_map = manifest["integrity"]["frame_blob_file_sha256"]

    identity_pairs = {
        "candidate_commit": freeze["candidate"]["commit"],
        "live_freeze_sha256": hashes["live_freeze"],
        "attempt_context_sha256": envelope["context_sha256"],
        "attempt_envelope_sha256": hashes["attempt_envelope"],
        "child_authority_sha256": hashes["child_authority"],
        "child_cleanup_certificate_sha256": hashes[
            "child_cleanup_certificate"
        ],
        "lease_final_sha256": hashes["lease_final"],
    }
    verification_identity_bound = all(
        verification["identity"].get(name) == expected
        for name, expected in identity_pairs.items()
    ) and verification["identity"]["child_process_result_sha256"] == hashes[
        "child_process_result"
    ]

    seal_identity_expected = {
        "candidate_commit": freeze["candidate"]["commit"],
        "code_sha256": freeze["candidate"]["code_sha256"],
        "live_freeze_sha256": hashes["live_freeze"],
        "attempt_context_sha256": envelope["context_sha256"],
        "attempt_envelope_sha256": hashes["attempt_envelope"],
        "target_config_sha256": envelope["context"]["target_config"]["sha256"],
        "capture_authorization_sha256": envelope["context"][
            "capture_authorization"
        ]["sha256"],
        "excitation_plan_id": attempt_contract.EXCITATION_PLAN_ID,
        "excitation_plan_sha256": attempt_contract.EXCITATION_PLAN_SHA256,
        "training_attestation_sha256": hashes["training_attestation"],
        "simulator_process_proof_sha256": hashes["process_prechild"],
        "simulator_final_process_proof_sha256": hashes["process_postchild"],
        "child_authority_sha256": hashes["child_authority"],
        "cleanup_authority_sha256": None,
        "lease_final_sha256": hashes["lease_final"],
        "bundle_verification_sha256": hashes["bundle_verification"],
    }
    seal_identity_bound = seal["identity"] == seal_identity_expected

    required_object_artifacts = {
        "live_freeze": hashes["live_freeze"],
        "attempt_envelope": hashes["attempt_envelope"],
        "training_attestation": hashes["training_attestation"],
        "process_prechild": hashes["process_prechild"],
        "process_postchild": hashes["process_postchild"],
        "child_authority": hashes["child_authority"],
        "child_cleanup_certificate": hashes["child_cleanup_certificate"],
        "lease_final": hashes["lease_final"],
        "bundle_verification": hashes["bundle_verification"],
        # The child's canonical process result is the runner-stdout record.
        "runner_stdout": hashes["child_process_result"],
    }
    object_artifacts_bound = all(
        artifact_hash(name) == digest
        for name, digest in required_object_artifacts.items()
    )

    bundle_refs_bound = (
        verification["bundle"]["dataset_hash"] == manifest["dataset_hash"]
        and verification["bundle"]["manifest"]["sha256"]
        == bundle_files.get("manifest_sha256")
        and verification["bundle"]["manifest"]["size_bytes"]
        == bundle_files.get("manifest_size_bytes")
        and verification["bundle"]["records"]["sha256"]
        == bundle_files.get("records_sha256")
        and verification["bundle"]["records"]["size_bytes"]
        == bundle_files.get("records_size_bytes")
        and verification_frames == expected_blob_map
        and verified_blob_map == expected_blob_map
        and artifact_hash("replay_manifest") == bundle_files.get("manifest_sha256")
        and artifact_hash("replay_records") == bundle_files.get("records_sha256")
        and all(
            artifact_hash(f"replay_frame/{digest}") == file_hash
            for digest, file_hash in expected_blob_map.items()
        )
    )

    semantic_capture_names = tuple(summary["capture_counts"])
    capture_counts_bound = all(
        seal["capture_stats"].get(name) == summary["capture_counts"][name]
        for name in semantic_capture_names
    )
    capture_loss_names = (
        "capture_drops",
        "decoded_frame_drops",
        "writer_queue_drops",
        "writer_errors",
        "ingress_drops",
        "observation_queue_drops",
        "collision_queue_drops",
        "outbound_trace_drops",
        "queue_overflows",
    )
    capture_loss_zero = all(
        seal["capture_stats"][name] == 0 for name in capture_loss_names
    )

    result_replay = process_result["artifacts"]["replay_bundle"]
    result_legacy = process_result["artifacts"]["legacy_record"]
    process_artifacts_bound = (
        result_replay["state"] == "closed"
        and result_replay["dataset_hash"] == manifest["dataset_hash"]
        and result_replay["manifest_sha256"]
        == bundle_files.get("manifest_sha256")
        and result_replay["records_sha256"] == bundle_files.get("records_sha256")
        and result_legacy["state"] == "closed"
        and result_legacy["sha256"] == artifact_hash("legacy_record")
    )

    cleanup_receipts_bound = (
        certificate["outbound_receipts"] == summary["outbound_receipts"]
        and process_result["outbound_audit"] == summary["outbound_audit"]
        and seal["outbound_audit"] == summary["outbound_audit"]
    )
    cleanup_zero_bound = (
        len(summary["cleanup_generated"]) == 1
        and len(summary["cleanup_sent"]) == 1
        and certificate["zero_command"]["state"] == "returned"
        and certificate["zero_command"]["generated"]
        == summary["cleanup_generated"][0]
        and certificate["zero_command"]["terminal"]
        == summary["cleanup_sent"][0]
    )
    endpoints = certificate["endpoints"]
    exclusive_binds_and_peers_exact = (
        endpoints["mavlink"]["state"] == "closed_with_peer"
        and endpoints["camera"]["state"] == "closed_with_peer"
        and endpoints["mavlink"]["rejected_source_count"] == 0
        and endpoints["camera"]["rejected_source_count"] == 0
        and endpoints["mavlink"]["frozen_peer"]["host"] == "127.0.0.1"
        and endpoints["camera"]["frozen_peer"]["host"] == "127.0.0.1"
        and endpoints["mavlink"]["bind"]["owner_process"]
        == authority["process"]
        and endpoints["camera"]["bind"]["owner_process"]
        == authority["process"]
        and {
            (
                item["port"],
                item["role"],
                item["pid"],
                item["creation_filetime_100ns"],
            )
            for item in postchild["ports"]["active_owner_observations"]
        }
        == {
            (
                14550,
                "powered_child",
                authority["process"]["pid"],
                authority["process"]["creation_filetime_100ns"],
            ),
            (
                5600,
                "powered_child",
                authority["process"]["pid"],
                authority["process"]["creation_filetime_100ns"],
            ),
        }
    )
    authority_bound = (
        authority["role"] == "powered_child"
        and process_result["producer_role"] == "powered_child"
        and process_result["process_authority_sha256"] == hashes["child_authority"]
        and certificate["producer_role"] == "powered_child"
        and certificate["authority"]["process_authority"]["sha256"]
        == hashes["child_authority"]
        and certificate["authority"]["producer"] == authority["process"]
        and certificate["authority"]["attempt_context_sha256"]
        == envelope["context_sha256"]
        and certificate["authority"]["attempt_envelope_sha256"]
        == hashes["attempt_envelope"]
        and authority["training_attestation_sha256"]
        == hashes["training_attestation"]
        and authority["simulator_process_proof_sha256"]
        == hashes["process_prechild"]
    )
    lease_record_bound = any(
        item["generation"] == certificate["lease"]["generation"]
        and item["sha256"] == certificate["lease"]["record_sha256"]
        for item in lease["records"]
    ) and bool(lease_records)

    topology_unchanged = (
        prechild["phase"] == "prechild"
        and postchild["phase"] == "postchild"
        and prechild["wrapper_process"] == postchild["wrapper_process"]
        and prechild["wrapper_process"] == authority["wrapper_process"]
        and prechild["launch"] == postchild["launch"]
        and prechild["launcher_process"] == postchild["launcher_process"]
        and prechild["payload_process"] == postchild["payload_process"]
        and prechild["window"] == postchild["window"]
        and prechild["build"] == postchild["build"] == 3385
        and prechild["topology"] == postchild["topology"]
        and prechild["scheduled_task"]["observations"]
        == postchild["scheduled_task"]["observations"][:-1]
    )
    seal_cleanup = seal["cleanup"]
    cleanup_expected = {
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
    cleanup_summary_exact = all(
        seal_cleanup[name] == expected for name, expected in cleanup_expected.items()
    ) and seal_cleanup["child_certificate_sha256"] == hashes[
        "child_cleanup_certificate"
    ]
    fallback_artifacts = {
        "cleanup_authority",
        "fallback_cleanup_certificate",
        "cleanup_stdout",
        "cleanup_stderr",
    }
    fallback_not_used = (
        cleanup_summary_exact
        and not fallback_artifacts.intersection(artifacts)
        and seal["identity"]["cleanup_authority_sha256"] is None
    )
    cleanup_confirmed = (
        certificate["outcome"] == "proved"
        and certificate["failure_codes"] == []
        and certificate["disarm"]["state"] == "confirmed"
        and certificate["reset"]["state"] == "confirmed"
        and certificate["final_state"]["state"] == "confirmed"
        and certificate["final_state"]["disarmed"] is True
        and certificate["zero_command"]["state"] == "returned"
        and all(certificate["transport"].values())
        and cleanup_zero_bound
        and cleanup_receipts_bound
        and cleanup_summary_exact
    )

    checks = {
        "identity_bound": (
            verification_identity_bound
            and seal_identity_bound
            and object_artifacts_bound
            and bundle_refs_bound
            and process_artifacts_bound
            and authority_bound
            and lease_record_bound
            and any(
                item["sha256"] == authority["lease_record_sha256"]
                for item in lease["records"]
            )
            and lease["task_id"] == attempt_contract.TASK_ID
            and lease["session_id"] == attempt_contract.SESSION_ID
            and lease["attempt_id"] == attempt_contract.ATTEMPT_ID
            and lease["attempt_envelope_sha256"] == hashes["attempt_envelope"]
            and summary["command_identity"]
            == {
                "candidate_commits": [freeze["candidate"]["commit"]],
                "attempt_context_sha256": [envelope["context_sha256"]],
            }
        ),
        "build3385_training_attested": (
            attestation["mode"] == "Training"
            and attestation["simulator_process_proof_sha256"]
            == hashes["process_prechild"]
            and prechild["build"] == 3385
        ),
        "bundle_complete": (
            summary["checks"].get("bundle_complete") is True
            and verification["valid"] is True
            and bundle_refs_bound
            and capture_counts_bound
        ),
        "frame_hashes_valid": (
            summary["checks"].get("frame_hashes_valid") is True
            and verification_frames == expected_blob_map
        ),
        "decoded_dimensions_640x360_stable": summary["checks"].get(
            "decoded_dimensions_640x360_stable"
        )
        is True,
        "camera_lineage_complete": summary["checks"].get(
            "camera_lineage_complete"
        )
        is True,
        "imu_lineage_complete": summary["checks"].get("imu_lineage_complete")
        is True,
        "race_heartbeat_actuator_collision_lineage_complete": summary[
            "checks"
        ].get("race_heartbeat_actuator_collision_lineage_complete")
        is True,
        "capture_loss_zero": capture_loss_zero,
        "ingress_loss_zero": (
            seal["capture_stats"]["ingress_drops"] == 0
            and seal["capture_stats"]["observation_queue_drops"] == 0
            and seal["capture_stats"]["collision_queue_drops"] == 0
        ),
        "outbound_allowlist_exact": (
            summary["checks"].get("outbound_allowlist_exact") is True
            and cleanup_receipts_bound
        ),
        "command_pairs_exact": summary["checks"].get("command_pairs_exact")
        is True,
        "ticks_0_through_244_accounted": summary["checks"].get(
            "ticks_0_through_244_accounted"
        )
        is True,
        "plan_exact": summary["checks"].get("plan_exact") is True,
        "watchdogs_passed": summary["checks"].get("watchdogs_passed") is True,
        "cleanup_confirmed": cleanup_confirmed,
        "fallback_not_used": fallback_not_used,
        "child_process_tree_exited": (
            process_result["outcome"] == "completed"
            and seal_cleanup["child_exit"] == "proved"
            and seal_cleanup["processes"] == "exited"
        ),
        "ports_released": seal_cleanup["ports"] == "free",
        "lease_released": (
            lease["release_proved"] is True
            and seal_cleanup["lease"] == "released"
        ),
        "simulator_topology_unchanged": (
            topology_unchanged and seal_cleanup["simulator_topology"] == "unchanged"
        ),
        "simulator_responsive": (
            postchild["responsive"] is True
            and postchild["window"]["responsive"] is True
            and seal_cleanup["simulator_responsive"] == "yes"
        ),
        "scheduled_task_absent": (
            all(
                item["absent"] is True
                for item in postchild["scheduled_task"]["observations"]
            )
            and seal_cleanup["scheduled_task"] == "absent"
        ),
        "exclusive_binds_and_peers_exact": exclusive_binds_and_peers_exact,
        "collection_invalidating_codes_empty": (
            certificate["collection_invalidating_codes"] == []
            and certificate["collisions"]["invalidating_occurrence_count"] == 0
        ),
        "conditional_on_nominal_gate_config": (
            seal["identity"]["target_config_sha256"]
            == envelope["context"]["target_config"]["sha256"]
            and summary["dimensions_admission"] is not None
            and summary["dimensions_admission"]["config_sha256"]
            == seal["identity"]["target_config_sha256"]
        ),
        "no_fit_or_rank_inspection": True,
    }
    if set(checks) != set(_REPORT_CHECKS):
        raise RuntimeError("internal report-check inventory drift")
    invalid_reasons = set(summary["invalid_reasons"])
    if not checks["identity_bound"]:
        invalid_reasons.add("artifact_mismatch")
    if not checks["outbound_allowlist_exact"]:
        invalid_reasons.add("unexpected_outbound")
    if not checks["cleanup_confirmed"]:
        invalid_reasons.add("cleanup_uncertain")
    if not all(checks.values()):
        invalid_reasons.add("capture_incomplete")
    result = {
        "valid": summary["valid"] is True
        and all(checks.values())
        and not invalid_reasons,
        "invalid_reasons": sorted(
            invalid_reasons, key=lambda item: item.encode("utf-8")
        ),
        "checks": checks,
        "hashes": hashes,
        "artifacts": _clone(artifacts),
        "live_freeze": freeze,
        "attempt_envelope": envelope,
        "bundle_verification": verification,
        "capture_seal": seal,
        "child_authority": authority,
        "child_process_result": process_result,
        "child_cleanup_certificate": certificate,
        "lease_final": lease,
        "training_attestation": attestation,
        "process_prechild": prechild,
        "process_postchild": postchild,
    }
    return _clone(result)


def _split_artifact_timing(
    phase_deadline: Any, *, prepared_monotonic_ns: int, path: str
) -> dict[str, Any]:
    deadline = _validated(
        path,
        attempt_contract.validate_phase_deadline,
        phase_deadline,
        expected_phase="split_publish",
    )
    timing = {**deadline, "prepared_monotonic_ns": prepared_monotonic_ns}
    return _validated(
        path,
        attempt_contract.validate_artifact_timing,
        timing,
        expected_phase="split_publish",
    )


def build_prospective_publications(
    reconciliation: Any,
    safety_evidence: Any,
    *,
    split_phase_deadline: Any,
    claim_path: str,
    registry_path: str,
    claimed_at_utc: str,
    claimed_monotonic_ns: int,
    published_at_utc: str,
    published_monotonic_ns: int,
    generated_at_utc: str,
    report_prepared_monotonic_ns: int,
) -> dict[str, Any]:
    """Construct claim -> registry -> report bytes entirely in memory.

    Every validation and cross-binding check must already pass.  No file is
    opened by this function, and the returned hashes use canonical complete
    file bytes.  The dependency direction is deliberately acyclic: the claim
    binds neither successor, the registry binds the claim, and the report binds
    both.
    """

    summary = _clone(reconciliation)
    safety = _clone(safety_evidence)
    if summary.get("valid") is not True or safety.get("valid") is not True:
        _fail("$prospective", "all replay and safety checks must pass before construction")
    checks = safety.get("checks")
    if type(checks) is not dict or set(checks) != set(_REPORT_CHECKS) or not all(
        value is True for value in checks.values()
    ):
        _fail("$prospective.checks", "must contain every exact true report check")
    if safety.get("invalid_reasons") != [] or summary.get("invalid_reasons") != []:
        _fail("$prospective.invalid_reasons", "must be empty")
    if not (
        type(claimed_monotonic_ns) is int
        and type(published_monotonic_ns) is int
        and type(report_prepared_monotonic_ns) is int
        and claimed_monotonic_ns <= published_monotonic_ns <= report_prepared_monotonic_ns
    ):
        _fail("$prospective", "publication preparation times must be ordered integers")

    claim_timing = _split_artifact_timing(
        split_phase_deadline,
        prepared_monotonic_ns=claimed_monotonic_ns,
        path="$prospective.claim.timing",
    )
    registry_timing = _split_artifact_timing(
        split_phase_deadline,
        prepared_monotonic_ns=published_monotonic_ns,
        path="$prospective.registry.timing",
    )
    report_timing = _split_artifact_timing(
        split_phase_deadline,
        prepared_monotonic_ns=report_prepared_monotonic_ns,
        path="$prospective.report.timing",
    )

    artifacts = safety["artifacts"]
    run_names = {
        "bundle_verification",
        "child_cleanup_certificate",
        "legacy_record",
        "replay_manifest",
        "replay_records",
        "runner_stdout",
        "runner_stderr",
    }
    run_names.update(
        f"replay_frame/{digest}" for digest in summary["decoded_content_sha256"]
    )
    if not run_names.issubset(artifacts):
        _fail(
            "$prospective.claim.run_artifacts",
            f"capture seal lacks {sorted(run_names - set(artifacts))!r}",
        )
    run_artifacts = sorted(
        (_clone(artifacts[name]) for name in run_names),
        key=lambda item: item["name"].encode("utf-8"),
    )
    seal = safety["capture_seal"]
    seal_identity = seal["identity"]
    capture_seal_sha256 = safety["hashes"]["capture_seal"]
    claim = {
        "schema": "aigp-vq2-package2-run-split-claim/1",
        "task_id": attempt_contract.TASK_ID,
        "session_id": attempt_contract.SESSION_ID,
        "attempt_id": attempt_contract.ATTEMPT_ID,
        "claimed_at_utc": claimed_at_utc,
        "claimed_monotonic_ns": claimed_monotonic_ns,
        "timing": claim_timing,
        "run_id": "F01-A01/reset-epoch-1/excitation-1",
        "assigned_split": "discovery_fit",
        "identity": {
            "attempt_context_sha256": seal_identity["attempt_context_sha256"],
            "attempt_envelope_sha256": seal_identity["attempt_envelope_sha256"],
            "capture_seal_sha256": capture_seal_sha256,
            "excitation_plan_id": attempt_contract.EXCITATION_PLAN_ID,
            "excitation_plan_sha256": attempt_contract.EXCITATION_PLAN_SHA256,
        },
        "reset_epochs": _clone(summary["reset_epochs"]),
        "run_artifacts": run_artifacts,
        "decoded_content_sha256": _clone(summary["decoded_content_sha256"]),
        "derivative_sha256": [],
        "collision_policy": (
            "f01_fixed_future_whole_run_discovery_fit_or_global_exclusion"
        ),
    }
    claim = _validated(
        "$prospective.claim", attempt_contract.validate_split_claim, claim
    )
    claim_sha256 = _file_hash(claim)

    registry = {
        "schema": "aigp-vq2-package2-split-registry/1",
        "task_id": attempt_contract.TASK_ID,
        "session_id": attempt_contract.SESSION_ID,
        "attempt_id": attempt_contract.ATTEMPT_ID,
        "published_at_utc": published_at_utc,
        "published_monotonic_ns": published_monotonic_ns,
        "timing": registry_timing,
        "registry_id": "vq2-package2-calibration",
        "revision": 1,
        "previous_registry_sha256": None,
        "claims": [
            {
                "claim_path": claim_path,
                "claim_sha256": claim_sha256,
                "session_id": attempt_contract.SESSION_ID,
                "attempt_id": attempt_contract.ATTEMPT_ID,
                "run_id": "F01-A01/reset-epoch-1/excitation-1",
                "assigned_split": "discovery_fit",
                "activation": "requires_matching_attempt_complete",
            }
        ],
        "content_groups": [
            {
                "decoded_sha256": digest,
                "run_ids": ["F01-A01/reset-epoch-1/excitation-1"],
                "assigned_split": "discovery_fit",
                "disposition": "assigned",
                "activation": "requires_matching_attempt_complete",
            }
            for digest in summary["decoded_content_sha256"]
        ],
    }
    registry = _validated(
        "$prospective.registry",
        attempt_contract.validate_split_registry,
        registry,
        split_claim=claim,
    )
    registry_sha256 = _file_hash(registry)

    seal_artifacts = safety["artifacts"]
    report_identity = {
        name: seal_identity[name]
        for name in (
            "candidate_commit",
            "live_freeze_sha256",
            "attempt_context_sha256",
            "attempt_envelope_sha256",
            "target_config_sha256",
            "capture_authorization_sha256",
            "excitation_plan_id",
            "excitation_plan_sha256",
            "training_attestation_sha256",
            "simulator_process_proof_sha256",
            "simulator_final_process_proof_sha256",
            "child_authority_sha256",
            "cleanup_authority_sha256",
            "lease_final_sha256",
            "bundle_verification_sha256",
        )
    }
    report = {
        "schema": "aigp-vq2-powered-calibration-acquisition-report/1",
        "task_id": attempt_contract.TASK_ID,
        "session_id": attempt_contract.SESSION_ID,
        "attempt_id": attempt_contract.ATTEMPT_ID,
        "generated_at_utc": generated_at_utc,
        "timing": report_timing,
        "collection_valid": True,
        "invalid_reasons": [],
        "reference_scope": {
            "conditional_on_nominal_gate_config": True,
            "geometry_status": "nominal_unverified_for_build_3385_training",
            "target_config_sha256": seal_identity["target_config_sha256"],
        },
        "identity": report_identity,
        "input_artifacts": {
            "capture_seal_sha256": capture_seal_sha256,
            "bundle_dataset_hash": summary["manifest"]["dataset_hash"],
            "bundle_verification_sha256": safety["hashes"][
                "bundle_verification"
            ],
            "bundle_manifest_sha256": seal_artifacts["replay_manifest"][
                "sha256"
            ],
            "bundle_records_sha256": seal_artifacts["replay_records"]["sha256"],
            "legacy_record_sha256": seal_artifacts["legacy_record"]["sha256"],
            "lease_final_sha256": safety["hashes"]["lease_final"],
            "runner_stdout_sha256": seal_artifacts["runner_stdout"]["sha256"],
            "runner_stderr_sha256": seal_artifacts["runner_stderr"]["sha256"],
            "child_cleanup_certificate_sha256": safety["hashes"][
                "child_cleanup_certificate"
            ],
            "fallback_cleanup_certificate_sha256": None,
        },
        "checks": _clone(checks),
        "counts": _clone(summary["counts"]),
        "command_accounting": _clone(summary["command_accounting"]),
        "excitation_accounting": _clone(summary["excitation_accounting"]),
        "descriptive_support": _clone(summary["descriptive_support"]),
        "calibration_status": {
            "intrinsics": "uncomputed",
            "distortion": "uncomputed",
            "camera_to_body_rotation": "uncomputed",
            "camera_imu_time_model": "uncomputed",
            "rank": "uncomputed",
            "covariance": "uncomputed",
            "empirical_limits": "uncomputed",
        },
        "unmeasured": [
            "absolute_host_phase",
            "accepted_calibration_coefficients",
            "command_to_actuator_response",
            "empirical_limits",
            "encode_queue_component_delays",
            "package2_acceptance",
            "render_exposure_delay",
        ],
        "split": {
            "assigned_split": "discovery_fit",
            "claim_path": claim_path,
            "claim_sha256": claim_sha256,
            "registry_path": registry_path,
            "registry_sha256": registry_sha256,
            "activation": "requires_matching_attempt_complete",
        },
    }
    report = _validated(
        "$prospective.report",
        attempt_contract.validate_acquisition_report,
        report,
    )
    report_sha256 = _file_hash(report)
    return {
        "claim": claim,
        "registry": registry,
        "report": report,
        "hashes": {
            "claim": claim_sha256,
            "registry": registry_sha256,
            "report": report_sha256,
        },
        "publication_order": ["claim", "registry", "report"],
    }


def validate_prospective_publications(value: Any) -> dict[str, Any]:
    """Validate an in-memory non-circular claim/registry/report triple."""

    row = _object(
        value,
        {"claim", "registry", "report", "hashes", "publication_order"},
        "$prospective",
    )
    claim = _validated(
        "$prospective.claim", attempt_contract.validate_split_claim, row["claim"]
    )
    registry = _validated(
        "$prospective.registry",
        attempt_contract.validate_split_registry,
        row["registry"],
        split_claim=claim,
    )
    report = _validated(
        "$prospective.report",
        attempt_contract.validate_acquisition_report,
        row["report"],
    )
    hashes = _object(row["hashes"], {"claim", "registry", "report"}, "$prospective.hashes")
    expected = {
        "claim": _file_hash(claim),
        "registry": _file_hash(registry),
        "report": _file_hash(report),
    }
    for name, digest in hashes.items():
        _hash(digest, f"$prospective.hashes.{name}")
    if hashes != expected:
        _fail("$prospective.hashes", "do not match canonical complete-file bytes")
    if row["publication_order"] != ["claim", "registry", "report"]:
        _fail("$prospective.publication_order", "must be claim, registry, report")
    if (
        report["split"]["claim_sha256"] != expected["claim"]
        or report["split"]["registry_sha256"] != expected["registry"]
    ):
        _fail("$prospective.report.split", "does not bind claim and registry")
    # A direct value search makes the non-circularity invariant explicit.  A
    # coincidental digest in unrelated data is treated fail-closed as well.
    claim_bytes = attempt_contract.canonical_json_bytes(claim)
    if expected["registry"].encode() in claim_bytes or expected["report"].encode() in claim_bytes:
        _fail("$prospective.claim", "must not bind registry or report hashes")
    registry_bytes = attempt_contract.canonical_json_bytes(registry)
    if expected["report"].encode() in registry_bytes:
        _fail("$prospective.registry", "must not bind report hash")
    return {
        "claim": claim,
        "registry": registry,
        "report": report,
        "hashes": expected,
        "publication_order": ["claim", "registry", "report"],
    }


def _validated_hash_array(value: Any, path: str, *, nonempty: bool) -> list[str]:
    items = _array(value, path)
    checked = [_hash(item, f"{path}[{index}]") for index, item in enumerate(items)]
    expected = sorted(set(checked), key=lambda item: item.encode("utf-8"))
    if checked != expected or len(checked) != len(expected):
        _fail(path, "must be unique and sorted by ordinal UTF-8 bytes")
    if nonempty and not checked:
        _fail(path, "must not be empty")
    return checked


def _validate_collision_runs(value: Any, *, anchor_run_id: str) -> list[dict[str, Any]]:
    rows = _array(value, "$runs")
    if not rows:
        _fail("$runs", "must not be empty")
    checked: list[dict[str, Any]] = []
    run_ids: list[str] = []
    for index, value_row in enumerate(rows):
        path = f"$runs[{index}]"
        row = _object(
            value_row,
            {
                "run_id",
                "assigned_split",
                "decoded_content_sha256",
                "derivative_sha256",
            },
            path,
        )
        run_id = _string(row["run_id"], f"{path}.run_id")
        assigned = _string(row["assigned_split"], f"{path}.assigned_split")
        decoded = _validated_hash_array(
            row["decoded_content_sha256"],
            f"{path}.decoded_content_sha256",
            nonempty=True,
        )
        derivatives = _validated_hash_array(
            row["derivative_sha256"],
            f"{path}.derivative_sha256",
            nonempty=False,
        )
        if set(decoded) & set(derivatives):
            _fail(path, "decoded and derivative hash sets must be disjoint within a run")
        run_ids.append(run_id)
        checked.append(
            {
                "run_id": run_id,
                "assigned_split": assigned,
                "decoded_content_sha256": decoded,
                "derivative_sha256": derivatives,
            }
        )
    expected_ids = sorted(set(run_ids), key=lambda item: item.encode("utf-8"))
    if run_ids != expected_ids or len(run_ids) != len(expected_ids):
        _fail("$runs", "must be unique and sorted by run_id")
    if anchor_run_id not in run_ids:
        _fail("$anchor_run_id", "must identify one supplied run")
    anchor = checked[run_ids.index(anchor_run_id)]
    if (
        anchor_run_id == "F01-A01/reset-epoch-1/excitation-1"
        and anchor["assigned_split"] != "discovery_fit"
    ):
        _fail("$anchor_run_id", "F01 is immutable discovery_fit")
    return checked


def plan_global_content_assignment(
    runs: Any,
    *,
    anchor_run_id: str,
    allow_anchor_whole_run_join: bool,
) -> dict[str, Any]:
    """Resolve global decoded/derivative collisions at whole-run granularity.

    Cross-split components containing the immutable F01 anchor either join its
    split as complete runs or are globally excluded.  Cross-split components
    without that anchor are conservatively excluded because this function has
    no reviewed authority to choose between their splits.  No frame, crop,
    label, derivative, or observation is reassigned independently.
    """

    if type(anchor_run_id) is not str or not anchor_run_id:
        _fail("$anchor_run_id", "must be a nonempty exact string")
    if type(allow_anchor_whole_run_join) is not bool:
        _fail("$allow_anchor_whole_run_join", "must be an exact boolean")
    checked = _validate_collision_runs(runs, anchor_run_id=anchor_run_id)
    index_by_id = {row["run_id"]: index for index, row in enumerate(checked)}
    parent = list(range(len(checked)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[str, list[int]] = {}
    hash_kinds: dict[str, set[str]] = {}
    for index, row in enumerate(checked):
        for kind, field in (
            ("decoded", "decoded_content_sha256"),
            ("derivative", "derivative_sha256"),
        ):
            for digest in row[field]:
                owners.setdefault(digest, []).append(index)
                hash_kinds.setdefault(digest, set()).add(kind)
    for indices in owners.values():
        first = indices[0]
        for other in indices[1:]:
            union(first, other)

    components: dict[int, list[int]] = {}
    for index in range(len(checked)):
        components.setdefault(find(index), []).append(index)
    anchor_split = checked[index_by_id[anchor_run_id]]["assigned_split"]
    resolution_by_index: dict[int, tuple[str | None, str]] = {}
    collision_rows: list[dict[str, Any]] = []
    for indices in components.values():
        component_runs = [checked[index] for index in indices]
        splits = sorted(
            {row["assigned_split"] for row in component_runs},
            key=lambda item: item.encode("utf-8"),
        )
        has_cross_split_collision = len(splits) > 1
        contains_anchor = any(row["run_id"] == anchor_run_id for row in component_runs)
        if not has_cross_split_collision:
            effective = splits[0]
            disposition = "assigned"
        elif contains_anchor and allow_anchor_whole_run_join:
            effective = anchor_split
            disposition = "whole_run_join"
        else:
            effective = None
            disposition = "globally_excluded"
        for index in indices:
            resolution_by_index[index] = (effective, disposition)
        if has_cross_split_collision:
            collision_rows.append(
                {
                    "run_ids": sorted(
                        (row["run_id"] for row in component_runs),
                        key=lambda item: item.encode("utf-8"),
                    ),
                    "claimed_splits": splits,
                    "resolution": disposition,
                    "effective_split": effective,
                }
            )

    output_runs: list[dict[str, Any]] = []
    for index, row in enumerate(checked):
        effective, disposition = resolution_by_index[index]
        root = find(index)
        component_run_ids = sorted(
            (checked[item]["run_id"] for item in components[root]),
            key=lambda item: item.encode("utf-8"),
        )
        output_runs.append(
            {
                **_clone(row),
                "effective_split": effective,
                "disposition": disposition,
                "component_run_ids": component_run_ids,
            }
        )

    content_groups: list[dict[str, Any]] = []
    for digest in sorted(owners, key=lambda item: item.encode("utf-8")):
        owner_indices = sorted(set(owners[digest]))
        # Every owner of one hash is necessarily in the same connected
        # component; apply the component-wide result to all its derivatives.
        effective, disposition = resolution_by_index[owner_indices[0]]
        content_groups.append(
            {
                "sha256": digest,
                "kinds": sorted(hash_kinds[digest]),
                "run_ids": sorted(
                    (checked[index]["run_id"] for index in owner_indices),
                    key=lambda item: item.encode("utf-8"),
                ),
                "effective_split": effective,
                "disposition": disposition,
            }
        )
    collision_rows.sort(key=lambda item: _canonical_key(item["run_ids"]))
    return {
        "anchor_run_id": anchor_run_id,
        "allow_anchor_whole_run_join": allow_anchor_whole_run_join,
        "runs": output_runs,
        "content_groups": content_groups,
        "collisions": collision_rows,
        "valid": True,
    }


def validate_global_content_assignment(
    value: Any,
    *,
    source_runs: Any,
    anchor_run_id: str,
    allow_anchor_whole_run_join: bool,
) -> dict[str, Any]:
    """Require byte-for-byte canonical equality with the frozen global plan."""

    expected = plan_global_content_assignment(
        source_runs,
        anchor_run_id=anchor_run_id,
        allow_anchor_whole_run_join=allow_anchor_whole_run_join,
    )
    candidate = _clone(value)
    if candidate != expected:
        _fail(
            "$global_content_assignment",
            "does not equal the deterministic whole-run assignment",
        )
    return expected


def _path_has_reparse_attribute(path: Path) -> bool:
    info = path.lstat()
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return path.is_symlink() or bool(attributes & reparse)


def _prepare_create_new_target(root: Path, target: Path) -> Path:
    if not root.is_absolute() or not target.is_absolute():
        _fail("$publication", "root and every target must be explicit absolute paths")
    root = Path(os.path.abspath(root))
    target = Path(os.path.abspath(target))
    if not root.exists() or not root.is_dir() or _path_has_reparse_attribute(root):
        _fail("$publication_root", "must be an existing non-reparse directory")
    if target == root:
        _fail("$publication.target", "cannot replace the publication root")
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise PoweredCalibrationAnalysisError(
            "$publication.target: escapes the caller-supplied root"
        ) from exc
    current = root
    for component in relative.parts[:-1]:
        if component in {"", ".", ".."}:
            _fail("$publication.target", "contains an unsafe path component")
        current = current / component
        if not current.exists() or not current.is_dir() or _path_has_reparse_attribute(current):
            _fail("$publication.target", "parent must be an existing non-reparse directory")
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PoweredCalibrationAnalysisError(
            "$publication.target: target absence could not be proved"
        ) from exc
    else:
        _fail("$publication.target", "already exists; overwrite is forbidden")
    return target


def publish_create_new_json_sequence(
    publications: Sequence[tuple[Path | str, Any]],
    *,
    publication_root: Path | str,
) -> tuple[dict[str, Any], ...]:
    """Create canonical JSON files in caller order with no overwrite/rollback.

    Every object, path, parent, and target absence is prevalidated before the
    first write.  A race or I/O failure after one successful immutable write is
    surfaced as :class:`PartialPublicationError`; already-published evidence is
    deliberately preserved for invalidation/forensics.
    """

    if isinstance(publications, (str, bytes, bytearray)) or not isinstance(
        publications, Sequence
    ):
        _fail("$publications", "must be an exact ordered sequence")
    if not publications:
        _fail("$publications", "must not be empty")
    root = Path(publication_root)
    prepared: list[tuple[Path, bytes, str]] = []
    normalized_targets: set[str] = set()
    for index, item in enumerate(publications):
        if type(item) is not tuple or len(item) != 2:
            _fail(f"$publications[{index}]", "must be an exact (path, object) tuple")
        raw_path, value = item
        target = _prepare_create_new_target(root, Path(raw_path))
        normalized = os.path.normcase(str(target))
        if normalized in normalized_targets:
            _fail("$publications", "targets must be distinct")
        normalized_targets.add(normalized)
        try:
            payload = attempt_contract.canonical_json_file_bytes(value)
            parsed = attempt_contract.parse_canonical_json_bytes(
                payload, file_form=True
            )
        except (TypeError, ValueError) as exc:
            raise PoweredCalibrationAnalysisError(
                f"$publications[{index}]: value is not canonical JSON: {exc}"
            ) from exc
        if parsed != value:
            _fail(f"$publications[{index}]", "canonical round trip changed value")
        prepared.append((target, payload, hashlib.sha256(payload).hexdigest()))

    published: list[dict[str, Any]] = []
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOINHERIT", 0))
    for target, payload, digest in prepared:
        descriptor: int | None = None
        target_created = False
        try:
            descriptor = os.open(target, flags, 0o600)
            target_created = True
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("create-new write made no progress")
                written += count
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            readback = target.read_bytes()
            if readback != payload or hashlib.sha256(readback).hexdigest() != digest:
                raise OSError("create-new readback identity mismatch")
            published.append(
                {
                    "path": str(target),
                    "size_bytes": len(payload),
                    "sha256": digest,
                }
            )
        except Exception as exc:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if target_created or published:
                partial_paths = [item["path"] for item in published]
                if target_created and str(target) not in partial_paths:
                    partial_paths.append(str(target))
                raise PartialPublicationError(
                    "publication failed after create-new made the sequence "
                    f"partial ({len(partial_paths)} path(s)): {exc}",
                    published=partial_paths,
                ) from exc
            raise PoweredCalibrationAnalysisError(
                f"publication failed before any complete file: {exc}"
            ) from exc
    return tuple(_clone(published))


def publish_prospective_publications(
    prospective: Any,
    *,
    publication_root: Path | str,
    claim_path: Path | str,
    registry_path: Path | str,
    report_path: Path | str,
) -> tuple[dict[str, Any], ...]:
    """Publish a validated triple in the sole allowed dependency order."""

    checked = validate_prospective_publications(prospective)
    if checked["registry"]["claims"][0]["claim_path"] != str(claim_path):
        _fail("$claim_path", "does not match the registry's frozen claim path")
    split = checked["report"]["split"]
    if split["claim_path"] != str(claim_path) or split["registry_path"] != str(
        registry_path
    ):
        _fail("$publication", "caller paths do not match report split bindings")
    return publish_create_new_json_sequence(
        (
            (claim_path, checked["claim"]),
            (registry_path, checked["registry"]),
            (report_path, checked["report"]),
        ),
        publication_root=publication_root,
    )


class PostReleaseServiceError(attempt_contract.PoweredAttemptContractError):
    """A post-release input cannot support an immutable success record."""


@dataclass(frozen=True)
class StablePostReleaseArtifact:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class StablePostReleasePayload:
    artifact: StablePostReleaseArtifact
    payload: bytes


class PostReleaseReadService(Protocol):
    def identify(self, path: str, *, maximum_bytes: int) -> StablePostReleaseArtifact: ...

    def read(self, path: str, *, maximum_bytes: int) -> StablePostReleasePayload: ...


class StablePostReleaseFileReader:
    """Bounded, no-write reader used only after the live lease is released."""

    @staticmethod
    def identify(path: str, *, maximum_bytes: int) -> StablePostReleaseArtifact:
        from scripts.aigp_vq2_powered_runtime import stable_file_identity

        identity = stable_file_identity(path, max_bytes=maximum_bytes)
        return StablePostReleaseArtifact(
            path=identity.path,
            size_bytes=identity.size_bytes,
            sha256=identity.sha256,
        )

    @classmethod
    def read(cls, path: str, *, maximum_bytes: int) -> StablePostReleasePayload:
        before = cls.identify(path, maximum_bytes=maximum_bytes)
        try:
            payload = Path(path).read_bytes()
        except OSError as exc:
            raise PostReleaseServiceError(f"{path}: stable payload read failed") from exc
        after = cls.identify(path, maximum_bytes=maximum_bytes)
        digest = hashlib.sha256(payload).hexdigest()
        if (
            before != after
            or len(payload) != before.size_bytes
            or digest != before.sha256
        ):
            raise PostReleaseServiceError(f"{path}: identity changed during payload read")
        return StablePostReleasePayload(before, payload)


@dataclass(frozen=True)
class PostReleaseInputs:
    """Explicit immutable sources; construction performs no filesystem access."""

    live_freeze_path: str
    implementation_inventory_path: str
    environment_inventory_path: str
    import_inventory_path: str
    paths: Mapping[str, str]
    supervision_snapshot: Callable[[], Mapping[str, Any]]

    def __post_init__(self) -> None:
        checked_paths = attempt_contract.validate_frozen_paths(dict(self.paths))
        if self.live_freeze_path != checked_paths["live_freeze"]:
            raise ValueError("live_freeze_path must equal the frozen live-freeze path")
        for name in (
            "implementation_inventory_path",
            "environment_inventory_path",
            "import_inventory_path",
        ):
            attempt_contract.validate_absolute_windows_path(
                getattr(self, name), path=f"$postrelease_inputs.{name}"
            )
        if not callable(self.supervision_snapshot):
            raise TypeError("supervision_snapshot must be callable")
        object.__setattr__(self, "paths", MappingProxyType(checked_paths))


@dataclass(frozen=True)
class _PostReleaseDocument:
    value: Mapping[str, Any]
    artifact: StablePostReleaseArtifact


_SUPERVISED_FILE_PROOF_KEYS = {
    "path",
    "final_path",
    "volume_id",
    "file_id",
    "file_state",
    "size_bytes",
    "sha256",
    "regular_file",
    "non_reparse",
    "hardlink_count_one",
    "owner_is_current_user",
    "current_user_only_dacl",
    "stable_before_after",
    "readback_twice_equal",
    "retained_handle",
}
_TREE_EXIT_KEYS = {
    "state",
    "root_process",
    "observations",
    "natural_exit_proved",
    "termination_attempted",
    "termination_returned",
    "termination_is_cleanup_proof",
}


class ProductionPostReleaseService:
    """Ordered pure/offline verifier and record builder for the sole F01/A01 run.

    The constructor is deliberately inert.  Every read occurs in a protocol
    method that the wrapper invokes only after the lease-release occurrence.
    No method opens a socket, mutex, process, or simulator resource, and split
    publication is prospective only: the wrapper remains the sole writer.
    """

    _MAX_JSON_BYTES = 64 * 1024 * 1024
    _MAX_STDERR_BYTES = 1 * 1024 * 1024
    _MAX_LEGACY_BYTES = 512 * 1024 * 1024
    _MAX_REPLAY_FILE_BYTES = 512 * 1024 * 1024
    _MAX_FRAME_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        inputs: PostReleaseInputs,
        *,
        now_ns: Callable[[], int],
        utc_now: Callable[[], str],
        reader: PostReleaseReadService | None = None,
        bundle_loader: Callable[[Path | str], Mapping[str, Any]] | None = None,
        reconciler: Callable[..., Mapping[str, Any]] | None = None,
        safety_validator: Callable[..., Mapping[str, Any]] | None = None,
        prospective_builder: Callable[..., Mapping[str, Any]] | None = None,
        prospective_validator: Callable[[Any], Mapping[str, Any]] | None = None,
        split_publications_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(inputs, PostReleaseInputs):
            raise TypeError("inputs must be PostReleaseInputs")
        if not callable(now_ns) or not callable(utc_now):
            raise TypeError("post-release clock providers must be callable")
        self.inputs = inputs
        self.now_ns = now_ns
        self.utc_now = utc_now
        self.reader = StablePostReleaseFileReader() if reader is None else reader
        if not callable(getattr(self.reader, "identify", None)) or not callable(
            getattr(self.reader, "read", None)
        ):
            raise TypeError("reader does not implement the post-release read API")
        self._bundle_loader = bundle_loader or load_and_verify_sealed_bundle
        self._reconciler = reconciler or reconcile_calibration_records
        self._safety_validator = safety_validator or validate_safety_evidence
        self._prospective_builder = prospective_builder or build_prospective_publications
        self._prospective_validator = prospective_validator or validate_prospective_publications
        self._split_factory = split_publications_factory
        for callback in (
            self._bundle_loader,
            self._reconciler,
            self._safety_validator,
            self._prospective_builder,
            self._prospective_validator,
        ):
            if not callable(callback):
                raise TypeError("post-release pure-function seam is not callable")
        self._stage = "new"
        self._documents: dict[str, _PostReleaseDocument] = {}
        self._supervision: dict[str, Any] | None = None
        self._bundle: dict[str, Any] | None = None
        self._bundle_artifacts: dict[str, StablePostReleaseArtifact] = {}
        self._resource_evidence: dict[str, Any] | None = None
        self._bundle_verification: dict[str, Any] | None = None
        self._capture_seal: dict[str, Any] | None = None
        self._reconciliation: dict[str, Any] | None = None
        self._safety: dict[str, Any] | None = None
        self._analysis_value: dict[str, Any] | None = None
        self._prospective: dict[str, Any] | None = None

    @staticmethod
    def _require_object(value: Any, keys: set[str], path: str) -> dict[str, Any]:
        if type(value) is not dict or set(value) != keys:
            raise PostReleaseServiceError(f"{path}: must be an exact object")
        return value

    @staticmethod
    def _artifact_ref(name: str, artifact: StablePostReleaseArtifact) -> dict[str, Any]:
        return attempt_contract.validate_artifact_ref(
            {
                "name": name,
                "path": artifact.path,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
            },
            root=attempt_contract.EVIDENCE_ROOT,
        )

    @staticmethod
    def _context_value(context: Mapping[str, Any], name: str) -> Any:
        if name not in context:
            raise PostReleaseServiceError(f"$context.{name}: missing")
        return context[name]

    @staticmethod
    def _member(value: Any, name: str, path: str) -> Any:
        if isinstance(value, Mapping):
            if name not in value:
                raise PostReleaseServiceError(f"{path}.{name}: missing")
            return value[name]
        if not hasattr(value, name):
            raise PostReleaseServiceError(f"{path}.{name}: missing")
        return getattr(value, name)

    def _read_identity(self, path: str, maximum_bytes: int) -> StablePostReleaseArtifact:
        try:
            value = self.reader.identify(path, maximum_bytes=maximum_bytes)
        except Exception as exc:
            raise PostReleaseServiceError(f"{path}: stable identity failed: {exc}") from exc
        if not isinstance(value, StablePostReleaseArtifact):
            raise PostReleaseServiceError(f"{path}: reader returned the wrong identity type")
        if (
            value.path != path
            or type(value.size_bytes) is not int
            or value.size_bytes < 0
            or type(value.sha256) is not str
            or len(value.sha256) != 64
            or any(character not in "0123456789abcdef" for character in value.sha256)
        ):
            raise PostReleaseServiceError(f"{path}: reader identity is invalid")
        return value

    def _read_payload(self, path: str, maximum_bytes: int) -> StablePostReleasePayload:
        try:
            value = self.reader.read(path, maximum_bytes=maximum_bytes)
        except Exception as exc:
            raise PostReleaseServiceError(f"{path}: stable read failed: {exc}") from exc
        if not isinstance(value, StablePostReleasePayload):
            raise PostReleaseServiceError(f"{path}: reader returned the wrong payload type")
        artifact = value.artifact
        if (
            artifact.path != path
            or artifact.size_bytes != len(value.payload)
            or artifact.sha256 != hashlib.sha256(value.payload).hexdigest()
        ):
            raise PostReleaseServiceError(f"{path}: payload identity mismatched")
        return value

    def _read_document(
        self,
        name: str,
        path: str,
        validator: Callable[[Any], Mapping[str, Any]],
        *,
        maximum_bytes: int | None = None,
    ) -> _PostReleaseDocument:
        payload = self._read_payload(
            path, self._MAX_JSON_BYTES if maximum_bytes is None else maximum_bytes
        )
        try:
            parsed = attempt_contract.parse_canonical_json_bytes(
                payload.payload, file_form=True
            )
            checked = dict(validator(parsed))
        except Exception as exc:
            raise PostReleaseServiceError(f"{name}: canonical validation failed: {exc}") from exc
        if payload.payload != attempt_contract.canonical_json_file_bytes(checked):
            raise PostReleaseServiceError(f"{name}: validated canonical bytes changed")
        return _PostReleaseDocument(checked, payload.artifact)

    def _phase(self, phase_deadline: Mapping[str, Any], expected: str) -> dict[str, Any]:
        try:
            checked = attempt_contract.validate_phase_deadline(
                phase_deadline, expected_phase=expected
            )
        except Exception as exc:
            raise PostReleaseServiceError(f"${expected}.deadline: {exc}") from exc
        if checked["duration_ns"] != attempt_contract.DEADLINE_DURATIONS_NS[expected]:
            raise PostReleaseServiceError(f"${expected}.duration_ns: frozen duration changed")
        return checked

    def _prepared_timing(
        self, phase_deadline: Mapping[str, Any], expected: str
    ) -> tuple[dict[str, Any], int]:
        deadline = self._phase(phase_deadline, expected)
        prepared = self.now_ns()
        if type(prepared) is not int or not (
            deadline["started_monotonic_ns"]
            <= prepared
            < deadline["deadline_monotonic_ns"]
        ):
            raise PostReleaseServiceError(f"${expected}: preparation missed its deadline")
        timing = attempt_contract.validate_artifact_timing(
            {**deadline, "prepared_monotonic_ns": prepared},
            expected_phase=expected,
        )
        return timing, prepared

    def _finish_unpublished_phase(
        self, phase_deadline: Mapping[str, Any], expected: str
    ) -> None:
        deadline = self._phase(phase_deadline, expected)
        completed = self.now_ns()
        if type(completed) is not int or not (
            deadline["started_monotonic_ns"]
            <= completed
            < deadline["deadline_monotonic_ns"]
        ):
            raise PostReleaseServiceError(f"${expected}: computation missed its deadline")

    def _validate_supervision_snapshot(self) -> dict[str, Any]:
        try:
            raw = attempt_contract.defensive_copy(self.inputs.supervision_snapshot())
        except Exception as exc:
            raise PostReleaseServiceError(f"$supervision_snapshot: {exc}") from exc
        row = self._require_object(
            raw,
            {"stable_files", "process_results", "cleanup_certificates", "tree_exit"},
            "$supervision_snapshot",
        )
        for name in row:
            if type(row[name]) is not dict:
                raise PostReleaseServiceError(f"$supervision_snapshot.{name}: must be an object")
        required_paths = {
            self.inputs.paths["child_authority"],
            self.inputs.paths["child_cleanup_certificate"],
            self.inputs.paths["runner_stdout"],
            self.inputs.paths["runner_stderr"],
        }
        if set(row["stable_files"]) != required_paths:
            raise PostReleaseServiceError(
                "$supervision_snapshot.stable_files: must be the exact no-fallback child set"
            )
        for path, proof in row["stable_files"].items():
            checked = self._require_object(
                proof, _SUPERVISED_FILE_PROOF_KEYS, f"$supervision_snapshot.stable_files[{path!r}]"
            )
            if (
                checked["path"] != path
                or checked["final_path"] != path
                or checked["regular_file"] is not True
                or checked["non_reparse"] is not True
                or checked["hardlink_count_one"] is not True
                or checked["owner_is_current_user"] is not True
                or checked["current_user_only_dacl"] is not True
                or checked["stable_before_after"] is not True
                or checked["readback_twice_equal"] is not True
                or checked["retained_handle"] is not True
            ):
                raise PostReleaseServiceError(
                    f"$supervision_snapshot.stable_files[{path!r}]: proof is not exact"
                )
            attempt_contract.validate_artifact_ref(
                {
                    "name": "supervised_file",
                    "path": path,
                    "size_bytes": checked["size_bytes"],
                    "sha256": checked["sha256"],
                },
                root=attempt_contract.EVIDENCE_ROOT,
            )
        if not (
            len(row["process_results"]) == 1
            and len(row["cleanup_certificates"]) == 1
            and len(row["tree_exit"]) == 1
        ):
            raise PostReleaseServiceError(
                "$supervision_snapshot: successful no-fallback branch requires one child result/certificate/tree proof"
            )
        tree = next(iter(row["tree_exit"].values()))
        tree = self._require_object(tree, _TREE_EXIT_KEYS, "$supervision_snapshot.tree_exit")
        observations = tree["observations"]
        if (
            tree["state"] != "exited"
            or tree["natural_exit_proved"] is not True
            or tree["termination_attempted"] is not False
            or tree["termination_returned"] is not None
            or tree["termination_is_cleanup_proof"] is not False
            or type(observations) is not list
            or not observations
            or observations[-1].get("root_signaled") is not True
            or observations[-1].get("active_pids") != []
        ):
            raise PostReleaseServiceError(
                "$supervision_snapshot.tree_exit: natural empty-tree exit is not proved"
            )
        return row

    def _bind_supervised_artifact(self, artifact: StablePostReleaseArtifact) -> None:
        assert self._supervision is not None
        proof = self._supervision["stable_files"].get(artifact.path)
        if proof is None or (
            proof["size_bytes"] != artifact.size_bytes
            or proof["sha256"] != artifact.sha256
        ):
            raise PostReleaseServiceError(
                f"{artifact.path}: post-release bytes do not bind retained supervision proof"
            )

    def _load_evidence_documents(self) -> None:
        implementation = self._read_document(
            "implementation_inventory",
            self.inputs.implementation_inventory_path,
            attempt_contract.validate_implementation_inventory,
        )
        freeze = self._read_document(
            "live_freeze",
            self.inputs.live_freeze_path,
            lambda value: attempt_contract.validate_live_freeze(
                value, implementation_inventory=implementation.value
            ),
        )
        environment = self._read_document(
            "environment_inventory",
            self.inputs.environment_inventory_path,
            attempt_contract.validate_environment_inventory,
        )
        imports = self._read_document(
            "import_inventory",
            self.inputs.import_inventory_path,
            attempt_contract.validate_import_inventory,
        )
        frozen = freeze.value
        expected_refs = {
            "implementation_inventory": frozen["candidate"]["implementation_inventory"],
            "environment_inventory": frozen["runtime"]["environment_inventory"],
            "import_inventory": frozen["runtime"]["import_inventory"],
        }
        for name, document in (
            ("implementation_inventory", implementation),
            ("environment_inventory", environment),
            ("import_inventory", imports),
        ):
            if expected_refs[name] != {
                "path": document.artifact.path,
                "sha256": document.artifact.sha256,
            }:
                raise PostReleaseServiceError(f"{name}: live-freeze reference mismatched")
        if frozen["paths"] != dict(self.inputs.paths):
            raise PostReleaseServiceError("live freeze paths differ from PostReleaseInputs")

        attempt = self._read_document(
            "attempt_envelope",
            self.inputs.paths["attempt_envelope"],
            lambda value: attempt_contract.validate_attempt(value, live_freeze=frozen),
        )
        authority = self._read_document(
            "child_authority",
            self.inputs.paths["child_authority"],
            lambda value: attempt_contract.validate_process_authority(
                value,
                attempt=attempt.value,
                argv=attempt.value["context"]["child_argv"],
            ),
        )
        certificate = self._read_document(
            "child_cleanup_certificate",
            self.inputs.paths["child_cleanup_certificate"],
            attempt_contract.validate_cleanup_certificate,
        )
        process_result = self._read_document(
            "child_process_result",
            self.inputs.paths["runner_stdout"],
            lambda value: attempt_contract.validate_process_result(
                value, cleanup_certificate=certificate.value
            ),
        )
        prechild = self._read_document(
            "process_prechild",
            self.inputs.paths["process_proof"],
            attempt_contract.validate_simulator_process_proof,
        )
        postchild = self._read_document(
            "process_postchild",
            self.inputs.paths["process_final_proof"],
            attempt_contract.validate_simulator_process_proof,
        )
        training = self._read_document(
            "training_attestation",
            self.inputs.paths["training_attestation"],
            lambda value: attempt_contract.validate_training_attestation(
                value, process_proof=prechild.value
            ),
        )
        lease = self._read_document(
            "lease_final",
            self.inputs.paths["lease_final"],
            validate_powered_live_lease_ledger,
        )
        if prechild.value["phase"] != "prechild" or postchild.value["phase"] != "postchild":
            raise PostReleaseServiceError("simulator process proof phases are not exact")

        documents = {
            "implementation_inventory": implementation,
            "live_freeze": freeze,
            "environment_inventory": environment,
            "import_inventory": imports,
            "attempt_envelope": attempt,
            "child_authority": authority,
            "child_cleanup_certificate": certificate,
            "child_process_result": process_result,
            "process_prechild": prechild,
            "process_postchild": postchild,
            "training_attestation": training,
            "lease_final": lease,
        }
        self._documents = documents
        for name in ("child_authority", "child_cleanup_certificate", "child_process_result"):
            self._bind_supervised_artifact(documents[name].artifact)
        supervised_result = next(iter(self._supervision["process_results"].values()))
        supervised_certificate = next(
            iter(self._supervision["cleanup_certificates"].values())
        )
        tree = next(iter(self._supervision["tree_exit"].values()))
        if (
            supervised_result != process_result.value
            or supervised_certificate != certificate.value
            or tree["root_process"] != authority.value["process"]
        ):
            raise PostReleaseServiceError(
                "supervision result/certificate/tree identity changed after validation"
            )

    def _load_bundle(self) -> None:
        try:
            loaded = dict(self._bundle_loader(self.inputs.paths["replay_bundle"]))
        except Exception as exc:
            raise PostReleaseServiceError(f"replay bundle verification failed: {exc}") from exc
        expected = {
            "manifest",
            "records",
            "manifest_bytes",
            "records_bytes",
            "frame_blob_file_sha256",
            "bundle_path",
        }
        if set(loaded) != expected or loaded["bundle_path"] != self.inputs.paths["replay_bundle"]:
            raise PostReleaseServiceError("replay loader returned a non-exact snapshot")
        if not isinstance(loaded["manifest_bytes"], bytes) or not isinstance(
            loaded["records_bytes"], bytes
        ):
            raise PostReleaseServiceError("replay loader omitted exact canonical bytes")
        manifest_path = ntpath.join(self.inputs.paths["replay_bundle"], "manifest.json")
        records_path = ntpath.join(self.inputs.paths["replay_bundle"], "records.jsonl")
        manifest_artifact = self._read_identity(manifest_path, self._MAX_REPLAY_FILE_BYTES)
        records_artifact = self._read_identity(records_path, self._MAX_REPLAY_FILE_BYTES)
        if (
            manifest_artifact.size_bytes != len(loaded["manifest_bytes"])
            or manifest_artifact.sha256
            != hashlib.sha256(loaded["manifest_bytes"]).hexdigest()
            or records_artifact.size_bytes != len(loaded["records_bytes"])
            or records_artifact.sha256
            != hashlib.sha256(loaded["records_bytes"]).hexdigest()
        ):
            raise PostReleaseServiceError("replay canonical-file identity changed after load")
        artifacts = {
            "replay_manifest": manifest_artifact,
            "replay_records": records_artifact,
        }
        frame_map = loaded["frame_blob_file_sha256"]
        if type(frame_map) is not dict:
            raise PostReleaseServiceError("replay frame identity map is invalid")
        for digest in sorted(frame_map, key=lambda item: item.encode("utf-8")):
            path = ntpath.join(
                self.inputs.paths["replay_bundle"], "frames", f"{digest}.npy"
            )
            artifact = self._read_identity(path, self._MAX_FRAME_BYTES)
            if artifact.sha256 != frame_map[digest]:
                raise PostReleaseServiceError(f"replay frame {digest} changed")
            artifacts[f"replay_frame/{digest}"] = artifact
        self._bundle = loaded
        self._bundle_artifacts = artifacts

    def _verify_manifest_outcome(self) -> dict[str, Any]:
        assert self._bundle is not None
        manifest = self._bundle["manifest"]
        outcome = self._require_object(
            manifest.get("outcome"),
            {
                "powered_stage_completed",
                "cleanup_certificate_outcome",
                "reason_codes",
                "vision_capture_stats",
                "powered_capture_resource_stats",
            },
            "$manifest.outcome",
        )
        if (
            outcome["powered_stage_completed"] is not True
            or outcome["cleanup_certificate_outcome"] != "proved"
            or outcome["reason_codes"] != []
        ):
            raise PostReleaseServiceError("$manifest.outcome: child outcome is not successful")
        resource = derive_powered_capture_resource_evidence(
            outcome["powered_capture_resource_stats"]
        )
        raw_vision = resource["raw"]["vision"]
        if outcome["vision_capture_stats"] != {
            name: value for name, value in raw_vision.items() if name != "constructed"
        }:
            raise PostReleaseServiceError("manifest vision stats do not bind raw stats")
        if resource["resource_stats_zero"] is not True:
            raise PostReleaseServiceError("sealed raw resource stats are not loss-free")
        if raw_vision["timing_ledger_entries"] != 0:
            raise PostReleaseServiceError("vision timing ledger remained buffered")
        for group, high, capacity in (
            (raw_vision, "timing_ledger_high_watermark", "timing_ledger_capacity"),
            (raw_vision, "receiver_buffer_high_watermark", "receiver_buffer_capacity"),
            (raw_vision, "capture_snapshot_queue_high_watermark", "capture_snapshot_queue_capacity"),
            (resource["raw"]["collision"], "high_watermark", "capacity"),
            (resource["raw"]["outbound_receipts"], "high_watermark", "capacity"),
        ):
            if group[capacity] <= 0 or group[high] > group[capacity]:
                raise PostReleaseServiceError("raw resource high-water/capacity relation failed")
        recorder = resource["raw"]["recorder"]
        if (
            recorder["written"] > recorder["enqueued"]
            or recorder["decoded_frames_written"]
            > recorder["decoded_frames_enqueued"]
        ):
            raise PostReleaseServiceError("pre-close recorder counters are inconsistent")
        self._resource_evidence = resource
        return resource

    def _bundle_checks(self) -> dict[str, bool]:
        assert self._bundle is not None and self._resource_evidence is not None
        manifest = self._bundle["manifest"]
        records = self._bundle["records"]
        decoded = [row for row in records if row.get("type") == "decoded_frame"]
        frame_rows = [
            row for row in records if row.get("type") in {"frame", "decoded_frame"}
        ]
        timing_rows = [
            row
            for row in records
            if row.get("type") == "event" and row.get("event") == "camera_frame_timing"
        ]
        timing_exact = len(timing_rows) == len(decoded)
        decoded_by_sequence = {row["sequence"]: row for row in decoded}
        for row in timing_rows:
            linked = decoded_by_sequence.get(row["linked_decoded_frame_record_sequence"])
            timing = row["observation"]
            identity = timing["identity"]
            if linked is None or (
                identity["generation"],
                identity["frame_id"],
                timing["camera_source_time_ns"],
            ) != (linked["generation"], linked["frame_id"], linked["sim_time_ns"]):
                timing_exact = False
        sequenced = [
            row["observation"]
            for row in records
            if row.get("type") == "event"
            and row.get("event") in _SEQUENCED_OBSERVATION_EVENTS
        ]
        resource = self._resource_evidence["raw"]
        boundary_resource = _reset_boundary_resource_evidence(
            records, final_resource_stats=resource
        )
        outbound_resource = _outbound_resource_evidence(
            records, final_resource_stats=resource
        )
        outbound_count = sum(
            row.get("type") == "event"
            and row.get("event") in {"attitude_target_outbound", "nonattitude_outbound"}
            for row in records
        )
        resource_bound = (
            resource["recorder"]["enqueued"] == len(records)
            and resource["recorder"]["decoded_frames_enqueued"] == len(decoded)
            and resource["snapshot_capture"]["observed_frames"] == len(decoded)
            and resource["vision"]["frames_decoded"] == len(decoded)
            and resource["outbound_receipts"]["returned"] == outbound_count
            and outbound_resource["exact"] is True
            and boundary_resource["required_occurrences_exact"] is True
        )
        process_result = self._documents["child_process_result"].value
        replay = process_result["artifacts"]["replay_bundle"]
        tree = next(iter(self._supervision["tree_exit"].values()))
        writer_closed = (
            manifest["complete"] is True
            and process_result["outcome"] == "completed"
            and replay["state"] == "closed"
            and replay["dataset_hash"] == manifest["dataset_hash"]
            and replay["manifest_sha256"]
            == self._bundle_artifacts["replay_manifest"].sha256
            and replay["records_sha256"]
            == self._bundle_artifacts["replay_records"].sha256
            and tree["state"] == "exited"
            and tree["natural_exit_proved"] is True
        )
        checks = {
            "manifest_schema_valid": manifest["schema"] == REPLAY_BUNDLE_SCHEMA,
            "records_schema_valid": all(
                row.get("schema") == REPLAY_RECORD_SCHEMA for row in records
            ),
            "dataset_hash_valid": type(manifest["dataset_hash"]) is str,
            "records_complete": (
                manifest["complete"] is True
                and manifest["record_count"] == len(records)
                and [row["sequence"] for row in records] == list(range(len(records)))
            ),
            "frame_blob_set_exact": set(self._bundle["frame_blob_file_sha256"])
            == set(manifest["integrity"]["frame_blob_file_sha256"]),
            "frame_blob_hashes_valid": self._bundle["frame_blob_file_sha256"]
            == manifest["integrity"]["frame_blob_file_sha256"],
            "decoded_frame_shape_valid": bool(decoded)
            and all(
                row["image_shape"] == [360, 640, 3]
                and row["image_dtype"] == "|u1"
                for row in frame_rows
            ),
            "camera_timing_links_exact": timing_exact,
            "observation_schemas_valid": True,
            "event_sequences_contiguous": [
                item["event_sequence"] for item in sequenced
            ]
            == list(range(len(sequenced))),
            "resource_stats_zero": self._resource_evidence["resource_stats_zero"]
            is True
            and boundary_resource["resource_stats_zero"] is True
            and resource_bound,
            "writer_closed": writer_closed,
        }
        if not all(checks.values()):
            failed = sorted(name for name, value in checks.items() if not value)
            raise PostReleaseServiceError(
                f"replay bundle checks failed: {','.join(failed)}"
            )
        return checks

    def verify_bundle(
        self, *, phase_deadline: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self._stage != "new":
            raise PostReleaseServiceError("verify_bundle is out of order")
        deadline = self._phase(phase_deadline, "bundle_verify")
        self._supervision = self._validate_supervision_snapshot()
        self._load_evidence_documents()
        stderr = self._read_identity(
            self.inputs.paths["runner_stderr"], self._MAX_STDERR_BYTES
        )
        self._bind_supervised_artifact(stderr)
        legacy = self._read_identity(
            self.inputs.paths["legacy_record"], self._MAX_LEGACY_BYTES
        )
        process_result = self._documents["child_process_result"].value
        if process_result["artifacts"]["legacy_record"] != {
            "path": legacy.path,
            "state": "closed",
            "sha256": legacy.sha256,
        }:
            raise PostReleaseServiceError("legacy artifact does not bind child result")
        self._bundle_artifacts["runner_stderr"] = stderr
        self._bundle_artifacts["legacy_record"] = legacy
        self._load_bundle()
        # _load_bundle replaces the mapping; restore non-bundle identities.
        self._bundle_artifacts["runner_stderr"] = stderr
        self._bundle_artifacts["legacy_record"] = legacy
        self._verify_manifest_outcome()
        checks = self._bundle_checks()
        prepared = self.now_ns()
        if type(prepared) is not int or not (
            deadline["started_monotonic_ns"] <= prepared < deadline["deadline_monotonic_ns"]
        ):
            raise PostReleaseServiceError("bundle verification missed its deadline")
        timing = attempt_contract.validate_artifact_timing(
            {**deadline, "prepared_monotonic_ns": prepared},
            expected_phase="bundle_verify",
        )
        freeze = self._documents["live_freeze"]
        attempt = self._documents["attempt_envelope"]
        frames = [
            self._artifact_ref(name, artifact)
            for name, artifact in sorted(
                self._bundle_artifacts.items(), key=lambda item: item[0].encode("utf-8")
            )
            if name.startswith("replay_frame/")
        ]
        value = {
            "schema": "aigp-vq2-replay-bundle-verification/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "verified_at_utc": self.utc_now(),
            "verified_monotonic_ns": prepared,
            "timing": timing,
            "identity": {
                "candidate_commit": freeze.value["candidate"]["commit"],
                "live_freeze_sha256": freeze.artifact.sha256,
                "attempt_context_sha256": attempt.value["context_sha256"],
                "attempt_envelope_sha256": attempt.artifact.sha256,
                "child_authority_sha256": self._documents["child_authority"].artifact.sha256,
                "child_process_result_sha256": self._documents["child_process_result"].artifact.sha256,
                "child_cleanup_certificate_sha256": self._documents["child_cleanup_certificate"].artifact.sha256,
                "lease_final_sha256": self._documents["lease_final"].artifact.sha256,
            },
            "bundle": {
                "path": self.inputs.paths["replay_bundle"],
                "dataset_hash": self._bundle["manifest"]["dataset_hash"],
                "manifest": self._artifact_ref(
                    "replay_manifest", self._bundle_artifacts["replay_manifest"]
                ),
                "records": self._artifact_ref(
                    "replay_records", self._bundle_artifacts["replay_records"]
                ),
                "frames": frames,
            },
            "checks": checks,
            "valid": True,
        }
        self._bundle_verification = attempt_contract.validate_bundle_verification(value)
        self._stage = "bundle_verified"
        return _clone(self._bundle_verification)

    def _refresh_artifact(self, artifact: StablePostReleaseArtifact, maximum: int) -> None:
        if self._read_identity(artifact.path, maximum) != artifact:
            raise PostReleaseServiceError(f"{artifact.path}: artifact changed before seal")

    def _capture_counts(self) -> dict[str, int]:
        assert self._bundle is not None and self._resource_evidence is not None
        records = self._bundle["records"]
        event_counts = {
            name: sum(
                row.get("type") == "event" and row.get("event") == event
                for row in records
            )
            for name, event in {
                "camera_timing_records": "camera_frame_timing",
                "mavlink_ingress_records": "mavlink_ingress",
                "race_records": "received_race_status",
                "heartbeat_records": "received_heartbeat",
                "actuator_records": "received_actuator_output_status",
                "collision_records": "runner_collision_observation",
                "generated_commands": "calibration_command_generated",
                "sent_commands": "calibration_command_sent",
                "not_sent_commands": "calibration_command_not_sent",
                "tick_dispositions": "calibration_tick_disposition",
            }.items()
        }
        boundary_resource = _reset_boundary_resource_evidence(
            records, final_resource_stats=self._resource_evidence["raw"]
        )
        aggregate_loss = {
            name: self._resource_evidence["loss"][name]
            + boundary_resource["loss"][name]
            for name in self._resource_evidence["loss"]
        }
        return {
            "record_count": len(records),
            "decoded_frames": sum(row.get("type") == "decoded_frame" for row in records),
            "frame_blobs": len(self._bundle["frame_blob_file_sha256"]),
            "imu_records": sum(row.get("type") == "imu" for row in records),
            **event_counts,
            **aggregate_loss,
        }

    def build_capture_seal(
        self, *, phase_deadline: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self._stage != "bundle_verified" or self._bundle_verification is None:
            raise PostReleaseServiceError("build_capture_seal is out of order")
        verification = self._read_document(
            "bundle_verification",
            self.inputs.paths["bundle_verification"],
            attempt_contract.validate_bundle_verification,
        )
        if verification.value != self._bundle_verification:
            raise PostReleaseServiceError("published bundle verification changed")
        for document in self._documents.values():
            self._refresh_artifact(document.artifact, self._MAX_JSON_BYTES)
        self._refresh_artifact(self._bundle_artifacts["runner_stderr"], self._MAX_STDERR_BYTES)
        self._refresh_artifact(self._bundle_artifacts["legacy_record"], self._MAX_LEGACY_BYTES)
        for name, artifact in self._bundle_artifacts.items():
            if name.startswith("replay_"):
                maximum = self._MAX_FRAME_BYTES if name.startswith("replay_frame/") else self._MAX_REPLAY_FILE_BYTES
                self._refresh_artifact(artifact, maximum)
        timing, _prepared = self._prepared_timing(phase_deadline, "capture_seal")
        freeze = self._documents["live_freeze"]
        attempt = self._documents["attempt_envelope"]
        named = {
            "live_freeze": freeze.artifact,
            "implementation_inventory": self._documents["implementation_inventory"].artifact,
            "environment_inventory": self._documents["environment_inventory"].artifact,
            "import_inventory": self._documents["import_inventory"].artifact,
            "attempt_envelope": attempt.artifact,
            "training_attestation": self._documents["training_attestation"].artifact,
            "process_prechild": self._documents["process_prechild"].artifact,
            "process_postchild": self._documents["process_postchild"].artifact,
            "child_authority": self._documents["child_authority"].artifact,
            "child_cleanup_certificate": self._documents["child_cleanup_certificate"].artifact,
            "lease_final": self._documents["lease_final"].artifact,
            "bundle_verification": verification.artifact,
            "runner_stdout": self._documents["child_process_result"].artifact,
            "runner_stderr": self._bundle_artifacts["runner_stderr"],
            "legacy_record": self._bundle_artifacts["legacy_record"],
            **{
                name: artifact
                for name, artifact in self._bundle_artifacts.items()
                if name.startswith("replay_")
            },
        }
        artifacts = [
            self._artifact_ref(name, artifact)
            for name, artifact in sorted(named.items(), key=lambda item: item[0].encode("utf-8"))
        ]
        child_hash = self._documents["child_cleanup_certificate"].artifact.sha256
        cleanup = {
            "child_certificate_sha256": child_hash,
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
        value = {
            "schema": "aigp-vq2-powered-calibration-capture-seal/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "sealed_at_utc": self.utc_now(),
            "timing": timing,
            "identity": {
                "candidate_commit": freeze.value["candidate"]["commit"],
                "code_sha256": freeze.value["candidate"]["code_sha256"],
                "live_freeze_sha256": freeze.artifact.sha256,
                "attempt_context_sha256": attempt.value["context_sha256"],
                "attempt_envelope_sha256": attempt.artifact.sha256,
                "target_config_sha256": attempt.value["context"]["target_config"]["sha256"],
                "capture_authorization_sha256": attempt.value["context"]["capture_authorization"]["sha256"],
                "excitation_plan_id": attempt_contract.EXCITATION_PLAN_ID,
                "excitation_plan_sha256": attempt_contract.EXCITATION_PLAN_SHA256,
                "training_attestation_sha256": self._documents["training_attestation"].artifact.sha256,
                "simulator_process_proof_sha256": self._documents["process_prechild"].artifact.sha256,
                "simulator_final_process_proof_sha256": self._documents["process_postchild"].artifact.sha256,
                "child_authority_sha256": self._documents["child_authority"].artifact.sha256,
                "cleanup_authority_sha256": None,
                "lease_final_sha256": self._documents["lease_final"].artifact.sha256,
                "bundle_verification_sha256": verification.artifact.sha256,
            },
            "artifacts": artifacts,
            "capture_stats": self._capture_counts(),
            "outbound_audit": _clone(
                self._documents["child_process_result"].value["outbound_audit"]
            ),
            "cleanup": cleanup,
        }
        self._capture_seal = attempt_contract.validate_capture_seal(value)
        self._documents["bundle_verification"] = verification
        self._stage = "sealed"
        return _clone(self._capture_seal)

    def analyze_capture(
        self, *, phase_deadline: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self._stage != "sealed" or self._capture_seal is None or self._bundle is None:
            raise PostReleaseServiceError("analyze_capture is out of order")
        seal = self._read_document(
            "capture_seal",
            self.inputs.paths["capture_seal"],
            attempt_contract.validate_capture_seal,
        )
        if seal.value != self._capture_seal:
            raise PostReleaseServiceError("published capture seal changed")
        try:
            reconciliation = dict(
                self._reconciler(
                    self._bundle["manifest"],
                    self._bundle["records"],
                    manifest_bytes=self._bundle["manifest_bytes"],
                    records_bytes=self._bundle["records_bytes"],
                    frame_blob_file_sha256=self._bundle["frame_blob_file_sha256"],
                )
            )
            safety = dict(
                self._safety_validator(
                    reconciliation,
                    live_freeze=self._documents["live_freeze"].value,
                    attempt_envelope=self._documents["attempt_envelope"].value,
                    bundle_verification=self._bundle_verification,
                    capture_seal=self._capture_seal,
                    child_authority=self._documents["child_authority"].value,
                    child_process_result=self._documents["child_process_result"].value,
                    child_cleanup_certificate=self._documents["child_cleanup_certificate"].value,
                    lease_final=self._documents["lease_final"].value,
                    training_attestation=self._documents["training_attestation"].value,
                    process_prechild=self._documents["process_prechild"].value,
                    process_postchild=self._documents["process_postchild"].value,
                )
            )
        except Exception as exc:
            raise PostReleaseServiceError(f"offline capture analysis failed: {exc}") from exc
        if reconciliation.get("valid") is not True or safety.get("valid") is not True:
            raise PostReleaseServiceError("offline capture analysis did not pass every check")
        self._finish_unpublished_phase(phase_deadline, "analysis")
        self._reconciliation = reconciliation
        self._safety = safety
        self._analysis_value = {
            "reconciliation": _clone(reconciliation),
            "safety_evidence": _clone(safety),
        }
        self._documents["capture_seal"] = seal
        self._stage = "analyzed"
        return _clone(self._analysis_value)

    def publish_split(
        self, *, analysis: Any, phase_deadline: Mapping[str, Any]
    ) -> Any:
        if self._stage != "analyzed" or self._analysis_value is None:
            raise PostReleaseServiceError("publish_split is out of order")
        if _clone(analysis) != self._analysis_value:
            raise PostReleaseServiceError("publish_split analysis input changed")
        deadline = self._phase(phase_deadline, "split_publish")

        def occurrence(label: str) -> tuple[str, int]:
            value = self.now_ns()
            if type(value) is not int or not (
                deadline["started_monotonic_ns"] <= value < deadline["deadline_monotonic_ns"]
            ):
                raise PostReleaseServiceError(f"split {label} missed its deadline")
            return self.utc_now(), value

        claimed_utc, claimed = occurrence("claim preparation")
        published_utc, published = occurrence("registry preparation")
        generated_utc, report_prepared = occurrence("report preparation")
        if not claimed <= published <= report_prepared:
            raise PostReleaseServiceError("split preparation clock regressed")
        try:
            prospective = dict(
                self._prospective_builder(
                    self._reconciliation,
                    self._safety,
                    split_phase_deadline=deadline,
                    claim_path=self.inputs.paths["split_claim"],
                    registry_path=self.inputs.paths["split_registry"],
                    claimed_at_utc=claimed_utc,
                    claimed_monotonic_ns=claimed,
                    published_at_utc=published_utc,
                    published_monotonic_ns=published,
                    generated_at_utc=generated_utc,
                    report_prepared_monotonic_ns=report_prepared,
                )
            )
            prospective = dict(self._prospective_validator(prospective))
        except Exception as exc:
            raise PostReleaseServiceError(f"prospective split construction failed: {exc}") from exc
        self._prospective = prospective
        self._stage = "published"
        factory = self._split_factory
        if factory is None:
            # Deliberately lazy: importing this seeded module cannot import the
            # wrapper module before its exact six-seed inventory is revalidated.
            from scripts.aigp_vq2_powered_calibration_probe import SplitPublications

            factory = SplitPublications
        return factory(
            claim=_clone(prospective["claim"]),
            registry=_clone(prospective["registry"]),
            report=_clone(prospective["report"]),
        )

    @staticmethod
    def _receipt_identity(value: Any, path: str) -> tuple[str, str, int]:
        receipt_path = ProductionPostReleaseService._member(value, "path", path)
        digest = ProductionPostReleaseService._member(value, "sha256", path)
        size = ProductionPostReleaseService._member(value, "size_bytes", path)
        if type(receipt_path) is not str or type(digest) is not str or type(size) is not int:
            raise PostReleaseServiceError(f"{path}: receipt identity is invalid")
        return receipt_path, digest, size

    def _require_complete_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if self._stage != "published" or self._prospective is None:
            raise PostReleaseServiceError("complete terminal requested before split construction")
        if not isinstance(context, Mapping):
            raise PostReleaseServiceError("$context: must be a mapping")
        if self._context_value(context, "reason_codes") != ():
            raise PostReleaseServiceError("complete terminal cannot carry reasons")
        if self._context_value(context, "fallback_used") is not False:
            raise PostReleaseServiceError("complete terminal forbids fallback")
        if self._context_value(context, "lease_release_proved") is not True:
            raise PostReleaseServiceError("complete terminal requires proved lease release")
        if self._context_value(context, "wrapper_alive") is not True:
            raise PostReleaseServiceError("complete terminal requires a live wrapper")
        if dict(self._context_value(context, "cleanup_state")) != {
            "child_exit": "proved",
            "fallback": "not_required",
            "ports": "free",
            "lease": "released",
            "processes": "exited",
            "transport": "closed",
            "scheduled_task": "absent",
            "simulator_topology": "unchanged",
            "simulator_responsive": "yes",
        }:
            raise PostReleaseServiceError("complete terminal cleanup tuple is not exact")
        if dict(self._context_value(context, "bundle")) != self._bundle_verification:
            raise PostReleaseServiceError("complete terminal bundle value changed")
        if dict(self._context_value(context, "seal")) != self._capture_seal:
            raise PostReleaseServiceError("complete terminal seal value changed")
        split = self._context_value(context, "split")
        for name in ("claim", "registry", "report"):
            if _clone(self._member(split, name, "$context.split")) != self._prospective[name]:
                raise PostReleaseServiceError(f"complete terminal split {name} changed")
        return dict(context)

    def build_complete_terminal(self, *, context: Mapping[str, Any]) -> Mapping[str, Any]:
        context = self._require_complete_context(context)
        lifecycle = attempt_contract.validate_wrapper_lifecycle(context["lifecycle"])
        lifecycle_hash = attempt_contract.canonical_file_sha256(lifecycle)
        lifecycle_receipt = context.get("lifecycle_receipt")
        if lifecycle_receipt is not None:
            path, digest, _size = self._receipt_identity(
                lifecycle_receipt, "$context.lifecycle_receipt"
            )
            if path != self.inputs.paths["wrapper_lifecycle"] or digest != lifecycle_hash:
                raise PostReleaseServiceError("wrapper lifecycle receipt changed")
        artifacts = context["artifacts"]
        expected_publications = {
            "bundle_verification": (
                self.inputs.paths["bundle_verification"],
                self._documents["bundle_verification"].artifact.sha256,
            ),
            "capture_seal": (
                self.inputs.paths["capture_seal"],
                self._documents["capture_seal"].artifact.sha256,
            ),
            "split_claim": (
                self.inputs.paths["split_claim"], self._prospective["hashes"]["claim"]
            ),
            "split_registry": (
                self.inputs.paths["split_registry"], self._prospective["hashes"]["registry"]
            ),
            "analysis_report": (
                self.inputs.paths["analysis_report"], self._prospective["hashes"]["report"]
            ),
        }
        for name, (expected_path, expected_hash) in expected_publications.items():
            if name not in artifacts:
                raise PostReleaseServiceError(f"$context.artifacts.{name}: missing")
            path, digest, _size = self._receipt_identity(
                artifacts[name], f"$context.artifacts.{name}"
            )
            if path != expected_path or digest != expected_hash:
                raise PostReleaseServiceError(f"$context.artifacts.{name}: identity changed")
        timing = attempt_contract.validate_terminal_publication_timing(
            context["publication_timing"], expected_phase="terminal_publish"
        )
        completed = context["completed_monotonic_ns"]
        if timing["prepared_monotonic_ns"] != completed:
            raise PostReleaseServiceError("complete terminal timing changed")
        freeze = self._documents["live_freeze"]
        attempt = self._documents["attempt_envelope"]
        seal = self._capture_seal
        refs = {item["name"]: item for item in seal["artifacts"]}
        hashes = {
            "bundle_dataset_hash": self._bundle["manifest"]["dataset_hash"],
            "bundle_verification_sha256": self._documents["bundle_verification"].artifact.sha256,
            "capture_seal_sha256": self._documents["capture_seal"].artifact.sha256,
            "analysis_report_sha256": self._prospective["hashes"]["report"],
            "split_claim_sha256": self._prospective["hashes"]["claim"],
            "split_registry_sha256": self._prospective["hashes"]["registry"],
            "bundle_manifest_sha256": refs["replay_manifest"]["sha256"],
            "bundle_records_sha256": refs["replay_records"]["sha256"],
            "legacy_record_sha256": refs["legacy_record"]["sha256"],
            "runner_stdout_sha256": refs["runner_stdout"]["sha256"],
            "runner_stderr_sha256": refs["runner_stderr"]["sha256"],
            "lease_final_sha256": refs["lease_final"]["sha256"],
            "training_attestation_sha256": refs["training_attestation"]["sha256"],
            "simulator_process_proof_sha256": refs["process_prechild"]["sha256"],
            "simulator_final_process_proof_sha256": refs["process_postchild"]["sha256"],
            "implementation_inventory_sha256": refs["implementation_inventory"]["sha256"],
            "environment_inventory_sha256": refs["environment_inventory"]["sha256"],
            "import_inventory_sha256": refs["import_inventory"]["sha256"],
            "child_authority_sha256": refs["child_authority"]["sha256"],
            "cleanup_authority_sha256": None,
            "child_cleanup_certificate_sha256": refs["child_cleanup_certificate"]["sha256"],
            "fallback_cleanup_certificate_sha256": None,
            "cleanup_stdout_sha256": None,
            "cleanup_stderr_sha256": None,
            "wrapper_lifecycle_sha256": lifecycle_hash,
        }
        value = {
            "schema": "aigp-vq2-powered-calibration-attempt-complete/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "completed_at_utc": context["utc"],
            "completed_monotonic_ns": completed,
            "deadline_monotonic_ns": timing["deadline_monotonic_ns"],
            "publication_timing": timing,
            "identity": {
                "candidate_commit": freeze.value["candidate"]["commit"],
                "code_sha256": freeze.value["candidate"]["code_sha256"],
                "live_freeze_sha256": freeze.artifact.sha256,
                "attempt_context_sha256": attempt.value["context_sha256"],
                "attempt_envelope_sha256": attempt.artifact.sha256,
                "target_config_sha256": attempt.value["context"]["target_config"]["sha256"],
                "capture_authorization_sha256": attempt.value["context"]["capture_authorization"]["sha256"],
                "excitation_plan_id": attempt_contract.EXCITATION_PLAN_ID,
                "excitation_plan_sha256": attempt_contract.EXCITATION_PLAN_SHA256,
                "wrapper_lifecycle_sha256": lifecycle_hash,
            },
            "artifact_hashes": hashes,
            "cleanup": _clone(seal["cleanup"]),
        }
        return attempt_contract.validate_attempt_complete(
            value, wrapper_lifecycle=lifecycle
        )

    def _attempt_identity_from_context(
        self, context: Mapping[str, Any]
    ) -> tuple[dict[str, Any], Any, Any]:
        admission = self._context_value(context, "admission")
        freeze = attempt_contract.validate_live_freeze(
            self._member(admission, "live_freeze", "$context.admission")
        )
        freeze_hash = self._member(
            admission, "live_freeze_sha256", "$context.admission"
        )
        material = self._context_value(context, "material")
        receipt = self._context_value(context, "attempt_receipt")
        return {"freeze": freeze, "freeze_hash": freeze_hash}, material, receipt

    def build_live_poison(self, *, context: Mapping[str, Any]) -> Mapping[str, Any]:
        identity, material, receipt = self._attempt_identity_from_context(context)
        timing = attempt_contract.validate_artifact_timing(
            self._context_value(context, "publication_timing"),
            expected_phase="poison_publish",
        )
        created = self._context_value(context, "created_monotonic_ns")
        if timing["prepared_monotonic_ns"] != created:
            raise PostReleaseServiceError("poison timing changed")
        reasons = tuple(self._context_value(context, "reason_codes"))
        if not reasons:
            raise PostReleaseServiceError("poison requires an invalidation reason")
        envelope = None if material is None else self._member(material, "envelope", "$context.material")
        valid_attempt = receipt is not None and envelope is not None
        artifact_state = dict(self._context_value(context, "artifact_state"))
        cleanup = attempt_contract.validate_invalid_cleanup_state(
            self._context_value(context, "cleanup_state")
        )
        publication = {
            "bundle_verification": artifact_state["bundle_verification"],
            "capture_seal": artifact_state["capture_seal"],
            "claim": artifact_state["split_claim"],
            "registry": artifact_state["split_registry"],
            "report": artifact_state["analysis_report"],
            "wrapper_lifecycle": artifact_state["wrapper_lifecycle"],
            "attempt_complete": artifact_state["attempt_complete"],
            "terminal": (
                "partial_complete"
                if artifact_state["attempt_complete"] == "partial"
                else "missing"
            ),
        }
        value = {
            "schema": "aigp-vq2-powered-calibration-live-poison/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "created_at_utc": context["utc"],
            "created_monotonic_ns": created,
            "publication_timing": timing,
            "phase": context["phase"],
            "reason_codes": sorted(set(reasons), key=lambda item: item.encode("utf-8")),
            "attempt_context_sha256": envelope["context_sha256"] if valid_attempt else None,
            "attempt_envelope_sha256": self._member(receipt, "sha256", "$context.attempt_receipt") if valid_attempt else None,
            "wrapper_process": envelope["context"]["wrapper_process"] if envelope is not None else None,
            "child_process": context["child_process"],
            "cleanup_process": context["cleanup_process"],
            "lease_state": {
                "phase": context["phase"],
                "owner_token_sha256": (
                    envelope["capabilities"]["lease_owner_sha256"]
                    if envelope is not None and context["lease_acquired"]
                    else None
                ),
                "release_proved": context["lease_release_proved"],
            },
            "port_state": {
                "mavlink_14550": cleanup["ports"],
                "camera_5600": cleanup["ports"],
            },
            "process_state": cleanup["processes"],
            "transport_state": cleanup["transport"],
            "scheduled_task_state": cleanup["scheduled_task"],
            "publication_state": publication,
            "simulator_state": {
                "topology": cleanup["simulator_topology"],
                "responsive": cleanup["simulator_responsive"],
            },
            "required_action": "new_reviewed_recovery_task_no_automatic_clear",
        }
        return attempt_contract.validate_live_poison(value)

    def build_invalid_terminal(self, *, context: Mapping[str, Any]) -> Mapping[str, Any]:
        identity, material, receipt = self._attempt_identity_from_context(context)
        timing = attempt_contract.validate_terminal_publication_timing(
            self._context_value(context, "publication_timing"),
            expected_phase="invalid_terminal_publish",
        )
        invalidated = self._context_value(context, "invalidated_monotonic_ns")
        if timing["prepared_monotonic_ns"] != invalidated:
            raise PostReleaseServiceError("invalid terminal timing changed")
        reasons = sorted(
            set(self._context_value(context, "reason_codes")),
            key=lambda item: item.encode("utf-8"),
        )
        if not reasons:
            raise PostReleaseServiceError("invalid terminal requires a reason")
        envelope = None if material is None else self._member(material, "envelope", "$context.material")
        valid_attempt = receipt is not None and envelope is not None
        poison_required = self._context_value(context, "poison_required")
        if type(poison_required) is not bool:
            raise PostReleaseServiceError("poison_required must be an exact boolean")
        poison_receipt = self._context_value(context, "poison_receipt")
        poison_hash = None
        if poison_receipt is not None:
            poison_path, poison_hash, _size = self._receipt_identity(
                poison_receipt, "$context.poison_receipt"
            )
            if poison_path != identity["freeze"]["paths"]["live_poison"]:
                raise PostReleaseServiceError("poison receipt path changed")
        freeze = identity["freeze"]
        value = {
            "schema": "aigp-vq2-powered-calibration-attempt-invalid/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "invalidated_at_utc": context["utc"],
            "invalidated_monotonic_ns": invalidated,
            "publication_timing": timing,
            "phase": context["phase"],
            "reason_codes": reasons,
            "reason_detail": context["reason_detail"],
            "identity": {
                "attempt_envelope_state": "valid" if valid_attempt else "partial",
                "live_freeze_sha256": identity["freeze_hash"],
                "attempt_context_sha256": envelope["context_sha256"] if valid_attempt else None,
                "attempt_envelope_sha256": self._member(receipt, "sha256", "$context.attempt_receipt") if valid_attempt else None,
                "candidate_commit": freeze["candidate"]["commit"],
                "target_config_sha256": freeze["inputs"]["target_config"]["sha256"],
                "capture_authorization_sha256": freeze["inputs"]["capture_authorization"]["sha256"],
                "excitation_plan_sha256": attempt_contract.EXCITATION_PLAN_SHA256,
            },
            "artifact_state": _clone(context["artifact_state"]),
            "cleanup_state": _clone(context["cleanup_state"]),
            "poison": {
                "required": poison_required,
                "path": freeze["paths"]["live_poison"],
                "sha256": poison_hash if poison_required else None,
            },
        }
        return attempt_contract.validate_attempt_invalid(value)


__all__ = [
    "PartialPublicationError",
    "PostReleaseInputs",
    "PostReleaseReadService",
    "PostReleaseServiceError",
    "ProductionPostReleaseService",
    "PoweredCalibrationAnalysisError",
    "SEMANTIC_AMBIGUITIES",
    "StablePostReleaseArtifact",
    "StablePostReleaseFileReader",
    "StablePostReleasePayload",
    "build_prospective_publications",
    "derive_powered_capture_resource_evidence",
    "load_and_verify_sealed_bundle",
    "plan_global_content_assignment",
    "publish_create_new_json_sequence",
    "publish_prospective_publications",
    "reconcile_calibration_records",
    "semantic_ambiguities",
    "validate_global_content_assignment",
    "validate_powered_capture_resource_stats",
    "validate_prospective_publications",
    "validate_replay_snapshot",
    "validate_safety_evidence",
]
