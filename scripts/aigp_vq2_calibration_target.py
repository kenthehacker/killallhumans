"""Strict offline admission for the VQ2 calibration collection protocol.

This module validates one explicitly supplied nominal-target configuration and
one explicitly supplied simulation-capture authorization.  It computes their
identities from the exact, stably read bytes.  It has no default input path,
field override, simulator, network, subprocess, asset, or write-back path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


TARGET_CONFIG_SCHEMA = "aigp-vq2-sim-calibration-collection-config/1"
CAPTURE_AUTHORIZATION_SCHEMA = "aigp-vq2-simulation-capture-authorization/1"
MAX_JSON_BYTES = 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_SIMULATOR = {"build": 3385, "mode": "Training"}
_SOURCE = {
    "kind": "public_technical_spec",
    "document_id": "VADR-TS-002",
    "issue": "00.02",
    "publication_date": "2026-05-08",
    "url": (
        "https://www.theaigrandprix.com/wp-content/uploads/2026/05/"
        "260508_Technical_Spec_0002.pdf"
    ),
    "stated_scope": "Virtual Qualifier 1",
    "use_scope": "nominal_geometry_only",
}
_APPLICABILITY = {
    "status": "nominal_unverified_for_build_3385_training",
    "result_semantics": "conditional_on_nominal_gate_config",
    "replacement_policy": "new_config_id_revision_and_hash",
}
_TARGET_FRAME = {
    "handedness": "right",
    "origin": "front_inner_aperture_center",
    "x_axis": "front_view_right",
    "y_axis": "front_view_down",
    "z_axis": "front_to_back",
}
_FEATURE_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]
_OBSERVED_STREAMS = {
    "camera": {
        "transport": "udp_jpeg",
        "stream_id": "vq2-camera-udp-5600",
        "frame_id_field": "frame_id",
        "source_time_field": "sim_time_ns",
        "source_time_semantics": (
            "opaque_ordering_token_not_calibrated_capture_time"
        ),
        "identity_schema": "aigp-vq2-frame-identity/1",
        "timing_schema": "aigp-vq2-frame-timing/1",
        "consume_timing_schema": "aigp-vq2-camera-frame-timing-observation/1",
        "expected_decoded_dimensions": {"width": 640, "height": 360},
        "decoded_dimensions_policy": (
            "observe_require_exact_before_arm_and_session_stability"
        ),
        "host_receipt_clock": "host-perf-counter",
    },
    "imu": {
        "message": "HIGHRES_IMU",
        "source_time_field": "time_usec",
        "source_time_semantics": (
            "opaque_source_clock_for_ordering_and_integration"
        ),
        "accel_fields": ["xacc", "yacc", "zacc"],
        "gyro_fields": ["xgyro", "ygyro", "zgyro"],
        "ingress_schema": "aigp-vq2-mavlink-ingress/1",
        "sample_schema": "aigp-vq2-received-imu/1",
        "host_receipt_clock": "host-perf-counter",
    },
    "race_status": {
        "required_fields": [
            "active_gate_index",
            "last_gate_race_time",
            "race_finish_time_ns",
            "race_start_boot_time_ms",
            "sim_boot_time_ms",
        ],
        "source_time_field": "sim_boot_time_ms",
        "ingress_schema": "aigp-vq2-mavlink-ingress/1",
        "host_receipt_clock": "host-perf-counter",
    },
    "heartbeat": {
        "required_fields": ["base_mode", "custom_mode"],
        "source_time_semantics": "no_admitted_source_timestamp",
        "ingress_schema": "aigp-vq2-mavlink-ingress/1",
        "host_receipt_clock": "host-perf-counter",
    },
    "safety_audit": {
        "actuator": {
            "message": "ACTUATOR_OUTPUT_STATUS",
            "ingress_schema": "aigp-vq2-mavlink-ingress/1",
            "host_receipt_clock": "host-perf-counter",
        },
        "collision": {
            "message": "COLLISION",
            "lineage_semantics": (
                "runner_drain_observation_order_only_no_receiver_"
                "receipt_timestamp"
            ),
        },
        "semantics": "watchdog_and_evidence_only_not_calibration_inputs",
    },
    "unavailable_as_truth": [
        "ATTITUDE",
        "LOCAL_POSITION_NED",
        "ODOMETRY",
        "track_gate_map",
    ],
}
_CALIBRATION_STATUS = {
    "intrinsics": "uncomputed",
    "distortion": "uncomputed",
    "camera_to_body_rotation": "uncomputed",
    "camera_imu_time_model": "uncomputed",
    "rank": "uncomputed",
    "covariance": "uncomputed",
    "empirical_limits": "uncomputed",
}
_DATA_SCOPE = {
    "private_simulation_capture": True,
    "physical_or_hil_use": False,
    "submitted_run_use": False,
    "public_release": False,
    "external_service_upload": False,
    "git_storage": False,
    "pak_access": False,
}

_AUTHORITY = {
    "kind": "user_operator",
    "authority_id": "conversation-2026-07-22-package2-f03-sim-capture",
    "authorized_on": "2026-07-22",
    "source": "direct_user_instruction",
}
_ALLOWED_PURPOSES = [
    "calibration_discovery",
    "independent_review",
    "integrity_audit",
    "offline_replay_and_analysis",
]
_ALLOWED_CLASSES = [
    "annotations_crops_features",
    "commands",
    "decoded_frames",
    "derived_replay_and_analysis",
    "highres_imu",
    "process_lease_cleanup",
    "race_heartbeat_actuator_collision",
    "reconstructed_jpegs",
    "source_and_host_timestamps",
    "udp_camera_datagrams",
]
_STORAGE = {
    "private_root": (
        r"C:\Users\John\aigp-evidence"
        r"\2026-07-22-package2-f03-powered-calibration-attempt"
    ),
    "git": False,
    "public_release": False,
    "network_export": False,
    "external_service_upload": False,
}
_RETENTION = {
    "through": "package2_simulator_audit_closeout",
    "after": "sealed_quarantine_pending_explicit_disposition",
    "automatic_deletion": False,
}
_TRANSFER = {
    "successor_task": False,
    "new_session": False,
    "new_build_or_mode": False,
    "submitted_run": False,
    "physical_or_hil": False,
}


class CalibrationTargetError(RuntimeError):
    """A fail-closed protocol, schema, identity, or file-read error."""


@dataclass(frozen=True)
class ValidatedJsonDocument:
    """An immutable validated value bound to its exact stable byte identity."""

    path: Path
    size_bytes: int
    sha256: str
    value: Mapping[str, Any]


def _reject_constant(value: str) -> None:
    raise CalibrationTargetError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationTargetError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _walk_finite(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        raise CalibrationTargetError("non-finite JSON number is forbidden")
    if type(value) is list:
        for item in value:
            _walk_finite(item)
    elif type(value) is dict:
        for item in value.values():
            _walk_finite(item)


def strict_json_bytes(payload: bytes) -> Any:
    """Decode strict UTF-8 JSON while rejecting ambiguous/non-finite forms."""

    if len(payload) > MAX_JSON_BYTES:
        raise CalibrationTargetError("JSON input exceeds the resource limit")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise CalibrationTargetError("UTF-8 BOM is forbidden")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CalibrationTargetError("JSON input is not strict UTF-8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        _walk_finite(value)
    except CalibrationTargetError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CalibrationTargetError("JSON input is malformed") from exc
    return value


def _signature(info: os.stat_result) -> tuple[Any, ...]:
    return tuple(
        getattr(info, field, None)
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def _read_stable_bytes(path: str | os.PathLike[str], *, label: str) -> tuple[Path, bytes]:
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str) or raw_path == "":
        raise CalibrationTargetError(f"{label} path must be an explicit nonempty path")
    supplied = Path(raw_path)
    flags = os.O_RDONLY
    for optional in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional, 0))
    try:
        named_before = os.stat(supplied, follow_symlinks=False)
        if not stat.S_ISREG(named_before.st_mode):
            raise CalibrationTargetError(f"{label} must be a regular non-symlink file")
        descriptor = os.open(supplied, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(
                named_before, opened
            ):
                raise CalibrationTargetError(f"{label} changed while opening")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                first = handle.read(MAX_JSON_BYTES + 1)
                if len(first) > MAX_JSON_BYTES:
                    raise CalibrationTargetError(
                        f"{label} exceeds the JSON resource limit"
                    )
                handle.seek(0)
                second = handle.read(MAX_JSON_BYTES + 1)
            after = os.fstat(descriptor)
            named_after = os.stat(supplied, follow_symlinks=False)
            if (
                first != second
                or _signature(opened) != _signature(after)
                or _signature(named_before) != _signature(named_after)
                or not stat.S_ISREG(named_after.st_mode)
                or not os.path.samestat(after, named_after)
            ):
                raise CalibrationTargetError(f"{label} changed while being read")
        finally:
            os.close(descriptor)
    except CalibrationTargetError:
        raise
    except OSError as exc:
        raise CalibrationTargetError(f"cannot read {label}: {exc}") from exc
    return Path(os.path.abspath(supplied)), first


def _require_object(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CalibrationTargetError(f"{label} must be a JSON object")
    return value


def _require_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, label: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unknown = sorted(actual - set(expected))
        raise CalibrationTargetError(
            f"{label} keys mismatch; missing={missing!r}, unknown={unknown!r}"
        )


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if type(expected) is dict:
        obj = _require_object(value, label=label)
        _require_keys(obj, set(expected), label=label)
        for key, expected_item in expected.items():
            _require_exact(obj[key], expected_item, label=f"{label}.{key}")
        return
    if type(expected) is list:
        if type(value) is not list or len(value) != len(expected):
            raise CalibrationTargetError(f"{label} must equal the frozen array")
        for index, (item, expected_item) in enumerate(zip(value, expected)):
            _require_exact(item, expected_item, label=f"{label}[{index}]")
        return
    if type(value) is not type(expected) or value != expected:
        raise CalibrationTargetError(f"{label} must equal {expected!r}")


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if type(value) is not str or value == "":
        raise CalibrationTargetError(f"{label} must be a nonempty string")
    return value


def _require_positive_number(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise CalibrationTargetError(f"{label} must be a JSON number, not a boolean")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise CalibrationTargetError(f"{label} must be finite") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise CalibrationTargetError(f"{label} must be finite and strictly positive")
    return converted


def _require_coordinate(value: Any, expected: tuple[float, float, float], *, label: str) -> None:
    if type(value) is not list or len(value) != 3:
        raise CalibrationTargetError(f"{label} must be a three-element JSON array")
    converted: list[float] = []
    for index, item in enumerate(value):
        if type(item) not in (int, float):
            raise CalibrationTargetError(
                f"{label}[{index}] must be a finite JSON number, not a boolean"
            )
        try:
            number = float(item)
        except (OverflowError, ValueError) as exc:
            raise CalibrationTargetError(f"{label}[{index}] must be finite") from exc
        if not math.isfinite(number):
            raise CalibrationTargetError(f"{label}[{index}] must be finite")
        converted.append(number)
    if tuple(converted) != expected:
        raise CalibrationTargetError(
            f"{label} does not match the configured inner dimensions and feature order"
        )


def validate_target_config(value: Any) -> None:
    """Validate the complete collection-config schema and nominal geometry."""

    obj = _require_object(value, label="target_config")
    _require_keys(
        obj,
        {
            "schema",
            "config_id",
            "revision",
            "simulator",
            "source",
            "applicability",
            "geometry",
            "observed_streams",
            "calibration_status",
            "data_scope",
        },
        label="target_config",
    )
    _require_exact(obj["schema"], TARGET_CONFIG_SCHEMA, label="target_config.schema")
    _require_nonempty_string(obj["config_id"], label="target_config.config_id")
    _require_nonempty_string(obj["revision"], label="target_config.revision")
    _require_exact(obj["simulator"], _SIMULATOR, label="target_config.simulator")
    _require_exact(obj["source"], _SOURCE, label="target_config.source")
    _require_exact(
        obj["applicability"], _APPLICABILITY, label="target_config.applicability"
    )

    geometry = _require_object(obj["geometry"], label="target_config.geometry")
    _require_keys(
        geometry,
        {
            "units",
            "target_frame",
            "outer",
            "inner",
            "depth",
            "feature",
            "uncertainty_status",
        },
        label="target_config.geometry",
    )
    _require_exact(geometry["units"], "m", label="target_config.geometry.units")
    _require_exact(
        geometry["target_frame"],
        _TARGET_FRAME,
        label="target_config.geometry.target_frame",
    )
    outer = _require_object(geometry["outer"], label="target_config.geometry.outer")
    inner = _require_object(geometry["inner"], label="target_config.geometry.inner")
    _require_keys(outer, {"width", "height"}, label="target_config.geometry.outer")
    _require_keys(inner, {"width", "height"}, label="target_config.geometry.inner")
    outer_width = _require_positive_number(
        outer["width"], label="target_config.geometry.outer.width"
    )
    outer_height = _require_positive_number(
        outer["height"], label="target_config.geometry.outer.height"
    )
    inner_width = _require_positive_number(
        inner["width"], label="target_config.geometry.inner.width"
    )
    inner_height = _require_positive_number(
        inner["height"], label="target_config.geometry.inner.height"
    )
    _require_positive_number(geometry["depth"], label="target_config.geometry.depth")
    if inner_width >= outer_width or inner_height >= outer_height:
        raise CalibrationTargetError(
            "target_config.geometry inner dimensions must be smaller than outer dimensions"
        )

    feature = _require_object(
        geometry["feature"], label="target_config.geometry.feature"
    )
    _require_keys(
        feature,
        {"kind", "order", "coordinates"},
        label="target_config.geometry.feature",
    )
    _require_exact(
        feature["kind"],
        "front_inner_aperture_boundary_intersections",
        label="target_config.geometry.feature.kind",
    )
    _require_exact(
        feature["order"], _FEATURE_ORDER, label="target_config.geometry.feature.order"
    )
    coordinates = feature["coordinates"]
    if type(coordinates) is not list or len(coordinates) != 4:
        raise CalibrationTargetError(
            "target_config.geometry.feature.coordinates must contain four vectors"
        )
    half_width = inner_width / 2.0
    half_height = inner_height / 2.0
    expected_coordinates = (
        (-half_width, -half_height, 0.0),
        (half_width, -half_height, 0.0),
        (half_width, half_height, 0.0),
        (-half_width, half_height, 0.0),
    )
    for index, expected in enumerate(expected_coordinates):
        _require_coordinate(
            coordinates[index],
            expected,
            label=f"target_config.geometry.feature.coordinates[{index}]",
        )
    _require_exact(
        geometry["uncertainty_status"],
        "unpublished_unknown",
        label="target_config.geometry.uncertainty_status",
    )
    _require_exact(
        obj["observed_streams"],
        _OBSERVED_STREAMS,
        label="target_config.observed_streams",
    )
    _require_exact(
        obj["calibration_status"],
        _CALIBRATION_STATUS,
        label="target_config.calibration_status",
    )
    _require_exact(obj["data_scope"], _DATA_SCOPE, label="target_config.data_scope")


def validate_capture_authorization(value: Any) -> None:
    """Validate the exact, non-transferable F03 simulation authority."""

    obj = _require_object(value, label="capture_authorization")
    _require_keys(
        obj,
        {
            "schema",
            "authority",
            "task_id",
            "domain",
            "simulator",
            "session_ids",
            "allowed_purposes",
            "allowed_classes",
            "storage",
            "retention",
            "transfer",
            "organizer_media_credential",
            "publication_permitted",
        },
        label="capture_authorization",
    )
    _require_exact(
        obj["schema"],
        CAPTURE_AUTHORIZATION_SCHEMA,
        label="capture_authorization.schema",
    )
    _require_exact(obj["authority"], _AUTHORITY, label="capture_authorization.authority")
    _require_exact(
        obj["task_id"],
        "vq2-package2-f03-powered-calibration-attempt",
        label="capture_authorization.task_id",
    )
    _require_exact(
        obj["domain"], "simulator_only", label="capture_authorization.domain"
    )
    _require_exact(obj["simulator"], _SIMULATOR, label="capture_authorization.simulator")
    _require_exact(
        obj["session_ids"], ["F03"], label="capture_authorization.session_ids"
    )
    _require_exact(
        obj["allowed_purposes"],
        _ALLOWED_PURPOSES,
        label="capture_authorization.allowed_purposes",
    )
    _require_exact(
        obj["allowed_classes"],
        _ALLOWED_CLASSES,
        label="capture_authorization.allowed_classes",
    )
    _require_exact(obj["storage"], _STORAGE, label="capture_authorization.storage")
    _require_exact(
        obj["retention"], _RETENTION, label="capture_authorization.retention"
    )
    _require_exact(obj["transfer"], _TRANSFER, label="capture_authorization.transfer")
    _require_exact(
        obj["organizer_media_credential"],
        False,
        label="capture_authorization.organizer_media_credential",
    )
    _require_exact(
        obj["publication_permitted"],
        False,
        label="capture_authorization.publication_permitted",
    )


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def validate_sha256(value: Any, *, label: str) -> str:
    """Require canonical lowercase SHA-256 text."""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise CalibrationTargetError(
            f"{label} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _read_document(
    path: str | os.PathLike[str],
    *,
    label: str,
    validator: Any,
    expected_sha256: str | None,
) -> ValidatedJsonDocument:
    absolute_path, payload = _read_stable_bytes(path, label=label)
    value = strict_json_bytes(payload)
    validator(value)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        expected = validate_sha256(expected_sha256, label=f"expected {label} SHA-256")
        if digest != expected:
            raise CalibrationTargetError(f"{label} SHA-256 mismatch")
    return ValidatedJsonDocument(
        path=absolute_path,
        size_bytes=len(payload),
        sha256=digest,
        value=_freeze(value),
    )


def read_target_config(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> ValidatedJsonDocument:
    """Read, validate, and identify one explicitly supplied target config."""

    return _read_document(
        path,
        label="target config",
        validator=validate_target_config,
        expected_sha256=expected_sha256,
    )


def read_capture_authorization(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> ValidatedJsonDocument:
    """Read, validate, and identify one explicitly supplied capture authority."""

    return _read_document(
        path,
        label="capture authorization",
        validator=validate_capture_authorization,
        expected_sha256=expected_sha256,
    )


def validate_collection_protocol(
    target_config: ValidatedJsonDocument,
    capture_authorization: ValidatedJsonDocument,
) -> None:
    """Check the cross-document simulator and simulation-only scope binding."""

    if target_config.value["schema"] != TARGET_CONFIG_SCHEMA:
        raise CalibrationTargetError("target document has the wrong validated schema")
    if capture_authorization.value["schema"] != CAPTURE_AUTHORIZATION_SCHEMA:
        raise CalibrationTargetError(
            "capture-authorization document has the wrong validated schema"
        )
    if target_config.value["simulator"] != capture_authorization.value["simulator"]:
        raise CalibrationTargetError("target and authorization simulator scopes differ")
    if target_config.value["data_scope"]["private_simulation_capture"] is not True:
        raise CalibrationTargetError("target does not permit private simulation capture")
    if capture_authorization.value["domain"] != "simulator_only":
        raise CalibrationTargetError("capture authority is not simulator-only")


def validate_config_replacement(
    previous: ValidatedJsonDocument, replacement: ValidatedJsonDocument
) -> None:
    """Require new config ID, revision, and hash for any byte-level change."""

    if previous.value["schema"] != TARGET_CONFIG_SCHEMA:
        raise CalibrationTargetError("previous document is not a target config")
    if replacement.value["schema"] != TARGET_CONFIG_SCHEMA:
        raise CalibrationTargetError("replacement document is not a target config")
    if previous.sha256 == replacement.sha256:
        return
    if previous.value["config_id"] == replacement.value["config_id"]:
        raise CalibrationTargetError("changed target config requires a new config_id")
    if previous.value["revision"] == replacement.value["revision"]:
        raise CalibrationTargetError("changed target config requires a new revision")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and identify an explicit VQ2 target config and private "
            "simulation-capture authorization."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--capture-authorization", required=True)
    parser.add_argument("--target-config-sha256")
    parser.add_argument("--capture-authorization-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = read_target_config(
            args.target_config, expected_sha256=args.target_config_sha256
        )
        authorization = read_capture_authorization(
            args.capture_authorization,
            expected_sha256=args.capture_authorization_sha256,
        )
        validate_collection_protocol(target, authorization)
    except CalibrationTargetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "capture_authorization_sha256": authorization.sha256,
                "target_config_sha256": target.sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
