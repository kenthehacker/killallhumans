"""Code-owned bounded yaw-calibration waveform for FlightSim build 3385.

The plan is deliberately small and exact.  Runtime code can bind evidence to
``YAW_CALIBRATION_PLAN_SHA256`` and use :func:`yaw_calibration_tick` without
reconstructing segment boundaries or accepting caller-provided tuning.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping


YAW_CALIBRATION_PLAN_SCHEMA = "aigp-vq2-yaw-calibration-plan/1"
YAW_CALIBRATION_PLAN_ID = (
    "vq2-build3385-training-yaw-envelope-calibration-v3"
)

YAW_CALIBRATION_CONTROL_PERIOD_NS = 20_000_000
YAW_CALIBRATION_TICK_COUNT = 45
YAW_CALIBRATION_NOMINAL_END_OFFSET_NS = 900_000_000
YAW_CALIBRATION_HARD_EXPIRY_OFFSET_NS = 1_000_000_000

YAW_CALIBRATION_THRUST = 0.235
YAW_CALIBRATION_ROLL_RATE_RAD_S = 0.0
YAW_CALIBRATION_PITCH_RATE_RAD_S = 0.0
YAW_CALIBRATION_RATE_RAD_S = 0.08


class YawCalibrationPlanError(ValueError):
    """The yaw-calibration plan or a tick lookup is not exactly admissible."""


_PLAN_LITERAL: dict[str, Any] = {
    "schema": YAW_CALIBRATION_PLAN_SCHEMA,
    "plan_id": YAW_CALIBRATION_PLAN_ID,
    "stage": "calibration-excite",
    "control_period_ns": YAW_CALIBRATION_CONTROL_PERIOD_NS,
    "tick_count": YAW_CALIBRATION_TICK_COUNT,
    "nominal_end_offset_ns": YAW_CALIBRATION_NOMINAL_END_OFFSET_NS,
    "powered_hard_expiry_offset_ns": YAW_CALIBRATION_HARD_EXPIRY_OFFSET_NS,
    "command": {
        "thrust": YAW_CALIBRATION_THRUST,
        "roll_rate_rad_s": YAW_CALIBRATION_ROLL_RATE_RAD_S,
        "pitch_rate_rad_s": YAW_CALIBRATION_PITCH_RATE_RAD_S,
    },
    "segments": [
        {
            "segment_id": "neutral-initial",
            "first_tick": 0,
            "last_tick": 11,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-positive",
            "first_tick": 12,
            "last_tick": 22,
            "yaw_rate_rad_s": YAW_CALIBRATION_RATE_RAD_S,
        },
        {
            "segment_id": "neutral-reversal",
            "first_tick": 23,
            "last_tick": 28,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-negative",
            "first_tick": 29,
            "last_tick": 39,
            "yaw_rate_rad_s": -YAW_CALIBRATION_RATE_RAD_S,
        },
        {
            "segment_id": "neutral-terminal",
            "first_tick": 40,
            "last_tick": 44,
            "yaw_rate_rad_s": 0.0,
        },
    ],
}

# SHA-256 of the canonical JSON object (sorted keys, compact separators, UTF-8).
# This is intentionally a literal rather than being derived at import time: a
# plan edit must make an explicit, reviewable identity change.
YAW_CALIBRATION_PLAN_SHA256 = (
    "9aa0a596e03ba685e3b5187b2940b0a3071a70e7e14686a398048bd1916ef91a"
)


def _fail(path: str, detail: str) -> None:
    raise YawCalibrationPlanError(f"{path}: {detail}")


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


FROZEN_YAW_CALIBRATION_PLAN = _freeze_json(_PLAN_LITERAL)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole canonical byte representation used for plan identity."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise YawCalibrationPlanError(
            f"$: value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_yaw_calibration_plan_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def yaw_calibration_plan() -> dict[str, Any]:
    """Return a defensive mutable copy of the exact code-owned plan."""

    return deepcopy(_PLAN_LITERAL)


def _exact_object(
    value: Any,
    fields: set[str],
    path: str,
) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an exact object")
    if any(type(key) is not str for key in value):
        _fail(path, "keys must be exact strings")
    actual = set(value)
    if actual != fields:
        _fail(
            path,
            "fields differ: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}",
        )
    return value


def _exact_int(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        _fail(path, "must be an exact int")
    if value < minimum or (maximum is not None and value > maximum):
        _fail(path, "is outside its admitted range")
    return value


def _exact_float(value: Any, path: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(path, "must be an exact finite float")
    return value


def _validate_shape(value: Any) -> None:
    plan = _exact_object(
        value,
        {
            "schema",
            "plan_id",
            "stage",
            "control_period_ns",
            "tick_count",
            "nominal_end_offset_ns",
            "powered_hard_expiry_offset_ns",
            "command",
            "segments",
        },
        "$plan",
    )
    for name in ("schema", "plan_id", "stage"):
        if type(plan[name]) is not str:
            _fail(f"$plan.{name}", "must be an exact string")

    period = _exact_int(plan["control_period_ns"], "$plan.control_period_ns")
    tick_count = _exact_int(plan["tick_count"], "$plan.tick_count", minimum=1)
    nominal_end = _exact_int(
        plan["nominal_end_offset_ns"],
        "$plan.nominal_end_offset_ns",
        minimum=1,
    )
    hard_expiry = _exact_int(
        plan["powered_hard_expiry_offset_ns"],
        "$plan.powered_hard_expiry_offset_ns",
        minimum=1,
    )
    if nominal_end != tick_count * period:
        _fail(
            "$plan.nominal_end_offset_ns",
            "must equal tick_count * control_period_ns",
        )
    if hard_expiry <= nominal_end:
        _fail(
            "$plan.powered_hard_expiry_offset_ns",
            "must be later than the nominal plan end",
        )

    command = _exact_object(
        plan["command"],
        {"thrust", "roll_rate_rad_s", "pitch_rate_rad_s"},
        "$plan.command",
    )
    for name in ("thrust", "roll_rate_rad_s", "pitch_rate_rad_s"):
        _exact_float(command[name], f"$plan.command.{name}")

    segments = plan["segments"]
    if type(segments) is not list or not segments:
        _fail("$plan.segments", "must be a nonempty exact array")
    next_tick = 0
    segment_ids: set[str] = set()
    for index, segment_value in enumerate(segments):
        path = f"$plan.segments[{index}]"
        segment = _exact_object(
            segment_value,
            {"segment_id", "first_tick", "last_tick", "yaw_rate_rad_s"},
            path,
        )
        segment_id = segment["segment_id"]
        if type(segment_id) is not str or not segment_id:
            _fail(f"{path}.segment_id", "must be a nonempty exact string")
        if segment_id in segment_ids:
            _fail(f"{path}.segment_id", "must be unique")
        segment_ids.add(segment_id)
        first_tick = _exact_int(segment["first_tick"], f"{path}.first_tick")
        last_tick = _exact_int(segment["last_tick"], f"{path}.last_tick")
        _exact_float(segment["yaw_rate_rad_s"], f"{path}.yaw_rate_rad_s")
        if first_tick != next_tick or last_tick < first_tick:
            _fail(
                path,
                "segments must be contiguous, ordered, non-overlapping, and nonempty",
            )
        next_tick = last_tick + 1
    if next_tick != tick_count:
        _fail("$plan.tick_count", "must equal the ticks covered by segments")


def _exact_sha256(value: Any, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(path, "must be a lowercase SHA-256 hex digest")
    return value


def validate_yaw_calibration_plan(
    value: Any,
    *,
    expected_sha256: str = YAW_CALIBRATION_PLAN_SHA256,
) -> dict[str, Any]:
    """Validate exact structure, content, and frozen plan identity."""

    _validate_shape(value)
    digest = _exact_sha256(expected_sha256, "$expected_sha256")
    if digest != YAW_CALIBRATION_PLAN_SHA256:
        _fail(
            "$expected_sha256",
            f"must equal frozen {YAW_CALIBRATION_PLAN_SHA256}",
        )
    actual = canonical_yaw_calibration_plan_sha256(value)
    if actual != YAW_CALIBRATION_PLAN_SHA256:
        _fail(
            "$plan",
            f"object SHA-256 must equal frozen {YAW_CALIBRATION_PLAN_SHA256}",
        )
    if value != _PLAN_LITERAL:
        _fail("$plan", "must equal the exact frozen plan literal")
    return deepcopy(value)


def yaw_calibration_command_for_tick(tick: Any) -> dict[str, float]:
    """Return the bounded body-rate/thrust command for one exact plan tick."""

    absolute_tick = _exact_int(
        tick,
        "$tick",
        maximum=YAW_CALIBRATION_TICK_COUNT - 1,
    )
    segment = next(
        item
        for item in _PLAN_LITERAL["segments"]
        if item["first_tick"] <= absolute_tick <= item["last_tick"]
    )
    return {
        "roll_rate_rad_s": YAW_CALIBRATION_ROLL_RATE_RAD_S,
        "pitch_rate_rad_s": YAW_CALIBRATION_PITCH_RATE_RAD_S,
        "yaw_rate_rad_s": segment["yaw_rate_rad_s"],
        "thrust": YAW_CALIBRATION_THRUST,
    }


def yaw_calibration_tick(
    tick: Any,
    *,
    anchor_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Resolve one tick to exact offsets or absolute monotonic deadlines."""

    absolute_tick = _exact_int(
        tick,
        "$tick",
        maximum=YAW_CALIBRATION_TICK_COUNT - 1,
    )
    anchor = (
        0
        if anchor_monotonic_ns is None
        else _exact_int(anchor_monotonic_ns, "$anchor_monotonic_ns")
    )
    segment = next(
        item
        for item in _PLAN_LITERAL["segments"]
        if item["first_tick"] <= absolute_tick <= item["last_tick"]
    )
    release = anchor + absolute_tick * YAW_CALIBRATION_CONTROL_PERIOD_NS
    end = release + YAW_CALIBRATION_CONTROL_PERIOD_NS
    expiry = anchor + YAW_CALIBRATION_HARD_EXPIRY_OFFSET_NS
    result = {
        "absolute_tick": absolute_tick,
        "segment_id": segment["segment_id"],
        "release_monotonic_ns": release,
        "end_monotonic_ns": end,
        "powered_expiry_monotonic_ns": expiry,
        "command": yaw_calibration_command_for_tick(absolute_tick),
    }
    if anchor_monotonic_ns is None:
        return {
            "absolute_tick": result["absolute_tick"],
            "segment_id": result["segment_id"],
            "release_offset_ns": result["release_monotonic_ns"],
            "end_offset_ns": result["end_monotonic_ns"],
            "powered_expiry_offset_ns": result[
                "powered_expiry_monotonic_ns"
            ],
            "command": result["command"],
        }
    return result


def iter_yaw_calibration_ticks(
    *,
    anchor_monotonic_ns: int | None = None,
) -> Iterable[dict[str, Any]]:
    for tick in range(YAW_CALIBRATION_TICK_COUNT):
        yield yaw_calibration_tick(
            tick,
            anchor_monotonic_ns=anchor_monotonic_ns,
        )


__all__ = [
    "FROZEN_YAW_CALIBRATION_PLAN",
    "YAW_CALIBRATION_CONTROL_PERIOD_NS",
    "YAW_CALIBRATION_HARD_EXPIRY_OFFSET_NS",
    "YAW_CALIBRATION_NOMINAL_END_OFFSET_NS",
    "YAW_CALIBRATION_PLAN_ID",
    "YAW_CALIBRATION_PLAN_SCHEMA",
    "YAW_CALIBRATION_PLAN_SHA256",
    "YAW_CALIBRATION_PITCH_RATE_RAD_S",
    "YAW_CALIBRATION_RATE_RAD_S",
    "YAW_CALIBRATION_ROLL_RATE_RAD_S",
    "YAW_CALIBRATION_THRUST",
    "YAW_CALIBRATION_TICK_COUNT",
    "YawCalibrationPlanError",
    "canonical_json_bytes",
    "canonical_yaw_calibration_plan_sha256",
    "iter_yaw_calibration_ticks",
    "validate_yaw_calibration_plan",
    "yaw_calibration_command_for_tick",
    "yaw_calibration_plan",
    "yaw_calibration_tick",
]
