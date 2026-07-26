"""Code-owned progressive yaw-capability sweep for FlightSim build 3385.

This plan is characterization evidence only.  It is deliberately distinct
from the accepted ``+/-0.08`` production yaw profile and cannot authorize a
visual-course command envelope by itself.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


YAW_CAPABILITY_PLAN_SCHEMA = "aigp-vq2-yaw-capability-plan/1"
YAW_CAPABILITY_PLAN_ID = "vq2-build3385-training-free-flight-yaw-sweep-v1"
YAW_CAPABILITY_CONTROL_PERIOD_NS = 20_000_000
YAW_CAPABILITY_TICK_COUNT = 81
YAW_CAPABILITY_NOMINAL_END_OFFSET_NS = 1_620_000_000
YAW_CAPABILITY_HARD_EXPIRY_OFFSET_NS = 1_750_000_000
YAW_CAPABILITY_TARGET_ROLL_RAD = 0.0
YAW_CAPABILITY_TARGET_PITCH_RAD = 0.0
YAW_CAPABILITY_THRUST = 0.285
YAW_CAPABILITY_LEVELS_RAD_S = (0.08, 0.12, 0.16, 0.20)


class YawCapabilityPlanError(ValueError):
    """The code-owned capability plan is not exactly admissible."""


_PLAN_LITERAL: dict[str, Any] = {
    "schema": YAW_CAPABILITY_PLAN_SCHEMA,
    "plan_id": YAW_CAPABILITY_PLAN_ID,
    "stage": "calibration-excite",
    "control_period_ns": YAW_CAPABILITY_CONTROL_PERIOD_NS,
    "tick_count": YAW_CAPABILITY_TICK_COUNT,
    "nominal_end_offset_ns": YAW_CAPABILITY_NOMINAL_END_OFFSET_NS,
    "powered_hard_expiry_offset_ns": YAW_CAPABILITY_HARD_EXPIRY_OFFSET_NS,
    "hold": {
        "target_roll_rad": YAW_CAPABILITY_TARGET_ROLL_RAD,
        "target_pitch_rad": YAW_CAPABILITY_TARGET_PITCH_RAD,
        "thrust": YAW_CAPABILITY_THRUST,
    },
    "segments": [
        {
            "segment_id": "neutral-initial",
            "first_tick": 0,
            "last_tick": 4,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-positive-0p08",
            "first_tick": 5,
            "last_tick": 9,
            "yaw_rate_rad_s": 0.08,
        },
        {
            "segment_id": "neutral-01",
            "first_tick": 10,
            "last_tick": 13,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-negative-0p08",
            "first_tick": 14,
            "last_tick": 18,
            "yaw_rate_rad_s": -0.08,
        },
        {
            "segment_id": "neutral-02",
            "first_tick": 19,
            "last_tick": 22,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-positive-0p12",
            "first_tick": 23,
            "last_tick": 27,
            "yaw_rate_rad_s": 0.12,
        },
        {
            "segment_id": "neutral-03",
            "first_tick": 28,
            "last_tick": 31,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-negative-0p12",
            "first_tick": 32,
            "last_tick": 36,
            "yaw_rate_rad_s": -0.12,
        },
        {
            "segment_id": "neutral-04",
            "first_tick": 37,
            "last_tick": 40,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-positive-0p16",
            "first_tick": 41,
            "last_tick": 45,
            "yaw_rate_rad_s": 0.16,
        },
        {
            "segment_id": "neutral-05",
            "first_tick": 46,
            "last_tick": 49,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-negative-0p16",
            "first_tick": 50,
            "last_tick": 54,
            "yaw_rate_rad_s": -0.16,
        },
        {
            "segment_id": "neutral-06",
            "first_tick": 55,
            "last_tick": 58,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-positive-0p20",
            "first_tick": 59,
            "last_tick": 63,
            "yaw_rate_rad_s": 0.20,
        },
        {
            "segment_id": "neutral-07",
            "first_tick": 64,
            "last_tick": 67,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-negative-0p20",
            "first_tick": 68,
            "last_tick": 72,
            "yaw_rate_rad_s": -0.20,
        },
        {
            "segment_id": "neutral-terminal",
            "first_tick": 73,
            "last_tick": 80,
            "yaw_rate_rad_s": 0.0,
        },
    ],
}

# SHA-256 of the canonical JSON object above.  A plan edit must update this
# literal so every compact live manifest binds the exact waveform.
YAW_CAPABILITY_PLAN_SHA256 = (
    "8ec6d12625bed7947682f2ddcdd104ce826c3baadd13ffc8d50dfcdaf6c7531b"
)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise YawCapabilityPlanError(
            f"plan is not canonical finite JSON: {exc}"
        ) from exc


def canonical_yaw_capability_plan_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def yaw_capability_plan() -> dict[str, Any]:
    return deepcopy(_PLAN_LITERAL)


def validate_yaw_capability_plan(value: Any) -> dict[str, Any]:
    if type(value) is not dict or value != _PLAN_LITERAL:
        raise YawCapabilityPlanError(
            "capability plan must equal the exact code-owned literal"
        )
    values = (
        value["control_period_ns"],
        value["tick_count"],
        value["nominal_end_offset_ns"],
        value["powered_hard_expiry_offset_ns"],
        *value["hold"].values(),
        *(
            item["yaw_rate_rad_s"]
            for item in value["segments"]
        ),
    )
    if not all(
        type(item) in {int, float} and math.isfinite(float(item))
        for item in values
    ):
        raise YawCapabilityPlanError("capability plan must be finite")
    if value["control_period_ns"] != YAW_CAPABILITY_CONTROL_PERIOD_NS:
        raise YawCapabilityPlanError("capability plan must run at 50 Hz")
    if (
        value["tick_count"] * value["control_period_ns"]
        != value["nominal_end_offset_ns"]
        or value["powered_hard_expiry_offset_ns"]
        <= value["nominal_end_offset_ns"]
    ):
        raise YawCapabilityPlanError("capability plan timing is inconsistent")
    cursor = 0
    pulse_levels: list[float] = []
    for segment in value["segments"]:
        if (
            type(segment) is not dict
            or set(segment)
            != {
                "segment_id",
                "first_tick",
                "last_tick",
                "yaw_rate_rad_s",
            }
            or segment["first_tick"] != cursor
            or segment["last_tick"] < segment["first_tick"]
        ):
            raise YawCapabilityPlanError(
                "capability segments must be exact and contiguous"
            )
        cursor = int(segment["last_tick"]) + 1
        yaw_rate = float(segment["yaw_rate_rad_s"])
        if yaw_rate != 0.0:
            pulse_levels.append(yaw_rate)
    expected_pulses = [
        signed
        for level in YAW_CAPABILITY_LEVELS_RAD_S
        for signed in (level, -level)
    ]
    if cursor != value["tick_count"] or pulse_levels != expected_pulses:
        raise YawCapabilityPlanError(
            "capability pulses must be symmetric and progressively increasing"
        )
    actual = canonical_yaw_capability_plan_sha256(value)
    if actual != YAW_CAPABILITY_PLAN_SHA256:
        raise YawCapabilityPlanError(
            "capability plan SHA-256 does not match its frozen identity"
        )
    return deepcopy(value)


def yaw_capability_tick(
    tick: int,
    *,
    anchor_monotonic_ns: int = 0,
) -> dict[str, Any]:
    if (
        type(tick) is not int
        or not 0 <= tick < YAW_CAPABILITY_TICK_COUNT
        or type(anchor_monotonic_ns) is not int
        or anchor_monotonic_ns < 0
    ):
        raise YawCapabilityPlanError("capability tick is outside the plan")
    segment: Mapping[str, Any] = next(
        item
        for item in _PLAN_LITERAL["segments"]
        if item["first_tick"] <= tick <= item["last_tick"]
    )
    release_ns = anchor_monotonic_ns + tick * YAW_CAPABILITY_CONTROL_PERIOD_NS
    return {
        "absolute_tick": tick,
        "segment_id": segment["segment_id"],
        "release_monotonic_ns": release_ns,
        "end_monotonic_ns": release_ns + YAW_CAPABILITY_CONTROL_PERIOD_NS,
        "powered_expiry_monotonic_ns": (
            anchor_monotonic_ns + YAW_CAPABILITY_HARD_EXPIRY_OFFSET_NS
        ),
        "command": {
            **_PLAN_LITERAL["hold"],
            "yaw_rate_rad_s": segment["yaw_rate_rad_s"],
        },
    }


def iter_yaw_capability_ticks(
    *,
    anchor_monotonic_ns: int = 0,
) -> Iterable[dict[str, Any]]:
    for tick in range(YAW_CAPABILITY_TICK_COUNT):
        yield yaw_capability_tick(
            tick,
            anchor_monotonic_ns=anchor_monotonic_ns,
        )


__all__ = [
    "YAW_CAPABILITY_CONTROL_PERIOD_NS",
    "YAW_CAPABILITY_HARD_EXPIRY_OFFSET_NS",
    "YAW_CAPABILITY_LEVELS_RAD_S",
    "YAW_CAPABILITY_NOMINAL_END_OFFSET_NS",
    "YAW_CAPABILITY_PLAN_ID",
    "YAW_CAPABILITY_PLAN_SCHEMA",
    "YAW_CAPABILITY_PLAN_SHA256",
    "YAW_CAPABILITY_TARGET_PITCH_RAD",
    "YAW_CAPABILITY_TARGET_ROLL_RAD",
    "YAW_CAPABILITY_THRUST",
    "YAW_CAPABILITY_TICK_COUNT",
    "YawCapabilityPlanError",
    "canonical_yaw_capability_plan_sha256",
    "iter_yaw_capability_ticks",
    "validate_yaw_capability_plan",
    "yaw_capability_plan",
    "yaw_capability_tick",
]
