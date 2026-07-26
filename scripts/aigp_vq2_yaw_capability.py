"""Code-owned progressive yaw-capability sweep for FlightSim build 3385.

This plan is characterization evidence only.  It is deliberately distinct
from the accepted composite yaw profile and cannot authorize a visual-course
command envelope by itself.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


YAW_CAPABILITY_PLAN_SCHEMA = "aigp-vq2-yaw-capability-plan/1"
YAW_CAPABILITY_PLAN_ID = "vq2-build3385-training-free-flight-yaw-sweep-v2"
YAW_CAPABILITY_CONTROL_PERIOD_NS = 20_000_000
YAW_CAPABILITY_TICK_COUNT = 45
YAW_CAPABILITY_NOMINAL_END_OFFSET_NS = 900_000_000
YAW_CAPABILITY_HARD_EXPIRY_OFFSET_NS = 1_000_000_000
YAW_CAPABILITY_TARGET_ROLL_RAD = 0.0
YAW_CAPABILITY_TARGET_PITCH_RAD = 0.05
YAW_CAPABILITY_THRUST = 0.285
YAW_CAPABILITY_LEVELS_RAD_S = (0.12,)


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
            "last_tick": 11,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-positive-0p12",
            "first_tick": 12,
            "last_tick": 22,
            "yaw_rate_rad_s": 0.12,
        },
        {
            "segment_id": "neutral-reversal",
            "first_tick": 23,
            "last_tick": 28,
            "yaw_rate_rad_s": 0.0,
        },
        {
            "segment_id": "yaw-negative-0p12",
            "first_tick": 29,
            "last_tick": 39,
            "yaw_rate_rad_s": -0.12,
        },
        {
            "segment_id": "neutral-terminal",
            "first_tick": 40,
            "last_tick": 44,
            "yaw_rate_rad_s": 0.0,
        },
    ],
}

# SHA-256 of the canonical JSON object above.  A plan edit must update this
# literal so every compact live manifest binds the exact waveform.
YAW_CAPABILITY_PLAN_SHA256 = (
    "06903f918dd89ddd41f684eb68280631efefedfe8bf0a22a320650cccc93d48d"
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
