"""Strict versioned configuration for VQ2 multi-gate visual navigation.

Only controller tuning lives here.  Reset/GO sequencing, source freshness,
collision policy, command-rate ceilings, attitude ceilings, per-segment hard
duration/yaw envelopes, race authority, disarm/reset, and cleanup are
code-owned runtime invariants and intentionally absent from this schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from planning.vq2_visual_servo import (
    MAX_NEXT_GATE_BLEND,
    MAX_VISUAL_SEGMENT_DURATION_S,
    VisualServoRefusal,
    VisualServoTuning,
)


VISUAL_CONFIG_SCHEMA = "aigp-vq2-visual-navigation-config/1"
VISUAL_CONTROLLER_FAMILY = "aigp-vq2-multigate-visual-servo/1"

_TOP_LEVEL_FIELDS = frozenset(
    {"schema", "controller_family", "servo", "lifecycle"}
)
_SERVO_FIELDS = frozenset(VisualServoTuning.__dataclass_fields__)
_LIFECYCLE_FIELDS = frozenset(
    {
        "next_gate_blend_max",
        "next_gate_blend_start_log_scale",
        "next_gate_blend_full_log_scale",
        "restricted_alignment_duration_s",
        "required_improving_frames",
        "launch_boost_duration_s",
        "launch_boost_thrust",
        "launch_pitch_blend_s",
    }
)


class VisualConfigError(ValueError):
    """A visual-navigation configuration is not exactly admissible."""


@dataclass(frozen=True)
class VisualLifecycleTuning:
    # The gain-only live candidate remained safely inside every yaw bound but
    # left substantial unused preview authority.  Use the existing immutable
    # blend ceiling while the current aperture remains inside its separate
    # passage corridor.
    next_gate_blend_max: float = 0.35
    next_gate_blend_start_log_scale: float = -1.80
    next_gate_blend_full_log_scale: float = -0.50
    restricted_alignment_duration_s: float = 0.90
    required_improving_frames: int = 3
    launch_boost_duration_s: float = 0.45
    launch_boost_thrust: float = 0.32
    launch_pitch_blend_s: float = 0.80

    def __post_init__(self) -> None:
        values = (
            self.next_gate_blend_max,
            self.next_gate_blend_start_log_scale,
            self.next_gate_blend_full_log_scale,
            self.restricted_alignment_duration_s,
            self.launch_boost_duration_s,
            self.launch_boost_thrust,
            self.launch_pitch_blend_s,
        )
        if not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in values
        ):
            raise VisualConfigError("lifecycle values must be finite numbers")
        if not 0.0 <= float(self.next_gate_blend_max) <= MAX_NEXT_GATE_BLEND:
            raise VisualConfigError("next-gate blend exceeds its fixed ceiling")
        if not -3.0 <= float(self.next_gate_blend_start_log_scale) <= -1.0:
            raise VisualConfigError("next-gate blend start scale is outside bounds")
        if not -1.0 <= float(self.next_gate_blend_full_log_scale) <= -0.20:
            raise VisualConfigError("next-gate blend full scale is outside bounds")
        if not (
            float(self.next_gate_blend_start_log_scale)
            < float(self.next_gate_blend_full_log_scale)
        ):
            raise VisualConfigError("next-gate blend scale interval is invalid")
        if not 0.40 <= float(self.restricted_alignment_duration_s) <= 0.90:
            raise VisualConfigError("restricted alignment duration is outside bounds")
        if type(self.required_improving_frames) is not int or not (
            3 <= self.required_improving_frames <= 6
        ):
            raise VisualConfigError("required improving frames are outside bounds")
        if not 0.45 <= float(self.launch_boost_duration_s) <= 0.60:
            raise VisualConfigError("launch boost duration is outside bounds")
        if not 0.30 <= float(self.launch_boost_thrust) <= 0.32:
            raise VisualConfigError("launch boost thrust is outside bounds")
        if not 0.80 <= float(self.launch_pitch_blend_s) <= 1.0:
            raise VisualConfigError("launch pitch blend duration is outside bounds")
        if float(self.restricted_alignment_duration_s) >= (
            MAX_VISUAL_SEGMENT_DURATION_S
        ):
            raise VisualConfigError("alignment duration must stay below hard segment time")


@dataclass(frozen=True)
class VisualNavigationConfig:
    schema: str
    controller_family: str
    servo: VisualServoTuning
    lifecycle: VisualLifecycleTuning

    def to_effective_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_family": self.controller_family,
            "servo": asdict(self.servo),
            "lifecycle": asdict(self.lifecycle),
        }

    @property
    def effective_config_sha256(self) -> str:
        return canonical_visual_config_sha256(self)


def default_visual_config_mapping() -> dict[str, Any]:
    return {
        "schema": VISUAL_CONFIG_SCHEMA,
        "controller_family": VISUAL_CONTROLLER_FAMILY,
        "servo": asdict(VisualServoTuning()),
        "lifecycle": asdict(VisualLifecycleTuning()),
    }


def _require_exact_fields(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualConfigError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise VisualConfigError(f"{label} keys must be exact strings")
    actual = frozenset(value)
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing or unknown:
        raise VisualConfigError(
            f"{label} fields differ: missing={missing}, unknown={unknown}"
        )
    return value


def _number(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise VisualConfigError(f"{label} must be a finite number")
    return float(value)


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise VisualConfigError(f"{label} must be an exact int")
    return value


def validate_visual_config(document: object) -> VisualNavigationConfig:
    top = _require_exact_fields(
        document,
        _TOP_LEVEL_FIELDS,
        label="visual_config",
    )
    if top["schema"] != VISUAL_CONFIG_SCHEMA:
        raise VisualConfigError("visual_config.schema is unsupported")
    if top["controller_family"] != VISUAL_CONTROLLER_FAMILY:
        raise VisualConfigError("visual_config.controller_family is unsupported")
    servo_value = _require_exact_fields(
        top["servo"],
        _SERVO_FIELDS,
        label="visual_config.servo",
    )
    lifecycle_value = _require_exact_fields(
        top["lifecycle"],
        _LIFECYCLE_FIELDS,
        label="visual_config.lifecycle",
    )
    try:
        servo = VisualServoTuning(
            **{
                field: (
                    _integer(
                        servo_value[field],
                        label=f"visual_config.servo.{field}",
                    )
                    if field == "required_corridor_frames"
                    else _number(
                        servo_value[field],
                        label=f"visual_config.servo.{field}",
                    )
                )
                for field in _SERVO_FIELDS
            }
        )
    except VisualServoRefusal as exc:
        raise VisualConfigError(str(exc)) from exc
    lifecycle = VisualLifecycleTuning(
        **{
            field: (
                _integer(
                    lifecycle_value[field],
                    label=f"visual_config.lifecycle.{field}",
                )
                if field == "required_improving_frames"
                else _number(
                    lifecycle_value[field],
                    label=f"visual_config.lifecycle.{field}",
                )
            )
            for field in _LIFECYCLE_FIELDS
        }
    )
    return VisualNavigationConfig(
        schema=VISUAL_CONFIG_SCHEMA,
        controller_family=VISUAL_CONTROLLER_FAMILY,
        servo=servo,
        lifecycle=lifecycle,
    )


def default_visual_config() -> VisualNavigationConfig:
    return validate_visual_config(default_visual_config_mapping())


def canonical_visual_config_sha256(
    value: Mapping[str, Any] | VisualNavigationConfig,
) -> str:
    config = (
        value
        if isinstance(value, VisualNavigationConfig)
        else validate_visual_config(value)
    )
    payload = json.dumps(
        config.to_effective_mapping(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "VISUAL_CONFIG_SCHEMA",
    "VISUAL_CONTROLLER_FAMILY",
    "VisualConfigError",
    "VisualLifecycleTuning",
    "VisualNavigationConfig",
    "canonical_visual_config_sha256",
    "default_visual_config",
    "default_visual_config_mapping",
    "validate_visual_config",
]
