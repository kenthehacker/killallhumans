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

MAX_VISUAL_SEGMENT_DURATION_S = 120.0
MIN_VISUAL_THRUST = 0.21
MAX_VISUAL_THRUST = 0.32
MAX_NEXT_GATE_BLEND = 0.35


class VisualServoRefusal(ValueError):
    """The observation or requested tuning cannot safely produce authority."""


@dataclass(frozen=True)
class VisualServoTuning:
    """Bounded controller choices below immutable runtime authority ceilings."""

    horizontal_corridor: float = 0.16
    vertical_corridor: float = 0.18
    edge_brake_x: float = 0.72
    edge_brake_y: float = 0.76
    stable_rate_norm_s: float = 0.30
    stable_scale_rate_s: float = 1.10
    brake_scale_rate_s: float = 2.00
    # The first two exact Gate-0 -> Gate-1 handoffs showed that a 0.15 gain
    # behind a 0.25 bearing blend produced only about 0.012 rad/s of preview
    # yaw and left the promoted track moving outward at 0.304-0.349 norm/s.
    # Use the reviewed tuning ceiling so the generic servo can exploit the
    # separately immutable yaw-rate and course-turn heading envelopes.
    yaw_error_gain: float = 0.30
    yaw_rate_gain: float = 0.035
    # Gate-1 live recovery saturated yaw while horizontal error grew from
    # 0.625 to 0.750.  Use materially stronger coordinated bank inside the
    # separately enforced 0.18-rad measured stage corridor.
    roll_error_gain: float = 0.20
    roll_rate_gain: float = 0.05
    # Retained in the serialized v1 configuration for manifest compatibility.
    # Vertical image feedback is no longer applied to pitch; collective is its
    # single control owner.
    vertical_error_gain: float = 0.16
    vertical_rate_gain: float = 0.035
    collective_error_gain: float = 0.060
    collective_rate_gain: float = 0.080
    advance_pitch_rad: float = -0.105
    brake_pitch_rad: float = 0.035
    # Repeated credited Gate-0 runs establish 0.275 as the generic
    # flight-support collective basis.  Forward closure is allocated through
    # pitch and the small continuous interpolation toward advance collective;
    # cutting airborne alignment to the 0.21 envelope minimum caused measured
    # vertical image divergence and top censorship.
    align_thrust: float = 0.275
    advance_thrust: float = 0.295
    brake_thrust: float = 0.275
    required_corridor_frames: int = 3

    def __post_init__(self) -> None:
        numeric = {
            name: float(value)
            for name, value in vars(self).items()
            if name != "required_corridor_frames"
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise VisualServoRefusal("visual-servo tuning must be finite")
        if not 0.08 <= self.horizontal_corridor <= 0.25:
            raise VisualServoRefusal("horizontal corridor is outside bounds")
        if not 0.08 <= self.vertical_corridor <= 0.28:
            raise VisualServoRefusal("vertical corridor is outside bounds")
        if not 0.55 <= self.edge_brake_x <= 0.85:
            raise VisualServoRefusal("horizontal edge brake is outside bounds")
        if not 0.55 <= self.edge_brake_y <= 0.85:
            raise VisualServoRefusal("vertical edge brake is outside bounds")
        if not 0.10 <= self.stable_rate_norm_s <= 0.60:
            raise VisualServoRefusal("stable image-rate bound is outside bounds")
        if not 0.50 <= self.stable_scale_rate_s <= 1.50:
            raise VisualServoRefusal("stable scale-rate bound is outside bounds")
        if not self.stable_scale_rate_s < self.brake_scale_rate_s <= 3.0:
            raise VisualServoRefusal("scale-rate braking bounds are invalid")
        if not 0.05 <= self.yaw_error_gain <= 0.30:
            raise VisualServoRefusal("yaw error gain is outside bounds")
        if not 0.0 <= self.yaw_rate_gain <= 0.08:
            raise VisualServoRefusal("yaw rate gain is outside bounds")
        if not 0.0 <= self.roll_error_gain <= 0.20:
            raise VisualServoRefusal("roll error gain is outside bounds")
        if not 0.0 <= self.roll_rate_gain <= 0.05:
            raise VisualServoRefusal("roll rate gain is outside bounds")
        if not 0.0 <= self.collective_error_gain <= 0.08:
            raise VisualServoRefusal("collective error gain is outside bounds")
        if not 0.0 <= self.collective_rate_gain <= 0.13:
            raise VisualServoRefusal("collective rate gain is outside bounds")
        if not -0.16 <= self.advance_pitch_rad <= -0.06:
            raise VisualServoRefusal("advance pitch is outside bounds")
        if not 0.0 <= self.brake_pitch_rad <= 0.08:
            raise VisualServoRefusal("brake pitch is outside bounds")
        if not MIN_VISUAL_THRUST <= self.align_thrust <= 0.29:
            raise VisualServoRefusal("alignment thrust is outside bounds")
        if not 0.27 <= self.advance_thrust <= MAX_VISUAL_THRUST:
            raise VisualServoRefusal("advance thrust is outside bounds")
        if not MIN_VISUAL_THRUST <= self.brake_thrust <= 0.29:
            raise VisualServoRefusal("brake thrust is outside bounds")
        if type(self.required_corridor_frames) is not int or not (
            3 <= self.required_corridor_frames <= 8
        ):
            raise VisualServoRefusal("required corridor frames are outside bounds")


VISUAL_CONFIG_SCHEMA = "aigp-vq2-visual-navigation-config/1"
VISUAL_CONTROLLER_FAMILY = "aigp-vq2-dynamic-image-course/1"

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
    # Retain the known-stable pre-credit preview while plant capability is
    # characterized independently of the visual-course lifecycle.
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
