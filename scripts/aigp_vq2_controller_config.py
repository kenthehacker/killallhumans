"""Strict, standalone configuration contract for the bounded VQ2 controller.

This module intentionally has no live-runner imports or side effects.  It
validates controller choices only; lifecycle safety authority remains owned by
``aigp_vq2_run``.  In particular, reset/GO sequencing, stream freshness,
watchdogs, collision and no-passage guards, command pacing, hard powered
deadlines, and cleanup are not configurable here.

The configurable phase timings are *requested* controller timings.  Their
upper bounds are the current hard runtime ceilings, so an integration may
shorten a phase but must continue enforcing the independent runtime deadline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


CONTROLLER_CONFIG_SCHEMA = "aigp-vq2-controller-config/1"
CONTROLLER_FAMILY = "aigp-vq2-gate0-gate1-recenter/1"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "controller_family",
        "phase_timing",
        "turn_cue",
        "roll_control",
        "yaw_control",
        "forward_braking",
    }
)
_GROUP_FIELDS = {
    "phase_timing": frozenset(
        {
            "gate0_boost_until_s",
            "gate0_pitch_blend_s",
            "gate0_yaw_brake_duration_s",
            "post_gate_observation_duration_s",
            "gate1_recenter_duration_s",
        }
    ),
    "turn_cue": frozenset(
        {
            "preturn_enabled",
            "exit_counterroll_enabled",
            "min_gate_area_scale",
            "min_abs_score",
            "preturn_gain",
            "preturn_roll_cap_rad",
            "preturn_taper_area_scale",
            "exit_counterroll_onset_area_scale",
            "exit_counterroll_cap_rad",
        }
    ),
    "roll_control": frozenset(
        {
            "gate0_centering_gain",
            "gate0_target_cap_rad",
            "gate1_error_gain",
            "gate1_error_rate_gain",
            "gate1_target_cap_rad",
            "command_rate_cap_rad_s",
        }
    ),
    "yaw_control": frozenset(
        {
            "gate0_turn_score_gain",
            "gate0_command_rate_cap_rad_s",
            "gate1_error_gain",
            "gate1_deadband_normalized_x",
            "command_rate_cap_rad_s",
        }
    ),
    "forward_braking": frozenset(
        {
            "gate0_turn_pitch_rad",
            "gate0_turn_thrust_cap",
            "gate1_target_pitch_rad",
            "pitch_command_rate_cap_rad_s",
            "gate1_forward_thrust",
        }
    ),
}

CONFIG_FIELD_ALLOWLIST = MappingProxyType(
    {
        "top_level": _TOP_LEVEL_FIELDS,
        **_GROUP_FIELDS,
    }
)

# Bounds never exceed the current build-3385 powered envelopes.  They are part
# of this schema version, not supplied by a caller.
NUMERIC_FIELD_BOUNDS = MappingProxyType(
    {
        "phase_timing.gate0_boost_until_s": (0.45, 1.0),
        "phase_timing.gate0_pitch_blend_s": (0.80, 1.0),
        "phase_timing.gate0_yaw_brake_duration_s": (0.04, 0.21),
        "phase_timing.post_gate_observation_duration_s": (0.10, 0.20),
        "phase_timing.gate1_recenter_duration_s": (0.10, 0.60),
        "turn_cue.min_gate_area_scale": (1.30, 3.50),
        "turn_cue.min_abs_score": (0.04, 0.25),
        "turn_cue.preturn_gain": (0.0, 0.80),
        "turn_cue.preturn_roll_cap_rad": (0.0, 0.13),
        "turn_cue.preturn_taper_area_scale": (3.50, 8.0),
        "turn_cue.exit_counterroll_onset_area_scale": (3.50, 8.0),
        "turn_cue.exit_counterroll_cap_rad": (0.0, 0.08),
        "roll_control.gate0_centering_gain": (0.0, 0.15),
        "roll_control.gate0_target_cap_rad": (0.0, 0.08),
        "roll_control.gate1_error_gain": (-0.24, 0.24),
        "roll_control.gate1_error_rate_gain": (-0.025, 0.025),
        "roll_control.gate1_target_cap_rad": (0.0, 0.12),
        "roll_control.command_rate_cap_rad_s": (0.02, 0.12),
        "yaw_control.gate0_turn_score_gain": (-0.80, 0.0),
        "yaw_control.gate0_command_rate_cap_rad_s": (0.0, 0.08),
        "yaw_control.gate1_error_gain": (-0.12, 0.0),
        "yaw_control.gate1_deadband_normalized_x": (0.20, 0.35),
        "yaw_control.command_rate_cap_rad_s": (0.0, 0.08),
        "forward_braking.gate0_turn_pitch_rad": (0.0, 0.08),
        "forward_braking.gate0_turn_thrust_cap": (0.21, 0.32),
        "forward_braking.gate1_target_pitch_rad": (0.0, 0.10),
        "forward_braking.pitch_command_rate_cap_rad_s": (0.02, 0.12),
        "forward_braking.gate1_forward_thrust": (0.21, 0.30),
    }
)

_DEFAULT_DOCUMENT: dict[str, Any] = {
    "schema": CONTROLLER_CONFIG_SCHEMA,
    "controller_family": CONTROLLER_FAMILY,
    "phase_timing": {
        "gate0_boost_until_s": 0.45,
        "gate0_pitch_blend_s": 0.80,
        "gate0_yaw_brake_duration_s": 0.21,
        "post_gate_observation_duration_s": 0.20,
        "gate1_recenter_duration_s": 0.60,
    },
    "turn_cue": {
        "preturn_enabled": True,
        "exit_counterroll_enabled": True,
        "min_gate_area_scale": 1.30,
        "min_abs_score": 0.04,
        "preturn_gain": 0.80,
        "preturn_roll_cap_rad": 0.13,
        "preturn_taper_area_scale": 8.0,
        "exit_counterroll_onset_area_scale": 3.5,
        "exit_counterroll_cap_rad": 0.08,
    },
    "roll_control": {
        "gate0_centering_gain": 0.15,
        "gate0_target_cap_rad": 0.08,
        "gate1_error_gain": -0.24,
        "gate1_error_rate_gain": 0.0,
        "gate1_target_cap_rad": 0.12,
        "command_rate_cap_rad_s": 0.12,
    },
    "yaw_control": {
        "gate0_turn_score_gain": 0.0,
        "gate0_command_rate_cap_rad_s": 0.0,
        "gate1_error_gain": 0.0,
        "gate1_deadband_normalized_x": 0.35,
        "command_rate_cap_rad_s": 0.0,
    },
    "forward_braking": {
        "gate0_turn_pitch_rad": 0.0,
        "gate0_turn_thrust_cap": 0.32,
        "gate1_target_pitch_rad": 0.10,
        "pitch_command_rate_cap_rad_s": 0.12,
        "gate1_forward_thrust": 0.275,
    },
}


class ControllerConfigError(ValueError):
    """Raised when a controller-config document is not exactly admissible."""


@dataclass(frozen=True)
class PhaseTimingConfig:
    gate0_boost_until_s: float
    gate0_pitch_blend_s: float
    gate0_yaw_brake_duration_s: float
    post_gate_observation_duration_s: float
    gate1_recenter_duration_s: float


@dataclass(frozen=True)
class TurnCueConfig:
    preturn_enabled: bool
    exit_counterroll_enabled: bool
    min_gate_area_scale: float
    min_abs_score: float
    preturn_gain: float
    preturn_roll_cap_rad: float
    preturn_taper_area_scale: float
    exit_counterroll_onset_area_scale: float
    exit_counterroll_cap_rad: float


@dataclass(frozen=True)
class RollControlConfig:
    gate0_centering_gain: float
    gate0_target_cap_rad: float
    gate1_error_gain: float
    gate1_error_rate_gain: float
    gate1_target_cap_rad: float
    command_rate_cap_rad_s: float


@dataclass(frozen=True)
class YawControlConfig:
    gate0_turn_score_gain: float
    gate0_command_rate_cap_rad_s: float
    gate1_error_gain: float
    gate1_deadband_normalized_x: float
    command_rate_cap_rad_s: float


@dataclass(frozen=True)
class ForwardBrakingConfig:
    gate0_turn_pitch_rad: float
    gate0_turn_thrust_cap: float
    gate1_target_pitch_rad: float
    pitch_command_rate_cap_rad_s: float
    gate1_forward_thrust: float


@dataclass(frozen=True)
class VQ2ControllerConfig:
    """Normalized, deeply immutable effective controller configuration."""

    schema: str
    controller_family: str
    phase_timing: PhaseTimingConfig
    turn_cue: TurnCueConfig
    roll_control: RollControlConfig
    yaw_control: YawControlConfig
    forward_braking: ForwardBrakingConfig

    def to_effective_mapping(self) -> dict[str, Any]:
        """Return the complete normalized mapping used for identity hashing."""

        return asdict(self)

    @property
    def effective_config_sha256(self) -> str:
        return _canonical_sha256(self.to_effective_mapping())


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


DEFAULT_CONTROLLER_CONFIG: Mapping[str, Any] = _deep_freeze(_DEFAULT_DOCUMENT)


def default_controller_config_mapping() -> dict[str, Any]:
    """Return a fresh mutable copy of the current-behavior default document."""

    return json.loads(
        json.dumps(
            _DEFAULT_DOCUMENT,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(
    value: object,
    *,
    expected_fields: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControllerConfigError(f"{path} must be a mapping")
    actual = set(value)
    missing = sorted(expected_fields - actual)
    unknown = sorted(actual - expected_fields, key=repr)
    if missing or unknown:
        raise ControllerConfigError(
            f"{path} must contain the exact field allowlist "
            f"(missing={missing!r}, unknown={unknown!r})"
        )
    return value


def _require_exact_string(value: object, expected: str, *, path: str) -> str:
    if type(value) is not str or value != expected:
        raise ControllerConfigError(f"{path} must equal {expected!r}")
    return value


def _require_bool(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        raise ControllerConfigError(f"{path} must be an exact bool")
    return value


def _require_number(value: object, *, path: str) -> float:
    if type(value) not in {int, float}:
        raise ControllerConfigError(f"{path} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ControllerConfigError(f"{path} must be a finite number")
    minimum, maximum = NUMERIC_FIELD_BOUNDS[path]
    if not minimum <= normalized <= maximum:
        raise ControllerConfigError(
            f"{path} must be within [{minimum}, {maximum}]"
        )
    return normalized


def validate_controller_config(document: object) -> VQ2ControllerConfig:
    """Validate and normalize one complete exact-schema config document."""

    root = _require_mapping(
        document,
        expected_fields=_TOP_LEVEL_FIELDS,
        path="controller_config",
    )
    schema = _require_exact_string(
        root["schema"],
        CONTROLLER_CONFIG_SCHEMA,
        path="controller_config.schema",
    )
    family = _require_exact_string(
        root["controller_family"],
        CONTROLLER_FAMILY,
        path="controller_config.controller_family",
    )
    groups = {
        name: _require_mapping(
            root[name],
            expected_fields=fields,
            path=f"controller_config.{name}",
        )
        for name, fields in _GROUP_FIELDS.items()
    }

    phase_value = groups["phase_timing"]
    phase = PhaseTimingConfig(
        **{
            field: _require_number(
                phase_value[field],
                path=f"phase_timing.{field}",
            )
            for field in _GROUP_FIELDS["phase_timing"]
        }
    )

    turn_value = groups["turn_cue"]
    turn = TurnCueConfig(
        preturn_enabled=_require_bool(
            turn_value["preturn_enabled"],
            path="turn_cue.preturn_enabled",
        ),
        exit_counterroll_enabled=_require_bool(
            turn_value["exit_counterroll_enabled"],
            path="turn_cue.exit_counterroll_enabled",
        ),
        min_gate_area_scale=_require_number(
            turn_value["min_gate_area_scale"],
            path="turn_cue.min_gate_area_scale",
        ),
        min_abs_score=_require_number(
            turn_value["min_abs_score"],
            path="turn_cue.min_abs_score",
        ),
        preturn_gain=_require_number(
            turn_value["preturn_gain"],
            path="turn_cue.preturn_gain",
        ),
        preturn_roll_cap_rad=_require_number(
            turn_value["preturn_roll_cap_rad"],
            path="turn_cue.preturn_roll_cap_rad",
        ),
        preturn_taper_area_scale=_require_number(
            turn_value["preturn_taper_area_scale"],
            path="turn_cue.preturn_taper_area_scale",
        ),
        exit_counterroll_onset_area_scale=_require_number(
            turn_value["exit_counterroll_onset_area_scale"],
            path="turn_cue.exit_counterroll_onset_area_scale",
        ),
        exit_counterroll_cap_rad=_require_number(
            turn_value["exit_counterroll_cap_rad"],
            path="turn_cue.exit_counterroll_cap_rad",
        ),
    )

    roll_value = groups["roll_control"]
    roll = RollControlConfig(
        **{
            field: _require_number(
                roll_value[field],
                path=f"roll_control.{field}",
            )
            for field in _GROUP_FIELDS["roll_control"]
        }
    )

    yaw_value = groups["yaw_control"]
    yaw = YawControlConfig(
        **{
            field: _require_number(
                yaw_value[field],
                path=f"yaw_control.{field}",
            )
            for field in _GROUP_FIELDS["yaw_control"]
        }
    )

    braking_value = groups["forward_braking"]
    braking = ForwardBrakingConfig(
        **{
            field: _require_number(
                braking_value[field],
                path=f"forward_braking.{field}",
            )
            for field in _GROUP_FIELDS["forward_braking"]
        }
    )

    if turn.exit_counterroll_enabled and not turn.preturn_enabled:
        raise ControllerConfigError(
            "turn_cue.exit_counterroll_enabled requires preturn_enabled"
        )
    if not (
        turn.min_gate_area_scale
        <= turn.exit_counterroll_onset_area_scale
        <= turn.preturn_taper_area_scale
    ):
        raise ControllerConfigError(
            "turn-cue area scales must satisfy "
            "min_gate_area_scale <= exit_counterroll_onset_area_scale "
            "<= preturn_taper_area_scale"
        )
    if (
        phase.post_gate_observation_duration_s
        + phase.gate1_recenter_duration_s
        > 0.80 + 1e-12
    ):
        raise ControllerConfigError(
            "requested post-gate controller phases exceed the fixed 0.80s "
            "combined ceiling"
        )
    if (yaw.gate0_turn_score_gain == 0.0) != (
        yaw.gate0_command_rate_cap_rad_s == 0.0
    ):
        raise ControllerConfigError(
            "yaw_control Gate-0 gain and command-rate cap must be zero or "
            "nonzero together"
        )
    if (yaw.gate1_error_gain == 0.0) != (
        yaw.command_rate_cap_rad_s == 0.0
    ):
        raise ControllerConfigError(
            "yaw_control gain and command-rate cap must be zero or nonzero "
            "together"
        )
    return VQ2ControllerConfig(
        schema=schema,
        controller_family=family,
        phase_timing=phase,
        turn_cue=turn,
        roll_control=roll,
        yaw_control=yaw,
        forward_braking=braking,
    )


def default_controller_config() -> VQ2ControllerConfig:
    """Return the validated effective config matching current runner behavior."""

    return validate_controller_config(DEFAULT_CONTROLLER_CONFIG)


def canonical_effective_config_sha256(
    value: object,
) -> str:
    """Return the canonical identity of a mapping or validated config."""

    config = (
        value
        if isinstance(value, VQ2ControllerConfig)
        else validate_controller_config(value)
    )
    return config.effective_config_sha256


__all__ = [
    "CONFIG_FIELD_ALLOWLIST",
    "CONTROLLER_CONFIG_SCHEMA",
    "CONTROLLER_FAMILY",
    "ControllerConfigError",
    "DEFAULT_CONTROLLER_CONFIG",
    "ForwardBrakingConfig",
    "NUMERIC_FIELD_BOUNDS",
    "PhaseTimingConfig",
    "RollControlConfig",
    "TurnCueConfig",
    "VQ2ControllerConfig",
    "YawControlConfig",
    "canonical_effective_config_sha256",
    "default_controller_config",
    "default_controller_config_mapping",
    "validate_controller_config",
]
