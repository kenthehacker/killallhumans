"""Deterministic offline body-rate system identification for VQ2.

This module owns no collection, simulator, socket, runner, supervisor,
scheduler, approval, or command-send capability.  It accepts already recorded
host-monotonic samples and fits one deliberately small model per body-rate
axis::

    command -- zero-order hold -- delay -- first-order lag/gain --> rate
                                                               + gyro bias

Delay and time constant are selected from reviewed finite grids using only a
chronological training partition.  Gain and bias are conditional linear least
squares parameters.  A later, disjoint gyro partition is then evaluated once
as a free run; validation data never selects or changes the model.  Synthetic
recovery tests establish implementation behavior only, not measured plant
truth or authority for a powered experiment.

The immutable dataclasses below are local offline-analysis contracts.  They
are deliberately not wire schemas and do not amend any frozen VQ2 interface.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional

import numpy as np


_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_MINIMUM_COMMAND_PERIOD_NS = 20_000_000
_MAXIMUM_EXPERIMENT_DURATION_NS = 10_000_000_000
_MINIMUM_FINAL_ZERO_SETTLING_NS = 500_000_000
_MAXIMUM_SIGNED_PREFIX_ANGLE_RAD = 0.20
_MAXIMUM_ADJACENT_RATE_STEP_RAD_S = 0.20
_VQ2_MAX_ROLL_PITCH_RATE_RAD_S = 0.25
_CONFIG_POLICY_SCHEMA = "vq2-rate-system-id-policy-v1"

_DEFAULT_DELAY_CANDIDATES_NS = tuple(range(0, 100_000_001, 5_000_000))
_DEFAULT_TIME_CONSTANT_CANDIDATES_S = tuple(
    value / 1_000.0 for value in range(20, 301, 10)
)
_DEFAULT_MINIMUM_TRAINING_SAMPLES = 80
_DEFAULT_MINIMUM_VALIDATION_SAMPLES = 40
_DEFAULT_MINIMUM_TRAINING_DURATION_S = 2.0
_DEFAULT_MINIMUM_VALIDATION_DURATION_S = 1.0
_DEFAULT_MAXIMUM_GYRO_GAP_S = 0.050
_DEFAULT_MAXIMUM_ABS_COMMAND_RATE_RAD_S = 0.25
_DEFAULT_MAXIMUM_ABS_GYRO_RATE_RAD_S = 4.0
_DEFAULT_MINIMUM_COMMAND_SPAN_RAD_S = 0.10
_DEFAULT_MINIMUM_COMMAND_STANDARD_DEVIATION_RAD_S = 0.025
_DEFAULT_MINIMUM_OUTPUT_STANDARD_DEVIATION_RAD_S = 0.010
_DEFAULT_MAXIMUM_DESIGN_CONDITION_NUMBER = 1.0e5
_DEFAULT_MINIMUM_GAIN = 0.05
_DEFAULT_MAXIMUM_GAIN = 2.0
_DEFAULT_MAXIMUM_ABS_BIAS_RAD_S = 0.30
_DEFAULT_PROFILE_DELTA_SIGMA2 = 3.841458820694124
_DEFAULT_MINIMUM_RESIDUAL_VARIANCE = 1.0e-12
_DEFAULT_MAXIMUM_DELAY_UNCERTAINTY_NS = 25_000_000
_DEFAULT_MAXIMUM_TIME_CONSTANT_UNCERTAINTY_S = 0.060
_DEFAULT_MAXIMUM_VALIDATION_NORMALIZED_RMSE = 0.35
_DEFAULT_MINIMUM_VALIDATION_IMPROVEMENT_FRACTION = 0.10


class SystemIdentificationError(ValueError):
    """Base class for deterministic fail-closed identification errors."""


class TraceValidationError(SystemIdentificationError):
    """Raised when timestamped evidence violates its local strict contract."""


class IdentifiabilityError(SystemIdentificationError):
    """Raised when training data cannot identify the reviewed model."""


class HeldOutValidationError(SystemIdentificationError):
    """Raised when the training-only model fails later held-out evidence."""


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded ASCII token")
    return value


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _exact_positive_int(value: object, label: str) -> int:
    result = _exact_nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_float(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_positive(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


class RateAxis(str, Enum):
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"


@dataclass(frozen=True, slots=True)
class RateCommandSample:
    """One inert commanded-rate record on an identified host clock.

    ``monotonic_ns`` is the recorded command-reference timestamp used by the
    offline fit.  Constructing this value does not approve, schedule, or send a
    command.  ``saturated`` preserves a disqualifying observation instead of
    silently clipping or dropping it.
    """

    host_clock_id: str
    sequence: int
    monotonic_ns: int
    axis: RateAxis
    commanded_rate_rad_s: float
    saturated: bool = False

    def __post_init__(self) -> None:
        _bounded_token(self.host_clock_id, "host_clock_id")
        _exact_nonnegative_int(self.sequence, "command sequence")
        _exact_nonnegative_int(self.monotonic_ns, "command monotonic_ns")
        if type(self.axis) is not RateAxis:
            raise TypeError("axis must be RateAxis")
        value = _finite_float(
            self.commanded_rate_rad_s, "commanded_rate_rad_s"
        )
        _exact_bool(self.saturated, "command saturated")
        object.__setattr__(self, "commanded_rate_rad_s", value)


@dataclass(frozen=True, slots=True)
class GyroRateSample:
    """One host-monotonic gyro-axis record; missing values are unrepresentable."""

    host_clock_id: str
    sequence: int
    monotonic_ns: int
    axis: RateAxis
    measured_rate_rad_s: float
    clipped: bool = False

    def __post_init__(self) -> None:
        _bounded_token(self.host_clock_id, "host_clock_id")
        _exact_nonnegative_int(self.sequence, "gyro sequence")
        _exact_nonnegative_int(self.monotonic_ns, "gyro monotonic_ns")
        if type(self.axis) is not RateAxis:
            raise TypeError("axis must be RateAxis")
        value = _finite_float(self.measured_rate_rad_s, "measured_rate_rad_s")
        _exact_bool(self.clipped, "gyro clipped")
        object.__setattr__(self, "measured_rate_rad_s", value)


def _validate_sample_stream(
    samples: tuple[object, ...],
    expected_type: type,
    *,
    host_clock_id: str,
    axis: RateAxis,
    label: str,
) -> None:
    if type(samples) is not tuple:
        raise TypeError(f"{label} samples must be an exact tuple")
    if not samples:
        raise TraceValidationError(f"{label} samples cannot be empty")
    previous_sequence: Optional[int] = None
    previous_time: Optional[int] = None
    for sample in samples:
        if type(sample) is not expected_type:
            raise TypeError(f"{label} stream contains a non-{expected_type.__name__}")
        if sample.host_clock_id != host_clock_id:
            raise TraceValidationError(f"{label} sample changed host clock")
        if sample.axis is not axis:
            raise TraceValidationError(f"{label} sample changed rate axis")
        if previous_sequence is not None and sample.sequence != previous_sequence + 1:
            raise TraceValidationError(
                f"{label} sequence must be contiguous; missing/replayed sample"
            )
        if previous_time is not None and sample.monotonic_ns <= previous_time:
            raise TraceValidationError(
                f"{label} timestamps must advance strictly"
            )
        previous_sequence = sample.sequence
        previous_time = sample.monotonic_ns


@dataclass(frozen=True, slots=True)
class RateAxisTrace:
    """Strict one-clock, one-axis command/gyro trace."""

    host_clock_id: str
    axis: RateAxis
    commands: tuple[RateCommandSample, ...]
    gyro: tuple[GyroRateSample, ...]

    def __post_init__(self) -> None:
        _bounded_token(self.host_clock_id, "host_clock_id")
        if type(self.axis) is not RateAxis:
            raise TypeError("axis must be RateAxis")
        _validate_sample_stream(
            self.commands,
            RateCommandSample,
            host_clock_id=self.host_clock_id,
            axis=self.axis,
            label="command",
        )
        _validate_sample_stream(
            self.gyro,
            GyroRateSample,
            host_clock_id=self.host_clock_id,
            axis=self.axis,
            label="gyro",
        )


@dataclass(frozen=True, slots=True)
class IdentificationWindow:
    """Inclusive host-monotonic outcome window."""

    start_monotonic_ns: int
    end_monotonic_ns: int

    def __post_init__(self) -> None:
        start = _exact_nonnegative_int(
            self.start_monotonic_ns, "window start_monotonic_ns"
        )
        end = _exact_nonnegative_int(
            self.end_monotonic_ns, "window end_monotonic_ns"
        )
        if end <= start:
            raise ValueError("identification window must have positive duration")


@dataclass(frozen=True, slots=True)
class ChronologicalHoldout:
    """Training then strictly later validation; overlap is unrepresentable."""

    training: IdentificationWindow
    validation: IdentificationWindow

    def __post_init__(self) -> None:
        if type(self.training) is not IdentificationWindow:
            raise TypeError("training must be IdentificationWindow")
        if type(self.validation) is not IdentificationWindow:
            raise TypeError("validation must be IdentificationWindow")
        if self.training.end_monotonic_ns >= self.validation.start_monotonic_ns:
            raise TraceValidationError(
                "training and validation windows must be disjoint and chronological"
            )


@dataclass(frozen=True, slots=True)
class SystemIdConfig:
    """Reviewed finite search and tighten-only identifiability bounds.

    Candidate grids are pinned to the reviewed full hard domains so a caller
    cannot truncate or coarsen a profile and report artificially narrow
    uncertainty.  Every other override must be at least as conservative as
    the default.  :attr:`semantic_identity` covers every dataclass field plus
    the local policy schema using canonical JSON and SHA-256; it never uses
    Python's process-randomized ``hash``.
    """

    delay_candidates_ns: tuple[int, ...] = _DEFAULT_DELAY_CANDIDATES_NS
    time_constant_candidates_s: tuple[float, ...] = (
        _DEFAULT_TIME_CONSTANT_CANDIDATES_S
    )
    minimum_training_samples: int = _DEFAULT_MINIMUM_TRAINING_SAMPLES
    minimum_validation_samples: int = _DEFAULT_MINIMUM_VALIDATION_SAMPLES
    minimum_training_duration_s: float = _DEFAULT_MINIMUM_TRAINING_DURATION_S
    minimum_validation_duration_s: float = _DEFAULT_MINIMUM_VALIDATION_DURATION_S
    maximum_gyro_gap_s: float = _DEFAULT_MAXIMUM_GYRO_GAP_S
    maximum_abs_command_rate_rad_s: float = (
        _DEFAULT_MAXIMUM_ABS_COMMAND_RATE_RAD_S
    )
    maximum_abs_gyro_rate_rad_s: float = _DEFAULT_MAXIMUM_ABS_GYRO_RATE_RAD_S
    minimum_command_span_rad_s: float = _DEFAULT_MINIMUM_COMMAND_SPAN_RAD_S
    minimum_command_standard_deviation_rad_s: float = (
        _DEFAULT_MINIMUM_COMMAND_STANDARD_DEVIATION_RAD_S
    )
    minimum_output_standard_deviation_rad_s: float = (
        _DEFAULT_MINIMUM_OUTPUT_STANDARD_DEVIATION_RAD_S
    )
    maximum_design_condition_number: float = (
        _DEFAULT_MAXIMUM_DESIGN_CONDITION_NUMBER
    )
    minimum_gain: float = _DEFAULT_MINIMUM_GAIN
    maximum_gain: float = _DEFAULT_MAXIMUM_GAIN
    maximum_abs_bias_rad_s: float = _DEFAULT_MAXIMUM_ABS_BIAS_RAD_S
    profile_delta_sigma2: float = _DEFAULT_PROFILE_DELTA_SIGMA2
    minimum_residual_variance: float = _DEFAULT_MINIMUM_RESIDUAL_VARIANCE
    maximum_delay_uncertainty_ns: int = _DEFAULT_MAXIMUM_DELAY_UNCERTAINTY_NS
    maximum_time_constant_uncertainty_s: float = (
        _DEFAULT_MAXIMUM_TIME_CONSTANT_UNCERTAINTY_S
    )
    maximum_validation_normalized_rmse: float = (
        _DEFAULT_MAXIMUM_VALIDATION_NORMALIZED_RMSE
    )
    minimum_validation_improvement_fraction: float = (
        _DEFAULT_MINIMUM_VALIDATION_IMPROVEMENT_FRACTION
    )

    def __post_init__(self) -> None:
        if type(self.delay_candidates_ns) is not tuple:
            raise TypeError("delay_candidates_ns must be an exact tuple")
        if len(self.delay_candidates_ns) < 3:
            raise ValueError("delay search requires at least three candidates")
        delays = tuple(
            _exact_nonnegative_int(value, f"delay_candidates_ns[{index}]")
            for index, value in enumerate(self.delay_candidates_ns)
        )
        if any(later <= earlier for earlier, later in zip(delays, delays[1:])):
            raise ValueError("delay candidates must increase strictly")
        if delays != _DEFAULT_DELAY_CANDIDATES_NS:
            raise ValueError(
                "delay_candidates_ns is pinned to the reviewed 0..100 ms grid"
            )
        if type(self.time_constant_candidates_s) is not tuple:
            raise TypeError("time_constant_candidates_s must be an exact tuple")
        if len(self.time_constant_candidates_s) < 3:
            raise ValueError("time-constant search requires at least three candidates")
        time_constants = tuple(
            _finite_positive(value, f"time_constant_candidates_s[{index}]")
            for index, value in enumerate(self.time_constant_candidates_s)
        )
        if any(
            later <= earlier
            for earlier, later in zip(time_constants, time_constants[1:])
        ):
            raise ValueError("time-constant candidates must increase strictly")
        if time_constants != _DEFAULT_TIME_CONSTANT_CANDIDATES_S:
            raise ValueError(
                "time_constant_candidates_s is pinned to the reviewed "
                "0.020..0.300 s grid"
            )

        integer_floors = {
            "minimum_training_samples": _DEFAULT_MINIMUM_TRAINING_SAMPLES,
            "minimum_validation_samples": _DEFAULT_MINIMUM_VALIDATION_SAMPLES,
        }
        for name, reviewed_minimum in integer_floors.items():
            value = _exact_positive_int(getattr(self, name), name)
            if value < reviewed_minimum:
                raise ValueError(
                    f"{name} cannot be lower than the reviewed default"
                )
        delay_uncertainty = _exact_positive_int(
            self.maximum_delay_uncertainty_ns,
            "maximum_delay_uncertainty_ns",
        )
        if delay_uncertainty > _DEFAULT_MAXIMUM_DELAY_UNCERTAINTY_NS:
            raise ValueError(
                "maximum_delay_uncertainty_ns cannot exceed the reviewed default"
            )

        float_names = (
            "minimum_training_duration_s",
            "minimum_validation_duration_s",
            "maximum_gyro_gap_s",
            "maximum_abs_command_rate_rad_s",
            "maximum_abs_gyro_rate_rad_s",
            "minimum_command_span_rad_s",
            "minimum_command_standard_deviation_rad_s",
            "minimum_output_standard_deviation_rad_s",
            "maximum_design_condition_number",
            "minimum_gain",
            "maximum_gain",
            "maximum_abs_bias_rad_s",
            "profile_delta_sigma2",
            "minimum_residual_variance",
            "maximum_time_constant_uncertainty_s",
            "maximum_validation_normalized_rmse",
            "minimum_validation_improvement_fraction",
        )
        for name in float_names:
            object.__setattr__(
                self, name, _finite_positive(getattr(self, name), name)
            )

        float_floors = {
            "minimum_training_duration_s": _DEFAULT_MINIMUM_TRAINING_DURATION_S,
            "minimum_validation_duration_s": (
                _DEFAULT_MINIMUM_VALIDATION_DURATION_S
            ),
            "minimum_command_span_rad_s": _DEFAULT_MINIMUM_COMMAND_SPAN_RAD_S,
            "minimum_command_standard_deviation_rad_s": (
                _DEFAULT_MINIMUM_COMMAND_STANDARD_DEVIATION_RAD_S
            ),
            "minimum_output_standard_deviation_rad_s": (
                _DEFAULT_MINIMUM_OUTPUT_STANDARD_DEVIATION_RAD_S
            ),
            "minimum_gain": _DEFAULT_MINIMUM_GAIN,
            "minimum_residual_variance": _DEFAULT_MINIMUM_RESIDUAL_VARIANCE,
            "minimum_validation_improvement_fraction": (
                _DEFAULT_MINIMUM_VALIDATION_IMPROVEMENT_FRACTION
            ),
        }
        for name, reviewed_minimum in float_floors.items():
            if getattr(self, name) < reviewed_minimum:
                raise ValueError(
                    f"{name} cannot be lower than the reviewed default"
                )

        float_caps = {
            "maximum_gyro_gap_s": _DEFAULT_MAXIMUM_GYRO_GAP_S,
            "maximum_abs_command_rate_rad_s": (
                _DEFAULT_MAXIMUM_ABS_COMMAND_RATE_RAD_S
            ),
            "maximum_abs_gyro_rate_rad_s": _DEFAULT_MAXIMUM_ABS_GYRO_RATE_RAD_S,
            "maximum_design_condition_number": (
                _DEFAULT_MAXIMUM_DESIGN_CONDITION_NUMBER
            ),
            "maximum_gain": _DEFAULT_MAXIMUM_GAIN,
            "maximum_abs_bias_rad_s": _DEFAULT_MAXIMUM_ABS_BIAS_RAD_S,
            "maximum_time_constant_uncertainty_s": (
                _DEFAULT_MAXIMUM_TIME_CONSTANT_UNCERTAINTY_S
            ),
            "maximum_validation_normalized_rmse": (
                _DEFAULT_MAXIMUM_VALIDATION_NORMALIZED_RMSE
            ),
        }
        for name, reviewed_maximum in float_caps.items():
            if getattr(self, name) > reviewed_maximum:
                raise ValueError(
                    f"{name} cannot exceed the reviewed default"
                )

        if self.profile_delta_sigma2 != _DEFAULT_PROFILE_DELTA_SIGMA2:
            raise ValueError(
                "profile_delta_sigma2 is pinned to the reviewed 95% cutoff"
            )
        if self.minimum_gain >= self.maximum_gain:
            raise ValueError("minimum_gain must be less than maximum_gain")
        if self.maximum_design_condition_number <= 1.0:
            raise ValueError("maximum_design_condition_number must exceed one")
        if self.minimum_validation_improvement_fraction >= 1.0:
            raise ValueError("minimum_validation_improvement_fraction must be < 1")
        if (
            self.minimum_command_span_rad_s
            > 2.0 * self.maximum_abs_command_rate_rad_s
        ):
            raise ValueError("command span floor exceeds the retained command range")
        if (
            self.minimum_command_standard_deviation_rad_s
            > self.maximum_abs_command_rate_rad_s
        ):
            raise ValueError(
                "command deviation floor exceeds the retained command range"
            )
        if (
            self.minimum_output_standard_deviation_rad_s
            > self.maximum_abs_gyro_rate_rad_s
        ):
            raise ValueError("output deviation floor exceeds the gyro bound")
        if self.maximum_abs_bias_rad_s > self.maximum_abs_gyro_rate_rad_s:
            raise ValueError("bias bound exceeds the retained gyro bound")
        object.__setattr__(self, "delay_candidates_ns", delays)
        object.__setattr__(self, "time_constant_candidates_s", time_constants)

    @property
    def semantic_identity(self) -> str:
        """Canonical identity for every reviewed config field and policy schema."""

        payload = {
            "config": {
                field.name: getattr(self, field.name) for field in fields(self)
            },
            "policy_schema": _CONFIG_POLICY_SCHEMA,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest = hashlib.sha256(encoded).hexdigest()
        return f"{_CONFIG_POLICY_SCHEMA}:sha256:{digest}"


@dataclass(frozen=True, slots=True)
class RateExperimentSegment:
    """One relative-duration element of an inert experiment definition."""

    duration_ns: int
    commanded_rate_rad_s: float

    def __post_init__(self) -> None:
        _exact_positive_int(self.duration_ns, "segment duration_ns")
        value = _finite_float(
            self.commanded_rate_rad_s, "segment commanded_rate_rad_s"
        )
        object.__setattr__(self, "commanded_rate_rad_s", value)


@dataclass(frozen=True, slots=True)
class RateExperimentDefinition:
    """Conservatively bounded inert data with no execution/conversion API.

    Zero net command area is only a definition-shape constraint.  A lagged
    biased plant need not return to its initial attitude, so this class never
    calls the definition attitude-restoring, safe to execute, or approved.
    """

    experiment_id: str
    axis: RateAxis
    command_period_ns: int
    segments: tuple[RateExperimentSegment, ...]

    def __post_init__(self) -> None:
        _bounded_token(self.experiment_id, "experiment_id")
        if type(self.axis) is not RateAxis:
            raise TypeError("axis must be RateAxis")
        if self.axis is RateAxis.YAW:
            raise ValueError(
                "nonzero yaw experiments are outside the frozen VQ2 safety contract"
            )
        period = _exact_positive_int(self.command_period_ns, "command_period_ns")
        if period < _MINIMUM_COMMAND_PERIOD_NS:
            raise ValueError("experiment command period cannot exceed the 50 Hz cap")
        if type(self.segments) is not tuple:
            raise TypeError("segments must be an exact tuple")
        if not 4 <= len(self.segments) <= 64:
            raise ValueError("experiment requires 4..64 bounded segments")
        for segment in self.segments:
            if type(segment) is not RateExperimentSegment:
                raise TypeError("segments contains a non-RateExperimentSegment")
            if segment.duration_ns % period != 0:
                raise ValueError("segment duration must be a command-period multiple")
            if (
                abs(segment.commanded_rate_rad_s)
                > _VQ2_MAX_ROLL_PITCH_RATE_RAD_S
            ):
                raise ValueError("experiment rate exceeds frozen VQ2 envelope")
        total_duration = sum(segment.duration_ns for segment in self.segments)
        if total_duration > _MAXIMUM_EXPERIMENT_DURATION_NS:
            raise ValueError("experiment exceeds the inert ten-second definition bound")
        if self.segments[0].commanded_rate_rad_s != 0.0:
            raise ValueError("experiment must begin at exact zero rate")
        if self.segments[-1].commanded_rate_rad_s != 0.0:
            raise ValueError("experiment must end at exact zero rate")
        if self.final_zero_settling_duration_ns < _MINIMUM_FINAL_ZERO_SETTLING_NS:
            raise ValueError(
                "experiment requires at least 0.5 s of final exact-zero settling"
            )
        values = tuple(segment.commanded_rate_rad_s for segment in self.segments)
        if any(
            abs(later - earlier) > _MAXIMUM_ADJACENT_RATE_STEP_RAD_S
            for earlier, later in zip(values, values[1:])
        ):
            raise ValueError("experiment adjacent rate step exceeds 0.20 rad/s")
        if not any(value > 0.0 for value in values) or not any(
            value < 0.0 for value in values
        ):
            raise ValueError("experiment must contain positive and negative excitation")
        segment_areas_rad = tuple(
            segment.commanded_rate_rad_s * (segment.duration_ns / 1e9)
            for segment in self.segments
        )
        prefix_areas_rad = tuple(
            math.fsum(segment_areas_rad[: index + 1])
            for index in range(len(segment_areas_rad))
        )
        if max(abs(value) for value in prefix_areas_rad) > (
            _MAXIMUM_SIGNED_PREFIX_ANGLE_RAD
        ):
            raise ValueError(
                "experiment signed prefix angle exceeds the inert 0.20 rad bound"
            )
        signed_area = math.fsum(segment_areas_rad)
        area_scale = max(
            1.0,
            math.fsum(abs(value) for value in segment_areas_rad),
        )
        if abs(signed_area) > 1e-12 * area_scale:
            raise ValueError("experiment must have zero net rate-command area")

    @property
    def total_duration_ns(self) -> int:
        return sum(segment.duration_ns for segment in self.segments)

    @property
    def maximum_abs_rate_rad_s(self) -> float:
        return max(abs(segment.commanded_rate_rad_s) for segment in self.segments)

    @property
    def maximum_signed_prefix_angle_rad(self) -> float:
        prefix = 0.0
        maximum = 0.0
        for segment in self.segments:
            prefix = math.fsum(
                (prefix, segment.commanded_rate_rad_s * segment.duration_ns / 1e9)
            )
            maximum = max(maximum, abs(prefix))
        return maximum

    @property
    def final_zero_settling_duration_ns(self) -> int:
        duration_ns = 0
        for segment in reversed(self.segments):
            if segment.commanded_rate_rad_s != 0.0:
                break
            duration_ns += segment.duration_ns
        return duration_ns

    @property
    def maximum_adjacent_rate_step_rad_s(self) -> float:
        values = tuple(segment.commanded_rate_rad_s for segment in self.segments)
        return max(
            abs(later - earlier) for earlier, later in zip(values, values[1:])
        )


@dataclass(frozen=True, slots=True)
class RateAxisModel:
    model_id: str
    config_semantic_id: str
    host_clock_id: str
    axis: RateAxis
    delay_ns: int
    time_constant_s: float
    steady_state_gain: float
    gyro_bias_rad_s: float

    def __post_init__(self) -> None:
        _bounded_token(self.model_id, "model_id")
        _bounded_token(self.config_semantic_id, "config_semantic_id")
        _bounded_token(self.host_clock_id, "host_clock_id")
        if type(self.axis) is not RateAxis:
            raise TypeError("axis must be RateAxis")
        _exact_nonnegative_int(self.delay_ns, "delay_ns")
        _finite_positive(self.time_constant_s, "time_constant_s")
        _finite_positive(self.steady_state_gain, "steady_state_gain")
        _finite_float(self.gyro_bias_rad_s, "gyro_bias_rad_s")


@dataclass(frozen=True, slots=True)
class RateAxisModelUncertainty:
    """Conditional/profile uncertainty; not a plant guarantee.

    Delay and time-constant intervals are fixed from training evidence only.
    Held-out residual variance may conservatively inflate the conditional
    gain/bias covariance after selection, but cannot change fitted parameters
    or either profile interval.
    """

    config_semantic_id: str
    method_id: str
    confidence_level: float
    delay_interval_ns: tuple[int, int]
    time_constant_interval_s: tuple[float, float]
    gain_standard_error: float
    bias_standard_error_rad_s: float
    gain_bias_covariance: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        _bounded_token(self.config_semantic_id, "config_semantic_id")
        _bounded_token(self.method_id, "uncertainty method_id")
        confidence = _finite_float(self.confidence_level, "confidence_level")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if (
            type(self.delay_interval_ns) is not tuple
            or len(self.delay_interval_ns) != 2
        ):
            raise TypeError("delay_interval_ns must be an exact pair")
        delay_low = _exact_nonnegative_int(
            self.delay_interval_ns[0], "delay interval lower"
        )
        delay_high = _exact_nonnegative_int(
            self.delay_interval_ns[1], "delay interval upper"
        )
        if delay_high < delay_low:
            raise ValueError("delay interval is reversed")
        if (
            type(self.time_constant_interval_s) is not tuple
            or len(self.time_constant_interval_s) != 2
        ):
            raise TypeError("time_constant_interval_s must be an exact pair")
        tau_low = _finite_positive(
            self.time_constant_interval_s[0], "time constant interval lower"
        )
        tau_high = _finite_positive(
            self.time_constant_interval_s[1], "time constant interval upper"
        )
        if tau_high < tau_low:
            raise ValueError("time constant interval is reversed")
        gain_error = _finite_float(
            self.gain_standard_error, "gain_standard_error"
        )
        bias_error = _finite_float(
            self.bias_standard_error_rad_s, "bias_standard_error_rad_s"
        )
        if gain_error < 0.0 or bias_error < 0.0:
            raise ValueError("standard errors must be nonnegative")
        if (
            type(self.gain_bias_covariance) is not tuple
            or len(self.gain_bias_covariance) != 2
            or any(
                type(row) is not tuple or len(row) != 2
                for row in self.gain_bias_covariance
            )
        ):
            raise TypeError("gain_bias_covariance must be an exact 2x2 tuple")
        covariance_values = tuple(
            tuple(
                _finite_float(value, f"gain_bias_covariance[{row}][{column}]")
                for column, value in enumerate(values)
            )
            for row, values in enumerate(self.gain_bias_covariance)
        )
        covariance = np.asarray(covariance_values, dtype=np.float64)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("gain/bias covariance must be finite")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("gain/bias covariance must be symmetric")
        if float(np.linalg.eigvalsh(covariance)[0]) < -1e-12:
            raise ValueError("gain/bias covariance must be positive semidefinite")
        object.__setattr__(self, "gain_bias_covariance", covariance_values)


@dataclass(frozen=True, slots=True)
class RateAxisFitDiagnostics:
    config_semantic_id: str
    selection_basis: str
    validation_basis: str
    training_scored_samples: int
    validation_scored_samples: int
    unused_gyro_samples: int
    evaluated_candidates: int
    command_span_rad_s: float
    command_standard_deviation_rad_s: float
    validation_command_span_rad_s: float
    training_rmse_rad_s: float
    validation_rmse_rad_s: float
    validation_mae_rad_s: float
    validation_normalized_rmse: float
    validation_initial_value_hold_rmse_rad_s: float
    validation_improvement_fraction: float
    design_condition_number: float
    residual_standard_deviation_rad_s: float

    def __post_init__(self) -> None:
        _bounded_token(self.config_semantic_id, "config_semantic_id")
        _bounded_token(self.selection_basis, "selection_basis")
        _bounded_token(self.validation_basis, "validation_basis")
        for name in (
            "training_scored_samples",
            "validation_scored_samples",
            "evaluated_candidates",
        ):
            _exact_positive_int(getattr(self, name), name)
        _exact_nonnegative_int(self.unused_gyro_samples, "unused_gyro_samples")
        for name in (
            "command_span_rad_s",
            "command_standard_deviation_rad_s",
            "validation_command_span_rad_s",
            "training_rmse_rad_s",
            "validation_rmse_rad_s",
            "validation_mae_rad_s",
            "validation_normalized_rmse",
            "validation_initial_value_hold_rmse_rad_s",
            "design_condition_number",
            "residual_standard_deviation_rad_s",
        ):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        improvement = _finite_float(
            self.validation_improvement_fraction,
            "validation_improvement_fraction",
        )
        if improvement > 1.0:
            raise ValueError("validation_improvement_fraction cannot exceed one")


@dataclass(frozen=True, slots=True)
class RateAxisFitResult:
    config_semantic_id: str
    model: RateAxisModel
    uncertainty: RateAxisModelUncertainty
    diagnostics: RateAxisFitDiagnostics

    def __post_init__(self) -> None:
        _bounded_token(self.config_semantic_id, "config_semantic_id")
        if type(self.model) is not RateAxisModel:
            raise TypeError("model must be RateAxisModel")
        if type(self.uncertainty) is not RateAxisModelUncertainty:
            raise TypeError("uncertainty must be RateAxisModelUncertainty")
        if type(self.diagnostics) is not RateAxisFitDiagnostics:
            raise TypeError("diagnostics must be RateAxisFitDiagnostics")
        if not (
            self.config_semantic_id
            == self.model.config_semantic_id
            == self.uncertainty.config_semantic_id
            == self.diagnostics.config_semantic_id
        ):
            raise ValueError(
                "result/model/uncertainty/diagnostics config identities disagree"
            )


@dataclass(frozen=True, slots=True)
class _CandidateFit:
    delay_ns: int
    time_constant_s: float
    gain: float
    bias: float
    sse: float
    condition_number: float
    design: np.ndarray
    targets: np.ndarray
    base: np.ndarray


def _gyro_partition(
    trace: RateAxisTrace,
    window: IdentificationWindow,
) -> tuple[GyroRateSample, ...]:
    return tuple(
        sample
        for sample in trace.gyro
        if window.start_monotonic_ns
        <= sample.monotonic_ns
        <= window.end_monotonic_ns
    )


def _command_arrays(
    commands: tuple[RateCommandSample, ...],
    delay_ns: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    return (
        tuple(sample.monotonic_ns + delay_ns for sample in commands),
        tuple(sample.commanded_rate_rad_s for sample in commands),
    )


def _command_at(
    command_times: tuple[int, ...],
    command_values: tuple[float, ...],
    monotonic_ns: int,
) -> float:
    index = bisect.bisect_right(command_times, monotonic_ns) - 1
    if index < 0:
        raise TraceValidationError(
            "command history does not cover delayed model initialization"
        )
    return command_values[index]


def _advance_affine_state(
    state: np.ndarray,
    *,
    start_ns: int,
    end_ns: int,
    time_constant_s: float,
    command_times: tuple[int, ...],
    command_values: tuple[float, ...],
) -> np.ndarray:
    """Propagate ``[initial basis, gain basis, bias basis]`` exactly for ZOH."""

    if end_ns <= start_ns:
        raise TraceValidationError("gyro interval must advance strictly")
    result = state.copy()
    cursor_ns = start_ns
    command_index = bisect.bisect_right(command_times, start_ns) - 1
    if command_index < 0:
        raise TraceValidationError(
            "command history does not cover delayed gyro interval"
        )
    command_value = command_values[command_index]
    next_index = command_index + 1
    while next_index < len(command_times) and command_times[next_index] < end_ns:
        transition_ns = command_times[next_index]
        if transition_ns > cursor_ns:
            duration_s = (transition_ns - cursor_ns) / 1e9
            decay = math.exp(-duration_s / time_constant_s)
            result = decay * result + (1.0 - decay) * np.array(
                (0.0, command_value, 1.0), dtype=np.float64
            )
            cursor_ns = transition_ns
        command_value = command_values[next_index]
        next_index += 1
    duration_s = (end_ns - cursor_ns) / 1e9
    decay = math.exp(-duration_s / time_constant_s)
    return decay * result + (1.0 - decay) * np.array(
        (0.0, command_value, 1.0), dtype=np.float64
    )


def _build_affine_design(
    gyro: tuple[GyroRateSample, ...],
    commands: tuple[RateCommandSample, ...],
    *,
    delay_ns: int,
    time_constant_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    command_times, command_values = _command_arrays(commands, delay_ns)
    _command_at(command_times, command_values, gyro[0].monotonic_ns)
    state = np.array(
        (gyro[0].measured_rate_rad_s, 0.0, 0.0), dtype=np.float64
    )
    bases: list[float] = []
    rows: list[tuple[float, float]] = []
    targets: list[float] = []
    previous_time = gyro[0].monotonic_ns
    for sample in gyro[1:]:
        state = _advance_affine_state(
            state,
            start_ns=previous_time,
            end_ns=sample.monotonic_ns,
            time_constant_s=time_constant_s,
            command_times=command_times,
            command_values=command_values,
        )
        bases.append(float(state[0]))
        rows.append((float(state[1]), float(state[2])))
        targets.append(sample.measured_rate_rad_s - float(state[0]))
        previous_time = sample.monotonic_ns
    return (
        np.asarray(bases, dtype=np.float64),
        np.asarray(rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )


def _fit_candidate(
    gyro: tuple[GyroRateSample, ...],
    commands: tuple[RateCommandSample, ...],
    *,
    delay_ns: int,
    time_constant_s: float,
    config: SystemIdConfig,
) -> Optional[_CandidateFit]:
    base, design, targets = _build_affine_design(
        gyro,
        commands,
        delay_ns=delay_ns,
        time_constant_s=time_constant_s,
    )
    if design.shape[0] < 2 or np.linalg.matrix_rank(design) != 2:
        return None
    condition = float(np.linalg.cond(design))
    if (
        not math.isfinite(condition)
        or condition > config.maximum_design_condition_number
    ):
        return None
    try:
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            design, targets, rcond=None
        )
    except np.linalg.LinAlgError:
        return None
    if rank != 2 or not np.all(np.isfinite(coefficients)):
        return None
    gain, bias = float(coefficients[0]), float(coefficients[1])
    if not config.minimum_gain <= gain <= config.maximum_gain:
        return None
    if abs(bias) > config.maximum_abs_bias_rad_s:
        return None
    errors = targets - design @ coefficients
    sse = float(errors @ errors)
    if not math.isfinite(sse):
        return None
    return _CandidateFit(
        delay_ns=delay_ns,
        time_constant_s=time_constant_s,
        gain=gain,
        bias=bias,
        sse=sse,
        condition_number=condition,
        design=design,
        targets=targets,
        base=base,
    )


def _simulate_model(
    gyro: tuple[GyroRateSample, ...],
    commands: tuple[RateCommandSample, ...],
    model: RateAxisModel,
) -> np.ndarray:
    command_times, command_values = _command_arrays(commands, model.delay_ns)
    _command_at(command_times, command_values, gyro[0].monotonic_ns)
    predicted = [gyro[0].measured_rate_rad_s]
    current = float(predicted[0])
    previous_time = gyro[0].monotonic_ns
    for sample in gyro[1:]:
        # The same exact ZOH propagation is expressed as a three-basis affine
        # state so command transitions inside irregular gyro intervals are not
        # snapped to a sample edge.
        state = _advance_affine_state(
            np.array((current, 0.0, 0.0), dtype=np.float64),
            start_ns=previous_time,
            end_ns=sample.monotonic_ns,
            time_constant_s=model.time_constant_s,
            command_times=command_times,
            command_values=command_values,
        )
        current = (
            float(state[0])
            + model.steady_state_gain * float(state[1])
            + model.gyro_bias_rad_s * float(state[2])
        )
        predicted.append(current)
        previous_time = sample.monotonic_ns
    return np.asarray(predicted, dtype=np.float64)


def _profile_interval_int(
    candidates: tuple[int, ...],
    acceptable: tuple[int, ...],
) -> tuple[int, int]:
    first = acceptable[0]
    last = acceptable[-1]
    first_index = candidates.index(first)
    last_index = candidates.index(last)
    if first_index == 0:
        lower = max(0, first - (candidates[1] - first) // 2)
    else:
        lower = max(0, first - (first - candidates[first_index - 1]) // 2)
    if last_index == len(candidates) - 1:
        upper = last + (last - candidates[last_index - 1]) // 2
    else:
        upper = last + (candidates[last_index + 1] - last) // 2
    return lower, upper


def _profile_interval_float(
    candidates: tuple[float, ...],
    acceptable: tuple[float, ...],
) -> tuple[float, float]:
    first = acceptable[0]
    last = acceptable[-1]
    first_index = candidates.index(first)
    last_index = candidates.index(last)
    if first_index == 0:
        lower = max(0.0, first - 0.5 * (candidates[1] - first))
    else:
        lower = first - 0.5 * (first - candidates[first_index - 1])
    if last_index == len(candidates) - 1:
        upper = last + 0.5 * (last - candidates[last_index - 1])
    else:
        upper = last + 0.5 * (candidates[last_index + 1] - last)
    return float(lower), float(upper)


def _command_values_at(
    commands: tuple[RateCommandSample, ...],
    times_ns: tuple[int, ...],
) -> np.ndarray:
    command_times = tuple(sample.monotonic_ns for sample in commands)
    command_values = tuple(sample.commanded_rate_rad_s for sample in commands)
    return np.asarray(
        [_command_at(command_times, command_values, time_ns) for time_ns in times_ns],
        dtype=np.float64,
    )


class VQ2RateSystemIdentifier:
    """Fit one offline FOPDT body-rate axis with chronological validation."""

    def __init__(self, config: Optional[SystemIdConfig] = None) -> None:
        if config is not None and type(config) is not SystemIdConfig:
            raise TypeError("config must be SystemIdConfig or None")
        self.config = config or SystemIdConfig()

    def fit(
        self,
        trace: RateAxisTrace,
        holdout: ChronologicalHoldout,
    ) -> RateAxisFitResult:
        if type(trace) is not RateAxisTrace:
            raise TypeError("trace must be RateAxisTrace")
        if type(holdout) is not ChronologicalHoldout:
            raise TypeError("holdout must be ChronologicalHoldout")
        config = self.config
        config_semantic_id = config.semantic_identity
        self._validate_trace_bounds(trace)
        training = _gyro_partition(trace, holdout.training)
        validation = _gyro_partition(trace, holdout.validation)
        self._validate_partition(
            training,
            holdout.training,
            minimum_samples=config.minimum_training_samples,
            minimum_duration_s=config.minimum_training_duration_s,
            label="training",
        )
        self._validate_partition(
            validation,
            holdout.validation,
            minimum_samples=config.minimum_validation_samples,
            minimum_duration_s=config.minimum_validation_duration_s,
            label="validation",
        )
        if set(training) & set(validation):
            raise TraceValidationError("training/validation gyro leakage detected")
        earliest_required = min(
            training[0].monotonic_ns,
            validation[0].monotonic_ns,
        ) - max(config.delay_candidates_ns)
        if earliest_required < 0:
            raise TraceValidationError("trace lacks nonnegative delayed history")
        if trace.commands[0].monotonic_ns > earliest_required:
            raise TraceValidationError(
                "command history does not cover the maximum reviewed delay"
            )

        # Give each phase only the command prefix that is causal at its final
        # gyro timestamp.  The propagation routines are causal themselves,
        # but materializing these prefixes makes future-command isolation an
        # explicit boundary rather than an incidental bisect behavior.
        training_commands = tuple(
            sample
            for sample in trace.commands
            if sample.monotonic_ns <= training[-1].monotonic_ns
        )
        validation_commands = tuple(
            sample
            for sample in trace.commands
            if sample.monotonic_ns <= validation[-1].monotonic_ns
        )

        training_command_values = _command_values_at(
            training_commands,
            tuple(sample.monotonic_ns for sample in training),
        )
        validation_command_values = _command_values_at(
            validation_commands,
            tuple(sample.monotonic_ns for sample in validation),
        )
        training_span, training_std = self._validate_excitation(
            training_command_values, "training"
        )
        validation_span, _validation_std = self._validate_excitation(
            validation_command_values, "validation"
        )

        candidates: list[_CandidateFit] = []
        for delay_ns in config.delay_candidates_ns:
            for time_constant_s in config.time_constant_candidates_s:
                candidate = _fit_candidate(
                    training,
                    training_commands,
                    delay_ns=delay_ns,
                    time_constant_s=time_constant_s,
                    config=config,
                )
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            raise IdentifiabilityError(
                "no finite, well-conditioned physical model fit the training data"
            )
        best = min(
            candidates,
            key=lambda item: (item.sse, item.delay_ns, item.time_constant_s),
        )
        if best.delay_ns == config.delay_candidates_ns[-1]:
            raise IdentifiabilityError("delay optimum reached the search boundary")
        if best.time_constant_s in {
            config.time_constant_candidates_s[0],
            config.time_constant_candidates_s[-1],
        }:
            raise IdentifiabilityError(
                "time-constant optimum reached the search boundary"
            )
        training_dof = max(1, best.design.shape[0] - 2)
        training_variance = max(
            best.sse / training_dof,
            config.minimum_residual_variance,
        )
        profile_threshold = best.sse + config.profile_delta_sigma2 * training_variance
        delay_profile = {
            delay: min(
                item.sse for item in candidates if item.delay_ns == delay
            )
            for delay in {
                item.delay_ns for item in candidates
            }
        }
        tau_profile = {
            tau: min(
                item.sse
                for item in candidates
                if item.time_constant_s == tau
            )
            for tau in {
                item.time_constant_s for item in candidates
            }
        }
        acceptable_delays = tuple(
            delay
            for delay in config.delay_candidates_ns
            if delay in delay_profile and delay_profile[delay] <= profile_threshold
        )
        acceptable_taus = tuple(
            tau
            for tau in config.time_constant_candidates_s
            if tau in tau_profile and tau_profile[tau] <= profile_threshold
        )
        if not acceptable_delays or not acceptable_taus:
            raise IdentifiabilityError("profile uncertainty could not be evaluated")
        if acceptable_delays[-1] == config.delay_candidates_ns[-1]:
            raise IdentifiabilityError("delay profile remains open at search boundary")
        if (
            acceptable_taus[0] == config.time_constant_candidates_s[0]
            or acceptable_taus[-1] == config.time_constant_candidates_s[-1]
        ):
            raise IdentifiabilityError(
                "time-constant profile remains open at search boundary"
            )
        delay_interval = _profile_interval_int(
            config.delay_candidates_ns, acceptable_delays
        )
        tau_interval = _profile_interval_float(
            config.time_constant_candidates_s, acceptable_taus
        )
        if delay_interval[1] - delay_interval[0] > config.maximum_delay_uncertainty_ns:
            raise IdentifiabilityError("delay uncertainty is too wide")
        if (
            tau_interval[1] - tau_interval[0]
            > config.maximum_time_constant_uncertainty_s
        ):
            raise IdentifiabilityError("time-constant uncertainty is too wide")

        model = RateAxisModel(
            model_id="vq2-rate-axis-fopdt-offline-v1",
            config_semantic_id=config_semantic_id,
            host_clock_id=trace.host_clock_id,
            axis=trace.axis,
            delay_ns=best.delay_ns,
            time_constant_s=best.time_constant_s,
            steady_state_gain=best.gain,
            gyro_bias_rad_s=best.bias,
        )

        # The first validation gyro is used only as the free-run initial state;
        # it is explicitly excluded from every validation score.  No later
        # validation gyro label enters propagation or model selection.
        validation_prediction = _simulate_model(
            validation, validation_commands, model
        )
        validation_observed = np.asarray(
            [sample.measured_rate_rad_s for sample in validation], dtype=np.float64
        )
        validation_errors = validation_observed[1:] - validation_prediction[1:]
        validation_sse = float(validation_errors @ validation_errors)
        validation_rmse = float(math.sqrt(validation_sse / len(validation_errors)))
        validation_mae = float(np.mean(np.abs(validation_errors)))
        validation_scale = float(np.std(validation_observed[1:]))
        if validation_scale < config.minimum_output_standard_deviation_rad_s:
            raise HeldOutValidationError(
                "held-out output variation is too weak for validation"
            )
        validation_normalized_rmse = validation_rmse / validation_scale
        initial_value_hold_errors = validation_observed[1:] - validation_observed[0]
        initial_value_hold_rmse = float(
            math.sqrt(
                float(initial_value_hold_errors @ initial_value_hold_errors)
                / len(initial_value_hold_errors)
            )
        )
        if initial_value_hold_rmse <= math.sqrt(config.minimum_residual_variance):
            raise HeldOutValidationError(
                "held-out initial-value hold comparator is uninformative"
            )
        validation_improvement = 1.0 - validation_rmse / initial_value_hold_rmse
        if validation_normalized_rmse > config.maximum_validation_normalized_rmse:
            raise HeldOutValidationError(
                "training-only model failed held-out normalized RMSE"
            )
        if validation_improvement < config.minimum_validation_improvement_fraction:
            raise HeldOutValidationError(
                "training-only model did not beat the held-out initial-value hold"
            )

        # Held-out labels have no path back into the selected model or the
        # already-fixed training profile intervals.  Their residual variance
        # can only make the conditional gain/bias covariance more conservative.
        uncertainty_variance = max(
            training_variance,
            validation_sse / max(1, len(validation_errors) - 1),
            config.minimum_residual_variance,
        )
        information = best.design.T @ best.design
        try:
            gain_bias_covariance = uncertainty_variance * np.linalg.inv(information)
        except np.linalg.LinAlgError as exc:
            raise IdentifiabilityError(
                "gain/bias uncertainty information is singular"
            ) from exc
        gain_bias_covariance = (
            gain_bias_covariance + gain_bias_covariance.T
        ) * 0.5
        if not np.all(np.isfinite(gain_bias_covariance)):
            raise IdentifiabilityError("gain/bias uncertainty became non-finite")
        uncertainty = RateAxisModelUncertainty(
            config_semantic_id=config_semantic_id,
            method_id="training-profile-plus-heldout-residual-v1",
            confidence_level=0.95,
            delay_interval_ns=delay_interval,
            time_constant_interval_s=tau_interval,
            gain_standard_error=math.sqrt(
                max(0.0, float(gain_bias_covariance[0, 0]))
            ),
            bias_standard_error_rad_s=math.sqrt(
                max(0.0, float(gain_bias_covariance[1, 1]))
            ),
            gain_bias_covariance=tuple(
                tuple(float(value) for value in row)
                for row in gain_bias_covariance
            ),  # type: ignore[arg-type]
        )
        training_rmse = math.sqrt(best.sse / best.design.shape[0])
        diagnostics = RateAxisFitDiagnostics(
            config_semantic_id=config_semantic_id,
            selection_basis="chronological_training_only_grid_v1",
            validation_basis="heldout_free_run_one_initializer_v1",
            training_scored_samples=len(training) - 1,
            validation_scored_samples=len(validation) - 1,
            unused_gyro_samples=(
                len(trace.gyro) - len(training) - len(validation)
            ),
            evaluated_candidates=len(candidates),
            command_span_rad_s=training_span,
            command_standard_deviation_rad_s=training_std,
            validation_command_span_rad_s=validation_span,
            training_rmse_rad_s=float(training_rmse),
            validation_rmse_rad_s=validation_rmse,
            validation_mae_rad_s=validation_mae,
            validation_normalized_rmse=validation_normalized_rmse,
            validation_initial_value_hold_rmse_rad_s=initial_value_hold_rmse,
            validation_improvement_fraction=validation_improvement,
            design_condition_number=best.condition_number,
            residual_standard_deviation_rad_s=math.sqrt(uncertainty_variance),
        )
        return RateAxisFitResult(
            config_semantic_id=config_semantic_id,
            model=model,
            uncertainty=uncertainty,
            diagnostics=diagnostics,
        )

    def _validate_trace_bounds(self, trace: RateAxisTrace) -> None:
        config = self.config
        if any(sample.saturated for sample in trace.commands):
            raise TraceValidationError(
                "saturated command samples cannot enter system identification"
            )
        if any(
            later.monotonic_ns - earlier.monotonic_ns
            < _MINIMUM_COMMAND_PERIOD_NS
            for earlier, later in zip(trace.commands, trace.commands[1:])
        ):
            raise TraceValidationError(
                "command sample interval exceeds the reviewed 50 Hz cap"
            )
        if any(sample.clipped for sample in trace.gyro):
            raise TraceValidationError(
                "clipped gyro samples cannot enter system identification"
            )
        if any(
            abs(sample.commanded_rate_rad_s)
            > config.maximum_abs_command_rate_rad_s
            for sample in trace.commands
        ):
            raise TraceValidationError("command sample exceeds reviewed fit bound")
        if trace.axis is RateAxis.YAW and any(
            sample.commanded_rate_rad_s != 0.0 for sample in trace.commands
        ):
            raise TraceValidationError(
                "nonzero yaw command evidence is outside the frozen VQ2 envelope"
            )
        if any(
            abs(sample.measured_rate_rad_s) > config.maximum_abs_gyro_rate_rad_s
            for sample in trace.gyro
        ):
            raise TraceValidationError("gyro sample exceeds reviewed fit bound")

    def _validate_partition(
        self,
        samples: tuple[GyroRateSample, ...],
        window: IdentificationWindow,
        *,
        minimum_samples: int,
        minimum_duration_s: float,
        label: str,
    ) -> None:
        if len(samples) < minimum_samples:
            raise IdentifiabilityError(f"{label} partition has too few gyro samples")
        actual_duration_s = (
            samples[-1].monotonic_ns - samples[0].monotonic_ns
        ) / 1e9
        if actual_duration_s < minimum_duration_s:
            raise IdentifiabilityError(f"{label} partition is too short")
        maximum_gap_ns = round(self.config.maximum_gyro_gap_s * 1e9)
        if any(
            later.monotonic_ns - earlier.monotonic_ns > maximum_gap_ns
            for earlier, later in zip(samples, samples[1:])
        ):
            raise TraceValidationError(
                f"{label} gyro timestamps contain a missing-data gap"
            )
        if not (
            window.start_monotonic_ns <= samples[0].monotonic_ns
            and samples[-1].monotonic_ns <= window.end_monotonic_ns
        ):
            raise AssertionError("partition escaped its declared window")

    def _validate_excitation(
        self,
        command_values: np.ndarray,
        label: str,
    ) -> tuple[float, float]:
        span = float(np.ptp(command_values))
        standard_deviation = float(np.std(command_values))
        if span < self.config.minimum_command_span_rad_s:
            raise IdentifiabilityError(f"{label} command span is too weak")
        if (
            standard_deviation
            < self.config.minimum_command_standard_deviation_rad_s
        ):
            raise IdentifiabilityError(
                f"{label} command standard deviation is too weak"
            )
        return span, standard_deviation


def fit_rate_axis_model(
    trace: RateAxisTrace,
    holdout: ChronologicalHoldout,
    *,
    config: Optional[SystemIdConfig] = None,
) -> RateAxisFitResult:
    """Convenience wrapper around :class:`VQ2RateSystemIdentifier`."""

    return VQ2RateSystemIdentifier(config).fit(trace, holdout)


__all__ = [
    "ChronologicalHoldout",
    "GyroRateSample",
    "HeldOutValidationError",
    "IdentificationWindow",
    "IdentifiabilityError",
    "RateAxis",
    "RateAxisFitDiagnostics",
    "RateAxisFitResult",
    "RateAxisModel",
    "RateAxisModelUncertainty",
    "RateAxisTrace",
    "RateCommandSample",
    "RateExperimentDefinition",
    "RateExperimentSegment",
    "SystemIdConfig",
    "SystemIdentificationError",
    "TraceValidationError",
    "VQ2RateSystemIdentifier",
    "fit_rate_axis_model",
]
