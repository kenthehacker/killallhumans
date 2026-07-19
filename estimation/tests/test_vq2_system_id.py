"""Adversarial synthetic tests for the fully offline VQ2 system identifier.

These traces are mathematical fixtures.  They are not simulator captures and
do not establish FlightSim plant parameters or authorize powered collection.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import replace

import numpy as np
import pytest

from estimation.vq2_system_id import (
    ChronologicalHoldout,
    GyroRateSample,
    HeldOutValidationError,
    IdentificationWindow,
    IdentifiabilityError,
    RateAxis,
    RateAxisTrace,
    RateCommandSample,
    RateExperimentDefinition,
    RateExperimentSegment,
    SystemIdConfig,
    TraceValidationError,
    fit_rate_axis_model,
)


_CLOCK = "synthetic-host-clock"
_BASE_NS = 1_000_000_000
_TRUE_DELAY_NS = 35_000_000
_TRUE_TIME_CONSTANT_S = 0.120
_TRUE_GAIN = 0.82
_TRUE_BIAS_RAD_S = 0.018

_COMMAND_SCHEDULE = (
    (0.00, 0.00),
    (0.30, 0.16),
    (0.65, -0.13),
    (0.98, 0.08),
    (1.27, -0.18),
    (1.62, 0.00),
    (1.88, 0.14),
    (2.19, -0.11),
    (2.46, 0.19),
    (2.82, -0.16),
    (3.15, 0.00),
    (3.40, -0.14),
    (3.73, 0.17),
    (4.08, -0.09),
    (4.41, 0.15),
    (4.75, 0.00),
    (5.00, -0.17),
    (5.31, 0.13),
    (5.63, -0.10),
    (5.94, 0.18),
    (6.31, 0.00),
    (6.55, -0.16),
    (6.89, 0.12),
    (7.21, -0.18),
    (7.55, 0.09),
    (7.86, 0.00),
    (8.03, 0.16),
    (8.30, -0.14),
    (8.55, 0.00),
    (8.80, 0.00),
)


def _test_config(**overrides: object) -> SystemIdConfig:
    values: dict[str, object] = {
        "delay_candidates_ns": tuple(range(0, 70_000_001, 5_000_000)),
        "time_constant_candidates_s": tuple(
            value / 1_000.0 for value in range(60, 201, 10)
        ),
    }
    values.update(overrides)
    return SystemIdConfig(**values)


def _commands(
    schedule: tuple[tuple[float, float], ...] = _COMMAND_SCHEDULE,
    *,
    axis: RateAxis = RateAxis.ROLL,
) -> tuple[RateCommandSample, ...]:
    return tuple(
        RateCommandSample(
            host_clock_id=_CLOCK,
            sequence=index,
            monotonic_ns=_BASE_NS + round(offset_s * 1e9),
            axis=axis,
            commanded_rate_rad_s=rate,
        )
        for index, (offset_s, rate) in enumerate(schedule)
    )


def _physical_advance(
    value: float,
    start_ns: int,
    end_ns: int,
    commands: tuple[RateCommandSample, ...],
    *,
    delay_ns: int,
    time_constant_s: float,
    gain: float,
    bias: float,
) -> float:
    """Independent exact ZOH propagation for a first-order synthetic plant."""

    transition_times = tuple(sample.monotonic_ns + delay_ns for sample in commands)
    transition_values = tuple(sample.commanded_rate_rad_s for sample in commands)
    command_index = bisect.bisect_right(transition_times, start_ns) - 1
    assert command_index >= 0
    cursor_ns = start_ns
    command = transition_values[command_index]
    next_index = command_index + 1

    def advance_segment(current: float, duration_ns: int, rate: float) -> float:
        decay = math.exp(-(duration_ns / 1e9) / time_constant_s)
        equilibrium = gain * rate + bias
        return decay * current + (1.0 - decay) * equilibrium

    while (
        next_index < len(transition_times)
        and transition_times[next_index] < end_ns
    ):
        transition_ns = transition_times[next_index]
        if transition_ns > cursor_ns:
            value = advance_segment(value, transition_ns - cursor_ns, command)
            cursor_ns = transition_ns
        command = transition_values[next_index]
        next_index += 1
    return advance_segment(value, end_ns - cursor_ns, command)


def _synthetic_trace(
    *,
    schedule: tuple[tuple[float, float], ...] = _COMMAND_SCHEDULE,
    axis: RateAxis = RateAxis.ROLL,
    delay_ns: int = _TRUE_DELAY_NS,
    time_constant_s: float = _TRUE_TIME_CONSTANT_S,
    gain: float = _TRUE_GAIN,
    bias: float = _TRUE_BIAS_RAD_S,
    noise_scale: float = 0.00035,
) -> RateAxisTrace:
    commands = _commands(schedule, axis=axis)
    increments_ns = (
        8_000_000,
        11_000_000,
        9_000_000,
        13_000_000,
        10_000_000,
        7_000_000,
        12_000_000,
    )
    timestamps: list[int] = [_BASE_NS + 120_000_000]
    increment_index = 0
    while timestamps[-1] < _BASE_NS + 8_620_000_000:
        timestamps.append(
            timestamps[-1] + increments_ns[increment_index % len(increments_ns)]
        )
        increment_index += 1

    delayed_times = tuple(sample.monotonic_ns + delay_ns for sample in commands)
    initial_index = bisect.bisect_right(delayed_times, timestamps[0]) - 1
    assert initial_index >= 0
    state = gain * commands[initial_index].commanded_rate_rad_s + bias
    gyro: list[GyroRateSample] = []
    previous_time = timestamps[0]
    for index, timestamp_ns in enumerate(timestamps):
        if index:
            state = _physical_advance(
                state,
                previous_time,
                timestamp_ns,
                commands,
                delay_ns=delay_ns,
                time_constant_s=time_constant_s,
                gain=gain,
                bias=bias,
            )
        deterministic_noise = noise_scale * (
            math.sin(0.37 * index) + 0.35 * math.cos(0.11 * index)
        )
        gyro.append(
            GyroRateSample(
                host_clock_id=_CLOCK,
                sequence=index,
                monotonic_ns=timestamp_ns,
                axis=axis,
                measured_rate_rad_s=state + deterministic_noise,
            )
        )
        previous_time = timestamp_ns
    return RateAxisTrace(
        host_clock_id=_CLOCK,
        axis=axis,
        commands=commands,
        gyro=tuple(gyro),
    )


def _holdout() -> ChronologicalHoldout:
    return ChronologicalHoldout(
        training=IdentificationWindow(
            _BASE_NS + 200_000_000,
            _BASE_NS + 4_900_000_000,
        ),
        validation=IdentificationWindow(
            _BASE_NS + 5_100_000_000,
            _BASE_NS + 8_500_000_000,
        ),
    )


def _renumber_gyro(
    samples: tuple[GyroRateSample, ...],
) -> tuple[GyroRateSample, ...]:
    return tuple(
        replace(sample, sequence=index) for index, sample in enumerate(samples)
    )


def test_irregular_trace_recovers_model_with_heldout_diagnostics_and_uncertainty():
    trace = _synthetic_trace()
    holdout = _holdout()

    result = fit_rate_axis_model(trace, holdout, config=SystemIdConfig())

    assert result.model.host_clock_id == _CLOCK
    assert result.model.axis is RateAxis.ROLL
    assert result.model.delay_ns == pytest.approx(_TRUE_DELAY_NS, abs=5_000_000)
    assert result.model.time_constant_s == pytest.approx(
        _TRUE_TIME_CONSTANT_S, abs=0.010
    )
    assert result.model.steady_state_gain == pytest.approx(_TRUE_GAIN, abs=0.015)
    assert result.model.gyro_bias_rad_s == pytest.approx(
        _TRUE_BIAS_RAD_S, abs=0.004
    )
    uncertainty = result.uncertainty
    assert (
        uncertainty.delay_interval_ns[0]
        <= _TRUE_DELAY_NS
        <= uncertainty.delay_interval_ns[1]
    )
    assert (
        uncertainty.time_constant_interval_s[0]
        <= _TRUE_TIME_CONSTANT_S
        <= uncertainty.time_constant_interval_s[1]
    )
    covariance = np.asarray(uncertainty.gain_bias_covariance)
    assert np.all(np.isfinite(covariance))
    assert np.linalg.eigvalsh(covariance)[0] >= -1e-12
    assert uncertainty.gain_standard_error > 0.0
    assert uncertainty.bias_standard_error_rad_s > 0.0

    training_count = sum(
        holdout.training.start_monotonic_ns
        <= sample.monotonic_ns
        <= holdout.training.end_monotonic_ns
        for sample in trace.gyro
    )
    validation_count = sum(
        holdout.validation.start_monotonic_ns
        <= sample.monotonic_ns
        <= holdout.validation.end_monotonic_ns
        for sample in trace.gyro
    )
    diagnostics = result.diagnostics
    assert diagnostics.selection_basis == "chronological_training_only_grid_v1"
    assert diagnostics.validation_basis == "heldout_free_run_one_initializer_v1"
    assert diagnostics.training_scored_samples == training_count - 1
    assert diagnostics.validation_scored_samples == validation_count - 1
    assert diagnostics.unused_gyro_samples == (
        len(trace.gyro) - training_count - validation_count
    )
    assert diagnostics.validation_normalized_rmse < 0.05
    assert diagnostics.validation_improvement_fraction > 0.90


def test_fit_is_deterministic_for_identical_irregular_evidence():
    trace = _synthetic_trace()
    holdout = _holdout()
    config = _test_config()

    first = fit_rate_axis_model(trace, holdout, config=config)
    second = fit_rate_axis_model(trace, holdout, config=config)

    future_commands = tuple(
        replace(sample, commanded_rate_rad_s=0.24)
        if sample.monotonic_ns > holdout.validation.end_monotonic_ns
        else sample
        for sample in trace.commands
    )
    changed_future = fit_rate_axis_model(
        replace(trace, commands=future_commands), holdout, config=config
    )

    assert first == second
    assert first == changed_future


def test_later_validation_labels_cannot_change_training_selected_model():
    trace = _synthetic_trace()
    holdout = _holdout()
    config = _test_config()
    validation_indices = [
        index
        for index, sample in enumerate(trace.gyro)
        if holdout.validation.start_monotonic_ns
        <= sample.monotonic_ns
        <= holdout.validation.end_monotonic_ns
    ]
    assert len(validation_indices) > 2
    changed = list(trace.gyro)
    for phase, index in enumerate(validation_indices[1:], start=1):
        changed[index] = replace(
            changed[index],
            measured_rate_rad_s=(
                changed[index].measured_rate_rad_s
                + 0.00012 * math.sin(0.41 * phase)
            ),
        )
    changed_trace = replace(trace, gyro=tuple(changed))

    original = fit_rate_axis_model(trace, holdout, config=config)
    perturbed = fit_rate_axis_model(changed_trace, holdout, config=config)

    assert original.model == perturbed.model
    assert (
        original.diagnostics.validation_rmse_rad_s
        != perturbed.diagnostics.validation_rmse_rad_s
    )


def test_corrupted_heldout_outcomes_reject_the_training_only_model():
    trace = _synthetic_trace()
    holdout = _holdout()
    validation_indices = [
        index
        for index, sample in enumerate(trace.gyro)
        if holdout.validation.start_monotonic_ns
        <= sample.monotonic_ns
        <= holdout.validation.end_monotonic_ns
    ]
    changed = list(trace.gyro)
    for index in validation_indices[1:]:
        changed[index] = replace(
            changed[index], measured_rate_rad_s=-changed[index].measured_rate_rad_s
        )

    with pytest.raises(HeldOutValidationError):
        fit_rate_axis_model(
            replace(trace, gyro=tuple(changed)),
            holdout,
            config=_test_config(),
        )


def test_weak_excitation_and_tight_uncertainty_gate_fail_closed():
    trace = _synthetic_trace()
    weak_commands = tuple(
        replace(sample, commanded_rate_rad_s=0.0) for sample in trace.commands
    )
    with pytest.raises(IdentifiabilityError, match="command span is too weak"):
        fit_rate_axis_model(
            replace(trace, commands=weak_commands),
            _holdout(),
            config=_test_config(),
        )

    with pytest.raises(IdentifiabilityError, match="delay uncertainty is too wide"):
        fit_rate_axis_model(
            trace,
            _holdout(),
            config=_test_config(maximum_delay_uncertainty_ns=1),
        )

    with pytest.raises(IdentifiabilityError, match="optimum reached"):
        fit_rate_axis_model(
            trace,
            _holdout(),
            config=_test_config(
                time_constant_candidates_s=(0.020, 0.030, 0.040)
            ),
        )


def test_delayed_fit_requires_causal_command_history():
    trace = _synthetic_trace()
    commands = tuple(
        replace(sample, sequence=index)
        for index, sample in enumerate(trace.commands[1:])
    )

    with pytest.raises(TraceValidationError, match="command history"):
        fit_rate_axis_model(
            replace(trace, commands=commands),
            _holdout(),
            config=_test_config(),
        )


@pytest.mark.parametrize(
    "kind", ["saturated", "clipped", "missing_gap", "command_burst"]
)
def test_saturation_clipping_and_missing_data_are_rejected(kind: str):
    trace = _synthetic_trace()
    if kind == "saturated":
        commands = list(trace.commands)
        commands[4] = replace(commands[4], saturated=True)
        changed_trace = replace(trace, commands=tuple(commands))
    elif kind == "clipped":
        gyro = list(trace.gyro)
        gyro[40] = replace(gyro[40], clipped=True)
        changed_trace = replace(trace, gyro=tuple(gyro))
    elif kind == "missing_gap":
        gap_start = next(
            index
            for index, sample in enumerate(trace.gyro)
            if sample.monotonic_ns >= _BASE_NS + 1_000_000_000
        )
        gyro = trace.gyro[:gap_start] + trace.gyro[gap_start + 8 :]
        changed_trace = replace(trace, gyro=_renumber_gyro(gyro))
    else:
        commands = list(trace.commands)
        commands[4] = replace(
            commands[4],
            monotonic_ns=commands[3].monotonic_ns + 19_999_999,
        )
        changed_trace = replace(trace, commands=tuple(commands))

    with pytest.raises(TraceValidationError):
        fit_rate_axis_model(changed_trace, _holdout(), config=_test_config())


def test_clock_sequence_and_timestamp_contracts_reject_gaps_replays_and_mixing():
    trace = _synthetic_trace()
    bad_clock = list(trace.gyro)
    bad_clock[1] = replace(bad_clock[1], host_clock_id="different-clock")
    with pytest.raises(TraceValidationError, match="changed host clock"):
        replace(trace, gyro=tuple(bad_clock))

    missing_sequence = list(trace.commands)
    missing_sequence[2] = replace(missing_sequence[2], sequence=3)
    with pytest.raises(TraceValidationError, match="missing/replayed"):
        replace(trace, commands=tuple(missing_sequence))

    duplicate_timestamp = list(trace.gyro)
    duplicate_timestamp[2] = replace(
        duplicate_timestamp[2], monotonic_ns=duplicate_timestamp[1].monotonic_ns
    )
    with pytest.raises(TraceValidationError, match="advance strictly"):
        replace(trace, gyro=tuple(duplicate_timestamp))


def test_training_and_validation_windows_cannot_overlap_or_reverse():
    with pytest.raises(TraceValidationError, match="disjoint and chronological"):
        ChronologicalHoldout(
            training=IdentificationWindow(100, 300),
            validation=IdentificationWindow(300, 500),
        )
    with pytest.raises(ValueError, match="positive duration"):
        IdentificationWindow(500, 100)


def test_samples_reject_bool_nonfinite_and_non_tuple_streams():
    with pytest.raises(TypeError, match="exact integer"):
        RateCommandSample(_CLOCK, True, 1, RateAxis.ROLL, 0.0)
    with pytest.raises(ValueError, match="finite"):
        GyroRateSample(_CLOCK, 0, 1, RateAxis.ROLL, math.nan)
    trace = _synthetic_trace()
    with pytest.raises(TypeError, match="exact tuple"):
        RateAxisTrace(
            host_clock_id=_CLOCK,
            axis=RateAxis.ROLL,
            commands=list(trace.commands),  # type: ignore[arg-type]
            gyro=trace.gyro,
        )


def test_inert_experiment_definition_is_symmetric_bounded_data_only():
    definition = RateExperimentDefinition(
        experiment_id="offline-roll-pulse-v1",
        axis=RateAxis.ROLL,
        command_period_ns=20_000_000,
        segments=(
            RateExperimentSegment(200_000_000, 0.0),
            RateExperimentSegment(400_000_000, 0.12),
            RateExperimentSegment(400_000_000, -0.12),
            RateExperimentSegment(200_000_000, 0.0),
        ),
    )

    assert definition.total_duration_ns == 1_200_000_000
    assert definition.maximum_abs_rate_rad_s == 0.12
    assert not hasattr(definition, "send")
    assert not hasattr(definition, "execute")


@pytest.mark.parametrize(
    "axis, period_ns, rate",
    [
        (RateAxis.YAW, 20_000_000, 0.12),
        (RateAxis.ROLL, 19_000_000, 0.12),
        (RateAxis.PITCH, 20_000_000, 0.26),
    ],
)
def test_inert_experiment_definition_rejects_unsafe_shapes(
    axis: RateAxis,
    period_ns: int,
    rate: float,
):
    with pytest.raises(ValueError):
        RateExperimentDefinition(
            experiment_id="rejected-definition",
            axis=axis,
            command_period_ns=period_ns,
            segments=(
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(400_000_000, rate),
                RateExperimentSegment(400_000_000, -rate),
                RateExperimentSegment(200_000_000, 0.0),
            ),
        )
