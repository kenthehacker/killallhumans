"""Adversarial synthetic tests for the fully offline VQ2 system identifier.

These traces are mathematical fixtures.  They are not simulator captures and
do not establish FlightSim plant parameters or authorize powered collection.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import fields, replace

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
    _advance_affine_state,
    fit_rate_axis_model,
)


_CLOCK = "synthetic-host-clock"
_BASE_NS = 1_000_000_000
_TRUE_DELAY_NS = 35_000_000
_TRUE_TIME_CONSTANT_S = 0.120
_TRUE_GAIN = 0.82
_TRUE_BIAS_RAD_S = 0.018
_REVIEWED_DELAY_GRID_NS = tuple(range(0, 100_000_001, 5_000_000))
_REVIEWED_TIME_CONSTANT_GRID_S = tuple(
    value / 1_000.0 for value in range(20, 301, 10)
)

_LOOSENED_CONFIG_OVERRIDES = (
    ("minimum_training_samples", 79),
    ("minimum_validation_samples", 39),
    ("minimum_training_duration_s", 1.99),
    ("minimum_validation_duration_s", 0.99),
    ("maximum_gyro_gap_s", 0.051),
    ("maximum_abs_command_rate_rad_s", 0.251),
    ("maximum_abs_gyro_rate_rad_s", 4.01),
    ("minimum_command_span_rad_s", 0.099),
    ("minimum_command_standard_deviation_rad_s", 0.024),
    ("minimum_output_standard_deviation_rad_s", 0.009),
    ("maximum_design_condition_number", 100_001.0),
    ("minimum_gain", 0.049),
    ("maximum_gain", 2.01),
    ("maximum_abs_bias_rad_s", 0.301),
    ("profile_delta_sigma2", 3.84),
    ("minimum_residual_variance", 0.9e-12),
    ("maximum_delay_uncertainty_ns", 25_000_001),
    ("maximum_time_constant_uncertainty_s", 0.061),
    ("maximum_validation_normalized_rmse", 0.351),
    ("minimum_validation_improvement_fraction", 0.099),
)

_TIGHTENED_CONFIG_OVERRIDES = (
    ("minimum_training_samples", 81),
    ("minimum_validation_samples", 41),
    ("minimum_training_duration_s", 2.1),
    ("minimum_validation_duration_s", 1.1),
    ("maximum_gyro_gap_s", 0.049),
    ("maximum_abs_command_rate_rad_s", 0.24),
    ("maximum_abs_gyro_rate_rad_s", 3.9),
    ("minimum_command_span_rad_s", 0.11),
    ("minimum_command_standard_deviation_rad_s", 0.026),
    ("minimum_output_standard_deviation_rad_s", 0.011),
    ("maximum_design_condition_number", 90_000.0),
    ("minimum_gain", 0.051),
    ("maximum_gain", 1.9),
    ("maximum_abs_bias_rad_s", 0.29),
    ("minimum_residual_variance", 2.0e-12),
    ("maximum_delay_uncertainty_ns", 24_000_000),
    ("maximum_time_constant_uncertainty_s", 0.050),
    ("maximum_validation_normalized_rmse", 0.34),
    ("minimum_validation_improvement_fraction", 0.11),
)

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
    return SystemIdConfig(**overrides)


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


def _noisy_training_clean_holdout_trace() -> RateAxisTrace:
    """Reviewer regression: noisy training labels and untouched holdout labels."""

    holdout = _holdout()
    base = _synthetic_trace(noise_scale=0.0)
    changed = list(base.gyro)
    for index, sample in enumerate(changed):
        if (
            holdout.training.start_monotonic_ns
            <= sample.monotonic_ns
            <= holdout.training.end_monotonic_ns
        ):
            changed[index] = replace(
                sample,
                measured_rate_rad_s=(
                    sample.measured_rate_rad_s
                    + 0.10
                    * (
                        math.sin(0.37 * index)
                        + 0.35 * math.cos(0.11 * index)
                    )
                ),
            )
    return replace(base, gyro=tuple(changed))


def _renumber_gyro(
    samples: tuple[GyroRateSample, ...],
) -> tuple[GyroRateSample, ...]:
    return tuple(
        replace(sample, sequence=index) for index, sample in enumerate(samples)
    )


@pytest.mark.parametrize("field_name,loosened_value", _LOOSENED_CONFIG_OVERRIDES)
def test_reviewed_config_rejects_every_looser_override(
    field_name: str,
    loosened_value: object,
):
    with pytest.raises(ValueError, match="reviewed"):
        SystemIdConfig(**{field_name: loosened_value})


@pytest.mark.parametrize(
    "override",
    [
        {"delay_candidates_ns": _REVIEWED_DELAY_GRID_NS[:-1]},
        {"delay_candidates_ns": _REVIEWED_DELAY_GRID_NS + (105_000_000,)},
        {
            "delay_candidates_ns": tuple(
                sorted(_REVIEWED_DELAY_GRID_NS + (2_500_000,))
            )
        },
        {"time_constant_candidates_s": _REVIEWED_TIME_CONSTANT_GRID_S[1:]},
        {
            "time_constant_candidates_s": (
                _REVIEWED_TIME_CONSTANT_GRID_S + (0.310,)
            )
        },
        {
            "time_constant_candidates_s": tuple(
                sorted(_REVIEWED_TIME_CONSTANT_GRID_S + (0.025,))
            )
        },
        {"profile_delta_sigma2": 3.841458820694125},
    ],
)
def test_profile_grids_and_95_percent_cutoff_are_pinned(override: dict[str, object]):
    with pytest.raises(ValueError, match="pinned"):
        SystemIdConfig(**override)


def test_config_semantic_identity_is_canonical_and_change_sensitive():
    default = SystemIdConfig()
    numerically_equivalent = SystemIdConfig(
        minimum_training_duration_s=2,
        minimum_validation_duration_s=1,
    )

    assert default.semantic_identity == SystemIdConfig().semantic_identity
    assert default.semantic_identity == numerically_equivalent.semantic_identity
    assert default.semantic_identity.startswith(
        "vq2-rate-system-id-policy-v1:sha256:"
    )
    assert len(default.semantic_identity.rsplit(":", 1)[1]) == 64

    tightened_identities = {
        SystemIdConfig(**{field_name: value}).semantic_identity
        for field_name, value in _TIGHTENED_CONFIG_OVERRIDES
    }
    assert default.semantic_identity not in tightened_identities
    assert len(tightened_identities) == len(_TIGHTENED_CONFIG_OVERRIDES)

    identity_covered_fields = {
        field_name for field_name, _value in _TIGHTENED_CONFIG_OVERRIDES
    } | {
        "delay_candidates_ns",
        "time_constant_candidates_s",
        "profile_delta_sigma2",
    }
    assert identity_covered_fields == {
        field.name for field in fields(SystemIdConfig)
    }


@pytest.mark.parametrize(
    "override",
    [
        {"minimum_gain": 1.9, "maximum_gain": 1.8},
        {"maximum_abs_command_rate_rad_s": 0.04},
        {
            "minimum_command_standard_deviation_rad_s": 0.21,
            "maximum_abs_command_rate_rad_s": 0.20,
        },
        {"maximum_abs_gyro_rate_rad_s": 0.20},
        {
            "maximum_abs_gyro_rate_rad_s": 0.009,
            "maximum_abs_bias_rad_s": 0.008,
        },
    ],
)
def test_individually_tighter_but_physically_inconsistent_config_is_rejected(
    override: dict[str, object],
):
    with pytest.raises(ValueError):
        SystemIdConfig(**override)


def test_irregular_trace_recovers_model_with_heldout_diagnostics_and_uncertainty():
    trace = _synthetic_trace()
    holdout = _holdout()
    config = SystemIdConfig()

    result = fit_rate_axis_model(trace, holdout, config=config)

    assert result.config_semantic_id == config.semantic_identity
    assert result.model.config_semantic_id == config.semantic_identity
    assert result.uncertainty.config_semantic_id == config.semantic_identity
    assert result.diagnostics.config_semantic_id == config.semantic_identity
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
    config = _test_config(minimum_training_samples=81)

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
    assert first.config_semantic_id == config.semantic_identity
    assert first.config_semantic_id != SystemIdConfig().semantic_identity
    with pytest.raises(ValueError, match="config identities disagree"):
        replace(
            first,
            config_semantic_id=SystemIdConfig().semantic_identity,
        )


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
        original.uncertainty.delay_interval_ns
        == perturbed.uncertainty.delay_interval_ns
    )
    assert (
        original.uncertainty.time_constant_interval_s
        == perturbed.uncertainty.time_constant_interval_s
    )
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


@pytest.mark.parametrize(
    "config",
    [
        SystemIdConfig(),
        SystemIdConfig(minimum_gain=0.90),
        SystemIdConfig(maximum_gain=0.85),
        SystemIdConfig(maximum_design_condition_number=10.0),
        SystemIdConfig(maximum_abs_bias_rad_s=0.015),
    ],
)
def test_tightened_selector_cannot_escape_default_profile_rejection(
    config: SystemIdConfig,
):
    with pytest.raises(IdentifiabilityError, match="delay uncertainty is too wide"):
        fit_rate_axis_model(
            _noisy_training_clean_holdout_trace(),
            _holdout(),
            config=config,
        )


def test_tightened_selector_either_rejects_or_preserves_model_and_profiles():
    trace = _synthetic_trace()
    holdout = _holdout()
    default = fit_rate_axis_model(trace, holdout, config=SystemIdConfig())
    accepted_config = SystemIdConfig(
        minimum_gain=default.model.steady_state_gain - 0.01,
        maximum_gain=default.model.steady_state_gain + 0.01,
        maximum_abs_bias_rad_s=abs(default.model.gyro_bias_rad_s) + 0.01,
        maximum_design_condition_number=(
            default.diagnostics.design_condition_number + 1.0
        ),
    )

    tightened = fit_rate_axis_model(trace, holdout, config=accepted_config)

    assert tightened.config_semantic_id == accepted_config.semantic_identity
    assert tightened.config_semantic_id != default.config_semantic_id
    assert tightened.model.delay_ns == default.model.delay_ns
    assert tightened.model.time_constant_s == default.model.time_constant_s
    assert tightened.model.steady_state_gain == default.model.steady_state_gain
    assert tightened.model.gyro_bias_rad_s == default.model.gyro_bias_rad_s
    assert (
        tightened.diagnostics.evaluated_candidates
        == default.diagnostics.evaluated_candidates
    )
    assert (
        tightened.diagnostics.design_condition_number
        == default.diagnostics.design_condition_number
    )
    assert (
        tightened.uncertainty.delay_interval_ns
        == default.uncertainty.delay_interval_ns
    )
    assert (
        tightened.uncertainty.time_constant_interval_s
        == default.uncertainty.time_constant_interval_s
    )

    rejecting_configs = (
        (
            SystemIdConfig(
                minimum_gain=default.model.steady_state_gain + 0.01
            ),
            "minimum gain",
        ),
        (
            SystemIdConfig(
                maximum_gain=default.model.steady_state_gain - 0.01
            ),
            "maximum gain",
        ),
        (
            SystemIdConfig(
                maximum_abs_bias_rad_s=abs(default.model.gyro_bias_rad_s) * 0.5
            ),
            "bias bound",
        ),
        (
            SystemIdConfig(
                maximum_design_condition_number=max(
                    1.000001,
                    default.diagnostics.design_condition_number * 0.99,
                )
            ),
            "condition bound",
        ),
    )
    for rejecting_config, error_match in rejecting_configs:
        with pytest.raises(IdentifiabilityError, match=error_match):
            fit_rate_axis_model(trace, holdout, config=rejecting_config)


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


def test_tightened_condition_bound_rejects_default_selected_model():
    with pytest.raises(IdentifiabilityError, match="tightened condition bound"):
        fit_rate_axis_model(
            _synthetic_trace(),
            _holdout(),
            config=_test_config(maximum_design_condition_number=1.000001),
        )


@pytest.mark.parametrize(
    "plant_override",
    [
        {"gain": 0.01, "noise_scale": 0.0},
        {"bias": 0.50, "noise_scale": 0.0},
    ],
)
def test_nonphysical_gain_and_bias_fits_are_rejected(
    plant_override: dict[str, float],
):
    with pytest.raises(IdentifiabilityError, match="no finite, well-conditioned"):
        fit_rate_axis_model(
            _synthetic_trace(**plant_override),
            _holdout(),
            config=SystemIdConfig(),
        )


def test_nonzero_yaw_fit_is_rejected_by_the_frozen_envelope():
    with pytest.raises(TraceValidationError, match="nonzero yaw"):
        fit_rate_axis_model(
            _synthetic_trace(axis=RateAxis.YAW),
            _holdout(),
            config=SystemIdConfig(),
        )


def test_exact_zoh_transition_at_interval_boundary_is_not_applied_early():
    time_constant_s = 0.100
    transition_ns = 100_000_000
    command_times = (0, transition_ns)
    command_values = (0.0, 1.0)

    at_transition = _advance_affine_state(
        np.zeros(3, dtype=np.float64),
        start_ns=0,
        end_ns=transition_ns,
        time_constant_s=time_constant_s,
        command_times=command_times,
        command_values=command_values,
    )
    after_transition = _advance_affine_state(
        at_transition,
        start_ns=transition_ns,
        end_ns=2 * transition_ns,
        time_constant_s=time_constant_s,
        command_times=command_times,
        command_values=command_values,
    )

    assert at_transition[1] == pytest.approx(0.0, abs=1e-15)
    assert after_transition[1] == pytest.approx(1.0 - math.exp(-1.0), abs=1e-15)


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
            RateExperimentSegment(200_000_000, 0.0),
            RateExperimentSegment(400_000_000, -0.12),
            RateExperimentSegment(500_000_000, 0.0),
        ),
    )

    assert definition.total_duration_ns == 1_700_000_000
    assert definition.maximum_abs_rate_rad_s == 0.12
    assert definition.maximum_signed_prefix_angle_rad == pytest.approx(0.048)
    assert definition.final_zero_settling_duration_ns == 500_000_000
    assert definition.maximum_adjacent_rate_step_rad_s == 0.12
    assert not hasattr(definition, "send")
    assert not hasattr(definition, "execute")
    assert not hasattr(definition, "restores_attitude")


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
                RateExperimentSegment(500_000_000, 0.0),
            ),
        )


@pytest.mark.parametrize(
    "segments,error_match",
    [
        (
            (
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(4_600_000_000, 0.10),
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(4_600_000_000, -0.10),
                RateExperimentSegment(500_000_000, 0.0),
            ),
            "ten-second",
        ),
        (
            (
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(400_000_000, 0.10),
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(200_000_000, -0.10),
                RateExperimentSegment(500_000_000, 0.0),
            ),
            "zero net",
        ),
        (
            (
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(2_100_000_000, 0.10),
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(2_100_000_000, -0.10),
                RateExperimentSegment(500_000_000, 0.0),
            ),
            "prefix angle",
        ),
        (
            (
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(400_000_000, 0.10),
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(400_000_000, -0.10),
                RateExperimentSegment(480_000_000, 0.0),
            ),
            "final exact-zero settling",
        ),
        (
            (
                RateExperimentSegment(200_000_000, 0.0),
                RateExperimentSegment(400_000_000, 0.12),
                RateExperimentSegment(400_000_000, -0.12),
                RateExperimentSegment(500_000_000, 0.0),
            ),
            "adjacent rate step",
        ),
    ],
)
def test_inert_experiment_definition_rejects_duration_area_prefix_settle_and_step(
    segments: tuple[RateExperimentSegment, ...],
    error_match: str,
):
    with pytest.raises(ValueError, match=error_match):
        RateExperimentDefinition(
            experiment_id="rejected-inert-shape",
            axis=RateAxis.ROLL,
            command_period_ns=20_000_000,
            segments=segments,
        )
