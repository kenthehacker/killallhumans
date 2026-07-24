from __future__ import annotations

from copy import deepcopy
import math

import pytest

from scripts import aigp_vq2_controller_config as config_module
from scripts import aigp_vq2_run as runner


def _default():
    return config_module.default_controller_config_mapping()


def _set_path(document, path, value):
    group, field = path.split(".", 1)
    document[group][field] = value
    return document


def _sustained_preshape():
    document = _default()
    document["phase_timing"]["gate0_preshape_max_duration_s"] = 1.20
    document["turn_cue"]["sustained_preshape_enabled"] = True
    document["turn_cue"]["exit_counterroll_enabled"] = False
    document["turn_cue"]["preshape_end_gate_area_scale"] = 20.0
    document["turn_cue"]["preturn_roll_cap_rad"] = 0.12
    document["yaw_control"]["gate0_turn_score_gain"] = -0.20
    document["yaw_control"]["gate0_command_rate_cap_rad_s"] = 0.04
    document["forward_braking"]["gate0_turn_pitch_rad"] = 0.04
    document["forward_braking"]["gate0_turn_thrust_cap"] = 0.24
    return document


def test_default_config_is_deeply_immutable_and_returns_fresh_copies():
    with pytest.raises(TypeError):
        config_module.DEFAULT_CONTROLLER_CONFIG["schema"] = "changed"
    with pytest.raises(TypeError):
        config_module.DEFAULT_CONTROLLER_CONFIG["phase_timing"][
            "gate0_boost_until_s"
        ] = 1.0

    first = _default()
    second = _default()
    first["phase_timing"]["gate0_boost_until_s"] = 1.0
    assert second["phase_timing"]["gate0_boost_until_s"] == 0.45


def test_default_effective_mapping_preserves_current_runner_behavior():
    effective = config_module.default_controller_config()
    mapping = effective.to_effective_mapping()

    assert mapping["schema"] == "aigp-vq2-controller-config/1"
    assert (
        mapping["controller_family"]
        == "aigp-vq2-gate0-gate1-recenter/15"
    )
    assert mapping["phase_timing"] == {
        "gate0_boost_until_s": 0.45,
        "gate0_pitch_blend_s": runner.GATE0_PITCH_BLEND_S,
        "gate0_preshape_max_duration_s": runner.SIGN_ID_YAW_PULSE_DURATION_S,
        "post_gate_observation_duration_s": (
            runner.POST_GATE_OBSERVATION_TIMEOUT_S
        ),
        "gate1_recenter_duration_s": runner.GATE1_RECENTER_DURATION_S,
    }
    assert mapping["turn_cue"] == {
        "preturn_enabled": True,
        "exit_counterroll_enabled": True,
        "sustained_preshape_enabled": False,
        "min_gate_area_scale": runner.COURSE_LINE_PRETURN_MIN_GATE_AREA_SCALE,
        "min_abs_score": runner.COURSE_LINE_PRETURN_MIN_SCORE,
        "preturn_gain": runner.COURSE_LINE_PRETURN_GAIN,
        "preturn_roll_cap_rad": runner.COURSE_LINE_PRETURN_LIMIT_RAD,
        "preshape_end_gate_area_scale": (
            runner.COURSE_LINE_PRETURN_TAPER_AREA_SCALE
        ),
        "exit_counterroll_onset_area_scale": (
            runner.COURSE_LINE_EXIT_COUNTERROLL_ONSET_AREA_SCALE
        ),
        "exit_counterroll_cap_rad": runner.COURSE_LINE_EXIT_COUNTERROLL_RAD,
    }
    assert mapping["roll_control"] == {
        "gate0_centering_gain": 0.15,
        "gate0_target_cap_rad": 0.08,
        "gate1_error_gain": runner.GATE1_RECENTER_ROLL_GAIN,
        "gate1_error_rate_gain": runner.GATE1_RECENTER_ROLL_RATE_GAIN,
        "gate1_target_cap_rad": runner.GATE1_RECENTER_MAX_ROLL_RAD,
        "command_rate_cap_rad_s": (
            runner.GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S
        ),
    }
    assert mapping["yaw_control"] == {
        "gate0_turn_score_gain": 0.0,
        "gate0_command_rate_cap_rad_s": 0.0,
        "gate1_error_gain": 0.0,
        "gate1_deadband_normalized_x": (
            runner.GATE1_RECENTER_CORRIDOR_NORMALIZED_X
        ),
        "command_rate_cap_rad_s": 0.0,
    }
    assert mapping["forward_braking"] == {
        "gate0_turn_pitch_rad": 0.0,
        "gate0_turn_thrust_cap": 0.32,
        "pitch_command_rate_cap_rad_s": (
            runner.GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S
        ),
        "gate1_forward_thrust": runner.GATE1_RECENTER_THRUST,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("schema", "aigp-vq2-controller-config/2", "must equal"),
        ("schema", 1, "must equal"),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/1",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/2",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/3",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/4",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/5",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/6",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/7",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/8",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/9",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/10",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/11",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/12",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/13",
            "must equal",
        ),
        (
            "controller_family",
            "aigp-vq2-gate0-gate1-recenter/14",
            "must equal",
        ),
        ("controller_family", "other", "must equal"),
        ("controller_family", True, "must equal"),
    ),
)
def test_schema_and_controller_family_are_exact(field, value, match):
    document = _default()
    document[field] = value
    with pytest.raises(config_module.ControllerConfigError, match=match):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize("document", (None, [], "config", 1, True))
def test_root_must_be_a_mapping(document):
    with pytest.raises(config_module.ControllerConfigError, match="mapping"):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("group", "field"),
    (
        ("root", "phase_timing"),
        ("phase_timing", "gate0_boost_until_s"),
        ("turn_cue", "min_abs_score"),
        ("roll_control", "gate1_error_gain"),
        ("yaw_control", "command_rate_cap_rad_s"),
        ("forward_braking", "gate1_forward_thrust"),
    ),
)
def test_missing_required_fields_are_rejected(group, field):
    document = _default()
    target = document if group == "root" else document[group]
    del target[field]
    with pytest.raises(config_module.ControllerConfigError, match="missing"):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    "group",
    (
        "root",
        "phase_timing",
        "turn_cue",
        "roll_control",
        "yaw_control",
        "forward_braking",
    ),
)
def test_unknown_fields_are_rejected_at_every_level(group):
    document = _default()
    target = document if group == "root" else document[group]
    target["unexpected"] = 1
    with pytest.raises(config_module.ControllerConfigError, match="unknown"):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("group", "new_field", "retired_field"),
    (
        (
            "phase_timing",
            "gate0_preshape_max_duration_s",
            "gate0_yaw_brake_duration_s",
        ),
        (
            "turn_cue",
            "preshape_end_gate_area_scale",
            "preturn_taper_area_scale",
        ),
    ),
)
def test_retired_family_one_field_names_are_rejected(
    group,
    new_field,
    retired_field,
):
    document = _default()
    document[group][retired_field] = document[group].pop(new_field)
    with pytest.raises(
        config_module.ControllerConfigError,
        match="missing=.*unknown=",
    ):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    "lifecycle_field",
    (
        "control_hz",
        "go_delay_s",
        "stream_freshness_s",
        "collision_impulse_limit",
        "max_measured_body_rate_rad_s",
        "max_preshape_pitch_objective_delta_rad",
        "gate0_longitudinal_brake_start_area_scale",
        "gate0_longitudinal_brake_max_duration_s",
        "cleanup_timeout_s",
        "reset_retry_count",
        "no_passage_width_px",
    ),
)
def test_lifecycle_safety_invariants_are_not_configurable(lifecycle_field):
    document = _default()
    document[lifecycle_field] = 1
    with pytest.raises(config_module.ControllerConfigError, match="unknown"):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        ("phase_timing.gate0_boost_until_s", True, "finite number"),
        ("turn_cue.min_abs_score", "0.04", "finite number"),
        ("roll_control.gate1_error_gain", None, "finite number"),
        ("yaw_control.gate1_error_gain", math.nan, "finite number"),
        ("yaw_control.command_rate_cap_rad_s", math.inf, "finite number"),
        ("forward_braking.gate1_forward_thrust", -math.inf, "finite number"),
        ("turn_cue.preturn_enabled", 1, "exact bool"),
        ("turn_cue.exit_counterroll_enabled", "true", "exact bool"),
        ("turn_cue.sustained_preshape_enabled", 1, "exact bool"),
    ),
)
def test_malformed_field_values_are_rejected(path, value, match):
    document = _set_path(_default(), path, value)
    with pytest.raises(config_module.ControllerConfigError, match=match):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("phase_timing.gate0_boost_until_s", 0.449),
        ("phase_timing.gate0_boost_until_s", 1.001),
        ("phase_timing.gate0_pitch_blend_s", 0.799),
        ("phase_timing.gate0_preshape_max_duration_s", 0.039),
        ("phase_timing.gate0_preshape_max_duration_s", 1.201),
        ("phase_timing.post_gate_observation_duration_s", 0.099),
        ("phase_timing.gate1_recenter_duration_s", 0.601),
        ("turn_cue.min_gate_area_scale", 1.299),
        ("turn_cue.min_abs_score", 0.251),
        ("turn_cue.preturn_gain", 0.801),
        ("turn_cue.preturn_roll_cap_rad", 0.131),
        ("turn_cue.preshape_end_gate_area_scale", 7.999),
        ("turn_cue.preshape_end_gate_area_scale", 20.001),
        ("turn_cue.exit_counterroll_onset_area_scale", 3.499),
        ("turn_cue.exit_counterroll_cap_rad", 0.081),
        ("roll_control.gate0_centering_gain", 0.151),
        ("roll_control.gate0_target_cap_rad", 0.081),
        ("roll_control.gate1_error_gain", -0.241),
        ("roll_control.gate1_error_gain", 0.241),
        ("roll_control.gate1_error_rate_gain", 0.026),
        ("roll_control.gate1_target_cap_rad", 0.121),
        ("roll_control.command_rate_cap_rad_s", 0.121),
        ("yaw_control.gate0_turn_score_gain", -0.801),
        ("yaw_control.gate0_command_rate_cap_rad_s", 0.081),
        ("yaw_control.gate1_error_gain", -0.121),
        ("yaw_control.gate1_error_gain", 0.001),
        ("yaw_control.gate1_deadband_normalized_x", 0.351),
        ("yaw_control.command_rate_cap_rad_s", 0.081),
        ("forward_braking.gate0_turn_pitch_rad", 0.081),
        ("forward_braking.gate0_turn_thrust_cap", 0.321),
        ("forward_braking.pitch_command_rate_cap_rad_s", 0.121),
        ("forward_braking.gate1_forward_thrust", 0.301),
    ),
)
def test_numeric_values_outside_conservative_bounds_are_rejected(path, value):
    document = _set_path(_default(), path, value)
    with pytest.raises(config_module.ControllerConfigError, match="within"):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("group", "field", "value"),
    (
        ("turn_cue", "required_frames", 3),
        ("turn_cue", "max_age_s", 0.25),
        ("forward_braking", "gate1_target_pitch_rad", -0.10),
        ("forward_braking", "post_gate_hold_thrust", 0.275),
    ),
)
def test_runtime_owned_fields_are_rejected(group, field, value):
    document = _default()
    document[group][field] = value
    with pytest.raises(config_module.ControllerConfigError, match="unknown"):
        config_module.validate_controller_config(document)


def test_exit_counterroll_requires_preturn():
    document = _default()
    document["turn_cue"]["preturn_enabled"] = False
    with pytest.raises(config_module.ControllerConfigError, match="requires"):
        config_module.validate_controller_config(document)


def test_turn_cue_area_phase_boundary_is_admitted():
    document = _default()
    document["turn_cue"]["exit_counterroll_onset_area_scale"] = 8.0
    document["turn_cue"]["preshape_end_gate_area_scale"] = 8.0
    config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gate0_preshape_max_duration_s", 0.22),
        ("preshape_end_gate_area_scale", 9.0),
    ),
)
def test_disabled_sustained_preshape_requires_exact_legacy_shape(field, value):
    document = _default()
    group = (
        "phase_timing"
        if field == "gate0_preshape_max_duration_s"
        else "turn_cue"
    )
    document[group][field] = value
    with pytest.raises(config_module.ControllerConfigError, match="exact legacy"):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("preturn_enabled", "exit_counterroll_enabled"),
    (
        (False, False),
        (True, True),
    ),
)
def test_sustained_preshape_requires_preturn_without_exit_counterroll(
    preturn_enabled,
    exit_counterroll_enabled,
):
    document = _sustained_preshape()
    document["turn_cue"]["preturn_enabled"] = preturn_enabled
    document["turn_cue"]["exit_counterroll_enabled"] = (
        exit_counterroll_enabled
    )
    with pytest.raises(
        config_module.ControllerConfigError,
        match="preturn enabled.*counterroll disabled",
    ):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("yaw_control.gate0_turn_score_gain", -0.401),
        ("yaw_control.gate0_command_rate_cap_rad_s", 0.041),
        ("forward_braking.gate0_turn_pitch_rad", 0.041),
        ("forward_braking.gate0_turn_thrust_cap", 0.239),
        ("turn_cue.preturn_roll_cap_rad", 0.121),
    ),
)
def test_sustained_preshape_has_tighter_control_envelope(path, value):
    document = _set_path(_sustained_preshape(), path, value)
    with pytest.raises(
        config_module.ControllerConfigError,
        match="sustained preshape exceeds",
    ):
        config_module.validate_controller_config(document)


def test_sustained_preshape_boundary_is_admitted():
    document = _sustained_preshape()
    document["yaw_control"]["gate0_turn_score_gain"] = -0.40
    document["yaw_control"]["gate0_command_rate_cap_rad_s"] = 0.04
    document["forward_braking"]["gate0_turn_pitch_rad"] = 0.04
    document["forward_braking"]["gate0_turn_thrust_cap"] = 0.24

    effective = config_module.validate_controller_config(document)

    assert effective.turn_cue.sustained_preshape_enabled is True
    assert effective.phase_timing.gate0_preshape_max_duration_s == 1.20
    assert effective.turn_cue.preshape_end_gate_area_scale == 20.0


def test_requested_post_gate_phases_cannot_extend_combined_ceiling():
    document = _default()
    document["phase_timing"]["post_gate_observation_duration_s"] = 0.20
    document["phase_timing"]["gate1_recenter_duration_s"] = 0.60
    config_module.validate_controller_config(document)

    # Each value remains independently in range, but their composition does not.
    document["phase_timing"]["post_gate_observation_duration_s"] = 0.19
    document["phase_timing"]["gate1_recenter_duration_s"] = 0.60
    config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("gain", "cap"),
    (
        (0.0, 0.08),
        (-0.08, 0.0),
    ),
)
def test_yaw_gain_and_cap_must_enable_or_disable_together(gain, cap):
    document = _default()
    document["yaw_control"]["gate1_error_gain"] = gain
    document["yaw_control"]["command_rate_cap_rad_s"] = cap
    with pytest.raises(config_module.ControllerConfigError, match="together"):
        config_module.validate_controller_config(document)


@pytest.mark.parametrize(
    ("gain", "cap"),
    (
        (0.0, 0.08),
        (-0.20, 0.0),
    ),
)
def test_gate0_yaw_gain_and_cap_must_enable_or_disable_together(gain, cap):
    document = _default()
    document["yaw_control"]["gate0_turn_score_gain"] = gain
    document["yaw_control"]["gate0_command_rate_cap_rad_s"] = cap
    with pytest.raises(config_module.ControllerConfigError, match="Gate-0.*together"):
        config_module.validate_controller_config(document)


def test_bounded_negative_early_turn_yaw_tuning_is_admitted():
    document = _default()
    document["yaw_control"]["gate0_turn_score_gain"] = -0.20
    document["yaw_control"]["gate0_command_rate_cap_rad_s"] = 0.08
    document["forward_braking"]["gate0_turn_pitch_rad"] = 0.04
    document["forward_braking"]["gate0_turn_thrust_cap"] = 0.28
    effective = config_module.validate_controller_config(document)

    assert effective.yaw_control.gate0_turn_score_gain == -0.20
    assert effective.yaw_control.gate0_command_rate_cap_rad_s == 0.08
    assert effective.forward_braking.gate0_turn_pitch_rad == 0.04
    assert effective.forward_braking.gate0_turn_thrust_cap == 0.28


def test_calibrated_negative_yaw_tuning_is_inside_the_schema_envelope():
    document = _default()
    document["yaw_control"]["gate1_error_gain"] = -0.08
    document["yaw_control"]["command_rate_cap_rad_s"] = 0.08
    effective = config_module.validate_controller_config(document)
    assert effective.yaw_control.gate1_error_gain == -0.08
    assert effective.yaw_control.command_rate_cap_rad_s == 0.08


def test_gate0_and_gate1_calibrated_yaw_can_be_enabled_together():
    document = _sustained_preshape()
    document["yaw_control"]["gate1_error_gain"] = -0.08
    document["yaw_control"]["command_rate_cap_rad_s"] = 0.08

    effective = config_module.validate_controller_config(document)

    assert effective.yaw_control.gate0_turn_score_gain == -0.20
    assert effective.yaw_control.gate1_error_gain == -0.08


def test_effective_config_hash_is_canonical_and_normalizes_numbers():
    first = _default()
    second = {
        key: deepcopy(first[key])
        for key in reversed(tuple(first))
    }
    second["phase_timing"] = {
        key: second["phase_timing"][key]
        for key in reversed(tuple(second["phase_timing"]))
    }
    second["turn_cue"]["min_abs_score"] = 0.04
    second["turn_cue"]["preturn_gain"] = 0.8

    first_hash = config_module.canonical_effective_config_sha256(first)
    second_hash = config_module.canonical_effective_config_sha256(second)
    assert first_hash == second_hash
    assert len(first_hash) == 64
    assert first_hash == config_module.default_controller_config().effective_config_sha256


def test_effective_config_hash_changes_for_a_valid_controller_change():
    baseline = config_module.default_controller_config()
    changed = _default()
    changed["roll_control"]["gate1_error_gain"] = -0.20
    changed_config = config_module.validate_controller_config(changed)

    assert (
        changed_config.effective_config_sha256
        != baseline.effective_config_sha256
    )


def test_effective_mapping_is_a_normalized_complete_round_trip():
    document = _default()
    document["phase_timing"]["gate0_boost_until_s"] = 1
    config = config_module.validate_controller_config(document)
    effective = config.to_effective_mapping()

    assert effective["phase_timing"]["gate0_boost_until_s"] == 1.0
    assert type(effective["phase_timing"]["gate0_boost_until_s"]) is float
    assert (
        config_module.validate_controller_config(effective)
        == config
    )
