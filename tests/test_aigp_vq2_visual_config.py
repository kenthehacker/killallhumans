from __future__ import annotations

import copy
import json

import pytest

from planning.vq2_visual_approach import (
    MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S,
    MAX_PASSAGE_SUSPENSION_EPOCHS,
    MAX_PASSAGE_SUSPENSION_FRESH_FRAMES,
    MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S,
    MAX_PASSAGE_SUSPENSION_TOTAL_FRESH_FRAMES,
)
from planning.vq2_visual_servo import (
    MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM,
    MAX_VISUAL_SEGMENT_DURATION_S,
    MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD,
    MAX_VISUAL_YAW_RATE_RAD_S,
    MIN_VISUAL_THRUST,
    PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S,
    PREPASS_CURRENT_MAX_ABS_X_NORM,
    PREPASS_CURRENT_MAX_ABS_Y_NORM,
    PREPASS_CURRENT_PROJECTION_HORIZON_S,
    PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S,
)
from scripts.aigp_vq2_visual_config import (
    VISUAL_CONFIG_SCHEMA,
    VISUAL_CONTROLLER_FAMILY,
    VisualConfigError,
    canonical_visual_config_sha256,
    default_visual_config,
    default_visual_config_mapping,
    validate_visual_config,
)


def test_default_visual_config_is_versioned_and_canonically_hashed():
    mapping = default_visual_config_mapping()
    config = default_visual_config()

    assert mapping["schema"] == VISUAL_CONFIG_SCHEMA
    assert mapping["controller_family"] == VISUAL_CONTROLLER_FAMILY
    assert mapping["servo"]["align_thrust"] == 0.275
    assert mapping["servo"]["brake_thrust"] == 0.275
    assert mapping["servo"]["advance_thrust"] == 0.295
    assert mapping["servo"]["yaw_error_gain"] == 0.30
    assert mapping["servo"]["roll_error_gain"] == 0.20
    assert mapping["servo"]["roll_rate_gain"] == 0.05
    assert mapping["lifecycle"]["next_gate_blend_max"] == 0.35
    assert config.to_effective_mapping() == mapping
    assert config.effective_config_sha256 == (
        "39da22a06d39e00c4168820d7828cde4bdb194d5c49e4d376adcce7ea53de615"
    )
    assert len(config.effective_config_sha256) == 64
    assert canonical_visual_config_sha256(mapping) == (
        config.effective_config_sha256
    )


def test_config_hash_is_independent_of_json_key_order():
    mapping = default_visual_config_mapping()
    reordered = json.loads(json.dumps(mapping, sort_keys=True))
    assert canonical_visual_config_sha256(reordered) == (
        canonical_visual_config_sha256(mapping)
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("servo", "required_corridor_frames"), True),
        (("servo", "roll_error_gain"), 0.201),
        (("servo", "advance_thrust"), 0.34),
        (("servo", "yaw_error_gain"), float("nan")),
        (("lifecycle", "next_gate_blend_max"), 0.36),
        (("lifecycle", "required_improving_frames"), 2),
        (("lifecycle", "restricted_alignment_duration_s"), 0.900001),
        (("lifecycle", "restricted_alignment_duration_s"), 2.0),
        (("lifecycle", "launch_boost_duration_s"), 0.449999),
        (("lifecycle", "launch_pitch_blend_s"), 0.799999),
    ],
)
def test_config_rejects_values_outside_reviewed_tuning_bounds(path, value):
    mapping = default_visual_config_mapping()
    mapping[path[0]][path[1]] = value
    with pytest.raises(VisualConfigError):
        validate_visual_config(mapping)


def test_config_rejects_unknown_or_missing_fields():
    unknown = default_visual_config_mapping()
    unknown["global_safety_caps"] = {"yaw": 9.0}
    with pytest.raises(VisualConfigError, match="unknown"):
        validate_visual_config(unknown)

    missing = default_visual_config_mapping()
    del missing["servo"]["brake_thrust"]
    with pytest.raises(VisualConfigError, match="missing"):
        validate_visual_config(missing)

    non_string = default_visual_config_mapping()
    non_string["servo"][7] = 0.0
    with pytest.raises(VisualConfigError, match="exact strings"):
        validate_visual_config(non_string)


def test_config_rejects_wrong_schema_and_controller_family():
    mapping = default_visual_config_mapping()
    mapping["schema"] = "aigp-vq2-visual-navigation-config/999"
    with pytest.raises(VisualConfigError, match="schema"):
        validate_visual_config(mapping)

    mapping = default_visual_config_mapping()
    mapping["controller_family"] = "retired-recovery-controller"
    with pytest.raises(VisualConfigError, match="controller_family"):
        validate_visual_config(mapping)


def test_immutable_safety_envelopes_are_not_configuration_fields():
    serialized = json.dumps(default_visual_config_mapping(), sort_keys=True)
    for forbidden in (
        "max_command_rate",
        "max_attitude",
        "max_collision",
        "cleanup",
        "watchdog",
        "max_segment_yaw_excursion",
        "max_segment_duration",
        "prepass_current_max",
        "prepass_next_max",
        "prepass_projection",
        "passage_suspension",
    ):
        assert forbidden not in serialized

    assert MAX_VISUAL_YAW_RATE_RAD_S == 0.12
    assert MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD > 0.05
    assert MAX_VISUAL_SEGMENT_DURATION_S == 8.0
    assert PREPASS_CURRENT_MAX_ABS_X_NORM == 0.20
    assert PREPASS_CURRENT_MAX_ABS_Y_NORM == 0.28
    assert PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S == 0.60
    assert PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S == 0.60
    assert PREPASS_CURRENT_PROJECTION_HORIZON_S == 0.10
    assert MAX_PASSAGE_SUSPENSION_FRESH_FRAMES == 2
    assert MAX_PASSAGE_SUSPENSION_TOTAL_FRESH_FRAMES == 4
    assert MAX_PASSAGE_SUSPENSION_EPOCHS == 3
    assert MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S == 0.12
    assert MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S == 0.20
    assert MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM == 0.01


def test_validation_returns_new_immutable_nested_values():
    mapping = default_visual_config_mapping()
    config = validate_visual_config(mapping)
    mutated = copy.deepcopy(mapping)
    mutated["servo"]["advance_thrust"] = 0.27

    assert config.servo.advance_thrust != mutated["servo"]["advance_thrust"]
    with pytest.raises(Exception):
        config.servo.advance_thrust = 0.27
