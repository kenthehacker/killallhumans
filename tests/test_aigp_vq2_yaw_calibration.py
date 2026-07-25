from __future__ import annotations

import copy

import pytest

from scripts import aigp_vq2_yaw_calibration as calibration


def test_plan_identity_timing_and_segment_counts_are_exact():
    plan = calibration.yaw_calibration_plan()

    assert plan["schema"] == "aigp-vq2-yaw-calibration-plan/1"
    assert plan["plan_id"] == (
        "vq2-build3385-training-yaw-envelope-calibration-v3"
    )
    assert plan["stage"] == "calibration-excite"
    assert plan["control_period_ns"] == 20_000_000
    assert plan["tick_count"] == 45
    assert plan["nominal_end_offset_ns"] == 900_000_000
    assert plan["powered_hard_expiry_offset_ns"] == 1_000_000_000

    assert [
        (
            segment["segment_id"],
            segment["last_tick"] - segment["first_tick"] + 1,
            segment["yaw_rate_rad_s"],
        )
        for segment in plan["segments"]
    ] == [
        ("neutral-initial", 12, 0.0),
        ("yaw-positive", 11, 0.08),
        ("neutral-reversal", 6, 0.0),
        ("yaw-negative", 11, -0.08),
        ("neutral-terminal", 5, 0.0),
    ]


def test_canonical_plan_hash_and_validation_are_deterministic():
    plan = calibration.yaw_calibration_plan()

    assert calibration.canonical_yaw_calibration_plan_sha256(plan) == (
        calibration.YAW_CALIBRATION_PLAN_SHA256
    )
    assert calibration.YAW_CALIBRATION_PLAN_SHA256 == (
        "9aa0a596e03ba685e3b5187b2940b0a3071a70e7e14686a398048bd1916ef91a"
    )
    assert calibration.validate_yaw_calibration_plan(plan) == plan

    reordered = {key: plan[key] for key in reversed(plan)}
    assert calibration.canonical_yaw_calibration_plan_sha256(reordered) == (
        calibration.YAW_CALIBRATION_PLAN_SHA256
    )


def test_plan_copy_is_defensive_and_frozen_plan_is_deeply_immutable():
    first = calibration.yaw_calibration_plan()
    second = calibration.yaw_calibration_plan()
    first["segments"][1]["yaw_rate_rad_s"] = 9.0
    assert second["segments"][1]["yaw_rate_rad_s"] == 0.08

    with pytest.raises(TypeError):
        calibration.FROZEN_YAW_CALIBRATION_PLAN["tick_count"] = 1
    with pytest.raises(TypeError):
        calibration.FROZEN_YAW_CALIBRATION_PLAN["command"]["thrust"] = 1.0
    with pytest.raises(TypeError):
        calibration.FROZEN_YAW_CALIBRATION_PLAN["segments"][1][
            "yaw_rate_rad_s"
        ] = 1.0


def test_every_tick_has_exact_timing_and_bounded_single_axis_command():
    ticks = list(calibration.iter_yaw_calibration_ticks())
    assert len(ticks) == 45
    assert ticks[0]["release_offset_ns"] == 0
    assert ticks[0]["end_offset_ns"] == 20_000_000
    assert ticks[-1]["release_offset_ns"] == 880_000_000
    assert ticks[-1]["end_offset_ns"] == 900_000_000
    assert all(
        tick["powered_expiry_offset_ns"] == 1_000_000_000 for tick in ticks
    )

    commands = [tick["command"] for tick in ticks]
    assert all(command["roll_rate_rad_s"] == 0.0 for command in commands)
    assert all(command["pitch_rate_rad_s"] == 0.0 for command in commands)
    assert all(command["thrust"] == 0.235 for command in commands)
    assert [command["yaw_rate_rad_s"] for command in commands] == (
        [0.0] * 12
        + [0.08] * 11
        + [0.0] * 6
        + [-0.08] * 11
        + [0.0] * 5
    )


def test_tick_lookup_resolves_exact_segments_and_absolute_deadlines():
    assert calibration.yaw_calibration_tick(11)["segment_id"] == (
        "neutral-initial"
    )
    assert calibration.yaw_calibration_tick(12)["segment_id"] == "yaw-positive"
    assert calibration.yaw_calibration_tick(22)["command"]["yaw_rate_rad_s"] == (
        0.08
    )
    assert calibration.yaw_calibration_tick(23)["segment_id"] == (
        "neutral-reversal"
    )
    assert calibration.yaw_calibration_tick(29)["segment_id"] == "yaw-negative"
    assert calibration.yaw_calibration_tick(40)["segment_id"] == (
        "neutral-terminal"
    )

    anchored = calibration.yaw_calibration_tick(
        12,
        anchor_monotonic_ns=8_000_000_000,
    )
    assert anchored == {
        "absolute_tick": 12,
        "segment_id": "yaw-positive",
        "release_monotonic_ns": 8_240_000_000,
        "end_monotonic_ns": 8_260_000_000,
        "powered_expiry_monotonic_ns": 9_000_000_000,
        "command": {
            "roll_rate_rad_s": 0.0,
            "pitch_rate_rad_s": 0.0,
            "yaw_rate_rad_s": 0.08,
            "thrust": 0.235,
        },
    }


@pytest.mark.parametrize("tick", [True, False, -1, 45, 1.0, "1", None])
def test_tick_lookup_rejects_non_exact_or_out_of_range_ticks(tick):
    with pytest.raises(calibration.YawCalibrationPlanError):
        calibration.yaw_calibration_tick(tick)
    with pytest.raises(calibration.YawCalibrationPlanError):
        calibration.yaw_calibration_command_for_tick(tick)


@pytest.mark.parametrize("anchor", [True, -1, 1.0, "1"])
def test_tick_lookup_rejects_invalid_monotonic_anchors(anchor):
    with pytest.raises(calibration.YawCalibrationPlanError):
        calibration.yaw_calibration_tick(0, anchor_monotonic_ns=anchor)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tick_count",), 44),
        (("control_period_ns",), 20_000_001),
        (("nominal_end_offset_ns",), 899_999_999),
        (("powered_hard_expiry_offset_ns",), 900_000_000),
        (("command", "thrust"), 0.236),
        (("command", "roll_rate_rad_s"), 0.01),
        (("segments", 1, "first_tick"), 13),
        (("segments", 1, "yaw_rate_rad_s"), 0.081),
        (("segments", 3, "yaw_rate_rad_s"), -0.081),
    ],
)
def test_validation_rejects_mutated_plan_content(path, value):
    plan = calibration.yaw_calibration_plan()
    target = plan
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(calibration.YawCalibrationPlanError):
        calibration.validate_yaw_calibration_plan(plan)


def test_validation_rejects_shape_types_and_identity_overrides():
    plan = calibration.yaw_calibration_plan()
    plan["unknown"] = 1
    with pytest.raises(calibration.YawCalibrationPlanError, match="unknown"):
        calibration.validate_yaw_calibration_plan(plan)

    plan = calibration.yaw_calibration_plan()
    plan["segments"] = tuple(plan["segments"])
    with pytest.raises(calibration.YawCalibrationPlanError, match="array"):
        calibration.validate_yaw_calibration_plan(plan)

    plan = calibration.yaw_calibration_plan()
    plan["command"]["thrust"] = 0
    with pytest.raises(calibration.YawCalibrationPlanError, match="float"):
        calibration.validate_yaw_calibration_plan(plan)

    with pytest.raises(
        calibration.YawCalibrationPlanError,
        match="must equal frozen",
    ):
        calibration.validate_yaw_calibration_plan(
            calibration.yaw_calibration_plan(),
            expected_sha256="f" * 64,
        )


def test_validation_returns_a_defensive_copy():
    plan = calibration.yaw_calibration_plan()
    validated = calibration.validate_yaw_calibration_plan(plan)
    validated["segments"][0]["last_tick"] = 99
    assert plan["segments"][0]["last_tick"] == 11

    nonfinite = copy.deepcopy(plan)
    nonfinite["command"]["thrust"] = float("nan")
    with pytest.raises(calibration.YawCalibrationPlanError):
        calibration.canonical_yaw_calibration_plan_sha256(nonfinite)
