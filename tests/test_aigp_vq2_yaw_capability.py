from copy import deepcopy

import pytest

from scripts import aigp_vq2_yaw_capability as capability


def test_progressive_yaw_capability_plan_is_exact_and_symmetric():
    plan = capability.validate_yaw_capability_plan(
        capability.yaw_capability_plan()
    )

    assert plan["stage"] == "calibration-excite"
    assert plan["plan_id"] == capability.YAW_CAPABILITY_PLAN_ID
    assert plan["control_period_ns"] == 20_000_000
    assert plan["tick_count"] == 45
    assert plan["nominal_end_offset_ns"] == 900_000_000
    assert plan["powered_hard_expiry_offset_ns"] == 1_000_000_000
    assert plan["hold"] == {
        "target_roll_rad": 0.0,
        "target_pitch_rad": 0.05,
        "thrust": 0.285,
    }
    pulses = [
        segment["yaw_rate_rad_s"]
        for segment in plan["segments"]
        if segment["yaw_rate_rad_s"] != 0.0
    ]
    assert pulses == [
        value
        for level in capability.YAW_CAPABILITY_LEVELS_RAD_S
        for value in (level, -level)
    ]
    assert pulses == [0.10, -0.10]
    assert capability.canonical_yaw_capability_plan_sha256(plan) == (
        capability.YAW_CAPABILITY_PLAN_SHA256
    )


def test_progressive_yaw_capability_ticks_cover_exact_50_hz_slots():
    ticks = list(
        capability.iter_yaw_capability_ticks(
            anchor_monotonic_ns=10_000_000_000
        )
    )

    assert len(ticks) == capability.YAW_CAPABILITY_TICK_COUNT
    assert ticks[0]["release_monotonic_ns"] == 10_000_000_000
    assert ticks[-1]["end_monotonic_ns"] == 10_900_000_000
    assert all(
        later["release_monotonic_ns"]
        - earlier["release_monotonic_ns"]
        == capability.YAW_CAPABILITY_CONTROL_PERIOD_NS
        for earlier, later in zip(ticks, ticks[1:])
    )
    assert all(
        tick["powered_expiry_monotonic_ns"] == 11_000_000_000
        for tick in ticks
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["hold"].__setitem__("thrust", 0.286),
        lambda plan: plan["segments"][1].__setitem__(
            "yaw_rate_rad_s", 0.081
        ),
        lambda plan: plan["segments"][2].__setitem__("first_tick", 11),
        lambda plan: plan.__setitem__("tick_count", 80),
    ],
)
def test_progressive_yaw_capability_plan_rejects_mutation(mutation):
    plan = deepcopy(capability.yaw_capability_plan())
    mutation(plan)

    with pytest.raises(capability.YawCapabilityPlanError):
        capability.validate_yaw_capability_plan(plan)


@pytest.mark.parametrize("tick", [True, -1, 45])
def test_progressive_yaw_capability_tick_rejects_invalid_index(tick):
    with pytest.raises(capability.YawCapabilityPlanError):
        capability.yaw_capability_tick(tick)
