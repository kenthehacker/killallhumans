"""
Iter-011 (closes iter-010 Composer-2 MAJOR + Codex#1 MAJOR 1 + Opus M4):
pin the drone-spec source-of-truth contract so silent drift between
`competition.drone_spec`, `DroneConstraints`, and `auto_velocity` is
caught immediately.

The iter-010 7-agent adversarial review found:
  - No test asserted `DroneConstraints().max_acceleration` matches the
    spec constant. A future "tuning" override to 18.0 would not fail.
  - `planning/auto_velocity.py` defines `DEFAULT_DRONE_MAX_ACCEL` and
    `DEFAULT_DRONE_MAX_SPEED_MPS` independently of `drone_spec`.
    Today the values coincide; tomorrow they could drift.

This module is intentionally tiny: it pins the contract and nothing
else. New duplicates added by future iters should be added here.
"""
from __future__ import annotations

from competition.drone_spec import (
    DEFAULT_MAX_ACCEL_MPS2,
    DEFAULT_MAX_VELOCITY_MPS,
    DEFAULT_MASS_KG,
    DEFAULT_MAX_THRUST_N,
    DEFAULT_MAX_TILT_RAD,
    DEFAULT_MAX_BODY_RATE_RAD_S,
    DEFAULT_MAX_JERK_MPS3,
    DEFAULT_GRAVITY_MPS2,
)
from planning.trajectory_optimizer import DroneConstraints


def test_drone_constraints_default_max_acceleration_matches_spec():
    """Iter-010 dropped DroneConstraints.max_acceleration default from
    20.0 to DEFAULT_MAX_ACCEL_MPS2 (15.0). A future change that
    reverts the import to a literal would silently re-introduce the
    bench-vs-planner mismatch this commit fixed."""
    assert DroneConstraints().max_acceleration == DEFAULT_MAX_ACCEL_MPS2


def test_drone_constraints_default_max_velocity_matches_spec():
    assert DroneConstraints().max_velocity == DEFAULT_MAX_VELOCITY_MPS


def test_drone_constraints_other_defaults_match_spec():
    """Single batch assertion across the remaining DroneConstraints
    fields that were rewired in iter-010."""
    c = DroneConstraints()
    assert c.mass == DEFAULT_MASS_KG
    assert c.max_thrust == DEFAULT_MAX_THRUST_N
    assert c.max_tilt_angle == DEFAULT_MAX_TILT_RAD
    assert c.max_body_rate == DEFAULT_MAX_BODY_RATE_RAD_S
    assert c.max_jerk == DEFAULT_MAX_JERK_MPS3
    assert c.gravity == DEFAULT_GRAVITY_MPS2


def test_auto_velocity_constants_match_drone_spec():
    """planning/auto_velocity.py still defines DEFAULT_DRONE_MAX_ACCEL
    and DEFAULT_DRONE_MAX_SPEED_MPS as local literals (avoids a
    planning→competition import). The values MUST match drone_spec
    or the 15-vs-20 class of mismatch reappears under different
    names."""
    from planning.auto_velocity import (
        DEFAULT_DRONE_MAX_ACCEL,
        DEFAULT_DRONE_MAX_SPEED_MPS,
    )
    assert DEFAULT_DRONE_MAX_ACCEL == DEFAULT_MAX_ACCEL_MPS2, (
        f"auto_velocity.DEFAULT_DRONE_MAX_ACCEL ({DEFAULT_DRONE_MAX_ACCEL}) "
        f"drifted from drone_spec.DEFAULT_MAX_ACCEL_MPS2 ({DEFAULT_MAX_ACCEL_MPS2})"
    )
    assert DEFAULT_DRONE_MAX_SPEED_MPS == DEFAULT_MAX_VELOCITY_MPS, (
        f"auto_velocity.DEFAULT_DRONE_MAX_SPEED_MPS ({DEFAULT_DRONE_MAX_SPEED_MPS}) "
        f"drifted from drone_spec.DEFAULT_MAX_VELOCITY_MPS ({DEFAULT_MAX_VELOCITY_MPS})"
    )
