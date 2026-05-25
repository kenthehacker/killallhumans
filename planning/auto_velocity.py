"""
Geometry-derived safe trajectory velocity (iter-006, addresses iter-005
F3 MAJOR — replaces the 8.0 / 6.0 magic numbers with a principled
centripetal-acceleration-limited formula).

Physics: a drone turning at radius r at velocity v experiences centripetal
acceleration v²/r. If this exceeds the drone's max lateral acceleration
the controller cannot track the trajectory, regardless of how good the
gains are.

For three consecutive gates A → B → C with bend angle β at B:
    chord       ≈ (|AB| + |BC|) / 2                              (arithmetic mean)
    turn_radius ≈ chord / (2 · sin(β/2))                         (inscribed-arc approx)
    v_max       ≈ min(√(a_max · r_min) · safety_factor, drone_max_speed)

Tracks with tight bends (slalom β≈90°, chord≈3m → r≈2.1m → v_max≈4.6m/s) get
auto-throttled; tracks with mostly straight runs (β<30°, chord≈8-10m → r≈30m)
fall through to the drone's max-speed cap.

This avoids the 15→8→6 sequence of hand-tuned globals and per-track overrides
that the iter-005 review flagged as charter-violating magic numbers.

Iter-009b: the previously-named `DEFAULT_ABSOLUTE_CAP_MPS` is now
`DEFAULT_DRONE_MAX_SPEED_MPS`. The value (15.0 m/s) is physically grounded
— it matches `scripts/benchmark.py`'s kinematic-sim velocity saturation,
which is the binding constraint in the synthetic bench. The trajectory
optimizer's `DroneConstraints.max_velocity` also happens to be 15.0 m/s.
The old name and "race_01 ILC sweep" comment were misleading — this cap
is a drone-spec property, not a course artifact.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional


# Drone max lateral accel (m/s²). Matches `scripts/benchmark.py`'s
# kinematic-sim saturation at line ~478 (`max_accel = 15.0`). Note that
# `DroneConstraints.max_acceleration` in trajectory_optimizer.py is
# *20.0* — the optimizer assumes more headroom than the bench actually
# delivers. We use the bench's value (15.0) because tracking-error
# bound is dominated by the binding physical limit, not the planner's
# assumption.
DEFAULT_DRONE_MAX_ACCEL: float = 15.0

# Safety factor on the centripetal limit. The √(a·r) formula is the
# *kinematic* limit; in practice the tracker's PD overshoot, drag, and
# discretization error eat ~20-30%. 0.8 leaves a margin against tracker
# saturation without flooring achievable velocity on long straights.
DEFAULT_SAFETY_FACTOR: float = 0.8

# Drone max speed cap (m/s) — used as the upper bound when the geometry
# has no binding bend (mostly-straight tracks). Matches both the bench's
# kinematic-sim velocity saturation and the trajectory optimizer's
# `DroneConstraints.max_velocity`. Real AIGP-class drones can exceed
# this; the cap reflects the *synthetic bench's* drone, not the
# competition drone (acknowledging the sim-vs-real drone mismatch).
DEFAULT_DRONE_MAX_SPEED_MPS: float = 15.0

# Iter-009b BACKWARD-COMPAT: keep the old name as an alias so any
# external imports (notebooks, sweeps) still work. Removable once we've
# confirmed nothing imports it.
DEFAULT_ABSOLUTE_CAP_MPS: float = DEFAULT_DRONE_MAX_SPEED_MPS

# Minimum bend angle (radians) to consider a 3-gate sequence "turning."
# Below this, the centripetal limit is too lax to bound velocity and
# we fall through to the absolute cap.
_MIN_BEND_RAD: float = math.radians(5.0)


def _gate_position(gate) -> tuple:
    """Duck-type position extraction — accepts GateSpec, GateWaypoint, or any
    object with a `.position` 3-tuple."""
    return tuple(gate.position)


def derive_safe_max_velocity(
    gates: Iterable,
    drone_max_accel: float = DEFAULT_DRONE_MAX_ACCEL,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    absolute_cap_mps: float = DEFAULT_DRONE_MAX_SPEED_MPS,
) -> float:
    """Compute a centripetal-acceleration-limited safe max velocity for a
    sequence of gates.

    Algorithm:
      1. Walk every interior triplet (g[i-1], g[i], g[i+1]).
      2. Compute the bend angle β at g[i].
      3. Approximate the inscribed-arc radius r ≈ chord / (2·sin(β/2)).
      4. v_max_local = √(a_max · r) · safety_factor.
      5. Return min over all triplets, capped at `absolute_cap_mps`.

    Tracks with no turns (straight only) or fewer than 3 gates short-
    circuit to the absolute cap.

    Args:
        gates: ordered iterable of objects with `.position` 3-tuples.
        drone_max_accel: m/s², lateral acceleration limit. Default 15.0
            matches the synthetic kinematic sim.
        safety_factor: multiplier in [0, 1] on the kinematic limit. 0.8 is
            empirically reasonable; 0.6 is very conservative.
        absolute_cap_mps: hard upper bound on the returned velocity.

    Returns:
        Safe max trajectory velocity in m/s. Always ≤ absolute_cap_mps,
        always > 0. Falls back to absolute_cap_mps when the geometry
        is degenerate (< 3 gates, all straight, zero-length segments).
    """
    gate_list = list(gates)
    if len(gate_list) < 3:
        return float(absolute_cap_mps)

    min_radius = math.inf
    for i in range(1, len(gate_list) - 1):
        a = _gate_position(gate_list[i - 1])
        b = _gate_position(gate_list[i])
        c = _gate_position(gate_list[i + 1])
        ab = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
        len_ab = math.sqrt(sum(x * x for x in ab))
        len_bc = math.sqrt(sum(x * x for x in bc))
        if len_ab < 1e-3 or len_bc < 1e-3:
            continue
        cos_theta = (
            sum(ab[k] * bc[k] for k in range(3)) / (len_ab * len_bc)
        )
        cos_theta = max(-1.0, min(1.0, cos_theta))
        interior_angle = math.acos(cos_theta)  # π if straight, 0 if hairpin
        bend = math.pi - interior_angle         # 0 if straight, π if hairpin
        if bend < _MIN_BEND_RAD:
            continue  # essentially straight — no centripetal limit
        # Iter-006 review Opus F2: arithmetic-mean chord is a better
        # representative for asymmetric triplets than `min(|AB|, |BC|)`.
        # `min` aggressively undercounts the radius when one arm is much
        # longer than the other (e.g. ascent into a tight corner that
        # exits onto a long straight). Mean averages the two arms and
        # tracks the polynomial trajectory's smoothed bend more closely.
        chord = (len_ab + len_bc) / 2.0
        r = chord / (2.0 * math.sin(bend / 2.0))
        if r < min_radius:
            min_radius = r

    if not math.isfinite(min_radius):
        return float(absolute_cap_mps)

    v_kinematic = math.sqrt(drone_max_accel * min_radius)
    v_safe = v_kinematic * safety_factor
    return float(min(v_safe, absolute_cap_mps))
