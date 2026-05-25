"""
Single source of truth for the synthetic-bench drone dynamics envelope.

Iter-010 (Opus + Composer 3-agent planning consensus, .loop/research/
next_iter_*.md): the iter-009 F9 fix decoupled racing-line SELECTION
velocity from EXECUTION velocity, but the same class of cross-module
mismatch was still open on the ACCELERATION axis:

  - scripts/benchmark.py:486 saturates lateral accel at 15 m/s²
    (kinematic sim's binding constraint)
  - planning/auto_velocity.py:33 derives v_max from 15 m/s²
  - planning/trajectory_optimizer.py:34 — DroneConstraints.max_acceleration
    defaulted to 20.0 m/s² (assumed ~2g headroom the bench cannot deliver)

A polynomial trajectory time-allocated under a 20 m/s² budget commands
accelerations the bench then clamps to 15 → feedforward mismatch and
gate-timing stress. Both planning agents independently rated this as
the #1 next iter (impact=High, effort=S-M, risk=Low because matrix
gates catch regressions).

This module centralises every drone-spec constant the synthetic stack
reads. Each value carries provenance — bench-empirical, spec-derived,
or placeholder pending calibration — so future maintainers can tell
the difference at a glance.

NB: this is the SYNTHETIC BENCH'S drone, not the AIGP competition
drone. Real AIGP-class 280×280×160mm racing drones likely exceed
these limits; calibration.py will bind real values once the SITL
binary lands. See competition/aigp_geometry.py for the actual
chassis dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- Bench-empirical: the kinematic sim's binding clamps ----------------------
# scripts/benchmark.py saturates at these values inside its 120Hz loop;
# all upstream planners/trackers must respect them or generate
# trajectories the bench cannot execute.

DEFAULT_MAX_ACCEL_MPS2: float = 15.0
"""Maximum lateral acceleration the synthetic bench will produce.

Source: scripts/benchmark.py:486 `max_accel = 15.0` (~1.5g). Anything
the planner commands above this gets vector-clamped, distorting the
intended trajectory. Was 20.0 in DroneConstraints pre-iter-010 — that
default was unreachable under the bench's actual saturation."""

DEFAULT_MAX_VELOCITY_MPS: float = 15.0
"""Maximum linear velocity the bench's tracker clamps to.

Source: scripts/benchmark.py:286 `max_speed = 15.0` in the synthetic
PD loop. Also doubles as the bench's max-speed sanity ceiling. The
auto-derived per-track velocity (planning/auto_velocity.py) caps at
this value."""


# --- Bench-proxy: synthetic stand-ins for a real drone ------------------------
# These are NOT the AIGP chassis; they're a Crazyflie-class 1 kg point-
# mass-with-drag model. Replace via calibration when SITL data lands.

DEFAULT_MASS_KG: float = 1.0
"""Drone mass for the synthetic kinematic sim. Placeholder — real
AIGP-class drones are heavier (~600-900 g for 280mm chassis, with
batteries). Source: scripts/benchmark.py:159 and TrackerConfig."""

DEFAULT_MAX_THRUST_N: float = 20.0
"""Maximum collective thrust the tracker can command. 20 N at 1 kg =
~2g lift envelope. Source: TrackerConfig.max_thrust_n. NOT used by
the kinematic bench (which is force/acceleration-saturated, not
thrust-modelled)."""

DEFAULT_LINEAR_DRAG_PER_MASS: float = 0.5
"""Linear drag coefficient applied to velocity in the synthetic kinematic
sim. `a_drag = -drag·v`. Source: scripts/benchmark.py:488. No physical
basis; tuned to keep the proxy stable."""


# --- Spec-derived / attitude limits -------------------------------------------

DEFAULT_MAX_TILT_RAD: float = 0.85
"""Maximum roll/pitch angle. ~49°. Source: DroneConstraints. Tilt-clamp
keeps the attitude loop tractable; aggressive AIGP corners can saturate
this without indicating a bug."""

DEFAULT_MAX_BODY_RATE_RAD_S: float = 6.0
"""Maximum body-rate magnitude (rad/s). Source: DroneConstraints."""

DEFAULT_MAX_JERK_MPS3: float = 50.0
"""Maximum jerk in min-snap polynomial time allocation. Source:
DroneConstraints. Bench has no jerk saturation; this is a planning-
side smoothness penalty only."""

DEFAULT_YAW_RATE_MAX_RAD_S: float = 4.0
"""Maximum yaw-rate the kinematic bench commands. Source:
scripts/benchmark.py:489."""

DEFAULT_GRAVITY_MPS2: float = 9.81
"""Standard gravity. Source: physics."""


@dataclass(frozen=True)
class DroneSpec:
    """Frozen drone-dynamics envelope. Instantiate at module boundaries
    where downstream code expects a dataclass; literal constants above
    are for direct import (faster, no allocation).

    Iter-010: this dataclass replaces the inline `DroneConstraints`
    default values in planning/trajectory_optimizer.py — that class
    becomes a thin wrapper that reads its defaults from here, so
    bench / planner / tracker all see the same numbers.
    """
    mass_kg: float = DEFAULT_MASS_KG
    max_thrust_n: float = DEFAULT_MAX_THRUST_N
    max_velocity_mps: float = DEFAULT_MAX_VELOCITY_MPS
    max_acceleration_mps2: float = DEFAULT_MAX_ACCEL_MPS2
    max_jerk_mps3: float = DEFAULT_MAX_JERK_MPS3
    max_tilt_rad: float = DEFAULT_MAX_TILT_RAD
    max_body_rate_rad_s: float = DEFAULT_MAX_BODY_RATE_RAD_S
    linear_drag_per_mass: float = DEFAULT_LINEAR_DRAG_PER_MASS
    yaw_rate_max_rad_s: float = DEFAULT_YAW_RATE_MAX_RAD_S
    gravity_mps2: float = DEFAULT_GRAVITY_MPS2
