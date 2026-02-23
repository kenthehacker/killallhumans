from typing import Tuple

from .types import DroneState, TargetState, MPCConfig


class MPCPlanner:
    def __init__(self, config: MPCConfig):
        self.config = config

    def plan(self, state: DroneState, target: TargetState) -> Tuple[Tuple[float, float, float], float]:
        best_cost = float("inf")
        best_velocity = state.velocity
        best_yaw = target.yaw

        accel_choices = _accel_candidates(self.config.max_acceleration)

        for ax, ay, az in accel_choices:
            cost, v_candidate = _simulate_cost(
                state,
                target,
                self.config,
                (ax, ay, az),
            )
            if cost < best_cost:
                best_cost = cost
                best_velocity = v_candidate
        return best_velocity, best_yaw


def _accel_candidates(max_accel: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], ...]:
    ax_max, ay_max, az_max = max_accel
    ax_values = (-ax_max, -0.5 * ax_max, 0.0, 0.5 * ax_max, ax_max)
    ay_values = (-ay_max, -0.5 * ay_max, 0.0, 0.5 * ay_max, ay_max)
    az_values = (-az_max, -0.5 * az_max, 0.0, 0.5 * az_max, az_max)
    candidates = []
    for ax in ax_values:
        for ay in ay_values:
            for az in az_values:
                candidates.append((ax, ay, az))
    return tuple(candidates)


def _simulate_cost(
    state: DroneState,
    target: TargetState,
    config: MPCConfig,
    accel: Tuple[float, float, float],
) -> Tuple[float, Tuple[float, float, float]]:
    pos_x, pos_y, pos_z = state.position
    vel_x, vel_y, vel_z = state.velocity
    target_px, target_py, target_pz = target.position
    target_vx, target_vy, target_vz = target.velocity
    max_vx, max_vy, max_vz = config.max_velocity
    ax, ay, az = accel
    cost = 0.0

    for _ in range(config.horizon_steps):
        vel_x = _clamp(vel_x + ax * config.dt, -max_vx, max_vx)
        vel_y = _clamp(vel_y + ay * config.dt, -max_vy, max_vy)
        vel_z = _clamp(vel_z + az * config.dt, -max_vz, max_vz)
        pos_x += vel_x * config.dt
        pos_y += vel_y * config.dt
        pos_z += vel_z * config.dt

        pos_error = (
            (target_px - pos_x) ** 2
            + (target_py - pos_y) ** 2
            + (target_pz - pos_z) ** 2
        )
        vel_error = (
            (target_vx - vel_x) ** 2
            + (target_vy - vel_y) ** 2
            + (target_vz - vel_z) ** 2
        )
        accel_cost = ax ** 2 + ay ** 2 + az ** 2
        cost += (
            config.position_weight * pos_error
            + config.velocity_weight * vel_error
            + config.acceleration_weight * accel_cost
        )

    terminal_pos_error = (
        (target_px - pos_x) ** 2
        + (target_py - pos_y) ** 2
        + (target_pz - pos_z) ** 2
    )
    terminal_vel_error = (
        (target_vx - vel_x) ** 2
        + (target_vy - vel_y) ** 2
        + (target_vz - vel_z) ** 2
    )
    cost += (
        config.terminal_position_weight * terminal_pos_error
        + config.terminal_velocity_weight * terminal_vel_error
    )

    return cost, (vel_x, vel_y, vel_z)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value
