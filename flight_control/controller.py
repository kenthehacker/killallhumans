import math
from typing import Tuple

from .mpc import MPCPlanner
from .pid import PIDController
from .types import ControlCommand, ControllerConfig, DroneState, TargetState


class FlightController:
    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.mpc = MPCPlanner(self.config.mpc)
        self.pid_vx = PIDController(self.config.velocity_pid)
        self.pid_vy = PIDController(self.config.velocity_pid)
        self.pid_vz = PIDController(self.config.velocity_pid)
        self.pid_yaw = PIDController(self.config.yaw_pid)

    def reset(self) -> None:
        self.pid_vx.reset()
        self.pid_vy.reset()
        self.pid_vz.reset()
        self.pid_yaw.reset()

    def step(self, state: DroneState, target: TargetState, dt: float | None = None) -> ControlCommand:
        dt = dt or self.config.mpc.dt
        desired_velocity, desired_yaw = self.mpc.plan(state, target)
        ax = self.pid_vx.update(desired_velocity[0], state.velocity[0], dt)
        ay = self.pid_vy.update(desired_velocity[1], state.velocity[1], dt)
        az = self.pid_vz.update(desired_velocity[2], state.velocity[2], dt)
        yaw_rate = self.pid_yaw.update(desired_yaw, state.yaw, dt)
        return ControlCommand(
            ax=ax,
            ay=ay,
            az=az,
            yaw_rate=yaw_rate,
            desired_velocity=desired_velocity,
            desired_yaw=desired_yaw,
        )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
