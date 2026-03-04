from __future__ import annotations

import math
from typing import Optional, Tuple

from .mpc import MPCPlanner
from .pid import PIDController
from .mixer import TRPYMixer, TRPYCommand, MixerConfig
from .types import ControlCommand, ControllerConfig, DroneState, TargetState


class FlightController:
    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        mixer_config: Optional[MixerConfig] = None,
    ):
        self.config = config or ControllerConfig()
        self.mpc = MPCPlanner(self.config.mpc)
        self.pid_vz = PIDController(self.config.vz_pid)
        self.pid_yaw = PIDController(self.config.yaw_pid)
        self.mixer = TRPYMixer(mixer_config or MixerConfig())

    def reset(self) -> None:
        self.pid_vz.reset()
        self.pid_yaw.reset()

    def step(self, state: DroneState, target: TargetState, dt: Optional[float] = None) -> ControlCommand:
        """
        MPC selects the best acceleration directly.
        XY accelerations pass through to the mixer unfiltered.
        Z uses a PID for altitude hold since gravity compensation is sensitive.
        Yaw uses a PID to track heading.
        """
        dt = dt or self.config.mpc.dt
        best_accel, desired_yaw = self.mpc.plan(state, target)

        ax = best_accel[0]
        ay = best_accel[1]

        # For Z, use PID on altitude error for stable hover + MPC az as feedforward
        target_vz = best_accel[2] * dt + state.velocity[2]
        az = self.pid_vz.update(target_vz, state.velocity[2], dt) + best_accel[2] * 0.5

        yaw_rate = self.pid_yaw.update(_wrap_angle(desired_yaw - state.yaw), 0.0, dt)

        return ControlCommand(
            ax=ax,
            ay=ay,
            az=az,
            yaw_rate=yaw_rate,
            desired_velocity=(best_accel[0], best_accel[1], best_accel[2]),
            desired_yaw=desired_yaw,
        )

    def step_trpy(
        self,
        state: DroneState,
        target: TargetState,
        dt: Optional[float] = None,
        current_roll: float = 0.0,
        current_pitch: float = 0.0,
    ) -> TRPYCommand:
        """
        Full pipeline: MPC → TRPY mixer.
        MPC acceleration goes directly to mixer (no velocity PID for XY).
        """
        cmd = self.step(state, target, dt)
        return self.mixer.mix(
            cmd,
            current_roll=current_roll,
            current_pitch=current_pitch,
            current_yaw=state.yaw,
        )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
