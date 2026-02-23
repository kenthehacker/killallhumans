from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DroneState:
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    yaw: float


@dataclass
class TargetState:
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0


@dataclass
class MPCConfig:
    dt: float = 0.05
    horizon_steps: int = 15
    max_velocity: Tuple[float, float, float] = (6.0, 6.0, 3.0)
    max_acceleration: Tuple[float, float, float] = (8.0, 8.0, 6.0)
    position_weight: float = 1.0
    velocity_weight: float = 0.3
    acceleration_weight: float = 0.05
    terminal_position_weight: float = 4.0
    terminal_velocity_weight: float = 0.5


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    integrator_limit: float = 1.0
    output_limit: float = 10.0


@dataclass
class ControllerConfig:
    mpc: MPCConfig = field(default_factory=MPCConfig)
    velocity_pid: PIDConfig = field(default_factory=lambda: PIDConfig(1.8, 0.1, 0.25, 2.0, 8.0))
    yaw_pid: PIDConfig = field(default_factory=lambda: PIDConfig(3.0, 0.05, 0.2, 1.5, 4.0))


@dataclass
class ControlCommand:
    ax: float
    ay: float
    az: float
    yaw_rate: float
    desired_velocity: Tuple[float, float, float]
    desired_yaw: float
