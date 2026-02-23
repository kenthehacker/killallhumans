from dataclasses import dataclass

from .types import PIDConfig


@dataclass
class PIDState:
    integrator: float = 0.0
    previous_error: float = 0.0


class PIDController:
    def __init__(self, config: PIDConfig):
        self.config = config
        self.state = PIDState()

    def reset(self) -> None:
        self.state = PIDState()

    def update(self, target: float, measurement: float, dt: float) -> float:
        error = target - measurement
        self.state.integrator += error * dt
        self.state.integrator = _clamp(
            self.state.integrator,
            -self.config.integrator_limit,
            self.config.integrator_limit,
        )
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.state.previous_error) / dt
        self.state.previous_error = error
        output = (
            self.config.kp * error
            + self.config.ki * self.state.integrator
            + self.config.kd * derivative
        )
        return _clamp(output, -self.config.output_limit, self.config.output_limit)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value
