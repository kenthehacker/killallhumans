import unittest

from flight_control.controller import FlightController
from flight_control.mpc import MPCPlanner
from flight_control.pid import PIDController
from flight_control.types import ControllerConfig, DroneState, MPCConfig, PIDConfig, TargetState


class TestPIDController(unittest.TestCase):
    def test_pid_moves_toward_target(self) -> None:
        pid = PIDController(PIDConfig(2.0, 0.0, 0.0, 1.0, 10.0))
        output = pid.update(1.0, 0.0, 0.1)
        self.assertGreater(output, 0.0)


class TestMPCPlanner(unittest.TestCase):
    def test_mpc_velocity_points_to_target(self) -> None:
        config = MPCConfig(dt=0.05, horizon_steps=10)
        planner = MPCPlanner(config)
        state = DroneState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        target = TargetState((2.0, 0.0, 0.0))
        desired_velocity, _ = planner.plan(state, target)
        self.assertGreater(desired_velocity[0], 0.0)


class TestFlightController(unittest.TestCase):
    def test_controller_step_output(self) -> None:
        controller = FlightController(ControllerConfig())
        state = DroneState((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), 0.0)
        target = TargetState((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        command = controller.step(state, target, 0.05)
        self.assertEqual(len(command.desired_velocity), 3)
        self.assertAlmostEqual(command.yaw_rate, 0.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
