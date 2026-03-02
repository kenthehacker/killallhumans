"""
Crazyflie CF2X drone using gym-pybullet-drones physics.

Uses CtrlAviary for real Crazyflie physics and a tilt-limited outer position
loop feeding into DSLPIDControl's inner attitude controller.

DSLPIDControl's built-in position PID has no tilt-angle limit, so large position
errors (8+ meters) command extreme pitch → flip. We replicate the position PID but
clip the horizontal thrust vector to enforce a max tilt angle (35 degrees) before
converting to target Euler angles for the attitude controller.

Exposes the same get_state() / get_camera_image() interface as the old QuadrotorDrone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pybullet as p
except ImportError:
    p = None

try:
    from gym_pybullet_drones.envs import CtrlAviary
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    _GPD_AVAILABLE = True
except ImportError:
    _GPD_AVAILABLE = False


# Maximum tilt angle before the drone risks losing altitude control.
# DSLPIDControl has no built-in tilt limit; we enforce it here.
_MAX_TILT_RAD = math.radians(35)


@dataclass
class GPDDroneConfig:
    # DSLPIDControl was tuned at 48 Hz. Running it at higher frequency makes the
    # attitude D-term (proportional to delta_error/dt) blow up → summersault.
    # Physics runs at 240 Hz; each drone.step() runs 5 physics sub-steps.
    pyb_freq: int = 240
    ctrl_freq: int = 48
    camera_fov: float = 90.0
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_near: float = 0.05
    camera_far: float = 100.0
    camera_pitch_offset: float = 0.0


class GPDDrone:
    """
    Crazyflie CF2X quadrotor backed by gym-pybullet-drones physics.

    Control loop per step():
      1. Position PD → target horizontal thrust vector (clamped to MAX_TILT)
      2. Altitude PD → thrust magnitude
      3. Rotation-matrix decomposition → target Euler angles
      4. DSLPIDControl._dslPIDAttitudeControl() → motor RPMs
      5. CtrlAviary.step(RPMs) → physics update

    Usage:
        drone = GPDDrone(start_position=(0, 0, 1.5), start_yaw=0.0)
        # add gate bodies to drone.CLIENT
        drone.step(target_pos, target_vel, target_yaw)
        state = drone.get_state()
    """

    def __init__(
        self,
        start_position: Tuple[float, float, float] = (0.0, 0.0, 1.5),
        start_yaw: float = 0.0,
        config: Optional[GPDDroneConfig] = None,
        gui: bool = False,
    ):
        if not _GPD_AVAILABLE:
            raise ImportError(
                "gym-pybullet-drones is required.\n"
                "pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git"
            )

        self.config = config or GPDDroneConfig()
        self._start_pos = np.array(start_position, dtype=float)
        self._start_yaw = float(start_yaw)

        self._env = CtrlAviary(
            num_drones=1,
            initial_xyzs=np.array([start_position], dtype=float),
            initial_rpys=np.array([[0.0, 0.0, start_yaw]], dtype=float),
            physics=Physics.PYB,
            pyb_freq=self.config.pyb_freq,
            ctrl_freq=self.config.ctrl_freq,
            gui=gui,
            record=False,
        )
        # Inner attitude controller (tuned by DSL for the real Crazyflie CF2X).
        self._ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self._ctrl_dt = 1.0 / self.config.ctrl_freq

        # Maximum horizontal thrust at max tilt angle (Newtons).
        # GRAVITY here = mass * g = 0.265 N.
        self._max_horiz_thrust = self._ctrl.GRAVITY * math.tan(_MAX_TILT_RAD)

        self.step_count: int = 0
        self._last_view_matrix: Optional[np.ndarray] = None
        self._last_proj_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def CLIENT(self) -> int:
        """PyBullet physics client ID — add gate bodies here after construction."""
        return self._env.CLIENT

    def step(
        self,
        target_pos: Tuple[float, float, float],
        target_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_yaw: float = 0.0,
    ) -> None:
        """
        Compute tilt-safe motor RPMs and advance physics by one control timestep.

        Position + velocity errors → target thrust vector (horizontal component
        clamped to ≤35 degrees of tilt) → target Euler angles → RPMs via
        DSLPIDControl attitude controller → CtrlAviary physics step.
        """
        sv = self._env._getDroneStateVector(0)
        cur_pos = sv[0:3]
        cur_quat = sv[3:7]
        cur_vel = sv[10:13]

        pos_e = np.array(target_pos, dtype=float) - cur_pos
        vel_e = np.array(target_vel, dtype=float) - cur_vel

        # --- Target thrust vector (Newtons, world frame) ---
        # Same PD structure as DSLPIDControl._dslPIDPositionControl but with
        # horizontal component clamped to prevent extreme tilt angles.
        target_thrust = (
            self._ctrl.P_COEFF_FOR * pos_e
            + self._ctrl.D_COEFF_FOR * vel_e
            + np.array([0.0, 0.0, self._ctrl.GRAVITY])
        )

        # Clip horizontal thrust so tilt stays within _MAX_TILT_RAD.
        horiz_norm = np.linalg.norm(target_thrust[:2])
        if horiz_norm > self._max_horiz_thrust:
            target_thrust[:2] *= self._max_horiz_thrust / horiz_norm

        # Clip total vertical component to achievable range.
        target_thrust[2] = np.clip(
            target_thrust[2],
            0.5 * self._ctrl.GRAVITY,
            3.0 * self._ctrl.GRAVITY,
        )

        # --- Scalar thrust along body Z axis (depends on current tilt) ---
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        scalar_thrust = max(0.0, float(np.dot(target_thrust, cur_rotation[:, 2])))

        # Convert force (N) → PWM value (the unit _dslPIDAttitudeControl expects).
        thrust_pwm = (
            math.sqrt(max(scalar_thrust, 1e-9) / (4.0 * self._ctrl.KF))
            - self._ctrl.PWM2RPM_CONST
        ) / self._ctrl.PWM2RPM_SCALE

        # --- Target Euler angles from thrust vector + desired yaw ---
        thrust_norm = np.linalg.norm(target_thrust)
        target_z_ax = target_thrust / max(thrust_norm, 1e-9)
        target_x_c = np.array([math.cos(target_yaw), math.sin(target_yaw), 0.0])
        cross = np.cross(target_z_ax, target_x_c)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-6:
            target_y_ax = np.array([0.0, 1.0, 0.0])
        else:
            target_y_ax = cross / cross_norm
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).T
        target_euler = Rotation.from_matrix(target_rotation).as_euler("XYZ", degrees=False)

        # --- Attitude control → motor RPMs ---
        rpm = self._ctrl._dslPIDAttitudeControl(
            control_timestep=self._ctrl_dt,
            thrust=thrust_pwm,
            cur_quat=cur_quat,
            target_euler=target_euler,
            target_rpy_rates=np.zeros(3),
        )

        action = np.zeros((1, 4))
        action[0] = rpm
        self._env.step(action)
        self.step_count += 1

    def get_state(self) -> dict:
        """
        Return drone state dict compatible with the old QuadrotorDrone interface.

        Keys: position, velocity, orientation_quat, orientation_euler,
              angular_velocity, roll, pitch, yaw.
        """
        sv = self._env._getDroneStateVector(0)
        pos = tuple(float(v) for v in sv[0:3])
        quat = tuple(float(v) for v in sv[3:7])
        rpy = tuple(float(v) for v in sv[7:10])   # (roll, pitch, yaw)
        vel = tuple(float(v) for v in sv[10:13])
        ang_vel = tuple(float(v) for v in sv[13:16])
        return {
            "position": pos,
            "velocity": vel,
            "orientation_quat": quat,
            "orientation_euler": rpy,
            "angular_velocity": ang_vel,
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
        }

    def get_sim_time(self) -> float:
        """Elapsed simulation time in seconds."""
        return self.step_count * self._ctrl_dt

    def reset(self) -> None:
        """
        Reset drone to start position WITHOUT wiping the PyBullet world.

        CtrlAviary.reset() calls p.resetSimulation() which deletes all bodies
        including the gates. Instead, manually reset just the drone body.
        """
        quat = p.getQuaternionFromEuler([0.0, 0.0, self._start_yaw])
        p.resetBasePositionAndOrientation(
            self._env.DRONE_IDS[0],
            self._start_pos.tolist(),
            list(quat),
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self._env.DRONE_IDS[0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.CLIENT,
        )
        self._ctrl.reset()
        self.step_count = 0
        self._last_view_matrix = None
        self._last_proj_matrix = None

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Camera / rendering
    # ------------------------------------------------------------------

    def _drone_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos, orn = p.getBasePositionAndOrientation(
            self._env.DRONE_IDS[0], physicsClientId=self.CLIENT
        )
        return np.array(pos), np.array(orn)

    def _compute_fpv_matrices(self):
        pos, orn = self._drone_pose()
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward = rot[:, 0]
        up = rot[:, 2]

        if self.config.camera_pitch_offset != 0.0:
            angle = self.config.camera_pitch_offset
            ca, sa = math.cos(angle), math.sin(angle)
            new_forward = forward * ca + up * sa
            up = -forward * sa + up * ca
            forward = new_forward

        cam_pos = pos + up * 0.03
        target = cam_pos + forward * 1.0

        w, h = self.config.camera_resolution
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self.CLIENT,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=w / h,
            nearVal=self.config.camera_near,
            farVal=self.config.camera_far,
            physicsClientId=self.CLIENT,
        )
        self._last_view_matrix = np.array(view_matrix).reshape(4, 4).T
        self._last_proj_matrix = np.array(proj_matrix).reshape(4, 4).T
        return view_matrix, proj_matrix

    def get_camera_image(self) -> np.ndarray:
        view_matrix, proj_matrix = self._compute_fpv_matrices()
        w, h = self.config.camera_resolution
        _, _, rgba, _, _ = p.getCameraImage(
            width=w, height=h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.CLIENT,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return rgb[:, :, ::-1].copy()

    def project_points_to_fpv(self, world_points: np.ndarray) -> np.ndarray:
        if self._last_view_matrix is None:
            self._compute_fpv_matrices()
        V = self._last_view_matrix
        P = self._last_proj_matrix
        w, h = self.config.camera_resolution
        results = []
        for pt in world_points:
            p_hom = np.array([pt[0], pt[1], pt[2], 1.0])
            cam = V @ p_hom
            clip = P @ cam
            if abs(clip[3]) < 1e-9:
                results.append([-1.0, -1.0, -1.0])
                continue
            ndc = clip[:3] / clip[3]
            px = (ndc[0] + 1.0) * 0.5 * w
            py = (1.0 - ndc[1]) * 0.5 * h
            results.append([px, py, float(-cam[2])])
        return np.array(results)

    def get_spectator_image(
        self,
        distance: float = 5.0,
        yaw_offset: float = 0.0,
        pitch_offset: float = 20.0,
        resolution: Tuple[int, int] = (640, 480),
    ) -> np.ndarray:
        state = self.get_state()
        drone_pos = np.array(state["position"])
        drone_yaw = state["yaw"]

        yaw_rad = math.radians(yaw_offset) + drone_yaw + math.pi
        pitch_rad = math.radians(pitch_offset)
        cam_x = drone_pos[0] + distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        cam_y = drone_pos[1] + distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_z = drone_pos[2] + distance * math.sin(pitch_rad)

        cam_pos = [cam_x, cam_y, max(cam_z, 0.3)]
        w, h = resolution
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=drone_pos.tolist(),
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.CLIENT,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=70, aspect=w / h, nearVal=0.1, farVal=200.0,
            physicsClientId=self.CLIENT,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=w, height=h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.CLIENT,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return rgb[:, :, ::-1].copy()
