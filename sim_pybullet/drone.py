"""
Quadrotor drone model for PyBullet simulation.

Creates a simple box-body drone with 4 rotor force application points.
Accepts (throttle, roll, pitch, yaw) commands and applies corresponding
forces/torques through PyBullet's rigid-body physics.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None


@dataclass
class DroneConfig:
    mass_kg: float = 1.0
    arm_length_m: float = 0.175
    body_size: Tuple[float, float, float] = (0.15, 0.15, 0.05)
    max_thrust_n: float = 20.0  # total max thrust (all 4 motors)
    max_roll_angle: float = 0.35  # radians (~20 deg) — keep conservative for stability
    max_pitch_angle: float = 0.35
    max_yaw_rate: float = 2.0  # rad/s
    gravity: float = 9.81
    # Attitude PD gains (tuned for per-motor differential thrust)
    attitude_kp: float = 12.0
    attitude_kd: float = 4.0
    yaw_kp: float = 6.0
    yaw_kd: float = 2.0
    # Yaw torque from propeller reaction (Nm per N of thrust)
    yaw_torque_coeff: float = 0.015
    yaw_torque_max: float = 0.5
    # Camera
    camera_fov: float = 90.0
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_near: float = 0.05
    camera_far: float = 100.0
    camera_pitch_offset: float = 0.0  # slight downward tilt if needed


class QuadrotorDrone:
    """
    A simplified quadrotor in PyBullet.

    Control inputs are normalized:
      - throttle: 0.0 (no thrust) to 1.0 (max thrust)
      - roll:    -1.0 (left) to 1.0 (right)
      - pitch:   -1.0 (nose down) to 1.0 (nose up)
      - yaw:     -1.0 (CCW) to 1.0 (CW)
    """

    def __init__(
        self,
        physics_client: int,
        config: Optional[DroneConfig] = None,
        start_position: Tuple[float, float, float] = (0.0, 0.0, 1.5),
        start_yaw: float = 0.0,
    ):
        self.client = physics_client
        self.config = config or DroneConfig()
        self._start_pos = start_position
        self._start_yaw = start_yaw
        self.body_id = self._create_body(start_position, start_yaw)

    def _create_body(
        self, position: Tuple[float, float, float], yaw: float
    ) -> int:
        """Build an X-shaped quadcopter: central hub + 4 arms + 4 motor discs."""
        arm = self.config.arm_length_m
        hub_r = 0.03
        hub_h = 0.02
        arm_r = 0.008
        motor_r = 0.022
        motor_h = 0.012

        # Central hub (collision body for physics)
        col_hub = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=hub_r, height=hub_h * 2,
            physicsClientId=self.client,
        )
        vis_hub = p.createVisualShape(
            p.GEOM_CYLINDER, radius=hub_r, length=hub_h * 2,
            rgbaColor=[0.15, 0.15, 0.15, 1.0],
            physicsClientId=self.client,
        )

        # 4 arm directions at 45, 135, 225, 315 degrees (X-pattern)
        arm_angles = [math.pi / 4, 3 * math.pi / 4,
                      5 * math.pi / 4, 7 * math.pi / 4]
        motor_colors = [
            [0.9, 0.2, 0.2, 1.0],  # front-right: red
            [0.9, 0.2, 0.2, 1.0],  # front-left: red
            [0.2, 0.2, 0.2, 1.0],  # rear-left: dark
            [0.2, 0.2, 0.2, 1.0],  # rear-right: dark
        ]

        link_cols = []
        link_viss = []
        link_positions = []
        link_orientations = []
        link_masses = []
        link_inertials = []
        link_inertial_pos = []
        link_inertial_orn = []
        link_parents = []
        link_joint_types = []
        link_joint_axes = []

        for i, angle in enumerate(arm_angles):
            dx = math.cos(angle) * arm * 0.5
            dy = math.sin(angle) * arm * 0.5

            # Arm segment (capsule-like box along the arm direction)
            arm_col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[arm * 0.5, arm_r, arm_r],
                physicsClientId=self.client,
            )
            arm_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[arm * 0.5, arm_r, arm_r],
                rgbaColor=[0.3, 0.3, 0.3, 1.0],
                physicsClientId=self.client,
            )
            arm_orn = p.getQuaternionFromEuler([0, 0, angle])

            link_cols.append(arm_col)
            link_viss.append(arm_vis)
            link_positions.append([dx, dy, 0])
            link_orientations.append(arm_orn)
            link_masses.append(0.0)
            link_inertials.append([0, 0, 0])
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parents.append(0)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append([0, 0, 0])

            # Motor disc at arm tip
            tip_dx = math.cos(angle) * arm
            tip_dy = math.sin(angle) * arm
            motor_col = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=motor_r, height=motor_h * 2,
                physicsClientId=self.client,
            )
            motor_vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=motor_r, length=motor_h * 2,
                rgbaColor=motor_colors[i],
                physicsClientId=self.client,
            )
            link_cols.append(motor_col)
            link_viss.append(motor_vis)
            link_positions.append([tip_dx, tip_dy, hub_h + motor_h])
            link_orientations.append([0, 0, 0, 1])
            link_masses.append(0.0)
            link_inertials.append([0, 0, 0])
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parents.append(0)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append([0, 0, 0])

        quat = p.getQuaternionFromEuler([0, 0, yaw])
        body_id = p.createMultiBody(
            baseMass=self.config.mass_kg,
            baseCollisionShapeIndex=col_hub,
            baseVisualShapeIndex=vis_hub,
            basePosition=list(position),
            baseOrientation=quat,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_cols,
            linkVisualShapeIndices=link_viss,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_pos,
            linkInertialFrameOrientations=link_inertial_orn,
            linkParentIndices=link_parents,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            body_id, -1,
            linearDamping=0.3,
            angularDamping=0.8,
            physicsClientId=self.client,
        )
        return body_id

    def reset(self):
        quat = p.getQuaternionFromEuler([0, 0, self._start_yaw])
        p.resetBasePositionAndOrientation(
            self.body_id, list(self._start_pos), quat,
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.body_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client,
        )

    def get_state(self) -> dict:
        """Return drone state as a dict with position, velocity, orientation."""
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.body_id, physicsClientId=self.client
        )
        euler = p.getEulerFromQuaternion(orn)
        return {
            "position": tuple(pos),
            "velocity": tuple(vel),
            "orientation_quat": tuple(orn),
            "orientation_euler": tuple(euler),  # (roll, pitch, yaw)
            "angular_velocity": tuple(ang_vel),
            "yaw": euler[2],
            "pitch": euler[1],
            "roll": euler[0],
        }

    def apply_command(self, throttle: float, roll: float, pitch: float, yaw: float):
        """
        Apply normalized control inputs via per-motor thrust.

        Differential thrust at X-config positions creates roll/pitch torques.
        Yaw is controlled via explicit torque (propeller reaction model).

        Motor layout (body frame, X-config, ENU: +x=fwd, +y=left):
          0: front-left  (+x, +y)  CW     angle=π/4
          1: rear-left   (-x, +y)  CCW    angle=3π/4
          2: rear-right  (-x, -y)  CW     angle=5π/4
          3: front-right (+x, -y)  CCW    angle=7π/4
        """
        throttle = max(0.0, min(1.0, throttle))
        roll = max(-1.0, min(1.0, roll))
        pitch = max(-1.0, min(1.0, pitch))
        yaw = max(-1.0, min(1.0, yaw))

        state = self.get_state()
        cur_roll, cur_pitch, _ = state["orientation_euler"]
        world_ang_vel = np.array(state["angular_velocity"])

        # Convert world-frame angular velocity to body-frame rates.
        # Critical: without this, roll/pitch damping uses wrong axis
        # when the drone has any yaw rotation.
        orn = state["orientation_quat"]
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        body_ang_vel = rot.T @ world_ang_vel

        desired_roll = roll * self.config.max_roll_angle
        desired_pitch = pitch * self.config.max_pitch_angle
        desired_yaw_rate = yaw * self.config.max_yaw_rate

        roll_error = desired_roll - cur_roll
        pitch_error = desired_pitch - cur_pitch

        max_differential = 0.35
        roll_signal = (
            self.config.attitude_kp * roll_error
            - self.config.attitude_kd * body_ang_vel[0]
        ) / self.config.max_thrust_n
        roll_signal = max(-max_differential, min(max_differential, roll_signal))

        pitch_signal = (
            self.config.attitude_kp * pitch_error
            - self.config.attitude_kd * body_ang_vel[1]
        ) / self.config.max_thrust_n
        pitch_signal = max(-max_differential, min(max_differential, pitch_signal))

        # Yaw torque computed separately — propeller reaction torques,
        # since thrust forces at X-config motor positions produce zero z-torque.
        yaw_rate_error = desired_yaw_rate - body_ang_vel[2]
        yaw_torque = (
            self.config.yaw_kp * yaw_rate_error
            - self.config.yaw_kd * body_ang_vel[2]
        )
        yaw_torque = max(-self.config.yaw_torque_max,
                         min(self.config.yaw_torque_max, yaw_torque))

        # X-config mixing matrix (motors at 45° angles in body frame):
        #   Motor 0: angle=π/4  → pos (+x,+y) = front-left,  CW
        #   Motor 1: angle=3π/4 → pos (-x,+y) = rear-left,   CCW
        #   Motor 2: angle=5π/4 → pos (-x,-y) = rear-right,  CW
        #   Motor 3: angle=7π/4 → pos (+x,-y) = front-right, CCW
        #
        # Roll torque (τ_x) = 0.707*arm*(F0+F1-F2-F3) → controlled by roll_signal
        # Pitch torque (τ_y) = 0.707*arm*(-F0+F1+F2-F3) → controlled by pitch_signal
        base = throttle
        motor_thrusts = [
            base + roll_signal - pitch_signal,   # Motor 0 (front-left, CW)
            base + roll_signal + pitch_signal,   # Motor 1 (rear-left, CCW)
            base - roll_signal + pitch_signal,   # Motor 2 (rear-right, CW)
            base - roll_signal - pitch_signal,   # Motor 3 (front-right, CCW)
        ]

        arm = self.config.arm_length_m
        motor_angles = [
            math.pi / 4,       # front-left
            3 * math.pi / 4,   # rear-left
            5 * math.pi / 4,   # rear-right
            7 * math.pi / 4,   # front-right
        ]

        rot_matrix = rot
        body_z = rot_matrix[:, 2]

        pos = np.array(state["position"])

        for i, (mt, angle) in enumerate(zip(motor_thrusts, motor_angles)):
            mt = max(0.0, min(1.0, mt))
            thrust_n = mt * (self.config.max_thrust_n / 4.0)
            force_world = body_z * thrust_n

            local_motor = np.array([
                math.cos(angle) * arm,
                math.sin(angle) * arm,
                0.0,
            ])
            world_motor = pos + rot_matrix @ local_motor

            p.applyExternalForce(
                self.body_id, -1,
                forceObj=force_world.tolist(),
                posObj=world_motor.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client,
            )

        # Apply yaw torque around body z-axis (models propeller reaction torques)
        torque_world = body_z * yaw_torque
        p.applyExternalTorque(
            self.body_id, -1,
            torqueObj=torque_world.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

    def _compute_fpv_matrices(self):
        """Compute and cache the FPV camera view/projection matrices."""
        state = self.get_state()
        pos = state["position"]
        orn = state["orientation_quat"]

        rot_matrix = np.array(
            p.getMatrixFromQuaternion(orn)
        ).reshape(3, 3)

        forward = rot_matrix[:, 0]
        up = rot_matrix[:, 2]

        if self.config.camera_pitch_offset != 0:
            angle = self.config.camera_pitch_offset
            ca, sa = math.cos(angle), math.sin(angle)
            new_forward = forward * ca + up * sa
            up = -forward * sa + up * ca   # use original forward before overwriting
            forward = new_forward

        cam_pos = np.array(pos) + up * 0.03
        target = cam_pos + forward * 1.0

        w, h = self.config.camera_resolution
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self.client,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=w / h,
            nearVal=self.config.camera_near,
            farVal=self.config.camera_far,
            physicsClientId=self.client,
        )
        self._last_view_matrix = np.array(view_matrix).reshape(4, 4).T
        self._last_proj_matrix = np.array(proj_matrix).reshape(4, 4).T
        return view_matrix, proj_matrix

    def get_camera_image(self) -> np.ndarray:
        """Render a forward-facing camera image from the drone's perspective."""
        view_matrix, proj_matrix = self._compute_fpv_matrices()

        w, h = self.config.camera_resolution
        _, _, rgba, _, _ = p.getCameraImage(
            width=w,
            height=h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return rgb[:, :, ::-1].copy()

    def project_points_to_fpv(self, world_points: np.ndarray) -> np.ndarray:
        """
        Project Nx3 world-space points into FPV pixel coordinates.
        Returns Nx3 array: (pixel_x, pixel_y, depth). Points behind camera
        will have depth <= 0.
        """
        if not hasattr(self, '_last_view_matrix'):
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
                results.append([-1, -1, -1])
                continue
            ndc = clip[:3] / clip[3]
            px = (ndc[0] + 1.0) * 0.5 * w
            py = (1.0 - ndc[1]) * 0.5 * h
            results.append([px, py, -cam[2]])
        return np.array(results)

    def get_spectator_image(
        self,
        distance: float = 5.0,
        yaw_offset: float = 0.0,
        pitch_offset: float = 20.0,
        resolution: Tuple[int, int] = (640, 480),
    ) -> np.ndarray:
        """
        Render an orbitable 3rd-person camera.

        Args:
            distance: radius from drone center
            yaw_offset: horizontal orbit angle in degrees (0 = behind drone)
            pitch_offset: vertical angle in degrees above horizontal
        """
        state = self.get_state()
        pos = np.array(state["position"])
        drone_yaw = state["yaw"]

        # Compute camera position on a sphere around the drone
        yaw_rad = math.radians(yaw_offset) + drone_yaw + math.pi
        pitch_rad = math.radians(pitch_offset)
        cam_x = pos[0] + distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        cam_y = pos[1] + distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_z = pos[2] + distance * math.sin(pitch_rad)

        cam_pos = [cam_x, cam_y, max(cam_z, 0.3)]
        target = pos.tolist()

        w, h = resolution
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.client,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=70, aspect=w / h, nearVal=0.1, farVal=200.0,
            physicsClientId=self.client,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=w, height=h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return rgb[:, :, ::-1].copy()
