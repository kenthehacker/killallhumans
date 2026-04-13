"""
PnP-based gate pose estimation and drift correction.

When gate corners are detected in the camera image, Perspective-n-Point (PnP)
gives us the camera-to-gate transform. Combined with known gate world position,
this provides an absolute position measurement to correct EKF drift.

Based on "On Your Own" (Romero et al., 2025):
  - YOLOv8n detects gate → corner regression → PnP → position measurement
  - Position corrections fed into EKF as measurement updates
  - RANSAC PnP for outlier rejection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GatePose:
    """6DOF pose of a gate relative to the camera."""
    position: Tuple[float, float, float]  # translation (x, y, z) in camera frame
    rotation: np.ndarray                   # 3x3 rotation matrix
    distance: float                        # Euclidean distance to gate center
    reprojection_error: float              # PnP quality metric


@dataclass
class CameraIntrinsics:
    """Camera calibration parameters."""
    fx: float = 462.0    # focal length x (pixels)
    fy: float = 462.0    # focal length y (pixels)
    cx: float = 320.0    # principal point x (pixels)
    cy: float = 240.0    # principal point y (pixels)
    fov_h_deg: float = 90.0
    image_width: int = 640
    image_height: int = 480

    @staticmethod
    def from_fov(
        fov_h_deg: float, width: int, height: int
    ) -> "CameraIntrinsics":
        """Compute intrinsics from horizontal FOV and image dimensions."""
        fx = width / (2 * math.tan(math.radians(fov_h_deg / 2)))
        fy = fx  # assume square pixels
        return CameraIntrinsics(
            fx=fx, fy=fy,
            cx=width / 2.0, cy=height / 2.0,
            fov_h_deg=fov_h_deg,
            image_width=width, image_height=height,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.zeros(5, dtype=np.float64)


@dataclass
class GateGeometry:
    """Physical gate dimensions for PnP."""
    interior_width_m: float = 1.2
    interior_height_m: float = 1.2

    @property
    def object_points(self) -> np.ndarray:
        """
        3D coordinates of gate corners in the gate's local frame.

        Order: top-left, top-right, bottom-right, bottom-left
        looking at the gate from the front.
        Gate center is at origin, normal along +Z in gate frame.
        """
        hw = self.interior_width_m / 2
        hh = self.interior_height_m / 2
        return np.array([
            [-hw, -hh, 0],  # top-left (image convention: -y is up)
            [hw, -hh, 0],   # top-right
            [hw, hh, 0],    # bottom-right
            [-hw, hh, 0],   # bottom-left
        ], dtype=np.float64)


class GatePnPEstimator:
    """
    Estimates gate 6DOF pose from detected corner pixels using PnP.

    Pipeline:
      1. Receive 4 corner pixel coordinates from gate detector
      2. Solve PnP with known gate geometry → rotation + translation
      3. Compute world position of the drone (inverse transform)
      4. Feed position to EKF as measurement update
    """

    def __init__(
        self,
        camera: CameraIntrinsics = None,
        gate: GateGeometry = None,
    ):
        self.camera = camera or CameraIntrinsics.from_fov(90.0, 640, 480)
        self.gate = gate or GateGeometry()
        self._last_pose: Optional[GatePose] = None

    def estimate_gate_pose(
        self,
        corner_pixels: np.ndarray,
        use_ransac: bool = True,
    ) -> Optional[GatePose]:
        """
        Solve PnP to get gate pose relative to camera.

        Args:
            corner_pixels: (4, 2) array of pixel coordinates
                           [top-left, top-right, bottom-right, bottom-left]
            use_ransac: use RANSAC for robustness (recommended)

        Returns:
            GatePose if PnP converges, None otherwise
        """
        if corner_pixels.shape != (4, 2):
            return None

        image_points = corner_pixels.astype(np.float64)
        object_points = self.gate.object_points
        camera_matrix = self.camera.matrix
        dist_coeffs = self.camera.dist_coeffs

        if use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=5.0,
                confidence=0.99,
            )
            if not success or inliers is None or len(inliers) < 3:
                return None
        else:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return None

        # Compute reprojection error
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec,
            camera_matrix, dist_coeffs,
        )
        reproj_error = float(np.mean(
            np.linalg.norm(projected.squeeze() - image_points, axis=1)
        ))

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        distance = float(np.linalg.norm(t))

        pose = GatePose(
            position=tuple(t),
            rotation=R,
            distance=distance,
            reprojection_error=reproj_error,
        )
        self._last_pose = pose
        return pose

    def gate_pose_to_drone_position(
        self,
        gate_pose: GatePose,
        gate_world_position: Tuple[float, float, float],
        gate_world_yaw: float,
        drone_orientation: Tuple[float, float, float],
        gate_world_pitch: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Compute drone world position from gate PnP result.

        The PnP gives us camera-to-gate transform. With known gate world
        position, we can infer the camera (drone) world position.

        Args:
            gate_pose: PnP result (camera frame)
            gate_world_position: known gate position in world frame (NED)
            gate_world_yaw: gate facing direction (radians)
            drone_orientation: (roll, pitch, yaw) of drone
            gate_world_pitch: gate pitch angle (radians, default 0)

        Returns:
            Estimated drone position in world frame (NED)
        """
        # Gate-to-camera: t_gc = gate_pose.position, R_gc = gate_pose.rotation
        # Camera-to-gate: t_cg = -R_gc^T @ t_gc
        R_gc = gate_pose.rotation
        t_gc = np.array(gate_pose.position)

        # Camera position in gate frame
        camera_in_gate = -R_gc.T @ t_gc

        # Gate frame to world frame: R = Rz(yaw) @ Ry(pitch)
        # Phase 4 fix: use full gate orientation, not yaw-only.
        cy, sy = math.cos(gate_world_yaw), math.sin(gate_world_yaw)
        cp, sp = math.cos(gate_world_pitch), math.sin(gate_world_pitch)

        R_yaw = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1],
        ])
        R_pitch = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp],
        ])
        R_gate_world = R_yaw @ R_pitch

        # Camera position in world frame
        gate_pos = np.array(gate_world_position)
        drone_world = gate_pos + R_gate_world @ camera_in_gate

        return tuple(drone_world)

    @property
    def last_reprojection_error(self) -> float:
        if self._last_pose is None:
            return float("inf")
        return self._last_pose.reprojection_error


def detect_gate_corners(
    contour: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Extract 4 ordered corners from a gate contour.

    Fits a quadrilateral to the contour and orders corners as:
    [top-left, top-right, bottom-right, bottom-left]

    Args:
        contour: OpenCV contour from detection
        image_shape: (height, width) for bounds checking

    Returns:
        (4, 2) array of corner pixels, or None if fitting fails
    """
    # Approximate contour to polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float64)
    else:
        # Use minimum area rectangle as fallback
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float64)

    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = _order_corners(corners)

    # Validate corners are within image bounds
    h, w = image_shape[:2]
    for cx, cy in corners:
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return None

    return corners


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    # Sort by y coordinate (top to bottom)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).squeeze()

    tl = pts[np.argmin(s)]    # smallest sum = top-left
    br = pts[np.argmax(s)]    # largest sum = bottom-right
    tr = pts[np.argmin(d)]    # smallest diff = top-right
    bl = pts[np.argmax(d)]    # largest diff = bottom-left

    return np.array([tl, tr, br, bl], dtype=np.float64)
