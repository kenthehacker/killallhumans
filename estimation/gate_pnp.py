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

# Defaults track VADR-TS-002 §3.8 — single source of truth.
from competition.aigp_geometry import (
    AIGP_CAM_CX,
    AIGP_CAM_CY,
    AIGP_CAM_FX,
    AIGP_CAM_FY,
    AIGP_CAM_HEIGHT_PX,
    AIGP_CAM_PITCH_OFFSET_RAD,
    AIGP_CAM_VFOV_DEG,
    AIGP_CAM_WIDTH_PX,
    AIGP_GATE_INTERIOR_M,
)


@dataclass
class GatePose:
    """6DOF pose of a gate relative to the camera."""
    position: Tuple[float, float, float]  # translation (x, y, z) in camera frame
    rotation: np.ndarray                   # 3x3 rotation matrix
    distance: float                        # Euclidean distance to gate center
    reprojection_error: float              # PnP quality metric


@dataclass
class CameraIntrinsics:
    """Camera calibration parameters.

    Defaults match VADR-TS-002 §3.8 (AIGP VQ1): 640×360 pinhole,
    fx=fy=320, cx=320, cy=180, tilted 20° UPWARD from body frame.
    Legacy 640×480 calibrations supply explicit args (or use `from_fov`).
    """
    fx: float = AIGP_CAM_FX                  # 320 px
    fy: float = AIGP_CAM_FY                  # 320 px
    cx: float = AIGP_CAM_CX                  # 320 px
    cy: float = AIGP_CAM_CY                  # 180 px
    fov_h_deg: float = 90.0                  # Horizontal FoV; matches AIGP intrinsics
                                             # (2·atan(320/320) = 90°). The spec's
                                             # literal "VFoV=90°" is inconsistent with
                                             # fx=fy=320 — see aigp_geometry.py for the
                                             # full derivation. We trust the intrinsics.
    image_width: int = AIGP_CAM_WIDTH_PX     # 640
    image_height: int = AIGP_CAM_HEIGHT_PX   # 360
    # Camera tilt about body-Y axis. Positive = nose-up. AIGP camera is
    # tilted 20° upward, so a feature on the world horizon projects
    # BELOW cy by `fy·tan(pitch_offset_rad) ≈ 116 px`. PnP must apply
    # `R_pitch(pitch_offset_rad)` when converting camera-frame pose to
    # body-frame drone position.
    pitch_offset_rad: float = AIGP_CAM_PITCH_OFFSET_RAD

    @staticmethod
    def from_fov(
        fov_h_deg: float, width: int, height: int
    ) -> "CameraIntrinsics":
        """Compute intrinsics from horizontal FOV and image dimensions.

        Preserves the legacy 640×480 path: callers that pass explicit
        width/height get principal point at (w/2, h/2) — NOT the AIGP
        default of (320, 180) — because `from_fov` is the "build for a
        custom camera" entry point. Pitch offset stays at the AIGP
        default; legacy callers that need a non-tilted camera should
        construct directly and override `pitch_offset_rad=0.0`.
        """
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
    """Physical gate dimensions for PnP.

    Iter-002 (5/7 reviews MINOR): defaults updated to AIGP VQ1 spec
    (1.5 m inner opening per VADR-TS-002 §3.7). Legacy tracks supply
    explicit smaller dimensions via constructor args.
    """
    interior_width_m: float = AIGP_GATE_INTERIOR_M
    interior_height_m: float = AIGP_GATE_INTERIOR_M

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
        # Iter-002 review M3 (5/7 MAJOR): when camera is None, fall back to
        # AIGP defaults (640x360, fx=fy=320, +20deg tilt) instead of the
        # legacy 640x480 / fov-derived intrinsics. The default CameraIntrinsics
        # constructor already encodes the spec'd values; using from_fov here
        # would silently revert anyone who instantiated GatePnPEstimator
        # without an explicit camera argument back to legacy geometry.
        self.camera = camera or CameraIntrinsics()
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
        gate_world_roll: float = 0.0,
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
            gate_world_roll: gate roll angle (radians, default 0)

        Returns:
            Estimated drone position in world frame (NED)
        """
        # Gate-to-camera: t_gc = gate_pose.position, R_gc = gate_pose.rotation
        # Camera-to-gate: t_cg = -R_gc^T @ t_gc
        R_gc = gate_pose.rotation
        t_gc = np.array(gate_pose.position)

        # Camera position in gate frame
        camera_in_gate = -R_gc.T @ t_gc

        R_gate_world = _gate_frame_to_world(
            gate_world_yaw,
            gate_world_pitch,
            gate_world_roll,
        )

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


def _gate_frame_to_world(
    yaw: float, pitch: float = 0.0, roll: float = 0.0
) -> np.ndarray:
    """
    Rotation matrix whose columns map gate-local axes into NED world axes.

    GateGeometry defines local +Z as the gate normal, local +X as the
    horizontal right direction, and local +Y as the downward opening axis.
    That differs from a conventional yaw/pitch/roll body frame, so build the
    basis explicitly.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    normal = np.array([cy * cp, sy * cp, -sp], dtype=np.float64)
    right0 = np.array([-sy, cy, 0.0], dtype=np.float64)
    down0 = np.cross(normal, right0)
    down_norm = np.linalg.norm(down0)
    if down_norm > 1e-12:
        down0 = down0 / down_norm
    else:
        down0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    cr, sr = math.cos(roll), math.sin(roll)
    right = right0 * cr + down0 * sr
    down = -right0 * sr + down0 * cr

    return np.column_stack((right, down, normal))
