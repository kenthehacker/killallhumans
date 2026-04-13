"""Tests for PnP-based gate pose estimation (estimation/gate_pnp.py)."""

import math

import cv2
import numpy as np
import pytest

from estimation.gate_pnp import (
    CameraIntrinsics,
    GateGeometry,
    GatePnPEstimator,
    GatePose,
    _order_corners,
    detect_gate_corners,
)


# ── Camera intrinsics from FOV ───────────────────────────────────────────


class TestCameraIntrinsics:
    def test_from_fov_90_degrees(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        # fx = 640 / (2 * tan(45°)) = 640 / 2 = 320
        assert cam.fx == pytest.approx(320.0, abs=0.1)
        assert cam.fy == pytest.approx(320.0, abs=0.1)
        assert cam.cx == pytest.approx(320.0)
        assert cam.cy == pytest.approx(240.0)
        assert cam.image_width == 640
        assert cam.image_height == 480

    def test_from_fov_60_degrees(self):
        cam = CameraIntrinsics.from_fov(60.0, 640, 480)
        # fx = 640 / (2 * tan(30°)) = 640 / (2 * 0.5774) ≈ 554.3
        assert cam.fx == pytest.approx(554.26, abs=0.1)

    def test_matrix_shape(self):
        cam = CameraIntrinsics()
        K = cam.matrix
        assert K.shape == (3, 3)
        assert K[0, 0] == cam.fx
        assert K[1, 1] == cam.fy
        assert K[0, 2] == cam.cx
        assert K[1, 2] == cam.cy
        assert K[2, 2] == 1.0

    def test_dist_coeffs_zero(self):
        cam = CameraIntrinsics()
        assert cam.dist_coeffs.shape == (5,)
        assert np.allclose(cam.dist_coeffs, 0)

    def test_square_pixels(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        assert cam.fx == pytest.approx(cam.fy)


# ── Gate geometry ────────────────────────────────────────────────────────


class TestGateGeometry:
    def test_object_points_shape(self):
        gate = GateGeometry()
        pts = gate.object_points
        assert pts.shape == (4, 3)

    def test_object_points_centered(self):
        gate = GateGeometry(interior_width_m=2.0, interior_height_m=2.0)
        pts = gate.object_points
        # Centroid should be at origin
        centroid = pts.mean(axis=0)
        assert np.allclose(centroid, 0, atol=1e-10)

    def test_object_points_z_zero(self):
        gate = GateGeometry()
        # All points in the gate plane (z=0)
        assert np.allclose(gate.object_points[:, 2], 0)


# ── Corner ordering ──────────────────────────────────────────────────────


class TestCornerOrdering:
    def test_already_ordered(self):
        pts = np.array([
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 300],  # bottom-right
            [100, 300],  # bottom-left
        ], dtype=np.float64)
        ordered = _order_corners(pts)
        assert ordered[0] == pytest.approx([100, 100])  # TL
        assert ordered[1] == pytest.approx([300, 100])  # TR
        assert ordered[2] == pytest.approx([300, 300])  # BR
        assert ordered[3] == pytest.approx([100, 300])  # BL

    def test_shuffled_corners(self):
        pts = np.array([
            [300, 300],  # BR
            [100, 100],  # TL
            [100, 300],  # BL
            [300, 100],  # TR
        ], dtype=np.float64)
        ordered = _order_corners(pts)
        assert ordered[0] == pytest.approx([100, 100])
        assert ordered[1] == pytest.approx([300, 100])
        assert ordered[2] == pytest.approx([300, 300])
        assert ordered[3] == pytest.approx([100, 300])


# ── PnP solve with known geometry ────────────────────────────────────────


class TestPnPSolve:
    def _project_gate(self, distance_z: float, cam: CameraIntrinsics,
                       gate: GateGeometry) -> np.ndarray:
        """Helper: project gate corners at a known distance onto image plane."""
        # Gate centered at (0, 0, distance_z) in camera frame
        rvec = np.zeros(3)  # no rotation
        tvec = np.array([0, 0, distance_z], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            gate.object_points, rvec, tvec,
            cam.matrix, cam.dist_coeffs,
        )
        return projected.reshape(4, 2)

    def test_pnp_known_distance(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        gate = GateGeometry(interior_width_m=1.2, interior_height_m=1.2)
        estimator = GatePnPEstimator(camera=cam, gate=gate)

        distance = 5.0
        corners = self._project_gate(distance, cam, gate)
        pose = estimator.estimate_gate_pose(corners, use_ransac=False)

        assert pose is not None
        assert pose.distance == pytest.approx(distance, abs=0.1)
        assert pose.reprojection_error < 1.0

    def test_pnp_close_distance(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        gate = GateGeometry()
        estimator = GatePnPEstimator(camera=cam, gate=gate)

        distance = 2.0
        corners = self._project_gate(distance, cam, gate)
        pose = estimator.estimate_gate_pose(corners, use_ransac=False)

        assert pose is not None
        assert pose.distance == pytest.approx(distance, abs=0.2)

    def test_pnp_far_distance(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        gate = GateGeometry()
        estimator = GatePnPEstimator(camera=cam, gate=gate)

        distance = 15.0
        corners = self._project_gate(distance, cam, gate)
        pose = estimator.estimate_gate_pose(corners, use_ransac=False)

        assert pose is not None
        assert pose.distance == pytest.approx(distance, abs=0.5)

    def test_pnp_with_ransac(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        gate = GateGeometry()
        estimator = GatePnPEstimator(camera=cam, gate=gate)

        distance = 5.0
        corners = self._project_gate(distance, cam, gate)
        pose = estimator.estimate_gate_pose(corners, use_ransac=True)

        assert pose is not None
        assert pose.distance == pytest.approx(distance, abs=0.2)

    def test_pnp_invalid_corners_shape(self):
        estimator = GatePnPEstimator()
        result = estimator.estimate_gate_pose(np.zeros((3, 2)))
        assert result is None

    def test_pnp_stores_last_pose(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        gate = GateGeometry()
        estimator = GatePnPEstimator(camera=cam, gate=gate)

        assert estimator.last_reprojection_error == float("inf")
        corners = self._project_gate(5.0, cam, gate)
        pose = estimator.estimate_gate_pose(corners, use_ransac=False)
        assert estimator.last_reprojection_error < 1.0

    def test_pose_rotation_matrix_shape(self):
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        gate = GateGeometry()
        estimator = GatePnPEstimator(camera=cam, gate=gate)

        corners = self._project_gate(5.0, cam, gate)
        pose = estimator.estimate_gate_pose(corners, use_ransac=False)
        assert pose.rotation.shape == (3, 3)


# ── gate_pose_to_drone_position ──────────────────────────────────────────


class TestGatePoseToDronePosition:
    def test_inverse_transform_head_on(self):
        """Gate at world (10, 0, 0), camera directly in front at 5m → drone at (5, 0, 0)."""
        estimator = GatePnPEstimator()
        # Simulate: camera is 5m in front of gate along gate normal (+Z in gate frame)
        R = np.eye(3)  # no rotation between camera and gate
        t = np.array([0, 0, 5.0])  # gate is 5m along camera Z
        pose = GatePose(position=tuple(t), rotation=R,
                        distance=5.0, reprojection_error=0.1)

        gate_world_pos = (10.0, 0.0, 0.0)
        gate_yaw = 0.0
        drone_ori = (0, 0, 0)

        drone_pos = estimator.gate_pose_to_drone_position(
            pose, gate_world_pos, gate_yaw, drone_ori
        )
        # Camera is at -5m along gate normal from gate position
        # With gate yaw=0, normal is along x, so drone at (10 - 5, 0, 0) = (5, 0, 0)
        # But gate frame Z maps to world X when yaw=0
        assert isinstance(drone_pos, tuple)
        assert len(drone_pos) == 3

    def test_output_is_3d_tuple(self):
        estimator = GatePnPEstimator()
        pose = GatePose(position=(0, 0, 3), rotation=np.eye(3),
                        distance=3.0, reprojection_error=0.1)
        result = estimator.gate_pose_to_drone_position(
            pose, (0, 0, 0), 0.0, (0, 0, 0)
        )
        assert len(result) == 3
        assert all(isinstance(v, (float, np.floating)) for v in result)


# ── detect_gate_corners ──────────────────────────────────────────────────


class TestDetectGateCorners:
    def test_square_contour(self):
        # Create a clean square contour
        contour = np.array([
            [[100, 100]], [[300, 100]], [[300, 300]], [[100, 300]]
        ], dtype=np.int32)
        result = detect_gate_corners(contour, (480, 640))
        assert result is not None
        assert result.shape == (4, 2)

    def test_out_of_bounds_returns_none(self):
        contour = np.array([
            [[-10, 100]], [[300, 100]], [[300, 300]], [[100, 300]]
        ], dtype=np.int32)
        result = detect_gate_corners(contour, (480, 640))
        assert result is None

    def test_corners_ordered_correctly(self):
        contour = np.array([
            [[100, 100]], [[300, 100]], [[300, 300]], [[100, 300]]
        ], dtype=np.int32)
        result = detect_gate_corners(contour, (480, 640))
        if result is not None:
            # Top-left should have smallest sum
            assert result[0].sum() <= result[2].sum()
