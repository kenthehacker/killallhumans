"""
Kalman-filter gate tracker for temporal consistency across frames.

Based on:
  - "Drift-Corrected Monocular VIO for Drone Racing" (arXiv 2512.20475, 2025):
    detected gates serve as corrective measurements within an EKF to eliminate
    accumulated VIO drift, reducing mean translational error by 45%.
  - "On Your Own" (Romero 2025): PnP solutions from gate corners are fused
    with VIO in a Kalman filter using the known track map.

Tracks gate bounding boxes in image space:
  State: [cx, cy, w, h, dcx, dcy, dw, dh]
  cx,cy = center pixel; w,h = width,height; d* = velocities

When detection drops out, the tracker predicts the gate position for
up to max_coast_frames before dropping the track. This bridges the gap
between low-frequency detection and high-frequency control.

The tracker also predicts corner positions from the tracked bbox state,
enabling PnP pose estimation even during detection dropouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrackedGate:
    """A single tracked gate with temporal state for Kalman filtering."""
    gate_id: str
    state: np.ndarray         # [cx, cy, w, h, dcx, dcy, dw, dh]
    covariance: np.ndarray    # 8x8
    age: int = 0              # frames since track creation
    hits: int = 0             # total successful updates
    coast_count: int = 0      # consecutive frames without measurement
    confidence: float = 0.0
    last_corners: Optional[np.ndarray] = None  # (4,2) last measured corners

    @property
    def is_coasting(self) -> bool:
        """True if the track is predicting without recent measurements."""
        return self.coast_count > 0

    @property
    def predicted_bbox(self) -> Tuple[float, float, float, float]:
        """Current predicted bounding box as (cx, cy, w, h)."""
        return tuple(self.state[:4])

    @property
    def predicted_velocity(self) -> Tuple[float, float, float, float]:
        """Current predicted velocity as (dcx, dcy, dw, dh)."""
        return tuple(self.state[4:])

    def predicted_corners(self) -> np.ndarray:
        """
        Predict gate corners from tracked bbox state.

        Returns (4, 2) array ordered: top-left, top-right, bottom-right, bottom-left.
        If real corners were previously measured, they are warped according to
        the bbox state change; otherwise corners are derived from the bbox.
        """
        cx, cy, w, h = self.state[:4]
        half_w, half_h = w / 2, h / 2

        if self.last_corners is not None and self.coast_count > 0:
            # Warp last measured corners to match current predicted bbox
            old_cx, old_cy = np.mean(self.last_corners, axis=0)
            old_w = np.max(self.last_corners[:, 0]) - np.min(self.last_corners[:, 0])
            old_h = np.max(self.last_corners[:, 1]) - np.min(self.last_corners[:, 1])
            if old_w > 0 and old_h > 0:
                scale_x = w / old_w
                scale_y = h / old_h
                centered = self.last_corners - np.array([old_cx, old_cy])
                centered[:, 0] *= scale_x
                centered[:, 1] *= scale_y
                return centered + np.array([cx, cy])

        # Default: axis-aligned rectangle corners from bbox
        return np.array([
            [cx - half_w, cy - half_h],  # top-left
            [cx + half_w, cy - half_h],  # top-right
            [cx + half_w, cy + half_h],  # bottom-right
            [cx - half_w, cy + half_h],  # bottom-left
        ], dtype=np.float64)


@dataclass
class GateTrackerConfig:
    """Tracker tuning knobs."""
    process_noise_pos: float = 5.0      # pixels^2 per frame
    process_noise_vel: float = 10.0     # pixels^2/frame^2
    measurement_noise: float = 8.0      # pixels^2
    max_coast_frames: int = 15          # frames to predict without measurement
    min_hits_to_confirm: int = 3        # measurements before track is confirmed
    iou_threshold: float = 0.3          # for associating detections to tracks
    initial_velocity_std: float = 20.0  # pixels/frame
    confidence_decay: float = 0.9       # per-frame multiplier during coasting


# Constant-velocity state transition (8x8)
_F = np.eye(8)
_F[0, 4] = 1.0  # cx += dcx
_F[1, 5] = 1.0  # cy += dcy
_F[2, 6] = 1.0  # w  += dw
_F[3, 7] = 1.0  # h  += dh

# Measurement matrix: observe [cx, cy, w, h]
_H = np.zeros((4, 8))
_H[0, 0] = 1.0
_H[1, 1] = 1.0
_H[2, 2] = 1.0
_H[3, 3] = 1.0


class GateTracker:
    """
    Multi-gate Kalman filter tracker.

    Maintains a track for each visible gate. Associates new detections
    with existing tracks by IoU overlap, then runs predict/update.
    """

    def __init__(self, config: GateTrackerConfig = None):
        self.config = config or GateTrackerConfig()
        self._tracks: Dict[str, TrackedGate] = {}
        self._next_id = 0

    def predict(self) -> None:
        """
        Predict all tracks forward one frame using the constant-velocity model.

        Tracks that exceed max_coast_frames without a measurement update are
        removed. Confidence decays geometrically while coasting so downstream
        consumers can threshold on reliability.
        """
        c = self.config
        Q = np.zeros((8, 8))
        Q[0, 0] = Q[1, 1] = Q[2, 2] = Q[3, 3] = c.process_noise_pos
        Q[4, 4] = Q[5, 5] = Q[6, 6] = Q[7, 7] = c.process_noise_vel

        to_remove = []
        for gid, track in self._tracks.items():
            track.state = _F @ track.state
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                track.covariance = _F @ track.covariance @ _F.T + Q
            track.covariance = (track.covariance + track.covariance.T) / 2
            np.clip(track.covariance, -1e8, 1e8, out=track.covariance)
            if np.any(np.isnan(track.covariance)):
                to_remove.append(gid)
                continue
            track.coast_count += 1
            track.age += 1
            # Decay confidence while coasting
            track.confidence *= c.confidence_decay

            if track.coast_count > c.max_coast_frames:
                to_remove.append(gid)

        for gid in to_remove:
            del self._tracks[gid]

    def update(
        self,
        detections: List[Tuple[str, Tuple[float, float, float, float], float]],
        corners_list: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Update tracks with new detections.

        Args:
            detections: list of (gate_id, (cx, cy, w, h), confidence)
                        gate_id can be "" for unidentified detections.
            corners_list: optional parallel list of (4,2) corner arrays.
                          When provided, measured corners are stored on the
                          track for warped prediction during coasting and
                          for direct PnP pose estimation.
        """
        c = self.config
        R = np.eye(4) * c.measurement_noise

        matched_tracks: set = set()
        matched_dets: set = set()

        def _get_corners(idx: int) -> Optional[np.ndarray]:
            if corners_list is not None and idx < len(corners_list):
                return corners_list[idx]
            return None

        # Associate by gate_id first (if known)
        for i, (gid, bbox, conf) in enumerate(detections):
            if gid and gid in self._tracks:
                self._kalman_update(self._tracks[gid], bbox, R)
                self._tracks[gid].confidence = conf
                crn = _get_corners(i)
                if crn is not None:
                    self._tracks[gid].last_corners = crn.copy()
                matched_tracks.add(gid)
                matched_dets.add(i)

        # Associate remaining by IoU
        unmatched_tracks = [
            (gid, t) for gid, t in self._tracks.items() if gid not in matched_tracks
        ]
        unmatched_dets = [
            (i, d) for i, d in enumerate(detections) if i not in matched_dets
        ]

        if unmatched_tracks and unmatched_dets:
            for gid, track in unmatched_tracks:
                best_iou = 0.0
                best_idx = -1
                t_bbox = tuple(track.state[:4])
                for j, (i, (_, bbox, _)) in enumerate(unmatched_dets):
                    iou = _iou(t_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_iou >= c.iou_threshold and best_idx >= 0:
                    det_i, (_, bbox, conf) = unmatched_dets[best_idx]
                    self._kalman_update(track, bbox, R)
                    track.confidence = conf
                    crn = _get_corners(det_i)
                    if crn is not None:
                        track.last_corners = crn.copy()
                    matched_tracks.add(gid)
                    matched_dets.add(det_i)

        # Create new tracks for unmatched detections
        for i, (gid, bbox, conf) in enumerate(detections):
            if i not in matched_dets:
                track_id = gid if gid else f"auto_{self._next_id}"
                self._next_id += 1
                t = self._create_track(track_id, bbox, conf)
                crn = _get_corners(i)
                if crn is not None:
                    t.last_corners = crn.copy()

    def get_tracked_gates(self) -> List[TrackedGate]:
        """Get all confirmed tracks."""
        return [
            t for t in self._tracks.values()
            if t.hits >= self.config.min_hits_to_confirm
        ]

    def get_gate(self, gate_id: str) -> Optional[TrackedGate]:
        """Get a specific tracked gate."""
        t = self._tracks.get(gate_id)
        if t and t.hits >= self.config.min_hits_to_confirm:
            return t
        return None

    def get_predicted_bbox(self, gate_id: str) -> Optional[Tuple[float, float, float, float]]:
        """Get the predicted bounding box for a gate (even during coast)."""
        t = self._tracks.get(gate_id)
        if t is None:
            return None
        return tuple(t.state[:4])

    def _kalman_update(
        self, track: TrackedGate, bbox: Tuple[float, float, float, float], R: np.ndarray
    ) -> None:
        z = np.array(bbox)
        y = z - _H @ track.state
        S = _H @ track.covariance @ _H.T + R
        try:
            K = track.covariance @ _H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        track.state = track.state + K @ y
        I_KH = np.eye(8) - K @ _H
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            track.covariance = I_KH @ track.covariance @ I_KH.T + K @ R @ K.T
        if np.any(np.isnan(track.covariance)) or np.any(np.isinf(track.covariance)):
            # Reset to initial covariance on divergence
            track.covariance = np.eye(8)
            track.covariance[:4, :4] *= self.config.measurement_noise
            track.covariance[4:, 4:] *= self.config.initial_velocity_std ** 2
            return
        track.covariance = (track.covariance + track.covariance.T) / 2
        np.clip(track.covariance, -1e8, 1e8, out=track.covariance)
        track.coast_count = 0
        track.hits += 1

    def _create_track(
        self, gate_id: str, bbox: Tuple[float, float, float, float], confidence: float
    ) -> TrackedGate:
        state = np.zeros(8)
        state[:4] = bbox
        P = np.eye(8)
        P[0, 0] = P[1, 1] = P[2, 2] = P[3, 3] = self.config.measurement_noise
        P[4, 4] = P[5, 5] = P[6, 6] = P[7, 7] = self.config.initial_velocity_std ** 2
        track = TrackedGate(
            gate_id=gate_id, state=state, covariance=P,
            hits=1, confidence=confidence,
        )
        self._tracks[gate_id] = track
        return track

    def step(
        self,
        detections: List[Tuple[str, Tuple[float, float, float, float], float]],
        corners_list: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[TrackedGate]:
        """
        Convenience: predict + update in one call, return confirmed tracks.

        This is the main entry point for frame-by-frame tracking.

        Args:
            detections: list of (gate_id, (cx, cy, w, h), confidence).
            corners_list: optional parallel list of (4,2) corner arrays.

        Returns:
            List of confirmed TrackedGate objects (hits >= min_hits_to_confirm).
        """
        self.predict()
        self.update(detections, corners_list=corners_list)
        return self.get_tracked_gates()

    def get_corners_for_pnp(self, gate_id: str) -> Optional[np.ndarray]:
        """
        Get predicted corner pixels for PnP pose estimation.

        Returns (4, 2) float64 array ordered [TL, TR, BR, BL], or None if
        the track doesn't exist or isn't confirmed yet.

        During coasting, corners are warped from the last measurement
        according to the predicted bbox motion.
        """
        track = self._tracks.get(gate_id)
        if track is None or track.hits < self.config.min_hits_to_confirm:
            return None
        return track.predicted_corners()

    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks (including unconfirmed)."""
        return len(self._tracks)

    @property
    def confirmed_track_count(self) -> int:
        """Number of confirmed tracks (hits >= threshold)."""
        return sum(
            1 for t in self._tracks.values()
            if t.hits >= self.config.min_hits_to_confirm
        )

    def reset(self) -> None:
        """Remove all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 0


def _iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU between two (cx, cy, w, h) bounding boxes."""
    ax1 = a[0] - a[2] / 2
    ay1 = a[1] - a[3] / 2
    ax2 = a[0] + a[2] / 2
    ay2 = a[1] + a[3] / 2
    bx1 = b[0] - b[2] / 2
    by1 = b[1] - b[3] / 2
    bx2 = b[0] + b[2] / 2
    by2 = b[1] + b[3] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)
