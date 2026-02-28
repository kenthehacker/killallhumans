from __future__ import annotations

from typing import Protocol, Tuple

from flight_control.adapter import CameraModel, gate_detection_to_target
from flight_control.types import DroneState, TargetState


class GateDetectionLike(Protocol):
    normalized_center_x: float
    normalized_center_y: float
    estimated_distance: float


def gate_detection_to_target_state(
    detection: GateDetectionLike,
    drone_state: DroneState,
    camera: CameraModel | None = None,
) -> TargetState:
    return gate_detection_to_target(detection, drone_state, camera)


def waypoint_to_target_state(point: Tuple[float, float, float], yaw: float = 0.0) -> TargetState:
    return TargetState(position=point, yaw=yaw)
