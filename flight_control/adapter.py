import math
from dataclasses import dataclass
from typing import Protocol, Tuple

from .types import DroneState, TargetState


class GateDetectionLike(Protocol):
    normalized_center_x: float
    normalized_center_y: float
    estimated_distance: float


@dataclass
class CameraModel:
    fov_horizontal_deg: float = 90.0
    fov_vertical_deg: float = 60.0
    min_distance: float = 0.2


def gate_detection_to_target(
    detection: GateDetectionLike,
    drone_state: DroneState,
    camera: CameraModel | None = None,
) -> TargetState:
    camera = camera or CameraModel()
    distance = max(detection.estimated_distance, camera.min_distance)

    angle_x = math.radians(camera.fov_horizontal_deg * 0.5) * detection.normalized_center_x
    angle_y = math.radians(camera.fov_vertical_deg * 0.5) * detection.normalized_center_y

    forward = distance
    right = math.tan(angle_x) * distance
    up = -math.tan(angle_y) * distance

    world_x, world_y = _rotate_xy(drone_state.yaw, forward, right)
    world_z = drone_state.position[2] + up

    target_position = (
        drone_state.position[0] + world_x,
        drone_state.position[1] + world_y,
        world_z,
    )

    return TargetState(position=target_position, yaw=drone_state.yaw)


def _rotate_xy(yaw: float, forward: float, right: float) -> Tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = cos_yaw * forward - sin_yaw * right
    world_y = sin_yaw * forward + cos_yaw * right
    return world_x, world_y
