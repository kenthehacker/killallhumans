from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .field import is_outside_bounds, visible_gate_prefilter
from .model_types import CameraFrame, CameraPose, Field, Gate, VisibleGateAnnotation


def get_camera_view(field: Field, pose: CameraPose, include_depth: bool = False) -> CameraFrame:
    width = pose.resolution_width
    height = pose.resolution_height
    rgb = [[(20, 20, 20) for _ in range(width)] for _ in range(height)]
    depth = [[float("inf") for _ in range(width)] for _ in range(height)] if include_depth else None

    annotations: List[VisibleGateAnnotation] = []
    prefiltered = visible_gate_prefilter(field, pose.pose.position, radius_m=pose.far_plane_m)
    for gate in prefiltered:
        projected = _project_gate_center(gate, pose)
        if projected is None:
            continue

        px, py, dist, camera_forward = projected
        box_half = max(2, int(140.0 * gate.config.width_m / max(camera_forward, 0.1)))
        bbox = (max(0, px - box_half), max(0, py - box_half), min(width - 1, px + box_half), min(height - 1, py + box_half))
        _draw_cross(rgb, px, py, radius=min(10, box_half), color=(220, 40, 40))
        _draw_rect(rgb, bbox, color=(40, 220, 40))

        if depth is not None:
            for y in range(bbox[1], bbox[3] + 1):
                row = depth[y]
                for x in range(bbox[0], bbox[2] + 1):
                    if dist < row[x]:
                        row[x] = dist

        annotations.append(
            VisibleGateAnnotation(
                gate_id=gate.gate_id,
                center_pixel=(px, py),
                approx_bbox=bbox,
                depth_m=dist,
            )
        )

    annotations.sort(key=lambda a: a.depth_m)
    return CameraFrame(
        rgb=rgb,
        pose=pose,
        visible_gates=annotations,
        outside_field=is_outside_bounds(field, pose.pose.position),
        depth=depth,
    )


def _project_gate_center(gate: Gate, camera: CameraPose) -> Optional[Tuple[int, int, float, float]]:
    rel_world = (
        gate.pose.x - camera.pose.x,
        gate.pose.y - camera.pose.y,
        gate.pose.z - camera.pose.z,
    )
    camera_space = _world_to_camera(rel_world, camera.pose.yaw, camera.pose.pitch, camera.pose.roll)
    forward_x, right_y, up_z = camera_space

    if forward_x <= camera.near_plane_m or forward_x >= camera.far_plane_m:
        return None

    tan_h = math.tan(math.radians(camera.fov_horizontal_deg * 0.5))
    tan_v = math.tan(math.radians(camera.fov_vertical_deg * 0.5))
    if tan_h <= 0 or tan_v <= 0:
        return None

    norm_x = right_y / (forward_x * tan_h)
    norm_y = -up_z / (forward_x * tan_v)

    if abs(norm_x) > 1.0 or abs(norm_y) > 1.0:
        return None

    px = int(round((norm_x + 1.0) * 0.5 * (camera.resolution_width - 1)))
    py = int(round((norm_y + 1.0) * 0.5 * (camera.resolution_height - 1)))

    distance = math.sqrt(rel_world[0] ** 2 + rel_world[1] ** 2 + rel_world[2] ** 2)
    return (px, py, distance, forward_x)


def _world_to_camera(vector: Tuple[float, float, float], yaw: float, pitch: float, roll: float) -> Tuple[float, float, float]:
    # Inverse rotation: world -> camera for yaw(z), pitch(y), roll(x)
    x, y, z = vector

    cy = math.cos(-yaw)
    sy = math.sin(-yaw)
    x1 = cy * x - sy * y
    y1 = sy * x + cy * y
    z1 = z

    cp = math.cos(-pitch)
    sp = math.sin(-pitch)
    x2 = cp * x1 + sp * z1
    y2 = y1
    z2 = -sp * x1 + cp * z1

    cr = math.cos(-roll)
    sr = math.sin(-roll)
    x3 = x2
    y3 = cr * y2 - sr * z2
    z3 = sr * y2 + cr * z2

    return (x3, y3, z3)


def _draw_cross(rgb: List[List[Tuple[int, int, int]]], cx: int, cy: int, radius: int, color: Tuple[int, int, int]) -> None:
    h = len(rgb)
    w = len(rgb[0]) if h else 0
    for dx in range(-radius, radius + 1):
        x = cx + dx
        if 0 <= x < w and 0 <= cy < h:
            rgb[cy][x] = color
    for dy in range(-radius, radius + 1):
        y = cy + dy
        if 0 <= cx < w and 0 <= y < h:
            rgb[y][cx] = color


def _draw_rect(rgb: List[List[Tuple[int, int, int]]], bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = bbox
    h = len(rgb)
    w = len(rgb[0]) if h else 0
    if h == 0 or w == 0:
        return

    for x in range(max(0, x0), min(w - 1, x1) + 1):
        if 0 <= y0 < h:
            rgb[y0][x] = color
        if 0 <= y1 < h:
            rgb[y1][x] = color
    for y in range(max(0, y0), min(h - 1, y1) + 1):
        if 0 <= x0 < w:
            rgb[y][x0] = color
        if 0 <= x1 < w:
            rgb[y][x1] = color
