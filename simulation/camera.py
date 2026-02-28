from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .field import is_outside_bounds, visible_gate_prefilter
from .model_types import CameraFrame, CameraPose, Field, Gate, VisibleGateAnnotation

Vec3 = Tuple[float, float, float]
ProjPoint = Tuple[float, float, float]  # (pixel_x, pixel_y, camera_forward_depth)

_COLOR_MAP = {
    "red": (220, 60, 60),
    "blue": (70, 120, 230),
    "green": (60, 180, 80),
    "yellow": (220, 210, 70),
    "orange": (235, 150, 70),
    "purple": (170, 95, 220),
    "white": (220, 220, 220),
}


def get_camera_view(field: Field, pose: CameraPose, include_depth: bool = False) -> CameraFrame:
    width = pose.resolution_width
    height = pose.resolution_height
    rgb = [[(12, 12, 18) for _ in range(width)] for _ in range(height)]
    depth_buffer = [[float("inf") for _ in range(width)] for _ in range(height)]

    annotations: List[VisibleGateAnnotation] = []
    prefiltered = visible_gate_prefilter(field, pose.pose.position, radius_m=pose.far_plane_m)
    for gate in prefiltered:
        gate_polys = _project_gate_polygons(gate, pose, width, height)
        if not gate_polys:
            continue

        for polygon, shade in sorted(gate_polys, key=lambda item: _avg_depth(item[0]), reverse=True):
            color = _shade_color(_color_from_name(gate.config.color), shade)
            _draw_polygon(rgb, depth_buffer, polygon, color)

        annotation = _build_annotation(gate.gate_id, gate_polys, width, height)
        if annotation is not None:
            annotations.append(annotation)

    annotations.sort(key=lambda a: a.depth_m)
    depth = depth_buffer if include_depth else None
    return CameraFrame(
        rgb=rgb,
        pose=pose,
        visible_gates=annotations,
        outside_field=is_outside_bounds(field, pose.pose.position),
        depth=depth,
    )


def _project_gate_polygons(gate: Gate, camera: CameraPose, width: int, height: int) -> List[Tuple[List[ProjPoint], float]]:
    polygons: List[Tuple[List[ProjPoint], float]] = []
    for prism in _gate_frame_prisms_world(gate):
        corners = [_project_world_point(point, camera) for point in prism]
        if any(point is None for point in corners):
            continue
        projected = [point for point in corners if point is not None]
        for face_indices, shade in _prism_faces():
            poly = [projected[idx] for idx in face_indices]
            if _polygon_completely_offscreen(poly, width, height):
                continue
            polygons.append((poly, shade))
    return polygons


def _build_annotation(
    gate_id: str,
    polygons: Sequence[Tuple[List[ProjPoint], float]],
    width: int,
    height: int,
) -> Optional[VisibleGateAnnotation]:
    all_points = [point for poly, _ in polygons for point in poly]
    if not all_points:
        return None

    min_x = max(0, int(math.floor(min(point[0] for point in all_points))))
    min_y = max(0, int(math.floor(min(point[1] for point in all_points))))
    max_x = min(width - 1, int(math.ceil(max(point[0] for point in all_points))))
    max_y = min(height - 1, int(math.ceil(max(point[1] for point in all_points))))
    if min_x > max_x or min_y > max_y:
        return None

    center_x = int(round((min_x + max_x) * 0.5))
    center_y = int(round((min_y + max_y) * 0.5))
    min_depth = min(point[2] for point in all_points)
    return VisibleGateAnnotation(
        gate_id=gate_id,
        center_pixel=(center_x, center_y),
        approx_bbox=(min_x, min_y, max_x, max_y),
        depth_m=min_depth,
    )


def _gate_frame_prisms_world(gate: Gate) -> List[List[Vec3]]:
    interior_w = max(gate.config.interior_width_m, 0.05)
    interior_h = max(gate.config.interior_height_m, 0.05)
    border = max(gate.config.border_width_m, 0.01)
    depth = max(gate.config.depth_m, 0.01)
    outer_w = interior_w + 2.0 * border

    segments = [
        ((0.0, 0.0, interior_h * 0.5 + border * 0.5), (outer_w, depth, border)),
        ((0.0, 0.0, -interior_h * 0.5 - border * 0.5), (outer_w, depth, border)),
        ((-interior_w * 0.5 - border * 0.5, 0.0, 0.0), (border, depth, interior_h)),
        ((interior_w * 0.5 + border * 0.5, 0.0, 0.0), (border, depth, interior_h)),
    ]

    prisms: List[List[Vec3]] = []
    for center, size in segments:
        local_corners = _box_corners(center, size)
        world_corners = [_local_to_world(corner, gate.pose.x, gate.pose.y, gate.pose.z, gate.pose.yaw, gate.pose.pitch, gate.pose.roll) for corner in local_corners]
        prisms.append(world_corners)
    return prisms


def _box_corners(center: Vec3, size: Vec3) -> List[Vec3]:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    return [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]


def _prism_faces() -> Tuple[Tuple[Tuple[int, int, int, int], float], ...]:
    # Ordered from likely back/side to front-ish shading for visual depth cues.
    return (
        ((0, 1, 2, 3), 0.60),
        ((4, 5, 6, 7), 0.70),
        ((0, 3, 7, 4), 0.75),
        ((1, 2, 6, 5), 0.75),
        ((0, 1, 5, 4), 0.85),
        ((3, 2, 6, 7), 1.00),
    )


def _draw_polygon(
    rgb: List[List[Tuple[int, int, int]]],
    depth: List[List[float]],
    polygon: Sequence[ProjPoint],
    color: Tuple[int, int, int],
) -> None:
    if len(polygon) < 3:
        return

    p0 = polygon[0]
    for idx in range(1, len(polygon) - 1):
        _draw_triangle(rgb, depth, p0, polygon[idx], polygon[idx + 1], color)


def _draw_triangle(
    rgb: List[List[Tuple[int, int, int]]],
    depth_buffer: List[List[float]],
    p0: ProjPoint,
    p1: ProjPoint,
    p2: ProjPoint,
    color: Tuple[int, int, int],
) -> None:
    h = len(rgb)
    w = len(rgb[0]) if h else 0
    if h == 0 or w == 0:
        return

    min_x = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(w - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(h - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))))
    if min_x > max_x or min_y > max_y:
        return

    denom = ((p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1]))
    if abs(denom) < 1e-8:
        return

    for y in range(min_y, max_y + 1):
        py = y + 0.5
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
            w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
            w2 = 1.0 - w0 - w1
            if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                continue

            point_depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
            if point_depth >= depth_buffer[y][x]:
                continue
            depth_buffer[y][x] = point_depth
            rgb[y][x] = color


def _project_world_point(point_world: Vec3, camera: CameraPose) -> Optional[ProjPoint]:
    rel_world = (
        point_world[0] - camera.pose.x,
        point_world[1] - camera.pose.y,
        point_world[2] - camera.pose.z,
    )
    forward_x, right_y, up_z = _world_to_camera(rel_world, camera.pose.yaw, camera.pose.pitch, camera.pose.roll)

    if forward_x <= camera.near_plane_m or forward_x >= camera.far_plane_m:
        return None

    tan_h = math.tan(math.radians(camera.fov_horizontal_deg * 0.5))
    tan_v = math.tan(math.radians(camera.fov_vertical_deg * 0.5))
    if tan_h <= 0 or tan_v <= 0:
        return None

    norm_x = right_y / (forward_x * tan_h)
    norm_y = -up_z / (forward_x * tan_v)
    px = (norm_x + 1.0) * 0.5 * (camera.resolution_width - 1)
    py = (norm_y + 1.0) * 0.5 * (camera.resolution_height - 1)
    return (px, py, forward_x)


def _local_to_world(point: Vec3, tx: float, ty: float, tz: float, yaw: float, pitch: float, roll: float) -> Vec3:
    x, y, z = point

    cr, sr = math.cos(roll), math.sin(roll)
    x1 = x
    y1 = cr * y - sr * z
    z1 = sr * y + cr * z

    cp, sp = math.cos(pitch), math.sin(pitch)
    x2 = cp * x1 + sp * z1
    y2 = y1
    z2 = -sp * x1 + cp * z1

    cy, sy = math.cos(yaw), math.sin(yaw)
    x3 = cy * x2 - sy * y2
    y3 = sy * x2 + cy * y2
    z3 = z2

    return (x3 + tx, y3 + ty, z3 + tz)


def _world_to_camera(vector: Vec3, yaw: float, pitch: float, roll: float) -> Vec3:
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


def _polygon_completely_offscreen(points: Sequence[ProjPoint], width: int, height: int) -> bool:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(xs) < 0.0 or min(xs) > (width - 1) or max(ys) < 0.0 or min(ys) > (height - 1)


def _shade_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return (
        max(0, min(255, int(color[0] * factor))),
        max(0, min(255, int(color[1] * factor))),
        max(0, min(255, int(color[2] * factor))),
    )


def _color_from_name(name: str) -> Tuple[int, int, int]:
    return _COLOR_MAP.get(name.lower(), (200, 200, 200))


def _avg_depth(points: Sequence[ProjPoint]) -> float:
    return sum(point[2] for point in points) / max(len(points), 1)
