from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from .model_types import PathPolyline, PathSpec, Vec3


def build_path(spec: PathSpec) -> PathPolyline:
    points = _collapse_repeats([tuple(p) for p in spec.control_points])
    if len(points) < 2:
        raise ValueError("Path collapses to fewer than 2 unique control points")

    if len(points) == 2:
        sampled = _sample_line(points[0], points[1], spec.samples_per_segment)
    else:
        sampled = _sample_catmull_rom(points, spec.samples_per_segment, closed=spec.closed)

    cumulative = [0.0]
    for idx in range(1, len(sampled)):
        cumulative.append(cumulative[-1] + _distance(sampled[idx - 1], sampled[idx]))
    return PathPolyline(points=sampled, cumulative_lengths=cumulative)


def _sample_line(p0: Vec3, p1: Vec3, samples_per_segment: int) -> List[Vec3]:
    out: List[Vec3] = []
    for i in range(samples_per_segment + 1):
        t = i / samples_per_segment
        out.append(_lerp(p0, p1, t))
    return out


def _sample_catmull_rom(control_points: List[Vec3], samples_per_segment: int, closed: bool) -> List[Vec3]:
    pts = list(control_points)
    if closed:
        pts = [pts[-1]] + pts + [pts[0], pts[1]]
        segment_start = 1
        segment_end = len(pts) - 2
    else:
        pts = [pts[0]] + pts + [pts[-1]]
        segment_start = 1
        segment_end = len(pts) - 2

    sampled: List[Vec3] = []
    for idx in range(segment_start, segment_end):
        p0, p1, p2, p3 = pts[idx - 1], pts[idx], pts[idx + 1], pts[idx + 2]
        for s in range(samples_per_segment):
            t = s / samples_per_segment
            sampled.append(_catmull_rom(p0, p1, p2, p3, t))
    sampled.append(control_points[0] if closed else control_points[-1])
    return sampled


def _catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
        0.5 * ((2 * p1[2]) + (-p0[2] + p2[2]) * t + (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * t2 + (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * t3),
    )


def _lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _collapse_repeats(points: Iterable[Vec3]) -> List[Vec3]:
    out: List[Vec3] = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    return out
