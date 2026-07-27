"""Deterministic image-space inner-aperture geometry for VQ2 gates.

The build-3385 detector's bounding box is contour support, not a gate edge.
This module instead extracts the boundary of the dark opening from the known
VQ2 colour mask.  Four visible sides are fit directly.  When exactly one side
is absent because that same image border clips the detection, the missing side
is censored and inferred with an explicitly named square-in-pixel-space prior.

This is not calibrated pose, distance, or physical square evidence.  Ambiguous
multiple openings and geometry with fewer than three supported sides are
rejected rather than assigned precise fitted values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
from typing import Optional

import cv2
import numpy as np

from gate_detection.src.vq2_detector import VQ2_GATE_HSV_RANGES


class ApertureSide(IntFlag):
    """Image-border and aperture-side mask in left/top/right/bottom order."""

    NONE = 0
    LEFT = 1
    TOP = 2
    RIGHT = 4
    BOTTOM = 8


_SIDES = (
    ApertureSide.LEFT,
    ApertureSide.TOP,
    ApertureSide.RIGHT,
    ApertureSide.BOTTOM,
)


@dataclass(frozen=True, slots=True)
class VQ2ApertureConfig:
    """Conservative thresholds for deterministic scanline fitting."""

    boundary_margin_px: int = 2
    morph_kernel_size: int = 3
    min_component_pixels: int = 24
    min_gap_px: int = 4
    min_line_samples: int = 4
    secondary_component_ratio: float = 0.45
    competing_gap_ratio: float = 0.65
    square_prior_relative_sigma: float = 0.12

    def __post_init__(self) -> None:
        for name in (
            "boundary_margin_px",
            "morph_kernel_size",
            "min_component_pixels",
            "min_gap_px",
            "min_line_samples",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if self.boundary_margin_px < 0:
            raise ValueError("boundary_margin_px must be non-negative")
        if self.morph_kernel_size < 1 or self.morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be a positive odd integer")
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be positive")
        if self.min_gap_px < 2:
            raise ValueError("min_gap_px must be at least two pixels")
        if self.min_line_samples < 2:
            raise ValueError("min_line_samples must be at least two")
        for name in (
            "secondary_component_ratio",
            "competing_gap_ratio",
            "square_prior_relative_sigma",
        ):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise TypeError(f"{name} must be finite numeric data")
        if not 0.0 < self.secondary_component_ratio < 1.0:
            raise ValueError("secondary_component_ratio must be in (0, 1)")
        if not 0.0 < self.competing_gap_ratio < 1.0:
            raise ValueError("competing_gap_ratio must be in (0, 1)")
        if not 0.0 < self.square_prior_relative_sigma <= 1.0:
            raise ValueError("square_prior_relative_sigma must be in (0, 1]")


Point = tuple[float, float]
Segment = tuple[Point, Point]
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class VQ2ApertureFit:
    """Result of an image-space aperture fit, including rejected diagnostics."""

    image_size_px: tuple[int, int]
    support_bbox_px: tuple[int, int, int, int]
    clipping: ApertureSide
    fitted_corners_px: Optional[Quad]
    visible_edges: ApertureSide
    visible_corners: tuple[bool, bool, bool, bool]
    visible_segments_px: tuple[
        Optional[Segment], Optional[Segment], Optional[Segment], Optional[Segment]
    ]
    geometry_model_id: Optional[str]
    covariance_model_id: Optional[str]
    covariance_diagonal: Optional[tuple[float, float, float, float, float]]
    residual_rms_px: Optional[float]
    inlier_count: int
    support_count: int
    confidence: float
    rejection_reason: Optional[str]

    @property
    def succeeded(self) -> bool:
        return self.fitted_corners_px is not None


@dataclass(frozen=True, slots=True)
class VQ2PassageGeometry:
    """Conservative normalized inner geometry permitted to claim passage."""

    center_norm: Point
    aperture_half_size_norm: Point
    log_scale: float
    measurement_std: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _LineFit:
    coefficients: tuple[float, float, float]
    inlier_points: tuple[Point, ...]
    residual_rms_px: float
    support_count: int


def _exact_bbox(
    bbox: object,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if type(bbox) is not tuple or len(bbox) != 4:
        raise TypeError("support_bbox_px must be an exact four-tuple")
    values: list[int] = []
    for index, value in enumerate(bbox):
        if type(value) is not int:
            raise TypeError(f"support_bbox_px[{index}] must be an exact integer")
        values.append(value)
    x, y, width, height = values
    if (
        x < 0
        or y < 0
        or width < 1
        or height < 1
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError("support_bbox_px must have positive area inside the image")
    return x, y, width, height


def _finite_confidence(value: object) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise TypeError("detection_confidence must be finite numeric data")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("detection_confidence must be in [0, 1]")
    return result


def _clipping_mask(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: int,
) -> ApertureSide:
    x, y, width, height = bbox
    result = ApertureSide.NONE
    if x <= margin:
        result |= ApertureSide.LEFT
    if y <= margin:
        result |= ApertureSide.TOP
    if x + width >= image_width - margin:
        result |= ApertureSide.RIGHT
    if y + height >= image_height - margin:
        result |= ApertureSide.BOTTOM
    return result


def _rejected(
    *,
    image_size_px: tuple[int, int],
    bbox: tuple[int, int, int, int],
    clipping: ApertureSide,
    reason: str,
    support_count: int = 0,
) -> VQ2ApertureFit:
    return VQ2ApertureFit(
        image_size_px=image_size_px,
        support_bbox_px=bbox,
        clipping=clipping,
        fitted_corners_px=None,
        visible_edges=ApertureSide.NONE,
        visible_corners=(False, False, False, False),
        visible_segments_px=(None, None, None, None),
        geometry_model_id=None,
        covariance_model_id=None,
        covariance_diagonal=None,
        residual_rms_px=None,
        inlier_count=0,
        support_count=support_count,
        confidence=0.0,
        rejection_reason=reason,
    )


def vq2_gate_mask_from_bgr(
    image_bgr: np.ndarray,
    *,
    config: VQ2ApertureConfig = VQ2ApertureConfig(),
) -> np.ndarray:
    """Return the deterministic build-3385 red/pink gate mask."""

    if type(config) is not VQ2ApertureConfig:
        raise TypeError("config must be VQ2ApertureConfig")
    if type(image_bgr) is not np.ndarray or image_bgr.dtype != np.uint8:
        raise TypeError("image_bgr must be a uint8 NumPy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")
    if image_bgr.shape[0] < 1 or image_bgr.shape[1] < 1:
        raise ValueError("image_bgr must have nonzero width and height")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in VQ2_GATE_HSV_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    kernel = np.ones(
        (config.morph_kernel_size, config.morph_kernel_size), dtype=np.uint8
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    split_at = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split_at)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _largest_unambiguous_gap(
    runs: list[tuple[int, int]],
    *,
    minimum: int,
    competing_ratio: float,
) -> tuple[Optional[tuple[int, int]], bool]:
    gaps = [
        (following[0] - preceding[1] - 1, preceding[1], following[0])
        for preceding, following in zip(runs, runs[1:])
        if following[0] - preceding[1] - 1 >= minimum
    ]
    if not gaps:
        return None, False
    gaps.sort(key=lambda row: (-row[0], row[1], row[2]))
    if len(gaps) > 1 and gaps[1][0] >= competing_ratio * gaps[0][0]:
        return None, True
    return (gaps[0][1], gaps[0][2]), False


def _scanline_pairs(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    horizontal_scan: bool,
    minimum_gap: int,
    competing_ratio: float,
) -> tuple[dict[ApertureSide, list[Point]], int]:
    x, y, width, height = bbox
    first_side = ApertureSide.LEFT if horizontal_scan else ApertureSide.TOP
    second_side = ApertureSide.RIGHT if horizontal_scan else ApertureSide.BOTTOM
    points = {first_side: [], second_side: []}
    ambiguous = 0
    scan_count = height if horizontal_scan else width
    for offset in range(scan_count):
        values = (
            component[y + offset, x : x + width]
            if horizontal_scan
            else component[y : y + height, x + offset]
        )
        nonzero = np.flatnonzero(values)
        gap, is_ambiguous = _largest_unambiguous_gap(
            _runs(nonzero),
            minimum=minimum_gap,
            competing_ratio=competing_ratio,
        )
        if is_ambiguous:
            ambiguous += 1
            continue
        if gap is None:
            continue
        before, after = gap
        if horizontal_scan:
            row = float(y + offset) + 0.5
            points[first_side].append((float(x + before) + 0.5, row))
            points[second_side].append((float(x + after) - 0.5, row))
        else:
            column = float(x + offset) + 0.5
            points[first_side].append((column, float(y + before) + 0.5))
            points[second_side].append((column, float(y + after) - 0.5))
    return points, ambiguous


def _interior_interval(
    first_points: list[Point],
    second_points: list[Point],
    *,
    x_axis: bool,
) -> Optional[tuple[int, int]]:
    coordinate = 0 if x_axis else 1
    if not first_points or not second_points:
        return None
    lower = math.ceil(max(point[coordinate] for point in first_points) + 1.0)
    upper = math.floor(min(point[coordinate] for point in second_points) - 1.0)
    if lower > upper:
        return None
    return lower, upper


def _recover_opposite_line(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    clipped_side: ApertureSide,
    adjacent_first: list[Point],
    adjacent_second: list[Point],
    margin: int,
) -> tuple[ApertureSide, list[Point]]:
    x, y, width, height = bbox
    if clipped_side in {ApertureSide.TOP, ApertureSide.BOTTOM}:
        interval = _interior_interval(
            adjacent_first, adjacent_second, x_axis=True
        )
        if interval is None:
            return ApertureSide.NONE, []
        recovered = (
            ApertureSide.BOTTOM
            if clipped_side == ApertureSide.TOP
            else ApertureSide.TOP
        )
        points: list[Point] = []
        for column in range(
            max(x, interval[0]), min(x + width - 1, interval[1]) + 1
        ):
            runs = _runs(np.flatnonzero(component[y : y + height, column]))
            if not runs:
                continue
            if clipped_side == ApertureSide.TOP:
                start, end = runs[-1]
                if end < height - 1 - margin or start == 0:
                    continue
                points.append((float(column) + 0.5, float(y + start) - 0.5))
            else:
                start, end = runs[0]
                if start > margin or end == height - 1:
                    continue
                points.append((float(column) + 0.5, float(y + end) + 0.5))
        return recovered, points

    interval = _interior_interval(adjacent_first, adjacent_second, x_axis=False)
    if interval is None:
        return ApertureSide.NONE, []
    recovered = (
        ApertureSide.RIGHT
        if clipped_side == ApertureSide.LEFT
        else ApertureSide.LEFT
    )
    points = []
    for row in range(
        max(y, interval[0]), min(y + height - 1, interval[1]) + 1
    ):
        runs = _runs(np.flatnonzero(component[row, x : x + width]))
        if not runs:
            continue
        if clipped_side == ApertureSide.LEFT:
            start, end = runs[-1]
            if end < width - 1 - margin or start == 0:
                continue
            points.append((float(x + start) - 0.5, float(row) + 0.5))
        else:
            start, end = runs[0]
            if start > margin or end == width - 1:
                continue
            points.append((float(x + end) + 0.5, float(row) + 0.5))
    return recovered, points


def _fit_line(
    points: list[Point],
    *,
    side: ApertureSide,
    minimum_samples: int,
) -> Optional[_LineFit]:
    if len(points) < minimum_samples:
        return None
    raw = np.asarray(points, dtype=np.float64)
    if side in {ApertureSide.LEFT, ApertureSide.RIGHT}:
        independent = raw[:, 1]
        dependent = raw[:, 0]
    else:
        independent = raw[:, 0]
        dependent = raw[:, 1]
    design = np.column_stack((independent, np.ones_like(independent)))
    slope, intercept = np.linalg.lstsq(design, dependent, rcond=None)[0]
    residuals = dependent - (slope * independent + intercept)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    threshold = max(0.75, 3.0 * 1.4826 * mad)
    inliers = np.abs(residuals - median) <= threshold
    if int(np.count_nonzero(inliers)) < minimum_samples:
        return None
    inlier_design = design[inliers]
    inlier_dependent = dependent[inliers]
    slope, intercept = np.linalg.lstsq(
        inlier_design, inlier_dependent, rcond=None
    )[0]
    if side in {ApertureSide.LEFT, ApertureSide.RIGHT}:
        coefficients = (1.0, -float(slope), -float(intercept))
    else:
        coefficients = (-float(slope), 1.0, -float(intercept))
    norm = math.hypot(coefficients[0], coefficients[1])
    inlier_rows = raw[inliers]
    distances = (
        coefficients[0] * inlier_rows[:, 0]
        + coefficients[1] * inlier_rows[:, 1]
        + coefficients[2]
    ) / norm
    residual_rms = math.sqrt(float(np.mean(np.square(distances))))
    return _LineFit(
        coefficients=coefficients,
        inlier_points=tuple((float(row[0]), float(row[1])) for row in inlier_rows),
        residual_rms_px=residual_rms,
        support_count=len(points),
    )


def _intersection(first: _LineFit, second: _LineFit) -> Optional[Point]:
    a1, b1, c1 = first.coefficients
    a2, b2, c2 = second.coefficients
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) <= 1e-9:
        return None
    return (
        (b1 * c2 - b2 * c1) / determinant,
        (c1 * a2 - c2 * a1) / determinant,
    )


def _unit_toward(line: _LineFit, *, axis: int, sign: int) -> Point:
    a, b, _ = line.coefficients
    dx, dy = b, -a
    length = math.hypot(dx, dy)
    dx, dy = dx / length, dy / length
    component = dx if axis == 0 else dy
    if component * sign < 0.0:
        dx, dy = -dx, -dy
    return dx, dy


def _distance(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _infer_one_side(
    lines: dict[ApertureSide, _LineFit],
    missing: ApertureSide,
) -> Optional[Quad]:
    left = lines.get(ApertureSide.LEFT)
    top = lines.get(ApertureSide.TOP)
    right = lines.get(ApertureSide.RIGHT)
    bottom = lines.get(ApertureSide.BOTTOM)
    if missing == ApertureSide.TOP and left and right and bottom:
        bottom_left = _intersection(left, bottom)
        bottom_right = _intersection(right, bottom)
        if bottom_left is None or bottom_right is None:
            return None
        span = _distance(bottom_left, bottom_right)
        dl = _unit_toward(left, axis=1, sign=-1)
        dr = _unit_toward(right, axis=1, sign=-1)
        return (
            (bottom_left[0] + dl[0] * span, bottom_left[1] + dl[1] * span),
            (bottom_right[0] + dr[0] * span, bottom_right[1] + dr[1] * span),
            bottom_right,
            bottom_left,
        )
    if missing == ApertureSide.BOTTOM and left and right and top:
        top_left = _intersection(left, top)
        top_right = _intersection(right, top)
        if top_left is None or top_right is None:
            return None
        span = _distance(top_left, top_right)
        dl = _unit_toward(left, axis=1, sign=1)
        dr = _unit_toward(right, axis=1, sign=1)
        return (
            top_left,
            top_right,
            (top_right[0] + dr[0] * span, top_right[1] + dr[1] * span),
            (top_left[0] + dl[0] * span, top_left[1] + dl[1] * span),
        )
    if missing == ApertureSide.LEFT and top and right and bottom:
        top_right = _intersection(top, right)
        bottom_right = _intersection(bottom, right)
        if top_right is None or bottom_right is None:
            return None
        span = _distance(top_right, bottom_right)
        dt = _unit_toward(top, axis=0, sign=-1)
        db = _unit_toward(bottom, axis=0, sign=-1)
        return (
            (top_right[0] + dt[0] * span, top_right[1] + dt[1] * span),
            top_right,
            bottom_right,
            (bottom_right[0] + db[0] * span, bottom_right[1] + db[1] * span),
        )
    if missing == ApertureSide.RIGHT and left and top and bottom:
        top_left = _intersection(left, top)
        bottom_left = _intersection(left, bottom)
        if top_left is None or bottom_left is None:
            return None
        span = _distance(top_left, bottom_left)
        dt = _unit_toward(top, axis=0, sign=1)
        db = _unit_toward(bottom, axis=0, sign=1)
        return (
            top_left,
            (top_left[0] + dt[0] * span, top_left[1] + dt[1] * span),
            (bottom_left[0] + db[0] * span, bottom_left[1] + db[1] * span),
            bottom_left,
        )
    return None


def _visible_quad(lines: dict[ApertureSide, _LineFit]) -> Optional[Quad]:
    corners = (
        _intersection(lines[ApertureSide.LEFT], lines[ApertureSide.TOP]),
        _intersection(lines[ApertureSide.TOP], lines[ApertureSide.RIGHT]),
        _intersection(lines[ApertureSide.RIGHT], lines[ApertureSide.BOTTOM]),
        _intersection(lines[ApertureSide.BOTTOM], lines[ApertureSide.LEFT]),
    )
    if any(corner is None for corner in corners):
        return None
    return corners  # type: ignore[return-value]


def _valid_clockwise_quad(corners: Quad, *, minimum_span: float) -> bool:
    if not all(math.isfinite(value) for point in corners for value in point):
        return False
    crosses = []
    for index in range(4):
        first = corners[index]
        second = corners[(index + 1) % 4]
        third = corners[(index + 2) % 4]
        crosses.append(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )
    if any(value <= 1e-6 for value in crosses):
        return False
    top_y = 0.5 * (corners[0][1] + corners[1][1])
    bottom_y = 0.5 * (corners[3][1] + corners[2][1])
    left_x = 0.5 * (corners[0][0] + corners[3][0])
    right_x = 0.5 * (corners[1][0] + corners[2][0])
    if not top_y < bottom_y or not left_x < right_x:
        return False
    return (
        min(_distance(corners[i], corners[(i + 1) % 4]) for i in range(4))
        >= minimum_span
    )


def _outside_clipping(corners: Quad, width: int, height: int) -> ApertureSide:
    result = ApertureSide.NONE
    for x, y in corners:
        if x < 0.0:
            result |= ApertureSide.LEFT
        elif x > width:
            result |= ApertureSide.RIGHT
        if y < 0.0:
            result |= ApertureSide.TOP
        elif y > height:
            result |= ApertureSide.BOTTOM
    return result


def _clip_segment_to_image(
    first: Point,
    second: Point,
    width: int,
    height: int,
) -> Optional[Segment]:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    t_min, t_max = 0.0, 1.0
    for p, q in (
        (-dx, first[0]),
        (dx, width - first[0]),
        (-dy, first[1]),
        (dy, height - first[1]),
    ):
        if abs(p) <= 1e-12:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            t_min = max(t_min, ratio)
        else:
            t_max = min(t_max, ratio)
        if t_min > t_max:
            return None
    start = (first[0] + t_min * dx, first[1] + t_min * dy)
    end = (first[0] + t_max * dx, first[1] + t_max * dy)
    if _distance(start, end) <= 1e-9:
        return None
    return start, end


def _covariance_diagonal(
    *,
    corners: Quad,
    image_width: int,
    image_height: int,
    residual_rms_px: float,
    inferred_side: Optional[ApertureSide],
    prior_relative_sigma: float,
) -> tuple[float, float, float, float, float]:
    spans = tuple(_distance(corners[i], corners[(i + 1) % 4]) for i in range(4))
    aperture_span = max(1.0, min(spans))
    base_sigma_px = max(0.5, residual_rms_px)
    sigma_x_px = base_sigma_px
    sigma_y_px = base_sigma_px
    scale_sigma = max(0.005, base_sigma_px / aperture_span)
    skew_sigma = max(0.01, 2.0 * base_sigma_px / aperture_span)
    if inferred_side is not None:
        prior_sigma_px = prior_relative_sigma * aperture_span
        if inferred_side in {ApertureSide.TOP, ApertureSide.BOTTOM}:
            sigma_x_px = max(sigma_x_px, 0.25 * prior_sigma_px)
            sigma_y_px = max(sigma_y_px, 0.50 * prior_sigma_px)
        else:
            sigma_x_px = max(sigma_x_px, 0.50 * prior_sigma_px)
            sigma_y_px = max(sigma_y_px, 0.25 * prior_sigma_px)
        scale_sigma = max(scale_sigma, prior_relative_sigma)
        skew_sigma = max(skew_sigma, 2.0 * prior_relative_sigma)
    values = (
        (2.0 * sigma_x_px / image_width) ** 2,
        (2.0 * sigma_y_px / image_height) ** 2,
        scale_sigma**2,
        skew_sigma**2,
        skew_sigma**2,
    )
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def fit_vq2_aperture_mask(
    mask: np.ndarray,
    support_bbox_px: tuple[int, int, int, int],
    *,
    detection_confidence: float,
    config: VQ2ApertureConfig = VQ2ApertureConfig(),
) -> VQ2ApertureFit:
    """Fit an inner aperture from a pre-thresholded gate-colour mask."""

    if type(config) is not VQ2ApertureConfig:
        raise TypeError("config must be VQ2ApertureConfig")
    if type(mask) is not np.ndarray or mask.ndim != 2:
        raise TypeError("mask must be a two-dimensional NumPy array")
    if mask.shape[0] < 1 or mask.shape[1] < 1:
        raise ValueError("mask must have nonzero width and height")
    if mask.dtype.kind not in {"b", "i", "u"}:
        raise TypeError("mask must contain boolean or integer data")
    image_height, image_width = mask.shape
    bbox = _exact_bbox(
        support_bbox_px,
        image_width=image_width,
        image_height=image_height,
    )
    confidence = _finite_confidence(detection_confidence)
    clipping = _clipping_mask(
        bbox,
        image_width,
        image_height,
        config.boundary_margin_px,
    )
    image_size = (image_width, image_height)
    x, y, width, height = bbox
    crop = np.asarray(mask[y : y + height, x : x + width] != 0, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        crop, connectivity=8
    )
    if component_count <= 1:
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="no_connected_gate_support",
        )
    components = sorted(
        (
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, component_count)
        ),
        reverse=True,
    )
    largest_area, largest_label = components[0]
    if largest_area < config.min_component_pixels:
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="insufficient_connected_support",
            support_count=largest_area,
        )
    if (
        len(components) > 1
        and components[1][0] >= config.secondary_component_ratio * largest_area
    ):
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="ambiguous_connected_components",
            support_count=largest_area + components[1][0],
        )
    component_left = int(stats[largest_label, cv2.CC_STAT_LEFT])
    component_top = int(stats[largest_label, cv2.CC_STAT_TOP])
    component_width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    component_height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    alignment_margin = max(config.boundary_margin_px, config.morph_kernel_size)
    if (
        component_left > alignment_margin
        or component_top > alignment_margin
        or component_left + component_width < width - alignment_margin
        or component_top + component_height < height - alignment_margin
    ):
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="connected_support_does_not_match_bbox",
            support_count=largest_area,
        )
    component = np.zeros(mask.shape, dtype=np.uint8)
    component[y : y + height, x : x + width] = labels == largest_label
    minimum_gap = max(config.min_gap_px, math.ceil(0.08 * min(width, height)))
    horizontal, ambiguous_rows = _scanline_pairs(
        component,
        bbox,
        horizontal_scan=True,
        minimum_gap=minimum_gap,
        competing_ratio=config.competing_gap_ratio,
    )
    vertical, ambiguous_columns = _scanline_pairs(
        component,
        bbox,
        horizontal_scan=False,
        minimum_gap=minimum_gap,
        competing_ratio=config.competing_gap_ratio,
    )
    if (
        ambiguous_rows >= config.min_line_samples
        or ambiguous_columns >= config.min_line_samples
    ):
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="ambiguous_multiple_aperture_gaps",
            support_count=largest_area,
        )
    samples: dict[ApertureSide, list[Point]] = {
        ApertureSide.LEFT: horizontal[ApertureSide.LEFT],
        ApertureSide.TOP: vertical[ApertureSide.TOP],
        ApertureSide.RIGHT: horizontal[ApertureSide.RIGHT],
        ApertureSide.BOTTOM: vertical[ApertureSide.BOTTOM],
    }

    # With one clipped side, scanlines parallel to that side may see only the
    # opposite coloured bar.  Recover that *visible opposite* boundary before
    # applying the censored square prior to the missing side itself.
    clipping_members = [side for side in _SIDES if clipping & side]
    if len(clipping_members) == 1:
        clipped = clipping_members[0]
        if clipped in {ApertureSide.TOP, ApertureSide.BOTTOM}:
            adjacent_first = samples[ApertureSide.LEFT]
            adjacent_second = samples[ApertureSide.RIGHT]
        else:
            adjacent_first = samples[ApertureSide.TOP]
            adjacent_second = samples[ApertureSide.BOTTOM]
        recovered_side, recovered_points = _recover_opposite_line(
            component,
            bbox,
            clipped_side=clipped,
            adjacent_first=adjacent_first,
            adjacent_second=adjacent_second,
            margin=config.boundary_margin_px,
        )
        if recovered_side != ApertureSide.NONE and len(
            samples[recovered_side]
        ) < config.min_line_samples:
            samples[recovered_side] = recovered_points

    lines: dict[ApertureSide, _LineFit] = {}
    for side in _SIDES:
        line = _fit_line(
            samples[side],
            side=side,
            minimum_samples=config.min_line_samples,
        )
        if line is not None:
            lines[side] = line
    missing = [side for side in _SIDES if side not in lines]
    inferred_side: Optional[ApertureSide] = None
    if not missing:
        corners = _visible_quad(lines)
        geometry_model_id = "vq2-visible-inner-quad-lines-v1"
    elif len(missing) == 1 and bool(clipping & missing[0]):
        inferred_side = missing[0]
        corners = _infer_one_side(lines, inferred_side)
        geometry_model_id = "vq2-censored-image-square-px-v1"
    else:
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="underconstrained_inner_aperture",
            support_count=sum(len(points) for points in samples.values()),
        )
    if corners is None or not _valid_clockwise_quad(
        corners, minimum_span=float(minimum_gap)
    ):
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="degenerate_inner_aperture",
            support_count=sum(len(points) for points in samples.values()),
        )
    clipping |= _outside_clipping(corners, image_width, image_height)
    # The frozen contract allows inferred geometry only to +/-4 normalized.
    if any(
        px < -1.5 * image_width
        or px > 2.5 * image_width
        or py < -1.5 * image_height
        or py > 2.5 * image_height
        for px, py in corners
    ):
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="inference_outside_contract_range",
            support_count=sum(len(points) for points in samples.values()),
        )
    visible_edges = ApertureSide.NONE
    for side in lines:
        visible_edges |= side
    adjacent_support = (
        bool(visible_edges & ApertureSide.LEFT and visible_edges & ApertureSide.TOP),
        bool(visible_edges & ApertureSide.TOP and visible_edges & ApertureSide.RIGHT),
        bool(visible_edges & ApertureSide.RIGHT and visible_edges & ApertureSide.BOTTOM),
        bool(visible_edges & ApertureSide.BOTTOM and visible_edges & ApertureSide.LEFT),
    )
    visible_corners = tuple(
        adjacent_support[index]
        and 0.0 <= corners[index][0] <= image_width
        and 0.0 <= corners[index][1] <= image_height
        for index in range(4)
    )
    corner_edges = (
        (corners[0], corners[3]),
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[3], corners[2]),
    )
    visible_segments = tuple(
        _clip_segment_to_image(*corner_edges[index], image_width, image_height)
        if visible_edges & side
        else None
        for index, side in enumerate(_SIDES)
    )
    if any(
        visible_edges & side and visible_segments[index] is None
        for index, side in enumerate(_SIDES)
    ):
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="visible_line_outside_image",
            support_count=sum(len(points) for points in samples.values()),
        )
    residual_numerator = sum(
        len(line.inlier_points) * line.residual_rms_px**2 for line in lines.values()
    )
    inlier_count = sum(len(line.inlier_points) for line in lines.values())
    support_count = sum(line.support_count for line in lines.values())
    residual_rms = math.sqrt(residual_numerator / max(inlier_count, 1))
    expected_edge_support = 2.0 * (
        _distance(corners[0], corners[1]) + _distance(corners[0], corners[3])
    )
    coverage = min(1.0, inlier_count / max(expected_edge_support, 1.0))
    residual_score = math.exp(
        -residual_rms / max(1.0, 0.02 * min(width, height))
    )
    censor_factor = 0.65 if inferred_side is not None else 1.0
    fit_confidence = max(
        0.0,
        min(
            1.0,
            confidence
            * (0.55 + 0.45 * coverage)
            * residual_score
            * censor_factor,
        ),
    )
    covariance = _covariance_diagonal(
        corners=corners,
        image_width=image_width,
        image_height=image_height,
        residual_rms_px=residual_rms,
        inferred_side=inferred_side,
        prior_relative_sigma=config.square_prior_relative_sigma,
    )
    return VQ2ApertureFit(
        image_size_px=image_size,
        support_bbox_px=bbox,
        clipping=clipping,
        fitted_corners_px=corners,
        visible_edges=visible_edges,
        visible_corners=visible_corners,
        visible_segments_px=visible_segments,  # type: ignore[arg-type]
        geometry_model_id=geometry_model_id,
        covariance_model_id=(
            "vq2-censored-aperture-diagonal-v1"
            if inferred_side is not None
            else "vq2-visible-aperture-diagonal-v1"
        ),
        covariance_diagonal=covariance,
        residual_rms_px=residual_rms,
        inlier_count=inlier_count,
        support_count=support_count,
        confidence=fit_confidence,
        rejection_reason=None,
    )


def fit_vq2_aperture_bgr(
    image_bgr: np.ndarray,
    support_bbox_px: tuple[int, int, int, int],
    *,
    detection_confidence: float,
    config: VQ2ApertureConfig = VQ2ApertureConfig(),
) -> VQ2ApertureFit:
    """Threshold a BGR frame and fit its VQ2 inner aperture."""

    mask = vq2_gate_mask_from_bgr(image_bgr, config=config)
    return fit_vq2_aperture_mask(
        mask,
        support_bbox_px,
        detection_confidence=detection_confidence,
        config=config,
    )


def passage_geometry_from_vq2_aperture_fit(
    fit: VQ2ApertureFit,
    *,
    minimum_confidence: float = 0.25,
) -> Optional[VQ2PassageGeometry]:
    """Return a conservative inscribed opening only from a nominal fit.

    Detector support bounds are deliberately excluded.  A clipped,
    under-supported, or low-confidence fit may remain useful diagnostic
    evidence, but it cannot manufacture aperture-relative crossing clearance.
    """

    if type(fit) is not VQ2ApertureFit:
        raise TypeError("fit must be an exact VQ2ApertureFit")
    confidence_floor = _finite_confidence(minimum_confidence)
    all_sides = (
        ApertureSide.LEFT
        | ApertureSide.TOP
        | ApertureSide.RIGHT
        | ApertureSide.BOTTOM
    )
    if (
        not fit.succeeded
        or fit.rejection_reason is not None
        or fit.clipping != ApertureSide.NONE
        or fit.visible_edges != all_sides
        or fit.visible_corners != (True, True, True, True)
        or fit.confidence < confidence_floor
        or fit.fitted_corners_px is None
        or fit.geometry_model_id
        != "vq2-visible-inner-quad-lines-v1"
        or fit.covariance_model_id
        != "vq2-visible-aperture-diagonal-v1"
        or fit.covariance_diagonal is None
        or fit.residual_rms_px is None
        or not math.isfinite(fit.residual_rms_px)
        or fit.inlier_count <= 0
        or fit.support_count < fit.inlier_count
    ):
        return None

    width, height = fit.image_size_px
    corners = tuple(
        (
            2.0 * point[0] / float(width) - 1.0,
            2.0 * point[1] / float(height) - 1.0,
        )
        for point in fit.fitted_corners_px
    )
    first_diagonal = (
        corners[2][0] - corners[0][0],
        corners[2][1] - corners[0][1],
    )
    second_diagonal = (
        corners[3][0] - corners[1][0],
        corners[3][1] - corners[1][1],
    )
    offset = (
        corners[1][0] - corners[0][0],
        corners[1][1] - corners[0][1],
    )
    cross = (
        first_diagonal[0] * second_diagonal[1]
        - first_diagonal[1] * second_diagonal[0]
    )
    if abs(cross) <= 1e-12:
        return None
    fraction = (
        offset[0] * second_diagonal[1]
        - offset[1] * second_diagonal[0]
    ) / cross
    center = (
        corners[0][0] + fraction * first_diagonal[0],
        corners[0][1] + fraction * first_diagonal[1],
    )

    # Find the maximum-area axis-aligned rectangle centred at the diagonal
    # intersection.  Each convex-quad edge contributes the half-plane
    # constraint ``|dy| * half_x + |dx| * half_y <= center_slack``.
    # Independent centerline spans are insufficient: their simultaneous
    # corner can fall outside a projective/trapezoidal opening.
    constraints: list[tuple[float, float, float]] = []
    orientation: Optional[float] = None
    for index, first in enumerate(corners):
        second = corners[(index + 1) % 4]
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        signed_slack = (
            dx * (center[1] - first[1])
            - dy * (center[0] - first[0])
        )
        if abs(signed_slack) <= 1e-12:
            return None
        if orientation is None:
            orientation = 1.0 if signed_slack > 0.0 else -1.0
        slack = orientation * signed_slack
        if slack <= 0.0:
            return None
        constraints.append((abs(dy), abs(dx), slack))

    candidates: list[Point] = []
    for coefficient_x, coefficient_y, slack in constraints:
        if coefficient_x > 1e-12 and coefficient_y > 1e-12:
            candidates.append(
                (
                    slack / (2.0 * coefficient_x),
                    slack / (2.0 * coefficient_y),
                )
            )
    for first_index, first in enumerate(constraints):
        for second in constraints[first_index + 1 :]:
            determinant = (
                first[0] * second[1] - second[0] * first[1]
            )
            if abs(determinant) <= 1e-12:
                continue
            half_x = (
                first[2] * second[1] - second[2] * first[1]
            ) / determinant
            half_y = (
                first[0] * second[2] - second[0] * first[2]
            ) / determinant
            candidates.append((half_x, half_y))
    feasible = tuple(
        candidate
        for candidate in candidates
        if candidate[0] > 0.0
        and candidate[1] > 0.0
        and all(
            coefficient_x * candidate[0]
            + coefficient_y * candidate[1]
            <= slack + 1e-10
            for coefficient_x, coefficient_y, slack in constraints
        )
    )
    if not feasible:
        return None
    half_size = max(
        feasible,
        key=lambda candidate: candidate[0] * candidate[1],
    )
    if (
        not all(math.isfinite(value) and value > 0.0 for value in half_size)
        or any(not math.isfinite(value) for value in center)
    ):
        return None
    measurement_std = tuple(
        math.sqrt(value) for value in fit.covariance_diagonal[:3]
    )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in measurement_std
    ):
        return None
    return VQ2PassageGeometry(
        center_norm=center,
        aperture_half_size_norm=half_size,
        # Match the dynamic controller's historical half-extent scale
        # convention.  An axis-aligned opening therefore changes no units.
        log_scale=0.5 * math.log(half_size[0] * half_size[1]),
        measurement_std=measurement_std,  # type: ignore[arg-type]
    )


__all__ = [
    "ApertureSide",
    "VQ2ApertureConfig",
    "VQ2ApertureFit",
    "VQ2PassageGeometry",
    "fit_vq2_aperture_bgr",
    "fit_vq2_aperture_mask",
    "passage_geometry_from_vq2_aperture_fit",
    "vq2_gate_mask_from_bgr",
]
