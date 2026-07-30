"""Deterministic image-space inner-aperture geometry for VQ2 gates.

The build-3385 detector's bounding box is contour support, not a gate edge.
This module instead extracts the boundary of the dark opening from the known
VQ2 colour mask.  Four visible sides are fit directly.  When exactly one side
is absent because that same image border clips the detection, the missing side
is censored and inferred with an explicitly named square-in-pixel-space prior.

This is not calibrated pose, distance, or physical square evidence.  Ambiguous
multiple openings and geometry with fewer than three supported sides are
rejected rather than assigned precise fitted values.

Panel-style gates (F80: a solid sponsor panel fills the blob below the true
opening) give some scanlines two comparable dark gaps, which the competing-gap
test must not silently pick between.  When exactly one dominant *enclosed*
dark region exists inside the support and it is bounded above by the
component's top bar, that region names the opening and each scanline selects
the unique gap overlapping it.  Comparable paired regions, regions touching
the crop border, and regions without top-bar support (e.g. a pylon sponsor
window) fall through to the unchanged legacy behaviour.
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
_PASSAGE_MINIMUM_CONFIDENCE = 0.25
_CONFIDENCE_UNCERTAINTY_EPSILON = 1e-6
_VISIBLE_GEOMETRY_MODEL_ID = "vq2-visible-inner-quad-lines-v1"
_VISIBLE_COVARIANCE_MODEL_ID = "vq2-visible-aperture-diagonal-v1"
_TEMPORAL_GEOMETRY_MODEL_ID = (
    "vq2-temporally-associated-inner-quad-lines-v1"
)
_TEMPORAL_COVARIANCE_MODEL_ID = (
    "vq2-temporally-associated-aperture-diagonal-v1"
)

# Enclosed-region disambiguation (panel-style gates).  A candidate opening
# must be bounded above by the component's top bar: walking upward from the
# region through (speckle-closed) support pixels must reach the component's
# top edge band in at least this fraction of the region's columns.  F80
# evidence: true openings score 0.43-1.00, the pylon sponsor window 0.00.
_APERTURE_REGION_TOP_BAR_CLOSE_KERNEL = 5
_APERTURE_REGION_TOP_BAR_MIN_FRACTION = 0.25
# Confidence calibration: the residual score normalises rms by two percent of
# the smaller support dimension, which drops below one pixel for gates under
# fifty pixels and over-punishes the ragged mask edges of resolved panel-gate
# openings (a verified-correct 3.9 px rms on a 90 px gate scored 0.12 and
# never reached the 0.20 steering floor).  A 4.5 px floor keeps correct
# small-gate fits steerable while a wrong hybrid fit (6.4 px rms on a 62 px
# support, F80 f1640361) still scores <= 0.17 confidence.  Gates above
# 225 px are unchanged.  This scales confidence only; covariance and
# measurement_std keep using the raw pixel rms.
_RESIDUAL_SCORE_MIN_SCALE_PX = 4.5


@dataclass(frozen=True, slots=True)
class VQ2ApertureConfig:
    """Conservative thresholds for deterministic scanline fitting."""

    boundary_margin_px: int = 2
    morph_kernel_size: int = 3
    min_component_pixels: int = 24
    min_gap_px: int = 4
    min_line_samples: int = 4
    min_aperture_region_pixels: int = 16
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
            "min_aperture_region_pixels",
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
        if self.min_aperture_region_pixels < 1:
            raise ValueError("min_aperture_region_pixels must be positive")
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
class VQ2ApertureTrackingPrior:
    """A current-frame inner-quad prediction used only to resolve ambiguity.

    The caller owns temporal propagation and supplies a conservative maximum
    boundary residual in pixels.  This prior cannot create an aperture when
    the ordinary scanline fit is merely absent or underconstrained; it is
    consulted only after the unchanged competing-gap test declares the frame
    ambiguous.
    """

    center_px: Point
    half_size_px: Point
    maximum_boundary_residual_px: float

    def __post_init__(self) -> None:
        normalized: dict[str, Point] = {}
        for name, point in (
            ("center_px", self.center_px),
            ("half_size_px", self.half_size_px),
        ):
            if type(point) is not tuple or len(point) != 2:
                raise TypeError(
                    f"{name} must be an exact two-tuple"
                )
            values: list[float] = []
            for axis, value in enumerate(point):
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                ):
                    raise TypeError(
                        f"{name}[{axis}] must be finite numeric data"
                    )
                values.append(float(value))
            normalized[name] = (values[0], values[1])
        if any(value <= 0.0 for value in normalized["half_size_px"]):
            raise ValueError(
                "half_size_px values must be positive"
            )
        residual = self.maximum_boundary_residual_px
        if (
            type(residual) not in {int, float}
            or not math.isfinite(float(residual))
        ):
            raise TypeError(
                "maximum_boundary_residual_px must be finite numeric data"
            )
        if float(residual) <= 0.0:
            raise ValueError(
                "maximum_boundary_residual_px must be positive"
            )
        object.__setattr__(
            self,
            "center_px",
            normalized["center_px"],
        )
        object.__setattr__(
            self,
            "half_size_px",
            normalized["half_size_px"],
        )
        object.__setattr__(
            self,
            "maximum_boundary_residual_px",
            float(residual),
        )

    @property
    def predicted_corners_px(self) -> Quad:
        center_x, center_y = self.center_px
        half_x, half_y = self.half_size_px
        return (
            (center_x - half_x, center_y - half_y),
            (center_x + half_x, center_y - half_y),
            (center_x + half_x, center_y + half_y),
            (center_x - half_x, center_y + half_y),
        )


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
    """Conservative normalized geometry from one complete visible inner fit."""

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


@dataclass(frozen=True, slots=True)
class _GapCandidate:
    width: int
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class _AmbiguousGapScanline:
    offset: int
    candidates: tuple[_GapCandidate, ...]


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


def _gap_candidates(
    runs: list[tuple[int, int]],
    *,
    minimum: int,
) -> tuple[_GapCandidate, ...]:
    gaps = tuple(
        _GapCandidate(
            width=following[0] - preceding[1] - 1,
            before=preceding[1],
            after=following[0],
        )
        for preceding, following in zip(runs, runs[1:])
        if following[0] - preceding[1] - 1 >= minimum
    )
    return tuple(
        sorted(
            gaps,
            key=lambda candidate: (
                -candidate.width,
                candidate.before,
                candidate.after,
            ),
        )
    )


def _largest_unambiguous_gap(
    runs: list[tuple[int, int]],
    *,
    minimum: int,
    competing_ratio: float,
) -> tuple[Optional[tuple[int, int]], bool]:
    gaps = _gap_candidates(runs, minimum=minimum)
    if not gaps:
        return None, False
    if (
        len(gaps) > 1
        and gaps[1].width >= competing_ratio * gaps[0].width
    ):
        return None, True
    return (gaps[0].before, gaps[0].after), False


def _enclosed_aperture_region(
    component_crop: np.ndarray,
    *,
    minimum_area: int,
    competing_ratio: float,
    top_bar_margin: int,
) -> Optional[np.ndarray]:
    """Name the opening as the dominant enclosed dark region, if unambiguous.

    Returns a crop-shaped boolean mask of the selected region, or ``None``
    when no enclosed region exists, when two comparable regions compete, or
    when the dominant region lacks top-bar support (pylon-style sponsor
    windows are enclosed dark regions too, but sit under a deep panel rather
    than directly below the top bar).  ``None`` means the caller keeps the
    unchanged legacy scanline behaviour.
    """

    dark = (component_crop == 0).astype(np.uint8)
    region_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark, connectivity=4
    )
    crop_height, crop_width = dark.shape
    enclosed: list[tuple[int, int]] = []
    for index in range(1, region_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if (
            left == 0
            or top == 0
            or left + width >= crop_width
            or top + height >= crop_height
        ):
            continue
        enclosed.append((area, index))
    if not enclosed:
        return None
    enclosed.sort(reverse=True)
    if (
        len(enclosed) > 1
        and enclosed[1][0] >= competing_ratio * enclosed[0][0]
    ):
        return None
    region = labels == enclosed[0][1]
    # Close support speckle (JPEG/morphology pinholes in the top bar) before
    # the upward walk; the walk, not the fit, uses this copy.
    walk = cv2.morphologyEx(
        component_crop,
        cv2.MORPH_CLOSE,
        np.ones(
            (_APERTURE_REGION_TOP_BAR_CLOSE_KERNEL,) * 2, dtype=np.uint8
        ),
    )
    columns = np.flatnonzero(region.any(axis=0))
    if columns.size == 0:
        return None
    supported = 0
    for column in columns:
        top_row = int(np.flatnonzero(region[:, column])[0])
        row = top_row - 1
        while row >= 0 and walk[row, column]:
            row -= 1
        if row <= top_bar_margin:
            supported += 1
    if supported < _APERTURE_REGION_TOP_BAR_MIN_FRACTION * columns.size:
        return None
    return region


def _scanline_pairs(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    horizontal_scan: bool,
    minimum_gap: int,
    competing_ratio: float,
    aperture_region: Optional[np.ndarray] = None,
) -> tuple[
    dict[ApertureSide, list[Point]],
    tuple[_AmbiguousGapScanline, ...],
]:
    x, y, width, height = bbox
    first_side = ApertureSide.LEFT if horizontal_scan else ApertureSide.TOP
    second_side = ApertureSide.RIGHT if horizontal_scan else ApertureSide.BOTTOM
    points = {first_side: [], second_side: []}
    ambiguous: list[_AmbiguousGapScanline] = []
    scan_count = height if horizontal_scan else width
    for offset in range(scan_count):
        values = (
            component[y + offset, x : x + width]
            if horizontal_scan
            else component[y : y + height, x + offset]
        )
        nonzero = np.flatnonzero(values)
        runs = _runs(nonzero)
        if aperture_region is not None:
            # Region-guided selection: keep the unique gap overlapping the
            # named opening.  Scanlines missing the region, hitting it more
            # than once, or holding only foreign gaps (pylon windows, panel
            # slits) are skipped rather than counted ambiguous.
            candidates = _gap_candidates(runs, minimum=minimum_gap)
            overlapping = []
            for candidate in candidates:
                segment = (
                    aperture_region[
                        y + offset, x + candidate.before + 1 : x + candidate.after
                    ]
                    if horizontal_scan
                    else aperture_region[
                        y + candidate.before + 1 : y + candidate.after, x + offset
                    ]
                )
                if segment.any():
                    overlapping.append(candidate)
            if len(overlapping) != 1:
                continue
            gap = (overlapping[0].before, overlapping[0].after)
        else:
            gap, is_ambiguous = _largest_unambiguous_gap(
                runs,
                minimum=minimum_gap,
                competing_ratio=competing_ratio,
            )
            if is_ambiguous:
                ambiguous.append(
                    _AmbiguousGapScanline(
                        offset=offset,
                        candidates=_gap_candidates(
                            runs,
                            minimum=minimum_gap,
                        ),
                    )
                )
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
    return points, tuple(ambiguous)


def _temporally_resolved_scanline_pairs(
    ambiguous: tuple[_AmbiguousGapScanline, ...],
    bbox: tuple[int, int, int, int],
    *,
    horizontal_scan: bool,
    prior: VQ2ApertureTrackingPrior,
) -> tuple[dict[ApertureSide, list[Point]], int]:
    """Select the unique fresh gap containing the predicted aperture center."""

    x, y, _width, _height = bbox
    center_x, center_y = prior.center_px
    half_x, half_y = prior.half_size_px
    first_side = (
        ApertureSide.LEFT if horizontal_scan else ApertureSide.TOP
    )
    second_side = (
        ApertureSide.RIGHT if horizontal_scan else ApertureSide.BOTTOM
    )
    points = {first_side: [], second_side: []}
    selected_count = 0
    for scanline in ambiguous:
        if horizontal_scan:
            independent = float(y + scanline.offset) + 0.5
            orthogonal_lower = center_y - half_y
            orthogonal_upper = center_y + half_y
            predicted_center = center_x
        else:
            independent = float(x + scanline.offset) + 0.5
            orthogonal_lower = center_x - half_x
            orthogonal_upper = center_x + half_x
            predicted_center = center_y
        if not orthogonal_lower <= independent <= orthogonal_upper:
            continue
        matches: list[tuple[float, float]] = []
        for candidate in scanline.candidates:
            if horizontal_scan:
                first_boundary = float(x + candidate.before) + 0.5
                second_boundary = float(x + candidate.after) - 0.5
            else:
                first_boundary = float(y + candidate.before) + 0.5
                second_boundary = float(y + candidate.after) - 0.5
            if first_boundary < predicted_center < second_boundary:
                matches.append((first_boundary, second_boundary))
        if len(matches) != 1:
            continue
        first_boundary, second_boundary = matches[0]
        if horizontal_scan:
            points[first_side].append((first_boundary, independent))
            points[second_side].append((second_boundary, independent))
        else:
            points[first_side].append((independent, first_boundary))
            points[second_side].append((independent, second_boundary))
        selected_count += 1
    return points, selected_count


def _quad_contains_tracking_prior_center(
    corners: Quad,
    prior: VQ2ApertureTrackingPrior,
) -> bool:
    """Require the fitted fresh opening to contain the predicted center."""

    center_x, center_y = prior.center_px
    return all(
        (
            (second[0] - first[0]) * (center_y - first[1])
            - (second[1] - first[1]) * (center_x - first[0])
        )
        > 1e-6
        for first, second in (
            (corners[index], corners[(index + 1) % 4])
            for index in range(4)
        )
    )


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
    temporal_boundary_sigma_px: Optional[float] = None,
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
    if temporal_boundary_sigma_px is not None:
        sigma_x_px = max(sigma_x_px, temporal_boundary_sigma_px)
        sigma_y_px = max(sigma_y_px, temporal_boundary_sigma_px)
        relative_sigma = temporal_boundary_sigma_px / aperture_span
        scale_sigma = max(scale_sigma, relative_sigma)
        skew_sigma = max(skew_sigma, 2.0 * relative_sigma)
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
    tracking_prior: Optional[VQ2ApertureTrackingPrior] = None,
) -> VQ2ApertureFit:
    """Fit an inner aperture from a pre-thresholded gate-colour mask."""

    if type(config) is not VQ2ApertureConfig:
        raise TypeError("config must be VQ2ApertureConfig")
    if tracking_prior is not None and (
        type(tracking_prior) is not VQ2ApertureTrackingPrior
    ):
        raise TypeError(
            "tracking_prior must be an exact VQ2ApertureTrackingPrior or None"
        )
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
    region_crop = _enclosed_aperture_region(
        component[y : y + height, x : x + width],
        minimum_area=max(
            config.min_aperture_region_pixels, minimum_gap * minimum_gap
        ),
        competing_ratio=config.competing_gap_ratio,
        top_bar_margin=alignment_margin,
    )
    aperture_region = None
    if region_crop is not None:
        aperture_region = np.zeros(mask.shape, dtype=np.uint8)
        aperture_region[y : y + height, x : x + width] = region_crop
    horizontal, ambiguous_rows = _scanline_pairs(
        component,
        bbox,
        horizontal_scan=True,
        minimum_gap=minimum_gap,
        competing_ratio=config.competing_gap_ratio,
        aperture_region=aperture_region,
    )
    vertical, ambiguous_columns = _scanline_pairs(
        component,
        bbox,
        horizontal_scan=False,
        minimum_gap=minimum_gap,
        competing_ratio=config.competing_gap_ratio,
        aperture_region=aperture_region,
    )
    row_ambiguity = len(ambiguous_rows) >= config.min_line_samples
    column_ambiguity = (
        len(ambiguous_columns) >= config.min_line_samples
    )
    temporally_associated = False
    if row_ambiguity or column_ambiguity:
        if tracking_prior is None:
            return _rejected(
                image_size_px=image_size,
                bbox=bbox,
                clipping=clipping,
                reason="ambiguous_multiple_aperture_gaps",
                support_count=largest_area,
            )
        if row_ambiguity:
            resolved, selected_count = (
                _temporally_resolved_scanline_pairs(
                    ambiguous_rows,
                    bbox,
                    horizontal_scan=True,
                    prior=tracking_prior,
                )
            )
            if selected_count < config.min_line_samples:
                return _rejected(
                    image_size_px=image_size,
                    bbox=bbox,
                    clipping=clipping,
                    reason="ambiguous_multiple_aperture_gaps",
                    support_count=largest_area,
                )
            horizontal = resolved
        if column_ambiguity:
            resolved, selected_count = (
                _temporally_resolved_scanline_pairs(
                    ambiguous_columns,
                    bbox,
                    horizontal_scan=False,
                    prior=tracking_prior,
                )
            )
            if selected_count < config.min_line_samples:
                return _rejected(
                    image_size_px=image_size,
                    bbox=bbox,
                    clipping=clipping,
                    reason="ambiguous_multiple_aperture_gaps",
                    support_count=largest_area,
                )
            vertical = resolved
        temporally_associated = True
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
        geometry_model_id = (
            _TEMPORAL_GEOMETRY_MODEL_ID
            if temporally_associated
            else _VISIBLE_GEOMETRY_MODEL_ID
        )
    elif temporally_associated:
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="underconstrained_inner_aperture",
            support_count=sum(len(points) for points in samples.values()),
        )
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
    if (
        temporally_associated
        and tracking_prior is not None
        and not _quad_contains_tracking_prior_center(
            corners,
            tracking_prior,
        )
    ):
        return _rejected(
            image_size_px=image_size,
            bbox=bbox,
            clipping=clipping,
            reason="ambiguous_multiple_aperture_gaps",
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
        -residual_rms
        / max(_RESIDUAL_SCORE_MIN_SCALE_PX, 0.02 * min(width, height))
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
        temporal_boundary_sigma_px=(
            tracking_prior.maximum_boundary_residual_px
            if temporally_associated and tracking_prior is not None
            else None
        ),
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
            else (
                _TEMPORAL_COVARIANCE_MODEL_ID
                if temporally_associated
                else _VISIBLE_COVARIANCE_MODEL_ID
            )
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
    tracking_prior: Optional[VQ2ApertureTrackingPrior] = None,
) -> VQ2ApertureFit:
    """Threshold a BGR frame and fit its VQ2 inner aperture."""

    mask = vq2_gate_mask_from_bgr(image_bgr, config=config)
    return fit_vq2_aperture_mask(
        mask,
        support_bbox_px,
        detection_confidence=detection_confidence,
        config=config,
        tracking_prior=tracking_prior,
    )


def _complete_geometry_from_vq2_aperture_fit(
    fit: VQ2ApertureFit,
    *,
    allow_outer_support_clipping: bool = False,
    allow_tracking_only_models: bool = False,
) -> Optional[VQ2PassageGeometry]:
    """Return an inscribed opening from an exact complete visible fit.

    Confidence affects uncertainty here, not structural admission.  Public
    passage admission applies its independent confidence floor before calling
    this helper.
    """

    if type(fit) is not VQ2ApertureFit:
        raise TypeError("fit must be an exact VQ2ApertureFit")
    if type(allow_outer_support_clipping) is not bool:
        raise TypeError("allow_outer_support_clipping must be an exact bool")
    if type(allow_tracking_only_models) is not bool:
        raise TypeError("allow_tracking_only_models must be an exact bool")
    fit_confidence = _finite_confidence(fit.confidence)
    model_pair = (
        fit.geometry_model_id,
        fit.covariance_model_id,
    )
    accepted_model_pairs = {
        (
            _VISIBLE_GEOMETRY_MODEL_ID,
            _VISIBLE_COVARIANCE_MODEL_ID,
        ),
    }
    if allow_tracking_only_models:
        accepted_model_pairs.add(
            (
                _TEMPORAL_GEOMETRY_MODEL_ID,
                _TEMPORAL_COVARIANCE_MODEL_ID,
            )
        )
    all_sides = (
        ApertureSide.LEFT
        | ApertureSide.TOP
        | ApertureSide.RIGHT
        | ApertureSide.BOTTOM
    )
    if (
        not fit.succeeded
        or fit.rejection_reason is not None
        or (
            fit.clipping != ApertureSide.NONE
            and not allow_outer_support_clipping
        )
        or fit.visible_edges != all_sides
        or fit.visible_corners != (True, True, True, True)
        or fit.fitted_corners_px is None
        or model_pair not in accepted_model_pairs
        or fit.covariance_diagonal is None
        or fit.residual_rms_px is None
        or not math.isfinite(fit.residual_rms_px)
        or fit.inlier_count <= 0
        or fit.support_count < fit.inlier_count
    ):
        return None

    width, height = fit.image_size_px
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or any(
            not all(math.isfinite(float(value)) for value in point)
            or not 0.0 <= float(point[0]) <= float(width)
            or not 0.0 <= float(point[1]) <= float(height)
            for point in fit.fitted_corners_px
        )
    ):
        return None
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
    # The fit covariance reflects line residuals, but confidence also carries
    # detector support and coverage quality.  Model variance as inversely
    # proportional to confidence below the passage floor so degraded tracking
    # cannot look as precise as a nominal passage measurement.
    confidence_uncertainty_multiplier = math.sqrt(
        _PASSAGE_MINIMUM_CONFIDENCE
        / max(fit_confidence, _CONFIDENCE_UNCERTAINTY_EPSILON)
    )
    confidence_uncertainty_multiplier = max(
        1.0,
        confidence_uncertainty_multiplier,
    )
    measurement_std = tuple(
        value * confidence_uncertainty_multiplier
        for value in measurement_std
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


def tracking_geometry_from_vq2_aperture_fit(
    fit: VQ2ApertureFit,
) -> Optional[VQ2PassageGeometry]:
    """Retain exact complete inner geometry, with confidence in uncertainty.

    A clipped detector support box does not censor a complete fitted inner
    quadrilateral. Incomplete, rejected, or non-production-model fits remain
    unavailable rather than borrowing the detector's outer support geometry.
    """

    return _complete_geometry_from_vq2_aperture_fit(
        fit,
        allow_outer_support_clipping=True,
        allow_tracking_only_models=True,
    )


def passage_geometry_from_vq2_aperture_fit(
    fit: VQ2ApertureFit,
    *,
    minimum_confidence: float = _PASSAGE_MINIMUM_CONFIDENCE,
) -> Optional[VQ2PassageGeometry]:
    """Return a conservative inscribed opening only from a nominal fit.

    Detector support bounds are deliberately excluded.  The configurable
    floor may make admission stricter but can never weaken the retained 0.25
    passage-confidence requirement.
    """

    if type(fit) is not VQ2ApertureFit:
        raise TypeError("fit must be an exact VQ2ApertureFit")
    confidence_floor = max(
        _PASSAGE_MINIMUM_CONFIDENCE,
        _finite_confidence(minimum_confidence),
    )
    if _finite_confidence(fit.confidence) < confidence_floor:
        return None
    return _complete_geometry_from_vq2_aperture_fit(fit)


__all__ = [
    "ApertureSide",
    "VQ2ApertureConfig",
    "VQ2ApertureFit",
    "VQ2ApertureTrackingPrior",
    "VQ2PassageGeometry",
    "fit_vq2_aperture_bgr",
    "fit_vq2_aperture_mask",
    "passage_geometry_from_vq2_aperture_fit",
    "tracking_geometry_from_vq2_aperture_fit",
    "vq2_gate_mask_from_bgr",
]
