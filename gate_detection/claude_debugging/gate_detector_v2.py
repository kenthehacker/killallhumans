"""
Gate Detection Module for AI Grand Prix — v2
=============================================
Supports two detection strategies:

1. **Contour mode** (original) — HSV threshold → contour → shape filter.
   Works well for brightly colored, fully visible gates.

2. **Bar-grouping mode** (new) — HSV threshold → find individual frame bars
   → pair vertical bars that could form a gate opening.
   Designed for the TII drone-racing gates (dark purple frame, low saturation).

Both modes run by default; results are merged and deduplicated.

Outputs rich per-gate features for ML (path planning, flight control).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateDetection:
    """Represents a detected gate in the image with rich features for ML."""

    # --- Core pixel geometry ---
    center_x: int
    center_y: int
    bbox: Tuple[int, int, int, int]          # (x, y, width, height)
    corners: np.ndarray                       # Shape: (4, 2), clockwise from TL
    area: int                                 # contour area in pixels

    # --- Distance & size ---
    estimated_distance: float                 # metres (pinhole model)
    apparent_width_px: float = 0.0
    apparent_height_px: float = 0.0

    # --- Viewing angle / orientation ---
    rotation_deg: float = 0.0
    aspect_ratio: float = 1.0
    rectangularity: float = 1.0

    # --- Position relative to frame ---
    normalized_center_x: float = 0.0          # -1 … +1
    normalized_center_y: float = 0.0
    is_above_center: bool = False
    is_below_center: bool = False
    is_left_of_center: bool = False
    is_right_of_center: bool = False

    # --- Quality ---
    confidence: float = 0.0
    detection_method: str = "contour"         # "contour" | "bar_group"

    def to_ml_features(self, image_width: int = 640, image_height: int = 480) -> Dict[str, Any]:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "normalized_center_x": self.normalized_center_x,
            "normalized_center_y": self.normalized_center_y,
            "is_above_center": float(self.is_above_center),
            "is_below_center": float(self.is_below_center),
            "is_left_of_center": float(self.is_left_of_center),
            "is_right_of_center": float(self.is_right_of_center),
            "bbox_x": self.bbox[0] / max(image_width, 1),
            "bbox_y": self.bbox[1] / max(image_height, 1),
            "bbox_w": self.bbox[2] / max(image_width, 1),
            "bbox_h": self.bbox[3] / max(image_height, 1),
            "area": self.area,
            "area_normalized": self.area / max(image_width * image_height, 1),
            "apparent_width_px": self.apparent_width_px,
            "apparent_height_px": self.apparent_height_px,
            "aspect_ratio": self.aspect_ratio,
            "rectangularity": self.rectangularity,
            "rotation_deg": self.rotation_deg,
            "estimated_distance": self.estimated_distance,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
        }


# ---------------------------------------------------------------------------
# HSV presets — including TII-specific ranges
# ---------------------------------------------------------------------------

# Format: dict[str, list[tuple[np.ndarray, np.ndarray]]]
# Each preset is a *list* of (lower, upper) pairs so we can cover
# non-contiguous hue ranges (e.g. red wraps around 0/180).

HSV_PRESETS: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    # --- Bright / saturated gates (original presets) ---
    "red": [
        (np.array([0, 100, 100]),   np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255])),
    ],
    "blue":   [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
    "orange": [(np.array([10, 100, 100]),  np.array([25, 255, 255]))],
    "green":  [(np.array([40, 100, 100]),  np.array([80, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]),  np.array([40, 255, 255]))],
    "purple": [(np.array([130, 100, 100]), np.array([160, 255, 255]))],

    # --- TII drone-racing gates (dark purple, low saturation) ---
    # Measured from real TII dataset frames:
    #   H ≈ 115-145,  S ≈ 20-120,  V ≈ 40-130
    "tii_purple": [
        (np.array([110, 20, 40]), np.array([150, 130, 135])),
    ],

    # Broader variant — catches more gate pixels but also more noise.
    # Use with bar-grouping mode for best results.
    "tii_purple_wide": [
        (np.array([105, 15, 35]), np.array([155, 140, 150])),
    ],
}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GateDetector:
    """
    Detects racing gates using colour-based segmentation.

    Two detection strategies run in parallel:

    **Contour mode** (original)
      HSV mask → morphological cleanup → find external contours →
      filter by area & aspect ratio → return detections.
      Best for: brightly coloured gates that form a single large contour.

    **Bar-grouping mode** (new for TII gates)
      HSV mask → find individual bar-like contours (tall/narrow or
      wide/short) → pair vertical bars whose heights, y-positions,
      and spacing are consistent with a gate opening.
      Best for: dark or low-saturation gate frames where the
      opening prevents a single contour from forming.
    """

    GATE_WIDTH_METERS  = 1.0   # placeholder – update when specs are known
    GATE_HEIGHT_METERS = 1.0

    def __init__(
        self,
        color_preset: str = "red",
        min_area: int = 500,
        max_area: int = 500_000,
        camera_fov_horizontal: float = 90.0,
        image_width: int = 640,
        image_height: int = 480,
        max_aspect_ratio: float = 3.0,
        max_aspect_ratio_partial: float = 4.5,
        # --- New options ---
        enable_contour_mode: bool = True,
        enable_bar_grouping: bool = True,
        bar_min_area: int = 600,
        bar_min_height_ratio: float = 2.0,
        morph_kernel_size: int = 5,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.camera_fov = camera_fov_horizontal
        self.image_width = image_width
        self.image_height = image_height
        self.max_aspect_ratio = max_aspect_ratio
        self.max_aspect_ratio_partial = max_aspect_ratio_partial

        self.enable_contour_mode = enable_contour_mode
        self.enable_bar_grouping = enable_bar_grouping
        self.bar_min_area = bar_min_area
        self.bar_min_height_ratio = bar_min_height_ratio
        self.morph_kernel_size = morph_kernel_size

        # Resolve HSV ranges
        if color_preset in HSV_PRESETS:
            self.hsv_ranges = HSV_PRESETS[color_preset]
        else:
            print(f"Warning: unknown preset '{color_preset}', falling back to 'red'")
            self.hsv_ranges = HSV_PRESETS["red"]

    # ------------------------------------------------------------------
    # Public helpers to override thresholds at runtime
    # ------------------------------------------------------------------
    def set_custom_thresholds(self, lower: np.ndarray, upper: np.ndarray):
        """Replace all HSV ranges with a single custom range."""
        self.hsv_ranges = [(lower, upper)]

    def add_hsv_range(self, lower: np.ndarray, upper: np.ndarray):
        """Add an additional HSV range (e.g. for a second hue band)."""
        self.hsv_ranges.append((lower, upper))

    # ------------------------------------------------------------------
    # Core mask building
    # ------------------------------------------------------------------
    def _build_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Union of all HSV ranges, followed by morphological cleanup."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        k = self.morph_kernel_size
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[GateDetection]:
        """
        Detect gates in a BGR image.

        Returns list of GateDetection sorted by area (largest first).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[GateDetection] = []

        if self.enable_contour_mode:
            detections.extend(self._detect_contour_mode(contours, image.shape))

        if self.enable_bar_grouping:
            detections.extend(self._detect_bar_grouping(contours, image.shape))

        # Deduplicate: if two detections overlap significantly keep higher-conf
        detections = self._deduplicate(detections)

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Strategy 1: original contour-based detection
    # ------------------------------------------------------------------
    def _detect_contour_mode(
        self, contours: List[np.ndarray], image_shape: Tuple
    ) -> List[GateDetection]:
        results = []
        for contour in contours:
            det = self._process_contour(contour, image_shape)
            if det is not None:
                results.append(det)
        return results

    def _process_contour(
        self, contour: np.ndarray, image_shape: Tuple
    ) -> Optional[GateDetection]:
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = np.int32(box)

        rect_w, rect_h = rect[1]
        if min(rect_w, rect_h) == 0:
            return None
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)

        img_h, img_w = image_shape[:2]
        margin = 3
        touches_edge = (
            x < margin or y < margin
            or (x + w) > (img_w - margin)
            or (y + h) > (img_h - margin)
        )
        max_ar = self.max_aspect_ratio_partial if touches_edge else self.max_aspect_ratio
        if aspect_ratio > max_ar:
            return None

        rect_area = rect_w * rect_h
        if rect_area == 0:
            return None
        rectangularity = area / rect_area

        return self._build_detection(
            x, y, w, h, corners, area, aspect_ratio, rectangularity,
            max(rect_w, rect_h), min(rect_w, rect_h),
            float(rect[2]), img_w, img_h, method="contour",
        )

    # ------------------------------------------------------------------
    # Strategy 2: bar-grouping for hollow-frame gates
    # ------------------------------------------------------------------
    def _detect_bar_grouping(
        self, contours: List[np.ndarray], image_shape: Tuple
    ) -> List[GateDetection]:
        """
        Find individual vertical bars in the mask, then pair them
        into plausible gate openings.
        """
        img_h, img_w = image_shape[:2]
        vertical_bars: List[Tuple[int, int, int, int, float]] = []   # x,y,w,h,area

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.bar_min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0:
                continue
            if h / w >= self.bar_min_height_ratio:
                vertical_bars.append((x, y, w, h, area))

        # Sort left→right
        vertical_bars.sort(key=lambda b: b[0])

        results: List[GateDetection] = []

        for i in range(len(vertical_bars)):
            for j in range(i + 1, len(vertical_bars)):
                x1, y1, w1, h1, a1 = vertical_bars[i]
                x2, y2, w2, h2, a2 = vertical_bars[j]

                # Height similarity (within 50 %)
                height_ratio = min(h1, h2) / max(h1, h2)
                if height_ratio < 0.50:
                    continue

                # Top-alignment: tops should be within 40 % of avg height
                avg_h = (h1 + h2) / 2
                if abs(y1 - y2) > avg_h * 0.40:
                    continue

                # Horizontal gap between inner edges of bars
                gap = x2 - (x1 + w1)
                if gap < avg_h * 0.25:       # too close / overlapping
                    continue
                if gap > avg_h * 2.5:         # too far apart
                    continue

                # Build gate bounding box
                gx = x1
                gy = min(y1, y2)
                gw = (x2 + w2) - x1
                gh = max(y1 + h1, y2 + h2) - gy

                # Gate aspect ratio check (should be roughly square-ish)
                gate_ar = max(gw, gh) / max(min(gw, gh), 1)
                if gate_ar > 2.5:
                    continue

                gcx = gx + gw // 2
                gcy = gy + gh // 2

                corners = np.array([
                    [gx, gy], [gx + gw, gy],
                    [gx + gw, gy + gh], [gx, gy + gh]
                ], dtype=np.int32)

                gate_area = gw * gh
                conf = self._bar_group_confidence(
                    height_ratio, gate_ar, a1, a2, gate_area
                )

                det = self._build_detection(
                    gx, gy, gw, gh, corners, gate_area,
                    gate_ar, height_ratio,
                    float(max(gw, gh)), float(min(gw, gh)),
                    0.0, img_w, img_h,
                    method="bar_group", confidence_override=conf,
                )
                results.append(det)

        return results

    @staticmethod
    def _bar_group_confidence(
        height_ratio: float,
        gate_ar: float,
        bar_area_1: float,
        bar_area_2: float,
        gate_area: float,
    ) -> float:
        """Heuristic confidence for a bar-grouped gate."""
        # Prefer bars of similar height
        h_score = height_ratio  # 0…1, 1 = identical

        # Prefer gate close to square
        ar_score = 1.0 - min(abs(gate_ar - 1.0) / 2.0, 1.0)

        # Prefer larger bars (more pixels = more reliable colour match)
        bar_area = bar_area_1 + bar_area_2
        size_score = min(bar_area / 15000.0, 1.0)

        return 0.35 * h_score + 0.35 * ar_score + 0.30 * size_score

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _build_detection(
        self, x, y, w, h, corners, area, aspect_ratio, rectangularity,
        apparent_w, apparent_h, rotation_deg,
        img_w, img_h, method="contour", confidence_override=None,
    ) -> GateDetection:
        cx = x + w // 2
        cy = y + h // 2
        norm_x = (2.0 * cx / max(img_w, 1)) - 1.0
        norm_y = (2.0 * cy / max(img_h, 1)) - 1.0
        mid_x = img_w / 2.0
        mid_y = img_h / 2.0

        dist = self._estimate_distance(max(apparent_w, apparent_h))
        conf = confidence_override if confidence_override is not None else \
            self._calculate_confidence(aspect_ratio, rectangularity, area)

        return GateDetection(
            center_x=cx, center_y=cy,
            bbox=(x, y, w, h), corners=corners, area=int(area),
            estimated_distance=dist,
            apparent_width_px=apparent_w, apparent_height_px=apparent_h,
            rotation_deg=rotation_deg,
            aspect_ratio=aspect_ratio, rectangularity=rectangularity,
            normalized_center_x=norm_x, normalized_center_y=norm_y,
            is_above_center=cy < mid_y, is_below_center=cy > mid_y,
            is_left_of_center=cx < mid_x, is_right_of_center=cx > mid_x,
            confidence=conf, detection_method=method,
        )

    def _estimate_distance(self, apparent_size_pixels: float) -> float:
        if apparent_size_pixels == 0:
            return float("inf")
        focal_length = self.image_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
        return (self.GATE_WIDTH_METERS * focal_length) / apparent_size_pixels

    def _calculate_confidence(self, aspect_ratio, rectangularity, area):
        aspect_score = 1.0 - min(abs(aspect_ratio - 1.0) / 2.0, 1.0)
        area_score = min(area / 10000, 1.0)
        return 0.6 * aspect_score + 0.4 * area_score

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate(detections: List[GateDetection], iou_threshold=0.35) -> List[GateDetection]:
        """Remove overlapping detections, keeping higher confidence."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence desc
        detections.sort(key=lambda d: d.confidence, reverse=True)
        keep = []

        for det in detections:
            dominated = False
            for kept in keep:
                if _bbox_iou(det.bbox, kept.bbox) > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                keep.append(det)
        return keep

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def get_debug_visualization(
        self, image: np.ndarray, detections: List[GateDetection],
        show_rich_info: bool = True,
    ) -> np.ndarray:
        vis = image.copy()
        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            color = (0, 255, 0) if det.confidence > 0.4 else (0, 255, 255)
            # Yellow bounding box, blue minAreaRect corners
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.drawContours(vis, [det.corners], 0, (255, 0, 0), 2)
            cv2.circle(vis, (det.center_x, det.center_y), 5, (0, 0, 255), -1)

            label = (
                f"Gate {i+1}: {det.estimated_distance:.1f}m "
                f"conf={det.confidence:.2f}"
            )
            cv2.putText(vis, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if show_rich_info:
                pos = "above" if det.is_above_center else "below"
                cv2.putText(
                    vis, f"angle={det.rotation_deg:.0f} deg {pos}",
                    (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1,
                )
                cv2.putText(
                    vis,
                    f"size={det.apparent_width_px:.0f}x{det.apparent_height_px:.0f}px",
                    (x, y + h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1,
                )
        return vis

    def get_mask_visualization(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return self._build_mask(hsv)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _bbox_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Intersection-over-union for (x, y, w, h) bounding boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0]+a[2], a[1]+a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0]+b[2], b[1]+b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return inter / max(union, 1)


def pixel_to_normalized(x, y, width, height):
    return (2 * x / width) - 1, (2 * y / height) - 1


def get_steering_error(detection: GateDetection, image_width, image_height):
    return pixel_to_normalized(
        detection.center_x, detection.center_y, image_width, image_height
    )
