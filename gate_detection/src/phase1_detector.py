"""
Phase 1 Gate Detector — optimized for Virtual Qualifier 1.

VQ1 environment characteristics (from competition email):
  - Desaturated (near-grayscale) environment
  - Gates are highlighted (high saturation or brightness)
  - High signal-to-noise ratio
  - Visual guidance aids may be active

Strategy: exploit the fact that gates are the only saturated/bright objects
in an otherwise gray scene. Much faster and more reliable than the full
classical pipeline for this specific scenario.

Detection pipeline:
  1. Convert to HSV
  2. Create a saturation mask (high-S pixels in a low-S scene)
  3. Also create a brightness anomaly mask (very bright in a dim scene)
  4. Combine masks
  5. Morphological cleanup
  6. Find contours → filter by area, aspect ratio → gate candidates
  7. Return GateDetection objects (same interface as GateDetector)

Corner detection (for PnP pose estimation):
  - After finding gate contours, fit a quadrilateral via approxPolyDP
  - Fall back to minAreaRect → boxPoints if approxPolyDP doesn't yield 4 pts
  - Order corners consistently: TL, TR, BR, BL
  - detect_with_corners() returns detections with proper fitted corners
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


# Import GateDetection for output compatibility
try:
    from gate_detector import GateDetection
except ImportError:
    from gate_detection.src.gate_detector import GateDetection


class Phase1GateDetector:
    """
    Simplified gate detector for Phase 1 (highlighted gates, desaturated env).

    Much faster than the full classical pipeline (~2-5ms vs ~30-50ms)
    because the detection problem is much easier: find the bright/saturated
    objects in an otherwise gray scene.
    """

    GATE_WIDTH_METERS = 1.0
    GATE_HEIGHT_METERS = 1.0

    def __init__(
        self,
        saturation_threshold: int = 60,
        brightness_threshold: int = 200,
        min_area: int = 500,
        max_area: int = 300_000,
        max_aspect_ratio: float = 3.0,
        min_confidence: float = 0.3,
        camera_fov_horizontal: float = 90.0,
        image_width: int = 640,
        image_height: int = 480,
    ):
        self.sat_thresh = saturation_threshold
        self.bright_thresh = brightness_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_ar = max_aspect_ratio
        self.min_conf = min_confidence
        self.camera_fov = camera_fov_horizontal
        self.image_width = image_width
        self.image_height = image_height

    def detect(self, image: np.ndarray) -> List[GateDetection]:
        """Detect highlighted gates in a desaturated environment."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_img, w_img = image.shape[:2]

        # Saturation mask: highlighted objects have high saturation
        sat_channel = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(
            sat_channel, self.sat_thresh, 255, cv2.THRESH_BINARY
        )

        # Brightness mask: highlighted objects are brighter than background
        val_channel = hsv[:, :, 2]
        _, bright_mask = cv2.threshold(
            val_channel, self.bright_thresh, 255, cv2.THRESH_BINARY
        )

        # Combine: either high saturation OR very bright
        combined = cv2.bitwise_or(sat_mask, bright_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ar = max(w, h) / max(min(w, h), 1)
            if ar > self.max_ar:
                continue

            # Confidence: based on how much of the bbox is filled + saturation strength
            bbox_area = w * h
            fill_ratio = area / max(bbox_area, 1)

            roi_sat = sat_channel[y:y+h, x:x+w]
            mean_sat = float(np.mean(roi_sat)) / 255.0

            confidence = 0.5 * fill_ratio + 0.5 * mean_sat
            if confidence < self.min_conf:
                continue

            # Detect dominant hue for color info
            roi_hue = hsv[y:y+h, x:x+w, 0]
            mask_roi = combined[y:y+h, x:x+w]
            if np.sum(mask_roi > 0) > 0:
                dominant_hue = int(np.median(roi_hue[mask_roi > 0]))
            else:
                dominant_hue = int(np.median(roi_hue))
            dominant_sat = int(np.mean(roi_sat))
            dominant_val = int(np.mean(val_channel[y:y+h, x:x+w]))

            cx, cy = x + w // 2, y + h // 2
            nxn = (2.0 * cx / max(w_img, 1)) - 1.0
            nyn = (2.0 * cy / max(h_img, 1)) - 1.0
            mid_x, mid_y = w_img / 2.0, h_img / 2.0

            apparent_w = float(max(w, h))
            dist = self._estimate_distance(apparent_w)

            corners = np.array(
                [[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32
            )

            detections.append(GateDetection(
                center_x=cx,
                center_y=cy,
                bbox=(x, y, w, h),
                corners=corners,
                area=area,
                estimated_distance=dist,
                apparent_width_px=float(w),
                apparent_height_px=float(h),
                rotation_deg=0.0,
                aspect_ratio=ar,
                rectangularity=fill_ratio,
                normalized_center_x=nxn,
                normalized_center_y=nyn,
                is_above_center=cy < mid_y,
                is_below_center=cy > mid_y,
                is_left_of_center=cx < mid_x,
                is_right_of_center=cx > mid_x,
                confidence=confidence,
                detection_method="phase1_highlight",
                detected_color_hsv=(dominant_hue, dominant_sat, dominant_val),
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = self._nms(detections)
        return detections

    def detect_with_corners(self, image: np.ndarray) -> List[GateDetection]:
        """
        Detect gates and fit proper quadrilateral corners for PnP estimation.

        Like ``detect()`` but replaces the axis-aligned bbox corners with
        actual fitted corners from the contour shape.  This enables accurate
        PnP pose estimation from Phase1 detections.

        Corner fitting strategy:
          1. ``cv2.approxPolyDP`` with adaptive epsilon to find a polygon.
          2. If the polygon has exactly 4 vertices, use them directly.
          3. Otherwise fall back to ``cv2.minAreaRect`` → ``cv2.boxPoints``.
          4. Corners are ordered: top-left, top-right, bottom-right, bottom-left.

        Returns:
            List of GateDetection with ``corners`` set to fitted (4,2) arrays.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_img, w_img = image.shape[:2]

        sat_channel = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(
            sat_channel, self.sat_thresh, 255, cv2.THRESH_BINARY
        )
        val_channel = hsv[:, :, 2]
        _, bright_mask = cv2.threshold(
            val_channel, self.bright_thresh, 255, cv2.THRESH_BINARY
        )
        combined = cv2.bitwise_or(sat_mask, bright_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ar = max(w, h) / max(min(w, h), 1)
            if ar > self.max_ar:
                continue

            bbox_area = w * h
            fill_ratio = area / max(bbox_area, 1)

            roi_sat = sat_channel[y:y+h, x:x+w]
            mean_sat = float(np.mean(roi_sat)) / 255.0
            confidence = 0.5 * fill_ratio + 0.5 * mean_sat
            if confidence < self.min_conf:
                continue

            # Fit corners from the contour
            corners = self._fit_corners(cnt, (h_img, w_img))
            if corners is None:
                # Fall back to bbox corners if fitting fails entirely
                corners = np.array(
                    [[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float64
                )

            # Color info
            roi_hue = hsv[y:y+h, x:x+w, 0]
            mask_roi = combined[y:y+h, x:x+w]
            if np.sum(mask_roi > 0) > 0:
                dominant_hue = int(np.median(roi_hue[mask_roi > 0]))
            else:
                dominant_hue = int(np.median(roi_hue))
            dominant_sat = int(np.mean(roi_sat))
            dominant_val = int(np.mean(val_channel[y:y+h, x:x+w]))

            cx, cy = x + w // 2, y + h // 2
            nxn = (2.0 * cx / max(w_img, 1)) - 1.0
            nyn = (2.0 * cy / max(h_img, 1)) - 1.0
            mid_x, mid_y = w_img / 2.0, h_img / 2.0

            apparent_w = float(max(w, h))
            dist = self._estimate_distance(apparent_w)

            detections.append(GateDetection(
                center_x=cx,
                center_y=cy,
                bbox=(x, y, w, h),
                corners=corners.astype(np.int32),
                area=area,
                estimated_distance=dist,
                apparent_width_px=float(w),
                apparent_height_px=float(h),
                rotation_deg=0.0,
                aspect_ratio=ar,
                rectangularity=fill_ratio,
                normalized_center_x=nxn,
                normalized_center_y=nyn,
                is_above_center=cy < mid_y,
                is_below_center=cy > mid_y,
                is_left_of_center=cx < mid_x,
                is_right_of_center=cx > mid_x,
                confidence=confidence,
                detection_method="phase1_corners",
                detected_color_hsv=(dominant_hue, dominant_sat, dominant_val),
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = self._nms(detections)
        return detections

    def _fit_corners(
        self,
        contour: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Fit a quadrilateral to a gate contour and return ordered corners.

        Strategy:
          1. Try approxPolyDP with epsilon sweep (2%-6% of perimeter).
          2. If any epsilon yields exactly 4 vertices, use them.
          3. Otherwise use minAreaRect → boxPoints.
          4. Order as [TL, TR, BR, BL] and validate within image bounds.

        Args:
            contour: OpenCV contour array.
            image_shape: (height, width) for bounds validation.

        Returns:
            (4, 2) float64 corner array, or None if fitting fails.
        """
        peri = cv2.arcLength(contour, True)
        if peri < 10:
            return None

        # Try several epsilon values to find one that gives exactly 4 points
        for eps_frac in (0.03, 0.04, 0.05, 0.02, 0.06):
            approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float64)
                corners = _order_corners(corners)
                if self._corners_in_bounds(corners, image_shape):
                    return corners

        # Fall back to minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = _order_corners(box.astype(np.float64))
        if self._corners_in_bounds(corners, image_shape):
            return corners

        return None

    @staticmethod
    def _corners_in_bounds(
        corners: np.ndarray, image_shape: Tuple[int, int]
    ) -> bool:
        """Check all corners are within the image boundaries."""
        h, w = image_shape[:2]
        return bool(
            np.all(corners[:, 0] >= 0)
            and np.all(corners[:, 0] < w)
            and np.all(corners[:, 1] >= 0)
            and np.all(corners[:, 1] < h)
        )

    def _estimate_distance(self, apparent_px: float) -> float:
        if apparent_px == 0:
            return float("inf")
        fl = self.image_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
        return (self.GATE_WIDTH_METERS * fl) / apparent_px

    def _nms(self, detections: List[GateDetection], iou_thresh: float = 0.4) -> List[GateDetection]:
        keep = []
        for det in detections:
            if all(self._iou(det.bbox, k.bbox) < iou_thresh for k in keep):
                keep.append(det)
        return keep

    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1e-6)

    def get_debug_visualization(
        self,
        image: np.ndarray,
        detections: List[GateDetection],
        show_rich_info: bool = True,
    ) -> np.ndarray:
        """Draw detection boxes on the image."""
        vis = image.copy()
        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (det.center_x, det.center_y), 5, (0, 0, 255), -1)
            label = f"Gate {i+1}: {det.estimated_distance:.1f}m conf={det.confidence:.2f}"
            cv2.putText(
                vis, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
            if show_rich_info and det.detected_color_hsv:
                h_val, s_val, v_val = det.detected_color_hsv
                cv2.putText(
                    vis, f"HSV: {h_val},{s_val},{v_val}",
                    (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                )
        return vis

    def get_mask_visualization(self, image: np.ndarray) -> np.ndarray:
        """Show the saturation + brightness mask used for detection."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, sat_mask = cv2.threshold(hsv[:, :, 1], self.sat_thresh, 255, cv2.THRESH_BINARY)
        _, bright_mask = cv2.threshold(hsv[:, :, 2], self.bright_thresh, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_or(sat_mask, bright_mask)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses the sum/difference heuristic:
      - top-left has the smallest x+y sum
      - bottom-right has the largest x+y sum
      - top-right has the smallest x-y difference
      - bottom-left has the largest x-y difference
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).squeeze()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float64)
