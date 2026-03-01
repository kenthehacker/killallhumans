"""
Gate Detection Module for AI Grand Prix — v3 (Color-Agnostic)
=============================================================
Detects racing gates **without requiring a color preset**.

Three parallel strategies, all fused and deduplicated:

A) **Dynamic color clustering** — estimates the background colour,
   finds foreground pixels via Otsu thresholding on colour-distance,
   clusters them with K-means, then runs bar-grouping per cluster.

B) **Edge-based rectangle detection** — bilateral filter → Canny →
   polygon approximation → filter for rectangular openings whose
   border colour is distinct from the estimated background.

C) **(Optional) Preset HSV mode** — the original colour-preset
   pipeline, kept for environments where gate colour is known.

Outputs the same GateDetection dataclass as v2 for ML compatibility.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateDetection:
    """Detected gate with rich features for downstream ML."""

    center_x: int
    center_y: int
    bbox: Tuple[int, int, int, int]
    corners: np.ndarray
    area: int

    estimated_distance: float
    apparent_width_px: float = 0.0
    apparent_height_px: float = 0.0

    rotation_deg: float = 0.0
    aspect_ratio: float = 1.0
    rectangularity: float = 1.0

    normalized_center_x: float = 0.0
    normalized_center_y: float = 0.0
    is_above_center: bool = False
    is_below_center: bool = False
    is_left_of_center: bool = False
    is_right_of_center: bool = False

    confidence: float = 0.0
    detection_method: str = "unknown"
    detected_color_hsv: Optional[Tuple[int, int, int]] = None

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
# HSV presets (optional — only used when caller explicitly sets one)
# ---------------------------------------------------------------------------

HSV_PRESETS: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255])),
    ],
    "blue":       [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
    "orange":     [(np.array([10, 100, 100]), np.array([25, 255, 255]))],
    "green":      [(np.array([40, 100, 100]), np.array([80, 255, 255]))],
    "yellow":     [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
    "purple":     [(np.array([130, 100, 100]), np.array([160, 255, 255]))],
    "tii_purple": [(np.array([110, 20, 40]), np.array([150, 130, 135]))],
}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GateDetector:
    """
    Color-agnostic racing-gate detector.

    Default mode (``color_preset=None``) runs dynamic colour clustering +
    edge-based rectangle detection — **no prior knowledge of gate colour
    is needed**.

    If you *do* know the gate colour, pass ``color_preset="red"`` (etc.)
    to additionally run the classic HSV-threshold pipeline, which can be
    faster and more precise when the colour is known.
    """

    GATE_WIDTH_METERS = 1.0
    GATE_HEIGHT_METERS = 1.0

    def __init__(
        self,
        # --- Colour preset (optional) ---
        color_preset: Optional[str] = None,
        # --- Camera / image params ---
        camera_fov_horizontal: float = 90.0,
        image_width: int = 640,
        image_height: int = 480,
        # --- Strategy toggles ---
        enable_dynamic_clustering: bool = True,
        enable_edge_rects: bool = True,
        enable_preset_mode: bool = False,      # auto-enabled if preset given
        # --- Tuning ---
        n_clusters: int = 4,
        min_bar_area: int = 800,
        bar_min_height_ratio: float = 2.0,
        morph_kernel_size: int = 7,
        min_confidence: float = 0.12,
        edge_min_area: int = 5000,
        # --- Classic-mode params ---
        min_area: int = 500,
        max_area: int = 500_000,
        max_aspect_ratio: float = 3.0,
        max_aspect_ratio_partial: float = 4.5,
    ):
        self.camera_fov = camera_fov_horizontal
        self.image_width = image_width
        self.image_height = image_height

        self.enable_dynamic = enable_dynamic_clustering
        self.enable_edge = enable_edge_rects
        self.n_clusters = n_clusters
        self.min_bar_area = min_bar_area
        self.bar_min_hr = bar_min_height_ratio
        self.morph_k = morph_kernel_size
        self.min_conf = min_confidence
        self.edge_min_area = edge_min_area

        # Classic (preset) mode
        self.min_area = min_area
        self.max_area = max_area
        self.max_ar = max_aspect_ratio
        self.max_ar_partial = max_aspect_ratio_partial

        self.hsv_ranges: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.enable_preset = enable_preset_mode

        if color_preset is not None:
            if color_preset in HSV_PRESETS:
                self.hsv_ranges = HSV_PRESETS[color_preset]
            else:
                print(f"Warning: unknown preset '{color_preset}', ignoring")
            self.enable_preset = True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_custom_thresholds(self, lower: np.ndarray, upper: np.ndarray):
        self.hsv_ranges = [(lower, upper)]
        self.enable_preset = True

    def add_hsv_range(self, lower: np.ndarray, upper: np.ndarray):
        if self.hsv_ranges is None:
            self.hsv_ranges = []
        self.hsv_ranges.append((lower, upper))
        self.enable_preset = True

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[GateDetection]:
        """
        Detect gates in a BGR image. Returns list sorted by confidence.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image.shape[:2]

        raw: List[Dict] = []

        # Strategy A: dynamic clustering + bar grouping
        if self.enable_dynamic:
            raw.extend(self._strategy_dynamic_clustering(hsv, h_img, w_img))

        # Strategy B: edge-based rectangles
        if self.enable_edge:
            raw.extend(self._strategy_edge_rects(gray, hsv, h_img, w_img))

        # Strategy C: preset HSV (optional)
        if self.enable_preset and self.hsv_ranges is not None:
            raw.extend(self._strategy_preset(hsv, h_img, w_img))

        # Filter, deduplicate, build GateDetection objects
        raw = [r for r in raw if r["confidence"] >= self.min_conf]
        raw.sort(key=lambda r: r["confidence"], reverse=True)
        raw = _deduplicate_dicts(raw, iou_thresh=0.35)

        detections = [self._dict_to_detection(r, w_img, h_img) for r in raw]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ==================================================================
    # Strategy A: dynamic clustering + bar grouping
    # ==================================================================
    def _strategy_dynamic_clustering(self, hsv, h_img, w_img) -> List[Dict]:
        bg_h = float(np.median(hsv[:, :, 0]))
        bg_s = float(np.median(hsv[:, :, 1]))
        bg_v = float(np.median(hsv[:, :, 2]))

        # Colour distance from background
        h_diff = np.minimum(
            np.abs(hsv[:, :, 0].astype(float) - bg_h),
            180 - np.abs(hsv[:, :, 0].astype(float) - bg_h),
        )
        s_diff = np.abs(hsv[:, :, 1].astype(float) - bg_s)
        v_diff = np.abs(hsv[:, :, 2].astype(float) - bg_v)
        cd = 2.0 * h_diff + 0.5 * s_diff + 0.5 * v_diff
        cd_u8 = (cd / max(cd.max(), 1) * 255).astype(np.uint8)

        _, fg = cv2.threshold(cd_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = np.ones((self.morph_k, self.morph_k), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        fg_px = hsv[fg > 0]
        fg_rc = np.argwhere(fg > 0)
        if len(fg_px) < 500:
            return []

        n_samp = min(len(fg_px), 8000)
        idx = np.random.choice(len(fg_px), n_samp, replace=False)
        samples = fg_px[idx].astype(np.float32)
        K = min(self.n_clusters, max(2, len(fg_px) // 500))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(samples, K, None, crit, 10, cv2.KMEANS_PP_CENTERS)

        all_fg = fg_px.astype(np.float32)
        dists = np.stack(
            [np.sqrt(np.sum((all_fg - c) ** 2, axis=1)) for c in centers], axis=1
        )
        labels = np.argmin(dists, axis=1)

        results: List[Dict] = []
        for cid in range(K):
            cidx = np.where(labels == cid)[0]
            if len(cidx) < 200:
                continue
            cmask = np.zeros((h_img, w_img), dtype=np.uint8)
            cmask[fg_rc[cidx, 0], fg_rc[cidx, 1]] = 255
            cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, k)
            cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bars = []
            for c in contours:
                a = cv2.contourArea(c)
                if a < self.min_bar_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w > 0 and h / w >= self.bar_min_hr:
                    bars.append((x, y, w, h, a))
            bars.sort(key=lambda b: b[0])

            for i in range(len(bars)):
                for j in range(i + 1, len(bars)):
                    g = _try_pair_bars(bars[i], bars[j])
                    if g is not None:
                        g["method"] = "dynamic_cluster"
                        g["color_hsv"] = tuple(int(v) for v in centers[cid])
                        results.append(g)
        return results

    # ==================================================================
    # Strategy B: edge-based rectangle detection
    # ==================================================================
    def _strategy_edge_rects(self, gray, hsv, h_img, w_img) -> List[Dict]:
        bg_h = float(np.median(hsv[:, :, 0]))
        bg_s = float(np.median(hsv[:, :, 1]))
        bg_v = float(np.median(hsv[:, :, 2]))

        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        canny = cv2.Canny(smooth, 30, 100)
        canny = cv2.dilate(canny, np.ones((5, 5), np.uint8), iterations=2)
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        results: List[Dict] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.edge_min_area or area > 0.7 * h_img * w_img:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = max(w, h) / max(min(w, h), 1)
            if ar > 2.0:
                continue
            rect_fill = area / max(w * h, 1)
            if rect_fill < 0.3:
                continue

            # Sample border colour (20 px thick band along contour)
            bmask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(bmask, [c], -1, 255, thickness=20)
            bpx = hsv[bmask > 0]
            if len(bpx) < 50:
                continue

            bh = float(np.median(bpx[:, 0]))
            bs = float(np.median(bpx[:, 1]))
            bv = float(np.median(bpx[:, 2]))
            hd = min(abs(bh - bg_h), 180 - abs(bh - bg_h))
            bg_dist = 2 * hd + 0.5 * abs(bs - bg_s) + 0.5 * abs(bv - bg_v)
            if bg_dist < 10:
                continue

            bg_dist_score = min(bg_dist / 50.0, 1.0)
            conf = (
                rect_fill
                * (1.0 / max(ar, 1))
                * min(area / 50000.0, 1.0)
                * bg_dist_score
                * 0.8
            )

            results.append({
                "bbox": (x, y, w, h),
                "center": (x + w // 2, y + h // 2),
                "confidence": conf,
                "method": "edge_rect",
                "color_hsv": (int(bh), int(bs), int(bv)),
            })
        return results

    # ==================================================================
    # Strategy C: preset HSV + bar grouping + contour
    # ==================================================================
    def _strategy_preset(self, hsv, h_img, w_img) -> List[Dict]:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        k = np.ones((self.morph_k, self.morph_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results: List[Dict] = []

        # Contour mode
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if min(rw, rh) == 0:
                continue
            ar = max(rw, rh) / min(rw, rh)
            margin = 3
            edge = x < margin or y < margin or (x+w) > (w_img-margin) or (y+h) > (h_img-margin)
            lim = self.max_ar_partial if edge else self.max_ar
            if ar > lim:
                continue
            ra = rw * rh
            if ra == 0:
                continue
            rect_fill = area / ra
            asp_score = 1.0 - min(abs(ar - 1.0) / 2.0, 1.0)
            area_score = min(area / 10000, 1.0)
            conf = 0.6 * asp_score + 0.4 * area_score
            results.append({
                "bbox": (x, y, w, h),
                "center": (x + w // 2, y + h // 2),
                "confidence": conf,
                "method": "preset_contour",
            })

        # Bar-grouping mode
        bars = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_bar_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w > 0 and h / w >= self.bar_min_hr:
                bars.append((x, y, w, h, a))
        bars.sort(key=lambda b: b[0])
        for i in range(len(bars)):
            for j in range(i + 1, len(bars)):
                g = _try_pair_bars(bars[i], bars[j])
                if g is not None:
                    g["method"] = "preset_bar_group"
                    results.append(g)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dict_to_detection(self, d: Dict, img_w: int, img_h: int) -> GateDetection:
        x, y, w, h = d["bbox"]
        cx, cy = d["center"]
        nxn = (2.0 * cx / max(img_w, 1)) - 1.0
        nyn = (2.0 * cy / max(img_h, 1)) - 1.0
        mid_x, mid_y = img_w / 2.0, img_h / 2.0
        aw = float(max(w, h))
        ah = float(min(w, h))
        ar = aw / max(ah, 1)
        dist = self._estimate_distance(aw)

        corners = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
        )

        return GateDetection(
            center_x=cx,
            center_y=cy,
            bbox=(x, y, w, h),
            corners=corners,
            area=w * h,
            estimated_distance=dist,
            apparent_width_px=aw,
            apparent_height_px=ah,
            rotation_deg=0.0,
            aspect_ratio=ar,
            rectangularity=1.0,
            normalized_center_x=nxn,
            normalized_center_y=nyn,
            is_above_center=cy < mid_y,
            is_below_center=cy > mid_y,
            is_left_of_center=cx < mid_x,
            is_right_of_center=cx > mid_x,
            confidence=d["confidence"],
            detection_method=d["method"],
            detected_color_hsv=d.get("color_hsv"),
        )

    def _estimate_distance(self, apparent_px: float) -> float:
        if apparent_px == 0:
            return float("inf")
        fl = self.image_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
        return (self.GATE_WIDTH_METERS * fl) / apparent_px

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def get_debug_visualization(
        self,
        image: np.ndarray,
        detections: List[GateDetection],
        show_rich_info: bool = True,
    ) -> np.ndarray:
        vis = image.copy()
        method_colors = {
            "dynamic_cluster": (0, 255, 0),
            "edge_rect": (255, 255, 0),
            "preset_contour": (0, 200, 255),
            "preset_bar_group": (255, 0, 200),
        }
        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            clr = method_colors.get(det.detection_method, (200, 200, 200))
            cv2.rectangle(vis, (x, y), (x + w, y + h), clr, 2)
            cv2.drawContours(vis, [det.corners], 0, (255, 0, 0), 2)
            cv2.circle(vis, (det.center_x, det.center_y), 5, (0, 0, 255), -1)

            label = f"Gate {i+1}: {det.estimated_distance:.1f}m conf={det.confidence:.2f}"
            cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

            if show_rich_info:
                pos = "above" if det.is_above_center else "below"
                cv2.putText(
                    vis,
                    f"{det.detection_method} | {pos}",
                    (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (200, 200, 200),
                    1,
                )
                cv2.putText(
                    vis,
                    f"size={det.apparent_width_px:.0f}x{det.apparent_height_px:.0f}px",
                    (x, y + h + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (200, 200, 200),
                    1,
                )
                if det.detected_color_hsv:
                    cv2.putText(
                        vis,
                        f"color HSV={det.detected_color_hsv}",
                        (x, y + h + 46),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (200, 200, 200),
                        1,
                    )
        return vis


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _try_pair_bars(bar1, bar2) -> Optional[Dict]:
    """Try to pair two vertical bars into a gate opening."""
    x1, y1, w1, h1, a1 = bar1
    x2, y2, w2, h2, a2 = bar2

    hr = min(h1, h2) / max(h1, h2)
    if hr < 0.5:
        return None
    avg_h = (h1 + h2) / 2
    if abs(y1 - y2) > avg_h * 0.4:
        return None
    gap = x2 - (x1 + w1)
    if gap < avg_h * 0.25 or gap > avg_h * 2.5:
        return None

    gx = x1
    gy = min(y1, y2)
    gw = (x2 + w2) - x1
    gh = max(y1 + h1, y2 + h2) - gy
    gar = max(gw, gh) / max(min(gw, gh), 1)
    if gar > 2.5:
        return None

    conf = hr * (1.0 / max(gar, 1)) * min((a1 + a2) / 15000.0, 1.0)
    return {
        "bbox": (gx, gy, gw, gh),
        "center": (gx + gw // 2, gy + gh // 2),
        "confidence": conf,
    }


def _bbox_iou(a, b) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1)


def _deduplicate_dicts(items: List[Dict], iou_thresh=0.35) -> List[Dict]:
    keep = []
    for r in items:
        if not any(_bbox_iou(r["bbox"], k["bbox"]) > iou_thresh for k in keep):
            keep.append(r)
    return keep


def pixel_to_normalized(x, y, width, height):
    return (2 * x / width) - 1, (2 * y / height) - 1


def get_steering_error(detection: GateDetection, image_width, image_height):
    return pixel_to_normalized(
        detection.center_x, detection.center_y, image_width, image_height
    )
