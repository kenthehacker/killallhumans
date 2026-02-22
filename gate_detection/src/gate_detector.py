"""
Gate Detection Module for AI Grand Prix
========================================
Uses classical computer vision (HSV thresholding + contour detection)
to detect racing gates from camera images.

Outputs rich per-gate features for ML (path planning, flight control).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class GateDetection:
    """Represents a detected gate in the image with rich features for ML."""

    # --- Core pixel geometry ---
    center_x: int
    center_y: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    corners: np.ndarray  # Shape: (4, 2), clockwise from top-left
    area: int  # contour area in pixels

    # --- Distance & size ---
    estimated_distance: float  # meters (pinhole model)
    apparent_width_px: float = 0.0   # width of min-area rect (pixels)
    apparent_height_px: float = 0.0  # height of min-area rect (pixels)

    # --- Viewing angle / orientation ---
    rotation_deg: float = 0.0   # in-plane rotation of gate (0 = axis-aligned)
    aspect_ratio: float = 1.0   # width/height of min-area rect (perspective cue)
    rectangularity: float = 1.0 # area / (width*height), shape quality

    # --- Position relative to frame (for above/below, left/right) ---
    normalized_center_x: float = 0.0  # -1 (left) to +1 (right), 0 = center
    normalized_center_y: float = 0.0  # -1 (above) to +1 (below), 0 = center
    is_above_center: bool = False
    is_below_center: bool = False
    is_left_of_center: bool = False
    is_right_of_center: bool = False

    # --- Quality ---
    confidence: float = 0.0  # 0-1, gate-like score

    def to_ml_features(self, image_width: int = 640, image_height: int = 480) -> Dict[str, Any]:
        """
        Flat feature dict for ML (path planning, flight path generation).
        All numeric values are normalized or in consistent units.
        """
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
        }


class GateDetector:
    """
    Detects racing gates using color-based segmentation.
    
    The approach:
    1. Convert image to HSV color space
    2. Threshold to isolate gate color (e.g., red, blue, orange)
    3. Find contours in the mask
    4. Filter contours by shape (looking for rectangles)
    5. Estimate gate pose/distance
    
    This is a starting point - you'll need to tune parameters
    based on the actual gate colors in the DCL simulator.
    """
    
    # Known gate dimensions (meters) - UPDATE THESE when DCL releases specs
    GATE_WIDTH_METERS = 1.0  # Placeholder
    GATE_HEIGHT_METERS = 1.0  # Placeholder
    
    def __init__(
        self,
        color_preset: str = "red",
        min_area: int = 500,
        max_area: int = 500000,
        camera_fov_horizontal: float = 90.0,  # degrees
        image_width: int = 640,
        image_height: int = 480,
    ):
        """
        Initialize the gate detector.
        
        Args:
            color_preset: One of "red", "blue", "orange", "green", or "custom"
            min_area: Minimum contour area to consider (filters noise)
            max_area: Maximum contour area to consider
            camera_fov_horizontal: Camera field of view in degrees
            image_width: Expected image width
            image_height: Expected image height
        """
        self.min_area = min_area
        self.max_area = max_area
        self.camera_fov = camera_fov_horizontal
        self.image_width = image_width
        self.image_height = image_height
        
        # Set HSV thresholds based on color preset
        self.hsv_lower, self.hsv_upper = self._get_color_thresholds(color_preset)
        
        # For red, we need two ranges (red wraps around in HSV)
        self.is_red = color_preset == "red"
        if self.is_red:
            self.hsv_lower2 = np.array([170, 100, 100])
            self.hsv_upper2 = np.array([180, 255, 255])
    
    def _get_color_thresholds(self, preset: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get HSV thresholds for common gate colors."""
        
        presets = {
            # (lower_hsv, upper_hsv)
            # H: 0-180, S: 0-255, V: 0-255 in OpenCV
            
            "red": (np.array([0, 100, 100]), np.array([10, 255, 255])),
            "blue": (np.array([100, 100, 100]), np.array([130, 255, 255])),
            "orange": (np.array([10, 100, 100]), np.array([25, 255, 255])),
            "green": (np.array([40, 100, 100]), np.array([80, 255, 255])),
            "yellow": (np.array([20, 100, 100]), np.array([40, 255, 255])),
            "purple": (np.array([130, 100, 100]), np.array([160, 255, 255])),
        }
        
        if preset not in presets:
            print(f"Warning: Unknown preset '{preset}', defaulting to red")
            preset = "red"
        
        return presets[preset]
    
    def set_custom_thresholds(self, lower: np.ndarray, upper: np.ndarray):
        """Set custom HSV thresholds."""
        self.hsv_lower = lower
        self.hsv_upper = upper
        self.is_red = False
    
    def detect(self, image: np.ndarray) -> List[GateDetection]:
        """
        Detect gates in an image.
        
        Args:
            image: BGR image from camera (OpenCV format)
            
        Returns:
            List of GateDetection objects, sorted by area (largest first)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Handle red (wraps around HSV spectrum)
        if self.is_red:
            mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
            mask = cv2.bitwise_or(mask, mask2)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        detections = []
        for contour in contours:
            detection = self._process_contour(contour, image.shape)
            if detection is not None:
                detections.append(detection)
        
        # Sort by area (largest first - closest gate)
        detections.sort(key=lambda d: d.area, reverse=True)
        
        return detections
    
    def _process_contour(self, contour: np.ndarray, image_shape: Tuple) -> Optional[GateDetection]:
        """Process a single contour and return a GateDetection if valid."""
        
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < self.min_area or area > self.max_area:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get minimum area rectangle (handles rotated gates)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = np.int32(box)
        
        # Min-area rect: (center, (width, height), angle in degrees)
        rect_angle_deg = rect[2]  # OpenCV: angle of the form [-90, 0)
        rect_w, rect_h = rect[1]
        if rect_h == 0:
            return None
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)

        # Gates are roughly square - filter extreme aspect ratios
        if aspect_ratio > 3.0:
            return None

        rect_area = rect_w * rect_h
        if rect_area == 0:
            return None
        rectangularity = area / rect_area

        center_x = x + w // 2
        center_y = y + h // 2

        # Image dimensions for normalized position
        img_h, img_w = image_shape[:2]
        norm_x = (2.0 * center_x / max(img_w, 1)) - 1.0
        norm_y = (2.0 * center_y / max(img_h, 1)) - 1.0
        mid_y = img_h / 2.0
        mid_x = img_w / 2.0

        estimated_distance = self._estimate_distance(max(rect_w, rect_h))
        confidence = self._calculate_confidence(aspect_ratio, rectangularity, area)

        return GateDetection(
            center_x=center_x,
            center_y=center_y,
            bbox=(x, y, w, h),
            corners=corners,
            area=int(area),
            estimated_distance=estimated_distance,
            apparent_width_px=max(rect_w, rect_h),
            apparent_height_px=min(rect_w, rect_h),
            rotation_deg=float(rect_angle_deg),
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity,
            normalized_center_x=norm_x,
            normalized_center_y=norm_y,
            is_above_center=center_y < mid_y,
            is_below_center=center_y > mid_y,
            is_left_of_center=center_x < mid_x,
            is_right_of_center=center_x > mid_x,
            confidence=confidence,
        )
    
    def _estimate_distance(self, apparent_size_pixels: float) -> float:
        """
        Estimate distance to gate based on apparent size.
        
        Uses simple pinhole camera model:
        distance = (real_size * focal_length) / apparent_size
        
        This is approximate - for better accuracy, use PnP with known corners.
        """
        if apparent_size_pixels == 0:
            return float('inf')
        
        # Focal length in pixels (approximate from FOV)
        focal_length = self.image_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
        
        # Estimate distance
        distance = (self.GATE_WIDTH_METERS * focal_length) / apparent_size_pixels
        
        return distance
    
    def _calculate_confidence(self, aspect_ratio: float, rectangularity: float, area: int) -> float:
        """Calculate detection confidence based on shape metrics."""
        
        # Ideal gate is square (aspect ratio = 1)
        aspect_score = 1.0 - min(abs(aspect_ratio - 1.0) / 2.0, 1.0)
        
        # Larger detections are more reliable
        area_score = min(area / 10000, 1.0)
        
        # Combine scores
        confidence = 0.6 * aspect_score + 0.4 * area_score
        
        return confidence
    
    def get_debug_visualization(
        self,
        image: np.ndarray,
        detections: List[GateDetection],
        show_rich_info: bool = True,
    ) -> np.ndarray:
        """
        Create a debug visualization showing detections and optional rich ML features.

        Args:
            image: Original BGR image
            detections: List of detections from detect()
            show_rich_info: If True, draw size, angle, above/below on overlay

        Returns:
            Annotated image for debugging
        """
        vis = image.copy()
        h_img, w_img = image.shape[:2]

        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            color = (0, 255, 0) if det.confidence > 0.5 else (0, 255, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.drawContours(vis, [det.corners], 0, (255, 0, 0), 2)
            cv2.circle(vis, (det.center_x, det.center_y), 5, (0, 0, 255), -1)

            info = f"Gate {i+1}: {det.estimated_distance:.1f}m conf={det.confidence:.2f}"
            cv2.putText(vis, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if show_rich_info:
                pos = "above" if det.is_above_center else "below"
                cv2.putText(
                    vis, f"angle={det.rotation_deg:.0f} deg {pos}",
                    (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
                )
                cv2.putText(
                    vis, f"size={det.apparent_width_px:.0f}x{det.apparent_height_px:.0f}px",
                    (x, y + h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
                )
        return vis
    
    def get_mask_visualization(self, image: np.ndarray) -> np.ndarray:
        """Get the color mask for debugging threshold values."""
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        if self.is_red:
            mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
            mask = cv2.bitwise_or(mask, mask2)
        
        return mask


def pixel_to_normalized(x: int, y: int, width: int, height: int) -> Tuple[float, float]:
    """
    Convert pixel coordinates to normalized coordinates (-1 to 1).
    
    Useful for control: (0, 0) is center, positive x is right, positive y is down.
    """
    norm_x = (2 * x / width) - 1
    norm_y = (2 * y / height) - 1
    return norm_x, norm_y


def get_steering_error(detection: GateDetection, image_width: int, image_height: int) -> Tuple[float, float]:
    """
    Calculate steering error from gate detection.
    
    Returns:
        (horizontal_error, vertical_error) in range [-1, 1]
        Negative = gate is to the left/above center
        Positive = gate is to the right/below center
    """
    norm_x, norm_y = pixel_to_normalized(
        detection.center_x, 
        detection.center_y,
        image_width,
        image_height
    )
    return norm_x, norm_y
