"""Low-latency gate detector for the AI Grand Prix VQ2 simulator.

The VQ2 training build renders gates as luminous red-to-pink frames.  During
the gate's animation, pixels move outside the narrower ``red`` preset used by
the generic detector (notably toward hue 150 and saturation 50).  The cyan
racing line remains well separated in hue.

This wrapper intentionally enables only the preset contour strategy.  The
generic edge and dynamic-clustering strategies add roughly a frame of latency
at 30 Hz and did not detect the VQ2 gate in the first live capture.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from gate_detection.src.gate_detector import GateDetector


# OpenCV HSV bounds measured from AIGP FlightSim build 3385 VQ2 Training.
# Keep these as tuples so each detector instance can take private array copies.
VQ2_GATE_HSV_RANGES: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0, 50, 100]), np.array([12, 255, 255])),
    (np.array([150, 50, 100]), np.array([180, 255, 255])),
)


class VQ2GateDetector(GateDetector):
    """Preset-only detector for the glowing red/pink VQ2 gate frames."""

    def __init__(
        self,
        *,
        camera_fov_horizontal: float = 90.0,
        image_width: int = 640,
        image_height: int = 360,
        min_area: int = 500,
        max_area: int = 500_000,
        max_aspect_ratio: float = 3.0,
        min_confidence: float = 0.10,
    ) -> None:
        super().__init__(
            color_preset=None,
            camera_fov_horizontal=camera_fov_horizontal,
            image_width=image_width,
            image_height=image_height,
            enable_edge_rects=False,
            enable_dynamic_clustering=False,
            enable_preset_mode=True,
            min_confidence=min_confidence,
            morph_kernel_size=5,
            min_area=min_area,
            max_area=max_area,
            max_aspect_ratio=max_aspect_ratio,
        )
        self.hsv_ranges = [
            (lower.copy(), upper.copy())
            for lower, upper in VQ2_GATE_HSV_RANGES
        ]
