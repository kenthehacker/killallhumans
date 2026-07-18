"""Regression tests for the VQ2 simulator's pulsing gate appearance."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gate_detection.src.vq2_detector import VQ2GateDetector


def _bgr_from_hsv(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
    pixel = np.uint8([[[hue, saturation, value]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def _vq2_scene(gate_hsv: tuple[int, int, int]) -> np.ndarray:
    image = np.full((360, 640, 3), 18, dtype=np.uint8)

    # The VQ2 course guidance is cyan and should not become a gate candidate.
    cyan = _bgr_from_hsv(92, 240, 255)
    cv2.line(image, (100, 359), (300, 145), cyan, 4)
    cv2.line(image, (540, 359), (345, 145), cyan, 4)

    gate_color = _bgr_from_hsv(*gate_hsv)

    # Near, head-on gate.
    cv2.rectangle(image, (282, 134), (361, 213), gate_color, -1)
    cv2.rectangle(image, (299, 152), (345, 196), (18, 18, 18), -1)

    # Far gate seen obliquely after the course turns.
    cv2.rectangle(image, (410, 138), (436, 182), gate_color, -1)
    cv2.rectangle(image, (418, 146), (430, 174), (18, 18, 18), -1)

    # Starting-light red is deliberately below the 500 px gate area floor.
    cv2.circle(image, (118, 140), 9, gate_color, -1)
    return image


@pytest.mark.parametrize(
    "gate_hsv",
    [
        (2, 220, 255),    # conventional bright red
        (177, 90, 250),   # red wraparound with reduced saturation
        (165, 60, 250),   # pink/magenta pulse seen in live VQ2 frames
    ],
)
def test_detects_both_vq2_gates_across_animation_hues(
    gate_hsv: tuple[int, int, int],
) -> None:
    detector = VQ2GateDetector()

    detections = detector.detect(_vq2_scene(gate_hsv))

    assert [d.bbox for d in detections] == [
        (282, 134, 80, 80),
        (410, 138, 27, 45),
    ]
    assert all(d.detection_method == "preset_contour" for d in detections)


def test_uses_only_fast_preset_strategy() -> None:
    detector = VQ2GateDetector()

    assert detector.enable_preset is True
    assert detector.enable_edge is False
    assert detector.enable_dynamic is False


def test_ignores_cyan_raceline_and_small_red_signal() -> None:
    detector = VQ2GateDetector()

    detections = detector.detect(_vq2_scene((165, 60, 250)))

    assert len(detections) == 2
    assert all(d.center_x > 250 for d in detections)
