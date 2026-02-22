"""
Visual GUI for Gate Detection
==============================
Run detection on synthetic or real images and show output in OpenCV windows.
Useful to verify the pipeline and inspect rich ML features.

Usage:
    # Synthetic test images (multiple scenarios)
    python visual_demo.py

    # Single image
    python visual_demo.py --image path/to/frame.png

    # With color preset
    python visual_demo.py --preset blue
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gate_detector import GateDetector, GateDetection
from test_detection import create_synthetic_gate_image


def make_features_panel(detections, image_width: int, image_height: int, panel_width: int = 380) -> np.ndarray:
    """Build a text panel image listing ML features for the first (or selected) detection."""
    panel = np.full((520, panel_width, 3), 28, dtype=np.uint8)
    if not detections:
        cv2.putText(panel, "No gates detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        return panel

    det = detections[0]
    features = det.to_ml_features(image_width, image_height)

    y = 28
    line_h = 22
    cv2.putText(panel, "ML features (first gate)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
    y += line_h + 4

    for k, v in features.items():
        if isinstance(v, float):
            text = f"  {k}: {v:.4f}"
        else:
            text = f"  {k}: {v}"
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        y += line_h
        if y > panel.shape[0] - 30:
            break

    return panel


def run_synthetic_demo(preset: str = "red"):
    """Cycle through synthetic test scenarios and show detection in GUI."""
    scenarios = [
        ("Center gate", {"gate_center": (320, 240), "gate_size": 150}),
        ("Offset gate", {"gate_center": (150, 120), "gate_size": 100}),
        ("Right of center", {"gate_center": (480, 240), "gate_size": 100}),
        ("Rotated gate", {"gate_center": (320, 240), "gate_size": 150, "rotation_deg": 25}),
        ("Large gate (close)", {"gate_center": (320, 240), "gate_size": 220}),
        ("Small gate (far)", {"gate_center": (320, 240), "gate_size": 70}),
    ]

    detector = GateDetector(color_preset=preset)
    idx = 0

    while True:
        name, kwargs = scenarios[idx]
        kwargs = dict(kwargs)
        if "gate_color_bgr" not in kwargs:
            colors = {"red": (0, 0, 255), "blue": (255, 0, 0), "orange": (0, 165, 255), "green": (0, 255, 0)}
            kwargs["gate_color_bgr"] = colors.get(preset, (0, 0, 255))

        image = create_synthetic_gate_image(**kwargs)
        detections = detector.detect(image)
        vis = detector.get_debug_visualization(image, detections, show_rich_info=True)
        mask = detector.get_mask_visualization(image)

        # Resize for display if needed
        scale = 1.0
        if image.shape[1] > 900:
            scale = 900 / image.shape[1]
        if scale != 1.0:
            w, h = int(image.shape[1] * scale), int(image.shape[0] * scale)
            image = cv2.resize(image, (w, h))
            vis = cv2.resize(vis, (w, h))
            mask = cv2.resize(mask, (w, h))

        h, w = image.shape[:2]
        features_panel = make_features_panel(detections, w, h)

        # Side-by-side: vis | mask; below: features
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        top = np.hstack([vis, mask_bgr])
        if top.shape[1] < features_panel.shape[1]:
            pad = np.full((top.shape[0], features_panel.shape[1] - top.shape[1], 3), 40, dtype=np.uint8)
            top = np.hstack([top, pad])
        combined = np.hstack([top, features_panel])
        # Title bar
        title = f"Scenario: {name}  |  Gates: {len(detections)}  [N]ext [P]rev [Q]uit"
        cv2.putText(combined, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Gate Detection - Visual Demo", combined)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            idx = (idx + 1) % len(scenarios)
        if key == ord("p"):
            idx = (idx - 1) % len(scenarios)

    cv2.destroyAllWindows()


def run_image_demo(image_path: str, preset: str = "red"):
    """Run detection on a single image and show GUI."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return

    detector = GateDetector(color_preset=preset)
    detections = detector.detect(image)
    vis = detector.get_debug_visualization(image, detections, show_rich_info=True)
    mask = detector.get_mask_visualization(image)
    h, w = image.shape[:2]
    features_panel = make_features_panel(detections, w, h)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    top = np.hstack([vis, mask_bgr])
    if top.shape[1] < features_panel.shape[1]:
        pad = np.full((top.shape[0], features_panel.shape[1] - top.shape[1], 3), 40, dtype=np.uint8)
        top = np.hstack([top, pad])
    combined = np.hstack([top, features_panel])
    cv2.putText(
        combined, f"Gates: {len(detections)}  [Q]uit",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )

    cv2.imshow("Gate Detection - Visual Demo", combined)
    print("Press 'q' in the window to quit.")
    while (cv2.waitKey(0) & 0xFF) != ord("q"):
        pass
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visual GUI for gate detection")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--preset", type=str, default="red", choices=["red", "blue", "orange", "green"],
                        help="Gate color preset for detection")
    args = parser.parse_args()

    if args.image:
        run_image_demo(args.image, args.preset)
    else:
        print("Synthetic demo: [N]ext scenario, [P]rev, [Q]uit")
        run_synthetic_demo(args.preset)


if __name__ == "__main__":
    main()
