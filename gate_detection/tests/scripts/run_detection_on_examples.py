#!/usr/bin/env python3
"""
Run our gate detector on all images in tests/example_gate_images/ (real drone racing frames).
Use this to verify detection on real-life images from the TII dataset or any JPEGs you drop in that folder.

Usage (from gate_detection/ or repo root):
    python3 tests/scripts/run_detection_on_examples.py
    python3 tests/scripts/run_detection_on_examples.py --preset orange --save
"""

import argparse
import sys
from pathlib import Path

# Add gate_detection/src so we can import gate_detector
script_dir = Path(__file__).resolve().parent
gate_detection_root = script_dir.parent.parent
sys.path.insert(0, str(gate_detection_root / "src"))

import cv2
from gate_detector import GateDetector


def main():
    parser = argparse.ArgumentParser(
        description="Run gate detection on all images in tests/example_gate_images/"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="orange",
        choices=["red", "blue", "orange", "green"],
        help="Gate color preset (TII gates are often orange; default orange)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save overlay images to tests/example_gate_images_output/",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override images directory (default: tests/example_gate_images)",
    )
    args = parser.parse_args()

    images_dir = args.images_dir or (gate_detection_root / "tests" / "example_gate_images")
    images_dir = images_dir.resolve()
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}")
        print("Populate it using tests/scripts/fetch_gate_images_from_dataset.py (see tests/example_gate_images/README.md)")
        return 1

    extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix in extensions
    )
    if not image_paths:
        print(f"No images found in {images_dir}")
        print("Add JPEGs or run fetch_gate_images_from_dataset.py first.")
        return 1

    out_dir = None
    if args.save:
        out_dir = gate_detection_root / "tests" / "example_gate_images_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving overlays to {out_dir}")

    detector = GateDetector(color_preset=args.preset)
    print(f"Running detector (preset={args.preset}) on {len(image_paths)} images. Press any key to advance, 'q' to quit.\n")

    for i, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  Skip (not readable): {path.name}")
            continue
        detections = detector.detect(image)
        vis = detector.get_debug_visualization(image, detections, show_rich_info=True)
        title = f"[{i+1}/{len(image_paths)}] {path.name}  Gates: {len(detections)}  [key]=next [q]=quit"
        cv2.putText(vis, title[:80], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("Gate detection (example images)", vis)
        if args.save and out_dir:
            out_path = out_dir / f"{path.stem}_detection.jpg"
            cv2.imwrite(str(out_path), vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("Quit.")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
