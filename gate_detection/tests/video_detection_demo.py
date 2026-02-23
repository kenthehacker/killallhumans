"""
Video-style gate detection demo using TII drone-racing-dataset frames.
========================================
Loads camera frames from data/autonomous/<flight>/camera_flight-<name>/ (or piloted),
runs gate detection on each frame, and displays the result so you can "play" through
the flight and see how gates are detected over time.

Usage (from gate_detection/ or repo root):
    # Use dataset path; picks first available flight
    python3 tests/video_detection_demo.py --dataset ../external_data/drone-racing-dataset

    # Limit to one flight and set playback delay (ms per frame)
    python3 tests/video_detection_demo.py --dataset ../external_data/drone-racing-dataset --flight flight-01a-ellipse --delay 33

    # Orange preset (common for TII gates)
    python3 tests/video_detection_demo.py --dataset ../external_data/drone-racing-dataset --preset orange

Keys: [SPACE] pause/resume, [Q] quit, [S] step one frame when paused.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add gate_detection/src
script_dir = Path(__file__).resolve().parent
gate_detection_root = script_dir.parent
sys.path.insert(0, str(gate_detection_root / "src"))

import cv2
from gate_detector import GateDetector


def find_camera_frames(dataset_root: Path, flight_name: Optional[str] = None) -> List[Path]:
    """
    Find all JPEG paths under data/autonomous and data/piloted, optionally filtered by flight.
    Returns paths sorted by name (temporal order).
    """
    dataset_root = dataset_root.resolve()
    frames = []
    for sub in ("autonomous", "piloted"):
        data_dir = dataset_root / "data" / sub
        if not data_dir.is_dir():
            continue
        for flight_dir in sorted(data_dir.iterdir()):
            if not flight_dir.is_dir():
                continue
            if flight_name is not None and flight_dir.name != flight_name:
                continue
            # camera_flight-<name> or camera_flight-<name>/ with images
            for subdir in flight_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("camera_"):
                    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
                        frames.extend(sorted(subdir.glob(ext)))
                    break
            if flight_name is not None and frames:
                return sorted(frames)
    return sorted(frames)


def main():
    parser = argparse.ArgumentParser(
        description="Play TII drone-racing-dataset frames with gate detection overlay."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to drone-racing-dataset repo (e.g. ../external_data/drone-racing-dataset)",
    )
    parser.add_argument(
        "--flight",
        type=str,
        default=None,
        help="Restrict to one flight (e.g. flight-01a-ellipse)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="orange",
        choices=["red", "blue", "orange", "green"],
        help="Gate color preset (default orange for TII)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=33,
        help="Delay in ms per frame (default 33 ~ 30 fps); 0 = wait for key",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to play (0 = all)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset
    if dataset_root is None:
        # Default relative to gate_detection
        dataset_root = gate_detection_root.parent / "external_data" / "drone-racing-dataset"
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        print(f"Dataset path not found: {dataset_root}")
        print("Usage: python3 tests/video_detection_demo.py --dataset /path/to/drone-racing-dataset")
        return 1

    frame_paths = find_camera_frames(dataset_root, args.flight)
    if not frame_paths:
        print("No camera frames found under data/autonomous or data/piloted.")
        print("Ensure the dataset is downloaded and camera_flight-* folders contain JPEGs.")
        return 1

    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    print(f"Playing {len(frame_paths)} frames (preset={args.preset}). [SPACE] pause, [Q] quit, [S] step.")
    detector = GateDetector(color_preset=args.preset)
    paused = False
    idx = 0

    while True:
        path = frame_paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            idx = (idx + 1) % len(frame_paths)
            continue
        detections = detector.detect(image)
        vis = detector.get_debug_visualization(image, detections, show_rich_info=True)
        h, w = vis.shape[:2]
        cv2.putText(
            vis, f"Frame {idx+1}/{len(frame_paths)}  Gates: {len(detections)}  [SPACE] pause [Q] quit",
            (10, min(30, h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.imshow("Gate detection (video)", vis)
        wait = args.delay if not paused else 1
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("s") and paused:
            idx = (idx + 1) % len(frame_paths)
            continue
        if not paused:
            idx = (idx + 1) % len(frame_paths)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
