#!/usr/bin/env python3
"""
Copy sample camera frames from the TII Racing drone-racing-dataset into
gate_detection/tests/example_gate_images/ for testing our gate detector on real images.

Dataset: https://github.com/tii-racing/drone-racing-dataset
We use only the images (not the labels) so we can test detection as if on unlabeled data.

Usage:
    # From gate_detection/ or repo root
    python3 tests/scripts/fetch_gate_images_from_dataset.py --dataset /path/to/drone-racing-dataset
    python3 tests/scripts/fetch_gate_images_from_dataset.py --dataset ../drone-racing-dataset --max-per-flight 8 --max-total 24
"""

import argparse
import shutil
from pathlib import Path


def find_camera_dirs(dataset_root: Path) -> list[tuple[str, Path]]:
    """Find all camera_flight-* directories under data/autonomous and data/piloted.
    Returns list of (flight_name, camera_dir_path).
    """
    out = []
    for data_sub in ("autonomous", "piloted"):
        data_dir = dataset_root / "data" / data_sub
        if not data_dir.is_dir():
            continue
        for flight_dir in sorted(data_dir.iterdir()):
            if not flight_dir.is_dir():
                continue
            # Each flight has camera_flight-<name>/ with JPEGs
            for sub in flight_dir.iterdir():
                if sub.is_dir() and sub.name.startswith("camera_flight-"):
                    flight_name = flight_dir.name  # e.g. flight-01a-ellipse
                    out.append((flight_name, sub))
                    break
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Copy sample frames from TII drone-racing-dataset into tests/example_gate_images/"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to cloned drone-racing-dataset repo (containing data/autonomous and data/piloted)",
    )
    parser.add_argument(
        "--max-per-flight",
        type=int,
        default=10,
        help="Max number of frames to copy per flight (default 10)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=40,
        help="Max total images to copy (default 40)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: gate_detection/tests/example_gate_images)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    if not dataset_root.is_dir():
        print(f"Error: dataset path is not a directory: {dataset_root}")
        return 1

    # Resolve script location: .../gate_detection/tests/scripts/fetch_...
    script_dir = Path(__file__).resolve().parent
    gate_detection_tests = script_dir.parent
    output_dir = (args.output_dir or (gate_detection_tests / "example_gate_images")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_dirs = find_camera_dirs(dataset_root)
    if not camera_dirs:
        print("No camera_flight-* directories found under data/autonomous or data/piloted.")
        print("Ensure you have run the dataset download script (data_downloader.sh / .cmd) first.")
        return 1

    print(f"Found {len(camera_dirs)} flights with camera data.")
    total_copied = 0

    for flight_name, cam_dir in camera_dirs:
        if total_copied >= args.max_total:
            break
        jpegs = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.jpeg")) + sorted(cam_dir.glob("*.JPG"))
        if not jpegs:
            print(f"  No JPEGs in {cam_dir}")
            continue
        # Evenly sample up to max_per_flight
        n = min(args.max_per_flight, len(jpegs), args.max_total - total_copied)
        if n == 0:
            continue
        step = max(1, len(jpegs) // n)
        indices = list(range(0, len(jpegs), step))[:n]
        count_this = 0
        for idx in indices:
            if total_copied >= args.max_total:
                break
            src = jpegs[idx]
            stem = src.stem
            dest_name = f"{flight_name}_{stem}.jpg"
            dest = output_dir / dest_name
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                continue
            shutil.copy2(src, dest)
            total_copied += 1
            count_this += 1
            print(f"  {dest_name}")
        if count_this:
            print(f"  -> {flight_name}: copied {count_this} images")

    print(f"\nDone. {total_copied} images in {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
