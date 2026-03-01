# Instructions

Project layout (gate_detection):
```
├── src/
│   ├── gate_detector.py      # Main detection module
│   └── color_calibrator.py   # HSV threshold tuning tool
├── tests/
│   ├── test_detection.py     # Unit tests and demos (2D, 3D-like, multiple gates, partial gates)
│   ├── visual_demo.py        # GUI: synthetic or single image + ML features panel
│   ├── video_detection_demo.py  # Play TII dataset frames with gate detection overlay (video-style)
│   ├── example_gate_images/  # Real gate images (e.g. from TII dataset)
│   │   └── README.md
│   └── scripts/
│       ├── fetch_gate_images_from_dataset.py  # Copy sample frames from TII dataset
│       ├── run_detection_on_examples.py       # Batch run detector on example_gate_images/
│       └── download_tii_dataset_curl.sh       # Download TII data with curl (macOS)
├── requirements.txt
└── README.md
```

For context, I kept the TODO's we have already accomplished under `TODO LIST (done)` so we do not have to do the stuff thats listed there. However, you need to do the tasks outlined under
`TODO List (not started)`

---

# TODO List (not started)

(none at present)

---

# TODO List (done)

1. **3D / non-head-on gates**
   Tests now include angled/rotated gates (e.g. 35, 45, -40 degrees) in `test_angled_gate_3d_like()` so the pipeline is validated for non-head-on views.

2. **Video-style detection on TII dataset**
   `tests/video_detection_demo.py` loads frames from `external_data/drone-racing-dataset/data/autonomous/<flight>/camera_flight-*/` (and piloted), runs gate detection on each frame, and displays the overlay so you can play through the flight. Usage:
   `python3 tests/video_detection_demo.py --dataset ../external_data/drone-racing-dataset [--flight flight-01a-ellipse] [--preset orange] [--delay 33]`

3. **Multiple gates per frame**
   The detector returns all valid contours (no limit of one). `test_multiple_gates_in_frame()` creates an image with 3 gates and asserts at least 2 (and typically 3) detections.

4. **Partial gates (not fully in frame)**
   When a contour's bounding box touches the image edge, the detector uses a relaxed aspect-ratio limit (`max_aspect_ratio_partial`, default 4.5) so partial gates are still accepted. `test_partial_gate_at_edge()` checks this. Constructor params: `max_aspect_ratio` (default 3.0), `max_aspect_ratio_partial` (default 4.5).

5. **Color-agnostic detection (TODO A)**
   The detector now defaults to color-agnostic mode (`color_preset=None`). Three strategies run in parallel and are fused via IoU-based NMS:
   - **Strategy A: Edge-based rectangles** -- bilateral filter + Canny + polygon approximation + colour-vs-background verification. Primary workhorse.
   - **Strategy B: Dynamic colour clustering + bar-grouping** -- Otsu foreground extraction, K-means clustering, then pairs vertical bar contours into gate openings. Catches hollow-frame gates (e.g. TII purple frames).
   - **Strategy C: Preset HSV (optional)** -- classic colour-threshold pipeline, enabled only when `color_preset` is explicitly passed.
   `GateDetection` dataclass gained `detection_method` and `detected_color_hsv` fields; `to_ml_features()` exposes them.
   New test: `test_color_agnostic_detection()` validates detection of cyan and white gates without any preset.
   Demo scripts (`visual_demo.py`, `video_detection_demo.py`) now default to color-agnostic mode and accept `--preset none`.
