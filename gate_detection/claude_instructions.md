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

---

# TODO List (done)

1. **3D / non–head-on gates**  
   Tests now include angled/rotated gates (e.g. 35°, 45°, -40°) in `test_angled_gate_3d_like()` so the pipeline is validated for non–head-on views.

2. **Video-style detection on TII dataset**  
   `tests/video_detection_demo.py` loads frames from `external_data/drone-racing-dataset/data/autonomous/<flight>/camera_flight-*/` (and piloted), runs gate detection on each frame, and displays the overlay so you can play through the flight. Usage:  
   `python3 tests/video_detection_demo.py --dataset ../external_data/drone-racing-dataset [--flight flight-01a-ellipse] [--preset orange] [--delay 33]`

3. **Multiple gates per frame**  
   The detector returns all valid contours (no limit of one). `test_multiple_gates_in_frame()` creates an image with 3 gates and asserts at least 2 (and typically 3) detections.

4. **Partial gates (not fully in frame)**  
   When a contour’s bounding box touches the image edge, the detector uses a relaxed aspect-ratio limit (`max_aspect_ratio_partial`, default 4.5) so partial gates are still accepted. `test_partial_gate_at_edge()` checks this. Constructor params: `max_aspect_ratio` (default 3.0), `max_aspect_ratio_partial` (default 4.5).

