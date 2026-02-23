# Example gate images (real drone racing frames)

This folder holds **real images of drone racing gates** for testing our gate detection pipeline. We do **not** use the dataset labels when running our detector—we run on the raw images and visually verify that gates are detected.

## Source: TII Racing drone-racing-dataset

Images are sampled from the [TII Racing drone-racing-dataset](https://github.com/tii-racing/drone-racing-dataset) (“Race Against the Machine”): high-speed autonomous and piloted flight with Arducam frames and gate annotations. The dataset includes:

- Fast (>21 m/s) quadrotor flight
- Autonomous and piloted trajectories
- Drone racing gates with bounding boxes and corner labels (we use only the camera frames here)
- [Paper (IEEE RA-L)](https://ieeexplore.ieee.org/document/10452776)

## How to populate this folder

1. **Clone and download the TII dataset** (one-time):

   ```bash
   git clone https://github.com/tii-racing/drone-racing-dataset.git
   cd drone-racing-dataset
   pip3 install -r requirements.txt
   ```

   Then download the actual data. The repo’s script uses `wget`, which is not installed on macOS by default.

   - **macOS (use curl):** from this repo’s root (`killallhumans`):
     ```bash
     ./gate_detection/tests/scripts/download_tii_dataset_curl.sh /path/to/drone-racing-dataset
     ```
     Example if the dataset is in `external_data/drone-racing-dataset`:
     ```bash
     ./gate_detection/tests/scripts/download_tii_dataset_curl.sh external_data/drone-racing-dataset
     ```
   - **Linux (wget):** `chmod +x data_downloader.sh && ./data_downloader.sh`  
   - **Windows:** run `data_downloader.cmd` from the dataset folder.

   This creates `data/piloted/` and `data/autonomous/` with flight folders and `camera_flight-.../` JPEGs.

2. **Copy sample frames into this folder**:

   From the `gate_detection` directory:

   ```bash
   python3 tests/scripts/fetch_gate_images_from_dataset.py --dataset /path/to/drone-racing-dataset [--max-per-flight 10] [--max-total 30]
   ```

   Example (if the dataset is cloned next to `killallhumans`):

   ```bash
   cd gate_detection
   python3 tests/scripts/fetch_gate_images_from_dataset.py --dataset ../drone-racing-dataset --max-per-flight 8 --max-total 24
   ```

   This finds all `camera_flight-*/` directories under `data/autonomous/` and `data/piloted/`, then copies up to `--max-per-flight` JPEGs per flight (evenly spaced) into `tests/example_gate_images/`, with filenames like `flight-01a-ellipse_00042.jpg`.

## Run our gate detector on these images

- **Single image:**

  ```bash
  python3 tests/visual_demo.py --image tests/example_gate_images/flight-01a-ellipse_00042.jpg
  ```

- **All images in this folder (batch):**

  ```bash
  python3 tests/scripts/run_detection_on_examples.py
  ```

  This runs the detector on every image in `example_gate_images/`, shows the overlay in a window (press a key to advance), and optionally saves results under `tests/example_gate_images_output/`.

Gates in the TII dataset may be orange/red/other colors; use `--preset orange` or `--preset red` in `visual_demo.py` if needed, or tune HSV with `python3 src/color_calibrator.py --image tests/example_gate_images/<file>.jpg`.
