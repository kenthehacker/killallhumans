# Gate Detection (AI Grand Prix)

Do this first before dev & running anything:

```bash
cd gate_detection
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Quick reference: scripts

| Script | Command | Purpose |
|--------|--------|--------|
| Unit tests | `python3 tests/test_detection.py` | Run all detection tests + save demo images |
| Visual GUI | `python3 tests/visual_demo.py` | Synthetic scenarios or `--image path.png`; shows mask + ML features |
| HSV calibrator | `python3 src/color_calibrator.py --image path.png` | Tune HSV thresholds on image/video/camera |
| Fetch real gate images | `python3 tests/scripts/fetch_gate_images_from_dataset.py --dataset /path/to/drone-racing-dataset` | Copy sample frames from [TII drone-racing-dataset](https://github.com/tii-racing/drone-racing-dataset) into `tests/example_gate_images/` |
| Run on real images | `python3 tests/scripts/run_detection_on_examples.py [--preset orange] [--save]` | Run detector on all images in `tests/example_gate_images/` (batch) |
| Video detection demo | `python3 tests/video_detection_demo.py --dataset /path/to/drone-racing-dataset [--flight flight-01a-ellipse] [--preset orange]` | Play TII dataset frames with gate detection overlay; [SPACE] pause, [Q] quit |

**Testing on real images:** Put real drone gate frames in `tests/example_gate_images/` (see that folder’s [README](tests/example_gate_images/README.md)). You can sample images from the TII Racing dataset with the fetch script, then run `run_detection_on_examples.py` to verify detection on unlabeled real-life frames.

---

# 1. `gate_detector.py` — Gate detection pipeline

## 1.1 What it does

This module is the **gate detection** stage of the autonomy stack. Given a single **BGR camera image**, it:

1. Finds **which pixels look like the gate color** (HSV thresholding).
2. Turns those pixels into **blobs**, then **contours** (outlines).
3. Keeps only contours that look **gate-shaped** (roughly square, not too small or huge).
4. For each kept contour, builds a **`GateDetection`** with center, bbox, corners, distance, orientation, and position (above/below, left/right).
5. Exposes that as **ML-friendly features** via `to_ml_features()` and helpers like `get_steering_error()`.

No neural nets — it's **classical CV** (OpenCV + HSV + contours + simple geometry).

---

## 1.2 The `GateDetection` dataclass

Each detection is one **gate instance** in the image. The dataclass holds everything we compute for that instance.

**Core pixel geometry**

- **`center_x`, `center_y`** — Center of the gate in image coordinates (from the axis-aligned bounding box).
- **`bbox`** — `(x, y, width, height)` of the axis-aligned bounding box. Used for drawing and cropping.
- **`corners`** — Four 2D points `(4, 2)` from the **minimum-area rotated rectangle** (OpenCV order). Used for pose/debug drawing; later you could feed these to PnP for 3D pose.
- **`area`** — Contour area in pixels. Used for sorting (bigger = usually closer) and filtering.

**Distance and size**

- **`estimated_distance`** — Distance to the gate in meters from a **pinhole model**: same real size → bigger in image → closer. Uses `GATE_WIDTH_METERS` and camera FOV.
- **`apparent_width_px`, `apparent_height_px`** — Width and height of the **min-area rectangle** in pixels (we store max/min of the two so “width” is the longer side). Tells you how big the gate looks; useful for ML and distance.

**Viewing angle / orientation**

- **`rotation_deg`** — Angle of the min-area rectangle (OpenCV convention, about -90° to 0°). In-plane “tilt” of the gate in the image.
- **`aspect_ratio`** — `max(w,h)/min(w,h)` of the min-area rect. Square gate head-on → ~1.0; viewing at an angle or elongated blob → larger. Used as a simple “perspective” / shape cue.
- **`rectangularity`** — `contour_area / (rect_w * rect_h)`. How well the contour fills its min-area box. Solid square → ~1; hollow frame → smaller. Used in confidence.

**Position relative to frame**

- **`normalized_center_x`, `normalized_center_y`** — Center mapped to **[-1, 1]** (0 = image center). Positive x = right, positive y = down. Handy for control and ML.
- **`is_above_center`, `is_below_center`, `is_left_of_center`, `is_right_of_center`** — Booleans from comparing center to image center. Easy features for “gate above/below/left/right of crosshair”.

**Quality**

- **`confidence`** — 0–1 score from aspect ratio (square-ish) and area (not tiny). Not a probability; just “how gate-like” this blob is.

**`to_ml_features(image_width, image_height)`**

Returns a **single flat dict** per gate with the above in a form suitable for ML (path planning / flight path model): normalized bbox, booleans as floats, distances, angles, etc. So one gate → one feature vector (or you can stack multiple gates).

---

## 1.3 `GateDetector` class — step-by-step

### Constructor `__init__(...)`

- **`color_preset`** — Chooses which gate color to segment: `"red"`, `"blue"`, `"orange"`, `"green"`, etc. Maps to **HSV ranges** via `_get_color_thresholds`.
- **`min_area`, `max_area`** — Contour area limits (pixels). Below `min_area` = noise; above `max_area` = bogus huge regions. Defaults 500 and 500000.
- **`camera_fov_horizontal`** — Horizontal FOV in degrees (default 90°). Used to get **focal length in pixels** for distance.
- **`image_width`, `image_height`** — Assumed image size for focal length and normalization.

Internal setup:

1. **`_get_color_thresholds(preset)`** — Returns `(hsv_lower, hsv_upper)` for that preset. OpenCV HSV: H 0–180, S/V 0–255. Stored as `self.hsv_lower`, `self.hsv_upper`.
2. **Red special case** — Red sits at the wraparound (0° and 180°). So for `"red"` we also set `hsv_lower2` / `hsv_upper2` (e.g. 170–180) and `is_red = True`. Later we **OR** two masks so both red lobes are kept.

### `_get_color_thresholds(preset)`

- Preset dict maps names to `(lower, upper)` numpy arrays `[H, S, V]`.
- Example: red low `[0,100,100]`, high `[10,255,255]` (first red lobe). Unknown preset → fallback to red.
- Returns the two arrays; red’s second range is fixed in `__init__`.

### `set_custom_thresholds(lower, upper)`

- Replaces `hsv_lower` / `hsv_upper` with user arrays (e.g. from the calibrator).
- Sets `is_red = False` so we don’t add the second red range.

### Main pipeline: `detect(image)`

Input: one **BGR** frame. Output: list of **`GateDetection`**, sorted by **area descending** (largest first ≈ closest gate).

**Step 1 — BGR → HSV**

- `cv2.cvtColor(image, cv2.COLOR_BGR2HSV)`.
- HSV separates **hue** (color), **saturation** (colorfulness), **value** (brightness), which makes “red gate” robust to lighting compared to raw BGR.

**Step 2 — Binary mask by color**

- `mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)`. Pixels inside the range → 255, else 0.
- If **`is_red`**: build a second mask with `hsv_lower2`/`hsv_upper2`, then `mask = cv2.bitwise_or(mask, mask2)` so both red lobes are 255.

**Step 3 — Clean up the mask (morphology)**

- **Close** (`MORPH_CLOSE`) with a 5×5 kernel: fills small holes inside the gate.
- **Open** (`MORPH_OPEN`) with same kernel: removes small isolated blobs. Result: cleaner, more solid regions.

**Step 4 — Find contours**

- `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
- **RETR_EXTERNAL**: only outer contours (no holes as separate contours).
- **CHAIN_APPROX_SIMPLE**: compress line segments (fewer points). Each contour is a list of (x,y) points.

**Step 5 — Process each contour**

- For each contour, call **`_process_contour(contour, image.shape)`**. If it returns a `GateDetection`, append it. So we only keep contours that pass size and shape checks.

**Step 6 — Sort by area**

- `detections.sort(key=lambda d: d.area, reverse=True)`. Biggest (usually closest) first. Caller can take `detections[0]` as “primary” gate.

---

## 1.4 `_process_contour(contour, image_shape)` — per-blob logic

This turns one contour into either **one `GateDetection`** or **`None`** (rejected).

**Step 1 — Area filter**

- `area = cv2.contourArea(contour)`.
- If `area < min_area` or `area > max_area` → return `None`. Drops tiny noise and huge mistakes.

**Step 2 — Axis-aligned bounding box**

- `x, y, w, h = cv2.boundingRect(contour)`. Used for center and bbox: `center_x = x + w//2`, `center_y = y + h//2`.

**Step 3 — Minimum-area rotated rectangle**

- `rect = cv2.minAreaRect(contour)` → `(center, (width, height), angle)`.
- `box = cv2.boxPoints(rect)` → four corners; `corners = np.int32(box)` for drawing.
- Gives **orientation** (angle) and **aspect ratio** even when the gate is tilted in the image.

**Step 4 — Aspect ratio check**

- `rect_w`, `rect_h` from `rect[1]`; if `rect_h == 0` → `None`.
- `aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)`. Gates are roughly square; if `aspect_ratio > 3.0` we reject (too elongated, likely not a gate).

**Step 5 — Rectangularity**

- `rect_area = rect_w * rect_h`; if 0 → `None`.
- `rectangularity = area / rect_area`. Measures how well the contour fills its rotated box (hollow frame vs solid square).

**Step 6 — Normalized position and above/below**

- `img_h`, `img_w = image_shape[:2]`.
- `norm_x = (2.0 * center_x / img_w) - 1.0`, same for y. So center of image = (0, 0), left = -1, right = +1, etc.
- `mid_x`, `mid_y` = image center. Then:
  - `is_above_center = center_y < mid_y`
  - `is_below_center = center_y > mid_y`
  - `is_left_of_center`, `is_right_of_center` similarly.

**Step 7 — Distance and confidence**

- **`_estimate_distance(max(rect_w, rect_h))`** — pinhole: `focal_length = image_width / (2 * tan(fov/2))`, then `distance = (GATE_WIDTH_METERS * focal_length) / apparent_size_pixels`. So bigger in image → smaller distance.
- **`_calculate_confidence(aspect_ratio, rectangularity, area)`** — aspect score (1 minus penalty for being non-square), area score (cap at 10000 px), then `0.6 * aspect_score + 0.4 * area_score`.

**Step 8 — Build and return `GateDetection`**

- Fill all fields: center, bbox, corners, area, estimated_distance, apparent_width_px/height_px (max/min of rect), rotation_deg, aspect_ratio, rectangularity, normalized coords, above/below/left/right booleans, confidence.

So: one contour → at most one `GateDetection`, with all the “rich” info used later for control and ML.

---

## 1.5 `_estimate_distance(apparent_size_pixels)`

- If `apparent_size_pixels == 0` → return `inf`.
- **Focal length (px):** `f = image_width / (2 * tan(radians(fov/2)))`.
- **Distance:** `d = (GATE_WIDTH_METERS * f) / apparent_size_pixels`. Assumes you know real gate size; when DCL gives exact dimensions, plug them into `GATE_WIDTH_METERS` (and optionally use height for a better estimate).

---

## 1.6 `_calculate_confidence(aspect_ratio, rectangularity, area)`

- **Aspect score:** 1.0 minus a penalty for deviating from 1.0 (square), capped so score stays in [0,1].
- **Area score:** `min(area / 10000, 1.0)` so bigger detections get higher score.
- **Confidence:** `0.6 * aspect_score + 0.4 * area_score`. So “square and not tiny” → high confidence.

---

## 1.7 `get_debug_visualization(image, detections, show_rich_info=True)`

- Copies `image`, then for each detection:
  - Draws **bbox** (green if confidence > 0.5, else yellow).
  - Draws **rotated corners** (blue).
  - Draws **center** (red dot).
  - Text: distance and confidence.
  - If `show_rich_info`: adds line with **angle** and **above/below**, and **size in px**.
- Returns the annotated image (for saving or display).

---

## 1.8 `get_mask_visualization(image)`

- Same as the first part of `detect`: BGR→HSV, inRange (with red dual range if needed), **no morphology**. Returns the raw binary mask so you can see what the color thresholds are selecting.

---

## 1.9 Helpers at module level

- **`pixel_to_normalized(x, y, width, height)`** — Maps pixel (x,y) to **(-1, 1)** with (0,0) at image center. Used for control/steering.
- **`get_steering_error(detection, image_width, image_height)`** — Returns `(norm_x, norm_y)` of the gate center. So you get **horizontal and vertical error** in [-1, 1] for a controller: negative = gate left/above, positive = right/below.

---

# 2. `color_calibrator.py` — HSV tuning tool

## 2.1 What it does

The **gate detector** depends on **HSV ranges** for the gate color. Those depend on lighting and exact gate color. This script is an **interactive calibrator**: you give it an image, video, or webcam, and with **trackbars** you adjust H, S, V min/max until the gate is cleanly isolated in the mask. You can **print** or **save** the values and then plug them into `GateDetector` via `set_custom_thresholds` or by adding a new preset.

So: **color_calibrator.py** = find good HSV; **gate_detector.py** = use those (or presets) to detect gates and output rich detections.

---

## 2.2 `HSVCalibrator` class — step-by-step

### `__init__`

- Window names: `"HSV Calibrator"`, `"Mask"`.
- **Default trackbar values:** H 0–180, S 100–255, V 100–255 (broad so you see something at start).
- Calls **`_setup_windows()`** to create windows and trackbars.

### `_setup_windows()`

- **`cv2.namedWindow(...)`** for the main window and the mask window.
- **Six trackbars** on the main window: "H Low", "H High", "S Low", "S High", "V Low", "V High". Each has a callback that only stores the value (e.g. `_on_h_low(val)` → `self.h_low = val`). So whenever you move a slider, the instance’s `h_low`, `h_high`, etc. are updated.

### `get_thresholds()`

- Returns `(lower, upper)` as numpy arrays `[H, S, V]` from the current trackbar state. Used both for **mask computation** and for **print/save**.

### `process_frame(frame)`

- **BGR → HSV.**
- **`cv2.inRange(hsv, lower, upper)`** with current thresholds.
- **Morphology:** same as detector — close then open with 5×5 kernel. So the calibrator shows you a “cleaned” mask similar to what the detector will see.
- Returns the binary mask (no second red range; calibrator is generic).

### `run_on_image(image_path)`

- **Load image** with `cv2.imread`; if fail, print and return.
- Print short instructions: adjust trackbars; **'p'** = print thresholds; **'s'** = save; **'q'** = quit.
- **Loop:**
  1. `mask = self.process_frame(image)`.
  2. `result = cv2.bitwise_and(image, image, mask=mask)` — original image but only where mask is white.
  3. **Info overlay:** copy image, draw current `lower` and `upper` as text.
  4. **Contours:** `cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`, draw all contours in green on the overlay.
  5. **Show three windows:** overlay (with contours + text), mask, filtered (result).
  6. **Keys:** 'q' break, 'p' `_print_values()`, 's' `_save_values()`.
- At exit, **`cv2.destroyAllWindows()`**.

So you see in real time: **what the mask looks like**, **what the image looks like where the mask is on**, and **contours** that the detector would later filter by size/shape.

### `run_on_video(video_path)`

- **Open video** with `cv2.VideoCapture`. If not opened, return.
- **Pause state:** `paused = False`, `frame = None`.
- **Loop:**
  - If not paused: read frame; if EOF, seek to 0 (loop).
  - If no frame yet, continue.
  - Same as image: `process_frame`, bitwise_and, overlay with lower/upper and “PAUSED” if paused, show three windows.
  - **Keys:** 'q' quit, 'p' print, 's' save, **space** toggle pause.
- **WaitKey:** 30 ms when playing, 1 ms when paused so sliders stay responsive.
- Release capture and destroy windows.

So you can scrub (by pausing) or let it play and tune HSV on real simulator or recorded footage.

### `run_on_camera(camera_id=0)`

- **Open camera** with `VideoCapture(camera_id)`.
- **Loop:** read frame; same mask + overlay + three windows.
- **Extra overlay:** **crosshair** at center and **HSV at center pixel** (`hsv[h/2, w/2]`). So you can point the camera at the gate and read the HSV of the gate color to set ranges.
- Keys: 'q', 'p', 's'.
- Release and destroy.

### `_print_values()`

- Gets current `(lower, upper)` from `get_thresholds()`.
- Prints two lines you can copy into code:
  - `hsv_lower = np.array([...])`
  - `hsv_upper = np.array([...])`

### `_save_values(filename="hsv_thresholds.txt")`

- Writes a short file: comment line, “H_low, S_low, V_low, H_high, S_high, V_high” comment, then one line with six numbers. So you can load these later and pass them to `set_custom_thresholds` or a config.

---

## 2.3 `create_test_image_with_gate(color)`

- **Image:** 480×640, dark gray (40,40,40) BGR.
- **Gate:** outer rectangle, inner rectangle cut out (hollow frame), color from dict (red/blue/orange/green in BGR).
- **Noise:** small random add for a bit of realism.
- Returns the image. Used when you run the calibrator with **`--test red`** (etc.) or no args to get a synthetic gate and tune thresholds without real footage.

---

## 2.4 `main()` — CLI

- **Parser:** `--image`, `--video`, `--camera`, `--test` (red/blue/orange/green).
- **One** `HSVCalibrator()` instance.
- **Branch:**
  - `--image path` → `run_on_image(path)`.
  - `--video path` → `run_on_video(path)`.
  - `--camera N` → `run_on_camera(N)`.
  - `--test color` → create test image with `create_test_image_with_gate(color)`, save to `test_gate_{color}.png`, then `run_on_image` on that file.
  - **No args** → create red test image, save `test_gate_red.png`, run calibrator on it.

So you can calibrate on a still, a video, a live camera, or a synthetic gate.

---

# 3. How the two files work together

- **color_calibrator.py** helps you **choose HSV bounds** (and optionally save/print them) for the gate color in your environment.
- **gate_detector.py** uses those bounds (via presets or `set_custom_thresholds`) to **segment by color**, clean the mask, find contours, filter by size and shape, and output **`GateDetection`** with geometry, distance, orientation, and above/below/left/right.
- **`GateDetection.to_ml_features()`** and **`get_steering_error()`** then feed into your **path planner / flight path ML** and control loop.

---

# 4. Project layout (from `claude_instructions.md`)

```
├── src/
│   ├── gate_detector.py      # Main detection module
│   └── color_calibrator.py   # HSV threshold tuning tool
├── tests/
│   ├── test_detection.py     # Unit tests and demos
│   └── visual_demo.py        # GUI: detection overlay + ML features panel
├── assets/                   # Images, models, etc. (optional)
├── requirements.txt
└── README.md
```
