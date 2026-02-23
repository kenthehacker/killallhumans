# Gate Detection: Diagnosis, Fixes & Limitations

## What's Going Wrong

I sampled the actual pixel HSV values from your TII dataset frames and the problem is clear:

### The gates aren't the color the detector expects

Your `video_detection_demo.py` defaults to the `orange` preset, which requires **S ≥ 100** and **V ≥ 100**. But the TII gates are dark purple frames with very low saturation:

| Gate region | H (hue) | S (saturation) | V (value) |
|---|---|---|---|
| Left bar (upper) | 24 | 42 | 124 |
| Right bar (mid) | 130 | 65 | 84 |
| Top bar | 17 | 46–64 | 109–111 |
| Bottom bar | 31 | 11 | 113 |
| **Orange preset expects** | **10–25** | **100–255** | **100–255** |

The orange preset matches **0.1%** of the image. The stock purple preset (`S ≥ 100, V ≥ 100`) matches **0.0%** — literally zero pixels.

That's why Frame 999 shows `Gates: 0` despite a gate filling half the screen. The one detection in Frame 25 is a false positive on a ceiling feature that happens to have a few orange-ish pixels.

### The gate frame is hollow — one contour won't cover it

Even with corrected HSV thresholds, the gate is a thin rectangular frame around a large opening. The opening breaks the frame into separate contours (individual bars), and each bar is tall and narrow with an aspect ratio of 3.5–4.3. The original detector rejects anything with aspect ratio > 3.0.

---

## What the Updated Detector Changes

The new `gate_detector_v2.py` addresses both problems:

### 1. `tii_purple` HSV preset

Measured from actual TII frames: `H ∈ [110, 150], S ∈ [20, 130], V ∈ [40, 135]`. This captures the dark purple gate bars, matching ~6% of the image vs 0.0% before.

### 2. Bar-grouping detection strategy

Instead of looking for one big contour shaped like a gate, the new mode:

1. Finds individual bar-like contours (tall and narrow, height/width ≥ 2.5)
2. Pairs vertical bars that have similar heights, aligned tops, and reasonable spacing
3. Infers the gate bounding box from each valid pair
4. Scores confidence based on bar similarity, gate squareness, and bar pixel area

### 3. Multi-range HSV support

The preset system now supports multiple `(lower, upper)` ranges per color (previously only red had this via a hardcoded second range). This makes it straightforward to add variants like `tii_purple_wide`.

### 4. IoU-based deduplication

When both detection modes run, overlapping detections are merged by keeping the higher-confidence one.

---

## Results on Your Screenshots

| Frame | Old detector (orange) | New detector (tii_purple + bar grouping) |
|---|---|---|
| Frame 999 (gate fills frame) | 0 gates (or 1 false positive) | **3 gates detected**, conf 0.89–0.93 |
| Frame 25 (gate is distant) | 1–2 false positives | **0 gates** |

Frame 999 is dramatically better — the main gate is detected with high confidence. The 3 detections correspond to different pairings of the 4 visible vertical bars (there appear to be 2 gate structures in view).

Frame 25 is still zero detections because the gate is too far away (~3 m estimated in your overlay) for individual bars to be large enough. At that distance, each bar is only ~35×80 pixels, and the purple color is weak.

---

## Where We've Hit Limitations

### Hard limits of HSV + contour for these gates

**Low color contrast.** The gate's purple (H≈130, S≈65, V≈80) is not very different from the surrounding environment. The ceiling has H≈19, S≈57, V≈114. Lowering thresholds to catch more gate pixels also catches more background. There's no clean separation in HSV space.

**Distance sensitivity.** Bar-grouping needs bars with area ≥ 3,000 pixels to be reliable. As the drone moves further from the gate, bars shrink below this threshold and detection drops out. Classical CV can't extrapolate "there's probably a gate there" from a few faint purple pixels the way a trained model can.

**Lighting variation.** The same gate frame reads as H≈24 (brownish) under direct lighting and H≈130 (purple) in shadow. A single HSV range can't cover both without also matching the walls. This is visible in the data: the left bar upper portion has H=24 while the lower portion has H=130.

**No shape model.** HSV knows nothing about "what a gate looks like." The checkered corner patterns — the most distinctive visual feature of TII gates — are invisible to color thresholding.

### What would actually solve this

This is the point where a **learned detector** (YOLO, SSD, or similar) would dramatically outperform classical CV. A trained model can:

- Learn the checkered pattern as a feature (not just color)
- Generalize across lighting conditions
- Detect gates at much greater distances from weaker visual cues
- Distinguish gates from non-gate rectangles by learned context

For the competition, the recommended path forward is to use the classical detector as a **training data bootstrapper**: run it on close-range frames where it works, manually verify/correct the bounding boxes, and use those as training labels for a YOLO or SSD model.

---

## How to Use the Updated Code

```python
from gate_detector_v2 import GateDetector

# For TII gates
detector = GateDetector(
    color_preset="tii_purple",
    image_width=1232,          # match your frame dimensions
    image_height=1020,
    bar_min_area=3000,         # increase if too many false positives
    bar_min_height_ratio=2.5,  # how "bar-like" a contour must be
    morph_kernel_size=7,       # larger = more noise removal
    enable_contour_mode=False, # disable for TII (too many false positives)
    enable_bar_grouping=True,  # the strategy that works for these gates
)

detections = detector.detect(image)
vis = detector.get_debug_visualization(image, detections)
```

### Tuning tips

- **Too many false positives?** Increase `bar_min_area` (try 5000–8000) and `bar_min_height_ratio` (try 3.0).
- **Missing gates?** Use `tii_purple_wide` preset, lower `bar_min_area` to 1000, lower `bar_min_height_ratio` to 2.0.
- **Different gate color?** Use `color_calibrator.py` on a frame to find the right HSV range, then call `detector.set_custom_thresholds(lower, upper)`.
- **Both bright and dark gates?** Enable both `enable_contour_mode` and `enable_bar_grouping`.
