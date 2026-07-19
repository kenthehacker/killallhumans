# VQ2 image-space gate geometry

**Target:** FlightSim build 3385, Training mode  
**Output contract:** `GateObservationV1` (`aigp-vq2-gate-observation/1`)

## Scope

`gate_detection/src/vq2_geometry.py` is a deterministic, offline-capable
inner-aperture fitter for the existing VQ2 red/pink detector. It consumes a BGR
frame and the detector's support bbox. It does not consume the legacy
`GateDetection.corners` or `estimated_distance` fields because both remain
placeholders. It does not estimate metric pose, physical range, gate identity,
track ownership, or passage.

The existing `gate_detection_to_observation_v1` bbox adapter is unchanged. The
new `gate_detection_with_aperture_to_observation_v1` function is an explicit
opt-in compatibility path. If geometry is rejected, that path emits a degraded
bbox-only observation with no fitted aperture, scale, or skew.

## Fit model

The fitter applies the verified build-3385 HSV ranges and deterministic
morphology, then selects connected gate-colour support inside the supplied
bbox. A second component near the largest component's area is ambiguous and is
rejected. Horizontal and vertical scanlines locate the dark aperture gap and
fit its left, top, right, and bottom boundaries with deterministic least
squares plus fixed median-deviation trimming. Scanlines containing competing
gaps of similar size are rejected instead of choosing one arbitrarily.

With four supported inner sides, line intersections produce a visible
quadrilateral. With exactly three supported sides, inference is allowed only
when the missing side matches a clipped image border. The missing side then
uses `vq2-censored-image-square-px-v1`: the opposite visible span is extended
along the adjacent fitted lines under a square-in-pixel-space prior. This is a
censored image model, not evidence of a physical square, camera calibration,
homography, or pose. Two missing sides, an unmarked missing side, a degenerate
quad, disconnected ambiguity, or out-of-contract extrapolation is rejected.

Corners are labeled top-left, top-right, bottom-right, bottom-left. Only a
corner supported by both adjacent measured lines and lying inside the image is
published as visible. The fitted quad may retain inferred off-image corners,
and every such corner adds its matching clipping bit. Visible line segments are
clipped to the image border. The coloured contour bbox remains support only and
is never relabeled as an outer gate edge.

## Diagnostics and uncertainty

Successful fits publish measured support/inlier counts and a residual scaled
to normalized image units. The covariance feature order is exactly:

```text
center_x_norm, center_y_norm, log_scale, skew_x, skew_y
```

The current covariance is diagonal and deliberately floors visible-line
uncertainty at half a pixel. A censored fit adds a 12% aperture-span prior floor
and increases center, scale, and skew variance; its confidence is also reduced.
These values are conservative model diagnostics, not calibrated probabilities.
Every censored fit is degraded. A complete fit below the reviewed confidence
floor is also degraded. Nominal health requires all four visible sides, no
clipping, and sufficient fit confidence.

## Evidence and remaining acceptance

The checked-in tests use synthetic VQ2-colour frames only. They cover all four
single-side clipping directions, perspective corner ordering and convexity,
visible-versus-inferred fields, covariance growth, low confidence, competing
gaps, disconnected components, solid/degenerate support, deterministic output,
and exact preservation of the legacy bbox adapter.

This evidence does **not** establish either M2 replay acceptance item:

- stable center and honest uncertainty on the recorded top-clipped Gate 1;
- no Gate 0 replay regression.

Those require an approved replay input and the final replay processor. No
new/private full capture was added or read for this implementation. Crossing
residue isolation and active/shadow tracking are separate tracker work; this
module emits one unassociated candidate and cannot seed or transfer a track.

