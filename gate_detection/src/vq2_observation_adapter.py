"""Compatibility adapter from the legacy VQ2 bbox detector to contracts.

The current detector's metric distance assumes placeholder geometry and its
``corners`` are synthesized outer bbox corners.  This adapter intentionally
copies neither into inner-aperture or metric-pose fields.
"""

from __future__ import annotations

import math
from typing import Any

from competition.vq2_contracts import (
    EdgeSetV1,
    FeatureCovarianceV1,
    FitDiagnosticsV1,
    FrameEdge,
    FrameTimingV1,
    GateAuthorityEpochV1,
    GateObservationV1,
    MeasurementTimeBasis,
    ObservationHealth,
)


def _exact_pixel_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _finite(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def _bearing_x(pixel: float, width: int) -> float:
    return 2.0 * pixel / float(width) - 1.0


def _bearing_y(pixel: float, height: int) -> float:
    return 2.0 * pixel / float(height) - 1.0


def gate_detection_to_observation_v1(
    detection: Any,
    *,
    frame_timing: FrameTimingV1,
    authority: GateAuthorityEpochV1,
    candidate_id: str,
    measurement_uncertainty_ns: int,
    center_covariance: FeatureCovarianceV1,
    image_width: int = 640,
    image_height: int = 360,
    boundary_margin_px: int = 2,
) -> GateObservationV1:
    """Build an honest degraded view of the current bbox-only detector output."""

    if type(frame_timing) is not FrameTimingV1:
        raise TypeError("frame_timing must be FrameTimingV1")
    if type(authority) is not GateAuthorityEpochV1:
        raise TypeError("authority must be GateAuthorityEpochV1")
    if type(center_covariance) is not FeatureCovarianceV1:
        raise TypeError("center_covariance must be FeatureCovarianceV1")
    if center_covariance.feature_order != ("center_x_norm", "center_y_norm"):
        raise ValueError("legacy bbox adapter accepts center-only covariance")
    image_width = _exact_pixel_int(image_width, "image_width")
    image_height = _exact_pixel_int(image_height, "image_height")
    boundary_margin_px = _exact_pixel_int(boundary_margin_px, "boundary_margin_px")
    if image_width < 1 or image_height < 1 or boundary_margin_px < 0:
        raise ValueError("image dimensions must be positive and margin non-negative")
    bbox = getattr(detection, "bbox", None)
    if type(bbox) is not tuple or len(bbox) != 4:
        raise TypeError("detection bbox must be an exact four-tuple")
    x, y, width, height = (
        _exact_pixel_int(value, f"bbox[{index}]")
        for index, value in enumerate(bbox)
    )
    if (
        x < 0
        or y < 0
        or width < 1
        or height < 1
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError("detection bbox must stay inside the declared image")
    center_x = _exact_pixel_int(getattr(detection, "center_x", None), "center_x")
    center_y = _exact_pixel_int(getattr(detection, "center_y", None), "center_y")
    if not 0 <= center_x < image_width or not 0 <= center_y < image_height:
        raise ValueError("detection center must stay inside the declared image")
    if not x <= center_x < x + width or not y <= center_y < y + height:
        raise ValueError("detection center must stay inside its support bbox")
    confidence = _finite(getattr(detection, "confidence", None), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")

    clipping = FrameEdge.NONE
    if x <= boundary_margin_px:
        clipping |= FrameEdge.LEFT
    if y <= boundary_margin_px:
        clipping |= FrameEdge.TOP
    if x + width >= image_width - boundary_margin_px:
        clipping |= FrameEdge.RIGHT
    if y + height >= image_height - boundary_margin_px:
        clipping |= FrameEdge.BOTTOM

    center_norm = (
        _bearing_x(center_x, image_width),
        _bearing_y(center_y, image_height),
    )
    support = (
        x / float(image_width),
        y / float(image_height),
        (x + width) / float(image_width),
        (y + height) / float(image_height),
    )
    health_reason = "legacy_bbox_has_no_inner_aperture"
    if clipping:
        health_reason += ":censored_" + "_".join(
            name
            for edge, name in (
                (FrameEdge.LEFT, "left"),
                (FrameEdge.TOP, "top"),
                (FrameEdge.RIGHT, "right"),
                (FrameEdge.BOTTOM, "bottom"),
            )
            if clipping & edge
        )
    return GateObservationV1(
        frame_timing=frame_timing,
        measurement_time_monotonic_ns=frame_timing.final_unique_packet_monotonic_ns,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=measurement_uncertainty_ns,
        authority=authority,
        candidate_id=candidate_id,
        image_size_px=(image_width, image_height),
        center_norm=center_norm,
        support_bounds_norm=support,
        # A contour bbox is support, not a fitted gate edge measurement.
        outer_edges=EdgeSetV1(),
        inner_edges=EdgeSetV1(),
        inner_corners_norm=(None, None, None, None),
        fitted_inner_aperture_corners_norm=None,
        geometry_model_id=None,
        log_scale=None,
        projective_skew=None,
        clipping=clipping,
        confidence=confidence,
        covariance=center_covariance,
        fit=FitDiagnosticsV1(
            residual_rms=None,
            inlier_count=0,
            support_count=0,
        ),
        health=ObservationHealth.DEGRADED,
        health_reason=health_reason,
        provenance="vq2_legacy_outer_bbox",
    )


def observation_to_legacy_gate_target_fields(
    observation: GateObservationV1,
    *,
    frame_timing: FrameTimingV1,
    expected_authority: GateAuthorityEpochV1,
) -> dict[str, Any]:
    """Narrow transitional projection to the current ``GateTarget`` shape."""

    if type(observation) is not GateObservationV1:
        raise TypeError("observation must be GateObservationV1")
    if type(frame_timing) is not FrameTimingV1:
        raise TypeError("frame_timing must be FrameTimingV1")
    if type(expected_authority) is not GateAuthorityEpochV1:
        raise TypeError("expected_authority must be GateAuthorityEpochV1")
    if observation.authority != expected_authority:
        raise ValueError("observation authority does not match the consumer epoch")
    if observation.health is ObservationHealth.UNUSABLE:
        raise ValueError("unusable observation cannot enter the legacy tracker")
    if (
        frame_timing != observation.frame_timing
    ):
        raise ValueError("frame_timing does not match the observation measurement")
    image_width, image_height = observation.image_size_px
    left, top, right, bottom = observation.support_bounds_norm
    center_x = round((observation.center_norm[0] + 1.0) * 0.5 * image_width)
    center_y = round((observation.center_norm[1] + 1.0) * 0.5 * image_height)
    x = round(left * image_width)
    y = round(top * image_height)
    width = round(right * image_width) - x
    height = round(bottom * image_height) - y
    if not (
        0 <= center_x < image_width
        and 0 <= center_y < image_height
        and 0 <= x < image_width
        and 0 <= y < image_height
        and width > 0
        and height > 0
        and x + width <= image_width
        and y + height <= image_height
    ):
        raise ValueError("observation is not representable by the legacy pixel target")
    if not x <= center_x < x + width or not y <= center_y < y + height:
        raise ValueError("legacy target center must lie inside its support bbox")
    represented_center = (
        _bearing_x(center_x, image_width),
        _bearing_y(center_y, image_height),
    )
    represented_support = (
        x / float(image_width),
        y / float(image_height),
        (x + width) / float(image_width),
        (y + height) / float(image_height),
    )
    if (
        represented_center != observation.center_norm
        or represented_support != observation.support_bounds_norm
    ):
        raise ValueError(
            "observation is not exactly representable on the legacy pixel grid"
        )
    return {
        "frame_id": observation.frame.frame_id,
        "sim_time_ns": frame_timing.camera_source_time_ns,
        "received_monotonic_s": observation.measurement_time_monotonic_ns / 1e9,
        "center_x": center_x,
        "center_y": center_y,
        "bbox": (x, y, width, height),
        "confidence": observation.confidence,
    }


def observation_to_replay_detection_v1(
    observation: GateObservationV1,
    *,
    selector_eligible: bool,
) -> dict[str, Any]:
    """Project to the historical minimal replay detector output."""

    if type(observation) is not GateObservationV1:
        raise TypeError("observation must be GateObservationV1")
    if type(selector_eligible) is not bool:
        raise TypeError("selector_eligible must be an exact bool")
    if selector_eligible and observation.health is ObservationHealth.UNUSABLE:
        raise ValueError("unusable observation cannot be selector eligible")
    if selector_eligible and not (
        -1.0 <= observation.center_norm[0] <= 1.0
        and -1.0 <= observation.center_norm[1] <= 1.0
    ):
        raise ValueError("out-of-frame observation cannot be selector eligible")
    width, height = observation.image_size_px
    return {
        "center_px": [
            (observation.center_norm[0] + 1.0) * 0.5 * width,
            (observation.center_norm[1] + 1.0) * 0.5 * height,
        ],
        "selector_eligible": selector_eligible,
        "confidence": observation.confidence,
    }


__all__ = [
    "gate_detection_to_observation_v1",
    "observation_to_legacy_gate_target_fields",
    "observation_to_replay_detection_v1",
]
