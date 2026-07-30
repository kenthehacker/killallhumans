"""Regression tests for the VQ2 simulator's pulsing gate appearance."""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import pytest

from competition.vq2_contracts import (
    FeatureCovarianceV1,
    FrameEdge,
    FrameIdentityV1,
    FrameTimingV1,
    GateAuthorityEpochV1,
    GateObservationV1,
    ObservationHealth,
)
from gate_detection.src import vq2_geometry as vq2_geometry_module
from gate_detection.src.vq2_detector import VQ2GateDetector
from gate_detection.src.vq2_geometry import (
    ApertureSide,
    VQ2ApertureConfig,
    VQ2ApertureTrackingPrior,
    fit_vq2_aperture_bgr,
    fit_vq2_aperture_mask,
    passage_geometry_from_vq2_aperture_fit,
    tracking_geometry_from_vq2_aperture_fit,
    vq2_gate_mask_from_bgr,
)
from gate_detection.src.vq2_observation_adapter import (
    gate_detection_to_observation_v1,
    gate_detection_with_aperture_to_observation_v1,
    observation_to_legacy_gate_target_fields,
)


def _bgr_from_hsv(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
    pixel = np.uint8([[[hue, saturation, value]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def _vq2_scene(gate_hsv: tuple[int, int, int]) -> np.ndarray:
    image = np.full((360, 640, 3), 18, dtype=np.uint8)

    # The VQ2 course guidance is cyan and should not become a gate candidate.
    cyan = _bgr_from_hsv(92, 240, 255)
    cv2.line(image, (100, 359), (300, 145), cyan, 4)
    cv2.line(image, (540, 359), (345, 145), cyan, 4)

    gate_color = _bgr_from_hsv(*gate_hsv)

    # Near, head-on gate.
    cv2.rectangle(image, (282, 134), (361, 213), gate_color, -1)
    cv2.rectangle(image, (299, 152), (345, 196), (18, 18, 18), -1)

    # Far gate seen obliquely after the course turns.
    cv2.rectangle(image, (410, 138), (436, 182), gate_color, -1)
    cv2.rectangle(image, (418, 146), (430, 174), (18, 18, 18), -1)

    # Starting-light red is deliberately below the 500 px gate area floor.
    cv2.circle(image, (118, 140), 9, gate_color, -1)
    return image


def _aperture_scene(
    inner_bounds: tuple[int, int, int, int],
    *,
    image_width: int = 200,
    image_height: int = 160,
    thickness_px: int = 10,
) -> np.ndarray:
    """Synthetic VQ2-colour frame; bounds may extend outside the image."""

    image = np.full((image_height, image_width, 3), 18, dtype=np.uint8)
    left, top, right, bottom = inner_bounds
    color = _bgr_from_hsv(165, 100, 250)
    cv2.rectangle(
        image,
        (left - thickness_px, top - thickness_px),
        (right + thickness_px, bottom + thickness_px),
        color,
        -1,
    )
    cv2.rectangle(image, (left, top), (right, bottom), (18, 18, 18), -1)
    return image


def _aperture_detection(image: np.ndarray):
    height, width = image.shape[:2]
    detections = VQ2GateDetector(
        image_width=width,
        image_height=height,
        min_area=100,
    ).detect(image)
    assert len(detections) == 1
    return detections[0]


def _frame_timing() -> FrameTimingV1:
    return FrameTimingV1(
        identity=FrameIdentityV1("camera0", 1, 7),
        camera_source_time_ns=123_456,
        host_clock_id="host-monotonic-geometry-test",
        publication_sequence=3,
        first_unique_packet_monotonic_ns=1_000,
        final_unique_packet_monotonic_ns=1_010,
        reassembly_complete_monotonic_ns=1_010,
        decode_start_monotonic_ns=1_011,
        decode_end_monotonic_ns=1_020,
        publish_monotonic_ns=1_021,
    )


def _authority() -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id="geometry-test-session",
        reset_epoch=2,
        gate_epoch=1,
        expected_gate_index=1,
        race_status_sequence=8,
        race_status_boot_ms=2_000,
        camera_host_clock_id="host-monotonic-geometry-test",
        camera_stream_id="camera0",
        camera_generation=1,
        frame_publication_sequence_not_before=3,
        frame_publish_monotonic_ns_not_before=1_021,
    )


def _fallback_covariance() -> FeatureCovarianceV1:
    return FeatureCovarianceV1(
        model_id="geometry-test-bbox-center-v1",
        feature_order=("center_x_norm", "center_y_norm"),
        matrix=((0.02, 0.0), (0.0, 0.02)),
    )


def _fitted_observation(image: np.ndarray) -> GateObservationV1:
    height, width = image.shape[:2]
    return gate_detection_with_aperture_to_observation_v1(
        _aperture_detection(image),
        image,
        frame_timing=_frame_timing(),
        authority=_authority(),
        candidate_id="gate-1-candidate-0",
        measurement_uncertainty_ns=33_333_334,
        fallback_center_covariance=_fallback_covariance(),
        image_width=width,
        image_height=height,
    )


def _crosses(corners: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    return tuple(
        (corners[(i + 1) % 4][0] - corners[i][0])
        * (corners[(i + 2) % 4][1] - corners[(i + 1) % 4][1])
        - (corners[(i + 1) % 4][1] - corners[i][1])
        * (corners[(i + 2) % 4][0] - corners[(i + 1) % 4][0])
        for i in range(4)
    )


@pytest.mark.parametrize(
    "gate_hsv",
    [
        (2, 220, 255),    # conventional bright red
        (177, 90, 250),   # red wraparound with reduced saturation
        (165, 60, 250),   # pink/magenta pulse seen in live VQ2 frames
    ],
)
def test_detects_both_vq2_gates_across_animation_hues(
    gate_hsv: tuple[int, int, int],
) -> None:
    detector = VQ2GateDetector()

    detections = detector.detect(_vq2_scene(gate_hsv))

    assert [d.bbox for d in detections] == [
        (282, 134, 80, 80),
        (410, 138, 27, 45),
    ]
    assert all(d.detection_method == "preset_contour" for d in detections)


def test_uses_only_fast_preset_strategy() -> None:
    detector = VQ2GateDetector()

    assert detector.enable_preset is True
    assert detector.enable_edge is False
    assert detector.enable_dynamic is False


def test_ignores_cyan_raceline_and_small_red_signal() -> None:
    detector = VQ2GateDetector()

    detections = detector.detect(_vq2_scene((165, 60, 250)))

    assert len(detections) == 2
    assert all(d.center_x > 250 for d in detections)


def test_visible_inner_aperture_fit_is_deterministic_and_supported() -> None:
    image = _aperture_scene((70, 50, 130, 110))
    detection = _aperture_detection(image)

    first = fit_vq2_aperture_bgr(
        image, detection.bbox, detection_confidence=detection.confidence
    )
    second = fit_vq2_aperture_bgr(
        image, detection.bbox, detection_confidence=detection.confidence
    )

    assert first == second
    assert first.succeeded
    assert first.rejection_reason is None
    assert first.geometry_model_id == "vq2-visible-inner-quad-lines-v1"
    assert first.visible_edges == (
        ApertureSide.LEFT
        | ApertureSide.TOP
        | ApertureSide.RIGHT
        | ApertureSide.BOTTOM
    )
    assert first.visible_corners == (True, True, True, True)
    assert first.inlier_count == first.support_count
    assert first.inlier_count > 100
    assert first.residual_rms_px == pytest.approx(0.0, abs=1e-12)
    assert np.asarray(first.fitted_corners_px) == pytest.approx(
        np.asarray(((69.5, 49.5), (130.5, 49.5), (130.5, 110.5), (69.5, 110.5)))
    )
    assert first.fitted_corners_px is not None
    assert all(value > 0.0 for value in _crosses(first.fitted_corners_px))


def test_visible_inner_fit_yields_conservative_passage_geometry() -> None:
    image = _aperture_scene((70, 50, 130, 110))
    detection = _aperture_detection(image)
    fit = fit_vq2_aperture_bgr(
        image,
        detection.bbox,
        detection_confidence=detection.confidence,
    )

    geometry = passage_geometry_from_vq2_aperture_fit(fit)

    assert geometry is not None
    assert geometry.center_norm == pytest.approx((0.0, 0.0), abs=1e-12)
    assert geometry.aperture_half_size_norm == pytest.approx(
        (0.305, 0.38125),
        abs=1e-12,
    )
    assert geometry.log_scale == pytest.approx(
        0.5 * math.log(0.305 * 0.38125),
    )
    assert geometry.measurement_std == pytest.approx(
        tuple(math.sqrt(value) for value in fit.covariance_diagonal[:3])
    )


@pytest.mark.parametrize(
    ("inner_bounds", "clipped", "expected_visible", "inferred_corner_indexes"),
    [
        (
            (70, -12, 130, 48),
            ApertureSide.TOP,
            ApertureSide.LEFT | ApertureSide.RIGHT | ApertureSide.BOTTOM,
            (0, 1),
        ),
        (
            (70, 112, 130, 172),
            ApertureSide.BOTTOM,
            ApertureSide.LEFT | ApertureSide.TOP | ApertureSide.RIGHT,
            (2, 3),
        ),
        (
            (-12, 50, 48, 110),
            ApertureSide.LEFT,
            ApertureSide.TOP | ApertureSide.RIGHT | ApertureSide.BOTTOM,
            (0, 3),
        ),
        (
            (152, 50, 212, 110),
            ApertureSide.RIGHT,
            ApertureSide.LEFT | ApertureSide.TOP | ApertureSide.BOTTOM,
            (1, 2),
        ),
    ],
)
def test_single_clipped_side_is_censored_with_ordered_convex_fit(
    inner_bounds: tuple[int, int, int, int],
    clipped: ApertureSide,
    expected_visible: ApertureSide,
    inferred_corner_indexes: tuple[int, int],
) -> None:
    image = _aperture_scene(inner_bounds)
    detection = _aperture_detection(image)

    fit = fit_vq2_aperture_bgr(
        image, detection.bbox, detection_confidence=detection.confidence
    )
    observation = _fitted_observation(image)

    assert fit.succeeded
    assert fit.geometry_model_id == "vq2-censored-image-square-px-v1"
    assert fit.clipping & clipped
    assert fit.visible_edges == expected_visible
    assert fit.fitted_corners_px is not None
    assert all(value > 0.0 for value in _crosses(fit.fitted_corners_px))
    assert (
        0.5 * (fit.fitted_corners_px[0][1] + fit.fitted_corners_px[1][1])
        < 0.5 * (fit.fitted_corners_px[3][1] + fit.fitted_corners_px[2][1])
    )
    assert (
        0.5 * (fit.fitted_corners_px[0][0] + fit.fitted_corners_px[3][0])
        < 0.5 * (fit.fitted_corners_px[1][0] + fit.fitted_corners_px[2][0])
    )
    for index in inferred_corner_indexes:
        assert fit.visible_corners[index] is False
    for index in set(range(4)) - set(inferred_corner_indexes):
        assert fit.visible_corners[index] is True
    assert GateObservationV1.from_primitive(observation.to_primitive()) == observation
    assert observation.clipping & FrameEdge(int(clipped))
    assert observation.inner_edges.visibility == FrameEdge(int(expected_visible))
    assert observation.fitted_inner_aperture_corners_norm is not None
    for index in inferred_corner_indexes:
        assert observation.inner_corners_norm[index] is None
    for index in set(range(4)) - set(inferred_corner_indexes):
        assert (
            observation.inner_corners_norm[index]
            == observation.fitted_inner_aperture_corners_norm[index]
        )


def test_clipping_increases_uncertainty_and_never_claims_nominal_health() -> None:
    visible_image = _aperture_scene((70, 50, 130, 110))
    clipped_image = _aperture_scene((70, -12, 130, 48))
    visible_fit = fit_vq2_aperture_bgr(
        visible_image,
        _aperture_detection(visible_image).bbox,
        detection_confidence=0.9,
    )
    clipped_fit = fit_vq2_aperture_bgr(
        clipped_image,
        _aperture_detection(clipped_image).bbox,
        detection_confidence=0.9,
    )

    assert visible_fit.covariance_diagonal is not None
    assert clipped_fit.covariance_diagonal is not None
    assert all(
        clipped > visible
        for clipped, visible in zip(
            clipped_fit.covariance_diagonal,
            visible_fit.covariance_diagonal,
        )
    )
    assert clipped_fit.confidence < visible_fit.confidence
    visible = _fitted_observation(visible_image)
    clipped = _fitted_observation(clipped_image)
    assert visible.health is ObservationHealth.NOMINAL
    assert visible.health_reason is None
    assert clipped.health is ObservationHealth.DEGRADED
    assert clipped.health_reason == "censored_image_aperture:top"


def test_fitted_observation_binds_visible_and_inferred_corner_semantics() -> None:
    observation = _fitted_observation(_aperture_scene((70, -12, 130, 48)))

    assert GateObservationV1.from_primitive(observation.to_primitive()) == observation
    assert observation.clipping & FrameEdge.TOP
    assert observation.outer_edges.visibility == FrameEdge.NONE
    assert observation.inner_edges.visibility == (
        FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.BOTTOM
    )
    assert observation.inner_edges.top is None
    assert observation.inner_corners_norm[:2] == (None, None)
    assert observation.fitted_inner_aperture_corners_norm is not None
    assert observation.inner_corners_norm[2:] == (
        observation.fitted_inner_aperture_corners_norm[2],
        observation.fitted_inner_aperture_corners_norm[3],
    )
    assert observation.log_scale is not None
    assert observation.projective_skew is not None
    assert observation.covariance.feature_order == (
        "center_x_norm",
        "center_y_norm",
        "log_scale",
        "skew_x",
        "skew_y",
    )
    assert observation.fit.inlier_count > 0
    assert observation.fit.inlier_count <= observation.fit.support_count


def test_two_missing_clipped_sides_are_rejected_without_invented_precision() -> None:
    image = _aperture_scene((-12, -12, 48, 48))
    detection = _aperture_detection(image)

    fit = fit_vq2_aperture_bgr(
        image, detection.bbox, detection_confidence=detection.confidence
    )
    observation = _fitted_observation(image)

    assert fit.clipping == (ApertureSide.LEFT | ApertureSide.TOP)
    assert not fit.succeeded
    assert fit.rejection_reason == "underconstrained_inner_aperture"
    assert observation.fitted_inner_aperture_corners_norm is None
    assert observation.log_scale is None
    assert observation.projective_skew is None


def test_complete_but_low_confidence_fit_is_degraded() -> None:
    image = _aperture_scene((70, 50, 130, 110))
    detection = _aperture_detection(image)
    detection.confidence = 0.1

    observation = gate_detection_with_aperture_to_observation_v1(
        detection,
        image,
        frame_timing=_frame_timing(),
        authority=_authority(),
        candidate_id="low-confidence-candidate",
        measurement_uncertainty_ns=33_333_334,
        fallback_center_covariance=_fallback_covariance(),
        image_width=200,
        image_height=160,
    )

    assert observation.fitted_inner_aperture_corners_norm is not None
    assert observation.health is ObservationHealth.DEGRADED
    assert observation.health_reason == "low_confidence_image_aperture"


def test_low_confidence_or_clipped_fit_cannot_claim_passage_geometry() -> None:
    visible_image = _aperture_scene((70, 50, 130, 110))
    clipped_image = _aperture_scene((70, -12, 130, 48))
    visible_detection = _aperture_detection(visible_image)
    clipped_detection = _aperture_detection(clipped_image)
    low_confidence = fit_vq2_aperture_bgr(
        visible_image,
        visible_detection.bbox,
        detection_confidence=0.1,
    )
    clipped = fit_vq2_aperture_bgr(
        clipped_image,
        clipped_detection.bbox,
        detection_confidence=clipped_detection.confidence,
    )
    outer_support_touch = dataclasses.replace(
        low_confidence,
        clipping=ApertureSide.TOP,
    )

    assert low_confidence.succeeded
    assert clipped.succeeded
    assert passage_geometry_from_vq2_aperture_fit(low_confidence) is None
    assert (
        passage_geometry_from_vq2_aperture_fit(
            low_confidence,
            minimum_confidence=0.0,
        )
        is None
    )
    assert passage_geometry_from_vq2_aperture_fit(clipped) is None
    assert (
        passage_geometry_from_vq2_aperture_fit(outer_support_touch)
        is None
    )
    degraded_tracking = tracking_geometry_from_vq2_aperture_fit(
        low_confidence
    )
    assert degraded_tracking is not None
    assert degraded_tracking.center_norm == pytest.approx((0.0, 0.0))
    assert degraded_tracking.log_scale == pytest.approx(
        0.5
        * math.log(
            degraded_tracking.aperture_half_size_norm[0]
            * degraded_tracking.aperture_half_size_norm[1]
        )
    )
    confidence_multiplier = math.sqrt(0.25 / low_confidence.confidence)
    assert low_confidence.covariance_diagonal is not None
    assert degraded_tracking.measurement_std == pytest.approx(
        tuple(
            math.sqrt(value) * confidence_multiplier
            for value in low_confidence.covariance_diagonal[:3]
        )
    )
    assert tracking_geometry_from_vq2_aperture_fit(clipped) is None
    assert (
        tracking_geometry_from_vq2_aperture_fit(outer_support_touch)
        is not None
    )
    assert (
        tracking_geometry_from_vq2_aperture_fit(
            dataclasses.replace(
                low_confidence,
                visible_edges=(
                    ApertureSide.LEFT
                    | ApertureSide.TOP
                    | ApertureSide.RIGHT
                ),
            )
        )
        is None
    )
    assert (
        tracking_geometry_from_vq2_aperture_fit(
            dataclasses.replace(
                low_confidence,
                geometry_model_id="unreviewed-inner-model",
            )
        )
        is None
    )


def test_perspective_inner_quad_remains_convex_and_corner_ordered() -> None:
    image = np.full((180, 240, 3), 18, dtype=np.uint8)
    color = _bgr_from_hsv(165, 100, 250)
    outer = np.asarray(((45, 28), (196, 40), (184, 153), (55, 146)), np.int32)
    inner = np.asarray(((68, 51), (171, 59), (162, 127), (76, 122)), np.int32)
    cv2.fillConvexPoly(image, outer, color)
    cv2.fillConvexPoly(image, inner, (18, 18, 18))
    detection = _aperture_detection(image)

    observation = gate_detection_with_aperture_to_observation_v1(
        detection,
        image,
        frame_timing=_frame_timing(),
        authority=_authority(),
        candidate_id="perspective-candidate",
        measurement_uncertainty_ns=33_333_334,
        fallback_center_covariance=_fallback_covariance(),
        image_width=240,
        image_height=180,
    )

    assert observation.health is ObservationHealth.NOMINAL
    assert observation.fitted_inner_aperture_corners_norm is not None
    assert all(
        value > 0.0 for value in _crosses(observation.fitted_inner_aperture_corners_norm)
    )
    assert observation.projective_skew != pytest.approx((0.0, 0.0), abs=1e-3)


def test_multiple_similar_aperture_gaps_are_rejected_as_ambiguous() -> None:
    mask = np.zeros((120, 180), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (160, 100), 255, 8)
    cv2.rectangle(mask, (86, 20), (94, 100), 255, -1)

    fit = fit_vq2_aperture_mask(
        mask,
        (16, 16, 149, 89),
        detection_confidence=0.9,
    )

    assert not fit.succeeded
    assert fit.rejection_reason == "ambiguous_multiple_aperture_gaps"
    assert fit.fitted_corners_px is None
    assert fit.covariance_diagonal is None


def test_unique_temporal_gap_association_is_tracking_only() -> None:
    mask = np.zeros((120, 180), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (160, 100), 255, 8)
    cv2.rectangle(mask, (86, 20), (94, 100), 255, -1)
    prior = VQ2ApertureTrackingPrior(
        center_px=(55.0, 60.0),
        # Extent is deliberately stale: identity comes from the one fresh gap
        # containing the predicted center, while current pixels refit extent.
        half_size_px=(15.0, 20.0),
        maximum_boundary_residual_px=0.1,
    )

    fit = fit_vq2_aperture_mask(
        mask,
        (16, 16, 149, 89),
        detection_confidence=0.9,
        tracking_prior=prior,
    )

    assert fit.succeeded
    assert fit.rejection_reason is None
    assert fit.geometry_model_id == (
        "vq2-temporally-associated-inner-quad-lines-v1"
    )
    assert fit.covariance_model_id == (
        "vq2-temporally-associated-aperture-diagonal-v1"
    )
    assert fit.fitted_corners_px is not None
    assert np.asarray(fit.fitted_corners_px) == pytest.approx(
        np.asarray(
            (
                (24.5, 24.5),
                (85.5, 24.5),
                (85.5, 95.5),
                (24.5, 95.5),
            )
        )
    )
    assert tracking_geometry_from_vq2_aperture_fit(fit) is not None
    assert (
        passage_geometry_from_vq2_aperture_fit(
            fit,
            minimum_confidence=0.0,
        )
        is None
    )


def test_temporal_prior_requires_fresh_quad_to_contain_predicted_center() -> None:
    prior = VQ2ApertureTrackingPrior(
        center_px=(90.0, 60.0),
        half_size_px=(30.0, 40.0),
        maximum_boundary_residual_px=1.0,
    )
    perspective_quad = (
        (50.0, 20.0),
        (130.0, 20.0),
        (110.0, 100.0),
        (70.0, 100.0),
    )

    assert vq2_geometry_module._quad_contains_tracking_prior_center(
        perspective_quad,
        prior,
    )


@pytest.mark.parametrize(
    "prior",
    [
        VQ2ApertureTrackingPrior(
            center_px=(90.0, 60.0),
            half_size_px=(30.5, 35.5),
            maximum_boundary_residual_px=36.0,
        ),
        VQ2ApertureTrackingPrior(
            center_px=(90.0, 60.0),
            half_size_px=(30.5, 35.5),
            maximum_boundary_residual_px=2.0,
        ),
    ],
)
def test_nonunique_or_incoherent_temporal_gap_prior_fails_closed(
    prior: VQ2ApertureTrackingPrior,
) -> None:
    mask = np.zeros((120, 180), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (160, 100), 255, 8)
    cv2.rectangle(mask, (86, 20), (94, 100), 255, -1)

    fit = fit_vq2_aperture_mask(
        mask,
        (16, 16, 149, 89),
        detection_confidence=0.9,
        tracking_prior=prior,
    )

    assert not fit.succeeded
    assert fit.rejection_reason == "ambiguous_multiple_aperture_gaps"
    assert fit.fitted_corners_px is None
    assert fit.covariance_diagonal is None


def test_tracking_prior_does_not_change_an_ordinary_unambiguous_fit() -> None:
    image = _aperture_scene((70, 50, 130, 110))
    detection = _aperture_detection(image)
    ordinary = fit_vq2_aperture_bgr(
        image,
        detection.bbox,
        detection_confidence=detection.confidence,
    )

    with_irrelevant_prior = fit_vq2_aperture_bgr(
        image,
        detection.bbox,
        detection_confidence=detection.confidence,
        tracking_prior=VQ2ApertureTrackingPrior(
            center_px=(10.0, 10.0),
            half_size_px=(2.0, 2.0),
            maximum_boundary_residual_px=1.0,
        ),
    )

    assert with_irrelevant_prior == ordinary
    assert ordinary.geometry_model_id == "vq2-visible-inner-quad-lines-v1"
    assert ordinary.covariance_model_id == (
        "vq2-visible-aperture-diagonal-v1"
    )


def test_connected_current_successor_union_fits_dominant_opening() -> None:
    """Merged current+successor supports now fit the dominant opening.

    Deliberate behaviour change from the enclosed-region disambiguation
    (F80 panel gates): the proposal names the larger enclosed opening —
    here the near gate's — and the fit is clean enough to claim passage.
    Previously this fixture survived only because the hybrid two-opening
    residuals kept confidence below the passage floor.  Comparable paired
    openings remain rejected; see
    ``test_multiple_similar_aperture_gaps_are_rejected_as_ambiguous``.
    """
    mask = np.zeros((160, 300), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (200, 140), 255, 10)
    cv2.rectangle(mask, (198, 32), (282, 112), 255, 8)
    cv2.rectangle(mask, (195, 30), (205, 48), 255, -1)

    fit = fit_vq2_aperture_mask(
        mask,
        (15, 15, 272, 130),
        detection_confidence=0.9,
    )

    assert fit.succeeded
    assert fit.fitted_corners_px is not None
    center_x = sum(point[0] for point in fit.fitted_corners_px) / 4.0
    center_y = sum(point[1] for point in fit.fitted_corners_px) / 4.0
    # The dominant (near gate) opening spans x 25-195, y 25-135.
    assert center_x == pytest.approx(110.0, abs=10.0)
    assert center_y == pytest.approx(80.0, abs=10.0)
    assert passage_geometry_from_vq2_aperture_fit(fit) is not None


def test_panel_gate_below_opening_slit_fits_true_opening() -> None:
    """F80 gate-1 topology: ring, solid sponsor panel, comparable slit.

    Columns crossing the panel see two comparable dark gaps (opening and
    below-panel slit), which the legacy competing-gap test must reject as
    ambiguous.  The single dominant enclosed region — the opening, bounded
    above by the top bar — resolves the ambiguity and the fit locks the
    true opening with passage-usable confidence.
    """
    mask = np.zeros((260, 220), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (180, 40), 255, -1)
    cv2.rectangle(mask, (20, 40), (40, 120), 255, -1)
    cv2.rectangle(mask, (160, 40), (180, 120), 255, -1)
    cv2.rectangle(mask, (20, 120), (180, 160), 255, -1)
    cv2.rectangle(mask, (20, 160), (60, 220), 255, -1)
    cv2.rectangle(mask, (150, 160), (180, 220), 255, -1)
    cv2.rectangle(mask, (20, 220), (180, 240), 255, -1)

    fit = fit_vq2_aperture_mask(
        mask,
        (18, 18, 164, 224),
        detection_confidence=0.9,
    )

    assert fit.succeeded
    assert np.asarray(fit.fitted_corners_px) == pytest.approx(
        np.asarray(
            (
                (40.5, 40.5),
                (159.5, 40.5),
                (159.5, 119.5),
                (40.5, 119.5),
            )
        ),
        abs=2.5,
    )
    assert passage_geometry_from_vq2_aperture_fit(fit) is not None


def test_enclosed_pylon_window_without_top_bar_is_not_an_aperture() -> None:
    """F80 f1640441 topology: the true opening leaks through the below-panel
    slit, so the only enclosed dark region is the pylon's sponsor window.
    The window is enclosed and dominant, but it is not bounded above by the
    component's top bar, so the region proposal must refuse it and the fit
    must fall through to the legacy ambiguity rejection rather than claim a
    usable aperture on solid structure.
    """
    mask = np.zeros((280, 240), dtype=np.uint8)
    cv2.rectangle(mask, (60, 20), (200, 40), 255, -1)
    cv2.rectangle(mask, (60, 40), (90, 120), 255, -1)
    cv2.rectangle(mask, (170, 40), (200, 120), 255, -1)
    cv2.rectangle(mask, (60, 120), (140, 150), 255, -1)
    cv2.rectangle(mask, (60, 150), (100, 240), 255, -1)
    cv2.rectangle(mask, (170, 150), (200, 240), 255, -1)
    cv2.rectangle(mask, (60, 240), (200, 258), 255, -1)
    cv2.rectangle(mask, (20, 100), (50, 240), 255, -1)
    cv2.rectangle(mask, (20, 120), (60, 150), 255, -1)
    cv2.rectangle(mask, (28, 160), (44, 200), 0, -1)

    fit = fit_vq2_aperture_mask(
        mask,
        (18, 18, 184, 242),
        detection_confidence=0.9,
    )

    assert not fit.succeeded
    assert fit.rejection_reason == "ambiguous_multiple_aperture_gaps"
    assert fit.fitted_corners_px is None
    assert passage_geometry_from_vq2_aperture_fit(fit) is None


def test_degenerate_solid_support_falls_back_without_scale_or_skew() -> None:
    image = np.full((160, 200, 3), 18, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (140, 120), _bgr_from_hsv(165, 100, 250), -1)
    detection = _aperture_detection(image)

    observation = gate_detection_with_aperture_to_observation_v1(
        detection,
        image,
        frame_timing=_frame_timing(),
        authority=_authority(),
        candidate_id="solid-rejected",
        measurement_uncertainty_ns=33_333_334,
        fallback_center_covariance=_fallback_covariance(),
        image_width=200,
        image_height=160,
    )

    assert observation.health is ObservationHealth.DEGRADED
    assert observation.health_reason == "aperture_fit_rejected:underconstrained_inner_aperture"
    assert observation.provenance == "vq2_aperture_fit_rejected"
    assert observation.fitted_inner_aperture_corners_norm is None
    assert observation.inner_corners_norm == (None, None, None, None)
    assert observation.log_scale is None
    assert observation.projective_skew is None
    assert observation.covariance == _fallback_covariance()


def test_two_large_disconnected_components_do_not_select_one_silently() -> None:
    mask = np.zeros((100, 180), dtype=np.uint8)
    cv2.rectangle(mask, (10, 20), (70, 80), 255, 6)
    cv2.rectangle(mask, (105, 20), (165, 80), 255, 6)

    fit = fit_vq2_aperture_mask(
        mask,
        (7, 17, 162, 67),
        detection_confidence=0.8,
    )

    assert not fit.succeeded
    assert fit.rejection_reason == "ambiguous_connected_components"


def test_gate_support_must_match_the_declared_detection_bbox() -> None:
    mask = np.zeros((120, 180), dtype=np.uint8)
    cv2.rectangle(mask, (60, 35), (120, 95), 255, 7)

    fit = fit_vq2_aperture_mask(
        mask,
        (0, 0, 180, 120),
        detection_confidence=0.8,
    )

    assert not fit.succeeded
    assert fit.rejection_reason == "connected_support_does_not_match_bbox"
    assert fit.fitted_corners_px is None


def test_legacy_bbox_adapter_remains_exact_and_ignores_placeholder_geometry() -> None:
    image = _aperture_scene((70, 50, 130, 110))
    detection = _aperture_detection(image)
    detection.corners = np.full((4, 2), 999_999, dtype=np.int32)
    detection.estimated_distance = 0.001
    arguments = {
        "frame_timing": _frame_timing(),
        "authority": _authority(),
        "candidate_id": "legacy-candidate",
        "measurement_uncertainty_ns": 33_333_334,
        "center_covariance": _fallback_covariance(),
        "image_width": 200,
        "image_height": 160,
    }

    before = gate_detection_to_observation_v1(detection, **arguments)
    _ = gate_detection_with_aperture_to_observation_v1(
        detection,
        image,
        frame_timing=arguments["frame_timing"],
        authority=arguments["authority"],
        candidate_id=arguments["candidate_id"],
        measurement_uncertainty_ns=arguments["measurement_uncertainty_ns"],
        fallback_center_covariance=arguments["center_covariance"],
        image_width=200,
        image_height=160,
    )
    after = gate_detection_to_observation_v1(detection, **arguments)

    assert before == after
    assert before.fitted_inner_aperture_corners_norm is None
    assert before.log_scale is None
    assert before.projective_skew is None
    assert observation_to_legacy_gate_target_fields(
        before,
        frame_timing=_frame_timing(),
        expected_authority=_authority(),
    ) == {
        "frame_id": 7,
        "sim_time_ns": 123_456,
        "received_monotonic_s": 1.01e-6,
        "center_x": 100,
        "center_y": 80,
        "bbox": (60, 40, 81, 81),
        "confidence": detection.confidence,
    }


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda mask: fit_vq2_aperture_mask(
            mask.astype(np.float32), (0, 0, 10, 10), detection_confidence=0.5
        ),
        lambda mask: fit_vq2_aperture_mask(
            mask, (0, 0, 10, 10), detection_confidence=math.nan
        ),
        lambda mask: fit_vq2_aperture_mask(
            mask, (0, 0, 10, 10), detection_confidence=True
        ),
        lambda mask: fit_vq2_aperture_mask(
            mask, (0, 0, 11, 10), detection_confidence=0.5
        ),
    ],
)
def test_geometry_rejects_ambiguous_or_invalid_numeric_inputs(bad_call) -> None:
    with pytest.raises((TypeError, ValueError)):
        bad_call(np.zeros((10, 10), dtype=np.uint8))


def test_geometry_config_rejects_even_morphology_kernel() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        dataclasses.replace(VQ2ApertureConfig(), morph_kernel_size=2)


def test_mask_generation_is_binary_deterministic_and_does_not_modify_input() -> None:
    image = _aperture_scene((70, 50, 130, 110))
    original = image.copy()

    first = vq2_gate_mask_from_bgr(image)
    second = vq2_gate_mask_from_bgr(image)

    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= {0, 255}
    assert np.array_equal(image, original)
