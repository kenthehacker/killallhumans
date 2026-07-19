"""Cross-workstream checks for the offline VQ2 Wave 1 perception path."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from competition.adapter import CameraFrame
from competition.vq2_contracts import (
    FeatureCovarianceV1,
    FrameIdentityV1,
    FrameTimingV1,
    GateAuthorityEpochV1,
    validate_relative_gate_state_source,
)
from competition.vq2_runtime import LatestFrameCursorV1, VQ2_HOST_CLOCK_ID
from competition.vq2_vision import VQ2VisionSnapshot
from estimation.vq2_relative_estimator import (
    MissingApertureScaleError,
    RelativePredictionTarget,
    VQ2RelativeGateEstimator,
)
from gate_detection.src.vq2_detector import VQ2GateDetector
from gate_detection.src.vq2_observation_adapter import (
    gate_detection_to_observation_v1,
    gate_detection_with_aperture_to_observation_v1,
)


_STREAM_ID = "camera0"
_FRAME_ID = 7
_GENERATION = 1
_PUBLICATION_SEQUENCE = 3
_SOURCE_TIME_NS = 123_456


def _gate_image() -> np.ndarray:
    image = np.full((160, 200, 3), 18, dtype=np.uint8)
    hsv = np.uint8([[[165, 100, 250]]])
    gate_color = tuple(
        int(channel) for channel in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    )
    cv2.rectangle(image, (50, 30), (150, 130), gate_color, -1)
    cv2.rectangle(image, (60, 40), (140, 120), (18, 18, 18), -1)
    image.flags.writeable = False
    return image


def _timing() -> FrameTimingV1:
    return FrameTimingV1(
        identity=FrameIdentityV1(_STREAM_ID, _GENERATION, _FRAME_ID),
        camera_source_time_ns=_SOURCE_TIME_NS,
        host_clock_id=VQ2_HOST_CLOCK_ID,
        publication_sequence=_PUBLICATION_SEQUENCE,
        first_unique_packet_monotonic_ns=1_000_000,
        final_unique_packet_monotonic_ns=1_010_000,
        reassembly_complete_monotonic_ns=1_010_000,
        decode_start_monotonic_ns=1_011_000,
        decode_end_monotonic_ns=1_020_000,
        publish_monotonic_ns=1_021_000,
    )


def _authority() -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id="wave1-integration-session",
        reset_epoch=2,
        gate_epoch=1,
        expected_gate_index=1,
        race_status_sequence=8,
        race_status_boot_ms=2_000,
        camera_host_clock_id=VQ2_HOST_CLOCK_ID,
        camera_stream_id=_STREAM_ID,
        camera_generation=_GENERATION,
        frame_publication_sequence_not_before=_PUBLICATION_SEQUENCE,
        frame_publish_monotonic_ns_not_before=1_021_000,
    )


def _center_covariance() -> FeatureCovarianceV1:
    return FeatureCovarianceV1(
        model_id="wave1-integration-bbox-center-v1",
        feature_order=("center_x_norm", "center_y_norm"),
        matrix=((0.02, 0.0), (0.0, 0.02)),
    )


def _selected_frame():
    image = _gate_image()
    timing = _timing()
    snapshot = VQ2VisionSnapshot(
        frame_id=_FRAME_ID,
        camera_frame=CameraFrame(
            timestamp_us=_SOURCE_TIME_NS // 1_000,
            image=image,
            width=200,
            height=160,
        ),
        sim_time_ns=_SOURCE_TIME_NS,
        received_monotonic_s=timing.final_unique_packet_monotonic_ns / 1e9,
        generation=_GENERATION,
        timing=timing,
    )
    cursor = LatestFrameCursorV1(
        expected_host_clock_id=VQ2_HOST_CLOCK_ID,
        expected_stream_id=_STREAM_ID,
    )
    selection = cursor.select(snapshot)
    assert selection is not None
    assert cursor.select(snapshot) is None
    return selection


def _detect(selection):
    frame = selection.snapshot.camera_frame
    detections = VQ2GateDetector(
        image_width=frame.width,
        image_height=frame.height,
        min_area=100,
    ).detect(frame.image)
    assert len(detections) == 1
    return frame, detections[0]


def test_timed_frame_aperture_fit_sources_relative_state() -> None:
    selection = _selected_frame()
    frame, detection = _detect(selection)
    observation = gate_detection_with_aperture_to_observation_v1(
        detection,
        frame.image,
        frame_timing=selection.timing,
        authority=_authority(),
        candidate_id="gate-1-candidate-0",
        measurement_uncertainty_ns=33_333_334,
        fallback_center_covariance=_center_covariance(),
        image_width=frame.width,
        image_height=frame.height,
    )

    assert observation.fitted_inner_aperture_corners_norm is not None
    assert observation.log_scale is not None
    estimator = VQ2RelativeGateEstimator("wave1-active-track")
    update = estimator.update(
        observation,
        RelativePredictionTarget.at_decision(
            VQ2_HOST_CLOCK_ID,
            selection.timing.publish_monotonic_ns,
        ),
    )

    assert update.measurement_accepted
    assert update.state.metric_position_body_frd_m is None
    validate_relative_gate_state_source(update.state, observation)


def test_bbox_only_detection_is_withheld_from_relative_estimator() -> None:
    selection = _selected_frame()
    frame, detection = _detect(selection)
    observation = gate_detection_to_observation_v1(
        detection,
        frame_timing=selection.timing,
        authority=_authority(),
        candidate_id="gate-1-bbox-only",
        measurement_uncertainty_ns=33_333_334,
        center_covariance=_center_covariance(),
        image_width=frame.width,
        image_height=frame.height,
    )

    assert observation.log_scale is None
    with pytest.raises(MissingApertureScaleError, match="fitted inner-aperture"):
        VQ2RelativeGateEstimator("wave1-active-track").update(
            observation,
            RelativePredictionTarget.at_decision(
                VQ2_HOST_CLOCK_ID,
                selection.timing.publish_monotonic_ns,
            ),
        )
