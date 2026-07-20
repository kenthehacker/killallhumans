from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from competition.vq2_contracts import (
    EdgeSetV1,
    FeatureCovarianceV1,
    FitDiagnosticsV1,
    FrameEdge,
    FrameIdentityV1,
    FrameTimingV1,
    GateAuthorityEpochV1,
    GateObservationV1,
    LineSegmentV1,
    MeasurementTimeBasis,
    ObservationHealth,
    PredictionBasis,
    RelativeStateHealth,
    TrackRole,
)
from estimation.vq2_relative_estimator import (
    EstimatorUnavailableError,
    MissingApertureScaleError,
    PredictionHorizonError,
    RelativeEstimatorConfig,
    RelativeEstimatorError,
    RelativePredictionTarget,
    StaleObservationError,
    VQ2ImuCorrelatedEstimatorCoast,
    VQ2ImuCorrelatedEstimatorUpdate,
    VQ2RelativeGateEstimator,
)
from estimation.vq2_imu_derotation import (
    SUPPORTED_CAMERA_RAY_MODEL_ID,
    VQ2AttitudeDerotationInput,
    VQ2CameraToBodyCalibration,
    VQ2DerotationModel,
    derotate_gate_observation,
)
from estimation.vq2_imu_provenance import VQ2ImuSource, VQ2TimestampedAttitude


_HOST_CLOCK_ID = "host-monotonic-test"
_STREAM_ID = "camera0"
_BASE_NS = 10_000_000_000


def _frame_timing(
    *,
    frame_id: int,
    measurement_offset_ns: int,
    publication_sequence: int | None = None,
    generation: int = 1,
    host_clock_id: str = _HOST_CLOCK_ID,
) -> FrameTimingV1:
    measurement_ns = _BASE_NS + measurement_offset_ns
    publication_sequence = (
        frame_id + 1 if publication_sequence is None else publication_sequence
    )
    return FrameTimingV1(
        identity=FrameIdentityV1(_STREAM_ID, generation, frame_id),
        camera_source_time_ns=1_000_000 + measurement_offset_ns,
        host_clock_id=host_clock_id,
        publication_sequence=publication_sequence,
        first_unique_packet_monotonic_ns=measurement_ns - 1_000,
        final_unique_packet_monotonic_ns=measurement_ns,
        reassembly_complete_monotonic_ns=measurement_ns,
        decode_start_monotonic_ns=measurement_ns + 100_000,
        decode_end_monotonic_ns=measurement_ns + 200_000,
        publish_monotonic_ns=measurement_ns + 300_000,
    )


def _authority(
    *,
    reset_epoch: int = 1,
    gate_epoch: int = 0,
    gate_index: int = 0,
    generation: int = 1,
    race_status_sequence: int = 10,
    race_status_boot_ms: int = 1_000,
    cutover_publication_sequence: int = 1,
    cutover_publish_ns: int = _BASE_NS,
    host_clock_id: str = _HOST_CLOCK_ID,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id="training-session-test",
        reset_epoch=reset_epoch,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=race_status_sequence,
        race_status_boot_ms=race_status_boot_ms,
        camera_host_clock_id=host_clock_id,
        camera_stream_id=_STREAM_ID,
        camera_generation=generation,
        frame_publication_sequence_not_before=cutover_publication_sequence,
        frame_publish_monotonic_ns_not_before=cutover_publish_ns,
    )


def _visible_segment(
    first: tuple[float, float],
    second: tuple[float, float],
) -> LineSegmentV1 | None:
    if all(-1.0 <= value <= 1.0 for point in (first, second) for value in point):
        return LineSegmentV1(first, second)
    return None


def _observation(
    *,
    frame_id: int,
    measurement_offset_ns: int,
    center: tuple[float, float] = (0.0, 0.0),
    log_scale: float = math.log(0.4),
    measurement_variance: float = 1e-4,
    confidence: float = 1.0,
    clipping: FrameEdge | None = None,
    health: ObservationHealth | None = None,
    health_reason: str | None = None,
    publication_sequence: int | None = None,
    generation: int = 1,
    authority: GateAuthorityEpochV1 | None = None,
    candidate_id: str | None = None,
    host_clock_id: str = _HOST_CLOCK_ID,
    fitted: bool = True,
) -> GateObservationV1:
    timing = _frame_timing(
        frame_id=frame_id,
        measurement_offset_ns=measurement_offset_ns,
        publication_sequence=publication_sequence,
        generation=generation,
        host_clock_id=host_clock_id,
    )
    authority = authority or _authority(host_clock_id=host_clock_id)
    if not fitted:
        return GateObservationV1(
            frame_timing=timing,
            measurement_time_monotonic_ns=timing.final_unique_packet_monotonic_ns,
            measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
            measurement_time_model_id=None,
            measurement_uncertainty_ns=1_000_000,
            authority=authority,
            candidate_id=candidate_id or f"gate-candidate-{frame_id}",
            image_size_px=(640, 360),
            center_norm=center,
            support_bounds_norm=(0.4, 0.4, 0.6, 0.6),
            outer_edges=EdgeSetV1(),
            inner_edges=EdgeSetV1(),
            inner_corners_norm=(None, None, None, None),
            fitted_inner_aperture_corners_norm=None,
            geometry_model_id=None,
            log_scale=None,
            projective_skew=None,
            clipping=FrameEdge.NONE,
            confidence=confidence,
            covariance=FeatureCovarianceV1(
                "bbox-center-only-v1",
                ("center_x_norm", "center_y_norm"),
                ((measurement_variance, 0.0), (0.0, measurement_variance)),
            ),
            fit=FitDiagnosticsV1(None, 0, 0),
            health=ObservationHealth.DEGRADED,
            health_reason="no_fitted_aperture",
            provenance="synthetic-test",
        )

    half_extent = math.exp(log_scale) * 0.5
    cx, cy = center
    top_left = (cx - half_extent, cy - half_extent)
    top_right = (cx + half_extent, cy - half_extent)
    bottom_right = (cx + half_extent, cy + half_extent)
    bottom_left = (cx - half_extent, cy + half_extent)
    fitted_corners = (top_left, top_right, bottom_right, bottom_left)
    required_clipping = FrameEdge.NONE
    for x, y in fitted_corners:
        if x < -1.0:
            required_clipping |= FrameEdge.LEFT
        elif x > 1.0:
            required_clipping |= FrameEdge.RIGHT
        if y < -1.0:
            required_clipping |= FrameEdge.TOP
        elif y > 1.0:
            required_clipping |= FrameEdge.BOTTOM
    clipping = required_clipping if clipping is None else clipping
    visible_corners = tuple(
        corner if all(-1.0 <= value <= 1.0 for value in corner) else None
        for corner in fitted_corners
    )
    inner_edges = EdgeSetV1(
        left=_visible_segment(top_left, bottom_left),
        top=_visible_segment(top_left, top_right),
        right=_visible_segment(top_right, bottom_right),
        bottom=_visible_segment(bottom_left, bottom_right),
    )
    clamped_x = [min(1.0, max(-1.0, point[0])) for point in fitted_corners]
    clamped_y = [min(1.0, max(-1.0, point[1])) for point in fitted_corners]
    left = (min(clamped_x) + 1.0) * 0.5
    right = (max(clamped_x) + 1.0) * 0.5
    top = (min(clamped_y) + 1.0) * 0.5
    bottom = (max(clamped_y) + 1.0) * 0.5
    if left == right:
        left, right = ((0.99, 1.0) if right == 1.0 else (0.0, 0.01))
    if top == bottom:
        top, bottom = ((0.99, 1.0) if bottom == 1.0 else (0.0, 0.01))
    if health is None:
        health = (
            ObservationHealth.NOMINAL
            if clipping is FrameEdge.NONE
            else ObservationHealth.DEGRADED
        )
    if health is ObservationHealth.NOMINAL:
        health_reason = None
    elif health_reason is None:
        health_reason = "clipped_or_degraded_geometry"
    covariance_order = (
        "center_x_norm",
        "center_y_norm",
        "log_scale",
        "skew_x",
        "skew_y",
    )
    covariance = tuple(
        tuple(
            measurement_variance if row == column else 0.0
            for column in range(len(covariance_order))
        )
        for row in range(len(covariance_order))
    )
    return GateObservationV1(
        frame_timing=timing,
        measurement_time_monotonic_ns=timing.final_unique_packet_monotonic_ns,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=1_000_000,
        authority=authority,
        candidate_id=candidate_id or f"gate-candidate-{frame_id}",
        image_size_px=(640, 360),
        center_norm=center,
        support_bounds_norm=(left, top, right, bottom),
        outer_edges=EdgeSetV1(),
        inner_edges=inner_edges,
        inner_corners_norm=visible_corners,
        fitted_inner_aperture_corners_norm=fitted_corners,
        geometry_model_id="synthetic-aperture-v1",
        log_scale=log_scale,
        projective_skew=(0.0, 0.0),
        clipping=clipping,
        confidence=confidence,
        covariance=FeatureCovarianceV1(
            "synthetic-aperture-covariance-v1",
            covariance_order,
            covariance,
        ),
        fit=FitDiagnosticsV1(0.001, 4, 4),
        health=health,
        health_reason=health_reason,
        provenance="synthetic-test",
    )


def _decision_target(observation: GateObservationV1) -> RelativePredictionTarget:
    return RelativePredictionTarget.at_decision(
        observation.host_clock_id,
        observation.frame_timing.publish_monotonic_ns,
    )


def _forecast_target(
    observation: GateObservationV1,
    *,
    forecast_from_measurement_ns: int,
) -> RelativePredictionTarget:
    return RelativePredictionTarget(
        host_clock_id=observation.host_clock_id,
        decision_time_monotonic_ns=observation.frame_timing.publish_monotonic_ns,
        prediction_time_monotonic_ns=(
            observation.measurement_time_monotonic_ns
            + forecast_from_measurement_ns
        ),
        prediction_basis=PredictionBasis.COMMAND_EFFECT_ESTIMATE,
        delay_model_id="synthetic-command-effect-v1",
        delay_uncertainty_ns=1_000_000,
    )


def _coast_target(monotonic_ns: int, host_clock_id: str = _HOST_CLOCK_ID):
    return RelativePredictionTarget.at_decision(host_clock_id, monotonic_ns)


def _derotation_attitude(
    observation: GateObservationV1,
    *,
    sequence: int,
    receive_monotonic_ns: int,
    yaw_rad: float = 0.0,
) -> VQ2AttitudeDerotationInput:
    half = yaw_rad * 0.5
    return VQ2AttitudeDerotationInput(
        attitude=VQ2TimestampedAttitude(
            source=VQ2ImuSource(
                session_id=observation.authority.session_id,
                reset_epoch=observation.authority.reset_epoch,
                host_clock_id=observation.host_clock_id,
                stream_id="highres-imu-test",
                generation=1,
            ),
            sample_sequence=sequence,
            source_time_us=1_000_000 + sequence * 10_000,
            receive_monotonic_ns=receive_monotonic_ns,
            orientation_body_to_ned_wxyz=(
                math.cos(half),
                0.0,
                0.0,
                math.sin(half),
            ),
            body_rates_rad_s=(0.0, 0.0, 0.0),
            gyro_bias_rad_s=(0.0, 0.0, 0.0),
            accel_trust=1.0,
            propagated=True,
        ),
        orientation_uncertainty_rad=0.001,
        host_time_uncertainty_ns=100_000,
    )


def _derotation_evidence(
    observation: GateObservationV1,
    *,
    target_yaw_rad: float = 0.05,
):
    target = _decision_target(observation)
    return derotate_gate_observation(
        observation,
        target,
        capture_attitude=_derotation_attitude(
            observation,
            sequence=10,
            receive_monotonic_ns=observation.measurement_time_monotonic_ns,
        ),
        target_attitude=_derotation_attitude(
            observation,
            sequence=11,
            receive_monotonic_ns=target.prediction_time_monotonic_ns,
            yaw_rad=target_yaw_rad,
        ),
        calibration=VQ2CameraToBodyCalibration(
            calibration_id="synthetic-camera-body-test-v1",
            camera_ray_model_id=SUPPORTED_CAMERA_RAY_MODEL_ID,
            camera_to_body_wxyz=(1.0, 0.0, 0.0, 0.0),
            rotation_uncertainty_rad=0.001,
        ),
        model=VQ2DerotationModel(
            model_id="synthetic-derotation-test-v1",
            attitude_time_model_id="synthetic-host-aligned-attitude-v1",
            max_capture_alignment_ns=20_000_000,
            max_target_extrapolation_ns=20_000_000,
            max_total_timing_uncertainty_ns=50_000_000,
            angular_rate_uncertainty_rad_s=0.01,
        ),
    )


def _first_imu_correlated_coast():
    first_observation = _observation(
        frame_id=0,
        measurement_offset_ns=0,
        center=(0.20, -0.10),
    )
    estimator = VQ2RelativeGateEstimator("correlated-coast-filter")
    estimator.update_with_imu_correlation(
        _derotation_evidence(first_observation, target_yaw_rad=0.0)
    )
    observation = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        center=(0.20, -0.10),
    )
    prior = estimator.update_with_imu_correlation(
        _derotation_evidence(observation, target_yaw_rad=0.0)
    )
    target = RelativePredictionTarget.at_decision(
        observation.host_clock_id,
        prior.state.timing.prediction_time_monotonic_ns + 20_000_000,
    )
    evidence = derotate_gate_observation(
        observation,
        target,
        capture_attitude=prior.evidence.capture_attitude,
        target_attitude=_derotation_attitude(
            observation,
            sequence=12,
            receive_monotonic_ns=target.prediction_time_monotonic_ns,
            yaw_rad=0.02,
        ),
        calibration=prior.evidence.calibration,
        model=prior.evidence.model,
    )
    state = estimator.coast(target)
    coast = VQ2ImuCorrelatedEstimatorCoast(
        prior_update=prior,
        state=state,
        evidence=evidence,
    )
    return coast, estimator


def _assert_psd(state) -> None:
    covariance = np.asarray(state.covariance.matrix, dtype=np.float64)
    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12, rtol=0.0)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_first_update_emits_exact_feature_contract_without_metric_pose():
    observation = _observation(frame_id=0, measurement_offset_ns=0)
    estimator = VQ2RelativeGateEstimator("active-gate-filter")

    result = estimator.update(observation, _decision_target(observation))

    assert result.measurement_accepted is True
    assert result.state.source_candidate_id == observation.candidate_id
    assert result.state.state_sequence == 0
    assert result.state.measurement_update_sequence == 0
    assert result.state.health is RelativeStateHealth.INITIALIZING
    assert result.state.metric_position_body_frd_m is None
    assert result.state.metric_velocity_body_frd_m_s is None
    assert result.state.metric_gate_orientation_body_frd_xyzw is None
    assert result.state.metric_covariance is None
    assert result.state.log_scale == pytest.approx(observation.log_scale, abs=1e-12)
    _assert_psd(result.state)


def test_irregular_timing_prediction_beats_latest_measurement_at_future_time():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    offsets_ns = (0, 27_000_000, 61_000_000, 103_000_000, 149_000_000)
    noises = (0.004, -0.003, 0.002, -0.002, 0.001)
    velocity_x = 0.65
    velocity_y = -0.25
    expansion_rate = 0.35
    base_log_scale = math.log(0.35)
    states = []

    for index, (offset_ns, noise) in enumerate(zip(offsets_ns, noises)):
        time_s = offset_ns / 1e9
        observation = _observation(
            frame_id=index,
            measurement_offset_ns=offset_ns,
            center=(
                -0.25 + velocity_x * time_s + noise,
                0.20 + velocity_y * time_s - noise * 0.5,
            ),
            log_scale=base_log_scale + expansion_rate * time_s + noise * 0.2,
            measurement_variance=2.5e-5,
        )
        target = (
            _forecast_target(observation, forecast_from_measurement_ns=80_000_000)
            if index == len(offsets_ns) - 1
            else _decision_target(observation)
        )
        states.append(estimator.update(observation, target).state)

    future_s = (offsets_ns[-1] + 80_000_000) / 1e9
    true_future_x = -0.25 + velocity_x * future_s
    latest_measurement_error = abs(
        (-0.25 + velocity_x * offsets_ns[-1] / 1e9 + noises[-1])
        - true_future_x
    )
    predicted_error = abs(states[-1].bearing_norm[0] - true_future_x)

    assert predicted_error < latest_measurement_error
    assert states[-1].bearing_rate_norm_s[0] > 0.0
    assert states[-1].expansion_rate_s > 0.0
    assert [state.measurement_update_sequence for state in states] == list(range(5))
    assert states[-1].health is RelativeStateHealth.HEALTHY
    for state in states:
        _assert_psd(state)


def test_innovation_rejection_coasts_without_advancing_measurement_update():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    first = _observation(frame_id=0, measurement_offset_ns=0, center=(0.0, 0.0))
    second = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        center=(0.01, -0.005),
    )
    estimator.update(first, _decision_target(first))
    accepted = estimator.update(second, _decision_target(second)).state
    outlier = _observation(
        frame_id=2,
        measurement_offset_ns=60_000_000,
        center=(0.85, -0.75),
        measurement_variance=1e-6,
    )

    rejected = estimator.update(outlier, _decision_target(outlier))

    assert rejected.measurement_accepted is False
    assert rejected.normalized_innovation_squared > rejected.innovation_gate_threshold
    assert rejected.state.source_candidate_id == accepted.source_candidate_id
    assert (
        rejected.state.measurement_update_sequence
        == accepted.measurement_update_sequence
    )
    assert rejected.state.state_sequence == accepted.state_sequence + 1
    assert rejected.state.dropout_count == 1
    assert rejected.state.health is RelativeStateHealth.COASTING
    assert rejected.state.normalized_innovation_squared is None
    assert rejected.state.innovation_gate_threshold is None
    assert rejected.state.innovation_accepted is None
    with pytest.raises(StaleObservationError, match="already processed"):
        estimator.update(outlier, _decision_target(outlier))


def test_control_rate_coast_is_bounded_and_covariance_grows_to_lost():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    with pytest.raises(EstimatorUnavailableError, match="initialization"):
        estimator.coast(_coast_target(_BASE_NS))
    first = _observation(frame_id=0, measurement_offset_ns=0)
    second = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        center=(0.02, 0.0),
    )
    estimator.update(first, _decision_target(first))
    accepted = estimator.update(second, _decision_target(second)).state

    coasting = estimator.coast(_coast_target(_BASE_NS + 130_000_000))
    lost = estimator.coast(_coast_target(_BASE_NS + 230_000_000))

    assert coasting.health is RelativeStateHealth.COASTING
    assert lost.health is RelativeStateHealth.LOST
    assert (coasting.dropout_count, lost.dropout_count) == (1, 2)
    assert lost.source_candidate_id == accepted.source_candidate_id
    assert lost.measurement_update_sequence == accepted.measurement_update_sequence
    assert np.trace(np.asarray(coasting.covariance.matrix)) > np.trace(
        np.asarray(accepted.covariance.matrix)
    )
    assert np.trace(np.asarray(lost.covariance.matrix)) > np.trace(
        np.asarray(coasting.covariance.matrix)
    )
    _assert_psd(coasting)
    _assert_psd(lost)
    with pytest.raises(StaleObservationError, match="advance"):
        estimator.coast(_coast_target(_BASE_NS + 230_000_000))
    with pytest.raises(EstimatorUnavailableError, match="expired"):
        estimator.coast(_coast_target(_BASE_NS + 430_000_001))


def test_clock_identity_and_measurement_relative_horizon_fail_closed():
    observation = _observation(frame_id=0, measurement_offset_ns=0)
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    wrong_clock_target = RelativePredictionTarget.at_decision(
        "other-host-clock",
        observation.frame_timing.publish_monotonic_ns,
    )
    with pytest.raises(RelativeEstimatorError, match="host clock"):
        estimator.update(observation, wrong_clock_target)

    late_decision = RelativePredictionTarget.at_decision(
        _HOST_CLOCK_ID,
        observation.measurement_time_monotonic_ns + 150_000_000,
    )
    with pytest.raises(PredictionHorizonError, match="measurement-relative"):
        estimator.update(observation, late_decision)
    assert estimator.is_initialized is False

    accepted = estimator.update(observation, _decision_target(observation)).state
    foreign_observation = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        publication_sequence=2,
        host_clock_id="other-host-clock",
    )
    with pytest.raises(RelativeEstimatorError, match="estimator clock"):
        estimator.update(
            foreign_observation,
            _decision_target(foreign_observation),
        )
    with pytest.raises(RelativeEstimatorError, match="host clock"):
        estimator.coast(
            _coast_target(
                accepted.timing.prediction_time_monotonic_ns + 20_000_000,
                "other-host-clock",
            )
        )
    too_long_target = RelativePredictionTarget(
        host_clock_id=_HOST_CLOCK_ID,
        decision_time_monotonic_ns=(
            accepted.timing.prediction_time_monotonic_ns + 20_000_000
        ),
        prediction_time_monotonic_ns=(
            accepted.timing.prediction_time_monotonic_ns + 120_000_001
        ),
        prediction_basis=PredictionBasis.COMMAND_EFFECT_ESTIMATE,
        delay_model_id="too-long-delay-v1",
        delay_uncertainty_ns=1,
    )
    with pytest.raises(PredictionHorizonError, match="requested"):
        estimator.coast(too_long_target)


def test_bbox_only_observation_is_withheld_instead_of_fabricating_scale():
    observation = _observation(
        frame_id=0,
        measurement_offset_ns=0,
        fitted=False,
    )
    estimator = VQ2RelativeGateEstimator("active-gate-filter")

    with pytest.raises(MissingApertureScaleError, match="inner-aperture"):
        estimator.update(observation, _decision_target(observation))

    assert estimator.last_state is None
    assert estimator.is_initialized is False


def test_clipping_and_low_confidence_inflate_covariance_and_degrade_health():
    nominal = _observation(
        frame_id=0,
        measurement_offset_ns=0,
        confidence=1.0,
    )
    degraded = _observation(
        frame_id=0,
        measurement_offset_ns=0,
        confidence=0.25,
        clipping=FrameEdge.RIGHT,
        health=ObservationHealth.DEGRADED,
        health_reason="right_edge_clipped",
    )

    nominal_state = VQ2RelativeGateEstimator("nominal-filter").update(
        nominal, _decision_target(nominal)
    ).state
    degraded_state = VQ2RelativeGateEstimator("degraded-filter").update(
        degraded, _decision_target(degraded)
    ).state

    assert degraded_state.health is RelativeStateHealth.DEGRADED
    assert degraded_state.last_clipping is FrameEdge.RIGHT
    for index in range(3):
        assert (
            degraded_state.covariance.matrix[index][index]
            > nominal_state.covariance.matrix[index][index]
        )
    _assert_psd(degraded_state)


def test_rate_clipping_is_explicit_and_preserves_psd_covariance():
    config = RelativeEstimatorConfig(
        max_abs_bearing_rate_norm_s=0.1,
        max_abs_expansion_rate_s=0.1,
    )
    estimator = VQ2RelativeGateEstimator(
        "rate-limited-filter",
        config=config,
    )
    first = _observation(frame_id=0, measurement_offset_ns=0)
    second = _observation(
        frame_id=1,
        measurement_offset_ns=50_000_000,
        center=(0.08, 0.0),
        log_scale=math.log(0.4) + 0.08,
    )
    estimator.update(first, _decision_target(first))

    state = estimator.update(second, _decision_target(second)).state

    assert state.health is RelativeStateHealth.DEGRADED
    assert state.health_reason == "feature_rate_limited"
    assert abs(state.bearing_rate_norm_s[0]) <= 0.1
    assert abs(state.expansion_rate_s) <= 0.1
    _assert_psd(state)


def test_long_measurement_gap_reinitializes_instead_of_extrapolating():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    first = _observation(frame_id=0, measurement_offset_ns=0)
    after_gap = _observation(
        frame_id=1,
        measurement_offset_ns=300_000_000,
        center=(0.5, -0.3),
    )
    estimator.update(first, _decision_target(first))

    state = estimator.update(after_gap, _decision_target(after_gap)).state

    assert state.health is RelativeStateHealth.INITIALIZING
    assert state.health_reason == "measurement_gap_reinitialized"
    assert state.bearing_rate_norm_s == (0.0, 0.0)
    assert state.expansion_rate_s == 0.0
    assert state.measurement_update_sequence == 1


def test_duplicate_replayed_and_nonadvancing_frames_are_rejected():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    first = _observation(frame_id=0, measurement_offset_ns=0)
    estimator.update(first, _decision_target(first))
    with pytest.raises(StaleObservationError, match="already processed"):
        estimator.update(first, _decision_target(first))

    same_frame_other_candidate = dataclasses.replace(
        first,
        candidate_id="same-frame-other-candidate",
    )
    with pytest.raises(StaleObservationError, match="already processed"):
        estimator.update(
            same_frame_other_candidate,
            _decision_target(same_frame_other_candidate),
        )

    fresh_estimator = VQ2RelativeGateEstimator("active-gate-filter")
    newer = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        publication_sequence=2,
    )
    older = _observation(
        frame_id=0,
        measurement_offset_ns=0,
        publication_sequence=1,
    )
    fresh_estimator.update(newer, _decision_target(newer))
    with pytest.raises(StaleObservationError, match="publication sequence"):
        fresh_estimator.update(older, _decision_target(older))


def test_distinct_frame_cannot_regress_the_control_decision_time():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    first = _observation(frame_id=0, measurement_offset_ns=0)
    estimator.update(
        first,
        RelativePredictionTarget.at_decision(
            _HOST_CLOCK_ID,
            first.measurement_time_monotonic_ns + 80_000_000,
        ),
    )
    second = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        publication_sequence=2,
    )
    regressed_decision = RelativePredictionTarget(
        host_clock_id=_HOST_CLOCK_ID,
        decision_time_monotonic_ns=_BASE_NS + 40_000_000,
        prediction_time_monotonic_ns=_BASE_NS + 90_000_000,
        prediction_basis=PredictionBasis.COMMAND_EFFECT_ESTIMATE,
        delay_model_id="synthetic-command-effect-v1",
        delay_uncertainty_ns=1_000_000,
    )

    with pytest.raises(StaleObservationError, match="decision time"):
        estimator.update(second, regressed_decision)


def test_gate_and_reset_authority_transitions_reinitialize_without_replay_amnesia():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    first = _observation(
        frame_id=0,
        measurement_offset_ns=0,
        publication_sequence=1,
    )
    first_state = estimator.update(first, _decision_target(first)).state

    gate_timing = _frame_timing(
        frame_id=1,
        measurement_offset_ns=40_000_000,
        publication_sequence=2,
    )
    gate_authority = _authority(
        gate_epoch=1,
        gate_index=1,
        race_status_sequence=11,
        race_status_boot_ms=1_100,
        cutover_publication_sequence=2,
        cutover_publish_ns=gate_timing.publish_monotonic_ns,
    )
    next_gate = _observation(
        frame_id=1,
        measurement_offset_ns=40_000_000,
        publication_sequence=2,
        authority=gate_authority,
        center=(0.3, -0.2),
    )
    gate_state = estimator.update(next_gate, _decision_target(next_gate)).state

    assert (gate_state.state_sequence, gate_state.measurement_update_sequence) == (0, 0)
    assert gate_state.bearing_rate_norm_s == (0.0, 0.0)
    assert gate_state.authority.expected_gate_index == 1

    reset_timing = _frame_timing(
        frame_id=0,
        measurement_offset_ns=80_000_000,
        publication_sequence=3,
        generation=2,
    )
    reset_authority = _authority(
        reset_epoch=2,
        gate_epoch=0,
        gate_index=0,
        generation=2,
        race_status_sequence=12,
        race_status_boot_ms=1_200,
        cutover_publication_sequence=3,
        cutover_publish_ns=reset_timing.publish_monotonic_ns,
    )
    after_reset = _observation(
        frame_id=0,
        measurement_offset_ns=80_000_000,
        publication_sequence=3,
        generation=2,
        authority=reset_authority,
        center=(-0.1, 0.1),
    )
    reset_state = estimator.update(
        after_reset,
        _decision_target(after_reset),
    ).state

    assert (reset_state.state_sequence, reset_state.measurement_update_sequence) == (0, 0)
    assert reset_state.bearing_rate_norm_s == (0.0, 0.0)
    assert reset_state.authority.reset_epoch == 2
    with pytest.raises(StaleObservationError, match="already processed"):
        estimator.update(first, _decision_target(first))
    assert first_state.source_candidate_id == first.candidate_id


def test_cross_session_authority_requires_a_new_estimator_instance():
    estimator = VQ2RelativeGateEstimator("active-gate-filter")
    first = _observation(frame_id=0, measurement_offset_ns=0)
    accepted = estimator.update(first, _decision_target(first)).state
    other_session_authority = dataclasses.replace(
        first.authority,
        session_id="replacement-training-session",
        race_status_sequence=first.authority.race_status_sequence + 1,
        race_status_boot_ms=first.authority.race_status_boot_ms + 1,
    )
    other_session = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        publication_sequence=2,
        authority=other_session_authority,
    )

    with pytest.raises(RelativeEstimatorError, match="safety session"):
        estimator.update(other_session, _decision_target(other_session))

    assert estimator.last_state is accepted


def test_same_replay_produces_bitwise_identical_contract_primitives():
    observations = tuple(
        _observation(
            frame_id=index,
            measurement_offset_ns=offset_ns,
            center=(0.1 + 0.4 * offset_ns / 1e9, -0.2),
            log_scale=math.log(0.3) + 0.2 * offset_ns / 1e9,
        )
        for index, offset_ns in enumerate((0, 31_000_000, 70_000_000, 116_000_000))
    )

    def run_replay():
        estimator = VQ2RelativeGateEstimator("deterministic-filter")
        trace = [
            estimator.update(observation, _decision_target(observation)).state.to_primitive()
            for observation in observations
        ]
        trace.append(
            estimator.coast(_coast_target(_BASE_NS + 150_000_000)).to_primitive()
        )
        return trace

    assert run_replay() == run_replay()


def test_imu_correlated_update_preserves_raw_camera_filter_basis_and_source():
    observation = _observation(
        frame_id=0,
        measurement_offset_ns=0,
        center=(0.20, -0.10),
    )
    evidence = _derotation_evidence(observation, target_yaw_rad=0.05)
    estimator = VQ2RelativeGateEstimator("correlated-filter")
    ordinary_estimator = VQ2RelativeGateEstimator("correlated-filter")

    result = estimator.update_with_imu_correlation(evidence)
    ordinary = ordinary_estimator.update(
        observation,
        evidence.prediction_target,
    )

    assert type(result) is VQ2ImuCorrelatedEstimatorUpdate
    assert result.evidence is evidence
    assert result.current_observation_accepted
    assert not result.derotation_applied_to_state
    assert result.estimator_update.measurement_accepted
    assert result.estimator_update == ordinary
    assert result.state.bearing_norm == observation.center_norm
    assert result.state.bearing_norm != evidence.derotated_center_norm
    assert result.state.source_candidate_id == observation.candidate_id
    assert result.state.timing.source_frame == observation.frame
    assert result.state.timing.measurement_time_monotonic_ns == (
        observation.measurement_time_monotonic_ns
    )
    assert result.state.timing.measurement_time_basis is (
        observation.measurement_time_basis
    )
    _assert_psd(result.state)


def test_imu_correlated_rejection_does_not_relabel_retained_camera_source():
    estimator = VQ2RelativeGateEstimator("correlated-filter")
    first = _observation(frame_id=0, measurement_offset_ns=0, center=(0.0, 0.0))
    second = _observation(
        frame_id=1,
        measurement_offset_ns=30_000_000,
        center=(0.01, 0.0),
    )
    accepted = estimator.update_with_imu_correlation(
        _derotation_evidence(first, target_yaw_rad=0.0)
    )
    accepted = estimator.update_with_imu_correlation(
        _derotation_evidence(second, target_yaw_rad=0.0)
    )
    outlier = _observation(
        frame_id=2,
        measurement_offset_ns=60_000_000,
        center=(0.85, -0.75),
        measurement_variance=1e-6,
    )
    evidence = _derotation_evidence(outlier, target_yaw_rad=0.0)

    rejected = estimator.update_with_imu_correlation(evidence)

    assert not rejected.current_observation_accepted
    assert not rejected.derotation_applied_to_state
    assert not rejected.estimator_update.measurement_accepted
    assert rejected.evidence is evidence
    assert rejected.state.source_candidate_id == accepted.state.source_candidate_id
    assert rejected.state.source_candidate_id != outlier.candidate_id
    assert rejected.state.measurement_update_sequence == (
        accepted.state.measurement_update_sequence
    )
    with pytest.raises(StaleObservationError, match="already processed"):
        estimator.update_with_imu_correlation(evidence)


def test_imu_correlated_update_requires_exact_evidence_before_mutating_estimator():
    estimator = VQ2RelativeGateEstimator("correlated-filter")
    with pytest.raises(TypeError, match="exact VQ2DerotationEvidence"):
        estimator.update_with_imu_correlation(object())
    assert estimator.last_state is None
    assert not estimator.is_initialized


def test_first_imu_correlated_coast_retains_source_and_advances_only_prediction():
    coast, estimator = _first_imu_correlated_coast()
    prior = coast.prior_update

    assert estimator.last_state is coast.state
    assert coast.evidence.observation is prior.evidence.observation
    assert coast.evidence.capture_attitude is prior.evidence.capture_attitude
    assert coast.state.timing.source_frame == prior.state.timing.source_frame
    assert coast.state.source_candidate_id == prior.state.source_candidate_id
    assert coast.state.tracker_id == prior.state.tracker_id
    assert coast.state.track_role is prior.state.track_role
    assert coast.state.authority == prior.state.authority
    assert coast.state.state_sequence == prior.state.state_sequence + 1
    assert (
        coast.state.measurement_update_sequence
        == prior.state.measurement_update_sequence
    )
    assert coast.state.dropout_count == 1
    assert coast.state.health is RelativeStateHealth.COASTING
    assert coast.state.health_reason == "observation_dropout"
    assert coast.state.normalized_innovation_squared is None
    assert coast.state.innovation_gate_threshold is None
    assert coast.state.innovation_accepted is None
    assert coast.state.covariance != prior.state.covariance
    assert not coast.derotation_applied_to_state
    coast.validate_integrity()


def test_first_imu_correlated_coast_requires_an_active_prior_state():
    coast, _estimator = _first_imu_correlated_coast()
    shadow_prior_state = dataclasses.replace(
        coast.prior_update.state,
        track_role=TrackRole.SHADOW,
    )
    shadow_prior = dataclasses.replace(
        coast.prior_update,
        estimator_update=dataclasses.replace(
            coast.prior_update.estimator_update,
            state=shadow_prior_state,
        ),
    )
    shadow_coast_state = dataclasses.replace(
        coast.state,
        track_role=TrackRole.SHADOW,
    )

    with pytest.raises(ValueError, match="healthy active non-dropout"):
        VQ2ImuCorrelatedEstimatorCoast(
            prior_update=shadow_prior,
            state=shadow_coast_state,
            evidence=coast.evidence,
        )


@pytest.mark.parametrize(
    "forgery",
    (
        "tracker",
        "role",
        "camera_source",
        "unchanged_covariance",
        "state_sequence",
        "measurement_sequence",
        "dropout_count",
        "health",
        "health_reason",
    ),
)
def test_first_imu_correlated_coast_rejects_state_envelope_forgery(forgery: str):
    coast, _estimator = _first_imu_correlated_coast()
    state = coast.state
    changes = {
        "tracker": {"tracker_id": "forged-tracker"},
        "role": {"track_role": TrackRole.SHADOW},
        "camera_source": {"source_candidate_id": "forged-candidate"},
        "unchanged_covariance": {"covariance": coast.prior_update.state.covariance},
        "state_sequence": {"state_sequence": state.state_sequence + 1},
        "measurement_sequence": {
            "measurement_update_sequence": state.measurement_update_sequence + 1
        },
        "dropout_count": {"dropout_count": 2},
        "health": {"health": RelativeStateHealth.LOST},
        "health_reason": {"health_reason": "forged_dropout_reason"},
    }[forgery]
    forged = dataclasses.replace(state, **changes)

    with pytest.raises(ValueError):
        VQ2ImuCorrelatedEstimatorCoast(
            prior_update=coast.prior_update,
            state=forged,
            evidence=coast.evidence,
        )


def test_first_imu_correlated_coast_rejects_compensated_variance_underreporting():
    coast, _estimator = _first_imu_correlated_coast()
    prior_diagonal = tuple(
        coast.prior_update.state.covariance.matrix[index][index]
        for index in range(6)
    )
    coast_diagonal = tuple(
        coast.state.covariance.matrix[index][index] for index in range(6)
    )
    forged_diagonal = (
        prior_diagonal[0] * 0.5,
        prior_diagonal[1] * 0.5,
        prior_diagonal[2] * 0.5,
        coast_diagonal[3] + 1.0,
        coast_diagonal[4] + 1.0,
        coast_diagonal[5] + 1.0,
    )
    covariance = dataclasses.replace(
        coast.state.covariance,
        matrix=tuple(
            tuple(
                forged_diagonal[row] if row == column else 0.0
                for column in range(6)
            )
            for row in range(6)
        ),
    )
    forged = dataclasses.replace(coast.state, covariance=covariance)

    with pytest.raises(ValueError, match="grow conservatively"):
        VQ2ImuCorrelatedEstimatorCoast(
            prior_update=coast.prior_update,
            state=forged,
            evidence=coast.evidence,
        )


def test_first_imu_correlated_coast_rejects_metric_state_injection():
    coast, _estimator = _first_imu_correlated_coast()
    metric_covariance = FeatureCovarianceV1(
        model_id="forged-coast-metric-v1",
        feature_order=(
            "position_x_body_frd_m",
            "position_y_body_frd_m",
            "position_z_body_frd_m",
            "velocity_x_body_frd_m_s",
            "velocity_y_body_frd_m_s",
            "velocity_z_body_frd_m_s",
            "orientation_error_x_rad",
            "orientation_error_y_rad",
            "orientation_error_z_rad",
        ),
        matrix=tuple(
            tuple(1.0 if row == column else 0.0 for column in range(9))
            for row in range(9)
        ),
    )
    forged = dataclasses.replace(
        coast.state,
        metric_position_body_frd_m=(1.0, 2.0, 3.0),
        metric_velocity_body_frd_m_s=(4.0, 5.0, 6.0),
        metric_gate_orientation_body_frd_xyzw=(0.0, 0.0, 0.0, 1.0),
        metric_covariance=metric_covariance,
    )

    with pytest.raises(ValueError, match="unsupported metric state"):
        VQ2ImuCorrelatedEstimatorCoast(
            prior_update=coast.prior_update,
            state=forged,
            evidence=coast.evidence,
        )


def test_first_imu_correlated_coast_rejects_changed_attitude_uncertainty_model():
    coast, _estimator = _first_imu_correlated_coast()
    target_attitude = dataclasses.replace(
        coast.evidence.target_attitude,
        orientation_uncertainty_rad=(
            coast.evidence.target_attitude.orientation_uncertainty_rad * 2.0
        ),
    )
    changed_evidence = derotate_gate_observation(
        coast.evidence.observation,
        coast.evidence.prediction_target,
        capture_attitude=coast.evidence.capture_attitude,
        target_attitude=target_attitude,
        calibration=coast.evidence.calibration,
        model=coast.evidence.model,
    )

    with pytest.raises(ValueError, match="uncertainty model"):
        VQ2ImuCorrelatedEstimatorCoast(
            prior_update=coast.prior_update,
            state=coast.state,
            evidence=changed_evidence,
        )


def test_first_imu_correlated_coast_revalidates_nested_covariance_integrity():
    coast, _estimator = _first_imu_correlated_coast()
    matrix = [list(row) for row in coast.state.covariance.matrix]
    matrix[0][1] = float("nan")
    matrix[1][0] = float("nan")
    object.__setattr__(
        coast.state.covariance,
        "matrix",
        tuple(tuple(row) for row in matrix),
    )

    with pytest.raises(ValueError):
        coast.validate_integrity()
