"""Deterministic feature-space filtering for the build-3385 VQ2 path.

This module is deliberately offline and authority-neutral.  It consumes the
frozen observation contract, emits the frozen relative-state contract, and
never arms, resets, sends, declares passage, or fabricates metric pose.  The
filter operates only on fitted inner-aperture bearing/scale features; a legacy
bbox observation without ``log_scale`` is withheld rather than relabeled as
aperture geometry.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import numpy as np

from competition.vq2_contracts import (
    FeatureCovarianceV1,
    FrameEdge,
    GateObservationV1,
    ObservationHealth,
    PredictionBasis,
    PredictionTimeV1,
    RelativeGateStateV1,
    RelativeStateHealth,
    TrackRole,
    validate_relative_gate_state_sequence,
    validate_relative_gate_state_source,
)

if TYPE_CHECKING:
    from estimation.vq2_imu_derotation import VQ2DerotationEvidence


_STATE_FEATURE_ORDER = (
    "bearing_x_norm",
    "bearing_y_norm",
    "log_scale",
    "bearing_rate_x_norm_s",
    "bearing_rate_y_norm_s",
    "expansion_rate_s",
)
_MEASUREMENT_FEATURE_ORDER = (
    "center_x_norm",
    "center_y_norm",
    "log_scale",
)
_MEASUREMENT_STATE_INDICES = (0, 1, 2)
_RATE_LIMITS_INDICES = (3, 4, 5)
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


class RelativeEstimatorError(ValueError):
    """Base class for deterministic, fail-closed estimator rejections."""


class MissingApertureScaleError(RelativeEstimatorError):
    """Raised when an observation has no fitted inner-aperture scale."""


class StaleObservationError(RelativeEstimatorError):
    """Raised for duplicate, replayed, or non-advancing camera observations."""


class PredictionHorizonError(RelativeEstimatorError):
    """Raised when a requested prediction exceeds the reviewed short horizon."""


class EstimatorUnavailableError(RelativeEstimatorError):
    """Raised when no bounded relative-state prediction remains available."""


def _finite_positive(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded token")
    return value


def _revalidate_relative_gate_state(
    state: RelativeGateStateV1,
) -> RelativeGateStateV1:
    """Reconstruct a state and every nested contract value."""

    source_frame = state.timing.source_frame
    if source_frame is None:
        raise ValueError("relative estimator state requires a source frame")
    timing = replace(
        state.timing,
        source_frame=replace(source_frame),
    )
    covariance = replace(state.covariance)
    metric_covariance = (
        None
        if state.metric_covariance is None
        else replace(state.metric_covariance)
    )
    return replace(
        state,
        timing=timing,
        authority=replace(state.authority),
        covariance=covariance,
        metric_covariance=metric_covariance,
    )


@dataclass(frozen=True, slots=True)
class RelativeEstimatorConfig:
    """Reviewed tuning surface for the small constant-velocity filter."""

    covariance_model_id: str = "vq2-feature-cv-kalman-v1"
    bearing_process_accel_std_norm_s2: float = 2.0
    scale_process_accel_std_s2: float = 2.5
    initial_bearing_rate_std_norm_s: float = 1.5
    initial_expansion_rate_std_s: float = 2.0
    minimum_measurement_variance: float = 1e-8
    covariance_eigenvalue_floor: float = 1e-12
    confidence_floor: float = 0.20
    clipping_variance_multiplier_per_edge: float = 1.5
    degraded_variance_multiplier: float = 2.0
    innovation_gate_threshold_3d: float = 11.344866730144373
    minimum_accepted_updates_for_healthy: int = 2
    max_measurement_gap_s: float = 0.250
    max_prediction_horizon_s: float = 0.100
    coast_lost_after_s: float = 0.150
    max_coast_s: float = 0.400
    dropout_variance_per_s: float = 0.25
    max_abs_bearing_rate_norm_s: float = 8.0
    max_abs_expansion_rate_s: float = 8.0

    def __post_init__(self) -> None:
        _bounded_token(self.covariance_model_id, "covariance_model_id")
        for name in (
            "bearing_process_accel_std_norm_s2",
            "scale_process_accel_std_s2",
            "initial_bearing_rate_std_norm_s",
            "initial_expansion_rate_std_s",
            "minimum_measurement_variance",
            "covariance_eigenvalue_floor",
            "confidence_floor",
            "clipping_variance_multiplier_per_edge",
            "degraded_variance_multiplier",
            "innovation_gate_threshold_3d",
            "max_measurement_gap_s",
            "max_prediction_horizon_s",
            "coast_lost_after_s",
            "max_coast_s",
            "max_abs_bearing_rate_norm_s",
            "max_abs_expansion_rate_s",
        ):
            _finite_positive(getattr(self, name), name)
        _finite_nonnegative(self.dropout_variance_per_s, "dropout_variance_per_s")
        _exact_nonnegative_int(
            self.minimum_accepted_updates_for_healthy,
            "minimum_accepted_updates_for_healthy",
        )
        if self.minimum_accepted_updates_for_healthy == 0:
            raise ValueError("minimum_accepted_updates_for_healthy must be positive")
        if self.confidence_floor > 1.0:
            raise ValueError("confidence_floor must be <= 1")
        if self.degraded_variance_multiplier < 1.0:
            raise ValueError("degraded_variance_multiplier must be >= 1")
        if self.coast_lost_after_s > self.max_coast_s:
            raise ValueError("coast_lost_after_s cannot exceed max_coast_s")


@dataclass(frozen=True, slots=True)
class RelativePredictionTarget:
    """Host-monotonic decision/prediction request; not a wire schema."""

    host_clock_id: str
    decision_time_monotonic_ns: int
    prediction_time_monotonic_ns: int
    prediction_basis: PredictionBasis = PredictionBasis.DECISION_TIME
    delay_model_id: Optional[str] = None
    delay_uncertainty_ns: int = 0

    def __post_init__(self) -> None:
        _bounded_token(self.host_clock_id, "host_clock_id")
        decision = _exact_nonnegative_int(
            self.decision_time_monotonic_ns, "decision_time_monotonic_ns"
        )
        prediction = _exact_nonnegative_int(
            self.prediction_time_monotonic_ns, "prediction_time_monotonic_ns"
        )
        _exact_nonnegative_int(self.delay_uncertainty_ns, "delay_uncertainty_ns")
        if prediction < decision:
            raise ValueError("prediction time cannot predate decision time")
        if type(self.prediction_basis) is not PredictionBasis:
            raise TypeError("prediction_basis must be PredictionBasis")
        if self.delay_model_id is not None:
            _bounded_token(self.delay_model_id, "delay_model_id")
        if self.prediction_basis is PredictionBasis.DECISION_TIME:
            if prediction != decision:
                raise ValueError("decision-time prediction must equal decision time")
            if self.delay_model_id is not None or self.delay_uncertainty_ns != 0:
                raise ValueError("decision-time prediction cannot claim delay")
        elif self.delay_model_id is None or self.delay_uncertainty_ns == 0:
            raise ValueError("estimated prediction requires model and uncertainty")

    @classmethod
    def at_decision(
        cls,
        host_clock_id: str,
        monotonic_ns: int,
    ) -> "RelativePredictionTarget":
        return cls(host_clock_id, monotonic_ns, monotonic_ns)


@dataclass(frozen=True, slots=True)
class RelativeEstimatorUpdate:
    """One local update outcome plus its contract state.

    Rejected innovations are diagnostics about the newly observed frame.  The
    returned contract state coasts from the last accepted source and therefore
    does not relabel the rejected frame as an applied measurement update.
    """

    state: RelativeGateStateV1
    measurement_accepted: bool
    observed_candidate_id: str
    normalized_innovation_squared: Optional[float]
    innovation_gate_threshold: Optional[float]
    reason: Optional[str]


@dataclass(frozen=True, slots=True)
class VQ2ImuCorrelatedEstimatorUpdate:
    """An ordinary camera-state update paired with exact local IMU evidence.

    The estimator state deliberately remains in the unchanged ``/1`` camera
    bearing model.  ``evidence.derotated_center_norm`` is *not* injected into
    that capture-time filter: doing so would mix a target-attitude bearing with
    a capture-time posterior and then predict it twice.  The retained evidence
    instead proves which bounded attitude sample was correlated with the same
    observation and target for the outer offline attitude adapter.
    """

    estimator_update: RelativeEstimatorUpdate
    evidence: "VQ2DerotationEvidence"

    def __post_init__(self) -> None:
        from estimation.vq2_imu_derotation import VQ2DerotationEvidence

        if type(self.estimator_update) is not RelativeEstimatorUpdate:
            raise TypeError("estimator_update must be exact RelativeEstimatorUpdate")
        if type(self.evidence) is not VQ2DerotationEvidence:
            raise TypeError("evidence must be exact VQ2DerotationEvidence")
        observation = self.evidence.observation
        target = self.evidence.prediction_target
        update = self.estimator_update
        if update.observed_candidate_id != observation.candidate_id:
            raise ValueError("IMU evidence candidate does not match estimator update")
        state = update.state
        if (
            state.timing.host_clock_id != target.host_clock_id
            or state.timing.decision_time_monotonic_ns
            != target.decision_time_monotonic_ns
            or state.timing.prediction_time_monotonic_ns
            != target.prediction_time_monotonic_ns
            or state.timing.prediction_basis is not target.prediction_basis
            or state.timing.delay_model_id != target.delay_model_id
            or state.timing.delay_uncertainty_ns != target.delay_uncertainty_ns
        ):
            raise ValueError(
                "IMU-correlated estimator state does not match its prediction target"
            )
        if update.measurement_accepted:
            validate_relative_gate_state_source(state, observation)

    def validate_integrity(self) -> None:
        """Revalidate nested state/evidence and their exact correlation."""

        state = _revalidate_relative_gate_state(self.estimator_update.state)
        estimator_update = replace(self.estimator_update, state=state)
        self.evidence.validate_integrity()
        replace(self, estimator_update=estimator_update)

    @property
    def state(self) -> RelativeGateStateV1:
        return self.estimator_update.state

    @property
    def current_observation_accepted(self) -> bool:
        return self.estimator_update.measurement_accepted

    @property
    def derotation_applied_to_state(self) -> bool:
        """The frozen ``/1`` state never claims the target-basis correction."""

        return False


@dataclass(frozen=True, slots=True)
class VQ2ImuCorrelatedEstimatorCoast:
    """One first-dropout coast bound to its accepted camera/IMU source.

    The coast retains the prior accepted raw-camera observation and advances
    only its prediction target, state sequence, covariance, and target IMU
    attitude.  The new derotation evidence remains standalone and is not
    applied to the frozen ``/1`` state.
    """

    prior_update: VQ2ImuCorrelatedEstimatorUpdate
    state: RelativeGateStateV1
    evidence: "VQ2DerotationEvidence"

    def __post_init__(self) -> None:
        from estimation.vq2_imu_derotation import VQ2DerotationEvidence

        if type(self.prior_update) is not VQ2ImuCorrelatedEstimatorUpdate:
            raise TypeError(
                "prior_update must be exact VQ2ImuCorrelatedEstimatorUpdate"
            )
        if type(self.state) is not RelativeGateStateV1:
            raise TypeError("state must be exact RelativeGateStateV1")
        if type(self.evidence) is not VQ2DerotationEvidence:
            raise TypeError("evidence must be exact VQ2DerotationEvidence")
        self.prior_update.validate_integrity()
        self.evidence.validate_integrity()
        _revalidate_relative_gate_state(self.state)

        prior = self.prior_update
        prior_state = prior.state
        observation = prior.evidence.observation
        if not prior.current_observation_accepted:
            raise ValueError("coast requires an accepted prior observation")
        if (
            prior_state.health is not RelativeStateHealth.HEALTHY
            or prior_state.dropout_count != 0
            or prior_state.track_role is not TrackRole.ACTIVE
        ):
            raise ValueError(
                "coast requires a healthy active non-dropout prior state"
            )
        if self.evidence.observation != observation:
            raise ValueError("coast evidence changed the accepted observation")
        if self.evidence.capture_attitude != prior.evidence.capture_attitude:
            raise ValueError("coast evidence changed the capture attitude")
        if (
            self.evidence.calibration != prior.evidence.calibration
            or self.evidence.model != prior.evidence.model
        ):
            raise ValueError("coast evidence changed calibration or model identity")

        target = self.evidence.prediction_target
        prior_target = prior.evidence.prediction_target
        if (
            target.prediction_basis is not PredictionBasis.DECISION_TIME
            or target.decision_time_monotonic_ns
            != target.prediction_time_monotonic_ns
            or target.prediction_time_monotonic_ns
            <= prior_target.prediction_time_monotonic_ns
        ):
            raise ValueError("coast requires a strictly newer at-decision target")
        timing = self.state.timing
        if (
            timing.host_clock_id != target.host_clock_id
            or timing.decision_time_monotonic_ns
            != target.decision_time_monotonic_ns
            or timing.prediction_time_monotonic_ns
            != target.prediction_time_monotonic_ns
            or timing.prediction_basis is not target.prediction_basis
            or timing.delay_model_id != target.delay_model_id
            or timing.delay_uncertainty_ns != target.delay_uncertainty_ns
        ):
            raise ValueError("coast state differs from its prediction target")
        validate_relative_gate_state_source(self.state, observation)
        validate_relative_gate_state_sequence((prior_state, self.state))
        if (
            self.state.tracker_id != prior_state.tracker_id
            or self.state.track_role is not prior_state.track_role
            or self.state.authority != prior_state.authority
            or self.state.source_candidate_id != prior_state.source_candidate_id
            or self.state.state_sequence != prior_state.state_sequence + 1
            or self.state.measurement_update_sequence
            != prior_state.measurement_update_sequence
            or self.state.dropout_count != 1
            or self.state.health is not RelativeStateHealth.COASTING
            or self.state.health_reason != "observation_dropout"
            or self.state.normalized_innovation_squared is not None
            or self.state.innovation_gate_threshold is not None
            or self.state.innovation_accepted is not None
        ):
            raise ValueError("coast state is not the exact first-dropout profile")
        metric_fields = (
            "metric_position_body_frd_m",
            "metric_velocity_body_frd_m_s",
            "metric_gate_orientation_body_frd_xyzw",
            "metric_covariance",
        )
        if any(
            getattr(prior_state, name) is not None
            or getattr(self.state, name) != getattr(prior_state, name)
            for name in metric_fields
        ):
            raise ValueError("coast injected unsupported metric state")
        if self.state.covariance == prior_state.covariance:
            raise ValueError("coast covariance must grow from the accepted state")
        if (
            self.state.covariance.model_id != prior_state.covariance.model_id
            or self.state.covariance.feature_order
            != prior_state.covariance.feature_order
        ):
            raise ValueError("coast changed the estimator covariance model")
        prior_diagonal = tuple(
            prior_state.covariance.matrix[index][index] for index in range(6)
        )
        coast_diagonal = tuple(
            self.state.covariance.matrix[index][index] for index in range(6)
        )
        if sum(coast_diagonal) <= sum(prior_diagonal) or any(
            coast_value <= prior_value
            for prior_value, coast_value in zip(
                prior_diagonal,
                coast_diagonal,
            )
        ):
            raise ValueError("coast covariance did not grow conservatively")
        elapsed_s = (
            target.prediction_time_monotonic_ns
            - prior_target.prediction_time_monotonic_ns
        ) / 1_000_000_000
        expected_features = (
            prior_state.bearing_norm[0]
            + prior_state.bearing_rate_norm_s[0] * elapsed_s,
            prior_state.bearing_norm[1]
            + prior_state.bearing_rate_norm_s[1] * elapsed_s,
            prior_state.log_scale + prior_state.expansion_rate_s * elapsed_s,
        )
        actual_features = (*self.state.bearing_norm, self.state.log_scale)
        if self.state.bearing_rate_norm_s != prior_state.bearing_rate_norm_s or (
            self.state.expansion_rate_s != prior_state.expansion_rate_s
        ):
            raise ValueError("coast changed the constant-velocity rate state")
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(actual_features, expected_features)
        ):
            raise ValueError("coast state is not the constant-velocity successor")

        prior_attitude = prior.evidence.target_attitude.attitude
        target_attitude = self.evidence.target_attitude.attitude
        if (
            self.evidence.target_attitude.orientation_uncertainty_rad
            != prior.evidence.target_attitude.orientation_uncertainty_rad
            or self.evidence.target_attitude.host_time_uncertainty_ns
            != prior.evidence.target_attitude.host_time_uncertainty_ns
        ):
            raise ValueError("coast target attitude changed uncertainty model")
        if target_attitude.source != prior_attitude.source:
            raise ValueError("coast target attitude changed IMU source")
        if (
            target_attitude.sample_sequence <= prior_attitude.sample_sequence
            or target_attitude.source_time_us <= prior_attitude.source_time_us
            or target_attitude.receive_monotonic_ns
            <= prior_attitude.receive_monotonic_ns
        ):
            raise ValueError("coast target attitude did not advance strictly")

    def validate_integrity(self) -> None:
        """Revalidate nested source, state, and derotation evidence."""

        self.prior_update.validate_integrity()
        self.evidence.validate_integrity()
        state = _revalidate_relative_gate_state(self.state)
        replace(
            self,
            prior_update=replace(self.prior_update),
            state=state,
        )

    @property
    def derotation_applied_to_state(self) -> bool:
        return False


class VQ2RelativeGateEstimator:
    """Single-track, deterministic 6D bearing/scale Kalman filter."""

    def __init__(
        self,
        tracker_id: str,
        *,
        track_role: TrackRole = TrackRole.ACTIVE,
        config: Optional[RelativeEstimatorConfig] = None,
    ) -> None:
        _bounded_token(tracker_id, "tracker_id")
        if type(track_role) is not TrackRole:
            raise TypeError("track_role must be TrackRole")
        if config is not None and type(config) is not RelativeEstimatorConfig:
            raise TypeError("config must be RelativeEstimatorConfig or None")
        self.tracker_id = tracker_id
        self.track_role = track_role
        self.config = config or RelativeEstimatorConfig()
        self._posterior_state: Optional[np.ndarray] = None
        self._posterior_covariance: Optional[np.ndarray] = None
        self._posterior_time_ns: Optional[int] = None
        self._last_accepted_observation: Optional[GateObservationV1] = None
        self._last_state: Optional[RelativeGateStateV1] = None
        self._accepted_updates = 0
        self._state_sequence = -1
        self._measurement_update_sequence = -1
        self._dropout_count = 0
        self._host_clock_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._seen_frames: set[tuple[str, object]] = set()
        self._last_processed_publication_sequence: Optional[int] = None
        self._last_processed_publish_ns: Optional[int] = None
        self._last_processed_measurement_ns: Optional[int] = None

    @property
    def last_state(self) -> Optional[RelativeGateStateV1]:
        return self._last_state

    @property
    def is_initialized(self) -> bool:
        return self._posterior_state is not None

    def update(
        self,
        observation: GateObservationV1,
        target: RelativePredictionTarget,
    ) -> RelativeEstimatorUpdate:
        """Process one distinct fitted-aperture observation transactionally."""

        self._validate_update_input(observation, target)
        measurement = self._measurement_vector(observation)
        measurement_covariance = self._measurement_covariance(observation)
        return self._update_validated_measurement(
            observation,
            target,
            measurement,
            measurement_covariance,
        )

    def update_with_imu_correlation(
        self,
        evidence: "VQ2DerotationEvidence",
    ) -> VQ2ImuCorrelatedEstimatorUpdate:
        """Update from the raw camera observation and retain IMU correlation.

        Rotation-only correction stays in ``VQ2DerotationEvidence`` for
        offline evaluation.  It cannot be represented honestly in the current
        capture-time ``/1`` filter, so this method intentionally delegates to
        the ordinary camera update without changing its measurement basis.
        """

        from estimation.vq2_imu_derotation import VQ2DerotationEvidence

        if type(evidence) is not VQ2DerotationEvidence:
            raise TypeError("evidence must be exact VQ2DerotationEvidence")
        update = self.update(
            evidence.observation,
            evidence.prediction_target,
        )
        return VQ2ImuCorrelatedEstimatorUpdate(update, evidence)

    def _update_validated_measurement(
        self,
        observation: GateObservationV1,
        target: RelativePredictionTarget,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> RelativeEstimatorUpdate:
        """Advance from already validated source and measurement inputs."""

        epoch_changed = self._epoch_changed(observation)
        gap_reinitialize = False

        if self._posterior_state is None or epoch_changed:
            posterior_state, posterior_covariance = self._initialize_filter(
                measurement, measurement_covariance
            )
            accepted = True
            nis = threshold = None
            accepted_updates = 1
            state_sequence = 0 if epoch_changed else self._state_sequence + 1
            update_sequence = 0 if epoch_changed else self._measurement_update_sequence + 1
        else:
            assert self._posterior_covariance is not None
            assert self._posterior_time_ns is not None
            dt_s = (observation.measurement_time_monotonic_ns - self._posterior_time_ns) / 1e9
            if dt_s <= 0.0:
                raise StaleObservationError("measurement time did not advance")
            if dt_s > self.config.max_measurement_gap_s:
                posterior_state, posterior_covariance = self._initialize_filter(
                    measurement, measurement_covariance
                )
                accepted = True
                nis = threshold = None
                accepted_updates = 1
                gap_reinitialize = True
            else:
                predicted_state, predicted_covariance = self._predict_filter(
                    self._posterior_state,
                    self._posterior_covariance,
                    dt_s,
                )
                (
                    accepted,
                    nis,
                    threshold,
                    posterior_state,
                    posterior_covariance,
                ) = self._measurement_update(
                    predicted_state,
                    predicted_covariance,
                    measurement,
                    measurement_covariance,
                )
                accepted_updates = self._accepted_updates + int(accepted)
            state_sequence = self._state_sequence + 1
            update_sequence = self._measurement_update_sequence + int(accepted)

        if not accepted:
            state = self._build_dropout_state(
                target,
                dropout_count=self._dropout_count + 1,
                reason="innovation_rejected",
            )
            self._validate_next_state(state, self._last_accepted_observation)
            self._commit_processed_observation(observation)
            self._last_state = state
            self._state_sequence = state.state_sequence
            self._dropout_count = state.dropout_count
            return RelativeEstimatorUpdate(
                state=state,
                measurement_accepted=False,
                observed_candidate_id=observation.candidate_id,
                normalized_innovation_squared=nis,
                innovation_gate_threshold=threshold,
                reason="innovation_rejected",
            )

        bounded_state, bounded_covariance, rate_limited = self._bound_rates(
            posterior_state, posterior_covariance
        )
        health, health_reason = self._accepted_health(
            observation,
            accepted_updates=accepted_updates,
            gap_reinitialize=gap_reinitialize,
            rate_limited=rate_limited,
        )
        self._validate_accepted_prediction_horizon(observation, target)
        state = self._build_state(
            source_observation=observation,
            posterior_state=bounded_state,
            posterior_covariance=bounded_covariance,
            posterior_time_ns=observation.measurement_time_monotonic_ns,
            target=target,
            state_sequence=state_sequence,
            measurement_update_sequence=update_sequence,
            dropout_count=0,
            health=health,
            health_reason=health_reason,
            normalized_innovation_squared=nis,
            innovation_gate_threshold=threshold,
            innovation_accepted=(None if nis is None else True),
        )
        self._validate_next_state(state, observation)

        self._posterior_state = bounded_state
        self._posterior_covariance = bounded_covariance
        self._posterior_time_ns = observation.measurement_time_monotonic_ns
        self._last_accepted_observation = observation
        self._last_state = state
        self._accepted_updates = accepted_updates
        self._state_sequence = state.state_sequence
        self._measurement_update_sequence = state.measurement_update_sequence
        self._dropout_count = 0
        self._host_clock_id = observation.host_clock_id
        self._session_id = observation.authority.session_id
        self._commit_processed_observation(observation)
        return RelativeEstimatorUpdate(
            state=state,
            measurement_accepted=True,
            observed_candidate_id=observation.candidate_id,
            normalized_innovation_squared=nis,
            innovation_gate_threshold=threshold,
            reason=health_reason,
        )

    def coast(self, target: RelativePredictionTarget) -> RelativeGateStateV1:
        """Predict from the last accepted source without applying a new frame.

        Control-rate consumers call this only when there is no distinct camera
        observation to apply.  The source and measurement-update sequence stay
        bound to the last accepted observation, while state sequence, dropout
        count, covariance, and health advance deterministically.
        """

        self._validate_coast_target(target)
        state = self._build_dropout_state(
            target,
            dropout_count=self._dropout_count + 1,
            reason="observation_dropout",
        )
        self._validate_next_state(state, self._last_accepted_observation)
        self._last_state = state
        self._state_sequence = state.state_sequence
        self._dropout_count = state.dropout_count
        return state

    def _validate_update_input(
        self,
        observation: GateObservationV1,
        target: RelativePredictionTarget,
    ) -> None:
        if type(observation) is not GateObservationV1:
            raise TypeError("observation must be GateObservationV1")
        if type(target) is not RelativePredictionTarget:
            raise TypeError("target must be RelativePredictionTarget")
        if target.host_clock_id != observation.host_clock_id:
            raise RelativeEstimatorError(
                "prediction target host clock does not match observation"
            )
        if (
            self._host_clock_id is not None
            and observation.host_clock_id != self._host_clock_id
        ):
            raise RelativeEstimatorError(
                "observation host clock does not match estimator clock"
            )
        if (
            self._session_id is not None
            and observation.authority.session_id != self._session_id
        ):
            raise RelativeEstimatorError(
                "observation safety session does not match estimator session"
            )
        if observation.health is ObservationHealth.UNUSABLE:
            raise RelativeEstimatorError("unusable observation cannot enter estimator")
        if observation.log_scale is None:
            raise MissingApertureScaleError(
                "fitted inner-aperture log_scale is required; bbox support is not scale"
            )
        if target.decision_time_monotonic_ns < observation.frame_timing.publish_monotonic_ns:
            raise RelativeEstimatorError("decision time predates observation publication")
        self._validate_requested_target_horizon(target)
        frame_key = (observation.host_clock_id, observation.frame)
        if frame_key in self._seen_frames:
            raise StaleObservationError("camera frame was already processed")
        if self._last_processed_publication_sequence is not None:
            if observation.frame_timing.publication_sequence <= self._last_processed_publication_sequence:
                raise StaleObservationError("frame publication sequence did not advance")
            if observation.frame_timing.publish_monotonic_ns <= self._last_processed_publish_ns:
                raise StaleObservationError("frame publication time did not advance")
            if observation.measurement_time_monotonic_ns <= self._last_processed_measurement_ns:
                raise StaleObservationError("frame measurement time did not advance")
        if (
            self._last_state is not None
            and target.decision_time_monotonic_ns
            < self._last_state.timing.decision_time_monotonic_ns
        ):
            raise StaleObservationError("relative-state decision time regressed")
        if (
            self._last_state is not None
            and target.prediction_time_monotonic_ns
            < self._last_state.timing.prediction_time_monotonic_ns
        ):
            raise StaleObservationError("relative-state prediction time regressed")

    def _validate_coast_target(self, target: RelativePredictionTarget) -> None:
        if type(target) is not RelativePredictionTarget:
            raise TypeError("target must be RelativePredictionTarget")
        if (
            self._host_clock_id is None
            or self._posterior_time_ns is None
            or self._last_state is None
        ):
            raise EstimatorUnavailableError("cannot coast before initialization")
        if target.host_clock_id != self._host_clock_id:
            raise RelativeEstimatorError(
                "prediction target host clock does not match estimator clock"
            )
        if (
            self._last_accepted_observation is None
            or target.decision_time_monotonic_ns
            < self._last_accepted_observation.frame_timing.publish_monotonic_ns
        ):
            raise RelativeEstimatorError(
                "decision time predates accepted source publication"
            )
        self._validate_requested_target_horizon(target)
        if (
            target.decision_time_monotonic_ns
            < self._last_state.timing.decision_time_monotonic_ns
        ):
            raise StaleObservationError("coast decision time regressed")
        if (
            target.prediction_time_monotonic_ns
            <= self._last_state.timing.prediction_time_monotonic_ns
        ):
            raise StaleObservationError(
                "coast prediction time must advance beyond the last state"
            )
        elapsed_ns = target.prediction_time_monotonic_ns - self._posterior_time_ns
        if elapsed_ns > round(self.config.max_coast_s * 1e9):
            raise EstimatorUnavailableError("bounded dropout horizon expired")

    def _validate_requested_target_horizon(
        self,
        target: RelativePredictionTarget,
    ) -> None:
        horizon_ns = (
            target.prediction_time_monotonic_ns - target.decision_time_monotonic_ns
        )
        if horizon_ns > round(self.config.max_prediction_horizon_s * 1e9):
            raise PredictionHorizonError(
                "requested prediction horizon exceeds config bound"
            )

    def _validate_accepted_prediction_horizon(
        self,
        observation: GateObservationV1,
        target: RelativePredictionTarget,
    ) -> None:
        elapsed_ns = (
            target.prediction_time_monotonic_ns
            - observation.measurement_time_monotonic_ns
        )
        if elapsed_ns > round(self.config.max_prediction_horizon_s * 1e9):
            raise PredictionHorizonError(
                "accepted prediction exceeds measurement-relative horizon"
            )

    def _measurement_vector(self, observation: GateObservationV1) -> np.ndarray:
        assert observation.log_scale is not None
        return np.array(
            [observation.center_norm[0], observation.center_norm[1], observation.log_scale],
            dtype=np.float64,
        )

    def _measurement_covariance(self, observation: GateObservationV1) -> np.ndarray:
        feature_indices = {
            name: index for index, name in enumerate(observation.covariance.feature_order)
        }
        try:
            indices = [feature_indices[name] for name in _MEASUREMENT_FEATURE_ORDER]
        except KeyError as exc:
            raise RelativeEstimatorError(
                "observation covariance lacks a required bearing/scale feature"
            ) from exc
        source = np.asarray(observation.covariance.matrix, dtype=np.float64)
        covariance = source[np.ix_(indices, indices)].copy()
        confidence = max(observation.confidence, self.config.confidence_floor)
        inflation = 1.0 / (confidence * confidence)
        inflation *= 1.0 + (
            int(observation.clipping).bit_count()
            * self.config.clipping_variance_multiplier_per_edge
        )
        if observation.health is ObservationHealth.DEGRADED:
            inflation *= self.config.degraded_variance_multiplier
        covariance *= inflation
        covariance += np.eye(3) * self.config.minimum_measurement_variance
        return self._stabilize_covariance(covariance)

    def _initialize_filter(
        self,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.zeros(6, dtype=np.float64)
        state[:3] = measurement
        covariance = np.zeros((6, 6), dtype=np.float64)
        covariance[:3, :3] = measurement_covariance
        covariance[3, 3] = self.config.initial_bearing_rate_std_norm_s**2
        covariance[4, 4] = self.config.initial_bearing_rate_std_norm_s**2
        covariance[5, 5] = self.config.initial_expansion_rate_std_s**2
        return state, self._stabilize_covariance(covariance)

    def _predict_filter(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(6, dtype=np.float64)
        for feature_index, rate_index in zip(
            _MEASUREMENT_STATE_INDICES, _RATE_LIMITS_INDICES
        ):
            transition[feature_index, rate_index] = dt_s
        process = np.zeros((6, 6), dtype=np.float64)
        for feature_index, rate_index, accel_std in (
            (0, 3, self.config.bearing_process_accel_std_norm_s2),
            (1, 4, self.config.bearing_process_accel_std_norm_s2),
            (2, 5, self.config.scale_process_accel_std_s2),
        ):
            variance = accel_std * accel_std
            process[feature_index, feature_index] = variance * dt_s**4 / 4.0
            process[feature_index, rate_index] = variance * dt_s**3 / 2.0
            process[rate_index, feature_index] = process[feature_index, rate_index]
            process[rate_index, rate_index] = variance * dt_s**2
        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + process
        return predicted_state, self._stabilize_covariance(predicted_covariance)

    def _measurement_update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[bool, float, float, np.ndarray, np.ndarray]:
        observation_matrix = np.zeros((3, 6), dtype=np.float64)
        observation_matrix[0, 0] = 1.0
        observation_matrix[1, 1] = 1.0
        observation_matrix[2, 2] = 1.0
        innovation = measurement - observation_matrix @ predicted_state
        innovation_covariance = (
            observation_matrix @ predicted_covariance @ observation_matrix.T
            + measurement_covariance
        )
        innovation_covariance = self._stabilize_covariance(innovation_covariance)
        solved = np.linalg.solve(innovation_covariance, innovation)
        nis = max(0.0, float(innovation @ solved))
        threshold = self.config.innovation_gate_threshold_3d
        if nis > threshold:
            return False, nis, threshold, predicted_state, predicted_covariance
        gain = np.linalg.solve(
            innovation_covariance,
            observation_matrix @ predicted_covariance,
        ).T
        updated_state = predicted_state + gain @ innovation
        identity_minus_gain_h = np.eye(6) - gain @ observation_matrix
        updated_covariance = (
            identity_minus_gain_h
            @ predicted_covariance
            @ identity_minus_gain_h.T
            + gain @ measurement_covariance @ gain.T
        )
        return (
            True,
            nis,
            threshold,
            updated_state,
            self._stabilize_covariance(updated_covariance),
        )

    def _bound_rates(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        bounded = state.copy()
        bounded_covariance = covariance.copy()
        limited = False
        limits = (
            self.config.max_abs_bearing_rate_norm_s,
            self.config.max_abs_bearing_rate_norm_s,
            self.config.max_abs_expansion_rate_s,
        )
        for index, limit in zip(_RATE_LIMITS_INDICES, limits):
            if abs(bounded[index]) > limit:
                bounded[index] = math.copysign(limit, bounded[index])
                bounded_covariance[index, index] = max(
                    bounded_covariance[index, index], limit * limit
                )
                limited = True
        return bounded, self._stabilize_covariance(bounded_covariance), limited

    def _accepted_health(
        self,
        observation: GateObservationV1,
        *,
        accepted_updates: int,
        gap_reinitialize: bool,
        rate_limited: bool,
    ) -> tuple[RelativeStateHealth, Optional[str]]:
        if gap_reinitialize:
            return RelativeStateHealth.INITIALIZING, "measurement_gap_reinitialized"
        if rate_limited:
            return RelativeStateHealth.DEGRADED, "feature_rate_limited"
        if observation.health is ObservationHealth.DEGRADED:
            return RelativeStateHealth.DEGRADED, "source_observation_degraded"
        if observation.clipping != FrameEdge.NONE:
            return RelativeStateHealth.DEGRADED, "source_observation_clipped"
        if accepted_updates < self.config.minimum_accepted_updates_for_healthy:
            return RelativeStateHealth.INITIALIZING, "insufficient_accepted_updates"
        return RelativeStateHealth.HEALTHY, None

    def _build_state(
        self,
        *,
        source_observation: GateObservationV1,
        posterior_state: np.ndarray,
        posterior_covariance: np.ndarray,
        posterior_time_ns: int,
        target: RelativePredictionTarget,
        state_sequence: int,
        measurement_update_sequence: int,
        dropout_count: int,
        health: RelativeStateHealth,
        health_reason: Optional[str],
        normalized_innovation_squared: Optional[float],
        innovation_gate_threshold: Optional[float],
        innovation_accepted: Optional[bool],
    ) -> RelativeGateStateV1:
        prediction_dt_s = (
            target.prediction_time_monotonic_ns - posterior_time_ns
        ) / 1e9
        if prediction_dt_s < 0.0:
            raise RelativeEstimatorError("prediction time predates filter posterior")
        predicted_state, predicted_covariance = self._predict_filter(
            posterior_state, posterior_covariance, prediction_dt_s
        )
        predicted_state, predicted_covariance, bounds_hit = self._bound_output(
            predicted_state, predicted_covariance
        )
        if bounds_hit:
            if dropout_count:
                health = RelativeStateHealth.LOST
            else:
                health = RelativeStateHealth.UNHEALTHY
            health_reason = "predicted_bearing_reached_contract_bound"
        timing = PredictionTimeV1(
            host_clock_id=source_observation.host_clock_id,
            source_frame=source_observation.frame,
            source_frame_publication_sequence=(
                source_observation.frame_timing.publication_sequence
            ),
            source_frame_publish_monotonic_ns=(
                source_observation.frame_timing.publish_monotonic_ns
            ),
            measurement_time_monotonic_ns=(
                source_observation.measurement_time_monotonic_ns
            ),
            measurement_time_basis=source_observation.measurement_time_basis,
            measurement_time_model_id=source_observation.measurement_time_model_id,
            measurement_uncertainty_ns=source_observation.measurement_uncertainty_ns,
            decision_time_monotonic_ns=target.decision_time_monotonic_ns,
            prediction_time_monotonic_ns=target.prediction_time_monotonic_ns,
            prediction_basis=target.prediction_basis,
            delay_model_id=target.delay_model_id,
            delay_uncertainty_ns=target.delay_uncertainty_ns,
        )
        covariance = FeatureCovarianceV1(
            model_id=self.config.covariance_model_id,
            feature_order=_STATE_FEATURE_ORDER,
            matrix=tuple(
                tuple(float(value) for value in row) for row in predicted_covariance
            ),
        )
        return RelativeGateStateV1(
            timing=timing,
            authority=source_observation.authority,
            tracker_id=self.tracker_id,
            state_sequence=state_sequence,
            measurement_update_sequence=measurement_update_sequence,
            source_candidate_id=source_observation.candidate_id,
            track_role=self.track_role,
            bearing_norm=(float(predicted_state[0]), float(predicted_state[1])),
            bearing_rate_norm_s=(
                float(predicted_state[3]),
                float(predicted_state[4]),
            ),
            log_scale=float(predicted_state[2]),
            expansion_rate_s=float(predicted_state[5]),
            covariance=covariance,
            metric_position_body_frd_m=None,
            metric_velocity_body_frd_m_s=None,
            metric_gate_orientation_body_frd_xyzw=None,
            metric_covariance=None,
            last_clipping=source_observation.clipping,
            outer_visibility=source_observation.outer_edges.visibility,
            inner_visibility=source_observation.inner_edges.visibility,
            normalized_innovation_squared=normalized_innovation_squared,
            innovation_gate_threshold=innovation_gate_threshold,
            innovation_accepted=innovation_accepted,
            dropout_count=dropout_count,
            health=health,
            health_reason=health_reason,
        )

    def _build_dropout_state(
        self,
        target: RelativePredictionTarget,
        *,
        dropout_count: int,
        reason: str,
    ) -> RelativeGateStateV1:
        if (
            self._posterior_state is None
            or self._posterior_covariance is None
            or self._posterior_time_ns is None
            or self._last_accepted_observation is None
        ):
            raise EstimatorUnavailableError("cannot coast before one accepted observation")
        elapsed_s = (
            target.prediction_time_monotonic_ns - self._posterior_time_ns
        ) / 1e9
        if elapsed_s < 0.0:
            raise StaleObservationError("dropout prediction time regressed")
        if elapsed_s > self.config.max_coast_s:
            raise EstimatorUnavailableError("bounded dropout horizon expired")
        health = (
            RelativeStateHealth.LOST
            if elapsed_s > self.config.coast_lost_after_s
            else RelativeStateHealth.COASTING
        )
        covariance = self._posterior_covariance.copy()
        covariance += (
            np.eye(6)
            * self.config.dropout_variance_per_s
            * max(elapsed_s, 0.0)
        )
        return self._build_state(
            source_observation=self._last_accepted_observation,
            posterior_state=self._posterior_state,
            posterior_covariance=self._stabilize_covariance(covariance),
            posterior_time_ns=self._posterior_time_ns,
            target=target,
            state_sequence=self._state_sequence + 1,
            measurement_update_sequence=self._measurement_update_sequence,
            dropout_count=dropout_count,
            health=health,
            health_reason=(reason if health is RelativeStateHealth.COASTING else f"{reason}:lost"),
            normalized_innovation_squared=None,
            innovation_gate_threshold=None,
            innovation_accepted=None,
        )

    def _bound_output(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        bounded = state.copy()
        bounded_covariance = covariance.copy()
        hit = False
        for index in (0, 1):
            if bounded[index] < -4.0 or bounded[index] > 4.0:
                bounded[index] = min(4.0, max(-4.0, bounded[index]))
                bounded_covariance[index, index] = max(
                    bounded_covariance[index, index], 16.0
                )
                hit = True
        return bounded, self._stabilize_covariance(bounded_covariance), hit

    def _stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        result = np.asarray(covariance, dtype=np.float64)
        if result.ndim != 2 or result.shape[0] != result.shape[1]:
            raise RelativeEstimatorError("covariance must be square")
        if not np.all(np.isfinite(result)):
            raise EstimatorUnavailableError("filter covariance became non-finite")
        result = (result + result.T) * 0.5
        eigenvalues = np.linalg.eigvalsh(result)
        minimum = float(eigenvalues[0])
        floor = self.config.covariance_eigenvalue_floor
        if minimum < floor:
            result = result + np.eye(result.shape[0]) * (floor - minimum)
        return (result + result.T) * 0.5

    def _epoch_changed(self, observation: GateObservationV1) -> bool:
        if self._last_state is None:
            return False
        previous = self._last_state.authority
        current = observation.authority
        return (
            previous.session_id,
            previous.reset_epoch,
            previous.gate_epoch,
            previous.expected_gate_index,
        ) != (
            current.session_id,
            current.reset_epoch,
            current.gate_epoch,
            current.expected_gate_index,
        )

    def _validate_next_state(
        self,
        state: RelativeGateStateV1,
        source_observation: Optional[GateObservationV1],
    ) -> None:
        if source_observation is None:
            raise EstimatorUnavailableError("relative state has no accepted source")
        validate_relative_gate_state_source(state, source_observation)
        if self._last_state is not None:
            validate_relative_gate_state_sequence((self._last_state, state))

    def _commit_processed_observation(self, observation: GateObservationV1) -> None:
        self._seen_frames.add((observation.host_clock_id, observation.frame))
        self._last_processed_publication_sequence = (
            observation.frame_timing.publication_sequence
        )
        self._last_processed_publish_ns = observation.frame_timing.publish_monotonic_ns
        self._last_processed_measurement_ns = observation.measurement_time_monotonic_ns


__all__ = [
    "EstimatorUnavailableError",
    "MissingApertureScaleError",
    "PredictionHorizonError",
    "RelativeEstimatorConfig",
    "RelativeEstimatorError",
    "RelativeEstimatorUpdate",
    "RelativePredictionTarget",
    "StaleObservationError",
    "VQ2ImuCorrelatedEstimatorUpdate",
    "VQ2ImuCorrelatedEstimatorCoast",
    "VQ2RelativeGateEstimator",
]
