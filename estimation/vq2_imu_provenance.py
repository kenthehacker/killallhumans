"""Exact local provenance for offline VQ2 ``HIGHRES_IMU`` attitude.

The build-3385 ``HIGHRES_IMU.time_usec`` value is a source-clock sample stamp.
It is suitable for strict ordering and the attitude estimator's integration
``dt`` only.  It is not a host timestamp, so this module never subtracts it
from ``receive_monotonic_ns`` or from any other host-clock value.

The values here are frozen local composition types, not frozen ``/1`` wire
contracts.  The wrapper performs no I/O and has no runtime, transport,
supervisor, or simulator authority.  A caller must bind it to one explicit
session/reset/clock/stream/generation identity and must explicitly rekey before
samples from another identity can enter a fresh estimator/bootstrap.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Optional

from estimation.imu_attitude import (
    AttitudeEstimate,
    ImuAttitudeConfig,
    ImuAttitudeEstimator,
)


Vector3 = tuple[float, float, float]
QuaternionWxyz = tuple[float, float, float, float]

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_BOOTSTRAP_REASONS = frozenset(
    {
        "calibrating",
        "bootstrap_not_stationary_or_upright",
        "bootstrap_variance_too_high",
    }
)


class VQ2ImuProvenanceError(ValueError):
    """Base class for fail-closed IMU provenance rejections."""


class VQ2ImuLineageError(VQ2ImuProvenanceError):
    """Raised when a sample is stale, discontinuous, or from another source."""


class VQ2ImuEstimateUnavailableError(VQ2ImuProvenanceError):
    """Raised when an accepted-looking sample cannot yield a healthy estimate."""


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded token")
    return value


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_tuple(
    value: object,
    label: str,
    *,
    length: int,
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    return tuple(
        _finite_float(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class VQ2ImuSource:
    """One explicitly bound local IMU stream epoch; not a wire schema."""

    session_id: str
    reset_epoch: int
    host_clock_id: str
    stream_id: str
    generation: int

    def __post_init__(self) -> None:
        _bounded_token(self.session_id, "session_id")
        _exact_nonnegative_int(self.reset_epoch, "reset_epoch")
        _bounded_token(self.host_clock_id, "host_clock_id")
        _bounded_token(self.stream_id, "stream_id")
        _exact_nonnegative_int(self.generation, "generation")

    @property
    def epoch_key(self) -> tuple[str, int, str, str, int]:
        """Exact identity that must remain fixed for one estimator bootstrap."""

        return (
            self.session_id,
            self.reset_epoch,
            self.host_clock_id,
            self.stream_id,
            self.generation,
        )


@dataclass(frozen=True, slots=True)
class VQ2TimedImuSample:
    """One immutable raw IMU sample with independent source/host stamps."""

    source: VQ2ImuSource
    sample_sequence: int
    source_time_us: int
    receive_monotonic_ns: int
    accel_mps2: Vector3
    gyro_rad_s: Vector3

    def __post_init__(self) -> None:
        if type(self.source) is not VQ2ImuSource:
            raise TypeError("source must be exact VQ2ImuSource")
        _exact_nonnegative_int(self.sample_sequence, "sample_sequence")
        _exact_nonnegative_int(self.source_time_us, "source_time_us")
        _exact_nonnegative_int(self.receive_monotonic_ns, "receive_monotonic_ns")
        accel = _finite_tuple(self.accel_mps2, "accel_mps2", length=3)
        gyro = _finite_tuple(self.gyro_rad_s, "gyro_rad_s", length=3)
        object.__setattr__(self, "accel_mps2", accel)
        object.__setattr__(self, "gyro_rad_s", gyro)

    @property
    def session_id(self) -> str:
        return self.source.session_id

    @property
    def reset_epoch(self) -> int:
        return self.source.reset_epoch

    @property
    def host_clock_id(self) -> str:
        return self.source.host_clock_id

    @property
    def stream_id(self) -> str:
        return self.source.stream_id

    @property
    def generation(self) -> int:
        return self.source.generation

    @property
    def lineage_key(self) -> tuple[str, int, str, str, int, int, int, int]:
        """Exact per-sample lineage without cross-clock arithmetic."""

        return (
            *self.source.epoch_key,
            self.sample_sequence,
            self.source_time_us,
            self.receive_monotonic_ns,
        )


@dataclass(frozen=True, slots=True)
class VQ2TimestampedAttitude:
    """One calibrated, healthy attitude bound to exactly one IMU sample.

    Construction itself is proof of the ``healthy`` and ``calibrated``
    invariants.  The explicit fields remain available so downstream local
    compositions can check those invariants without relying on type folklore.
    ``propagated`` is false for the bootstrap-completing estimate and true for
    ordinary gyro propagation; both are calibrated and healthy.
    """

    source: VQ2ImuSource
    sample_sequence: int
    source_time_us: int
    receive_monotonic_ns: int
    orientation_body_to_ned_wxyz: QuaternionWxyz
    body_rates_rad_s: Vector3
    gyro_bias_rad_s: Vector3
    accel_trust: float
    propagated: bool
    yaw_observable: bool = False
    healthy: bool = True
    calibrated: bool = True

    def __post_init__(self) -> None:
        if type(self.source) is not VQ2ImuSource:
            raise TypeError("source must be exact VQ2ImuSource")
        _exact_nonnegative_int(self.sample_sequence, "sample_sequence")
        _exact_nonnegative_int(self.source_time_us, "source_time_us")
        _exact_nonnegative_int(self.receive_monotonic_ns, "receive_monotonic_ns")
        orientation = _finite_tuple(
            self.orientation_body_to_ned_wxyz,
            "orientation_body_to_ned_wxyz",
            length=4,
        )
        norm = math.sqrt(sum(component * component for component in orientation))
        if abs(norm - 1.0) > 1e-9:
            raise ValueError("orientation_body_to_ned_wxyz must be unit length")
        body_rates = _finite_tuple(self.body_rates_rad_s, "body_rates_rad_s", length=3)
        gyro_bias = _finite_tuple(self.gyro_bias_rad_s, "gyro_bias_rad_s", length=3)
        accel_trust = _finite_float(self.accel_trust, "accel_trust")
        if not 0.0 <= accel_trust <= 1.0:
            raise ValueError("accel_trust must be in [0, 1]")
        _exact_bool(self.propagated, "propagated")
        yaw_observable = _exact_bool(self.yaw_observable, "yaw_observable")
        healthy = _exact_bool(self.healthy, "healthy")
        calibrated = _exact_bool(self.calibrated, "calibrated")
        if yaw_observable:
            raise ValueError("HIGHRES_IMU-only yaw cannot be observable")
        if not healthy or not calibrated:
            raise ValueError("timestamped attitude must be healthy and calibrated")
        object.__setattr__(self, "orientation_body_to_ned_wxyz", orientation)
        object.__setattr__(self, "body_rates_rad_s", body_rates)
        object.__setattr__(self, "gyro_bias_rad_s", gyro_bias)
        object.__setattr__(self, "accel_trust", accel_trust)

    @property
    def session_id(self) -> str:
        return self.source.session_id

    @property
    def reset_epoch(self) -> int:
        return self.source.reset_epoch

    @property
    def host_clock_id(self) -> str:
        return self.source.host_clock_id

    @property
    def stream_id(self) -> str:
        return self.source.stream_id

    @property
    def generation(self) -> int:
        return self.source.generation

    @property
    def lineage_key(self) -> tuple[str, int, str, str, int, int, int, int]:
        return (
            *self.source.epoch_key,
            self.sample_sequence,
            self.source_time_us,
            self.receive_monotonic_ns,
        )

    @property
    def roll_rad(self) -> float:
        return self._euler()[0]

    @property
    def pitch_rad(self) -> float:
        return self._euler()[1]

    @property
    def yaw_rad(self) -> float:
        return self._euler()[2]

    def _euler(self) -> Vector3:
        w, x, y, z = self.orientation_body_to_ned_wxyz
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (roll, pitch, yaw)


class VQ2ImuProvenanceEstimator:
    """Stateful, I/O-free provenance wrapper around ``ImuAttitudeEstimator``.

    Bootstrap samples are accepted into a private candidate estimator and
    return ``None`` until calibration completes.  Invalid lineage is rejected
    before estimator evaluation.  Estimator evaluation itself happens on a
    clone; an unhealthy/gap or malformed result therefore cannot partially
    advance the accepted estimator or per-sample lineage.
    """

    def __init__(
        self,
        source: VQ2ImuSource,
        *,
        config: Optional[ImuAttitudeConfig] = None,
        initial_yaw_rad: float = 0.0,
    ) -> None:
        if type(source) is not VQ2ImuSource:
            raise TypeError("source must be exact VQ2ImuSource")
        if config is not None and type(config) is not ImuAttitudeConfig:
            raise TypeError("config must be exact ImuAttitudeConfig or None")
        yaw = _finite_float(initial_yaw_rad, "initial_yaw_rad")
        selected_config = config if config is not None else ImuAttitudeConfig()
        estimator = ImuAttitudeEstimator(selected_config, initial_yaw_rad=yaw)

        self._source = source
        self._config = selected_config
        self._initial_yaw_rad = yaw
        self._estimator = estimator
        self._bound_epoch_keys = frozenset({source.epoch_key})
        self._last_sample: Optional[VQ2TimedImuSample] = None
        self._last_attitude: Optional[VQ2TimestampedAttitude] = None

    @property
    def source(self) -> VQ2ImuSource:
        return self._source

    @property
    def config(self) -> ImuAttitudeConfig:
        return self._config

    @property
    def is_ready(self) -> bool:
        return self._estimator.is_ready

    @property
    def calibration_progress(self) -> float:
        return self._estimator.calibration_progress

    @property
    def last_sample(self) -> Optional[VQ2TimedImuSample]:
        return self._last_sample

    @property
    def last_attitude(self) -> Optional[VQ2TimestampedAttitude]:
        return self._last_attitude

    @property
    def expected_sample_sequence(self) -> Optional[int]:
        if self._last_sample is None:
            return None
        return self._last_sample.sample_sequence + 1

    def rekey(self, source: VQ2ImuSource) -> None:
        """Bind a changed source to a fresh estimator and pad bootstrap.

        A same-identity reset is forbidden because it would make old samples
        replayable.  Within one session/reset epoch, only a strictly advancing
        generation of the same host clock and stream is a legal rekey.
        """

        if type(source) is not VQ2ImuSource:
            raise TypeError("source must be exact VQ2ImuSource")
        self._validate_rekey(source)
        candidate = ImuAttitudeEstimator(
            self._config,
            initial_yaw_rad=self._initial_yaw_rad,
        )
        bound_epoch_keys = self._bound_epoch_keys | {source.epoch_key}
        self._source = source
        self._estimator = candidate
        self._bound_epoch_keys = bound_epoch_keys
        self._last_sample = None
        self._last_attitude = None

    def update(
        self,
        sample: VQ2TimedImuSample,
    ) -> Optional[VQ2TimestampedAttitude]:
        """Consume one exact sample, withholding output until safe calibration."""

        if type(sample) is not VQ2TimedImuSample:
            raise TypeError("sample must be exact VQ2TimedImuSample")
        self._validate_source(sample.source)
        self._validate_ordering(sample)

        candidate = copy.deepcopy(self._estimator)
        try:
            estimate = candidate.update(
                sample.source_time_us,
                sample.accel_mps2,
                sample.gyro_rad_s,
            )
        except Exception as exc:
            raise VQ2ImuEstimateUnavailableError(
                "attitude estimator failed without committing the sample"
            ) from exc

        if estimate is None:
            reason = candidate.last_rejection_reason
            if candidate.is_ready or reason not in _BOOTSTRAP_REASONS:
                raise VQ2ImuEstimateUnavailableError(
                    f"attitude estimator rejected sample: {reason or 'unknown'}"
                )
            self._commit(candidate, sample, attitude=None)
            return None

        if type(estimate) is not AttitudeEstimate:
            raise VQ2ImuEstimateUnavailableError(
                "attitude estimator returned an unexpected value type"
            )
        if (
            type(estimate.timestamp_us) is not int
            or estimate.timestamp_us != sample.source_time_us
        ):
            raise VQ2ImuEstimateUnavailableError(
                "attitude estimator relabelled the sample source time"
            )
        if not candidate.is_ready or not estimate.healthy:
            raise VQ2ImuEstimateUnavailableError(
                f"unhealthy attitude withheld: {estimate.reason or 'unknown'}"
            )
        if estimate.reason is not None:
            raise VQ2ImuEstimateUnavailableError(
                "healthy attitude unexpectedly carried a rejection reason"
            )

        orientation = estimate.orientation
        try:
            attitude = VQ2TimestampedAttitude(
                source=sample.source,
                sample_sequence=sample.sample_sequence,
                source_time_us=sample.source_time_us,
                receive_monotonic_ns=sample.receive_monotonic_ns,
                orientation_body_to_ned_wxyz=(
                    orientation.w,
                    orientation.x,
                    orientation.y,
                    orientation.z,
                ),
                body_rates_rad_s=estimate.body_rates,
                gyro_bias_rad_s=estimate.gyro_bias,
                accel_trust=estimate.accel_trust,
                propagated=estimate.propagated,
                yaw_observable=estimate.yaw_observable,
                healthy=estimate.healthy,
                calibrated=candidate.is_ready,
            )
        except (TypeError, ValueError) as exc:
            raise VQ2ImuEstimateUnavailableError(
                "attitude estimator returned an invalid calibrated estimate"
            ) from exc

        if attitude.lineage_key != sample.lineage_key:
            raise VQ2ImuEstimateUnavailableError(
                "timestamped attitude did not preserve exact sample lineage"
            )
        self._commit(candidate, sample, attitude=attitude)
        return attitude

    def _validate_source(self, source: VQ2ImuSource) -> None:
        labels = (
            "session_id",
            "reset_epoch",
            "host_clock_id",
            "stream_id",
            "generation",
        )
        changed = tuple(
            label
            for label in labels
            if getattr(source, label) != getattr(self._source, label)
        )
        if changed:
            raise VQ2ImuLineageError(
                "sample changed bound IMU source fields: " + ", ".join(changed)
            )

    def _validate_ordering(self, sample: VQ2TimedImuSample) -> None:
        previous = self._last_sample
        if previous is None:
            return
        if sample.sample_sequence <= previous.sample_sequence:
            raise VQ2ImuLineageError("sample sequence duplicated or regressed")
        if sample.sample_sequence != previous.sample_sequence + 1:
            raise VQ2ImuLineageError("sample sequence is not contiguous")
        if sample.source_time_us <= previous.source_time_us:
            raise VQ2ImuLineageError("IMU source time duplicated or regressed")
        source_delta_us = sample.source_time_us - previous.source_time_us
        if source_delta_us > self._config.max_dt_s * 1_000_000.0:
            raise VQ2ImuEstimateUnavailableError(
                "IMU source timestamp_gap exceeds the estimator bound"
            )
        if sample.receive_monotonic_ns <= previous.receive_monotonic_ns:
            raise VQ2ImuLineageError("IMU host receive time duplicated or regressed")

    def _validate_rekey(self, source: VQ2ImuSource) -> None:
        previous = self._source
        if source == previous:
            raise VQ2ImuLineageError("same-identity estimator reset is forbidden")
        if source.epoch_key in self._bound_epoch_keys:
            raise VQ2ImuLineageError("previously bound IMU source cannot be reused")
        if source.session_id != previous.session_id:
            return
        if source.reset_epoch < previous.reset_epoch:
            raise VQ2ImuLineageError("reset epoch cannot regress within a session")
        if source.reset_epoch != previous.reset_epoch:
            return
        if source.host_clock_id != previous.host_clock_id:
            raise VQ2ImuLineageError(
                "host clock cannot change inside one session/reset epoch"
            )
        if source.stream_id != previous.stream_id:
            raise VQ2ImuLineageError(
                "IMU stream cannot change inside one session/reset epoch"
            )
        if source.generation <= previous.generation:
            raise VQ2ImuLineageError(
                "IMU generation must advance for an in-epoch rekey"
            )

    def _commit(
        self,
        estimator: ImuAttitudeEstimator,
        sample: VQ2TimedImuSample,
        *,
        attitude: Optional[VQ2TimestampedAttitude],
    ) -> None:
        self._estimator = estimator
        self._last_sample = sample
        self._last_attitude = attitude


__all__ = [
    "VQ2ImuEstimateUnavailableError",
    "VQ2ImuLineageError",
    "VQ2ImuProvenanceError",
    "VQ2ImuProvenanceEstimator",
    "VQ2ImuSource",
    "VQ2TimedImuSample",
    "VQ2TimestampedAttitude",
]
