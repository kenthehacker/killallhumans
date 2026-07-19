"""Direct and adversarial tests for the local VQ2 IMU provenance seam."""

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from competition.adapter import Quaternion
from estimation.imu_attitude import ImuAttitudeConfig, ImuAttitudeEstimator
from estimation.vq2_imu_provenance import (
    VQ2ImuEstimateUnavailableError,
    VQ2ImuLineageError,
    VQ2ImuProvenanceEstimator,
    VQ2ImuSource,
    VQ2TimedImuSample,
    VQ2TimestampedAttitude,
)


G = 9.80665


def _source(**overrides) -> VQ2ImuSource:
    values = {
        "session_id": "training-session-7",
        "reset_epoch": 3,
        "host_clock_id": "host-monotonic-1",
        "stream_id": "highres-imu-0",
        "generation": 4,
    }
    values.update(overrides)
    return VQ2ImuSource(**values)


def _fast_config(**overrides) -> ImuAttitudeConfig:
    values = {
        "calibration_min_samples": 3,
        "calibration_min_duration_s": 0.02,
        "max_dt_s": 0.02,
    }
    values.update(overrides)
    return ImuAttitudeConfig(**values)


def _sample(
    source: VQ2ImuSource,
    sequence: int,
    source_time_us: int,
    receive_monotonic_ns: int,
    *,
    accel=(0.0, 0.0, -G),
    gyro=(0.0, 0.0, 0.0),
) -> VQ2TimedImuSample:
    return VQ2TimedImuSample(
        source=source,
        sample_sequence=sequence,
        source_time_us=source_time_us,
        receive_monotonic_ns=receive_monotonic_ns,
        accel_mps2=accel,
        gyro_rad_s=gyro,
    )


def _bootstrap(
    wrapper: VQ2ImuProvenanceEstimator,
    *,
    sequence: int = 0,
    source_time_us: int = 4_000_000_000_000,
    receive_monotonic_ns: int = 11,
):
    attitude = None
    sample = None
    for offset in range(3):
        sample = _sample(
            wrapper.source,
            sequence + offset,
            source_time_us + offset * 10_000,
            receive_monotonic_ns + offset,
        )
        attitude = wrapper.update(sample)
        if offset < 2:
            assert attitude is None
            assert wrapper.last_attitude is None
    assert sample is not None
    assert attitude is not None
    assert wrapper.is_ready
    return sample, attitude


def _direct_attitude(**overrides) -> VQ2TimestampedAttitude:
    values = {
        "source": _source(),
        "sample_sequence": 2,
        "source_time_us": 120_000,
        "receive_monotonic_ns": 9_000_000,
        "orientation_body_to_ned_wxyz": (1.0, 0.0, 0.0, 0.0),
        "body_rates_rad_s": (0.0, 0.0, 0.0),
        "gyro_bias_rad_s": (0.0, 0.0, 0.0),
        "accel_trust": 1.0,
        "propagated": True,
    }
    values.update(overrides)
    return VQ2TimestampedAttitude(**values)


def test_local_values_are_exact_deeply_immutable_and_expose_stable_lineage():
    source = _source()
    sample = _sample(source, 8, 123_456, 8_765_432, accel=(0, 0, -G))
    attitude = _direct_attitude(source=source)

    assert not hasattr(source, "__dict__")
    assert not hasattr(sample, "__dict__")
    assert not hasattr(attitude, "__dict__")
    assert sample.accel_mps2 == (0.0, 0.0, -G)
    assert sample.lineage_key == (
        source.session_id,
        source.reset_epoch,
        source.host_clock_id,
        source.stream_id,
        source.generation,
        8,
        123_456,
        8_765_432,
    )
    assert attitude.session_id == source.session_id
    assert attitude.reset_epoch == source.reset_epoch
    assert attitude.host_clock_id == source.host_clock_id
    assert attitude.stream_id == source.stream_id
    assert attitude.generation == source.generation
    assert attitude.healthy and attitude.calibrated

    with pytest.raises(FrozenInstanceError):
        sample.sample_sequence = 9
    with pytest.raises(FrozenInstanceError):
        source.generation = 5
    with pytest.raises(FrozenInstanceError):
        attitude.accel_trust = 0.0


@pytest.mark.parametrize(
    ("factory", "error_type"),
    [
        (lambda: _source(reset_epoch=True), TypeError),
        (lambda: _source(host_clock_id="space is not a token"), ValueError),
        (
            lambda: VQ2TimedImuSample(
                source=_source(),
                sample_sequence=0,
                source_time_us=0,
                receive_monotonic_ns=0,
                accel_mps2=[0.0, 0.0, -G],
                gyro_rad_s=(0.0, 0.0, 0.0),
            ),
            TypeError,
        ),
        (
            lambda: _sample(
                _source(),
                0,
                0,
                0,
                gyro=(math.nan, 0.0, 0.0),
            ),
            ValueError,
        ),
        (lambda: _direct_attitude(healthy=False), ValueError),
        (lambda: _direct_attitude(calibrated=False), ValueError),
        (lambda: _direct_attitude(yaw_observable=True), ValueError),
        (
            lambda: _direct_attitude(
                orientation_body_to_ned_wxyz=(2.0, 0.0, 0.0, 0.0)
            ),
            ValueError,
        ),
        (lambda: _direct_attitude(accel_trust=1.01), ValueError),
        (lambda: _direct_attitude(propagated=1), TypeError),
    ],
)
def test_local_values_reject_ambiguous_or_unsafe_construction(factory, error_type):
    with pytest.raises(error_type):
        factory()


def test_bootstrap_withholds_output_then_preserves_exact_sample_lineage():
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(
        source,
        config=_fast_config(),
        initial_yaw_rad=0.25,
    )

    final_sample, attitude = _bootstrap(wrapper)

    assert attitude.source is source
    assert attitude.lineage_key == final_sample.lineage_key
    assert attitude.source_time_us == 4_000_000_020_000
    assert attitude.receive_monotonic_ns == 13
    assert attitude.healthy and attitude.calibrated
    assert not attitude.propagated
    assert not attitude.yaw_observable
    assert attitude.yaw_rad == pytest.approx(0.25, abs=1e-12)
    assert wrapper.last_sample is final_sample
    assert wrapper.last_attitude is attitude
    assert wrapper.expected_sample_sequence == 3


def test_source_time_drives_dt_but_is_never_mixed_with_host_receive_time():
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(
        source,
        config=_fast_config(gravity_correction_kp=0.0, gyro_bias_ki=0.0),
    )
    last_sample, initial = _bootstrap(wrapper)

    # A huge host-clock jump does not become estimator dt.  Only the 10 ms raw
    # source-clock delta is integrated, while host receive time is copied.
    next_sample = _sample(
        source,
        3,
        last_sample.source_time_us + 10_000,
        10**18,
        gyro=(0.0, 0.0, 1.0),
    )
    propagated = wrapper.update(next_sample)
    assert propagated is not None
    assert propagated.yaw_rad - initial.yaw_rad == pytest.approx(0.01, abs=1e-12)
    assert propagated.receive_monotonic_ns == 10**18
    assert propagated.source_time_us == next_sample.source_time_us

    # Conversely, a source-clock gap is unhealthy even if host arrival advances
    # by only one nanosecond.  Clone-before-update keeps the accepted state fixed.
    gap = _sample(
        source,
        4,
        next_sample.source_time_us + 30_000,
        10**18 + 1,
    )
    with pytest.raises(VQ2ImuEstimateUnavailableError, match="timestamp_gap"):
        wrapper.update(gap)
    assert wrapper.last_sample is next_sample
    assert wrapper.last_attitude is propagated

    retry = replace(gap, source_time_us=next_sample.source_time_us + 10_000)
    recovered = wrapper.update(retry)
    assert recovered is not None and recovered.propagated
    assert wrapper.last_sample is retry


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "sample_sequence": 7,
                "source_time_us": 1_010_000,
                "receive_monotonic_ns": 2_000,
            },
            "sequence",
        ),
        (
            {
                "sample_sequence": 6,
                "source_time_us": 1_010_000,
                "receive_monotonic_ns": 2_000,
            },
            "sequence",
        ),
        (
            {
                "sample_sequence": 9,
                "source_time_us": 1_010_000,
                "receive_monotonic_ns": 2_000,
            },
            "contiguous",
        ),
        (
            {
                "sample_sequence": 8,
                "source_time_us": 1_000_000,
                "receive_monotonic_ns": 2_000,
            },
            "source time",
        ),
        (
            {
                "sample_sequence": 8,
                "source_time_us": 999_999,
                "receive_monotonic_ns": 2_000,
            },
            "source time",
        ),
        (
            {
                "sample_sequence": 8,
                "source_time_us": 1_010_000,
                "receive_monotonic_ns": 1_000,
            },
            "receive time",
        ),
        (
            {
                "sample_sequence": 8,
                "source_time_us": 1_010_000,
                "receive_monotonic_ns": 999,
            },
            "receive time",
        ),
    ],
)
def test_duplicate_regressed_or_relabelled_ordering_rejects_transactionally(
    changes,
    message,
):
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())
    first = _sample(source, 7, 1_000_000, 1_000)
    assert wrapper.update(first) is None
    progress = wrapper.calibration_progress

    bad = replace(first, **changes)
    with pytest.raises(VQ2ImuLineageError, match=message):
        wrapper.update(bad)
    assert wrapper.last_sample is first
    assert wrapper.calibration_progress == progress
    assert wrapper.last_attitude is None

    valid = replace(
        first,
        sample_sequence=8,
        source_time_us=1_010_000,
        receive_monotonic_ns=2_000,
    )
    assert wrapper.update(valid) is None
    assert wrapper.last_sample is valid


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "different-session"),
        ("reset_epoch", 4),
        ("host_clock_id", "different-host-clock"),
        ("stream_id", "different-stream"),
        ("generation", 5),
    ],
)
def test_cross_source_epoch_clock_stream_or_generation_is_transactional(field, value):
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())
    first = _sample(source, 0, 1_000_000, 1_000)
    assert wrapper.update(first) is None
    progress = wrapper.calibration_progress

    changed_source = replace(source, **{field: value})
    bad = _sample(changed_source, 1, 1_010_000, 2_000)
    with pytest.raises(VQ2ImuLineageError, match=field):
        wrapper.update(bad)
    assert wrapper.source is source
    assert wrapper.last_sample is first
    assert wrapper.calibration_progress == progress

    valid = _sample(source, 1, 1_010_000, 2_000)
    assert wrapper.update(valid) is None
    assert wrapper.last_sample is valid


def test_nonstationary_calibration_is_consumed_but_never_exposed_as_attitude():
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())
    moving = _sample(
        source,
        0,
        1_000_000,
        1_000,
        gyro=(1.0, 0.0, 0.0),
    )

    assert wrapper.update(moving) is None
    assert wrapper.last_sample is moving
    assert wrapper.last_attitude is None
    assert not wrapper.is_ready
    assert wrapper.calibration_progress == 0.0

    attitude = None
    for offset in range(1, 4):
        attitude = wrapper.update(
            _sample(
                source,
                offset,
                1_000_000 + offset * 10_000,
                1_000 + offset,
            )
        )
    assert attitude is not None and attitude.healthy and attitude.calibrated


def test_source_timestamp_gap_cannot_complete_bootstrap_and_is_transactional():
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())
    first = _sample(source, 0, 1_000_000, 1_000)
    assert wrapper.update(first) is None
    progress = wrapper.calibration_progress

    # The ordering gate compares source microseconds without first converting
    # an attacker-sized integer to float.
    gap = _sample(source, 1, 10**1_000, 1_001)
    with pytest.raises(VQ2ImuEstimateUnavailableError, match="timestamp_gap"):
        wrapper.update(gap)
    assert wrapper.last_sample is first
    assert wrapper.calibration_progress == progress
    assert wrapper.last_attitude is None

    replacement = replace(gap, source_time_us=1_010_000)
    assert wrapper.update(replacement) is None
    assert wrapper.last_sample is replacement


def test_rekey_discards_attitude_and_requires_fresh_bootstrap():
    old_source = _source()
    wrapper = VQ2ImuProvenanceEstimator(old_source, config=_fast_config())
    old_sample, old_attitude = _bootstrap(wrapper)

    with pytest.raises(VQ2ImuLineageError, match="same-identity"):
        wrapper.rekey(old_source)
    assert wrapper.is_ready
    assert wrapper.last_attitude is old_attitude

    with pytest.raises(VQ2ImuLineageError, match="cannot regress"):
        wrapper.rekey(replace(old_source, reset_epoch=2))
    assert wrapper.source is old_source
    assert wrapper.last_sample is old_sample

    new_source = replace(old_source, generation=old_source.generation + 1)
    wrapper.rekey(new_source)
    assert wrapper.source is new_source
    assert not wrapper.is_ready
    assert wrapper.calibration_progress == 0.0
    assert wrapper.last_sample is None
    assert wrapper.last_attitude is None

    stale_old = _sample(
        old_source,
        old_sample.sample_sequence + 1,
        old_sample.source_time_us + 10_000,
        old_sample.receive_monotonic_ns + 1,
    )
    with pytest.raises(VQ2ImuLineageError, match="generation"):
        wrapper.update(stale_old)
    assert wrapper.last_sample is None

    _, new_attitude = _bootstrap(
        wrapper,
        sequence=0,
        source_time_us=0,
        receive_monotonic_ns=20,
    )
    assert new_attitude.source is new_source
    assert new_attitude.calibrated


def test_rekey_cannot_cycle_to_a_retired_source_or_replay_its_sample():
    source_a = _source()
    wrapper = VQ2ImuProvenanceEstimator(source_a, config=_fast_config())
    old_sample, _ = _bootstrap(wrapper)
    source_b = replace(source_a, generation=source_a.generation + 1)
    wrapper.rekey(source_b)

    with pytest.raises(VQ2ImuLineageError, match="previously bound"):
        wrapper.rekey(source_a)
    assert wrapper.source is source_b
    assert wrapper.last_sample is None
    assert wrapper.last_attitude is None
    assert not wrapper.is_ready

    with pytest.raises(VQ2ImuLineageError, match="generation"):
        wrapper.update(old_sample)
    assert wrapper.last_sample is None
    assert not wrapper.is_ready


def test_in_epoch_rekey_cannot_relabel_clock_stream_or_regress_generation():
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())

    with pytest.raises(VQ2ImuLineageError, match="host clock"):
        wrapper.rekey(replace(source, host_clock_id="other-clock", generation=5))
    with pytest.raises(VQ2ImuLineageError, match="stream"):
        wrapper.rekey(replace(source, stream_id="other-stream", generation=5))
    with pytest.raises(VQ2ImuLineageError, match="generation"):
        wrapper.rekey(replace(source, generation=3))

    assert wrapper.source is source
    assert wrapper.last_sample is None
    assert not wrapper.is_ready


def test_new_reset_epoch_rekey_may_change_clock_but_still_starts_uncalibrated():
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())
    _bootstrap(wrapper)
    reset_source = VQ2ImuSource(
        session_id=source.session_id,
        reset_epoch=source.reset_epoch + 1,
        host_clock_id="host-monotonic-2",
        stream_id="highres-imu-1",
        generation=0,
    )

    wrapper.rekey(reset_source)

    assert wrapper.source is reset_source
    assert not wrapper.is_ready
    assert wrapper.last_attitude is None
    assert wrapper.update(_sample(reset_source, 100, 0, 1)) is None
    assert not wrapper.is_ready


def test_wrapper_requires_exact_local_sample_and_config_types():
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())

    with pytest.raises(TypeError, match="exact VQ2TimedImuSample"):
        wrapper.update(object())
    with pytest.raises(TypeError, match="exact ImuAttitudeConfig"):
        VQ2ImuProvenanceEstimator(source, config=object())
    with pytest.raises(TypeError, match="numeric and not bool"):
        VQ2ImuProvenanceEstimator(source, initial_yaw_rad=True)

    assert wrapper.last_sample is None
    assert wrapper.last_attitude is None
    assert not wrapper.is_ready


def test_estimator_timestamp_requires_exact_integer_and_rejects_transactionally(
    monkeypatch,
):
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())
    last_sample, last_attitude = _bootstrap(wrapper)
    next_sample = _sample(
        source,
        last_sample.sample_sequence + 1,
        last_sample.source_time_us + 10_000,
        last_sample.receive_monotonic_ns + 1,
    )
    original_update = ImuAttitudeEstimator.update

    def relabelled_update(estimator, timestamp_us, accel, gyro):
        assert estimator.last_estimate is not None
        return replace(estimator.last_estimate, timestamp_us=float(timestamp_us))

    monkeypatch.setattr(ImuAttitudeEstimator, "update", relabelled_update)
    with pytest.raises(VQ2ImuEstimateUnavailableError, match="source time"):
        wrapper.update(next_sample)
    assert wrapper.last_sample is last_sample
    assert wrapper.last_attitude is last_attitude

    monkeypatch.setattr(ImuAttitudeEstimator, "update", original_update)
    recovered = wrapper.update(next_sample)
    assert recovered is not None
    assert wrapper.last_sample is next_sample


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("unhealthy", "unhealthy attitude"),
        ("healthy_with_reason", "rejection reason"),
        ("malformed_orientation", "invalid calibrated estimate"),
        ("unexpected_type", "unexpected value type"),
        ("none_after_ready", "rejected sample"),
        ("exception", "failed without committing"),
    ],
)
def test_dependency_failures_do_not_advance_estimator_or_lineage(
    monkeypatch,
    mode,
    message,
):
    source = _source()
    wrapper = VQ2ImuProvenanceEstimator(source, config=_fast_config())
    last_sample, last_attitude = _bootstrap(wrapper)
    next_sample = _sample(
        source,
        last_sample.sample_sequence + 1,
        last_sample.source_time_us + 10_000,
        last_sample.receive_monotonic_ns + 1,
    )
    original_update = ImuAttitudeEstimator.update

    def adversarial_update(estimator, timestamp_us, accel, gyro):
        if mode == "exception":
            raise RuntimeError("synthetic estimator failure")
        if mode == "unexpected_type":
            return object()
        if mode == "none_after_ready":
            return None
        assert estimator.last_estimate is not None
        if mode == "unhealthy":
            return replace(
                estimator.last_estimate,
                timestamp_us=timestamp_us,
                healthy=False,
                propagated=False,
                reason="synthetic_unhealthy",
            )
        if mode == "healthy_with_reason":
            return replace(
                estimator.last_estimate,
                timestamp_us=timestamp_us,
                reason="synthetic_reason",
            )
        assert mode == "malformed_orientation"
        return replace(
            estimator.last_estimate,
            timestamp_us=timestamp_us,
            orientation=Quaternion(w=math.nan, x=0.0, y=0.0, z=0.0),
        )

    monkeypatch.setattr(ImuAttitudeEstimator, "update", adversarial_update)
    with pytest.raises(VQ2ImuEstimateUnavailableError, match=message):
        wrapper.update(next_sample)
    assert wrapper.last_sample is last_sample
    assert wrapper.last_attitude is last_attitude

    monkeypatch.setattr(ImuAttitudeEstimator, "update", original_update)
    recovered = wrapper.update(next_sample)
    assert recovered is not None
    assert wrapper.last_sample is next_sample
