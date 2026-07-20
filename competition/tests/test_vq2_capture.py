from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

from competition.adapter import IMUData
from competition.vq2_capture import (
    ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
    HEARTBEAT_MESSAGE_TYPE,
    HIGHRES_IMU_MESSAGE_TYPE,
    HOST_PERF_COUNTER_CLOCK_ID,
    MavlinkIngressV1,
    RACE_STATUS_MESSAGE_TYPE,
    ReceivedIMUSampleV1,
    SUPPORTED_MAVLINK_MESSAGE_TYPES,
)


def _ingress(**overrides: object) -> MavlinkIngressV1:
    values: dict[str, object] = {
        "stream_id": "vq2-mavlink-udp-14550",
        "generation": 3,
        "sequence": 17,
        "message_type": HIGHRES_IMU_MESSAGE_TYPE,
        "host_clock_id": HOST_PERF_COUNTER_CLOCK_ID,
        "received_monotonic_ns": 8_000_000_123,
        "source_time_value": 4_200_000,
        "source_time_unit": "us",
    }
    values.update(overrides)
    return MavlinkIngressV1(**values)  # type: ignore[arg-type]


def _imu(**overrides: object) -> IMUData:
    values: dict[str, object] = {
        "timestamp_us": 4_200_000,
        "accel": (0.1, -0.2, -9.8),
        "gyro": (0.01, -0.02, 0.03),
        "mag": (0.4, 0.5, 0.6),
    }
    values.update(overrides)
    return IMUData(**values)  # type: ignore[arg-type]


def _sample(**overrides: object) -> ReceivedIMUSampleV1:
    values: dict[str, object] = {"ingress": _ingress(), "imu": _imu()}
    values.update(overrides)
    return ReceivedIMUSampleV1(**values)  # type: ignore[arg-type]


def test_supported_ingress_types_and_source_units_round_trip_exactly():
    cases = (
        (HIGHRES_IMU_MESSAGE_TYPE, 1_000, "us"),
        (HEARTBEAT_MESSAGE_TYPE, None, None),
        (RACE_STATUS_MESSAGE_TYPE, 2_000, "ms"),
        (ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE, 3_000, "us"),
    )
    assert SUPPORTED_MAVLINK_MESSAGE_TYPES == frozenset(item[0] for item in cases)

    for sequence, (message_type, source_value, source_unit) in enumerate(cases):
        ingress = _ingress(
            sequence=sequence,
            message_type=message_type,
            source_time_value=source_value,
            source_time_unit=source_unit,
        )
        primitive = ingress.to_primitive()
        assert set(primitive) == {
            "schema",
            "stream_id",
            "generation",
            "sequence",
            "message_type",
            "host_clock_id",
            "received_monotonic_ns",
            "source_time_value",
            "source_time_unit",
        }
        assert primitive["schema"] == "aigp-vq2-mavlink-ingress/1"
        assert primitive["source_time_value"] == source_value
        assert primitive["source_time_unit"] == source_unit
        assert MavlinkIngressV1.from_primitive(primitive) == ingress


def test_source_timestamp_is_optional_but_never_partially_declared():
    without_source = _ingress(source_time_value=None, source_time_unit=None)
    assert without_source.source_time_value is None
    assert without_source.source_time_unit is None

    with pytest.raises(ValueError, match="both present or both absent"):
        _ingress(source_time_value=1, source_time_unit=None)
    with pytest.raises(ValueError, match="both present or both absent"):
        _ingress(source_time_value=None, source_time_unit="us")


@pytest.mark.parametrize(
    ("message_type", "source_value", "source_unit", "match"),
    (
        (HEARTBEAT_MESSAGE_TYPE, 1, "us", "no admitted source"),
        (HIGHRES_IMU_MESSAGE_TYPE, 1, "ms", "must be 'us'"),
        (RACE_STATUS_MESSAGE_TYPE, 1, "us", "must be 'ms'"),
        (ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE, 1, "ms", "must be 'us'"),
    ),
)
def test_message_specific_source_units_fail_closed(
    message_type: str,
    source_value: int,
    source_unit: str,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        _ingress(
            message_type=message_type,
            source_time_value=source_value,
            source_time_unit=source_unit,
        )


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    (
        ("stream_id", "", ValueError),
        ("stream_id", "contains space", ValueError),
        ("stream_id", 3, TypeError),
        ("generation", True, TypeError),
        ("generation", -1, ValueError),
        ("sequence", 1.0, TypeError),
        ("sequence", -1, ValueError),
        ("message_type", "ENCAPSULATED_DATA", ValueError),
        ("message_type", 4, TypeError),
        ("host_clock_id", "host-monotonic", ValueError),
        ("host_clock_id", False, TypeError),
        ("received_monotonic_ns", True, TypeError),
        ("received_monotonic_ns", -1, ValueError),
        ("received_monotonic_ns", 2**64, ValueError),
        ("source_time_value", True, TypeError),
        ("source_time_value", -1, ValueError),
        ("source_time_value", 2**64, ValueError),
        ("source_time_unit", 4, TypeError),
    ),
)
def test_ingress_rejects_coercion_bool_unknown_and_out_of_range_values(
    field: str,
    value: object,
    exception: type[Exception],
):
    with pytest.raises(exception):
        _ingress(**{field: value})


def test_ingress_codec_requires_an_exact_object_schema_and_key_set():
    primitive = _ingress().to_primitive()
    with pytest.raises(TypeError, match="exact object"):
        MavlinkIngressV1.from_primitive(tuple(primitive.items()))

    for mutation in ("missing", "unknown", "schema"):
        malformed = dict(primitive)
        if mutation == "missing":
            del malformed["sequence"]
        elif mutation == "unknown":
            malformed["extra"] = 1
        else:
            malformed["schema"] = "aigp-vq2-mavlink-ingress/2"
        with pytest.raises(ValueError):
            MavlinkIngressV1.from_primitive(malformed)


def test_ingress_is_frozen_and_low_level_corruption_is_revalidated():
    ingress = _ingress()
    with pytest.raises(FrozenInstanceError):
        ingress.sequence = 18  # type: ignore[misc]

    object.__setattr__(ingress, "received_monotonic_ns", False)
    with pytest.raises(TypeError, match="received_monotonic_ns"):
        ingress.validate_integrity()
    with pytest.raises(TypeError, match="received_monotonic_ns"):
        ingress.to_primitive()


def test_received_imu_defensively_copies_and_binds_source_microseconds():
    original = _imu()
    sample = _sample(imu=original)

    assert sample.imu is not original
    assert sample.imu == original
    original.timestamp_us = 99
    original.accel = (9.0, 9.0, 9.0)
    original.gyro = (8.0, 8.0, 8.0)
    original.mag = None
    assert sample.imu.timestamp_us == 4_200_000
    assert sample.imu.accel == (0.1, -0.2, -9.8)
    assert sample.imu.gyro == (0.01, -0.02, 0.03)
    assert sample.imu.mag == (0.4, 0.5, 0.6)

    primitive = sample.to_primitive()
    assert primitive["schema"] == "aigp-vq2-received-imu/1"
    assert primitive["imu"] == {
        "timestamp_us": 4_200_000,
        "accel": [0.1, -0.2, -9.8],
        "gyro": [0.01, -0.02, 0.03],
        "mag": [0.4, 0.5, 0.6],
    }
    restored = ReceivedIMUSampleV1.from_primitive(primitive)
    assert restored == sample
    assert restored.imu is not sample.imu


def test_received_imu_supports_an_explicit_absent_magnetometer():
    sample = _sample(imu=_imu(mag=None))
    assert sample.imu.mag is None
    assert sample.to_primitive()["imu"]["mag"] is None
    assert ReceivedIMUSampleV1.from_primitive(sample.to_primitive()) == sample


@pytest.mark.parametrize(
    ("ingress", "imu", "match"),
    (
        (
            _ingress(
                message_type=ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
                source_time_unit="us",
            ),
            _imu(),
            "must be HIGHRES_IMU",
        ),
        (
            _ingress(source_time_value=None, source_time_unit=None),
            _imu(),
            "must be in us",
        ),
        (_ingress(source_time_value=4_200_001), _imu(), "does not match"),
    ),
)
def test_received_imu_rejects_wrong_message_unit_or_source_binding(
    ingress: MavlinkIngressV1,
    imu: IMUData,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        ReceivedIMUSampleV1(ingress=ingress, imu=imu)


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    (
        ("timestamp_us", True, TypeError),
        ("timestamp_us", -1, ValueError),
        ("timestamp_us", "4200000", TypeError),
        ("accel", [0.1, -0.2, -9.8], TypeError),
        ("accel", (True, 0.0, 0.0), TypeError),
        ("accel", (float("nan"), 0.0, 0.0), ValueError),
        ("gyro", (0.0, 0.0), TypeError),
        ("gyro", (float("inf"), 0.0, 0.0), ValueError),
        ("mag", [0.0, 0.0, 0.0], TypeError),
        ("mag", (0.0, float("-inf"), 0.0), ValueError),
    ),
)
def test_received_imu_rejects_malformed_raw_imu_fields(
    field: str,
    value: object,
    exception: type[Exception],
):
    with pytest.raises(exception):
        _sample(imu=_imu(**{field: value}))


def test_received_imu_requires_exact_nested_types_and_is_frozen():
    class DerivedIMUData(IMUData):
        pass

    with pytest.raises(TypeError, match="exact IMUData"):
        _sample(imu=DerivedIMUData(**vars(_imu())))
    with pytest.raises(TypeError, match="exact MavlinkIngressV1"):
        ReceivedIMUSampleV1(ingress=object(), imu=_imu())  # type: ignore[arg-type]

    sample = _sample()
    with pytest.raises(FrozenInstanceError):
        sample.ingress = _ingress(sequence=18)  # type: ignore[misc]


def test_received_imu_codec_rejects_unknown_missing_and_coerced_nested_fields():
    primitive = _sample().to_primitive()
    with pytest.raises(TypeError, match="exact object"):
        ReceivedIMUSampleV1.from_primitive([])

    for mutation in ("outer_missing", "outer_unknown", "schema"):
        malformed = copy.deepcopy(primitive)
        if mutation == "outer_missing":
            del malformed["ingress"]
        elif mutation == "outer_unknown":
            malformed["extra"] = 1
        else:
            malformed["schema"] = "aigp-vq2-received-imu/2"
        with pytest.raises(ValueError):
            ReceivedIMUSampleV1.from_primitive(malformed)

    for mutation in ("imu_missing", "imu_unknown", "tuple_vector", "bool"):
        malformed = copy.deepcopy(primitive)
        imu_row = malformed["imu"]
        assert type(imu_row) is dict
        if mutation == "imu_missing":
            del imu_row["gyro"]
        elif mutation == "imu_unknown":
            imu_row["extra"] = 1
        elif mutation == "tuple_vector":
            imu_row["accel"] = tuple(imu_row["accel"])
        else:
            imu_row["timestamp_us"] = False
        with pytest.raises((TypeError, ValueError)):
            ReceivedIMUSampleV1.from_primitive(malformed)


def test_nested_mutation_cannot_be_serialized_as_valid_received_imu_evidence():
    sample = _sample()
    sample.imu.timestamp_us += 1
    with pytest.raises(ValueError, match="does not match"):
        sample.validate_integrity()
    with pytest.raises(ValueError, match="does not match"):
        sample.to_primitive()

    fresh = _sample()
    object.__setattr__(fresh.ingress, "source_time_value", False)
    with pytest.raises(TypeError, match="source_time_value"):
        fresh.validate_integrity()
