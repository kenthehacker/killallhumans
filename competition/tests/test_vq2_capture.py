from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

import competition.vq2_capture as capture_contracts
from competition.adapter import IMUData
from competition.vq2_capture import (
    ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
    ActuatorOutputStatusPayloadV1,
    AttitudeTargetOutboundV1,
    AttitudeTargetWireV1,
    CommandLongWireV1,
    GCSHeartbeatWireV1,
    HEARTBEAT_MESSAGE_TYPE,
    HIGHRES_IMU_MESSAGE_TYPE,
    HOST_PERF_COUNTER_CLOCK_ID,
    HeartbeatPayloadV1,
    MavlinkIngressV1,
    NonAttitudeOutboundV1,
    RACE_STATUS_MESSAGE_TYPE,
    RaceStatusPayloadV1,
    ReceivedActuatorOutputStatusV1,
    ReceivedHeartbeatV1,
    ReceivedIMUSampleV1,
    ReceivedRaceStatusV1,
    SUPPORTED_MAVLINK_MESSAGE_TYPES,
    TimesyncWireV1,
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


def _heartbeat_payload(**overrides: object) -> HeartbeatPayloadV1:
    values: dict[str, object] = {"base_mode": 129, "custom_mode": 7}
    values.update(overrides)
    return HeartbeatPayloadV1(**values)  # type: ignore[arg-type]


def _race_payload(**overrides: object) -> RaceStatusPayloadV1:
    values: dict[str, object] = {
        "sim_boot_time_ms": 12_345,
        "race_start_boot_time_ms": -1,
        "race_finish_time_ns": -1,
        "active_gate_index": 0,
        "last_gate_race_time": -1,
    }
    values.update(overrides)
    return RaceStatusPayloadV1(**values)  # type: ignore[arg-type]


def _actuator_payload(**overrides: object) -> ActuatorOutputStatusPayloadV1:
    values: dict[str, object] = {
        "time_usec": 4_200_000,
        "active": 0xF,
        "actuator": tuple(index / 10 for index in range(32)),
    }
    values.update(overrides)
    return ActuatorOutputStatusPayloadV1(**values)  # type: ignore[arg-type]


def _attitude_wire(**overrides: object) -> AttitudeTargetWireV1:
    values: dict[str, object] = {
        "time_boot_ms": 2_345,
        "target_system": 1,
        "target_component": 1,
        "type_mask": 128,
        "q_wxyz": (1.0, 0.0, 0.0, 0.0),
        "body_rates_rad_s": (0.08, -0.06, 0.0),
        "thrust": 0.235,
    }
    values.update(overrides)
    return AttitudeTargetWireV1(**values)  # type: ignore[arg-type]


def _attitude_receipt(**overrides: object) -> AttitudeTargetOutboundV1:
    values: dict[str, object] = {
        "stream_id": "vq2-mavlink-udp-14550",
        "reset_generation": 3,
        "outbound_sequence": 8,
        "host_clock_id": HOST_PERF_COUNTER_CLOCK_ID,
        "call_start_monotonic_ns": 8_100_000_000,
        "call_end_monotonic_ns": 8_100_001_000,
        "api": "send_attitude_rate",
        "outcome": "returned",
        "error_type": None,
        "wire": _attitude_wire(),
    }
    values.update(overrides)
    return AttitudeTargetOutboundV1(**values)  # type: ignore[arg-type]


def _command_long_wire(**overrides: object) -> CommandLongWireV1:
    values: dict[str, object] = {
        "target_system": 1,
        "target_component": 1,
        "command": 400,
        "confirmation": 0,
        "params": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return CommandLongWireV1(**values)  # type: ignore[arg-type]


def _nonattitude_receipt(
    category: str = "arm",
    **overrides: object,
) -> NonAttitudeOutboundV1:
    api: str
    wire: CommandLongWireV1 | TimesyncWireV1 | GCSHeartbeatWireV1
    if category in {"arm", "disarm", "sim_reset"}:
        api = "command_long_send"
        wire = _command_long_wire()
    elif category == "timesync":
        api = "timesync_send"
        wire = TimesyncWireV1(tc1=0, ts1=8_100_000_000)
    else:
        api = "heartbeat_send"
        wire = GCSHeartbeatWireV1(
            type=6,
            autopilot=8,
            base_mode=0,
            custom_mode=0,
            system_status=4,
        )
    values: dict[str, object] = {
        "stream_id": "vq2-mavlink-udp-14550",
        "reset_generation": 3,
        "outbound_sequence": 9,
        "host_clock_id": HOST_PERF_COUNTER_CLOCK_ID,
        "call_start_monotonic_ns": 8_200_000_000,
        "call_end_monotonic_ns": 8_200_001_000,
        "category": category,
        "api": api,
        "outcome": "returned",
        "error_type": None,
        "wire": wire,
    }
    values.update(overrides)
    return NonAttitudeOutboundV1(**values)  # type: ignore[arg-type]


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


def test_new_capture_contracts_are_publicly_exported():
    expected = {
        "ActuatorOutputStatusPayloadV1",
        "AttitudeTargetOutboundV1",
        "AttitudeTargetWireV1",
        "CommandLongWireV1",
        "GCSHeartbeatWireV1",
        "HeartbeatPayloadV1",
        "NonAttitudeOutboundV1",
        "RaceStatusPayloadV1",
        "ReceivedActuatorOutputStatusV1",
        "ReceivedHeartbeatV1",
        "ReceivedRaceStatusV1",
        "TimesyncWireV1",
    }
    assert expected <= set(capture_contracts.__all__)
    for name in expected:
        assert getattr(capture_contracts, name) is not None


def test_received_payload_envelopes_defensively_copy_and_round_trip_exactly():
    heartbeat = _heartbeat_payload()
    race = _race_payload()
    actuator = _actuator_payload()
    cases = (
        (
            ReceivedHeartbeatV1(
                ingress=_ingress(
                    message_type=HEARTBEAT_MESSAGE_TYPE,
                    source_time_value=None,
                    source_time_unit=None,
                ),
                heartbeat=heartbeat,
            ),
            "heartbeat",
            heartbeat,
            ReceivedHeartbeatV1,
            "aigp-vq2-received-heartbeat/1",
            {"base_mode", "custom_mode"},
        ),
        (
            ReceivedRaceStatusV1(
                ingress=_ingress(
                    message_type=RACE_STATUS_MESSAGE_TYPE,
                    source_time_value=race.sim_boot_time_ms,
                    source_time_unit="ms",
                ),
                race_status=race,
            ),
            "race_status",
            race,
            ReceivedRaceStatusV1,
            "aigp-vq2-received-race-status/1",
            {
                "sim_boot_time_ms",
                "race_start_boot_time_ms",
                "race_finish_time_ns",
                "active_gate_index",
                "last_gate_race_time",
            },
        ),
        (
            ReceivedActuatorOutputStatusV1(
                ingress=_ingress(
                    message_type=ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
                    source_time_value=actuator.time_usec,
                    source_time_unit="us",
                ),
                actuator_output_status=actuator,
            ),
            "actuator_output_status",
            actuator,
            ReceivedActuatorOutputStatusV1,
            "aigp-vq2-received-actuator-output-status/1",
            {"time_usec", "active", "actuator"},
        ),
    )

    for envelope, payload_name, original, envelope_type, schema, payload_keys in cases:
        copied = getattr(envelope, payload_name)
        assert copied == original
        assert copied is not original
        primitive = envelope.to_primitive()
        assert set(primitive) == {"schema", "ingress", payload_name}
        assert primitive["schema"] == schema
        assert set(primitive[payload_name]) == payload_keys
        restored = envelope_type.from_primitive(copy.deepcopy(primitive))
        assert restored == envelope
        assert getattr(restored, payload_name) is not copied

    actuator_primitive = cases[2][0].to_primitive()
    assert type(actuator_primitive["actuator_output_status"]["actuator"]) is list
    assert len(actuator_primitive["actuator_output_status"]["actuator"]) == 32


@pytest.mark.parametrize(
    ("factory", "overrides", "exception"),
    (
        (_heartbeat_payload, {"base_mode": True}, TypeError),
        (_heartbeat_payload, {"base_mode": 256}, ValueError),
        (_heartbeat_payload, {"custom_mode": 2**32}, ValueError),
        (_race_payload, {"sim_boot_time_ms": False}, TypeError),
        (_race_payload, {"sim_boot_time_ms": -1}, ValueError),
        (_race_payload, {"race_start_boot_time_ms": 2**63}, ValueError),
        (_race_payload, {"race_finish_time_ns": -(2**63) - 1}, ValueError),
        (_race_payload, {"active_gate_index": 2**32}, ValueError),
        (_race_payload, {"last_gate_race_time": 1.0}, TypeError),
        (_actuator_payload, {"time_usec": True}, TypeError),
        (_actuator_payload, {"active": 2**32}, ValueError),
        (_actuator_payload, {"actuator": [0.0] * 32}, TypeError),
        (_actuator_payload, {"actuator": (0.0,) * 31}, TypeError),
        (
            _actuator_payload,
            {"actuator": (0.0,) * 31 + (float("nan"),)},
            ValueError,
        ),
    ),
)
def test_received_payloads_reject_bool_range_shape_and_nonfinite_values(
    factory: object,
    overrides: dict[str, object],
    exception: type[Exception],
):
    with pytest.raises(exception):
        factory(**overrides)  # type: ignore[operator]


def test_received_envelopes_bind_exact_message_unit_and_source_timestamp():
    with pytest.raises(ValueError, match="must be HEARTBEAT"):
        ReceivedHeartbeatV1(ingress=_ingress(), heartbeat=_heartbeat_payload())

    race = _race_payload()
    with pytest.raises(ValueError, match="must be in ms"):
        ReceivedRaceStatusV1(
            ingress=_ingress(
                message_type=RACE_STATUS_MESSAGE_TYPE,
                source_time_value=None,
                source_time_unit=None,
            ),
            race_status=race,
        )
    with pytest.raises(ValueError, match="does not match"):
        ReceivedRaceStatusV1(
            ingress=_ingress(
                message_type=RACE_STATUS_MESSAGE_TYPE,
                source_time_value=race.sim_boot_time_ms + 1,
                source_time_unit="ms",
            ),
            race_status=race,
        )

    actuator = _actuator_payload()
    with pytest.raises(ValueError, match="must be ACTUATOR_OUTPUT_STATUS"):
        ReceivedActuatorOutputStatusV1(
            ingress=_ingress(),
            actuator_output_status=actuator,
        )
    with pytest.raises(ValueError, match="does not match"):
        ReceivedActuatorOutputStatusV1(
            ingress=_ingress(
                message_type=ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
                source_time_value=actuator.time_usec + 1,
                source_time_unit="us",
            ),
            actuator_output_status=actuator,
        )


def test_received_envelope_codecs_reject_unknown_missing_and_wrong_json_shapes():
    envelopes = (
        ReceivedHeartbeatV1(
            ingress=_ingress(
                message_type=HEARTBEAT_MESSAGE_TYPE,
                source_time_value=None,
                source_time_unit=None,
            ),
            heartbeat=_heartbeat_payload(),
        ),
        ReceivedRaceStatusV1(
            ingress=_ingress(
                message_type=RACE_STATUS_MESSAGE_TYPE,
                source_time_value=12_345,
                source_time_unit="ms",
            ),
            race_status=_race_payload(),
        ),
        ReceivedActuatorOutputStatusV1(
            ingress=_ingress(
                message_type=ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
                source_time_value=4_200_000,
                source_time_unit="us",
            ),
            actuator_output_status=_actuator_payload(),
        ),
    )
    payload_names = ("heartbeat", "race_status", "actuator_output_status")

    for envelope, payload_name in zip(envelopes, payload_names, strict=True):
        envelope_type = type(envelope)
        primitive = envelope.to_primitive()
        with pytest.raises(TypeError, match="exact object"):
            envelope_type.from_primitive(tuple(primitive.items()))

        malformed = copy.deepcopy(primitive)
        malformed["unknown"] = 1
        with pytest.raises(ValueError, match="unknown"):
            envelope_type.from_primitive(malformed)

        malformed = copy.deepcopy(primitive)
        del malformed[payload_name]
        with pytest.raises(ValueError, match="missing"):
            envelope_type.from_primitive(malformed)

        malformed = copy.deepcopy(primitive)
        malformed["schema"] = False
        with pytest.raises(TypeError, match="schema"):
            envelope_type.from_primitive(malformed)

        malformed = copy.deepcopy(primitive)
        malformed[payload_name]["unknown"] = 1
        with pytest.raises(ValueError, match="unknown"):
            envelope_type.from_primitive(malformed)

    malformed_actuator = envelopes[2].to_primitive()
    malformed_actuator["actuator_output_status"]["actuator"] = tuple(
        malformed_actuator["actuator_output_status"]["actuator"]
    )
    with pytest.raises(TypeError, match="array"):
        ReceivedActuatorOutputStatusV1.from_primitive(malformed_actuator)


def test_received_envelopes_are_frozen_and_defensive_against_nested_corruption():
    original = _actuator_payload()
    envelope = ReceivedActuatorOutputStatusV1(
        ingress=_ingress(
            message_type=ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
            source_time_value=original.time_usec,
            source_time_unit="us",
        ),
        actuator_output_status=original,
    )
    with pytest.raises(FrozenInstanceError):
        envelope.actuator_output_status = _actuator_payload()  # type: ignore[misc]

    object.__setattr__(original, "active", True)
    assert envelope.actuator_output_status.active == 0xF
    envelope.to_primitive()

    object.__setattr__(envelope.actuator_output_status, "active", True)
    with pytest.raises(TypeError, match="active"):
        envelope.validate_integrity()
    with pytest.raises(TypeError, match="active"):
        envelope.to_primitive()


def test_attitude_outbound_receipt_defensively_copies_and_round_trips_exactly():
    wire = _attitude_wire(q_wxyz=(1, -0.0, 0.0, 0.0))
    receipt = _attitude_receipt(wire=wire)
    assert receipt.wire == wire
    assert receipt.wire is not wire
    assert receipt.wire.q_wxyz == (1.0, 0.0, 0.0, 0.0)

    primitive = receipt.to_primitive()
    assert set(primitive) == {
        "schema",
        "stream_id",
        "reset_generation",
        "outbound_sequence",
        "host_clock_id",
        "call_start_monotonic_ns",
        "call_end_monotonic_ns",
        "api",
        "outcome",
        "error_type",
        "wire",
    }
    assert primitive["schema"] == "aigp-vq2-attitude-target-outbound/1"
    assert set(primitive["wire"]) == {
        "time_boot_ms",
        "target_system",
        "target_component",
        "type_mask",
        "q_wxyz",
        "body_rates_rad_s",
        "thrust",
    }
    assert type(primitive["wire"]["q_wxyz"]) is list
    assert type(primitive["wire"]["body_rates_rad_s"]) is list
    restored = AttitudeTargetOutboundV1.from_primitive(copy.deepcopy(primitive))
    assert restored == receipt
    assert restored.wire is not receipt.wire


@pytest.mark.parametrize(
    ("overrides", "exception"),
    (
        ({"stream_id": "contains space"}, ValueError),
        ({"reset_generation": True}, TypeError),
        ({"outbound_sequence": -1}, ValueError),
        ({"host_clock_id": "host-monotonic"}, ValueError),
        ({"call_start_monotonic_ns": False}, TypeError),
        ({"call_end_monotonic_ns": 8_000_000_000}, ValueError),
        ({"api": "send_position_target"}, ValueError),
        ({"api": False}, TypeError),
        ({"outcome": "unknown"}, ValueError),
        ({"outcome": False}, TypeError),
        ({"error_type": "RuntimeError"}, ValueError),
        (
            {"outcome": "raised", "error_type": None},
            TypeError,
        ),
        (
            {"outcome": "raised", "error_type": "contains space"},
            ValueError,
        ),
        ({"wire": object()}, TypeError),
    ),
)
def test_attitude_outbound_rejects_bad_metadata_discriminants_and_nested_type(
    overrides: dict[str, object],
    exception: type[Exception],
):
    with pytest.raises(exception):
        _attitude_receipt(**overrides)


@pytest.mark.parametrize(
    ("overrides", "exception"),
    (
        ({"time_boot_ms": True}, TypeError),
        ({"time_boot_ms": 2**32}, ValueError),
        ({"target_system": 256}, ValueError),
        ({"target_component": -1}, ValueError),
        ({"type_mask": False}, TypeError),
        ({"q_wxyz": [1.0, 0.0, 0.0, 0.0]}, TypeError),
        ({"q_wxyz": (1.0, 0.0, 0.0)}, TypeError),
        ({"body_rates_rad_s": (float("inf"), 0.0, 0.0)}, ValueError),
        ({"thrust": True}, TypeError),
        ({"thrust": float("nan")}, ValueError),
    ),
)
def test_attitude_wire_rejects_bool_range_shape_and_nonfinite_values(
    overrides: dict[str, object],
    exception: type[Exception],
):
    with pytest.raises(exception):
        _attitude_wire(**overrides)


@pytest.mark.parametrize(
    ("category", "wire_type", "wire_keys"),
    (
        (
            "arm",
            CommandLongWireV1,
            {"target_system", "target_component", "command", "confirmation", "params"},
        ),
        (
            "disarm",
            CommandLongWireV1,
            {"target_system", "target_component", "command", "confirmation", "params"},
        ),
        (
            "sim_reset",
            CommandLongWireV1,
            {"target_system", "target_component", "command", "confirmation", "params"},
        ),
        ("timesync", TimesyncWireV1, {"tc1", "ts1"}),
        (
            "gcs_heartbeat",
            GCSHeartbeatWireV1,
            {"type", "autopilot", "base_mode", "custom_mode", "system_status"},
        ),
    ),
)
def test_nonattitude_receipt_alternatives_round_trip_with_exact_wire_shape(
    category: str,
    wire_type: type[object],
    wire_keys: set[str],
):
    receipt = _nonattitude_receipt(category)
    original_wire = receipt.wire
    primitive = receipt.to_primitive()
    assert set(primitive) == {
        "schema",
        "stream_id",
        "reset_generation",
        "outbound_sequence",
        "host_clock_id",
        "call_start_monotonic_ns",
        "call_end_monotonic_ns",
        "category",
        "api",
        "outcome",
        "error_type",
        "wire",
    }
    assert primitive["schema"] == "aigp-vq2-nonattitude-outbound/1"
    assert set(primitive["wire"]) == wire_keys
    restored = NonAttitudeOutboundV1.from_primitive(copy.deepcopy(primitive))
    assert restored == receipt
    assert type(restored.wire) is wire_type
    assert restored.wire is not original_wire


@pytest.mark.parametrize(
    ("category", "overrides", "exception"),
    (
        ("unknown", {}, ValueError),
        ("arm", {"api": "timesync_send"}, ValueError),
        ("timesync", {"api": "command_long_send"}, ValueError),
        ("gcs_heartbeat", {"api": False}, TypeError),
        ("arm", {"wire": TimesyncWireV1(tc1=0, ts1=1)}, TypeError),
        ("timesync", {"wire": _command_long_wire()}, TypeError),
        (
            "gcs_heartbeat",
            {"wire": TimesyncWireV1(tc1=0, ts1=1)},
            TypeError,
        ),
        ("arm", {"outcome": "raised", "error_type": None}, TypeError),
    ),
)
def test_nonattitude_receipt_rejects_category_api_wire_and_outcome_mismatch(
    category: str,
    overrides: dict[str, object],
    exception: type[Exception],
):
    with pytest.raises(exception):
        _nonattitude_receipt(category, **overrides)


def test_raised_outbound_receipts_require_and_preserve_exact_error_type():
    attitude = _attitude_receipt(outcome="raised", error_type="RuntimeError")
    nonattitude = _nonattitude_receipt(
        "arm",
        outcome="raised",
        error_type="OSError",
    )
    assert AttitudeTargetOutboundV1.from_primitive(
        attitude.to_primitive()
    ).error_type == "RuntimeError"
    assert NonAttitudeOutboundV1.from_primitive(
        nonattitude.to_primitive()
    ).error_type == "OSError"


@pytest.mark.parametrize(
    ("factory", "overrides", "exception"),
    (
        (_command_long_wire, {"target_system": True}, TypeError),
        (_command_long_wire, {"target_component": 256}, ValueError),
        (_command_long_wire, {"command": 2**16}, ValueError),
        (_command_long_wire, {"confirmation": -1}, ValueError),
        (_command_long_wire, {"params": [0.0] * 7}, TypeError),
        (_command_long_wire, {"params": (0.0,) * 6}, TypeError),
        (
            _command_long_wire,
            {"params": (0.0,) * 6 + (float("-inf"),)},
            ValueError,
        ),
    ),
)
def test_command_long_wire_rejects_bool_range_shape_and_nonfinite_values(
    factory: object,
    overrides: dict[str, object],
    exception: type[Exception],
):
    with pytest.raises(exception):
        factory(**overrides)  # type: ignore[operator]


def test_timesync_and_gcs_heartbeat_wire_integer_ranges_are_exact():
    assert TimesyncWireV1(tc1=-(2**63), ts1=(2**63) - 1).to_primitive() == {
        "tc1": -(2**63),
        "ts1": (2**63) - 1,
    }
    with pytest.raises(TypeError):
        TimesyncWireV1(tc1=True, ts1=0)
    with pytest.raises(ValueError):
        TimesyncWireV1(tc1=-(2**63) - 1, ts1=0)
    with pytest.raises(ValueError):
        TimesyncWireV1(tc1=0, ts1=2**63)

    with pytest.raises(TypeError):
        GCSHeartbeatWireV1(
            type=True,
            autopilot=8,
            base_mode=0,
            custom_mode=0,
            system_status=4,
        )
    with pytest.raises(ValueError):
        GCSHeartbeatWireV1(
            type=6,
            autopilot=8,
            base_mode=0,
            custom_mode=2**32,
            system_status=4,
        )


@pytest.mark.parametrize(
    "receipt",
    (_attitude_receipt(), _nonattitude_receipt("arm")),
)
def test_outbound_codecs_reject_unknown_missing_schema_and_non_json_vectors(
    receipt: AttitudeTargetOutboundV1 | NonAttitudeOutboundV1,
):
    receipt_type = type(receipt)
    primitive = receipt.to_primitive()
    with pytest.raises(TypeError, match="exact object"):
        receipt_type.from_primitive(tuple(primitive.items()))

    malformed = copy.deepcopy(primitive)
    malformed["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        receipt_type.from_primitive(malformed)

    malformed = copy.deepcopy(primitive)
    del malformed["wire"]
    with pytest.raises(ValueError, match="missing"):
        receipt_type.from_primitive(malformed)

    malformed = copy.deepcopy(primitive)
    malformed["schema"] = 1
    with pytest.raises(TypeError, match="schema"):
        receipt_type.from_primitive(malformed)

    malformed = copy.deepcopy(primitive)
    malformed["wire"]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        receipt_type.from_primitive(malformed)

    malformed = copy.deepcopy(primitive)
    vector_name = "q_wxyz" if isinstance(receipt, AttitudeTargetOutboundV1) else "params"
    malformed["wire"][vector_name] = tuple(malformed["wire"][vector_name])
    with pytest.raises(TypeError, match="array"):
        receipt_type.from_primitive(malformed)


def test_outbound_receipts_are_frozen_and_revalidate_low_level_corruption():
    original = _attitude_wire()
    receipt = _attitude_receipt(wire=original)
    with pytest.raises(FrozenInstanceError):
        receipt.outcome = "raised"  # type: ignore[misc]

    object.__setattr__(original, "type_mask", True)
    assert receipt.wire.type_mask == 128
    receipt.to_primitive()

    object.__setattr__(receipt.wire, "thrust", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        receipt.validate_integrity()
    with pytest.raises(ValueError, match="finite"):
        receipt.to_primitive()

    nonattitude = _nonattitude_receipt("timesync")
    object.__setattr__(nonattitude, "api", "heartbeat_send")
    with pytest.raises(ValueError, match="mismatch"):
        nonattitude.validate_integrity()
