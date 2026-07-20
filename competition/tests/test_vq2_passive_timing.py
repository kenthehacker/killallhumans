"""Direct contract and aggregate tests for passive VQ2 camera timing."""

from __future__ import annotations

from copy import deepcopy

import pytest

from competition.vq2_contracts import FrameIdentityV1, FrameTimingV1
from competition.vq2_passive_timing import (
    CONTROL_PERIOD_NS,
    UNMEASURED_PASSIVE_TIMING_ITEMS,
    CameraFrameTimingObservationV1,
    summarize_camera_frame_timing_observations,
    validate_camera_frame_timing_observations,
)


def _timing(
    index: int = 0,
    *,
    base_ns: int | None = None,
    publication_sequence: int | None = None,
    frame_id: int | None = None,
    generation: int = 0,
    stream_id: str = "vq2-camera-udp-5600",
    host_clock_id: str = "host-perf-counter",
    source_time_ns: int | None = None,
    packet_span_ns: int = 10,
    reassembly_tail_ns: int = 2,
    decode_ns: int = 4,
    decode_to_publish_ns: int = 6,
) -> FrameTimingV1:
    base = index * 100_000_000 if base_ns is None else base_ns
    final = base + packet_span_ns
    reassembled = final + reassembly_tail_ns
    decode_end = reassembled + decode_ns
    published = decode_end + decode_to_publish_ns
    return FrameTimingV1(
        identity=FrameIdentityV1(
            stream_id=stream_id,
            generation=generation,
            frame_id=index if frame_id is None else frame_id,
        ),
        camera_source_time_ns=(
            1_000_000 + index if source_time_ns is None else source_time_ns
        ),
        host_clock_id=host_clock_id,
        publication_sequence=(
            index + 1 if publication_sequence is None else publication_sequence
        ),
        first_unique_packet_monotonic_ns=base,
        final_unique_packet_monotonic_ns=final,
        reassembly_complete_monotonic_ns=reassembled,
        decode_start_monotonic_ns=reassembled,
        decode_end_monotonic_ns=decode_end,
        publish_monotonic_ns=published,
    )


def _observation(
    index: int = 0,
    *,
    timing: FrameTimingV1 | None = None,
    publish_to_consume_ns: int = 8,
    detection_ns: int = 10,
    tracking_ns: int = 12,
    work_padding_ns: int = 0,
) -> CameraFrameTimingObservationV1:
    frame_timing = _timing(index) if timing is None else timing
    consumed = frame_timing.publish_monotonic_ns + publish_to_consume_ns
    work_start = consumed
    detection_start = work_start
    detection_end = detection_start + detection_ns
    tracking_start = detection_end
    tracking_end = tracking_start + tracking_ns
    work_end = tracking_end + work_padding_ns
    return CameraFrameTimingObservationV1(
        frame_timing=frame_timing,
        consume_monotonic_ns=consumed,
        work_start_monotonic_ns=work_start,
        detection_start_monotonic_ns=detection_start,
        detection_end_monotonic_ns=detection_end,
        tracking_start_monotonic_ns=tracking_start,
        tracking_end_monotonic_ns=tracking_end,
        work_end_monotonic_ns=work_end,
    )


def test_observation_schema_is_exact_and_round_trips_bit_exactly():
    observation = _observation()
    primitive = observation.to_primitive()

    assert primitive["schema"] == "aigp-vq2-camera-frame-timing-observation/1"
    assert set(primitive) == CameraFrameTimingObservationV1._FIELDS
    assert primitive["frame_timing"] == observation.frame_timing.to_primitive()
    assert CameraFrameTimingObservationV1.from_primitive(primitive) == observation
    assert CameraFrameTimingObservationV1.from_primitive(
        observation.to_primitive()
    ).to_primitive() == primitive


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("consume_monotonic_ns", True, TypeError),
        ("work_start_monotonic_ns", -1, ValueError),
        ("detection_start_monotonic_ns", 1, ValueError),
        ("detection_end_monotonic_ns", 1, ValueError),
        ("tracking_start_monotonic_ns", 1, ValueError),
        ("tracking_end_monotonic_ns", 1, ValueError),
        ("work_end_monotonic_ns", 1, ValueError),
    ],
)
def test_observation_rejects_nonexact_or_regressing_stage_points(field, value, error):
    fields = _observation().to_primitive()
    fields.pop("schema")
    fields["frame_timing"] = FrameTimingV1.from_primitive(fields["frame_timing"])
    fields[field] = value
    with pytest.raises(error):
        CameraFrameTimingObservationV1(**fields)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "schema", "nested"])
def test_observation_codec_rejects_nonexact_or_mislabeled_objects(mutation):
    primitive = deepcopy(_observation().to_primitive())
    if mutation == "missing":
        primitive.pop("work_end_monotonic_ns")
    elif mutation == "unknown":
        primitive["authority"] = "none"
    elif mutation == "schema":
        primitive["schema"] = "aigp-vq2-camera-frame-timing-observation/2"
    else:
        primitive["frame_timing"]["unknown"] = 1

    with pytest.raises((TypeError, ValueError)):
        CameraFrameTimingObservationV1.from_primitive(primitive)


def test_observation_rejects_a_noncontract_nested_timing():
    fields = _observation().to_primitive()
    fields.pop("schema")
    with pytest.raises(TypeError, match="exact FrameTimingV1"):
        CameraFrameTimingObservationV1(**fields)


def test_sequence_accepts_uint32_frame_wrap_without_using_frame_id_as_order():
    first = _observation(
        timing=_timing(
            0,
            frame_id=(1 << 32) - 1,
            publication_sequence=10,
            source_time_ns=20,
        )
    )
    second = _observation(
        1,
        timing=_timing(
            1,
            frame_id=0,
            publication_sequence=11,
            source_time_ns=21,
        ),
    )

    validate_camera_frame_timing_observations((first, second))


@pytest.mark.parametrize(
    "replacement",
    [
        _timing(1, host_clock_id="different-clock"),
        _timing(1, stream_id="different-camera"),
        _timing(1, generation=1),
    ],
    ids=("clock", "stream", "generation"),
)
def test_sequence_rejects_mixed_clock_stream_or_generation(replacement):
    with pytest.raises(ValueError, match="cannot mix"):
        validate_camera_frame_timing_observations(
            (_observation(0), _observation(1, timing=replacement))
        )


@pytest.mark.parametrize(
    "replacement",
    [
        _timing(1, frame_id=0),
        _timing(1, publication_sequence=1),
        _timing(1, base_ns=0, publication_sequence=2, source_time_ns=2_000_000),
        _timing(1, source_time_ns=1_000_000),
    ],
    ids=("duplicate-frame", "publication-regression", "publish-regression", "source-regression"),
)
def test_sequence_rejects_duplicate_or_regressing_frame_evidence(replacement):
    with pytest.raises(ValueError):
        validate_camera_frame_timing_observations(
            (_observation(0), _observation(1, timing=replacement))
        )


def test_sequence_rejects_consume_regression_and_overlapping_passive_work():
    first = _observation(0, work_padding_ns=100_000_000)
    second = _observation(1)
    with pytest.raises(ValueError, match="overlap"):
        validate_camera_frame_timing_observations((first, second))

    delayed_first = _observation(0, publish_to_consume_ns=200_000_000)
    regressed = _observation(1)
    with pytest.raises(ValueError, match="consumption time"):
        validate_camera_frame_timing_observations((delayed_first, regressed))


def test_sequence_requires_an_exact_tuple_and_contract_values():
    with pytest.raises(TypeError, match="exact tuple"):
        validate_camera_frame_timing_observations([_observation(0)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-contract"):
        validate_camera_frame_timing_observations((_observation(0), object()))  # type: ignore[arg-type]


def test_summary_reports_all_distributions_gaps_shortfalls_and_unmeasured_items():
    observations = tuple(
        _observation(
            index,
            timing=_timing(
                index,
                publication_sequence=(1, 2, 5, 6)[index],
                packet_span_ns=(10, 20, 30, 40)[index],
                reassembly_tail_ns=(1, 2, 3, 4)[index],
                decode_ns=(2, 4, 6, 8)[index],
                decode_to_publish_ns=(3, 6, 9, 12)[index],
            ),
            publish_to_consume_ns=(4, 8, 12, 16)[index],
            detection_ns=(5, 10, 15, 20)[index],
            tracking_ns=(6, 12, 18, 24)[index],
            work_padding_ns=(0, CONTROL_PERIOD_NS, 0, 0)[index],
        )
        for index in range(4)
    )

    summary = summarize_camera_frame_timing_observations(
        observations,
        receiver_frames_reassembled=7,
        receiver_frames_decoded=6,
        receiver_queue_high_watermark=3,
        receiver_queue_capacity=8,
    )

    packet = summary.distribution("camera_packet_span")
    assert packet is not None
    assert packet.count == 4
    assert packet.p50_ns == pytest.approx(25.0)
    assert packet.p95_ns == pytest.approx(38.5)
    assert packet.p99_ns == pytest.approx(39.7)
    assert packet.maximum_ns == 40
    assert summary.distribution("frame_publish_interval").count == 3  # type: ignore[union-attr]
    assert summary.distribution("frame_consume_interval").count == 3  # type: ignore[union-attr]
    assert summary.distribution("unknown") is None
    assert tuple(name for name, _distribution in summary.distributions) == (
        "camera_packet_span",
        "reassembly_after_final_packet",
        "decode",
        "decode_to_publish",
        "camera_first_packet_to_publish",
        "publish_to_consume",
        "detection",
        "tracking",
        "total_frame_work",
        "frame_publish_interval",
        "frame_consume_interval",
    )
    assert summary.host_clock_id == "host-perf-counter"
    assert summary.stream_id == "vq2-camera-udp-5600"
    assert summary.generation == 0
    assert summary.observation_count == 4
    assert summary.publication_sequence_gap_events == 1
    assert summary.missing_publications == 2
    assert summary.work_over_control_period_count == 1
    assert summary.receiver_frames_reassembled == 7
    assert summary.receiver_frames_decoded == 6
    assert summary.receiver_decode_shortfall == 1
    assert summary.receiver_observation_shortfall == 2
    assert summary.receiver_queue_high_watermark == 3
    assert summary.receiver_queue_capacity == 8
    assert summary.unmeasured_items == UNMEASURED_PASSIVE_TIMING_ITEMS
    assert "control_scheduler_deadlines" in summary.unmeasured_items
    assert "command_send_timing" in summary.unmeasured_items
    assert "command_to_actuator_causal_response" in summary.unmeasured_items
    assert "command_to_gyro_causal_response" in summary.unmeasured_items
    assert "camera_measurement_clock_model" in summary.unmeasured_items
    assert "calibrated_camera_imu_offset" in summary.unmeasured_items


@pytest.mark.parametrize(
    "overrides",
    [
        {"receiver_frames_reassembled": True},
        {"receiver_frames_reassembled": -1},
        {"receiver_frames_reassembled": 1, "receiver_frames_decoded": 2},
        {"receiver_frames_reassembled": 2, "receiver_frames_decoded": 1},
        {"receiver_queue_high_watermark": 9},
        {"receiver_queue_capacity": 0},
    ],
    ids=(
        "bool",
        "negative",
        "decoded-exceeds-reassembled",
        "observations-exceed-decoded",
        "queue-exceeds-capacity",
        "zero-capacity",
    ),
)
def test_summary_rejects_invalid_receiver_diagnostics(overrides):
    arguments = {
        "receiver_frames_reassembled": 2,
        "receiver_frames_decoded": 2,
        "receiver_queue_high_watermark": 1,
        "receiver_queue_capacity": 8,
        **overrides,
    }
    with pytest.raises((TypeError, ValueError)):
        summarize_camera_frame_timing_observations(
            (_observation(0), _observation(1)), **arguments
        )


def test_summary_rejects_insufficient_or_non_tuple_evidence():
    arguments = {
        "receiver_frames_reassembled": 2,
        "receiver_frames_decoded": 2,
        "receiver_queue_high_watermark": 0,
        "receiver_queue_capacity": 8,
    }
    with pytest.raises(ValueError, match="at least two"):
        summarize_camera_frame_timing_observations((_observation(0),), **arguments)
    with pytest.raises(TypeError, match="exact tuple"):
        summarize_camera_frame_timing_observations(  # type: ignore[arg-type]
            [_observation(0), _observation(1)], **arguments
        )
