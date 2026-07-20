"""Pure passive camera-timing evidence for FlightSim build 3385.

This module has no simulator, socket, replay, reset, arm, command, transport,
or cleanup authority.  It validates one runner observation around an already
published :class:`~competition.vq2_contracts.FrameTimingV1` and summarizes a
single receiver generation deterministically.

All occurrence times belong to the exact host-monotonic clock named by the
nested frame timing.  The camera source timestamp remains an opaque ordering
token and is never subtracted from a host occurrence time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional

from competition.vq2_contracts import (
    FrameTimingV1,
    validate_frame_timing_sequence,
)


CONTROL_PERIOD_NS = 20_000_000

UNMEASURED_PASSIVE_TIMING_ITEMS = (
    "control_scheduler_deadlines",
    "control_scheduler_skip_behavior",
    "command_send_timing",
    "command_to_actuator_causal_response",
    "command_to_gyro_causal_response",
    "camera_measurement_clock_model",
    "imu_measurement_clock_model",
    "command_effect_delay_model",
)

_DISTRIBUTION_NAMES = (
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


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _exact_object(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} fields must be exact; "
            f"missing={sorted(keys - actual)}, unknown={sorted(actual - keys)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class CameraFrameTimingObservationV1:
    """One passively consumed camera frame and its same-clock work stages.

    The nested ``FrameTimingV1`` proves receiver through publication timing.
    These additional points begin only after publication and terminate before
    the caller releases this frame's passive work.  Equal adjacent points are
    allowed because a host clock may observe a zero-duration stage.
    """

    SCHEMA: ClassVar[str] = "aigp-vq2-camera-frame-timing-observation/1"
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "frame_timing",
            "consume_monotonic_ns",
            "work_start_monotonic_ns",
            "detection_start_monotonic_ns",
            "detection_end_monotonic_ns",
            "tracking_start_monotonic_ns",
            "tracking_end_monotonic_ns",
            "work_end_monotonic_ns",
        }
    )

    frame_timing: FrameTimingV1
    consume_monotonic_ns: int
    work_start_monotonic_ns: int
    detection_start_monotonic_ns: int
    detection_end_monotonic_ns: int
    tracking_start_monotonic_ns: int
    tracking_end_monotonic_ns: int
    work_end_monotonic_ns: int

    def __post_init__(self) -> None:
        if type(self.frame_timing) is not FrameTimingV1:
            raise TypeError("frame_timing must be an exact FrameTimingV1")
        points = (
            self.frame_timing.publish_monotonic_ns,
            self.consume_monotonic_ns,
            self.work_start_monotonic_ns,
            self.detection_start_monotonic_ns,
            self.detection_end_monotonic_ns,
            self.tracking_start_monotonic_ns,
            self.tracking_end_monotonic_ns,
            self.work_end_monotonic_ns,
        )
        for index, point in enumerate(points[1:], start=1):
            _exact_nonnegative_int(point, f"observation timing point {index}")
        if any(later < earlier for earlier, later in zip(points, points[1:])):
            raise ValueError(
                "passive observation points must follow frame publication "
                "without regression"
            )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "frame_timing": self.frame_timing.to_primitive(),
            "consume_monotonic_ns": self.consume_monotonic_ns,
            "work_start_monotonic_ns": self.work_start_monotonic_ns,
            "detection_start_monotonic_ns": self.detection_start_monotonic_ns,
            "detection_end_monotonic_ns": self.detection_end_monotonic_ns,
            "tracking_start_monotonic_ns": self.tracking_start_monotonic_ns,
            "tracking_end_monotonic_ns": self.tracking_end_monotonic_ns,
            "work_end_monotonic_ns": self.work_end_monotonic_ns,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "CameraFrameTimingObservationV1":
        row = _exact_object(value, cls._FIELDS, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported camera frame timing observation schema")
        return cls(
            frame_timing=FrameTimingV1.from_primitive(row["frame_timing"]),
            consume_monotonic_ns=row["consume_monotonic_ns"],
            work_start_monotonic_ns=row["work_start_monotonic_ns"],
            detection_start_monotonic_ns=row["detection_start_monotonic_ns"],
            detection_end_monotonic_ns=row["detection_end_monotonic_ns"],
            tracking_start_monotonic_ns=row["tracking_start_monotonic_ns"],
            tracking_end_monotonic_ns=row["tracking_end_monotonic_ns"],
            work_end_monotonic_ns=row["work_end_monotonic_ns"],
        )


@dataclass(frozen=True, slots=True)
class TimingDistribution:
    """Deterministic linear-interpolated nanosecond distribution."""

    count: int
    p50_ns: float
    p95_ns: float
    p99_ns: float
    maximum_ns: int

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 1:
            raise ValueError("distribution count must be a positive exact integer")
        if type(self.maximum_ns) is not int or self.maximum_ns < 0:
            raise ValueError("distribution maximum must be a non-negative exact integer")
        percentiles = (self.p50_ns, self.p95_ns, self.p99_ns)
        if any(type(value) is not float or not math.isfinite(value) for value in percentiles):
            raise ValueError("distribution percentiles must be finite exact floats")
        if not (
            0.0
            <= self.p50_ns
            <= self.p95_ns
            <= self.p99_ns
            <= float(self.maximum_ns)
        ):
            raise ValueError("distribution percentiles must be ordered within the maximum")


@dataclass(frozen=True, slots=True)
class CameraPassiveTimingSummary:
    """Aggregate evidence for one host clock, camera stream, and generation."""

    host_clock_id: str
    stream_id: str
    generation: int
    observation_count: int
    distributions: tuple[tuple[str, TimingDistribution], ...]
    publication_sequence_gap_events: int
    missing_publications: int
    work_over_control_period_count: int
    receiver_frames_reassembled: int
    receiver_frames_decoded: int
    receiver_decode_shortfall: int
    receiver_observation_shortfall: int
    receiver_queue_high_watermark: int
    receiver_queue_capacity: int
    unmeasured_items: tuple[str, ...]

    def distribution(self, name: str) -> Optional[TimingDistribution]:
        return dict(self.distributions).get(name)


def validate_camera_frame_timing_observations(
    observations: tuple[CameraFrameTimingObservationV1, ...],
) -> None:
    """Reject incomparable, repeated, regressing, or overlapping observations."""

    if type(observations) is not tuple:
        raise TypeError("camera timing observations must be an exact tuple")
    if any(type(item) is not CameraFrameTimingObservationV1 for item in observations):
        raise TypeError("camera timing sequence contains a non-contract value")
    if not observations:
        return

    clocks = {item.frame_timing.host_clock_id for item in observations}
    streams = {item.frame_timing.identity.stream_id for item in observations}
    generations = {item.frame_timing.identity.generation for item in observations}
    if len(clocks) != 1:
        raise ValueError("camera timing observations cannot mix host clocks")
    if len(streams) != 1:
        raise ValueError("camera timing observations cannot mix camera streams")
    if len(generations) != 1:
        raise ValueError("camera timing observations cannot mix receiver generations")

    validate_frame_timing_sequence(
        tuple(item.frame_timing for item in observations)
    )
    for previous, current in zip(observations, observations[1:]):
        if current.consume_monotonic_ns <= previous.consume_monotonic_ns:
            raise ValueError("frame consumption time must progress strictly")
        if current.work_start_monotonic_ns < previous.work_end_monotonic_ns:
            raise ValueError("passive frame work cannot overlap across observations")


def summarize_camera_frame_timing_observations(
    observations: tuple[CameraFrameTimingObservationV1, ...],
    *,
    receiver_frames_reassembled: int,
    receiver_frames_decoded: int,
    receiver_queue_high_watermark: int,
    receiver_queue_capacity: int,
) -> CameraPassiveTimingSummary:
    """Summarize validated passive evidence without adding authority claims."""

    validate_camera_frame_timing_observations(observations)
    if len(observations) < 2:
        raise ValueError("a timing summary requires at least two observations")

    reassembled = _exact_nonnegative_int(
        receiver_frames_reassembled, "receiver_frames_reassembled"
    )
    decoded = _exact_nonnegative_int(receiver_frames_decoded, "receiver_frames_decoded")
    queue_high = _exact_nonnegative_int(
        receiver_queue_high_watermark, "receiver_queue_high_watermark"
    )
    queue_capacity = _exact_nonnegative_int(
        receiver_queue_capacity, "receiver_queue_capacity"
    )
    if queue_capacity < 1:
        raise ValueError("receiver_queue_capacity must be >= 1")
    if decoded > reassembled:
        raise ValueError("decoded receiver frames cannot exceed reassembled frames")
    if len(observations) > decoded:
        raise ValueError("timing observations cannot exceed decoded receiver frames")
    if queue_high > queue_capacity:
        raise ValueError("receiver queue high watermark cannot exceed capacity")

    samples: dict[str, list[int]] = {name: [] for name in _DISTRIBUTION_NAMES}
    for item in observations:
        timing = item.frame_timing
        samples["camera_packet_span"].append(
            timing.final_unique_packet_monotonic_ns
            - timing.first_unique_packet_monotonic_ns
        )
        samples["reassembly_after_final_packet"].append(
            timing.reassembly_complete_monotonic_ns
            - timing.final_unique_packet_monotonic_ns
        )
        samples["decode"].append(
            timing.decode_end_monotonic_ns - timing.decode_start_monotonic_ns
        )
        samples["decode_to_publish"].append(
            timing.publish_monotonic_ns - timing.decode_end_monotonic_ns
        )
        samples["camera_first_packet_to_publish"].append(
            timing.publish_monotonic_ns
            - timing.first_unique_packet_monotonic_ns
        )
        samples["publish_to_consume"].append(
            item.consume_monotonic_ns - timing.publish_monotonic_ns
        )
        samples["detection"].append(
            item.detection_end_monotonic_ns - item.detection_start_monotonic_ns
        )
        samples["tracking"].append(
            item.tracking_end_monotonic_ns - item.tracking_start_monotonic_ns
        )
        samples["total_frame_work"].append(
            item.work_end_monotonic_ns - item.work_start_monotonic_ns
        )

    for previous, current in zip(observations, observations[1:]):
        samples["frame_publish_interval"].append(
            current.frame_timing.publish_monotonic_ns
            - previous.frame_timing.publish_monotonic_ns
        )
        samples["frame_consume_interval"].append(
            current.consume_monotonic_ns - previous.consume_monotonic_ns
        )

    sequence_deltas = [
        current.frame_timing.publication_sequence
        - previous.frame_timing.publication_sequence
        - 1
        for previous, current in zip(observations, observations[1:])
    ]
    distributions = tuple(
        (name, _distribution(samples[name])) for name in _DISTRIBUTION_NAMES
    )
    first = observations[0].frame_timing
    return CameraPassiveTimingSummary(
        host_clock_id=first.host_clock_id,
        stream_id=first.identity.stream_id,
        generation=first.identity.generation,
        observation_count=len(observations),
        distributions=distributions,
        publication_sequence_gap_events=sum(delta > 0 for delta in sequence_deltas),
        missing_publications=sum(sequence_deltas),
        work_over_control_period_count=sum(
            value > CONTROL_PERIOD_NS for value in samples["total_frame_work"]
        ),
        receiver_frames_reassembled=reassembled,
        receiver_frames_decoded=decoded,
        receiver_decode_shortfall=reassembled - decoded,
        receiver_observation_shortfall=decoded - len(observations),
        receiver_queue_high_watermark=queue_high,
        receiver_queue_capacity=queue_capacity,
        unmeasured_items=UNMEASURED_PASSIVE_TIMING_ITEMS,
    )


def _distribution(values: list[int]) -> TimingDistribution:
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("timing distributions require non-negative exact integers")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return TimingDistribution(
        count=len(ordered),
        p50_ns=percentile(0.50),
        p95_ns=percentile(0.95),
        p99_ns=percentile(0.99),
        maximum_ns=ordered[-1],
    )


__all__ = [
    "CONTROL_PERIOD_NS",
    "UNMEASURED_PASSIVE_TIMING_ITEMS",
    "CameraFrameTimingObservationV1",
    "TimingDistribution",
    "CameraPassiveTimingSummary",
    "validate_camera_frame_timing_observations",
    "summarize_camera_frame_timing_observations",
]
