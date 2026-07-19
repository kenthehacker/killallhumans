"""Pure runtime timing primitives for the build-3385 VQ2 stack.

This module owns no simulator, transport, reset, arm, gate, or cleanup
authority.  It turns the frozen ``/1`` timing values into three small runtime
tools:

* :class:`LatestFrameCursorV1` consumes a latest-value camera snapshot once;
* :class:`FixedRateControlSchedulerV1` exposes one bounded 50 Hz tick at a
  time and explicitly skips a tick whose deadline has already elapsed;
* latency helpers build and summarize exact :class:`LatencyEventV1` traces.

All scheduling uses integer host-monotonic nanoseconds.  Camera source time is
handled only as the opaque token carried by :class:`FrameTimingV1`.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Optional

from competition.vq2_contracts import (
    EventOutcome,
    FrameIdentityV1,
    FrameTimingV1,
    LatencyEventKind,
    LatencyEventV1,
    validate_frame_timing_sequence,
    validate_latency_event_sequence,
)


MINIMUM_CONTROL_PERIOD_NS = 20_000_000
VQ2_HOST_CLOCK_ID = "host-perf-counter"


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


@dataclass(frozen=True, slots=True)
class LatestFrameSelectionV1:
    """One newly selected latest-value snapshot and its observed gap."""

    snapshot: Any
    timing: FrameTimingV1
    overwritten_publications: int

    def __post_init__(self) -> None:
        if type(self.timing) is not FrameTimingV1:
            raise TypeError("timing must be FrameTimingV1")
        _exact_nonnegative_int(
            self.overwritten_publications, "overwritten_publications"
        )


class LatestFrameCursorV1:
    """Consume each distinct publication from a latest-value slot at most once.

    A publication-sequence gap reports intermediate frames that were
    overwritten before the consumer polled; no backlog is queued or replayed.
    Re-reading the exact current snapshot returns ``None``.  Relabeling the
    same frame identity, changing host clocks/streams, or presenting regressing
    timing fails closed.
    """

    def __init__(
        self,
        *,
        expected_host_clock_id: Optional[str] = None,
        expected_stream_id: Optional[str] = None,
    ) -> None:
        if expected_host_clock_id is not None and (
            type(expected_host_clock_id) is not str or not expected_host_clock_id
        ):
            raise ValueError("expected_host_clock_id must be a non-empty string")
        if expected_stream_id is not None and (
            type(expected_stream_id) is not str or not expected_stream_id
        ):
            raise ValueError("expected_stream_id must be a non-empty string")
        self._expected_host_clock_id = expected_host_clock_id
        self._expected_stream_id = expected_stream_id
        self._previous: Optional[FrameTimingV1] = None
        self._repeated_reads = 0
        self._overwritten_publications = 0

    @property
    def repeated_reads(self) -> int:
        return self._repeated_reads

    @property
    def overwritten_publications(self) -> int:
        return self._overwritten_publications

    @property
    def previous_timing(self) -> Optional[FrameTimingV1]:
        return self._previous

    def select(self, snapshot: Any) -> Optional[LatestFrameSelectionV1]:
        timing = getattr(snapshot, "timing", None)
        if type(timing) is not FrameTimingV1:
            raise TypeError("latest-value snapshot requires exact FrameTimingV1 timing")
        identity = timing.identity
        try:
            snapshot_identity = (snapshot.generation, snapshot.frame_id)
            source_token = snapshot.sim_time_ns
        except AttributeError as exc:
            raise TypeError("latest-value snapshot is missing its frame identity") from exc
        if any(type(value) is not int for value in (*snapshot_identity, source_token)):
            raise TypeError("snapshot frame identity and source token must be exact integers")
        if snapshot_identity != (identity.generation, identity.frame_id):
            raise ValueError("snapshot identity does not match its frame timing")
        if source_token != timing.camera_source_time_ns:
            raise ValueError("snapshot source token does not match its frame timing")

        expected_clock = self._expected_host_clock_id
        if expected_clock is not None and timing.host_clock_id != expected_clock:
            raise ValueError("snapshot changed the expected host monotonic clock")
        expected_stream = self._expected_stream_id
        if expected_stream is not None and identity.stream_id != expected_stream:
            raise ValueError("snapshot changed the expected camera stream")

        previous = self._previous
        if previous is not None and timing.identity == previous.identity:
            if timing != previous:
                raise ValueError("a frame identity cannot be relabeled with new timing")
            self._repeated_reads += 1
            return None

        if previous is not None:
            if timing.host_clock_id != previous.host_clock_id:
                raise ValueError("latest-value cursor cannot cross host clock domains")
            if timing.identity.stream_id != previous.identity.stream_id:
                raise ValueError("latest-value cursor cannot switch camera streams")
            validate_frame_timing_sequence((previous, timing))
            overwritten = timing.publication_sequence - previous.publication_sequence - 1
        else:
            overwritten = 0

        selection = LatestFrameSelectionV1(
            snapshot=snapshot,
            timing=timing,
            overwritten_publications=overwritten,
        )
        self._previous = timing
        self._overwritten_publications += overwritten
        return selection


class LatencyTraceRecorderV1:
    """Thread-safe append-only builder for one host-clock latency trace."""

    def __init__(
        self,
        host_clock_id: str = VQ2_HOST_CLOCK_ID,
        *,
        starting_event_sequence: int = 1,
    ) -> None:
        if type(host_clock_id) is not str or not host_clock_id:
            raise ValueError("host_clock_id must be a non-empty string")
        first = _exact_nonnegative_int(
            starting_event_sequence, "starting_event_sequence"
        )
        LatencyEventV1(
            event_sequence=first,
            host_clock_id=host_clock_id,
            monotonic_ns=0,
            kind=LatencyEventKind.CONTROL_TICK_DUE,
            frame=None,
            control_tick_id=0,
            command_id=None,
            sensor_sample_id=None,
            sensor_source_time_ns=None,
            outcome=EventOutcome.OK,
            reason_code=None,
            queue_depth=0,
        )
        self.host_clock_id = host_clock_id
        self._next_event_sequence = first
        self._events: list[LatencyEventV1] = []
        self._last_monotonic_ns: Optional[int] = None
        self._lock = threading.Lock()

    def record(
        self,
        kind: LatencyEventKind,
        monotonic_ns: int,
        *,
        frame: Optional[FrameIdentityV1] = None,
        control_tick_id: Optional[int] = None,
        command_id: Optional[int] = None,
        sensor_sample_id: Optional[int] = None,
        sensor_source_time_ns: Optional[int] = None,
        outcome: EventOutcome = EventOutcome.OK,
        reason_code: Optional[str] = None,
        queue_depth: Optional[int] = None,
    ) -> LatencyEventV1:
        timestamp = _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        with self._lock:
            if (
                self._last_monotonic_ns is not None
                and timestamp < self._last_monotonic_ns
            ):
                raise ValueError("latency event occurrence time regressed")
            event = LatencyEventV1(
                event_sequence=self._next_event_sequence,
                host_clock_id=self.host_clock_id,
                monotonic_ns=timestamp,
                kind=kind,
                frame=frame,
                control_tick_id=control_tick_id,
                command_id=command_id,
                sensor_sample_id=sensor_sample_id,
                sensor_source_time_ns=sensor_source_time_ns,
                outcome=outcome,
                reason_code=reason_code,
                queue_depth=queue_depth,
            )
            self._events.append(event)
            self._next_event_sequence += 1
            self._last_monotonic_ns = timestamp
            return event

    def snapshot(self, *, validate: bool = True) -> tuple[LatencyEventV1, ...]:
        with self._lock:
            result = tuple(self._events)
        if validate:
            validate_latency_event_sequence(result)
        return result


@dataclass(frozen=True, slots=True)
class ControlTickLeaseV1:
    """A single scheduler-owned tick; it carries no command authority."""

    control_tick_id: int
    due_monotonic_ns: int
    deadline_monotonic_ns: int
    start_monotonic_ns: int
    frame: Optional[FrameIdentityV1]

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.control_tick_id, "control_tick_id")
        due = _exact_nonnegative_int(self.due_monotonic_ns, "due_monotonic_ns")
        deadline = _exact_nonnegative_int(
            self.deadline_monotonic_ns, "deadline_monotonic_ns"
        )
        start = _exact_nonnegative_int(self.start_monotonic_ns, "start_monotonic_ns")
        if not due <= start <= deadline:
            raise ValueError("control tick lease must start inside its deadline window")
        if type(self.frame) not in {FrameIdentityV1, type(None)}:
            raise TypeError("frame must be FrameIdentityV1 or None")


class FixedRateControlSchedulerV1:
    """One-at-a-time fixed-rate scheduler with no catch-up path.

    The scheduler emits a due event only when the caller polls a due tick.  If
    that poll occurs after the tick's one-period validity window, the tick is
    recorded as deadline-missed and skipped, and the next due time is rebased
    a full period after the observation.  Work that fits retains the nominal
    grid; work that crosses the next nominal slot is likewise rebased a full
    period after completion.  Consequently a stall can reduce the rate but can
    never create back-to-back catch-up ticks.
    """

    def __init__(
        self,
        *,
        start_monotonic_ns: int,
        host_clock_id: str = VQ2_HOST_CLOCK_ID,
        period_ns: int = MINIMUM_CONTROL_PERIOD_NS,
        first_control_tick_id: int = 0,
        trace: Optional[LatencyTraceRecorderV1] = None,
    ) -> None:
        start = _exact_nonnegative_int(start_monotonic_ns, "start_monotonic_ns")
        period = _exact_nonnegative_int(period_ns, "period_ns")
        if period < MINIMUM_CONTROL_PERIOD_NS:
            raise ValueError("control period cannot exceed the reviewed 50 Hz cap")
        tick = _exact_nonnegative_int(
            first_control_tick_id, "first_control_tick_id"
        )
        if trace is not None and type(trace) is not LatencyTraceRecorderV1:
            raise TypeError("trace must be LatencyTraceRecorderV1 or None")
        self.period_ns = period
        self.trace = trace or LatencyTraceRecorderV1(host_clock_id)
        if self.trace.host_clock_id != host_clock_id:
            raise ValueError("scheduler and latency trace host clocks must match")
        self._next_due_ns = start
        self._next_tick_id = tick
        self._active: Optional[ControlTickLeaseV1] = None
        self._last_due_event_ns: Optional[int] = None

    @property
    def next_due_monotonic_ns(self) -> int:
        return self._next_due_ns

    @property
    def next_control_tick_id(self) -> int:
        return self._next_tick_id

    @property
    def active(self) -> Optional[ControlTickLeaseV1]:
        return self._active

    def wait_ns(self, now_monotonic_ns: int) -> int:
        now = _exact_nonnegative_int(now_monotonic_ns, "now_monotonic_ns")
        return max(0, self._next_due_ns - now)

    def begin_due(
        self,
        now_monotonic_ns: int,
        *,
        frame: Optional[FrameIdentityV1] = None,
        queue_depth: Optional[int] = 0,
    ) -> Optional[ControlTickLeaseV1]:
        """Start the due tick, or record and skip it if already expired."""

        now = _exact_nonnegative_int(now_monotonic_ns, "now_monotonic_ns")
        if self._active is not None:
            raise RuntimeError("a control tick is already active")
        if now < self._next_due_ns:
            return None
        if type(frame) not in {FrameIdentityV1, type(None)}:
            raise TypeError("frame must be FrameIdentityV1 or None")

        tick_id = self._next_tick_id
        scheduled_due = self._next_due_ns
        deadline = scheduled_due + self.period_ns
        self.trace.record(
            LatencyEventKind.CONTROL_TICK_DUE,
            now,
            frame=frame,
            control_tick_id=tick_id,
            queue_depth=queue_depth,
        )
        self._last_due_event_ns = now
        if now > deadline:
            self.trace.record(
                LatencyEventKind.DEADLINE_MISSED,
                now,
                frame=frame,
                control_tick_id=tick_id,
                outcome=EventOutcome.SKIPPED,
                reason_code="tick_deadline_elapsed",
                queue_depth=queue_depth,
            )
            self.trace.record(
                LatencyEventKind.CONTROL_TICK_SKIPPED,
                now,
                frame=frame,
                control_tick_id=tick_id,
                outcome=EventOutcome.SKIPPED,
                reason_code="tick_deadline_elapsed",
                queue_depth=queue_depth,
            )
            self._advance_after(now)
            return None

        lease = ControlTickLeaseV1(
            control_tick_id=tick_id,
            due_monotonic_ns=scheduled_due,
            deadline_monotonic_ns=deadline,
            start_monotonic_ns=now,
            frame=frame,
        )
        self.trace.record(
            LatencyEventKind.CONTROL_TICK_START,
            now,
            frame=frame,
            control_tick_id=tick_id,
            queue_depth=queue_depth,
        )
        self._active = lease
        return lease

    def skip_due(
        self,
        now_monotonic_ns: int,
        *,
        reason_code: str,
        frame: Optional[FrameIdentityV1] = None,
        queue_depth: Optional[int] = 0,
    ) -> bool:
        """Explicitly skip one due, not-yet-started tick."""

        now = _exact_nonnegative_int(now_monotonic_ns, "now_monotonic_ns")
        if self._active is not None:
            raise RuntimeError("an active control tick cannot be skipped")
        if now < self._next_due_ns:
            return False
        if type(frame) not in {FrameIdentityV1, type(None)}:
            raise TypeError("frame must be FrameIdentityV1 or None")
        tick_id = self._next_tick_id
        deadline = self._next_due_ns + self.period_ns
        missed = now > deadline
        effective_reason = "tick_deadline_elapsed" if missed else reason_code
        # Validate caller-owned skip metadata before appending the due event;
        # a bad reason or queue depth must not leave a half-mutated scheduler
        # trace that can no longer be resumed.
        LatencyEventV1(
            event_sequence=0,
            host_clock_id=self.trace.host_clock_id,
            monotonic_ns=now,
            kind=LatencyEventKind.CONTROL_TICK_SKIPPED,
            frame=frame,
            control_tick_id=tick_id,
            command_id=None,
            sensor_sample_id=None,
            sensor_source_time_ns=None,
            outcome=EventOutcome.SKIPPED,
            reason_code=effective_reason,
            queue_depth=queue_depth,
        )
        self.trace.record(
            LatencyEventKind.CONTROL_TICK_DUE,
            now,
            frame=frame,
            control_tick_id=tick_id,
            queue_depth=queue_depth,
        )
        self._last_due_event_ns = now
        if missed:
            self.trace.record(
                LatencyEventKind.DEADLINE_MISSED,
                now,
                frame=frame,
                control_tick_id=tick_id,
                outcome=EventOutcome.SKIPPED,
                reason_code=effective_reason,
                queue_depth=queue_depth,
            )
        self.trace.record(
            LatencyEventKind.CONTROL_TICK_SKIPPED,
            now,
            frame=frame,
            control_tick_id=tick_id,
            outcome=EventOutcome.SKIPPED,
            reason_code=effective_reason,
            queue_depth=queue_depth,
        )
        self._advance_after(now)
        return True

    def finish(
        self,
        lease: ControlTickLeaseV1,
        now_monotonic_ns: int,
        *,
        outcome: EventOutcome = EventOutcome.OK,
        reason_code: Optional[str] = None,
        queue_depth: Optional[int] = 0,
    ) -> LatencyEventV1:
        """End the active tick and rebase the next due time safely."""

        now = _exact_nonnegative_int(now_monotonic_ns, "now_monotonic_ns")
        if type(lease) is not ControlTickLeaseV1 or lease != self._active:
            raise ValueError("only the active scheduler lease can finish")
        if now < lease.start_monotonic_ns:
            raise ValueError("control tick end cannot predate its start")
        if outcome not in {EventOutcome.OK, EventOutcome.ERROR}:
            raise ValueError("control tick end outcome must be ok or error")
        actual_outcome = outcome
        actual_reason = reason_code
        if now > lease.deadline_monotonic_ns:
            actual_outcome = EventOutcome.ERROR
            actual_reason = "tick_work_overrun"
        elif actual_reason == "tick_work_overrun":
            raise ValueError("tick_work_overrun is reserved for an elapsed deadline")
        event = self.trace.record(
            LatencyEventKind.CONTROL_TICK_END,
            now,
            frame=lease.frame,
            control_tick_id=lease.control_tick_id,
            outcome=actual_outcome,
            reason_code=actual_reason,
            queue_depth=queue_depth,
        )
        self._active = None
        self._advance_after(now)
        return event

    def _advance_after(self, now_monotonic_ns: int) -> None:
        self._next_tick_id += 1
        nominal_next_due = self._next_due_ns + self.period_ns
        if now_monotonic_ns <= nominal_next_due:
            minimum_observed_spacing = (
                nominal_next_due
                if self._last_due_event_ns is None
                else self._last_due_event_ns + self.period_ns
            )
            self._next_due_ns = max(nominal_next_due, minimum_observed_spacing)
        else:
            self._next_due_ns = now_monotonic_ns + self.period_ns


def latency_events_from_frame_timings(
    timings: tuple[FrameTimingV1, ...],
    *,
    starting_event_sequence: int = 1,
) -> tuple[LatencyEventV1, ...]:
    """Expand published frame timings into a chronological ``/1`` trace."""

    validate_frame_timing_sequence(timings)
    sequence = _exact_nonnegative_int(
        starting_event_sequence, "starting_event_sequence"
    )
    clocks = {timing.host_clock_id for timing in timings}
    if len(clocks) > 1:
        raise ValueError("one latency trace cannot order incomparable host clocks")
    points: list[
        tuple[int, int, int, FrameTimingV1, LatencyEventKind]
    ] = []
    stages = (
        (LatencyEventKind.CAMERA_FIRST_PACKET, "first_unique_packet_monotonic_ns"),
        (LatencyEventKind.CAMERA_FINAL_PACKET, "final_unique_packet_monotonic_ns"),
        (LatencyEventKind.FRAME_REASSEMBLED, "reassembly_complete_monotonic_ns"),
        (LatencyEventKind.DECODE_START, "decode_start_monotonic_ns"),
        (LatencyEventKind.DECODE_END, "decode_end_monotonic_ns"),
        (LatencyEventKind.FRAME_PUBLISHED, "publish_monotonic_ns"),
    )
    for timing in timings:
        for order, (kind, field) in enumerate(stages):
            points.append(
                (
                    getattr(timing, field),
                    timing.publication_sequence,
                    order,
                    timing,
                    kind,
                )
            )
    points.sort(key=lambda item: item[:3])
    events = tuple(
        LatencyEventV1(
            event_sequence=sequence + index,
            host_clock_id=timing.host_clock_id,
            monotonic_ns=timestamp,
            kind=kind,
            frame=timing.identity,
            control_tick_id=None,
            command_id=None,
            sensor_sample_id=None,
            sensor_source_time_ns=None,
            outcome=EventOutcome.OK,
            reason_code=None,
            queue_depth=0,
        )
        for index, (timestamp, _publication, _order, timing, kind) in enumerate(points)
    )
    validate_latency_event_sequence(events)
    return events


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    """Deterministic linear-interpolated latency percentiles."""

    count: int
    p50_ns: float
    p95_ns: float
    p99_ns: float
    maximum_ns: int


@dataclass(frozen=True, slots=True)
class RuntimeLatencySummary:
    distributions: tuple[tuple[str, LatencyDistribution], ...]
    deadline_misses: int
    skipped_ticks: int
    repeated_frame_ticks: int
    frame_drops: int
    maximum_queue_depth: int

    def distribution(self, name: str) -> Optional[LatencyDistribution]:
        return dict(self.distributions).get(name)


def summarize_latency_events(
    events: tuple[LatencyEventV1, ...],
) -> RuntimeLatencySummary:
    """Summarize validated events without mixing host clock domains."""

    validate_latency_event_sequence(events)
    samples: dict[str, list[int]] = {}

    def add(name: str, value: int) -> None:
        if value < 0:
            raise ValueError("latency duration cannot be negative")
        samples.setdefault(name, []).append(value)

    frame_points: dict[tuple[str, FrameIdentityV1], dict[LatencyEventKind, int]] = {}
    starts: dict[tuple[str, LatencyEventKind, tuple[Any, ...]], int] = {}
    stage_pairs = {
        LatencyEventKind.DECODE_END: (
            LatencyEventKind.DECODE_START,
            "decode",
        ),
        LatencyEventKind.DETECTION_END: (
            LatencyEventKind.DETECTION_START,
            "detection",
        ),
        LatencyEventKind.TRACKING_END: (
            LatencyEventKind.TRACKING_START,
            "tracking",
        ),
        LatencyEventKind.ESTIMATOR_UPDATE_END: (
            LatencyEventKind.ESTIMATOR_UPDATE_START,
            "estimator_update",
        ),
        LatencyEventKind.PREDICTION_END: (
            LatencyEventKind.PREDICTION_START,
            "prediction",
        ),
        LatencyEventKind.CONTROLLER_END: (
            LatencyEventKind.CONTROLLER_START,
            "controller",
        ),
        LatencyEventKind.CONTROL_TICK_END: (
            LatencyEventKind.CONTROL_TICK_START,
            "control_tick_work",
        ),
        LatencyEventKind.COMMAND_SEND_END: (
            LatencyEventKind.COMMAND_SEND_START,
            "command_send",
        ),
    }
    start_kinds = {start for start, _name in stage_pairs.values()}
    publish_times: dict[str, list[int]] = {}
    send_times: dict[str, list[int]] = {}
    previous_tick_frame: dict[str, Optional[FrameIdentityV1]] = {}
    repeated_frame_ticks = 0

    def correlation(event: LatencyEventV1) -> tuple[Any, ...]:
        if event.kind in {
            LatencyEventKind.DECODE_START,
            LatencyEventKind.DECODE_END,
            LatencyEventKind.DETECTION_START,
            LatencyEventKind.DETECTION_END,
            LatencyEventKind.TRACKING_START,
            LatencyEventKind.TRACKING_END,
            LatencyEventKind.ESTIMATOR_UPDATE_START,
            LatencyEventKind.ESTIMATOR_UPDATE_END,
        }:
            return (event.frame,)
        if event.kind in {
            LatencyEventKind.PREDICTION_START,
            LatencyEventKind.PREDICTION_END,
            LatencyEventKind.CONTROLLER_START,
            LatencyEventKind.CONTROLLER_END,
        }:
            return (event.frame, event.control_tick_id)
        if event.kind in {
            LatencyEventKind.CONTROL_TICK_START,
            LatencyEventKind.CONTROL_TICK_END,
        }:
            return (event.control_tick_id,)
        return (event.control_tick_id, event.command_id)

    for event in events:
        if event.frame is not None:
            frame_points.setdefault(
                (event.host_clock_id, event.frame), {}
            )[event.kind] = event.monotonic_ns
        if event.kind in start_kinds:
            starts[(event.host_clock_id, event.kind, correlation(event))] = (
                event.monotonic_ns
            )
        elif event.kind in stage_pairs:
            start_kind, name = stage_pairs[event.kind]
            started = starts[(event.host_clock_id, start_kind, correlation(event))]
            add(name, event.monotonic_ns - started)
        if event.kind is LatencyEventKind.FRAME_PUBLISHED:
            publish_times.setdefault(event.host_clock_id, []).append(event.monotonic_ns)
        elif event.kind is LatencyEventKind.COMMAND_SEND_START:
            send_times.setdefault(event.host_clock_id, []).append(event.monotonic_ns)
        elif event.kind is LatencyEventKind.CONTROL_TICK_START:
            prior = previous_tick_frame.get(event.host_clock_id)
            if event.frame is not None and event.frame == prior:
                repeated_frame_ticks += 1
            previous_tick_frame[event.host_clock_id] = event.frame

    for points in frame_points.values():
        first = points.get(LatencyEventKind.CAMERA_FIRST_PACKET)
        final = points.get(LatencyEventKind.CAMERA_FINAL_PACKET)
        reassembled = points.get(LatencyEventKind.FRAME_REASSEMBLED)
        published = points.get(LatencyEventKind.FRAME_PUBLISHED)
        if first is not None and final is not None:
            add("camera_packet_span", final - first)
        if final is not None and reassembled is not None:
            add("reassembly_after_final_packet", reassembled - final)
        if first is not None and published is not None:
            add("camera_first_packet_to_publish", published - first)

    for times in publish_times.values():
        for earlier, later in zip(times, times[1:]):
            add("frame_publish_interval", later - earlier)
    for times in send_times.values():
        for earlier, later in zip(times, times[1:]):
            add("command_send_interval", later - earlier)

    distributions = tuple(
        (name, _distribution(values))
        for name, values in sorted(samples.items())
    )
    return RuntimeLatencySummary(
        distributions=distributions,
        deadline_misses=sum(
            event.kind is LatencyEventKind.DEADLINE_MISSED for event in events
        )
        + sum(
            event.kind is LatencyEventKind.CONTROL_TICK_END
            and event.outcome is EventOutcome.ERROR
            and event.reason_code == "tick_work_overrun"
            for event in events
        ),
        skipped_ticks=sum(
            event.kind is LatencyEventKind.CONTROL_TICK_SKIPPED for event in events
        ),
        repeated_frame_ticks=repeated_frame_ticks,
        frame_drops=sum(event.kind is LatencyEventKind.FRAME_DROPPED for event in events),
        maximum_queue_depth=max(
            (event.queue_depth for event in events if event.queue_depth is not None),
            default=0,
        ),
    )


def _distribution(values: list[int]) -> LatencyDistribution:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency distribution requires at least one sample")

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return LatencyDistribution(
        count=len(ordered),
        p50_ns=percentile(0.50),
        p95_ns=percentile(0.95),
        p99_ns=percentile(0.99),
        maximum_ns=ordered[-1],
    )


__all__ = [
    "ControlTickLeaseV1",
    "FixedRateControlSchedulerV1",
    "LatencyDistribution",
    "LatencyTraceRecorderV1",
    "LatestFrameCursorV1",
    "LatestFrameSelectionV1",
    "MINIMUM_CONTROL_PERIOD_NS",
    "RuntimeLatencySummary",
    "VQ2_HOST_CLOCK_ID",
    "latency_events_from_frame_timings",
    "summarize_latency_events",
]
