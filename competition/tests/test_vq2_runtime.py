"""Offline tests for VQ2 latest-value and fixed-rate runtime timing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from competition.vq2_contracts import (
    EventOutcome,
    FrameIdentityV1,
    FrameTimingV1,
    LatencyEventKind,
    validate_latency_event_sequence,
)
from competition.vq2_runtime import (
    FixedRateControlSchedulerV1,
    LatencyTraceRecorderV1,
    LatestFrameCursorV1,
    MINIMUM_CONTROL_PERIOD_NS,
    VQ2_HOST_CLOCK_ID,
    latency_events_from_frame_timings,
    summarize_latency_events,
)


def _timing(
    *,
    publication_sequence: int = 1,
    generation: int = 0,
    frame_id: int = 10,
    source_time_ns: int = 1_000,
    base_ns: int = 100,
    stream_id: str = "camera0",
    host_clock_id: str = "host-monotonic-1",
    packet_span_ns: int = 2,
) -> FrameTimingV1:
    return FrameTimingV1(
        identity=FrameIdentityV1(stream_id, generation, frame_id),
        camera_source_time_ns=source_time_ns,
        host_clock_id=host_clock_id,
        publication_sequence=publication_sequence,
        first_unique_packet_monotonic_ns=base_ns,
        final_unique_packet_monotonic_ns=base_ns + packet_span_ns,
        reassembly_complete_monotonic_ns=base_ns + packet_span_ns + 1,
        decode_start_monotonic_ns=base_ns + packet_span_ns + 2,
        decode_end_monotonic_ns=base_ns + packet_span_ns + 5,
        publish_monotonic_ns=base_ns + packet_span_ns + 6,
    )


def _snapshot(timing: FrameTimingV1):
    return SimpleNamespace(
        generation=timing.identity.generation,
        frame_id=timing.identity.frame_id,
        sim_time_ns=timing.camera_source_time_ns,
        timing=timing,
    )


def test_latest_frame_cursor_consumes_once_and_reports_overwritten_gap():
    cursor = LatestFrameCursorV1(
        expected_host_clock_id="host-monotonic-1",
        expected_stream_id="camera0",
    )
    first = _snapshot(_timing())
    selected = cursor.select(first)
    assert selected is not None
    assert selected.overwritten_publications == 0
    assert cursor.select(first) is None
    assert cursor.repeated_reads == 1

    latest = _snapshot(
        _timing(
            publication_sequence=4,
            frame_id=13,
            source_time_ns=1_300,
            base_ns=200,
        )
    )
    selected = cursor.select(latest)
    assert selected is not None
    assert selected.snapshot is latest
    assert selected.overwritten_publications == 2
    assert cursor.overwritten_publications == 2


def test_latest_frame_cursor_rejects_same_identity_relabel_and_source_regression():
    cursor = LatestFrameCursorV1()
    first = _timing()
    cursor.select(_snapshot(first))
    with pytest.raises(ValueError, match="cannot be relabeled"):
        cursor.select(
            _snapshot(
                _timing(
                    source_time_ns=1_001,
                    base_ns=101,
                )
            )
        )

    bad_snapshot = _snapshot(first)
    bad_snapshot.generation = True
    with pytest.raises(TypeError, match="exact integers"):
        LatestFrameCursorV1().select(bad_snapshot)

    cursor = LatestFrameCursorV1()
    cursor.select(_snapshot(first))
    with pytest.raises(ValueError, match="source time"):
        cursor.select(
            _snapshot(
                _timing(
                    publication_sequence=2,
                    frame_id=11,
                    source_time_ns=999,
                    base_ns=200,
                )
            )
        )

    cursor = LatestFrameCursorV1()
    cursor.select(_snapshot(first))
    with pytest.raises(ValueError, match="publication_sequence"):
        cursor.select(
            _snapshot(
                _timing(
                    publication_sequence=1,
                    frame_id=11,
                    source_time_ns=1_100,
                    base_ns=200,
                )
            )
        )


def test_latest_frame_cursor_accepts_uint32_wrap_and_generation_restart():
    cursor = LatestFrameCursorV1()
    cursor.select(
        _snapshot(
            _timing(
                frame_id=(1 << 32) - 1,
                source_time_ns=10_000,
            )
        )
    )
    wrapped = cursor.select(
        _snapshot(
            _timing(
                publication_sequence=2,
                frame_id=0,
                source_time_ns=11_000,
                base_ns=200,
            )
        )
    )
    assert wrapped is not None
    restarted = cursor.select(
        _snapshot(
            _timing(
                publication_sequence=3,
                generation=1,
                frame_id=0,
                source_time_ns=5,
                base_ns=300,
            )
        )
    )
    assert restarted is not None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("host_clock_id", "host-monotonic-2", "host clock"),
        ("stream_id", "camera1", "camera stream"),
    ],
)
def test_latest_frame_cursor_rejects_clock_or_stream_switch(field, value, message):
    cursor = LatestFrameCursorV1()
    cursor.select(_snapshot(_timing()))
    kwargs = {
        "publication_sequence": 2,
        "frame_id": 11,
        "source_time_ns": 1_100,
        "base_ns": 200,
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        cursor.select(_snapshot(_timing(**kwargs)))


def test_scheduler_runs_one_tick_at_a_time_and_keeps_nominal_grid_when_work_fits():
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=1_000_000_000,
        host_clock_id="host-monotonic-1",
    )
    frame = FrameIdentityV1("camera0", 0, 10)
    first = scheduler.begin_due(1_000_000_000, frame=frame)
    assert first is not None
    with pytest.raises(RuntimeError, match="already active"):
        scheduler.begin_due(1_000_000_001, frame=frame)
    scheduler.finish(first, 1_001_000_000)
    assert scheduler.next_due_monotonic_ns == 1_020_000_000
    assert scheduler.begin_due(1_019_999_999, frame=frame) is None
    second = scheduler.begin_due(1_020_000_000, frame=frame)
    assert second is not None
    scheduler.finish(second, 1_021_000_000)

    events = scheduler.trace.snapshot()
    due_times = [
        event.monotonic_ns
        for event in events
        if event.kind is LatencyEventKind.CONTROL_TICK_DUE
    ]
    assert due_times == [1_000_000_000, 1_020_000_000]
    assert due_times[1] - due_times[0] == MINIMUM_CONTROL_PERIOD_NS


def test_scheduler_late_poll_is_deadline_missed_skipped_and_rebased():
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=100,
        host_clock_id="host-monotonic-1",
    )
    late = 100 + MINIMUM_CONTROL_PERIOD_NS + 1
    assert scheduler.begin_due(late) is None
    assert scheduler.active is None
    assert scheduler.next_due_monotonic_ns == late + MINIMUM_CONTROL_PERIOD_NS
    events = scheduler.trace.snapshot()
    assert [event.kind for event in events] == [
        LatencyEventKind.CONTROL_TICK_DUE,
        LatencyEventKind.DEADLINE_MISSED,
        LatencyEventKind.CONTROL_TICK_SKIPPED,
    ]
    assert events[-1].outcome is EventOutcome.SKIPPED


def test_scheduler_in_window_late_poll_cannot_create_a_short_following_interval():
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=0,
        host_clock_id="host-monotonic-1",
    )
    observed_due = MINIMUM_CONTROL_PERIOD_NS // 2
    lease = scheduler.begin_due(observed_due)
    assert lease is not None
    scheduler.finish(lease, observed_due + 1_000_000)
    assert scheduler.next_due_monotonic_ns == observed_due + MINIMUM_CONTROL_PERIOD_NS
    next_lease = scheduler.begin_due(scheduler.next_due_monotonic_ns)
    assert next_lease is not None
    scheduler.finish(next_lease, next_lease.start_monotonic_ns)
    due_times = [
        event.monotonic_ns
        for event in scheduler.trace.snapshot()
        if event.kind is LatencyEventKind.CONTROL_TICK_DUE
    ]
    assert due_times[1] - due_times[0] == MINIMUM_CONTROL_PERIOD_NS


def test_scheduler_exact_deadline_is_eligible_but_late_work_ends_error():
    start = 1_000
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=start,
        host_clock_id="host-monotonic-1",
    )
    deadline = start + MINIMUM_CONTROL_PERIOD_NS
    lease = scheduler.begin_due(deadline)
    assert lease is not None
    end = scheduler.finish(lease, deadline + 1)
    assert end.outcome is EventOutcome.ERROR
    assert end.reason_code == "tick_work_overrun"
    assert scheduler.next_due_monotonic_ns == deadline + 1 + MINIMUM_CONTROL_PERIOD_NS
    summary = summarize_latency_events(scheduler.trace.snapshot())
    assert summary.deadline_misses == 1


def test_scheduler_reserves_work_overrun_reason_for_elapsed_deadline():
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=0,
        host_clock_id="host-monotonic-1",
    )
    lease = scheduler.begin_due(0)
    assert lease is not None
    with pytest.raises(ValueError, match="reserved"):
        scheduler.finish(
            lease,
            1,
            outcome=EventOutcome.ERROR,
            reason_code="tick_work_overrun",
        )
    assert scheduler.active is lease
    scheduler.finish(
        lease,
        1,
        outcome=EventOutcome.ERROR,
        reason_code="controller_failed",
    )


def test_scheduler_explicit_skip_has_no_deadline_miss_and_never_catches_up():
    start = 500
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=start,
        host_clock_id="host-monotonic-1",
    )
    assert scheduler.skip_due(start, reason_code="no_usable_state")
    assert not scheduler.skip_due(start + 1, reason_code="no_usable_state")
    assert scheduler.next_due_monotonic_ns == start + MINIMUM_CONTROL_PERIOD_NS
    events = scheduler.trace.snapshot()
    assert [event.kind for event in events] == [
        LatencyEventKind.CONTROL_TICK_DUE,
        LatencyEventKind.CONTROL_TICK_SKIPPED,
    ]


def test_scheduler_invalid_skip_metadata_does_not_append_a_due_event():
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=0,
        host_clock_id="host-monotonic-1",
    )
    with pytest.raises(ValueError, match="reason_code"):
        scheduler.skip_due(0, reason_code="not a token with spaces")
    assert scheduler.trace.snapshot() == ()
    assert scheduler.next_control_tick_id == 0


def test_scheduler_command_trace_summarizes_intervals_and_repeated_frame_ticks():
    scheduler = FixedRateControlSchedulerV1(
        start_monotonic_ns=0,
        host_clock_id="host-monotonic-1",
    )
    frame = FrameIdentityV1("camera0", 0, 10)
    first = scheduler.begin_due(0, frame=frame, queue_depth=2)
    assert first is not None
    scheduler.trace.record(
        LatencyEventKind.COMMAND_SEND_START,
        1_000_000,
        frame=frame,
        control_tick_id=first.control_tick_id,
        command_id=100,
        queue_depth=1,
    )
    scheduler.trace.record(
        LatencyEventKind.COMMAND_SEND_END,
        2_000_000,
        frame=frame,
        control_tick_id=first.control_tick_id,
        command_id=100,
        queue_depth=0,
    )
    scheduler.finish(first, 3_000_000)

    second = scheduler.begin_due(MINIMUM_CONTROL_PERIOD_NS, frame=frame)
    assert second is not None
    scheduler.trace.record(
        LatencyEventKind.COMMAND_SEND_START,
        MINIMUM_CONTROL_PERIOD_NS + 1_000_000,
        frame=frame,
        control_tick_id=second.control_tick_id,
        command_id=101,
        queue_depth=0,
    )
    scheduler.trace.record(
        LatencyEventKind.COMMAND_SEND_END,
        MINIMUM_CONTROL_PERIOD_NS + 2_000_000,
        frame=frame,
        control_tick_id=second.control_tick_id,
        command_id=101,
        queue_depth=0,
    )
    scheduler.finish(second, MINIMUM_CONTROL_PERIOD_NS + 3_000_000)

    summary = summarize_latency_events(scheduler.trace.snapshot())
    assert summary.repeated_frame_ticks == 1
    assert summary.maximum_queue_depth == 2
    command_interval = summary.distribution("command_send_interval")
    assert command_interval is not None
    assert command_interval.maximum_ns == MINIMUM_CONTROL_PERIOD_NS
    assert summary.distribution("command_send").maximum_ns == 1_000_000


def test_scheduler_rejects_rate_over_50_hz_and_trace_clock_mismatch():
    with pytest.raises(ValueError, match="50 Hz"):
        FixedRateControlSchedulerV1(
            start_monotonic_ns=0,
            period_ns=MINIMUM_CONTROL_PERIOD_NS - 1,
        )
    trace = LatencyTraceRecorderV1("clock-a")
    with pytest.raises(ValueError, match="host clocks"):
        FixedRateControlSchedulerV1(
            start_monotonic_ns=0,
            host_clock_id="clock-b",
            trace=trace,
        )


def test_runtime_timing_components_share_one_default_qpc_clock():
    trace = LatencyTraceRecorderV1()
    scheduler = FixedRateControlSchedulerV1(start_monotonic_ns=0)
    assert trace.host_clock_id == VQ2_HOST_CLOCK_ID
    assert scheduler.trace.host_clock_id == VQ2_HOST_CLOCK_ID
    trace.record(
        LatencyEventKind.FRAME_DROPPED,
        1,
        frame=FrameIdentityV1("camera0", 0, 9),
        outcome=EventOutcome.DROPPED,
        reason_code="latest_value_overwritten",
        queue_depth=3,
    )
    summary = summarize_latency_events(trace.snapshot())
    assert summary.frame_drops == 1
    assert summary.maximum_queue_depth == 3
    with pytest.raises(ValueError, match="unsupported characters"):
        LatencyTraceRecorderV1("clock id with spaces")


def test_frame_timing_expansion_is_chronological_for_overlapping_frames():
    first = _timing(packet_span_ns=20)
    second = _timing(
        publication_sequence=2,
        frame_id=11,
        source_time_ns=1_100,
        base_ns=120,
        packet_span_ns=2,
    )
    events = latency_events_from_frame_timings((first, second))
    assert len(events) == 12
    assert [event.monotonic_ns for event in events] == sorted(
        event.monotonic_ns for event in events
    )
    validate_latency_event_sequence(events)

    summary = summarize_latency_events(events)
    packet_span = summary.distribution("camera_packet_span")
    assert packet_span is not None
    assert packet_span.count == 2
    assert packet_span.p50_ns == pytest.approx(11.0)
    assert packet_span.maximum_ns == 20
    assert summary.distribution("decode").maximum_ns == 3


def test_frame_timing_expansion_rejects_mixed_host_clock_domains():
    with pytest.raises(ValueError, match="incomparable host clocks"):
        latency_events_from_frame_timings(
            (
                _timing(),
                _timing(
                    publication_sequence=2,
                    frame_id=11,
                    source_time_ns=1_100,
                    base_ns=200,
                    host_clock_id="other-clock",
                ),
            )
        )


def test_latency_trace_rejects_time_regression_without_mutating_trace():
    trace = LatencyTraceRecorderV1("host-monotonic-1")
    frame = FrameIdentityV1("camera0", 0, 10)
    trace.record(
        LatencyEventKind.FRAME_PUBLISHED,
        100,
        frame=frame,
        queue_depth=0,
    )
    with pytest.raises(ValueError, match="regressed"):
        trace.record(
            LatencyEventKind.FRAME_PUBLISHED,
            99,
            frame=FrameIdentityV1("camera0", 0, 11),
            queue_depth=0,
        )
    events = trace.snapshot()
    assert len(events) == 1
    assert events[0].event_sequence == 1
