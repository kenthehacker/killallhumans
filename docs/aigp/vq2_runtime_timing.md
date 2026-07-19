# VQ2 runtime timing foundation

**Target:** FlightSim build 3385, Training mode

**Status:** offline Wave 1 foundation; no simulator measurement or powered
integration is claimed

## Implemented boundary

`competition.vq2_vision.VQ2VisionThread` now publishes an exact frozen
`FrameTimingV1` with every production snapshot. The timing covers:

```text
first unique UDP packet
  -> final unique UDP packet
  -> reassembly complete
  -> decode start/end
  -> latest-value publication
```

The snapshot retains its existing `received_monotonic_s` freshness field for
the proved runner watchdog. On the target Windows Python build that clock is
the coarse monotonic `GetTickCount64` clock. The `/1` frame timing instead uses
the monotonic high-resolution `QueryPerformanceCounter` clock through
`time.perf_counter_ns()` and labels it `host-perf-counter`. Those are separate
domains: code must never subtract the legacy freshness value from a `/1`
timing point or claim a calibrated mapping between them.
`VQ2_HOST_CLOCK_ID` supplies that same default to vision, latency recording,
and fixed-rate scheduling so correlated `/1` events do not silently mix clocks.

The camera source timestamp remains only the frozen opaque uint64 source token.
It is never converted to host time. Publication sequence is process-lifetime
monotonic and does not reset when camera generation advances. A timing-ledger
overflow latches publication stale until the reviewed reset boundary instead
of evicting the first-packet timestamp and manufacturing a shorter latency.

`competition.vq2_runtime` provides pure offline primitives:

- `LatestFrameCursorV1` consumes one exact latest publication once, reports
  publication gaps as overwritten intermediate frames, accepts uint32 frame-ID
  wrap and forward generation restart, and rejects timing/clock/stream
  regression;
- `FixedRateControlSchedulerV1` exposes at most one active tick, rejects periods
  faster than 20 ms, retains the nominal grid while work fits, and rebases one
  full period after a missed slot so catch-up bursts are impossible;
- `LatencyTraceRecorderV1` builds append-only frozen `LatencyEventV1` traces;
- `latency_events_from_frame_timings` expands overlapping camera timings in
  chronological order without comparing different host clocks;
- `summarize_latency_events` reports deterministic p50, p95, p99, and maximum
  durations, publish and command intervals, deadline misses, skipped and
  repeated-frame ticks, frame drops, and maximum observed queue depth.

An unstarted expired tick is recorded as `DEADLINE_MISSED` followed by
`CONTROL_TICK_SKIPPED`. A tick that already started cannot be relabeled as
skipped under the frozen contract; if its work crosses the deadline, its end is
`ERROR/tick_work_overrun`, and the summary counts that as a deadline miss.

## Runner compatibility

The powered runner still owns all existing safety checks and uses its existing
send path and stage pacing. This branch does not route a command through the
new pure scheduler. Its only runner behavior change is to recognize a distinct
camera frame by `(generation, frame_id)` instead of the opaque source timestamp.
Repeated reads cannot re-run perception merely because that source token
changes, while a receiver generation restart may safely reuse a frame ID.

No command envelope, yaw-zero rule, exact-zero crossing confirmation, watchdog,
reset proof, arm/disarm freshness rule, stage deadline, or cleanup path changes.

## Remaining M1 work

The following still need an integration-owned implementation and measured
evidence before the timing dossier is complete:

- connect detection, tracking, estimator, prediction, controller, approved
  command send, actuator, and gyro events to one process-scoped latency trace;
- integrate the fixed-rate scheduler behind the reviewed supervisor/transport
  seam while preserving single-use approval and actual send pacing;
- record bounded queue depth and drop reasons at each real handoff;
- measure p50/p95/p99/max, stream rates, duplicates, frame drops, repeated-frame
  ticks, host load, simulator/wall ratio, graphics/focus state, and response
  timing under separately authorized simulator work;
- calibrate command-effect and measurement-clock models rather than inferring
  either from raw source tokens.

Until those steps exist, this module is tested runtime infrastructure, not
FlightSim timing evidence and not authority for any powered stage.
