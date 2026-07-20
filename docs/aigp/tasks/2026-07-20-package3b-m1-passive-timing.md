# Package 3B M1 passive receiver-timing dossier

- Task ID: `vq2-package3b-m1-passive-timing`
- Parent: `2026-07-20-vq2-development-continuation-handoff`
- State: `active`
- Objective: preserve and summarize real build-3385 camera receiver,
  reassembly, decode, publication, passive detection, and tracker timing from
  bounded Training-mode preflights, without sending an arm request, reset, or
  flight target.
- Starting main commit: `b7935b4e2d9b685a70ac59b944631a0d48a1d919`
- Branch: `package3b-m1-passive-timing`
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-package3b-m1-passive-timing`
- Integration owner, task owner, and simulator lease owner: `/root`
- Heartbeat date: `2026-07-20`
- Simulator access: `passive`
- User authorization: on `2026-07-20` the user approved private full-frame
  storage/use, replay work, calibration collection, and passive build-3385
  simulator access. This task does not treat that blanket approval as fresh
  authorization for any powered stage.

## Frozen scope

Owned tracked interfaces:

- passive frame-timing observation in `scripts/aigp_vq2_run.py`;
- a pure fail-closed timing summarizer and narrow CLI, if needed;
- directly affected tests;
- `docs/aigp/vq2_runtime_timing.md`; and
- this task record.

Serialized promotion policy and trusted-manifest files remain integration-owner
files. They may change only after the behavior candidate is accepted. Private
captures, replay bundles, raw timing records, process samples, and generated
reports remain outside Git under
`C:\Users\John\aigp-evidence\2026-07-20-package3b-m1-passive-timing`.

The implementation may record only already-published `FrameTimingV1` points
and same-clock passive runner observations. It must preserve the frozen
`aigp-vq2-frame-timing/1` meaning, keep the simulator camera source timestamp
opaque, and never subtract `time.monotonic()` freshness values from
`host-perf-counter` timing points. Aggregate output must distinguish measured
receiver evidence from generated scheduler evidence.

Explicitly excluded interfaces and claims:

- no reset, countdown, arm/disarm, flight target, body-rate/thrust send,
  command-pacing change, transport selection, supervisor change, Gate 0 or
  Gate 1 motion, or powered cleanup path;
- no command-send-to-actuator or gyro causal-delay measurement;
- no camera/IMU clock calibration, camera intrinsics/extrinsics calibration,
  measurement-time model, or calibrated capture-time claim;
- no replay labels, corpus split, replay acceptance, M2 acceptance, runtime
  promotion, or authority to select an offline proposal for transport; and
- no changes to replay record schemas, controller envelopes, watchdogs,
  exact-zero behavior, or gate-passage authority.

## Passive simulator lease and stop rules

Immediately before every probe, `/root` must confirm:

1. the running executable resolves to the documented build-3385 installation;
2. no other agent/process owns the simulator lease;
3. local UDP ports `14550` and `5600` are unowned before the receiver binds;
4. the task worktree is clean at the committed candidate; and
5. the only requested stage is `preflight` with explicit private output paths.

The task permits at most one pre-instrumentation baseline and five
post-instrumentation passive preflights. Each individual runner call retains
the existing ten-second preflight ceiling and must terminate its vision thread,
disconnect MAVLink, seal or permanently invalidate the requested replay
bundle, and release both UDP ports. A bind conflict, stale stream, incomplete
capture, receiver termination failure, transport disconnect failure, changed
simulator process/build, unexpected command record, or uncleared port is a
failed probe and stops simulator work pending review.

The exact live boundary is:

```powershell
python -m scripts.aigp_vq2_run --stage preflight `
  --record <private-jsonl-path> `
  --replay-bundle <private-bundle-path> `
  --recording-approved
```

No command whose `--stage` is `sign-id`, `hover`, `gate0`, or
`gate0-observe` is within this contract.

## Acceptance and evidence

Direct acceptance requires:

- exact frame identity and publication sequence in every timing observation;
- validation of all frozen timing points before aggregation;
- p50/p95/p99/max and sample counts for packet span, final-packet-to-
  reassembly, decode, decode-to-publication, first-packet-to-publication,
  publication-to-passive-consumption, detection, tracking, and total passive
  frame work where those stages are observed;
- publication and consumption intervals, publication-sequence gaps, malformed
  or duplicate observation rejection, and an explicit count of passive frame
  work exceeding the frozen 20 ms control period;
- receiver counters, host/process-load context, simulator/wall ratio where it
  can be measured without calibrating the opaque camera timestamp, graphics /
  focus state, capture completeness, exact artifact hashes, and port-release
  proof for each accepted session; and
- explicit `unmeasured` results for control scheduler deadlines, command send,
  actuator response, and gyro response.

Required checks after each edit are the directly affected timing and runner
tests. Candidate acceptance requires `scripts\dev.cmd test-vq2`; promotion
requires `test-fast`, `test-unit`, and `test-full-non-live`, plus an exact
hash-pinned VQ2 run from a fresh candidate. No test command may contact the
simulator.

The implementation stop condition is a reviewed, deterministic passive timing
summarizer plus accepted private build-3385 preflight evidence, or an honestly
recorded failure at one of the frozen stop rules. Completion of this task can
advance only the passive receiver portion of M1. Powered causal-delay work
remains a separately named and freshly authorized task.

## Evidence record

### Independent-review correction

The contract-freeze commit is
`944d245c570ee84119d6d66df34fd832035a3304`. Independent timing,
calibration, and replay reviews found that the original freeze was too narrow
to support honest M1 or calibration-timing evidence. This correction is an
immutable superseding contract for all later implementation and simulator
work; the earlier baseline is retained as failure provenance rather than
silently upgraded.

This task is now the shared **Package 2A / Package 3B passive capture-ingress
tranche**. Its exit does not complete Package 2 or M1. It establishes exact
receiver-boundary evidence needed by both. A later M1 task must still add a
no-send shadow 50 Hz scheduler and production handoff/deadline/skip evidence.
Actual send-to-actuator/gyro delay remains powered and separately authorized.

Additional owned tracked interfaces are:

- `competition/aigp_mavlink.py` for additive receiver-boundary arrival
  envelopes, bounded ingress diagnostics, and outbound-message audit;
- `aigp_loop/replay.py` for backward-compatible, versioned timing events that
  preserve the existing `/1` core replay records;
- a pure passive-timing validator/summarizer;
- a Windows passive-probe/lease wrapper and its tests; and
- the directly affected MAVLink, replay, runner, and timing tests.

The frozen event identities are
`aigp-vq2-mavlink-ingress/1`, `aigp-vq2-received-imu/1`, and
`aigp-vq2-camera-frame-timing-observation/1`. All receiver-boundary arrivals
and camera timing points use integer nanoseconds on exact clock ID
`host-perf-counter`. A MAVLink connection/reset generation and strict receive
sequence bind each sample. The legacy coarse freshness clock and existing
core replay records remain unchanged; new exact evidence is additive and old
`aigp-vq2-replay-record/1` bundles remain readable. Queue capacity, high-water
mark, and overflow count are evidence. Any overflow invalidates a session for
timing acceptance.

The live lease is the Windows named mutex
`Global\AIGP-FlightSim-LiveLease-v1`. The probe wrapper must acquire it
non-blockingly before process or port checks, hold it across the entire child
runner lifetime, transport/vision disconnect, bundle finalization, and
post-probe port checks, and then release it. It writes a private
`aigp-vq2-live-lease-evidence/1` envelope containing a random owner token,
wrapper PID, acquisition and one-second heartbeat timestamps, phase,
child PID, and clean release timestamp. A busy, abandoned, inaccessible, or
unverifiable mutex aborts before simulator contact; an abandoned mutex is
never silently reclaimed for the same probe.

The build boundary is both of these exact files:

- launcher `C:\Users\John\AIGP\AIGP_3385\FlightSim.exe`, SHA-256
  `0d3217fa72e9fee847b2c154432476a687f21b79f0ab6b910728a6254b4dce32`;
- payload
  `C:\Users\John\AIGP\AIGP_3385\FlightSim\Binaries\Win64\DCGame-Win64-Shipping.exe`,
  SHA-256
  `9064dd1547a30afea1e3fb87652cc8194c3f5af556be40629dc491bb4f681362`.

The environment context supplied by the user attests Training mode. The probe
must bind `simulator_mode="Training"` and
`simulator_mode_basis="operator-attested-2026-07-20"` into its private
provenance; it must not claim that executable paths, window title, or race
packets independently detect the GUI selection. A separately typed
`aigp-vq2-passive-probe-context/1` envelope records fresh process IDs, parent
relationship, paths and hashes, command lines, process start/session, window
visibility/minimized/foreground state, commit and dirty-diff hash, exact child
command, ports, QPC frequency, wall timestamps, artifact hashes, and fresh
per-process CPU/working-set samples at one-second cadence. Camera source tokens
remain opaque; simulator/wall ratio is `unmeasured` until a reviewed clock
model exists.

Only GCS heartbeat and TIMESYNC sends are permitted during a passive probe.
The adapter's outbound audit must report zero SIM_RESET, arm, disarm,
attitude-target, position-target, and other command sends. Any nonzero
disallowed count fails the session even if the runner otherwise returns
success.

The original baseline is complete contextual evidence from exact commit
`944d245c570ee84119d6d66df34fd832035a3304`:

- private bundle
  `C:\Users\John\aigp-evidence\2026-07-20-package3b-m1-passive-timing\baseline.vq2replay`;
- dataset SHA-256 identity
  `22917a45cd7e6e940d34eafd6f6389732d78a2bd686b5f1f3109d460c94d25f6`;
- 184 records, 31 decoded and 31 processed frames, five unique blobs, zero
  command records, zero capture drops/errors, and queue high-water mark three;
- 31 frames at 31 fps and 51,032 duplicates in the returned stage details;
  the final manifest outcome records 51,037 lifetime duplicates;
- successful full bundle verification and released UDP `14550`/`5600`.

It consumed the sole pre-instrumentation allowance. It has no durable lease
proof or `FrameTimingV1` records. Its 117 IMU records contain only 59 distinct
poll timestamps, including batches of up to four samples assigned one drain
time. Its CPU/focus host-context fields are invalid. It is therefore not M1,
clock-offset, process-load, or production-latency evidence and must never be
used to calibrate those quantities.

No further simulator probe occurs until the corrected implementation and all
affected offline tests plus `test-vq2` pass. The corrected collection is
exactly three accepted capture-loaded passive sessions. Each session uses a
five-second healthy dwell after readiness, remains inside the existing
ten-second preflight ceiling, and must contain at least 140 exact camera timing
observations and 600 exact HIGHRES_IMU arrival observations. Sessions with a
shortfall, queue overflow, capture incompleteness, mismatched frame count,
invalid timing, nonzero disallowed outbound audit, lease failure, changed
build/process, or uncleared receive port are rejected and stop collection.

These three sessions characterize the approved replay-capture-loaded path.
They do not claim no-capture production latency. A successor tranche must
separately compare timing-only/no-frame-write load before runtime promotion.
All acceptance output must explicitly report control scheduler deadlines,
command send, actuator effect, gyro effect, calibrated camera/IMU offset,
simulator/wall ratio, and Training-mode machine detection as `unmeasured`.
