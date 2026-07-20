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

Pending.
