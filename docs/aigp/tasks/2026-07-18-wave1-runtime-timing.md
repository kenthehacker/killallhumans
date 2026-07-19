# Wave 1 runtime-timing task manifest

- Task ID: `vq2-wave1-runtime-timing`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `post_merge_verified`
- Objective: implement the fully offline M1 foundation for end-to-end host
  timing, latest-value camera consumption, and 50 Hz control-tick scheduling
  using the frozen VQ2 `/1` timing and latency contracts.
- Non-goals: no FlightSim process launch, connection, or passive probe; no
  external network access; no reset, arm/disarm, flight target, powered stage,
  Gate 1 maneuver, plant identification, or safety-supervisor replacement; no
  claim of measured simulator timing.
- Starting commit: `3de33c3a568bc86638d9d7ac4dac6124f1e15397`
- Branch: `wave1-runtime-timing`
- Worktree: `C:\Users\John\aigp-worktrees\wt-runtime-timing`
- Integration owner: `/root`
- Task owner: `/root/wave1_runtime_timing`
- Heartbeat date: `2026-07-18`
- Simulator access: `none`
- Owned interfaces: VQ2 vision publication timing, offline latest-value and
  fixed-rate runtime primitives, their direct tests, and runtime-timing docs.
- Serialized files excluded from this branch:
  `config/t1_pytest_policy.json`, `config/promotion_trusted_files.json`, and
  `competition/vq2_contracts.py`.
- Frozen dependency: `docs/aigp/vq2_contracts.md` and the exact `/1` classes in
  `competition/vq2_contracts.py`; this task must not change their wire meaning.
- Required direct tests: runtime timing, VQ2 vision, and VQ2 runner timing tests.
- Required candidate gate: `scripts\dev.cmd test-vq2`; an exact-count-only
  policy failure is reported to the integration owner rather than changing the
  serialized policy on this branch.
- Acceptance: each published camera frame carries complete same-clock
  first/final-packet, reassembly, decode, and publication timing; publication
  identity/sequence remains safe across uint32 frame-ID wrap and receiver
  generations; latest-value consumption never reprocesses a frame or queues
  backlog; the fixed-rate scheduler never catches up, never produces a tick
  faster than 50 Hz, and records stale/deadline skips deterministically; all
  new timing traces validate against the frozen contracts.

## Adversarial review matrix

- repeated latest-frame reads and overwritten intermediate frames;
- stale or regressing publication sequences and host-clock mismatches;
- uint32 frame-ID wrap versus receiver restart generation;
- simulator camera source time treated only as an opaque ordering token;
- decode/reset races and old-generation publication attempts;
- long loop stalls, exact deadline boundaries, skipped ticks, and prevention
  of catch-up bursts;
- injected monotonic clocks and deterministic percentile/trace behavior;
- unchanged Gate 0 watchdog, command bounds, exact-zero confirmation, and
  cleanup behavior.

## Evidence record

- Implementation commit: `11c35a5431587b892fd3950c7a589ce3ff312652`.
- Implemented files:
  `competition/vq2_runtime.py`, `competition/vq2_vision.py`, and the narrow
  frame-identity correction in `scripts/aigp_vq2_run.py`; direct tests live in
  `competition/tests/test_vq2_runtime.py`,
  `competition/tests/test_vq2_vision.py`, and
  `tests/test_aigp_vq2_runner.py`.
- Direct runtime/vision/runner tests: `92 passed` in 1.41 seconds.
- Broader competition and legacy vision tests: `235 passed` in 3.60 seconds.
- `test-vq2`: `381 passed` in 5.61 seconds. The branch left serialized policy
  and manifest ownership to integration; the combined candidate now records
  exactly 418 tests and the reviewed 119-entry manifest in the integration task.
- Pre-final `test-fast`: `1,472 passed, 20 skipped, 42 deselected` in 71.99
  seconds. The later generation-lifetime identity regression is included in
  the final direct, broader, and VQ2 evidence above.
- Mandatory trace evidence boundary:
  `LatencyTraceRecorderV1.snapshot(validate=True)`; an invalid caller sequence
  can poison its append-only diagnostic buffer but cannot validate or promote.
- `git diff --check`: clean.
- Adversarial results: repeated reads, stale publication, source-token relabel,
  uint32 wrap, generation restart, bounded-cache identity replay, timing-ledger
  churn, host-clock mixing, exact deadline boundaries, long stalls, active
  overruns, invalid skip metadata, repeated-frame ticks, queue depth, and
  deterministic percentile summaries all fail closed or retain the documented
  latest-value behavior.
- Simulator access: none; no official-simulator or flight-domain evidence is
  claimed by this task.
