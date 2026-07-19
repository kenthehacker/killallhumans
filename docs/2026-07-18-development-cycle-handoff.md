# Development-Cycle Acceleration Handoff

**Date:** 2026-07-18<br>
**Repository:** `C:\Users\John\killallhumans`<br>
**Baseline commit:** `c7c37c047039bcac055d77c57a234effe36f73e1` (`aigp: add bounded gate1 observation stage`)<br>
**Branch state before this document:** `main`, clean, three commits ahead of `origin/main`

## New-session starting instruction

Implement the workflow improvements in this handoff, beginning with **P0 only**.
Do not alter VQ2 flight behavior or weaken any safety invariant merely to make
the development loop faster. Do not run a powered FlightSim stage as part of
this tooling work unless the user explicitly authorizes that live flight.

Read these first:

1. This handoff.
2. `docs/aigp/2026-07-18-vq2-handoff.md` for the current build-3385 flight
   state and safety contract.
3. `docs/2026-06-09-logic-audit.md` for historical benchmark-honesty issues.

Recommended first-session scope:

1. Fix/bound the two 480-second VQ1 tests.
2. Add pytest markers, hard test timeouts, and fast/default test selection.
3. Add a canonical Windows development command surface.
4. Update agent instructions so VQ2, not the legacy synthetic stack, is the
   default inner loop.
5. Measure and report the resulting wall times before starting cache work.

## Executive conclusion

The slow development cycle is primarily caused by workflow and evaluation
design, not by gate inference:

- Two legacy tests can consume about **16 minutes** because each reaches the
  real 480-second race timeout.
- A cold `race_01` synthetic evaluation took **58.030 seconds**, while the same
  evaluation with a cache hit took **3.170 seconds** (18.3x faster).
- `tests/test_benchmark_matrix.py` requests **26 synthetic track executions**,
  including three separate full seven-track matrices.
- The standing autonomous-agent instructions target the old VQ1/PyBullet
  objective, not the current VQ2 IMU-plus-vision system.
- The historical `benchmark_history.jsonl` contains 43 pretty-printed objects
  over 7,345 lines; all 43 reported overall success while their PyBullet tier
  was skipped.
- The active VQ2 detector is already sub-millisecond. Optimizing it will not
  materially improve development latency.

The intended outcome is a layered loop in which normal VQ2 edits receive
feedback in roughly five seconds, expensive deterministic artifacts are
reused, and only promoted candidates reach the live simulator.

## Current verified baseline

The current VQ2 scoped suite is green at the baseline commit:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  competition/tests `
  estimation/tests `
  gate_detection/tests `
  tests/test_aigp_vq2_runner.py `
  tests/test_vision_udp.py `
  tests/test_vision_udp_listener.py
```

Measured on this host after the gate-1 observation commit:

```text
294 passed, 3 warnings in 4.09s
end-to-end process wall time: 4.334s
```

The three warnings are existing `PytestReturnNotNoneWarning` warnings in
`gate_detection/tests/test_detection.py`.

Do **not** use an unfiltered full `pytest` as the first validation command until
P0 below is complete.

## Measured bottlenecks and evidence

### 1. Two tests can wait 480 seconds each

The following tests are marked `slow`, but no pytest configuration registers
or excludes that marker:

- `tests/test_aigp_vq1_runner.py:80`
- `tests/test_aigp_vq1_runner.py:115`

Their fake adapter has a static position, so the race never completes. The
session then uses the production timeout and real sleeps:

- `competition/session.py:36` defines `MAX_RUN_DURATION_S = 480`.
- `competition/session.py:168-171` waits for that timeout.
- `competition/session.py:199-202` rate-limits with real `asyncio.sleep`.

The audit's full-suite process exceeded 120 seconds and survived the parent
command timeout until explicitly terminated. If allowed to finish, these two
tests can cost approximately 960 seconds total.

Required resolution:

- Prefer deterministic virtual time or an injected short maximum duration in
  these tests.
- Also classify them correctly so the default inner loop never invokes a
  production-length timeout.
- Add a hard per-test wall timeout so a future regression cannot orphan a
  long-running child process.

### 2. Planning dominates synthetic benchmark startup

Measured with the existing code:

| Track / state | Actual wall time |
|---|---:|
| `race_01`, cold | 58.030s |
| `race_01`, warm | 3.170s |
| `vertical_cliff`, cold | 9.629s |
| `vertical_cliff`, warm | 1.793s |

Reducing simulated duration barely changes cold startup time. Most work occurs
before rollout:

- `planning/racing_line.py:157` uses ten optimizer starts.
- `planning/racing_line.py:348-359` runs L-BFGS-B for every start.
- `planning/racing_line.py:440-528` builds and simulates trajectories for the
  candidates and up to three interpolations.
- `planning/trajectory_optimizer.py:1794-1860` uses a numerically
  differentiated time-allocation objective.

One instrumented run spent nearly all of its time in the trajectory objective
and finite differences. This is setup cost, not simulated flight duration.

### 3. Benchmark timing hides the setup cost

`scripts/benchmark.py` performs racing-line optimization, trajectory
generation, validation, and ILC before starting its wall timer around line
528. A cold 9.629-second evaluation reported only 0.440 seconds.

Autonomous agents cannot optimize the development cycle if the result omits
the dominant phase. Future results must report at least:

```text
startup
config_load
cache_lookup
racing_line
trajectory
plan_validation
ilc
rollout
metrics
total_wall
cache_hit_or_miss
```

### 4. The racing-line cache thrashes across tracks

The cache is a single JSON record:

- `planning/racing_line.py:70-72` defines one cache file.
- `planning/racing_line.py:203-225` accepts it only when the one stored key
  matches.
- `planning/racing_line.py:229-251` overwrites that record after a miss.
- `.gitignore:22-27` explicitly describes the last-writer-wins behavior and
  notes that tests often leave a toy layout in the cache.

`scripts/benchmark_matrix.py:65-80` visits tracks sequentially, so nearly every
track evicts the previous track. Multiple autonomous workers would also race
on this shared file.

### 5. Matrix tests recompute identical deterministic results

`tests/test_benchmark_matrix.py` requests 26 track evaluations:

- single-track calls near lines 39, 80, and 119;
- full seven-track matrices near lines 179 and 243;
- seven direct track evaluations near lines 298-304;
- a two-track matrix near line 357.

One session-scoped seven-track result could support most of these assertions,
reducing executions from 26 to 7 before any cache improvement.

### 6. Perception is not the bottleneck

Repository measurements on real VQ2 frames:

- active red-only detector: approximately **0.673ms median / 1.05ms p95**;
- generic detector: approximately 32.277ms median;
- JPEG decode: approximately 0.54ms;
- targeted detector tests: roughly 1.6 seconds.

The dedicated duplicate-suppressing vision thread is necessary because the
simulator sends roughly 55,000 datagrams per second with extensive duplicate
chunks. Preserve that separation from the 50Hz control loop.

The host has a Ryzen 7 7800X3D (8 cores / 16 logical processors) and an RTX
4090. The active detector is CPU-based and fast enough that GPU migration is
not currently justified.

## P0: Correct the default inner loop

### P0.1 Fix and classify long tests

Add central pytest configuration, preferably in `pyproject.toml`, with strict
marker registration for at least:

```text
unit
slow
benchmark
live
```

Add a hard timeout facility (for example, a pinned `pytest-timeout` development
dependency). The default command should exclude `slow`, `benchmark`, and
`live`. Explicit commands must still be available for each tier.

Do not merely hide the two broken VQ1 tests: make them use a fake clock or a
short injected duration so they remain useful when their tier is explicitly
run.

Acceptance criteria:

- A deliberately stuck test is killed by a short test timeout.
- The two VQ1 flow tests complete deterministically in seconds, not minutes.
- The default suite cannot invoke a live simulator or production-length wait.
- Unknown pytest markers fail collection (`--strict-markers`).

### P0.2 Make VQ2 the canonical agent workflow

The following are stale and currently misdirect autonomous agents:

- `Agents.md` still says the simulator/API details have not been released.
- `CLAUDE.md` identifies VQ1 and requires `scripts/benchmark.py --mode full`
  after changes.
- `docs/autonomous_iteration.md` describes a recurring ten-minute VQ1-style
  synthetic optimization prompt.

Update the canonical instructions to say:

- current target is FlightSim build 3385, Training mode;
- VQ2 lacks usable pose and gate-map data;
- current production path is vision + `HIGHRES_IMU` + race status;
- directly affected tests are the per-edit gate;
- the 294-test VQ2 suite is the per-candidate gate;
- synthetic/PyBullet matrices are module-specific or pre-merge gates, not the
  universal objective;
- powered FlightSim stages require explicit authorization and must retain the
  reset/countdown/watchdog/cleanup contract.

Keep `docs/aigp/2026-07-18-vq2-handoff.md` authoritative for flight status.

### P0.3 Add one Windows command surface

Create a small, documented PowerShell entry point or equivalent task runner
with stable commands resembling:

```text
test-target <paths>
test-fast
test-vq2
test-benchmark
test-full-non-live
preflight
```

`preflight` is passive. Do not include powered stages in a generic test task.

The existing launcher also needs to stop hardcoding the old user and build:

- stale: `C:\Users\Kenichi\...\AIGP_3364`
- current: `C:\Users\John\AIGP\AIGP_3385\FlightSim.exe`

Parameterize the simulator path and discover the active interactive session;
do not hardcode session ID 1. Preserve the existing double-launch guard. GUI
selection of Training mode may still require an interactive desktop action.

### P0.4 Pin and split environments

There is no lockfile and the requirements use broad lower bounds plus an
unpinned Git dependency. The current environment resolved future major
versions such as NumPy 2.5, OpenCV 5, and pytest 9, while declared PyBullet
packages are absent.

Split at least:

```text
runtime-vq2
development/test
legacy-simulation
optional-training
```

Lock exact versions for the development environment. A fast resolver such as
`uv` is reasonable, but preserve whatever submission format the competition
requires.

## P1: Reuse expensive deterministic work

### P1.1 Introduce content-addressed prepared artifacts

Replace the single record with versioned, content-addressed entries such as:

```text
.cache/
  racing-lines/<hash>.json
  trajectories/<hash>.npz
  ilc/<hash>.npz
  benchmark-results/<hash>.json
```

The key must include:

- canonical track geometry and start state;
- fully resolved racing-line, planner, controller, ILC, and drone settings
  relevant to the cached layer;
- an explicit schema/algorithm version or source digest;
- dependency/environment fields that affect numeric results.

Use atomic writes and per-key locks. Tests must receive an isolated temporary
cache root instead of sharing the developer cache.

Split the benchmark API conceptually:

```python
prepared = prepare_course(track, planning_config)
result = simulate(prepared, controller_config, seed)
```

Cache the racing line, trajectory, validation result, sampled reference arrays,
and ILC tables. Controller/residual sweeps should not rebuild planning.

Acceptance criteria:

- cold and warm runs produce identical metrics;
- cache corruption or a partial write fails closed and rebuilds safely;
- concurrent distinct keys do not overwrite each other;
- a config/schema change invalidates the correct artifact;
- repeated warm `race_01` evaluation remains in the low-single-digit seconds.

### P1.2 Reuse one matrix result across assertions

Create a session-scoped prepared matrix/result fixture and make the independent
assertions read from it. Keep focused single-track tests only when they exercise
a genuinely different config or trace mode.

After cache isolation, run independent tracks in worker **processes**, not
threads. Begin with 4-6 workers on this host and cap BLAS/OpenCV threads per
worker to avoid oversubscription.

### P1.3 Make benchmark results honest

Every benchmark result should include:

- end-to-end wall time and phase timings;
- resolved configuration and hash;
- code commit plus dirty-diff hash;
- evaluator/schema version;
- dependency fingerprint;
- seed;
- cache hit/miss state;
- stdout/stderr failure summary;
- safety, validity, and completion fields.

Other correctness gaps to address while touching this interface:

- `--config` currently does not feed the synthetic mode in the main CLI.
- unknown dataclass override keys are silently discarded;
- a full run can still pass when PyBullet is skipped unless strict mode is
  selected;
- the current environment does not have PyBullet installed.

Do not silently reinterpret historical metrics after changing evaluator
semantics. Version the evaluator and start a new comparison series.

## P2: Build a durable autonomous improvement loop

### P2.1 Use a promotion ladder

The intended validation ladder is:

| Tier | Trigger | Gate | Target wall time |
|---|---|---|---:|
| T0 | Every edit | Directly affected tests/import checks | <2s |
| T1 | Every accepted code candidate | VQ2 scoped suite + golden replay | ~5s |
| T2 | Relevant control/planning candidate | One warm prepared simulation | 2-5s |
| T3 | Promising simulation candidate | Changed-domain track subset | Tens of seconds |
| T4 | Pre-merge/nightly | Parallel seven-track matrix + full non-live suite | Minutes, outside inner loop |
| T5 | Explicitly promoted candidate | Bounded official-simulator trial | Live and authorized only |

Completion and safety are hard gates. Optimize lexicographically:

1. no collision, disqualification, stale-stream flight, or cleanup failure;
2. correct gate sequence and completion reliability;
3. centering/stability margins;
4. race time.

Do not collapse these into a scalar that can trade a safety failure for speed.

Use successive halving rather than fully evaluating every proposal: one cheap
trial for all candidates, more tracks/repetitions only for survivors, and live
validation only for finalists.

### P2.2 Build a VQ2 replay corpus

Current VQ2 captures retain useful telemetry and detected targets, but there is
not yet a durable scored corpus containing all required frame pixels. Some
saved detector benchmark metadata references JPEGs absent from this checkout.

Capture deduplicated, decoded frames synchronized with:

- IMU samples and estimator state;
- race status and active gate index;
- detector outputs and tracker state;
- generated/sent commands;
- target-loss, transition, collision, abort, and cleanup events.

Retain immutable full sessions outside normal Git. Keep a small labeled golden
set in an appropriate artifact store or Git LFS if desired. Split validation by
session/flight, not adjacent video frames.

Score at least:

- gate recall and false positives per frame;
- center/corner error;
- consecutive missed frames;
- temporal target stability;
- post-gate reacquisition latency;
- p50/p95 detector latency;
- estimator and open-loop generated-command behavior.

Replay can validate perception, estimation, and open-loop command generation
quickly. It cannot replace closed-loop FlightSim evaluation for changes whose
commands alter the future observations.

### P2.3 Add a resumable trial scheduler and structured ledger

Do not use the existing recurring ten-minute prompt as the long-running loop.
It has no lease, overlap prevention, isolated worktree, deduplication, or
resume protocol.

Use one scheduler and one merger. Each code candidate gets an isolated Git
worktree. Store one structured row per trial in SQLite with at least:

```text
trial_id
parent_trial_id
status
lease_owner
heartbeat
started_at / finished_at
commit_hash / dirty_diff_hash
resolved_config / config_hash
dataset_hash / artifact_hashes
simulator_build / evaluator_version
environment_fingerprint
seed
phase_timings
safety_and_completion_metrics
failure_reason
stdout_stderr_tail
```

Deduplicate by `(code hash, config hash, dataset hash, seed, evaluator
version)`. Checkpoint after every promotion tier. A stopped session must resume
without repeating completed expensive evaluations.

Use LLM agents for bounded hypothesis generation, diagnosis, and code review.
Use deterministic search/optimization workers for parameter sweeps. The
repository's previous 150-agent reconciliation produced useful findings but is
not an efficient per-iteration operating model.

### P2.4 Keep the simulator warm without weakening safety

The current CLI reconnects, runs preflight, executes one stage, cleans up, and
disconnects. A future campaign process may keep the transport/vision process
warm, but every powered trial must still:

- prove a fresh reset epoch;
- observe countdown and GO;
- arm on fresh authoritative state;
- enforce all stream/attitude/rate/collision watchdogs;
- disarm/reset and prove cleanup.

Record simulator build, process uptime, session trial count, and preflight
health. Interleave a known-good baseline periodically to detect simulator drift
before attributing a regression to code. The older VQ1 simulator was observed
to degrade after roughly 25 runs; do not assume that exact threshold applies to
VQ2 without measurement.

## Long-running training warnings

### YOLO pose experiment

Do not spend autonomous iteration budget on the checked-in YOLO run yet:

- training took 62,629.7 seconds (17.4 hours);
- `ARCH.md` names extraction/train/validate/export scripts that are absent;
- the dataset is absent and ignored;
- `args.yaml` contains an absolute macOS data path;
- Torch and Ultralytics are absent from requirements and the current venv;
- `best.pt` has no runtime code references;
- identical 6.4MB best weights are stored twice;
- pose mAP50-95 was already about 0.978 after 10 epochs and 0.989 after 25,
  so 100-epoch full trials were inefficient.

Before resuming learned detection, establish data provenance, locked
dependencies, scripts, grouped splits, runtime integration, early smoke
epochs, successive halving, periodic checkpoints, and resume support. Compare
ONNX/TensorRT accuracy and latency against the existing sub-millisecond
classical detector before adopting them.

### Residual-controller training

The default residual trainer can run a seven-track baseline plus another
seven-track matrix every 25 epochs, up to approximately 147 track runs over 500
epochs. It repeatedly rebuilds planning. Its checkpoint evaluator runs for 15
simulated seconds while current `race_01` completion is about 29.9 seconds, so
otherwise viable checkpoints can all receive the same hard-failure score.

Before an unattended residual campaign:

- reuse prepared trajectories and baseline results;
- score truncated-prefix behavior separately or run long enough to complete;
- make training and acceptance horizons consistent;
- hold out complete sessions/tracks rather than randomly interleaving samples
  from a 100Hz time series;
- save model, optimizer, RNG, epoch, history, and best-candidate state after
  every expensive evaluation.

## Experiment history cleanup

`docs/autonomous_iteration.md` tells agents to append the benchmark output to
`benchmark_history.jsonl`, but `scripts/benchmark.py` pretty-prints JSON with
indentation. The resulting file is a stream of multiline JSON objects, not
JSONL:

```text
43 objects
7,345 lines
0 individually valid JSON lines
43 overall_passed=true
43 simulation.skipped=true
```

Do not append new results to it. Preserve it as historical evidence, write a
one-time importer if needed, and store new trials in the structured ledger.

The repository also tracks 369 `.loop`/`.research_loop` files out of 631 total
tracked files. This is not a major filesystem bottleneck, but it is substantial
search and context noise for coding agents. Preserve the historical data, but
keep a short current decision digest and explicitly exclude the archive from
routine agent search/indexing. Do not delete or relocate it without user
approval.

## Suggested commit sequence

Keep changes reviewable and measure after every commit:

1. **Test safety:** virtualize/bound VQ1 waits; markers; strict markers; test
   timeout; fast default command.
2. **Canonical workflow:** Windows task entry point; current VQ2 agent docs;
   parameterized build-3385 launcher.
3. **Honest timing:** end-to-end and per-phase benchmark timing, without
   changing benchmark semantics.
4. **Artifact cache:** content-addressed per-key prepared artifacts and
   isolated test cache.
5. **Matrix reuse:** one session result, process-parallel affected tracks.
6. **Experiment ledger:** schema, idempotent trial runner, resume/deduplication.
7. **Replay corpus:** capture bundle format, labeled golden set, scoring CLI.

After each commit, run directly affected tests and the current VQ2 scoped
suite. Run expensive matrices only once the relevant cache/matrix commit is
ready to measure.

## Expected wins

| Change | Expected effect |
|---|---|
| Bound/exclude two legacy VQ1 tests | Removes up to ~960s from accidental full runs |
| Canonical scoped VQ2 gate | Normal feedback in ~4-5s |
| Content-addressed prepared cache | Measured cold-to-warm improvement up to 18.3x |
| One matrix fixture | Reduces one module from 26 track runs to 7 |
| Process-parallel tracks | Uses the available 8 CPU cores instead of sequential execution |
| Replay-first promotion | Eliminates unnecessary live trials for perception/open-loop regressions |
| Structured resumable scheduler | Prevents duplicate work, overlapping jobs, and lost long-run progress |

## Audit side effects and repository state

The audit made no source or tracked-file changes. Benchmark diagnostics did
refresh the existing ignored generated file
`planning/racing_line_cache.json`; its content is not part of Git. A timed-out
pytest child created by the audit was identified by its exact command line and
terminated.

At the time this handoff was created, the user's gate-1 observation changes had
already been committed at the baseline commit above and the worktree was clean.
This handoff document is the only intended new tracked file from this session.
