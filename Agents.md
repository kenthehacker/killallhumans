# AI Grand Prix engineering instructions

## Current target

The active target is FlightSim **build 3385** in **Training** mode. Treat
`docs/aigp/2026-07-18-vq2-handoff.md` as authoritative for live flight status,
the verified interface, and the safety contract.

Public VQ1 material is useful historical context, but it does not match every
empirical build-3385 behavior. In VQ2 there is no usable pose or gate-map
stream. The production path is:

```text
UDP JPEG vision + HIGHRES_IMU + race status
                  -> target tracking and IMU attitude estimation
                  -> safety-gated body-rate/thrust commands
```

Do not invent qualifier dates or assume an older public interface overrides a
verified build-specific finding. Flag any conflict for human review.

## Default development loop

Use the canonical Windows task surface from the repository root:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_runner.py
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
.\scripts\dev.cmd test-vq2
```

Run directly affected tests after each edit. Run `test-vq2` for each accepted
candidate; it is the fast, dedicated non-live safety suite, and its collected
test count is expected to grow with the stack. The default pytest policy
excludes `slow`, `benchmark`, and `live`, enforces strict markers, and applies
a hard wall timeout.

Synthetic and PyBullet matrices are module-specific or pre-merge evidence, not
the universal optimization objective. Invoke them explicitly with
`test-benchmark`; use `test-promotion` only at a promotion boundary. The
promotion suite normally takes 10-13 minutes because it includes the slow and
benchmark tiers. Its external, commit-keyed state lets a later caller attach
to an active run or reuse its terminal result; never launch a duplicate merely
because a console or API wait expired. Use `--fresh` only after reviewing an
incomplete or terminal attempt. `test-full-non-live` remains a compatibility
alias. Promotion runs require a fresh exact-commit worktree with no ignored
files, so ignored model/data inputs cannot bypass the durable key. Do not
append to `benchmark_history.jsonl`, whose historical records
are multiline objects and whose skipped PyBullet tier made old
`overall_passed` values misleading.

The explicit isolated and bounded legacy tiers are
`.\scripts\dev.cmd test-unit` and `.\scripts\dev.cmd test-slow`. A `live` marker is
run directly with `python -m pytest -m live` only after explicit authorization;
there is intentionally no generic powered-test task.

## Simulator boundary

`preflight` is passive and sends no arm or flight targets:

```powershell
.\scripts\dev.cmd preflight
```

Any powered FlightSim stage requires the user's explicit authorization. Never
put `sign-id`, `hover`, `gate0`, `gate0-observe`, or a future powered stage in a
generic test command. Powered work must preserve the proved reset epoch,
countdown/GO, fresh-stream checks, watchdogs, command bounds, disarm/reset, and
cleanup confirmation in the VQ2 handoff. For promotion or submission evidence,
a cleanup failure is a failed stage. During explicitly authorized rapid
Training-mode course iteration, use the standing policy below instead.

For rapid simulator iteration, a direct user instruction to run or continually
iterate a specific powered stage is the authorization for that scoped work. Do
not ask for another confirmation on every attempt while that instruction still
applies. Use the dedicated, noninteractive command (default stage is the
calibration excitation):

```powershell
.\scripts\dev.cmd flight-cycle [sign-id|hover|gate0|gate0-observe|calibration-excite]
```

`flight-cycle` is not a test task. It automatically writes one compact run
manifest, one JSONL trace, one result, and the small canonical live-lease record
under the private external evidence root. It does not run a separate passive
preflight, take screenshots, request a console challenge, wait for manual
approval, derive duplicate freezes, inventory the full
repository/environment/import graph, or synchronously review/analyze the
capture before returning. Do not add those gates back unless the user asks for
a forensic or promotion workflow. A failure before simulator contact does not
require a new F-number or poisoned attempt.

## Standing rapid-course iteration policy

This is the standing policy for every remaining build-3385 Training-mode
`visual-course` development iteration until authoritative `race_finished` or
the user explicitly revokes it. It applies after context compaction and in new
sessions. It overrides broader per-candidate validation, repeated-review, and
cleanup-promotion workflows for these rapid simulator iterations; it does not
relax final promotion or submission requirements.

Iteration speed and completing the course are the priority. A failed simulator
attempt is acceptable and useful evidence. For each diagnosed candidate:

1. Run directly affected tests.
2. Run `test-vq2` once.
3. If green, commit and push the exact clean candidate.
4. Run one bounded `visual-course` attempt.
5. Analyze its first causal navigation blocker and iterate.

Target roughly 15-25 minutes from a diagnosed live failure to the next powered
attempt. Do not run `test-fast`, `test-unit`, promotion, benchmark, or broad
matrices before every flight; reserve them for meaningful course milestones
or final promotion. Once directly affected tests are green, do not add another
review round, replay, abstraction, proof layer, or test-fidelity enhancement
before `test-vq2` and flight unless it exposes a hard blocker involving:

- nonfinite or out-of-envelope commands;
- wrong authoritative race-gate ownership;
- stale control inputs;
- the in-flight collision watchdog;
- live-lease concurrency;
- or failure to establish a usable simulator state for the next run.

Keep bounded commands, fresh-stream checks, the hard attempt timeout,
collision abort, and the host-wide live lease. Cleanup is best-effort zero,
disarm, reset, and final disarm from a `finally` path. Report navigation and
cleanup outcomes separately. Recorder/diagnostic failures, evidence
completeness, exact post-wire heartbeat proofs, and post-reset pad-contact
classification must not erase achieved navigation milestones or block the next
development iteration. If cleanup cannot establish a usable next-run state,
relaunch FlightSim before continuing.

The fast path still must retain reset/GO/freshness/watchdog/command-bound
invariants and the best-effort cleanup actions above. Training mode is not
machine-readable in build 3385; record it as configured session state without
claiming a fresh visual proof. The command takes the existing nonblocking
host-wide FlightSim live lease and refuses immediately if any legacy or fast
live workflow owns it.

The parameterized Windows launcher is:

```powershell
$env:AIGP_FLIGHTSIM_PATH = 'C:\path\to\AIGP_3385\FlightSim.exe'
.\scripts\dev.cmd launch-sim
```

It discovers the active interactive desktop and refuses to double-launch.
Training-mode GUI selection can still require a human desktop action.

## Data, dependencies, and disclosure

Use `requirements/development-test.lock.txt` for normal development. Keep the
VQ2 runtime, test, legacy simulation, and optional training environments
separate as documented in `docs/development_environment.md`.

Keep new/private full captures, credentials, simulator data, and generated
dependency inventories out of Git; preserve existing tracked historical
capture evidence. Before a submission, generate an inventory with
`.\scripts\dev.cmd sbom` and complete
`docs/disclosures/ai-and-tools-template.md` through human review. The generated
metadata is evidence, not a guarantee of license or rules compliance.

Historical `.loop` and `.research_loop` files are evidence. Preserve them, but
exclude them from routine searches unless the task explicitly needs history.
