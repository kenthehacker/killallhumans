# Package 2 F02 powered calibration attempt

- Task ID: `vq2-package2-f02-powered-calibration-attempt`.
- Parent: `vq2-package2-launcher-session-recovery`.
- Starting main commit:
  `7be5111199161adbf545fd4e58034a31bd2f3a0a`.
- Simulator target: FlightSim build 3385, Training mode.
- State: `offline preparation, promotion, publication, and review only`.
- Simulator access before the final checkpoint: passive host/process/task/port
  absence proofs only. No launch, fixed-port contact, preflight stream contact,
  reset, arm/disarm, target, or powered command is authorized by this record.
- Contract date: `2026-07-21`.

## Authority and stop boundary

The user's 2026-07-21 instruction authorizes preparation of a fresh F02 task,
private simulator-capture authority, exact candidate, pristine live worktree,
bounded `calibration-excite` plan, inventories, freeze, and independent review.
It does not replace the controlling successor gate in the parent task: powered
authority must be a new explicit checkpoint after the exact F02 publication and
review exist. L0 therefore stops after post-publication revalidation and a
passive pre-L1 absence attestation. The exact candidate, bundle, freeze, and
wrapper identities are then presented to the user for that checkpoint.

If that later checkpoint is granted, it authorizes at most one consumption of
`F02-A01` by the exact no-selector wrapper. It does not authorize `sign-id`,
`hover`, `gate0`, `gate0-observe`, Gate 1 steering or passage, a lap, another
attempt, another build or mode, physical/HIL/submission use, fitting, calibration
acceptance, cross-task data access, training/corpus reuse, or public release.
Any semantic, stage, command, candidate, freeze, or environment change requires
a new task and checkpoint.

## Immutable predecessor and launcher provenance

`F01-A01` is consumed, terminal-invalid, and poisoned. It must not be retried,
renamed, cleared, repaired, copied into this root, or used as authority. The
successor review must bind these immutable files in the predecessor root
`C:\Users\John\aigp-evidence\2026-07-21-package2-import-environment-recovery`:

- attempt-invalid: size `2772`, SHA-256
  `edf2424bbf60fb305fc805a28b561fa37982c77e9f8321ed35c91305e3dbaeb0`;
- live poison: size `1886`, SHA-256
  `009edcb1b7f48d120c1f46b393a9094073e378ed122d3ac75c2a565a65fae91d`;
- attempt envelope: size `9759`, SHA-256
  `e939c1f35e5a41ff350a7319f3f7f8c22b4a733c9104b428682e1ac38d12eab5`;
- wrapper lifecycle: size `4698`, SHA-256
  `218fe368f6971ab309be4c09c7a87e4d86b9d49624590641fe7d4a5083bc880f`;
  and
- last lease generation: size `1493`, SHA-256
  `9e69def312f13f7b90b5a265851e0f83f91ec8419780f5df37836c41286fb898`.

F01 failed before simulator launch because the verified active-session table
arrived with `query.exe session` status `1`. The launcher correction is exact
main commit `7be5111199161adbf545fd4e58034a31bd2f3a0a`; its tracked launcher script
SHA-256 is `e3e96f2268f6b8b9877448dae67eef652add72e11a37d94fc35154b1c8c73595`.
The selector admits only status `0` or `1`, and either admitted status still
requires bounded output, exactly one expected English header, and at least one
strictly parsed Active row.

The conservative early-launch failure lane remains unchanged: launcher output
is discarded, and a failed launcher return after attempt consumption is
ambiguous until contemporaneous topology is proved. It must retain the lease
and produce invalid/poison if cleanup and topology cannot be proved. Later OS
absence never repairs F01 and must not be used to release or rewrite it.

## Frozen F02 identities

- session: `F02`;
- sole attempt: `F02-A01`;
- private evidence root:
  `C:\Users\John\aigp-evidence\2026-07-21-package2-f02-powered-calibration-attempt`;
- attempt directory: `<private-root>\F02-A01`;
- detached live worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-f02-powered-calibration-live`;
- independent review root:
  `C:\Users\John\aigp-review\2026-07-21-package2-f02-powered-calibration-attempt`;
- freeze ID:
  `vq2-package2-f02-powered-calibration-attempt-f02-a01-live-freeze`;
- freeze path: `<private-root>\live-freeze-F02-A01.json`;
- scheduled task: `AIGP-P2-F02-A01-Launch`;
- run ID: `F02-A01/reset-epoch-1/excitation-1`;
- split: `discovery_fit`;
- plan ID: `vq2-build3385-training-f02-excite-v1`;
- plan canonical-object SHA-256:
  `efe51d3a674d6a4a68245423424036362f5063d785ae1f45d0103817de410132`;
  and
- capture authority ID:
  `conversation-2026-07-21-package2-f02-sim-capture`.

The capture authority is simulator-only and task-local. It permits only the
private F02 acquisition classes already admitted for Package 2, sealing,
bundle verification, acquisition analysis, and independent integrity review.
It grants no transfer to a successor, other session, other build/mode,
submission, physical/HIL use, external service, Git, or public release.

## Exact bounded excitation and safety contract

The plan changes only fresh identity. Its stage, tick timing, waveform, thrust,
and safety values are byte-semantically unchanged from the reviewed discovery
plan: 245 absolute ticks at 20 ms (`4.9 s` nominal, `5.0 s` hard expiry), thrust
`0.235`, yaw rate exact zero, and these nonzero roll/pitch segments:

| Ticks, inclusive | Roll / pitch rate (rad/s) |
| ---: | ---: |
| `30-44` | `(+0.08, 0.00)` |
| `54-73` | `(-0.06, 0.00)` |
| `86-105` | `(0.00, +0.07)` |
| `116-133` | `(0.00, -0.08)` |
| `150-164` | `(+0.06, +0.04)` |
| `165-179` | `(-0.06, +0.04)` |
| `180-194` | `(-0.06, -0.04)` |
| `195-209` | `(+0.06, -0.04)` |

Every other tick is zero-rate. Missed ticks drop and never replay; command
pacing is at most 50 Hz. There is no amplitude, waveform, thrust, duration, or
stage selector.

Before power, the wrapper must prove exact build-3385/Training topology and the
local interactive Training challenge, a fresh exclusive lease, free fixed
ports, same-loopback peers, stopped vision during reset, race- and IMU-clock
rollback with multiple advancing samples, vision restarted only in that epoch,
normalized and proved disarm, countdown, GO plus 150 ms, actual stable 640x360
decode, and three stable target frames. Arm/disarm confirmation accepts only a
newer heartbeat.

Before every send it rechecks parent, lease, deadlines, fresh heartbeat,
actuator, camera, target, capture health, advancing IMU and race streams,
estimator health, collision state, gate index zero, and the exact command
envelope. Any collision has no launch-pad exception. Any gate-index change,
second source, unknown outbound, queue drop/overflow, target loss, identity or
lineage drift, nonfinite value, or stale evidence aborts before another send.
Roll and pitch excursion from start remain at most `0.05 rad`; commanded roll
and pitch remain within `0.25 rad/s`; yaw remains exactly zero. Target center
must stay in closed `[0.10W,0.90W] x [0.10H,0.90H]`, width and height at most
160 px, and area at most `2*A0`. The stage never enters crossing confirmation
and never approaches or passes a gate.

Completion or abort latches production, sends exact zero rate/thrust when
required, confirms disarm on a newer heartbeat, resets, proves a clean advancing
race/IMU epoch and final disarm, closes vision/MAVLink/workers/handles, proves
child-tree/task/port/topology cleanup, and releases the lease. Any fallback use
invalidates the collection even if cleanup succeeds. Any cleanup failure fails
the stage and requires terminal invalid plus poison.

## Implementation, verification, and promotion

Production identity rollover is limited to the existing Package 2 capture,
attempt, probe, and analysis contracts plus their direct tests. It must not
change the runner waveform, controller, estimator, detector, transport,
launcher behavior, capture schemas, lease protocol, deadlines, watchdogs,
command bounds, or cleanup truth table.

After each affected edit run its direct target. Before acceptance run:

```powershell
$env:AIGP_PYTHON = 'C:\Users\John\killallhumans\.venv\Scripts\python.exe'
.\scripts\dev.cmd test-target tests/test_aigp_vq2_calibration_target.py tests/test_aigp_vq2_powered_attempt.py tests/test_aigp_vq2_powered_calibration_probe.py tests/test_aigp_vq2_powered_calibration_analysis.py tests/test_aigp_vq2_powered_cleanup.py tests/test_aigp_vq2_runner.py tests/test_aigp_live_lease.py tests/test_windows_tooling.py
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
.\scripts\dev.cmd test-vq2
```

At the promotion boundary update only exact trusted digests and count arithmetic.
Use separate fresh external worktrees for `test-full-non-live` and the isolated
hash-pinned VQ2 policy, integrate the exact candidate, run detached post-merge
`test-vq2`, inventory physical side effects, and prove tracked `main` clean.
No automated test may acquire the production live mutex/lease, bind or send on
UDP 14550/5600, launch/query FlightSim, or read/write either private evidence
root.

## L0 publication and checkpoint package

L0 uses a detached physically pristine live worktree at the exact integrated
candidate. It binds exact Python, PowerShell, development lock, launcher script,
FlightSim launcher/payload, target configuration, plan, implementation,
environment, and import identities. Two isolated derivations must be
byte-identical. `PYTHONNOUSERSITE=1` and `PYTHONDONTWRITEBYTECODE=1` are exact;
`PYTHONHOME`, `PYTHONPATH`, and `PYTHONSTARTUP` are absent.

The private root is create-new on the fixed local volume, current-user owned,
inheritance-disabled, non-reparse, current-user-only effective DACL, stable
readback, and one-link for every file. Publish capture authority, plan,
implementation inventory, environment inventory, import inventory, and freeze
in that order with the freeze last. No unexplained entry, attempt directory,
split registry, or poison is permitted.

Independent review must bind the exact candidate, F01 provenance, launcher,
environment/import inventories, wrapper argv, bounded plan, early-launch
ambiguity, cleanup contract, review bundle, and all hashes. Immediately before
the checkpoint package, passive proof must show the old and new wrappers,
scheduled tasks, FlightSim launcher/payload, and UDP 14550/5600 owners absent.
That proof does not repair F01. The exact F02 attempt must independently acquire
its own lease only after the later powered checkpoint.

The checkpoint package contains the candidate commit, review-bundle SHA-256,
live-freeze SHA-256, capture-authority SHA-256, plan identities, wrapper argv,
passive absence attestation, and independent review result. Stop there and ask
the user to explicitly release exactly that one F02-A01 attempt.

## Post-attempt quarantine

After the one attempt, E0 may only validate the exact terminal/poison choice,
lifecycle, lease, cleanup, capture seal, replay bundle, acquisition report, and
split identities and obtain independent acquisition-integrity review. All F02
data stays sealed quarantine. Fitting, rank/identifiability claims, limit or
calibration acceptance, held-out design, successor access, Gate 1 work, or data
disposition requires a separate new hash-bound task and authority.
