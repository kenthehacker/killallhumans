# Package 2 F03 powered calibration attempt

- Task ID: `vq2-package2-f03-powered-calibration-attempt`.
- Parent: `vq2-package2-f02-powered-calibration-attempt`.
- Starting local-main commit:
  `1479a85827bca72613b63bfecc81a395136cb36a`.
- Simulator target: FlightSim build 3385, Training mode.
- State: `terminal-invalid and poisoned; no retry or automatic clear`.
- Contract date: `2026-07-22`.

## Authority and stop boundary

The user's 2026-07-22 instruction to resume the session handoff and prioritize
actual runs authorizes preparation, publication, and one consumption of exact
`F03-A01` by the existing no-selector `calibration-excite` wrapper. The capture
authority is `conversation-2026-07-22-package2-f03-sim-capture`, direct user
instruction, simulator-only, and task-local.

This authority does not permit another attempt or retry, `sign-id`, `hover`,
`gate0`, `gate0-observe`, Gate 1 control, a lap, another build/mode, physical or
HIL use, fitting, calibration acceptance, cross-task data access, public
release, or external upload. Any candidate, freeze, environment, stage,
waveform, or command change requires a new checkpoint.

## Immutable F02 predecessor

`F02-A01` is terminal-invalid and poisoned. It failed before simulator contact
or powered commands because the 91,968,000-byte payload exceeded the generic
64 MiB stable-file hash ceiling. Preserve the predecessor root
`C:\Users\John\aigp-evidence\2026-07-21-package2-f02-powered-calibration-attempt`
and bind these exact files:

- attempt envelope: size `9927`, SHA-256
  `f8e7071dc35f6d336d6127c6164c87194313c4e88723adb9435531b15b7df8b3`;
- attempt-invalid: size `2809`, SHA-256
  `d64b943184796c10bc91cdc257f57e5cfab16765de8d92a9e6889319604c481b`;
- wrapper lifecycle: size `4751`, SHA-256
  `c82528d2ed4bd766823b74694b9ba04e8c57a413f0b765a48c30593dc0a6d71a`;
- live poison: size `1891`, SHA-256
  `63a215a250bedde84cfdd2714745e7762b2b97e27ed0a2782e4283592e9c2005`;
  and
- last lease generation: size `1493`, SHA-256
  `faec921e6ef7518a7c91364e031cfbec9ba99d32aef887cf210ebe503267661e`.

F03 fixes only process-image hashing: process identity admits at most 128 MiB,
while the generic stable-file limit remains 64 MiB. It adds one portable
regression test and does not change flight behavior or safety limits.

## Frozen F03 identities

- session: `F03`;
- sole attempt: `F03-A01`;
- private evidence root:
  `C:\Users\John\aigp-evidence\2026-07-22-package2-f03-powered-calibration-attempt`;
- detached live worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-f03-powered-calibration-attempt-live`;
- freeze ID:
  `vq2-package2-f03-powered-calibration-attempt-f03-a01-live-freeze`;
- freeze path: `<private-root>\live-freeze-F03-A01.json`;
- scheduled task: `AIGP-P2-F03-A01-Launch`;
- run ID: `F03-A01/reset-epoch-1/excitation-1`;
- split: `discovery_fit`;
- plan ID: `vq2-build3385-training-f03-excite-v1`;
- plan canonical-object SHA-256:
  `73a1134906edeb6480e189cdf9df1d9d30eac697537d3da44d5d0de075237e7b`;
  and
- plan canonical-file SHA-256:
  `d69ea76c6f3d4d44b8ec17c70e813eb79fe8530e995ced4e65a91a77333c221a`.

## Runtime and safety contract

The exact waveform, 20 ms pacing, 245 ticks, 0.235 thrust, zero yaw, command
bounds, watchdogs, reset-epoch proof, GO plus 150 ms, cleanup, invalidation, and
poison truth table remain those frozen for F02 and the authoritative build-3385
handoff. The run never enters crossing confirmation or attempts a gate pass.

FlightSim may already be running. The wrapper must validate and adopt only the
exact unique build-3385 launcher/payload topology, then prove it is unchanged
after the frozen launcher returns. Do not close a valid simulator merely to
satisfy F02's old optional external absence checkpoint. The production wrapper
itself still must prove topology, Training mode, fixed-port ownership, fresh
streams, reset epoch, disarm, countdown, target, all powered watchdogs, and full
cleanup. A cleanup failure fails the stage; fallback use invalidates F03.

Before consumption, publish create-new in this exact order: capture authority,
plan, implementation inventory, environment inventory, import inventory, and
live freeze last. The detached candidate must be physically pristine and the
two isolated derivations byte-identical. The production wrapper must run in an
attached local console so the operator can visually attest Training mode using
the generated challenge. No F02 external preparer, recovery, or debug launcher
is reusable as F03 authority.

## Minimum verification

Run the five directly affected test files and then the canonical dedicated
suite:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_calibration_target.py tests/test_aigp_vq2_powered_attempt.py tests/test_aigp_vq2_powered_calibration_analysis.py tests/test_aigp_vq2_powered_calibration_probe.py tests/test_aigp_vq2_powered_runtime.py
.\scripts\dev.cmd test-vq2
```

## Terminal result

Candidate commit `bdbd45649c7d9baab7d5025434205930bcb84561` was
validated in a detached pristine worktree. The five directly affected test
files passed `471` tests, and `test-vq2` passed `2254` tests with one skip. A
passive preflight also passed before publication.

The reviewed six-file L0 bundle was published create-new with live-freeze
SHA-256
`ca7a5d7dc08aaf6a7fecd3d38675918090a08fafa75121349a19ceac114ea2e9`.
The first external launcher invocation refused before wrapper creation because
its review-only `sys.orig_argv[0]` gate expected the venv path instead of
CPython's base-executable path. The launcher was narrowly corrected and
independently re-reviewed; no candidate or frozen-evidence byte changed.

The sole authorized wrapper consumption then reached
`topology_and_training_attestation`, but no matching operator attestation was
received within the 30-second deadline. `F03-A01` failed before the powered
child was created, so there was no transport connection, reset, arm, attitude
target, thrust command, capture, or powered cleanup child. The wrapper proved
the lease released, ports free, transport closed, scheduled task absent,
simulator responsive, and the original launcher/payload topology unchanged.

Preserve the terminal evidence exactly:

- attempt envelope SHA-256:
  `4082339925ad6da687161b05ade3bdbc3c1f7ce39c045a4e119acaf552193e4a`;
- attempt-invalid SHA-256:
  `b8df4b33b30f9a5b267b4650e6db91a55c6e9236b0f5c46651eb87f36e3a820f`;
- live-poison SHA-256:
  `185bdb4a3b970ab323e9a40f4fcbfb805134aca32ad9c463d973a78e298f1fb3`;
- process-final-proof SHA-256:
  `732ccdabf97283ee62cffb9b767e802fa349d8f90c608cc0d23eb2c8fcf72a88`;
  and
- live-lease SHA-256:
  `8908f04433eab6ae0794a0cf357e0364a7e1f9075cd9c56ed755beb792824f4b`.

The poison requires `new_reviewed_recovery_task_no_automatic_clear`. Do not
recover, clear, or rerun `F03-A01`; another powered attempt requires a new task,
new identities and freeze, and fresh explicit user authorization.
