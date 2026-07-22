# VQ2 session handoff — 2026-07-22

> **Process update:** The F00-F04 freeze, screenshot, attached-console
> challenge, duplicate-inventory, and attempt-rollover workflow described below
> is retained as historical evidence but is no longer the default iteration
> path. Use `docs/aigp/fast-flight-cycle.md` and
> `.\scripts\dev.cmd flight-cycle`. The compact path preserves the authoritative
> July 18 runtime safety contract and removes the pre-contact ceremony. Do not
> create a new F-number or poison an attempt for a failure before simulator
> contact.

Start here. Do not recover, clear, or rerun `F02-A01` or `F03-A01`.

## What completed

- Latest local `main` was used as requested. It started at
  `1479a85827bca72613b63bfecc81a395136cb36a`; remote `origin/main` remained
  `7be5111199161adbf545fd4e58034a31bd2f3a0a`.
- The F02 process-image hash-ceiling bug was fixed without changing the generic
  64 MiB stable-file limit. Process images now use an explicit 128 MiB ceiling,
  with a 91,968,000-byte regression case.
- The fresh F03 candidate is
  `bdbd45649c7d9baab7d5025434205930bcb84561` (`Roll Package 2 powered
  calibration to F03`).
- In an external detached pristine validation worktree, the five directly
  affected files passed `471` tests, and `test-vq2` passed `2254` tests with one
  skip.
- Passive preflight passed at 31 FPS with fresh camera/IMU/race streams, gate 0,
  bbox `(282, 134, 80, 80)`, center `(322, 174)`, pitch `-17.80°`, and confirmed
  cleanup. It sent no powered commands.
- The live detached worktree is
  `C:\Users\John\aigp-worktrees\wt-package2-f03-powered-calibration-attempt-live`
  at the exact F03 candidate and remains pristine including ignored files.

## Frozen publication

The private evidence root is:

```text
C:\Users\John\aigp-evidence\2026-07-22-package2-f03-powered-calibration-attempt
```

The reviewed L0 publisher created exactly the five inputs followed by the live
freeze as the final mutation. The protected current-user-only root and all six
files were re-proved against their pinned hashes. The live-freeze SHA-256 is:

```text
ca7a5d7dc08aaf6a7fecd3d38675918090a08fafa75121349a19ceac114ea2e9
```

The first external launcher invocation safely refused before wrapper creation:
the review-only gate expected the venv path in `sys.orig_argv[0]`, while Windows
CPython reports the base executable there. The external launcher was narrowly
fixed to pin and hash both identities, independently re-reviewed at SHA-256
`5924e76d706eddfac5b8e0847a2de60f4ee20c2618d2a0cf2285d3dba7774f83`,
and then invoked. No candidate or frozen-evidence byte changed.

## F03-A01 terminal result

The sole authorized wrapper consumption ended during
`topology_and_training_attestation`: the exact operator Training attestation
was not received within its 30-second deadline. The powered child was never
created. Therefore F03 made no transport connection and sent no reset, arm,
attitude-target, thrust, or other powered command; it produced no capture.

`F03-A01` is terminal-invalid and poisoned. Preserve these exact files:

- attempt envelope:
  `4082339925ad6da687161b05ade3bdbc3c1f7ce39c045a4e119acaf552193e4a`;
- attempt-invalid:
  `b8df4b33b30f9a5b267b4650e6db91a55c6e9236b0f5c46651eb87f36e3a820f`;
- live-poison:
  `185bdb4a3b970ab323e9a40f4fcbfb805134aca32ad9c463d973a78e298f1fb3`;
- process-final-proof:
  `732ccdabf97283ee62cffb9b767e802fa349d8f90c608cc0d23eb2c8fcf72a88`;
  and
- live-lease:
  `8908f04433eab6ae0794a0cf357e0364a7e1f9075cd9c56ed755beb792824f4b`.

Terminal cleanup proof says: powered child not created, fallback not eligible,
lease released, ports 14550/5600 free, transport closed, wrapper processes
exited, scheduled task absent, simulator responsive, and simulator topology
unchanged. FlightSim remains launcher PID `7072` with payload child PID `26176`.

## Stop boundary and clean next step

The poison requires `new_reviewed_recovery_task_no_automatic_clear`. Do not
modify the F03 evidence, clear its poison, reuse its attempt identity, or retry
it. Another powered attempt needs a fresh task/session/attempt/root/freeze and
fresh explicit user authorization. Arrange for the operator to be actively
watching the attached console before release so its generated `TRAINING
<challenge>` can be entered within the deadline.

No flight-stack change is indicated by F03's terminal result. The external
launcher should retain the separately pinned venv and base-interpreter identity
checks for any new reviewed task.

To recover disk space before validation, 22 old detached promotion/check
worktrees were removed after proving them clean apart from disposable pytest
caches. Their commits and source states are fully recreatable; the deleted
caches are not. Do not remove the live F03 worktree or either poisoned evidence
root.
