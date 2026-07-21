# Package 2 launcher session-admission recovery

- Task ID: `vq2-package2-launcher-session-recovery`.
- Parent: `vq2-package2-import-environment-recovery`.
- State: `offline implementation and promotion only`.
- Starting main commit:
  `0970f4b74c8529fb6fcd72d37c765d7d18025c3a`.
- Simulator target: FlightSim build 3385, Training mode.
- Simulator access: `none`. This task records no powered authority. A future
  bounded calibration attempt requires a new explicit user checkpoint after
  implementation, promotion, fresh publication/freeze, and independent review.
- Contract date: `2026-07-21`.

## Immutable predecessor state

F01-A01 is consumed, terminal-invalid, and poisoned. It must not be retried,
renamed, cleared, repaired, or used as authority for another attempt. Its
wrapper failed before simulator launch because the trusted System32
`query.exe session` emitted a valid active-console table but returned exit code
`1`; the frozen launcher rejected every nonzero code before parsing the table.

The wrapper then correctly treated the launcher call as ambiguous, could not
construct the required prechild/postchild topology proof, retained the lease,
and published poison plus attempt-invalid. Current OS absence does not
retroactively create the missing cleanup evidence. The exact predecessor hashes
are recorded in the parent task and remain private evidence outside Git.

## Narrow objective

Recover only the verified Windows session-admission boundary:

1. isolate active-session selection in a pure PowerShell function;
2. admit only `query.exe` statuses `0` and the empirically observed `1`;
3. for either admitted status, require bounded output, exactly one expected
   English session-table header, and at least one strictly parsed `Active` row;
4. preserve selection priority: current session, then console, then lowest
   numeric session ID;
5. preserve the global launch mutex, preexisting-simulator refusal, exact
   simulator path, non-overwriting scheduled-task creation, interactive/highest
   task flags, bounded launch wait, and mandatory task deletion; and
6. add non-live tests that execute only the exact selector function extracted
   from the production script's PowerShell AST.

Status `1` by itself is never success. Missing/duplicate header, no active row,
oversized output, any other status, parse failure, or changed selection remains
fail-closed. This task does not weaken simulator topology, port, transport,
watchdog, command, cleanup, lease, or evidence contracts.

## Verification and promotion

Run the focused Windows tooling tests after every edit. For an accepted
candidate, use a clean external worktree whose candidate root does not contain
the venv, set `AIGP_PYTHON` to the pinned development interpreter, and run:

```powershell
.\scripts\dev.cmd test-target tests/test_windows_tooling.py
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
.\scripts\dev.cmd test-vq2
.\scripts\dev.cmd test-full-non-live
```

Also execute the selector alone against the host's passive `query.exe session`
output and record only its exit status and selected session facts. Do not run
`launch-sim`, a powered wrapper, a live marker, or any generic powered command.

## Successor boundary

Promotion and commit do not authorize simulator use. Before another powered
attempt, a successor must:

- use a fresh task/session/attempt identity, private evidence root, task name,
  capture authorization, plan, inventories, and freeze;
- bind the immutable F01 invalid/poison and this correction as provenance;
- independently review the exact candidate, environment, launcher, wrapper,
  bounded calibration plan, and cleanup contract;
- prove current wrapper/simulator/task/port absence without claiming that it
  repairs F01 cleanup; and
- stop for a new explicit powered checkpoint.

The successor should separately review the conservative early-launch failure
lane: launcher stdout/stderr are currently discarded, and a failed launcher
call cannot release the lease without contemporaneous topology proof. That gap
does not affect nominal session selection and must not be silently weakened or
treated as repaired by this narrow change.
