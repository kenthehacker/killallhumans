# Fast VQ2 flight cycle

This is the default loop for rapid FlightSim build-3385 Training iteration:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_fast_cycle.py tests/test_aigp_vq2_runner.py
.\scripts\dev.cmd test-vq2
.\scripts\dev.cmd flight-cycle
```

`flight-cycle` defaults to `calibration-excite`; a different bounded stage may
be supplied as its sole argument. It is a dedicated powered command, not a
generic test, and it never prompts for interactive confirmation. Invocation
asserts that an existing direct user instruction authorizes that powered stage
or a continuing scoped iteration; the command itself is not proof of
authorization.

## What happens before flight

The command does only the following pre-contact work:

1. Reserves a unique private evidence directory outside Git.
2. Takes the canonical nonblocking host-wide FlightSim live lease, shared with
   legacy live workflows; a busy lease leaves only that empty reservation.
3. Writes one compact manifest containing the stage, build/mode target,
   commit/tree plus dirty-diff identity, hashes of the bounded runtime source
   set, target/plan identities, Python identity, and output paths.
4. Rechecks the bounded runtime source hashes and starts the runner in the
   launcher's existing `-E -s -B` Python isolation.

It does not take a screenshot, read the console, wait for a challenge or manual
approval file, run a separate passive preflight, require a clean/detached
worktree, derive duplicate data, inventory the full repository/environment or
import graph, create a freeze bundle, or synchronously run post-flight analysis.
Training is recorded honestly as configured session state because build 3385
has no machine-readable mode field.

Dirty development candidates are allowed so controller changes can be tested
quickly. The manifest records both Git status/diff hashes and exact hashes of
the runtime source files used by the fast path. Promotion remains a separate
clean exact-commit boundary.

## What is not shortened

The powered runner still:

- obtains advancing race and IMU baselines and proves both clocks rolled back
  after `SIM_RESET`;
- restarts vision only inside the proved epoch;
- normalizes disarmed, witnesses countdown, and waits until GO + 150 ms with
  fresh camera/target, IMU, race, heartbeat, actuator, and estimator state;
- proves at least 20 decoded frames/s over the post-reset countdown and exact
  640x360 dimensions before arm, then rechecks dimensions on calibration ticks;
- confirms arm/disarm only on newer heartbeats;
- sends no faster than 50 Hz, never catches up missed ticks, keeps yaw rate at
  zero, and retains the command envelope;
- aborts on stale/nonadvancing streams, target/corridor loss, estimator or
  attitude failure, collision, gate change, non-finite/out-of-bounds command,
  or missed waveform deadline;
- on every runner exit path, latches command production, sends the safe stop
  when applicable, disarms, resets, proves the clean epoch, and marks failed
  cleanup as a failed stage.

The calibration waveform is the existing reviewed 245-slot, 4.9-second plan.
Its values and safety limits are code-owned and cannot be changed through the
manifest.

## Evidence and failures

By default artifacts are written to
`%USERPROFILE%\aigp-evidence\fast-flight-cycles\<run-id>`:

- `run-manifest.json`
- `live-lease.json`
- `session.jsonl.gz`
- `result.json`

There are no diagnostic PNGs or full replay bundle on this fast path. A
pre-contact manifest/source failure writes a failed result when possible and
may be fixed and retried without creating a new task/F-number or poisoning an
unused attempt. A cleanup failure remains terminal for that actual flight.

The Package 2 F00-F04 freeze/challenge/publication documents and wrapper remain
historical evidence. They are not prerequisites for `flight-cycle`.

The fast path intentionally uses the established same-process bounded runner.
A hard interpreter/process termination or a native transport call that never
returns can bypass its `finally` cleanup; use the legacy promotion wrapper only
when independent parent-death/hang containment is required.
