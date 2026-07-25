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

The available stages are `sign-id`, `hover`, `gate0`, `gate0-observe`,
`gate1-recenter`, `visual-shadow`, `visual-align`, `visual-course`, and
`calibration-excite`. `visual-course` is the generic rolling-graph lifecycle
that continues until authoritative race status reports `race_finished`;
`full-lap` remains quarantined.

`sign-id` is a yaw-only below-hover epoch at fixed `0.235` thrust. It records a
`0.24 s` neutral baseline, applies isolated `+0.08` and `-0.08 rad/s` yaw
pulses for 11 slots (`0.22 s`) each with a 6-slot (`0.12 s`) zero-rate
reversal dwell, then enters zero rate for 5 slots (`0.10 s`) and cleanup.
The exact 45-slot plan ends at `0.90 s` and has a `1.00 s` hard expiry. Fresh
frames observed at a segment boundary
are attributed to the command that was still active, using its exact
wire-release timestamp.

The stage requires exact outbound wire receipts, at least four strictly newer
gyro samples and four distinct fresh primary-target frames per measured
window, and opposite paired gyro and differential image-slope effects with
consistent gains. Roll and pitch remain exact zero. Calibration collection
stays inside a `0.05 rad` attitude safety bound and a `0.50 rad/s` measured
yaw-rate bound; this collection limit does not itself grant production
authority. The existing collision watchdog remains unchanged. The result
records controller-to-body and controller-to-image signs plus paired gyro and
image-rate gains. Only an independently reviewed, fixed-hash yaw profile may
grant bounded authority to `visual-course`.

`gate1-recenter` is a user-authorized bounded no-passage diagnostic. Its
horizontal pixel-rate gain remains exactly zero pending M2 recorded-replay and
tracker-isolation acceptance; the current diagnostic combines horizontal
position error with a fixed bounded pitch brake. Its single structural
hypothesis enables the existing bounded Gate 0 cyan-course-line preturn while
enabling its bounded late exit counterroll to align the Gate 1 handoff with the
frozen recenter direction. Applied preturn and exit-counterroll objectives are
recorded in the trace. It preserves exact `0.275` thrust with zero attitude
rates during the visually armed Gate 0 crossing confirmation and authoritative
Gate 1 acquisition so the motors do not cut during the continuation handoff.
Once Gate 1 is authoritative, the accepted and raw no-passage geometry guards
apply before every such powered observation setpoint. It stops after 0.60
seconds, on any gate-index change, or before a primary target reaches 160
pixels wide or 23,040 square pixels; the existing raw large-geometry contact
guard is enforced independently. Stage success still requires cleanup
confirmation.

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
- sends no faster than 50 Hz, never catches up missed ticks, keeps
  uncalibrated axes at zero, and retains the command envelope;
- aborts on stale/nonadvancing streams, target/corridor loss, estimator or
  attitude failure, unsafe collision, gate change, non-finite/out-of-bounds command,
  or missed waveform deadline;
- on every runner exit path, latches command production, sends the safe stop
  when applicable, disarms, resets, proves the clean epoch, and marks failed
  cleanup as a failed stage.

The compact yaw-calibration waveform is a balanced 45-slot, 0.90-second
system-ID burst with a 1.00-second hard expiry. Its values and safety limits
are code-owned and cannot be changed through the manifest. Both `sign-id` and
`calibration-excite` bind this exact plan identity into the run manifest.

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
