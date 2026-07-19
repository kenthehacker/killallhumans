# VQ2 Wave 2 offline guidance/controller adapter

## Scope

`competition/vq2_wave2_adapter.py` is a pure, immutable offline composition of
the mapless guidance state machine and predictive controller. It consumes no
transport, runtime scheduler, supervisor approval, system-identification
result, simulator connection, or powered authority. Its output is only a
`CommandProposalV1`; the existing safety supervisor remains the sole component
that may approve any proposal for transport.

The adapter is intentionally not runtime-ready. `ControllerAttitudeInput` and
the Gate 0 initial-pitch basis have no timestamp, clock identity, or source
correlation, and the frozen proposal cannot bind either input. A reviewed IMU
provenance and derotation seam is required before shadow or runtime wiring.

## State ownership

The caller threads `VQ2Wave2AdapterMemory` through every invocation. The
adapter, not an external decision producer, calls `step_vq2_guidance` with the
previous accepted `VQ2GuidanceMemory`. It does not accept a preconstructed
`VQ2GuidanceDecision` or `VQ2GuidanceTransition`.

This makes the prior phase start part of the adapter's immutable state. A
same-phase update that renews the start is rejected even if the current safety
input, fresh state, and tick all repeat the renewed value. The rejected call
keeps the prior memory and emits source-less exact zero. A future runtime
caller must keep this memory inside the reviewed supervisor boundary; a pure
Python value cannot authenticate an untrusted owner by itself.

Gate 0's pitch basis is latched exactly once at accepted `APPROACH` entry and
bound to session, reset epoch, gate epoch/index, host clock, and phase start.
Changing it within the phase fails closed. If it is absent at entry, supplying
it later cannot enable motion. The latch is cleared outside Gate 0 approach.

## Exhaustive controller mapping

Only these exact tuples can call the controller:

| Gate index | Race | Guidance phase | Objective | Controller mode |
|---:|---|---|---|---|
| 0 | `UNDERWAY` | `APPROACH` | `APPROACH_ACTIVE_GATE` | `GATE0_APPROACH` |
| 1 | `UNDERWAY` | `ALIGN` | `RECENTER_ACTIVE_GATE` | `GATE1_RECENTER` |

Every other tuple is source-less exact zero. This includes Gate 0 alignment,
Gate 1 approach, every acquire/confirmation/reacquire or terminal hold, every
gate index above one, and every `COMMIT` objective even when guidance considers
that local planning objective eligible. Unsupported phases still advance valid
guidance memory so legal confirmation, credit, and reacquisition transitions
remain reachable; they simply cannot reach control.

## Exact binding before control

For a supported tuple, the adapter requires:

- the decision's authority, phase, race state, evaluation clock/time, and phase
  start to equal the accepted guidance memory;
- the decision source and retained active source to equal all ten correlation
  fields reconstructed from the same immutable `RelativeGateStateV1` passed to
  the controller;
- exact authority equality across decision, guidance memory, state, and tick;
- one exact host clock across authority, decision, state/source, and tick;
- tick phase-start and phase-evaluation watermarks equal to the current
  accepted values, not merely older lower bounds;
- tick state-decision and state-sequence watermarks equal to the supplied
  state; and
- an active source, exact centered target, permitted objective, and no
  withholding reason.

The controller then applies its own health, freshness, prediction-delay,
covariance, envelope, dwell, and saturation checks. A sourced result is
validated again with `validate_command_proposal_source`. A controller
withholding result remains source-less exact zero and retains its controller
diagnostics, including Gate 1 corridor-unconfirmed and timeout reasons.

The adapter constructs its own exact-zero proposal for guidance or composition
failures. All ten source fields are `None`, rates and thrust are exact zero,
and only tick identity/timing plus the tick's expected authority are copied.
This zero is deliberately tick-scoped: when tick authority or clock is the
failed check, the proposal is not relabeled with accepted decision authority.
Only a sourced proposal is required to share decision authority.

## Offline evidence boundary

The direct adapter suite covers legal Gate 0 and credited Gate 1 proposals,
coordinated phase-start renewal, exact tick watermarks/clock/authority, pitch
latching, unsupported but guidance-permitted phases, commit isolation, Gate 1
uncertainty/corridor/timeout behavior, shadow non-promotion, deterministic
state transitions, and mid-course bootstrap rejection.

One synthetic cross-layer test carries an immutable generated, already-decoded
timed frame through latest-frame selection, red-gate detection, aperture
fitting, the relative estimator, guidance, the adapter, and exact
observation-to-state and proposal-to-state source validation. Multiple distinct
frames are required before the estimator's rate uncertainty qualifies for Gate
0 approach. A bbox-only observation remains withheld by the estimator. These
are deterministic generated image-space tests, not JPEG receiver/reassembly,
recorded replay, measured plant response, official-simulator evidence, passage
evidence, or powered flight evidence.
