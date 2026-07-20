# VQ2 generated offline scheduler composition

`competition/vq2_wave3_offline_runtime.py` is a deterministic composition for
the already-decoded generated path. It joins the existing latest-frame cursor,
50 Hz scheduler, gate detector and aperture adapter, raw-camera relative
estimator, local IMU provenance/derotation evidence, and Wave 3 adapter. It
terminates at a quarantined `CommandProposalV1` and has no approval or send
path.

The verified FlightSim build-3385 Training premise remains UDP JPEG vision,
`HIGHRES_IMU`, and race status. There is no usable pose or gate-map stream. This
module starts after frame decoding and supplies generated offline mechanism
evidence only; it is not a receiver, recorded replay, simulator, or live runtime.

## State and input boundary

One `VQ2Wave3OfflineRuntime` owns its frame cursor, scheduler, IMU estimator and
bounded attitude history, relative estimator, Wave 3 adapter memory (including
an optional one-use coast lease), proposal sequence, and cumulative trace. The
caller supplies exact immutable timed IMU samples and one latest
`VQ2VisionSnapshot`, safety input, and immutable timing plan per poll.
Calibration, derotation model, camera/IMU identities, covariance, and attitude
timing/orientation uncertainties are required configuration; the runtime does
not infer production values.

`enable_single_tick_correlated_coast` is an exact-boolean, default-off
allowance. Enabling it requires the reviewed exact 20 ms scheduler period. A
tick timing plan can carry either the eight distinct-frame perception times or
the four coast prediction/estimator times, never both. Coast timing is required
exactly when the retained frame and pending lease make that successor tick
eligible; extra or missing work timing is rejected before commit.

Before a due tick commits, the composition validates the snapshot, safety
epoch, camera/IMU identity, timing plan, distinct-frame status, candidate
estimator update or correlated coast, outer adapter transition, and a preview of
the merged trace. Malformed work cannot leave the scheduler active or partially
advance the cursor, estimator, adapter memory, proposal sequence, or trace. An
expired or planned-overrun tick is skipped before perception state changes.

The exported result is not a loose pairing of intent and diagnostics. Its
constructor binds the current due/start/controller/end or skipped lifecycle to
the scheduler lease, proposal, frame, outcome, reason, and retained decision
time. A distinct result also requires its exact six camera facts, staged
perception success or failure, dropped-frame evidence when applicable, and the
selected capture/target IMU occurrences. A coast result additionally binds the
attempted coast timing, consumed proof lease, accepted/rejected disposition,
local correlated-coast transition, and exact eligible source-frame/tick
identity. Result construction is exercised against the preview trace and
preview leases before the scheduler or pipeline state commits.

The snapshot boundary also revalidates its mutable legacy `CameraFrame`
payload: exact class and integer metadata, simulator timestamp binding, exact
`uint8` BGR shape, contiguous layout, and read-only storage. The legacy
floating-point freshness stamp must be finite and nonnegative, but is not
equated to the exact `/1` final-packet occurrence because the receiver samples
those two clocks independently.

## Distinct and repeated frames

A distinct publication runs detection once, uses the contract tracking stage
slots for aperture geometry/observation adaptation, constructs one decision-
time prediction target, selects bounded causal capture/target attitudes, and
pairs the ordinary raw-camera estimator update with standalone derotation
evidence. The corrected bearing is not applied to the filter, guidance, or
proposal.

With the default configuration, re-reading the latest publication still runs no
perception, supplies no correlated update or coast to the Wave 3 adapter, and
returns source-less exact zero. Wave 3C does not relabel a retained frame as a
camera measurement update and does not add stable-frame corrected-ray use.

When explicitly enabled, a fully accepted distinct tick can arm one frozen
`VQ2Wave3CoastLease`. Its source must be an accepted `HEALTHY`, zero-dropout,
active estimator update and a nonzero sourced proposal under the exact accepted
safety lifecycle. The lease binds that update, proposal, safety value, source
tick/deadline, and the immediate successor tick/due/deadline. The prior proposal
must itself lie within the source tick window, and the eligible deadline is
exactly 20 ms after the successor due time. Arming additionally requires the
source scheduler tick to start exactly at its scheduled due. A valid source
proposal that starts late but remains inside its own deadline window is still
accepted, while the scheduler-rebased successor follows the ordinary path and
no unusable nominal-successor lease is created.

Only a repeated source frame on that exact successor tick can attempt the coast.
The runtime reuses the prior raw observation, capture attitude, calibration,
derotation model, and target-uncertainty inputs, then selects a strictly newer,
causal attitude from the same IMU source. The relative estimator produces the
constant-velocity first-dropout successor with the same measurement sequence,
`COASTING` health, no innovation diagnostics, and strictly greater marginal
variance for all six state coordinates. A separate
`VQ2ImuCorrelatedEstimatorCoast` binds the prior update, coast state, and new
standalone derotation evidence. The corrected bearing remains unapplied to the
raw-camera filter state.

Public guidance, controller, and Wave 2 entry points continue to reject dropout
state. The Wave 3 adapter is the sole production call site for the private
capability-checked first-dropout path. Any sourced coast proposal is explicitly
uncertainty-limited with reason `first_observation_dropout_coast`; a coast cannot
open new Gate 0 pitch provenance. Full coast and attitude proof remains in the
local transition/result and is not added to `CommandProposalV1`.

The lease is one-use. An accepted or rejected coast attempt consumes it, as does
a committed scheduler skip or selection of any newer distinct frame. A new
fully accepted healthy distinct update may arm a fresh lease. Missing, stale,
noncausal, mismatched, or malformed coast evidence fails closed to source-less
exact zero; locally detected construction failure is reported as
`imu_correlated_coast_unavailable`, while the adapter retains its more specific
withholding reasons. After consumption, a second repeat follows the ordinary
source-less exact-zero path. Pre-due polls and validation failures do not
partially advance the scheduler, cursor, estimator, adapter memory, proposal
sequence, or trace.

Every consuming result also retains the exact prior source transition as an
independent immutable anchor. Reconstruction requires that transition's pending
lease, correlated update, sourced proposal, and accepted safety to equal the
consumed lease. This prevents a valid but unrelated lease from being substituted
into a skipped, distinct-frame, accepted-coast, or rejected-coast result.

Estimator/tracker ownership is gate-scoped. Gate 0 retains the configured base
tracker identity; a changed authority gate epoch/index derives a bounded
gate-specific identity and resets the candidate estimator. Gate 1 therefore
cannot reuse guidance-retired Gate 0 tracker ownership after accepted gate
credit.

## Timing evidence

The scheduler defaults to the reviewed 20 ms period and rejects any faster
configuration. Deadline misses and planned overruns skip rather than catch up.
Camera timing facts, IMU occurrences, scheduler events, and generated pipeline
stages are merged by host occurrence time and deterministically resequenced
before frozen trace validation. This permits honest interleaving without
appending historical camera facts after later IMU events.

A coast tick contributes only its current prediction and estimator stage facts,
followed by controller and scheduler facts. It emits no new camera, detection,
tracking, or retained-frame-drop fact. Genuine prior `GYRO_SAMPLE` occurrences
remain visible only because the result carries the cumulative trace; no IMU
occurrence is duplicated or rewritten as command causality. Failed coast
construction closes its estimator stage with `ERROR` and the stable unavailable
reason.

Any consumed lease is bound back to exactly one accepted source camera,
perception, prediction, estimator, controller, and scheduler lifecycle plus its
exact IMU occurrence facts. Skipped outcomes are bound to the lease due window,
and a cumulative result rejects future tick identities or any occurrence after
its current terminal tick event.

`GYRO_SAMPLE` records that an IMU sample occurred, including its source-time
token converted from microseconds to nanoseconds. It carries no command ID and
no frame/tick correlation, and does not claim approval, send, actuator
response, gyro response, or command causality. Every offline event has exact
zero queue depth. Send and actuator kinds are rejected even when injected into
an otherwise valid trace. The generated camera timings and caller timing plan
are mechanism evidence, not receiver measurements or recorded replay.

## Promotion boundary

The module directly imports no runner, UDP receiver, MAVLink, transport,
supervisor, simulator, or powered surface and constructs no approved command,
send, or actuator event. Runtime/shadow promotion still requires production
per-sample arrival capture, calibrated timing and extrinsics, a
supervisor-verifiable provenance envelope, approved replay, and separate
review. Operational T0-T4 and every powered stage retain their existing
external prerequisites and authorization boundary.

## Baseline and Wave 3C integration record

Behavior commit `8eab146e3a9a7a1a1b28070d3e0234adff900595`, reconciliation
merge `7904fbadbc4b220b81afb846a69b15a7b30ef4bb`, and promotion/trust
commit `28b7d782404d6b825cebae3b65a8443d756be234` are integrated on main.
The focused runtime suite passed 38 tests; the coupled matrix passed 181; an
independent compatibility/adversarial matrix passed 199 and explicitly cleared
the tranche. Canonical and isolated-manifest VQ2 runs passed exactly 910 tests,
fast and unit each passed 2,005 with 20 skips and 42 deselections, and the full
non-live boundary passed 2,046 with 21 skips. The 127-file trusted manifest has
semantic identity
`cdd0db402b6f1c8bb0c90c1b8d445ca64741d3bfc3aa03a78c3fe4d73c8dcce2`
and file SHA-256
`e270a194031d463accfb50b28bd3296eb672004d1c41241fab3cb368bab1640a`.

Those commits, counts, and manifest identities remain the historical Wave 3B
baseline. Wave 3C is now integrated by behavioral commit
`84674fd8c7379b327e25725010ca58a57f4fd910` and promotion/trust commit
`168220ba7060d07743335d0e9c56bcd2d05d669d`.

Wave 3C focused runtime, Wave 3 adapter, and relative-estimator suites pass 74,
95, and 32 tests; the frozen six-module affected matrix passes 477. Independent
final contract review passes a 201-test deep matrix and explicitly clears the
coast envelope, lease, memory, transition, reconstruction, scheduler edge, and
cumulative-trace scope. Canonical and isolated-manifest VQ2 runs pass exactly
1,019 tests, including the post-merge main run. Fast and unit each pass 2,114
with 20 skips and 42 deselections; full non-live passes 2,155 with 21 skips.

The reviewed manifest remains 127 files with semantic identity
`f9118fad5fdbdd8e5e355cf0e153492525b853b9b7c32239ab4d2d81f6d63b2b`,
file SHA-256
`29b306e41a6954552ef7693f0e0c3d853cc4b60aeedfb59f6a2c9592ece9d8c6`,
and policy SHA-256
`29eb2dcd627a8f5dbbea4bf88c249a87ca741ca5c9d743c0c646404f40e8748e`.
Exactly six changed test hashes plus the policy hash were replaced; no trusted
file was added or removed. Main fast-forwarded to the promotion commit,
post-merge VQ2 passed all 1,019 tests, and tracked status was empty before this
documentation closeout.

No simulator, network, preflight, reset, arm/disarm, target, transport, shadow,
or powered action was used for the baseline or integrated Wave 3C work described
here. No live, recorded-replay, production timing/extrinsics, measured command
response, or supervisor-verifiable provenance claim follows from this document.
