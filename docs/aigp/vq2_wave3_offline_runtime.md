# VQ2 generated offline scheduler composition

`competition/vq2_wave3_offline_runtime.py` is a deterministic composition for
the already-decoded generated path. It joins the existing latest-frame cursor,
50 Hz scheduler, gate detector and aperture adapter, raw-camera relative
estimator, local IMU provenance/derotation evidence, and Wave 3 adapter. It
terminates at a quarantined `CommandProposalV1` and has no approval or send
path.

## State and input boundary

One `VQ2Wave3OfflineRuntime` owns its frame cursor, scheduler, IMU estimator and
bounded attitude history, relative estimator, Wave 3 adapter memory, proposal
sequence, and cumulative trace. The caller supplies exact immutable timed IMU
samples and one latest `VQ2VisionSnapshot`, safety input, and immutable timing
plan per poll. Calibration, derotation model, camera/IMU identities, covariance,
and attitude timing/orientation uncertainties are required configuration; the
runtime does not infer production values.

Before a due tick commits, the composition validates the snapshot, safety
epoch, camera/IMU identity, timing plan, distinct-frame status, candidate
estimator update, outer adapter transition, and a preview of the merged trace.
Malformed work cannot leave the scheduler active or partially advance the
cursor, estimator, adapter memory, proposal sequence, or trace. An expired or
planned-overrun tick is skipped before perception state changes.

The exported result is not a loose pairing of intent and diagnostics. Its
constructor binds the current due/start/controller/end or skipped lifecycle to
the lease, proposal, frame, outcome, reason, and retained decision time. A
distinct result also requires its exact six camera facts, staged perception
success or failure, dropped-frame evidence when applicable, and the selected
capture/target IMU occurrences. Result construction is exercised against the
preview trace and preview lease before the scheduler or pipeline state commits.

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

Re-reading the latest publication runs no perception and supplies no correlated
update to the Wave 3 adapter. The result is therefore source-less exact zero.
There is deliberately no correlated coast or stable-frame relabeling in this
tranche.

## Timing evidence

The scheduler defaults to the reviewed 20 ms period and rejects any faster
configuration. Deadline misses and planned overruns skip rather than catch up.
Camera timing facts, IMU occurrences, scheduler events, and generated pipeline
stages are merged by host occurrence time and deterministically resequenced
before frozen trace validation. This permits honest interleaving without
appending historical camera facts after later IMU events.

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
