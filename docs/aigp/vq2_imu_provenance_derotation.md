# VQ2 offline IMU provenance and rotation-only derotation

## Scope and authority boundary

This tranche adds a local envelope around HIGHRES_IMU-derived attitude and a
pure rotation-only camera-ray correction. It does not modify the frozen VQ2
`/1` wire contracts and grants no supervisor, transport, reset, arm, cleanup,
simulator, or powered authority.

The existing `RelativeGateStateV1` and `CommandProposalV1` bind camera/state
lineage but have no fields for IMU, attitude, pitch-basis, calibration, or
derotation identity. New local types can therefore prove deterministic offline
composition, but a bare `/1` proposal is not supervisor-verifiable evidence of
those inputs. Runtime promotion requires a separately reviewed `/2` envelope
or a supervisor-owned out-of-band provenance registry.

## Two time domains remain separate

Each IMU sample retains both:

- its HIGHRES source timestamp in microseconds, used only for source ordering
  and attitude-estimator integration; and
- its per-sample receive time on an identified host-monotonic clock, used for
  freshness, causality, and correlation with camera/state/controller events.

The source timestamp is never converted to or subtracted from host-monotonic
time. A drained batch cannot be relabeled with one shared arrival time. Exact
session/reset, clock, stream/generation, sample sequence, source time, and host
receipt lineage must advance consistently or fail without mutating accepted
state.

Calibration-incomplete, timestamp-gap, unhealthy, future, stale, or
excessively uncertain estimates are ineligible for the outer controller
composition. Reusing an exact immutable sample may be allowed where explicitly
bounded; changing any field under an existing identity is relabeling and is
rejected.

## Rotation-only camera-ray correction

The derotator treats normalized image bearing `(u, v)` as the FRD-like camera
ray `(forward, right, down) = (1, u, v)`. It rotates that ray from the capture
attitude to a bounded target attitude through one explicit camera-to-body
quaternion. There is no default calibration and every calibration carries a
model identity.

The result is rejected when epochs, clocks, streams, source chronology,
frame/candidate/authority, or target timing disagree; when either attitude is
not healthy and calibrated; when the correction interval or extrapolation is
outside the reviewed local bound; when the corrected ray is not forward; or
when normalized output leaves the `/1` numerical envelope. The operation
models camera rotation only. It makes no translation, metric distance, planar
pose, or calibrated production-extrinsics claim, and its uncertainty cannot be
tighter than the input uncertainty.

## Outer adapter behavior

`competition/vq2_wave3_imu_adapter.py` is an offline-only provenance gate around
the unchanged Wave 2 guidance/controller adapter. It accepts an ordinary raw-
camera estimator update paired with exact IMU/derotation evidence for the same
observation and prediction target. The corrected bearing remains standalone
evidence: it is not injected into the capture-time Kalman filter, guidance, or
the frozen `/1` state. Injecting a target-attitude bearing there would mix time
bases and predict the correction twice.

The controller attitude has separate propagated evidence targeted exactly to
the proposal time. Constant-body-rate extrapolation is capped at 20 ms;
extrapolation plus host-time uncertainty is capped at 50 ms; and the combined
orientation, host-time/rate, and propagation uncertainty is capped at five
degrees. These are local conservative eligibility bounds, not measured runtime
latency or plant-identification claims.

For Gate 0 approach, a second propagated-attitude record is targeted exactly to
the phase-start time. Its source must have arrived no later than phase entry and
must pass the same extrapolation, effective-age, and angular-uncertainty bounds.
The initial pitch is derived from that effective quaternion. The latch retains
the complete phase-entry evidence and the same session/reset/gate/clock/phase
identity as the inner pitch latch. It cannot be supplied late, changed, or
relabeled; an entry that starts with an empty latch stays empty for that phase.
The latch clears outside that exact Gate 0 phase, and Gate 1 never retains it.

Every provenance, correlation, freshness, uncertainty, or pitch-source failure
yields a source-less exact-zero outer proposal. An invalid supplied update is
quarantined before it can advance Wave 2 visual ownership. Requested safety is
retained only after the unchanged inner transition accepts it; a rejected
safety transition also preserves prior outer attitude lineage. Caller-threaded
pure memory remains a trust boundary; neither inner nor outer Python values
authenticate an untrusted owner.

## Evidence boundary

Tests use exact immutable synthetic samples, generated bearings, explicit test
calibrations, and local temporary state. They can establish deterministic
lineage rejection, quaternion/ray signs, uncertainty monotonicity, pitch
derivation, and exact-zero composition. They are not recorded replay, ingress
arrival evidence, calibrated camera/IMU timing or extrinsics, measured
actuator/gyro delay, official-simulator behavior, shadow selection, passage, or
powered flight evidence.

Promoting the corrected ray into state estimation is a separate design task.
It needs a stable-frame or explicitly time-aligned measurement model; the
current target-time correction cannot be treated as a capture-time Kalman
measurement.
