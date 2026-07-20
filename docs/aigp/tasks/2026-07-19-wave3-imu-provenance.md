# Wave 3 offline IMU provenance and derotation task manifest

- Task ID: `vq2-wave3-imu-provenance-derotation`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `post_merge_verified`
- Objective: add an exact local provenance envelope for HIGHRES_IMU-derived
  attitude, a bounded pure camera-ray derotation primitive, and an outer
  offline-only adapter that derives the Gate 0 pitch basis from a correlated
  attitude sample.
- Starting main commit:
  `a5c11ab9924b1f250948e9ebdee459427489c924`.
- Branch: `wave3-imu-provenance`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-wave3-imu-provenance`.
- Owner: `/root`; module implementation is split across independently scoped
  collaborators and reconciled here.
- Heartbeat date: `2026-07-19`.
- Simulator access: `none`.
- Frozen interface: the checked-in `/1` contracts remain byte-for-byte
  unchanged. The local envelope must not relabel camera timing as
  `IMU_PROPAGATED`, add fields to `RelativeGateStateV1` or
  `CommandProposalV1`, or imply supervisor-verifiable attitude provenance.
- Authority boundary: every new type and adapter result is offline evidence
  only. No supervisor approval, projection, runner, scheduler, MAVLink,
  transport, reset, arm/disarm, cleanup, simulator, or system-identification
  dependency is allowed.
- Timing rule: HIGHRES_IMU source microseconds are opaque ordering and estimator
  integration input. They are never subtracted from host-monotonic time.
  Per-sample host receipt/estimate times carry freshness and correlation.
- Source rule: session/reset, host clock, IMU stream/generation, sample
  sequence, source time, and host receipt time are exact and monotonic. A
  sample cannot be relabeled across any source or epoch.
- Health rule: calibration-incomplete, gap, unhealthy, future, stale, or
  excessively uncertain attitudes cannot produce a sourced proposal.
- Derotation rule: explicit camera-to-body calibration/model identity is
  mandatory; rotation-only correction makes no translation or metric-pose
  claim, must keep the camera ray forward and normalized output bounded, and
  cannot reduce input uncertainty.
- Gate 0 pitch rule: derive pitch from the accepted timestamped quaternion at
  the exact phase-start host time, retain the complete propagated evidence and
  uncertainty with the latch, latch once, reject later fill, change, or relabel
  attempts, and clear it on phase/gate/reset exit. Gate 1 never retains a Gate
  0 pitch latch.
- Filter-basis rule: the rotation-corrected target bearing remains standalone
  evidence. The ordinary raw-camera estimator output must remain bit-for-bit
  unchanged; a target-basis bearing must not enter its capture-time posterior,
  guidance, or a `/1` proposal.
- Controller-attitude rule: separately propagate the correlated timestamped
  attitude to the exact proposal time, with 20 ms extrapolation, 50 ms
  effective-age, and five-degree angular-uncertainty hard caps.
- Expected implementation ownership:
  - `estimation/vq2_imu_provenance.py` and direct tests: exact IMU/attitude
    lineage and estimator wrapper;
  - `estimation/vq2_imu_derotation.py` and direct tests: bounded rotation-only
    geometry/evidence;
  - an integration-owned outer adapter and tests: provenance eligibility,
    pitch derivation/latching, exact-zero failures, and delegation to the
    unchanged Wave 2 adapter.
- Required evidence: direct and adversarial targets, combined affected tests,
  canonical exact `test-vq2`, `test-fast`, isolated trusted-manifest review if
  discovery changes, clean committed candidate, independent review, and
  post-merge `test-vq2` on main.
- Promotion claim: deterministic generated/offline evidence only. It is not a
  recorded replay, calibrated production camera/IMU model, measured latency or
  plant response, official-simulator result, shadow result, passage result, or
  powered evidence.
- Known promotion blocker: frozen `CommandProposalV1` cannot carry attitude,
  pitch-basis, or derotation identity to the supervisor. Runtime promotion
  therefore requires a separately reviewed `/2` envelope or a
  supervisor-owned out-of-band provenance registry, plus per-sample host
  arrival capture in production ingress and calibrated camera/IMU timing and
  extrinsics.
- State-estimation blocker: applying the corrected ray requires a reviewed
  stable-frame or explicitly time-aligned filter. The current capture-time
  camera estimator cannot ingest a target-attitude bearing without mixing
  coordinate/time bases and potentially double-predicting it.

## Integration and evidence record

- Behavioral implementation:
  `f53718da892c4ab5aecc567a61249b21a8cb6ffa`.
- Main reconciliation merge:
  `e3f386d460d012c7b9710ae440c2ac405447f1f3`.
- Promotion-policy and trusted-manifest closeout:
  `ecaa794aeaed87a169b7b87b284d1440f1768a28`.
- Affected IMU-adapter, provenance, derotation, and relative-estimator tests:
  143 passed.
- Canonical exact `test-vq2`: 872 passed. Main fast-forwarded to
  `ecaa794aeaed87a169b7b87b284d1440f1768a28`; post-merge `test-vq2` also
  passed all 872 tests and tracked Git status was empty.
- `test-fast` and `test-unit`: 1,967 passed, 20 skipped, and 42 deselected
  each.
- Promotion-boundary `test-full-non-live`: 2,008 passed and 21 skipped.
  Skipped optional coverage is not positive evidence.
- Strict trusted manifest: 126 files, semantic identity
  `f074019f30858b9fcc5fb06a90a8df7cf57770e84791893ddbaa082861eca5eb`,
  file SHA-256
  `ba07b6ea73b5fc88f99e6c8824ea4d7039c956391de2c5730e6716af76cad9b1`,
  and exact VQ2 policy file SHA-256
  `4352163c57b06f8bb12a7b7750c8a279d76b0c45d933dacf3d5149238ee970ef`.
- Independent lifecycle and adversarial review cleared the integrated
  candidate with no remaining tranche-local blocker.
- Simulator access remained `none`. No FlightSim launch or connection,
  preflight, external network access, reset, arm/disarm, target, transport,
  shadow selection, or powered command occurred.

## Residual boundary

The promotion and state-estimation blockers above remain open. In particular,
the corrected ray is standalone rotation-only evidence and is not applied to
the raw-camera estimator, guidance, or a `/1` proposal. Full production
arrival correlation, calibrated camera/IMU timing and extrinsics, measured
command/actuator/gyro latency, supervisor-verifiable provenance, recorded
replay, shadow/runtime evidence, and all powered evidence remain future work.
