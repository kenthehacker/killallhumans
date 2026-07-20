# Wave 3C one-tick IMU-correlated coast lease

- Task ID: `vq2-wave3c-correlated-coast`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `post_merge_verified`
- Objective: permit exactly one proof-bound first-dropout prediction on the
  immediate repeated-frame scheduler tick while preserving the existing
  source-less exact-zero default and quarantined proposal boundary.

## Ownership

- `estimation/vq2_relative_estimator.py`
- `planning/vq2_guidance.py`
- `competition/vq2_controller.py`
- `competition/vq2_wave2_adapter.py`
- `competition/vq2_wave3_imu_adapter.py`
- `competition/vq2_wave3_offline_runtime.py`
- the six directly corresponding test modules
- `docs/aigp/vq2_wave3_offline_runtime.md`
- this task record

No frozen `/1` contract, runner, receiver, supervisor, approval, command
projection, transport, simulator, reset, arm, cleanup, promotion scheduler, or
powered surface is owned by this tranche. Promotion metadata and shared
handoffs remain integration-owner work after independent review.

## Required invariants

- The ordinary estimator `coast()` remains general. Wave 3C adds a separate
  exact local `VQ2ImuCorrelatedEstimatorCoast` envelope; it never relabels a
  coast as a camera measurement update.
- The envelope binds an accepted `HEALTHY` zero-dropout active update, exact
  observation/frame/candidate/authority/tracker/role, unchanged capture input,
  calibration/model and target uncertainty model, and a strictly newer
  same-source causal target attitude.
- The coast state is the constant-velocity successor with the same measurement
  update, exact state-sequence increment, strictly growing six marginal
  variances, `dropout_count == 1`, `COASTING`, `observation_dropout`, and no
  innovation diagnostics. Corrected bearing remains standalone evidence and is
  not applied to the raw-camera state.
- Public guidance, controller, and Wave 2 entry points continue to reject all
  dropout state. Only private capability-checked helpers may admit the exact
  first-dropout profile, and the Wave 3 adapter is their sole call site.
- Every sourced coast proposal is explicitly uncertainty-limited with reason
  `first_observation_dropout_coast`.
- A frozen `VQ2Wave3CoastLease` retains the prior correlated update, sourced
  proposal, accepted safety lifecycle, source tick/deadline, and exact
  successor tick/due/deadline. It is armed only by an opt-in, fully accepted,
  healthy, sourced distinct update whose scheduler tick starts exactly at its
  scheduled due time. A valid late source proposal remains accepted but cannot
  arm an already-unusable nominal-successor lease.
- Opt-in requires the reviewed exact 20 ms scheduler period. The coast tick is
  exactly the immediate successor, occurs inside its due/deadline window,
  retains authority/phase/race/phase-start and source-frame identity, and uses
  a strictly newer causal IMU attitude.
- A committed skip, any new distinct-frame selection or failure, invalid or
  unavailable coast evidence, lifecycle mismatch, success, or attempted reuse
  consumes the pending lease. A malformed/pre-due call does not partially
  mutate runtime state. The second repeat remains source-less exact zero.
- Gate-scoped estimator/tracker identity rotates after accepted gate credit so
  Gate 1 cannot reuse guidance-retired Gate 0 ownership.
- Coast work has its own exact prediction/estimator timing plan. Its trace
  emits only current-tick prediction and estimator stages plus genuine prior
  IMU occurrence facts; it never re-emits camera/detection/tracking or a
  retained-frame drop.
- The exported result retains the attempted coast timing, consumed lease,
  exact prior source transition, and reviewed disposition. Preview construction
  reconstructs and validates trace, state, proposal, lease, source transition,
  disposition, and terminal snapshot before scheduler/estimator/adapter commit.
- No supervisor approval, send, actuator response, transport, network,
  simulator, reset, arm/disarm, cleanup, or powered value is constructed.

## Evidence boundary

This is generated, already-decoded, offline mechanism evidence. It is not UDP
receiver/reassembly or recorded-replay evidence, production per-sample arrival
capture, calibrated camera/IMU timing or extrinsics, measured
command/actuator/gyro response, a supervisor-verifiable provenance envelope,
shadow/runtime acceptance, or powered FlightSim evidence. The frozen proposal
still cannot carry the full coast and attitude proof.

Stable-frame corrected-ray application remains separate. It still requires a
reviewed reference lifecycle, inverse output transform, and full
bearing/rate/log-scale covariance transform before filter use.

## Closeout record

- Direct focused tests cover estimator-envelope forgeries, public
  lower-layer rejection and private-call ownership, Gate 0 and full credited
  Gate 1 success, one-use/skip/new-frame consumption, lifecycle and IMU
  failures, exact trace/result binding, transaction retry, and recovery.
- The frozen six-module affected matrix passes `477` tests. Independent final
  contract review passes a `201`-test deep matrix and explicitly clears the
  coast envelope, lease, memory, transition, result, reconstruction, scheduler,
  and cumulative-trace scope with no remaining promotion blocker.
- Behavioral implementation:
  `84674fd8c7379b327e25725010ca58a57f4fd910`.
- Promotion-policy and trusted-manifest closeout:
  `168220ba7060d07743335d0e9c56bcd2d05d669d`.
- Focused runtime, Wave 3 adapter, and relative-estimator suites pass 74, 95,
  and 32 tests respectively. The six-module affected matrix passes 477.
- Canonical and isolated-manifest VQ2 policy runs pass exactly 1,019 tests,
  including the post-merge main run.
- `test-fast` and `test-unit` each pass 2,114 tests with 20 skips and 42
  deselections. Promotion-boundary `test-full-non-live` passes 2,155 tests
  with 21 skips; skipped optional coverage is not positive evidence.
- The strict trusted manifest remains 127 files, with semantic identity
  `f9118fad5fdbdd8e5e355cf0e153492525b853b9b7c32239ab4d2d81f6d63b2b`,
  file SHA-256
  `29b306e41a6954552ef7693f0e0c3d853cc4b60aeedfb59f6a2c9592ece9d8c6`,
  and exact policy SHA-256
  `29eb2dcd627a8f5dbbea4bf88c249a87ca741ca5c9d743c0c646404f40e8748e`.
  The trust review replaced exactly the six changed test hashes plus the
  policy hash, with no file added or removed and no trust-root expansion.
- Main fast-forwarded to the promotion commit. Post-merge `test-vq2` passed all
  1,019 tests, and tracked Git status was empty before this documentation
  closeout.
- Simulator access remained `none`. No FlightSim launch or connection,
  preflight, external network access, reset, arm/disarm, target, transport,
  shadow selection, or powered command occurred.

Recorded replay, calibrated production timing/extrinsics, a
supervisor-verifiable proof carrier, stable-frame corrected-ray application,
shadow/runtime acceptance, and all powered work remain outside this completed
task.
