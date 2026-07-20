# Wave 3C one-tick IMU-correlated coast lease

- Task ID: `vq2-wave3c-correlated-coast`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `candidate_complete`
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

## Candidate verification

- Direct focused tests cover estimator-envelope forgeries, public
  lower-layer rejection and private-call ownership, Gate 0 and full credited
  Gate 1 success, one-use/skip/new-frame consumption, lifecycle and IMU
  failures, exact trace/result binding, transaction retry, and recovery.
- The frozen six-module affected matrix passes `477` tests. Independent final
  contract review passes a `201`-test deep matrix and explicitly clears the
  coast envelope, lease, memory, transition, result, reconstruction, scheduler,
  and cumulative-trace scope with no remaining promotion blocker.
- Canonical VQ2 policy, promotion-boundary non-live suites, trusted-manifest
  reconciliation, merge, and post-merge verification remain pending and must
  replace this section at closeout.
- Simulator access for the implementation and tests is `none`.
