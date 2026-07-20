# Wave 3D stable-reference local-feature transform

- Task ID: `vq2-wave3d-stable-reference`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `contract_frozen`
- Objective: add a standalone, immutable, bidirectional stable-orientation
  transform for a distinctly named six-dimensional local-differential pinhole
  feature state, with exact reference lineage, rate chain rules, and full
  covariance propagation.
- Starting main commit:
  `7c31f86e1be38d335aaa161af787392cddbb67c4`.
- Branch: `wave3d-stable-reference`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-wave3d-stable-reference`.
- Owner: `/root`.
- Heartbeat date: `2026-07-19`.
- Simulator access: `none`.

## Owned files

- `estimation/vq2_stable_reference.py`
- `estimation/tests/test_vq2_stable_reference.py`
- `docs/aigp/vq2_stable_reference.md`
- this task record

Promotion policy and trusted-manifest metadata remain integration-owner files
and belong in a separate commit after behavioral review. Shared handoffs are
updated only after integration and post-merge verification.

The tranche does not own or modify `competition/vq2_contracts.py`,
`estimation/vq2_imu_derotation.py`, `estimation/vq2_imu_provenance.py`,
`estimation/vq2_relative_estimator.py`, any adapter/runtime/controller/guidance
module, supervisor, runner, transport, or powered surface. It must not import
private helpers from the existing derotation module.

## Frozen semantic boundary

The existing `/1` `log_scale` is one half of the logarithm of a finite fitted
quadrilateral's polygon area. A center bearing, finite area, projective skew,
and their rates do not retain enough shape information to transform that area
exactly through a projective camera rotation. Therefore:

- Wave 3D introduces a local-only `local_log_scale`, defined as one half of the
  logarithm of the positive normalized-image area density of a fixed canonical
  aperture element at its center. The canonical square is `[-1,1]^2`; for its
  fitted homography `G`, the definition is
  `log(2) + 0.5 log(det(D project(G)(0)))`;
- its six-dimensional transform is exact only for that named local
  differential feature model;
- no Wave 3D function accepts or returns `GateObservationV1`,
  `RelativeGateStateV1`, or a frozen `/1` scale/state as its feature value;
- existing public `/1` semantics, schemas, estimator behavior, and proposal
  behavior remain byte-for-byte and behaviorally unchanged; and
- future estimator use requires either a reviewed local-scale measurement and
  covariance derived from the full fitted homography, or a shape-augmented
  state carrying the full quadrilateral, rates, and covariance.

The design document records a constructive counterexample. Any implementation
or documentation that calls the local scale an exact transform of current
`/1` finite polygon area fails this task.

The frozen public surface is local and non-wire:

- `VQ2StableReferenceModel` carries an explicit local-feature semantic,
  synthetic chart calibration, reference-age/conditioning/output bounds,
  attitude/rate/timing limits, and an angular-acceleration bound;
- `VQ2StableReferenceKey` carries the stable owner/source/model identity;
- `VQ2StableReference` retains the immutable seed evidence and derived
  capture-time chart;
- `VQ2LocalDifferentialFeatureState` carries the exact local six-vector,
  basis/time/model identity, and dense covariance;
- `VQ2StableFeatureTransformEvidence` retains forward/inverse inputs, derived
  homography and derivative, analytic Jacobian, separated covariance terms,
  and output; and
- the exact functions below provide the only operations. There is no stateful
  rollover API.

```python
establish_stable_reference(
    *,
    reference_id: str,
    tracker_id: str,
    track_role: TrackRole,
    seed_evidence: VQ2DerotationEvidence,
    model: VQ2StableReferenceModel,
) -> VQ2StableReference

camera_to_stable_local_differential(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    state: VQ2LocalDifferentialFeatureState,
    *,
    camera_time: VQ2CameraFeatureTime,
) -> VQ2StableFeatureTransformEvidence

stable_to_camera_local_differential(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    state: VQ2LocalDifferentialFeatureState,
    *,
    camera_time: VQ2CameraFeatureTime,
) -> VQ2StableFeatureTransformEvidence

validate_stable_measurement_sequence(
    reference: VQ2StableReference,
    transforms: tuple[VQ2StableFeatureTransformEvidence, ...],
) -> None
```

`CAPTURE` binds the camera state time exactly to the observation measurement
time. `TARGET` binds it exactly to the prediction-target time. Forward input
must use the selected camera basis; inverse input must use the stable reference
basis at that same time. Feature semantic, chart model, basis ID, host clock,
and covariance model must match exactly. The sequence validator accepts only
camera-to-stable `CAPTURE` measurement transforms.

## Reference lifecycle

- A reference is created atomically from one integrity-revalidated public
  `VQ2DerotationEvidence` capture-time camera orientation plus an explicit
  owner tracker ID, owner role, reference ID, and stable-transform model.
- The reference orientation is immutable. There is no implicit rebase or
  rollover of a live posterior.
- Its lifecycle key binds owner tracker and role; session, reset, gate epoch,
  expected gate index, camera host clock/stream/generation; exact IMU
  `epoch_key`; calibration and camera-ray model; derotation and attitude-time
  model; image size; explicit stable chart calibration/model; and observation
  measurement-time basis/model. It binds the entire immutable calibration,
  derotation model, stable model, and chart-matrix values, not their IDs alone.
- Race-status refresh sequence and authority cutover watermarks are not a
  stable-basis identity. They remain validated inside each source observation
  but do not force a reference rebase by themselves. Across a validated
  measurement sequence, race-status sequence/boot time and cutover watermarks
  follow the complete same-epoch authority-snapshot transition rules: they are
  monotonic, and an unchanged `race_status_sequence` requires exactly unchanged
  `race_status_boot_ms`. Publication sequence, publish time, and measurement
  time are strictly increasing, and decision/prediction time cannot regress.
  An exact attitude input may be reused. Otherwise sample sequence, opaque
  source time, and host receipt time must all advance coherently; source time is
  never subtracted from host time.
- Reference establishment requires a usable, complete fitted quadrilateral
  with all four visible, non-clipped corners. Establishment latches only
  reference orientation and lineage. The seed may be paired once with an
  independently supplied local-differential feature whose value and covariance
  were produced under the exact canonical local-scale model. Wave 3D never
  derives that feature from `/1` center, finite scale, skew, or covariance
  summaries. Every later source must be a strictly newer distinct frame with
  advancing publication and measurement chronology and coherent IMU lineage.
- A lifecycle-key change, tracker replacement, or explicit retirement requires
  discarding the old reference and
  rebootstrap. Innovation rejection, a transient missing frame, or missing IMU
  evidence withholds a transform and must not move the reference.
- A future estimator integration must retire the reference on estimator gap
  reinitialization; the standalone module has no estimator event input and
  makes no directly testable gap-transition claim.
- Reference mismatch, replay, chronology conflict, malformed evidence, excess
  uncertainty, nonforward geometry, or output-envelope failure rejects without
  mutable state because the module is pure.

## Transform and covariance invariants

- The stable model supplies an explicit nonsingular homogeneous chart-to-camera
  ray matrix. The first implementation accepts the synthetic
  `(forward, right, down) = (1, u, v)` chart only; it does not silently treat
  border-normalized image coordinates as calibrated production pinhole slopes.
  The stable basis is the seed capture camera orientation frozen in the
  attitude estimator's yaw-unobservable NED gauge; it is an orientation chart,
  not verified absolute heading, a metric position, or a map frame.
- Public derotation evidence supplies exact capture/target orientation lineage.
  The new module independently derives only public quaternion/rate values and
  cross-checks compatible bearing signs; it does not depend on private
  implementation functions.
- Both capture-camera and target-camera bases may be transformed to stable and
  back. Input state time and basis must exactly match the selected evidence
  time and calibration identity.
- For the determinant-one chart homography induced by the proper ray-space
  rotation, `p' = project(A [1,p])` and local scale transforms by
  `local_log_scale' = local_log_scale + 0.5 log(det(dp'/dp))`. Homography
  rescaling is forbidden.
- Bearing rates and local expansion use the complete time-varying chain rule
  from `A` and `A_dot`; inverse transformation uses `A^-1` and
  `d(A^-1)/dt = -A^-1 A_dot A^-1`. Forward followed by inverse recovers the
  local feature state and the two analytic state Jacobians compose to identity
  within the reviewed numerical tolerance.
- The analytic `6x6` state Jacobian propagates every dense covariance and
  retains all bearing/scale/rate cross terms. Because current evidence does not
  publish the full common/current nuisance covariance, the explicit stable
  model supplies one exact dense `6x6` PSD joint-nuisance envelope and one exact
  dense `6x6` PSD model-floor covariance in output local-feature coordinates.
  They are bounds, not probabilistic-sigma claims, and must dominate unknown
  common/current cross-correlation. The result exposes congruence, joint-
  nuisance envelope, model floor, and total matrices deterministically.
- The result is a one-shot conditional envelope whose input covariance excludes
  previously added transform nuisances and feature/nuisance cross-covariance.
  A total covariance from an earlier transform cannot be fed through and have
  the same nuisance envelope added again. Sequential filtering requires an
  augmented/Schmidt state, retained cross-covariance, or a separately proved
  dominating common-mode construction.
- Rate and expansion timing uncertainty uses the full time sensitivity of `A`
  and `A_dot`, including angular-rate-squared/projective terms and `A_ddot`.
  Bounding `A_ddot` requires an explicit angular-acceleration bound. If timing
  uncertainty also moves physical feature time, a feature-acceleration/model
  bound is additionally required; otherwise the rate transform is withheld.
- Every input, reference, rotation, derivative, Jacobian, covariance component,
  and output is finite, dimensionally exact, symmetric where required, and
  positive semidefinite within a scale-aware tolerance. Numerical stabilization
  may add uncertainty but never erase a negative eigenvalue as if it were
  evidence.
- Projective denominators and local area Jacobians must remain strictly
  positive; input/output bearings and rates remain inside explicit hard bounds.
  Homography/Jacobian condition, magnification, reference age, relative
  orientation uncertainty, rate uncertainty, and timing uncertainty remain
  bounded. No uncalibrated default model exists.
- A frozen transform result re-derives all nested and derived values during
  `validate_integrity()` so low-level mutation cannot relabel evidence.

## Authority and evidence boundary

This module is pure generated/offline math. It grants no gate ownership,
guidance, controller, supervisor, proposal, approval, projection, transport,
reset, arm/disarm, cleanup, simulator, shadow, or powered authority. It makes
no translation, metric pose, physical gate size, calibrated production
extrinsics/timing, finite-quad `/1` scale, replay, measured latency, or plant
response claim.

The transform remains default-off and has no production call site. It must not
be wired into the current estimator, adapters, runtime, `/1` state/proposal, or
transport in this tranche. The existing corrected ray remains standalone from
the production path.

## Required evidence

- Direct tests for identity and axis signs; capture and target bases; negative
  capture alignment;
  nontrivial forward/inverse state and Jacobian round trips; local scale,
  bearing-rate, and expansion chain rules; analytic versus independent finite-
  difference Jacobians; dense covariance congruence, cross terms, PSD nuisance
  growth, and determinism.
- Direct lifecycle tests for authority, tracker/role, host clock, camera and IMU
  epochs, calibration/ray/time models, seed reuse, distinct-frame chronology,
  replay, and explicit rebootstrap boundaries.
- Adversarial tests for exact types, booleans, nonfinite values, quaternion and
  covariance corruption, singular/nonforward/out-of-envelope transforms,
  excessive uncertainty, forged derived results, and transactional rejection.
- A finite-quadrilateral counterexample proving that identical current `/1`
  center/area/skew can yield different rotated finite areas.
- Static boundary tests proving no production import/call site, no private
  derotation-helper import, no `/1` contract edit, and no authority/transport
  dependency.
- Tests proving the joint nuisance envelope remains a single one-shot bound and
  cannot be presented as independent per-frame noise, fed back as fresh input,
  or averaged away by this stateless primitive.
- Direct target, the new test plus existing derotation/provenance/relative-
  estimator compatibility matrix, canonical and isolated-manifest `test-vq2`,
  `test-fast`, `test-unit`, promotion-boundary `test-full-non-live`, independent
  contract review, clean candidate, integration, and post-merge `test-vq2`.

Skipped optional coverage is never positive evidence. No simulator, network,
preflight, reset, arm/disarm, target, transport, shadow, or powered action is
permitted for this task.

## Contract-freeze review

Two independent read-only reviews cleared this contract before implementation:

- the mathematical review verified the finite-quadrilateral counterexample,
  determinant-one chart and inverse equations, canonical local-scale semantic,
  full rate/expansion chain rule, analytic Jacobian boundary, joint one-shot
  covariance envelope, and timing/acceleration caveats; and
- the lifecycle review verified the exact public surface, immutable reference
  key, basis/time binding, complete authority snapshot and IMU chronology,
  independent local-feature input, deep integrity boundary, and no-wiring/non-
  authority scope.

Both reviews reported no remaining contract-freeze blocker. No existing source,
contract, estimator, adapter, runtime, simulator, network, or powered surface
was changed or exercised during the freeze.
