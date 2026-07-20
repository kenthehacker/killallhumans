# Wave 3D stable-reference local-feature transform

- Task ID: `vq2-wave3d-stable-reference`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `post_merge_verified`
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
  attitude/rate/timing limits, and explicit angular- and feature-acceleration
  bounds;
- `VQ2StableReferenceKey` carries the stable owner/source/model identity;
- `VQ2StableReference` retains the immutable seed evidence and derived
  capture-time chart, and exposes a deterministic local `basis_id` fingerprint
  over the caller reference ID plus the complete key, seed, and orientation;
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
basis at that same time. Feature semantic, chart model, derived seed-bound basis
ID, host clock, and covariance model must match exactly. Reusing the same
caller-supplied reference ID cannot alias two distinct seed charts. The sequence
validator accepts only camera-to-stable `CAPTURE` measurement transforms.

## Reference lifecycle

- A reference is created atomically from one integrity-revalidated public
  `VQ2DerotationEvidence` capture-time camera orientation plus an explicit
  owner tracker ID, owner role, reference ID, and stable-transform model.
- The reference orientation is immutable. There is no implicit rebase or
  rollover of a live posterior.
- The caller-supplied reference ID is a label, not sufficient chart identity.
  Stable states bind the deterministic local `reference.basis_id` fingerprint
  over that label and the complete reference key, seed evidence, and derived
  orientation. The fingerprint is an in-process integrity identity, not a wire
  schema or cross-version persistence format.
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
  `race_status_boot_ms`. Camera opaque source time within the fixed generation,
  publication sequence, publish time, and measurement time are strictly
  increasing, and decision/prediction time cannot regress. Camera and IMU
  opaque source times are ordering tokens and are never subtracted from host
  time. Each capture and target attitude input may be exactly reused;
  otherwise its sample sequence, opaque source time, and host receipt time must
  all advance coherently.
- Reference establishment requires a usable, complete fitted quadrilateral
  with all four visible, non-clipped corners. Establishment latches only
  reference orientation and lineage. The seed may be paired once with an
  independently supplied local-differential feature whose value and covariance
  were produced under the exact canonical local-scale model. Wave 3D never
  derives that feature from `/1` center, finite scale, skew, or covariance
  summaries. Reuse of the seed frame requires the exact original complete
  evidence context; it cannot be retargeted or relabelled. Every later source
  must be a strictly newer distinct frame with advancing source/publication/
  measurement chronology and coherent capture and target IMU lineage. Because
  the local feature is independently supplied, a later usable derotation
  source need not retain complete finite-quad `/1` scale/skew/corners.
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
  The functions reject a directly returned state while it remains marked
  `TRANSFORM_TOTAL`. `covariance_scope` is nevertheless a caller assertion: a
  stateless value transform cannot detect a newly constructed or dishonestly
  relabelled covariance. No stronger anti-recycling, sequential-filter, or
  averaging claim is made. Sequential use requires an approved provenance
  carrier plus an augmented/Schmidt state, retained cross-covariance, or a
  separately proved dominating common-mode construction.
- Rate and expansion timing sensitivity includes `A`, `A_dot`, rate-squared/
  projective terms, and `A_ddot`; physical feature-time displacement may also
  require a feature-acceleration bound. The stable model therefore requires
  explicit angular- and feature-acceleration bounds and binds them into the
  reference. In this standalone version, those scalars and the supplied joint
  nuisance matrix are declarative model assumptions: the model author asserts
  that the matrix dominates the full bounded sensitivity. The module hard-
  validates the values and matrix but does not derive or prove that dominance.
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
  `validate_integrity()`. Nested public observations are round-tripped through
  the public `/1` primitive schema so shallow frozen-object mutation cannot
  relabel frame timing, authority, geometry, or covariance evidence.

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
- Tests proving the joint nuisance envelope remains a single one-shot bound,
  directly returned total-labelled states are rejected, and the stateless
  primitive makes no claim to detect caller reconstruction/relabeling or to
  support independent-noise averaging.
- Direct target, the new test plus existing derotation/provenance/relative-
  estimator compatibility matrix, canonical and isolated-manifest `test-vq2`,
  `test-fast`, `test-unit`, promotion-boundary `test-full-non-live`, independent
  contract review, clean candidate, integration, and post-merge `test-vq2`.

Skipped optional coverage is never positive evidence. No simulator, network,
preflight, reset, arm/disarm, target, transport, shadow, or powered action is
permitted for this task.

## Behavioral review

- The standalone implementation adds only the owned module, direct tests, and
  these design/task records. Repository-wide static inspection finds no
  production import or call site and no private derotation-helper dependency.
- Direct stable-reference tests pass `82`; the stable-reference plus existing
  derotation/provenance/relative-estimator compatibility matrix passes `186`;
  and the pre-promotion canonical VQ2 candidate passes exactly `1,101` tests
  (`1,019 + 82`).
- Independent mathematical re-review cleared the homography and derivative,
  local-area scale/expansion law, inverse, analytic `6x6` Jacobian, covariance
  congruence, normalized extrinsic, and PSD changes. A 2,000-case independent
  sweep bounded oracle error at `8.88e-16`, finite-difference Jacobian error at
  `4.52e-9`, inverse-state error at `1.11e-15`, inverse-Jacobian composition at
  `1.78e-15`, and finite-time rate/scale error at `4.13e-10`.
- Independent lifecycle and test re-reviews cleared source/authority/IMU
  chronology, exact seed context, derived basis identity, nested public-schema
  integrity, uncertainty/geometry boundaries, covariance labeling, public
  signatures/types, finite-quad semantic separation, and no-wiring scope.
- Review-discovered defects were corrected before acceptance: scale-insensitive
  PSD tolerance, shallow nested observation validation, target-IMU relabeling,
  same-label reference aliasing, camera source-time replay, corrupted model
  establishment, near-unit extrinsic rate scaling, and unnecessary later-frame
  finite-quad coupling.
- Simulator access remained `none`. No network, preflight, replay, transport,
  shadow/runtime, reset, arm/disarm, target, or powered action occurred.

## Promotion review

- Behavioral implementation commit:
  `c21a742004d1d3bc485a866babb9759b6aee62fb`.
- Promotion/trust and integrated main commit:
  `46df0adee76070e10509fa5e807b986a9469c68e`.
- The exact T1 policy advances from `1,019` to `1,101` passed tests and adds
  only `estimation/tests/test_vq2_stable_reference.py` to its sorted inventory.
  The policy file SHA-256 is
  `a98b2d4d618b6999927d1c997ca0a65c63aebef742c53bef31a6c05dcd53b020`;
  its semantic SHA-256 is
  `0053eaf52fa6a9b273abb558e83c89aa1b4560888eaa7f9db11a371d5c3c9ab4`.
- The reviewed trusted manifest advances from `127` to `128` files. Its only
  addition is the new test digest
  `b9dddab8dffca4417c651e85aac07d661c2429dd42edb76cd919d72934efbf45`;
  its only changed existing digest is the policy; and no entry is removed. All
  `128` hashes independently match disk. Manifest semantic identity is
  `2f70415dd7cdfa0675c6dc778406cdccfdca09757e79b1a8f1a3e0d4752e9268`;
  manifest file SHA-256 is
  `2c965f2f5a6486f506d51c8e290b09d6a22166f6f277fbff1234690e510d63d9`.
- Canonical and cache-clean isolated-manifest VQ2 gates each pass exactly
  `1,101` tests. `test-fast` and `test-unit` each pass `2,196` with `20`
  skips and `42` deselections. Promotion-boundary `test-full-non-live` passes
  `2,237` with `21` skips. The direct and focused counts remain `82` and `186`.
- Independent promotion metadata review cleared the exact count arithmetic,
  sorted/unique policy inventory, manifest delta, every on-disk digest, and
  file/semantic identities. Generated cache directories were removed before
  the final isolated run; they contained no source or evidence.
- Main fast-forwarded from the reviewed Wave 3C closeout to the promotion
  commit. Post-merge canonical `test-vq2` passed all `1,101` tests, and tracked
  Git status was empty before documentation closeout.
- No simulator, external network, replay corpus, preflight, runtime/shadow,
  transport, reset, arm/disarm, target, or powered action was used.

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
