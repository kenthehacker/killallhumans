# Package 2 production calibration and timing dossier

- Task ID: `vq2-package2-production-calibration`
- Parent: `2026-07-20-vq2-development-continuation-handoff`
- State:
  `post_merge_verified — stopped at empirical readiness gate; Package 2 not accepted`
- Objective: freeze Package 2's algorithm-independent calibration obligations
  and determine whether current authorized inputs can honestly support
  build-3385 pixel-to-camera-FRD, camera-to-body, and camera/IMU timing evidence.
- Starting main commit:
  `094dd6fe065f483ce7283e80d2910d03dbbea0b1`.
- Branch: `package2-production-calibration`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-production-calibration`.
- Owner and integration owner: `/root`.
- Heartbeat date: `2026-07-20`.
- Contract/readiness freeze:
  `d52652452de130e8d3990316e156c340dac5e785`.
- Behavioral and promotion commits: `not applicable`.
- Simulator access exercised: `none`.
- Maximum future access under this freeze: `none`; passive collection requires
  an exact reviewed successor contract and current authority reconciliation.
- Reserved private artifact root (not created by this task):
  `C:\Users\John\aigp-evidence\2026-07-20-package2-production-calibration`.

## Entry authority and exact stop boundary

Package 2's disjunctive entry gate permits approved build-3385 inputs or
separately authorized collection. The accepted Package 2A / Package 3B task
record preserves the user's 2026-07-20 authorization for private full-frame
storage/use, calibration collection, and passive build-3385 simulator access.
Together with the current request, that record supports selecting Package 2
and drafting this readiness contract. It is not transferable authority for a
new private capture or simulator contact, evidence that calibration inputs
already exist, or evidence that the package is accepted. Any future collection
must reconcile current explicit user authority with the governing handoff
before it contacts FlightSim or creates a private frame.

This freeze stops at a reviewed readiness decision. Package 2 itself can exit
only at a later reviewed calibration dossier. Neither boundary produces a
homography, detector observation, estimator input, sequential state, runtime
selection, supervisor carrier, transport command, or powered result. No
historical VQ1 intrinsic or extrinsic, Wave 3D synthetic identity calibration,
border-normalized pixel, fitted `/1` corner, or heuristic covariance may be
promoted as build-3385 calibration.

If a build-specific oracle or a passive dataset satisfying every readiness
condition below is unavailable, the task stops before new simulator collection
or behavioral implementation. Motion produced by reset, arm, motor output,
body-rate/thrust targets, or flight is powered calibration work and requires a
separately named task plus fresh explicit powered authorization.

## Owned and excluded surfaces

This documentation-only freeze owns only:

- this task record;
- the read-only readiness audit of already accepted evidence; and
- the algorithm-independent obligations and stop rules below.

No behavioral file, schema, dependency, API, configuration, collector, private
artifact, test surface, promotion policy, or trusted-manifest entry is owned or
authorized. Resumption requires an exact-base independently reviewed correction
or child task that freezes all such filenames/interfaces, artifact/schema
versions, dependencies, configuration formats, data/reference prerequisites,
simulator access, non-goals, tests, and stop conditions before collection or
implementation.

The contract does not own or modify the VQ2 detector, fitted observation,
Wave 3D/E reducers, relative estimator, offline runtime, controller, guidance,
supervisor, runner, MAVLink command path, vision reset lifecycle, live safety
contract, replay promotion policy, or gate-passage authority. Calibration
coefficients, full frames, annotations, reports, operational configs, and
generated inventories remain private and outside Git.

## Frozen coordinate and evidence conventions

- The only target is FlightSim build 3385 in Training mode at decoded BGR
  resolution `640 x 360`.
- Pixel coordinates use the center of the top-left decoded pixel as `(0, 0)`,
  increase right/down, and have the closed center domain
  `[0, 639] x [0, 359]`.
- Decoded-image optical coordinates are OpenCV-like `(z forward, x right,
  y down)`. Camera-FRD is exactly `(X, Y, Z) = (z, x, y)`. Rectified chart
  output is the forward-normalized slope `(Y / X, Z / X)` with a strictly
  positive finite `X`.
- `R_BC` is an active rotation with `v_B = R_BC v_C`, mapping camera-FRD
  vectors into body-FRD vectors. Its Hamilton quaternion order is
  `(w, x, y, z)` with the first nonzero component positive to remove the
  double-cover ambiguity in serialization only; this sign rule does not alter
  tangent perturbations or covariance at the pi discontinuity. Extrinsic
  perturbations are right-multiplicative,
  `R_BC(delta) = R_BC Exp([delta_theta_C]x)`, and their dense `3 x 3`
  covariance is expressed in the camera-FRD tangent coordinates
  `delta_theta_C`, ordered `(delta_X, delta_Y, delta_Z)` in radians.
  Basis-vector and handedness checks must verify this active mapping
  independently of the fitter.
- `host-perf-counter` integer nanoseconds are receiver occurrence times.
  HIGHRES_IMU `time_usec` and the camera `sim_time_ns` token remain distinct
  source clocks until a reviewed mapping proves otherwise. Legacy float
  freshness timestamps are never mixed with or relabeled as exact timing.
- Camera host receipt is sampled in Python after `recvfrom()` returns. MAVLink
  host receipt is sampled after `pymavlink.recv_match()` returns a decoded
  message. Neither value is a kernel, wire, sensor-sampling, render, or exposure
  timestamp.
- Calibration/model uncertainty is fixed or session-shared nuisance. It must
  remain separate from every future per-frame `CONDITIONAL_FIT` covariance;
  repeated frames cannot average the shared term toward zero.

## Required immutable artifact set

An accepted dossier binds, by content hash and canonical identity:

1. the exact executable/build, mode attestation, decoded resolution, capture
   sessions, split manifest, source code, dependencies, configuration, and
   reference labels or oracle;
2. the nonlinear pixel-to-camera-FRD ray model, inverse projection, validity
   domain, units, parameter order, coefficients, dense parameter covariance,
   and analytic forward/inverse Jacobians;
3. the camera-to-body rotation, perturbation convention, dense tangent-space
   uncertainty, fit diagnostics, and independent axis/sign evidence;
4. three separately named timing results: the relative camera-source to
   equivalent-IMU-source affine alignment; any source-to-host measurement map
   supported by a sender/render/sample-time oracle; and observed receiver/
   transport/queue timing. They retain exact per-sample host-arrival semantics,
   timestamp-phase assumptions, validity intervals, load envelope, and bounded
   uncertainty without claiming that one result identifies the others;
5. a nuisance ledger and full joint shared covariance retaining correlations
   among continuous fitted/shared intrinsics, extrinsics, timing, reference,
   gyro-bias, and attitude parameters, while keeping every future per-frame
   conditional-fit term separate. Discrete model form, oracle systematic, and
   nonlinear remainder retain separately named conservative bounds or sets and
   are never relabeled as Gaussian covariance. The artifact freezes the exact
   continuous tangent-space parameter order, units, and scaling; every zero
   cross-block requires independent evidence rather than an independence
   default; and
6. fit, calibration, sealed held-out, and repeat-session reports with exact
   sample identities and no overlap.

Changing any material field changes the artifact identity. A report that
omits a coefficient, correlation, rejected sample, failed readiness check,
split identity, or uncertainty contribution is invalid rather than partial
success.

## Independent reference and split protocol

The reference source must be one of:

- an organizer/build-specific render-side pixel-to-ray, camera-mount, and
  sensor-time oracle with independently verified semantics; or
- an independently specified calibration target/scene protocol providing
  nondegenerate known geometry across the image plus known rigid camera/body
  rotation relative to a stationary/reference scene and time-correlated
  HIGHRES_IMU measurements.

A moving target observed by a static body supplies visual motion but no
camera/body hand-eye rotation or temporal correlation. Dynamic evidence must
therefore use independently observed rigid-body relative rotations or an exact
render-side pose/time oracle, not merely commanded axes or moving scenery.

Detector boxes, current fitted corners, the pixel-square prior, aggregate
residuals, and values generated by the candidate model are not reference
labels. Fit and reference production must not share the same implementation.

Sessions are content-addressed and assigned once to discovery/fit,
limit-calibration, sealed held-out, or sealed repeat-session stability. One
reset/epoch/excitation run and every derived image, crop, annotation, duplicate,
or content-identical hash from that run remain in one split. Equal decoded-
content hashes are grouped globally: if a hash occurs across sessions, all
containing runs share one split or that content is excluded from evaluation.
No trajectory or scene realization crosses a split. Model family,
parameterization, preprocessing, rejection rules, covariance construction,
parameter nondimensionalization, and the algebraic rank rule are frozen in an
immutable pre-fit design commit before fitting or inspecting singular values.
All empirical uncertainty/error/coverage limits are then frozen in an
immutable limit-freeze commit after fit/limit-calibration work and before
held-out **or repeat-session** content is opened. A repeat session is
independently fitted only after the limit freeze and is never used for threshold
tuning. No held-out or repeat failure may be repaired by relabeling,
resplitting, threshold changes, or silent sample rejection.

Parameter nondimensionalization and the algebraic rank definition are chosen
before singular values are inspected. After the frozen scaling, numerical rank
uses `tau = max(m, p) * eps * sigma_max` for an `m x p` profiled Jacobian.
Passing that algebraic floor is necessary but not sufficient: practical
identifiability must also meet the separately budgeted uncertainty and
conditioning limits frozen before held-out data opens.

## Empirical readiness gate

Before implementation or new simulator collection, an independent review must
prove that the proposed inputs can identify all claimed quantities:

- across-image geometric support sufficient to distinguish principal point,
  focal scaling, the selected distortion terms, and reference pose/geometry;
- at least two nonparallel independently observed rigid-body relative
  rotations, including rate changes, reversals, and angular acceleration, plus
  independent axis/sign/handedness checks;
- time-varying, time-correlated visual and gyro angular velocity and angular
  acceleration that are jointly persistently exciting. A profiled joint
  Jacobian for extrinsic rotation, effective temporal alignment, clock
  scale/drift, and gyro bias must remain full rank after eliminating per-view
  target pose, per-run initial orientation/gauge, and reference nuisance;
- exact camera and IMU sample lineage on their source clocks and
  `host-perf-counter` arrivals, with no batch-arrival relabeling;
- immutable disjoint sessions adequate for fit, pre-limit calibration,
  held-out evaluation, and repeat-session stability; and
- a reference protocol whose uncertainty is separately bounded rather than
  absorbed into per-frame fit covariance.

Structural design, planned excitation, split independence, and reference
semantics pass before collection. Observed support, excitation, lineage, and
arrival distributions pass before fitting. Numerical local rank, conditioning,
profile likelihood, drift, and covariance are evaluated on discovery/fit and
limit-calibration data after fitting but before the limit freeze opens held-out
or repeat data. Failure at any gate does not select a simpler model by default.

## Frozen timing-model semantics

Within one exact connection/reset generation, the primary empirically
identifiable timing model maps a camera-source instant `c` to its equivalent
IMU-source instant:

```text
i_equiv(c) = i_ref + rho_CI * (c - c_ref) + delta_C_to_I
```

Camera `c_ref`, IMU `i_ref`, units, rate ratio `rho_CI`, phase correction
`delta_C_to_I`, dense parameter covariance/correlation, fit interval, maximum
interpolation gap, and maximum extrapolation horizon are explicit artifact
fields. Positive `delta_C_to_I` means the camera measurement pairs with a later
IMU-source instant than the zero-delta reference map. Because offset changes
with skew away from the reference, no phase value is reported without the
exact `(c_ref, i_ref)` pair. This relative model is identified from the same
independently referenced visual/gyro motion; it is an effective alignment that
includes unresolved timestamp phase and path delay.

An absolute source-to-host measurement map for stream `j`,

```text
m_j(s) = h_ref_j + a_j + b_j * (s - s_ref_j),
```

is admitted only with a sender/render/sample-time oracle or explicit bounded
delay/phase assumptions. It must carry the same reference, units, covariance,
validity, gap, horizon, and generation fields. Without that independent anchor,
the common host phase is unbounded and the artifact retains only the relative
inter-sensor model plus exact source/arrival pairs. It never chooses zero,
minimum arrival, or another arbitrary phase gauge and calls `a_j` calibrated.

Only for an oracle-anchored `m_j` is

```text
r_j = h_receive_j - m_j(s_j)
```

a measurement-time-to-receipt residual. Arrival pairs alone otherwise observe
clock phase plus fixed delay and clock skew plus delay trend. Correlated
visual/gyro motion does not separately identify transport, encode,
packetization, decode, queue, exposure/render phase, or IMU sample-to-send
delay. Those terms remain named shared nuisance with bounded timestamp-phase
uncertainty, not silently absorbed and relabeled as physical clock offset.

Minimum/lower-envelope arrivals, TIMESYNC traffic, the wall-like magnitude of
the camera token, and differences between independently fitted arrival
intercepts cannot establish calibrated offset or one-way delay. Finite-sample
p50/p95/p99/max values are observed distributions only. Any future operational
delay or jitter bound additionally requires a pre-frozen host/graphics/capture
load envelope plus held-out and repeat sessions; capture-loaded passive timing
does not establish no-capture latency or a future hard maximum.

A source duplicate, regression, wrap, or change point; connection/reset or host
boot change; QPC regression or frequency change; batch-arrival relabeling;
sample gap, drop, overflow, or incomplete lineage; or rank, uncertainty,
held-out, or repeat-session limit failure invalidates the affected session and
stops calibration. It is never repaired by fitting across the discontinuity.

## Acceptance and verification

Empirical acceptance must include:

- held-out reprojection and angular residual distributions across a frozen
  image grid, with tail and maximum failures retained;
- forward/inverse round trips throughout the validity domain;
- analytic pixel/ray and parameter Jacobians against an independently
  implemented finite-difference oracle;
- Schur-complement/projected information after eliminating per-view/per-run
  nuisance, full joint shared-covariance cross-block influence, PSD checks,
  perturbation coverage, and fixed/shared-nuisance non-averaging tests;
- independent camera/body basis-vector, axis, sign, and handedness checks;
- separate held-out coverage for effective inter-sensor alignment, any
  oracle-supported source-to-host mapping, timestamp-phase/unresolved-delay
  nuisance, and observed arrival distributions, without converting finite-
  sample maxima into untested hard bounds;
- repeat-session coefficient and prediction stability under the pre-frozen
  limits; and
- fail-closed behavior outside resolution, domain, build, mode, time validity,
  provenance, excitation, and uncertainty bounds.

The readiness review identified these possible future direct-test locations,
but this freeze neither owns nor creates them:

- `estimation/tests/test_vq2_camera_calibration.py`;
- `estimation/tests/test_vq2_sensor_timing_calibration.py`; and
- `tests/test_aigp_vq2_calibration_script.py` when a private-data CLI exists.

A reviewed successor must freeze its exact direct and compatibility paths.
Expected compatibility includes IMU provenance/derotation, local-differential
measurement, VQ2 contracts/capture/passive timing, and legacy PnP only as a
non-regression surface—not as positive build-3385 evidence. It must follow the
complete parent promotion checklist: affected tests after each edit;
`test-vq2`, `test-fast`, and `test-unit`; `test-full-non-live` from a fresh exact
promotion-test worktree with every physical side effect inventoried; a separate
fresh exact hash-pinned VQ2 worktree; integration of the unchanged candidate;
and post-merge VQ2 verification. Synthetic tests are T0 mechanism evidence and
never substitute for the private empirical dossier.

The machine-precision algebraic rank floor is frozen above. No empirical pixel,
millisecond, degree, RMS, tail, practical-identifiability, conditioning, or
stability threshold is invented by this initial contract. Exact empirical
definitions and numerical values are derived only from discovery/fit and
limit-calibration data, independently reviewed, and committed at the immutable
limit freeze before sealed held-out or repeat data is opened. Existing `20 ms`
capture-alignment, `50 ms` total-timing, and `5 degree` combined-angular gates
are downstream compatibility ceilings, not empirical Package 2 acceptance
limits or proof that the remaining budget is available to calibration error.

## Simulator and data safety

No simulator contact occurs until the readiness contract, reference protocol,
private output roots, exact collection command, attempt bounds, exclusive live
lease, outbound audit, process/build proof, port cleanup, and failure
invalidation rules pass independent review. Passive means no `SIM_RESET`, arm,
disarm, attitude/position target, motor-producing command, or flight cleanup
path. Only the already reviewed passive heartbeat/TIMESYNC traffic is allowed.
Any disallowed send, bind conflict, changed build/process, incomplete artifact,
receiver termination failure, or uncleared port invalidates the attempt and
stops simulator work.

This documentation-only freeze requires `git diff --check`, independent
calibration/identifiability, timing, lifecycle/provenance, and authority reviews,
and a clean committed worktree. Behavioral, canonical, broad, promotion,
hash-pinned, and post-merge test results are `not applicable` because this task
owns no executable or promotion surface. A future successor cannot inherit that
`not applicable` result.

## Entry audit result

At contract drafting, the repo, official v2 devkit, and exposed build-3385
text/config contain no approved camera intrinsic, distortion, mount, or
measurement-time artifact. No discovered numeric camera value is an approved
build-3385 calibration: historical VQ1 constants are excluded, and VQ2 runner/
image-normalization heuristics are not calibration inputs. The official
receiver example comments that the camera token is a server epoch timestamp,
but defines neither render/exposure phase nor its relationship to HIGHRES_IMU
time or host receipt. The verified VQ2 contract therefore correctly keeps it
opaque until an oracle or empirical model supplies those missing semantics and
uncertainty.

The three accepted Package 2A / Package 3B passive sessions contain only a
stationary spawn view and explicitly leave camera/IMU clock calibration
unmeasured. Sessions 02 through 04 contain respectively `6`, `5`, and `5`
unique decoded frame hashes across `182`, `181`, and `181` processed frames.
Every selected target box is exactly `(282, 134, 80, 80)`, and all `2,110`
accepted HIGHRES_IMU gyro vectors are exact zero. These facts prove absence of
hand-eye and temporal-offset excitation; they are receiver-boundary evidence,
not fit samples, and will not be recollected or upgraded.

They also contain no independent known-geometry label: the same unknown-pose
planar gate occupies one box and pose. Repetition cannot separate focal scale,
principal point, distortion, and scene/pose nuisance or validate residuals
across the image. The current inputs therefore fail the intrinsic as well as
the full-rotation and temporal-offset readiness conditions.

No new simulator probe or behavioral module is authorized by this finding.
Resumption requires either an approved build-3385
oracle/non-flight reference protocol satisfying every readiness item or fresh
explicit authorization for a separately frozen powered calibration stage.

## Review and integration record

Independent calibration/identifiability review cleared the active FRD/optical
and SO(3) conventions, joint nuisance treatment, split isolation, pre-fit rank
rule, empirical-limit sequencing, and the conclusion that current intrinsics,
extrinsics, and temporal alignment are unidentifiable. Independent timing
review cleared the relative source-clock estimand, oracle-only absolute host
map, phase-gauge prohibition, arrival-boundary semantics, uncertainty split,
and discontinuity invalidation. Independent lifecycle/provenance/authority
review cleared only this documentation/readiness boundary and explicitly did
not clear capture, simulator access, or behavioral implementation.

The exact staged delta was one new `374`-line task record. `git diff --check`
was clean, UTF-8 validation succeeded, and no line exceeded 88 characters.
Behavioral, canonical, broad, promotion, hash-pinned, and post-merge test gates
are `not applicable` under the frozen documentation-only scope. No executable,
policy, trusted-manifest, private artifact, dependency, or generated inventory
changed.

Integration owner `/root` committed the reviewed freeze at `d526524` and
fast-forwarded tracked-clean local `main` from `094dd6f` to that exact commit.
Post-merge tracked status was empty. No FlightSim process was launched or
contacted; no network/port, private capture, replay, reset, arm/disarm, target,
transport, shadow, or powered action occurred.
