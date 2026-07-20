# VQ2 stable-reference local-feature transform

## Purpose and non-authority

This standalone Wave 3D module implements the pure mathematics and lifecycle for a
fixed-orientation feature basis. It consumes integrity-checked public
`VQ2DerotationEvidence` only for attitude, time, calibration, and source
lineage. It neither changes nor feeds the existing estimator.

The stable reference is the first eligible capture camera orientation frozen
in the attitude estimator's yaw-unobservable NED gauge for one exact
authority/tracker/source/model lifecycle. Common arbitrary yaw cancels in the
relative chart, but gyro drift remains uncertainty. It is not verified absolute
heading, a world position, map, metric pose, or assertion that the camera
remained at the same translation. Rotation is the only modeled coordinate
change.

There is no supervisor, command, runner, transport, simulator, reset, arm,
cleanup, or powered authority in this design.

## Why the frozen `/1` scale cannot be transformed in six dimensions

The current `/1` observation defines

```text
log_scale = 0.5 * log(finite fitted-quadrilateral polygon area).
```

A projective rotation does not transform every finite polygon by the Jacobian
at its center. Center, area, and the two published skew summaries do not encode
the complete quadrilateral. For example, under a 20-degree yaw, centered
`0.8 x 0.2` and `0.4 x 0.4` rectangles both start with area `0.16`, identical
center, and zero skew, but their transformed finite areas are approximately
`0.2012663` and `0.1948845`.

Consequently no function of the current six-state

```text
[bearing_x, bearing_y, log_scale,
 bearing_rate_x, bearing_rate_y, expansion_rate]
```

can be an exact, invertible camera-to-stable transform of `/1` semantics. Even
adding the two current skew summaries is insufficient. An exact finite-polygon
model must retain the complete quadrilateral plus its rates and covariance.

Wave 3D therefore uses a different, local-only scale semantic. For a fixed
canonical aperture coordinate with square `[-1,1]^2` and fitted homography `G`,
define

```text
local_log_scale = log(2) + 0.5 log(det(D project(G)(0))).
```

This is one half of the logarithm of the positive normalized-image area density
of the fixed canonical aperture element at its center. The `log(2)` convention
makes it equal finite `/1` scale for an affine quadrilateral, but not for a
general perspective quadrilateral. It is a candidate mathematical primitive
for a future local `/2` measurement model, not a relabeling or replacement of
`/1`. A future producer would need to derive its value and covariance from the
full fitted homography or carry a shape-augmented state. That producer,
estimator wiring, controller retuning, replay, and runtime acceptance are
outside this tranche.

## Local feature transform

Let the local state in one normalized pinhole chart be

```text
x = [u, v, ell, u_dot, v_dot, ell_dot].
```

Let `A(t)` be the determinant-one homogeneous chart map induced by the proper
ray-space rotation from the input camera chart to the output chart, and let
`A_dot(t)` be its time derivative. Arbitrary homography rescaling is forbidden.
Define

```text
r     = [1, u, v]^T
r_dot = [0, u_dot, v_dot]^T
s     = A r       = [X, Y, Z]^T
s_dot = A_dot r + A r_dot = [X_dot, Y_dot, Z_dot]^T.
```

For a general fixed-scale homography, define

```text
alpha     = 0.5 log(det(A)) - 1.5 log(X)
alpha_dot = 0.5 trace(A^-1 A_dot) - 1.5 X_dot / X.
```

The output state is

```text
U       = Y / X
V       = Z / X
ell'    = ell + alpha
U_dot   = (Y_dot X - Y X_dot) / X^2
V_dot   = (Z_dot X - Z X_dot) / X^2
ell_dot'= ell_dot + alpha_dot.
```

For a proper rotational homography, the local image-area Jacobian determinant
is `1 / X^3`. The canonical similarity construction below has `det(A)=1` and
`trace(A^-1 A_dot)=0`, yielding the simplified `-1.5 log(X)` and
`-1.5 X_dot/X` terms. These equations are exact for the stated local
differential semantic.

The general calibrated chart uses an explicit nonsingular homogeneous matrix
`K` from chart coordinates `[1,u,v]` to camera-FRD rays. With the current
camera-to-NED rotation `C(t)` and frozen reference camera-to-NED rotation
`C_ref`,

```text
M     = C_ref^T C
M_dot = M [omega_camera]x
A     = K^-1 M K
A_dot = K^-1 M_dot K.
```

For the inverse stable-to-camera transformation,

```text
A_inverse     = A^-1
A_inverse_dot = -A^-1 A_dot A^-1.
```

Thus the inverse includes the camera-rotation contribution to apparent bearing
rates and expansion rates; it is not merely an inverse point rotation. The
initial implementation permits only an explicitly supplied synthetic
identity `K`, matching the existing `(1,u,v)` ray convention. Build-3385
production intrinsics and distortion remain uncalibrated and must not be
inferred from historical VQ1 values.

`K` represents a rectified pinhole chart only. Lens distortion cannot be
encoded in this matrix and requires separately calibrated nonlinear ray/project
functions and their Jacobians in a future model.

`omega_camera` is the selected evidence body rate rotated from body FRD into
camera FRD by the exact camera-to-body extrinsic. With the active camera-to-NED
convention, `C_dot = C [omega_camera]x`; compatibility tests freeze the yaw,
pitch, and roll signs.

## Analytic state Jacobian

For each input component, derive `dr`, `dr_dot`, then

```text
ds     = A dr
ds_dot = A_dot dr + A dr_dot.
```

Quotient differentiation supplies the bearing rows. The scale rows are

```text
d ell'     = d ell - 1.5 dX / X
d ell_dot' = d ell_dot
             - 1.5 (dX_dot X - X_dot dX) / X^2.
```

Differentiating the two quotient-rate expressions supplies the remaining
rows. The implementation exposes the resulting analytic `6x6` Jacobian `G`.
Independent central finite differences verify it away from projective guards;
the forward and inverse Jacobians must compose to identity at the transformed
state.

## Full covariance accounting

For a dense input covariance `P`, the deterministic coordinate contribution is

```text
P_coordinate = G P G^T.
```

No diagonal-only shortcut is permitted: all bearing/scale/rate cross terms are
retained.

Attitude, calibration, and angular-rate uncertainty are transform nuisances,
not new independent feature measurements. The mathematically general form is

```text
B = [B_common B_now]
P_nuisance = B S_joint B^T
P_output = P_coordinate + P_nuisance + P_model,
```

where `S_joint` retains common/current cross blocks. Current public evidence
does not publish this joint covariance or its feature cross-covariance. The
standalone v1 model therefore requires one explicit dense `6x6` PSD joint-
nuisance envelope in output feature coordinates that is declared to dominate
those unknown cross terms, plus a separate explicit dense model-floor
covariance. These are deterministic covariance envelopes, not claims that the
underlying scalar bounds are probabilistic sigmas.

The evidence exposes congruence, joint envelope, model floor, and total. A
forward transform followed by an inverse transform recovers state and
coordinate Jacobian, but adding a second nuisance envelope on the way back is
intentionally not described as a covariance round trip. Repeated measurements
also must not average a fixed reference or calibration error toward zero. A
future filter needs an augmented or Schmidt nuisance state, retained cross-
covariance, or an explicit proved common-mode construction; Wave 3D does not
hide that choice.

`P_output = G P G^T + ...` is a one-shot conditional envelope only when `P`
excludes prior transform nuisances and feature/nuisance cross-covariance. The
implementation rejects a directly returned state while its scope remains
`TRANSFORM_TOTAL`. That scope is a caller assertion, not an unforgeable
provenance carrier: a stateless mathematical transform cannot detect a newly
constructed or dishonestly relabelled covariance. Therefore Wave 3D makes no
stronger anti-recycling, sequential-filter, or independent-noise averaging
claim. Any such use needs an approved provenance carrier and retained nuisance
cross-covariance or another separately proved construction.

Current provenance provides point angular rate but not angular acceleration.
Rate and expansion timing sensitivity includes the full time sensitivity of
`A` and `A_dot`, angular-rate-squared/projective terms, and `A_ddot`. If timing
uncertainty moves physical feature time, feature acceleration also matters.
The stable model therefore requires explicit angular- and feature-acceleration
bounds. In this standalone version those scalars and the supplied joint
nuisance envelope are declarative model assumptions: the model author asserts
that the matrix dominates the full bounded sensitivity. The module validates
and binds the values and matrix, but it does not derive or independently prove
that dominance.

Positive-semidefinite validation is scale-aware. Tiny symmetric roundoff may
be floored only within an explicit tolerance. A materially negative
eigenvalue, nonfinite value, nonpositive marginal variance, singular projective
denominator, or nonpositive local area determinant rejects rather than being
silently repaired.

## Reference and chronology contract

Reference creation binds:

- explicit reference ID, owner tracker ID, and owner role;
- session/reset/gate epoch and expected gate index;
- camera host clock, stream, and generation;
- exact IMU source `epoch_key`;
- calibration and camera-ray model identity;
- image size and explicit stable chart calibration identity;
- derotation and attitude-time model identity;
- observation measurement-time basis and model; and
- the seed frame, publication, measurement time, candidate, capture attitude,
  and derived capture camera-to-NED orientation.

The caller reference ID is only a label. `reference.basis_id` is a deterministic
local fingerprint over that label plus the complete key, seed evidence, and
derived reference orientation. Stable-basis feature states bind this
fingerprint, so reusing a caller ID cannot alias two distinct seed charts. The
fingerprint is an in-process integrity identity, not a wire schema or promised
cross-version persistence format.

The key binds the entire exact immutable camera-to-body calibration,
derotation model, stable model, and chart matrix values, not their IDs alone.

Reference creation requires a complete, all-four-visible, non-clipped fitted
quadrilateral, although Wave 3D still cannot derive an honest local-scale
covariance from the current summaries. Establishment latches orientation and
lineage only. The seed may be paired once with an independently supplied local-
differential feature produced under the exact canonical semantic; no `/1`
feature or covariance is converted. Reuse of the seed frame requires the exact
original evidence context; it cannot be retargeted. The reference orientation
never rolls forward. A later transform must use a distinct newer frame with
strictly advancing camera opaque source time within the fixed generation,
publication and measurement time, causal evidence available at decision,
coherent capture and target IMU sample/source/receipt lineage, and the same
lifecycle key. Because the local feature is supplied independently, a later
usable derotation source need not contain complete finite-quad `/1` scale,
skew, or corner summaries. Any lifecycle change requires explicit retirement
and a fresh filter bootstrap; no posterior is silently re-expressed under a new
reference.

Race-status refresh sequence, race boot time, and authority cutover watermarks
remain checked inside each observation but are not themselves the stable chart
identity. The sequence validator additionally requires those snapshot fields
to obey the complete same-epoch transition semantics: they are monotonic, and
an unchanged `race_status_sequence` requires exactly unchanged
`race_status_boot_ms`. It also requires strictly increasing camera opaque
source time within a generation, publication sequence, publish time, and
measurement time, and nonregressing decision/prediction times. Each capture and
target IMU input may be exactly reused; otherwise its sample sequence, opaque
source time, and host receipt time all advance coherently. Camera and IMU source
times are ordering tokens and are never subtracted from host time.
Innovation rejection or temporary missing evidence withholds an update and
does not move the reference. Estimator gap reinitialization must retire the
reference when integration eventually exists.

The exact public operations are:

```python
establish_stable_reference(
    *, reference_id, tracker_id, track_role, seed_evidence, model
) -> VQ2StableReference

camera_to_stable_local_differential(
    reference, evidence, state, *, camera_time
) -> VQ2StableFeatureTransformEvidence

stable_to_camera_local_differential(
    reference, evidence, state, *, camera_time
) -> VQ2StableFeatureTransformEvidence

validate_stable_measurement_sequence(reference, transforms) -> None
```

`camera_time` is exact enum `CAPTURE` or `TARGET`. Capture basis/time equals
the observation measurement time; target basis/time equals the prediction-
target time. The local semantic, model, derived seed-bound basis ID, host clock,
and covariance identity must match. Sequence validation accepts only camera-
to-stable capture-measurement transforms.

The standalone implementation independently evaluates the small quaternion
propagation and chart kernel from public values. It never imports private
derotation helpers. Compatibility tests cover negative capture alignment and
cross-check point bearings against the existing public derotation evidence.
Integrity validation additionally round-trips every nested source observation
through the public `/1` primitive schema before re-deriving transform fields.

## Evidence boundary and next decision

Tests use synthetic local states, dense covariances, explicit calibration and
attitude evidence, and finite-quadrilateral counterexamples. They can prove the
local projective/rate algebra, inverse, covariance congruence, nuisance growth,
lineage rejection, and immutability. They are not a calibrated camera model,
current `/1` scale conversion, recorded replay, translation model, measured
timing/plant response, runtime/shadow evidence, or powered evidence.

After this standalone primitive is reviewed, the next state-estimation design
decision remains explicit: either introduce a local-scale measurement with
full fitted-homography covariance in a new local/provenance envelope, or retain
finite `/1` semantics by carrying complete quadrilateral shape, rates, and
covariance. Neither may be inferred from this six-state prototype.
