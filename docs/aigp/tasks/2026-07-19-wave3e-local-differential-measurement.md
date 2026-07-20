# Wave 3E local-differential measurement reducer

- Task ID: `vq2-wave3e-local-differential-measurement`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `promotion_verified`
- Objective: add a standalone, immutable reducer from one externally produced,
  rectified, center-gauge-fixed full homography and dense conditional
  covariance to a three-component measurement that shares Wave 3D's local
  scale semantic but is not accepted by or callable from Wave 3D.
- Starting main commit:
  `f8b0e4095a15413bf04601bc5264f12842bdbc66`.
- Branch: `wave3e-local-measurement`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-wave3e-local-measurement`.
- Owner: `/root`.
- Heartbeat date: `2026-07-20`.
- Simulator access: `none`.
- Contract freeze: `c7dcb612318eb9d26868fa1364c1a027d2b8edcd`.
- Review-driven contract correction:
  `aab44d48a032444faeaf5cd1020e90dc9dbd24ed`.

## Owned files

- `estimation/vq2_local_differential_measurement.py`
- `estimation/tests/test_vq2_local_differential_measurement.py`
- `docs/aigp/vq2_local_differential_measurement.md`
- this task record

Promotion policy and trusted-manifest metadata remain integration-owner files
and belong in a separate commit after behavioral review. Shared handoffs are
updated only after integration and post-merge verification.

This tranche does not own or modify `competition/vq2_contracts.py`, the gate
detector or observation adapter, `estimation/vq2_stable_reference.py`, the
current estimator, any runtime/controller/guidance module, supervisor, runner,
transport, or powered surface.

## Why this reducer is the bounded next step

The current fitted observation retains a four-corner mean, but publishes only
a heuristic diagonal covariance over center, finite polygon scale, and two
skew summaries. Their differential has rank at most five over the eight
projective shape degrees of freedom and is generically rank five. At a valid
representative homography, its three-dimensional nullspace contains directions
that change local area density. Therefore the published `5x5` covariance
cannot be lifted uniquely into an honest full-homography or local-scale
covariance.

The required constructive differential test uses

```text
theta = [0.12, -0.08, 0.15, 0.72, 0.09, -0.11, -0.06, 0.55].
```

An independent central-difference Jacobian of `[center_x, center_y,
finite_log_scale, skew_x, skew_y]` has five nonzero singular values
approximately `[2.2121, 1.9941, 1.1674, 1.0000, 0.9825]`. The norm of the
local-log-scale gradient projected into its numerical nullspace is
approximately `0.09570`, not zero. This is T0 differential evidence, not a
calibrated covariance claim.

Wave 3E therefore starts strictly after a future producer has supplied all
eight gauge-fixed homography degrees of freedom and their dense conditional
covariance in a rectified pinhole chart. The reducer proves the remaining
homography-to-local-feature algebra and provenance boundary. It does not claim
that the present detector, camera calibration, or replay evidence can produce
its input.

## Frozen non-wire surface

The module exposes only these immutable local values:

- `VQ2RectifiedHomographyMeasurementModel` freezes the canonical aperture,
  rectified chart, gauge, parameter order, output semantic, numerical envelope,
  and covariance tolerances;
- `VQ2RectifiedHomographyInput` carries declared reducer provenance, exact
  producer/config/calibration identities, one fixed-gauge homography, and a
  dense conditional `8x8` covariance;
- `VQ2RectifiedHomographyMeasurementEvidence` is the indivisible output. It
  retains the complete model and input, deterministic input and derivation
  fingerprints, projected canonical corners, local differential, analytic
  Jacobian, covariance congruence, diagnostics, and the instantaneous camera
  measurement-time `[bearing_x_norm, bearing_y_norm, local_log_scale]` value.
  `validate_integrity()` deeply reconstructs and rederives the result.

The sole operation is:

```python
derive_local_differential_measurement(
    source: VQ2RectifiedHomographyInput,
    *,
    model: VQ2RectifiedHomographyMeasurementModel,
) -> VQ2RectifiedHomographyMeasurementEvidence
```

There is no mutable producer, association, sequence, rate, state, filter,
stable-transform, adapter, or runtime API.

The exact semantic constants are:

```text
LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID = "vq2-local-differential-area-v1"
CANONICAL_APERTURE_MODEL_ID = "vq2-canonical-square-aperture-v1"
RECTIFIED_CHART_MODEL_ID = "vq2-rectified-camera-frd-slope-chart-v1"
HOMOGRAPHY_GAUGE_MODEL_ID = "vq2-center-forward-h00-one-v1"
CANONICAL_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
CANONICAL_CORNERS = ((-1.0,-1.0), (1.0,-1.0), (1.0,1.0), (-1.0,1.0))
HOMOGRAPHY_PARAMETER_ORDER =
    ("h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")
LOCAL_MEASUREMENT_ORDER =
    ("bearing_x_norm", "bearing_y_norm", "local_log_scale")
```

The exact public aliases, errors, enum, and frozen dataclass annotations are:

```python
Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Vector8 = tuple[float, float, float, float, float, float, float, float]
Matrix2 = tuple[Vector2, Vector2]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Matrix8 = tuple[Vector8, Vector8, Vector8, Vector8,
                Vector8, Vector8, Vector8, Vector8]
Matrix3x8 = tuple[Vector8, Vector8, Vector8]
ProjectedQuad = tuple[Vector2, Vector2, Vector2, Vector2]

class VQ2LocalDifferentialMeasurementError(ValueError): ...
class VQ2LocalDifferentialProvenanceError(
    VQ2LocalDifferentialMeasurementError
): ...
class VQ2LocalDifferentialGeometryError(
    VQ2LocalDifferentialMeasurementError
): ...
class VQ2LocalDifferentialCovarianceError(
    VQ2LocalDifferentialMeasurementError
): ...

class VQ2HomographyCovarianceScope(str, Enum):
    CONDITIONAL_FIT = "conditional_fit"

@dataclass(frozen=True, slots=True)
class VQ2RectifiedHomographyMeasurementModel:
    model_id: str
    local_feature_model_id: str
    canonical_aperture_model_id: str
    canonical_corner_order: tuple[str, str, str, str]
    rectified_chart_model_id: str
    homography_gauge_model_id: str
    homography_parameter_order: tuple[str, str, str, str, str, str, str, str]
    image_size_px: tuple[int, int]
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: str | None
    geometry_producer_model_id: str
    geometry_producer_config_sha256: str
    homography_fit_model_id: str
    rectification_model_id: str
    rectification_calibration_id: str
    rectification_calibration_sha256: str
    homography_covariance_model_id: str
    output_covariance_model_id: str
    max_measurement_uncertainty_ns: int
    minimum_forward: float
    minimum_local_area_determinant: float
    minimum_projected_edge_length: float
    minimum_projected_corner_cross: float
    max_homography_condition: float
    max_local_differential_condition: float
    max_abs_center_bearing_norm: float
    max_abs_projected_corner_bearing_norm: float
    max_abs_local_log_scale: float
    max_input_variance: float
    max_output_variance: float
    covariance_psd_tolerance: float

@dataclass(frozen=True, slots=True)
class VQ2RectifiedHomographyInput:
    frame_timing: FrameTimingV1
    authority: GateAuthorityEpochV1
    measurement_time_monotonic_ns: int
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: str | None
    measurement_uncertainty_ns: int
    candidate_id: str
    image_size_px: tuple[int, int]
    canonical_aperture_model_id: str
    rectified_chart_model_id: str
    homography_gauge_model_id: str
    geometry_producer_model_id: str
    geometry_producer_config_sha256: str
    homography_fit_model_id: str
    rectification_model_id: str
    rectification_calibration_id: str
    rectification_calibration_sha256: str
    homography_covariance_model_id: str
    homography_covariance_scope: VQ2HomographyCovarianceScope
    homography_parameter_order: tuple[str, str, str, str, str, str, str, str]
    homography: Matrix3
    homography_covariance: Matrix8

@dataclass(frozen=True, slots=True)
class VQ2RectifiedHomographyMeasurementEvidence:
    model: VQ2RectifiedHomographyMeasurementModel
    source: VQ2RectifiedHomographyInput
    feature_model_id: str
    rectified_chart_model_id: str
    host_clock_id: str
    measurement_time_monotonic_ns: int
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: str | None
    measurement_uncertainty_ns: int
    covariance_model_id: str
    covariance_scope: VQ2HomographyCovarianceScope
    input_fingerprint_sha256: str
    derivation_fingerprint_sha256: str
    measurement_order: tuple[str, str, str]
    values: Vector3
    projected_canonical_corners: ProjectedQuad
    local_differential: Matrix2
    local_area_jacobian_determinant: float
    measurement_jacobian: Matrix3x8
    canonical_homography_covariance: Matrix8
    conditional_covariance: Matrix3
    minimum_canonical_forward: float
    homography_condition: float
    local_differential_condition: float
    minimum_projected_edge_length: float
    minimum_projected_corner_cross: float
    maximum_projected_abs_bearing: float

    def validate_integrity(self) -> None: ...
```

All tuple shapes, enum instances, nested public contracts, and integers use
exact types. Numeric scalar fields permit exact `int` or `float` but reject
`bool`, then store finite `float` values. SHA-256 identities and both derived
fingerprints are exactly 64 lowercase hexadecimal characters.

### Exact numerical policy

The implementation constants and non-relaxable limits are:

```text
HARD_MAX_MEASUREMENT_UNCERTAINTY_NS = 200_000_000
HARD_MIN_FORWARD = 1e-6
HARD_MIN_LOCAL_AREA_DETERMINANT = 1e-12
HARD_MIN_PROJECTED_EDGE_LENGTH = 1e-6
HARD_MIN_PROJECTED_CORNER_CROSS = 1e-12
HARD_MAX_HOMOGRAPHY_CONDITION = 1e6
HARD_MAX_LOCAL_DIFFERENTIAL_CONDITION = 1e6
HARD_MAX_ABS_CENTER_BEARING_NORM = 4.0
HARD_MAX_ABS_PROJECTED_CORNER_BEARING_NORM = 4.0
HARD_MAX_ABS_LOCAL_LOG_SCALE = 20.0
HARD_MAX_INPUT_VARIANCE = 1e6
HARD_MAX_OUTPUT_VARIANCE = 1e6
HARD_MAX_COVARIANCE_PSD_TOLERANCE = 1e-10
```

`max_measurement_uncertainty_ns` is an exact positive integer no greater than
its hard maximum. Every `minimum_*` model value is finite, positive, and no
smaller than its corresponding hard minimum; increasing it only tightens
admission. The two condition-number caps are finite and in
`[1.0, corresponding hard maximum]`, because a spectral condition number
cannot be below one. Every other `max_*` numeric value is finite, positive, and
no greater than its corresponding hard maximum; decreasing it only tightens
admission.
`covariance_psd_tolerance` is finite and in
`(0, HARD_MAX_COVARIANCE_PSD_TOLERANCE]`; decreasing it only tightens
admission. A model cannot relax any hard limit.

The fixed semantic/order fields must equal the constants above. The model's
image size, time basis/model, producer model/config hash, fit model,
rectification model/calibration ID/hash, and homography covariance model must
equal the corresponding input fields exactly. The model accepts only the two
camera measurement-time bases. A calibrated basis requires a non-`None` model
ID; a final-packet proxy requires `None`. The input scope and evidence scope
are exact `VQ2HomographyCovarianceScope.CONDITIONAL_FIT`. The output covariance
identity is `model.output_covariance_model_id`; it is available only through
the indivisible evidence and never inferred from the input ID.

For either an input or derived covariance `P`, define

```text
scale = max_ij(abs(P[i,j]))
tolerance = model.covariance_psd_tolerance * scale.
```

Strictly positive marginals make `scale > 0`. Reject nonfinite data, any
diagonal `<= 0`, or any marginal above the applicable input/output variance
limit. The standalone input carrier retains its exact submitted `P_H`, after
only the hard-tolerance structural check, so a later reducer model can enforce
a tighter tolerance and so its input fingerprint covers the submitted matrix.
During derivation, reject when `max(abs(P_H-P_H^T))` exceeds the active model
tolerance; otherwise expose `(P_H+P_H^T)/2` separately as
`canonical_homography_covariance`. On that symmetrized matrix, reject when
`lambda_min(P) < -tolerance`. A negative eigenvalue within tolerance is
accepted as numerical roundoff and retained; it is never clipped, floored, or
represented as exactly positive semidefinite. There is no artificial
covariance floor.
The congruence uses only `canonical_homography_covariance`; its result is
subjected to the same symmetry/PSD and strictly positive output-marginal checks
before storage. Evidence retains both the exact raw source and canonical
derived view, and integrity rederives the latter. A stricter model must reject
raw asymmetry that a looser model would admit; input construction must not
erase that evidence.

Evidence rederivation uses the same deterministic in-process operations and
requires fingerprints, enums/orders, derived floating values, vectors, and
matrices to match exactly. There is no evidence-comparison tolerance. This
proof-only local value is not a cross-platform or wire serialization contract;
any low-level mutation, including one smaller than covariance admission
tolerance, must fail `validate_integrity()`.

## Exact source and lifecycle binding

`VQ2RectifiedHomographyInput` contains:

- an exact `FrameTimingV1` and `GateAuthorityEpochV1`;
- measurement host time, camera measurement-time basis/model, and positive
  uncertainty;
- source image size and candidate ID;
- exact canonical-aperture, rectified-chart, and homography-gauge IDs;
- geometry-producer model ID and SHA-256 config identity;
- homography-fit model ID;
- rectification model and calibration artifact IDs plus the calibration
  artifact's SHA-256 content identity;
- homography-covariance model ID and exact `CONDITIONAL_FIT` scope;
- the exact homography parameter order; and
- the fixed-gauge homography and dense `8x8` covariance.

The frame host clock, stream, and generation must equal the authority values.
Publication and publish time use inclusive cutover boundaries:
`publication_sequence >= frame_publication_sequence_not_before` and
`publish_monotonic_ns >= frame_publish_monotonic_ns_not_before`. The only
accepted time bases are calibrated camera capture and camera-final-packet
proxy. Calibrated capture requires a non-`None` bounded model ID and positive
uncertainty. Final-packet proxy requires model ID `None`, positive uncertainty,
and exact equality between measurement time and
`frame_timing.final_unique_packet_monotonic_ns`. Either measurement time must
be no later than frame publication, and uncertainty must not exceed the model
limit. Model and input time basis/model must match exactly.

Nested public timing and authority values are round-tripped through their
public primitive codecs so a low-level mutation cannot become trusted local
evidence. Authority/cutover checks are implemented independently from those
public fields; the module must not import `_validate_authority_frame_cutover`
or any other private contract helper.

The input fingerprint covers every model-independent source field, all nested
timing and authority values, every homography and covariance value, and all
content hashes. The derivation fingerprint separately covers the complete
exact model plus the complete exact input. Both are rederived during evidence
integrity validation so the same source admitted under a different numerical
or semantic envelope has a different derivation identity. They are in-process
integrity identities, not wire schemas, signatures, or proof that a caller
supplied truthful calibration or covariance.

The evidence is indivisible: consumers must retain the complete evidence and
must not detach `values` or `conditional_covariance` from its timing, frame,
candidate, authority, model, and fingerprint provenance. Its measurement order
equals `LOCAL_MEASUREMENT_ORDER`; its semantic, chart, and output covariance
model are exactly those in the retained model; its host clock, camera
measurement time, basis/model/uncertainty, frame, and candidate are exactly
those in the retained source; its covariance scope is exactly
`CONDITIONAL_FIT`; and `conditional_covariance` is exactly the rederived
`J P_H_canonical J^T`, where `P_H_canonical` is the separately retained
active-model-admitted symmetric view of the raw source covariance.

`geometry_producer_config_sha256` covers every geometry-producer,
homography-fit, and covariance-setting value, not merely a detector subset.
The fit-model and covariance-model IDs name the algorithms; the configuration
content hash binds their complete parameterization.

No function accepts `GateObservationV1`, `FeatureCovarianceV1`, fitted corners,
`VQ2DerotationEvidence`, `VQ2LocalDifferentialFeatureState`, or
`RelativeGateStateV1`. This prevents accidental conversion of current `/1`
summaries or border-normalized JPEG coordinates into rectified evidence.

## Frozen homography convention

The canonical aperture is `[-1,1]^2` with coordinates ordered
top-left, top-right, bottom-right, bottom-left and axes right/down. The input
homography maps a canonical point `[1,a,b]^T` to a rectified camera-FRD ray:

```text
[X,Y,Z]^T = H [1,a,b]^T
bearing = [u,v] = [Y/X,Z/X].
```

The gauge and covariance parameterization are exact:

```text
H = [[1,   h01, h02],
     [h10, h11, h12],
     [h20, h21, h22]]

theta = [h01,h02,h10,h11,h12,h20,h21,h22].
```

`H[0,0]` must equal exactly `1.0`. The reducer never rescales or silently
normalizes a homography because doing so without transforming its covariance
would change the declared random variables. A gauge-ambiguous `9x9`
covariance is not accepted.

At the canonical center:

```text
u = h10
v = h20

L = [[h11 - h10*h01, h12 - h10*h02],
     [h21 - h20*h01, h22 - h20*h02]]

delta = det(L)
local_log_scale = log(2) + 0.5*log(delta).
```

The local scale is exactly the distinct
`vq2-local-differential-area-v1` semantic. It is not frozen `/1` finite polygon
area except in the affine special case.

## Analytic Jacobian and covariance

For the parameter order above, the bearing rows of the `3x8` Jacobian are
selectors for `h10` and `h20`. Let

```text
A = h11 - h10*h01
B = h12 - h10*h02
C = h21 - h20*h01
D = h22 - h20*h02
delta = A*D - B*C.
```

The derivative of `delta` is frozen as:

```text
d_delta = [
    -h10*D + B*h20,
    -A*h20 + h10*C,
    -h01*D + h02*C,
    D,
    -C,
    -A*h02 + B*h01,
    -B,
    A,
].
```

The local-scale row is `0.5*d_delta/delta`. For raw dense conditional
homography covariance `P_H`, let the active-model-admitted canonical view be
`P_H_canonical = (P_H + P_H^T)/2`. The only output covariance is the
first-order conditional congruence

```text
P_local = J P_H_canonical J^T.
```

Every off-diagonal term is retained. No calibration, timing, detector-model,
common-mode, nonlinear-remainder, attitude, or sequential nuisance is added or
called independent noise. There is no `TOTAL` output. The covariance remains
conditional on the exact rectification/calibration and fit model named by the
input, and it is a linearized covariance rather than a coverage guarantee.

The covariance scope is a caller assertion. Deep fingerprints and integrity
checks prevent accidental mutation/relabeling of a returned object, but a pure
stateless reducer cannot detect a newly constructed dishonest input. Future
sequential use requires an approved producer and an augmented/Schmidt or other
reviewed shared-nuisance treatment.

## Fail-closed geometry and numerical envelope

The reducer rejects before emitting evidence when any of these holds:

- wrong exact type, token, SHA-256 content identity, enum, dimension,
  parameter order, gauge, or feature semantic;
- nonfinite values, booleans used as numbers, malformed nested public values,
  authority/frame mismatch, invalid timing semantics, or excessive timing
  uncertainty;
- a covariance asymmetric beyond tolerance or materially indefinite below
  negative tolerance, nonpositive marginal variance, or variance outside the
  model envelope;
- a canonical-corner forward denominator below the model minimum;
- nonpositive local area determinant, wrong projected corner order,
  nonconvex/self-intersecting output, or a collapsed projected edge;
- homography or local-differential conditioning outside the model envelope;
  or
- projected bearing, center bearing, local scale, or output covariance outside
  explicit hard/model bounds.

The four canonical corner denominators are sufficient for positivity over the
whole square because the denominator is affine in `(a,b)`. All inputs and
derived fields are finite and deterministically rederived. PSD and symmetry
tolerances scale with the actual matrix, not an implicit unit scale.

Diagnostics use exact definitions. With projected corners `p_i=(u_i,v_i)` in
the frozen cyclic order and indices modulo four:

```text
minimum_canonical_forward = min_i X_i
minimum_projected_edge_length = min_i ||p_(i+1) - p_i||_2
corner_cross_i = cross_2d(p_(i+1)-p_i, p_(i+2)-p_(i+1))
minimum_projected_corner_cross = min_i corner_cross_i
maximum_projected_abs_bearing = max_i(max(abs(u_i), abs(v_i)))
homography_condition = sigma_max(H) / sigma_min(H)
local_differential_condition = sigma_max(L) / sigma_min(L)
```

`cross_2d((x1,y1),(x2,y2)) = x1*y2 - y1*x2`. In the x-right,
y-down convention, the frozen top-left/top-right/bottom-right/bottom-left
ordering has positive cross values. Edge length is in normalized-bearing units
and corner cross is in squared normalized-bearing units. Both condition
numbers are spectral two-norm/SVD condition numbers.

## Explicitly excluded and blocked

Wave 3E must not:

- consume current `/1` covariance or infer `8x8` covariance from RMS, inlier
  counts, half-pixel floors, diagonal independence, bootstrap guesses, or a
  square prior;
- derive a homography from current `/1` corners or treat image-border
  normalization as camera-ray calibration;
- accept arbitrary homography scaling or a `9x9` covariance;
- create a detached measurement, zero rates, finite differences, a six-state
  object, or a finite-quad state;
- invoke or feed Wave 3D, the current estimator, controller, runtime,
  supervisor, transport, or simulator; or
- claim production, replay, calibration, association, timing, covariance
  coverage, or powered evidence.

An actual producer remains blocked on calibrated image-to-ray rectification,
an independently reviewed full-homography fit covariance, explicit shared
calibration/model nuisance, and recorded replay coverage. Filter use remains
blocked on cross-frame association, rate/state design, sequential nuisance
cross-covariance, estimator integration, and runtime/authority review.

## Required verification

Direct tests must cover:

- exact dataclass types, tokens, hashes, enums, dimensions, parameter order,
  gauge, bounds, booleans, nonfinite values, and nested corruption;
- every exact hard-limit boundary and tighten-only model direction;
- frame/authority/candidate/time/producer/config/calibration/covariance binding,
  including exact calibrated/proxy rules and inclusive authority watermarks;
- fingerprints changing when any homography, covariance, producer config,
  calibration artifact, or reducer-model value changes;
- affine equality with finite polygon scale and perspective disagreement with
  `/1` finite area;
- the documented five-summary rank/nullspace counterexample;
- independent projective-oracle values across randomized well-conditioned
  homographies;
- denominator, orientation, convexity, edge collapse, determinant, condition,
  bearing, scale, and covariance guards;
- the analytic `3x8` Jacobian against central finite differences across a
  randomized matrix;
- dense covariance congruence, off-diagonal influence, scale-separated
  symmetry/PSD cases, raw-input preservation, active-model tolerance before
  canonical symmetrization, stricter-model rejection, retained tiny-negative
  roundoff, rejection below tolerance, zero-marginal rejection, no-floor
  behavior, sub-tolerance covariance/derived-field tamper rejection, and exact
  deterministic integrity rederivation;
- rejection/non-acceptance of `/1`, corner, rate, state, derotation, and total
  covariance inputs; and
- static proof that no production module imports or calls the reducer, no
  private contract helper is imported, and frozen public `/1` primitive codecs
  remain unchanged.

The focused compatibility matrix is the direct test plus:

- `estimation/tests/test_vq2_stable_reference.py`
- `estimation/tests/test_vq2_imu_derotation.py`
- `estimation/tests/test_vq2_relative_estimator.py`
- `gate_detection/tests/test_vq2_detector.py`
- `competition/tests/test_vq2_contracts.py`

After behavioral review, run canonical `test-vq2`, `test-fast`, and `test-unit`.
Run `test-full-non-live` only at the promotion boundary. Promotion must add only
the new test path to the exact policy/trusted manifest, verify every digest,
and retain the no-wiring boundary. No simulator, preflight, external network,
replay, reset, arm/disarm, target, transport, shadow/runtime, or powered action
is permitted for this task.

## Behavioral evidence

The pure reducer and its dedicated direct suite are implemented in the two
owned estimation files with no production call site. Accepted candidate
evidence before promotion is:

- `224` direct tests;
- `450` focused compatibility tests across the reducer, stable-reference,
  derotation, relative estimator, detector, and frozen contracts;
- exactly `1,325` canonical VQ2 tests;
- `test-fast`: `2,420` passed, `20` skipped, `42` deselected;
- `test-unit`: `2,420` passed, `20` skipped, `42` deselected; and
- three independent final reviews clearing lifecycle/provenance/no-wiring,
  API/test coverage, and mathematics/numerics.

The mathematical re-review independently exercised `2,000` admitted random
homographies, `11` exact one-ULP boundaries, `260` raw/canonical covariance
cases from `1e-250` through `1e6`, and tolerated/material negative modes. The
largest reported value, scaled finite-difference Jacobian, finite-difference
covariance, and `det(H)-det(L)` errors were respectively `4.44e-16`,
`2.11e-10`, `5.88e-10`, and `4.44e-16`.

Review found and closed two evidence-integrity bugs before behavioral freeze:
raw covariance is now retained and fingerprinted until the active model
admits a separate canonical symmetric view, and exact stored float types plus
bit-exact same-process rederivation now reject one-ULP or equal-valued
float-to-int low-level mutation.

No detector producer, `/1` conversion, corner covariance, rate/state object,
Wave 3D call, estimator/runtime/controller/supervisor/transport wiring,
simulator, preflight, external network, replay, reset, arm/disarm, target,
shadow/runtime, or powered action contributed to this evidence.

## Promotion review

The accepted behavioral implementation is commit
`ceed9c854b0066d4f00d4add796fb968d449593a`. Promotion changed the canonical
VQ2 policy only from `1,101` to `1,325` expected passes and added only
`estimation/tests/test_vq2_local_differential_measurement.py`; the policy
inventory remains sorted and unique. The policy file identity is
`7daa46ec4dfd025c18f12076add06d70b6463f07d6320b20487a63bd78d0851e`
and its canonical semantic identity is
`b8bc5228b12eafc75c10b3d2aa658cfe57a0d1ed820b3fefa6e0317d7c5cdc90`.

The trusted manifest advanced from `128` to `129` files. It added the new
test digest
`683aa081103e6e9ae22281b1e1f573bc821218f57df75cdfa688587b0ad84382`,
changed only the policy digest, and removed nothing. Independent strict
review found all `129/129` files present, regular, and digest-matching. The
manifest file identity is
`e88363ef096bba83fe4660a4903abb6ae063f41682246b38ba9c69481008fffc`
and its canonical semantic identity is
`46e77cbbe8a131517444b141293b1fe8c2bab546a6f5630f711ffe0d621d5ea2`.
Canonical semantic identities are SHA-256 over parsed JSON serialized with
sorted keys and compact separators.

Observed promotion evidence is:

- direct: `224` passed;
- focused compatibility: `450` passed;
- canonical VQ2: `1,325` passed;
- cache-clean isolated hash-pinned VQ2: `1,325` passed;
- `test-fast`: `2,420` passed, `20` skipped, `42` deselected;
- `test-unit`: `2,420` passed, `20` skipped, `42` deselected; and
- `test-full-non-live`: `2,461` passed, `21` skipped in `487.69s`.

The full non-live boundary was rerun to an observed pytest summary after two
shorter outer shell ceilings expired; neither interrupted attempt produced a
test failure, and no orphaned runner remained. Promotion used no simulator,
preflight, external network, replay, reset, arm/disarm, target, transport,
shadow/runtime, or powered surface. Post-merge status and the integrated main
commit remain to be recorded after fast-forward and canonical verification.
