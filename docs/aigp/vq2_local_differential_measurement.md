# VQ2 rectified-homography local measurement

## Purpose and non-authority

This Wave 3E module is a pure, non-wire reducer for one externally supplied
rectified homography. It converts a complete eight-degree-of-freedom
projective shape and its dense conditional covariance into indivisible
evidence containing the instantaneous three-component local-differential
camera measurement-time value

```text
[bearing_x_norm, bearing_y_norm, local_log_scale].
```

It does not obtain a homography from the current detector, rectify pixels,
estimate rates, create an estimator state, perform association, call the Wave
3D stable transform, or feed any production path. There is no supervisor,
command, runner, transport, simulator, reset, cleanup, or powered authority.

The frozen identifiers are

```text
feature:    vq2-local-differential-area-v1
canonical:  vq2-canonical-square-aperture-v1
chart:      vq2-rectified-camera-frd-slope-chart-v1
gauge:      vq2-center-forward-h00-one-v1
```

The chart identifier means that coordinates are already rectified camera-FRD
ray slopes. It does not authorize treating border-normalized image coordinates
as slopes. The exact rectification model, calibration artifact ID, and content
SHA-256 remain mandatory independent provenance.

## Why current `/1` evidence is not an input

The frozen `/1` observation covariance covers five summaries: center, finite
fitted-quadrilateral scale, and two skew values. A general homography has eight
shape degrees of freedom after gauge fixing. The five-summary differential has
rank at most five and is generically rank five. A documented representative
homography has a three-dimensional numerical nullspace whose projection of the
local-scale gradient is nonzero. Consequently neither the current `5x5`
covariance nor aggregate fit residuals can be expanded uniquely into an honest
homography covariance.

The reducer therefore accepts neither `GateObservationV1` nor fitted corners.
Its input begins after a separately reviewed producer has supplied a
center-gauge homography and dense `8x8` conditional covariance in rectified
camera-FRD slope coordinates. Such a producer does not yet exist in this
repository.

## Coordinate and gauge convention

Canonical aperture coordinates use the square `[-1,1]^2`, with corners in
top-left, top-right, bottom-right, bottom-left order and axes right/down. For
canonical coordinate `(a,b)`,

```text
[X,Y,Z]^T = H [1,a,b]^T
u = Y/X
v = Z/X.
```

The exact center-forward gauge is

```text
H = [[1,   h01, h02],
     [h10, h11, h12],
     [h20, h21, h22]],
```

and covariance uses the fixed parameter order

```text
[h01,h02,h10,h11,h12,h20,h21,h22].
```

The module requires `H[0,0] == 1.0` and never normalizes caller input. A
homography's arbitrary scalar is not a ninth stochastic degree of freedom;
accepting a rescaled matrix without the matching covariance transform would
silently change the model.

## Local measurement

At the canonical center, define

```text
u = h10
v = h20

L = [[h11 - h10*h01, h12 - h10*h02],
     [h21 - h20*h01, h22 - h20*h02]]

delta = det(L).
```

For positive `delta`,

```text
local_log_scale = log(2) + 0.5*log(delta).
```

This is one half of the logarithm of the local normalized-image area density
of the canonical aperture element. It uses the exact
`vq2-local-differential-area-v1` semantic introduced by Wave 3D. For an affine
quadrilateral it equals one half of the log finite polygon area; for a general
perspective quadrilateral it intentionally differs.

## Analytic `3x8` Jacobian

Write

```text
A = h11 - h10*h01
B = h12 - h10*h02
C = h21 - h20*h01
D = h22 - h20*h02
delta = A*D - B*C.
```

The first two Jacobian rows select `h10` and `h20`. In the frozen parameter
order, the derivative of `delta` is

```text
[-h10*D + B*h20,
 -A*h20 + h10*C,
 -h01*D + h02*C,
 D,
 -C,
 -A*h02 + B*h01,
 -B,
 A].
```

The third row is this vector multiplied by `0.5/delta`. Direct central finite
differences and independent projective oracles are required acceptance
evidence.

## Conditional covariance only

For dense gauge-fixed homography covariance `P_H`, the reducer returns

```text
P_local = J P_H J^T.
```

This is a first-order conditional covariance. It conditions on the exact
rectification/calibration artifact, fit model, producer configuration, and
timing declarations identified by the input. It does not include calibration,
timing, detector-model, common-mode, nonlinear-remainder, attitude, or
sequential uncertainty. No diagonal shortcut, artificial variance floor, or
`TOTAL` label is permitted.

The input and output covariance scope is exactly `CONDITIONAL_FIT`. The scope
and content hashes are caller assertions: immutable values, fingerprints, and
deep rederivation catch accidental corruption but cannot prove that an
external producer supplied calibrated or statistically valid evidence.
Repeated measurements may share calibration and model error, so future filter
use must retain the corresponding nuisance state or a separately proved
cross-covariance treatment.

## Provenance boundary

The input independently carries exact frame timing, gate authority, candidate,
measurement-time semantics, image size, producer/config, homography-fit,
rectification/calibration, and covariance-model identity. Public timing and
authority objects are deep-round-tripped, and their host clock, stream,
generation, publication, and cutover fields must agree.

The reducer model repeats and exactly matches the input image size, time
basis/model, producer/config hash, fit model, rectification model/calibration
ID/hash, and homography covariance model. Inclusive authority watermarks accept
equality. Calibrated camera time requires its named model; final-packet proxy
requires no model and exact equality with the frame's final-packet host time.
These checks are reimplemented from public fields without importing private
contract helpers.

This is declared reducer provenance, not complete perception lineage. The
envelope deliberately does not retain a source observation or image, clipping
or visibility, detector health, support/inlier counts, residuals, or fit
quality. Those omissions are another reason this proof-only result cannot by
itself be admitted as an estimator measurement. All candidate, algorithm, and
content-hash fields remain caller assertions.

SHA-256 content identities bind the producer configuration and rectification
calibration artifacts. They identify exact external content; they do not make
that content available, validate its rules compliance, or establish its
physical accuracy. The local input fingerprint additionally covers the entire
source value, homography, and covariance. A separate derivation fingerprint
covers the complete model plus complete input, so changing a numerical or
semantic admission envelope changes the result identity. They are integrity
identities, not wire schemas or signatures.

The module deliberately has no detached measurement dataclass. Values and
conditional covariance remain inside the complete evidence with their exact
frame, candidate, authority, measurement-time basis/model/uncertainty, model,
and fingerprints. Consumers must accept or reject that complete evidence as a
unit. The geometry-producer configuration content hash covers every producer,
homography-fit, and covariance-setting value; algorithm IDs alone are not
treated as exact configuration binding.

## Geometry guards

The model requires positive forward denominator over the complete canonical
square, positive local orientation/area, a convex order-preserving projected
quad with noncollapsed edges, bounded homography and local-differential
condition numbers, bounded projected bearings and local scale, and bounded
input/output variances. Because the projective denominator is affine in
canonical coordinates, checking all four square corners proves its minimum
over the square.

The exact canonical corner tuples are `(-1,-1)`, `(1,-1)`, `(1,1)`, and
`(-1,1)`. Homography and local-differential condition numbers are spectral
two-norm/SVD condition numbers. Projected edge length is Euclidean. Convexity
uses the minimum cyclic signed cross
`cross(p[i+1]-p[i], p[i+2]-p[i+1])`; it is positive for the frozen corner order
in the x-right/y-down chart and has squared normalized-bearing units.

All values must be finite. Covariances must be exact-size, symmetric within
tolerance, numerically positive-semidefinite with no eigenvalue below negative
tolerance, have positive marginal variances, and remain within explicit model
bounds. Symmetry and PSD tolerances are relative to the covariance's actual
scale. Evidence integrity rederives every output and diagnostic from the
retained immutable input and model.

More exactly, covariance scale is the maximum absolute matrix entry and the
admissible roundoff is that scale times the model tolerance, which itself
cannot exceed `1e-10`. Asymmetry within tolerance is averaged; greater
asymmetry is rejected. An eigenvalue below negative tolerance is rejected. A
negative eigenvalue within tolerance remains unmodified numerical roundoff:
the reducer never clips eigenvalues or adds a variance floor. Input and output
marginals must remain strictly positive and at or below their explicit caps.

Hard limits cap measurement-time uncertainty at `200,000,000 ns`, both
spectral condition numbers at `1e6`, absolute center/corner bearings at `4`,
absolute local log scale at `20`, input/output marginal variance at `1e6`, and
the covariance tolerance at `1e-10`. Non-relaxable geometry floors are `1e-6`
for forward denominator and projected edge, and `1e-12` for local determinant
and signed corner cross. A concrete model may only tighten those limits.

## Promotion boundary

Synthetic tests prove only the reducer's T0 mathematics and lifecycle guards.
They do not prove a detector producer, calibrated pixel-to-ray mapping,
covariance coverage, replay performance, association, rates, estimator
behavior, runtime timing, or authority integration.

Before production use, a separate tranche must provide and independently
validate the full-homography covariance source and rectification calibration
against approved recorded replay. A later filter design must address
cross-frame association, rate/state construction, shared nuisance
cross-covariance, estimator integration, and runtime/authority review. Powered
work remains separately authorized under the authoritative VQ2 handoff.
