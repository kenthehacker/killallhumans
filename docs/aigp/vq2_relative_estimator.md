# VQ2 feature-space relative estimator

`estimation/vq2_relative_estimator.py` is the offline Wave 1 estimator seam for
build 3385. It consumes a frozen `GateObservationV1` and emits a frozen
`RelativeGateStateV1`; it does not change either `/1` wire schema.

## State and measurement

The filter state is the constant-velocity feature vector

```text
[bearing_x, bearing_y, log_scale,
 bearing_rate_x, bearing_rate_y, expansion_rate]
```

The measurement is the fitted inner-aperture
`[center_x_norm, center_y_norm, log_scale]`. `log_scale` retains its contract
meaning: the natural logarithm of the square root of fitted aperture area in
normalized image units. A legacy bbox-only observation has no interchangeable
scale measurement and is withheld with `MissingApertureScaleError`.

The implementation uses a variable-time constant-velocity Kalman prediction,
a Joseph-form covariance update, a positive-semidefinite covariance floor, and
a three-dimensional normalized-innovation-squared gate. Source confidence,
clipping, and degraded observation health conservatively inflate measurement
covariance. Feature rates and contract-bounded predicted bearings are clipped
only with explicit degraded/unhealthy health and covariance inflation.

No metric position, velocity, or gate orientation is inferred. All metric
fields remain absent.

## Timing and source rules

`RelativePredictionTarget` carries an exact host-clock identity plus decision
and prediction times. The target clock must match both the observation and the
estimator instance. A normal accepted update is bounded in two ways:

- prediction minus decision cannot exceed `max_prediction_horizon_s`; and
- prediction minus the accepted measurement time cannot exceed that same
  reviewed horizon.

The second bound prevents a late decision from disguising arbitrarily old
evidence as a short prediction. The estimator applies at most one measurement
update for one distinct `(host_clock_id, FrameIdentityV1)`. Publication
sequence, publication time, measurement time, decision time, and prediction
time cannot regress. Validation failures are transactional: the frame is not
committed and a corrected request may retry it.

The emitted timing and visibility fields are copied from the exact accepted
source observation and are checked with
`validate_relative_gate_state_source`. The emitted trace is checked with
`validate_relative_gate_state_sequence`.

## Rejection and dropout behavior

An accepted distinct observation advances `measurement_update_sequence`.
An innovation rejected by the NIS gate does not. The local
`RelativeEstimatorUpdate` reports the rejected candidate and its NIS, while
the wire state coasts from the last accepted source. Therefore the wire state
does not falsely cite the rejected frame or claim that its measurement was
applied.

At control rate, `coast(target)` predicts from the last accepted posterior. It
keeps the source frame, candidate, authority, and measurement-update sequence;
advances state sequence and dropout count; grows covariance; transitions from
`COASTING` to `LOST`; and fails closed after `max_coast_s`. Every requested
decision-to-prediction delay still obeys the short prediction-horizon bound.

## Authority lifecycle

The estimator only echoes the safety-issued authority in its source
observation. A forward reset or gate/index authority change within the bound
safety session reinitializes the filter from the first post-cutover fitted
observation and restarts the per-authority state/update sequences at zero.
Seen-frame and monotonic replay history are retained across those transitions.

There is intentionally no unguarded public reset method: erasing that history
could admit old frames. Process/session replacement owns construction of a new
estimator instance. The estimator never issues reset, passage, arm, target, or
command authority.

## Offline verification

The dedicated test module covers irregular frame timing, future-error
reduction, NIS rejection, deterministic coasting/loss, covariance
positive-semidefiniteness, clock and horizon mismatches, bbox-scale
withholding, confidence/clipping inflation, rate clipping, long-gap
reinitialization, duplicate and replayed frames, gate/reset cutovers, and
deterministic replay equivalence.

```powershell
.\scripts\dev.cmd test-target estimation/tests/test_vq2_relative_estimator.py
.\scripts\dev.cmd test-target competition/tests/test_vq2_contracts.py
.\scripts\dev.cmd test-vq2
```
