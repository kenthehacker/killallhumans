# VQ2 pure predictive controller

## Boundary

`competition/vq2_controller.py` is a deterministic, offline controller with
this effective interface:

```text
RelativeGateStateV1
  + ControllerAttitudeInput
  + ControllerTickInput
  + ControllerPhaseInput
  + PredictiveControllerConfig
    -> exact CommandProposalV1
```

The module imports only the frozen VQ2 contract layer and Python pure-math
utilities. It cannot connect to FlightSim, arm, reset, approve, project to the
legacy transport DTO, send, declare passage, or advance gate authority. A
proposal remains untrusted intent until the external safety supervisor checks
it and issues a separate `SupervisorApprovedCommandV1`.

The local tick input echoes caller-owned proposal/tick IDs, host-monotonic
times, exact expected `GateAuthorityEpochV1`, minimum state-decision/state-
sequence watermarks, the safety-owned expected phase start, and a minimum
guidance/safety evaluation-time watermark. The phase input carries the same
host-clock identity, a stable `phase_started_monotonic_ns`, the objective's
`evaluation_monotonic_ns`, the reviewed mode, initial Gate 0 pitch basis,
guidance target bearing, and paired objective-permission/withholding fields.
The controller requires an exact phase-clock/start echo and
`start <= evaluation <= proposal`, rejects an evaluation below the tick
watermark, and applies a tighten-only 100 ms objective-age ceiling. It computes
phase elapsed time from `proposal_monotonic_ns - phase_started_monotonic_ns`;
there is no caller-supplied elapsed duration to renew.

The pure controller can compare these local values but cannot authenticate
that safety owns them, and frozen `CommandProposalV1` has no fields that bind
phase start or objective evaluation provenance. The integration adapter and
supervisor must validate the guidance decision's exact authority, complete
source correlation, and evaluation-time echo before mapping it to
`ControllerPhaseInput`; set transition phase start equal to that evaluation;
and preserve the exact same start on every later evaluation in that phase.
The controller independently binds every nonzero proposal source field that
the frozen command contract can carry with `validate_command_proposal_source`.

The local attitude input is a unit body-to-world quaternion in `(w, x, y, z)`
order plus FRD body rates, but has no timestamp, clock identity, or source
correlation. `CommandProposalV1` cannot bind attitude provenance. This
candidate is therefore ineligible for shadow, runtime, or powered wiring until
a reviewed IMU timing/derotation seam exists. The bounded guidance target is
also frozen to exact image center `(0, 0)`; an off-center objective is rejected
rather than permitted to steer toward an unreviewed image-edge target.

## Fail-closed eligibility

Malformed local values raise because no contract-valid proposal can represent
them. A well-formed but ineligible state returns an exact-zero, source-less
`CommandProposalV1` carrying the caller's tick IDs and expected authority. It
therefore cannot be mistaken for a sourced nonzero proposal. Withholding covers:

- host clock, exact authority, expected gate, or active-track mismatch;
- a mismatched/future phase start, a future/regressing objective evaluation,
  or an objective evaluation older than 100 ms;
- a decision timestamp or state sequence below the caller watermark;
- a future or older-than-100 ms decision, an effective measurement age above
  150 ms after adding measurement-time uncertainty, or a prediction budget
  `max(0, prediction - proposal) + delay_uncertainty` above 100 ms, including
  when nominal prediction lead is zero or in the past;
- camera measurement-time or prediction-delay uncertainty above 50 ms;
- out-of-envelope bearing/rate or covariance diagonal;
- a guidance-withheld objective;
- a state carrying an explicitly rejected innovation;
- all `INITIALIZING`, `COASTING`, `UNHEALTHY`, and `LOST` states;
- every non-`HEALTHY` Gate 0 state; and
- an expired Gate 1 recenter window.

Gate 1 additionally admits a `DEGRADED` state only when its clipping mask is
nonzero, guidance explicitly permits the objective, and all ordinary age,
authority, bearing, and covariance gates pass. Any accepted Gate 1 proposal
from a degraded or clipped state sets `uncertainty.limited=true` with reason
`bounded_gate1_recenter_degraded_or_clipped`.

Every Gate 0 legacy-law and schedule constant is frozen to its documented
default: roll gain/limit, pitch blend, launch/boost boundaries and thrusts,
shared vertical coefficients, attitude gains, and Gate 0 body-rate/thrust
envelopes. A configuration deviation is rejected before any proposal can carry
the `legacy_gate0_pixel_pd` reason. Gate 1 gains and output envelopes, plus
state/objective timing, bearing, and covariance evidence thresholds, are
tighten-only. Gate 1's completion corridor is not tunable in either direction:
its exact reviewed thresholds remain `(x, y, rate)=(0.10, 0.12, 0.25 norm/s)`.

## Gate 0 regression mapping

The controller preserves only the legacy behavior representable by the frozen
feature state and explicit attitude/phase inputs. With a centered guidance
target, normalized bearing is the legacy 640x360 image normalization:

```text
normalized_x = (center_x_px - 320) / 320
bearing_y     = (control_y_px - 180) / 180
bearing_rate_y_norm_s = filtered_control_y_rate_px_s / 180
```

The default Gate 0 law is exactly:

```text
target_roll = clamp(0.15 * normalized_x, -0.08, +0.08)
elapsed_s = (proposal_monotonic_ns - phase_started_monotonic_ns) / 1e9
target_pitch = (1 - min(1, elapsed_s / 0.8)) * initial_pitch_rad

thrust = 0.26                         when elapsed_s < 0.15
thrust = 0.32                         when 0.15 <= elapsed_s < 0.45
thrust = clamp(0.275
               + 0.040 * clamp((180 - control_y_px) / 90, -1, +1)
               - 0.00070 * clamp(control_y_rate_px_s, -300, +300),
               0.21, 0.32)           otherwise
```

The target attitude is converted to FRD roll/pitch rates with the same
quaternion-error gains `(kp_roll, kp_pitch)=(1.0, 0.5)`, damping
`(kd_roll, kd_pitch)=(0.4, 0.2)`, and `+/-0.25 rad/s` clamp. Desired yaw equals
current yaw for the attitude error, but requested yaw rate is always exact
positive `0.0`.

The `+/-0.25 rad/s` clamp limits proposal intent; it does not establish that
the measured vehicle attitude or body rates are safe. The external supervisor
and runtime watchdogs retain all actual attitude/rate abort responsibility.

This is feature/control-law equivalence, not live runner equivalence. The new
state bearing comes from a fitted inner aperture, while the proved runner uses
legacy bbox/square-center inference and its own filtered pixel-rate history.
Exact damping equivalence therefore requires an input bearing rate equal to
that legacy filtered rate. Bbox area, close-crossing qualification, target-loss
confirmation, race credit, watchdogs, attitude-estimator health, command
pacing, and cleanup are not reconstructable here and remain outside this pure
controller. The controller is not wired into the runner.

## Gate 1 bounded recenter

`GATE1_RECENTER` exists only for authority whose expected gate index is 1. It
has no passage, approach-through-gate, crossing, or race-credit mode. Defaults
are intentionally tighter than Gate 0:

- exact-zero target pitch basis, so no forward-progress target is introduced;
- roll target `clamp(0.12*x_error + 0.025*x_rate, +/-0.05 rad)`;
- roll/pitch body-rate clamp `+/-0.12 rad/s`, yaw exact zero;
- the same normalized vertical PD with thrust clamped to `[0.21, 0.30]`;
- a hard 0.60-second proposal window; and
- an inclusive bearing/rate corridor that returns exact zero when reached.

Only a `HEALTHY`, unclipped state may return the source-less
`gate1_recenter_corridor_reached` result. A degraded or clipped state inside
the same corridor instead returns the distinct source-less
`gate1_recenter_corridor_unconfirmed_limited` result with
`uncertainty.limited=true`; it never claims completion. Corridor withholding
and timeout contain no passage claim. They are not cleanup proof and do not
authorize a future powered stage.

## Evidence boundary

Direct tests use constructed immutable contract values only. They cover fixed
Gate 0 regression fixtures, host-derived phase schedule/timeout boundaries,
stable phase-start and objective-evaluation watermarks, source binding,
determinism, yaw zero, saturation diagnostics, health/age/covariance gates,
unconditional prediction-delay budget arithmetic, metric-pose independence,
fully frozen Gate 0 tuning and Gate 1 corridor thresholds, the degraded/clipped
Gate 1 exception, tighter recenter envelopes, and source-less corridor/timeout
behavior. They are offline unit evidence, not replay, simulator,
actuator-response, powered, or safety-supervisor evidence.
