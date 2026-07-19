# VQ2 shared contract reference

**Target:** DCL FlightSim build 3385, Training mode

**Contract generation:** `/1`

**Live-state authority:** `2026-07-18-vq2-handoff.md`

## Scope and versioning

The VQ2 contracts are immutable, exact data values for the production path:

```text
UDP JPEG + HIGHRES_IMU + race status
  -> GateObservationV1
  -> RelativeGateStateV1
  -> CommandProposalV1
  -> safety supervisor
  -> SupervisorApprovedCommandV1
  -> transport compatibility projection
```

They do not arm, reset, send, advance a gate, declare passage, or prove
cleanup. Race status and the existing safety supervisor remain authoritative.
The transport projection constructs a legacy DTO but performs no send.

Codecs reject missing or unknown fields, unknown enums or mask bits, booleans
used as numbers, non-finite values, ambiguous evidence, and inconsistent
timing, covariance, health, or authority data. Every finite floating-point zero
is canonicalized to positive `0.0`, so exact-zero wire values and hashes have
one representation. A change to fields, units,
coordinate frames, enum/mask meaning, ordering, or safety semantics requires a
new schema version. Adding stricter validation that merely enforces an already
documented `/1` invariant does not change the wire meaning.

The frozen top-level schemas are:

| Value | Schema |
|---|---|
| Frame identity | `aigp-vq2-frame-identity/1` |
| Frame timing | `aigp-vq2-frame-timing/1` |
| Prediction timing | `aigp-vq2-prediction-time/1` |
| Latency event | `aigp-vq2-latency-event/1` |
| Gate authority epoch | `aigp-vq2-gate-authority-epoch/1` |
| Gate observation | `aigp-vq2-gate-observation/1` |
| Relative gate state | `aigp-vq2-relative-gate-state/1` |
| Command proposal | `aigp-vq2-command-proposal/1` |
| Supervisor-approved command | `aigp-vq2-supervisor-approved-command/1` |
| Tier evidence scope | `aigp-tier-evidence-scope/1` |

## Time and frame identity

Frame identity is exactly `(stream_id, generation, uint32 frame_id)`. Camera
generation scopes receiver restarts and is not reset proof. The camera source
timestamp is an opaque uint64 ordering and integrity token. It is not identity,
reset-relative time, or calibrated capture time, and it is never subtracted
from a host-monotonic timestamp.

All host times are comparable only when their `host_clock_id` matches.
`FrameTimingV1` records first and final unique-packet arrival, reassembly,
decode, and publication on that clock; those six points are nondecreasing.
Its sequence validator is scoped by host clock and stream. It rejects a
repeated frame identity, a non-increasing publication sequence, decreasing
generation, non-increasing publish-monotonic time, and non-increasing source
time within one generation. It does not infer ordering from `frame_id`.

The source-frame identity, publication sequence, and publication time in
`PredictionTimeV1` are all present or all absent. Measurement, decision, and
prediction times obey `measurement <= decision <= prediction`; a source frame
cannot be published after the decision, and a camera measurement cannot be
later than that publication. A final-packet measurement in
`GateObservationV1` is explicitly a proxy: it equals the nested frame's final
unique-packet time and carries nonzero uncertainty. `PredictionTimeV1` carries
only frame publication metadata, not full `FrameTimingV1`, so its codec cannot
independently re-prove that equality.

A calibrated camera capture time or propagated IMU time requires its own
mapping-model ID and nonzero uncertainty. Raw IMU samples and final-packet
proxies cannot claim a mapping model. Measurement-clock models and
command-delay models are separate. Decision-time predictions equal the
decision and carry no delay model; send/effect-time predictions require a
delay model and nonzero uncertainty.

Latency events carry required correlation minima by kind:

| Event group | Required correlation |
|---|---|
| Camera/frame/decode/detection/tracking/estimator and frame-drop | frame |
| Prediction and controller | frame and control tick |
| Tick due/start/end/skip and deadline miss | control tick |
| Command send start/end | control tick and command ID |
| Gyro and actuator sample | sensor-sample ID and opaque uint64 source time |

The command ID is not intrinsically an approved-command identity; only the
approval/latency correlation validator establishes that binding. Sensor-sample
correlation fields are forbidden on non-sample events. Raw gyro and actuator
samples need no command ID, and an optional command ID does not itself prove
causality. `FRAME_DROPPED` requires `DROPPED`; `CONTROL_TICK_SKIPPED` and
`DEADLINE_MISSED` require `SKIPPED`. Other kinds cannot use those two outcomes.
An `OK` event has no reason; every non-`OK` event has one.

The sequence validator rejects per-host event-sequence or time regression,
mismatched command-to-tick correlation, duplicate open stages, and an end
without its matching start. An unfinished start remains representable for a
truncated trace. Tick IDs become due once and increase strictly; due ticks and
command-send starts are at least `20_000_000 ns` apart. A control-tick start or
skip requires an earlier due event, a started/ended/sent tick cannot be skipped,
and an unstarted tick is explicitly skipped before a newer tick becomes due.
Every deadline miss precedes and ends in its skipped-tick event. Runtime
transport still owns actual pacing and single-use enforcement; this validator
checks trace evidence.

## Gate authority

`GateAuthorityEpochV1` is issued by the safety supervisor from proved reset and
race-status state. Perception, estimation, and control may only echo it. It
contains the session, reset epoch, gate epoch, expected gate index, and the
race-status sequence and boot time in milliseconds that support the snapshot.
It also freezes an inclusive camera cutover:

```text
camera_host_clock_id, camera_stream_id, camera_generation,
frame_publication_sequence_not_before,
frame_publish_monotonic_ns_not_before
```

An observation, relative state, or proposal with source-frame correlation must
match that host clock, stream, and generation, and its publication sequence and
publication time must be greater than or equal to both watermarks. There is no
`frame_id` cutoff. Camera identity, visual continuity, confidence, or geometry
can never manufacture or advance this authority.

## Observation geometry

`GateObservationV1` is one raw, unassociated candidate. Active/shadow roles
begin in tracking. Image coordinates use x-right/y-down with borders at
`-1/+1`. Truly visible edges and corners remain in that interval. A censored
model-inferred center may extend to `+/-4`, but an out-of-frame center requires
the matching clipping bit. Support bbox coordinates alone use `[0,1]` and
must have positive area. A clipped outer edge cannot also be marked visible.

Visible inner corners and the separate fitted inner-aperture quad are ordered
top-left, top-right, bottom-right, bottom-left. The fitted quad is censored,
model-inferred geometry: its coordinates may extend to `+/-4`, and it must be a
strictly convex clockwise quad. A supplied visible inner corner must exactly
equal the corresponding fitted corner. Top/bottom and left/right edge-midpoint
ordering anchors those labels so a cyclically rotated corner list is rejected.
Every fitted corner outside the image requires its corresponding clipping bit.
The fitted projected center is the quadrilateral diagonal intersection, and
`center_norm` matches it with relative and absolute tolerance `1e-9`.

The fitted quad, geometry-model ID, `log_scale`, and two-axis projective skew
are all present or all absent. `log_scale` is the natural logarithm of the
square root of fitted inner-aperture polygon area in normalized image units.
Projective skew is `(log(right/left edge length), log(bottom/top edge length))`.
Zero denotes symmetric fronto-parallel projection. A fitted aperture requires
positive support and inlier counts plus a residual, and nominal health requires
a fitted aperture. Degraded and unusable observations require a health reason.

Every covariance has a model ID, exact feature order, symmetric
positive-semidefinite matrix, and strictly positive variances. Center features
are mandatory; scale and both skew covariance features appear exactly when the
corresponding estimates do. Feature names are unique and covariance dimension
is at most nine.

One observation batch uses one exact frame timing and authority binding per
camera frame, and candidate IDs are unique within that frame. This prevents two
different candidates from sharing an indistinguishable source identity.

## Legacy observation adapter

The current detector's bbox is contour support, not a fitted gate edge. Its
corner and metric-distance fields are placeholders. The adapter therefore
copies neither into inner/outer edge, corner, scale, skew, or pose evidence.
It emits a degraded observation with caller-supplied conservative center
covariance and nonzero final-packet-proxy timing uncertainty.

The reverse legacy projection requires the exact nested frame timing and the
current safety authority and rejects unusable observations. It also requires
an exact pixel-grid round trip and a center inside the projected bbox; it never
silently rounds model geometry into a different legacy target. Replay
projection cannot mark an unusable or out-of-frame observation selector-eligible.
The legacy output field named `sim_time_ns` still contains the opaque camera
source token; that historical field name does not make it calibrated time.

## Relative state

`RelativeGateStateV1` requires a source frame and the exact six-feature order:

```text
bearing_x_norm, bearing_y_norm, log_scale,
bearing_rate_x_norm_s, bearing_rate_y_norm_s, expansion_rate_s
```

Bearing uses x-right/y-down normalized image coordinates and may extend to
`+/-4`; bearing-rate and expansion fields are per second. Each state carries
`tracker_id`, `state_sequence`, `measurement_update_sequence`,
`source_candidate_id`, and active/shadow `track_role`. The sequence validator
is keyed by host clock, session, reset epoch, gate epoch, expected gate index,
and tracker. It requires strictly increasing state sequence and nondecreasing
prediction time. Every previously seen `(source_frame, source_candidate_id)`
keeps its original measurement-update sequence, even after intervening frames;
a new source advances it. Active/shadow ownership is keyed by stable
session/reset/gate/index identity, so refreshing race-status snapshot fields
cannot transfer one frame/candidate to another tracker. Snapshot race/camera
metadata must progress without regression within that stable epoch.

`validate_relative_gate_state_source` rejects unusable observations and binds
authority, host clock, frame, publication sequence/time, candidate,
measurement time/basis/model/uncertainty, clipping, and inner/outer visibility
exactly to the cited observation. A state cannot relabel an old frame by
inventing a newer publication watermark.

Innovation score, threshold, and accept/reject result are all-or-none.
Dropout predictions cannot claim a current-frame innovation result, require
coasting/lost health, and carry a nonzero dropout count. Healthy state cannot
carry dropouts, a rejection, or a health reason. Clipped outer edges cannot be
simultaneously visible.

Metric state is optional and all-or-none: gate-center position and relative
velocity in body FRD, a unit gate orientation quaternion in `(x,y,z,w)`, and an
exact 9x9 covariance. The quaternion actively rotates gate-local vectors into
body FRD. Gate-local +x is aperture-right, +y is down, and +z is the reviewed
travel-direction normal. If that normal or metric scale is ambiguous, the
producer leaves the entire metric state absent.

The metric covariance order is exactly:

```text
position_x_body_frd_m, position_y_body_frd_m, position_z_body_frd_m,
velocity_x_body_frd_m_s, velocity_y_body_frd_m_s, velocity_z_body_frd_m_s,
orientation_error_x_rad, orientation_error_y_rad, orientation_error_z_rad
```

## Command authority seam

A proposal contains proposal and control-tick IDs, host clock and gate
authority, requested body rates/thrust, the tick deadline, phase/reason, and
explicit controller saturation and uncertainty diagnostics. Requested body
rates are finite intent rather than an approved envelope; requested thrust is
in `[0,1]`. The proposal time does not exceed the tick deadline.

The following source-state correlation bundle is all present or all absent:

```text
source_state_decision_monotonic_ns,
source_state_prediction_monotonic_ns,
source_frame, source_frame_publication_sequence,
source_frame_publish_monotonic_ns, source_tracker_id, source_track_role,
source_state_sequence,
source_measurement_update_sequence, source_candidate_id
```

When present, the source decision time does not postdate proposal creation or
the source prediction horizon, and frame publication does not postdate that
decision. The prediction horizon may postdate proposal creation when the state
is predicted to command-send or command-effect time. The bundle obeys the
authority camera cutover. `validate_command_proposal_source` binds every field,
plus host clock and authority, to the cited `RelativeGateStateV1`.

A nonzero proposal requires the complete source bundle. Only an exact-zero
rate/exact-zero-thrust failsafe may omit it; that source-less value has no state
correlation and the cross-object validator rejects attempts to claim one.

The legacy adapter accepts only the exact `AttitudeRateCommand` DTO and
caller-supplied diagnostics; it does not invent unsaturated or certain status.

Only `SupervisorApprovedCommandV1` can project to the transport DTO. Approval
is tied to the nested proposal, command ID, host clock, current control tick,
gate/reset authority, approval time, and a validity time no later than the tick
deadline. Approved roll/pitch rates obey the build-3385 `+/-0.25 rad/s` bound
and yaw rate is exact zero; approved thrust remains in `[0,1]`. Approval cannot
predate its proposal, and validity starts no earlier than approval. The
supervisor may only reduce each requested rate's magnitude without reversing
its sign or creating a nonzero axis, and may not increase thrust. A limit reason
is present exactly when approved values differ. Stage-specific thrust and
tighter envelopes remain identified by the safety policy and are not weakened
by this shared schema. A shadow-track proposal may be evaluated open loop, but
the supervisor can approve it only after reducing it to exact zero; any nonzero
approved command requires an active source track.

The trusted projection additionally receives the current host clock, send
time, control-tick ID, trusted tick deadline, authority, safety-policy ID, and a
positive maximum approval age in nanoseconds. All identities must match; send
time is at or after approval, at or before validity expiry, and no older than
the trusted maximum. Projection preserves the approved values exactly,
constructs the legacy DTO, and performs no send. That pure projection neither
authenticates a supervisor caller nor makes the mutable legacy DTO single-use;
the trusted stateful transport boundary owns both guarantees.

The approval sequence validator rejects reused proposal/command IDs per host,
non-increasing control ticks, approval-time regression, authority switching
without a forward gate/reset transition, and reset/gate/race-status or camera
cutover regression. The command/latency validator requires exactly one matching
tick due/start and send-start/send-end pair with the same host, tick, and command
ID. Tick due/start precede proposal creation; tick and send events are `OK`,
preserve the source frame when present, and both send events finish inside the
inclusive approval window.
Stateful transport still owns single-use enforcement, 50 Hz pacing, stream
watchdogs, arm state, and cleanup. The four-field proposal replay projection is
open-loop diagnostic output only; it is never a send path.

## Tier evidence

`TierEvidenceScopeV1` fixes the ordinary evidence boundary:

| Tier | Allowed claim |
|---|---|
| T0 | affected tests; no flight-domain claim |
| T1 | causal open-loop replay; no powered, closed-loop, official-simulator, or gate-authority claim |
| T2-T4 | deterministic synthetic closed-loop sequence evidence; explicitly nonpowered |
| T5 | unavailable in the ordinary scheduler |

T0/T1 reject forbidden claims recursively even when their value is `false`;
absence is different from claiming the fact was verified. T1 accepts only the
reviewed recorded or candidate causal replay provenance:

| Source | Perception | Estimator | Open-loop commands |
|---|---|---|---|
| Candidate | `candidate_detector_on_all_decoded_frames` | `candidate_estimator_on_ordered_sanitized_stream` | `candidate_generator_on_ordered_sanitized_stream` |
| Recorded | `recorded_processed_frames` | `recorded_bundle_context` | `recorded_bundle_command_stream` |

Candidate worker transport, when present, is exactly
`candidate_worktree_code_hash` and is not accepted with recorded provenance.
T2 uses `race_01`; T3 uses `grand_tour`, `slalom`, and `vertical_cliff`; T4
uses those tracks plus `aigp_default`, `figure8`, `race_01`, and
`straight_hairpin`. T2-T4 require
an exact six-field `domain_provenance` object:

```text
execution: deterministic_synthetic_kinematic_nonpowered
powered_resources_used: false
cleanup_gate_semantics: vacuously_true_only_after_synthetic_domain_proof
stale_stream_gate_semantics: vacuously_true_only_after_synthetic_domain_proof
centering_proxy: negative_worst_p95_tracking_error_m
stability_proxy: negative_worst_max_tracking_error_m
```

Evidence discovery accepts one canonical replay-corpus envelope as one unit
without relabeling its per-session children as sibling evidence. Distinct or
aliased sibling payloads are ambiguous and rejected. Both scheduler checkpoint
binding and final promotion-chain validation enforce these rules.

## Current limitations

No usable pose or gate-map stream exists in build 3385. There is no calibrated
camera capture mapping, production aperture fit, metric pose, approved private
golden corpus, or shipped powered evaluator in this checkout. The contracts
make those absences explicit; they are not evidence that those capabilities or
authorizations exist.
