# VQ2 visual-course architecture-reset handoff - 2026-07-28

> This is a research and implementation handoff for FlightSim build 3385 in
> Training mode. It does not supersede the verified interface or live-flight
> safety contract in `docs/aigp/2026-07-18-vq2-handoff.md`.
>
> Source state audited: branch
> `redesign/vq2-single-governor-20260727`, commit
> `e56c563518ffddb52b37cdb964065ba16f2b58b4`.
>
> The conclusion is architectural: preserve the working transport,
> perception, runner, and real safety shell, but bypass the existing
> visual-course navigation coordinator wholesale. Do not resume the
> run-specific patch loop inside that coordinator.

## Required reading and authority order

Read these before changing or flying anything:

1. Repository `AGENTS.md`.
2. `docs/aigp/2026-07-18-vq2-handoff.md` for the build-3385 interface and
   live-flight safety contract.
3. This handoff for the current failure analysis and replacement boundary.
4. `docs/aigp/2026-07-26-vq2-capability-envelope-audit.md` for measured
   capability versus historical navigation limits.
5. `docs/aigp/fast-flight-cycle.md` for the rapid powered-attempt path.

If this handoff conflicts with the July 18 interface or real safety contract,
stop and flag the conflict for human review. Historical VQ1 pose/gate-map
assumptions are not authority for build 3385.

## Definition of done

The task is not complete at Gate 0 or Gate 1. Completion requires:

- one bounded build-3385 Training-mode `visual-course` attempt;
- monotonic authoritative race progress through the entire course;
- authoritative `race_finished`;
- the final and maximum gate indexes agreeing with race status;
- navigation outcome and cleanup outcome recorded separately; and
- best-effort zero, disarm, reset, and final disarm from the runner's
  `finally` path.

Vision never declares a gate pass. `active_gate_index` and
`race_finish_time_ns` are authoritative.

## Current state and evidence

The current controller has repeatedly passed the first gate but has never
passed the second gate:

| Evidence window | Result |
| --- | ---: |
| Actual flights in the latest approximately 20-hour window | 99 |
| Passed Gate 0, authoritative `0 -> 1` | 86 |
| Passed Gate 1, authoritative `1 -> 2` | 0 |
| Ended on navigation proof/lineage refusal | 51 |
| Ended on a local recovery/mode timer | 21 |
| Ended on a real simulator collision | 25 |
| Ended on a real command/attitude-envelope stop | 2 |

All 74 non-collision controller aborts in that window still had a fresh visual
graph frame:

- median age: `1.503 ms`;
- p95 age: `26.257 ms`;
- maximum age: `64.57 ms`.

The dominant "freshness" problem was therefore semantic proof freshness, not a
stale UDP camera stream.

Since July 25, the evidence set contains 334 `visual-course` runs and no
`race_finished`. The broad outcome census includes 201 internal semantic
boundaries, 23 timeouts, 12 raw freshness/stale outcomes, and 41 collision
outcomes.

The two latest powered attempts before this handoff were:

### `20260729T040518Z-visual-course-797413fd`

Evidence:
`C:\Users\John\aigp-evidence\fast-flight-cycles\20260729T040518Z-visual-course-797413fd`

- source commit: `e56c563518ffddb52b37cdb964065ba16f2b58b4`;
- authoritative Gate 0 credit: yes;
- authoritative Gate 1 credit: no;
- `race_finished`: false;
- course elapsed before failure: `3.891 s`;
- real collision: object `1002`, threat 2, impulse `1.153`, followed by
  additional contact and immediate body rate `7.51 rad/s`;
- no stale raw stream;
- cleanup confirmed.

The intended pitch/roll changes reached the wire, but `0.21` collective was
retained after vertical observability was lost. Once vertical velocity
reversed, the stale sub-support command continued and the vehicle hit the
ground.

### `20260729T034617Z-visual-course-c6407bd0`

Evidence:
`C:\Users\John\aigp-evidence\fast-flight-cycles\20260729T034617Z-visual-course-c6407bd0`

- authoritative Gate 0 credit: yes;
- authoritative Gate 1 credit: no;
- course elapsed before failure: `5.641 s`;
- real Gate collision: object `1001`, threat 1, impulse `0.3438`;
- no stale raw stream;
- cleanup confirmed.

These contacts are reasons to retain the collision watchdog. They are not
evidence that the semantic navigation proof system should remain.

## Why the existing architecture failed

The current navigation path is not one controller. It is a chain of partially
overlapping controllers, proof systems, recovery state machines, and
wire-time overrides.

Approximate production navigation size at the audited commit:

| Module | Lines |
| --- | ---: |
| `scripts/aigp_vq2_visual_course_stage.py` | 17,144 |
| `planning/vq2_dynamic_visual_approach.py` | 4,465 |
| `planning/vq2_dynamic_course.py` | 4,771 |
| `planning/vq2_visual_approach.py` | 1,478 |
| `planning/vq2_visual_servo.py` | 1,844 |
| `planning/vq2_gate_graph.py` | 3,090 |
| `planning/vq2_course_lifecycle.py` | 1,291 |
| Total | 34,083 |

The main visual-course stage coroutine is approximately 9,600 lines. A
nominal command can pass through roughly 18 layers:

1. detector/tracker;
2. rolling gate graph;
3. dynamic wrapper;
4. dynamic core;
5. roll-reference handoff;
6. dynamic image-servo adapter;
7. legacy approach servo;
8. stage vertical controller;
9. launch override;
10. TOP-FOV override;
11. TOP recovery;
12. horizontal-FOV brake;
13. crossing coast;
14. yaw soft stop;
15. attitude PD;
16. roll/yaw transport;
17. clamp and secondary wire governor;
18. exact-frame/race lease and acceptance state.

The same target pitch, roll, and thrust are overwritten repeatedly. Diagnostic
and evidence dictionaries can later grant command authority. This makes
causal reasoning and safe tuning impractical.

The implementation also carries overlapping lifecycle modes for approach,
passage, near-plane, credit wait, promotion/reacquisition, one-edge
censorship, TOP recovery, inner-aperture dropout, propagated visibility gaps,
same-gate rebind, ambiguity quarantine, successor steering, roll handoffs,
FOV retention, collective retention, and wire continuity.

Since July 25, 286 commits touched core navigation. Across the audited
controller/test scope, that work added approximately 44,044 lines and removed
only 263. The pattern is patch accretion rather than convergence.

## Fundamental causal defects

### 1. Navigation bookkeeping poisons perception

`VQ2Runner._sample()` in `scripts/aigp_vq2_run.py` combines:

- IMU and race ingestion;
- camera snapshot consumption;
- detector execution;
- aperture fitting;
- multi-target tracker update;
- aperture-lineage assertion;
- gate-graph observation and role mutation;
- legacy tracker update; and
- evidence generation.

`visual_gate_graph.observe()` runs inside that shared failure boundary. An
exception in graph roles, lineage, or evidence becomes `_detection_error`, and
the raw watchdog then aborts with a detector failure even though the JPEG and
detector result were healthy.

Course-mode perception must stop after real detection/tracking. Optional
target-selection or evidence failures must not poison raw detector health.

### 2. Race authority is subordinate to visual proof

The simulator's sequential `active_gate_index` increment is authoritative,
but the current path can reject it because a prior passage proof, clipped-miss
classification, exact camera prefix, reviewed preview identity, or graph
promotion is incomplete.

The same problem exists at final success: the runner currently requires a
complete rolling-graph transition chain in addition to `race_finished`.

The replacement must accept a newer sequential race increment immediately,
retire the previous target hypothesis, and promote the retained successor or
enter search. Visual inference must not veto race credit.

### 3. Raw race ownership is coupled to exact JPEG pinning

`VQ2Runner._send_flight_command()` currently permits a guarded
`wire_race_gate_index` only when an exact visual publication token, wire
receipt, and narrow call-start deadline are also supplied.

Atomic race ownership is a real invariant. Exact JPEG pinning and a `12 ms`
validation-to-wire proof are navigation/evidence policy. Split them:

- retain an atomic race-active transport call;
- do not require a camera publication lease to perform it;
- if the race index changes before wire, skip the obsolete command, sample the
  new state, accept the authoritative promotion, and continue.

### 4. Vertical control uses contradictory physical semantics

The effective build-3385 camera/body response is represented as `Rx(pi)`.
With the stable image-down vertical error used by the live controller,
collective above support drives error acceleration negative. Stable feedback
therefore has one global form:

```text
collective = support/feedforward
           + Kp * stable_vertical_error
           + Kd * qualified_stable_vertical_rate
```

The sign must not change at Gate 1. Gate 0 may use launch boost as feedforward,
but not an opposite closed-loop sign.

At the audited commit:

- Gate 0 consumes the same stable coordinate with the opposite feedback sign;
- later gates use the physically stable sign;
- multiple later layers can replace the requested collective; and
- loss handling can retain a saturated sub-support command.

When the vertical axis or rate becomes unqualified:

- do not inject a censored-axis zero rate as if it meant stationary;
- discard the derivative correction;
- inflate uncertainty; and
- decay smoothly toward tilt-compensated support collective.

Never retain a saturated `0.21` collective merely because its former authority
object remains live.

### 5. Lateral control is a patch around a disabled plant model

Production explicitly sets `roll_to_lateral_bearing_accel=0.0`. The controller
therefore does not predict how bank changes gate bearing. Full-bank
hysteresis, rebound commands, and roll handoffs compensate for that missing
model and produce visible jerks.

The clean controller should begin with one empirically correct sign and
bounded proportional/derivative image-space control. A plant gain can be
identified later without adding new lifecycle modes.

### 6. Successor steering is structurally suppressed

Successor turn authority is currently the product of visibility, ambiguity,
rate qualification, scale qualification, positive clearance, a dwell, a
ramp, passage alignment, TTC progress, current-yaw release, and a maximum
weight. Any failed term resets steering to zero.

This directly explains the repeated "fly straight, then turn too late"
behavior.

Successor lookahead should be a continuous blend based on current-gate
alignment, scale/closure, successor confidence, and uncertainty. A weak
successor estimate should reduce the blend, not erase it through an unrelated
binary authority failure.

### 7. Search often does not search

When exact geometry expires and no already-qualified unique candidate is
visible, the current "active search" can command zero horizontal error, zero
yaw, level roll, braking pitch, and support thrust.

The clean `SEARCH` mode must issue a real bounded yaw sweep. Initialize its
direction from the last observed target/successor bearing and reverse on a
bounded schedule while holding safe pitch and support collective.

### 8. Tests fossilize the failed architecture

There are at least 767 navigation-specific test functions before pytest
parameter expansion:

- 416 across ten active `planning/tests/test_vq2_*.py` navigation files;
- 189 in `tests/test_aigp_vq2_visual_course.py`;
- 39 recorded/replay/reacquisition functions;
- 26 visual-shadow functions;
- 7 visual-config functions; and
- approximately 90 Gate-0/Gate-1-recenter/course-line/full-lap functions in
  `tests/test_aigp_vq2_runner.py`.

Many tests assert exact timers, event dictionaries, graph lineages, internal
mode sequences, or behavior named after one run/commit. Keeping all of them
green while replacing the architecture forces every obsolete mechanism to
survive.

Freeze those tests as legacy evidence. Do not use them as active acceptance
for the clean controller.

## True hard boundaries to retain

Retain these unchanged unless new measured evidence and human review explicitly
change them:

- host-wide live lease;
- reset epoch and rollback proof;
- countdown and GO timing;
- fresh heartbeat, IMU, race, actuator, and raw camera streams;
- healthy IMU attitude estimate;
- armed-state confirmation;
- collision abort;
- broad measured attitude/body-rate watchdog;
- finite commands;
- roll/pitch body-rate cap `+/-0.25 rad/s`;
- yaw cap from the accepted v3 profile, `+/-0.15 rad/s`;
- active visual-course thrust envelope, with exact-zero commands reserved for
  the existing crossing/abort/cleanup semantics;
- 50 Hz pacing with missed-tick drop;
- one hard attempt timeout;
- atomic authoritative race-gate ownership at wire;
- best-effort zero, disarm, reset, and final disarm;
- cleanup outcome recorded separately from navigation milestones.

The runner watchdog already provides the desired raw boundary when called with
`require_target=False`: it still checks heartbeat, IMU, race, actuator, raw
camera, detector execution, estimator, armed state, collisions, finite
attitude, and measured envelopes. It does not need gate-graph authority.

The July 18 bounded credible-crossing/credit wait remains one explicit
contract state. Do not recreate multiple passage, censorship, and
post-credit timers around it.

## Navigation boundaries to remove or demote

These must not terminate a clean-course attempt:

- exact graph role or track-lineage proof;
- visual passage proof as a veto over authoritative race credit;
- exact clipped-miss classification;
- exact camera prefix at race credit;
- exact camera publication pinning at wire;
- `12 ms` validation-to-wire expiry as an attempt-fatal outcome;
- passage preview identity;
- graph-vetted promotion/reacquisition/rebind;
- ambiguity quarantine expiry;
- propagated visibility-gap expiry;
- TOP-FOV authority expiry;
- inner-aperture dropout expiry;
- post-credit fresh-frame opportunity timer;
- segment-local and passage-local hard timers;
- evidence/diagnostic completeness;
- exact internal mode/event sequence;
- exact old command retention.

Ordinary clipping, ambiguity, weak confidence, missing aperture geometry,
temporary target loss, and frame supersession should increase uncertainty,
reduce aggressiveness, skip one command, or enter search. They are not raw
stream failures.

## Exact replacement seam

The true replacement point is:

```text
scripts/aigp_vq2_run.py
    VQ2Runner._run_visual_course()
```

Do not replace only `VisualCourseStageRuntime.servo_factory`. That leaves the
9,600-line proof/timer coordinator active.

The retained outer path is:

```text
scripts/aigp_vq2_fast_cycle.py
    host-wide lease, clean commit/manifest, compact evidence
        ->
scripts/aigp_vq2_run.py
    run_live()
        ->
    VQ2Runner.run_powered_stage()
        reset -> disarm -> countdown/GO -> initial vision -> arm
        -> VQ2Runner._run_visual_course()    REPLACE HERE
        -> safe_cleanup() in finally
        ->
    vision stop and adapter disconnect
```

Retain:

- `execute_fast_cycle()` and its live lease/manifest;
- `run_live()`;
- `run_powered_stage()` reset/GO/arm/finally lifecycle;
- `establish_reset_epoch()`;
- `wait_for_go()`;
- `arm_confirmed()`;
- `_watchdog(require_target=False)`;
- `_wait_for_next_flight_command_slot()`;
- command validation;
- `safe_cleanup()`;
- outer vision shutdown and transport disconnect.

Course-specific integration changes required:

1. Add one clean course-stage module.
2. Redirect `_run_visual_course()` to the clean stage. Do not construct or
   call `DynamicVisualCourseSession` or the old course coordinator.
3. For `visual-course` only, make `_sample()` publish detector/tracker output
   without gate-graph observation, role mutation, or fatal aperture-lineage
   evidence checks. Preserve legacy behavior for shadow/alignment stages if
   they remain in use.
4. Skip `_bind_initial_visual_gate()` for `visual-course`. Initialize the
   clean selector from the tracker update, authoritative Gate 0, and
   `StartContext`'s initial gate center/area.
5. Add a race-only atomic send guard that does not require a visual
   publication lease.
6. Determine final success from authoritative `race_finished`, sequential
   race-index observations, matching final/maximum gate state, yaw-profile
   identity, and cleanup. Do not require a rolling-graph proof chain.
7. Add the clean executable module to the fast-cycle runtime source manifest.

## Minimal trustworthy observation contract

### Camera

Reuse:

- `competition.vq2_vision.VQ2VisionSnapshot`;
- `VQ2VisionThread.snapshot()`;
- exact `(stream_id, generation, frame_id)` identity;
- host packet/publication timing for freshness.

`camera_source_time_ns` is an opaque ordering/integrity token. Do not subtract
it from host monotonic time.

### Detector and tracker

Reuse:

- `VisualDetectionFrame.from_vision_snapshot()`;
- `MultiTargetVisualTracker.update()`;
- `VisualTrackerUpdate.visible_tracks`;
- `VisualTrack.track_id`, `center_norm`, `bbox_norm`, `apparent_scale`,
  `center_velocity_norm_s`, `log_scale_rate_s`, confidence, association
  confidence, clipping/censorship, visibility, history, and optional inner
  aperture.

Prefer a valid fitted inner-aperture center/size. Fall back to outer bbox
center/size with larger covariance.

Important: the tracker may force a censored-axis rate to zero. That zero means
"unobserved", not "stationary".

Do not treat detector `estimated_distance` as metric distance. It uses a
placeholder gate width. Detector bbox corners are not physical gate corners.

### IMU

Reuse the received IMU ingress and `ImuAttitudeEstimator`:

- source `timestamp_us` for integration/order;
- ingress `received_monotonic_ns` for freshness and cross-stream pairing;
- relative attitude and bias-corrected body rates.

Yaw is relative, not globally observable.

### Race

Reuse exact received race status:

- `active_gate_index` is authoritative progress;
- `race_finish_time_ns >= 0` is authoritative completion.

### Identity separation

Keep these concepts distinct:

- frame token: transport identity;
- `track_id`: local visual-continuity hypothesis;
- `active_gate_index`: authoritative race progress.

A visual `track_id` is never a gate number.

## Minimal controller architecture

### Estimator

Maintain one small variable-dt filter for each retained current/successor
hypothesis:

```text
[x_image_right,
 y_image_down,
 x_rate,
 y_rate,
 log_scale,
 expansion_rate,
 covariance]
```

Use measurement covariance/confidence rather than binary authority classes.
Clipping updates observable axes and predicts censored axes. Ambiguity and
dropout mean prediction with growing covariance.

Pair frames with the nearest host-received IMU attitude/rates for short-term
rotation compensation. Inflate covariance for unknown capture latency.

### Target selector

Maintain only:

- authoritative current gate index;
- `current_track_id` as a hypothesis;
- optional `successor_track_id` as a hypothesis;
- last reliable current/successor bearings;
- confidence/covariance.

When race status increments:

1. accept it immediately;
2. retire the prior current hypothesis;
3. promote the cached successor if still credible; otherwise enter `SEARCH`;
4. continue the same controller for the new gate.

Do not require pre-credit graph promotion proof.

### Runtime states

Use only four controller states:

1. `TRACK`: fresh credible current target.
2. `PREDICT`: short target gap; same estimator/controller on predicted state.
3. `COAST_FOR_CREDIT`: one bounded credible-crossing wait governed by the
   July 18 safety contract.
4. `SEARCH`: covariance too large; safe closure/braking, support collective,
   and a real bounded yaw sweep.

Authoritative gate promotion is an event, not another state machine.

### One continuous control law

Navigation produces exactly:

```text
target_roll_rad
target_pitch_rad
yaw_rate_rad_s
thrust
```

Guidance principles:

- image-right error commands the verified negative canonical yaw direction;
- under the effective `Rx(pi)` response, positive stable lateral error
  requires negative bank;
- vertical collective uses one global stable sign at every gate;
- takeoff boost is feedforward only;
- pitch controls closure continuously: advance when aligned and confident,
  brake progressively with angular error, uncertainty, rapid expansion, or
  near-plane risk;
- clipping saturates corrective steering and increases covariance rather than
  entering a new FOV recovery state;
- successor lookahead blends continuously as current-gate scale/closure grows;
- loss of qualified vertical state removes derivative/corrective authority and
  decays collective toward tilt-compensated support;
- `SEARCH` actively sweeps yaw from the last known direction.

After navigation, apply exactly:

1. one attitude PD for roll/pitch;
2. one explicit yaw channel;
3. one transparent final clamp/slew;
4. command validation;
5. one atomic race-active send.

No downstream subsystem may reinterpret target pitch or thrust.

## Course map policy

Build 3385 provides no usable pose or gate-map stream. Do not invent a metric
world-coordinate gate map or absolute-yaw course map.

Permitted memory:

- gate-indexed last-observed successor bearing;
- relative bearing/elevation/scale/rate history;
- turn-direction/search hints.

This is a local visual cache, not metric geometry. Bearings change under
translation and must retain uncertainty.

## Test migration

### Retain as active safety/perception evidence

Keep or port tests for:

- live lease;
- UDP vision receiver and duplicate suppression;
- raw camera freshness;
- detector mechanics;
- basic tracker association and ambiguity facts;
- IMU attitude estimation;
- reset epoch and delayed old-packet rejection;
- GO and arm admission;
- 50 Hz missed-tick drop;
- collision classification/abort;
- finite command and measured envelope checks;
- atomic race-active transport send;
- hard attempt timeout;
- emergency reset and cleanup;
- yaw-profile identity and measured yaw envelope;
- cleanup on every exit.

### Freeze as legacy evidence

Do not require the clean controller to satisfy the current navigation contracts
in:

- `planning/tests/test_vq2_course_lifecycle.py`;
- `planning/tests/test_vq2_dynamic_course.py`;
- `planning/tests/test_vq2_dynamic_visual_approach.py`;
- `planning/tests/test_vq2_gate_reacquisition.py`;
- `planning/tests/test_vq2_guidance.py`;
- `planning/tests/test_vq2_recorded_gate_promotion.py`;
- `planning/tests/test_vq2_visual_alignment.py`;
- `planning/tests/test_vq2_visual_approach.py`;
- `planning/tests/test_vq2_visual_recovery.py`;
- `planning/tests/test_vq2_visual_servo.py`;
- the navigation section of `tests/test_aigp_vq2_visual_course.py`;
- recorded-facts, old coordinator replay, old reacquisition, and
  graph/lifecycle shadow tests;
- old Gate-0/Gate-1-recenter/course-line/full-lap runner behavior tests;
- Wave 1-3 exact-zero/proof-stack behavior as clean-course acceptance.

Preserve these files for historical/regression research. Move them out of the
active clean-controller acceptance surface rather than deleting the evidence.

Tests that assert exact internal event dictionaries, proof identities, mode
sequences, incident-specific commands, or names such as `run24`, `run25`,
`run28`, `attempt*`, or commit hashes must not dictate the new architecture.

### Required replacement behavior tests

Create focused behavior tests for:

1. Controller:
   - identical global vertical sign at every gate;
   - loss of vertical observability decays toward support instead of retaining
     saturation;
   - verified left/right yaw and roll directions;
   - clipping increases uncertainty but does not abort;
   - `PREDICT -> SEARCH` on fresh empty frames;
   - real bounded yaw sweep;
   - finite bounded output;
   - one command owner and one clamp/slew.
2. Runner safety:
   - truly stale camera/IMU/race aborts;
   - fresh camera with no detection is not raw staleness;
   - collision, estimator failure, command envelope, and one hard timeout
     abort;
   - camera supersession skips one tick and continues;
   - race-index change skips an obsolete command, promotes, and continues;
   - cleanup executes on every exit.
3. Course integration:
   - authoritative `0 -> 1 -> 2 -> ... -> race_finished`;
   - vision alone cannot award credit;
   - authoritative credit needs no passage-preview/graph proof;
   - clipping, ambiguity, and detection gaps lead to prediction/search rather
     than fatal semantic refusal;
   - navigation and cleanup outcomes remain separate.
4. Recorded regressions:
   - assert envelope and directional behavior only;
   - do not assert exact commands, evidence dictionaries, lineage identities,
     or internal mode sequences.

Update the canonical `test-vq2` surface so it tests the real safety kernel and
clean controller. Keep legacy navigation tests runnable separately, but do not
make green legacy architecture a prerequisite for a clean-course flight.

## Implementation and powered-iteration sequence

The first architecture-reset candidate is a meaningful boundary and may take
longer than one normal 15-25 minute iteration. After it exists, return to the
standing rapid-course loop.

For the first candidate:

1. Inspect current status and preserve unrelated/user changes.
2. Implement the clean stage and narrow runner seams above with one
   implementation owner.
3. Add the replacement behavior tests.
4. Run directly affected tests.
5. Run canonical `test-vq2` once.
6. If green, commit and push the exact clean candidate.
7. Run exactly one bounded:

   ```powershell
   .\scripts\dev.cmd flight-cycle visual-course
   ```

8. Analyze the first causal navigation blocker.
9. Make the smallest causal change in the clean architecture.
10. Repeat until authoritative `race_finished`.

Do not run `test-fast`, `test-unit`, benchmark, promotion, or broad matrices
before every powered attempt. Reserve them for meaningful course milestones or
final promotion.

Do not add another proof layer, replay abstraction, forensic capture,
simulation model, or review round before a flight once directly affected tests
and `test-vq2` are green, unless a hard blocker involves:

- nonfinite/out-of-envelope command;
- wrong authoritative race-gate ownership;
- stale raw control input;
- collision watchdog;
- live-lease concurrency; or
- unusable simulator state.

## Explicit non-goals

Do not:

- patch the old 17,144-line coordinator;
- swap only its servo and call that an architecture reset;
- preserve old navigation tests by reimplementing their internal modes;
- build a metric gate map without pose/geometry;
- make evidence dictionaries controller state;
- add gate-dependent vertical feedback signs;
- retain saturated collective through unobservable vertical state;
- call zero-yaw level hold "search";
- treat a fresh empty camera frame as a stale stream;
- let visual proof veto authoritative race credit;
- stop after Gate 0 or Gate 1;
- weaken collision, raw freshness, lease, command-envelope, reset/GO, hard
  timeout, or cleanup invariants.

## Handoff state

No controller code, tests, commits, or simulator state were changed during the
research audit that produced this handoff. The audited source commit was clean
and pushed. This handoff document is the intentional documentation change
created afterward.

Before the next powered attempt, ensure the exact controller candidate,
including this handoff if retained in the worktree, is committed and pushed so
the fast-cycle manifest sees a clean exact source state.
