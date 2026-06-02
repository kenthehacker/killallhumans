# AI Grand Prix — Gate-7 Failure Diagnosis And Research-Backed Plan

**Date**: 2026-04-03  
**Status**: Active  
**Scope**: Make the `sim_pybullet.runner` path trustworthy on the exact April 3 reproducer, and align the plan with the strongest relevant drone-racing literature.

---

## Exact Reproducer

```bash
python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json --use-detection --detector phase1
```

Latest reported result from that command:

- passed gates: `6 / 12`
- crash time: `28.71s`
- crash condition: ground impact at `z=0.09m`
- telemetry file: `logs/race_20260403_195735.csv`

This is not pure metadata mode and not pure perception mode:

- the runner attempts `detection` first
- if detection yields nothing, it falls back to `metadata`
- the actual failure is therefore a hybrid perception / guidance / control problem

---

## What Changed Since The Older Plan

The earlier stabilization document was based on a gate-5 / gate-6 era failure narrative.
That is now stale.

The current run shows:

- the drone now survives through gate 6
- the failure has moved into the helix, on the approach to gate 7
- repeated recoveries occur before the final crash
- the dominant problem is no longer just lookahead and altitude shaping
- the dominant problem is the **perception-to-guidance contract**

That means further progress will not come from tuning one more spline heuristic in isolation.

---

## Measured Facts From `race_20260403_195735.csv`

These are observations, not guesses:

- total logged control frames: `1378`
- target source usage:
  - `metadata`: `979` frames
  - `detection`: `399` frames
- target mode usage:
  - `spline`: `493`
  - `spline_tight`: `266`
  - `blend_gate`: `220`
  - `detection`: `399`
- recovery entries: `7`
- max absolute roll: `2.1944 rad` (`125.7 deg`)
- max absolute pitch: `0.9769 rad` (`56.0 deg`)
- target jumps larger than `1.0 m` between consecutive frames: `80`
- largest single target jump: `13.60 m` at `t=28.2083s`

Examples of impossible or highly suspicious target jumps:

- `t=12.2500s`: detection target jumps to `(46.55, -9.81, 0.52)`
- `t=14.7500s`: detection target jumps to `(58.06, -0.06, 6.06)`
- `t=24.1875s`: detection target jumps to `(43.21, -3.64, 9.16)`
- `t=28.2083s`: detection target jumps to `(36.71, -1.51, 0.97)` while the drone is near gate 7

Important logging caveat:

- the current CSV column named `dist_to_gate` is actually **distance from drone to commanded target**, not true distance to the current gate
- that means the present logs are useful, but still slightly misleading

---

## Diagnosis

## 1. The current detection-guidance algorithm is fundamentally flawed for tight racing

This statement applies to the **runner detection path**, not to the repo’s long-term architecture.

The current runner does this:

- run `Phase1GateDetector.detect()`
- take the single highest-confidence detection
- convert bbox center + heuristic distance directly into a world-frame target with `gate_detection_to_target()`
- send that target straight into the controller

That is fundamentally brittle for the helix because it has:

- no gate ID association
- no map consistency check
- no temporal track
- no outlier rejection
- no corner geometry
- no PnP pose estimate
- no filtered state correction
- no gate-region pass-through target

In other words: the controller is often being asked to chase a raw image-space guess, not a validated gate-relative flight target.

## 2. The repo’s overall architecture is not fundamentally flawed

The stronger direction already exists elsewhere in the repo:

- `race_pipeline.py`
- `estimation/gate_pnp.py`
- `estimation/ekf.py`
- `estimation/state_predictor.py`
- `estimation/gate_tracker.py`

Those modules are closer to what the literature recommends.

The gap is that `sim_pybullet/runner.py` still bypasses most of them and flies from a much thinner heuristic stack.

## 3. The failure is now mostly a perception / estimation / guidance problem, not a pure controller problem

The controller is not innocent, but it is not the first thing to fix.

Evidence:

- repeated multi-meter target jumps precede instability
- detection frames frequently command targets inconsistent with the local helix geometry
- the controller enters recovery `7` times, which is a symptom of upstream target instability
- a better controller cannot fully compensate for a target stream that is discontinuous or wrong-gate

## 4. The hybrid switching logic is still unstable

Even with improved metadata heuristics, the system continues to switch between:

- metadata spline / blend targets
- direct detection targets

That switching is not stateful and not quality-gated enough.
The result is target discontinuity during the exact section of the course that demands the most continuity.

---

## Papers We Actually Need For This Failure

These are the papers that should drive the fix path.

### Primary, immediate

1. **On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas**  
   Link: https://arxiv.org/abs/2510.13644  
   Why it matters:
   - closest match to this problem setting: onboard vision in an uninstrumented arena
   - supports the use of gate observations as filtered state corrections instead of direct one-frame flight targets
   - directly motivates the repo modules for EKF, latency prediction, and gate-based correction

2. **Robust Tightly-Coupled Filter-Based Monocular Visual-Inertial State Estimation and Graph-Based Evaluation for Autonomous Drone Racing**  
   Link: https://arxiv.org/abs/2603.02742  
   Why it matters:
   - shows the direction beyond loose coupling: corner-level visual constraints and robust filtering
   - useful as the end-state reference for what “not brittle to bad detections” should look like
   - not the first implementation step, but the right research north star for the estimator

3. **Time-Optimal Gate-Traversing Planner for Autonomous Drone Racing**  
   Link: https://lbfd.github.io/papers/TRO24_Hanover.pdf  
   Why it matters:
   - demonstrates that racing should optimize through **gate regions**, not just chase gate centers
   - reinforces that the target generator must respect drone dynamics and gate geometry
   - directly relevant once the target stream is filtered enough to support a real trajectory

4. **Perception-Aware Time-Optimal Planning for Quadrotor Waypoint Flight**  
   Link: https://arxiv.org/abs/2603.04305  
   Why it matters:
   - the helix is not only dynamically hard, it is visually hard
   - planning must preserve gate visibility margin instead of assuming perception is always available
   - this is the right way to think about the gate-7 to gate-12 section

5. **MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints**  
   Link: https://www.roboticsproceedings.org/rss20/p109.pdf  
   Why it matters:
   - strong reference for the eventual controller / tracker endgame
   - especially relevant for safety tunnels, collision avoidance at gates, and dynamic robustness
   - this is a later-phase controller upgrade, not the first fix

6. **Autonomous Drone Racing: A Survey**  
   Link: https://arxiv.org/abs/2301.01755  
   Why it matters:
   - broad framing across perception, planning, control, and learning
   - useful sanity check against overfitting to one subsystem

### Secondary, useful but not first

1. **Time-Optimal Online Replanning for Agile Quadrotor Flight**  
   Link: https://arxiv.org/abs/2203.09839  
   Why it matters:
   - relevant for online trajectory updates once perception is trustworthy

2. **Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning**  
   Link: https://arxiv.org/abs/2310.10943  
   Why it matters:
   - useful for deciding what not to do first
   - the current problem does not justify jumping to RL

### What not to lead with

- End-to-end RL is not the right first response to this failure.
- Learned inertial odometry is not the right first response either.
- The current stack is failing before those methods would become the bottleneck.

---

## Where The Algorithm Needs To Change

## A. `sim_pybullet/runner.py`

This is the main problem area.

Current issues:

- `_target_from_detection()` uses `detect()` instead of `detect_with_corners()`
- it picks the highest-confidence detection with no current-gate association
- it converts the detection directly into a flight target
- it has no rejection path for implausible target jumps
- it does not use `GateTracker`, `GatePnPEstimator`, `DroneEKF`, or `StatePredictor`

Required direction:

- treat detection as a measurement, not as the final target
- associate detections to the expected gate
- reject outliers before they reach control
- generate gate-relative targets from filtered state, not raw image center

## B. `flight_control/adapter.py`

Current issue:

- `gate_detection_to_target()` assumes bbox center + apparent width are sufficient to produce a trustworthy world target

That is acceptable as a debug fallback, but not as the main flight primitive in the helix.

Required direction:

- demote this to fallback / debug use
- stop using it as the authoritative target generator for aggressive flight

## C. `gate_detection/src/phase1_detector.py`

Current issue:

- the detector itself is not the main flaw
- the flaw is how its output is consumed

Required direction:

- use `detect_with_corners()` when running flight-relevant perception
- expose enough quality signals for gating:
  - confidence
  - bbox size
  - corner fit quality
  - optional candidate count

## D. `estimation/`

Current opportunity:

- the repo already contains the right building blocks, but the runner bypasses them

Required direction:

- integrate `GateTracker`
- integrate `GatePnPEstimator`
- use loose-coupled EKF correction first
- use `StatePredictor` for latency compensation before higher-aggression control work

## E. Planning / guidance

Current issue:

- the system still thinks in terms of “next target point”
- that is too weak for a tight helical gate sequence

Required direction:

- move toward gate-region guidance:
  - entry waypoint
  - gate pass-through point / region
  - exit waypoint
- preserve continuity and dynamic feasibility
- later, incorporate visibility margin / perception-aware cost

---

## Research-Backed Plan

## Phase 0 — Fix The Evidence

**Goal**: make the logs truthful enough that each failure is classifiable.

Record in telemetry:

- true distance to current gate center
- target jump magnitude from previous frame
- detection candidate count
- selected detection confidence
- selected detection estimated distance
- whether the selected detection matched current gate, next gate, or neither
- rejection reason when a detection is discarded
- reprojection error if PnP is used
- true recovery entry / exit timestamps

Acceptance:

- one log is enough to answer why a target was accepted
- no more misnamed `dist_to_gate` field

## Phase 1 — Stop Letting Raw Detections Command The Drone

**Goal**: eliminate one-frame detection spikes as a direct control input.

Immediate algorithmic changes:

- use `detect_with_corners()` in the runner path
- only consider detections geometrically plausible for the current or next gate
- reject detections that imply impossible target jumps
- if detection is rejected, keep the filtered metadata / trajectory target instead of snapping
- add temporal gate tracking or coasting, rather than frame-by-frame re-acquisition

Acceptance:

- zero multi-meter one-frame target jumps without an explicit logged fallback / rejection path
- gate-7 approach is continuous enough that recovery count does not increase rapidly

## Phase 2 — Convert Perception Into State Correction, Not Into A Raw Target

**Goal**: make perception a measurement source.

Algorithm:

- fitted corners -> PnP pose
- PnP pose -> gate-relative drone position estimate
- update EKF with loose-coupled gate correction
- forward-predict state for controller latency

This is the first serious paper-backed architecture step and should follow the `On Your Own` style first.

Acceptance:

- temporary perception dropouts no longer cause target discontinuities
- filtered pose remains plausible through the helix
- the vehicle can keep progressing when detection is intermittent

## Phase 3 — Replace Point Chasing With Gate-Region Guidance

**Goal**: stop treating the next gate as a single point.

Algorithm:

- define gate entry and exit waypoints using gate normal
- keep continuous trajectory targets through the helix
- use speed scheduling tied to local curvature / gate spacing
- begin migrating toward TOGT-style gate-region planning rather than center-hit heuristics

Acceptance:

- gate 7 through gate 12 are approached with continuous, dynamically plausible targets
- fewer recovery entries in the helix
- no “correct gate, wrong side / wrong angle” behavior

## Phase 4 — Make Planning Perception-Aware

**Goal**: stop planning paths that are dynamically feasible but visually fragile.

Algorithm:

- incorporate gate visibility margin
- penalize aggressive turns that drop the next gate out of the safe field of view
- bias the planner toward trajectories that preserve perception quality in the helix

Acceptance:

- detection availability remains higher during helix entry and mid-helix transitions
- metadata fallback becomes rarer for the right reasons, not because detection is ignored

## Phase 5 — Only Then Revisit The Controller

**Goal**: improve the tracker only after the target stream is sane.

Order:

1. keep the current controller as the baseline while fixing perception and guidance
2. if needed, migrate to `GeometricTracker`
3. later, evaluate MPCC++-style contouring control with safety constraints

Acceptance:

- no controller redesign is accepted unless the same run is already stable under a sane target stream

---

## Architecture Decision

The fastest credible path is:

- stop growing `sim_pybullet.runner` as an ad hoc autonomy stack
- either:
  - port the runner onto the existing `RacePipeline` estimation / planning modules, or
  - keep the runner as a thin PyBullet harness and move all serious autonomy into the shared pipeline

The slow path is:

- continuing to patch the raw `detect()` -> `gate_detection_to_target()` -> controller chain

This plan recommends the first path.

---

## What Success Looks Like

Near-term trustworthy success:

- same reproducer passes gate 7 reliably
- no multi-meter target spikes from accepted detections
- recovery entries drop sharply
- log explains every accepted / rejected perception update

Meaningful next milestone:

- full helix completion through gate 12 in the same runner path
- target generation driven by filtered gate-relative state, not raw one-frame image geometry

Long-term:

- PyBullet evaluation and competition-facing stack share the same perception / estimation / planning / control architecture

---

## Non-Goals For This Plan

- no code changes are described here as already completed
- no claim that the current controller alone is the root cause
- no claim that RL is the next required step
- no claim that tightly coupled VIO should be implemented before a loose-coupled baseline is working

---

## Bottom Line

The current runner failure is not just “more tuning needed.”

The **current detection-to-target algorithm in `sim_pybullet.runner` is fundamentally too weak for tight drone racing** because it turns raw single-frame image detections directly into flight targets.

The **repo’s broader architecture is not fundamentally flawed**. The stronger path is already present in the estimation and pipeline modules.

The correct plan is to:

1. make the telemetry truthful,
2. stop raw detections from commanding the drone directly,
3. convert perception into filtered state correction,
4. move guidance toward gate-region planning,
5. only then tune or replace the controller.
