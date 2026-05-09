# 150-agent cross-validation reconciliation — 2026-05-09

149 of 150 Opus 4.7 sub-agents returned reports (one prompt had a
typo'd `subagent_type=general-repurpose`). Findings are deduplicated,
disagreements resolved, and ranked by severity × ease-of-fix.

Numbers in brackets are agent count corroborating each finding.

## P0 — ship-stopping bugs surfaced by validation

These either contradict the goals of the current diff or break
production paths the diff was meant to harden.

### P0-1. Replanned racing line mis-targets after first replan [agent #150]
`sim_pybullet/runner.py:_target_from_sim_metadata` computes
`min_arc = self._racing_line.waypoint_arc(self.sequencer.gates_passed)`,
but after `_maybe_replan` rebuilds the line from
`[drone_pos, ...remaining_gate_centres]` the new spline's waypoint
indexing starts fresh. `gates_passed` is cumulative across the race.
A replan at `gates_passed=2` on a 5-gate course requests
`waypoint_arc(2)` on a 4-waypoint line — clamped to the spline end —
so the controller targets the *final* gate, skipping every intermediate
remaining gate. **Every replan poisons control.**

Fix: snapshot the baseline (`self._replan_gates_baseline =
self.sequencer.gates_passed`) on replan, then compute
`local_idx = sequencer.gates_passed - self._replan_gates_baseline`
in `_target_from_sim_metadata`. Reset to 0 in `__init__` and `_reset`.

### P0-2. `_reset()` does not rebuild the racing line [27 hits, including #50, #54-55]
`_maybe_replan` overwrites `self._racing_line` from the drone's mid-flight
position. Pressing `r` calls `env.reset()` (drone teleports back to
start) and `sequencer.reset()` (gate state cleared) — but
`self._racing_line` is left as the post-replan stub. The drone will then
chase a line that doesn't pass through the start position and may skip
already-passed gates.

Fix: extract the original line construction (`__init__:177-181`) into
`_build_initial_racing_line()` and call it from both `__init__` and
`_reset`. Also clear `self._prev_target_z = None` and
`self._last_lateral_err = 0.0`.

### P0-3. Mixer inverted-attitude sends MAX throttle into the ground [agent #112]
`flight_control/mixer.py:104-114` floors `cos_correction` at `0.3`,
but when the drone is inverted (cos(roll)*cos(pitch) negative), the
clamp forces the denominator positive and `thrust_needed` becomes
large positive — saturating throttle to **1.0 while inverted**, full
thrust toward the ground. No log, no fault. Aerobatic recovery →
crash.

Fix: replace the `max(..., 0.3)` floor with an inversion guard:
```python
if cos_tilt < 0.1:
    throttle = 0.05  # idle motors, don't fly into the ground
else:
    thrust_needed = (mass * (g + az_world)) / cos_tilt
    throttle = _clamp(thrust_needed / max_thrust_n, 0.05, 1.0)
```

### P0-4. `fused_gate_detector.py` does not exist [6 hits]
`sim_pybullet/runner.py:269` does
`from fused_gate_detector import FusedGateDetector` (lazy). Any
invocation of `--detector fused` (mentioned in `CLAUDE.md`) raises
`ModuleNotFoundError` mid-flight. `ARCH.md:138` documents the class
as if it exists.

Fix: either restore the module from git history (or
`gate_detection/claude_debugging/gate_detector_v3.py`) **or** remove
the `--detector fused` branch from runner + `argparse` choices.

### P0-5. Cascaded rate-vs-angle PID nesting is absent [agent #120]
`flight_control/pid.py` exposes a single flat `PIDController`; only
`pid_vz` and `pid_yaw` are wired (`controller.py:20-45`). Roll/pitch
loops don't exist. A single PID on attitude error has no inner
rate-loop and is **structurally unstable beyond hover**. The
`max_body_rate` field in `TrackerConfig` is declared but never
applied (agent #109 confirmed) — same shape as the missing nesting.

Fix: implement `CascadedAttitudePID` (outer P angle → inner PID rate)
per axis. Until then, document the limitation in `ARCH.md` and clamp
target attitudes more aggressively.

## P1 — major correctness gaps in the new diff

These violate the iff-highlighted contract or the dynamic-replan
contract under realistic conditions.

### P1-1. Cooldown bypassed for level-triggered signals [agents #29, #30, #50, #150]
`off_track` and `sustained_lateral_error` are level-triggered. While
the condition persists, they flip True every tick. With
`cooldown_seconds=0.5` and a sustained recovery state, the replanner
re-fires every 120 frames at 240 Hz — a slow replan storm. Worse,
post-replan the lateral counter resets to 0 (correct) but `off_track`
stays latched on `state.RECOVERY`.

Fix: edge-trigger both. Track `_was_off_track`, `_was_sustained` bools
and only set the trigger field on the rising edge. Reset both in
`mark_replanned` and `reset`.

### P1-2. `off_track` trigger is dead in production [agents #34, #80, #149, #150]
`sim_pybullet/sequencer.state` only emits `COMPLETED`/`RACING` —
never `RECOVERY` (no off-track logic implemented in this sequencer).
`DynamicReplanner.evaluate` reads `state.name == "RECOVERY"`. So
`off_track` never fires under PyBullet. Tests pass because they go
through the platform-agnostic sequencer.

Fix: port the off-track detection from `gate_sequencing/sequencer.py:247-252`
into `sim_pybullet/sequencer.py:update`, expose
`state.name == "RECOVERY"` accordingly, and add transition tests.
Better still: P2-1 below (single sequencer).

### P1-3. Sequencer reset desyncs replanner counters silently [agents #28, #149]
`DynamicReplanner._last_seen_crashes` is a monotonic baseline. After
`sequencer.reset()` (without paired `replanner.reset()`), `n_crashes=0
< _last_seen=N`, so the next N real crashes are swallowed.
`RaceRunner._reset` does pair them today, but any future caller (test,
sub-race restart, RL rollout) breaks the contract silently.

Fix: replace count-deltas with set-of-IDs. Track
`_seen_crash_ids: set[str]`; trigger on `set(crashed_gate_ids) -
_seen_crash_ids`. Set-diff auto-resyncs when the sequencer's set
shrinks. Or, defensively, add `if n < self._last_seen: self._last_seen
= n` clamp.

### P1-4. Same-tick crash + pass — `_last_event` overwritten [agents #11, #12, #144, #146]
`runner.run`: `mark_collision(G1)` runs at lines 307-318 (sets
`_last_event = "crash"`), then `sequencer.update(pos)` at line 321 can
classify the same segment as `"pass"` and overwrite the event. Result:
G1 ends up in **both** `crashed_gate_ids` and `passed_gate_ids` with
`last_event = "pass"`. Replanner downstream sees no crash trigger;
`crashed_into_gate` still returns G1 — invariant violation.

Fix: in `sim_pybullet/sequencer.update`, short-circuit when the current
gate already has a fresh `mark_collision` from this tick — store
`_collision_marked_this_tick: Optional[str]` set inside `mark_collision`,
reset at the end of `update`. The crash mark wins.

### P1-5. `mark_collision` not state-gated, not idempotent [agents #14, #15, #50, #144]
`mark_collision`:
- Records crashes during `WAITING` (pre-start) and `COMPLETED` (post-finish).
- Has no dedupe — repeat calls append duplicates; only the runner's
  `_last_contact_gate_id` saves us today.
- Doesn't advance `_current_idx`, so a crashed gate is silently treated
  as still-highlighted, producing repeat crash events.

Fix: early-return if `state in (WAITING, COMPLETED, TIMED_OUT)`.
Idempotent appending: skip if `last_event == "crash"` and `_crashes[-1][0]
== gate_id`. Optionally consume the gate (advance `_current_idx`,
clear `_prev_position`) so the next pass-through check can't re-credit it.

### P1-6. Geometric crash zone empty under production margin [agents #2, #4, #136, #150]
At `pass_through_margin=1.5` with default border 0.15m, the
margin-stretched opening (0.9m half) exceeds the outer frame (0.75m
half). Any crossing inside the frame is classified as `pass`, so
geometric crash detection is dead code in production. `mark_collision`
(PyBullet contacts) is the only authoritative path.

Fix (one of):
- Cap the opening at the outer frame: `half_w_pass = min(half_w *
  margin, half_w + border_width)`. Guarantees the crash zone is
  never empty regardless of margin.
- Decouple: `crash_margin: float = 1.0` separate from
  `pass_through_margin`. Crash always uses the bare opening.

### P1-7. Crash/miss lists not deduped per fly-by [agents #1, #25, #144]
Every tick that re-crosses the highlighted gate's plane outside the
opening appends to `_crashes` / `_misses`. A drone oscillating against
the frame produces N entries for one physical event. Replanner
doesn't double-fire (cooldown), but counts are wrong and downstream
reasoning over the lists is corrupted.

Fix: gate the classification on a state transition. Skip append if
`_last_event` is already `"crash"` / `"miss"` for the current target
(it clears when target advances).

### P1-8. NaN-poisoned `_last_replan_time` permanently disables cooldown [agent #143]
`should_replan`: `sim_time - self._last_replan_time < cooldown` —
when `sim_time` is NaN, comparison returns False per IEEE-754 → falls
through to `return True`. NaN forces replan, then `mark_replanned`
writes NaN into `_last_replan_time` → cooldown permanently disabled
for the rest of the race.

Fix: `if not math.isfinite(sim_time): return False`.
Refuse non-finite writes in `mark_replanned`. Same guard for
`lateral_error` in `evaluate` (NaN currently silently disables the
sustained-error counter — agent #146).

### P1-9. `_prev_target_z = None` after replan → unbounded z-step [agents #52, #126, #148]
After replan, the slew limiter is disabled for one tick. The new spline
samples at lookahead distance can sit on the next gate's altitude. If
drone z=1.5m and next gate z=4.0m, target z jumps 2.5m in one tick —
~180× the normal 0.0083m budget. Worst time to drop the limiter:
post-perturbation, drone is already off-nominal.

Fix: seed instead of nuke:
`self._prev_target_z = float(drone_state.position[2])`.

### P1-10. iff-highlighted: stray cross-gate passes are silent [agent #16]
Drone passes G2 cleanly while G1 is highlighted. Sequencer correctly
issues no credit (good). But it also doesn't *log* the off-sequence
event. The replanner can't react to "drone is now downstream of where
it should be" — could happen during a recovery overshoot.

Fix: scan non-current, non-passed gates for plane crossings inside
the opening; record into `_off_sequence_passes` (id+position+ts);
add a `ReplanTrigger.off_sequence_pass` reason that rebuilds the
line from the drone's actual position.

### P1-11. Skip-after-N-failures policy missing [agents #146, #149]
After multiple crashes/misses on the same gate (drone wedged in
frame), the replanner replans repeatedly back to the same centre.
Cooldown spaces the replans but doesn't escalate. Drone orbits
indefinitely.

Fix: track `_miss_counts[gate_id]` / `_crash_counts[gate_id]` in the
sequencer; once `>= miss_skip_threshold` (default 2), advance
`_current_idx` past the gate and add to `skipped_gate_ids`. Add a
`ReplanTrigger.skip` action.

### P1-12. Telemetry doesn't log the new fields [agents #59, #137, #146]
`_log_frame` writes 26 columns, none of `replan_count`,
`replan_reasons`, or `crashed_into_gate`. Post-mortem cannot answer
"did the drone crash?" or "did the replanner fire?". Cross-validation
runs are blind.

Fix: append three columns to header + row. Pipe-delimit reasons
(commas would break CSV). Initialise `_last_replan_reasons` to `[]`
each tick after logging so reasons appear only on the firing tick.
Bump CSV schema version; write the version as row 0.

### P1-13. `runner._maybe_replan` consumes one-tick-stale lateral error [agents #46, #146]
`_maybe_replan` reads `self._last_lateral_err`, which is set in
`_target_from_sim_metadata` (called *after* `_maybe_replan` in the
loop). Tick 0 reads the init value `0.0`. Single-tick staleness is
masked today by the multi-frame `sustained_frames` debounce, but a
detection-mode run (when `_target_from_sim_metadata` may be skipped)
keeps the value permanently stale.

Fix: compute lateral error once at the top of the loop body, before
both `_maybe_replan` and `_get_target`. Drop the assignment inside
`_target_from_sim_metadata`.

### P1-14. `race_pipeline.py` doesn't use the replanner [agent #149]
The dynamic replanner was only wired into `sim_pybullet/runner.py`.
The real-flight path (`race_pipeline.py:_control_callback`) only does
a sequencer-driven reference *override*; it never rebuilds
`self.trajectory` after a crash/miss. Sim passes, hardware regresses
silently.

Fix: import `DynamicReplanner` in `race_pipeline.py`, instantiate in
`__init__`, evaluate after `sequencer.update`, regenerate trajectory
on `should_replan`. Add a parallel integration test under `tests/`.

### P1-15. `scripts/visual_demo.py` still uses the old gate_fallback path [agent #149]
`visual_demo.py:580-612` keeps the heuristic gate_fallback target source
that PRD `race_01_fail_prd.md` explicitly calls out as the cause of
the gate-7 failure mode. The PRD acceptance criterion ("no gate is
passed via `gate_fallback`") cannot be satisfied by the current diff.

Fix: port the contact-poll + replan integration from `runner.py`.

## P2 — architectural recommendations

### P2-1. Collapse the two sequencers into one [agent #144]
`sim_pybullet/sequencer.py` and `gate_sequencing/sequencer.py` ship
parallel implementations. Drift is already present (state-shim
incompleteness, margin defaults, missing detection_active kwarg,
`Gate.pose.x/y/z` vs `GateSpec.position`). The bug class **already
hit** us during this diff (off_track dead, P1-2).

Fix: delete `sim_pybullet/sequencer.py`. Replace with a thin adapter
`adapt_gate_to_spec(g) -> GateSpec` and have the runner use
`gate_sequencing.GateSequencer` directly. One-arg `update(pos)` becomes
`update(pos, gate_detected=True, detection_active=False)`. Saves ~240 LOC.

### P2-2. Promote `RaceState` to a shared contract [agents #144, #150]
Move `RaceState` enum to `gate_sequencing/race_state.py` (or
`common/`). Have both sequencers import it. Replace the
`SimpleNamespace(name=...)` shim with the real enum. Then
`DynamicReplanner.evaluate` can use `is RaceState.RECOVERY` instead
of stringly `getattr`.

### P2-3. `_SequencerLike` runtime check [agent #131]
Add `@runtime_checkable` and an `isinstance` assertion at
`DynamicReplanner.__init__`. Catches contract drift the next time
either sequencer drops a member. Pair with `mypy --strict planning/`
in CI for the static side.

### P2-4. Per-config racing-line cache [agent #146]
`racing_line_cache.json` is a single-entry store. Alternating configs
clobbers the cache, defeating iter-33's determinism guarantee. Switch
to per-config files keyed by hash: `racing_line_cache/<hash>.json`.

### P2-5. Telemetry CSV schema versioning [agent #138]
Embed `#schema_version,2` as row 0; bump on additive vs structural
changes. Centralize column list as a module constant.

### P2-6. Cache-key invalidation includes controller gains [agent #95]
`racing_line.py:_compute_cache_key` doesn't hash `TrackerConfig`. The
sim oracle (`_kinematic_eval`) gains drift from `mpc_tracker.py` are
silent — iter-40 sync was unverified because cache replay bypassed
re-selection. Hash all inputs that affect selection; bump cache version.

## P3 — module-level optimization recommendations

Highest-leverage suggestions across modules, ranked.

### P3-1. EKF: variable measurement noise R from PnP quality [agent #62]
Current: fixed `pnp_position_noise_std=0.3` isotropic. Reality: PnP
position error is range-dependent and anisotropic (depth axis ~10×
noisier than lateral). Pass `GatePose.distance` and
`reprojection_error` into `update_pnp_position` and build R per-update.

### P3-2. Trajectory: gate-region cost (TOGT) [agents #81, #86, #112]
Per Qin 2024, gates are regions, not points. Add per-gate
`(u_i, v_i) ∈ [-1, 1]²` decision variables clipped to gate aperture;
co-optimize with time. Subsumes the post-hoc `_inflate_sharp_turns`
and `_inflate_vertical_climbs` heuristics. Expected: 3-7% lap-time
gain on tracks with >60° turns.

### P3-3. DroneConstraints / TrackerConfig calibration mismatch [agents #83, #106, #107]
`DroneConstraints` defaults are for a 1kg, 7N-class racer; the sim
flies a 27g CF2X. Mass ~37×, max_thrust ~75×, max_accel ~3× over the
sim's 6.87 m/s² tilt-clamped ceiling. `feedforward_accel=0.50` halves
the literature value (1.0). Result: planner generates references the
controller can't track.

Fix: ground all constraints in CF2X physics (`mass=0.027`,
`max_acceleration=g·tan(0.35)≈3.58`, etc.) or guard with assertions
that prevent silent mismatches.

### P3-4. State predictor: actuator first-order model [agents #79, #146]
Predictor advances kinematic state by `total_latency=42ms` but
ignores motor τ≈30-50ms. Effectively reintroduces the bug Romero's
predictor was supposed to fix. Add command-queue + first-order rise
model.

### P3-5. PnP: 3-of-4 fallback + Mahalanobis ID [agents #66, #67, #71, #72]
`estimate_gate_pose` rejects any non-(4,2) input. SOLVEPNP_P3P with
3 corners + EKF prior to disambiguate would salvage edge-occluded
gates — exactly the case during gate transit. Combine with Hungarian
+ Mahalanobis assignment in the gate tracker (agent #72) for
ID-stability on overlapping helix gates.

### P3-6. Phase1 detector: per-frame percentile auto-tune [agent #121]
S/V thresholds (60/200) are scene-absolute. VQ1 will shift the
histogram and the detector will silently starve. Replace with
`max(self.sat_thresh, np.percentile(sat, 95))` — exploits the
"top-tail of scene" premise the docstring already states.

### P3-7. Phase1 NMS IoU 0.4 → 0.6 + center-distance gate [agent #123]
At 0.4, adjacent helix gates suppress each other. Lift to 0.6 and
add a centre-distance check (suppress only if centres within 30%
of smaller dim) — preserves dedup but keeps neighbouring gates.

### P3-8. Mixer: nonlinear thrust magnitude [agent #111]
Replace `(g+az)/cos*cos` with `m·sqrt(ax² + ay² + (g+az)²) /
max_thrust_n`. Removes small-angle error. Drop the cos floor of 0.3
(now unnecessary). Surface saturation as `TRPYCommand.mixer_saturated:
bool` so MPC can shed lateral demand instead of being silently truncated.

### P3-9. PID: derivative-on-measurement + LP filter [agent #118]
`PIDController` derivatives are derivative-on-error and unfiltered →
derivative kick + noise amplification. Switch to D-on-measurement
with a first-order LPF. Reset `prev_measurement` lazily on first call.

### P3-10. Anti-windup back-calculation [agent #117]
PID integrator clamps but doesn't couple to actuator saturation. Add
back-calculation: `_integrator += kb * (saturated - unsaturated) * dt`
where `kb ≈ 1/Ti`.

### P3-11. ILC residual model is too weak [agent #102]
Time-indexed lerp assumes repeatable disturbance at the same `t`.
Breaks under replans / changed initial conditions. Re-key on arc-length
`s` along the racing line; add state-conditioned residual `ff_acc(s,
v, ψ)`. Bump schema version.

### P3-12. PyBullet seed / NumPy seed in env [agent #134]
`DroneRaceEnv.__init__` doesn't seed NumPy / Python random / PyBullet
contact pair iteration. Determinism only works because
`scripts/benchmark.py` seeds NumPy externally — `env.py` itself offers
no contract.

Fix: add `seed: Optional[int]` to `RaceConfig`; on init, seed NumPy +
random + `setPhysicsEngineParameter(deterministicOverlappingPairs=1)`.
Re-apply on `reset`.

### P3-13. Benchmark covers only race_01 [agent #132]
`scripts/benchmark.py` hardcodes `race_01.json`. Five sibling configs
exist (`figure8`, `grand_tour`, `slalom`, `straight_hairpin`,
`vertical_cliff`); all silently uncovered. ILC section boundaries are
also race_01-specific.

Fix: discover `sim_pybullet/configs/*.json`; loop benchmark over all.
Move ILC section boundaries to per-config metadata.

### P3-14. Last-resort descent-protection clips low gates [agent #149]
`runner.run`: `alt<1.0 ∧ vz<-2.0 → climb to 1.5m at zero vel`. But
race_01 gates 1/3 sit at z=1.5m with interior_height=1.2m → window
[0.9, 2.1]m. The override fires *inside* the gate, blanks XY tangent,
and the climb pulls the drone through the upper bar.

Fix: gate the override on horizontal proximity to the next gate. Drop
alt threshold to 0.5m and climb target to 0.8m (under all gate
bottoms in the corpus).

### P3-15. Replan_count threshold in benchmark [agents #60, #146]
`benchmark.py` thresholds: gate-pass-rate, tracking error, no-crash —
but no `max_replan_count`. A replan storm passes the benchmark. Add
`max_replan_count: 1` (one replan tolerates a recoverable
perturbation).

## Disagreements resolved

| Topic | Disagreement | Resolution |
|-------|--------------|------------|
| RacingLine 2-waypoint | Agents A: "no bug, math is fine"; agent B: "zero-tangent endpoints" | B wins. Reflected ghosts (`padded = [2*wps[0]-wps[1], ..., 2*wps[-1]-wps[-2]]`) preserve direction at endpoints; correct for any N≥2. |
| Replanner cooldown init=-inf | Agents flag as bug; agent #142 confirms intentional | Intentional. First-tick replan must fire. Document with a comment. |
| `pass_through_margin` 1.5 default | "Lower to 1.2 / 1.0" vs "Decouple crash margin" | Decouple. Production keeps 1.5 for pass tolerance; crash zone uses bare opening (or `crash_margin=1.0`). Captures both intents. |
| EKF process noise vs measurement noise | Both flagged as primary issue | Both apply. Process noise tuning unblocks racing-rate dynamics; measurement-noise-from-PnP-quality unblocks correction quality. Independent fixes. |
| iff-highlighted "stray pass detection" needed | One agent says yes, others silent | Useful but NOT P0. Add as P1-10 (low-risk addition; off-sequence trigger could meaningfully replan when drone overshoots into next gate). |

## Meta findings

- 1 of 150 sub-agents errored on a typo'd `subagent_type=general-repurpose`
  in the dispatch prompt. Functional output: 149/150 = 99.3% yield.
- ~15% of agents flagged the "single sequencer / shared contract"
  architectural smell independently — strong signal P2-1 should be
  promoted to P1 or done first.
- Several agents over-confidently classified findings as P0/CRITICAL
  that turned out to be misreads (e.g. claimed `mark_replanned` could
  be called twice per tick when it cannot, claimed `_reset` had race
  conditions in a single-threaded loop). I down-graded these in the
  reconciliation; raw severity ratings in the agent reports should not
  be trusted without rechecking.
- Five agents confused the missing `fused_gate_detector.py` with the
  existing `gate_detector.py`. The file IS missing; treat any analysis
  of the "fused" path as void.

## Suggested next-step ordering (1-week plan)

1. **Day 1 (must-do)**: P0-1, P0-2, P0-4, P1-1, P1-2, P1-3, P1-4,
   P1-8, P1-12. All are 1-3 line fixes for correctness; tests are
   straightforward. P0-1 is the highest priority — every replan today
   is poisoning control.
2. **Day 2**: P1-5, P1-6, P1-7, P1-9, P1-13. Sequencer/replanner
   hardening. Add tests covering the matrix of trigger + cooldown +
   reset paths.
3. **Day 3**: P2-1 (collapse sequencers) + P2-2 (RaceState contract).
   Eliminates entire bug class; unlocks P1-2 for free.
4. **Day 4**: P0-3 (mixer inversion), P0-5 (cascaded PID), P3-3
   (CF2X calibration). Real-hardware safety.
5. **Day 5**: P1-14, P1-15 (wire replanner into race_pipeline.py and
   visual_demo.py). Without these, the diff doesn't satisfy the PRD.
6. **Days 6-7**: P3 optimizations, ranked by which benchmark
   threshold they'd improve.

---

149 of 150 reports captured. Full per-agent transcripts available in
the prior tool-call output (search by `agentId`). Open
follow-up agents via `SendMessage` with the agent ID for deeper
investigation on any specific finding.
