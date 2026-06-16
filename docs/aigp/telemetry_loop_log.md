# AIGP Telemetry-Driven Iteration Loop

Autonomous loop started 2026-06-12. Mandate: **at each iteration read flight
telemetry, detect anything gone wrong (esp. the drone "spinning in circles"),
identify an improvement, test it, find gaps, repeat** — max 50 iterations.

Regression gate (fast, reliable on this Windows host, no pybullet needed):

```
python -m pytest control/tests estimation planning gate_sequencing competition flight_control/tests -q -o addopts="" -p no:randomly
```
Baseline: **288 passed** (~19 s).

Telemetry anomaly detector (run every iteration):
```
python scripts/analyze_telemetry.py --all captures/
```

---

## Iteration 1 — diagnosis + tooling

**Telemetry read:** all committed `captures/telemetry_*.jsonl.gz` (5 files,
~480 s each) show **position/velocity/yaw frozen at 0, gates_passed=0, and
`cmd_yaw` pinned at −3.0866 rad for 100% of ticks** (bit-for-bit identical
command every tick).

**Findings:**
1. The 4 committed captures were produced by `aigp_vq1_run.py --dry-run`
   (`FakeAdapter`): `get_telemetry()` returns a static origin snapshot and
   `send_attitude()` is a no-op. **They cannot diagnose real flight.** A 5th
   capture has the *same* constant command but a sim-style timestamp
   (t_us≈446 s, not the host's monotonic ≈424 634 s) — consistent with a real
   run that received **frozen zero-position telemetry** (controller flying
   blind → constant max-yaw command → physical spin). This matches the
   2026-06-09 audit MAJOR: "subscription tasks die silently → frozen
   telemetry, control flies blind on stale state."
2. **Recorder bug:** `aigp_vq1_run.py` called `trajectory.at_time()` which does
   not exist (only `sample()` / `find_closest_forward()`); the AttributeError
   was silently swallowed so `ref_pos` was `None` in *every* sample — the
   planned-vs-actual comparison never ran. **Fixed** → use
   `trajectory.sample(_ref_progress_time)`.
3. **Spinning root causes already fixed** by the recent yaw-spin commits
   (a7343d0/fb5d079), post-dating the audit: GeometricTracker now rotates
   thrust into the yaw frame (mpc_tracker.py:290-294, Blocker 1), saturates
   commanded descent (≈234-235, Blocker 2), and SimplePositionTracker uses
   NED-correct signs (≈447-463, Blocker 3). Verified by 30 passing control
   tests. **Still needs real-sim validation.**
4. Sim is running (DCGame-Win64-Shipping, session 1) but **not in VQ mode** —
   no MAVLink on 14550; it only listens on UDP 14560/5601. Entering VQ mode is
   a GUI action that can't be automated from a headless session, so fresh
   real telemetry is currently blocked.

**Changes:** fixed recorder `at_time`→`sample`; added
`scripts/analyze_telemetry.py` (flags FROZEN / SPINNING / CIRCLING / CMD
SATURATION / CONSTANT COMMAND / NO REFERENCE / GATE STALL); fixed a
Windows-path-separator test failure in `tests/test_benchmark_matrix.py`.

**Tests:** 288 core tests pass; analyzer flags all 5 captures correctly.

**Gaps for next iterations:** real-sim telemetry blocked on VQ-mode entry;
the live path can fly blind on frozen telemetry with no detection → harden
the pipeline to *catch* stale/frozen telemetry (iteration 2). Remaining audit
blockers: EKF observability (4), replan EKF re-init (5), PnP gating (6,12),
replanner trigger loss (7,8,9), trajectory feasibility (10,11), sim-time
stamping (13,14), benchmark honesty (15-19).

---

## Iteration 2 — frozen-telemetry watchdog

**Telemetry read:** re-confirmed via `analyze_telemetry.py` that the failure
signature is a *non-advancing* state — the controller never saw the vehicle
move. The previous code had **no detection** for this; it is exactly the
"AI could not catch the spinning" gap.

**Improvement:** added `RacePipeline._check_telemetry_freshness()` — a
detection-only watchdog called first in the control callback. It counts
consecutive ticks whose `telem.timestamp_us` does not advance (or is missing)
and logs one loud `ERROR` per stall once the feed has been frozen for
`_telem_stale_tick_limit` (50 ≈ 0.5 s @ 100 Hz) ticks; `_telem_frozen_ticks`
accumulates the total and is reported in the run summary
(`aigp_vq1_run.py`). Flight commands are unchanged (no new safety risk).

**Tests:** `tests/test_telemetry_watchdog.py` (4 cases: advancing feed never
flags; frozen feed counts and warns exactly once; recovery resets and
re-arms; missing timestamp treated as frozen). Regression gate: **304
passed**.

**Gaps:** watchdog is detection-only — a future iteration could decide a safe
response (e.g. controlled hover/abort) when frozen, but that needs a
trustworthy fallback state. Next: audit Blocker 5 (mid-race replan
re-initializes the EKF — wipes yaw/biases at the worst moment).

---

## Iteration 3 — replanner trigger-loss fix (Blocker 7); triage of 5

**Triage (verify before fixing):**
- **Blocker 5 (replan re-inits EKF): ALREADY FIXED.** `ekf.initialize` is
  called only in `configure()`, *outside* `_build_trajectory_from`;
  `_maybe_replan` rebuilds the trajectory without touching the EKF. No change
  needed.
- **Blocker 7 (replanner trigger loss): STILL PRESENT — fixed this iteration.**

**Bug:** `DynamicReplanner.evaluate()` reports each crash/miss/off-track event
once on its rising edge then consumes it (seen-sets + level latches advance
every tick). If that single edge lands inside the 0.5 s replan cooldown,
`should_replan()` rejects it and the event is **lost forever** — a strut graze
during cooldown ⇒ no recovery; an off-track entry during cooldown ⇒ stuck in
RECOVERY. The existing unit tests *enshrine* the fire-once `evaluate()`
contract, so the fix belongs at the integration layer, not in `evaluate()`.

**Fix:** `RacePipeline._maybe_replan` now keeps a `_pending_trigger`: a
triggered-but-unserved evaluation (blocked by cooldown, or whose rebuild threw)
is merged (`_merge_triggers`) into later ticks and served the moment the
cooldown expires. Cleared only after a rebuild succeeds. `evaluate()` is
unchanged, so all fire-once edge tests still hold.

**Tests:** new `test_cooldown_deferred_crash_is_served_after_cooldown`;
31 replanner tests pass; regression gate **304 passed**.

**Gaps:** the same consume-on-read loss exists in
`sim_pybullet/runner.py` (practice sim, audit Blocker 7 cont.) — lower
priority than the live path. Audit Blocker 9 (replan runs ~1.8 s of
optimization synchronously in the 100 Hz control callback) is real and
architecturally bigger — candidate for a later iteration.

---

## Iteration 4 — closed-loop "no-spin" regression test

**Why:** Blockers 1-3 (the spinning root causes) are fixed in code but had
**no test that would catch a regression** — exactly the "AI could not catch
the spinning" gap. The VQ1 start requires facing gate-0 behind the drone
(yaw ≈ ±π), which is where the world-frame-extraction bug diverged.

**Improvement:** `control/tests/test_tracker_no_spin.py` — a black-box closed
loop that feeds the tracker's `AttitudeCommand` through textbook NED quad
dynamics and integrates, asserting the drone **converges** to the target from
every heading (0, ±π/2, ±π, ±3.0 rad). Verified the test has teeth: with the
old world-frame extraction the yaw=π case diverges 5.83 m → **293 m** (spin
signature) and fails; with the fix it converges to 0.00 m.

**Tests:** 10 new cases pass; regression gate **314 passed**.

**Status after 4 iterations:** the user's two core concerns are addressed —
spinning is fixed *and* now guarded by a regression test, and frozen/blind
telemetry is now detected live + offline. Remaining backlog (lower urgency,
verify-before-fix): Blockers 6/12 (PnP gating/scale), 9 (sync replan in
control loop), 13/14 (sim-time stamping / gate-normal sign — may be fixed),
benchmark honesty (15-19), and real-sim validation (blocked on GUI VQ-mode
entry).

---

## Iteration 5 — closed-loop flight harness (real telemetry without the GUI sim)

**Why:** the GUI sim won't enter VQ mode headlessly and the committed captures
are frozen dry-runs, so there was *no* source of real flight telemetry to read
each iteration. Built one.

**Tool:** `scripts/sim_closed_loop.py` drives the **real** `RacePipeline`
control callback (geometric tracker + gate sequencer + dynamic replanner)
against textbook NED point-mass dynamics, integrated in sim time — runs in
<1 s, writes the standard telemetry schema, and the analyzer reads it
directly. Scope: `use_ekf=False`, `use_detection=False` to isolate
tracking+sequencing (EKF/detection are a later layer).

**Telemetry read (first clean signal):** with the sequencer inactive the drone
**tracked the full trajectory to gate-5's position** (net 165 m, circle_ratio
1.0, ref_pos populated) — confirming the tracker is healthy and *not* spinning
in closed loop. Two harness-setup bugs were found and fixed along the way
(must call `sequencer.start()`; must seed a positive sim timestamp / real
`_race_start_time` or the wall-clock fallback trips the 8-min timeout on tick
0 — note: a real sim whose first stamp is exactly 0 hits that same fallback).

## Iteration 6 — off-track = cross-track, not distance-to-gate (audit MAJOR)

**Telemetry read:** once the sequencer was active, the harness showed a
**replan storm** (10 replans/15 s), altitude divergence, and **0 gates**. Trace
showed `RECOVERY` at t=0.01 with reason `off_track` — the drone, at the start,
perfectly on the line but 23 m from gate-0.

**Root cause (sequencer.py:510-515):** off-track was point distance to the
target gate (`off_track_distance*3` = 15 m). On a course with gates ~20 m
apart the drone is **always > 15 m from its target gate at the start of every
leg**, so it latched RECOVERY constantly → reference overridden to the raw
gate centre + thrust cut + replan storm.

**Fix:** off-track now measures **cross-track distance to the current leg's
segment** (`previous gate → current gate`, with a captured `_track_origin`
for the first leg) via `_point_to_segment_distance`. Reset clears the origin.

**Result:** the closed-loop harness now reports **`race_complete`, 6/6 gates,
0 replans, circle_ratio 1.0** in 15.3 s — analyzer says *OK, no anomalies*.
Also removed a false-positive analyzer rule (a yaw *angle* setpoint near ±π on
a −X course is the correct heading, not "saturation"; the real frozen-feed
case is still caught by CONSTANT COMMAND / FROZEN STATE).

**Tests:** 2 new sequencer regression tests (on-line-far-from-gate stays
RACING; genuine lateral excursion still triggers RECOVERY); regression gate
**316 passed**. Sequencer suite 69 passed.

**Biggest win so far:** a telemetry-surfaced bug that the benchmark never
caught took the drone from 0 → 6/6 gates on the VQ1 course.

---

## Iteration 7 — flight-envelope sweep + closed-loop regression test

**Telemetry read (max_speed sweep through the harness):**

| max_speed | result | gates | replans | note |
|-----------|--------|-------|---------|------|
| 6 / 8 / 10 | race_complete | 6/6 | 0 | clean, faster with speed (20→12.8 s) |
| 12 | **crash** | 0/6 | 2 | clips gate-0 frame (~1.7 m high) |
| 14 / 16 | race_complete | 6/6 | 4 | completes only via replan recovery |

**Finding (non-monotonic):** at 12 m/s the tracker saturates **max pitch
(−0.85) and max thrust (0.95) simultaneously** for ~1 s, so the vertical
thrust component exceeds hover and the drone climbs ~2 m above gate-0's plane
and clips the frame. This is the infeasible-trajectory / control-saturation
issue (audit Blocker 11: trajectory accel peaks ~108 m/s² ≈ 11 g) surfacing at
high speed. **The competition default (8 m/s) is well inside the safe
envelope.** Re-tuning the high-speed thrust/tilt coupling (or making the
trajectory feasible) is a candidate future iteration; lower priority than the
default working.

**Improvement:** `tests/test_closed_loop_flight.py` — a capstone end-to-end
regression that runs the real control stack vs point-mass physics at the
default speed and asserts: race completes, 6/6 gates, net progress > 150 m,
path/net < 1.5 (no spin/circling), ref populated > 99 %, 0 replans on the
clean course. A regression that reintroduces spinning / off-track / divergence
now fails CI even while the unit suites stay green — closing the original
"unit tests pass but the drone spins" gap at the integration level.

**Housekeeping:** harness/sweep telemetry artifacts gitignored (two were
accidentally tracked in the iter 5-6 commit; removed).

**Tests:** 4 new closed-loop cases pass (7.5 s).

---

## Iteration 8 — recovery-stack resilience test

**Considered EKF probing** (live path uses it) but the VQ1 case is
odometry-rich, so the EKF mostly fuses clean position/velocity and uses
telemetry yaw directly (the audit's yaw-unobservable blocker bites only in
vision dead-reckoning); feeding the IMU with the wrong specific-force
convention risked false conclusions. Chose the higher-value, lower-risk target
instead: prove the recovery stack fixed in iters 3 & 6 actually recovers.

**Telemetry read (perturbation injection):** added a one-shot velocity-impulse
option to the harness (`perturb=(t, (dvx,dvy,dvz))`). A **+15 m/s lateral gust
at t=5 s** → the replanner fires **1 replan** and the drone **re-converges and
completes 6/6 gates** (race_complete, 15.8 s). The disturbance is detected and
handled gracefully (sustained-lateral-error replan, below the harsher RECOVERY
threshold). Larger impulses escalate to RECOVERY.

**Caveat:** the harness runs replans synchronously (no sim time elapses during
the ~1.8 s optimize), so it does NOT model the "blind during replan" window
(audit Blocker 9) — recovery success here validates *detection + re-convergence*,
not the real-time blind gap. Blocker 9 (async replan) remains a future target.

**Improvement:** `test_recovers_from_midcourse_disturbance` — asserts a mid-
course gust still finishes 6/6 gates AND was actually noticed (replan or
RECOVERY). Guards iters 3+6 working together.

**Tests:** closed-loop suite now 5 cases (20.7 s); related suites 116 passed.

---

## Iteration 9 — quantify Blocker 9 (synchronous replan blinds the loop)

**Improvement:** added a `replan_blind_s` option to the harness that models
the live pipeline's synchronous rebuild — every tick a replan fires is followed
by that many seconds of "blind" flight holding the last command (the drone
coasts) while the sequencer still observes physics (so a mid-blind crash is
caught). This turns an architectural concern into a measurable one.

**Telemetry read (gust at t=5 s, 8 m/s, with vs without the blind window):**

| blind window | gates | replans | blind ticks | outcome |
|--------------|-------|---------|-------------|---------|
| 0.0 s (instantaneous replan — prior assumption) | **6/6** | 1 | 0 | race_complete |
| 1.8 s (realistic synchronous optimize) | **2/6** | 20 | **3425 (~76 % of flight)** | diverged to z=−840 m, timed out |

**Finding:** the synchronous replan blinds the drone ~1.8 s → it drifts
off-line during the blind gap → triggers another replan → blinds again: a
**recovery death-spiral**. The instantaneous-replan assumption (iter 8's test)
completely masked it. Blocker 9 is severe *during recovery* — though the
nominal clean course (8 m/s, 0 replans) never triggers it.

**Gap / next:** fix Blocker 9 by computing the rebuild **off** the control
thread (keep flying the still-valid old trajectory, atomic-swap when ready) so
the control loop never goes blind. Deferred to iteration 10 to implement
carefully (concurrency) rather than rush it.

**Tests:** harness refactor preserves the 5 closed-loop cases (default
`replan_blind_s=0`).

---

## Iteration 10 — async replan fix (Blocker 9)

**Fix:** `RacePipeline` now rebuilds the trajectory on a **background thread**
(`async_replan=True`, default) while the control loop keeps tracking the
still-valid current trajectory; the new trajectory is swapped in atomically on
the control thread when the worker finishes (`_apply_pending_rebuild`). The
~1.8 s optimisation no longer blocks the 100 Hz callback, so the loop never
goes blind. `_build_trajectory_from` was split into a pure `_compute_trajectory`
(thread-safe; touches no shared state) + a thin assigner; the legacy
synchronous path is kept under `async_replan=False`. Concurrency is GIL-safe:
the worker writes only the guarded `_rebuild_result` handoff, and the swap is a
single `self.trajectory =` assignment.

**Telemetry read (gust at t=5 s, recovery — sync vs async):**

| pipeline | gates | replans | outcome |
|----------|-------|---------|---------|
| sync, blind=1.8 s (Blocker 9) | 2/6 | 20 | diverged, timed out (death-spiral) |
| **async, rebuild=1.8 s sim** | **6/6** | **2** | **race_complete** |

Flying the old trajectory during the rebuild (vs going blind) turns
catastrophic recovery into clean recovery.

**Harness:** added `rebuild_sim_s` to model the async rebuild's latency in sim
time (joins the worker after that much sim time so the result lands while the
drone flew the old trajectory) — needed because the real worker runs in wall
time, decoupled from harness sim time.

**Tests:** new `test_async_replan_is_nonblocking_and_swaps_when_ready`
(deterministic: callback returns promptly while a fake slow rebuild is in
flight, no second rebuild starts, atomic swap on completion). The default
recovery test now exercises the async path. Regression: closed-loop +
replan + sequencer + watchdog 112 passed; core modules 234 passed.

**Gaps:** remaining audit items — Blocker 11 (infeasible trajectory accel
peaks, ~108 m/s²; limits the high-speed envelope), Blocker 12 (PnP
outer/inner-corner scale), and the large-disturbance RECOVERY-state behavior
(reference-override + thrust-cut can still storm at ≥25 m/s gusts).

---

## Iteration 11 — Blocker 11 investigation (negative result) + observability

**Telemetry read (tilt/thrust saturation vs speed):** at the **default 8 m/s**
the tracker is at max tilt **11.9 %** of the time (thrust-sat 4.8 %), rising to
**37 %** at 11 m/s — the trajectory genuinely demands 108–165 m/s² (11–17 g)
accelerations the drone can't deliver. Confirms Blocker 11 bites (and is what
crashes the drone at 12 m/s, iter 7).

**Root cause located:** `_project_accel_peaks` stretches over-budget segments
(min-snap accel ∝ 1/T²) but **skips segments shorter than `min_seg_time`
(0.15 s)** — and the through-gate segments where the peaks live are ~0.11 s, so
they're skipped and never corrected (residual stays 108).

**Attempted fix → REVERTED.** Un-skipping short segments (stretching them) and
adding passes:
- dropped the peak 108 → 61 m/s² (still > 15), but
- **slowed the 8 m/s race 15.3 s → 21.9 s (+44 %)**, and
- **caused out-of-order DQs at 10/12 m/s** — stretching one segment re-solves
  the *global* min-snap spline and distorts the path/gate geometry.

Post-hoc stretching is the wrong lever. The real fixes are deeper and riskier:
(a) the upstream L-BFGS time-allocation penalty that ignores direction-change
accel (audit MAJOR P-2), or (b) wiring the **discarded** curvature-aware
SpeedProfiler output into `optimize()` (it currently computes per-waypoint
speeds, logs them, and throws them away — audit MAJOR). Both need careful work
beyond a safe autonomous change; deferred with this analysis recorded.

**Safe positive delivered:** the infeasibility signal (`residual peak …`) was a
bare `print()` to stdout; converted it to a proper `logger.warning` with
actionable guidance, so infeasible trajectories are visible in logs rather than
lost in noise.

**Tests:** trajectory suite 30 passed; revert verified clean (empty diff,
8 m/s back to 15.3 s / 6/6).

---

## Iteration 12 — large-disturbance RECOVERY (async win confirmed; slow-down is load-bearing)

**Telemetry read (large gusts on the current async pipeline):** a **25 m/s**
lateral gust now triggers only **3 replans** (vs **23** on the pre-async build,
iter 8) — async (iter 10) crushed the recovery replan-storm. The residual
failure at 25 m/s is **altitude divergence** (z → −53 m) during RECOVERY, the
same tracker thrust-saturation coupling as the 12 m/s case (Blocker 11 family),
not a storm. Moderate gusts (15 m/s) still recover 6/6.

**Attempted two fixes to the `should_slow_down` thrust cut → BOTH REVERTED.**
The audit flags `thrust*0.7` as sinking the drone toward the floor:
1. Floor it at hover → broke the 15 m/s recovery (6/6 → 2/6, z → −248 m).
2. Leave thrust unchanged → also broke it (6/6 → 2/6).

Root insight: the cut is **load-bearing** for off-track recovery on the
**descending** VQ1 course — during lateral recovery the tracker saturates
thrust high and the ×0.7 tempers the climb; the course legitimately needs
thrust *below* hover to descend, so flooring/removing it diverges the drone
upward. The audit's "sinking" concern is a real but *different* (level-flight
detection-dropout) scenario not reproducible in the current harness. A correct
fix needs the tracker's saturated thrust allocation reworked (Blocker 11
family), not this scale factor.

**Delivered:** a code comment documenting why the cut must stay (so a future
"obvious fix" doesn't reintroduce the regression) + this analysis. No
functional change (logic reverted to original; verified 5/5 closed-loop pass).

**Net:** the recovery stack is robust to realistic disturbances (≤15 m/s →
6/6); extreme 3 g impulses (≥25 m/s) hit the same tracker-saturation ceiling as
the high-speed envelope, all converging on Blocker 11 as the next real lever.

---

## Iteration 13 — exercise the EKF/estimation path (the live-competition path)

**Why:** the live VQ1 pipeline runs state through the EKF (IMU predict +
`LOCAL_POSITION_NED` odometry fusion); the harness had only run EKF-off. The
audit flagged EKF concerns (Blocker 4 observability, covariance collapse), so
broadened coverage to surface any.

**Convention work (read telemetry / verify first):**
- Odometry-only (no IMU predict) **lags** badly — 100 updates tracking a ramp
  to (10,0,−5) estimate only (5,0,−2.5): without velocity propagation the
  filter can't follow motion. So a faithful test **must** include IMU predict.
- Verified the EKF's IMU input is **body-frame specific force**
  `R_world→body·(a_world − g)`: hover → (0,0,−9.81) gives no motion; free-fall
  → (0,0,0) gives z=19.62 m at 2 s (=½·9.81·2²). Added
  `_specific_force_body` to the harness with this verified convention.

**Telemetry read (EKF-on closed loop):**

| scenario | result |
|----------|--------|
| clean 8 m/s | race_complete, **6/6**, 15.5 s (vs 15.3 s EKF-off) |
| 15 m/s gust | race_complete, **6/6**, 4 replans |

The EKF estimate stays accurate enough to fly the whole course (completing 6/6
*through the estimate* means no drift/divergence) and survives a disturbance
together with the recovery stack. The audit's EKF-divergence concerns bite in
*vision dead-reckoning*, not the odometry-rich VQ1 path exercised here.

**Improvement:** harness gained a faithful `use_ekf` mode (IMU specific force +
odometry + sim-time stamps); new `test_completes_with_ekf_enabled` regression.
Closed-loop suite now 6 cases, all pass (26 s).

---

## Iterations 21-30 (2026-06-13, continuous session) — ROOT-CAUSED the instability; drone now flies

**Mandate restated:** code was written for PyBullet before the real sim; be
skeptical of all prior "fixes". Test ONLY on the live AIGP sim (offline/harness
forbidden). Reset the drone every run (stale flight muddies telemetry).

### The P0 minimal controller (new): `control/minimal_controller.py`
Bypasses the ENTIRE min-snap/racing/replan/should_slow_down stack. Pure-pursuit:
v_des = cruise*unit(aim - pos); accel = kv*(v_des - vel); horizontal accel
clamped to g*tan(max_tilt); roll/pitch from NED thrust vector; yaw held at pi.
Wired via `PipelineConfig.minimal_control` + `scripts/aigp_vq1_run.py --minimal`.
Uses RAW telemetry (`use_ekf=False` for minimal) — the EKF diverged to NaN ~1s
in and silently blinded the controller (it fell to a fixed hover and climbed
away). Robust to non-finite telemetry (hover fallback) so a bad tick can't crash
the adapter.

### ROOT CAUSES found (all via the live bench `scripts/aigp_bench.py`, reset/phase):
1. **Inner attitude->rate loop was an unstable ~9Hz limit cycle.** Old gains
   (kp=2.0,kd=2.5,max=1.0) limit-cycled (gyro p95 4.5) and the jitter rectified
   thrust into a runaway CLIMB — this, not the trajectory, was the flight
   failure. Pure ZERO body-rate is perfectly clean (gyro~0). FIX: cut gains to
   **kp=0.5,kd=0.2,max_rate=0.5** (`competition/aigp_mavlink.py`). Two Opus
   reviews + bench converged on "gain limit cycle from the ~2.5x sim rate
   amplification", NOT a sign error on damping.
2. **Real hover throttle ~0.26-0.27** (not 0.20/0.234). max_thrust_n 42->**37**
   in MinimalControllerConfig.
3. **PITCH axis of the inner loop was POSITIVE FEEDBACK.** Commanded pitch -0.5
   drove measured pitch the WRONG way; in flight pitch -0.62 diverged to +1.5
   (inverts, then yaw re-charts pi->0 via euler singularity, then climb runaway).
   FIX: per-axis `_rate_sign = (-1, +1, -1)` — flip PITCH only (roll tracked
   correctly with -1; bench: roll+0.3 -> measured +0.26). This made X/pitch
   stable: yaw rock-steady at pi for 95s.
4. **ROLL extraction direction inverted vs the sim.** Sim does +roll -> +Y at
   yaw=pi (bench), but the standard NED extraction assumes +roll -> -Y, so the
   controller fed the Y drift (slid off +Y at constant speed). FIX: negate roll
   in `minimal_controller` extraction (inner roll loop is fine; this is an outer
   sign).

### RESULT: stable, accurate flight (HUGE milestone)
After 1-4: hover hold is rock-solid (20s, 0 drift). The drone flies STRAIGHT and
STABLE to gate0's centre at cruise 1.5 (e.g. min_v8/min_v10: reached
(-23.1,-0.34,-0.04) ~= gate0 (-23.3,-0.4,-0.03), yaw steady at pi). The core
instability (immediate flip / climb / divergence) is SOLVED.

### Gate passing — partially solved, 2 open issues
- **Aim BEYOND the gate along its NORMAL** (`race_pipeline` minimal branch) so
  it flies THROUGH instead of parking (parking -> frame collisions + false/early
  sequencer pass). 
- **The opening is ~0.86m ABOVE gate.position (NED -z).** Flying at gate.position
  hit the bottom bar (128 collisions, stuck 0.2m short of the plane). Aiming
  `--aim-z -1.0` -> **0 collisions** and the drone crossed gate0's plane cleanly.
  So gate.position is NOT the opening centre vertically — there is a vertical
  offset to characterise (and apply to BOTH the aim AND the sequencer's opening
  check, which still assumes gate.position is the centre -> it credits a miss/DQ
  when the drone crosses at the real opening height).
- **best gate count so far: 2/6** (min_v9, but off-centre; the SIM's own count
  was 0 — our sequencer over-credited). Need clean centre passes that the SIM
  counts.

### CRITICAL RELIABILITY BUG discovered
**The SIM_RESET track-data transfer INTERMITTENTLY returns GARBAGE gate
positions.** min_v13 chased an aim of (-922, 6.85, 576) because
`current_gate.position` was ~(-918, 6.85, 577) instead of (-23.3,-0.4,-0.03) —
the run before (min_v12) had correct gates. So some "divergent" runs are
garbage-gate runs, not control failures. **Next: add a gate-position sanity
guard at configure (bounds-check / re-fetch the track on garbage) so runs are
trustworthy.**

### NEXT STEPS (priority)
1. Gate-data sanity guard + re-fetch (the corruption invalidates runs silently).
2. Characterise the vertical opening offset (sweep --aim-z; find where the SIM
   credits the pass) and apply it to gate.position for aim AND sequencer.
3. Once gate0 passes cleanly + SIM-credited, run the full 6-gate course; then
   raise cruise toward the <10s/gate target.
4. Decide yaw _rate_sign (currently -1, never excited; revisit if yaw drifts).

Files changed this session: `control/minimal_controller.py` (new),
`competition/aigp_mavlink.py` (inner-loop gains + per-axis _rate_sign),
`race_pipeline.py` (PipelineConfig minimal_* fields, minimal control branch,
configure skip-trajectory), `scripts/aigp_vq1_run.py` (--minimal/--cruise-speed/
--max-tilt/--aim-z, reset-and-settle, dump-on-crash, dbg logging),
`scripts/aigp_bench.py` (hover sweep, per-axis rate ID, mask-7 retest, PD sweep).

### Update (same session): 2/6 gates, gate-guard live
- Baked the ~0.85m vertical opening offset into the gate map (runner, after the
  sanity check) so the AIM and the SEQUENCER both use the real opening centre.
  Result (min_v15): **gate 0 AND gate 1 pass cleanly (2/6)**, drone tracks the
  gate line exactly (-23.4, -46.9).
- Gate-map sanity guard (`_gate_map_is_sane` + reset/re-fetch) is working — it
  caught two corrupt maps this session (gates at ~-1400 and ~-2680) and
  recovered to the correct course.
- REMAINING blocker to 6/6: the pure-pursuit DESCENT LAGS the horizontal on the
  steeper descending legs, so the drone arrives ~1 m too high at gate 2
  (z=11.7 vs opening ~12.8) and oscillates without passing. Next: decouple the
  vertical channel in `minimal_controller` (track a desired vertical velocity to
  close the z-gap independent of horizontal cruise), or slow horizontal cruise
  when the altitude error is large. Then full 6/6, then raise cruise toward the
  <10s/gate target.

### *** 6/6 GATES — full course complete, 0 collisions (min_v17) ***
Decoupled the vertical channel in `minimal_controller.desired_velocity`
(horizontal pure-pursuit at cruise + INDEPENDENT vertical-velocity tracking of
the altitude error, `vert_gain=1.0`, `max_vert_speed=2.0`) — this fixed the
descent lag. At cruise 1.5 -> 5/6 (ran out of time at gate5). At **cruise 2.5
-> 6/6 in 66.9s, 0 collisions**, "All gates passed! Race complete." Gates at
t=9.8/19.4/30.8/46.0/55.9/65.6s (~10-15s/gate). The whole-stack fix chain:
inner-loop gains (kill 9Hz limit cycle) + pitch _rate_sign +1 (was positive
feedback) + roll extraction negate + EKF bypass (raw telem) + hover cal 37N +
fly-through aim along gate normal + vertical opening offset baked into the gate
map (aim+sequencer) + gate-map sanity guard + decoupled vertical descent.

CAVEAT TO VERIFY NEXT: `competition.session` printed the SIM's race_status as
"0 gates" while our geometric sequencer credits 6/6 with 0 collisions and exact
gate-line tracking. Confirm the SIM officially credits these passes (the drone
clearly flew through every opening) — check race_status / active_gate_index
parsing; if the sim wants a different trigger, align to it.

NEXT: (1) verify sim-credited passes; (2) raise cruise toward <10 s/gate
(currently ~11s/gate avg) — the controller is stable, so push speed and re-tune
max_tilt/gains as needed; (3) tighten centering (still ~80 collisions at cruise
1.5, 0 at 2.5 — interesting; characterise).

### *** SPEED SWEEP — 6.45 s/gate clean (cruise 4.5), sim-credited ***
Optimizations (2 Opus reviews): through_dist 4->2 (halve geometric crossing
offset), inner-loop max_rate 0.5->0.8 (attitude bandwidth, no limit cycle),
max_vert_speed 2->3, and — key — make the SIM's race_finished the SOLE
completion authority (stop on it, not our geometric is_complete which can lead
the sim and cut the run before the drone is through the last gate; keep flying
THROUGH the last gate until the sim credits it).

SIM-credited (race_finished=True) results, full 6-gate course:
| cruise m/s | total s | s/gate | collisions | gyro p95/max | max gate offset |
|---|---|---|---|---|---|
| 3.0 | 55.3 | 9.2 | 0 | 0.14/0.72 | 0.31 |
| 3.5 | 48.9 | 8.15 | 0 | 0.21/0.82 | 0.25 |
| 4.0 | 42.4 | 7.07 | 0 | 0.27/0.85 | 0.27 |
| 4.5 | 38.7 | 6.45 | 0 | 0.37/0.88 | 0.48 |
| 5.0 | 35.5 | 5.9  | 2 | 0.46/16.2 | 0.48 |

**Clean limit = cruise 4.5 (6.45 s/gate, 0 collisions).** At 5.0 the centering at
the later/faster descending gates degrades past the frame margin -> 2 collisions
+ a gyro spike (16 rad/s, a frame-clip kick). The crossing offset is mostly
cross-track (Y); vertical is nailed by the decoupled descent. NEXT lever to go
faster cleanly: a cross-track / centerline-tracking term (reviewer B fix 2) to
tighten the offset, and/or inner-loop kp 0.5->0.7 (reviewer A) for crisper
tracking. Recommended stable race config TODAY: `--minimal --cruise-speed 4.5
--max-tilt 0.62 --aim-z -0.85`.

### *** PER-AXIS ROLL GAIN -> 4.6 s/gate (cruise 6.5), sim-credited, 0 collisions ***
Two Opus reviews both isolated the ROLL axis as the high-speed limiter: the sim
amplifies pitch/yaw ~2.1x but roll only ~1.0x, so a uniform inner-loop kp=0.5
left roll at HALF the closed-loop bandwidth of pitch (roll under-tracked 0.46x
amplitude, ~0.6s lag) -> the cross-track centering oscillated and clipped frames
at cruise >=5. FIX: PER-AXIS inner-loop gains in `_attitude_error_body_rates`
(now accepts scalar or 3-tuple kp/kd) -> roll kp 0.5->1.0, kd 0.2->0.4 (effective
gain ~1.0, matching pitch's proven-safe ~1.05); pitch/yaw unchanged. This killed
the 16 rad/s frame-clip spike and roughly HALVED the high-speed gate offsets.

Full SIM-credited (race_finished=True) speed sweep, 0 collisions unless noted:
| cruise | s/gate | gyro p95/max | max gate offset | note |
|---|---|---|---|---|
| 4.5 | 6.45 | 0.37/0.88 | 0.48 | pre roll-fix |
| 5.0 | 5.86 | 0.45/1.52 | 0.27 | roll-fix (was 2 collisions pre-fix) |
| 5.5 | 5.37 | 0.50/1.60 | 0.35 | |
| 6.0 | 5.00 | 0.53/1.68 | 0.44 | RECOMMENDED (margin) |
| 6.5 | 4.60 | 0.57/1.86 | 0.48 | clean edge (gyro near 2.0 abort) |

**RECOMMENDED RACE CONFIG: `--minimal --cruise-speed 6.0 --max-tilt 0.62
--aim-z -0.85` = 5.0 s/gate (30s course), 0 collisions, comfortable margins.**
Aggressive: cruise 6.5 = 4.6 s/gate (at the centering/gyro edge).

NEXT LEVERS to push past 6.5 cleanly (diminishing returns): (1) reviewer A's
cross-track / centerline term in `minimal_controller.desired_velocity`
(along-track along the gate normal + a capped perpendicular pull, kc~1.0,
max_cross_speed~1.5; needs the gate normal threaded into compute) to tighten the
gate-4/5 offset; (2) further inner-loop headroom if gyro p95 becomes the limit.
We went 0/6 (flipping) -> 6/6 sim-credited @ 4.6 s/gate, 0 collisions this session.

### *** CRUISE SWEEP (roadmap #1+#2) -> 4.09 s/gate, sim-credited ***
A 5-dimension workflow (each finding adversarially verified) deflated the
speculative levers and ranked the cruise sweep as the proven #1. Executed it:
| cruise | course s | s/gate | gyro MAX | worst gate offset (margin) | sim-credited |
|---|---|---|---|---|---|
| 6.5 | 27.8 | 4.6  | 1.86 | 0.48 (0.27) | yes |
| 6.8 | 26.7 | 4.44 | 2.02 | 0.50 (0.25) | yes |
| 7.0 | 26.0 | 4.33 | 1.56 | 0.50 (0.25) | yes |
| 7.5 | 24.6 | 4.09 | 1.67 | 0.59 (0.16) | yes |
(gyro MAX is stochastic gate-transition transients, not monotonic in speed.)

**RECOMMENDED RACE CONFIG: `--minimal --cruise-speed 7.0 --max-tilt 0.62
--aim-z -0.85` = ~26.0 s (4.33 s/gate), comfortable centering margin (0.25m).**
Aggressive: cruise 7.5 = 24.6 s (4.09 s/gate) but gate-2 margin only 0.16m
(repeatability risk). The BINDING CONSTRAINT is now CENTERING (cross-track
UNDERSHOOT/lag, not oscillation — verified): worst gate offset grows
0.48->0.59 over cruise 6.5->7.5 vs the 0.75m half-opening. NEXT LEVER (roadmap
#4) to push past 7.5: a cross-track centerline term (or anticipatory aim toward
the next gate's Y) in minimal_controller.desired_velocity to cut the undershoot
lag — A/B at the SAME cruise first; watch lateral-accel-clamp fraction + gyro
MAX. NOT worth: variable speed profile / kv (startup is accel-saturation
limited, 75/101 startup frames pinned on the 7.0 m/s2 clamp). Journey: 0/6
flipping -> 6/6 sim-credited @ 4.09 s/gate, 0 collisions.

### iter-34: cross-track centering term (roadmap #4) — TESTED, NEGATIVE RESULT
Implemented the verified roadmap's last lever: a decoupled horizontal law
(X=along-track cruise, Y=capped high-gain convergence) behind a `cross_gain`
flag (default 0 = pure pursuit). A/B at the SAME cruise 7.0 (vs pure-pursuit
baseline: 0 collisions, gyro MAX 1.56, worst offset 0.50):
- cross_gain 1.5 -> 3 collisions, gyro MAX 2.22 (over abort line), lateral-accel
  clamp 22% (was ~3%), gate1 OVERSHOT to 0.70.
- cross_gain 0.5 -> 14 collisions, gyro spike 38 rad/s, offsets 0.86/0.73.
Root cause: decoupling loses pure pursuit's natural cross-track DECELERATION, so
the Y converges then OVERSHOOTS (bandwidth without damping) and clips frames —
exactly what the workflow's adversarial verifier predicted. Left OFF by default
(cross_gain=0); pure pursuit is the clean law. **The cross-track lever is
exhausted.**

CONCLUSION: we are at the practical optimum for this controller architecture.
Fastest clean = cruise 7.0 (26.0s / 4.33 s/gate) reliable, 7.5 (24.6s / 4.09)
aggressive. Going meaningfully faster would need a fundamentally different
approach (a smooth trajectory / racing line with DAMPED cross-track tracking),
which reintroduces the min-snap complexity we stripped out — diminishing returns
vs the (crushed) <10 s/gate goal. NEXT highest-value work is REPEATABILITY
validation (all results are single runs; gyro MAX and offsets vary run-to-run),
not chasing more tenths.

### iter-35: re-enable the TRAJECTORY STACK (racing-line + GeometricTracker) — implemented + reviewed, LIVE A/B PENDING (sim out of VQ mode)
User direction: stop iterating the pure-pursuit optimum; test the principled
alternative — a smooth racing-line trajectory flown with DAMPED feedforward
tracking (the thing the failed hand-rolled cross_gain term couldn't do) — and
A/B it live vs minimal cruise-7.0.

IMPLEMENTED a clean `--trajectory` mode (race_pipeline.PipelineConfig.trajectory_race):
flies the precomputed min-snap trajectory with control/mpc_tracker.GeometricTracker
on RAW telemetry, BYPASSING replan/state-predictor/should_slow_down (apples-to-
apples: only the controller differs from minimal). The tracker routes through the
SAME fixed body-rate inner loop (send_attitude mask 128). Key correctness work —
the GeometricTracker's attitude extraction is BYTE-IDENTICAL to the minimal
controller's EXCEPT minimal adds `roll=-roll` (live-sim convention) and holds
yaw=pi; so the tracker now has `sim_roll_sign=-1` + the trajectory path pins
ref.yaw=pi. Tracker also given the calibrated 37 N thrust (not 42) + tilt clamp
0.62. Optimizer constrained to the REAL envelope.

TWO Opus-4.8 reviewers (correctness + tuning), fixes applied:
- (correctness BLOCKER, fixed) GeometricTracker had NO non-finite guard; on raw
  telem a single NaN would crash the run ("thrust must be finite"). Added input+
  output hover guards to the trajectory_race branch (mirrors minimal).
- (correctness RISK, fixed) tracker clamps tilt ANGLE not lateral ACCEL, so the
  ~17 m/s2 min-snap kink peaks would use the unclamped thrust magnitude ->
  realized-accel overshoot + a spurious CLIMB transient at every kink. Added an
  optional `TrackerConfig.max_lateral_accel` (clamps accel_des[:2] BEFORE thrust
  extraction, like minimal); set to g*tan(0.62)=7.0 for the live sim. Now
  "plan-smooth/clamp-safe" actually caps at the envelope.
- (tuning) 7.5 m/s2 PLANNING budget made _project_accel_peaks over-stretch every
  through-gate segment -> 35.7s (slower than minimal's 26s). NOT inflation
  (helix/proximity/climb don't fire on this descending slalom; S-turn only
  3-4%). Raised planning budget to 15 -> 27.6s, kink peaks clamped at tracker.
- (tuning) Raising entry_exit_offset is WRONG here (min-snap corner-cuts the
  entry/exit chord -> reference leaves gate center: 0.4m->17cm, 2.5m->112cm).
  Kept ee=0.4. The real centering lever was RacingLineOptimizer's
  max_lateral_offset=0.6 (=0.45m fixed offset on every 1.5m gate); dropped to
  0.15 (~0.11m) -> worst-gate reference offset 45cm->13cm at ~no time cost.

VALIDATED OFFLINE (dry-run): trajectory 27.6s / 2759 pts, residual interior peak
22.5 m/s2 (clamped to ~7 at the tracker). Expected live: ~13cm gate centering vs
minimal's 0.48-0.59m undershoot — the hypothesis. Run-1 config:
  --trajectory --max-speed 10 --aim-z -0.85   (A/B vs --minimal --cruise-speed 7.0)

BLOCKER: the live AIGP sim is out of Virtual Qualifier mode — MAVLink heartbeat
+ telemetry flow on 14550 but SIM_RESET returns NO track map across retries
(connect() fails at "AIGP track data not received after SIM_RESET"). DCGame is
responding, memory normal (~249MB), no zombie python / port holder — classic
wedged post-race state. Cannot re-enter VQ mode over MAVLink; needs the sim
restarted into Virtual Qualifier in the GUI. Implementation is ready; run the
live A/B the moment VQ mode is restored.

---

## Iteration 36 (2026-06-15) — trajectory-mode FALSIFIED live; kv lever FALSIFIED live; binding constraint re-diagnosed

Fresh 75-iteration budget (user-granted). Sim restarted into VQ mode. All runs
below are LIVE, each with the runner's built-in fresh SIM_RESET + settle.

**Baseline re-confirmed (minimal cruise 7.0, --max-tilt 0.62 --aim-z -0.85):**
6/6 SIM-credited (race_finished=True), 0 collisions, 26.26 s (4.38 s/gate),
worst plane-cross lat(Y) 0.47 m (gate4). gyro p95 0.65 / max 2.04. Controller
healthy on the restarted sim. (Our geometric sequencer credits 5–6/6 run to run;
the SIM is authoritative.)

**Trajectory-race mode (iter-35's pending A/B) — FALSIFIED.** `--trajectory
--max-speed 10 --aim-z -0.85`: 1/6 gates, **2 collisions**, stopped 10.35 s. The
min-snap reference is INFEASIBLE (residual peak 27.5 m/s² vs the drone's ~7 m/s²
real lateral limit), so the damped GeometricTracker clamps + lags badly (overall
cross-track avg 0.99 m) and clips the gate-1 frame. The iter-35 hypothesis
(racing-line → 13 cm centering) is dead: the offline 13 cm reference offset is
meaningless when the drone physically can't track the reference. Racing-line /
min-snap is a dead end for this drone+course; pure-pursuit is the law.

**kv (velocity-tracking gain) lever — FALSIFIED.** A 4-agent workflow (control /
adversarial / alternatives + synthesis, all grounded in the capture) recommended
sweeping kv 3.0→4.5. Live sweep (cruise 7.0, fresh resets):

| kv | worst lat(Y) | gate2 | gate4 | clamp% | gyro max |
|----|------|------|------|------|------|
| 3.0 | 0.47 | -0.34 | -0.47 | 3.5 | 2.04 |
| 3.5 | 0.54 | -0.30 | -0.54 | 8.9 | 1.51 |
| 4.0 | 0.56 | -0.29 | -0.53 | 11.7 | 1.51 |

kv IMPROVES the non-saturated gate (gate2 monotonically better) but WORSENS the
binding/saturated gates (gate4/5); worst-case lat(Y) rises monotonically and
clamp engagement climbs. **Root cause (live-confirmed): the worst gates are
tilt-SATURATED at the gate-flip (cmd_roll pinned 94–100% even at kv=3.0) and
suffer a ~2× inner-loop roll attenuation (measured_roll/cmd_roll≈0.45–0.57), so
they are lateral-AUTHORITY-limited, not kv-limited.** kv only feeds the
discontinuous v_des flip step → more saturation. kv default stays 3.0.

**Also falsified:** anticipatory aim toward the next gate (HARMFUL on this slalom
— next gate is always the opposite Y side → corner-cutting); raising
max_lateral_accel alone (the clamp is 0% engaged in the steady approach window).

**Delivered:** `--kv` CLI knob (default 3.0); coupled `max_lateral_accel =
g·tan(max_tilt)` in the minimal config so `--max-tilt` is no longer a no-op
(it was clamped at 7.0 first → 0.62 rad regardless); new comparator
`scripts/iter36_compare.py` (per-gate plane-cross lat/vert decomposition, gyro,
clamp, per-flip windows) — and per user request, **FRAME CLEARANCE**
(0.75 − max(|lat|,|vert|)) = the "how close to crashing" margin.

---

## Iteration 37 (2026-06-15) — VERTICAL channel unlocks speed; cruise 8.0 reliable @ 3.85 s/gate

With centering levers exhausted, pushed CRUISE directly (sim collisions + frame
clearance = ground truth). The frame-clearance metric immediately re-diagnosed
the binding constraint at speed: it is the **VERTICAL channel**, not lateral.
(Course is DESCENDING — NED z increases gate0→gate5 — so at speed the drone
arrives ~0.6–0.7 m ABOVE each opening, skimming the TOP bar.)

| cruise | mvs | s/gate | collisions | worst frame clearance | gyro max | verdict |
|----|----|------|------|------|------|------|
| 7.0 | 3.0 | 4.33 | 0 | 0.28 m | 2.04 | prior baseline |
| 8.0 | 3.0 | 3.87 | 0 | **0.11 m** (gate2 vert) | 2.12 | vertical lag → unsafe |
| **8.0** | **5.0** | **3.85** | **0 (3/3)** | **0.42 m** | 1.74 | ✅ RELIABLE — new config |
| 8.5 | 5.5 | 3.66 | 0 (2/2) | 0.35 m | 2.03 | aggressive (3rd run blocked by wedge) |
| 9.0 | 6.0 | 3.50 | **0 then 84** | 0.23 m | 2.19 | ❌ collision COIN-FLIP |
| 10.0 | 8.0 | 3.20 | 0 | **0.006 m** | 2.18 | ❌ razor-thin |
| 10.0 | 8.0 + **vert_gain 2.0** | — | **128** | diverged | **8.45** | ❌ vert_gain destabilises |

**Key findings (all live):**
1. **The vertical CAP (`max_vert_speed`) is the speed lever, not lateral.** At
   cruise 8.0 the descent lagged (gate2 clearance 0.11 m); raising mvs 3.0→5.0
   restored 0.42 m and made cruise 8.0 reliably 0-collision (3/3). Default mvs
   bumped 3.0→**5.0** (`control/minimal_controller.py`).
2. **Only the vertical CAP is safe to raise — NOT the vertical GAIN.** vert_gain
   2.0 → gyro 8.45 limit cycle, 128 collisions, divergence (aggressive vertical
   accel swings the thrust vector → roll extraction atan2(zy_h,−z_b[2]) blows up
   as −z_b[2] shrinks). Keep vert_gain=1.0.
3. **Collisions are decoupled from my plane-cross clearance below ~0.25 m.**
   Cruise 9.0 logged 0 collisions on run 1 and **84 on run 2 with near-identical
   trajectories** (both min-clearance ~0.23 m, zero ticks <0.15 m). At ~0.23 m
   center-clearance the drone's finite body + cm-level sim non-determinism flip
   between clean and scrape. **Reliable 0-collision needs worst frame clearance
   ≳0.35–0.4 m** → cruise 8.0 (0.42 m) is the safe sweet spot; 8.5 (0.35 m) is
   borderline-aggressive (2/2 clean but needs a 3rd confirm); ≥9.0 is unsafe.

**NEW RECOMMENDED RACE CONFIG (reliable, 3/3 clean):**
`--minimal --cruise-speed 8.0 --max-tilt 0.70 --aim-z -0.85 --max-vert-speed 5.0`
= **3.85 s/gate** (~23.1 s course), 0 collisions, 0.42 m worst clearance, gyro
p95 0.74 — a reliable ~11 % speedup over the old cruise-7.0 (4.33 s/gate).
Aggressive (verify a 3rd run): cruise 8.5 = 3.66 s/gate, 0.35 m.

**NEXT LEVERS to go faster than 8.0–8.5 cleanly** (the binding constraint is now
vertical descent lag + the ~2× inner-loop roll attenuation):
1. Improve vertical descent tracking WITHOUT raising vert_gain — e.g. a vertical
   FEEDFORWARD (descend along the known leg slope) so the cap-limited lag shrinks
   without the destabilising high-gain accel swing.
2. Recover the ~2× inner-loop roll attenuation (raise the attitude→body-rate gain
   in `aigp_mavlink.py` — DOCUMENTED limit-cycle hazard; do it on the bench).
3. Slew-limit / low-pass the discontinuous v_des gate-flip step (~0.15–0.2 s) to
   cut the flip transient that pins cmd_roll at the clamp.

**SIM WEDGED at end of iter-37:** after the cruise-10/vert_gain-2 divergence run
(flew far past the finish to x≈−50), DCGame memory ballooned 490→1891 MB and
SIM_RESET stopped returning a track map ("AIGP track data not received") — the
documented post-race wedge. Needs a GUI restart into Virtual Qualifier to
continue. Regression gate green (302 passed; fixed the stale per-axis
`_rate_sign` test in `competition/tests/test_aigp_mavlink.py`).
