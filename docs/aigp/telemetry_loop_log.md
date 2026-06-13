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
