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
