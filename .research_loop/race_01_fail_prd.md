# PRD: Fix `race_01` failure — drone misses gates, overshoots, crashes

## Problem

Running the visual demo against `sim_pybullet/configs/race_01.json` fails
consistently: the drone completes only **4 / 12 gates**, crashes around gate-5
after altitude collapses to `z < 0.1 m`, and exhibits the catastrophic behaviour
of **overshooting gate-1 by a wide margin and then looping back to pass it from
the wrong side**.

Reproducing command:
```bash
cd ~/Personal/killallhumans
python3 scripts/visual_demo.py --config sim_pybullet/configs/race_01.json --no-render
```

Latest failure run transcript (PRD input):
```
PASSED gate-1 [1/12] t=20.79s
PASSED gate-2 [2/12] t=24.67s
PASSED gate-3 [3/12] t=29.73s
PASSED gate-4 [4/12] t=32.42s
Crash! Alt=0.04m
Results: gates_passed: 4, total_gates: 12, sim_time: 36.938, complete: False
```

Note that the **planner prints "Trajectory: 14.2s, 700 points"**, yet gate-1 is
only reached at t=20.79s — the drone has already flown past all the early gates
on the original trajectory pass and had to loop back via the `gate_fallback`
branch in `scripts/visual_demo.py`.

## Telemetry Evidence (from `logs/visual_demo_*.csv`)

Analysis of the 2026-04-16 17:56 CSV run (a newer, similar-failure variant) shows:

| sim_time | pos (x, y, z) | gates_passed | target_source |
|----------|--------------|--------------|---------------|
| 0.00 s   | (0.00, 0.00, 1.50) | 0 | trajectory (cmd vel 5 m/s) |
| 1.85 s   | (8.13, 0.18, 1.61) | 1 | trajectory |
| 12.50 s  | (54.14, 0.54, 2.83) | 1 | trajectory  ← drone at gate-6 region |
| 16.67 s  | (46.07, 1.90, 2.21) | 1 | gate_fallback ← looping back |
| 19.21 s  | (17.91, 3.95, 2.20) | 2 | gate_fallback ← finally at gate-2 |
| 23.31 s  | (28.07, -2.01, 1.49) | 3 | gate_fallback |
| 27.15 s  | (37.89, 1.37, 2.19) | 4 | gate_fallback |
| 30.83 s  | (39.88, -2.45, 0.05) | 4 | gate_fallback ← crash |

### Observations
1. **Initial state/trajectory mismatch**: at t=0 drone is at rest (v=0), but the
   first reference is at (1.98, 0.11, 1.55) with `v=5.0 m/s` — the trajectory
   assumes the drone is already in motion, causing immediate large tracking
   error.
2. **Drone overshoots far past every gate except gate-1** during the original
   trajectory — by t=12.5 s it is near gate-6 (x=54) but sequencer still
   reports `gates_passed=1`. The geometric gate-pass-through check in
   `gate_sequencing/sequencer.py::_point_in_gate_opening` rejects crossings
   where the drone plane-crosses outside the interior w/h.
3. **Gate-2 through Gate-6 are missed entirely** on the first pass because the
   planned trajectory does not actually fly through the gate openings (likely
   lateral offset > 0.6 m half-width).
4. **Fallback mode thrashes**: after `sim_time > trajectory.total_time` the
   fallback in `scripts/visual_demo.py` drives the drone straight back toward
   the first un-passed gate. This creates huge overshoots and sharp reversals
   because there is no trajectory replanning.
5. **Crash at gate-5**: the drone ends up at z=0.05 while chasing gate-5 via
   fallback. The fallback commands `target_vel = direction / dist * min(dist*2,
   5.0)`; at short dist this gives a small thrust command while the drone is
   falling, so altitude collapses.

## Action items

1. **Instrument telemetry** (already largely in place via `logs/visual_demo_*.csv`):
   * Ensure the CSV captures `distance_to_target_gate_center_m` (3D) and the
     `pass_through_offset_m` (signed lateral + vertical offset at the moment
     each gate is passed). Add these columns if missing.
   * Log per-gate **pass-through distance from gate center** (single summary
     line per pass-through event).
2. **Root-cause + fix the trajectory** so the **drone actually flies through
   each gate opening** on the first pass. Candidate areas:
   * `planning/racing_line.py` — offset optimization may have drifted outside
     the gate interior after the cached `racing_line_cache.json` settings.
   * `planning/trajectory_optimizer.py` — time allocation / velocity clamp may
     be producing a trajectory whose actual path is outside gate openings
     (interior 1.2 m × 1.2 m).
   * Initial condition: trajectory must start at `v=0` if the drone starts at
     rest; otherwise the tracker is behind for the first ~1 s and diverges.
3. **Remove the catastrophic fallback**: replace the "navigate straight to the
   next gate after trajectory ends" heuristic with one that never flies
   backward through already-passed gates (e.g., re-plan from current state or
   hold position).
4. **Gate-pass detection hardening**: if the drone legitimately passes within
   a small lateral/vertical margin of a gate but the plane-crossing test is
   rejecting it due to geometric edge cases, extend the sequencer with a
   proximity-pass rule (currently `proximity_pass_distance: 0.0` means
   disabled). Only enable if it is the actual root cause — the primary fix is
   to get the trajectory through the gate centres.
5. **Cross-validate every change with Codex red-team** via the ralph-loop
   red-team stage (configured in runbook). Never merge an iteration that
   regresses `gates_passed` below the previous iteration's best.

## Tools to use

* **`giga_chad_llm` Ralph loop** (`mcp__giga_chad_llm__giga_chad_llm_ralph_*`)
  with **≥ 25 iterations** (`max_iters: 25`).
* Each iteration MUST:
  1. Run `python3 scripts/visual_demo.py --config sim_pybullet/configs/race_01.json --no-render` and capture the JSON/CSV output.
  2. Parse `gates_passed`, `complete`, `sim_time`, `avg_tracking_error`, and
     crash altitude from the CSV if needed.
  3. Propose and apply a **minimal** change (one module, one concern).
  4. Re-run the demo and compare.
  5. Invoke **Codex red-team review** on the diff (`ralph_red_team`), rejecting
     changes with severity ≥ high.
  6. Commit the iteration with a descriptive message.

## Acceptance Criteria

Primary (must-hit): **all met simultaneously in a single run**
* `gates_passed == total_gates == 12`
* `complete == True`
* `crashed == False` (drone altitude stays > 0.3 m the entire run)
* `sim_time <= 60.0 s`
* **First gate-1 pass-through happens in the drone's forward direction and
  within `t <= 4 s`** (not a backward loop-around).
* **No gate is passed via `gate_fallback` target_source** — every gate pass
  must occur while `target_source == "trajectory"`.

Secondary (stretch):
* `sim_time <= 30 s`
* Mean pass-through offset from each gate centre `<= 0.3 m`
* `avg_tracking_error <= 0.5 m`
* `avg_loop_hz >= 200`

## Do NOT

* Modify `sim_pybullet/` physics — treat as ground truth.
* Weaken the gate pass-through geometric test to "pass by proximity" just to
  make the number go up. If proximity is used, it must also require the drone
  to have crossed the gate plane in the forward direction.
* Add new runtime dependencies.
* Claim success without running the demo end-to-end headless and parsing the
  final `Results:` block.

## Signal for completion

Output the promise tag exactly:

```
<promise>RACE01_PASS</promise>
```

only when a full run satisfies all primary acceptance criteria above and the
diff has been red-team-reviewed and committed.
