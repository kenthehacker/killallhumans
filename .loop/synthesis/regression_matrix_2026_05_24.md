# Multi-Track Regression Matrix — 2026-05-24 Baseline

Result of running `scripts/benchmark_matrix.py` across all 7 track
configs in `sim_pybullet/configs/` after iter-003d.

## Outcome: only race_01 passes

| track             | gates passed | rate  | complete | crashed | DQ | termination                |
|-------------------|--------------|-------|----------|---------|----|-----------------------------|
| race_01           | 12 / 12      | 100%  | YES      | NO      | NO | race_complete (13.7s)       |
| aigp_default*     | 0 / 6        | 0%    | no       | YES     | no | crash_gate:gate-1 (1.2s)    |
| figure8           | 0 / 8        | 0%    | no       | no      | YES| out_of_order:gate-2 (1.7s)  |
| grand_tour        | 1 / 14       | 7%    | no       | no      | YES| out_of_order:gate-3 (2.7s)  |
| slalom            | 0 / 8        | 0%    | no       | YES     | no | crash_gate:gate-8 (4.8s)    |
| straight_hairpin  | 1 / 6        | 17%   | no       | no      | YES| out_of_order:gate-3 (2.9s)  |
| vertical_cliff    | 0 / 4        | 0%    | no       | no      | YES| out_of_order:gate-2 (2.4s)  |

\* placeholder track (no real DCL data backing).

## Interpretation: this is the user's overfitting concern, surfaced empirically

Five tracks DQ out-of-order in under 3 seconds. The pattern: the
controller's trajectory crosses gate N+1's opening *before* gate N is
credited, and the new strict in-order DQ (iter-001 A5, tightened in
iter-002) terminates the run.

Possible root causes (require iter-004+ investigation):
1. **Trajectory optimizer overshoots gate-1 area** — produces a path
   whose first segment crosses gate-2's plane geometrically. Most
   likely for tightly-packed gates (slalom, figure8, hairpin).
2. **Kinematic drone's initial-acceleration overshoot** — the synthetic
   drone starts at rest and over-corrects on the first segment. With
   race_01's wide opening spacing this is benign; on tighter tracks
   it crosses the next gate's plane.
3. **The new ILC default `(low → high)` partition** — without race_01's
   hand-tuned section schedule, the trajectory's correction profile
   may push the path off-axis. (Reading the deltas vs race_01:
   max_tracking_error 0.66 m on race_01 vs 1.44 m on slalom — the
   tracker IS getting worse.)

## Crashes (not just DQs)
- **slalom**: gate-8 strut hit at 4.77s. The drone reaches halfway-ish
  before clipping a frame.
- **aigp_default**: gate-1 hit at 1.19s. Likely my placeholder
  geometry put gate-1 close enough to the start that the kinematic
  drone's initial overshoot clips the frame.

## Implications for iter-004+
- **Do NOT trust race_01 results in isolation.** Any controller tweak
  that improves race_01 by X% but doesn't improve at least one other
  track is suspect.
- **The honesty bar is working.** The bench correctly distinguishes
  these failure modes (DQ vs crash vs incomplete) and surfaces them.
  Iter-001 and iter-002 fixes are doing their job.
- **Plausible next directions** (don't commit to one without research swarm):
  - Soften the in-order DQ: only DQ if a future-gate crossing is
    inside the lenient pass-through opening AND the drone is NOT in
    a recovery/replanning state. Currently every overshoot is fatal.
  - Trajectory-optimizer corridor constraint: penalise paths that
    cross future-gate planes.
  - Move the controller to MPCC++ contour control (paper backed,
    deferred from iter-001 plan).
  - Multi-track training of the ML tracker residual (iter-001 A15
    landed the model but never trained against multi-track data).

Per the optimization-stall protocol in `.loop/specs/0_charter.md`,
**if the controller doesn't improve in iter-004 via direct tuning,
fan out the 4-model research swarm** (Gemini + Codex + Opus + Composer)
for proven techniques. Don't grind tuning knobs against race_01 only.

## Baseline JSON
Saved at `.loop/state/regression_baseline_2026_05_24.json` for diff
against future iterations.
