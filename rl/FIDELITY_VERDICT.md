# RL Replica — Fidelity Verdict (2026-06-16, verified)

**Question (user):** for RL we need the training sim to match the AI Grand Prix
(DCGame) sim — run the *same exact flight commands* through both; if identical,
the telemetry should be very similar. **Can we replicate the drone physics?**

**Answer: NOT with the current model + data.** The replica reproduces DCGame's
*aggregate, closed-loop* champion metrics but **fails the open-loop same-command
test** — the rigorous fidelity check.

## The test that matters (open-loop command replay)

Replayed the exact logged champion command sequence (`cmd_roll/pitch/yaw/thrust`
per tick) open-loop through the replica from the real initial state, no feedback,
and compared to the real DCGame telemetry (`captures/rel_12`):

| t (s) | real (N,E,D) | replica (N,E,D) | err (m) |
|---|---|---|---|
| 2 | (−13.0,−0.2,−1.7) | (−7.4,−0.1,−1.5) | 5.6 |
| 8 | (−68.0, 1.0, 11.9) | (−57.4, 1.0, 17.9) | 12.2 |
| 16 | (−130.2,−1.7, 24.1) | (−118.6,−1.1, 49.4) | 27.9 |
| 19 (end) | (−159.2,−3.3, 24.7) | (−141.6,−1.3, 54.9) | **37.3** |

**37 m full-lap divergence; the replica descends ~2× too fast** (real −24 m,
replica −49 m by t=16 s). They are NOT "very similar" under identical commands.

## Why the agent's "PASS" was misleading

The fidelity gate reported PASS on **closed-loop** champion-in-replica metrics
(lap 16.85 s, descent 2.33 m/s) + **windowed short-horizon** RMS (0.33 m@0.5 s).
But:
- The closed-loop match is the **champion's feedback correcting the replica's
  wrong dynamics** — not the replica being faithful. The descent "wall" does NOT
  emerge open-loop; it only appeared because the champion throttles back when it
  sees (the replica's) altitude.
- Even closed-loop, the champion **breaches gates 0/2/3** in the replica
  (margins −0.13/−0.49/−0.46 m vs real +0.235 m).
- The "held-out rel_9" was effectively **in-sample** — all 15 captures are the
  same champion lap fit together (n=27943), so nothing tested generalization.

## Root cause — it's the DATA, not just the model

- **Champion-only telemetry:** 15 repeats of one lap. No excitation (chirps,
  multi-axis), no **level-hover thrust steps** (so `k_t` — which drives the
  descent — can't be pinned; the agent flagged a launch-phase `k_t` cliff), no
  off-champion / descent-wall-boundary data.
- **Corrupt per-step timestamps** (`t_us` deltas ~half out-of-range/negative) —
  forced a fixed-cadence fit.
- **No IMU-accel channel** in the captures — translation fit from finite-diff
  velocity only.

A gray-box model fit to one closed-loop trajectory cannot recover the open-loop
descent dynamics that aren't excited in that trajectory.

## Decision: RL is NOT viable here — two converging reasons

1. **Floored even with a perfect twin:** the lap is bandwidth-walled at ~16.2 s
   (descent forced through gates 1–3 at ~2 m/s; established over 9 real-sim
   iterations). A faithful replica imports that wall → RL trained on it is
   floored at ~16.2 s. RL beats classical via a *better objective*, not by
   violating physics (Science Robotics 2023) — and the champion is already at
   the wall.
2. **Can't build a faithful twin from champion-only data:** the open-loop test
   above (37 m divergence) proves it. An RL policy trained on this replica would
   exploit its free-descent error and fail to transfer to DCGame's walled descent.

## What it WOULD take (if ever revisited)

A real **excitation campaign on DCGame** (level-hover thrust steps to pin k_t;
roll/pitch chirps at amplitude; steep-descent sweeps up to the tumble boundary;
coupled multi-axis), a fixed-timestamp telemetry fix, a richer dynamics model
(or a learned residual), then Stage-E sim→DCGame residual fine-tuning (Swift
recipe). Multi-week effort — for a result the bandwidth wall says is floored at
~16.2 s. **Not recommended.** The race-ready champion (16.2 s, 15/15 6/6) stands.

## What's preserved

`rl/` (replica + fitter + the fidelity tooling that revealed this) is committed
on `aigp-telemetry-loop` (NOT merged to main — exploratory). The fidelity
validator + this open-loop replay are reusable if richer DCGame data is ever
collected.
