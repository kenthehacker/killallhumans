# AIGP VQ1 — Speed Ladder + Structural Spline Handoff (2026-06-16)

Covers the autonomous-loop work of **iters 36–56** (continues
`2026-06-13-control-handoff.md`). The system is now **race-ready** with two
validated controllers. This doc is the single place to start.

---

## TL;DR

- **Two clean controllers**, both SIM-credited 6/6, 0 collisions, ~50 km/h peak:
  - **Gate-by-gate** (default `--minimal`) — **fastest lap (18.4 s g0→g5, avg 32 km/h)**, frame margin ~0.32 m.
  - **Spline** (`--minimal --spline`) — 20.6 s lap, **best margin (~0.35 m, most gates 0.44–0.71 m)**.
- **Pick gate-by-gate for race time, spline for reliability.** They are ~2 s apart; the gap is the steep-descent section (see "Hard limit").
- **The speed/precision ceiling is HARD** and well-characterised: the sim's
  inner attitude loop attenuates achieved roll to ~0.53× commanded during fast
  maneuvers. Three inner-loop fixes were tried and **all falsified** — see below.
  Do **not** re-try them.

---

## How to run (live sim, VQ mode active)

**Gate-by-gate (recommended for fastest lap):**
```
python -m scripts.aigp_vq1_run --minimal --cruise-speed 16 --max-tilt 0.82 \
  --aim-z -0.85 --kv 3.0 --max-vert-speed 9.5 --lookahead 0.7 \
  --speed-brake 1.3 --speed-min-frac 0.55 --speed-descent-gain 1.5 \
  --aim-slew 12 --final-aim-z 0.5 --final-brake-band 26 \
  --record captures/run.jsonl.gz --max-seconds 75
```

**Spline (recommended for max margin / reliability):** the winning config is
baked into the spline defaults, so just:
```
python -m scripts.aigp_vq1_run --minimal --spline --cruise-speed 16 \
  --max-tilt 0.82 --aim-z -0.85 --kv 3.0 --max-vert-speed 9.5 \
  --record captures/run.jsonl.gz --max-seconds 75
```
(Defaults baked: lookahead 8, a_lat 6.5, v_min 6, v_descent 2.0, vert_ff 1.0,
v_final 10, final_region 50.)

**Analyse a run** (per-gate frame clearance + crash location, ignores post-crash telemetry):
```
python -m scripts.iter36_compare captures/run.jsonl.gz
```

---

## What changed this session (the journey)

1. **DSQ root cause — FALSE START (iter 39, CRITICAL).** Runs were disqualified
   for commanding flight before the 3 s GO countdown. The old runner used a blind
   ~3.5 s timer; GO jitters 3.3–3.7 s. Fix: `_reset_and_settle` waits for the
   authoritative GO crossing `sim_boot_time_ms >= race_start_boot_time_ms + 120 ms`
   against a FRESH post-reset status. **Never** gate flight on a timer or on
   `RaceStatus.race_started` (that flips at ~0.6 s = "GO scheduled", not "racing").

2. **Speed ladder 28 → 50 km/h.** Variable speed: fly a fast base and BRAKE
   per-leg into geometrically tight / steep-descent gates (`--speed-brake`,
   `--speed-descent-gain`, `--speed-min-frac`, slew-limited). Smoothing: slew the
   lateral aim (`--aim-slew`) to kill the "rock-back" at each gate. Clean 6/6 at
   40 → 45 → 48 → 50.

3. **gate5 TOP-crash fixed (iter 46).** At speed the drone arrived ~0.7–0.9 m
   HIGH at the finish gate — a deceleration *balloon* on the short fast finish
   leg, not a lateral miss. Fix: `--final-aim-z` aims the final gate lower.

4. **Clean 50 km/h (iter 47).** The final reversal (g4→g5) overshot at 50.
   `--final-brake-band` proximity-brakes the closing reversal so it's made at a
   speed the rate-limited roll can handle, while the straight keeps the peak.

5. **Structural racing-line spline (iters 52–54).** `planning/racing_spline.py`:
   one C2 arc-length spline through all gates + a curvature/descent/final-region
   speed profile. Opt-in (`--spline`). Verified insight: the slalom is *gentle*
   as a smooth path (~30 m radius); the gate-by-gate's difficulty was the sharp
   corners, not the geometry. Net: the spline matches peak speed with ~2× the
   frame margin, at the cost of ~2 s lap time (slower descents).

6. **Telemetry tooling.** `iter36_compare` now reports per-gate frame **clearance
   + crash edge** (TOP/BOTTOM/±Y) and **ignores post-crash telemetry** (truncates
   at a tumble or backward shove, so a crash-at-gate1 run no longer reports
   garbage gate2–5 numbers).

---

## The hard limit (read before trying to go faster)

Both the slow steep-descent (~22 km/h through gate2) **and** the 50-km/h
wall trace to ONE root cause: **the sim's inner attitude loop attenuates
achieved roll to ~0.53× commanded during a fast maneuver** (rate-limited
attitude tracking). Demanding a faster descent or a harder turn requires an
attitude change the loop can't track → tumble.

**Everything falsified (do NOT repeat):**
| Attempt | Result |
|---|---|
| Raise roll kp / kd | 9 Hz limit cycle, achieved roll *worse* (0.53→0.33) |
| Raise roll rate-clamp 0.8→1.0 | Neutral (the rate rarely binds) |
| Attitude-error integral (PI-via-INDI, `--att-ki`) | **Destabilises** even the clean champion (3/6 + collisions) |
| Static FF pre-comp (÷0.53) | Doesn't help at the clamp (the hard turns are already clamped) |
| Spline descend-at-speed (vert_ff 0/1/1.2, far-lookahead, gate-Z, waypoint-Z) | All faster-than-22 tumble or lag/diverge |

**Only remaining lever:** full **measured-angular-acceleration INDI** (Tal &
Karaman) — invert the incremental rate→ang-accel using a filtered gyro
derivative + a *bench-identified* control-effectiveness matrix `G`. Needs bench
ID we don't have, is error-prone, and "cannot exceed physics" if 0.53 is a true
actuator-bandwidth limit. **High effort, uncertain payoff — not recommended
unless a faster lap is mission-critical.**

---

## Key code pointers

- `scripts/aigp_vq1_run.py` — the live runner + all CLI knobs. `_reset_and_settle`
  (GO-crossing start), `run_vq1` (params → `PipelineConfig`), recorder callback.
- `race_pipeline.py` — `_control_callback` minimal branch: gate-by-gate aim +
  variable-speed brake + slew + final-gate handling, and the `minimal_spline_path`
  branch. `_gate_difficulty` (per-leg brake). All `minimal_*` config fields.
- `planning/racing_spline.py` — `RacingSpline` (spline + speed profile). Offline
  tests in `tests/test_racing_spline.py` (7 pass).
- `control/minimal_controller.py` — the pure-pursuit law (desired velocity →
  accel → attitude/thrust). `vert_ff` glide-slope, envelope clamps.
- `competition/aigp_mavlink.py` — `_attitude_error_body_rates` (the inner PD
  body-rate loop, gains `(1.0,0.5,0.5)/(0.4,0.2,0.2)`, clamp 0.8, `_rate_sign
  (-1,+1,-1)`), `send_attitude`. **The 0.53 attenuation lives here — see the
  FALSIFIED comments before touching gains.**
- `scripts/iter36_compare.py` — the per-run analysis (clearance, crash edge,
  gyro, post-crash truncation).

---

## Operational notes (race day)

- **Sim degrades after ~25 runs/session** (DCGame process): start over-climb
  grows (healthy ~−1.7 m, degraded ≥−2.4 m), and the gate-map transfer returns
  garbage (sign-flipped X or Z≈−350). The runner's `_gate_map_is_sane` re-fetch
  catches the bad map; a degraded *process* needs a full .exe restart into VQ
  mode (a per-run SIM_RESET does NOT fix it). HEALTH PROBE: a fresh run's
  first-3 s peak climb should be Z≈−1.7.
- **Reset discipline:** the runner does SIM_RESET + settle per run, so each run
  is freshly mapped — but if a run's start-climb is ≥−2.4 m or the collision
  count is wildly high with a clean trajectory, suspect degradation and restart.
- **No DSQ signal over MAVLink** — the verdict is sim-GUI-side only; can't be
  detected in telemetry.
- **Frame is FRD/NED.** Comms: loopback UDP MAVLink2 `udpin:127.0.0.1:14550` +
  JPEG vision UDP 5600 (not wifi/bluetooth — compliant).

---

## Open items / next levers (ranked)

1. **Reliability hardening** (low risk, high race-day value): harden the
   gate-map sanity check against *uniform* shifts (current check catches
   out-of-bounds but not a uniform offset); add automatic sim-degradation
   detection + abort.
2. **Full measured-accel INDI** (high risk/effort, uncertain): the only path to
   a faster lap; needs bench ID of `G`.
3. **Perception / VQ2** (out of scope here): onboard gate detection + EKF fusion
   (AlphaPilot/Swift recipe) — a separate, larger effort.

Full iteration-by-iteration detail: `docs/aigp/telemetry_loop_log.md` and the
`aigp-loop iter NN` commit messages.
