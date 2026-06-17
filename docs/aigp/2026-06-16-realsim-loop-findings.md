# AIGP — Real-Sim Validation Loop Findings (2026-06-16)

First end-to-end **real-DCGame-sim** validation of the deep-research work (the
prior report `2026-06-16-deep-research-improvement-report.md` was offline-only).
8 live flights. **Use `python -m scripts.sim_connect_check` + the `aigp-sim-connect`
skill to connect** (the sim was reachable: VQ mode, MAVLink udp 14550).

## TL;DR

- **Performance envelope (live):** gate-by-gate **cruise 16 = clean 6/6, 16.2 s
  g0→g5, 0.27 m worst frame margin** — the practical SPEED CEILING. Spline =
  clean 6/6 ~21.5 s, big margins (0.35–0.69 m), but *curvature-limited*
  (cruise-insensitive — cruise 20 ≈ cruise 16). Both verified by the SIM's own
  `race_finished=True` scoring.
- **The 0.53 roll crux is a TRUE rate/bandwidth wall** — confirmed FOUR ways
  (below). Not recoverable by the inner loop.
- **INDI is shelved** — it diverges on the real sim (a sign/frame bug); off by
  default, not race-usable.
- **The new monitors work live:** gate-map integrity + session-reference drift
  check + sim-health probe all fired correctly. One real tool bug fixed
  (`iter36_compare`).
- **Next real speed lever = a racing-line / speed-profile redesign that reduces
  the *required* roll rate and the vertical balloon** — a substantial planner
  effort (the team's prior focus), uncertain incremental payoff.

## The envelope (8 flights)

| iter | config | result (SIM-authoritative) |
|---|---|---|
| 1 | spline, cruise 16 | clean **6/6**, 21.3 s, worst margin 0.33 m @ gate3 |
| 2 | gate-by-gate, cruise 16 | clean **6/6**, **16.2 s** g0→g5, worst 0.268 m @ gate5 |
| 3 | gate-by-gate, cruise 16, **--indi** | **DIVERGED** (out-of-box, 2.3 s) |
| 4 | --indi, cruise 8 | **DIVERGED** (out-of-box, 1.9 s) |
| 5 | --indi (clamp+gentler gains), cruise 8 | **DIVERGED** (sign bug) |
| 6 | gate-by-gate, cruise 18 | **CRASH @ gate0** (1 collision) |
| 7 | spline, cruise 20 | clean **6/6**, 21.5 s (≈ cruise 16 → curvature-limited) |
| 8 | gate-by-gate, cruise 17 | **5/6**, 1.3 cm margin @ gate1, **tumble @ gate5** |

Scoring note: trust the SIM's `race_status.race_finished` / `active_gate_index`,
NOT our center-based sequencer — the sequencer undercounts the final gate by
~0.16 m (it credits *center*-crossing; the sim credits *body*-crossing). Iter 1
logged "5/6" but the sim scored `race_finished=True` (full 6/6).

## The bandwidth wall — four independent confirmations

1. **PD champion telemetry (iter 2):** the roll-rate command saturates the ±0.8
   rad/s clamp **60–66 %** of the hard-turn ticks (gates 1, 2, 4) while achieved
   roll stays < commanded — the controller is already asking for max roll rate.
2. **INDI online-G (iters 3–5):** converges to a low roll effectiveness (~0.42),
   and a measured-accel inversion does NOT restore the roll.
3. **Cruise 18 (iter 6):** +2 m/s over the clean run → immediate gate-0 crash
   (the harder bank exceeds the rate-limited roll).
4. **Cruise 17 (iter 8):** gates ballooned to **1.3 cm** from the TOP bar and the
   final reversal tumbled — the same balloon/overshoot the bandwidth limit causes.

Matches the handoff's own conclusion: *reduce the required roll via the
trajectory, don't add inner-loop bandwidth.*

## INDI status — diagnosed and shelved (off by default)

The opt-in measured-accel INDI loop **diverges on the real sim** (climbs out of
box in ~2 s). Telemetry root cause:
- **Gyro-derivative spikes:** the telem feed updates slower than the control
  loop → `alpha_meas = dω/dt` spikes to ±89 rad/s² while barely rolling.
  *Fixed* (iter 66): clamp the raw derivative to ±`max_ang_accel` (30) +
  freeze online-G on a clamped axis.
- **Hot startup gains:** module-default `kp_att=18` slams the rate clamp on the
  large startup attitude error → windup. *Fixed* (iter 66): gentler live gains
  `kp_att=(6,6,4)`.
- **RESIDUAL SIGN/FRAME BUG (open):** the sim applies body rates with
  `_rate_sign=(−1,+1,−1)`, so the true roll/yaw control-effectiveness is
  **negative**, but the online-G is clamped positive-only → it can't represent
  it (g floors at 0.05, achieved/commanded roll goes *negative*) → positive
  feedback → divergence. To make INDI fly, the online-G + `g_clip` must be
  per-axis SIGNED (seed `(−1, +2.1, −2.1)` to match `_rate_sign`), or the INDI
  must work in the sim command frame (apply `_rate_sign` internally and not
  re-apply at the caller). **Even fixed, the evidence says INDI won't beat the
  wall** — it would only confirm BANDWIDTH-LIMITED cleanly.

## Monitors validated LIVE (the iter 57–63 work)

- **Gate-map integrity (iter 60):** "Gate map integrity: OK (6 gates)" on every
  run; wrote `captures/gate_map_reference.json` on iter 1; iter 2+ confirmed
  cross-run stability ("matches the reference within tolerance, max residual
  0.021 m"). The uniform-drift detector is live.
- **Sim-health probe (iter 63):** reported healthy peak climb Z≈−1.67…−1.72
  (vs the −2.4 degraded threshold) on clean runs; `insufficient_data` on the
  early-abort runs (correctly non-alarming).
- **`iter36_compare` bug fix (iter 65):** it `KeyError`'d on the iter-63
  `sim_health` capture row (hard `["pos"]` index); now filters to telemetry rows.

## What's left (ranked)

1. **Racing-line / speed-profile redesign** (the only real speed lever): cut the
   *required* roll rate at the binding gates (1, 2, 4) and the vertical balloon at
   the early gates, so the line can carry more speed within the ±0.8 roll-rate
   budget. This is roadmap #3 (bandwidth-aware re-timing) made concrete now that
   the bandwidth (±0.8 rad/s clamp, ~0.42 roll effectiveness) is measured.
   Substantial, multi-iteration, uncertain payoff (the team already optimized the
   line heavily). Validate every change on the live sim.
2. **INDI sign fix** (medium effort): per-axis signed online-G → a clean crux
   verdict. Likely confirms BANDWIDTH-LIMITED; low race payoff.
3. **Reliability stress test:** run toward the ~25-run degradation point to
   confirm the sim-health + gate-map monitors catch it live (they're wired and
   passed the clean-run checks).

## Reproduce

```
python -m scripts.sim_connect_check                      # confirm READY
python -m scripts.aigp_vq1_run --minimal --cruise-speed 16 --max-tilt 0.82 \
  --aim-z -0.85 --kv 3.0 --max-vert-speed 9.5 --lookahead 0.7 --speed-brake 1.3 \
  --speed-min-frac 0.55 --speed-descent-gain 1.5 --aim-slew 12 --final-aim-z 0.5 \
  --final-brake-band 26 --record captures/run.jsonl.gz --max-seconds 75   # 16.2s 6/6
python -m scripts.iter36_compare captures/run.jsonl.gz   # per-gate margins
```
Captures from this loop: `captures/iter{1..8}_*.jsonl.gz` (+ `*_run.log`).

## Racing-line redesign attempt (iter 9) — the lap is DESCENT-FLOORED

Pushed the spline speed profile toward the bandwidth budget: `--spline-v-descent
2.0→3.0`, `--spline-a-lat 6.5→8.0`, cruise 18. Result: **1/6, gate2 breached by
8.6 m on the TOP edge** (vert Z=−8.59), gates 1 & 3 also breached high; roll-rate
clamp 67% saturated. The failure mode is decisive:

- **Faster descent → the drone flies HIGH over gates 1–3.** Raising `v_descent`
  lets it go faster horizontally on the steep legs, so it covers the ground
  before it can descend the forced Δz → arrives above the gate → sails over the
  top. `v_descent=2.0` isn't conservative — it's holding horizontal speed down
  so the (bandwidth-limited ~2 m/s) vertical can keep up. The descent cap is the
  binding wall.
- **`a_lat=8.0` saturates roll (67%)** — the smooth spline has *less* lateral
  roll-rate headroom than hoped; pushing it just re-hits the roll wall.

**The lap time is physics-floored, not tuning-limited.** Gates 1–3 force a
~19.5 m (gate0→gate3: ~24.6 m) descent; the descent rate is walled at ~2 m/s
vertical (faster → tumble or fly-high). That's a hard ~10–12 s floor for the
descent section alone. The gate-by-gate champion already flies **gate0→gate3 in
10.8 s (≈2.27 m/s vertical — AT the wall)**; the near-level gate3→gate5 finish is
fast. Net: **16.2 s is essentially the physical floor for this course on this
sim.** The racing line cannot beat it — pushing the descent flies high, pushing
the lateral saturates roll. Confirmed across iters 2, 6, 8, 9.

**Implication:** further *speed* work is futile without beating the descent/roll
bandwidth wall (RL-territory, deep-research-ranked not-first). The achievable
race-day value is **RELIABILITY** — guaranteeing the 16.2 s 6/6 every race
despite sim degradation / gate-map corruption (monitors already built + validated
live). Optionally, port gate-by-gate's descent handling into the spline to get a
*fast-AND-high-margin* controller (~matches 16.2 s with more clearance) — a
reliability win, not a speed win.

## Reliability campaign (15/15) + the controller frontier

**Champion reliability — PROVEN.** 15 consecutive champion runs (gate-by-gate
cruise-16, `--abort-on-degraded`): **15/15 sim-credited 6/6** (`race_finished=True`),
total **19.0–19.08 s (g0->g5 16.2 s) +/- 0.04 s**, **0 collisions**, worst frame
margin 0.235–0.295 m. Monitors held across all 15: gate-map <=0.021 m drift vs the
session reference; sim-health Z=-1.66..-1.74 (healthy). The sim did NOT degrade
this session (~24 runs), so the monitors are proven to NOT false-trigger live; the
degradation *catch* stays offline-validated (unit tests on synthetic over-climb /
sign-flip / drift).

**Reporting fix (iter 69):** the runner now headlines the SIM-authoritative result
(`OFFICIAL RESULT (SIM): FULL COURSE COMPLETE [OK]`), so the geometric sequencer's
center-based 5/6 undercount no longer reads as a failure on race day.

**"Fast-AND-high-margin" (option b) is NOT achievable — speed and margin trade off
via the same wall.** Spline with `--spline-v-descent` raised from 2.0 to the
champion's measured 2.27 m/s (2.3): 6/6, g0->g5 16.94 s, but worst margin 0.219 m
at gate3 (the steep descent leg now arrives high). The spline's margin advantage
*came from* its slow descent; speeding it up tightens the descent gate to match.
spline-vd2.3 (16.94 s, 0.219 m) is slightly *dominated* by gate-by-gate
(16.2 s, 0.235 m).

**FRONTIER CONCLUSION:** the gate-by-gate champion (16.2 s, 6/6, 0.235 m worst
margin, 15/15 reliable) sits on the speed/margin frontier set by the descent/roll
bandwidth wall — no config is both faster and higher-margin. Further improvement
needs a different control architecture (RL — deep-research-ranked not-first for
this deadline/single-track setting), not tuning. **Race-day posture: fly the
champion; the armed gate-map + sim-health monitors plus the `.exe`-restart
procedure (`aigp-sim-connect` skill) cover the documented sim-degradation risk.**

## Best-of-many speed hunt (2026-06-17) — objective INVERTED

**Competition format: every run is uploaded and only your BEST run counts** —
there is no single race. This inverts the strategy: reliability is NOT the goal;
the single fastest sim-credited 6/6 is, and **crashes are free**. So the "16.2 s
floor" (a *reliability*-conservative limit) re-opens — push past the safe envelope
and keep the lucky fast landings.

**Result — a faster best run:**

| config | total | g0→g5 | completion | notes |
|---|---|---|---|---|
| champion `--cruise-speed 16` (band 26) | 19.05 s | 16.2 s | ~100% | clean, the SAFE upload (guaranteed 6/6) |
| **`--cruise-speed 17 --final-brake-band 38`** | **18.27 s** | **15.45 s** | **~30–50%** | the FAST upload; sim-credited 6/6, achievable 0-collision |

~0.8 s (~5%) faster. Achieved repeatedly (batches 1 & 3). Full command:
```
python -m scripts.aigp_vq1_run --minimal --cruise-speed 17 --max-tilt 0.82 \
  --aim-z -0.85 --kv 3.0 --max-vert-speed 9.5 --lookahead 0.7 --speed-brake 1.3 \
  --speed-min-frac 0.55 --speed-descent-gain 1.5 --aim-slew 12 --final-aim-z 0.5 \
  --final-brake-band 38 --record captures/run.jsonl.gz --max-seconds 75
```

**Key findings:**
- **The aggressive margin is stochastic** (unlike the deterministic champion,
  ±0.04 s). The *same* cruise-17/band-38 config completed 6/6 (18.27 s) on some
  runs and diverged on others — so **best-of-many luck IS exploitable here**: run
  it many times, keep the best clean 6/6.
- **cruise 17 is the sweet spot.** cruise 17.5+ fails at the **gate-0 launch**
  (too aggressive on the initial accel/bank), not the descent. cruise 18 lands
  *only* with launch-softening so heavy (aim-slew 6, lookahead 1.6, descent-gain
  2.5) that it's **slower** (24.33 s) — no good.
- **Decoupling (fast straights + hard descent braking) did NOT unlock cruise 18**
  — the limiter is the launch, not the descent. The descent is still walled
  (~10.8 s), and that + the gate-0 launch cap a *fast* landing at ~cruise 17, so
  **~18.3 s is near the best-of-many floor** for this control architecture.
- One 18.27 s landing logged 1 collision (grazed a frame but the sim credited
  6/6); the other was clean — so **clean 18.27 s runs exist**; upload a clean one.

**Race-day upload strategy:** bank the champion (cruise 16, ~100%, 19.05 s) as a
guaranteed 6/6, AND run cruise-17/band-38 several times — upload the fastest clean
sim-credited 6/6 (≈18.27 s when it lands). Monitors (gate-map + sim-health) stay
armed to discard degraded runs. Beating ~18 s needs a different controller
(descent wall) — RL was ruled out (see `rl/FIDELITY_VERDICT.md`).

**Launch-cap experiment (iter 74, `--launch-speed`):** added an opt-in cap on the
spawn→gate0 leg only (champion untouched, default 0=off; sanity-verified: launch
off → 6/6 @ 19.05 s). Hypothesis: the gate-0 launch caps fast cruise. Result:
with `--cruise-speed 18 --launch-speed 16` cruise-18 **does now land 6/6 — but at
23.78 s (slow).** Capping the launch fixed gate0, but then the **descent at cruise
18 overshoots/struggles** and the lap is slow. So the launch was *a* limiter but
the **descent wall is the binding one** — cruise 18 cannot do a *fast* clean lap
regardless of launch handling. **Conclusion: cruise-17 / 18.27 s is the firm
best-of-many floor**; the only way below it is beating the descent bandwidth wall
(new controller / RL — ruled out). `--launch-speed` is kept as a (harmless,
opt-in) tool.
