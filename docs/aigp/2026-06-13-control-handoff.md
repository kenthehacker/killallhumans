# AIGP Live-Sim Control Handoff — 2026-06-13

Handoff for the next agent. This session moved the drone from "spins wildly,
flies 2 km away, 0 gates" to "no spin, tracks the −X course for ~4 s, gets
within ~20 m of gate 0" — by debugging against the **live AIGP simulator**
(not the point-mass harness, which cannot see the sim's command interface).

Iteration count: durable log `docs/aigp/telemetry_loop_log.md` is numbered
through **Iteration 13** (pre-compaction, control-logic work). This session is
~iterations 14–20 (continuous diagnose→fix→test, not separately numbered).
Combined budget is 50 + 25 = 75 iterations; ~20 used.

---

## TL;DR — read this first

The original "drone spins in circles" was **real and live**, and its cause was
NOT in our control math — it was the **sim's command interface**:

1. **The sim mishandles `SET_ATTITUDE_TARGET` attitude mode** (type_mask `0b111`).
   Sending a held attitude makes the drone spin ~9 rad/s once airborne. The sim
   DOES honor **body-rate mode** (type_mask `128`). FIXED: `send_attitude` now
   converts desired attitude → body rate (quaternion-error PD) and sends rate
   mode.
2. **All three body-rate axes are sign-flipped** vs our FRD convention
   (gyro test: cmd +0.8 rad/s → measured gyro −0.76 / −2.10 / −1.78 on
   roll/pitch/yaw). Sending FRD rates straight = positive feedback = erratic.
   FIXED: `_rate_sign = (-1,-1,-1)` applied on send.
3. **Thrust was 2× miscalibrated.** `DEFAULT_MAX_THRUST_N` was 20 N (PyBullet
   placeholder); the live drone hovers at throttle ~0.20, i.e. real
   `max_thrust_n ≈ 42–48 N`. FIXED: set to 42.0.
4. **The logger was blind to erratic flight** — it only recorded yaw. FIXED:
   now records measured roll/pitch, gyro, and distance-to-target-gate; the
   analyzer flags oscillation / flip / tumble / vertical divergence.

After all four fixes the drone no longer spins and flies the course for ~4 s,
but **still flips at the gate-2/3 descent and climbs away (0/6 gates).** The
remaining root cause and recommended next steps are below.

---

## 🛑 P0 — STOP PATCHING, SIMPLIFY. Read this before you plan anything.

**Thesis: this stack is over-engineered for VQ1, and the over-engineering is
now the thing breaking us.** Do NOT continue tuning deterministic fudge factors
(accel targets, `should_slow_down` ×0.7 thrust / ×0.5 tilt, projection passes).
Those are band-aids on a mis-sequenced architecture. The next agent's FIRST job
is to strip the flight path down to basics and get a real, repeatable **6/6 at
low speed**, which has never once been achieved on the live sim.

### Why this is P0 (the honest diagnosis)

1. **We optimized before we could fly.** We built a ~2000-line min-snap RACING
   trajectory optimizer (racing line, TOPP-RA retiming, S-turn/helix detection,
   8-pass accel projection, FOV penalties) before validating basic stable flight
   on the real platform. "Make it fast" before "make it work."
2. **The optimizer is now the failure.** It emits references demanding ~18 m/s²
   (61° tilt) when the drone's limit is 49° (~11 m/s²). The drone *physically
   cannot* track the reference → falls behind → saturates → climbs/flips. We
   were then patching this with hand-tuned scale factors (`should_slow_down`).
   That is the wrong layer to work at.
3. **"We were fine before" was an illusion.** Every pre-AIGP green result came
   from unit tests or the point-mass harness (`sim_closed_loop.py`), which grades
   the controller against physics that MIRRORS the controller's own assumptions
   (NED, instant attitude, perfect telemetry). It cannot fail the way the real
   sim fails. The AIGP sim is our first ground truth; "failing now" = "finally
   measuring reality," not a regression.
4. **VQ1 does not need most of this.** Per `2026-06-10-first-contact-findings.md`:
   the 6 gate positions are **known/downloaded**, and the sim provides **accurate
   local position** (`LOCAL_POSITION_NED`/`ODOMETRY`). So VQ1 = "fly through 6
   known waypoints with good position feedback." That is a SOLVED problem with
   basic control. **No min-snap optimizer, no racing line, and no CV are required
   for VQ1.** (CV matters for VQ2, where visual guidance is off — not here.)

### The P0 plan (do this first)

Build a **minimal gate-to-gate controller** and get 6/6 slowly before any
optimization:

- **Controller**: position PID on the error to the current target gate →
  desired acceleration → desired attitude (roll/pitch from the horizontal accel,
  yaw pointed at the next gate, thrust from vertical accel) → the **body-rate
  inner loop we already fixed** (`_attitude_error_body_rates`, mask 128, rate
  signs negated). Fly straight lines, gate to gate, at **2–3 m/s**.
- **Bypass** the whole trajectory stack for now: no `TrajectoryOptimizer`, no
  racing line, no TOPP, no accel projection, no `should_slow_down`.
- **Keep** what the platform work validated: EKF/state, the downloaded gate map,
  the gate-pass sequencer, and the fixed body-rate inner loop.
- **Success = repeatable 6/6 on the live sim at low speed.** Only after that,
  re-introduce speed/optimization — and only with the optimizer constrained to
  the REAL measured limits (49° tilt / ~11 m/s², hover 0.20), not placeholders.

### Specific areas to investigate (use these to build your plan)

Work bottom-up; do not skip a rung. Each is a `scripts/aigp_bench.py`-style
controlled live-sim test:

1. **Hover hold** — can the drone hold a FIXED position (not just attitude)?
   Close a position loop around hover thrust; confirm it doesn't drift/climb.
   (Bench already shows attitude hold + hover thrust work; add the position
   outer loop.)
2. **Single-waypoint step** — command "go to a point 10 m ahead and stop." Tune
   the position-PID gains for a clean approach with no overshoot/oscillation.
   This is where you find the real usable speed and tilt the platform tolerates.
3. **Real flight envelope** — from the step tests, measure the max tilt / accel /
   climb-descent rate the drone reliably achieves WITHOUT flipping. This number
   (not the placeholder 49°/11 m/s²) bounds everything downstream.
4. **Gate-to-gate sequencing** — string the 6 known gate positions as waypoints;
   advance the target when within a capture radius. Verify the existing
   `sequencer` pass-detection actually registers a pass at low speed (it may be
   tuned for fast fly-through; check `pass_through_margin`).
5. **Yaw strategy** — decide whether to hold yaw = π the whole course (gates all
   face −X, so this is fine and simplest) or point yaw at the next gate. Simplest
   first.
6. **What to delete vs keep** — explicitly decide the bypass list. The min-snap
   stack and `should_slow_down` are prime candidates to disable for VQ1. Confirm
   nothing essential (state estimation, gate map, pass detection) depends on them.
7. **Decide the optimizer's future** — is the racing optimizer worth keeping at
   all for VQ1, or should it be reserved for a later "go fast" phase once 6/6 is
   solid? (Recommendation: shelve it for VQ1.)

**The P1 items in "RECOMMENDED NEXT STEPS" below (lower accel target, gate
`should_slow_down`) only matter if you choose to KEEP the optimizer. The P0 path
makes most of them moot.** Prefer P0.

---

## Live-sim ground truth (bench-measured — trust these)

Measured with `scripts/aigp_imu_probe.py` (passive) and `scripts/aigp_bench.py`
(controlled open-loop setpoints) against the running VQ sim:

| Property | Value | How measured |
|---|---|---|
| Body frame | **FRD / NED** (z down) | rest IMU `zacc = −9.64` (negative). NOT FLU. |
| Spawn pose | yaw = π (faces −X course), pitched ~−0.31 rad, at rest at origin | passive probe |
| Attitude mode (mask 7) | **SPINS** ~9 rad/s airborne (do not use) | bench A vs F |
| Body-rate mode (mask 128) | **honored**, holds attitude with zero rates | bench F |
| Body-rate sign | **all 3 axes inverted** (cmd +0.8 → gyro −0.76/−2.10/−1.78) | bench gyro test |
| Rate amplification | sim applies ~**2.5×** the commanded rate (cmd 0.8 → ~2.0) | bench gyro test |
| Real hover throttle | **~0.20** (vz≈0 at thr 0.18–0.21) | bench thrust sweep |
| Implied `max_thrust_n` | **~42–48 N** at mass 1 kg | a_z = g − T·Fmax/m |
| Telemetry rates | LOCAL_POSITION_NED ~95 Hz, ATTITUDE ~117 Hz, HIGHRES_IMU ~117 Hz, ODOMETRY ~74 Hz | first-contact doc |

**Frozen-telemetry warning is a FALSE ALARM.** The run logs "FROZEN for N
control ticks" because `timestamp_us` is written by three handlers using two
different clocks (LPN/ATTITUDE = ms, ODOMETRY = µs) that clobber each other and
even step backward. The feed is healthy 30–50 Hz. Optional fix: only let
`_handle_local_position` set `timestamp_us` (`competition/aigp_mavlink.py`
~:381 / :395 / :411). It is NOT a cause of the divergence.

---

## Current flight state (capture `captures/show_run.jsonl.gz`, default max_speed=8)

```
yaw: revs=-0.44 (NO SPIN ✓)   gyro_p95=3.3 rad/s
attitude: roll_osc 2.1Hz pitch_osc 1.4Hz  flip_frac=0.93  z=[-78, 1919] m
next-gate: closest approach to gate 0 = 20.9 m ; ended 2001 m from it
0/6 gates, 0 collisions
```

Behaviour: lifts cleanly, holds heading, tracks −X for ~4 s reaching the
gate-2/3 region, **then flips (tilted >75° for 93% of ticks) and climbs to
~1900 m**, diverging. Misses gate 0 because it is ~15 m too high when crossing
its plane (the climb), not because of lateral error at that point.

---

## Remaining ROOT CAUSE (physics, not a guess)

**The reference trajectory demands more lateral acceleration than the drone can
produce within its tilt limit.**

- Tilt clamp `max_tilt_rad = 0.85` (49°) → max lateral accel = g·tan(49°) ≈ **11 m/s²**.
- The min-snap trajectory peaks at **~18 m/s²** (even after the iter-14 accel
  projection fix), which needs **61° tilt** > the 49° clamp.
- So the drone physically cannot track the reference → falls behind → position
  & velocity error grow → `accel_des` grows → **thrust saturates at 0.95 (climbs)
  and tilt saturates** → at the steeper gate-2/3 descent it flips.

The `should_slow_down` block (`race_pipeline.py:547-553`) is a **mixed** factor:
its ×0.5 roll/pitch cut strips tilt authority to ~24° (4.4 m/s²) when off-track
(death-spiral, per the iter-15 adversarial review), but its ×0.7 thrust cut
actually *reduces* the climb (thrust is saturated at 0.95 either way). So do NOT
just delete it — gate it.

---

## P1 — optimizer-path fixes (ONLY if you keep the min-snap stack)

> ⚠️ These are SUPERSEDED by the P0 strip-down plan above for VQ1. They are the
> "if you insist on flying the racing optimizer" fixes. The P0 minimal
> controller makes #1 and #2 here mostly moot. Listed for completeness.

1. **Make the trajectory feasible within the tilt envelope (highest leverage).**
   Lower the accel target so the reference never demands >~10–11 m/s² (≤49° tilt).
   `competition/drone_spec.py:49 DEFAULT_MAX_ACCEL_MPS2: 15.0 → ~10.0`.
   **VERIFY OFFLINE FIRST for time-bloat**: the iter-14 change set
   `_project_accel_peaks` to 8 passes, which can inflate segment times a lot.
   (Live max_speed=3 already produced a pathological 623 s / 62k-pt trajectory —
   though offline `max_speed` does NOT change total_time at all, because the
   trajectory is curvature-limited, so max_speed is NOT a useful lever.) Build
   the trajectory offline and confirm total_time stays sane (~30–45 s) and peak
   accel drops to ≤11 m/s² before flying.
   Quick offline check:
   ```python
   from scripts.sim_closed_loop import _build_gates
   from race_pipeline import RacePipeline, PipelineConfig
   import numpy as np
   traj = RacePipeline(PipelineConfig())._compute_trajectory((0,0,20),(0,0,0),_build_gates())
   # sample velocity, finite-diff → accel magnitude; check max & p95
   ```

2. **Gate `should_slow_down` on lateral error** (`race_pipeline.py:547`).
   Only attenuate when ON-track and near a gate (e.g. `self._last_lateral_err <
   ~2 m`); when off-track, keep FULL tilt authority so the drone can recover
   instead of death-spiralling. Keep a (smaller) thrust easing for gate accuracy.

3. **Re-tune the inner attitude→rate loop once the trajectory is feasible.**
   Current gains (`competition/aigp_mavlink.py` `_att_rate_kp=2.0, _att_rate_kd=2.5,
   _att_rate_max=1.0`) are conservative/non-flipping but sluggish (bench: looser
   combos flip within 2–3 s because of the 2.5× amplification; `kd≥kp` was
   required). If the reference is gentler, you may be able to raise `max_rate`
   for crisper tracking. Tune with `scripts/aigp_bench.py` (roll-step, watch
   `max_tilt`/`flip` and reversal count).

4. **Fix the 3 failing harness tests** (`tests/test_closed_loop_flight.py`).
   They regressed because `configure()` builds a trajectory feasible for the
   42 N drone (`drone_spec` SSOT) while the harness physics is pinned to 20 N
   (`scripts/sim_closed_loop.py`), so the harness drone can't fly it → gate-0
   crash. Either build the harness trajectory with 20 N constraints too, or
   accept the harness as a deprecated control-logic-only guard and adjust the
   test. The harness CANNOT validate the real-sim command interface anyway
   (it consumes `AttitudeCommand` directly in NED, never through `send_attitude`).

---

## Files changed this session

- `competition/aigp_mavlink.py`
  - `send_attitude` now converts attitude → body-rate (mask 128) via
    `_attitude_error_body_rates` (module fn) when `self._use_rate_control`
    (default True); legacy attitude-mode path kept as fallback.
  - `send_attitude_rate` applies `self._rate_sign`.
  - `__init__`: `_use_rate_control=True`, `_att_rate_kp=2.0`, `_att_rate_kd=2.5`,
    `_att_rate_max=1.0`, `_rate_sign=(-1,-1,-1)`.
- `competition/drone_spec.py`
  - `DEFAULT_MAX_THRUST_N` 20.0 → **42.0** (live calibration).
  - `DEFAULT_ACCEL_PROJECTION_MIN_SEG_TIME_S` 0.15 → **0.05** (iter-14, un-skip
    through-gate segments in accel projection).
- `planning/trajectory_optimizer.py`
  - `_project_accel_peaks` default `max_passes` 3 → **8** (iter-14). Dropped peak
    86→18 m/s² offline, total 26.5→34.6 s. NOTE: still 18 m/s² > 11 m/s² tilt
    envelope — see next-step #1.
- `scripts/aigp_vq1_run.py`
  - Added `--max-seconds` (clean early-exit that STILL writes the capture; the
    pipeline's own timeout is 480 s, so a stuck run otherwise costs 8 min).
  - Recorder now logs `roll`, `pitch`, `gyro`, `target_gate`, `dist_target_gate`.
- `scripts/analyze_telemetry.py`
  - New anomalies: ATTITUDE OSCILLATION, ATTITUDE FLIP/TUMBLE, HIGH BODY RATES,
    VERTICAL DIVERGENCE; new stats incl. per-target-gate closest approach.
- `scripts/sim_closed_loop.py`
  - Pinned harness + its tracker to 20 N (decoupled from `drone_spec`) — see
    next-step #4 (3 tests still failing).
- `competition/tests/test_aigp_mavlink.py`
  - Split into legacy-attitude-mode, rate-control-default, and rate-negation
    tests (all pass).
- NEW tooling:
  - `scripts/aigp_imu_probe.py` — passive rest-state IMU/frame probe (zero cmds).
  - `scripts/aigp_bench.py` — controlled open-loop attitude/rate/thrust bench
    (reset→arm→hold setpoint→measure). Modes: attitude, att+yawrate, rate,
    att_via_rate (PD), idle. Has the gyro sign test and PD gain sweep.

---

## How to run (and gotchas)

```bash
# Live race, recorded, clean 75 s cap:
python -m scripts.aigp_vq1_run --record captures/run.jsonl.gz --max-seconds 75
# Analyze (erratic-flight flags + next-gate distance):
python scripts/analyze_telemetry.py captures/run.jsonl.gz
# Controlled attitude/rate bench (tuning):
python -m scripts.aigp_bench --hold 2.0
# Passive frame/IMU probe:
python -m scripts.aigp_imu_probe
```

- **Run as a module** (`python -m scripts.aigp_vq1_run`), not `python scripts/...`,
  or the `competition` import fails.
- **Port 14550 must be free** before each run. Check & kill stale runners:
  `netstat -ano -p UDP | grep 1455` then `Stop-Process -Id <pid> -Force`.
  A stale runner holds 14550 and keeps commanding the drone.
- **The recorder only flushes the capture on CLEAN exit.** `--max-seconds` exits
  cleanly; a `Stop-Process`/kill loses the capture. Let runs finish.
- **Sim must be in Virtual Qualifier mode** (vision firehose on UDP 5600,
  MAVLink to 14550). It is a fire-and-forget UDP sender, so `netstat` cannot
  confirm VQ mode — receive-probe instead. Sim is in Downloads (AI-GP Simulator
  v1.0.3364); login `kenichimatsuo1775@gmail.com`.
- **Frozen-telemetry ERROR in run output is a false alarm** (see above).

---

## Mandate reminders (from the user)

- Test every flight change against the **live AIGP simulator**, not matplotlib/
  pybullet/the point-mass harness (the harness hid all four bugs above).
- Read telemetry every iteration and check for anomalies (the logger/analyzer
  now does this — `analyze_telemetry.py` exits nonzero on any anomaly).
- Track how far the drone gets from the **next/target gate** (now logged).
- Loop budget: 50 + 25 = 75 iterations max; ~20 used.

See also memory: `project_aigp_sim_control_interface.md`.
