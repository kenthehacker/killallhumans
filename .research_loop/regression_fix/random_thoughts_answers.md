# Answers to `random_thoughts.md`

## Q1. Why are we hard-coding values like `max_compression_helix`? Isn't there a programmatic way of setting these?

**Short answer: yes, and you're right — we shouldn't be.** Nearly every tuning parameter in this codebase is a hardcoded magic number tuned by hand-iteration against `race_01.json`. The comments next to them ("iter 36: Pareto rebalance from 0.76", "iter 44: 12%→6%") are a scar-tissue history of empirical sweeps.

Three levels of "programmatic" we could move to, in increasing ambition:

| Level | What it means | Example |
|---|---|---|
| **1. Config-driven** | Move magic numbers from code to per-race `race_*.json` so different courses can override | `max_compression_helix` moves out of `trajectory_optimizer.py:963` into `race_01.json` under `planner.timing.max_compression_helix` |
| **2. First-principles derivation** | Compute the value from physical constants at runtime | `max_compression_helix = f(helix_radius, max_lateral_accel, controller_bandwidth)`. For a helix of radius R at speed v, centripetal accel = v²/R. Solve backward from "accel must be ≤ 80% of controller saturation" to derive the maximum allowed speed per segment, which dictates the minimum time inflation. |
| **3. Auto-tuning** | Automated search (Bayesian optimization, CMA-ES) over the parameter set against a gate-pass + tracking-error objective | Run 100 trials of (compression, inflate, lookahead, max_speed) and pick the Pareto frontier. Requires a faster sim loop to be practical. |

**Where we are now:** level 0. Everything is hand-tuned.

**Recommended next step:** Level 1 for the obviously-course-specific knobs (below), Level 2 for things with a clean physical derivation (speed caps from actuator limits), and defer Level 3 until we have a second course to validate against.

---

## Q2. I need the list of parameters and what each one does

Complete inventory of current hardcoded tuning values and their meanings:

### Planner / trajectory (file: `planning/trajectory_optimizer.py`)

| Param | Value | Where | What it does | Course-sensitive? |
|---|---|---|---|---|
| `DroneConstraints.max_velocity` | 15.0 m/s | L33 | Absolute velocity ceiling used in min-snap QP (overridden in `visual_demo.py` to 4.0) | Yes — depends on drone & course |
| `DroneConstraints.max_acceleration` | 20.0 m/s² | L34 | Acceleration ceiling used in QP penalty | Yes — drone-specific |
| `max_accel` (kinematic sim) | 15.0 m/s² | L226 | Used during `_simulate_time` to pick segment times | Mostly drone, not course |
| `kp_xy, kd_xy` (kinematic sim) | 6.0, 4.0 | L229 | Simulated PD gains during segment-time search | Drone-specific, should match PyBullet PID |
| `ENTRY_EXIT_OFFSET` | 0.4 m | L558 | Distance along gate normal where entry/exit waypoints are placed | Yes — gate-spacing-dependent |
| `helix_entry_inflate` | 1.06 (6%) | L904 | Minimum time inflation on first helix entry segment | Yes — helix-specific; different for slalom-only course |
| `helix_interior_inflate` | 1.06 (6%) | L912 | Minimum time inflation on every interior helix segment (our iter-7 change) | Yes — helix-specific |
| `max_compression_helix` | 0.72 | L963 | Hard floor on how much the time-compression pass can shrink helix segments (0.72 = never below 72% of original) | **Highly course-specific** |

### Racing-line optimizer (file: `planning/racing_line.py`)

| Param | Value | Where | What it does | Course-sensitive? |
|---|---|---|---|---|
| `max_lateral_offset` | 0.6 | L78 | Max offset from gate center as fraction of half-width (0.6 × 0.6m = 0.36m) | Track-agnostic (fractional) |
| `corner_cut_aggressiveness` | 0.7 | L83 | Seed weighting: 0=center, 1=max corner cut | Somewhat course-specific |
| `speed_weight` | 1.0 | L84 | Objective weight for lap-time in L-BFGS cost | General |
| `smoothness_weight` | 0.40 | L85 | Objective weight for path smoothness | Tuned for race_01 (comment: "sweet spot for helix") |
| `lookahead_gates` | 3 | L92 | How many future gates the optimizer considers at each waypoint | Track-agnostic |
| `N_STARTS` | 10 | L112 | Multi-start count for L-BFGS escape from local minima | General |
| `_select_by_sim` `max_accel` | 15.0 | L515 | Simulated accel in racing-line tiebreak | Drone-specific |
| `_select_by_sim` `max_speed` | 15.0 | L516 | **Bug:** hardcoded, ignores `PLAN_MAX_SPEED` (codex iter 4) | Should inherit from caller |
| `_select_by_sim` `kp_xy, kd_xy` | 7.0, 5.5 | L518 | PD gains in racing-line tiebreak sim | Drone-specific |
| `SpeedProfiler.max_speed` default | 15.0 m/s | L675 | Speed cap in TOPP-RA forward/backward | Drone-specific |
| `SpeedProfiler.min_speed` default | 2.0 m/s | L676 | Speed floor | Drone-specific |
| `SpeedProfiler.max_accel` default | 8.0 m/s² | L677 | Longitudinal accel cap in TOPP | Drone-specific |

### Gate sequencer (file: `gate_sequencing/sequencer.py`)

| Param | Value | Where | What it does | Course-sensitive? |
|---|---|---|---|---|
| `pass_through_margin` | 1.0 | L49 | Multiplier on half-opening for plane-crossing check (1.0 = exactly the gate opening) | General |
| `proximity_pass_distance` | 0.0 default, **1.2 in visual_demo** | L50 | If >0, alternative pass-through: drone comes within this distance of center | **This is the one causing false positives — see below** |
| `off_track_distance` | 5.0 m | L52 | Distance from expected path before RECOVERY triggers | General |
| `max_approach_angle` | 1.2 rad | L53 | Unused currently | General |
| `detection_dropout_frames` | 30 | L54 | Frames without gate detection before slow-down | General |
| `recovery_speed_factor` | 0.3 | L55 | Speed reduction during RECOVERY | General |

### Visual demo / runtime (file: `scripts/visual_demo.py`)

| Param | Value | Where | What it does | Course-sensitive? |
|---|---|---|---|---|
| `PLAN_MAX_SPEED` | 4.0 m/s | L288 | Cap passed to SpeedProfiler + `DroneConstraints.max_velocity` | Course-specific |
| `MAX_CMD_SPEED` | 4.0 m/s | L438 | Runtime clamp on commanded velocity into `drone.step()` | Course-specific |
| `lookahead_time` | `closest.time + 0.3s` | L427 | How far ahead on the reference the tracker reads | Somewhat course-specific |
| `recovery slow-down scale` | 0.3 | L461 | Velocity scale when `should_slow_down()` | General |
| `proximity_pass_distance` override | 1.2 m | L256 | Overrides sequencer default — **source of false-positive passes** | **Should be ≤ half-opening (0.6m)** |

### Summary: which knobs actually bind right now?

1. **`PLAN_MAX_SPEED` / `MAX_CMD_SPEED`** — directly controls both trajectory and commanded velocity.
2. **`max_compression_helix`** — controls how tight the helix timing gets post-TOPP.
3. **`helix_interior_inflate`** — last iter's winning change.
4. **`proximity_pass_distance`** — **false-positive accelerator**, fixing this now.

---

## Q3. Do hardcoded parameters mean we perform well on race_01 but could do horribly on another course?

**Yes, almost certainly.** Evidence:

- `max_compression_helix = 0.72` is labelled "Helix floor (iter 36: Pareto rebalance from 0.76 — swept 0.70-0.74)". It was found by manual sweep against race_01's specific helix geometry (6 gates, radius ~4m, 60°-increment yaws).
- `helix_interior_inflate` / `helix_entry_inflate` only apply when `segment.in_helix == True`. A straight-sprint course would never trigger them.
- `ENTRY_EXIT_OFFSET = 0.4m` assumes gate spacing ≥ ~1m. A dense slalom with 0.5m gate-to-gate spacing would produce overlapping entry/exit waypoints and ill-conditioned min-snap.
- `max_lateral_offset = 0.6` (60% of half-width) assumes 1.2m gates. A course with smaller gates would have less effective corner-cutting room.
- `smoothness_weight = 0.40` was tuned against race_01's helix+slalom mix; a pure slalom course probably wants ≤0.2, a pure high-speed loop probably wants ≥0.6.
- `proximity_pass_distance = 1.2` was introduced in iter 6 *specifically to mask* that the helix trajectory skims outside the opening (see Q2 false-positive note).

**Concrete risks moving to a new course:**
1. **Tight slalom (gates <1m apart):** ENTRY_EXIT_OFFSET waypoints overlap → numerical instability.
2. **Large gates (>2m opening):** max_lateral_offset=0.6 gives ≥1.2m of corner cut room, would produce unnecessarily aggressive racing lines.
3. **Fast straight sections with only end-gates:** PLAN_MAX_SPEED=4.0 is leaving speed on the table; should derive from trajectory curvature + actuator limit.
4. **Acute corners (>120° gate-to-gate angle change):** helix inflations don't apply so we'd hit the bare min-snap geometry with no margin — likely miss.

**What "general" fixes would look like:**
- **Config-driven:** move `PLAN_MAX_SPEED`, `max_compression_*`, `*_inflate`, `ENTRY_EXIT_OFFSET`, `proximity_pass_distance`, `smoothness_weight` into the race JSON. Per-course tuning becomes data, not code.
- **Auto-derivation:** compute `max_compression_helix` from `helix_radius * max_lateral_accel / (PLAN_MAX_SPEED²)` + margin. Compute `PLAN_MAX_SPEED` from racing-line peak curvature. Compute `ENTRY_EXIT_OFFSET` from min gate spacing / 3.
- **Course-diversity validation:** add at least one more race JSON (slalom-heavy, sprint-heavy) to smoke-test every commit. Right now we have a test suite of one.

---

## On top of the above: a **bug** the PRD exposed

The PRD output shows:
```
PASSED gate-10 [10/12] miss=1.12m
PASSED gate-11 [11/12] miss=0.99m
PASSED gate-12 [12/12] miss=0.72m
```

Gate interior width = 1.2m, so **half_opening = 0.6m**. A miss distance of 1.12m means the drone physically passed 1.12m from the gate center — that is **outside the gate opening by 0.52m**. The drone is being credited for passing gate-10 while actually flying around it.

Root cause: `proximity_pass_distance = 1.2` in `visual_demo.py:256` credits any pass within 1.2m of gate center regardless of whether the drone went through the opening. This was introduced iter 6 to mask a trajectory that skims the helix gates.

**The right fix** (doing it next): tighten the proximity check so it only rescues genuine through-passes (proximity ≤ half-opening = 0.6m). This will expose the real gate-pass count and force us to fix the *trajectory* to actually fly through the lit openings — which is what the PRD meant by "pivot aggressively into the gate that is not perpendicular to the racing line."
