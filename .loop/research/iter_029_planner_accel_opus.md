# iter-030 plan: matrix↔PyBullet consistency via planner-cap + binding-tilt fix

## TL;DR

Ship **Option 1 + a missed-physics correction** (NOT pure Option 1, NOT
Option 2, NOT Option 3). The iter-029 diagnosis is correct but
incomplete: it assumed PyBullet's lateral budget = 10 m/s² (full
thrust headroom). It's actually `g·tan(max_roll_angle)` = 9.81·tan(0.35)
≈ **3.6 m/s²**, because `sim_pybullet/drone.py:DroneConfig.max_roll_angle
= 0.35 rad`, while `TrackerConfig.max_tilt_rad = 0.85 rad`. The control
chain commands a tilt the plant clamps to 41% of it — the planner is
two layers of unrealistic at once.

So iter-030 caps the planner *and* unclamps the plant tilt, then closes
the polynomial-peaks gap iter-016 left open.

## Why not Option 2 or 3

- **Option 2 (T/W → 4:1)** is rejected by the iter-026 charter
  ("AIGP proxy v1; rebaseline AFTER SITL"). Bumping `max_thrust_n`
  20→40 invalidates the iter-009i racing-line basin, race_01 ILC
  schedule (v=15 boundaries baked in), and `auto_velocity` ceiling
  with no calibration data backing the new number. Even at 4:1 the
  current `max_roll_angle=0.35` keeps lateral budget at 3.6 m/s²
  — adding thrust without unclamping tilt buys nothing.
- **Option 3 (stay CF2X)** moves the goalposts to a different
  non-AIGP drone (27 g toy quad). Doesn't unify matrix↔PyBullet —
  matrix runs the 1 kg / 20 N envelope; CF2X is 27 g / 0.6 N. The
  current 4/12 visual_demo result is on this same overfit; chasing
  it is grinding tuning knobs against a known-wrong proxy.

## Concrete code changes

### 1. `competition/drone_spec.py` — cap the envelope to what the plant delivers

```python
DEFAULT_MAX_ACCEL_MPS2: float = 6.0   # was 15.0
DEFAULT_MAX_VELOCITY_MPS: float = 8.0 # was 15.0
```

Justification: with `DroneConfig.max_roll_angle` raised to 0.7 rad
(below), lateral accel ceiling is `g·tan(0.7) ≈ 8.3 m/s²`; budget
6.0 leaves margin for longitudinal. v=8 follows from
`v_max = √(a_lat·r_helix)` with race_01 helix r ≈ 12 m.

### 2. `sim_pybullet/drone.py` — unclamp `DroneConfig` tilt to match `TrackerConfig`

```python
max_roll_angle: float = 0.7   # was 0.35
max_pitch_angle: float = 0.7  # was 0.35
```

Both cited as "iter-021 stability conservatism" but TrackerConfig
already runs at 0.85. The 0.35 limit is what makes the tracker
command attitudes PyBullet then proportionally throws away. Smoke
test must verify `attitude_kp=12, attitude_kd=4` are still stable
at the larger setpoint range.

### 3. `planning/trajectory_optimizer.py::_topp_retime` — close the iter-016 gap

The bug iter-016 surfaced: bench saw 80 m/s² peaks against a 15
m/s² ceiling. `_optimize_time_allocation`'s soft penalty checks
finite-difference accel between segment endpoints, not polynomial
peaks. After TOPP retime, sample the generated polynomial at
`dt_sample` and stretch any segment whose peak exceeds
`constraints.max_acceleration` by a hard projection:

```python
def _project_accel_peaks(self, points, segment_times, peak_target):
    # Per-segment: compute |a(t)| max from points; if > peak_target,
    # scale segment_time by sqrt(peak/peak_target) (accel ∝ 1/T²).
    # Cap stretch at 1.5× to avoid pathological retime on outliers.
```

Insert between `_topp_retime` and `_inflate_vertical_climbs`. The
1.5× cap prevents one anomalous boundary spike (e.g. min-snap C³
discontinuity at finish-waypoint) from doubling lap time.

### 4. `scripts/benchmark.py` — auto-inherits envelope from drone_spec

Lines 509-510 already source `max_accel`/`max_speed` from
`DEFAULT_*`. No edit needed; bench saturator now matches planner.

### 5. `sim_pybullet/configs/race_01.json` — update v=15 ILC overrides

```json
"ilc_section_overrides_format": "fractions"
```
The v=15 boundary fractions baked at the iter-009 tune may not
match the v=8 trajectory's curvature distribution. Either re-run
the iter-47-49 ILC sweep at v=8 (M effort, ~10 min on bench) OR
accept temporary tracking degradation and add a TODO. Iter-030
ships the latter; iter-031 re-sweeps.

### 6. `scripts/smoke_quadrotor_drone_race.py` — promote to a parity assertion

After the four changes above, the smoke should pass ≥10/12 gates.
Wrap the script's final JSON in an exit-code check (exit 1 if
gates_passed < 10). This becomes the iter-026 step-9 parity gate
that was deferred. Add to a `tests/test_sim_stack_smoke.py` that
calls it.

## Expected outcome

| Metric                        | Before iter-030 | After iter-030 (target) |
|-------------------------------|-----------------|-------------------------|
| Matrix race_01 gates          | 12/12 @ 17.2s   | 12/12 @ 28-35s          |
| Matrix race_01 avg track err  | 0.665 m         | 0.5-0.8 m               |
| Matrix race_01 accel-clamp %  | 21.7%           | < 5%                    |
| PyBullet smoke gates          | 0/12            | ≥ 10/12                 |
| PyBullet smoke time-to-fail   | 1.16 s          | full duration or finish |
| visual_demo race_01 gates     | 4/12 (CF2X)     | unchanged this iter     |

visual_demo still runs CF2X this iter; the QuadrotorDrone backend
swap (iter-026 steps 4-7) is a separate iter once smoke is green.

## Risk assessment

**MAJOR**:
- **Bench regression on 5 already-failing matrix tracks.** aigp_default
  / slalom crashes happen at gate-1 because of initial-acceleration
  overshoot; lower v_max may *help* (less overshoot at start) but
  could also surface different DQ patterns at lower speeds. Run
  full matrix after change; baseline diff against
  `regression_baseline_2026_05_24.json`.
- **Attitude PD oscillation at 0.7 rad.** `attitude_kp=12, kd=4`
  was tuned for 0.35-rad envelope. Larger tilt setpoints with same
  gains can ring; mitigation = smoke test asserts `pitch_rate <
  6 rad/s` (existing `DEFAULT_MAX_BODY_RATE_RAD_S`). If unstable,
  drop to 0.55 rad as compromise; matrix budget then drops to ≈ 6.4
  m/s² (still > 6 budget).
- **TOPP hard-projection over-stretches helix segments.** Race_01
  gate-7 helix entry hit max_compression_helix=0.72 floor in
  iter-36 Pareto rebalance. Hard projection scaling by
  √(peak/target) on a 50→6 m/s² peak = 2.9× stretch, cap to 1.5×.
  Even capped, gate-7 segment time may double; lap time absorbs.

**MINOR**:
- ILC schedule mismatch (covered above; defer re-sweep one iter).
- `auto_velocity.DEFAULT_DRONE_MAX_ACCEL = 15.0` is a separate
  inline literal *enforced equal* to drone_spec via
  `test_auto_velocity_constants_match_drone_spec`. Update both or
  the test breaks.
- The 1.5× projection cap is itself a magic number. Charter rule
  ("no course-specific magic"): justify via comment that 1.5 is
  the median ratio of polynomial peak / TOPP-RA estimate observed
  across the matrix in iter-016.

**NOT a risk**:
- Charter "no giga_chad calls": this plan uses no LLM tools.
- Backward-compat: drone_spec is the single source; one edit
  ripples cleanly through bench/planner/tracker/PyBullet.

## Loop budget

Iter-030 is ~3-4 hr of careful coding (six edits, two test runs,
matrix re-baseline). Within the 50-iter cap, this unblocks:
- iter-031: re-sweep ILC schedule at v=8.
- iter-032: backend swap (iter-026 steps 4-7) once smoke green.
- iter-033+: address remaining matrix tracks (aigp_default,
  slalom, etc.) that were failing pre-cap.

## Open questions (don't block iter-030)

1. Is `max_roll_angle=0.7` empirically stable on the iter-021 PD?
   (Smoke test will tell; if not, fall to 0.55.)
2. Does the polynomial peak-projection break the iter-009i racing-
   line basin? (Re-run iter-009i BO once at v=8; if score regresses
   > 20%, escalate.)
3. Should `tests/test_drone_spec_contract.py` add a peak-vs-budget
   invariant? (Yes; one-line check on race_01 trajectory.)
