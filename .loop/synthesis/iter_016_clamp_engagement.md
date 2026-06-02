# Iter-016: empirical clamp-engagement data (Composer-4 verification)

Closes the iter-010 review hypothesis-check that Composer-4 flagged:
*"instrument benchmark.py:486 to log clamp-active fraction before
declaring victory on alignment."*

## Method

Added two metrics to the synthetic bench's `controller_trace_summary`:
- `accel_clamp_active_frac` — fraction of steps where the bench's
  acceleration saturator engaged (`accel_mag > max_accel = 15 m/s²`)
- `speed_clamp_active_frac` — fraction where speed saturator engaged
  (`speed > max_speed = 15 m/s`)
- `max_accel_mag_pre_clamp` — peak commanded accel before clamp

Probed all six non-figure8 matrix tracks at `duration=30s` against
the iter-013 baseline (DroneConstraints.max_acceleration=15.0 inherited
from drone_spec).

## Findings (2026-05-24, commit ~ee0f0b1/3fb0a83)

| Track             | accel_clamp_active | peak_accel_mag | speed_clamp |
|-------------------|--------------------|----------------|-------------|
| aigp_default      | 72.2%              | 79.88 m/s²     | 0.0%        |
| grand_tour        |  9.9%              | 33.68 m/s²     | 0.0%        |
| race_01           | 21.7%              | 52.83 m/s²     | 0.0%        |
| slalom            | 36.1%              | 37.54 m/s²     | 0.0%        |
| straight_hairpin  | 11.5%              | 40.84 m/s²     | 0.0%        |
| vertical_cliff    |  4.2%              | 44.78 m/s²     | 0.0%        |

## Interpretation

**iter-010 hypothesis was directionally correct.** The bench DOES
saturate — frequently, especially on aigp_default (72.2% of steps)
and slalom (36.1%). The bench is NOT a hypothetical clamp — it's an
active constraint that the planner regularly hits.

**BUT the planner's soft penalty is too soft.** Peak commanded
accelerations of 33-80 m/s² are *5×* the bench's 15 m/s² ceiling.
Dropping `DroneConstraints.max_acceleration` from 20 → 15 in iter-010
narrowed the gap but didn't close it — the polynomial-trajectory
optimizer's accel penalty (Opus M3 noted: scalar finite-difference at
segment boundaries, *not* peak polynomial accel) doesn't actually
constrain the peaks the bench then has to clamp.

**Implication for aigp_default's tracking degradation (0.205 →
0.233m).** The hypothesis "planner commands 20 m/s², bench delivers
15, fix mismatch → tracking improves" is incomplete: the planner was
ALWAYS commanding *80 m/s²*, well above either ceiling. Dropping the
soft bound 20 → 15 changed the BO scorer's basin (per iter-010 review),
producing a different optimal racing line. The new basin happens to
saturate slightly more on aigp_default → marginally worse tracking.

## What this implies for future iters

1. **Iter-010's accel drop was right but cosmetic.** The bench was
   always operating at saturation; the change reduced the planner's
   *announced* envelope to match its *actual* effective envelope, but
   the gap between planner-commanded and bench-deliverable accel
   remains large.

2. **Real fix:** TOPP-style peak-acceleration retiming
   (`_topp_retime` already exists in trajectory_optimizer; just needs
   to be called consistently and given the bench's actual bounds, not
   the planner's soft-bound). Or move the bound from a soft penalty
   to a hard projection.

3. **Composer-4 was right to insist on measurement.** Without this
   data we'd assume the iter-010 change "fixed" something. It moved
   numbers around but the underlying lie — planner thinks it can
   command 4–5× what the bench delivers — is still open.

4. **race_01 lap time is suspiciously low (17.2s) given 21.7%
   clamping.** If we got the planner to respect the bench's ceiling
   the laps would necessarily slow down. So the matrix gate at 22.5s
   has comfortable headroom for a planner that's more honest about
   its envelope.

## Next iter candidates

- **A: TOPP-style peak retiming** (~M effort, high impact — closes
  the actual envelope gap; likely lap-time slowdown).
- **B: Replace optimizer soft penalty with hard projection on peak
  polynomial accel** (~S, lower impact).
- **C: Re-tune the racing-line BO with the clamp engagement as a
  score component** (~M; better basin selection at low ratios).

None implemented in iter-016 — this iter is observation only. The
metrics are now permanently captured in `controller_trace_summary`
for ongoing tracking.
