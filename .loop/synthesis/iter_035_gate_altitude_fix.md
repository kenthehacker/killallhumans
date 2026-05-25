# iter-035: gate-altitude planner bug (user-reported via visualizer)

## Problem

User watched the 3D PyBullet replay and noticed the drone wasn't
passing through gate centers vertically. Diagnostic:

| Track | Gate Δz_mean | Gate Δz_max |
|---|---:|---:|
| race_01 | (n/a logged) | (n/a logged) |
| **straight_hairpin** | **−0.349 m** | **0.361 m** |
| figure8 | −0.164 m | 0.363 m |
| grand_tour | −0.094 m | 0.498 m |
| vertical_cliff | −0.006 m | 0.364 m |

The smoking gun was straight_hairpin: all 6 gates at z=2.0, drone
passed at z=1.65 every time — a consistent 35 cm offset below.

## Root cause (TWO compounding bugs)

Diagnostic showed: drone position trace and planned trajectory both
sat at z=1.654 (Δ=−0.003 m between them — the tracker was tracking
perfectly). The trajectory itself was wrong by 0.346 m at gate-pass.

Tracing back to `RacingLineOptimizer._apply_offsets`:

```python
up = np.array([0, 0, -1])   # comment says "NED: -z is up"
pos = pos + up * vert_off * gate.height * 0.5
```

**Bug 1**: `up = [0, 0, -1]` is NED convention. Every other component
(bench, trajectory, visualizer, sim configs with z=1.5..10.5) uses
**ENU** (+z up). So `up * vert_off` pushed gates DOWN by `vert_off`,
exactly opposite of intent.

**Bug 2**: BO objective rewards short path. With start at z=1.5 and
gates at z=2.0, the optimizer found "lower the gates" cuts path
length by ~0.3 m per gate. Combined with the inverted `up`, it
converged on `vert_off > 0` to mean "lower the gate" — and the
straight_hairpin trajectory was 0.35 m too low on every gate.

## Fix

`planning/racing_line.py`:

1. `up = np.array([0, 0, 1])` — ENU correct. If anyone re-enables
   vertical offsets, the sign is now right.
2. New `RacingLineConfig.max_vertical_offset: float = 0.0` — default
   bounds the BO's vertical search to {0}. The drone passes through
   actual gate centers, maximising frame clearance (gates are
   1.5 m tall → 75 cm of vertical margin above and below center).
3. `bounds = [(-lat, lat)]*n + [(-vert, vert)]*n` — uses the new
   separate vertical bound.

Also invalidated `planning/racing_line_cache.json` (it had the buggy
offsets baked in; was already gitignored).

## Measured impact

| Track | Δz_mean before | Δz_mean after | Δz_max before | Δz_max after | sim_time after | avg_err after |
|---|---:|---:|---:|---:|---:|---:|
| race_01 | (n/a) | **−0.016** | (n/a) | **0.052** | 24.46s | 0.065m |
| aigp_default | (n/a) | **−0.003** | (n/a) | **0.018** | 14.87s | 0.043m |
| slalom | (n/a) | **−0.000** | (n/a) | **0.004** | 13.81s | 0.045m |
| grand_tour | −0.094 | **−0.025** | 0.498 | 0.278 | 24.04s | 0.066m |
| **straight_hairpin** | **−0.349** | **−0.001** | **0.361** | **0.004** | 10.45s | 0.069m |
| vertical_cliff | −0.006 | −0.008 | **0.364** | **0.020** | 14.27s | 0.027m |
| figure8 | −0.164 | **−0.005** | 0.363 | 0.025 | 16.73s | 0.047m |

straight_hairpin went from -0.35 m on every gate → -0.001 m (drone
passes through actual centers). vertical_cliff max gate error dropped
0.36 m → 0.02 m. figure8 max 0.36 m → 0.025 m.

**aigp_default tracking error dropped 37%** (0.068 → 0.043 m) — the
broken BO was previously cheating to shorten its path. With honest
gate altitudes, the planner has to climb properly; turns out doing
that produces better tracking too.

## Side effects

- aigp_default lap time: 11.78s → 14.87s (+26%). The buggy BO was
  shaving ~3s by lowering gates. Real lap time now reflects honest
  trajectory through actual gate centers. `SIM_TIME_CEILINGS` for
  aigp_default bumped 14.0 → 17.0 s.
- vertical_cliff lap time: 13.25 → 14.27s (+8%). Same effect, smaller
  magnitude.
- All other lap times unchanged.

## Test gates

ADDED `test_iter035_drone_passes_through_gate_centers_vertically` —
asserts mean |Δz| < 100 mm and max |Δz| < 400 mm per track. straight_
hairpin's 0.35 m bug would have failed this gate trivially.

RELAXED `test_matrix_pass_rate_at_least_six_of_seven`:
`SIM_TIME_CEILINGS["aigp_default"]` 14.0 → 17.0 s to absorb the
honest-gate-altitude lap time.

## Charter compliance

- No course-specific magic numbers. `max_vertical_offset` is a project-
  wide constant in `RacingLineConfig`.
- drone_spec.py SSOT preserved.
- No giga_chad pipelines used.
- Matrix 7/7 still PASS; figure8 still 8/8.

## Open follow-ups

- `_project_accel_peaks` print warnings still fire on straight_hairpin
  (residual 121 m/s² > 15 target) and vertical_cliff (43-301 m/s²). The
  honest trajectory through actual gate centers has even sharper
  polynomial peaks than the cheating one did. Not a regression
  (matrix 9/9 PASS) but bumping `max_passes=4` or relaxing
  `per_pass_cap` might converge further.
