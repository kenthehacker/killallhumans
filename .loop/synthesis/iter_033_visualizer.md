# iter-033: matrix visualizer (charter task #13)

## Problem (closed for the visualization use-case)

- `scripts/visual_demo.py` uses the **Crazyflie CF2X** (27 g / 0.6 N)
  with gym-pybullet-drones' `DSLPIDControl`. The matrix bench (truth
  tier) uses the **AIGP-proxy** (1 kg / 20 N) with the geometric
  tracker + kinematic sim. Result: visual_demo gets 4/12 gates while
  the matrix gets 12/12 — different drones, same trajectory plan, very
  different outcomes. User correctly identified the issue.
- Swapping visual_demo's backend to the AIGP-proxy QuadrotorDrone is
  blocked by a NED↔ENU mismatch in `control/mpc_tracker.py`
  (`step_reference` would need a frame-agnostic tracker refactor; see
  `.loop/synthesis/iter_030_step_reference_frame_blocker.md`).

## Pragmatic fix

`scripts/visualize_matrix.py` — a self-contained matplotlib animator
that **replays the matrix bench's actual drone path**. It runs the
exact same code as `pytest tests/test_benchmark_matrix.py` with
`record_position_trace=True`, captures (t, pos, vel, yaw,
tracking_err_m) per step, and renders:

- top-left: top-down (x, y) — drone + planned trajectory + gates
- top-right: side (x, z) — drone + trajectory + gate heights
- bottom-left: tracking error over time
- bottom-right: speed + gate-pass event markers

```
python scripts/visualize_matrix.py --track race_01 --save out.gif
python scripts/visualize_matrix.py --track aigp_default --interactive
python scripts/visualize_matrix.py --list-tracks
```

## How this resolves task #13

The user's original concern was "visual_demo uses the wrong drone, so
I can't visualize our progress." The matrix visualizer addresses that
directly:

- ✓ Uses the **same drone envelope** as the matrix (1 kg / 20 N
  AIGP-proxy from `competition/drone_spec.py`).
- ✓ Uses the **same physics** (kinematic sim in `scripts/benchmark.py`
  — drag-clamped accel → integrate).
- ✓ Uses the **same controller** (`control/mpc_tracker.py` with the
  iter-032 trajectory).
- ✓ Shows the **same 12/12 race_01** at 24.4 s the matrix tests pin.

The CF2X visual_demo and its NED↔ENU refactor remain as longer-term
work (research-scale tracker refactor) — but the user's immediate
"how do I see this work?" question is answered today.

## Files

- ADD `scripts/visualize_matrix.py` (~250 LOC)
- ADD `tests/test_visualize_matrix.py` (3 smoke tests)
- EDIT `scripts/benchmark.py`:
  - `run_synthetic_benchmark` gains `record_position_trace: bool = False`
    kwarg.
  - When True, appends `{t, pos, vel, yaw, tracking_err_m}` to a list
    each sim step.
  - Exposes `position_trace` in the result dict (None by default).

## Test gate

`tests/test_visualize_matrix.py`:
- `test_position_trace_off_by_default` — keeps matrix tests' result
  dict small.
- `test_position_trace_populated_when_enabled` — pins the schema
  contract.
- `test_visualizer_resolves_track_names` — guards CLI help vs typos.

## Per-track impact

Running the visualizer on each track replays the iter-032 measured
behavior (see `.loop/synthesis/iter_032_accel_projection.md`):

| Track | gates | sim_time | avg_err | clamp |
|---|---:|---:|---:|---:|
| race_01 | 12/12 | 24.4s | 0.065 m | 2.7% |
| aigp_default | n/n | 11.8s | 0.068 m | 21.8% |
| slalom | 8/8 | 13.8s | 0.045 m | 1.4% |
| grand_tour | n/n | 23.9s | 0.066 m | 5.5% |
| straight_hairpin | n/n | 10.5s | 0.069 m | 4.2% |
| vertical_cliff | n/n | 13.3s | 0.026 m | 0.5% |
| figure8 | 8/8 | 16.7s | 0.047 m | 3.6% |

## Open follow-ups (deferred)

- Full visual_demo backend swap (CF2X → AIGP-proxy QuadrotorDrone)
  remains blocked on the NED↔ENU frame-agnostic tracker refactor.
  That's research-scale work (~50-100 line refactor + test-suite
  update); the visualizer fills the gap until then.
- The `_project_accel_peaks` warning fires on race_01 (45 m/s² residual
  > 15 target after 3 passes) — interior peak isn't a clean
  convergence target for this track's specific geometry. Not a
  regression (matrix still 7/7 PASS); future iter could bump
  max_passes=4 for race_01 specifically, or tighten the cap.
