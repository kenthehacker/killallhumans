# iter-029: QuadrotorDrone smoke test fails — physics blocker

## Smoke test result

Built minimum-viable PyBullet harness in
`scripts/smoke_quadrotor_drone_race.py`:
- Connect headless PyBullet client
- Construct `QuadrotorDrone(config=DroneConfig())` at race_01 start
- Generate race_01 trajectory via planner (max_v ~15 m/s from
  `derive_safe_max_velocity`)
- Loop: `drone.step_reference(ref)` + `p.stepSimulation()`

Outcome:
```
crash_ground at t=1.16s, pos=(-0.85, -0.66, 0.04)
gates_passed: 0/12
avg_tracking_error_m: 0.66
max_tracking_error_m: 1.81
```

**Drone falls within 1.2 s.** Started at z=1.5, ended at z=0.04.

## Diagnosis

The race_01 trajectory commands peaks of 50-80 m/s² (per iter-016
clamp engagement instrumentation). QuadrotorDrone has:
- mass=1 kg, max_thrust=20 N → T/W = 2:1
- max accel @ full throttle = (20/1) - 9.81 = ~10 m/s² lateral
  (and 10 m/s² vertical headroom)

So the drone can deliver AT MOST 10 m/s² maneuvering accel. The
trajectory wants 50-80 m/s² peaks. Saturation on every aggressive
move → no thrust budget for hover → drone falls.

For reference, AIGP-class 280 mm racing drones have T/W = 3-5:1
(typically 600-900 g + 30-50 N total thrust). The 1 kg / 20 N
"AIGP proxy v1" in `drone_spec.py` is **systematically underpowered**
for race_01's planner-derived speeds.

## Why this wasn't caught earlier

The synthetic kinematic bench `accel_clamp` saturates lateral accel
at 15 m/s² (iter-016 documented this). At those saturated levels,
race_01 still passes 12/12 — because the bench's controller doesn't
have a gravity-vs-tilt trade-off. The kinematic sim just clips
commanded accel at 15 m/s² and accepts the implicit hover-while-
maneuvering assumption.

The PyBullet rigid-body QuadrotorDrone has NO such cheat. Tilting
30° to get 5 m/s² lateral STEALS thrust from hover → drone falls.

## Implications for the iter-026 plan

Opus's plan assumed QuadrotorDrone at 1 kg / 20 N is a viable
backend for the matrix-vs-demo unification. It's NOT viable for
race_01 as currently planned. Three paths forward:

1. **Reduce planner aggression**: lower `max_velocity_mps` and
   `max_acceleration` so the trajectory commands ≤ 8 m/s² (a number
   QuadrotorDrone can deliver while hovering). Cost: race_01 lap
   time goes from 17.2 s → 30-40 s. Probably necessary anyway for
   PyBullet honesty.

2. **Increase drone T/W ratio**: bump `max_thrust_n` from 20 → 40 N
   in `drone_spec`. Now T/W = 4:1 which matches AIGP-class. But this
   changes ALL paths (kinematic bench, tracker, etc.) — see iter-010
   review's warning about envelope rebaseline. Should be done with
   SITL data, not guesses.

3. **Use DSLPIDControl (gym-pybullet-drones)**: the existing GPDDrone
   path uses DSLPIDControl which handles thrust+tilt trade-offs
   automatically. The user's 4/12 demo result already uses this path.
   Stick with GPDDrone but feed it AIGP-class mass/thrust via URDF
   override or env params.

## Recommendation

For the loop budget remaining (~46 iters), tackle option 1 or 3,
NOT option 2 (which requires SITL calibration we don't have).

Option 1 is cleanest: cap the planner's max accel via a "PyBullet-
proxy" envelope much tighter than the kinematic bench's. This makes
the planner generate trajectories the QuadrotorDrone can actually
fly, at the cost of slower lap times. Honest about reality.

Option 3 stays on CF2X gym-pybullet-drones but tunes everything to
match it. This is what visual_demo currently does — user gets 4/12.
We'd improve by ALSO fitting the planner to CF2X (max_v ~4 m/s).

Neither option lands in one iter. Both are multi-iter projects.

## Per the brick-wall rule

iter-029 is a documented blocker, not a working iter. The smoke
script `scripts/smoke_quadrotor_drone_race.py` is shipped as
diagnostic infrastructure for whoever picks up the sim-stack
unification next. The QuadrotorDrone + step_reference plumbing
from iter-026b/c remains correct — the issue is the PLANNER, not
the plumbing.

Closing the iter with the script + this synthesis doc; task #13
stays open with revised understanding.
