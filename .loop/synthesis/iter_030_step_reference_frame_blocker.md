# iter-030: NED↔ENU mismatch is the real step_reference blocker

## What iter-030 attempted

Composer's 3-agent recommendation was correct: iter-029's smoke test
used `derive_safe_max_velocity` (15 m/s) instead of race_01.json's
`plan_max_speed_mps=4.0` (the PyBullet matrix path's velocity). Fixed
that — smoke now plans at 4 m/s, matching the matrix path.

Also applied composer's #2 fix: tracker's `max_tilt_rad` now clamps
to plant's `max_roll/pitch_angle` so the tracker can't command
tilts the plant will throw away.

## What remained broken

Even at 4 m/s with consistent tilt caps, the smoke STILL crashes
because:

1. **PyBullet timestep / control-rate mismatch**: `applyExternalForce`
   only persists for one stepSimulation. Smoke ran 240 Hz physics
   with 120 Hz control → drone got force half the time → effective
   half-gravity hover. **Fixed**: set physics timestep = control
   timestep (1/120 s).

2. **NED vs ENU coordinate frame mismatch**: GeometricTracker is
   hardcoded for NED (`mpc_tracker.py:203` subtracts `(0,0,+g)`,
   line 216 says `z_b_des = (0,0,-1)` for hover, etc.). PyBullet is
   ENU (z-up). Naive z-flip conversion partially fixed the thrust
   sign (drone falls slower) but **the drone now flies in the WRONG
   xy direction**: target (5, 0, 1.5) → drone ends up at (-7, -2.6).

## Why the naive z-flip is wrong

NED↔ENU has 4 coupled sign issues:

| Quantity | NED convention | ENU convention | Conversion |
|---|---|---|---|
| z position | down positive | up positive | flip sign |
| Gravity vector | (0,0,+g) | (0,0,-g) | flip sign |
| Body z-axis (hover) | (0,0,-1) | (0,0,+1) | flip sign |
| Yaw rotation | CW from north | CCW from east | reverse direction |
| Pitch (nose-up) | negative | positive | flip sign |
| Roll (right-wing-down) | positive | positive | same |

Flipping only z gets gravity / body-z right but leaves pitch/yaw
inverted → drone tilts the WRONG way → flies wrong direction. The
0.41 m vs target 5.0 m horizontal error after 1.75 s is consistent
with the tracker commanding +5 m/s acceleration in -x.

## Why fixing this is research-scale

Three architectural options:

1. **Refactor mpc_tracker.py to be frame-agnostic.** Add a
   `convention: Literal["NED", "ENU"]` to TrackerConfig that
   parameterises the gravity vector, hover body-z, and yaw sign.
   Then a `TrackerConfig(convention="ENU")` works directly with
   PyBullet state. Cost: ~30-50 line refactor + test suite update
   (lots of tests assume NED). Worth doing but not in one iter.

2. **Full ENU→NED→ENU conversion in step_reference.** Convert
   position/velocity/acceleration/yaw on input; convert
   cmd.roll/pitch/yaw on output. 6+ sign flips, fragile, easy to
   regress.

3. **Write an ENU-native tracker for QuadrotorDrone.** Subclass
   GeometricTracker, override the gravity-vector and body-z
   constants. Less code than (1) but creates two parallel control
   stacks to maintain.

## Net findings from iter-029 + iter-030

- The iter-029 "QuadrotorDrone is underpowered" diagnosis was
  WRONG. The drone is fine at 4 m/s; the smoke test was running it
  at 15 m/s due to a max-velocity-resolution bug.
- The iter-026b/c plumbing (drone_spec routing, step_reference) is
  STRUCTURALLY correct but blocked on the NED↔ENU mismatch.
- The real path to closing task #13 (matrix-vs-demo divergence) is
  option (1) above — frame-agnostic tracker — and that's a
  separate multi-iter project.

## What this iter ships

- Smoke max-velocity resolution fixed (now respects race_01.json's
  plan_max_speed_mps).
- Smoke timestep / control-rate matched.
- Tracker tilt cap matches plant tilt cap in step_reference.
- step_reference's broken naive-z-flip conversion REVERTED to
  unconverted (which is also broken but at least consistent with
  the "this needs the proper tracker refactor" finding).
- This synthesis doc as the next-iter playbook.

## Recommendation

Don't continue chasing this iter's loop. The frame-agnostic tracker
refactor needs to be its own iter (or its own session) with proper
testing across NED matrix and ENU PyBullet paths. iter-031+ should
either:
- Land the refactor (Option 1) end-to-end with a parity test that
  proves matrix == PyBullet on race_01, OR
- Pivot to remaining charter items that aren't blocked on this
  (task #10 peak accel gap, or new substantive items the user
  surfaces).

Per the brick-wall rule, iter-030 is "diagnostic infra shipped; the
underlying bug needs research-scale work to fix properly". Loop
iter 4/50 used.
