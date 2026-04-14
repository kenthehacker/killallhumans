# Iteration 8 Plan: Activate Feedforward Acceleration in Benchmark

## Objective
Fix the feedforward acceleration bug in the benchmark's kinematic sim loop. Pass trajectory-computed acceleration, jerk, and yaw_rate to the controller instead of zeros. This should break the speed-accuracy Pareto frontier by enabling the controller to anticipate trajectory changes rather than reacting to position errors.

**Target metrics:**
- Avg tracking error: 0.285m → <0.22m (est. 20%+ improvement)
- Race time: maintain 16.74s or potentially recover speed by reducing inflation
- All existing thresholds continue to pass

## Research Basis
1. **"Leveling the Playing Field"** (Kunapuli et al., 2025): "Feedforward is the most important single fix for geometric controllers"
2. **"Accurate Tracking of Aggressive Quadrotor Trajectories"** (Tal & Karaman, 2021): Up to 4th-order feedforward → 6.6cm RMS at 12.9 m/s
3. **DATT** (Huang et al., 2023): Feedforward-feedback structure → 34-36% error reduction
4. **"Differential Flatness with Rotor Drag"** (Faessler et al., 2018): Flatness-based feedforward for high-speed tracking

## Files to Modify

### 1. `scripts/benchmark.py` (PRIMARY FIX)
**Lines 376-384** — Change the ref_point construction to pass actual trajectory data:

```python
# BEFORE (bug):
ref_point = TrajectoryPoint(
    time=sim_time,
    position=tuple(target_pos),
    velocity=tuple(target_vel),
    acceleration=(0, 0, 0),
    jerk=(0, 0, 0),
    yaw=target_yaw,
    yaw_rate=0.0,
)

# AFTER (fix):
ref_point = TrajectoryPoint(
    time=sim_time,
    position=tuple(target_pos),
    velocity=tuple(target_vel),
    acceleration=ref.acceleration,
    jerk=ref.jerk,
    yaw=target_yaw,
    yaw_rate=ref.yaw_rate,
)
```

**Lines 364-373** — Gate-seeking fallback case must continue using zero acceleration (no trajectory data available in fallback). Need to track whether we're in fallback mode:

```python
# When in gate-seeking fallback (after trajectory ends):
use_fallback = sim_time > trajectory.total_time and not seq.is_complete
if use_fallback:
    # Override target from gate position (existing logic)
    ref_accel = (0, 0, 0)
    ref_jerk = (0, 0, 0)
    ref_yaw_rate = 0.0
else:
    ref_accel = ref.acceleration
    ref_jerk = ref.jerk
    ref_yaw_rate = ref.yaw_rate
```

### 2. `planning/trajectory_optimizer.py` (OPTIONAL — speed recovery)
If the feedforward fix significantly improves accuracy, we may have headroom to:
- Reduce centripetal acceleration inflation (currently threshold=3.5, coeff=0.25)
- Or increase time_weight from 2.0 to 2.5 for faster straight segments

**Only attempt this if the feedforward fix alone shows >15% accuracy improvement.**

## Algorithm Changes

### Before
```
Controller loop:
  ref = trajectory.sample(t)  → has acceleration, jerk, yaw_rate
  target_pos = ref.position
  target_vel = ref.velocity
  target_yaw = ref.yaw
  ref_point = TrajectoryPoint(pos, vel, acc=(0,0,0), ...)  ← ZEROED
  cmd = tracker.track(pos, vel, yaw, ref_point)
  accel = tracker.last_desired_acceleration  ← pure PD, no feedforward
```

### After
```
Controller loop:
  ref = trajectory.sample(t)  → has acceleration, jerk, yaw_rate
  target_pos = ref.position
  target_vel = ref.velocity
  target_yaw = ref.yaw
  ref_point = TrajectoryPoint(pos, vel, ref.acc, ref.jerk, yaw, ref.yaw_rate)  ← REAL
  cmd = tracker.track(pos, vel, yaw, ref_point)
  accel = tracker.last_desired_acceleration  ← PD + feedforward acceleration
```

The feedforward effect:
- **On straight segments**: ref_acc ≈ 0, feedforward adds little → no regression
- **Entering turns**: ref_acc has centripetal component → controller anticipates turn
- **Exiting turns**: ref_acc shifts back → controller anticipates straightening
- **Net effect**: controller leads the trajectory instead of lagging behind

## Risk Assessment
- **Low risk**: This is a bug fix. The code infrastructure already exists.
- **Regression risk**: Very low. Feedforward acceleration from min-snap polynomials should always improve tracking. The `feedforward_accel=1.0` weight can be reduced if needed.
- **Possible concern**: Very aggressive acceleration feedforward on short segments near gates might cause overshoot. Monitor per-gate errors at gate-1 (start) and gate-12 (finish) for any anomalies.

## Rollback Criteria
- If avg tracking error increases by >10%: revert
- If any gate fails to pass: revert
- If drone crashes: revert

## Test Plan
1. Run unit tests first (should be unaffected since benchmark code doesn't affect unit tests)
2. Run synthetic benchmark
3. Compare per-gate errors to identify which gates benefit most
4. If accuracy improves >15%, attempt speed recovery (increase time_weight or reduce inflation)
