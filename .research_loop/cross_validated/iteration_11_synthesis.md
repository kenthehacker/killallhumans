# Iteration 11 — Research Synthesis: Reference-Velocity Drag Feedforward

## Bottleneck: Control — Helix Tracking Error (gates 7-12)

The PD controller (kp=6, kd=4, ff=0.4) averages 0.641m error on helix gates vs 0.31m on straight gates. The controller saturates at 0.85 rad tilt during consecutive tight turns.

## Mathematical Root Cause Analysis

The kinematic sim dynamics are:
```
accel_des = kp * ep + kd * ev + ff * ref_acc    (controller output)
accel_actual = accel_des - drag * vel             (sim physics, drag=0.5)
```

Defining tracking error e = ref_pos - pos, the error dynamics are:
```
e_ddot + (kd + drag) * e_dot + kp * e = (1 - ff) * ref_acc + drag * ref_vel
```

The **forcing term** `(1-ff)*ref_acc + drag*ref_vel` drives steady-state error:
```
e_ss = ((1-ff) * ref_acc + drag * ref_vel) / kp
```

At helix parameters (ref_acc ≈ 5 m/s², ref_vel ≈ 10 m/s, ff=0.4):
```
e_ss = (0.6 * 5 + 0.5 * 10) / 6 = 8 / 6 = 1.33 m
```

**The drag term (0.5 * 10 = 5.0) is larger than the feedforward deficiency (0.6 * 5 = 3.0)!**
Drag forcing dominates the steady-state error budget.

## Key Insight: Reference-Velocity Drag Feedforward

Adding `drag_ff * ref_vel` to accel_des (where drag_ff = sim drag coefficient = 0.5):

```
accel_des = kp*ep + kd*ev + ff*ref_acc + drag_ff*ref_vel
```

New error dynamics:
```
e_ddot + (kd + drag)*e_dot + kp*e = (1-ff)*ref_acc
```

The drag*ref_vel forcing term is **completely eliminated**. The damping ratio remains unchanged at ζ = (kd + drag)/(2*sqrt(kp)) = 0.92 because drag on actual velocity is preserved.

New steady-state error:
```
e_ss = (1-ff) * ref_acc / kp
```

With ff=0.7, ref_acc=5: e_ss = 0.3 * 5 / 6 = 0.25m (vs 1.33m before — 5x improvement!)
With ff=0.8, ref_acc=5: e_ss = 0.2 * 5 / 6 = 0.17m

## Why This Is Different From Iteration 9's Failed Drag Compensation

Iter 9 added `drag_coeff * vel` (CURRENT velocity) to accel_des:
```
accel_actual = (kp*ep + kd*ev + ff*ref_acc + drag*vel) - drag*vel = kp*ep + kd*ev + ff*ref_acc
```
This completely cancelled drag on vel, reducing effective damping from kd+drag=4.5 to kd=4.0.
The damping ratio dropped from 0.92 to 0.82, causing oscillation and 73-86% error regression.

This iteration adds `drag_ff * ref_vel` (REFERENCE velocity):
```
accel_actual = (kp*ep + kd*ev + ff*ref_acc + drag*ref_vel) - drag*vel
             = kp*ep + kd*ev + ff*ref_acc + drag*(ref_vel - vel)
             = kp*ep + (kd+drag)*ev + ff*ref_acc
```
Drag on velocity error (ev) is PRESERVED, maintaining the full 0.92 damping ratio.
Only the steady-state forcing from drag is eliminated.

## Paper Support

1. **Tal & Karaman 2018** (aggressive_tracking_tal_2018): "feedforward is the most important single fix for geometric controllers." Full 4th-order feedforward achieves 6.6cm RMS.
2. **L1Quad 2025** (l1quad_adaptive_geometric_wu_2025): L1 adaptive drag compensation on geometric controller achieves 5x smaller RMSE. Our approach is the analytical equivalent for known drag.
3. **DATT 2023** (datt_adaptive_tracking_2023): L1 adaptive outperforms learned adaptation for drag. Ref-velocity feedforward is the closed-form version.
4. **Leveling the Playing Field 2025**: "feedforward is the most important single fix for geometric controllers" — our ff=0.4 is too conservative.

## Consensus
All papers agree: feedforward acceleration must be near 1.0 for aggressive tracking. The barrier to higher ff in our system is drag-induced model mismatch. Reference-velocity drag feedforward resolves this analytically.

## Proposed Implementation
1. Add `velocity_feedforward` parameter to TrackerConfig (default 0.0 for backward compatibility)
2. Add `velocity_feedforward * ref_vel[i]` to accel_des in GeometricTracker.track()
3. Set velocity_feedforward = 0.5 (matching sim drag coefficient)
4. Increase feedforward_accel from 0.4 to 0.7
5. Update benchmark.py's TrackerConfig to match

## Expected Impact
- Helix gates 7-12 error: 0.641m → ~0.35m (45% reduction)
- Overall avg error: 0.481m → ~0.35m
- Race time: maintained at 13.34s (no trajectory change)
- Risk: if ref_vel feedforward is too strong, could cause oscillation near trajectory boundaries. Mitigate by starting at velocity_feedforward=0.3 and increasing.
