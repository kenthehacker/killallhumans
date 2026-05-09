# Iteration 9 Plan — Drag-Compensated Feedforward + Inflation Reduction

## Objective
Recover race time from 16.69s toward 14s target by:
1. Adding drag compensation to enable higher feedforward weight → better tracking at speed
2. Reducing inflation parameters → faster trajectories leveraging improved controller

Target: race time <15.5s, avg error <0.25m, 100% gate pass

## Research Basis
- Faessler et al. 2018: drag-compensated differential flatness
- Zhang et al. 2024 (L4DC): drag-aware planning reduces tracking error 83%
- Teissing et al. 2024 (RA-L): drag modeling critical at high speeds
- Wu et al. 2025 (IEEE TCST): drag compensation enables 5x tracking improvement

## Files to Modify

### 1. `control/mpc_tracker.py` — Add drag compensation
**Change**: In `GeometricTracker.track()`, add drag compensation term:
```python
# Add drag compensation (Faessler 2018)
# The kinematic sim applies -drag*vel as aerodynamic damping.
# Compensating for this known force lets feedforward be more accurate.
drag_compensation = self.config.drag_coefficient * vel
accel_des = accel_des + drag_compensation
```

**TrackerConfig changes**:
- Add `drag_coefficient: float = 0.5` (matching kinematic sim's drag=0.5)
- Change `feedforward_accel: float` from 0.4 to optimal value from sweep

### 2. `planning/trajectory_optimizer.py` — Reduce inflation
**Change** in `_inflate_sharp_turns()`:
- Reduce `a_centripetal_threshold` from 3.5 to 4.5 (less sensitive)
- Reduce centripetal inflation coefficient from 0.25 to 0.15
- Reduce angle-based inflation coefficient from 0.35 to 0.25

### 3. `scripts/benchmark.py` — Update TrackerConfig
**Change**: Update TrackerConfig instantiation to use new defaults (drag_coefficient, feedforward_accel).

## Algorithm Changes (Pseudocode)

### Step 1: Drag compensation
```
# In GeometricTracker.track():
accel_des[xy] = kp_xy * ep[xy] + kd_xy * ev[xy] + ff * ref_acc[xy] + drag_coeff * vel[xy]
accel_des[z]  = kp_z * ep[z]   + kd_z * ev[z]   + ff * ref_acc[z]  + drag_coeff * vel[z]
```

Math verification:
- Sim computes: `actual_accel = accel_des - 0.5 * vel`
- With compensation: `actual_accel = (kp*ep + kd*ev + ff*ref_acc + 0.5*vel) - 0.5*vel`
- Simplifies to: `actual_accel = kp*ep + kd*ev + ff*ref_acc`
- Drag is perfectly cancelled! Feedforward can now be 1.0.

### Step 2: Feedforward sweep (in benchmark)
Test ff_weight in {0.6, 0.8, 0.9, 1.0} WITH drag compensation active.
Select optimal by avg_error + gate pass rate.

### Step 3: Inflation reduction
With optimal ff+drag_comp active, reduce inflation and benchmark.

## Risk Assessment
- **Drag compensation**: Near-zero risk. The math is exact for kinematic sim.
- **High feedforward**: Low risk with drag compensation. Without it, ff>0.5 regresses. With it, ff=1.0 should work.
- **Inflation reduction**: Medium risk. Could cause gate misses if too aggressive. Sweep conservatively.

## Rollback Criteria
- If avg error increases >15% OR gate pass rate <100% OR crash: revert inflation changes, keep drag compensation
- If even drag compensation regresses: full revert

## Test Plan
1. Add drag compensation, run benchmark with ff=0.4 (should improve slightly)
2. Sweep ff={0.6, 0.8, 0.9, 1.0} with drag comp
3. With best ff, sweep inflation reduction
4. Full benchmark to confirm final configuration
