# A Data-Driven Aggressive Autonomous Racing Framework Utilizing Local Trajectory Planning with Velocity Prediction
- **URL**: https://arxiv.org/abs/2410.11570
- **Authors**: Zhouheng Li, Bei Zhou, Cheng Hu, Lei Xie, Hongye Su (Zhejiang University)
- **Year**: 2024 (submitted October 2024, revised March 2025)
- **Venue**: arXiv preprint / under review

---

## Key Contribution

VPMPCC (Velocity Prediction based MPCC) extends the CiMPCC framework by learning the optimal velocity profile through Bayesian Optimization rather than using a hand-designed curvature mapping. The key insight is that the optimal velocity profile depends not just on curvature but on vehicle dynamics, track geometry, and safety margins — all of which are hard to model analytically. By using data-driven optimization with a novel Objective Function for Racing (OFR), the system automatically finds velocity profiles that balance racing performance with vehicle safety.

The second contribution is the OFR itself, which combines lap time minimization with safety penalties (lateral deviation, control effort) in a way that enables efficient Bayesian Optimization convergence — 42.86% fewer training iterations than standard alternatives.

---

## Technical Approach

### Architecture
1. **Offline phase**: Bayesian Optimization learns optimal mapping parameters for the curvature→velocity function
2. **Online phase**: VPMPCC uses the learned mapping to generate real-time velocity references within MPCC

### Velocity Profile Learning
The velocity mapping function is parameterized:
v_ref(s) = f(κ_norm(s); θ)
where θ are learnable parameters. BO searches over θ to minimize OFR:
OFR(θ) = w_time × T_lap + w_safety × max(e_lateral) + w_effort × sum(u²)

### Objective Function for Racing (OFR)
The OFR is designed to avoid common BO failure modes:
- **Time-only objective**: BO finds aggressive profiles that crash → infeasible
- **Safety-only objective**: BO finds slow, conservative profiles → not racing
- **OFR combines both**: lap time × (1 + penalty) where penalty activates only when safety thresholds are exceeded

### Sharp Turn Handling
The paper explicitly addresses tracks with sharp corners (their custom-built track has 90° turns). The learned velocity profiles show distinctive behavior at sharp turns:
- Deceleration begins 1-2 vehicle lengths BEFORE the turn apex
- The minimum velocity at the apex is proportional to the turn radius
- Acceleration begins at or slightly after the apex
- For consecutive sharp turns (chicanes), the velocity stays low between turns

This "decelerate early, stay slow through compound turns" pattern is exactly what we need for gates 3-4.

---

## Results

### Key Metrics
| Method | Mean Projected Velocity | Lap Time | Training Iterations |
|--------|------------------------|----------|-------------------|
| Standard MPCC | 82% of limit | baseline | N/A |
| CiMPCC (hand-tuned) | ~89% of limit | -11.4% | N/A |
| VPMPCC (BO) | **93.18%** of limit | **-12.5%** | 20 iterations |
| Alternative BO (standard) | ~91% of limit | -10.8% | 35 iterations |

### Training Efficiency
OFR-based BO converged in 42.86% fewer iterations than standard BO alternatives. This is important because each BO iteration requires a full lap simulation.

### Sim-to-Real Transfer
The learned velocity profiles transferred to physical F1TENTH hardware without retraining, achieving the same 93.18% velocity utilization. This suggests the learned profiles capture fundamental physics rather than simulator artifacts.

---

## Relevance to Our System

1. **Velocity profile learning for S-turns**: The VPMPCC approach of learning optimal velocity profiles through BO could be applied to our TOPP retimer. Instead of hand-tuning S-turn inflation factors, we could run BO over the inflation parameters using our synthetic_kinematic sim as the evaluation function.

2. **"Decelerate early" principle**: The learned profiles consistently show early deceleration before turns. Our `_inflate_sharp_turns` currently inflates the turn segments themselves but doesn't inflate the APPROACH segments. For gate-4 (the S-turn's second half), the approach segment (from gate-3 exit to gate-4 entry) should be slowed to allow lateral velocity reversal.

3. **Compound turn behavior**: The learned profiles for chicanes show sustained low velocity across the entire compound turn. This matches the CiMPCC insight: treat the S-turn as a single compound maneuver, not two independent turns.

---

## Actionable Takeaways

1. **Inflate approach segments, not just turn segments**: For S-turns, the segment BEFORE the second turn (gate-3 exit → gate-4 entry) needs extra time, not just the gate-4 entry/through segments.

2. **Use a compound turn detection heuristic**: If two consecutive gates have turns with opposite cross-product signs and are within 15m of each other, treat them as a compound S-turn and increase effective curvature by 1.3-1.5x.

3. **Consider BO for inflation parameter tuning**: Run Bayesian Optimization over the S-turn inflation factor using benchmark race time + tracking error as the objective. Our sim runs in <0.2s, so 50-100 BO iterations would take <20s.

4. **Early deceleration pattern**: The approach segment to an S-turn should be inflated by 10-15%, not just the turn segments. This allows the controller to begin lateral velocity reversal earlier.

---

## Limitations & Caveats

1. **Ground vehicle, not quadrotor**: Same limitation as CiMPCC — different dynamics, but the velocity profiling principle is universal.

2. **Requires simulation for BO**: The BO approach needs a fast simulator. Our synthetic_kinematic sim runs at ~8000 Hz, more than fast enough.

3. **Learned profiles are track-specific**: The optimal velocity profile depends on the specific track geometry. Different race configurations would need retraining.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| BO iterations to converge | ~20 | With OFR objective |
| Velocity utilization achieved | 93.18% | Of physical limits |
| Deceleration anticipation | 1-2 vehicle lengths | Before turn apex |
| Chicane velocity dip | Sustained low | Stays low between turns |
| OFR weights | track-dependent | Balanced time + safety |
