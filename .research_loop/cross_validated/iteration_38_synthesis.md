# Iteration 38 — Research Synthesis: Controller Feedforward and Gain Optimization

## Bottleneck: CONTROL (PD gains kp=6, kd=4, feedforward=0.4)

## Papers Analyzed (new + existing)

### New papers (this iteration):
1. **Almost Global Trajectory Tracking on S²** (arXiv:2409.05702, CDC 2024)
2. **RL-Based PID Gain Prediction for Quadrotor UAVs** (arXiv:2502.04552, Feb 2025)
3. **Quadrotor MPC Trajectory Tracking** (arXiv:2411.06707, Nov 2024)

### Key existing papers (re-consulted):
4. **Leveling the Playing Field** (arXiv:2506.17832, June 2025) — feedforward is the most important single fix
5. **Accurate Tracking of Aggressive Quadrotor Trajectories** (Tal & Karaman, 2018/2021) — full flatness feedforward achieves 6.6cm RMS
6. **TACO** (arXiv:2511.02060, Nov 2025) — trajectory-aware gain optimization
7. **NGTC** (arXiv:2510.12611, Oct 2025) — neural-augmented geometric control

---

## Consensus Across Papers

### 1. Full feedforward acceleration (weight=1.0) is the correct baseline
**Unanimously supported.** Every paper that implements a geometric/DFBC controller uses full reference acceleration feedforward (weight=1.0):
- Leveling the Playing Field: "Feedforward information is often replaced by an integral loop or simply omitted... This information is crucial and contributes a significant performance gain."
- NGTC: `F_des = m*(x_ddot_ref - K_v*(v-v_ref) - K_x*(x-x_ref) + g*e3)` — feedforward weight is implicitly 1.0
- Tal & Karaman: Full 4th-order feedforward through differential flatness
- TACO: Same geometric controller with full feedforward

**Our system uses feedforward_accel=0.4.** This is the single largest known deviation from the research consensus.

### 2. PD gains in the literature are significantly higher than ours
Our gains: kp_xy=6, kd_xy=4 (acceleration units, mass-normalized)

Literature comparison (normalized to acceleration units for 1 kg drone):
| Source | kp (position) | kd (velocity) | Notes |
|--------|---------------|---------------|-------|
| Our system | 6 | 4 | Current |
| NGTC (Pries 2025) | 25 (=18/0.72) | 11 (=8/0.72) | Mass-normalized from 0.72kg |
| Leveling the Playing Field | ~15 (from reward tuning) | ~8 | Bayesian-optimized |
| TACO suggested range | 2-15 | 1-10 | Training distribution |
| S² CDC 2024 | ~12-20 | ~6-10 | Theoretical analysis |

Our gains (kp=6, kd=4) are at the **low end** of the TACO training range and **4x lower** than NGTC's optimized values.

### 3. Damping ratio analysis for our kinematic sim
The effective error dynamics: `e_ddot + (kd+drag)*e_dot + kp*e = forcing`

With drag=0.5:
| kp | kd | ζ = (kd+0.5)/(2√kp) | ω_n = √kp | Behavior |
|----|----|----|----|----|
| 6 | 4 | 0.92 | 2.45 | Near critical damping (SLOW) |
| 10 | 5 | 0.87 | 3.16 | Slightly underdamped (better) |
| 12 | 6 | 0.94 | 3.46 | Near critical (good) |
| 12 | 5 | 0.79 | 3.46 | Underdamped (FAST response) |
| 15 | 6 | 0.84 | 3.87 | Good racing tradeoff |

Racing controllers typically use ζ ≈ 0.7-0.85 for fast response with moderate overshoot.

### 4. Steady-state error analysis
For our kinematic sim with drag=0.5:
```
steady_state_error = [(1-ff)*ref_acc + drag*ref_vel] / kp
```

At helix (ref_acc≈5, ref_vel≈8):
| ff | kp | Steady-state error |
|----|----|----|
| 0.4 | 6 | (3.0+4.0)/6 = 1.17m |
| 1.0 | 6 | (0+4.0)/6 = 0.67m |
| 1.0 | 10 | (0+4.0)/10 = 0.40m |
| 1.0 | 12 | (0+4.0)/12 = 0.33m |
| 1.0 | 15 | (0+4.0)/15 = 0.27m |

**Increasing ff from 0.4 to 1.0 at current kp reduces helix steady-state error by 43%.
Combining ff=1.0 with kp=12 reduces error by 72%.**

---

## Contradictions / Cautions

1. **Gain scheduling failed in iter 12** — but that was DYNAMIC gain scheduling. Static gain increase is different.
2. **Velocity feedforward (vff) failed in iter 11** — but only tested with ff=0.4. With ff=1.0, the system is closer to steady-state, so vff may work. Save for future iteration.
3. **TACO max speed was only 3.5 m/s** — our racing speeds are 10-15 m/s. Gains that work at low speed may not transfer.
4. **NGTC gains (kp=25, kd=11) may be too aggressive** — but NGTC used real dynamics with motor time constants. Our sim is simpler.

---

## Recommended Implementation Direction

### Primary: Full feedforward (ff=1.0) + PD gain sweep

**Evidence strength: Very strong (5 papers unanimous)**

1. Set feedforward_accel = 1.0 (the research baseline)
2. Sweep kp_xy from 6 to 15, kd_xy from 4 to 8
3. Keep kp_z and kd_z at current values (vertical tracking is not the bottleneck)
4. Measure avg error, max error, race time, worst gate

### Why NOT:
- SE(3) upgrade: our kinematic sim bypasses attitude entirely (uses raw accel_des). SE(3) gains (kR, kω) have no effect.
- MPCC: requires full dynamics model, optimization solver, huge implementation effort. Overkill for kinematic sim.
- NGTC/neural augmentation: requires training infrastructure, PyBullet only. Not applicable to kinematic sim.
- Gain scheduling: proven to fail in this sim (iter 12).

### Why this approach is safe:
- All changes are in benchmark.py TrackerConfig values only
- Easy to sweep systematically (each benchmark is ~10s)
- Fully reversible (just restore the three numbers)
- Strong theoretical basis (steady-state error formula)
