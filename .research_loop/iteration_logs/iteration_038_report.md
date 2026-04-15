# Iteration 38 — PD Gain + Feedforward Optimization

**Date**: 2026-04-15
**Bottleneck**: control (PD gains kp=6, kd=4, feedforward=0.4 — suboptimal per literature)
**Status**: COMMITTED — avg error 0.174→0.151m (-13.5%), largest improvement since iter 35
**Commit**: 2a7f540

---

## Section 1: Summary
- Iteration 38, timestamp 2026-04-15T11:22Z
- Bottleneck: control — PD gains and feedforward weight suboptimal per literature (NGTC uses kp=25, "Leveling the Playing Field" says feedforward is most important single fix)
- One-line outcome: **ff 0.4→0.50, kp 6→7, kd 4→5.5 via 40+ config sweep → avg error -13.5%, gate-3 -24%, gate-7 -24%**

---

## Section 2: Research

### Papers analyzed (new in this iteration)
1. **Almost Global Trajectory Tracking on S²** (CDC 2024, arXiv:2409.05702)
   - Thrust direction control on unit sphere with backstepping stability certificate
   - Confirms full feedforward + PD is the standard approach

2. **RL-Based PID Gain Prediction for Quadrotor UAVs** (Feb 2025, arXiv:2502.04552)
   - DDPG-based online PID gain fine-tuning
   - Shows RL-tuned gains outperform hand-tuned by reducing attitude errors

3. **Quadrotor MPC Trajectory Tracking** (Nov 2024, arXiv:2411.06707)
   - Linear and Nonlinear MPC comparison
   - MPC is overkill for our kinematic sim; PD+FF is sufficient

### Key existing papers re-consulted
4. **Leveling the Playing Field** (Kunapuli 2025) — feedforward is the MOST important single fix
5. **Tal & Karaman (2018)** — full flatness feedforward achieves 6.6cm RMS at 12.9 m/s
6. **TACO** (Sanghvi 2025) — gain ranges kp=[2,15], kd=[1,10]
7. **NGTC** (Pries 2025) — DFBC baseline uses kp=25, kd=11 (mass-normalized from 0.72kg)

### Key insight from synthesis
The literature unanimously uses feedforward_accel=1.0, but our kinematic sim with drag=0.5 causes catastrophic basin switching at ff>0.52. The viable parameter space is extremely narrow: ff=[0.48-0.50], kp=[7.0-8.5], kd=[5.0-6.5]. Outside this window, the controller can't track the trajectory and race time explodes to 27+ seconds.

### Research consensus vs contradictions
- **Strong consensus**: Full feedforward (ff=1.0) is correct for real dynamics
- **Contradiction**: Our kinematic sim's drag model makes ff>0.52 unstable (not addressed by any paper)
- **Consensus**: Literature gains are 2-4x higher than our original kp=6, kd=4
- **Practical finding**: Gains must be scaled with ff to maintain damping ratio

---

## Section 3: Implementation

### Changes made
1. **File**: `scripts/benchmark.py` (line 334-339)
   - `feedforward_accel`: 0.4 → 0.50
   - `kp_xy`: 6.0 → 7.0
   - `kd_xy`: 4.0 → 5.5

2. **File**: `control/mpc_tracker.py` (TrackerConfig defaults)
   - Same parameter changes in default values
   - Updated comments with research references and iteration 38 analysis

### Sweep methodology
- **Phase A**: Feedforward sweep [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] at kp=6, kd=4
  - ALL values ff≥0.5 caused race time to explode to 27+ seconds at kp=6
  - Root cause: higher ff amplifies feedforward acceleration, overwhelming the PD correction at low gains

- **Phase B**: kp sweep [7-12] at ff=0.8, kd=4
  - kp≥8 restores fast tracking at higher ff
  - But avg error WORSE than baseline due to S-turn oscillation (gate-3)

- **Phase C**: kd sweep at ff=0.4, kp=8
  - Higher kd dramatically improves: kd=6 gives avg=0.1638 (-5.9% from baseline)
  - Key finding: damping is the primary bottleneck, not just position gain

- **Phase D**: Combined ff×kp×kd sweep (18 configs)
  - ff=0.50, kp=7, kd=5.5: **best avg=0.1507**
  - ff=0.50, kp=7.5, kd=6.0: best max=0.7221
  - ff=0.60, kp=10, kd=8.0: best gate-7=0.1797

- **Phase E**: Ultra-fine sweep (15 configs around best)
  - Confirmed ff=0.50, kp=7.0, kd=5.5 is the global optimum
  - Sharp basin boundaries: ff=0.52 at kp=7 or kd=6.5 at kp=7 both catastrophic

- **Phase F**: Z-axis gain sweep — no improvement from kpz/kdz changes

### Plan adherence
Followed the plan's Phase A-C structure. The plan expected ff=1.0 to work based on theory. The sweep revealed the kinematic sim's drag model creates a narrow stability window around ff=0.50. Adapted by finding the optimal (ff, kp, kd) triple within this window.

---

## Section 4: Benchmark Comparison

### Metrics: Before vs After
| Metric | Before | After | Delta | Direction |
|--------|--------|-------|-------|-----------|
| Unit tests | 9/9 (100%) | 9/9 (100%) | = | = |
| Gates passed | 12/12 (100%) | 12/12 (100%) | = | = |
| Avg tracking error | 0.1741m | **0.1507m** | **-13.5%** | **Better** |
| Max tracking error | 0.7159m | 0.7421m | +3.7% | Slightly worse |
| P50 tracking error | 0.1636m | 0.1286m | -21.4% | Better |
| P95 tracking error | 0.3409m | 0.3530m | +3.5% | Slightly worse |
| EKF uncertainty | 0.0119m | 0.0119m | = | = |
| Race time | 14.02s | 14.01s | -0.1% | ≈ |
| Deterministic | YES | YES | = | = |
| Worst gate | gate-7 (0.247m) | gate-2 (0.218m) | -11.8% | Better |

### Per-gate error breakdown
| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.1109 | 0.1128 | +1.7% | ≈ |
| **gate-2** | 0.1787 | **0.2183** | **+22.2%** | **Regressed** |
| gate-3 | 0.2446 | **0.1860** | **-24.0%** | **Improved** |
| gate-4 | 0.2021 | **0.1655** | **-18.1%** | **Improved** |
| gate-5 | 0.1720 | **0.1334** | **-22.4%** | **Improved** |
| gate-6 | 0.1583 | **0.0877** | **-44.6%** | **Improved** |
| gate-7 | 0.2474 | **0.1884** | **-23.8%** | **Improved** |
| gate-8 | 0.2099 | **0.1636** | **-22.1%** | **Improved** |
| gate-9 | 0.1448 | 0.1432 | -1.1% | ≈ |
| gate-10 | 0.1535 | 0.1602 | +4.4% | Slightly worse |
| gate-11 | 0.1480 | **0.1139** | **-23.0%** | **Improved** |
| gate-12 | 0.1343 | 0.1471 | +9.5% | Regressed |

**Key observations**:
- Gates 3-8 all improved 18-45% — the tight S-turn and helix sections benefit most from better PD tracking
- Gate-6 had the largest improvement (-44.6%) — likely entering helix with better velocity control
- Gate-2 regressed 22% — this gate has a straight approach where the higher feedforward causes slight overshoot
- Gate-12 regressed 9.5% — the final gate approach is affected by accumulated velocity from higher ff

---

## Section 5: Deep Diagnostic

### 5b.1 — Root cause diagnosis
The original controller gains (kp=6, kd=4, ff=0.4) were set conservatively during early development. The literature uses 2-4x higher gains with full feedforward. Our kinematic sim's drag model (drag=0.5) creates a unique constraint: ff>0.52 causes catastrophic tracking failure because the feedforward acceleration (applied directly in the sim) overwhelms the PD correction when drag is high. The optimal operating point (ff=0.50, kp=7, kd=5.5) increases feedforward by 25% and PD correction by 30%, maintaining damping ratio ζ≈1.13 (slightly overdamped).

The gate-2 regression (+22%) is likely caused by the higher feedforward amplifying the acceleration during the straight→turn transition before gate-2. This is a transient effect that the steady-state error formula doesn't capture.

### 5b.2 — Telemetry signals
| Signal | Before | After |
|--------|--------|-------|
| max_roll | 0.85 rad | 0.85 rad (unchanged — still saturating) |
| max_pitch | 0.85 rad | 0.85 rad |
| avg_thrust | 0.796 | 0.836 (+5% — more aggressive control) |
| avg_pitch | -0.107 rad | -0.097 rad (less average forward lean) |

### 5b.3 — Basin switching discovery
**Critical finding**: The kinematic sim's parameter space has extremely sharp basin boundaries. Changing ff by 0.02 (from 0.50 to 0.52) at kp=7 causes race time to jump from 14s to 27.5s. This is NOT racing line basin switching (the racing line optimizer has hardcoded gains) — it's controller-induced tracking failure where the drone can't follow the pre-computed trajectory.

The basin boundary exists because the drag model (drag=0.5) creates a feedback loop: higher ff → more acceleration → higher velocity → more drag → more error → PD can't compensate → divergence.

### 5b.4 — Trend analysis
| Iter | Area | Approach | Avg Err Delta | Trend |
|------|------|----------|---------------|-------|
| 35 | trajectory | Helix TOPP 0.65→0.76 | -7.9% | BREAKTHROUGH |
| 36 | trajectory | Helix TOPP rebalance | ≈ | Pareto optimization |
| 37 | trajectory | S-turn TOPP 0.65→0.67 | -1.1% | Diminishing returns |
| **38** | **control** | **ff=0.50, kp=7, kd=5.5** | **-13.5%** | **BREAKTHROUGH** |

**Trend: NEW AREA BREAKTHROUGH.** Switching from trajectory_planning to control bottleneck produced the largest single-iteration improvement since iter 35. The bottleneck identification from iter 37 was correct: TOPP floor tuning was exhausted, and controller gains were the true limiter.

### 5b.5 — Architectural issues
1. **REMAINING**: Controller parameter space has dangerously narrow stability window
2. **REMAINING**: Max tracking error slightly increased (+3.7%) — p95 also +3.5%
3. **NEW**: Gate-2 regression (+22%) — straight→turn transition handling
4. **REMAINING**: Kinematic sim only — no PyBullet validation
5. **REMAINING**: Drag model mismatch — literature ff=1.0 doesn't work in this sim

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **[control] Gate-2 tracking fix — per-section gain scheduling or ILC refinement**
   - Gate-2 regressed 22% with new gains. This is a straight→turn transition.
   - Options: section-specific ILC alpha, or per-section lookahead_s tuning
   - Expected: fix gate-2 regression without losing gate-3-8 gains
   - Priority: 1
   - Research: TACO (trajectory-aware gains), per-section ILC (iter 28)

2. **[control] Velocity feedforward with ff=0.50 base**
   - With ff=0.50 (closer to steady state than ff=0.4), vff=drag may now work
   - Previously failed at ff=0.4 (iter 11) because "helix is all transients"
   - With better PD (kp=7, kd=5.5), transient behavior should be improved
   - Expected: further 5-10% avg error reduction
   - Priority: 2
   - Research: Faessler 2018, iter 11 analysis

3. **[trajectory_planning] TOPP fine-tuning in new controller regime**
   - The controller change affects the trackability curve
   - Old TOPP floors (helix=0.72, S-turn=0.67) may no longer be optimal
   - Expected: potential 2-5% further improvement
   - Priority: 3
   - Research: iter 35-37 floor analysis

4. **[system_integration] PyBullet validation**
   - All gains tuned on kinematic sim. Need to validate on real physics.
   - Priority: 4

5. **[control] Wider gain sweep with basin-aware search**
   - The basin boundary (ff≈0.52 at kp=7) may shift with different lookahead_s
   - Try ff=0.55 with lookahead_s=0.04 or ff=0.48 with lookahead_s=0.06
   - Priority: 5

### Next bottleneck
**control** — Gate-2 regression needs fixing. The controller is still the limiter but the focus shifts from global gain optimization to gate-2 specific tracking. May require per-section tuning or ILC refinement for the gate-2 approach.

### What NOT to try
- ff > 0.52 at any kp < 8 (basin switching — verified catastrophically)
- kp > 10 with kd < 6 (underdamped oscillation at gate-3)
- Z-axis gain changes (no effect — verified in sweep)
- Full feedforward ff=1.0 (kinematic sim drag model prevents this)

---

## Section 7: Lessons Learned

### What worked
- **Systematic parameter sweep**: 40+ configurations tested in ~5 minutes. Much more reliable than theory-based predictions.
- **Correct bottleneck identification**: Switching from trajectory_planning to control produced 13.5% improvement vs 1.1% from the last trajectory iteration.
- **Literature-guided search space**: NGTC and TACO papers gave the gain ranges to sweep.
- **Damping ratio framework**: ζ analysis correctly predicted which (kp, kd) combos would be stable.

### What didn't work
- **Full feedforward (ff=1.0)**: The literature unanimously recommends this, but the kinematic sim's drag model makes it catastrophically unstable. The sim's `accel = accel_des - drag*vel` model creates a unique feedback loop not present in real quadrotor dynamics.
- **Higher kp without higher kd**: Pure kp increase causes gate-3 S-turn oscillation.

### Surprises
- **Basin boundaries are SHARP**: ff=0.50→0.52 causes race time 14→27.5s. No gradual degradation.
- **Gate-6 improved 44.6%**: The largest single-gate improvement ever. The helix entry section benefited enormously from better velocity damping.
- **Gate-2 regression**: Higher feedforward causes overshoot on straight→turn transitions. This was not predicted by the steady-state analysis.
- **P50 improved more than avg**: Median error dropped 21.4% while avg dropped 13.5%, indicating the improvement is concentrated in the middle of the distribution.

### Process improvements
- The multi-phase sweep approach (ff sweep → kp sweep → kd sweep → combined → fine) is efficient and systematic
- Basin switching detection should be automated: flag configs where race_time > 20s
- The kinematic sim's drag model is a significant difference from real dynamics. Future iterations should consider whether gains optimized here will transfer.
