# Iteration 12 — Research Synthesis: Trajectory-Aware Gain Scheduling

## Papers Analyzed (New)
1. **Deep Q-Learning-Based Gain Scheduling** (arXiv:2603.03127, 2026) — DQN selects from 625 pre-certified gains; phase variable enables trajectory-aware anticipation; 14-dim state input
2. **Task-Parameter Nexus** (arXiv:2412.12448, 2024) — Speed×curvature grid determines optimal gains; D-gain should be HIGH for hover, REDUCED for aggressive trajectories; 12 parameters via Batch-DiffTune
3. **Adaptive Gain Scheduling using RL** (arXiv:2403.07216, 2024) — PPO reactive gain scheduling; 43-49% ISE reduction; observes current error only (no trajectory lookahead)

## Previously Analyzed (Key References)
4. **TACO** (arXiv:2511.02060, 2025) — Trajectory-aware gain optimization at 2Hz; MLP surrogate predicts tracking cost; matches static oracle with zero-shot generalization
5. **Tal & Karaman** (arXiv:1809.04048, 2018/2021) — Jerk/snap feedforward via differential flatness; 6.6cm RMS at 12.9 m/s
6. **Aggressiveness-Aware Control** (arXiv:2602.21936, 2026) — Formal gain scheduling framework; minimize aggressiveness subject to tracking bounds; GP-based disturbance compensation

## Consensus Across Papers
1. **Static gains are a fundamental compromise** — All 6 papers agree that fixed gains perform poorly across diverse trajectory regimes. TACO shows 62% error reduction (0.80→0.30m) on hard trajectories.
2. **Gain scheduling provides 30-50% tracking improvement** — PPO (43-49% ISE), TACO (62% on hard), DQN (demonstrated on aggressive maneuvers), TPN (generalizable to unseen trajectories).
3. **Trajectory curvature/acceleration is the best scheduling signal** — TPN shows speed×curvature are the two dominant factors. TACO uses trajectory lookahead. DQN uses phase variable (trajectory progress). PPO uses error (reactive, less effective than predictive).
4. **Predictive beats reactive** — TACO (trajectory-aware) outperforms PPO (error-reactive) approaches because gains adapt BEFORE the error builds.

## Key Contradiction: D-gain During Turns
- **TPN**: D-gain should be REDUCED during aggressive/high-curvature maneuvers (high D-gain causes oscillation at high speed)
- **TACO/DQN**: Both increase gains during turns
- **Resolution**: The contradiction is about PROPORTIONAL boost. TPN says the RATIO of P/D should shift toward P during turns. All papers agree P-gain should increase for turns. D-gain should increase less or stay flat.

## Actionable Direction: Simple Curvature-Based Gain Scheduling

Given our constraints (kinematic sim, pre-computed trajectory, no ML training infrastructure), the optimal approach is:

**Use trajectory acceleration magnitude (at lookahead point) as curvature signal → scale gains proportionally**

### Why This Works in Our System
1. Our trajectory is pre-computed — we know acceleration at every point
2. We already compute ref_ahead for the 50ms feedforward lookahead
3. The acceleration magnitude directly correlates with maneuver difficulty
4. During turns, accel_des saturates at max_accel=15 m/s² — a higher kp shifts the clamped direction toward error correction
5. Straights keep base gains → no regression on straight-segment tracking

### Design Parameters (from research)
- kp boost: 1.0-2.0x during turns (TACO suggests kp can go from 6→15 safely)
- kd boost: 1.0-1.3x (TPN warns against excessive D-gain on aggressive trajectories)
- Smoothing: EMA to avoid spiky behavior at polynomial segment boundaries
- Curvature reference scale: ~30 m/s² acceleration (moderate turn threshold)

### Expected Impact
- Gate-7 error: 0.659→0.45-0.50m (TACO-like improvement on hard sections)
- Gate-8 error: 0.528→0.40m
- Helix avg: 0.455→0.35m
- Straight gates: maintained or slightly improved
- Overall avg error: 0.358→0.30m
- Race time: maintained (gain scheduling doesn't slow down the trajectory)

### Risk Assessment
- Too much kp boost → oscillation on turn recovery
- Too much kd boost → oscillation during high-speed turns (TPN warning)
- Poor smoothing → gain discontinuities at segment boundaries
- Mitigation: conservative initial boost (0.5x), systematic sweep
