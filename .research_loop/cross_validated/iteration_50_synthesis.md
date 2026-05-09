# Iteration 50 — Research Synthesis: Competition Robustness & Final Polish

## Context
This is the **final iteration** of 50. The system is at peak performance:
- Race time: 13.31s (all-time best, held 5 consecutive iterations)
- Avg tracking error: 0.162m, all thresholds passing
- 100% gate pass, deterministic, no crashes

The focus is competition readiness, not further optimization.

## Papers Analyzed This Iteration (3 new, 139 total)
1. **"A Step-by-Step Guide to Creating a Robust Autonomous Drone Testing Pipeline"** (2025)
2. **"What Matters in Learning Zero-Shot Sim-to-Real RL Policy (SimpleFlight)"** (2024)
3. **"The Reality Gap in Robotics: Challenges, Solutions, and Best Practices"** (UZH RPG, 2025)

## Research Consensus

### 1. Staged Testing is Non-Negotiable
All three papers agree: SIL → HIL → Controlled Real → Field is the standard pipeline.
Our system is currently at Stage 1 (SIL with kinematic sim). For VQ1, the remote
qualification phase uses a virtual environment (Stage 1 equivalent), so our testing
level is sufficient for the immediate milestone.

### 2. Deterministic Seeding is a Best Practice
SimpleFlight and the testing pipeline paper both use fixed random seeds for reproducibility.
Our benchmark uses `np.random.normal()` without a seed — although the noise amplitude
(σ=5mm) is too small to affect results in practice, adding a fixed seed ensures
reproducibility even if noise parameters change in the future.

### 3. Key Sim-to-Real Gaps for Drone Racing (UZH RPG)
- **Dynamics**: Motor latency (20-80ms), thrust-to-weight uncertainty (±10-25%), drag (±50%)
- **Sensing**: IMU vibration aliasing, VIO latency (50-200ms), camera motion blur
- **Environment**: Wind (0-3 m/s), lighting variation
- **Timing**: Controller execution jitter, inference latency

### 4. Our System's Robustness Profile
From the UZH RPG analysis, our primary sim-to-real risks are:
- **Motor latency**: Our kinematic sim has zero actuator delay. Real Crazyflie has ~20ms.
- **State estimation latency**: Our EKF updates instantly. Real VIO adds 50-200ms.
- **Wind**: Our sim has no wind. Competition will have indoor airflow.
- **Thrust uncertainty**: Our sim uses idealized force model.

## Actionable Items for This Iteration

### A. Add Deterministic Random Seed (Low Risk, High Value)
Set `np.random.seed(42)` at benchmark start for guaranteed reproducibility.
This is a competition deployment best practice cited by SimpleFlight and the
testing pipeline paper. Zero regression risk.

### B. Add Race Time Threshold to Competition Metrics
Our benchmark doesn't explicitly check race_time_s against max_total_time_s in the
threshold_failures output. Adding this makes the benchmark more complete.

### C. Run Multi-Trial Robustness Verification
Run 3 sequential benchmark runs and verify bit-identical results. Document this
as competition readiness evidence.

## What NOT to Do
- Do NOT attempt further ILC/trajectory tuning — diminishing returns confirmed
- Do NOT add noise injection to the main benchmark — it changes the optimization target
- Do NOT attempt controller upgrades — too risky for the final iteration
- Do NOT modify the racing line or offsets — basin switching risk

## Cross-Validation Assessment
The research strongly supports our iteration 49 diagnosis: the system is at its
performance ceiling under the current kinematic sim + PD controller architecture.
The next performance tier requires either MPC/MPCC (Krinner 2024) or real hardware
deployment with system identification (UZH RPG best practice).

For this final iteration, the safest high-value action is adding deterministic
seeding and comprehensive documentation.
