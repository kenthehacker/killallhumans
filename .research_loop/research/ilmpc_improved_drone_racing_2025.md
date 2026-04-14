# Improving Drone Racing Performance Through Iterative Learning MPC
- **URL**: https://arxiv.org/abs/2508.01103
- **Authors**: Haocheng Zhao, Niklas Schlüter, Lukas Brunke, Angela P. Schoellig (Learning Systems and Robotics Lab, Technical University of Munich)
- **Year**: 2025
- **Venue**: arXiv (accepted for oral presentation at IROS 2025)

---

## Key Contribution

This paper presents an enhanced Iterative Learning Model Predictive Control (ILMPC) framework for autonomous drone racing. The core problem it addresses is that direct application of LMPC to racing — a minimum-time problem with gate-passing constraints — suffers from two failure modes: (1) excessive corner-cutting that causes gate misses in early iterations, and (2) a tendency to converge prematurely to suboptimal solutions when the safe set is too restrictive. The authors resolve both by proposing three targeted modifications to standard LMPC.

The practical outcome is striking: a drone starting from a naive PID controller (17.09s lap time) converges to 6.69s — a **60.85% improvement** — without any handcrafted trajectory design. Even when initialized from an aggressively-tuned MPCC++ controller (already near-optimal at 6.45s), the method still extracts a **6.05% improvement**, reaching 6.06s. This is meaningful because MPCC++ is itself a state-of-the-art racing controller, and closing the gap further with a generic learning framework is non-trivial.

---

## Technical Approach

### Problem Formulation

The racing task is cast as a minimum-time optimal control problem: navigate through N_g gates in order, satisfying gate-passing constraints and velocity bounds. The quadrotor state vector is:

```
x = [p, v, φ, θ, ψ]  (position, velocity, Euler angles)
u = [f_Σ, φ_cmd, θ_cmd, ψ_cmd]  (collective thrust, commanded angles)
```

Translational dynamics follow Newton's law; rotational dynamics are modeled as three independent first-order integrators (attitude as integrators of commanded angles, valid for inner-loop attitude-controlled drones).

The corridor safety constraint is a circular cross-section tube around the central path:

```
A_c(s) = { p ∈ ℝ³ : ||p - p_c(s)|| ≤ R_c(s),  (p - p_c(s))^T T(s) = 0 }
```

where s is arc-length along the path, p_c(s) is the path centerline, R_c(s) is a smoothly varying radius (tight near gates, relaxed in straights), and T(s) is the unit tangent. The corridor radius transitions between gate and straight-section values via sigmoid functions to avoid sharp constraint boundaries.

The central path p_c(s) is constructed via **piecewise cubic Hermite interpolation** through gate centers, satisfying gate positions and tangent directions simultaneously.

---

### Modification 1: Cartesian-Based Formulation with Arc-Length Parametrization

Standard LMPC for racing uses Frenet-frame coordinates (progress along track, lateral/normal offsets). This causes two problems: (1) the frame transformation becomes singular at high-curvature sections, and (2) accumulated arc-length integration drifts over time.

The paper instead augments the state with arc-length s as a dynamic variable:

```
ṡ = v^T · T(s)
```

where T(s) is the tangent vector at the current arc-length. The full augmented state is x̃ = [p, v, φ, θ, ψ, s]. This is **Cartesian-based**: position p is tracked in world coordinates; s is used only for parametrizing the corridor and cost, not for expressing position.

To avoid arc-length integration drift (accumulation of ṡ = v^T T(s) over long horizons), the authors use **feedback-based arc-length estimation** at each control step:

```
s_k = arg min_s ||p_c(s) - p_k||
```

This is solved efficiently by: (1) discretizing the path into bins, (2) nearest-neighbor search via a k-d tree to get an initial guess, (3) L-BFGS-B local optimization over a bounded interval around the guess. This reduces computation from 4.21 ± 0.28 ms to 0.68 ± 0.17 ms per call — a 6x speedup critical for real-time operation at 30 Hz.

---

### Modification 2: Adaptive Cost Function

Standard LMPC uses a time-optimal cost (penalize time elapsed and control effort). For drone racing this causes early iterations to cut corners aggressively, missing gates before the safe set is sufficiently populated.

The paper decomposes the stage cost into two components:

**Time-optimal component:**
```
l_t(u) = c + ||u||²_R
```
where c is a constant per-step time penalty and R penalizes control effort.

**Lateral deviation component:**
```
l_d(x) = ||[p - p_c(s)] / R_c(s)||²_{Q_d}
```
This penalizes Euclidean distance from the centerline, normalized by the corridor radius so the penalty is scale-invariant across different corridor widths.

**Combined adaptive cost:**
```
h(x, u) = l_t(u) + γ(s) · l_d(x)
```

The weighting function γ(s) uses **mirrored sigmoid functions**:
- γ(s) is HIGH near gates → strong centerline-following pressure through gate windows
- γ(s) is LOW in straights and near-straight sections → optimization free to find time-optimal shortcuts

The sigmoid transition width and peak value are hyperparameters. In early iterations, l_d dominates near gates (safe traversal); as the safe set matures, the time-optimal component increasingly drives behavior.

---

### Modification 3: Shifted Local Safe Set

LMPC builds a "safe set" from previously observed successful trajectories. The local safe set S_local is a subset of the full safe set near the current operating region. The standard approach takes the K nearest states to the current state as the terminal constraint.

**Problem:** When the drone consistently passes gates on one side of the centerline (due to aggressive optimization), the entire safe set is biased to that side. The terminal constraint then only allows the drone to stay on that side, preventing exploration of the other side and potentially locking into a suboptimal path.

**Solution — Shifted Safe Set:** For each state p_k in the local safe set:
1. Compute the average lateral deviation of the local set: Δp_avg
2. Create a "shifted" state p̂_k = p_k - 2·Δp_avg (mirror to opposite side of centerline)
3. Assign a penalized cost-to-go: Ĵ = J(p_k) + ||p̂_k - p_k||²_K

The penalized cost ensures the optimizer prefers real safe-set states but can fall back to shifted states when needed. This effectively gives the optimizer a synthetic "alternative path" on the other side, preventing premature convergence to one-sided trajectories.

Additionally, the shifted safe set addresses the **single-demonstration problem**: standard LMPC requires multiple varied initial demonstrations to populate the safe set with diverse states. By synthesizing opposite-side states from a single trajectory, the method achieves good coverage from one initialization.

---

### System Integration

The full pipeline at each control step (30 Hz):
1. Receive state from EKF (fused motion capture + onboard IMU)
2. Estimate current arc-length s via k-d tree + L-BFGS-B
3. Extract local safe set (K=20 nearest states in arc-length) + shifted augmentation
4. Solve LMPC QP (horizon N=8) via Acados/SQP
5. Apply first control input; record (x, u, J) for safe set update

**Platform:** Crazyflie 2.1 (small quadrotor, ~30g)
**Solver:** Acados v0.4.1 with SQP (Sequential Quadratic Programming)
**State Estimation:** External motion capture at 200 Hz, fused with EKF
**Control Frequency:** 30 Hz (MPC step), vs. 90 Hz for MPCC++ baseline

---

## Results

### Simulation (Split-S Track)

Ablation over cost components:

| Cost Configuration | Lap Time (s) | Gate Misses |
|---|---|---|
| Time-optimal only | 7.31 | Yes (after iter 2) |
| Lateral deviation only | 23.24 | No |
| Adaptive cost only | 9.64 | Occasional |
| Adaptive cost + shifted safe set | **8.47** | No |

The full method (adaptive + shifted) achieves the best time while maintaining 100% gate pass rate.

Hyperparameter study over prediction horizon N ∈ {5, 8, 10, 15} and safe set size K ∈ {10, 15, 20}: N=8, K=20 provides best tradeoff. Lower N (5) causes myopic behavior near gates; higher N (15) increases solver time (72ms average, exceeding the 33ms budget at 30 Hz).

### Real-World Experiments (Figure-Eight Track)

| Initial Controller | Initial Time (s) | Final Time (s) | Improvement |
|---|---|---|---|
| PID (v=0.5 m/s) | 17.09 ± 0.11 | 6.69 ± 0.22 | 60.85% |
| MPCC++ (μ=0.02) | 10.79 ± 0.27 | 7.51 ± 0.38 | 30.40% |
| MPCC++ (μ=0.05) | 7.62 ± 0.11 | 6.97 ± 0.17 | 8.53% |
| MPCC++ (μ=0.10) | 6.45 ± 0.12 | 6.06 ± 0.25 | 6.05% |

Simulation with MPCC++ (μ=0.02) initialization achieved 48.99% improvement (11.84s → 6.04s), showing the sim-to-real gap is modest.

Convergence typically occurs within **3–10 iterations** in both simulation and hardware.

---

## Relevance to Our System

Our system uses:
- **Trajectory planner**: min-snap polynomials + L-BFGS time optimization + TOPP-style retiming + turn inflation
- **Controller**: PD with feedforward in `mpc_tracker.py` (geometric SE(3) tracker)
- **Estimator**: EKF in `ekf.py`
- **Gate sequencing**: `sequencer.py` with pass-through margin detection
- **Current bottleneck**: helix gate accuracy while maintaining race speed; avg error ~0.248m, race time ~13.62s

**Direct relevance by module:**

1. **`trajectory_optimizer.py` / `racing_line.py`** — The adaptive cost function concept is the most directly applicable idea. Our current trajectory is pre-computed offline and fixed. The paper shows that dynamically weighting time-optimality vs. centerline adherence near gates (via sigmoid γ(s)) is the key to maintaining gate accuracy while not sacrificing speed on straights. Our turn inflation heuristic is a coarser version of this idea — the paper provides a principled, parameterized formulation.

2. **`mpc_tracker.py`** — Our PD tracker is the equivalent of the LMPC's initialized controller. The paper's key insight: even a well-tuned PD tracker can be improved 6% by ILMPC. However, implementing full ILMPC requires a QP solver (Acados/CasADi) which we don't currently have. The adaptive cost weighting idea could be extracted and applied to our trajectory generation offline.

3. **`racing_line.py`** (lateral offset optimization) — The shifted safe set concept maps directly to our lateral offset optimizer. Currently we optimize the racing line offline with L-BFGS. We could implement a multi-iteration approach where: (a) run sim, (b) observe which side of gates we miss, (c) shift the centerline laterally for those gates, (d) re-optimize. This is the spirit of the shifted safe set without requiring full LMPC infrastructure.

4. **Arc-length estimation** — Our trajectory parametrization already uses arc-length internally. The k-d tree + L-BFGS approach for fast projection is directly implementable in Python and could improve our TOPP retiming speed.

5. **Helix gate accuracy** — Gate 3 (helix) has avg error 0.326m in our current system. The adaptive γ(s) cost — increasing centerline pressure near gates — is the most targeted fix. We could implement this by increasing the curvature penalty or reducing the speed limit specifically within a radius of each gate center.

---

## Actionable Takeaways

1. **Implement gate-proximity adaptive speed limits**: Add a sigmoid-weighted speed cap in `racing_line.py` that reduces max speed within ~2m of each gate center. This is a direct translation of γ(s) without needing full LMPC infrastructure. The sigmoid could transition over ~1m approach distance.

2. **Iterative racing line refinement**: After each benchmark run, parse `per_gate_avg_error` and identify which side of each gate the drone drifts. Apply a lateral offset correction to the racing line for that gate (shift centerline toward the error-reducing direction), then re-run. This replicates the shifted safe set logic in our offline optimizer.

3. **Arc-length projection speedup**: Implement the k-d tree + bounded L-BFGS approach for arc-length estimation in `trajectory_optimizer.py`. This is particularly useful if TOPP retiming becomes a bottleneck. Expected speedup: ~6x vs. naive minimization.

4. **Gate corridor radius tapering**: Explicitly model a safety corridor with radius that tightens near gates (like R_c(s) in the paper). Use this to tighten trajectory constraints near gate centers in the min-snap optimizer, replacing our current constant-margin approach.

5. **Multi-iteration benchmarking loop with trajectory correction**: Formalize the iteration loop: run sim → observe per-gate errors → adjust racing line lateral offsets and speed profile → re-run. The paper shows 3-10 iterations typically suffice. Our current iteration protocol is manual; this should be scriptable.

6. **Adaptive feedforward near gates**: In `mpc_tracker.py`, increase the feedforward gain (or add a position-error correction term) specifically within a gate proximity window. This replicates the increased l_d weighting near gates without full LMPC — it's a gain-scheduled PD where kP increases near gates.

7. **If implementing full LMPC**: Use N=8 prediction horizon, K=20 safe set size, 30 Hz control frequency. These are validated hyperparameters on real hardware. Use Acados with SQP solver. The L-BFGS-B arc-length projection with k-d tree initialization is essential for real-time feasibility.

---

## Limitations & Caveats

1. **Requires a working initial controller**: ILMPC only improves upon existing successful trajectories. If the drone crashes or misses gates, the safe set cannot be populated. Our current system must already pass all gates before ILMPC would help.

2. **Motion capture dependency**: The real hardware experiments use a 200 Hz motion capture system for state estimation. Our competition setting uses onboard EKF + gate PnP. The EKF uncertainty in our system (~0.1–0.5m) is larger than the precision assumed by LMPC's safe set construction, which could degrade convergence.

3. **30 Hz control rate**: The paper runs at 30 Hz due to MPC solve time. Our current PD controller runs at >100 Hz. Switching to LMPC would reduce control frequency by 3x, which may hurt tracking performance on fast maneuvers. Our PD+feedforward at 100+ Hz may actually outperform a 30 Hz MPC for our specific track.

4. **Crazyflie-scale results**: The Crazyflie 2.1 is a micro-quadrotor (~30g) with different dynamics from competition drones (heavier, faster, larger). The specific gains, corridor radii, and convergence rates may not transfer directly to our drone model.

5. **Figure-eight track only**: Real-world experiments use a single figure-eight track. Transfer to tracks with 3D helices, split-S maneuvers, or rapidly varying gate orientations (like our track) is validated only in simulation.

6. **Single demonstration assumption**: The shifted safe set is designed to overcome the need for multiple diverse demonstrations. However, it introduces synthetic states that have never been visited, which could cause infeasibility in edge cases near constraint boundaries.

7. **No aerodynamic drag model**: The dynamics model treats rotational dynamics as independent first-order integrators. This is a simplification — in reality, aerodynamic drag and rotor interactions matter at high speed. Our TOPP retiming accounts for drag to some extent; LMPC as formulated here does not.

8. **Offline vs. online**: Our system pre-computes the trajectory offline and runs a tracking controller online. ILMPC is inherently an online, iterative method that improves lap-over-lap. For a single-run competition (no warm-up laps allowed), the iterative improvement may be irrelevant unless we can pre-compute offline via simulation rollouts.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|---|---|---|
| Prediction horizon N | 8 | MPC look-ahead steps (validated optimal in N ∈ {5,8,10,15}) |
| Safe set size K | 20 | Number of historical states in local safe set (validated in K ∈ {10,15,20}) |
| Control frequency | 30 Hz | MPC step rate on Crazyflie |
| State estimation rate | 200 Hz | Motion capture + EKF fusion rate |
| Arc-length bin count | Not specified | Used for k-d tree discretization; 0.68ms solve time achieved |
| Arc-length solve time | 0.68 ± 0.17 ms | After k-d tree + L-BFGS-B (vs. 4.21 ± 0.28 ms naive) |
| LMPC solver (Acados) | v0.4.1, SQP | Sequential quadratic programming |
| Avg solver time | 12–72 ms | Depending on N and K (N=8, K=20 → ~30ms typical) |
| Cost constant c | Not specified | Per-step time penalty in l_t(u) |
| γ(s) near gate | HIGH (not specified numerically) | Lateral deviation weight near gate centers |
| γ(s) in straights | LOW (not specified numerically) | Lateral deviation weight on straights |
| Corridor radius R_c(s) | Varies (not specified numerically) | Tight at gate centers, relaxed in straights |
| Sigmoid transition width | Not specified numerically | Controls γ(s) transition sharpness |
| Convergence iterations | 3–10 | Typical number of laps to reach near-optimal time |
| Improvement (best init) | 6.05% | MPCC++ μ=0.10, real hardware |
| Improvement (poor init) | 60.85% | PID baseline, real hardware |
| Platform | Crazyflie 2.1 | ~30g micro-quadrotor |
| Track type (sim) | Split-S | 3D aerobatic maneuver track |
| Track type (real) | Figure-eight | 2D planar figure-eight |

**Note on missing numerics**: The paper does not provide explicit values for γ(s) magnitudes, sigmoid widths, corridor radii R_c(s), or cost matrix entries (R, Q_d, K). These must be tuned empirically. The authors indicate these are dataset/track-specific.
