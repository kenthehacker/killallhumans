# Rethinking Reference Trajectories in Agile Drone Racing: A Unified Reference-Free Model-Based Controller via MPPI

- **URL**: https://arxiv.org/abs/2509.14726
- **Authors**: Fangguo Zhao, Xin Guan, Shuo Li
- **Year**: 2025 (submitted September 18, 2025)
- **Venue**: arXiv preprint (cs.RO)

---

## Key Contribution

This paper directly challenges the dominant paradigm in agile drone racing: the requirement for a pre-computed reference trajectory. In virtually every state-of-the-art pipeline — including our own — the controller is designed to track a reference path generated offline. This reference creates a cascaded dependency: the quality of the trajectory planner determines an upper bound on achievable performance, and any mismatch between the planned path and reality (due to disturbances, estimation error, or model mismatch) compounds into tracking error.

The authors propose a unified framework using Model Predictive Path Integral (MPPI) control with a novel "gate progress objective" that sidesteps this dependency entirely. Instead of minimizing deviation from a reference path, the controller directly rewards progression through sequential gates — the actual task objective. This makes the approach genuinely reference-free: no offline trajectory optimization is required. The paper also serves as a fair empirical benchmark, implementing three distinct objective functions (trajectory tracking, contouring control, and gate progress) within an identical MPPI framework with consistent dynamics and parameters, enabling apples-to-apples comparison without the confounds of different dynamics models or solvers.

---

## Technical Approach

### MPPI Background

MPPI (Model Predictive Path Integral) is a sampling-based stochastic optimal control method. Rather than computing a gradient of the cost with respect to control inputs, MPPI rolls forward M stochastic perturbations of the current control sequence, evaluates the cost of each rollout, and computes an information-theoretically optimal update as a weighted average:

```
U* = Σ_m w_m * U^m
w_m = exp(-J(U^m) / λ) / Σ_i exp(-J(U^i) / λ)
```

Here λ is the temperature parameter controlling how sharply the weighting concentrates on low-cost samples. A low λ (0.01 in this paper) produces near-greedy behavior, weighting the best samples most heavily.

The key strength of MPPI is that the cost function J need not be differentiable. This allows encoding discontinuous objectives like "did the drone cross through a gate?" — which is geometrically sharp and non-smooth.

### Quadrotor Dynamics Model

The state vector is x = [position, velocity, quaternion, angular velocity], a 13-dimensional state. Control input is u = Ṫ = [Ṫ₁, Ṫ₂, Ṫ₃, Ṫ₄], the time-derivatives of individual rotor thrusts. This actuator model naturally incorporates motor slew-rate limits and produces smoother thrust commands compared to controlling thrust magnitudes directly. The model incorporates aerodynamic drag effects.

### Gate Progress Objective

The core innovation is a cost function designed to directly maximize progress toward sequential racing gates:

```
J_gate = Σ_k [ ||p_{k+1} - p_gate||² - ||p_k - p_gate||² ] + Q_near * ||p_k - p_gate||²
```

The first term rewards any step that reduces distance to the next active gate. The second term, weighted by Q_near, adds a direct proximity reward. Gate switching (transitioning from gate i to gate i+1) is handled by a temporally consistent geometric check: a gate pass is registered only when the drone crosses the gate plane while the previous timestep's position was on the approach side. This prevents spurious gate registrations from trajectory roll-outs that "cut through" a gate from the wrong direction.

This construction avoids the need for a smooth reference entirely. The controller has no memory of a planned path — only knowledge of gate positions.

### Comparison Objectives

For fair benchmarking, the paper also implements:

1. **Trajectory Tracking**: Standard quadratic penalty J = ||x - x_ref||²_Q + ||u||²_R, where x_ref comes from a pre-computed time-parameterized trajectory. This is the closest analog to our current system.

2. **Contouring Control (MPCC-style)**: Decomposes error into contouring error (perpendicular to the path) and lag error (along-path position error), with the path parameterized by arc length. A virtual progress state θ is optimized jointly with control inputs.

### Implementation Parameters

The paper reports specific MPPI configuration values that were validated experimentally:

- Sample count: M = 8192 (simulation/desktop), M = 2048 (embedded Jetson)
- Prediction horizon: K = 20 steps
- Time step: Δt = 0.03 s (yielding a 0.6 s lookahead)
- Temperature: λ = 0.01
- Hardware: NVIDIA RTX 3090 (desktop), Jetson Orin NX (embedded)

---

## Results

### Simulation Results (Three Test Tracks)

**Circle Track:**
- Gate progress: 7.47 s flight time
- Optimal reference tracking: 7.29 s
- Waypoint error: 0.429 m for gate progress

**Figure-8 Track:**
- Gate progress: 11.16 s
- Contouring controller (gradient-based): 10.26 s (faster but requires reference)

**Split-S Track:**
- Gate progress: 17.16 s
- Tracking RMSE: 0.24 ± 0.14 m

The gate progress objective consistently showed the most aggressive thrust saturation behavior across all tracks, demonstrating that it is driving the system harder against actuator limits — behavior consistent with genuine time-optimality.

### Real-World Validation

On a circular track with a 340 g quadrotor (thrust-to-weight ratio = 3):
- Gate progress flight time: **6.56 s** (vs. 6.33 s theoretical optimum — only 3.6% suboptimal)
- Tracking RMSE: **0.46 ± 0.18 m**
- Gate-passing success rate: superior to reference-based methods

### Compute Performance

| Platform | MPPI Solve Time | Gradient Solver |
|---|---|---|
| RTX 3090 (desktop) | 0.4 ms | 3–4 ms |
| Jetson Orin NX (embedded) | 6.7 ms | 3–4 ms |

MPPI matches or beats gradient-based solvers on GPU-accelerated hardware but is disadvantaged on CPU-only or limited-GPU embedded platforms. At 6.7 ms on the Jetson, the control loop rate would be capped at ~149 Hz — acceptable for our 100 Hz target, but with little margin.

---

## Relevance to Our System

This paper is highly relevant to our specific problem: non-deterministic basin selection in the racing line optimizer. Our current architecture pre-computes a racing line (`racing_line.py`) via lateral offset optimization, then runs trajectory optimization (`trajectory_optimizer.py`) against that line, and the `_select_by_sim()` method introduces stochasticity by running full kinematic simulations on multiple candidates. The racing line acts as the reference that everything downstream depends on.

The MPPI gate progress approach offers a radically different architecture that would **eliminate the racing line entirely** from the control loop. Instead of:

```
racing_line → trajectory_optimizer → MPC tracker
```

We would have:

```
gate positions → MPPI (gate progress objective) → thrust commands
```

This directly addresses the root cause of basin non-determinism: there is no reference trajectory to have a basin around. The only inputs are gate positions (which are fixed) and current state, so the controller is deterministic in terms of goal specification even if the rollouts are stochastic internally.

The most directly relevant module is `control/mpc_tracker.py` (our geometric SE(3) tracker), which would be replaced or augmented by an MPPI controller. The `planning/racing_line.py` and `planning/trajectory_optimizer.py` modules could potentially be eliminated or kept only for pre-flight analysis.

The approach also bears on `race_pipeline.py`: the top-level orchestrator would simplify considerably if no offline planning phase is needed.

---

## Actionable Takeaways

1. **Replace `_select_by_sim()` with gate-progress scoring**: Rather than running full kinematic sims to select among racing line candidates, score each candidate by a gate-progress proxy: the sum of gate crossing times predicted by a fast kinematic model. This is deterministic and much cheaper than a full sim.

2. **Freeze the racing line, decouple from trajectory optimizer**: If we keep the reference-based architecture, the key insight is to separate the `_select_by_sim()` basin selection from the trajectory optimizer. Pre-select the racing line using a cheap deterministic criterion (e.g., minimum integrated curvature, or fixed lateral offset = 0), then run trajectory optimization once on the frozen line. This eliminates the source of non-determinism.

3. **Prototype MPPI gate-progress controller**: Implement the gate progress objective on top of our existing MPPI infrastructure (or build a simple Python MPPI against our PyBullet adapter). Use M=2048 samples, K=20 steps, Δt=0.03 s, λ=0.01 as starting parameters.

4. **Encode gate-crossing as a hard constraint, not soft**: The paper's temporally-consistent gate check (registering a pass only when the previous state was on the approach side) is directly applicable to our `gate_sequencing/sequencer.py`. This prevents the controller from "cheating" by flying through gates from the wrong direction in simulation rollouts.

5. **Use rotor thrust derivative as control input**: Our current `mpc_tracker.py` uses collective thrust + attitude. Switching to thrust-derivative inputs (u = Ṫ) would give smoother commands and natural motor slew-rate limiting — this is implementable in our current SE(3) framework with a minor reformulation.

6. **Benchmark on embedded hardware early**: The 6.7 ms solve time on Jetson Orin NX is close to our 10 ms budget for 100 Hz. Validate MPPI compute times in our PyBullet environment before committing to this architecture.

---

## Limitations & Caveats

**Near-optimal, not optimal**: The gate progress results are consistently 2–8% slower than reference-based trajectory tracking on the same tracks. For competition settings where lap time differences are small, this gap matters.

**GPU requirement**: The 0.4 ms solve time on RTX 3090 is excellent, but competition hardware will be more constrained. The Jetson Orin NX result (6.7 ms) is the more realistic benchmark for our use case.

**No perception integration**: The paper assumes gate positions are known exactly. Our pipeline uses `gate_pnp.py` + `gate_tracker.py` to estimate gate poses with noise. MPPI's cost function would need to account for gate position uncertainty.

**Short horizon (0.6 s)**: With K=20 and Δt=0.03 s, the prediction horizon is only 600 ms. At competition speeds (10+ m/s), this covers only ~6 m lookahead, potentially insufficient for tight gate sequences.

**Tracks tested are simple**: Circle, figure-8, and split-S are relatively simple compared to competition multi-gate layouts. Generalization to DCL-style complex tracks is unproven.

**Real-world gap**: The 0.46 m tracking RMSE in real-world experiments is close to our current threshold of 0.5 m. The reference-based methods weren't tested on real hardware in this paper, making the real-world comparison incomplete.

---

## Key Parameters / Constants

- MPPI samples: M = 8192 (GPU), M = 2048 (embedded)
- Prediction horizon: K = 20 steps
- Time step: Δt = 0.03 s (30 Hz prediction grid, not same as control rate)
- Temperature: λ = 0.01
- Quadrotor mass: 340 g (validation platform)
- Thrust-to-weight ratio: 3
- Gate progress proximity weight: Q_near (specific value not reported; tune empirically)
- Real-world RMSE: 0.46 ± 0.18 m
- Compute time (GPU): 0.4 ms per MPPI solve
- Compute time (Jetson Orin NX): 6.7 ms per MPPI solve
