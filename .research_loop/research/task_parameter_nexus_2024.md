# Task-Parameter Nexus: Task-Specific Parameter Learning for Model-Based Control
- **URL**: https://arxiv.org/abs/2412.12448
- **Year**: 2024 (submitted Dec 17, 2024; revised Apr 9, 2025)
- **Authors**: Zhe Shen et al.

---

## Key Contribution

Task-Parameter Nexus (TPN) addresses a fundamental limitation in model-based control: a single fixed set of control parameters cannot be near-optimal across diverse trajectory types. A controller tuned for hovering (high D-gain for damping) performs poorly on aggressive high-curvature maneuvers, and vice versa. Manual re-tuning per task is intractable in real deployments, and existing auto-tuning methods (like DiffTune) require per-task optimization at runtime which is too slow.

TPN solves this with a lightweight neural network that takes a trajectory as input and outputs a full set of near-optimal control parameters at runtime — essentially learning the mapping `trajectory → optimal_params` offline so it can be applied instantly online. The network is trained on a dataset of `(trajectory, optimal_params)` pairs where the optimal params are computed via an extended "Batch-DiffTune" algorithm that ensures robustness to task variation, not just single-trajectory optimality.

The key insight is that trajectory characteristics (speed and curvature) are the dominant factors determining what control parameters work best. By spanning this 2D space systematically during training, TPN learns a generalizable mapping that transfers to unseen trajectory types.

---

## Technical Approach

### Problem Formulation

Given a quadrotor with geometric controller parameterized by `θ ∈ Θ` (12 parameters: 6 translational, 6 rotational — position P/D gains plus attitude P/D gains), find the mapping `φ: T → Θ` such that for any new tracking task `T`, `φ(T) ≈ θ*(T)` where `θ*(T)` is the task-optimal parameter set.

The controller is a standard geometric SE(3) controller (Lee et al.), with the tracking objective being RMSE of position error over the trajectory duration.

### Trajectory Bank Construction

The training distribution spans a 2D grid of:
- **Speed bins**: S = {1, 2, 3} m/s
- **Curvature bins**: C = {[0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8)} (in units of 1/m, via Menger curvature)

This yields 12 trajectory categories (3 speeds × 4 curvature ranges). For each category, 20 "parent" trajectories are generated. Waypoints are sampled to satisfy:

```
‖p_{a+1} - p_a‖ = v_i * Δt           (speed constraint)
1 / R(p_a, p_{a+1}, p_{a+2}) ∈ C_j   (Menger curvature constraint)
```

Each trajectory segment is 2 seconds long, generated via minimum-snap polynomial planning. This gives 20 parent × 5 pieces × 12 categories = 1,200 trajectory-parameter training pairs.

### Batch-DiffTune: Robust Expert Parameter Generation

Standard DiffTune optimizes parameters for a single trajectory instance — the resulting parameters can overfit to that specific waypoint sequence. Batch-DiffTune instead creates `C` child trajectories by perturbing parent waypoints within a ball of radius `r = 0.05m`, then jointly minimizes:

```
min_{θ ∈ Θ}  (1/C) * Σ_{k=1}^{C}  L_m(θ; T_k)
```

Gradient updates use the average gradient across children:

```
θ ← P_Θ(θ - α * (1/C) * Σ_{k=1}^{C} ∇_θ L_m(θ; T_k))
```

where `P_Θ` is a projection operator ensuring `θ ≥ 0.01` (stability constraint). The averaged gradient prevents overfitting to a specific trajectory instance, yielding "expert parameters" that are robust across the local neighborhood. Batch-DiffTune runs for 100 iterations with step size `α = 0.1`.

The key differentiation from vanilla DiffTune is that gradients flow through the closed-loop simulation dynamics (autodiff through the controller and integrator), making this a form of differentiable simulation-based optimization.

### TPN Neural Network Architecture

- **Input**: Trajectory represented as a 201×2 waypoint sequence (x, y positions), flattened to a 402-dimensional vector
- **Hidden layers**: [128, 64, 12] neurons with ReLU activations
- **Output**: 12 control parameters
- **Constraint layer**: RAYEN-based output layer enforces `θ ≥ 0.01` (all gains must be positive and non-negligible)

Training uses supervised learning with MSE loss:

```
MSE = Σ_{i,j,p,s}  ‖θ*(B_{p,s}^{i,j}) - φ(T_{p,s}^{i,j})‖²
```

Training details: 50 epochs, Adam optimizer, learning rate 0.001, batch size 32. Training loss dropped from 88.07 to 0.30; validation loss from 10.84 to 0.27, demonstrating good convergence without overfitting.

### Runtime Operation

At the start of each new task/trajectory, TPN takes the upcoming trajectory waypoints, runs a single forward pass through the small MLP, and outputs updated control parameters instantly. These parameters are then loaded into the geometric controller before execution begins. No online optimization is required.

---

## Results

### Trained Categories (Table 1 — In-Distribution Performance)

TPN achieves near-expert tracking performance across all 12 trained categories with at most 4.9% degradation versus expert Batch-DiffTune parameters:

| Category | Expert RMSE (m) | TPN RMSE (m) | Untrained RMSE (m) |
|---|---|---|---|
| S1C1 (slow, low curv) | 0.180 ± 0.035 | 0.181 ± 0.034 | ~0.22 |
| S1C5 (slow, high curv) | 0.182 ± 0.037 | 0.180 ± 0.036 | 0.215 ± 0.058 |
| Higher speed/curv | Degrades gradually | Near-expert | Significantly worse |

The "Untrained" baseline uses a single fixed default parameter set, confirming the need for task-specific parameters.

### Generalization to Unseen Categories (Table 2)

For out-of-distribution speeds (>3 m/s) and curvatures (0.8–1.2), TPN still outperforms the fixed-parameter baseline:
- TPN maintains RMSE advantage even when extrapolating beyond training range
- Performance degrades gracefully, not catastrophically

### Unseen Trajectory Parametrizations (Table 3)

The most critical test: entirely different trajectory shapes not in the training set (circular, lemniscate/figure-eight):

| Trajectory | TPN RMSE (m) | Untrained RMSE (m) |
|---|---|---|
| Circular(4) | 0.211 ± 0.056 | 0.861 ± 0.071 |
| Lemniscate(1) | 0.131 ± 0.025 | 0.143 ± 0.033 |

Circular trajectory shows a dramatic 4× improvement over fixed parameters. Lemniscate shows more modest gain (the fixed params happen to be reasonable for this shape). This confirms TPN generalizes to trajectory shapes, not just interpolation within the training curvature bins.

---

## Relevance to Our System

Our system uses a PD+feedforward controller with fixed gains `kp=6, kd=4, feedforward_accel=0.4` and a 50ms predictive lookahead. The current bottleneck is the helix turn at gate-7 with 0.659m error, while straight segments achieve 0.12–0.27m. This exactly mirrors the paper's core finding: a single parameter set cannot handle both low-curvature and high-curvature regimes.

**Direct Relevance:**

1. **D-gain reduction for high curvature**: The paper explicitly validates that aggressive/high-curvature trajectories need reduced D-gains. Our kd=4 is likely too high in helix turns — it introduces drag that fights against the centripetal acceleration the controller is trying to apply, causing the drone to lag behind the desired trajectory.

2. **P-gain increase for high curvature/speed**: Higher curvature requires stronger proportional correction to stay on the racing line. Our kp=6 may be insufficient during turns.

3. **Trajectory-aware scheduling, not reactive tuning**: TPN predicts parameters from the upcoming trajectory (feedforward scheduling), not from current error (reactive). This prevents overshoot artifacts from abrupt gain changes mid-flight.

4. **The 201-waypoint input representation**: Our trajectory optimizer already computes full polynomial trajectories. We can extract waypoint sequences around upcoming gates to serve as input to a TPN-like gain scheduler.

5. **Curvature as the key feature**: Menger curvature of the upcoming trajectory segment is the dominant predictor of optimal gains. We can compute this analytically from our min-snap polynomials.

The paper's result on circular trajectories (0.211m vs 0.861m with fixed gains) is extremely relevant — helix turns are essentially circular arcs, and our gate-7 error (0.659m) is in exactly the range where TPN's circular trajectory improvement would apply.

---

## Actionable Takeaways

1. **Implement curvature-based D-gain scheduling**: Compute Menger curvature along the upcoming trajectory segment (using the min-snap polynomial derivatives already available in `trajectory_optimizer.py`). Reduce `kd` proportionally to curvature: e.g., `kd_effective = kd_base / (1 + alpha * curvature)`. Start with `alpha ≈ 2.0` and tune via benchmark.

2. **Simultaneous P-gain increase for high-curvature segments**: Mirror the TPN finding that P-gains should increase for aggressive maneuvers. Apply `kp_effective = kp_base * (1 + beta * curvature)` with `beta ≈ 0.5`. This increases responsiveness without adding the velocity-drag that hurts in turns.

3. **Lookahead window for gain prediction**: Use the trajectory waypoints 50–200ms ahead (matching our existing 50ms lookahead). Compute average curvature and speed in this window to determine gain scaling. This is a simplified TPN that avoids training a neural network.

4. **Speed-dependent gain scaling**: The paper uses a 2D grid (speed × curvature). Add speed-dependent scaling: at higher speeds, reduce derivative gains further since the same positional error corresponds to larger relative velocity error.

5. **Gate-proximity gain scheduling**: Near gates (within 1–2m), where trajectory curvature is highest in our race course, switch to a high-curvature parameter set. Gate positions are known a priori from the race config, so this can be precomputed.

6. **If implementing full TPN**: Train on our specific race course trajectory segments. Use 2-second windows of the polynomial trajectory, compute Batch-DiffTune-style optimal gains per segment via grid search over (kp, kd) pairs, then fit a small MLP. The architecture is deliberately simple (128→64→12 neurons) and would add negligible inference overhead.

7. **Validate the feedforward term interaction**: The paper uses geometric SE(3) control, which has separate translational and rotational gains (12 parameters). Our feedforward_accel=0.4 interacts with kp/kd — when adjusting gains, re-tune feedforward_accel jointly to maintain the acceleration compensation balance.

8. **Trajectory bank construction for our domain**: Generate training segments by sampling 2-second windows from the race course polynomial. The helix gates (gate-7 area) will naturally fall in the high-curvature bins. Ensure the training set includes our specific curvature range (the helix radius determines peak Menger curvature).

---

## Limitations & Caveats

1. **Simulation only**: All results are in simulation (PyBullet). Real-world motor lag, aerodynamic effects, and battery voltage sag are not modeled. Transfer to hardware may require additional adaptation.

2. **2-second trajectory windows**: TPN uses fixed 2-second segments. For our variable-speed trajectory, shorter segments near gates and longer segments on straights would be more natural, but the fixed-length input may not capture gate approach dynamics well.

3. **No closed-loop guarantee**: TPN is trained via supervised learning on offline-computed optimal parameters. There is no stability proof for the learned parameter schedule. The RAYEN constraint (θ ≥ 0.01) only ensures non-negativity, not closed-loop stability.

4. **Generalization degrades for extreme curvature**: Table 2 shows TPN struggles when curvature exceeds training range (>0.8 1/m). If our helix has tighter curvature, TPN's gains may not be optimal. Our helix radius and the corresponding Menger curvature should be checked against the training distribution.

5. **12-parameter geometric controller**: TPN is designed for full SE(3) geometric control with 12 independent parameters. Our simpler PD+feedforward controller has 3 parameters (kp, kd, feedforward_accel). Direct application requires adaptation to our reduced parameter space.

6. **No temporal consistency constraint**: Gains change abruptly at segment boundaries. In our system, rapid gain changes could excite oscillations, especially if the transition occurs during a high-speed maneuver. A low-pass filter on the predicted gains is advisable.

7. **1,200 training pairs**: This is a small dataset by deep learning standards. The network works because the problem is low-dimensional (trajectory → 12 params) and the features are smooth. For our 3-parameter case, even fewer samples may suffice — or we can use a simple lookup table indexed by (speed_bin, curvature_bin).

---

## Key Parameters / Constants

| Parameter | Value | Description |
|---|---|---|
| Trajectory segment length | 2 seconds | Duration per training segment |
| Waypoints per segment | 201 | Input resolution (201×2 coordinates) |
| Curvature bins | [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8) | Training curvature ranges (1/m) |
| Speed bins | {1, 2, 3} m/s | Training speed values |
| Training categories | 12 | 3 speeds × 4 curvature bins |
| Parent trajectories per category | 20 | Training diversity |
| Batch-DiffTune children | C (not specified) | Perturbed task variants per parent |
| Waypoint perturbation radius | r = 0.05m | Perturbation ball radius for batch generation |
| Batch-DiffTune iterations | 100 | Optimization steps per trajectory |
| Batch-DiffTune step size | α = 0.1 | Gradient step |
| TPN hidden layers | [128, 64, 12] | MLP architecture |
| TPN training epochs | 50 | Supervised training |
| TPN learning rate | 0.001 | Adam optimizer LR |
| TPN batch size | 32 | Training batch size |
| Min gain constraint | θ ≥ 0.01 | RAYEN layer constraint |
| Circular trajectory improvement | 4× | vs fixed params (0.211m vs 0.861m RMSE) |
| Max in-distribution degradation | 4.9% | vs expert Batch-DiffTune params |
