# Multi-Fidelity Reinforcement Learning for Time-Optimal Quadrotor Re-planning

- **URL**: https://arxiv.org/abs/2403.08152
- **Authors**: Gilhyun Ryou, Geoffrey Wang, Sertac Karaman (MIT)
- **Year**: 2024 (accepted IJRR)
- **Repository**: https://github.com/mit-aera/mfrlTrajectory

---

## Key Contribution

This paper addresses online trajectory re-planning for quadrotors in environments where the planned path must be modified mid-flight due to waypoint perturbations (e.g., moving obstacles, updated gate positions, GPS drift). The central insight is that time-optimal trajectory planning is computationally intractable in real time at high fidelity, but can be decomposed into (a) a fast learned policy that outputs time allocation ratios and smoothness weights, and (b) a deterministic inner-loop solver (minimum-snap QP) that converts these ratios to polynomial coefficients in microseconds.

The multi-fidelity aspect addresses a second problem: training a policy that generalizes to real hardware requires flight data, but collecting enough real-world trajectories is expensive and slow. The paper trains a reward estimator using Gaussian process correlation across three fidelity levels — analytical dynamics, high-fidelity simulation (FlightGoggles), and real flights — so each real-world data point is worth many simulated ones.

The end result: trajectory re-planning in **2 ms** versus several minutes for the minimum-snap baseline at equivalent quality, with 4.7% reduction in total trajectory time and stable tracking error of 0.13 ± 0.14 m under waypoint deviations up to 3 m.

---

## Technical Approach

### Problem Decomposition: Inner Loop vs. Outer Loop

The core technical insight is a two-level decomposition of the trajectory optimization:

**Inner loop (closed-form, ~microseconds):** Given fixed segment times `T_1, ..., T_m` and waypoints `p`, solve for polynomial coefficients by QP minimizing weighted snap:

```
minimize  sum_{i=1}^{m} x_{w,i} * integral( mu_r * ||p_r^(4)||^2 + mu_psi * (p_psi^(2))^2 ) dt
```

This is a standard minimum-snap formulation. With fixed time allocations, the coefficient solve is a banded linear system — extremely fast (~microseconds).

**Outer loop (learned policy, ~2 ms):** The policy outputs two things:
1. **Time allocation ratios** `x_t in R^m` — relative segment durations
2. **Smoothness weights** `x_w in R^m` — per-segment snap penalty weights (via softmax)

The policy operates in a latent space with GRU encoder (256 hidden dims) and VAE bottleneck (64 dims). Outputs are decoded from this latent vector and transformed:

```
x_t = T_avg * exp(x_tilde_t) / dim(x_tilde_t)   # absolute segment times
x_w = softmax(x_tilde_w)                          # normalized per-segment weights
```

The exponential transformation ensures positive segment times; the softmax ensures weights sum to 1.

### Time Allocation Scaling and the Line Search for Feasibility

This is the most directly applicable part to our system. After the outer-loop policy outputs time allocation ratios `x_tilde`, the paper performs a **uniform scaling line search** to find the minimum total time that still satisfies feasibility:

**Step 1**: Outer loop finds normalized time ratios `x_tilde^MS` (the "shape" of time allocation across segments — which segment gets more time relative to others).

**Step 2**: Inner loop solves for polynomial coefficients `chi(x_tilde, p)` given those ratios.

**Step 3**: Binary line search over a scalar `alpha > 0`:

```
minimize_{alpha > 0}  alpha
subject to  chi(alpha * x_tilde, p) in P_feasible
```

where `P_feasible` is the set of trajectories satisfying all fidelity constraints. The scalar `alpha` uniformly scales all segment times, which is key: **uniform time scaling preserves the spatial shape of the trajectory while shifting motor commands away from their stationary hover commands**. Faster = more thrust required; if `alpha` is too small, motors saturate.

The binary search runs **10 evaluations** to find the minimum feasible `alpha`. At Level 1 (analytical dynamics), each evaluation costs 75 ms, so the full binary search takes ~750 ms — still much faster than global re-optimization.

### Three-Level Fidelity Hierarchy

**Level 1 — Ideal dynamics**: Check that reference motor commands stay within bounds `[omega_min, omega_max]^4` where `omega_max = 2200 rad/s`. This is purely analytical — evaluate the trajectory's thrust/torque demands through differential flatness.

**Level 2 — Simulation (FlightGoggles)**: Execute the trajectory in sim with full attitude controller. Feasibility = position tracking error ≤ 20 cm AND yaw error ≤ 15 degrees throughout the trajectory.

**Level 3 — Real world**: Same thresholds as Level 2, verified via motion capture. Each real evaluation costs ~2 minutes (flight + setup).

The key insight: feasibility at Level 1 is necessary but not sufficient. A trajectory can satisfy motor limits analytically but still fail in sim due to attitude transients. The MFGPC (Multi-Fidelity Gaussian Process Classifier) correlates these three levels so high-fidelity failures provide gradients that update Level 1 predictions.

### Training Protocol

- **Pretraining**: The policy is pretrained on a minimum-snap dataset (10^5 sequences, 5-14 waypoints) to predict time allocations matching the existing optimizer. This gives a warm start.
- **RL fine-tuning**: PPO with clipping epsilon=0.2, discount gamma=0.9. Reward = trajectory time (primary) + feasibility bonus (secondary).
- **MFGPC reward model**: Gaussian process with 128 inducing points, regularization 1e-4. Correlates Level 1/2/3 observations to predict feasibility without running high-fidelity evaluation on every candidate.
- **Training duration**: 1800 epochs, ~3 weeks total (dominated by the 1800 real-world trajectories collected across training).

### Architectural Details

- GRU encoder: 256 hidden dimensions
- VAE bottleneck: 64-dimensional embedding
- Optimizer: Adam, learning rate 1e-4
- Smoothness weights: mu_r=1, mu_psi=1 (equal position and yaw snap)
- Waypoint domain: rooms up to [20m, 20m, 4m], distances 0-30m, curvature 5-20 (Menger)

---

## Results

| Metric | Value |
|--------|-------|
| Inference time | 2 ms |
| Baseline (min-snap) time | several minutes |
| Trajectory time reduction (0m deviation) | 4.70% |
| Trajectory time reduction (1m/15 deg dev) | 4.50% |
| Trajectory time reduction (2m/30 deg dev) | 3.51% |
| Trajectory time reduction (3m/45 deg dev) | 2.49% |
| Tracking error, MFRL (3m dev) | 0.13 ± 0.14 m |
| Tracking error, baseline (3m dev) | 0.36 ± 3.09 m |

The baseline minimum-snap method catastrophically degrades under waypoint perturbation (±3.09 m std dev vs ±0.14 m for MFRL). This is because minimum-snap re-optimization from scratch can settle into very different local minima when the waypoint geometry changes, while the learned policy produces smooth, consistent time allocations.

---

## Relevance to Our System

Our system currently uses L-BFGS optimization over segment times for min-snap polynomials. Current state:
- Race time: 17.70s (regressed from 13.31s in iter 13 due to smoothness tradeoff)
- Avg tracking error: 0.179m (improved from 0.358m)
- Target: ~14-15s race time with tracking error ≤ 0.25m

### Where MFRL's Time Allocation Approach Applies Directly

**The line search for feasibility is directly applicable.** Our L-BFGS optimizer currently tries to minimize total time subject to per-axis velocity/acceleration constraints. The MFRL paper's approach suggests an alternative structure:

1. Run L-BFGS to find optimal **time allocation ratios** (which segment gets more/less time relative to others). This determines the "shape" of the trajectory.
2. Separately, binary search for the minimum **global time scale** `alpha` that makes the trajectory feasible for our kinematic sim + PD controller.

This two-phase structure would let us decouple the "shape" optimization (where the smoothness penalty matters) from the "speed" optimization (where we just scale alpha down until we crash or succeed).

**The key formula for our use:**
```python
# After L-BFGS finds normalized time ratios x_norm = segment_times / sum(segment_times)
# Binary search for minimum feasible total time T_total:
alpha_lo, alpha_hi = T_min, T_max
for _ in range(10):
    alpha = (alpha_lo + alpha_hi) / 2
    segment_times = alpha * x_norm
    if is_feasible(segment_times):  # run sim or check motor bounds
        alpha_hi = alpha
    else:
        alpha_lo = alpha
```

This directly addresses our race time regression: we found a good "shape" (smooth local minimum with 0.179m error) but the total time is too conservative at 17.70s. The binary search would let us find how much we can compress time before tracking error exceeds 0.25m.

### The Local Minimum Problem

Our iteration 13 diagnostic identified two basins:
- Fast basin: 12.78s, tracking error too high (>0.5m at helix)
- Smooth basin: 17.70s, tracking error 0.179m

The MFRL paper's approach suggests that the basin problem may be solved by fixing the **ratio** (shape) and searching over **scale** (speed) separately. The smooth basin has better shape; we just need to find how fast we can fly that shape. This is exactly what the line search does.

### Practical Adaptation Without RL

We do not need to implement the full RL + MFGPC system. The directly useful primitive is:

1. Keep our L-BFGS optimizer to find the smooth time allocation ratios (current behavior, iteration 13)
2. Add a post-optimization binary search over total trajectory time `T_total`, holding ratios fixed
3. Feasibility oracle: run `_check_dynamics_feasibility()` (our existing motor-bound check) OR better, define feasibility as "tracking error < threshold when simulated"

This would compress 17.70s toward the 14-15s target while staying in the smooth basin.

---

## Actionable Takeaways

1. **Implement two-phase time optimization**: Separate the "shape" (ratio) from the "speed" (scale). Our L-BFGS currently tries to do both simultaneously, which causes local minimum sensitivity. Run L-BFGS to convergence for ratios, then binary search for minimum feasible scale.

2. **Binary line search for minimum alpha**: Run 10-15 binary search steps over a scalar `alpha` multiplying all segment times simultaneously. This is O(log(N)) simulation evaluations for N precision levels. At 20Hz simulation, 20s trajectory, each feasibility check takes ~1s wall clock.

3. **Feasibility oracle options** (in order of fidelity):
   - Level 1: Check motor thrust bounds via differential flatness (analytical, ~1ms). Already partially implemented in our `_check_dynamics_feasibility()`.
   - Level 2: Run kinematic sim for 5-10s and check if average tracking error < 0.25m. (~0.5s wall clock at our sim speed.)
   - Use Level 1 for coarse search, Level 2 for final verification (mimics their multi-fidelity approach without RL).

4. **Smoothness weights per segment**: Their `x_w = softmax(x_tilde_w)` approach (per-segment snap penalty weights) is a simple extension to min-snap that could help the S-turn (gate-3/4). Higher weight on the S-turn segment reduces snap there, which our uniform-weight optimizer cannot do. No RL needed — just extend L-BFGS to also optimize `x_w`.

5. **Do not implement the full MFRL/RL system**: The 3-week training time and requirement for real-world flight data makes this impractical for our competition timeline. The key primitives (decoupled ratio/scale optimization, line search for feasibility) can be extracted and implemented in our existing framework.

---

## Limitations & Caveats

1. **Training cost**: 3 weeks total, 1800 real-world trajectories. Completely impractical for our system. Only the structural insights (two-phase optimization) are relevant.

2. **Fixed-route assumption**: The MFRL system is designed for re-planning when waypoints move. Our gates are fixed before the race, so the online re-planning capability is not needed. However, the time-allocation structure applies equally to pre-race offline optimization.

3. **Modest time reduction**: 4.7% time reduction over minimum-snap is small. If our L-BFGS is already near-optimal in ratio space, the line search would give perhaps 10-20% compression (from 17.70s to 14-15s) — but this is pure conjecture. The actual gain depends on how conservatively our current optimizer sets the scale.

4. **Controller fidelity gap**: Their Level 2 feasibility threshold (20 cm tracking error) is tighter than our current avg error (179 mm). However, their controller (FlightGoggles with full attitude dynamics) is more capable than our kinematic PD controller. We may need to use a less aggressive feasibility threshold.

5. **MFGPC complexity**: The Gaussian process reward model is sophisticated. Without it, a naive binary search over alpha requires running simulation for each evaluation. At our current benchmark speed (7939 Hz control loop), a 20s trajectory runs in ~2.5ms sim time, so each feasibility check is fast enough for 15-20 binary search evaluations.

6. **Local minimum in ratio space**: The paper implicitly assumes L-BFGS or the policy finds a good ratio. If the ratio is poor (e.g., all time on short straight segments, none on tight turns), scaling alpha cannot fix it. This is exactly our basin problem. The line search only helps if we start in the smooth basin.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Motor speed limit | 2200 rad/s | Level 1 feasibility bound |
| Level 2 tracking threshold | 20 cm | Sim feasibility criterion |
| Level 2 yaw threshold | 15 degrees | Sim feasibility criterion |
| Binary search evaluations | 10 | Per line search pass |
| Level 1 eval time | 75.35 ms | Analytical feasibility check |
| Level 2 eval time | 2.30 s | Simulation feasibility check |
| Level 3 eval time | ~2 min | Real-world evaluation |
| GRU hidden dim | 256 | Policy network |
| VAE bottleneck | 64 | Latent space |
| MFGPC inducing points | 128 | GP approximation |
| PPO clip | 0.2 | RL training |
| Discount factor | 0.9 | RL training |
| Snap weights | mu_r=1, mu_psi=1 | Min-snap objective |
| Training epochs | 1800 | Full training |
| Real trajectories | 1800 | Across all training |
| Pretraining time scale | alpha_pretrain=0.9 | Warm-start compression |
| Waypoints per sequence | 5-14 | Training domain |
| Room size (large) | 20x20x4 m | Training domain |

---

## Summary for State Update

The most immediately actionable idea from this paper is the **decoupled ratio/scale optimization with binary line search for feasibility**. Our current L-BFGS optimizer conflates the shape of time allocation (which segment gets proportionally more time) with the absolute speed (total trajectory duration). Separating these and binary searching for the minimum feasible total time — starting from our current smooth-basin solution — could compress 17.70s toward 14-15s without touching the ratio shape that gives us 0.179m tracking error. Implementation is ~50 lines of Python, no RL required.
