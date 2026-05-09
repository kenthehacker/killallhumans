# Improving Drone Racing Performance Through Iterative Learning MPC

- **URL**: https://arxiv.org/abs/2508.01103
- **Authors**: Haocheng Zhao, Niklas Schlüter, Lukas Brunke, Angela P. Schoellig
- **Year**: 2025 (IROS 2025 oral)

## Key Contribution

This paper presents three targeted enhancements to Learning Model Predictive Control (LMPC) that together dramatically improve lap times in autonomous drone racing, both in simulation and on physical hardware. Vanilla LMPC builds a "safe set" from previous lap trajectories and uses it as terminal constraint, iteratively improving toward time-optimal behavior. However, naive LMPC suffers from excessive path shortcutting (missing gates), singularities in Frenet-frame formulations, and cost functions that push too hard toward time-optimality at the expense of safety.

The three innovations — an adaptive cost function, a shifted local safe set, and a Cartesian arc-length parametrization — together allow LMPC to achieve up to 60.85% lap time improvement from a naive PID baseline, and 6.05% improvement even when initialized from the already-aggressive MPCC++ controller. The method is validated on a Crazyflie 2.1 quadrotor with motion capture at 200 Hz and runs the MPC online at 30 Hz using acados.

## Technical Approach

### System Dynamics

The drone is modeled as a nonlinear control-affine system with state **x** ∈ ℝ⁹ = [position, velocity, Euler angles] and control input **u** ∈ ℝ⁴ = [collective thrust, commanded roll, pitch, yaw rates]. Rotational dynamics use first-order integrators with experimentally identified parameters: αϕ = -6.00, αθ = -3.96, βϕ = 6.21, βθ = 4.08.

### Arc-Length Parametrization (avoiding Frenet singularities)

Instead of Frenet-Serret frames (which have singularities when curvature approaches zero and accumulate integration error), the state vector is augmented with a scalar arc-length parameter s. The central path is a piecewise cubic Hermite interpolant through gate centers, with corridor radii that transition smoothly via sigmoid:

```
Rc(s) = Rc,gate + (Rc,max - Rc,gate) · σ(s - s_gate)
```

Arc-length estimation at each step solves:

```
sₖ = argmin_{s ∈ [s_prev - δ, s_prev + δ]} ‖pc(s) - pₖ‖
```

using L-BFGS-B on a k-d tree accelerated lookup. This reduces estimation time from 4.21 ± 0.28 ms (brute-force) to 0.68 ± 0.17 ms, making it negligible at 30 Hz control.

### Adaptive Cost Function

The stage cost is:

```
h(x, u) = l_t(u) + γ(s) · l_a(x)
```

where:
- `l_t(u) = c + ‖u‖²_R` penalizes control effort (promotes time-optimality via constant c)
- `l_a(x) = ‖(p - pc(s)) / Rc(s)‖²_Ψ` is normalized lateral deviation from centerline
- `γ(s)` is a mirrored sigmoid function that peaks near gates (to encourage accurate traversal) and is near zero between gates (to allow aggressive shortcuts)

This adaptive weight solves the tension between time-optimality and gate-passing accuracy. Pure time-optimal cost leads to gate misses at iteration 4; pure lateral cost barely improves lap time; the combined adaptive form converges reliably.

### Modified Local Safe Set

Standard LMPC uses a local safe set 𝒞𝒮ʲ = convex hull of K nearest neighbors from previous trajectories around the current arc-length s. The issue: as iterations progress, all past states cluster on the same (shortcutted) side of the track, causing the convex hull to degenerate and permitting no further exploration.

The fix is to augment the safe set by artificially shifting previous trajectory states to the *opposite* side of the centerline at reference arc-length s̄:

```
p̂ₖ = pₖ + 2 · (pc(s̄) - pₖ)_⊥
```

Shifted states are included in the safe set with a penalized cost-to-go:

```
Ĵₖ,ₜʲ = Jₖ,ₜʲ + ‖p̂ₖ - pₖ‖²_K
```

This injects diversity and prevents premature convergence. Ablation confirms this is essential: without shifted safe set, the combined adaptive cost still misses gates at iteration 6.

### MPC Formulation

At each step k, the following OCP is solved:

```
min  Σᵢ₌₀^{N-1} h(xₖ₊ᵢ|ₖ, uₖ₊ᵢ|ₖ) + Q̃^{j-1}(xₖ₊N|ₖ)
s.t. x_{i+1} = f(xᵢ, uᵢ)        (dynamics)
     x ∈ 𝕏, u ∈ 𝕌               (box constraints)
     x_{N} ∈ conv(K-NN of CS^j)  (terminal safe set)
     ‖p - pc(s)‖ ≤ Rc(s)         (corridor constraint)
```

Default hyperparameters: N = 8, K = 20, fd = 20 Hz (discretization), control at 30 Hz. Solved via acados v0.4.1 with SQP (max 5 iterations) and HPIPM QP backend (max 20 iterations), convergence tolerance 10⁻⁴.

### Computational Cost

| (N, K) config | Mean solve time |
|---------------|-----------------|
| (5, 20)       | 16.66 ± 2.28 ms |
| (8, 20)       | 33.26 ms        |
| (10, 20)      | 34.26 ± 5.18 ms |
| (15, 20)      | 72.24 ± 7.31 ms |

N = 8 fits within a 33 ms window (30 Hz) on an Intel i7-11700H. N ≥ 15 violates real-time.

## Results

### Simulation (Split-S track, 0.25× scaled)

| Baseline | Initial lap | Final LMPC lap | Improvement |
|----------|-------------|----------------|-------------|
| PID (0.5 m/s) | 23.55 s | 8.42 s | 64.25% |
| MPCC++ μ=0.02 | 11.84 s | 6.04 s | 48.99% |
| MPCC++ μ=0.10 | 7.71 s | 5.92 s | 23.22% |

### Real-World (Crazyflie 2.1, figure-eight track, 0.4 m square gates)

| Baseline | Initial lap | Final LMPC lap | Improvement |
|----------|-------------|----------------|-------------|
| PID | 17.09 s | 6.69 s | 60.85% |
| MPCC++ μ=0.02 | 10.79 s | 7.51 s | 30.40% |
| MPCC++ μ=0.10 | 6.45 s | 6.06 s | 6.05% |

All results averaged over 8 trials. Yaw held constant throughout.

### Ablation (Split-S, K=20)

| Configuration | Lap time | Gate misses |
|---------------|----------|-------------|
| Time-optimal cost only | 7.31 s | Yes (iter 4, gates 4-5) |
| Lateral deviation only | ~23 s | No (minimal improvement) |
| Combined + no shifted set | 9.64 s | Yes (iter 6) |
| Combined + shifted set | 8.47 s | None — reliable convergence |

## Relevance to Our System

Our system currently uses min-snap polynomial trajectories with L-BFGS optimization plus per-section ILC with Butterworth Q-filter. Our bottleneck is helix gate tracking (gate-7 at 0.284 m, avg 0.185 m). This paper is highly relevant in two ways:

**1. The LMPC terminal cost formulation is a direct template for our ILC improvement loop.** Our ILC already iterates over laps and corrects feedforward signals. The LMPC "safe set" concept maps onto our ILC memory: previous lap errors form the correction basis. The shifted safe set idea directly addresses our S-curve and helix sections where the drone tends to settle on one side of the gate passage corridor. We could inject a "mirrored correction" from a prior lap into the ILC feedforward candidate pool to avoid local convergence.

**2. The adaptive γ(s) cost is directly applicable to our per-gate error weighting.** Our current ILC applies uniform weighting across the section. Peaking γ near gate-7 (the helix) while relaxing it on straight sections would focus correction bandwidth on the high-curvature problem. The sigmoid transition profile can be parameterized by arc-length from each gate center.

**3. Arc-length parametrization with k-d tree** could replace our current time-parametric tracking in `mpc_tracker.py`. The 0.68 ms arc-length lookup is fast enough to run inside our 10 ms control loop budget.

**4. The corridor constraint formulation** (circular cross-sections with sigmoid-varying radius) maps directly onto our gate tube representation and could be used to tighten the feasible corridor on helix segments where we currently allow too much lateral freedom.

The acados SQP backend (N=8, 33 ms) is slightly too slow for our 100 Hz loop target, but N=5 at 16 ms could work if we maintain 60 Hz.

## Actionable Takeaways

1. **Implement adaptive ILC gain γ(s)**: Modify the per-section Q-filter in `planning/racing_line.py` so that the Butterworth cutoff is tighter (more correction accepted) near gate-7 and looser on straights. Use a sigmoid profile centered on the gate arc-length.

2. **Add mirrored feedforward candidates**: In our ILC update step, for each gate neighborhood compute the mirror-image correction (flip sign of lateral error) and add it to the candidate pool with a cost penalty proportional to distance from current trajectory. This directly implements the shifted safe set idea.

3. **Switch from time-parametric to arc-length parametric tracking**: Implement the L-BFGS-B arc-length projection with k-d tree acceleration in `control/mpc_tracker.py`. This eliminates the "running ahead/behind" desync that inflates tracking error at variable-speed sections like the helix.

4. **Tighten corridor radius near gates**: In `planning/trajectory_optimizer.py`, parameterize the corridor constraint radius as `Rc(s)` with sigmoid transitions — narrower at gates (e.g., Rc,gate = 0.3 m) and wider between gates (Rc,max = 1.5 m). This forces the optimizer to commit to accurate gate passage.

5. **Use acados N=5 config for online re-planning**: If we move to MPC-based re-planning, use N=5 with HPIPM to stay within 17 ms per solve, giving headroom in a 100 Hz loop.

6. **Tune shifted-set penalty K**: Start with K proportional to gate clearance distance (e.g., K = 1/Rc,gate²) so tight gates get strong centering correction, loose sections get weak correction.

## Limitations & Caveats

- **Crazyflie platform**: Results are on a miniature 27 g quadrotor at low speed. Our competition drone is larger with different actuator dynamics. The identified αϕ, αθ, βϕ, βθ parameters will need re-identification.

- **Constant yaw assumption**: The paper holds yaw constant throughout all experiments. Our helix section requires active yaw tracking for gate passage. The arc-length formulation would need extension to 3D curved paths with yaw components.

- **Track scale**: The Split-S track is scaled to 0.25×, figure-eight uses 0.4 m gates. Our gates may have different clearance margins, requiring re-tuning of Rc,gate and Rc,max.

- **30 Hz control rate**: Their outer loop runs at 30 Hz. We target 100+ Hz. The N=8 SQP config at 33 ms is not feasible for us. Must use N=5 or find a lighter QP backend.

- **Online computation**: The method requires online MPC solve at each timestep. Our current pipeline is purely feedforward (ILC). Switching to online MPC adds latency risk if the solver occasionally takes longer than the worst-case 2× mean time.

- **No wind/external disturbance testing**: The paper does not evaluate robustness to wind or model mismatch. Our competition environment may have drafts from gate structures.

## Key Parameters / Constants

| Parameter | Value | Usage |
|-----------|-------|-------|
| Prediction horizon N | 8 (sim), 5 (real-time budget) | MPC window |
| Safe set neighbors K | 20 | Terminal constraint diversity |
| Discretization frequency fd | 20 Hz | Dynamics model |
| Control frequency | 30 Hz | Outer loop |
| Arc-length search window δ | ~0.5 m (implied) | L-BFGS-B bracket |
| k-d tree arc-length lookup | 0.68 ± 0.17 ms | Negligible overhead |
| SQP iterations | 5 max | acados solver |
| HPIPM QP iterations | 20 max | Inner QP |
| Convergence tolerance | 10⁻⁴ | Solver stopping criterion |
| Rotational model αϕ | -6.00 | Roll dynamics (Crazyflie) |
| Rotational model αθ | -3.96 | Pitch dynamics (Crazyflie) |
| Rotational model βϕ | 6.21 | Roll input gain |
| Rotational model βθ | 4.08 | Pitch input gain |
| N=5 mean solve time | 16.66 ± 2.28 ms | Fits 60 Hz |
| N=8 mean solve time | 33.26 ms | Fits 30 Hz |
