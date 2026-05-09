# Reduce Lap Time for Autonomous Racing with Curvature-Integrated MPCC

**URL:** https://arxiv.org/abs/2502.03695
**Authors:** Zhouheng Li, Lei Xie, Cheng Hu, Hongye Su
**Year:** 2025
**Venue:** arXiv preprint

---

## 1. Key Contribution

This paper introduces CiMPCC (Curvature-Integrated Model Predictive Contouring Control), a local trajectory planning method that embeds racetrack centerline curvature directly into the MPCC cost function. The core idea is straightforward: standard MPCC maximizes progress along the centerline via a linear reward term (gamma * v_p * T_s) but is agnostic to upcoming curvature, leading the vehicle to carry excessive speed into sharp turns or brake too conservatively in gentle sections. CiMPCC replaces this uniform speed incentive with a curvature-modulated velocity reference that interpolates between aggressive and conservative velocity bounds using an exponential mapping of local curvature. On a physical 1:10-scale F1TENTH platform, CiMPCC achieves 11.4--12.5% lap time reduction over baseline MPCC and the RDM+OTG global planning method, while maintaining real-time solve times under 20.6 ms per iteration.

---

## 2. Technical Approach

### 2.1 Baseline MPCC Recap

The vehicle state is zeta = [X, Y, phi, s] (position, heading, centerline progress) with controls u = [v_l, delta, v_p] (longitudinal velocity, steering angle, projected velocity along the path). The standard MPCC cost is:

```
J_MPCC = sum(||xi_k||^2_Q  -  gamma * v_p,k * T_s)  +  sum(||Delta_u_k||^2_R1)  +  sum(||u_k - u_ref||^2_R2)
```

where xi_k = [e_con, e_lag] are contour (lateral) and lag (longitudinal) errors relative to the centerline, Q penalizes deviation, and the gamma term rewards forward progress. The problem is that the progress reward is a constant multiplier regardless of geometry — no notion of "slow down for corners."

### 2.2 Curvature Extraction and Normalization

From discrete centerline waypoints, curvature kappa_i is computed via finite differences and then smoothed with a moving-average filter to remove noise. Min-max normalization maps the result to K^n in [0, 1].

### 2.3 Exponential Velocity Mapping

An exponential function converts normalized curvature to a blending coefficient:

```
g(K^n) = exp(-alpha * (K^n)^2),   alpha > 0
```

At zero curvature g = 1 (full aggressive speed); at maximum curvature g = exp(-alpha) > 0 (conservative but nonzero). The Gaussian-like shape means the transition is smooth and differentiable, which is critical for NLP solver convergence.

### 2.4 Curvature-Integrated Cost

CiMPCC adds a velocity-tracking term that interpolates between an upper velocity bound v_bar (aggressive) and a lower bound v_underline (safe):

```
J_Ci = sum[ (1 - beta) * ||v_k - v_underline||^2_R3  +  beta * ||v_k - v_bar||^2_R3 ]
```

where beta = g(K^n_cur). When curvature is low (beta near 1), the cost penalizes deviation from the aggressive bound; when curvature is high (beta near 0), it penalizes deviation from the conservative bound. The total cost is J_MPCC + J_Ci.

### 2.5 Weight Matrices

Reported tuning: Q = diag(800, 800), R1 = diag(10, 3500, 0), R2(CiMPCC) = diag(0, 10, 0), R3 = diag(40, 40), gamma = 40. Notably, CiMPCC zeros out the longitudinal velocity weight in R2 because velocity regulation is now handled entirely by J_Ci — this prevents conflicting objectives.

### 2.6 Solver

The NLP is solved with CasADi at a prediction/control horizon of N_p = N_c = 10 steps. Over 95% of iterations complete in under 20.6 ms.

---

## 3. Results

Experiments were conducted on a physical 1:10 F1TENTH car on a custom racetrack with varying curvature sections.

| Method    | Mean Lap Time (s) | Mean Velocity (m/s) | Lap Time Improvement |
|-----------|--------------------|---------------------|-----------------------|
| MPCC      | 16.106             | 2.910               | baseline              |
| RDM+OTG  | 16.083             | 2.864               | baseline              |
| CiMPCC    | 14.202             | 3.351               | 11.4--12.5%           |

CiMPCC increased mean velocity by 15.2% while reducing lap time by 11.8% versus baseline MPCC. The velocity profile shows clear deceleration in high-curvature segments and aggressive acceleration through straights, validating the curvature-speed coupling. No crashes or constraint violations were reported across 17 laps. However, no explicit tracking error or lateral deviation statistics are reported, which is a notable omission.

---

## 4. Relevance to Our System

Our drone racing pipeline (in `planning/racing_line.py`) selects the best racing line from a pool of L-BFGS candidates by running a lightweight kinematic simulation and scoring each candidate with:

```
score = 0.7 * avg_error + 0.3 * worst_gate_error
```

This composite is purely tracking-error-based: it has no incentive to prefer faster trajectories. A candidate that flies slowly and perfectly tracks has a better score than one that flies aggressively with marginally higher error but much lower race time. This is exactly the deficiency that CiMPCC addresses in the ground-vehicle domain.

**How CiMPCC informs our composite cost design:**

1. **Curvature-weighted velocity targets, not raw time.** CiMPCC does not simply minimize lap time — that would encourage reckless speed everywhere. Instead it defines segment-appropriate velocity targets and penalizes deviation from them. For our composite, this suggests we should not just add a raw `race_time` penalty. Instead, we should define expected velocities per segment (higher on straights between distant gates, lower near tight gate sequences) and penalize trajectories that are slower than their curvature-appropriate target.

2. **Interpolation between aggressive and conservative bounds.** The beta-blending idea translates directly: for each trajectory segment, compute a curvature-based blending coefficient and set the target speed as `beta * v_max + (1-beta) * v_safe`. The composite score can then include a term like `||v_actual - v_target||` averaged over the trajectory.

3. **Practical simplified version.** For immediate implementation without per-segment curvature analysis, we can add a normalized race_time term to the composite. The paper's 11.8% improvement came from curvature awareness, but even a simple time penalty would break the current bias toward slow-and-safe. A reasonable starting composite: `0.5 * avg_error + 0.2 * worst_gate_error + 0.3 * (race_time / race_time_baseline)` where race_time_baseline is the median across candidates.

4. **Exponential mapping for smooth trade-off.** The Gaussian mapping g(K^n) = exp(-alpha * K^2) is a good functional form for any curvature-to-weight conversion because it is smooth, monotonic on [0, inf), and has a single tuning knob (alpha). We could use this to modulate per-gate tracking error weights: tolerate more error at high-curvature gates, demand precision at low-curvature gates where speed should be high.

---

## 5. Actionable Takeaways

1. **Add a time/speed component to the racing line composite score.** The `_select_best_by_sim` method in `planning/racing_line.py` already computes `race_time` from the kinematic sim but discards it during scoring (only used as a tiebreaker in sort). Incorporate it with a meaningful weight.

2. **Use curvature-modulated error tolerance.** When evaluating per-gate tracking error, weight each gate's error inversely with its approach curvature. High-curvature approaches naturally have larger error; penalizing them equally biases selection toward slow trajectories.

3. **Define aggressive/conservative velocity bounds for our drone.** Analogous to CiMPCC's v_bar and v_underline, set a max velocity (e.g., 15 m/s from our DroneConstraints) and a curvature-safe velocity (e.g., 5--8 m/s) and penalize candidates whose velocity profiles deviate from the curvature-mapped target.

4. **Exponential sensitivity parameter alpha.** Start with alpha = 2.0 (giving g(1) = e^{-2} approx 0.135) and tune via benchmark. Higher alpha means sharper speed reduction in corners.

5. **Zero out conflicting weights when adding new terms.** CiMPCC explicitly zeroes the longitudinal velocity weight in R2 when J_Ci takes over velocity regulation. Similarly, if we add a time penalty, we may need to reduce the avg_error weight to avoid conflicting objectives.

---

## 6. Limitations & Caveats

- **Ground vehicle only.** CiMPCC uses a kinematic bicycle model with no altitude dynamics. Drone racing has 6-DOF dynamics where curvature manifests in 3D, and aggressive banking/pitching means curvature-speed coupling is more complex.
- **No tracking error statistics reported.** The paper claims improved lap times but never quantifies whether lateral tracking error increased. For our application, we need both.
- **No ablation studies.** The sensitivity to alpha, smoothing window w, and velocity bounds is never analyzed. We cannot know which parameters matter most without our own tuning.
- **Single track, single vehicle.** All results are from one track on one car. Generalization is undemonstrated.
- **Curvature is precomputed from a fixed centerline.** In drone racing with sim-selected racing lines, curvature changes with each candidate, so the mapping must be recomputed per candidate evaluation.
- **No obstacle avoidance or multi-agent considerations.** Purely single-vehicle time-trial setting.

---

## 7. Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_p, N_c | 10 | Prediction and control horizon steps |
| L | 0.324 m | Vehicle wheelbase |
| v_bar | [4.18, 3.80] m/s | Aggressive velocity bounds (lon, steering) |
| v_underline | [2.72, 2.47] m/s | Safe velocity bounds |
| Q | diag(800, 800) | Contour/lag error weight |
| R1 | diag(10, 3500, 0) | Control change weight |
| R2 (CiMPCC) | diag(0, 10, 0) | Control reference weight (lon zeroed) |
| R3 | diag(40, 40) | Velocity tracking weight |
| gamma | 40 | Progress reward weight |
| alpha | not disclosed | Curvature sensitivity in exponential map |
| w | not disclosed | Moving average smoothing window |
| Solver | CasADi | NLP solver framework |
| Solve time | < 20.6 ms (95th pct) | Real-time capable |
