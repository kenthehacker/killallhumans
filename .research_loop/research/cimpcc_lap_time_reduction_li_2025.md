# Reduce Lap Time for Autonomous Racing with Curvature-Integrated MPCC
- **URL**: https://arxiv.org/abs/2502.03695
- **Authors**: Zhouheng Li, Lei Xie, Cheng Hu, Hongye Su
- **Year**: 2025
- **Venue**: arXiv preprint (submitted February 6, 2025); presented at ITSC 2024
- **Code**: https://github.com/zhouhengli/CiMPCC

---

## Key Contribution

This paper introduces CiMPCC (Curvature-Integrated Model Predictive Contouring Control), a local trajectory planning method for autonomous racing that explicitly maps racetrack curvature into the velocity reference signal embedded in the MPCC cost function. The central insight is that standard MPCC applies a uniform progress reward (gamma * v_p * T_s) that is geometry-agnostic: the optimizer has no mechanism to reduce speed through tight corners or to push velocity aggressively on straights. CiMPCC replaces this fixed incentive with a continuously varying velocity target that interpolates between an aggressive upper bound and a conservative lower bound according to local curvature, using a smooth exponential mapping that is differentiable throughout and thus compatible with NLP solvers.

The practical result on a 1:10-scale F1TENTH ground vehicle platform across 17 consecutive laps is an 11.4-12.5% reduction in lap time relative to both baseline MPCC and the global planning method RDM+OTG, accompanied by a 15-17% increase in mean velocity, with no reported crashes or constraint violations. The method remains real-time capable with over 95% of solver iterations completing in under 20.6 ms on an Intel NUC embedded computer. This work is relevant to any racing system where the controller does not have an explicit curvature-speed coupling in its cost formulation.

---

## Technical Approach

### Baseline MPCC Formulation

The vehicle state is zeta = [X, Y, phi, s]^T where (X, Y) is position, phi is heading, and s is arc-length progress along the centerline. Control inputs are u = [v_l, delta, v_p]^T representing longitudinal velocity, steering angle, and projected velocity along the path. The kinematic bicycle model is discretized at sampling period T_s with wheelbase L = 0.324 m. Standard MPCC minimizes:

```
J_MPCC = sum_k( ||xi_k||^2_Q  -  gamma * v_p,k * T_s )
       + sum_k( ||Delta_u_k||^2_R1 )
       + sum_k( ||u_k - u_ref||^2_R2 )
```

where xi_k = [e_con, e_lag] are contouring (lateral) and lag (longitudinal) errors relative to the centerline. The problem: gamma is a scalar constant, so the optimizer gains equal reward for every meter of progress regardless of whether the upcoming geometry demands braking.

### Curvature Extraction and Smoothing

Curvature kappa_i is computed from discrete centerline waypoints via the cross-product formula:

```
kappa_i = || Delta_x_i * Delta^2_y_i  -  Delta^2_x_i * Delta_y_i ||
          / ( Delta_x_i^2 + Delta_y_i^2 )^{3/2}
```

where Delta denotes finite differences. Raw curvature is noisy at the resolution of the waypoints, so a Moving Average Filter (MAF) smooths it with window width w:

```
K_i = (1/w) * sum_{m = i-(w-1)/2}^{i+(w-1)/2}  kappa_m
```

The smoothed values are then min-max normalized to the Normalized Smooth Curvature (NSC) K^n in [0, 1]:

```
K_i^n = (K_i - K_min) / (K_max - K_min)
```

Normalization ensures the curvature signal is scale-invariant across different tracks.

### Exponential Velocity Mapping

The core functional form converts normalized curvature into a blending coefficient beta:

```
g(K^n) = exp( -alpha * (K^n)^2 ),    alpha > 0
```

This has three desirable properties:
1. At zero curvature: g(0) = 1 (full aggressive speed — called the Upper Truncation Coefficient, UTC).
2. At maximum curvature: g(1) = exp(-alpha) > 0 (conservative but nonzero — the Lower Truncation Coefficient, LTC).
3. The function is a Gaussian in K^n — smooth, differentiable, monotonically decreasing, with a single tunable parameter alpha that controls how aggressively speed is reduced as curvature increases.

The specific value of alpha is not disclosed in the paper; it is a tunable hyperparameter. A value of alpha = 2.0 gives g(1) ≈ 0.135, meaning the conservative velocity bound dominates strongly at maximum curvature. A value of alpha = 1.0 gives g(1) ≈ 0.368, a gentler transition.

### Curvature-Integrated Cost Term

CiMPCC introduces a new cost term J_Ci that drives velocity toward a curvature-appropriate target:

```
J_Ci = sum_k[ (1 - beta) * ||v_k - v_underline||^2_R3  +  beta * ||v_k - v_overline||^2_R3 ]
```

where beta = g(K^n_cur) is the curvature-derived blending coefficient at the current planning step, v_overline is the aggressive velocity bound, v_underline is the conservative bound, and R3 is the velocity tracking weight matrix. The interpretation is:

- High beta (low curvature, straight): the cost heavily penalizes falling below v_overline — the optimizer accelerates.
- Low beta (high curvature, tight corner): the cost heavily penalizes exceeding v_underline — the optimizer decelerates.
- The transition is smooth and continuous, preventing discontinuous velocity commands.

The total CiMPCC cost is J_MPCC + J_Ci. Crucially, to prevent conflicting velocity objectives, the longitudinal velocity weight in R2 is zeroed out (R2 = diag(0, 10, 0)), transferring all velocity regulation responsibility to J_Ci. Only the steering reference tracking weight in R2 is retained.

### Complete Optimization Problem

```
Minimize:  J_Ci + J_MPCC

Subject to:
  zeta(0) = zeta_cur          (initial condition)
  zeta(k+1) = f_d(zeta(k), u(k))  (bicycle dynamics)
  zeta_min <= zeta(k) <= zeta_max  (state bounds)
  u_min <= u(k) <= u_max           (control bounds)
```

Solved with CasADi over prediction/control horizon N_p = N_c = 10 steps. The NLP is warm-started at each iteration. Computation: >95% of solves complete in under 20.6 ms on Intel NUC hardware, confirming real-time feasibility at 50+ Hz.

---

## Results

Experiments were conducted on a physical 1:10-scale F1TENTH autonomous racing car (DDRA platform) running ROS Melodic on an Intel NUC. The racetrack features sections with "sharp curvature." Localization uses a particle filter over a Cartographer SLAM map. Each method completed 17 consecutive laps.

### Lap Time Comparison

| Metric       | MPCC (s) | RDM+OTG (s) | CiMPCC (s) | CiMPCC vs MPCC |
|--------------|----------|-------------|------------|----------------|
| Maximum      | 16.654   | 16.404      | 14.568     | -12.5%         |
| Minimum      | 15.801   | 15.618      | 13.945     | -11.7%         |
| Mean         | 16.106   | 16.083      | 14.202     | -11.8%         |

CiMPCC mean lap time 14.202 s is 11.8% faster than MPCC (16.106 s) and 11.7% faster than RDM+OTG (16.083 s).

### Velocity Performance

| Metric    | MPCC (m/s) | CiMPCC (m/s) | Improvement |
|-----------|------------|--------------|-------------|
| Maximum   | 2.963      | 3.404        | +14.9%      |
| Minimum   | 2.816      | 3.294        | +17.0%      |
| Mean      | 2.910      | 3.351        | +15.2%      |

CiMPCC increases mean velocity by 15.2%, confirming that the improvement is from higher speeds on straights rather than line optimization alone.

### Tracking Error

No explicit lateral/contouring tracking error statistics are reported in the paper — a notable omission. The qualitative velocity profile plots show that CiMPCC achieves clear deceleration at high-curvature sections and strong acceleration in straights, validating the curvature-speed coupling. No crashes or constraint violations were reported across 17 laps.

### Computation

Solver time: >95th percentile below 20.6 ms (approximately 48 Hz minimum, typically faster). The method is real-time capable on embedded hardware.

---

## Relevance to Our System

Our drone racing pipeline uses `planning/racing_line.py` with a three-term normalized composite score for racing line candidate selection:

```
score = 0.5 * norm_avg_err + 0.2 * norm_worst_gate + 0.3 * norm_time
```

This composite already includes race time (added in iteration 23, motivated in part by a prior read of CiMPCC), but it applies a uniform time weight independent of where that time comes from. A candidate that saves 0.05 s by flying faster through gate-3 (S-turn, 0.247 m error — our hardest section) is scored identically to one that saves 0.05 s on the straight leading to gate-7 (helix entry, lower curvature, more tracking margin). CiMPCC's key contribution — curvature-modulated velocity incentives — is not yet present in our stack.

**Our specific bottleneck.** We need to recover 0.09 s (14.09 s to sub-14.0 s) without regressing gate-3 tracking error. This is precisely the trade-off CiMPCC is designed for: identify where curvature is low (easy sections) and preferentially increase speed there, while leaving the hard high-curvature gates alone. The exponential mapping g(K^n) = exp(-alpha * K^2) provides the exact mechanism.

**Drone vs. ground vehicle differences.** Our SpeedProfiler in `racing_line.py` already implements a curvature-based speed profile using the physical formula v = sqrt(a_max / kappa), with a forward-backward pass for acceleration limits. This is analogous to the curvature-speed idea in CiMPCC, but it operates at the trajectory generation level (input to TrajectoryOptimizer), not at the racing line selection level. The CiMPCC insight is that curvature-based speed targeting should also influence which racing line is selected — not just how fast we fly along whatever line was chosen.

**The missing link.** Our `_kinematic_eval` function in `_select_by_sim` computes `race_time` by integrating a PD controller over the trajectory. The race_time term in the composite score thus implicitly captures some curvature effect (tighter lines through corners are slower). However, the curvature of the path is never explicitly computed or used to modulate per-gate error tolerances during selection. The CiMPCC approach suggests we should explicitly compute path curvature at each gate approach, use the exponential mapping to derive a curvature-modulated error tolerance, and apply that tolerance when scoring candidates — being more lenient on high-curvature (hard) gates and more demanding on low-curvature (easy) ones.

**Concrete application to gate-3 S-turn.** Gate-3 has inherently high approach curvature due to the S-turn geometry. Penalizing its 0.247 m error equally to a straight-approach gate biases selection away from candidates that would be fast on the low-curvature sections. Explicitly downweighting gate-3's error contribution (via curvature-modulated weight) could allow the selection of a faster overall trajectory without regressing the gate-3 error that is physically bounded by dynamics.

---

## Actionable Takeaways

1. **Compute per-gate approach curvature in `_select_by_sim`.** For each gate, compute the curvature of the trajectory in the 1–2 m window before gate passage. Use the NSC formula: normalize across all gates in the candidate, then apply g(K^n) = exp(-alpha * K^2) with alpha = 2.0 as starting point.

2. **Curvature-modulated per-gate error weighting.** Replace the uniform worst-gate term in the composite with a curvature-weighted version: for gate i, weight its per-gate error by (1 / g(K_i^n)) so high-curvature gates are penalized less. This directly implements the CiMPCC insight that hard corners deserve looser speed/error targets.

3. **Target race time recovery via low-curvature sections.** When comparing candidate trajectories, compute the velocity profile's curvature-weighted deviation: sum_i g(K_i^n) * (v_target - v_actual)^2. Candidates that are slow on low-curvature segments are penalized; those that are slow on high-curvature segments are not. This recovers the 0.09 s time target by pushing speed on straights, not by risking gate-3.

4. **Apply exponential mapping to the SpeedProfiler's turn_speed_factor.** The SpeedProfiler in `racing_line.py` uses a fixed `turn_speed_factor = 0.4` to scale minimum speed in turns. Replace this with the CiMPCC exponential: g(K^n) * v_max + (1 - g(K^n)) * v_min. This creates a smooth, continuously-varying speed profile rather than a binary fast/slow split. Start with alpha = 2.0 and tune via benchmark.

5. **Zero conflicting weight when adding curvature-modulated terms.** Analogous to CiMPCC zeroing R2's longitudinal weight when J_Ci takes over: if we add a curvature-modulated velocity term to the composite, reduce the flat `_W_TIME` weight from 0.3 to 0.15 to avoid double-counting time via two mechanisms.

6. **Use the MAF smoothing before curvature computation.** Our existing `_compute_curvatures` in SpeedProfiler does not smooth before differentiating. Adding a window-3 or window-5 moving average will reduce noise-driven curvature spikes that incorrectly signal tight corners. The CiMPCC MAF with odd window width w is directly implementable in one line of numpy.

7. **Do not attempt to port the full MPCC formulation.** CiMPCC is a ground-vehicle NLP solved at 50 Hz with a kinematic bicycle model. Our system uses a PD geometric tracker with pre-computed min-snap trajectories — a fundamentally different architecture. The portable insight is the curvature-velocity mapping concept and the exponential functional form, not the MPC cost structure.

---

## Limitations & Caveats

- **Ground vehicle only.** CiMPCC uses a 2D kinematic bicycle model. Drone racing has 6-DOF dynamics: curvature in 3D involves roll/pitch coupling, and the saturating attitude limit (0.85 rad on our system) means the effective speed-curvature relationship is more complex than v = sqrt(a_max / kappa). Direct parameter translation is not valid.

- **No tracking error statistics reported.** The paper demonstrates improved lap times but never quantifies lateral contouring error. For a system like ours where tracking error is a primary metric, we cannot know whether CiMPCC's faster lap times come at the cost of larger cross-track excursions. This is a significant gap for our use case.

- **No ablation studies on alpha or w.** The sensitivity to the exponential decay parameter alpha and the MAF window width w is not analyzed. We must tune these empirically via benchmark, starting with the alpha = 2.0 suggestion.

- **Single track, single vehicle.** All 17 laps are on one custom racetrack with one car. The track geometry and vehicle dynamics are not described in enough detail to assess generalizability. Whether the 11.8% improvement holds on tracks with different curvature distributions is unknown.

- **Curvature precomputed from centerline, not optimal line.** CiMPCC computes curvature from the fixed centerline, not the optimized racing line. In our system, curvature changes with each candidate offset, so the exponential mapping must be recomputed per candidate during `_select_by_sim` — adding some computational overhead.

- **No multi-agent or obstacle considerations.** The method is designed for solo time-trial racing. In the AI Grand Prix competition format, interaction effects are not addressed.

- **Prediction horizon N = 10 is short.** A 10-step horizon (approximately 0.2 s at 50 Hz) means the planner has limited lookahead for upcoming curvature changes. In drone racing at 10–15 m/s, 0.2 s lookahead corresponds to 2–3 m, which may not be enough to anticipate sharp turns. Our pre-planned min-snap trajectory with full-lap planning has much longer effective lookahead.

- **The 11-12% improvement baseline is weak.** Both MPCC and RDM+OTG are reported to perform similarly (~16.1 s mean). These baselines may be poorly tuned. The improvement of CiMPCC could be partially attributable to suboptimal baseline configuration rather than purely to the curvature integration. No statistical significance tests are reported.

---

## Key Parameters / Constants

| Parameter          | Value                     | Description                                                  |
|--------------------|---------------------------|--------------------------------------------------------------|
| N_p, N_c           | 10                        | Prediction and control horizon (steps)                       |
| T_s                | ~0.02 s (50 Hz)           | Sampling period (inferred from solve time requirement)       |
| L                  | 0.324 m                   | Vehicle wheelbase (1:10 scale F1TENTH)                       |
| v_overline (lon)   | 4.18 m/s                  | Aggressive longitudinal velocity bound                       |
| v_overline (path)  | 3.80 m/s                  | Aggressive projected path velocity bound                     |
| v_underline (lon)  | 2.72 m/s                  | Safe/conservative longitudinal velocity bound                |
| v_underline (path) | 2.47 m/s                  | Safe projected path velocity bound                           |
| u_min              | [-10, -0.35, -10]         | Control lower bounds [m/s, rad, m/s]                         |
| u_max              | [10, 0.35, 10]            | Control upper bounds [m/s, rad, m/s]                         |
| Q                  | diag(800, 800)            | Contour/lag error penalty weight                             |
| R1                 | diag(10, 3500, 0)         | Control increment weight (high steering smoothing)           |
| R2 (MPCC)          | diag(40, 10, 40)          | Control reference weight in baseline MPCC                    |
| R2 (CiMPCC)        | diag(0, 10, 0)            | Control reference weight in CiMPCC (lon zeroed)              |
| R3                 | diag(40, 40)              | Velocity tracking weight for J_Ci                            |
| gamma              | 40                        | Progress reward weight in J_MPCC                             |
| alpha              | not disclosed             | Curvature sensitivity in exponential mapping; try 2.0        |
| w                  | not disclosed             | MAF smoothing window width; try 5 (odd)                      |
| Solver             | CasADi (NLP)              | Optimization backend                                         |
| Solve time P95     | < 20.6 ms                 | 95th percentile, Intel NUC embedded hardware                 |
| Lap time baseline  | 16.1 s (MPCC), 16.1 s (RDM+OTG) | F1TENTH 1:10 scale, custom track            |
| Lap time CiMPCC    | 14.2 s mean               | 11.8% improvement vs MPCC                                    |
| Mean velocity gain | +15.2%                    | 2.910 → 3.351 m/s                                           |

**For our drone system, the translatable constants are:**
- Exponential form g(K^n) = exp(-alpha * K^2) with alpha ∈ [1.0, 3.0]; start at 2.0.
- MAF window w = 5 for curvature smoothing before normalization.
- Zeroing conflicting velocity weight when curvature-modulated term is added.
- The normalized curvature K^n in [0,1] as the universal input to curvature-based modulation.
