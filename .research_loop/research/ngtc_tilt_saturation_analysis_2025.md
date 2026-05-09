# NGTC: Tilt Saturation Analysis for Aggressive Flight (Pries & Ryll 2025)

- **URL**: https://arxiv.org/abs/2510.12611
- **Authors**: Lukas Pries, Markus Ryll
- **Year**: 2025
- **Venue**: arXiv preprint

## Key Contribution

The Neural Geometric Tracking Controller (NGTC) paper provides critical data on the relationship between tilt angle constraints and tracking performance in aggressive quadrotor flight. The paper uses a maximum tilt angle β = 56° (0.977 rad) as the platform limit, and systematically evaluates how Differential Flatness-Based Control (DFBC) — the controller class most similar to our GeometricTracker — degrades when trajectories push the drone into saturation. The paper demonstrates that DFBC tracking error degrades from 0.23m on feasible trajectories to 2.39m on infeasible trajectories (10.4× degradation), directly attributable to tilt saturation.

## Technical Approach

The paper compares four controller paradigms:
1. **DFBC**: Standard differential flatness PD + feedforward (like our system)
2. **NMPC**: Nonlinear Model Predictive Control (constraint-aware)
3. **NGTC**: Neural-augmented geometric control (their contribution)
4. **RL**: End-to-end reinforcement learning

The DFBC controller uses gains Kx=(18,18,18) for position and Kv=(8,8,8) for velocity — notably 2-3× higher than our kp=7, kd=5.5. This aligns with the literature standard for aggressive flight and suggests our gains may still be conservative.

The max tilt β=56° corresponds to 0.977 rad. Our current max_tilt_rad=0.85 rad (49°) is 13% more conservative. The paper notes that "the proportional nature of DFBC can often be over-aggressive in correcting errors, significantly degrading control performance at saturation limits." This suggests a nuanced relationship: higher gains improve unsaturated performance but worsen saturated behavior.

## Results

| Controller | Feasible Error | Infeasible Error | Degradation Factor |
|-----------|----------------|------------------|--------------------|
| DFBC | 0.23m | 2.39m | 10.4× |
| NMPC | 0.23m | 1.77m | 7.7× |
| NGTC | 0.20m | 1.42m | 7.1× |

Key finding: NGTC achieves 40% lower tracking error than DFBC on infeasible trajectories because the neural augmentation learns to compensate for saturation effects.

## Relevance to Our System

Our system uses a DFBC-equivalent GeometricTracker with max_tilt_rad=0.85. The benchmark data shows both roll and pitch saturating at this limit. The NGTC paper demonstrates that:

1. **0.85 rad (49°) is conservative** compared to the 0.977 rad (56°) used in this paper
2. **DFBC degrades catastrophically at saturation** — our gates 2, 3, 7 (highest error) are likely the points where tilt saturates
3. **Increasing max_tilt_rad** from 0.85 to ~0.97 would expand the non-saturating envelope, potentially reducing error at the worst gates

## Actionable Takeaways

1. Increase max_tilt_rad from 0.85 to 0.97 rad (56°) to match NGTC's demonstrated safe limit
2. Monitor whether saturation frequency decreases at gates 2, 3, 7
3. Consider that our gains (kp=7, kd=5.5) are conservative relative to literature (Kx=18, Kv=8)
4. The ILC inner sim has no tilt constraint, so ILC corrections are already calibrated for an unconstrained controller — increasing max_tilt makes the benchmark MORE aligned with ILC assumptions

## Limitations & Caveats

- NGTC results are from a different platform (Crazyflie 2.1 model) with different mass/thrust properties
- Our kinematic sim has a simplified drag model (drag=0.5) that interacts with tilt changes
- The paper's 56° limit may be platform-specific; our sim might have different stability boundaries
- Higher tilt means faster velocity buildup, which increases drag-induced deceleration

## Key Parameters / Constants

| Parameter | NGTC Paper | Our System | Gap |
|-----------|-----------|------------|-----|
| Max tilt | 56° (0.977 rad) | 49° (0.85 rad) | +7° |
| Kp (position) | 18 | 7 | 2.6× |
| Kv (velocity) | 8 | 5.5 | 1.5× |
| Body rate limit | 6 rad/s | 6 rad/s | Match |
