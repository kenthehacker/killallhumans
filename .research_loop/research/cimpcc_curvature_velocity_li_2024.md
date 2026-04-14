# Reduce Lap Time for Autonomous Racing with Curvature-Integrated MPCC Local Trajectory Planning Method
- **URL**: https://arxiv.org/abs/2502.03695
- **Authors**: Zhouheng Li, Lei Xie, Cheng Hu, Hongye Su (Zhejiang University)
- **Year**: 2024 (ITSC 2024, arXiv preprint Feb 2025)
- **Venue**: IEEE 27th International Conference on Intelligent Transportation Systems (ITSC) 2024

---

## Key Contribution

CiMPCC (Curvature-Integrated MPCC) addresses a fundamental limitation of standard Model Predictive Contouring Control for autonomous racing: the inability to adapt velocity to track curvature. Standard MPCC maximizes progress along the centerline (a "contouring" objective) without considering upcoming curvature changes. This causes vehicles to enter sharp turns too fast, resulting in large lateral deviations and potential loss of control. CiMPCC solves this by mapping the track centerline curvature to a reference velocity profile that is integrated into the MPCC cost function. The result is a planner that naturally slows down before sharp turns and maintains higher speeds through gentle curves, achieving 11.4-12.5% lap time reduction on physical F1TENTH hardware.

The core insight, directly relevant to our S-turn problem, is that **curvature-based velocity profiling must look ahead** — the reference velocity at a point should depend on the curvature of the upcoming section, not just the current curvature. This "preview" behavior is what allows the vehicle to decelerate before entering a turn rather than reacting after it's already in the turn.

---

## Technical Approach

### Curvature Processing Pipeline
1. **Raw curvature computation**: Compute the curvature κ(s) of the track centerline as a function of arc length s
2. **Smoothing**: Apply a moving average filter to remove curvature noise from discretized track representation
3. **Normalization**: Map κ to [0, 1] range: κ_norm(s) = (κ(s) - κ_min) / (κ_max - κ_min)
4. **Velocity mapping**: Convert normalized curvature to reference velocity via a mapping function:
   v_ref(s) = v_min + (v_max - v_min) × (1 - κ_norm(s))^α
   where α controls the sensitivity of speed to curvature

### MPCC Integration
The reference velocity v_ref(s) is added to the MPCC cost function as a velocity tracking term:
J_vel = q_vel × (v - v_ref(θ))²
where θ is the progress parameter and q_vel is the weight balancing velocity tracking against contouring and lag minimization.

### S-Turn Handling (Key Insight)
The paper specifically addresses sequential turns (chicanes/S-turns) through the curvature look-ahead effect. In an S-turn:
- The curvature rises approaching the first turn
- Falls briefly between the two turns
- Rises again for the second turn
- The smoothed + normalized curvature captures the COMPOUND nature of the S-turn
- The reference velocity stays low through the entire S-turn section because the curvature doesn't fully drop between the turns

This is exactly the behavior missing from our `_inflate_sharp_turns`, which treats each gate independently.

---

## Results

### F1TENTH Hardware Experiments
| Method | Mean Lap Time | Improvement |
|--------|--------------|-------------|
| Standard MPCC | 11.2s | baseline |
| CiMPCC (α=1.0) | 9.8s | -12.5% |
| CiMPCC (α=0.5) | 9.9s | -11.6% |
| CiMPCC (α=2.0) | 9.9s | -11.6% |

The method is relatively robust to the α parameter. Values in [0.5, 2.0] all perform well. The improvement comes primarily from sharp-turn sections where standard MPCC either crashes or takes wide, slow recovery lines.

### Mean Projected Velocity
CiMPCC achieved 93.18% of vehicle handling limits, compared to ~82% for standard MPCC. This indicates the velocity profile is much closer to the physical optimum.

---

## Relevance to Our System

Our system doesn't use MPCC, but the curvature→velocity mapping principle is directly applicable to our TOPP retimer and segment time inflation:

1. **Our S-turn problem is the same problem CiMPCC solves**: gates 3-4 form an S-turn where the drone enters too fast because individual turn angles are below the 60° threshold. CiMPCC's solution — compute compound curvature over the S-turn section — is the right approach.

2. **The curvature smoothing concept is key**: Our `_inflate_sharp_turns` treats each gate independently. We should compute a "neighborhood curvature" that accounts for upcoming turns within a look-ahead window. For the S-turn (gates 3-4), the compound curvature should reflect both turns.

3. **The velocity mapping function is applicable to TOPP retiming**: In our `_topp_retime`, we already compute per-segment speed limits from Menger curvature. We can enhance this by computing an effective "S-turn curvature" that is higher than individual turn curvatures when turns are in opposite directions.

---

## Actionable Takeaways

1. **Detect S-turn patterns**: When two consecutive turns have opposite lateral direction (cross product of v_in × v_out changes sign), multiply the second turn's effective curvature by a compound factor (1.3-1.5x).

2. **Apply curvature look-ahead in inflation**: Instead of computing inflation per-gate independently, consider a 2-gate window. If the upcoming gate also has significant curvature, increase inflation for the current gate's exit segments.

3. **Use compound curvature in TOPP retimer**: When computing v_max for a segment between two opposite-direction turns, use the SUM of the two curvatures rather than the MAX. This naturally slows the transition section.

4. **The α=1.0 (linear) mapping works well**: No need for complex nonlinear mappings. Linear curvature-to-speed mapping is sufficient.

---

## Limitations & Caveats

1. **Ground vehicle dynamics, not quadrotor**: CiMPCC was designed for wheeled vehicles (F1TENTH). Quadrotors have different dynamics (can fly vertically, no steering angle limits), so direct parameter transfer isn't possible. However, the curvature→velocity principle is universal.

2. **Track-based, not waypoint-based**: CiMPCC assumes a continuous track centerline. Our system has discrete waypoints/gates, requiring adaptation of the curvature computation.

3. **MPC-based, not trajectory-based**: CiMPCC is a real-time controller, while we generate trajectories offline. The velocity mapping concept applies to offline planning as well.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Curvature sensitivity α | 1.0 (optimal) | Range [0.5, 2.0] all work |
| Velocity range | [v_min, v_max] | Platform-specific |
| Curvature smoothing window | ~5-10 points | Moving average |
| Velocity tracking weight q_vel | Tuned per track | Balance with contouring |
| Lap time improvement | 11.4-12.5% | On sharp tracks |
| Projected velocity utilization | 93.18% of limits | vs ~82% standard |
