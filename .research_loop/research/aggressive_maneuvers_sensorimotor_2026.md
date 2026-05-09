# Precise Aggressive Aerial Maneuvers with Sensorimotor Policies

- **URL**: https://arxiv.org/abs/2604.05828
- **Authors**: Tianyue Wu, Guangtong Xu, et al. (9 researchers)
- **Affiliations**: Not specified in abstract; inferred robotics/ML research group
- **Year**: 2026
- **Venue**: arXiv:2604.05828 (April 2026, multiple revisions June 2025–April 2026)

---

## Key Contribution

This paper addresses one of the hardest open problems in agile drone flight: traversing narrow gaps at extreme orientations using **only onboard sensors**, without any prior knowledge of gap pose or access to external localization (no motion capture, no pre-mapped environment).

The central contribution is a **two-stage teacher-student reinforcement learning framework** that:
1. Trains an oracle RL policy with privileged gap geometry observations (gap edge point cloud + attitude + velocity)
2. Distills that oracle into a deployable vision-based policy that reads only a monocular camera image and IMU

The key achievement is navigating through a **20cm × 60cm rectangular gap** (only 5cm clearance from propeller tips) at tilt angles up to **90° roll** and **60° pitch** — without knowing the gap's orientation in advance. This surpasses prior work by ~2× in both tilt angle range and clearance margin.

The paper is particularly significant for its **informed reset strategy** during RL training: instead of starting episodes from hover (which leads to pathological exploration in such a constrained solution space), trajectories from a model-based planner are used to initialize the RL agent near the gap. This reduces required training samples by ~3× and boosts success rate from 70% to 96%.

---

## Technical Approach

### Problem Formulation

The quadrotor must traverse a gap of arbitrary orientation (roll up to 90°, pitch up to 60°) with no prior pose knowledge. The task is decomposed into:
1. **Approach phase**: Detect gap via vision, generate approach trajectory
2. **Traversal phase**: Execute aggressive maneuver through the gap using the learned policy
3. **Recovery phase**: Stabilize after traversal using a PD controller + optical flow hover

The RL formulation treats gap traversal as a Markov Decision Process (MDP) where:
- **State**: Gap edge points (32 uniformly sampled 3D points), attitude, velocity
- **Action**: Collective thrust [0.61g–2.04g] and body rates [up to 6 rad/s]
- **Horizon**: Single-gap traversal episode with termination on collision or successful traversal

### Two-Stage Policy Architecture

**Stage 1 — Oracle RL Policy:**
- MLP processes gap point cloud via global max-pooling (permutation-invariant)
- Fused with attitude quaternion and velocity vector
- Trained with PPO, initialized via informed-reset from model-based planner trajectories
- Achieves 96% simulation success rate on diverse gap orientations

**Stage 2 — Vision Distillation (DAgger):**
- Lightweight CNN encoder on 320×256 masked binary images (color-threshold HSV segmentation)
- Single-layer GRU captures temporal context across historical observations
- MLP fuses CNN features with attitude and previous action
- Separate output head for traversal completion detection (triggers recovery)
- Distilled via DAgger (Dataset Aggregation), collecting expert labels from oracle policy on student rollouts

### Reward Function

The traversal reward is the primary signal, scaling progress through the gap plane when the drone is collision-free. Auxiliary components:
- **Shaping reward** (λ=0.3): Encourages approach toward gap centroid
- **Smoothness penalties**: Penalize large action magnitudes and action rate-of-change
- **Speed constraint**: Penalizes exceeding 4 m/s
- **Pitch compensation factor**: For pitched gaps, traversal reward is multiplied by `exp(−|θ_k − θ^g| / 20°)` to encourage attitude matching to the gap normal

The 20° exponential decay constant in the pitch compensation factor is a key design choice — it creates a soft constraint that strongly rewards aligning with the gap orientation within ±20° while gracefully degrading rather than hard-failing beyond.

### Trajectory Planning (Model-Based Component)

A model-based planner using **differential flatness** under SE(3) constraints generates reference trajectories. This is **not used during deployment** — only during training to initialize episodes near the gap (informed reset). The flatness-based planner provides geometrically consistent approach paths that give the RL agent a warm start rather than requiring it to discover gap approach from scratch.

### Sim-to-Real Transfer

Four mechanisms bridge the simulation-reality gap:
1. **Perturbation forces**: Random persistent external forces (applied for tens of steps) simulate unmodeled aerodynamics and prevent simulator overfitting
2. **Response delay simulation**: `a_k(n) = (1/w) * Σ a_(k-i)(n)` — a weighted average of past actions models flight controller latency
3. **Response randomization**: Multiplicative factor `ã_k(n) = â_k(n) · c`, where `c ∈ U(1 − c(n), 1 + c(n))`, randomizes motor response characteristics across episodes
4. **Perceptual latency simulation**: Calibrated ~4ms image processing delay added to observation pipeline

Total control loop latency is ~20ms (bodyrate commands to actuator).

---

## Results

### Tilt Angle Performance (Real-World Trials, 30 per condition)

| Gap Orientation | Success Rate |
|-----------------|-------------|
| Roll ≤ 60°      | 97% (29/30) |
| Roll > 60°–90°  | 90% (27/30) |
| Pitch 30°       | 100% (30/30)|
| Pitch 45°       | 80% (24/30) |
| Pitch 60°       | 73.3% (22/30)|

The asymmetry between roll and pitch performance is physically meaningful: a 90° roll gap requires the drone to knife-edge through on its side, which is achievable with proper trajectory planning. A 60° pitch gap requires the drone to dive or climb steeply through the gap while simultaneously aligning its body axis to the gap normal — a more dynamically demanding maneuver involving thrust vectoring.

### Gap Clearance

- Quadrotor dimensions: 38cm × 10cm (propeller-to-propeller)
- Collision model: 34cm × 34cm × 11cm
- Gap size: 20cm × 60cm
- Effective clearance: **5cm** — roughly one propeller width on each side

### Generalization

- Successfully traversed **moving/rotating** gaps despite training only on static gaps
- Handled geometrically diverse shapes: triangular, parallelogram, elliptical, diamond, arch
- Chained **3 consecutive gaps** with only 0.8m inter-gap spacing
- All generalization achieved with the same trained policy (zero-shot)

### Baselines Comparison

| Method | Max Tilt | Clearance | Localization Required |
|--------|----------|-----------|----------------------|
| Falanga et al. (prior SOTA) | 45° | 8cm | Yes (VIO + PnP) |
| Wang et al. | Unknown | Unknown | Yes (motion capture) |
| **This work** | **90°** | **5cm** | **No** |

The Falanga et al. baseline is particularly relevant — it uses vision-inertial odometry with PnP pose estimation (similar to our gate_pnp.py approach) and achieves 45° maximum tilt with 8cm clearance. The learned approach doubles the angular range and tightens clearance by 37%.

---

## Relevance to Our System

Our system (`race_pipeline.py`, `mpc_tracker.py`, `trajectory_optimizer.py`) uses a classical planning-and-control stack: min-snap polynomial trajectories through gate centers, tracked by a geometric SE(3) controller. Several findings from this paper apply directly:

### 1. Tilt Angle Handling in Classical Controllers

Our SE(3) geometric tracker (Lee et al.) in `mpc_tracker.py` does not have an explicit tilt angle constraint — it tracks attitude error with PD gains. The paper implies that **tilt angles up to ~60° are reliably achievable** even with classical bodyrate controllers (PX4 in their case), provided the trajectory through the gap is geometrically consistent with the drone's collision envelope.

The critical question is not whether the controller can command 60° bank angle, but whether the **trajectory planner computes a path that is geometrically compatible with the gap normal** at the moment of traversal. For upright racing gates (typical of AI Grand Prix), gate normals are near-horizontal, and bank angles during traversal are modest — probably 20–35° for tight turns.

### 2. Recovery After Gate Traversal

The paper's post-traversal recovery (PD stabilization + optical flow hover) is analogous to our virtual finish waypoint approach. The key insight: **decouple the aggressive traversal phase from the recovery phase**. Our `sequencer.py` gate pass-through detection could similarly trigger a mode switch.

### 3. Vision Without Explicit Pose Estimation

Their approach succeeds without PnP pose estimation (no `gate_pnp.py` equivalent). The vision-based distillation policy implicitly encodes gap pose in the GRU hidden state. This is a longer-term direction for us but not immediately actionable given our current stack.

### 4. Informed Reset as Training Insight

The 3× sample efficiency gain from trajectory-based initialization is a strong argument for using model-based trajectories as a **prior** even when the final system is learned. If we pursue RL fine-tuning of our controller, warm-starting from our min-snap trajectories would be the right approach.

### 5. Speed Limits and Clearance

The policy's speed penalty at 4 m/s for a 20cm × 60cm gap is informative. Racing gates (typically 1.5m × 1.5m or larger) allow much higher traversal speeds. Scaling the clearance ratio (gap_size / vehicle_size), our system should safely handle 8–12 m/s gate traversal speeds with our current trajectory optimizer.

---

## Actionable Takeaways

For our immediate iteration targets on the AI Grand Prix system:

1. **Tilt angle budget**: Classical bodyrate control is reliable up to ~60° bank angle (90° is achievable but with degraded success rate). Our trajectory optimizer should budget for 30–45° maximum bank during tight turns to maintain high reliability. Consider adding a tilt angle constraint to `trajectory_optimizer.py` DroneConstraints.

2. **Traversal attitude alignment**: The pitch compensation reward `exp(−|θ_k − θ^g| / 20°)` implies that misalignment beyond ~20° significantly degrades traversal probability. Our trajectory should ensure the drone's attitude at gate crossing aligns with the gate normal within ±20°. This is already implicit in min-snap (flat flight at gate centers), but should be verified for banked turns.

3. **Speed vs. clearance tradeoff**: At 4 m/s through a 5cm-clearance gap, the drone must be nearly perfect. Our much larger gates (clearance ~50cm typical) mean we can fly significantly faster. The constraint is attitude alignment, not speed per se.

4. **Domain randomization for robustness**: If we pursue any learned components (even a learned gain scheduler), the four sim-to-real techniques (force perturbations, delay modeling, response randomization, perceptual latency) should be adopted wholesale. Specifically, adding random force perturbations to our PyBullet test environment would make our classical controller gains more robustly tuned.

5. **Control loop latency**: Their 20ms bodyrate latency (and 4ms vision latency) matches typical PX4 setups. Our `state_predictor.py` forward-prediction approach is the correct response to this latency — latency compensation is essential for aggressive flight.

6. **Consecutive gap handling**: Their success at 0.8m inter-gap spacing with 3 gates suggests that 1–2m inter-gate distances (common in racing) are feasible with good trajectory planning. Our sequencer must handle rapid sequential gate transitions cleanly — the gate completion detection (output head in their policy) is analogous to our `pass_through_margin` in `sequencer.py`.

---

## Limitations & Caveats

1. **Not a racing paper**: The task is gap traversal at a static or slowly moving target, not continuous high-speed lap racing through multiple gates. The learned policy optimizes a single traversal, not a multi-gate racing line. Generalization to racing gates is promising but not directly demonstrated.

2. **Low absolute speed**: Maximum speed of 3–5 m/s (domain randomization dependent) is significantly below competitive drone racing speeds (15–25 m/s). The extreme tilt angles (90° roll) are impressive but occur at slow approach speeds where the tight gap is the challenge, not the turn rate.

3. **No racing line optimization**: The approach trajectory is generated by a model-based planner to the gap, but no attempt is made to optimize the racing line (lateral offset, curvature-aware speed) across multiple gates. Our `racing_line.py` provides capabilities this paper does not address.

4. **Vision segmentation fragility**: HSV color-threshold segmentation is brittle to outdoor lighting changes and requires a visually distinctive gap. Competition gates will have standardized markers, making color-based detection feasible, but reliance on a single simple cue is a risk.

5. **Single maneuver evaluation**: Success rates (73–100%) are measured over 30 trials for a single gap crossing. In a multi-gate race, even a 90% per-gate success rate means ~35% chance of completing a 10-gate course — composition of independent per-gate probabilities is a concern.

6. **Hardware-specific**: The 38cm quad with PX4 bodyrate interface is specific. Mapping to other vehicles or autopilots requires re-tuning the sim-to-real transfer components.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Max tilt angle (roll) | 90° | Gap orientation, not drone bank angle |
| Max tilt angle (pitch) | 60° | Gap orientation |
| Success rate knee point (roll) | 60° | Above this, success drops 97% → 90% |
| Success rate knee point (pitch) | 30°–45° | 100% → 80% |
| Minimum gap clearance | 5 cm | Per side, with 34cm × 34cm drone |
| Max traversal speed penalty | 4 m/s | RL reward constraint |
| Max body rate | 6 rad/s | Action space limit |
| Thrust range | 0.61g–2.04g (6–20 m/s²) | Action space limit |
| Control loop latency | ~20 ms | PX4 bodyrate |
| Vision inference latency | ~4 ms | MobileNetv3 segmentation |
| Pitch compensation decay constant | 20° | `exp(−|θ_k − θ^g| / 20°)` |
| Shaping reward weight | λ = 0.3 | Relative to traversal reward |
| Inter-gap spacing tested | 0.8 m | Consecutive gap experiment |
| Training time | ~1.5 hours | RTX 4090, PPO |
| Informed reset sample savings | ~3× | vs. hover-start initialization |
| Image resolution (inference) | 320 × 256 | Downsampled from 1280 × 1024 |
| Camera FOV | 82° × 72° | Monocular |
| Policy architecture | CNN + GRU + MLP | Vision distillation stage |
| Gap point samples | 32 | Oracle policy point cloud |
