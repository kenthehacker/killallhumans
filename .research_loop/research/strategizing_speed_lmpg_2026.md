# Strategizing at Speed: A Learned Model Predictive Game
- **URL**: https://arxiv.org/abs/2602.06925
- **Authors**: Andrei-Carlo Papuc, Lasse Peters, Sihao Sun, Laura Ferranti, Javier Alonso-Mora
- **Year**: 2026
- **Venue**: arXiv:2602.06925 (cs.RO / cs.GT), submitted February 6, 2026

---

## Key Contribution

The paper addresses a fundamental tension in competitive autonomous drone racing: how much strategic computation should an agent perform before committing to an action? At high flight speeds, the environment evolves faster than slow planners can respond, meaning that a more strategically sophisticated (but slower) planner may actually perform worse than a simpler, faster one.

The central contribution is the **Learned Model Predictive Game (LMPG)**, a method that "amortizes" the expensive game-theoretic computation offline via neural network training, enabling real-time deployment at 14x lower latency than the full online solver while preserving — and in practice exceeding — the strategic quality of the original Model Predictive Game (MPG). The paper also provides a rigorous comparative analysis of three planning paradigms (MPG, MPC, LMPG) across simulation tournaments and real hardware flights, making it one of the most thorough studies of multi-agent drone racing dynamics to date.

---

## Technical Approach

### Problem Formulation

The system models two-drone racing as a **Nash Equilibrium Problem (NEP)**. Each agent solves a coupled optimal control problem, treating its opponent as a strategic (responsive) actor rather than a passive obstacle. The state space is 11-dimensional: position (p_x, p_y, p_z), velocity (v_x, v_y, v_z), acceleration (a_x, a_y, a_z), track progress (θ), and progress velocity (v_θ). Control inputs are 4-dimensional: linear jerk (j_x, j_y, j_z) and path acceleration (Δv_θ). This jerk-level model gives a smooth, differentiable dynamics stack amenable to gradient-based optimization.

### Cost Function

Each agent minimizes a composite cost over a finite prediction horizon K, with five weighted terms:
1. **Contouring error**: Lateral and longitudinal deviation from the racing line, penalized separately with weights q_l and q_c.
2. **Input regularization**: Quadratic penalty q_u on control magnitude.
3. **Progress competition**: A term μ·(v^opponent_θ - v^ego_θ) that directly rewards closing the speed gap with the opponent.
4. **Collision penalty**: Soft barrier q_col applied only to the attacker role when within separation distance r_col = 0.35 m.
5. **Velocity limit penalty**: Soft constraint q_vel·Ψ(‖v‖, v_max) preventing constraint violations from crashing the solver.

### Three Planners Compared

**MPG (Model Predictive Game)**: Solves the full coupled NEP online at each time step using the PATH complementarity solver. In zero-latency (synchronous) execution, this is the theoretically ideal planner and achieves a 100% win rate against MPC. However, mean solve time is ~60 ms with worst-case spikes exceeding 2000 ms in complex interaction scenarios, causing timeout failures at high speed.

**MPC (contouring Model Predictive Control)**: Ignores opponent reactivity; treats opponent as a non-responsive obstacle predicted under constant velocity. Much faster to solve, but strategically blind — achieves 0 successful overtakes in 20 synchronous races against MPG.

**LMPG (Learned Model Predictive Game)**: Two MLPs trained offline to predict Nash equilibrium strategies directly from observation. A differentiable trajectory optimization layer is appended to guarantee kinodynamic feasibility. Mean inference time is ~3.5 ms, a 14x speedup over MPG. Training uses simultaneous gradient play with data aggregation that includes simulated delays (Bernoulli-distributed) and Gaussian control noise, making the policy robust to real-world asynchrony.

### Observation Encoding

Each network receives a structured observation vector: opponent position and velocity in the ego body frame, ego absolute position, three equidistant reference track points at 0.75 m spacing ahead, ego velocity and acceleration, progress velocity, and an attacker-role boolean flag. This encoding is compact and domain-general.

### Hardware Platform

Real-world validation uses custom quadrotors running the Agilicious framework on a Raspberry Pi 5, tracked via motion capture at 100 Hz in an 8 m × 5 m × 6 m arena. Low-level control uses hierarchical geometric control with Incremental Nonlinear Dynamic Inversion (INDI).

---

## Results

### Simulation Tournaments

In high-speed asynchronous mode (the most realistic condition):
- LMPG dominates MPC across all three track layouts.
- LMPG consistently beats MPG despite MPG's superior theoretical strategy, because MPG's solver timeouts cause it to fly off-track or collide.
- MPC is competitive with MPG in asynchronous mode — not because MPC is strategic, but because MPG's latency failures level the playing field.

On the Lemniscate track, LMPG vs. MPG in synchronous mode produces 187 vs. 177 overtakes for players 1 and 2 respectively, showing that LMPG has learned more aggressive overtaking behavior than the online solver (which is prone to local equilibrium traps).

### Speed Configurations

| Configuration | Attacker v_max | Defender v_max |
|---------------|---------------|----------------|
| Low speed     | 2 m/s         | 1 m/s          |
| High speed    | 3 m/s         | 2 m/s          |

The performance gap between LMPG and MPG widens at high speed, confirming the core thesis that latency becomes the binding constraint at competition-relevant velocities.

### Real Hardware

Eight races per comparison group on the Lemniscate track confirm simulation trends. LMPG outperforms MPG in physical flights, with MPG exhibiting the same timeout-driven instability observed in simulation.

---

## Relevance to Our System

Our current system is single-agent (no opponent), but this paper is highly relevant for several reasons:

**Speed-Latency Tradeoff in the Control Loop**: The paper's central finding — that computational latency can negate the value of a better planner at high speeds — applies directly to our ILC/MPC tracker. Our benchmark shows avg_loop_hz must stay above 100 Hz. Any trajectory optimization that takes more than 10 ms per step risks the same failure mode that defeats MPG here. The LMPG result quantifies this concretely: 60 ms solve time is catastrophic at 3 m/s; 3.5 ms is fine.

**Offline Amortization Pattern**: LMPG's strategy of training a neural network to predict the output of an expensive online optimizer is directly applicable to our trajectory_optimizer.py and racing_line.py. If our min-snap trajectory replanning or ILC update steps become bottlenecks, amortizing them offline (learning a fast surrogate) could maintain loop frequency while preserving plan quality.

**Contouring Error Decomposition**: The separation of lateral vs. longitudinal error (q_l vs. q_c weights) in the cost function is a cleaner formulation than a single tracking error norm. Our mpc_tracker.py currently uses a combined geometric error. Splitting into contouring (cross-track) and lag (along-track) components could give finer control over the speed-accuracy tradeoff, allowing the drone to accept more lag error in favor of speed while keeping cross-track tight through gates.

**Progress Velocity as a Reward Signal**: The term μ·(v^opponent_θ - v^ego_θ) reduces racing to a single scalar competition metric: path progress rate. In our single-agent case, this collapses to maximizing v_θ, which aligns with minimizing race time. Explicitly including v_θ as a reward signal in our trajectory optimizer (rather than just minimizing snap) could yield more aggressive speed profiles.

**Soft Constraint Handling**: The paper's soft penalty formulation for collision and velocity limits (Equations 4d, 4e) ensures solver feasibility in all cases. Our current hard constraint handling in trajectory_optimizer.py occasionally causes solver failures near tight gate margins. Adopting soft barrier functions with adaptive tightening during training (as in LMPG's data aggregation) is worth considering.

---

## Actionable Takeaways

1. **Benchmark control loop latency explicitly**: Add per-iteration solve time logging to benchmark.py. If any module exceeds 5-8 ms, it is a candidate for offline amortization or algorithmic simplification. The LMPG result shows 3.5 ms is achievable for a full game-theoretic planner; our single-agent tracker should be well within this.

2. **Split tracking error into contouring (lateral) and lag (longitudinal) components**: Modify mpc_tracker.py to separately penalize cross-track vs. along-track error. This enables the controller to be speed-aggressive (tolerating lag) while being precision-tight through gates (minimizing cross-track). This directly addresses the speed-accuracy tradeoff the paper formalizes.

3. **Maximize path progress rate (v_θ) explicitly**: In trajectory_optimizer.py or racing_line.py, add a term rewarding progress velocity rather than relying solely on minimum-snap objectives. This can yield lap time improvements without sacrificing safety margins.

4. **Adopt soft barrier functions for constraint handling**: Replace hard box constraints on velocity and position with soft penalty terms (log-barrier or quadratic penalty) to prevent solver failures near constraint boundaries, especially at high speeds near gate margins.

5. **Consider offline surrogate for ILC update**: If the Butterworth/ILC update step (currently adding measurable overhead per iteration) becomes a bottleneck at higher loop rates, train a small MLP to predict ILC corrections from error history, amortizing the online optimization cost.

6. **Use training data augmentation with delay injection**: LMPG's robustness to latency is partly due to training with simulated Bernoulli delays and control noise. If we train any learned component, include latency perturbation in the training distribution.

---

## Limitations & Caveats

- **Perfect state information**: The entire system assumes external motion capture at 100 Hz. There is no perception, no EKF drift, and no sensor noise in the strategic planner's state estimate. In our system, EKF uncertainty is a real degradation source that this paper does not address.

- **Local Nash equilibrium only**: The PATH solver finds first-order necessary conditions, not globally optimal equilibria. LMPG may also inherit this limitation if trained data is dominated by local solutions. For single-agent racing, this maps to local minima in trajectory optimization.

- **Limited track generalization**: LMPG is trained and validated on specific track layouts (Lemniscate and two others). Generalization to arbitrary gate configurations (as required in the AI Grand Prix) is not demonstrated.

- **Low absolute speeds**: Maximum tested velocity is 3 m/s, which is conservative for competitive drone racing (DCL races reach 20+ m/s). The strategy-vs-latency tradeoff may shift further against sophisticated planners at competition speeds.

- **Two-agent scope**: Results are specific to the 1v1 racing scenario. Multi-agent generalization (more than two drones) is not explored.

- **Arena size**: Hardware validation is limited to a single small track (8m × 5m × 6m) due to facility constraints, limiting the diversity of real-world validation.

---

## Key Parameters / Constants

| Parameter | Symbol | Value | Context |
|-----------|--------|-------|---------|
| Collision radius | r_col | 0.35 m | Minimum drone separation |
| Overtake lead distance | — | 0.75 m | Required ahead distance to claim overtake |
| Track deviation limit | — | 2 m | Off-track disqualification |
| Race length | — | 5 laps | Maximum race duration |
| Reference track point spacing | — | 0.75 m | 3 equidistant points in observation |
| MPG mean solve time | — | ~60 ms | Full online Nash solver |
| MPG worst-case solve time | — | >2000 ms | Complex interaction scenarios |
| LMPG mean inference time | — | ~3.5 ms | Neural surrogate, 14x speedup |
| Low-speed attacker v_max | — | 2 m/s | Slow racing configuration |
| Low-speed defender v_max | — | 1 m/s | Slow racing configuration |
| High-speed attacker v_max | — | 3 m/s | Fast racing configuration |
| High-speed defender v_max | — | 2 m/s | Fast racing configuration |
| State dimension | — | 11 | p, v, a, θ, v_θ |
| Control dimension | — | 4 | jerk (3D) + Δv_θ |
| Motion capture frequency | — | 100 Hz | Real-world localization rate |
| Hardware races per group | — | 8 | Real-world experiment replication |
| LMPG vs. MPG sync overtakes | — | 187 / 177 | Player 1 / Player 2, Lemniscate |

**Code repository**: https://github.com/andrejcarlo/ral26_strategizing_at_speed
