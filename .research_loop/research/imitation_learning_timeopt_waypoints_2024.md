# Imitation Learning-Based Online Time-Optimal Control with Multiple-Waypoint Constraints for Quadrotors

- **URL**: https://arxiv.org/abs/2402.11570
- **Authors**: Jin Zhou, Jiahao Mei, Fangguo Zhao, Jiming Chen, Shuo Li (Zhejiang University)
- **Year**: 2024
- **Venue**: arXiv cs.RO (submitted February 18, 2024)

---

## Key Contribution

This paper addresses a fundamental tension in time-optimal quadrotor control: offline optimizers (like CPC) compute near-perfect trajectories but take on the order of minutes per problem instance, making them unsuitable for real-time operation or dynamic environments. The authors train neural networks — called WN&CNets — to imitate the control policy learned from ~37,000 CPC-generated time-optimal trajectories, achieving online inference at 30 Hz with a planning time of 0.033 seconds versus CPC's 11 minutes.

The secondary contribution is a polynomial-based "transition phase" strategy that bridges the gap created by a key training limitation: the network was trained only on 2-waypoint hover-to-hover scenarios, so naive deployment on multi-waypoint missions produces stop-and-go behavior at each waypoint. The polynomial interpolation "jumps over" these stop-and-go pauses, stitching together successive neural-network-controlled segments into a smooth multi-waypoint flight. Together, these two components yield a deployable system that flies 7-waypoint missions at up to 7.0 m/s in a 6m x 4m x 2m arena with only ~5% increase in flight time versus CPC.

---

## Technical Approach

### Quadrotor Dynamics and Problem Formulation

The quadrotor is modeled with a 13-dimensional state vector: position **p** ∈ ℝ³, velocity **v** ∈ ℝ³, quaternion attitude **q** ∈ ℝ⁴, and angular rates **ω** ∈ ℝ³. Control inputs are angular rate commands **ω**_c ∈ ℝ³ and scalar thrust T_c.

The optimization problem minimizes total flight time subject to: (1) quadrotor dynamics, (2) boundary conditions (start/end states), (3) actuation limits, (4) "subsequence constraints" ensuring waypoints are visited in order, and (5) "complementary progress constraints" that enforce each waypoint is passed within a spatial tolerance d_tol. This last constraint is what distinguishes waypoint-constrained time-optimal control from simpler point-to-point problems — the drone must pass through each gate/waypoint region without stopping.

### CPC Offline Optimizer (Teacher)

The CPC (Complementary Progress Constraints) algorithm solves the above optimization offline using discrete time segments. Each trajectory has approximately 80 nodes and accounts for system latency by predicting controls d time steps ahead. CPC is accurate and time-optimal but requires ~11 minutes per solve, making it unusable online.

### Imitation Learning (WN&CNets)

**Dataset generation**: 37,044 CPC-solved trajectories were generated, each covering a hover-to-hover flight through exactly 2 waypoints. With ~80 nodes per trajectory and accounting for latency variants (d = 0, 1, 2, 3 time steps), this yields approximately 3,000,000 state-control training pairs.

**Input features**: Relative positions to the next two waypoints (Δ**p** ∈ ℝ³ˣ²), current velocity **v** ∈ ℝ³, current acceleration **a** ∈ ℝ³, current angular rates **ω** ∈ ℝ³.

**Output**: Angular rate commands **ω**_c ∈ ℝ³ and thrust command T_c.

**Training**: Mean Absolute Error (MAE) loss with Adam optimizer, 80/20 train/validation split. Converges in 40 epochs to 99.80% accuracy and loss ~0.04.

**Latency compensation**: The network is trained on shifted data (input at time k-d predicts output at time k+1), implicitly learning to compensate for system delay.

**Deployment**: Runs at 30 Hz, takes 0.033 seconds per inference — five orders of magnitude faster than CPC.

### Transition Phase Strategy (The "Jump-Over" Approach)

The core limitation is that WN&CNets only learned 2-waypoint behavior ending in hover. On a multi-waypoint mission, each time the drone reaches a waypoint, the network naturally decelerates to hover before the next segment begins — producing inefficient stop-and-go flight.

The authors solve this with a polynomial bridge that activates near each intermediate waypoint, bypassing the stop-and-go phase. The overall control architecture becomes:

```
Neural Network (WN&CNets) → [Transition Phase Trigger] → Polynomial Interpolation → [Exit Trigger] → Neural Network (next segment)
```

The transition is parameterized differently depending on the relationship between consecutive waypoint vectors:

**Case 1 — Same-direction flight** (consecutive waypoints lie roughly in the same direction):

- **Entry condition**: When the predicted (neural network) velocity **v** drops below threshold v_in (expressed as a fraction of v_max).
- **Exit condition**: When the predicted velocity rises back above threshold v_out (also a fraction of v_max).
- The polynomial bridge fills the gap between entry and exit points.
- Tested with v_in = 100%, 90%, 80% of v_max.

**Case 2 — Opposite-direction flight** (consecutive waypoints require reversing direction):

- **Entry condition**: When predicted acceleration **a** drops below threshold a_in (fraction of a_max).
- **Exit condition**: When predicted acceleration rises above a_out.
- Tested with a_in = 100%, 90%, 50% of a_max.

**Traverse time estimation**: Within the transition phase, the drone is assumed to have constant acceleration. This allows the polynomial traverse time to be computed analytically given entry/exit positions, velocities, and accelerations — no optimization required.

**Polynomial degree**: The paper uses polynomial interpolation with boundary conditions set at positions, velocities, and accelerations at the entry/exit points. The degree is not explicitly stated but matching position, velocity, and acceleration at both endpoints requires at least a quintic (5th-degree) polynomial, consistent with minimum-snap approaches.

Note: The paper makes no reference to MINCO (Minimum Control effort) trajectory formulation — the polynomial transition is a simpler boundary-condition-matching approach, not an energy-optimal formulation.

### Dynamic Waypoint Extension

Because inference is online at 30 Hz, the system can re-query with updated waypoint positions. The authors demonstrate tracking of a waypoint moving on a circular trajectory, with position errors of 0.07–0.36 m across four test flights — a capability entirely absent from offline-optimized approaches.

---

## Results

### Simulation (Multi-Waypoint Benchmark)

| Scenario | Method | Flight Time | Overhead vs CPC |
|----------|--------|-------------|-----------------|
| Exp 1 (same-dir) | WN&CNets (v_in=v_out=90% v_max) | 2.80 s | +9.8% |
| Exp 1 | WN&CNets (v_in=v_out=100% v_max) | varies | +16.9% |
| Exp 1 | MSTG&C baseline | 3.25 s | +27.5% |
| Exp 2 (opp-dir) | WN&CNets (a_in=90% a_max) | 3.46 s | +9.15% |

Best results use intermediate threshold values (90% of max velocity/acceleration) rather than 100% (too eager to switch) or 80%/50% (too conservative, too much stop-and-go).

### Real-World Experiments (7 Waypoints, Confined Arena)

| Metric | CPC (offline) | WN&CNets (online) |
|--------|--------------|-------------------|
| Flight time | 6.85 s | 7.18 s (+4.8%) |
| Max velocity | 7.0 m/s | 7.0 m/s |
| Waypoint error | 0.47 m | **0.26 m** (−45%) |
| Planning time | 11 minutes | **0.033 seconds** |

Notably, WN&CNets achieves *lower* waypoint error than CPC in real hardware flights (0.26 m vs 0.47 m). The authors attribute this to CPC's open-loop nature: it plans offline and cannot adapt to disturbances, while WN&CNets continuously re-evaluates state and re-issues commands at 30 Hz.

### Dynamic Waypoint Tracking

Four flight tests with a circular-moving waypoint: tracking errors of 0.07, 0.14, 0.16, and 0.36 m (best to worst). This demonstrates real-time replanning capability that no offline method can provide.

---

## Relevance to Our System

Our system uses min-snap polynomial trajectories with L-BFGS racing line optimization and a kinematic-sim PD controller. The current bottleneck is the S-turn at gate-3, where the drone arrives too fast at the entry of the first S-turn bend, producing 0.463 m average tracking error. This is structurally identical to the "opposite-direction flight" scenario studied in this paper: consecutive gates require the drone to reverse lateral direction, and the current time inflation/deceleration approach is insufficient.

**Directly applicable insights**:

1. **Transition phase concept**: Our current L-BFGS optimizer inflates segment times to slow the drone at curves, but does so uniformly. This paper validates that triggering deceleration based on a velocity threshold relative to v_max (rather than curvature or segment index alone) produces better results for direction-reversal scenarios. For gate-3, we should trigger additional deceleration when predicted velocity exceeds a fraction of v_max on approach.

2. **Opposite-direction flight is the hard case**: The paper treats same-direction and opposite-direction waypoints as fundamentally different problems requiring different triggering logic. Our S-turn is opposite-direction, so we should apply the acceleration-threshold logic (a_in = 90% a_max) rather than velocity-threshold logic.

3. **Constant-acceleration approximation**: The traverse time within a transition is estimated assuming constant acceleration. This is equivalent to what a trapezoidal velocity profile planner would produce. For our trajectory time allocation, explicitly computing the constant-acceleration traverse time for the S-turn segment (rather than relying on curvature heuristics) may yield more accurate time budgets.

4. **Latency compensation**: The network is trained with shifted inputs to account for system delay. Our PD controller already runs with a state predictor, but this paper confirms that explicitly modeling k-d step delays in the control loop is important at high speeds.

5. **Online re-evaluation beats offline planning under disturbance**: Even a slightly suboptimal online policy (WN&CNets) outperforms an optimal offline plan (CPC) on real hardware because it can compensate for disturbances. This suggests our fixed trajectory approach should be augmented with online tracking error feedback to adjust approach speed.

---

## Actionable Takeaways

1. **Implement velocity-threshold entry into S-turn deceleration**: On approach to gate-3 (first S-turn), detect when the drone's predicted speed exceeds 90% of v_max for that segment. At that point, apply additional time inflation to the segment rather than waiting for the existing curvature penalty to kick in.

2. **Treat opposite-direction gates differently from same-direction gates**: In `planning/racing_line.py`, classify each consecutive gate pair as same-direction or opposite-direction (dot product of approach vectors). Apply stronger deceleration budgets to opposite-direction pairs. This mirrors the paper's two-case transition strategy.

3. **Use constant-acceleration time estimate for S-turn segments**: For the gate-3 S-turn, compute the minimum traverse time assuming constant deceleration from entry speed to the speed required to make the turn radius. Use this as a lower bound on segment time in the L-BFGS optimizer, preventing it from allocating insufficient time.

4. **Add an acceleration-threshold exit condition for the S-turn**: After the first turn of the S-turn pair, the drone should re-accelerate. Set an exit condition analogous to a_out: only allow the trajectory planner to allocate aggressive time to the second turn once the predicted acceleration recovers above a_out threshold. This avoids the compounded error of arriving at the second S-turn bend also too fast.

5. **Consider per-segment speed cap as a pre-pass**: Before running L-BFGS, enforce a per-gate speed cap based on the gate geometry (approach vector reversal angle). Gates requiring >90-degree direction change should have a v_max cap computed from kinematic constraints (v = sqrt(a_max * r) for turn radius r).

6. **Dataset insight for potential NN approach**: If we later pursue learned control for gate-to-gate segments, this paper's dataset strategy (hover-to-hover, 2-waypoint, ~37K examples) is a tractable starting point. The ~10% overhead vs. optimal is acceptable for racing if it enables online re-planning around disturbances.

7. **Validate with threshold sweep**: The paper shows 90% of max is better than 100% or 80%. Run a sweep of time inflation multipliers for gate-3 approach (e.g., 1.2x, 1.5x, 2.0x of current allocation) to find the empirical optimum, analogous to their v_in parameter sweep.

---

## Limitations & Caveats

1. **Training data covers only 2-waypoint hover-to-hover scenarios**: The network never learned multi-waypoint or high-speed pass-through behavior. The polynomial bridge is a workaround for this limitation, not an elegant generalization.

2. **30 Hz control frequency**: The paper's experiments run at 30 Hz. Modern racing drones typically require 100–400 Hz control loops for stability at high speeds. The method's applicability at higher speeds or tighter arenas is unclear.

3. **10% overhead vs. optimal is non-trivial in racing**: In a race where margins are sub-second, a 10% time penalty from imitation learning is significant. The real-world experiment showed only +4.8% overhead, but this was in a relatively low-speed (7 m/s), confined scenario.

4. **No discussion of gate-passing orientation constraints**: Drone racing requires passing through gates at the correct yaw angle. The paper only addresses positional waypoints with a tolerance sphere d_tol, not orientation-constrained gate traversal.

5. **Confined arena limits speed**: All real-world experiments were in a 6m × 4m × 2m space at 7 m/s. Whether the approach scales to competition speeds (15–25 m/s) on larger tracks is not validated.

6. **Threshold sensitivity**: The transition thresholds (v_in, v_out, a_in, a_out) require tuning per scenario. The paper sweeps only a few discrete values; continuous optimization of these thresholds is not addressed.

7. **No aerodynamic or drag modeling**: The quadrotor model uses simplified thrust-only dynamics without rotor drag or aerodynamic effects, which become significant above ~5 m/s. This may limit accuracy of the CPC training data at higher speeds.

8. **No obstacle avoidance**: The method is pure waypoint-to-waypoint; it does not reason about collision with gate structures, which requires positional accuracy better than the gate aperture width.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Training trajectories | 37,044 | CPC-generated, hover-to-hover, 2-waypoint |
| Nodes per trajectory | ~80 | Discretization of each CPC solution |
| State-control pairs | ~3,000,000 | Total training data points |
| Network accuracy | 99.80% | After 40 epochs on validation set |
| Network loss | ~0.04 | MAE at convergence |
| Training split | 80/20 | Train / validation |
| Optimizer | Adam | Standard gradient-based |
| Loss function | MAE | Mean Absolute Error |
| Control frequency | 30 Hz | Deployment frequency |
| Planning time (WN&CNets) | 0.033 s | Per inference |
| Planning time (CPC) | ~11 min | Per trajectory solve |
| v_in sweep | 80%, 90%, 100% v_max | Entry threshold for same-direction transition |
| v_out sweep | 80%, 90%, 100% v_max | Exit threshold for same-direction transition |
| a_in sweep | 50%, 90%, 100% a_max | Entry threshold for opposite-direction transition |
| Best v_in / v_out | 90% v_max | Empirically optimal for same-direction |
| Best a_in | 90% a_max | Empirically optimal for opposite-direction |
| Max real-world speed | 7.0 m/s | Experiment arena constraint |
| Arena size | 6.0m × 4.0m × 2.0m | Real-world test environment |
| Waypoints (real test) | 7 | Multi-waypoint mission |
| CPC flight time | 6.85 s | 7-waypoint real-world baseline |
| WN&CNets flight time | 7.18 s | +4.8% vs CPC |
| WN&CNets waypoint error | 0.26 m | vs CPC's 0.47 m (−45%) |
| Dynamic waypoint error | 0.07–0.36 m | Across 4 circular-tracking tests |
| Simulation overhead | ~9.8–16.9% | WN&CNets vs CPC in sim |
| Latency delay modeled | d = 0,1,2,3 steps | System delay compensation |
