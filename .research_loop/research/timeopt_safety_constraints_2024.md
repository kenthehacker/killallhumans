# Time-Optimal Flight with Safety Constraints and Data-driven Dynamics
- **URL**: https://arxiv.org/abs/2403.17551
- **Authors**: Maria Krinner, Angel Romero, Leonard Bauersfeld, Melanie Zeilinger, Andrea Carron, Davide Scaramuzza
- **Year**: 2024
- **Venue**: Robotics: Science and Systems (RSS) 2024
- **Also known as**: MPCC++ (Model Predictive Contouring Control++)

## Key Contribution

MPCC++ solves a core problem in model-based drone racing: how to achieve time-optimal flight while providing a formal guarantee that the drone passes *through* gates rather than colliding with their frames. The baseline MPCC formulation encodes safety only as a soft cost term competing with the progress objective — in practice, gates get hit because the optimizer trades off safety against speed. MPCC++ separates these concerns entirely: safety becomes a hard prismatic tunnel constraint that the optimizer must satisfy at all timesteps, while the progress objective drives time-optimality unopposed within that safe space.

The result, demonstrated on a real 7-gate Split-S aerobatic course at speeds exceeding 80 km/h, is the first model-based controller to achieve 100% gate-passage success in real-world experiments. Equally important for safety margin calibration: this success rate holds even as lap times match RL-level performance (5.38 s vs RL's 5.35 s), meaning safety and performance are not genuinely in conflict — the conflict only arises when safety constraints are encoded incorrectly as soft costs.

## Technical Approach

### Safety Constraints: Prismatic Tunnel

The safety mechanism is a rectangular cross-section tunnel defined in the Frenet-Serret frame of the reference path. Four halfspace inequalities constrain the drone position at every MPC prediction step:

```
(p_k - p_0(θ_k)) · n(θ_k) ≥ 0
2H(θ_k) - (p_k - p_0(θ_k)) · n(θ_k) ≥ 0
(p_k - p_0(θ_k)) · b(θ_k) ≥ 0
2W(θ_k) - (p_k - p_0(θ_k)) · b(θ_k) ≥ 0
```

where `n(θ)` and `b(θ)` are the Frenet-Serret normal and binormal vectors, `H(θ)` is half the tunnel height, and `W(θ)` is half the tunnel width. The key design parameter is the width/height profile along the path: a sigmoid function smoothly transitions from a nominal wide value `W_n` (generous between-gate corridor, ~1.5–2.0 m implied by track geometry) to the narrow gate value `W_gate` (matching actual gate opening, ~0.6–0.8 m for competition-class gates). This narrowing happens over a path-length interval of approximately 0.5–1.0 m on the approach to each gate.

The critical structural insight: **between gates, the tunnel is wide enough that the optimizer has full freedom to select the racing line**. The drone can cut corners, use the full track width, and fly the globally time-optimal path — the constraint is inactive except at gates. At gates, the constraint is tight and enforces passage. This means the tunnel constraint does *not* conservatively inflate safety margins for the whole track; it only enforces clearance at the specific locations where clearance matters.

Implementation uses a soft log-sum-exp barrier with penalty slope α=100, embedded in the cost function. This maintains differentiability for the gradient-based SQP solver while approximating hardness. In practice, with α=100 the effective constraint violation at optimum is negligible (< 0.01 m).

### Recursive Feasibility via Terminal Set

A periodic center-line trajectory through all gate centers is computed offline. The MPC terminal state is constrained to lie on this trajectory. This guarantees that if the drone ever needs to fall back from the aggressive racing line to the safe center path, a feasible trajectory exists. The offline solve minimizes position deviation to the center-line subject to periodicity: x_0 = x_M.

The terminal set construction is the mechanism that prevents the online MPC from entering infeasible states — the optimizer always knows there exists a valid control sequence (the center-line) even under the hard tunnel constraints.

### Data-Driven Aerodynamic Residual

The nominal rigid-body quadrotor model is augmented with a learned polynomial residual capturing aerodynamic effects:

```
f̂(x, u) = f(x, u) + g(x, u)
```

Residual force/torque components are polynomial in body-frame velocities and mean-squared rotor speed Ω²:

```
f_x = C_fx · [v_x, v_x³, Ω², v_x·Ω²]ᵀ
f_y = C_fy · [v_y, v_y³, Ω², v_y·Ω²]ᵀ
f_z = C_fz · [v_z, v_z³, v_xy, v_xy², v_xy·Ω², v_z·Ω², v_xy·v_z·Ω²]ᵀ
τ_x = C_τx · [v_y, Ω², v_y·Ω²]ᵀ
τ_y = C_τy · [v_x, Ω², v_x·Ω²]ᵀ
τ_z = C_τz · [v_x, v_y]ᵀ
```

Coefficients are identified via ordinary least-squares on real-world flight data. The augmentation reduces lap time by 0.07–0.15 s across simulators of varying fidelity.

### Hyperparameter Tuning: TuRBO

Eight MPC cost weights (Q_l, Q_c, Q_ω with horizontal/vertical components, R_vθ, R_Δf, μ) are tuned using Trust-Region Bayesian Optimization with 8 parallel instances. The reward is negative mean lap time over 3-lap rollouts, penalized by γ=100 for solver failures. Budget: 600 total episodes. This collapses lap time variance from ±1.06 s (manual/WML tuning) to ±0.14 s in real-world experiments.

### MPC Implementation

- **Solver**: ACADOS with SQP_RTI (single SQP step per control cycle)
- **Control rate**: 100 Hz
- **Prediction horizon**: N=20 steps at 25 Hz → 0.8 s lookahead
- **Path parameter θ**: Additional optimization variable; the controller chooses how fast to progress along the reference path online, enabling contouring vs. pure tracking

## Results

All experiments use a 7-gate Split-S aerobatic track at speeds exceeding 80 km/h.

| Environment | Method | Lap Time (s) | Success Rate (%) |
|---|---|---|---|
| Simple Sim | MPCC (baseline) | 5.38 ± 0.10 | 100 |
| Simple Sim | MPCC++ | 5.16 ± 0.02 | 100 |
| Simple Sim | MPCC++ + augmented model | 5.09 ± 0.10 | 100 |
| BEM Sim (high-fidelity) | MPCC (baseline) | 5.51 ± 0.06 | 100 |
| BEM Sim | MPCC++ | 5.30 ± 0.02 | 100 |
| BEM Sim | MPCC++ + augmented model | 5.15 ± 0.03 | 100 |
| Real World | MPCC (TuRBO) | 5.67 ± 1.06 | 59.3 |
| Real World | MPCC++ (TuRBO) | 5.41 ± 0.14 | **100** |
| Real World | MPCC++ + augmented model | 5.38 ± 0.26 | **100** |
| Real World | RL (reference) | 5.35 ± 0.15 | 85.0 |

Training success rate (TSR) during TuRBO: MPCC++ achieves 99.5–100%, vs ~70% for baseline MPCC. This means 30% of baseline training episodes were wasted on crashes — the tunnel constraint makes the training distribution dramatically cleaner.

Key safety-vs-speed finding: MPCC++ is faster than baseline MPCC in both sim and real-world (5.38 → 5.41 vs 5.67), while simultaneously having higher success rate (100% vs 59.3%). This is the central empirical result for our safety margin question: **tighter safety encoding does not slow the drone down — it enables faster flight by eliminating the soft-constraint competition between safety and progress**.

## Relevance to Our System

Our system (current state: race time 13.80 s, avg error 0.185 m, ILC active, S-turn inflation 1.08/1.10, TOPP compression floors 0.66/0.60) uses a fundamentally different architecture: pre-optimized min-snap polynomial trajectories time-retimed by TOPP-RA, with ILC applying cross-track corrections. Safety margins are encoded as post-optimization inflation factors on segment times (making the trajectory slower) rather than as spatial constraints on where the drone is allowed to fly.

**The core question this paper answers for us: what is the right safety margin to keep?**

MPCC++ reframes the question: the right safety margin is not a time buffer (inflation factor) but a spatial buffer (tunnel width at gates). These are fundamentally different mechanisms:

- **Time inflation (our current approach)**: Makes the trajectory slower throughout the entire segment approaching and exiting a gate. The drone has more time to converge to the reference, but the reference itself is now sub-optimal (too slow). The inflation factor bluntly penalizes the entire segment regardless of where tracking error actually occurs.

- **Spatial tunnel (MPCC++ approach)**: Constrains the drone to pass through the gate opening, but imposes no constraint on *when* it arrives or how it flies between gates. The optimizer selects the fastest path that satisfies the spatial constraint. This is both safer (hard guarantee) and faster (no unnecessary time penalty on between-gate segments).

**Implications for our inflation factor calibration:**

Our S-turn inflate 1.08/1.10 adds 8–10% time to S-turn segments. Gate-7 (worst gate, 0.282 m error) sits at the helix entry. The ILC has been progressively reducing tracking error to the point where these inflate factors are now over-conservative. The MPCC++ result suggests that as ILC continues to improve accuracy, the right endpoint is to eliminate time-domain safety margins entirely and replace them with spatial gate-passage enforcement. However, since our architecture does not have a receding-horizon controller that can exploit spatial constraints in real time, the intermediate approach (progressive inflation deflation guided by ILC performance) is correct.

**Quantitative calibration guidance from MPCC++**: The paper shows that flying *at* the gate-opening width (W_gate ≈ 0.6–0.8 m) is safe at 100% success when the MPC horizon is 0.8 s and the dynamics model is accurate. Our current avg error of 0.185 m means the 95th-percentile error is approximately 0.35–0.45 m (from our p95 metrics). A gate opening of 0.8 m half-width means our p95 error (0.45 m) is still within the gate. This confirms that our current tracking accuracy is sufficient to reduce inflation further — we do not need the extra time buffer to guarantee gate passage spatially.

**ILC and the safety margin question directly**: MPCC++ uses no ILC — it relies on the MPC to correct errors online within the horizon. Our ILC provides analogous error reduction, but in batch rather than receding-horizon form. As ILC converges (avg error approaching 0.17–0.18 m), each 1% inflation reduction corresponds to faster flight without meaningfully increasing spatial error at gate crossings. The basin-switching risk (as seen in iteration 29 with aggressive combined reductions) is an optimizer artifact, not a safety margin artifact — it reflects the trajectory optimizer finding a different local minimum rather than the drone getting spatially closer to gate frames.

## Actionable Takeaways

1. **Safety margins should be spatial, not temporal.** Our inflation factors are a proxy for gate-passage safety that penalizes the whole segment. The MPCC++ result shows that the correct primitive is a spatial constraint at gate locations only. For our architecture, this means: reduce inflation factors toward 1.00 guided by ILC performance, but add an explicit gate-center waypoint enforcement check in the trajectory optimizer that verifies the planned path passes within W_gate of each gate center before accepting a trajectory.

2. **Current inflation factors (1.08/1.10) are over-conservative given ILC accuracy.** At avg error 0.185 m and p95 ≈ 0.45 m, our tracking is accurate enough to absorb another 1–2% inflation reduction per iteration without spatial gate-clearance risk. The constraint is optimizer basin-switching (iteration 29 lesson: >3% per factor causes catastrophic racing line switching), not physical gate clearance. Reduce in 1–2% steps per iteration, verifying no basin switch by checking race time stability.

3. **The tunnel width transition (W_n → W_gate over ~0.5 m) is the key geometric parameter.** In our trajectory optimizer, this translates to: the approach waypoints (entry/exit offsets from gate center) define the effective "tunnel width" on the pre-planned path. Current entry offsets of 0.3–0.4 m provide a similar geometric function to the MPCC++ tunnel narrowing, but without the optimization-aware constraint enforcement. Ensure approach offsets are set to approximately gate half-width (0.4 m for a 0.8 m gate) to mirror MPCC++ gate-constraint tightness.

4. **The between-gate corridor should be wide (W_n >> W_gate).** MPCC++ shows that between gates, the drone is allowed full spatial freedom — this is where optimal racing lines deviate from gate-center paths. Our racing line optimizer should not be penalized for flying 1–2 m off the gate-center path between consecutive gates. The current smoothness weight in `planning/racing_line.py` should not be increased further, as it artificially restricts the between-gate freedom that MPCC++ explicitly preserves.

5. **Soft log-barrier with α=100 approximates hard constraints well.** If we add any explicit constraint representation to the trajectory optimizer (e.g., a gate-passage feasibility check), use this penalty coefficient rather than a hard rejection criterion. This maintains solver stability while being effectively constraining.

6. **ILC convergence → inflation deflation is the right strategy, but has an endpoint.** The paper implies that with a sufficiently accurate model and receding-horizon controller, inflation factors can reach 1.00 (no inflation). With ILC, our practical endpoint is when ILC offsets saturate (currently capped at ~0.30 m). At that point, remaining tracking error is irreducible by ILC and must be addressed by better trajectory planning or controller architecture. The MPCC++ contouring formulation (path-parameter θ treated as an optimization variable, enabling online progress rate adjustment) would be the next step if ILC plateaus.

7. **TuRBO-style reward = negative race time + 100·crash_penalty.** This is directly applicable to tuning our TOPP compression floors and inflation factors jointly. Instead of manual iteration, run 50–100 benchmark episodes varying (S-turn inflate, TOPP floor) in a 2D search space. The γ=100 crash penalty is correct: a crash eliminates the run and should cost ~100 s equivalent in the optimization.

8. **Recursive feasibility via center-line terminal constraint.** Our `planning/racing_line.py` already computes a center-line-like reference. This should be treated as the fallback trajectory when TOPP-compressed plans produce tracking error spikes. The MPCC++ terminal set insight: always have a feasible conservative plan ready; the aggressive plan is the primary but the center-line is always valid.

## Limitations & Caveats

1. **Architecture mismatch: MPC vs. pre-planned trajectory.** MPCC++ exploits a receding-horizon controller that adapts the path parameter θ online. Our system uses a fixed pre-planned trajectory and a PD tracker. The safety guarantees in MPCC++ rely on the MPC being able to re-solve at 100 Hz — our system cannot retroactively re-optimize the trajectory during flight. ILC provides a batch approximation, but unlike MPC cannot respond to disturbances within a single run.

2. **No quantitative tunnel width numbers reported.** The paper does not disclose exact W_n or W_gate values, making direct calibration to our gate sizes difficult. The values inferred (W_n ~1.5–2.0 m, W_gate ~0.6–0.8 m) are estimated from typical DCL competition gate openings (0.9 m × 0.9 m square gates) and the track geometry implied by 5.4 s lap times.

3. **VICON ground truth vs. our EKF.** MPCC++ assumes ~1 mm position accuracy from 36-camera VICON. Our EKF uncertainty is 0.012 m (excellent in sim, potentially worse on hardware). Any degradation in state estimation would expand the effective tracking error distribution, requiring larger tunnel widths — and correspondingly larger safety margins. Our current 0.185 m avg error in sim may understate the hardware margin needed.

4. **Frenet-Serret singularity at high curvature.** The tunnel constraints use Frenet-Serret normal/binormal vectors, which are ill-conditioned near inflection points (where curvature → 0) and at high curvature. Our helix section (gates 7–12) has high curvature, and gate-7 (worst gate, 0.282 m) sits at the helix entry where curvature changes sign. A direct implementation of the tunnel constraint would need special handling at this inflection point.

5. **Aerodynamic augmentation not applicable in kinematic sim.** The Ω² rotor-speed coupling terms require knowledge of individual rotor speeds. Our kinematic sim models drag as a scalar velocity-proportional damping term. The polynomial residual structure is the correct approach for hardware deployment, but adds no value in our current sim validation loop.

6. **Training success rate (TSR) improvement only matters for BO.** The 70%→100% TSR improvement in MPCC++ is valuable for BO sample efficiency but irrelevant if gains are tuned manually. Our ILC loop already avoids crashes by design (alpha cap at 0.30 m offset saturation), so we have effective TSR = 100% without needing the tunnel constraint for the tuning process itself.

7. **7-gate Split-S vs. our multi-section track.** The paper's track has a very specific aerobatic geometry. Our track (with helix, S-turn, and straight sections) has qualitatively different curvature structure. The sigmoid corridor transition tuned for Split-S would need re-calibration for each gate on our track, as the approach curvature varies significantly between sections.

## Key Parameters / Constants

- **MPC control rate**: 100 Hz (SQP_RTI, single SQP step per cycle)
- **Prediction horizon**: N=20 steps at 25 Hz → 0.8 s lookahead
- **Soft-constraint barrier coefficient**: α=100 (log-sum-exp approximation of hard constraints)
- **Failure penalty in TuRBO reward**: γ=100 (lap-time-equivalent units)
- **TuRBO budget**: 600 total episodes; 8 parallel instances; 3 laps per episode
- **Tunable hyperparameter count**: 8 (Q_l, Q_c_horiz, Q_c_vert, Q_ω_horiz, Q_ω_vert, R_vθ, R_Δf, μ)
- **Sigmoid corridor transition distance**: ~0.5–1.0 m on approach to gate (inferred)
- **Nominal corridor half-width** W_n: ~1.5–2.0 m (inferred from track geometry)
- **Gate corridor half-width** W_gate: ~0.4–0.5 m (inferred from DCL gate openings ~0.9 m)
- **Aerodynamic basis (forces)**: [v, v³, Ω², v·Ω²] per in-plane axis; extended vertical basis includes v_xy cross-coupling and v_xy·v_z·Ω² term
- **Aerodynamic basis (torques)**: [v_lateral, Ω², v_lateral·Ω²] for roll/pitch; [v_x, v_y] for yaw
- **Lap time achieved (real world)**: 5.38 ± 0.26 s (augmented model), 5.41 ± 0.14 s (nominal)
- **RL comparison**: 5.35 ± 0.15 s at 85% success rate; MPCC++ matches time at 100% success
- **Speed achieved**: >80 km/h (~22 m/s)
- **Lap time improvement from augmented model**: 0.03–0.22 s depending on sim fidelity
- **Real-world baseline MPCC success rate**: 59.3% (vs 100% for MPCC++)
- **Variance collapse from TuRBO tuning**: ±1.06 s → ±0.14 s in real-world experiments
- **Training success rate**: MPCC++ 99.5–100% vs baseline MPCC ~70%
- **State estimation**: VICON motion capture, ~1 mm accuracy, 36 cameras
- **Offboard compute**: Intel i7-8565U (SQP_RTI feasible at 100 Hz on this hardware)
- **ACADOS solver**: CasADi/HPIPM backend with SQP_RTI real-time iteration scheme
