# Improving Drone Racing Performance Through Iterative Learning MPC

- **URL**: https://arxiv.org/abs/2508.01103
- **Authors**: Haocheng Zhao, Niklas Schlüter, Lukas Brunke, Angela P. Schoellig (University of Toronto / UTIAS DSL lab)
- **Year**: 2025 (submitted August 1, 2025; accepted for IROS 2025)
- **Venue**: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025 — Oral presentation

---

## Key Contribution

This paper extends **Learning Model Predictive Control (LMPC)** (Rosolia & Borrelli 2017) to the drone racing domain with three specific innovations that address failure modes of vanilla LMPC when applied to fast, constrained quadrotor racing:

1. **Adaptive cost function** that dynamically weights time-optimal behavior against centerline adherence using gate-proximity-based sigmoid weighting
2. **Shifted local safe set** that prevents the LMPC from learning a degenerate shortcutting trajectory by injecting mirrored states on the opposite side of the centerline
3. **Cartesian-based arc-length formulation** that avoids the singularities and integration drift of Frenet-frame parametrization, using cubic Hermite interpolation and k-d tree lookup for efficient arc-length queries

The paper is directly relevant to our system as the first demonstrated application of iterative learning MPC to real quadrotor racing, achieving a 60.85% lap time reduction over conservative baselines and 6.05% improvement over the state-of-the-art MPCC++ controller. The LMPC framework provides theoretical guarantees of non-decreasing performance improvement across iterations, making it fundamentally more principled than our current ILC offset table approach.

**Important note on Q-filters**: This paper does NOT use a Q-filter in the classical ILC sense. LMPC is a fundamentally different framework from standard ILC—there is no explicit Q-filter, no learning gain `L`, and no trial-to-trial correction signal. Instead, LMPC builds a terminal cost and safe set from previously recorded trajectories and uses standard MPC online. The connections to Q-filter ILC design are indirect (through the terminal cost regularization that provides robustification analogous to Q-filter damping).

---

## Technical Approach

### LMPC background

Standard LMPC (Rosolia & Borrelli 2017) applies to minimum-time problems by:
1. Running an initial feasible (but suboptimal) trajectory
2. Constructing a safe set `SS` from all states visited in successful past iterations
3. Building a terminal cost `V_f(x)` as the minimum cost-to-go estimated from past trajectories
4. At each subsequent iteration, solving MPC with `x_N ∈ SS` and `V_f(x_N)` as terminal constraint and cost

The guarantee is that each new iteration has cost at most equal to the previous iteration—monotonically non-increasing lap time. However, convergence is to a local optimum, not global, and the safe set must be finite and representable in the MPC solver.

### Adaptive cost function

The stage cost at each MPC timestep:
```
h(x, u) = l_t(u) + γ(s) * l_d(x)
```

**Time-optimal term**:
```
l_t(u) = c + ||u||²_R
```
where `c` is a constant cost per timestep (penalizing long lap durations) and `R` penalizes control effort.

**Lateral deviation penalty**:
```
l_d(x) = ||[p(x) - p_c(s)] / R_c(s)||²_{Q_d}
```
where `p(x)` is the drone position, `p_c(s)` is the centerline position at arc-length `s`, `R_c(s)` is the track corridor radius at that arc-length, and `Q_d` weights the deviation.

**Adaptive weight** `γ(s)`:
A mirrored sigmoid function of arc-length that is low in open sections (allowing speed optimization) and high near gates (enforcing gate centerline passage):
```
γ(s) = γ_max * sigmoid-based_gate_proximity(s)
```
The sigmoid increases sharply within a few meters of each gate, creating a "funnel" that drives the drone toward the gate centerline exactly where precision matters, while allowing free optimization of the racing line in between gates.

This adaptive weighting resolves the core tension in drone racing between "go fast" (deviating from centerline to find shorter paths) and "pass through gates" (must hit the gate opening). Without `γ(s)`, LMPC quickly learns to shortcut gates; with a constant `γ`, speed optimization is overly penalized in open sections.

### Shifted local safe set

**Problem with vanilla LMPC safe set**: In a racing scenario with no symmetry constraint, the safe set becomes biased toward one side of the track (whichever side the initialization trajectory happened to approach each gate from). LMPC then repeatedly samples terminal states from this biased set, causing the learned trajectory to progressively hug one wall rather than finding the true racing line.

**Solution**: At each gate `g`, after collecting a new successful iteration:
1. Compute the mean lateral deviation from the centerline of all safe set states near gate `g`
2. Generate "shifted" mirror states on the opposite side: `x_shifted = reflect(x_safe, centerline)`
3. Over-approximate the cost-to-go of shifted states with: `V_f(x_shifted) = V_f(x_safe) + penalty * ||x_shifted - x_safe||²`
4. Add these shifted states to the safe set

This prevents the convex hull of the safe set from collapsing to one side. The ablation study shows that without this modification, collisions occur after 2-3 iterations as the drone learns to cut the corner too aggressively.

### Cartesian arc-length formulation

**Frenet-frame problems**: Standard track-following MPC formulates dynamics in Frenet coordinates `(s, n, ψ_e)` (arc-length, lateral offset, heading error). This introduces:
- Singularities when curvature approaches the inverse of the lateral offset (`κ * n → 1`)
- Integration drift: `s` is computed by integrating `ds/dt = v * cos(ψ_e) / (1 - κn)`, which accumulates error over a lap

**Cartesian solution**: State is `x_aug = [s, x_cartesian]^T` where:
- `x_cartesian = [p, v, q, ω]` are the standard Cartesian drone states (position, velocity, quaternion, angular rate)
- `s` (arc-length) is **not** integrated—it is computed at each MPC step from Cartesian position via nearest-point lookup on the pre-computed centerline

**Arc-length lookup**: A cubic Hermite spline through gate centers provides a smooth centerline. The nearest arc-length for a given Cartesian position is found via:
1. k-d tree search on the discretized centerline to find candidate segments
2. Newton iteration within the segment to find exact `s`

**Computational benefit**: k-d tree lookup reduces arc-length computation from `4.21 ± 0.28 ms` to `0.68 ± 0.17 ms` per query—a 6x speedup that is critical for enabling the 30 Hz MPC update rate.

**Feedback correction**: At each MPC step, `s` is re-estimated from the current Cartesian position measurement, so integration drift does not accumulate. Any one-step estimation error is corrected at the next step.

### MPC formulation details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Control frequency | 30 Hz | vs. 90 Hz for MPCC++ baseline; lower due to LMPC solve time |
| Prediction horizon N | 8 steps | 8/30 ≈ 267 ms look-ahead |
| Safe set cardinality K | 20 | Local K-nearest-neighbor safe set |
| SQP iterations | max 5 | Sequential quadratic programming iterations per solve |
| QP iterations (inner) | max 20 | OSQP iterations per SQP step |
| QP convergence tolerance | 10⁻⁴ | — |
| Discretization frequency | 16-24 Hz | Affects shortcutting vs. acceleration trade-off |
| Solver runtime | 16.66±2.28 ms (N=5,K=20) | OSQP on Intel i7-11700H |
| Solver runtime (N=15,K=20) | 72.24±7.31 ms | Upper feasibility bound for 30 Hz |

The solver uses OSQP (operator splitting QP) via CasADi. The MPC problem is cast as a nonlinear program solved via SQP, where each SQP iteration calls OSQP.

### Quadrotor model

The linearized attitude dynamics used for system identification:
```
φ̈ = α_φ * φ̇ + β_φ * u_φ
θ̈ = α_θ * θ̇ + β_θ * u_θ
ψ̈ = α_ψ * ψ̇ + β_ψ * u_ψ
```

Identified parameters on Crazyflie 2.1:
- `[α_φ, α_θ, α_ψ] = [-6.00, -3.96, 0.0]` (roll/pitch/yaw damping)
- `[β_φ, β_θ, β_ψ] = [6.21, 4.08, 0.0]` (roll/pitch/yaw input gain)
- Mass-normalized thrust model: `T/m = c_T * ||cmd_thrust||²`

Yaw is held constant (ψ = const) throughout all experiments, reducing the effective state to 10 DOF.

### State estimation

An EKF fuses motion capture measurements (200 Hz) with the attitude dynamics model. Motion capture provides ground-truth position and orientation at 200 Hz with sub-millimeter accuracy—no vision-based estimation is used. This represents a significant gap from competition conditions where only onboard sensing is available.

---

## Results

### Simulation (PyBullet, PID initialization)
- PID initialization lap time: 23.55 s
- LMPC after convergence: 8.42 s
- **Improvement: 64.25%**
- No crashes reported after incorporating shifted safe set

### Real-world hardware (Crazyflie 2.1)

| Initialization | Init. lap time | Final lap time | Improvement |
|----------------|----------------|----------------|-------------|
| PID (conservative, 0.5 m/s) | 17.09 s | 6.69 s | **60.85%** |
| MPCC++ (aggressive, μ=0.10) | 6.45 s | 6.06 s | **6.05%** |

Final real-world lap time of **6.06 s** represents near-human expert level performance on the test track.

### Ablation study
- Time-optimal cost only (no deviation penalty): crashes after 2-3 iterations
- Adaptive cost without shifted safe set: converges but suboptimally (wall-hugging)
- Full method: consistent convergence to 8.47 s (simulation), 6.69 s (hardware)
- Arc-length Cartesian vs. Frenet: Frenet showed singularity-induced trajectory corruption in 3/10 experiments; Cartesian had 0/10 failures

---

## Relevance to Our System

This paper is the most directly applicable to our autonomous drone racing context. Key connections:

### Strategic relevance: LMPC as a superior alternative to our current ILC

Our current system uses a static ILC position-offset table (iteration 26): pre-computed gate-by-gate corrections that are applied as fixed offsets to the min-snap trajectory. This approach:
- Has no formal convergence guarantee
- Cannot adjust to sensor noise or environment variation between trials
- Requires careful manual gain tuning to avoid instability

LMPC replaces this with a principled framework that:
- Formally guarantees non-decreasing improvement (Rosolia & Borrelli 2017)
- Uses the drone's actual experienced states (not just gate crossings) to build the safe set
- Naturally handles state and input constraints via MPC

Transitioning from our current ILC offset table to LMPC would require implementing: (a) safe set construction from recorded trajectories, (b) MPC terminal constraint, (c) the adaptive cost function described here.

### Immediate relevance: adaptive gate proximity weighting

Our current per-section ILC applies uniform correction across all parts of a trajectory section. The gate-proximity sigmoid weighting (`γ(s)`) is immediately applicable: weight the ILC correction more heavily near gates (where passing matters) and less in between (where we want speed optimization). This is a 5-10 line modification to our existing ILC update.

### Cartesian arc-length for our trajectory

Our min-snap trajectory uses a parametric time representation. Switching to Cartesian arc-length parametrization (as described here) would give us a more robust "where am I on the track" estimate that doesn't drift. The k-d tree lookup at 0.68 ms overhead is acceptable at 400 Hz control rate.

### Safe set as trajectory initialization library

Even without full LMPC, we could use the safe set concept: maintain a library of previously successful trajectory segments, and when beginning a new race attempt, initialize the trajectory from the best-performing historical segment. This requires less infrastructure than full LMPC but captures much of the benefit.

### Caution: gap from our competition context

This paper uses:
- Motion capture at 200 Hz (we will have onboard vision only in competition)
- Crazyflie 2.1 at modest speeds (slower than competition drone specs)
- Offboard Intel i7 CPU at 30 Hz MPC rate (our onboard compute is more constrained)
- A simple 3-5 gate track (competition may have more complex configurations)

These differences mean the specific MPC parameters (horizon N=8, solver tolerances, arc-length discretization) need re-tuning for our system.

---

## Actionable Takeaways

1. **Adopt gate-proximity sigmoid weighting in existing ILC**: Immediately applicable. Before the next iteration's offset update, multiply the error signal by `γ(s) = 1 + γ_extra * sigmoid((gate_dist_to_nearest - d_threshold) / d_scale)` where `d_threshold ≈ 1-2 m` and `d_scale ≈ 0.5 m`. This biases learning toward gate-crossing precision.

2. **Implement local safe set from past trajectories**: After each successful lap, log all drone states `(position, velocity, attitude)` along with their lap-time cost-to-go. Build a k-d tree over these states. When initializing the next lap's trajectory, allow the MPC terminal constraint to be satisfied by nearest-neighbor states in this safe set.

3. **Switch to Cartesian arc-length parametrization**: Replace our current time-parametrized trajectory lookup with arc-length lookup using cubic Hermite spline through gates + k-d tree. This eliminates the integration drift that currently causes gate timing errors late in the lap.

4. **N=8 horizon at 30 Hz = 267 ms look-ahead**: This is our target MPC horizon configuration. At 100 Hz control rate, N=8 gives 80 ms look-ahead—may need to increase to N=26 (260 ms) to match the paper's effective look-ahead.

5. **Use SQP + OSQP solver infrastructure**: CasADi + OSQP is the specific solver stack used. If we implement MPC, this is the validated combination. Max 5 SQP iterations, 20 OSQP iterations, tolerance 10⁻⁴.

6. **Shifted safe set to prevent shortcutting**: When building the safe set, always include mirrored states on the opposite side of the centerline at each gate, weighted by quadratic distance penalty. This prevents convergence to degenerate corner-cutting solutions.

7. **System identification via linear attitude model**: The `[α, β]` parametrization for roll/pitch/yaw dynamics is a clean way to identify a simple linearized model from flight data. We should run the same identification procedure on our specific drone platform before implementing LMPC.

---

## Limitations & Caveats

1. **No onboard sensing**: The entire paper uses motion capture. All of the tracking error numbers and convergence behavior depend on having near-perfect state estimation. In competition, our EKF-based state estimation will introduce additional uncertainty that is not modeled in the LMPC framework.

2. **30 Hz control rate is low**: For agile drone racing, 30 Hz MPC may be insufficient for fast maneuvers. The paper's hardware results (6.06 s lap) involve relatively gentle flight on the Crazyflie; a faster, heavier racing drone at higher speeds requires higher control rates. The 30 Hz was necessary due to LMPC solve time.

3. **Crazyflie is not a racing drone**: The Crazyflie 2.1 has a mass of ~27 g and max speed ~3 m/s in normal operation. Competition drones are ~250-500 g with 10-20 m/s capability. The dynamics scaling (higher speed = stronger aerodynamic drag, faster attitude dynamics) will require re-tuning all parameters.

4. **Local convergence only**: The paper is explicit that LMPC converges to a local optimum. The initialization trajectory strongly determines the final converged solution. Poor initialization (e.g., from a controller that takes a very non-optimal path) may converge to a poor local optimum.

5. **No Q-filter**: This framework does not address Q-filter design at all. If we want to combine LMPC with Q-filter robustification (e.g., to handle model uncertainty in the LMPC's internal model), we must look to other literature (e.g., Longman 2019, Schoellig 2012).

6. **Safe set size and memory**: At 30 Hz with 6-7 second laps, each iteration adds ~200 states to the safe set. With K=20 local neighbors, the k-d tree lookup is fast, but after 50+ iterations the safe set becomes large. Memory management and safe set pruning are not addressed.

7. **Gate detection not included**: The paper assumes perfect knowledge of gate positions via motion capture and a known track layout. In real competition, gate pose estimation from cameras introduces uncertainty that must be propagated through the LMPC.

---

## Key Parameters / Constants

| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| Control frequency | 30 | Hz | LMPC update rate |
| Prediction horizon N | 8 | steps | 8/30 = 267 ms look-ahead |
| Safe set cardinality K | 20 | neighbors | Local k-NN safe set |
| SQP iterations | max 5 | — | Per MPC solve |
| QP iterations | max 20 | — | OSQP inner loop |
| QP convergence tolerance | 10⁻⁴ | — | OSQP |
| Solver runtime (N=5,K=20) | 16.66 ± 2.28 | ms | Intel i7-11700H |
| Solver runtime (N=15,K=20) | 72.24 ± 7.31 | ms | Intel i7-11700H |
| MoCap rate | 200 | Hz | State estimation input |
| Discretization frequency | 16–24 | Hz | Trajectory knot points |
| k-d tree lookup time | 0.68 ± 0.17 | ms | vs. 4.21 ms naive |
| Roll damping α_φ | -6.00 | s⁻¹ | Crazyflie 2.1 identified |
| Pitch damping α_θ | -3.96 | s⁻¹ | Crazyflie 2.1 identified |
| Roll input gain β_φ | 6.21 | s⁻² | Crazyflie 2.1 identified |
| Pitch input gain β_θ | 4.08 | s⁻² | Crazyflie 2.1 identified |
| Lap time improvement (PID init) | 60.85 | % | Real hardware |
| Lap time improvement (MPCC++) | 6.05 | % | Real hardware |
| Final hardware lap time | 6.06 | s | MPCC++ baseline |
| Track size | 3–5 gates | — | Small indoor track |
| γ_max (gate proximity weight) | Not stated explicitly | — | Tunable hyperparameter |
| Gate approach distance threshold | ~1–2 m (estimated) | m | Sigmoid activation region |
