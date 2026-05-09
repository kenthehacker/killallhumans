# Perception-Aware Time-Optimal Planning (ETH 2026)

- **URL**: https://arxiv.org/abs/2603.04305
- **Authors**: Chao Qin, Jiaxu Xing, Rudolf Reiter, Angel Romero, Yifan Lin, Hugh H.-T. Liu, Davide Scaramuzza
- **Affiliations**: University of Toronto Institute for Aerospace Studies; Robotics and Perception Group, University of Zurich
- **Year**: 2026
- **Venue**: arXiv:2603.04305v1 (March 4, 2026) — Robotics/CS

---

## Key Contribution

The paper's central thesis is that **dynamically feasible, time-optimal trajectories can fail in closed-loop execution because they degrade the quality of visual state estimation** — and that this failure mode should be addressed at the planning stage rather than patched in the estimator.

Their unified framework is the first to jointly optimize:
1. Time-optimal trajectory generation (min-time, physically constrained)
2. Visual perception quality via an information-theoretic metric (FIM-based position uncertainty)
3. Camera field-of-view (FOV) feasibility as a soft constraint
4. Look-ahead gaze alignment to keep future waypoints in view

The practical payoff is dramatic: closed-loop success rate improved from **55% to 100%** on a challenging Split-S track, while maintaining near-time-optimal performance (worst case +17% lap time overhead for the fully perception-aware variant).

---

## Technical Approach

### Three-Step Planning Pipeline

**Step 1 — Time-Optimal Seed:** A polynomial trajectory is generated using differential flatness (TOGT framework). This produces a dynamically feasible time-optimal reference without perception awareness.

**Step 2 — Segment Density Refinement:** The trajectory is re-parameterized to concentrate shooting nodes in high-curvature regions, achieving a <5% optimality gap compared to a naive dense discretization. This dramatically reduces compute while preserving solution quality.

**Step 3 — Perception-Aware NLP Refinement:** The trajectory is re-optimized via direct multiple-shooting (IPOPT + CasADi) with additional cost terms for perception objectives. The seed from steps 1–2 warm-starts the solver.

### Quadrotor Dynamics

The state vector is: **x** = [position (3), quaternion (4), velocity (3), body rates (3), rotor thrusts (4)] — 17 states total.

Constraints enforced:
- Body rate limits: |**ω**| ≤ **ω**_max
- Rotor thrust bounds: f_min ≤ f ≤ f_max
- Thrust rate limits (motor dynamics)
- Waypoint spherical tolerance regions δᵢ
- Polyhedral gate constraints: **A**ᵢ**p** ≤ **b**ᵢ (gate represented as convex polytope)

Integration uses Runge-Kutta 4 with dynamic sampling time adjustment.

### Position Uncertainty Metric (FIM-Based)

This is the paper's most novel technical contribution. They derive a **rotation-invariant, closed-form position uncertainty bound** from Fisher Information.

**Camera model:** Fisheye (equidistant projection). Angle of incidence: θ = arctan2(√(X²+Y²), Z).

**Key insight:** Under isotropic pixel measurement noise, the Fisher Information Matrix (FIM) for position reduces to a form that eliminates the rotation matrix — decoupling camera orientation from uncertainty computation. This makes the metric fast to evaluate inside an NLP.

The FIM summed over all visible gate landmarks is:

    I_FIM(p) = Σ v_{i,j} · A_{i,j}^T · Σ_ρ · A_{i,j}

where v_{i,j} is a visibility weighting and A_{i,j} encodes the bearing geometry.

**Fast Evaluation:** For computational tractability, they combine:
- Multi-landmark geometry via landmark centroids (captures relative viewing angle)
- Single-landmark distance estimation using physical gate size L and observed pixel size l

**PUM Cost:** ℒ_PUM = -log det(I_FIM) — maximizes the determinant of the FIM, which is equivalent to minimizing the volume of the localization uncertainty ellipsoid (D-optimal design criterion).

### Camera FOV Constraints

FOV is enforced as **soft constraints** with slack variables **S**:

    ℒ_FOV = w_FOV · (1^T · S + ½||S||₂²)

The slack formulation avoids infeasibility (hard FOV constraints would make many trajectories infeasible during aggressive banked turns) while heavily penalizing constraint violation. Camera FOV parameters used: horizontal α_max = 128.1°, vertical β_max = 72.2°, minimum depth Z_min = 0.3 m.

### Look-Ahead Gaze Alignment

To ensure upcoming gates are centered in the camera frame before arrival, they add a look-ahead term that aligns the camera optical axis z_c with the bearing vector **b** toward a future trajectory point at horizon t_LA:

    ℒ_LA = -w_LA · exp(-λ_LA · arccos⁴(⟨b, z_c⟩))

The exp-arccos⁴ shaping creates a smooth, differentiable cost that is zero when perfectly aligned and grows steeply off-axis.

### Motion Regulation

A jerk regularization term reduces camera vibration during flight:

    ℒ_MR = w_MR · ||j||₂², w_MR = 1e-7

This is critical because IMU/camera fusion degrades when vibration introduces aliasing.

### Model Predictive Contouring Controller (MPCC)

The trajectory tracker uses an MPCC that explicitly decomposes position error into two orthogonal components at each shooting node k:

**Contouring (lateral) error:**
    e_k^c = (I₃ - t_k^d (t_k^d)^T) e_k

This is the component of position error perpendicular to the local path tangent t_k^d.

**Progress (longitudinal) error:**
    e_k^l = (e_k^T t_k^d) · t_k^d

This is the component along the path — essentially lead/lag.

**Why this decomposition matters:** Standard MPC penalizes total Euclidean error equally in all directions. At high speed, a drone naturally cuts corners (undershoot in the lateral direction while making rapid progress). MPCC applies **higher weight to contouring error** to enforce lateral precision, while allowing some flexibility in progress error. This prevents corner-cutting without over-constraining the longitudinal motion, which would cause speed oscillations.

The MPCC OCP minimizes a weighted sum of:
- Contouring error (highest weight)
- Progress error
- Orientation error
- Velocity, angular velocity
- Thrust and thrust rates

Solver: acados with HPIPM quadratic programming solver, sampling interval Δt_mpc = 0.02 s (50 Hz).

---

## Results

### Benchmark Track: Split-S

The Split-S is a classic aerobatic maneuver requiring a banked half-roll followed by a descending half-loop — exactly the kind of aggressive geometry that stresses both dynamics and perception.

### Time-Optimality Trade-off

| Configuration | Lap Time | vs. Pure Time-Optimal |
|---|---|---|
| Pure time-optimal (TOGT) | 13.33 s | baseline |
| With PUM only | 14.20 s | +6.5% |
| With FOV only | 14.41 s | +8.1% |
| With Look-Ahead only | 14.62 s | +9.7% |
| Full (LA+FOV+PUM) | 15.65 s | +17.4% |

The full perception-aware variant costs ~1.5 seconds per lap on a ~13-second track — a modest penalty given the reliability improvement.

### Real-World Flight (Split-S Track)

- Maximum speed: **9.8 m/s**
- Average tracking error: **0.07 m**
- Peak tracking error: **0.23 m**
- Closed-loop success rate: **100%** (with perception constraints)
- Closed-loop success rate without perception: **55%**

The 0.07 m average tracking error at 9.8 m/s is exceptional. For context, many published systems achieve 0.3–1.0 m at comparable speeds. This performance is enabled by:
1. High-quality trajectory reference (perception-aware, smooth)
2. MPCC's lateral/progress decomposition preventing corner-cutting
3. 50 Hz control loop with HPIPM solver

### Computational Performance

| Method | Planning Time |
|---|---|
| Their method | 68.63 s |
| Fast-Fly (baseline) | 256.48 s |

The 3.7x speedup vs. Fast-Fly comes from the segment density refinement step (Step 2), which concentrates nodes intelligently rather than using uniform high-density sampling.

### Position Uncertainty Validation

Their analytical FIM-based uncertainty estimate was validated against sampling-based PnP ground truth — showing close agreement. This confirms the theoretical derivation is practically sound and not just a planning heuristic.

---

## Relevance to Our System

This paper is directly relevant to the AI Grand Prix system in three ways:

**1. Perception-Aware Trajectory Planning:** Our current `trajectory_optimizer.py` (min-snap polynomial) generates smooth trajectories but does not consider camera visibility of gates. On sharp turns, our EKF likely loses gate visibility, causing drift. Incorporating even a simplified FOV soft constraint into our planning step would improve localization continuity.

**2. MPCC for Gate Racing:** Our `mpc_tracker.py` uses geometric SE(3) control. Replacing or augmenting it with MPCC's contouring/progress error decomposition would directly address corner-cutting behavior — a likely contributor to our current tracking error. The decomposition is not complex to implement and maps naturally onto our existing path-following architecture.

**3. FIM-Based Gate Visibility:** Our `gate_pnp.py` and `gate_tracker.py` perform gate-based drift correction, but the trajectory planner doesn't try to maximize the informativeness of gate observations. Adding a lightweight FIM term (using the simplification in the paper) to our trajectory optimization could improve EKF correction frequency and quality.

**Current gap:** Our system has 0.07m aspirational tracking error target vs. their 0.07m achieved — which indicates the target is feasible, but our current implementation is far from it.

---

## Actionable Takeaways

1. **Add FOV soft constraints to trajectory optimizer.** Implement the slack-variable FOV penalty ℒ_FOV in `planning/trajectory_optimizer.py`. Model each gate's polytope (already done) and add bearing-angle bounds for the planned camera pose.

2. **Implement MPCC in the tracker.** Modify `control/mpc_tracker.py` to decompose position error into contouring (perpendicular) and progress (parallel) components. Weight contouring error 3–5x higher than progress. This alone should reduce corner-cutting at high speed.

3. **Add look-ahead gaze term.** In the trajectory optimizer, add ℒ_LA to pre-align the drone's camera axis toward upcoming gates. This requires knowing the camera mounting orientation relative to body frame (already implicit in `gate_pnp.py`).

4. **Use segment density refinement.** Our current trajectory sampling is uniform. Concentrating nodes at high-curvature segments (near gates) would improve both solver convergence and tracking controller resolution where it matters most.

5. **Validate EKF with FIM metric.** Use the FIM determinant as a diagnostic metric during simulation runs — log it per gate to identify which gates cause the most localization uncertainty. This is a zero-cost analysis using existing `gate_tracker.py` state.

6. **Target the 0.07m tracking error.** The paper demonstrates it is achievable at 9.8 m/s with MPCC + perception-aware planning. Our current aspirational target matches this; the path to get there is now clear.

---

## Limitations & Caveats

- **Planning latency (68s) is offline-only.** The full perception-aware NLP refinement is not real-time capable. It must be pre-computed for known tracks. For our competition (known track VQ1), this is acceptable.
- **Assumes known gate positions.** The FIM metric and FOV constraints require known gate locations. In the competition setting, gates are pre-surveyed, so this holds. Dynamic gate detection during flight is not addressed.
- **Fisheye camera assumed.** The FIM derivation uses an equidistant projection model. Our system uses standard perspective cameras (gate_pnp.py). The FIM simplification may differ slightly, though the approach is directly portable.
- **+17% lap time overhead.** The full perception-aware configuration trades 1.5s per lap for reliability. For a 500K prize race this is well worth it — but if the competition penalizes lap time heavily, a tuned partial configuration (PUM-only at +6.5%) may be preferable.
- **Real-world platform is small (1.0 kg, 125mm arm).** Our platform specs may differ. The gains and noise parameters need re-tuning for our specific drone.
- **No wind model.** Aerodynamic effects are modeled (thrust-to-drag) but atmospheric turbulence is not addressed. The 100% success rate may degrade in outdoor conditions.

---

## Key Parameters / Constants

| Parameter | Value | Purpose |
|---|---|---|
| MPC sampling interval Δt_mpc | 0.02 s (50 Hz) | MPCC control loop |
| Initial shooting node interval | 2 ms | Dense seed for Step 2 |
| NLP convergence tolerance ε_tol | 1e-5 | IPOPT stopping criterion |
| Max NLP iterations | 5000 | IPOPT limit |
| Motion regulation weight w_jerk | 1e-7 | Jerk regularization |
| Sharpness parameter λ_v | 10 | Sigmoid transition sharpness |
| Position uncertainty threshold | 2 m | Alarm threshold for EKF |
| Camera H-FOV α_max | 128.1° | Fisheye horizontal limit |
| Camera V-FOV β_max | 72.2° | Fisheye vertical limit |
| Minimum depth Z_min | 0.3 m | FOV near-plane |
| Image noise σ_u/v | 10 pixels | FIM noise model |
| Sim drone mass | 0.7 kg | RPG quadrotor |
| Real drone mass | 1.0 kg | Experiment platform |
| Arm length (both) | 0.125 m | Rotor geometry |
| Sim inertia J_diag | [2.4, 1.8, 3.7] g·m² | Moment of inertia |
| Real inertia J_diag | [2.5, 2.1, 4.3] g·m² | Moment of inertia |
| Max speed achieved | 9.8 m/s | Split-S real flight |
| Avg tracking error | 0.07 m | Split-S real flight |
| Peak tracking error | 0.23 m | Split-S real flight |
| Planning time (their method) | 68.63 s | Offline NLP solve |
| Planning time (Fast-Fly) | 256.48 s | Comparison baseline |
| Closed-loop success (w/ perception) | 100% | Reliability metric |
| Closed-loop success (w/o perception) | 55% | Reliability metric |
| Pure time-optimal lap | 13.33 s | Lower bound reference |
| Full perception-aware lap | 15.65 s | +17.4% overhead |
