# TOPPQuad: Dynamically-Feasible Time-Optimal Path Parametrization for Quadrotors

- **URL**: https://arxiv.org/abs/2309.11637
- **Authors**: Katherine Mao, Igor Spasojevic, M. Ani Hsieh, Vijay Kumar
- **Year**: 2024
- **Venue**: IROS 2024

---

## Key Contribution

TOPPQuad contributes a time-optimal path parametrization (TOPP) algorithm that respects the **full rigid-body dynamics and individual motor thrust constraints** of a quadrotor, rather than relying on convex relaxations or overly conservative simplifications. The method takes a pre-planned geometric path (from any collision-avoidance planner) and computes the fastest dynamically-feasible speed profile along that path. The core insight is that existing min-snap and min-jerk trajectory methods either (a) ignore motor-level constraints, producing trajectories that saturate actuators at aggressive segments, or (b) apply conservative scaling that leaves significant time on the table. TOPPQuad closes this gap by directly embedding thrust constraints into the optimization.

A secondary contribution is the treatment of **rotational dynamics inside the TOPP formulation**. Most prior TOPP methods for quadrotors ignore the coupling between translational and rotational motion, handling only thrust magnitude bounds. TOPPQuad explicitly integrates quaternion kinematics and angular rate constraints into the optimization, producing speed profiles that are feasible not just translationally but also rotationally. This is particularly important for sharp turns and aggressive maneuvers where the required attitude change rate can saturate body-rate limits even when thrust bounds are satisfied.

---

## Technical Approach

### Problem Setup

The input is a geometric path γ: [0, S_end] → ℝ³ parameterized by arc-length s. The goal is to find a time parametrization χ(t) such that **p**(t) = γ(χ(t)) and the total execution time T is minimized subject to actuator feasibility.

### Square Speed Reparametrization

The algorithm transforms the time-domain problem into a path-domain problem by defining:

```
h(s) := (ds/dt)²   [square speed as a function of arc-length]
```

This change of variables converts the time-minimization objective into:

```
T = ∫₀^S_end 1/√h(s) ds
```

and converts time derivatives into spatial derivatives via d/dt = √h(s) · d/ds. This transformation is the classical TOPP trick (Bobrow 1985), but TOPPQuad extends it to incorporate full quadrotor rigid-body dynamics.

### Quadrotor Dynamics Model

Standard rigid-body dynamics:

```
ṗ = v
v̇ = R e₃ (c/m) + g
q̇ = (1/2) Ω(ω) q
ω̇ = J⁻¹(τ - ω × Jω)
```

where p is position, v velocity, R ∈ SO(3) rotation, q ∈ S³ quaternion, ω body angular rate, c total collective thrust, τ body torque, m mass, J inertia matrix. Motor thrusts u ∈ ℝ⁴ map to collective thrust and torque via:

```
[c, τ]ᵀ = F u
```

where F is the allocation matrix (function of arm length and motor thrust/torque coefficients).

### Constraint Derivation (Eq. 18)

The critical coupling equation relates motor thrusts directly to the speed profile h(s) and its derivative h'(s):

```
m[½ γ'(s) h'(s) + γ''(s) h(s) - g]     [R(s)e₃   0 ]
[J[½ω(s)h'(s) + α(s)h(s)] + ω×Jω h(s)] = [0        I₃] F u
```

where α(s) := ω'(s) is angular acceleration with respect to arc-length. This equation means that for any candidate speed profile {h(s), h'(s)}, there is a unique required motor thrust vector u(s). The feasibility constraint becomes simply:

```
u_min ≤ u(s) ≤ u_max  ∀ s ∈ [0, S_end]
```

### Quaternion Integration

Orientation is propagated along the path using:

```
q'(s) = (1/2) Ω(ω(s)) q(s)
```

with normalized forward Euler updates (Eq. 23) to maintain unit-norm quaternion constraint:

```
q_{i+1} = (q_i + (1/2) Ω(ω_i) q_i Δs) / ‖q_i + (1/2) Ω(ω_i) q_i Δs‖
```

This is a simple first-order projection step, not a full quaternion exponential map, which introduces small integration error but is sufficient at fine discretization.

### Decision Variables and Solver

The full NLP has decision variables: {h(·), h'(·), q(·), ω(·), α(·), u(·)} discretized over N=300 grid points. The problem is solved using IPOPT via the CasADi automatic-differentiation interface. An initial guess is provided by the baseline min-snap trajectory's speed profile.

### Bidirectional Motor Support

The formulation optionally allows u_min < 0, which handles reversible-thrust motor configurations. For standard quadrotors with one-directional motors, u_min = 0 is enforced.

---

## Results

### Simulation Benchmarks (200 randomized 4-waypoint trajectories, 10×10×10 m³)

- **40-50% faster** than α-scaled minimum-snap trajectories
- **10% faster** than prior TOPP methods that use relaxed (not per-motor) thrust constraints
- Consistently maintained motor thrust within bounds, while baselines violated constraints in aggressive segments
- Average speed increase of ~1.8 m/s over unconstrained baseline trajectories

### Hardware Validation (CrazyFlie 2.0, m=32 g, u_max=0.14375 N per motor)

| Trajectory | TOPPQuad | Min-Snap (1 m/s) | Notes |
|---|---|---|---|
| Line | 2.6 s | 3.4 s | 24% faster |
| L-curve | 3.6 s | 4.0 s | 10% faster |
| 3D X-curve | — | — | MS 2 m/s crashed |
| Lissajous | 9.1 s | MS 2 m/s crashed | TOPPQuad succeeded |

The Lissajous and 3D X-curve cases are particularly notable: aggressive min-snap trajectories crashed the hardware, while TOPPQuad trajectories of shorter duration tracked successfully because the speed profile was dynamically consistent.

### Real-World Forest Flight

Applied to a 336 m GPS-logged flight path: TOPPQuad trajectory time 136 s vs. original 236 s, a **43% reduction** with verified thrust feasibility.

### SE(3) Controller Gains Used (Table III)

These gains were used in hardware validation and represent a tuned set for aggressive CrazyFlie flight:

```
K_p (position):  [8, 8, 19]
K_d (velocity):  [5.5, 5.5, 8.7]
K_R (rotation):  [2812, 2812, 163]
K_ω (body rate): [128, 128, 73]
```

Note that K_R and K_ω are substantially larger in xy than z, reflecting the anisotropy of quadrotor attitude authority.

---

## Relevance to Our System

This paper is **directly relevant** to the current controller saturation problem in our racing stack. Our system uses min-snap polynomial trajectories with L-BFGS time allocation, and we are observing controller saturation at 0.85 rad tilt during moderate 48°/38° turns with long approach distances. The root cause is that L-BFGS time allocation optimizes a smoothness objective, not dynamic feasibility: it can assign segment times that require physically infeasible thrust levels at the turn entry/exit.

**Affected modules:**

1. `planning/trajectory_optimizer.py` — The min-snap optimizer. TOPPQuad's insight applies directly: after computing the min-snap geometric path, we should run a TOPP pass to find the fastest feasible speed profile rather than relying on time-scaling heuristics.

2. `planning/racing_line.py` — The speed profiling step. Currently uses curvature-aware heuristics. Replacing or augmenting this with a thrust-constraint-aware speed profile (even a simplified version of TOPPQuad) would directly address the saturation issue.

3. `control/mpc_tracker.py` — The geometric (SE(3)) tracker. Controller saturation at 0.85 rad tilt means the tracker is being commanded attitudes that exceed what the thrust allocation can achieve. A dynamically feasible trajectory from TOPPQuad would not produce such commands.

The specific problem scenario — long approach at speed into a moderate turn — is exactly the case where TOPP provides the most benefit. During straight segments, thrust constraints are not active (ample margin). At the turn entry, the required centripetal force plus attitude change rate may simultaneously saturate multiple motors. TOPPQuad's per-motor constraint enforcement detects this and slows the approach, whereas our current time scaling may not.

---

## Actionable Takeaways

1. **Implement a post-hoc TOPP feasibility check.** After min-snap generates a trajectory, evaluate Eq. 18 at each discretized path point to compute required motor thrusts. Flag any segment where u > u_max. This is a read-only diagnostic that immediately identifies which gates are causing saturation without modifying any existing code.

2. **Replace curvature-based speed scaling with a thrust-constrained speed cap.** For each waypoint in `racing_line.py`, compute the maximum speed at which motor thrusts remain feasible given the local curvature and required attitude change rate. This is a simplified scalar version of TOPPQuad that avoids the full NLP but captures the dominant constraint.

3. **Add body-rate constraints to the speed profile.** The paper explicitly includes ω_max as a constraint. Our system likely has an implicit angular rate limit from the motor model, but it may not be enforced during trajectory generation. Adding ω_max ≤ ω_limit to the speed profile in `trajectory_optimizer.py` would prevent commands that exceed attitude controller bandwidth.

4. **Use N=300 discretization for offline TOPP.** If we implement full TOPPQuad offline (pre-race), 300 grid points over a 20–35 m trajectory segment (Δs ≈ 0.025 m resolution) is shown to be sufficient for accurate constraint enforcement while remaining solvable in ~30 s. Since our race trajectories are precomputed, the 30 s solve time is acceptable.

5. **Use CasADi + IPOPT for the NLP.** The authors use CasADi for automatic differentiation and IPOPT for the non-convex NLP. Both are available in Python and can be added without modifying any physics modules. The integration fits naturally into `planning/trajectory_optimizer.py` as an optional post-processing stage.

6. **Adopt the SE(3) gain ratios as a sanity check.** The controller gains from Table III (K_R_xy/K_R_z ≈ 17:1, K_ω_xy/K_ω_z ≈ 1.75:1) reflect well-tuned aggressive flight. Compare these ratios against our current `TrackerConfig` in `mpc_tracker.py` to check if our rotational gains are appropriately anisotropic.

7. **Warm-start the TOPP NLP with the min-snap speed profile.** The paper requires a good initial guess for IPOPT convergence. Using the L-BFGS min-snap trajectory's speed profile as initialization is exactly what the paper recommends and exploits our existing optimizer output.

---

## Limitations & Caveats

**Computational cost.** The full TOPPQuad NLP takes ~30 seconds per trajectory on a desktop. This is fine for offline pre-race planning but unsuitable for online re-planning mid-race. Our current architecture pre-computes trajectories, so this limitation does not apply unless we want online adaptation (e.g., after gate detection updates pose estimates).

**No aerodynamic drag model.** TOPPQuad ignores air drag and motor dynamics. At high speeds (>5 m/s), drag significantly affects required thrust. The computed speed profiles will be slightly optimistic at high-speed straight segments. For the turn-entry saturation problem we are solving, this matters less because saturation occurs at moderate speed with high centripetal demand, where drag is secondary.

**Non-convex optimization without completeness guarantees.** IPOPT may converge to a local minimum. The paper addresses this with good initialization but does not guarantee global time-optimality. In practice, the solutions are significantly faster than baselines, suggesting good-quality local optima.

**Requires a pre-planned geometric path.** TOPPQuad is a path *parametrizer*, not a path *planner*. It takes a fixed geometric path and finds the fastest timing. If the geometric path itself is suboptimal (bad racing line), TOPPQuad cannot correct it. We still need `racing_line.py` to produce good geometry; TOPPQuad handles the timing layer.

**CrazyFlie-specific hardware validation.** The hardware experiments use a 32 g CrazyFlie 2.0, which is much smaller and lighter than competition racing quads (typically 200–400 g). Thrust bounds, inertia ratios, and gain magnitudes will differ. The method transfers, but the specific numerical parameters (u_max = 0.14375 N, SE(3) gains) require re-tuning for our platform.

**Quaternion integration accuracy.** The paper uses a simple normalized forward Euler quaternion update (Eq. 23) rather than a more accurate exponential map or Lie-group integrator. At N=300 and Δs ≈ 0.025 m this is likely sufficient, but at coarser discretization integration error accumulates. If we use fewer grid points for speed, we should switch to a higher-order quaternion integrator.

**No handling of gate-crossing constraints.** TOPPQuad treats the path as a free-space curve and enforces only dynamic feasibility. It does not handle gate pass-through geometry (the constraint that the drone must be within the gate opening at the crossing point). For racing, we need to couple TOPPQuad with gate-aware path generation (e.g., TOGT or our existing `gate_sequencing/sequencer.py`).

---

## Key Parameters / Constants

| Parameter | Value | Context |
|---|---|---|
| Discretization grid N | 300 | Experiments; Δs ≈ 0.025 m for 7.5 m paths |
| Motor thrust u_max (CrazyFlie) | 0.14375 N per motor | CrazyFlie 2.0 hardware |
| Vehicle mass (CrazyFlie) | 32 g = 0.032 kg | Hardware experiments |
| Position gain K_p | [8, 8, 19] | SE(3) controller, xy vs z asymmetry |
| Velocity gain K_d | [5.5, 5.5, 8.7] | SE(3) controller |
| Rotation gain K_R | [2812, 2812, 163] | SE(3) controller, large xy vs z ratio |
| Body rate gain K_ω | [128, 128, 73] | SE(3) controller |
| Solve time | ~30 s | Desktop CPU, IPOPT via CasADi |
| Test trajectory space | 10×10×10 m³ | Simulation benchmark |
| Forest flight path length | 336 m | Real-world validation |
| Forest time reduction | 43% (236 s → 136 s) | Real-world result |
| Simulation time reduction | 40–50% | vs α-scaled min-snap baselines |
| Integration method | Forward Euler + quaternion normalization | Eq. 23 |

The ratio K_R_xy / K_R_z ≈ 17.2 and K_ω_xy / K_ω_z ≈ 1.75 are notable: they reflect the physical asymmetry of quadrotor attitude control (much higher authority and stiffness in roll/pitch than yaw). These ratios are a useful sanity check for our own controller tuning.
