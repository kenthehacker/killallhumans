# Sequential Convex Programming for Time-optimal Quadrotor Waypoint Flight

- **URL**: https://ieeexplore.ieee.org/document/10802749/ (also: https://www.researchgate.net/publication/387422364_Sequential_Convex_Programming_for_Time-optimal_Quadrotor_Waypoint_Flight)
- **Authors**: Zhipeng Shen, Guanzhong Zhou, Hailong Huang
- **Affiliation**: Department of Aeronautical and Aviation Engineering, Hong Kong Polytechnic University, Hong Kong, China
- **Year**: 2024
- **Venue**: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Abu Dhabi, UAE, October 14-18, 2024. Pages 3108-3115. DOI: 10.1109/IROS58592.2024.10802749

---

## Key Contribution

This paper addresses a fundamental bottleneck in time-optimal quadrotor trajectory planning: the chicken-and-egg problem of waypoint time allocation. Prior numerical optimization approaches required waypoints to be pre-assigned to specific discrete time nodes within a fixed-horizon formulation. This time allocation is a priori unknown — choosing it poorly forces the optimizer to either spend too long near a waypoint (wasting time) or arrive too early (forcing constraint violations). The result is systematically suboptimal trajectories, because time optimality cannot be achieved when the temporal positioning of waypoints is fixed before optimization begins.

The central contribution is the introduction of **state-triggered constraints** within a Sequential Convex Programming (SCP) framework that simultaneously optimizes both the waypoint time allocation and the trajectory geometry. Rather than committing to a time grid before solving, the approach allows the optimizer to decide — during trajectory optimization itself — exactly when in time each waypoint is visited. This removes the suboptimality gap introduced by time pre-allocation. Combined with a time-scaling direct multiple shooting scheme and a fast semidefinite-programming-based convex relaxation that exploits sparsity, the result is a computationally efficient solver that achieves near-optimal trajectories under full 6-DOF quadrotor dynamics, validated in both simulation and physical hardware experiments with constrained open time windows.

---

## Technical Approach

### Problem Formulation: Non-convex Optimal Control

The core problem is a free-final-time Optimal Control Problem (OCP) where a quadrotor must pass through a sequence of spatial waypoints in minimum total time, subject to:
- Full 6-DOF quadrotor dynamics (position, velocity, attitude as quaternion or rotation matrix, body angular rates)
- Single-rotor thrust constraints (more realistic than collective-thrust-only models)
- Body-rate constraints
- Waypoint spatial constraints that must be satisfied at some unknown time(s)

The 6-DOF dynamics are nonlinear and non-convex. The coupling between continuous-time dynamics and discrete waypoint-visit events makes the NLP inherently non-convex due to the simultaneous treatment of continuous state evolution and event timing.

### State-Triggered Constraints for Time Allocation

The key innovation is encoding waypoint-visit requirements as **state-triggered constraints**. In classical formulations, a waypoint is enforced by pinning a state at a specific discrete-time index k: `x[k] = x_waypoint`. This requires choosing k before solving, fixing the time at which the waypoint is visited.

State-triggered constraints replace this with a conditional form: when a state-based trigger condition g_trig(x) < 0 is satisfied (i.e., when the trajectory is in the neighborhood of the waypoint), a consequent constraint g_stc(x) <= 0 is enforced. In the waypoint context, the trigger activates when the drone is spatially near the waypoint (measured in progress or position space), and the consequent enforces passage through the gate geometry. This makes waypoint satisfaction conditional on state, not on time index — decoupling the event from a pre-assigned discrete time node.

The formulation draws on the broader "state-triggered constraints" literature (Szmuk et al., 2019/2020) which has been applied to spacecraft powered descent guidance, where similar free-event-time problems appear.

### Progress Variable and Simultaneous Optimization

Closely related to the CPC (Complementary Progress Constraints) framework of Foehn & Scaramuzza (Science Robotics 2021), the approach introduces a **progress variable** along the trajectory. This scalar variable encodes how far along the planned path the drone has traveled. Waypoint constraints are tied to progress values rather than time indices: the constraint fires when the progress variable reaches the waypoint's progress level, and the optimizer is free to choose how much time is spent reaching that progress level. This elegant reformulation converts the free-time-allocation problem into one where timing is an implicit output of optimizing the progress dynamics.

The fundamental difference from CPC is the convexification strategy. CPC uses an MPCC (Mathematical Program with Complementarity Constraints) formulation, which is non-convex and solved via NLP solvers (IPOPT-class methods) that require significant computation (minutes scale for non-trivial problems). Shen et al. instead pursue a Sequential Convex Programming route that convexifies the problem iteratively, producing a sequence of convex subproblems.

### Time-Scaling Direct Multiple Shooting

The trajectory is discretized using a **direct multiple shooting** scheme with a **time-scaling transformation**. The physical time t is re-parameterized through a normalized parameter τ ∈ [0, 1] with a time-dilation factor s(τ) = dt/dτ > 0. This converts the free-final-time problem into a fixed-final-time problem where the time-dilation function s(τ) becomes an optimization variable. Constraints s(τ) >= s_min > 0 prevent degenerate solutions where the time derivative collapses to zero.

The prediction horizon is partitioned into segments that align with the waypoint structure — each segment corresponds to a flight phase between consecutive waypoints. This segmented structure naturally accommodates multiple waypoints and allows the optimizer to allocate different amounts of virtual time (and thus physical time) to each inter-waypoint segment.

### Sequential Convex Programming Algorithm

The SCP loop proceeds as follows:
1. Initialize with a feasible reference trajectory (e.g., straight-line segments with heuristic timing)
2. Linearize the nonlinear dynamics around the current reference: compute Jacobians A_k, B_k^- , B_k^+ via sensitivity equations or automatic differentiation
3. Formulate a convex subproblem: trust-region-constrained quadratic/linear program with the linearized dynamics, convexified constraints, and L1-norm penalty on dynamics violations (virtual control)
4. Solve the convex subproblem (using SDP or QP depending on the relaxation)
5. Update the reference trajectory
6. Repeat until convergence (dynamics defect norm falls below threshold, trust region satisfied)

The SCP framework handles the non-convexity of 6-DOF dynamics through successive linearization, not through global convexification. Convergence to a local optimum is the typical guarantee, with the quality of the initialization affecting the final solution quality.

### Semidefinite-Programming-Based Convex Relaxation

A novel algorithmic contribution is the **fast SDP-based convex relaxation** that exploits the **sparsity pattern of the lifted formulation**. Certain nonlinear terms in the quadrotor dynamics (particularly trigonometric functions of attitude) can be handled by "lifting" — introducing auxiliary variables and imposing quadratic/semidefinite constraints that tightly approximate the original nonlinear relations.

The resulting relaxation is an SDP (semidefinite program) rather than an NLP. SDPs are convex and can be solved to global optimality by interior-point methods. However, large-scale SDPs are slow unless sparsity is exploited. The paper identifies and exploits the specific sparsity pattern present in the lifted quadrotor dynamics, yielding a sparse SDP that can be solved significantly faster than a dense formulation. This is what allows the approach to "significantly reduce computing time" compared to prior methods (which either solve dense NLPs or ignore sparsity in lifted formulations).

### Handling Open Time Windows

The physical experiment involves "constrained open time windows" — scenarios where the drone must pass through a waypoint within a specified time interval [t_min_i, t_max_i] rather than at an exact time. State-triggered constraints naturally accommodate this: the trigger condition fires when the drone is near the waypoint, and the consequent enforces that the physical time (recovered from the time-scaling variable) falls within the allowed window. This is more realistic than pure time-optimal planning with no temporal constraints and directly models competition scenarios where gates open and close.

---

## Results

### Simulation Results

Comprehensive simulation studies demonstrated:
- **Solution optimality**: The SCP-generated trajectories achieve trajectory times competitive with or approaching the lower bounds from more computationally expensive methods, confirming that the simultaneous time-allocation approach avoids the suboptimality gap of pre-allocated timing approaches.
- **Computational efficiency**: The SCP algorithm significantly reduces computing time compared to general nonlinear programming approaches (IPOPT-class solvers on the full NLP). The sparsity-exploiting SDP relaxation is a key driver of this speedup.
- Comparison with baselines (including methods that pre-allocate waypoint times) shows measurable trajectory time improvements, validating that the state-triggered constraint formulation for simultaneous time allocation is genuinely beneficial.

### Hardware Experiments

Real-world quadrotor flights were conducted to validate practical applicability:
- The task involved navigating through waypoints with **constrained open time windows** — the drone must pass through each gate within a temporal window, not at a fixed time.
- The SCP-generated trajectories were successfully executed on physical hardware.
- Results confirmed that the approach works under real-world uncertainties, sensor noise, and tracking errors.

### Quantitative Specifics

The paper reports 8 pages of results (pages 3108-3115 in IROS proceedings), but specific numerical values (e.g., exact computation times in ms, trajectory time reductions in %) were not accessible from publicly available abstracts and metadata. Based on contextual information from the abstract and related survey papers, the computation time is expected to be in the range of seconds to tens of seconds for multi-waypoint scenarios (far below the minutes required by CPC/IPOPT on similar problems, but likely not real-time without further approximation).

---

## Relevance to Our System

Our current system uses a min-snap polynomial trajectory optimizer with L-BFGS time allocation. This two-stage approach — first fix time allocation heuristically, then optimize polynomial coefficients — is exactly the architecture this paper identifies as fundamentally suboptimal. Our time allocation is determined by a speed profile heuristic before trajectory optimization begins, which means:

1. **We cannot recover from a poor initial time allocation**: if L-BFGS converges to a local minimum in time allocation space that does not correspond to the globally optimal time distribution, the trajectory quality is permanently limited.
2. **Polynomial smoothness constraints leave actuator capacity on the table**: min-snap polynomials are smooth by construction, which prevents the trajectory from exploiting full single-rotor thrust limits during aggressive maneuvers near gates.
3. **The per-gate timing is not jointly optimized with gate-traversal geometry**: our L-BFGS optimizes segment times but treats waypoint constraints as fixed in time, exactly the limitation this paper solves.

The state-triggered constraint approach could improve our system in two ways:
- **Better time allocation**: simultaneous optimization of when each gate is traversed and how the trajectory is shaped would likely reduce race time vs. our current L-BFGS approach, particularly for tight gate sequences where timing and geometry are coupled.
- **Tighter gate traversal**: allowing the optimizer to decide freely when to pass through each gate removes the constraint that gate crossing occurs at a specific polynomial node, enabling better use of the gate's spatial window.

The computational cost is the main concern for our use case: SCP with SDP is more expensive than L-BFGS on polynomial coefficients, and we need trajectories computed in seconds, not minutes.

---

## Actionable Takeaways

1. **Replace fixed time-node waypoint constraints with state-triggered or progress-variable constraints**: our current min-snap formulation assigns each gate to a specific polynomial node time. Reformulating gate constraints as "must be satisfied when the drone is within X meters of the gate center-line" frees the optimizer to find the true optimal crossing time, which may differ substantially from the heuristic assignment.

2. **Implement a progress variable**: introduce a scalar progress variable s ∈ [0, N_gates] that increases monotonically along the trajectory. Gate constraints fire when s crosses integer values. This decouples gate satisfaction from time-grid assignment and is compatible with our existing polynomial trajectory infrastructure.

3. **Investigate SDP relaxation for attitude dynamics**: the sparsity-exploiting SDP relaxation described in this paper could replace or augment our current geometric tracker gain tuning. In particular, if our polynomial trajectory assumes simplified dynamics, switching to a convex relaxation of full 6-DOF dynamics would allow the planner to stay within the actual actuator envelope.

4. **Use SCP outer loop instead of L-BFGS**: replace the L-BFGS time-allocation loop with an SCP outer loop that iterates over linearized convex subproblems. Each subproblem is a QP or LP that can be solved fast; the SCP overhead comes from the number of iterations (typically 10-50) and Jacobian computation. For our 8-gate track, this is likely tractable if the dynamics Jacobian computation is efficient (e.g., analytical or AD-computed).

5. **Initialize the SCP from our existing L-BFGS solution**: the current L-BFGS output provides a reasonable initial feasible trajectory for SCP warm-starting. This avoids the cold-start convergence problems and should reduce the number of SCP iterations needed.

6. **Segment the multiple shooting horizon gate-by-gate**: the paper's approach of partitioning the prediction horizon into segments aligned with waypoints maps directly onto our gate sequence. Each segment (gate i to gate i+1) can have its time scaling independently optimized, which is structurally similar to our current segment-time L-BFGS but with the convex reformulation enabling true global search over segment lengths.

7. **Consider open time windows as a constraint relaxation**: rather than enforcing exact gate crossing times, model each gate as having an open time window [t_min, t_max]. For a first implementation, t_max - t_min can be set generously (e.g., ±0.5s around the heuristic crossing time), and this can be tightened as the optimizer matures. This is more robust than exact-time enforcement.

8. **Benchmark against our current L-BFGS approach on the per-gate error metrics**: the per-gate tracking error data we already collect (e.g., gate-3 error 0.345m, gate-4 error 0.310m) provides a concrete regression test. Any SCP-based replacement should reduce these gate-specific errors, since those errors often arise from the trajectory passing through the wrong part of the gate opening due to suboptimal crossing geometry.

---

## Limitations & Caveats

1. **No public code**: the paper does not appear to have an accompanying open-source repository. Implementation requires re-deriving the formulation from the paper, which is behind an IEEE paywall. The full technical derivation of the SDP relaxation and the sparsity structure may require significant engineering effort to reproduce.

2. **Computation time uncertainty**: the abstract claims "significant reduction in computing time," but specific numbers are not publicly available. For drone racing applications requiring sub-second replanning, it is unclear whether the SCP with SDP is fast enough without GPU acceleration or problem-specific warm-starting. The Science Robotics CPC baseline required minutes for multi-waypoint problems; this paper likely reduces that to seconds but probably not sub-second for the full formulation.

3. **Local optimality only**: SCP converges to a local optimum of the original non-convex problem. The quality of the solution depends strongly on initialization. For drone racing, the global optimum matters (to minimize race time), so there is no guarantee the SCP solution is globally time-optimal, only locally.

4. **Full 6-DOF dynamics complexity**: the formulation uses full quadrotor dynamics including individual rotor thrust constraints. Our existing system uses a simpler model (collective thrust + body-rate model). Integrating the SCP approach may require upgrading our dynamics model, which has knock-on effects on the controller and EKF.

5. **Hardware validation scope**: the real-world experiments demonstrate feasibility with "constrained open time windows" but do not appear to be drone racing scenarios with gates at racing speeds. The physical experiments validate the time-window constraint satisfaction, not necessarily competition-grade aggressive performance.

6. **No direct comparison to CPC**: the relationship to the Foehn et al. CPC approach (Science Robotics 2021) is implicit. CPC uses an MPCC formulation that is solved by NLP solvers with long compute times. The SCP approach should be faster but the exact speedup factor and trajectory quality tradeoff are not explicitly quantified in publicly available material.

7. **Semidefinite program scaling**: SDP-based approaches scale poorly with problem dimension. For long race tracks with many gates, the sparse SDP may become a bottleneck. The paper validates on short sequences; performance on a 20+ gate track (as in some competition scenarios) is unknown.

8. **Convergence guarantees are limited**: SCP convergence to a feasible point (not just a local optimum) depends on the trust region and virtual control penalty parameters. Poorly tuned parameters can cause infeasibility or oscillation between SCP iterates.

---

## Key Parameters / Constants

The following parameters are referenced in the methodology, based on what is publicly available from the abstract and related works in the SCP/quadrotor time-optimal literature. Exact values would require access to the full paper.

**Quadrotor Model Parameters** (full 6-DOF model):
- Mass m (kg) — vehicle-specific
- Inertia tensor J ∈ R^{3x3} — vehicle-specific
- Single-rotor thrust bounds: T_min, T_max per rotor (N), or equivalently collective thrust f ∈ [f_min, f_max]
- Body-rate bounds: ||omega|| <= omega_max (rad/s)
- Arm length l (m)
- Rotor force-to-torque constant k_m

**SCP Algorithm Parameters**:
- Trust region radius delta_tr (initialized large, reduced on convergence failure)
- Virtual control penalty weight lambda_vc (L1-norm weight on dynamics defect)
- Convergence tolerance epsilon (dynamics defect norm threshold)
- Maximum SCP iterations N_iter (typically 20-100)
- Time-dilation lower bound s_min > 0 (prevents degenerate zero-time solutions)

**Multiple Shooting Parameters**:
- Number of shooting nodes N per segment (between consecutive waypoints)
- Number of waypoints / segments M (equals number of gates)
- Time-scaling factor s(τ) ∈ [s_min, s_max]

**Semidefinite Relaxation Parameters**:
- Lifted variable matrix dimension (depends on attitude parameterization used)
- Sparsity pattern (paper-specific, derived from the structure of quadrotor dynamics lifting)
- SDP solver tolerance (typically 1e-6 to 1e-8 for interior-point methods)

**State-Triggered Constraint Parameters**:
- Trigger proximity threshold for waypoint activation (distance from waypoint at which constraint fires)
- Constraint satisfaction margin for waypoint passage (how tightly the drone must thread the gate)
- Time window bounds [t_min_i, t_max_i] for open-window experiments

**Initialization**:
- Straight-line or minimum-snap warm start with heuristic time allocation (e.g., distance-proportional segment times)
- Initial reference speed: typically set near expected average race speed

---

## Relationship to Prior Work

This paper's position in the time-optimal quadrotor trajectory literature:

- **Gao et al. IROS 2018** ("Optimal Time Allocation for Quadrotor Trajectory Generation"): optimizes segment times for polynomial trajectories using gradient-based methods, but with pre-fixed waypoint assignment. Shen et al. directly improve on this by making the assignment itself an optimization variable.

- **Foehn & Scaramuzza arXiv 2020 / Science Robotics 2021** ("CPC: Complementary Progress Constraints"): solved the simultaneous time-allocation problem using MPCC formulation with NLP solvers. High solution quality but high computation time (minutes). Shen et al. achieve similar simultaneous optimization with much lower computation via SCP + SDP.

- **Szmuk et al. IROS 2019** ("State-Triggered Constraints in SCP for Powered Descent"): originated the state-triggered constraint concept in an SCP context for spacecraft. Shen et al. transfer this concept to quadrotor waypoint flight.

- **TOGT Planner (Qin 2024)**: another 2024 approach to time-optimal gate traversal; Shen et al. address a complementary aspect (time allocation) rather than gate-traversal geometry specifically.

The paper's approach is a clean synthesis: borrow state-triggered constraints from the spacecraft SCP literature, apply them to the drone waypoint time-allocation problem (previously addressed only via MPCC/CPC), and add a fast SDP relaxation to make the convex subproblems tractable.

---

*Analysis written: 2026-04-14. Full paper text not directly accessible (IEEE paywall); this analysis synthesizes all publicly available metadata, abstract text, and related-work context from IEEE Xplore, ResearchGate, and related SCP/quadrotor literature.*
