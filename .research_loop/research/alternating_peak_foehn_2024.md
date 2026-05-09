# An Alternating Peak-Optimization Method for Optimal Trajectory Generation of Quadrotor Drones

- **URL**: https://arxiv.org/abs/2312.02944
- **Authors**: Wytze A.B. de Vries, Ming Li, Qirui Song, Zhiyong Sun (Eindhoven University of Technology)
- **Year**: 2024 (submitted December 2023; published ECC — European Control Conference 2024)
- **Venue**: European Control Conference (ECC) 2024 / arXiv:2312.02944

---

## Key Contribution

The paper introduces an **alternating peak-optimization** method for time-optimal polynomial trajectory generation for quadrotor drones. The core contribution is a computationally efficient decomposition of the joint trajectory optimization problem — which is typically expensive when optimizing polynomial coefficients and segment time allocations simultaneously — into two alternating sub-problems that can each be solved cheaply in closed form or via simple gradient steps.

The key insight is that the **binding constraint** on any trajectory segment is the maximum (peak) value of some motion derivative — velocity, acceleration, jerk, or snap — not the average. Existing methods (Mellinger et al., Bry et al.) use a joint Quadratic Program or L-BFGS-based optimization that uses proxy objectives or gradient approximations over all variables simultaneously. This paper's method instead iterates: (1) solve for polynomial coefficients given fixed segment times, then (2) update segment times based on how close each segment's derivative peak is to its hard limit. This "peak-normalized" update is straightforward and the two steps alternate until each segment is within 5% of saturating its hardest constraint. The result is 20–40% faster trajectories than Mellinger's joint method, with comparable or better tracking accuracy, validated on real Crazyflie 2.1 hardware with VICON at 100 Hz.

---

## Technical Approach

### Polynomial Representation

Each of K trajectory segments uses a degree-5 or degree-7 polynomial (minimum-jerk or minimum-snap respectively). For segment k in one axis:

```
p_k(t) = sum_{i=0}^{N} c_{k,i} * t^i,    t in [0, T_k]
```

The full polynomial vector `p` across all K segments and 3 axes is optimized to minimize the cost functional `J = p^T Q p` (sum of squared jerk or squared snap), subject to continuity constraints at internal waypoints:

```
A p = b
```

where `A` encodes endpoint position, velocity, and acceleration matching across segment boundaries, and `b` contains the boundary conditions (start/end states and waypoint constraints). This is the well-known Mellinger/Richter minimum-snap QP, solvable in closed form via KKT conditions.

### The Two Sub-Problems

**Sub-problem 1 — Coefficient optimization (fixed times):**
Given the current segment durations `{T_k}`, solve the QP for polynomial coefficients. This step is efficient and closed-form; it produces the smoothest trajectory (minimum integrated jerk/snap) achievable with those durations.

**Sub-problem 2 — Time allocation (fixed coefficients):**
Given the current polynomial, evaluate the **normalized peak** of each motion derivative per segment:

```
kappa_k(s) = max_{t in [0, T_k]} || d^s r_k(t) / dt^s || / rho_s
```

where `rho_s` is the hard upper limit on derivative order `s` (velocity, acceleration, jerk, snap). For acceleration, physical feasibility requires including gravity:

```
kappa_k(2) = max( sqrt( x_k''(t)^2 + y_k''(t)^2 + (z_k''(t) + g)^2 ) ) / rho_accel
```

If `kappa_k(s) > 1`, the segment violates that derivative limit. If `kappa_k(s) < 1`, there is unused capacity — the segment could be made faster.

**Segment time update rule (primary):**
```
T_k,new = T_k,old * (1 - delta_1 * (1 - max_s(kappa_k(s))))
```

This shrinks the segment time when the peak is below the limit (speeding up underutilized segments) and grows it when above (relaxing overloaded segments).

**Neighboring segment update (secondary — accounts for continuity coupling):**
```
T_{k-1},new = T_{k-1},old * (1 - delta_2 * (1 - max_s(kappa_k(s))))
T_{k+1},new = T_{k+1},old * (1 - delta_2 * (1 - max_s(kappa_k(s))))
where delta_2 < delta_1
```

Because polynomial continuity constraints couple adjacent segments (a change in T_k affects boundary conditions in k-1 and k+1), the neighboring segments are also adjusted at a smaller learning rate.

**Global scaling fallback (infeasibility recovery):**
If any segment is globally infeasible (kappa > 1.05) after the local update, the method falls back to uniform scaling of all times:
```
T_all,new = T_all,old * (1 - delta_1 * (1 - max_{k,s}(kappa(k,s))))
```

### Convergence Criterion

The algorithm terminates when all `kappa_k` are within **[0.95, 1.05]** — within 5% of their derivative limits. This guarantees (1) feasibility (no limit exceeded) and (2) near-time-optimality (each segment is close to saturating its hardest constraint). The convergence band is a key design choice: stopping at 100% exact saturation is numerically fragile, while 5% tolerance gives robust convergence with negligible suboptimality.

### Full Algorithm (Algorithm 1)

```
Initialize: compute T_k from distance / speed heuristic
Repeat until converged:
  1. Solve QP for polynomial coefficients (Mellinger, fixed times T_k)
  2. For each segment k:
       - Sample polynomial at 20-50 interior points
       - Compute kappa_k(s) for s in {velocity, accel, jerk, snap}
       - Take max_s kappa_k
  3. If max kappa > 1.05: apply global scaling fallback
  4. Else: apply per-segment update with neighbor coupling
Until: all kappa in [0.95, 1.05]
Return: final T_k and polynomial coefficients
```

---

## Results

The paper presents simulation and real-hardware comparisons on a waypoint sequence with tight early turns and a long straight (6 waypoints, Crazyflie 2.1 drone, VICON tracking at 100 Hz):

### Race Time Comparison

| Method                    | Min-Jerk Time | Min-Snap Time |
|---------------------------|---------------|---------------|
| Initial guess (heuristic) | 7.54 s        | 8.13 s        |
| Mellinger optimization    | 6.32 s        | 6.16 s        |
| Peak optimization         | 4.50 s        | 4.92 s        |

Peak optimization achieves **29% faster than Mellinger (min-jerk), 20% faster (min-snap)**, and **40–42% faster than the heuristic initialization**.

### Tracking Quality (RMS Position Error)

| Method               | Min-Jerk  | Min-Snap  |
|----------------------|-----------|-----------|
| Initial guess        | 7.9 cm    | 12.2 cm   |
| Mellinger            | 11.4 cm   | 11.7 cm   |
| Peak optimization    | 10.2 cm   | 9.0 cm    |

Crucially, the peak-optimized trajectory achieves **better tracking than Mellinger despite being faster**. The authors attribute this to more uniform constraint utilization: by saturating each segment close to its limit rather than creating local hotspots of constraint violation, the trajectory avoids the bang-coast-bang artifacts that cause large tracking errors.

### Maximum Speed

Peak optimization achieved 125% higher maximum speed than the initial guess and 88% higher than Mellinger's method, confirming that the method genuinely pushes toward the physical envelope of the drone.

### Physical Constraint Limits Used in Experiments

| Derivative    | Min-Jerk Limit | Min-Snap Limit |
|---------------|----------------|----------------|
| Velocity      | 5.0 m/s        | 5.0 m/s        |
| Acceleration  | 15.5 m/s²      | 14.5 m/s²      |
| Jerk          | 62 m/s³        | 54 m/s³        |
| Snap          | 800 m/s⁴       | 800 m/s⁴       |

---

## Relevance to Our System

Our system is a near-exact analog of the problem this paper solves. We use minimum-snap degree-5 polynomials (same representation), we have K=25 segments through 12 gates (entry + exit waypoints per gate plus start and virtual finish), and our current time allocation uses **L-BFGS-B on log-transformed segment times** — which is precisely the kind of joint black-box optimization the paper's method outperforms.

The current problem — trajectory allocating insufficient time at moderate turns (48°/38°) with long approach distances, causing controller saturation at 0.85 rad tilt — has a direct diagnosis through this paper's lens: our L-BFGS objective uses **proxy constraints** (average velocity per segment, velocity-difference-based acceleration estimates) rather than evaluating the actual polynomial peak derivatives. The polynomial peaks inside each segment can exceed average values by 2–3x on curved segments, particularly at the entry/exit waypoints flanking a sharp turn. This means:

1. Our optimizer underestimates the true constraint violations at turn segments, so it doesn't penalize those segments enough and leaves their times too short.
2. The long straight segments have very low true derivative peaks, so their times could be compressed further — but the L-BFGS objective doesn't detect this because average velocity is already near the limit.

The alternating peak method addresses both problems simultaneously:
- It detects the actual peak (not average) in turn segments and grows their times until the peak is feasible.
- It detects the actual slack in straight segments and shrinks their times until the peak is near the limit.

This directly maps to our `planning/trajectory_optimizer.py`, specifically `_optimize_time_allocation()` (which currently calls `scipy.optimize.minimize` with L-BFGS-B) and the objective function `_time_optimization_objective()` (which uses average-speed and difference-based constraint proxies).

Our `DroneConstraints` currently has:
- `max_velocity = 15.0 m/s`
- `max_acceleration = 20.0 m/s²`
- `max_jerk = 50.0 m/s³`
- `max_tilt_angle = 0.85 rad` (~49°)

The peak-normalization method works identically regardless of limit magnitudes — just substitute our `rho_s` values. Given our track has 39% gap between 23s current time and 14s target (matching the paper's 29–40% speedup over Mellinger), implementing this method is the most direct path to closing that gap.

---

## Actionable Takeaways

1. **Replace the L-BFGS time optimizer with a peak-normalization loop** in `planning/trajectory_optimizer.py`, function `_optimize_time_allocation()`. The new method iterates: solve QP for coefficients → evaluate polynomial peaks → update segment times → repeat until convergence in [0.95, 1.05] band.

2. **Implement a `_compute_peak_kappas()` function** that samples each polynomial segment at 20–50 interior points and computes the maximum norm of each derivative (velocity, acceleration with gravity correction, jerk). This replaces the current heuristic constraint proxies with exact polynomial peak evaluation.

   ```python
   def _compute_peak_kappas(self, waypoints, times, start_velocity):
       kappas = []
       for i, T in enumerate(times):
           # re-solve QP for coefficients at current times
           coeffs_per_axis = [...]
           t_samples = np.linspace(0, T, 40)
           peak_vel = peak_accel = peak_jerk = 0.0
           for t in t_samples:
               v = np.array([_poly_deriv_eval(c, t, 1) for c in coeffs_per_axis])
               a = np.array([_poly_deriv_eval(c, t, 2) for c in coeffs_per_axis])
               j = np.array([_poly_deriv_eval(c, t, 3) for c in coeffs_per_axis])
               a_phys = a.copy(); a_phys[2] += self.constraints.gravity
               peak_vel   = max(peak_vel,   np.linalg.norm(v))
               peak_accel = max(peak_accel, np.linalg.norm(a_phys))
               peak_jerk  = max(peak_jerk,  np.linalg.norm(j))
           kappa = max(
               peak_vel   / self.constraints.max_velocity,
               peak_accel / self.constraints.max_acceleration,
               peak_jerk  / self.constraints.max_jerk,
           )
           kappas.append(kappa)
       return kappas
   ```

3. **Set convergence criterion to kappa in [0.95, 1.05]** — not a gradient tolerance. This ensures the optimizer stops when physically meaningful convergence is achieved, not when a mathematical tolerance is met.

4. **Use neighbor coupling** (delta_2 ≈ delta_1 / 3) to handle polynomial continuity coupling between adjacent segments. This prevents oscillation where shrinking one segment causes the adjacent segment to become infeasible.

5. **Apply global scaling fallback** whenever any kappa > 1.05 after the local update. This prevents the alternating update from diverging in the early iterations when far from feasibility.

6. **Tune learning rates**: Start with delta_1 = 0.3 and delta_2 = 0.1 as the paper's implied range. For our 25-segment track, convergence should occur in 50–150 iterations; each iteration is cheap (25 QP solves + 25×40 polynomial evaluations = sub-second in Python).

7. **Include gravity in acceleration peak computation**: `a_phys = [ax, ay, az + g]` as per Equation 10. This is critical because a drone hovering requires thrust to counteract gravity, so the effective acceleration limit from the motors is `rho_accel` measured in the frame including gravity.

8. **Use the paper's constraint limits as validation cross-check**: For Crazyflie (5 m/s, 15.5 m/s², 62 m/s³), the method achieves 4.50s for a 6-waypoint track. Scaling by our higher limits (15 m/s velocity → 3× higher) suggests our 25-waypoint track should be achievable in roughly 14–16s — aligning with our target.

---

## Limitations & Caveats

1. **Not globally optimal.** The peak-normalization update is a gradient-descent heuristic on segment times. It may converge to a local minimum for tracks with complex geometry (U-turns, near-parallel gates). Our current Strait of Hormuz track likely has well-separated gates, making this less of a concern, but it is not guaranteed to find the global time minimum.

2. **No gate region optimization.** The paper uses simple point waypoints, not gate pass-through *regions*. Our gates have 1.2m × 1.2m openings, so the optimal path may pass off-center to reduce curvature. This paper does not help optimize the entry/exit waypoint lateral offsets — that is handled separately by our `planning/racing_line.py`. The two optimizations are complementary: racing_line provides waypoint placement, alternating peak provides time allocation.

3. **No snap continuity at segment boundaries.** The per-segment polynomial formulation guarantees C2 continuity (position, velocity, acceleration) but not C3 or higher across boundaries. For very aggressive maneuvers this causes discontinuities in jerk (and hence motor thrust commands). Our existing implementation has the same limitation — this paper does not introduce a new issue here.

4. **Learning rate tuning still manual.** The paper demonstrates effectiveness but does not provide a systematic procedure for choosing delta_1 and delta_2 for different platform speeds or track geometries. For our racing drone (3× faster than Crazyflie), the optimal learning rates may differ.

5. **Computation time for online replanning.** For 25 segments, each iteration requires 25 QP solves. In Python this could be 50–200ms per outer iteration; 100 iterations = 5–20s. This is acceptable for pre-race offline planning (our use case) but rules out online replanning from this method alone. The paper does not address online replanning.

6. **Platform scaling uncertainty.** The Crazyflie is a 27g micro-drone with a velocity limit of 5 m/s. Our racing drone operates at 15 m/s — 3× higher velocity — with correspondingly higher jerk and snap. The method's convergence properties and speedup percentages may differ at higher speeds, though the algorithm itself is platform-agnostic.

7. **Assumes fixed waypoints.** The method does not co-optimize waypoint positions and time allocation. For our system, combining this with `racing_line.py` lateral offset optimization would require an outer loop, adding complexity.

---

## Key Parameters / Constants

| Parameter              | Paper Value           | Our System Default         | Notes                                      |
|------------------------|-----------------------|----------------------------|--------------------------------------------|
| Primary learning rate δ₁ | ~0.2–0.4 (inferred) | N/A (uses L-BFGS)         | Start at 0.3; tune in [0.1, 0.5]           |
| Neighbor learning rate δ₂ | < δ₁ (inferred)  | N/A                        | Start at δ₁/3 ≈ 0.1                       |
| Convergence band       | [0.95, 1.05]          | ftol=1e-6 (L-BFGS)        | 5% tolerance; key design choice            |
| Max iterations         | ~50–100 (inferred)    | 200 (L-BFGS maxiter)      | Should converge in ~100 for 25 segments    |
| Polynomial order       | 5 (min-jerk) or 7    | 5 (min-snap per comment)  | Both work; paper validates both            |
| Interior samples/segment | 20–50             | N/A (not currently sampled)| 40 is a good default                      |
| Velocity limit         | 5.0 m/s (Crazyflie)  | 15.0 m/s                  | Substitute our value for rho_v             |
| Acceleration limit     | 15.5 m/s² (min-jerk) | 20.0 m/s²                 | Include gravity: check ||[ax, ay, az+g]||  |
| Jerk limit             | 62 m/s³               | 50.0 m/s³                 | Substitute our value for rho_j             |
| Snap limit             | 800 m/s⁴              | N/A (not constrained)     | Add constraint if snap violations appear   |
| Min segment time       | not stated            | 0.1 s                     | Clamp to prevent numerical issues          |
| Gravity correction     | +9.81 m/s² in z      | gravity = 9.81            | Must include in acceleration peak check    |
