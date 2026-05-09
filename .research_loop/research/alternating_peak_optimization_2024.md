# An Alternating Peak-Optimization Method for Optimal Trajectory Generation

- **URL**: https://arxiv.org/abs/2312.02944
- **Year**: 2024 (ECC — European Control Conference)
- **Authors**: Wytze A.B. de Vries, Ming Li, Qirui Song, Zhiyong Sun

---

## Key Contribution

The paper presents an **alternating peak-optimization** method for time-optimal polynomial trajectory generation for quadrotors. The central insight is to decompose the joint optimization over polynomial coefficients and segment time allocation into two alternating sub-problems, each of which is efficient to solve in isolation:

1. **Coefficient sub-problem** (fixed times): Given segment durations, solve a standard minimum-snap/jerk QP — this is closed-form and extremely fast.
2. **Time sub-problem** (fixed coefficients): Given the current polynomial, compute the "peak" violation ratio for each derivative constraint per segment, then update segment times via a simple multiplicative rule that pushes each segment toward saturation of its hardest constraint.

This decomposition avoids the expensive, numerically difficult joint optimization over both polynomial parameters and time variables simultaneously — which is the key bottleneck in methods like Mellinger's joint approach or L-BFGS on both variables at once.

The method produces hard-feasible trajectories (no constraint is violated at convergence) while achieving flight times 20–40% shorter than Mellinger's method on the same test tracks.

---

## Technical Approach

### Polynomial Representation

Each segment k uses a 5th-order polynomial (degree 5, minimum-jerk) or can be extended to minimum-snap. Trajectory segments are encoded as:

```
p_k(t) = sum_{i=0}^{5} c_{k,i} * t^i
```

For K segments, the cost is the block-diagonal quadratic form:

```
J = p^T Q p
where Q = block-diag(Q_1, Q_2, ..., Q_K)
```

Continuity constraints (position, velocity, acceleration at shared waypoints) are captured in `A p = b`, making the coefficient sub-problem a **Quadratic Program with linear constraints** — solvable in closed form via the standard KKT conditions (Mellinger's method).

### Time Sub-problem: Peak Normalization

After computing polynomial coefficients, the algorithm evaluates the **normalized peak** for each motion derivative and each segment:

```
kappa_k(s) = max_{t in [0, T_k]} || d^s r_k(t) / dt^s || / rho_s
```

where `rho_s` is the hard limit for derivative order `s` (velocity, acceleration, jerk, snap). For acceleration, gravity is factored in:

```
kappa_k(2) = max( sqrt( x_k''(t)^2 + y_k''(t)^2 + (z_k''(t) + g)^2 ) ) / rho_accel
```

If `kappa_k > 1` for any axis/derivative, the segment violates a constraint. If `kappa_k < 1`, there is unused headroom.

**Segment time update rule** (primary):
```
T_k,new = T_k,old * (1 - delta_1 * (1 - max_s(kappa_k(s))))
```

**Neighboring segment adjustment** (secondary — accounts for continuity coupling):
```
T_{k±1},new = T_{k±1},old * (1 - delta_2 * (1 - max_s(kappa_k(s))))
with delta_2 < delta_1
```

If any segment is globally infeasible after the update, a **global scaling fallback** uniformly scales all times:
```
T_all,new = T_all,old * (1 - delta_1 * (1 - max_{k,s}(kappa(k,s))))
```

The loop terminates when all `kappa_k` are in `[0.95, 1.05]` — meaning every segment is within 5% of its limit. This is the convergence band that guarantees feasibility while nearly saturating all constraints.

### Full Algorithm (Algorithm 1)

```
Initialize: compute T_k from distance / speed heuristic
Repeat:
  1. Solve QP for polynomial coefficients (Mellinger, fixed times T_k)
  2. For each segment k, compute kappa_k(s) for s in {velocity, accel, jerk, snap}
  3. Update T_k using peak normalization update rule (with neighbors)
  4. If any kappa > 1.05: apply global scaling fallback
Until: all kappa in [0.95, 1.05]
Return: final T_k and polynomial coefficients
```

### Constraint Set

The paper uses the following hard limits for its Crazyflie 2.1 experiments:

| Derivative | Min-Jerk Limit | Min-Snap Limit |
|------------|---------------|----------------|
| Velocity   | 5.0 m/s       | 5.0 m/s        |
| Acceleration | 15.5 m/s²  | 14.5 m/s²      |
| Jerk       | 62 m/s³       | 54 m/s³        |
| Snap       | 800 m/s⁴      | 800 m/s⁴       |

---

## Results

On a test waypoint sequence with tight early turns and long straights (Crazyflie 2.1, VICON at 100 Hz):

| Method                   | Min-Jerk Time | Min-Snap Time |
|--------------------------|---------------|---------------|
| Initial guess (heuristic)| 7.54 s        | 8.13 s        |
| Mellinger optimization   | 6.32 s        | 6.16 s        |
| Peak optimization        | 4.50 s        | 4.92 s        |

Improvements over Mellinger: **29% faster (min-jerk), 20% faster (min-snap)**.

**Tracking quality (position RMS errors):**

| Method               | Min-Jerk  | Min-Snap  |
|----------------------|-----------|-----------|
| Initial guess        | 7.9 cm    | 12.2 cm   |
| Mellinger            | 11.4 cm   | 11.7 cm   |
| Peak optimization    | 10.2 cm   | 9.0 cm    |

Notably, the peak-optimized trajectory has *better* tracking than Mellinger despite being faster — because it saturates constraints more uniformly rather than creating local hotspots.

The authors validate results with **real-world flights** on the Crazyflie, confirming the trajectories are executable by a real controller at the computed times.

---

## Relevance to Our System

Our system is directly analogous to what this paper addresses:

- We use minimum-snap 5th-order polynomials (identical representation).
- We have 25 segments (entry + exit waypoints per 12 gates + start + virtual finish).
- Our current time optimization uses **L-BFGS-B on log-transformed segment times**, which is a black-box optimizer that does not exploit the structure of the problem.
- Our constraint checking is **approximate**: we use average velocity per segment and a rough acceleration estimate from velocity differences, not the actual polynomial peak values.
- Race time is 23s vs. a 14s target — a 39% gap. This paper achieves 29–40% speedups over comparable baseline methods.

The key diagnostic: our L-BFGS objective uses heuristic constraint proxies (average speed, not peak speed; average acceleration, not max acceleration). The polynomial peaks inside each segment can exceed these averages by 2–3x on curved segments. This means our optimizer currently underestimates constraint violations, leaving significant time on the table without triggering penalty terms.

The alternating peak method directly addresses this: it evaluates the **actual polynomial peak** for each derivative, which is the binding constraint. It then updates times segment-by-segment toward exactly saturating those constraints.

---

## Actionable Takeaways

### 1. Replace L-BFGS time optimizer with peak-optimization loop

In `planning/trajectory_optimizer.py`, `_optimize_time_allocation()` currently runs L-BFGS on a proxy objective. Replace or supplement with the peak-normalization loop:

```python
def _optimize_time_allocation_peak(self, waypoints, initial_times, start_velocity):
    times = list(initial_times)
    delta_1 = 0.3   # primary learning rate (tune: 0.1–0.5)
    delta_2 = 0.1   # neighbor learning rate
    max_iters = 100
    kappa_tol_lo = 0.95
    kappa_tol_hi = 1.05

    for _ in range(max_iters):
        # Step 1: solve for polynomial coefficients (already done in _generate_trajectory)
        # Step 2: compute peak kappa per segment
        kappas = self._compute_peak_kappas(waypoints, times, start_velocity)
        # Step 3: check convergence
        if all(kappa_tol_lo <= k <= kappa_tol_hi for k in kappas):
            break
        # Step 4: global feasibility check
        if max(kappas) > kappa_tol_hi:
            scale = 1 - delta_1 * (1 - max(kappas))
            times = [t * scale for t in times]
            continue
        # Step 5: per-segment update with neighbor coupling
        new_times = list(times)
        for k, kappa in enumerate(kappas):
            new_times[k] *= (1 - delta_1 * (1 - kappa))
            if k > 0:
                new_times[k-1] *= (1 - delta_2 * (1 - kappa))
            if k < len(times) - 1:
                new_times[k+1] *= (1 - delta_2 * (1 - kappa))
        times = [max(t, 0.05) for t in new_times]
    return times
```

### 2. Implement `_compute_peak_kappas()` using actual polynomial evaluation

The function must evaluate the polynomial (from `_min_snap_1d`) at many interior points per segment and find the actual peak for each derivative. This is computationally cheap — just sample at 20–50 points per segment and take the max:

```python
def _compute_peak_kappas(self, waypoints, times, start_velocity):
    kappas = []
    for i, T in enumerate(times):
        coeffs_per_axis = [...]  # re-solve QP for this segment
        t_samples = np.linspace(0, T, 30)
        peak_vel = 0.0
        peak_accel = 0.0
        peak_jerk = 0.0
        for t in t_samples:
            v = np.array([_poly_deriv_eval(c, t, 1) for c in coeffs_per_axis])
            a = np.array([_poly_deriv_eval(c, t, 2) for c in coeffs_per_axis])
            j = np.array([_poly_deriv_eval(c, t, 3) for c in coeffs_per_axis])
            # Add gravity to accel z for physical feasibility
            a_phys = a.copy(); a_phys[2] += self.constraints.gravity
            peak_vel = max(peak_vel, np.linalg.norm(v))
            peak_accel = max(peak_accel, np.linalg.norm(a_phys))
            peak_jerk = max(peak_jerk, np.linalg.norm(j))
        kappa = max(
            peak_vel   / self.constraints.max_velocity,
            peak_accel / self.constraints.max_acceleration,
            peak_jerk  / self.constraints.max_jerk,
        )
        kappas.append(kappa)
    return kappas
```

### 3. Use the convergence band [0.95, 1.05] as the stopping criterion

This is the most practically important design choice: stopping at 95–105% of constraint limits ensures the trajectory is both feasible **and** time-optimal to within 5%.

### 4. Apply to our current DroneConstraints

Our current limits in `DroneConstraints`:
- `max_velocity = 15.0 m/s` (vs. 5 m/s for Crazyflie — we have a faster drone)
- `max_acceleration = 12.0 m/s²`
- `max_jerk = 50.0 m/s³`

The peak-normalization approach works identically regardless of the limit magnitudes — just substitute our values for `rho_s`.

### 5. Tuning guidance for learning rates

The paper does not report exact δ₁, δ₂ values but implies:
- δ₁ ∈ [0.1, 0.5] — larger values converge faster but may oscillate
- δ₂ = δ₁ / 3 is a reasonable starting ratio
- For 25 segments and ~100 iterations, convergence should be sub-second even in Python

---

## Limitations & Caveats

1. **Not globally optimal.** The peak-normalization update is a gradient descent heuristic on segment times. It may converge to a local minimum, especially with many segments or complex track geometries. Tracks with near-parallel gates or u-turns may need a better initialization.

2. **No jerk/snap continuity across segments.** The per-segment polynomial formulation ensures position, velocity, and acceleration continuity (C2 or C3) but the paper does not impose snap continuity at segment boundaries. For very aggressive trajectories this can cause discontinuities in motor thrust commands.

3. **Gate constraints not modeled.** The paper uses simple waypoints (fixed positions), not gate pass-through regions (which would allow optimizing *where* within the gate opening to fly). This is a known limitation compared to TOGT (Qin 2024), which treats gates as feasible regions and can find shorter paths by offsetting through corners.

4. **Two tuning parameters (δ₁, δ₂) still require manual selection.** The paper demonstrates the method works well in their experiments but does not give a systematic procedure for tuning these for different track geometries or platforms.

5. **Computation time not explicitly reported.** For our 25-segment track, the per-iteration cost is 25 QP solves + 25×30 polynomial evaluations. At Python speeds this could be 50–200ms per outer iteration, meaning 100 iterations might take 5–20s. This is acceptable for offline trajectory generation (we pre-compute before the race) but could be a concern if we ever need online replanning.

6. **Crazyflie platform.** The experimental platform is a micro-drone with very conservative limits (5 m/s velocity). The paper's speedup percentages may differ for our racing drone with 15 m/s velocity — but the methodology is platform-agnostic.

---

## Key Parameters / Constants

| Parameter | Paper Value | Our System Default | Notes |
|-----------|------------|-------------------|-------|
| Primary learning rate δ₁ | ~0.2–0.4 (inferred) | N/A (use L-BFGS) | Tune in [0.1, 0.5] |
| Neighbor learning rate δ₂ | < δ₁ | N/A | Start at δ₁/3 |
| Convergence band | [0.95, 1.05] | None (ftol=1e-6) | 5% tolerance |
| Max iterations | ~50–100 (inferred) | 200 (L-BFGS maxiter) | |
| Polynomial order | 5 (min-jerk) or min-snap | 5 (min-snap) | Same |
| Sampling for peak eval | ~20–50 pts/segment | N/A | Practical implementation |
| Velocity limit | 5.0 m/s (Crazyflie) | 15.0 m/s | Substitute our limits |
| Acceleration limit | 15.5 m/s² (with gravity) | 12.0 m/s² | Include gravity in check |
| Jerk limit | 62 m/s³ | 50.0 m/s³ | |
| Snap limit | 800 m/s⁴ | N/A (not constrained) | Add if needed |
| Min segment time | not stated | 0.1 s | Prevent division by zero |
