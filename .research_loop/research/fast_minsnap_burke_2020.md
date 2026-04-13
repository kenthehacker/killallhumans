# Generating Minimum-Snap Trajectories Really Fast (Burke 2020)

- **URL**: https://arxiv.org/abs/2008.00595
- **Authors**: Declan Burke, Airlie Chapman, Iman Shames (University of Melbourne)
- **Venue**: IEEE/RSJ IROS 2020, pp. 1487–1492
- **Year**: 2020

---

## Key Contribution

Burke et al. present an algorithm for generating minimum-snap polynomial trajectories for quadrotors that achieves **linear computational complexity O(N) in the number of spline segments N**, compared to the cubic O(N³) complexity of prior closed-form methods (Mellinger & Kumar 2011, Richter et al. 2015). The algorithm:

1. Exploits the banded/structured form of the minimum-snap QP to enable linear-time solve.
2. Reformulates the QP in nondimensional (normalized) coordinates to eliminate ill-conditioning arising from large segment times.
3. Embeds this O(N) fixed-time solver inside an iterative time allocation loop, where each iteration also runs in O(N).

Demonstrated scale: a C++ implementation generates a 500,000-segment trajectory in 156 seconds on an Intel Core i7-8650U (1.9 GHz, 16 GB RAM), where the bottleneck becomes memory allocation rather than arithmetic.

---

## Technical Approach

### Polynomial Trajectory Representation

The trajectory for each flat output (x, y, z, ψ) is a continuous piecewise polynomial spline π(t; p) over k segments defined by non-decreasing segment boundary times T = [t₀, …, t_k]. The conventional choice is a polynomial of degree 2r−1 per segment, where r is the derivative order being minimized. For minimum snap, r = 4 (minimizing the 4th derivative), giving **degree-7 (order-8) polynomials**. For yaw ψ, r = 2 (minimum acceleration) suffices.

Each segment i uses the monomial basis shifted to t_{i-1}:

    π_i(t) = Σ_{l=0}^{2r-1} p_i^l · (t − t_{i-1})^l

With k segments and 2r coefficients per segment, there are k·2r total unknowns.

### QP Problem Structure

The problem is formulated as four independent QPs (one per flat output). The cost minimizes the integral of the squared rth derivative:

    minimize  p^T Q p
    subject to  A p = d

Where:
- **Q** is block-diagonal with per-segment cost submatrices Q_i (symmetric positive-definite)
- **A** encodes waypoint constraints at segment boundaries: position equality at each waypoint, and continuity of derivatives 1…2r−1 at interior junctions
- **d** is the vector of boundary values (positions, and specified endpoint derivatives)

The standard unconstrained closed-form solution is obtained by solving the KKT conditions of the equality-constrained QP. The critical issue is that the constraint matrix A contains **transposed confluent Vandermonde matrices** Γ_i, which are notoriously ill-conditioned when segment times are large or vary widely.

### The Ill-Conditioning Problem

The condition number κ(Γ_i · A_i^{-1}) grows rapidly with segment duration Δ_i = t_i − t_{i-1}. For a segment of duration Δ_i, the polynomial basis evaluations t^l scale as Δ_i^l for l up to 2r−1 = 7, creating a ratio of ~Δ_i^7 between smallest and largest basis terms. This makes the Vandermonde matrix ill-conditioned for even modestly large Δ_i (e.g., Δ_i = 2 s gives a ratio of 128; Δ_i = 10 s gives a ratio of 10^7).

Prior work by Richter et al. improved numerical stability over the naive formulation but still suffered from conditioning issues for very large segment counts.

### Nondimensional Reformulation (Key Fix)

Burke et al. show that the condition number κ(Γ_i · A_i^{-1}) **remains constant when Δ_i = 1**, i.e., it depends only on the segment duration, not on the absolute times t_{i-1} or t_i. The fix is to normalize each segment to unit duration by working in scaled local time:

    τ = (t − t_{i-1}) / Δ_i ∈ [0, 1]

In this nondimensional program (referred to as "Program (4)" in the paper), each segment is evaluated over τ ∈ [0, 1] regardless of its actual duration. The actual segment duration Δ_i only appears in the derivative scaling (chain rule), not in the Vandermonde matrix entries. This keeps the condition number **bounded and independent of segment time**.

This is the critical insight: **numerical instability is caused by large Δ_i entering the Vandermonde matrix as powers, and normalizing to [0,1] prevents this entirely.**

### Linear Complexity via Structured Solve (Algorithm 1)

The key computational gain comes from recognizing that the KKT system for the multi-segment QP has a **block-banded structure**:

- The cost matrix Q is block-diagonal (each block handles one segment independently)
- The constraint matrix A, while coupling adjacent segments via continuity constraints, is block-bidiagonal or block-tridiagonal
- The resulting KKT (saddle-point) matrix therefore has **bandwidth proportional to the polynomial order r, independent of k**

Naive Gaussian elimination on a k×n system costs O(k³n³); exploiting the banded structure reduces this to **O(k · n³)** where n = 2r is the fixed polynomial order per segment. Since n is constant (fixed at 8 for degree-7 snap-minimizing polynomials), the overall complexity is **O(k)**.

This is the same structural insight used in banded linear solvers (e.g., LAPACK's `dgbsv`): when the non-zero bandwidth is O(1), Gaussian elimination is O(N).

The upper bound on the number of constraint equations is k·n + (k−1)·s, where s is the number of continuity conditions at interior waypoints (s = 2r−1 = 7 for full C³ continuity). Solving this banded system is O(k) vs. O(k³) for the dense approach.

### Time Allocation

Burke et al. address two related problems:

1. **Fixed-time problem**: Given segment times T, find minimum-snap coefficients → solved by Algorithm 1 in O(k)
2. **Free-time problem**: Find segment times T* that minimize total time (or some proxy) subject to dynamics constraints → solved iteratively

For the iterative time allocation, each outer iteration:
- Evaluates trajectory quality / constraint violations at the current T
- Updates T using gradient information or line search
- Re-solves the fixed-time QP (O(k) per call)

Each iteration thus costs O(k), and with a bounded number of iterations (typically 10–50 for convergence), the total time allocation also scales as O(k) in practice.

The initial time allocation heuristic used is **distance-proportional**: T_i = d_i / v_avg, where d_i is the Euclidean distance between waypoints i and i+1, and v_avg is a fraction of maximum speed. This is the same heuristic used in most prior work.

### Benchmark Results

On an Intel Core i7-8650U (1.9 GHz, 16 GB RAM) in C++:
- 500,000-segment trajectory generated in **156.01 seconds**
- This equates to ~3,200 segments/second, or ~0.31 ms per segment at this scale
- For typical racing trajectories (10–50 segments), extrapolation implies sub-millisecond generation
- Memory allocation becomes the bottleneck before arithmetic at extreme scale

Comparison with prior state-of-the-art (Richter/Bry closed-form):
- Burke et al. is both **faster** and handles **larger trajectories** than prior methods
- The advantage is most significant at k > ~100 segments; for k < 20, overhead differences are minor
- The ZJU-FAST-Lab "Double Description" method (Wang et al., ICRA 2021) subsequently claims further improvement, with analytical gradient support

---

## Results

- O(N) complexity for both fixed-time polynomial solve and iterative time reallocation
- Numerically stable for 500,000+ segments (condition number bounded by nondimensional reformulation)
- Outperforms Richter/Bry closed-form and Bry et al. on both speed and scalability benchmarks
- Experimental validation via a real quadrotor flight test demonstrating smooth, executable trajectories
- Applicable to any differentially flat system, not just quadrotors

---

## Relevance to Our System

Our current implementation in `planning/trajectory_optimizer.py` has several structural gaps relative to Burke et al.:

### 1. Polynomial Order: 5th vs 7th Degree

Our `_min_snap_1d()` function uses a **5th-order (degree-5) polynomial** with 6 boundary conditions (pos, vel, accel at each end). True minimum-snap requires a **7th-order (degree-7) polynomial** (8 coefficients) that minimizes the integral of the 4th derivative (snap). Our polynomial minimizes the integral of the 3rd derivative (jerk), making it technically a **minimum-jerk** formulation, not minimum-snap. This reduces trajectory smoothness and dynamical feasibility for aggressive maneuvers.

### 2. Per-Segment Independent Solve (No Global Continuity)

Our implementation solves each segment independently with manually-specified endpoint velocities and accelerations (`end_vel`, `end_accel`). Burke et al. solve a **global joint QP** that enforces exact continuity of all derivatives up to order 2r−1 simultaneously. Our approach introduces artificial breaks in higher derivatives at waypoints, which creates impulsive control inputs at gate transitions.

### 3. No Nondimensional Normalization

Our `_min_snap_1d()` directly forms a 3×3 system in physical time `T`. For segment times of T = 1–3 seconds (typical race segments), the Vandermonde matrix entries scale as T^3 to T^5 (up to 243 at T=3), which risks conditioning issues. Burke's normalization to τ ∈ [0,1] would be cheap to implement and would ensure stability.

### 4. Time Allocation Heuristic

Our `_initial_time_allocation` uses `dist / (0.6 · v_max)` — nearly identical to Burke's recommendation. However, our `_optimize_time_allocation` uses L-BFGS on log-time variables with a handcrafted penalty, rather than Burke's iterative approach using the fast O(N) solver in the inner loop. Our approach decouples time allocation from the polynomial solve, losing information about true constraint satisfaction.

### 5. Trajectory Evaluation Performance

Our trajectory evaluation loop (in `_generate_trajectory`) iterates over each axis and each sample time in Python, with `_poly_deriv_eval` rebuilding derivative coefficients on every call. This is O(k · n_samples · order²) in Python and likely the dominant runtime cost. Burke's approach computes coefficients analytically once per segment.

---

## Actionable Takeaways

In priority order for our racing system:

**P1 — Fix polynomial order (5th → 7th degree)**
Replace `_min_snap_1d` with a true minimum-snap 7th-order polynomial formulation. The boundary conditions become: pos, vel, accel, jerk at each endpoint (8 BCs for 8 unknowns). This directly fixes the jerk discontinuities at gate transitions.

**P2 — Apply nondimensional normalization**
Before solving the 3×3 or 4×4 system in `_min_snap_1d`, normalize τ = t/T so the Vandermonde-like matrix entries are in [0,1]. Scale derivatives back using chain rule: dp/dt = (1/T) dp/dτ. This is a 2-line change and eliminates conditioning risk for long segments.

**P3 — Vectorize polynomial evaluation**
Replace the Python loop in `_generate_trajectory` with `np.polyval` or precomputed coefficient matrices. For each axis, evaluate all n_samples at once using `np.polyval(coeffs[::-1], t_local)`. This can reduce evaluation time by 10–50x and is the fastest path to hitting the >100 Hz loop frequency target.

**P4 — Consider a global multi-segment QP for smooth junctions**
For final competition preparation, replacing per-segment independent solves with a global QP (even a small banded one for 10–20 gate race courses) would yield smoother trajectories. This is the full Burke approach and is O(N) with banded solve. Implementation complexity is moderate; potential gain in tracking error is significant.

**P5 — Time allocation: curvature-aware speed (not just distance)**
Burke and follow-on work (TOGT, AM-Traj) suggest penalizing curvature-induced centripetal acceleration as a tighter proxy for dynamics feasibility. Our current code estimates curvature from turn angles but doesn't propagate this into segment-level polynomial feasibility checks. Using the polynomial coefficients to compute max acceleration per segment (analytically, at the critical points) would replace the heuristic with exact constraint checking.

---

## Limitations & Caveats

- **Fixed time allocation only**: Algorithm 1 (the O(N) fast solve) assumes segment times are given. The time-optimal free-time problem still requires iteration; Burke's contribution is making each iteration O(N) rather than O(N³).
- **No inequality constraints in the fast path**: The O(N) result holds for equality-constrained QP (waypoints + continuity). Adding obstacle-avoidance inequality constraints returns the problem to standard QP territory (no longer trivially O(N) banded). Burke et al. acknowledge this and reference separate work for the inequality case.
- **C++ implementation**: The 156s/500k-segment result is in C++. A Python implementation would be ~10–100x slower, meaning the true benefit for our system is through the vectorization approach (P3 above) rather than raw algorithm scaling.
- **Advantage appears mainly at k > ~100**: For a typical 10-gate race course with 10 segments, the O(N) vs O(N³) difference is negligible (10³ = 1000 operations either way). The real benefit is the nondimensional conditioning fix and the cleaner global QP formulation.
- **Superseded by Wang et al. (ICRA 2021)**: The ZJU-FAST-Lab "Double Description" method achieves the same O(N) complexity with additional support for analytical gradients w.r.t. time allocation and waypoints, enabling joint optimization. Burke's method has no published open-source reference implementation.

---

## Key Parameters / Constants

| Parameter | Burke's Choice | Our Current Value | Notes |
|---|---|---|---|
| Polynomial degree | 7 (minimum-snap) | 5 (minimum-jerk) | Fix: upgrade to degree 7 |
| BCs per segment | 8 (pos/vel/accel/jerk at each end) | 6 (pos/vel/accel at each end) | Fix: add jerk BCs |
| Derivative order minimized | r = 4 (snap) | r = 3 (jerk, implicit) | Fix: reformulate cost |
| Time normalization | τ ∈ [0,1] per segment | None (raw physical time T) | Fix: normalize |
| Initial time heuristic | dist / v_avg | dist / (0.6 · v_max) | Essentially identical |
| Complexity | O(N) per iteration | O(N · n_samples) Python | Fix: vectorize |
| Constraint structure | Global banded QP | Per-segment independent | Future: global QP |
| Min segment time | Not specified | 0.2–0.3 s | Keep for stability |
