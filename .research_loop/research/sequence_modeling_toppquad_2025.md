# Sequence Modeling for Time-Optimal Quadrotor Trajectory Optimization

- **URL**: https://arxiv.org/abs/2506.13915
- **Authors**: Katherine Mao, Hongzhan Yu, Ruipeng Zhang, Igor Spasojevic, M Ani Hsieh, Sicun Gao, Vijay Kumar
- **Affiliations**: University of Pennsylvania (Mao, Spasojevic, Hsieh, Kumar); UC San Diego (Yu, Zhang, Gao)
- **Year**: 2025 (submitted June 16, 2025)
- **Code**: github.com/lbTOPPQuad

---

## Key Contribution

The paper trains sequence models (primarily LSTM encoder-decoders) to imitate TOPPQuad, a classical Time-Optimal Path Parameterization solver for quadrotors. The central insight is that the full time-optimal trajectory can be reconstructed from just two scalar sequences along the path: the **squared-speed profile** h(s) and the **yaw angle** cos θ_z(s), where s is arc-length progress. By predicting only these two outputs instead of solving the 16×N-variable optimization problem that TOPPQuad operates on, the learned model achieves a 136× speedup (0.078 s vs. 10.656 s per trajectory) with zero failure rate and nearly identical tracking performance.

The paper also introduces a **sampling-based Backward Reachable Tube (BRT) approximation** for verifying dynamic feasibility of the predicted profiles — a rare theoretical contribution that connects learning-based planners to reachability analysis.

---

## Technical Approach

### Separation of Geometric Path and Time Parameterization

The approach exploits the standard TOPP decomposition: trajectory design separates into (1) a **collision-free geometric path** γ(s) parameterized by arc length, and (2) a **time parameterization** that assigns a speed to every point along the path. Once the speed profile is fixed, the full state sequence (position, velocity, acceleration, attitude, motor thrusts) follows from the quadrotor's **differential flatness** — the flat outputs [x, y, z, ψ] determine all other states algebraically.

This is critical: the learned model does not need to predict positions, velocities, or attitudes explicitly. It only predicts two scalars per waypoint:
- **h(s)** = squared speed at arc-length position s (squared-speed representation avoids numerical issues at zero speed)
- **cos θ_z(s)** = yaw encoding, which determines heading from which attitude is recovered via geometric controller

Full motor thrusts are recovered post-hoc from the predicted flat outputs using standard inverse dynamics.

### Input Representation

The path is discretized into **100 equally-spaced waypoints**. At each waypoint, the input features are:
- 3D position γ(s): 3 dimensions
- First derivative γ'(s): 3 dimensions (encodes tangent direction)
- Second derivative γ''(s): 3 dimensions (encodes curvature)

Total: 9-dimensional input per waypoint. The derivatives encode geometric information the model can exploit: high curvature → lower predicted h(s); low curvature straight → higher h(s).

### Architecture: LSTM Encoder-Decoder

The best model is an **LSTM encoder-decoder with non-parameterized attention**:
- Encoder LSTM reads the full 100-waypoint path sequence
- Attention mechanism provides a context vector summarizing the global path
- Decoder LSTM generates h(s) and cos θ_z(s) **auto-regressively** (each output conditions on all previous outputs)

Auto-regressive decoding is key: speed at position s naturally depends on speeds downstream (you must decelerate before a turn) and upstream (acceleration constraints). The LSTM's hidden state implicitly represents this bidirectional dependency.

Other tested architectures and their failure modes:
- **Transformer encoder-decoder**: 76% failure rate — likely due to attention over-smoothing speed profiles, losing the sharp deceleration needed before turns
- **Encoder-only Transformer**: 4–6% failure rate — acceptable but LSTM was better
- **Per-step MLP**: 0.010 s (1066× faster) but 6% failure rate — cannot model inter-step dependencies, so it cannot ensure feasibility

### Data Augmentation for Robustness

Random perturbations with scales 0.001–0.1 m are applied to input path waypoints during training. The model designated LSTM-0.1 (trained with ε = 0.1 m perturbations) maintained >92% probability of staying within the Backward Reachable Tube across all tested perturbation levels. Without augmentation, small perturbations caused BRT exit. This finding directly motivates the BRT robustness framework.

### Backward Reachable Tube (BRT) Framework

The BRT analysis asks: given a reference speed profile h*(s) from TOPPQuad, what set of predicted profiles h(s) will result in a tracking controller staying within acceptable thrust bounds? The paper approximates the BRT via sampling: perturb the input path, collect the resulting predicted profiles, and check whether resulting thrust demands stay within physical limits. This gives an empirical "in-BRT probability" metric (90–94% for augmented models).

---

## Results

### Computational Speed

| Method | Compute Time | Speedup vs. TOPPQuad | Failure Rate |
|--------|-------------|----------------------|--------------|
| TOPPQuad (baseline) | 10.656 s | 1× | 0% |
| LSTM encoder-decoder | 0.078 s | **136×** | 0% |
| Per-step MLP | 0.010 s | 1066× | 6% |
| AllocNet (prior work) | 0.277 s | 38× | 28% |
| MFBOTrajectory | 13,609 s | 0.001× | 0% |

The LSTM is the best practical choice: 136× faster than TOPPQuad with zero failure rate.

### Trajectory Quality

Simulation (test set of unseen paths):
- **LSTM maximum position deviation**: 0.074 m (vs. TOPPQuad baseline 0.053 m) — 40% higher but still well within practical tolerances
- **LSTM thrust violation**: 0.002 N — essentially negligible
- **Average speed**: LSTM 3.498 m/s vs. TOPPQuad 3.477 m/s — nearly identical, confirming time-optimality is preserved

Hardware (CrazyFlie 2.0 in motion capture, 8 unseen paths):
- **LSTM tracking error**: 0.355 m vs. TOPPQuad 0.347 m — within 2.3% of optimal
- Hardware experiments confirm that sim-to-real gap is modest and does not negate the speedup benefit

### Generalization

The model successfully generalizes to path lengths **outside the training distribution**, demonstrating that the LSTM learns generalizable geometric-to-speed mappings rather than memorizing specific routes.

---

## Relevance to Our System

Our current system generates min-snap polynomial trajectories with L-BFGS time allocation. The key problem we face is the **local minimum trap** in the racing line optimizer: the fast basin (12.78 s) is untrackable, and the smooth basin (17.70 s) is too slow. We need a principled way to find intermediate time allocations that the kinematic PD controller can actually follow.

This paper is highly relevant in three ways:

### 1. Squared-Speed Profile as a Separate Optimization Variable

We currently couple path optimization and time parameterization inside the same L-BFGS call, letting the optimizer co-optimize path shape and timing simultaneously. This produces local minima sensitivity because the loss surface has ridges where path smoothness and speed tradeoff discontinuously.

The TOPPQuad approach says: **fix the geometric path first, then solve for speed separately**. Concretely:
- Our `RacingLineOptimizer` produces a fixed waypoint sequence (path only, no timing)
- A separate `SpeedProfiler` then assigns speeds using the forward-backward curvature scan

The missing piece: our `SpeedProfiler` uses a heuristic `sqrt(a_max / curvature)` formula, but does not enforce global time-optimality subject to thrust constraints. A proper TOPP solve — even a simple one using the classical bang-bang structure — would guarantee that the speed profile is physically achievable while minimizing time.

### 2. Improving Our SpeedProfiler with TOPP Principles

Our `SpeedProfiler._compute_curvatures()` → `sqrt(a_max / k)` formula is a curvature-speed heuristic but ignores:
- Thrust saturation (we have a 20 N limit; speed profiles that require >20 N thrust to follow will cause tracking error)
- The joint constraint that v(s), v'(s), v''(s) must all satisfy thrust bounds simultaneously

A TOPP-style solver on our waypoint sequences would:
1. Compute the Maximum Velocity Curve (MVC): at each arc-length position s, find the maximum speed consistent with thrust limits
2. Apply the forward-backward integration (already in our code as the two-pass scan) but starting from the MVC, not from a heuristic curvature formula
3. The result is the globally time-optimal speed profile that the actual quadrotor physics can follow

This is implementable without neural networks: the classical TOPP algorithm (Bobrow 1985, Shin & McKay 1985) is a well-understood solve.

### 3. Race Time Recovery via Speed Profile Decoupling

Our priority-2 backlog item is recovering from the 13.31 → 17.70 s race time regression. The current approach (binary search on smoothness_weight) is heuristic and risks flipping back to the fast-but-untrackable basin.

An alternative strategy inspired by this paper:
1. Keep the smooth racing line (smoothness_weight = 0.40) — it produces a geometrically tractable path
2. Re-optimize the speed profile independently using a TOPP-style or learning-free bang-bang solver against actual thrust constraints
3. The decoupled speed solve has no local minimum problem — it is a convex problem on the MVC

Expected benefit: we might recover 2–4 s of race time (from 17.70 s toward 14–15 s) while maintaining <0.3 m tracking error, because the smooth geometric path means curvature is well-behaved and higher speeds are achievable on straights.

### 4. The 136× Speedup Is Not Directly Relevant

Our system pre-computes trajectories before the race; latency is not a bottleneck. However, the speedup matters indirectly: if trajectory computation is fast (sub-second), we could run a **multi-start strategy** — try 10–20 random initializations of the racing line optimizer and pick the one that is simultaneously fast and trackable. This breaks the local minimum trap without neural networks.

---

## Actionable Takeaways

1. **Implement TOPP-style speed profile on top of the fixed racing line.** After `RacingLineOptimizer.optimize()` returns waypoints, replace the `SpeedProfiler` heuristic with a proper forward-backward integration starting from the thrust-constrained Maximum Velocity Curve. This is ~100 lines of Python, no new dependencies, and directly addresses the race time regression while guaranteeing physical feasibility.

2. **Separate the time allocation optimization from path shape optimization.** Currently, `trajectory_optimizer.py`'s L-BFGS optimizes segment times with an acceleration penalty that conflates geometry and timing. Consider fixing segment times from the TOPP speed profile and running min-snap only on polynomial coefficients, not on times. This removes the local minimum that manifests as the fast/slow basin bifurcation.

3. **Use the squared-speed parameterization h(s) = v(s)^2 internally.** Squared speed avoids numerical singularities at velocity reversals and is the natural variable for the thrust-speed feasibility constraint `h_max(s) = f(curvature, thrust_limit)`. Our `SpeedProfiler` uses speed directly; switching to h(s) makes the forward-backward passes simpler and more numerically stable.

4. **Multi-start L-BFGS with a speed-profile fitness criterion.** Run racing line L-BFGS from 5–10 random initializations. Score each solution by: (TOPP-computed race time) × (1 + max_curvature_normalized). Pick the minimum. This exploits the fast TOPP evaluation (<<1 s per trajectory) to escape the local minimum trap.

5. **Do not attempt to train the LSTM model.** Our track is a fixed course; the training-data generation, ML infrastructure, and generalization concerns are not justified when the underlying TOPP algorithm is implementable directly and solves our specific local-minimum problem.

---

## Limitations & Caveats

1. **Requires retraining for new quadrotor platforms.** The LSTM learns implicit dynamics constraints specific to the training platform (CrazyFlie at 5 m/s limit). For our system (up to 15 m/s, kinematic sim with PD controller), we would need to retrain — but this is irrelevant since we are not using the neural approach.

2. **The 6% failure rate of the per-step MLP is a fundamental limitation of feed-forward architectures for speed profiles.** Any learned model that cannot condition on future path curvature will occasionally predict infeasible speed at gates. Our TOPP-style implementation avoids this by using the analytical backward pass.

3. **Sim-to-real gap expands at higher speeds.** The paper tests at 2 m/s (hardware) vs. 5 m/s (sim). At our target speeds of 5–10 m/s, thrust feasibility constraints are significantly tighter and the BRT becomes smaller. Our kinematic sim already approximates this gap via the PD controller's effective speed limit.

4. **The BRT analysis is approximate (sampling-based, not guaranteed).** It verifies feasibility probabilistically. For competition, we need deterministic guarantees, which requires either conservative constraint margins or physics-based forward simulation (which our benchmark already provides).

5. **Path discretization at 100 points is fine for smooth paths but may undersample aggressive turns.** Our race track has helix sections where curvature changes rapidly within a 1–2 m segment. 100 points over a 100 m path gives 1 m resolution — adequate, but the S-turn issue (gate-3, 0.422 m error) might benefit from adaptive resolution near high-curvature waypoints.

6. **The paper focuses on single-lap trajectories on smooth, pre-planned paths.** It does not address gate uncertainty, replanning on detection failure, or the kind of trajectory-tracking interaction that our kinematic sim PD controller exhibits. The theoretical guarantees apply only to the offline planning phase.

---

## Key Parameters / Constants

From the paper's implementation:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Path discretization | 100 equally-spaced waypoints | Input to LSTM |
| Input features per waypoint | 9 | position (3) + γ'(s) (3) + γ''(s) (3) |
| Output per waypoint | 2 | h(s) = squared speed, cos θ_z(s) = yaw |
| Training dataset size | 10,000 trajectories | Main experiments |
| Data augmentation range (ε) | 0.001–0.1 m | Path perturbation |
| Simulation speed limit | 5 m/s | |
| Hardware speed limit | 2 m/s | CrazyFlie |
| TOPPQuad solve time | 10.656 s | Baseline |
| LSTM inference time | 0.078 s | 136× speedup |
| LSTM tracking error (sim) | 0.074 m | vs. 0.053 m for TOPPQuad |
| LSTM tracking error (hardware) | 0.355 m | vs. 0.347 m for TOPPQuad |
| LSTM failure rate | 0% | On test set |
| In-BRT probability (augmented) | >92% | For LSTM-0.1 |

### Relevant Constraint Formulation

The key physical feasibility constraint used internally by TOPPQuad (and what any TOPP-style implementation must enforce) is:

```
f_i(s) ∈ [f_i_min, f_i_max]  for all i, s
```

where f_i are motor thrusts. Given the flat output derivatives and h(s), motor thrusts are computed via the inverse dynamics chain: flat outputs → attitude (quaternion) → body frame accelerations → motor mixing. The MVC is the envelope where at least one motor is exactly at its saturation limit — the tightest achievable speed profile.

For our system: `max_thrust = 20.0 N`, `mass = 1.0 kg`, `g = 9.81 m/s²` → maximum specific thrust ≈ 20 m/s². This is the constraint our TOPP-style implementation must respect when computing h_max(s) at each waypoint.
