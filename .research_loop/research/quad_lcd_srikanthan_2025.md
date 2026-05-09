# Quad-LCD: Layered Control Decomposition Enables Actuator-Feasible Quadrotor Trajectory Planning
- **URL**: https://arxiv.org/abs/2505.10228
- **Authors**: Anusha Srikanthan, Hanli Zhang, Spencer Folk, Vijay Kumar, Nikolai Matni (UPenn GRASP Lab)
- **Year**: 2025
- **Venue**: ICRA 2025 Workshop on 25 Years of Aerial Robotics / arXiv

---

## Key Contribution

Quad-LCD addresses the practical gap between trajectory planning and execution in aggressive quadrotor flight: minimum-snap planners produce smooth, dynamically clean trajectories in theory, but when executed on real hardware with fixed feedback controllers, motor saturation causes uncontrolled drift and crashes. The standard solution — tightening trajectory constraints globally or re-tuning controllers — sacrifices speed and generality.

The paper's core insight is to **keep the controller fixed** and instead reshape the trajectory to lie within the controller's actual tracking capability. They do this by adding a learned, controller-aware cost term g^ctrl(ξ, c) to the standard minimum-snap optimization. This cost function is trained offline via simulation rollouts that expose which polynomial coefficient patterns lead to tracking failure (motor saturation, large errors). The result is a planner that is aware of what its downstream controller can actually execute, not just what the kinematic/dynamic model permits. This decomposition — treating planning and control as a layered system with a learned interface — is the titular "Layered Control Decomposition."

---

## Technical Approach

### Problem Formulation

Trajectory planning is cast as:

    minimize  c^T H c  +  g^ctrl(ξ, c)
    subject to  A c = b

where:
- `c` is the stacked vector of piecewise polynomial coefficients (order n, s segments)
- `H` is block-diagonal with snap cost matrices (minimizes snap = 4th derivative of position)
- `A c = b` enforces endpoint continuity and smoothness constraints (position, velocity, acceleration matching at segment boundaries and at waypoints)
- `g^ctrl(ξ, c)` is the learned controller-tracking cost

The standard minimum-snap formulation only has the first term. The learned term g^ctrl is the contribution.

### Layered Control Decomposition

The architecture has two layers:
1. **Feedback controller (fixed)**: A nonlinear geometric controller (or any fixed controller). Its tracking capability is treated as a black box that can be probed via simulation. The controller is NOT modified — only the reference trajectory feeding into it is changed.
2. **Reference trajectory planner (optimized)**: The planner knows about the controller's limitations through g^ctrl and avoids generating references that will cause saturation.

This is "control decomposition" because the overall task (fly aggressively) is decomposed into a planning layer and a control layer with a learned interface between them. The interface is the tracking cost function.

### Learning g^ctrl

The function g^ctrl maps polynomial trajectory coefficients to a scalar tracking cost. Key design choices:

**Why not learn on trajectory states?** Prior learned-cost approaches parameterize inputs as time-discretized state sequences. The input dimension then scales with `n_timesteps × state_dim`, making it impractical to train with enough data for dense time grids.

**Why polynomial coefficients?** The input dimension is fixed at `n_segments × (n_order + 1) × 3` regardless of trajectory duration. For typical race configs this is on the order of hundreds of values — manageable for an MLP.

**Architecture**: 3-layer MLP with hidden sizes {100, 100, 20}, ReLU activations. Lightweight by design — intended to be evaluated in an inner optimization loop.

**Training procedure**:
1. Sample 200,000 random minimum-snap trajectories (random waypoints in a 10×10×10 m³ volume, 1–3 m spacing, ~2 m/s average speed).
2. Roll out each trajectory in RotorPy simulator with the fixed controller. Record max tracking error and/or motor saturation events.
3. Label each trajectory with a tracking cost (crash = high cost, clean tracking = low cost).
4. Train the MLP on (polynomial coefficients → tracking cost) pairs.
5. Separate networks trained for 5 drag coefficient configurations (spanning [0.002, 0.008] N/(m/s) for horizontal, [0.007, 0.013] for vertical).

**Optimization at runtime**: The full objective (snap cost + g^ctrl) is optimized jointly. Since both terms are smooth in `c`, gradient-based methods (L-BFGS or similar) apply. The learned term provides gradient information pushing the optimizer away from coefficient patterns that cause tracking failures.

### Sim-to-Real Transfer

Validated on Crazyflie 2.0 with motion capture at 100 Hz. The learned cost function trained entirely in simulation transfers zero-shot to hardware. The paper attributes this to: (a) coefficient-space learning captures structural patterns rather than noise-sensitive state sequences, and (b) training across 5 drag configurations provides implicit robustness to model mismatch.

---

## Results

### Simulation (RotorPy)

Crash is defined as max tracking error > 1.5 m over a trajectory:

| Method | Crash Rate |
|--------|-----------|
| Quad-LCD (proposed) | **6%** |
| Min-snap + geometric controller (MS-GC) | 41% |
| Min-snap + drag compensation (MS-GCD) | 54% |

This is a 49% relative crash rate reduction vs. MS-GC (the most natural baseline).

Notably, adding drag compensation (MS-GCD) made things *worse* than the baseline. The paper's interpretation: drag compensation alters the effective dynamics but the controller is still not aware of when the trajectory exceeds tracking capability, so it fails more in high-speed regimes.

### Hardware (Crazyflie 2.0)

Successful zero-shot transfer on aggressive maneuvers. Quantitative hardware metrics are not fully detailed in the 4-page paper (it is a workshop paper), but the authors report stable flight on hardware trajectories that caused crashes in the baseline planners.

### Data collection cost

200,000 simulated trajectories across 40 CPU cores takes 6 hours. This is a one-time offline cost. At test time, the MLP forward pass is cheap (~microseconds), so runtime overhead is negligible.

---

## Relevance to Our System

Our system's core issue is exactly the one Quad-LCD targets: **the trajectory planner does not know what the controller can track**. Our current mitigations are:

1. A global `max_compression = 0.68` floor in `_topp_retime()` that prevents over-squeezing all segments uniformly (too coarse).
2. Per-gate `_inflate_sharp_turns()` that applies angle-based and centripetal-acceleration-based inflation to specific segments.
3. S-turn compound inflation at 10% for the second gate of an S-turn pair.

These are all hand-tuned heuristics derived from empirical observation of which gates had high tracking error. The problem is that the inflation logic is reactive (tuned to observed failures) and doesn't generalize: helix-style sections with closely-spaced gates need segment-specific timing that the global `max_compression` floor can't provide, and the turn-angle heuristic misses failures caused by velocity/acceleration direction changes that aren't pure angle changes.

**Quad-LCD's approach addresses this at the root**: instead of hand-coding inflation rules, it learns a cost function from simulation rollouts that tells the optimizer exactly how much each polynomial pattern stresses the controller. Applied to our system:

- **Affected module**: `planning/trajectory_optimizer.py`, specifically `_topp_retime()` and `_inflate_sharp_turns()`. The learned cost would replace or augment the hand-tuned inflation rules.
- **Secondary benefit**: `control/mpc_tracker.py` remains unchanged — the approach explicitly does not require modifying the controller, which is important for us since our PD+feedforward controller is already tuned and stable.
- **EKF/estimation modules**: Not directly affected. The benefit is cleaner reference trajectories that stay within the controller's tracking band, which indirectly reduces state estimation stress (lower residuals → EKF stays confident).

The most direct analog in our system is replacing the `max_compression` global floor with a **per-segment feasibility score** derived from the polynomial coefficients of that segment. This score could be computed by a small learned model or by a data-driven lookup that maps (segment polynomial, approach speed, turn angle) → compression limit.

Our sim (PyBullet via `sim_pybullet/`) can serve the same role as RotorPy in Quad-LCD: we can collect rollouts of candidate trajectories, measure tracking error per segment, and train a mapping from polynomial coefficients to per-segment tracking cost.

---

## Actionable Takeaways

1. **Collect per-segment tracking error data from our PyBullet sim.** Run ~5,000–20,000 varied trajectories through the race course (or simplified sub-courses around the helix section), record polynomial coefficients and per-segment tracking error. This is the Quad-LCD data collection step adapted to our sim.

2. **Train a lightweight segment feasibility predictor.** A small MLP (or even a linear model with polynomial features) mapping (segment polynomial coefficients, approach velocity magnitude, turn angle) → max tracking error. This can replace the ad-hoc `_inflate_sharp_turns` heuristics.

3. **Replace `max_compression` global floor with per-segment compression limits.** During `_topp_retime()`, instead of `new_time = max(new_time, times[i] * 0.68)`, query the feasibility predictor for segment i and set the compression floor segment-specifically. Segments the predictor rates as high-stress get a higher floor (less compression), low-stress segments can be compressed more aggressively.

4. **Augment the L-BFGS objective with the learned cost.** If the feasibility predictor is differentiable (MLP with smooth activations), add it to the objective in `_optimize_time_allocation()`. This allows the optimizer to jointly optimize time allocation and trajectory shape to avoid high-stress regions rather than post-hoc correcting with inflation.

5. **Use the Quad-LCD training regime for multi-drag robustness.** Train across PyBullet configurations with varied motor response times and drag (our drone's mass/drag parameters have uncertainty). This gives the predictor implicit robustness to model mismatch without requiring explicit online adaptation.

6. **Prioritize the helix sub-course in data collection.** Our current per-gate error shows gate-3 is consistently the worst. Concentrate data collection on trajectories that pass through the high-curvature sections to get the predictor more accurate where it matters most.

7. **Keep the controller fixed.** Quad-LCD validates that reshaping the reference is sufficient — do not attempt to re-tune MPC gains simultaneously. Changes should be isolated to `trajectory_optimizer.py`.

---

## Limitations & Caveats

**Scale and speed regime.** Quad-LCD experiments use ~2 m/s average speed in a 10×10×10 m³ domain with 1–3 m waypoint spacing on a lightweight Crazyflie. Our race course runs significantly faster (target < 14s total, segment speeds up to 10–15 m/s). The polynomial coefficient patterns that cause saturation at 2 m/s may differ qualitatively from those at 10 m/s. We would need to collect training data at our actual operating speeds, not Quad-LCD's slower regime.

**Motor saturation vs. tracking error.** Quad-LCD's training signal is motor saturation events. Our failure mode is tracking error on a PD+feedforward controller in a kinematic-ish sim. These are related but not identical — we need to define our own training signal (e.g., per-segment max cross-track error) rather than directly reusing their crash labels.

**4-page workshop paper.** This is a short workshop paper, not a full conference paper. Implementation details are sparse. The exact loss function for training g^ctrl, optimizer hyperparameters, and the full hardware comparison are not disclosed. We would need to make implementation choices that the paper leaves open.

**Fixed controller assumption.** The method is only valid if the controller is genuinely fixed. If we later switch from PD+feedforward to a proper MPC (as suggested by MASTERPLAN.md), the learned cost function becomes stale and must be retrained. Retraining cost is 200,000 sim rollouts, which is non-trivial but manageable.

**No gate-passage constraints.** Quad-LCD plans through waypoints but does not explicitly enforce gate passage geometry (entry/exit normals, gate width). Our system has hard gate passage constraints that narrow the feasible trajectory space. The learned cost may need to be conditioned on gate geometry, or the data collection needs to be done on gate-constrained trajectories only.

**Generalization to new courses.** The learned cost is trained on random waypoint configurations. Its generalization to our specific race course geometry (helix, S-turns, etc.) depends on whether those patterns appear in the training distribution. Data collection should explicitly include helix-like and S-turn-like sub-trajectories.

---

## Key Parameters / Constants

The following numerical values from Quad-LCD are potentially useful as starting points:

- **MLP architecture**: 3 hidden layers, sizes {100, 100, 20}, ReLU. Total parameters ~11,000 — fast to evaluate.
- **Training set size**: 200,000 trajectories. For our faster iteration, 20,000–50,000 may suffice given our smaller course.
- **Polynomial order**: Not explicitly stated, but consistent with typical minimum-snap (order 7–9 per segment).
- **Motor response time modeled**: 5 ms. Our PyBullet sim may have different actuation delays — check `sim_pybullet/` config.
- **Motor noise**: 100 rad/s std dev. Relevant if we add noise to sim rollouts for robustness.
- **Crash threshold used as label**: max tracking error > 1.5 m. For our system, a relevant threshold might be max per-segment tracking error > 0.5 m (our avg error threshold).
- **Drag coefficient range trained**: [0.002, 0.008] N/(m/s) horizontal, [0.007, 0.013] N/(m/s) vertical. If we want multi-config robustness, vary our PyBullet drag in a similar relative range (~4× span).
- **Waypoint spacing for training**: 1–3 m. For our course, use actual gate-to-gate distances (which may be 3–8 m) to keep the training distribution on-manifold.
- **Domain size for training**: 10×10×10 m³. Our course extent should bound the sampling region.
- **Data collection wall time**: 6 hours on 40 CPU cores ≈ 240 CPU-hours for 200,000 trajectories. At 50,000 trajectories on 8 cores this scales to ~30 CPU-hours.
