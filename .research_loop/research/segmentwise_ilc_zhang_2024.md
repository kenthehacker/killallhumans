# Segment-wise Learning Control for Trajectory Tracking

- **URL**: https://link.springer.com/article/10.1007/s11432-023-3845-6 (also at https://www.sciengine.com/SCIS/doi/10.1007/s11432-023-3845-6)
- **Authors**: Fan Zhang, Deyuan Meng, Kaiquan Cai
- **Year**: 2024 (published March 2024, received/available July 2023)
- **Venue**: Science China Information Sciences, Vol. 67, No. 3, Article 132203

---

## Key Contribution

The paper addresses a fundamental limitation of conventional adaptive iterative learning control (AILC): the requirement that every trial run (every "iteration") be executed over exactly the same time duration. In physical robotics, this assumption is routinely violated. A robot arm may finish early due to an obstacle, or be interrupted mid-task, producing trials of unequal length. Traditional AILC cannot reuse information from trials of different duration without corrupting the learned signal.

The core contribution is a **segment-wise AILC** scheme that decomposes the trajectory into discrete segments and maintains **virtual memory slots** — fixed-length storage buffers assigned to each segment. On each trial, only the segments that were actually traversed contribute new data; segments not reached are either held at prior values or filled from historical estimates stored in the virtual slots. This decomposition allows the algorithm to guarantee both **amplitude boundedness** (signals never diverge) and **energy boundedness** (total accumulated error remains finite) even under iteration-dependent trial lengths.

A secondary contribution is a proof that the practical AILC achieves the **perfect tracking objective** — i.e., tracking error converges to zero — provided the robot satisfies a "persistent full learning property," meaning that over a sufficiently long sequence of iterations, every segment of the trajectory is traversed at least once. When no segment is perpetually skipped, the virtual memory slots for all segments eventually receive real data, and the learned offset for each segment converges to its true value.

---

## Technical Approach

### The Problem Setting

The paper models a nonlinear robot manipulator executing a desired trajectory repeatedly. On each iteration k, the trial has duration T_k, which is an unknown, iteration-dependent quantity drawn from some set. The desired trajectory is defined over a fixed horizon [0, T_d]. When T_k < T_d, part of the trajectory is not executed; when T_k > T_d, the robot overshoots and redundant data is discarded.

Conventional AILC uses **point-wise learning**: the error at each individual time instant t on iteration k is used to update the control signal at that same instant t for iteration k+1. This requires T_k = T_d for all k. When T_k varies, there is simply no error signal at missing time instants, and zeroing missing entries introduces systematic bias.

### Segment Decomposition and Virtual Memory

Zhang et al. partition [0, T_d] into N fixed segments [0, s_1], [s_1, s_2], ..., [s_{N-1}, T_d]. For each segment i, a **virtual memory slot** M_i stores a compact representation of the learned control correction (an estimated offset or feedforward signal) for that segment. The key properties:

1. **Independence per segment**: The update rule for M_i depends only on tracking error measured while the robot is executing segment i. If a trial ends before reaching segment i, M_i is carried forward unchanged from the previous iteration — it is not corrupted by zeros or missing data.

2. **Compact storage**: Unlike point-wise methods that must store a full time series of length T_d, segment-wise storage only needs one representative value (or a low-dimensional parameterization) per segment. This is explicitly noted as a memory efficiency advantage over point-wise AILC.

3. **Adaptive gain**: The learning gain applied within each segment is adapted based on the inertia matrix estimate and task-specific uncertainty, following the standard AILC paradigm for nonlinear robotic systems (typically Lyapunov-based or composite energy function-based design).

### Amplitude and Energy Boundedness

The main stability result proceeds via a **composite energy function (CEF)** argument. The CEF is a Lyapunov-like function that combines:
- Squared tracking error norms across all segments
- Parameter estimation error terms for the adaptive weights
- Cross terms coupling adjacent segments

By showing that the CEF is monotonically non-increasing along the iteration axis (under appropriate learning gain selection), the authors prove:
- **Amplitude boundedness**: every element of the control input and parameter estimate remains within a compact set for all k
- **Energy boundedness**: the integral (or sum) of tracking error squared over all trials is finite

These are stronger than simple convergence-in-expectation results and do not rely on probabilistic assumptions about T_k.

### Persistent Full Learning and Convergence

The paper defines the **persistent full learning property**: there exists a finite window length L such that for every iteration k and every segment i, at least one of the L consecutive trials from k to k+L-1 traverses segment i. Under this condition (analogous to persistent excitation in adaptive control), the virtual memory slots receive fresh real data with bounded frequency, and the standard CEF decrease argument can be extended to show tracking error for every segment converges to zero.

This is the segment-wise counterpart of the classical ILC convergence theorem. Without persistent full learning (some segment permanently skipped), perfect tracking is unachievable by construction, but amplitude/energy boundedness still holds — a graceful degradation property.

### Experimental Validation

The authors validate on a visual robot manipulator platform built in **CoppeliaSim** (formerly V-REP) with MATLAB. The platform simulates a multi-joint arm executing repeated pick-and-place trajectories where the stopping time varies due to simulated perception delays. Quantitative results show:
- Tracking error converges segment-by-segment over iterations
- Memory usage is reduced substantially versus point-wise AILC
- Amplitude of parameter estimates remains bounded throughout

---

## Results

The paper reports that segment-wise AILC with virtual memory slots achieves:
- **Zero steady-state tracking error** across all trajectory segments under persistent full learning
- **Bounded parameter estimates and control inputs** for all iterations, even in early trials where many segments are unvisited
- **Lower memory footprint** than conventional point-wise AILC — the required storage per iteration is O(N) (number of segments) rather than O(T_d / dt) (full time-series length)
- Successful CoppeliaSim simulation results demonstrating convergence in tracking error over approximately 20-50 iterations across different trial length distributions

The main theoretical results are presented as formal lemmas and theorems with complete proofs. The convergence rate is not quantified explicitly (this is common in adaptive ILC — only asymptotic convergence is typically shown), but the empirical curves in simulation show geometric decay of segment-wise error norms over iterations.

---

## Relevance to Our System

**Our system** applies an offline ILC to compute position-offset corrections layered on top of a PD controller that tracks polynomial (min-snap) trajectories. The issue is that a global ILC offset — a single correction vector applied uniformly across the full trajectory — improves tracking through the helix gates but simultaneously degrades the S-turn gates. This anti-correlation occurs because the helix section and the S-turn section have structurally different aerodynamic and geometric demands: the helix requires sustained centripetal correction in one direction, while the S-turn requires corrections that alternate sign rapidly. A single global offset is a scalar compromise that cannot simultaneously satisfy both.

Zhang et al.'s segment-wise ILC directly addresses this class of problem:

1. **Per-segment independence**: By assigning each trajectory section (S-turn, helix, straight connector) its own virtual memory slot and learning update, the ILC for the helix section learns purely from helix-phase tracking error, and the ILC for the S-turn section learns purely from S-turn error. The two corrections are fully decoupled in the update law.

2. **Applicability even without iteration-varying trial lengths**: Our drone does complete all gates every run (assuming no crash), so we do not face the missing-data problem the paper was primarily designed to solve. However, the segment decomposition idea is independently valuable: we can partition the trajectory by gate indices or arc-length thresholds, maintain a separate learned offset vector per segment, and update each vector from the per-gate tracking error provided in `simulation.per_gate_avg_error`. This is exactly the structure the paper proposes.

3. **Virtual memory slot implementation**: In our offline ILC, each segment's slot corresponds to a small numpy array (e.g., shape [3,] for xyz offset) that is updated as:
   ```
   M_i[k+1] = M_i[k] + gamma_i * mean(error_i[k])
   ```
   where `error_i[k]` is the tracking error measured while the drone was within the arc-length bounds of segment i. This is functionally identical to the virtual memory slot update in Zhang et al., specialized to the case where all segments are always visited.

4. **Amplitude boundedness as a safety property**: The paper's proof that segment-wise AILC keeps all corrections bounded is directly useful. In our drone system, an unbounded ILC offset would cause a crash. The boundedness guarantee (achievable by standard gain-clipping or by choosing learning rate gamma_i < 1/sigma_max(L_i) where L_i is the local Lipschitz constant of the dynamics near segment i) is essential for safe offline iteration.

5. **Persistent full learning**: Since our drone visits all gates in every iteration (no skipped segments), we automatically satisfy persistent full learning. This means we get the full convergence guarantee — per-segment offsets converge to the true systematic error in each phase — not merely the weaker boundedness result.

---

## Actionable Takeaways

1. **Split the trajectory at gate boundaries** (or at arc-length thresholds between the S-turn cluster and helix cluster) and maintain a separate ILC offset array per section. Do not pool error gradients across sections.

2. **Use per-gate error from `simulation.per_gate_avg_error`** directly as the signal for each segment's virtual memory update. Each gate (or gate group) gets its own slot M_i.

3. **Set section-specific learning rates** gamma_i. The helix section, with its tighter geometry and larger centripetal demands, may require a smaller gamma to avoid oscillation. The S-turn section, with rapidly alternating curvature, may require an even smaller gamma and possibly a position-along-arc weighting to avoid the correction averaging out to zero.

4. **Do not share offset state between sections when resetting**: If the helix offset is reset (e.g., after a major trajectory change), the S-turn offset should be reset independently. Virtual memory slots are logically independent.

5. **Apply the "persistent full learning" framing as a convergence check**: After N iterations, plot the per-segment error reduction. If a segment's error is not decreasing, the learning rate for that segment may be too high (oscillation) or the segment boundaries may be misaligned (the segment captures mixed dynamics).

6. **For the helix specifically**, consider that the tracking error may have a strong radial component (centripetal direction) that is nearly constant along the helix. A per-segment offset learned over a few iterations should correct this efficiently. The S-turn error is more phase-sensitive; consider further sub-segmenting the S-turn into left-curve and right-curve halves with separate offsets.

7. **Memory cost is negligible**: N segments × 3 xyz components × float64 = trivially small. There is no computational argument against fine-grained segmentation.

---

## Limitations & Caveats

1. **Asymptotic convergence only**: The theoretical results show convergence as k → ∞ but give no finite-iteration bound. In practice, our offline ILC budget is ~25-50 iterations. The paper does not provide a formula for how many iterations are needed to reach a given error level. Empirical tuning of gamma_i is required.

2. **Nonlinear manipulator model, not quadrotor**: The paper's dynamics are a serial rigid-body robot arm with joint torques as inputs. The quadrotor is a fundamentally different dynamical system with underactuation, rotor lag, aerodynamic coupling, and thrust-limited attitude tracking. The paper's CEF proofs do not transfer directly. The segment-wise decomposition idea transfers; the specific gain design formulas (which depend on inertia matrix bounds) do not.

3. **No treatment of trajectory phase error**: The paper assumes the mapping from time to trajectory position is fixed and known. In our system, the drone may arrive at a gate early or late, meaning the "segment" membership of each timestep is uncertain. Spatial (arc-length) rather than temporal indexing of segments avoids this issue. The paper implicitly uses temporal segmentation; we should use spatial (arc-length) segmentation.

4. **Iteration-dependent period as the primary motivation**: Much of the paper's technical machinery (virtual memory, CEF extension) is designed to handle the case where T_k varies. Since our drone completes all gates reliably (when not crashed), this machinery is more than we need. The segment decomposition idea is the transferable element; the full adaptive apparatus may be overkill.

5. **Single trajectory only**: The paper addresses repetitive execution of one fixed desired trajectory. Our racing line changes between iterations as the optimizer may update the polynomial waypoints. If the trajectory itself changes significantly, the accumulated ILC offsets become stale and must be discarded or decayed. A forgetting factor (M_i[k+1] = lambda * M_i[k] + gamma_i * error_i[k], with lambda slightly below 1) would help.

6. **No perception or gate detection noise**: The paper assumes perfect state feedback. Our system has EKF uncertainty (reported as ekf_uncertainty_m). Noisy error signals will cause noise in the ILC update. Low-pass filtering (e.g., exponential moving average of per-gate error before feeding into the ILC update) mitigates this.

---

## Key Parameters / Constants

| Symbol | Meaning | Relevance to Our System |
|--------|---------|------------------------|
| N | Number of trajectory segments | Set to number of gate groups (e.g., 2–6 depending on track structure) |
| M_i | Virtual memory slot for segment i | Per-segment ILC offset array, shape [3,] (xyz) |
| gamma_i | Learning rate for segment i | Tune per segment; start ~0.3–0.5, reduce for oscillating segments |
| L | Window length for persistent full learning | Equals 1 if all gates visited every run (our case) |
| lambda | Forgetting factor (if trajectory changes) | Recommended 0.90–0.95 to handle trajectory drift between iterations |
| T_k | Trial duration (iteration-dependent) | Not an issue for our system (all trials complete) |
| CEF | Composite Energy Function | Lyapunov-like stability certificate; guides gain selection |
| sigma_max(L_i) | Max singular value of local Lipschitz constant | Upper bound for gamma_i to ensure convergence; estimated from per-segment error sensitivity |

The paper is published in *Science China Information Sciences* Vol. 67, No. 3 (March 2024), DOI [10.1007/s11432-023-3845-6](https://link.springer.com/article/10.1007/s11432-023-3845-6). The SciEngine version is available at [https://www.sciengine.com/SCIS/doi/10.1007/s11432-023-3845-6](https://www.sciengine.com/SCIS/doi/10.1007/s11432-023-3845-6).
