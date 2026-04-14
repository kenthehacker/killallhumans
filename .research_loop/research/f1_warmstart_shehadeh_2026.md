# Efficient Trajectory Optimization via F1 Data-Driven Initialization
- **URL**: https://arxiv.org/abs/2603.07126
- **Year**: 2026
- **Authors**: Samir Shehadeh, Lukas Kutsch, Nils Dengler, Sicong Pan, Maren Bennewitz
- **Venue**: arXiv:2603.07126 [cs.RO], submitted March 7, 2026

---

## Key Contribution

Trajectory optimization solvers for autonomous racing are highly sensitive to their initialization: a poor starting point can lead to slow convergence or convergence to a suboptimal local minimum. Standard baselines (centerline initialization or minimum-curvature path) provide no domain knowledge about what expert racing looks like, placing the optimizer in an arbitrary region of the objective landscape. This paper proposes addressing that gap by learning from Formula 1 telemetry. The authors construct a multi-track F1 dataset from 17 circuits and train a neural network that predicts expert-quality lateral raceline offsets `d(s)` from local track geometry alone — without explicitly modeling vehicle dynamics. This predicted raceline is used as a warm start (initialization vector) for a minimum-time optimal control solver (IPOPT), not as a replacement for the optimizer. The result is faster convergence and better-conditioned trajectories while preserving physics-based guarantees.

The central insight — and the paper's most important finding — is that the value of a good initialization is not primarily its Euclidean closeness to the optimal path. The F1-NN initialization achieves only marginally better geometric accuracy than the centerline baseline (RMSE 3.02 m vs. 3.07 m), yet achieves 17% fewer solver iterations. The authors explicitly state that "geometric agreement alone does not fully explain optimization performance," concluding that the learned initialization provides *structural proximity* to the optimal basin of attraction, not just metric proximity to the optimal point. This distinction has broad implications for any system using gradient-based trajectory optimization.

---

## Technical Approach

### Data Pipeline Construction

The dataset is built from GPS telemetry collected via the FastF1 Python library, covering 2024–2025 F1 race sessions across 17 circuits. Raw GPS data is sampled at approximately 3 Hz — too coarse for precise raceline representation. To increase density, 15 consecutive clean laps per driver are averaged together, augmenting spatial sampling. Laps corrupted by rain, safety car phases, or traffic incidents are filtered out to ensure ground truth reflects genuine maximum-performance driving.

Each track's centerline and boundary geometry is obtained from the TUM Racetrack Database (6 tracks) and Assetto Corsa simulation (11 tracks). A coarse-to-fine grid search optimization aligns the noisy GPS measurements to the track geometry. The aligned trajectories are resampled to a standardized 2.0 m spatial resolution along arc-length and represented as a reference-line with scalar lateral offset `d(s)` measured from the centerline. The dataset uses a 14/3 train/test split with 6 cross-validation folds — each fold uses a different 3-track test set so that every track is evaluated on a model that has never seen it during training.

### Neural Network Architecture

The model is a sequence-to-sequence predictor that operates in a sliding-window fashion over track geometry:

**Input:** Three windows over the 1D track representation parameterized by arc-length `s`:
- A *history window* carrying both local track geometry (curvature, left/right boundary offsets) and the previously predicted raceline offsets `d(s)`. This allows the model to condition on its own prior predictions, maintaining consistency along the predicted sequence.
- A *target window* at the current position carrying geometry only (no raceline history), representing what is to be predicted.
- A *future lookahead window* carrying only geometry for upcoming track sections. This enables the model to anticipate upcoming curves and plan the current offset accordingly.

**Encoder:** Dilated Temporal Convolutional Networks (TCNs) with two residual 1D convolutional blocks. Dilated convolutions expand the receptive field exponentially with depth without proportionally increasing parameter count, allowing the model to capture both local turn geometry and longer-range track structure within a single forward pass.

**Fusion:** A convolutional fusion module aggregates the three encoded windows, followed by multi-head temporal attention layers. The attention mechanism allows the model to focus on the most relevant parts of the future lookahead when predicting the current offset — effectively learning which upcoming features (tight hairpin, chicane, long straight) should influence the current lateral position decision.

**Output:** A two-layer MLP projecting the fused representation to a scalar lateral offset `d(s)` for the target position.

### Loss Function

Training uses a hybrid loss combining two complementary objectives:

```
L = 0.5 * (1 - cosine_similarity(p_flat, g_flat)) + 0.5 * mean(||p - g||_2)
```

- The **cosine alignment term** (weight 0.5) evaluates directional and structural consistency of the predicted raceline shape by treating the flattened offset vectors as directions. It penalizes predictions that have correct local offsets but the wrong global trajectory structure — e.g., apexing at the wrong gate in a chicane. This is the "global trajectory alignment" term.
- The **Euclidean distance term** (weight 0.5) enforces per-point metric accuracy. Training and validation loss tracked closely with this formulation, indicating no overfitting.

The equal weighting of these two terms reflects a deliberate design choice: pointwise accuracy alone is insufficient if the overall path structure is wrong, and structural alignment alone is insufficient if local offsets are far off.

### Warm-Starting the Trajectory Optimizer

The predicted raceline `d(s)` from the neural network is converted into an initial trajectory in (x, y) coordinates and speed profile `v(s)` by applying the offset to the track centerline. This initial trajectory is passed as the warm start `x0` to IPOPT, a primal-dual interior-point solver. The optimization problem minimizes lap time subject to vehicle dynamics constraints (simplified tire friction model with constant friction coefficient calibrated to F1 specifications), track boundary constraints (the vehicle must stay within the left/right boundaries), and friction/acceleration limits (combined lateral and longitudinal acceleration within the tire friction circle).

The key distinction from simply "using the neural network's output" is that the optimizer retains full freedom to modify the warm start. It will converge to whatever local minimum its initialization leads it toward. The hypothesis — validated by the results — is that F1-expert-like initializations are closer to high-quality local minima in the non-convex landscape of minimum-time optimization.

### Why Warm-Starting Avoids Local Minima

Non-convex trajectory optimization has multiple local minima corresponding to qualitatively different racing lines (e.g., taking a chicane tight vs. wide, late-apex vs. early-apex strategies). Gradient-based solvers like IPOPT are guaranteed to converge to *a* local minimum but not the *global* minimum. The quality of the local minimum found depends entirely on which basin of attraction the initialization falls in.

A centerline initialization is, by construction, a high-symmetry point that treats all gates equally and cuts no corners. When a track has a tight chicane that requires coordinated multi-gate offset (e.g., outside-inside-outside over three consecutive gates), the centerline initialization may sit in the basin of a degenerate local minimum that cannot discover this coordinated offset because the gradient direction at the centerline points away from the joint optimum. The F1 data provides a prior that reflects what coordinated multi-gate corner cutting looks like in expert human driving, placing the initialization in or near the correct basin.

---

## Results

### Simulation Across 17 F1 Circuits

| Method | Solver Iterations | Opt Time (s) | Gen Time (s) | Total Time (s) | Lap Time (s) |
|--------|-------------------|--------------|--------------|----------------|--------------|
| Centerline | 521.6 ± 149.0 | 149.5 ± 47.5 | — | 149.5 ± 47.5 | 85.70 ± 11.20 |
| Min-Curvature | 483.1 ± 112.1 | 136.2 ± 38.7 | 121.4 ± 50.3 | 257.6 ± 63.4 | 85.26 ± 11.17 |
| **F1-NN (proposed)** | **434.5 ± 103.1*** | **123.4 ± 40.5** | **0.63 ± 0.1** | **124.0 ± 40.5*†** | **85.22 ± 11.21*** |
| F1-GT (oracle) | 400.1 ± 67.6 | 112.3 ± 26.2 | — | 112.3 ± 26.2 | 85.14 ± 11.13 |

*p < 0.05 vs. centerline; †p < 0.05 vs. min-curvature

Key observations:
- F1-NN reduces solver iterations by **17%** (521.6 → 434.5) compared to centerline.
- F1-NN reduces total wall-clock time by **~17%** (149.5 → 124.0 s).
- The minimum-curvature baseline is *worse* in total runtime than cold-start centerline because its 121 s geometric pre-computation outweighs the modest iteration savings.
- F1-NN matches the oracle upper bound (F1-GT) within noise on lap time (85.22 vs. 85.14 s).
- Lap time differences between methods are small (~0.6%). The primary benefit is convergence speed and consistency, not a large lap time gain.
- The F1-GT oracle only saves ~35 more iterations than F1-NN, suggesting the network captures ~82% of the possible gain from expert initialization.

Geometric prediction accuracy (relative to F1 ground truth):
- RMSE: F1-NN 3.02 ± 0.69 m vs. Centerline 3.07 ± 0.61 m (difference: 0.05 m, ~1.6%)
- MAE: F1-NN 2.45 ± 0.63 m vs. Centerline 2.57 ± 0.55 m (difference: 0.12 m, ~4.7%)

The tiny geometric improvement vs. the 17% iteration reduction confirms the structural basin-of-attraction hypothesis.

### Hardware Validation (RoboRacer 1:10 Scale Platform)

| Initialization | Lap Time (s) | Avg Speed (m/s) | Lateral Tracking Error (m) |
|----------------|-------------|-----------------|---------------------------|
| Centerline | 7.090 ± 0.061 | 3.504 ± 1.010 | 0.165 ± 0.155 |
| **F1-NN** | **6.640 ± 0.069*** | **3.672 ± 1.014*** | **0.109 ± 0.075*** |

*p < 0.01

Hardware results are substantially more dramatic than simulation:
- **6.3% lap time reduction** (7.090 → 6.640 s).
- **34% reduction in lateral tracking error** (0.165 → 0.109 m).
- The higher real-world improvement (6.3% vs. 0.6% in simulation) is attributed to the physical controller tracking better-conditioned trajectories more faithfully — smoother paths generated from better initializations reduce the demands placed on the controller.

---

## Relevance to Our System

Our system's `planning/racing_line.py` uses L-BFGS-B with 10 multi-start initializations to optimize gate pass-through lateral offsets `d_i` for minimum path length plus smoothness penalty. We have documented that our L-BFGS-B optimizer converges to only 2 distinct qualitative basins from 10 random starts, and that the S-turn region (gate-3) persistently shows elevated tracking error (0.402–0.422 m) despite smoothness weight tuning.

The connection to Shehadeh et al. is direct and multi-layered:

**Same fundamental problem.** Our zero-initialization (`x0 = np.zeros(n*2)`) is the exact analog of centerline initialization in the F1 paper — it places the optimizer at the symmetric center of the search space, which for a multi-gate S-turn is precisely where the basin-of-attraction problem is worst. The optimal racing line through an S-turn requires coordinated offsets across gates 2–4 simultaneously (outside → inside → outside pattern), but the zero-initialization gradient at this neutral point may not point toward this coordinated solution.

**Our 2-basin convergence problem.** The F1 paper provides a direct explanation for why our 10 random starts only find 2 qualitatively distinct solutions: random uniform initialization (`rng.uniform(-max_off, max_off, n*2)`) does not systematically cover the distinct homotopy classes of the problem. For a track with k turns, the objective landscape has at least 2^k local minima corresponding to different combinations of turn-cutting strategies. Random starts are unlikely to sample all of these — they cluster around the most accessible basins (which correspond to "cut all turns to the right" and "cut all turns to the left" for a simple course). Expert-informed initialization targets specific homotopy classes that reflect real racing strategies.

**Data-driven vs. geometry-based initialization.** We lack F1 telemetry, but the paper's finding that its network only marginally outperforms the centerline geometrically (RMSE 3.02 vs. 3.07 m) suggests the actual mechanism is geometric heuristics, not learned dynamics. The network learned "cut inside of tight turns, stay wide on entry, position for next gate" — all of which can be computed analytically from our gate geometry. Our existing `_late_apex_init()` method already implements a version of this, but it currently only handles the case where `i+2 < len(centers)`, missing some gates.

**Interpolated initializations between the two known basins.** Since we have already identified that our optimizer converges to 2 basins, we can explicitly extract the offset vectors `x_basin1` and `x_basin2` from the two distinct solutions, then generate additional initializations as convex combinations `alpha * x_basin1 + (1-alpha) * x_basin2` for `alpha in {0.1, 0.3, 0.5, 0.7, 0.9}`. These interpolated points lie on the path between the two known basins and may pass through saddle points that lead to a third, better basin — this is directly motivated by the paper's insight that the solution landscape has multiple qualitatively different regions.

**Hardware error reduction transferability.** The RoboRacer result (34% lateral error reduction) is particularly relevant because the same mechanism — better-conditioned trajectories from better-initialized optimization — applies to our MPC tracker. Our gate-3 tracking error is dominated by the MPC needing to make aggressive corrections to a poorly-shaped trajectory, not by controller gain inadequacy. A qualitatively better racing line through the S-turn would reduce the demand on the controller.

---

## Actionable Takeaways

1. **Increase `maxiter` from 100 to at least 300 in `racing_line.py`.** The F1 paper's baseline solver ran 400–520 iterations before converging. Our current `options={"maxiter": 300}` cap (raised from 100 in iteration 23) may still be premature termination for complex S-turn geometry. The racing line is computed once offline; there is no runtime penalty for more iterations.

2. **Implement interpolated basin-crossing initializations.** After running all 10 L-BFGS-B starts, identify which converge to each of the 2 known basins. Generate 5 additional initializations as linear interpolations `alpha * x_basin_A + (1-alpha) * x_basin_B` for alpha in {0.1, 0.25, 0.5, 0.75, 0.9}. Run L-BFGS-B from these interpolated points. The paper's insight predicts these may explore regions of the objective landscape not accessible from random starts.

3. **Improve the `_late_apex_init()` geometric prior to cover all gates.** Currently the method skips the first and last gate (`if i == 0 or i >= n-1: continue`). For our 8-gate course, gates 1 and 8 are the start/finish region and may have valid turn geometry. Remove or loosen this restriction and add an entry for gate indices where the previous gate direction is well-defined.

4. **Add a "chicane recognition" initialization.** For consecutive left-right-left gate triplets (S-turns, like our gates 2–4), explicitly initialize the middle gate with a larger offset toward the inside of the first turn rather than a small zero or random offset. Specifically: if `cross_z(gate_i-1, gate_i) > 0` (left turn) then set `x0[i] = -0.5 * max_off` and `x0[i+1] = +0.5 * max_off` for the next gate simultaneously. This targets the coordinated multi-gate offset pattern that the paper shows random initialization misses.

5. **Log optimizer convergence metadata per run.** After each L-BFGS-B call in `_select_by_sim`, record `result.nit` (iterations used), `result.fun` (final objective value), and `result.success` (convergence flag). This distinguishes between the two failure modes the paper identifies: hitting `maxiter` (iteration budget insufficient) vs. converging to a poor local minimum (landscape problem requiring a different initialization). These require different remedies.

6. **Consider a pre-computed geometry-based raceline prior for gate-3.** Based on the F1 paper's finding that structural proximity matters more than Euclidean proximity, manually specify a "known good" initialization for the gate-3 S-turn based on manual inspection of the track layout. This is equivalent to the oracle (F1-GT) initialization in the paper and costs zero compute — just a hard-coded offset vector for known track `race_01.json`.

7. **Do not use minimum-curvature as a competing initialization method.** The paper quantitatively confirms that minimum-curvature initialization is worse than cold-start centerline in total runtime (257.6 vs. 149.5 s). While our context is different (we already compute a min-curvature path via `_late_apex_init()`), this result warns against treating minimum-curvature as a reliable warm start — its objective landscape properties are not systematically better than the centerline's.

---

## Limitations & Caveats

**Paper limitations:**

- **Simulation lap time gains are small** (~0.6%, 85.70 → 85.22 s). The primary contribution is convergence speed reduction, not solution quality. For our use case, solution quality (tracking error) is more important than compute time, so the simulation results are less directly motivating than the hardware results.
- **Geometric prediction accuracy nearly identical to centerline** (RMSE difference 0.05 m, ~1.6%). The network's actual mechanism is structural proximity, not metric accuracy. This means a carefully designed hand-crafted heuristic can replicate the initialization quality without requiring training data or a neural network.
- **Constant friction coefficient assumption** in vehicle dynamics. Real F1 tires exhibit highly variable friction depending on compound, temperature, and wear. The paper's vehicle model is a simplification. For drones, this is even more severe — quadrotor dynamics (thrust limits, rotor drag, inertia) are qualitatively different from ground vehicle dynamics.
- **Long straights degrade prediction quality** (noted explicitly for COTA circuit). The GPS density augmentation method averages over 15 laps, which may smooth out fine-grained steering variations on long straights. In our drone racing context this is less critical since gates are dense.
- **3 Hz GPS data is low-resolution**, requiring 15-lap averaging. This likely misses aggressive track edges and final braking points that expert drivers use, potentially under-representing the aggressiveness of expert racing lines.
- **Domain gap: F1 cars are not drones.** F1 vehicles operate in 2D (planar) with Ackermann steering, tire friction limits, and aerodynamic downforce. Quadrotors operate in 3D with thrust-limited dynamics, no friction-based turning, and different inertia characteristics. The neural network weights are not transferable; only the structural principle (expert-like initialization) is.
- **Only tested on a single hardware track** for the RoboRacer validation. The 34% lateral error reduction may not generalize across all track geometries.

**Caveats for our application:**

- Our optimization problem is low-dimensional: 8 gates × 2 DOF = 16 scalar variables. The F1 paper's track representations have hundreds to thousands of discretization points, making the objective landscape substantially more complex. In our lower-dimensional problem, random multi-start may already be sufficient to cover the main basins — the marginal benefit of data-driven initialization may be smaller.
- Our sim-based selection criterion (iteration 22–23) already partially compensates for poor initializations by using a kinematic simulation to evaluate all 10 L-BFGS-B results and pick the best. This is the AERO-MPPI ensemble approach. The F1 paper only evaluates the final optimizer output; it does not run competing initializations and select among them. Our approach is already more robust.
- The paper's 17% iteration reduction translates to roughly 26 seconds of saved compute time in a full-scale minimum-time solver. In our system, L-BFGS-B on 16 variables runs in milliseconds per start, so compute savings are irrelevant — only solution quality matters.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Track circuits in dataset | 17 | F1 circuits from FastF1 + TUM Racetrack DB |
| Track sources | 11 Assetto Corsa + 6 TUM | Dataset construction |
| GPS sample rate | ~3 Hz | Raw FastF1 telemetry |
| Laps averaged per driver | 15 | Spatial density augmentation |
| Spatial resampling resolution | 2.0 m | Arc-length standardization interval |
| Train / test split | 14 / 3 tracks | Cross-track generalization |
| Cross-validation folds | 6 | Each track tested on unseen model |
| Loss weights | 0.5 / 0.5 | Cosine alignment + Euclidean distance |
| Predicted raceline RMSE (F1-NN) | 3.02 ± 0.69 m | vs. F1 ground truth |
| Predicted raceline RMSE (centerline) | 3.07 ± 0.61 m | Reference baseline |
| Network inference time | 0.63 ± 0.1 s | Negligible vs. optimizer time |
| Solver iterations (centerline) | 521.6 ± 149.0 | Cold-start IPOPT baseline |
| Solver iterations (F1-NN) | 434.5 ± 103.1 | 17% reduction vs. centerline |
| Solver iterations (F1-GT oracle) | 400.1 ± 67.6 | Upper bound with perfect initialization |
| Total time (centerline) | 149.5 ± 47.5 s | Optimization only |
| Total time (min-curvature) | 257.6 ± 63.4 s | Including 121 s pre-computation |
| Total time (F1-NN) | 124.0 ± 40.5 s | ~17% faster than centerline |
| Simulation lap time (F1-NN) | 85.22 ± 11.21 s | vs. 85.70 s centerline (~0.6% faster) |
| Hardware lap time (centerline) | 7.090 ± 0.061 s | RoboRacer 1:10 |
| Hardware lap time (F1-NN) | 6.640 ± 0.069 s | 6.3% faster |
| Hardware lateral error (centerline) | 0.165 ± 0.155 m | RoboRacer |
| Hardware lateral error (F1-NN) | 0.109 ± 0.075 m | 34% reduction |
| Statistical significance | p < 0.05 (sim), p < 0.01 (hw) | Paired t-test across tracks/laps |

**Application to our L-BFGS-B optimizer:**

| Our Parameter | Current Value | F1-Paper Analogue | Recommended Change |
|---------------|---------------|-------------------|--------------------|
| `maxiter` | 300 | ~520 iterations to convergence | Raise to 500 for offline opt |
| `N_STARTS` | 10 | Single start (NN) + single oracle | Add 5 interpolated basin starts |
| Initialization 0 | `np.zeros(n*2)` | Centerline | Keep as fallback |
| Initialization 1 | `_late_apex_init()` | F1-NN warm start | Extend to cover all gates |
| Initialization 2–9 | Random uniform | N/A in paper | Replace some with interpolated |
| Gate-3 specific | None | Track-specific GT oracle | Add hard-coded S-turn prior |
