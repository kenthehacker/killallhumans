# Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization

- **URL**: https://arxiv.org/abs/2603.07126
- **Authors**: Samir Shehadeh, Lukas Kutsch, Nils Dengler, Sicong Pan, Maren Bennewitz
- **Year**: 2026 (submitted March 7, 2026)
- **Venue**: arXiv preprint (cs.RO)

---

## Key Contribution

This paper attacks a fundamental problem in trajectory optimization for racing: the optimization landscape has multiple local minima (basins), and convergence to a good solution is highly sensitive to the initialization. Starting from a naive initial guess (like the centerline) can lead the solver to a suboptimal basin, or can require many more iterations to escape a poor initialization and find a competitive lap time. This is precisely the same multi-basin problem that afflicts our racing line optimizer's `_select_by_sim()` method.

The authors propose learning a warm-start initializer from real Formula 1 telemetry data across 17 tracks. A neural network trained on F1 GPS data learns to predict lateral offset profiles d(s) — how far from the track centerline the expert driver positions the vehicle at each arc-length position s. This predicted racing line is then used to initialize a standard minimum-time nonlinear programming (NLP) solver, replacing the centerline or min-curvature heuristics that are typically used as default initializations.

The key insight is that expert human drivers have already solved the multi-basin problem empirically through thousands of laps: the GPS data implicitly encodes which basin is globally competitive. By using this data as a prior, the solver is initialized near the correct basin, requires fewer iterations, and converges faster — all while reaching essentially the same final lap time quality as the optimal solution.

---

## Technical Approach

### Dataset Construction

The dataset covers 17 Formula 1 circuits, including Baku, Barcelona, COTA, Hungaroring, Imola, Interlagos, and others. Track geometries come from two sources: the Assetto Corsa racing simulation and the TUM Racetrack Database. F1 GPS telemetry is obtained via the FastF1 Python library at approximately 3 Hz sampling rate, which is then up-sampled and aligned to the track reference line.

For each track and each driver session, a mean trajectory is computed across 15 consecutive laps to reduce noise. The trajectory is represented in a track-aligned Frenet frame: arc-length s along the centerline, and lateral offset d(s) perpendicular to it. Track curvature κ(s) and lateral boundary offsets (left and right track edges) are encoded at Δs = 2.0 m spatial resolution.

The dataset is split: 14 tracks for training, 3 tracks for test. This leave-out evaluation prevents the model from simply memorizing track-specific solutions and forces generalization from geometry alone.

### Neural Network Architecture

The model takes as input a context window of track geometry (curvature κ(s) and boundary offsets) both behind and ahead of the current position, along with expert raceline context from the history window. The architecture uses:

- **Encoding**: Dilated temporal convolutional networks (TCN) with residual 1D convolutional blocks. Dilated convolutions allow the model to see a wide receptive field (capturing long-range track features like approaching corners) without proportionally increasing parameter count.
- **Fusion**: A convolutional fusion module combines geometry and raceline context, followed by multi-head temporal attention.
- **Output**: An MLP head predicts lateral offset d(s) for target arc-length segments.

The training loss is a weighted hybrid of two terms:
- Cosine alignment loss (weight 0.5): penalizes angular deviation between predicted and ground-truth raceline tangent directions
- Euclidean distance loss (weight 0.5): penalizes spatial offset between predicted and ground-truth lateral positions

This combination encourages the model to match both the shape (direction) and position of the expert raceline.

### Optimization Integration

The learned raceline prediction is used to seed the minimum-time optimal control solver (IPOPT interior-point NLP solver). Concretely, the neural network outputs d(s) which is converted to x(s), y(s) Cartesian coordinates. This initial trajectory x₀ is passed to the solver as the starting iterate, replacing the default centerline initialization.

The minimum-time optimal control formulation minimizes:

```
min_{x(s), y(s), v(s)} ∫ (1/v(s)) ds
```

subject to:
- Track boundary constraints: x(s), y(s) within left/right boundaries
- Vehicle dynamic feasibility via simplified point-mass model
- Longitudinal and lateral force constraints from tire friction circle
- Constant friction coefficient μ (a key limiting assumption)

IPOPT solves this as a structured NLP with thousands of variables (one per arc-length discretization point per track). The number of solver iterations and total wall-clock time are recorded as convergence metrics.

---

## Results

### Neural Network Accuracy

| Metric | Value |
|--------|-------|
| RMSE vs F1 ground truth | 3.02 ± 0.69 m |
| MAE vs F1 ground truth | 2.45 ± 0.63 m |

A 2.45–3.02 m prediction error might seem large, but the key point is that this is sufficient to place the solver near the correct basin. The optimization then refines from this warm start to the actual local optimum. The warm start doesn't need to be perfect — it needs to be close enough to the right basin.

### Optimization Convergence (17-track average)

| Initialization Method | Iterations | Opt. Time (s) | Total Runtime (s) | Lap Time (s) |
|---|---|---|---|---|
| Centerline (CL) | 521.6 ± 149.0 | 149.5 ± 47.5 | 149.5 ± 47.5 | 85.70 ± 11.20 |
| Min-Curvature (MC) | 483.1 ± 112.1 | 136.2 ± 38.7 | 257.6 ± 63.4 | 85.26 ± 11.17 |
| **F1-NN (Ours)** | **434.5 ± 103.1** | **123.4 ± 40.5** | **124.0 ± 40.5** | **85.22 ± 11.21** |
| F1-GT (Oracle) | 400.1 ± 67.6 | 112.3 ± 26.2 | 112.3 ± 26.2 | 85.14 ± 11.13 |

Key observations:
- **17% reduction in solver iterations** (521.6 → 434.5) vs. centerline
- **17% reduction in total runtime** (149.5 s → 124.0 s)
- **Negligible inference cost**: 0.63 ± 0.1 s for the neural network prediction
- **Final lap time preserved**: 85.22 s vs. 85.70 s (0.6% improvement) — essentially the same optimum
- Min-curvature has higher total runtime because it requires a second optimization pass to compute the MC line before using it as initialization

The F1-GT oracle result (using actual F1 data directly) is the upper bound: 400 iterations, 112.3 s runtime, 85.14 s lap time. The F1-NN result is 92% of the way to the oracle, with only 0.63 s of inference overhead.

### Hardware Validation (1:10 Scale RoboRacer Platform)

| Metric | Centerline Init | F1-NN Init |
|--------|-----|-------|
| Lap Time | 7.090 ± 0.061 s | **6.640 ± 0.069 s** |
| Average Speed | 3.504 ± 1.010 m/s | **3.672 ± 1.014 m/s** |
| Lateral Error | 0.165 ± 0.155 m | **0.109 ± 0.075 m** |

The F1-NN initialization produces a 6.3% faster lap time, 4.8% higher average speed, and 34% lower lateral tracking error on real hardware. All improvements are statistically significant (p < 0.01). This demonstrates successful transfer from full-scale F1 simulation to a 1:10 scale platform with entirely different dynamics.

---

## Relevance to Our System

This paper is directly relevant to our non-deterministic basin selection problem in `planning/racing_line.py`. Our `_select_by_sim()` method runs full trajectory optimization + kinematic simulation on multiple racing line candidates to determine which one performs best. This approach is:

1. **Slow**: Running full trajectory optimization N times is expensive
2. **Non-deterministic**: The optimizer may converge to different local optima depending on initialization, making `_select_by_sim()` non-reproducible
3. **Circular**: We're using trajectory quality to select the racing line, but the racing line quality determines trajectory quality — a feedback loop

The F1 paper's solution is the complementary approach to our problem: instead of trying to evaluate which of N basins is best (our approach), use a data-driven prior to directly predict a good initialization that lands in the right basin from the start.

**Concretely, applied to our system:**

Our racing line optimizer (`racing_line.py`) optimizes lateral offsets {d_i} for each gate. Currently, the initial offset guess is either zero (centerline) or some heuristic. The F1 paper suggests we could train a small neural network or use a lookup table from previous successful races to predict good initial offsets d_i, bypassing the need for `_select_by_sim()` entirely.

The core insight that transfers directly: a good warm start eliminates the multi-basin problem by ensuring the solver starts near the globally competitive solution. For our drone racing context, a "good warm start" means initializing the lateral offsets at a configuration consistent with the known racing line solution — which we can fix deterministically.

**Immediate application**: Freeze the racing line offsets to the result from the previous successful run (our "F1-GT" equivalent). Store the offsets in `state.json` and load them as the warm start for trajectory optimization. This completely eliminates `_select_by_sim()` and makes the system deterministic.

---

## Actionable Takeaways

1. **Store the winning racing line offsets in `state.json`** as a warm start. After each successful benchmark run, serialize the optimized lateral offsets {d_i} for each gate. Load these as the initialization for the next run's trajectory optimizer. This gives us the "F1-GT oracle" behavior — the best possible warm start — without needing a neural network.

2. **Eliminate `_select_by_sim()` entirely**: If we initialize from the stored winning offsets, trajectory optimization will reliably converge to the same (or better) solution in fewer iterations. We don't need to run N parallel candidates and select by simulation.

3. **Use min-curvature as a deterministic fallback**: For cold starts (no prior solution available), min-curvature initialization is deterministic and converges 7% faster than centerline, at the cost of a second optimization pass. This is better than the current approach of random/multiple initializations.

4. **Train a small gate-offset predictor**: For future generalization to new tracks, train a lightweight TCN or MLP that maps gate positions + track curvatures to lateral offsets. The architecture doesn't need to be as complex as the full F1-NN — our gate count is small (10–20 gates), so even a simple polynomial fit on curvature might suffice.

5. **Separate neural network inference from solver**: Run the offset predictor once at startup (0.63 s inference budget) before the first trajectory optimization. This cost is negligible vs. the 120–150 s solver time the paper benchmarks, and even more negligible vs. our competition timeline.

6. **Validate basin consistency**: After fixing the warm start, run the trajectory optimizer 10 times with identical initialization and confirm that the resulting lap time variance drops to near-zero. This verifies that we've eliminated the non-determinism.

---

## Limitations & Caveats

**4-wheel ground vehicle, not UAV**: The F1 paper optimizes a point-mass car dynamics model with tire friction constraints. Our drone has fundamentally different dynamics (thrust-limited, 3D, rotor aerodynamics). The specific architecture and dataset do not transfer directly — we would need to replace the F1 GPS data with our own simulation run data.

**Constant friction coefficient**: The paper assumes μ is constant along the track. For drones, the analogous assumption is constant air density and no wind — reasonable for indoor competition.

**Frenet frame representation**: The lateral offset d(s) parameterization makes sense for ground vehicles following a well-defined track centerline. For drone gate racing, the "centerline" is less well-defined between gates (it's a straight line or spline through gate centers). The Frenet frame analogy still applies, but the geometry is simpler.

**No real-time replanning**: The F1 approach is an offline initialization for a lap-time optimizer. It does not support online replanning during the race. In our context, this is acceptable — we precompute the trajectory before the race.

**Scale gap**: The real-hardware validation is on a 1:10 scale car, not full-scale F1. The paper notes "significant domain shift" and still achieves good results, which is encouraging for our sim-to-real transfer scenario.

**Data availability**: The paper uses proprietary F1 telemetry via FastF1. For our drone system, we'd need to generate equivalent data from simulation runs of our own system, which means bootstrapping from existing benchmark results.

---

## Key Parameters / Constants

- Training tracks: 14; test tracks: 3 (of 17 total F1 circuits)
- Spatial resolution: Δs = 2.0 m along arc-length
- Laps averaged per driver per track: 15 consecutive laps
- GPS sampling rate: ~3 Hz (upsampled during preprocessing)
- Neural network inference time: 0.63 ± 0.1 s
- Training loss weights: cosine alignment = 0.5, Euclidean distance = 0.5
- RMSE of NN prediction: 3.02 ± 0.69 m lateral
- Solver iterations (CL baseline): 521.6 ± 149.0
- Solver iterations (F1-NN): 434.5 ± 103.1 (17% reduction)
- Solver iterations (F1-GT oracle): 400.1 ± 67.6
- Total runtime (CL): 149.5 ± 47.5 s
- Total runtime (F1-NN): 124.0 ± 40.5 s (17% reduction)
- Hardware lap time improvement: 7.090 s → 6.640 s (6.3% faster)
- Hardware lateral error improvement: 0.165 m → 0.109 m (34% lower)
- Statistical significance: p < 0.01 for all hardware metrics
