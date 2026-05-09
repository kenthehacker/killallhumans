# Efficient Trajectory Optimization via F1 Data-Driven Initialization

**URL:** https://arxiv.org/abs/2603.07126
**Authors:** Samir Shehadeh, Lukas Kutsch, Nils Dengler, Sicong Pan, Maren Bennewitz
**Year:** 2026
**Venue:** arXiv:2603.07126 [cs.RO] (submitted March 7, 2026)

---

## 1. Key Contribution

The paper addresses a fundamental weakness in trajectory optimization for autonomous racing: cold-start initialization. Standard approaches initialize the optimizer from a geometric heuristic (centerline or minimum-curvature path), which can be far from the optimal solution in the objective landscape — particularly in complex track sections. The core contribution is a learned neural network that ingests local track geometry and predicts an expert-like lateral raceline offset `d(s)`, trained on real-world Formula 1 telemetry from 17 circuits (via the FastF1 library). This predicted raceline serves as a warm start for a minimum-time optimal control solver.

The key claim is that a better initialization point places the optimizer structurally closer to the global (or a high-quality local) minimum, reducing both the number of solver iterations and total wall-clock time — without sacrificing the final optimized lap time. The paper validates this claim both in simulation across all 17 F1 tracks and on physical hardware (1:10 scale RoboRacer platform).

---

## 2. Technical Approach

### 2.1 Data Pipeline

- **Source:** FastF1 library, GPS telemetry from 2024–2025 F1 race sessions at approximately 3 Hz.
- **Density augmentation:** 15 consecutive laps per driver are averaged to increase spatial sampling density beyond the raw 3 Hz limit.
- **Filtering:** Rain-affected laps, safety car phases, and traffic-affected segments are removed to ensure the ground truth reflects genuine high-performance driving.
- **Standardization:** Trajectories are resampled to 2.0 m spatial resolution along arc-length and represented as a reference-line with lateral offset `d(s)`.
- **Track coverage:** 17 circuits used; 14/3 train/test split with 6 cross-validation folds to ensure each track is tested with a model that has never seen it.

### 2.2 Network Architecture

The network is a sequence-to-sequence raceline predictor operating in a sliding window fashion:

- **Input:** Three window types over local track geometry — (a) a history window carrying both geometry and the predicted raceline offset so far, (b) a target window with geometry only, and (c) a future lookahead window with geometry only. This asymmetric design allows the model to condition on previously predicted offsets while predicting the next segment.
- **Encoder:** Dilated Temporal Convolutional Networks (TCNs) with two residual 1D convolutional blocks. Dilated convolutions expand the receptive field without proportionally increasing parameters.
- **Fusion:** A convolutional fusion module followed by multi-head temporal attention layers to aggregate information across the three window types.
- **Output:** A two-layer MLP predicting the scalar lateral offset `d(s)` at the target position.

### 2.3 Loss Function

A hybrid weighted loss combining two complementary terms:

```
L_train = 0.5 * (1 - cos(p_flat, g_flat)) + 0.5 * mean(||p - g||_2)
```

- **Cosine alignment term** (0.5 weight): Encourages directional and structural consistency of the raceline shape — the "global trajectory alignment."
- **Euclidean distance term** (0.5 weight): Enforces per-point metric accuracy between prediction and ground truth. Training is described as "smooth and stable" with training and validation loss tracking closely.

### 2.4 How Initialization Affects Convergence and Local Minima

The paper's analysis reveals that geometric prediction accuracy (RMSE/MAE of the predicted raceline vs. F1 ground truth) only marginally differs between F1-NN and the centerline baseline:

- RMSE: F1-NN 3.02 ± 0.69 m vs. Centerline 3.07 ± 0.61 m
- MAE: F1-NN 2.45 ± 0.63 m vs. Centerline 2.57 ± 0.55 m

These differences are small. Yet the convergence speedup is substantial (17% fewer iterations). The authors explicitly note: "geometric agreement alone does not fully explain optimization performance," concluding that the learned initialization must be providing structural proximity to the optimal solution — placing the optimizer in a basin of attraction for a better local minimum, not merely a closer point in Euclidean space. This is a critical insight: the value of a good initialization is its position in the *objective landscape*, not just its physical proximity to the optimal path.

For the minimum-curvature baseline, the total runtime is *worse* than cold-start centerline despite better geometric initialization. This is because minimum-curvature computation itself takes ~121 s, outweighing any optimization savings. F1-NN initialization costs only 0.63 s for generation while still reducing solver iterations.

---

## 3. Results (Quantitative)

### Simulation (17 F1 tracks, minimum-time optimal control solver)

| Method | Iterations | Opt Time (s) | Gen Time (s) | Total Time (s) | Lap Time (s) |
|--------|-----------|-------------|-------------|---------------|-------------|
| Centerline | 521.6 ± 149.0 | 149.5 ± 47.5 | — | 149.5 ± 47.5 | 85.70 ± 11.20 |
| Min-Curvature | 483.1 ± 112.1 | 136.2 ± 38.7 | 121.4 ± 50.3 | 257.6 ± 63.4 | 85.26 ± 11.17 |
| F1-NN (proposed) | 434.5 ± 103.1* | 123.4 ± 40.5 | 0.63 ± 0.1 | 124.0 ± 40.5*† | 85.22 ± 11.21* |
| F1-GT (oracle) | 400.1 ± 67.6 | 112.3 ± 26.2 | — | 112.3 ± 26.2 | 85.14 ± 11.13 |

*p < 0.05 vs. centerline; †p < 0.05 vs. min-curvature

Key takeaways:
- F1-NN reduces solver iterations by **17%** vs. centerline.
- F1-NN reduces total runtime by **~17%** vs. centerline (149.5 → 124.0 s).
- F1-NN matches oracle (F1-GT) on lap time within noise.
- Lap times are nearly identical across all methods — the benefit is convergence speed and consistency, not a large lap time delta.
- The F1-GT upper bound shows that even perfect initialization only saves ~25 more iterations over F1-NN, suggesting the network captures most of the benefit.

### Hardware Validation (RoboRacer 1:10 scale vehicle, single track)

| Initialization | Lap Time (s) | Avg Speed (m/s) | Lateral Error (m) |
|---|---|---|---|
| Centerline | 7.090 ± 0.061 | 3.504 ± 1.010 | 0.165 ± 0.155 |
| F1-NN | 6.640 ± 0.069* | 3.672 ± 1.014* | 0.109 ± 0.075* |

*p < 0.01

Surprisingly large improvements on hardware despite domain shift: **6.3% lap time reduction**, **34% reduction in lateral tracking error** (0.165 → 0.109 m). The authors attribute this to better-conditioned trajectories from the improved initialization producing paths the controller can track more faithfully.

---

## 4. Relevance to Our System

Our system uses L-BFGS-B in `planning/racing_line.py` to optimize gate pass-through offsets (`d_i` per gate) for minimum path length + smoothness. The objective landscape is non-convex, and we initialize from `x0 = np.zeros(n * 2)` (all gate centers). We have explicitly documented that the S-turn region (gate-3) is a persistent problem where the optimizer converges to a suboptimal local minimum — high tracking error (0.402–0.422 m) persisting across multiple iterations despite smoothness weight tuning.

The connection to Shehadeh et al. is direct:

1. **Same fundamental problem:** Cold-start initialization from a neutral point (gate centers = centerline analog) may sit in a poor basin of attraction in the L-BFGS-B objective landscape, especially for multi-gate S-curve sections where the optimal path requires coordinated offsets across gates 2–4 simultaneously.

2. **Same solution class:** Providing an informed initial offset vector `x0` that reflects "expert-like" behavior (aggressive corner-cutting through the S-turn) could land the optimizer in a better basin and yield a qualitatively different (better) local minimum.

3. **Hardware validation directly applicable:** The 34% lateral error reduction on the RoboRacer suggests that better-initialized trajectories produce smoother paths that physical controllers track more accurately — our MPC tracker would benefit similarly.

4. **Our constraint:** We lack F1 telemetry, but the *principle* is transferable: any domain knowledge about the optimal racing line through an S-turn (e.g., the late-apex strategy: cut inside on first bend, position outside for second bend) can serve as the initialization prior. This doesn't require a trained neural network — a hand-crafted heuristic initialization per gate type is sufficient for our small gate count (8 gates).

---

## 5. Actionable Takeaways

1. **Replace zero-initialization with a heuristic warm start for the S-turn.** For gate-3 (the first S-turn bend), initialize the lateral offset to a late-apex value (e.g., +0.4 of gate half-width toward the inside of the turn) rather than 0. For gate-4 (second bend), initialize to the opposite sign. This directly mimics the "expert prior" concept without requiring a neural network.

2. **Try multi-start L-BFGS-B for the S-turn section.** Run the optimizer from 5–10 random initializations (sampled uniformly in `[-max_lateral_offset, max_lateral_offset]^n`) and take the best result. The paper's insight is that initialization quality matters more than iteration count — a few extra runs from diverse starts is cheap relative to the benefit.

3. **Use the hybrid loss insight for objective design.** Our current objective combines path length + curvature penalty. The paper's hybrid cosine + Euclidean loss suggests that adding a "directional alignment" term (penalizing abrupt heading changes rather than just curvature) may steer L-BFGS-B toward smoother local minima structurally.

4. **Pre-compute a "racing line prior" from geometry.** For each gate pair (i, i+1), compute the sign of the turn (left/right) and initialize gate i's lateral offset to cut the inside corner. This is the minimum-curvature heuristic implemented cheaply without running a full optimizer. Use this as `x0` instead of zeros.

5. **Increase `maxiter` beyond 100 for offline optimization.** The paper's solver ran 400–520 iterations to convergence. Our current cap of 100 iterations in `minimize(..., options={"maxiter": 100})` may be premature termination. Since racing line optimization runs once offline, raising to 500 is free. A better initialization + more iterations together should improve solution quality.

6. **Log `result.fun` and `result.nit` after optimization.** Currently we discard the optimizer's convergence metadata. Logging these values per run would let us confirm whether the S-turn local minimum is a convergence issue (nit hitting maxiter) or a landscape issue (converged but to a bad basin) — these require different fixes.

7. **Consider gate-local subproblem decomposition.** For the S-turn (gates 2–5), run a focused sub-optimization with lookahead_gates = 4, initialized with the geometric prior, before folding the result into the global optimization. This "divide and conquer" reduces the dimensionality of the local problem and matches the paper's finding that better structural proximity (not just Euclidean distance) matters most.

---

## 6. Limitations and Caveats

**Paper limitations:**

- **Lap time gains are small in simulation** (85.70 → 85.22 s, ~0.6%). The primary win is convergence speed, not solution quality. For our use case, we care far more about solution quality (tracking error) than compute time, so the direct simulation results are less motivating than the hardware results.
- **Geometric prediction accuracy is nearly identical to the centerline** (RMSE difference: 0.05 m). This suggests the network's value is in capturing trajectory *structure* (which gates to cut, when to apex), not in precise lateral position. Our hand-crafted heuristic can capture the same structure.
- **Constant friction assumption** in the vehicle dynamics model. Real-world tracks violate this, as do drone aerodynamics (drag, downwash). The paper's vehicle model may not generalize directly.
- **Long straights degrade prediction quality** (noted explicitly for COTA circuit). In our gate track, this is less relevant since gates are densely spaced.
- **Domain shift is large for drones.** F1 cars are subject to different dynamics (tire friction, Ackermann steering) than quadrotors (thrust-limited, 6-DOF). The neural network weights are not directly transferable. The *principle* transfers but not the model.
- **3 Hz GPS telemetry is low-resolution** and requires averaging 15 laps to increase density. This limits fine-grained precision of the ground truth raceline.

**Caveats for our application:**

- Our gate count is only 8, making neural network overkill. A lookup-table or geometric rule per gate type suffices.
- The paper optimizes over a continuous track parameterized by arc-length, while our problem is discrete (8 gate offsets). The optimization landscape is lower-dimensional.
- The paper's minimum-time optimal control solver is likely IPOPT or CasADi-based with full dynamics, whereas our L-BFGS-B uses a simplified proxy cost. Better initialization may have smaller effect in lower-dimensional, simpler landscapes.

---

## 7. Key Parameters and Constants

From the paper's methodology and results:

| Parameter | Value | Context |
|---|---|---|
| Spatial resolution | 2.0 m | Resampling interval along arc-length for standardization |
| Laps averaged per driver | 15 | For GPS density augmentation |
| Training circuits | 14 of 17 | Cross-validation split |
| Test circuits | 3 of 17 | Per fold |
| Cross-val folds | 6 | Ensures each track tested without seeing it in training |
| Loss weights | 0.5 / 0.5 | Cosine alignment vs. Euclidean distance |
| TCN residual blocks | 2 | Dilated 1D convolutional blocks in encoder |
| Optimizer iterations (centerline) | 521.6 ± 149.0 | Reference baseline |
| Optimizer iterations (F1-NN) | 434.5 ± 103.1 | 17% reduction |
| Generation time (F1-NN) | 0.63 ± 0.1 s | Neural network inference cost |
| Hardware lap time improvement | 6.3% | Centerline → F1-NN on RoboRacer |
| Hardware lateral error reduction | 34% | 0.165 → 0.109 m on RoboRacer |
| Significance threshold | p < 0.05 (sim), p < 0.01 (hw) | Statistical testing |
| F1 GPS sample rate | ~3 Hz | Raw telemetry frequency |
| Track count in dataset | 17 F1 circuits | From FastF1 + TUM Racetrack Database |

For our system, the most actionable analogues are:
- **Initialization offset magnitude:** The network predicts offsets on the order of meters (RMSE ~3 m relative to F1 ground truth on 15–20 m wide tracks). Scaled to our 1.2 m wide gates, this suggests initializing at ~0.3–0.5 of half-width is a reasonable geometric prior, consistent with our `max_lateral_offset = 0.6`.
- **Iteration budget:** The baseline solver runs ~520 iterations to convergence. Our `maxiter=100` cap is likely causing premature termination in complex sections. Recommend raising to at least 300.
