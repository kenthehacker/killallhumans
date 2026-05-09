# Sequence Modeling for Time-Optimal Quadrotor Trajectory Optimization with Sampling-based Robustness Analysis

- **URL**: https://arxiv.org/abs/2506.13915
- **Authors**: Katherine Mao, Hongzhan Yu, Ruipeng Zhang, Igor Spasojevic, M. Ani Hsieh, Sicun Gao, Vijay Kumar
- **Year**: 2025
- **Venue**: CoRL 2025 (Conference on Robot Learning); submitted June 16, 2025 via arXiv cs.RO

---

## Key Contribution

This paper addresses a core bottleneck in time-optimal quadrotor trajectory planning: classical optimizers like TOPPQuad are accurate but computationally expensive (on the order of 10+ seconds per trajectory), making them infeasible for onboard real-time use. The authors propose a learning-based surrogate that trains an LSTM encoder-decoder to imitate TOPPQuad's per-segment speed output from geometric path inputs alone. The central insight is that time-optimal speed profiles have strong sequential dependencies across path segments — the velocity at segment k constrains what is achievable at segment k+1 — which makes recurrent sequence modeling a natural fit.

Beyond raw speedup, the paper introduces a formal framework for analyzing the robustness of the learned model's outputs: rather than just asking "does the trajectory track well on average," the authors link the model's local sensitivity properties to Backward Reachable Tubes (BRTs), a concept from formal verification and safe control theory. This allows them to certify, for any given input path, whether the predicted speed profile keeps the drone safely inside a tube around the nominal trajectory under bounded input perturbations. This robustness analysis is the secondary contribution that elevates the work above a plain learning-to-optimize paper.

---

## Technical Approach

### Problem Formulation

The paper decomposes trajectory generation into two stages — a classical approach in the spirit of TOPPQuad (Mao, IROS 2024):
1. **Geometric path**: A collision-free path through gates is fixed (e.g., from a minimum-snap planner or waypoint list).
2. **Speed profile / timing**: Given the fixed path geometry, find the time allocation across segments that minimizes total flight time while respecting the drone's dynamic constraints (thrust, angular velocity, jerk).

The learning model addresses stage 2 only: given the geometry of path segments as input, predict per-segment time durations or, equivalently, per-segment average speeds.

### LSTM Encoder-Decoder Architecture

The authors compare four architectures:
- **LSTM Encoder-Decoder** (their primary contribution)
- Full Transformer (encoder-decoder with attention)
- Encoder-only Transformer
- Per-step MLP (baseline)

The LSTM encoder-decoder treats the sequence of geometric path segments (waypoints at 100 equally-spaced points per trajectory) as a sequence-to-sequence problem. The encoder reads the full geometric path and produces a latent context; the decoder autoregressively outputs per-segment speed/time predictions. This captures the causal dependency structure: segment k's time allocation depends on the preceding k-1 segments (the entrance velocity is set by the prior segment).

Key architecture results from Table 1:
| Model | Compute Time | Failure Rate |
|---|---|---|
| LSTM Encoder-Decoder | 0.078 s | 2.0% |
| Transformer (full) | 1.012 s | 76.0% |
| Encoder-only Transformer | 1.042 s | 72.0% |
| Per-step MLP | 0.010 s | 0.0% (poor tracking) |

The full Transformer fails catastrophically (76% failure rate) despite being a superficially more powerful architecture. The authors attribute this to the inductive bias of the LSTM being better matched to the causal left-to-right structure of speed planning: segment k truly depends on segments 1..k, not on future segments, so the bidirectional attention of a Transformer introduces noise. The per-step MLP is fastest but does not account for inter-segment continuity, resulting in velocity discontinuities that degrade tracking quality even if failures are rare.

### Training Data

- Simulation dataset: 10,000 trajectories, each with 100 equally spaced points, solved by TOPPQuad as ground truth labels.
- Hardware experiments: 9,000 trajectories.
- Data augmentation: random perturbations are added to input paths during training, forcing the model to learn smoother, more conservative speed predictions that remain feasible under small path deviations.

### Sampling-Based Robustness Analysis

The robustness framework is the methodologically novel part of the paper. The authors define the **Backward Reachable Tube (BRT)**:

```
BRT(x, Δt) = { x₀ | ∃ τ ≤ Δt, s.t. x₀(τ) = x under control U }
```

This is the set of initial states from which the drone can reach target state x within time Δt using admissible controls U. If the drone's predicted trajectory lies within the BRT of the nominal optimal trajectory at each time step, then the trajectory is certifiably robust to disturbances bounded by the tube radius.

The **sampling-based** approach works as follows:
1. Sample ε-bounded random perturbations to the input geometric path.
2. Run the LSTM to get predicted speed profiles for each perturbed input.
3. Check whether the resulting trajectory stays inside the BRT of the nominal trajectory.
4. Report the fraction of perturbed inputs whose trajectories remain in-tube.

This converts the intractable question "is this model robust?" into a Monte Carlo estimate "what fraction of paths within ε of the nominal path produce in-tube trajectories?" Results from Table 2 show that LSTM-0.1 (trained with perturbation scale 0.1) achieves 92.9% in-BRT probability at ε=0.1 perturbation scale, with a maximum deviation of 0.127 m.

The data augmentation during training (random path perturbations) directly improves this metric — models trained without augmentation have worse in-BRT probability even if their nominal-path tracking error is similar.

---

## Results

### Computational Speedup

The LSTM model computes in **0.078 seconds** vs. TOPPQuad's **10.656 seconds**, a **~137x speedup**. This transforms the trajectory timing step from an offline precomputation into something feasible for onboard re-planning.

### Time-Optimality (TD Ratio)

The Time Difference (TD) ratio — how much longer the LSTM trajectory takes vs. TOPPQuad optimal — is:
- Training set: -0.70% (LSTM is marginally faster, likely due to slight infeasibility that reduces conservatism)
- Test set: -0.40%

These near-zero TD ratios indicate the LSTM closely matches the optimal timing without significant degradation. The slightly negative values suggest the learned model occasionally produces trajectories marginally faster than TOPPQuad (at the cost of the 2% failure rate).

### Tracking Error

- Simulation maximum position deviation: **0.074 m** for LSTM
- Hardware (CrazyFlie 2.0): **0.355 m** maximum position deviation for LSTM vs. **0.347 m** for TOPPQuad

The hardware tracking errors are slightly worse for the LSTM (0.355 m vs. 0.347 m), which is expected — the optimization-based method has exact constraint satisfaction whereas the learned model approximates. However, the difference is small (0.008 m) and both are within practical safety margins for the test platform.

### Robustness (Table 2)

| Model | ε | In-BRT Probability | Max Deviation |
|---|---|---|---|
| LSTM-0.0 (no augmentation) | 0.1 | lower | higher |
| LSTM-0.1 (augmented) | 0.1 | 92.9% | 0.127 m |

Data augmentation meaningfully improves robustness without sacrificing nominal performance.

### Generalization

The LSTM generalizes to path lengths exceeding those in the training set (longer trajectories than seen during training), confirming that the model learns structural patterns about curvature-to-speed relationships rather than memorizing trajectory shapes.

---

## Relevance to Our System

Our system (`planning/trajectory_optimizer.py`) already implements a TOPP-RA-style retiming step (`_topp_retime`) and references TOPPQuad (Mao, IROS 2024) explicitly. The segment time allocation problem this paper solves is exactly what our `_optimize_time_allocation` and `_topp_retime` methods handle analytically. The question is: does the learned surrogate add value in our context?

**Where the learned model would help:**
1. **Re-planning latency**: Our analytical TOPP retimer runs offline before the race. If we needed to re-time a trajectory mid-race (e.g., after gate detection updates gate positions), our analytical method is too slow. The 0.078 s LSTM would allow ~12 Hz re-timing updates.
2. **Per-segment margin identification**: The robustness analysis's in-BRT probability per segment could identify which segments are at the dynamic limit (low margin, high failure rate under perturbation) vs. which have slack. This is directly useful for deciding where to back off speed to reduce crash risk.
3. **Warm-starting the optimizer**: Even if we keep the analytical TOPP for final solutions, the LSTM's output could serve as a warm start that reduces optimizer iterations.

**Where the learned model is less necessary:**
1. **Pre-race optimization**: We run the TOPP retimer offline and bake it into the race trajectory. Computational cost is not a bottleneck here. Our analytical approach gives exact constraint satisfaction with no failure rate.
2. **Known gate layouts**: Unlike a fully autonomous mission where the path is unknown until runtime, we know the race course before the race (or can precompute during qualification). This eliminates the real-time planning pressure.

**The robustness analysis is the more transferable idea**: The BRT-based framework can tell us, for each segment in our pre-computed trajectory, whether a given tracking error tolerance is achievable under the drone's dynamics. Our current approach uses a uniform speed backing-off heuristic in `_inflate_sharp_turns`; a BRT-per-segment analysis would let us apply speed reduction only where the reachable tube is genuinely tight.

---

## Actionable Takeaways

1. **Use the BRT per-segment margin concept to refine `_inflate_sharp_turns`**: Instead of heuristically inflating sharp-turn segment times, compute whether the gate's nominal crossing velocity is inside the BRT margin for that segment geometry. Segments with tight BRT margin should be slowed; segments with large BRT margin can be kept aggressive.

2. **Evaluate LSTM surrogate for mid-race re-planning**: If gate detection updates a gate's estimated position during the race, we need to re-time the remaining trajectory. The 0.078 s inference time of an LSTM would enable this at ~12 Hz. Worth implementing as a fallback re-planner.

3. **Replicate the sampling-based robustness analysis on our trajectory**: Generate ε-bounded perturbations to our pre-computed path, simulate the resulting deviations, and measure which gates have high probability of tube violation. This directly identifies our fragile segments — currently we are guessing via per-gate tracking error in the benchmark.

4. **Use data augmentation strategy for any learned components we add**: If we train any learned components (e.g., a learned speed predictor or MPC terminal cost), the paper's result that augmenting with path perturbations improves robustness (from <92% to 92.9% in-BRT) is a clear training recipe to adopt.

5. **The LSTM beats Transformers for this causal problem**: If we ever benchmark sequence models for trajectory planning, this paper confirms that the causal inductive bias of LSTMs is superior to bidirectional attention for left-to-right speed scheduling. Skip Transformer-based approaches for this specific sub-problem.

6. **Hardware gap is small (0.008 m) but real**: The learned model's hardware tracking error is slightly worse than the optimizer's. For competition, where every centimeter of gate margin matters, the analytical TOPP with exact constraint satisfaction is preferred for the pre-race trajectory. Reserve the LSTM for online adaptation only.

7. **Checkout the reference implementation**: `https://github.com/maokat12/lbTOPPQuad` provides working LSTM code trained on TOPPQuad outputs. This is directly usable to generate the BRT-margin analysis for our gate segments without needing to implement from scratch.

---

## Limitations & Caveats

1. **2% failure rate**: Even the best LSTM model has a 2% trajectory failure rate on test data. For competition use in a single-attempt race, 2% is a meaningful crash probability. Analytical TOPP has 0% constraint violation by construction (if the optimizer converges).

2. **Small test platform**: Hardware experiments use the CrazyFlie 2.0, a micro-quadrotor limited to 2 m/s and 10 m/s² acceleration. Competition racing drones fly at 15-25 m/s with much higher accelerations. Whether the LSTM's learned patterns transfer to higher-performance regimes is untested.

3. **BRT analysis is sampling-based, not certified**: The 92.9% in-BRT probability is a Monte Carlo estimate, not a formal guarantee. With probability 7.1%, a perturbed path produces an out-of-tube trajectory. For hard safety requirements, this is insufficient.

4. **No obstacle avoidance integration**: The method fixes the geometric path first and only learns the timing. Dynamic re-routing through obstacles is outside scope.

5. **Generalization to unseen path lengths is qualitative**: The paper shows Figure 3 qualitatively, but quantitative generalization error for out-of-distribution lengths is not reported.

6. **The LSTM is an imitator, not an improver**: The model targets TOPPQuad's output, which is already time-optimal under that planner's constraints. If TOPPQuad is sub-optimal (e.g., due to its own simplifications), the LSTM learns the same sub-optimality.

7. **Per-segment boundary conditions**: The encoder-decoder must maintain velocity continuity at segment boundaries through the decoder's autoregressive structure. This is implicit rather than hard-constrained, which may cause the 2% failure cases.

---

## Key Parameters / Constants

| Parameter | Value | Source |
|---|---|---|
| Training set size | 10,000 trajectories | Paper |
| Hardware training size | 9,000 trajectories | Paper |
| Waypoints per trajectory | 100 equally spaced | Paper |
| LSTM compute time | 0.078 s | Table 1 |
| TOPPQuad compute time | 10.656 s | Table 1 |
| Speedup factor | ~137x | Computed |
| LSTM failure rate | 2.0% | Table 1 |
| TD ratio (training) | -0.70% | Paper |
| TD ratio (test) | -0.40% | Paper |
| Max sim position deviation | 0.074 m | Table 1 |
| Max HW position deviation (LSTM) | 0.355 m | Table 4 |
| Max HW position deviation (TOPPQuad) | 0.347 m | Table 4 |
| In-BRT probability (LSTM-0.1, ε=0.1) | 92.9% | Table 2 |
| Max deviation in BRT test | 0.127 m | Table 2 |
| HW max speed | 2 m/s | Paper |
| HW max acceleration | 10 m/s² | Paper |
| HW max angular velocity | 10 rad/s | Paper |
| Perturbation scale ε | 0.1 | Table 2 |
| Code repository | https://github.com/maokat12/lbTOPPQuad | Paper |
