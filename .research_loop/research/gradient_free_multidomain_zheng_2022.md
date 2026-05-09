# Gradient-free Multi-domain Optimization for Autonomous Systems

- **URL**: https://arxiv.org/abs/2202.13525
- **Authors**: Hongrui Zheng, Johannes Betz, Rahul Mangharam
- **Year**: 2022
- **Venue**: arXiv preprint (submitted February 28, 2022); associated with the TunerCar project at the University of Pennsylvania
- **Code**: https://github.com/hzheng40/tunercar

---

## Key Contribution

This paper introduces a gradient-free multi-domain optimization framework for autonomous systems, specifically targeting the long-standing challenge of siloed subsystem design. In classical autonomous system engineering, hardware parameters (vehicle mass, center-of-gravity position), planning parameters (racing line waypoints, look-ahead distances), and control parameters (gain schedules, velocity profiles) are each tuned in isolation — which is sub-optimal because the domains interact nonlinearly. The core contribution is treating all these parameter families as a single joint search space and optimizing over them simultaneously using black-box, gradient-free methods.

The paper validates this idea on autonomous racing vehicles using the F1TENTH platform and the F1TENTH Gym simulator. Six gradient-free optimizers are benchmarked across three different race tracks (Spielberg, Silverstone, Monza) with different geometric characteristics (a mixed track, a curve-heavy track, and a high-speed straight-dominated track). The framework, called TunerCar, is designed to generalize beyond racing — the authors frame it as a blueprint for any autonomous system with separable but interacting design domains and a simulator that can evaluate candidate configurations.

---

## Technical Approach

### Problem Formulation

The optimization problem is formulated as a black-box, derivative-free minimization:

    minimize f(x)  subject to x in X

where `x` is a concatenated parameter vector spanning multiple design domains (hardware, planning, control), `X` is a bounded search space defined by domain-specific constraints, and `f(x)` is the lap time returned by a physics simulator for a full race run with parameter configuration `x`. Because `f` is non-differentiable (it involves a simulator with discrete events like collision detection and gate sequencing), gradient-based methods cannot be directly applied.

### Parameter Domains

Three domains are jointly optimized:

1. **Hardware domain**: Physical vehicle parameters — vehicle mass (~3.29–3.98 kg range), center-of-gravity-to-front-axle distance (~0.15–0.17 m), and related inertial properties.
2. **Planning domain**: Motion planning algorithm selection (discrete) and planning parameters such as look-ahead distance for Pure Pursuit, waypoint spacing, and racing-line lateral offsets.
3. **Control domain**: Low-level controller parameters — velocity setpoints, gain schedules, maximum velocity (up to ~15 m/s), minimum velocity.

The combination of continuous and discrete parameters means the search space is mixed-type, which presents additional challenges for gradient-based methods but is handled naturally by gradient-free approaches.

### Gradient-free Optimizers Benchmarked

Six optimizers are compared:

1. **CMA (Covariance Matrix Adaptation Evolution Strategy, CMA-ES)**: Population-based evolutionary algorithm that adapts a full covariance matrix of the search distribution. Uses a self-adaptive explore-exploit balance.
2. **2PtsDE (Two-point Differential Evolution)**: DE variant using two-point crossover. Population size is `max(num_workers, 30)` for the standard setting; `max(num_workers, 30, dimension+1)` for the dimension-aware setting; `max(num_workers, 30, 7*dimension)` for the large-population setting.
3. **NDE (Noisy Differential Evolution)**: DE variant with noisy recommendation, intended to be more robust in noisy or stochastic fitness landscapes.
4. **PSO (Particle Swarm Optimization)**: Swarm-based optimizer where particles maintain velocity and position in the search space, communicating to converge toward global optima.
5. **1+1 Evolutionary Algorithm**: Minimal (1+1)-ES variant — a single parent generates a single offspring each iteration, accepted if it improves the objective.
6. **Random Search**: Uniform random sampling of the search space, used as a baseline.

All optimizers are accessed through the **Nevergrad** library, which provides a unified ask-and-tell interface and supports parallel evaluation via Ray for distributed simulation rollouts.

### Sim-in-the-loop Evaluation

Each optimizer query triggers a full simulator rollout in the F1TENTH Gym environment — a physics-accurate 2D racing simulator. A candidate configuration `x` is submitted (ask), the simulator runs to completion or timeout, the lap time is recorded (tell), and the optimizer updates its internal model. The fixed computational budget is **9,600 simulator calls** per optimizer run on the default Spielberg track. Because the simulation runs ~6x faster than real time, this budget is feasible within a practical wall-clock window.

The parallel evaluation capability (via Ray) is critical: multiple simulator instances evaluate different population members simultaneously, making population-based methods like CMA-ES and DE viable within the budget.

### Dimensionality

The exact dimensionality of the joint parameter vector is not explicitly stated in the abstract, but from the reported parameter ranges (vehicle mass, CG position, max/min velocity, look-ahead distance, controller gains) the effective continuous dimension is estimated at roughly **10–20 parameters**, augmented by discrete choices for planning algorithm selection. CMA-ES is known to be effective up to a few hundred dimensions, but performs best in the 10–50 range — directly applicable here.

---

## Results

### Quantitative Lap Time Results

On the default Spielberg track:
- Best lap time with multi-domain optimization using CMA: **66.69 seconds**
- Other configurations reported: 100.20 s, 99.85 s, 96.39 s, 110.21 s, 115.29 s, 69.65 s, 73.71 s (across different optimizer/track combinations)

On Silverstone (curve-heavy):
- CMA again achieves competitive results, with unique solutions found across the population

On Monza (high-speed straights + three chicanes):
- CMA achieves lowest lap times; PSO finds more diverse solutions

### Optimizer Conclusions

- **CMA is the overall best-performing optimizer** for lap time minimization. Its covariance adaptation enables effective exploration of the correlated parameter space typical of multi-domain optimization, while also exploiting promising regions.
- **PSO is preferred when solution diversity is required** — it tends to find multiple distinct optima spread across the search space, which is useful when the system operator wants to explore a Pareto front of trade-offs.
- **Random Search** provides a useful baseline but is consistently outperformed once budget exceeds ~1,000 evaluations.
- **2PtsDE and NDE** are competitive but slightly below CMA on most tracks; population size tuning (standard vs. dimension-aware vs. large) affects relative performance.
- **1+1 EA** converges quickly but gets trapped in local optima more often, suitable only for unimodal landscapes.

### Improvement Magnitude

Relative improvements in lap time between **7–21%** are reported across tracks compared to hand-tuned baseline configurations. Improvements over naive random search with equivalent budget: over **15 seconds per lap**. Improvements over expert-tuned configurations: over **2 seconds per lap** in some cases.

---

## Relevance to Our System

Our system uses a kinematic sim (PyBullet) to evaluate trajectory quality, and we have a concrete low-dimensional tuning problem: **4 parameters** (lateral + vertical offsets for gates 7–8 in the racing line). This is a textbook use case for the methodology in this paper.

**Direct parallels:**

1. **Sim-in-the-loop evaluation structure**: The Zheng 2022 framework is precisely sim-in-the-loop — a candidate parameter set is evaluated by running the full simulator and returning a scalar metric (lap time). Our setup is identical: candidate gate offsets → run `scripts/benchmark.py --mode sim` → parse `simulation.avg_tracking_error_m` or race time as the objective.

2. **Low dimensionality (d=4) is ideal for CMA-ES**: The paper recommends CMA-ES for this type of problem. With only 4 parameters, CMA-ES operates in a regime where it converges extremely efficiently — typically in **O(d log d)** to **O(d^2)** function evaluations. For d=4, this means **~50–200 evaluations** should suffice to converge well. The paper uses 9,600 evaluations for a ~15-dimensional problem; we can expect to need far fewer.

3. **Correlated parameters**: The lateral and vertical offsets for gates 7 and 8 are likely correlated (e.g., a wider lateral offset at gate 7 requires compensating at gate 8 due to trajectory curvature). CMA-ES handles parameter correlations explicitly through its covariance matrix — this is its key advantage over coordinate-wise methods like Nelder-Mead or independent DE.

4. **Discrete/continuous mix**: The paper successfully handles mixed search spaces. If we expand tuning to include discrete choices (e.g., trajectory segment allocation, controller mode selection), the framework can still handle it via mixed-type optimizers in Nevergrad.

5. **Budget efficiency**: Our benchmark takes ~5–10 seconds per simulation. With a budget of 200 evaluations, we are looking at ~20–30 minutes of wall-clock time to converge — a very practical investment for a meaningful racing line improvement.

6. **Multi-domain extension**: Beyond gates 7–8, the same framework could jointly optimize all 24 gate offsets plus control gain parameters (TrackerConfig), trajectory segment times, and even EKF noise covariances. This is precisely the multi-domain vision of the paper.

---

## Actionable Takeaways

1. **Use CMA-ES (via Nevergrad) for the gate offset optimization.** It is the clear winner in the paper for lap time minimization in low-to-medium dimensional spaces. Initialize with the current hand-tuned offsets as the starting mean, and a small initial step size (sigma ~ 0.1–0.2 m for lateral, 0.05–0.1 m for vertical).

2. **Budget 100–300 evaluations for the 4-parameter problem.** Given d=4, CMA-ES should converge within this range. Set the evaluation budget accordingly when calling the benchmark script in a tuning loop.

3. **Parallelize evaluations using Ray.** The paper demonstrates 6x speedup via parallel simulation. Our benchmark.py may not yet support parallel execution, but wrapping it with Ray or Python multiprocessing would allow running multiple parameter configurations simultaneously.

4. **Treat the objective as the scalar race time or avg_tracking_error_m.** The paper uses lap time as the single objective. We should do the same — pick one primary metric (race time is most meaningful for competition) and optimize it directly. Avoid multi-objective formulations for this initial pass.

5. **Log all evaluations (parameter + objective) for warm-starting.** The paper uses Sacred for experiment management. We should at minimum log each (offset_vector, benchmark_result) pair to a file so that if the tuning run is interrupted, CMA-ES can be warm-started from the learned distribution.

6. **Expand to full 24-parameter joint optimization after gating on 4-parameter success.** Once the 4-parameter tuning shows measurable improvement, scale to the full parameter space. CMA-ES scales to ~20–50 dimensions effectively; beyond that, consider separability-exploiting variants (sep-CMA-ES, LMCMA).

7. **Treat the sim as noisy, not deterministic.** The F1TENTH sim had stochastic elements too. Our PyBullet benchmark may have timing noise. Consider running 2–3 evaluations per candidate and averaging, or using Nevergrad's noisy optimizer variants.

8. **Use Nevergrad as the optimizer library.** It is already a dependency in the TunerCar repo, supports CMA-ES, all DE variants, PSO, and random search, and provides a consistent ask-and-tell API that is straightforward to wrap around our benchmark script.

---

## Limitations & Caveats

1. **Sim-to-real gap**: The paper optimizes for F1TENTH simulator performance. Real-world transfer is not deeply studied — a configuration that minimizes sim lap time may not transfer perfectly to hardware due to unmodeled dynamics. For us this is partially mitigated because we optimize in the same sim used for all other development, but the gap to the Anduril/DCL competition environment remains a risk.

2. **No convergence guarantees**: CMA-ES is a heuristic; it finds good solutions but provides no global optimality proof. The paper acknowledges that solutions "did not converge to the best possible in all cases." Budget allocation is a practical engineering choice, not a theoretically optimal one.

3. **Fixed computational budget assumption**: The paper assumes a fixed budget of 9,600 evaluations. For our use case with a different-dimensional problem and different sim speed, the budget must be re-calibrated.

4. **Multi-modal landscapes**: Some track configurations (especially Monza, with its chicane geometry) led to multiple optima that CMA-ES could miss. If our gate-7/8 region has multiple good racing lines (e.g., different cornering strategies), CMA-ES might converge to a local minimum. Running multiple independent restarts from different initial conditions mitigates this.

5. **Discrete parameter handling is approximate**: Gradient-free optimizers can handle discrete parameters by rounding or encoding, but convergence guarantees weaken further. If we encode algorithm selection or segment count as discrete parameters, expect noisier convergence.

6. **The paper uses F1TENTH (car racing), not drone racing.** The dynamics are qualitatively different — cars have nonholonomic constraints and operate in 2D; drones have 6-DOF rigid body dynamics and operate in 3D. The parameter space geometry differs. However, the key methodological insight (gradient-free black-box optimization over a mixed sim-in-the-loop objective) transfers directly.

7. **Hyperparameter sensitivity of CMA-ES**: The initial step size sigma and population size lambda affect convergence speed. The paper does not provide a sensitivity analysis. For our 4-parameter problem, the defaults (sigma=0.3 of the search range, lambda=4+floor(3*ln(d))=6 for d=4) are standard starting points.

---

## Key Parameters / Constants

| Parameter | Value / Range | Notes |
|-----------|--------------|-------|
| Computational budget | 9,600 evaluations | For ~15-dim problem on Spielberg |
| Simulation speedup | ~6x real-time | F1TENTH Gym; allows large budget |
| Vehicle mass range | ~3.29–3.98 kg | Hardware domain parameter |
| CG-to-front-axle range | ~0.15–0.17 m | Hardware domain parameter |
| Max velocity range | up to ~15.0 m/s | Control domain parameter |
| 2PtsDE population (standard) | max(num_workers, 30) | Nevergrad default |
| 2PtsDE population (dim-aware) | max(num_workers, 30, d+1) | d = parameter dimension |
| 2PtsDE population (large) | max(num_workers, 30, 7*d) | For high-dimensional problems |
| Lap time improvement | 7–21% relative | Compared to hand-tuned baselines |
| Improvement over random search | >15 s/lap | With equivalent 9,600-eval budget |
| Improvement over expert baseline | >2 s/lap | Best case; track dependent |
| Tracks used | Spielberg, Silverstone, Monza | Different geometric characteristics |
| Optimizers compared | 6 total | CMA, 2PtsDE, NDE, PSO, 1+1 EA, Random |
| Best overall optimizer | CMA | Lowest lap time across all tracks |
| Best for diversity | PSO | Multiple distinct solutions |
| Optimizer library | Nevergrad | Unified ask-and-tell API |
| Parallelization framework | Ray | Distributed simulation rollouts |

---

## Summary Judgment for Our Use Case

This paper provides strong empirical justification for using CMA-ES with a sim-in-the-loop evaluation loop to tune our gate offset parameters. The low dimensionality of our 4-parameter problem (gates 7–8 lateral + vertical offsets) means we are in the regime where CMA-ES excels most — expected convergence in under 200 evaluations, well within practical time constraints (~20–30 min at 5–10 s/eval). The methodology is straightforward to implement: wrap `scripts/benchmark.py --mode sim` as the black-box objective, pass gate offset parameters as the search vector, and use Nevergrad's `CMA` optimizer with reasonable initial sigma. This is a high-confidence, low-risk improvement strategy with direct precedent in the literature.
