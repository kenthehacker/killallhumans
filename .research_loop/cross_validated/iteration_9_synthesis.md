# Iteration 9 — Research Synthesis: Speed Recovery via Drag Compensation + Reduced Inflation

## Papers Analyzed (3 new, 21 prior)

1. **"Why Change Your Controller When You Can Change Your Planner: Drag-Aware Trajectory Generation"** (Zhang et al., L4DC 2024) — drag-aware planning reduces tracking error by 83% vs min-snap baseline
2. **"Real-time Planning of Minimum-time Trajectories for Agile UAV Flight"** (Teissing et al., RA-L 2024) — norm constraint on acceleration (vs per-axis) gives >20% faster trajectories; includes linear drag model in planner
3. **"ℒ₁Quad: ℒ₁ Adaptive Augmentation of Geometric Control"** (Wu et al., IEEE TCST 2025) — adaptive layer compensates unmodeled drag, enables 5x smaller tracking error

## Consensus Across Papers

All three papers agree on the core issue: **unmodeled aerodynamic drag creates a systematic mismatch between the planned trajectory and what the controller can achieve**. The solutions differ in approach:

- **Zhang 2024**: Modify the planner (generate trajectories the controller can follow despite drag)
- **Teissing 2024**: Modify constraint formulation (norm vs per-axis) AND include drag in planner
- **Wu 2025**: Modify the controller (add adaptive layer to compensate drag online)
- **Faessler 2018** (already analyzed): Modify the controller feedforward (analytical drag-aware flatness)

For our system, where the drag model is **known and linear** (coefficient = 0.5), the Faessler approach is the simplest and most direct: add `drag * vel` to the controller's feedforward term. This is what all papers converge on as the foundational fix.

## Contradictions

- Zhang 2024 argues planner modification is better than controller modification. But their experiments use unknown/complex drag. For known linear drag, direct controller compensation is cleaner.
- Wu 2025's ℒ₁ framework provides formal guarantees but requires ~30 lines of adaptation logic, a state predictor, and careful filter tuning. Over-engineering for a known 0.5 coefficient.

## Ranked Actionable Takeaways

### Priority 1: Direct drag compensation in controller (THIS ITERATION)
- Add `drag_coefficient * current_velocity` to `accel_des` in `GeometricTracker.track()`
- This cancels the sim's `- drag * vel` term, making the controller model-accurate
- Enables feedforward weight increase from 0.4 → 0.8-1.0
- Expected impact: avg error -15-25%, enabling faster trajectories

### Priority 2: Reduce inflation parameters (THIS ITERATION)
- With drag-compensated feedforward active, the controller handles turns much better
- Reduce centripetal inflation coefficient from 0.25 to 0.15
- Reduce angle-based inflation from 0.35 to 0.25
- Expected impact: race time -1.5-2.5s

### Priority 3: Norm-based acceleration constraint (FUTURE)
- Replace per-axis penalty in L-BFGS with norm penalty (Teissing 2024)
- Expected impact: additional 10-20% trajectory time reduction
- Higher risk — changes optimizer convergence behavior

## Proposed Implementation Direction

**Two-step approach within this iteration:**
1. Add drag compensation to controller → sweep feedforward weight → find optimal
2. Reduce inflation parameters → sweep reduction amounts → find Pareto-optimal speed/accuracy

This is backed by the strongest evidence (all 3 papers + Faessler 2018) and is the lowest-risk path to significant speed improvement.
