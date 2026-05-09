# Iterative Learning Control of Fast, Nonlinear, Oscillatory Dynamics

- **URL**: https://arxiv.org/abs/2405.20045
- **Authors**: John W. Brooks, Christine M. Greve
- **Year**: 2024
- **Venue**: arXiv (cs.LG)

## Key Contribution
This paper addresses ILC for systems where the controlled dynamics are much faster than the controller's sampling rate. By combining ILC with Time-Lagged Phase Portraits (TLPP) and Gaussian Process Regression (GPR), the authors develop a framework that can iteratively tune control parameters for fast oscillatory systems even when the controller operates at a much lower frequency than the dynamics. The key novelty is using TLPP as a diagnostic tool to characterize the system's dynamical state and GPR to map control parameters to TLPP features, enabling optimization without a closed-form model.

## Technical Approach
The methodology has three components:
1. **Time-Lagged Phase Portraits (TLPP)**: Reconstruct the system's attractor from time series data using delay embedding (Takens' theorem). This provides a geometric characterization of the dynamics that is robust to noise and sampling artifacts.
2. **Gaussian Process Regression (GPR)**: Learn the mapping from control parameters to TLPP features (e.g., orbit diameter, period, stability margin). This enables gradient-based optimization of control parameters.
3. **Iterative Parameter Tuning**: Use the GPR model to predict which parameter adjustments will move the TLPP features toward the desired trajectory, then validate on the real system and update the GP model.

The ILC update is conceptual rather than the standard u_{j+1} = Q(u_j + Le_j) form. Instead, it's parameter-space ILC: θ_{j+1} = θ_j + Δθ where Δθ is computed via GPR-guided optimization.

## Results
Demonstrated on the Lorenz system, achieving stable tracking of desired oscillatory trajectories. The method identifies continuous and bounded regions of achievable dynamical trajectories and shows robustness to incomplete information. Convergence typically achieved in 10-20 parameter-space iterations.

## Relevance to Our System
Moderate relevance. While our drone racing system doesn't have "fast oscillatory dynamics" in the classical sense, the parameter-space ILC concept is relevant to our challenge of tuning ILC section parameters (alpha, max_correction, vel_scale, cutoff_hz) across multiple track sections. Rather than manually tuning these 4-6 parameters per section × 4 sections = 16-24 parameters, a GPR-guided approach could systematically explore the parameter space.

However, this is a longer-term architectural improvement, not immediately actionable for this iteration.

## Actionable Takeaways
1. The idea of parameter-space ILC (tuning controller parameters iteratively based on performance feedback) could be applied to our section boundary parameters — systematically optimize alpha, max_correction, vel_scale per section.
2. GPR model of (section_params → per_gate_error) could identify optimal configurations without exhaustive grid search.
3. For this iteration, the concept validates our per-section tuning approach — treating each track section as having different optimal ILC parameters.

## Limitations & Caveats
- Demonstrated only on Lorenz system — a low-dimensional attractor, not a high-dimensional robotics system.
- The GPR approach requires many evaluations (10-20 iterations × multiple parameter samples), which is expensive in our sim.
- Not directly applicable to our current ILC correction computation, which operates in trajectory-space not parameter-space.

## Key Parameters / Constants
- Typical convergence: 10-20 parameter-space iterations
- GPR kernel: Squared Exponential with automatic relevance determination
- Delay embedding dimension: 2-3 for Lorenz system
