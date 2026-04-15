# ILC with Mismatch Compensation for Residual Vibration Suppression in Delta Robots

- **URL**: https://arxiv.org/abs/2411.07862
- **Authors**: Mingkun Wu, Alisa Rupenyan, Burkhard Corves
- **Year**: 2024
- **Venue**: IEEE (under review), arXiv preprint

## Key Contribution

This paper proposes an adaptive mismatch-compensated iterative learning controller (MC-ILC) that explicitly addresses the gap between the theoretical plant model used during ILC design and the actual system behavior. The central idea is that standard ILC assumes the plant model P is accurate, but in practice, the actual plant P̃ differs from P due to unmodeled dynamics, parameter uncertainty, and configuration-dependent variations. The paper introduces a fuzzy logic structure to approximate and compensate for this mismatch term ΔP = P̃ - P during ILC iterations.

The mismatch compensation is combined with input shaping (for vibration suppression in the Delta robot context) and an adaptive ILC framework. The key theoretical contribution is showing that ILC convergence can be maintained even with significant model mismatch if the mismatch is estimated and compensated iteratively.

## Technical Approach

The standard P-type ILC update law is:
```
u_{k+1} = Q(u_k + L * e_k)
```
where Q is a robustness filter, L is the learning operator, and e_k is the tracking error at iteration k.

With mismatch compensation:
```
u_{k+1} = Q(u_k + L * e_k + ΔL * Δe_k)
```
where ΔL compensates for the model mismatch and Δe_k is the error residual attributed to model uncertainty.

The fuzzy logic estimator approximates ΔP online from the iteration-to-iteration error evolution. When the error pattern doesn't match the expected reduction from the nominal model, the fuzzy system attributes the residual to model mismatch and adjusts the learning law accordingly.

The convergence condition for ILC with mismatch is:
```
||I - P̃L|| < 1 (with Q-filter)
```
For standard ILC without mismatch compensation: ||I - PL|| < 1 is required, which may not hold when P ≠ P̃.

## Results

Results demonstrated through high-fidelity Simscape simulations of a Delta robot. The MC-ILC shows improved convergence speed and lower residual error compared to standard ILC when model mismatch is present. Specific numerical improvements were not available from the abstract, but the paper demonstrates that mismatch compensation prevents the error floor that standard ILC hits when the plant model is inaccurate.

## Relevance to Our System

This paper is directly relevant to our ILC-controller coupling problem:

1. **Our ILC has explicit model mismatch**: The ILC inner sim uses (kp=6, kd=4, ff=0.4, no tilt limit) while the benchmark uses (kp=7, kd=5.5, ff=0.50, max_tilt=0.85 rad). This is a classic P ≠ P̃ scenario.

2. **Why naive gain sync failed**: When we synchronized the ILC gains to match the benchmark (+15.1% regression), we changed P to match P̃, but the converged corrections u_k were optimized for the old P. The corrections + new plant dynamics produced worse tracking than the mismatched equilibrium.

3. **The paper suggests two paths**:
   a. Compensate for the mismatch adaptively (complex, requires modifying the ILC update law)
   b. Reduce the mismatch at the source (simpler — align the plant models)

4. **For our system, reducing the mismatch by increasing max_tilt_rad is the simplest approach**: The ILC inner sim has no tilt limit (unconstrained plant). Increasing max_tilt_rad in the benchmark makes the benchmark plant closer to the ILC's unconstrained model. This reduces ΔP without requiring any ILC modification.

## Actionable Takeaways

1. **Reduce model mismatch at the source** by increasing max_tilt_rad rather than trying to compensate for it
2. **If ILC still needs modification**, consider adding a mismatch compensation term to the P-type update law
3. **The convergence condition ||I - P̃L|| < 1 explains** why some ILC parameter changes cause basin switching: when ΔP is large enough, convergence breaks
4. **Q-filter cutoff choice matters more with mismatch**: the filter must be conservative enough to maintain ||Q(I - P̃L)|| < 1

## Limitations & Caveats

- Delta robot dynamics are fundamentally different from quadrotor dynamics
- The fuzzy logic mismatch estimator adds complexity
- Results are simulation-only; no hardware validation
- The approach assumes the mismatch is slowly varying; our mismatch is structural (tilt constraint)

## Key Parameters / Constants

- Standard ILC convergence: ||I - PL|| < 1
- With mismatch: ||I - P̃L|| < 1 (harder to satisfy)
- Q-filter essential for robustness to mismatch
- Mismatch compensation reduces error floor
