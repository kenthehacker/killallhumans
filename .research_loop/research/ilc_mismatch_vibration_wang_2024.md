# ILC with Mismatch Compensation for Residual Vibration Suppression

- **URL**: https://arxiv.org/abs/2411.07862
- **Year**: 2024

## Key Contribution

This paper presents an Adaptive Mismatch-Compensated Iterative Learning Controller (AMCILC) combined with an optimal input shaper for Delta robots driven by permanent magnet synchronous motors (PMSMs). The three main contributions are: (1) an integrated rigid-flexible coupling dynamic model that accounts for PMSM dynamics and joint stiffness, (2) a dual-objective input shaper that minimizes both maximum and average residual vibration across the robot's workspace, and (3) an adaptive ILC framework that uses fuzzy logic structures (FLS) to approximate model mismatch while enforcing velocity constraints via barrier Lyapunov functions (BLFs). The key novelty is that the controller does not require an accurate plant model -- it learns the mismatch between the nominal model and the real system iteratively, converging to high-precision tracking despite significant modeling errors, parameter uncertainties, and unmodeled PMSM dynamics.

## Technical Approach

### Model Decomposition via Singular Perturbation

The rigid-flexible coupling dynamics of the Delta robot are decomposed using the Singular Perturbation Method (SPM) into a slow subsystem (rigid-body motion) and a fast subsystem (structural vibration). This separation enables independent controller design: ILC handles trajectory tracking in the slow subsystem, while input shaping suppresses vibration in the fast subsystem.

### Mismatch Compensation via Fuzzy Logic

The core mismatch compensation mechanism works as follows. The true plant dynamics contain uncertain terms: mass matrix errors (Delta_M), Coriolis/centrifugal errors (Delta_C), gravity errors (Delta_G), and unknown damping (B_r). These are lumped into a single mismatch function. A fuzzy logic structure (FLS) approximates this mismatch as theta^T * phi(x), where theta are fuzzy weights and phi(x) are basis functions. The universal approximation theorem guarantees that for any continuous mismatch function on a compact set, there exists a FLS such that the approximation error epsilon satisfies |epsilon_i| <= epsilon_bar_i (a known bound).

Across iterations, the fuzzy weight estimates theta_hat_k and error bound estimates epsilon_hat_k are updated adaptively. The update laws are gradient-based, driven by the tracking error signal. Each iteration refines the mismatch approximation, so the controller progressively learns the gap between the nominal model and the real system without requiring explicit system identification.

### AMCILC Control Law

The control input at iteration k is:

u_k = C_bar(theta_k, theta_dot_k) * theta_dot_k + G_bar(theta_k) - theta_hat_k^T * phi(x_k) + M_bar(theta_k) * (theta_ddot_r - sigma * e_dot - k * eta_k) - epsilon_hat_k

where sigma and k are positive design constants, eta_k = e_dot + sigma * e is a combined error metric, and the terms with bars denote nominal model components. The first two terms provide model-based feedforward, the FLS term compensates mismatch, the M_bar term provides feedback, and epsilon_hat corrects for FLS approximation residuals.

### Velocity Constraint Enforcement

Angular velocity constraints |theta_dot_i| <= theta_dot_max are enforced using a Barrier Lyapunov Function: V_b,i = (v_c^2 / pi) * tan(pi * eta_i^2 / (2 * v_c^2)). This function grows to infinity as the error metric eta approaches the constraint boundary v_c, effectively creating a soft wall that prevents constraint violation. A Barrier Composite Energy Function (BCEF) combines the BLF with tracking error metrics to provide a unified convergence and constraint satisfaction guarantee.

### Convergence Guarantees

The convergence analysis proceeds through the BCEF framework. The main guarantees are: (1) tracking errors converge to zero along the iteration axis (k -> infinity), (2) velocity constraints are never violated during the learning process, (3) fuzzy weight estimates converge to values that accurately compensate the true mismatch, and (4) error reduction is monotonic across iterations. The proof relies on showing that the BCEF decreases at each iteration under the proposed update laws. Convergence depends on the mismatch being bounded and continuous (so the FLS can approximate it) and on the adaptive gains satisfying stability conditions from the Lyapunov analysis.

### Input Shaper Design

A three-impulse input shaper with amplitudes A_1, A_2, A_3 and timings t_1=0, t_2=T, t_3=2T is optimized. The lag T = k_t * T_d / 2 is parameterized by k_t in [0, 1]. Because the Delta robot's natural frequency varies across its workspace (16-24 Hz for the first mode), the optimization minimizes a weighted sum J = 0.5 * J_max + 0.5 * J_avg, where J_max is the worst-case residual vibration and J_avg is the average across configurations. Grid search (resolution 0.01) yields optimal parameters: f_n = 16.4 Hz, k_t = 0.83, with design damping ratio zeta = 0.075.

### Learning Rate / Gain Structure

The paper does not use explicit decaying learning rates. Instead, the adaptive update laws for theta_hat and epsilon_hat implicitly regulate the learning speed through the Lyapunov-derived gain conditions. The feedback gain k and proportional constant sigma are fixed design parameters. No explicit low-pass filter is applied to the learning signal; monotonic convergence is enforced structurally through the energy function design rather than through signal filtering.

## Results

The validation uses high-fidelity multi-domain Simscape simulation incorporating PMSM electrical dynamics, rigid-flexible mechanical coupling, gear backlash, and damping. Two case studies compare performance against the mathematical model and the high-fidelity simulation. The AMCILC demonstrates: (1) convergence of tracking errors to near-zero over iterations, (2) effective vibration suppression through the input shaper reducing residual vibration across the workspace, (3) velocity constraints respected throughout all iterations, and (4) robustness to the significant gap between the simplified mathematical model used for controller design and the high-fidelity simulation plant. The paper is currently simulation-only; no hardware experiments are reported.

## Relevance to Our System

The relevance to our drone racing system is moderate but contains transferable ideas:

1. **Iterative learning across laps**: If the drone repeatedly flies similar trajectories (e.g., practice laps on a known track), an ILC-style approach could reduce tracking error lap-over-lap. Our current controller uses fixed PD/geometric gains; an iterative refinement layer could learn feedforward corrections that compensate for unmodeled aerodynamic effects, motor lag, or EKF bias.

2. **Mismatch compensation concept**: Our system has model mismatch between the assumed quadrotor dynamics and reality (drag, motor response, propeller interactions). The FLS-based mismatch learning could inspire an online or iteration-based compensation term added to our controller output. However, for a single race pass this requires adaptation within a single trial rather than across iterations.

3. **Barrier Lyapunov for constraint enforcement**: The BLF approach to enforcing velocity/rate limits could be adapted for our attitude rate constraints or speed limits near gates. This is cleaner than hard saturation and provides formal guarantees.

4. **Input shaping for vibration**: Less directly relevant since quadrotors do not have the same structural flexibility as Delta robots, but the concept of shaping reference trajectories to avoid exciting undesirable dynamics (e.g., oscillatory modes in our PD controller) is broadly applicable.

5. **Singular perturbation decomposition**: Separating slow (position) and fast (attitude) dynamics for independent control design parallels standard quadrotor control architectures. The paper reinforces this design pattern with formal justification.

## Actionable Takeaways

1. **Lap-to-lap feedforward learning**: Implement a simple ILC correction term u_ff[k+1](t) = u_ff[k](t) + L * e[k](t) where L is a learning gain and e[k] is the tracking error from lap k. This could be applied to thrust and moment commands. Start with L = 0.1-0.3 and verify convergence empirically.

2. **Mismatch estimation**: Rather than full FLS, a simpler approach for our system would be to fit a polynomial or lookup table of feedforward corrections as a function of velocity and acceleration, updated after each lap. This captures drag and motor lag without needing fuzzy logic machinery.

3. **Barrier functions for gate speed limits**: If we need to enforce maximum speed through gates (for safety or detection reliability), a BLF-based soft constraint in the trajectory optimizer could replace hard clipping, giving smoother trajectories.

4. **No decaying learning rate needed**: The paper shows that fixed adaptive gains with proper Lyapunov design achieve monotonic convergence. For our simpler ILC application, a fixed learning gain with a low-pass Q-filter on the learned signal should suffice for convergence.

## Limitations & Caveats

- **Simulation-only validation**: No hardware experiments are reported. The authors acknowledge this and state hardware implementation is future work. The gap between Simscape simulation and real hardware is unknown.
- **Iteration-domain only**: ILC requires repeated execution of the same trajectory. For a single race attempt, this framework does not directly apply. It would need adaptation to learn across practice runs before the race.
- **Fixed trajectory assumption**: Standard ILC assumes the reference trajectory is identical across iterations. In racing, slight variations in gate positions or wind conditions would require robust ILC extensions.
- **FLS complexity**: The fuzzy logic structure adds design complexity (choosing membership functions, number of rules). Simpler approximators (polynomials, neural networks) might achieve similar results with less tuning.
- **Convergence rate not quantified**: While monotonic convergence is proven, the paper does not provide explicit convergence rate bounds (e.g., geometric rate with specific contraction factor). Practical convergence speed depends on gain tuning.
- **Delta robot specifics**: The rigid-flexible coupling model and PMSM dynamics are specific to parallel robots. Direct transfer to quadrotors requires re-deriving the mismatch structure for rotorcraft dynamics.

## Key Parameters / Constants

| Parameter | Value / Range | Description |
|-----------|--------------|-------------|
| f_n (optimal) | 16.4 Hz | Input shaper design frequency |
| k_t (optimal) | 0.83 | Input shaper lag coefficient |
| zeta_design | 0.075 | Design damping ratio |
| Natural freq range | 16-24 Hz | First-mode frequency across workspace |
| k_t range | [0, 1] | Lag coefficient search space |
| Grid resolution | 0.01 | Optimization search granularity |
| w_1, w_2 | 0.5, 0.5 | Optimization objective weights |
| sigma | > 0 (tunable) | PD-like proportional constant in error metric |
| k | > 0 (tunable) | Feedback gain in control law |
| v_c | < theta_dot_max | BLF constraint parameter |
| epsilon_bar | bounded | FLS approximation error bound |
| Number of impulses | 3 | Input shaper structure |
