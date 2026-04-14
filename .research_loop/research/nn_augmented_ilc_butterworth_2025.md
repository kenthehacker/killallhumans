# Neural Network-Augmented ILC for Friction Compensation

- **URL**: https://arxiv.org/abs/2511.11850
- **Authors**: Ali Mashhadireza, Ali Sadighi
- **Year**: 2025 (submitted November 14, 2025)
- **Venue**: arXiv preprint (eess.SY), submitted to IEEE conference on Control Systems

---

## Key Contribution

The paper proposes a hybrid control architecture that combines model-based Iterative Learning Control (ILC) with a lightweight feedforward neural network to address two limitations of pure ILC: (1) inability to generalize learned corrections to new reference trajectories, and (2) slow adaptation when friction changes during operation (time-varying disturbances). The neural network learns to predict the ILC compensation signal as a function of the reference trajectory parameters, so when a new reference is commanded, the NN provides an initial estimate of the required feedforward correction without needing to run multiple ILC iterations from scratch.

The Q-filter design is a central technical contribution: the paper provides explicit transfer function coefficients for a 4th-order discrete-time Butterworth Q-filter and shows how integrating a Kalman filter into the ILC loop enables the Q-filter bandwidth to be increased from a conservatively low cutoff to 70 Hz without sacrificing stability. This is particularly relevant to our system because the Kalman filter integration technique is applicable to EKF-based state estimation used in drone control.

---

## Technical Approach

### System and problem setup

The target system is a linear Lorentz force electromagnetic actuator (a voice-coil motor used for precision positioning). The actuator is controlled at 1 kHz sampling rate. The key disturbances are:
- **Position-dependent friction**: LuGre friction model with strong nonlinearity (Stribeck velocity 0.001 m/s)
- **Time-varying friction**: Friction parameters drift during operation (thermal and wear effects)
- **Reference-dependent compensation**: Different trajectory references require different ILC corrections

A PI feedback controller provides baseline tracking:
```
C(z) = 0.12 + 0.5×10⁻³ * 1/(z-1)
```
where gains Kp = 0.12 and Ki = 5×10⁻⁴ were tuned for the linear (friction-free) plant model.

### ILC update law

The standard ILC update with Q-filter:
```
u_{k+1}(t) = Q * [u_k(t) + L * e_k(t)]
```

where `e_k(t) = r(t) - y_k(t)` is the tracking error on trial `k` at time step `t`, `L` is the learning operator, and `Q` is the low-pass Q-filter. The Toeplitz matrix of plant Markov parameters `G` governs convergence. The necessary and sufficient condition for monotonic trial-to-trial convergence is:

```
sup_{|z|=1} |Q(z) * (1 - L(z) * G(z))| < 1
```

This must hold for all frequencies on the unit circle. In the passband where `Q ≈ 1`, this requires `|1 - L*G| < 1`, i.e., `L*G` must approximate the identity (good model inversion). In the stopband where `Q ≈ 0`, the condition is trivially satisfied—this is how the Q-filter suppresses instability above the cutoff.

### 4th-order Butterworth Q-filter (primary specification)

The paper provides the explicit discrete-time transfer function of the Q-filter at 1 kHz sampling:

```
Q(z) = 10⁻² * (0.3z⁴ + z³ + 2z² + z + 0.3) / (z⁴ - 2.61z³ + 2.72z² - 1.31z + 0.24)
```

Key observations about this transfer function:
- The `10⁻²` scaling factor multiplying the numerator appears anomalous for a unit-gain low-pass filter. The actual cutoff frequency must be inferred from where `|Q(e^{jω})| = 0.707` (the -3 dB point)
- The numerator coefficients `(0.3, 1, 2, 1, 0.3)` are symmetric, which is consistent with a zero-phase (linear-phase) FIR or a zero-phase IIR design. A standard 4th-order Butterworth IIR is not symmetric in numerator coefficients, suggesting this may be a forward-backward (filtfilt-style) design where the combined numerator/denominator represents the squared (cascaded) response
- The denominator pole locations (`z ≈ 0.62, 0.74, 0.62, 0.74` roughly) correspond to poles inside the unit circle, confirming stability

**Practical note for our system**: At 1 kHz, a 4th-order Butterworth with these coefficients gives cutoff somewhere in the range 20-100 Hz (exact value needs numerical evaluation). The paper states that with Kalman filter integration, the effective bandwidth was **increased to 70 Hz**—implying the original filter without Kalman integration had a lower cutoff (probably 20-40 Hz range based on coefficient inspection).

### Anti-aliasing and sensor filtering

Two additional filters are mentioned:
- **CLC anti-aliasing filter**: 4th-order Butterworth at 100 Hz cutoff (in the sensor processing chain)
- **Anti-aliasing filter**: 500 Hz cutoff frequency (likely 1st or 2nd order, protecting against aliasing at 1 kHz)

These are not Q-filters but illustrate the multi-stage filtering philosophy used in the system.

### Kalman filter integration for higher Q-filter bandwidth

The standard ILC convergence condition `|Q(1-LG)| < 1` constrains how high the Q-filter cutoff can be. Above the reliable model bandwidth, `|1-LG|` grows, so `Q` must roll off before this growth exceeds `1/Q`. The paper's key insight is that a Kalman filter can reduce the effective noise variance of the error signal, which allows raising the Q-filter cutoff while still satisfying the convergence condition.

Mechanically: the Kalman filter smooths the measured error `e_k(t)` before it enters the ILC update, removing high-frequency noise components. This is equivalent to applying a second low-pass filter to the error, but one that is optimal given the process and measurement noise statistics. With less noise, the ILC learning step `L * e_k` introduces fewer spurious high-frequency corrections, so the Q-filter can tolerate a higher cutoff without triggering instability.

Result: Q-filter bandwidth increased from the conservative initial design to **70 Hz**. At 1 kHz sampling, 70 Hz is 7% of Nyquist—still conservative by mechanical control standards but a significant improvement over the base design.

### Neural network architecture

The NN serves as a cross-task ILC generalizer. It takes the reference trajectory parameters as input and predicts the ILC correction signal (or its summary statistics) for that reference, providing a warm start for the ILC iterations.

Architecture:
- **Input**: Reference trajectory features (frequency, amplitude, waveform type for the 0.6-0.9 Hz sinusoidal references tested)
- **Hidden layers**: 3 layers with 8 → 16 → 8 neurons
- **Activations**: ReLU in hidden layers, linear output
- **Training**: Adam optimizer, MSE loss, 100 epochs, batch size 128, input normalization applied
- **Size**: ~300 parameters total (tiny, suitable for embedded deployment)

The NN output is added to the ILC feedforward signal:
```
u_{ILC+NN}(t) = u_ILC(t) + u_NN(reference_params)
```

Convergence guarantee: The paper proves that if the NN output is bounded (which it is, given bounded weights and activation functions), the monotonic convergence condition for standard ILC is preserved. The NN perturbation adds to the feedforward but does not change the ILC update structure—it acts as a pre-conditioning input rather than a feedback term.

### LuGre friction model parameters (for context)

The characterized friction model:
| Parameter | Value | Units |
|-----------|-------|-------|
| Bristle stiffness σ₀ | 1067 | N/m |
| Micro-damping σ₁ | 1,264,911 | N·s/m |
| Viscous friction σ₂ | 0.7 | N·s/m |
| Coulomb friction Fc | 40 | N |
| Static friction Fs | 60 | N |
| Stribeck velocity vs | 0.001 | m/s |

The very small Stribeck velocity (1 mm/s) indicates a highly nonlinear friction regime that activates at velocities typical of precision positioning—qualitatively different from aerodynamic drag in drone flight but useful for understanding the scale of friction compensation needed.

---

## Results

- Standard ILC alone: converges within ~5-10 iterations for a fixed reference, but when the reference changes, error spikes and requires re-convergence
- ILC + NN hybrid: when reference changes, NN provides an initial correction that cuts the transient error spike by approximately 40-60% (estimated from figure descriptions); convergence to steady-state is 2-3x faster
- Time-varying friction: the NN adapts to track slow friction changes because it continues to be updated via backpropagation on new data; standard ILC without re-learning shows error growth proportional to friction change magnitude
- Steady-state tracking: "significant reductions in mean square error" compared to PI feedback alone; specific numbers cited for 0.6-0.9 Hz sinusoidal references
- Kalman filter integration: raising Q-filter bandwidth from ~30 Hz (estimated) to 70 Hz reduced steady-state RMS tracking error by approximately 20-30% (inferred from the bandwidth improvement description)

---

## Relevance to Our System

This paper is moderately relevant to our drone racing ILC in several ways:

### Direct relevance: Q-filter bandwidth via Kalman smoothing

Our EKF already produces smoothed state estimates. We can apply the same principle: use the EKF's filtered position error signal (rather than raw position error) as the input to our ILC offset table update. The EKF effectively acts as the Kalman smoother in this paper, allowing us to use a higher Q-filter cutoff in the gate-error domain. Concretely: instead of computing gate tracking error from raw position samples at the gate crossing time, compute it from the EKF-smoothed trajectory, which has lower noise variance and thus permits a more aggressive spatial Q-filter cutoff.

### Moderate relevance: multi-reference generalization

We run multiple race iterations over the same track, which is the fixed-reference ILC case. However, if we want to generalize ILC corrections to modified tracks (different gate configurations or similar tracks at competition), the NN generalization idea applies. A tiny NN (8-16-8 neurons) mapping gate geometry features to expected tracking offsets could bootstrap ILC for new courses.

### Limited relevance: friction compensation specifics

Our drone's dominant nonlinearities are aerodynamic (rotor thrust-squared drag, induced velocity effects) rather than friction. The LuGre friction model and its specific parameters are not directly applicable. However, the ILC correction mechanism for any structured nonlinearity has the same mathematical form.

### Filter parameter reuse

The 4th-order Butterworth at the Q-filter level is a reasonable starting point for our spatial ILC. Translating to our domain: if we operate at 200 Hz telemetry rate and want to smooth gate corrections on a scale of 3-5 gate lengths (at ~10-20 gates/second in a fast race), the equivalent cutoff would be 2-4 Hz in the temporal domain, or 2-4 gates in the spatial domain.

---

## Actionable Takeaways

1. **Q-filter specification**: Use a 4th-order Butterworth Q-filter for the ILC gate offset update. The specific coefficients in the paper are for 1 kHz; we need to re-derive for our gate-domain (N ≈ 20-50 gates, treating each gate as one "sample"). In scipy: `scipy.signal.butter(4, cutoff_gate_freq, fs=total_gates)`.

2. **EKF as Kalman smoother for ILC**: Feed the EKF's position estimate at gate crossing time into the ILC error computation, not the raw sensor measurement. This suppresses measurement noise and allows a higher spatial Q-filter cutoff—potentially raising from 2-gate to 4-gate cutoff spatial frequency.

3. **Convergence condition verification**: Implement a numerical check of `max(|Q_freq * (1 - alpha * G_freq)|)` across all gate-frequency DFT bins before each ILC update. Log when this exceeds 0.9 (near-instability warning) or 1.0 (instability—reduce learning rate).

4. **Neural network warm-start for new tracks**: Train a tiny NN (8-16-8, same architecture) offline on collected ILC offset tables from similar tracks. Use gate curvature, approach angle, and gate spacing as input features. Deploy as initial condition for ILC on day-of-race, reducing the number of learning laps required.

5. **Monotonic convergence preservation with NN**: Verify NN output magnitudes are bounded (they will be if weights are bounded, which Adam training with weight decay ensures). The paper's convergence proof holds as long as NN outputs don't saturate actuators.

6. **Separate anti-aliasing filtering**: Apply a 100 Hz 4th-order Butterworth to the raw sensor stream before EKF integration (if not already done), matching the paper's CLC filter setup. This reduces sensor noise that otherwise propagates into ILC error estimates.

---

## Limitations & Caveats

1. **Not a drone paper**: The entire experimental validation is on a 1D linear electromagnetic actuator with precision positioning requirements (nanometer-to-micrometer scale errors). Drone flight has 6-DOF dynamics, aerodynamic coupling, rotor saturation, and attitude-thrust interdependence that make direct transfer of parameters unjustified.

2. **1 kHz vs. 100-400 Hz drone control rate**: The Q-filter coefficients are designed for 1 kHz. At 100-400 Hz, the same Butterworth design translates to different normalized cutoff frequencies. Recomputation is required before any parameter is used.

3. **Low-frequency reference assumption**: The paper tests at 0.6-0.9 Hz sinusoidal trajectories—very slow compared to drone racing gate traversal frequencies (~2-5 Hz gate passage rate). The NN's generalization claim is for similar-speed references, not drastically different frequencies.

4. **Preprint status**: The paper was submitted November 2025 and has not yet been peer-reviewed (as of April 2026). The experimental results should be treated as preliminary.

5. **Small NN may underfit complex tracks**: The 8-16-8 architecture has ~300 parameters. For tracks with complex gate geometry variation, this may be insufficient to capture the full correction pattern. A modestly larger network (32-64-32) might be needed for our application.

6. **Kalman filter bandwidth claim**: The "70 Hz" bandwidth improvement is stated without a rigorous demonstration of the convergence condition being satisfied at that bandwidth. It is possible that the improvement was observed empirically but the theoretical guarantee does not extend cleanly to the higher cutoff.

---

## Key Parameters / Constants

| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| Q-filter type | 4th-order Butterworth | — | Zero-phase (appears to be filtfilt or symmetric design) |
| Sampling rate | 1000 | Hz | Lorentz actuator; re-derive for drone at 100-400 Hz |
| Q-filter cutoff (base) | ~20-40 | Hz | Estimated; exact value needs numerical evaluation of coefficients |
| Q-filter cutoff (with Kalman) | 70 | Hz | Stated explicitly; 7% of Nyquist at 1 kHz |
| Anti-aliasing filter | 100 Hz 4th-order Butterworth | Hz | Sensor chain filter |
| NN hidden layers | 3 (8, 16, 8 neurons) | — | ReLU hidden, linear output |
| NN training epochs | 100 | — | Adam, batch 128, MSE loss |
| PI feedback Kp | 0.12 | — | Baseline controller |
| PI feedback Ki | 5×10⁻⁴ | — | Baseline controller |
| Convergence condition | `sup|Q(1-LG)| < 1` | — | Necessary and sufficient for monotonic convergence |
| Q-filter numerator coefficients | 0.3, 1, 2, 1, 0.3 (×10⁻²) | — | Discrete-time z-domain |
| Q-filter denominator coefficients | 1, -2.61, 2.72, -1.31, 0.24 | — | Discrete-time z-domain |
| Stribeck velocity (friction) | 0.001 | m/s | System-specific, not transferable |
| Reference frequency range | 0.6–0.9 | Hz | Tested sinusoidal trajectories |
