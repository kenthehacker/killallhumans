# Accurate Tracking of Aggressive Quadrotor Trajectories Using INDI and Differential Flatness
- **URL**: https://arxiv.org/abs/1809.04048
- **Authors**: Ezra Tal, Sertac Karaman
- **Year**: 2018/2021
- **Venue**: IEEE Transactions on Control Systems Technology, Vol. 29, No. 3, pp. 1203–1218 (2021); first presented at CDC 2018

---

## Key Contribution

This paper introduces a unified control architecture that tracks position, yaw, and all their derivatives up to fourth order: position, velocity, acceleration, jerk, and snap, along with yaw rate and yaw acceleration. This is qualitatively different from typical trajectory tracking controllers that only close feedback loops on position and velocity and treat acceleration as a feedforward at best. By exploiting the differential flatness of quadrotor dynamics to derive analytic feedforward terms for angular rate and angular acceleration (corresponding to jerk and snap in the flat output space), the controller effectively anticipates future attitude requirements rather than reacting to them.

A second key contribution is the application of Incremental Nonlinear Dynamic Inversion (INDI) for both linear and angular acceleration tracking. Unlike standard NDI (feedback linearization), which requires an accurate model of all forces and moments, INDI uses onboard accelerometer and gyroscope measurements to implicitly estimate the current disturbance state and apply corrections incrementally. This renders the controller robust to aerodynamic drag without needing to model or identify it.

The combination of these two ideas — differential flatness-derived feedforward up to snap, plus INDI-based disturbance rejection — enables a 1 kg quadrotor to track complex 3D trajectories with an RMS position error of 6.6 cm at speeds up to 12.9 m/s and accelerations up to 2.1g (approximately 20.6 m/s²).

---

## Technical Approach

### Quadrotor Model and Differential Flatness

The quadrotor is modeled as a 6-DOF rigid body with state (position, velocity, Euler attitude, body-frame angular rate) and control inputs (total thrust and three-axis body torque), which relate to the four rotor speeds through a thrust/torque coefficient matrix.

Differential flatness means the entire state and all inputs can be expressed analytically as functions of four flat outputs and their derivatives. For a quadrotor, the flat outputs are the three inertial position components and the yaw angle. Given a reference trajectory defined as a smooth function of time:

- Position: `p_ref(t)` (class C4 in position, C2 in yaw)
- Velocity: first derivative
- Acceleration: second derivative
- Jerk: third derivative
- Snap: fourth derivative

The differential flatness mapping proceeds as follows. The reference thrust direction (the body z-axis) is determined by the commanded acceleration plus gravity, normalized. The reference attitude is then fully specified by this thrust direction combined with the yaw angle. By differentiating the jerk equation — which is affine in angular rate — one can invert it to recover the reference angular rate `omega_ref` as a closed-form function of the reference trajectory derivatives. Differentiating once more and inverting gives the reference angular acceleration `alpha_ref` as a function of the reference snap.

These `omega_ref` and `alpha_ref` terms are injected as feedforward signals into the attitude rate and attitude acceleration controllers, respectively. The core insight is that if the trajectory is smooth and the derivatives are available (e.g., from a min-snap polynomial planner), the controller already knows exactly what angular motion will be needed before the positional error has time to build up.

### Full Control Architecture (Six Nested Layers)

The paper presents a structured hierarchy of six control components, summarized in their Table I:

1. **PD Position and Velocity Control** — computes a commanded acceleration from position and velocity errors against the reference.

2. **INDI Linear Acceleration Control** — uses accelerometer measurements (gravity-corrected, low-pass filtered, in the inertial frame) to compute the incremental update to the specific thrust vector. The key equation takes the current estimated external force `f_ext` (computed from the difference between measured acceleration and the thrust contribution) as approximately constant, and applies a delta thrust to drive the measured acceleration toward the commanded value. This eliminates integrators and is robust to unmodeled drag.

3. **Jerk and Snap Tracking via Differential Flatness** — computes `omega_ref` and `alpha_ref` from the trajectory jerk and snap using the analytic flatness mapping. These are passed forward as feedforward reference signals.

4. **NDI Attitude and Attitude Rate Control** — uses feedback linearization on the angular kinematics (not the full torque dynamics) to define a linear double-integrator equivalent system. The commanded attitude rate includes the feedforward `omega_ref` from differential flatness; the commanded angular acceleration combines feedback terms with the feedforward `alpha_ref` from snap.

5. **INDI Angular Acceleration Control** — given the commanded angular acceleration `alpha_cmd`, uses gyroscope-derived angular acceleration measurements and the current control moment (computed from measured motor speeds) to compute the incremental torque update needed to drive the angular acceleration to `alpha_cmd`. This is analogous to the linear INDI loop but for rotational dynamics. The external moment `tau_ext` is estimated from the difference between measured angular acceleration and the torque contribution, and treated as constant over each time step.

6. **Inversion-Based Motor Speed Control with Integral Action** — inverts the nonlinear thrust/torque coefficient matrix (using Newton's method on the discretized finite-difference equation) to obtain commanded rotor speeds. Optical encoders on each motor hub provide high-rate, accurate rotor speed feedback at 1000 Hz, enabling tight closed-loop motor control. Integral action compensates for battery voltage sag. This closed-loop motor speed control is what makes snap tracking physically achievable: snap corresponds to angular acceleration, which corresponds to applied body torque, which requires accurate real-time torque delivery.

### INDI Core Equations

For linear acceleration: the incremental control law computes the new commanded specific thrust vector as the current specific thrust plus a correction term driven by the error between commanded and measured acceleration. Formally, `f_cmd = f_current + K_indi * (a_cmd - a_measured)`, where both `f_current` and `a_measured` pass through identical low-pass filters to ensure phase-matched comparison.

For angular acceleration: similarly, `tau_cmd = tau_current + J * (alpha_cmd - alpha_measured)`, where `tau_current` is derived from measured motor speeds. The identical filter is applied to both IMU signals and motor speed measurements.

### Trajectory Definition

The experiments use a figure-8-like 3D reference trajectory parametrized by a frequency parameter `omega_xy`. The lemniscate shape has sinusoidal components in x, y, and z, with separate yaw rate `omega_psi`. The trajectory derivatives are computed analytically (all four orders) and fed into the flatness mapping at runtime.

---

## Results

**Primary result**: 6.6 cm RMS position tracking error at 12.9 m/s top speed, 2.1g peak acceleration, in an 18 m × 7 m × 3 m flight volume using a 1 kg quadrotor.

**Table II summary** (from the paper) across multiple trajectory parameter settings:
- At baseline `omega_xy`, RMS position error 6.6 cm, max error reported; yaw tracking also quantified with RMS and max errors in degrees.
- With high yaw rate (`omega_psi = 3 rad/s`): RMS position error no more than ~10–11 cm.
- At `omega_psi = 6 rad/s` (very high yaw rate): error increases moderately to ~12 cm, but remains well-bounded.

**Robustness validation**: Tests with a drag plate attached to the airframe and a rope pulling on the vehicle during hover confirmed that the INDI disturbance rejection maintains stability and reasonable tracking without any model re-identification.

**Hardware**: Nvidia Jetson TX2 at 500 Hz control loop, IMU at 1000 Hz, motion capture at 200 Hz (for state), optical encoders at 1000 Hz.

---

## Relevance to Our System

Our current system (`control/mpc_tracker.py`) implements a geometric SE(3) tracker (Lee et al.) plus a simple PD tracker. The Lee controller closes loops on attitude and angular rate but does not use jerk or snap feedforward from the trajectory. This is precisely the gap that Tal & Karaman address.

Our trajectory planner (`planning/trajectory_optimizer.py`) already produces min-snap polynomial trajectories, meaning all four orders of trajectory derivatives are analytically available. This is the raw material needed for INDI+flatness feedforward.

The INDI approach is also highly relevant to our situation because we cannot rely on an accurate aerodynamic model at high speed. The paper shows that INDI's implicit disturbance cancellation achieves 6.6 cm RMS without any drag model — comparable to or better than approaches that explicitly model drag.

The specific combination of lessons for our system:
1. We already have min-snap trajectories, so `p_ref`, `v_ref`, `a_ref`, `j_ref`, `s_ref` are available.
2. We could add jerk/snap feedforward to the geometric tracker without a full INDI redesign as a first step.
3. A full INDI inner loop for angular acceleration would require high-rate motor speed or angular acceleration feedback, which depends on our simulation adapter.

---

## Actionable Takeaways

1. **Add jerk feedforward to angular rate command**: In `control/mpc_tracker.py`, the attitude rate reference `omega_ref` should include the differential flatness term derived from the reference jerk. This term is `omega_ref = (I_body^-1) * (jerk_cross_thrust_direction_term)`. This is a direct, minimal change to the existing geometric tracker.

2. **Add snap feedforward to angular acceleration command**: Similarly, the reference angular acceleration `alpha_ref` computed from the reference snap can be added to the attitude controller output. Even without full INDI, feeding `alpha_ref` as a feedforward into the torque command reduces the phase lag induced by pure feedback.

3. **Extract derivatives from min-snap planner**: Verify that `planning/trajectory_optimizer.py` exposes all four derivative orders at query time. If only position/velocity/acceleration are returned, extend the polynomial evaluation to also return jerk and snap (third and fourth derivative of the polynomial coefficients).

4. **Consider INDI for the angular rate loop**: Replace or augment the current `mpc_tracker.py` angular rate PD feedback with an INDI-style incremental update using gyroscope-derived angular acceleration. This does not require optical encoders; angular acceleration can be estimated by differencing filtered gyro signals, which the EKF or a dedicated filter can provide.

5. **Low-pass filter consistency**: The INDI approach requires that the same LPF phase response is applied to both measured signals and commanded/predicted signals (e.g., IMU acceleration vs. computed thrust). In our software simulation this is straightforward to implement.

6. **Trajectory smoothness is a prerequisite**: INDI+flatness feedforward only works if the trajectory is genuinely C4 (four times differentiable and continuous). Min-snap polynomials satisfy this by construction. Piecewise trajectories with discontinuous snap at segment boundaries will cause impulsive feedforward commands; ensure continuity conditions are enforced at junctions.

---

## Limitations & Caveats

1. **Hardware dependency for snap tracking**: Full snap tracking as demonstrated in the paper requires closed-loop motor speed control with optical encoders providing 1000 Hz feedback. In our PyBullet simulation, motor torque/speed dynamics may not be modeled at this fidelity. The feedforward jerk term (angular rate) is implementable without this; the snap feedforward (angular acceleration) benefits from it but can provide partial improvement even without tight motor speed control.

2. **Computational cost**: The INDI update laws are simple algebraic operations (no matrix inversions beyond 3×3 at most), but the motor speed inversion uses Newton's method on a 4×4 nonlinear system at 500 Hz. This is fast in practice but requires implementation care.

3. **IMU-dependent**: The INDI linear acceleration loop requires a body-frame accelerometer measurement at high rate (1000 Hz in the paper). If our EKF state estimate latency is significant, the INDI correction may lag. The paper addresses this with the state predictor; our `estimation/state_predictor.py` plays a similar role.

4. **Motion capture state**: The experiments in this paper use motion capture for position and velocity, not onboard VIO. In a competition setting with gate-based drift correction, positional state uncertainty is larger. The INDI approach mitigates this somewhat because the disturbance rejection is local (based on accelerometers) rather than requiring globally accurate position estimates.

5. **No explicit gate/waypoint constraints**: The paper tracks a smooth analytic trajectory; it does not address gate-passing geometry or the problem of dynamically re-planning around gate passes. This limits direct applicability to the racing scenario without integration with a gate sequencer.

6. **Yaw at very high rates degrades tracking**: The paper acknowledges that at `omega_psi = 6 rad/s`, position error increases moderately. For our racing application where gate alignment dominates the yaw profile, yaw transitions should be planned smoothly with bounded yaw rate.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Vehicle mass | 1 kg | Experimental quadrotor |
| Top speed demonstrated | 12.9 m/s | Peak velocity in flight test |
| Peak acceleration | ~20.6 m/s² (2.1g) | During aggressive maneuver |
| RMS position tracking error | 6.6 cm | Main experimental result |
| Control loop rate | 500 Hz | Jetson TX2 |
| IMU rate | 1000 Hz | Linear accel + angular rate |
| Motion capture rate | 200 Hz | Position/velocity/attitude |
| Optical encoder rate | 1000 Hz | Motor speed measurement |
| LPF cutoff | 50 Hz (314 rad/s) | Applied to IMU and motor signals |
| LPF type | 2nd-order Butterworth | Software implementation |
| Trajectory differentiability | C4 in position, C2 in yaw | Required by flatness mapping |
| Trajectory class needed | min-snap polynomial | Provides analytic 4th derivatives |
| RMS tracking at high yaw (3 rad/s) | ~10–11 cm | Moderate degradation |
| RMS tracking at very high yaw (6 rad/s) | ~12 cm | Higher degradation |

**Control Table I summary**:
- Layer 1: PD on position/velocity → `a_cmd`
- Layer 2: INDI on linear acceleration → thrust vector command
- Layer 3: Differential flatness → `omega_ref`, `alpha_ref` from jerk, snap
- Layer 4: NDI on attitude and attitude rate → `alpha_cmd` (with `alpha_ref` feedforward)
- Layer 5: INDI on angular acceleration → torque command
- Layer 6: Nonlinear inversion + integrative motor speed control → PWM commands
