# Accurate Tracking of Aggressive Quadrotor Trajectories using INDI and Differential Flatness
- **URL**: https://arxiv.org/abs/1809.04048
- **Authors**: Ezra Tal, Sertac Karaman (MIT)
- **Year**: 2018 (conference); 2021 (journal, IEEE Transactions on Control Systems Technology, Vol. 29, No. 3, pp. 1203–1218)

---

## Key Contribution

Tal & Karaman introduce a unified control law that tracks position, yaw, and all their time derivatives up to fourth order: position, velocity, acceleration, jerk, and snap, together with yaw rate and yaw acceleration. This is a qualitative departure from conventional quadrotor controllers, which typically close loops on position and velocity and treat reference acceleration as a feedforward signal at best. By tracking jerk and snap, the controller effectively anticipates the rapid attitude changes required before positional error has time to accumulate.

Two technically distinct mechanisms enable this. First, the differential flatness of quadrotor dynamics is exploited to derive closed-form feedforward terms for the reference angular rate (`omega_ref`) and reference angular acceleration (`alpha_ref`) directly from the trajectory's jerk and snap. Second, Incremental Nonlinear Dynamic Inversion (INDI) provides disturbance rejection for both the linear acceleration loop (handling aerodynamic drag) and the angular acceleration loop (handling unmodeled torque perturbations), without requiring any prior aerodynamic model. The result is a 1 kg quadrotor achieving 6.6 cm RMS position error at 12.9 m/s with accelerations up to 2.1g.

---

## Technical Approach

### Differential Flatness and the Feedforward Chain

The quadrotor has four flat outputs: inertial position `p = [x, y, z]^T` and yaw angle `psi`. Given a reference trajectory that is C4 in position and C2 in yaw, every required state and control input — including body attitude, angular rate, angular acceleration, thrust, and torques — can be computed algebraically from those four quantities and their derivatives.

The feedforward derivation chain proceeds as follows:

1. **Thrust direction**: the reference specific force vector is `f_ref = a_ref - g*e_z` (reference acceleration minus gravity). Normalized, this defines the body z-axis (the thrust direction).
2. **Reference attitude (R_ref)**: fully specified by the thrust direction and yaw, via the standard ZXY decomposition.
3. **Reference angular rate (omega_ref)** — the jerk feedforward term: differentiating the thrust direction equation with respect to time yields an equation affine in body angular rate. Inverting this (a closed-form 3-vector calculation) gives `omega_ref` as a direct function of `a_ref`, `j_ref` (jerk), and `psi_dot`.
4. **Reference angular acceleration (alpha_ref)** — the snap feedforward term: differentiating the jerk equation once more and inverting yields `alpha_ref` as a function of `a_ref`, `j_ref`, `s_ref` (snap), `psi_dot`, and `psi_ddot`.

These two feedforward signals are injected into the attitude rate and attitude acceleration controllers, respectively. Because they are derived analytically from the trajectory — not from feedback error — they carry no phase lag. When the trajectory begins curving (e.g., entering a turn), `omega_ref` and `alpha_ref` instantly reflect the required attitude motion even before any positional deviation has occurred.

### INDI Control Law

INDI operates by treating the system's current dynamics as approximately constant over one control timestep and computing only the increment to the control input needed to achieve the commanded state change. This sidesteps the need for an accurate aerodynamic model.

For the linear acceleration loop: the measured inertial acceleration (from the accelerometer, gravity-corrected) is compared to the commanded acceleration. The difference between measured acceleration and the contribution attributable to the known thrust vector gives an implicit estimate of the external force (drag, etc.). The new thrust command is then: `f_cmd = f_current + K_indi * (a_cmd - a_measured)`. Both `f_current` and `a_measured` pass through identical low-pass filters (2nd-order Butterworth, 50 Hz cutoff) to ensure phase-matched comparison.

For the angular acceleration loop: the measured angular acceleration (from gyro differentiation) is compared to the commanded angular acceleration `alpha_cmd`. The new torque command is: `tau_cmd = tau_current + J * (alpha_cmd - alpha_measured)`, where `tau_current` is reconstructed from optical encoder motor speed measurements.

### Six-Layer Control Architecture

| Layer | Inputs | Outputs |
|-------|--------|---------|
| 1. PD position/velocity | `p_err`, `v_err` | `a_cmd` |
| 2. INDI linear acceleration | `a_cmd`, accel measurement | thrust direction + magnitude |
| 3. Diff. flatness mapping | `j_ref`, `s_ref`, `psi_dot`, `psi_ddot` | `omega_ref`, `alpha_ref` |
| 4. NDI attitude + rate | attitude error, `omega_err`, `omega_ref` | `alpha_cmd` (+ `alpha_ref` FF) |
| 5. INDI angular acceleration | `alpha_cmd`, gyro-derived `alpha_meas` | torque command |
| 6. Nonlinear motor inversion | thrust + torque → PWM | motor speed commands |

### Jerk Feedforward and Overshoot Reduction

The paper explicitly demonstrates that enabling the jerk feedforward (layer 3 → layer 4 path) reduces response overshoot in simulation. The mechanism is direct: without the jerk feedforward, the attitude rate controller only reacts after positional error begins to grow. With `omega_ref` derived from `j_ref`, the vehicle begins rotating toward the new thrust direction at the onset of the curvature change, before error accumulates. The snap feedforward `alpha_ref` provides a further improvement by pre-commanding the torque required to accelerate the angular rate, reducing the lag in the angular rate tracking itself.

The paper reports that overshoot is visually reduced in step response tests and in the trajectory tracking results when higher-order derivatives are included. This is not a lookahead in the time-domain sense (sampling a future trajectory point) but rather a mathematical consequence of tracking derivatives of the current trajectory state.

### Snap Tracking and Motor Encoder Requirement

Snap corresponds to position's fourth derivative, which maps through the flatness transform to angular acceleration, which is generated by applied body torque. Accurate snap tracking therefore requires precise, closed-loop torque delivery. The paper achieves this with optical encoders on each motor hub at 1000 Hz, enabling tight motor speed control. Integral action compensates for battery voltage sag. Without this, the snap feedforward still improves behavior in practice (because it improves commanded angular acceleration targets) but the physical accuracy of torque delivery is reduced.

---

## Results

- **RMS position tracking error**: 6.6 cm at 12.9 m/s peak velocity and 2.1g (20.6 m/s²) peak acceleration
- **Flight volume**: 18 m × 7 m × 3 m
- **High yaw rate tests**: RMS error rises to ~10–11 cm at `omega_psi = 3 rad/s`, ~12 cm at `omega_psi = 6 rad/s`
- **Robustness test**: drag plate tripling frontal area attached without controller re-tuning — no significant degradation in tracking error, demonstrating INDI's implicit disturbance rejection
- **External force test**: rope tension applied during hover — position maintained within several centimeters

---

## Relevance to Our System

Our system (`control/mpc_tracker.py`) implements a geometric SE(3) tracker (Lee et al.) plus a PD tracker with an acceleration feedforward term (`feedforward_accel = 0.50`). The ILC optimizer in `planning/trajectory_optimizer.py` uses a 50 ms positional lookahead (`ff_lookahead_s = 0.05`) to sample the trajectory 50 ms ahead and use that future acceleration as the feedforward signal during ILC simulation.

**The 50ms lookahead and overshoot**: The positional lookahead used in the ILC loop is a time-domain preview, sampling `a_ref(t + 0.05s)` rather than `a_ref(t)`. At straight→turn transitions this causes overshoot because the feedforward acceleration from 50 ms in the future already reflects the turn's lateral component before the drone needs it. The drone begins turning early, overshoots the entry of the turn, and then the feedback controller corrects. This is the opposite of what the Tal & Karaman jerk feedforward achieves.

**The correct fix per this paper**: Tal & Karaman do not use positional lookahead at all. Instead, `omega_ref` (derived from jerk) is fed into the angular rate controller as an instantaneous feedforward of the current trajectory's first derivative of acceleration. This is mathematically anticipatory — it tells the controller "the thrust direction is about to change at this rate" — but it does not shift reference positions forward in time. The result is that the drone's attitude smoothly follows the trajectory curvature without generating positional overshoot.

**Recommendation on lookahead**: Reducing the lookahead from 50 ms toward 0 ms should reduce overshoot at straight→turn transitions, at the cost of slightly reduced anticipation of curvature onset. The right solution longer-term is to replace the positional lookahead entirely with a jerk feedforward term for angular rate, using the existing min-snap trajectory's analytically available jerk (`j_ref`). This is a direct implementation of layer 3 of the Tal & Karaman architecture.

**Structural compatibility**: Our `planning/trajectory_optimizer.py` produces min-snap polynomials whose third and fourth derivative coefficients are already stored. Exposing a `jerk` and `snap` field from `trajectory.sample()` and passing them into `mpc_tracker.py` to compute `omega_ref` via the flatness mapping is a focused, measurable change. A full INDI inner loop for angular acceleration would require gyro-derived angular acceleration feedback; a partial implementation (jerk feedforward only, keeping the existing PD angular rate loop) is meaningful on its own.

---

## Actionable Takeaways

1. **Reduce or eliminate the 50ms positional lookahead at straight→turn transitions**: The current `ff_lookahead_s = 0.05` in the ILC optimizer prematures the curvature-induced lateral acceleration command, causing overshoot when the drone enters a turn. Reducing this to 0 ms (or gating it off near trajectory curvature maxima) should reduce the entry overshoot observed at those transitions.

2. **Add jerk feedforward for angular rate**: In `control/mpc_tracker.py`, extend the reference angular rate target to include `omega_ref` derived from `j_ref` via the differential flatness mapping. The core formula is: compute the thrust direction's time derivative from `j_ref`, then solve for `omega_ref` using the cross-product inversion described in the paper. This is a ~20-line addition that requires exposing jerk from `planning/trajectory_optimizer.py`.

3. **Expose jerk (and optionally snap) from trajectory sampling**: Verify that `TrajectoryPoint` returned by `trajectory.sample(t)` includes the third (and fourth) polynomial derivative. If not, add these fields to the polynomial evaluation in `trajectory_optimizer.py`. This is a prerequisite for any jerk/snap feedforward work.

4. **Evaluate INDI for the angular rate loop as a follow-on**: The existing geometric tracker applies PD feedback on angular rate. Augmenting this with an INDI-style correction using differentiated IMU gyro signals would improve disturbance rejection at high speed. This does not require optical encoders; numerical differentiation of the gyro signal through a 50 Hz LPF is sufficient for the angular rate INDI loop (snap-level INDI requires motor feedback).

5. **Ensure trajectory is C4 at segment junctions**: Min-snap polynomial planners enforce C4 continuity by construction within a segment, but verify that the segment boundary conditions in `trajectory_optimizer.py` impose continuity of jerk and snap. Discontinuities at boundaries produce impulsive feedforward terms.

---

## Limitations & Caveats

1. **Optical encoders required for full snap tracking**: Achieving the paper's 6.6 cm result with snap feedforward requires 1000 Hz closed-loop motor speed control. In our PyBullet simulation, motor dynamics may not be modeled at this fidelity. The jerk feedforward (angular rate) is implementable without this and provides the larger portion of the benefit.

2. **Positional lookahead vs. derivative feedforward are architecturally different**: The 50ms lookahead shifts the reference position in time; derivative feedforward uses the trajectory's instantaneous higher-order information. These are not interchangeable. Removing lookahead may cause a temporary degradation in response to curvature onset that jerk feedforward then compensates; both changes should be made together.

3. **Motion capture state in experiments**: The paper uses motion capture for ground-truth position and velocity. Our competition system relies on EKF with gate-based drift correction. State estimation latency can degrade INDI performance; our `estimation/state_predictor.py` partially compensates but does not fully eliminate this.

4. **Smooth analytic trajectory only**: The paper does not address discrete waypoints, dynamic re-planning, or gate sequencing. The INDI+flatness approach requires a smooth, parameterized trajectory with continuous fourth derivatives. Racing trajectories that are re-planned online on gate detection must maintain C4 continuity through the gate transition.

5. **No explicit gate pass-through geometry**: The paper's lemniscate test trajectory avoids the gate-centering constraint specific to drone racing. Integrating the jerk feedforward with gate passage accuracy requires verifying that the flatness transform's thrust direction references are compatible with the gate pass-through margin at full speed.

6. **High yaw rate degrades tracking moderately**: At 6 rad/s yaw rate, RMS error rises to ~12 cm. Racing courses requiring rapid yaw alignment through gates should plan smooth, bounded yaw profiles rather than aggressive yaw transitions.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Vehicle mass | 1 kg | Experimental quadrotor |
| Top speed | 12.9 m/s | Peak in flight test |
| Peak acceleration | ~20.6 m/s² (2.1g) | Aggressive maneuver |
| RMS position tracking error | 6.6 cm | Primary experimental result |
| Control loop rate | 500 Hz | Jetson TX2 onboard computer |
| IMU rate | 1000 Hz | Linear accelerometer + gyroscope |
| Motion capture rate | 200 Hz | Position, velocity, attitude |
| Optical encoder rate | 1000 Hz | Per-motor speed feedback |
| LPF cutoff frequency | 50 Hz (314 rad/s) | Applied to IMU and motor signals |
| LPF type | 2nd-order Butterworth | Matched filter on all INDI signals |
| Trajectory differentiability | C4 in position, C2 in yaw | Minimum required by flatness mapping |
| RMS error at omega_psi = 3 rad/s | ~10–11 cm | High yaw rate degradation |
| RMS error at omega_psi = 6 rad/s | ~12 cm | Very high yaw rate |
| Drag plate frontal area multiplier | 3× | Used in robustness validation |
| Angular rate feedforward | `omega_ref` from jerk via flatness | Layer 3 output |
| Angular acceleration feedforward | `alpha_ref` from snap via flatness | Layer 3 output, requires motor encoder |
| Our current positional lookahead | 50 ms | `ff_lookahead_s = 0.05` in `trajectory_optimizer.py` |
