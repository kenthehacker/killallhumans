# The Reality Gap in Robotics: Challenges, Solutions, and Best Practices

- **URL**: https://rpg.ifi.uzh.ch/docs/AR25_Aljalbout.pdf
- **Authors**: Elie Aljalbout et al. (UZH Robotics and Perception Group — RPG)
- **Year**: 2025
- **Venue**: Annual Reviews (AR25) / arXiv 2510.20808

---

## Key Contribution

This paper is a comprehensive survey and taxonomy of the simulation-to-reality (sim-to-real) transfer problem across all of robotics, with a particular focus on learning-based controllers (reinforcement learning, imitation learning, and neural augmentation). The central contribution is a structured framework that categorizes (1) the *sources* of the reality gap, (2) the *mitigation strategies* and their tradeoffs, and (3) concrete *best practices* distilled from successful real-world deployments.

The paper's value is not a single novel algorithm but rather an authoritative synthesis of what actually works in practice, backed by the UZH RPG group's own experience deploying drone racing policies in physical hardware competitions (including the work behind the Swift autonomous racing champion). For a competition-bound system like ours, this survey functions as a checklist: every class of sim-to-real gap it describes is a potential failure mode we need to address before VQ1.

The authors draw from hundreds of prior works and organize them into a coherent reference that practitioners can use to diagnose and fix sim-to-real failures. The key insight threaded throughout is that the reality gap is not one problem but a *family* of orthogonal problems, and conflating them leads to poorly targeted fixes.

---

## Technical Approach

### 2.1 Taxonomy of Reality Gap Sources

The paper categorizes the reality gap into four primary source domains:

**1. Dynamics Gaps**
- Unmodeled forces: aerodynamic drag, motor back-EMF, gyroscopic precession, rotor downwash, ground effect
- Parameter uncertainty: mass, inertia tensor, center-of-mass location, motor thrust curves
- Actuator dynamics: motor spin-up/spin-down latency (typically 20–80 ms for brushless motors), ESC filtering, PWM-to-thrust nonlinearity
- Structural flexibility: propeller flex, arm vibration modes
- Battery voltage sag: thrust output degrades ~10–20% over a flight as cell voltage drops from 4.2V to 3.5V

**2. Sensor and Perception Gaps**
- Camera: motion blur, rolling shutter distortion, lens vignetting, HDR handling, latency (USB cameras: 30–100 ms; MIPI cameras: 5–15 ms)
- IMU: vibration aliasing (propeller harmonics at 2× and 4× rotor frequency), bias drift, temperature sensitivity
- State estimation lag: VIO pipelines typically add 50–200 ms of latency relative to ground truth
- Depth/LiDAR: reflectance variance, multipath returns, range noise not matching Gaussian assumption

**3. Environmental Gaps**
- Lighting variation: directional vs. ambient light, shadows, reflective surfaces
- Wind and turbulence: quiescent indoor air in simulation vs. airflow from other drones, HVAC, competition arena effects
- Ground clutter and texture diversity in visual scenes
- Gate/obstacle material properties: reflectivity, color calibration

**4. Compute and Timing Gaps**
- Simulation timestep vs. real controller execution jitter
- GPU inference latency (neural policies): not constant — varies 1–10 ms with thermal throttling
- Memory bandwidth bottlenecks on embedded hardware
- OS scheduling: real-time kernel vs. standard kernel latency differences

### 2.2 Mitigation Strategies

The paper organizes mitigation strategies along a spectrum from "fix the simulation" to "fix the policy."

**A. System Identification (SysID)**

The most effective first step. Measure the actual parameters of your hardware and update the simulation accordingly. The paper emphasizes that even basic SysID (thrust curve fitting, mass measurement, IMU calibration) recovers most of the gap for dynamics before any randomization. Key protocol:

- Hover tests: measure steady-state throttle vs. altitude to calibrate thrust-to-weight ratio
- Step response tests: command step throttle inputs and fit first-order motor model (time constant τ_motor)
- Chirp tests: identify rotor inertia and drag coefficients across frequency
- Battery profiling: measure thrust at multiple SOC levels; fit polynomial voltage-to-thrust correction

**B. Domain Randomization (DR)**

After SysID, apply randomization *around* the identified mean. DR works by training the policy over a distribution of environments; if the real world falls within that distribution, the policy generalizes.

Effective DR ranges cited in the paper for quadrotors:
- Mass: ±15–25% of nominal
- Inertia: ±20–30%
- Motor time constant (τ): 15–80 ms range
- Thrust coefficient (kT): ±10%
- Drag coefficient: ±50% (less well-characterized)
- IMU noise: scale by 0.5×–3×
- Observation delay: 0–3 timesteps (at 50–100 Hz: 0–60 ms)
- Wind: uniform random 0–3 m/s with random direction

**C. Dynamics Randomization vs. Visual Randomization**

The paper distinguishes these carefully. Dynamics randomization (physics parameters) is essential for control policies. Visual randomization (textures, lighting, camera artifacts) is essential for perception pipelines. For gate-racing with known gate positions (classical planning), visual randomization matters primarily for the gate detection front-end, not the tracker/controller.

**D. Residual / Neural-Augmented Dynamics**

A learned residual model `f_residual(s, a)` is added to the nominal simulator:

```
s_{t+1} = f_nominal(s_t, a_t) + f_residual(s_t, a_t)
```

The residual is trained on rollout data from the real robot. This approach is particularly effective for unmodeled aerodynamic effects that are difficult to hand-code (blade flapping, asymmetric drag). The paper reports that residual models trained on as few as 50–200 real-world rollouts can cut the dynamics gap by 40–70%.

**E. Adaptive Policies and Online System Identification**

Policies augmented with an online estimator that infers current system parameters (e.g., drag coefficient, wind) from recent state-action history. The estimator output is appended to the policy's observation. This allows the policy to adapt to conditions outside the training distribution without retraining. The paper links this to the "context encoder" architecture (also called LSTM-augmented or privileged information distillation).

**F. Sim-to-Real Fine-Tuning**

Starting from a simulation-trained policy and fine-tuning on real hardware. Key finding: fine-tuning with as few as 10–50 real episodes recovers substantial performance, but only if the simulation policy is already within a "competent but imperfect" regime — if the sim policy crashes on transfer, fine-tuning is unstable. The implication is that simulation training must be "good enough" to at least hover and fly semi-stably before hardware deployment.

### 2.3 Best Practices for Competition Deployment

The paper distills a deployment checklist derived from successful competition systems (explicitly including the UZH RPG drone racing work):

1. **Measure before you randomize.** SysID first, then wrap a distribution around the measured values. Randomizing around an uncalibrated nominal wastes compute and produces conservative policies.

2. **Close the loop on latency.** Measure total perception-to-actuator latency on your actual hardware stack (camera → VIO → planner → motor command). Add this measured latency explicitly to the simulation training loop. The paper identifies unmodeled latency as the single most common cause of transfer failures.

3. **Match the control frequency.** Train at the same frequency you deploy. If your real controller runs at 50 Hz, train at 50 Hz. Running at 200 Hz in simulation and 50 Hz in deployment causes subtle instabilities.

4. **Test distribution coverage.** Before competition, characterize the arena: measure ambient light levels, gate dimensions, floor texture, and airflow. Verify that these fall within the training distribution. If not, expand DR or retrain.

5. **Use privileged information during training only.** During simulation training, give the policy access to ground-truth wind, exact parameters, etc. (privileged information). At deployment, use only real sensor inputs. This "asymmetric actor-critic" approach consistently outperforms training with only real-sensor inputs.

6. **Deploy conservatively first.** First flight should be at reduced speed/aggressiveness. Establish that the policy is stable before pushing to competition velocities. A crash on the first run can damage hardware and lose competition slots.

7. **Log everything.** IMU, motor commands, state estimates, and timestamps must all be logged with hardware timestamps (not software timestamps) to allow post-flight reality gap diagnostics.

---

## Results

The paper aggregates quantitative results across many systems. Key findings relevant to drone racing:

**Tracking error reduction from SysID alone:** Systems that applied systematic SysID before training reported 30–60% reduction in sim-to-real tracking error compared to using simulator defaults, without any domain randomization.

**Domain randomization diminishing returns:** Adding DR beyond ±30% variation in dynamics parameters yields little additional benefit and can *hurt* performance by forcing overly conservative policies. The "useful" randomization window is narrow.

**Motor latency dominates at high speed:** At flight speeds above 5 m/s, unmodeled motor response time (τ_motor) was the dominant source of tracking error in transferred policies. Policies trained without motor dynamics models showed 3–5× higher tracking error at high speed than at low speed — the gap is speed-dependent.

**Observation delay sensitivity:** Every 20 ms of unmodeled observation delay corresponded to approximately 0.15–0.3 m of additional cross-track error at 10 m/s flight speed. This is directly relevant: our EKF + state predictor latency compensation stack must be calibrated against actual sensor timestamps.

**Residual model benefit:** Neural residual dynamics trained on 100 real rollouts reduced position tracking RMSE by ~45% compared to nominal simulation baseline on quadrotor hover tasks.

**Competition-specific finding (Swift / UZH RPG context):** The Swift autonomous racing system (Kaufmann et al., Nature 2023) achieved champion-level performance through systematic SysID + DR. The paper notes that the key was **not** the neural architecture but the fidelity of the simulation: "the policy architecture matters less than the simulation accuracy."

---

## Relevance to Our System

Our pipeline (`race_pipeline.py`, `control/mpc_tracker.py`, `estimation/ekf.py`) is a classical (non-learned) planning and control stack. The reality gap manifests for us differently than for a pure RL policy, but most of the paper's findings still apply directly:

**1. Motor dynamics not modeled.** Our `pybullet_adapter.py` likely uses instantaneous thrust application. Real brushless motors have τ_motor ≈ 20–50 ms. At racing speeds (10+ m/s), a 30 ms lag means the drone overshoots turns by 0.3–0.5 m. Our `state_predictor.py` compensates for perception latency but not actuator latency. This is a gap.

**2. EKF latency calibration.** Our `ekf.py` uses process noise parameters (`EKFConfig`) tuned in simulation. If real-world VIO or IMU noise differs from simulation noise, the EKF will be miscalibrated, leading to poor state estimates and high tracking error. The paper recommends logging real sensor data and fitting noise parameters empirically.

**3. Drag not modeled (likely).** Our min-snap trajectory optimizer (`trajectory_optimizer.py`) computes segment times assuming a drag-free model or a simple drag estimate. Real aerodynamic drag is nonlinear and speed-dependent. At high speed, this leads to systematic trajectory time errors: the drone arrives at gates later than planned, and the sequencer (`sequencer.py`) may miss pass-through detections.

**4. Battery voltage sag.** If testing across multiple runs without recharging, thrust output degrades. Our `TrackerConfig` gains are fixed. A feedforward thrust correction based on estimated battery state would improve consistency.

**5. PyBullet physics fidelity.** The paper notes that PyBullet is a reasonable baseline for rigid-body dynamics but lacks aerodynamic modeling, rotor interactions, and ground effect. Our benchmark (`scripts/benchmark.py`) uses PyBullet as "ground truth" — but PyBullet is itself a simplified simulation. Our real-world performance will likely differ from benchmark scores in ways that the paper's taxonomy predicts.

**6. Gate detection latency.** If we use `gate_pnp.py` for drift correction, the PnP solve adds processing latency. The paper's finding — every 20 ms of unmodeled delay ≈ 0.15–0.3 m tracking error at 10 m/s — means our PnP latency must be profiled and compensated.

---

## Actionable Takeaways

1. **Profile and compensate motor latency in `mpc_tracker.py`.** Add a first-order motor model with τ_motor = 30 ms (or measure from real hardware). Apply command pre-compensation: send commands that anticipate the lag. This is likely the highest-impact single fix for high-speed tracking.

2. **Calibrate EKF noise parameters from real sensor data.** Before any competition flight, collect a slow hover flight, record IMU and position outputs, and fit the `EKFConfig.process_noise` and `EKFConfig.measurement_noise` parameters to match. Do not use PyBullet defaults.

3. **Add drag feedforward to trajectory timing.** In `trajectory_optimizer.py`, incorporate a quadratic drag term into the segment time estimator. At 10 m/s with a typical drag coefficient of ~0.3 N·s/m per axis, drag deceleration is ~0.3 g. Segment times need to be 10–15% longer at high speed to remain feasible.

4. **Implement battery state correction.** Track estimated voltage from flight time and throttle average. Apply a scalar correction to thrust commands. A simple linear model (thrust_correction = 1 - k*(4.2V - V_batt)) prevents performance degradation mid-race.

5. **Stress-test EKF with injected noise.** Run `benchmark.py` with artificially increased sensor noise levels (2×, 5×) to characterize how much EKF degradation we can tolerate. This gives a safety margin estimate for real-world conditions.

6. **Use the paper's deployment checklist before VQ1.** Specifically: measure total pipeline latency end-to-end, verify gate detection operates within training distribution light levels, and perform at least 3 slow test flights before full-speed competition runs.

7. **Consider residual dynamics if hardware access is available.** If we get arena access before VQ1, log 50+ flights and train a small residual model to correct PyBullet-vs-real errors. The paper reports this is the most effective post-SysID improvement.

8. **Match control loop frequency to deployment target.** If the competition hardware runs at a different rate than our `benchmark.py` simulation, retune gains. The paper identifies frequency mismatch as a hidden failure mode.

---

## Limitations & Caveats

**Survey scope, not experimental paper.** This is a review article, not a primary research paper. The quantitative claims (e.g., "residual model reduces error by 45%") are aggregated from other papers with different hardware, tasks, and conditions. The numbers should be treated as rough guides, not precise predictions for our system.

**RL/learning focus.** Most of the paper's solutions assume a learned policy (RL or imitation) that can be retrained. Our system uses classical planning and control; the solutions that require retraining (DR, privileged information, fine-tuning) do not apply directly. The portions most relevant to us are the SysID, latency modeling, and diagnostic sections.

**PyBullet implicit coverage.** The paper discusses simulation fidelity across many engines but does not specifically benchmark PyBullet's aerodynamic fidelity for drones at racing speeds. We cannot directly infer from this paper how large the PyBullet-to-real gap is for our specific benchmark.

**No racing-line specifics.** The paper covers control policies broadly but does not discuss min-snap trajectory optimization, racing line computation, or gate sequencing. It treats "trajectory tracking error" as the metric but does not analyze the trajectory generation side. Our planning stack is not directly addressed.

**Competition context varies.** The paper's best practices are derived from platforms (legged robots, manipulation arms, general navigation drones) that may have different priorities than a gate-racing competition drone. Specifically: competition racing prioritizes lap time aggressiveness over safety margin, which inverts some of the paper's "deploy conservatively" recommendations.

**No discussion of gate-passing specifically.** Pass-through detection, gate sequencing, and drift-corrected VIO (our `gate_tracker.py`, `gate_pnp.py`) are not covered. The paper's sensor gap discussion applies to cameras and IMUs generally, but not to the specific PnP-based gate pose estimation pipeline.

---

## Key Parameters / Constants

| Parameter | Value / Range | Source / Context |
|-----------|---------------|-----------------|
| Motor time constant τ_motor | 20–80 ms | Brushless motor first-order model; typical for racing drones |
| Typical total perception-to-actuator latency | 50–200 ms | VIO-based state estimation pipeline |
| Camera latency (USB) | 30–100 ms | Consumer USB cameras |
| Camera latency (MIPI) | 5–15 ms | Embedded MIPI CSI cameras |
| Domain randomization — mass | ±15–25% | Recommended range for quadrotors |
| Domain randomization — inertia | ±20–30% | Recommended range for quadrotors |
| Domain randomization — motor τ | 15–80 ms range | Span entire plausible range |
| Domain randomization — thrust coefficient kT | ±10% | Motor-to-motor variation |
| Domain randomization — drag coefficient | ±50% | High uncertainty, large range needed |
| Domain randomization — IMU noise scale | 0.5×–3× nominal | Cover vibration regimes |
| Domain randomization — observation delay | 0–3 timesteps | At 50–100 Hz controller |
| Domain randomization — wind | 0–3 m/s uniform random | Indoor competition conditions |
| Tracking error per 20 ms unmodeled latency | 0.15–0.3 m | At ~10 m/s flight speed |
| Residual model training data | 50–200 real rollouts | Sufficient for 40–70% gap reduction |
| Sim-to-real fine-tuning data | 10–50 real episodes | Sufficient if sim policy is pre-competent |
| DR useful range (dynamics) | ≤±30% | Beyond this, diminishing or negative returns |
| Battery voltage range (LiPo 1S) | 3.5–4.2 V | Full to nominal; ~15% thrust loss across range |
| High-speed drag deceleration estimate | ~0.3 g at 10 m/s | For typical racing drone aero configuration |
| Trajectory timing correction (drag) | +10–15% at high speed | Segment time inflation for drag feasibility |
| IMU vibration aliasing frequencies | 2× and 4× rotor RPM | Propeller harmonic contamination bands |
| Useful SysID: thrust-curve test | Hover: 0–100% throttle | Fit kT, kD from steady-state measurements |
| Useful SysID: step response | Rise time → τ_motor | Command step; fit first-order exponential |
| Useful SysID: frequency chirp | 0–50 Hz bandwidth | Identify rotor inertia, resonance modes |
