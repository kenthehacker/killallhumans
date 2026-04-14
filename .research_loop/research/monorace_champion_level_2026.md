# MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI

- **URL**: https://arxiv.org/abs/2601.15222
- **Authors**: Bahnam, Ferede, Blaha, Lang, Lucassen, Missinne, Verraest, De Wagter, de Croon
- **Year**: 2026
- **Venue**: arXiv (A2RL competition winner, TU Delft)

---

## Key Contribution

MonoRace is the first autonomous drone racing system to defeat human world champion FPV pilots in direct head-to-head competition. It won the 2025 Abu Dhabi Autonomous Drone Racing Competition (A2RL), outperforming all competing AI teams and three human world champions in direct knockout rounds. The system achieved this using only a single monocular rolling-shutter camera and IMU — no external motion capture, no stereo cameras, no LiDAR.

The central technical contribution is a robust, fully-onboard perception-control pipeline that handles real-world failure modes (IMU saturation at >16g, electromagnetic camera interference causing up to 50% corrupted frames, calibration drift) that destroy naive approaches. The key insight is that champion-level performance requires not just a fast nominal policy, but a system that degrades gracefully under sensor failures.

A secondary contribution is the Guidance-and-Control Network (G&CNet) approach: a compact 3-layer MLP running at 500 Hz directly on the flight controller that outputs raw motor commands, bypassing all traditional inner-loop PID controllers. This achieves 2ms command rise times and 100 km/h peak speeds.

---

## Technical Approach

### System Overview

The pipeline is: Camera (90 Hz) → GateNet segmentation → QuAdGate corner detection → Multi-gate PnP pose estimation → EKF (fusing 1000 Hz IMU + vision) → G&CNet policy (500 Hz, motor commands).

Hardware: 966g quadcopter, NVIDIA Jetson Orin NX companion computer, STM32H743 flight controller, monocular rolling-shutter CMOS camera (155° × 115° FOV).

### Perception Pipeline

**Image capture and adaptive cropping:** Raw images are 820×616 pixels at 90 Hz. Rather than processing the full frame, the system uses adaptive cropping to extract a 384×384 region containing the two closest visible gates. Crop region is guided by predicted gate locations from the EKF. This balances computational efficiency (mandatory on Jetson Orin NX) against pixel accuracy — a naive fixed crop would lose gates during aggressive maneuvers when the camera angle shifts dramatically.

**GateNet (gate segmentation):** A U-Net-style encoder-decoder with skip connections. Trained using Dice loss combined with binary cross-entropy for 100 epochs with learning rate scheduling. Produces binary segmentation masks for gate regions. The U-Net architecture was chosen for robustness to partial occlusion and varying lighting conditions. Output is a mask, not bounding boxes, which is more robust to the non-rectangular gate appearances that occur at oblique angles.

**QuAdGate (corner extraction):** Extracts precise sub-pixel corner positions from segmentation masks using line detection and intersection computation. Uses RANSAC-based affine registration to reject erroneous corners. The sub-pixel precision is critical for downstream PnP accuracy — coarse pixel-level corners would introduce unacceptable heading estimation error at racing speeds.

**Multi-gate PnP pose estimation:** The key innovation here is combining corners from multiple gates (typically the two closest visible) in a single PnP optimization, rather than solving each gate independently. This dramatically improves heading (yaw) estimation because the relative geometry between two gates provides strong rotational constraints — analogous to how a wider baseline improves stereo depth estimation. Approximately 27% of frames benefit from multi-gate corner fusion. Single-gate PnP requires a minimum of 4 corners; the multi-gate approach meets this threshold more frequently and produces more accurate estimates. A fallback uses EKF attitude with PnP-estimated relative translation when gate distance is outside 2–5m or fewer than 6 corners are detected.

**Offline self-supervised calibration:** Camera extrinsic calibration is refined offline using Intersection over Union (IoU) between reprojected and observed gate masks, leveraging the known gate geometry (1.5m inner opening). This achieves sub-degree accuracy without ground truth. Critically, this calibration runs on actual onboard flight data, accounting for vibration-induced sensor misalignment that laboratory calibration misses. The camera angular offset (θ_cam) is 43–50 degrees depending on the trained network — a significant forward pitch to keep gates in view at high speed.

### Extended Kalman Filter (State Estimation)

**State vector (16 states):** position [x, y, z], velocity [v_x, v_y, v_z], orientation quaternion [q_w, q_x, q_y, q_z], accelerometer biases [b_x, b_y, b_z], gyroscope biases [b_p, b_q, b_r].

**Prediction:** Continuous-time dynamics discretized with Δt. Process model includes gravity and quaternion kinematics. Control input is [a_x, a_y, a_z, p, q, r] from IMU at 1000 Hz (accelerometer) and 2000 Hz (gyroscope).

**Vision update:** PnP pose measurements update position and optionally attitude. Outlier rejection uses a Mahalanobis distance gate: measurements rejected if ||x_pos - x_PnP||² ≥ 16·N_c²·trace(P_pos), where N_c is the number of corners used and P_pos is the position covariance. This adaptive threshold scales with corner count and uncertainty — fewer corners means a tighter gate to avoid integrating noisy measurements.

**Measurement noise model:** Position noise variance is distance-dependent: σ_pos² = 0.02·d_gate²/(N_c²·N_g), where d_gate is gate distance, N_c is corner count, N_g is number of gates. Far gates with few detected corners get high noise variance, appropriately down-weighting their contribution.

**IMU saturation fallback:** During aggressive maneuvers, accelerometers saturate beyond ±16g. MonoRace detects saturation by computing the Euclidean norm of the difference between low-pass-filtered model-predicted acceleration and measured IMU acceleration. When this exceeds 22 m/s², the EKF switches from measured IMU accelerations to model-predicted accelerations and inflates uncertainty (increasing reliance on visual measurements). The aerodynamic model captures linear and quadratic drag, angle of attack effects, advance ratio dependencies, gyroscopic coupling from battery/compute asymmetry, and motor response saturation. Parameters identified via linear regression on high-speed flight data. This fallback raised aggressive maneuver success rates from 50% to 100%.

**Camera interference handling:** Electromagnetic interference (common in competition arenas with motors, ESCs, and video transmitters) corrupts up to 50% of camera frames. RANSAC in QuAdGate rejects corrupted frame corners; the Kalman filter's outlier rejection discards measurements too far from prediction. Combined, the system tolerates extended periods of visual-only IMU propagation.

### Guidance and Control Network (G&CNet)

A 3-layer MLP with 64 neurons per layer (ReLU activations) takes 24-dimensional observations as input and outputs 4 motor commands directly. No inner-loop controllers (attitude control, rate control) are involved.

**Observations (24D):** Relative position and velocity to current and next gate, current attitude (represented as rotation matrix or Euler angles), angular rates, and current motor commands. The gate-centric state representation is critical — it makes the policy invariant to global position and heading, enabling generalization across different sections of the track.

**Training:** Proximal Policy Optimization (PPO) in a custom quadrotor simulator. Extensive domain randomization across 50+ model parameters with 30–55% variation ranges. Motor response delays, drag coefficients, mass, inertia, and sensor noise are all randomized. Zero-shot sim-to-real transfer achieved.

**Speed-robustness tradeoff explored:** Network M16 achieved 16.56s lap time but lower success rate (priority on speed). Network M23 achieved 88.4% success rate across 43 flights at slightly slower pace. Competition used M16 for the Grand Challenge time trial and M23 for reliability-critical knockout rounds.

---

## Results

- **Grand Challenge (fastest lap):** 16.56 seconds for two laps on an 11-gate, 76×18×5.4m track
- **Peak velocity:** 28.23 m/s (~100 km/h) — fastest fully onboard autonomous result to date
- **AI vs Human:** Three consecutive knockout victories against world champion FPV pilots
- **Multi-Drone Race:** Third place (no collision avoidance implemented)
- **Drag Race:** Won
- **Robustness:** M23 network: 88.4% success rate across 43 flights; M17 network: 71.4% (55/77 flights)
- **Sim-to-real fidelity:** Real-world lap times within ~2 seconds of simulation predictions
- **Multi-gate PnP benefit:** ~2° heading improvement, 27% of frames benefit
- **IMU fallback impact:** Success rate under saturation improved from 50% to 100%

Track: 11 square gates (1.5m inner opening, 2.7m outer frame), two double-gates, one split-S maneuver, 100×30m indoor arena.

Comparison: "Swift" (Kaufmann, Nature 2023) used stereo cameras and external motion capture on a different track — not a fair direct comparison. MonoRace is the first champion-level result under full competition constraints (monocular, onboard only).

---

## Relevance to Our System

### FOV Constraints: Perception Penalty vs. Integrated Cost

This is the critical question for our system. MonoRace's approach to FOV/gate visibility is **integrated into the reward function during RL training**, not post-processed or geometrically constrained after the fact:

> "if θ_cam > π/3, [penalize], encouraging keeping the next gate within the camera's view."

This is a **soft penalty during training** — the G&CNet learns implicitly to keep gates in the field of view as part of optimizing the total reward, not an explicit geometric constraint during execution. There is no hard FOV constraint enforcement at deployment time; robustness comes from the trained policy having internalized this preference.

**Contrast with our current architecture:** Our system uses a pre-computed polynomial trajectory (min-snap) with the MPC tracker following it. We do not have a G&CNet and cannot directly apply MonoRace's training-time FOV penalty. However, the insight translates: we should add perception-aware costs to our trajectory optimization so that the offline-planned path keeps gates in the camera FOV during aggressive bank angles. The paper arXiv:2512.20475 (already in our research) addresses exactly this for polynomial trajectory planning.

**For our implementation:** The most direct takeaway is to add a FOV visibility cost term to `trajectory_optimizer.py` and/or `racing_line.py` that penalizes trajectory segments where the predicted drone attitude would point the camera away from upcoming gates. The camera is forward-pitched ~45°, so fast forward flight naturally keeps gates in view — the problem arises in tight turns where bank angles can exceed 60°.

### EKF Design

MonoRace's 16-state EKF is very close to our existing `estimation/ekf.py` (15-state). Key differences:
1. **IMU saturation fallback with model-based prediction** — we should implement this. Our EKF currently trusts IMU readings unconditionally. At >10g maneuvers, this will degrade.
2. **Distance-dependent measurement noise model** (σ_pos² = 0.02·d_gate²/(N_c²·N_g)) — our current fixed noise model is suboptimal. Far gates and few corner detections should get higher noise variance.
3. **Multi-gate PnP fusion** — combining corners from multiple visible gates in a single solve improves heading accuracy by ~2°. Our `gate_pnp.py` should be checked for this.

### Control Architecture

We use a geometric tracker (SE(3) Lee et al.) rather than a G&CNet. MonoRace demonstrates that end-to-end neural control can surpass human performance, but it requires extensive RL training infrastructure we do not currently have. The G&CNet approach is aspirational for our system but not immediately implementable. Stick with the geometric tracker for now; focus improvements on trajectory quality and EKF robustness.

### Adaptive Cropping

Our visual pipeline (if we add one) should use EKF-predicted gate positions to guide adaptive cropping rather than fixed-region processing. This is particularly important for the AI Grand Prix competition if we need to run perception on constrained hardware.

### Domain Randomization for Sim-to-Real

MonoRace achieves zero-shot sim-to-real transfer through 50+ randomized parameters at 30–55% variation. Our PyBullet simulator should be stress-tested similarly. The failure modes MonoRace identified (IMU saturation, camera interference) should be explicitly simulated.

---

## Actionable Takeaways

**High priority (directly applicable to our system):**

1. **Add IMU saturation detection and model-based fallback to `estimation/ekf.py`.** Use a simple aerodynamic model to predict expected acceleration. When measured vs. predicted diverges by >threshold, substitute model prediction and inflate process noise. This is a critical robustness fix for any maneuver above ~5g.

2. **Switch to distance-dependent measurement noise in `estimation/ekf.py`.** Replace fixed R matrix with σ_pos² ∝ d_gate²/(N_c²·N_g). Distant gates with few corner detections contribute less to the state estimate.

3. **Add multi-gate PnP fusion in `estimation/gate_pnp.py`.** When two gates are simultaneously visible, solve PnP jointly rather than independently. The ~2° heading improvement is meaningful for tight gate clearances.

4. **Add perception-aware FOV cost to trajectory optimization.** In `planning/trajectory_optimizer.py` or `planning/racing_line.py`, add a soft penalty for trajectory segments where the predicted attitude (from dynamics) would point the camera away from the next gate (θ_cam > π/3 from gate direction). See arXiv:2512.20475 for geometric formulation.

**Medium priority:**

5. **Implement RANSAC-based corner outlier rejection in gate detection.** Before sending PnP measurements to the EKF, reject corners that are inconsistent with an affine transformation model of the gate's expected appearance.

6. **Calibrate camera extrinsics using flight data IoU optimization.** Rather than relying solely on laboratory calibration, run the offline IoU-based refinement using actual flight recordings to correct vibration-induced misalignment.

7. **Explore a FOV reward term if we add RL components.** If we add any learning-based components, include a penalty for θ_cam > π/3 to next gate.

**Not directly applicable (RL/G&CNet approach):**

8. The G&CNet end-to-end approach would require rebuilding our control stack. Not recommended for current iteration. Our geometric tracker with good trajectory quality is the right path.

---

## Limitations & Caveats

**Decoupled vision and control:** The authors acknowledge that "vision and control remain decoupled, requiring explicit reward shaping." The G&CNet never directly observes raw images — it receives processed state estimates from the EKF. True end-to-end vision-to-motor control is identified as future work. This means MonoRace's robustness still depends critically on the EKF state estimate quality.

**Gate shape specificity:** GateNet is trained specifically on rectangular gate shapes. The system cannot generalize to novel gate geometries without retraining. The AI Grand Prix may use different gate shapes than A2RL.

**No multi-drone collision avoidance:** MonoRace placed third in the multi-drone event due to the complete absence of collision avoidance. In any multi-drone scenario (or our competition if it involves simultaneous racing), this is a hard gap.

**Split-S handling not detailed:** The paper mentions a mandatory split-S maneuver but does not detail how the G&CNet handles this specifically, or whether it required special treatment. Our system has identified the S-turn compound inflation issue (iteration 16) — it is unclear whether MonoRace's approach handles this better or differently.

**Track-specific training:** The G&CNet is trained for a specific track layout. Generalization to novel tracks (as required in the AI Grand Prix where track layout may not be pre-known) would require either (a) re-training for each new track, or (b) a more general policy. The paper does not address zero-shot generalization to unseen tracks.

**Lap time context:** 16.56 seconds over a 76×18×5.4m track with 11 gates. Our current benchmark target is <14s over a different track. Direct comparison is not meaningful without track-normalized metrics. Human world champions ran the same track at slightly higher speeds in non-knockout rounds.

**RL training infrastructure not open-sourced:** The custom simulator, domain randomization parameters, and training code are not released. Reproducing the G&CNet approach from scratch would require significant engineering effort.

---

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Camera FOV | 155° × 115° | Monocular rolling-shutter CMOS |
| Image resolution | 820×616 px | Captured at 90 Hz |
| Cropped resolution | 384×384 px | Adaptive, guided by EKF gate predictions |
| Camera angular offset (θ_cam) | 43°–50° | Forward pitch, network-dependent |
| FOV visibility reward threshold | π/3 (60°) | Penalty if next gate angle > 60° from camera axis |
| IMU accelerometer rate | 1000 Hz | Fused in EKF prediction step |
| IMU gyroscope rate | 2000 Hz | Fused in EKF prediction step |
| EKF state dimension | 16 | pos(3) + vel(3) + quat(4) + acc_bias(3) + gyro_bias(3) |
| EKF measurement noise (position) | σ_pos² = 0.02·d_gate²/(N_c²·N_g) | Distance- and corner-count-dependent |
| EKF outlier rejection gate | \|\|x_pos - x_PnP\|\|² < 16·N_c²·trace(P_pos) | Mahalanobis-based, scales with corner count |
| PnP acceptance range | 2–5 m | Gate distance for full pose update |
| PnP minimum corners | 4 (full), 6 (with attitude) | Fallback to position-only below threshold |
| IMU saturation threshold | ±16g | Hardware accelerometer limit |
| Model fallback activation threshold | 22 m/s² | Norm of (model_acc - measured_acc) |
| G&CNet architecture | 3 layers × 64 neurons, ReLU | MLP, motor commands output |
| G&CNet input dimension | 24 | Gate-relative state + attitude + rates |
| G&CNet update rate | 500 Hz | Runs on STM32H743 flight controller |
| G&CNet latency | 2 ms | Rise time for motor command |
| Domain randomization range | 30–55% | Parameter variation during RL training |
| Multi-gate PnP heading improvement | ~2° | vs. single-gate PnP |
| Frames benefiting from multi-gate | 27% | Fraction of flight frames |
| Gate inner opening | 1.5 m | Competition specification |
| Gate outer frame | 2.7 m | Competition specification |
| Track dimensions | 76×18×5.4 m | A2RL 2025 Grand Challenge |
| Total gates on track | 11 | Including two double-gates, one split-S |
| Best two-lap time | 16.56 s | Grand Challenge result |
| Peak velocity | 28.23 m/s (100 km/h) | Fastest fully onboard autonomous result |
| Drone mass | 966 g | Competition class |
| Success rate (M23, conservative) | 88.4% | Over 43 flights |
| Success rate (M17, aggressive) | 71.4% | 55/77 flights |
| Vision latency | ~17 ms | Camera to EKF update |
| IMU-vision sync latency | 0.5 ms | Timestamping compensation |
