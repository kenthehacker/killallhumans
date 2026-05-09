# MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI

- **URL:** https://arxiv.org/abs/2601.15222
- **Authors:** Stavrow A. Bahnam, Robin Ferede, Till M. Blaha, Anton E. Lang, Erin Lucassen, Quentin Missinne, Aderik E.C. Verraest, Christophe De Wagter, Guido C.H.E. de Croon
- **Year:** 2026 (submitted January 21, 2026)
- **Venue:** arXiv:2601.15222 (cs.RO); competition results from the 2025 Abu Dhabi Autonomous Drone Racing (A2RL) Championship

---

## Key Contribution

MonoRace is the first autonomous drone racing system to defeat human FPV world champions in a direct knockout tournament using only monocular vision and onboard computation — no external motion capture, no precise track surveying, no ground-truth pose infrastructure. The core claim is champion-level performance at up to 100 km/h achieved from a single rolling-shutter camera plus IMU, running entirely on a Jetson Orin NX companion computer and an STM32H743 flight controller.

The result is significant because every prior champion-level autonomous system (Swift, Agile Flight, CPC) depended on external state estimation or highly precise environment maps. MonoRace closes that gap by demonstrating that the robustness bottleneck is addressable at the system-engineering level through three targeted mechanisms: self-supervised camera calibration, model-based IMU saturation handling, and a perception-robust RL policy.

---

## Technical Approach

### Perception Pipeline

**Gate segmentation (GateNet):** A U-Net with five multi-scale output maps, trained on 3500 synthetic composites plus 500 real images. Input is 384×384 adaptive crops at 90 Hz; the crop follows the expected gate position so the network always sees a properly scaled gate. Channel scaling factor f=4. Augmentation: affine transforms, HSV shifts, motion blur kernels 5–15 px, thermal noise simulation.

**Corner detection (QuAdGate):** Line Segment Detector extracts edges from the segmentation mask, intersections produce corner candidates, a 4-pixel handcrafted descriptor matches candidates to geometric priors, RANSAC (threshold 5.0 px) rejects outliers. Key innovation: when a single gate offers ambiguous heading, corners from *multiple coplanar gates* are solved simultaneously in one PnP optimization, improving heading accuracy by approximately 2°.

**State estimation (EKF):** 16-state EKF (position, velocity, quaternion, accel bias, gyro bias) fusing vision at variable rate with IMU at up to 2000 Hz. Measurement noise scales dynamically:
```
σ_pos²  = 0.02 × d_gate² / (N_c² × N_g)
σ_quat² = 0.01 × d_gate² / (N_c² × N_g)
```
Closer gates with more visible corners produce tighter measurement noise, naturally down-weighting far or partially visible detections.

**Temporal sync:** Camera has 17 ms hardware delay; IMU has 0.5 ms. The system time-stamps each image at capture, then integrates subsequent IMU data to propagate the state forward. This alone reduced SITL trajectory error from 0.289 m to 0.103 m.

### IMU Saturation Handling

Aggressive 7g maneuvers saturate the ±16g accelerometers. The fix: maintain a drone dynamics model in parallel with the EKF. When `||a_model - a_IMU||₂ > 22 m/s²`, the system substitutes model-predicted acceleration for the corrupted IMU reading and inflates Kalman gain uncertainty. Gyroscope threshold: 1700°/s. Effect: success rate on the affected trajectory segment improved from 50% to 100%.

### Camera Calibration Recovery

MIPI cable electromagnetic interference corrupted 8–75% of pixels in some flights. The defense is layered: GateNet tolerates partial corruption, RANSAC discards bad corners, and the EKF innovation gate rejects large residuals. The system flew 25 m+ with zero valid images during one competition run.

For the camera extrinsics, a self-supervised Bayesian optimization procedure uses IoU between re-projected gate masks and actual segmentation output as the objective — no external reference needed. 40 iterations corrects a 2° initialization error to under 0.5°; real-flight validation improved mask IoU from 0.64 to 0.78.

### Control: Guidance & Control Network (G&CNet)

A 3-layer fully connected network (64 neurons per layer, ReLU) runs at 500 Hz and outputs four direct motor commands, bypassing any inner-loop PID or geometric controller. Input: 24-element observation vector (position, velocity, Euler angles in current gate frame; angular rates; relative position and yaw to next gate).

Trained with PPO (Stable-Baselines3) in simulation with heavy domain randomization: all physical parameters randomized 30–55% per policy iteration. The aerodynamic model includes quadratic drag `v|v|`, angle-of-attack and advance-ratio dependencies, gyroscopic coupling, and a motor response model with τ = 0.025 s.

**Reward shaping:** Progress (capped at v_max × Δt), gate-passing bonus λ_gate, penalties for angular rate, gate offset at crossing, perception angle, motor jerk, low-thrust commands, and collisions. Entropy coefficient swept 0–0.005 across policy variants.

**Policy family trained:**
- M16 (aggressive): Gate size reduced to 0.45 m in training, faster but less robust
- M23 (conservative): 88.4% success rate across 43 flights; used for qualification rounds
- M16 used in the final: 16.56 second lap, peak 28.23 m/s (100 km/h)

---

## Results

- Won the 2025 A2RL Grand Challenge: 16.56 s lap, beat three FPV world champions in direct knockout
- Won the Drag Race event (fastest AI, fastest overall)
- Third place in Multi-Drone Race (no collision avoidance)
- Six distinct policies trained; success rates spanned 55/77 flights (M16) to 88.4% (M23)
- Temporal synchronization alone: 0.289 m → 0.103 m SITL trajectory error
- Camera calibration: IoU 0.64 → 0.78; extrinsic error < 0.5° after 40 BO steps
- IMU saturation fix: 50% → 100% success on critical maneuver segments

---

## Relevance to Our System (Competition Preparation Focus)

Our system is in iteration 48 of 50, targeting sub-14 s race times with < 0.25 m average tracking error. MonoRace is the closest published competitor to our target operating regime and its lessons map directly onto our architecture.

**EKF design:** MonoRace's dynamic noise scaling (`σ ∝ d_gate / (N_c × N_g)`) is more principled than fixed-noise EKF. Our `ekf.py` uses static `EKFConfig` parameters; at competition, gate distances and occlusion will vary. Borrowing this distance- and corner-count-weighted measurement noise could reduce EKF uncertainty at longer ranges.

**Temporal compensation:** Our `state_predictor.py` does forward prediction for latency compensation. MonoRace quantifies the gain precisely: 17 ms camera delay caused 0.186 m → 0.103 m error reduction when properly compensated. If our sim runs with lower artificial latency than the real competition hardware, actual performance could degrade significantly.

**IMU saturation:** Our pipeline does not appear to handle accelerometer saturation during aggressive maneuvers. The 50% → 100% success improvement from MonoRace's model-based substitution is critical for gates with tight entry angles. Our `ekf.py` should detect large innovation residuals and inflate uncertainty rather than integrating corrupted measurements.

**Policy diversity / robustness dial:** MonoRace trained six policy variants covering the speed-robustness tradeoff and selected M23 for qualifying and M16 for finals. Our ILC iterations implicitly do something similar by tuning alpha, but we lack an explicit per-condition policy selection mechanism. For a real competition, having a "safe" and "fast" trajectory variant ready is directly actionable.

**Gate PnP for drift correction:** Our `gate_pnp.py` already does PnP-based drift correction. MonoRace's multi-gate coplanar PnP is a direct improvement worth considering — solving adjacent gates jointly in one PnP is feasible when gates share a known geometric relationship (which they do in our `race_01.json`).

**Calibration robustness:** Our visual pipeline currently assumes fixed camera extrinsics. MonoRace's Bayesian optimization over gate reprojection IoU could serve as a pre-competition calibration step, especially if our drone is transport-damaged or has component swaps.

---

## Actionable Takeaways (Numbered)

1. **Distance-weighted EKF measurement noise.** In `ekf.py`, scale position and orientation measurement covariances by `d_gate² / (N_corners² × N_gates_visible)` rather than fixed values. Reduces EKF divergence at long-range gate approaches.

2. **IMU saturation detection and model substitution.** Add a dynamics-model-based acceleration prediction in `ekf.py`. When `||a_model - a_meas|| > threshold` (e.g., 22 m/s²), substitute model prediction and inflate process noise. Critical for the split-S and tight-apex gates in our track.

3. **Latency budget audit.** Measure the true end-to-end latency of our pipeline (camera → EKF → MPC output) and verify `state_predictor.py` compensates by the correct amount. MonoRace's 17 ms camera delay, if uncompensated, caused 0.186 m additional SITL error — equivalent to most of our current tracking error budget.

4. **Dual-policy competition strategy.** Create a "fast" trajectory variant (current ILC-optimized) and a "robust" variant with +10% gate margins and reduced speed at high-curvature gates. Use the robust variant until gate 5, switch to fast for the final sprint. MonoRace used this exact approach (M23 for qualifiers, M16 for finals).

5. **Multi-gate joint PnP.** In `gate_pnp.py`, when two or more gates are simultaneously visible, solve a joint PnP using all visible corners with the known inter-gate geometry as an additional constraint. Expected heading improvement: ~2°, which translates directly into reduced cross-track error on gate approach.

6. **Adaptive crop or region-of-interest for gate detection.** MonoRace reports a 36% corner detection improvement from adaptive cropping vs. fixed-size processing. If our gate detector uses fixed-size inputs, adding a predicted gate position prior to crop the ROI before inference will improve detection reliability at high speeds.

7. **EKF innovation gating for camera interference.** Add a Mahalanobis-distance check on gate detection residuals in the EKF update step. Reject measurements beyond a threshold (e.g., 3σ) rather than applying them directly. This provides hardware-fault tolerance at competition.

8. **Bayesian optimization pre-competition calibration.** Run 40–50 iterations of BO over camera extrinsic parameters before each competition day using gate reprojection IoU as the objective. No external calibration target needed — only onboard flight data from warmup laps.

9. **Domain randomization coverage for competition track.** MonoRace randomized physical parameters 30–55%. Our sim uses a single fixed model. Adding ±20% randomization on drag coefficients, motor time constants, and mass during benchmark sweeps would expose controller brittleness to slight hardware variation on competition day.

10. **ILC alpha selection as function of gate curvature.** MonoRace's per-policy tuning maps loosely onto our per-section alpha rebalancing (iteration 47). The insight is that the speed-robustness tradeoff should be selected per maneuver type, not globally. For our next iteration: assign higher correction authority (alpha) in straight sections, lower alpha near apexes to avoid overshooting the correction.

---

## Limitations & Caveats

**Not applicable to our current sim stage.** MonoRace's contributions are primarily in the perception and robustness layers. Our pipeline uses the PyBullet adapter with ground-truth state, so GateNet, QuAdGate, and camera calibration are not directly transferable until we move to hardware deployment. The control and estimation insights apply now; the vision pipeline insights apply at integration.

**Policy training from scratch.** The G&CNet approach (direct motor commands from RL) is architecturally different from our geometric SE(3) tracker + MPC. Adopting it wholesale would require rewriting the control layer. The relevant takeaway is the *design principles* (domain randomization coverage, reward shaping for perception angle) rather than the network itself.

**No ILC in MonoRace.** The system uses offline-trained RL with policy selection at competition, not iterative learning control. It does not address our specific use case of correcting systematic errors on a fixed track with repeated laps. Our ILC approach is more directly aligned with CPC (Foehn 2021) and constrained ILC literature.

**Gate geometry specificity.** The multi-gate PnP and corner detection pipeline is tuned for rectangular 1.5 m inner / 2.7 m outer gates. Our gate geometry may differ. Verify `gate_pnp.py` gate dimensions match competition specs before applying multi-gate improvements.

**Collision avoidance gap.** MonoRace finished third in the multi-drone event due to no collision avoidance. If our competition format includes simultaneous multi-drone heats, this is a gap that MonoRace explicitly does not solve.

**Single-track generalization unknown.** All results are for the specific Abu Dhabi track (11 gates, 2 laps, includes split-S). Success rate on unseen tracks during the competition warm-up period is not reported.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Camera resolution | 820 × 616 pixels | Raw capture |
| Processing resolution | 384 × 384 crops | Adaptive ROI for GateNet |
| Camera capture rate | 90 Hz | Rolling-shutter CMOS |
| Camera latency | 17 ms | Hardware delay to be compensated |
| IMU accelerometer rate | 1000 Hz | Integration rate |
| IMU gyroscope rate | 2000 Hz | Integration rate |
| IMU accel range | ±16g | Saturation threshold for 7g maneuvers |
| IMU saturation detection threshold | 22 m/s² | `\|\|a_model - a_IMU\|\|₂` |
| Gyroscope saturation threshold | 1700°/s | Angular velocity |
| Control frequency | 500 Hz | G&CNet motor command output |
| Motor time constant | τ = 0.025 s | Step response model |
| Drag coefficients | k_x = k_y = 5.37×10⁻⁵, k_x2 = 4.10×10⁻³ | Quadratic drag terms |
| Drone mass | 966 g | Total with battery |
| Gate inner dimension | 1.5 m | Competition track |
| Gate outer dimension | 2.7 m | Competition track |
| Track layout | 11 gates, 2 laps | A2RL Abu Dhabi 2025 |
| RANSAC corner threshold | 5.0 px | QuAdGate outlier rejection |
| BO calibration steps | 40 | Camera extrinsic recovery |
| EKF state dimension | 16 | pos, vel, quat, biases |
| G&CNet architecture | 3 × 64 FC, ReLU | Direct motor command output |
| G&CNet observation dim | 24 | Gate-frame pose + angular rates |
| Domain randomization range | 30–55% | Physical parameter variation per episode |
| Policy entropy coefficient | 0–0.005 | PPO training, varies by policy variant |
| M16 lap time | 16.56 s | Fastest policy, competition finals |
| M23 success rate | 88.4% | Conservative policy, 43 flights |
| Peak speed achieved | 28.23 m/s (100 km/h) | Grand Challenge final |
| Temporal sync improvement | 0.289 m → 0.103 m | SITL trajectory error |
| Calibration IoU improvement | 0.64 → 0.78 | Camera extrinsic optimization |
| Heading accuracy after multi-gate PnP | ~2° improvement | vs. single-gate PnP |
