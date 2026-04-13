# Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing

- **URL**: https://arxiv.org/abs/2512.20475
- **Authors**: Maulana Bisyir Azhari, Donghun Han, Je In You, Sungjun Park, David Hyunchul Shim (KAIST)
- **Year**: 2025 (submitted December 23, 2025)
- **Competition**: Abu Dhabi Autonomous Racing League (A2RL) × Drone Champions League (DCL)

---

## Key Contribution

This paper presents a complete autonomous drone racing stack that operates under severe sensor constraints — a single monocular camera and a low-quality IMU — and still achieves competitive performance in the A2RL×DCL events. The two headline contributions are:

1. **Drift-Corrected VIO**: An EKF-based Kalman filter that fuses monocular VIO (OpenVINS) position output with gate-derived global position measurements from a YOLOv8-Pose detector. The correction layer tracks only 6-state drift (position + velocity) rather than the full navigation state, making it lightweight and stable.

2. **Perception-Aware Heading Control**: A decoupled yaw planner layered on top of TOGT trajectories that continuously blends heading toward visible gates as a function of distance. This keeps gates in the camera's field of view for more reliable detection and correction, yielding an 8.88% improvement in gate visibility metrics.

The system achieved **3rd place (AI Grand Challenge), 2nd place (AI Drag Race at 59 km/h), and 2nd place (AI Multi-Drone Race)**, making it the strongest monocular-only result in the competition and approximately twice the speed of prior monocular racing systems.

---

## Technical Approach

### 1. VIO Selection and Benchmarking

The team evaluated six open-source VIO algorithms (ROVIO, LARVIO, OpenVINS, SVO, VINS-Mono, DM-VIO) against a custom dataset (TII-RATM). Selection criteria included accuracy, robustness, processing time (<33 ms), CPU usage (<200%), and memory (<500 MB). **OpenVINS** was selected for its robust initialization, SLAM feature persistence, and low CPU footprint.

### 2. Gate Detection Pipeline (YOLOv8-Pose)

- Model: YOLOv8s at 640×640 resolution
- Output: bounding box + 4 corner keypoints with visibility flags `[O, cx, cy, w, h, tl, tr, br, bl]`
- Training: 2,000 manually labeled images with augmentation (contrast, brightness, motion blur, Gaussian blur, MixUp, Mosaic)
- mAP: 0.877 (bounding box), 0.971 (keypoints)
- Inference: ~16.1 ms on Jetson Orin NX (TensorRT 8.5.2, FP16)

### 3. PnP-Based Gate Pose Estimation

Given a detection, OpenCV `SOLVEPNP_IPPE_SQUARE` recovers the 6-DOF camera-to-gate transform. Critically, **only the position estimate is used**; the rotation estimate is discarded because VIO's orientation is already superior. The recovered gate center position in world frame serves as a measurement for the drift-correction filter.

**Detection filtering rules** (important for robustness):
- Distance: reject if `d < 1 m` or `d > 13 m`
- Aspect ratio: reject if skew ratio `a_i > 2` (gate appears too oblique)
- Occlusion: reject detection `S_i` if a larger gate `S_j` is within 20 px and has area ratio `A(S_j)/A(S_i) > 1.2`

Gate-to-map association uses the **Hungarian algorithm** minimizing reprojection error across all visible corners.

### 4. Drift Correction Kalman Filter

This is a 6-state Kalman filter running in parallel with (not replacing) VIO:

```
x_d = [p_d^T, v_d^T]^T ∈ ℝ^6   (position drift + velocity drift)
```

- **Prediction**: Constant-velocity model; process noise `σ_p = 0.1`, `σ_v = 0.2`
- **Measurement**: `z_k = p_VIO,k + p_drift,k` where the measurement is the raw VIO position anchored to the PnP-derived absolute gate position
- **Measurement noise**: `R = λ_r · diag(σ_rx, σ_ry, σ_rz)`, where `λ_r` is a scalar quality factor derived from detection confidence
- **Final corrected state**: `X_c = [p_VIO + p_drift, q_VIO, v_VIO, ω_VIO]^T` — position is corrected, orientation/velocity come directly from VIO

Result: average translational error reduced from **1.04 m to 0.56 m (45% reduction)** over 8 competition sequences.

### 5. Trajectory Planning with TOGT

The system uses the **TOGT planner (Qin et al. 2024)** for time-optimal trajectory generation:

```
min_{p, T}  T
subject to:  p(0) = p_start,  p(T) = p_finish,
             ∃ 0 < t_1 < ... < t_L < T  such that  p(t_i) ∈ G_i
```

Gates are modeled as **spatial volumes** rather than point waypoints, allowing the optimizer to find the fastest path through the gate opening. The solver uses a **change-of-variable technique** with **L-BFGS gradient descent** and a 1-second planning horizon for re-optimization.

The TOGT formulation does not inherently constrain yaw, which is where the perception-aware layer is added.

### 6. Perception-Aware Heading Control

Yaw is decoupled from position trajectory and planned separately using a **distance-weighted blending** between the current gate heading and the next gate heading:

```
λ_i = { 1                              if d_i < d_min
       { 0                              if d_i > d_max
       { (d_max - d_i)/(d_max - d_min)  otherwise

ψ_g = (λ_i · ψ_g,i + λ_{i+1} · ψ_g,{i+1}) / (λ_i + λ_{i+1})
ψ_des = λ_g · ψ_g + (1 - λ_g) · ψ_c
```

Where `ψ_g,i` is the heading toward gate `i`, `ψ_c` is the current heading from the position trajectory, and `λ_g` is a proximity-based weight. This ensures the drone faces the next gate early, keeping it in the camera FOV for detection and correction.

For aggressive split-s maneuvers, the planner uses a **fixed-step incremental yaw** (`ψ_{k+1} = ψ_k ± Δψ_step`) rather than blending, to avoid gimbal-lock-like instability during rapid roll inversions.

Gate visibility improvement from perception-aware planning:
- Full FOV (155°×115°): **80.24% vs 71.36%** (baseline TOGT)
- Constrained FOV (120°×90°): **60.19% vs 51.82%**

### 7. MPC Control

Model Predictive Controller with `N = 20` steps, 1-second horizon (50 ms per step):

```
min_u  Σ_{k=0..N-1} [ ||p_k - p_ref||²_Qp + ||v_k - v_ref||²_Qv
                      + ||q_k ⊖ q_ref||²_Qq + ||ω_k - ω_ref||²_Qω
                      + ||u_k||²_Ru ]
```

Implemented via **acados** framework with **qpOASES** solver. Commands dispatched to flight controller at **200 Hz** over UART using MultiWii Serial Protocol.

---

## Results

| Event | Placement | Key Metrics |
|---|---|---|
| AI Grand Challenge | 3rd | 36.8 s race time, mean 9.54 m/s, peak 12.0 m/s, mean tracking error 0.35 m |
| AI Drag Race (90 m) | 2nd | 5.4 s, peak 16.42 m/s (~59 km/h) |
| AI Multi-Drone Race | 2nd | — |

**State estimation** (8 competition sequences, ADR-Comp dataset):
- OpenVINS baseline ATE: 1.04 m average
- Drift-corrected ATE: 0.56 m average (45% improvement)
- Best single sequence: 1.25 m → 0.49 m

**Gate visibility** (perception-aware vs. baseline TOGT): 8.88 percentage-point improvement.

**Hardware**: ~960 g quad, Jetson Orin NX (16 GB), Foxeer H7 flight controller, Arducam 8MP IMX219 (155°H × 115°V, 30 FPS at 820×626).

---

## Relevance to Our System

Our system shares the same TOGT backbone and a similar EKF-based estimation stack. The paper is directly applicable in several ways:

**1. EKF drift correction pattern**: Our `ekf.py` and `gate_pnp.py` already implement the building blocks for drift correction. The paper provides a clean reference for the 6-state additive drift model (`x_d = [p_d, v_d]`), which is simpler and more stable than fusing absolute position directly into the main EKF state. This separation of concerns (VIO handles fast dynamics, drift filter handles slow global error) is worth replicating exactly.

**2. Gate waypoint parameterization**: The paper confirms that modeling gates as **spatial volumes** (not point waypoints) is the correct approach for TOGT. Our `trajectory_optimizer.py` should treat gate constraints as regions, not hard waypoints through gate centers — this directly addresses our current focus on gate-region parameterization.

**3. Trajectory extension past final gate**: The TOGT formulation sets `p(T) = p_finish` explicitly, meaning the trajectory must extend to a defined finish point beyond the last gate. The paper handles this as a standard `p_finish` constraint in the optimization. For our system, we should ensure the trajectory continues through and past the final gate to a finish waypoint rather than terminating at the gate center.

**4. Perception-aware yaw**: Our `trajectory_optimizer.py` currently couples yaw to position trajectory heading. The paper's decoupled yaw blending (`ψ_des = λ_g · ψ_g + (1-λ_g) · ψ_c`) is directly implementable in `mpc_tracker.py` or `racing_line.py` as a post-processing step. This is low-risk and could improve EKF correction frequency.

**5. Detection filtering thresholds**: The `d < 1 m` / `d > 13 m` distance filter and `a > 2` aspect-ratio filter in `gate_pnp.py` are directly usable parameter values if our current filters are not tuned.

**6. MPC horizon and rate**: Their `N=20`, 1-second horizon at 200 Hz matches what we should target. If our `mpc_tracker.py` loop Hz is below 100 Hz, the acados/qpOASES framework choice is worth examining.

---

## Actionable Takeaways

1. **Separate drift correction from main EKF**: Implement a lightweight 6-state Kalman filter (`[p_drift, v_drift]`) that runs alongside the main EKF and corrects position output using PnP gate measurements. Keep VIO orientation untouched.

2. **Gate volumes, not gate centers**: In `trajectory_optimizer.py`, replace point gate waypoints with volumetric (box or cylinder) constraints. The TOGT formulation's `p(t_i) ∈ G_i` condition means the solver should be free to pick any point inside the gate opening. This reduces required precision and allows faster traversal.

3. **Add a finish waypoint**: Ensure the optimized trajectory has an explicit `p_finish` point set 5–10 m beyond the last gate, aligned with the course exit direction. This prevents the trajectory from ending abruptly at the gate, which can cause deceleration at the wrong time.

4. **Implement decoupled perception-aware yaw**: Add the distance-weighted heading blend from section 6 above to `racing_line.py` or `mpc_tracker.py`. Parameters `d_min` and `d_max` are tunable (likely 2–5 m and 8–15 m based on gate sizes and FOV).

5. **Tighten PnP filtering**: Review `gate_pnp.py` against their three filters (distance, aspect ratio, occlusion). The `d > 13 m` upper bound is particularly useful; detections at range are noisy and should not update the EKF.

6. **Use SOLVEPNP_IPPE_SQUARE**: If our `gate_pnp.py` uses a generic PnP solver, switching to `SOLVEPNP_IPPE_SQUARE` is a direct improvement for planar square-gate geometry.

7. **Discard rotation from PnP in EKF updates**: Only feed position from PnP into EKF correction steps — orientation from VIO/IMU integration is more reliable than from a single gate detection.

---

## Limitations & Caveats

- **Competition-specific tuning**: The 1 m / 13 m detection range thresholds and the `d_min`/`d_max` blending parameters are tuned for the A2RL course layout, which uses large visible gates. Our course gates may require different values.

- **Drift model assumes slow drift**: The constant-velocity drift model (`σ_v = 0.2`) assumes drift evolves slowly. In aggressive high-speed maneuvers with rapid IMU excitation, this model may lag. Our EKF already handles the fast dynamics so the additive correction should remain stable.

- **No full TOGT re-optimization at runtime**: The paper appears to use TOGT for pre-planned trajectories with perception-aware yaw as a real-time overlay, not full re-planning in flight. True online TOGT re-optimization would be more powerful but computationally demanding.

- **Monocular-only gate detection**: PnP position accuracy degrades with distance and when gate corners are partially occluded. Their 45% ATE improvement is impressive but the absolute residual error (0.56 m) is still non-trivial; this is after competition-specific training data collection.

- **Hardware dependency**: Their system uses an Arducam with a 155° FOV. Systems with narrower FOVs will see worse gate visibility even with the perception-aware planner, as their own constrained-FOV comparison (60% vs 80%) shows.

- **No ablation of gate volume vs. gate point**: The paper does not explicitly isolate the benefit of volumetric gate constraints vs. point waypoints in TOGT; this benefit comes from the TOGT paper itself (Qin 2024) which they build upon.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|---|---|---|
| Drift KF process noise (position) | `σ_p = 0.1` | Drift position process noise |
| Drift KF process noise (velocity) | `σ_v = 0.2` | Drift velocity process noise |
| Gate detection min distance | 1 m | Below this, reject PnP measurement |
| Gate detection max distance | 13 m | Above this, reject PnP measurement |
| Aspect ratio reject threshold | `a > 2` | Reject highly skewed gate detections |
| Occlusion proximity threshold | 20 px | For occlusion filtering |
| Occlusion area ratio threshold | 1.2 | `A(S_j)/A(S_i)` for occlusion |
| MPC horizon steps | N = 20 | |
| MPC horizon duration | 1 second | 50 ms per step |
| Control rate | 200 Hz | Via UART to flight controller |
| YOLOv8s input resolution | 640 × 640 | |
| YOLOv8s inference time | ~16.1 ms | On Jetson Orin NX, TensorRT FP16 |
| Camera FOV (full) | 155° H × 115° V | Arducam IMX219 |
| Gate visibility improvement | +8.88 pp | Perception-aware vs. baseline TOGT |
| VIO drift reduction (ATE) | 1.04 m → 0.56 m (45%) | 8-sequence average |
| Race mean tracking error | 0.35 m | Grand Challenge event |
| Drag race tracking error | 0.20 m | Drag race event |
| Peak race speed | 16.42 m/s (~59 km/h) | Drag race event |
| Drone mass | ~960 g | With 6S 1400 mAh LiPo |
