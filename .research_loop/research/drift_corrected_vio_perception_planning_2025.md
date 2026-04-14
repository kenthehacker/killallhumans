# Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing

- **URL**: https://arxiv.org/abs/2512.20475
- **Authors**: Maulana Bisyir Azhari, Donghun Han, Je In You, Sungjun Park, David Hyunchul Shim (KAIST)
- **Year**: 2025 (submitted December 23, 2025)
- **Venue**: arXiv preprint (competition system paper, A2RL × DCL)

---

## Key Contribution

This paper presents a complete autonomous drone racing stack operating under severe sensor constraints — a single monocular camera and a low-quality IMU — that achieves podium finishes in the Abu Dhabi Autonomous Racing League × Drone Champions League (A2RL×DCL) competition. The two headline technical contributions are: (1) a lightweight 6-state drift-correction Kalman filter that fuses monocular VIO output with gate-derived global position measurements to reduce accumulated odometry error; and (2) a perception-aware heading (yaw) planner layered on top of TOGT time-optimal trajectories that keeps upcoming gates in the camera FOV using distance-weighted heading blending, without modifying the underlying position trajectory at all.

The primary relevance here is the second contribution. Unlike the ETH 2026 paper (arXiv:2603.04305), which formulates FOV visibility as a hard or soft constraint inside an NLP that jointly re-optimizes the entire trajectory, the KAIST approach decouples yaw entirely from the position plan and controls heading as a separate, real-time-computed reference signal. This makes it computationally trivial — no re-optimization is needed. The consequence is that gate visibility is maintained not by slowing the trajectory down (as our current `_relax_for_fov()` does) but by rotating the drone's camera toward the gate early, even while the position trajectory is executing at full speed. Gate visibility improvement: **+8.88 percentage points** with zero added race time.

---

## Technical Approach

### 1. VIO Selection

Six open-source VIO algorithms were benchmarked on a custom dataset (TII-RATM). Selection criteria: accuracy, robustness, processing time (<33 ms per frame), CPU usage (<200%), memory footprint (<500 MB). **OpenVINS** was chosen for its robust initialization, SLAM feature persistence, and low CPU footprint.

### 2. Gate Detection Pipeline

- Model: YOLOv8s at 640×640 resolution
- Output: bounding box + 4 corner keypoints with per-corner visibility flags: `[O, cx, cy, w, h, tl, tr, br, bl]`
- Training: 2,000 manually labeled images with augmentation (contrast, brightness, motion blur, Gaussian blur, MixUp, Mosaic)
- mAP: 0.877 (bounding box), 0.971 (keypoints)
- Inference: ~16.1 ms on Jetson Orin NX (TensorRT 8.5.2, FP16)

### 3. PnP-Based Gate Pose Estimation

Given a keypoint detection, `SOLVEPNP_IPPE_SQUARE` (OpenCV) recovers the 6-DOF camera-to-gate transform. Only the **position estimate** is used; the rotation estimate is discarded because VIO orientation is already superior. Gate-to-map association uses the Hungarian algorithm minimizing reprojection error across visible corners.

Detection filtering rules (critical for robustness):
- Distance: reject if `d < 1 m` or `d > 13 m`
- Aspect ratio: reject if skew ratio `a_i > 2` (gate appears too oblique)
- Occlusion: reject if a larger gate `S_j` is within 20 px with area ratio `A(S_j)/A(S_i) > 1.2`

### 4. Drift-Correction Kalman Filter (6-State)

This filter runs **in parallel with, not replacing, VIO**. It estimates additive drift only:

```
x_d = [p_d^T, v_d^T]^T  ∈ ℝ^6    (position drift + velocity drift)
```

- **Prediction**: Constant-velocity model; process noise `σ_p = 0.1`, `σ_v = 0.2`
- **Measurement**: `z_k = p_VIO,k + p_drift,k` where the measurement is the VIO position anchored to the PnP-derived absolute gate position in world frame
- **Measurement noise**: `R = λ_r · diag(σ_rx, σ_ry, σ_rz)`, where `λ_r` is a scalar quality factor from detection confidence
- **Final corrected state**: `X_c = [p_VIO + p_drift, q_VIO, v_VIO, ω_VIO]^T` — position is corrected, orientation/velocity come directly from VIO unchanged

ATE improvement: **1.04 m → 0.56 m (45% reduction)** over 8 competition sequences.

### 5. Trajectory Planning with TOGT

The system uses TOGT (Qin et al. 2024) for time-optimal trajectory generation:

```
min_{p, T}  T
subject to:  p(0) = p_start,  p(T) = p_finish,
             ∃ 0 < t_1 < ... < t_L < T  s.t.  p(t_i) ∈ G_i
```

Gates are modeled as **spatial volumes** (not point waypoints), allowing the optimizer to find the fastest path through any point in the gate opening. The solver uses L-BFGS with a change-of-variable technique. TOGT does **not** natively constrain yaw — the perception-aware layer addresses this independently.

### 6. Perception-Aware Heading Control (Core FOV Contribution)

This is the key innovation for FOV management. Yaw is **completely decoupled** from the position trajectory and planned using distance-weighted blending between current-gate and next-gate headings:

```
λ_i = { 1                              if d_i < d_min
       { 0                              if d_i > d_max
       { (d_max - d_i)/(d_max - d_min)  otherwise

ψ_g = (λ_i · ψ_{g,i} + λ_{i+1} · ψ_{g,i+1}) / (λ_i + λ_{i+1})
ψ_des = λ_g · ψ_g + (1 - λ_g) · ψ_c
```

Where:
- `ψ_{g,i}` = heading angle directly toward gate `i`
- `ψ_c` = heading derived from the position trajectory (velocity direction)
- `λ_g` = proximity-based weight: 1 when close to gate (look at gate), 0 when far (follow trajectory)
- `d_min`, `d_max` = near/far blending thresholds (course-dependent, ~2–8 m range)

For aggressive split-s maneuvers with rapid roll inversions, blending is replaced by **fixed-step incremental yaw** (`ψ_{k+1} = ψ_k ± Δψ_step`) to avoid instability.

Gate visibility results:
- Full FOV (155° H × 115° V): **80.24% vs 71.36%** (baseline TOGT without perception-aware yaw)
- Constrained FOV (120° H × 90° V): **60.19% vs 51.82%**
- Net improvement: **+8.88 percentage points**

### 7. MPC Control

MPC with N=20 steps, 1-second horizon (50 ms per step):

```
min_u  Σ_{k=0..N-1} [ ||p_k - p_ref||²_Qp + ||v_k - v_ref||²_Qv
                      + ||q_k ⊖ q_ref||²_Qq + ||ω_k - ω_ref||²_Qω
                      + ||u_k||²_Ru ]
```

Implemented via acados + qpOASES solver. Commands at 200 Hz over UART using MultiWii Serial Protocol.

---

## Results

| Event | Placement | Key Metrics |
|---|---|---|
| AI Grand Challenge | 3rd | 36.8 s race time, mean 9.54 m/s, peak 12.0 m/s, mean tracking error 0.35 m |
| AI Drag Race (90 m straight) | 2nd | 5.4 s, peak 16.42 m/s (~59 km/h) |
| AI Multi-Drone Race | 2nd | — |

State estimation (8 competition sequences, ADR-Comp dataset):
- OpenVINS baseline ATE: 1.04 m average
- Drift-corrected ATE: 0.56 m average (45% improvement)
- Best single sequence: 1.25 m → 0.49 m

Hardware: ~960 g quad, Jetson Orin NX (16 GB), Foxeer H7 flight controller, Arducam 8MP IMX219 (155° H × 115° V, 30 FPS at 820×626).

---

## Relevance to Our System

### The FOV Bottleneck: What We Have vs. What They Do

Our current `_relax_for_fov()` method in `planning/trajectory_optimizer.py` addresses FOV loss by iteratively **inflating segment times** for high-curvature segments until the geometric FOV penalty drops below a threshold. It runs up to 5 iterations, increasing each high-curvature segment time by 10% per iteration. The measured overhead is approximately **+3.5 seconds on a ~12-second trajectory** — a ~29% race time penalty from a method designed to be a safety net, not a primary planning strategy.

The KAIST paper reveals a fundamentally better approach: **do not slow the trajectory down for FOV; instead, rotate the camera toward the gate earlier using yaw control**.

The key insight is that FOV loss during aggressive maneuvers is primarily a **yaw alignment problem**, not a speed problem. When a drone tilts aggressively to execute a sharp turn, the camera (mounted forward on the body) tilts with it. The TOGT-generated trajectory optimizes position and implicitly generates body attitude (via differential flatness) but does not explicitly control yaw. By commanding yaw to face the upcoming gate rather than to align with the velocity vector, the drone's camera can maintain gate visibility throughout the maneuver — without needing to slow down.

This is architecturally simple to implement:
1. Compute `ψ_g` = bearing angle from drone to next gate center in the world frame
2. Compute `ψ_c` = heading from velocity direction (already available in our trajectory)
3. Blend: `ψ_des = λ_g · ψ_g + (1 - λ_g) · ψ_c` where `λ_g` rises as drone approaches the gate
4. Pass `ψ_des` to the attitude controller as the yaw reference, overriding the trajectory-derived yaw

Our `mpc_tracker.py` already accepts a full pose reference including yaw; this blend can be computed in `racing_line.py` or as a preprocessing step in `race_pipeline.py` before commands are dispatched.

### Why Our Current Approach Is Suboptimal

`_relax_for_fov()` is a post-optimization patch that applies a geometric heuristic (penalize curvature) as a proxy for actual FOV loss. It has two failure modes:

1. **Over-relaxation**: Slowing every high-curvature segment even when yaw could compensate, adding unnecessary time.
2. **Under-relaxation**: Even after slowing down, if the body tilt during the maneuver causes the camera to miss the gate, slowing further doesn't necessarily help — the gate may be out of FOV at the apex of the turn regardless of segment time.

The perception-aware yaw approach avoids both: it does not alter the position trajectory at all, and it directly controls the degree of freedom (yaw) most responsible for gate visibility.

### Complementarity with ETH 2026 Paper

The ETH 2026 paper (arXiv:2603.04305) takes a different angle: it adds FOV as a soft constraint inside the trajectory NLP, modifying the entire trajectory (position + yaw) to maintain perception quality, at a cost of +17% lap time and ~68 seconds of offline planning. For our system, this is the "full solution" — but it is complex to implement.

The KAIST paper's approach is the **lightweight, incremental improvement**: decouple yaw and control it reactively to face gates. It gains +8.88 percentage points of visibility with zero trajectory replanning. These are not mutually exclusive — the optimal system would use perception-aware NLP planning offline (ETH approach) and perception-aware yaw blending as a real-time correction layer (KAIST approach).

**Immediate recommendation**: Implement the KAIST yaw-blending approach first. It eliminates `_relax_for_fov()` overhead and is implementable in ~50 lines. The ETH NLP approach is a future optimization.

---

## Actionable Takeaways

1. **Replace `_relax_for_fov()` with perception-aware yaw blending.** Implement the distance-weighted heading blend (`ψ_des = λ_g · ψ_g + (1 - λ_g) · ψ_c`) as a real-time yaw reference generator in `planning/racing_line.py` or `race_pipeline.py`. This should eliminate the ~3.5s overhead from iterative segment-time inflation. The yaw blend runs at control-loop frequency — no trajectory re-solve needed.

2. **Parameterize `d_min` and `d_max` for our gate sizes.** The blending thresholds control how early the drone starts facing the next gate. Starting point: `d_min = 2 m` (within 2m, full gate-facing), `d_max = 10 m` (beyond 10m, pure velocity heading). Tune empirically in simulation by watching gate visibility metrics per gate.

3. **Handle split-s / aggressive inversion separately.** For segments where yaw changes faster than the drone can track (inverted maneuvers), use fixed-step yaw increments rather than blending, to avoid discontinuity in the yaw reference.

4. **Remove or disable `_relax_for_fov()` after implementing yaw blending.** The iterative relaxation adds 3.5s and is no longer needed. Verify in simulation that gate visibility metrics remain stable without it.

5. **Gate volumes, not gate centers, in TOGT.** The paper confirms that TOGT's `p(t_i) ∈ G_i` formulation should treat gates as volumetric regions. Our `trajectory_optimizer.py` should not force trajectories through gate centers — the optimizer should be free to pick any point inside the gate opening to minimize time. This also reduces the need for tight tracking near gates.

6. **Add explicit finish waypoint.** The TOGT formulation requires `p(T) = p_finish` — a defined point beyond the last gate. Ensure our trajectory optimizer includes a 5–10 m extension past the final gate to a finish point, preventing premature deceleration.

7. **Tighten PnP detection filters.** Review `gate_pnp.py` against their three filters: `d < 1 m` (near rejection), `d > 13 m` (far rejection), `a_i > 2` (oblique rejection). The far-distance cutoff is particularly important — at >13 m, PnP position estimates become noisy enough to degrade EKF corrections rather than improve them.

8. **Use `SOLVEPNP_IPPE_SQUARE`.** If `gate_pnp.py` uses a generic PnP method, switching to `SOLVEPNP_IPPE_SQUARE` is a drop-in improvement for planar square-gate geometry — it gives better position estimates from 4-corner correspondences.

9. **Only use PnP position, not orientation.** The paper explicitly discards the PnP rotation estimate and uses only the position for drift correction, because VIO orientation is already superior. If our EKF updates use the rotation from PnP, this should be removed.

10. **Log per-gate FOV visibility rate during benchmarks.** Add a metric to `scripts/benchmark.py` or the simulation that tracks what fraction of time each gate is visible in the camera FOV. This will reveal which specific gates drive the FOV problem and allow targeted tuning of `d_min`/`d_max`.

---

## Limitations & Caveats

- **d_min / d_max are course-specific.** The KAIST blending thresholds are tuned for the A2RL course (large, well-lit, widely spaced gates). Our VQ1 course may have different gate spacing that requires different transition distances.

- **Yaw tracking bandwidth matters.** The perception-aware yaw approach works only if the drone's yaw controller can track the desired heading fast enough to point the camera toward the gate before the gate is passed. At high speed and short gate-to-gate distances, the commanded yaw rate may exceed actuator limits. If yaw tracking is slow, the gate will be missed even with perfect yaw reference generation.

- **Camera FOV assumed forward-facing.** The analysis assumes the camera mounts along the body x-axis (forward). If our camera mount differs (e.g., tilted downward for FPV style), the yaw-blending approach needs adjustment to account for the mounting angle.

- **Constant-velocity drift model may lag at high speed.** The `σ_v = 0.2` process noise on drift velocity assumes drift evolves slowly. During the most aggressive maneuvers with rapid IMU excitation, this model may be too conservative. This is acceptable because the main EKF handles fast dynamics — the drift filter only needs to capture slow systematic error.

- **VIO reliance on good initialization.** OpenVINS was selected partly for its robust initialization. In our system, if VIO loses tracking during startup or after a crash recovery, the drift filter has no useful measurements to correct. Our existing EKF initialization procedure should be robust to this.

- **No ablation of yaw-blending vs. trajectory slowing.** The paper does not directly compare their yaw-blending approach to an approach that slows the trajectory (like our `_relax_for_fov()`). The 8.88 pp improvement is vs. baseline TOGT with no FOV handling at all. Whether yaw-blending alone is sufficient, or whether some trajectory adjustment is also needed, is not characterized.

- **Split-s handling is ad hoc.** The fixed-step yaw increment for inversion maneuvers is described qualitatively, not quantitatively. The step size `Δψ_step` is not specified. This will require empirical tuning.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|---|---|---|
| Drift KF process noise (position) | `σ_p = 0.1` | Position drift process noise |
| Drift KF process noise (velocity) | `σ_v = 0.2` | Velocity drift process noise |
| Gate detection min distance | 1 m | Below this, reject PnP measurement |
| Gate detection max distance | 13 m | Above this, reject PnP measurement |
| Aspect ratio reject threshold | `a > 2` | Reject highly skewed gate detections |
| Occlusion proximity threshold | 20 px | Gate overlap proximity for occlusion filter |
| Occlusion area ratio threshold | 1.2 | `A(S_j)/A(S_i)` for occlusion rejection |
| MPC horizon steps | N = 20 | Prediction horizon |
| MPC horizon duration | 1.0 s | 50 ms per step |
| Control rate | 200 Hz | Via UART to flight controller |
| Camera FOV (full, Arducam IMX219) | 155° H × 115° V | Hardware parameter |
| Gate visibility (perception-aware, full FOV) | 80.24% | vs. 71.36% baseline TOGT |
| Gate visibility (perception-aware, constrained FOV) | 60.19% | vs. 51.82% baseline TOGT |
| Net visibility improvement | +8.88 pp | Zero race time cost |
| VIO drift reduction (ATE) | 1.04 m → 0.56 m | 45% improvement over 8 sequences |
| Race mean tracking error | 0.35 m | AI Grand Challenge |
| Drag race tracking error | 0.20 m | AI Drag Race |
| Peak race speed | 16.42 m/s (~59 km/h) | AI Drag Race |
| Drone mass | ~960 g | With 6S 1400 mAh LiPo |
| YOLOv8s inference time | ~16.1 ms | Jetson Orin NX, TensorRT FP16 |
| YOLOv8s keypoint mAP | 0.971 | On custom gate dataset |
