# On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas

- **URL**: https://arxiv.org/abs/2510.13644
- **Authors**: Michael Bosello, Flavio Pinzarrone, Sara Kiade, Davide Aguiari, and 9 others (full author list not retrieved)
- **Year**: 2025 (submitted Oct 15, 2025; revised Jan 30, 2026)
- **Venue**: IEEE Robotics and Automation Letters, Vol. 11, No. 3, March 2026

---

## Key Contribution

This paper closes the gap between laboratory-controlled autonomous drone racing and real-world deployment. Prior work (e.g., Swift/Kaufmann 2023, Champion-level pilots vs. RL) achieved superhuman lap times but only in heavily instrumented arenas with external positioning systems (MoCap/UWB/RTK) providing ground truth. The key novelty here is demonstrating **pro-level autonomous racing in uninstrumented environments** — without any external ground-truth infrastructure — using onboard sensing alone.

The system achieves performance parity with a professional FPV champion pilot (MCK) across both instrumented and uninstrumented tracks:
- **Instrumented**: Autonomous 4.65s avg lap vs. MCK 4.71s avg lap (autonomous wins), with 0 crashes vs. 5 crashes (human).
- **Uninstrumented**: Autonomous 6.02s avg vs. MCK 5.80s avg (human slightly faster), but 4 crashes vs. 2 crashes.

This is significant because it removes the dependency on expensive arena infrastructure, making real-world deployment viable.

---

## Technical Approach (focus on dual-waypoint gate traversal)

### System Overview

The pipeline follows the classical structure: vision-based gate detection → pose estimation (PnP) → dual-stage state estimation → trajectory optimization → MPC → Betaflight PID attitude control. All compute runs onboard on an NVIDIA Orin NX.

### Dual-Waypoint Gate Traversal

The core gate-passing strategy uses **exactly two waypoints per gate**, both positioned at the gate center in the y- and z-axes, with offsets along the gate's local x-axis (the axis perpendicular to the gate plane):

- **Entry waypoint**: gate_center + (-0.4m along gate x-axis) → i.e., 0.4m before the gate
- **Exit waypoint**: gate_center + (+0.4m along gate x-axis) → i.e., 0.4m after the gate

These waypoints are defined in the gate's local frame, then **transformed to world frame using the known gate yaw angle**. This is a critical detail: the offset is along the gate's heading direction, not the world x-axis.

For Split-S maneuvers (where the drone flips and reverses direction), the offsets are larger: **±1.25m** instead of ±0.4m, to accommodate the higher speed and the geometric requirement of executing a half-loop through the gate.

The rationale is that placing waypoints symmetrically before and after each gate:
1. Forces the trajectory optimizer to plan a path that actually passes through the gate opening (not around it).
2. Provides velocity direction control — the drone approaches and departs along the gate's axis, which is the correct traversal heading.
3. Avoids the optimizer taking aggressive shortcuts that might clip gate edges.

### Trajectory Optimizer

The trajectory optimizer used is described as **open-source and time-optimal**, minimizing time to specified waypoints while incorporating full rigid-body drone dynamics and actuator constraints. Key parameters:

- **Thrust-to-weight ratio**: Conservative value of **3.8** used for feasibility (note: actual hardware achieves ~7 at full battery — they deliberately cap it for trajectory safety margin).
- **Linear aerodynamic drag**: Explicitly excluded from the optimization model (simplified dynamics).
- The optimizer respects per-rotor thrust bounds derived from the 3.8 TWR cap.

This optimizer is distinct from MINCO/TOGT — it appears to be a separate open-source time-optimal planner (likely RAPTOR or similar), though the exact library name is not specified in the HTML content retrieved.

### Controller: MPC + Betaflight PID

The tracking controller uses **Model Predictive Control (MPC)** based on an open-source perception-aware MPC framework. Key configuration choices:

- **Perception-aware objectives disabled** — the perception features of the base framework are turned off; only trajectory tracking and robustness cost weights are active.
- **Cost weights carefully tuned** for racing (aggressive tracking, not smooth hovering).
- **Command interface**: MPC outputs Collective Thrust and Body Rates (CTBR).
- **Inner loop**: CTBR commands sent to **Betaflight PID** running on the flight controller, which maps them to PWM rotor signals with internal gyro feedback at high rate.

### Latency Compensation

A **state predictor** compensates for three distinct latency sources:
1. MPC computation time
2. Flight controller communication delay
3. Motor actuation delay

The predictor forward-integrates the state estimate over the combined delay before feeding it to the MPC, ensuring the control action is planned for the predicted future state rather than the stale observed state.

### Tilt and Thrust Constraints

The trajectory optimizer enforces:
- **Thrust constraints**: Per-rotor minimum and maximum thrust, derived from the 3.8 TWR (conservative) cap.
- **Tilt angle limits**: Maximum pitch/roll angles are constrained (exact numerical values not specified in retrieved content).
- **Angular rate limits**: Body rate bounds enforced through the full rigid-body dynamics model.

The conservative TWR of 3.8 (vs. hardware max ~7) gives a safety margin that prevents the optimizer from planning trajectories that require full-throttle bursts, which are difficult to track reliably.

---

## Results

### Instrumented Track (with MoCap ground truth)

| Metric | Autonomous (VIO) | Champion Pilot (MCK) |
|--------|-----------------|----------------------|
| Avg lap time | **4.65s** | 4.71s |
| Top speed | **20.98 m/s** | 20.87 m/s |
| Path length | 48.93m ± 1.11m | 51.29m ± 5.87m |
| Crashes | **0** | 5 |

The autonomous system beats the champion on lap time and consistency (much lower path length variance). The shorter path length (48.93 vs 51.29m) indicates the drone follows a tighter, more optimal racing line.

### Uninstrumented Track (no external infrastructure)

| Metric | Autonomous | Champion Pilot (MCK) |
|--------|-----------|----------------------|
| Avg lap time | 6.02s | **5.80s** |
| Battery runs | 45 | 63 |
| Crashes | 4 | 2 |

The human pilot is slightly faster and more reliable in the uninstrumented setting. The gap is attributed to accumulated VIO drift in the larger, uncontrolled environment without the correction opportunities of a well-mapped instrumented track.

### Gate Detection Accuracy (Uninstrumented)

- Position errors: order of a few centimeters
- Orientation errors: at most ten degrees

---

## Relevance to Our System

This paper is directly relevant to our racing pipeline. The specific takeaways map closely to our architecture:

1. **Dual-waypoint approach**: We should verify that our `trajectory_optimizer.py` places entry/exit waypoints ±0.4m before/after each gate center along the gate's local x-axis (not world x-axis). If we are using world-frame offsets, that is a bug on non-axis-aligned gates.

2. **Conservative TWR in optimizer**: Our `DroneConstraints` in `trajectory_optimizer.py` should use a conservative thrust cap (e.g., TWR 3.8 not max) to ensure trajectory feasibility and trackability.

3. **Dual-stage EKF**: Their two-stage state estimation (VIO + gate-based drift correction via PnP) matches our design intent in `ekf.py` + `gate_pnp.py`. The specific process noise used for position drift is σₐ² = 8 — this is worth comparing to our EKF config.

4. **CTBR command interface**: Their MPC outputs collective thrust + body rates (CTBR), which is the correct low-level interface. Our `mpc_tracker.py` should target this command format.

5. **Latency compensation**: Their state predictor (compensating MPC computation + FC comms + motor delay) matches our `state_predictor.py` intent.

---

## Actionable Takeaways

### Immediate (high confidence)

1. **Gate waypoint placement**: Place entry waypoint at `gate_center - 0.4m * gate_x_axis` and exit at `gate_center + 0.4m * gate_x_axis`, where `gate_x_axis` is the unit vector in gate heading direction (transformed from gate local frame using gate yaw). For Split-S gates, use ±1.25m.

2. **TWR cap at 3.8**: In `trajectory_optimizer.py`, cap the thrust-to-weight ratio at 3.8 (not the hardware maximum) for trajectory generation. This creates a feasibility buffer.

3. **Dual-stage filtering process noise**: Try σₐ² = 8 as process noise for the position drift state in the VIO correction Kalman filter (the outer stage).

4. **Disable perception-aware costs in MPC**: If using a perception-aware MPC framework, disable the perception objectives and focus purely on trajectory tracking cost.

### Secondary (worth investigating)

5. **State predictor**: Ensure latency compensation accounts for all three delay sources: MPC solve time, FC comms, and motor actuation. Even 20-30ms total delay at 20 m/s means 40-60cm of positional error if uncompensated.

6. **Vision detection**: Their two-stage detection (YOLOv8n for bounding box → MobileNetV3 for 4 corner keypoints) with TensorRT acceleration achieves 24-30ms per frame. Our `gate_pnp.py` pipeline should target similar latency.

7. **VIO configuration**: Disable relocalization and pose jumping in VIO; use mapping with fixed exposure parameters tuned to minimize motion blur.

---

## Limitations & Caveats

1. **Instrumented map required**: The dual-waypoint approach requires knowing gate positions in world frame ahead of time. The system uses a pre-surveyed map of gate locations — it is not truly map-free. In the uninstrumented track, they pre-map the arena before racing.

2. **TWR 3.8 is conservative**: The conservative thrust cap means the optimizer generates slower trajectories than the hardware is capable of. The champion pilot's slight edge in the uninstrumented track may partly reflect this conservatism.

3. **No aerodynamic drag model**: Excluding drag from the optimizer means the planned trajectory is only an approximation of the true optimal path. At 20+ m/s, aerodynamic effects are non-negligible.

4. **VIO drift accumulates**: In the uninstrumented track results, the autonomous system is slower and crashes more. VIO drift without frequent gate-based corrections leads to position error accumulation over multi-lap runs. The system lacks robust re-localization.

5. **Gate orientation uncertainty**: 10-degree orientation errors in gate detection can cause systematic waypoint misplacement. With ±0.4m offsets and 10° error, the lateral displacement of the waypoints can be ~0.07m — acceptable but not negligible at high speeds.

6. **Hardware-specific tuning**: The Betaflight PID gains, MPC cost weights, and state predictor delays are all tuned to their specific hardware (T-motor F60 PRO, HQProp R38, Orin NX). Direct parameter transfer to different hardware requires re-tuning.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Entry waypoint offset | -0.4m along gate x-axis | Standard gate traversal |
| Exit waypoint offset | +0.4m along gate x-axis | Standard gate traversal |
| Split-S entry offset | -1.25m along gate x-axis | Split-S maneuver gates |
| Split-S exit offset | +1.25m along gate x-axis | Split-S maneuver gates |
| Conservative TWR | 3.8 | Trajectory optimizer thrust cap |
| Hardware TWR | ~7 at full battery | Actual drone capability |
| VIO output rate | 200 Hz | Intel T265 tracking camera |
| VIO camera rate | 30 Hz | T265 grayscale fisheye |
| IMU rate | 500 Hz | FC IMU via MSP at 1MBaud |
| FC communication | 1 MBaud MultiWii Serial Protocol | MSP protocol |
| Gate detection latency | 24-30ms per frame | YOLOv8n + MobileNetV3 + TensorRT |
| Drift KF process noise | σₐ² = 8 | Position drift state |
| EKF state dimension | 16 | Quaternion + pos + vel + gyro bias + accel bias |
| EKF error state | 6-dimensional | Euler angle error representation |
| PnP solver | SOLVEPNP_ITERATIVE + Levenberg-Marquardt | OpenCV gate pose estimation |
| MoCap validation | 32-camera Arqus A12 Qualisys, 275Hz | Millimeter precision |
| Instrumented avg lap | 4.65s | Autonomous VIO |
| Instrumented top speed | 20.98 m/s | Autonomous VIO |
| Uninstrumented avg lap | 6.02s | Autonomous VIO |
| Champion avg lap (instrumented) | 4.71s | MCK human pilot |
| Champion avg lap (uninstrumented) | 5.80s | MCK human pilot |
| Gate position error | few centimeters | PnP detection, uninstrumented |
| Gate orientation error | ≤ 10 degrees | PnP detection, uninstrumented |
| YOLOv8n parameters | 3.2M | Gate bounding box detector |
| MobileNetV3-Small parameters | 1.1M | Keypoint detector |
| Vision input resolution | 640×640 (YOLO), 256×256 (MobileNet) | Detection pipeline |
| ONNX opset | v17 | Model export format |
| TensorRT precision | FP16 | Inference acceleration |
| Compute platform | NVIDIA Orin NX | Onboard GPU |
| JetPack version | 5.1.2, CUDA 11.4 | Software stack |
| Motor | T-motor F60 PRO V 2020KV | Brushless motor |
| Propeller | HQProp HeadsUp R38 | Racing prop |
