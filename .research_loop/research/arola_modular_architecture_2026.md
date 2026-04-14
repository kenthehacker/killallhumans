# AROLA: A Modular Layered Architecture for Scaled Autonomous Racing
- **URL**: https://arxiv.org/abs/2602.02730
- **Year**: 2026
- **Venue**: arXiv preprint
- **Authors**: Fam Shihata, Mohammed Abdelazim (German International University in Berlin), Ahmed Hussein (IAV GmbH, Berlin)

## Key Contribution

AROLA proposes a standardized, open-layered modular software architecture for autonomous racing that decomposes the full autonomy stack into eight functional stages connected through standardized ROS 2 interfaces. The key insight is that the fragmented state of autonomous racing codebases — where teams build monolithic, tightly-coupled systems — hampers reproducibility, benchmarking, and incremental improvement. By enforcing strict interface contracts between layers, any single module (e.g., a controller, a planner, a localizer) can be swapped without touching the rest of the stack.

The second major contribution is the Race Monitor framework, a companion benchmarking and telemetry suite that provides real-time per-lap performance logging, trajectory error analysis, computational resource monitoring, and standardized evaluation output. This addresses a chronic problem in racing research: the inability to fairly compare different algorithms because evaluation setups differ. AROLA was validated at the 2025 RoboRacer IV25 competition where it achieved third place with a 10.1-second lap time.

## Technical Approach

### Eight-Layer Architecture

AROLA decomposes the autonomous racing pipeline into eight sequential layers:

1. **Sensing** — Raw sensor acquisition (LiDAR, cameras, IMU)
2. **Preprocessing** — Filtering, cleaning, and fusing raw data streams
3. **Perception** — Semantic interpretation of sensor data (obstacle detection, gate detection)
4. **Localization & Mapping** — Coupled SLAM for navigation state estimation
5. **Planning** — Feasible trajectory generation based on perception and map
6. **Behavior** — High-level decision-making via FSMs or learned policies (e.g., overtaking, pit-stop)
7. **Control** — Trajectory tracking with deviation compensation (MPC, Pure Pursuit, LQR, etc.)
8. **Actuation** — Physical command execution to motors/servos

Data flows sequentially through these layers, though optional cross-layer interactions are permitted for performance (e.g., perception feeding directly to control for reactive avoidance).

### Standardized ROS 2 Interfaces

Each layer communicates via standardized ROS 2 topics with fixed message types:
- `/scan/scan` -> `sensor_msgs/LaserScan`
- `/odom/odom` -> `nav_msgs/Odometry`
- `/map/map` -> `nav_msgs/OccupancyGrid`
- `/drive/drive` -> `ackermann_msgs/AckermannDriveStamped`
- `tf` messages for coordinate frame alignment

Namespacing supports multi-agent scenarios (e.g., `/ego_racecar/scan`). Lowercase naming conventions are enforced throughout.

### Race Monitor Framework

The Race Monitor publishes real-time telemetry on dedicated ROS 2 topics:
- `/race_monitor/lap_count` (Int32)
- `/race_monitor/lap_time` (Float32)
- `/race_monitor/race_running` (Bool)
- `/race_monitor/race_status` (String)
- `/race_monitor/total_distance` (Float32)
- `/race_monitor/current_trajectory` (nav_msgs/Path)
- `/race_monitor/trajectory_metrics` (String)

Configuration is YAML-based, specifying module selection, reference trajectories, and logging parameters. It supports CSV, TUM, and KITTI trajectory formats for reference trajectories. It integrates the `evo` package for standardized trajectory evaluation (APE/RPE metrics). Interactive RViz tools allow defining start/finish zones for lap detection.

## Results

### RoboRacer IV25 Competition (Three Controllers Compared)

| Metric | Gap Follower | MPC | Pure Pursuit |
|--------|-------------|-----|--------------|
| Best Lap Time (s) | 12.60 | 10.24 | 10.10 |
| Avg Lap Time (s) | 12.85 | 10.40 | 10.35 |
| Consistency Score | -- | 0.98 | 0.92 |
| Avg Speed (m/s) | 4.30 | 4.99 | 4.95 |
| Control Latency (ms) | 12 | 42 | 18 |
| CPU Load (%) | 22 | 55 | 28 |
| RPE Mean | -- | 4.2 | 4.3 |
| APE Mean (m) | -- | 0.19 | 0.22 |

Competition outcome: third place, 10.1s best lap (0.02s behind second, 1.24s behind first).

### LQR Controller (Berlin Map, 7-Lap Test)

| Metric | Value |
|--------|-------|
| Best Lap Time (s) | 19.53 |
| Avg Lap Time (s) | 19.75 |
| Lap Time Std Dev (s) | 0.15 |
| Consistency Score | 0.99 |
| Avg Speed (m/s) | 3.40 |
| Max Speed (m/s) | 3.79 |
| APE Mean (m) | 4.044 |
| RPE Mean | 4.42 |

The MPC controller had the best balance of speed and consistency (0.98 consistency score) but at the cost of higher CPU load (55%) and control latency (42ms). Pure Pursuit was nearly as fast with much lower computational cost (28% CPU, 18ms latency).

## Relevance to Our System

This paper is directly relevant to our drone racing stack in several ways:

**Benchmarking and Reproducibility**: Our `scripts/benchmark.py` already follows a similar philosophy to Race Monitor — structured JSON output with per-gate metrics, tracking error, loop frequency, and crash detection. However, AROLA's approach of publishing telemetry on standardized topics and supporting multiple trajectory output formats (CSV, TUM, KITTI) is more mature. We could adopt their consistency score metric (std_dev of lap/race times across runs) to quantify controller reliability beyond single-run benchmarks.

**Modular Architecture**: Our stack (`estimation/`, `planning/`, `control/`, `gate_sequencing/`) already follows a roughly layered decomposition, but the interfaces between modules are Python function calls rather than standardized contracts. AROLA's strict interface approach would benefit us if we wanted to swap controllers (e.g., testing MPC vs. geometric tracker vs. PD) without touching `race_pipeline.py`. Currently, our `race_pipeline.py` orchestrator is more tightly coupled.

**Logging/Telemetry**: The Race Monitor's per-lap breakdown and trajectory error visualization using the `evo` package is something we lack. Our benchmark outputs aggregate metrics but does not produce trajectory-level error plots or per-segment analysis that would help diagnose where on the track performance degrades.

**Controller Comparison Methodology**: Their side-by-side comparison of Gap Follower, MPC, Pure Pursuit, and LQR on the same track with identical sensing/planning is a methodology we should adopt. Our `mpc_tracker.py` contains both a geometric (SE(3) Lee) tracker and a PD tracker, but we lack a systematic A/B comparison framework.

## Actionable Takeaways

1. **Add a consistency score metric** to `scripts/benchmark.py`: run N races (e.g., 5-7) and compute lap time standard deviation and a consistency score (best/avg ratio). This captures reliability, not just peak performance.

2. **Standardize module interfaces** with explicit dataclass contracts between pipeline stages. Define `EstimationOutput`, `TrajectoryOutput`, `ControlOutput` types that are the sole interface between modules, making it easy to swap implementations.

3. **Integrate the `evo` trajectory evaluation package** for post-run analysis. Export our drone trajectory and reference trajectory in TUM format, then compute APE/RPE for standardized comparison with the literature.

4. **Build a controller A/B testing harness** that runs the same trajectory with different controllers (geometric, PD, future MPC) and produces a comparison table like AROLA's, including tracking error, computational cost, and consistency.

5. **Add per-gate trajectory error logging** to the benchmark. AROLA's Race Monitor breaks down error by track segment; we should do the same per gate to identify which gates cause the most tracking error.

6. **Add computational monitoring** — log per-loop CPU time for each pipeline stage (EKF, trajectory evaluation, control) separately, not just overall loop Hz. AROLA tracks CPU load % per controller, which helped them identify MPC's 55% CPU cost.

7. **YAML-based run configuration** for benchmark experiments. Instead of command-line flags, use a YAML config specifying which modules to use, which track to run, and which metrics to collect. This improves reproducibility.

## Limitations & Caveats

- **Ground vehicle focus**: AROLA targets 1/10-scale autonomous cars (RoboRacer), not quadrotors. The eight-layer decomposition maps imperfectly to drones — we have no "actuation" layer separate from control (our attitude commands go directly to the flight controller), and our "behavior" layer is essentially the gate sequencer.

- **ROS 2 dependency**: The standardized interfaces assume ROS 2 as middleware. Our stack is pure Python with no ROS dependency, and adding ROS would be over-engineering for our competition setting. The interface standardization concept is valuable, but we should implement it as Python protocols/ABCs, not ROS topics.

- **LiDAR-centric**: Their sensing layer centers on 2D LiDAR (`LaserScan`). Our primary sensing is monocular camera with PnP gate detection — a fundamentally different perception pipeline.

- **No aggressive dynamics**: Their best speeds are ~5 m/s on ground vehicles. Our drone operates at much higher speeds with 3D dynamics, making their specific controller parameters and latency tolerances not directly transferable. Their 42ms MPC latency is tolerable at 5 m/s but would be problematic at 15+ m/s drone speeds.

- **Limited evaluation scope**: They acknowledge that metrics focus on lap time and trajectory error, omitting safety margins and multi-agent robustness. Similarly, their evaluation is on a single platform and sensor configuration.

- **No learning-based components**: The architecture is classical-only. They note integration of end-to-end learned components as future work, which limits applicability to hybrid RL/classical approaches.

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| MPC control latency | 42 ms | RoboRacer IV25 |
| Pure Pursuit control latency | 18 ms | RoboRacer IV25 |
| Gap Follower control latency | 12 ms | RoboRacer IV25 |
| MPC CPU load | 55% | RoboRacer IV25 |
| Pure Pursuit CPU load | 28% | RoboRacer IV25 |
| Gap Follower CPU load | 22% | RoboRacer IV25 |
| Best competition lap time | 10.10 s | Pure Pursuit, IV25 |
| MPC APE mean | 0.19 m | Trajectory tracking accuracy |
| Pure Pursuit APE mean | 0.22 m | Trajectory tracking accuracy |
| LQR consistency score | 0.99 | 7-lap test, Berlin map |
| MPC consistency score | 0.98 | IV25 competition |
| LQR lap time std dev | 0.15 s | 7-lap test |
| LQR avg speed | 3.40 m/s | Berlin map |
| LQR max speed | 3.79 m/s | Berlin map |
| Competition margin (2nd to 3rd) | 0.02 s | IV25 final results |
