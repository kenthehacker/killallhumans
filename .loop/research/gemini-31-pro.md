# Multi-Track Generalization Research — Gemini 3.1 Pro

## Summary
The current controller is overfit to wide tracks because the unconstrained trajectory optimizer and time-based ILC do not respect the spatial boundaries of tight tracks, causing early overshoot and out-of-order DQs. To fix this, we need to inject spatial awareness into the planning and learning modules. The most promising path is to apply hard spatial bounds (safe flight corridors or virtual tubes) during trajectory generation or learning, ensuring the drone is physically constrained from crossing future gate planes prematurely.

## Top 3 candidate techniques (ranked)

### C1. Time-Optimal Gate-Traversing (TOGT) Planner
- **Citation**: Qin et al., "Time-Optimal Gate-Traversing Planner for Autonomous Drone Racing," *IEEE International Conference on Robotics and Automation (ICRA)*, 2024.
- **What it does**: Instead of treating gates as zero-dimensional waypoints, it models gates as convex polygons/polytopes. The trajectory optimization natively incorporates these spatial configurations to generate time-optimal, dynamically feasible polynomial trajectories.
- **Why it unblocks our stall**: By integrating the physical gate boundaries directly into the optimization problem, the planner is forced to respect the track topology. This prevents the initial acceleration phase from generating a path that overshoots widely into adjacent gate openings on tight tracks.
- **Cost (S/M/L)**: M
- **Expected gain (small/medium/large)**: Large
- **Risks/constraints**: Requires formulating and solving convex optimization problems (e.g., using MINCO or similar); might slightly increase offline generation time.

### C2. Model Predictive Contouring Control with Safety Constraints (MPCC++)
- **Citation**: Krinner et al., "MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints," *Robotics: Science and Systems (RSS)*, 2024.
- **What it does**: Enhances standard MPCC for drone racing by introducing a "track constraint" defined as a spatial prismatic tunnel. This explicitly prevents gate collisions while allowing the optimizer to maximize progress along the track.
- **Why it unblocks our stall**: The prismatic tunnel strictly bounds the drone's allowable position. Even if the drone tries to accelerate aggressively, the hard tunnel constraints mathematically preclude it from crossing into a future gate's plane out of sequence.
- **Cost (S/M/L)**: L
- **Expected gain (small/medium/large)**: Large
- **Risks/constraints**: Requires replacing the geometric tracker with a real-time Nonlinear MPC solver, which poses a significant compute risk for maintaining >100 Hz loop frequencies.

### C3. Spatial Iterative Learning Control within a Virtual Tube
- **Citation**: Lv et al., "Autonomous Drone Racing: Time-Optimal Spatial Iterative Learning Control within a Virtual Tube," *IEEE International Conference on Robotics and Automation (ICRA)*, 2023.
- **What it does**: A model-free ILC approach that defines a spatial "virtual tube" around the track. It iteratively learns control commands based on the drone's spatial position (arc-length) rather than time, constraining all exploration and corrections to stay inside the tube.
- **Why it unblocks our stall**: Our current offline ILC pushes cross-track corrections too aggressively on tight geometries. A spatial ILC bounded by a virtual tube inherently clips unsafe cross-track corrections, stopping the drone from cutting corners into future gates.
- **Cost (S/M/L)**: M
- **Expected gain (small/medium/large)**: Medium
- **Risks/constraints**: Relies on consistent initial paths and requires multiple laps to converge; less effective if the base trajectory is completely broken.

## Other candidates (don't pick, but flag)
- **FAST-Racing (Wang et al., IEEE RA-L 2021)**: Generates Safe Flight Corridors (SFC) as overlapping convex polyhedra to confine SE(3) trajectory planning. Great classical alternative to TOGT.
- **Iterative Learning MPC (Völk et al., 2025)**: Adapts the cost function dynamically and shifts local safe sets. Too heavy for our compute budget but conceptually relevant.
- **Dream to Fly (Romero et al., 2025)**: End-to-end model-based RL mapping raw pixels to thrust/body rates. Exciting, but requires heavy GPU training.

## What NOT to do
- **Heavy End-to-End RL**: Avoid pixel-to-control or massive multi-task PPO/SAC approaches (e.g., DreamerV3) as they violate our compute/GPU constraints and take immense effort to integrate into the existing modular stack.
- **Online NMPC without heavy simplification**: Standard MPCC is computationally expensive. Unless we have a highly optimized solver like acados running in C++, implementing MPCC in Python might destroy our 100 Hz loop requirement.

## My #1 pick if I had to ship one in iter-004
**Add Safe Flight Corridors (SFC) / Prismatic Tunnels to the Trajectory Optimizer.**

Since our base problem is the unconstrained polynomial trajectory overshooting early gates, modifying `planning/trajectory_optimizer.py` is the highest ROI fix. 
**Implementation plan:**
1. Given the gate sequence, generate simple convex overlapping polyhedra (a "corridor") connecting each gate.
2. Add a bounding constraint to the polynomial solver (e.g., sampling points along the spline must lie within the respective polyhedron).
3. Scale the ILC max_corr hyperparameters dynamically based on corridor width (or gate proximity), ensuring `planning/ilc_sections.py` never outputs a correction larger than the distance to the track boundary.
This fixes the root cause directly at the planning stage while retaining our efficient geometric tracker and meeting compute constraints.