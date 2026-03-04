# Issues
There is an issue with the first simulated run 

1. WASD doesn't actually orbit around the drone it just makes the world x-ray on one of the pop up screen and on the other one it doesnt do anything

2. This is just a screenshot but in `claude_run/Screenshot 2026-03-01 at 3.34.06 PM.png` i noticed that after passing the first gate it veers to the right even though the second gate is on the left side of the POV. Something in the flight control is going awry

3. We should be logging the test runs by logging the drone position & the gate positions per frame so that way we can see the trends. It should also be friendly for machine learning training should we swap the flight control to ML from deterministic logic

4. Scan the internet for literature regarding flight planning logic that can be useful 

# Need to build:
In addition to resolving the issues I mentioned above we need to do the following

1. If we dont have it yet, we need to add physics logic to add thrust for the propellers like in a regular FPV because we have to simulate realistic drone physics. So if our algo puts more thrust in the rear propellers it should tilt forwards more and go faster

2. The flight plan logic seems not very good because it just veers in a straight direction instead of turning and twisting to the right gate. Also it veered in the wrong direction 

3. Make some of the gates varying in different vertical heights but they shouldn't be drastically different heights so that the next gate will be in the camera frame

4. make the animations slightly faster its slow at the moment.

---

# Fixes Applied (Claude)

## Issue 1: WASD orbit broken
**Root cause**: PyBullet GUI window was stealing keyboard events. WASD/arrow keys were
being intercepted by PyBullet's internal camera controller, causing the "x-ray" visual artifacts.
**Fix**:
- Disabled PyBullet's keyboard shortcuts (`COV_ENABLE_KEYBOARD_SHORTCUTS=0`)
- Defaulted to `p.DIRECT` mode (no PyBullet window) since we render our own FPV+Spectator views
- Fixed arrow key codes for macOS OpenCV compatibility
- Use `--pybullet-gui` flag only if you need PyBullet's own 3D viewer

## Issue 2: Drone veers wrong direction
**Root cause**: Two bugs found:
1. **Mixer sign inversion**: In PyBullet's ENU coordinate system, positive roll (right-hand rule
   about +x) tilts the drone RIGHT. But the mixer was computing `atan2(ay_body, g)` directly,
   producing positive roll when the drone needed to go LEFT (+y). Both roll AND pitch signs
   were inverted. Fixed by negating both: `desired_pitch = -atan2(ax_body, g)`,
   `desired_roll = -atan2(ay_body, g)`.
2. **MPC returning terminal velocity**: The MPC was returning the velocity at the END of the
   15-step horizon instead of the velocity after the FIRST step. This caused the PID to track
   a velocity setpoint that was way ahead of what the drone should be doing now. Fixed to
   return `first_step_vel`.

## Issue 3: Telemetry logging
**Fix**: Added per-frame CSV logging to `logs/race_YYYYMMDD_HHMMSS.csv` with columns:
sim_time, frame, pos_xyz, vel_xyz, roll/pitch/yaw, target_gate_id, target_xyz,
dist_to_gate, throttle/roll/pitch/yaw commands, gates_passed/total.
Format is ML-training-friendly (flat CSV, one row per control step).

## Issue 4: Flight planning literature
See section below.

## Build 1: Per-motor thrust physics
**Fix**: Replaced aggregate force+torque model with individual motor forces. 4 motors at arm
tips apply thrust along body Z. Differential thrust naturally creates roll/pitch torques.
Mixing matrix converts TRPY to 4 motor thrusts with proper signs for X-config layout.

## Build 2: Flight plan improvements
Fixed the mixer/MPC bugs above. For next-level path planning, see literature below.

## Build 3: Varying gate heights
**Fix**: Updated `race_01.json` with gates at z=1.5, 2.2, 1.3, 2.8m (gradual variation,
each within camera FOV of previous gate).

## Build 4: Faster animation
**Fix**:
- Defaulted to `p.DIRECT` (no double-rendering overhead from PyBullet GUI)
- Added `--sim-speed` flag (default 2x) that runs multiple physics steps per control cycle
- Combine with `--sim-speed 4` or higher for faster runs

---

# Flight Planning Literature Research

## Approaches ranked by relevance to our competition:

### 1. Time-Optimal Gate-Traversing (TOGT) Planner - Most relevant
- **Paper**: "Time-Optimal Gate-Traversing Planner for Autonomous Drone Racing" (2024)
- **Source**: https://arxiv.org/html/2309.06837v3
- **Code**: https://github.com/FSC-Lab/TOGT-Planner (C++, Python wrapper available)
- **Key insight**: Models gate shapes/sizes as constraints, not just waypoints. Drones
  utilize the full free space within gates rather than flying through centers.
- **Performance**: Computes full trajectories through dozens of gates in seconds.
- **Why relevant**: Directly applicable to our competition format. Could replace our
  simple MPC with a pre-computed optimal trajectory.

### 2. Model Predictive Contouring Control (MPCC)
- **Paper**: "Time-Optimal Planning for Quadrotor Waypoint Flight" (Science Robotics, 2022)
- **Source**: https://rpg.ifi.uzh.ch/docs/TRO22_MPCC_Romero.pdf
- **Key insight**: Solves time allocation and control simultaneously in real-time.
  Achieved 60+ km/h and beat world-class human pilots.
- **Why relevant**: Real-time replanning capability for handling detection errors.

### 3. Minimum Snap Trajectory + NMPC
- **Key insight**: 4th-order polynomial trajectory optimization (minimize snap = 4th
  derivative of position) creates smooth, aggressive paths. Combined with Nonlinear MPC
  for real-time tracking with dynamic adjustments.
- **Why relevant**: Well-established, smooth trajectories good for gate sequencing.

### 4. Deep Reinforcement Learning (DRL)
- **Paper**: "Obstacle-Aware Autonomous Racing" (2024)
- **Key insight**: End-to-end learned policies via domain randomization. Reached 70 km/h
  in unseen environments.
- **Why relevant**: Could be our long-term approach if we collect enough sim data.
  Our telemetry logging is designed to support this.

### 5. AlphaPilot Architecture
- **Paper**: ArXiv 2005.12813 (2020, 2nd place AlphaPilot Challenge)
- **Key insight**: Vision-based detection -> nonlinear filtering -> time-optimal planning.
  Builds global gate map from multiple detections, compensates state drift.
- **Why relevant**: Most similar to our current architecture. Good reference for
  detection-to-planning pipeline integration.

## Recommended upgrade path:
1. **Short-term**: Fix current MPC bugs (done), tune PID gains
2. **Medium-term**: Implement minimum-snap trajectory generation between gates
3. **Long-term**: Integrate TOGT planner or train DRL policy from telemetry data
