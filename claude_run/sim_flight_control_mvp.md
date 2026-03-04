# Simulation + Flight Control MVP — Implementation Log

## Date: 2026-02-22

## What was built

### 1. PyBullet Simulation (`sim_pybullet/`)
New package with closed-loop drone racing simulation:
- `drone.py` — `QuadrotorDrone` with box-body physics, attitude-level controller, FPV + spectator cameras
- `gate_models.py` — gate frame creation with highlight/dim/reset color management
- `env.py` — `DroneRaceEnv` managing the full PyBullet world (ground, gates, drone)
- `sequencer.py` — `GateSequencer` with plane-crossing pass-through detection
- `runner.py` — `RaceRunner` closed-loop: physics → detect → plan → control → render

### 2. TRPY Mixer (`flight_control/mixer.py`)
- `TRPYMixer` converts world-frame accelerations to competition-format (throttle, roll, pitch, yaw)
- `TRPYCommand` dataclass for normalized control outputs
- Standard quadrotor decomposition: body-frame accel → attitude angles + thrust

### 3. Flight Control Upgrades (`flight_control/controller.py`)
- Added `step_trpy()` method: full pipeline MPC → PID → TRPY
- Fixed yaw error wrapping for proper angle tracking
- Backward compatible — existing `step()` still works

### 4. Phase 1 Detector (`gate_detection/src/phase1_detector.py`)
- `Phase1GateDetector` for VQ1 (highlighted gates in desaturated environment)
- Saturation + brightness thresholding (2-5ms vs 30-50ms for full pipeline)
- Same `GateDetection` output interface as the classical detector

### 5. Dual Camera Views + HUD
- FPV (1st person) with detection overlay
- Spectator (3rd person chase cam)
- HUD: speed, altitude, target gate, progress, distance
- All rendered via OpenCV `imshow`

### 6. Race Configuration
- JSON-based config format in `sim_pybullet/configs/`
- First race: `race_01.json` with 4 gates in a straight-ish course

## Files created/modified
- NEW: `sim_pybullet/__init__.py`, `__main__.py`, `env.py`, `drone.py`, `gate_models.py`, `sequencer.py`, `runner.py`
- NEW: `sim_pybullet/configs/race_01.json`
- NEW: `flight_control/mixer.py`
- NEW: `gate_detection/src/phase1_detector.py`
- MOD: `flight_control/controller.py` (added `step_trpy()`)
- MOD: `flight_control/__init__.py` (exports TRPYMixer, TRPYCommand, MixerConfig)
- MOD: `requirements.txt` (added pybullet)
- MOD: `ARCH.md` (full architecture update)
- MOD: `RUN.md` (added sim_pybullet commands)

## What's next
- Install pybullet and test the full loop: `pip install pybullet && python3 -m sim_pybullet.runner`
- Tune MPC/PID parameters based on actual PyBullet drone dynamics
- Add more race configs with varying difficulty
- Integrate with the competition's DCL platform when access is provided
