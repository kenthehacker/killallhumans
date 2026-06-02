# AI-GP — Windows Sim Autonomous Loop Prompt

**For:** a teammate running this on a **Windows PC with the official AI-GP simulator**, driving **Claude Code** with `/goal` + `/loop` on our behalf.
**You are continuing real work, not starting fresh.** Read `PLAN.md` (the port plan) and §2 "Reference" before doing anything. When unsure, **stop and ask Ken** — do not guess (Rule 7).

---

## How to run this

**Prereqs on the Windows machine:**
1. Official AI-GP sim installed and **running** (`FlightSim.exe`), logged in to the virtual qualifier (sim creds — ask Ken).
2. This repo cloned. **Check out the branch that contains this file** (`AIGP_WINDOWS_LOOP_PROMPT.md`). It should be `aigp-vq1-loop`; if `git checkout aigp-vq1-loop` doesn't contain this file, run `git branch -a` and pick the branch that does. Confirm with `dir AIGP_WINDOWS_LOOP_PROMPT.md` (cmd) / `ls AIGP_WINDOWS_LOOP_PROMPT.md`.
3. **Minimal venv for first contact** (do NOT `pip install -r requirements.txt` yet — it pulls PyQt6/pybullet/MAVSDK/a git dep you don't need and may fail on 3.14):
   ```
   py -3.14 -m venv .venv && .venv\Scripts\activate
   py -3.14 -m pip install pymavlink numpy opencv-python pytest
   ```
   If a wheel fails to build on 3.14, fall back to `py -3.12 -m venv .venv` and record it in `FIRST_CONTACT_FINDINGS.md`. Install full `requirements.txt` only once Action Item 6 needs it.
4. Claude Code installed in the repo.

**Then, in Claude Code, set the goal (one line):**
```
/goal Fly the killallhumans stack through ALL official AI-GP VQ1 gates in order with zero crashes (VQ1 is scored on completion, not speed). Build a live pymavlink CompetitionInterface adapter, drive the sim's runtime gate map + telemetry + camera into RacePipeline, recalibrate the drone model to the real sim, keep the ML residual OFF, and drop race_01/PyBullet-specific tuning. Stop and report at the first clean full-course completion.
```

**Then start the loop (this MUST be a single line — a bare `/loop` is a usage error and won't start):**
```
/loop Follow ./AIGP_WINDOWS_LOOP_PROMPT.md — execute the LOOP BODY. Each iteration: re-read this file, pick exactly ONE Action Item (the next incomplete one, from git log + FIRST_CONTACT_FINDINGS.md), do only that item, run its gate, commit (and push, once) only on green, then END the iteration. If blocked or ambiguous, STOP and write the question to FIRST_CONTACT_FINDINGS.md instead of guessing.
```

**Loop bounds (hard):** stop after **20 iterations** or **8 hours** wall-clock, whichever first; write `LOOP_STOPPED.md` with the last completed item and why. **One Action Item per iteration.** Items 5–7 require **live sim evidence** (a log line / JSONL) — passing pytest is NOT sufficient and must never be reported as a completed flight gate.

---

# LOOP BODY

## 1. What
- **Repo / branch:** `killallhumans` @ the branch holding this file (see prereq 2). Do work on a child branch, created idempotently: `git switch -c aigp/live-sim-bringup 2>/dev/null || git switch aigp/live-sim-bringup`.
- **Task:** make our autonomy stack (`race_pipeline.py` → EKF → SE(3) tracker) fly the **official AI-GP sim** by building the missing **live pymavlink transport adapter** + a bring-up script that wires the sim's runtime gate map, telemetry, camera, race-status and collisions into `RacePipeline`; then recalibrate to real physics and get a VQ1 completion.
- **Scope:** VQ1 = **completion only** (all gates, in order, no crash, ≤8 min). Do **not** optimize for speed yet (VQ2).

## 2. Reference impl (READ THIS FIRST)
- **`PLAN.md`** (this repo) — the port plan. §1 = exact interface contract; §4 = open-questions checklist; §5 = phase plan. This LOOP BODY executes PLAN Phases 0→2.
- **PR #9** landed the scaffolding you build on (do **not** re-derive):
  - `competition/aigp_messages.py` — byte-exact parsers (race-status `"<BQqqIq"`, track gate `"<Hfffffffff"`, `TrackInfoReassembler`).
  - `competition/track_data.py` — `track_data_to_gatespecs()` (sim gate map → `GateSpec`s). Tested in `competition/tests/test_track_data.py`.
  - `competition/aigp_recorder.py` — passive MAVLink → JSONL recorder. **Limitations (you will fix in Item 2):** it records `track_data` as only `{"num_gates": N}` (no per-gate bytes), does **not** capture the vision stream, and does **not** send any setpoints (so it cannot tell you whether the sim accepts `SET_ATTITUDE_TARGET`).
  - `competition/vision_udp.py` — camera receiver (`VisionUdpListener`, port 5600). Reuse it.
- **Official template** (the byte formats already live in `aigp_messages.py`): `PyAIPilotExample/{main,setup,mavlink_rx,vision_rx,timesync,controller}.py`. `setup.py` connects with `mavutil.mavlink_connection('udpin:<ip>:14550')` → `wait_heartbeat()` → `recv_match()`. `controller.py` shows arm (`MAV_CMD_COMPONENT_ARM_DISARM`), `SIM_RESET` (cmd 31000), and `SET_ATTITUDE_TARGET` usage (attitude-quaternion+thrust, or body-rate+thrust via the `ATTITUDE_IGNORE` mask).
- **Closest existing adapters to mirror:** `competition/mavlink_bridge.py` (`MAVLinkBridge(CompetitionInterface)`, MAVSDK — **do not copy its offboard/connect semantics blindly**) and `competition/pybullet_adapter.py`. You are writing a **third** `CompetitionInterface` (`competition/adapter.py:154`) using **pymavlink**.

## 3. Context — interface contract (authoritative; full detail in `PLAN.md §1`)
- MAVLink 2 over UDP, **`udpin:127.0.0.1:14550`** (single-machine). Sim→client: `HEARTBEAT, ATTITUDE, HIGHRES_IMU, TIMESYNC, LOCAL_POSITION_NED, ODOMETRY, COLLISION`, plus `ENCAPSULATED_DATA` (race-status + chunked track-info) and `DATA_TRANSMISSION_HANDSHAKE`. Client→sim: `SET_ATTITUDE_TARGET`, `SET_POSITION_TARGET_LOCAL_NED`, arm, `SIM_RESET` (31000).
- Geometry/timing constants live in `competition/aigp_geometry.py` (gate 1.5 m interior; camera 640×360, fx=fy=320, cx=320, cy=180, +20° up-tilt; physics 120 Hz, cmd <100 Hz). **Import from there; never hard-code.**
- Frames: internal NED; origin = arm point (local, not global). No GPS.

## 4. Platform / domain specifics
- **Transport = raw pymavlink, port 14550** — NOT MAVSDK, NOT 14540. `RacePipeline.run(address=…)`'s default `udp://:14540` is MAVSDK-only; on the live path **pass `address="udpin:127.0.0.1:14550"`** and make `AIGPMavlinkInterface.connect` parse that pymavlink form. The sim's custom `ENCAPSULATED_DATA`/`COLLISION`/`SIM_RESET` are surfaced by raw pymavlink, not MAVSDK.
- **Vision** is a separate UDP stream on 5600 (`competition/vision_udp.py`, `VisionUdpListener`, bind `127.0.0.1:5600` single-machine).
- **Gate map arrives at runtime** via the track-info packet → `track_data_to_gatespecs()` → `RacePipeline.configure(gates=…)`. **Never read `sim_pybullet/configs/*.json` for gates** — practice fixtures.
- **Sequencing + crashes come from the sim:** race-status `active_gate_index` (ground-truth next gate) and `COLLISION` (`id` 1001=gate, 1002=environment). The pipeline does **not** consume these today — you must wire them (§5).
- **Control:** the SE(3) tracker emits `AttitudeCommand(roll,pitch,yaw,thrust)`. Send `SET_ATTITUDE_TARGET` with attitude-quaternion+thrust (clear `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`). Body-rate mode is a VQ2 optimization.

## 5. Adapter contract + bring-up orchestration (build these; they don't exist yet)
**`competition/aigp_mavlink.py` → `class AIGPMavlinkInterface(CompetitionInterface)`** must implement every ABC method (`competition/adapter.py:154`) and additionally expose:
- `async connect(address="udpin:127.0.0.1:14550")` — parse the pymavlink URL; `wait_heartbeat()`; start the RX thread + `VisionUdpListener`.
- `async get_telemetry() -> TelemetryState` — from `ATTITUDE` + `HIGHRES_IMU` + (`LOCAL_POSITION_NED`/`ODOMETRY` if Item-2 findings say they're populated; else dead-reckon per Rule 4).
- `async get_camera_frame()` — from `VisionUdpListener.latest_frame()`.
- `async send_attitude(cmd)` → `SET_ATTITUDE_TARGET` (quaternion+thrust).
- `start_offboard()` — **define explicitly**: AIGP uses raw `SET_ATTITUDE_TARGET`, not MAVSDK offboard. Either a no-op (with a comment proving setpoints are accepted without a mode switch — confirm in Item 2) or "stream neutral/hover `SET_ATTITUDE_TARGET` at ≥10 Hz until armed." **Must not require MAVSDK.**
- **Side channels (new — `TelemetryState` has no fields for these):** `get_race_status() -> RaceStatus | None` (latest, incl. `active_gate_index`), `get_collisions() -> list` (drained), and `async wait_for_track_data(timeout_s) -> TrackData` (blocks until the chunked track-info packet is fully reassembled).

**`scripts/aigp_live_race.py`** — the only correct run order (a bare `RacePipeline.run()` raises if `configure()` wasn't called, `race_pipeline.py:290`):
```text
iface = AIGPMavlinkInterface()
await iface.connect("udpin:127.0.0.1:14550")
track = await iface.wait_for_track_data(timeout_s=120)        # log num_gates; abort if none
pipeline = RacePipeline(iface, PipelineConfig(max_speed=1.5, target_hz=50.0))   # conservative
assert pipeline.tracker.config.use_residual is False           # Rule 1 guard
start_pos = (await iface.get_telemetry()).position_ned
pipeline.configure(gates=track_data_to_gatespecs(track), start_position=start_pos)
# wire safety: each tick, if a COLLISION arrived → sequencer.mark_collision(current_gate) (gate id 1001)
#              or stop the run (environment id 1002); log active_gate_index vs sequencer progress.
await pipeline.run(address="udpin:127.0.0.1:14550")
```

## 6. Hard rules (each has already-known failure modes — do not skip)
1. **ML residual stays OFF.** `use_residual=False` default (`control/mpc_tracker.py:109`); no `control/residual_weights.npz` exists; what training exists was fit to the *kinematic bench* (not even PyBullet) → iter-031 closed-loop score **−1e6**, broke slalom completion. **Do NOT** set `use_residual=True`, **do NOT** run `scripts/train_tracker_residual.py` / `scripts/init_residual_weights.py`, **do NOT** set `trace_tracker_features=True` on the live sim. `assert pipeline.tracker.config.use_residual is False` in the entry script. `tests/test_residual_matrix_gain.py` **SKIPping** (no weights) is expected and fine — "green" there means "skipped," not "validated." (`.loop/synthesis/iter_031_ml_result.md`)
2. **Gate map from the sim, never JSON.** `track_data_to_gatespecs()` on the runtime packet. `race_01.json` (1.2 m openings, hand-built helix) is wrong for AIGP (1.5 m). Do not edit `race_01.json`. (`competition/track_data.py:25`)
3. **Drone model + gains are kinematic-sim artifacts — recalibrate, don't trust; `calibration.py` is a STUB.** `feedforward_accel=0.50` and `drag_coefficient=0.0` are tuned to a fake `drag=0.5` bench (`control/mpc_tracker.py:59-70`); `drone_spec` `mass=1 kg`/`thrust=20 N` are "NOT verified" placeholders (`competition/drone_spec.py:26-32`). `competition/calibration.py` is explicitly a stub ("validation against the DCL binary deferred"). **First live run clamps:** cap `PipelineConfig.max_speed` at **1.5** (default is 8.0), tilt at ~**0.35 rad**, normalized thrust at ~**0.65** (set the matching `TrackerConfig`/adapter fields — verify exact names in `control/mpc_tracker.py`). Abort the run if telemetry is stale >0.5 s, thrust pins >0.65 for >0.5 s, or any `COLLISION` arrives. Lift caps only after a documented calibration produces real thrust/drag numbers.
4. **Record before you build; the state path forks on what's actually populated.** Confirm from Item-2 JSONL whether `LOCAL_POSITION_NED`/`ODOMETRY` carry real (non-zero) local position. If yes → feed the EKF. If absent/zero (the no-GPS branch) → dead-reckon from velocity + correct with gate-PnP against the known map. Report which branch in `FIRST_CONTACT_FINDINGS.md`.
5. **Drop only the *overfit*, don't delete generic logic.** Remove/override race_01/fixture/ILC-specific constants and PyBullet-proven values — but do NOT delete generic curvature/S-turn/helix detection in `planning/trajectory_optimizer.py` (it may generalize). Concretely: set the planner's speed envelope from calibrated limits via `PlannerConfig(plan_max_speed_mps=…)` (default 4.0 is a CF2X artifact, `:634`), ignore `race_01.json` `ilc_section_overrides`, and recalibrate the detector (`gate_detection/src/phase1_detector.py:52-65`: `saturation_threshold=60`, `brightness_threshold=200`, `image_height=480`→360, `GATE_*_METERS=1.0`→1.5) against the real render via `gate_detection/src/color_calibrator.py`. Only disable a heuristic if a live-sim run shows it's harmful.
6. **Sim time, not wall clock** (currently `RaceSession` rate-limits and `RacePipeline._maybe_replan` use wall clock, `race_pipeline.py:424`). Make this an Item-6 refactor: drive timeout/replan/logging off `TelemetryState.timestamp_us` (or race-status sim time) when populated; wall clock only for sleep/backoff. Add a unit test with simulated non-realtime timestamps.
7. **Stop-and-ask + no fabrication.** If telemetry you need is absent, the sim rejects a setpoint type, or the spec is ambiguous, **STOP the iteration** and write the blocker to `FIRST_CONTACT_FINDINGS.md`. Never invent constants. Items 5–7 are only "done" with a live log/JSONL artifact proving it — never claim a flight gate from pytest alone.
8. **Git discipline.** Branch creation idempotent (`git switch -c X || git switch X`). Commit on each green gate. **Push at most once per iteration**; if push fails (no creds/network), commit locally + write `PUSH_FAILED.txt` and continue — do not retry-loop. **Never force-push, never delete branches/history.**
9. **Human gate after first contact.** After Item 2, **halt the loop** until Ken confirms in chat OR the file `.loop/ALLOW_LIVE_ADAPTER` exists. Do not start Item 3 (building the live adapter) before that — the telemetry findings may change the design.

## 7. Smoke test / run commands (run from the repo root; PowerShell shown — cmd/bash equivalents noted)
```powershell
# Branch (idempotent)
git switch -c aigp/live-sim-bringup 2>$null; if ($LASTEXITCODE -ne 0) { git switch aigp/live-sim-bringup }

# Green baseline (PowerShell). cmd: set "PYTHONPATH=%CD%" && py -3.14 -m pytest ...   bash: PYTHONPATH=. python -m pytest ...
$env:PYTHONPATH = (Get-Location).Path
py -3.14 -m pytest competition/ tests/test_vision_udp.py tests/test_vision_udp_listener.py -q
#   (do NOT run repo-root `pytest` with no path — it collects 300+ unrelated tests, some needing pybullet)

# FIRST CONTACT — MAVLink stream (the recorder is MAVLink-only today; ~60 s with the sim running)
py -3.14 -m competition.aigp_recorder --out FIRST_CONTACT.jsonl --duration 60
py -3.14 -c "import json,collections; print(collections.Counter(json.loads(l)['type'] for l in open('FIRST_CONTACT.jsonl')))"
py -3.14 -c "import json; [print(l.strip()) for l in open('FIRST_CONTACT.jsonl') if json.loads(l)['type'] in ('LOCAL_POSITION_NED','ODOMETRY','track_data','race_status','COLLISION')][:20]"
```
Confirm: heartbeat connects; whether `LOCAL_POSITION_NED`/`ODOMETRY` are present **and non-zero**; whether a `track_data` record appears. Record answers in `FIRST_CONTACT_FINDINGS.md`.

## 8. Action items (one per loop iteration; each ends at its gate)
1. **Setup & baseline.** Branch; minimal venv (prereq 3); run the pytest baseline to green; confirm `FlightSim.exe` running + logged in. Gate: tests green + sim up.
2. **Stock-pilot smoke (PLAN §0.2).** Run the official `PyAIPilotExample/main.py` **unmodified** ~30 s; confirm heartbeat connects and arm + a `SET_ATTITUDE_TARGET`/motor command **actually moves the drone**. Gate: stock pilot moves the drone. **If this fails (firewall/login/port), STOP — fix the environment before any custom code.**
3. **First contact + recorder upgrade.** Extend `competition/aigp_recorder.py` so `track_data` records carry every gate `{gate_id, position_ned, orientation_wxyz, width, height}` (or raw base64 chunk payloads) AND add a vision capture via `VisionUdpListener` (log frame size/FPS). Run it; write `FIRST_CONTACT_FINDINGS.md` answering all PLAN §4 questions + "did the stock pilot's `SET_ATTITUDE_TARGET` move the drone?" + vision FPS. Gate: findings committed; replay fixture can reconstruct `TrackData` (not just `num_gates`). **Then HALT for the human gate (Rule 9).**
4. **Build `AIGPMavlinkInterface`** per §5 (all ABC methods + side channels). TDD against the `FIRST_CONTACT.jsonl` replay fixture: (a) `TelemetryState` from ATTITUDE/IMU/LOCAL_POSITION_NED/ODOMETRY; (b) `get_track_data()` + round-trip through `track_data_to_gatespecs`; (c) `get_race_status()`/`get_collisions()`. A tiny JSONL replayer is enough — no live sim for unit tests. Gate: new unit tests RUN green; replay reproduces the streams.
5. **Connect-smoke ONLY (no flight).** `await iface.connect("udpin:127.0.0.1:14550"); await iface.get_telemetry(); await iface.wait_for_track_data(120)`. **Do NOT call `RacePipeline.run()`** (it would arm + fly). Gate: live telemetry + a full track-data packet received, logged; zero motor commands sent.
6. **First live flight** via `scripts/aigp_live_race.py` (§5) with the Rule-3 clamps + collision/active_gate wiring. Gate: **a live log shows `active_gate_index ≥ 1` after arm** (or sequencer "Gate X passed" + telemetry position). PyBullet smokes do NOT satisfy this. Commit + push (once).
7. **Calibrate + de-overfit + sim-time.** Identify real thrust/drag from live telemetry (extend `calibration.py` with a real collection+fit; document thresholds), set `drone_spec`, re-tune gains conservatively; apply Rule 5 (parameterize, don't delete) and Rule 6 (sim time). Gate: stable hover + step response on the real sim, commanded-vs-achieved accel within a stated tolerance; then iterate to **all gates in order, no crash, ≤8 min** (the `/goal`). **STOP and report.**
8. **Persist.** Write `FIRST_CONTACT_FINDINGS.md` + `docs/aigp/<date>-live-bringup.md` (telemetry availability, calibrated drone numbers, gotchas, evidence). Push `aigp/live-sim-bringup`, open a PR, tell Ken.

## 9. Gotchas
- **Run pytest from the repo root** with `PYTHONPATH` set to the root (PowerShell `$env:PYTHONPATH=(Get-Location).Path`; cmd `set "PYTHONPATH=%CD%"`; bash `PYTHONPATH=.`). No `pytest.ini`/`conftest.py` — imports resolve only with the root on `sys.path`. Don't run bare repo-root `pytest`.
- **Python 3.14.2** per the kit; if `pymavlink`/`opencv-python` wheels fail, use a 3.12 venv and record it (pure parsers don't care; `vision_udp` decode needs `cv2`).
- **The recorder is passive + MAVLink-only today** (Item 3 upgrades it). It cannot, by itself, prove the sim accepts your control setpoints — that's what the Item-2 stock-pilot smoke is for.
- **Gate dimensions:** trust the track packet's width/height + `aigp_geometry.py` (1.5 m), not `phase1_detector.py`'s `1.0` or `race_01.json`'s `1.2`.
- **`active_gate_index`/`COLLISION` aren't in `TelemetryState`** — they come via the adapter's side channels you build (§5). The pipeline ignores them until you wire them.
- If `LOCAL_POSITION_NED` is all-zero/absent, that's the no-GPS branch — dead-reckon + gate-PnP (Rule 4); report it, don't assume position.

---
*This runbook is precise on purpose. When in doubt, prefer Rule 7 (stop and report) over a plausible guess — Ken would rather answer a question than debug a fabricated constant on the real drone.*
