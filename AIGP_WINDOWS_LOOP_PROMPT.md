# AI-GP — Windows Sim Autonomous Loop Prompt

**For:** a teammate running this on a **Windows PC with the official AI-GP simulator**, driving **Claude Code** with `/goal` + `/loop` on our behalf.
**You are continuing real work, not starting fresh.** Read `PLAN.md` (the port plan) and §2 "Reference" before doing anything. When unsure, **stop and ask Ken** — do not guess (Rule 7).

---

## How to run this

**Prereqs on the Windows machine:**
1. Official AI-GP sim installed and **running** (`FlightSim.exe`), logged in to the virtual qualifier (sim creds — ask Ken).
2. **Extract the official kit** (`AIGP_3364.zip` + `PyAIPilotExample.zip`) to a known path, e.g. `$env:USERPROFILE\AIGP\`. Note `$env:USERPROFILE\AIGP\PyAIPilotExample\` — **this is NOT in this repo**; the runbook refers to it by that path.
3. This repo cloned. **Check out the branch/PR that contains this file** (`AIGP_WINDOWS_LOOP_PROMPT.md`). Once the prompt PR merges it will be on **`aigp-vq1-loop`**; until then it's on the PR branch **`aigp/windows-loop-prompt`**. Don't trust the branch *name* — verify the file is present: `Test-Path AIGP_WINDOWS_LOOP_PROMPT.md` must print `True`; if not, `git branch -a` / check the open PRs and switch to the one that has it. (Ask Ken which branch to use if unsure.)
4. **Minimal venv for first contact** (do NOT `pip install -r requirements.txt` yet — it pulls PyQt6/pybullet/MAVSDK/a git dep you don't need and may fail on 3.14). PowerShell:
   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install pymavlink numpy opencv-python pytest
   ```
   If a wheel fails to build on 3.14, use `py -3.12 -m venv .venv` instead and record it in `FIRST_CONTACT_FINDINGS.md`. Install full `requirements.txt` only once Action Item 7 needs it.
5. Claude Code installed in the repo.

**Set the goal (one line):**
```
/goal Fly the killallhumans stack through ALL official AI-GP VQ1 gates in order with zero crashes (VQ1 is scored on completion, not speed). Build a live pymavlink CompetitionInterface adapter, drive the sim's runtime gate map + telemetry + camera into the race stack, recalibrate the drone model to the real sim, keep the ML residual OFF, and drop race_01/PyBullet-specific tuning. Stop and report at the first clean full-course completion.
```

**Start the loop (MUST be one line — a bare `/loop` is a usage error):**
```
/loop Follow ./AIGP_WINDOWS_LOOP_PROMPT.md — execute the LOOP BODY. First read LOOP_STATE.json (create it if missing). HUMAN GATE: if last_item_completed >= 3 AND NOT (human_gate_cleared==true OR Test-Path .loop/ALLOW_LIVE_ADAPTER), write LOOP_STOPPED.md ("awaiting human gate") and END the iteration — do NOT start Items 4-7. Otherwise: re-read this file, pick exactly ONE Action Item (the next incomplete one), do only that item, run its gate, update LOOP_STATE.json, commit (and push, once) only on green, then END the iteration. If blocked/ambiguous, STOP and write the question to FIRST_CONTACT_FINDINGS.md instead of guessing. Also stop if iteration>=20 or >8h since wall_start_iso.
```

**`LOOP_STATE.json` (create on first iteration, update every iteration):**
```json
{"iteration": 0, "wall_start_iso": "<set once>", "last_item_completed": 0, "human_gate_cleared": false}
```
- **Hard stop** at `iteration >= 20` or 8 h since `wall_start_iso`; write `LOOP_STOPPED.md` and end.
- **One Action Item per iteration.**
- **Items 4+ forbidden while `human_gate_cleared` is false** AND `.loop/ALLOW_LIVE_ADAPTER` absent (Rule 9). **Only Ken sets these** — the agent must never set/flip them.
- **Items 5–7 require live-sim evidence** (a quoted log line / JSONL path in the commit message). Passing pytest is NEVER a flight-gate completion.

---

# LOOP BODY

## 1. What
- **Repo:** `killallhumans` (this worktree). Branch holding this file per prereq 3. Work on a child branch, created idempotently (PowerShell): `git switch -c aigp/live-sim-bringup; if ($LASTEXITCODE -ne 0) { git switch aigp/live-sim-bringup }`.
- **Task:** make our autonomy stack (`race_pipeline.py` → EKF → SE(3) tracker) fly the **official AI-GP sim** by building the missing **live pymavlink transport adapter** + a bring-up script that owns the flight loop (wiring the sim's runtime gate map, telemetry, camera, race-status, collisions); then recalibrate to real physics and get a VQ1 completion.
- **Scope:** VQ1 = **completion only** (all gates, in order, no crash, ≤8 min). Do **not** optimize for speed yet (VQ2).

## 2. Reference impl (READ THIS FIRST)
- **`PLAN.md`** — §1 = exact interface contract; §4 = open-questions checklist; §5 = phase plan. This LOOP BODY executes PLAN Phases 0→2.
- **PR #9** landed the scaffolding (do **not** re-derive):
  - `competition/aigp_messages.py` — parsers (race-status `"<BQqqIq"`, track gate `"<Hfffffffff"`, `TrackInfoReassembler`); `COLLISION_ID_GATE=1001`, `COLLISION_ID_ENVIRONMENT=1002`.
  - `competition/track_data.py` — `track_data_to_gatespecs()`; tested in `competition/tests/test_track_data.py`.
  - `competition/aigp_recorder.py` — passive MAVLink → JSONL recorder. **Limitations (you fix in Item 3):** records `track_data` as only `{"num_gates": N}`, no vision, sends no setpoints (so it cannot tell you whether the sim accepts `SET_ATTITUDE_TARGET`).
  - `competition/vision_udp.py` — `VisionUdpListener` (port 5600; default `bind_host="0.0.0.0"` → **construct with `bind_host="127.0.0.1"`** single-machine); `latest_frame()` returns the freshest `CameraFrame`.
- **Official kit** at `$env:USERPROFILE\AIGP\PyAIPilotExample\` (external): `{main,setup,mavlink_rx,vision_rx,timesync,controller}.py`. `setup.py` connects `mavutil.mavlink_connection('udpin:<ip>:14550')` → `wait_heartbeat()` → `recv_match()`; `controller.py` shows arm (`MAV_CMD_COMPONENT_ARM_DISARM`), `SIM_RESET` (31000), `SET_ATTITUDE_TARGET`.
- **Closest adapters to mirror:** `competition/mavlink_bridge.py` (MAVSDK — **do not copy its offboard/connect semantics**) and `competition/pybullet_adapter.py`. You write a **third** `CompetitionInterface` (`competition/adapter.py:154`) using **pymavlink**.

## 3. Context — interface contract (full detail in `PLAN.md §1`)
- MAVLink 2 over UDP, **`udpin:127.0.0.1:14550`**. Sim→client: `HEARTBEAT, ATTITUDE, HIGHRES_IMU, TIMESYNC, LOCAL_POSITION_NED, ODOMETRY, COLLISION`, + `ENCAPSULATED_DATA` (race-status + chunked track-info) and `DATA_TRANSMISSION_HANDSHAKE`. Client→sim: `SET_ATTITUDE_TARGET`, `SET_POSITION_TARGET_LOCAL_NED`, arm, `SIM_RESET` (31000).
- Constants in `competition/aigp_geometry.py` (gate 1.5 m interior; camera 640×360, fx=fy=320, cx=320, cy=180, +20° up-tilt; physics 120 Hz, cmd <100 Hz).
- Frames: internal NED; origin = arm point. No GPS.
- Tracker emits `AttitudeCommand(roll_rad, pitch_rad, yaw_rad, thrust)` (`competition/adapter.py:109-112`). Send `SET_ATTITUDE_TARGET` with the quaternion (from roll/pitch/yaw) + `thrust`, clearing `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`.

## 4. Platform / domain specifics
- **Transport = raw pymavlink, port 14550** — NOT MAVSDK, NOT 14540. Custom `ENCAPSULATED_DATA`/`COLLISION`/`SIM_RESET` are surfaced by raw pymavlink only.
- **Vision** = separate UDP stream on 5600 (`VisionUdpListener(bind_host="127.0.0.1")`).
- **Gate map arrives at runtime** via the track-info packet → `track_data_to_gatespecs()` → `pipeline.configure(gates=…)`. **Never read `sim_pybullet/configs/*.json` for gates.**
- **Sequencing + crashes come from the sim:** race-status `active_gate_index`, `COLLISION` (`id` 1001=gate, 1002=environment). The pipeline doesn't consume these — you wire them in the flight loop (§5).

## 5. Adapter contract + bring-up (build these; they don't exist yet)
**`competition/aigp_mavlink.py` → `class AIGPMavlinkInterface(CompetitionInterface)`** — implement **all** ABC methods (`competition/adapter.py:162-210`): `connect, disconnect, arm, start_offboard, stop_offboard, get_telemetry, get_camera_frame, send_attitude, send_attitude_rate, send_position, is_connected, is_armed`. Notes:
- `connect(address="udpin:127.0.0.1:14550")` — parse the pymavlink URL; `wait_heartbeat()`; start the RX thread + `VisionUdpListener(bind_host="127.0.0.1")`. **Idempotent** — a second `connect()` is a no-op (no duplicate RX thread / no re-bind of UDP 5600).
- `get_telemetry()` — `ATTITUDE`+`HIGHRES_IMU` + (`LOCAL_POSITION_NED`/`ODOMETRY` if Item-3 says populated; else dead-reckon, Rule 4).
- `get_camera_frame()` → `VisionUdpListener.latest_frame()`. `send_attitude(cmd)` → `SET_ATTITUDE_TARGET` (quaternion+thrust).
- `start_offboard()` — AIGP uses raw `SET_ATTITUDE_TARGET`, not MAVSDK offboard: either a no-op (with a comment proving setpoints are accepted with no mode switch — confirm in Item 3) or "stream neutral `SET_ATTITUDE_TARGET` at ≥10 Hz until armed". **Must not require MAVSDK.**
- **Side channels** (not in `TelemetryState`): `async wait_for_track_data(timeout_s) -> TrackData`, `get_race_status() -> RaceStatus | None`, `get_collisions() -> list[int]` (drains queued `COLLISION.id`s).

**`scripts/aigp_live_race.py` — own the flight loop; do NOT call bare `RacePipeline.run()`** (`RaceSession.run()` re-`connect()`s and has no hook for the side channels). Because you bypass `run()`, you must reproduce the **startup lifecycle it owns** (`competition/session.py` connect→arm→start_offboard→`sequencer.start()`→timer init→loop). Add two thin, unit-tested helpers to `RacePipeline` rather than hand-duplicating EKF/tracker math: `begin_live_race(first_telem)` (calls `sequencer.start()` + initializes the race/sim-time timers `_race_start_time`/`_race_start_sim_time_s`/`_ref_progress_time`) and `step(telem, frame) -> AttitudeCommand` (the per-tick body of `_control_callback`, `race_pipeline.py:327`).
```python
iface = AIGPMavlinkInterface()
await iface.connect("udpin:127.0.0.1:14550")            # idempotent
track = await iface.wait_for_track_data(timeout_s=120)  # log num_gates; abort if None
pipeline = RacePipeline(iface, PipelineConfig(max_speed=1.5, target_hz=50.0))
assert pipeline.tracker.config.use_residual is False                  # Rule 1
pipeline.tracker.config.max_tilt_rad = 0.35                           # Rule 3 (run() does NOT set these)
pipeline.tracker.config.max_thrust_normalized = 0.65
first = await iface.get_telemetry()
pipeline.configure(gates=track_data_to_gatespecs(track), start_position=first.position_ned)
await iface.arm(); await iface.start_offboard()
pipeline.begin_live_race(first)            # sequencer.start() + timer init — else update() ignores ticks (WAITING)
period = 1.0 / 50.0
while True:
    telem = await iface.get_telemetry()
    for cid in iface.get_collisions():
        if cid == COLLISION_ID_GATE and pipeline.sequencer.current_gate:
            pipeline.sequencer.mark_collision(pipeline.sequencer.current_gate.gate_id, position=telem.position_ned)
        elif cid == COLLISION_ID_ENVIRONMENT:
            break_and_disarm("environment collision")
    rs = iface.get_race_status()                         # may be None — guard before use
    if (rs and rs.race_finished) or pipeline.sequencer.state in (RaceState.COMPLETED, RaceState.TIMED_OUT, RaceState.DISQUALIFIED):
        break
    if sim_elapsed_s(telem) > 8 * 60:                    # VQ1 cap (use sim time, Rule 6)
        break
    cmd = pipeline.step(telem, frame=await iface.get_camera_frame())
    await iface.send_attitude(cmd)
    if rs: log(rs.active_gate_index)                     # ground-truth progress
    if telem_stale(telem) > 0.5 or thrust_pinned(cmd, 0.65, 0.5):
        break_and_disarm("safety abort")
    await asyncio.sleep(period)                          # 50 Hz pacing (or gate on new telemetry)
await iface.stop_offboard(); await iface.disconnect()    # stop setpoints, land/disarm on exit
```

## 6. Hard rules (each has already-known failure modes — do not skip)
1. **ML residual stays OFF.** `use_residual=False` default (`control/mpc_tracker.py:109`); no `control/residual_weights.npz` exists; kinematic-bench-fit → iter-031 score **−1e6**, broke slalom. **Do NOT** set `use_residual=True`, run `scripts/train_tracker_residual.py` / `scripts/init_residual_weights.py`, or set `trace_tracker_features=True` on the live sim. Keep `assert pipeline.tracker.config.use_residual is False`. `tests/test_residual_matrix_gain.py` SKIPping (no weights) is expected — "green" = "skipped," not validated.
2. **Gate map from the sim, never JSON.** `track_data_to_gatespecs()` on the runtime packet. `race_01.json` (1.2 m openings, hand-built helix) is wrong for AIGP (1.5 m). Don't edit `race_01.json`.
3. **Drone model + gains are kinematic-sim artifacts; `calibration.py` is a STUB.** `feedforward_accel=0.50`/`drag_coefficient=0.0` are bench-tuned (`control/mpc_tracker.py:59-70`); `drone_spec` mass=1 kg/thrust=20 N are "NOT verified" (`competition/drone_spec.py:26-32`). **First-flight clamps (set explicitly — run() does NOT apply them):** `PipelineConfig(max_speed=1.5)` (default 8.0), `pipeline.tracker.config.max_tilt_rad=0.35` (default 0.85), `pipeline.tracker.config.max_thrust_normalized=0.65` (default 0.95). Abort if telemetry stale >0.5 s, thrust pinned >0.65 for >0.5 s, or any `COLLISION`. Raise caps only after a documented calibration.
4. **Record before you build; the state path forks on what's populated.** From Item-3 JSONL, confirm whether `LOCAL_POSITION_NED`/`ODOMETRY` carry real (non-zero) local position. Yes → feed the EKF. Absent/zero → dead-reckon from velocity + gate-PnP against the known map. Report which in `FIRST_CONTACT_FINDINGS.md`.
5. **Drop only the *overfit*, don't delete generic logic.** Override race_01/fixture/ILC constants + PyBullet-proven values — but don't delete generic curvature/S-turn/helix detection in `planning/trajectory_optimizer.py`. Set the planner speed envelope from calibrated limits via `PlannerConfig(plan_max_speed_mps=…)` (default 4.0 is a CF2X artifact, `:634`); ignore `race_01.json` `ilc_section_overrides`; recalibrate the detector (`gate_detection/src/phase1_detector.py:52-65`: `saturation_threshold=60`, `brightness_threshold=200`, `image_height=480`→360, `GATE_*_METERS=1.0`→1.5) via `gate_detection/src/color_calibrator.py`. Only disable a heuristic if a live-sim run shows it harmful.
6. **Sim time, not wall clock** (pipeline still uses wall clock in places, e.g. `_maybe_replan` near `race_pipeline.py:408-425`). Item-7 refactor: drive timeout/replan/logging off `TelemetryState.timestamp_us` (or race-status sim time) when populated; wall clock only for sleep/backoff. Unit-test with non-realtime timestamps.
7. **Stop-and-ask + no fabrication.** If telemetry is absent, the sim rejects a setpoint, or the spec is ambiguous, **STOP the iteration** and write the blocker to `FIRST_CONTACT_FINDINGS.md`. Never invent constants. Items 5–7 are "done" only with a live log/JSONL artifact quoted in the commit — never from pytest alone.
8. **Git discipline.** Idempotent branch. Commit each green gate; update `LOOP_STATE.json`. **Push at most once per iteration**; if push fails, commit locally + write `PUSH_FAILED.txt` and continue — no retry loop. **Never force-push, never delete branches/history.**
9. **Human gate after first contact.** After **Item 3** (findings committed), **halt** until **Ken** sets `human_gate_cleared: true` in `LOOP_STATE.json` OR creates `.loop/ALLOW_LIVE_ADAPTER`. **The agent must NOT create that file or flip that flag.** Do not start **Item 4** (the live adapter) before then. The `/loop` payload enforces this each iteration.

## 7. Smoke / run commands (repo root; PowerShell)
```powershell
git switch -c aigp/live-sim-bringup; if ($LASTEXITCODE -ne 0) { git switch aigp/live-sim-bringup }
$env:PYTHONPATH = (Get-Location).Path
python -m pytest competition/ tests/test_vision_udp.py tests/test_vision_udp_listener.py -q
#   (do NOT run bare repo-root `pytest` — it collects 300+ unrelated tests, some needing pybullet)

# FIRST CONTACT — MAVLink stream (recorder is MAVLink-only today; ~60 s with the sim running)
python -m competition.aigp_recorder --out FIRST_CONTACT.jsonl --duration 60
python -c "import json,collections; print(collections.Counter(json.loads(l)['type'] for l in open('FIRST_CONTACT.jsonl')))"
python -c "import json; [print(l.strip()) for l in open('FIRST_CONTACT.jsonl') if json.loads(l)['type'] in ('LOCAL_POSITION_NED','ODOMETRY','track_data','race_status','COLLISION')][:20]"
```
(cmd: `set "PYTHONPATH=%CD%"` then same `python …`. Git-Bash: `PYTHONPATH=. python …`.) Confirm heartbeat connects; whether `LOCAL_POSITION_NED`/`ODOMETRY` are present **and non-zero**; whether a `track_data` record appears. Record in `FIRST_CONTACT_FINDINGS.md`.

## 8. Action items (one per loop iteration; each ends at its gate)
1. **Setup & baseline.** Branch; minimal venv (prereq 4); pytest baseline green; `FlightSim.exe` running + logged in. Gate: tests green + sim up.
2. **Stock-pilot smoke (PLAN §0.2).** Run `python "$env:USERPROFILE\AIGP\PyAIPilotExample\main.py"` **unmodified** ~30 s (its own deps); confirm heartbeat connects and arm + a `SET_ATTITUDE_TARGET`/motor command **moves the drone**. Gate: stock pilot moves the drone. **If it fails (firewall/login/port), STOP — fix the environment before custom code.**
3. **First contact + recorder upgrade.** Extend `competition/aigp_recorder.py` so `track_data` records carry every gate `{gate_id, position_ned, orientation_wxyz, width, height}` (or raw base64 chunks) AND add a vision capture via `VisionUdpListener` (log frame size/FPS). Run it; write `FIRST_CONTACT_FINDINGS.md` answering all PLAN §4 questions + "did the stock pilot's `SET_ATTITUDE_TARGET` move the drone?" + vision FPS. Gate: findings committed; replay fixture can reconstruct `TrackData` (not just `num_gates`). **Then HALT for the human gate (Rule 9) — do not start Item 4.**
4. **Build `AIGPMavlinkInterface`** per §5 (all ABC methods + idempotent connect + `wait_for_track_data`/`get_race_status`/`get_collisions`) and the `begin_live_race`/`step` helpers on `RacePipeline`. TDD against the `FIRST_CONTACT.jsonl` replay fixture: (a) `TelemetryState` from ATTITUDE/IMU/LOCAL_POSITION_NED/ODOMETRY; (b) `wait_for_track_data()` → round-trip through `track_data_to_gatespecs`; (c) `get_race_status()`/`get_collisions()`; (d) `step()`/`begin_live_race()`. A tiny JSONL replayer suffices — no live sim for unit tests. Gate: new unit tests RUN green. (Requires the human gate cleared.)
5. **Connect-smoke ONLY (no flight).** `await iface.connect("udpin:127.0.0.1:14550"); await iface.get_telemetry(); await iface.wait_for_track_data(timeout_s=120)`. **Do NOT fly.** Gate: live telemetry + a full track-data packet received and logged; zero motor commands sent.
6. **First live flight** via `scripts/aigp_live_race.py` (§5 own-loop: reuse the connected `iface`, arm→offboard→`begin_live_race`→50 Hz loop, Rule-3 clamps, collision/active_gate wiring, abort conditions). Gate: a live log shows `active_gate_index ≥ 1` after arm (quote it in the commit). PyBullet smokes do NOT satisfy this. Commit + push (once).
7. **Calibrate + de-overfit + sim-time.** Extend `calibration.py` with a real live thrust/drag collection+fit (document thresholds); set `drone_spec`; re-tune gains conservatively; apply Rule 5 (parameterize, don't delete) and Rule 6 (sim time). Gate: stable hover + step response within a stated tolerance; then iterate to **all gates in order, no crash, ≤8 min** (the `/goal`). **STOP and report.**
8. **Persist.** Finalize `FIRST_CONTACT_FINDINGS.md` and add `docs/aigp/<date>-live-bringup.md` (telemetry availability, calibrated numbers, gotchas, evidence). Push `aigp/live-sim-bringup`, open a PR, tell Ken.

## 9. Gotchas
- **Run pytest from repo root** with `$env:PYTHONPATH=(Get-Location).Path`. No `pytest.ini`/`conftest.py` — imports resolve only with root on `sys.path`. Don't run bare repo-root `pytest`.
- **Python 3.14.2** per the kit; if `pymavlink`/`opencv-python` wheels fail, use a 3.12 venv (cv2 needed for vision decode).
- **The recorder is passive + MAVLink-only today** (Item 3 upgrades it). It cannot prove the sim accepts your setpoints — that's the Item-2 stock-pilot smoke.
- **`active_gate_index`/`COLLISION` aren't in `TelemetryState`** — they come via the §5 side channels; the pipeline ignores them until the §5 own-loop wires them.
- **Gate dimensions:** trust the track packet + `aigp_geometry.py` (1.5 m), not `phase1_detector.py`'s 1.0 or `race_01.json`'s 1.2.
- If `LOCAL_POSITION_NED` is all-zero/absent → no-GPS branch: dead-reckon + gate-PnP (Rule 4); report it.

---
*This runbook is precise on purpose. When in doubt, prefer Rule 7 (stop and report) over a plausible guess — Ken would rather answer a question than debug a fabricated constant on the real drone.*
