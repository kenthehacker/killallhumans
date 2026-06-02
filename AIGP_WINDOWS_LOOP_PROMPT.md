# AI-GP — Windows Sim Autonomous Loop Prompt

**For:** a teammate running this on a **Windows PC with the official AI-GP simulator** on our behalf, driving **Claude Code** with `/goal` + `/loop`.
**Repo:** `killallhumans`, branch **`aigp-vq1-loop`** (this branch — it has all the AIGP code + `PLAN.md`).
**You are continuing real work, not starting fresh.** Read `PLAN.md` (the port plan) and the "Reference" section below before doing anything.

---

## How to run this

Prereqs on the Windows machine:
1. Official AI-GP sim installed and **running** (`FlightSim.exe`), logged in to the virtual qualifier (sim account creds — ask Ken).
2. This repo cloned, on branch `aigp-vq1-loop`. `pip install -r requirements.txt` (Python 3.14.2 per the official kit; `pymavlink`, `opencv-python`, `numpy` are the load-bearing deps).
3. Claude Code installed in the repo.

Then, in Claude Code:

```
/goal Fly the killallhumans stack through ALL official AI-GP VQ1 gates in order with zero crashes (VQ1 is scored on completion, not speed). Build a live pymavlink CompetitionInterface adapter, drive the sim's runtime gate map + telemetry + camera into RacePipeline, and recalibrate the drone model to the real sim. Keep the ML residual OFF and drop all race_01/PyBullet-specific tuning. Stop and report at the first clean full-course completion.
```

```
/loop
Follow ./AIGP_WINDOWS_LOOP_PROMPT.md — execute the "LOOP BODY" below. Each iteration: re-read this file, determine the current Action Item from git state + the findings report, do exactly that one item, run its gate (tests/smoke), commit + push on green, then stop the iteration. If you hit a decision you cannot infer (missing telemetry, sim rejects setpoints, ambiguous spec), STOP and write the question to FIRST_CONTACT_FINDINGS.md instead of guessing.
```

Self-paced loop (no interval) is correct — this is a build task, not a poll. Do **one Action Item per iteration**.

---

# LOOP BODY

## 1. What
- **Repo / branch:** `killallhumans` @ `aigp-vq1-loop`. Work on a child branch: `git checkout -b aigp/live-sim-bringup`.
- **Task:** make our existing autonomy stack (`race_pipeline.py` → EKF → SE(3) tracker) fly the **official AI-GP sim** by building the missing **live pymavlink transport adapter** and wiring the sim's runtime gate map + telemetry + camera into it. Then recalibrate the drone model to real physics and get a VQ1 completion.
- **Scope:** VQ1 = **completion only** (all gates, in order, no crash, ≤8 min). Do **not** optimize for speed (that's VQ2, later).

## 2. Reference impl (READ THIS FIRST)
- **`PLAN.md`** (this repo) — the full port plan. §1 is the exact official interface contract; §4 is the open-questions checklist; §5 is the phase plan. This LOOP BODY executes PLAN §5 Phases 0→2.
- **PR #9** (`feat(competition): AIGP official interface…`) — landed the scaffolding you will build on:
  - `competition/aigp_messages.py` — byte-exact parsers for the sim's custom payloads (race-status `"<BQqqIq"`, track-info gates `"<Hfffffffff"`, `TrackInfoReassembler`). **Use these; do not re-derive the wire format.**
  - `competition/track_data.py` — `track_data_to_gatespecs()` (sim gate map → `GateSpec` list).
  - `competition/aigp_recorder.py` — the first-contact black-box recorder.
  - `competition/vision_udp.py` — the camera receiver (port 5600, already spec-compliant). **Reuse `VisionUdpListener`.**
- **Official template** (Ken has it; the byte formats already live in `aigp_messages.py`): `PyAIPilotExample/{main,setup,mavlink_rx,vision_rx,timesync,controller}.py`. `main.py`/`setup.py` connect with `mavutil.mavlink_connection('udpin:<ip>:14550')` → `wait_heartbeat()` → `recv_match()`.
- **Closest existing adapter to copy structure from:** `competition/mavlink_bridge.py` (`MAVLinkBridge(CompetitionInterface)`, MAVSDK) and `competition/pybullet_adapter.py` (`PyBulletAdapter`). You are writing a **third** implementation of the same `CompetitionInterface` ABC (`competition/adapter.py:154`), using **pymavlink** instead of MAVSDK.

## 3. Context
- Interface contract (authoritative), all in `PLAN.md §1`: MAVLink 2 over UDP `udpin:…:14550`; sim→client `HEARTBEAT, ATTITUDE, HIGHRES_IMU, TIMESYNC, LOCAL_POSITION_NED, ODOMETRY, COLLISION`, plus `ENCAPSULATED_DATA` (race-status + chunked track-info) and `DATA_TRANSMISSION_HANDSHAKE`; client→sim `SET_ATTITUDE_TARGET`, `SET_POSITION_TARGET_LOCAL_NED`, arm, custom `SIM_RESET` cmd 31000.
- Geometry/timing constants: `competition/aigp_geometry.py` (gate 1.5 m interior / 2.7 m outer / 0.26 m depth; camera 640×360, fx=fy=320, cx=320, cy=180, **+20° up-tilt**; physics 120 Hz, cmd <100 Hz). **Import from there; never hard-code.**
- Frames: everything internal is **NED**. NED origin = the point where the drone armed (local, not global). No GPS/global position.

## 4. Platform / domain specifics
- **Transport is pymavlink, port 14550** — NOT the MAVSDK `mavlink_bridge.py` (which uses port 14540). The sim's custom `ENCAPSULATED_DATA`/`COLLISION`/`SIM_RESET` messages are surfaced cleanly by raw pymavlink, not by MAVSDK.
- **Vision is a separate UDP stream on 5600** (JPEG, 24-byte header) — already handled by `competition/vision_udp.py`. Bind `0.0.0.0:5600`.
- **The gate map arrives at runtime** from the sim's track-info packet → `track_data_to_gatespecs()` → `RacePipeline.configure(gates=...)`. **Do not read `sim_pybullet/configs/*.json` for gate positions** — those are practice fixtures.
- **Sequencing + crashes come from the sim:** race-status `active_gate_index` (ground-truth next gate) and `COLLISION` (`id` 1001=gate, 1002=environment). Prefer these over pure geometric inference.
- **Control:** `RacePipeline`'s SE(3) tracker emits `AttitudeCommand(roll,pitch,yaw,thrust)`. Send it as `SET_ATTITUDE_TARGET` with the **attitude quaternion + thrust** (clear the `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` bit). Body-rate mode is a VQ2 optimization, not now.

## 5. Hard rules (each has already-known failure modes — do not skip)
1. **Keep the ML residual OFF (`use_residual=False`).** It is **untrained** (no `control/residual_weights.npz` exists; default resolves to zero-init = no-op) and what training exists was fit to the *kinematic bench*, not real physics — iter-031 closed-loop score was **−1e6 (hard fail)** and it **broke slalom completion**. On the real drone its residuals are meaningless or harmful. Do **not** enable it; do not train it on practice data. Re-enable only after recollecting a dataset against the **real sim** AND `tests/test_residual_matrix_gain.py` goes green. (`control/mpc_tracker.py:109`; `.loop/synthesis/iter_031_ml_result.md`)
2. **Gate map from the sim, never from JSON.** Build `GateSpec`s via `track_data_to_gatespecs()` on the runtime track-info packet. `sim_pybullet/configs/race_01.json` uses 1.2 m gate openings and a hand-built helix — **wrong** for AIGP (1.5 m, unknown layout). (`competition/track_data.py:25`; `sim_pybullet/configs/race_01.json`)
3. **Controller gains + drone model are kinematic-sim artifacts — recalibrate, don't trust.** `feedforward_accel=0.50` and `drag_coefficient=0.0` are tuned to a fake `drag=0.5` bench; `drone_spec` `mass=1 kg`/`thrust=20 N`/`drag=0.5` are explicit "NOT verified" placeholders. Fly **conservatively** until you identify real thrust/drag via `competition/calibration.py`, then set `drone_spec` and re-tune. (`control/mpc_tracker.py:59-70`; `competition/drone_spec.py:26-32,91-102`)
4. **Record before you build.** Run `competition/aigp_recorder.py` and confirm from the JSONL **which telemetry the sim actually populates** (is `LOCAL_POSITION_NED`/`ODOMETRY` non-zero? is track-info sent? what do `COLLISION`s look like?) BEFORE designing the adapter's state path — the whole estimator design forks on this (PLAN §4). If local position IS provided, feed it to the EKF; if not, dead-reckon from velocity + correct with gate PnP against the known map.
5. **Drop race_01 / track-topology overfit before flying the real course.** Disable helix/S-turn name-based special-casing and the `plan_max_speed_mps=4.0` CF2X ceiling in `planning/trajectory_optimizer.py:1022-1058,615-635`; ignore `race_01.json` `ilc_section_overrides`; the detector's `saturation_threshold=60`/`brightness_threshold=200`/`image_height=480`/`GATE_WIDTH_METERS=1.0` (`gate_detection/src/phase1_detector.py:52-65`) are VQ1-scene guesses — recalibrate against the real render (use `gate_detection/src/color_calibrator.py`) and set width/height from the track packet (1.5 m).
6. **Use sim time, not wall clock**, for the control loop and logging — `TIMESYNC` + the `sim_time_ns` in vision frames are authoritative; the sim may not run real-time.
7. **Stop-and-ask, don't guess.** If telemetry you need is absent, the sim rejects a setpoint type, or the spec is ambiguous, **STOP the iteration** and write the blocker to `FIRST_CONTACT_FINDINGS.md`. Never invent constants or fabricate "it worked."
8. **Commit + push every green gate.** After each Action Item passes its gate, `git commit` + `git push` on `aigp/live-sim-bringup`. Keep live runs ≤ 8 min (VQ1 cap). Never command continuous max thrust.

## 6. Smoke test / run commands (copy-paste; run from the repo root)
```bash
# Branch for this work
git checkout aigp-vq1-loop && git pull && git checkout -b aigp/live-sim-bringup

# Install + green baseline (run from repo root so 'competition'/'control' import)
pip install -r requirements.txt
PYTHONPATH=. python -m pytest competition/ tests/test_vision_udp.py tests/test_vision_udp_listener.py -q
#   (Windows cmd: set PYTHONPATH=%CD%  &&  python -m pytest ... )

# FIRST CONTACT: record the live sim's message stream to JSONL (~60 s with the sim running)
PYTHONPATH=. python -m competition.aigp_recorder --out FIRST_CONTACT.jsonl --duration 60
#   then inspect what actually arrived populated:
PYTHONPATH=. python -c "import json,collections; c=collections.Counter(json.loads(l)['type'] for l in open('FIRST_CONTACT.jsonl')); print(c)"
#   confirm local position is real (not all-zero) and track-data was delivered:
PYTHONPATH=. python -c "import json; [print(l.strip()) for l in open('FIRST_CONTACT.jsonl') if json.loads(l)['type'] in ('LOCAL_POSITION_NED','ODOMETRY','track_data','race_status','COLLISION')][:20]"
```
Expected on success: heartbeat connects; the Counter shows `ATTITUDE`, `HIGHRES_IMU`, and (hopefully) `LOCAL_POSITION_NED`/`ODOMETRY` and a `track_data` record. Record the answers in `FIRST_CONTACT_FINDINGS.md`.

## 7. Action items (one per loop iteration; each ends at its gate)
1. **Setup & baseline.** Clone/branch, `pip install`, run the pytest baseline above to green, confirm `FlightSim.exe` is running + logged in. Gate: tests green + sim up. Commit nothing (no code change yet); proceed.
2. **First contact (PLAN §0.3 / §4).** Run `aigp_recorder.py` against the live sim; write `FIRST_CONTACT_FINDINGS.md` answering every PLAN §4 question (is `LOCAL_POSITION_NED`/`ODOMETRY` populated? is `track_data` sent? what are `COLLISION`/`active_gate_index` semantics? does the sim accept `SET_ATTITUDE_TARGET`?). Gate: findings file committed. **STOP and report the findings** before building.
3. **Build the live adapter** `competition/aigp_mavlink.py` → `class AIGPMavlinkInterface(CompetitionInterface)`: pymavlink `udpin:0.0.0.0:14550`; RX thread parsing the message set into `TelemetryState` (use `aigp_messages` for ENCAPSULATED_DATA + `TrackInfoReassembler`; reuse `VisionUdpListener` for `get_camera_frame`); `send_attitude` → `SET_ATTITUDE_TARGET` (quaternion+thrust); arm + `SIM_RESET`. **TDD against the `FIRST_CONTACT.jsonl` replay fixture** (a small JSONL replayer is sufficient — do not require the live sim for unit tests). Gate: new unit tests green; replaying the fixture reproduces the `TelemetryState` stream.
4. **Wire `RacePipeline`.** On receiving the track-info packet, `track_data_to_gatespecs()` → `RacePipeline.configure(gates=...)`; `RacePipeline.run(address=...)` driving `AIGPMavlinkInterface`. Keep `use_residual=False`. State source per Action Item 2 findings (Rule 4). Gate: pipeline constructs + connects against the live sim without exception (no flight yet).
5. **First live closed-loop run.** Arm, fly **conservatively** (low speed cap). Use `active_gate_index` to track progress and `COLLISION` to detect crashes. Gate: **drone passes ≥ 1 gate, in order, live.** Commit + push.
6. **Calibrate + de-overfit (Rule 3, Rule 5).** Identify real thrust/drag from the live telemetry (`competition/calibration.py`), set `drone_spec`, re-tune gains conservatively; disable the race_01/helix overfit and recalibrate the detector. Gate: stable hover + step response on the real sim; commanded-vs-achieved accel error within tolerance.
7. **Iterate to VQ1 completion.** Keep iterating perception/guidance until the drone clears **all** gates in order, no crash, ≤ 8 min. Gate: one clean full-course completion (the `/goal`). **STOP and report.**
8. **Persist.** Write `FIRST_CONTACT_FINDINGS.md` + a short `docs/aigp/<date>-live-bringup.md` report (what worked, real telemetry availability, calibrated drone numbers, gotchas). Push `aigp/live-sim-bringup` and open a PR. Tell Ken.

## 8. Gotchas
- **Run pytest from the repo root** with `PYTHONPATH=.` (Windows: `set PYTHONPATH=%CD%`). There is no `pytest.ini`/`conftest.py`; imports like `competition.*` resolve only if the root is on `sys.path`.
- **Python 3.14.2** per the official kit; ensure `pymavlink` + `opencv-python` wheels install on it (fall back to a 3.12 venv only if a wheel is missing — the pure parsers don't care, but `vision_udp` decode needs `cv2`).
- **The recorder's live `run()` is not covered by CI** — it's validated here, on first contact. Its pure core is unit-tested.
- **`tests/test_residual_matrix_gain.py` SKIPs** when `control/residual_weights.npz` is absent (normal on a clean checkout) — that is expected, not a failure. Do not "fix" it by training the residual (Rule 1).
- **Gate dimensions:** trust the track packet's width/height and `aigp_geometry.py` (1.5 m), not `phase1_detector.py`'s `GATE_*_METERS=1.0` or `race_01.json`'s 1.2 m.
- If the sim's `LOCAL_POSITION_NED` is **all zeros / absent**, that's the no-GPS branch — do not assume position; dead-reckon from velocity + gate-PnP (Rule 4). Report it.

---
*This prompt is precise on purpose. When in doubt, prefer Rule 7 (stop and report) over a plausible guess — Ken would rather answer a question than debug a fabricated constant on the real drone.*
