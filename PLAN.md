# AI Grand Prix — Official Competition Port Plan

> **HISTORICAL / SUPERSEDED (build 3364 and early VQ1 planning).** Do not use
> this file as an execution plan or current interface contract. The authoritative
> build-3385 VQ2 flight-status, interface, and safety handoff is
> `docs/aigp/2026-07-18-vq2-handoff.md`; the workflow handoff is
> `docs/2026-07-18-development-cycle-handoff.md`. Use `RUN.md`, `Agents.md`, and
> `docs/autonomous_iteration.md` for current commands and safety policy. The
> speculative dates, telemetry assumptions, host paths, and VQ1 priorities below
> are preserved only as development evidence.

> Archived workflow note: the checkbox plan below is not an instruction to
> agents working on the current build.

**Goal:** Take the autonomy stack we practiced with on PyBullet and make it fly the **official AI-GP simulator** end-to-end — VQ1 (completion) first, then VQ2 (fastest valid time).

**Architecture:** The official sim is a separate Windows/Unreal process. Our Python pilot connects over **MAVLink2/UDP (`udpin:…:14550`)** for telemetry + control and a **JPEG-over-UDP stream (port 5600)** for the FPV camera. The pilot keeps our existing `RacePipeline` (EKF → state-predict → trajectory → SE(3) tracker → attitude command); we **replace the transport layer** (MAVSDK → raw pymavlink to match the official template and to surface the sim's custom messages), **feed the gate map from the sim's runtime track-data packet** instead of hardcoded JSON, and **recalibrate the drone model/control gains against the real sim** (today's are overfit to a fake CF2X PyBullet model).

**Tech Stack:** Python 3.14.2, `pymavlink`, `opencv-python`, `numpy` (official kit deps); our existing `competition/`, `estimation/`, `control/`, `planning/`, `gate_detection/` modules; AWS EC2 GPU (Windows) to host the sim.

**Date:** 2026-06-01 · **Branch:** `aigp-vq1-loop` (worktree `~/Personal/killallhumans-aigp-vq1`) · Supersedes `plan_archive_2026-04-03_pybullet-gate7.md`.

**Deadline:** VQ1 and VQ2 are run as **one virtual phase** — VQ1 opened May and stays open until VQ2 closes, **~mid-to-late July 2026** (exact day emailed to registrants). So ~6–7 weeks for both qualifiers. Physical qualifier **Sept 2026** (SoCal); finals **Nov 2026** (Ohio). Register at `theaigrandprix.com` → sim download links + credentials arrive by email; team portal `https://teams.theaigrandprix.com`; **online login + broadband required at runtime**.

---

## 0. TL;DR — the strategic reframe

A read of the **official `PyAIPilotExample` template** (the authoritative contract, newer than the spec PDF) changes the picture dramatically:

- **This is an integration + calibration job, not a rebuild.** Our practice stack was built around two assumptions an earlier analysis feared were fatal — *known gate positions in a world frame* and *a usable local position*. The real interface **provides both**: the sim hands us the **gate map at runtime** (`on_track_data`: per-gate NED position + orientation + size) and has live handlers for **`LOCAL_POSITION_NED` / `ODOMETRY`** (local pose relative to the arm-point origin). "No GPS / no global coordinates" ≠ "no local position."
- The sim even gives us **gate sequencing for free** (`active_gate_index` + per-gate pass times in the race-status message) and **crash detection for free** (`COLLISION` message: gate vs environment, impulse magnitude).
- Our **SE(3) geometric tracker already outputs the right command type** — `SET_ATTITUDE_TARGET` accepts either attitude-quaternion+thrust *or* body-rates+thrust via a type-mask.
- We have already built a **spec-compliant vision receiver** (`competition/vision_udp.py`, port 5600, identical 24-byte header to the official `vision_rx.py`) and **codified the geometry constants** (`competition/aigp_geometry.py`). These drop straight in.

**The one thing that gates everything:** *which of those telemetry messages the real binary actually populates.* The template has handlers for them, but a handler is not a guarantee the sim sends data. **Only running the binary answers this** — and the sim is **Windows 11 + NVIDIA-only**, so step zero is standing up a GPU host and logging the real message stream.

**Biggest real risks (in priority order):** (1) telemetry availability unknown until first contact; (2) drone dynamics/control gains are overfit to a *fake* PyBullet model and will misbehave on real physics; (3) MAVSDK (what we built on) may not surface the sim's custom encapsulated messages — pymavlink (what the template uses) does; (4) VQ2 perception (3D-scanned, guidance off) is a genuinely hard CV problem.

---

## 1. The official interface contract (key reference)

Source of truth: official `PyAIPilotExample/` (extracted to `/tmp/aigp_pilot_example/`) + Technical Spec **VADR-TS-002 v0002** (cached `/tmp/aigp_spec_0002.pdf`). Where they disagree, **the template is newer (2026-05-28) — trust it, then verify live.**

### 1.1 Transport
| Channel | Detail |
|---|---|
| Control/telemetry | **MAVLink 2 over UDP**, `mavutil.mavlink_connection('udpin:<ip>:14550')`, `wait_heartbeat()` then `recv_match()`. Library = **pymavlink** (`c_library_v2` dialect). |
| Vision | **JPEG over UDP, port 5600**, chunked; 24-byte LE header `"<IHHIIQ"` = `frame_id u32, chunk_id u16, total_chunks u16, jpeg_size u32, payload_size u32, sim_time_ns u64`; reassemble by `frame_id`, `cv2.imdecode(..., IMREAD_COLOR)` → **color** 640×360 @ 30 Hz. |
| Time | Client sends `TIMESYNC` at 10 Hz (`tc1=now_ns, ts1=0`); sim replies. Authoritative clock = **sim time** (`sim_time_ns`, `sim_boot_time_ms`), not wall-clock. |
| ⚠ Drift vs our code | Our `competition/mavlink_bridge.py` uses **MAVSDK on port 14540**. Official = **pymavlink on 14550**. |

### 1.2 Messages the sim → client (from `mavlink_rx.py`)
- `HEARTBEAT` (armed flag), `ATTITUDE` (roll/pitch/yaw + rates), `HIGHRES_IMU` (accel+gyro), `TIMESYNC`.
- **`LOCAL_POSITION_NED`** (x,y,z + vx,vy,vz) and **`ODOMETRY`** (pos + quaternion + body rates) — *local* pose; **populate-or-not is the #1 open question.**
- `ACTUATOR_OUTPUT_STATUS` (4 motor outputs — useful for calibration).
- `COLLISION` (`id`: 1001=Gate, 1002=Environment; `threat_level` 1–2; impulse magnitude).
- `ENCAPSULATED_DATA` carrying two custom payloads:
  - **Race status** (`<BQqqIq`): `sim_boot_time_ms`, `race_start_boot_time_ms`, `race_finish_time_ns`, **`active_gate_index`**, `last_gate_race_time`.
  - **Track info** (chunked via repurposed `DATA_TRANSMISSION_HANDSHAKE`): **`num_gates`, then per gate `<Hfffffffff`** = `gate_id`, `position_ned_x/y/z`, `orientation_ned_w/x/y/z`, `width`, `height`.

### 1.3 Messages client → sim (from `controller.py`)
- `MAV_CMD_COMPONENT_ARM_DISARM` (arm), custom `command_long` **id 31000 = SIM_RESET**.
- **`SET_ATTITUDE_TARGET`** — quaternion + body roll/pitch/yaw rate + `thrust∈[0,1]`. Type-mask `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` selects **body-rate+thrust** (template default: pitch-rate −0.3, thrust 0.6). Clearing it sends **attitude-quaternion+thrust** (matches our SE(3) tracker).
- `SET_POSITION_TARGET_LOCAL_NED` — position and/or velocity setpoints in `MAV_FRAME_LOCAL_NED` (template example: 2 m/s forward velocity).
- `set_actuator_control_target` — direct motor control (advanced).
- Loop rate: template uses **250 Hz**; spec §4.4 says command **< 100 Hz**, physics 120 Hz, heartbeat ≥ 2 Hz. **Verify what the sim accepts.**

### 1.4 Geometry / rules (VADR-TS-002 + website)
- Frames **NED**; NED origin = fixed ground point where the drone armed. Camera shares body origin, **tilted +20° up**, pinhole **fx=fy=320, cx=320, cy=180** (the stated "VFoV 90°" contradicts the intrinsics ≈58.7° — **use the intrinsics**).
- Gates: outer 2700 mm, **inner passable 1500 mm**, depth 260 mm. Drone chassis 280×280×160 mm. Gates passed **in order**, **8-min** max run, **unlimited attempts**, course **downloaded in advance & deterministic** (→ course-specific tuning is *legitimate* for the qualifiers).
- **VQ1**: <10 gates, desaturated + highlighted gates, **scored on completion**. **VQ2**: <20 gates, 3D-scanned, lighting/distractions, **visual guidance off**, **scored on fastest valid time**. Both share the interface.
- Runtime: **Windows 11 only**, Python 3.14.2 verified. Mass/TWR/top-speed of the real drone are **not published** → must be system-identified from the sim.

---

## 2. Reuse map — what ports easily vs what's real work

### 2.1 ✅ Easy ports (reuse with little/no change)
| Asset | Path | Note |
|---|---|---|
| Vision UDP receiver | `competition/vision_udp.py` | Already byte-identical to official `vision_rx` (port 5600, 24-B header) + 24 tests. Just wire into the new transport. |
| AIGP geometry constants | `competition/aigp_geometry.py` | Gate sizes, camera intrinsics, timing, FoV reconciliation — already codified from the spec. |
| Interface contract | `competition/adapter.py` | `CompetitionInterface` ABC + `TelemetryState`/`AttitudeCommand`/`CameraFrame` dataclasses — keep as the seam. |
| PnP solver + known-gate→drone-pose | `estimation/gate_pnp.py` | `solvePnPRansac` from 4 corners, and "known gate world pose → infer drone pose" — **exactly** the localization primitive for known-map + vision. |
| Image-space gate track + latency predict | `estimation/gate_tracker.py`, `estimation/state_predictor.py` | Map-agnostic; directly reusable. |
| SE(3) geometric tracker | `control/mpc_tracker.py` | Outputs `AttitudeCommand`; reuse structure, **retune gains** (see §2.2). |
| Race orchestration | `race_pipeline.py` | Already interface-agnostic and already consumes `List[GateSpec]` — the track-data adapter slots straight in. |
| Gate detector | `gate_detection/src/phase1_detector.py` | Detects gates from the image; reuse algorithm, **retune thresholds** per environment. |
| Dynamic replanner + gate-region philosophy | `planning/dynamic_replanner.py`, `plan_archive_2026-04-03_pybullet-gate7.md` | When-to-replan policy + "perception → filtered state → gate-region guidance" carry forward. |

### 2.2 🔨 Significant work (the real porting effort)
1. **pymavlink transport adapter** (replaces MAVSDK) — handle the full message set incl. custom encapsulated race-status + track-data + `COLLISION` + `SIM_RESET`. *Biggest single piece.* Port **14550**.
2. **Runtime gate map** — parse track-data → `List[GateSpec]` → `RacePipeline.configure()`; stop reading gates from `sim_pybullet/configs/*.json`.
3. **Telemetry-source verification + EKF rewire** — if `LOCAL_POSITION_NED`/`ODOMETRY` are populated, use them; else dead-reckon from velocity + correct with known-gate PnP. Today the EKF is effectively a passthrough of ground-truth odometry.
4. **Drone calibration to the real sim** — system-identify thrust curve + drag from commanded thrust vs measured accel/`ACTUATOR_OUTPUT_STATUS`; fill `competition/drone_spec.py` (currently a `1 kg / 20 N` placeholder) and **de-overfit controller gains** (current `drag_coefficient=0.0`, `feedforward_accel=0.50` are *PyBullet artifacts*, explicitly noted as such — they will be wrong on real physics).
5. **Control output mapping** — `AttitudeCommand` → `SET_ATTITUDE_TARGET`: start with attitude-quaternion+thrust (matches tracker), evaluate body-rate+thrust for VQ2 speed; calibrate thrust normalization.
6. **Sim-time-driven loop** — drive control/logging on sim clock + TIMESYNC, not wall-clock.
7. **Perception robustness** — VQ1 highlighted-gate retune → VQ2 realistic/guidance-off (the hard problem; may need a learned detector / YOLO-pose).
8. **Telemetry logger + offline replay harness** — record the live MAVLink+vision stream; replay offline to develop without burning EC2 hours. *Build this in Phase 0; it pays for itself immediately.*
9. **End-to-end validation against the real binary** — never done; the whole point.

### 2.3 🗑 De-prioritize / throwaway for the official sim
- `sim_pybullet/*` (Crazyflie **CF2X** physics + `DSLPIDControl`) — keep only as a *secondary* local sanity check; it is the wrong vehicle and the wrong control interface. CLAUDE.md already says "do not treat its physics as ground truth."
- `planning/ilc_runtime.py` — learned offset keyed to a `sha256(race_01.json)` + CF2X; not transferable.
- Name-based track-topology special-casing (helix/S-turn) in `planning/trajectory_optimizer.py` — replace with principled curvature logic. (Caveat: because the VQ course is *known and fixed*, deliberate, clean per-course tuning is allowed — just don't bake it into name heuristics.)

---

## 3. Host — ShadowPC now → your own RTX PC later (decided 2026-06-01)

**Decision:** Use **ShadowPC** (cloud Windows 11 gaming desktop, flat ~$34–50/mo) as the sim host while away from your PC; switch to **your own RTX PC** (free, best latency, unlimited hours) once back with it. Keep **Claude Code + the full MCP stack on the Mac** — this is a *hybrid*, not a single box.

**Why this fits:**
- The sim is a **Windows Unreal-Engine (UE4/DX11) GPU binary** (`FlightSim.exe`). Rosetta/Conda only ran the *practice PyBullet* stack (pure Python); it cannot host a Windows DirectX GPU app — so the sim needs a real Windows+NVIDIA host.
- **ShadowPC is a full Windows 11 desktop with a real NVIDIA GPU** (Power tier ≈ RTX 3070 Ti / 20 GB — above the RTX 3070 the sim was tested on; the 640×360 render is trivially light). Install `FlightSim.exe` like on any PC.
- **Critical: the control loop is immune to streaming latency.** You run BOTH the sim and the Python pilot *on the ShadowPC*; MAVLink/UDP/vision are all localhost there. Only the *desktop image* streams to the Mac — so the 120 Hz physics / 250 Hz control loop never crosses the internet. Streaming lag only affects you watching/clicking.
- **Lower "phone-home" risk than a VM.** The sim requires online login + broadband (calls home via its Tencent PGOS backend). ShadowPC presents as a normal NVIDIA gaming PC, so it's far less likely to be rejected than a Parallels Windows-on-ARM VM (§3.3).

**Hybrid topology (keep your MCPs intact):**
- **Mac** = brain: Claude Code + all MCP servers + offline dev (recorder, transport adapter, replay against recorded fixtures — none need the live sim).
- **ShadowPC (then own PC)** = the sim + the Python pilot, run together. Sync code via **git** (clone the repo there; push from Mac, pull on the box).
- Don't move the MCP stack onto Windows — it's OAuth-heavy (needs a browser; breaks headless) and has Mac-only servers (imessage, cua, keychain creds).

### 3.1 ShadowPC setup (one-time, ~30 min)
- [ ] Sign up at shadow.tech (Boost, or **Power** for VQ2 headroom); confirm regional availability (occasional queue).
- [ ] Launch the Windows 11 desktop from the Mac client.
- [ ] **Re-download the dev kit inside ShadowPC** from the email links (faster than uploading the 1.95 GB zip from the Mac); extract `AIGP_3364.zip`, run `FlightSim.exe`, log in with the sim account.
- [ ] Install Python 3.14.2 + `git`; `git clone` the repo; `pip install -r requirements.txt`.
- [ ] Cancel ShadowPC once back on your own PC (flat-rate, month-to-month — no lock-in).

### 3.2 Cloud-VM fallback (only if ShadowPC is unavailable)
Viable Windows+DirectX options ranked by cost (verified 2026-06-01; stop-when-idle): **Azure NV A10 v5 Spot ~$0.17/hr** (cheapest; ~30 s eviction risk) · **Tensordock RTX 4090 ~$0.37/hr** (marketplace, self-install Win) · **AWS g4dn.xlarge ~$0.71/hr** (most predictable, T4 16 GB) · **Azure NV A10 On-Demand ~$1.46/hr**. Co-locate the pilot on the VM; view via RDP. Linux-only clouds (RunPod / Lambda / Modal / vast.ai / Oracle / Paperspace-for-new-users) **cannot** run the sim.

### 3.3 Free experiment you can run on the M4 Max anytime: Parallels (NOT UTM)
Worth a $0 hour — the binary is friendlier than expected: **UE4 + DirectX 11** (not UE5/DX12), **no kernel anti-cheat bundled**, which Parallels supports. Verdict **MAYBE** (not no-go). Risks: (1) DX11→Metal rendering correctness ("D3D device lost"/corruption), (2) the **online login may reject a Windows-on-ARM VM**. Test: install Windows 11 ARM in Parallels → launch `FlightSim.exe` (or force DX11 via `…\FlightSim\Binaries\Win64\DCGame-Win64-Shipping.exe -d3d11 -windowed -ResX=640 -ResY=360`) → does it render the scene, pass login, and hold ~30 Hz with the pilot connected? If device-lost/corruption or login rejection → stop, use ShadowPC. **UTM = no-go** (no DirectX acceleration for Windows guests).

> **Sim account:** in-sim login uses `kenichimatsuo1775@gmail.com` (password in your password manager — **not stored in this repo**).

---

## 4. Open questions — answered only by first contact (Phase 0 checklist)

Build the logger (Task 0.3) and answer **all** of these from one recorded run before designing Phase 2+:
- [ ] Are `LOCAL_POSITION_NED` / `ODOMETRY` **populated** with usable local position, or zeros/absent? *(Determines: known-map planner directly, vs vision+dead-reckoning.)*
- [ ] Is the **track-data packet actually sent** — in VQ1? in VQ2? *(Determines: runtime gate map vs build-from-vision. "Guidance off" in VQ2 may or may not withhold it.)*
- [ ] Does the sim accept `SET_ATTITUDE_TARGET` **and** `SET_POSITION_TARGET_LOCAL_NED`? Which yields better control? Is there a downstream "stabilized controller" expecting one type?
- [ ] What command rate does the sim accept (<100 Hz spec vs 250 Hz template)? Round-trip latency?
- [ ] Vision: real resolution/FPS/JPEG quality/latency; gate appearance in VQ1 (color, highlight).
- [ ] Arm + `SIM_RESET (31000)` behavior; how a run starts/ends; `active_gate_index` semantics; `COLLISION` thresholds.
- [ ] Commanded-thrust → measured-acceleration mapping (for calibration); does `ACTUATOR_OUTPUT_STATUS` populate?
- [ ] Confirm NED origin = arm point and the +20° camera tilt in practice.

---

## 5. Phased execution plan

> Phase 0–1 are concrete (the contract is known). Phase 2–3 are specified as task-lists with acceptance criteria because their detail legitimately depends on the Phase-0 telemetry findings — writing fictional TDD steps against unknown telemetry would be a placeholder. Re-expand Phase 2+ into bite-sized TDD tasks once Phase 0 lands.

### Phase 0 — Host + first contact (the gating milestone)
**Outcome:** the real sim runs, our pilot connects, and we have a recorded telemetry+vision log that answers §4.

- [ ] **0.1 Stand up the sim host (ShadowPC).** Per §3.1: sign up, launch the Windows 11 desktop, re-download + extract the dev kit inside it, install Python 3.14.2 + git. Pilot and sim both run here (localhost UDP 14550/5600). Document in `docs/aigp/host_setup.md`. *(Switch to your own RTX PC when available — same steps, minus signup.)*
- [ ] **0.2 Run the official kit unmodified.** Extract `AIGP_3364.zip`, launch `FlightSim.exe`, log in (sim account), then run the **stock** `PyAIPilotExample/main.py` (its own venv) to confirm heartbeat + that motor/attitude commands move the drone. This validates the environment before our code touches it.
- [ ] **0.3 Build the telemetry logger** (highest-leverage task). Create `competition/aigp_recorder.py`: a pymavlink `udpin:…:14550` client that subscribes to **every** message in §1.2, plus the `vision_udp.py` receiver, and writes a timestamped JSONL of message types/fields + a frame index (+ optional JPEG dump). Run one full manual/auto traversal. **Acceptance:** JSONL shows which messages arrive populated → §4 checklist filled in. This log becomes the Phase-1 replay fixture (develop offline, no EC2 burn).
- [ ] **0.4 Decide transport library.** Default **pymavlink** (matches the template, cleanly exposes the custom encapsulated messages MAVSDK may hide). Record the decision + rationale in `docs/aigp/decisions.md`.

### Phase 1 — Minimal end-to-end pilot on the real sim → "any clean lap" / VQ1 completion
**Outcome:** our `RacePipeline` flies the real sim through the VQ1 gates in order without a crash.

- [ ] **1.1 pymavlink transport adapter.** `competition/aigp_mavlink.py` implementing `CompetitionInterface`: connect/arm/`SIM_RESET`; RX thread parsing ATTITUDE, HIGHRES_IMU, LOCAL_POSITION_NED, ODOMETRY, race-status, track-data, COLLISION → fill `TelemetryState`; reuse `vision_udp.VisionUdpListener` for `get_camera_frame()`. TDD against the Phase-0 JSONL replay fixture. **Acceptance:** replaying the recording reproduces the same `TelemetryState` stream; unit tests for each parser (esp. the `<BQqqIq` race-status and `<Hfffffffff` per-gate unpacks).
- [ ] **1.2 Track-data → gates.** `competition/track_data.py`: convert the parsed gate list (NED pos + quaternion + w/h) into `List[GateSpec]` and call `RacePipeline.configure(gates)`. **Acceptance:** a synthetic track-data payload round-trips to the correct `GateSpec` list (positions, normals from quaternion, ordering by `gate_id`).
- [ ] **1.3 State source wiring.** If §4 says local position is populated → feed EKF `update_odometry`; else integrate velocity + correct with `gate_pnp` known-gate fixes. **Acceptance:** on the replay, estimated position tracks the logged position (or, in dead-reckon mode, stays bounded between gate sightings).
- [ ] **1.4 Control output.** Map `AttitudeCommand` → `SET_ATTITUDE_TARGET` (attitude-quaternion+thrust first). **Acceptance:** a known tracker command produces the expected MAVLink quaternion+thrust+type-mask (unit test); thrust within [0,1].
- [ ] **1.5 Sim-time loop.** Drive the control loop + logging on sim time / TIMESYNC. **Acceptance:** loop cadence holds against sim clock; no wall-clock dependence in the hot path.
- [ ] **1.6 First live closed-loop run.** On EC2: arm → fly VQ1. Use `active_gate_index` to confirm sequencing and `COLLISION` to detect crashes. **Acceptance: drone passes ≥1 gate in order, live.** Iterate to **all VQ1 gates, no crash** (VQ1 = completion).
- [ ] **1.7 Calibration pass.** Use the live thrust↔accel data to fill `competition/drone_spec.py` with real mass/thrust/drag and re-tune the tracker gains on the real sim (retire the PyBullet-artifact constants). **Acceptance:** commanded vs achieved acceleration error < target; stable hover + step response on the real sim.

### Phase 2 — VQ1 hardening (reliable completion + submit)
- [ ] 2.1 Perception backup/correction: retune `phase1_detector` for the desaturated/highlighted VQ1 gates; use PnP fixes to correct drift even if local position is provided (robustness if it's noisy).
- [ ] 2.2 Crash-resilience: wire `COLLISION` + `active_gate_index` regressions into the `dynamic_replanner`; recover or safely continue.
- [ ] 2.3 Reliability matrix: N consecutive full-course completions on the real sim (deterministic seed). **Acceptance:** ≥ X/X clean completions within 8-min cap.
- [ ] 2.4 Submit VQ1 per the team-portal flow; capture the submission/eval mechanics in `docs/aigp/submission.md`.

### Phase 3 — VQ2 (fastest valid time)
- [ ] 3.1 Re-run the Phase-0 logger on the VQ2 environment — **confirm whether track-data + local position are still provided with "guidance off."** Branch the plan on the answer.
- [ ] 3.2 Perception for 3D-scanned/realistic scenes: likely upgrade `phase1_detector` → a learned gate detector (YOLO-pose / corner regression), trained on recorded VQ2 frames; tighten `gate_tracker` temporal association.
- [ ] 3.3 Speed: re-enable the racing-line optimizer + speed profiler calibrated to the **real** drone envelope; evaluate body-rate+thrust control for aggression; consider the off-by-default learned residual now that physics is real.
- [ ] 3.4 Course-specific tuning (legitimate: known course + unlimited attempts) — clean, reproducible per-course parameters, not name-based hacks. Optimize lap time against the 1500 mm aperture margins.
- [ ] 3.5 Submit VQ2; iterate on fastest valid time.

### Cross-cutting (continuous)
- [ ] Keep the Phase-0 recorder as a permanent black-box logger on every live run (offline replay = cheap iteration).
- [ ] CI: keep the 306-test suite green; add transport/track-data/control-mapping tests; run the matrix bench as a regression guard for the autonomy core.
- [ ] Decision log + setup docs under `docs/aigp/`.

---

## 6. Immediate next actions (this week)
1. **Stand up ShadowPC** and get `FlightSim.exe` running + sim-account login (Task 0.1–0.2).
2. **Run the stock `PyAIPilotExample`** to confirm a live heartbeat and that commands move the drone (Task 0.2).
3. **Write `competition/aigp_recorder.py` and capture one full run** — then fill in the §4 open-questions checklist. *Everything after this is shaped by that log.* (Task 0.3)
4. Lock the **pymavlink** transport decision and start `competition/aigp_mavlink.py` against the recorded fixture (Task 0.4 → 1.1).

---

## 7. Self-review notes
- Spec coverage: interface (§1) ↔ reuse (§2) ↔ host (§3) ↔ unknowns (§4) ↔ phased tasks (§5) ↔ near-term (§6) — each requirement maps to a task.
- The plan deliberately front-loads the *unknown-resolving* work (Phase 0 logger) before committing to a perception/estimation design, because the single largest design fork (is local position + track-data provided?) is unknowable without the binary.
- Names kept consistent: `competition/aigp_mavlink.py` (transport), `competition/track_data.py` (gate map), `competition/aigp_recorder.py` (logger), reused `RacePipeline.configure(gates: List[GateSpec])`, `competition/drone_spec.py` (calibration target).
