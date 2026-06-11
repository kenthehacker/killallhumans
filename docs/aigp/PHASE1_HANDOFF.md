# Phase 1 Handoff — AIGP VQ1 pilot (for the next implementer)

> Written 2026-06-10/11 after Phase 0 first contact. Audience: a fresh (cheaper) agent
> picking up implementation. **Read `docs/aigp/2026-06-10-first-contact-findings.md` first**
> (it is ground truth), then this. Then `PLAN.md` §5 for the phase map.

## 0. TL;DR — where we are, what you're doing

**Phase 0 is complete and on `main`** (commit `963f1ae`). We made first contact with the real
AI-GP simulator and characterized the whole interface. The headline: **best-case branch** — the
sim hands us **usable local position (`LOCAL_POSITION_NED` ~95 Hz) + odometry (`ODOMETRY` ~74 Hz,
with quaternion + covariance + quality) + the runtime gate map**. So VQ1 is *fly-the-known-line*,
not a vision problem.

**Your job is two interlocking workstreams:**
- **A — Build the pilot transport** (Tasks 1.1–1.3): a pymavlink adapter, gate-map→`GateSpec`
  conversion, and wiring the provided pose into the EKF, so `RacePipeline` runs on real telemetry.
  Mostly offline/TDD against a recording — low risk.
- **B — Fix the control-stack bugs** (Task B): the SE(3) tracker has audit-confirmed frame/sign
  bugs that real physics WILL hit. These are **latent under a green 427-test suite** because they
  were masked by mutually-cancelling pairs and by benches that don't run the live control path.
  **Must be fixed before the first live `SET_ATTITUDE_TARGET` lap.**

**Invariants that hold everywhere on the AIGP path (memorize):**
- Frame is **NED** (x≈north, y≈east, **z DOWN**). The VQ1 course *descends*, so gate z *increases*
  (gate 0 z≈−0.03 → gate 5 z≈+25.97). Add **zero** frame conversions on this path.
- All quaternions are **`[w, x, y, z]`** — `ODOMETRY.q`, `TrackGate.orientation`,
  `SET_ATTITUDE_TARGET.q`, and `competition.adapter.Quaternion` all agree.
- Origin = spawn point (drone rests at ≈ (0, 0, 0.02)). Angles rad, accel m/s², thrust ∈ [0,1].
- We **own the control loop** via `SET_ATTITUDE_TARGET`. The sim has **no usable velocity/position
  autopilot** (a velocity setpoint produced a runaway climb to −27 m @ 28 m/s — verified live).

## 1. How to operate the sim host (needed to capture fixtures + do live runs)

The sim runs on Ken's Windows PC; the Mac is the brain. Control it over SSH:
```
ssh -i ~/.ssh/id_ed25519_winpc -o IdentitiesOnly=yes Kenichi@100.122.0.79 "<powershell>"
```
- `IdentitiesOnly=yes` is REQUIRED. Remote default shell = PowerShell. Mac must be on the personal
  Tailscale profile: `tailscale switch kenichimatsuo1775@gmail.com` (⇄ `doordash.com` for corp).
- Repo on the PC: `C:\Users\Kenichi\killallhumans` (Mac pushes → PC pulls).
- **The sim must be in the Virtual Qualifier mode** (NOT the ACRO free-flight/practice scene) to
  serve the MAVLink (14550) + vision (5600) interface. See `host_setup.md`.
- **Launch the sim:** `scripts/launch_sim.ps1` (a `schtasks /IT` bridge from SSH session 0 into the
  desktop session 1 — a plain SSH launch won't render/GPU). The login persists to a save file with
  an AutoLogin path; if the PGOS token expired you'll need a one-time Parsec/console login
  (ideally "remember me"). **Never put the password in the repo** (Ken's password manager has it;
  DPAPI `cmdkey` is the fallback store).
- **Fetching the gate map:** it is a one-shot, only sent on `SIM_RESET` (cmd 31000) and only after
  the client *announces* itself (TIMESYNC + GCS heartbeat). It is NOT in the steady stream, and a
  naive `wait_heartbeat()` discards the connect-time transfer. The Task 1.1 adapter handles this.

## 2. Constraints & gotchas (hard rules)

- **Credentials:** AIGP sim login is in Ken's password manager — NEVER write it to the repo or any
  log. Parsec "remember me" or Windows Credential Manager (DPAPI) only.
- **SafetyBelts:** pushes to `kenthehacker/killallhumans` pass because the repo is MIT-licensed.
  Never use `--no-verify` or a test-mode bypass.
- **`~/.ssh` on the Mac is off-limits** — do not `ls`/`cat`/`keygen` there; invoking
  `ssh -i ~/.ssh/<key>` as an argument is fine.
- **Tests:** TDD. The full `pytest` suite must stay green. **Some existing tests enshrine the very
  bugs you're fixing** (e.g. `test_downward_error_increases_thrust`) — when a correct fix flips an
  assertion, *update the test* and cite the audit; don't contort the fix.
- **Branch:** `main` now holds Phase 0. Do Phase 1 on a feature branch (e.g. `aigp-phase1-pilot`);
  push it (SafetyBelts-OK) and merge when green.
- The pedregal / giga_chad_llm / UMS rules in global memory do **not** apply here — this is the
  personal drone repo.

## 3. Recommended sequencing

The two workstreams interact at two seams — mind them:
- **Gate-normal convention (Task 1.2 ↔ Task B-6b):** the gate "normal" must point along the
  *flight axis*. The real VQ1 gates are `q=[0.707,0,0,0.707]` and the course runs −X; the repo's
  x-forward assumption yields a normal *perpendicular* to travel (→ **zero gate passes detected**).
  Both specs converge on: local **+Y** through-axis → world **−X** for VQ1, with a sign/axis check
  against course progression. Fix the converter (1.2) and the `_gate_normal` consumers (B-6) together.
- **State→control yaw (Task 1.3 ↔ Task B-4):** the EKF yaw is structurally unobservable and
  inits blind-to-0, then overwrites the sim's true yaw. Orientation must be a **telemetry
  passthrough**. 1.3 wires it; B-4 stops the overwrite. Do them aware of each other.

Suggested order:
1. **Task 1.1** transport adapter — biggest piece, offline TDD; capture the canonical fixture here.
2. **Task 1.2** track_data→GateSpec — fix the existing buggy converter; resolve the normal axis (B-6b).
3. **Task 1.3** state-source wiring — provided pose → EKF, orientation passthrough.
4. **Task B** control-stack fixes — B-1…B-5 are BLOCKING; B-6/B-7 as tagged. Required before any
   live lap.
5. *Then* (separate, not specced here) PLAN 1.5/1.6: the runner + first live closed-loop lap —
   capture-validate each step on the sim host.

**Fan-out note:** these four specs were drafted by parallel sub-agents that each read the actual
code on `main@963f1ae`; line numbers were verified then but re-confirm as you edit. Every spec is
TDD-first with explicit acceptance criteria and an explicit out-of-scope list — respect both.

---

## 4. Task specs

### ────────────────────────────────────────────────────────────────────
""" The four sections below are the detailed, code-grounded specs. """

<!-- ===================== TASK 1.1 + 1.4 ===================== -->

## Task 1.1 + 1.4 — pymavlink transport adapter

**Deliverable:** `competition/aigp_mavlink.py` — class `AIGPMavlinkAdapter(CompetitionInterface)` — plus `competition/tests/test_aigp_mavlink.py` and a recorded replay fixture. This is the production transport to the official AI-GP simulator (`FlightSim.exe`, Virtual Qualifier mode), replacing the speculative MAVSDK bridge.

### 0. Ground truth — read these first, do not re-derive

| File | What you take from it |
|---|---|
| `docs/aigp/2026-06-10-first-contact-findings.md` | Live-verified message rates, gate-map-via-SIM_RESET behavior, control findings, vision specs. Treat every number in it as ground truth. |
| `competition/adapter.py` | The ABC you implement (`CompetitionInterface`, all methods `async`) + the dataclasses (`TelemetryState`, `Quaternion`, `IMUData`, `AttitudeCommand`, `AttitudeRateCommand`, `PositionCommand`, `CameraFrame`). **Do not modify this file.** |
| `competition/aigp_messages.py` | Pure parsers: `parse_race_status`, `parse_track_data`, `TrackInfoReassembler`, `encode_race_status`, `encode_track_data`, constants. **Reuse — never re-implement the struct unpacking.** |
| `competition/aigp_recorder.py` | The *proven* live RX handling: exact `DATA_TRANSMISSION_HANDSHAKE`/`ENCAPSULATED_DATA` dispatch, the `<BH` chunk header strip, `bytes(msg.data)`. Copy this dispatch logic; it worked against the real sim on 2026-06-10. |
| `competition/vision_udp.py` | `VisionUdpListener` (asyncio, port 5600, decode-cached `latest_frame()`). Reuse as-is. |
| `competition/track_data.py` | Downstream consumer: `track_data_to_gatespecs(TrackData)` (Task 1.2). You only need to *expose* `TrackData`. |

**Do NOT touch** `competition/mavlink_bridge.py` (old MAVSDK bridge, superseded for the real sim), `adapter.py`, `aigp_messages.py`, `vision_udp.py` (additive imports only). **Import discipline:** `pymavlink` imported **lazily inside `connect()`** so `import competition.aigp_mavlink` succeeds without pymavlink and unit tests never need it.

### 1. Connect sequence (the load-bearing logic)
`async def connect()`:
1. Open `mavutil.mavlink_connection("udpin:127.0.0.1:14550")` (lazy pymavlink; default source_system 255 = GCS; common dialect has every message). `udpin` = bind+listen.
2. **Start the RX thread immediately — before any waiting.** The gate-map transfer
   (`DATA_TRANSMISSION_HANDSHAKE` → `ENCAPSULATED_DATA` track chunks) is a **one-shot** at scene
   init / on `SIM_RESET`; `wait_heartbeat()` (which discards non-matching messages) would eat a
   connect-time transfer. **Never call `wait_heartbeat()`** — detect the first HEARTBEAT inside the
   RX handler; `recv_match` auto-populates `target_system`/`target_component` anyway.
3. **Start the ANNOUNCE thread immediately** (10 Hz, under a send-lock): `timesync_send(tc1=0,
   ts1=now_ns)` + `heartbeat_send(MAV_TYPE_GCS=6, MAV_AUTOPILOT_INVALID=8, base_mode=0,
   custom_mode=0, MAV_STATE_ACTIVE=4)`. **Required** — the sim only pushes the full stream once a
   client announces. (udpin drops outbound writes until the sim's first inbound packet sets the
   peer, so announcing early is harmless.)
4. Wait (via `asyncio.to_thread`) for `_heartbeat_event` (timeout → `ConnectionError("…Virtual
   Qualifier mode?")`), then `_telemetry_ready_event` (set once both an attitude and a position
   message have arrived).
5. Start vision (`VisionUdpListener(port=5600)`), if enabled.
6. **Fetch the gate map:** if not already captured at connect, send `SIM_RESET`
   (`command_long_send(tsys, tcomp, 31000, 0, 0,0,0,0,0,0,0)`) and wait for the track event; retry
   N times; hard-fail or warn per `require_track`. SIM_RESET re-spawns the drone at start, resets
   the race clock, re-sends the gate map; `active_gate_index → 0`. This is the *normal* path.

`is_connected` = `_conn is not None and (monotonic − last_heartbeat) < 3 s`. `disconnect()`
idempotent: stop event → join RX+announce threads (2 s) → stop vision → close conn.

**Concurrency:** a `threading.Lock` around *every* `self._conn.mav.*_send(...)` (the MAVLink object
shares a seq counter + pack buffer; announce + control send concurrently). A second lock around all
state mutation. Single `_handle_message(msg)` method (takes any object with `.get_type()` + wire
attrs) is the unit-test seam; `_rx_loop` is a thin `recv_match(blocking=True, timeout=0.5)` shell.

### 2. RX dispatch → `TelemetryState` (single-source-of-truth, deterministic → replayable)
`TelemetryState` fields: `timestamp_us`, `position_ned`, `velocity_ned`, `orientation: Quaternion`,
`angular_velocity`, `imu: Optional[IMUData]`. Also expose (not in TelemetryState) race status,
track data, a collision deque, actuator outputs. Mapping:

| MAVLink (rate) | Wire fields | → | Rule |
|---|---|---|---|
| `LOCAL_POSITION_NED` (~95 Hz) | `x,y,z`, `vx,vy,vz`, `time_boot_ms` | `position_ned`, `velocity_ned` | **sole** pos/vel source |
| `ODOMETRY` (~74 Hz) | `q=[w,x,y,z]`, `quality`, `reset_counter`, `time_usec` | `orientation = Quaternion(q[0],q[1],q[2],q[3])` | **primary** orientation. Do NOT take its pos/vel (child-frame, unverified). |
| `ATTITUDE` (~117 Hz) | `roll,pitch,yaw`, `*speed` | `angular_velocity`=(rollspeed,pitchspeed,yawspeed); orientation fallback **only until first ODOMETRY** | sole body-rate source |
| `HIGHRES_IMU` (~117 Hz) | `xacc..`, `xgyro..`, `time_usec` | `imu = IMUData(...)`, `mag=None` (NaN on this sim) | predict-step input |
| `ACTUATOR_OUTPUT_STATUS` (~95 Hz) | `time_usec`, `active=15`, `actuator[32]` | side state | calibration (1.7) |
| `COLLISION` (event) | `id` (1001 gate/1002 env), `threat_level`, `horizontal_minimum_delta` | deque | `horizontal_minimum_delta` IS the impulse magnitude, not a delta |
| `ENCAPSULATED_DATA` p[0]==1 (~4 Hz) | `data` | `parse_race_status` | race status |
| `DATA_TRANSMISSION_HANDSHAKE` | `width`=transfer_id, `packets`=num_chunks | `reassembler.begin_transfer` | one-shot |
| `ENCAPSULATED_DATA` p[0]==2 | `data`, `seqnr` | strip `<BH` (transfer_id=p[1:3] LE), `feed_chunk(tid, seqnr, p[3:])` | sets track event when complete |

`timestamp_us`: `time_boot_ms*1000` for ATTITUDE/LOCAL_POSITION_NED, `time_usec` as-is for
ODOMETRY/HIGHRES_IMU. `is_armed = bool(base_mode & 0x80)` (spawn base_mode=193 → armed). Wrap RX body
in try/except (one bad message must not kill the thread).

### 3. arm / reset / offboard / camera
- `arm()`: spawns armed → return if `is_armed`; else send `MAV_CMD_COMPONENT_ARM_DISARM` (400, param1=1), **warn (don't raise)** if still disarmed.
- `reset()` (adapter-specific): clear track event → SIM_RESET → return the fresh `TrackData`.
- `start_offboard()/stop_offboard()`: no-ops (sim accepts setpoints with no mode handshake).
- `get_camera_frame()`: `return self._vision.latest_frame() if self._vision else None` (640×360×3 BGR).

### 4. Task 1.4 — `AttitudeCommand` → `SET_ATTITUDE_TARGET`
Primary path (SE(3) tracker, attitude-quaternion + thrust):
```python
q = Quaternion.from_euler(cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad)
thrust = min(1.0, max(0.0, cmd.thrust))           # NaN → raise (tracker bugs must not reach wire)
with self._send_lock:
    self._conn.mav.set_attitude_target_send(
        self._time_boot_ms(), tsys, tcomp,
        0b00000111,                                # mask=7: ignore body rates; attitude+thrust active
        [q.w, q.x, q.y, q.z], 0.0, 0.0, 0.0, thrust)
```
Type-mask bits (set = ignore): roll_rate 1, pitch_rate 2, yaw_rate 4, THRUST_BODY 32, THROTTLE 64,
ATTITUDE 128. Attitude+thrust ⇒ **7**. Body-rate+thrust (VQ2 alt, implement as `send_attitude_rate`):
mask **128**, `q=[1,0,0,0]` placeholder, rates from `AttitudeRateCommand`. Never set bit 64.
`send_position` (`SET_POSITION_TARGET_LOCAL_NED`, frame=1, mask=2496): implement for parity only —
**docstring must warn it does not track velocity cleanly** (runaway climb live); attitude path only.
Caller owns rate (50 Hz accepted live).

### 5. CLI / fixture capture
Add `__main__`: `python -m competition.aigp_mavlink --record out.jsonl [--duration 30] [--no-vision]
[--jpeg-dir DIR] [--attitude-test]`. It runs the full connect (announce + SIM_RESET inherent — the
exact "recorder must announce+reset" enhancement first contact called for) and tees every message
through the recorder's existing pure fns (`record_for_message`, `mavlink_msg_to_fields`,
`race_status_fields`, `track_data_fields`, `write_jsonl` — import, don't duplicate). Prints per-type
rates, gate count + gate-0 NED pos, first frame shape, armed state.

### 6. Tests — `competition/tests/test_aigp_mavlink.py` (TDD, no pymavlink/sockets/cv2)
Fakes: `FakeMsg(type, **fields)` with `.get_type()`; `FakeMav` recording every `*_send(...)`.
1. Parser round-trips (`<BQqqIq` w/ negative sentinels; `<Hfffffffff` over the real VQ1 map — use
   `pytest.approx(abs=1e-3)`, values transit float32).
2. Chunked gate-map through the adapter (handshake + 2 chunks built as `bytes([2]) +
   tid.to_bytes(2,"little") + body_slice`); out-of-order assembles; orphan chunk ignored.
3. `TelemetryState` population from synthetic ATTITUDE/LOCAL_POSITION_NED/HIGHRES_IMU/ODOMETRY;
   assert orientation euler yaw ≈ π/2 from `q=[0.707,0,0,0.707]` (**proves q[0]=w**); attitude
   orientation fallback only until first ODOMETRY; ODOMETRY velocity NOT used; timestamp normalization.
4. Side state: COLLISION `impulse==horizontal_minimum_delta`, drained once; actuator stored;
   race-status via ENCAPSULATED_DATA; heartbeat armed bit (193→True, 65→False).
5. 1.4 wire: known command → exactly one `set_attitude_target_send`, `type_mask==7`, q≈from_euler,
   thrust clamped, zero rates; rate path mask 128; position frame 1/mask 2496; NaN raises.
6. SIM_RESET wire (command 31000, 7 zero params); arm noop-when-armed / 400-when-disarmed.
7. **Replay against the recorded JSONL fixture** (PLAN 1.1 acceptance): map records→FakeMsg
   (`race_status`/`track_data` rows re-encoded via `encode_*` to exercise the real parse path),
   stream through `_handle_message`, assert no exceptions, `timestamp_us` non-decreasing, first pos
   ≈ (0,0,0.02)±0.1, final TrackData matches the fixture's own `track_data` record (6 gates, poses
   to 1e-3), a race_status with `active_gate_index==0`, ≥100 each of the 4 telemetry types.
   `@pytest.mark.skipif(not FIXTURE.exists())` + a synthetic-fixture twin so CI is green pre-capture.

**Fixture capture (live, on WinPC):** sim in Virtual Qualifier, then
`python -m competition.aigp_mavlink --record aigp_vq1_capture.jsonl --duration 30 --jpeg-dir frames/`.
Commit gzipped at `competition/tests/fixtures/aigp_vq1_capture.jsonl.gz` (JPEGs NOT committed).
Validity gate before commit: ≥1 `track_data` (6 gates), ≥100 each telemetry type, ≥1 `race_status`,
≥10 `vision_frame`. **Do not** capture with the old `aigp_recorder.run()` alone — it never announces
or resets and its `wait_heartbeat()` discards the gate map.

### 7. Acceptance
1. Adapter implements every ABC method + `reset()`; `import competition.aigp_mavlink` works without
   pymavlink. 2. `pytest competition/tests/ -q` fully green incl. §6; existing suites untouched.
3. Replay reproduces a deterministic TelemetryState stream incl. the 6-gate TrackData. 4. Known
   command → asserted SET_ATTITUDE_TARGET (q `[w,x,y,z]`, mask 7, thrust clamped). 5. Gate-map
   robustness unit-proven (connect-time capture + SIM_RESET-trigger-with-retry). 6. Concurrency
   hygiene (locks; idempotent disconnect; no thread leaks). 7. Live smoke (manual): rates
   ≈117/95/74 Hz, 6 gates (gate 0 ≈ (−23.3,−0.4,−0.03)), 640×360 frame, armed; fixture committed.
**Out of scope:** calibration (1.7), GateSpec wiring into RacePipeline, vision perception, TIMESYNC
clock-offset, any change to `mavlink_bridge.py`/PyBullet adapter.

<!-- ===================== TASK 1.2 ===================== -->

## Task 1.2 — track_data → GateSpec

**Goal:** convert a parsed `TrackData` (sim gate map, fetched on `SIM_RESET`) into the
`List[GateSpec]` `RacePipeline.configure()` consumes — correct normals, sizes, ordering.

**Status:** `competition/track_data.py` (`track_data_to_gatespecs`) EXISTS and is WRONG twice — do
not rewrite from scratch, fix it:
- **Bug 1:** maps packet `width/height` → `interior_width/height`; first contact proved the packet
  carries the **outer** 2.72 m dimension.
- **Bug 2:** sets `yaw/pitch/roll` from `Quaternion.to_euler()` — for the real VQ1 quaternion this
  makes gate normals **perpendicular to the flight axis** (drone plans through gates sideways).
Only `competition/track_data.py` + `competition/tests/test_track_data.py` change.

**Target type** `gate_sequencing/sequencer.py:35` `GateSpec(gate_id:str, position, yaw, pitch, roll,
interior_width=1.5, interior_height=1.5, border_width=0.6, depth=0.26, sequence_index)`.
Consumer `race_pipeline.py:208 configure(gates, start_position=(0,0,0), start_velocity=(0,0,0))`.
**Critical convention** — both consumers (`race_pipeline.py:620`, `sequencer.py:701`) derive the
normal as `_gate_normal(yaw,pitch) = (cos y·cos p, sin y·cos p, sin p)` (NED, **+sin p**). So
`GateSpec.yaw/pitch` must be chosen so this equals the gate's **fly-through axis in world NED** (and
`yaw` is also used as the drone's reference heading through the gate).

**Mapping:** `gate_id→str`; `position→position_ned` passthrough (no flip/negation); `yaw=atan2(n_y,
n_x)`, `pitch=asin(clamp(n_z))` from the world through-axis `n`; `roll=0` (VQ1 gates upright);
`interior_* = packet_dim − 2·AIGP_GATE_BORDER_M` with a `<0.5 m → treat as already-interior`
fallback (2.72→1.52; keeps a 1.5-interior test green); border/depth = defaults; `sequence_index` =
index after sorting by `gate_id`.

**Orientation math (core):** the sim gates face along the flight axis; VQ1 `q=[0.707,0,0,0.707]`
(+90° yaw), course runs −X. Rotating gate-local **+Y** by q gives **−X** (the flight axis), so
`GATE_LOCAL_THROUGH_AXIS=(0,1,0)`. Per gate: (1) normalize q (norm<1e-6 → ValueError); (2) world
through-axis = R·(0,1,0) = `(2(xy−wz), 1−2(x²+z²), 2(yz+wx))` (worked check for VQ1: `(−1,0,0)` ✓);
(3) **disambiguate sign vs course progression** — tangent `t_i=normalize(pos[i+1]−pos[i])` (last:
back-diff; single gate/zero-norm: skip); if `dot(n,t_i)<0` flip n; if 0 leave; (4) `yaw=atan2(n_y,
n_x)`, `pitch=asin(clamp(n_z))` (consumers use **+sin p** — NOT aerospace −sin θ), `roll=0`. VQ1
result: every gate `yaw=π, pitch=0`, normal `(−1,0,0)`.

**Ordering/validation:** sort by `gate_id` asc (don't mutate input), `sequence_index=0..N−1`
(matches `race_status.active_gate_index`); duplicate gate_id → ValueError; empty → `[]`.

**Run-flow contract (lands in the 1.5/1.6 runner, not here):**
```python
track = await interface.wait_for_track_data(timeout_s=10.0)
gates = track_data_to_gatespecs(track)
pipeline.configure(gates, start_position=(0.0, 0.0, 0.0))   # configure() must precede run()
```
Runner note: SIM_RESET restarts the race clock, so fetch-map → configure (slow precompute) →
re-SIM_RESET → fly.

**Tests** (`competition/tests/test_track_data.py`): keep empty/single/sorted tests; **rewrite**
`test_orientation_quaternion_becomes_yaw` (euler-passthrough is now wrong). Add: (1)
`test_vq1_capture_end_to_end` — build the 6 captured gates (`q=(0.7071,0,0,0.7071)`,
`width=height=2.72`, the position table from the findings doc), deliver **shuffled**, go
`encode_track_data → parse_track_data → track_data_to_gatespecs`; assert ids 0..5, seq 0..5,
positions (abs=1e-4), every normal ≈(−1,0,0), `yaw≈π`, `pitch≈0`, `roll=0`, interior≈1.52,
border 0.6, depth 0.26, outer 2.72; (2) `test_normal_flipped_to_match_course_direction`
(`q=(0.7071,0,0,−0.7071)` → still (−1,0,0)); (3) interior fallback at width=1.5; (4) duplicate ids →
ValueError; (5) zero-quaternion → ValueError.

**Acceptance:** signature unchanged; exports `GATE_LOCAL_THROUGH_AXIS`; VQ1 end-to-end passes;
`_gate_normal(spec.yaw,spec.pitch)` reproduces the quaternion-rotated through-axis to 1e-6; sign
disambiguation works; old euler test gone; full suite green; docstring updated (packet=outer,
roll=0 rationale, NED no-conversion). **Out of scope:** transport (1.1), the runner + double-reset
recipe (1.5/1.6), EKF/control (1.3/1.4), changing `GateSpec`.

<!-- ===================== TASK 1.3 ===================== -->

## Task 1.3 — state-source wiring

**Goal:** make the sim's provided pose the primary state. `LOCAL_POSITION_NED` (~95 Hz) → EKF
position/velocity measurement; orientation = `ODOMETRY.q` passthrough (NOT estimated); `HIGHRES_IMU`
→ predict step; `gate_pnp` → health-gated backup. NED end-to-end — **add zero frame conversions.**

**EKF API** (`estimation/ekf.py`, do not redesign): 15-dim state (pos NED, vel NED, euler, accel
bias, gyro bias); NED world / FRD body; gravity via `accel_ned[2]+=9.81` (correct — don't touch).
`initialize(pos, vel, orientation, timestamp_s)`; `predict(accel_body, gyro_body, timestamp_s)`
(self-dedups dt≤0); `update_odometry(position, velocity)` (6-dim POS+VEL — **this is what
LOCAL_POSITION_NED feeds**, NOT reserved for the ODOMETRY message; NOT stamp-deduped → you must gate
by stamp); `update_pnp_position(position)` (3-dim POS, R=0.3).

**Quaternion:** `q=msg.q` is `[w,x,y,z]`; `Quaternion(w=q[0],x=q[1],y=q[2],z=q[3])`. Do **not** route
through scipy (it's `[x,y,z,w]`) — a w↔x swap is a ~180° error that "looks fine" at hover.
Cross-check test: ODOMETRY-q euler vs the sim's ATTITUDE euler agree ≤ 2°.

**Frame — convert NOTHING.** The sim is MAVLink-standard NED/FRD; `TelemetryState`, EKF,
`StatePredictor`, `GateSpec`, gate map are all NED. The audit found NED↔ENU sign bugs at nearly
every existing boundary (incl. mutually-cancelling pairs) — **do not copy patterns from**
`pybullet_adapter.py:111-122` (ENU→NED, attitude not converted), `mpc_tracker.py:446-447`
(SimplePositionTracker ENU signs — keep `use_geometric_tracker=True`), `racing_line.py` (ENU
up-vector masked by `max_vertical_offset=0`), `plan_validator.py` (z-up). Sign invariants to assert:
VQ1 gate z **increases** along the course (`gates[-1].position[2] > gates[0].position[2]`); rest z ≈
+0.02; rest `HIGHRES_IMU.zacc ≈ −9.81` (if a recording shows +9.8, STOP — gravity handling would
need inverting; escalate, don't silently flip). **No `*-1` on y/z and no `[0,0,±1]` up-vectors in
the diff.** Don't consume `ODOMETRY.x/y/z/vx/vy/vz` (child-frame unverified); 2-line recorder
follow-up (in scope): add `frame_id`/`child_frame_id` to the curated ODOMETRY dict in
`aigp_recorder.py` so the next *moving* recording settles it.

**Why orientation passthrough (audit Blocker 4, verified):** no EKF measurement touches
orientation/gyro-bias and `F` omits the couplings → their Kalman gains are zero forever (a 0.5 rad
yaw error survived 600 perfect updates); and the pipeline inits yaw blind-0 then overwrites the
telemetry yaw (`race_pipeline.py:280,355`). The sim gives `ODOMETRY.q` @74 Hz (quality 100) + ATTITUDE
@117 Hz agreeing — ground-truth attitude. So orientation is a passthrough; EKF estimates pos/vel only.
**Only permitted `ekf.py` edit:** add `set_orientation(roll,pitch,yaw)` that hard-sets `x[ORI_IDX]`
(wrapped), called once per tick **before** `predict()`. No F-Jacobian changes, no attitude update,
no bias estimation, no gravity change.

**RacePipeline routing** (`race_pipeline.py`): replace callback step-1 with
`_update_state_estimate(telem)` doing, in order: (1) live init/re-init on first tick or when
`telem.odom_reset_counter` changes (`initialize(pos, vel, orientation=euler, timestamp_s=us/1e6)`);
(2) `set_orientation(*telem.orientation.to_euler())`; (3) `predict(imu.accel, imu.gyro,
imu.timestamp_us/1e6)`; (4) **stamp-gated** `update_odometry(pos,vel)` only when `lpn_time_boot_ms`
changed AND source healthy; (5) return pos/vel from EKF, **yaw from telemetry** (delete `_,_,yaw =
self.ekf.orientation` at :355). Add `TelemetryState` fields (additive, default None):
`lpn_time_boot_ms, odom_time_usec, odom_quality, odom_reset_counter` (the 1.1↔1.3 seam; 1.1 fills
them; require an immutable per-tick snapshot + sim-clock stamps). **Move `ekf.initialize(...)` out of
`_build_trajectory_from` (`:279-281`) into `configure()`** so a mid-race replan doesn't wipe the
filter (audit Blocker 5). New pipeline fields reset in `configure()`:
`_ekf_live_initialized=False, _last_lpn_stamp_ms=None, _last_odom_reset_counter=None`.

**gate_pnp backup** (API unchanged; NED position-only fix via `update_pnp_position`): health state
machine from per-tick freshness — PRIMARY (LPN age ≤0.05 s & quality≥50): update on fresh stamps;
DEGRADED (age 0.05–0.3 s or quality<50): predict-only, PnP if `pnp_mode!="off"`; DEAD_RECKON (age
>0.3 s): predict-only, PnP sole corrector, warn ≤1 Hz. **Default `pnp_mode="backup"`** because audit
Blocker 12 (detector feeds 2700 mm OUTER corners to a 1500 mm INNER PnP model → fixes land ~44% of
true range with 0.000 px error, inside the 3 m gate) means PnP currently *poisons* a healthy EKF;
the geometry fix is Phase 2.1.

**Clock:** sim time only (`time_usec/1e6`, `time_boot_ms/1e3`); no `time.time()`/`monotonic()` in the
state path (audit Blocker 13). Assert LPN and IMU/ODOMETRY epochs agree (≤50 ms); if not, normalize
to `HIGHRES_IMU.time_usec` in the 1.1 adapter and note in `decisions.md`.

**Tests** (`competition/tests/test_state_source_wiring.py` + EKF unit additions): commit a trimmed
(~15-30 s, must include a *moving* segment) Phase-0 JSONL slice under `competition/tests/fixtures/` +
a synthetic generator (no skip-and-pass). (9.1) Replay primary tracking: position RMS ≤0.10 m / max
≤0.25 m, vel RMS ≤0.25 m/s; yaw == ODOMETRY-q yaw; ODOMETRY-q vs ATTITUDE ≤2°; rest zacc∈[−10.3,−9.3],
rest z∈[−0.2,0.2], epoch consistency. (9.2) Dead-reckon: mask LPN after t, inject N(0,0.3²) PnP @1 Hz
→ error <2.0 m always, ≤0.75 m within 0.2 s of a fix; pure-IMU 1 s coast ≤1.5 m. (9.3) Units: stamp
dedup (same stamp → one update); reset_counter bump → re-init (no Kalman lag through a 5 m jump);
quality 10 → no update; 0.5 rad yaw error corrected in one tick; replan preserves bias/covariance;
descending-course z sign preserved. Add `decisions.md` D5 (LPN primary, ODOMETRY attitude+health,
PnP backup). **Out of scope:** NED↔ENU conversions anywhere, EKF observability surgery, scipy quats,
the Blocker-12 PnP geometry, the tracker yaw-frame bugs (Task B).

<!-- ===================== TASK B ===================== -->

## Task B — control-stack bug fixes (pre-live-lap)

All sites verified against `main@963f1ae` (control modules unchanged since audited `291bd4b`). Live
path: `RacePipeline._control_callback → GeometricTracker.track() → AttitudeCommand → SET_ATTITUDE_
TARGET`. Source audit: `docs/2026-06-09-logic-audit.md`. **Rule:** some tests enshrine these bugs —
when a fix flips an assertion, fix the test (cite the audit). `pytest` after each item.

**B-1 [BLOCKING] — GeometricTracker never rotates desired accel into the yaw frame**
`control/mpc_tracker.py:277-278`. roll/pitch are extracted from **world-frame** thrust components;
the correct desired frame (`:252-265`) is built then discarded. At yaw=90° a +x error → pure pitch →
sim accelerates **east** (90° wrong); at yaw=180° the accel is **inverted** → divergence. VQ1 flies
−X (~yaw 180°) → worst case. **Fix:** rotate the thrust-direction horizontal components into the
commanded-yaw frame before extraction (byte-identical at yaw=0):
```python
cpsi, spsi = math.cos(yaw_des), math.sin(yaw_des)
zx_h =  cpsi*z_b_des[0] + spsi*z_b_des[1]
zy_h = -spsi*z_b_des[0] + cpsi*z_b_des[1]
desired_pitch = -math.asin(np.clip(zx_h, -1.0, 1.0))
desired_roll  =  math.atan2(zy_h, -z_b_des[2])
```
Use `yaw_des` (atomic attitude setpoint). Tests: yaw=π/2 ref (10,0,0) → `roll<0`, `|pitch|<0.01`;
yaw=π ref (10,0,0) → `pitch>0`; yaw=0 tests unchanged.

**B-2 [BLOCKING] — saturated descent flips the thrust vector** `control/mpc_tracker.py:234-246`
(enshrined by `test_tracker.py:157-168`). A rotor only pushes body-up (`thrust_vec[2]≤0` in NED), but
a commanded down-accel > g (e.g. `kp_z=8·1.226 > 9.81`, or any recovery target below) makes
`thrust_vec` point down; it's normalized as if achievable (→0.95 thrust) and `z_b_des=[0,0,+1]` →
`roll=atan2(0,−1)=π` clamped to 49° → **banks 49% @95% thrust, climbs sideways** instead of
descending. VQ1 descends 26 m → fires on the lap. **Fix** (insert before `:234`):
```python
a_down_max = c.gravity - c.min_thrust_normalized * c.max_thrust_n / c.mass   # 9.81-0.05*20 = 8.81
accel_des[2] = min(accel_des[2], a_down_max)
```
Update `test_downward_error_increases_thrust` to expect thrust ≤ hover (≈ min_thrust) and
`|roll|,|pitch|<0.01`; keep the upward test.

**B-3 [BLOCKING] — SimplePositionTracker uses ENU signs in the NED pipeline**
`control/mpc_tracker.py:446-447` (reachable via `use_geometric_tracker=False`). Both angle signs are
inverted for NED+FRD → +x error commands +0.85 rad nose-up → ≈−0.75 g *away* → divergence. The
designated operator fallback must not be a guaranteed crash. **Fix:** `desired_pitch =
-math.atan2(ax_body, c.gravity)`, `desired_roll = math.atan2(ay_body, c.gravity)`. Add directional
tests at yaw 0 and π/2 (this tracker DOES use `current_yaw`, which is NED-correct).

**B-4 [BLOCKING] — EKF orientation/gyro-bias unobservable; yaw inits 0 and overwrites good yaw**
`estimation/ekf.py:189-193,222-224,252-253`; `race_pipeline.py:280,355`. (Same finding as Task 1.3
Blocker 4 — coordinate with that.) **Fix:** add `DroneEKF.set_orientation(roll,pitch,yaw)`, slave it
from telemetry every tick before `predict()`; stop overwriting yaw at `:355` (keep `yaw=telem.yaw`);
init with `orientation=` from telemetry. Proper observability redesign (attitude update + F
couplings) is **[deferrable]** (VQ2). Lap-1 runbook: `use_detection=False, use_pnp=False`.

**B-5 [BLOCKING] — every replan re-initializes the EKF (yaw→0, biases wiped, clock rewound)**
`race_pipeline.py:228-236,279-281,515`. `_build_trajectory_from` (shared by configure + replan) ends
with `ekf.initialize(...,timestamp_s=0.0)`; a mid-race replan thus lobotomizes the filter exactly
during recovery. **Fix:** move the 2 EKF-init lines into `configure()` only; replan rebuilds
trajectory only. Test: force a replan, assert `x[ABIAS_IDX]`/`P` not reset. (Recovery-stack Blockers
7-9 are a different task.)

**B-6 — gate-normal pitch sign + axis convention vs the real sim**
- **B-6a [deferrable but do now, 5 lines]:** `_gate_normal` builds `+sin p`; the repo's quaternion
  convention puts body-forward z at **−sin θ**. Flip `sp → -sp` at `sequencer.py:704,712,770`,
  `race_pipeline.py:623`, `gate_pnp.py:353` (verify upright case still gives `down0=[0,0,1]`). VQ1
  gates are upright (pitch=0) so it's masked today — becomes BLOCKING on any pitched-gate course.
- **B-6b [BLOCKING — verification/decode rule]:** the decoded VQ1 gates (`q=[0.707,0,0,0.707]`) give
  repo normal `[0,1,0]` (east) but the course runs −X → normal ⟂ travel → **zero gate passes ever
  detected**. The sim's gate opening is along local **+Y** (→ world `[−1,0,0]`, aligned). **This ties
  to Task 1.2:** in `track_data_to_gatespecs`, after conversion assert `normal_k·(pos_{k+1}−pos_k)>0`
  for all consecutive gates; if x-forward fails but y-forward passes, adopt y-forward (`yaw_gate =
  yaw_quat − π/2`). Validate against the recorded VQ1 map before the first lap — data-driven, not a
  guess.

**B-7 — mutually-cancelling bug PAIRS (fix BOTH halves, never one)**
- **Pair 1 [BLOCKING] — racing-line ENU up-vector ↔ masked by `max_vertical_offset=0` + stale cache.**
  `planning/racing_line.py:755-763` hardcodes `up=[0,0,1]` (ENU; in NED that's **down**), currently
  ×0 because `:93 max_vertical_offset=0.0`. The mask is config+cache, not correctness: the cache key
  (`:163-201`) omits `max_vertical_offset`, so a stale `racing_line_cache.json` can apply ±0.35 m
  offsets through the 1.5 m aperture regardless of the bound. **Fix both:** `up=[0,0,-1]` for NED (fix
  the `:742-746` docstring too); add `max_vertical_offset` + an up-vector schema version to the cache
  key (bump v3→v4); pre-lap, purge/disable the racing-line cache and assert all offsets are 0 after
  `optimize()`. Test: `vert_off=+0.5` moves the waypoint **up in NED** (z decreases); v3 cache entry
  misses under the new key.
- **Pair 2 [deferrable] — mirrored synthetic camera ↔ mirrored detection-steering.**
  `flight_control/adapter.py:34-39,53-58` (steer) ↔ `simulation/camera.py:203-223` (render). Not in
  the lap-1 loop (`gate_detection_to_target` is consumed only by PyBullet/sim adapters, never by
  `race_pipeline`) → deferrable, but becomes BLOCKING the moment detection-steer is ported as a VQ1
  backup. The real FPV camera is genuine/unmirrored (61/61 frames), so the mask is gone — reusing the
  steer would diverge. Fix both halves frame-explicitly together; re-validate the legacy demos whose
  "pass" depends on the double bug.

**Deferrable control-adjacent:** terminal/abort hover thrust 0.4 should be `mass·g/max_thrust`
(≈0.49) — every terminal state currently sinks (`race_pipeline.py:380,388,392,401`);
`should_slow_down` cuts lift 30% instead of reducing reference speed (`:481-487`); EKF
`tan(pitch)→copysign(5)` + element-wise covariance clip (`ekf.py:157,203`).

**OUT of scope for Task B:** benchmark honesty (Blockers 15-19), ML residual (20-21, keep
`use_residual=False`), PyBullet-only frame bugs, recovery-stack semantics (7-9), trajectory
feasibility (10-11), PnP aperture (12), sim-time/TIMESYNC/COLLISION transport (13 — owned by 1.1/1.5).

**Order within B:** B-1 → B-2 → B-3 (same file) → B-4 → B-5 (same callback) → B-6a → B-6b (verify vs
the VQ1 capture) → B-7 Pair 1 → B-7 Pair 2. Full `pytest` after each; only intentionally-changed
assertions: `test_downward_error_increases_thrust` + the SimplePositionTracker directional tests.

---

## 5. Definition of done — Phase 1 (pre-live-lap)
- `competition/aigp_mavlink.py` implements `CompetitionInterface`; replay test green; canonical VQ1
  fixture captured + committed (gzipped).
- `competition/track_data.py` produces correct `GateSpec`s (normals along the flight axis, verified
  by the `normal·course>0` rule); VQ1 end-to-end test green.
- State wiring: provided pose → EKF, replay tracks logged pose (RMS ≤0.10 m); orientation passthrough.
- Task B blocking bugs (B-1..B-5, B-6b, B-7 Pair 1) fixed; enshrining tests updated; full `pytest`
  green.
- **Then (separate, PLAN 1.5/1.6):** a runner script + first live closed-loop lap on the sim host —
  arm → fetch map → configure → fly VQ1, confirm ≥1 gate via `active_gate_index`. Capture every live
  run with the recorder (cheap offline iteration).
