# AIGP First Contact — Phase 0 Findings (2026-06-10)

First successful end-to-end contact with the **official AI-GP simulator** (`FlightSim.exe`,
build v1.0.3364) running the Virtual Qualifier, driven from the Mac over SSH to the
Windows sim host (RTX 3060). This answers the PLAN.md §4 open questions that only the
live binary could settle, and it reshapes the Phase 1+ design. **Net: we are in the
best-case branch — usable local position + the runtime gate map are both provided.**

## How contact was made
- Host: Windows PC `DESKTOP-M5VJ10H`, sim + Python co-located (localhost UDP). Mac =
  brain over SSH (Tailscale personal tailnet). pymavlink 2.4.49 / opencv 4.13 / numpy 2.4.
- The sim only serves the pilot interface in the **Virtual Qualifier** mode (not the
  ACRO free-flight/practice scene). On entering VQ the drone spawns armed at the NED
  origin and the sim streams MAVLink (14550) + JPEG vision (5600) to `127.0.0.1`.

## §4 checklist — answered
| Question | Answer |
|---|---|
| `LOCAL_POSITION_NED` populated? | ✅ **YES**, ~95 Hz, real x/y/z + v. Drone rests at ~(0,0,0.02). |
| `ODOMETRY` populated? | ✅ **YES**, ~74 Hz — pose + quaternion (`q=[w,x,y,z]`), `pose_covariance`/`velocity_covariance` (21-elem), `reset_counter`, `estimator_type=2`, `quality=100`. |
| Track-data (gate map) sent? | ✅ **YES, but only on `SIM_RESET` (cmd 31000)** — a one-shot chunked transfer (`DATA_TRANSMISSION_HANDSHAKE` → track chunks). **Not in the steady stream.** A passive client never sees it; you must reset (or be connected at scene init) to fetch it. |
| Sim accepts `SET_POSITION_TARGET_LOCAL_NED`? | ⚠️ Accepts the message and the drone responds, but **does not cleanly track velocity** — a `vx=-2, vy=0, vz=0` setpoint produced an unstable climb to z=−27 m @ 28 m/s. **No turnkey velocity autopilot.** |
| `SET_ATTITUDE_TARGET`? | Not yet tested — this is the intended path (our SE(3) tracker output). Test in Phase 1.4. |
| Command rate accepted? | Sent setpoints at 50 Hz without rejection; precise max not yet measured (template uses 250 Hz, spec says <100 Hz). |
| Vision res/fps/quality? | ✅ **640×360×3 color** JPEG, ~38 KB/frame, **28 chunks/frame**, ~28k pkt/s firehose. `VisionUdpReceiver` reassembled 61 frames with **0 dropped partials**. |
| Arm + `SIM_RESET` semantics? | Drone spawns armed (`base_mode=193`, `system_status=ACTIVE`). `SIM_RESET` re-spawns at start, resets the race clock, and re-sends the gate map. `active_gate_index=0` = first target gate. |
| `ACTUATOR_OUTPUT_STATUS`? | ✅ ~95 Hz, `active=15` (4 motors), 32-float array → calibration data available (Phase 1.7). |
| NED origin = arm point, +20° cam tilt? | ✅ origin = spawn point (rest ≈ 0,0,0); camera tilt consistent with spec. |

## The VQ1 gate map ("decoded" course)
6 gates, **2.72 m outer** (= spec 2700 mm), all oriented `q=[0.707,0,0,0.707]` (facing the
flight axis). Course runs ~159 m in **−X**, **descending ~26 m** (NED +z down):

| gate | NED x | NED y | NED z |
|---|---|---|---|
| 0 | −23.3 | −0.4 | −0.03 |
| 1 | −46.9 | −2.5 | 5.07 |
| 2 | −74.6 | 1.2 | 13.67 |
| 3 | −111.5 | −5.1 | 24.57 |
| 4 | −135.5 | −0.8 | 25.36 |
| 5 | −159.2 | −4.4 | 25.97 |

(The course is downloaded/deterministic → this map is a legitimate fixture for VQ tuning.)

## Telemetry message catalog (steady state, VQ idle)
`ATTITUDE` ~117 Hz · `HIGHRES_IMU` ~117 Hz (accel/gyro real; mag/baro/temp = NaN) ·
`LOCAL_POSITION_NED` ~95 Hz · `ACTUATOR_OUTPUT_STATUS` ~95 Hz · `ODOMETRY` ~74 Hz ·
`HEARTBEAT` ~10 Hz · `ENCAPSULATED_DATA`(race_status) ~4 Hz. No `COLLISION` at idle.
Byte layouts in `competition/aigp_messages.py` confirmed exact vs the official
`mavlink_rx.py` (incl. `COLLISION.horizontal_minimum_delta` = impulse magnitude, not a delta).

## Vision / perception (the raceline question)
The FPV camera image (5600) **has the blue guidance raceline and red-highlighted gates
painted directly into it**, over a desaturated grayscale world — i.e. the spec's
"desaturated + highlighted" VQ1 look. So:
- The raceline is a **rendered visual, not a telemetry feed** (no raceline message exists;
  the only spatial guidance in telemetry is the gate map).
- It is **VQ1-only** ("visual guidance off" in VQ2). **Do not build on it.** It (and the red
  gate highlight) are usable VQ1 cross-checks, not the foundation.

## Implications for the plan / logic rewrite
1. **Primary state & guidance = provided.** Use `LOCAL_POSITION_NED`/`ODOMETRY` for state
   and the `SIM_RESET`-fetched gate map for the plan. VQ1 completion is fly-the-known-line,
   not a vision problem. The "no local position" fear that shaped the old stack is moot.
2. **We own the control loop.** No usable velocity/position autopilot in the sim → drive
   `SET_ATTITUDE_TARGET` from our SE(3) tracker; calibrate thrust/gains to real physics
   (the PyBullet-fit `drone_spec` constants are wrong, as flagged).
3. **Transport: pymavlink** (confirmed; cleanly exposes the custom encapsulated messages).
4. **Recorder/adapter must announce + `SIM_RESET`** to capture the gate map — and must log
   from t=0 (the current recorder's `wait_heartbeat()` eats the connect-time track-data).
5. **Perception ladder:** VQ1 = red-gate color threshold + raceline as backup to the map;
   VQ2 = learned gate detector (guidance off, realistic textures), gate map TBD-confirm.

## Status of Phase 0 tasks
- 0.1 host ✅ (own RTX PC; deps + repo in place) · 0.2 link + commands-move-drone ✅
- 0.3 recorder ✅ (fixed: vision capture + full gate-map + generic/crash-safe field dump;
  validated live; needs the announce/reset + from-t0 enhancement noted above)
- 0.4 transport decision ✅ pymavlink
