# AIGP Decision Log

## D1 — Transport library: pymavlink (Task 0.4)
**Decision:** Use **pymavlink** (`udpin:<ip>:14550`), not MAVSDK.
**Rationale:** matches the official `PyAIPilotExample`; cleanly exposes the sim's custom
`ENCAPSULATED_DATA` (race-status + track-data) and `DATA_TRANSMISSION_HANDSHAKE` track transfer,
which MAVSDK may hide. Confirmed working on first contact (2026-06-10): heartbeat + full
telemetry + gate-map decode + control all via pymavlink.

## D2 — We own the control loop (SET_ATTITUDE_TARGET), not the sim
**Decision:** Drive the drone with `SET_ATTITUDE_TARGET` from our SE(3) geometric tracker;
calibrate thrust/gains to the real sim.
**Rationale:** first-contact control test showed the sim has **no turnkey velocity autopilot** —
a `SET_POSITION_TARGET_LOCAL_NED` velocity setpoint (`vx=-2, vz=0`) produced an unstable climb
(z→−27 m @ 28 m/s), not tracking. The sim accepts commands and the drone is dynamic, so the link
works; stabilization is ours to provide. (Re-evaluate body-rate+thrust for VQ2 aggression.)

## D3 — Gate map is fetched via SIM_RESET, then cached
**Decision:** On connect, issue `SIM_RESET` (cmd 31000) to trigger the one-shot track-data
transfer, parse it into `List[GateSpec]`, and cache for the run.
**Rationale:** track-data is NOT in the steady stream — it's pushed once (chunked) on scene
init / reset. A passive late-connecting client never receives it. `SIM_RESET` also gives the
clean race start we want. (Recorder/adapter must also log from t=0 so `wait_heartbeat()` doesn't
discard the connect-time transfer.)

## D4 — State source: use the provided LOCAL_POSITION_NED / ODOMETRY
**Decision:** Feed the EKF from the sim's `LOCAL_POSITION_NED`/`ODOMETRY` (populated, ~75–95 Hz,
with covariance + `quality`), rather than vision dead-reckoning, for VQ1.
**Rationale:** the "no local position" assumption that shaped the old stack is false here. Known
map + provided local pose ⇒ VQ1 is fly-the-line, with vision as a backup/correction. Revisit for
VQ2 (confirm whether local pose + map persist with "guidance off").
