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

## D5 — EKF position only; attitude is telemetry passthrough
**Decision:** Use `LOCAL_POSITION_NED` as the sole position/velocity measurement, use
`ODOMETRY.q` as the attitude passthrough plus quality/reset health signal, and keep gate PnP in
backup mode for VQ1.
**Rationale:** the current EKF does not observe orientation, so estimating yaw from blind init can
overwrite good simulator attitude. The sim provides high-rate quaternion attitude with quality 100,
while PnP geometry still uses the inner-aperture model and can poison a healthy state estimate.

## D6 — VQ2 build 3385 uses vision plus HIGHRES_IMU, not VQ1 pose/map
**Decision:** Run the build-3385 training stack with `telemetry_mode="imu"`, disable track fetching,
estimate attitude from `HIGHRES_IMU`, and navigate from the live camera feed.
**Rationale:** live inspection of build 3385 found `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, and
usable gate-map geometry unavailable. Heartbeat, race status, actuator status, collision events,
`HIGHRES_IMU`, and UDP vision remain live. The older D3-D5 choices still describe VQ1 only.

## D7 — Advance live VQ2 control through bounded, reset-proved stages
**Decision:** Permit powered commands only after race and IMU clock rollback prove a fresh reset,
countdown is witnessed, GO has passed, and a newer heartbeat confirms arm. Keep stages bounded and
always reset/disarm in cleanup; hold yaw command at zero until calibrated.
**Rationale:** build 3385 removes pose telemetry, its vision epoch does not reset with race/IMU time,
and the simulator can boot with the heartbeat armed bit set despite zero actuator demand. Explicit
epoch, freshness, and cleanup proofs make early calibration failures observable and fail-closed.

## D8 — Race status is the sole gate-pass authority
**Decision:** A close/full-frame visual approach may arm a crossing-confirmation state, but vision
never declares passage. Once the target disappears, command zero rate/thrust and wait at most
0.40 seconds for a race-status packet with a strictly newer simulator clock. Gate index 1 passes;
a newer index 0, invalid index, timeout, stale stream, or collision aborts.
**Rationale:** the first stable gate-0 trace reset exactly when the next 4 Hz race packet arrived,
discarding its gate index. Preserving that packet proved the same trajectory was a valid,
collision-free pass. Timestamp causality prevents stale gate values from authorizing flight.
