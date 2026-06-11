"""Raw pymavlink transport for the official AIGP simulator.

The official sim speaks MAVLink2 on UDP 14550 plus a separate JPEG-over-UDP
vision stream. This adapter deliberately uses pymavlink directly so custom
``ENCAPSULATED_DATA`` race-status and track-info packets are visible.

``pymavlink`` is imported lazily in :meth:`connect`; importing this module and
running unit tests does not require the package or a live socket.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from competition.adapter import (
    AttitudeCommand,
    AttitudeRateCommand,
    CameraFrame,
    CompetitionInterface,
    IMUData,
    PositionCommand,
    Quaternion,
    TelemetryState,
)
from competition.aigp_geometry import AIGP_CAM_UDP_PORT
from competition.aigp_messages import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    RaceStatus,
    TrackData,
    TrackInfoReassembler,
    parse_race_status,
)
from competition.aigp_recorder import (
    mavlink_msg_to_fields,
    race_status_fields,
    record_for_message,
    track_data_fields,
    write_jsonl,
)
from competition.vision_udp import VisionUdpListener

logger = logging.getLogger(__name__)

DEFAULT_MAVLINK_URL = "udpin:127.0.0.1:14550"
SIM_RESET_COMMAND = 31000
MAV_CMD_COMPONENT_ARM_DISARM = 400
SET_ATTITUDE_TARGET_MASK_ATTITUDE_THRUST = 0b00000111
SET_ATTITUDE_TARGET_MASK_RATES_THRUST = 128
SET_POSITION_TARGET_LOCAL_NED_MASK = 2496
MAV_FRAME_LOCAL_NED = 1


class AIGPMavlinkAdapter(CompetitionInterface):
    """CompetitionInterface implementation for the official AIGP sim."""

    def __init__(
        self,
        *,
        enable_vision: bool = True,
        vision_port: int = AIGP_CAM_UDP_PORT,
        require_track: bool = True,
        track_retries: int = 3,
    ) -> None:
        self.enable_vision = enable_vision
        self.require_track = require_track
        self.track_retries = int(track_retries)

        self._conn = None
        self._target_system = 1
        self._target_component = 1
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_event = threading.Event()
        self._telemetry_ready_event = threading.Event()
        self._track_event = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._announce_thread: Optional[threading.Thread] = None
        self._vision: Optional[VisionUdpListener] = (
            VisionUdpListener(port=vision_port) if enable_vision else None
        )

        self._latest_telem: Optional[TelemetryState] = None
        self._race_status: Optional[RaceStatus] = None
        self._track_data: Optional[TrackData] = None
        self._collisions: Deque[Dict] = deque(maxlen=128)
        self._actuator_outputs: Optional[Dict] = None
        self._reassembler = TrackInfoReassembler()

        self._last_heartbeat_monotonic = 0.0
        self._armed = False
        self._have_attitude = False
        self._have_lpn = False
        self._have_odometry = False

    async def connect(self, address: str = DEFAULT_MAVLINK_URL) -> None:
        """Open the UDP MAVLink socket, announce as GCS, then fetch track data.

        The RX thread starts before any heartbeat wait so a connect-time
        track transfer cannot be discarded by ``wait_heartbeat()``.

        Idempotent: if already connected, returns immediately without spawning
        duplicate threads or re-fetching track data. ``RaceSession`` calls
        ``connect()`` unconditionally; the runner calls it first with the
        correct address, so this guard is required.
        """
        if self._conn is not None:
            return

        try:
            from pymavlink import mavutil
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError("pymavlink is required for live AIGP transport") from exc

        self._conn = mavutil.mavlink_connection(address)
        self._target_system = getattr(self._conn, "target_system", 1) or 1
        self._target_component = getattr(self._conn, "target_component", 1) or 1
        self._stop_event.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="aigp-mavlink-rx", daemon=True)
        self._announce_thread = threading.Thread(
            target=self._announce_loop,
            name="aigp-mavlink-announce",
            daemon=True,
        )
        self._rx_thread.start()
        self._announce_thread.start()

        heartbeat_ok = await asyncio.to_thread(self._heartbeat_event.wait, 10.0)
        if not heartbeat_ok:
            raise ConnectionError("No AIGP heartbeat received. Is the sim in Virtual Qualifier mode?")
        telemetry_ok = await asyncio.to_thread(self._telemetry_ready_event.wait, 10.0)
        if not telemetry_ok:
            raise ConnectionError("AIGP telemetry did not become ready")

        if self._vision is not None:
            try:
                await self._vision.start()
            except OSError:
                logger.exception("Could not start AIGP vision listener")

        if self._track_data is None:
            for _ in range(max(1, self.track_retries)):
                await self._send_sim_reset(clear_track_event=True)
                if await asyncio.to_thread(self._track_event.wait, 5.0):
                    break
            if self._track_data is None and self.require_track:
                raise ConnectionError("AIGP track data not received after SIM_RESET")
            if self._track_data is None:
                logger.warning("AIGP track data not received after SIM_RESET")

    async def disconnect(self) -> None:
        self._stop_event.set()
        for thread in (self._rx_thread, self._announce_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self._rx_thread = None
        self._announce_thread = None
        if self._vision is not None:
            await self._vision.stop()
        if self._conn is not None:
            close = getattr(self._conn, "close", None)
            if close is not None:
                close()
        self._conn = None

    async def arm(self) -> None:
        if self._armed:
            return
        self._require_conn()
        with self._send_lock:
            self._conn.mav.command_long_send(
                self._target_system,
                self._target_component,
                MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        if not self._armed:
            logger.warning("Arm command sent but vehicle still reports disarmed")

    async def start_offboard(self) -> None:
        """No-op: the sim accepts setpoints without a PX4 offboard handshake."""

    async def stop_offboard(self) -> None:
        """No-op: the sim accepts setpoints without a PX4 offboard handshake."""

    async def get_telemetry(self) -> TelemetryState:
        with self._state_lock:
            if self._latest_telem is None:
                raise RuntimeError("No telemetry received yet")
            return self._latest_telem

    async def get_camera_frame(self) -> Optional[CameraFrame]:
        return self._vision.latest_frame() if self._vision is not None else None

    async def send_attitude(self, cmd: AttitudeCommand) -> None:
        self._require_conn()
        thrust = _clamp_thrust(cmd.thrust)
        q = Quaternion.from_euler(cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad)
        # Canonicalize: MAVLink requires w >= 0 (positive scalar convention).
        # Double-cover means q and -q represent the same rotation, but a
        # negative w component can cause sim-side interpolation glitches.
        if q.w < 0:
            q = Quaternion(-q.w, -q.x, -q.y, -q.z)
        with self._send_lock:
            self._conn.mav.set_attitude_target_send(
                self._time_boot_ms(),
                self._target_system,
                self._target_component,
                SET_ATTITUDE_TARGET_MASK_ATTITUDE_THRUST,
                [q.w, q.x, q.y, q.z],
                0.0,
                0.0,
                0.0,
                thrust,
            )

    async def send_attitude_rate(self, cmd: AttitudeRateCommand) -> None:
        self._require_conn()
        thrust = _clamp_thrust(cmd.thrust)
        with self._send_lock:
            self._conn.mav.set_attitude_target_send(
                self._time_boot_ms(),
                self._target_system,
                self._target_component,
                SET_ATTITUDE_TARGET_MASK_RATES_THRUST,
                [1.0, 0.0, 0.0, 0.0],
                cmd.roll_rate,
                cmd.pitch_rate,
                cmd.yaw_rate,
                thrust,
            )

    async def send_position(self, cmd: PositionCommand) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED for parity only.

        First-contact live testing showed this path does not track velocity
        cleanly and can produce runaway climb. Use attitude targets for VQ1.
        """
        self._require_conn()
        n, e, d = cmd.position_ned
        vn, ve, vd = cmd.velocity_ned
        with self._send_lock:
            self._conn.mav.set_position_target_local_ned_send(
                self._time_boot_ms(),
                self._target_system,
                self._target_component,
                MAV_FRAME_LOCAL_NED,
                SET_POSITION_TARGET_LOCAL_NED_MASK,
                n,
                e,
                d,
                vn,
                ve,
                vd,
                0.0,
                0.0,
                0.0,
                cmd.yaw_rad,
                0.0,
            )

    async def reset(self) -> Optional[TrackData]:
        await self._send_sim_reset(clear_track_event=True)
        await asyncio.to_thread(self._track_event.wait, 5.0)
        return self.track_data

    async def wait_for_track_data(self, timeout_s: float = 10.0) -> Optional[TrackData]:
        await asyncio.to_thread(self._track_event.wait, timeout_s)
        return self.track_data

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and (time.monotonic() - self._last_heartbeat_monotonic) < 3.0

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def latest_telemetry(self) -> Optional[TelemetryState]:
        with self._state_lock:
            return self._latest_telem

    @property
    def race_status(self) -> Optional[RaceStatus]:
        with self._state_lock:
            return self._race_status

    @property
    def track_data(self) -> Optional[TrackData]:
        with self._state_lock:
            return self._track_data

    @property
    def actuator_outputs(self) -> Optional[Dict]:
        with self._state_lock:
            return self._actuator_outputs

    def drain_collisions(self):
        with self._state_lock:
            out = list(self._collisions)
            self._collisions.clear()
            return out

    async def _send_sim_reset(self, clear_track_event: bool = False) -> None:
        self._require_conn()
        if clear_track_event:
            self._track_event.clear()
        with self._send_lock:
            self._conn.mav.command_long_send(
                self._target_system,
                self._target_component,
                SIM_RESET_COMMAND,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def _handle_message(self, msg) -> None:
        try:
            msg_type = msg.get_type()
            if msg_type == "BAD_DATA":
                return
            if msg_type == "HEARTBEAT":
                self._handle_heartbeat(msg)
            elif msg_type == "LOCAL_POSITION_NED":
                self._handle_local_position(msg)
            elif msg_type == "ODOMETRY":
                self._handle_odometry(msg)
            elif msg_type == "ATTITUDE":
                self._handle_attitude(msg)
            elif msg_type == "HIGHRES_IMU":
                self._handle_highres_imu(msg)
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self._handle_actuator(msg)
            elif msg_type == "COLLISION":
                self._handle_collision(msg)
            elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
                self._reassembler.begin_transfer(msg.width, msg.packets)
            elif msg_type == "ENCAPSULATED_DATA":
                self._handle_encapsulated(msg)
        except Exception:
            logger.exception("AIGP MAVLink message handler failed")

    def _handle_heartbeat(self, msg) -> None:
        with self._state_lock:
            self._last_heartbeat_monotonic = time.monotonic()
            self._armed = bool(msg.base_mode & 0x80)
            if self._conn is not None:
                self._target_system = getattr(self._conn, "target_system", self._target_system) or self._target_system
                self._target_component = getattr(self._conn, "target_component", self._target_component) or self._target_component
            self._heartbeat_event.set()

    def _handle_local_position(self, msg) -> None:
        with self._state_lock:
            old = self._latest_telem
            self._latest_telem = _telem_with(
                old,
                timestamp_us=int(msg.time_boot_ms) * 1000,
                position_ned=(msg.x, msg.y, msg.z),
                velocity_ned=(msg.vx, msg.vy, msg.vz),
                lpn_time_boot_ms=int(msg.time_boot_ms),
            )
            self._have_lpn = True
            self._maybe_ready()

    def _handle_odometry(self, msg) -> None:
        q = msg.q
        with self._state_lock:
            old = self._latest_telem
            self._latest_telem = _telem_with(
                old,
                timestamp_us=int(msg.time_usec),
                orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]),
                odom_time_usec=int(msg.time_usec),
                odom_quality=getattr(msg, "quality", None),
                odom_reset_counter=getattr(msg, "reset_counter", None),
            )
            self._have_odometry = True

    def _handle_attitude(self, msg) -> None:
        with self._state_lock:
            orientation = None
            if not self._have_odometry:
                orientation = Quaternion.from_euler(msg.roll, msg.pitch, msg.yaw)
            old = self._latest_telem
            self._latest_telem = _telem_with(
                old,
                timestamp_us=int(msg.time_boot_ms) * 1000,
                orientation=orientation,
                angular_velocity=(msg.rollspeed, msg.pitchspeed, msg.yawspeed),
            )
            self._have_attitude = True
            self._maybe_ready()

    def _handle_highres_imu(self, msg) -> None:
        imu = IMUData(
            timestamp_us=int(msg.time_usec),
            accel=(msg.xacc, msg.yacc, msg.zacc),
            gyro=(msg.xgyro, msg.ygyro, msg.zgyro),
            mag=None,
        )
        with self._state_lock:
            self._latest_telem = _telem_with(self._latest_telem, imu=imu)

    def _handle_actuator(self, msg) -> None:
        with self._state_lock:
            self._actuator_outputs = {
                "time_usec": getattr(msg, "time_usec", None),
                "active": getattr(msg, "active", None),
                "actuator": list(getattr(msg, "actuator", [])),
            }

    def _handle_collision(self, msg) -> None:
        with self._state_lock:
            self._collisions.append({
                "id": msg.id,
                "threat_level": msg.threat_level,
                "impulse": msg.horizontal_minimum_delta,
            })

    def _handle_encapsulated(self, msg) -> None:
        payload = bytes(msg.data)
        if not payload:
            return
        data_type = payload[0]
        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            race_status = parse_race_status(payload)
            with self._state_lock:
                self._race_status = race_status
            return
        if data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            if len(payload) < 3:
                return
            transfer_id = int.from_bytes(payload[1:3], "little")
            track = self._reassembler.feed_chunk(transfer_id, msg.seqnr, payload[3:])
            if track is not None:
                with self._state_lock:
                    self._track_data = track
                    self._track_event.set()

    def _maybe_ready(self) -> None:
        if self._have_attitude and self._have_lpn:
            self._telemetry_ready_event.set()

    def _rx_loop(self) -> None:  # pragma: no cover - live socket loop
        while not self._stop_event.is_set():
            try:
                msg = self._conn.recv_match(blocking=True, timeout=0.5)
            except Exception:
                logger.exception("AIGP MAVLink recv failed")
                continue
            if msg is not None:
                self._handle_message(msg)

    def _announce_loop(self) -> None:  # pragma: no cover - live socket loop
        while not self._stop_event.is_set():
            try:
                now_ns = time.time_ns()
                with self._send_lock:
                    self._conn.mav.timesync_send(0, now_ns)
                    self._conn.mav.heartbeat_send(6, 8, 0, 0, 4)
            except Exception:
                logger.exception("AIGP MAVLink announce failed")
            self._stop_event.wait(0.1)

    def _time_boot_ms(self) -> int:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF

    def _require_conn(self) -> None:
        if self._conn is None:
            raise RuntimeError("AIGP MAVLink adapter is not connected")


def _clamp_thrust(thrust: float) -> float:
    if not math.isfinite(thrust):
        raise ValueError("thrust must be finite")
    return max(0.0, min(1.0, thrust))


def _default_telem() -> TelemetryState:
    return TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
    )


def _telem_with(old: Optional[TelemetryState], **updates) -> TelemetryState:
    base = old or _default_telem()
    values = {
        "timestamp_us": base.timestamp_us,
        "position_ned": base.position_ned,
        "velocity_ned": base.velocity_ned,
        "orientation": base.orientation,
        "angular_velocity": base.angular_velocity,
        "imu": base.imu,
        "lpn_time_boot_ms": base.lpn_time_boot_ms,
        "odom_time_usec": base.odom_time_usec,
        "odom_quality": base.odom_quality,
        "odom_reset_counter": base.odom_reset_counter,
    }
    for key, value in updates.items():
        if value is not None:
            values[key] = value
    return TelemetryState(**values)


def _iter_records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def main(argv=None) -> None:  # pragma: no cover - live CLI
    parser = argparse.ArgumentParser(description="AIGP pymavlink transport utility")
    parser.add_argument("--record", default=None, help="write JSONL capture to this path")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--jpeg-dir", default=None)
    parser.add_argument("--attitude-test", action="store_true")
    args = parser.parse_args(argv)

    async def _run():
        adapter = AIGPMavlinkAdapter(enable_vision=not args.no_vision)
        await adapter.connect()
        if args.attitude_test:
            await adapter.send_attitude(AttitudeCommand(0.0, 0.0, 0.0, 0.5))
        if args.record:
            records = []
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                await asyncio.sleep(0.02)
                telem = adapter.latest_telemetry
                if telem is not None:
                    records.append(record_for_message("telemetry_snapshot", {
                        "timestamp_us": telem.timestamp_us,
                        "position_ned": list(telem.position_ned),
                        "velocity_ned": list(telem.velocity_ned),
                    }, time.time_ns()))
            with open(args.record, "w") as out:
                write_jsonl(records, out)
        await adapter.disconnect()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
