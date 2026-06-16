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
        self._indi_debug: Optional[Dict] = None
        self._reassembler = TrackInfoReassembler()
        # Diagnostics for the DSQ investigation (iter-39): the sim announces
        # human-readable verdicts (e.g. disqualification) via STATUSTEXT, which
        # we previously dropped silently. Capture them, and log every OTHER
        # message type the first time we see it so a DSQ on any unexpected
        # channel becomes visible instead of invisible.
        self._status_texts: Deque[Dict] = deque(maxlen=256)
        self._seen_msg_types: set = set()

        self._last_heartbeat_monotonic = 0.0
        self._armed = False
        self._have_attitude = False
        self._have_lpn = False
        self._have_odometry = False

        # The live AIGP sim MISHANDLES SET_ATTITUDE_TARGET attitude mode
        # (type_mask 0b111): a held level attitude makes the drone spin up to
        # ~9 rad/s the moment it is airborne (bench-confirmed 2026-06-13,
        # scripts/aigp_bench.py). It DOES honor body-rate mode (mask 128).
        # So `send_attitude` converts the commanded attitude into a body-rate
        # setpoint via a quaternion attitude-error P loop and sends rate mode.
        # Off-switch + gains exposed for tuning / fallback.
        self._use_rate_control = True
        # Inner attitude->rate loop gains, RE-tuned 2026-06-13 (iter-022) after
        # an isolation bench overturned the prior tuning. Key findings
        # (scripts/aigp_bench.py): (1) pure zero body-rate is perfectly clean
        # (gyro p95~0); (2) the old (2.0,2.5,1.0) loop LIMIT-CYCLES at ~9 Hz
        # (gyro p95~4.5) and the jitter rectifies thrust into a runaway climb
        # — this was THE flight failure, not the trajectory; (3) per-axis rate
        # ID shows the sim sign-flips all axes (so _rate_sign=-1 is right) but
        # the gain is axis-dependent (~1.0 roll, ~2.1 pitch/yaw), NOT a uniform
        # 2.5x. The old loop gain (2*kp * ~2.1 amp = ~8.4) was far past the
        # delay-limited stability margin -> the limit cycle. Cutting the gain
        # ~4x removes the oscillation AND the climb while still tracking a 0.3
        # rad roll step cleanly (gyro p95<0.6, no flip). kd is the SAME sign as
        # before (genuine damping: the sim's flip applies to the kd term too,
        # so it stays negative feedback) — just much smaller.
        # PER-AXIS body-rate P/D gains (roll, pitch, yaw). The sim amplifies
        # the rate channels asymmetrically (~1.0x roll vs ~2.1x pitch/yaw,
        # bench rate-ID), so a uniform kp=0.5 leaves ROLL at HALF the
        # closed-loop bandwidth of pitch — roll under-tracks (0.46x amplitude,
        # ~0.6s lag, captures min_v28) and the cross-track centering oscillates
        # and clips frames at cruise >=5. iter-32: raise ROLL only to kp=1.0
        # (effective gain ~1.0, matching pitch's proven-safe ~1.05) with kd=0.4
        # to damp the now-faster roll loop; pitch/yaw unchanged (eff ~1.05,
        # already crisp+stable). Watch gyro p95 (<1.0 clean, abort >2.0).
        # iter-44 TESTED LIVE & FALSIFIED — KEEP 1.0/0.4/0.8. To make the fast
        # gate5 turn (achieved roll only 0.53x commanded), raising the ROLL loop
        # to kp 1.3 / kd 0.55 / rate-clamp 1.1 made tracking WORSE, not better:
        # achieved/cmd roll fell 0.53 -> 0.33, gyro p95 rose 1.0 -> 1.62, and it
        # nearly clipped gate0 (0.049 m). The higher kp+clamp under-damped the
        # cascade into oscillation (the 9 Hz limit cycle the kd term suppresses;
        # the adversarial workflow predicted exactly this). The 0.53 roll
        # attenuation is the sim's INHERENT behaviour — fighting it with gains
        # destabilises. The fast-turn cap must be solved by reducing the REQUIRED
        # turn (racing line / variable speed), not by more inner-loop bandwidth.
        self._att_rate_kp = (1.0, 0.5, 0.5)   # (roll, pitch, yaw)
        self._att_rate_kd = (0.4, 0.2, 0.2)
        # iter-51 TESTED & NEUTRAL — keep 0.8. Raising ONLY the roll rate clamp
        # to 1.0 (kp/kd unchanged) is SAFE (gyro p95/max unchanged at 1.16/2.13,
        # NO limit cycle — so the iter-44 limit cycle was the kp, not the clamp),
        # but it does NOT help: the roll rate rarely reaches 0.8 on the slalom
        # turns, so the lateral undershoot (the 50-km/h clearance limiter at
        # gates 2/3) is TILT/attenuation-limited, not rate-limited. Reverted.
        self._att_rate_max = 0.8      # rad/s clamp per axis
        # Per-axis CLOSED-LOOP sign, corrected 2026-06-13 (iter-023). The
        # open-loop gyro probe suggested all three axes were sign-flipped, but
        # that conflated the body-rate actuator sign with the spawn yaw=pi
        # frame rotation and the euler-rate mapping. The CLOSED loop is the
        # ground truth, and it disagrees per axis:
        #   * ROLL  (-1): bench att-hold commanding roll +0.30 drove measured
        #     roll to +0.26 — converges. Correct.
        #   * PITCH (+1): commanding pitch -0.50 drove measured pitch the WRONG
        #     way (-0.31 -> -0.08), and in flight pitch -0.62 diverged to +1.5
        #     (positive feedback) — the cause of "hover-stable, flips the moment
        #     it translates". Un-flip pitch so the loop is negative feedback.
        #   * YAW   (-1): held cleanly at pi in every run (never excited with a
        #     real error); left at -1. Revisit if yaw drifts after this fix.
        # Two independent Opus reviews + the bench/flight captures all point to
        # the pitch axis being the single inverted sign.
        self._rate_sign = (-1.0, 1.0, -1.0)

        # --- OPT-IN measured-accel INDI inner loop (roadmap #2) -------------
        # OFF by default. When _use_indi is True, send_attitude computes the
        # body-rate setpoint via control.indi_inner_loop.IndiInnerLoop (filtered
        # gyro-derivative inversion + online-G) INSTEAD of the PD law in
        # _attitude_error_body_rates. It STILL applies self._rate_sign and sends
        # rates mode exactly as the PD path, so the only difference is how the
        # rate vector is produced. When False, the code path below is unchanged
        # (byte-identical to the validated champion PD path). The INDI object is
        # lazily built on first use so importing this module never requires the
        # control package. See the module docstring for the discriminator
        # read-out ("recovered => mismatch; still clamped => bandwidth limit").
        self._use_indi = False
        self._indi_config = None  # optional control.indi_inner_loop.IndiConfig
        self._indi = None         # lazily-built IndiInnerLoop
        self._indi_last_t_us: Optional[int] = None

    def _ensure_indi(self):
        """Lazily construct the IndiInnerLoop (kept out of __init__ so importing
        this module does not pull in the control package)."""
        if self._indi is None:
            from control.indi_inner_loop import IndiInnerLoop
            # Default the INDI rate clamp to the SAME envelope as the PD path
            # (_att_rate_max) so the opt-in branch never commands outside the
            # validated rate range, unless the caller supplied an explicit cfg.
            cfg = self._indi_config
            if cfg is None:
                from control.indi_inner_loop import IndiConfig
                cfg = IndiConfig(max_rate=self._att_rate_max)
            self._indi = IndiInnerLoop(cfg)
        return self._indi

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

        # The sim mishandles attitude mode (it spins) — convert the desired
        # attitude into a body-rate command it DOES honor. See __init__.
        if self._use_rate_control:
            with self._state_lock:
                telem = self._latest_telem
            q_cur = telem.orientation if (telem and telem.orientation) else Quaternion()
            omega = (telem.angular_velocity if (telem and telem.angular_velocity)
                     else (0.0, 0.0, 0.0))
            if self._use_indi:
                # OPT-IN: measured-accel INDI inner loop. dt from telemetry
                # timestamps (us). Produces the SAME (roll,pitch,yaw) rate
                # setpoint contract as the PD law; sign + send below are
                # identical. See control/indi_inner_loop.py.
                t_us = telem.timestamp_us if telem else None
                if t_us is not None and self._indi_last_t_us is not None:
                    dt = (t_us - self._indi_last_t_us) * 1e-6
                else:
                    dt = 0.0  # first tick (or no stamp): INDI guard holds command
                self._indi_last_t_us = t_us
                indi = self._ensure_indi()
                rr, pr, yr = indi.compute(q_cur, q, omega=omega, dt=dt)
                with self._state_lock:
                    self._indi_debug = indi.debug_dict()
            else:
                # Desired body rate in our FRD convention (kp on attitude error,
                # kd damping on measured gyro — gyro is FRD-consistent).
                rr, pr, yr = _attitude_error_body_rates(
                    q_cur, q, omega=omega, kp=self._att_rate_kp,
                    kd=self._att_rate_kd, max_rate=self._att_rate_max,
                )
            sx, sy, sz = self._rate_sign  # sim applies rates with opposite sign
            with self._send_lock:
                self._conn.mav.set_attitude_target_send(
                    self._time_boot_ms(),
                    self._target_system,
                    self._target_component,
                    SET_ATTITUDE_TARGET_MASK_RATES_THRUST,
                    [1.0, 0.0, 0.0, 0.0],
                    sx * rr, sy * pr, sz * yr,
                    thrust,
                )
            return

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
        sx, sy, sz = self._rate_sign  # sim applies body rates with opposite sign
        with self._send_lock:
            self._conn.mav.set_attitude_target_send(
                self._time_boot_ms(),
                self._target_system,
                self._target_component,
                SET_ATTITUDE_TARGET_MASK_RATES_THRUST,
                [1.0, 0.0, 0.0, 0.0],
                sx * cmd.roll_rate,
                sy * cmd.pitch_rate,
                sz * cmd.yaw_rate,
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

    @property
    def indi_debug(self) -> Optional[Dict]:
        """Latest INDI inner-loop debug snapshot (None unless _use_indi is on).

        Mirrors ``actuator_outputs`` so the recorder/runner can log the INDI
        read-out (alpha_des, alpha_meas, Ghat, saturation flags, u) per tick.
        """
        with self._state_lock:
            return self._indi_debug

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
            elif msg_type == "STATUSTEXT":
                self._handle_statustext(msg)
            else:
                # First-sighting log of any unhandled type. The sim may report a
                # disqualification (DSQ) or other verdict on a channel we don't
                # decode; surface it once rather than dropping it silently.
                if msg_type not in self._seen_msg_types:
                    self._seen_msg_types.add(msg_type)
                    logger.info("AIGP: first %s message seen (unhandled): %s",
                                msg_type, msg.to_dict() if hasattr(msg, "to_dict") else msg)
        except Exception:
            logger.exception("AIGP MAVLink message handler failed")

    def _handle_statustext(self, msg) -> None:
        """Capture + log STATUSTEXT. The DSQ verdict (if the sim sends one over
        MAVLink) almost certainly arrives here. Always log it at WARNING so it
        is impossible to miss in a run's output."""
        text = getattr(msg, "text", "")
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8", "replace")
        text = str(text).strip("\x00").strip()
        severity = getattr(msg, "severity", None)
        with self._state_lock:
            self._status_texts.append({
                "severity": severity,
                "text": text,
                "monotonic": time.monotonic(),
            })
        logger.warning("AIGP STATUSTEXT (sev=%s): %s", severity, text)

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


def _attitude_error_body_rates(q_cur, q_des, omega=(0.0, 0.0, 0.0),
                               kp=5.0, kd=0.0, max_rate=4.0):
    """PD body-rate command (FRD) that drives q_cur toward q_des.

    The error quaternion ``q_err = conj(q_cur) (x) q_des`` is expressed in the
    body frame; its vector part is ``sin(theta/2)*axis``, so ``2*kp*vec`` is a
    proportional, singularity-free body-rate (euler-error control cross-couples
    when tilted). The ``-kd*omega`` term damps the cascade (sim rate loop + our
    P loop limit-cycles ~5 Hz without it). Shortest-path via ``w >= 0``.
    Returns (roll_rate, pitch_rate, yaw_rate) in FRD, clamped to +/- max_rate.
    The CALLER applies the sim's per-axis rate sign — see __init__._rate_sign.

    ``kp``/``kd`` may be scalars or 3-tuples (per-axis roll/pitch/yaw). PER-AXIS
    gains matter because the sim amplifies the rate channels asymmetrically
    (~1.0x roll vs ~2.1x pitch/yaw, bench-measured), so a single kp leaves the
    ROLL loop at half the closed-loop bandwidth of pitch — roll under-tracks
    (0.46x amplitude, ~0.6s lag) and the cross-track centering oscillates at
    speed. Raising ONLY roll's kp equalises the bandwidth.

    Used because the AIGP sim honors body-rate (mask 128) but spins under
    attitude mode (mask 7) — see AIGPMavlinkAdapter.__init__.
    """
    kpx, kpy, kpz = (kp, kp, kp) if isinstance(kp, (int, float)) else kp
    kdx, kdy, kdz = (kd, kd, kd) if isinstance(kd, (int, float)) else kd
    qc = (q_cur.w, q_cur.x, q_cur.y, q_cur.z)
    qd = (q_des.w, q_des.x, q_des.y, q_des.z)
    # conj(qc) (x) qd
    cw, cx, cy, cz = qc[0], -qc[1], -qc[2], -qc[3]
    ew = cw * qd[0] - cx * qd[1] - cy * qd[2] - cz * qd[3]
    ex = cw * qd[1] + cx * qd[0] + cy * qd[3] - cz * qd[2]
    ey = cw * qd[2] - cx * qd[3] + cy * qd[0] + cz * qd[1]
    ez = cw * qd[3] + cx * qd[2] - cy * qd[1] + cz * qd[0]
    if ew < 0:
        ex, ey, ez = -ex, -ey, -ez
    rates = (
        2.0 * kpx * ex - kdx * omega[0],
        2.0 * kpy * ey - kdy * omega[1],
        2.0 * kpz * ez - kdz * omega[2],
    )
    # max_rate may be a scalar or a 3-tuple (per-axis). ROLL gets more rate
    # headroom (iter-44): at a fast slalom/finish turn the roll command saturates
    # the clamp and builds too slowly (~1 s to 0.8 rad), so achieved roll is only
    # ~0.53x commanded -> under-turn -> over-command -> tumble (gate5 @ base 15.5,
    # gyro 33). A higher ROLL clamp lets the bank build in time; pitch/yaw stay
    # at the proven 0.8 (they were never the bottleneck).
    mxx, mxy, mxz = (max_rate, max_rate, max_rate) if isinstance(
        max_rate, (int, float)) else max_rate
    return (
        max(-mxx, min(mxx, rates[0])),
        max(-mxy, min(mxy, rates[1])),
        max(-mxz, min(mxz, rates[2])),
    )


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
