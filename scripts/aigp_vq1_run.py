"""Phase 1.5/1.6 entry-point for the AIGP VQ1 autonomous race.

Usage (on the Windows sim host over SSH, with VQ mode active):

    python3 scripts/aigp_vq1_run.py [--address udpin:127.0.0.1:14550]
                                     [--max-speed 8.0]
                                     [--record captures/vq1.jsonl.gz]
                                     [--dry-run]

The ``--dry-run`` flag substitutes a ``FakeAdapter`` for offline testing;
it exercises the full configure→reset→run flow without the sim.

Race flow
---------
1. Connect adapter and wait for heartbeat + telemetry (sim must be in VQ mode).
2. Fetch the track-gate map via SIM_RESET and convert to ``GateSpec`` objects.
3. Configure the ``RacePipeline`` from the gate map.
4. Issue SIM_RESET again to give the race clock a clean start *after* the
   trajectory has been pre-computed (trajectory optimisation can take ~1-2 s).
5. Run the pipeline; the session loop sends attitude setpoints at ~100 Hz until
   ``sequencer.is_complete`` (or timeout/crash/DQ).
6. Print a completion summary (gates passed, collisions, elapsed wall time).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from typing import List, Optional

import gzip
import json

from competition.adapter import AttitudeCommand, CameraFrame, CompetitionInterface, Quaternion, TelemetryState
from competition.aigp_messages import TrackData, TrackGate
from competition.track_data import track_data_to_gatespecs
from gate_sequencing.sequencer import GateSpec
from race_pipeline import PipelineConfig, RacePipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fake adapter for --dry-run (no live sim needed)
# ---------------------------------------------------------------------------

class FakeAdapter(CompetitionInterface):
    """Minimal fake adapter for offline ``--dry-run`` testing.

    Simulates the minimum sequence the runner exercises:
    - ``connect()`` records the address.
    - ``wait_for_track_data()`` returns a fixed 6-gate VQ1 map.
    - ``reset()`` returns the same map.
    - ``get_telemetry()`` returns a static telemetry snapshot.
    - ``send_attitude()`` is a no-op (captures call count).
    - ``drain_collisions()`` always returns an empty list.
    """

    # VQ1 first-contact gate map (NED, metres) from 2026-06-10 Phase 0 capture.
    VQ1_GATES: List[tuple] = [
        (0, (-23.3, -0.4, -0.03), (0.707, 0.0, 0.0, 0.707), 2.72),
        (1, (-46.9, -2.5,  5.07), (0.707, 0.0, 0.0, 0.707), 2.72),
        (2, (-74.6,  1.2, 13.67), (0.707, 0.0, 0.0, 0.707), 2.72),
        (3, (-111.5, -5.1, 24.57), (0.707, 0.0, 0.0, 0.707), 2.72),
        (4, (-135.5, -0.8, 25.36), (0.707, 0.0, 0.0, 0.707), 2.72),
        (5, (-159.2, -4.4, 25.97), (0.707, 0.0, 0.0, 0.707), 2.72),
    ]

    def __init__(self) -> None:
        self._connected = False
        self._connected_address: Optional[str] = None
        self._reset_count = 0
        self._attitude_send_count = 0
        self._track: Optional[TrackData] = None

    def _make_track(self) -> TrackData:
        gates = []
        for gate_id, pos, quat, size in self.VQ1_GATES:
            g = TrackGate(
                gate_id=gate_id,
                position_ned=pos,
                orientation=Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3]),
                width=size,
                height=size,
            )
            gates.append(g)
        return TrackData(gates=gates)

    async def connect(self, address: str = "udpin:127.0.0.1:14550") -> None:
        if self._connected:
            return
        self._connected_address = address
        self._connected = True
        self._track = self._make_track()

    async def disconnect(self) -> None:
        self._connected = False

    async def arm(self) -> None:
        pass

    async def start_offboard(self) -> None:
        pass

    async def stop_offboard(self) -> None:
        pass

    async def get_telemetry(self) -> TelemetryState:
        return TelemetryState(
            timestamp_us=int(time.monotonic() * 1_000_000),
            position_ned=(0.0, 0.0, 0.0),
            velocity_ned=(0.0, 0.0, 0.0),
            orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )

    async def get_camera_frame(self):
        return None

    async def send_attitude(self, cmd) -> None:
        self._attitude_send_count += 1

    async def send_attitude_rate(self, cmd) -> None:
        pass

    async def send_position(self, cmd) -> None:
        pass

    async def wait_for_track_data(self, timeout_s: float = 10.0) -> Optional[TrackData]:
        return self._track

    async def reset(self) -> Optional[TrackData]:
        self._reset_count += 1
        return self._track

    def drain_collisions(self):
        return []

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_armed(self) -> bool:
        return True

    @property
    def latest_telemetry(self) -> Optional[TelemetryState]:
        return None

    @property
    def race_status(self):
        return None

    @property
    def track_data(self) -> Optional[TrackData]:
        return self._track


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_vq1(
    address: str = "udpin:127.0.0.1:14550",
    max_speed: float = 8.0,
    record: Optional[str] = None,
    dry_run: bool = False,
    max_seconds: Optional[float] = None,
    minimal: bool = False,
    cruise_speed: float = 3.0,
    max_tilt_rad: float = 0.62,
    aim_z_offset: float = 0.0,
    cross_gain: float = 0.0,
    kv: float = 3.0,
    vert_gain: float = -1.0,
    max_vert_speed: float = -1.0,
    trajectory: bool = False,
) -> None:
    """Full Phase 1.5/1.6 run sequence.

    ``max_seconds`` caps the race wall-clock. The pipeline's own race
    timeout is 480 s, so a run that never completes (e.g. the drone is
    stuck off-track and replanning) otherwise costs the full 8 minutes
    before the capture is written. With ``max_seconds`` set, the run is
    cancelled cleanly after the cap and the telemetry log is STILL
    written (it is accumulated live in the recording callback), so a
    failing run produces an analyzable capture in seconds-to-minutes
    rather than 8 minutes. Used for fast live-sim iteration.
    """

    # 1. Build adapter
    if dry_run:
        adapter: CompetitionInterface = FakeAdapter()
        logger.info("DRY RUN — using FakeAdapter (no live sim connection)")
    else:
        from competition.aigp_mavlink import AIGPMavlinkAdapter
        adapter = AIGPMavlinkAdapter(enable_vision=True)

    # 2. Connect + wait for heartbeat + telemetry
    logger.info("Connecting to sim at %s …", address)
    await adapter.connect(address)
    logger.info("Connected")

    # 3. Fetch gate map — with a sanity check + retry. The SIM_RESET track
    #    transfer (chunked) intermittently delivers GARBAGE gate positions
    #    (observed: gate0 at (-918, 6.85, 577) instead of (-23.3,-0.4,-0.03)),
    #    which silently sends the controller chasing a point ~1 km away. Reject
    #    out-of-bounds maps and re-fetch rather than fly a corrupt course.
    track = await adapter.wait_for_track_data(timeout_s=10.0)
    if track is None:
        raise RuntimeError("No track data received — is the sim in Virtual Qualifier mode?")
    gates: List[GateSpec] = track_data_to_gatespecs(track)
    for attempt in range(4):
        if _gate_map_is_sane(gates):
            break
        logger.error(
            "Track map attempt %d looks CORRUPT (out-of-bounds gate positions): %s — "
            "re-fetching via SIM_RESET.",
            attempt + 1, [tuple(round(c, 1) for c in g.position) for g in gates],
        )
        track = await adapter.reset()
        if track is not None:
            gates = track_data_to_gatespecs(track)
    if not _gate_map_is_sane(gates):
        raise RuntimeError(
            "Track map still corrupt after retries — aborting rather than flying a "
            "garbage course."
        )
    # The passable OPENING sits ~0.85 m above gate.position in NED -z (the
    # drone hit the bottom bar when flying at gate.position). Bake the vertical
    # offset into the gate map ONCE so BOTH the controller aim AND the
    # sequencer's pass/opening check use the real opening centre — otherwise
    # the sequencer credits a pass only when the drone happens to cross within
    # its (wrong) assumed opening, and the target never advances past gate 1.
    if (minimal or trajectory) and aim_z_offset != 0.0:
        import dataclasses
        gates = [
            dataclasses.replace(
                g, position=(g.position[0], g.position[1], g.position[2] + aim_z_offset)
            )
            for g in gates
        ]
        logger.info(
            "Applied vertical opening offset %.2f m (NED z) to the gate map "
            "(aim + sequencer).", aim_z_offset,
        )
        aim_z_offset = 0.0  # now baked into gate.position; don't double-apply
    logger.info("Track map: %d gates", len(gates))
    for i, g in enumerate(gates):
        logger.info(
            "  Gate %d: pos=(%.1f, %.1f, %.1f) yaw=%.2f pitch=%.2f",
            i, *g.position, g.yaw, g.pitch,
        )

    # 4. Get initial position for pipeline configure
    telem = await adapter.get_telemetry()
    start_pos = telem.position_ned

    # 5. Configure pipeline (pre-computes trajectory — may take 1-2 s)
    config = PipelineConfig(
        max_speed=max_speed,
        minimal_control=minimal,
        minimal_cruise_speed=cruise_speed,
        minimal_max_tilt_rad=max_tilt_rad,
        minimal_aim_z_offset=aim_z_offset,
        minimal_cross_gain=cross_gain,
        minimal_kv=kv,
        minimal_vert_gain=vert_gain,
        minimal_max_vert_speed=max_vert_speed,
        trajectory_race=trajectory,
        # Both the minimal and the trajectory-race paths use the sim's raw
        # LOCAL_POSITION_NED directly. The EKF was diverging to NaN ~1 s into
        # flight, which silently blinded the controller (it fell back to a fixed
        # hover and climbed away). Raw telemetry is clean and is all VQ1 needs
        # (known gate map + accurate sim position). Only the legacy full
        # trajectory+replan path keeps the EKF on.
        use_ekf=(not minimal and not trajectory),
    )
    if minimal:
        logger.info(
            "MINIMAL CONTROL mode: pure-pursuit gate-to-gate @ cruise=%.1f m/s, "
            "max_tilt=%.2f rad — trajectory optimizer BYPASSED.",
            cruise_speed, max_tilt_rad,
        )
    elif trajectory:
        logger.info(
            "TRAJECTORY-RACE mode: precomputed racing-line + GeometricTracker "
            "(velocity-feedforward) on RAW telemetry; replan/predictor/slow-down "
            "BYPASSED. Optimizer constrained to the REAL envelope "
            "(max_accel=%.1f m/s², max_tilt=%.2f rad, thrust=%.0f N), "
            "plan_max_speed=%.1f m/s. A/B baseline: --minimal --cruise-speed 7.0.",
            config.traj_max_accel_mps2, config.traj_max_tilt_rad,
            config.traj_max_thrust_n, max_speed,
        )
    pipeline = RacePipeline(adapter, config)
    logger.info("Configuring pipeline…")
    pipeline.configure(gates, start_position=start_pos)
    logger.info("Trajectory pre-computed")

    # 6. SIM_RESET again so the race clock starts *after* trajectory computation,
    #    and WAIT until the drone has actually settled back at spawn. A previous
    #    run (or a --max-seconds cancel) can leave the drone airborne and still
    #    commanding; starting before it resets measures stale flight, not a fresh
    #    spawn (muddied telemetry). Poll until pos≈0 and vel≈0.
    logger.info("Resetting sim for clean race start…")
    await _reset_and_settle(adapter)

    # 7. Wrap pipeline callback to record actual path vs planned trajectory.
    telem_log: list = []
    _orig_callback = pipeline._control_callback

    def _recording_callback(
        telem: TelemetryState,
        frame: Optional[CameraFrame],
    ) -> Optional[AttitudeCommand]:
        cmd = _orig_callback(telem, frame)
        pos = list(telem.position_ned)
        vel = list(telem.velocity_ned)
        # Find the reference point the tracker is currently tracking.
        ref_pos = ref_vel = ref_yaw = None
        if pipeline.trajectory is not None and pipeline._ref_progress_time is not None:
            try:
                # ``_ref_progress_time`` is updated to the tracked reference's
                # time after each control callback, so sampling at that time
                # reproduces the point the tracker is actually following.
                # (The old ``at_time`` call did not exist on Trajectory and
                # silently swallowed an AttributeError, so ``ref_pos`` was
                # ``None`` in every recorded sample — the planned-vs-actual
                # comparison never ran.)
                pt = pipeline.trajectory.sample(pipeline._ref_progress_time)
                ref_pos = list(pt.position)
                ref_vel = list(pt.velocity)
                ref_yaw = float(pt.yaw)
            except Exception:
                pass
        # Full measured attitude + body rates — without these, erratic flight
        # (roll/pitch oscillation, flips, tumble) is INVISIBLE in the capture.
        # The original recorder logged only yaw, which is exactly why the
        # "bouncing left/right" had to be reported by eye instead of read off
        # the telemetry. roll/pitch from the orientation quaternion; gyro is
        # the sim's measured body angular rate (rad/s).
        try:
            m_roll, m_pitch, m_yaw = telem.orientation.to_euler()
        except Exception:
            m_roll = m_pitch = m_yaw = None
        gyro = list(telem.angular_velocity) if telem.angular_velocity else None
        # Distance to the gate the sequencer is currently targeting (the "next
        # gate"). This is what we actually care about — how close the drone got
        # to the gate it was trying to fly through, not just gate 0.
        tgt_idx = None
        dist_target = None
        if pipeline.sequencer is not None:
            cg = pipeline.sequencer.current_gate
            if cg is not None:
                tgt_idx = cg.sequence_index
                gx, gy, gz = cg.position
                dist_target = math.sqrt(
                    (pos[0] - gx) ** 2 + (pos[1] - gy) ** 2 + (pos[2] - gz) ** 2
                )
        entry = {
            "t_wall": time.monotonic(),
            "t_us": telem.timestamp_us,
            "pos": pos,
            "vel": vel,
            "yaw": _q_to_yaw(telem.orientation),
            "roll": m_roll,
            "pitch": m_pitch,
            "gyro": gyro,
            "gates_passed": pipeline.sequencer.gates_passed if pipeline.sequencer else 0,
            "target_gate": tgt_idx,
            "dist_target_gate": dist_target,
            "ref_pos": ref_pos,
            "ref_vel": ref_vel,
            "ref_yaw": ref_yaw,
        }
        if cmd is not None:
            entry["cmd_roll"] = round(cmd.roll_rad, 4)
            entry["cmd_pitch"] = round(cmd.pitch_rad, 4)
            entry["cmd_yaw"] = round(cmd.yaw_rad, 4)
            entry["cmd_thrust"] = round(cmd.thrust, 4)
        # Minimal-controller diagnostics: the exact gate position + internal
        # accel/thrust vectors the controller used this tick (ground truth for
        # explaining a capture).
        dbg = getattr(getattr(pipeline, "minimal_controller", None), "last_debug", None)
        if dbg is not None:
            entry["dbg"] = dbg
        # SIM's authoritative race status — does the sim itself credit our gate
        # passes? active_gate_index = the gate the SIM wants next (it increments
        # when the sim credits a pass); race_finished = the sim flagged the race
        # done. Our geometric sequencer is separate; this is the ground truth.
        try:
            rs = pipeline.interface.race_status
            if rs is not None:
                entry["sim_active_gate"] = rs.active_gate_index
                entry["sim_finished"] = bool(rs.race_finished)
                entry["sim_started"] = bool(rs.race_started)
        except Exception:
            pass
        telem_log.append(entry)
        return cmd

    pipeline._control_callback = _recording_callback

    # 8. Run
    wall_start = time.monotonic()
    logger.info("Starting race run…")
    timed_out = False
    try:
        if max_seconds is not None:
            await asyncio.wait_for(pipeline.run(address=address), timeout=max_seconds)
        else:
            await pipeline.run(address=address)
    except asyncio.TimeoutError:
        timed_out = True
        logger.warning(
            "Run hit --max-seconds=%.0fs cap before completion — cancelling "
            "cleanly and writing the partial capture for analysis.",
            max_seconds,
        )
    except Exception:
        # A crash in the control loop (e.g. a non-finite command rejected by
        # the adapter) must NOT swallow the telemetry collected so far — that
        # capture is exactly what we need to diagnose the crash. Log it and
        # fall through to the dump below.
        logger.exception(
            "Run crashed before completion — writing the partial capture "
            "collected so far for analysis."
        )
    elapsed = time.monotonic() - wall_start

    # 9. Dump telemetry log
    if record and telem_log:
        _write_telem_log(telem_log, record)
        logger.info("Telemetry log: %d samples → %s", len(telem_log), record)
    elif telem_log:
        # Always write to a timestamped file beside the script if --record not given
        default_path = f"captures/telemetry_{int(time.time())}.jsonl.gz"
        import os
        os.makedirs("captures", exist_ok=True)
        _write_telem_log(telem_log, default_path)
        logger.info("Telemetry log: %d samples → %s", len(telem_log), default_path)

    # 10. Path comparison summary
    if telem_log:
        _log_path_comparison(telem_log, gates)

    # 11. Completion summary
    collisions = adapter.drain_collisions()
    gates_passed = pipeline.sequencer.gates_passed if pipeline.sequencer else 0
    total_gates = pipeline.sequencer.total_gates if pipeline.sequencer else len(gates)
    logger.info(
        "Race finished: %d/%d gates in %.2f s, %d collision(s)",
        gates_passed, total_gates, elapsed, len(collisions),
    )
    # SIM-authoritative result (the sim credits the passes, not our sequencer).
    # active_gate_index = the next gate the SIM wants; == total_gates and
    # race_finished both mean the SIM scored a full clean run.
    try:
        rs = adapter.race_status
        if rs is not None:
            logger.info(
                "SIM race_status: active_gate_index=%d/%d, race_finished=%s "
                "(the SIM's official scoring%s)",
                rs.active_gate_index, total_gates, rs.race_finished,
                " — FULL COURSE COMPLETE ✓" if rs.race_finished else "",
            )
    except Exception:
        pass
    frozen = getattr(pipeline, "_telem_frozen_ticks", 0)
    if frozen:
        logger.error(
            "Telemetry feed was FROZEN for %d control ticks during the run — "
            "the controller flew (partly) blind on a stale state estimate. "
            "Investigate the MAVLink RX subscription before trusting this run.",
            frozen,
        )
    if collisions:
        for c in collisions:
            logger.info("  Collision: id=%s impulse=%.3f", c.get("id"), c.get("impulse", 0))

    # Stop the drone before we leave so it is not left flying into the next
    # iteration (which would muddy that run's telemetry). SIM_RESET respawns
    # it at the origin at rest.
    try:
        await adapter.reset()
    except Exception:
        logger.warning("Final SIM_RESET (stop drone) failed", exc_info=True)
    await adapter.disconnect()


def _gate_map_is_sane(gates: List[GateSpec]) -> bool:
    """Reject obviously-corrupt gate maps from a bad SIM_RESET track transfer.

    The real VQ1 course spans x in [-160, 0], |y| < 10, z in [-1, 27] (NED).
    Garbage transfers have produced gates ~1 km out. Bounds are generous so
    only true corruption is rejected, not legitimate course variation.
    """
    if not gates:
        return False
    for g in gates:
        x, y, z = g.position
        if not all(math.isfinite(c) for c in (x, y, z)):
            return False
        if not (-300.0 <= x <= 20.0 and -50.0 <= y <= 50.0 and -50.0 <= z <= 60.0):
            return False
    return True


async def _reset_and_settle(
    adapter,
    pos_tol: float = 1.0,
    vel_tol: float = 0.5,
    countdown_s: float = 3.5,
    timeout_s: float = 9.0,
) -> None:
    """SIM_RESET, wait out the race COUNTDOWN, and block until the drone is
    actually back at spawn before flight commands begin.

    The sim runs a ~``countdown_s`` countdown after a restart; the drone does
    NOT accept flight commands during it (commanding through the countdown
    wastes the first seconds and can leave the sim in a bad state). So we wait
    at least ``countdown_s`` from the SIM_RESET AND for the drone to be settled
    at spawn (|pos| & |vel| within tolerance — also guards against a prior run's
    drone still being airborne) before returning. ``race_status.race_started``
    is logged for observability (the sim flips it when the countdown ends).
    """
    await adapter.reset()
    t0 = time.monotonic()
    settled = False
    while time.monotonic() - t0 < timeout_s:
        await asyncio.sleep(0.2)
        elapsed = time.monotonic() - t0
        telem = adapter.latest_telemetry
        if telem is not None:
            p = telem.position_ned
            v = telem.velocity_ned
            if all(math.isfinite(x) for x in (*p, *v)):
                pos_mag = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                vel_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                settled = pos_mag < pos_tol and vel_mag < vel_tol
        # Require BOTH the countdown to have elapsed AND the drone settled.
        if settled and elapsed >= countdown_s:
            rs = getattr(adapter, "race_status", None)
            started = getattr(rs, "race_started", None) if rs is not None else None
            logger.info(
                "Countdown elapsed (%.1fs) and drone settled at spawn — starting "
                "flight (sim race_started=%s).", elapsed, started,
            )
            return
    logger.warning(
        "Reset wait timed out at %.1fs (settled=%s) — proceeding anyway.",
        timeout_s, settled,
    )


def _q_to_yaw(q) -> float:
    """Extract NED yaw (radians) from a Quaternion."""
    import math
    if q is None:
        return 0.0
    # ZYX Euler: yaw = atan2(2(wz+xy), 1-2(y²+z²))
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _write_telem_log(telem_log: list, path: str) -> None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        for row in telem_log:
            f.write(json.dumps(row) + "\n")


def _log_path_comparison(telem_log: list, gates: list) -> None:
    """Log a compact per-gate cross-track error summary."""
    import math
    if not telem_log or not gates:
        return
    logger.info("--- Path vs plan ---")
    for i, g in enumerate(gates):
        gx, gy, gz = g.position
        errors = []
        for row in telem_log:
            if row.get("gates_passed", 0) == i:
                px, py, pz = row["pos"]
                errors.append(math.sqrt((px-gx)**2 + (py-gy)**2 + (pz-gz)**2))
        if errors:
            logger.info(
                "  Gate %d: n=%d samples approaching, min_dist=%.2f m avg_dist=%.2f m",
                i, len(errors), min(errors), sum(errors)/len(errors),
            )
    # Overall cross-track error vs reference
    cross_errs = []
    for row in telem_log:
        if row.get("ref_pos") is not None:
            dx = row["pos"][0] - row["ref_pos"][0]
            dy = row["pos"][1] - row["ref_pos"][1]
            dz = row["pos"][2] - row["ref_pos"][2]
            cross_errs.append(math.sqrt(dx*dx + dy*dy + dz*dz))
    if cross_errs:
        logger.info(
            "  Overall: %d samples, avg_cross_track=%.3f m, p95=%.3f m",
            len(cross_errs),
            sum(cross_errs)/len(cross_errs),
            sorted(cross_errs)[int(0.95*len(cross_errs))],
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="AIGP VQ1 autonomous race runner")
    parser.add_argument(
        "--address",
        default="udpin:127.0.0.1:14550",
        help="pymavlink connection URL",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=8.0,
        dest="max_speed",
        help="Maximum trajectory speed in m/s",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Path to write JSONL capture (gzipped if .gz suffix)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Use FakeAdapter — full flow test without a live sim",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        dest="max_seconds",
        help="Cap race wall-clock (s); cancels cleanly and still writes the "
             "capture. Avoids the 480 s pipeline timeout on a stuck run.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="P0 strip-down: minimal pure-pursuit gate-to-gate control, "
             "bypassing the min-snap trajectory optimizer entirely.",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Iter-035 trajectory-race mode: fly the precomputed racing-line "
             "trajectory (optimizer constrained to the REAL measured envelope) "
             "with the GeometricTracker; replan/predictor/slow-down bypassed. "
             "The A/B alternative to --minimal. Use --max-speed for plan speed, "
             "--aim-z for the vertical opening offset.",
    )
    parser.add_argument(
        "--cruise-speed",
        type=float,
        default=3.0,
        dest="cruise_speed",
        help="Minimal-control cruise speed (m/s). Only used with --minimal.",
    )
    parser.add_argument(
        "--max-tilt",
        type=float,
        default=0.62,
        dest="max_tilt_rad",
        help="Minimal-control roll/pitch clamp (rad). Only used with --minimal.",
    )
    parser.add_argument(
        "--aim-z",
        type=float,
        default=0.0,
        dest="aim_z_offset",
        help="Vertical (NED z) offset added to the gate aim point (m). "
             "Negative aims higher. Only used with --minimal.",
    )
    parser.add_argument(
        "--kv",
        type=float,
        default=3.0,
        dest="kv",
        help="Velocity-tracking gain (1/s) for the minimal controller: "
             "accel=kv*(v_des-v). Raising it tightens cross-track tracking at "
             "the Y-staggered gates (the undershoot is kv-limited, not "
             "accel-clamp-limited). Only used with --minimal.",
    )
    parser.add_argument(
        "--vert-gain",
        type=float,
        default=-1.0,
        dest="vert_gain",
        help="Vertical-channel gain (1/s): vz=clip(vert_gain*dz,...). -1=default "
             "(1.0). Raise to close the climb gap at speed. Only with --minimal.",
    )
    parser.add_argument(
        "--max-vert-speed",
        type=float,
        default=-1.0,
        dest="max_vert_speed",
        help="Cap on commanded vertical speed (m/s). -1=default (3.0). Raise so "
             "steep climbs keep pace with cruise. Only with --minimal.",
    )
    parser.add_argument(
        "--cross-gain",
        type=float,
        default=0.0,
        dest="cross_gain",
        help="Cross-track (Y) convergence gain (1/s). 0=pure pursuit. >0 "
             "decouples horizontal (X cruise + capped high-gain Y) to fix the "
             "cross-track undershoot at speed. Only used with --minimal.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    asyncio.run(run_vq1(
        address=args.address,
        max_speed=args.max_speed,
        record=args.record,
        dry_run=args.dry_run,
        max_seconds=args.max_seconds,
        minimal=args.minimal,
        cruise_speed=args.cruise_speed,
        max_tilt_rad=args.max_tilt_rad,
        aim_z_offset=args.aim_z_offset,
        cross_gain=args.cross_gain,
        kv=args.kv,
        vert_gain=args.vert_gain,
        max_vert_speed=args.max_vert_speed,
        trajectory=args.trajectory,
    ))


if __name__ == "__main__":
    main()
