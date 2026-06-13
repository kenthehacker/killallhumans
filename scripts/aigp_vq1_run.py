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
) -> None:
    """Full Phase 1.5/1.6 run sequence."""

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

    # 3. Fetch gate map
    track = await adapter.wait_for_track_data(timeout_s=10.0)
    if track is None:
        raise RuntimeError("No track data received — is the sim in Virtual Qualifier mode?")
    gates: List[GateSpec] = track_data_to_gatespecs(track)
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
    config = PipelineConfig(max_speed=max_speed)
    pipeline = RacePipeline(adapter, config)
    logger.info("Configuring pipeline…")
    pipeline.configure(gates, start_position=start_pos)
    logger.info("Trajectory pre-computed")

    # 6. SIM_RESET again so the race clock starts *after* trajectory computation.
    logger.info("Resetting sim for clean race start…")
    await adapter.reset()

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
        entry = {
            "t_wall": time.monotonic(),
            "t_us": telem.timestamp_us,
            "pos": pos,
            "vel": vel,
            "yaw": _q_to_yaw(telem.orientation),
            "gates_passed": pipeline.sequencer.gates_passed if pipeline.sequencer else 0,
            "ref_pos": ref_pos,
            "ref_vel": ref_vel,
            "ref_yaw": ref_yaw,
        }
        if cmd is not None:
            entry["cmd_roll"] = round(cmd.roll_rad, 4)
            entry["cmd_pitch"] = round(cmd.pitch_rad, 4)
            entry["cmd_yaw"] = round(cmd.yaw_rad, 4)
            entry["cmd_thrust"] = round(cmd.thrust, 4)
        telem_log.append(entry)
        return cmd

    pipeline._control_callback = _recording_callback

    # 8. Run
    wall_start = time.monotonic()
    logger.info("Starting race run…")
    await pipeline.run(address=address)
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

    await adapter.disconnect()


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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    asyncio.run(run_vq1(
        address=args.address,
        max_speed=args.max_speed,
        record=args.record,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
