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
from types import SimpleNamespace
from typing import List, Optional

import gzip
import json

from competition.adapter import AttitudeCommand, CameraFrame, CompetitionInterface, Quaternion, TelemetryState
from competition.aigp_messages import TrackData, TrackGate
from competition.track_data import track_data_to_gatespecs
from competition.gate_map_integrity import (
    check_gate_map,
    read_reference_json,
    write_reference_json,
)
from competition.sim_health import SimHealthProbe
from competition.session import _positive_finite_float
from gate_sequencing.sequencer import GateSpec
from race_pipeline import PipelineConfig, RacePipeline

logger = logging.getLogger(__name__)

# Default location of the SESSION gate-map reference (see --gate-map-ref). The
# first sane fetched map in a session is written here; subsequent runs compare
# against it so a UNIFORM offset / drift (the failure the bounds miss — e.g. the
# sim degrading after ~25 runs) is caught ACROSS separate run processes.
DEFAULT_GATE_MAP_REF = "captures/gate_map_reference.json"


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
        self._epoch_started = time.monotonic()

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
        self._epoch_started = time.monotonic()

    async def disconnect(self) -> None:
        self._connected = False

    async def arm(self) -> None:
        pass

    async def start_offboard(self) -> None:
        pass

    async def stop_offboard(self) -> None:
        pass

    def _make_telemetry(self) -> TelemetryState:
        return TelemetryState(
            timestamp_us=int(time.monotonic() * 1_000_000),
            position_ned=(0.0, 0.0, 0.0),
            velocity_ned=(0.0, 0.0, 0.0),
            orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )

    async def get_telemetry(self) -> TelemetryState:
        return self._make_telemetry()

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
        self._epoch_started = time.monotonic()
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
        return self._make_telemetry()

    @property
    def race_status(self):
        # Exercise the same reset/countdown/GO contract as the real runner
        # without sleeping for the simulator's multi-second countdown. The
        # first poll sees pre-GO time, proving freshness; later polls cross GO.
        sim_boot_ms = int((time.monotonic() - self._epoch_started) * 1000.0)
        return SimpleNamespace(
            race_start_boot_time_ms=100,
            sim_boot_time_ms=sim_boot_ms,
            race_started=sim_boot_ms >= 100,
            race_finished=False,
            active_gate_index=0,
        )

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
    vert_ff: float = 0.0,
    lookahead_m: float = 0.0,
    lat_lead_m: float = 0.0,
    speed_brake: float = 0.0,
    speed_min_frac: float = 0.5,
    speed_descent_gain: float = 0.0,
    arrival_radius: float = 8.0,
    aim_slew: float = 0.0,
    final_aim_z: Optional[float] = None,
    final_brake_band: float = 0.0,
    spline_path: bool = False,
    spline_lookahead: float = 8.0,
    spline_a_lat: float = 6.5,
    spline_a_long: float = 12.0,
    spline_v_min: float = 6.0,
    spline_v_descent: float = 2.0,
    spline_vert_ff: float = 1.0,
    spline_v_final: float = 10.0,
    spline_final_region: float = 50.0,
    trajectory: bool = False,
    indi: bool = False,
    gate_map_ref: Optional[str] = DEFAULT_GATE_MAP_REF,
    refresh_gate_map_ref: bool = False,
    abort_on_degraded: bool = False,
) -> None:
    """Full Phase 1.5/1.6 run sequence.

    ``max_seconds`` caps the race wall-clock. The pipeline's own race
    timeout is 480 s, so a run that never completes (e.g. the drone is
    stuck off-track and replanning) otherwise costs the full 8 minutes
    before the capture is written. With ``max_seconds`` set, that shorter
    duration is injected into the session so it returns through normal
    cleanup; a slightly larger outer guard catches a stuck transport cleanup.
    The telemetry log is still written, so a failing run produces an
    analyzable capture in seconds-to-minutes rather than 8 minutes. Used for
    fast live-sim iteration.
    """

    # Validate the programmatic boundary before constructing or connecting an
    # adapter. ``bool`` is an ``int`` subclass and strings can be coerced by
    # ``float()``, but neither is an intentional race-duration request.
    if max_seconds is not None:
        max_seconds = _positive_finite_float("max_seconds", max_seconds)

    # 1. Build adapter
    if dry_run:
        adapter: CompetitionInterface = FakeAdapter()
        logger.info("DRY RUN — using FakeAdapter (no live sim connection)")
    else:
        from competition.aigp_mavlink import AIGPMavlinkAdapter
        adapter = AIGPMavlinkAdapter(enable_vision=True)
        # OPT-IN measured-accel INDI inner loop (roadmap #2). Default OFF: the
        # validated champion PD body-rate path is byte-identical unless --indi
        # is passed. When on, send_attitude computes the body-rate setpoint via
        # control.indi_inner_loop.IndiInnerLoop with online-G. CANNOT be
        # validated without the DCGame sim — this is the crux DISCRIMINATOR
        # experiment, not a claimed lap-time win. Read-out: achieved roll
        # restored => model mismatch (recoverable); still clamped => true
        # rate/bandwidth limit. See control/indi_inner_loop.py.
        if indi:
            adapter._use_indi = True
            logger.warning(
                "INDI inner loop ENABLED (--indi): EXPERIMENTAL measured-accel "
                "rate-INDI with online-G replaces the PD body-rate law. This is "
                "the crux discriminator, NOT a validated speed lever — watch the "
                "achieved-vs-commanded roll to read mismatch vs bandwidth limit."
            )

    # 2. Connect + wait for heartbeat + telemetry
    logger.info("Connecting to sim at %s …", address)
    await adapter.connect(address)
    logger.info("Connected")

    # 3. Fetch gate map — with an integrity check + retry. The SIM_RESET track
    #    transfer (chunked) intermittently delivers GARBAGE gate positions
    #    (observed: gate0 at (-918, 6.85, 577) instead of (-23.3,-0.4,-0.03);
    #    also sign-flipped X and Z≈-350 once the sim process degrades after ~25
    #    runs), which silently sends the controller chasing a bad course. The
    #    monitor (competition.gate_map_integrity) diagnoses WHICH corruption it
    #    is, not just that the map "looks corrupt", and — when a SESSION
    #    reference exists — also catches a UNIFORM offset / drift that stays in
    #    bounds. Reject and re-fetch rather than fly a corrupt course.
    #
    #    SESSION REFERENCE: a known-good map from the first healthy run of the
    #    session, persisted to ``gate_map_ref`` so a uniform drift is caught
    #    across SEPARATE run processes (the sim-degradation signature). Absent
    #    file => no reference (default behaviour unchanged for a fresh checkout).
    #    Skipped entirely on --dry-run: a dry run is offline testing on a fixed
    #    fake map, not a real session, so it neither reads nor writes the
    #    persistent baseline (keeps the offline flow side-effect-free).
    session_ref_path = None if dry_run else gate_map_ref
    reference = _load_gate_map_reference(session_ref_path, refresh_gate_map_ref)

    track = await adapter.wait_for_track_data(timeout_s=10.0)
    if track is None:
        raise RuntimeError("No track data received — is the sim in Virtual Qualifier mode?")
    gates: List[GateSpec] = track_data_to_gatespecs(track)
    verdict = check_gate_map(gates, reference=reference)
    for attempt in range(4):
        if verdict.ok:
            break
        logger.error(
            "Track map attempt %d FAILED integrity check [%s]: %s — positions=%s — "
            "re-fetching via SIM_RESET.",
            attempt + 1, verdict.diagnosis, verdict.message,
            [tuple(round(c, 1) for c in g.position) for g in gates],
        )
        if verdict.suggested_correction:
            logger.error(
                "  Suggested correction for [%s]: %s (NOT auto-applied — the "
                "runner re-fetches a clean map instead of flying a 'fixed' one).",
                verdict.diagnosis, verdict.suggested_correction,
            )
        track = await adapter.reset()
        if track is not None:
            gates = track_data_to_gatespecs(track)
        verdict = check_gate_map(gates, reference=reference)
    if not verdict.ok:
        raise RuntimeError(
            f"Track map still corrupt after retries [{verdict.diagnosis}]: "
            f"{verdict.message} — aborting rather than flying a garbage course."
        )
    logger.info("Gate map integrity: OK (%s)", verdict.message)

    # Persist this map as the SESSION reference the first time we see a sane map
    # and no reference exists yet (or --refresh-gate-map-ref was passed). Done
    # BEFORE the aim-z offset is baked in below, so the reference is the raw
    # sim-transferred geometry, comparable to future raw fetches.
    if session_ref_path and (reference is None or refresh_gate_map_ref):
        try:
            write_reference_json(gates, session_ref_path)
            logger.info(
                "Wrote SESSION gate-map reference (%d gates) -> %s — future runs "
                "will be checked for uniform drift against it.",
                len(gates), session_ref_path,
            )
        except Exception:
            logger.warning(
                "Could not write gate-map reference to %s (continuing — "
                "reference is optional).", gate_map_ref, exc_info=True,
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
        minimal_vert_ff=vert_ff,
        minimal_lookahead_m=lookahead_m,
        minimal_lat_lead_m=lat_lead_m,
        minimal_speed_brake=speed_brake,
        minimal_speed_min_frac=speed_min_frac,
        minimal_speed_descent_gain=speed_descent_gain,
        minimal_arrival_radius=arrival_radius,
        minimal_aim_slew=aim_slew,
        # iter-46: applied at RUNTIME relative to the baked opening centre (the
        # --aim-z bias is already baked into the gate map and zeroed here), so a
        # POSITIVE value aims the final gate LOWER to cancel the decel balloon.
        minimal_final_aim_z_offset=final_aim_z,
        minimal_final_brake_band_m=final_brake_band,
        minimal_spline_path=spline_path,
        minimal_spline_lookahead_m=spline_lookahead,
        minimal_spline_a_lat=spline_a_lat,
        minimal_spline_a_long=spline_a_long,
        minimal_spline_v_min=spline_v_min,
        minimal_spline_v_descent=spline_v_descent,
        minimal_spline_vert_ff=spline_vert_ff,
        minimal_spline_v_final=spline_v_final,
        minimal_spline_final_region_m=spline_final_region,
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
    # Collisions drained live inside the callback (so the health probe can see
    # them in real time) are accumulated here; the end-of-run summary reads this
    # list instead of re-draining, so no collision is lost to the probe.
    all_collisions: list = []
    _orig_callback = pipeline._control_callback

    # --- Sim-degradation health probe (race-day reliability item 1, half 2) ---
    # The companion to the gate-map integrity check above: that catches a CORRUPT
    # gate map; this watches the FLIGHT-DYNAMICS degradation signature — the
    # start OVER-CLIMB growing past the healthy ~-1.7 m toward the degraded
    # <= -2.4 m (NED, climb = negative Z), and/or a wildly-high early collision
    # count (see docs/aigp/2026-06-16-speed-and-spline-handoff.md "Operational
    # notes"). We feed the first ~window_s of post-GO flight to the streaming
    # probe and EVALUATE ONCE when the window elapses. This block is purely
    # ADDITIVE: it reads telemetry the recorder already has and NEVER alters the
    # control command or the GO logic. Default behaviour is WARN-ONLY; the
    # opt-in --abort-on-degraded ends the run early (clean disconnect) after the
    # warning. Skipped on --dry-run (FakeAdapter has no real climb/collisions —
    # the probe would just read insufficient_data; we avoid the noise entirely).
    health_probe: Optional[SimHealthProbe] = None if dry_run else SimHealthProbe()
    # Mutable cell so the nested callback can flip it; read after run() returns
    # to record the verdict into the capture and (optionally) report the abort.
    health_state = {"verdict": None, "abort": False, "t0": None}

    def _health_time_s(telem: TelemetryState) -> float:
        """Seconds clock for the probe: prefer the sim stamp (matches the clock
        the pipeline uses; immune to a non-realtime sim), else wall monotonic."""
        ts = telem.timestamp_us
        if ts is not None and ts > 0:
            return ts / 1e6
        return time.monotonic()

    def _recording_callback(
        telem: TelemetryState,
        frame: Optional[CameraFrame],
    ) -> Optional[AttitudeCommand]:
        cmd = _orig_callback(telem, frame)
        pos = list(telem.position_ned)
        vel = list(telem.velocity_ned)
        # --- Feed the sim-degradation health probe (additive; never touches cmd).
        # Sample z (NED, down-positive) + any collisions drained THIS tick, and
        # evaluate ONCE when the window has elapsed. Collisions are drained into
        # ``all_collisions`` so the end-of-run summary still sees every one (it
        # reads that list instead of re-draining). Wrapped so a probe hiccup can
        # never break the control loop / the run.
        if health_probe is not None and not health_probe.done:
            try:
                t_h = _health_time_s(telem)
                if health_state["t0"] is None:
                    health_state["t0"] = t_h
                health_probe.add_sample(t_h, pos[2])
                drained = adapter.drain_collisions()
                if drained:
                    all_collisions.extend(drained)
                    health_probe.add_collisions(len(drained))
                if health_probe.window_elapsed(t_h):
                    verdict = health_probe.evaluate()
                    health_state["verdict"] = verdict
                    if verdict.degraded:
                        # LOUD warning naming the signal + the action.
                        logger.warning("SIM HEALTH PROBE: %s", verdict.message)
                        if abort_on_degraded:
                            logger.warning(
                                "--abort-on-degraded set: ending this run early "
                                "(clean disconnect) — restart the DCGame .exe "
                                "into VQ mode before the next run."
                            )
                            health_state["abort"] = True
                            pipeline._diverged = True  # _should_stop ends the run
                    else:
                        # healthy or insufficient_data — informational only.
                        logger.info("SIM HEALTH PROBE: %s", verdict.message)
            except Exception:
                logger.debug("sim health probe tick failed (ignored)", exc_info=True)
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
        # INDI inner-loop read-out (only present when --indi is on). Logs
        # alpha_des/alpha_meas/Ghat/saturation/u per tick so the achieved-vs-
        # commanded roll (mismatch vs bandwidth-limit) can be read off the
        # capture. Mirrors actuator_outputs; None on the PD path.
        indi_dbg = getattr(adapter, "indi_debug", None)
        if indi_dbg is not None:
            entry["indi"] = indi_dbg
        # iter-45 (user): live "how close to the opening centre" metric (lateral
        # Y, vertical Z, and the in-plane distance to the current gate's centre).
        gco = getattr(pipeline, "_gate_center_offset", None)
        if gco is not None:
            entry["gate_center"] = {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in gco.items()}
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
    outer_guard_timed_out = False
    try:
        if max_seconds is not None:
            # Inject the short duration into RaceSession so it exits through
            # its normal timeout and cleanup path. Keep a slightly larger
            # outer guard only for a stuck cleanup/transport implementation.
            cleanup_guard_s = max(1.0, min(5.0, max_seconds * 0.1))
            await asyncio.wait_for(
                pipeline.run(
                    address=address,
                    max_run_duration_s=max_seconds,
                ),
                timeout=max_seconds + cleanup_guard_s,
            )
        else:
            await pipeline.run(address=address)
    except asyncio.TimeoutError:
        if max_seconds is None:
            raise
        outer_guard_timed_out = True
        logger.warning(
            "Run failed to finish cleanup within %.1fs after the "
            "--max-seconds=%.1fs race bound — cancelling and writing the "
            "partial capture for analysis.",
            cleanup_guard_s,
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
    if max_seconds is not None and not outer_guard_timed_out:
        logger.info(
            "Bounded run returned through normal session cleanup "
            "(--max-seconds=%.1fs).",
            max_seconds,
        )

    # 8b. Record the sim-health verdict as a ONE-SHOT entry in the capture so
    #     post-run analysis sees it. If the window never elapsed (a very short
    #     run), evaluate now to capture the (likely insufficient_data) verdict.
    if health_probe is not None:
        try:
            hv = health_state["verdict"]
            if hv is None:
                hv = health_probe.evaluate()
                health_state["verdict"] = hv
            telem_log.append({
                "sim_health": {
                    "healthy": hv.healthy,
                    "degraded": hv.degraded,
                    "diagnosis": hv.diagnosis,
                    "message": hv.message,
                    "details": hv.details,
                    "aborted_run": bool(health_state["abort"]),
                },
            })
            if hv.degraded:
                logger.warning(
                    "SIM HEALTH VERDICT (recorded into capture): [%s] %s%s",
                    hv.diagnosis, hv.message,
                    " — RUN ABORTED EARLY" if health_state["abort"] else "",
                )
            else:
                logger.info(
                    "SIM HEALTH VERDICT (recorded into capture): [%s]",
                    hv.diagnosis,
                )
        except Exception:
            logger.debug("recording sim-health verdict failed (ignored)", exc_info=True)

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

    # 11. Completion summary. The health probe drains collisions live into
    #     ``all_collisions``; combine those with any drained after the probe
    #     finished (or all of them on --dry-run, where the probe is off) so the
    #     reported count is complete regardless of when collisions arrived.
    collisions = all_collisions + adapter.drain_collisions()
    gates_passed = pipeline.sequencer.gates_passed if pipeline.sequencer else 0
    total_gates = pipeline.sequencer.total_gates if pipeline.sequencer else len(gates)
    # OFFICIAL RESULT: the SIM credits the passes, not our geometric sequencer
    # (which is center-based and undercounts the FINAL gate by ~0.16 m -- it
    # often logs 5/6 on a run the SIM actually scored 6/6, see the reliability
    # batch in docs/aigp/2026-06-16-realsim-loop-findings.md). Lead with the
    # SIM's authoritative scoring so a race-day operator reads the TRUE result,
    # not the misleading proxy.
    sim_finished = sim_gates = None
    try:
        rs = adapter.race_status
        if rs is not None:
            sim_finished = bool(rs.race_finished)
            sim_gates = int(rs.active_gate_index)
    except Exception:
        rs = None
    if sim_finished is not None:
        headline = (
            "FULL COURSE COMPLETE [OK] -- %d/%d gates" % (total_gates, total_gates)
            if sim_finished else
            "INCOMPLETE -- %d/%d gates credited (race_finished=False)"
            % (sim_gates, total_gates)
        )
        logger.info("OFFICIAL RESULT (SIM): %s in %.2f s, %d collision(s)",
                    headline, elapsed, len(collisions))
        logger.info("  (geometric sequencer reported %d/%d -- proxy only, it "
                    "undercounts the final gate ~0.16 m; trust the SIM result)",
                    gates_passed, total_gates)
    else:
        logger.info("Race finished: %d/%d gates in %.2f s, %d collision(s) "
                    "(sequencer; no SIM race_status available)",
                    gates_passed, total_gates, elapsed, len(collisions))
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


def _gate_map_is_sane(gates: List[GateSpec], reference=None) -> bool:
    """Reject obviously-corrupt gate maps from a bad SIM_RESET track transfer.

    Thin bool wrapper over :func:`competition.gate_map_integrity.check_gate_map`
    so the existing fetch-and-retry call sites keep their boolean contract while
    gaining sign-flip, self-consistency, and (when ``reference`` is supplied)
    uniform-offset / drift detection. The new module's OUTER bounds floor is a
    strict superset of this function's historical box (x in [-300, 20],
    |y| <= 50, z in [-50, 60]), so behaviour can only get stricter.

    The real VQ1 course spans x in ~[-160, 0], |y| < 10, z in ~[-1, 27] (NED);
    garbage transfers have produced gates ~1 km out, sign-flipped X, or Z≈-350.
    """
    return check_gate_map(gates, reference=reference).ok


def _load_gate_map_reference(path: Optional[str], refresh: bool):
    """Load the session gate-map reference, or None.

    Returns None (so the run behaves exactly as before) when: no path is given,
    ``--refresh-gate-map-ref`` was passed (we will OVERWRITE it with this run's
    map), the file is absent, or the file is malformed (a broken reference must
    never block an otherwise-healthy run). Logs why on each None path.
    """
    if not path:
        return None
    if refresh:
        logger.info(
            "--refresh-gate-map-ref: ignoring any existing reference; this "
            "run's sane map will overwrite %s.", path,
        )
        return None
    import os
    if not os.path.exists(path):
        logger.info(
            "No gate-map reference at %s yet — the first sane map this session "
            "will be saved there as the drift baseline.", path,
        )
        return None
    try:
        ref = read_reference_json(path)
        logger.info(
            "Loaded gate-map reference (%d gates) from %s — fetched maps will be "
            "checked for uniform drift against it.", len(ref), path,
        )
        return ref
    except Exception:
        logger.warning(
            "Gate-map reference at %s is unreadable/malformed — ignoring it "
            "(run continues without drift comparison).", path, exc_info=True,
        )
        return None


async def _reset_and_settle(
    adapter,
    pos_tol: float = 1.0,
    vel_tol: float = 0.5,
    go_margin_ms: float = 120.0,
    per_attempt_s: float = 8.0,
    max_resets: int = 4,
) -> None:
    """SIM_RESET, then block until the sim's 3 s countdown has actually elapsed
    (the race is GO) AND the drone is settled at spawn, before ANY flight command
    (arm / setpoint) is issued.

    DSQ ROOT CAUSE (iter-39, probed live): SIM_RESET resets the sim race clock
    (``sim_boot_time_ms``) to ~0 and schedules the GO time at
    ``race_start_boot_time_ms`` ≈ 3300 ms (the 3 s countdown). The race actually
    starts only when ``sim_boot_time_ms`` reaches that GO time (~3.3 s, and it
    JITTERS — observed 3.3–3.7 s). The ``race_started`` property
    (``race_start_boot_time_ms >= 0``) flips True at ~0.6 s and means only that a
    GO time is SCHEDULED, NOT that racing has begun. Gating on a 3.5 s timer (the
    old code) or on ``race_started`` (an earlier wrong fix) therefore commands
    the already-armed drone DURING the countdown → it jumps the start → the run
    is DISQUALIFIED. We instead read the authoritative GO crossing from
    telemetry: ``sim_boot_time_ms >= race_start_boot_time_ms`` (+ a small
    ``go_margin_ms``), confirmed against a FRESH post-reset status so a stale
    pre-reset frame (large ``sim_boot_time_ms``) cannot read GO spuriously.

    RETRY (iter-40): the sim IGNORES a SIM_RESET that arrives too soon after a
    previous one (the runner already resets once in connect() to fetch the gate
    map), so the race clock never drops and no fresh countdown appears. If an
    attempt doesn't reach a fresh GO crossing within ``per_attempt_s``, re-issue
    the reset (up to ``max_resets`` times) — never just proceed and false-start.
    """
    for attempt in range(1, max_resets + 1):
        # Race clock BEFORE this reset. SIM_RESET resets sim_boot_time_ms to ~0,
        # so a later value well below this proves the reset took effect (robust
        # even if reset() returns after the pre-GO window).
        rs0 = getattr(adapter, "race_status", None)
        pre_boot_ms = getattr(rs0, "sim_boot_time_ms", None) if rs0 is not None else None
        await adapter.reset()
        t0 = time.monotonic()
        settled = False
        fresh = False  # confirmed we see the FRESH post-reset race clock
        go = False
        now_ms = start_ms = -1
        while time.monotonic() - t0 < per_attempt_s:
            await asyncio.sleep(0.05)
            elapsed = time.monotonic() - t0
            telem = adapter.latest_telemetry
            if telem is not None:
                p = telem.position_ned
                v = telem.velocity_ned
                if all(math.isfinite(x) for x in (*p, *v)):
                    pos_mag = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                    vel_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                    settled = pos_mag < pos_tol and vel_mag < vel_tol
            rs = getattr(adapter, "race_status", None)
            if rs is not None:
                start_ms = rs.race_start_boot_time_ms
                now_ms = rs.sim_boot_time_ms
                # Fresh post-reset clock: the brief -1 blip, the clock still
                # before GO, or the clock having dropped well below its pre-reset
                # value (handles a slow reset() that returns past GO).
                if (start_ms < 0 or now_ms < start_ms
                        or (pre_boot_ms is not None and now_ms < pre_boot_ms - 500)):
                    fresh = True
                if fresh and start_ms >= 0 and now_ms >= start_ms + go_margin_ms:
                    go = True
            if go and settled:
                logger.info(
                    "Race GO crossing reached after %.2fs (attempt %d, "
                    "sim_boot=%dms >= start=%dms) and drone settled at spawn — "
                    "safe to arm + fly (countdown elapsed, no false start).",
                    elapsed, attempt, now_ms, start_ms,
                )
                return
            # A real reset drops the clock within ~1 s; if we haven't seen a
            # fresh countdown by 2.5 s the sim ignored this reset — retry now
            # instead of burning the full per-attempt budget.
            if elapsed > 2.5 and not fresh:
                break
        logger.warning(
            "Reset attempt %d/%d did NOT reach a fresh GO crossing in %.1fs "
            "(fresh=%s, go=%s, settled=%s, sim_boot=%sms, start=%sms) — the sim "
            "likely ignored the reset; re-issuing.",
            attempt, max_resets, per_attempt_s, fresh, go, settled, now_ms, start_ms,
        )
    logger.error(
        "Could not reach a clean GO crossing after %d SIM_RESETs — the sim may be "
        "wedged (needs a GUI restart into VQ mode). NOT flying (would false-start "
        "/ DSQ).", max_resets,
    )
    raise RuntimeError("no clean race start after repeated SIM_RESET")


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
            # Skip non-telemetry rows (e.g. the one-shot sim_health record).
            if "pos" not in row:
                continue
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
        if "pos" not in row:
            continue
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
        "--indi",
        action="store_true",
        help="OPT-IN (default OFF): use the measured-accel INDI inner loop with "
             "online-G (control/indi_inner_loop.py) for the body-rate setpoint "
             "instead of the validated PD law. EXPERIMENTAL crux DISCRIMINATOR — "
             "NOT a validated speed lever, cannot be confirmed without DCGame. "
             "Read-out: achieved roll restored => model mismatch (recoverable); "
             "still clamped => true rate/bandwidth limit. Leaves the PD path "
             "byte-identical when omitted.",
    )
    parser.add_argument(
        "--gate-map-ref",
        default=DEFAULT_GATE_MAP_REF,
        dest="gate_map_ref",
        help="Path to the SESSION gate-map reference JSON. On the first run "
             "whose fetched map is sane AND this file is absent, the map is "
             "saved here; on later runs it is loaded and the fetched map is "
             "checked for a UNIFORM offset / drift against it (catches the sim "
             "degrading across separate run processes). Absent file = no "
             "reference (default behaviour unchanged). Set to '' to disable.",
    )
    parser.add_argument(
        "--refresh-gate-map-ref",
        action="store_true",
        dest="refresh_gate_map_ref",
        help="Ignore any existing gate-map reference and OVERWRITE it with this "
             "run's sane map (use after a legitimate course/sim change so a new "
             "healthy baseline is captured). No-op if --gate-map-ref is ''.",
    )
    parser.add_argument(
        "--abort-on-degraded",
        action="store_true",
        dest="abort_on_degraded",
        help="OPT-IN (default OFF = WARN ONLY): if the sim-health probe finds "
             "the run DEGRADED in its first ~3 s (start over-climb Z <= -2.4 m "
             "vs healthy ~-1.7, or a wildly-high early collision count), end the "
             "run early with a clean disconnect after the warning. Default is to "
             "WARN ONLY and let the run continue (never surprise-abort). Either "
             "way the verdict is logged loudly and recorded into the capture; a "
             "degraded sim needs a full DCGame .exe restart into VQ mode (a "
             "SIM_RESET does NOT fix it). No-op on --dry-run.",
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
        "--vert-ff",
        type=float,
        default=0.0,
        dest="vert_ff",
        help="Vertical glide-slope FEEDFORWARD. 0=off (proportional law, lags "
             "descent and crosses gates above centre). 1.0=descend at "
             "speed*dz/horiz_dist so the drone arrives at gate altitude on time. "
             ">1.0 biases low (more top-bar margin). Only with --minimal.",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=0.0,
        dest="lookahead_m",
        help="VERTICAL anticipatory descent: aim the altitude DOWN toward the "
             "NEXT (lower) gate by up to this many METRES (bounded), ramped in "
             "over the last ~12 m of approach. Kills the vertical lag (drone "
             "arriving above the opening at speed) with no lateral corner-cut. "
             "0=off. Try 0.3-0.5 m. Only with --minimal.",
    )
    parser.add_argument(
        "--speed-brake",
        type=float,
        default=0.0,
        dest="speed_brake",
        help="VARIABLE SPEED gain (1/rad): leg cruise = --cruise-speed * "
             "clip(1 - speed_brake*turn_angle, speed-min-frac, 1). Brakes into "
             "the tight (steep-descent/Y-reversal) gates, full speed on straight "
             "legs. 0=constant speed. Try 0.6-1.2. Only with --minimal.",
    )
    parser.add_argument(
        "--speed-min-frac",
        type=float,
        default=0.5,
        dest="speed_min_frac",
        help="Floor on the braked cruise fraction (default 0.5). With --speed-brake.",
    )
    parser.add_argument(
        "--speed-descent-gain",
        type=float,
        default=0.0,
        dest="speed_descent_gain",
        help="Also brake steep-but-straight DESCENT gates: difficulty = "
             "max(turn_angle, gain*slope). Try ~1.5. With --speed-brake.",
    )
    parser.add_argument(
        "--aim-slew",
        type=float,
        default=0.0,
        dest="aim_slew",
        help="Slew-limit the aim's lateral (Y) target (m/s): spreads the "
             "cross-track course correction across the leg instead of an instant "
             "jerk at the gate pass (smooths the rock-back). 0=off. Try 6-12. --minimal.",
    )
    parser.add_argument(
        "--arrival-radius",
        type=float,
        default=8.0,
        dest="arrival_radius",
        help="Final-gate (gate5) approach-ramp radius (m). Larger brakes the "
             "finish-gate approach earlier to bleed cross-track velocity and "
             "stop the high-speed overshoot/tumble. Default 8. Try 12-16. --minimal.",
    )
    parser.add_argument(
        "--final-aim-z",
        type=float,
        default=None,
        dest="final_aim_z",
        help="Vertical aim offset for the FINAL gate only (m, NED, applied at "
             "runtime vs the baked opening centre). POSITIVE aims LOWER to cancel "
             "the decel pitch-up balloon that makes the drone clip gate5's TOP "
             "frame at speed. Try +0.5..+0.8. None=same as other gates. --minimal.",
    )
    parser.add_argument(
        "--spline", action="store_true", dest="spline_path",
        help="STRUCTURAL racing line: fly ONE continuous arc-length spline "
             "through all gates with curvature-limited speed (replaces the "
             "gate-by-gate aim + per-leg brake). Removes the sharp-corner "
             "undershoot/reversal-tumble. --minimal.",
    )
    parser.add_argument("--spline-lookahead", type=float, default=8.0,
                        dest="spline_lookahead",
                        help="Pursuit lookahead (m) along the spline. Try 4-10.")
    parser.add_argument("--spline-a-lat", type=float, default=6.5,
                        dest="spline_a_lat",
                        help="Lateral-accel cap (m/s^2) for the curvature speed "
                             "profile. 6.5 caps the slalom lateral tilt (the "
                             "shared -z_b[2] with descent tumbles it at speed); "
                             "10 never engages. Lower=slower turns.")
    parser.add_argument("--spline-a-long", type=float, default=12.0,
                        dest="spline_a_long",
                        help="Longitudinal-accel cap (m/s^2) for the speed profile.")
    parser.add_argument("--spline-v-min", type=float, default=6.0,
                        dest="spline_v_min",
                        help="Speed floor (m/s) in the tightest turn.")
    parser.add_argument("--spline-v-descent", type=float, default=2.0,
                        dest="spline_v_descent",
                        help="Vertical descent-RATE cap (m/s): steep legs slow "
                             "so v*|tangent_z|<=this (drone destabilises "
                             "descending fast at speed). 0=off. 2.0 is the stable "
                             "ceiling at cruise 16; 2.5+ eats gate3 margin.")
    parser.add_argument("--spline-vert-ff", type=float, default=1.0,
                        dest="spline_vert_ff",
                        help="Spline vertical: 1.0=glide-slope feedforward "
                             "(vz=speed*slope, aggressive); 0=steady capped-"
                             "proportional descent (gentler, avoids the roll "
                             "limit cycle at speed). Try 0.")
    parser.add_argument("--spline-v-final", type=float, default=10.0,
                        dest="spline_v_final",
                        help="Speed cap (m/s) over the closing reversal region "
                             "(ports --final-brake-band to the spline; the gentle "
                             "spline curvature there doesn't auto-brake the final "
                             "lateral move). 0=off. 10 validated.")
    parser.add_argument("--spline-final-region", type=float, default=50.0,
                        dest="spline_final_region",
                        help="Length (m) of the final-region brake zone. Default 50 "
                             "(covers the whole g3->g4->g5 reversal). With --spline-v-final.")
    parser.add_argument(
        "--final-brake-band",
        type=float,
        default=0.0,
        dest="final_brake_band",
        help="Proximity-brake the FINAL leg within this many metres of gate5 "
             "(m). Keeps the straight fast (peak) but bleeds speed for the "
             "slalom reversal the rate-limited roll can't make at 50 km/h. "
             "0=off. Try 12-18. Needs --speed-brake. --minimal.",
    )
    parser.add_argument(
        "--lat-lead",
        type=float,
        default=0.0,
        dest="lat_lead_m",
        help="Lateral lead (metres): aim PAST each gate's Y in the slalom travel "
             "direction so the undershooting drone arrives centred at high cruise. "
             "Bounded; ramped over the last ~12 m. 0=off. Try 0.3-0.6. --minimal.",
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
        vert_ff=args.vert_ff,
        lookahead_m=args.lookahead_m,
        lat_lead_m=args.lat_lead_m,
        speed_brake=args.speed_brake,
        speed_min_frac=args.speed_min_frac,
        speed_descent_gain=args.speed_descent_gain,
        arrival_radius=args.arrival_radius,
        aim_slew=args.aim_slew,
        final_aim_z=args.final_aim_z,
        final_brake_band=args.final_brake_band,
        spline_path=args.spline_path,
        spline_lookahead=args.spline_lookahead,
        spline_a_lat=args.spline_a_lat,
        spline_a_long=args.spline_a_long,
        spline_v_min=args.spline_v_min,
        spline_v_descent=args.spline_v_descent,
        spline_vert_ff=args.spline_vert_ff,
        spline_v_final=args.spline_v_final,
        spline_final_region=args.spline_final_region,
        trajectory=args.trajectory,
        indi=args.indi,
        # Empty string disables the reference entirely.
        gate_map_ref=(args.gate_map_ref or None),
        refresh_gate_map_ref=args.refresh_gate_map_ref,
        abort_on_degraded=args.abort_on_degraded,
    ))


if __name__ == "__main__":
    main()
