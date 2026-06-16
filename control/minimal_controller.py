"""Minimal gate-to-gate pure-pursuit controller (AIGP VQ1 P0 strip-down).

Rationale (see docs/aigp/2026-06-13-control-handoff.md, the "STOP PATCHING,
SIMPLIFY" P0 section):

The min-snap racing optimizer emits references that demand ~18 m/s² (61° tilt)
when the live drone's usable envelope is ~11 m/s² (49°). The drone physically
cannot track that reference → it saturates tilt + thrust → flips and diverges
(0/6 gates on every live run). The fix is NOT to tune the optimizer's fudge
factors; it is to fly a *feasible* reference.

This controller is that feasible reference, computed directly from the current
target gate — no trajectory optimization at all:

  desired velocity  = cruise_speed · unit(gate − pos)      (pure pursuit)
  desired accel     = kv · (v_des − v)                     (velocity tracking)
  horizontal accel  is HARD-CLAMPED to g·tan(max_tilt)     (never demand a tilt
                                                            the drone can't make)
  thrust + roll/pitch are extracted from the NED thrust vector, identical math
  to GeometricTracker (which was never the bug).

Yaw is held at π (all VQ1 gates face −X, so a fixed heading is correct and
simplest — handoff §5). The AttitudeCommand this returns is converted to the
sim's body-rate command by AIGPMavlinkAdapter.send_attitude (the fixed inner
loop we keep).

The whole point is that EVERY command this emits is inside the flight envelope,
so the drone can always physically follow it. Get a repeatable 6/6 at low speed
first; only then re-introduce speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from competition.adapter import AttitudeCommand


@dataclass
class MinimalControllerConfig:
    """Tunable knobs for the minimal pure-pursuit controller.

    Defaults are deliberately CONSERVATIVE — the first job is a stable,
    repeatable 6/6, not speed. Crank ``cruise_speed`` / ``max_tilt_rad`` up
    only after low-speed flight is proven stable on the live sim.
    """

    # Commanded ground/air speed along the line to the current gate (m/s).
    cruise_speed: float = 3.0
    # Velocity-error → acceleration gain (1/s). Higher = snappier velocity
    # convergence but more tilt transient at start-up (clamped below).
    kv: float = 3.0
    # When approaching the FINAL gate, ramp speed down within this radius so
    # the drone arrives instead of rocketing off the end of the course (m).
    arrival_radius: float = 8.0
    # Floor on the ramped final-approach speed so the drone always reaches and
    # passes the last gate (m/s).
    min_speed: float = 1.0
    # Vertical channel is DECOUPLED from horizontal pursuit: a pure-pursuit unit
    # vector gives a tiny vertical component when the gate is far horizontally,
    # so the descent LAGS on steep legs and the drone arrives above the opening
    # (min_v15: ~1 m high at gate 2, oscillates, never passes). Instead track a
    # desired vertical velocity proportional to the altitude error, capped, so
    # the drone closes the z-gap regardless of horizontal distance.
    vert_gain: float = 1.0          # desired vz per metre of z-error (1/s)
    # iter-31 speed-up: 2.0->3.0. iter-37: 3.0->5.0. Required descent rate scales
    # with cruise (same Δz per leg, less time). At cruise 8.0 the steep descending
    # legs made the drone arrive ~0.6 m ABOVE the opening (frame clearance only
    # 0.11 m at gate2 — nearly clipping the top bar); raising the CAP to 5.0
    # restored 0.42 m clearance and made cruise 8.0 reliably collision-free
    # (3/3 runs). NOTE: only the CAP is safe to raise — raising vert_GAIN above
    # 1.0 destabilises (iter-37: vert_gain 2.0 -> gyro 8.45 limit cycle, 128
    # collisions) because aggressive vertical accel swings the thrust vector and
    # the roll extraction (atan2(zy_h,-z_b[2])) blows up as -z_b[2] shrinks.
    max_vert_speed: float = 5.0     # cap on commanded vertical speed (m/s)
    # iter-34 cross-track centering (verified roadmap #4) — TESTED LIVE, FAILED.
    # DO NOT ENABLE (leave cross_gain=0). Decoupling the horizontal into
    # X=along-track-cruise + Y=capped-high-gain-convergence DESTABILISES: it
    # loses pure pursuit's natural cross-track deceleration, so Y overshoots and
    # the drone clips frames. cross_gain 1.5 -> 3 collisions, gyro MAX 2.22,
    # lateral-accel clamp 22%; cross_gain 0.5 -> 14 collisions, gyro spike 38.
    # The workflow's adversarial pass predicted exactly this (the term adds
    # cross-track BANDWIDTH without damping). Kept off-by-default as a record;
    # pure pursuit (cross_gain=0) is the clean, recommended law.
    cross_gain: float = 0.0         # KEEP 0 — >0 destabilises (see above)
    max_cross_speed: float = 2.0    # cap on commanded cross-track (Y) speed (m/s)

    # --- Envelope clamps (the load-bearing part) ---------------------------
    # Max horizontal accel we will ever command (m/s²). g·tan(35.5°) ≈ 7.0.
    # This is the single guarantee that the drone is never asked for a tilt it
    # cannot achieve — the exact failure mode that flips it today.
    max_lateral_accel: float = 7.0
    # Hard roll/pitch clamp (rad). ~35.5° — comfortably inside the measured
    # ~49° flip limit, with margin for the inner-loop's 2.5× rate overshoot.
    max_tilt_rad: float = 0.62
    # Clamp on commanded vertical accel magnitude (m/s²).
    max_climb_accel: float = 6.0

    # Heading hold. All VQ1 gates face −X; spawn yaw ≈ π. Hold it.
    yaw_hold: float = math.pi

    # --- Drone constants ---------------------------------------------------
    # max_thrust_n RE-calibrated 2026-06-13 (iter-022) from the bench
    # zero-body-rate thrust sweep: vz≈0 at throttle ~0.26-0.27 (level), i.e.
    # hover force 9.81 N at ~0.26 → max_thrust_n ≈ 37-38 N (NOT the 42 in
    # drone_spec, which gave a hover command of 0.234 — too low, so the drone
    # under-commanded vertical and the controller had to fight it). 37.0 makes
    # the controller's hover command land on the measured hover throttle.
    mass: float = 1.0
    gravity: float = 9.81
    max_thrust_n: float = 37.0
    min_thrust: float = 0.05
    max_thrust: float = 0.95


class MinimalController:
    """Stateless pure-pursuit gate-to-gate controller.

    One public call: ``compute(pos, vel, yaw, gate_pos, is_final_gate)`` →
    ``AttitudeCommand``. No trajectory, no replan, no internal state.
    """

    def __init__(self, config: MinimalControllerConfig | None = None) -> None:
        self.cfg = config or MinimalControllerConfig()

    def desired_velocity(
        self,
        pos: np.ndarray,
        gate_pos: np.ndarray,
        is_final_gate: bool,
    ) -> tuple[np.ndarray, float]:
        """Desired velocity toward the gate (HORIZONTAL pursuit + DECOUPLED
        vertical), plus full 3-D distance.

        Horizontal: pure-pursuit at cruise speed toward the gate's x/y.
        Vertical: track the altitude error directly (vz = vert_gain * dz,
        capped) so the descent/climb keeps pace independent of how far the gate
        is horizontally — fixes the "arrives above the opening on steep legs"
        lag.
        """
        cfg = self.cfg
        to_gate = gate_pos - pos
        dist = float(np.linalg.norm(to_gate))

        horiz = to_gate[:2]
        horiz_dist = float(np.linalg.norm(horiz))
        speed = cfg.cruise_speed
        if is_final_gate:
            # Ramp horizontal speed down on the final approach so we settle on
            # the last gate rather than overshooting off the end of the course.
            ramped = cfg.cruise_speed * (dist / cfg.arrival_radius)
            speed = max(cfg.min_speed, min(cfg.cruise_speed, ramped))
        if cfg.cross_gain > 0.0:
            # DECOUPLED horizontal (VQ1 gates face -X): along-track X at cruise,
            # cross-track Y as a capped high-gain convergence so the drone closes
            # the Y gap before the gate plane (fixes the cross-track undershoot).
            v_x = math.copysign(speed, to_gate[0]) if abs(to_gate[0]) > 1e-3 else 0.0
            v_y = float(np.clip(cfg.cross_gain * to_gate[1],
                                -cfg.max_cross_speed, cfg.max_cross_speed))
            v_xy = np.array([v_x, v_y])
        elif horiz_dist > 1e-3:
            v_xy = (horiz / horiz_dist) * speed
        else:
            v_xy = np.zeros(2)

        v_z = float(
            np.clip(cfg.vert_gain * to_gate[2], -cfg.max_vert_speed, cfg.max_vert_speed)
        )
        return np.array([v_xy[0], v_xy[1], v_z]), dist

    def _hover(self, yaw: float) -> AttitudeCommand:
        """Level, gravity-cancelling hover command — the safe fallback when
        inputs are unusable (non-finite telemetry)."""
        hover_thrust = float(
            np.clip(
                self.cfg.mass * self.cfg.gravity / self.cfg.max_thrust_n,
                self.cfg.min_thrust,
                self.cfg.max_thrust,
            )
        )
        yaw_safe = float(yaw) if math.isfinite(yaw) else self.cfg.yaw_hold
        return AttitudeCommand(0.0, 0.0, yaw_safe, hover_thrust)

    def compute(
        self,
        pos,
        vel,
        yaw: float,
        gate_pos,
        is_final_gate: bool = False,
    ) -> AttitudeCommand:
        cfg = self.cfg
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        gate_pos = np.asarray(gate_pos, dtype=float)

        # Robustness to bad telemetry. The live sim occasionally emits a
        # non-finite position/velocity (odom reset, dropped field); the old
        # GeometricTracker would silently turn that into a NaN thrust and the
        # adapter crashes ("thrust must be finite"), killing the whole run.
        # If position or the gate is unusable we cannot even compute a
        # direction → hover. A non-finite velocity we can safely treat as
        # zero (the velocity-tracking term just sees full error this tick).
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(gate_pos))):
            return self._hover(yaw)
        if not np.all(np.isfinite(vel)):
            vel = np.zeros(3)

        v_des, _dist = self.desired_velocity(pos, gate_pos, is_final_gate)

        # Velocity-tracking acceleration (NED). Pointing v_des straight at the
        # gate makes this a pursuit law: cross-track error decays as the drone
        # is continuously steered toward the gate. No position-integral term —
        # we want feasibility and stability, not millimetre tracking.
        accel = cfg.kv * (v_des - vel)

        # --- Envelope clamps ---------------------------------------------
        # Horizontal: never command more lateral accel than g·tan(max_tilt).
        ah = accel[:2]
        ah_mag = float(np.linalg.norm(ah))
        if ah_mag > cfg.max_lateral_accel:
            ah = ah * (cfg.max_lateral_accel / ah_mag)
        accel[0], accel[1] = ah[0], ah[1]
        # Vertical: clamp magnitude, then saturate descent so the thrust vector
        # never flips downward (which would bank the drone sideways — the
        # a_down_max guard mirrors GeometricTracker).
        accel[2] = float(np.clip(accel[2], -cfg.max_climb_accel, cfg.max_climb_accel))
        a_down_max = cfg.gravity - cfg.min_thrust * cfg.max_thrust_n / cfg.mass
        accel[2] = min(accel[2], a_down_max)

        # --- Thrust vector → thrust + attitude (NED, z-down) -------------
        # Thrust must counter gravity: T = m·(a − g_vec), g_vec = (0,0,+g).
        thrust_vec = cfg.mass * (accel - np.array([0.0, 0.0, cfg.gravity]))
        thrust_mag = float(np.linalg.norm(thrust_vec))
        thrust_norm = float(
            np.clip(thrust_mag / cfg.max_thrust_n, cfg.min_thrust, cfg.max_thrust)
        )
        if thrust_mag > 1e-3:
            z_b = thrust_vec / thrust_mag      # desired body-up axis (NED)
        else:
            z_b = np.array([0.0, 0.0, -1.0])   # hover: thrust points up (−z)

        # Yaw: hold the course heading, wrapped to shortest path from the
        # measured yaw so the rate loop never spins the long way round the
        # ±π seam (spawn yaw ≈ +π or −π are the same heading).
        yaw_err = math.atan2(
            math.sin(cfg.yaw_hold - yaw), math.cos(cfg.yaw_hold - yaw)
        )
        yaw_cmd = yaw + yaw_err

        # Roll/pitch from the thrust direction, expressed in the heading
        # frame. Identical extraction to GeometricTracker.track (NED).
        cpsi, spsi = math.cos(yaw_cmd), math.sin(yaw_cmd)
        zx_h = cpsi * z_b[0] + spsi * z_b[1]
        zy_h = -spsi * z_b[0] + cpsi * z_b[1]
        pitch = -math.asin(float(np.clip(zx_h, -1.0, 1.0)))
        roll = math.atan2(zy_h, -z_b[2])

        # LIVE-SIM ROLL CONVENTION (bench-measured 2026-06-13, iter-023): on
        # this sim a +roll command produces +Y motion at yaw=pi, whereas the
        # standard NED extraction above assumes +roll -> -Y. With the standard
        # sign the controller commands +roll to correct a +Y drift, which feeds
        # the drift (positive feedback) and the drone slides off in +Y at
        # constant speed (min_v7). The roll inner loop itself tracks correctly
        # (measured roll follows commanded), so the fix belongs here in the
        # extraction, not in the inner-loop _rate_sign: invert the roll sign so
        # desired lateral accel maps to the roll the sim actually needs. (Pitch
        # was a separate, inner-loop instability fixed via _rate_sign.)
        roll = -roll

        roll = float(np.clip(roll, -cfg.max_tilt_rad, cfg.max_tilt_rad))
        pitch = float(np.clip(pitch, -cfg.max_tilt_rad, cfg.max_tilt_rad))

        # Final guard: never hand the adapter a non-finite command (it raises
        # and kills the run). If anything went non-finite, hover instead.
        if not all(math.isfinite(x) for x in (roll, pitch, yaw_cmd, thrust_norm)):
            return self._hover(yaw)

        # Diagnostic stash (read by the recorder) — the exact inputs and
        # internal vectors this tick, so a capture can be replayed/explained
        # without guessing the gate position or frame.
        self.last_debug = {
            "yaw_in": float(yaw),
            "yaw_cmd": float(yaw_cmd),
            "gate": [float(x) for x in gate_pos],
            "vdes": [round(float(x), 3) for x in v_des],
            "accel": [round(float(x), 3) for x in accel],
            "tvec": [round(float(x), 3) for x in thrust_vec],
        }

        return AttitudeCommand(
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=float(yaw_cmd),
            thrust=thrust_norm,
        )
