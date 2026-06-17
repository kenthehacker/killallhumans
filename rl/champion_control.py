"""Standalone, deterministic re-implementation of the CHAMPION control law.

The champion is `race_pipeline.RacePipeline._control_callback`'s minimal /
gate-by-gate branch driving `control.minimal_controller.MinimalController`, run
with the champion CLI config (findings doc reproduce block):

    --minimal --cruise-speed 16 --max-tilt 0.82 --aim-z -0.85 --kv 3.0
    --max-vert-speed 9.5 --lookahead 0.7 --speed-brake 1.3 --speed-min-frac 0.55
    --speed-descent-gain 1.5 --aim-slew 12 --final-aim-z 0.5 --final-brake-band 26

We reuse `MinimalController` UNCHANGED (it is the real control law and emits an
AttitudeCommand). What we re-host here is the thin gate-by-gate AIM + variable-
speed-brake + aim-slew layer that the pipeline wraps around it, plus the
sequencer's plane-crossing gate advance — exactly the minimal branch logic, but:
  * driven by SIM-TIME dt instead of wall-clock `time.monotonic()` (so a replay
    is deterministic and step-rate-independent), and
  * decoupled from the EKF / detection / replan stack (the champion bypasses all
    of it on the minimal path — state in, attitude out).

`--aim-z -0.85` is BAKED into the gate map by the runner (it shifts every gate's
NED z by the offset and zeroes the controller's aim_z_offset), so we apply the
same bake here and pass aim_z_offset=0 to the per-tick aim, matching the live run.

This is read-only w.r.t. the live code: it imports MinimalController and mirrors
the pipeline; it does not modify either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from competition.adapter import AttitudeCommand
from control.minimal_controller import MinimalController, MinimalControllerConfig


# Champion CLI config -> the knobs the minimal branch reads.
@dataclass
class ChampionConfig:
    cruise_speed: float = 16.0
    max_tilt_rad: float = 0.82
    aim_z_offset: float = -0.85          # baked into the gate map (see module doc)
    kv: float = 3.0
    max_vert_speed: float = 9.5
    lookahead_m: float = 0.7
    lookahead_band_m: float = 12.0       # pipeline default
    speed_brake: float = 1.3
    speed_min_frac: float = 0.55
    speed_descent_gain: float = 1.5
    aim_slew: float = 12.0
    final_aim_z_offset: Optional[float] = 0.5
    final_brake_band_m: float = 26.0
    through_dist: float = 2.0            # pipeline default minimal_through_dist
    lat_lead_m: float = 0.0              # not set by champion config
    cruise_slew_rate: float = 12.0       # the pipeline's hard-coded 12 m/s^2 slew


def _gate_normal(yaw: float, pitch: float = 0.0) -> Tuple[float, float, float]:
    """Identical to race_pipeline._gate_normal."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (cy * cp, sy * cp, -sp)


@dataclass
class Gate:
    """Minimal GateSpec stand-in: NED position + opening orientation."""
    position: Tuple[float, float, float]
    yaw: float
    pitch: float
    sequence_index: int


class ChampionDriver:
    """Closed-loop champion controller + plane-crossing gate sequencer.

    Usage:
        drv = ChampionDriver(gates, ChampionConfig())
        att = drv.step(pos, vel, yaw, dt)   # AttitudeCommand, advances sequencer
        ... feed att+thrust into the replica ...
        drv.finished  / drv.gates_passed / drv.leg_times

    Mirrors the minimal branch of race_pipeline._control_callback. Gate advance
    uses the same geometric plane-crossing test the GateSequencer uses for the
    minimal path (cross the gate plane within the opening), which is what credits
    a leg in the live run.
    """

    def __init__(self, gates: Sequence[Gate], cfg: Optional[ChampionConfig] = None,
                 opening_half: float = 0.75):
        self.cfg = cfg or ChampionConfig()
        # Bake the aim-z offset into the gate map exactly as the runner does.
        z = self.cfg.aim_z_offset
        self.gates: List[Gate] = [
            Gate((g.position[0], g.position[1], g.position[2] + z),
                 g.yaw, g.pitch, i)
            for i, g in enumerate(gates)
        ]
        self.opening_half = float(opening_half)
        cc = MinimalControllerConfig(
            cruise_speed=self.cfg.cruise_speed,
            max_tilt_rad=self.cfg.max_tilt_rad,
            kv=self.cfg.kv,
            max_vert_speed=self.cfg.max_vert_speed,
            max_lateral_accel=9.81 * math.tan(self.cfg.max_tilt_rad),
            vert_ff=0.0,
        )
        self.ctrl = MinimalController(cc)

        # sequencer / timing state
        self.target_idx = 0
        self.gates_passed = 0
        self.finished = False
        self.t = 0.0
        self.gate_cross_time: List[Optional[float]] = [None] * len(self.gates)
        self._prev_pos: Optional[np.ndarray] = None
        # slew memory (sim-time driven)
        self._cruise_cmd: Optional[float] = None
        self._aim_y_cmd: Optional[float] = None
        self._diff_cache: dict = {}

    # -- per-gate difficulty (identical formula to _gate_difficulty) --------- #
    def _gate_difficulty(self, idx: int) -> float:
        if idx in self._diff_cache:
            return self._diff_cache[idx]
        specs = self.gates
        diff = 0.0
        if 0 < idx < len(specs):
            p_prev = np.array(specs[idx - 1].position, float)
            p_cur = np.array(specs[idx].position, float)
            inc = p_cur - p_prev
            nxt = (np.array(specs[idx + 1].position, float) - p_cur
                   if idx + 1 < len(specs) else inc)
            ni, nn = float(np.linalg.norm(inc)), float(np.linalg.norm(nxt))
            theta = 0.0
            if ni > 1e-6 and nn > 1e-6:
                theta = math.acos(float(np.clip(np.dot(inc, nxt) / (ni * nn), -1.0, 1.0)))
            slope = abs(float(inc[2])) / max(1e-6, float(np.linalg.norm(inc[:2])))
            diff = max(theta, self.cfg.speed_descent_gain * slope)
        self._diff_cache[idx] = diff
        return diff

    def _advance_sequencer(self, pos: np.ndarray) -> None:
        """Plane-crossing gate advance. When the segment prev_pos->pos crosses the
        current target gate's plane (gate faces ~ -X, normal along its yaw) within
        the opening half-extent, credit the gate and advance the target."""
        if self.finished or self._prev_pos is None:
            return
        # advance through possibly several gates in one big step (rare)
        for _ in range(len(self.gates)):
            if self.target_idx >= len(self.gates):
                self.finished = True
                return
            g = self.gates[self.target_idx]
            gpos = np.array(g.position, float)
            n = np.array(_gate_normal(g.yaw, g.pitch), float)
            # signed distance to plane along the normal, prev vs cur
            d_prev = float(np.dot(self._prev_pos - gpos, n))
            d_cur = float(np.dot(pos - gpos, n))
            crossed = (d_prev <= 0.0 < d_cur) or (d_prev >= 0.0 > d_cur)
            if not crossed:
                return
            # interpolate crossing point; check it's within the opening
            denom = (d_cur - d_prev)
            frac = 0.5 if abs(denom) < 1e-9 else -d_prev / denom
            cross = self._prev_pos + frac * (pos - self._prev_pos)
            in_plane = np.linalg.norm((cross - gpos) - np.dot(cross - gpos, n) * n)
            if in_plane <= self.opening_half + 0.5:   # generous, matches body-crossing credit
                self.gate_cross_time[self.target_idx] = self.t
                self.gates_passed += 1
                self.target_idx += 1
            else:
                return
        if self.target_idx >= len(self.gates):
            self.finished = True

    def _aim_point(self, pos: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Compute the champion aim point for the current target gate (mirrors the
        minimal branch: through-dist along normal, final/normal z-bias, bounded
        anticipatory descent lookahead, lateral lead, aim-slew)."""
        cfg = self.cfg
        idx = min(self.target_idx, len(self.gates) - 1)
        gate = self.gates[idx]
        cur = np.array(gate.position, float)
        normal = np.array(_gate_normal(gate.yaw, gate.pitch), float)
        if float(np.dot(normal, cur - pos)) < 0:
            normal = -normal
        aim = cur + cfg.through_dist * normal
        last_idx = len(self.gates) - 1
        is_final = (idx == last_idx)
        # z bias: aim-z is baked into gate map already (so base offset 0); the
        # FINAL gate gets its smaller bias = (final_aim_z - aim_z) so the net
        # final aim height matches the live run (which applies final_aim_z_offset
        # on top of the already-baked map, but with aim_z_offset zeroed). The live
        # code sets _z_off = final_aim_z_offset directly (a small absolute bias).
        if is_final and cfg.final_aim_z_offset is not None:
            aim[2] += cfg.final_aim_z_offset
        # bounded anticipatory descent toward next (lower) gate
        if cfg.lookahead_m > 0.0 and idx + 1 <= last_idx:
            nxt = self.gates[idx + 1]
            nxt_z = float(nxt.position[2])      # aim_z baked in; no extra offset
            d_cur = float(np.linalg.norm(cur - pos))
            band = max(1e-3, cfg.lookahead_band_m)
            ramp = max(0.0, min(1.0, 1.0 - d_cur / band))
            aim[2] += min(cfg.lookahead_m, max(0.0, nxt_z - aim[2])) * ramp
        # lateral lead (champion: 0, so inert) — kept for parity
        if cfg.lat_lead_m > 0.0 and idx > 0:
            prev = self.gates[idx - 1]
            d_y = cur[1] - float(prev.position[1])
            if abs(d_y) > 1e-3:
                d_cur = float(np.linalg.norm(cur - pos))
                band = max(1e-3, cfg.lookahead_band_m)
                ramp = max(0.0, min(1.0, 1.0 - d_cur / band))
                aim[1] += cfg.lat_lead_m * (1.0 if d_y > 0 else -1.0) * ramp
        return aim, is_final

    def step(self, pos, vel, yaw: float, dt: float) -> AttitudeCommand:
        """One champion control tick. Advances the sequencer on the crossing,
        updates the sim-time slews, returns the AttitudeCommand to apply."""
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        dt = float(dt)
        self.t += dt
        self._advance_sequencer(pos)
        self._prev_pos = pos.copy()
        if self.finished or self.target_idx >= len(self.gates):
            self.finished = True
            return AttitudeCommand(0.0, 0.0, yaw, 0.4)   # hover (done)

        cfg = self.cfg
        idx = self.target_idx
        gate = self.gates[idx]
        last_idx = len(self.gates) - 1

        # --- variable-speed brake (per-leg) + sim-time slew -----------------
        if cfg.speed_brake > 0.0:
            base = cfg.cruise_speed
            diff = self._gate_difficulty(idx)
            if (cfg.final_brake_band_m > 0.0 and idx == last_idx and last_idx >= 2):
                rev = self._gate_difficulty(last_idx - 1)
                d_cur = float(np.linalg.norm(np.array(gate.position, float) - pos))
                ramp = max(0.0, min(1.0, 1.0 - d_cur / cfg.final_brake_band_m))
                diff = max(diff, rev * ramp)
            factor = max(cfg.speed_min_frac, min(1.0, 1.0 - cfg.speed_brake * diff))
            target_cruise = base * factor
            if self._cruise_cmd is None:
                self._cruise_cmd = target_cruise
            step = cfg.cruise_slew_rate * max(1e-3, min(0.1, dt))
            self._cruise_cmd += max(-step, min(step, target_cruise - self._cruise_cmd))
            self.ctrl.cfg.cruise_speed = self._cruise_cmd

        # --- aim point + aim-slew (sim-time) --------------------------------
        aim, _is_final = self._aim_point(pos)
        if cfg.aim_slew > 0.0:
            if self._aim_y_cmd is None:
                self._aim_y_cmd = float(aim[1])
            step = cfg.aim_slew * max(1e-3, min(0.1, dt))
            self._aim_y_cmd += max(-step, min(step, float(aim[1]) - self._aim_y_cmd))
            aim[1] = self._aim_y_cmd

        return self.ctrl.compute(pos, vel, yaw, tuple(aim), is_final_gate=False)

    # -- results ------------------------------------------------------------- #
    def leg_times(self) -> List[Optional[float]]:
        """gate_cross_time[i] is when the drone crossed gate i (sim-time s)."""
        return list(self.gate_cross_time)
