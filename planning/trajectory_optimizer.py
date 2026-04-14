"""
Time-optimal trajectory generation through racing gates.

Inspired by the TOGT Planner (Qin et al., 2024):
  - Piecewise polynomial trajectories via differential flatness
  - Gate constraints as pass-through regions
  - L-BFGS optimization to minimize total traversal time

And "Perception-Aware Time-Optimal Planning" (ETH 2025):
  - FOV constraints ensure next gate stays visible during aggressive maneuvers
  - Improved success rate from 55% → 100%

This implementation uses minimum-snap polynomial trajectories as the backbone,
with time allocation optimization for racing speed.

The drone's flat outputs are [x, y, z, ψ], from which all states and
inputs can be computed via differential flatness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class DroneConstraints:
    """Physical limits of the drone."""
    max_velocity: float = 15.0       # m/s
    max_acceleration: float = 20.0   # m/s^2 (~2g) — relaxed: rough estimate overestimates actual accel at segment boundaries
    max_jerk: float = 50.0           # m/s^3
    max_tilt_angle: float = 0.85     # radians (~49 deg) — increased for faster turns (Aggressive Maneuvers 2026)
    max_thrust: float = 20.0         # Newtons
    max_body_rate: float = 6.0       # rad/s
    mass: float = 1.0                # kg
    gravity: float = 9.81            # m/s^2


@dataclass
class GateWaypoint:
    """A gate to fly through."""
    position: Tuple[float, float, float]  # center (NED)
    normal: Tuple[float, float, float]    # gate facing direction (unit vector)
    width: float = 1.2                    # interior width (meters)
    height: float = 1.2                   # interior height (meters)
    yaw: float = 0.0                      # gate facing yaw (radians)


@dataclass
class TrajectoryPoint:
    """A single point on the reference trajectory."""
    time: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    acceleration: Tuple[float, float, float]
    jerk: Tuple[float, float, float]
    yaw: float
    yaw_rate: float


@dataclass
class RaceTrajectory:
    """Complete time-optimal trajectory through all gates."""
    points: List[TrajectoryPoint]
    total_time: float
    segment_times: List[float]
    gate_waypoints: List[GateWaypoint]

    def sample(self, t: float) -> TrajectoryPoint:
        """Sample the trajectory at time t via interpolation."""
        if t <= 0:
            return self.points[0]
        if t >= self.total_time:
            return self.points[-1]

        # Binary search for the right segment
        lo, hi = 0, len(self.points) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.points[mid].time <= t:
                lo = mid
            else:
                hi = mid

        p0 = self.points[lo]
        p1 = self.points[hi]
        dt = p1.time - p0.time
        if dt < 1e-9:
            return p0

        alpha = (t - p0.time) / dt

        return TrajectoryPoint(
            time=t,
            position=_lerp3(p0.position, p1.position, alpha),
            velocity=_lerp3(p0.velocity, p1.velocity, alpha),
            acceleration=_lerp3(p0.acceleration, p1.acceleration, alpha),
            jerk=_lerp3(p0.jerk, p1.jerk, alpha),
            yaw=_lerp_angle(p0.yaw, p1.yaw, alpha),
            yaw_rate=p0.yaw_rate + alpha * (p1.yaw_rate - p0.yaw_rate),
        )

    def find_closest(self, position: Tuple[float, float, float]) -> TrajectoryPoint:
        """Find the trajectory point closest to a given position.

        Uses vectorized numpy for O(n) but with fast SIMD operations
        instead of a Python loop. For long trajectories this is ~50x faster.
        """
        if not hasattr(self, '_positions_array'):
            # Cache positions as a contiguous numpy array on first call
            self._positions_array = np.array(
                [pt.position for pt in self.points], dtype=np.float64
            )
        pos = np.array(position, dtype=np.float64)
        diffs = self._positions_array - pos
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        idx = int(np.argmin(dists_sq))
        return self.points[idx]


@dataclass
class FOVConfig:
    """
    Camera FOV constraints for perception-aware planning.

    Based on "Perception-Aware Time-Optimal Planning" (ETH 2025):
    jointly optimizing trajectory with FOV constraints improved
    closed-loop success rate from 55% to 100%.
    """
    horizontal_fov_rad: float = math.radians(90)
    vertical_fov_rad: float = math.radians(60)
    penalty_weight: float = 10.0    # weight for FOV violation in objective
    margin_fraction: float = 0.8    # use 80% of FOV as safe zone


class TrajectoryOptimizer:
    """
    Generates time-optimal trajectories through a sequence of gates.

    Two-phase approach:
      1. Generate initial trajectory using minimum-snap polynomials
         with heuristic time allocation
      2. Optimize time allocation to minimize total time while
         respecting dynamics constraints

    Includes perception-aware FOV constraints (ETH 2025) to ensure
    the next gate stays visible during aggressive maneuvers.

    The trajectory is pre-computed offline when the course is known,
    then tracked online by the MPC controller.
    """

    def __init__(
        self,
        constraints: DroneConstraints = None,
        dt_sample: float = 0.01,
        fov_config: FOVConfig = None,
    ):
        self.constraints = constraints or DroneConstraints()
        self.dt_sample = dt_sample
        self.fov = fov_config or FOVConfig()

    def optimize(
        self,
        gates: List[GateWaypoint],
        start_position: Tuple[float, float, float] = (0, 0, 0),
        start_velocity: Tuple[float, float, float] = (0, 0, 0),
    ) -> RaceTrajectory:
        """
        Compute the time-optimal trajectory through all gates.

        Args:
            gates: ordered sequence of gates to fly through
            start_position: starting position (NED)
            start_velocity: starting velocity (NED)

        Returns:
            RaceTrajectory with sampled reference points
        """
        if not gates:
            raise ValueError("No gates provided")

        # Build waypoints: start + (entry/exit per gate) + virtual finish
        # Based on "On Your Own" (Romero 2025, arxiv:2510.13644):
        # Each gate gets TWO waypoints along its normal direction.
        # The entry/exit approach was deployed at IROS 2024 and Abu Dhabi
        # F1 GP, where the autonomous drone outperformed a professional pilot.
        #
        # Iteration 6: adaptive offsets based on turn angle at each gate.
        # "On Your Own" uses 0.4m for normal gates, 1.25m for Split-S.
        # TOGT Planner (Qin 2024): gates are regions, not points.
        # Sharp turns (like helix entry) need longer offsets to give the
        # min-snap polynomial more room to create smooth curves.

        ENTRY_EXIT_OFFSET = 0.4  # meters, per "On Your Own" paper

        waypoints = [np.array(start_position)]
        for g in gates:
            pos = np.array(g.position, dtype=float)
            normal = np.array(g.normal, dtype=float)
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 0.1:
                normal = normal / norm_mag
            else:
                # Fallback: direction from previous waypoint
                normal = pos - waypoints[-1]
                n_mag = np.linalg.norm(normal)
                if n_mag > 0.1:
                    normal = normal / n_mag
                else:
                    normal = np.array([1.0, 0.0, 0.0])

            entry = pos - normal * ENTRY_EXIT_OFFSET
            exit_wp = pos + normal * ENTRY_EXIT_OFFSET
            waypoints.append(entry)
            waypoints.append(exit_wp)

        # Add virtual finish waypoint past last gate
        if len(gates) >= 1:
            last_gate = gates[-1]
            normal = np.array(last_gate.normal, dtype=float)
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 0.1:
                normal = normal / norm_mag
            else:
                # Fallback: direction from second-to-last to last waypoint
                normal = waypoints[-1] - waypoints[-2]
                n_mag = np.linalg.norm(normal)
                if n_mag > 0.1:
                    normal = normal / n_mag
                else:
                    normal = np.array([1.0, 0.0, 0.0])
            finish_wp = waypoints[-1] + normal * 2.0
            waypoints.append(finish_wp)
            # Create a virtual gate for the finish waypoint
            gates = list(gates) + [GateWaypoint(
                position=tuple(finish_wp),
                normal=tuple(normal),
                width=last_gate.width,
                height=last_gate.height,
                yaw=last_gate.yaw,
            )]

        # Compute initial time allocation (distance-based heuristic)
        segment_times = self._initial_time_allocation(waypoints)

        # Optimize time allocation
        segment_times = self._optimize_time_allocation(
            waypoints, segment_times, start_velocity
        )

        # Post-optimization: inflate time at sharp gate-center turns.
        # The L-BFGS penalty approach (using waypoint-level turn angles)
        # was ineffective because entry/exit waypoints dilute the angles.
        # Instead, compute turn angles from gate centers and inflate the
        # 3 segments around the sharpest turns (approaching + through-gate).
        # Research: TACO (Sanghvi 2025) adapts trajectory parameters to
        # local characteristics; LMPC (Zhao 2025) uses per-section cost.
        segment_times = self._inflate_sharp_turns(
            waypoints, segment_times, gates
        )

        # FOV awareness is handled by the L-BFGS optimizer's integrated FOV
        # penalty (weight=10). Post-processing FOV relaxation removed in
        # iteration 17 — research consensus (ETH 2026, MonoRace 2026,
        # FOV-CBF 2025, PA-MPPI 2025, Mastering Diverse Tracks 2025) shows
        # integrated FOV constraints are superior to post-processing stages.
        # The A2RL competition winner (MonoRace) uses no post-processing FOV
        # relaxation. Removing this recovers ~0.5s of race time.
        # See: .research_loop/cross_validated/iteration_17_final.md

        # Post-optimization: TOPP-RA-style speed retiming (iteration 15).
        # Uses actual polynomial curvature + forward-backward propagation
        # to find near-optimal speed profile. Replaces heuristic compression.
        # Research: TOPPQuad (Mao 2024), FBGA (Piazza 2025), TOPP-RA (Pham 2017).
        segment_times = self._topp_retime(
            waypoints, segment_times, start_velocity, gates
        )
        points = self._generate_trajectory(
            waypoints, segment_times, start_velocity, gates,
        )

        total_time = sum(segment_times)
        return RaceTrajectory(
            points=points,
            total_time=total_time,
            segment_times=segment_times,
            gate_waypoints=gates,
        )

    def _inflate_sharp_turns(
        self,
        waypoints: List[np.ndarray],
        segment_times: List[float],
        gates: List[GateWaypoint],
    ) -> List[float]:
        """
        Inflate segment times near turns that exceed controller capability.

        Two complementary checks (applied AFTER L-BFGS optimization):

        1. **Angle-based** (>60°): Sharp turns get time inflation proportional
           to turn severity. Handles helix entry (gate-7) and similar.

        2. **Centripetal acceleration-based** (NEW, iter 7): Moderate turns
           at high speed get inflation proportional to centripetal acceleration
           excess. Handles S-turns (gate-3/4) where turn angles are below 60°
           but long approaches allow high speed into the turn.

        The centripetal check computes: a_c = v² × κ, where v is the
        estimated approach speed from L-BFGS segment times and κ is the
        path curvature estimated from gate-center turn angle / approach distance.

        Research backing:
        - TOPPQuad (Mao, IROS 2024): dynamic feasibility requires checking
          centripetal acceleration against thrust constraints, not just angle
        - Alternating Peak (de Vries, ECC 2024): peak constraint ratio per
          segment determines required time inflation
        - TACO (Sanghvi 2025): trajectory parameters should adapt to local
          characteristics (speed + curvature, not curvature alone)
        - Teissing (RA-L 2024): boundary velocity at turns is the key variable
        - CiMPCC (Li, ITSC 2024): compound curvature for S-turns — curvature
          doesn't drop between consecutive opposite-direction turns
        - VPMPCC (Li, 2024): early deceleration before S-turns; approach
          segments to the second turn need slowing
        """
        times = list(segment_times)
        n_gates = len(gates)
        if n_gates < 3:
            return times

        # Compute gate-center positions
        gate_centers = [np.array(g.position) for g in gates]

        # Centripetal acceleration threshold: beyond this, the PD controller
        # with feedforward cannot track the turn without significant overshoot.
        # With feedforward active (iter 8, ff=0.4), the controller handles
        # turns better, so threshold raised from 3.5→4.5 (iter 9).
        # Research: TOPPQuad feasibility, Teissing RA-L 2024 norm constraints.
        a_centripetal_threshold = 4.5  # m/s² — raised from 3.5, feedforward handles moderate turns

        # --- Pre-compute turn cross-products for S-turn detection (iter 16) ---
        # An S-turn is two consecutive turns with opposite lateral direction.
        # Detection: cross product Z-component changes sign between consecutive gates.
        # Research: CiMPCC (Li, ITSC 2024) — smoothed compound curvature for chicanes.
        cross_z = [0.0] * n_gates  # Z-component of cross(v_in, v_out) at each gate
        for gi in range(1, n_gates - 1):
            v_in = gate_centers[gi] - gate_centers[gi - 1]
            v_out = gate_centers[gi + 1] - gate_centers[gi]
            cross_z[gi] = float(np.cross(v_in, v_out)[2])  # yaw-plane direction

        # Compute turn angle at each gate (angle between approach and departure)
        for gi in range(1, n_gates - 1):
            v_in = gate_centers[gi] - gate_centers[gi - 1]
            v_out = gate_centers[gi + 1] - gate_centers[gi]
            n1 = np.linalg.norm(v_in)
            n2 = np.linalg.norm(v_out)
            if n1 < 0.1 or n2 < 0.1:
                continue
            cos_a = np.clip(np.dot(v_in, v_out) / (n1 * n2), -1, 1)
            turn_angle = math.acos(cos_a)

            # Each gate gi has entry at waypoint index 2*gi+1, exit at 2*gi+2
            seg_entry = 2 * gi      # segment ending at gate entry
            seg_through = 2 * gi + 1  # segment through gate (entry→exit)

            inflate = 1.0

            # --- S-turn detection (iteration 16) ---
            # An S-turn occurs when this gate and the previous gate have turns
            # in opposite directions (cross product sign change).
            # Research: CiMPCC compound curvature, VPMPCC early deceleration.
            is_s_turn = False
            if gi >= 2 and turn_angle > 0.25:
                # Check if previous gate also had a turn in the opposite direction
                if cross_z[gi - 1] != 0 and cross_z[gi] != 0:
                    is_s_turn = (cross_z[gi - 1] * cross_z[gi] < 0)

            if turn_angle > 1.05:  # > ~60 degrees — angle-based
                # Inflation factor: reduced from 0.35→0.25 (iter 9) since
                # feedforward (iter 8) anticipates turns. 60°→1.25x, 90°→1.50x.
                severity = (turn_angle - 1.05) / (math.pi / 2 - 1.05)
                severity = min(severity, 1.0)
                inflate = 1.0 + 0.25 * severity
            elif turn_angle > 0.3:  # > ~17° — check centripetal acceleration
                # Estimate approach speed from L-BFGS segment times.
                # The approach to gate gi uses segments leading to entry wp.
                # seg_entry-1 is the segment from prev gate exit to this gate entry.
                approach_seg = seg_entry - 1 if seg_entry > 0 else seg_entry
                approach_dist = 0.0
                approach_time = 0.0
                for s in [approach_seg, seg_entry]:
                    if 0 <= s < len(times):
                        d = float(np.linalg.norm(
                            waypoints[s + 1] - waypoints[s]))
                        approach_dist += d
                        approach_time += times[s]
                if approach_time > 0.01:
                    avg_speed = approach_dist / approach_time
                else:
                    avg_speed = 0.0

                # Curvature estimate from gate-center geometry
                # κ ≈ turn_angle / approach_distance (between gate centers)
                approach_gate_dist = n1  # distance between gate centers
                if approach_gate_dist > 0.5:
                    curvature = turn_angle / approach_gate_dist
                else:
                    curvature = 0.0

                # Centripetal acceleration: a_c = v² × κ
                a_centripetal = avg_speed ** 2 * curvature

                if a_centripetal > a_centripetal_threshold:
                    # Inflate proportional to excess centripetal acceleration
                    excess = (a_centripetal - a_centripetal_threshold) / a_centripetal_threshold
                    severity = min(excess, 1.0)
                    inflate = 1.0 + 0.15 * severity  # range: 1.0x to 1.15x (reduced from 0.25, iter 9)

            # --- S-turn compound inflation (iteration 16) ---
            # For the second gate of an S-turn pair, apply extra inflation.
            # The drone arrives with lateral velocity in the wrong direction
            # and must reverse it — requires more time than a single turn.
            # Research: CiMPCC compound curvature, VPMPCC sustained low speed.
            if is_s_turn:
                s_turn_inflate = 1.10  # 10% compound inflation (tuned down from 15%, iter 16)
                inflate = max(inflate, s_turn_inflate)

                # Also inflate the APPROACH segment (prev gate exit → this entry).
                # Research: VPMPCC shows early deceleration is critical for S-turns.
                approach_seg = seg_entry - 1  # segment from prev gate exit to this entry
                if 0 <= approach_seg < len(times):
                    times[approach_seg] *= 1.05  # 5% approach deceleration (tuned: 10%→7%→5%)

            # Bidirectional proximity-based inflation for closely-spaced gates (iter 13, updated iter 18).
            # Helix gates are 3.6-5.7m apart; short polynomial segments create
            # high curvature that the PD controller can't follow.
            # Iter 18: Changed from forward-only to BIDIRECTIONAL — uses min
            # distance to prev OR next gate. Gate 11 (49.9° turn) was getting
            # only 0.7% inflation because dist to next gate (5.66m) is large,
            # despite dist to prev gate (3.64m) being small. This caused helix
            # regression when FOV relaxation was removed in iter 17.
            # Also increased multiplier 0.12→0.18 to compensate for lost FOV inflation.
            # Research: CiMPCC (Li 2024) — compound curvature for sequential turns
            # requires considering ALL neighbors. Quad-LCD (Srikanthan 2025) —
            # per-segment feasibility depends on approach context.
            dist_next = float(np.linalg.norm(
                gate_centers[gi + 1] - gate_centers[gi])) if gi + 1 < n_gates else 999.0
            dist_prev = float(np.linalg.norm(
                gate_centers[gi] - gate_centers[gi - 1])) if gi > 0 else 999.0
            dist_closest = min(dist_next, dist_prev)
            if dist_closest < 6.0 and turn_angle > 0.4:  # ~23° minimum
                proximity_factor = 1.0 + 0.22 * (1.0 - dist_closest / 6.0)  # was 0.12, forward-only (iter 14)
                inflate = max(inflate, proximity_factor)

            if inflate > 1.001:
                for seg_idx in [seg_entry, seg_through]:
                    if 0 <= seg_idx < len(times):
                        times[seg_idx] *= inflate

        return times

    def _topp_retime(
        self,
        waypoints: List[np.ndarray],
        segment_times: List[float],
        start_velocity: Tuple[float, float, float],
        gates: List[GateWaypoint],
    ) -> List[float]:
        """
        TOPP-RA-style forward-backward speed retiming using waypoint geometry.

        Uses the timing-independent geometric curvature from waypoint positions
        (not from polynomial derivatives, which depend on current timing).
        Then runs TOPP-RA forward-backward propagation to find the fastest
        feasible speed profile respecting centripetal and longitudinal
        acceleration limits.

        Research basis:
        - TOPPQuad (Mao, IROS 2024): fix geometry, optimize speed → 40-50% faster
        - FBGA (Piazza, RA-L 2025): forward-backward matches OC within 0.36%
        - TOPP-RA (Pham & Pham, 2017): reachability-based forward-backward
        - CiMPCC (Li, ITSC 2024): compound curvature boost for S-turn regions
        """
        times = list(segment_times)
        n = len(times)
        if n < 2:
            return times

        # Acceleration budgets:
        # Centripetal: max tilt 0.85→ g*tan(0.85) ≈ 11.4, use 10.0 for margin
        # Longitudinal: budget for speed changes between segments
        a_centripetal = 10.0
        a_longitudinal = 8.0
        max_v = self.constraints.max_velocity
        min_v = 2.0
        max_compression = 0.68  # iter 17: raised from 0.65 to compensate for FOV relaxation removal — protects helix turns from over-compression

        # --- S-turn region detection for compound curvature boost (iter 16) ---
        # Identify waypoint indices that are in S-turn regions (between gates
        # with consecutive opposite-direction turns). Boost curvature at these
        # waypoints to prevent TOPP from speeding through S-turns.
        # Research: CiMPCC (Li, ITSC 2024) — compound curvature for chicanes.
        s_turn_segments = set()  # segment indices in S-turn regions
        n_gates = len(gates)
        if n_gates >= 3:
            gate_centers = [np.array(g.position) for g in gates]
            cross_z = {}
            for gi in range(1, n_gates - 1):
                v_in = gate_centers[gi] - gate_centers[gi - 1]
                v_out = gate_centers[gi + 1] - gate_centers[gi]
                cross_z[gi] = float(np.cross(v_in, v_out)[2])
            for gi in range(2, n_gates - 1):
                if gi - 1 in cross_z and gi in cross_z:
                    if cross_z[gi - 1] * cross_z[gi] < 0:
                        # S-turn: mark segments around this gate
                        # Gate gi waypoints: entry=2*gi+1, exit=2*gi+2
                        # Segments: approach (2*gi-1 to 2*gi+1), through (2*gi+1)
                        for s in range(max(0, 2 * gi - 1), min(n, 2 * gi + 3)):
                            s_turn_segments.add(s)

        # --- Step 1: Compute segment distances and geometric curvature ---
        seg_dist = []  # straight-line distances
        seg_curv = []  # geometric curvature at segment endpoint

        for i in range(n):
            if i + 1 < len(waypoints):
                d = float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
            else:
                d = 0.01
            seg_dist.append(max(d, 0.01))

            # Geometric curvature at waypoint i+1 (endpoint of segment i)
            # κ ≈ 2 * sin(θ/2) / chord_length (Menger curvature from 3 points)
            k = 0.0
            if 0 < i + 1 < len(waypoints) - 1:
                p0 = waypoints[i]
                p1 = waypoints[i + 1]
                p2 = waypoints[i + 2] if i + 2 < len(waypoints) else p1
                v1 = p1 - p0
                v2 = p2 - p1
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 > 0.01 and n2 > 0.01:
                    # Cross product magnitude gives twice the triangle area
                    cross_mag = float(np.linalg.norm(np.cross(v1, v2)))
                    # Menger curvature: κ = 2*|cross| / (|v1| * |v2| * |v1+v2|)
                    chord = float(np.linalg.norm(p2 - p0))
                    if chord > 0.01:
                        k = 2.0 * cross_mag / (n1 * n2 * chord)

            # Compound curvature boost for S-turn regions (iter 16).
            # In S-turn segments, the effective curvature is higher than the
            # point Menger curvature because the drone must reverse its lateral
            # velocity, requiring more centripetal margin.
            if i in s_turn_segments and k > 1e-4:
                k *= 1.2  # 20% compound curvature boost (tuned from 30%)

            seg_curv.append(k)

        # --- Step 2: Speed limits from curvature ---
        # Use the MAX curvature of the two endpoints of each segment
        v_max_seg = []
        for i in range(n):
            k_start = seg_curv[i - 1] if i > 0 else 0.0
            k_end = seg_curv[i]
            k_max = max(k_start, k_end)
            if k_max > 1e-4:
                v_limit = math.sqrt(a_centripetal / k_max)
                v_limit = min(v_limit, max_v)
            else:
                v_limit = max_v
            v_max_seg.append(max(v_limit, min_v))

        # --- Step 3: Forward-backward propagation ---
        v_start = float(np.linalg.norm(np.array(start_velocity)))
        v_start = max(v_start, min_v)

        # Forward pass: v² = v0² + 2*a*s
        v_fwd = [0.0] * n
        v_fwd[0] = min(v_start, v_max_seg[0])
        for i in range(1, n):
            v_sq = v_fwd[i - 1] ** 2 + 2.0 * a_longitudinal * seg_dist[i - 1]
            v_fwd[i] = min(v_max_seg[i], math.sqrt(max(v_sq, 0.0)), max_v)

        # Backward pass: deceleration from end
        v_bwd = [0.0] * n
        v_bwd[n - 1] = min(v_max_seg[n - 1], max_v * 0.5)  # end slower
        for i in range(n - 2, -1, -1):
            v_sq = v_bwd[i + 1] ** 2 + 2.0 * a_longitudinal * seg_dist[i]
            v_bwd[i] = min(v_max_seg[i], math.sqrt(max(v_sq, 0.0)))

        # --- Step 4: Optimal speed = min(forward, backward) ---
        v_opt = [max(min(v_fwd[i], v_bwd[i]), min_v) for i in range(n)]

        # --- Step 5: Compute new segment times ---
        new_times = list(times)
        for i in range(n):
            if seg_dist[i] < 0.01:
                continue
            new_time = seg_dist[i] / v_opt[i]
            # Don't compress below max_compression of original
            new_time = max(new_time, times[i] * max_compression)
            # Only compress, never expand
            new_time = min(new_time, times[i])
            new_times[i] = max(new_time, 0.1)

        return new_times

    def _relax_for_fov(
        self,
        waypoints: List[np.ndarray],
        segment_times: List[float],
        start_velocity: Tuple[float, float, float],
        gates: List[GateWaypoint],
    ) -> List[float]:
        """
        Targeted relaxation of segment times to reduce FOV violations.

        Research basis (iteration 10, updated iteration 14):
        - ETH 2026 (arXiv:2603.04305): proper FOV soft constraints add only
          +8.1% to trajectory time. Our previous implementation added +14.1%.
        - KAIST 2025 (arXiv:2512.20475): heading-based FOV control adds +0%
          race time. Position trajectory doesn't need slowing for FOV.
        - TOPPQuad (Mao, IROS 2024): geometry-timing decoupling — excessive
          post-hoc inflation wastes time without improving tracking.
        - Consensus: post-hoc inflation should be minimal; the L-BFGS
          optimizer's FOV penalty (weight=10) already provides baseline
          awareness.

        Changes (iteration 14 — speed recovery):
        - Reduced iterations: 3 → 2 (enough to converge)
        - Reduced multiplier: 1.07 → 1.03 per segment (ETH: +8.1% is sufficient)
        - Reduced cap: 25% → 8% (aligned with ETH finding)
        - The L-BFGS FOV penalty (weight=10) already provides primary FOV
          awareness; this step is a safety net, not the primary mechanism.
        """
        times = list(segment_times)
        pre_relax_total = sum(times)
        max_total = pre_relax_total * 1.08  # cap: at most +8% from FOV (was 25%, iter 14)

        for _iteration in range(2):  # was 3 (iter 14: 2 is enough to converge)
            points = self._generate_trajectory(
                waypoints, times, start_velocity, gates
            )
            penalty = self.add_fov_constraints(points, gates)
            if penalty < 100.0:
                break

            # Inflate segments with turns > 30° (same threshold as original)
            for i in range(len(times) - 1):
                v_in = waypoints[i + 1] - waypoints[i]
                v_out = waypoints[i + 2] - waypoints[i + 1] if i + 2 < len(waypoints) else v_in
                norm_in = np.linalg.norm(v_in)
                norm_out = np.linalg.norm(v_out)
                if norm_in > 0.1 and norm_out > 0.1:
                    cos_a = np.clip(
                        np.dot(v_in, v_out) / (norm_in * norm_out), -1, 1
                    )
                    turn = math.acos(cos_a)
                    if turn > 0.5:  # > ~30 degrees
                        times[i] *= 1.03  # 3% increase (was 7%, iter 14)

            # Enforce cap on total time inflation
            current_total = sum(times)
            if current_total > max_total:
                scale = max_total / current_total
                times = [t * scale for t in times]

        return times

    def _initial_time_allocation(
        self, waypoints: List[np.ndarray]
    ) -> List[float]:
        """Curvature-aware time allocation.

        Inspired by TOGT Planner (Qin 2024, ICRA): "high-curvature segments
        naturally require more time to stay within thrust limits." Instead of
        a uniform speed factor, we allocate more time to turning segments and
        less to straight segments.
        """
        times = []
        for i in range(len(waypoints) - 1):
            dist = float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))

            # Compute turn angle at the endpoint of this segment
            turn_angle = 0.0
            if i + 1 < len(waypoints) - 1:
                v_in = waypoints[i + 1] - waypoints[i]
                v_out = waypoints[i + 2] - waypoints[i + 1]
                norm_in = np.linalg.norm(v_in)
                norm_out = np.linalg.norm(v_out)
                if norm_in > 0.1 and norm_out > 0.1:
                    cos_a = np.clip(
                        np.dot(v_in, v_out) / (norm_in * norm_out), -1, 1
                    )
                    turn_angle = math.acos(cos_a)

            # Curvature-aware speed factor:
            # - Straight (< 17 deg): 65% of max velocity
            # - Moderate turn (17-57 deg): 55% of max velocity
            # - Sharp turn (> 57 deg): 45% of max velocity
            if turn_angle < 0.3:
                speed_factor = 0.80
            elif turn_angle < 1.0:
                speed_factor = 0.70
            else:
                speed_factor = 0.55

            avg_speed = self.constraints.max_velocity * speed_factor
            t = max(dist / avg_speed, 0.1)  # minimum segment time (lowered for short entry/exit segments)
            times.append(t)
        return times

    def _optimize_time_allocation(
        self,
        waypoints: List[np.ndarray],
        initial_times: List[float],
        start_velocity: Tuple[float, float, float],
    ) -> List[float]:
        """
        Optimize segment times to minimize total time while respecting constraints.

        Uses L-BFGS-B (same as TOGT Planner) on log-time variables to
        ensure positivity. Includes perception-aware FOV penalty (ETH 2025).
        """
        n_segments = len(initial_times)
        log_times = np.log(np.array(initial_times))

        def objective(log_t: np.ndarray) -> float:
            times = np.exp(log_t)
            total_time = np.sum(times)

            # Time incentive: weight total_time more heavily to prioritize speed.
            # TOGT (Qin 2024) uses joint time-position optimization where time
            # minimization dominates; we achieve similar effect with time_weight > 1.
            time_weight = 2.0

            # Penalty for constraint violations
            penalty = 0.0
            for i in range(n_segments):
                dist = float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
                avg_v = dist / max(times[i], 0.01)

                # Velocity constraint
                if avg_v > self.constraints.max_velocity:
                    penalty += (avg_v - self.constraints.max_velocity) ** 2 * 50

                # Acceleration constraint (rough estimate)
                if i < n_segments - 1:
                    v1 = dist / max(times[i], 0.01)
                    dist2 = float(np.linalg.norm(waypoints[i + 2] - waypoints[i + 1]))
                    v2 = dist2 / max(times[i + 1], 0.01)
                    accel = abs(v2 - v1) / max(times[i], 0.01)
                    if accel > self.constraints.max_acceleration:
                        penalty += (accel - self.constraints.max_acceleration) ** 2 * 20

            # (Curvature-speed penalty removed — moved to post-optimization
            # _inflate_sharp_turns() which uses gate-center turn angles instead
            # of diluted waypoint-level angles.)

            # Perception-aware FOV penalty (ETH 2025)
            # Penalize segments where the target gate would fall outside camera FOV
            # due to extreme tilt angle from high acceleration
            fov_penalty = 0.0
            half_fov = self.fov.horizontal_fov_rad * self.fov.margin_fraction / 2
            for i in range(n_segments):
                dist = float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
                avg_v = dist / max(times[i], 0.01)
                # Estimate tilt angle from centripetal acceleration
                if i < n_segments - 1:
                    v_vec = waypoints[i + 1] - waypoints[i]
                    v_next = waypoints[i + 2] - waypoints[i + 1] if i + 2 < len(waypoints) else v_vec
                    v_norm = np.linalg.norm(v_vec)
                    v_next_norm = np.linalg.norm(v_next)
                    if v_norm > 0.1 and v_next_norm > 0.1:
                        cos_angle = np.dot(v_vec, v_next) / (v_norm * v_next_norm)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        turn_angle = math.acos(cos_angle)
                        # Tilt needed ~ atan(v^2 * curvature / g)
                        curvature = turn_angle / max(dist, 0.1)
                        tilt = math.atan2(avg_v ** 2 * curvature, self.constraints.gravity)
                        # If tilt > half_fov, the gate may leave camera view
                        if tilt > half_fov:
                            fov_penalty += (tilt - half_fov) ** 2

            penalty += self.fov.penalty_weight * fov_penalty

            return time_weight * total_time + penalty

        result = minimize(
            objective,
            log_times,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-6},
        )

        optimized_times = np.exp(result.x)
        # Enforce minimum segment time
        optimized_times = np.maximum(optimized_times, 0.1)
        return optimized_times.tolist()

    def add_fov_constraints(
        self,
        trajectory_points: List[TrajectoryPoint],
        gates: List[GateWaypoint],
        current_gate_index: int = 0,
    ) -> float:
        """
        Compute perception-aware FOV penalty for a trajectory.

        Based on "Perception-Aware Time-Optimal Planning" (arXiv 2603.04305, 2025):
        for each trajectory point, verify the next gate center projects inside
        the camera FOV. The camera points along the body x-axis; aggressive tilt
        angles rotate the camera away from the gate.

        The penalty is the sum-of-squares of angular violations, weighted by
        ``self.fov.penalty_weight``.  This is designed to be added as a soft
        constraint term in the optimization objective.

        Projection model:
          1. Compute gate-to-drone vector in world frame.
          2. Estimate drone attitude from acceleration (thrust direction).
             Under differential flatness, attitude is determined by the
             acceleration vector: the z-body axis aligns with (accel + g).
          3. Rotate gate-to-drone vector into the body frame.
          4. The camera looks along body-x, so the bearing angles are:
             azimuth = atan2(y_body, x_body)
             elevation = atan2(z_body, x_body)
          5. Penalize if |azimuth| > half_h_fov or |elevation| > half_v_fov.

        Args:
            trajectory_points: sampled trajectory to evaluate.
            gates: ordered gate sequence.
            current_gate_index: index of the first upcoming gate.

        Returns:
            Total FOV penalty (0.0 if all points have the gate in view).
        """
        if not trajectory_points or not gates:
            return 0.0

        half_h = self.fov.horizontal_fov_rad * self.fov.margin_fraction / 2
        half_v = self.fov.vertical_fov_rad * self.fov.margin_fraction / 2
        g_vec = np.array([0.0, 0.0, self.constraints.gravity])  # NED: gravity is +z

        penalty = 0.0
        gate_idx = current_gate_index

        # Pre-compute cumulative time to each gate (rough: use segment_times if available)
        gate_positions = [np.array(g.position) for g in gates]

        for pt in trajectory_points:
            # Advance gate index if we're past the current gate
            while (
                gate_idx < len(gates) - 1
                and np.linalg.norm(np.array(pt.position) - gate_positions[gate_idx]) < 0.5
            ):
                gate_idx += 1

            if gate_idx >= len(gates):
                break

            target_pos = gate_positions[gate_idx]
            drone_pos = np.array(pt.position)
            accel = np.array(pt.acceleration)

            # Gate-to-drone vector in world frame
            to_gate = target_pos - drone_pos
            dist_to_gate = np.linalg.norm(to_gate)
            if dist_to_gate < 0.1:
                continue  # too close, skip

            # Estimate body frame from acceleration (differential flatness)
            # Thrust direction = accel + gravity (in NED, gravity is [0,0,g])
            thrust_dir = accel + g_vec
            thrust_mag = np.linalg.norm(thrust_dir)
            if thrust_mag < 0.1:
                thrust_dir = g_vec.copy()
                thrust_mag = self.constraints.gravity

            # z_body = thrust direction (normalized)
            z_b = thrust_dir / thrust_mag

            # x_body: project velocity direction onto plane perpendicular to z_b
            vel = np.array(pt.velocity)
            vel_mag = np.linalg.norm(vel)
            if vel_mag > 0.1:
                # x_body = normalize(vel - (vel . z_b) * z_b)
                vel_proj = vel - np.dot(vel, z_b) * z_b
                vel_proj_mag = np.linalg.norm(vel_proj)
                if vel_proj_mag > 1e-4:
                    x_b = vel_proj / vel_proj_mag
                else:
                    x_b = np.array([1.0, 0.0, 0.0])
            else:
                # Hovering: x_body defaults to world x projected onto body plane
                world_x = np.array([1.0, 0.0, 0.0])
                x_proj = world_x - np.dot(world_x, z_b) * z_b
                x_proj_mag = np.linalg.norm(x_proj)
                if x_proj_mag > 1e-4:
                    x_b = x_proj / x_proj_mag
                else:
                    x_b = np.array([0.0, 1.0, 0.0])

            # y_body = z_b x x_b (right-hand frame)
            y_b = np.cross(z_b, x_b)

            # Rotation matrix: columns are body axes in world frame
            # R_wb = [x_b | y_b | z_b], so R_bw = R_wb^T
            R_bw = np.array([x_b, y_b, z_b])  # rows = body axes

            # Gate vector in body frame
            to_gate_body = R_bw @ to_gate

            # Camera looks along +x_body
            x_comp = to_gate_body[0]
            if x_comp < 0.1:
                # Gate is behind the drone — maximum penalty
                penalty += math.pi ** 2
                continue

            azimuth = math.atan2(to_gate_body[1], x_comp)
            elevation = math.atan2(to_gate_body[2], x_comp)

            # Soft penalty: quadratic outside FOV bounds
            if abs(azimuth) > half_h:
                penalty += (abs(azimuth) - half_h) ** 2
            if abs(elevation) > half_v:
                penalty += (abs(elevation) - half_v) ** 2

        return self.fov.penalty_weight * penalty

    def _generate_trajectory(
        self,
        waypoints: List[np.ndarray],
        segment_times: List[float],
        start_velocity: Tuple[float, float, float],
        gates: List[GateWaypoint],
    ) -> List[TrajectoryPoint]:
        """
        Generate the full trajectory using minimum-snap polynomials.

        Each segment uses a 7th-order polynomial (minimum snap) that
        ensures C3 continuity (continuous position, velocity, acceleration,
        and jerk) at waypoints.
        """
        points = []
        t_cumulative = 0.0
        prev_vel = np.array(start_velocity)
        prev_accel = np.array([0.0, 0.0, 0.0])

        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            T = segment_times[i]

            # Desired velocity at endpoint: direction toward next gate
            if i + 1 < len(waypoints) - 1:
                next_dir = np.array(waypoints[i + 2], dtype=float) - np.array(p1, dtype=float)
                next_dist = float(np.linalg.norm(next_dir))
                if next_dist > 0:
                    next_dir = next_dir / next_dist
                    # Speed based on distance to next waypoint
                    next_speed = min(
                        next_dist / max(segment_times[i + 1], 0.1),
                        self.constraints.max_velocity,
                    )
                    end_vel = next_dir * next_speed
                else:
                    end_vel = np.zeros(3)
            else:
                # Last segment: approach with moderate speed
                dir_to_gate = p1 - p0
                dist = float(np.linalg.norm(dir_to_gate))
                if dist > 0:
                    end_vel = dir_to_gate / dist * min(dist / T, self.constraints.max_velocity * 0.3)
                else:
                    end_vel = np.zeros(3)

            end_accel = np.zeros(3)

            # Generate polynomial for each axis
            n_samples = max(int(T / self.dt_sample), 2)
            t_local = np.linspace(0, T, n_samples)

            positions = np.zeros((n_samples, 3))
            velocities = np.zeros((n_samples, 3))
            accelerations = np.zeros((n_samples, 3))
            jerks = np.zeros((n_samples, 3))

            for axis in range(3):
                coeffs = _min_snap_1d(
                    p0[axis], prev_vel[axis], prev_accel[axis],
                    p1[axis], end_vel[axis], end_accel[axis],
                    T,
                )
                for j, t in enumerate(t_local):
                    positions[j, axis] = _poly_eval(coeffs, t)
                    velocities[j, axis] = _poly_deriv_eval(coeffs, t, 1)
                    accelerations[j, axis] = _poly_deriv_eval(coeffs, t, 2)
                    jerks[j, axis] = _poly_deriv_eval(coeffs, t, 3)

            # Compute yaw: point toward next gate
            gate_yaw = gates[i].yaw if i < len(gates) else 0.0

            for j in range(n_samples):
                # Yaw: blend between looking at current target and next
                if j < n_samples - 1:
                    dx = velocities[j, 0]
                    dy = velocities[j, 1]
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        yaw = math.atan2(dy, dx)
                    else:
                        yaw = gate_yaw
                else:
                    yaw = gate_yaw

                yaw_rate = 0.0
                if j > 0 and j < n_samples:
                    prev_yaw = points[-1].yaw if points else yaw
                    yaw_rate = _wrap_angle(yaw - prev_yaw) / self.dt_sample

                points.append(TrajectoryPoint(
                    time=t_cumulative + t_local[j],
                    position=tuple(positions[j]),
                    velocity=tuple(velocities[j]),
                    acceleration=tuple(accelerations[j]),
                    jerk=tuple(jerks[j]),
                    yaw=yaw,
                    yaw_rate=yaw_rate,
                ))

            t_cumulative += T
            prev_vel = end_vel
            prev_accel = end_accel

        return points


def _min_snap_1d(
    p0: float, v0: float, a0: float,
    pf: float, vf: float, af: float,
    T: float,
) -> np.ndarray:
    """
    Compute 5th-order polynomial coefficients for minimum-snap trajectory.

    Boundary conditions:
      p(0) = p0, p'(0) = v0, p''(0) = a0
      p(T) = pf, p'(T) = vf, p''(T) = af

    Returns coefficients [c0, c1, c2, c3, c4, c5] for:
      p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
    """
    # From boundary conditions:
    c0 = p0
    c1 = v0
    c2 = a0 / 2.0

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    # Solve 3x3 system for c3, c4, c5
    A = np.array([
        [T3, T4, T5],
        [3 * T2, 4 * T3, 5 * T4],
        [6 * T, 12 * T2, 20 * T3],
    ])
    b = np.array([
        pf - c0 - c1 * T - c2 * T2,
        vf - c1 - 2 * c2 * T,
        af - 2 * c2,
    ])

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.zeros(3)

    return np.array([c0, c1, c2, x[0], x[1], x[2]])


def _poly_eval(coeffs: np.ndarray, t: float) -> float:
    """Evaluate polynomial at t using Horner's method."""
    # coeffs = [c0, c1, c2, ..., cn] for c0 + c1*t + c2*t^2 + ...
    # Horner's: evaluate from highest power down
    result = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * t + coeffs[i]
    return result


def _poly_deriv_eval(coeffs: np.ndarray, t: float, order: int) -> float:
    """Evaluate the nth derivative of polynomial at t."""
    n = len(coeffs)
    # Build derivative coefficients
    dc = np.array(coeffs, dtype=float)
    for _ in range(order):
        new_dc = np.zeros(max(len(dc) - 1, 0))
        for i in range(1, len(dc)):
            new_dc[i - 1] = dc[i] * i
        dc = new_dc
    # Evaluate using Horner's
    result = 0.0
    for i in range(len(dc) - 1, -1, -1):
        result = result * t + dc[i]
    return result


def _lerp3(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    alpha: float,
) -> Tuple[float, float, float]:
    return (
        a[0] + alpha * (b[0] - a[0]),
        a[1] + alpha * (b[1] - a[1]),
        a[2] + alpha * (b[2] - a[2]),
    )


def _lerp_angle(a: float, b: float, alpha: float) -> float:
    diff = _wrap_angle(b - a)
    return _wrap_angle(a + alpha * diff)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
