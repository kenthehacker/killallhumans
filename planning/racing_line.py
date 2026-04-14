"""
Racing line optimizer — compute the fastest path through gates.

Unlike the current Catmull-Rom spline which simply connects gate centers,
this module optimizes the lateral offset within each gate opening and
plans multi-gate lookahead to cut corners.

Key insights from the research:
  - TOGT Planner (Qin 2024): gates are regions, not points. The optimal
    path doesn't pass through gate centers but through strategically
    chosen points within the gate opening.
  - Swift (Kaufmann 2023): the RL agent learned to optimize trajectories
    on a longer timescale than human pilots, cutting corners aggressively.
  - Corner cutting: when two consecutive gates require a turn, fly through
    the inside edge of the first gate to reduce path curvature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .trajectory_optimizer import GateWaypoint


@dataclass
class RacingLineConfig:
    """Configuration for racing line optimization."""
    max_lateral_offset: float = 0.6    # max offset from gate center (fraction of half-width)
                                       # Increased from 0.4 (iter 13): optimizer was maxing out
                                       # at 0.339m, constraining corner-cutting.
                                       # TOGT (Qin 2024): gates are regions, not points.
                                       # 0.6 * 0.6m half-width = 0.36m offset, leaves 0.24m margin.
    corner_cut_aggressiveness: float = 0.7  # 0=center, 1=max corner cut
    speed_weight: float = 1.0          # importance of minimizing time
    smoothness_weight: float = 0.40    # importance of path smoothness
                                       # Increased from 0.3 (iter 13): with offset=0.6, smooth≥0.35
                                       # steers the racing line L-BFGS into a qualitatively
                                       # smoother local minimum. smooth=0.40 is the sweet spot:
                                       # helix tracking drops 73% (gate-7: 0.659→0.180m) while
                                       # S-turn (gate-3) stays acceptable (0.402→0.422m).
                                       # ILMPC (Zhao 2025): trajectory quality > controller tuning.
    lookahead_gates: int = 3           # gates to consider for corner cutting


class RacingLineOptimizer:
    """
    Optimizes the path through gate openings for minimum lap time.

    For each gate, finds the optimal pass-through point within the
    gate opening that minimizes total path length and curvature.
    """

    def __init__(self, config: RacingLineConfig = None):
        self.config = config or RacingLineConfig()

    def optimize(
        self,
        gates: List[GateWaypoint],
        start_position: Tuple[float, float, float] = (0, 0, 0),
    ) -> List[GateWaypoint]:
        """
        Optimize gate pass-through points for minimum time.

        Returns a new list of GateWaypoints with optimized positions
        (offset within gate openings for corner cutting).
        """
        if len(gates) < 2:
            return list(gates)

        n = len(gates)
        # Optimization variables: lateral offset for each gate [-1, 1]
        # and vertical offset [-1, 1]
        x0 = np.zeros(n * 2)  # (lateral, vertical) per gate

        def objective(offsets: np.ndarray) -> float:
            points = self._apply_offsets(gates, offsets)
            all_pts = [np.array(start_position)] + [np.array(p) for p in points]

            # Total path length (proxy for time)
            path_length = sum(
                np.linalg.norm(all_pts[i + 1] - all_pts[i])
                for i in range(len(all_pts) - 1)
            )

            # Path curvature penalty (smoother = faster in practice)
            curvature = 0.0
            for i in range(1, len(all_pts) - 1):
                v1 = all_pts[i] - all_pts[i - 1]
                v2 = all_pts[i + 1] - all_pts[i]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0.01 and n2 > 0.01:
                    cos_angle = np.dot(v1, v2) / (n1 * n2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = math.acos(cos_angle)
                    curvature += angle ** 2

            return (
                self.config.speed_weight * path_length
                + self.config.smoothness_weight * curvature
            )

        # Bounds: offsets limited to gate opening
        max_off = self.config.max_lateral_offset
        bounds = [(-max_off, max_off)] * (n * 2)

        result = minimize(
            objective, x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100},
        )

        optimized_positions = self._apply_offsets(gates, result.x)
        optimized_gates = []
        for i, gate in enumerate(gates):
            optimized_gates.append(GateWaypoint(
                position=optimized_positions[i],
                normal=gate.normal,
                width=gate.width,
                height=gate.height,
                yaw=gate.yaw,
            ))

        return optimized_gates

    def _apply_offsets(
        self,
        gates: List[GateWaypoint],
        offsets: np.ndarray,
    ) -> List[Tuple[float, float, float]]:
        """Apply lateral/vertical offsets to gate positions."""
        n = len(gates)
        positions = []
        for i in range(n):
            gate = gates[i]
            lat_off = offsets[i]
            vert_off = offsets[n + i]

            # Compute gate local axes
            cy = math.cos(gate.yaw)
            sy = math.sin(gate.yaw)
            right = np.array([-sy, cy, 0])  # local right
            up = np.array([0, 0, -1])        # NED: -z is up

            pos = np.array(gate.position)
            pos = pos + right * lat_off * gate.width * 0.5
            pos = pos + up * vert_off * gate.height * 0.5

            positions.append(tuple(pos))

        return positions


@dataclass
class SpeedProfile:
    """Speed targets along the trajectory."""
    times: List[float]
    speeds: List[float]
    gate_indices: List[int]  # which points correspond to gate passages


class SpeedProfiler:
    """
    Generates curvature-aware speed profiles.

    Slows down before turns, accelerates on straights.
    Target: 5-15 m/s depending on track geometry.
    """

    def __init__(
        self,
        max_speed: float = 15.0,
        min_speed: float = 2.0,
        max_accel: float = 8.0,
        max_decel: float = 10.0,
        turn_speed_factor: float = 0.4,
    ):
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.turn_speed_factor = turn_speed_factor

    def profile(
        self,
        waypoints: List[Tuple[float, float, float]],
    ) -> List[float]:
        """
        Compute target speed at each waypoint.

        Uses a two-pass approach:
          1. Forward pass: accelerate from current speed, limited by max_accel
          2. Backward pass: decelerate toward turns, limited by max_decel
          3. Take the minimum of both passes
        """
        n = len(waypoints)
        if n < 2:
            return [self.min_speed]

        # Compute curvature at each waypoint
        curvatures = self._compute_curvatures(waypoints)

        # Curvature → max speed: tighter turns → slower
        speed_from_curvature = []
        for k in curvatures:
            if k < 0.01:
                speed_from_curvature.append(self.max_speed)
            else:
                # v_max = sqrt(a_max / curvature)
                v = math.sqrt(self.max_accel / k)
                v = max(self.min_speed, min(v, self.max_speed))
                speed_from_curvature.append(v)

        # Forward pass: limited acceleration
        forward = [self.min_speed]
        for i in range(1, n):
            dist = _dist3(waypoints[i], waypoints[i - 1])
            v_max = math.sqrt(forward[-1] ** 2 + 2 * self.max_accel * dist)
            v_max = min(v_max, speed_from_curvature[i], self.max_speed)
            forward.append(v_max)

        # Backward pass: limited deceleration
        backward = [0.0] * n
        backward[-1] = speed_from_curvature[-1]
        for i in range(n - 2, -1, -1):
            dist = _dist3(waypoints[i], waypoints[i + 1])
            v_max = math.sqrt(backward[i + 1] ** 2 + 2 * self.max_decel * dist)
            v_max = min(v_max, speed_from_curvature[i], self.max_speed)
            backward[i] = v_max

        # Take minimum of both passes
        speeds = [
            max(min(f, b), self.min_speed)
            for f, b in zip(forward, backward)
        ]
        return speeds

    def _compute_curvatures(
        self, waypoints: List[Tuple[float, float, float]]
    ) -> List[float]:
        """Estimate curvature at each waypoint using discrete approximation."""
        n = len(waypoints)
        curvatures = [0.0] * n
        for i in range(1, n - 1):
            p0 = np.array(waypoints[i - 1])
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i + 1])
            v1 = p1 - p0
            v2 = p2 - p1
            cross = np.linalg.norm(np.cross(v1, v2))
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            denom = l1 * l2 * (l1 + l2) / 2
            if denom > 1e-6:
                curvatures[i] = cross / denom
        return curvatures


def _dist3(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )
