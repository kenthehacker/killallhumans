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

    Uses multi-start L-BFGS-B (iteration 19) to escape local minima.
    Research: T-MPC (de Groot, T-RO 2024) — parallel optimization from
    diverse homotopy seeds with fallback guarantee (Theorem 2).
    AERO-MPPI (Chen, ICRA 2026) — ensemble of M parallel optimizers
    from structurally different initializations.
    F1-Init (Shehadeh 2026) — initialization sensitivity causes
    convergence to suboptimal local minima.
    """

    N_STARTS = 10  # 1 zero + 1 late-apex + 8 random (deterministic seed)

    def __init__(self, config: RacingLineConfig = None):
        self.config = config or RacingLineConfig()

    def optimize(
        self,
        gates: List[GateWaypoint],
        start_position: Tuple[float, float, float] = (0, 0, 0),
    ) -> List[GateWaypoint]:
        """
        Optimize gate pass-through points for minimum time.

        Uses multi-start L-BFGS-B to explore multiple basins of attraction.
        Includes zero-initialization as fallback (guaranteed no regression
        per T-MPC Theorem 2, de Groot et al. T-RO 2024).

        Returns a new list of GateWaypoints with optimized positions
        (offset within gate openings for corner cutting).
        """
        if len(gates) < 2:
            return list(gates)

        n = len(gates)
        max_off = self.config.max_lateral_offset
        bounds = [(-max_off, max_off)] * (n * 2)

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

        # --- Multi-start initialization (iteration 19) ---
        # Research: AERO-MPPI runs M=15 parallel instances from diverse seeds.
        # T-MPC runs P=4 parallel MPCs from distinct homotopy classes.
        # F1-Init shows initialization quality determines which basin L-BFGS finds.
        candidates = []

        # Start 0: zero initialization (current baseline — fallback guarantee)
        candidates.append(np.zeros(n * 2))

        # Start 1: late-apex geometric prior for S-turns
        # For each gate, compute turn direction and offset to cut inside.
        # Research: F1-Init — geometric prior places optimizer in better basin.
        candidates.append(self._late_apex_init(gates, start_position, n, max_off))

        # Starts 2..N_STARTS-1: random initializations (deterministic seed)
        rng = np.random.default_rng(42)
        for _ in range(self.N_STARTS - 2):
            candidates.append(rng.uniform(-max_off, max_off, n * 2))

        # Run L-BFGS-B from each candidate, select best by objective value
        best_result = None
        for x0 in candidates:
            result = minimize(
                objective, x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300},  # raised from 100 (F1-Init: baseline needs 520 iters)
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        optimized_positions = self._apply_offsets(gates, best_result.x)
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

    def _late_apex_init(
        self,
        gates: List[GateWaypoint],
        start_position: Tuple[float, float, float],
        n: int,
        max_off: float,
    ) -> np.ndarray:
        """
        Compute a late-apex initialization for the racing line.

        For each gate, determines the turn direction (left/right) and
        sets the lateral offset to cut the inside of the turn.
        Research: F1-Init (Shehadeh 2026) — expert-like initialization
        places optimizer closer to the optimal basin.
        """
        x0 = np.zeros(n * 2)
        centers = [np.array(start_position)]
        for g in gates:
            centers.append(np.array(g.position))

        for i in range(n):
            if i == 0 or i >= n - 1:
                continue  # first and last gate: keep centered
            v_in = centers[i + 1] - centers[i]
            v_out = centers[i + 2] - centers[i + 1] if i + 2 < len(centers) else v_in
            # Cross product Z-component determines turn direction (NED: -Z is up)
            cross_z = v_in[0] * v_out[1] - v_in[1] * v_out[0]
            if abs(cross_z) > 0.1:
                # Cut inside the turn: positive cross_z = left turn → offset right
                x0[i] = -np.sign(cross_z) * max_off * 0.5
        return x0

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
