"""
Gate sequencer for drone racing.

Tracks which gate the drone should fly through next, detects gate pass-through
events using plane-crossing geometry, and manages gate highlight states.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from simulation.model_types import Gate, Pose3D


class GateSequencer:
    """
    Manages the ordered sequence of gates in a race.

    Pass-through detection: when the drone's position crosses the gate plane
    (signed distance flips sign between frames) and the crossing point is
    within the gate opening, the gate is considered passed.
    """

    def __init__(self, gates: List[Gate]):
        sorted_gates = sorted(
            [g for g in gates if g.sequence_index is not None],
            key=lambda g: g.sequence_index,
        )
        if not sorted_gates:
            raise ValueError("No gates with sequence_index found")

        self._gates = sorted_gates
        self._current_idx = 0
        self._passed: List[str] = []
        self._prev_position: Optional[Tuple[float, float, float]] = None

    @property
    def current_gate(self) -> Optional[Gate]:
        if self._current_idx < len(self._gates):
            return self._gates[self._current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self._current_idx >= len(self._gates)

    @property
    def gates_passed(self) -> int:
        return len(self._passed)

    @property
    def total_gates(self) -> int:
        return len(self._gates)

    @property
    def passed_gate_ids(self) -> List[str]:
        return list(self._passed)

    @property
    def all_gates(self) -> List[Gate]:
        return list(self._gates)

    def update(self, drone_position: Tuple[float, float, float]) -> Optional[Gate]:
        """
        Check if the drone has passed through the current gate.

        Call once per physics tick with the drone's current position.
        Returns the Gate that was just passed, or None.
        """
        if self.is_complete:
            return None

        passed_gate = None

        if self._prev_position is not None:
            gate = self._gates[self._current_idx]
            if self._check_pass_through(
                self._prev_position, drone_position, gate
            ):
                passed_gate = gate
                self._passed.append(gate.gate_id)
                self._current_idx += 1

        self._prev_position = drone_position
        return passed_gate

    def _check_pass_through(
        self,
        prev_pos: Tuple[float, float, float],
        curr_pos: Tuple[float, float, float],
        gate: Gate,
    ) -> bool:
        """
        Detect if the drone crossed through the gate plane between prev and curr.

        Uses signed-distance plane crossing: the gate's local X-axis is its
        forward/normal direction. If the signed distances from the gate plane
        have opposite signs for prev_pos and curr_pos, the drone crossed the
        plane. Then check if the crossing point is within the gate opening.
        """
        gate_pos = np.array([gate.pose.x, gate.pose.y, gate.pose.z])
        normal = self._gate_normal(gate.pose)

        prev = np.array(prev_pos)
        curr = np.array(curr_pos)

        d_prev = np.dot(prev - gate_pos, normal)
        d_curr = np.dot(curr - gate_pos, normal)

        if d_prev * d_curr > 0:
            return False

        # Find the crossing point via linear interpolation
        denom = d_curr - d_prev
        if abs(denom) < 1e-9:
            return False
        t = -d_prev / denom
        crossing = prev + t * (curr - prev)

        return self._point_in_gate_opening(crossing, gate)

    def _point_in_gate_opening(
        self, point: np.ndarray, gate: Gate
    ) -> bool:
        """Check if a 3D point falls within the gate's rectangular opening."""
        gate_pos = np.array([gate.pose.x, gate.pose.y, gate.pose.z])
        relative = point - gate_pos

        # Gate local axes
        normal = self._gate_normal(gate.pose)
        right = self._gate_right(gate.pose)
        up = np.array([0.0, 0.0, 1.0])

        # Apply pitch/roll for the up vector
        cy, sy = math.cos(gate.pose.yaw), math.sin(gate.pose.yaw)
        cp, sp = math.cos(gate.pose.pitch), math.sin(gate.pose.pitch)
        cr, sr = math.cos(gate.pose.roll), math.sin(gate.pose.roll)
        up = np.array([
            sy * sr + cy * sp * cr,
            -cy * sr + sy * sp * cr,
            cp * cr,
        ])

        local_right = np.dot(relative, right)
        local_up = np.dot(relative, up)

        half_w = gate.config.interior_width_m / 2.0
        half_h = gate.config.interior_height_m / 2.0

        # Allow some tolerance (1.5x the opening) for imprecise flight
        margin = 1.5
        return (
            abs(local_right) < half_w * margin
            and abs(local_up) < half_h * margin
        )

    @staticmethod
    def _gate_normal(pose: Pose3D) -> np.ndarray:
        """Gate forward direction (local +X after rotation)."""
        cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
        cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
        return np.array([cy * cp, sy * cp, sp])

    @staticmethod
    def _gate_right(pose: Pose3D) -> np.ndarray:
        """Gate rightward direction (local +Y after yaw rotation)."""
        cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
        return np.array([-sy, cy, 0.0])

    def reset(self):
        """Reset sequencer to the beginning."""
        self._current_idx = 0
        self._passed.clear()
        self._prev_position = None
