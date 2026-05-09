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

    def __init__(
        self,
        gates: List[Gate],
        pass_through_margin: float = 1.5,
    ):
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
        # Crash/miss tracking. See gate_sequencing.sequencer for the
        # platform-agnostic version of this contract — both surfaces are
        # kept aligned so DynamicReplanner can consume either one.
        self._crashes: List[Tuple[str, Tuple[float, float, float]]] = []
        self._misses: List[str] = []
        self._last_event: Optional[str] = None  # 'pass'|'crash'|'miss'|None
        # Pass-through margin: lateral/vertical opening is stretched by this
        # factor when classifying a plane crossing as 'pass'. Default 1.5
        # preserves the original imprecise-flight tolerance baked into
        # the runner. Tests use 1.0 to make the crash zone non-empty.
        self._pass_margin = float(pass_through_margin)

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

        Side effects: classifies non-pass plane crossings as either
        'crash' (inside outer frame, outside opening) or 'miss' (outside
        outer frame). Both are recorded for the dynamic replanner.
        """
        if self.is_complete:
            return None

        passed_gate = None

        if self._prev_position is not None:
            gate = self._gates[self._current_idx]
            crossing = self._compute_crossing(
                self._prev_position, drone_position, gate
            )
            if crossing is not None:
                if self._point_in_gate_opening(crossing, gate):
                    passed_gate = gate
                    self._passed.append(gate.gate_id)
                    self._current_idx += 1
                    self._last_event = "pass"
                elif self._point_in_outer_frame(crossing, gate):
                    self._crashes.append(
                        (gate.gate_id, tuple(float(c) for c in crossing))
                    )
                    self._last_event = "crash"
                else:
                    self._misses.append(gate.gate_id)
                    self._last_event = "miss"

        self._prev_position = drone_position
        return passed_gate

    def _check_pass_through(
        self,
        prev_pos: Tuple[float, float, float],
        curr_pos: Tuple[float, float, float],
        gate: Gate,
    ) -> bool:
        """Detect drone-crossed-gate-opening between prev_pos and curr_pos."""
        crossing = self._compute_crossing(prev_pos, curr_pos, gate)
        if crossing is None:
            return False
        return self._point_in_gate_opening(crossing, gate)

    def _compute_crossing(
        self,
        prev_pos: Tuple[float, float, float],
        curr_pos: Tuple[float, float, float],
        gate: Gate,
    ) -> Optional[np.ndarray]:
        gate_pos = np.array([gate.pose.x, gate.pose.y, gate.pose.z])
        normal = self._gate_normal(gate.pose)
        prev = np.array(prev_pos)
        curr = np.array(curr_pos)
        d_prev = np.dot(prev - gate_pos, normal)
        d_curr = np.dot(curr - gate_pos, normal)
        if d_prev * d_curr > 0:
            return None
        denom = d_curr - d_prev
        if abs(denom) < 1e-9:
            return None
        t = -d_prev / denom
        return prev + t * (curr - prev)

    def _point_in_outer_frame(self, point: np.ndarray, gate: Gate) -> bool:
        """True iff `point` lies inside the gate's outer (frame-included) bounds."""
        gate_pos = np.array([gate.pose.x, gate.pose.y, gate.pose.z])
        relative = point - gate_pos

        right = self._gate_right(gate.pose)
        # Same `up` derivation as _point_in_gate_opening.
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

        half_outer_w = (
            gate.config.interior_width_m / 2.0 + gate.config.border_width_m
        )
        half_outer_h = (
            gate.config.interior_height_m / 2.0 + gate.config.border_width_m
        )
        return (
            abs(local_right) < half_outer_w
            and abs(local_up) < half_outer_h
        )

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

        # Margin tolerates imprecise flight when classifying a pass.
        # Configured at construction; default 1.5 preserves prior behavior.
        return (
            abs(local_right) < half_w * self._pass_margin
            and abs(local_up) < half_h * self._pass_margin
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

    @property
    def crashed_gate_ids(self) -> List[str]:
        return [gid for gid, _ in self._crashes]

    @property
    def last_crash(self) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        return self._crashes[-1] if self._crashes else None

    @property
    def missed_gate_ids(self) -> List[str]:
        return list(self._misses)

    @property
    def last_event(self) -> Optional[str]:
        return self._last_event

    @property
    def state(self):
        """Lightweight RaceState shim — DynamicReplanner inspects state.name."""
        from types import SimpleNamespace
        if self.is_complete:
            return SimpleNamespace(name="COMPLETED")
        return SimpleNamespace(name="RACING")

    @property
    def gates_passed(self) -> int:
        return len(self._passed)

    def mark_collision(
        self,
        gate_id: str,
        position: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Record a physics-reported collision (e.g. PyBullet contact).

        Bypasses geometric heuristics — authoritative."""
        gate = next((g for g in self._gates if g.gate_id == gate_id), None)
        if gate is None:
            raise ValueError(f"Unknown gate_id: {gate_id!r}")
        if position is not None:
            pt = tuple(float(c) for c in position)
        elif self._prev_position is not None:
            pt = tuple(float(c) for c in self._prev_position)
        else:
            pt = (gate.pose.x, gate.pose.y, gate.pose.z)
        self._crashes.append((gate_id, pt))
        self._last_event = "crash"

    def reset(self):
        """Reset sequencer to the beginning."""
        self._current_idx = 0
        self._passed.clear()
        self._prev_position = None
        self._crashes.clear()
        self._misses.clear()
        self._last_event = None
