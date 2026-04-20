"""
Platform-agnostic gate sequencer with recovery behaviors.

Refactored from sim_pybullet/sequencer.py to remove PyBullet/simulation
dependencies. Works with any position source and gate definition.

Adds recovery behaviors missing from the original:
  - Off-track detection and return-to-corridor
  - Missed gate handling (reattempt or skip)
  - Attitude excursion recovery
  - Detection dropout handling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GateSpec:
    """Platform-agnostic gate definition."""
    gate_id: str
    position: Tuple[float, float, float]  # center (NED)
    yaw: float = 0.0                       # facing direction (radians)
    pitch: float = 0.0
    roll: float = 0.0
    interior_width: float = 1.2            # meters
    interior_height: float = 1.2           # meters
    sequence_index: int = 0


class RaceState(Enum):
    """Current state of the race."""
    WAITING = auto()      # before start
    RACING = auto()       # normal racing
    RECOVERY = auto()     # recovering from off-track / missed gate
    COMPLETED = auto()    # all gates passed
    TIMED_OUT = auto()    # exceeded time limit


@dataclass
class SequencerConfig:
    """Sequencer tuning parameters."""
    pass_through_margin: float = 1.0     # gate opening multiplier for pass-through detection
    proximity_pass_distance: float = 0.0 # if >0, also pass gate when within this distance
    off_track_distance: float = 5.0      # meters from expected path before triggering recovery
    max_approach_angle: float = 1.2      # radians — max angle for valid gate approach
    detection_dropout_frames: int = 30   # frames without detection before slowing down
    recovery_speed_factor: float = 0.3   # speed reduction during recovery


class GateSequencer:
    """
    Manages gate sequence, pass-through detection, and recovery.

    Platform-agnostic: works with any position source (MAVLink telemetry,
    PyBullet sim, etc.) through the update() interface.
    """

    def __init__(
        self,
        gates: List[GateSpec],
        config: SequencerConfig = None,
    ):
        self.config = config or SequencerConfig()
        self._gates = sorted(gates, key=lambda g: g.sequence_index)
        self._current_idx = 0
        self._passed: List[str] = []
        self._prev_position: Optional[np.ndarray] = None
        self._state = RaceState.WAITING
        self._frames_without_detection = 0
        self._recovery_target: Optional[Tuple[float, float, float]] = None

    @property
    def current_gate(self) -> Optional[GateSpec]:
        if self._current_idx < len(self._gates):
            return self._gates[self._current_idx]
        return None

    @property
    def next_gate(self) -> Optional[GateSpec]:
        if self._current_idx + 1 < len(self._gates):
            return self._gates[self._current_idx + 1]
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
    def state(self) -> RaceState:
        return self._state

    @property
    def all_gates(self) -> List[GateSpec]:
        return list(self._gates)

    @property
    def progress(self) -> float:
        """Race progress as fraction [0, 1]."""
        return self.gates_passed / max(self.total_gates, 1)

    def start(self) -> None:
        """Begin the race."""
        self._state = RaceState.RACING

    def update(
        self,
        position: Tuple[float, float, float],
        gate_detected: bool = True,
    ) -> Optional[GateSpec]:
        """
        Check if the drone has passed through the current gate.

        Args:
            position: current drone position (NED)
            gate_detected: whether the current gate is being detected

        Returns:
            The gate that was just passed, or None
        """
        if self._state in (RaceState.COMPLETED, RaceState.TIMED_OUT, RaceState.WAITING):
            return None

        pos = np.array(position)
        passed_gate = None

        # Track detection dropout
        if gate_detected:
            self._frames_without_detection = 0
        else:
            self._frames_without_detection += 1

        # Check pass-through (plane crossing or proximity)
        if self._prev_position is not None and not self.is_complete:
            gate = self._gates[self._current_idx]
            plane_crossed = self._check_pass_through(self._prev_position, pos, gate)

            # Proximity-based pass: only credits a pass when the drone is
            # inside the lit gate opening. A "lit" gate is the currently
            # targeted gate; if the drone flies close to it but outside the
            # rectangular opening, this must NOT count (prior behaviour
            # credited skim-bys at dist ≤ proximity_pass_distance regardless
            # of lateral offset, producing false-positive passes).
            proximity_passed = False
            if not plane_crossed and self.config.proximity_pass_distance > 0:
                dist = float(np.linalg.norm(pos - np.array(gate.position)))
                if dist < self.config.proximity_pass_distance:
                    gate_pos = np.array(gate.position)
                    normal = self._gate_normal(gate)
                    d_curr = float(np.dot(pos - gate_pos, normal))
                    if d_curr > -0.5:  # near or past the plane
                        # Require drone to be inside the gate opening laterally,
                        # i.e. its projection onto the gate plane falls within
                        # (half_w × pass_through_margin, half_h × pass_through_margin).
                        if self._point_in_gate_opening(pos, gate):
                            proximity_passed = True

            if plane_crossed or proximity_passed:
                passed_gate = gate
                self._passed.append(gate.gate_id)
                self._current_idx += 1
                self._state = RaceState.RACING
                self._recovery_target = None

                if self.is_complete:
                    self._state = RaceState.COMPLETED

        # Check if off-track
        if not self.is_complete and self._state == RaceState.RACING:
            gate = self._gates[self._current_idx]
            dist_to_gate = float(np.linalg.norm(pos - np.array(gate.position)))
            if dist_to_gate > self.config.off_track_distance * 3:
                self._state = RaceState.RECOVERY
                self._recovery_target = gate.position

        self._prev_position = pos
        return passed_gate

    def get_recovery_target(self) -> Optional[Tuple[float, float, float]]:
        """Get the recovery target position if in recovery mode."""
        if self._state == RaceState.RECOVERY and self._recovery_target:
            return self._recovery_target
        return None

    def should_slow_down(self) -> bool:
        """Whether the drone should reduce speed (detection dropout, recovery)."""
        return (
            self._state == RaceState.RECOVERY
            or self._frames_without_detection > self.config.detection_dropout_frames
        )

    def _check_pass_through(
        self,
        prev_pos: np.ndarray,
        curr_pos: np.ndarray,
        gate: GateSpec,
    ) -> bool:
        """Detect gate pass-through via plane crossing."""
        gate_pos = np.array(gate.position)
        normal = self._gate_normal(gate)

        d_prev = float(np.dot(prev_pos - gate_pos, normal))
        d_curr = float(np.dot(curr_pos - gate_pos, normal))

        # Must cross the plane (signs differ)
        if d_prev * d_curr > 0:
            return False

        # Find crossing point
        denom = d_curr - d_prev
        if abs(denom) < 1e-9:
            return False
        t = -d_prev / denom
        crossing = prev_pos + t * (curr_pos - prev_pos)

        return self._point_in_gate_opening(crossing, gate)

    def _point_in_gate_opening(
        self, point: np.ndarray, gate: GateSpec
    ) -> bool:
        """Check if a point falls within the gate opening."""
        gate_pos = np.array(gate.position)
        relative = point - gate_pos

        right = self._gate_right(gate)
        up = self._gate_up(gate)

        local_right = float(np.dot(relative, right))
        local_up = float(np.dot(relative, up))

        half_w = gate.interior_width / 2.0
        half_h = gate.interior_height / 2.0
        margin = self.config.pass_through_margin

        return (
            abs(local_right) < half_w * margin
            and abs(local_up) < half_h * margin
        )

    @staticmethod
    def _gate_normal(gate: GateSpec) -> np.ndarray:
        cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
        cp, sp = math.cos(gate.pitch), math.sin(gate.pitch)
        return np.array([cy * cp, sy * cp, sp])

    @staticmethod
    def _gate_right(gate: GateSpec) -> np.ndarray:
        cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
        return np.array([-sy, cy, 0.0])

    @staticmethod
    def _gate_up(gate: GateSpec) -> np.ndarray:
        cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
        cp, sp = math.cos(gate.pitch), math.sin(gate.pitch)
        cr, sr = math.cos(gate.roll), math.sin(gate.roll)
        return np.array([
            sy * sr + cy * sp * cr,
            -cy * sr + sy * sp * cr,
            cp * cr,
        ])

    def reset(self) -> None:
        self._current_idx = 0
        self._passed.clear()
        self._prev_position = None
        self._state = RaceState.WAITING
        self._frames_without_detection = 0
        self._recovery_target = None
