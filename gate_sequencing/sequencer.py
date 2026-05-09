"""
Platform-agnostic gate sequencer with recovery behaviors.

Refactored from sim_pybullet/sequencer.py to remove PyBullet/simulation
dependencies. Works with any position source and gate definition.

Adds recovery behaviors missing from the original:
  - Off-track detection and return-to-corridor
  - Missed gate handling (reattempt or skip)
  - Attitude excursion recovery
  - Detection dropout handling
  - Geometric crash-into-gate detection
  - External collision marking (e.g., PyBullet contact manifold)
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
    border_width: float = 0.15             # meters — frame thickness around the opening.
                                           # Used by geometric crash detection: a plane
                                           # crossing inside (interior + 2*border) but
                                           # outside the interior opening = hit the frame.
    sequence_index: int = 0

    @property
    def outer_width(self) -> float:
        return self.interior_width + 2.0 * self.border_width

    @property
    def outer_height(self) -> float:
        return self.interior_height + 2.0 * self.border_width


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
        # Crash + miss tracking. A crash is a plane crossing inside the outer
        # frame but outside the interior opening, OR an externally reported
        # collision via mark_collision(). A miss is a plane crossing outside
        # both the opening and the outer frame (drone flew completely around
        # the highlighted gate). Both are race-relevant signals for the
        # replanner upstream.
        self._crashes: List[Tuple[str, Tuple[float, float, float]]] = []
        self._misses: List[str] = []
        # The most recent terminal event for the current target gate.
        # Cleared when the target advances. One of: 'pass' | 'crash' | 'miss' | None.
        self._last_event: Optional[str] = None

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
        detection_active: bool = True,
    ) -> Optional[GateSpec]:
        """
        Check if the drone has passed through the current gate.

        Args:
            position: current drone position (NED)
            gate_detected: whether the current gate is being detected
                this tick. Only meaningful when ``detection_active`` is
                True; ignored otherwise.
            detection_active: whether perception is actually running
                this tick (camera frame available AND detector enabled).
                When False, the dropout counter is reset rather than
                incremented — a tick with no camera feed is not the
                same thing as a tick where the detector looked and saw
                nothing. Without this distinction, a competition run
                that has no vision stream yet (mavlink_bridge returns
                None for camera frames) would latch ``should_slow_down``
                permanently after ~0.3 s of nominal flight.

        Returns:
            The gate that was just passed, or None
        """
        if self._state in (RaceState.COMPLETED, RaceState.TIMED_OUT, RaceState.WAITING):
            return None

        pos = np.array(position)
        passed_gate = None

        # Track detection dropout: only accrue the counter when the
        # perception stack is actually running and failed to see the
        # current gate. "No camera feed" must not trigger recovery-grade
        # slowdown (the existing behaviour was a permanent 0.5×/0.7×
        # scale latch once detection_dropout_frames ticks elapsed).
        if not detection_active:
            self._frames_without_detection = 0
        elif gate_detected:
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
                self._last_event = "pass"

                if self.is_complete:
                    self._state = RaceState.COMPLETED
            else:
                # Pass-through *iff highlighted*: if the geometry shows the
                # drone crossed the highlighted gate's plane but missed the
                # opening, classify the event so upstream can react. We do
                # not advance the target — the gate stays highlighted until
                # the drone either passes it or skips ahead deliberately.
                if self._plane_was_crossed(self._prev_position, pos, gate):
                    crossing = self._compute_crossing(
                        self._prev_position, pos, gate
                    )
                    if crossing is not None:
                        if self._point_in_outer_frame(crossing, gate) \
                                and not self._point_in_gate_opening(crossing, gate):
                            self._crashes.append(
                                (gate.gate_id, tuple(float(c) for c in crossing))
                            )
                            self._last_event = "crash"
                        else:
                            self._misses.append(gate.gate_id)
                            self._last_event = "miss"

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

    @property
    def crashed_gate_ids(self) -> List[str]:
        """All gate IDs the drone has hit so far (frame collisions)."""
        return [gid for gid, _ in self._crashes]

    @property
    def last_crash(self) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        """(gate_id, crossing_point) of the most recent crash, or None."""
        return self._crashes[-1] if self._crashes else None

    @property
    def missed_gate_ids(self) -> List[str]:
        """Gate IDs whose plane was crossed completely outside the frame."""
        return list(self._misses)

    @property
    def last_event(self) -> Optional[str]:
        """One of 'pass' | 'crash' | 'miss' | None — for the current target."""
        return self._last_event

    @property
    def passed_gate_ids(self) -> List[str]:
        """Gate IDs the drone has passed through, in order."""
        return list(self._passed)

    def mark_collision(
        self,
        gate_id: str,
        position: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Record an externally observed collision (e.g. PyBullet contact).

        Use this when a physics layer reports the drone touched a gate body —
        it bypasses the geometric heuristic and is authoritative. Position
        defaults to the last known drone position; if none is available we
        fall back to the gate centre.
        """
        gate = next((g for g in self._gates if g.gate_id == gate_id), None)
        if gate is None:
            raise ValueError(f"Unknown gate_id: {gate_id!r}")
        if position is not None:
            pt = tuple(float(c) for c in position)
        elif self._prev_position is not None:
            pt = tuple(float(c) for c in self._prev_position)
        else:
            pt = tuple(float(c) for c in gate.position)
        self._crashes.append((gate_id, pt))
        self._last_event = "crash"

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
        crossing = self._compute_crossing(prev_pos, curr_pos, gate)
        if crossing is None:
            return False
        return self._point_in_gate_opening(crossing, gate)

    def _plane_was_crossed(
        self,
        prev_pos: np.ndarray,
        curr_pos: np.ndarray,
        gate: GateSpec,
    ) -> bool:
        gate_pos = np.array(gate.position)
        normal = self._gate_normal(gate)
        d_prev = float(np.dot(prev_pos - gate_pos, normal))
        d_curr = float(np.dot(curr_pos - gate_pos, normal))
        return d_prev * d_curr <= 0 and abs(d_curr - d_prev) >= 1e-9

    def _compute_crossing(
        self,
        prev_pos: np.ndarray,
        curr_pos: np.ndarray,
        gate: GateSpec,
    ) -> Optional[np.ndarray]:
        gate_pos = np.array(gate.position)
        normal = self._gate_normal(gate)
        d_prev = float(np.dot(prev_pos - gate_pos, normal))
        d_curr = float(np.dot(curr_pos - gate_pos, normal))
        if d_prev * d_curr > 0:
            return None
        denom = d_curr - d_prev
        if abs(denom) < 1e-9:
            return None
        t = -d_prev / denom
        return prev_pos + t * (curr_pos - prev_pos)

    def _point_in_outer_frame(
        self, point: np.ndarray, gate: GateSpec
    ) -> bool:
        """True iff `point` is inside the gate's outer frame bounds.

        The outer frame is `(interior + 2*border)` × `(interior + 2*border)`
        in the gate's local right/up plane. Combined with the opening check,
        a point inside the outer frame but outside the opening = hit the
        actual gate frame (crash).
        """
        gate_pos = np.array(gate.position)
        relative = point - gate_pos
        right = self._gate_right(gate)
        up = self._gate_up(gate)
        local_right = float(np.dot(relative, right))
        local_up = float(np.dot(relative, up))
        half_outer_w = gate.outer_width / 2.0
        half_outer_h = gate.outer_height / 2.0
        return (
            abs(local_right) < half_outer_w
            and abs(local_up) < half_outer_h
        )

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
        cp, sp = math.cos(gate.pitch), math.sin(gate.pitch)
        cr, sr = math.cos(gate.roll), math.sin(gate.roll)

        normal = np.array([cy * cp, sy * cp, sp])
        right0 = np.array([-sy, cy, 0.0])
        down0 = np.cross(normal, right0)
        down_norm = np.linalg.norm(down0)
        if down_norm > 1e-12:
            down0 = down0 / down_norm

        return right0 * cr + down0 * sr

    @staticmethod
    def _gate_up(gate: GateSpec) -> np.ndarray:
        cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
        cp, sp = math.cos(gate.pitch), math.sin(gate.pitch)
        cr, sr = math.cos(gate.roll), math.sin(gate.roll)

        normal = np.array([cy * cp, sy * cp, sp])
        right0 = np.array([-sy, cy, 0.0])
        down0 = np.cross(normal, right0)
        down_norm = np.linalg.norm(down0)
        if down_norm > 1e-12:
            down0 = down0 / down_norm

        return -right0 * sr + down0 * cr

    def reset(self) -> None:
        self._current_idx = 0
        self._passed.clear()
        self._prev_position = None
        self._state = RaceState.WAITING
        self._frames_without_detection = 0
        self._recovery_target = None
        self._crashes.clear()
        self._misses.clear()
        self._last_event = None
