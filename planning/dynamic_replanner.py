"""
Dynamic race-line replanner.

Owns the decision of *when* to rebuild the racing line during a live race
and the construction of the new waypoint list from the current drone
state. Pure logic — no PyBullet, no RacingLine ownership. The caller
holds whichever spline/poly representation it uses and rebuilds on a
positive trigger.

Triggers (any one fires a replan, subject to cooldown):

  - **gate_collision**: sequencer has a fresh crash on the highlighted
    gate (geometric crash detection or external mark_collision()).
  - **missed_gate**: drone crossed the highlighted gate's plane outside
    the opening — fly-around. Sequencer reports this as `last_event ==
    'miss'`.
  - **off_track**: sequencer is in `RECOVERY` state.
  - **sustained_lateral_error**: drone has stayed `> threshold` metres
    from the racing line for `sustained_frames` consecutive ticks. This
    catches gradual drift that the crash/miss heuristics never trip.

Cooldown (`cooldown_seconds`) prevents a single perturbation from
producing a replanning storm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple


@dataclass
class ReplanConfig:
    cooldown_seconds: float = 0.5
    lateral_error_threshold_m: float = 2.0
    sustained_frames: int = 30


@dataclass
class ReplanTrigger:
    gate_collision: bool = False
    missed_gate: bool = False
    off_track: bool = False
    sustained_lateral_error: bool = False
    crashed_gate_id: Optional[str] = None

    @property
    def triggered(self) -> bool:
        return (
            self.gate_collision
            or self.missed_gate
            or self.off_track
            or self.sustained_lateral_error
        )

    @property
    def reasons(self) -> List[str]:
        out = []
        if self.gate_collision:
            out.append("gate_collision")
        if self.missed_gate:
            out.append("missed_gate")
        if self.off_track:
            out.append("off_track")
        if self.sustained_lateral_error:
            out.append("sustained_lateral_error")
        return out


class _SequencerLike(Protocol):
    """Just the interface the replanner needs from a sequencer."""
    @property
    def state(self): ...
    @property
    def current_gate(self): ...
    @property
    def all_gates(self) -> list: ...
    @property
    def gates_passed(self) -> int: ...
    @property
    def crashed_gate_ids(self) -> List[str]: ...
    @property
    def missed_gate_ids(self) -> List[str]: ...
    @property
    def last_event(self) -> Optional[str]: ...


class DynamicReplanner:
    """Stateful replanner. One instance per race; reset() between runs."""

    def __init__(self, config: Optional[ReplanConfig] = None):
        self.config = config or ReplanConfig()
        self._last_replan_time: float = -math.inf
        self._consecutive_high_lateral: int = 0
        self._replan_count: int = 0
        self._last_seen_crashes: int = 0
        self._last_seen_misses: int = 0
        self._last_trigger: Optional[ReplanTrigger] = None

    @property
    def replan_count(self) -> int:
        return self._replan_count

    @property
    def last_trigger(self) -> Optional[ReplanTrigger]:
        return self._last_trigger

    def reset(self) -> None:
        self._last_replan_time = -math.inf
        self._consecutive_high_lateral = 0
        self._replan_count = 0
        self._last_seen_crashes = 0
        self._last_seen_misses = 0
        self._last_trigger = None

    def evaluate(
        self,
        sim_time: float,
        sequencer: _SequencerLike,
        lateral_error: float,
    ) -> ReplanTrigger:
        """Inspect the current race state and return a trigger record.

        The trigger is computed each tick whether or not the cooldown
        allows a replan — `should_replan()` is the one that gates on
        cooldown. This split is deliberate: callers may want to log
        triggers even when not acting on them.
        """
        n_crashes = len(sequencer.crashed_gate_ids)
        n_misses = len(sequencer.missed_gate_ids)
        crashed_gate = (
            sequencer.crashed_gate_ids[-1]
            if n_crashes > self._last_seen_crashes
            else None
        )

        gate_collision = n_crashes > self._last_seen_crashes
        missed_gate = n_misses > self._last_seen_misses

        # Sustained lateral error counter
        if lateral_error > self.config.lateral_error_threshold_m:
            self._consecutive_high_lateral += 1
        else:
            self._consecutive_high_lateral = 0
        sustained = (
            self._consecutive_high_lateral >= self.config.sustained_frames
        )

        # Off-track is whatever the sequencer says — keeps the policy in
        # one place. RaceState lives in the sequencer module; we compare
        # by string to avoid a hard import.
        state_name = getattr(sequencer.state, "name", str(sequencer.state))
        off_track = state_name == "RECOVERY"

        self._last_seen_crashes = n_crashes
        self._last_seen_misses = n_misses

        trigger = ReplanTrigger(
            gate_collision=gate_collision,
            missed_gate=missed_gate,
            off_track=off_track,
            sustained_lateral_error=sustained,
            crashed_gate_id=crashed_gate,
        )
        return trigger

    def should_replan(self, trigger: ReplanTrigger, sim_time: float) -> bool:
        if not trigger.triggered:
            return False
        if sim_time - self._last_replan_time < self.config.cooldown_seconds:
            return False
        return True

    def waypoints_for_replan(
        self,
        drone_position: Tuple[float, float, float],
        sequencer: _SequencerLike,
    ) -> List[Tuple[float, float, float]]:
        """Build the new racing-line waypoint list from current state.

        Always starts at the drone's current position. Then enumerates
        the remaining gate centres in sequence order — i.e. the current
        target gate first, followed by every gate after it that hasn't
        been passed.

        Accepts both the platform-agnostic GateSpec (with ``.position``)
        and the sim_pybullet Gate (with ``.pose.x/y/z``) shape.
        """
        wps: List[Tuple[float, float, float]] = [
            tuple(float(c) for c in drone_position)
        ]
        all_gates = sequencer.all_gates
        start_idx = sequencer.gates_passed
        for g in all_gates[start_idx:]:
            wps.append(_gate_centre(g))
        return wps

    def mark_replanned(self, sim_time: float, trigger: ReplanTrigger) -> None:
        self._last_replan_time = sim_time
        self._replan_count += 1
        self._last_trigger = trigger
        # On replan, reset sustained counter — the new line is now the
        # reference, lateral error against the OLD line is irrelevant.
        self._consecutive_high_lateral = 0


def _gate_centre(gate) -> Tuple[float, float, float]:
    """Extract the (x, y, z) centre of a gate, accepting either GateSpec
    (gate_sequencing) or Gate (sim_pybullet) shape."""
    pos = getattr(gate, "position", None)
    if pos is not None:
        return tuple(float(c) for c in pos)
    pose = getattr(gate, "pose", None)
    if pose is not None:
        return (float(pose.x), float(pose.y), float(pose.z))
    raise TypeError(
        f"Gate object {gate!r} has neither 'position' nor 'pose' attribute"
    )
