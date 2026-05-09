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
        # Set-of-IDs detection (P1-3): track which crash/miss IDs the
        # sequencer has reported so far. The list length is tracked
        # alongside so that a sequencer.reset() (which empties the lists)
        # is detected as a length-shrink and resynced — set-difference
        # alone would miss this when the post-reset crash happens to
        # repeat a previously-seen ID.
        self._seen_crash_ids: set = set()
        self._seen_miss_ids: set = set()
        self._n_crashes_seen: int = 0
        self._n_misses_seen: int = 0
        self._last_trigger: Optional[ReplanTrigger] = None
        # Edge-trigger latches for level signals. The sequencer's RECOVERY
        # state and the sustained-lateral-error condition each persist for
        # the duration of the perturbation; without these we'd report the
        # trigger field True every tick. See P1-1.
        self._was_off_track: bool = False
        self._was_sustained: bool = False

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
        self._seen_crash_ids = set()
        self._seen_miss_ids = set()
        self._n_crashes_seen = 0
        self._n_misses_seen = 0
        self._last_trigger = None
        self._was_off_track = False
        self._was_sustained = False

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
        # Set-of-IDs crash/miss detection. A length-shrink on the
        # sequencer's authoritative list signals a reset; resync our
        # baseline so a post-reset event with a previously-seen ID still
        # registers as new.
        crashed_list = sequencer.crashed_gate_ids
        missed_list = sequencer.missed_gate_ids
        if len(crashed_list) < self._n_crashes_seen:
            self._seen_crash_ids = set()
        if len(missed_list) < self._n_misses_seen:
            self._seen_miss_ids = set()
        self._n_crashes_seen = len(crashed_list)
        self._n_misses_seen = len(missed_list)

        crashed_set = set(crashed_list)
        missed_set = set(missed_list)
        new_crashes = crashed_set - self._seen_crash_ids
        new_misses = missed_set - self._seen_miss_ids

        gate_collision = bool(new_crashes)
        missed_gate = bool(new_misses)
        crashed_gate = (
            crashed_list[-1] if gate_collision else None
        )

        self._seen_crash_ids = crashed_set
        self._seen_miss_ids = missed_set

        # Sustained lateral error counter. Skip the update entirely when
        # lateral_error is non-finite — a NaN tick must neither increment
        # (false fire eventually) nor reset (silent disable).
        if math.isfinite(lateral_error):
            if lateral_error > self.config.lateral_error_threshold_m:
                self._consecutive_high_lateral += 1
            else:
                self._consecutive_high_lateral = 0
        level_sustained = (
            self._consecutive_high_lateral >= self.config.sustained_frames
        )

        # Off-track is whatever the sequencer says — keeps the policy in
        # one place. RaceState lives in the sequencer module; we compare
        # by string to avoid a hard import.
        state_name = getattr(sequencer.state, "name", str(sequencer.state))
        level_off_track = state_name == "RECOVERY"

        # Edge-trigger: the level signals stay True for the duration of the
        # perturbation, so report them only on the rising edge. Latches
        # self-update from level each tick — mark_replanned deliberately
        # leaves them alone so a still-True condition stays suppressed past
        # the cooldown window.
        off_track = level_off_track and not self._was_off_track
        sustained = level_sustained and not self._was_sustained
        self._was_off_track = level_off_track
        self._was_sustained = level_sustained

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
        # NaN/inf would slip through the cooldown check (NaN comparisons
        # are False per IEEE-754; inf - finite = inf > cooldown). Reject.
        if not math.isfinite(sim_time):
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
        # Refuse non-finite writes — a single NaN tick would otherwise
        # poison _last_replan_time and disable cooldown for the rest of
        # the race.
        if not math.isfinite(sim_time):
            return
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
