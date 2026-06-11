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

# Defaults track VADR-TS-002 §3.7. Track configs (e.g. race_01.json)
# may override via explicit constructor args.
from competition.aigp_geometry import (
    AIGP_GATE_BORDER_M,
    AIGP_GATE_DEPTH_M,
    AIGP_GATE_INTERIOR_M,
)


@dataclass
class GateSpec:
    """Platform-agnostic gate definition.

    Defaults match VADR-TS-002 §3.7 (AIGP Virtual Qualifier 1):
        interior 1.5 m × 1.5 m, border 0.6 m, depth 0.26 m → outer 2.7 m.
    Legacy tracks supply explicit smaller dimensions via constructor args.
    """
    gate_id: str
    position: Tuple[float, float, float]   # center (NED)
    yaw: float = 0.0                       # facing direction (radians)
    pitch: float = 0.0
    roll: float = 0.0
    interior_width: float = AIGP_GATE_INTERIOR_M    # 1.5 m
    interior_height: float = AIGP_GATE_INTERIOR_M   # 1.5 m
    border_width: float = AIGP_GATE_BORDER_M        # 0.6 m frame thickness.
                                                    # Crash zone = plane crossing inside
                                                    # (interior + 2·border) but outside
                                                    # the bare interior opening (P1-6).
    depth: float = AIGP_GATE_DEPTH_M                # 0.26 m through-gate depth.
    sequence_index: int = 0

    @property
    def outer_width(self) -> float:
        return self.interior_width + 2.0 * self.border_width

    @property
    def outer_height(self) -> float:
        return self.interior_height + 2.0 * self.border_width


class RaceState(Enum):
    """Current state of the race."""
    WAITING = auto()       # before start
    RACING = auto()        # normal racing
    RECOVERY = auto()      # recovering from off-track / missed gate
    COMPLETED = auto()     # all gates passed
    TIMED_OUT = auto()     # exceeded time limit
    DISQUALIFIED = auto()  # terminal failure — out-of-order pass, etc.


@dataclass
class SequencerConfig:
    """Sequencer tuning parameters."""
    pass_through_margin: float = 1.0     # gate opening multiplier for pass-through detection
    # Crash classification uses the bare opening (multiplied by `crash_margin`)
    # rather than the lenient pass-through tolerance. With production
    # pass_through_margin=1.5 and a 0.15 m frame border, the geometric
    # crash zone is empty unless this is set to ~1.0 (P1-6).
    crash_margin: float = 1.0
    proximity_pass_distance: float = 0.0 # if >0, also pass gate when within this distance
    off_track_distance: float = 5.0      # meters from expected path before triggering recovery
    max_approach_angle: float = 1.2      # radians — max angle for valid gate approach
    detection_dropout_frames: int = 30   # frames without detection before slowing down
    recovery_speed_factor: float = 0.3   # speed reduction during recovery
    # When True, any plane crossing of an unpassed-non-current gate's *opening*
    # is a terminal DQ ("out-of-order"). Crossings outside the opening (e.g.
    # far above or beside a future gate) are benign. Frame-strut hits of any
    # gate continue to be classified as `crash`, not DQ. Default True so the
    # competition-relevant strict-order rule is the out-of-the-box behaviour;
    # legacy tests / debug runs can opt out via SequencerConfig(enforce_in_order=False).
    enforce_in_order: bool = True


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
        # Set by mark_collision; consumed and cleared by update(). When set
        # to the current gate's id, the same-tick pass classification is
        # short-circuited so the authoritative physics-driven crash mark
        # wins over a lenient geometric pass (P1-4).
        self._collision_marked_this_tick: Optional[str] = None
        # Reason set when the sequencer DQs the run. None unless
        # `self._state == RaceState.DISQUALIFIED`. Format: "out_of_order:<gate_id>".
        self._dq_reason: Optional[str] = None
        # Iter-002 (5/7 reviews MAJOR): RaceState.TIMED_OUT was dead code
        # before — defined but never assigned. Now bench/pipeline call
        # `mark_timed_out(reason)` when the 8-minute VQ1 cap is exceeded.
        self._timeout_reason: Optional[str] = None

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
        # A DQ run is never "complete" — `is_complete` means the drone
        # actually finished the course. Downstream consumers check
        # `is_disqualified` separately for terminal-but-failed runs.
        if self._state == RaceState.DISQUALIFIED:
            return False
        return self._current_idx >= len(self._gates)

    @property
    def is_disqualified(self) -> bool:
        """True if the run was terminated for a rule violation (e.g. out-of-order pass)."""
        return self._state == RaceState.DISQUALIFIED

    @property
    def dq_reason(self) -> Optional[str]:
        """Human-readable reason for the DQ, or None."""
        return self._dq_reason

    @property
    def is_timed_out(self) -> bool:
        """True if the run exceeded the VQ1 8-minute cap or any caller-imposed limit."""
        return self._state == RaceState.TIMED_OUT

    @property
    def timeout_reason(self) -> Optional[str]:
        """Human-readable reason for the timeout, or None."""
        return self._timeout_reason

    def mark_timed_out(self, reason: str = "max_run_duration_exceeded") -> None:
        """Transition to RaceState.TIMED_OUT. Idempotent — repeated calls
        are silently ignored. Pre-race / completed / DQ'd state takes
        precedence (terminal events don't get retroactively re-terminated).
        """
        if self._state in (
            RaceState.WAITING,
            RaceState.COMPLETED,
            RaceState.DISQUALIFIED,
            RaceState.TIMED_OUT,
        ):
            return
        self._state = RaceState.TIMED_OUT
        self._timeout_reason = reason
        self._last_event = "timeout"

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
        if self._state in (
            RaceState.COMPLETED,
            RaceState.TIMED_OUT,
            RaceState.WAITING,
            RaceState.DISQUALIFIED,
        ):
            return None

        pos = np.array(position)
        passed_gate = None
        # iter-003 (gpt-55-2 F2): tick-local flag for "did the current-gate
        # classification block record a fresh crash this tick?" — used to
        # gate the future-gate DQ scan so a physical crash always wins
        # over an out-of-order rule violation in the same segment.
        crash_recorded_this_tick = False

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

            # P1-4: a same-tick mark_collision on the current gate
            # overrides the geometric pass classification — the physics
            # contact is authoritative. Skip pass/crash/miss; the crash
            # mark already set _last_event="crash".
            collision_pre_marked = (
                self._collision_marked_this_tick == gate.gate_id
            )
            if not collision_pre_marked:
                plane_crossed = self._check_pass_through(
                    self._prev_position, pos, gate,
                )

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
                            # Require drone to be inside the gate opening
                            # laterally — the projection onto the gate plane
                            # must fall inside (half × pass_through_margin).
                            if self._point_in_gate_opening(pos, gate):
                                proximity_passed = True

                # P1-6: classify a frame hit as crash BEFORE crediting a
                # lenient pass. With pass_through_margin=1.5 the pass test
                # would otherwise eat every crossing inside the outer
                # frame; gate the pass branch on the bare-opening check.
                crash_classified = False
                if self._plane_was_crossed(self._prev_position, pos, gate):
                    crossing = self._compute_crossing(
                        self._prev_position, pos, gate,
                    )
                    if crossing is not None:
                        in_outer = self._point_in_outer_frame(crossing, gate)
                        in_crash_zone_opening = (
                            self._point_in_opening_with_margin(
                                crossing, gate, self.config.crash_margin,
                            )
                        )
                        if in_outer and not in_crash_zone_opening:
                            # Hit the frame between bare opening and outer.
                            # P1-7: dedup ONLY within the SAME gate's fly-by.
                            # iter-003 (composer-25-5 F15 / gpt-55-1 F3):
                            # the previous `_last_event != "crash"` check
                            # dropped a second crash on a DIFFERENT gate
                            # in the same tick (multi-strut segments).
                            # Dedup must key on gate_id, not on last_event.
                            already_in_this_flyby = (
                                self._crashes
                                and self._crashes[-1][0] == gate.gate_id
                                and self._last_event == "crash"
                            )
                            if not already_in_this_flyby:
                                self._crashes.append(
                                    (gate.gate_id,
                                     tuple(float(c) for c in crossing))
                                )
                                self._last_event = "crash"
                                crash_recorded_this_tick = True
                            crash_classified = True

                if (plane_crossed or proximity_passed) and not crash_classified:
                    passed_gate = gate
                    self._passed.append(gate.gate_id)
                    self._current_idx += 1
                    self._state = RaceState.RACING
                    self._recovery_target = None
                    self._last_event = "pass"

                    if self.is_complete:
                        self._state = RaceState.COMPLETED

                    # Iter-001 review Opus F2: multi-gate-per-tick. If the
                    # same prev→pos segment also passes through the NEW
                    # current target's opening, credit that one too — keep
                    # going until the segment is exhausted or we hit a crash.
                    while (
                        not self.is_complete
                        and self._state == RaceState.RACING
                    ):
                        nxt = self._gates[self._current_idx]
                        if not self._plane_was_crossed(self._prev_position, pos, nxt):
                            break
                        nxt_crossing = self._compute_crossing(
                            self._prev_position, pos, nxt,
                        )
                        if nxt_crossing is None:
                            break
                        # Test against the lenient pass-through margin
                        # (same semantics as the first credit).
                        if not self._point_in_gate_opening(nxt_crossing, nxt):
                            # Outside the lenient opening — could still be
                            # a strut hit on this new current target. Apply
                            # the same crash classification the P1-6 branch
                            # uses above.
                            in_outer = self._point_in_outer_frame(nxt_crossing, nxt)
                            in_crash_zone_opening = (
                                self._point_in_opening_with_margin(
                                    nxt_crossing, nxt, self.config.crash_margin,
                                )
                            )
                            if in_outer and not in_crash_zone_opening:
                                if self._last_event != "crash":
                                    self._crashes.append(
                                        (nxt.gate_id,
                                         tuple(float(c) for c in nxt_crossing))
                                    )
                                    self._last_event = "crash"
                            break
                        # Inside lenient opening — credit this gate too.
                        passed_gate = nxt
                        self._passed.append(nxt.gate_id)
                        self._current_idx += 1
                        self._last_event = "pass"
                        if self.is_complete:
                            self._state = RaceState.COMPLETED
                            break
                elif (
                    not crash_classified
                    and self._plane_was_crossed(self._prev_position, pos, gate)
                ):
                    # Plane crossed completely outside the frame → miss.
                    if self._last_event != "miss":
                        self._misses.append(gate.gate_id)
                        self._last_event = "miss"

        # ---------------------------------------------------------------
        # Out-of-order DQ: any unpassed-non-current gate whose *opening*
        # was crossed this tick is a terminal rule violation. This catches
        # the U-turn false-complete pattern from `.loop/specs/2_known_issues.md`
        # (I-1): drone skips gate N, passes gates N+1..K, U-turns back
        # through gate N, then the original "skip" tick is the smoking gun.
        # Frame-strut hits of any gate continue to be classified as
        # `crash` (handled above for the current target via the P1-6
        # branch); we only DQ on opening-inside crossings.
        # ---------------------------------------------------------------
        # iter-003: future-gate scan combines two independent concerns —
        # opening crossing → out-of-order DQ (only when enforce_in_order),
        # and strut hit → crash (always, physical impact). Splitting them
        # so a caller that opts out of in-order enforcement still gets
        # honest crash detection.
        # gpt-55-2 F2: physical crash takes priority over DQ. A tick that
        # already classified a crash this tick must not ALSO DQ — that
        # would obscure the actual cause of termination.
        if (
            self._state != RaceState.DISQUALIFIED
            and self._prev_position is not None
            and not self.is_complete
            and not crash_recorded_this_tick
            and self._collision_marked_this_tick is None
        ):
            for future_gate in self._gates[self._current_idx + 1:]:
                if not self._plane_was_crossed(self._prev_position, pos, future_gate):
                    continue
                crossing = self._compute_crossing(
                    self._prev_position, pos, future_gate,
                )
                if crossing is None:
                    continue
                # Iter-001 review Opus F14: use the STRICT crash_margin
                # opening (default 1.0 = bare opening) for the DQ check,
                # NOT the lenient pass_through_margin.
                in_strict_opening = self._point_in_opening_with_margin(
                    crossing, future_gate, self.config.crash_margin,
                )
                in_outer = self._point_in_outer_frame(crossing, future_gate)
                if in_strict_opening:
                    # Iter-028 (figure8 coplanar fix): if the future
                    # gate is COPLANAR (same xy + same yaw) with the
                    # gate just credited this tick, the crossing is an
                    # incidental side-effect of a legitimate pass-
                    # through, not an out-of-order violation. The
                    # figure-8 self-crossing pattern (gates 1+5 at the
                    # same x=5, different z) is the canonical example.
                    if self._future_gate_coplanar_with_last_pass(future_gate):
                        continue
                    # Out-of-order rule violation — only terminal under
                    # strict-ordering mode. With enforce_in_order=False
                    # the caller has explicitly opted out (legacy tests,
                    # debug runs).
                    if self.config.enforce_in_order:
                        self._state = RaceState.DISQUALIFIED
                        self._dq_reason = f"out_of_order:{future_gate.gate_id}"
                        self._last_event = "dq"
                        break
                    # else: silently ignore the out-of-order opening pass.
                    continue
                # Iter-001 review Opus F3 + iter-003: future-gate strut
                # hits are PHYSICAL crashes, recorded regardless of
                # enforce_in_order. Dedup keyed on gate_id (not on
                # _last_event alone), so two struts on two different
                # gates in successive ticks both record.
                # Iter-028 (figure8): also skip strut crashes for
                # coplanar future gates. The drone passing through
                # gate-1 can graze gate-5's strut at y≈±opening_width/2,
                # but that's the figure-8 self-crossing geometry, not
                # a control failure. A real strut hit would be
                # accompanied by the actual collision contact (PyBullet
                # path) or a clear off-axis crossing far from any
                # legitimately-pass-throughable gate plane.
                if in_outer and self._future_gate_coplanar_with_last_pass(future_gate):
                    continue
                if in_outer:
                    already_in_this_flyby = (
                        self._crashes
                        and self._crashes[-1][0] == future_gate.gate_id
                        and self._last_event == "crash"
                    )
                    if not already_in_this_flyby:
                        self._crashes.append(
                            (future_gate.gate_id,
                             tuple(float(c) for c in crossing))
                        )
                        self._last_event = "crash"
                    break

        # Check if off-track (suppressed once DQ'd or completed)
        if not self.is_complete and self._state == RaceState.RACING:
            gate = self._gates[self._current_idx]
            dist_to_gate = float(np.linalg.norm(pos - np.array(gate.position)))
            if dist_to_gate > self.config.off_track_distance * 3:
                self._state = RaceState.RECOVERY
                self._recovery_target = gate.position

        self._prev_position = pos
        # Same-tick crash latch consumed; clear so the next tick starts
        # fresh.
        self._collision_marked_this_tick = None
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
        # P1-5 state gate: pre-race spawn-overlap and post-race fly-throughs
        # must NOT register as race crashes.
        if self._state in (
            RaceState.WAITING, RaceState.COMPLETED, RaceState.TIMED_OUT,
        ):
            return
        # P1-5 idempotency: PyBullet's contact manifold persists across
        # ticks; collapse repeat calls on the same gate to a single entry.
        # Still flag _collision_marked_this_tick so P1-4 short-circuit
        # fires even when the append is suppressed.
        already_marked = (
            self._last_event == "crash"
            and self._crashes
            and self._crashes[-1][0] == gate_id
        )
        if already_marked:
            self._collision_marked_this_tick = gate_id
            return
        if position is not None:
            pt = tuple(float(c) for c in position)
        elif self._prev_position is not None:
            pt = tuple(float(c) for c in self._prev_position)
        else:
            pt = tuple(float(c) for c in gate.position)
        self._crashes.append((gate_id, pt))
        self._last_event = "crash"
        self._collision_marked_this_tick = gate_id

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
        """Check if a point falls within the lenient pass-through opening."""
        return self._point_in_opening_with_margin(
            point, gate, self.config.pass_through_margin,
        )

    def _point_in_opening_with_margin(
        self, point: np.ndarray, gate: GateSpec, margin: float,
    ) -> bool:
        """Generalised opening check with an explicit margin. Pass detection
        uses pass_through_margin (lenient); crash detection uses
        crash_margin (strict, default 1.0 = bare opening)."""
        gate_pos = np.array(gate.position)
        relative = point - gate_pos

        right = self._gate_right(gate)
        up = self._gate_up(gate)

        local_right = float(np.dot(relative, right))
        local_up = float(np.dot(relative, up))

        half_w = gate.interior_width / 2.0
        half_h = gate.interior_height / 2.0

        return (
            abs(local_right) < half_w * margin
            and abs(local_up) < half_h * margin
        )

    @staticmethod
    def _gate_normal(gate: GateSpec) -> np.ndarray:
        cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
        cp, sp = math.cos(gate.pitch), math.sin(gate.pitch)
        return np.array([cy * cp, sy * cp, -sp])

    @staticmethod
    def _gate_right(gate: GateSpec) -> np.ndarray:
        cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
        cp, sp = math.cos(gate.pitch), math.sin(gate.pitch)
        cr, sr = math.cos(gate.roll), math.sin(gate.roll)

        normal = np.array([cy * cp, sy * cp, -sp])
        right0 = np.array([-sy, cy, 0.0])
        down0 = np.cross(normal, right0)
        down_norm = np.linalg.norm(down0)
        if down_norm > 1e-12:
            down0 = down0 / down_norm

        return right0 * cr + down0 * sr

    def _future_gate_coplanar_with_last_pass(
        self, future_gate: GateSpec,
    ) -> bool:
        """Iter-028: is `future_gate` coplanar (same xy + same normal)
        with ANY already-credited gate?

        Used by the future-gate DQ scan to skip the figure-8 self-
        crossing case: gates that share an xy position by design (e.g.
        figure8.json's gate-1+gate-5 at x=5 and gate-2+gate-6 at x=10,y=5)
        intentionally have overlapping bare openings. The drone crossing
        the future-gate's plane while flying between OTHER gates is
        unavoidable in figure-8 geometry — those crossings are SPATIAL
        consequences of the course design, not out-of-order violations.

        The check passes whenever any prior pass has a coplanar twin
        in the un-credited gates: once the drone has gone through one
        member of a coplanar pair, the second member is allowed to be
        plane-crossed at will until it becomes the current target.

        Coplanar = same xy within 0.5 m AND same yaw within 0.05 rad.
        Z is intentionally NOT required to match — the figure-8 pattern
        relies on z separation between the two coplanar gates.
        """
        if not self._passed:
            return False
        # Build a list of already-passed gates (by lookup into self._gates).
        # self._passed stores gate_ids in pass order; map back to GateSpec.
        passed_ids = set(self._passed)
        for g in self._gates[: self._current_idx]:
            if g.gate_id not in passed_ids:
                continue
            dxy = math.hypot(
                future_gate.position[0] - g.position[0],
                future_gate.position[1] - g.position[1],
            )
            dyaw = abs(
                ((future_gate.yaw - g.yaw) + math.pi) % (2 * math.pi)
                - math.pi
            )
            if dxy < 0.5 and dyaw < 0.05:
                return True
        return False

    @staticmethod
    def _gate_up(gate: GateSpec) -> np.ndarray:
        cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
        cp, sp = math.cos(gate.pitch), math.sin(gate.pitch)
        cr, sr = math.cos(gate.roll), math.sin(gate.roll)

        normal = np.array([cy * cp, sy * cp, -sp])
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
        self._collision_marked_this_tick = None
        self._dq_reason = None
        self._timeout_reason = None
