"""Tests for gate sequencer (gate_sequencing/sequencer.py)."""

import math

import numpy as np
import pytest

from gate_sequencing.sequencer import (
    GateSequencer,
    GateSpec,
    RaceState,
    SequencerConfig,
)


def _make_gate(gate_id: str, position, yaw=0.0, idx=0, **kwargs) -> GateSpec:
    return GateSpec(
        gate_id=gate_id,
        position=position,
        yaw=yaw,
        sequence_index=idx,
        **kwargs,
    )


def _make_course() -> list:
    """Simple 3-gate straight course along X axis, gates facing +X."""
    return [
        _make_gate("G1", (5, 0, 0), yaw=0.0, idx=0),
        _make_gate("G2", (10, 0, 0), yaw=0.0, idx=1),
        _make_gate("G3", (15, 0, 0), yaw=0.0, idx=2),
    ]


class TestGateFrameAxes:
    def test_positive_pitch_tilts_normal_up_in_ned(self):
        gate = _make_gate("G1", (0, 0, 0), yaw=0.0, pitch=0.25)
        normal = GateSequencer._gate_normal(gate)
        assert normal == pytest.approx((math.cos(0.25), 0.0, -math.sin(0.25)))


# ── Gate pass-through detection ──────────────────────────────────────────


class TestPassThroughDetection:
    def test_fly_through_gate_plane(self):
        """Flying straight through a gate should detect pass-through."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        # Approach from behind
        seq.update((3, 0, 0))
        # Fly through
        result = seq.update((7, 0, 0))
        assert result is not None
        assert result.gate_id == "G1"

    def test_no_detection_when_missing_gate(self):
        """Flying past gate but outside the opening should not trigger."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2)]
        seq = GateSequencer(gates)
        seq.start()

        # Approach far off to the side
        seq.update((3, 10, 0))
        result = seq.update((7, 10, 0))
        assert result is None

    def test_fly_parallel_no_crossing(self):
        """Moving parallel to gate plane should not trigger."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        seq.update((5, -2, 0))
        result = seq.update((5, 2, 0))
        # Moving along the gate plane, not crossing it
        assert result is None

    def test_crossing_in_opening(self):
        """Crossing through the center of the opening triggers detection."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        cfg = SequencerConfig(pass_through_margin=1.0)
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        seq.update((4.9, 0, 0))
        result = seq.update((5.1, 0, 0))
        assert result is not None


# ── Sequence progression ─────────────────────────────────────────────────


class TestSequenceProgression:
    def test_gates_must_be_passed_in_order(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        # Try to pass gate 2 first (skip gate 1) — should not register
        seq.update((9, 0, 0))
        result = seq.update((11, 0, 0))
        assert result is None  # G2 is not the current target yet
        assert seq.gates_passed == 0

    def test_sequential_gate_passing(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        # Pass gate 1
        seq.update((4, 0, 0))
        r1 = seq.update((6, 0, 0))
        assert r1 is not None
        assert r1.gate_id == "G1"
        assert seq.gates_passed == 1

        # Pass gate 2
        seq.update((9, 0, 0))
        r2 = seq.update((11, 0, 0))
        assert r2 is not None
        assert r2.gate_id == "G2"
        assert seq.gates_passed == 2

        # Pass gate 3
        seq.update((14, 0, 0))
        r3 = seq.update((16, 0, 0))
        assert r3 is not None
        assert r3.gate_id == "G3"
        assert seq.gates_passed == 3

    def test_progress_fraction(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        assert seq.progress == pytest.approx(0.0)

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.progress == pytest.approx(1 / 3)

    def test_current_gate_advances(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        assert seq.current_gate.gate_id == "G1"

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.current_gate.gate_id == "G2"

    def test_next_gate_property(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        assert seq.next_gate.gate_id == "G2"

    def test_gates_sorted_by_sequence_index(self):
        # Provide gates out of order
        gates = [
            _make_gate("G3", (15, 0, 0), idx=2),
            _make_gate("G1", (5, 0, 0), idx=0),
            _make_gate("G2", (10, 0, 0), idx=1),
        ]
        seq = GateSequencer(gates)
        assert seq.current_gate.gate_id == "G1"


# ── Recovery state transitions ───────────────────────────────────────────


class TestRecoveryTransitions:
    def test_initial_state_waiting(self):
        seq = GateSequencer(_make_course())
        assert seq.state == RaceState.WAITING

    def test_start_transitions_to_racing(self):
        seq = GateSequencer(_make_course())
        seq.start()
        assert seq.state == RaceState.RACING

    def test_off_track_triggers_recovery(self):
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Move very far from the expected path
        seq.update((0, 0, 0))
        seq.update((0, 100, 0))  # way off track
        assert seq.state == RaceState.RECOVERY

    def test_recovery_provides_target(self):
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        seq.update((0, 0, 0))
        seq.update((0, 100, 0))

        target = seq.get_recovery_target()
        assert target is not None
        # Should point toward current gate
        assert target == gates[0].position

    def test_passing_gate_exits_recovery(self):
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Go off track
        seq.update((0, 0, 0))
        seq.update((0, 100, 0))
        assert seq.state == RaceState.RECOVERY

        # Come back and pass through gate 1
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.state == RaceState.RACING

    def test_on_line_far_from_gate_does_not_trigger_recovery(self):
        """Regression: off-track must be cross-track distance to the racing
        line, NOT point distance to the next gate. A drone on the straight
        line to a gate 25 m away (>> off_track_distance*3) used to latch
        RECOVERY at the start of every leg on a real (gate-spaced) course."""
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = [
            _make_gate("G1", (25, 0, 0), yaw=0.0, idx=0),
            _make_gate("G2", (50, 0, 0), yaw=0.0, idx=1),
        ]
        seq = GateSequencer(gates, config=cfg)
        seq.start()
        # Fly straight along the line toward G1 — always far from the gate
        # itself, but perfectly on the path.
        for x in (0, 2, 5, 10, 15, 20):
            seq.update((x, 0, 0))
            assert seq.state == RaceState.RACING, (
                f"false RECOVERY at x={x} while on-line"
            )

    def test_cross_track_excursion_still_triggers_recovery(self):
        """The cross-track metric must still catch a genuine lateral excursion
        even when the drone is close (along-track) to the gate."""
        cfg = SequencerConfig(off_track_distance=5.0)
        gates = [
            _make_gate("G1", (25, 0, 0), yaw=0.0, idx=0),
            _make_gate("G2", (50, 0, 0), yaw=0.0, idx=1),
        ]
        seq = GateSequencer(gates, config=cfg)
        seq.start()
        seq.update((0, 0, 0))
        seq.update((12, 20, 0))  # 20 m lateral off the line (> 15 m)
        assert seq.state == RaceState.RECOVERY

    def test_detection_dropout_triggers_slow_down(self):
        cfg = SequencerConfig(detection_dropout_frames=5)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        for i in range(10):
            seq.update((0, 0, 0), gate_detected=False)

        assert seq.should_slow_down()

    def test_detection_recovery_clears_slow_down(self):
        cfg = SequencerConfig(detection_dropout_frames=5)
        gates = _make_course()
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Lose detection
        for _ in range(10):
            seq.update((0, 0, 0), gate_detected=False)
        assert seq.should_slow_down()

        # Regain detection
        seq.update((0, 0, 0), gate_detected=True)
        # In RACING state with detection → should not slow down
        if seq.state == RaceState.RACING:
            assert not seq.should_slow_down()


# ── Completion detection ─────────────────────────────────────────────────


class TestCompletion:
    def test_all_gates_passed_is_complete(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        # Pass all 3 gates
        for gx in [5, 10, 15]:
            seq.update((gx - 1, 0, 0))
            seq.update((gx + 1, 0, 0))

        assert seq.is_complete
        assert seq.state == RaceState.COMPLETED
        assert seq.gates_passed == 3

    def test_update_after_completion_returns_none(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.is_complete

        result = seq.update((100, 0, 0))
        assert result is None

    def test_waiting_state_returns_none(self):
        seq = GateSequencer(_make_course())
        # Not started
        result = seq.update((4, 0, 0))
        assert result is None

    def test_current_gate_none_after_completion(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.current_gate is None


# ── Proximity pass-through must respect the gate opening ───────────────


class TestProximityRespectsOpening:
    """A proximity-based pass-through must only count passes inside the lit
    gate opening. If the drone flies close to the gate centre but outside
    the rectangular opening, it must NOT be credited — otherwise the drone
    can skim past unlit parts of the gate frame and still score.
    """

    def test_skim_outside_opening_does_not_count(self):
        """Drone flies within proximity distance but outside the opening
        box on the far side of the gate plane. Must not count."""
        # 1.2m-wide gate at origin, facing +X.
        gates = [_make_gate("G1", (0, 0, 0), yaw=0.0, idx=0,
                             interior_width=1.2, interior_height=1.2)]
        cfg = SequencerConfig(proximity_pass_distance=1.2)
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Previous frame: in front of the gate plane, 0.9m laterally right
        # (outside the half-width of 0.6m), 0.8m ahead on +X.
        seq.update((-0.8, 0.9, 0))
        # Current frame: past the gate plane, still 0.9m laterally right.
        # Distance to centre: sqrt(0.8^2 + 0.9^2) = 1.20m ≈ proximity limit.
        # Plane sign did NOT change in a way that crosses the opening box
        # (lateral_right = 0.9m > half_w=0.6m), so this is a skim-by.
        result = seq.update((0.8, 0.9, 0))
        # Old (buggy) behaviour: would return the gate because distance
        # < proximity_pass_distance and d_curr > -0.5.
        # New behaviour: must return None because drone is outside the
        # opening box.
        assert result is None, (
            "Drone skimmed outside the gate opening but was credited "
            "with a pass-through — proximity fallback is not respecting "
            "the lit gate opening."
        )

    def test_close_through_opening_still_counts(self):
        """Drone passes through the gate opening at reasonable distance from
        centre — this is still a legitimate pass and should count even when
        the plane-crossing sample straddles the plane by only a hair."""
        gates = [_make_gate("G1", (0, 0, 0), yaw=0.0, idx=0,
                             interior_width=1.2, interior_height=1.2)]
        cfg = SequencerConfig(proximity_pass_distance=1.2)
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Drone passes through the opening near its edge but inside it:
        # lateral offset 0.4m (< half_w=0.6m).
        seq.update((-0.5, 0.4, 0))
        result = seq.update((0.5, 0.4, 0))
        assert result is not None
        assert result.gate_id == "G1"

    def test_proximity_behind_plane_inside_opening_counts(self):
        """Drone is inside the gate opening and within proximity distance but
        the plane sign didn't change (sampling gap). Proximity fallback
        should still credit the pass because the drone is in the lit
        frame."""
        gates = [_make_gate("G1", (0, 0, 0), yaw=0.0, idx=0,
                             interior_width=1.2, interior_height=1.2)]
        cfg = SequencerConfig(proximity_pass_distance=1.2)
        seq = GateSequencer(gates, config=cfg)
        seq.start()

        # Previous frame: in front of plane, 0.3m laterally (inside opening).
        seq.update((-0.3, 0.3, 0))
        # Current frame: just past plane (within 0.5m), still inside opening.
        result = seq.update((0.1, 0.3, 0))
        assert result is not None
        assert result.gate_id == "G1"


# ── Gate orientation basis ───────────────────────────────────────────────


class TestGateOrientationBasis:
    def test_pitched_gate_basis_is_orthonormal(self):
        gate = _make_gate(
            "G1",
            (0, 0, 0),
            yaw=math.radians(25.0),
            idx=0,
            pitch=math.radians(30.0),
            roll=math.radians(15.0),
        )

        normal = GateSequencer._gate_normal(gate)
        right = GateSequencer._gate_right(gate)
        up = GateSequencer._gate_up(gate)

        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-9)
        assert float(np.dot(normal, right)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.dot(normal, up)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.dot(right, up)) == pytest.approx(0.0, abs=1e-9)

    def test_pitched_gate_pass_through_uses_tilted_opening(self):
        gate = _make_gate(
            "G1",
            (0, 0, 0),
            yaw=0.0,
            idx=0,
            pitch=math.radians(30.0),
            interior_width=1.2,
            interior_height=1.2,
        )
        seq = GateSequencer([gate], config=SequencerConfig(pass_through_margin=1.0))
        seq.start()

        normal = GateSequencer._gate_normal(gate)
        up = GateSequencer._gate_up(gate)
        center = np.array(gate.position, dtype=float)

        seq.update(tuple(center - normal + up * 0.7))
        result = seq.update(tuple(center + normal + up * 0.7))

        assert result is None


# ── Crash-into-gate detection ────────────────────────────────────────────


class TestCrashIntoGate:
    """A 'crash' is a plane crossing inside the outer frame bounds but
    outside the interior opening (drone hit the frame). A 'miss' is a
    plane crossing completely outside the outer frame bounds (drone flew
    around the gate). Both are classified on the highlighted gate only."""

    def test_strut_hit_records_crash_not_pass(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2,
                            border_width=0.15)]
        seq = GateSequencer(gates)
        seq.start()
        # Drone flies into the right strut: lateral offset 0.7m
        # (interior half = 0.6m, outer half = 0.75m → inside frame, outside opening)
        seq.update((4.5, 0.7, 0))
        passed = seq.update((5.5, 0.7, 0))
        assert passed is None
        assert seq.gates_passed == 0
        assert seq.crashed_gate_ids == ["G1"]
        assert seq.last_event == "crash"
        # Crash position must be reported (used by the dynamic replanner).
        gid, crash_pt = seq.last_crash
        assert gid == "G1"
        assert abs(crash_pt[0] - 5.0) < 1e-6  # crossing happened on the gate plane
        assert abs(crash_pt[1] - 0.7) < 1e-6

    def test_top_bar_hit_records_crash(self):
        """Vertical hit on the top bar — same crash classification, vertical axis."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2,
                            border_width=0.15)]
        seq = GateSequencer(gates)
        seq.start()
        # 0.7m above centre → inside outer frame (half_outer = 0.75m), outside opening (half = 0.6m).
        seq.update((4.5, 0, 0.7))
        passed = seq.update((5.5, 0, 0.7))
        assert passed is None
        assert seq.crashed_gate_ids == ["G1"]
        assert seq.last_event == "crash"

    def test_complete_miss_classified_as_miss_not_crash(self):
        """Plane crossed completely outside the frame — miss, not crash."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2,
                            border_width=0.15)]
        seq = GateSequencer(gates)
        seq.start()
        # Lateral 5m from centre — well outside outer frame.
        seq.update((4.5, 5.0, 0))
        seq.update((5.5, 5.0, 0))
        assert seq.gates_passed == 0
        assert seq.crashed_gate_ids == []
        assert seq.missed_gate_ids == ["G1"]
        assert seq.last_event == "miss"

    def test_clean_pass_does_not_record_crash_or_miss(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()
        seq.update((4, 0, 0))
        passed = seq.update((6, 0, 0))
        assert passed is not None
        assert seq.crashed_gate_ids == []
        assert seq.missed_gate_ids == []
        assert seq.last_event == "pass"

    def test_mark_collision_records_crash_authoritatively(self):
        """External collision sources (e.g. PyBullet contact) bypass geometry."""
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()
        # Drone flying nominally — no geometric crash detected
        seq.update((1, 0, 0))
        seq.update((2, 0, 0))
        assert seq.crashed_gate_ids == []
        # Physics layer reports a contact with G1
        seq.mark_collision("G1", position=(2.0, 0.0, 0.0))
        assert seq.crashed_gate_ids == ["G1"]
        assert seq.last_event == "crash"
        gid, pt = seq.last_crash
        assert gid == "G1"
        assert pt == (2.0, 0.0, 0.0)

    def test_mark_collision_unknown_gate_raises(self):
        seq = GateSequencer(_make_course())
        seq.start()
        with pytest.raises(ValueError):
            seq.mark_collision("does-not-exist")

    def test_mark_collision_uses_last_known_position_by_default(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()
        seq.update((1.5, 0.2, 0.3))
        seq.mark_collision("G1")
        gid, pt = seq.last_crash
        assert gid == "G1"
        # Falls back to the last-known drone position
        assert pt == (1.5, 0.2, 0.3)

    def test_crash_does_not_advance_target(self):
        """A crash on the highlighted gate must not advance the target.
        The drone keeps trying to pass the same gate (or the upstream
        replanner clears the situation).

        Test uses explicit 1.2 m / 0.15 m geometry — y=0.7 is in the
        [0.6, 0.75] strut annulus under that geometry. With AIGP defaults
        (1.5 m opening, 0.6 m border) the same y would be a clean pass,
        which is fine for the AIGP-default path but isn't what THIS test
        is checking.
        """
        gates = [
            _make_gate(
                f"G{i+1}", (5.0 * (i + 1), 0.0, 0.0), yaw=0.0, idx=i,
                interior_width=1.2, interior_height=1.2, border_width=0.15,
            )
            for i in range(3)
        ]
        seq = GateSequencer(gates)
        seq.start()
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))
        assert seq.current_gate.gate_id == "G1"  # still target
        assert seq.gates_passed == 0

    # ── P1-4: same-tick mark_collision wins over geometric pass ────────

    def test_mark_collision_wins_over_same_tick_pass(self):
        """If a physics layer reports a collision and the same-tick geometry
        ALSO classifies the crossing as a pass (lenient pass_through_margin),
        the crash mark must win — `_last_event=='crash'` and the gate is NOT
        added to `passed_gate_ids`."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2,
                            border_width=0.15)]
        # Lenient pass margin (sim_pybullet's production default).
        seq = GateSequencer(
            gates, config=SequencerConfig(pass_through_margin=1.5),
        )
        seq.start()
        # Step 1: drone before the gate plane.
        seq.update((4.5, 0, 0))
        # Step 2: physics says the drone hit the frame, then geometry runs.
        # The drone is at (5.5, 0, 0) — clean centre crossing, would normally
        # classify as pass under margin=1.5. Crash mark must override.
        seq.mark_collision("G1", position=(5.0, 0.7, 0))
        result = seq.update((5.5, 0, 0))
        assert seq.last_event == "crash", (
            f"crash mark overwritten; last_event={seq.last_event!r}"
        )
        assert "G1" not in seq.passed_gate_ids
        assert seq.gates_passed == 0
        assert result is None

    # ── P1-5: state-gated + idempotent mark_collision ──────────────────

    def test_mark_collision_pre_start_is_silent_noop(self):
        """A pre-race spawn-overlap should not register a phantom crash."""
        seq = GateSequencer(_make_course())
        # NO seq.start() — state is WAITING.
        seq.mark_collision("G1", position=(5.0, 0.7, 0))
        assert seq.crashed_gate_ids == []
        assert seq.last_event is None

    def test_mark_collision_after_completion_is_noop(self):
        """A post-race fly-through must not corrupt the crash log."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))  # passes G1 → state COMPLETED
        assert seq.is_complete

        seq.mark_collision("G1", position=(5.0, 0.7, 0))
        assert seq.crashed_gate_ids == []  # noop; pass status preserved
        assert seq.last_event == "pass"

    def test_mark_collision_idempotent_dedupes_repeat_calls(self):
        """Repeat mark_collision calls on the same gate (e.g. PyBullet
        contact manifold persists) must NOT append duplicate entries."""
        seq = GateSequencer(_make_course())
        seq.start()
        seq.mark_collision("G1", position=(5.0, 0.7, 0))
        seq.mark_collision("G1", position=(5.0, 0.7, 0))
        seq.mark_collision("G1", position=(5.0, 0.7, 0))
        assert seq.crashed_gate_ids == ["G1"]

    # ── P1-6: decoupled crash_margin ───────────────────────────────────

    def test_lenient_pass_margin_still_detects_geometric_crashes(self):
        """With production pass_through_margin=1.5 and crash_margin=1.0,
        a frame hit at lateral 0.7 (inside outer 0.75, outside bare 0.6)
        must still classify as a crash — not as a pass."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2,
                            border_width=0.15)]
        seq = GateSequencer(
            gates,
            config=SequencerConfig(
                pass_through_margin=1.5,  # production sim_pybullet default
                crash_margin=1.0,         # bare opening for crash classification
            ),
        )
        seq.start()
        seq.update((4.5, 0.7, 0))
        result = seq.update((5.5, 0.7, 0))
        assert seq.crashed_gate_ids == ["G1"], (
            "frame hit not classified as crash under lenient pass margin"
        )
        assert seq.last_event == "crash"
        assert result is None

    # ── P1-7: per-fly-by dedupe of crashes/misses ──────────────────────

    def test_oscillating_against_frame_records_one_crash(self):
        """Multiple plane re-crossings on the same gate during one fly-by
        (drone wedged in the frame) record ONE crash entry."""
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2,
                            border_width=0.15)]
        seq = GateSequencer(gates)
        seq.start()
        # Cross gate plane back and forth at lateral 0.7m (in crash zone).
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))   # crash 1
        seq.update((4.5, 0.7, 0))   # cross back — should NOT dupe-append
        seq.update((5.5, 0.7, 0))   # cross again — should NOT dupe-append
        assert seq.crashed_gate_ids == ["G1"]

    def test_oscillating_outside_frame_records_one_miss(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0)]
        seq = GateSequencer(gates)
        seq.start()
        seq.update((4.5, 5.0, 0))
        seq.update((5.5, 5.0, 0))  # miss 1
        seq.update((4.5, 5.0, 0))
        seq.update((5.5, 5.0, 0))  # repeat — dedupe
        assert seq.missed_gate_ids == ["G1"]

    def test_reset_clears_crashes_and_misses(self):
        gates = [_make_gate("G1", (5, 0, 0), yaw=0.0, idx=0,
                            interior_width=1.2, interior_height=1.2,
                            border_width=0.15)]
        seq = GateSequencer(gates)
        seq.start()
        seq.update((4.5, 0.7, 0))
        seq.update((5.5, 0.7, 0))
        assert seq.crashed_gate_ids == ["G1"]
        seq.reset()
        assert seq.crashed_gate_ids == []
        assert seq.missed_gate_ids == []
        assert seq.last_event is None


# ── Pass-through if and only if highlighted ──────────────────────────────


class TestPassIfAndOnlyIfHighlighted:
    """A pass-through must be reported if AND only if the gate the drone
    crossed was the highlighted (current target) gate at the time of
    crossing. Crossing a non-highlighted gate must NEVER be credited —
    even if the drone goes through it geometrically."""

    def test_crossing_highlighted_gate_credits_the_pass(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()
        # G1 is highlighted at the start
        assert seq.current_gate.gate_id == "G1"
        seq.update((4, 0, 0))
        result = seq.update((6, 0, 0))
        assert result is not None
        assert result.gate_id == "G1"
        assert seq.gates_passed == 1
        assert "G1" in seq.passed_gate_ids

    def test_crossing_non_highlighted_gate_does_not_credit(self):
        """Drone flies cleanly through G2 while G1 is highlighted. No credit."""
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()
        assert seq.current_gate.gate_id == "G1"  # highlighted

        # Two straight crossings of G2's plane (x=10), inside its opening.
        seq.update((9, 0, 0))
        result = seq.update((11, 0, 0))
        assert result is None
        assert seq.gates_passed == 0
        assert "G2" not in seq.passed_gate_ids
        # Sequencer must NOT have advanced — G1 is still the highlighted target.
        assert seq.current_gate.gate_id == "G1"

    def test_late_crossing_of_previously_highlighted_gate_does_not_credit(self):
        """After G1 is passed, G2 is highlighted. Re-crossing G1's plane must
        not register — G1 is no longer highlighted."""
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()
        # Pass G1 cleanly
        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.gates_passed == 1
        assert seq.current_gate.gate_id == "G2"

        # Fly back through G1 (no longer highlighted)
        # We need to clear prev_position bias by walking around it. Two
        # passes both on the upstream side first, then crossing back.
        seq.update((6, 5, 0))
        seq.update((4, 5, 0))   # parallel — no crossing
        result = seq.update((4, 0, 0))   # now back at G1's near side
        # Either the back-cross of G1 is detected as nothing (current is G2),
        # or as nothing on G2 either. Either way, gates_passed stays at 1.
        assert result is None
        assert seq.gates_passed == 1
        assert seq.current_gate.gate_id == "G2"  # G2 is still highlighted

    def test_passing_iff_in_a_4_gate_course(self):
        """Each pass-through must correspond exactly to the highlighted
        gate at the time of the crossing. Build a 4-gate course and walk
        the drone through them in order, asserting credit aligns with the
        highlighted ID at each step."""
        gates = [
            _make_gate("G1", (5, 0, 0), yaw=0.0, idx=0),
            _make_gate("G2", (10, 0, 0), yaw=0.0, idx=1),
            _make_gate("G3", (15, 0, 0), yaw=0.0, idx=2),
            _make_gate("G4", (20, 0, 0), yaw=0.0, idx=3),
        ]
        seq = GateSequencer(gates)
        seq.start()

        sequence_observed = []
        for x in range(0, 22):
            highlighted_before = seq.current_gate.gate_id if seq.current_gate else None
            r = seq.update((float(x), 0, 0))
            if r is not None:
                # Pass must match what was highlighted *before* the update.
                assert r.gate_id == highlighted_before
                sequence_observed.append(r.gate_id)

        assert sequence_observed == ["G1", "G2", "G3", "G4"]
        assert seq.gates_passed == 4

    def test_skip_then_correct_path(self):
        """Drone strays past a later gate while G1 is highlighted — not
        credited. Only the highlighted gate (G1) earns a pass."""
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        # Stray pass of G3 (highlighted=G1). Crossing G3's plane in its
        # opening MUST NOT credit because G3 isn't highlighted.
        seq.update((14, 0, 0))
        r = seq.update((16, 0, 0))
        assert r is None
        assert seq.gates_passed == 0
        assert seq.current_gate.gate_id == "G1"
        # Importantly G3 is NOT in passed_gate_ids even though we crossed it.
        assert "G3" not in seq.passed_gate_ids


# ── Reset ────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self):
        gates = _make_course()
        seq = GateSequencer(gates)
        seq.start()

        seq.update((4, 0, 0))
        seq.update((6, 0, 0))
        assert seq.gates_passed == 1

        seq.reset()
        assert seq.gates_passed == 0
        assert seq.state == RaceState.WAITING
        assert not seq.is_complete
        assert seq.current_gate.gate_id == "G1"
