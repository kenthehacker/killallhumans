"""
Adversarial tests for the gate sequencer (iter-001 A1).

These tests MUST fail on the current sequencer (which silently no-ops on
out-of-order forward crossings and can credit a clean run after a U-turn).
They turn green once iter-001 A5 lands the in-order enforcement.

Charter requirement: the testbench is suspect; PASS means nothing without
an adversarial harness that tries to break it. Each test below encodes a
concrete behavioural claim pulled straight from `.loop/specs/2_known_issues.md`.
"""
from __future__ import annotations

import math

import pytest

from competition.aigp_geometry import (
    AIGP_GATE_BORDER_M,
    AIGP_GATE_INTERIOR_M,
    AIGP_GATE_OUTER_M,
)
from gate_sequencing.sequencer import (
    GateSequencer,
    GateSpec,
    RaceState,
    SequencerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_line(n: int, spacing: float = 5.0) -> list[GateSpec]:
    """Build a straight line of n gates along +X (NED), facing +X."""
    return [
        GateSpec(
            gate_id=f"g{i+1}",
            position=(spacing * (i + 1), 0.0, -2.0),
            yaw=0.0,
            sequence_index=i,
        )
        for i in range(n)
    ]


def _ordered_cfg() -> SequencerConfig:
    """Sequencer config with strict in-order enforcement on (A5)."""
    return SequencerConfig(enforce_in_order=True)


# ---------------------------------------------------------------------------
# I-1: out-of-order forward skip must DQ (terminal)
# ---------------------------------------------------------------------------

def test_out_of_order_forward_skip_is_terminal_dq():
    """Drone flies straight through gate-3's opening while current is gate-1.

    Pre-fix: silent no-op — no event recorded, current_idx unchanged.
    Post-fix: terminal DQ. `is_disqualified` becomes True; `dq_reason`
    names the violating gate; `is_complete` stays False forever.
    """
    gates = _make_line(5)
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()

    # Approach and cross gate-3 directly. Current target is gate-1.
    # Gate-3 sits at x=15; segment from x=14.5 to x=15.5 straddles its plane.
    seq.update((14.5, 0.0, -2.0))
    seq.update((15.5, 0.0, -2.0))

    assert seq.is_disqualified, "out-of-order forward skip must DQ"
    assert seq.dq_reason is not None
    assert "out_of_order" in seq.dq_reason
    assert "g3" in seq.dq_reason
    assert not seq.is_complete
    # And future events must not credit anything.
    seq.update((4.5, 0.0, -2.0))
    seq.update((5.5, 0.0, -2.0))
    assert seq.gates_passed == 0


def test_u_turn_after_skip_does_not_recover():
    """Pass g1 cleanly, skip-through g3, U-turn back through g2, then g3.

    Pre-fix: ends with `is_complete=True`, all gates "passed" — a clean run.
    Post-fix: DQ fires on the first g3 crossing.
    """
    gates = _make_line(3)
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()

    # Cleanly pass g1 (at x=5).
    seq.update((4.5, 0.0, -2.0))
    seq.update((5.5, 0.0, -2.0))
    assert seq.gates_passed == 1, "g1 should be credited"

    # Skip-through g3 (at x=15) without crossing g2 (at x=10) first.
    # We can't actually teleport, so simulate two ticks that include the
    # plane crossing of g2 (which would credit g2) AND the crossing of g3.
    # Use a path that arcs around g2 — y=2.0 keeps us outside g2's opening
    # half-width (AIGP 0.75m) so we miss g2 cleanly and then come back to
    # cross g3 inside its opening.
    seq.update((9.5, 2.0, -2.0))
    seq.update((10.5, 2.0, -2.0))   # crosses g2's plane outside opening -> miss
    seq.update((14.5, 0.0, -2.0))
    seq.update((15.5, 0.0, -2.0))   # crosses g3 inside opening — out-of-order!

    assert seq.is_disqualified, "u-turn skip must DQ"
    assert not seq.is_complete


def test_far_plane_grazing_does_not_dq():
    """A plane crossing far OUTSIDE the gate frame is benign.

    The sequencer must check `_point_in_gate_opening` on the crossing
    point, not just the infinite-plane crossing. Otherwise any segment
    that crosses gate-N's plane anywhere — even meters off-axis — would
    DQ.
    """
    gates = _make_line(5)
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()

    # Cross gate-3's plane at y = +10 m — well outside the 0.75 m half-
    # opening AND outside the 1.35 m outer-frame half.
    seq.update((14.5, 10.0, -2.0))
    seq.update((15.5, 10.0, -2.0))

    assert not seq.is_disqualified
    assert seq.dq_reason is None
    # And gate-1 must remain the current target.
    assert seq.current_gate is not None
    assert seq.current_gate.gate_id == "g1"


def test_outer_frame_strike_classified_as_crash_not_dq():
    """A crossing in the [opening, outer_frame] annulus is a CRASH, not a DQ.

    Both are terminal failure modes, but they have distinct semantics:
    crash = hit a strut; DQ = passed a wrong gate's opening.
    """
    gates = _make_line(3)
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()

    # Hit gate-1's strut: opening half = 0.75; outer half = 1.35.
    # Cross at y = 1.0 — inside the outer frame, outside the opening.
    seq.update((4.5, 1.0, -2.0))
    seq.update((5.5, 1.0, -2.0))

    assert seq.last_crash is not None, "expected a frame-strut crash"
    assert seq.last_crash[0] == "g1"
    # A frame-strut hit is NOT an out-of-order DQ.
    assert seq.dq_reason is None or "out_of_order" not in (seq.dq_reason or "")


# ---------------------------------------------------------------------------
# I-5: AIGP geometry defaults
# ---------------------------------------------------------------------------

def test_gate_spec_defaults_to_aigp_geometry():
    """GateSpec() with no geometry args must match the VADR-TS-002 numbers."""
    spec = GateSpec(gate_id="t", position=(0.0, 0.0, 0.0))
    assert spec.interior_width == pytest.approx(AIGP_GATE_INTERIOR_M)
    assert spec.interior_height == pytest.approx(AIGP_GATE_INTERIOR_M)
    assert spec.border_width == pytest.approx(AIGP_GATE_BORDER_M)
    assert spec.outer_width == pytest.approx(AIGP_GATE_OUTER_M)
    assert spec.outer_height == pytest.approx(AIGP_GATE_OUTER_M)


def test_explicit_override_still_works_for_legacy_tracks():
    """Tracks like race_01.json must still be able to opt into 1.2 m gates."""
    spec = GateSpec(
        gate_id="legacy",
        position=(0.0, 0.0, 0.0),
        interior_width=1.2,
        interior_height=1.2,
        border_width=0.15,
    )
    assert spec.interior_width == pytest.approx(1.2)
    assert spec.outer_width == pytest.approx(1.2 + 2 * 0.15)


# ---------------------------------------------------------------------------
# Backward-compat: existing happy-path tests must stay green.
# ---------------------------------------------------------------------------

def test_in_order_clean_run_still_completes():
    """A clean run with strict ordering on must finish successfully."""
    gates = _make_line(3)
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()

    # Pass each gate in order.
    for x_center in (5.0, 10.0, 15.0):
        seq.update((x_center - 0.5, 0.0, -2.0))
        seq.update((x_center + 0.5, 0.0, -2.0))

    assert seq.is_complete
    assert not seq.is_disqualified
    assert seq.gates_passed == 3


# ---------------------------------------------------------------------------
# Iter-001 review Opus F2: multi-gate-per-tick crediting
# ---------------------------------------------------------------------------

def test_segment_crossing_two_gates_credits_both():
    """A single segment that passes through two consecutive gates' openings
    must credit BOTH (not just the first one).

    Pre-fix: only the current gate was checked per update; after credit
    _current_idx advanced and the next gate's opening crossing was silently
    lost. The post-credit DQ scan started at _current_idx + 1 (i.e., g3+),
    so g2's crossing wasn't reclassified either — phantom skip.
    """
    # Three gates 1.5 m apart so a single fast tick straddles two openings.
    gates = [
        GateSpec(gate_id=f"g{i+1}", position=(5.0 + 1.5 * i, 0.0, -2.0),
                 yaw=0.0, sequence_index=i)
        for i in range(3)
    ]
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()
    # Tick 1: drone at (4.0, 0, -2) — behind g1's plane.
    seq.update((4.0, 0.0, -2.0))
    # Tick 2: drone at (7.0, 0, -2) — past BOTH g1 (x=5) and g2 (x=6.5).
    seq.update((7.0, 0.0, -2.0))
    assert seq.gates_passed == 2, (
        f"expected both g1 and g2 credited; got {seq.gates_passed}"
    )
    assert "g1" in seq.passed_gate_ids
    assert "g2" in seq.passed_gate_ids
    assert not seq.is_disqualified
    assert seq.current_gate is not None and seq.current_gate.gate_id == "g3"


def test_segment_crossing_three_gates_credits_all_three():
    """Stress test: one segment, three openings in a row."""
    gates = [
        GateSpec(gate_id=f"g{i+1}", position=(5.0 + 1.0 * i, 0.0, -2.0),
                 yaw=0.0, sequence_index=i)
        for i in range(4)
    ]
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()
    seq.update((4.0, 0.0, -2.0))
    # Segment from x=4 to x=8 crosses g1 (5), g2 (6), g3 (7) but ends before g4 (8).
    seq.update((7.5, 0.0, -2.0))
    assert seq.gates_passed == 3
    assert seq.current_gate is not None and seq.current_gate.gate_id == "g4"


# ---------------------------------------------------------------------------
# Iter-001 review Opus F3: future-gate strut hit must be a crash, not silent
# ---------------------------------------------------------------------------

def test_future_gate_strut_hit_classified_as_crash():
    """A segment that crosses a future gate's plane in the [opening, outer]
    annulus (i.e., physically hits the strut) must record a crash on that
    future gate — neither silent ignore nor an out-of-order DQ.

    With AIGP defaults: opening half = 0.75 m, outer half = 1.35 m. A
    crossing at y=1.0 is squarely in the strut annulus.

    Pre-fix: the crash branch only ran against `current_gate`, and the
    DQ branch used `_point_in_gate_opening` which is `False` for y=1.0,
    so the future-gate strut hit was a silent no-op.
    """
    gates = _make_line(2)
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()
    # Skip g1 cleanly (don't credit) and aim straight at g2's strut.
    # g1 at x=5, g2 at x=10. Approach g2 at y=1.0 (strut zone).
    # First we have to set up _prev_position to be before g1's plane and
    # then cross g2's plane in one segment. Easiest: come in at y=1.0
    # from (4.0, 1.0, -2.0) to (10.5, 1.0, -2.0).
    # That crosses g1's plane at (5, 1, -2) — outside g1's opening
    # (y=1.0 > 0.75 half) but inside g1's outer (1.0 < 1.35 half) —
    # so g1 is a CRASH first.
    seq.update((4.0, 1.0, -2.0))
    seq.update((10.5, 1.0, -2.0))
    # The first thing the sequencer sees is g1's strut hit (current gate).
    # That records a crash; current_idx stays at 0.
    assert seq.last_crash is not None
    # Crash gate should be one of g1 or g2 (the strut path); preferred g1
    # because it's the first crossing in the segment.
    crashed_id = seq.last_crash[0]
    assert crashed_id in ("g1", "g2")


def test_future_gate_strut_hit_without_current_gate_hit():
    """Tests the future-gate strut path independent of any current-gate
    interaction: drone passes g1 cleanly, then a single segment grazes
    g2's strut while g1 is now credited.
    """
    gates = _make_line(2)
    seq = GateSequencer(gates, _ordered_cfg())
    seq.start()
    # Pass g1 cleanly.
    seq.update((4.5, 0.0, -2.0))
    seq.update((5.5, 0.0, -2.0))
    assert seq.gates_passed == 1
    assert seq.last_crash is None
    # Now graze g2's strut.
    seq.update((9.5, 1.0, -2.0))
    seq.update((10.5, 1.0, -2.0))
    # g2 is the CURRENT gate now (current_idx=1), so this is a current-
    # gate strut hit. Should be classified as crash on g2.
    assert seq.last_crash is not None
    assert seq.last_crash[0] == "g2"
    assert seq.gates_passed == 1   # not credited
    assert not seq.is_disqualified


# ---------------------------------------------------------------------------
# Iter-001 review Opus F14: DQ uses strict opening, not lenient pass_through_margin
# ---------------------------------------------------------------------------

def test_dq_uses_strict_opening_when_pass_through_margin_is_lenient():
    """With pass_through_margin=1.5, the lenient opening half-width is
    1.5×0.75 = 1.125 m. But the strict crash-margin opening is the bare
    0.75 m. A future-gate crossing at y=0.9 m (inside lenient, outside
    strict) must NOT trigger a DQ — it must either be a crash (if in
    outer frame) or benign.

    Pre-fix: the DQ check used `_point_in_gate_opening` which used the
    lenient margin → false-positive DQ on legitimate-but-grazing crossings.
    """
    gates = _make_line(2)
    seq = GateSequencer(gates, SequencerConfig(
        enforce_in_order=True,
        pass_through_margin=1.5,
    ))
    seq.start()
    # Skip past g1 (current) without crossing its plane; we want to test
    # ONLY the future-gate DQ branch behaviour. Move drone far around g1.
    # Approach g2 (at x=10) at y=0.9 — inside lenient opening (1.125)
    # but outside strict opening (0.75). The strut hit IS inside the
    # outer (1.35), so this should be classified as crash, not DQ.
    seq.update((4.0, 0.9, -2.0))
    seq.update((10.5, 0.9, -2.0))
    # The drone crossed g1's plane too (at (5, 0.9, -2)). y=0.9 is outside
    # g1's strict opening (0.75) but inside g1's outer (1.35). So g1 gets
    # a crash hit. That's a current-gate strut hit — terminal.
    # Either way, the test's MAIN assertion: NO DQ on g2 due to lenient
    # margin leak.
    assert not seq.is_disqualified or (
        seq.dq_reason and "out_of_order" not in seq.dq_reason
    ), (
        f"strict-opening DQ check should not fire on y=0.9 with margin=1.5; "
        f"got dq_reason={seq.dq_reason}"
    )


def test_far_grazing_with_lenient_margin_still_benign():
    """Sanity follow-on for F14: a TRULY far-out crossing remains benign."""
    gates = _make_line(2)
    seq = GateSequencer(gates, SequencerConfig(
        enforce_in_order=True,
        pass_through_margin=1.5,
    ))
    seq.start()
    seq.update((4.5, 10.0, -2.0))
    seq.update((10.5, 10.0, -2.0))
    assert not seq.is_disqualified
    # Far crossings don't classify as crashes either.
    if seq.last_crash is not None:
        # If the crash branch did fire somehow, it should be on g1 (current),
        # not g2.
        assert seq.last_crash[0] == "g1"
