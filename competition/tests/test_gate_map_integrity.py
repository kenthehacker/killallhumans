"""Tests for competition.gate_map_integrity — the gate-map corruption monitor
(VQ2 robustness item 3).

Each corruption case proves a specific failure mode the deep-research report /
race-day handoff named: sign-flipped X, Z≈−350, uniform offset / drift (the one
the old bounding-box check misses), single-gate outlier, wrong gate count, and
scrambled spacing. The healthy VQ1 map (and jittered copies of it) must PASS.

Tests are fully offline: they build gates from a tiny ``_Gate`` stand-in (only a
``.position`` + ``.gate_id``), and one test also uses the real ``GateSpec`` to
prove the duck type matches.
"""
import dataclasses
import math
from typing import List, Optional, Tuple

import pytest

from competition.gate_map_integrity import (
    Bounds,
    GateMapVerdict,
    ReferenceGate,
    check_gate_map,
    read_reference_json,
    write_reference_json,
)


# Healthy VQ1 first-contact course (NED, metres) — same map as
# FakeAdapter.VQ1_GATES / the track_data end-to-end test.
HEALTHY_POSITIONS: List[Tuple[float, float, float]] = [
    (-23.3, -0.4, -0.03),
    (-46.9, -2.5, 5.07),
    (-74.6, 1.2, 13.67),
    (-111.5, -5.1, 24.57),
    (-135.5, -0.8, 25.36),
    (-159.2, -4.4, 25.97),
]


@dataclasses.dataclass
class _Gate:
    """Minimal GateSpec stand-in: just what check_gate_map reads."""
    position: Tuple[float, float, float]
    gate_id: Optional[str] = None


def _healthy(positions=HEALTHY_POSITIONS) -> List[_Gate]:
    return [_Gate(position=p, gate_id=str(i)) for i, p in enumerate(positions)]


# ---------------------------------------------------------------------------
# Healthy map + jitter
# ---------------------------------------------------------------------------

def test_healthy_map_passes():
    verdict = check_gate_map(_healthy())
    assert verdict.ok is True
    assert verdict.diagnosis == "ok"
    assert verdict.suggested_correction is None


def test_healthy_real_gatespec_passes():
    # Prove the real GateSpec duck-types into check_gate_map (no MAVLink needed
    # to construct a GateSpec — it only pulls in competition.aigp_geometry).
    from gate_sequencing.sequencer import GateSpec

    gates = [
        GateSpec(gate_id=str(i), position=p, sequence_index=i)
        for i, p in enumerate(HEALTHY_POSITIONS)
    ]
    verdict = check_gate_map(gates)
    assert verdict.ok is True
    assert verdict.diagnosis == "ok"


def test_small_float_jitter_still_passes():
    # Sub-decimetre jitter (the kind a clean re-fetch shows) must not trip any
    # check, with OR without a reference.
    jittered = []
    for i, (x, y, z) in enumerate(HEALTHY_POSITIONS):
        d = 0.05 * ((-1) ** i)  # +/-5 cm
        jittered.append((x + d, y - d, z + d))
    gates = _healthy(jittered)

    assert check_gate_map(gates).ok is True
    # And against a reference of the pristine map, still a PASS (jitter < tol).
    ref = _healthy()
    v = check_gate_map(gates, reference=ref)
    assert v.ok is True
    assert v.diagnosis == "ok"


# ---------------------------------------------------------------------------
# Sign-flip X (the named "sign-flipped X" failure)
# ---------------------------------------------------------------------------

def test_sign_flipped_x():
    flipped = [(-x, y, z) for (x, y, z) in HEALTHY_POSITIONS]  # x -> +x
    v = check_gate_map(_healthy(flipped))
    assert v.ok is False
    assert v.diagnosis == "sign_flip_x"
    assert v.suggested_correction == "negate_x"


# ---------------------------------------------------------------------------
# Z ≈ −350 (the named "Z≈−350" failure) and a clean Z flip
# ---------------------------------------------------------------------------

def test_z_approx_minus_350_flagged():
    # All z driven to ~-350 (far below the -50 floor) => NOT ok, and the
    # diagnosis must name the z corruption (out_of_bounds with no clean single
    # negation recovering it, OR sign_flip_z).
    corrupt = [(x, y, -350.0) for (x, y, z) in HEALTHY_POSITIONS]
    v = check_gate_map(_healthy(corrupt))
    assert v.ok is False
    # -350 is uniform, so no single negation restores the per-gate z spread;
    # it should be reported as out_of_bounds, explicitly mentioning z range.
    assert v.diagnosis in {"out_of_bounds", "sign_flip_z"}
    assert "z" in v.message.lower()


def test_clean_z_sign_flip_recovered():
    # A genuine z SIGN flip (z -> -z) keeps the per-gate spread, so negate_z
    # restores the expected region => sign_flip_z + negate_z.
    flipped = [(x, y, -z) for (x, y, z) in HEALTHY_POSITIONS]
    v = check_gate_map(_healthy(flipped))
    assert v.ok is False
    assert v.diagnosis == "sign_flip_z"
    assert v.suggested_correction == "negate_z"


# ---------------------------------------------------------------------------
# Uniform offset / drift — WHY the reference exists
# ---------------------------------------------------------------------------

def test_uniform_offset_passes_bounds_without_reference():
    # +12 m on every axis stays inside the generous bounds AND the expected
    # signed region AND self-consistency (spacing/polyline unchanged), so with
    # NO reference it PASSES. This documents the gap the reference closes.
    shifted = [(x + 12.0, y + 12.0, z + 12.0) for (x, y, z) in HEALTHY_POSITIONS]
    v = check_gate_map(_healthy(shifted))
    assert v.ok is True
    assert v.diagnosis == "ok"


def test_uniform_offset_caught_with_reference():
    shifted = [(x + 12.0, y + 12.0, z + 12.0) for (x, y, z) in HEALTHY_POSITIONS]
    v = check_gate_map(_healthy(shifted), reference=_healthy())
    assert v.ok is False
    assert v.diagnosis == "uniform_offset"
    assert v.suggested_correction is not None
    assert v.suggested_correction.startswith("subtract_offset")
    # The recovered offset vector should be ~(+12, +12, +12).
    off = v.details["offset_m"]
    assert off == pytest.approx([12.0, 12.0, 12.0], abs=1e-6)


# ---------------------------------------------------------------------------
# Single-gate outlier vs reference
# ---------------------------------------------------------------------------

def test_single_gate_outlier_vs_reference():
    bad = list(HEALTHY_POSITIONS)
    x, y, z = bad[3]
    bad[3] = (x + 8.0, y - 3.0, z + 2.0)  # one gate displaced, rest pristine
    v = check_gate_map(_healthy(bad), reference=_healthy())
    assert v.ok is False
    assert v.diagnosis == "reference_mismatch"
    assert v.details["worst_gate"] == 3


# ---------------------------------------------------------------------------
# Wrong gate count
# ---------------------------------------------------------------------------

def test_gate_count_too_few():
    v = check_gate_map(_healthy(HEALTHY_POSITIONS[:5]))
    assert v.ok is False
    assert v.diagnosis == "gate_count"
    assert v.details["count"] == 5


def test_gate_count_too_many():
    extra = list(HEALTHY_POSITIONS) + [(-175.0, -2.0, 26.0)]
    v = check_gate_map(_healthy(extra))
    assert v.ok is False
    assert v.diagnosis == "gate_count"
    assert v.details["count"] == 7


# ---------------------------------------------------------------------------
# Scrambled spacing — one gate teleported within bounds
# ---------------------------------------------------------------------------

def test_scrambled_spacing_one_gate_moved_within_bounds():
    # Move gate 2 by ~ -100 m in x (still inside the -300..20 box) so the legs
    # around it blow past the 60 m spacing ceiling. No reference needed.
    bad = list(HEALTHY_POSITIONS)
    x, y, z = bad[2]
    bad[2] = (x - 100.0, y, z)
    v = check_gate_map(_healthy(bad))
    assert v.ok is False
    assert v.diagnosis == "spacing_anomaly"


def test_scrambled_spacing_or_mismatch_with_reference():
    # Same teleport, but a reference is available: either spacing_anomaly fires
    # first (self-consistency runs before the reference) or it would be a
    # reference_mismatch. Assert it's caught and names one of the two.
    bad = list(HEALTHY_POSITIONS)
    x, y, z = bad[2]
    bad[2] = (x - 100.0, y, z)
    v = check_gate_map(_healthy(bad), reference=_healthy())
    assert v.ok is False
    assert v.diagnosis in {"spacing_anomaly", "reference_mismatch"}


# ---------------------------------------------------------------------------
# Empty / non-finite never raise
# ---------------------------------------------------------------------------

def test_empty_map():
    v = check_gate_map([])
    assert v.ok is False
    assert v.diagnosis == "empty"


def test_non_finite_does_not_raise():
    bad = list(HEALTHY_POSITIONS)
    bad[0] = (float("nan"), 0.0, 0.0)
    v = check_gate_map(_healthy(bad))
    assert v.ok is False
    assert v.diagnosis == "non_finite"


def test_out_of_bounds_far_away():
    # The historical ~1 km garbage signature.
    bad = list(HEALTHY_POSITIONS)
    bad[0] = (-918.0, 6.85, 577.0)
    v = check_gate_map(_healthy(bad))
    assert v.ok is False
    assert v.diagnosis == "out_of_bounds"


# ---------------------------------------------------------------------------
# JSON round-trip of a reference map
# ---------------------------------------------------------------------------

def test_reference_json_round_trip(tmp_path):
    path = tmp_path / "gate_map_reference.json"
    gates = _healthy()
    write_reference_json(gates, path)
    assert path.exists()

    loaded = read_reference_json(path)
    assert isinstance(loaded, list)
    assert len(loaded) == len(gates)
    assert all(isinstance(g, ReferenceGate) for g in loaded)
    for g, (x, y, z) in zip(loaded, HEALTHY_POSITIONS):
        assert g.position == pytest.approx((x, y, z))

    # The loaded reference round-trips through the checker: healthy map vs the
    # just-written reference passes; a uniform shift vs it is caught.
    assert check_gate_map(_healthy(), reference=loaded).ok is True
    shifted = [(x + 10.0, y, z) for (x, y, z) in HEALTHY_POSITIONS]
    assert check_gate_map(_healthy(shifted), reference=loaded).diagnosis == "uniform_offset"


def test_reference_json_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1}')  # missing 'gates'
    with pytest.raises(ValueError, match="missing required key 'gates'"):
        read_reference_json(bad)


def test_write_reference_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        write_reference_json([], tmp_path / "x.json")


# ---------------------------------------------------------------------------
# Bounds dataclass / custom bounds
# ---------------------------------------------------------------------------

def test_custom_bounds_superset_of_legacy():
    # Legacy box admits these; a tighter custom box rejects gate 0.
    legacy = check_gate_map(_healthy())
    assert legacy.ok is True
    tight = check_gate_map(_healthy(), bounds=Bounds(x_min=-100.0))
    assert tight.ok is False
    assert tight.diagnosis in {"out_of_bounds", "sign_flip_x"}


def test_verdict_is_dataclass_with_bool_contract():
    v = check_gate_map(_healthy())
    assert isinstance(v, GateMapVerdict)
    assert isinstance(v.ok, bool)
