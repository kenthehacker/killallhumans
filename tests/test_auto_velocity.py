"""
Tests for geometry-derived max_velocity (iter-006 Phase 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import pytest

from planning.auto_velocity import (
    DEFAULT_DRONE_MAX_ACCEL,
    DEFAULT_DRONE_MAX_SPEED_MPS,
    DEFAULT_SAFETY_FACTOR,
    derive_safe_max_velocity,
)

# Iter-009d: alias kept import-time so a future audit can `git grep`
# for the deprecated name. Tests below use the new name everywhere.
from planning.auto_velocity import DEFAULT_ABSOLUTE_CAP_MPS  # noqa: F401


@dataclass
class _StubGate:
    position: Tuple[float, float, float]


def test_fewer_than_3_gates_returns_cap():
    assert derive_safe_max_velocity([]) == DEFAULT_DRONE_MAX_SPEED_MPS
    assert derive_safe_max_velocity([_StubGate((0, 0, 0))]) == DEFAULT_DRONE_MAX_SPEED_MPS
    assert derive_safe_max_velocity([
        _StubGate((0, 0, 0)), _StubGate((5, 0, 0)),
    ]) == DEFAULT_DRONE_MAX_SPEED_MPS


def test_three_gates_in_a_straight_line_returns_cap():
    # No turn — should be unconstrained.
    gates = [
        _StubGate((0, 0, 2)), _StubGate((5, 0, 2)), _StubGate((10, 0, 2)),
    ]
    v = derive_safe_max_velocity(gates)
    assert v == pytest.approx(DEFAULT_DRONE_MAX_SPEED_MPS)


def test_tight_90_turn_three_gates_at_3m_spacing():
    """Slalom-like: 90° bend at gate-2, 3m spacing.
    r ≈ chord / (2·sin(45°)) = 3 / √2 ≈ 2.12 m
    v_max ≈ √(15·2.12) · 0.8 ≈ 4.6 m/s
    """
    gates = [
        _StubGate((0, 0, 2)),    # A
        _StubGate((3, 0, 2)),    # B — corner
        _StubGate((3, 3, 2)),    # C — 90° turn
    ]
    v = derive_safe_max_velocity(gates)
    # Expected: √(15 · 2.121) · 0.8 = √31.82 · 0.8 = 5.64 · 0.8 = 4.51 m/s
    expected = math.sqrt(DEFAULT_DRONE_MAX_ACCEL * (3.0 / (2.0 * math.sin(math.pi / 4)))) * DEFAULT_SAFETY_FACTOR
    assert v == pytest.approx(expected, rel=0.01)
    assert v < DEFAULT_DRONE_MAX_SPEED_MPS  # binding constraint
    assert 4.0 < v < 6.0  # rough sanity bound


def test_wide_gentle_bend_returns_cap_or_close():
    """Race_01-like: ~10m spacing with shallow bends → r large → v hits cap."""
    # 15° bend at 10m spacing — very gentle
    bend = math.radians(15.0)
    gates = [
        _StubGate((0, 0, 2)),
        _StubGate((10, 0, 2)),
        _StubGate((10 + 10 * math.cos(bend), 10 * math.sin(bend), 2)),
    ]
    v = derive_safe_max_velocity(gates)
    # Should hit the cap (or be very close to it)
    assert v >= DEFAULT_DRONE_MAX_SPEED_MPS * 0.9


def test_returns_min_over_all_triplets():
    """If one triplet has tight bend and others are loose, min wins."""
    # 3-gate setup: gates 1-2-3 wide, gates 2-3-4 tight.
    gates = [
        _StubGate((0, 0, 2)),
        _StubGate((10, 0, 2)),
        _StubGate((20, 0, 2)),
        _StubGate((20, 3, 2)),   # 90° at gate-3
    ]
    v = derive_safe_max_velocity(gates)
    # Tight 3-gate triplet (gate-2,3,4) dominates.
    assert v < DEFAULT_DRONE_MAX_SPEED_MPS


def test_zero_length_segments_skipped_safely():
    """Duplicate gates (length-0 segments) shouldn't crash with ZeroDivisionError."""
    gates = [
        _StubGate((0, 0, 2)),
        _StubGate((5, 0, 2)),
        _StubGate((5, 0, 2)),    # duplicate
        _StubGate((10, 0, 2)),
    ]
    v = derive_safe_max_velocity(gates)
    assert v > 0
    assert math.isfinite(v)


def test_safety_factor_scales_output():
    gates = [
        _StubGate((0, 0, 2)), _StubGate((3, 0, 2)), _StubGate((3, 3, 2)),
    ]
    v08 = derive_safe_max_velocity(gates, safety_factor=0.8)
    v04 = derive_safe_max_velocity(gates, safety_factor=0.4)
    assert v04 == pytest.approx(v08 * 0.5, rel=0.01)


def test_higher_drone_accel_yields_higher_velocity():
    gates = [
        _StubGate((0, 0, 2)), _StubGate((3, 0, 2)), _StubGate((3, 3, 2)),
    ]
    v_15 = derive_safe_max_velocity(gates, drone_max_accel=15.0)
    v_30 = derive_safe_max_velocity(gates, drone_max_accel=30.0)
    # v scales with √a, so 2× accel → √2× velocity
    assert v_30 == pytest.approx(v_15 * math.sqrt(2), rel=0.02)


def test_absolute_cap_respected_even_with_aggressive_settings():
    gates = [
        _StubGate((0, 0, 2)),
        _StubGate((100, 0, 2)),    # wide spacing
        _StubGate((100, 0.1, 2)),  # tiny bend
    ]
    v = derive_safe_max_velocity(gates, drone_max_accel=100.0, absolute_cap_mps=10.0)
    assert v <= 10.0
