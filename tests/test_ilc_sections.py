"""
Tests for the curvature-derived ILC section partition (iter-001 A8).

These tests pin the contract of `planning.ilc_sections.derive_section_boundaries`
so the benchmark refactor (A9) has a concrete bar to clear:
  - smooth tracks fall back to a single global section
  - tracks with curvature peaks get multiple sections
  - all sections together cover [0, n_total_steps] without gap or overlap
  - section count is capped per config
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Duck-typed trajectory stubs — keep the tests independent of
# RaceTrajectory's full surface.
# ---------------------------------------------------------------------------

@dataclass
class _StubPoint:
    time: float
    acceleration: Tuple[float, float, float]


@dataclass
class _StubTraj:
    points: List[_StubPoint]
    total_time: float


def _flat_traj(n_points: int = 200, dt_pt: float = 0.01) -> _StubTraj:
    """Zero acceleration everywhere — a perfectly smooth path."""
    return _StubTraj(
        points=[
            _StubPoint(time=i * dt_pt, acceleration=(0.0, 0.0, 0.0))
            for i in range(n_points)
        ],
        total_time=n_points * dt_pt,
    )


def _spiky_traj(
    n_points: int = 1000, dt_pt: float = 0.01, peak_step: int = 500,
    peak_width: int = 50, peak_mag: float = 5.0, base_mag: float = 0.05,
) -> _StubTraj:
    """Mostly flat, with one acceleration spike in the middle."""
    return _StubTraj(
        points=[
            _StubPoint(
                time=i * dt_pt,
                acceleration=(
                    0.0,
                    0.0,
                    peak_mag if abs(i - peak_step) < peak_width else base_mag,
                ),
            )
            for i in range(n_points)
        ],
        total_time=n_points * dt_pt,
    )


def _alternating_spiky(
    n_points: int = 2000, dt_pt: float = 0.01, run_len: int = 80,
) -> _StubTraj:
    """Many alternating high/low runs — exercises the n_sections_max cap."""
    pts = []
    for i in range(n_points):
        high = (i // run_len) % 2 == 0
        a = (5.0, 0.0, 0.0) if high else (0.05, 0.0, 0.0)
        pts.append(_StubPoint(time=i * dt_pt, acceleration=a))
    return _StubTraj(points=pts, total_time=n_points * dt_pt)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def test_load_ilc_config_returns_required_keys():
    from planning.ilc_sections import load_ilc_config
    cfg = load_ilc_config()
    for key in ("global", "sections", "partition"):
        assert key in cfg, f"missing required top-level key {key!r}"
    for hp in ("alpha", "max_iterations", "max_correction_m",
               "convergence_threshold", "filter_cutoff_hz", "momentum_gamma"):
        assert hp in cfg["global"], f"missing global hyperparam {hp!r}"
    for cls in ("low", "high"):
        assert cls in cfg["sections"], f"missing section class {cls!r}"
        for fld in ("alpha", "max_correction_m", "filter_cutoff_hz", "vel_scale"):
            assert fld in cfg["sections"][cls], f"missing {cls}.{fld!r}"


# ---------------------------------------------------------------------------
# Smooth track → single section
# ---------------------------------------------------------------------------

def test_smooth_trajectory_returns_single_low_section():
    from planning.ilc_sections import derive_section_boundaries, load_ilc_config
    cfg = load_ilc_config()
    sections = derive_section_boundaries(_flat_traj(), dt=0.01, config=cfg)
    assert len(sections) == 1
    s, e, alpha, max_corr, cutoff, vel_scale = sections[0]
    assert s == 0
    assert e > 0
    low = cfg["sections"]["low"]
    assert alpha == pytest.approx(low["alpha"])
    assert max_corr == pytest.approx(low["max_correction_m"])
    assert cutoff == pytest.approx(low["filter_cutoff_hz"])
    assert vel_scale == pytest.approx(low["vel_scale"])


def test_empty_trajectory_returns_safe_single_section():
    from planning.ilc_sections import derive_section_boundaries, load_ilc_config
    cfg = load_ilc_config()
    empty = _StubTraj(points=[], total_time=0.0)
    # Passing n_total_steps explicitly so we don't divide-by-dt-of-zero.
    sections = derive_section_boundaries(
        empty, dt=0.01, config=cfg, n_total_steps=500,
    )
    assert len(sections) == 1
    assert sections[0][0] == 0
    assert sections[0][1] >= 1


# ---------------------------------------------------------------------------
# Spiky track → multiple sections with at least one "high" class
# ---------------------------------------------------------------------------

def test_spiky_trajectory_yields_multiple_sections():
    from planning.ilc_sections import derive_section_boundaries
    sections = derive_section_boundaries(_spiky_traj(), dt=0.01)
    assert len(sections) >= 2


def test_spiky_section_contains_high_class_tuple_values():
    """At least one section should match the 'high' class hyperparameters."""
    from planning.ilc_sections import derive_section_boundaries, load_ilc_config
    cfg = load_ilc_config()
    high = cfg["sections"]["high"]
    sections = derive_section_boundaries(_spiky_traj(), dt=0.01, config=cfg)
    high_match = any(
        s[2] == pytest.approx(high["alpha"])
        and s[3] == pytest.approx(high["max_correction_m"])
        and s[5] == pytest.approx(high["vel_scale"])
        for s in sections
    )
    assert high_match, (
        "expected at least one section to use the 'high' class tuple; "
        f"got sections {sections}"
    )


# ---------------------------------------------------------------------------
# Geometric invariants
# ---------------------------------------------------------------------------

def test_sections_cover_full_step_range_without_gap():
    from planning.ilc_sections import derive_section_boundaries
    traj = _spiky_traj()
    n_total_steps = int(traj.total_time / 0.01) + 50
    sections = derive_section_boundaries(traj, dt=0.01)
    assert sections[0][0] == 0, "first section must start at 0"
    assert sections[-1][1] == n_total_steps, (
        f"last section must end at {n_total_steps}, got {sections[-1][1]}"
    )
    for a, b in zip(sections, sections[1:]):
        assert a[1] == b[0], (
            f"sections must abut without gap or overlap: {a} then {b}"
        )


def test_sections_are_monotonic_and_non_empty():
    from planning.ilc_sections import derive_section_boundaries
    sections = derive_section_boundaries(_spiky_traj(), dt=0.01)
    for s in sections:
        assert s[1] > s[0], f"section has non-positive length: {s}"


def test_section_count_capped_to_n_sections_max():
    from planning.ilc_sections import derive_section_boundaries, load_ilc_config
    cfg = load_ilc_config()
    n_max = cfg["partition"]["n_sections_max"]
    sections = derive_section_boundaries(_alternating_spiky(), dt=0.01, config=cfg)
    assert len(sections) <= n_max, (
        f"expected at most {n_max} sections, got {len(sections)}: {sections}"
    )


def test_section_tuples_are_six_element_for_compute_ilc_offset_table():
    """`compute_ilc_offset_table` accepts (start, end, alpha, max_corr, cutoff, vel_scale)."""
    from planning.ilc_sections import derive_section_boundaries
    sections = derive_section_boundaries(_spiky_traj(), dt=0.01)
    for s in sections:
        assert len(s) == 6, f"section tuple must have 6 elements: {s}"
        s_start, s_end, alpha, max_corr, cutoff, vel = s
        assert isinstance(s_start, int) and isinstance(s_end, int)
        assert isinstance(alpha, float) and 0 < alpha < 1
        assert isinstance(max_corr, float) and max_corr > 0
        assert isinstance(cutoff, float) and cutoff > 0
        assert isinstance(vel, float)
