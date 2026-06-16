"""iter-38: tests for the divergence / inner-loop-instability guard.

The guard aborts a run that has gone unstable (sustained high gyro — the
vert_gain=2.0 limit-cycle tumble) or flown off the course box, before it logs
garbage telemetry or wedges the sim. The two decision helpers are pure and
tested directly; the consecutive-tick trip logic is replayed against the real
iter-37 captures (skipped if those captures are not present).
"""
import gzip
import json
import math
import os

import pytest

from race_pipeline import (
    _COURSE_BOX_NED,
    _GUARD_TRIP_TICKS,
    _GYRO_INSTABILITY_RADS,
    _gyro_unstable,
    _in_course_box,
)

CAP = "captures"


def test_in_course_box_accepts_real_course_points():
    # origin (spawn), a mid-course gate, and the finish (~2 m past gate5) are IN.
    assert _in_course_box((0.0, 0.0, 0.0))
    assert _in_course_box((-74.6, 1.2, 12.8))     # gate 2
    assert _in_course_box((-162.0, -4.4, 25.1))   # just past the final gate


def test_in_course_box_rejects_flyaway_and_nonfinite():
    assert not _in_course_box((-300.0, 0.0, 0.0))           # flew past finish
    assert not _in_course_box((-50.0, 40.0, 0.0))           # huge +Y
    assert not _in_course_box((-50.0, 0.0, 60.0))           # huge +Z
    assert not _in_course_box((float("nan"), 0.0, 0.0))     # non-finite
    assert not _in_course_box((float("inf"), 0.0, 0.0))
    # the box must actually bound the real course (not be degenerate)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = _COURSE_BOX_NED
    assert xlo < -162.0 and xhi > 0.0 and yhi >= 10.0 and zhi >= 27.0


def test_gyro_unstable_threshold_and_nonfinite():
    assert not _gyro_unstable((0.5, 0.5, 0.5))                  # normal flight
    assert not _gyro_unstable((2.0, 0.0, 0.0))                  # worst clean ~2.2
    assert not _gyro_unstable(None)                             # no gyro: abstain
    assert _gyro_unstable((5.0, 0.0, 0.0))                      # > 4.0 rad/s
    assert _gyro_unstable((float("nan"), 0.0, 0.0))            # non-finite -> unstable
    # threshold is between the clean max (~2.2) and the divergence (~8.4)
    assert 2.2 < _GYRO_INSTABILITY_RADS < 8.4


def _replay_trips(path):
    """Replay the guard's consecutive-tick counter over a capture; return the
    tick index it would trip at, or None if it never trips."""
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    ticks = 0
    for i, r in enumerate(rows):
        if _gyro_unstable(r.get("gyro")) or not _in_course_box(r["pos"]):
            ticks += 1
            if ticks >= _GUARD_TRIP_TICKS:
                return i
        else:
            ticks = 0
    return None


@pytest.mark.parametrize("name,should_trip", [
    ("iter37_c100_vg2.jsonl.gz", True),    # vert_gain=2.0 divergence (gyro 8.4)
    ("iter37_c90.jsonl.gz", False),        # clean cruise 9.0
    ("iter37_c90_rep.jsonl.gz", False),    # 84-collision but NOT unstable (gyro 2.2)
    ("iter37_c80_r2.jsonl.gz", False),     # clean recommended config
    ("iter36_min70_baseline.jsonl.gz", False),
])
def test_guard_replay_against_real_captures(name, should_trip):
    path = os.path.join(CAP, name)
    if not os.path.exists(path):
        pytest.skip(f"capture {name} not present")
    trip = _replay_trips(path)
    if should_trip:
        assert trip is not None, f"{name}: guard should have tripped on divergence"
    else:
        assert trip is None, f"{name}: guard FALSE-tripped at tick {trip}"
