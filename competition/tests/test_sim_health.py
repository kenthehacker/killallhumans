"""Tests for competition.sim_health — the sim-degradation / health-probe monitor
(race-day reliability item 1, half 2).

Each case proves a specific part of the documented degradation signature from
``docs/aigp/2026-06-16-speed-and-spline-handoff.md`` ("Operational notes"):

* a healthy climb (peak ~-1.7) PASSES;
* an over-climb (peak <= -2.4) is flagged "over_climb" with the right peak;
* a high collision count within the window is flagged "excessive_collisions";
* too-few / no samples -> "insufficient_data" (does NOT raise, NOT alarming);
* a borderline peak just inside -2.4 stays healthy (threshold boundary);
* the streaming SimHealthProbe matches the batch probe_health on the same data;
* SIGN-CONVENTION GUARD: a DESCENDING stream (positive Z) is NOT flagged
  over-climb — only excessive negative-Z climb trips it.

Fully offline: synthetic ``(t_s, z_ned)`` samples, numpy/stdlib only, no sim.
"""
import math

import pytest

from competition.sim_health import (
    DIAGNOSES,
    HEALTHY_REF_Z,
    PEAK_CLIMB_WARN_Z,
    SimHealthProbe,
    SimHealthVerdict,
    probe_health,
)


# ---------------------------------------------------------------------------
# Synthetic sample builders
# ---------------------------------------------------------------------------

def _stream(peak_z: float, *, hz: float = 100.0, window_s: float = 3.0):
    """A first-window flight stream whose MOST-NEGATIVE Z (peak climb) is
    ``peak_z``. Models the start climb-out: spawn at Z=0, climb (negative Z) to
    ``peak_z`` at mid-window, then ease back a little — exactly the transient the
    probe watches. Returns a list of (t_s, z_ned)."""
    n = int(window_s * hz) + 1
    samples = []
    for i in range(n):
        t = i / hz
        frac = i / (n - 1)  # 0..1 over the window
        # Half-sine climb-out: 0 -> peak at the middle -> ~0.6*peak at the end.
        # min(z) over the stream is exactly peak_z (at frac=0.5).
        z = peak_z * math.sin(math.pi * min(frac, 0.5) / 0.5) if frac <= 0.5 else \
            peak_z * (0.6 + 0.4 * math.cos(math.pi * (frac - 0.5) / 0.5))
        samples.append((t, z))
    return samples


def _descending_stream(depth_z: float = 5.0, *, hz: float = 100.0, window_s: float = 3.0):
    """A stream that DESCENDS the whole time: Z goes 0 -> +depth_z (down in NED).
    Its min(z) is ~0 (it never climbs), so it must read healthy — this is the
    sign-convention guard."""
    n = int(window_s * hz) + 1
    return [(i / hz, depth_z * (i / (n - 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# Healthy
# ---------------------------------------------------------------------------

def test_healthy_climb_passes():
    # Peak climb ~-1.7 (the handoff's healthy reference) => healthy.
    v = probe_health(_stream(-1.7))
    assert v.healthy is True
    assert v.diagnosis == "healthy"
    assert v.degraded is False
    # Peak recorded in details is the most-negative Z (~-1.7).
    assert v.details["peak_climb_z"] == pytest.approx(-1.7, abs=1e-6)


def test_healthy_with_a_couple_collisions_still_passes():
    # 1-2 early collisions are below the "wildly high" threshold => still healthy.
    v = probe_health(_stream(-1.6), collisions=2)
    assert v.healthy is True
    assert v.diagnosis == "healthy"
    assert v.details["collisions"] == 2


# ---------------------------------------------------------------------------
# Over-climb (the PRIMARY degradation signature)
# ---------------------------------------------------------------------------

def test_over_climb_flagged():
    # Peak climb -2.6 (<= -2.4) => degraded / over_climb, peak in details.
    v = probe_health(_stream(-2.6))
    assert v.healthy is False
    assert v.diagnosis == "over_climb"
    assert v.degraded is True
    assert v.details["peak_climb_z"] == pytest.approx(-2.6, abs=1e-6)
    # Message names the signal AND the action (restart .exe, not SIM_RESET).
    msg = v.message.lower()
    assert "peak climb" in msg
    assert "-2.6" in v.message
    assert ".exe" in msg
    assert "sim_reset will not fix" in msg


def test_over_climb_takes_priority_over_collisions():
    # Both signals fire; the PRIMARY (over_climb) wins, collisions still recorded.
    v = probe_health(_stream(-3.0), collisions=10)
    assert v.diagnosis == "over_climb"
    assert v.details["collisions"] == 10


# ---------------------------------------------------------------------------
# Threshold boundary
# ---------------------------------------------------------------------------

def test_peak_exactly_at_threshold_is_degraded():
    # Boundary is inclusive on the degraded side: peak == -2.4 => over_climb.
    v = probe_health(_stream(PEAK_CLIMB_WARN_Z))
    assert v.healthy is False
    assert v.diagnosis == "over_climb"


def test_peak_just_inside_threshold_stays_healthy():
    # A peak just shallower than -2.4 (e.g. -2.39) must stay healthy — proves
    # the boundary doesn't false-trip on ordinary climb jitter near the edge.
    v = probe_health(_stream(-2.39))
    assert v.healthy is True
    assert v.diagnosis == "healthy"
    assert v.details["peak_climb_z"] == pytest.approx(-2.39, abs=1e-6)


# ---------------------------------------------------------------------------
# Excessive collisions (the SECONDARY tell)
# ---------------------------------------------------------------------------

def test_excessive_collisions_flagged():
    # Clean climb (-1.5, well shy of -2.4) but a wildly high collision count
    # within the window => excessive_collisions.
    v = probe_health(_stream(-1.5), collisions=8)
    assert v.healthy is False
    assert v.diagnosis == "excessive_collisions"
    assert v.degraded is True
    assert v.details["collisions"] == 8
    assert ".exe" in v.message.lower()


def test_collisions_at_threshold_not_flagged():
    # Exactly at the threshold (3) is NOT excessive (trigger is strictly greater).
    v = probe_health(_stream(-1.5), collisions=3)
    assert v.healthy is True
    assert v.diagnosis == "healthy"


# ---------------------------------------------------------------------------
# Insufficient data — must NOT raise, must be non-alarming
# ---------------------------------------------------------------------------

def test_no_samples_is_insufficient_data():
    v = probe_health([])  # must not raise
    assert isinstance(v, SimHealthVerdict)
    assert v.healthy is False
    assert v.diagnosis == "insufficient_data"
    # NON-alarming: degraded is False, so the runner won't cry "restart .exe".
    assert v.degraded is False
    assert v.details["peak_climb_z"] is None
    assert v.details["n_samples"] == 0


def test_too_few_samples_is_insufficient_data():
    # A handful of points (below MIN_SAMPLES) -> insufficient_data, not a verdict
    # read off one or two points.
    v = probe_health([(0.0, -1.7), (0.01, -1.8)])
    assert v.healthy is False
    assert v.diagnosis == "insufficient_data"
    assert v.degraded is False


def test_window_not_covered_is_insufficient_data():
    # Plenty of samples, but they only span ~0.5 s of a 3 s window (run ended
    # early) => insufficient_data, NOT a premature healthy/degraded read.
    samples = [(i / 100.0, -1.5) for i in range(51)]  # 0.50 s @ 100 Hz
    v = probe_health(samples, window_s=3.0)
    assert v.healthy is False
    assert v.diagnosis == "insufficient_data"
    assert v.degraded is False


def test_non_finite_samples_do_not_raise():
    # A NaN/inf frame in the stream must not crash the probe; the bad frame is
    # dropped and the rest still yields a verdict.
    samples = _stream(-1.7)
    samples[10] = (samples[10][0], float("nan"))
    samples[20] = (samples[20][0], float("inf"))
    v = probe_health(samples)  # must not raise
    assert isinstance(v, SimHealthVerdict)
    assert v.diagnosis in DIAGNOSES


def test_all_diagnoses_are_in_canonical_set():
    for s, c in [(_stream(-1.7), 0), (_stream(-3.0), 0),
                 (_stream(-1.5), 9), ([], 0)]:
        assert probe_health(s, collisions=c).diagnosis in DIAGNOSES


# ---------------------------------------------------------------------------
# Sign-convention guard — DESCENT is not over-climb
# ---------------------------------------------------------------------------

def test_descending_stream_is_not_over_climb():
    # A stream that only ever DESCENDS (Z positive, down in NED) has min(z) ~ 0,
    # which is NOT <= -2.4, so it must NOT be flagged over_climb. This is the
    # sign guard: only EXCESS CLIMB (negative Z) trips the probe, never descent.
    v = probe_health(_descending_stream(depth_z=8.0))
    assert v.healthy is True
    assert v.diagnosis == "healthy"
    # Peak "climb" is ~0 (the start), never negative enough to trip.
    assert v.details["peak_climb_z"] >= -1e-6


def test_large_positive_z_never_trips_over_climb():
    # Even a huge POSITIVE Z (deep descent) must not read as over-climb — proves
    # the threshold is not accidentally comparing magnitude / abs(z).
    samples = [(i / 100.0, 50.0 * (i / 300.0)) for i in range(301)]
    v = probe_health(samples)
    assert v.diagnosis == "healthy"


# ---------------------------------------------------------------------------
# Streaming SimHealthProbe == batch probe_health
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("peak_z,collisions", [(-1.7, 0), (-2.6, 0), (-1.5, 8)])
def test_streaming_matches_batch(peak_z, collisions):
    samples = _stream(peak_z)
    batch = probe_health(samples, collisions=collisions)

    probe = SimHealthProbe()
    for t, z in samples:
        probe.add_sample(t, z)
    probe.add_collisions(collisions)
    # window_elapsed becomes true once we pass window_s past the first sample.
    assert probe.window_elapsed(samples[-1][0]) is True
    stream = probe.evaluate()

    assert stream.healthy == batch.healthy
    assert stream.diagnosis == batch.diagnosis
    assert stream.details["peak_climb_z"] == pytest.approx(batch.details["peak_climb_z"])
    assert stream.details["collisions"] == batch.details["collisions"]


def test_streaming_is_fire_once_and_idempotent():
    probe = SimHealthProbe()
    for t, z in _stream(-2.6):
        probe.add_sample(t, z)
    v1 = probe.evaluate()
    assert probe.done is True
    # Late samples/collisions after evaluate() are ignored; verdict is latched.
    probe.add_sample(99.0, -10.0)
    probe.add_collisions(100)
    v2 = probe.evaluate()
    assert v2 is v1
    assert v2.diagnosis == "over_climb"


def test_streaming_window_elapsed_false_before_any_sample():
    probe = SimHealthProbe()
    assert probe.window_elapsed(100.0) is False
    # And evaluating with nothing buffered is the non-alarming insufficient_data.
    v = probe.evaluate()
    assert v.diagnosis == "insufficient_data"
    assert v.degraded is False


def test_streaming_bounded_buffer_drops_out_of_window_samples():
    # Samples well past the window are dropped from the buffer (bounded memory)
    # and do not affect the verdict.
    probe = SimHealthProbe(window_s=3.0)
    for t, z in _stream(-1.7):
        probe.add_sample(t, z)
    # A late deep-climb sample at t=10 s must NOT change the healthy verdict.
    probe.add_sample(10.0, -9.0)
    v = probe.evaluate()
    assert v.diagnosis == "healthy"


# ---------------------------------------------------------------------------
# Verdict dataclass contract
# ---------------------------------------------------------------------------

def test_verdict_is_dataclass_with_bool_contract():
    v = probe_health(_stream(-1.7))
    assert isinstance(v, SimHealthVerdict)
    assert isinstance(v.healthy, bool)
    assert isinstance(v.degraded, bool)


def test_custom_thresholds_are_honoured():
    # Pass a stricter degraded threshold so a -1.9 peak (healthy by default)
    # now trips — proves the thresholds are real parameters, not baked literals.
    v = probe_health(_stream(-1.9), peak_climb_warn_z=-1.8)
    assert v.diagnosis == "over_climb"
    assert v.details["peak_climb_warn_z"] == -1.8
