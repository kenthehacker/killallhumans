"""Tests for the frozen-telemetry watchdog in RacePipeline.

A silently-dead telemetry feed used to be invisible: the controller kept
emitting the same command every tick (the "spinning in circles" failure
mode) with nothing flagging it. ``_check_telemetry_freshness`` now counts
consecutive ticks with a non-advancing telemetry timestamp and logs once
when the feed has clearly stalled.
"""

import logging

from competition.adapter import Quaternion, TelemetryState
from race_pipeline import PipelineConfig, RacePipeline


def _watchdog_stub(limit=50):
    pipe = RacePipeline.__new__(RacePipeline)
    pipe.config = PipelineConfig()
    pipe._last_telem_stamp_us = None
    pipe._telem_stale_ticks = 0
    pipe._telem_frozen_ticks = 0
    pipe._telem_stale_warned = False
    pipe._telem_stale_tick_limit = limit
    return pipe


def _telem(timestamp_us):
    return TelemetryState(
        timestamp_us=timestamp_us,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )


def test_advancing_timestamps_never_flag_stale():
    pipe = _watchdog_stub()
    for i in range(200):
        pipe._check_telemetry_freshness(_telem(1_000_000 + i * 10_000))
    assert pipe._telem_stale_ticks == 0
    assert pipe._telem_frozen_ticks == 0
    assert pipe._telem_stale_warned is False


def test_frozen_timestamp_counts_and_warns_once(caplog):
    pipe = _watchdog_stub(limit=50)
    # One fresh sample to seed the last-stamp, then the feed freezes.
    pipe._check_telemetry_freshness(_telem(1_000_000))
    assert pipe._telem_stale_ticks == 0

    with caplog.at_level(logging.ERROR):
        for _ in range(120):
            pipe._check_telemetry_freshness(_telem(1_000_000))

    # 120 frozen ticks accumulated, and the consecutive counter kept climbing.
    assert pipe._telem_frozen_ticks == 120
    assert pipe._telem_stale_ticks == 120
    # Warned exactly once despite 120 frozen ticks.
    frozen_errors = [r for r in caplog.records if "FROZEN" in r.getMessage()]
    assert len(frozen_errors) == 1
    assert pipe._telem_stale_warned is True


def test_recovery_resets_counter_and_rearms_warning(caplog):
    pipe = _watchdog_stub(limit=10)
    pipe._check_telemetry_freshness(_telem(1_000_000))
    with caplog.at_level(logging.ERROR):
        for _ in range(15):  # freeze -> warns once
            pipe._check_telemetry_freshness(_telem(1_000_000))
        assert pipe._telem_stale_warned is True

        # Feed recovers: a fresh timestamp resets the consecutive counter
        # and re-arms the warning for a future stall.
        pipe._check_telemetry_freshness(_telem(2_000_000))
        assert pipe._telem_stale_ticks == 0
        assert pipe._telem_stale_warned is False

        for _ in range(15):  # freeze again -> warns a second time
            pipe._check_telemetry_freshness(_telem(2_000_000))

    frozen_errors = [r for r in caplog.records if "FROZEN" in r.getMessage()]
    assert len(frozen_errors) == 2


def test_missing_timestamp_is_treated_as_frozen():
    pipe = _watchdog_stub(limit=5)
    for _ in range(8):
        pipe._check_telemetry_freshness(_telem(None))
    # None never "advances", so every tick counts as frozen.
    assert pipe._telem_frozen_ticks == 8
    assert pipe._telem_stale_warned is True
