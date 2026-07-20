"""Tests for the Phase 1.5/1.6 AIGP VQ1 runner (scripts/aigp_vq1_run.py).

Uses ``FakeAdapter`` to exercise the full connect→track-fetch→configure→
reset→run flow without the live sim or pymavlink.
"""
import asyncio
import math
import time

import pytest

import scripts.aigp_vq1_run as vq1_module
from scripts.aigp_vq1_run import FakeAdapter, run_vq1
from competition.session import MAX_RUN_DURATION_S, RaceSession
from competition.track_data import track_data_to_gatespecs


# ---------------------------------------------------------------------------
# FakeAdapter unit tests
# ---------------------------------------------------------------------------

def test_fake_adapter_returns_vq1_6_gates():
    async def _run():
        adapter = FakeAdapter()
        await adapter.connect("udpin:127.0.0.1:14550")
        assert adapter.is_connected
        track = await adapter.wait_for_track_data()
        assert track is not None
        assert track.num_gates == 6
    asyncio.run(_run())


def test_fake_adapter_gates_convert_to_gatespecs():
    async def _run():
        adapter = FakeAdapter()
        await adapter.connect()
        track = await adapter.wait_for_track_data()
        gates = track_data_to_gatespecs(track)
        assert len(gates) == 6
        for g in gates:
            assert hasattr(g, "position")
            assert hasattr(g, "yaw")
            # VQ1 gate quaternion (0.707,0,0,0.707) → local +Y maps to world -X → yaw ≈ π
            assert abs(g.yaw - math.pi) < 0.05, (
                f"VQ1 gate yaw should be ≈ π (facing -X); got {g.yaw:.4f}"
            )
    asyncio.run(_run())


def test_fake_adapter_reset_count():
    async def _run():
        adapter = FakeAdapter()
        await adapter.connect()
        await adapter.reset()
        await adapter.reset()
        assert adapter._reset_count == 2
    asyncio.run(_run())


def test_fake_adapter_connect_is_idempotent():
    async def _run():
        adapter = FakeAdapter()
        await adapter.connect("udpin:127.0.0.1:14550")
        first_track = adapter.track_data
        await adapter.connect("udpin:127.0.0.1:9999")
        assert adapter._connected_address == "udpin:127.0.0.1:14550", (
            "FakeAdapter.connect() should ignore second call"
        )
        assert adapter.track_data is first_track
    asyncio.run(_run())


def test_fake_adapter_drain_collisions_empty():
    adapter = FakeAdapter()
    assert adapter.drain_collisions() == []


def test_production_session_timeout_remains_eight_minutes():
    session = RaceSession(FakeAdapter())
    assert MAX_RUN_DURATION_S == 480
    assert session.max_run_duration_s == MAX_RUN_DURATION_S


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        pytest.param("max_run_duration_s", True, TypeError, id="duration-bool"),
        pytest.param("max_run_duration_s", "1", TypeError, id="duration-string"),
        pytest.param("max_run_duration_s", 0.0, ValueError, id="duration-zero"),
        pytest.param("max_run_duration_s", -1.0, ValueError, id="duration-negative"),
        pytest.param("max_run_duration_s", math.nan, ValueError, id="duration-nan"),
        pytest.param("max_run_duration_s", math.inf, ValueError, id="duration-infinite"),
        pytest.param("target_hz", False, TypeError, id="rate-bool"),
        pytest.param("target_hz", "100", TypeError, id="rate-string"),
        pytest.param("target_hz", 0.0, ValueError, id="rate-zero"),
        pytest.param("target_hz", math.nan, ValueError, id="rate-nan"),
    ],
)
def test_race_session_rejects_ambiguous_or_invalid_numeric_boundaries(
    keyword, value, error
):
    with pytest.raises(error, match=keyword):
        RaceSession(FakeAdapter(), **{keyword: value})


@pytest.mark.parametrize(
    ("value", "error"),
    [
        pytest.param(True, TypeError, id="bool"),
        pytest.param("0.1", TypeError, id="string"),
        pytest.param(0.0, ValueError, id="zero"),
        pytest.param(-0.1, ValueError, id="negative"),
        pytest.param(math.nan, ValueError, id="nan"),
        pytest.param(math.inf, ValueError, id="infinite"),
    ],
)
def test_run_vq1_rejects_invalid_programmatic_duration_before_adapter(
    monkeypatch, value, error
):
    def adapter_must_not_be_built():
        raise AssertionError("duration validation must precede adapter construction")

    monkeypatch.setattr(vq1_module, "FakeAdapter", adapter_must_not_be_built)
    with pytest.raises(error, match="max_seconds"):
        asyncio.run(run_vq1(dry_run=True, max_seconds=value))


# ---------------------------------------------------------------------------
# Full dry-run flow test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.timeout(15)
def test_dry_run_full_flow(monkeypatch, tmp_path):
    """``run_vq1(dry_run=True)`` exercises a bounded offline flow."""
    record_path = tmp_path / "dry-run-telemetry.jsonl.gz"
    adapter = FakeAdapter()
    monkeypatch.setattr(vq1_module, "FakeAdapter", lambda: adapter)
    original_wait_for = asyncio.wait_for
    outer_guard_timed_out = [False]

    async def tracked_wait_for(awaitable, timeout):
        try:
            return await original_wait_for(awaitable, timeout)
        except asyncio.TimeoutError:
            outer_guard_timed_out[0] = True
            raise

    monkeypatch.setattr(vq1_module.asyncio, "wait_for", tracked_wait_for)
    started = time.monotonic()
    asyncio.run(
        run_vq1(
            dry_run=True,
            max_speed=4.0,
            record=str(record_path),
            max_seconds=0.10,
            minimal=True,
        )
    )
    assert time.monotonic() - started < 10.0
    assert not outer_guard_timed_out[0]
    assert adapter._attitude_send_count > 0
    assert adapter._reset_count >= 2
    assert not adapter.is_connected
    assert record_path.is_file()
    assert record_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Call-order assertion via instrumented FakeAdapter
# ---------------------------------------------------------------------------

class OrderCheckAdapter(FakeAdapter):
    """Records the order of key calls for sequencing assertions."""

    def __init__(self):
        super().__init__()
        self.call_log = []

    async def connect(self, address="udpin:127.0.0.1:14550"):
        self.call_log.append("connect")
        await super().connect(address)

    async def wait_for_track_data(self, timeout_s=10.0):
        self.call_log.append("wait_for_track_data")
        return await super().wait_for_track_data(timeout_s)

    async def reset(self):
        self.call_log.append("reset")
        return await super().reset()

    async def send_attitude(self, cmd):
        if "send_attitude" not in self.call_log:
            self.call_log.append("send_attitude")
        await super().send_attitude(cmd)


@pytest.mark.slow
@pytest.mark.timeout(15)
def test_call_order_connect_track_configure_reset_before_send():
    """connect → wait_for_track_data → (configure) → reset → send_attitude."""
    adapter = OrderCheckAdapter()

    from race_pipeline import PipelineConfig, RacePipeline

    async def _instrumented_run():
        await adapter.connect()
        track = await adapter.wait_for_track_data()
        gates = track_data_to_gatespecs(track)
        telem = await adapter.get_telemetry()
        pipeline = RacePipeline(
            adapter,
            PipelineConfig(max_speed=4.0, minimal_control=True),
        )
        pipeline.configure(gates, start_position=telem.position_ned)
        await adapter.reset()
        await pipeline.run(
            address="udpin:127.0.0.1:14550",
            max_run_duration_s=0.10,
        )

    asyncio.run(_instrumented_run())

    log = adapter.call_log
    assert "send_attitude" in log, "bounded flow must still exercise command send"
    assert log.index("connect") < log.index("wait_for_track_data"), "connect must precede wait_for_track_data"
    assert log.index("wait_for_track_data") < log.index("reset"), "wait_for_track_data must precede reset"
    reset_idx = log.index("reset")
    if "send_attitude" in log:
        assert reset_idx < log.index("send_attitude"), "reset must precede first send_attitude"
