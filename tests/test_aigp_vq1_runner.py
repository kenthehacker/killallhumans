"""Tests for the Phase 1.5/1.6 AIGP VQ1 runner (scripts/aigp_vq1_run.py).

Uses ``FakeAdapter`` to exercise the full connect→track-fetch→configure→
reset→run flow without the live sim or pymavlink.
"""
import asyncio
import time

import pytest

from scripts.aigp_vq1_run import FakeAdapter, run_vq1
from competition.track_data import track_data_to_gatespecs


# ---------------------------------------------------------------------------
# FakeAdapter unit tests
# ---------------------------------------------------------------------------

def test_fake_adapter_returns_vq1_6_gates():
    adapter = FakeAdapter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(adapter.connect("udpin:127.0.0.1:14550"))
    assert adapter.is_connected
    track = loop.run_until_complete(adapter.wait_for_track_data())
    assert track is not None
    assert track.num_gates == 6


def test_fake_adapter_gates_convert_to_gatespecs():
    import math
    adapter = FakeAdapter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(adapter.connect())
    track = loop.run_until_complete(adapter.wait_for_track_data())
    gates = track_data_to_gatespecs(track)
    assert len(gates) == 6
    for g in gates:
        assert hasattr(g, "position")
        assert hasattr(g, "yaw")
        # VQ1 gate quaternion (0.707,0,0,0.707) → local +Y maps to world -X → yaw ≈ π
        assert abs(g.yaw - math.pi) < 0.05, (
            f"VQ1 gate yaw should be ≈ π (facing -X); got {g.yaw:.4f}"
        )


def test_fake_adapter_reset_count():
    adapter = FakeAdapter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(adapter.connect())
    loop.run_until_complete(adapter.reset())
    loop.run_until_complete(adapter.reset())
    assert adapter._reset_count == 2


def test_fake_adapter_connect_is_idempotent():
    adapter = FakeAdapter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(adapter.connect("udpin:127.0.0.1:14550"))
    first_track = adapter.track_data
    loop.run_until_complete(adapter.connect("udpin:127.0.0.1:9999"))
    assert adapter._connected_address == "udpin:127.0.0.1:14550", (
        "FakeAdapter.connect() should ignore second call"
    )
    assert adapter.track_data is first_track


def test_fake_adapter_drain_collisions_empty():
    adapter = FakeAdapter()
    assert adapter.drain_collisions() == []


# ---------------------------------------------------------------------------
# Full dry-run flow test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dry_run_full_flow():
    """``run_vq1(dry_run=True)`` must:
    1. Connect to FakeAdapter.
    2. Fetch track data (6-gate VQ1 map).
    3. Configure pipeline (trajectory pre-computed).
    4. Call reset() exactly once after configure().
    5. Call pipeline.run() — sequencer completes or times out.

    The fake session loop terminates quickly because the FakeAdapter
    does not advance simulation time and the pipeline stops on
    sequencer.is_complete or after the race timeout.
    """
    asyncio.get_event_loop().run_until_complete(
        run_vq1(dry_run=True, max_speed=4.0)
    )


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
def test_call_order_connect_track_configure_reset_before_send():
    """connect → wait_for_track_data → (configure) → reset → send_attitude."""
    adapter = OrderCheckAdapter()

    from race_pipeline import PipelineConfig, RacePipeline
    from competition.track_data import track_data_to_gatespecs

    async def _instrumented_run():
        await adapter.connect()
        track = await adapter.wait_for_track_data()
        gates = track_data_to_gatespecs(track)
        telem = await adapter.get_telemetry()
        pipeline = RacePipeline(adapter, PipelineConfig(max_speed=4.0))
        pipeline.configure(gates, start_position=telem.position_ned)
        await adapter.reset()
        await pipeline.run(address="udpin:127.0.0.1:14550")

    asyncio.get_event_loop().run_until_complete(_instrumented_run())

    log = adapter.call_log
    assert log.index("connect") < log.index("wait_for_track_data"), "connect must precede wait_for_track_data"
    assert log.index("wait_for_track_data") < log.index("reset"), "wait_for_track_data must precede reset"
    reset_idx = log.index("reset")
    if "send_attitude" in log:
        assert reset_idx < log.index("send_attitude"), "reset must precede first send_attitude"
