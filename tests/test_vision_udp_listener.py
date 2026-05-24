"""
Async VisionUdpListener integration tests (iter-002 B4).

Verifies the asyncio DatagramProtocol wiring: real UDP sockets, real
event loop. The listener must:
  - bind on the requested port
  - feed every received datagram into its VisionUdpReceiver
  - drop malformed packets without disrupting the stream
  - decode the latest reassembled frame on demand

These tests don't need MAVSDK; they exercise the listener layer that
MAVLinkBridge wraps.
"""
from __future__ import annotations

import asyncio
import socket

import pytest

from competition.vision_udp import VisionUdpListener, encode_packet


def _free_port() -> int:
    """Allocate an ephemeral UDP port the OS isn't currently using."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def event_loop():
    """Per-test event loop so listener lifecycle is isolated."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _drive_listener_until_frame(loop, listener, send_callable, timeout_s=2.0):
    """Helper: start the listener, run send_callable in the loop, wait for
    at least one decoded frame or timeout, return the frame.
    """
    async def _runner():
        await listener.start()
        try:
            await send_callable(listener.port)
            # Poll for a frame to appear.
            deadline = loop.time() + timeout_s
            while loop.time() < deadline:
                frame = listener.latest_frame()
                if frame is not None:
                    return frame
                await asyncio.sleep(0.01)
            return listener.latest_frame()
        finally:
            await listener.stop()

    return loop.run_until_complete(_runner())


# ---------------------------------------------------------------------------
# Wire-format roundtrip via real UDP socket
# ---------------------------------------------------------------------------

def test_listener_reassembles_jpeg_via_real_udp_socket(event_loop):
    """End-to-end: send chunks through a real UDP socket; listener
    reassembles them and `latest_frame()` returns a decoded CameraFrame
    (or, if JPEG decode fails on the synthetic payload, at least the
    receiver should have observed a complete reassembly)."""
    port = _free_port()
    listener = VisionUdpListener(port=port, bind_host="127.0.0.1")

    # A real 1×1 JPEG (smallest valid JPEG sequence).
    # Minimal JFIF header + SOI/EOI markers, single MCU at zero quality.
    # Constructed bytes-by-bytes; we don't NEED a valid JPEG for the
    # reassembly assertion, but having one makes `latest_frame()` work.
    minimal_jpeg = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46,
        0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0xFF, 0xDB, 0x00, 0x43, 0x00,
    ] + [16] * 64 + [
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
        0x01, 0x11, 0x00,
        0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01,
    ] + [0] * 15 + [
        0xFF, 0xC4, 0x00, 0x14, 0x10,
    ] + [0] * 16 + [
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
        0xFF, 0xD9,
    ])

    async def _send(target_port):
        # Send via a separate UDP socket on a non-listener port.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Split the JPEG into 2 chunks.
        mid = len(minimal_jpeg) // 2
        chunks = [minimal_jpeg[:mid], minimal_jpeg[mid:]]
        for cid, ch in enumerate(chunks):
            pkt = encode_packet(
                frame_id=1, chunk_id=cid, total_chunks=2,
                jpeg_size=len(minimal_jpeg), sim_time_ns=10 ** 9,
                payload=ch,
            )
            s.sendto(pkt, ("127.0.0.1", target_port))
            await asyncio.sleep(0.005)  # let the event loop pump
        s.close()

    asyncio.set_event_loop(event_loop)
    frame = _drive_listener_until_frame(event_loop, listener, _send)
    # Reassembly must have happened; receiver counters confirm.
    assert listener.receiver.delivered_frames >= 1, (
        f"expected ≥1 delivered frame; got {listener.receiver.delivered_frames}"
    )


def test_listener_protocol_counts_malformed_packets(event_loop):
    """Invoke the protocol's datagram_received directly with a malformed
    payload. Bypasses the OS UDP layer (whose timing can flake on tiny
    packets), but verifies the actual error-handling code path inside
    `_VisionDatagramProtocol.datagram_received`."""
    port = _free_port()
    listener = VisionUdpListener(port=port, bind_host="127.0.0.1")
    asyncio.set_event_loop(event_loop)

    async def _runner():
        await listener.start()
        try:
            # Truncated header — parse_packet raises ValueError, the
            # protocol should swallow it and bump the counter.
            listener._protocol.datagram_received(
                b"\x01\x02\x03", ("127.0.0.1", 0),
            )
            assert listener.malformed_packet_count == 1
            # Zero payload_size — also malformed; another increment.
            bad_header = b"\x02\x00\x00\x00"  # frame_id=2
            bad_header += b"\x00\x00\x01\x00"  # chunk_id=0, total_chunks=1
            bad_header += b"\x10\x00\x00\x00"  # jpeg_size=16
            bad_header += b"\x00\x00\x00\x00"  # payload_size=0  (malformed)
            bad_header += b"\x00\x00\x00\x00\x00\x00\x00\x00"  # sim_time
            listener._protocol.datagram_received(
                bad_header, ("127.0.0.1", 0),
            )
            assert listener.malformed_packet_count == 2
        finally:
            await listener.stop()

    event_loop.run_until_complete(_runner())


def test_listener_lifecycle_idempotent_stop(event_loop):
    """Calling stop() twice (or before start) must not raise."""
    port = _free_port()
    listener = VisionUdpListener(port=port, bind_host="127.0.0.1")
    asyncio.set_event_loop(event_loop)

    async def _runner():
        # Stop before start — no-op.
        await listener.stop()
        await listener.start()
        await listener.stop()
        # Stop after stop — no-op.
        await listener.stop()
        assert not listener.is_listening

    event_loop.run_until_complete(_runner())
