"""
Vision UDP receiver tests (iter-001 A10).

Covers the VADR-TS-002 §4.6 wire format:
  - 24-byte little-endian header
  - chunked JPEG reassembly
  - out-of-order delivery
  - partial-frame timeout
  - duplicate-chunk idempotency
  - duplicate-frame-id post-completion
"""
from __future__ import annotations

import struct

import pytest

from competition.aigp_geometry import AIGP_CAM_UDP_PORT
from competition.vision_udp import (
    DEFAULT_REASSEMBLY_TIMEOUT_MS,
    HEADER_FMT,
    HEADER_SIZE,
    VisionUdpReceiver,
    encode_packet,
    parse_packet,
)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

def test_header_size_is_24_bytes():
    assert HEADER_SIZE == 24


def test_header_fmt_is_little_endian():
    # AIGP spec calls out little-endian explicitly.
    assert HEADER_FMT.startswith("<")


def test_encode_then_parse_roundtrip():
    payload = b"\xff\xd8\xff" + b"\x00" * 100  # JPEG magic + filler
    raw = encode_packet(
        frame_id=42, chunk_id=2, total_chunks=4, jpeg_size=400,
        sim_time_ns=10**9, payload=payload,
    )
    assert len(raw) == HEADER_SIZE + len(payload)
    pkt = parse_packet(raw)
    assert pkt.frame_id == 42
    assert pkt.chunk_id == 2
    assert pkt.total_chunks == 4
    assert pkt.jpeg_size == 400
    assert pkt.payload_size == len(payload)
    assert pkt.sim_time_ns == 10**9
    assert pkt.payload == payload


def test_parse_too_short_raises():
    with pytest.raises(ValueError):
        parse_packet(b"\x00" * (HEADER_SIZE - 1))


def test_parse_payload_size_mismatch_raises():
    # Header claims 100-byte payload but we only attach 50 bytes.
    bad_header = struct.pack(HEADER_FMT, 1, 0, 1, 100, 100, 0)
    with pytest.raises(ValueError):
        parse_packet(bad_header + b"\x00" * 50)


def test_parse_chunk_id_out_of_range_raises():
    bad_header = struct.pack(HEADER_FMT, 1, 5, 3, 10, 10, 0)
    with pytest.raises(ValueError):
        parse_packet(bad_header + b"\x00" * 10)


def test_parse_zero_total_chunks_raises():
    bad_header = struct.pack(HEADER_FMT, 1, 0, 0, 10, 10, 0)
    with pytest.raises(ValueError):
        parse_packet(bad_header + b"\x00" * 10)


# ---------------------------------------------------------------------------
# Iter-001 review Opus F9 (a): zero-size payload must be rejected
# ---------------------------------------------------------------------------

def test_parse_zero_payload_size_raises():
    """A chunk with payload_size=0 is malformed — the encoder should never
    emit empty chunks. Accepting them would produce a JPEG buffer with
    zero-filled holes, which cv2.imdecode silently fails on."""
    bad = struct.pack(HEADER_FMT, 1, 0, 1, 10, 0, 0)
    with pytest.raises(ValueError, match="zero"):
        parse_packet(bad)


# ---------------------------------------------------------------------------
# Iter-001 review Opus F9 (b) + 7-way MAJOR: jpeg_size must equal the sum
# of chunk payload sizes after assembly
# ---------------------------------------------------------------------------

def test_assembled_payload_smaller_than_jpeg_size_is_dropped():
    """Three chunks whose total payload < jpeg_size produces a zero-tailed
    buffer. cv2.imdecode silently fails on that; the receiver must drop
    the frame, not propagate corrupt bytes."""
    r = VisionUdpReceiver()
    # 3 chunks of total payload 6 bytes, but jpeg_size says 10.
    chunks_payloads = [b"\xff\xd8\xff", b"\xab\xcd", b"\x99"]
    for cid, pl in enumerate(chunks_payloads):
        raw = encode_packet(
            frame_id=99, chunk_id=cid, total_chunks=3,
            jpeg_size=10,                # claims 10 bytes total
            sim_time_ns=42, payload=pl,  # but chunks sum to 6
        )
        result = r.feed_packet(raw)
    # The last feed_packet must NOT return a corrupt frame.
    assert result is None, (
        "size-mismatched assembly must be dropped, not yielded"
    )
    assert r.delivered_frames == 0
    assert r.dropped_partial_frames >= 1


def test_assembled_payload_larger_than_jpeg_size_is_dropped():
    """Two chunks whose payload > jpeg_size means the encoder lied; drop."""
    r = VisionUdpReceiver()
    chunks_payloads = [b"\xff\xd8\xff" + b"\xab" * 5, b"\xcd" * 5]  # total 13
    for cid, pl in enumerate(chunks_payloads):
        raw = encode_packet(
            frame_id=100, chunk_id=cid, total_chunks=2,
            jpeg_size=8,                  # claims 8 bytes total
            sim_time_ns=43, payload=pl,
        )
        result = r.feed_packet(raw)
    assert result is None
    assert r.delivered_frames == 0
    assert r.dropped_partial_frames >= 1


def test_assembled_payload_equals_jpeg_size_succeeds():
    """Sanity: a correctly-sized frame still works after validation."""
    r = VisionUdpReceiver()
    chunks_payloads = [b"\xff\xd8\xff\xab", b"\xcd\xef\x99\xd9"]
    total = sum(len(c) for c in chunks_payloads)
    last = None
    for cid, pl in enumerate(chunks_payloads):
        raw = encode_packet(
            frame_id=101, chunk_id=cid, total_chunks=2,
            jpeg_size=total, sim_time_ns=44, payload=pl,
        )
        last = r.feed_packet(raw) or last
    assert last is not None
    assert last.jpeg_bytes == b"\xff\xd8\xff\xab\xcd\xef\x99\xd9"


# ---------------------------------------------------------------------------
# Receiver default port and config
# ---------------------------------------------------------------------------

def test_receiver_defaults_to_aigp_port_5600():
    r = VisionUdpReceiver()
    assert r.port == AIGP_CAM_UDP_PORT == 5600


# ---------------------------------------------------------------------------
# In-order reassembly
# ---------------------------------------------------------------------------

def _chunked_jpeg(frame_id: int, jpeg: bytes, total_chunks: int, sim_time_ns: int):
    """Split `jpeg` into `total_chunks` evenly-sized chunks and yield
    (chunk_id, raw_packet) pairs in chunk-id order."""
    chunk_size = (len(jpeg) + total_chunks - 1) // total_chunks
    for cid in range(total_chunks):
        payload = jpeg[cid * chunk_size:(cid + 1) * chunk_size]
        yield cid, encode_packet(
            frame_id=frame_id, chunk_id=cid, total_chunks=total_chunks,
            jpeg_size=len(jpeg), sim_time_ns=sim_time_ns, payload=payload,
        )


def test_in_order_chunks_reassemble_to_original_jpeg():
    r = VisionUdpReceiver()
    jpeg = b"\xff\xd8" + bytes(range(256)) * 4  # 1026 bytes
    chunks = list(_chunked_jpeg(frame_id=7, jpeg=jpeg, total_chunks=4, sim_time_ns=1000))
    out = None
    for cid, raw in chunks:
        out = r.feed_packet(raw) or out
    assert out is not None
    assert out.frame_id == 7
    assert out.sim_time_ns == 1000
    assert out.jpeg_bytes == jpeg
    assert r.delivered_frames == 1


# ---------------------------------------------------------------------------
# Out-of-order reassembly
# ---------------------------------------------------------------------------

def test_out_of_order_chunks_still_yield_one_frame():
    r = VisionUdpReceiver()
    jpeg = b"\xff\xd8" + bytes(range(256)) * 2
    chunks = list(_chunked_jpeg(frame_id=11, jpeg=jpeg, total_chunks=4, sim_time_ns=2000))
    # Deliver in [2, 0, 3, 1] order.
    order = [chunks[2], chunks[0], chunks[3], chunks[1]]
    out = None
    for cid, raw in order:
        result = r.feed_packet(raw)
        out = result or out
    assert out is not None
    assert out.jpeg_bytes == jpeg
    assert r.delivered_frames == 1


# ---------------------------------------------------------------------------
# Partial-frame timeout
# ---------------------------------------------------------------------------

def test_partial_frame_gcs_after_timeout():
    r = VisionUdpReceiver(reassembly_timeout_ms=10)  # 10 ms = 10_000_000 ns
    jpeg = b"\xff\xd8" + bytes(50)
    chunks = list(_chunked_jpeg(frame_id=1, jpeg=jpeg, total_chunks=4, sim_time_ns=0))
    # Feed 3 of 4 chunks.
    for cid, raw in chunks[:3]:
        r.feed_packet(raw)
    assert r.delivered_frames == 0
    # Feed an unrelated chunk far in the future — that triggers GC of
    # the stale buffer.
    far_future = encode_packet(
        frame_id=2, chunk_id=0, total_chunks=1, jpeg_size=4,
        sim_time_ns=100_000_000, payload=b"\xff\xd8\xff\xd9",  # tiny SOI+EOI JPEG
    )
    r.feed_packet(far_future)
    assert r.dropped_partial_frames >= 1


# ---------------------------------------------------------------------------
# Duplicate chunks within a frame
# ---------------------------------------------------------------------------

def test_duplicate_chunk_within_frame_does_not_break_assembly():
    r = VisionUdpReceiver()
    jpeg = b"\xff\xd8" + bytes(range(64))
    chunks = list(_chunked_jpeg(frame_id=5, jpeg=jpeg, total_chunks=3, sim_time_ns=500))
    # Deliver chunk 0 twice plus 1 and 2.
    sequence = [chunks[0], chunks[0], chunks[1], chunks[2]]
    out = None
    for cid, raw in sequence:
        result = r.feed_packet(raw)
        out = result or out
    assert out is not None
    assert out.jpeg_bytes == jpeg
    assert r.duplicate_chunks == 1


# ---------------------------------------------------------------------------
# Duplicate frame_id after completion
# ---------------------------------------------------------------------------

def test_duplicate_frame_id_after_completion_is_dropped():
    r = VisionUdpReceiver()
    jpeg = b"\xff\xd8" + bytes(8)
    chunks = list(_chunked_jpeg(frame_id=9, jpeg=jpeg, total_chunks=2, sim_time_ns=300))
    out = None
    for cid, raw in chunks:
        out = r.feed_packet(raw) or out
    assert out is not None
    assert r.delivered_frames == 1

    # Re-send chunk 0 of frame 9 — must NOT re-emit.
    again = r.feed_packet(chunks[0][1])
    assert again is None
    assert r.delivered_frames == 1
    assert r.dropped_late_packets >= 1


# ---------------------------------------------------------------------------
# pop_latest_frame
# ---------------------------------------------------------------------------

def test_pop_latest_frame_returns_most_recent_completion():
    r = VisionUdpReceiver()
    jpeg_a = b"\xff\xd8" + bytes(16)
    jpeg_b = b"\xff\xd8" + bytes(32)
    for _cid, raw in _chunked_jpeg(frame_id=1, jpeg=jpeg_a, total_chunks=2, sim_time_ns=100):
        r.feed_packet(raw)
    assert r.pop_latest_frame().jpeg_bytes == jpeg_a
    for _cid, raw in _chunked_jpeg(frame_id=2, jpeg=jpeg_b, total_chunks=2, sim_time_ns=200):
        r.feed_packet(raw)
    latest = r.pop_latest_frame()
    assert latest is not None
    assert latest.frame_id == 2
    assert latest.jpeg_bytes == jpeg_b


# ---------------------------------------------------------------------------
# Eviction under buffer pressure
# ---------------------------------------------------------------------------

def test_max_buffered_frames_evicts_oldest():
    r = VisionUdpReceiver(reassembly_timeout_ms=DEFAULT_REASSEMBLY_TIMEOUT_MS * 1000,
                          max_buffered_frames=2)
    # Start three in-flight frames; the first must get evicted.
    for fid, ts in [(1, 10), (2, 20), (3, 30)]:
        pkt = encode_packet(
            frame_id=fid, chunk_id=0, total_chunks=2, jpeg_size=16,
            sim_time_ns=ts, payload=b"\xff\xd8" + bytes(6),
        )
        r.feed_packet(pkt)
    assert r.dropped_partial_frames >= 1
    assert 1 not in r._buffers  # frame 1 evicted
    assert 3 in r._buffers
