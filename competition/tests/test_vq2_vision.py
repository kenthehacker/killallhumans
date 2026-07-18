"""Focused tests for the build-3385 duplicate-suppressing vision thread.

No test opens a real network socket or sends simulator traffic.  Datagrams are
fed directly or through an in-memory fake socket.
"""

from __future__ import annotations

import queue
import socket
import threading
import time

import cv2
import numpy as np
import pytest

import competition.vq2_vision as vq2_vision_module
from competition.vision_udp import encode_packet
from competition.vq2_vision import VQ2VisionThread


def _jpeg_packets(
    *,
    frame_id: int = 7,
    sim_time_ns: int = 1_000_000_000,
    total_chunks: int = 3,
):
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[120:220, 270:370] = (0, 0, 255)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()
    chunk_size = (len(jpeg) + total_chunks - 1) // total_chunks
    packets = []
    for chunk_id in range(total_chunks):
        payload = jpeg[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        packets.append(encode_packet(
            frame_id=frame_id,
            chunk_id=chunk_id,
            total_chunks=total_chunks,
            jpeg_size=len(jpeg),
            sim_time_ns=sim_time_ns,
            payload=payload,
        ))
    return packets


def test_rejects_duplicate_chunk_keys_before_reassembly_and_decodes_once():
    receiver = VQ2VisionThread()
    packets = _jpeg_packets(total_chunks=3)

    # Build 3385 sends about 56 copies of every chunk.  The underlying
    # reassembler must see only one copy of each key.
    for packet in packets:
        for _ in range(56):
            receiver.feed_datagram(packet, received_monotonic_s=10.0)

    snapshot = receiver.snapshot()
    assert snapshot is not None
    assert snapshot.frame_id == 7
    assert snapshot.camera_frame.image.shape == (360, 640, 3)

    stats = receiver.stats()
    assert stats.datagrams_received == 3 * 56
    assert stats.unique_datagrams == 3
    assert stats.duplicate_datagrams == 3 * 55
    assert stats.frames_reassembled == 1
    assert stats.frames_decoded == 1
    assert stats.receiver_duplicate_chunks == 0
    assert stats.receiver_dropped_late_packets == 0


def test_snapshot_freshness_and_reset_accept_restarted_frame_ids():
    receiver = VQ2VisionThread()
    packet = _jpeg_packets(frame_id=19, total_chunks=1)[0]

    first = receiver.feed_datagram(packet, received_monotonic_s=20.0)
    assert first is not None
    assert receiver.is_fresh(0.20, now_monotonic_s=20.19)
    assert receiver.snapshot(max_age_s=0.20, now_monotonic_s=20.21) is None

    # Same frame/chunk is an early duplicate until the SIM_RESET boundary.
    assert receiver.feed_datagram(packet, received_monotonic_s=21.0) is None
    receiver.reset()
    assert receiver.snapshot() is None
    assert not receiver.is_fresh(1.0, now_monotonic_s=21.0)

    restarted = receiver.feed_datagram(packet, received_monotonic_s=22.0)
    assert restarted is not None
    assert restarted.frame_id == 19
    assert restarted.generation == first.generation + 1
    assert receiver.stats().resets == 1


def test_malformed_first_copy_does_not_poison_duplicate_key():
    receiver = VQ2VisionThread()
    good = _jpeg_packets(frame_id=23, total_chunks=1)[0]

    # Retain the same frame/chunk key but make payload_size inconsistent.
    malformed = bytearray(good)
    # payload_size is the u32 beginning at byte offset 12.
    malformed[12:16] = (999_999).to_bytes(4, "little")
    assert receiver.feed_datagram(bytes(malformed)) is None
    assert receiver.feed_datagram(good) is not None

    stats = receiver.stats()
    assert stats.malformed_datagrams == 1
    assert stats.unique_datagrams == 1
    assert stats.duplicate_datagrams == 0


def test_reset_prevents_in_progress_old_generation_decode_from_publishing(monkeypatch):
    receiver = VQ2VisionThread()
    packet = _jpeg_packets(frame_id=31, total_chunks=1)[0]
    decode_started = threading.Event()
    allow_decode = threading.Event()
    real_decode = vq2_vision_module.decode_jpeg_to_camera_frame

    def delayed_decode(frame):
        decode_started.set()
        assert allow_decode.wait(timeout=1.0)
        return real_decode(frame)

    monkeypatch.setattr(
        vq2_vision_module,
        "decode_jpeg_to_camera_frame",
        delayed_decode,
    )
    worker = threading.Thread(target=receiver.feed_datagram, args=(packet,))
    worker.start()
    assert decode_started.wait(timeout=1.0)

    receiver.reset()
    allow_decode.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert receiver.snapshot() is None
    assert receiver.stats().reset_generation_drops == 1


def test_older_frame_completing_late_cannot_regress_published_snapshot():
    receiver = VQ2VisionThread()
    newer = _jpeg_packets(frame_id=50, sim_time_ns=2_000, total_chunks=1)[0]
    older = _jpeg_packets(frame_id=49, sim_time_ns=1_000, total_chunks=1)[0]

    assert receiver.feed_datagram(newer) is not None
    assert receiver.feed_datagram(older) is None

    snapshot = receiver.snapshot()
    assert snapshot is not None
    assert snapshot.frame_id == 50
    stats = receiver.stats()
    assert stats.frames_reassembled == 2
    assert stats.frames_decoded == 1
    assert stats.out_of_order_frame_drops == 1


class _FakeSocket:
    _CLOSED = object()

    def __init__(self):
        self.incoming = queue.Queue()
        self.bound = None
        self.timeout = None
        self.options = []
        self.closed = False

    def setsockopt(self, *args):
        self.options.append(args)

    def settimeout(self, timeout):
        self.timeout = timeout

    def bind(self, address):
        self.bound = address

    def recvfrom(self, _size):
        try:
            item = self.incoming.get(timeout=0.02)
        except queue.Empty as exc:
            raise socket.timeout() from exc
        if item is self._CLOSED:
            raise OSError("fake socket closed")
        return item, ("127.0.0.1", 5601)

    def close(self):
        if not self.closed:
            self.closed = True
            self.incoming.put(self._CLOSED)

    def push(self, packet):
        self.incoming.put(packet)


def test_start_stop_are_idempotent_and_fake_socket_thread_publishes_frame():
    fake = _FakeSocket()
    receiver = VQ2VisionThread(
        port=45678,
        bind_host="127.0.0.1",
        socket_factory=lambda: fake,
    )

    receiver.stop()  # stop-before-start is a no-op
    receiver.start()
    receiver.start()  # no second socket/thread
    assert receiver.is_running
    assert fake.bound == ("127.0.0.1", 45678)
    assert fake.options

    fake.push(_jpeg_packets(frame_id=41, total_chunks=1)[0])
    deadline = time.monotonic() + 1.0
    while receiver.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert receiver.snapshot() is not None
    assert receiver.snapshot().frame_id == 41
    receiver.stop()
    receiver.stop()
    assert not receiver.is_running
    assert fake.closed


@pytest.mark.parametrize("max_age", [-1.0, float("nan"), float("inf")])
def test_freshness_rejects_invalid_max_age(max_age):
    receiver = VQ2VisionThread()
    with pytest.raises(ValueError):
        receiver.is_fresh(max_age)
