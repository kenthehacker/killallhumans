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
from competition.vq2_contracts import validate_frame_timing_sequence
from competition.vq2_runtime import (
    VQ2_HOST_CLOCK_ID,
    latency_events_from_frame_timings,
)
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
    assert snapshot.timing is not None
    assert snapshot.timing.host_clock_id == VQ2_HOST_CLOCK_ID
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
    assert stats.timing_ledger_entries == 0
    assert stats.timing_ledger_high_watermark == 1
    assert stats.timing_ledger_capacity == receiver.max_remembered_chunks
    assert stats.timing_overflow_latched is False
    assert stats.receiver_buffered_partial_frames == 0
    assert stats.receiver_buffer_high_watermark == 1
    assert stats.receiver_buffer_capacity == receiver.max_buffered_frames


def test_snapshot_callback_receives_every_published_decoded_frame_once():
    delivered = []
    receiver = VQ2VisionThread(
        on_snapshot=delivered.append,
        capture_snapshot_queue_enabled=True,
    )
    for frame_id in (7, 8):
        packet = _jpeg_packets(
            frame_id=frame_id,
            sim_time_ns=frame_id * 1_000,
            total_chunks=1,
        )[0]
        assert receiver.feed_datagram(packet) is not None
        assert receiver.feed_datagram(packet) is None
    assert [(item.frame_id, item.sim_time_ns) for item in delivered] == [
        (7, 7_000),
        (8, 8_000),
    ]
    assert receiver.stats().frames_decoded == len(delivered)
    assert delivered[0].camera_frame.image.flags.writeable is False
    with pytest.raises(ValueError):
        delivered[0].camera_frame.image[0, 0, 0] = 255
    assert receiver.capture_snapshot_queue_depth() == 2
    assert receiver.pop_capture_snapshot().frame_id == 7
    assert receiver.pop_capture_snapshot().frame_id == 8
    assert receiver.pop_capture_snapshot() is None
    stats = receiver.stats()
    assert stats.capture_snapshot_queue_entries == 0
    assert stats.capture_snapshot_queue_high_watermark == 2
    assert stats.capture_snapshot_queue_capacity == 256
    assert stats.capture_snapshot_queue_dropped == 0


def test_capture_snapshot_queue_overflow_is_latched_in_diagnostics():
    delivered = []
    receiver = VQ2VisionThread(
        on_snapshot=delivered.append,
        capture_snapshot_queue_enabled=True,
        capture_snapshot_queue_capacity=1,
    )
    for frame_id in (7, 8):
        packet = _jpeg_packets(
            frame_id=frame_id,
            sim_time_ns=frame_id * 1_000,
            total_chunks=1,
        )[0]
        assert receiver.feed_datagram(packet) is not None

    stats = receiver.stats()
    assert len(delivered) == 2
    assert stats.capture_snapshot_queue_entries == 1
    assert stats.capture_snapshot_queue_high_watermark == 1
    assert stats.capture_snapshot_queue_dropped == 1
    assert stats.processing_errors == 1


def test_capture_snapshot_queue_requires_explicit_callback_and_exact_flag():
    with pytest.raises(ValueError, match="requires an on_snapshot"):
        VQ2VisionThread(capture_snapshot_queue_enabled=True)
    with pytest.raises(TypeError, match="exact bool"):
        VQ2VisionThread(
            on_snapshot=lambda _snapshot: None,
            capture_snapshot_queue_enabled=1,
        )


def test_published_snapshot_has_complete_same_clock_frame_timing():
    clock_values = iter((130, 140, 150, 160))
    receiver = VQ2VisionThread(
        stream_id="camera0",
        host_clock_id="host-monotonic-1",
        monotonic_ns=lambda: next(clock_values),
    )
    packets = _jpeg_packets(frame_id=7, sim_time_ns=9_000, total_chunks=3)
    assert receiver.feed_datagram(packets[0], received_monotonic_ns=100) is None
    assert receiver.feed_datagram(packets[1], received_monotonic_ns=110) is None
    snapshot = receiver.feed_datagram(packets[2], received_monotonic_ns=120)

    assert snapshot is not None
    timing = snapshot.timing
    assert timing is not None
    assert timing.identity.stream_id == "camera0"
    assert timing.identity.generation == snapshot.generation
    assert timing.identity.frame_id == snapshot.frame_id
    assert timing.camera_source_time_ns == snapshot.sim_time_ns == 9_000
    assert timing.publication_sequence == 1
    assert (
        timing.first_unique_packet_monotonic_ns,
        timing.final_unique_packet_monotonic_ns,
        timing.reassembly_complete_monotonic_ns,
        timing.decode_start_monotonic_ns,
        timing.decode_end_monotonic_ns,
        timing.publish_monotonic_ns,
    ) == (100, 120, 130, 140, 150, 160)
    validate_frame_timing_sequence((timing,))
    assert len(latency_events_from_frame_timings((timing,))) == 6


def test_publication_sequence_survives_reset_while_generation_scopes_reused_id():
    clock_values = iter((20, 21, 22, 23, 30, 31, 32, 33))
    receiver = VQ2VisionThread(
        stream_id="camera0",
        host_clock_id="host-monotonic-1",
        monotonic_ns=lambda: next(clock_values),
    )
    packet = _jpeg_packets(frame_id=19, sim_time_ns=5, total_chunks=1)[0]
    first = receiver.feed_datagram(packet, received_monotonic_ns=10)
    assert first is not None and first.timing is not None

    receiver.reset()
    restarted = receiver.feed_datagram(packet, received_monotonic_ns=25)
    assert restarted is not None and restarted.timing is not None
    assert restarted.generation == first.generation + 1
    assert restarted.frame_id == first.frame_id
    assert restarted.timing.publication_sequence == 2
    validate_frame_timing_sequence((first.timing, restarted.timing))


def test_feed_rejects_ambiguous_or_regressing_host_timestamp_inputs():
    packet = _jpeg_packets(total_chunks=1)[0]
    dual_clock = iter((3, 4, 5, 6))
    dual = VQ2VisionThread(monotonic_ns=lambda: next(dual_clock))
    snapshot = dual.feed_datagram(
        packet,
        received_monotonic_s=1.0,
        received_monotonic_ns=2,
    )
    assert snapshot is not None and snapshot.timing is not None
    assert snapshot.received_monotonic_s == 1.0
    assert snapshot.timing.first_unique_packet_monotonic_ns == 2

    receiver = VQ2VisionThread(monotonic_ns=lambda: 1)
    with pytest.raises(ValueError, match=">= 0"):
        receiver.feed_datagram(packet, received_monotonic_s=-1.0)
    with pytest.raises(TypeError, match="not bool"):
        receiver.feed_datagram(packet, received_monotonic_s=True)

    # The injected host clock moving behind packet arrival fails the frozen
    # same-clock ordering instead of clamping or inventing timing evidence.
    with pytest.raises(ValueError, match="monotonic"):
        receiver.feed_datagram(packet, received_monotonic_ns=2)


def test_nonincreasing_publish_clock_drops_frame_without_advancing_sequence():
    clock_values = iter((2, 3, 4, 10, 11, 12, 13, 10))
    receiver = VQ2VisionThread(monotonic_ns=lambda: next(clock_values))
    first = receiver.feed_datagram(
        _jpeg_packets(frame_id=1, sim_time_ns=10, total_chunks=1)[0],
        received_monotonic_ns=1,
    )
    assert first is not None and first.timing is not None
    assert first.timing.publication_sequence == 1

    dropped = receiver.feed_datagram(
        _jpeg_packets(frame_id=2, sim_time_ns=20, total_chunks=1)[0],
        received_monotonic_ns=9,
    )
    assert dropped is None
    assert receiver.snapshot() is first
    assert receiver.stats().processing_errors == 1


def test_partial_frame_timing_overflow_latches_stale_until_reset():
    clock_values = iter((100, 101, 102, 103))
    receiver = VQ2VisionThread(
        max_buffered_frames=2,
        max_remembered_chunks=1,
        monotonic_ns=lambda: next(clock_values),
    )
    first = _jpeg_packets(frame_id=1, sim_time_ns=10, total_chunks=2)
    second = _jpeg_packets(frame_id=2, sim_time_ns=20, total_chunks=2)
    assert receiver.feed_datagram(first[0], received_monotonic_ns=1) is None
    assert receiver.feed_datagram(second[0], received_monotonic_ns=2) is None
    assert receiver.feed_datagram(first[1], received_monotonic_ns=3) is None
    assert receiver.snapshot() is None
    assert receiver.stats().processing_errors == 1

    receiver.reset()
    recovered = receiver.feed_datagram(
        _jpeg_packets(frame_id=1, sim_time_ns=1, total_chunks=1)[0],
        received_monotonic_ns=99,
    )
    assert recovered is not None


def test_snapshot_callback_failure_is_counted_without_losing_publication():
    def broken(_snapshot):
        raise RuntimeError("capture callback failed")

    receiver = VQ2VisionThread(on_snapshot=broken)
    snapshot = receiver.feed_datagram(_jpeg_packets(total_chunks=1)[0])
    assert snapshot is not None
    assert receiver.snapshot() is snapshot
    assert receiver.stats().snapshot_callback_errors == 1


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


def test_stop_fails_if_decode_thread_survives_and_success_proves_no_late_callback(monkeypatch):
    delivered = []
    receiver = VQ2VisionThread(on_snapshot=delivered.append)
    packet = _jpeg_packets(frame_id=32, total_chunks=1)[0]
    decode_started = threading.Event()
    allow_decode = threading.Event()
    real_decode = vq2_vision_module.decode_jpeg_to_camera_frame

    def delayed_decode(frame):
        decode_started.set()
        assert allow_decode.wait(timeout=2.0)
        return real_decode(frame)

    monkeypatch.setattr(vq2_vision_module, "decode_jpeg_to_camera_frame", delayed_decode)
    worker = threading.Thread(target=receiver.feed_datagram, args=(packet,))
    with receiver._lifecycle_lock:
        receiver._thread = worker
    worker.start()
    assert decode_started.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="did not terminate"):
        receiver.stop(timeout_s=0.01)
    assert delivered == []
    allow_decode.set()
    worker.join(timeout=1.0)
    receiver.stop(timeout_s=0.1)
    delivered_after_success = len(delivered)
    time.sleep(0.02)
    assert len(delivered) == delivered_after_success


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


def test_bounded_packet_caches_cannot_republish_one_generation_frame_identity():
    receiver = VQ2VisionThread(
        max_buffered_frames=1,
        max_remembered_chunks=1,
    )
    for frame_id in range(1, 11):
        snapshot = receiver.feed_datagram(
            _jpeg_packets(
                frame_id=frame_id,
                sim_time_ns=frame_id * 1_000,
                total_chunks=1,
            )[0]
        )
        assert snapshot is not None

    # Both the packet-key cache and the underlying delivered-ID window have
    # forgotten frame 1 by now. The lifetime generation identity has not.
    repeated = receiver.feed_datagram(
        _jpeg_packets(frame_id=1, sim_time_ns=20_000, total_chunks=1)[0]
    )
    assert repeated is None
    assert receiver.snapshot().frame_id == 10
    stats = receiver.stats()
    assert stats.frames_decoded == 10
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


class _FakePoweredSocket(_FakeSocket):
    def __init__(
        self,
        *,
        actual=("0.0.0.0", 49_152),
        family=socket.AF_INET,
        close_error=False,
    ):
        super().__init__()
        self.family = family
        self.actual = actual
        self.bind_calls = []
        self.close_error = close_error

    def getsockname(self):
        return self.actual

    def bind(self, address):
        self.bind_calls.append(address)
        raise AssertionError(
            "exclusive socket factory must return an already-bound socket"
        )

    def close(self):
        if self.close_error:
            raise OSError("injected powered socket close failure")
        super().close()

    def recvfrom(self, _size):
        try:
            item = self.incoming.get(timeout=0.02)
        except queue.Empty as exc:
            raise socket.timeout() from exc
        if item is self._CLOSED:
            raise OSError("fake socket closed")
        return item

    def push_from(self, packet, addr):
        self.incoming.put((packet, addr))


def _wait_until(predicate, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


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


def test_powered_vision_requires_one_unambiguous_exclusive_factory():
    with pytest.raises(ValueError, match="requires an exclusive_socket_factory"):
        VQ2VisionThread(powered_exclusive=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        VQ2VisionThread(
            powered_exclusive=True,
            socket_factory=_FakeSocket,
            exclusive_socket_factory=lambda _host, _port: _FakePoweredSocket(),
        )
    with pytest.raises(ValueError, match="only in powered mode"):
        VQ2VisionThread(
            exclusive_socket_factory=lambda _host, _port: _FakePoweredSocket()
        )


@pytest.mark.parametrize(
    ("fake", "message"),
    [
        (_FakePoweredSocket(family=socket.AF_INET6), "must be AF_INET"),
        (_FakePoweredSocket(actual=("127.0.0.1", 49_152)), "bind host changed"),
        (_FakePoweredSocket(actual=("0.0.0.0", 0)), "ephemeral port is invalid"),
    ],
)
def test_powered_vision_closes_socket_when_bind_proof_fails(fake, message):
    receiver = VQ2VisionThread(
        port=0,
        powered_exclusive=True,
        exclusive_socket_factory=lambda _host, _port: fake,
    )
    with pytest.raises(OSError, match=message):
        receiver.start()
    assert fake.closed
    assert receiver.source_diagnostics().state == "not_opened"


def test_powered_vision_freezes_valid_loopback_peer_and_rejects_before_feed():
    fake = _FakePoweredSocket()
    factory_calls = []

    def exclusive_factory(host, port):
        factory_calls.append((host, port))
        return fake

    receiver = VQ2VisionThread(
        port=0,
        powered_exclusive=True,
        exclusive_socket_factory=exclusive_factory,
    )
    with pytest.raises(RuntimeError, match="socket source gate"):
        receiver.feed_datagram(_jpeg_packets(total_chunks=1)[0])
    initial = receiver.source_diagnostics()
    assert initial.state == "not_opened"
    assert initial.actual_host is None
    assert initial.frozen_peer is None
    assert initial.socket_policy == "ipv4-exclusive-address-use"

    receiver.start()
    assert factory_calls == [("0.0.0.0", 0)]
    assert fake.bind_calls == []
    bound = receiver.source_diagnostics()
    assert (bound.actual_host, bound.actual_port) == ("0.0.0.0", 49_152)
    assert bound.state == "bound"

    # A malformed packet cannot choose the peer or reach production
    # reassembly, but remains visible in cumulative malformed diagnostics.
    fake.push_from(b"not-a-frame-chunk", ("127.0.0.1", 50_001))
    _wait_until(lambda: receiver.stats().malformed_datagrams == 1)
    assert receiver.source_diagnostics().frozen_peer is None
    assert receiver.stats().datagrams_received == 1

    # Non-loopback input is rejected and latched before production mutation.
    first_packet = _jpeg_packets(frame_id=41, total_chunks=1)[0]
    fake.push_from(first_packet, ("192.0.2.10", 50_001))
    _wait_until(lambda: receiver.source_diagnostics().rejected_source_count == 1)
    rejected = receiver.source_diagnostics()
    assert rejected.source_rejected_latched is True
    assert rejected.frozen_peer is None
    assert receiver.stats().datagrams_received == 1

    # The first syntactically valid loopback chunk freezes before normal feed.
    fake.push_from(first_packet, ("127.0.0.1", 50_001))
    _wait_until(lambda: receiver.snapshot() is not None)
    frozen = receiver.source_diagnostics()
    assert frozen.state == "peer_frozen"
    assert frozen.frozen_peer == ("127.0.0.1", 50_001)
    assert receiver.snapshot().frame_id == 41
    received_before_second_source = receiver.stats().datagrams_received

    # A second endpoint is rejected before it can publish frame 42.
    fake.push_from(
        _jpeg_packets(frame_id=42, total_chunks=1)[0],
        ("127.0.0.1", 50_002),
    )
    _wait_until(lambda: receiver.source_diagnostics().rejected_source_count == 2)
    assert receiver.snapshot().frame_id == 41
    assert receiver.stats().datagrams_received == received_before_second_source

    # The latch is diagnostic, not a cleanup denial: the frozen peer remains
    # usable, and reset must not erase attempt-lifetime peer/count evidence.
    fake.push_from(
        _jpeg_packets(
            frame_id=43,
            sim_time_ns=2_000_000_000,
            total_chunks=1,
        )[0],
        ("127.0.0.1", 50_001),
    )
    _wait_until(lambda: receiver.snapshot().frame_id == 43)
    receiver.reset()
    after_reset = receiver.source_diagnostics()
    assert after_reset.frozen_peer == ("127.0.0.1", 50_001)
    assert after_reset.rejected_source_count == 2
    assert after_reset.source_rejected_latched is True

    receiver.stop()
    stopped = receiver.source_diagnostics()
    assert stopped.state == "closed_with_peer"
    assert stopped.frozen_peer == ("127.0.0.1", 50_001)


def test_powered_vision_close_failure_is_cleanup_failure_not_closed_state():
    fake = _FakePoweredSocket(close_error=True)
    receiver = VQ2VisionThread(
        port=0,
        powered_exclusive=True,
        exclusive_socket_factory=lambda _host, _port: fake,
    )
    receiver.start()
    with pytest.raises(RuntimeError, match="close was not proved"):
        receiver.stop()
    diagnostics = receiver.source_diagnostics()
    assert diagnostics.state == "bound"
    assert receiver.stats().socket_errors >= 1


@pytest.mark.parametrize("max_age", [-1.0, float("nan"), float("inf")])
def test_freshness_rejects_invalid_max_age(max_age):
    receiver = VQ2VisionThread()
    with pytest.raises(ValueError):
        receiver.is_fresh(max_age)


def test_snapshot_publication_lease_blocks_reset_until_release():
    receiver = VQ2VisionThread()
    reset_started = threading.Event()
    reset_completed = threading.Event()

    def reset_receiver():
        reset_started.set()
        receiver.reset()
        reset_completed.set()

    with receiver.snapshot_publication_lease():
        worker = threading.Thread(target=reset_receiver)
        worker.start()
        assert reset_started.wait(1.0)
        worker.join(0.02)
        assert worker.is_alive()
        assert not reset_completed.is_set()

    worker.join(1.0)
    assert not worker.is_alive()
    assert reset_completed.is_set()


def test_snapshot_publication_lease_acquisition_is_deadline_bounded():
    receiver = VQ2VisionThread()
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_publication_lock():
        with receiver._data_lock:
            lock_held.set()
            assert release_lock.wait(1.0)

    worker = threading.Thread(target=hold_publication_lock)
    worker.start()
    assert lock_held.wait(1.0)
    deadline_ns = receiver._read_monotonic_ns() + 20_000_000
    try:
        with pytest.raises(
            TimeoutError,
            match="acquisition deadline",
        ):
            with receiver.snapshot_publication_lease(
                acquire_deadline_monotonic_ns=deadline_ns,
            ):
                pytest.fail("expired publication lease was entered")
    finally:
        release_lock.set()
        worker.join(1.0)

    assert not worker.is_alive()
