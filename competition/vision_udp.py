"""
AIGP vision stream UDP receiver (iter-001 A11).

The AIGP DCL simulator publishes the forward-facing camera stream on UDP
port 5600 as JPEG-encoded frames split into chunks, each carrying a
24-byte little-endian header (VADR-TS-002 §4.6):

    frame_id      uint32   unique per image frame
    chunk_id      uint16   0 .. total_chunks - 1
    total_chunks  uint16   chunks required to reassemble this frame
    jpeg_size     uint32   size of the final reconstructed JPEG (bytes)
    payload_size  uint32   size of the JPEG slice in THIS packet (bytes)
    sim_time_ns   uint64   simulation epoch timestamp (nanoseconds)

The receiver is split into a synchronous reassembler (`VisionUdpReceiver`)
and the JPEG decode step (`decode_jpeg_to_camera_frame`). Unit tests cover
the reassembler without needing OpenCV; the decode step is only invoked
when a complete frame is wanted.

Design notes:
- Reassembly buffers are keyed by `frame_id`. A late packet for a frame
  that has already completed is dropped (avoids re-emitting the same
  image; downstream consumers see one CameraFrame per frame_id).
- Duplicate chunks are idempotent — the last copy wins.
- Stale partial frames are GC'd by sim_time_ns gap, NOT wall clock: the
  receiver must work in playback / accelerated-time scenarios where wall
  clock doesn't track sim time.
- `max_buffered_frames` is the upper bound on simultaneous in-flight
  frames; eviction drops the oldest by sim_time_ns.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from competition.aigp_geometry import AIGP_CAM_UDP_PORT

# Little-endian header layout: I=u32, H=u16, Q=u64.
# Order: frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns.
HEADER_FMT: str = "<IHHIIQ"
HEADER_SIZE: int = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 24, f"VADR-TS-002 §4.6 specifies 24-byte header; got {HEADER_SIZE}"

# Default reassembly timeout — 100 ms at 30 Hz ≈ 3 frames. Long enough to
# tolerate reordering on a healthy LAN, short enough that a single dropped
# chunk doesn't deadlock memory growth.
DEFAULT_REASSEMBLY_TIMEOUT_MS: int = 100
DEFAULT_MAX_BUFFERED_FRAMES: int = 8


@dataclass
class VisionPacket:
    """One UDP datagram from the AIGP vision stream."""
    frame_id: int
    chunk_id: int
    total_chunks: int
    jpeg_size: int
    payload_size: int
    sim_time_ns: int
    payload: bytes


@dataclass
class ReassembledFrame:
    """A full JPEG-encoded frame, ready for decode."""
    frame_id: int
    jpeg_bytes: bytes
    sim_time_ns: int


@dataclass
class _ReassemblyBuf:
    """In-flight buffer for one in-progress frame."""
    total_chunks: int
    jpeg_size: int
    sim_time_ns: int               # of the first packet that started this buffer
    chunks: Dict[int, bytes] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return len(self.chunks) == self.total_chunks

    def assemble(self) -> bytes:
        """Concatenate chunks in ID order. Caller must check `.complete` first."""
        out = bytearray(self.jpeg_size)
        # Chunks may have different sizes (e.g. the last chunk is often
        # smaller). Reconstruct by writing each chunk into the right slice.
        # We don't know offsets from the header alone, so assume contiguous
        # ID-order concatenation matches the encoder's slicing — AIGP spec
        # implies this (chunk_id 0..total-1 with a fixed slice strategy).
        offset = 0
        for cid in range(self.total_chunks):
            chunk = self.chunks[cid]
            out[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
        return bytes(out)


def parse_packet(data: bytes) -> VisionPacket:
    """Parse one UDP datagram. Raises ValueError on malformed input."""
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"packet too short: got {len(data)} bytes, need ≥ {HEADER_SIZE}"
        )
    header = data[:HEADER_SIZE]
    payload = data[HEADER_SIZE:]
    fid, cid, total, jpeg_sz, pay_sz, ts = struct.unpack(HEADER_FMT, header)
    if pay_sz != len(payload):
        raise ValueError(
            f"payload_size mismatch: header says {pay_sz}, got {len(payload)} "
            f"(frame_id={fid}, chunk_id={cid})"
        )
    if total == 0:
        raise ValueError(f"total_chunks must be > 0 (frame_id={fid})")
    if cid >= total:
        raise ValueError(
            f"chunk_id {cid} out of range for total_chunks {total} "
            f"(frame_id={fid})"
        )
    # Iter-001 review Opus F9 (a): reject zero-size payloads. The encoder
    # should never emit empty chunks; accepting them leaves zero-filled
    # holes in the assembled JPEG that cv2.imdecode silently fails on,
    # and the caller would just see a None frame with no diagnostic.
    if pay_sz == 0:
        raise ValueError(
            f"zero payload_size (frame_id={fid}, chunk_id={cid})"
        )
    return VisionPacket(
        frame_id=fid, chunk_id=cid, total_chunks=total,
        jpeg_size=jpeg_sz, payload_size=pay_sz, sim_time_ns=ts,
        payload=bytes(payload),
    )


def encode_packet(
    frame_id: int,
    chunk_id: int,
    total_chunks: int,
    jpeg_size: int,
    sim_time_ns: int,
    payload: bytes,
) -> bytes:
    """Build a wire-format packet. Used by tests and by any local mock sender."""
    header = struct.pack(
        HEADER_FMT,
        frame_id, chunk_id, total_chunks, jpeg_size, len(payload), sim_time_ns,
    )
    return header + payload


class VisionUdpReceiver:
    """
    Synchronous AIGP vision-stream reassembler.

    Call `feed_packet(raw_bytes)` for each datagram; when a frame becomes
    complete, that call returns a `ReassembledFrame`. Use
    `pop_latest_frame()` to retrieve the most recent completed frame
    without feeding new data (useful when the control loop wants the
    freshest image regardless of how many packets arrived since the last
    poll).

    The async UDP listener that drives this in production is layered on
    top via `asyncio.DatagramProtocol`; this class itself is sync so it
    can be unit-tested without an event loop.
    """

    def __init__(
        self,
        port: int = AIGP_CAM_UDP_PORT,
        reassembly_timeout_ms: int = DEFAULT_REASSEMBLY_TIMEOUT_MS,
        max_buffered_frames: int = DEFAULT_MAX_BUFFERED_FRAMES,
    ):
        self.port: int = int(port)
        self.reassembly_timeout_ns: int = int(reassembly_timeout_ms) * 1_000_000
        self.max_buffered: int = int(max_buffered_frames)
        self._buffers: Dict[int, _ReassemblyBuf] = {}
        # Frame IDs that have already been emitted — additional packets for
        # them are dropped silently. Bounded so it can't grow unboundedly.
        self._delivered_ids: List[int] = []
        self._delivered_id_cap: int = max_buffered_frames * 8
        self._latest_frame: Optional[ReassembledFrame] = None
        # Diagnostic counters.
        self.dropped_partial_frames: int = 0
        self.delivered_frames: int = 0
        self.duplicate_chunks: int = 0
        self.dropped_late_packets: int = 0

    def feed_packet(self, raw: bytes) -> Optional[ReassembledFrame]:
        """Feed one UDP datagram. Returns the completed frame if this was
        the final chunk; else None."""
        pkt = parse_packet(raw)
        self._gc_stale(pkt.sim_time_ns)

        if pkt.frame_id in self._delivered_ids:
            # Late packet for a frame we already emitted.
            self.dropped_late_packets += 1
            return None

        buf = self._buffers.get(pkt.frame_id)
        if buf is None:
            # First sighting of this frame_id. Make room if we're at the cap.
            if len(self._buffers) >= self.max_buffered:
                self._evict_oldest()
            buf = _ReassemblyBuf(
                total_chunks=pkt.total_chunks,
                jpeg_size=pkt.jpeg_size,
                sim_time_ns=pkt.sim_time_ns,
            )
            self._buffers[pkt.frame_id] = buf
        else:
            if buf.total_chunks != pkt.total_chunks:
                # Inconsistent header — drop and reset.
                self._buffers.pop(pkt.frame_id, None)
                self.dropped_partial_frames += 1
                raise ValueError(
                    f"inconsistent total_chunks for frame_id={pkt.frame_id}: "
                    f"buffer had {buf.total_chunks}, packet has {pkt.total_chunks}"
                )

        if pkt.chunk_id in buf.chunks:
            self.duplicate_chunks += 1
        buf.chunks[pkt.chunk_id] = pkt.payload

        if not buf.complete:
            return None

        # Iter-001 review Opus F9 (b) + 7-way MAJOR consensus: validate
        # that the chunk payload sizes actually sum to jpeg_size before
        # yielding. Otherwise a malformed encoder (or a corrupted-on-wire
        # packet that slipped past payload_size checks) produces a
        # zero-tailed or over-grown buffer that cv2.imdecode silently
        # fails on — caller gets None with no diagnostic.
        total_payload = sum(len(c) for c in buf.chunks.values())
        if total_payload != buf.jpeg_size:
            self._buffers.pop(pkt.frame_id, None)
            self.dropped_partial_frames += 1
            return None

        # Complete — emit.
        jpeg_bytes = buf.assemble()
        frame = ReassembledFrame(
            frame_id=pkt.frame_id,
            jpeg_bytes=jpeg_bytes,
            sim_time_ns=buf.sim_time_ns,
        )
        self._buffers.pop(pkt.frame_id, None)
        self._delivered_ids.append(pkt.frame_id)
        if len(self._delivered_ids) > self._delivered_id_cap:
            # Drop the oldest delivered IDs — late-packet detection is
            # best-effort over a sliding window.
            self._delivered_ids = self._delivered_ids[-self._delivered_id_cap:]
        self.delivered_frames += 1
        self._latest_frame = frame
        return frame

    def pop_latest_frame(self) -> Optional[ReassembledFrame]:
        """Return the most recent completed frame, or None."""
        return self._latest_frame

    def reset(self) -> None:
        """Discard all pre-reset frame state.

        ``SIM_RESET`` may restart frame ids.  Keeping delivered ids or a cached
        pre-reset image would make a fresh race consume stale vision (or drop
        the first repeated ids), so callers must clear the reassembler at the
        same boundary as the simulator clock reset.
        """

        self._buffers.clear()
        self._delivered_ids.clear()
        self._latest_frame = None

    def _gc_stale(self, now_sim_ns: int) -> None:
        """Drop in-flight buffers whose first packet is older than the
        reassembly timeout, measured in SIM time (not wall time)."""
        stale = [
            fid for fid, b in self._buffers.items()
            if now_sim_ns - b.sim_time_ns > self.reassembly_timeout_ns
        ]
        for fid in stale:
            self._buffers.pop(fid, None)
            self.dropped_partial_frames += 1

    def _evict_oldest(self) -> None:
        if not self._buffers:
            return
        oldest = min(
            self._buffers.items(), key=lambda kv: kv[1].sim_time_ns,
        )
        self._buffers.pop(oldest[0], None)
        self.dropped_partial_frames += 1


def decode_jpeg_to_camera_frame(
    frame: ReassembledFrame, width: int = 640, height: int = 360,
):
    """Decode a reassembled JPEG to a `competition.adapter.CameraFrame`.

    Kept out of the receiver hot path so unit tests don't need cv2.
    """
    import cv2  # local import — only paid when actually decoding
    import numpy as np

    from competition.adapter import CameraFrame

    arr = np.frombuffer(frame.jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if (w, h) != (width, height):
        # Soft-warn via the shape; the spec is 640×360 but we surface the
        # actual dimensions rather than silently resizing.
        pass
    return CameraFrame(
        image=img,
        width=w,
        height=h,
        timestamp_us=frame.sim_time_ns // 1000,
    )


# ---------------------------------------------------------------------------
# Async UDP listener — wires the reassembler into an asyncio event loop
# (iter-002 B4: MAVLinkBridge integration).
# ---------------------------------------------------------------------------

class _VisionDatagramProtocol:
    """asyncio.DatagramProtocol that feeds raw bytes into a VisionUdpReceiver.

    Defined as a duck-typed protocol (not subclassing DatagramProtocol) so
    the module imports cleanly without asyncio at module-load time;
    asyncio is only paid by the live listener path.
    """

    def __init__(self, receiver: VisionUdpReceiver):
        self.receiver = receiver
        self.errors = 0

    def connection_made(self, transport) -> None:   # noqa: D401
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        try:
            self.receiver.feed_packet(data)
        except ValueError:
            # Malformed packet — count it and keep listening. Real-world
            # AIGP traffic is JPEG over UDP; one bad packet shouldn't
            # poison the stream.
            self.errors += 1

    def error_received(self, exc) -> None:
        self.errors += 1

    def connection_lost(self, exc) -> None:
        # Transport closed — caller handles cleanup via stop().
        pass


class VisionUdpListener:
    """Asyncio wrapper that opens a UDP socket on `port` and feeds every
    received datagram into a `VisionUdpReceiver`.

    Lifecycle: ``await listener.start()`` to bind the socket and begin
    receiving; ``await listener.stop()`` to close it. ``latest_frame()``
    returns the most recent successfully-decoded `CameraFrame`, or None.

    The listener doesn't decode JPEG on each packet — decoding happens
    on demand inside `latest_frame()` so a control loop polling at 100 Hz
    pays the decode cost only when it actually wants the freshest image,
    while the receiver continues reassembling at network speed.
    """

    def __init__(
        self,
        port: int = AIGP_CAM_UDP_PORT,
        receiver: Optional[VisionUdpReceiver] = None,
        bind_host: str = "0.0.0.0",
    ):
        self.port: int = int(port)
        self.bind_host: str = bind_host
        self.receiver: VisionUdpReceiver = (
            receiver if receiver is not None else VisionUdpReceiver(port=port)
        )
        self._transport = None  # asyncio.DatagramTransport (set by start)
        self._protocol: Optional[_VisionDatagramProtocol] = None
        # Iter-002 review M2 (7/7 reviews MAJOR): cache the last decoded
        # CameraFrame so a 100Hz poll doesn't re-decode the same JPEG 30
        # times per source-frame. Decode only fires on frame_id transition.
        self._last_decoded_frame_id: int = -1
        self._last_decoded_camera_frame = None

    async def start(self) -> None:
        """Open the UDP socket and begin receiving.

        Iter-002 review M1 (6/7 reviews MAJOR): idempotent. A second
        start() with a transport already bound is a no-op; the prior
        version silently overwrote `self._transport`, leaking the first
        socket's bound port and protocol object. Real exposure: any
        reconnect path in MAVLinkBridge would leak.
        """
        if self._transport is not None:
            return
        import asyncio  # local — keeps module importable without asyncio
        loop = asyncio.get_event_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: _VisionDatagramProtocol(self.receiver),
            local_addr=(self.bind_host, self.port),
        )

    async def stop(self) -> None:
        """Close the UDP socket. Idempotent."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None
            self._protocol = None

    def reset(self) -> None:
        """Clear reassembly and decoded-image caches after ``SIM_RESET``."""

        self.receiver.reset()
        self._last_decoded_frame_id = -1
        self._last_decoded_camera_frame = None

    @property
    def is_listening(self) -> bool:
        return self._transport is not None

    @property
    def malformed_packet_count(self) -> int:
        return self._protocol.errors if self._protocol else 0

    def latest_frame(self):
        """Decode and return the most recent CameraFrame, or None.

        Iter-002 review M2: decode-and-cache by frame_id. A poll that
        sees the same frame_id as the previous poll returns the cached
        CameraFrame instead of re-decoding the JPEG. Cache invalidates
        on frame_id change; cv2.imdecode is paid at most once per
        source-frame.
        """
        rf = self.receiver.pop_latest_frame()
        if rf is None:
            return None
        if rf.frame_id == self._last_decoded_frame_id:
            return self._last_decoded_camera_frame
        cf = decode_jpeg_to_camera_frame(rf)
        self._last_decoded_frame_id = rf.frame_id
        self._last_decoded_camera_frame = cf
        return cf
