"""Dedicated, duplicate-suppressing vision receiver for AIGP VQ2.

FlightSim build 3385 has been observed sending every UDP/5600 JPEG chunk
roughly 50--60 times.  Feeding that traffic through an ``asyncio`` protocol on
the control-loop thread can starve a 50 Hz controller even though the source
camera is only 30 Hz.  :class:`VQ2VisionThread` isolates socket draining in a
daemon thread and rejects repeated ``(frame_id, chunk_id)`` keys before doing
the full packet validation/reassembly work.

The wire contract, reassembler, and JPEG decoder remain owned by
``competition.vision_udp``.  This module only adds lifecycle, concurrency,
freshness, and the build-3385 duplicate guard.  ``reset()`` is a hard race
boundary: pre-reset partial frames, delivered IDs, duplicate keys, and the
published snapshot are discarded, and an in-progress pre-reset decode cannot
publish after the boundary.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

from competition.adapter import CameraFrame
from competition.aigp_geometry import AIGP_CAM_UDP_PORT
from competition.vision_udp import (
    DEFAULT_MAX_BUFFERED_FRAMES,
    DEFAULT_REASSEMBLY_TIMEOUT_MS,
    HEADER_SIZE,
    VisionUdpReceiver,
    decode_jpeg_to_camera_frame,
    parse_packet,
)


_DUPLICATE_KEY_FMT = "<IH"  # frame_id u32, chunk_id u16
_MAX_UDP_DATAGRAM_BYTES = 65_535


@dataclass(frozen=True)
class VQ2VisionSnapshot:
    """The newest decoded frame published by :class:`VQ2VisionThread`.

    ``received_monotonic_s`` is recorded when the final unique chunk arrives,
    making freshness independent of the simulator epoch and wall-clock time.
    The contained ``CameraFrame.image`` is shared for speed and must be treated
    as read-only by consumers.
    """

    frame_id: int
    camera_frame: CameraFrame
    sim_time_ns: int
    received_monotonic_s: float
    generation: int

    def age_s(self, now_monotonic_s: Optional[float] = None) -> float:
        """Return non-negative receive age measured on the monotonic clock."""

        now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
        if not math.isfinite(now):
            raise ValueError("now_monotonic_s must be finite")
        return max(0.0, now - self.received_monotonic_s)

    def is_fresh(
        self,
        max_age_s: float,
        now_monotonic_s: Optional[float] = None,
    ) -> bool:
        """Whether this frame is no older than ``max_age_s``."""

        max_age = _validate_max_age(max_age_s)
        return self.age_s(now_monotonic_s) <= max_age


@dataclass(frozen=True)
class VQ2VisionStats:
    """Thread-safe cumulative diagnostics plus current receiver state."""

    datagrams_received: int
    unique_datagrams: int
    duplicate_datagrams: int
    malformed_datagrams: int
    frames_reassembled: int
    frames_decoded: int
    decode_failures: int
    out_of_order_frame_drops: int
    reset_generation_drops: int
    processing_errors: int
    socket_errors: int
    snapshot_callback_errors: int
    resets: int
    remembered_chunk_keys: int
    receiver_dropped_partial_frames: int
    receiver_duplicate_chunks: int
    receiver_dropped_late_packets: int


class VQ2VisionThread:
    """Drain and decode the VQ2 JPEG stream outside the control-loop thread.

    ``start`` binds UDP/5600 synchronously, so port conflicts are reported to
    the caller rather than hidden in the worker.  ``start`` and ``stop`` are
    idempotent.  Tests and replay tools can call :meth:`feed_datagram` directly
    without opening a socket.
    """

    def __init__(
        self,
        *,
        port: int = AIGP_CAM_UDP_PORT,
        bind_host: str = "0.0.0.0",
        reassembly_timeout_ms: int = DEFAULT_REASSEMBLY_TIMEOUT_MS,
        max_buffered_frames: int = DEFAULT_MAX_BUFFERED_FRAMES,
        max_remembered_chunks: int = 4_096,
        receive_buffer_bytes: int = 4 * 1024 * 1024,
        socket_timeout_s: float = 0.10,
        socket_factory: Optional[Callable[[], socket.socket]] = None,
        on_snapshot: Optional[Callable[[VQ2VisionSnapshot], object]] = None,
    ) -> None:
        if not 0 <= int(port) <= 65_535:
            raise ValueError("port must be in [0, 65535]")
        if int(max_remembered_chunks) < 1:
            raise ValueError("max_remembered_chunks must be >= 1")
        if int(receive_buffer_bytes) < 1:
            raise ValueError("receive_buffer_bytes must be >= 1")
        if not math.isfinite(socket_timeout_s) or socket_timeout_s <= 0.0:
            raise ValueError("socket_timeout_s must be finite and > 0")

        self.port = int(port)
        self.bind_host = str(bind_host)
        self.reassembly_timeout_ms = int(reassembly_timeout_ms)
        self.max_buffered_frames = int(max_buffered_frames)
        self.max_remembered_chunks = int(max_remembered_chunks)
        self.receive_buffer_bytes = int(receive_buffer_bytes)
        self.socket_timeout_s = float(socket_timeout_s)
        self._socket_factory = socket_factory or self._new_udp_socket
        self._on_snapshot = on_snapshot

        self._data_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

        self._receiver = self._new_receiver()
        self._seen_keys: set[Tuple[int, int]] = set()
        self._seen_order: Deque[Tuple[int, int]] = deque()
        self._latest_snapshot: Optional[VQ2VisionSnapshot] = None
        self._generation = 0

        self._datagrams_received = 0
        self._unique_datagrams = 0
        self._duplicate_datagrams = 0
        self._malformed_datagrams = 0
        self._frames_reassembled = 0
        self._frames_decoded = 0
        self._decode_failures = 0
        self._out_of_order_frame_drops = 0
        self._reset_generation_drops = 0
        self._processing_errors = 0
        self._socket_errors = 0
        self._snapshot_callback_errors = 0
        self._resets = 0

    @staticmethod
    def _new_udp_socket() -> socket.socket:
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _new_receiver(self) -> VisionUdpReceiver:
        return VisionUdpReceiver(
            port=self.port,
            reassembly_timeout_ms=self.reassembly_timeout_ms,
            max_buffered_frames=self.max_buffered_frames,
        )

    def start(self) -> None:
        """Bind the configured UDP port and start the receiver daemon.

        Binding happens before the thread starts.  An ``OSError`` therefore
        reaches the caller immediately when another process owns UDP/5600.
        """

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            sock = self._socket_factory()
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.receive_buffer_bytes)
                sock.settimeout(self.socket_timeout_s)
                sock.bind((self.bind_host, self.port))
            except Exception:
                try:
                    sock.close()
                finally:
                    raise

            self._stop_event.clear()
            thread = threading.Thread(
                target=self._receive_loop,
                args=(sock,),
                name="aigp-vq2-vision",
                daemon=True,
            )
            self._socket = sock
            self._thread = thread
            thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the worker and close its socket.  Safe before/after ``start``."""

        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and >= 0")

        with self._lifecycle_lock:
            self._stop_event.set()
            sock = self._socket
            thread = self._thread

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=float(timeout_s))

        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            raise RuntimeError(
                "VQ2 vision thread did not terminate before stop timeout"
            )

        with self._lifecycle_lock:
            if self._socket is sock:
                self._socket = None
            # Preserve a still-live thread reference after a caller-requested
            # zero/short timeout.  This prevents a subsequent start() from
            # launching a second worker while the first finishes decoding.
            if self._thread is thread and (
                thread is None or not thread.is_alive()
            ):
                self._thread = None

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def reset(self) -> None:
        """Clear all frame/dedup state at a ``SIM_RESET`` boundary.

        Lifetime counters are intentionally preserved for diagnostics.  The
        incremented generation prevents a JPEG decode that began before this
        call from publishing a stale frame afterward.
        """

        with self._data_lock:
            self._generation += 1
            self._receiver.reset()
            self._seen_keys.clear()
            self._seen_order.clear()
            self._latest_snapshot = None
            self._resets += 1

    def feed_datagram(
        self,
        raw: bytes,
        *,
        received_monotonic_s: Optional[float] = None,
    ) -> Optional[VQ2VisionSnapshot]:
        """Process one wire datagram, returning a newly decoded frame if any.

        The first six header bytes are inspected for the duplicate key.  A
        repeated key is returned immediately; only a first-seen key pays for
        full :func:`parse_packet` validation and
        :class:`VisionUdpReceiver` reassembly.
        """

        received_at = (
            time.monotonic()
            if received_monotonic_s is None
            else float(received_monotonic_s)
        )
        if not math.isfinite(received_at):
            raise ValueError("received_monotonic_s must be finite")

        with self._data_lock:
            self._datagrams_received += 1
            if len(raw) < HEADER_SIZE:
                self._malformed_datagrams += 1
                return None

            # This deliberately reads only the duplicate key.  Full header and
            # payload validation remains centralized in vision_udp.parse_packet.
            try:
                frame_id, chunk_id = struct.unpack_from(_DUPLICATE_KEY_FMT, raw, 0)
            except struct.error:
                self._malformed_datagrams += 1
                return None
            key = (int(frame_id), int(chunk_id))
            if key in self._seen_keys:
                self._duplicate_datagrams += 1
                return None

            try:
                parse_packet(raw)
            except ValueError:
                # Do not remember malformed first copies: a later valid packet
                # with the same frame/chunk key must still be accepted.
                self._malformed_datagrams += 1
                return None

            self._remember_key(key)
            self._unique_datagrams += 1
            generation = self._generation
            try:
                completed = self._receiver.feed_packet(raw)
            except ValueError:
                # Defensive: parse_packet above already validated the packet,
                # but receiver-level consistency checks can still reject a
                # conflicting chunk header inside an in-progress frame.
                self._malformed_datagrams += 1
                return None
            if completed is None:
                return None
            self._frames_reassembled += 1

        # JPEG decode can be comparatively expensive; keep reset/snapshot calls
        # responsive and use generation validation when publishing afterward.
        try:
            camera_frame = decode_jpeg_to_camera_frame(completed)
        except Exception:
            # A corrupt JPEG/OpenCV failure must not kill the dedicated socket
            # worker.  The watchdog will see freshness expire if this persists.
            with self._data_lock:
                if generation == self._generation:
                    self._decode_failures += 1
            return None

        with self._data_lock:
            if generation != self._generation:
                self._reset_generation_drops += 1
                return None
            if camera_frame is None:
                self._decode_failures += 1
                return None
            # Snapshots are shared with control and the asynchronous replay
            # writer without a copy.  Enforce the documented ownership
            # contract at publication so no consumer can mutate evidence
            # before persistence.
            camera_frame.image.setflags(write=False)
            if (
                self._latest_snapshot is not None
                and completed.sim_time_ns <= self._latest_snapshot.sim_time_ns
            ):
                # Interleaved UDP frames can complete out of order.  A control
                # snapshot is monotonic within one reset generation.
                self._out_of_order_frame_drops += 1
                return None
            snapshot = VQ2VisionSnapshot(
                frame_id=completed.frame_id,
                camera_frame=camera_frame,
                sim_time_ns=completed.sim_time_ns,
                received_monotonic_s=received_at,
                generation=generation,
            )
            self._latest_snapshot = snapshot
            self._frames_decoded += 1
        # A replay listener must be non-blocking by contract.  It runs outside
        # the publication lock so even a faulty optional recorder cannot delay
        # freshness reads/reset.  Callback failures affect capture diagnostics,
        # never receiver/control liveness.
        if self._on_snapshot is not None:
            try:
                self._on_snapshot(snapshot)
            except Exception:
                with self._data_lock:
                    self._snapshot_callback_errors += 1
        return snapshot

    def snapshot(
        self,
        *,
        max_age_s: Optional[float] = None,
        now_monotonic_s: Optional[float] = None,
    ) -> Optional[VQ2VisionSnapshot]:
        """Return the latest snapshot, optionally only when it is fresh."""

        if max_age_s is not None:
            max_age = _validate_max_age(max_age_s)
        else:
            max_age = None
        if now_monotonic_s is not None and not math.isfinite(now_monotonic_s):
            raise ValueError("now_monotonic_s must be finite")

        with self._data_lock:
            snapshot = self._latest_snapshot
        if snapshot is None:
            return None
        if max_age is not None and not snapshot.is_fresh(max_age, now_monotonic_s):
            return None
        return snapshot

    def is_fresh(
        self,
        max_age_s: float,
        *,
        now_monotonic_s: Optional[float] = None,
    ) -> bool:
        """Whether a decoded frame exists within the requested receive age."""

        return self.snapshot(
            max_age_s=max_age_s,
            now_monotonic_s=now_monotonic_s,
        ) is not None

    def stats(self) -> VQ2VisionStats:
        """Return an atomic diagnostics snapshot."""

        with self._data_lock:
            receiver = self._receiver
            return VQ2VisionStats(
                datagrams_received=self._datagrams_received,
                unique_datagrams=self._unique_datagrams,
                duplicate_datagrams=self._duplicate_datagrams,
                malformed_datagrams=self._malformed_datagrams,
                frames_reassembled=self._frames_reassembled,
                frames_decoded=self._frames_decoded,
                decode_failures=self._decode_failures,
                out_of_order_frame_drops=self._out_of_order_frame_drops,
                reset_generation_drops=self._reset_generation_drops,
                processing_errors=self._processing_errors,
                socket_errors=self._socket_errors,
                snapshot_callback_errors=self._snapshot_callback_errors,
                resets=self._resets,
                remembered_chunk_keys=len(self._seen_keys),
                receiver_dropped_partial_frames=receiver.dropped_partial_frames,
                receiver_duplicate_chunks=receiver.duplicate_chunks,
                receiver_dropped_late_packets=receiver.dropped_late_packets,
            )

    def _remember_key(self, key: Tuple[int, int]) -> None:
        while len(self._seen_order) >= self.max_remembered_chunks:
            expired = self._seen_order.popleft()
            self._seen_keys.discard(expired)
        self._seen_order.append(key)
        self._seen_keys.add(key)

    def _receive_loop(self, sock: socket.socket) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    raw, _addr = sock.recvfrom(_MAX_UDP_DATAGRAM_BYTES)
                except socket.timeout:
                    continue
                except OSError:
                    if not self._stop_event.is_set():
                        with self._data_lock:
                            self._socket_errors += 1
                    break
                try:
                    self.feed_datagram(raw)
                except Exception:
                    # Keep draining after an unexpected per-datagram failure.
                    # Deliberately do not catch BaseException/SystemExit.
                    with self._data_lock:
                        self._processing_errors += 1
        finally:
            try:
                sock.close()
            except OSError:
                pass


def _validate_max_age(max_age_s: float) -> float:
    max_age = float(max_age_s)
    if not math.isfinite(max_age) or max_age < 0.0:
        raise ValueError("max_age_s must be finite and >= 0")
    return max_age
