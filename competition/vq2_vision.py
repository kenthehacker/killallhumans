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

import ipaddress
import math
import socket
import struct
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Deque, Iterator, Optional, Tuple

from competition.adapter import (
    CameraFrame,
    VisionPublicationLeaseAcquisitionTimeout,
)
from competition.aigp_geometry import AIGP_CAM_UDP_PORT
from competition.vq2_contracts import FrameIdentityV1, FrameTimingV1
from competition.vq2_runtime import VQ2_HOST_CLOCK_ID
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
    # Optional only for source compatibility with callers that construct a
    # legacy snapshot directly.  VQ2VisionThread always publishes exact /1
    # timing.
    timing: Optional[FrameTimingV1] = None

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
    timing_ledger_entries: int
    timing_ledger_high_watermark: int
    timing_ledger_capacity: int
    timing_overflow_latched: bool
    receiver_buffered_partial_frames: int
    receiver_buffer_high_watermark: int
    receiver_buffer_capacity: int
    capture_snapshot_queue_entries: int
    capture_snapshot_queue_high_watermark: int
    capture_snapshot_queue_capacity: int
    capture_snapshot_queue_dropped: int
    capture_snapshot_queue_enabled: bool
    receiver_dropped_partial_frames: int
    receiver_duplicate_chunks: int
    receiver_dropped_late_packets: int


@dataclass(frozen=True)
class VQ2VisionSourceDiagnostics:
    """Powered-only exclusive bind and source-freeze diagnostics."""

    powered_exclusive: bool
    state: str
    requested_host: str
    requested_port: int
    actual_host: Optional[str]
    actual_port: Optional[int]
    socket_policy: str
    frozen_peer: Optional[Tuple[str, int]]
    rejected_source_count: int
    source_rejected_latched: bool


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
        capture_snapshot_queue_enabled: bool = False,
        capture_snapshot_queue_capacity: int = 256,
        stream_id: str = "vq2-camera-udp-5600",
        host_clock_id: str = VQ2_HOST_CLOCK_ID,
        monotonic_ns: Optional[Callable[[], int]] = None,
        powered_exclusive: bool = False,
        exclusive_socket_factory: Optional[
            Callable[[str, int], socket.socket]
        ] = None,
    ) -> None:
        if not 0 <= int(port) <= 65_535:
            raise ValueError("port must be in [0, 65535]")
        if int(max_remembered_chunks) < 1:
            raise ValueError("max_remembered_chunks must be >= 1")
        if int(max_buffered_frames) < 1:
            raise ValueError("max_buffered_frames must be >= 1")
        if int(receive_buffer_bytes) < 1:
            raise ValueError("receive_buffer_bytes must be >= 1")
        if (
            type(capture_snapshot_queue_capacity) is not int
            or capture_snapshot_queue_capacity < 1
        ):
            raise ValueError("capture_snapshot_queue_capacity must be >= 1")
        if type(capture_snapshot_queue_enabled) is not bool:
            raise TypeError("capture_snapshot_queue_enabled must be an exact bool")
        if capture_snapshot_queue_enabled and on_snapshot is None:
            raise ValueError(
                "capture snapshot queue requires an on_snapshot capture callback"
            )
        if not math.isfinite(socket_timeout_s) or socket_timeout_s <= 0.0:
            raise ValueError("socket_timeout_s must be finite and > 0")
        if monotonic_ns is not None and not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable or None")
        if type(powered_exclusive) is not bool:
            raise TypeError("powered_exclusive must be an exact bool")
        if powered_exclusive and not callable(exclusive_socket_factory):
            raise ValueError(
                "powered_exclusive requires an exclusive_socket_factory"
            )
        if powered_exclusive and socket_factory is not None:
            raise ValueError(
                "socket_factory cannot be combined with powered_exclusive"
            )
        if not powered_exclusive and exclusive_socket_factory is not None:
            raise ValueError(
                "exclusive_socket_factory is available only in powered mode"
            )

        self.port = int(port)
        self.bind_host = str(bind_host)
        self.reassembly_timeout_ms = int(reassembly_timeout_ms)
        self.max_buffered_frames = int(max_buffered_frames)
        self.max_remembered_chunks = int(max_remembered_chunks)
        self.receive_buffer_bytes = int(receive_buffer_bytes)
        self.socket_timeout_s = float(socket_timeout_s)
        self._socket_factory = socket_factory or self._new_udp_socket
        self._powered_exclusive = powered_exclusive
        self._exclusive_socket_factory = exclusive_socket_factory
        self._on_snapshot = on_snapshot
        self._capture_snapshot_queue_enabled = capture_snapshot_queue_enabled
        self.capture_snapshot_queue_capacity = int(
            capture_snapshot_queue_capacity
        )
        self.stream_id = stream_id
        self.host_clock_id = host_clock_id
        # Windows ``time.monotonic`` is backed by coarse GetTickCount64 on the
        # target Python build.  QueryPerformanceCounter supplies the required
        # sub-millisecond instrumentation while remaining monotonic.
        self._monotonic_ns = monotonic_ns or time.perf_counter_ns

        # Reuse the frozen contract's exact token validation at construction,
        # before a live receiver could publish a partially identified stream.
        probe_identity = FrameIdentityV1(self.stream_id, 0, 0)
        FrameTimingV1(
            identity=probe_identity,
            camera_source_time_ns=0,
            host_clock_id=self.host_clock_id,
            publication_sequence=0,
            first_unique_packet_monotonic_ns=0,
            final_unique_packet_monotonic_ns=0,
            reassembly_complete_monotonic_ns=0,
            decode_start_monotonic_ns=0,
            decode_end_monotonic_ns=0,
            publish_monotonic_ns=0,
        )

        self._data_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._endpoint_state = "not_opened"
        self._actual_bind: Optional[Tuple[str, int]] = None
        self._frozen_peer: Optional[Tuple[str, int]] = None
        self._rejected_source_count = 0
        self._source_rejected_latched = False
        self._socket_close_proved = False

        self._receiver = self._new_receiver()
        self._seen_keys: set[Tuple[int, int]] = set()
        self._seen_order: Deque[Tuple[int, int]] = deque()
        self._frame_first_packet_ns: OrderedDict[int, int] = OrderedDict()
        self._published_frame_ids: set[int] = set()
        self._timing_overflow_latched = False
        self._timing_ledger_high_watermark = 0
        self._receiver_buffer_high_watermark = 0
        self._latest_snapshot: Optional[VQ2VisionSnapshot] = None
        self._capture_snapshot_queue: Deque[VQ2VisionSnapshot] = deque()
        self._capture_snapshot_queue_high_watermark = 0
        self._capture_snapshot_queue_dropped = 0
        self._generation = 0
        self._publication_sequence = 0
        self._last_publish_monotonic_ns: Optional[int] = None

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

            if self._powered_exclusive:
                assert self._exclusive_socket_factory is not None
                sock = self._exclusive_socket_factory(self.bind_host, self.port)
            else:
                sock = self._socket_factory()
            try:
                if self._powered_exclusive:
                    if getattr(sock, "family", None) != socket.AF_INET:
                        raise OSError("powered vision socket must be AF_INET")
                    getsockname = getattr(sock, "getsockname", None)
                    if not callable(getsockname):
                        raise OSError(
                            "powered vision socket cannot prove bind identity"
                        )
                    actual = getsockname()
                    if (
                        type(actual) is not tuple
                        or len(actual) < 2
                        or type(actual[0]) is not str
                        or type(actual[1]) is not int
                    ):
                        raise OSError("powered vision socket has invalid bind identity")
                    actual_host = str(actual[0])
                    actual_port = int(actual[1])
                    if actual_host != self.bind_host:
                        raise OSError("powered vision socket bind host changed")
                    if self.port == 0:
                        if not 1 <= actual_port <= 65_535:
                            raise OSError("powered vision ephemeral port is invalid")
                    elif actual_port != self.port:
                        raise OSError("powered vision socket bind port changed")
                else:
                    actual_host = self.bind_host
                    actual_port = self.port
                sock.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_RCVBUF,
                    self.receive_buffer_bytes,
                )
                sock.settimeout(self.socket_timeout_s)
                if not self._powered_exclusive:
                    sock.bind((self.bind_host, self.port))
                    getsockname = getattr(sock, "getsockname", None)
                    if callable(getsockname):
                        actual = getsockname()
                        if type(actual) is tuple and len(actual) >= 2:
                            actual_host = str(actual[0])
                            actual_port = int(actual[1])
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
            with self._data_lock:
                self._socket_close_proved = False
                self._actual_bind = (actual_host, actual_port)
                self._endpoint_state = (
                    "peer_frozen" if self._frozen_peer is not None else "bound"
                )
            try:
                thread.start()
            except Exception:
                self._socket = None
                self._thread = None
                try:
                    sock.close()
                except OSError:
                    with self._data_lock:
                        self._socket_errors += 1
                    raise
                else:
                    with self._data_lock:
                        self._socket_close_proved = True
                        self._endpoint_state = (
                            "closed_with_peer"
                            if self._frozen_peer is not None
                            else "closed_without_peer"
                        )
                raise

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
                with self._data_lock:
                    self._socket_errors += 1
            else:
                with self._data_lock:
                    self._socket_close_proved = True
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
        with self._data_lock:
            close_proved = self._socket_close_proved
            if self._actual_bind is not None and close_proved:
                self._endpoint_state = (
                    "closed_with_peer"
                    if self._frozen_peer is not None
                    else "closed_without_peer"
                )
        if self._powered_exclusive and sock is not None and not close_proved:
            raise RuntimeError("powered vision socket close was not proved")

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def source_diagnostics(self) -> VQ2VisionSourceDiagnostics:
        """Return immutable bind/source state without weakening source gates."""

        with self._data_lock:
            actual = self._actual_bind
            peer = self._frozen_peer
            return VQ2VisionSourceDiagnostics(
                powered_exclusive=self._powered_exclusive,
                state=self._endpoint_state,
                requested_host=self.bind_host,
                requested_port=self.port,
                actual_host=None if actual is None else actual[0],
                actual_port=None if actual is None else actual[1],
                socket_policy=(
                    "ipv4-exclusive-address-use"
                    if self._powered_exclusive
                    else "legacy-default"
                ),
                frozen_peer=None if peer is None else (peer[0], peer[1]),
                rejected_source_count=self._rejected_source_count,
                source_rejected_latched=self._source_rejected_latched,
            )

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
            self._frame_first_packet_ns.clear()
            self._published_frame_ids.clear()
            self._timing_overflow_latched = False
            self._latest_snapshot = None
            self._capture_snapshot_queue.clear()
            self._resets += 1

    def feed_datagram(
        self,
        raw: bytes,
        *,
        received_monotonic_s: Optional[float] = None,
        received_monotonic_ns: Optional[int] = None,
    ) -> Optional[VQ2VisionSnapshot]:
        """Feed replay/test traffic when the powered source gate is disabled."""

        if self._powered_exclusive:
            raise RuntimeError(
                "powered vision datagrams must pass the socket source gate"
            )
        return self._feed_admitted_datagram(
            raw,
            received_monotonic_s=received_monotonic_s,
            received_monotonic_ns=received_monotonic_ns,
        )

    def _feed_admitted_datagram(
        self,
        raw: bytes,
        *,
        received_monotonic_s: Optional[float] = None,
        received_monotonic_ns: Optional[int] = None,
    ) -> Optional[VQ2VisionSnapshot]:
        """Process one wire datagram, returning a newly decoded frame if any.

        The first six header bytes are inspected for the duplicate key.  A
        repeated key is returned immediately; only a first-seen key pays for
        full :func:`parse_packet` validation and
        :class:`VisionUdpReceiver` reassembly.

        ``received_monotonic_ns`` belongs to the configured `/1` host clock.
        ``received_monotonic_s`` belongs only to the legacy freshness clock;
        when omitted, freshness samples :func:`time.monotonic` independently.
        """

        if received_monotonic_ns is not None:
            received_at_ns = _validate_monotonic_ns(
                received_monotonic_ns, "received_monotonic_ns"
            )
        else:
            received_at_ns = self._read_monotonic_ns()
        if received_monotonic_s is not None:
            if type(received_monotonic_s) not in {int, float}:
                raise TypeError("received_monotonic_s must be numeric and not bool")
            received_at = float(received_monotonic_s)
            if not math.isfinite(received_at) or received_at < 0.0:
                raise ValueError("received_monotonic_s must be finite and >= 0")
        else:
            received_at = time.monotonic()

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
                packet = parse_packet(raw)
            except ValueError:
                # Do not remember malformed first copies: a later valid packet
                # with the same frame/chunk key must still be accepted.
                self._malformed_datagrams += 1
                return None

            if self._timing_overflow_latched:
                return None
            if packet.frame_id in self._published_frame_ids:
                # Frame identity is (stream, generation, uint32 frame_id).
                # Bounded duplicate-key caches may forget an old packet, but
                # one generation can never publish that identity twice.
                self._out_of_order_frame_drops += 1
                return None
            if (
                packet.frame_id not in self._frame_first_packet_ns
                and len(self._frame_first_packet_ns) >= self.max_remembered_chunks
            ):
                # Never evict a first-packet timestamp and later relabel a
                # surviving partial frame.  This adversarial churn condition
                # latches vision publication stale until the reviewed reset
                # boundary clears all receiver state.
                self._timing_overflow_latched = True
                self._processing_errors += 1
                return None

            self._remember_key(key)
            self._unique_datagrams += 1
            if packet.frame_id not in self._frame_first_packet_ns:
                self._frame_first_packet_ns[packet.frame_id] = received_at_ns
            generation = self._generation
            try:
                completed = self._receiver.feed_packet(raw)
            except ValueError:
                # Defensive: parse_packet above already validated the packet,
                # but receiver-level consistency checks can still reject a
                # conflicting chunk header inside an in-progress frame.
                self._malformed_datagrams += 1
                self._frame_first_packet_ns.pop(packet.frame_id, None)
                return None
            self._timing_ledger_high_watermark = max(
                self._timing_ledger_high_watermark,
                len(self._frame_first_packet_ns),
            )
            self._receiver_buffer_high_watermark = max(
                self._receiver_buffer_high_watermark,
                len(self._receiver._buffers),
            )
            if completed is None:
                return None
            self._frames_reassembled += 1
            first_packet_ns = self._frame_first_packet_ns.pop(completed.frame_id, None)
            if first_packet_ns is None:
                # A bounded timing ledger may evict only under adversarial
                # partial-frame churn.  Drop rather than relabel the final
                # packet as the first and manufacture a shorter latency.
                self._processing_errors += 1
                return None
            final_packet_ns = received_at_ns
            reassembly_complete_ns = self._read_monotonic_ns()

        # JPEG decode can be comparatively expensive; keep reset/snapshot calls
        # responsive and use generation validation when publishing afterward.
        decode_start_ns = self._read_monotonic_ns()
        try:
            camera_frame = decode_jpeg_to_camera_frame(completed)
        except Exception:
            # A corrupt JPEG/OpenCV failure must not kill the dedicated socket
            # worker.  The watchdog will see freshness expire if this persists.
            with self._data_lock:
                if generation == self._generation:
                    self._decode_failures += 1
            return None
        decode_end_ns = self._read_monotonic_ns()

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
            publish_ns = self._read_monotonic_ns()
            if (
                self._last_publish_monotonic_ns is not None
                and publish_ns <= self._last_publish_monotonic_ns
            ):
                self._processing_errors += 1
                return None
            publication_sequence = self._publication_sequence + 1
            timing = FrameTimingV1(
                identity=FrameIdentityV1(
                    stream_id=self.stream_id,
                    generation=generation,
                    frame_id=completed.frame_id,
                ),
                camera_source_time_ns=completed.sim_time_ns,
                host_clock_id=self.host_clock_id,
                publication_sequence=publication_sequence,
                first_unique_packet_monotonic_ns=first_packet_ns,
                final_unique_packet_monotonic_ns=final_packet_ns,
                reassembly_complete_monotonic_ns=reassembly_complete_ns,
                decode_start_monotonic_ns=decode_start_ns,
                decode_end_monotonic_ns=decode_end_ns,
                publish_monotonic_ns=publish_ns,
            )
            self._publication_sequence = publication_sequence
            self._last_publish_monotonic_ns = publish_ns
            snapshot = VQ2VisionSnapshot(
                frame_id=completed.frame_id,
                camera_frame=camera_frame,
                sim_time_ns=completed.sim_time_ns,
                received_monotonic_s=received_at,
                generation=generation,
                timing=timing,
            )
            self._latest_snapshot = snapshot
            self._published_frame_ids.add(completed.frame_id)
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
            if self._capture_snapshot_queue_enabled:
                with self._data_lock:
                    if (
                        len(self._capture_snapshot_queue)
                        >= self.capture_snapshot_queue_capacity
                    ):
                        self._capture_snapshot_queue_dropped += 1
                        self._processing_errors += 1
                    else:
                        self._capture_snapshot_queue.append(snapshot)
                        self._capture_snapshot_queue_high_watermark = max(
                            self._capture_snapshot_queue_high_watermark,
                            len(self._capture_snapshot_queue),
                        )
        return snapshot

    def pop_capture_snapshot(self) -> Optional[VQ2VisionSnapshot]:
        """Pop the oldest capture-loaded publication, if capture is enabled."""

        with self._data_lock:
            if not self._capture_snapshot_queue:
                return None
            return self._capture_snapshot_queue.popleft()

    def capture_snapshot_queue_depth(self) -> int:
        with self._data_lock:
            return len(self._capture_snapshot_queue)

    @contextmanager
    def snapshot_publication_lease(
        self,
        *,
        max_age_s: Optional[float] = None,
        now_monotonic_s: Optional[float] = None,
        acquire_deadline_monotonic_ns: Optional[int] = None,
    ) -> Iterator[Optional[VQ2VisionSnapshot]]:
        """Pin the latest publication during one bounded synchronous action.

        This narrow lease is for a wire-call boundary that must prove no newer
        JPEG can publish between an exact-token check and the synchronous
        transport return.  Callers must not invoke another receiver method
        while holding it.
        """

        max_age = (
            None
            if max_age_s is None
            else _validate_max_age(max_age_s)
        )
        if (
            now_monotonic_s is not None
            and not math.isfinite(now_monotonic_s)
        ):
            raise ValueError("now_monotonic_s must be finite")
        if (
            acquire_deadline_monotonic_ns is not None
            and (
                type(acquire_deadline_monotonic_ns) is not int
                or acquire_deadline_monotonic_ns < 0
            )
        ):
            raise ValueError(
                "acquire_deadline_monotonic_ns must be a non-negative "
                "exact integer"
            )

        if acquire_deadline_monotonic_ns is None:
            acquired = self._data_lock.acquire()
        else:
            now_ns = self._read_monotonic_ns()
            if now_ns >= acquire_deadline_monotonic_ns:
                raise VisionPublicationLeaseAcquisitionTimeout(
                    "vision publication lease acquisition deadline was reached"
                )
            remaining_s = (
                acquire_deadline_monotonic_ns - now_ns
            ) / 1_000_000_000.0
            acquired = self._data_lock.acquire(timeout=remaining_s)
        if not acquired:
            raise VisionPublicationLeaseAcquisitionTimeout(
                "vision publication lease acquisition deadline was reached"
            )
        try:
            snapshot = self._latest_snapshot
            if (
                snapshot is not None
                and max_age is not None
                and not snapshot.is_fresh(max_age, now_monotonic_s)
            ):
                snapshot = None
            yield snapshot
        finally:
            self._data_lock.release()

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
                timing_ledger_entries=len(self._frame_first_packet_ns),
                timing_ledger_high_watermark=self._timing_ledger_high_watermark,
                timing_ledger_capacity=self.max_remembered_chunks,
                timing_overflow_latched=self._timing_overflow_latched,
                receiver_buffered_partial_frames=len(receiver._buffers),
                receiver_buffer_high_watermark=(
                    self._receiver_buffer_high_watermark
                ),
                receiver_buffer_capacity=receiver.max_buffered,
                capture_snapshot_queue_entries=len(
                    self._capture_snapshot_queue
                ),
                capture_snapshot_queue_high_watermark=(
                    self._capture_snapshot_queue_high_watermark
                ),
                capture_snapshot_queue_capacity=(
                    self.capture_snapshot_queue_capacity
                ),
                capture_snapshot_queue_dropped=(
                    self._capture_snapshot_queue_dropped
                ),
                capture_snapshot_queue_enabled=(
                    self._capture_snapshot_queue_enabled
                ),
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

    def _read_monotonic_ns(self) -> int:
        return _validate_monotonic_ns(self._monotonic_ns(), "monotonic_ns clock")

    @staticmethod
    def _normalize_loopback_peer(addr: object) -> Optional[Tuple[str, int]]:
        if type(addr) is not tuple or len(addr) != 2:
            return None
        host, port = addr
        if type(host) is not str or type(port) is not int:
            return None
        if not 1 <= port <= 65_535:
            return None
        try:
            parsed_host = ipaddress.ip_address(host)
        except ValueError:
            return None
        if parsed_host.version != 4 or not parsed_host.is_loopback:
            return None
        return str(parsed_host), port

    def _reject_powered_source_locked(self) -> None:
        self._rejected_source_count += 1
        self._source_rejected_latched = True

    def _accept_powered_source(self, raw: bytes, addr: object) -> bool:
        """Freeze one valid loopback source before production receiver use."""

        if not self._powered_exclusive:
            return True

        peer = self._normalize_loopback_peer(addr)
        with self._data_lock:
            frozen_peer = self._frozen_peer
            if peer is None or (
                frozen_peer is not None and peer != frozen_peer
            ):
                self._reject_powered_source_locked()
                return False
            if frozen_peer is not None:
                return True

        # This parse is intentionally scratch-only.  No duplicate, reassembly,
        # publication, or production receiver state is touched before a source
        # has proved one syntactically valid build-3385 frame chunk.
        try:
            parse_packet(raw)
        except (ValueError, struct.error):
            with self._data_lock:
                self._datagrams_received += 1
                self._malformed_datagrams += 1
            return False

        with self._data_lock:
            if self._frozen_peer is None:
                assert peer is not None
                self._frozen_peer = peer
                self._endpoint_state = "peer_frozen"
                return True
            if self._frozen_peer != peer:
                self._reject_powered_source_locked()
                return False
            return True

    def _receive_loop(self, sock: socket.socket) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    raw, addr = sock.recvfrom(_MAX_UDP_DATAGRAM_BYTES)
                except socket.timeout:
                    continue
                except OSError:
                    if not self._stop_event.is_set():
                        with self._data_lock:
                            self._socket_errors += 1
                    break
                try:
                    if self._accept_powered_source(raw, addr):
                        self._feed_admitted_datagram(raw)
                except Exception:
                    # Keep draining after an unexpected per-datagram failure.
                    # Deliberately do not catch BaseException/SystemExit.
                    with self._data_lock:
                        self._processing_errors += 1
        finally:
            try:
                sock.close()
            except OSError:
                with self._data_lock:
                    self._socket_errors += 1
                    close_proved = self._socket_close_proved
            else:
                with self._data_lock:
                    self._socket_close_proved = True
                    close_proved = True
            with self._data_lock:
                if self._actual_bind is not None and close_proved:
                    self._endpoint_state = (
                        "closed_with_peer"
                        if self._frozen_peer is not None
                        else "closed_without_peer"
                    )


def _validate_max_age(max_age_s: float) -> float:
    max_age = float(max_age_s)
    if not math.isfinite(max_age) or max_age < 0.0:
        raise ValueError("max_age_s must be finite and >= 0")
    return max_age


def _validate_monotonic_ns(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must return an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value
