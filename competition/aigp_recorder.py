"""AIGP first-contact black-box recorder.

Connects to the AIGP sim exactly like the official ``PyAIPilotExample``
(MAVLink over ``udpin:<ip>:14550`` + the JPEG vision stream on UDP 5600) and
writes every received message to a timestamped JSONL log. The point is to
answer the open questions that only the live binary can settle (plan §4):

  * Are ``LOCAL_POSITION_NED`` / ``ODOMETRY`` populated (local position)?
  * Is the track-info packet actually sent (runtime gate map)?
  * What do ``COLLISION`` / race-status / actuator messages look like?
  * Vision: does the port-5600 stream arrive, at what rate / size?

Faithfulness over prettiness — this log is how we *discover* the sim's
behaviour, so the recorder never silently drops data:
  * messages we model get clean, normalized field names;
  * everything else is dumped verbatim via ``msg.to_dict()``;
  * a curated extractor that hits a field the real wire doesn't carry falls
    back to that same generic dump instead of crashing mid-capture;
  * the track-info packet logs every gate's full NED pose, not just a count.

The recorded JSONL then doubles as an offline replay fixture so the transport
adapter can be developed without burning sim time.

Design: the parsing/normalization is pure (``record_for_message``,
``mavlink_msg_to_fields``, ``race_status_fields``, ``track_data_fields``,
``vision_frame_record``, ``write_jsonl``) and unit tested. ``run`` is the thin
socket/pymavlink shell (plus a daemon vision-capture thread) — it imports
pymavlink lazily and is validated on first contact, not in CI.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from typing import Dict, Iterable, List, Optional, TextIO

from competition.aigp_messages import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    TrackInfoReassembler,
    parse_race_status,
)
from competition.vision_udp import ReassembledFrame, VisionUdpReceiver

# Defaults from the official PyAIPilotExample (main.py / setup.py).
DEFAULT_SIM_IP: str = "127.0.0.1"
DEFAULT_MAVLINK_PORT: int = 14550
DEFAULT_VISION_PORT: int = 5600


# ---------------------------------------------------------------------------
# Pure core (unit tested)
# ---------------------------------------------------------------------------
def record_for_message(
    msg_type: str, fields: Dict, recv_wall_ns: int
) -> Dict:
    """Build one JSONL record: a type tag, the receive timestamp, and fields."""
    return {"type": msg_type, "recv_wall_ns": recv_wall_ns, **fields}


def race_status_fields(payload: bytes) -> Dict:
    """Normalize a race-status ENCAPSULATED_DATA payload to a flat dict."""
    rs = parse_race_status(payload)
    return {
        "sim_boot_time_ms": rs.sim_boot_time_ms,
        "race_start_boot_time_ms": rs.race_start_boot_time_ms,
        "race_finish_time_ns": rs.race_finish_time_ns,
        "active_gate_index": rs.active_gate_index,
        "last_gate_race_time": rs.last_gate_race_time,
        "race_started": rs.race_started,
        "race_finished": rs.race_finished,
    }


def track_data_fields(track) -> Dict:
    """Flatten a parsed ``TrackData`` to JSON — every gate's full NED pose.

    The gate map is a primary §4 question, so we log each gate (id, NED
    position, orientation quaternion w,x,y,z, width, height), not just the
    count the way the first draft did.
    """
    return {
        "num_gates": track.num_gates,
        "gates": [
            {
                "gate_id": g.gate_id,
                "position_ned": list(g.position_ned),
                "orientation_wxyz": [
                    g.orientation.w, g.orientation.x,
                    g.orientation.y, g.orientation.z,
                ],
                "width": g.width,
                "height": g.height,
            }
            for g in track.gates
        ],
    }


def _curated_fields(t: str, msg) -> Optional[Dict]:
    """Clean, normalized fields for the message types we model explicitly.

    Returns ``None`` for types we don't model (caller falls back to the
    generic dump). May raise ``AttributeError`` if the live message lacks a
    field we assumed — the caller catches that and falls back too, so a wrong
    field-name guess degrades to a verbatim dump instead of killing capture.
    """
    if t == "ATTITUDE":
        return {
            "roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw,
            "rollspeed": msg.rollspeed, "pitchspeed": msg.pitchspeed,
            "yawspeed": msg.yawspeed, "time_boot_ms": msg.time_boot_ms,
        }
    if t == "LOCAL_POSITION_NED":
        return {
            "x": msg.x, "y": msg.y, "z": msg.z,
            "vx": msg.vx, "vy": msg.vy, "vz": msg.vz,
            "time_boot_ms": msg.time_boot_ms,
        }
    if t == "ODOMETRY":
        return {
            "x": msg.x, "y": msg.y, "z": msg.z, "q": list(msg.q),
            "vx": msg.vx, "vy": msg.vy, "vz": msg.vz,
            "time_usec": msg.time_usec,
        }
    if t == "HIGHRES_IMU":
        return {
            "xacc": msg.xacc, "yacc": msg.yacc, "zacc": msg.zacc,
            "xgyro": msg.xgyro, "ygyro": msg.ygyro, "zgyro": msg.zgyro,
            "time_usec": msg.time_usec,
        }
    if t == "COLLISION":
        return {
            "id": msg.id, "threat_level": msg.threat_level,
            "impulse": msg.horizontal_minimum_delta,
        }
    if t == "HEARTBEAT":
        return {"base_mode": msg.base_mode, "custom_mode": msg.custom_mode}
    if t == "TIMESYNC":
        return {"tc1": msg.tc1, "ts1": msg.ts1}
    if t == "ACTUATOR_OUTPUT_STATUS":
        return {"time_usec": msg.time_usec, "actuator": list(msg.actuator)}
    return None


def _generic_fields(msg) -> Dict:
    """Verbatim dump of every field on a pymavlink message.

    Used for any type ``_curated_fields`` doesn't model (or can't extract).
    Drops the redundant ``mavpackettype`` tag — the record already carries
    ``type``.
    """
    d = dict(msg.to_dict())
    d.pop("mavpackettype", None)
    return d


def mavlink_msg_to_fields(msg) -> Dict:
    """Extract the fields of interest from a pymavlink message object.

    Modeled types return curated, normalized names. Anything else — or any
    modeled type whose assumed field is missing on the real wire — is dumped
    verbatim via ``to_dict()`` so the first-contact log never loses data.
    """
    t = msg.get_type()
    try:
        curated = _curated_fields(t, msg)
    except AttributeError:
        curated = None
    if curated is not None:
        return curated
    return _generic_fields(msg)


def vision_frame_record(
    frame: ReassembledFrame, recv_wall_ns: int, jpeg_path: Optional[str] = None
) -> Dict:
    """Build a JSONL record for one reassembled vision frame (the frame index).

    Records metadata (frame id, sim timestamp, JPEG size); the JPEG bytes
    themselves are only persisted when ``jpeg_path`` is given (``--jpeg-dir``).
    """
    rec = {
        "type": "vision_frame",
        "recv_wall_ns": recv_wall_ns,
        "frame_id": frame.frame_id,
        "sim_time_ns": frame.sim_time_ns,
        "jpeg_size": len(frame.jpeg_bytes),
    }
    if jpeg_path is not None:
        rec["jpeg_path"] = jpeg_path
    return rec


def _json_default(o):
    """Make non-JSON-native values (e.g. bytes array fields) serializable."""
    if isinstance(o, (bytes, bytearray)):
        return list(o)
    return str(o)


def write_jsonl(records: Iterable[Dict], fileobj: TextIO) -> int:
    """Write each record as one JSON object per line. Returns the count."""
    n = 0
    for r in records:
        fileobj.write(json.dumps(r, default=_json_default))
        fileobj.write("\n")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Live loop (thin shell — validated on first contact, not in CI)
# ---------------------------------------------------------------------------
def _vision_capture_loop(
    port: int,
    out_q: "queue.Queue",
    stop_event: "threading.Event",
    jpeg_dir: Optional[str],
) -> None:  # pragma: no cover - requires a live UDP vision stream
    """Bind UDP ``port`` (5600), reassemble JPEG frames, enqueue a record per
    completed frame. Daemon thread; the main loop is the sole file writer."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    receiver = VisionUdpReceiver(port=port)
    if jpeg_dir:
        os.makedirs(jpeg_dir, exist_ok=True)
    print(f"[recorder] vision capture on udp/{port}", flush=True)
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            frame = receiver.feed_packet(data)
        except ValueError:
            continue
        if frame is None:
            continue
        recv_ns = time.time_ns()
        jpeg_path = None
        if jpeg_dir:
            jpeg_path = os.path.join(jpeg_dir, f"frame_{frame.frame_id:08d}.jpg")
            try:
                with open(jpeg_path, "wb") as jf:
                    jf.write(frame.jpeg_bytes)
            except OSError:
                jpeg_path = None
        out_q.put(vision_frame_record(frame, recv_ns, jpeg_path))
    sock.close()


def run(
    out_path: str,
    ip: str = DEFAULT_SIM_IP,
    port: int = DEFAULT_MAVLINK_PORT,
    vision_port: int = DEFAULT_VISION_PORT,
    jpeg_dir: Optional[str] = None,
    record_vision: bool = True,
    duration_s: Optional[float] = None,
) -> None:  # pragma: no cover - requires a live MAVLink endpoint
    """Connect to the sim and stream every MAVLink message (and the vision
    frame index) to ``out_path``.

    Mirrors PyAIPilotExample's connection (``udpin:ip:port`` → wait_heartbeat
    → recv_match). ENCAPSULATED_DATA is dispatched to the race-status parser
    or the track-info reassembler; everything else is normalized by
    ``mavlink_msg_to_fields``. A daemon thread captures the port-5600 vision
    stream in parallel; the main loop drains its queue and is the only writer.
    Stops after ``duration_s`` (None = until Ctrl-C).
    """
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pymavlink is required for the live recorder: pip install pymavlink"
        ) from exc

    vision_q: "queue.Queue" = queue.Queue()
    vision_stop = threading.Event()
    vision_thread: Optional[threading.Thread] = None
    if record_vision:
        vision_thread = threading.Thread(
            target=_vision_capture_loop,
            args=(vision_port, vision_q, vision_stop, jpeg_dir),
            daemon=True,
        )
        vision_thread.start()

    conn = mavutil.mavlink_connection(f"udpin:{ip}:{port}")
    print(f"[recorder] waiting for heartbeat on udpin:{ip}:{port} ...", flush=True)
    conn.wait_heartbeat()
    print(f"[recorder] connected to system {conn.target_system}", flush=True)

    reassembler = TrackInfoReassembler()
    start = time.monotonic()
    written = 0

    def _drain_vision(fileobj) -> int:
        n = 0
        while True:
            try:
                vrec = vision_q.get_nowait()
            except queue.Empty:
                break
            write_jsonl([vrec], fileobj)
            n += 1
        return n

    with open(out_path, "w") as f:
        try:
            while duration_s is None or (time.monotonic() - start) < duration_s:
                written += _drain_vision(f)
                msg = conn.recv_match(blocking=False)
                if msg is None:
                    time.sleep(0.001)
                    continue
                t = msg.get_type()
                if t == "BAD_DATA":
                    continue
                recv_ns = time.time_ns()

                if t == "DATA_TRANSMISSION_HANDSHAKE":
                    # Repurposed by the sim to open a chunked track-info transfer.
                    reassembler.begin_transfer(msg.width, msg.packets)
                    rec = record_for_message(
                        "track_transfer_begin",
                        {"transfer_id": msg.width, "num_chunks": msg.packets},
                        recv_ns,
                    )
                elif t == "ENCAPSULATED_DATA":
                    payload = bytes(msg.data)
                    data_type = payload[0] if payload else -1
                    if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
                        rec = record_for_message(
                            "race_status", race_status_fields(payload), recv_ns
                        )
                    elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
                        # Per-chunk header is "<BH" (data_type, transfer_id);
                        # the chunk body is the remainder.
                        transfer_id = int.from_bytes(payload[1:3], "little")
                        track = reassembler.feed_chunk(
                            transfer_id, msg.seqnr, payload[3:]
                        )
                        if track is not None:
                            rec = record_for_message(
                                "track_data", track_data_fields(track), recv_ns
                            )
                        else:
                            rec = record_for_message(
                                "track_chunk",
                                {"transfer_id": transfer_id, "seqnr": msg.seqnr},
                                recv_ns,
                            )
                    else:
                        rec = record_for_message(
                            "encapsulated_unknown", {"data_type": data_type}, recv_ns
                        )
                else:
                    rec = record_for_message(t, mavlink_msg_to_fields(msg), recv_ns)

                write_jsonl([rec], f)
                written += 1
                if written % 200 == 0:
                    f.flush()
                    print(f"[recorder] {written} records", flush=True)
        except KeyboardInterrupt:
            print("\n[recorder] stopped by user", flush=True)
        finally:
            vision_stop.set()
            written += _drain_vision(f)
            f.flush()
    if vision_thread is not None:
        vision_thread.join(timeout=1.0)
    print(f"[recorder] wrote {written} records to {out_path}", flush=True)


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="AIGP first-contact MAVLink + vision recorder"
    )
    ap.add_argument("--out", default="aigp_recording.jsonl", help="output JSONL path")
    ap.add_argument("--ip", default=DEFAULT_SIM_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_MAVLINK_PORT)
    ap.add_argument("--vision-port", type=int, default=DEFAULT_VISION_PORT)
    ap.add_argument(
        "--jpeg-dir", default=None,
        help="if set, dump each frame's JPEG into this dir (frame index always logged)",
    )
    ap.add_argument(
        "--no-vision", action="store_true",
        help="skip the port-5600 vision capture thread",
    )
    ap.add_argument(
        "--duration", type=float, default=None,
        help="seconds (default: until Ctrl-C)",
    )
    args = ap.parse_args(argv)
    run(
        out_path=args.out, ip=args.ip, port=args.port,
        vision_port=args.vision_port, jpeg_dir=args.jpeg_dir,
        record_vision=not args.no_vision, duration_s=args.duration,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
