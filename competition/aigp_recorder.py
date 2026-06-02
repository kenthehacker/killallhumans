"""AIGP first-contact black-box recorder.

Connects to the AIGP sim exactly like the official ``PyAIPilotExample``
(MAVLink over ``udpin:<ip>:14550`` + the JPEG vision stream on UDP 5600) and
writes every received message to a timestamped JSONL log. The point is to
answer the open questions that only the live binary can settle (plan §4):

  * Are ``LOCAL_POSITION_NED`` / ``ODOMETRY`` populated (local position)?
  * Is the track-info packet actually sent (runtime gate map)?
  * What do ``COLLISION`` / race-status / actuator messages look like?

The recorded JSONL then doubles as an offline replay fixture so the transport
adapter can be developed without burning sim time.

Design: the parsing/normalization is pure (``record_for_message``,
``mavlink_msg_to_fields``, ``race_status_fields``, ``write_jsonl``) and unit
tested. ``run`` is the thin socket/pymavlink shell — it imports pymavlink
lazily and is validated on first contact, not in CI.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Iterable, List, Optional, TextIO

from competition.aigp_messages import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    TrackInfoReassembler,
    parse_race_status,
)

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


def mavlink_msg_to_fields(msg) -> Dict:
    """Extract the fields of interest from a pymavlink message object.

    Unknown / not-yet-modeled types return ``{}`` — they're still recorded by
    their type label so the first-contact log shows everything the sim emits.
    """
    t = msg.get_type()
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
    return {}


def write_jsonl(records: Iterable[Dict], fileobj: TextIO) -> int:
    """Write each record as one JSON object per line. Returns the count."""
    n = 0
    for r in records:
        fileobj.write(json.dumps(r))
        fileobj.write("\n")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Live loop (thin shell — validated on first contact, not in CI)
# ---------------------------------------------------------------------------
def run(
    out_path: str,
    ip: str = DEFAULT_SIM_IP,
    port: int = DEFAULT_MAVLINK_PORT,
    duration_s: Optional[float] = None,
) -> None:  # pragma: no cover - requires a live MAVLink endpoint
    """Connect to the sim and stream every MAVLink message to ``out_path``.

    Mirrors PyAIPilotExample's connection (``udpin:ip:port`` → wait_heartbeat
    → recv_match). ENCAPSULATED_DATA is dispatched to the race-status parser
    or the track-info reassembler; everything else is normalized by
    ``mavlink_msg_to_fields``. Stops after ``duration_s`` (None = until Ctrl-C).
    """
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pymavlink is required for the live recorder: pip install pymavlink"
        ) from exc

    conn = mavutil.mavlink_connection(f"udpin:{ip}:{port}")
    print(f"[recorder] waiting for heartbeat on udpin:{ip}:{port} ...", flush=True)
    conn.wait_heartbeat()
    print(f"[recorder] connected to system {conn.target_system}", flush=True)

    reassembler = TrackInfoReassembler()
    start = time.monotonic()
    written = 0
    with open(out_path, "w") as f:
        try:
            while duration_s is None or (time.monotonic() - start) < duration_s:
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
                                "track_data", {"num_gates": track.num_gates}, recv_ns
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
    print(f"[recorder] wrote {written} records to {out_path}", flush=True)


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="AIGP first-contact MAVLink recorder")
    ap.add_argument("--out", default="aigp_recording.jsonl", help="output JSONL path")
    ap.add_argument("--ip", default=DEFAULT_SIM_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_MAVLINK_PORT)
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl-C)")
    args = ap.parse_args(argv)
    run(out_path=args.out, ip=args.ip, port=args.port, duration_s=args.duration)


if __name__ == "__main__":  # pragma: no cover
    main()
