"""Bridge the sim's runtime track-info packet to the autonomy stack.

The AIGP sim delivers the gate map at runtime (see
:func:`competition.aigp_messages.parse_track_data`): each gate carries a
``gate_id``, an NED position, an orientation quaternion, and a width/height.
:func:`track_data_to_gatespecs` converts that into the platform-agnostic
``GateSpec`` list that ``RacePipeline.configure(gates=...)`` consumes — so the
race plan is built from the sim-provided map instead of a hardcoded JSON.

First contact with the VQ1 sim showed the packet width/height are the gate's
outer dimensions, while ``GateSpec`` expects the passable interior opening.
The sim orientation is also not a raw euler passthrough: the fly-through axis
is gate-local ``+Y`` rotated into world NED, then sign-disambiguated against
course progression. VQ1 gates are upright, so roll is set to zero.
"""
from __future__ import annotations

import math
from typing import List

from competition.aigp_messages import TrackData
from competition.aigp_geometry import AIGP_GATE_BORDER_M
from gate_sequencing.sequencer import GateSpec


GATE_LOCAL_THROUGH_AXIS = (0.0, 1.0, 0.0)


def _normalize_quaternion(gate_id: int, w: float, x: float, y: float, z: float):
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-6:
        raise ValueError(f"gate_id {gate_id} has zero-norm orientation quaternion")
    return w / norm, x / norm, y / norm, z / norm


def _rotated_local_y(w: float, x: float, y: float, z: float):
    # Rotation-matrix column for local +Y with quaternion order [w, x, y, z].
    return (
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    )


def _normalize_vector(v):
    norm = math.sqrt(sum(c * c for c in v))
    if norm < 1e-9:
        return None
    return tuple(c / norm for c in v)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _interior_dimension(packet_dim: float) -> float:
    interior = packet_dim - 2.0 * AIGP_GATE_BORDER_M
    if interior < 0.5:
        return packet_dim
    return interior


def _heading_from_normal(n) -> float:
    yaw = math.atan2(n[1], n[0])
    if math.isclose(yaw, -math.pi, abs_tol=1e-12):
        return math.pi
    return yaw


def track_data_to_gatespecs(track: TrackData) -> List[GateSpec]:
    """Convert a :class:`TrackData` into ordered ``GateSpec``s.

    Gates are ordered by ``gate_id`` (the sim numbers them in pass order),
    and ``sequence_index`` is assigned in that order. Positions stay in NED
    with no frame conversion. Packet dimensions are treated as outer size
    and converted to the passable interior opening, with a fallback for legacy
    tests/fixtures that already provide interior dimensions.
    """
    seen = set()
    for gate in track.gates:
        if gate.gate_id in seen:
            raise ValueError(f"duplicate gate_id {gate.gate_id}")
        seen.add(gate.gate_id)

    ordered = sorted(track.gates, key=lambda g: g.gate_id)
    specs: List[GateSpec] = []
    for seq, gate in enumerate(ordered):
        q = gate.orientation
        w, x, y, z = _normalize_quaternion(gate.gate_id, q.w, q.x, q.y, q.z)
        n = _normalize_vector(_rotated_local_y(w, x, y, z))
        if n is None:
            raise ValueError(f"gate_id {gate.gate_id} has invalid through-axis")

        if len(ordered) > 1:
            if seq < len(ordered) - 1:
                other = ordered[seq + 1]
                tangent = tuple(b - a for a, b in zip(gate.position_ned, other.position_ned))
            else:
                other = ordered[seq - 1]
                tangent = tuple(a - b for a, b in zip(gate.position_ned, other.position_ned))
            t = _normalize_vector(tangent)
            if t is not None and _dot(n, t) < 0.0:
                n = tuple(-c for c in n)

        nz = max(-1.0, min(1.0, n[2]))
        yaw = _heading_from_normal(n)
        pitch = -math.asin(nz)

        specs.append(
            GateSpec(
                gate_id=str(gate.gate_id),
                position=gate.position_ned,
                yaw=yaw,
                pitch=pitch,
                roll=0.0,
                interior_width=_interior_dimension(gate.width),
                interior_height=_interior_dimension(gate.height),
                sequence_index=seq,
            )
        )
    return specs
