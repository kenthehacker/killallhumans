"""Bridge the sim's runtime track-info packet to the autonomy stack.

The AIGP sim delivers the gate map at runtime (see
:func:`competition.aigp_messages.parse_track_data`): each gate carries a
``gate_id``, an NED position, an orientation quaternion, and a width/height.
:func:`track_data_to_gatespecs` converts that into the platform-agnostic
``GateSpec`` list that ``RacePipeline.configure(gates=...)`` consumes — so the
race plan is built from the sim-provided map instead of a hardcoded JSON.

Assumption (flag for first-contact verification): the packet's ``width``/
``height`` are treated as the **passable interior opening** (mapped to
``GateSpec.interior_width/height``). VADR-TS-002 gives interior 1.5 m vs outer
2.7 m; the recorder logs the real values so we can confirm which they are.
Frame thickness (``border_width``) and ``depth`` are not in the packet, so
they stay at the AIGP spec defaults baked into ``GateSpec``.
"""
from __future__ import annotations

from typing import List

from competition.aigp_messages import TrackData
from gate_sequencing.sequencer import GateSpec


def track_data_to_gatespecs(track: TrackData) -> List[GateSpec]:
    """Convert a :class:`TrackData` into ordered ``GateSpec``s.

    Gates are ordered by ``gate_id`` (the sim numbers them in pass order),
    and ``sequence_index`` is assigned in that order.
    """
    specs: List[GateSpec] = []
    for seq, gate in enumerate(sorted(track.gates, key=lambda g: g.gate_id)):
        roll, pitch, yaw = gate.orientation.to_euler()
        specs.append(
            GateSpec(
                gate_id=str(gate.gate_id),
                position=gate.position_ned,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                interior_width=gate.width,
                interior_height=gate.height,
                sequence_index=seq,
            )
        )
    return specs
