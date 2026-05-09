"""Adapter: simulation.model_types.Gate -> gate_sequencing.GateSpec.

Lets the platform-agnostic GateSequencer (gate_sequencing.sequencer)
drive the sim_pybullet runner without a duplicate sequencer
implementation. See P2-1 in
`.research_loop/2026-05-09-150-agent-reconciliation.md`.
"""

from __future__ import annotations

from gate_sequencing.sequencer import GateSpec
from simulation.model_types import Gate


def to_spec(gate: Gate) -> GateSpec:
    """Project a sim Gate into the platform-agnostic GateSpec shape.

    Sequence index defaults to 0 if the sim Gate didn't set one — the
    sequencer accepts that, just be aware sort order will fall through to
    list position.
    """
    pose = gate.pose
    cfg = gate.config
    return GateSpec(
        gate_id=gate.gate_id,
        position=(float(pose.x), float(pose.y), float(pose.z)),
        yaw=float(pose.yaw),
        pitch=float(pose.pitch),
        roll=float(pose.roll),
        interior_width=float(cfg.interior_width_m),
        interior_height=float(cfg.interior_height_m),
        border_width=float(cfg.border_width_m),
        sequence_index=int(gate.sequence_index) if gate.sequence_index is not None else 0,
    )
