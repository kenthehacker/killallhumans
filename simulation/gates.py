from __future__ import annotations

from .model_types import Gate, GateConfig, Pose3D


def generate_gate(config: GateConfig, pose: Pose3D, gate_id: str, sequence_index: int | None = None) -> Gate:
    if not gate_id:
        raise ValueError("gate_id must be non-empty")
    return Gate(gate_id=gate_id, config=config, pose=pose, sequence_index=sequence_index)
