from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from .model_types import Field, FieldConfig, Gate, Vec3


def generate_field(config: FieldConfig, gates: Iterable[Gate]) -> Field:
    gate_list = list(gates)
    ids = [g.gate_id for g in gate_list]
    if len(set(ids)) != len(ids):
        raise ValueError("Field requires unique gate_id values")
    return Field(config=config, gates=gate_list)


def get_gate(field: Field, gate_id: str) -> Optional[Gate]:
    for gate in field.gates:
        if gate.gate_id == gate_id:
            return gate
    return None


def nearest_gate(field: Field, position: Vec3) -> Optional[Gate]:
    if not field.gates:
        return None

    def distance_sq(gate: Gate) -> float:
        gx, gy, gz = gate.pose.position
        return (gx - position[0]) ** 2 + (gy - position[1]) ** 2 + (gz - position[2]) ** 2

    return min(field.gates, key=distance_sq)


def visible_gate_prefilter(field: Field, camera_position: Vec3, radius_m: float = 50.0) -> List[Gate]:
    if radius_m <= 0:
        return []
    radius_sq = radius_m * radius_m
    out: List[Gate] = []
    for gate in field.gates:
        gx, gy, gz = gate.pose.position
        d_sq = (gx - camera_position[0]) ** 2 + (gy - camera_position[1]) ** 2 + (gz - camera_position[2]) ** 2
        if d_sq <= radius_sq:
            out.append(gate)
    return out


def is_outside_bounds(field: Field, position: Vec3) -> bool:
    mins = field.config.bounds_min
    maxs = field.config.bounds_max
    return any(position[idx] < mins[idx] or position[idx] > maxs[idx] for idx in range(3))
