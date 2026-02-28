from __future__ import annotations

from .field import generate_field
from .gates import generate_gate
from .pathing import build_path
from .model_types import Field, FieldConfig, GateConfig, PathPolyline, PathSpec, Pose3D


def build_sample_field() -> Field:
    gates = [
        generate_gate(GateConfig(color="red"), Pose3D(8.0, 0.0, 1.8), "gate-1", sequence_index=1),
        generate_gate(GateConfig(color="blue"), Pose3D(16.0, 4.0, 2.0), "gate-2", sequence_index=2),
        generate_gate(GateConfig(color="green"), Pose3D(24.0, -2.0, 1.8), "gate-3", sequence_index=3),
    ]
    return generate_field(FieldConfig(bounds_min=(0.0, -10.0, 0.0), bounds_max=(35.0, 10.0, 8.0)), gates)


def build_sample_path() -> PathPolyline:
    spec = PathSpec(
        control_points=[
            (0.0, 0.0, 1.5),
            (8.0, 0.0, 1.8),
            (16.0, 4.0, 2.0),
            (24.0, -2.0, 1.8),
            (32.0, 0.0, 1.5),
        ],
        samples_per_segment=20,
    )
    return build_path(spec)
