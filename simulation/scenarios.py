from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .field import generate_field
from .gates import generate_gate
from .pathing import build_path
from .model_types import Field, FieldConfig, GateConfig, PathPolyline, PathSpec, Pose3D

DEFAULT_SCENE_CONFIG = Path(__file__).resolve().parent / "configs" / "field_demo.yaml"


def build_sample_field(config_path: Path | None = None) -> Field:
    scene = _load_scene_config(config_path or DEFAULT_SCENE_CONFIG)
    return _build_field_from_scene(scene)


def build_sample_path(config_path: Path | None = None) -> PathPolyline:
    scene = _load_scene_config(config_path or DEFAULT_SCENE_CONFIG)
    path_data = scene.get("path", {})
    control_points = [tuple(point) for point in path_data.get("control_points", [])]
    if not control_points:
        raise ValueError("Scene config path.control_points must include at least 2 points")
    spec = PathSpec(
        control_points=control_points,
        samples_per_segment=int(path_data.get("samples_per_segment", 20)),
        closed=bool(path_data.get("closed", False)),
    )
    return build_path(spec)


def _load_scene_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scene config not found: {path}")

    text = path.read_text(encoding="utf-8")
    loaded = None
    if yaml is not None:
        loaded = yaml.safe_load(text)
    else:
        # Dependency-free fallback: support JSON-formatted YAML files.
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ValueError(f"Scene config must be a YAML mapping: {path}")
    return loaded


def _build_field_from_scene(scene: Dict[str, Any]) -> Field:
    field_cfg = scene.get("field", {})
    config = FieldConfig(
        name=str(field_cfg.get("name", "default-field")),
        bounds_min=tuple(field_cfg.get("bounds_min", (0.0, -10.0, 0.0))),
        bounds_max=tuple(field_cfg.get("bounds_max", (35.0, 10.0, 8.0))),
    )

    gate_defaults = scene.get("gate_defaults", {})
    gates_data = scene.get("gates", [])
    gates = [_build_gate_from_data(gate_defaults, gate_data) for gate_data in gates_data]
    return generate_field(config, gates)


def _build_gate_from_data(gate_defaults: Dict[str, Any], gate_data: Dict[str, Any]):
    merged = dict(gate_defaults)
    merged.update(gate_data.get("config", {}))
    pose_raw = gate_data.get("pose", {})
    pose = Pose3D(
        x=float(pose_raw.get("x", 0.0)),
        y=float(pose_raw.get("y", 0.0)),
        z=float(pose_raw.get("z", 0.0)),
        yaw=float(pose_raw.get("yaw", 0.0)),
        pitch=float(pose_raw.get("pitch", 0.0)),
        roll=float(pose_raw.get("roll", 0.0)),
    )
    gate_config = GateConfig(
        gate_type=str(merged.get("gate_type", "square")),
        interior_width_m=float(merged.get("interior_width_m", 1.0)),
        interior_height_m=float(merged.get("interior_height_m", 1.0)),
        border_width_m=float(merged.get("border_width_m", 0.15)),
        depth_m=float(merged.get("depth_m", 0.08)),
        color=str(merged.get("color", "red")),
        label=str(merged.get("label", "")),
    )
    gate_id = str(gate_data["id"])
    seq = gate_data.get("sequence_index")
    sequence_index = int(seq) if seq is not None else None
    return generate_gate(gate_config, pose, gate_id=gate_id, sequence_index=sequence_index)
