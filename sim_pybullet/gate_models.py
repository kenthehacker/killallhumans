"""
Gate creation and visual management for the PyBullet simulation.

Creates gate frames as static collision bodies with visual meshes.
Supports highlighting (changing color) for gate sequencing.
"""

import math
from typing import Tuple, Optional, Dict, List

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None

# Re-use the shared Gate / GateConfig types from the lightweight simulation
from simulation.model_types import Gate, GateConfig, Pose3D

# Visual states
COLOR_DEFAULT = {
    "red": [0.85, 0.2, 0.2, 1.0],
    "blue": [0.25, 0.45, 0.9, 1.0],
    "green": [0.2, 0.75, 0.3, 1.0],
    "yellow": [0.9, 0.85, 0.2, 1.0],
    "orange": [0.9, 0.55, 0.2, 1.0],
    "purple": [0.65, 0.35, 0.85, 1.0],
    "white": [0.85, 0.85, 0.85, 1.0],
}
COLOR_HIGHLIGHT = [0.1, 1.0, 0.2, 1.0]  # bright green
COLOR_PASSED = [0.3, 0.3, 0.3, 0.5]  # dim gray
COLOR_FUTURE = None  # use default


def _gate_color(gate: Gate) -> list:
    name = gate.config.color.lower()
    return list(COLOR_DEFAULT.get(name, [0.7, 0.7, 0.7, 1.0]))


def create_gate_body(
    physics_client: int,
    gate: Gate,
) -> List[int]:
    """
    Create a gate frame in PyBullet as 4 box segments (top, bottom, left, right).
    Returns list of body IDs for all segments.
    """
    iw = max(gate.config.interior_width_m, 0.05)
    ih = max(gate.config.interior_height_m, 0.05)
    bw = max(gate.config.border_width_m, 0.01)
    depth = max(gate.config.depth_m, 0.02)
    ow = iw + 2 * bw

    color = _gate_color(gate)

    segments = [
        # (local_center, half_extents)
        # Gate lies in the local YZ plane so it faces the approaching drone in +X.
        # Top bar — wide in Y
        ((0.0, 0.0, ih / 2 + bw / 2), (depth / 2, ow / 2, bw / 2)),
        # Bottom bar — wide in Y
        ((0.0, 0.0, -ih / 2 - bw / 2), (depth / 2, ow / 2, bw / 2)),
        # Left bar — positioned at -Y
        ((0.0, -iw / 2 - bw / 2, 0.0), (depth / 2, bw / 2, ih / 2)),
        # Right bar — positioned at +Y
        ((0.0, iw / 2 + bw / 2, 0.0), (depth / 2, bw / 2, ih / 2)),
    ]

    body_ids = []
    for local_center, half_ext in segments:
        world_pos, world_orn = _local_to_world(
            local_center, gate.pose
        )

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=list(half_ext),
            physicsClientId=physics_client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(half_ext),
            rgbaColor=color,
            physicsClientId=physics_client,
        )
        body_id = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=world_pos,
            baseOrientation=world_orn,
            physicsClientId=physics_client,
        )
        body_ids.append(body_id)

    return body_ids


def highlight_gate(
    physics_client: int,
    body_ids: List[int],
):
    """Set gate segments to bright highlight color (current target)."""
    for bid in body_ids:
        p.changeVisualShape(
            bid, -1, rgbaColor=COLOR_HIGHLIGHT,
            physicsClientId=physics_client,
        )


def dim_gate(
    physics_client: int,
    body_ids: List[int],
):
    """Set gate segments to dim gray (already passed)."""
    for bid in body_ids:
        p.changeVisualShape(
            bid, -1, rgbaColor=COLOR_PASSED,
            physicsClientId=physics_client,
        )


def reset_gate_color(
    physics_client: int,
    body_ids: List[int],
    gate: Gate,
):
    """Restore gate to its default color."""
    color = _gate_color(gate)
    for bid in body_ids:
        p.changeVisualShape(
            bid, -1, rgbaColor=color,
            physicsClientId=physics_client,
        )


def _local_to_world(
    local_pos: Tuple[float, float, float],
    pose: Pose3D,
) -> Tuple[list, list]:
    """Transform a local-frame position to world frame given a Pose3D."""
    x, y, z = local_pos

    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    x1, y1, z1 = x, cr * y - sr * z, sr * y + cr * z

    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    x2, y2, z2 = cp * x1 + sp * z1, y1, -sp * x1 + cp * z1

    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    x3, y3, z3 = cy * x2 - sy * y2, sy * x2 + cy * y2, z2

    world_pos = [x3 + pose.x, y3 + pose.y, z3 + pose.z]

    # Compose gate orientation quaternion
    gate_quat = p.getQuaternionFromEuler([pose.roll, pose.pitch, pose.yaw])
    return world_pos, list(gate_quat)
