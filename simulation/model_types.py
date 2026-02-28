from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Pixel = Tuple[int, int]


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def position(self) -> Vec3:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class GateConfig:
    gate_type: str = "square"
    width_m: float = 1.0
    height_m: float = 1.0
    frame_thickness_m: float = 0.1
    color: str = "red"
    label: str = ""

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("Gate dimensions must be positive")
        if self.frame_thickness_m <= 0:
            raise ValueError("Gate frame thickness must be positive")


@dataclass(frozen=True)
class Gate:
    gate_id: str
    config: GateConfig
    pose: Pose3D
    sequence_index: Optional[int] = None


@dataclass(frozen=True)
class FieldConfig:
    bounds_min: Vec3 = (0.0, 0.0, 0.0)
    bounds_max: Vec3 = (50.0, 50.0, 10.0)
    seed: Optional[int] = None
    name: str = "default-field"

    def __post_init__(self) -> None:
        if any(mn >= mx for mn, mx in zip(self.bounds_min, self.bounds_max)):
            raise ValueError("Field bounds_min must be strictly less than bounds_max")


@dataclass
class Field:
    config: FieldConfig
    gates: List[Gate]


@dataclass(frozen=True)
class CameraPose:
    pose: Pose3D
    fov_horizontal_deg: float = 90.0
    resolution_width: int = 640
    resolution_height: int = 480
    near_plane_m: float = 0.05
    far_plane_m: float = 200.0

    def __post_init__(self) -> None:
        if not (1.0 < self.fov_horizontal_deg < 179.0):
            raise ValueError("fov_horizontal_deg must be in (1, 179)")
        if self.resolution_width <= 0 or self.resolution_height <= 0:
            raise ValueError("Camera resolution must be positive")
        if self.near_plane_m <= 0 or self.far_plane_m <= self.near_plane_m:
            raise ValueError("Invalid near/far plane configuration")

    @property
    def fov_vertical_deg(self) -> float:
        aspect = self.resolution_height / self.resolution_width
        half_h = math.tan(math.radians(self.fov_horizontal_deg) * 0.5)
        half_v = half_h * aspect
        return math.degrees(2.0 * math.atan(half_v))


@dataclass(frozen=True)
class VisibleGateAnnotation:
    gate_id: str
    center_pixel: Pixel
    approx_bbox: Tuple[int, int, int, int]
    depth_m: float


@dataclass
class CameraFrame:
    rgb: List[List[Tuple[int, int, int]]]
    pose: CameraPose
    visible_gates: List[VisibleGateAnnotation] = field(default_factory=list)
    outside_field: bool = False
    depth: Optional[List[List[float]]] = None


@dataclass(frozen=True)
class PathSpec:
    control_points: Sequence[Vec3]
    method: str = "catmull_rom"
    samples_per_segment: int = 30
    closed: bool = False

    def __post_init__(self) -> None:
        if len(self.control_points) < 2:
            raise ValueError("PathSpec requires at least 2 control points")
        if self.samples_per_segment <= 0:
            raise ValueError("samples_per_segment must be positive")
        if self.method != "catmull_rom":
            raise ValueError("Only 'catmull_rom' path method is supported in MVP")


@dataclass
class PathPolyline:
    points: List[Vec3]
    cumulative_lengths: List[float]

    @property
    def total_length(self) -> float:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0.0


@dataclass
class DetectionLike:
    normalized_center_x: float
    normalized_center_y: float
    estimated_distance: float
