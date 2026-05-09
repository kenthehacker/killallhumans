"""
PyBullet drone racing environment.

Sets up the world: ground plane (via CtrlAviary), gates, and the Crazyflie drone.
Uses gym-pybullet-drones (CtrlAviary + DSLPIDControl) for real Crazyflie physics.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from simulation.model_types import Gate, GateConfig, FieldConfig, Pose3D

from .gpd_drone import GPDDrone, GPDDroneConfig
from .gate_models import create_gate_body, highlight_gate, dim_gate, reset_gate_color


@dataclass
class RaceConfig:
    """Configuration for a drone race."""
    field_bounds_min: Tuple[float, float, float] = (-5.0, -15.0, 0.0)
    field_bounds_max: Tuple[float, float, float] = (50.0, 15.0, 15.0)
    gates: List[Gate] = None
    start_position: Tuple[float, float, float] = (0.0, 0.0, 1.5)
    start_yaw: float = 0.0
    timestep: float = 1.0 / 240.0
    gravity: float = -9.81

    # Iter 10 (Phase A L1): per-race optional overrides for the planner,
    # racing-line, and sequencer knobs that previously lived as magic
    # literals in Python. Empty dict (default) → components use their
    # baked-in defaults exactly, so existing configs without the matching
    # section are byte-identical to pre-iter-10 behavior. Populated dicts
    # are merged onto the corresponding dataclass defaults at construction.
    planner_overrides: Dict[str, float] = None
    racing_line_overrides: Dict[str, float] = None
    sequencer_overrides: Dict[str, float] = None

    def __post_init__(self):
        if self.gates is None:
            self.gates = []
        if self.planner_overrides is None:
            self.planner_overrides = {}
        if self.racing_line_overrides is None:
            self.racing_line_overrides = {}
        if self.sequencer_overrides is None:
            self.sequencer_overrides = {}


class DroneRaceEnv:
    """
    PyBullet-based drone racing environment using real Crazyflie CF2X physics.

    CtrlAviary creates the PyBullet world (ground plane, gravity, drone URDF).
    Gates are added to the same physics client after drone creation.
    """

    def __init__(
        self,
        race_config: Optional[RaceConfig] = None,
        drone_config: Optional[GPDDroneConfig] = None,
        gui: bool = False,
    ):
        self.race_config = race_config or RaceConfig()
        self.drone_config = drone_config or GPDDroneConfig()

        # GPDDrone creates the PyBullet world internally via CtrlAviary.
        self.drone = GPDDrone(
            start_position=self.race_config.start_position,
            start_yaw=self.race_config.start_yaw,
            config=self.drone_config,
            gui=gui,
        )

        # Expose the PyBullet client for gate/debug-line operations.
        self.client = self.drone.CLIENT

        # Create gate collision/visual bodies in the same physics world.
        # gate_id -> list of pybullet body IDs (4 segments per gate)
        self.gate_bodies: Dict[str, List[int]] = {}
        for gate in self.race_config.gates:
            body_ids = create_gate_body(self.client, gate)
            self.gate_bodies[gate.gate_id] = body_ids

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of control steps taken (1 step = race_config.timestep seconds)."""
        return self.drone.step_count

    def get_sim_time(self) -> float:
        return self.drone.get_sim_time()

    # ------------------------------------------------------------------
    # Gate visual management
    # ------------------------------------------------------------------

    def highlight_gate(self, gate_id: str):
        if gate_id in self.gate_bodies:
            highlight_gate(self.client, self.gate_bodies[gate_id])

    def dim_gate(self, gate_id: str):
        if gate_id in self.gate_bodies:
            dim_gate(self.client, self.gate_bodies[gate_id])

    def reset_gate_color(self, gate_id: str, gate: Gate):
        if gate_id in self.gate_bodies:
            reset_gate_color(self.client, self.gate_bodies[gate_id], gate)

    # ------------------------------------------------------------------
    # Collision queries
    # ------------------------------------------------------------------

    def gate_contact(self) -> Optional[str]:
        """Return the gate_id the drone is currently touching, else None.

        Walks every gate-segment body and asks PyBullet for contact points
        against the drone. Cheap (~O(n_gates)) and deterministic — bullet
        already maintains the contact manifold from the last step.
        """
        import pybullet as p
        drone_id = self.drone.body_id
        for gate_id, body_ids in self.gate_bodies.items():
            for bid in body_ids:
                contacts = p.getContactPoints(
                    bodyA=drone_id, bodyB=bid, physicsClientId=self.client
                )
                if contacts:
                    return gate_id
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset drone position without wiping the PyBullet world (preserves gates)."""
        self.drone.reset()
        for gate in self.race_config.gates:
            self.reset_gate_color(gate.gate_id, gate)

    def close(self):
        self.drone.close()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_config(config_path: str) -> RaceConfig:
        """Load a race configuration from a JSON file."""
        path = Path(config_path)
        with open(path) as f:
            data = json.load(f)

        field_data = data.get("field", {})
        bounds_min = tuple(field_data.get("bounds_min", [-5.0, -15.0, 0.0]))
        bounds_max = tuple(field_data.get("bounds_max", [50.0, 15.0, 15.0]))

        gate_defaults = data.get("gate_defaults", {})
        default_config = GateConfig(
            gate_type=gate_defaults.get("gate_type", "square"),
            interior_width_m=gate_defaults.get("interior_width_m", 1.0),
            interior_height_m=gate_defaults.get("interior_height_m", 1.0),
            border_width_m=gate_defaults.get("border_width_m", 0.15),
            depth_m=gate_defaults.get("depth_m", 0.08),
            color=gate_defaults.get("color", "red"),
        )

        gates = []
        for gd in data.get("gates", []):
            pose_data = gd.get("pose", {})
            pose = Pose3D(
                x=pose_data.get("x", 0.0),
                y=pose_data.get("y", 0.0),
                z=pose_data.get("z", 1.5),
                yaw=pose_data.get("yaw", 0.0),
                pitch=pose_data.get("pitch", 0.0),
                roll=pose_data.get("roll", 0.0),
            )

            gc = gd.get("config", {})
            config = GateConfig(
                gate_type=gc.get("gate_type", default_config.gate_type),
                interior_width_m=gc.get("interior_width_m", default_config.interior_width_m),
                interior_height_m=gc.get("interior_height_m", default_config.interior_height_m),
                border_width_m=gc.get("border_width_m", default_config.border_width_m),
                depth_m=gc.get("depth_m", default_config.depth_m),
                color=gc.get("color", default_config.color),
            )

            gate = Gate(
                gate_id=gd["id"],
                config=config,
                pose=pose,
                sequence_index=gd.get("sequence_index"),
            )
            gates.append(gate)

        start_data = data.get("start", {})
        start_pos = tuple(start_data.get("position", [0.0, 0.0, 1.5]))
        start_yaw = start_data.get("yaw", 0.0)

        # Iter 10 (Phase A L1): optional top-level ``planner``,
        # ``racing_line``, and ``sequencer`` sections carry per-race
        # overrides of knobs that used to be hardcoded in the planners
        # and the visual demo. Unknown keys are ignored so the loader
        # is forward-compatible with future additions.
        planner_data = data.get("planner", {})
        racing_line_data = data.get("racing_line", {})
        sequencer_data = data.get("sequencer", {})

        return RaceConfig(
            field_bounds_min=bounds_min,
            field_bounds_max=bounds_max,
            gates=gates,
            start_position=start_pos,
            start_yaw=start_yaw,
            timestep=data.get("timestep", 1.0 / 240.0),
            gravity=data.get("gravity", -9.81),
            planner_overrides=dict(planner_data),
            racing_line_overrides=dict(racing_line_data),
            sequencer_overrides=dict(sequencer_data),
        )
