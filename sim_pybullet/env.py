"""
PyBullet drone racing environment.

Sets up the world: ground plane (via CtrlAviary), gates, and the Crazyflie drone.
Uses gym-pybullet-drones (CtrlAviary + DSLPIDControl) for real Crazyflie physics.
"""

import json
import math
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

    # Iter-007 (Opus F3 BLOCKER): top-level max_velocity_mps field so the
    # PyBullet bench AND the synthetic bench read the same value from the
    # same JSON key. iter-005b's `getattr(race_config, "max_velocity_mps",
    # None)` was dead code because the dataclass had no such field —
    # caught by all 3 iter-006 reviewers.
    max_velocity_mps: Optional[float] = None

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
        """Number of GPD control steps taken (1 step = 1 / ctrl_freq seconds).

        ``race_config.timestep`` describes the historical physics timestep and
        is not the elapsed time of one :meth:`GPDDrone.step` call, which may
        execute multiple physics substeps.
        """
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

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key in race config: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"non-standard JSON numeric constant: {value}")

        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        if not isinstance(data, dict):
            raise TypeError("race config root must be a JSON object")

        def mapping(name, value):
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a JSON object")
            return value

        def finite(name, value, *, positive=False):
            if type(value) not in {int, float}:
                raise TypeError(f"{name} must be an exact JSON number")
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError(f"{name} must be finite")
            if positive and resolved <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
            return resolved

        def vector3(name, value):
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(f"{name} must contain exactly three numbers")
            return tuple(
                finite(f"{name}[{index}]", item)
                for index, item in enumerate(value)
            )

        def exact_string(name, value):
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty string")
            return value

        field_data = mapping("field", data.get("field", {}))
        bounds_min = vector3(
            "field.bounds_min", field_data.get("bounds_min", [-5.0, -15.0, 0.0])
        )
        bounds_max = vector3(
            "field.bounds_max", field_data.get("bounds_max", [50.0, 15.0, 15.0])
        )
        if any(lower >= upper for lower, upper in zip(bounds_min, bounds_max)):
            raise ValueError("field bounds_min must be strictly below bounds_max")

        gate_defaults = mapping("gate_defaults", data.get("gate_defaults", {}))
        default_config = GateConfig(
            gate_type=exact_string(
                "gate_defaults.gate_type", gate_defaults.get("gate_type", "square")
            ),
            interior_width_m=finite(
                "gate_defaults.interior_width_m",
                gate_defaults.get("interior_width_m", 1.0),
                positive=True,
            ),
            interior_height_m=finite(
                "gate_defaults.interior_height_m",
                gate_defaults.get("interior_height_m", 1.0),
                positive=True,
            ),
            border_width_m=finite(
                "gate_defaults.border_width_m",
                gate_defaults.get("border_width_m", 0.15),
                positive=True,
            ),
            depth_m=finite(
                "gate_defaults.depth_m",
                gate_defaults.get("depth_m", 0.08),
                positive=True,
            ),
            color=exact_string(
                "gate_defaults.color", gate_defaults.get("color", "red")
            ),
        )

        raw_gates = data.get("gates", [])
        if not isinstance(raw_gates, list):
            raise TypeError("gates must be a JSON list")
        gates = []
        gate_ids = set()
        sequence_indices = set()
        for gate_number, raw_gate in enumerate(raw_gates):
            gd = mapping(f"gates[{gate_number}]", raw_gate)
            gate_id = exact_string(f"gates[{gate_number}].id", gd.get("id"))
            if gate_id in gate_ids:
                raise ValueError(f"duplicate gate id: {gate_id}")
            gate_ids.add(gate_id)
            sequence_index = gd.get("sequence_index", gate_number)
            if type(sequence_index) is not int or sequence_index < 0:
                raise TypeError(
                    f"gates[{gate_number}].sequence_index must be a non-negative integer"
                )
            if sequence_index in sequence_indices:
                raise ValueError(f"duplicate gate sequence_index: {sequence_index}")
            sequence_indices.add(sequence_index)
            pose_data = mapping(f"gates[{gate_number}].pose", gd.get("pose", {}))
            pose = Pose3D(
                x=finite(f"gates[{gate_number}].pose.x", pose_data.get("x", 0.0)),
                y=finite(f"gates[{gate_number}].pose.y", pose_data.get("y", 0.0)),
                z=finite(f"gates[{gate_number}].pose.z", pose_data.get("z", 1.5)),
                yaw=finite(
                    f"gates[{gate_number}].pose.yaw", pose_data.get("yaw", 0.0)
                ),
                pitch=finite(
                    f"gates[{gate_number}].pose.pitch", pose_data.get("pitch", 0.0)
                ),
                roll=finite(
                    f"gates[{gate_number}].pose.roll", pose_data.get("roll", 0.0)
                ),
            )

            gc = mapping(f"gates[{gate_number}].config", gd.get("config", {}))
            config = GateConfig(
                gate_type=exact_string(
                    f"gates[{gate_number}].config.gate_type",
                    gc.get("gate_type", default_config.gate_type),
                ),
                interior_width_m=finite(
                    f"gates[{gate_number}].config.interior_width_m",
                    gc.get("interior_width_m", default_config.interior_width_m),
                    positive=True,
                ),
                interior_height_m=finite(
                    f"gates[{gate_number}].config.interior_height_m",
                    gc.get("interior_height_m", default_config.interior_height_m),
                    positive=True,
                ),
                border_width_m=finite(
                    f"gates[{gate_number}].config.border_width_m",
                    gc.get("border_width_m", default_config.border_width_m),
                    positive=True,
                ),
                depth_m=finite(
                    f"gates[{gate_number}].config.depth_m",
                    gc.get("depth_m", default_config.depth_m),
                    positive=True,
                ),
                color=exact_string(
                    f"gates[{gate_number}].config.color",
                    gc.get("color", default_config.color),
                ),
            )

            gate = Gate(
                gate_id=gate_id,
                config=config,
                pose=pose,
                sequence_index=sequence_index,
            )
            gates.append(gate)

        start_data = mapping("start", data.get("start", {}))
        start_pos = vector3(
            "start.position", start_data.get("position", [0.0, 0.0, 1.5])
        )
        start_yaw = finite("start.yaw", start_data.get("yaw", 0.0))

        # Iter 10 (Phase A L1): optional top-level ``planner``,
        # ``racing_line``, and ``sequencer`` sections carry per-race
        # overrides of knobs that used to be hardcoded in the planners
        # and the visual demo. Unknown keys are ignored so the loader
        # is forward-compatible with future additions.
        planner_data = mapping("planner", data.get("planner", {}))
        racing_line_data = mapping("racing_line", data.get("racing_line", {}))
        sequencer_data = mapping("sequencer", data.get("sequencer", {}))
        timestep = finite("timestep", data.get("timestep", 1.0 / 240.0), positive=True)
        gravity = finite("gravity", data.get("gravity", -9.81))
        if gravity >= 0.0:
            raise ValueError("gravity must be negative")
        max_velocity = data.get("max_velocity_mps")
        if max_velocity is not None:
            max_velocity = finite("max_velocity_mps", max_velocity, positive=True)

        return RaceConfig(
            field_bounds_min=bounds_min,
            field_bounds_max=bounds_max,
            gates=gates,
            start_position=start_pos,
            start_yaw=start_yaw,
            timestep=timestep,
            gravity=gravity,
            # Iter-007 Opus F3: load the top-level `max_velocity_mps` so
            # the PyBullet path picks up the same per-track value the
            # synthetic bench reads. Default None means "fall through to
            # auto-derive in the bench" (matches synthetic bench logic).
            max_velocity_mps=max_velocity,
            planner_overrides=dict(planner_data),
            racing_line_overrides=dict(racing_line_data),
            sequencer_overrides=dict(sequencer_data),
        )
