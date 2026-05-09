"""
PyBullet-based drone racing simulation with realistic physics.

Runs alongside the lightweight `simulation/` package — this one provides
closed-loop physics, camera rendering, and gate sequencing for testing
the full autonomy stack.
"""

from .env import DroneRaceEnv
from .gpd_drone import GPDDrone, GPDDroneConfig
from .gate_models import create_gate_body, highlight_gate, dim_gate
# GateSequencer collapsed into the platform-agnostic version. See
# gate_sequencing.sequencer + sim_pybullet/_gate_to_spec.py (P2-1).
from gate_sequencing.sequencer import GateSequencer
from ._gate_to_spec import to_spec as _gate_to_spec  # noqa: F401
