"""
PyBullet-based drone racing simulation with realistic physics.

Runs alongside the lightweight `simulation/` package — this one provides
closed-loop physics, camera rendering, and gate sequencing for testing
the full autonomy stack.
"""

from .env import DroneRaceEnv
from .gpd_drone import GPDDrone, GPDDroneConfig
from .gate_models import create_gate_body, highlight_gate, dim_gate
from .sequencer import GateSequencer
