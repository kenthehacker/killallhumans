"""rl/ — gray-box telemetry-fit REPLICA of the DCGame sim + the fidelity gate.

This package is the FOUNDATION of the RL effort (see rl/README.md for the staged
plan). It is numpy/scipy-only by mandate (CLAUDE.md): NO torch, NO PyBullet here.

The replica does NOT model true rigid-body physics. We never command or observe
the drone's real dynamics — we send body-rate setpoints + thrust over MAVLink to
DCGame's closed-source inner autopilot. What is observable, and what RL actually
consumes, is the COMPOSITE map

    (attitude/body-rate setpoint + thrust, current state) -> next state

system-identified from telemetry (captures/rel_*.jsonl.gz) and replayed here.

Public surface:
    DCGameReplica, ReplicaParams, ReplicaState   (rl.dcgame_replica)
    attitude_to_body_rate                         (rl.dcgame_replica)
    fit_dynamics_from_captures                     (rl.fit_dynamics)
    validate_fidelity                              (rl.validate_fidelity)
"""

from rl.dcgame_replica import (  # noqa: F401
    DCGameReplica,
    ReplicaParams,
    ReplicaState,
    attitude_to_body_rate,
)

__all__ = [
    "DCGameReplica",
    "ReplicaParams",
    "ReplicaState",
    "attitude_to_body_rate",
]
