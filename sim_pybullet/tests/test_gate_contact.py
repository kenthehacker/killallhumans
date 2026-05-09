"""
Tests for DroneRaceEnv.gate_contact().

Mocks pybullet so we don't need a real physics client. Verifies:
  - Returns None when no contact points exist.
  - Returns the gate_id whose body has a contact point.
  - Iterates every body in a gate's segment list (gates are 4-segment).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_pybullet(contact_table):
    """
    Install a fake pybullet module whose getContactPoints returns the
    contact list for (bodyA, bodyB) keyed lookups.
    """
    fake = types.ModuleType("pybullet")

    def get_contact_points(bodyA, bodyB, physicsClientId=0):
        return contact_table.get((bodyA, bodyB), [])

    fake.getContactPoints = get_contact_points
    sys.modules["pybullet"] = fake
    return fake


def _make_env_stub(drone_body_id: int, gate_bodies: dict, client_id: int = 0):
    """
    Build a DroneRaceEnv-shaped object that has just enough surface for
    gate_contact() to work, without importing the real env (which pulls
    in CtrlAviary and gym-pybullet-drones).
    """
    from sim_pybullet.env import DroneRaceEnv

    env = DroneRaceEnv.__new__(DroneRaceEnv)  # bypass __init__
    env.client = client_id
    env.gate_bodies = gate_bodies
    env.drone = MagicMock()
    env.drone.body_id = drone_body_id
    return env


def test_gate_contact_returns_none_when_no_contacts():
    _install_fake_pybullet({})
    env = _make_env_stub(
        drone_body_id=1,
        gate_bodies={"G1": [10, 11, 12, 13], "G2": [20, 21, 22, 23]},
    )
    assert env.gate_contact() is None


def test_gate_contact_returns_gate_id_on_strut_hit():
    # Drone (body 1) is touching the second segment of gate G2 (body 21).
    _install_fake_pybullet({(1, 21): [("contact",)]})
    env = _make_env_stub(
        drone_body_id=1,
        gate_bodies={"G1": [10, 11, 12, 13], "G2": [20, 21, 22, 23]},
    )
    assert env.gate_contact() == "G2"


def test_gate_contact_first_match_wins():
    # Touching both G1's first segment and G2's first segment. Iteration
    # order is dict-insertion-order (Python 3.7+), so G1 should win.
    _install_fake_pybullet({
        (1, 10): [("c1",)],
        (1, 20): [("c2",)],
    })
    env = _make_env_stub(
        drone_body_id=1,
        gate_bodies={"G1": [10, 11, 12, 13], "G2": [20, 21, 22, 23]},
    )
    assert env.gate_contact() == "G1"


def test_gate_contact_passes_client_id():
    """Must thread physicsClientId through so multi-client setups work."""
    seen = []

    fake = types.ModuleType("pybullet")

    def get_contact_points(bodyA, bodyB, physicsClientId):
        seen.append(physicsClientId)
        return []

    fake.getContactPoints = get_contact_points
    sys.modules["pybullet"] = fake

    env = _make_env_stub(
        drone_body_id=7,
        gate_bodies={"G1": [42]},
        client_id=99,
    )
    env.gate_contact()
    assert seen == [99]
