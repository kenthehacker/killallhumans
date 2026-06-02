"""Tests for competition.track_data — converting the sim's runtime track-info
packet (TrackData / TrackGate, NED) into the autonomy stack's GateSpec list
that RacePipeline.configure() consumes.
"""
import math

import pytest

from competition.adapter import Quaternion
from competition.aigp_messages import TrackData, TrackGate
from competition.track_data import track_data_to_gatespecs
from competition.aigp_geometry import AIGP_GATE_BORDER_M, AIGP_GATE_DEPTH_M
from gate_sequencing.sequencer import GateSpec


def _gate(gate_id, pos, quat=(1.0, 0.0, 0.0, 0.0), width=1.5, height=1.5):
    return TrackGate(
        gate_id=gate_id,
        position_ned=pos,
        orientation=Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3]),
        width=width,
        height=height,
    )


def test_empty_track_yields_no_gatespecs():
    assert track_data_to_gatespecs(TrackData(gates=[])) == []


def test_single_gate_maps_fields():
    td = TrackData(gates=[_gate(0, (8.0, 0.0, -1.5), width=1.5, height=1.5)])
    specs = track_data_to_gatespecs(td)
    assert len(specs) == 1
    s = specs[0]
    assert isinstance(s, GateSpec)
    assert s.gate_id == "0"          # GateSpec.gate_id is a string
    assert s.position == pytest.approx((8.0, 0.0, -1.5))
    assert s.sequence_index == 0
    assert s.interior_width == pytest.approx(1.5)
    assert s.interior_height == pytest.approx(1.5)
    # frame thickness / depth stay at AIGP spec defaults (not in track packet)
    assert s.border_width == pytest.approx(AIGP_GATE_BORDER_M)
    assert s.depth == pytest.approx(AIGP_GATE_DEPTH_M)


def test_orientation_quaternion_becomes_yaw():
    # A pure yaw rotation of +0.5 rad about the down axis.
    q = Quaternion.from_euler(0.0, 0.0, 0.5)
    td = TrackData(gates=[_gate(0, (1.0, 2.0, -3.0), quat=(q.w, q.x, q.y, q.z))])
    s = track_data_to_gatespecs(td)[0]
    assert s.yaw == pytest.approx(0.5, abs=1e-5)
    assert s.pitch == pytest.approx(0.0, abs=1e-5)
    assert s.roll == pytest.approx(0.0, abs=1e-5)


def test_gates_sorted_by_id_with_sequence_index():
    # Delivered out of order — output must be ordered by gate_id with
    # sequence_index assigned in that order (gates are passed in order).
    td = TrackData(gates=[_gate(2, (2, 0, 0)), _gate(0, (0, 0, 0)), _gate(1, (1, 0, 0))])
    specs = track_data_to_gatespecs(td)
    assert [s.gate_id for s in specs] == ["0", "1", "2"]
    assert [s.sequence_index for s in specs] == [0, 1, 2]
    assert specs[0].position == pytest.approx((0, 0, 0))
    assert specs[2].position == pytest.approx((2, 0, 0))
