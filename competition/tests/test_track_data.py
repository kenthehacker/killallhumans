"""Tests for competition.track_data — converting the sim's runtime track-info
packet (TrackData / TrackGate, NED) into the autonomy stack's GateSpec list
that RacePipeline.configure() consumes.
"""
import math

import pytest

from competition.adapter import Quaternion
from competition.aigp_messages import (
    TrackData,
    TrackGate,
    encode_track_data,
    parse_track_data,
)
import competition.track_data as track_data_module
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


def _gate_normal(spec: GateSpec):
    cy, sy = math.cos(spec.yaw), math.sin(spec.yaw)
    cp, sp = math.cos(spec.pitch), math.sin(spec.pitch)
    return (cy * cp, sy * cp, -sp)


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


def test_vq1_capture_end_to_end():
    # VQ1 first-contact course: q=[w,x,y,z] rotates gate-local +Y to world -X.
    positions = [
        (-23.3, -0.4, -0.03),
        (-46.9, -2.5, 5.07),
        (-74.6, 1.2, 13.67),
        (-111.5, -5.1, 24.57),
        (-135.5, -0.8, 25.36),
        (-159.2, -4.4, 25.97),
    ]
    shuffled = [2, 0, 5, 1, 4, 3]
    gates = [
        _gate(i, positions[i], quat=(0.7071, 0.0, 0.0, 0.7071), width=2.72, height=2.72)
        for i in shuffled
    ]
    parsed = parse_track_data(encode_track_data(gates))

    specs = track_data_to_gatespecs(parsed)

    assert track_data_module.GATE_LOCAL_THROUGH_AXIS == pytest.approx((0.0, 1.0, 0.0))
    assert [s.gate_id for s in specs] == ["0", "1", "2", "3", "4", "5"]
    assert [s.sequence_index for s in specs] == [0, 1, 2, 3, 4, 5]
    for spec, pos in zip(specs, positions):
        assert spec.position == pytest.approx(pos, abs=1e-4)
        assert _gate_normal(spec) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-4)
        assert spec.yaw == pytest.approx(math.pi, abs=1e-4)
        assert spec.pitch == pytest.approx(0.0, abs=1e-5)
        assert spec.roll == pytest.approx(0.0, abs=1e-5)
        assert spec.interior_width == pytest.approx(1.52, abs=1e-5)
        assert spec.interior_height == pytest.approx(1.52, abs=1e-5)
        assert spec.border_width == pytest.approx(AIGP_GATE_BORDER_M)
        assert spec.depth == pytest.approx(AIGP_GATE_DEPTH_M)
        assert spec.outer_width == pytest.approx(2.72, abs=1e-5)


def test_normal_flipped_to_match_course_direction():
    td = TrackData(gates=[
        _gate(0, (0.0, 0.0, 0.0), quat=(0.7071, 0.0, 0.0, -0.7071), width=2.72, height=2.72),
        _gate(1, (-10.0, 0.0, 0.0), quat=(0.7071, 0.0, 0.0, -0.7071), width=2.72, height=2.72),
    ])
    specs = track_data_to_gatespecs(td)
    assert _gate_normal(specs[0]) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-4)
    assert specs[0].yaw == pytest.approx(math.pi, abs=1e-4)
    assert specs[0].pitch == pytest.approx(0.0, abs=1e-5)
    assert specs[0].roll == pytest.approx(0.0, abs=1e-5)


def test_pitch_uses_consumer_negative_sin_convention():
    q = Quaternion.from_euler(math.pi / 2, 0.0, 0.0)
    td = TrackData(gates=[_gate(
        0,
        (0.0, 0.0, 0.0),
        quat=(q.w, q.x, q.y, q.z),
        width=2.72,
        height=2.72,
    )])
    spec = track_data_to_gatespecs(td)[0]
    assert _gate_normal(spec) == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert spec.pitch == pytest.approx(-math.pi / 2, abs=1e-6)


def test_packet_dimension_falls_back_when_already_interior():
    td = TrackData(gates=[_gate(0, (0.0, 0.0, 0.0), width=1.5, height=1.5)])
    spec = track_data_to_gatespecs(td)[0]
    assert spec.interior_width == pytest.approx(1.5)
    assert spec.interior_height == pytest.approx(1.5)


def test_duplicate_gate_ids_raise_value_error():
    td = TrackData(gates=[
        _gate(0, (0.0, 0.0, 0.0)),
        _gate(0, (1.0, 0.0, 0.0)),
    ])
    with pytest.raises(ValueError, match="duplicate gate_id"):
        track_data_to_gatespecs(td)


def test_zero_orientation_quaternion_raises_value_error():
    td = TrackData(gates=[_gate(0, (0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="zero-norm"):
        track_data_to_gatespecs(td)


def test_gates_sorted_by_id_with_sequence_index():
    # Delivered out of order — output must be ordered by gate_id with
    # sequence_index assigned in that order (gates are passed in order).
    td = TrackData(gates=[_gate(2, (2, 0, 0)), _gate(0, (0, 0, 0)), _gate(1, (1, 0, 0))])
    specs = track_data_to_gatespecs(td)
    assert [s.gate_id for s in specs] == ["0", "1", "2"]
    assert [s.sequence_index for s in specs] == [0, 1, 2]
    assert specs[0].position == pytest.approx((0, 0, 0))
    assert specs[2].position == pytest.approx((2, 0, 0))
