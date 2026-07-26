"""Replay exact near-plane target-boundary facts from visual-course attempt 3.

Source:

* run: ``20260726T030710Z-visual-course-3403f1e5``
* commit: ``e8fcd57090cdb18b8c4bd0f1bad9046aab524a9f``
* config: ``ca07ccf0b60840e77db43137e9c5e2f33449d23bb6f7933994695e62fbb93e11``
* trace: ``612e7d08b4496ccc50f0f414fb2d4efa755cd6c57f28edab60d0c128c05a7249``

The run retained tracker, graph, command-wire, IMU, and race facts but no JPEG
replay bundle. These tests replay the exact logged target/accepted-wire
boundary into the production lifecycle reducer. They are not detector, UDP,
IMU-estimator, vehicle-dynamics, or full-image replay.
"""

from __future__ import annotations

from dataclasses import replace
import math

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import CameraFrameToken, VisualTrackRole
from planning.vq2_course_lifecycle import (
    LatchedMeasurementMode,
    NearPlaneEvidence,
    NearPlaneWireSample,
    advance_near_plane_evidence,
    classify_latched_measurement,
)
from planning.vq2_gate_graph import DEFAULT_ROLLING_GATE_GRAPH_CONFIG


_STREAM_ID = "vq2-camera-udp-5600"
_TRACK_ID = "vq2-track-000001"
_CROSSING_MIN_LOG_SCALE = -0.80

# publication, observation ns, publication ns, wire start ns, wire return ns,
# x, y-down, vx, vy, apparent scale, log-scale rate, confidence, association,
# roll/pitch/yaw/thrust final bounded command.
_ACCEPTED_WIRE_ROWS = (
    (
        159,
        168_080_068_485_100,
        168_080_069_594_700,
        168_080_100_129_900,
        168_080_100_207_900,
        0.06562500000000004,
        -0.10555555555555551,
        0.09097910460810257,
        0.3763050272183972,
        0.5666781555207898,
        1.8603279243415711,
        0.9182944864629641,
        0.8932093941133568,
        -0.0006811912852831715,
        0.011320412028434406,
        0.0,
        0.23651185156400012,
    ),
    (
        160,
        168_080_103_254_300,
        168_080_104_149_400,
        168_080_131_898_200,
        168_080_131_965_500,
        0.05312500000000009,
        -0.0888888888888889,
        -0.1567918787958008,
        0.4329805634075414,
        0.6016398308881411,
        1.7841669019083968,
        0.9229418212339152,
        0.9070421818042729,
        -0.0005144568175131827,
        0.014542224085169761,
        0.0,
        0.23046552649573276,
    ),
    (
        161,
        168_080_132_237_600,
        168_080_133_114_800,
        168_080_162_676_200,
        168_080_162_744_100,
        0.04062499999999991,
        -0.07777777777777772,
        -0.3077619086617517,
        0.40569064304773983,
        0.6313909358621557,
        1.7187967895460923,
        0.9297946224749698,
        0.9210062619665919,
        -0.00043582329946673776,
        0.020101248165450815,
        0.0,
        0.23074630480773928,
    ),
)


def _token(publication: int) -> CameraFrameToken:
    return CameraFrameToken(
        generation=1,
        frame_id=1_385_068 + publication,
        publication_sequence=publication,
        stream_id=_STREAM_ID,
    )


def _sample(row) -> NearPlaneWireSample:
    (
        publication,
        observation_ns,
        publication_ns,
        wire_start_ns,
        wire_return_ns,
        x,
        y,
        vx,
        vy,
        scale,
        scale_rate,
        confidence,
        association,
        roll,
        pitch,
        yaw,
        thrust,
    ) = row
    token = _token(publication)
    return NearPlaneWireSample(
        gate_index=0,
        track_id=_TRACK_ID,
        camera_token=token,
        wire_camera_token=token,
        observation_monotonic_ns=observation_ns,
        publication_monotonic_ns=publication_ns,
        wire_start_monotonic_ns=wire_start_ns,
        wire_return_monotonic_ns=wire_return_ns,
        wire_race_gate_index=0,
        publication_pinned_through_transport_return=True,
        normalized_x=x,
        normalized_y_down=y,
        normalized_x_rate_s=vx,
        normalized_y_rate_down_s=vy,
        log_scale=math.log(scale),
        log_scale_rate_s=scale_rate,
        confidence=confidence,
        association_confidence=association,
        clipping=FrameEdge.NONE,
        center_censored=False,
        ambiguous=False,
        command_roll_rate=roll,
        command_pitch_rate=pitch,
        command_yaw_rate=yaw,
        command_thrust=thrust,
    )


def _advance(evidence, sample):
    config = DEFAULT_ROLLING_GATE_GRAPH_CONFIG
    return advance_near_plane_evidence(
        evidence,
        sample,
        required_corridor_frames=3,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        min_track_confidence=config.min_track_confidence,
        min_association_confidence=config.min_association_confidence,
    )


def _exact_latch():
    evidence = NearPlaneEvidence()
    latch = None
    for row in _ACCEPTED_WIRE_ROWS:
        evidence, latch = _advance(evidence, _sample(row))
    assert latch is not None
    return latch


def _classify_pub164(latch, **changes):
    config = DEFAULT_ROLLING_GATE_GRAPH_CONFIG
    facts = {
        "previous_camera_token": _token(162),
        "camera_token": _token(164),
        "current_gate_index": 0,
        "current_track_id": _TRACK_ID,
        "track_latest_camera_token": _token(164),
        "track_role": VisualTrackRole.CURRENT,
        "track_authoritative_gate_index": 0,
        "visible": True,
        "missed_frame_count": 0,
        "ambiguous": False,
        "clipping": FrameEdge.TOP | FrameEdge.BOTTOM,
        "center_censored": True,
        "normalized_x": 0.0031250000000000444,
        "normalized_y_down": 0.0,
        "normalized_x_rate_s": -0.19808378450363373,
        "normalized_y_rate_down_s": 0.7924858044955384,
        "apparent_scale": 0.8100925873009825,
        "confidence": 0.9498046113839211,
        "association_confidence": 0.8003389499432206,
        "min_track_confidence": config.min_track_confidence,
        "min_association_confidence": (
            config.min_association_confidence
        ),
    }
    facts.update(changes)
    return classify_latched_measurement(latch, **facts)


def test_exact_attempt3_wire_suffix_latches_and_vertical_censor_coasts():
    latch = _exact_latch()

    assert [
        sample.camera_token.publication_sequence
        for sample in latch.evidence.samples
    ] == [159, 160, 161]
    assert latch.anchor_camera_token == _token(161)
    assert _classify_pub164(latch) is LatchedMeasurementMode.COAST


def test_attempt3_nonexpansion_and_off_center_mutations_stay_fail_closed():
    first, second, third = map(_sample, _ACCEPTED_WIRE_ROWS)
    second = replace(second, log_scale=first.log_scale - 0.01)

    evidence = NearPlaneEvidence()
    latch = None
    for sample in (first, second, third):
        evidence, latch = _advance(evidence, sample)
    assert latch is None
    assert len(evidence.samples) == 2

    latch = _exact_latch()
    assert (
        _classify_pub164(
            latch,
            clipping=FrameEdge.NONE,
            center_censored=False,
            normalized_x=0.50,
        )
        is LatchedMeasurementMode.UNSAFE
    )
    assert (
        _classify_pub164(latch, ambiguous=True)
        is LatchedMeasurementMode.UNSAFE
    )
