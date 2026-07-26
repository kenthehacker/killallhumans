"""Regress the course lifecycle against compact build-3385 live facts.

This is an exact logged tracker/graph/IMU/race/wire-state replay.  It is not
JPEG replay, detector replay, or full receiver replay: the source manifests
explicitly recorded ``replay_bundle: null``.  The tracked constants below are
the smallest state excerpts needed to exercise the pure lifecycle decisions.
"""

from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import CameraFrameToken, VisualTrackRole
from planning.vq2_course_lifecycle import (
    CourseLifecycle,
    LatchedMeasurementMode,
    NEAR_PLANE_LATCH_BASIS,
    NearPlaneEvidence,
    NearPlaneLatch,
    NearPlaneWireSample,
    advance_near_plane_evidence,
    classify_latched_measurement,
)
from planning.vq2_gate_graph import DEFAULT_ROLLING_GATE_GRAPH_CONFIG
from planning.vq2_visual_servo import (
    PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S,
    PREPASS_CURRENT_MAX_ABS_X_NORM,
    PREPASS_CURRENT_MAX_ABS_Y_NORM,
    PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S,
    PREPASS_CURRENT_PROJECTION_HORIZON_S,
    VisualServoTuning,
)
from scripts.aigp_vq2_visual_course_stage import (
    VisualCourseStageLimits,
    _current_snapshot_ready,
)


_SOURCES = (
    {
        "run_id": "20260725T202342Z-visual-course-2e71fae8",
        "commit": "4c67777fbe7b80ab4b0aead9bc6c108b4b6ca953",
        "configuration_sha256": (
            "ca07ccf0b60840e77db43137e9c5e2f33449d23bb6f7933994695e62fbb93e11"
        ),
        "trace_sha256": (
            "6efb867b047144878ed9096eaed0f639b8e8b44e6f81114a812d9984a726b16e"
        ),
        "result_sha256": (
            "720d07a6b18be79e73f90a0706a3075f5f037d4efc90efe3d7e5bf15d2737149"
        ),
        "replay_bundle": None,
    },
    {
        "run_id": "20260725T221536Z-visual-course-508a76b3",
        "commit": "dec83dbb535d1c5bf94ee30f472da9307d900229",
        "configuration_sha256": (
            "ca07ccf0b60840e77db43137e9c5e2f33449d23bb6f7933994695e62fbb93e11"
        ),
        "trace_sha256": (
            "fb4a5e71312f6d7fc6c0164e02d10fa424fffdbcb855812b0ef83c81b3845044"
        ),
        "result_sha256": (
            "a1cf685dfb7358dbaac4583d66241eb588f47c43a8f22ff0c5f9a52b7ddafe90"
        ),
        "replay_bundle": None,
    },
    {
        "run_id": "20260726T035026Z-visual-course-5630e810",
        "commit": "4e12c8e874b2dc3bf3f938345144400cacb716ff",
        "configuration_sha256": (
            "ca07ccf0b60840e77db43137e9c5e2f33449d23bb6f7933994695e62fbb93e11"
        ),
        "trace_sha256": (
            "6202ea6765b380b06d6c072821cdfc817a714d820833b623301c00ec696fdb04"
        ),
        "result_sha256": (
            "ea7e679ecad8ee6c1b077595757c87fe13e133ef3b2e893008c3960485cad005"
        ),
        "replay_bundle": None,
    },
    {
        "run_id": "20260726T042145Z-visual-course-93d97b28",
        "commit": "cab574e138f552d20d5cca80b9dcc07c02d46790",
        "configuration_sha256": (
            "ca07ccf0b60840e77db43137e9c5e2f33449d23bb6f7933994695e62fbb93e11"
        ),
        "trace_sha256": (
            "4013133b12496e813e650ff4a6c1d0fa02a784b8df7d329cea592ac63b5eddec"
        ),
        "result_sha256": (
            "67aee8be848b7b1b0db85a4453a8453da1793f171101506e5a494c250e1b4d86"
        ),
        "replay_bundle": None,
    },
)
_CAMERA_STREAM = "vq2-camera-udp-5600"
_CURRENT_TRACK = "vq2-track-000001"
_PROMOTED_TRACK = "vq2-track-000002"
_CREDITED_RACE_RECEIVED_NS = 143_875_626_649_400
_TUNING = VisualServoTuning()
_LIMITS = VisualCourseStageLimits()
_GRAPH_CONFIG = DEFAULT_ROLLING_GATE_GRAPH_CONFIG
_REDUCER_KWARGS = {
    "required_corridor_frames": _TUNING.required_corridor_frames,
    "crossing_min_log_scale": _LIMITS.crossing_arm_min_log_scale,
    "min_track_confidence": _GRAPH_CONFIG.min_track_confidence,
    "min_association_confidence": (
        _GRAPH_CONFIG.min_association_confidence
    ),
}
_CLASSIFIER_CONFIDENCE_KWARGS = {
    "min_track_confidence": _GRAPH_CONFIG.min_track_confidence,
    "min_association_confidence": (
        _GRAPH_CONFIG.min_association_confidence
    ),
}

# The latest run reached its clean near-plane suffix with only two commands
# classified as "advance".  These three accepted wire publications are exact
# graph/wire/tick facts from publications 162--164.
_LATEST_ADVANCE_COMMAND_COUNT = 2
_LATEST_NEAR_PLANE_ROWS = (
    {
        "sequence": 162,
        "frame_id": 860514,
        "observation_ns": 150_589_378_772_800,
        "publication_ns": 150_589_379_693_500,
        "wire_start_ns": 150_589_405_759_200,
        "wire_return_ns": 150_589_405_827_100,
        "x": 0.0625,
        "y": -0.03888888888888886,
        "x_rate": -0.0757491799205273,
        "y_rate": 0.406498395364429,
        "scale": 0.5981452814975454,
        "scale_rate": 1.8659175074399745,
        "confidence": 0.9227527278675461,
        "association": 0.8826683444770271,
        "command": (
            -0.001138285769011149,
            0.015276810429744868,
            0.0,
            0.23019011539728196,
        ),
    },
    {
        "sequence": 163,
        "frame_id": 860515,
        "observation_ns": 150_589_404_970_500,
        "publication_ns": 150_589_405_998_600,
        "wire_start_ns": 150_589_436_024_100,
        "wire_return_ns": 150_589_436_069_600,
        "x": 0.046875,
        "y": -0.033333333333333326,
        "x_rate": -0.3621216530787741,
        "y_rate": 0.2995587746658279,
        "scale": 0.630241906819208,
        "scale_rate": 1.9370313522491318,
        "confidence": 0.928706200252549,
        "association": 0.9125382607877941,
        "command": (
            -0.0005170745309682602,
            0.019328224138863204,
            -0.0013882421422429046,
            0.23716605344202168,
        ),
    },
    {
        "sequence": 164,
        "frame_id": 860516,
        "observation_ns": 150_589_438_596_000,
        "publication_ns": 150_589_439_699_100,
        "wire_start_ns": 150_589_468_839_700,
        "wire_return_ns": 150_589_468_873_800,
        "x": 0.028124999999999956,
        "y": -0.01666666666666672,
        "x_rate": -0.46964163329973285,
        "y_rate": 0.4074120169678739,
        "scale": 0.6705615391429617,
        "scale_rate": 1.8859676905184997,
        "confidence": 0.9377144002831386,
        "association": 0.9121601209548152,
        "command": (
            -9.946973190635446e-05,
            0.020538953607333738,
            0.007999957165490665,
            0.2281495237006383,
        ),
    },
)

# Exact accepted-wire publications 159--164 and the following tracker-only
# bottom-censored publication 165 from:
# C:\Users\John\aigp-evidence\fast-flight-cycles\
# 20260726T035026Z-visual-course-5630e810\session.jsonl.gz
#
# This is a tracker/graph/wire boundary replay.  The source manifest has no
# replay bundle, so these facts do not claim JPEG, detector, UDP, or complete
# receiver replay.
_ATTEMPT4_NEAR_PLANE_ROWS = (
    {
        "sequence": 159,
        "frame_id": 1_463_132,
        "observation_ns": 170_676_898_922_800,
        "publication_ns": 170_676_899_987_000,
        "wire_start_ns": 170_676_920_939_600,
        "wire_return_ns": 170_676_921_004_300,
        "x": 0.07499999999999996,
        "y": -0.033333333333333326,
        "x_rate": 1.3670281359508214,
        "y_rate": 0.34718036440944966,
        "scale": 0.5295192394993783,
        "scale_rate": 3.4055413450667666,
        "confidence": 0.9367381488700184,
        "association": 0.761315303114898,
        "command": (
            -0.0006574963399372036,
            0.03594935938874102,
            0.0,
            0.23374933587420715,
        ),
    },
    {
        "sequence": 160,
        "frame_id": 1_463_133,
        "observation_ns": 170_676_935_927_800,
        "publication_ns": 170_676_936_994_000,
        "wire_start_ns": 170_676_952_630_100,
        "wire_return_ns": 170_676_952_710_900,
        "x": 0.06875000000000009,
        "y": -0.011111111111111072,
        "x_rate": 0.5222698088606171,
        "y_rate": 0.48651686111226855,
        "scale": 0.55263730068745,
        "scale_rate": 2.167619291582423,
        "confidence": 0.9216282182441906,
        "association": 0.8487041451779588,
        "command": (
            -0.000592990098805232,
            0.027438026370037723,
            0.0,
            0.22085971615220473,
        ),
    },
    {
        "sequence": 161,
        "frame_id": 1_463_134,
        "observation_ns": 170_676_968_534_900,
        "publication_ns": 170_676_969_550_000,
        "wire_start_ns": 170_676_984_170_600,
        "wire_return_ns": 170_676_984_254_000,
        "x": 0.05624999999999991,
        "y": 0.005555555555555536,
        "x_rate": 0.024177763371304534,
        "y_rate": 0.5000574549884801,
        "scale": 0.5825366583600841,
        "scale_rate": 1.8641803072048202,
        "confidence": 0.9214867304679504,
        "association": 0.8929918451264776,
        "command": (
            -0.0006210429609931206,
            0.01710802157008043,
            0.0,
            0.21624549026662657,
        ),
    },
    {
        "sequence": 162,
        "frame_id": 1_463_135,
        "observation_ns": 170_677_003_017_700,
        "publication_ns": 170_677_004_032_300,
        "wire_start_ns": 170_677_016_533_400,
        "wire_return_ns": 170_677_016_583_300,
        "x": 0.04062499999999991,
        "y": 0.02777777777777768,
        "x_rate": -0.23833845742077187,
        "y_rate": 0.5794698738564363,
        "scale": 0.6186202164674399,
        "scale_rate": 1.7974658079441244,
        "confidence": 0.9269723994970945,
        "association": 0.9057259744103434,
        "command": (
            -0.0005933535075613167,
            0.02581454535164186,
            0.0,
            0.21,
        ),
    },
    {
        "sequence": 163,
        "frame_id": 1_463_136,
        "observation_ns": 170_677_031_177_700,
        "publication_ns": 170_677_032_117_900,
        "wire_start_ns": 170_677_048_192_500,
        "wire_return_ns": 170_677_048_268_300,
        "x": 0.03125,
        "y": 0.050000000000000044,
        "x_rate": -0.29035777458934564,
        "y_rate": 0.694789221013177,
        "scale": 0.6492784456610277,
        "scale_rate": 1.7535885583754687,
        "confidence": 0.9333213253567314,
        "association": 0.9228193169681045,
        "command": (
            -0.000523756693300895,
            0.0225629183223547,
            0.0007875221106270983,
            0.21,
        ),
    },
    {
        "sequence": 164,
        "frame_id": 1_463_137,
        "observation_ns": 170_677_065_321_400,
        "publication_ns": 170_677_066_354_700,
        "wire_start_ns": 170_677_079_012_900,
        "wire_return_ns": 170_677_079_067_500,
        "x": 0.009374999999999911,
        "y": 0.061111111111111116,
        "x_rate": -0.48303200697964366,
        "y_rate": 0.49163724896802374,
        "scale": 0.6970921746799343,
        "scale_rate": 1.9337122824651742,
        "confidence": 0.9439139512492389,
        "association": 0.8979388920100646,
        "command": (
            -0.000521126996433585,
            0.01727273064623086,
            0.014093620244287558,
            0.21,
        ),
    },
)
_ATTEMPT4_BOTTOM_CENSOR = {
    "sequence": 165,
    "frame_id": 1_463_138,
    "x": -0.0031250000000000444,
    "y": 0.06666666666666665,
    "x_rate": -0.41389685600878345,
    "y_rate": 0.3085845188658079,
    "scale": 0.7560864148142504,
    "confidence": 0.9471791885099187,
    "association": 0.827260665014193,
}

# The credited comparison latched at publication 158.  Publications 165, 166,
# 171, and 172 then supply the exact BOTTOM -> TOP|BOTTOM -> full-frame -> loss
# evolution.  The gaps are real camera-publication replacement, not invented
# adjacent tokens.
_CREDITED_NEAR_PLANE_ROWS = (
    {
        "sequence": 156,
        "frame_id": 659078,
        "observation_ns": 143_874_842_790_600,
        "publication_ns": 143_874_843_950_900,
        "wire_start_ns": 143_874_870_779_600,
        "wire_return_ns": 143_874_870_826_500,
        "x": -0.009375000000000022,
        "y": -0.050000000000000044,
        "x_rate": 0.12262423164115838,
        "y_rate": 0.23078629418568575,
        "scale": 0.4092676385936225,
        "scale_rate": 1.1144505987291886,
        "confidence": 0.9820524393803381,
        "association": 0.9273506180084061,
        "command": (
            -0.0008138930404383224,
            0.039583051863755034,
            0.0,
            0.2555596088694337,
        ),
    },
    {
        "sequence": 157,
        "frame_id": 659079,
        "observation_ns": 143_874_878_240_600,
        "publication_ns": 143_874_879_360_500,
        "wire_start_ns": 143_874_902_593_600,
        "wire_return_ns": 143_874_902_677_900,
        "x": -0.009375000000000022,
        "y": -0.050000000000000044,
        "x_rate": 0.05518090423852126,
        "y_rate": 0.10385383238355858,
        "scale": 0.4310688025130095,
        "scale_rate": 1.3066965144902496,
        "confidence": 0.9820235977211522,
        "association": 0.917866701002499,
        "command": (
            -0.001111300660357516,
            0.0415224200402323,
            0.0,
            0.2637637457651319,
        ),
    },
    {
        "sequence": 158,
        "frame_id": 659080,
        "observation_ns": 143_874_905_646_700,
        "publication_ns": 143_874_906_653_500,
        "wire_start_ns": 143_874_933_521_300,
        "wire_return_ns": 143_874_933_567_400,
        "x": -0.009375000000000022,
        "y": -0.0444444444444444,
        "x_rate": 0.024831406907334565,
        "y_rate": 0.15822602951951487,
        "scale": 0.4539324701979359,
        "scale_rate": 1.6251694670673007,
        "confidence": 0.9809106189745185,
        "association": 0.923499758396885,
        "command": (
            -0.0005943738190254716,
            0.03918836610569788,
            0.0,
            0.25971237467284425,
        ),
    },
)


def _camera_token(sequence: int, frame_id: int) -> CameraFrameToken:
    return CameraFrameToken(
        generation=1,
        frame_id=frame_id,
        publication_sequence=sequence,
        stream_id=_CAMERA_STREAM,
    )


def _wire_sample(
    row: dict[str, object],
    *,
    gate_index: int = 0,
    track_id: str = _CURRENT_TRACK,
) -> NearPlaneWireSample:
    token = _camera_token(
        int(row["sequence"]),
        int(row["frame_id"]),
    )
    command = row["command"]
    assert isinstance(command, tuple)
    return NearPlaneWireSample(
        gate_index=gate_index,
        track_id=track_id,
        camera_token=token,
        wire_camera_token=token,
        observation_monotonic_ns=int(row["observation_ns"]),
        publication_monotonic_ns=int(row["publication_ns"]),
        wire_start_monotonic_ns=int(row["wire_start_ns"]),
        wire_return_monotonic_ns=int(row["wire_return_ns"]),
        wire_race_gate_index=gate_index,
        publication_pinned_through_transport_return=True,
        normalized_x=float(row["x"]),
        normalized_y_down=float(row["y"]),
        normalized_x_rate_s=float(row["x_rate"]),
        normalized_y_rate_down_s=float(row["y_rate"]),
        log_scale=math.log(float(row["scale"])),
        log_scale_rate_s=float(row["scale_rate"]),
        confidence=float(row["confidence"]),
        association_confidence=float(row["association"]),
        clipping=FrameEdge.NONE,
        center_censored=False,
        ambiguous=False,
        command_roll_rate=float(command[0]),
        command_pitch_rate=float(command[1]),
        command_yaw_rate=float(command[2]),
        command_thrust=float(command[3]),
    )


def _latch(
    rows: tuple[dict[str, object], ...],
    *,
    gate_index: int = 0,
    track_id: str = _CURRENT_TRACK,
) -> NearPlaneLatch:
    evidence = NearPlaneEvidence()
    latch = None
    for row in rows:
        evidence, latch = advance_near_plane_evidence(
            evidence,
            _wire_sample(
                row,
                gate_index=gate_index,
                track_id=track_id,
            ),
            **_REDUCER_KWARGS,
        )
    assert latch is not None
    return latch


def _classify(
    latch: NearPlaneLatch,
    *,
    previous_sequence: int,
    previous_frame_id: int,
    sequence: int,
    frame_id: int,
    clipping: FrameEdge,
    visible: bool = True,
    missed_frame_count: int = 0,
    ambiguous: bool = False,
    x: float | None = 0.0,
    y: float | None = 0.0,
    x_rate: float | None = 0.0,
    y_rate: float | None = 0.0,
    scale: float | None = 1.0,
    confidence: float | None = 0.9,
    association: float | None = 0.9,
    track_latest_sequence: int | None = None,
    track_latest_frame_id: int | None = None,
) -> LatchedMeasurementMode:
    camera_token = _camera_token(sequence, frame_id)
    if track_latest_sequence is None:
        track_latest_sequence = sequence
    if track_latest_frame_id is None:
        track_latest_frame_id = frame_id
    track_latest_token = _camera_token(
        track_latest_sequence,
        track_latest_frame_id,
    )
    return classify_latched_measurement(
        latch,
        previous_camera_token=_camera_token(
            previous_sequence,
            previous_frame_id,
        ),
        camera_token=camera_token,
        current_gate_index=latch.gate_index,
        current_track_id=latch.track_id,
        track_latest_camera_token=track_latest_token,
        track_role=VisualTrackRole.CURRENT,
        track_authoritative_gate_index=latch.gate_index,
        visible=visible,
        missed_frame_count=missed_frame_count,
        ambiguous=ambiguous,
        clipping=clipping,
        center_censored=clipping != FrameEdge.NONE,
        normalized_x=x,
        normalized_y_down=y,
        normalized_x_rate_s=x_rate,
        normalized_y_rate_down_s=y_rate,
        apparent_scale=scale,
        confidence=confidence,
        association_confidence=association,
        **_CLASSIFIER_CONFIDENCE_KWARGS,
    )


def _promoted_snapshot(
    *,
    sequence: int,
    frame_id: int,
    x: float,
    y: float,
    scale: float,
    confidence: float,
    association: float,
    gate_index: int = 1,
    track_id: str = _PROMOTED_TRACK,
    role: VisualTrackRole = VisualTrackRole.CURRENT,
    authoritative_gate_index: int | None = None,
    visible: bool = True,
    missed_frame_count: int = 0,
    ambiguous: bool = False,
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
    latest_token: CameraFrameToken | None = None,
    observation_ns: int | None = None,
    publication_ns: int | None = None,
):
    token = _camera_token(sequence, frame_id)
    if authoritative_gate_index is None:
        authoritative_gate_index = gate_index
    history = ()
    if observation_ns is not None or publication_ns is not None:
        assert observation_ns is not None
        assert publication_ns is not None
        history = (
            SimpleNamespace(
                token=token,
                observation_monotonic_ns=observation_ns,
                publication_monotonic_ns=publication_ns,
            ),
        )
    track = SimpleNamespace(
        track_id=track_id,
        latest_token=token if latest_token is None else latest_token,
        role=role,
        authoritative_gate_index=authoritative_gate_index,
        visible=visible,
        missed_frame_count=missed_frame_count,
        ambiguous=ambiguous,
        clipping=clipping,
        center_censored=center_censored,
        center_norm=(x, y),
        apparent_scale=scale,
        confidence=confidence,
        association_confidence=association,
        history=history,
    )
    return SimpleNamespace(
        latest_camera_token=token,
        current_gate_index=gate_index,
        current_track_id=track_id,
        current_track=track,
        authority_usable=True,
        race_finished=False,
    )


def test_sources_are_compact_logged_state_not_full_image_replay():
    assert {source["replay_bundle"] for source in _SOURCES} == {None}
    assert all(len(str(source["trace_sha256"])) == 64 for source in _SOURCES)
    assert all(len(str(source["result_sha256"])) == 64 for source in _SOURCES)


def test_latest_run_latches_before_censor_despite_two_advance_commands():
    assert _LATEST_ADVANCE_COMMAND_COUNT == 2
    assert _LATEST_ADVANCE_COMMAND_COUNT < _TUNING.required_corridor_frames

    latch = _latch(_LATEST_NEAR_PLANE_ROWS)

    assert latch.lifecycle is CourseLifecycle.NEAR_PLANE_LATCHED
    assert latch.basis == NEAR_PLANE_LATCH_BASIS
    assert latch.anchor_camera_token.publication_sequence == 164
    assert [sample.camera_token.publication_sequence for sample in (
        latch.evidence.samples
    )] == [162, 163, 164]
    assert latch.accepted_command == pytest.approx(
        _LATEST_NEAR_PLANE_ROWS[-1]["command"]
    )


def test_attempt4_contour_union_frames_cannot_contribute() -> None:
    publication_159 = _wire_sample(_ATTEMPT4_NEAR_PLANE_ROWS[0])
    publication_160 = _wire_sample(_ATTEMPT4_NEAR_PLANE_ROWS[1])

    assert abs(
        publication_159.normalized_x
        + publication_159.normalized_x_rate_s
        * PREPASS_CURRENT_PROJECTION_HORIZON_S
    ) > PREPASS_CURRENT_MAX_ABS_X_NORM
    assert (
        publication_159.log_scale_rate_s
        > PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
    )
    assert (
        publication_160.log_scale_rate_s
        > PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
    )

    evidence = NearPlaneEvidence()
    for sample in (publication_159, publication_160):
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            **_REDUCER_KWARGS,
        )
        assert evidence.samples == ()
        assert latch is None


def test_attempt4_projected_corridor_latches_before_bottom_censor() -> None:
    samples = tuple(
        _wire_sample(row)
        for row in _ATTEMPT4_NEAR_PLANE_ROWS[2:5]
    )
    terminal = samples[-1]

    assert (
        abs(terminal.normalized_y_rate_down_s)
        > PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S
    )
    for sample in samples:
        assert abs(
            sample.normalized_x
            + sample.normalized_x_rate_s
            * PREPASS_CURRENT_PROJECTION_HORIZON_S
        ) <= PREPASS_CURRENT_MAX_ABS_X_NORM
        assert abs(
            sample.normalized_y_down
            + sample.normalized_y_rate_down_s
            * PREPASS_CURRENT_PROJECTION_HORIZON_S
        ) <= PREPASS_CURRENT_MAX_ABS_Y_NORM

    evidence = NearPlaneEvidence()
    latch = None
    for sample in samples:
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            **_REDUCER_KWARGS,
        )

    assert latch is not None
    assert [
        sample.camera_token.publication_sequence
        for sample in latch.evidence.samples
    ] == [161, 162, 163]
    assert latch.anchor_camera_token == _camera_token(163, 1_463_136)

    last_clean = _ATTEMPT4_NEAR_PLANE_ROWS[-1]
    censored = _ATTEMPT4_BOTTOM_CENSOR
    mode = _classify(
        latch,
        previous_sequence=int(last_clean["sequence"]),
        previous_frame_id=int(last_clean["frame_id"]),
        sequence=int(censored["sequence"]),
        frame_id=int(censored["frame_id"]),
        clipping=FrameEdge.BOTTOM,
        x=float(censored["x"]),
        y=float(censored["y"]),
        x_rate=float(censored["x_rate"]),
        y_rate=float(censored["y_rate"]),
        scale=float(censored["scale"]),
        confidence=float(censored["confidence"]),
        association=float(censored["association"]),
    )

    assert mode is LatchedMeasurementMode.COAST


def test_credited_censor_fragment_and_loss_are_one_generic_measurement_mode():
    latch = _latch(_CREDITED_NEAR_PLANE_ROWS)

    bottom = _classify(
        latch,
        previous_sequence=158,
        previous_frame_id=659080,
        sequence=165,
        frame_id=659087,
        clipping=FrameEdge.BOTTOM,
        x=0.0,
        y=0.022222222222222143,
        x_rate=-0.31476596612987606,
        y_rate=0.23894140672499853,
        scale=0.7757305024942618,
        confidence=0.9595058777998756,
        association=0.82034291642588,
    )
    top_bottom = _classify(
        latch,
        previous_sequence=165,
        previous_frame_id=659087,
        sequence=166,
        frame_id=659088,
        clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
        x=0.0031250000000000444,
        y=0.0,
        x_rate=-0.09244913046229414,
        y_rate=-0.24231141974636716,
        scale=0.835725732522339,
        confidence=0.9417915725865457,
        association=0.8661413767327067,
    )
    full_frame = _classify(
        latch,
        previous_sequence=166,
        previous_frame_id=659088,
        sequence=171,
        frame_id=659093,
        clipping=(
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM
        ),
        x=0.0,
        y=0.0,
        x_rate=-0.02664925020826258,
        y_rate=-0.004471327195188508,
        scale=1.0,
        confidence=0.7743949823768358,
        association=0.951054399368475,
    )
    lost = _classify(
        latch,
        previous_sequence=171,
        previous_frame_id=659093,
        sequence=172,
        frame_id=659094,
        clipping=(
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM
        ),
        visible=False,
        missed_frame_count=1,
        x=None,
        y=None,
        x_rate=None,
        y_rate=None,
        scale=None,
        confidence=None,
        association=None,
        track_latest_sequence=171,
        track_latest_frame_id=659093,
    )

    # Gaps 158 -> 165 and 166 -> 171 are accepted without exact adjacency.
    assert bottom is LatchedMeasurementMode.COAST
    assert top_bottom is LatchedMeasurementMode.COAST
    assert full_frame is LatchedMeasurementMode.CREDIT_WAIT
    assert lost is LatchedMeasurementMode.CREDIT_WAIT


@pytest.mark.parametrize(
    "mutation",
    ("off_center", "ambiguous", "stale_publication", "divergent_scale"),
)
def test_unsafe_or_discontinuous_near_plane_sequences_do_not_latch(mutation):
    samples = [_wire_sample(row) for row in _LATEST_NEAR_PLANE_ROWS]
    if mutation == "off_center":
        samples[-1] = replace(samples[-1], normalized_x=0.21)
    elif mutation == "ambiguous":
        samples[-1] = replace(samples[-1], ambiguous=True)
    elif mutation == "stale_publication":
        samples[-1] = replace(
            samples[-1],
            camera_token=samples[-2].camera_token,
            wire_camera_token=samples[-2].camera_token,
        )
    else:
        samples[-1] = replace(
            samples[-1],
            log_scale=samples[-2].log_scale - 0.01,
            log_scale_rate_s=-0.1,
        )

    evidence = NearPlaneEvidence()
    latch = None
    for sample in samples:
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            **_REDUCER_KWARGS,
        )

    assert latch is None


def test_observable_off_center_and_ambiguous_censor_are_unsafe():
    latch = _latch(_CREDITED_NEAR_PLANE_ROWS)

    off_center = _classify(
        latch,
        previous_sequence=158,
        previous_frame_id=659080,
        sequence=165,
        frame_id=659087,
        clipping=FrameEdge.BOTTOM,
        x=0.21,
        y=0.0,
        x_rate=0.0,
        y_rate=0.0,
        scale=0.7757305024942618,
    )
    ambiguous = _classify(
        latch,
        previous_sequence=158,
        previous_frame_id=659080,
        sequence=165,
        frame_id=659087,
        clipping=FrameEdge.BOTTOM,
        ambiguous=True,
        scale=0.7757305024942618,
    )
    stale = _classify(
        latch,
        previous_sequence=158,
        previous_frame_id=659080,
        sequence=158,
        frame_id=659080,
        clipping=FrameEdge.BOTTOM,
        scale=0.7757305024942618,
    )

    assert off_center is LatchedMeasurementMode.UNSAFE
    assert ambiguous is LatchedMeasurementMode.UNSAFE
    assert stale is LatchedMeasurementMode.UNSAFE


@pytest.mark.parametrize(
    (
        "sequence",
        "frame_id",
        "x",
        "y",
        "scale",
        "confidence",
        "association",
        "observation_ns",
        "publication_ns",
    ),
    (
        (
            180,
            659102,
            0.6125,
            -0.6611111111111111,
            0.20071486824182544,
            0.7954918245413743,
            0.9291081576354688,
            143_875_641_569_700,
            143_875_642_550_600,
        ),
        (
            181,
            659103,
            0.625,
            -0.6777777777777778,
            0.205986784905138,
            0.7981153855597475,
            0.9394529007027884,
            143_875_676_881_600,
            143_875_677_668_400,
        ),
    ),
)
def test_first_fresh_promoted_current_is_ready_without_exact_adjacency(
    sequence,
    frame_id,
    x,
    y,
    scale,
    confidence,
    association,
    observation_ns,
    publication_ns,
):
    credit_watermark = _camera_token(179, 659101)
    snapshot = _promoted_snapshot(
        sequence=sequence,
        frame_id=frame_id,
        x=x,
        y=y,
        scale=scale,
        confidence=confidence,
        association=association,
        observation_ns=observation_ns,
        publication_ns=publication_ns,
    )

    assert _current_snapshot_ready(
        snapshot,
        gate_index=1,
        track_id=_PROMOTED_TRACK,
        newer_than=credit_watermark,
        observed_after_ns=_CREDITED_RACE_RECEIVED_NS,
    )
    assert sequence - credit_watermark.publication_sequence in {1, 2}


def test_credit_boundary_straddle_is_history_only_not_command_authority():
    credit_watermark = _camera_token(179, 659101)
    snapshot = _promoted_snapshot(
        sequence=180,
        frame_id=659102,
        x=0.6125,
        y=-0.6611111111111111,
        scale=0.20071486824182544,
        confidence=0.7954918245413743,
        association=0.9291081576354688,
        observation_ns=_CREDITED_RACE_RECEIVED_NS - 1,
        publication_ns=_CREDITED_RACE_RECEIVED_NS + 1,
    )

    assert not _current_snapshot_ready(
        snapshot,
        gate_index=1,
        track_id=_PROMOTED_TRACK,
        newer_than=credit_watermark,
        observed_after_ns=_CREDITED_RACE_RECEIVED_NS,
    )


def test_first_fresh_one_edge_current_retains_observable_axis_authority():
    credit_watermark = _camera_token(179, 659101)
    snapshot = _promoted_snapshot(
        sequence=181,
        frame_id=659103,
        x=0.628125,
        y=-0.7944444444444445,
        scale=0.1897,
        confidence=0.648,
        association=0.897,
        clipping=FrameEdge.TOP,
        center_censored=True,
        observation_ns=143_875_676_881_600,
        publication_ns=143_875_677_668_400,
    )
    snapshot.current_track.center_velocity_norm_s = (0.373, -0.637)

    assert not _current_snapshot_ready(
        snapshot,
        gate_index=1,
        track_id=_PROMOTED_TRACK,
        newer_than=credit_watermark,
        observed_after_ns=_CREDITED_RACE_RECEIVED_NS,
    )
    assert _current_snapshot_ready(
        snapshot,
        gate_index=1,
        track_id=_PROMOTED_TRACK,
        newer_than=credit_watermark,
        observed_after_ns=_CREDITED_RACE_RECEIVED_NS,
        allow_one_edge_censored=True,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "not_newer",
        "wrong_role",
        "wrong_authoritative_gate",
        "token_mismatch",
        "clipped",
        "censored",
        "missed",
        "ambiguous",
    ),
)
def test_promoted_current_readiness_remains_fail_closed(mutation):
    credit_watermark = _camera_token(179, 659101)
    kwargs = {
        "sequence": 181,
        "frame_id": 659103,
        "x": 0.625,
        "y": -0.6777777777777778,
        "scale": 0.205986784905138,
        "confidence": 0.7981153855597475,
        "association": 0.9394529007027884,
    }
    if mutation == "not_newer":
        kwargs.update(sequence=179, frame_id=659101)
    elif mutation == "wrong_role":
        kwargs["role"] = VisualTrackRole.NEXT
    elif mutation == "wrong_authoritative_gate":
        kwargs["authoritative_gate_index"] = 0
    elif mutation == "token_mismatch":
        kwargs["latest_token"] = _camera_token(180, 659102)
    elif mutation == "clipped":
        kwargs["clipping"] = FrameEdge.TOP
    elif mutation == "censored":
        kwargs["center_censored"] = True
    elif mutation == "missed":
        kwargs["missed_frame_count"] = 1
    else:
        kwargs["ambiguous"] = True
    snapshot = _promoted_snapshot(**kwargs)

    assert not _current_snapshot_ready(
        snapshot,
        gate_index=1,
        track_id=_PROMOTED_TRACK,
        newer_than=credit_watermark,
    )


def test_same_recorded_lifecycle_is_gate_generic_for_gate1_to_gate2():
    reindexed_track = "vq2-track-gate1-current"
    latch = _latch(
        _LATEST_NEAR_PLANE_ROWS,
        gate_index=1,
        track_id=reindexed_track,
    )

    bottom = _classify(
        latch,
        previous_sequence=164,
        previous_frame_id=860516,
        sequence=166,
        frame_id=860518,
        clipping=FrameEdge.BOTTOM,
        x=0.0031250000000000444,
        y=0.005555555555555536,
        x_rate=-0.26958645153182076,
        y_rate=0.2871081869228447,
        scale=0.8018939170280983,
        confidence=0.9497871551140489,
        association=0.8063537755276018,
    )

    assert latch.gate_index == 1
    assert latch.track_id == reindexed_track
    assert latch.lifecycle is CourseLifecycle.NEAR_PLANE_LATCHED
    assert bottom is LatchedMeasurementMode.COAST
