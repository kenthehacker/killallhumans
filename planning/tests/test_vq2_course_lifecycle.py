"""Logged-state regression tests for the generic visual-course lifecycle.

These are exact compact tracker, receiver-publication, wire-receipt, and final
command facts.  The fast-flight-cycle records do not contain JPEG payloads or
the full detector call, so these tests intentionally make no full-image replay
claim.
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
    DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
    DYNAMIC_NEAR_PLANE_LATCH_BASIS,
    LatchedMeasurementMode,
    NearPlaneEvidence,
    NearPlaneLatch,
    NearPlaneWireSample,
    PostCreditMeasurementMode,
    advance_dynamic_near_plane_evidence,
    advance_near_plane_evidence,
    classify_post_credit_measurement,
    classify_latched_measurement,
)


_STREAM = "vq2-camera-udp-5600"
_TRACK = "vq2-track-000001"
_REQUIRED_FRAMES = 3
_CROSSING_MIN_LOG_SCALE = -0.80
_MIN_TRACK_CONFIDENCE = 0.20
_MIN_ASSOCIATION_CONFIDENCE = 0.10


def _token(frame_id: int, publication: int, *, generation: int = 1):
    return CameraFrameToken(
        generation=generation,
        frame_id=frame_id,
        publication_sequence=publication,
        stream_id=_STREAM,
    )


def _sample(
    *,
    frame_id: int,
    publication: int,
    observation_ns: int,
    publication_ns: int,
    wire_start_ns: int,
    wire_return_ns: int,
    x: float,
    y: float,
    x_rate: float,
    y_rate: float,
    apparent_scale: float,
    log_scale_rate: float,
    confidence: float,
    association_confidence: float,
    command: tuple[float, float, float, float],
) -> NearPlaneWireSample:
    token = _token(frame_id, publication)
    return NearPlaneWireSample(
        gate_index=0,
        track_id=_TRACK,
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
        normalized_x_rate_s=x_rate,
        normalized_y_rate_down_s=y_rate,
        log_scale=math.log(apparent_scale),
        log_scale_rate_s=log_scale_rate,
        confidence=confidence,
        association_confidence=association_confidence,
        clipping=FrameEdge.NONE,
        center_censored=False,
        ambiguous=False,
        command_roll_rate=command[0],
        command_pitch_rate=command[1],
        command_yaw_rate=command[2],
        command_thrust=command[3],
    )


def _post_credit_snapshot(
    publication: int,
    *,
    visible: bool = True,
    clipping: FrameEdge = FrameEdge.NONE,
    ambiguous: bool = False,
    latest_track_publication: int | None = None,
):
    token = _token(1_500_000 + publication, publication)
    track_publication = (
        publication
        if latest_track_publication is None
        else latest_track_publication
    )
    latest_track_token = _token(
        1_500_000 + track_publication,
        track_publication,
    )
    track = SimpleNamespace(
        track_id=_TRACK,
        latest_token=latest_track_token,
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=1,
        visible=visible,
        missed_frame_count=0 if visible else 1,
        ambiguous=ambiguous,
        clipping=clipping,
        center_censored=clipping != FrameEdge.NONE,
        center_norm=(0.60, -0.70),
        center_velocity_norm_s=(0.30, -0.70),
    )
    return SimpleNamespace(
        latest_camera_token=token,
        current_gate_index=1,
        current_track_id=_TRACK,
        current_track=track,
        authority_usable=visible and not ambiguous,
    )


# Exact publications 156-158 from:
# C:\Users\John\aigp-evidence\fast-flight-cycles\
# 20260725T202342Z-visual-course-2e71fae8\session.jsonl.gz
_CREDITED_NEAR_PLANE = (
    _sample(
        frame_id=659078,
        publication=156,
        observation_ns=143_874_842_790_600,
        publication_ns=143_874_843_950_900,
        wire_start_ns=143_874_870_779_600,
        wire_return_ns=143_874_870_847_900,
        x=-0.009375000000000022,
        y=-0.050000000000000044,
        x_rate=0.12262423164115838,
        y_rate=0.23078629418568575,
        apparent_scale=0.4092676385936225,
        log_scale_rate=1.1144505987291886,
        confidence=0.9820524393803381,
        association_confidence=0.9273506180084061,
        command=(
            -0.0008138930404383224,
            0.039583051863755034,
            0.0,
            0.2555596088694337,
        ),
    ),
    _sample(
        frame_id=659079,
        publication=157,
        observation_ns=143_874_878_240_600,
        publication_ns=143_874_879_360_500,
        wire_start_ns=143_874_902_593_600,
        wire_return_ns=143_874_902_711_800,
        x=-0.009375000000000022,
        y=-0.050000000000000044,
        x_rate=0.05518090423852126,
        y_rate=0.10385383238355858,
        apparent_scale=0.4310688025130095,
        log_scale_rate=1.3066965144902496,
        confidence=0.9820235977211522,
        association_confidence=0.917866701002499,
        command=(
            -0.001111300660357516,
            0.0415224200402323,
            0.0,
            0.2637637457651319,
        ),
    ),
    _sample(
        frame_id=659080,
        publication=158,
        observation_ns=143_874_905_646_700,
        publication_ns=143_874_906_653_500,
        wire_start_ns=143_874_933_521_300,
        wire_return_ns=143_874_933_597_800,
        x=-0.009375000000000022,
        y=-0.0444444444444444,
        x_rate=0.024831406907334565,
        y_rate=0.15822602951951487,
        apparent_scale=0.4539324701979359,
        log_scale_rate=1.6251694670673007,
        confidence=0.9809106189745185,
        association_confidence=0.923499758396885,
        command=(
            -0.0005943738190254716,
            0.03918836610569788,
            0.0,
            0.25971237467284425,
        ),
    ),
)


# Exact publications 157-159 from:
# C:\Users\John\aigp-evidence\fast-flight-cycles\
# 20260725T221536Z-visual-course-508a76b3\session.jsonl.gz
_LATEST_NEAR_PLANE = (
    _sample(
        frame_id=860509,
        publication=157,
        observation_ns=150_589_210_072_800,
        publication_ns=150_589_211_121_700,
        wire_start_ns=150_589_217_560_900,
        wire_return_ns=150_589_217_634_700,
        x=0.006250000000000089,
        y=-0.09999999999999998,
        x_rate=0.11872072885028712,
        y_rate=0.3141879016683444,
        apparent_scale=0.4133198922545748,
        log_scale_rate=1.2488526265213373,
        confidence=0.9773336531922997,
        association_confidence=0.925719274434676,
        command=(
            -0.00018267622154919438,
            0.026147634053156288,
            -0.006030225509760076,
            0.24493819547655876,
        ),
    ),
    _sample(
        frame_id=860510,
        publication=158,
        observation_ns=150_589_239_321_800,
        publication_ns=150_589_240_368_700,
        wire_start_ns=150_589_248_088_900,
        wire_return_ns=150_589_248_168_000,
        x=0.006250000000000089,
        y=-0.09444444444444444,
        x_rate=0.053424327982629194,
        y_rate=0.24585156506921185,
        apparent_scale=0.4299951550114644,
        log_scale_rate=1.3057236320217758,
        confidence=0.979021249464173,
        association_confidence=0.9297253788584732,
        command=(
            -0.00013983557561611295,
            0.0265550864272439,
            -0.0037448514793920486,
            0.24943902787210168,
        ),
    ),
    _sample(
        frame_id=860511,
        publication=159,
        observation_ns=150_589_272_303_300,
        publication_ns=150_589_273_564_200,
        wire_start_ns=150_589_279_796_100,
        wire_return_ns=150_589_279_865_100,
        x=0.009374999999999911,
        y=-0.08333333333333337,
        x_rate=0.07615349553572429,
        y_rate=0.2959222636359682,
        apparent_scale=0.45694866597171874,
        log_scale_rate=1.60142925833799,
        confidence=0.977202419401735,
        association_confidence=0.9181027530135752,
        command=(
            -0.00020683431026789854,
            0.023803971639617484,
            -0.005477872343750323,
            0.24528411000436037,
        ),
    ),
)


def _advance(
    samples: tuple[NearPlaneWireSample, ...],
) -> tuple[NearPlaneEvidence, NearPlaneLatch]:
    evidence = NearPlaneEvidence()
    latch = None
    for index, sample in enumerate(samples):
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
        if index < len(samples) - 1:
            assert latch is None
    assert latch is not None
    return evidence, latch


def _latest_censored_kwargs(latch: NearPlaneLatch):
    """Exact latest-failure publication 166 after the publication-159 latch."""

    return {
        "previous_camera_token": latch.anchor_camera_token,
        "camera_token": _token(860518, 166),
        "current_gate_index": 0,
        "current_track_id": _TRACK,
        "track_latest_camera_token": _token(860518, 166),
        "track_role": VisualTrackRole.CURRENT,
        "track_authoritative_gate_index": 0,
        "visible": True,
        "missed_frame_count": 0,
        "ambiguous": False,
        "clipping": FrameEdge.BOTTOM,
        "center_censored": True,
        "normalized_x": 0.0031250000000000444,
        "normalized_y_down": 0.005555555555555536,
        "normalized_x_rate_s": -0.26958645153182076,
        "normalized_y_rate_down_s": 0.2871081869228447,
        "apparent_scale": 0.8018939170280983,
        "confidence": 0.9497871551140489,
        "association_confidence": 0.8063537755276018,
        "min_track_confidence": _MIN_TRACK_CONFIDENCE,
        "min_association_confidence": _MIN_ASSOCIATION_CONFIDENCE,
    }


def test_course_lifecycle_is_gate_generic_and_explicit():
    assert tuple(CourseLifecycle) == (
        CourseLifecycle.APPROACH,
        CourseLifecycle.PASSAGE_ARMED,
        CourseLifecycle.NEAR_PLANE_LATCHED,
        CourseLifecycle.CREDIT_WAIT,
        CourseLifecycle.PROMOTE_REACQUIRE,
    )


def test_primary_credited_wire_facts_latch_before_censorship():
    evidence, latch = _advance(_CREDITED_NEAR_PLANE)

    assert len(evidence.samples) == 3
    assert latch.lifecycle is CourseLifecycle.NEAR_PLANE_LATCHED
    assert latch.anchor_camera_token == _token(659080, 158)
    assert latch.gate_index == 0
    assert latch.track_id == _TRACK
    assert latch.accepted_command == (
        -0.0005943738190254716,
        0.03918836610569788,
        0.0,
        0.25971237467284425,
    )


def test_latest_failure_wire_facts_latch_without_advance_command_count():
    evidence, latch = _advance(_LATEST_NEAR_PLANE)

    assert len(evidence.samples) == 3
    assert latch.anchor_camera_token == _token(860511, 159)
    assert all(sample.log_scale_rate_s > 0.0 for sample in evidence.samples)
    assert evidence.samples[-1].log_scale >= _CROSSING_MIN_LOG_SCALE


def test_dynamic_derotated_history_latches_and_allows_bounded_raw_motion():
    dynamic_samples = tuple(
        replace(
            sample,
            normalized_x=-0.04,
            normalized_y_down=-0.08,
            normalized_x_rate_s=-0.70,
            normalized_y_rate_down_s=0.20,
            log_scale=math.log(scale),
            log_scale_rate_s=expansion,
            geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
            normalized_x_std=0.015,
            normalized_y_std=0.015,
            log_scale_std=0.02,
            crossing_prediction_horizon_s=0.50,
            predicted_crossing_x_norm=0.10,
            predicted_crossing_y_down_norm=-0.12,
            predicted_crossing_x_std_norm=0.015,
            predicted_crossing_y_std_norm=0.015,
            crossing_allowance_x_norm=0.30,
            crossing_allowance_y_norm=0.30,
            crossing_swept_x_occupancy_norm=0.13,
            crossing_swept_y_occupancy_norm=0.15,
            current_crossing_x_q=0.10,
            current_crossing_y_q=-0.12,
            crossing_x_q_rate_s=-0.10,
            crossing_y_q_rate_s=0.10,
            post_governor_contact_budget_s=0.50,
        )
        for sample, scale, expansion in zip(
            _CREDITED_NEAR_PLANE,
            (0.50, 0.55, 0.60),
            (1.64, 1.74, 1.86),
        )
    )
    evidence = NearPlaneEvidence()
    latch = None
    for sample in dynamic_samples:
        evidence, latch = advance_dynamic_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            horizontal_corridor=0.16,
            vertical_corridor=0.18,
            minimum_post_governor_contact_budget_s=0.12,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )

    assert latch is not None
    assert latch.basis == DYNAMIC_NEAR_PLANE_LATCH_BASIS
    facts = _latest_censored_kwargs(latch)
    facts["normalized_x"] = -0.40
    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.COAST
    )


def test_dynamic_terminal_clearance_can_commit_before_legacy_scale_gate():
    sample = replace(
        _CREDITED_NEAR_PLANE[0],
        normalized_x=0.01,
        normalized_y_down=-0.17,
        normalized_x_rate_s=0.03,
        normalized_y_rate_down_s=-0.02,
        log_scale=-1.47,
        log_scale_rate_s=0.89,
        geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
        normalized_x_std=0.01,
        normalized_y_std=0.01,
        log_scale_std=0.03,
        crossing_prediction_horizon_s=1.12,
        predicted_crossing_x_norm=0.20,
        predicted_crossing_y_down_norm=-0.08,
        predicted_crossing_x_std_norm=0.047,
        predicted_crossing_y_std_norm=0.083,
        crossing_allowance_x_norm=0.50,
        crossing_allowance_y_norm=0.45,
        crossing_swept_x_occupancy_norm=0.30,
        crossing_swept_y_occupancy_norm=0.40,
        current_crossing_x_q=0.04,
        current_crossing_y_q=-0.57,
        crossing_x_q_rate_s=0.15,
        crossing_y_q_rate_s=0.43,
        post_governor_contact_budget_s=1.02,
    )

    evidence, latch = advance_dynamic_near_plane_evidence(
        NearPlaneEvidence(),
        sample,
        required_corridor_frames=1,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        horizontal_corridor=0.16,
        vertical_corridor=0.18,
        minimum_post_governor_contact_budget_s=0.12,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert sample.log_scale < _CROSSING_MIN_LOG_SCALE
    assert latch is not None
    assert latch.basis == DYNAMIC_NEAR_PLANE_LATCH_BASIS
    assert latch.evidence == evidence


def test_qualified_propagated_clip_mints_only_safe_local_passage_latch():
    sample = replace(
        _CREDITED_NEAR_PLANE[0],
        normalized_x=0.02,
        normalized_y_down=-0.25,
        normalized_x_rate_s=0.20,
        normalized_y_rate_down_s=-0.06,
        log_scale=-0.92,
        log_scale_rate_s=1.64,
        confidence=0.83,
        clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
        center_censored=True,
        geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
        normalized_x_std=0.02,
        normalized_y_std=0.03,
        log_scale_std=0.12,
        crossing_prediction_horizon_s=0.61,
        predicted_crossing_x_norm=0.33,
        predicted_crossing_y_down_norm=0.20,
        predicted_crossing_x_std_norm=0.055,
        predicted_crossing_y_std_norm=0.065,
        crossing_allowance_x_norm=0.50,
        crossing_allowance_y_norm=0.45,
        crossing_swept_x_occupancy_norm=0.44,
        crossing_swept_y_occupancy_norm=0.63,
        current_crossing_x_q=0.08,
        current_crossing_y_q=-1.39,
        crossing_x_q_rate_s=0.12,
        crossing_y_q_rate_s=1.87,
        post_governor_contact_budget_s=0.53,
        propagated_state_horizon_remaining_s=0.80,
        propagated_state_dynamics_qualified=True,
    )

    evidence, latch = advance_dynamic_near_plane_evidence(
        NearPlaneEvidence(),
        sample,
        required_corridor_frames=1,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        horizontal_corridor=0.16,
        vertical_corridor=0.18,
        minimum_post_governor_contact_budget_s=0.12,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert latch is not None
    assert latch.basis == DYNAMIC_NEAR_PLANE_LATCH_BASIS
    assert latch.evidence == evidence
    assert latch.anchor_sample.clipping == FrameEdge.TOP | FrameEdge.BOTTOM
    assert latch.anchor_sample.center_censored is True

    multi_edge_clipped = replace(
        sample,
        clipping=(
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM
        ),
    )
    evidence, latch = advance_dynamic_near_plane_evidence(
        NearPlaneEvidence(),
        multi_edge_clipped,
        required_corridor_frames=1,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        horizontal_corridor=0.16,
        vertical_corridor=0.18,
        minimum_post_governor_contact_budget_s=0.12,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )
    assert latch is not None
    assert latch.anchor_sample.clipping == multi_edge_clipped.clipping

    unsafe = replace(
        multi_edge_clipped,
        predicted_crossing_y_down_norm=0.40,
        predicted_crossing_y_std_norm=0.10,
    )
    evidence, latch = advance_dynamic_near_plane_evidence(
        NearPlaneEvidence(),
        unsafe,
        required_corridor_frames=1,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        horizontal_corridor=0.16,
        vertical_corridor=0.18,
        minimum_post_governor_contact_budget_s=0.12,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )
    assert evidence.samples == ()
    assert latch is None


def test_propagated_passage_state_must_outlive_crossing_prediction():
    with pytest.raises(ValueError, match="expires before crossing"):
        replace(
            _CREDITED_NEAR_PLANE[0],
            geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
            crossing_prediction_horizon_s=0.61,
            predicted_crossing_x_norm=0.10,
            predicted_crossing_y_down_norm=-0.12,
            predicted_crossing_x_std_norm=0.015,
            predicted_crossing_y_std_norm=0.015,
            crossing_allowance_x_norm=0.30,
            crossing_allowance_y_norm=0.30,
            crossing_swept_x_occupancy_norm=0.13,
            crossing_swept_y_occupancy_norm=0.15,
            current_crossing_x_q=0.10,
            current_crossing_y_q=-0.12,
            crossing_x_q_rate_s=-0.10,
            crossing_y_q_rate_s=0.10,
            post_governor_contact_budget_s=0.50,
            propagated_state_horizon_remaining_s=0.60,
            propagated_state_dynamics_qualified=True,
        )


def test_5dffc517_predicted_clearance_latches_while_braking() -> None:
    """Replay the final three clean dynamic states before its bottom clip."""

    replay_facts = (
        {
            "normalized_x": 0.08755705173981657,
            "normalized_y_down": -0.6663688234686937,
            "normalized_x_rate_s": -0.07266507862197541,
            "normalized_y_rate_down_s": 0.17732009079152664,
            "log_scale": -0.4888813534482041,
            "log_scale_rate_s": 1.8323514969963652,
            "confidence": 0.9162014714288149,
            "normalized_x_std": 0.0166015902854566,
            "normalized_y_std": 0.02192254240992148,
            "log_scale_std": 0.046214360066108545,
            "crossing_prediction_horizon_s": 0.545746818576689,
            "predicted_crossing_x_norm": 0.04790031626024851,
            "predicted_crossing_y_down_norm": -0.5695969480494885,
            "predicted_crossing_x_std_norm": 0.01805489185789611,
            "predicted_crossing_y_std_norm": 0.02546894422575201,
            "crossing_allowance_x_norm": 0.43144731788958546,
            "crossing_allowance_y_norm": 0.7585534078076088,
            "crossing_swept_x_occupancy_norm": (
                0.04790031626024851 + 2.0 * 0.01805489185789611
            ),
            "crossing_swept_y_occupancy_norm": (
                0.5695969480494885 + 2.0 * 0.02546894422575201
            ),
            "current_crossing_x_q": 0.10,
            "current_crossing_y_q": -0.12,
            "crossing_x_q_rate_s": 0.0,
            "crossing_y_q_rate_s": 0.0,
            "post_governor_contact_budget_s": 0.50,
        },
        {
            "normalized_x": 0.07326644915155803,
            "normalized_y_down": -0.6481111955218952,
            "normalized_x_rate_s": -0.18452990189829785,
            "normalized_y_rate_down_s": 0.2734674067961661,
            "log_scale": -0.4330575882676292,
            "log_scale_rate_s": 1.7524624980079508,
            "confidence": 0.9147545431145462,
            "normalized_x_std": 0.01641192699254262,
            "normalized_y_std": 0.02166944716677914,
            "log_scale_std": 0.04583523085268929,
            "crossing_prediction_horizon_s": 0.5706256203123972,
            "predicted_crossing_x_norm": -0.03203104058534398,
            "predicted_crossing_y_down_norm": -0.4920636868836103,
            "predicted_crossing_x_std_norm": 0.020102525030508578,
            "predicted_crossing_y_std_norm": 0.02713879530270246,
            "crossing_allowance_x_norm": 0.4541013180219873,
            "crossing_allowance_y_norm": 0.8247772155886629,
            "crossing_swept_x_occupancy_norm": (
                0.03203104058534398 + 2.0 * 0.020102525030508578
            ),
            "crossing_swept_y_occupancy_norm": (
                0.4920636868836103 + 2.0 * 0.02713879530270246
            ),
            "current_crossing_x_q": 0.10,
            "current_crossing_y_q": -0.12,
            "crossing_x_q_rate_s": 0.0,
            "crossing_y_q_rate_s": 0.0,
            "post_governor_contact_budget_s": 0.50,
        },
        {
            "normalized_x": 0.05873231376519952,
            "normalized_y_down": -0.6256504945104799,
            "normalized_x_rate_s": -0.1965476878653643,
            "normalized_y_rate_down_s": 0.3957952406238849,
            "log_scale": -0.3691951134967805,
            "log_scale_rate_s": 1.7749301488475058,
            "confidence": 0.9068386259584491,
            "normalized_x_std": 0.01647398117883298,
            "normalized_y_std": 0.021738417343367785,
            "log_scale_std": 0.046005207747579116,
            "crossing_prediction_horizon_s": 0.5634024531327715,
            "predicted_crossing_x_norm": -0.05200313573572099,
            "predicted_crossing_y_down_norm": -0.40265848500470763,
            "predicted_crossing_x_std_norm": 0.020404934936140266,
            "predicted_crossing_y_std_norm": 0.029654322155845483,
            "crossing_allowance_x_norm": 0.4903944647215772,
            "crossing_allowance_y_norm": 0.8982225951480185,
            "crossing_swept_x_occupancy_norm": (
                0.05200313573572099 + 2.0 * 0.020404934936140266
            ),
            "crossing_swept_y_occupancy_norm": (
                0.40265848500470763 + 2.0 * 0.029654322155845483
            ),
            "current_crossing_x_q": 0.10,
            "current_crossing_y_q": -0.12,
            "crossing_x_q_rate_s": 0.0,
            "crossing_y_q_rate_s": 0.0,
            "post_governor_contact_budget_s": 0.50,
        },
    )
    dynamic_samples = tuple(
        replace(
            sample,
            geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
            **facts,
        )
        for sample, facts in zip(_LATEST_NEAR_PLANE, replay_facts)
    )

    evidence = NearPlaneEvidence()
    latch = None
    for sample in dynamic_samples:
        evidence, latch = advance_dynamic_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            # Retained only as a validated compatibility surface.  The first
            # sample is far outside these obsolete fixed center corridors.
            horizontal_corridor=0.16,
            vertical_corridor=0.18,
            minimum_post_governor_contact_budget_s=0.12,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )

    assert latch is not None
    assert latch.basis == DYNAMIC_NEAR_PLANE_LATCH_BASIS
    assert len(evidence.samples) == _REQUIRED_FRAMES
    assert all(
        sample.crossing_allowance_x_norm
        - sample.crossing_swept_x_occupancy_norm
        >= 0.0
        and sample.crossing_allowance_y_norm
        - sample.crossing_swept_y_occupancy_norm
        >= 0.0
        for sample in evidence.samples
    )


def test_1cab9da6_terminal_window_latches_after_pre_scale_history() -> None:
    """Admit its converging terminal envelope, not its unsafe sweep start."""

    base_samples = (
        _sample(
            frame_id=464795,
            publication=152,
            observation_ns=15_715_654_389_100,
            publication_ns=15_715_655_331_900,
            wire_start_ns=15_715_675_209_800,
            wire_return_ns=15_715_675_289_300,
            x=0.005764799323353661,
            y=-1.1231932479269657,
            x_rate=0.12525669811216106,
            y_rate=-0.008273899560235554,
            apparent_scale=math.exp(-0.8290109598193538),
            log_scale_rate=1.1465154650938152,
            confidence=0.9199250784072561,
            association_confidence=0.9199250784072561,
            command=(
                0.001674900342085917,
                0.040946787969841636,
                0.033094757316025764,
                0.2511672635095732,
            ),
        ),
        _sample(
            frame_id=464796,
            publication=153,
            observation_ns=15_715_682_402_100,
            publication_ns=15_715_683_246_500,
            wire_start_ns=15_715_707_048_200,
            wire_return_ns=15_715_707_097_900,
            x=0.012866218978527365,
            y=-1.115962050043321,
            x_rate=0.13313577786259895,
            y_rate=0.02327797829561726,
            apparent_scale=math.exp(-0.7809297271930684),
            log_scale_rate=1.2358089083301733,
            confidence=0.9272929937598157,
            association_confidence=0.9272929937598157,
            command=(
                0.0013993488862322915,
                0.038429543405929446,
                0.03226922676223867,
                0.24639907850957318,
            ),
        ),
        _sample(
            frame_id=464797,
            publication=154,
            observation_ns=15_715_716_894_800,
            publication_ns=15_715_717_745_600,
            wire_start_ns=15_715_738_030_200,
            wire_return_ns=15_715_738_119_000,
            x=0.06630000578183558,
            y=-1.1096365234848757,
            x_rate=0.13825717970038356,
            y_rate=0.05309540065020987,
            apparent_scale=math.exp(-0.6755779173200508),
            log_scale_rate=1.2969035199480012,
            confidence=0.747236218916346,
            association_confidence=0.747236218916346,
            command=(
                0.000639909443465941,
                0.03701451226687369,
                0.0312928717386639,
                0.2417594135095732,
            ),
        ),
    )
    replay_facts = (
        {
            "log_scale": -0.8290109598193538,
            "log_scale_std": 0.04517566370817692,
            "normalized_x_std": 0.016057561508406433,
            "normalized_y_std": 0.02130477103991473,
            "crossing_prediction_horizon_s": 0.872208034209267,
            "predicted_crossing_x_norm": 0.29185011994412513,
            "predicted_crossing_y_down_norm": -0.009951378950815437,
            "predicted_crossing_x_std_norm": 0.049930954979794794,
            "predicted_crossing_y_std_norm": 0.13463579007645196,
            "crossing_allowance_x_norm": 0.50,
            "crossing_allowance_y_norm": 0.45,
            "crossing_swept_x_occupancy_norm": 0.39171202990371473,
            "crossing_swept_y_occupancy_norm": 1.7475401400798543,
            "current_crossing_x_q": 0.015400081813554532,
            "current_crossing_y_q": -1.548843085556051,
            "crossing_x_q_rate_s": 0.31695424404247413,
            "crossing_y_q_rate_s": 1.7643631407275169,
            "post_governor_contact_budget_s": 0.5177596108121123,
        },
        {
            "log_scale": -0.7809297271930684,
            "log_scale_std": 0.044457095321514845,
            "normalized_x_std": 0.01578456892757788,
            "normalized_y_std": 0.020688955127417703,
            "crossing_prediction_horizon_s": 0.8091865928942051,
            "predicted_crossing_x_norm": 0.2710737804470746,
            "predicted_crossing_y_down_norm": 0.02457502965613867,
            "predicted_crossing_x_std_norm": 0.04705602966410842,
            "predicted_crossing_y_std_norm": 0.12831316633166714,
            "crossing_allowance_x_norm": 0.50,
            "crossing_allowance_y_norm": 0.45,
            "crossing_swept_x_occupancy_norm": 0.3651858397752914,
            "crossing_swept_y_occupancy_norm": 1.6394003020165357,
            "current_crossing_x_q": 0.032373897898515074,
            "current_crossing_y_q": -1.455960320338022,
            "crossing_x_q_rate_s": 0.29498744127087595,
            "crossing_y_q_rate_s": 1.8296587746205148,
            "post_governor_contact_budget_s": 0.4865260694970506,
        },
        {
            "log_scale": -0.6755779173200508,
            "log_scale_std": 0.052343769710160595,
            "normalized_x_std": 0.018808537220433994,
            "normalized_y_std": 0.024304489328331846,
            "crossing_prediction_horizon_s": 0.7710673805867182,
            "predicted_crossing_x_norm": 0.25298049972837156,
            "predicted_crossing_y_down_norm": 0.05128657209432386,
            "predicted_crossing_x_std_norm": 0.05534993410482621,
            "predicted_crossing_y_std_norm": 0.14059388915832344,
            "crossing_allowance_x_norm": 0.50,
            "crossing_allowance_y_norm": 0.45,
            "crossing_swept_x_occupancy_norm": 0.36368036793802394,
            "crossing_swept_y_occupancy_norm": 1.5964812450733752,
            "current_crossing_x_q": 0.1573332768127754,
            "current_crossing_y_q": -1.3900652360978418,
            "crossing_x_q_rate_s": 0.12404522007248783,
            "crossing_y_q_rate_s": 1.8692942335278362,
            "post_governor_contact_budget_s": 0.4793379571895635,
        },
    )
    samples = tuple(
        replace(
            sample,
            geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
            **facts,
        )
        for sample, facts in zip(base_samples, replay_facts)
    )

    evidence = NearPlaneEvidence()
    latch = None
    retained_counts = []
    for sample in samples:
        evidence, latch = advance_dynamic_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            horizontal_corridor=0.16,
            vertical_corridor=0.18,
            minimum_post_governor_contact_budget_s=0.12,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
        retained_counts.append(len(evidence.samples))

    assert retained_counts == [1, 2, 3]
    assert latch is not None
    assert latch.basis == DYNAMIC_NEAR_PLANE_LATCH_BASIS
    assert latch.anchor_camera_token == _token(464797, 154)
    assert all(
        sample.log_scale - 2.0 * sample.log_scale_std
        < _CROSSING_MIN_LOG_SCALE
        for sample in evidence.samples[:-1]
    )
    assert (
        evidence.samples[-1].log_scale
        - 2.0 * evidence.samples[-1].log_scale_std
        >= _CROSSING_MIN_LOG_SCALE
    )
    assert all(
        sample.crossing_allowance_x_norm
        - (
            abs(sample.predicted_crossing_x_norm)
            + 2.0 * sample.predicted_crossing_x_std_norm
        )
        >= 0.0
        and sample.crossing_allowance_y_norm
        - (
            abs(sample.predicted_crossing_y_down_norm)
            + 2.0 * sample.predicted_crossing_y_std_norm
        )
        >= 0.0
        and sample.crossing_allowance_y_norm
        - sample.crossing_swept_y_occupancy_norm
        < 0.0
        and sample.current_crossing_y_q
        * sample.crossing_y_q_rate_s
        < 0.0
        for sample in evidence.samples
    )


def test_e38e73be_historical_thrust_settle_does_not_erase_corridor() -> None:
    """Require contact budget on the live anchor, not its geometry history."""

    base_samples = (
        _sample(
            frame_id=513779,
            publication=161,
            observation_ns=17_348_448_085_200,
            publication_ns=17_348_448_990_200,
            wire_start_ns=17_348_459_330_800,
            wire_return_ns=17_348_459_366_400,
            x=0.05053606646517718,
            y=-0.6223224697616248,
            x_rate=0.05025334644147395,
            y_rate=-0.033858194627571905,
            apparent_scale=math.exp(-0.7146496419146361),
            log_scale_rate=1.260834536632019,
            confidence=0.7123055934205911,
            association_confidence=0.7123055934205911,
            command=(
                0.010134807198040273,
                0.08262205795505823,
                -0.006088170006393641,
                0.3039436872898472,
            ),
        ),
        _sample(
            frame_id=513780,
            publication=162,
            observation_ns=17_348_483_033_900,
            publication_ns=17_348_483_939_100,
            wire_start_ns=17_348_506_283_200,
            wire_return_ns=17_348_506_332_900,
            x=0.08653616414151748,
            y=-0.6273651659041501,
            x_rate=0.04091165485339476,
            y_rate=0.03821930443582828,
            apparent_scale=math.exp(-0.6180676601303003),
            log_scale_rate=1.3064908467203953,
            confidence=0.806924018259926,
            association_confidence=0.806924018259926,
            command=(
                0.011376099245041552,
                0.07577596723099314,
                -0.02245869711492665,
                0.2969078922898472,
            ),
        ),
        _sample(
            frame_id=513781,
            publication=163,
            observation_ns=17_348_517_665_600,
            publication_ns=17_348_518_525_600,
            wire_start_ns=17_348_537_364_000,
            wire_return_ns=17_348_537_409_200,
            x=0.10140027539268794,
            y=-0.617641999064406,
            x_rate=-0.028659852912795117,
            y_rate=0.20990216181945368,
            apparent_scale=math.exp(-0.5380412056929045),
            log_scale_rate=1.3315694955408057,
            confidence=0.8718116627788197,
            association_confidence=0.8718116627788197,
            command=(
                0.014274192011554394,
                0.0725351021480228,
                -0.03499463959022381,
                0.2922525972898472,
            ),
        ),
    )
    model_facts = (
        {
            "log_scale_std": 0.05393956114685996,
            "normalized_x_std": 0.019313857093835304,
            "normalized_y_std": 0.024757701006388015,
            "crossing_prediction_horizon_s": 0.7931254823263578,
            "predicted_crossing_x_norm": 0.10799004469669124,
            "predicted_crossing_y_down_norm": -0.0419069483569362,
            "predicted_crossing_x_std_norm": 0.06044461697473255,
            "predicted_crossing_y_std_norm": 0.11445338758528793,
            "crossing_allowance_x_norm": 0.50,
            "crossing_allowance_y_norm": 0.45,
            "crossing_swept_x_occupancy_norm": 0.2563536037477585,
            "crossing_swept_y_occupancy_norm": 1.153212086131139,
            "current_crossing_x_q": 0.1369235861305697,
            "current_crossing_y_q": -0.9711712520718878,
            "crossing_x_q_rate_s": -0.03648040830690749,
            "crossing_y_q_rate_s": 1.1716485277831168,
            "post_governor_contact_budget_s": 0.15617276046455952,
        },
        {
            "log_scale_std": 0.05404086897226105,
            "normalized_x_std": 0.019777280285008324,
            "normalized_y_std": 0.025594730559642065,
            "crossing_prediction_horizon_s": 0.7654091128998257,
            "predicted_crossing_x_norm": 0.06363920563947952,
            "predicted_crossing_y_down_norm": 0.040047913656949485,
            "predicted_crossing_x_std_norm": 0.05262943186855274,
            "predicted_crossing_y_std_norm": 0.10494126918460928,
            "crossing_allowance_x_norm": 0.50,
            "crossing_allowance_y_norm": 0.45,
            "crossing_swept_x_occupancy_norm": 0.2752598670183933,
            "crossing_swept_y_occupancy_norm": 1.0217687753139812,
            "current_crossing_x_q": 0.17586593085114566,
            "current_crossing_y_q": -0.8588629917415753,
            "crossing_x_q_rate_s": -0.14662318924644685,
            "crossing_y_q_rate_s": 1.174418869920316,
            "post_governor_contact_budget_s": 0.10602316430084424,
        },
        {
            "log_scale_std": 0.051482961581514705,
            "normalized_x_std": 0.018899687686455658,
            "normalized_y_std": 0.024728570304547046,
            "crossing_prediction_horizon_s": 0.7509934730022172,
            "predicted_crossing_x_norm": -0.04245051623961754,
            "predicted_crossing_y_down_norm": 0.20356659687707312,
            "predicted_crossing_x_std_norm": 0.05402858274012277,
            "predicted_crossing_y_std_norm": 0.0996599382573406,
            "crossing_allowance_x_norm": 0.50,
            "crossing_allowance_y_norm": 0.45,
            "crossing_swept_x_occupancy_norm": 0.29513572728859017,
            "crossing_swept_y_occupancy_norm": 0.9436038576650027,
            "current_crossing_x_q": 0.19999170865222554,
            "current_crossing_y_q": -0.7976093977128692,
            "crossing_x_q_rate_s": -0.32282867109702207,
            "crossing_y_q_rate_s": 1.333135414063694,
            "post_governor_contact_budget_s": 0.12264282440323582,
        },
    )
    samples = tuple(
        replace(
            sample,
            geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
            **facts,
        )
        for sample, facts in zip(base_samples, model_facts)
    )

    evidence = NearPlaneEvidence()
    latch = None
    retained_counts = []
    for sample in samples:
        evidence, latch = advance_dynamic_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            horizontal_corridor=0.16,
            vertical_corridor=0.18,
            minimum_post_governor_contact_budget_s=0.12,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
        retained_counts.append(len(evidence.samples))

    assert retained_counts == [1, 2, 3]
    assert samples[1].post_governor_contact_budget_s < 0.12
    assert samples[2].post_governor_contact_budget_s >= 0.12
    assert latch is not None
    assert latch.anchor_camera_token == _token(513781, 163)

    under_budget_anchor = replace(
        samples[2],
        post_governor_contact_budget_s=0.119,
    )
    evidence = NearPlaneEvidence()
    latch = None
    for sample in (*samples[:2], under_budget_anchor):
        evidence, latch = advance_dynamic_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            horizontal_corridor=0.16,
            vertical_corridor=0.18,
            minimum_post_governor_contact_budget_s=0.12,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )

    assert len(evidence.samples) == 3
    assert latch is None


def test_zero_dynamic_crossing_allowance_is_valid_but_cannot_latch() -> None:
    sample = replace(
        _LATEST_NEAR_PLANE[0],
        geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
        crossing_prediction_horizon_s=0.0,
        predicted_crossing_x_norm=0.0,
        predicted_crossing_y_down_norm=0.0,
        predicted_crossing_x_std_norm=0.0,
        predicted_crossing_y_std_norm=0.0,
        crossing_allowance_x_norm=0.0,
        crossing_allowance_y_norm=0.30,
        crossing_swept_x_occupancy_norm=0.0,
        crossing_swept_y_occupancy_norm=0.0,
        current_crossing_x_q=0.0,
        current_crossing_y_q=0.0,
        crossing_x_q_rate_s=0.0,
        crossing_y_q_rate_s=0.0,
        post_governor_contact_budget_s=0.50,
    )

    evidence, latch = advance_dynamic_near_plane_evidence(
        NearPlaneEvidence(),
        sample,
        required_corridor_frames=_REQUIRED_FRAMES,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        horizontal_corridor=0.16,
        vertical_corridor=0.18,
        minimum_post_governor_contact_budget_s=0.12,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert evidence.samples == ()
    assert latch is None


@pytest.mark.parametrize(
    ("changed", "message"),
    (
        ({"crossing_prediction_horizon_s": True}, "horizon"),
        ({"crossing_prediction_horizon_s": 1.21}, "horizon"),
        ({"predicted_crossing_x_norm": math.nan}, "crossing_x"),
        ({"predicted_crossing_y_std_norm": -0.01}, "crossing_y_std"),
        ({"crossing_allowance_x_norm": -0.01}, "allowance_x"),
    ),
)
def test_dynamic_crossing_sample_rejects_malformed_model_facts(
    changed,
    message,
) -> None:
    valid = replace(
        _LATEST_NEAR_PLANE[0],
        geometry_basis=DYNAMIC_NEAR_PLANE_GEOMETRY_BASIS,
        crossing_prediction_horizon_s=0.50,
        predicted_crossing_x_norm=0.10,
        predicted_crossing_y_down_norm=-0.12,
        predicted_crossing_x_std_norm=0.015,
        predicted_crossing_y_std_norm=0.015,
        crossing_allowance_x_norm=0.30,
        crossing_allowance_y_norm=0.30,
        crossing_swept_x_occupancy_norm=0.13,
        crossing_swept_y_occupancy_norm=0.15,
        current_crossing_x_q=0.10,
        current_crossing_y_q=-0.12,
        crossing_x_q_rate_s=-0.10,
        crossing_y_q_rate_s=0.10,
        post_governor_contact_budget_s=0.50,
    )

    with pytest.raises(ValueError, match=message):
        replace(valid, **changed)


def test_latest_bottom_censor_is_safe_coast_across_skipped_publications():
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)

    # Publications 160-165 may be processed elsewhere or superseded.  Physical
    # authority is strict-newer, not exact-adjacent.
    mode = classify_latched_measurement(
        latch,
        **_latest_censored_kwargs(latch),
    )

    assert mode is LatchedMeasurementMode.COAST


def test_censored_axis_never_creates_fake_offcenter_evidence():
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(latch)
    facts.update(
        normalized_y_down=100.0,
        normalized_y_rate_down_s=100.0,
    )

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.COAST
    )


def test_observable_axis_offcenter_after_latch_is_unsafe():
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(latch)
    facts["normalized_x"] = 0.21

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.UNSAFE
    )


def test_post_latch_projection_divergence_remains_bounded_coast():
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(latch)
    facts.update(
        clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
        center_censored=True,
        normalized_x=-0.196875,
        normalized_x_rate_s=-0.5354,
    )
    assert abs(facts["normalized_x"]) < 0.20
    assert abs(
        facts["normalized_x"] + facts["normalized_x_rate_s"] * 0.10
    ) > 0.20

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.COAST
    )


def test_high_raw_center_rate_inside_projected_corridor_latches_and_coasts():
    high_rate_samples = tuple(
        replace(
            sample,
            normalized_x=-0.10,
            normalized_y_down=-0.07,
            normalized_x_rate_s=1.00,
            normalized_y_rate_down_s=0.70,
        )
        for sample in _LATEST_NEAR_PLANE
    )

    evidence, latch = _advance(high_rate_samples)

    assert len(evidence.samples) == _REQUIRED_FRAMES
    assert all(
        sample.normalized_x_rate_s > 0.60
        and sample.normalized_y_rate_down_s > 0.60
        and abs(
            sample.normalized_x
            + sample.normalized_x_rate_s * 0.10
        )
        <= 0.20
        and abs(
            sample.normalized_y_down
            + sample.normalized_y_rate_down_s * 0.10
        )
        <= 0.28
        for sample in evidence.samples
    )

    facts = _latest_censored_kwargs(latch)
    facts.update(
        clipping=FrameEdge.NONE,
        center_censored=False,
        normalized_x=-0.10,
        normalized_y_down=-0.07,
        normalized_x_rate_s=1.00,
        normalized_y_rate_down_s=0.70,
    )

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.COAST
    )


def test_projected_divergence_does_not_count_or_erase_safe_evidence():
    evidence = NearPlaneEvidence()
    latch = None
    for sample in _LATEST_NEAR_PLANE[:2]:
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
        assert latch is None
    assert len(evidence.samples) == 2

    projected_outside = replace(
        _LATEST_NEAR_PLANE[2],
        normalized_x=0.15,
        normalized_y_down=-0.07,
        normalized_x_rate_s=1.00,
        normalized_y_rate_down_s=0.70,
    )
    assert projected_outside.normalized_x_rate_s > 0.60
    assert (
        projected_outside.normalized_x
        + projected_outside.normalized_x_rate_s * 0.10
        > 0.20
    )

    evidence, latch = advance_near_plane_evidence(
        evidence,
        projected_outside,
        required_corridor_frames=_REQUIRED_FRAMES,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert evidence.samples == _LATEST_NEAR_PLANE[:2]
    assert latch is None

    _evidence, safe_latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(safe_latch)
    facts.update(
        clipping=FrameEdge.NONE,
        center_censored=False,
        normalized_x=projected_outside.normalized_x,
        normalized_y_down=projected_outside.normalized_y_down,
        normalized_x_rate_s=projected_outside.normalized_x_rate_s,
        normalized_y_rate_down_s=projected_outside.normalized_y_rate_down_s,
    )

    assert (
        classify_latched_measurement(safe_latch, **facts)
        is LatchedMeasurementMode.COAST
    )


def test_run_8c5e_transients_do_not_require_exact_adjacent_latch_frames():
    rows = (
        (-0.18125, -0.105556, -0.0213, 0.3203, 0.4218, 1.225),
        (-0.1875, -0.088889, -0.1095, 0.4105, 0.4458, 1.435),
        (-0.190625, -0.066667, -0.0989367, 0.538, 0.4739, 1.618),
        (-0.103125, -0.055556, 1.6805, 0.4611, 0.5462, 3.527),
        (-0.103125, -0.038889, 0.7562, 0.4723, 0.5767, 2.451),
        (-0.1125, -0.022222, 0.1909, 0.4782, 0.6074, 1.930),
    )
    samples = tuple(
        _sample(
            frame_id=3_244_380 + index,
            publication=157 + index,
            observation_ns=1_000_000_000 + index * 33_000_000,
            publication_ns=1_001_000_000 + index * 33_000_000,
            wire_start_ns=1_020_000_000 + index * 33_000_000,
            wire_return_ns=1_021_000_000 + index * 33_000_000,
            x=row[0],
            y=row[1],
            x_rate=row[2],
            y_rate=row[3],
            apparent_scale=row[4],
            log_scale_rate=row[5],
            confidence=0.95,
            association_confidence=0.85,
            command=(0.0, 0.035, 0.15, 0.25),
        )
        for index, row in enumerate(rows)
    )
    assert abs(
        samples[2].normalized_x
        + samples[2].normalized_x_rate_s * 0.10
    ) > 0.20
    assert samples[3].log_scale_rate_s > 2.0
    assert samples[4].log_scale_rate_s > 2.0

    evidence = NearPlaneEvidence()
    latch = None
    retained_counts = []
    continuity_publications = []
    for sample in samples:
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
        retained_counts.append(len(evidence.samples))
        continuity_publications.append(
            None
            if evidence.last_observed_sample is None
            else (
                evidence.last_observed_sample
                .camera_token.publication_sequence
            )
        )

    assert retained_counts == [1, 2, 2, 2, 2, 3]
    assert continuity_publications == [157, 158, 159, 160, 161, 162]
    assert latch is not None
    assert [
        sample.camera_token.publication_sequence
        for sample in latch.evidence.samples
    ] == [157, 158, 162]

    evidence = NearPlaneEvidence()
    for sample in samples[:3]:
        evidence, _latch = advance_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
    regressed_scale = replace(
        samples[3],
        log_scale=samples[2].log_scale - 0.01,
    )
    evidence, latch = advance_near_plane_evidence(
        evidence,
        regressed_scale,
        required_corridor_frames=_REQUIRED_FRAMES,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )
    assert evidence.samples == ()
    assert latch is None


def test_run_14e9_uses_noncounting_close_scale_anchor():
    qualified = tuple(
        replace(
            sample,
            log_scale=math.log(scale),
        )
        for sample, scale in zip(
            _LATEST_NEAR_PLANE,
            (0.3791, 0.3999, 0.4176),
            strict=True,
        )
    )
    evidence = NearPlaneEvidence()
    latch = None
    for sample in qualified:
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
    assert len(evidence.samples) == 3
    assert latch is None

    previous = qualified[-1]
    anchor_token = _token(3_266_127, 160)
    anchor = replace(
        previous,
        camera_token=anchor_token,
        wire_camera_token=anchor_token,
        observation_monotonic_ns=(
            previous.observation_monotonic_ns + 33_000_000
        ),
        publication_monotonic_ns=(
            previous.publication_monotonic_ns + 33_000_000
        ),
        wire_start_monotonic_ns=(
            previous.wire_start_monotonic_ns + 33_000_000
        ),
        wire_return_monotonic_ns=(
            previous.wire_return_monotonic_ns + 33_000_000
        ),
        normalized_x=-0.20,
        normalized_x_rate_s=-0.1373,
        log_scale=math.log(0.4905),
        log_scale_rate_s=1.578,
        command_yaw_rate=0.15,
    )
    assert abs(
        anchor.normalized_x
        + anchor.normalized_x_rate_s * 0.10
    ) > 0.20

    evidence, latch = advance_near_plane_evidence(
        evidence,
        anchor,
        required_corridor_frames=_REQUIRED_FRAMES,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert latch is not None
    assert latch.evidence.samples == qualified
    assert latch.anchor_sample == anchor
    assert latch.anchor_camera_token == anchor_token
    assert latch.accepted_command == anchor.command


@pytest.mark.parametrize(
    "updates",
    (
        {"ambiguous": True},
        {"track_role": VisualTrackRole.AMBIGUOUS},
        {"current_gate_index": 1},
        {"current_track_id": "vq2-track-different"},
        {"track_authoritative_gate_index": 1},
    ),
)
def test_ambiguous_or_wrong_authority_is_unsafe(updates):
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(latch)
    facts.update(updates)

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.UNSAFE
    )


@pytest.mark.parametrize(
    "updates",
    (
        {"confidence": 0.19},
        {"association_confidence": 0.09},
        {"apparent_scale": 0.40},
    ),
)
def test_low_confidence_or_fragmentation_cuts_to_credit_wait(updates):
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(latch)
    facts.update(updates)

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.CREDIT_WAIT
    )


def test_credited_one_edge_then_two_edges_then_full_censor_is_generic():
    _evidence, latch = _advance(_CREDITED_NEAR_PLANE)
    common = {
        "current_gate_index": 0,
        "current_track_id": _TRACK,
        "track_role": VisualTrackRole.CURRENT,
        "track_authoritative_gate_index": 0,
        "visible": True,
        "missed_frame_count": 0,
        "ambiguous": False,
        "center_censored": True,
        "min_track_confidence": _MIN_TRACK_CONFIDENCE,
        "min_association_confidence": _MIN_ASSOCIATION_CONFIDENCE,
    }
    bottom = dict(
        common,
        previous_camera_token=latch.anchor_camera_token,
        camera_token=_token(659087, 165),
        track_latest_camera_token=_token(659087, 165),
        clipping=FrameEdge.BOTTOM,
        normalized_x=0.0,
        normalized_y_down=0.022222222222222143,
        normalized_x_rate_s=-0.31476596612987606,
        normalized_y_rate_down_s=0.23894140672499853,
        apparent_scale=0.7757305024942618,
        confidence=0.9595058777998756,
        association_confidence=0.82034291642588,
    )
    top_bottom = dict(
        common,
        previous_camera_token=_token(659087, 165),
        camera_token=_token(659088, 166),
        track_latest_camera_token=_token(659088, 166),
        clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
        normalized_x=0.0031250000000000444,
        normalized_y_down=0.0,
        normalized_x_rate_s=-0.09244913046229414,
        normalized_y_rate_down_s=-0.24231141974636716,
        apparent_scale=0.835725732522339,
        confidence=0.9417915725865457,
        association_confidence=0.8661413767327067,
    )
    full = dict(
        common,
        previous_camera_token=_token(659088, 166),
        camera_token=_token(659091, 169),
        track_latest_camera_token=_token(659091, 169),
        clipping=(
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM
        ),
        normalized_x=0.0,
        normalized_y_down=0.0,
        normalized_x_rate_s=-0.13160123559635845,
        normalized_y_rate_down_s=-0.0220806281243877,
        apparent_scale=1.0,
        confidence=0.8073908824001383,
        association_confidence=0.8901329788113191,
    )

    assert (
        classify_latched_measurement(latch, **bottom)
        is LatchedMeasurementMode.COAST
    )
    assert (
        classify_latched_measurement(latch, **top_bottom)
        is LatchedMeasurementMode.COAST
    )
    assert (
        classify_latched_measurement(latch, **full)
        is LatchedMeasurementMode.CREDIT_WAIT
    )


def test_primary_credited_target_loss_cuts_to_credit_wait():
    _evidence, latch = _advance(_CREDITED_NEAR_PLANE)

    mode = classify_latched_measurement(
        latch,
        previous_camera_token=_token(659093, 171),
        camera_token=_token(659094, 172),
        current_gate_index=0,
        current_track_id=_TRACK,
        track_latest_camera_token=_token(659093, 171),
        track_role=VisualTrackRole.CURRENT,
        track_authoritative_gate_index=0,
        visible=False,
        missed_frame_count=1,
        ambiguous=False,
        clipping=(
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM
        ),
        center_censored=True,
        normalized_x=None,
        normalized_y_down=None,
        normalized_x_rate_s=None,
        normalized_y_rate_down_s=None,
        apparent_scale=None,
        confidence=None,
        association_confidence=None,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert mode is LatchedMeasurementMode.CREDIT_WAIT


@pytest.mark.parametrize(
    "previous,current",
    (
        (_token(860511, 159), _token(860511, 159)),
        (_token(860518, 166), _token(860517, 165)),
        (
            _token(860511, 159),
            _token(860518, 166, generation=2),
        ),
    ),
)
def test_stale_or_cross_epoch_lineage_is_unsafe(previous, current):
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(latch)
    facts.update(
        previous_camera_token=previous,
        camera_token=current,
        track_latest_camera_token=current,
    )

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.UNSAFE
    )


@pytest.mark.parametrize(
    ("publication", "clipping", "expected"),
    (
        (180, FrameEdge.NONE, PostCreditMeasurementMode.CLEAN),
        (183, FrameEdge.TOP, PostCreditMeasurementMode.ONE_EDGE_CENSORED),
        (
            184,
            FrameEdge.TOP | FrameEdge.RIGHT,
            PostCreditMeasurementMode.REACQUIRE,
        ),
    ),
)
def test_post_credit_measurement_modes_are_gate_generic(
    publication,
    clipping,
    expected,
):
    assert (
        classify_post_credit_measurement(
            _post_credit_snapshot(publication, clipping=clipping),
            gate_index=1,
            track_id=_TRACK,
            previous_camera_token=_token(
                1_500_000 + publication - 1,
                publication - 1,
            ),
            last_track_token=_token(
                1_500_000 + publication - 1,
                publication - 1,
            ),
        )
        is expected
    )


def test_post_credit_retained_loss_rejects_regressed_track_token():
    previous = _token(1_500_183, 183)
    retained = _post_credit_snapshot(
        184,
        visible=False,
        clipping=FrameEdge.TOP,
        latest_track_publication=183,
    )
    skipped_camera = _post_credit_snapshot(
        185,
        visible=False,
        clipping=FrameEdge.TOP,
        latest_track_publication=184,
    )
    regressed = _post_credit_snapshot(
        184,
        visible=False,
        clipping=FrameEdge.TOP,
        latest_track_publication=182,
    )

    assert (
        classify_post_credit_measurement(
            retained,
            gate_index=1,
            track_id=_TRACK,
            previous_camera_token=previous,
            last_track_token=previous,
        )
        is PostCreditMeasurementMode.REACQUIRE
    )
    assert (
        classify_post_credit_measurement(
            skipped_camera,
            gate_index=1,
            track_id=_TRACK,
            previous_camera_token=previous,
            last_track_token=previous,
        )
        is PostCreditMeasurementMode.REACQUIRE
    )
    assert (
        classify_post_credit_measurement(
            regressed,
            gate_index=1,
            track_id=_TRACK,
            previous_camera_token=previous,
            last_track_token=previous,
        )
        is PostCreditMeasurementMode.UNSAFE
    )


def test_8853bd30_retained_ambiguous_loss_uses_bounded_reacquire() -> None:
    track_id = "vq2-track-000002"
    previous = _token(538260, 181)
    snapshot = _post_credit_snapshot(
        182,
        visible=False,
        clipping=FrameEdge.TOP | FrameEdge.RIGHT,
        latest_track_publication=181,
    )
    snapshot.latest_camera_token = _token(538261, 182)
    snapshot.current_track_id = track_id
    snapshot.current_track.track_id = track_id
    snapshot.current_track.latest_token = previous
    snapshot.current_track.center_norm = (
        0.7906249999999999,
        -0.8722222222222222,
    )
    snapshot.current_track.role = VisualTrackRole.AMBIGUOUS
    snapshot.current_track.ambiguous = True

    assert (
        classify_post_credit_measurement(
            snapshot,
            gate_index=1,
            track_id=track_id,
            previous_camera_token=previous,
            last_track_token=previous,
        )
        is PostCreditMeasurementMode.REACQUIRE
    )

    snapshot.current_track.visible = True
    snapshot.current_track.missed_frame_count = 0
    snapshot.current_track.latest_token = snapshot.latest_camera_token
    assert (
        classify_post_credit_measurement(
            snapshot,
            gate_index=1,
            track_id=track_id,
            previous_camera_token=previous,
            last_track_token=previous,
        )
        is PostCreditMeasurementMode.UNSAFE
    )


@pytest.mark.parametrize(
    "mutation",
    ("ambiguous", "wrong_gate", "stale_camera", "outside_image"),
)
def test_post_credit_unsafe_authority_fails_closed(mutation):
    snapshot = _post_credit_snapshot(180)
    previous = _token(1_500_179, 179)
    if mutation == "ambiguous":
        snapshot.current_track.ambiguous = True
    elif mutation == "wrong_gate":
        snapshot.current_track.authoritative_gate_index = 2
    elif mutation == "outside_image":
        snapshot.current_track.clipping = FrameEdge.TOP
        snapshot.current_track.center_censored = True
        snapshot.current_track.center_norm = (1.01, -0.70)
    else:
        previous = snapshot.latest_camera_token

    assert (
        classify_post_credit_measurement(
            snapshot,
            gate_index=1,
            track_id=_TRACK,
            previous_camera_token=previous,
            last_track_token=previous,
        )
        is PostCreditMeasurementMode.UNSAFE
    )


def test_duplicate_wire_publication_does_not_count_twice():
    evidence, latch = advance_near_plane_evidence(
        NearPlaneEvidence(),
        _LATEST_NEAR_PLANE[0],
        required_corridor_frames=_REQUIRED_FRAMES,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )
    assert latch is None

    evidence, latch = advance_near_plane_evidence(
        evidence,
        _LATEST_NEAR_PLANE[0],
        required_corridor_frames=_REQUIRED_FRAMES,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert evidence.samples == ()
    assert latch is None


@pytest.mark.parametrize(
    "changed",
    (
        {"clipping": FrameEdge.BOTTOM, "center_censored": True},
        {"ambiguous": True},
        {"confidence": 0.19},
        {"normalized_x": 0.21},
    ),
)
def test_hard_unsafe_wire_fact_clears_prior_latch_evidence(changed):
    evidence = NearPlaneEvidence(samples=_LATEST_NEAR_PLANE[:2])
    sample = replace(_LATEST_NEAR_PLANE[2], **changed)

    evidence, latch = advance_near_plane_evidence(
        evidence,
        sample,
        required_corridor_frames=_REQUIRED_FRAMES,
        crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
        min_track_confidence=_MIN_TRACK_CONFIDENCE,
        min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
    )

    assert evidence.samples == ()
    assert latch is None


def test_generic_adjacent_gate_uses_identical_lifecycle():
    gate_one_samples = tuple(
        replace(
            sample,
            gate_index=1,
            wire_race_gate_index=1,
            track_id="vq2-track-000002",
        )
        for sample in _LATEST_NEAR_PLANE
    )

    _evidence, latch = _advance_with_gate(gate_one_samples)

    assert latch.gate_index == 1
    assert latch.track_id == "vq2-track-000002"
    assert latch.lifecycle is CourseLifecycle.NEAR_PLANE_LATCHED


def _advance_with_gate(
    samples: tuple[NearPlaneWireSample, ...],
) -> tuple[NearPlaneEvidence, NearPlaneLatch]:
    evidence = NearPlaneEvidence()
    latch = None
    for sample in samples:
        evidence, latch = advance_near_plane_evidence(
            evidence,
            sample,
            required_corridor_frames=_REQUIRED_FRAMES,
            crossing_min_log_scale=_CROSSING_MIN_LOG_SCALE,
            min_track_confidence=_MIN_TRACK_CONFIDENCE,
            min_association_confidence=_MIN_ASSOCIATION_CONFIDENCE,
        )
    assert latch is not None
    return evidence, latch


def test_wire_fact_requires_the_same_pinned_publication():
    sample = _LATEST_NEAR_PLANE[0]

    with pytest.raises(
        ValueError,
        match="observation and accepted wire tokens differ",
    ):
        replace(sample, wire_camera_token=_token(860510, 158))
    with pytest.raises(ValueError, match="did not pin"):
        replace(
            sample,
            publication_pinned_through_transport_return=False,
        )
