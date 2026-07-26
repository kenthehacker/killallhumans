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
    LatchedMeasurementMode,
    NearPlaneEvidence,
    NearPlaneLatch,
    NearPlaneWireSample,
    PostCreditMeasurementMode,
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


def test_observable_axis_projection_divergence_is_unsafe():
    _evidence, latch = _advance(_LATEST_NEAR_PLANE)
    facts = _latest_censored_kwargs(latch)
    facts.update(
        clipping=FrameEdge.NONE,
        center_censored=False,
        normalized_x=0.19,
        normalized_x_rate_s=0.20,
    )

    assert (
        classify_latched_measurement(latch, **facts)
        is LatchedMeasurementMode.UNSAFE
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


def test_high_raw_center_rate_outside_projected_corridor_resets_and_is_unsafe():
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

    assert evidence.samples == ()
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
        is LatchedMeasurementMode.UNSAFE
    )


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
        {"log_scale_rate_s": 2.01},
        {"normalized_x": 0.21},
    ),
)
def test_unusable_clean_wire_fact_cannot_advance_latch(changed):
    sample = replace(_LATEST_NEAR_PLANE[0], **changed)

    evidence, latch = advance_near_plane_evidence(
        NearPlaneEvidence(),
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
