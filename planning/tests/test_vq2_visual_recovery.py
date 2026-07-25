"""Focused tests for transition-anchor post-promotion recovery admission."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualTrack,
    VisualTrackRole,
    VisualTrackSample,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateTransition,
)
from planning.vq2_visual_alignment import (
    VisualAlignmentRefusal,
    require_visual_alignment_entry,
)
from planning.vq2_visual_recovery import (
    RECOVERY_HARD_DURATION_S,
    RECOVERY_MAX_ANCHOR_CREDIT_AGE_S,
    RECOVERY_MAX_CONTINUATION_AGE_S,
    RECOVERY_MAX_PROJECTED_ABS_Y_NORM,
    RECOVERY_MAX_RAW_CENTER_RATE_NORM_S,
    RecoveryContinuationAdmission,
    TransitionRecoveryAdmission,
    VisualRecoveryRefusal,
    require_recovery_continuation,
    require_transition_recovery_admission,
)
from planning.vq2_visual_servo import VisualTarget
from planning.vq2_visual_alignment import (
    POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM,
)


# Exact compact excerpt from
# 20260724T224756Z-visual-align-bd25a045/session.jsonl.gz
# (trace sha256 14c2c65cbd5f64cda3a67a26e35636684bd327f3eb88d73d937a00565b99cd19).
_ROWS = (
    (
        1974827,
        168,
        66_129_481_540_600,
        66_129_482_509_500,
        (0.5, -0.6166666666666667),
        (0.6890625, 0.08888888888888889, 0.8125, 0.2972222222222222),
        0.1603625449827151,
        0.7368920473689649,
        0.9317637477940568,
    ),
    (
        1974828,
        169,
        66_129_508_638_800,
        66_129_509_835_100,
        (0.5093749999999999, -0.6166666666666667),
        (0.6921875, 0.08611111111111111, 0.8171875, 0.2972222222222222),
        0.16244657241348273,
        0.7484904213160342,
        0.9390432596316921,
    ),
    (
        1974829,
        170,
        66_129_544_587_600,
        66_129_545_498_300,
        (0.51875, -0.6277777777777778),
        (0.6953125, 0.07777777777777778, 0.825, 0.29444444444444445),
        0.16762743908242864,
        0.7561404038779297,
        0.9284982404586473,
    ),
    (
        1974830,
        171,
        66_129_578_314_400,
        66_129_579_366_000,
        (0.53125, -0.6333333333333333),
        (0.6984375, 0.07222222222222222, 0.8328125, 0.29444444444444445),
        0.1728036779443976,
        0.7734291589391382,
        0.9328634141534373,
    ),
    (
        1974831,
        172,
        66_129_613_164_100,
        66_129_614_288_300,
        (0.5406249999999999, -0.6444444444444444),
        (0.7015625, 0.06388888888888888, 0.840625, 0.2916666666666667),
        0.1779756927847795,
        0.7673048622633529,
        0.9347801148583332,
    ),
)
_RACE_RECEIVED_NS = 66_129_618_666_500


def _fixture() -> tuple[
    VisualTrack,
    ConfirmedGateTransition,
]:
    samples = tuple(
        VisualTrackSample(
            tracker_frame_sequence=100 + index,
            token=CameraFrameToken(
                generation=1,
                frame_id=row[0],
                publication_sequence=row[1],
                stream_id="vq2-camera-udp-5600",
            ),
            observation_monotonic_ns=row[2],
            publication_monotonic_ns=row[3],
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            camera_source_time_ns=1_784_933_283_805_814_400 + index,
            source_index=0,
            center_norm=row[4],
            bbox_norm=row[5],
            apparent_scale=row[6],
            confidence=row[7],
            clipping=FrameEdge.NONE,
            center_censored=False,
            association_confidence=row[8],
        )
        for index, row in enumerate(_ROWS)
    )
    race = AuthoritativeRaceStatusRef.live(
        session_id=(
            "41b0b7e1a0ed8a2f8646b9af1b0c8c8e0c17da19ae29c1abe17462a9653686a3"
        ),
        reset_epoch=1,
        race_generation=2,
        race_status_sequence=1476,
        race_status_boot_ms=6063,
        active_gate_index=1,
        received_monotonic_ns=_RACE_RECEIVED_NS,
        host_clock_id="host-perf-counter",
    )
    track = VisualTrack(
        track_id="vq2-track-000002",
        first_token=samples[0].token,
        latest_token=samples[-1].token,
        center_norm=samples[-1].center_norm,
        bbox_norm=samples[-1].bbox_norm,
        apparent_scale=samples[-1].apparent_scale,
        center_velocity_norm_s=(
            0.2953815824130328,
            -0.2605013862598165,
        ),
        log_scale_rate_s=0.8368765126029618,
        confidence=samples[-1].confidence,
        association_confidence=samples[-1].association_confidence,
        consecutive_frame_count=len(samples),
        total_observation_count=len(samples),
        missed_frame_count=0,
        clipping=FrameEdge.NONE,
        center_censored=False,
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=1,
        authority_race_status_sequence=1476,
        authority_race_status_boot_ms=6063,
        ambiguous=False,
        visible=True,
        history=samples,
    )
    transition = ConfirmedGateTransition(
        from_gate_index=0,
        to_gate_index=1,
        retired_track_id="vq2-track-000001",
        promoted_track_id=track.track_id,
        race_status=race,
        camera_token_at_credit=track.latest_token,
        promoted_first_token=track.first_token,
        promoted_latest_token_before_credit=track.latest_token,
        pretransition_frame_tokens=tuple(
            sample.token for sample in samples
        ),
        history_length_before_promotion=len(samples),
        history_length_after_promotion=len(samples),
    )
    return track, transition


def _admit(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
) -> TransitionRecoveryAdmission:
    return require_transition_recovery_admission(
        track,
        transition,
        tracker_time_basis_id="host-perf-counter",
        measured_pitch_rad=-0.04001,
        now_monotonic_ns=_RACE_RECEIVED_NS + 1_000_000,
    )


def _continued_fixture(
    *,
    center_norm: tuple[float, float] = (0.53, -0.63),
    center_velocity_norm_s: tuple[float, float] = (-0.20, 0.20),
    apparent_scale: float = 0.18,
    log_scale_rate_s: float = 0.30,
) -> tuple[
    VisualTrack,
    ConfirmedGateTransition,
    CameraFrameToken,
    int,
    int,
]:
    anchor, transition = _fixture()
    previous = anchor.history[-1]
    assert previous.publication_monotonic_ns is not None
    delta_x_image = 0.5 * (center_norm[0] - previous.center_norm[0])
    delta_y_image = 0.5 * (center_norm[1] - previous.center_norm[1])
    left, top, right, bottom = previous.bbox_norm
    observation_ns = previous.observation_monotonic_ns + 35_000_000
    publication_ns = observation_ns + 1_000_000
    latest = replace(
        previous,
        tracker_frame_sequence=previous.tracker_frame_sequence + 1,
        token=replace(
            previous.token,
            frame_id=previous.token.frame_id + 1,
            publication_sequence=(
                previous.token.publication_sequence + 1
            ),
        ),
        observation_monotonic_ns=observation_ns,
        publication_monotonic_ns=publication_ns,
        camera_source_time_ns=(
            None
            if previous.camera_source_time_ns is None
            else previous.camera_source_time_ns + 35_000_000
        ),
        center_norm=center_norm,
        bbox_norm=(
            left + delta_x_image,
            top + delta_y_image,
            right + delta_x_image,
            bottom + delta_y_image,
        ),
        apparent_scale=apparent_scale,
        confidence=0.80,
        association_confidence=0.95,
    )
    history = anchor.history + (latest,)
    track = replace(
        anchor,
        latest_token=latest.token,
        center_norm=latest.center_norm,
        bbox_norm=latest.bbox_norm,
        apparent_scale=latest.apparent_scale,
        center_velocity_norm_s=center_velocity_norm_s,
        log_scale_rate_s=log_scale_rate_s,
        confidence=latest.confidence,
        association_confidence=latest.association_confidence,
        consecutive_frame_count=len(history),
        total_observation_count=len(history),
        history=history,
    )
    recovery_started_ns = _RACE_RECEIVED_NS + 1_000_000
    now_ns = publication_ns + 1_000_000
    return (
        track,
        transition,
        previous.token,
        recovery_started_ns,
        now_ns,
    )


def _exact_failed_trace_continuation_fixture():
    """Append exact live token 173 to the authoritative token-172 anchor."""

    anchor, transition = _fixture()
    previous = anchor.history[-1]
    latest = VisualTrackSample(
        tracker_frame_sequence=previous.tracker_frame_sequence + 1,
        token=CameraFrameToken(
            generation=previous.token.generation,
            frame_id=1_974_832,
            publication_sequence=173,
            stream_id=previous.token.stream_id,
        ),
        observation_monotonic_ns=66_129_647_607_200,
        publication_monotonic_ns=66_129_648_481_400,
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        camera_source_time_ns=(
            None
            if previous.camera_source_time_ns is None
            else previous.camera_source_time_ns + 1
        ),
        source_index=0,
        center_norm=(0.553125, -0.6611111111111111),
        bbox_norm=(
            451.0 / 640.0,
            19.0 / 360.0,
            543.0 / 640.0,
            104.0 / 360.0,
        ),
        apparent_scale=math.sqrt((92.0 * 85.0) / (640.0 * 360.0)),
        confidence=0.769348,
        clipping=FrameEdge.NONE,
        center_censored=False,
        association_confidence=0.928976,
    )
    history = anchor.history + (latest,)
    track = replace(
        anchor,
        latest_token=latest.token,
        center_norm=latest.center_norm,
        bbox_norm=latest.bbox_norm,
        apparent_scale=latest.apparent_scale,
        center_velocity_norm_s=(
            0.3325263,
            -0.3833650,
        ),
        log_scale_rate_s=0.9281754,
        confidence=latest.confidence,
        association_confidence=latest.association_confidence,
        consecutive_frame_count=len(history),
        total_observation_count=len(history),
        history=history,
    )
    return (
        track,
        transition,
        previous.token,
        _RACE_RECEIVED_NS + 1_000_000,
        latest.publication_monotonic_ns + 1_000_000,
    )


def _continue(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
    previous_token: CameraFrameToken,
    recovery_started_ns: int,
    now_ns: int,
) -> RecoveryContinuationAdmission:
    return require_recovery_continuation(
        track,
        transition,
        previous_token=previous_token,
        tracker_time_basis_id="host-perf-counter",
        measured_pitch_rad=-0.04001,
        recovery_started_monotonic_ns=recovery_started_ns,
        now_monotonic_ns=now_ns,
    )


def test_exact_latest_transition_anchor_admits_only_predictive_recovery():
    track, transition = _fixture()
    target = VisualTarget.from_visual_track(
        track,
        expected_gate_index=1,
    )
    with pytest.raises(VisualAlignmentRefusal, match="horizontal motion"):
        require_visual_alignment_entry(
            target,
            measured_pitch_rad=-0.04001,
        )

    admission = _admit(track, transition)

    assert type(admission) is TransitionRecoveryAdmission
    assert admission.track_id == "vq2-track-000002"
    assert admission.anchor_token.publication_sequence == 172
    assert admission.anchor_credit_age_s == pytest.approx(0.0043782)
    assert admission.max_raw_horizontal_rate_s == pytest.approx(
        0.3706251408375522
    )
    assert admission.max_raw_vertical_rate_down_s == pytest.approx(
        0.31882946226541614
    )
    assert admission.projected_abs_horizontal_error < 0.67
    assert admission.projected_abs_vertical_error_image_down < 0.71
    assert admission.projected_bbox_norm_ltrb[1] > 6.0 / 360.0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (
            lambda track, transition: (
                replace(
                    track,
                    association_confidence=0.89,
                    history=track.history[:-1]
                    + (
                        replace(
                            track.history[-1],
                            association_confidence=0.89,
                        ),
                    ),
                ),
                transition,
            ),
            "confidence is insufficient",
        ),
        (
            lambda track, transition: (
                replace(
                    track,
                    clipping=FrameEdge.TOP,
                    history=track.history[:-1]
                    + (
                        replace(
                            track.history[-1],
                            clipping=FrameEdge.TOP,
                        ),
                    ),
                ),
                transition,
            ),
            "ambiguous or censored",
        ),
        (
            lambda track, transition: (
                track,
                replace(
                    transition,
                    camera_token_at_credit=track.history[-2].token,
                ),
            ),
            "transition identity is inconsistent",
        ),
        (
            lambda track, transition: (
                replace(
                    track,
                    history=track.history[:2]
                    + (
                        replace(
                            track.history[2],
                            token=replace(
                                track.history[2].token,
                                publication_sequence=171,
                            ),
                        ),
                    )
                    + track.history[3:],
                ),
                transition,
            ),
            "not bound to the transition",
        ),
    ),
)
def test_recovery_refuses_provenance_identity_and_censoring_faults(
    mutate,
    reason,
):
    track, transition = _fixture()
    track, transition = mutate(track, transition)
    with pytest.raises(VisualRecoveryRefusal, match=reason):
        _admit(track, transition)


def test_recovery_refuses_stale_credit_and_raw_motion_hidden_by_filter():
    track, transition = _fixture()
    stale_race = replace(
        transition.race_status,
        received_monotonic_ns=(
            track.history[-1].publication_monotonic_ns
            + round(
                (RECOVERY_MAX_ANCHOR_CREDIT_AGE_S + 0.001)
                * 1_000_000_000
            )
        ),
    )
    stale = replace(transition, race_status=stale_race)
    with pytest.raises(VisualRecoveryRefusal, match="stale at race credit"):
        require_transition_recovery_admission(
            track,
            stale,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=stale_race.received_monotonic_ns,
        )

    previous = track.history[-2]
    latest = track.history[-1]
    assert latest.publication_monotonic_ns is not None
    assert previous.publication_monotonic_ns is not None
    dt_s = (
        latest.publication_monotonic_ns
        - previous.publication_monotonic_ns
    ) / 1_000_000_000.0
    unsafe_latest = replace(
        latest,
        center_norm=(
            previous.center_norm[0]
            + (RECOVERY_MAX_RAW_CENTER_RATE_NORM_S + 0.01) * dt_s,
            latest.center_norm[1],
        ),
    )
    unsafe_track = replace(
        track,
        center_norm=unsafe_latest.center_norm,
        history=track.history[:-1] + (unsafe_latest,),
    )
    with pytest.raises(VisualRecoveryRefusal, match="raw center motion"):
        _admit(unsafe_track, transition)


def test_continuation_admits_exact_next_postcredit_publication():
    fixture = _continued_fixture()
    track, transition, previous, started_ns, now_ns = fixture

    admission = _continue(*fixture)

    assert type(admission) is RecoveryContinuationAdmission
    assert admission.track_id == track.track_id
    assert admission.previous_token == previous
    assert admission.frame_token == track.latest_token
    assert admission.frame_token.publication_sequence == (
        previous.publication_sequence + 1
    )
    assert admission.capture.track_id == track.track_id
    assert admission.capture.frame_token.publication_sequence == 173
    assert admission.capture.horizontal_error == pytest.approx(0.53)
    assert admission.capture.vertical_error_image_down == pytest.approx(-0.63)
    assert admission.observation_age_s == pytest.approx(
        (
            now_ns - track.history[-1].observation_monotonic_ns
        )
        / 1_000_000_000.0
    )
    assert admission.recovery_elapsed_s == pytest.approx(
        (now_ns - started_ns) / 1_000_000_000.0
    )
    assert admission.max_raw_horizontal_rate_s > abs(
        track.center_velocity_norm_s[0]
    )
    assert admission.max_raw_vertical_rate_down_s > abs(
        track.center_velocity_norm_s[1]
    )
    assert admission.projected_abs_horizontal_error == pytest.approx(
        abs(track.center_norm[0])
        + admission.max_raw_horizontal_rate_s
        * admission.projection_horizon_s
        + 4.0 / 640.0
    )
    assert admission.projected_abs_vertical_error_image_down == pytest.approx(
        abs(track.center_norm[1])
        + admission.max_raw_vertical_rate_down_s
        * admission.projection_horizon_s
        + 4.0 / 360.0
    )


def test_exact_failed_trace_token_173_has_only_narrow_recovery_margin():
    fixture = _exact_failed_trace_continuation_fixture()

    admission = _continue(*fixture)

    assert admission.frame_token.publication_sequence == 173
    assert admission.projection_horizon_s == pytest.approx(0.080)
    assert admission.max_raw_vertical_rate_down_s == pytest.approx(
        0.48389,
        rel=1e-5,
    )
    assert (
        admission.projected_abs_vertical_error_image_down
        > POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM
    )
    remaining_margin = (
        RECOVERY_MAX_PROJECTED_ABS_Y_NORM
        - admission.projected_abs_vertical_error_image_down
    )
    assert 0.0 < remaining_margin < 0.005
    assert admission.projected_bbox_norm_ltrb[1] > 6.0 / 360.0


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("skip_publication", "publication did not advance exactly"),
        ("wrong_previous", "publication did not advance exactly"),
        ("new_generation", "publication did not advance exactly"),
    ),
)
def test_continuation_requires_exact_next_token_chain(mutation, reason):
    track, transition, previous, started_ns, now_ns = _continued_fixture()
    if mutation == "wrong_previous":
        previous = replace(previous, frame_id=previous.frame_id - 1)
    else:
        latest = track.history[-1]
        if mutation == "skip_publication":
            token = replace(
                latest.token,
                publication_sequence=(
                    latest.token.publication_sequence + 1
                ),
            )
        else:
            token = replace(
                latest.token,
                generation=latest.token.generation + 1,
            )
        latest = replace(latest, token=token)
        track = replace(
            track,
            latest_token=token,
            history=track.history[:-1] + (latest,),
        )

    with pytest.raises(VisualRecoveryRefusal, match=reason):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            now_ns,
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (
            lambda track, transition: (
                track,
                replace(
                    transition,
                    promoted_track_id="vq2-track-wrong",
                ),
            ),
            "track authority disagrees",
        ),
        (
            lambda track, transition: (
                replace(track, authoritative_gate_index=2),
                transition,
            ),
            "track authority disagrees",
        ),
        (
            lambda track, transition: (
                replace(track, authority_race_status_sequence=1477),
                transition,
            ),
            "track authority disagrees",
        ),
        (
            lambda track, transition: (
                track,
                replace(
                    transition,
                    race_status=replace(
                        transition.race_status,
                        active_gate_index=2,
                    ),
                ),
            ),
            "unfinished adjacent transition",
        ),
        (
            lambda track, transition: (
                track,
                replace(transition, from_gate_index=1),
            ),
            "unfinished adjacent transition",
        ),
    ),
)
def test_continuation_binds_promoted_identity_to_authoritative_race(
    mutate,
    reason,
):
    track, transition, previous, started_ns, now_ns = _continued_fixture()
    track, transition = mutate(track, transition)

    with pytest.raises(VisualRecoveryRefusal, match=reason):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            now_ns,
        )


def test_continuation_rejects_precredit_stale_and_future_frames():
    track, transition, previous, started_ns, now_ns = _continued_fixture()
    latest = replace(
        track.history[-1],
        observation_monotonic_ns=_RACE_RECEIVED_NS,
        publication_monotonic_ns=_RACE_RECEIVED_NS + 1,
    )
    precredit = replace(
        track,
        history=track.history[:-1] + (latest,),
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="observation is not post-credit",
    ):
        _continue(
            precredit,
            transition,
            previous,
            started_ns,
            now_ns,
        )

    stale_now_ns = (
        track.history[-1].observation_monotonic_ns
        + round(
            (RECOVERY_MAX_CONTINUATION_AGE_S + 0.001)
            * 1_000_000_000
        )
    )
    with pytest.raises(VisualRecoveryRefusal, match="frame is stale"):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            stale_now_ns,
        )

    future_now_ns = track.history[-1].observation_monotonic_ns - 1
    with pytest.raises(VisualRecoveryRefusal, match="future-dated"):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            future_now_ns,
        )


@pytest.mark.parametrize(
    ("center_norm", "reason"),
    (
        ((0.6001, -0.63), "horizontal position is unsafe"),
        ((0.53, -0.6801), "vertical position is unsafe"),
    ),
)
def test_continuation_enforces_immutable_actual_center_caps(
    center_norm,
    reason,
):
    fixture = _continued_fixture(center_norm=center_norm)

    with pytest.raises(VisualRecoveryRefusal, match=reason):
        _continue(*fixture)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (
            lambda sample: replace(sample, confidence=0.64),
            "confidence is insufficient",
        ),
        (
            lambda sample: replace(
                sample,
                association_confidence=0.89,
            ),
            "confidence is insufficient",
        ),
        (
            lambda sample: replace(sample, clipping=FrameEdge.RIGHT),
            "clipped or censored",
        ),
    ),
)
def test_continuation_rejects_low_authority_and_clipped_history(
    mutate,
    reason,
):
    track, transition, previous, started_ns, now_ns = _continued_fixture()
    latest = mutate(track.history[-1])
    track = replace(
        track,
        confidence=latest.confidence,
        association_confidence=latest.association_confidence,
        clipping=latest.clipping,
        history=track.history[:-1] + (latest,),
    )

    with pytest.raises(VisualRecoveryRefusal, match=reason):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            now_ns,
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (
            lambda track: replace(
                track,
                center_norm=(math.nan, track.center_norm[1]),
                history=track.history[:-1]
                + (
                    replace(
                        track.history[-1],
                        center_norm=(
                            math.nan,
                            track.history[-1].center_norm[1],
                        ),
                    ),
                ),
            ),
            "fields must be finite",
        ),
        (
            lambda track: replace(
                track,
                bbox_norm=(
                    track.bbox_norm[0],
                    math.nan,
                    track.bbox_norm[2],
                    track.bbox_norm[3],
                ),
                history=track.history[:-1]
                + (
                    replace(
                        track.history[-1],
                        bbox_norm=(
                            track.history[-1].bbox_norm[0],
                            math.nan,
                            track.history[-1].bbox_norm[2],
                            track.history[-1].bbox_norm[3],
                        ),
                    ),
                ),
            ),
            "bbox coordinate must be finite",
        ),
        (
            lambda track: replace(
                track,
                apparent_scale=math.nan,
                history=track.history[:-1]
                + (
                    replace(
                        track.history[-1],
                        apparent_scale=math.nan,
                    ),
                ),
            ),
            "apparent scale is invalid",
        ),
        (
            lambda track: replace(
                track,
                history=track.history[:-1] + (object(),),
            ),
            "track structure lacks current authority",
        ),
        (
            lambda track: replace(
                track,
                history=track.history[:-1]
                + (
                    replace(
                        track.history[-1],
                        tracker_frame_sequence=(
                            track.history[-2].tracker_frame_sequence + 2
                        ),
                    ),
                ),
            ),
            "not contiguous",
        ),
    ),
)
def test_continuation_rejects_malformed_or_nan_recent_history(
    mutate,
    reason,
):
    track, transition, previous, started_ns, now_ns = _continued_fixture()
    track = mutate(track)

    with pytest.raises(VisualRecoveryRefusal, match=reason):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            now_ns,
        )


def test_continuation_projection_uses_worse_raw_or_filtered_motion():
    previous_track, transition = _fixture()
    previous = previous_track.history[-1]
    assert previous.publication_monotonic_ns is not None
    dt_s = 0.035
    raw_rate = 0.49
    center = (
        previous.center_norm[0] + raw_rate * dt_s,
        -0.63,
    )
    fixture = _continued_fixture(
        center_norm=center,
        center_velocity_norm_s=(0.0, 0.0),
        log_scale_rate_s=0.0,
    )
    admission = _continue(*fixture)
    assert admission.max_raw_horizontal_rate_s == pytest.approx(raw_rate)
    assert admission.projected_abs_horizontal_error == pytest.approx(
        abs(center[0])
        + raw_rate * admission.projection_horizon_s
        + 4.0 / 640.0
    )

    track, transition, previous, started_ns, now_ns = _continued_fixture(
        center_norm=(0.54, -0.63),
        center_velocity_norm_s=(0.40, 0.0),
        log_scale_rate_s=0.0,
    )
    filtered_admission = _continue(
        track,
        transition,
        previous,
        started_ns,
        now_ns,
    )
    assert (
        abs(track.center_velocity_norm_s[0])
        > filtered_admission.max_raw_horizontal_rate_s
    )
    assert filtered_admission.projected_abs_horizontal_error == pytest.approx(
        abs(track.center_norm[0])
        + abs(track.center_velocity_norm_s[0])
        * filtered_admission.projection_horizon_s
        + 4.0 / 640.0
    )


def test_continuation_rejects_bbox_edge_loss_and_contour_deformation():
    track, transition, previous, started_ns, now_ns = _continued_fixture()
    edge_history = tuple(
        replace(
            sample,
            bbox_norm=(
                sample.bbox_norm[0],
                0.018,
                sample.bbox_norm[2],
                0.290,
            ),
        )
        for sample in track.history
    )
    edge_track = replace(
        track,
        bbox_norm=edge_history[-1].bbox_norm,
        history=edge_history,
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="projected bbox lacks edge margin",
    ):
        _continue(
            edge_track,
            transition,
            previous,
            started_ns,
            now_ns,
        )

    latest = track.history[-1]
    center_image_x = 0.5 * (latest.center_norm[0] + 1.0)
    previous_width = (
        track.history[-2].bbox_norm[2]
        - track.history[-2].bbox_norm[0]
    )
    widened_half_width = 0.5 * previous_width * 1.06
    widened_bbox = (
        center_image_x - widened_half_width,
        latest.bbox_norm[1],
        center_image_x + widened_half_width,
        latest.bbox_norm[3],
    )
    widened = replace(latest, bbox_norm=widened_bbox)
    deformation_track = replace(
        track,
        bbox_norm=widened_bbox,
        history=track.history[:-1] + (widened,),
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="contour deformation is unsafe",
    ):
        _continue(
            deformation_track,
            transition,
            previous,
            started_ns,
            now_ns,
        )


def test_continuation_hard_duration_is_inclusive_and_immutable():
    track, transition, previous, started_ns, _now_ns = _continued_fixture()
    at_limit_ns = started_ns + round(
        RECOVERY_HARD_DURATION_S * 1_000_000_000
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="projection horizon is exhausted|frame is stale",
    ):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            at_limit_ns,
        )

    beyond_limit_ns = at_limit_ns + 1
    with pytest.raises(
        VisualRecoveryRefusal,
        match="exceeded its hard duration",
    ):
        _continue(
            track,
            transition,
            previous,
            started_ns,
            beyond_limit_ns,
        )
