"""Focused tests for transition-anchor post-promotion recovery admission."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

import planning.vq2_visual_recovery as recovery_policy
from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    AssociationEvidence,
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualTrack,
    VisualTrackRole,
    VisualTrackSample,
    visual_track_history_sha256,
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
    RECOVERY_HISTORY_SAMPLE_COUNT,
    RECOVERY_MAX_ANCHOR_CREDIT_AGE_S,
    RECOVERY_MAX_CONTINUATION_AGE_S,
    RECOVERY_MAX_PROJECTED_ABS_Y_NORM,
    RECOVERY_MAX_RAW_CENTER_RATE_NORM_S,
    RECOVERY_MAX_RAW_LOG_DIMENSION_RATE_S,
    RECOVERY_MAX_REACQUISITION_CENTER_RATE_NORM_S,
    RECOVERY_MAX_REACQUISITION_GAP_S,
    RECOVERY_MAX_REACQUISITION_LOG_SCALE_RATE_S,
    RECOVERY_MAX_REACQUISITION_MISSED_FRAMES,
    RECOVERY_MIN_ASSOCIATION_CONFIDENCE,
    RECOVERY_MIN_HISTORY_SPAN_S,
    RECOVERY_MIN_REACQUISITION_ASSOCIATION_CONFIDENCE,
    PromotionHistoryAuthority,
    ReacquisitionBridgeAdmission,
    RecoveryContinuationAdmission,
    TransitionRecoveryAdmission,
    VisualRecoveryRefusal,
    require_promotion_history_authority,
    require_recovery_continuation,
    require_transition_recovery_admission,
)
from planning.vq2_visual_servo import VisualTarget
from planning.vq2_visual_alignment import (
    POST_PROMOTION_ENTRY_MAX_ABS_X_NORM,
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


# Exact compact frame 167--171 excerpt from
# 20260725T000858Z-visual-align-e874ec54/session.jsonl.gz
# (trace sha256 e440488d71613084992e2efcd9d21c18a5cec3b27a1554f893a5483e56f320ff).
# Frame 167 is the one-frame contour-change association dip.  The following
# four exact frames are a contiguous, high-authority 102.8 ms stable tail.
_STABLE_PRE_GAP_ROWS = (
    (
        2_120_680,
        152,
        70_991_244_206_500,
        70_991_245_094_800,
        (0.3999999999999999, -0.5055555555555555),
        (
            0.659375,
            0.16666666666666666,
            0.7421875,
            0.3277777777777778,
        ),
        0.11550763563024256,
        0.6674738461538461,
        0.9464679927969937,
        1_784_938_145_566_568_200,
    ),
    (
        2_120_681,
        153,
        70_991_276_501_400,
        70_991_277_612_100,
        (0.40625, -0.5055555555555555),
        (
            0.6609375,
            0.16666666666666666,
            0.7453125,
            0.3277777777777778,
        ),
        0.11659223816361025,
        0.6766184905660377,
        0.9373433084474938,
        1_784_938_145_601_276_400,
    ),
    (
        2_120_682,
        154,
        70_991_311_203_200,
        70_991_312_001_400,
        (0.4125000000000001, -0.5166666666666666),
        (0.6625, 0.16111111111111112, 0.75, 0.325),
        0.1197508988599993,
        0.6466540547474844,
        0.9271955866409487,
        1_784_938_145_636_013_000,
    ),
    (
        2_120_683,
        155,
        70_991_347_906_400,
        70_991_348_864_400,
        (0.4156249999999999, -0.5166666666666666),
        (0.6640625, 0.15833333333333333, 0.753125, 0.325),
        0.12183492931011208,
        0.6705908806363146,
        0.9351533809405791,
        1_784_938_145_670_730_400,
    ),
)

_STABLE_TAIL_ROWS = (
    (
        2_120_695,
        167,
        70_991_742_542_700,
        70_991_744_195_800,
        (0.4937499999999999, -0.6055555555555556),
        (0.6875, 0.09722222222222222, 0.8078125, 0.3),
        0.15619443456438803,
        0.7450733333333333,
        0.6655837362714977,
        1_784_938_146_066_574_300,
    ),
    (
        2_120_696,
        168,
        70_991_777_650_100,
        70_991_778_375_200,
        (0.503125, -0.6111111111111112),
        (0.690625, 0.09166666666666666, 0.8140625, 0.3),
        0.1603625449827151,
        0.7534437837837838,
        0.9331590326212811,
        1_784_938_146_101_320_400,
    ),
    (
        2_120_697,
        169,
        70_991_810_878_500,
        70_991_812_087_100,
        (0.515625, -0.6222222222222222),
        (0.69375, 0.08333333333333333, 0.821875, 0.2972222222222222),
        0.16554308771099113,
        0.7568631578947368,
        0.9282073353948699,
        1_784_938_146_136_070_600,
    ),
    (
        2_120_698,
        170,
        70_991_847_936_300,
        70_991_848_706_300,
        (0.5249999999999999, -0.6333333333333333),
        (0.696875, 0.075, 0.828125, 0.29444444444444445),
        0.16971176545346917,
        0.7665692307692307,
        0.938133611089553,
        1_784_938_146_170_748_000,
    ),
    (
        2_120_699,
        171,
        70_991_880_465_000,
        70_991_881_509_600,
        (0.534375, -0.6444444444444444),
        (0.7, 0.06666666666666667, 0.8359375, 0.2916666666666667),
        0.17488835724541532,
        0.77176,
        0.9363999145783396,
        1_784_938_146_205_467_500,
    ),
)
_STABLE_TAIL_RACE_RECEIVED_NS = 70_991_899_845_000


# Exact recorded scalars from the six-publication reacquisition epoch in
# 20260725T013545Z-visual-align-e1439c6e/session.jsonl.gz
# (trace sha256 da5882c0162938d0a9081c76ac316b5639f73bd3246f14b3636ccb95de07052f).
# Publication 152 is the excerpt boundary; its association to 151 is omitted.
_SIX_EPOCH_PRE_GAP_ROWS = (
    (
        2_276_896,
        152,
        76_198_445_439_600,
        76_198_446_361_300,
        (0.3968750000000001, -0.5111111111111111),
        (
            0.6578125,
            0.16666666666666666,
            0.740625,
            0.325,
        ),
        0.11450755069717744,
        0.6726830769230769,
        0.9343610519413472,
        1_784_943_352_769_850_500,
    ),
    (
        2_276_897,
        153,
        76_198_480_662_400,
        76_198_481_500_700,
        (0.40312499999999996, -0.5111111111111111),
        (0.659375, 0.1638888888888889, 0.74375, 0.325),
        0.11659223816361017,
        0.6764584905660377,
        0.936356123439321,
        1_784_943_352_804_590_400,
    ),
    (
        2_276_898,
        154,
        76_198_514_327_900,
        76_198_515_505_500,
        (0.40625, -0.5166666666666666),
        (
            0.6609375,
            0.16111111111111112,
            0.746875,
            0.32222222222222224,
        ),
        0.11766684372035782,
        0.6847533333333333,
        0.944615557833968,
        1_784_943_352_839_307_200,
    ),
    (
        2_276_899,
        155,
        76_198_541_992_200,
        76_198_543_067_700,
        (0.4125000000000001, -0.5222222222222221),
        (
            0.6625,
            0.15555555555555556,
            0.75,
            0.32222222222222224,
        ),
        0.12076147288491201,
        0.6824818181818182,
        0.9369815497402262,
        1_784_943_352_867_098_000,
    ),
)

_SIX_EPOCH_ROWS = (
    (
        2_276_911,
        167,
        76_198_946_068_500,
        76_198_946_888_800,
        (0.4906250000000001, -0.6055555555555556),
        (
            0.6859375,
            0.09722222222222222,
            0.80625,
            0.3,
        ),
        0.15619443456438803,
        0.7447333333333332,
        0.7177611644314987,
        1_784_943_353_269_865_500,
    ),
    (
        2_276_912,
        168,
        76_198_979_573_000,
        76_198_980_260_000,
        (0.5, -0.6111111111111112),
        (
            0.6890625,
            0.09166666666666666,
            0.8125,
            0.2972222222222222,
        ),
        0.1592898737801273,
        0.7471720547945205,
        0.9397230908814789,
        1_784_943_353_304_582_600,
    ),
    (
        2_276_913,
        169,
        76_199_014_758_400,
        76_199_015_720_500,
        (0.5093749999999999, -0.6222222222222222),
        (
            0.6921875,
            0.08333333333333333,
            0.81875,
            0.2972222222222222,
        ),
        0.16453058226360232,
        0.7606905263157895,
        0.930226705465556,
        1_784_943_353_339_318_300,
    ),
    (
        2_276_914,
        170,
        76_199_043_532_900,
        76_199_044_715_500,
        (0.51875, -0.6277777777777778),
        (
            0.69375,
            0.07777777777777778,
            0.825,
            0.29444444444444445,
        ),
        0.1686342195404005,
        0.7600233766233766,
        0.9394573212354178,
        1_784_943_353_367_088_900,
    ),
    (
        2_276_915,
        171,
        76_199_077_024_300,
        76_199_078_014_900,
        (0.528125, -0.6388888888888888),
        (
            0.696875,
            0.06944444444444445,
            0.8328125,
            0.2916666666666667,
        ),
        0.17380544678845172,
        0.7660777215189873,
        0.9353120615583702,
        1_784_943_353_401_824_300,
    ),
    (
        2_276_916,
        172,
        76_199_111_299_700,
        76_199_112_132_000,
        (0.5375000000000001, -0.65),
        (0.7, 0.06111111111111111, 0.8390625, 0.28888888888888886),
        0.17797569278477957,
        0.7749340740740741,
        0.9403938961628551,
        1_784_943_353_436_536_600,
    ),
)
_SIX_EPOCH_RACE_RECEIVED_NS = 76_199_121_890_000


def _accepted_sample(
    previous: VisualTrackSample,
    sample: VisualTrackSample,
    *,
    track_id: str = "vq2-track-000002",
    missed_frames: int = 0,
    cost: float | None = None,
    residual: float = 0.005,
    bbox_iou: float = 0.95,
    log_area_residual: float = 0.0,
    clipping_continuity: float = 1.0,
    ambiguous: bool = False,
    track_ambiguous_before: bool = False,
) -> VisualTrackSample:
    previous_width = previous.bbox_norm[2] - previous.bbox_norm[0]
    previous_height = previous.bbox_norm[3] - previous.bbox_norm[1]
    width = sample.bbox_norm[2] - sample.bbox_norm[0]
    height = sample.bbox_norm[3] - sample.bbox_norm[1]
    assert previous.publication_monotonic_ns is not None
    assert sample.publication_monotonic_ns is not None
    association_cost = (
        (1.0 - sample.association_confidence) * 0.82
        if cost is None
        else cost
    )
    evidence = AssociationEvidence(
        track_id=track_id,
        previous_token=previous.token,
        current_token=sample.token,
        detection_source_index=sample.source_index,
        cost=association_cost,
        confidence=sample.association_confidence,
        predicted_center_residual_norm=residual,
        bbox_iou=bbox_iou,
        log_width_change=math.log(width / previous_width),
        log_height_change=math.log(height / previous_height),
        log_area_residual=log_area_residual,
        clipping_continuity=clipping_continuity,
        temporal_consistency=1.0 / (missed_frames + 1),
        appearance_distance=None,
        ambiguous=ambiguous,
        missed_frame_count_before_association=missed_frames,
        observation_gap_ns=(
            sample.observation_monotonic_ns
            - previous.observation_monotonic_ns
        ),
        publication_gap_ns=(
            sample.publication_monotonic_ns
            - previous.publication_monotonic_ns
        ),
        track_ambiguous_before_association=track_ambiguous_before,
    )
    return replace(sample, accepted_association=evidence)


def _bind_continuous_history(
    samples: tuple[VisualTrackSample, ...],
) -> tuple[VisualTrackSample, ...]:
    bound = [samples[0]]
    for sample in samples[1:]:
        bound.append(_accepted_sample(bound[-1], sample))
    return tuple(bound)


def _prepend_established_identity(
    samples: tuple[VisualTrackSample, ...],
) -> tuple[VisualTrackSample, ...]:
    first = samples[0]
    assert first.publication_monotonic_ns is not None
    prefix = tuple(
        replace(
            first,
            tracker_frame_sequence=first.tracker_frame_sequence - (4 - index),
            token=replace(
                first.token,
                frame_id=first.token.frame_id - (4 - index),
                publication_sequence=(
                    first.token.publication_sequence - (4 - index)
                ),
            ),
            observation_monotonic_ns=(
                first.observation_monotonic_ns - (4 - index) * 33_000_000
            ),
            publication_monotonic_ns=(
                first.publication_monotonic_ns - (4 - index) * 33_000_000
            ),
            camera_source_time_ns=None,
            confidence=0.72,
            association_confidence=0.95,
            accepted_association=None,
        )
        for index in range(4)
    )
    return _bind_continuous_history(prefix + samples)


def _fixture() -> tuple[
    VisualTrack,
    ConfirmedGateTransition,
]:
    transition_samples = tuple(
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
    samples = _prepend_established_identity(transition_samples)
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
        promoted_history_sha256=visual_track_history_sha256(samples),
    )
    return track, transition


def _stable_tail_fixture() -> tuple[
    VisualTrack,
    ConfirmedGateTransition,
]:
    # This fixture copies every recorded scalar in the exact nine-observation
    # proof window around the live visibility interruption.  It intentionally
    # starts at publication 152, so only that boundary sample's association to
    # publication 151 is omitted; the fixture does not claim to reproduce the
    # full 157-observation promotion history.  Publications 153--155 prove
    # established identity before the gap, while 167--171 prove bounded
    # reacquisition and the stable post-gap tail.
    prefix_samples = tuple(
        VisualTrackSample(
            tracker_frame_sequence=149 + index,
            token=CameraFrameToken(
                generation=1,
                frame_id=row[0],
                publication_sequence=row[1],
                stream_id="vq2-camera-udp-5600",
            ),
            observation_monotonic_ns=row[2],
            publication_monotonic_ns=row[3],
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            camera_source_time_ns=row[9],
            source_index=1,
            center_norm=row[4],
            bbox_norm=row[5],
            apparent_scale=row[6],
            confidence=row[7],
            clipping=FrameEdge.NONE,
            center_censored=False,
            association_confidence=row[8],
        )
        for index, row in enumerate(_STABLE_PRE_GAP_ROWS)
    )
    bound_prefix = [prefix_samples[0]]
    exact_prefix_associations = (
        (
            0.051378487073055104,
            0.0062579784421234125,
            0.9487289421554578,
            -0.007524335180372432,
        ),
        (
            0.05969961895442211,
            0.008497638581617142,
            0.9315341703531249,
            0.029738540689553616,
        ),
        (
            0.05317422762872512,
            0.008305011894605346,
            0.9535420156878494,
            -0.007884607082040418,
        ),
    )
    for sample, evidence in zip(
        prefix_samples[1:],
        exact_prefix_associations,
        strict=True,
    ):
        bound_prefix.append(
            _accepted_sample(
                bound_prefix[-1],
                sample,
                cost=evidence[0],
                residual=evidence[1],
                bbox_iou=evidence[2],
                log_area_residual=evidence[3],
            )
        )
    transition_samples = tuple(
        VisualTrackSample(
            tracker_frame_sequence=164 + index,
            token=CameraFrameToken(
                generation=1,
                frame_id=row[0],
                publication_sequence=row[1],
                stream_id="vq2-camera-udp-5600",
            ),
            observation_monotonic_ns=row[2],
            publication_monotonic_ns=row[3],
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            camera_source_time_ns=row[9],
            source_index=1 if index == 0 else 0,
            center_norm=row[4],
            bbox_norm=row[5],
            apparent_scale=row[6],
            confidence=row[7],
            clipping=FrameEdge.NONE,
            center_censored=False,
            association_confidence=row[8],
        )
        for index, row in enumerate(_STABLE_TAIL_ROWS)
    )
    bridge = _accepted_sample(
        bound_prefix[-1],
        transition_samples[0],
        missed_frames=11,
        cost=0.2742213362573719,
        residual=0.05951303777040391,
        bbox_iou=0.572322349916915,
        log_area_residual=0.0877005386799139,
    )
    bound_transition = [bridge]
    exact_tail_associations = (
        (
            0.05480959325054955,
            0.0036325981417213975,
            0.9448896353763697,
            0.011979818808735221,
        ),
        (
            0.05886998497620671,
            0.007715960019892727,
            0.9376353097257341,
            0.018839024019028372,
        ),
        (
            0.05073043890656653,
            0.002535576710918654,
            0.9514767932489457,
            -0.011722432168477237,
        ),
        (
            0.05215207004576156,
            0.0020214073054181975,
            0.941677309493401,
            0.011801333153957927,
        ),
    )
    for sample, evidence in zip(
        transition_samples[1:],
        exact_tail_associations,
        strict=True,
    ):
        bound_transition.append(
            _accepted_sample(
                bound_transition[-1],
                sample,
                cost=evidence[0],
                residual=evidence[1],
                bbox_iou=evidence[2],
                log_area_residual=evidence[3],
            )
        )
    samples = tuple(bound_prefix + bound_transition)
    race = AuthoritativeRaceStatusRef.live(
        session_id=(
            "3041ea9cab1daa0f773d38d06155822993d660d0005e9f51864f2292313c09a3"
        ),
        reset_epoch=1,
        race_generation=2,
        race_status_sequence=1476,
        race_status_boot_ms=6069,
        active_gate_index=1,
        received_monotonic_ns=_STABLE_TAIL_RACE_RECEIVED_NS,
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
            0.2831292677274865,
            -0.3140795585237663,
        ),
        log_scale_rate_s=0.8420567390858966,
        confidence=0.765302508859699,
        association_confidence=samples[-1].association_confidence,
        consecutive_frame_count=len(bound_transition),
        total_observation_count=len(samples),
        missed_frame_count=0,
        clipping=FrameEdge.NONE,
        center_censored=False,
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=1,
        authority_race_status_sequence=1476,
        authority_race_status_boot_ms=6069,
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
            sample.token for sample in bound_transition
        ),
        history_length_before_promotion=len(samples),
        history_length_after_promotion=len(samples),
        promoted_history_sha256=visual_track_history_sha256(samples),
    )
    return track, transition


def _six_frame_epoch_fixture() -> tuple[
    VisualTrack,
    ConfirmedGateTransition,
]:
    prefix_samples = tuple(
        VisualTrackSample(
            tracker_frame_sequence=150 + index,
            token=CameraFrameToken(
                generation=1,
                frame_id=row[0],
                publication_sequence=row[1],
                stream_id="vq2-camera-udp-5600",
            ),
            observation_monotonic_ns=row[2],
            publication_monotonic_ns=row[3],
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            camera_source_time_ns=row[9],
            source_index=1,
            center_norm=row[4],
            bbox_norm=row[5],
            apparent_scale=row[6],
            confidence=row[7],
            clipping=FrameEdge.NONE,
            center_censored=False,
            association_confidence=row[8],
        )
        for index, row in enumerate(_SIX_EPOCH_PRE_GAP_ROWS)
    )
    bound_prefix = [prefix_samples[0]]
    exact_prefix_associations = (
        (
            0.052187978779756825,
            0.00795877570372995,
            0.9519854981376425,
            -0.0008049934776899192,
        ),
        (
            0.045415242576146195,
            0.002972858922451899,
            0.9682849759606584,
            -0.016485597546768815,
        ),
        (
            0.05167512921301456,
            0.003482668878445401,
            0.9494047619047616,
            0.03074573304372752,
        ),
    )
    for sample, evidence in zip(
        prefix_samples[1:],
        exact_prefix_associations,
        strict=True,
    ):
        bound_prefix.append(
            _accepted_sample(
                bound_prefix[-1],
                sample,
                cost=evidence[0],
                residual=evidence[1],
                bbox_iou=evidence[2],
                log_area_residual=evidence[3],
            )
        )

    epoch_samples = tuple(
        VisualTrackSample(
            tracker_frame_sequence=164 + index,
            token=CameraFrameToken(
                generation=1,
                frame_id=row[0],
                publication_sequence=row[1],
                stream_id="vq2-camera-udp-5600",
            ),
            observation_monotonic_ns=row[2],
            publication_monotonic_ns=row[3],
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            camera_source_time_ns=row[9],
            source_index=1 if index == 0 else 0,
            center_norm=row[4],
            bbox_norm=row[5],
            apparent_scale=row[6],
            confidence=row[7],
            clipping=FrameEdge.NONE,
            center_censored=False,
            association_confidence=row[8],
        )
        for index, row in enumerate(_SIX_EPOCH_ROWS)
    )
    bound_epoch = [
        _accepted_sample(
            bound_prefix[-1],
            epoch_samples[0],
            missed_frames=10,
            cost=0.23143584516617108,
            residual=0.015530124412322066,
            bbox_iou=0.5977584059775841,
            log_area_residual=-0.04170914425679895,
        )
    ]
    exact_epoch_associations = (
        (
            0.0494270654771873,
            0.003240237653215293,
            0.9614459664977612,
            -0.004974292674191805,
        ),
        (
            0.057214101518244125,
            0.004990318344826129,
            0.9373096039762703,
            0.02117376376835356,
        ),
        (
            0.04964499658695737,
            0.0027071155715102323,
            0.9519230769230772,
            0.004117688592047308,
        ),
        (
            0.05304410952213641,
            0.003764036772643817,
            0.941379310344828,
            0.005217962331851123,
        ),
        (
            0.04887700514645889,
            0.0015217735110336194,
            0.9536859413537945,
            -0.011999331758205223,
        ),
    )
    for sample, evidence in zip(
        epoch_samples[1:],
        exact_epoch_associations,
        strict=True,
    ):
        bound_epoch.append(
            _accepted_sample(
                bound_epoch[-1],
                sample,
                cost=evidence[0],
                residual=evidence[1],
                bbox_iou=evidence[2],
                log_area_residual=evidence[3],
            )
        )

    samples = tuple(bound_prefix + bound_epoch)
    race = AuthoritativeRaceStatusRef.live(
        session_id=(
            "4b336a6a696a82066b7d7154337bed19dc7a3e451c605ae4bdd343fffc6bf151"
        ),
        reset_epoch=1,
        race_generation=2,
        race_status_sequence=1478,
        race_status_boot_ms=6068,
        active_gate_index=1,
        received_monotonic_ns=_SIX_EPOCH_RACE_RECEIVED_NS,
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
            0.2791059905697063,
            -0.30498914969929664,
        ),
        log_scale_rate_s=0.7705316781003424,
        confidence=0.7686564964069769,
        association_confidence=samples[-1].association_confidence,
        consecutive_frame_count=len(bound_epoch),
        total_observation_count=len(samples),
        missed_frame_count=0,
        clipping=FrameEdge.NONE,
        center_censored=False,
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=1,
        authority_race_status_sequence=1478,
        authority_race_status_boot_ms=6068,
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
            sample.token for sample in bound_epoch
        ),
        history_length_before_promotion=len(samples),
        history_length_after_promotion=len(samples),
        promoted_history_sha256=visual_track_history_sha256(samples),
    )
    return track, transition


def _shift_six_epoch_timing(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
    *,
    observation_shift_ns: int,
    publication_shift_ns: int,
    rehash_transition: bool,
) -> tuple[VisualTrack, ConfirmedGateTransition]:
    epoch_start = len(track.history) - track.consecutive_frame_count
    shifted = list(track.history[:epoch_start])
    for sample in track.history[epoch_start:]:
        shifted_sample = replace(
            sample,
            observation_monotonic_ns=(
                sample.observation_monotonic_ns + observation_shift_ns
            ),
            publication_monotonic_ns=(
                sample.publication_monotonic_ns + publication_shift_ns
                if sample.publication_monotonic_ns is not None
                else None
            ),
        )
        assert shifted_sample.accepted_association is not None
        previous = shifted[-1]
        shifted_sample = replace(
            shifted_sample,
            accepted_association=replace(
                shifted_sample.accepted_association,
                observation_gap_ns=(
                    shifted_sample.observation_monotonic_ns
                    - previous.observation_monotonic_ns
                ),
                publication_gap_ns=(
                    shifted_sample.publication_monotonic_ns
                    - previous.publication_monotonic_ns
                ),
            ),
        )
        shifted.append(shifted_sample)
    history = tuple(shifted)
    shifted_track = replace(track, history=history)
    assert transition.race_status.received_monotonic_ns is not None
    shifted_race = replace(
        transition.race_status,
        received_monotonic_ns=(
            transition.race_status.received_monotonic_ns
            + publication_shift_ns
        ),
    )
    shifted_transition = replace(
        transition,
        race_status=shifted_race,
        promoted_history_sha256=(
            visual_track_history_sha256(history)
            if rehash_transition
            else transition.promoted_history_sha256
        ),
    )
    return shifted_track, shifted_transition


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
        accepted_association=None,
    )
    latest = _accepted_sample(previous, latest)
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
        consecutive_frame_count=anchor.consecutive_frame_count + 1,
        total_observation_count=anchor.total_observation_count + 1,
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
    latest = _accepted_sample(
        previous,
        latest,
        cost=0.05823968,
        residual=0.01,
        bbox_iou=0.90,
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
        consecutive_frame_count=anchor.consecutive_frame_count + 1,
        total_observation_count=anchor.total_observation_count + 1,
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
    assert admission.reacquisition_bridge is None


def test_exact_weak_frame_before_four_sample_stable_tail_admits():
    track, transition = _stable_tail_fixture()
    pre_gap = track.history[:4]
    established_pre_gap = pre_gap[-3:]
    stable_tail = track.history[-RECOVERY_HISTORY_SAMPLE_COUNT:]
    pre_gap_span_s = (
        established_pre_gap[-1].observation_monotonic_ns
        - established_pre_gap[0].observation_monotonic_ns
    ) / 1_000_000_000.0
    history_span_s = (
        stable_tail[-1].observation_monotonic_ns
        - stable_tail[0].observation_monotonic_ns
    ) / 1_000_000_000.0

    assert RECOVERY_HISTORY_SAMPLE_COUNT == 4
    assert RECOVERY_MIN_HISTORY_SPAN_S == pytest.approx(0.090)
    assert RECOVERY_MIN_ASSOCIATION_CONFIDENCE == pytest.approx(0.90)
    assert (
        track.history[-RECOVERY_HISTORY_SAMPLE_COUNT - 1]
        .association_confidence
        < RECOVERY_MIN_ASSOCIATION_CONFIDENCE
    )
    assert all(
        sample.association_confidence
        >= RECOVERY_MIN_ASSOCIATION_CONFIDENCE
        for sample in stable_tail
    )
    assert history_span_s == pytest.approx(0.1028149)
    assert history_span_s >= RECOVERY_MIN_HISTORY_SPAN_S
    assert tuple(
        sample.token.publication_sequence for sample in pre_gap
    ) == (152, 153, 154, 155)
    assert pre_gap_span_s == pytest.approx(0.071405)
    assert min(
        sample.confidence for sample in established_pre_gap
    ) == pytest.approx(0.6466540547474844)
    assert min(
        sample.association_confidence for sample in established_pre_gap
    ) == pytest.approx(0.9271955866409487)

    admission = require_transition_recovery_admission(
        track,
        transition,
        tracker_time_basis_id="host-perf-counter",
        measured_pitch_rad=-0.04001,
        now_monotonic_ns=_STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000,
    )

    assert admission.history_tokens == tuple(
        sample.token for sample in stable_tail
    )
    assert tuple(
        token.publication_sequence for token in admission.history_tokens
    ) == (168, 169, 170, 171)
    assert admission.anchor_credit_age_s == pytest.approx(0.0183354)
    assert admission.stable_history_span_s == pytest.approx(0.1028149)
    assert admission.min_history_detection_confidence == pytest.approx(
        0.7534437837837838
    )
    assert admission.min_history_association_confidence == pytest.approx(
        0.9282073353948699
    )
    assert type(admission.reacquisition_bridge) is ReacquisitionBridgeAdmission
    bridge = admission.reacquisition_bridge
    assert bridge.predecessor_token.publication_sequence == 155
    assert bridge.reacquisition_token.publication_sequence == 167
    assert bridge.missed_frame_count == 11
    assert bridge.tracker_frame_delta == 12
    assert bridge.publication_delta == 12
    assert bridge.unobserved_publication_count == 0
    assert bridge.observation_gap_s == pytest.approx(0.3946363)
    assert bridge.publication_gap_s == pytest.approx(0.3953314)
    assert bridge.association_confidence == pytest.approx(
        RECOVERY_MIN_REACQUISITION_ASSOCIATION_CONFIDENCE
        + 0.0155837362714977
    )
    assert bridge.bbox_iou == pytest.approx(0.572322349916915)
    assert bridge.direct_bbox_iou == pytest.approx(0.3104797796782144)
    assert bridge.predicted_center_residual_norm == pytest.approx(
        0.05951303777040391
    )
    assert bridge.average_horizontal_rate_norm_s == pytest.approx(
        0.19796709020432232
    )
    assert bridge.average_vertical_rate_norm_s == pytest.approx(
        -0.22524255596580706
    )
    assert bridge.average_log_scale_rate_s == pytest.approx(
        0.629527786654983
    )


def test_exact_six_frame_epoch_with_one_publication_skip_admits():
    track, transition = _six_frame_epoch_fixture()

    admission = require_transition_recovery_admission(
        track,
        transition,
        tracker_time_basis_id="host-perf-counter",
        measured_pitch_rad=-0.04001,
        now_monotonic_ns=_SIX_EPOCH_RACE_RECEIVED_NS + 1_000_000,
    )

    assert track.consecutive_frame_count == 6
    assert len(transition.pretransition_frame_tokens) == 6
    assert tuple(
        token.publication_sequence
        for token in transition.pretransition_frame_tokens
    ) == (167, 168, 169, 170, 171, 172)
    assert tuple(
        token.publication_sequence for token in admission.history_tokens
    ) == (169, 170, 171, 172)
    assert admission.anchor_credit_age_s == pytest.approx(0.009758)
    assert admission.stable_history_span_s == pytest.approx(0.0965413)
    assert admission.min_history_detection_confidence == pytest.approx(
        0.7600233766233766
    )
    assert admission.min_history_association_confidence == pytest.approx(
        0.930226705465556
    )
    bridge = admission.reacquisition_bridge
    assert type(bridge) is ReacquisitionBridgeAdmission
    assert bridge.missed_frame_count == 10
    assert bridge.tracker_frame_delta == 11
    assert bridge.publication_delta == 12
    assert bridge.publication_delta == bridge.tracker_frame_delta + 1
    assert bridge.unobserved_publication_count == 1
    assert bridge.observation_gap_s == pytest.approx(0.4040763)
    assert bridge.publication_gap_s == pytest.approx(0.4038211)
    assert 0.400 < bridge.observation_gap_s <= (
        RECOVERY_MAX_REACQUISITION_GAP_S
    )
    assert RECOVERY_MAX_REACQUISITION_GAP_S == pytest.approx(0.410)
    assert bridge.direct_bbox_iou == pytest.approx(
        0.3112863191706818
    )
    assert bridge.average_horizontal_rate_norm_s == pytest.approx(
        0.19334219799577457
    )
    assert bridge.average_vertical_rate_norm_s == pytest.approx(
        -0.2062316778621599
    )
    assert bridge.average_log_scale_rate_s == pytest.approx(
        0.636722086948461
    )
    assert admission.max_raw_log_width_rate_s == pytest.approx(
        1.2638844869893338
    )
    assert 1.25 < admission.max_raw_log_width_rate_s <= (
        RECOVERY_MAX_RAW_LOG_DIMENSION_RATE_S
    )
    assert RECOVERY_MAX_RAW_LOG_DIMENSION_RATE_S == pytest.approx(1.30)
    assert admission.projected_abs_horizontal_error == pytest.approx(
        0.5698147448261486
    )
    assert (
        POST_PROMOTION_ENTRY_MAX_ABS_X_NORM
        - admission.projected_abs_horizontal_error
        == pytest.approx(0.10018525517385146)
    )
    assert admission.projected_abs_vertical_error_image_down == (
        pytest.approx(0.6876519212560703)
    )
    assert admission.projected_bbox_norm_ltrb[1] > 6.0 / 360.0


@pytest.mark.parametrize(
    (
        "observation_gap_ns",
        "publication_gap_ns",
        "admitted",
    ),
    (
        (410_000_000, 410_000_000, True),
        (410_000_001, 410_000_000, False),
        (410_000_000, 410_000_001, False),
    ),
)
def test_reacquisition_gap_four_ten_boundary_is_inclusive(
    observation_gap_ns,
    publication_gap_ns,
    admitted,
):
    track, transition = _six_frame_epoch_fixture()
    bridge_index = len(track.history) - track.consecutive_frame_count
    predecessor = track.history[bridge_index - 1]
    bridge = track.history[bridge_index]
    assert predecessor.publication_monotonic_ns is not None
    assert bridge.publication_monotonic_ns is not None
    observation_shift_ns = (
        observation_gap_ns
        - (
            bridge.observation_monotonic_ns
            - predecessor.observation_monotonic_ns
        )
    )
    publication_shift_ns = (
        publication_gap_ns
        - (
            bridge.publication_monotonic_ns
            - predecessor.publication_monotonic_ns
        )
    )
    track, transition = _shift_six_epoch_timing(
        track,
        transition,
        observation_shift_ns=observation_shift_ns,
        publication_shift_ns=publication_shift_ns,
        rehash_transition=admitted,
    )
    now_ns = (
        transition.race_status.received_monotonic_ns + 1_000_000
    )

    if admitted:
        admission = require_transition_recovery_admission(
            track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=now_ns,
        )
        assert admission.reacquisition_bridge is not None
        assert admission.reacquisition_bridge.observation_gap_s == (
            pytest.approx(0.410)
        )
        assert admission.reacquisition_bridge.publication_gap_s == (
            pytest.approx(0.410)
        )
    else:
        with pytest.raises(
            VisualRecoveryRefusal,
            match="reacquisition timing is unsafe",
        ):
            require_transition_recovery_admission(
                track,
                transition,
                tracker_time_basis_id="host-perf-counter",
                measured_pitch_rad=-0.04001,
                now_monotonic_ns=now_ns,
            )


def test_reacquisition_bridge_rejects_two_unobserved_publications():
    track, transition = _six_frame_epoch_fixture()
    bridge_index = len(track.history) - track.consecutive_frame_count
    shifted = list(track.history[:bridge_index])
    for offset, sample in enumerate(track.history[bridge_index:]):
        assert sample.accepted_association is not None
        shifted_sample = replace(
            sample,
            tracker_frame_sequence=sample.tracker_frame_sequence - 1,
            accepted_association=replace(
                sample.accepted_association,
                missed_frame_count_before_association=(
                    9 if offset == 0 else 0
                ),
                temporal_consistency=(0.1 if offset == 0 else 1.0),
            ),
        )
        shifted.append(shifted_sample)
    track = replace(track, history=tuple(shifted))

    with pytest.raises(
        VisualRecoveryRefusal,
        match="association gap is inconsistent",
    ):
        require_transition_recovery_admission(
            track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=_SIX_EPOCH_RACE_RECEIVED_NS + 1_000_000,
        )


def test_six_frame_epoch_rejects_last_five_token_relabel_or_hidden_ambiguity():
    track, transition = _six_frame_epoch_fixture()
    shortened = replace(
        transition,
        pretransition_frame_tokens=transition.pretransition_frame_tokens[-5:],
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="visibility epoch is inconsistent",
    ):
        require_transition_recovery_admission(
            track,
            shortened,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=_SIX_EPOCH_RACE_RECEIVED_NS + 1_000_000,
        )

    epoch_start = len(track.history) - track.consecutive_frame_count
    hidden = track.history[epoch_start + 1]
    assert hidden.accepted_association is not None
    hidden = replace(
        hidden,
        accepted_association=replace(
            hidden.accepted_association,
            ambiguous=True,
        ),
    )
    track = replace(
        track,
        history=(
            track.history[: epoch_start + 1]
            + (hidden,)
            + track.history[epoch_start + 2 :]
        ),
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="association provenance is inconsistent",
    ):
        require_transition_recovery_admission(
            track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=_SIX_EPOCH_RACE_RECEIVED_NS + 1_000_000,
        )


def test_clean_stable_tail_rejects_an_unobserved_publication():
    track, transition = _six_frame_epoch_fixture()
    changed_index = len(track.history) - 3
    changed = track.history[changed_index]
    assert changed.accepted_association is not None
    changed_token = replace(
        changed.token,
        publication_sequence=changed.token.publication_sequence + 1,
    )
    changed = replace(
        changed,
        token=changed_token,
        accepted_association=replace(
            changed.accepted_association,
            current_token=changed_token,
        ),
    )
    history = (
        track.history[:changed_index]
        + (changed,)
        + track.history[changed_index + 1 :]
    )
    epoch_start = len(history) - track.consecutive_frame_count
    transition = replace(
        transition,
        pretransition_frame_tokens=tuple(
            sample.token for sample in history[epoch_start:]
        ),
    )
    track = replace(track, history=history)

    with pytest.raises(
        VisualRecoveryRefusal,
        match="association gap is inconsistent",
    ):
        require_transition_recovery_admission(
            track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=_SIX_EPOCH_RACE_RECEIVED_NS + 1_000_000,
        )


def test_prevalidated_promotion_history_avoids_wire_time_rehash(
    monkeypatch,
):
    track, transition = _stable_tail_fixture()
    authority = require_promotion_history_authority(
        track,
        transition,
        tracker_time_basis_id="host-perf-counter",
    )
    assert type(authority) is PromotionHistoryAuthority
    assert authority.history_sha256 == transition.promoted_history_sha256

    def unexpected_rehash(_history):
        raise AssertionError("wire-time promotion history was rehashed")

    monkeypatch.setattr(
        recovery_policy,
        "visual_track_history_sha256",
        unexpected_rehash,
    )
    admission = require_transition_recovery_admission(
        track,
        transition,
        tracker_time_basis_id="host-perf-counter",
        measured_pitch_rad=-0.04001,
        now_monotonic_ns=_STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000,
        promotion_history_authority=authority,
    )

    assert admission.promotion_identity_sha256 == authority.history_sha256


def test_prevalidated_promotion_history_rejects_forgery_or_prefix_change():
    track, transition = _stable_tail_fixture()
    authority = require_promotion_history_authority(
        track,
        transition,
        tracker_time_basis_id="host-perf-counter",
    )
    forged_authority = replace(authority, _validator_seal=object())

    with pytest.raises(
        VisualRecoveryRefusal,
        match="prevalidated history changed",
    ):
        require_transition_recovery_admission(
            track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=(
                _STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000
            ),
            promotion_history_authority=forged_authority,
        )

    changed_sample = replace(
        track.history[0],
        confidence=track.history[0].confidence + 1e-6,
    )
    changed_track = replace(
        track,
        history=(changed_sample,) + track.history[1:],
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="prevalidated history changed",
    ):
        require_transition_recovery_admission(
            changed_track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=(
                _STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000
            ),
            promotion_history_authority=authority,
        )

    forged_authority = replace(
        authority,
        history=changed_track.history[
            : transition.history_length_after_promotion
        ],
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="prevalidated history changed",
    ):
        require_transition_recovery_admission(
            changed_track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=(
                _STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000
            ),
            promotion_history_authority=forged_authority,
        )


def _admit_stable_trace(
    track: VisualTrack,
    transition: ConfirmedGateTransition,
) -> TransitionRecoveryAdmission:
    return require_transition_recovery_admission(
        track,
        transition,
        tracker_time_basis_id="host-perf-counter",
        measured_pitch_rad=-0.04001,
        now_monotonic_ns=_STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000,
    )


def test_reacquisition_bridge_requires_full_untruncated_track_history():
    track, transition = _stable_tail_fixture()
    truncated_history = track.history[-5:]
    truncated = replace(
        track,
        first_token=truncated_history[0].token,
        total_observation_count=len(truncated_history),
        consecutive_frame_count=len(truncated_history),
        history=truncated_history,
    )
    truncated_transition = replace(
        transition,
        promoted_first_token=truncated.first_token,
        history_length_before_promotion=len(truncated_history),
        history_length_after_promotion=len(truncated_history),
    )

    with pytest.raises(
        VisualRecoveryRefusal,
        match="lacks established pre-gap identity",
    ):
        _admit_stable_trace(truncated, truncated_transition)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (
            lambda evidence: None,
            "reacquisition bridge is absent",
        ),
        (
            lambda evidence: replace(
                evidence,
                track_ambiguous_before_association=True,
            ),
            "association provenance is inconsistent",
        ),
        (
            lambda evidence: replace(evidence, ambiguous=True),
            "association provenance is inconsistent",
        ),
        (
            lambda evidence: replace(
                evidence,
                detection_source_index=99,
            ),
            "association provenance is inconsistent",
        ),
        (
            lambda evidence: replace(
                evidence,
                observation_gap_ns=evidence.observation_gap_ns + 1,
            ),
            "association gap is inconsistent",
        ),
        (
            lambda evidence: replace(
                evidence,
                temporal_consistency=0.09,
            ),
            "association authority is insufficient",
        ),
        (
            lambda evidence: replace(evidence, cost=0.291),
            "association authority is insufficient",
        ),
        (
            lambda evidence: replace(
                evidence,
                bbox_iou=0.549,
            ),
            "reacquisition geometry is unsafe",
        ),
        (
            lambda evidence: replace(
                evidence,
                predicted_center_residual_norm=0.066,
            ),
            "reacquisition geometry is unsafe",
        ),
        (
            lambda evidence: replace(
                evidence,
                log_width_change=0.321,
            ),
            "reacquisition geometry is unsafe",
        ),
        (
            lambda evidence: replace(
                evidence,
                log_height_change=0.221,
            ),
            "reacquisition geometry is unsafe",
        ),
        (
            lambda evidence: replace(
                evidence,
                log_area_residual=0.101,
            ),
            "reacquisition geometry is unsafe",
        ),
        (
            lambda evidence: replace(
                evidence,
                clipping_continuity=0.999,
            ),
            "reacquisition geometry is unsafe",
        ),
    ),
)
def test_reacquisition_bridge_rejects_missing_ambiguous_or_unsafe_evidence(
    mutate,
    reason,
):
    track, transition = _stable_tail_fixture()
    bridge_index = (
        transition.history_length_after_promotion
        - len(transition.pretransition_frame_tokens)
    )
    bridge = track.history[bridge_index]
    changed = replace(
        bridge,
        accepted_association=mutate(bridge.accepted_association),
    )
    track = replace(
        track,
        history=(
            track.history[:bridge_index]
            + (changed,)
            + track.history[bridge_index + 1 :]
        ),
    )

    with pytest.raises(VisualRecoveryRefusal, match=reason):
        _admit_stable_trace(track, transition)


def test_reacquisition_bridge_rejects_a_coherent_gap_over_four_ten_ms():
    track, transition = _stable_tail_fixture()
    bridge_index = (
        transition.history_length_after_promotion
        - len(transition.pretransition_frame_tokens)
    )
    shifted: list[VisualTrackSample] = list(track.history[:bridge_index])
    for sample in track.history[bridge_index:]:
        shifted_sample = replace(
            sample,
            observation_monotonic_ns=(
                sample.observation_monotonic_ns + 16_000_000
            ),
            publication_monotonic_ns=(
                sample.publication_monotonic_ns + 16_000_000
                if sample.publication_monotonic_ns is not None
                else None
            ),
        )
        previous = shifted[-1]
        assert shifted_sample.accepted_association is not None
        shifted_sample = replace(
            shifted_sample,
            accepted_association=replace(
                shifted_sample.accepted_association,
                previous_token=previous.token,
                current_token=shifted_sample.token,
                observation_gap_ns=(
                    shifted_sample.observation_monotonic_ns
                    - previous.observation_monotonic_ns
                ),
                publication_gap_ns=(
                    shifted_sample.publication_monotonic_ns
                    - previous.publication_monotonic_ns
                ),
            ),
        )
        shifted.append(shifted_sample)
    shifted_history = tuple(shifted)
    track = replace(track, history=shifted_history)

    with pytest.raises(
        VisualRecoveryRefusal,
        match="reacquisition timing is unsafe",
    ):
        _admit_stable_trace(track, transition)


def test_reacquisition_bridge_rejects_low_direct_bbox_overlap():
    track, transition = _stable_tail_fixture()
    bridge_index = (
        transition.history_length_after_promotion
        - len(transition.pretransition_frame_tokens)
    )
    bridge = track.history[bridge_index]
    bridge = replace(
        bridge,
        bbox_norm=(
            bridge.bbox_norm[0] + 0.002,
            bridge.bbox_norm[1],
            bridge.bbox_norm[2] + 0.002,
            bridge.bbox_norm[3],
        ),
    )
    track = replace(
        track,
        history=(
            track.history[:bridge_index]
            + (bridge,)
            + track.history[bridge_index + 1 :]
        ),
    )

    with pytest.raises(
        VisualRecoveryRefusal,
        match="reacquisition contour is inconsistent",
    ):
        _admit_stable_trace(track, transition)


@pytest.mark.parametrize(
    "unsafe_dimension",
    ("center", "scale"),
)
def test_reacquisition_bridge_rejects_excessive_center_or_scale_rate(
    unsafe_dimension,
):
    track, transition = _stable_tail_fixture()
    bridge_index = (
        transition.history_length_after_promotion
        - len(transition.pretransition_frame_tokens)
    )
    bridge = track.history[bridge_index]
    predecessor = track.history[bridge_index - 1]
    gap_s = (
        bridge.observation_monotonic_ns
        - predecessor.observation_monotonic_ns
    ) / 1_000_000_000.0
    if unsafe_dimension == "center":
        bridge = replace(bridge, center_norm=(0.515, bridge.center_norm[1]))
        assert (
            abs(bridge.center_norm[0] - predecessor.center_norm[0]) / gap_s
            > RECOVERY_MAX_REACQUISITION_CENTER_RATE_NORM_S
        )
    else:
        bridge = replace(
            bridge,
            apparent_scale=(
                predecessor.apparent_scale
                * math.exp(
                    (
                        RECOVERY_MAX_REACQUISITION_LOG_SCALE_RATE_S
                        + 0.001
                    )
                    * gap_s
                )
            ),
        )
        assert (
            abs(
                math.log(
                    bridge.apparent_scale / predecessor.apparent_scale
                )
                / gap_s
            )
            > RECOVERY_MAX_REACQUISITION_LOG_SCALE_RATE_S
        )
    track = replace(
        track,
        history=(
            track.history[:bridge_index]
            + (bridge,)
            + track.history[bridge_index + 1 :]
        ),
    )

    with pytest.raises(
        VisualRecoveryRefusal,
        match="reacquisition motion is unsafe",
    ):
        _admit_stable_trace(track, transition)


def test_reacquisition_bridge_rejects_twelve_misses_and_delta_thirteen():
    track, transition = _stable_tail_fixture()
    bridge_index = (
        transition.history_length_after_promotion
        - len(transition.pretransition_frame_tokens)
    )
    shifted = list(track.history[:bridge_index])
    for offset, sample in enumerate(track.history[bridge_index:]):
        assert sample.accepted_association is not None
        token = replace(
            sample.token,
            frame_id=sample.token.frame_id + 1,
            publication_sequence=sample.token.publication_sequence + 1,
        )
        shifted_sample = replace(
            sample,
            tracker_frame_sequence=sample.tracker_frame_sequence + 1,
            token=token,
            accepted_association=replace(
                sample.accepted_association,
                previous_token=shifted[-1].token,
                current_token=token,
                missed_frame_count_before_association=(
                    RECOVERY_MAX_REACQUISITION_MISSED_FRAMES + 1
                    if offset == 0
                    else 0
                ),
                temporal_consistency=(
                    1.0 / (RECOVERY_MAX_REACQUISITION_MISSED_FRAMES + 2)
                    if offset == 0
                    else 1.0
                ),
            ),
        )
        shifted.append(shifted_sample)
    history = tuple(shifted)
    bridge = history[bridge_index]
    track = replace(
        track,
        latest_token=history[-1].token,
        history=history,
    )
    transition = replace(
        transition,
        camera_token_at_credit=history[-1].token,
        promoted_latest_token_before_credit=history[-1].token,
        pretransition_frame_tokens=tuple(
            sample.token for sample in history[-5:]
        ),
    )

    assert RECOVERY_MAX_REACQUISITION_MISSED_FRAMES == 11
    assert (
        bridge.token.publication_sequence
        - history[bridge_index - 1].token.publication_sequence
    ) == 13
    with pytest.raises(
        VisualRecoveryRefusal,
        match="reacquisition gap is outside bounds",
    ):
        _admit_stable_trace(track, transition)


def test_reacquisition_bridge_keeps_a_separate_point_sixty_five_floor():
    track, transition = _stable_tail_fixture()
    bridge_index = (
        transition.history_length_after_promotion
        - len(transition.pretransition_frame_tokens)
    )
    bridge = track.history[bridge_index]
    assert bridge.accepted_association is not None
    low_confidence = (
        RECOVERY_MIN_REACQUISITION_ASSOCIATION_CONFIDENCE - 0.001
    )
    bridge = replace(
        bridge,
        association_confidence=low_confidence,
        accepted_association=replace(
            bridge.accepted_association,
            confidence=low_confidence,
        ),
    )
    track = replace(
        track,
        history=(
            track.history[:bridge_index]
            + (bridge,)
            + track.history[bridge_index + 1 :]
        ),
    )

    with pytest.raises(
        VisualRecoveryRefusal,
        match="history confidence is insufficient",
    ):
        _admit_stable_trace(track, transition)


@pytest.mark.parametrize("history_index", (-1, -3, -6))
def test_recovery_requires_accepted_association_on_both_identity_tails(
    history_index,
):
    track, transition = _stable_tail_fixture()
    sample = replace(
        track.history[history_index],
        accepted_association=None,
    )
    index = len(track.history) + history_index
    track = replace(
        track,
        history=track.history[:index] + (sample,) + track.history[index + 1 :],
    )

    with pytest.raises(
        VisualRecoveryRefusal,
        match="association provenance is absent",
    ):
        _admit_stable_trace(track, transition)


def test_low_association_inside_four_sample_stable_tail_refuses():
    track, transition = _stable_tail_fixture()
    low_authority = replace(
        track.history[-3],
        association_confidence=(
            RECOVERY_MIN_ASSOCIATION_CONFIDENCE - 0.01
        ),
    )
    track = replace(
        track,
        history=(
            track.history[:-3]
            + (low_authority,)
            + track.history[-2:]
        ),
    )

    with pytest.raises(
        VisualRecoveryRefusal,
        match="history confidence is insufficient",
    ):
        require_transition_recovery_admission(
            track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=_STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000,
        )


def test_four_sample_stable_tail_requires_ninety_milliseconds():
    track, transition = _stable_tail_fixture()
    latest_observation_ns = track.history[-1].observation_monotonic_ns
    compressed_tail = tuple(
        replace(
            sample,
            observation_monotonic_ns=(
                latest_observation_ns - (3 - index) * 20_000_000
            ),
            publication_monotonic_ns=(
                latest_observation_ns
                - (3 - index) * 20_000_000
                + 1_000_000
            ),
        )
        for index, sample in enumerate(track.history[-4:])
    )
    track = replace(
        track,
        history=track.history[:-4] + compressed_tail,
    )

    with pytest.raises(
        VisualRecoveryRefusal,
        match="clean history span is insufficient",
    ):
        require_transition_recovery_admission(
            track,
            transition,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            now_monotonic_ns=_STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000,
        )


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
            "clean live provenance",
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
                    history=track.history[:-3]
                    + (
                        replace(
                            track.history[-3],
                            token=replace(
                                track.history[-3].token,
                                publication_sequence=171,
                            ),
                        ),
                    )
                    + track.history[-2:],
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


def test_continuation_revalidates_the_frozen_reacquisition_bridge():
    anchor, transition = _stable_tail_fixture()
    previous = anchor.history[-1]
    assert previous.publication_monotonic_ns is not None
    latest = replace(
        previous,
        tracker_frame_sequence=previous.tracker_frame_sequence + 1,
        token=replace(
            previous.token,
            frame_id=previous.token.frame_id + 1,
            publication_sequence=previous.token.publication_sequence + 1,
        ),
        observation_monotonic_ns=(
            previous.observation_monotonic_ns + 33_000_000
        ),
        publication_monotonic_ns=(
            previous.publication_monotonic_ns + 33_000_000
        ),
        center_norm=(0.53, -0.63),
        bbox_norm=(
            previous.bbox_norm[0] - 0.0021875,
            previous.bbox_norm[1] + 0.00722222222222222,
            previous.bbox_norm[2] - 0.0021875,
            previous.bbox_norm[3] + 0.00722222222222222,
        ),
        apparent_scale=0.18,
        confidence=0.78,
        association_confidence=0.94,
        accepted_association=None,
    )
    latest = _accepted_sample(previous, latest)
    history = anchor.history + (latest,)
    track = replace(
        anchor,
        latest_token=latest.token,
        center_norm=latest.center_norm,
        bbox_norm=latest.bbox_norm,
        apparent_scale=latest.apparent_scale,
        center_velocity_norm_s=(-0.10, 0.15),
        log_scale_rate_s=0.30,
        confidence=latest.confidence,
        association_confidence=latest.association_confidence,
        consecutive_frame_count=anchor.consecutive_frame_count + 1,
        total_observation_count=anchor.total_observation_count + 1,
        history=history,
    )
    started_ns = _STABLE_TAIL_RACE_RECEIVED_NS + 1_000_000
    now_ns = latest.publication_monotonic_ns + 1_000_000

    admission = require_recovery_continuation(
        track,
        transition,
        previous_token=previous.token,
        tracker_time_basis_id="host-perf-counter",
        measured_pitch_rad=-0.04001,
        recovery_started_monotonic_ns=started_ns,
        now_monotonic_ns=now_ns,
    )
    assert type(admission.reacquisition_bridge) is ReacquisitionBridgeAdmission

    bridge_index = (
        transition.history_length_after_promotion
        - len(transition.pretransition_frame_tokens)
    )
    bridge = track.history[bridge_index]
    assert bridge.accepted_association is not None
    forged_bridge = replace(
        bridge,
        accepted_association=replace(
            bridge.accepted_association,
            bbox_iou=0.60,
            predicted_center_residual_norm=0.058,
        ),
    )
    forged_track = replace(
        track,
        history=(
            track.history[:bridge_index]
            + (forged_bridge,)
            + track.history[bridge_index + 1 :]
        ),
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="history digest changed after promotion",
    ):
        require_recovery_continuation(
            forged_track,
            transition,
            previous_token=previous.token,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            recovery_started_monotonic_ns=started_ns,
            now_monotonic_ns=now_ns,
        )

    bridge = replace(
        bridge,
        accepted_association=replace(
            bridge.accepted_association,
            track_ambiguous_before_association=True,
        ),
    )
    track = replace(
        track,
        history=(
            track.history[:bridge_index]
            + (bridge,)
            + track.history[bridge_index + 1 :]
        ),
    )
    with pytest.raises(
        VisualRecoveryRefusal,
        match="association provenance is inconsistent",
    ):
        require_recovery_continuation(
            track,
            transition,
            previous_token=previous.token,
            tracker_time_basis_id="host-perf-counter",
            measured_pitch_rad=-0.04001,
            recovery_started_monotonic_ns=started_ns,
            now_monotonic_ns=now_ns,
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
