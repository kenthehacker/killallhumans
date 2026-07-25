from __future__ import annotations

from dataclasses import replace

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualDetection,
    VisualDetectionFrame,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    GateRelationshipBasis,
    GateGraphSnapshot,
    RollingVisualGateGraph,
)
from planning.vq2_visual_approach import (
    MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S,
    MAX_PASSAGE_SUSPENSION_EPOCHS,
    MAX_PASSAGE_SUSPENSION_FRESH_FRAMES,
    MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S,
    MAX_PASSAGE_SUSPENSION_TOTAL_FRESH_FRAMES,
    RollingVisualApproachServo,
    VISUAL_PASSAGE_ADMISSION_BASIS,
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
    VisualApproachPassageAdmission,
    VisualApproachPassageLease,
    VisualApproachPassageSafetyUnavailable,
    VisualApproachRefusal,
)


_FRAME_PERIOD_NS = 33_000_000
_BASE_OBSERVATION_NS = 10_000_000_000
_HOST_CLOCK_ID = "host-perf-counter"
_CONFIGURED_NEXT_BLEND = 0.25


def _detection(
    source_index: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    *,
    confidence: float = 0.9,
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
) -> VisualDetection:
    center_unit_x = 0.5 * (center_x + 1.0)
    center_unit_y = 0.5 * (center_y + 1.0)
    left = max(0.0, center_unit_x - width / 2.0)
    right = min(1.0, center_unit_x + width / 2.0)
    top = max(0.0, center_unit_y - height / 2.0)
    bottom = min(1.0, center_unit_y + height / 2.0)
    return VisualDetection(
        source_index=source_index,
        center_norm=(center_x, center_y),
        bbox_norm=(left, top, right, bottom),
        confidence=confidence,
        clipping=clipping,
        center_censored=center_censored,
    )


def _frame(
    sequence: int,
    detections: tuple[VisualDetection, ...],
) -> VisualDetectionFrame:
    observation_ns = _BASE_OBSERVATION_NS + sequence * _FRAME_PERIOD_NS
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=1,
            frame_id=1_000 + sequence,
            publication_sequence=sequence,
            stream_id="vq2-camera",
        ),
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        time_basis_id=_HOST_CLOCK_ID,
        image_size_px=(640, 360),
        detections=detections,
        camera_source_time_ns=20_000_000_000 + sequence * _FRAME_PERIOD_NS,
        final_unique_packet_monotonic_ns=observation_ns,
        publish_monotonic_ns=observation_ns + 1_000_000,
    )


def _race(
    received_ns: int,
    *,
    active_gate_index: int = 0,
) -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef.live(
        session_id="visual-approach-test",
        reset_epoch=1,
        race_generation=1,
        race_status_sequence=1,
        race_status_boot_ms=5_000,
        active_gate_index=active_gate_index,
        received_monotonic_ns=received_ns,
        host_clock_id=_HOST_CLOCK_ID,
    )


def _current_detection(
    *,
    center_y: float = 0.0,
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
) -> VisualDetection:
    return _detection(
        0,
        0.0,
        center_y,
        0.32,
        0.34,
        clipping=clipping,
        center_censored=center_censored,
    )


def _next_detection(
    *,
    center_x: float = 0.30,
    source_index: int = 1,
) -> VisualDetection:
    return _detection(source_index, center_x, 0.0, 0.14, 0.16)


def _build_bound_graph(
    *,
    include_next: bool = True,
    current_center_y: float = 0.0,
    current_gate_index: int = 0,
) -> tuple[
    MultiTargetVisualTracker,
    RollingVisualGateGraph,
    GateGraphSnapshot,
    str,
    str | None,
    int,
]:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    next_id: str | None = None
    snapshot: GateGraphSnapshot | None = None
    final_sequence = 5 if include_next else 3
    for sequence in range(1, final_sequence + 1):
        detections = (
            _current_detection(center_y=current_center_y),
        )
        if include_next:
            detections += (_next_detection(),)
        update = tracker.update(_frame(sequence, detections))
        if sequence == 1:
            current_id = update.visible_track_ids[0]
            if include_next:
                next_id = update.visible_track_ids[1]
        if sequence == 3:
            assert update.publish_monotonic_ns is not None
            snapshot = graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_race(
                    update.publish_monotonic_ns + 1_000_000,
                    active_gate_index=current_gate_index,
                ),
            )
        elif sequence > 3:
            snapshot = graph.observe(tracker)
    assert snapshot is not None
    return tracker, graph, snapshot, current_id, next_id, final_sequence


def _advance(
    tracker: MultiTargetVisualTracker,
    graph: RollingVisualGateGraph,
    sequence: int,
    *,
    include_next: bool = True,
    include_competing_next: bool = False,
    include_provisional: bool = False,
    next_center_x: float = 0.30,
    current_center_y: float = 0.0,
    current_clipping: FrameEdge = FrameEdge.NONE,
    current_center_censored: bool = False,
) -> GateGraphSnapshot:
    detections = (
        _current_detection(
            center_y=current_center_y,
            clipping=current_clipping,
            center_censored=current_center_censored,
        ),
    )
    if include_next:
        detections += (_next_detection(center_x=next_center_x),)
    if include_competing_next:
        detections += (
            _next_detection(center_x=-0.30, source_index=2),
        )
    if include_provisional:
        detections += (
            _detection(2, -0.55, 0.20, 0.12, 0.12),
        )
    tracker.update(_frame(sequence, detections))
    return graph.observe(tracker)


def _now_s(tracker: MultiTargetVisualTracker) -> float:
    update = tracker.latest_update
    assert update is not None
    return update.observation_monotonic_ns / 1_000_000_000.0 + 0.005


def _observe(
    approach: RollingVisualApproachServo,
    snapshot: GateGraphSnapshot,
    tracker: MultiTargetVisualTracker,
    *,
    mode: VisualApproachMode = VisualApproachMode.APPROACH,
    passage_admission: VisualApproachPassageAdmission | None = None,
):
    return approach.observe(
        snapshot,
        tracker,
        now_monotonic_s=_now_s(tracker),
        segment_elapsed_s=0.5,
        segment_yaw_excursion_rad=0.0,
        mode=mode,
        passage_admission=passage_admission,
    )


def _approach(
    current_track_id: str,
    *,
    current_gate_index: int = 0,
) -> RollingVisualApproachServo:
    return RollingVisualApproachServo(
        current_track_id,
        current_gate_index,
        next_gate_blend=_CONFIGURED_NEXT_BLEND,
    )


def test_stable_exact_next_track_starts_only_after_narrow_corridor() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = _approach(current_id)

    proposal = _observe(approach, snapshot, tracker)
    assert proposal.mode is VisualApproachMode.APPROACH
    assert proposal.passage_admission is None
    assert proposal.servo_output.next_gate_blend == 0.0
    assert proposal.servo_output.corridor_frames == 1
    assert not proposal.servo_output.advance_enabled
    assert (
        proposal.withholding_reason
        == "current_passage_corridor_not_ready"
    )

    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)

    assert proposal.current_target.track_id == current_id
    assert proposal.next_target is not None
    assert proposal.next_target.track_id == next_id
    assert (
        proposal.next_target.frame_token
        == proposal.current_target.frame_token
    )
    assert proposal.candidate_track_ids == (next_id,)
    assert proposal.provisional_track_ids == ()
    assert proposal.relationship_basis is GateRelationshipBasis.SIMULTANEOUS_IMAGE
    assert (
        proposal.servo_output.next_gate_blend
        == _CONFIGURED_NEXT_BLEND
    )
    assert not proposal.servo_output.advance_enabled
    assert proposal.withholding_reason is None
    assert proposal.latched_next_track_id == next_id
    assert approach.latched_next_track_id == next_id
    assert not proposal.servo_output.advance_enabled
    assert type(proposal.passage_admission) is VisualApproachPassageAdmission


def test_next_preview_honors_current_scale_ramp_and_exact_attempt_five_frame():
    approach = RollingVisualApproachServo(
        "vq2-track-000001",
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
    )

    assert approach._requested_next_gate_blend(-1.81) == 0.0
    assert approach._requested_next_gate_blend(-0.50) == 0.35
    assert approach._requested_next_gate_blend(-0.40) == 0.35
    assert approach._requested_next_gate_blend(
        -1.7610061763715115
    ) == pytest.approx(0.0104983371307469)

    current_x = -0.009375000000000022
    current_x_rate = -0.0637333103798879
    next_x = 0.31562500000000004
    next_x_rate = -0.0238675148682927
    blend = approach._requested_next_gate_blend(-1.7610061763715115)
    effective_x = (1.0 - blend) * current_x + blend * next_x
    effective_x_rate = (
        (1.0 - blend) * current_x_rate + blend * next_x_rate
    )
    yaw_request = -0.30 * effective_x - 0.035 * effective_x_rate

    assert effective_x == pytest.approx(-0.00596304043250727)
    assert effective_x_rate == pytest.approx(-0.06331478581862175)
    assert yaw_request == pytest.approx(0.004004929633403943)

    tracker, graph, snapshot, current_id, _next_id, sequence = (
        _build_bound_graph()
    )
    integrated = RollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
    )
    proposal = _observe(integrated, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(integrated, snapshot, tracker)
    integrated_fraction = (
        proposal.current_target.log_scale + 1.80
    ) / 1.30
    expected_integrated_blend = 0.35 * max(
        0.0,
        min(1.0, integrated_fraction),
    )

    assert proposal.servo_output.next_gate_blend == pytest.approx(
        expected_integrated_blend
    )
    assert 0.0 < proposal.servo_output.next_gate_blend < 0.35


def test_generic_gate_seven_passage_uses_admission_without_preview() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph(current_gate_index=7)
    )
    assert next_id is not None
    approach = _approach(current_id, current_gate_index=7)

    proposal = _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)

    admission = proposal.passage_admission
    assert type(admission) is VisualApproachPassageAdmission
    assert admission.basis == VISUAL_PASSAGE_ADMISSION_BASIS
    assert admission.current_gate_index == 7
    assert admission.current_target == proposal.current_target
    assert admission.camera_token == snapshot.latest_camera_token
    assert (
        admission.tracker_frame_sequence
        == snapshot.tracker_frame_sequence
    )
    assert admission.preview_track_id == next_id
    assert admission.preview_blend == _CONFIGURED_NEXT_BLEND
    assert proposal.mode is VisualApproachMode.APPROACH
    assert not proposal.servo_output.advance_enabled

    sequence += 1
    snapshot = _advance(tracker, graph, sequence)
    passage = _observe(
        approach,
        snapshot,
        tracker,
        mode=VisualApproachMode.PASSAGE,
        passage_admission=admission,
    )

    assert passage.mode is VisualApproachMode.PASSAGE
    assert passage.current_target.track_id == current_id
    assert passage.next_target is not None
    assert passage.next_target.track_id == next_id
    assert passage.candidate_track_ids == (next_id,)
    assert passage.passage_admission == admission
    assert passage.servo_output.advance_enabled
    assert passage.servo_output.next_gate_blend == 0.0
    assert passage.servo_output.next_horizontal_error is None
    assert passage.servo_output.next_vertical_error_image_down is None
    assert (
        passage.withholding_reason
        == "passage_advance_excludes_next_preview"
    )

    sequence += 1
    snapshot = _advance(tracker, graph, sequence)
    continued = _observe(
        approach,
        snapshot,
        tracker,
        mode=VisualApproachMode.PASSAGE,
        passage_admission=admission,
    )
    assert continued.servo_output.advance_enabled
    assert continued.passage_admission == admission

    with pytest.raises(VisualApproachRefusal, match="cannot return"):
        _observe(approach, snapshot, tracker)


def test_passage_requires_latest_exact_reviewed_admission() -> None:
    tracker, graph, snapshot, current_id, _, sequence = _build_bound_graph(
        include_next=False,
        current_gate_index=4,
    )
    approach = _approach(current_id, current_gate_index=4)

    with pytest.raises(
        VisualApproachRefusal,
        match="requires exact reviewed admission",
    ):
        _observe(
            approach,
            snapshot,
            tracker,
            mode=VisualApproachMode.PASSAGE,
        )

    proposal = _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 3):
        snapshot = _advance(
            tracker,
            graph,
            sequence,
            include_next=False,
        )
        proposal = _observe(approach, snapshot, tracker)

    prior_admission = proposal.passage_admission
    assert type(prior_admission) is VisualApproachPassageAdmission

    sequence += 1
    snapshot = _advance(
        tracker,
        graph,
        sequence,
        include_next=False,
    )
    proposal = _observe(approach, snapshot, tracker)
    admission = proposal.passage_admission
    assert type(admission) is VisualApproachPassageAdmission
    assert admission != prior_admission

    with pytest.raises(
        VisualApproachRefusal,
        match="latest reviewed evidence",
    ):
        _observe(
            approach,
            snapshot,
            tracker,
            mode=VisualApproachMode.PASSAGE,
            passage_admission=prior_admission,
        )

    forged = replace(admission, current_gate_index=5)
    with pytest.raises(
        VisualApproachRefusal,
        match="latest reviewed evidence",
    ):
        _observe(
            approach,
            snapshot,
            tracker,
            mode=VisualApproachMode.PASSAGE,
            passage_admission=forged,
        )

    sequence += 1
    snapshot = _advance(
        tracker,
        graph,
        sequence,
        include_next=False,
    )
    passage = _observe(
        approach,
        snapshot,
        tracker,
        mode=VisualApproachMode.PASSAGE,
        passage_admission=admission,
    )
    assert passage.servo_output.advance_enabled
    assert passage.servo_output.next_gate_blend == 0.0


def test_passage_survives_next_only_identity_ambiguity() -> None:
    tracker, graph, snapshot, current_id, _, sequence = _build_bound_graph(
        current_gate_index=4,
    )
    approach = _approach(current_id, current_gate_index=4)

    proposal = _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    admission = proposal.passage_admission
    assert type(admission) is VisualApproachPassageAdmission

    sequence += 1
    snapshot = _advance(tracker, graph, sequence)
    passage = _observe(
        approach,
        snapshot,
        tracker,
        mode=VisualApproachMode.PASSAGE,
        passage_admission=admission,
    )
    assert passage.servo_output.advance_enabled

    saw_withheld_identity = False
    for sequence in range(sequence + 1, sequence + 5):
        snapshot = _advance(
            tracker,
            graph,
            sequence,
            include_competing_next=True,
        )
        passage = _observe(
            approach,
            snapshot,
            tracker,
            mode=VisualApproachMode.PASSAGE,
            passage_admission=admission,
        )
        assert passage.servo_output.advance_enabled
        assert passage.servo_output.next_gate_blend == 0.0
        assert passage.servo_output.next_horizontal_error is None
        if passage.withholding_reason == "passage_next_identity_withheld":
            saw_withheld_identity = True

    assert saw_withheld_identity


def test_current_outside_narrow_start_corridor_withholds_next_blend() -> None:
    tracker, _, snapshot, current_id, next_id, _ = _build_bound_graph(
        current_center_y=-0.27,
    )
    assert next_id is not None
    approach = _approach(current_id)

    proposal = _observe(approach, snapshot, tracker)

    assert proposal.next_target is not None
    assert proposal.next_target.track_id == next_id
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert (
        proposal.withholding_reason
        == "current_passage_corridor_not_ready"
    )
    assert proposal.latched_next_track_id is None


def test_latched_next_continues_through_broader_passage_corridor() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = _approach(current_id)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert proposal.latched_next_track_id == next_id

    for step_index in range(1, 21):
        sequence += 1
        snapshot = _advance(
            tracker,
            graph,
            sequence,
            current_center_y=-0.01 * step_index,
        )
        proposal = _observe(approach, snapshot, tracker)

    assert abs(proposal.current_target.normalized_y_down) > 0.18
    assert (
        proposal.servo_output.next_gate_blend
        == _CONFIGURED_NEXT_BLEND
    )
    assert proposal.latched_next_track_id == next_id
    assert not proposal.servo_output.advance_enabled


def test_latched_current_passage_violation_is_specialized_refusal() -> None:
    tracker, graph, snapshot, current_id, _, sequence = (
        _build_bound_graph()
    )
    approach = _approach(current_id)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert proposal.latched_next_track_id is not None

    snapshot = _advance(
        tracker,
        graph,
        sequence + 1,
        current_center_y=-0.27,
    )
    with pytest.raises(
        VisualApproachPassageSafetyUnavailable,
        match="retired passage authority",
    ):
        _observe(approach, snapshot, tracker)


def test_passage_lease_resumes_exact_latest_two_frame_excursion() -> None:
    """Regress exact publication provenance from the latest live handoff."""

    lease = VisualApproachPassageLease()
    tokens = tuple(
        CameraFrameToken(
            stream_id="vq2-camera-udp-5600",
            generation=1,
            frame_id=2_426_752 + publication,
            publication_sequence=publication,
        )
        for publication in (116, 117, 118, 119)
    )

    initial = lease.observe(
        tokens[0],
        observation_monotonic_s=20.0,
        passage_safe=True,
        blend_active=True,
    )
    first = lease.observe(
        tokens[1],
        observation_monotonic_s=20.033,
        passage_safe=False,
        blend_active=False,
    )
    second = lease.observe(
        tokens[2],
        observation_monotonic_s=20.066,
        passage_safe=False,
        blend_active=False,
    )
    resumed = lease.observe(
        tokens[3],
        observation_monotonic_s=20.099,
        passage_safe=True,
        blend_active=True,
    )

    assert MAX_PASSAGE_SUSPENSION_FRESH_FRAMES == 2
    assert MAX_PASSAGE_SUSPENSION_TOTAL_FRESH_FRAMES == 4
    assert MAX_PASSAGE_SUSPENSION_EPOCHS == 3
    assert MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S == 0.12
    assert MAX_PASSAGE_SUSPENSION_TOTAL_DURATION_S == 0.20
    assert initial.suspension_streak == 0
    assert first.suspension_streak == 1
    assert second.suspension_streak == 2
    assert not first.retirement_required
    assert not second.retirement_required
    assert resumed.camera_token == tokens[3]
    assert resumed.recovered
    assert resumed.resumed
    assert resumed.suspension_streak == 0
    assert resumed.total_suspended_fresh_frames == 2
    assert resumed.suspension_epoch_count == 1
    assert resumed.suspension_epoch_duration_s == pytest.approx(0.066)
    assert resumed.total_suspension_duration_s == pytest.approx(0.066)
    assert resumed.resume_count == 1
    assert not resumed.retirement_required


def test_passage_lease_retires_on_third_fresh_refusal_and_never_revives() -> None:
    lease = VisualApproachPassageLease()
    states = tuple(
        lease.observe(
            CameraFrameToken(
                stream_id="vq2-camera-udp-5600",
                generation=4,
                frame_id=900 + publication,
                publication_sequence=publication,
            ),
            observation_monotonic_s=publication * 0.033,
            passage_safe=False,
            blend_active=False,
        )
        for publication in (1, 2, 3)
    )
    after_retirement = lease.observe(
        CameraFrameToken(
            stream_id="vq2-camera-udp-5600",
            generation=4,
            frame_id=904,
            publication_sequence=4,
        ),
        observation_monotonic_s=0.132,
        passage_safe=True,
        blend_active=False,
    )

    assert [state.suspension_streak for state in states] == [1, 2, 3]
    assert [state.retirement_required for state in states] == [
        False,
        False,
        True,
    ]
    assert states[-1].retirement_reason == (
        "consecutive_fresh_frames_exhausted"
    )
    assert after_retirement.retirement_required
    assert after_retirement.retirement_reason == (
        "consecutive_fresh_frames_exhausted"
    )
    assert not after_retirement.recovered
    assert not after_retirement.resumed
    assert after_retirement.resume_count == 0

    with pytest.raises(VisualApproachRefusal, match="cannot reactivate"):
        lease.observe(
            CameraFrameToken(
                stream_id="vq2-camera-udp-5600",
                generation=4,
                frame_id=905,
                publication_sequence=5,
            ),
            observation_monotonic_s=0.165,
            passage_safe=True,
            blend_active=True,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda token: token,
        lambda token: replace(token, publication_sequence=116),
        lambda token: replace(token, generation=2),
        lambda token: replace(token, stream_id="different-camera"),
    ),
)
def test_passage_lease_rejects_replayed_or_cross_stream_token(mutation) -> None:
    lease = VisualApproachPassageLease()
    token = CameraFrameToken(
        stream_id="vq2-camera-udp-5600",
        generation=1,
        frame_id=2_426_868,
        publication_sequence=116,
    )
    lease.observe(
        token,
        observation_monotonic_s=1.0,
        passage_safe=True,
        blend_active=True,
    )

    with pytest.raises(VisualApproachRefusal, match="strictly advance"):
        lease.observe(
            mutation(token),
            observation_monotonic_s=1.033,
            passage_safe=False,
            blend_active=False,
        )


def test_passage_lease_never_allows_blend_on_unsafe_publication() -> None:
    lease = VisualApproachPassageLease()
    with pytest.raises(VisualApproachRefusal, match="cannot retain"):
        lease.observe(
            CameraFrameToken(
                stream_id="vq2-camera-udp-5600",
                generation=1,
                frame_id=2_426_869,
                publication_sequence=117,
            ),
            observation_monotonic_s=1.0,
            passage_safe=False,
            blend_active=True,
        )


def test_passage_lease_retires_alternating_epochs_at_whole_segment_cap() -> None:
    """The exact replay pattern cannot manufacture an unlimited lease."""

    lease = VisualApproachPassageLease()

    def observe(publication: int, *, safe: bool):
        return lease.observe(
            CameraFrameToken(
                stream_id="vq2-camera-udp-5600",
                generation=1,
                frame_id=2_426_752 + publication,
                publication_sequence=publication,
            ),
            observation_monotonic_s=(
                20.0 + 0.033 * (publication - 116)
            ),
            passage_safe=safe,
            blend_active=safe,
        )

    observe(116, safe=True)
    for publication, safe in (
        (117, False),
        (118, False),
        (119, True),
        (122, False),
        (123, True),
        (124, False),
        (125, True),
    ):
        state = observe(publication, safe=safe)
        assert not state.retirement_required

    assert state.total_suspended_fresh_frames == 4
    assert state.suspension_epoch_count == 3
    exhausted = observe(153, safe=False)
    assert exhausted.retirement_required
    assert exhausted.total_suspended_fresh_frames == 5
    assert exhausted.suspension_epoch_count == 4
    assert exhausted.retirement_reason == "total_fresh_frames_exhausted"


def test_passage_lease_keeps_pending_until_same_identity_blend_resumes() -> None:
    lease = VisualApproachPassageLease()

    def observe(
        publication: int,
        observation_s: float,
        *,
        safe: bool,
        blend: bool,
    ):
        return lease.observe(
            CameraFrameToken(
                stream_id="vq2-camera-udp-5600",
                generation=1,
                frame_id=3_000_000 + publication,
                publication_sequence=publication,
            ),
            observation_monotonic_s=observation_s,
            passage_safe=safe,
            blend_active=blend,
        )

    observe(200, 30.000, safe=True, blend=True)
    suspended = observe(201, 30.033, safe=False, blend=False)
    no_next = observe(202, 30.066, safe=True, blend=False)
    still_no_next = observe(203, 30.077, safe=True, blend=False)
    resumed = observe(204, 30.099, safe=True, blend=True)

    assert suspended.suspension_streak == 1
    assert no_next.recovered
    assert not no_next.resumed
    assert no_next.suspension_streak == 1
    assert not still_no_next.recovered
    assert not still_no_next.resumed
    assert still_no_next.suspension_streak == 1
    assert resumed.recovered
    assert resumed.resumed
    assert resumed.suspension_streak == 0
    assert resumed.resume_count == 1
    assert resumed.total_suspension_duration_s == pytest.approx(0.066)


def test_passage_lease_retires_reentry_after_epoch_wall_duration() -> None:
    lease = VisualApproachPassageLease()
    first = CameraFrameToken(
        stream_id="vq2-camera-udp-5600",
        generation=1,
        frame_id=400,
        publication_sequence=1,
    )
    second = replace(first, frame_id=401, publication_sequence=2)
    lease.observe(
        first,
        observation_monotonic_s=1.0,
        passage_safe=False,
        blend_active=False,
    )
    retired = lease.observe(
        second,
        observation_monotonic_s=(
            1.0 + MAX_PASSAGE_SUSPENSION_EPOCH_DURATION_S + 0.001
        ),
        passage_safe=True,
        blend_active=True,
    )

    assert retired.retirement_required
    assert not retired.resumed
    assert retired.retirement_reason == (
        "suspension_epoch_duration_exhausted"
    )


@pytest.mark.parametrize(
    "token",
    (
        CameraFrameToken(generation=1, frame_id=1),
        CameraFrameToken(
            generation=1,
            frame_id=1,
            publication_sequence=1,
        ),
    ),
)
def test_passage_lease_rejects_partial_live_provenance(
    token: CameraFrameToken,
) -> None:
    with pytest.raises(
        VisualApproachRefusal,
        match="live publication provenance",
    ):
        VisualApproachPassageLease().observe(
            token,
            observation_monotonic_s=1.0,
            passage_safe=False,
            blend_active=False,
        )


@pytest.mark.parametrize(
    "value",
    (True, float("nan"), -0.01, 0.36),
)
def test_next_blend_configuration_is_bounded_and_exact(value: object) -> None:
    with pytest.raises(
        VisualApproachRefusal,
        match="next_gate_blend",
    ):
        RollingVisualApproachServo(
            "vq2-track-000001",
            0,
            next_gate_blend=value,
        )


@pytest.mark.parametrize(
    ("start", "full"),
    (
        (None, -0.5),
        (-1.8, None),
        (True, -0.5),
        (-1.8, float("nan")),
        (-0.9, -0.5),
        (-1.8, -1.1),
        (-1.8, -1.8),
    ),
)
def test_next_blend_scale_ramp_is_bounded_and_complete(start, full):
    with pytest.raises(VisualApproachRefusal, match="scale ramp"):
        RollingVisualApproachServo(
            "vq2-track-000001",
            0,
            next_gate_blend=_CONFIGURED_NEXT_BLEND,
            next_gate_blend_start_log_scale=start,
            next_gate_blend_full_log_scale=full,
        )


def test_graph_next_identity_ambiguity_refuses_authority() -> None:
    tracker, _, snapshot, current_id, _, _ = _build_bound_graph()
    approach = _approach(current_id)

    with pytest.raises(VisualApproachRefusal, match="ambiguous"):
        _observe(
            approach,
            replace(snapshot, next_selection_ambiguous=True),
            tracker,
        )


def test_zero_preview_authority_ignores_next_identity_ambiguity():
    tracker, _, snapshot, current_id, _, _ = _build_bound_graph()
    approach = RollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.0,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
    )

    proposal = _observe(
        approach,
        replace(snapshot, next_selection_ambiguous=True),
        tracker,
    )

    assert proposal.current_target.track_id == current_id
    assert proposal.servo_output.next_gate_blend == 0.0
    assert proposal.servo_output.advance_enabled is False


def test_provisional_contour_withholds_blend_without_changing_latch() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = _approach(current_id)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert (
        proposal.servo_output.next_gate_blend
        == _CONFIGURED_NEXT_BLEND
    )
    assert proposal.latched_next_track_id == next_id

    snapshot = _advance(
        tracker,
        graph,
        sequence + 1,
        include_provisional=True,
    )
    proposal = _observe(approach, snapshot, tracker)

    assert proposal.next_target is None
    assert len(proposal.provisional_track_ids) == 1
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert (
        proposal.withholding_reason
        == "provisional_next_identity_unresolved"
    )
    assert proposal.latched_next_track_id == next_id


def test_stale_qpc_observation_refuses_authority() -> None:
    tracker, _, snapshot, current_id, _, _ = _build_bound_graph()
    update = tracker.latest_update
    assert update is not None
    approach = _approach(current_id)

    with pytest.raises(VisualApproachRefusal, match="stale"):
        approach.observe(
            snapshot,
            tracker,
            now_monotonic_s=(
                update.observation_monotonic_ns / 1_000_000_000.0
                + 0.101
            ),
            segment_elapsed_s=0.5,
            segment_yaw_excursion_rad=0.0,
        )


def test_clipped_or_censored_current_aperture_refuses_authority() -> None:
    tracker, graph, _, current_id, _, sequence = _build_bound_graph()
    snapshot = _advance(
        tracker,
        graph,
        sequence + 1,
        current_clipping=FrameEdge.TOP,
        current_center_censored=True,
    )
    approach = _approach(current_id)

    with pytest.raises(
        VisualApproachCurrentGeometryUnavailable,
        match="clipped or censored",
    ):
        _observe(approach, snapshot, tracker)


def test_latched_next_identity_cannot_silently_switch() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = _approach(current_id)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert proposal.latched_next_track_id == next_id

    replacement_id: str | None = None
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(
            tracker,
            graph,
            sequence,
            next_center_x=-0.50,
        )
        visible_replacements = tuple(
            track_id
            for track_id in tracker.latest_update.visible_track_ids
            if track_id != current_id
        )
        assert len(visible_replacements) == 1
        replacement_id = visible_replacements[0]
    assert replacement_id is not None
    assert replacement_id != next_id
    assert tuple(
        candidate.track_id for candidate in snapshot.next_candidates
        if candidate.promotable and tracker.track(candidate.track_id).visible
    ) == (replacement_id,)

    with pytest.raises(VisualApproachRefusal, match="identity changed"):
        _observe(approach, snapshot, tracker)


def test_latched_next_loss_withdraws_blend_without_changing_identity() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = _approach(current_id)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert proposal.latched_next_track_id == next_id

    snapshot = _advance(
        tracker,
        graph,
        sequence + 1,
        include_next=False,
    )
    proposal = _observe(approach, snapshot, tracker)

    assert proposal.next_target is None
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert proposal.withholding_reason == "latched_next_track_unavailable"
    assert proposal.latched_next_track_id == next_id


def test_no_next_candidate_remains_current_only_and_never_advances() -> None:
    tracker, _, snapshot, current_id, _, _ = _build_bound_graph(
        include_next=False
    )
    approach = _approach(current_id)

    proposal = _observe(approach, snapshot, tracker)

    assert proposal.current_target.track_id == current_id
    assert proposal.next_target is None
    assert proposal.candidate_track_ids == ()
    assert proposal.provisional_track_ids == ()
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert proposal.withholding_reason == "no_next_candidate"
    assert proposal.latched_next_track_id is None
