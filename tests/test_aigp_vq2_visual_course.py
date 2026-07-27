from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import math
from types import SimpleNamespace

import pytest

from competition.adapter import (
    AttitudeRateCommand,
    RaceActiveBoundaryChangedBeforeWire,
)
from competition.aigp_messages import RaceStatus
from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    AssociationEvidence,
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualInnerApertureGeometry,
    VisualTrackRole,
    VisualTrackSample,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateTransition,
)
from planning.vq2_visual_approach import (
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
    VisualApproachPassageAdmission,
    VisualApproachPassageSafetyUnavailable,
    VisualApproachRefusal,
)
from planning.vq2_visual_servo import (
    ImageVisualServo,
    MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD,
    MAX_VISUAL_YAW_RATE_RAD_S,
    PassageSafetyViolation,
    PassageSafetyViolationDetail,
    ServoFrameToken,
    VisualServoOutput,
    VisualTarget,
)
from scripts.aigp_vq2_visual_config import default_visual_config
from scripts import aigp_vq2_run as vq2_run
from scripts import aigp_vq2_visual_course_stage as course_stage
from scripts.aigp_vq2_visual_course_stage import (
    VISUAL_COURSE_YAW_PROFILE_SCHEMA,
    VisualCourseStageLimits,
    VisualCourseStageRuntime,
    VisualCourseYawProfile,
    run_visual_course_stage,
)
from scripts.aigp_vq2_yaw_profile import (
    YAW_CALIBRATION_PLAN_ID,
    YAW_CALIBRATION_PLAN_SHA256,
)
from scripts.aigp_vq2_yaw_profile import (
    YAW_CALIBRATION_PROFILE_ID,
    YAW_CALIBRATION_PROFILE_SHA256,
    YAW_CALIBRATION_SOURCE_COMMIT,
    load_yaw_calibration_profile,
    yaw_calibration_profile_evidence,
)


class SafetyAbort(RuntimeError):
    pass


class _Orientation:
    def to_euler(self):
        return 0.0, 0.0, 0.0


class _Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event, **payload):
        self.events.append((event, payload))


def _token(sequence: int) -> CameraFrameToken:
    return CameraFrameToken(
        generation=7,
        frame_id=100 + sequence,
        publication_sequence=sequence,
        stream_id="camera-live",
    )


def test_dynamic_continuity_seed_is_not_crossing_evidence():
    token = _token(1)
    accepted = course_stage._AcceptedVisualCommand(
        command=AttitudeRateCommand(0.0, 0.0, 0.0, 0.26),
        yaw_soft_stop_zeroed=False,
        observation_monotonic_ns=1_000,
        publication_monotonic_ns=2_000,
        wire_start_monotonic_ns=3_000,
        wire_return_monotonic_ns=4_000,
        wire_camera_token=token,
        wire_race_gate_index=0,
        publication_pinned_through_transport_return=True,
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        next_preview_collective_delta=0.0,
        dynamic_evidence={
            "schema": "aigp-vq2-dynamic-command/1",
            "dynamic_command_count": 0,
        },
    )

    assert (
        course_stage._dynamic_near_plane_wire_sample(
            accepted,
            gate_index=0,
            track_id="vq2-track-000001",
            target=SimpleNamespace(),
            clipping=FrameEdge.NONE,
        )
        is None
    )
    assert (
        course_stage._derive_approach_top_recovery_authority(
            accepted,
            gate_index=0,
            track_id="vq2-track-000001",
            raw_vertical_rate_down_s=0.0,
            requested_thrust=0.275,
            minimum_brake_pitch_rad=0.035,
            maximum_recovery_duration_s=0.12,
        )
        is None
    )


def _accepted_dynamic_near_plane_command(
    evidence: dict[str, object],
) -> course_stage._AcceptedVisualCommand:
    token = _token(2)
    return course_stage._AcceptedVisualCommand(
        command=AttitudeRateCommand(0.01, -0.02, 0.03, 0.27),
        yaw_soft_stop_zeroed=False,
        observation_monotonic_ns=1_000,
        publication_monotonic_ns=2_000,
        wire_start_monotonic_ns=3_000,
        wire_return_monotonic_ns=4_000,
        wire_camera_token=token,
        wire_race_gate_index=0,
        publication_pinned_through_transport_return=True,
        target_roll_rad=0.04,
        target_pitch_rad=-0.05,
        next_preview_collective_delta=0.0,
        dynamic_evidence=evidence,
    )


def _valid_dynamic_near_plane_evidence() -> dict[str, object]:
    return {
        "schema": "aigp-vq2-dynamic-command/1",
        "dynamic_command_count": 1,
        "gate_index": 0,
        "current_track_id": "vq2-track-000001",
        "passage_error_norm": [0.10, -0.20],
        "current_bearing_std_norm": [0.01, 0.02],
        "residual_translation_rate_norm_s": [0.30, -0.40],
        "time_to_contact_s": 0.45,
        "crossing_prediction_horizon_s": 0.45,
        "crossing_coordinate_basis": (
            course_stage.DYNAMIC_CROSSING_COORDINATE_BASIS
        ),
        "current_crossing_error_q": [0.10, -0.20],
        "crossing_rate_q_s": [0.30, -0.40],
        "predicted_crossing_error_norm": [0.235, -0.38],
        "predicted_crossing_std_norm": [0.03, 0.04],
        "crossing_allowance_norm": [0.50, 0.60],
        "crossing_swept_occupancy_norm": [0.295, 0.46],
        "predicted_crossing_clearance_norm": [0.205, 0.14],
        "terminal_crossing_occupancy_norm": [0.295, 0.46],
        "terminal_crossing_clearance_norm": [0.205, 0.14],
        "post_governor_contact_budget_s": 0.25,
        "current_censored_axes": [False, False],
        "current_bearing_rate_qualified": [True, True],
        "current_scale_rate_qualified": True,
        "current_log_scale": -0.50,
        "expansion_rate_s": 0.75,
        "current_confidence": 0.90,
        "current_ambiguous": False,
        "dropout_held": False,
        "current_visible": True,
        "current_log_scale_std": 0.05,
    }


def _dynamic_near_plane_sample(
    evidence: dict[str, object],
):
    return course_stage._dynamic_near_plane_wire_sample(
        _accepted_dynamic_near_plane_command(evidence),
        gate_index=0,
        track_id="vq2-track-000001",
        target=SimpleNamespace(
            association_confidence=0.80,
            center_censored=False,
        ),
        clipping=FrameEdge.NONE,
    )


def test_dynamic_near_plane_wire_sample_maps_crossing_prediction():
    sample = _dynamic_near_plane_sample(
        _valid_dynamic_near_plane_evidence()
    )

    assert sample is not None
    assert sample.crossing_prediction_horizon_s == pytest.approx(0.45)
    assert sample.predicted_crossing_x_norm == pytest.approx(0.235)
    assert sample.predicted_crossing_y_down_norm == pytest.approx(-0.38)
    assert sample.predicted_crossing_x_std_norm == pytest.approx(0.03)
    assert sample.predicted_crossing_y_std_norm == pytest.approx(0.04)
    assert sample.crossing_allowance_x_norm == pytest.approx(0.50)
    assert sample.crossing_allowance_y_norm == pytest.approx(0.60)
    assert sample.crossing_swept_x_occupancy_norm == pytest.approx(0.295)
    assert sample.crossing_swept_y_occupancy_norm == pytest.approx(0.46)
    assert sample.current_crossing_x_q == pytest.approx(0.10)
    assert sample.current_crossing_y_q == pytest.approx(-0.20)
    assert sample.crossing_x_q_rate_s == pytest.approx(0.30)
    assert sample.crossing_y_q_rate_s == pytest.approx(-0.40)
    assert sample.post_governor_contact_budget_s == pytest.approx(0.25)


def test_dynamic_near_plane_wire_sample_maps_1cab_terminal_window():
    evidence = _valid_dynamic_near_plane_evidence()
    evidence.update(
        {
            "time_to_contact_s": 0.7710673805867182,
            "crossing_prediction_horizon_s": 0.7710673805867182,
            "current_crossing_error_q": [
                0.1573332768127754,
                -1.3900652360978418,
            ],
            "crossing_rate_q_s": [
                0.12404522007248783,
                1.8692942335278362,
            ],
            "predicted_crossing_error_norm": [
                0.25298049972837156,
                0.05128657209432386,
            ],
            "predicted_crossing_std_norm": [
                0.05534993410482621,
                0.14059388915832344,
            ],
            "crossing_allowance_norm": [0.50, 0.45],
            "crossing_swept_occupancy_norm": [
                0.36368036793802394,
                1.5964812450733752,
            ],
            "predicted_crossing_clearance_norm": [
                0.13631963206197606,
                -1.1464812450733752,
            ],
            "terminal_crossing_occupancy_norm": [
                0.36368036793802394,
                0.33247435041097073,
            ],
            "terminal_crossing_clearance_norm": [
                0.13631963206197606,
                0.11752564958902928,
            ],
            "post_governor_contact_budget_s": 0.4793379571895635,
        }
    )

    sample = _dynamic_near_plane_sample(evidence)

    assert sample is not None
    assert sample.current_crossing_x_q == pytest.approx(
        0.1573332768127754
    )
    assert sample.current_crossing_y_q == pytest.approx(
        -1.3900652360978418
    )
    assert sample.crossing_x_q_rate_s == pytest.approx(
        0.12404522007248783
    )
    assert sample.crossing_y_q_rate_s == pytest.approx(
        1.8692942335278362
    )
    assert sample.post_governor_contact_budget_s == pytest.approx(
        0.4793379571895635
    )


def test_unqualified_dynamic_rates_cannot_become_crossing_evidence():
    evidence = _valid_dynamic_near_plane_evidence()
    evidence["current_bearing_rate_qualified"] = [True, False]
    evidence["current_scale_rate_qualified"] = False

    sample = _dynamic_near_plane_sample(evidence)

    assert sample is not None
    assert sample.ambiguous is True


@pytest.mark.parametrize(
    "missing_name",
    [
        "crossing_prediction_horizon_s",
        "current_crossing_error_q",
        "crossing_rate_q_s",
        "predicted_crossing_error_norm",
        "predicted_crossing_std_norm",
        "crossing_allowance_norm",
        "crossing_swept_occupancy_norm",
        "predicted_crossing_clearance_norm",
        "terminal_crossing_occupancy_norm",
        "terminal_crossing_clearance_norm",
        "post_governor_contact_budget_s",
    ],
)
def test_dynamic_near_plane_wire_sample_rejects_missing_crossing_prediction(
    missing_name,
):
    evidence = _valid_dynamic_near_plane_evidence()
    del evidence[missing_name]

    with pytest.raises(ValueError, match=missing_name):
        _dynamic_near_plane_sample(evidence)


@pytest.mark.parametrize(
    ("name", "malformed"),
    [
        ("crossing_prediction_horizon_s", True),
        ("crossing_prediction_horizon_s", -0.01),
        ("crossing_prediction_horizon_s", 1.21),
        ("current_crossing_error_q", [0.1, math.nan]),
        ("crossing_rate_q_s", [0.3, True]),
        ("predicted_crossing_error_norm", [0.1, math.nan]),
        ("predicted_crossing_std_norm", [0.01, -0.01]),
        ("crossing_allowance_norm", [0.50, -0.01]),
        ("terminal_crossing_occupancy_norm", [0.295, -0.01]),
        ("terminal_crossing_clearance_norm", [0.205, math.nan]),
        ("post_governor_contact_budget_s", True),
    ],
)
def test_dynamic_near_plane_wire_sample_rejects_malformed_crossing_prediction(
    name,
    malformed,
):
    evidence = _valid_dynamic_near_plane_evidence()
    evidence[name] = malformed

    with pytest.raises(ValueError, match=name):
        _dynamic_near_plane_sample(evidence)


def test_dynamic_near_plane_wire_sample_accepts_zero_allowance() -> None:
    evidence = _valid_dynamic_near_plane_evidence()
    evidence["crossing_allowance_norm"] = [0.50, 0.0]
    evidence["predicted_crossing_error_norm"] = [0.235, 0.0]
    evidence["predicted_crossing_std_norm"] = [0.03, 0.0]
    evidence["crossing_swept_occupancy_norm"] = [0.295, 0.0]
    evidence["predicted_crossing_clearance_norm"] = [0.205, 0.0]
    evidence["terminal_crossing_occupancy_norm"] = [0.295, 0.0]
    evidence["terminal_crossing_clearance_norm"] = [0.205, 0.0]

    sample = _dynamic_near_plane_sample(evidence)

    assert sample is not None
    assert sample.crossing_allowance_y_norm == 0.0


def _c25_approach_top_recovery_command():
    evidence = _valid_dynamic_near_plane_evidence()
    evidence.update(
        {
            "current_crossing_error_q": [
                0.7872345285230035,
                -1.7416199837792337,
            ],
            "crossing_rate_q_s": [
                0.1328463672794427,
                2.1763541677795883,
            ],
            "predicted_crossing_error_norm": [
                0.8870006716301947,
                -0.10720222888568531,
            ],
            "predicted_crossing_std_norm": [
                0.09250658369983762,
                0.16712723618103104,
            ],
            "crossing_allowance_norm": [0.50, 0.45],
            "crossing_swept_occupancy_norm": [
                1.7258056275019444,
                2.2968614309187667,
            ],
            "predicted_crossing_clearance_norm": [
                -1.2258056275019444,
                -1.8468614309187668,
            ],
            "camera_current_center_norm": [
                0.0599040959869629,
                -0.3384720638086744,
            ],
            "time_to_contact_s": 0.7509888689491439,
            "successor_yaw_contribution_rad": 0.0,
            "expansion_rate_s": 1.3315776589329433,
            "braking": True,
            "brake_reason": "vertical_alignment_unsettled",
            "passage_scale_ready": False,
        }
    )
    accepted = replace(
        _accepted_dynamic_near_plane_command(evidence),
        command=AttitudeRateCommand(
            0.02879612662393212,
            0.03948135293628269,
            -0.010391620866495196,
            0.30799241399874683,
        ),
        target_roll_rad=0.08288152550333001,
        target_pitch_rad=0.12,
    )
    return accepted


def test_c25_top_censor_replay_admits_only_bounded_approach_recovery():
    accepted = _c25_approach_top_recovery_command()

    authority = course_stage._derive_approach_top_recovery_authority(
        accepted,
        gate_index=0,
        track_id="vq2-track-000001",
        raw_vertical_rate_down_s=0.6039185423722361,
        requested_thrust=0.2892416792249238,
        minimum_brake_pitch_rad=0.035,
        maximum_recovery_duration_s=0.12,
    )

    assert authority is not None
    assert authority.command.anchor_camera_token == (
        accepted.wire_camera_token
    )
    assert authority.command.requested_thrust == pytest.approx(
        0.2892416792249238
    )
    assert authority.command.requested_thrust < accepted.command.thrust
    assert authority.current_vertical_q == pytest.approx(
        -1.7416199837792337
    )
    assert authority.vertical_q_rate_s == pytest.approx(
        2.1763541677795883
    )
    assert authority.vertical_endpoint_occupancy_q == pytest.approx(
        0.4414567012477474
    )
    assert authority.vertical_endpoint_occupancy_q < (
        authority.vertical_allowance_q
    )
    assert authority.thrust_settle_s == pytest.approx(
        0.1250048984921535
    )
    assert authority.post_settle_contact_budget_s > 0.54


@pytest.mark.parametrize(
    ("mutation", "raw_rate", "requested_thrust"),
    [
        (
            {"successor_yaw_contribution_rad": 0.01},
            0.6039185423722361,
            0.2892416792249238,
        ),
        (
            {
                "predicted_crossing_error_norm": [
                    0.8870006716301947,
                    0.4842752,
                ],
                "predicted_crossing_std_norm": [
                    0.09250658369983762,
                    0.1035347,
                ],
            },
            1.76237,
            0.2892416792249238,
        ),
        (
            {"time_to_contact_s": None},
            0.6039185423722361,
            0.2892416792249238,
        ),
        ({}, -0.01, 0.2892416792249238),
        ({}, 0.6039185423722361, 0.21),
    ],
)
def test_c25_top_recovery_rejects_yaw_endpoint_rate_or_collective_lag(
    mutation,
    raw_rate,
    requested_thrust,
):
    accepted = _c25_approach_top_recovery_command()
    accepted.dynamic_evidence.update(mutation)

    authority = course_stage._derive_approach_top_recovery_authority(
        accepted,
        gate_index=0,
        track_id="vq2-track-000001",
        raw_vertical_rate_down_s=raw_rate,
        requested_thrust=requested_thrust,
        minimum_brake_pitch_rad=0.035,
        maximum_recovery_duration_s=0.12,
    )

    assert authority is None


def test_dynamic_near_plane_wire_sample_rejects_inconsistent_clearance() -> None:
    evidence = _valid_dynamic_near_plane_evidence()
    evidence["predicted_crossing_clearance_norm"] = [0.205, 0.15]

    with pytest.raises(ValueError, match="clearance is inconsistent"):
        _dynamic_near_plane_sample(evidence)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("terminal_crossing_occupancy_norm", [0.295, 0.45]),
        ("terminal_crossing_clearance_norm", [0.205, 0.15]),
    ],
)
def test_dynamic_near_plane_wire_sample_rejects_inconsistent_terminal_window(
    name,
    value,
) -> None:
    evidence = _valid_dynamic_near_plane_evidence()
    evidence[name] = value

    with pytest.raises(
        ValueError,
        match="terminal crossing window is inconsistent",
    ):
        _dynamic_near_plane_sample(evidence)


def _history_sample(
    track_id: str,
    token: CameraFrameToken,
    *,
    previous_token: CameraFrameToken | None = None,
    missed_frames_before_association: int = 0,
    apparent_scale: float = math.exp(-0.50),
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
) -> VisualTrackSample:
    publication_sequence = token.publication_sequence
    assert publication_sequence is not None
    # The deterministic host starts at 0.20 s with publication 10 already
    # available, then publishes before the following 50 Hz wire slot.  Keep
    # the synthetic receiver timestamps on that same clock instead of placing
    # each publication one control period in the future.
    publication_monotonic_ns = (
        publication_sequence - 1
    ) * 20_000_000
    accepted_association = None
    if previous_token is not None:
        previous_publication_sequence = previous_token.publication_sequence
        assert previous_publication_sequence is not None
        gap_ns = (
            publication_sequence - previous_publication_sequence
        ) * 20_000_000
        accepted_association = AssociationEvidence(
            track_id=track_id,
            previous_token=previous_token,
            current_token=token,
            detection_source_index=0,
            cost=0.10,
            confidence=0.80,
            predicted_center_residual_norm=0.01,
            bbox_iou=0.90,
            log_width_change=0.0,
            log_height_change=0.0,
            log_area_residual=0.0,
            clipping_continuity=1.0,
            temporal_consistency=(
                1.0 / (missed_frames_before_association + 1)
            ),
            appearance_distance=0.0,
            ambiguous=False,
            missed_frame_count_before_association=(
                missed_frames_before_association
            ),
            observation_gap_ns=gap_ns,
            publication_gap_ns=gap_ns,
            track_ambiguous_before_association=False,
        )
    return VisualTrackSample(
        tracker_frame_sequence=publication_sequence,
        token=token,
        observation_monotonic_ns=publication_monotonic_ns,
        publication_monotonic_ns=publication_monotonic_ns,
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        camera_source_time_ns=publication_sequence * 20_000_000,
        source_index=0,
        center_norm=(0.04, 0.03),
        bbox_norm=(-0.20, -0.20, 0.20, 0.20),
        apparent_scale=apparent_scale,
        confidence=0.90,
        clipping=clipping,
        center_censored=center_censored,
        association_confidence=(
            0.90 if accepted_association is None else 0.80
        ),
        accepted_association=accepted_association,
    )


def _track(
    track_id: str,
    *,
    gate_index: int,
    visible: bool = True,
    token: CameraFrameToken,
):
    publication_sequence = token.publication_sequence
    assert publication_sequence is not None
    if publication_sequence == 1:
        history = (_history_sample(track_id, token),)
    else:
        previous_token = _token(publication_sequence - 1)
        history = (
            _history_sample(track_id, previous_token),
            _history_sample(
                track_id,
                token,
                previous_token=previous_token,
            ),
        )
    return SimpleNamespace(
        track_id=track_id,
        latest_token=token,
        visible=visible,
        ambiguous=False,
        missed_frame_count=0 if visible else 1,
        consecutive_frame_count=len(history),
        confidence=0.90,
        association_confidence=0.80,
        clipping=FrameEdge.NONE,
        center_censored=False,
        apparent_scale=math.exp(-0.50),
        role=VisualTrackRole.CURRENT,
        authoritative_gate_index=gate_index,
        center_norm=(0.04, 0.03),
        center_velocity_norm_s=(0.0, 0.0),
        history=history,
    )


def _snapshot(
    gate_index: int,
    track_id: str,
    sequence: int,
    *,
    visible: bool = True,
    race_finished: bool = False,
):
    token = _token(sequence)
    track_token = token if visible else _token(sequence - 1)
    return SimpleNamespace(
        tracker_frame_sequence=sequence,
        latest_camera_token=token,
        current_gate_index=gate_index,
        current_track_id=track_id,
        current_track=_track(
            track_id,
            gate_index=gate_index,
            visible=visible,
            token=track_token,
        ),
        authority_usable=visible and not race_finished,
        race_finished=race_finished,
    )


class _Graph:
    def __init__(self, snapshot):
        self.latest_snapshot = snapshot
        self.finish_calls = []

    def confirm_race_finished(
        self,
        tracker,
        *,
        race_status,
        camera_token_at_finish,
    ):
        self.finish_calls.append(
            (tracker, race_status, camera_token_at_finish)
        )
        current = self.latest_snapshot
        self.latest_snapshot = _snapshot(
            race_status.active_gate_index,
            current.current_track_id,
            current.latest_camera_token.publication_sequence,
            visible=False,
            race_finished=True,
        )
        return self.latest_snapshot


class _Tracker:
    time_basis_id = "host-perf-counter"

    def __init__(self, host):
        self.host = host

    def track(self, track_id):
        adjacent = getattr(self.host, "adjacent_track", None)
        if (
            track_id != self.host.current_track_id
            and adjacent is not None
            and adjacent.track_id == track_id
        ):
            return adjacent
        assert track_id == self.host.current_track_id
        return self.host.visual_gate_graph.latest_snapshot.current_track


def _target(snapshot, track_id):
    token = snapshot.latest_camera_token
    return VisualTarget(
        track_id=track_id,
        frame_token=ServoFrameToken(
            stream_id=token.stream_id,
            generation=token.generation,
            frame_id=token.frame_id,
            publication_sequence=token.publication_sequence,
        ),
        received_monotonic_s=(token.publication_sequence - 1) * 0.02,
        normalized_x=0.04,
        normalized_y_down=0.03,
        normalized_x_rate_s=0.0,
        normalized_y_rate_down_s=0.0,
        log_scale=-0.50,
        log_scale_rate_s=0.20,
        confidence=0.9,
        association_confidence=0.8,
        consecutive_frames=5,
    )


class _Servo:
    """Small deterministic stand-in for the already-focused planner tests."""

    def __init__(
        self,
        expected_current_track_id,
        expected_gate_index,
        tuning,
        *,
        next_gate_blend,
        next_gate_blend_start_log_scale=None,
        next_gate_blend_full_log_scale=None,
        yaw_rate=0.02,
        passage_advances=True,
        preview_track_id=None,
        passage_preview_blend=0.0,
        passage_preview_retire_once=False,
        calls=None,
    ):
        self.track_id = expected_current_track_id
        self.gate_index = expected_gate_index
        self.tuning = tuning
        self.next_gate_blend = next_gate_blend
        self.next_gate_blend_start_log_scale = (
            next_gate_blend_start_log_scale
        )
        self.next_gate_blend_full_log_scale = (
            next_gate_blend_full_log_scale
        )
        self.yaw_rate = yaw_rate
        self.passage_advances = passage_advances
        self.preview_track_id = (
            preview_track_id
            if preview_track_id is not None
            else f"track-{expected_gate_index + 1}"
        )
        self.passage_preview_blend = passage_preview_blend
        self.passage_preview_retire_once = (
            passage_preview_retire_once
        )
        self.passage_preview_retirement_emitted = False
        self.passage_preview_retired = False
        self.generic_passage_sample_count = 0
        self.calls = calls if calls is not None else []

    def retire_passage_preview(self, expected_track_id):
        assert expected_track_id == self.preview_track_id
        self.passage_preview_retired = True

    def observe_promotable_adjacent(
        self,
        snapshot,
        tracker,
        now_monotonic_s,
        segment_elapsed_s,
        segment_yaw_excursion_rad,
    ):
        del tracker, now_monotonic_s, segment_yaw_excursion_rad
        self.calls.append(
            (
                self.gate_index,
                VisualApproachMode.ADJACENT_RECENTER,
                None,
                False,
                segment_elapsed_s,
            )
        )
        target = replace(
            _target(snapshot, self.track_id),
            normalized_x=0.55,
            normalized_y_down=-0.55,
            normalized_x_rate_s=0.20,
            normalized_y_rate_down_s=-0.20,
            log_scale=-1.70,
        )
        output = VisualServoOutput(
            target_roll_rad=0.0,
            target_pitch_rad=0.08,
            yaw_rate_rad_s=self.yaw_rate,
            thrust=0.27,
            corridor_frames=0,
            advance_enabled=False,
            next_gate_blend=0.0,
            horizontal_error=0.55,
            vertical_error_image_down=-0.55,
            effective_horizontal_error=0.55,
            effective_vertical_error_image_down=-0.55,
            effective_horizontal_rate_s=0.20,
            effective_vertical_rate_down_s=-0.20,
            next_horizontal_error=None,
            next_vertical_error_image_down=None,
            horizontal_abs_error_delta=0.0,
            vertical_abs_error_delta=0.0,
            brake_reason="target_edge_or_clipping",
            yaw_envelope_limited=False,
        )
        return SimpleNamespace(
            current_target=target,
            next_target=None,
            servo_output=output,
            candidate_track_ids=(self.track_id,),
            provisional_track_ids=(),
            withholding_reason=None,
            relationship_basis=None,
            latched_next_track_id=None,
            mode=VisualApproachMode.ADJACENT_RECENTER,
            passage_admission=None,
        )

    def observe(
        self,
        snapshot,
        tracker,
        now_monotonic_s,
        segment_elapsed_s,
        segment_yaw_excursion_rad,
        *,
        mode,
        passage_admission,
        passage_forward_closure_authorized=True,
    ):
        del tracker, now_monotonic_s
        del segment_yaw_excursion_rad
        self.calls.append(
            (
                self.gate_index,
                mode,
                passage_admission,
                passage_forward_closure_authorized,
                segment_elapsed_s,
            )
        )
        if not snapshot.current_track.visible:
            raise VisualApproachRefusal("current identity is no longer visible")
        target = _target(snapshot, self.track_id)
        if mode is VisualApproachMode.PASSAGE:
            self.generic_passage_sample_count += 1
            target = replace(
                target,
                log_scale=(
                    -0.90
                    + 0.06 * self.generic_passage_sample_count
                ),
            )
        advance = bool(
            mode is VisualApproachMode.PASSAGE
            and self.passage_advances
            and passage_forward_closure_authorized
        )
        retirement_details = ()
        if (
            mode is VisualApproachMode.PASSAGE
            and self.passage_preview_retire_once
            and not self.passage_preview_retirement_emitted
        ):
            self.passage_preview_retirement_emitted = True
            self.passage_preview_retired = True
            retirement_details = (
                PassageSafetyViolationDetail(
                    violation=(
                        PassageSafetyViolation.CURRENT_APPARENT_SCALE
                    ),
                    observed=0.56,
                    limit=0.55,
                    excess=0.01,
                ),
            )
        active_preview_blend = (
            self.passage_preview_blend
            if not self.passage_preview_retired
            else 0.0
        )
        output = VisualServoOutput(
            target_roll_rad=0.0,
            target_pitch_rad=-0.105 if advance else 0.0,
            yaw_rate_rad_s=self.yaw_rate,
            thrust=0.295 if advance else 0.21,
            corridor_frames=5,
            advance_enabled=advance,
            next_gate_blend=(
                active_preview_blend
                if mode is VisualApproachMode.PASSAGE
                else 0.0
            ),
            horizontal_error=0.04,
            vertical_error_image_down=0.03,
            effective_horizontal_error=0.04,
            effective_vertical_error_image_down=0.03,
            effective_horizontal_rate_s=0.0,
            effective_vertical_rate_down_s=0.0,
            next_horizontal_error=(
                0.30
                if (
                    mode is VisualApproachMode.PASSAGE
                    and active_preview_blend > 0.0
                )
                else None
            ),
            next_vertical_error_image_down=(
                -0.20
                if (
                    mode is VisualApproachMode.PASSAGE
                    and active_preview_blend > 0.0
                )
                else None
            ),
            horizontal_abs_error_delta=0.0,
            vertical_abs_error_delta=0.0,
            brake_reason=None if advance else "aligning",
            yaw_envelope_limited=False,
            passage_preview_retired=self.passage_preview_retired,
            passage_preview_retirement_violations=(
                retirement_details
            ),
        )
        admission = passage_admission
        if mode is VisualApproachMode.APPROACH:
            admission = VisualApproachPassageAdmission(
                basis="tight-current-corridor-dwell-v1",
                current_gate_index=self.gate_index,
                current_target=target,
                camera_token=snapshot.latest_camera_token,
                tracker_frame_sequence=snapshot.tracker_frame_sequence,
                corridor_frames=5,
                preview_track_id=self.preview_track_id,
                preview_blend=(
                    self.next_gate_blend
                    if self.preview_track_id is not None
                    else 0.0
                ),
            )
        return SimpleNamespace(
            current_target=target,
            next_target=(
                replace(
                    target,
                    track_id=self.preview_track_id,
                    normalized_x=0.30,
                    normalized_x_rate_s=0.10,
                )
                if (
                    mode is VisualApproachMode.PASSAGE
                    and active_preview_blend > 0.0
                )
                else None
            ),
            servo_output=output,
            passage_admission=admission,
            mode=mode,
        )


class _Host:
    def __init__(
        self,
        *,
        initial_gate=3,
        finish_gate=4,
        fresh_after_samples=2,
        lose_before_credit=False,
        credit_on_approach=False,
        disable_credit=False,
    ):
        self._visual_tracking_enabled = True
        self._last_flight_command_started_ns = None
        self.clock = 0.20
        self.sequence = 10
        self.visual_config = default_visual_config()
        self.visual_tracker = _Tracker(self)
        self.current_gate = initial_gate
        self.current_track_id = f"track-{initial_gate}"
        self.visual_gate_graph = _Graph(
            _snapshot(initial_gate, self.current_track_id, self.sequence)
        )
        self.estimate = SimpleNamespace(
            orientation=_Orientation(),
            body_rates=(0.0, 0.0, 0.0),
        )
        self.recorder = _Recorder()
        self.finish_gate = finish_gate
        self.fresh_after_samples = fresh_after_samples
        self.lose_before_credit = lose_before_credit
        self.credit_on_approach = credit_on_approach
        self.disable_credit = disable_credit
        self.passage_counts = {}
        self.after_promotion_samples = None
        self.crossing_zero_count = 0
        self.crossing_hold_count = 0
        self.commands = []
        self.wire_receipts_by_command_index = []
        self.ticks = []
        self.watchdogs = 0
        self.control_ingress_samples = 0
        self.credit_token = None
        self.confirmed_race_statuses = []
        self.promotion_tokens = []
        self.requested_promotion_track_ids = []
        self.race = AuthoritativeRaceStatusRef.live(
            session_id="test-session",
            reset_epoch=2,
            race_generation=9,
            race_status_sequence=1,
            race_status_boot_ms=1000,
            active_gate_index=initial_gate,
            received_monotonic_ns=100_000_000,
            host_clock_id="host-perf-counter",
        )

    def _advance_race(self):
        if self.disable_credit or self.race.race_finished:
            return
        self.credit_token = self.visual_gate_graph.latest_snapshot.latest_camera_token
        finished = self.current_gate == self.finish_gate
        next_gate = self.current_gate if finished else self.current_gate + 1
        self.race = AuthoritativeRaceStatusRef.live(
            session_id=self.race.session_id,
            reset_epoch=self.race.reset_epoch,
            race_generation=self.race.race_generation,
            race_status_sequence=self.race.race_status_sequence + 1,
            race_status_boot_ms=self.race.race_status_boot_ms + 250,
            active_gate_index=next_gate,
            received_monotonic_ns=max(
                self.race.received_monotonic_ns + 1,
                round(self.clock * 1_000_000_000),
            ),
            host_clock_id=self.race.host_clock_id,
            race_finished=finished,
        )

    def _sample(self):
        self.sequence += 1
        visible = True
        if self.after_promotion_samples is not None:
            self.after_promotion_samples += 1
            visible = (
                self.after_promotion_samples >= self.fresh_after_samples
            )
        if (
            self.lose_before_credit
            and self.passage_counts.get(self.current_gate, 0) >= 3
            and self.race.active_gate_index == self.current_gate
        ):
            visible = False
        self.visual_gate_graph.latest_snapshot = _snapshot(
            self.current_gate,
            self.current_track_id,
            self.sequence,
            visible=visible,
        )

    def _sample_control_ingress(self):
        self.control_ingress_samples += 1

    def _watchdog(self, **kwargs):
        assert kwargs["enforce_benign_pad_budget"] is True
        assert kwargs["allow_benign_pad_contact"] is (
            self.clock < 0.55
        )
        self.watchdogs += 1

    async def _wait_for_next_flight_command_slot(self):
        if self._last_flight_command_started_ns is not None:
            self.clock = max(
                self.clock,
                self._last_flight_command_started_ns / 1e9 + 0.02,
            )
        return self.clock

    async def _send_flight_command(self, command, **kwargs):
        self.commands.append((command, dict(kwargs), self.current_gate))
        self._last_flight_command_started_ns = round(self.clock * 1e9)
        passage_wire_command = bool(
            command.thrust == 0.295
            or (
                self.current_gate == 0
                and command.pitch_rate == -0.105
                and command.thrust > 0.0
            )
        )
        if passage_wire_command:
            self.passage_counts[self.current_gate] = (
                self.passage_counts.get(self.current_gate, 0) + 1
            )
            if (
                self.lose_before_credit
                and not (
                    self.visual_gate_graph.latest_snapshot
                    .current_track.visible
                )
            ):
                self.crossing_hold_count += 1
                if self.crossing_hold_count >= 2:
                    self._advance_race()
            if (
                self.passage_counts[self.current_gate] >= 3
                and not self.lose_before_credit
            ):
                self._advance_race()
        elif command.thrust == 0.21 and self.credit_on_approach:
            self._advance_race()
        elif (
            command.roll_rate
            == command.pitch_rate
            == command.yaw_rate
            == command.thrust
            == 0.0
        ):
            self.crossing_zero_count += 1
            if self.lose_before_credit and self.crossing_zero_count >= 2:
                self._advance_race()
        if kwargs.get("require_wire_receipt"):
            receipt = {
                "call_start_monotonic_ns": (
                    self._last_flight_command_started_ns
                ),
            }
            token = kwargs.get("wire_visual_token")
            if token is not None:
                receipt["visual_receiver_authority"] = {
                    "schema": "aigp-vq2-visual-wire-authority/1",
                    "frame_token": asdict(token),
                    "call_start_monotonic_ns": (
                        self._last_flight_command_started_ns
                    ),
                    "transport_return_monotonic_ns": (
                        self._last_flight_command_started_ns
                    ),
                    "publication_pinned_through_transport_return": True,
                }
            self.wire_receipts_by_command_index.append(
                (len(self.commands) - 1, receipt)
            )
            return receipt
        return None

    def _assert_visual_receiver_token_current(self, expected_token):
        receiver_token = (
            self.visual_gate_graph.latest_snapshot.latest_camera_token
        )
        if expected_token != receiver_token:
            exc = SafetyAbort(
                course_stage.VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
            )
            exc.expected_visual_token = expected_token
            exc.receiver_visual_token = receiver_token
            raise exc
        return receiver_token

    def _visual_race_status_ref(self):
        return self.race

    def _visual_camera_token_at_race_credit(self, race_status):
        assert race_status == self.race
        return self.credit_token

    def _confirm_visual_transition(
        self,
        *,
        from_gate_index,
        to_gate_index,
        race_status,
        promoted_track_id=None,
    ):
        assert from_gate_index == self.current_gate
        assert to_gate_index == self.current_gate + 1
        assert race_status is self.race
        self.confirmed_race_statuses.append(race_status)
        self.requested_promotion_track_ids.append(promoted_track_id)
        retired = self.current_track_id
        promoted = promoted_track_id or f"track-{to_gate_index}"
        promotion_token = (
            self.visual_gate_graph.latest_snapshot.latest_camera_token
        )
        self.promotion_tokens.append(promotion_token)
        credit_token = self.credit_token
        assert credit_token is not None
        credit_sequence = credit_token.publication_sequence
        promotion_sequence = promotion_token.publication_sequence
        assert credit_sequence is not None
        assert promotion_sequence is not None
        history_length = 17
        history_length_at_credit = (
            history_length
            if promotion_sequence == credit_sequence
            else history_length - 1
        )
        transition = ConfirmedGateTransition(
            from_gate_index=from_gate_index,
            to_gate_index=to_gate_index,
            retired_track_id=retired,
            promoted_track_id=promoted,
            race_status=race_status,
            camera_token_at_credit=credit_token,
            promoted_first_token=_token(1),
            promoted_latest_token_before_credit=credit_token,
            promoted_history_length_at_credit=history_length_at_credit,
            promoted_latest_token_at_promotion=promotion_token,
            pretransition_frame_tokens=tuple(
                _token(sequence)
                for sequence in range(
                    credit_sequence - 2,
                    credit_sequence + 1,
                )
            ),
            history_length_before_promotion=history_length,
            history_length_after_promotion=history_length,
            promoted_history_sha256="a" * 64,
        )
        self.current_gate = to_gate_index
        self.current_track_id = promoted
        self.after_promotion_samples = 0
        self.visual_gate_graph.latest_snapshot = _snapshot(
            self.current_gate,
            self.current_track_id,
            promotion_token.publication_sequence,
            visible=False,
        )
        return transition

    def _confirm_visual_course_advance(self, **kwargs):
        """Return the typed course boundary used by the generic coordinator."""

        kwargs["promoted_track_id"] = kwargs.pop("reviewed_track_id")
        return self._confirm_visual_transition(**kwargs)

    def _record_tick(self, stage, elapsed_s, command):
        self.ticks.append((stage, elapsed_s, command))


def _yaw_profile():
    return VisualCourseYawProfile.load_tracked()


def _context():
    return SimpleNamespace(spawn_pitch_rad=-0.31)


def _assert_course_zero_receipts(
    host: _Host,
) -> dict[str, list[int]]:
    """Prove each coordinator handoff zero had one outbound receipt."""

    assert len(host.commands) == len(host.ticks)
    receipt_indices = [
        command_index
        for command_index, receipt
        in host.wire_receipts_by_command_index
        if isinstance(receipt, dict)
    ]
    by_phase = {
        "crossing-zero": [],
        "post-credit-zero": [],
        "credited-unbound-zero": [],
    }
    for command_index, (
        (command, kwargs, _gate_index),
        (stage, _elapsed_s, recorded_command),
    ) in enumerate(zip(host.commands, host.ticks)):
        matched_phase = next(
            (
                phase
                for phase in by_phase
                if stage.endswith(phase)
            ),
            None,
        )
        if matched_phase is None:
            continue
        assert recorded_command == command
        assert (
            command.roll_rate
            == command.pitch_rate
            == command.yaw_rate
            == command.thrust
            == 0.0
        )
        assert kwargs.get("require_wire_receipt") is True
        assert receipt_indices.count(command_index) == 1
        by_phase[matched_phase].append(command_index)
    return by_phase


def _runtime(host, *, yaw_profile=True, servo_options=None, limits=None):
    calls = []
    options = dict(servo_options or {})
    host.intercept_response_authorities = []

    def servo_factory(*args, **kwargs):
        return _Servo(*args, **kwargs, calls=calls, **options)

    async def sleep(delay):
        assert delay >= 0.0
        host.clock += delay

    def next_deadline(previous, now, period):
        return max(previous + period, now + period)

    def attitude_rate(
        _estimate,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
        intercept_response_authority=0.0,
    ):
        assert 0.0 <= intercept_response_authority <= 1.0
        host.intercept_response_authorities.append(
            intercept_response_authority
        )
        return AttitudeRateCommand(
            target_roll_rad,
            target_pitch_rad,
            0.0,
            thrust,
        )

    def limit(command, maximum):
        return AttitudeRateCommand(
            max(-maximum, min(maximum, command.roll_rate)),
            max(-maximum, min(maximum, command.pitch_rate)),
            0.0,
            command.thrust,
        )

    def validate(command):
        assert max(
            abs(command.roll_rate),
            abs(command.pitch_rate),
            abs(command.yaw_rate),
        ) <= 0.25
        assert 0.0 <= command.thrust <= 0.35

    profile = _yaw_profile() if yaw_profile else None
    runtime = VisualCourseStageRuntime(
        safety_abort_type=SafetyAbort,
        cancelled_error_type=asyncio.CancelledError,
        monotonic=lambda: host.clock,
        perf_counter_ns=lambda: round(host.clock * 1e9),
        sleep=sleep,
        next_control_deadline=next_deadline,
        attitude_rate_command=attitude_rate,
        limit_command_rates=limit,
        validate_command=validate,
        yaw_profile=profile,
        expected_yaw_profile_sha256=(
            None if profile is None else profile.profile_sha256
        ),
        limits=limits or VisualCourseStageLimits(),
        servo_factory=servo_factory,
    )
    return runtime, calls


def test_command_deadline_conversion_preserves_remaining_budget_across_clocks():
    validation_ns = 225_761_902_017_200

    deadline_ns = course_stage._perf_counter_deadline_from_monotonic(
        deadline_monotonic_s=71_313.75,
        now_monotonic_s=71_313.625,
        validation_perf_counter_ns=validation_ns,
    )

    assert deadline_ns == validation_ns + 125_000_000
    assert course_stage._perf_counter_deadline_from_monotonic(
        deadline_monotonic_s=71_313.50,
        now_monotonic_s=71_313.625,
        validation_perf_counter_ns=validation_ns,
    ) == validation_ns


def test_generic_course_repeats_lifecycle_from_nonzero_gate_until_finish():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["outcome"] == "race_finished"
    assert result["first_causal_blocker"] is None
    assert result["passage_authority_enabled"] is True
    assert result["initial_gate_index"] == 3
    assert result["maximum_authoritative_gate_index"] == 4
    assert result["final_gate_index"] == 4
    assert host.control_ingress_samples == 1
    assert [
        (item["from_gate_index"], item["to_gate_index"])
        for item in result["authoritative_transitions"]
    ] == [(3, 4)]
    assert [item["gate_index"] for item in result["segments"]] == [3, 4]
    assert all(
        item["passage_authority_enabled"] is True
        for item in result["segments"]
    )
    assert all(
        item["advance_command_count"] >= 3
        for item in result["segments"]
    )
    assert [call[1] for call in calls] == [
        VisualApproachMode.APPROACH,
        VisualApproachMode.PASSAGE,
        VisualApproachMode.PASSAGE,
        VisualApproachMode.PASSAGE,
        VisualApproachMode.PROMOTE_REACQUIRE,
        VisualApproachMode.PROMOTE_REACQUIRE,
        VisualApproachMode.APPROACH,
        VisualApproachMode.PASSAGE,
        VisualApproachMode.PASSAGE,
        VisualApproachMode.PASSAGE,
    ]
    transition = result["authoritative_transitions"][0]
    assert transition["promotion_confirmed"] is True
    assert transition["pre_transition_approach_command_count"] == 1
    assert transition["pre_transition_passage_command_count"] == 3
    assert transition["post_transition_navigation_command_count"] == 6
    assert (
        transition["recovery_admission"]["admitted_frame_token"]
        == transition["recovery_admission"]["wire_frame_token"]
    )
    recovery_segment = result["segments"][1]
    assert recovery_segment["recovery_navigation_command_count"] == 2
    assert recovery_segment["recovery_clean_command_count"] == 2
    recovery_completed = [
        fields
        for name, fields in host.recorder.events
        if name == "visual_course_recovery_completed"
    ]
    assert len(recovery_completed) == 1
    assert (
        recovery_completed[0]["camera_token"][
            "publication_sequence"
        ]
        > transition["recovery_admission"]["wire_frame_token"][
            "publication_sequence"
        ]
    )
    assert transition["passage_authority_enabled"] is True
    assert transition["history_length_before_promotion"] == (
        transition["history_length_after_promotion"]
    )
    assert host.visual_gate_graph.finish_calls
    assert host.recorder.events[-1][0] == "visual_course_complete"


def test_transition_carries_exact_reviewed_passage_preview_identity():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"preview_track_id": "reviewed-track-4"},
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert host.requested_promotion_track_ids == ["reviewed-track-4"]
    transition = result["authoritative_transitions"][0]
    assert transition["promoted_track_id"] == "reviewed-track-4"


def test_passage_preview_wire_count_is_compact_and_transition_scoped():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"passage_preview_blend": 0.20},
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment_count = sum(
        segment["passage_next_preview_command_count"]
        for segment in result["segments"]
    )
    assert segment_count > 0
    assert result["passage_next_preview_command_count"] == segment_count
    transition = result["authoritative_transitions"][0]
    assert (
        transition[
            "pre_transition_passage_next_preview_command_count"
        ]
        > 0
    )


def test_passage_preview_hard_retirement_is_recorded_and_nonfatal():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"passage_preview_retire_once": True},
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["passage_next_preview_command_count"] == 0
    for segment in result["segments"]:
        assert segment["next_preview_retired"] is True
        assert segment["next_preview_withdrawal_count"] == 1
        withdrawal = segment["next_preview_withdrawal"]
        assert withdrawal["reason"] == (
            "passage_preview_envelope_retired"
        )
        assert withdrawal["violation_codes"] == [
            "current_apparent_scale"
        ]
        assert segment["passage_command_count"] > 0
    events = [
        payload
        for event, payload in host.recorder.events
        if (
            event == "visual_course_next_preview_withdrawn"
            and payload["reason"]
            == "passage_preview_envelope_retired"
        )
    ]
    assert len(events) == len(result["segments"])


def test_nonterminal_transition_refuses_without_reviewed_preview_identity():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"preview_track_id": ""},
    )

    with pytest.raises(
        SafetyAbort,
        match="nonterminal transition lacks its reviewed next-track identity",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.requested_promotion_track_ids == []
    transition = host._visual_course_summary[
        "authoritative_transitions"
    ][0]
    assert transition["promotion_confirmed"] is False


def test_terminal_transition_completes_without_next_preview_identity():
    host = _Host(initial_gate=3, finish_gate=3, fresh_after_samples=1)
    runtime, _calls = _runtime(
        host,
        servo_options={"preview_track_id": ""},
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["maximum_authoritative_gate_index"] == 3
    assert host.requested_promotion_track_ids == []


def test_crossing_loss_latches_only_after_credible_passage_and_holds():
    host = _Host(
        initial_gate=6,
        finish_gate=7,
        fresh_after_samples=1,
        lose_before_credit=True,
    )
    runtime, _calls = _runtime(
        host,
        servo_options={
            "passage_preview_blend": 0.30,
            "yaw_rate": -0.15,
        },
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["race_finished"] is True
    assert result["segments"][0]["crossing_anchor"] is not None
    assert result["segments"][0][
        "crossing_wait_zero_command_count"
    ] == 0
    assert result["segments"][0][
        "crossing_wait_coast_command_count"
    ] == 2
    gate6_crossing_indices = [
        command_index
        for command_index, (stage, _elapsed, _command)
        in enumerate(host.ticks)
        if stage == "visual-course/gate6/credit-wait"
    ]
    first_gate7_navigation_index = next(
        command_index
        for command_index, (command, kwargs, gate_index)
        in enumerate(host.commands)
        if gate_index == 7
        and command.thrust > 0.0
        and kwargs.get("wire_visual_token") is not None
    )
    assert len(gate6_crossing_indices) == 2
    assert all(
        host.commands[index][0].thrust > 0.0
        and host.commands[index][0].pitch_rate
        == pytest.approx(host.visual_config.servo.brake_pitch_rad)
        and host.commands[index][0].yaw_rate == pytest.approx(-0.15)
        and host.commands[index][1].get("wire_visual_token") is not None
        for index in gate6_crossing_indices
    )
    assert all(
        command_index < first_gate7_navigation_index
        for command_index in gate6_crossing_indices
    )


def test_post_credit_zero_is_receipted_before_promoted_gate_navigation():
    host = _Host(
        initial_gate=3,
        finish_gate=4,
        fresh_after_samples=2,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    phase_receipts = _assert_course_zero_receipts(host)
    post_credit_indices = phase_receipts["post-credit-zero"]
    first_gate4_navigation_index = next(
        command_index
        for command_index, (command, kwargs, gate_index)
        in enumerate(host.commands)
        if gate_index == 4
        and command.thrust > 0.0
        and kwargs.get("wire_visual_token") is not None
    )
    assert post_credit_indices
    assert all(
        command_index < first_gate4_navigation_index
        for command_index in post_credit_indices
    )


def test_latched_crossing_without_authoritative_credit_times_out_bounded():
    host = _Host(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="gate 6 credit timed out"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    segment = host._visual_course_summary["segments"][0]
    assert segment["near_plane_latch"] is not None
    assert segment["crossing_wait_zero_command_count"] == 0
    max_crossing_commands = math.ceil(
        runtime.limits.crossing_status_timeout_s
        / runtime.limits.control_period_s
    )
    assert (
        1
        <= segment["crossing_wait_coast_command_count"]
        <= max_crossing_commands
    )
    assert host.race.active_gate_index == 6
    assert host.race.race_finished is False


def test_censored_near_plane_passage_transfers_to_successor_coast():
    class CoastBeforeCreditHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.allow_credit = False
            self.near_plane_coast_seen = False

        def _advance_race(self):
            if self.allow_credit:
                super()._advance_race()

        def _sample(self):
            super()._sample()
            summary = getattr(self, "_visual_course_summary", {})
            segments = summary.get("segments", ())
            if (
                segments
                and segments[-1].get("near_plane_latch") is not None
            ):
                track = self.visual_gate_graph.latest_snapshot.current_track
                track.clipping = FrameEdge.TOP | FrameEdge.BOTTOM
                track.center_censored = True
                track.center_norm = (-0.196875, 0.0)
                track.center_velocity_norm_s = (-0.5354, 0.0)

        async def _send_flight_command(self, command, **kwargs):
            if (
                self.passage_counts.get(self.current_gate, 0) >= 3
                and command.pitch_rate > 0.0
                and command.yaw_rate
                == -MAX_VISUAL_YAW_RATE_RAD_S
            ):
                self.near_plane_coast_seen = True
                self.allow_credit = True
            return await super()._send_flight_command(command, **kwargs)

    class RefuseCensoredCurrentServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            if snapshot.current_track.clipping is not FrameEdge.NONE:
                raise VisualApproachCurrentGeometryUnavailable(
                    "current aperture entered expected passage censorship"
                )
            return super().observe(snapshot, *args, **kwargs)

    host = CoastBeforeCreditHost(initial_gate=6, finish_gate=6)
    runtime, calls = _runtime(
        host,
        servo_options={"passage_preview_blend": 0.3},
    )
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: RefuseCensoredCurrentServo(
            *args,
            **kwargs,
            calls=calls,
            passage_preview_blend=0.3,
        ),
    )

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert host.near_plane_coast_seen
    coast_commands = [
        command
        for stage, _elapsed, command in host.ticks
        if stage.endswith("/censored-passage")
    ]
    assert len(coast_commands) == 1
    assert coast_commands[0].yaw_rate == -MAX_VISUAL_YAW_RATE_RAD_S
    assert coast_commands[0].pitch_rate > 0.0


@pytest.mark.parametrize("case", ("stale", "ambiguous", "off_center"))
def test_latched_unsafe_target_is_refused_without_a_wire_send(case):
    class UnsafePostLatchHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.unsafe_token = None

        def _sample(self):
            super()._sample()
            summary = getattr(self, "_visual_course_summary", {})
            segments = summary.get("segments", ())
            if (
                not segments
                or segments[-1].get("near_plane_latch") is None
            ):
                return
            snapshot = self.visual_gate_graph.latest_snapshot
            track = snapshot.current_track
            track.center_norm = (0.04, 0.03)
            track.center_velocity_norm_s = (0.0, 0.0)
            track.authoritative_gate_index = self.current_gate
            if case == "stale":
                track.latest_token = _token(self.sequence - 1)
            elif case == "ambiguous":
                track.ambiguous = True
                track.role = VisualTrackRole.AMBIGUOUS
            else:
                track.center_norm = (0.80, 0.03)
            self.unsafe_token = snapshot.latest_camera_token

    class UnsafeRefusingServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            track = snapshot.current_track
            center = getattr(track, "center_norm", (0.0, 0.0))
            if (
                track.latest_token != snapshot.latest_camera_token
                or track.ambiguous
                or abs(center[0]) > 0.50
            ):
                raise VisualApproachRefusal(
                    "unsafe current target in deterministic fixture"
                )
            return super().observe(snapshot, *args, **kwargs)

    host = UnsafePostLatchHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: UnsafeRefusingServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="latched near-plane measurement became unsafe",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.unsafe_token is not None
    sent_tokens = {
        kwargs.get("wire_visual_token")
        for _command, kwargs, _gate in host.commands
    }
    assert host.unsafe_token not in sent_tokens


@pytest.mark.parametrize(
    "case",
    (
        "schema",
        "nested_start",
        "top_start",
        "host_start",
        "frame_token",
        "not_pinned",
    ),
)
def test_visual_send_rejects_unbound_wire_timing(case):
    class UnboundReceiptHost(_Host):
        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(command, **kwargs)
            if receipt is None:
                return receipt
            authority = receipt["visual_receiver_authority"]
            if case == "schema":
                authority["schema"] = "other"
            elif case == "nested_start":
                authority["call_start_monotonic_ns"] = True
            elif case == "top_start":
                receipt["call_start_monotonic_ns"] += 1
            elif case == "host_start":
                self._last_flight_command_started_ns += 1
            elif case == "frame_token":
                authority["frame_token"] = asdict(
                    _token(
                        kwargs[
                            "wire_visual_token"
                        ].publication_sequence
                        + 1
                    )
                )
            else:
                authority[
                    "publication_pinned_through_transport_return"
                ] = False
            return receipt

    host = UnboundReceiptHost(initial_gate=3, finish_gate=3)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="lacks exact visual wire timing",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )


@pytest.mark.parametrize("case", ("token", "observation"))
def test_current_target_rejects_inconsistent_observation_provenance(case):
    snapshot = _snapshot(0, "track-0", 10)
    target = _target(snapshot, "track-0")
    sample = snapshot.current_track.history[-1]
    if case == "token":
        snapshot.current_track.history = (
            replace(sample, token=_token(11)),
        )
    else:
        snapshot.current_track.history = (
            replace(sample, observation_monotonic_ns=200_000_001),
        )

    with pytest.raises(SafetyAbort, match="provenance is inconsistent"):
        course_stage._current_target_observation_monotonic_ns(
            snapshot,
            target,
            abort_type=SafetyAbort,
        )


def test_crossing_wait_accepts_newer_same_gate_status_before_credit():
    class SameGateHeartbeatHost(_Host):
        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(
                command,
                **kwargs,
            )
            if (
                self.crossing_hold_count == 1
                and command.thrust > 0.0
                and not (
                    self.visual_gate_graph.latest_snapshot
                    .current_track.visible
                )
            ):
                self.race = AuthoritativeRaceStatusRef.live(
                    session_id=self.race.session_id,
                    reset_epoch=self.race.reset_epoch,
                    race_generation=self.race.race_generation,
                    race_status_sequence=(
                        self.race.race_status_sequence + 1
                    ),
                    race_status_boot_ms=(
                        self.race.race_status_boot_ms + 100
                    ),
                    active_gate_index=self.current_gate,
                    received_monotonic_ns=max(
                        self.race.received_monotonic_ns + 1,
                        round(self.clock * 1_000_000_000),
                    ),
                    host_clock_id=self.race.host_clock_id,
                )
            return receipt

    host = SameGateHeartbeatHost(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["segments"][0][
        "crossing_wait_zero_command_count"
    ] == 0
    assert result["segments"][0][
        "crossing_wait_coast_command_count"
    ] == 2


def test_atomic_no_wire_credit_uses_latched_passage_and_finishes():
    class AtomicCreditHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.refused_attempts = 0
            self.post_credit_navigation_attempts = 0
            self.commands_at_credit = None

        async def _send_flight_command(self, command, **kwargs):
            if self.race.race_finished and command.thrust > 0.0:
                self.post_credit_navigation_attempts += 1
            if (
                command.thrust == 0.295
                and self.passage_counts.get(self.current_gate, 0) >= 3
                and not self.race.race_finished
            ):
                self.refused_attempts += 1
                self.disable_credit = False
                try:
                    self._advance_race()
                finally:
                    self.disable_credit = True
                self.commands_at_credit = len(self.commands)
                raise RaceActiveBoundaryChangedBeforeWire(
                    "synthetic atomic race credit before wire"
                )
            return await super()._send_flight_command(command, **kwargs)

    host = AtomicCreditHost(
        initial_gate=6,
        finish_gate=6,
        disable_credit=True,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["segments"][0]["crossing_anchor"] is not None
    assert result["segments"][0]["passage_command_count"] == 3
    assert host.refused_attempts == 1
    assert host.commands_at_credit == len(host.commands)
    assert host.post_credit_navigation_attempts == 0


@pytest.mark.parametrize(
    ("vertical", "vertical_rate", "expected"),
    [
        (-0.08333333333333333, -0.23, 0.3106466666666667),
        (-1.0, 0.5, 0.252),
        (1.0, -0.5, 0.298),
        (-1.0, -1.0, 0.32),
        (1.0, 1.0, 0.21),
    ],
)
def test_gate0_proved_vertical_collective_is_exact_and_bounded(
    vertical,
    vertical_rate,
    expected,
):
    assert course_stage._gate0_proved_vertical_collective(
        vertical,
        vertical_rate,
    ) == pytest.approx(expected)


def test_gate0_proved_vertical_collective_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="must be finite"):
        course_stage._gate0_proved_vertical_collective(
            float("nan"),
            0.0,
        )


@pytest.mark.parametrize(
    "elapsed_s",
    (0.0, 0.12, 0.40, 0.90),
)
def test_launch_pitch_reference_is_finite_convex_and_zero_slope_at_go(
    elapsed_s,
):
    spawn = -0.3100692828034804
    responsive = 0.120
    target, blend = course_stage._allocate_launch_pitch_target(
        spawn_pitch_rad=spawn,
        responsive_target_pitch_rad=responsive,
        launch_elapsed_s=elapsed_s,
    )
    acceleration = course_stage.LAUNCH_PITCH_REFERENCE_ACCEL_RAD_S2
    maximum_rate = course_stage.LAUNCH_PITCH_REFERENCE_MAX_RATE_RAD_S
    ramp_s = maximum_rate / acceleration
    traveled = (
        0.5 * acceleration * elapsed_s * elapsed_s
        if elapsed_s <= ramp_s
        else (
            0.5 * acceleration * ramp_s * ramp_s
            + maximum_rate * (elapsed_s - ramp_s)
        )
    )
    expected_blend = min(1.0, traveled / (responsive - spawn))

    assert blend == pytest.approx(expected_blend)
    assert target == pytest.approx(
        spawn + expected_blend * (responsive - spawn)
    )
    assert math.isfinite(target)
    assert spawn <= target <= responsive


def test_opposite_launch_demand_changes_reference_on_the_same_tick():
    common = {
        "spawn_pitch_rad": -0.3100692828034804,
        "launch_elapsed_s": 0.10,
    }
    forward, _forward_blend = course_stage._allocate_launch_pitch_target(
        responsive_target_pitch_rad=-0.35,
        **common,
    )
    brake, _brake_blend = course_stage._allocate_launch_pitch_target(
        responsive_target_pitch_rad=0.120,
        **common,
    )

    assert math.isfinite(forward)
    assert math.isfinite(brake)
    assert -0.35 <= forward < common["spawn_pitch_rad"] < brake <= 0.120
    assert brake > forward


def test_launch_destination_cannot_raise_gain_before_reference_allocates_it():
    target, _blend = course_stage._allocate_launch_pitch_target(
        spawn_pitch_rad=-0.3100692828034804,
        responsive_target_pitch_rad=0.120,
        launch_elapsed_s=0.437,
    )

    assert target < 0.0
    assert course_stage._pitch_response_authority(
        allocated_target_pitch_rad=target,
        intercept_response_authority=0.0,
    ) == 0.0
    assert course_stage._pitch_response_authority(
        allocated_target_pitch_rad=0.060,
        intercept_response_authority=0.0,
    ) == pytest.approx(
        0.060 / course_stage.MAX_VISUAL_TARGET_PITCH_RAD
    )
    assert course_stage._pitch_response_authority(
        allocated_target_pitch_rad=target,
        intercept_response_authority=1.0,
    ) == 1.0


def test_live_top_fov_outer_fallback_holds_nose_up_before_support_clips():
    raw_top = course_stage._raw_bbox_top_image_down(
        (
            282.0 / 640.0,
            134.0 / 360.0,
            362.0 / 640.0,
            214.0 / 360.0,
        )
    )
    capture_pitch = -0.3100692828034804
    proposal = course_stage._propose_top_fov_pitch_reference(
        capture_pitch_rad=capture_pitch,
        raw_top_edge_image_down=raw_top,
        raw_top_edge_rate_down_s=-0.10,
        requested_target_pitch_rad=0.120,
        prior_target_pitch_rad=-0.101,
        vertical_angle_scale_rad=0.55,
        active_before=False,
    )
    expected_maximum = (
        capture_pitch
        + math.atan(raw_top * 0.55)
        - math.atan(
            course_stage.TOP_FOV_SAFE_EDGE_IMAGE_DOWN * 0.55
        )
    )

    assert raw_top == pytest.approx(-0.25555555555555554)
    assert proposal.maximum_observable_target_pitch_rad == pytest.approx(
        expected_maximum
    )
    assert proposal.maximum_observable_target_pitch_rad == pytest.approx(
        -0.082201,
        abs=1e-6,
    )
    assert proposal.predicted_requested_top_edge_image_down < -1.0
    assert proposal.protected_target_pitch_rad == pytest.approx(
        capture_pitch
    )
    assert proposal.predicted_protected_top_edge_image_down == pytest.approx(
        raw_top
    )
    assert (
        proposal.predicted_protected_top_edge_image_down
        >= course_stage.TOP_FOV_SAFE_EDGE_IMAGE_DOWN
    )
    assert proposal.active_after is True
    assert proposal.limited is True


def test_decreasing_top_clearance_cannot_worsen_predicted_clipping():
    common = {
        "capture_pitch_rad": -0.20,
        "raw_top_edge_image_down": -0.20,
        "requested_target_pitch_rad": 0.0,
        "prior_target_pitch_rad": -0.15,
        "vertical_angle_scale_rad": 0.55,
        "active_before": True,
    }
    decreasing = course_stage._propose_top_fov_pitch_reference(
        raw_top_edge_rate_down_s=-0.10,
        **common,
    )
    recovered = course_stage._propose_top_fov_pitch_reference(
        raw_top_edge_rate_down_s=0.10,
        **common,
    )

    assert decreasing.protected_target_pitch_rad <= common[
        "prior_target_pitch_rad"
    ]
    assert (
        decreasing.predicted_protected_top_edge_image_down
        >= decreasing.predicted_requested_top_edge_image_down
    )
    assert decreasing.active_after is True
    assert recovered.protected_target_pitch_rad == pytest.approx(
        common["requested_target_pitch_rad"]
    )
    assert recovered.active_after is False
    assert recovered.limited is False


def test_adverse_nonrotational_edge_motion_tightens_pitch_before_clipping():
    common = {
        "capture_pitch_rad": -0.27,
        "raw_top_edge_image_down": -0.61,
        "raw_top_edge_rate_down_s": -0.20,
        "requested_target_pitch_rad": 0.12,
        "prior_target_pitch_rad": -0.20,
        "vertical_angle_scale_rad": 0.55,
        "active_before": True,
        "prediction_horizon_s": 0.10,
    }
    stationary = course_stage._propose_top_fov_pitch_reference(
        raw_top_edge_nonrotational_angle_rate_rad_s=0.0,
        **common,
    )
    closing = course_stage._propose_top_fov_pitch_reference(
        raw_top_edge_nonrotational_angle_rate_rad_s=-0.60,
        **common,
    )

    assert closing.forecast_top_edge_image_down < (
        stationary.forecast_top_edge_image_down
    )
    assert closing.maximum_observable_target_pitch_rad < (
        stationary.maximum_observable_target_pitch_rad
    )
    assert closing.protected_target_pitch_rad < (
        stationary.protected_target_pitch_rad
    )
    assert (
        closing.predicted_protected_top_edge_image_down
        >= course_stage.TOP_FOV_SAFE_EDGE_IMAGE_DOWN
    )


def test_top_fov_nonrotational_rate_is_invariant_under_pure_pitch():
    previous_top = -0.42
    scale = 0.55
    elapsed_s = 0.05
    pitch_rate = 0.40
    current_top = math.tan(
        math.atan(previous_top * scale) - pitch_rate * elapsed_s
    ) / scale

    assert course_stage._top_fov_nonrotational_angle_rate_rad_s(
        current_top_edge_image_down=current_top,
        previous_top_edge_image_down=previous_top,
        vertical_angle_scale_rad=scale,
        elapsed_s=elapsed_s,
        measured_pitch_rate_rad_s=pitch_rate,
    ) == pytest.approx(0.0, abs=1e-12)


def _inner_aperture(
    *,
    center_y: float,
    half_y: float,
    std_y: float = 0.01,
    std_log_scale: float = 0.03,
    confidence: float = 0.90,
    geometry_model_id: str = "vq2-visible-inner-quad-lines-v1",
    covariance_model_id: str = "vq2-visible-aperture-diagonal-v1",
    visible_edges: FrameEdge = (
        FrameEdge.LEFT
        | FrameEdge.TOP
        | FrameEdge.RIGHT
        | FrameEdge.BOTTOM
    ),
    health_reason: str | None = None,
) -> VisualInnerApertureGeometry:
    return VisualInnerApertureGeometry(
        center_norm=(0.0, center_y),
        half_size_norm=(0.20, half_y),
        log_scale=-1.0,
        measurement_std=(0.01, std_y, std_log_scale),
        confidence=confidence,
        clipping=FrameEdge.NONE,
        visible_edges=visible_edges,
        geometry_model_id=geometry_model_id,
        covariance_model_id=covariance_model_id,
        health_reason=health_reason,
    )


def test_top_fov_pitch_limit_uses_fitted_inner_extent_not_center_alone():
    short_top = course_stage._conservative_inner_aperture_top_image_down(
        _inner_aperture(center_y=-0.20, half_y=0.10)
    )
    tall_top = course_stage._conservative_inner_aperture_top_image_down(
        _inner_aperture(center_y=-0.20, half_y=0.30)
    )
    common = {
        "capture_pitch_rad": -0.20,
        "raw_top_edge_rate_down_s": 0.10,
        "requested_target_pitch_rad": 0.12,
        "prior_target_pitch_rad": -0.20,
        "vertical_angle_scale_rad": 0.55,
        "active_before": False,
    }
    short = course_stage._propose_top_fov_pitch_reference(
        raw_top_edge_image_down=short_top,
        **common,
    )
    tall = course_stage._propose_top_fov_pitch_reference(
        raw_top_edge_image_down=tall_top,
        **common,
    )

    assert tall.maximum_observable_target_pitch_rad < (
        short.maximum_observable_target_pitch_rad
    )
    assert (
        tall.predicted_protected_top_edge_image_down
        == pytest.approx(
            course_stage.TOP_FOV_SAFE_EDGE_IMAGE_DOWN
        )
    )


def test_top_fov_prefers_complete_low_confidence_inner_aperture_geometry():
    inner = _inner_aperture(
        center_y=-0.195,
        half_y=0.301,
        std_y=0.010,
        std_log_scale=0.030,
        confidence=0.17,
        health_reason="aperture_fit_low_confidence",
    )
    sample = SimpleNamespace(
        # The outer support has crossed the reserve, while the complete fitted
        # inner aperture remains observable.
        bbox_norm=(0.10, 0.06945, 0.90, 0.95),
        confidence=0.90,
        clipping=FrameEdge.NONE,
        center_censored=False,
        inner_aperture=inner,
    )

    edge = course_stage._top_fov_raw_edge(sample)
    expected_std = math.sqrt(0.010**2 + (0.301 * 0.030) ** 2)
    expected = -0.195 - 0.301 - 2.0 * expected_std

    assert edge.basis == course_stage.TOP_FOV_INNER_EDGE_BASIS
    assert edge.confidence == pytest.approx(0.17)
    assert edge.top_edge_image_down == pytest.approx(expected)
    assert edge.top_edge_image_down > (
        course_stage.TOP_FOV_SAFE_EDGE_IMAGE_DOWN
    )


def test_top_fov_uses_fresh_temporally_associated_tracking_geometry():
    inner = _inner_aperture(
        center_y=-0.18,
        half_y=0.29,
        std_y=0.04,
        std_log_scale=0.12,
        geometry_model_id=(
            "vq2-temporally-associated-inner-quad-lines-v1"
        ),
        covariance_model_id=(
            "vq2-temporally-associated-aperture-diagonal-v1"
        ),
        health_reason="aperture_fit_tracking_prior_only",
    )
    sample = SimpleNamespace(
        bbox_norm=(0.0, 0.0, 1.0, 1.0),
        confidence=0.90,
        clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
        center_censored=True,
        inner_aperture=inner,
    )

    edge = course_stage._top_fov_raw_edge(sample)

    assert not inner.passage_usable
    assert edge.basis == course_stage.TOP_FOV_INNER_EDGE_BASIS
    assert edge.top_edge_image_down < inner.center_norm[1]


def test_top_fov_recovery_requires_same_source_and_exceeds_uncertainty():
    previous = course_stage._top_fov_raw_edge(
        SimpleNamespace(
            bbox_norm=(0.10, 0.10, 0.90, 0.90),
            confidence=0.90,
            clipping=FrameEdge.NONE,
            center_censored=False,
            inner_aperture=_inner_aperture(
                center_y=-0.20,
                half_y=0.30,
                std_y=0.01,
                std_log_scale=0.03,
            ),
        )
    )
    noisy_nominal_recovery = course_stage._top_fov_raw_edge(
        SimpleNamespace(
            bbox_norm=(0.10, 0.10, 0.90, 0.90),
            confidence=0.90,
            clipping=FrameEdge.NONE,
            center_censored=False,
            inner_aperture=_inner_aperture(
                center_y=-0.18,
                half_y=0.30,
                std_y=0.01,
                std_log_scale=0.03,
            ),
        )
    )
    proved_recovery = replace(
        noisy_nominal_recovery,
        nominal_top_edge_image_down=(
            previous.nominal_top_edge_image_down + 0.08
        ),
    )
    outer_source = course_stage._TopFovRawEdge(
        top_edge_image_down=-0.40,
        nominal_top_edge_image_down=-0.40,
        top_edge_std_image_down=0.0,
        basis=course_stage.TOP_FOV_OUTER_EDGE_FALLBACK_BASIS,
        confidence=0.90,
    )

    assert (
        course_stage._top_fov_edge_recovery_rate_down_s(
            current=noisy_nominal_recovery,
            previous=previous,
            elapsed_s=0.05,
        )
        < 0.0
    )
    assert (
        course_stage._top_fov_edge_recovery_rate_down_s(
            current=proved_recovery,
            previous=previous,
            elapsed_s=0.05,
        )
        > 0.0
    )
    assert (
        course_stage._top_fov_edge_recovery_rate_down_s(
            current=outer_source,
            previous=previous,
            elapsed_s=0.05,
        )
        is None
    )


def test_top_fov_outer_fallback_refuses_clipped_or_censored_support():
    sample = SimpleNamespace(
        bbox_norm=(0.10, 0.05, 0.90, 0.95),
        confidence=0.90,
        clipping=FrameEdge.TOP,
        center_censored=True,
        inner_aperture=None,
    )

    with pytest.raises(ValueError, match="clean outer fallback"):
        course_stage._top_fov_raw_edge(sample)


def test_approach_inner_dropout_hold_is_bounded_to_prior_fov_authority():
    anchor_token = _token(40)
    dropout_token = _token(41)
    anchor_inner = _inner_aperture(
        center_y=-0.25,
        half_y=0.30,
    )
    rejected_inner = VisualInnerApertureGeometry(
        center_norm=None,
        half_size_norm=None,
        log_scale=None,
        measurement_std=None,
        confidence=0.0,
        clipping=FrameEdge.TOP,
        visible_edges=FrameEdge.NONE,
        geometry_model_id=None,
        covariance_model_id=None,
        health_reason="aperture_fit_rejected:ambiguous_multiple_aperture_gaps",
    )
    anchor = SimpleNamespace(
        token=anchor_token,
        observation_monotonic_ns=1_000_000_000,
        inner_aperture=anchor_inner,
    )
    dropout = SimpleNamespace(
        token=dropout_token,
        observation_monotonic_ns=1_032_000_000,
        inner_aperture=rejected_inner,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    track = SimpleNamespace(
        track_id="track-0",
        latest_token=dropout_token,
        role=VisualTrackRole.CURRENT,
        visible=True,
        ambiguous=False,
        missed_frame_count=0,
        clipping=FrameEdge.TOP,
        center_censored=True,
        history=(anchor, dropout),
    )
    snapshot = SimpleNamespace(
        current_gate_index=0,
        current_track_id="track-0",
        current_track=track,
        latest_camera_token=dropout_token,
        authority_usable=True,
    )
    fov_summary = {
        "last_inner_active": True,
        "last_inner_track_id": "track-0",
        "last_inner_camera_token": asdict(anchor_token),
        "last_inner_wire_start_monotonic_ns": 1_010_000_000,
        "last_inner_raw_top_edge_basis": (
            course_stage.TOP_FOV_INNER_EDGE_BASIS
        ),
        "last_inner_protected_target_pitch_rad": -0.31,
    }

    authority = course_stage._derive_approach_inner_dropout_authority(
        snapshot=snapshot,
        expected_gate_index=0,
        expected_track_id="track-0",
        maximum_age_s=0.12,
        now_monotonic_ns=1_042_000_000,
        fov_summary=fov_summary,
    )

    assert authority is not None
    assert authority.age_s == pytest.approx(0.032)
    assert authority.maximum_target_pitch_rad == pytest.approx(-0.31)
    assert math.isfinite(authority.maximum_target_pitch_rad)
    assert (
        course_stage.MIN_VISUAL_TARGET_PITCH_RAD
        <= authority.maximum_target_pitch_rad
        <= course_stage.MAX_VISUAL_TARGET_PITCH_RAD
    )

    mismatched_fov_summary = dict(fov_summary)
    mismatched_fov_summary["last_inner_camera_token"] = asdict(
        _token(39)
    )
    assert (
        course_stage._derive_approach_inner_dropout_authority(
            snapshot=snapshot,
            expected_gate_index=0,
            expected_track_id="track-0",
            maximum_age_s=0.12,
            now_monotonic_ns=1_042_000_000,
            fov_summary=mismatched_fov_summary,
        )
        is None
    )
    assert (
        course_stage._derive_approach_inner_dropout_authority(
            snapshot=snapshot,
            expected_gate_index=0,
            expected_track_id="track-0",
            maximum_age_s=0.120_001,
            now_monotonic_ns=1_042_000_000,
            fov_summary=fov_summary,
        )
        is None
    )

    fallback_token = _token(42)
    fallback = SimpleNamespace(
        token=fallback_token,
        observation_monotonic_ns=1_064_000_000,
        inner_aperture=rejected_inner,
        clipping=FrameEdge.NONE,
        center_censored=False,
    )
    bottom_token = _token(43)
    bottom = SimpleNamespace(
        token=bottom_token,
        observation_monotonic_ns=1_097_000_000,
        inner_aperture=rejected_inner,
        clipping=FrameEdge.BOTTOM,
        center_censored=True,
    )
    track.latest_token = bottom_token
    track.clipping = FrameEdge.BOTTOM
    track.history = (anchor, dropout, fallback, bottom)
    snapshot.latest_camera_token = bottom_token
    bridged = course_stage._derive_approach_inner_dropout_authority(
        snapshot=snapshot,
        expected_gate_index=0,
        expected_track_id="track-0",
        maximum_age_s=0.12,
        now_monotonic_ns=1_107_000_000,
        fov_summary=fov_summary,
    )
    assert bridged is not None
    assert bridged.anchor_camera_token == anchor_token
    assert bridged.last_camera_token == bottom_token
    assert bridged.age_s == pytest.approx(0.097)

    skipped = SimpleNamespace(
        token=_token(42),
        observation_monotonic_ns=1_054_000_000,
        inner_aperture=rejected_inner,
        clipping=FrameEdge.NONE,
        center_censored=False,
    )
    after_gap_token = _token(43)
    after_gap = SimpleNamespace(
        token=after_gap_token,
        observation_monotonic_ns=1_076_000_000,
        inner_aperture=rejected_inner,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    track.latest_token = after_gap_token
    track.history = (anchor, dropout, skipped, after_gap)
    snapshot.latest_camera_token = after_gap_token
    assert (
        course_stage._derive_approach_inner_dropout_authority(
            snapshot=snapshot,
            expected_gate_index=0,
            expected_track_id="track-0",
            maximum_age_s=0.12,
            now_monotonic_ns=1_086_000_000,
            fov_summary=fov_summary,
            existing=authority,
        )
        is None
    )

    expired_token = _token(42)
    expired = SimpleNamespace(
        token=expired_token,
        observation_monotonic_ns=1_121_000_000,
        inner_aperture=rejected_inner,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    track.latest_token = expired_token
    track.clipping = FrameEdge.TOP
    track.history = (anchor, dropout, expired)
    snapshot.latest_camera_token = expired_token

    assert (
        course_stage._derive_approach_inner_dropout_authority(
            snapshot=snapshot,
            expected_gate_index=0,
            expected_track_id="track-0",
            maximum_age_s=0.12,
            now_monotonic_ns=1_131_000_000,
            fov_summary=fov_summary,
            existing=authority,
        )
        is None
    )


def test_top_fov_uses_full_bounded_pitch_when_reserve_is_infeasible():
    proposal = course_stage._propose_top_fov_pitch_reference(
        capture_pitch_rad=-0.284,
        raw_top_edge_image_down=-0.90,
        raw_top_edge_rate_down_s=-0.70,
        requested_target_pitch_rad=0.12,
        prior_target_pitch_rad=-0.338,
        vertical_angle_scale_rad=0.55,
        active_before=True,
    )

    assert proposal.envelope_saturated is True
    assert proposal.maximum_observable_target_pitch_rad < (
        course_stage.MIN_VISUAL_TARGET_PITCH_RAD
    )
    assert proposal.protected_target_pitch_rad == (
        course_stage.MIN_VISUAL_TARGET_PITCH_RAD
    )
    assert proposal.active_after is True


def test_top_fov_capture_pitch_uses_active_body_to_reference_quaternion():
    pitch = 0.23
    quaternion = (
        math.cos(pitch / 2.0),
        0.0,
        math.sin(pitch / 2.0),
        0.0,
    )

    assert course_stage._body_to_reference_pitch_rad(
        quaternion
    ) == pytest.approx(pitch)


def test_8319198e_dynamic_launch_does_not_discard_proved_collective():
    thrust, phase = course_stage._allocate_launch_collective(
        launch_elapsed_s=0.422,
        post_preload_thrust=0.280,
        configured_boost_duration_s=0.45,
        configured_boost_thrust=0.32,
        dynamic_collective_owns_post_preload=True,
    )
    legacy_thrust, legacy_phase = (
        course_stage._allocate_launch_collective(
            launch_elapsed_s=0.422,
            post_preload_thrust=0.280,
            configured_boost_duration_s=0.45,
            configured_boost_thrust=0.32,
            dynamic_collective_owns_post_preload=False,
        )
    )

    assert thrust == pytest.approx(0.280)
    assert phase == "proved-current-aperture"
    assert legacy_thrust == pytest.approx(0.32)
    assert legacy_phase == "boost"


def test_current_aperture_collective_recreates_new_frame_rate_filter():
    state = course_stage._CurrentApertureProvedCollectiveState()

    def observed(sequence, received, vertical):
        target = _target(
            _snapshot(0, "track-0", sequence),
            "track-0",
        )
        return replace(
            target,
            received_monotonic_s=received,
            normalized_y_down=vertical,
        )

    first = observed(1, 1.0, 0.0)
    second = observed(2, 1.1, -0.02)
    first_thrust, first_rate = state.observe(first)
    second_thrust, second_rate = state.observe(second)
    duplicate_thrust, duplicate_rate = state.observe(
        replace(
            second,
            frame_token=replace(
                second.frame_token,
                publication_sequence=3,
            ),
            received_monotonic_s=1.15,
        )
    )
    third_thrust, third_rate = state.observe(
        observed(4, 1.2, -0.03)
    )

    assert first_rate == 0.0
    assert first_thrust == pytest.approx(0.275)
    assert second_rate == pytest.approx(-0.07)
    assert second_thrust == pytest.approx(0.28542)
    assert duplicate_rate == pytest.approx(-0.07)
    assert duplicate_thrust == pytest.approx(0.28542)
    assert third_rate == pytest.approx(-0.0805)
    assert third_thrust == pytest.approx(0.287543)


def test_105f607_terminal_vertical_replay_commands_below_support():
    """Freeze the last clean current-aperture facts from live run 250407d7."""

    state = course_stage._CurrentApertureProvedCollectiveState(
        track_id="track-0",
        # Exact value retained in the authoritative compact result.  Receiver
        # timestamps are intentionally compacted, so seed the proved filter
        # at its recorded terminal state instead of fabricating precision.
        filtered_vertical_rate=0.6221695111650704,
    )
    target = replace(
        _target(_snapshot(0, "track-0", 166), "track-0"),
        received_monotonic_s=241806.953,
        normalized_y_down=0.05555555555555558,
        normalized_y_rate_down_s=0.6993507654430058,
    )

    proposal = course_stage._propose_current_aperture_collective(
        state,
        target,
        authoritative_current_track_id="track-0",
    )

    assert proposal.filtered_vertical_rate_down_s == pytest.approx(
        0.6221695111650704
    )
    assert proposal.requested_thrust == pytest.approx(0.21)
    assert proposal.requested_thrust < 0.275
    assert proposal.vertical_censored is False
    assert proposal.held_last_observable_collective is False


def test_054358af_expansion_q_rate_cannot_unload_collective():
    """Freeze the first causal collective blocker from live run 054358af."""

    decision = SimpleNamespace(
        passage_error_norm=(0.0, -1.111),
        current_aperture_half_size_norm=(
            0.10,
            0.198,
        ),
        # Expansion made q look convergent even though derotated translation
        # was still strongly topward.
        crossing_rate_q_s=(0.0, 0.963),
    )
    residual_rate_down_s = -1.117
    vertical_angle_scale_rad = 0.55
    current = SimpleNamespace(
        residual_translational_rate_rad_s=(
            0.0,
            residual_rate_down_s * vertical_angle_scale_rad,
        ),
    )
    error, translation_rate_down_s, basis = (
        course_stage._dynamic_current_aperture_collective_inputs(
            decision,
            current,
            vertical_angle_scale_rad=vertical_angle_scale_rad,
        )
    )
    state = course_stage._CurrentApertureProvedCollectiveState(
        track_id="track-0"
    )
    target = _target(_snapshot(0, "track-0", 152), "track-0")
    proposal = course_stage._propose_current_aperture_collective(
        state,
        target,
        authoritative_current_track_id="track-0",
        control_vertical_error_image_down=error,
        control_vertical_rate_down_s=translation_rate_down_s,
        control_basis=basis,
    )
    falsified_expansion_request = (
        course_stage._gate0_proved_vertical_collective(
            error,
            decision.crossing_rate_q_s[1]
            * decision.current_aperture_half_size_norm[1],
        )
    )

    assert translation_rate_down_s == pytest.approx(-1.117)
    assert basis == course_stage.CURRENT_APERTURE_PROVED_COLLECTIVE_BASIS
    assert falsified_expansion_request == pytest.approx(
        0.290975076,
    )
    assert proposal.requested_thrust == pytest.approx(0.32)
    assert proposal.requested_thrust > falsified_expansion_request


def test_current_aperture_collective_holds_through_censorship_and_dropout():
    state = course_stage._CurrentApertureProvedCollectiveState(
        track_id="track-3",
        filtered_vertical_rate=0.6221695111650704,
    )
    clean = replace(
        _target(_snapshot(3, "track-3", 20), "track-3"),
        received_monotonic_s=12.0,
        normalized_y_down=0.05555555555555558,
    )
    clean_proposal = course_stage._propose_current_aperture_collective(
        state,
        clean,
        authoritative_current_track_id="track-3",
    )
    censored = replace(
        clean,
        frame_token=replace(
            clean.frame_token,
            frame_id=clean.frame_token.frame_id + 1,
            publication_sequence=(
                clean.frame_token.publication_sequence + 1
            ),
        ),
        received_monotonic_s=12.03,
        normalized_y_down=-1.0,
        normalized_y_rate_down_s=-8.0,
        clipped=True,
        center_censored=True,
        vertical_censored=True,
    )
    censored_proposal = (
        course_stage._propose_current_aperture_collective(
            state,
            censored,
            authoritative_current_track_id="track-3",
        )
    )
    fallback = replace(
        censored,
        frame_token=replace(
            censored.frame_token,
            frame_id=censored.frame_token.frame_id + 1,
            publication_sequence=(
                censored.frame_token.publication_sequence + 1
            ),
        ),
        received_monotonic_s=12.06,
        normalized_y_down=0.75,
        normalized_y_rate_down_s=4.0,
        clipped=False,
        center_censored=False,
        vertical_censored=False,
    )
    fallback_proposal = (
        course_stage._propose_current_aperture_collective(
            state,
            fallback,
            authoritative_current_track_id="track-3",
            current_aperture_observable=False,
        )
    )
    adjacent = _target(
        _snapshot(4, "track-4", 22),
        "track-4",
    )
    dropout_proposal = (
        course_stage._propose_current_aperture_collective(
            state,
            adjacent,
            authoritative_current_track_id="track-3",
        )
    )

    assert clean_proposal.requested_thrust == pytest.approx(0.21)
    assert censored_proposal.requested_thrust == pytest.approx(
        clean_proposal.requested_thrust
    )
    assert censored_proposal.vertical_censored is True
    assert censored_proposal.held_last_observable_collective is True
    assert fallback_proposal.requested_thrust == pytest.approx(
        clean_proposal.requested_thrust
    )
    assert fallback_proposal.current_aperture_dropout is True
    assert fallback_proposal.held_last_observable_collective is True
    assert fallback_proposal.control_vertical_error_image_down == (
        clean.normalized_y_down
    )
    assert dropout_proposal.requested_thrust == pytest.approx(
        clean_proposal.requested_thrust
    )
    assert dropout_proposal.current_aperture_dropout is True
    assert dropout_proposal.held_last_observable_collective is True

    promoted_state = (
        course_stage._CurrentApertureProvedCollectiveState(
            track_id="track-4"
        )
    )
    promoted = replace(
        adjacent,
        normalized_y_down=-0.10,
    )
    promoted_proposal = (
        course_stage._propose_current_aperture_collective(
            promoted_state,
            promoted,
            authoritative_current_track_id="track-4",
        )
    )
    assert promoted_proposal.requested_thrust > 0.275


def test_faa7cee6_collective_uses_derotated_state_not_pitch_motion():
    target = replace(
        _target(_snapshot(0, "track-0", 10), "track-0"),
        normalized_y_down=-0.3166666667,
        normalized_y_rate_down_s=0.4157384466,
    )
    first_state = course_stage._CurrentApertureProvedCollectiveState(
        track_id="track-0"
    )
    first = course_stage._propose_current_aperture_collective(
        first_state,
        target,
        authoritative_current_track_id="track-0",
        control_vertical_error_image_down=-1.0283863293,
        control_vertical_rate_down_s=0.1110041644,
        control_basis=(
            course_stage.CURRENT_APERTURE_PROVED_COLLECTIVE_BASIS
        ),
    )
    pitch_shifted = replace(
        target,
        normalized_y_down=0.20,
        normalized_y_rate_down_s=-0.70,
    )
    second_state = course_stage._CurrentApertureProvedCollectiveState(
        track_id="track-0"
    )
    second = course_stage._propose_current_aperture_collective(
        second_state,
        pitch_shifted,
        authoritative_current_track_id="track-0",
        control_vertical_error_image_down=-1.0283863293,
        control_vertical_rate_down_s=0.1110041644,
        control_basis=(
            course_stage.CURRENT_APERTURE_PROVED_COLLECTIVE_BASIS
        ),
    )
    falsified_raw_request = (
        course_stage._gate0_proved_vertical_collective(
            -0.3166666667,
            0.4157384466,
        )
    )

    assert falsified_raw_request == pytest.approx(0.2479502891)
    assert first.requested_thrust == pytest.approx(0.3010134753)
    assert second.requested_thrust == pytest.approx(
        first.requested_thrust
    )
    assert first.control_basis == (
        course_stage.CURRENT_APERTURE_PROVED_COLLECTIVE_BASIS
    )


def test_initial_gate_uses_hashed_launch_bootstrap_only_once():
    host = _Host(
        initial_gate=0,
        finish_gate=1,
        fresh_after_samples=1,
    )
    runtime, calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    initial, later = result["segments"]
    launch = initial["launch_bootstrap"]
    assert launch["enabled"] is True
    assert launch["preload_duration_s"] == 0.15
    assert launch["preload_thrust"] == 0.26
    assert launch["boost_duration_s"] == (
        host.visual_config.lifecycle.launch_boost_duration_s
    )
    assert launch["boost_thrust"] == (
        host.visual_config.lifecycle.launch_boost_thrust
    )
    assert launch["post_boost_collective_basis"] == "generic-visual-servo"
    assert launch["post_boost_collective_base"] is None
    assert launch["post_boost_collective_error_gain"] is None
    assert launch["post_boost_collective_rate_gain"] is None
    assert launch["post_boost_collective_max_abs_error"] is None
    assert launch["post_boost_collective_max_abs_rate"] is None
    assert launch["post_boost_collective_rate_filter_alpha"] is None
    assert launch["post_boost_next_preview_collective_basis"] is None
    assert launch[
        "post_boost_next_preview_collective_error_gain"
    ] is None
    assert launch[
        "post_boost_next_preview_collective_max_thrust_delta"
    ] is None
    assert launch["pitch_blend_s"] == (
        host.visual_config.lifecycle.launch_pitch_blend_s
    )
    assert launch["pitch_reference_basis"] == (
        course_stage.LAUNCH_PITCH_REFERENCE_BASIS
    )
    assert launch["pitch_reference_max_rate_rad_s"] == 0.60
    assert launch["pitch_reference_accel_rad_s2"] == 2.50
    assert "passage_admission_withheld_count" not in launch
    assert later["launch_bootstrap"]["enabled"] is False
    gate0_passage_calls = [
        call
        for call in calls
        if call[0] == 0 and call[1] is VisualApproachMode.PASSAGE
    ]
    assert gate0_passage_calls
    assert any(not call[3] for call in gate0_passage_calls)
    assert any(call[3] for call in gate0_passage_calls)
    assert all(
        call[3]
        is (
            call[4]
            >= host.visual_config.lifecycle.launch_pitch_blend_s
        )
        for call in gate0_passage_calls
    )

    gate0_commands = [
        (command, kwargs)
        for command, kwargs, gate_index in host.commands
        if gate_index == 0
    ]
    assert launch["first_target_pitch_rad"] == pytest.approx(-0.31)
    assert gate0_commands[0][0].pitch_rate == pytest.approx(-0.25)
    assert gate0_commands[0][0].thrust == 0.26
    assert any(
        command.thrust
        == host.visual_config.lifecycle.launch_boost_thrust
        for command, _kwargs in gate0_commands
    )
    assert launch["last_thrust_phase"] == "generic-visual-servo"
    assert launch["last_thrust"] == pytest.approx(0.295)
    assert launch["last_current_vertical_error_image_down"] == 0.03
    assert launch["last_current_vertical_rate_down_s"] == 0.0
    assert launch["last_proved_filtered_vertical_rate_down_s"] == 0.0
    assert launch["next_preview_collective_command_count"] == 0
    assert launch["max_next_preview_collective_delta"] == 0.0
    assert launch["last_next_preview_collective_delta"] == 0.0
    assert launch["last_next_preview_collective_track_id"] is None
    assert any(
        command.thrust == pytest.approx(0.295)
        for command, _kwargs in gate0_commands
    )
    assert all(
        kwargs["wire_race_gate_index"] == 0
        for _command, kwargs in gate0_commands
        if kwargs.get("require_wire_receipt")
    )
    later_navigation = [
        command
        for command, _kwargs, gate_index in host.commands
        if gate_index == 1 and command.thrust > 0.0
    ]
    assert later_navigation
    assert later_navigation[0].thrust == 0.21
    assert later["launch_bootstrap"][
        "post_boost_collective_basis"
    ] is None
    assert later["launch_bootstrap"][
        "post_boost_next_preview_collective_basis"
    ] is None
    gate0_passage = [
        command
        for stage, _elapsed, command in host.ticks
        if stage == "visual-course/gate0/passage"
    ]
    gate1_passage = [
        command
        for stage, _elapsed, command in host.ticks
        if stage == "visual-course/gate1/passage"
    ]
    assert gate0_passage
    assert any(command.thrust == 0.26 for command in gate0_passage)
    assert any(
        command.thrust
        == host.visual_config.lifecycle.launch_boost_thrust
        for command in gate0_passage
    )
    assert gate0_passage[-1].thrust == pytest.approx(0.295)
    assert gate1_passage
    assert all(command.thrust == 0.295 for command in gate1_passage)


def test_initial_gate_arms_from_finite_preblend_admission_window():
    """Keep the exact attempt-3 admission window independent of launch shaping."""

    admission_window = (0.172, 0.797)
    planners = []

    class FiniteAdmissionWindowServo(_Servo):
        def observe(
            self,
            snapshot,
            tracker,
            now_monotonic_s,
            segment_elapsed_s,
            segment_yaw_excursion_rad,
            **kwargs,
        ):
            proposal = super().observe(
                snapshot,
                tracker,
                now_monotonic_s,
                segment_elapsed_s,
                segment_yaw_excursion_rad,
                **kwargs,
            )
            if (
                kwargs["mode"] is VisualApproachMode.APPROACH
                and not (
                    admission_window[0]
                    <= segment_elapsed_s
                    <= admission_window[1]
                )
            ):
                proposal.passage_admission = None
            return proposal

    def servo_factory(*args, **kwargs):
        planner = FiniteAdmissionWindowServo(
            *args,
            **kwargs,
            calls=calls,
        )
        planners.append(planner)
        return planner

    host = _Host(initial_gate=0, finish_gate=0, fresh_after_samples=1)
    runtime, calls = _runtime(host)
    runtime = replace(runtime, servo_factory=servo_factory)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert len(planners) == 1
    segment = result["segments"][0]
    assert segment["passage_authority_enabled"] is True
    assert segment["lifecycle"] in {
        "near_plane_latched",
        "credit_wait",
    }
    assert segment["passage_admission"] is not None
    assert "passage_admission_withheld_count" not in (
        segment["launch_bootstrap"]
    )

    approach_calls = [
        call
        for call in calls
        if call[1] is VisualApproachMode.APPROACH
    ]
    passage_calls = [
        call
        for call in calls
        if call[1] is VisualApproachMode.PASSAGE
    ]
    assert approach_calls
    assert admission_window[0] <= approach_calls[-1][4] <= admission_window[1]
    assert passage_calls
    assert passage_calls[0][2] is not None
    assert passage_calls[0][3] is False
    assert any(call[3] for call in passage_calls)
    assert segment["advance_command_count"] >= 3


@pytest.mark.parametrize(
    ("unsafe_advance", "unsafe_target_pitch"),
    (
        (True, 0.0),
        (False, -0.105),
        (0, 0.0),
    ),
)
def test_initial_gate_rejects_planner_that_escapes_closure_inhibit(
    unsafe_advance,
    unsafe_target_pitch,
):
    calls = []

    class UnsafeClosureServo(_Servo):
        def observe(self, *args, **kwargs):
            proposal = super().observe(*args, **kwargs)
            if (
                kwargs["mode"] is VisualApproachMode.PASSAGE
                and not kwargs["passage_forward_closure_authorized"]
            ):
                proposal.servo_output = replace(
                    proposal.servo_output,
                    target_pitch_rad=unsafe_target_pitch,
                    thrust=0.295,
                    advance_enabled=unsafe_advance,
                    brake_reason=None,
                )
            return proposal

    def servo_factory(*args, **kwargs):
        return UnsafeClosureServo(*args, **kwargs, calls=calls)

    host = _Host(initial_gate=0, finish_gate=0, fresh_after_samples=1)
    runtime, _unused_calls = _runtime(host)
    runtime = replace(runtime, servo_factory=servo_factory)

    with pytest.raises(
        SafetyAbort,
        match="escaped its launch forward-closure inhibit",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert [call[1] for call in calls] == [
        VisualApproachMode.APPROACH,
        VisualApproachMode.PASSAGE,
    ]
    navigation = [
        command
        for command, kwargs, _gate_index in host.commands
        if kwargs.get("require_wire_receipt")
    ]
    assert len(navigation) == 1
    assert navigation[0].pitch_rate != -0.105


def test_course_wires_the_hashed_next_preview_scale_ramp_to_every_segment():
    host = _Host(initial_gate=3, finish_gate=4, fresh_after_samples=1)
    runtime, _calls = _runtime(host)
    factory_calls = []

    def capture_factory(*args, **kwargs):
        factory_calls.append(dict(kwargs))
        return _Servo(*args, **kwargs)

    runtime = replace(runtime, servo_factory=capture_factory)
    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert len(factory_calls) == 2
    assert all(
        call["next_gate_blend"]
        == host.visual_config.lifecycle.next_gate_blend_max
        and call["next_gate_blend_start_log_scale"]
        == host.visual_config.lifecycle.next_gate_blend_start_log_scale
        and call["next_gate_blend_full_log_scale"]
        == host.visual_config.lifecycle.next_gate_blend_full_log_scale
        and "required_next_track_id" not in call
        for call in factory_calls
    )


def test_passage_safety_refusal_after_entry_remains_fatal():
    host = _Host(initial_gate=3, finish_gate=3, disable_credit=True)

    class PassageRefusingServo(_Servo):
        def observe(self, snapshot, *args, **kwargs):
            if kwargs["mode"] is VisualApproachMode.PASSAGE:
                raise VisualApproachPassageSafetyUnavailable(
                    "passage authority left its corridor",
                    violation_codes=("current_vertical_rate",),
                    violation_evidence=(
                        (
                            "current_vertical_rate",
                            0.61,
                            0.60,
                            0.01,
                        ),
                    ),
                    camera_observation_monotonic_s=(
                        snapshot.latest_camera_token.publication_sequence
                        * 0.02
                    ),
                )
            return super().observe(snapshot, *args, **kwargs)

    runtime, _calls = _runtime(host)
    runtime = replace(
        runtime,
        servo_factory=lambda *args, **kwargs: PassageRefusingServo(
            *args,
            **kwargs,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match=(
            "visual authority refused: passage authority left its corridor"
        ),
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )


def test_newer_receiver_publication_drops_unsent_proposal_and_replans():
    class SupersedingHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.superseded_expected = None
            self.superseded_receiver = None

        async def _send_flight_command(self, command, **kwargs):
            if (
                self.superseded_expected is None
                and kwargs.get("wire_visual_token") is not None
            ):
                expected = kwargs["wire_visual_token"]
                receiver = _token(expected.publication_sequence + 1)
                self.superseded_expected = expected
                self.superseded_receiver = receiver
                self.sequence = receiver.publication_sequence
                self.clock = max(
                    self.clock,
                    receiver.publication_sequence * 0.02,
                )
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
            return await super()._send_flight_command(command, **kwargs)

    host = SupersedingHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    segment = result["segments"][0]
    assert segment["superseded_proposal_count"] == 1
    assert segment["launch_bootstrap"]["command_count"] == (
        result["visual_navigation_command_count"]
    )
    sent_tokens = [
        kwargs["wire_visual_token"]
        for _command, kwargs, _gate_index in host.commands
        if kwargs.get("wire_visual_token") is not None
    ]
    assert host.superseded_expected not in sent_tokens
    assert all(
        token.publication_sequence
        > host.superseded_expected.publication_sequence
        for token in sent_tokens
    )
    events = [
        payload
        for event, payload in host.recorder.events
        if event == "visual_course_proposal_superseded"
    ]
    assert len(events) == 1
    assert events[0]["expected_frame_token"] == {
        "generation": host.superseded_expected.generation,
        "frame_id": host.superseded_expected.frame_id,
        "publication_sequence": (
            host.superseded_expected.publication_sequence
        ),
        "stream_id": host.superseded_expected.stream_id,
    }
    assert events[0]["receiver_frame_token"] == {
        "generation": host.superseded_receiver.generation,
        "frame_id": host.superseded_receiver.frame_id,
        "publication_sequence": (
            host.superseded_receiver.publication_sequence
        ),
        "stream_id": host.superseded_receiver.stream_id,
    }


def test_precheck_publication_supersession_replans_before_wire():
    class PrecheckSupersedingHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.superseded_expected = None

        def _assert_visual_receiver_token_current(self, expected_token):
            if self.superseded_expected is None:
                receiver = _token(
                    expected_token.publication_sequence + 1
                )
                self.superseded_expected = expected_token
                self.sequence = receiver.publication_sequence
                self.clock = max(
                    self.clock,
                    receiver.publication_sequence * 0.02,
                )
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected_token
                exc.receiver_visual_token = receiver
                raise exc
            return super()._assert_visual_receiver_token_current(
                expected_token
            )

    host = PrecheckSupersedingHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["segments"][0]["superseded_proposal_count"] == 1
    sent_tokens = [
        kwargs["wire_visual_token"]
        for _command, kwargs, _gate_index in host.commands
        if kwargs.get("wire_visual_token") is not None
    ]
    assert host.superseded_expected not in sent_tokens
    assert result["segments"][0]["launch_bootstrap"][
        "command_count"
    ] == result["visual_navigation_command_count"]


def test_receiver_supersession_burst_drops_ticks_until_latest_wire():
    class BurstSupersedingHost(_Host):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.rejected_tokens = []

        async def _send_flight_command(self, command, **kwargs):
            expected = kwargs.get("wire_visual_token")
            if expected is not None and len(self.rejected_tokens) < 6:
                receiver = _token(expected.publication_sequence + 1)
                self.rejected_tokens.append(expected)
                self.sequence = receiver.publication_sequence
                self.clock = max(
                    self.clock,
                    receiver.publication_sequence * 0.02,
                )
                self.visual_gate_graph.latest_snapshot = _snapshot(
                    self.current_gate,
                    self.current_track_id,
                    self.sequence,
                )
                exc = SafetyAbort(
                    course_stage
                    .VISUAL_RECEIVER_PROPOSAL_SUPERSEDED_REASON
                )
                exc.expected_visual_token = expected
                exc.receiver_visual_token = receiver
                raise exc
            return await super()._send_flight_command(command, **kwargs)

    host = BurstSupersedingHost(
        initial_gate=0,
        finish_gate=0,
    )
    runtime, _calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["segments"][0]["superseded_proposal_count"] == 6
    sent_tokens = [
        kwargs["wire_visual_token"]
        for _command, kwargs, _gate_index in host.commands
        if kwargs.get("wire_visual_token") is not None
    ]
    assert all(token not in sent_tokens for token in host.rejected_tokens)
    assert min(
        token.publication_sequence for token in sent_tokens
    ) > max(
        token.publication_sequence for token in host.rejected_tokens
    )


def test_promotion_failure_retains_observed_authoritative_credit_and_counts():
    class PromotionFailureHost(_Host):
        def _confirm_visual_transition(self, **_kwargs):
            raise SafetyAbort("synthetic promotion failure")

    host = PromotionFailureHost(
        initial_gate=3,
        finish_gate=4,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="synthetic promotion failure"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    summary = host._visual_course_summary
    assert summary["maximum_authoritative_gate_index"] == 4
    assert summary["final_gate_index"] == 4
    assert summary["first_causal_blocker"] == (
        "synthetic promotion failure"
    )
    assert len(summary["authoritative_transitions"]) == 1
    transition = summary["authoritative_transitions"][0]
    assert transition["from_gate_index"] == 3
    assert transition["to_gate_index"] == 4
    assert transition["promotion_confirmed"] is False
    assert transition["pre_transition_navigation_command_count"] == 4
    assert transition["post_transition_navigation_command_count"] == 0


def test_nonzero_yaw_requires_the_accepted_calibration_profile():
    host = _Host(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host, yaw_profile=False)

    with pytest.raises(
        SafetyAbort,
        match="nonzero yaw lacks calibrated authority",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []


def test_authoritative_transition_without_passage_evidence_fails_closed():
    host = _Host(
        initial_gate=8,
        finish_gate=9,
        credit_on_approach=True,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="without credible passage evidence",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )


def test_passage_timer_is_tight_and_cannot_run_without_advance():
    host = _Host(initial_gate=2, finish_gate=2, disable_credit=True)
    limits = replace(
        VisualCourseStageLimits(),
        passage_hard_duration_s=0.40,
    )
    runtime, _calls = _runtime(
        host,
        servo_options={"passage_advances": False, "yaw_rate": 0.0},
        limits=limits,
    )

    with pytest.raises(SafetyAbort, match="passage expired"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert all(
        command.thrust != 0.295
        for command, _kwargs, _gate in host.commands
    )


def test_promoted_gate_fresh_frame_opportunity_has_a_hard_timeout():
    host = _Host(
        initial_gate=0,
        finish_gate=1,
        fresh_after_samples=999,
    )
    limits = replace(
        VisualCourseStageLimits(),
        post_credit_fresh_frame_timeout_s=0.05,
    )
    runtime, _calls = _runtime(host, limits=limits)

    with pytest.raises(SafetyAbort, match="fresh post-credit"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    zeros = [
        command
        for command, _kwargs, _gate in host.commands
        if command.thrust == 0.0
    ]
    assert 1 <= len(zeros) <= 3


def test_yaw_profile_loads_only_the_exact_tracked_multi_run_authority():
    profile = _yaw_profile()

    assert profile.schema == VISUAL_COURSE_YAW_PROFILE_SCHEMA
    assert profile.profile_id == YAW_CALIBRATION_PROFILE_ID
    assert profile.profile_sha256 == YAW_CALIBRATION_PROFILE_SHA256
    assert profile.source_commit == YAW_CALIBRATION_SOURCE_COMMIT
    assert profile.plan_id == YAW_CALIBRATION_PLAN_ID
    assert profile.plan_sha256 == YAW_CALIBRATION_PLAN_SHA256
    assert profile.max_abs_yaw_rate_command_rad_s == 0.15
    assert profile.max_attitude_excursion_rad == 0.10
    assert profile.max_abs_measured_yaw_rate_rad_s == 0.5
    assert profile.observed_max_abs_measured_yaw_rate_rad_s == (
        0.4234286031848751
    )
    assert VisualCourseStageLimits().max_yaw_rate_rad_s == 0.15
    assert (
        VisualCourseStageLimits().max_yaw_rate_rad_s
        == profile.max_abs_yaw_rate_command_rad_s
    )
    assert (
        VisualCourseStageLimits().max_segment_yaw_excursion_rad
        == MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD
    )
    with pytest.raises(TypeError, match="tracked loader"):
        VisualCourseYawProfile(
            issuer=object(),
            **{
                field: getattr(profile, field)
                for field in profile.__dataclass_fields__
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("control_period_s", 0.019, "exactly 50 Hz"),
        ("passage_hard_duration_s", 8.01, "passage duration"),
        ("crossing_status_timeout_s", 0.76, "crossing wait"),
        (
            "censored_passage_coast_max_duration_s",
            0.301,
            "censored passage coast",
        ),
        (
            "censored_passage_coast_max_fresh_frames",
            9,
            "discrete bounds",
        ),
        ("post_credit_fresh_frame_timeout_s", 0.21, "fresh-frame"),
        (
            "max_segment_yaw_excursion_rad",
            MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD + 0.001,
            "yaw excursion",
        ),
    ),
)
def test_course_limits_refuse_widened_safety_envelopes(
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        replace(VisualCourseStageLimits(), **{field: value})


class _SettableOrientation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def to_euler(self):
        return self.roll, self.pitch, self.yaw


def _set_attitude(
    host,
    *,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    rates=(0.0, 0.0, 0.0),
):
    host.estimate = SimpleNamespace(
        orientation=_SettableOrientation(roll, pitch, yaw),
        body_rates=rates,
    )


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_yaw_at_hard_excursion_zeroes_outward_but_allows_inward(direction):
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    excursion = direction * limits.max_segment_yaw_excursion_rad

    outward = course_stage._limit_calibrated_yaw_request(
        direction * 0.02,
        excursion_rad=excursion,
        measured_euler_yaw_rate_rad_s=0.0,
        limits=limits,
        profile=profile,
        abort_type=SafetyAbort,
    )
    assert outward == 0.0

    admitted = course_stage._limit_calibrated_yaw_request(
        -direction * 0.02,
        excursion_rad=excursion,
        measured_euler_yaw_rate_rad_s=0.0,
        limits=limits,
        profile=profile,
        abort_type=SafetyAbort,
    )
    assert admitted == -direction * 0.02


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_yaw_delayed_projection_at_soft_boundary_zeroes_outward(direction):
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    response_reserve = (
        profile.observed_max_abs_measured_yaw_rate_rad_s
        * profile.control_hold_horizon_s
    )
    soft_boundary = (
        limits.max_segment_yaw_excursion_rad - response_reserve
    )
    excursion = direction * (soft_boundary - 0.01)
    rate_to_boundary = direction * (
        0.01 / profile.max_gyro_response_delay_s
        + 1e-9
    )

    admitted = course_stage._limit_calibrated_yaw_request(
        direction * 0.02,
        excursion_rad=excursion,
        measured_euler_yaw_rate_rad_s=rate_to_boundary,
        limits=limits,
        profile=profile,
        abort_type=SafetyAbort,
    )
    assert admitted == 0.0


def test_course_wires_exact_zero_at_yaw_soft_stop_and_keeps_hard_guards():
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    response_reserve = (
        profile.observed_max_abs_measured_yaw_rate_rad_s
        / math.cos(
            max(
                abs(limits.min_measured_pitch_rad),
                abs(limits.max_measured_pitch_rad),
            )
        )
        * profile.control_hold_horizon_s
    )
    soft_boundary = (
        limits.max_segment_yaw_excursion_rad - response_reserve
    )
    excursion = soft_boundary - 0.01
    rate_to_soft_stop = (
        0.01 / profile.max_gyro_response_delay_s
        + 1e-6
    )

    class SoftStopHost(_Host):
        sample_count = 0

        def _sample(self):
            super()._sample()
            self.sample_count += 1
            if self.sample_count == 1:
                _set_attitude(
                    self,
                    yaw=excursion,
                    rates=(0.0, 0.0, rate_to_soft_stop),
                )
            else:
                _set_attitude(self)

    host = SoftStopHost(initial_gate=3, finish_gate=3)
    runtime, calls = _runtime(host, limits=limits)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    segment = result["segments"][0]
    assert result["success"] is True
    assert segment["yaw_soft_stop_zero_command_count"] == 1
    assert (
        segment["passage_admission_yaw_soft_stop_withheld_count"]
        == 1
    )
    assert calls[0][1] is VisualApproachMode.APPROACH
    assert calls[1][1] is VisualApproachMode.APPROACH
    assert calls[2][1] is VisualApproachMode.PASSAGE
    navigation = [
        command
        for command, kwargs, _gate in host.commands
        if kwargs.get("require_wire_receipt")
    ]
    assert navigation
    assert navigation[0].yaw_rate == 0.0
    assert all(command.yaw_rate == 0.02 for command in navigation[1:])
    events = [
        payload
        for event, payload in host.recorder.events
        if event == "visual_course_yaw_soft_stop_zeroed"
    ]
    assert len(events) == 1
    assert events[0]["requested_yaw_rate_rad_s"] == pytest.approx(0.02)
    assert events[0]["admitted_yaw_rate_rad_s"] == 0.0


@pytest.mark.parametrize(
    ("direction", "yaw_rate"),
    (
        (1.0, 0.4234286031848751),
        (-1.0, -0.50),
    ),
)
def test_measured_yaw_momentum_projects_outside_hard_boundary(
    direction,
    yaw_rate,
):
    limits = VisualCourseStageLimits()
    excursion = direction * (
        limits.max_segment_yaw_excursion_rad - 0.01
    )
    host = _Host()
    _set_attitude(host, yaw=excursion, rates=(0.0, 0.0, yaw_rate))

    with pytest.raises(SafetyAbort, match="momentum projects outside"):
        course_stage._assert_course_attitude_state(
            host,
            yaw_reference_rad=0.0,
            limits=limits,
            yaw_profile=_yaw_profile(),
            abort_type=SafetyAbort,
            phase="regression",
        )


def test_cross_axis_body_rates_use_exact_euler_yaw_kinematics():
    limits = VisualCourseStageLimits()
    host = _Host()
    _set_attitude(
        host,
        roll=0.18,
        pitch=-0.35,
        yaw=-(limits.max_segment_yaw_excursion_rad - 0.01),
        rates=(0.0, -0.5, -0.5),
    )

    with pytest.raises(SafetyAbort, match="momentum projects outside"):
        course_stage._assert_course_attitude_state(
            host,
            yaw_reference_rad=0.0,
            limits=limits,
            yaw_profile=_yaw_profile(),
            abort_type=SafetyAbort,
            phase="cross-axis-regression",
        )


@pytest.mark.parametrize(
    ("roll", "pitch"),
    (
        (0.180001, 0.0),
        (0.0, -0.350001),
        (0.0, 0.150001),
    ),
)
def test_measured_roll_pitch_envelopes_fail_closed(
    roll,
    pitch,
):
    host = _Host()
    _set_attitude(host, roll=roll, pitch=pitch)

    with pytest.raises(SafetyAbort, match="measured attitude envelope"):
        course_stage._assert_course_attitude_state(
            host,
            yaw_reference_rad=0.0,
            limits=VisualCourseStageLimits(),
            yaw_profile=_yaw_profile(),
            abort_type=SafetyAbort,
            phase="regression",
        )


@pytest.mark.parametrize(
    "rates",
    (
        (0.500001, 0.0, 0.0),
        (0.0, -0.500001, 0.0),
    ),
)
def test_measured_roll_pitch_rate_corridor_is_recorded_as_diagnostic(rates):
    host = _Host()
    _set_attitude(host, rates=rates)

    _excursion, admitted_rates, _euler_yaw_rate = (
        course_stage._assert_course_attitude_state(
            host,
            yaw_reference_rad=0.0,
            limits=VisualCourseStageLimits(),
            yaw_profile=_yaw_profile(),
            abort_type=SafetyAbort,
            phase="diagnostic-regression",
        )
    )

    assert admitted_rates == rates
    events = [
        payload
        for event, payload in host.recorder.events
        if event == "visual_course_measured_body_rate_corridor_exceeded"
    ]
    assert len(events) == 1
    assert events[0]["phase"] == "diagnostic-regression"
    assert events[0]["disposition"] == "diagnostic_only"
    assert events[0]["threshold_rad_s"] == 0.50
    assert events[0]["peak_abs_body_rate_rad_s"] == pytest.approx(0.500001)
    assert events[0]["measured_body_rates_rad_s"] == list(rates)


def test_repeated_same_sign_yaw_requests_zero_before_hard_boundary():
    profile = _yaw_profile()
    limits = VisualCourseStageLimits()
    excursion = 0.0
    admitted = 0

    while True:
        request = course_stage._limit_calibrated_yaw_request(
            0.02,
            excursion_rad=excursion,
            measured_euler_yaw_rate_rad_s=0.0,
            limits=limits,
            profile=profile,
            abort_type=SafetyAbort,
        )
        if request == 0.0:
            break
        admitted += 1
        excursion += (
            profile.observed_max_abs_measured_yaw_rate_rad_s
            * limits.control_period_s
        )
        assert admitted < 200

    assert admitted > 0
    assert excursion < limits.max_segment_yaw_excursion_rad


def test_duplicate_camera_frame_cannot_hide_new_unsafe_attitude():
    limits = VisualCourseStageLimits()

    class DuplicateUnsafeHost(_Host):
        sample_count = 0

        def _sample(self):
            self.sample_count += 1
            if self.sample_count == 2:
                _set_attitude(
                    self,
                    yaw=limits.max_segment_yaw_excursion_rad + 0.000001,
                )
                return
            super()._sample()

    host = DuplicateUnsafeHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="yaw envelope"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert len(host.commands) == 1
    assert host._visual_course_summary["success"] is False


def test_crossing_hold_checks_attitude_before_sending():
    class CrossingUnsafeHost(_Host):
        crossing_loss_observed = False

        def _sample(self):
            if self.crossing_loss_observed:
                super()._sample()
                _set_attitude(self, roll=0.180001)
                return
            super()._sample()
            if (
                self.passage_counts.get(self.current_gate, 0) >= 3
                and not self.visual_gate_graph.latest_snapshot.current_track.visible
            ):
                self.crossing_loss_observed = True

    host = CrossingUnsafeHost(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="measured attitude envelope",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert len(host.commands) == 4
    assert not any(command.thrust == 0.0 for command, _kwargs, _gate in host.commands)


def test_crossing_hold_aborts_new_observable_axis_divergence():
    class CrossingDivergentHost(_Host):
        def _sample(self):
            super()._sample()
            if self.crossing_hold_count >= 1:
                snapshot = self.visual_gate_graph.latest_snapshot
                track = snapshot.current_track
                track.visible = True
                track.missed_frame_count = 0
                track.latest_token = snapshot.latest_camera_token
                track.center_norm = (1.0, 0.0)
                snapshot.authority_usable = True

    host = CrossingDivergentHost(
        initial_gate=6,
        finish_gate=6,
        lose_before_credit=True,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="credit-wait measurement became unsafe",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.crossing_hold_count == 1
    assert not any(command.thrust == 0.0 for command, _kwargs, _gate in host.commands)


def test_credit_wait_uses_one_stable_adjacent_without_advance():
    class AdjacentCreditWaitHost(_Host):
        def __init__(self):
            super().__init__(
                initial_gate=6,
                finish_gate=7,
                lose_before_credit=True,
            )
            self.adjacent_track = None
            self.adjacent_stable_frames = 0
            self.adjacent_command_count = 0

        def _sample(self):
            super()._sample()
            snapshot = self.visual_gate_graph.latest_snapshot
            snapshot.next_candidates = ()
            snapshot.next_selection_ambiguous = False
            snapshot.provisional_track_ids = ()
            if (
                self.current_gate == 6
                and not snapshot.current_track.visible
            ):
                self.adjacent_stable_frames += 1
                token = snapshot.latest_camera_token
                adjacent = _track(
                    "track-7",
                    gate_index=7,
                    token=token,
                )
                adjacent.role = VisualTrackRole.NEXT
                adjacent.authoritative_gate_index = None
                adjacent.center_norm = (0.55, -0.55)
                adjacent.center_velocity_norm_s = (0.20, -0.20)
                self.adjacent_track = adjacent
                if self.adjacent_stable_frames >= 3:
                    snapshot.next_candidates = (
                        SimpleNamespace(
                            track_id=adjacent.track_id,
                            latest_token=token,
                            promotable=False,
                            stable_frame_count=3,
                            confidence=0.80,
                            association_confidence=0.80,
                            relationship=None,
                        ),
                    )

        async def _send_flight_command(self, command, **kwargs):
            receipt = await super()._send_flight_command(
                command,
                **kwargs,
            )
            if self.current_gate == 6 and command.thrust == 0.27:
                self.adjacent_command_count += 1
                if self.adjacent_command_count >= 2:
                    self._advance_race()
            return receipt

    host = AdjacentCreditWaitHost()
    runtime, calls = _runtime(host)

    result = asyncio.run(
        run_visual_course_stage(host, _context(), runtime=runtime)
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    transition = result["authoritative_transitions"][0]
    assert transition["from_gate_index"] == 6
    assert transition["to_gate_index"] == 7
    assert transition["crossing_wait_coast_command_count"] == 1
    assert transition["crossing_wait_adjacent_command_count"] == 2
    assert transition["crossing_wait_zero_command_count"] == 0
    adjacent_tick_indexes = [
        index
        for index, (stage, _elapsed, _command) in enumerate(host.ticks)
        if stage == "visual-course/gate6/credit-wait-adjacent"
    ]
    assert len(adjacent_tick_indexes) == 2
    assert all(
        host.commands[index][2] == 6
        and host.commands[index][0].thrust == 0.27
        for index in adjacent_tick_indexes
    )
    assert host.intercept_response_authorities.count(1.0) >= transition[
        "crossing_wait_adjacent_command_count"
    ]
    assert any(
        gate_index == 7
        and mode is VisualApproachMode.ADJACENT_RECENTER
        for gate_index, mode, *_rest in calls
    )
    assert any(
        gate_index == 7
        and mode is VisualApproachMode.PROMOTE_REACQUIRE
        for gate_index, mode, *_rest in calls
    )


def test_post_credit_wait_checks_attitude_before_sending_zero():
    class RecoveryUnsafeHost(_Host):
        def _sample(self):
            super()._sample()
            if (
                self.after_promotion_samples is not None
                and self.after_promotion_samples > 0
            ):
                _set_attitude(self, pitch=0.150001)

    host = RecoveryUnsafeHost(
        initial_gate=1,
        finish_gate=2,
        fresh_after_samples=999,
    )
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="measured attitude envelope",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert [
        command
        for command, _kwargs, gate_index in host.commands
        if gate_index == 2
    ] == []


def test_command_slot_wait_cannot_invalidate_attitude_guard():
    class SlotUnsafeHost(_Host):
        async def _wait_for_next_flight_command_slot(self):
            ready = await super()._wait_for_next_flight_command_slot()
            _set_attitude(self, roll=0.180001)
            return ready

    host = SlotUnsafeHost(initial_gate=0, finish_gate=0)
    runtime, _calls = _runtime(host)

    with pytest.raises(
        SafetyAbort,
        match="measured attitude envelope",
    ):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert host.commands == []


def test_terminal_race_finish_checks_attitude_before_success_return():
    limits = VisualCourseStageLimits()

    class TerminalUnsafeHost(_Host):
        def _visual_race_status_ref(self):
            if self.race.race_finished:
                _set_attitude(
                    self,
                    yaw=-(limits.max_segment_yaw_excursion_rad + 0.000001),
                )
            return self.race

    host = TerminalUnsafeHost(initial_gate=4, finish_gate=4)
    runtime, _calls = _runtime(host)

    with pytest.raises(SafetyAbort, match="yaw envelope"):
        asyncio.run(
            run_visual_course_stage(host, _context(), runtime=runtime)
        )

    assert len(host.commands) == 4
    assert host._visual_course_summary["success"] is False
    assert all(
        event != "visual_course_complete"
        for event, _payload in host.recorder.events
    )


def test_production_servo_sign_matches_frozen_image_sign_authority():
    profile = _yaw_profile()
    snapshot = _snapshot(0, "track-0", 10)
    target = _target(snapshot, "track-0")
    servo = ImageVisualServo(default_visual_config().servo)

    output = servo.step(
        target,
        now_monotonic_s=target.received_monotonic_s,
        segment_elapsed_s=0.0,
        segment_yaw_excursion_rad=0.0,
        allow_advance=False,
    )

    assert profile.controller_to_image_sign == 1
    assert target.normalized_x > 0.0
    assert output.yaw_rate_rad_s < 0.0
    assert (
        output.yaw_rate_rad_s
        * target.normalized_x
        * profile.controller_to_image_sign
        < 0.0
    )


class _RunnerAdapter:
    enable_vision = False
    telemetry_mode = "imu"
    fetch_track_on_connect = False

    def __init__(self):
        self.race_status = RaceStatus(1000, 0, -1, 0, -1)


class _RunnerVision:
    pass


@pytest.mark.parametrize(
    ("race_finished", "ordered_transitions", "cleanup_confirmed", "success"),
    (
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ),
)
def test_runner_course_boundary_requires_finish_chain_and_cleanup(
    monkeypatch,
    race_finished,
    ordered_transitions,
    cleanup_confirmed,
    success,
):
    adapter = _RunnerAdapter()
    profile = load_yaw_calibration_profile()
    profile_evidence = yaw_calibration_profile_evidence(profile)
    runner = vq2_run.VQ2Runner(
        adapter,
        _RunnerVision(),
        yaw_calibration_profile=profile,
        yaw_calibration_profile_evidence=profile_evidence,
    )
    context = vq2_run.StartContext(
        0.0,
        -0.31,
        320,
        180,
        6_400,
        1_000,
    )

    async def no_result(*_args, **_kwargs):
        return None

    async def wait_for_go():
        adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
        return context

    async def run_course(_context):
        assert _context is context
        adapter.race_status = RaceStatus(
            1500,
            0,
            50_000_000 if race_finished else -1,
            2,
            500,
        )
        pairs = (
            [(0, 1), (1, 2)]
            if ordered_transitions
            else [(0, 2)]
        )
        graph_transitions = tuple(
            SimpleNamespace(
                from_gate_index=from_index,
                to_gate_index=to_index,
                retired_track_id=f"track-{from_index}",
                promoted_track_id=f"track-{to_index}",
            )
            for from_index, to_index in pairs
        )
        finish_ref = AuthoritativeRaceStatusRef.live(
            session_id="course-boundary-test",
            reset_epoch=1,
            race_generation=3,
            race_status_sequence=9,
            race_status_boot_ms=1500,
            active_gate_index=2,
            received_monotonic_ns=20_000_000,
            host_clock_id="host-perf-counter",
            race_finished=race_finished,
        )
        runner.visual_gate_graph = SimpleNamespace(
            latest_snapshot=SimpleNamespace(
                confirmed_transitions=graph_transitions,
                race_finished=race_finished,
                current_gate_index=2,
                latest_race_status=finish_ref,
            )
        )
        runner._visual_transition = graph_transitions[-1]
        summary = {
            "stage": vq2_run.VISUAL_COURSE_STAGE,
            "success": True,
            "race_finished": race_finished,
            "initial_gate_index": 0,
            "maximum_authoritative_gate_index": 2,
            "final_gate_index": 2,
            "authoritative_transitions": [
                {
                    "from_gate_index": item.from_gate_index,
                    "to_gate_index": item.to_gate_index,
                }
                for item in graph_transitions
            ],
            "segments": [
                {
                    "gate_index": 0,
                    "passage_authority_enabled": True,
                }
            ],
            "visual_navigation_command_count": 8,
            "exact_zero_command_count": 3,
            "yaw_calibration_profile": profile_evidence,
        }
        runner._visual_course_summary = dict(summary)
        return summary

    async def cleanup():
        return cleanup_confirmed

    monkeypatch.setattr(runner, "establish_reset_epoch", no_result)
    monkeypatch.setattr(runner, "normalize_disarmed", no_result)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(
        runner,
        "_bind_initial_visual_gate",
        lambda _context: None,
    )
    monkeypatch.setattr(runner, "arm_confirmed", no_result)
    monkeypatch.setattr(runner, "_run_visual_course", run_course)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(
        runner.run_powered_stage(
            vq2_run.VISUAL_COURSE_STAGE,
            write_diagnostic_pngs=False,
        )
    )

    assert result.success is success
    assert result.cleanup_confirmed is cleanup_confirmed
    assert result.details["authoritative_cleanup_entry"][
        "transitions"
    ] == [list(pair) for pair in (
        [(0, 1), (1, 2)]
        if ordered_transitions
        else [(0, 2)]
    )]
    assert result.details["visual_course"]["cleanup_confirmed"] is (
        cleanup_confirmed
    )
    if not race_finished or not ordered_transitions:
        assert "race_finished/ordered rolling-graph proof" in result.reason
    if not cleanup_confirmed:
        assert "cleanup unconfirmed" in result.reason
