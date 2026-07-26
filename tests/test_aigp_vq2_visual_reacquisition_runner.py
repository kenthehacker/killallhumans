from __future__ import annotations

from types import SimpleNamespace

import pytest

from competition.vq2_visual_tracker import (
    CameraFrameToken,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateReacquisition,
    CreditedUnboundGateAdvance,
    GateGraphError,
    GateReacquisitionPending,
)
from scripts.aigp_vq2_run import SafetyAbort, VQ2Runner


def _race() -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef.live(
        session_id="runner-reacquisition",
        reset_epoch=4,
        race_generation=7,
        race_status_sequence=12,
        race_status_boot_ms=2_400,
        active_gate_index=1,
        received_monotonic_ns=1_000_000_000,
        host_clock_id="host-perf-counter",
    )


def _token(sequence: int) -> CameraFrameToken:
    return CameraFrameToken(
        generation=4,
        frame_id=8_000 + sequence,
        publication_sequence=sequence,
        stream_id="runner-reacquisition-camera",
    )


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, payload))


def _advance(
    race: AuthoritativeRaceStatusRef,
    credit_token: CameraFrameToken,
) -> CreditedUnboundGateAdvance:
    return CreditedUnboundGateAdvance(
        from_gate_index=0,
        to_gate_index=1,
        retired_track_id="current-zero",
        reviewed_track_id="reviewed-next",
        race_status=race,
        camera_token_at_credit=credit_token,
        reviewed_first_token=_token(2),
        reviewed_latest_token_before_credit=_token(8),
        reviewed_history_length_at_credit=7,
        reviewed_history_length_at_advance=7,
        reviewed_history_sha256="a" * 64,
        alternative_reacquisition_track_ids_at_credit=(),
    )


@pytest.mark.parametrize("same_reviewed_identity", (False, True))
def test_runner_records_unbound_credit_then_exact_reacquisition_boundary(
    same_reviewed_identity,
):
    runner = object.__new__(VQ2Runner)
    race = _race()
    credit_token = _token(10)
    binding_token = _token(13)
    advance = _advance(race, credit_token)
    reacquired_track_id = (
        "reviewed-next" if same_reviewed_identity else "fresh-next"
    )
    reacquisition = ConfirmedGateReacquisition(
        credited_advance=advance,
        gate_index=1,
        reacquired_track_id=reacquired_track_id,
        camera_token_at_binding=binding_token,
        reacquired_first_token=_token(11),
        stable_frame_tokens=(_token(11), _token(12), binding_token),
        history_length_at_binding=3,
        history_sha256="b" * 64,
        cross_gap_identity_claimed=False,
    )

    class Graph:
        latest_snapshot = object()

        def confirm_reviewed_advance(self, *_args, **kwargs):
            assert kwargs == {
                "race_status": race,
                "camera_token_at_credit": credit_token,
                "reviewed_track_id": "reviewed-next",
            }
            return advance

        def try_confirm_reacquired_current(self, *_args, **kwargs):
            assert kwargs == {
                "credited_advance": advance,
                "camera_token_at_binding": binding_token,
            }
            return reacquisition

    runner.visual_gate_graph = Graph()
    runner.visual_tracker = SimpleNamespace(
        latest_update=SimpleNamespace(token=binding_token)
    )
    runner.recorder = _Recorder()
    runner._visual_transition = None
    runner._visual_unbound_advance = None
    runner._visual_latest_graph_snapshot = None
    runner._visual_camera_token_at_race_credit = (
        lambda supplied_race: credit_token
    )
    runner._visual_graph_summary = lambda _snapshot: {"phase": "test"}

    observed_advance = runner._confirm_visual_course_advance(
        from_gate_index=0,
        to_gate_index=1,
        race_status=race,
        reviewed_track_id="reviewed-next",
    )
    observed_reacquisition = runner._try_visual_reacquired_current()

    assert observed_advance is advance
    assert observed_reacquisition is reacquisition
    assert runner._visual_unbound_advance is None
    assert runner._visual_transition is reacquisition
    assert [event for event, _payload in runner.recorder.events] == [
        "visual_gate_credited_unbound",
        "visual_gate_reacquired",
    ]
    assert runner.recorder.events[-1][1]["identity_basis"] == (
        "retained-reviewed-local-track"
        if same_reviewed_identity
        else "fresh-unique-local-track"
    )


def test_runner_returns_soft_reacquisition_pending_without_recording_boundary():
    runner = object.__new__(VQ2Runner)
    race = _race()
    credit_token = _token(10)
    binding_token = _token(11)
    advance = _advance(race, credit_token)
    pending = GateReacquisitionPending(
        reason="no unique clean locally stable successor is ready",
        ambiguous=False,
    )

    class Graph:
        latest_snapshot = object()

        def try_confirm_reacquired_current(self, *_args, **kwargs):
            assert kwargs == {
                "credited_advance": advance,
                "camera_token_at_binding": binding_token,
            }
            return pending

    runner.visual_gate_graph = Graph()
    runner.visual_tracker = SimpleNamespace(
        latest_update=SimpleNamespace(token=binding_token)
    )
    runner.recorder = _Recorder()
    runner._visual_transition = None
    runner._visual_unbound_advance = advance
    runner._visual_latest_graph_snapshot = None

    outcome = runner._try_visual_reacquired_current()

    assert outcome is pending
    assert runner._visual_transition is None
    assert runner._visual_unbound_advance is advance
    assert runner.recorder.events == []


def test_runner_converts_hard_course_advance_graph_error_to_safety_abort():
    runner = object.__new__(VQ2Runner)
    race = _race()
    credit_token = _token(10)

    class Graph:
        def confirm_reviewed_advance(self, *_args, **_kwargs):
            raise GateGraphError("current gate is not bound")

    runner.visual_gate_graph = Graph()
    runner.visual_tracker = object()
    runner._visual_camera_token_at_race_credit = (
        lambda supplied_race: credit_token
    )

    with pytest.raises(SafetyAbort, match="current gate is not bound"):
        runner._confirm_visual_course_advance(
            from_gate_index=0,
            to_gate_index=1,
            race_status=race,
            reviewed_track_id="reviewed-next",
        )
