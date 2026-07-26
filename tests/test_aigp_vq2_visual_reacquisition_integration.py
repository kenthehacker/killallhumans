"""Derived-state integration for the credited-unbound course handoff.

This is deliberately not JPEG, detector, UDP receiver, FlightSim dynamics, or
full recorded-sequence replay.  It constructs detections from the compact
candidate-1 tracker facts, holds a fixed healthy IMU attitude, and uses a
deterministic servo.  A blank image exists only so the production receiver
publication lease can prove the exact camera token at the wire boundary.

Within those limits, the boundary under test is production code end to end:
``MultiTargetVisualTracker`` -> ``RollingVisualGateGraph`` -> typed
``VQ2Runner`` course advance/reacquisition -> visual-course coordinator ->
``VQ2Runner._send_flight_command`` publication lease and atomic race-active
wire admission.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import math
import time

import numpy as np

from competition.adapter import (
    CameraFrame,
    RaceActiveBoundaryChangedBeforeWire,
)
from competition.aigp_messages import RaceStatus
from competition.vq2_capture import (
    MavlinkIngressV1,
    RaceStatusPayloadV1,
    ReceivedRaceStatusV1,
)
from competition.vq2_contracts import FrameIdentityV1, FrameTimingV1
from competition.vq2_vision import VQ2VisionSnapshot, VQ2VisionThread
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualDetection,
    VisualDetectionFrame,
    VisualTrackRole,
)
from planning.vq2_gate_graph import (
    ConfirmedGateReacquisition,
    GateGraphPhase,
)
from scripts.aigp_vq2_run import (
    VISUAL_COURSE_STAGE,
    VQ2Runner,
)
from scripts.aigp_vq2_visual_course_stage import run_visual_course_stage
from tests.test_aigp_vq2_runner import _FakeAdapter, _estimate
from tests.test_aigp_vq2_visual_course import (
    _context,
    _runtime,
)
from tests.test_aigp_vq2_visual_course_coordinator_replay import (
    _CadencedCoordinatorServo,
)


_FRAME_PERIOD_NS = 33_000_000
_PUBLISH_DELAY_NS = 1_000_000
_RECORDED_REVIEWED_TO_SUCCESSOR_GAP_NS = 464_904_700
_RECORDED_GAP_EXTRA_NS = 2_904_700
_HOST_CLOCK_ID = "host-perf-counter"
_CAMERA_STREAM_ID = "vq2-camera-udp-5600"
_GENERATION = 8


def _detection(
    source_index: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    *,
    confidence: float = 0.9,
) -> VisualDetection:
    center_unit_x = 0.5 * (center_x + 1.0)
    center_unit_y = 0.5 * (center_y + 1.0)
    return VisualDetection(
        source_index=source_index,
        center_norm=(center_x, center_y),
        bbox_norm=(
            center_unit_x - width / 2.0,
            center_unit_y - height / 2.0,
            center_unit_x + width / 2.0,
            center_unit_y + height / 2.0,
        ),
        confidence=confidence,
    )


def _current_detection(sequence: int) -> VisualDetection:
    return _detection(
        0,
        -0.03 + 0.001 * sequence,
        0.02,
        0.42,
        0.44,
        confidence=0.95,
    )


def _reviewed_detection(sequence: int) -> VisualDetection:
    return _detection(
        1,
        0.435 + 0.006 * (sequence - 1),
        -0.443 - 0.006 * (sequence - 1),
        0.105,
        0.135,
    )


def _successor_detection(sequence: int) -> VisualDetection:
    return _detection(
        1,
        0.553 + 0.005 * (sequence - 19),
        -0.572 - 0.005 * (sequence - 19),
        0.140,
        0.183,
    )


class _EventRecorder:
    path = None
    capture_enabled = False
    capture_fifo_enabled = False

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, payload))


def _received_race_status(
    *,
    sequence: int,
    boot_ms: int,
    gate_index: int,
    received_ns: int,
    race_finished: bool = False,
) -> ReceivedRaceStatusV1:
    payload = RaceStatusPayloadV1(
        sim_boot_time_ms=boot_ms,
        race_start_boot_time_ms=500,
        race_finish_time_ns=(9_000_000_000 if race_finished else -1),
        active_gate_index=gate_index,
        last_gate_race_time=(-1 if gate_index == 0 else 750_000_000),
    )
    return ReceivedRaceStatusV1(
        ingress=MavlinkIngressV1(
            stream_id="vq2-mavlink-udp-14550",
            generation=3,
            sequence=sequence,
            message_type="RACE_STATUS",
            host_clock_id=_HOST_CLOCK_ID,
            received_monotonic_ns=received_ns,
            source_time_value=boot_ms,
            source_time_unit="ms",
        ),
        race_status=payload,
    )


class _RaceWireAdapter(_FakeAdapter):
    """In-memory transport retaining the runner's exact wire contract."""

    def __init__(self) -> None:
        super().__init__()
        self.is_armed = True
        self.latest_received_race_status: ReceivedRaceStatusV1 | None = None
        self.race_active_wires: list[dict[str, object]] = []
        self.finish_pending = False

    def publish_race(
        self,
        *,
        sequence: int,
        boot_ms: int,
        gate_index: int,
        race_finished: bool = False,
    ) -> ReceivedRaceStatusV1:
        received = _received_race_status(
            sequence=sequence,
            boot_ms=boot_ms,
            gate_index=gate_index,
            received_ns=time.perf_counter_ns(),
            race_finished=race_finished,
        )
        payload = received.race_status
        self.latest_received_race_status = received
        self.race_status = RaceStatus(
            sim_boot_time_ms=payload.sim_boot_time_ms,
            race_start_boot_time_ms=payload.race_start_boot_time_ms,
            race_finish_time_ns=payload.race_finish_time_ns,
            active_gate_index=payload.active_gate_index,
            last_gate_race_time=payload.last_gate_race_time,
        )
        return received

    async def send_attitude_rate_if_race_active(
        self,
        command,
        *,
        expected_active_gate_index,
        **kwargs,
    ):
        received = self.latest_received_race_status
        if (
            received is None
            or received.race_status.active_gate_index
            != expected_active_gate_index
            or received.race_status.race_finish_time_ns >= 0
        ):
            raise RaceActiveBoundaryChangedBeforeWire(
                "derived race-active boundary changed before wire"
            )
        await self.send_attitude_rate(command, **kwargs)
        receipt = self.outbound_receipts[-1]
        self.race_active_wires.append(
            {
                "gate_index": expected_active_gate_index,
                "command": command,
                "call_start_monotonic_ns": receipt[
                    "call_start_monotonic_ns"
                ],
                "race_status": received,
            }
        )
        if (
            expected_active_gate_index == 1
            and any(
                value != 0.0
                for value in (
                    command.roll_rate,
                    command.pitch_rate,
                    command.yaw_rate,
                    command.thrust,
                )
            )
        ):
            self.finish_pending = True
        return {
            "schema": "aigp-vq2-race-active-send-authority/1",
            "expected_active_gate_index": expected_active_gate_index,
            "received_race_status": received.to_primitive(),
        }


class _DerivedStateCourseRunner(VQ2Runner):
    """Runner with only camera/race ingress and physical watchdog simulated."""

    def __init__(self) -> None:
        self.test_recorder = _EventRecorder()
        adapter = _RaceWireAdapter()
        vision = VQ2VisionThread()
        super().__init__(
            adapter,
            vision,
            recorder=self.test_recorder,
            visual_session_id="derived-reacquisition-integration",
        )
        self._visual_tracking_enabled = True
        self._visual_active_stage = VISUAL_COURSE_STAGE
        self.estimate = _estimate(roll=0.0, pitch=-0.05, yaw=0.0)
        self.watchdog_calls = 0
        self.reviewed_retired_before_successor = False
        self.successor_track_id: str | None = None
        self.credit_received: ReceivedRaceStatusV1 | None = None
        self.finish_received: ReceivedRaceStatusV1 | None = None
        self.last_published_sequence = 0
        self._next_sequence = 1
        self._credit_published = False

        now_ns = time.perf_counter_ns()
        # Publication 5 is fresh at coordinator entry.  The prescribed
        # 33 ms deltas then place successor publication 19 exactly
        # 464,904,700 ns after the reviewed identity's last observation.
        self._origin_ns = (
            now_ns
            - 5 * _FRAME_PERIOD_NS
            - _PUBLISH_DELAY_NS
            - 2_000_000
        )
        self._postcredit_observation_floor_ns: int | None = None

        first_track_ids: tuple[str, ...] = ()
        for sequence in range(1, 6):
            update = self._publish_frame(
                sequence,
                (
                    _current_detection(sequence),
                    _reviewed_detection(sequence),
                ),
                observe_graph=False,
            )
            if sequence == 1:
                first_track_ids = update.created_track_ids
        assert len(first_track_ids) == 2
        self.current_track_id, self.reviewed_track_id = first_track_ids
        baseline = adapter.publish_race(
            sequence=100,
            boot_ms=1_000,
            gate_index=0,
        )
        baseline_ref = self._visual_race_status_ref()
        assert (
            baseline_ref.received_monotonic_ns
            == baseline.ingress.received_monotonic_ns
        )
        self._visual_latest_graph_snapshot = (
            self.visual_gate_graph.bind_initial_current(
                self.visual_tracker,
                track_id=self.current_track_id,
                race_status=baseline_ref,
            )
        )
        self._visual_latest_tracker_update = (
            self.visual_tracker.latest_update
        )
        self._next_sequence = 6

    @property
    def replay_adapter(self) -> _RaceWireAdapter:
        assert type(self.adapter) is _RaceWireAdapter
        return self.adapter

    def _frame_times(self, sequence: int) -> tuple[int, int]:
        seam_extra_ns = (
            _RECORDED_GAP_EXTRA_NS if sequence >= 19 else 0
        )
        observation_ns = (
            self._origin_ns
            + sequence * _FRAME_PERIOD_NS
            + seam_extra_ns
        )
        if (
            sequence >= 26
            and self._postcredit_observation_floor_ns is not None
        ):
            observation_ns = max(
                observation_ns,
                self._postcredit_observation_floor_ns,
            )
        return observation_ns, observation_ns + _PUBLISH_DELAY_NS

    def _snapshot(
        self,
        sequence: int,
        observation_ns: int,
        publication_ns: int,
    ) -> VQ2VisionSnapshot:
        frame_id = 120_000 + sequence
        camera_source_ns = 30_000_000_000 + sequence * _FRAME_PERIOD_NS
        timing = FrameTimingV1(
            identity=FrameIdentityV1(
                _CAMERA_STREAM_ID,
                _GENERATION,
                frame_id,
            ),
            camera_source_time_ns=camera_source_ns,
            host_clock_id=_HOST_CLOCK_ID,
            publication_sequence=sequence,
            first_unique_packet_monotonic_ns=observation_ns - 1_000_000,
            final_unique_packet_monotonic_ns=observation_ns,
            reassembly_complete_monotonic_ns=observation_ns,
            decode_start_monotonic_ns=observation_ns,
            decode_end_monotonic_ns=publication_ns,
            publish_monotonic_ns=publication_ns,
        )
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image.setflags(write=False)
        return VQ2VisionSnapshot(
            frame_id=frame_id,
            camera_frame=CameraFrame(
                timestamp_us=camera_source_ns // 1_000,
                image=image,
            ),
            sim_time_ns=camera_source_ns,
            received_monotonic_s=observation_ns / 1_000_000_000.0,
            generation=_GENERATION,
            timing=timing,
        )

    def _publish_frame(
        self,
        sequence: int,
        detections: tuple[VisualDetection, ...],
        *,
        observe_graph: bool = True,
    ):
        observation_ns, publication_ns = self._frame_times(sequence)
        snapshot = self._snapshot(
            sequence,
            observation_ns,
            publication_ns,
        )
        with self.vision._data_lock:
            self.vision._latest_snapshot = snapshot
        frame = VisualDetectionFrame(
            token=CameraFrameToken(
                generation=_GENERATION,
                frame_id=120_000 + sequence,
                publication_sequence=sequence,
                stream_id=_CAMERA_STREAM_ID,
            ),
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            time_basis_id=_HOST_CLOCK_ID,
            image_size_px=(640, 360),
            detections=detections,
            camera_source_time_ns=snapshot.sim_time_ns,
            final_unique_packet_monotonic_ns=observation_ns,
            publish_monotonic_ns=publication_ns,
        )
        update = self.visual_tracker.update(frame)
        if observe_graph:
            self._visual_latest_graph_snapshot = (
                self.visual_gate_graph.observe(self.visual_tracker)
            )
            self._visual_latest_tracker_update = (
                self.visual_tracker.latest_update
            )
        self.last_published_sequence = sequence
        return update

    def _next_detections(
        self,
        sequence: int,
    ) -> tuple[VisualDetection, ...]:
        if self._credit_published:
            return (_successor_detection(sequence),)
        if sequence <= 18:
            return (_current_detection(sequence),)
        return (
            _current_detection(sequence),
            _successor_detection(sequence),
        )

    def _near_plane_latched(self) -> bool:
        return any(
            event == "visual_course_near_plane_latched"
            for event, _payload in self.test_recorder.events
        )

    def _sample(self) -> None:
        adapter = self.replay_adapter
        if adapter.finish_pending and self.finish_received is None:
            self.finish_received = adapter.publish_race(
                sequence=102,
                boot_ms=1_500,
                gate_index=1,
                race_finished=True,
            )
            return

        if (
            not self._credit_published
            and self.last_published_sequence >= 25
            and self._near_plane_latched()
        ):
            self.credit_received = adapter.publish_race(
                sequence=101,
                boot_ms=1_250,
                gate_index=1,
            )
            self._credit_published = True
            self._postcredit_observation_floor_ns = (
                self.credit_received.ingress.received_monotonic_ns
                + 5_000_000
            )
            return

        if self._next_sequence <= 40:
            _observation_ns, publication_ns = self._frame_times(
                self._next_sequence
            )
            if time.perf_counter_ns() >= publication_ns:
                sequence = self._next_sequence
                if sequence == 19:
                    reviewed = self.visual_tracker.track(
                        self.reviewed_track_id
                    )
                    self.reviewed_retired_before_successor = (
                        reviewed.role is VisualTrackRole.RETIRED
                    )
                update = self._publish_frame(
                    sequence,
                    self._next_detections(sequence),
                )
                if sequence == 19:
                    created = tuple(
                        track_id
                        for track_id in update.created_track_ids
                        if track_id != self.current_track_id
                    )
                    assert len(created) == 1
                    self.successor_track_id = created[0]
                self._next_sequence += 1
                return

    def _watchdog(self, **kwargs) -> None:
        # Physical dynamics and stream transport are outside this regression;
        # still fail the fixture if its fixed safety state is not benign.
        assert kwargs["enforce_benign_pad_budget"] is True
        assert self.replay_adapter.is_armed
        assert self.replay_adapter.drain_collisions() == []
        assert self.estimate is not None and self.estimate.healthy
        roll, pitch, yaw = self.estimate.orientation.to_euler()
        assert all(
            math.isfinite(value)
            for value in (roll, pitch, yaw, *self.estimate.body_rates)
        )
        self.watchdog_calls += 1


def _real_course_runtime(
    host: _DerivedStateCourseRunner,
):
    base_runtime, _calls = _runtime(host)
    servo_calls: list[object] = []

    def servo_factory(*args, **kwargs):
        return _CadencedCoordinatorServo(
            *args,
            **kwargs,
            calls=servo_calls,
            yaw_rate=0.0,
            preview_track_id=host.reviewed_track_id,
        )

    return replace(
        base_runtime,
        monotonic=time.monotonic,
        perf_counter_ns=time.perf_counter_ns,
        sleep=asyncio.sleep,
        servo_factory=servo_factory,
    )


def _nonzero(command) -> bool:
    return any(
        value != 0.0
        for value in (
            command.roll_rate,
            command.pitch_rate,
            command.yaw_rate,
            command.thrust,
        )
    )


def test_derived_candidate1_handoff_reaches_real_gate1_wire_admission():
    host = _DerivedStateCourseRunner()

    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_real_course_runtime(host),
        )
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    assert result["maximum_authoritative_gate_index"] == 1
    transition = result["authoritative_transitions"][0]
    assert (
        transition["from_gate_index"],
        transition["to_gate_index"],
    ) == (0, 1)
    assert transition["promotion_mode"] == "fresh_reacquisition"
    assert transition["post_transition_navigation_command_count"] >= 1

    assert host.reviewed_retired_before_successor is True
    assert host.successor_track_id is not None
    reviewed = host.visual_tracker.track(host.reviewed_track_id)
    successor = host.visual_tracker.track(host.successor_track_id)
    assert reviewed.role is VisualTrackRole.RETIRED
    assert successor.authoritative_gate_index == 1
    assert (
        successor.history[0].observation_monotonic_ns
        - reviewed.history[-1].observation_monotonic_ns
        == _RECORDED_REVIEWED_TO_SUCCESSOR_GAP_NS
    )

    reacquisition = host._visual_transition
    assert type(reacquisition) is ConfirmedGateReacquisition
    assert reacquisition.reacquired_track_id == host.successor_track_id
    assert reacquisition.identity_basis == "fresh-unique-local-track"
    assert reacquisition.cross_gap_identity_claimed is False
    assert (
        host.visual_gate_graph.latest_snapshot.phase
        is GateGraphPhase.RACE_FINISHED
    )

    event_names = [
        event for event, _payload in host.test_recorder.events
    ]
    credited_event_index = event_names.index(
        "visual_gate_credited_unbound"
    )
    reacquired_event_index = event_names.index("visual_gate_reacquired")
    assert credited_event_index < reacquired_event_index

    gate1_wires = [
        wire
        for wire in host.replay_adapter.race_active_wires
        if wire["gate_index"] == 1 and _nonzero(wire["command"])
    ]
    assert gate1_wires
    first_gate1_wire = gate1_wires[0]
    assert host.credit_received is not None
    assert (
        first_gate1_wire["call_start_monotonic_ns"]
        > host.credit_received.ingress.received_monotonic_ns
    )

    binding_token = reacquisition.camera_token_at_binding
    assert (
        binding_token.publication_sequence
        > (
            reacquisition.credited_advance.camera_token_at_credit
            .publication_sequence
        )
    )
    binding_sample = next(
        sample
        for sample in successor.history
        if sample.token == binding_token
    )
    assert (
        binding_sample.observation_monotonic_ns
        > host.credit_received.ingress.received_monotonic_ns
    )
    assert (
        binding_sample.publication_monotonic_ns
        > host.credit_received.ingress.received_monotonic_ns
    )

    wire_authorities = [
        payload["authority"]
        for event, payload in host.test_recorder.events
        if event == "visual_receiver_wire_authority"
    ]
    race_authorities = [
        payload["authority"]
        for event, payload in host.test_recorder.events
        if event == "race_active_wire_authority"
    ]
    first_gate1_visual_authority = next(
        authority
        for authority in wire_authorities
        if authority["call_start_monotonic_ns"]
        == first_gate1_wire["call_start_monotonic_ns"]
    )
    assert (
        first_gate1_visual_authority["frame_token"][
            "publication_sequence"
        ]
        >= binding_token.publication_sequence
    )
    assert (
        first_gate1_visual_authority["frame_token"][
            "publication_sequence"
        ]
        > (
            reacquisition.credited_advance.camera_token_at_credit
            .publication_sequence
        )
    )
    assert any(
        authority["expected_active_gate_index"] == 1
        for authority in race_authorities
    )
