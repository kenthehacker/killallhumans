"""Focused async coordinator for the bounded VQ2 visual-alignment stage.

The coordinator deliberately does not own simulator transport, reset/GO,
watchdogs, race authority, command dispatch, or cleanup.  Those remain on the
live runner and are reached only through the explicit host boundary below.
Image-space control and acceptance remain in the planning modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, Tuple

from competition.adapter import AttitudeRateCommand
from competition.vq2_visual_tracker import VisualTrack
from planning.vq2_visual_alignment import (
    RestrictedAlignmentMonitor,
    VisualAlignmentRefusal,
    VisualAlignmentTrend,
)
from planning.vq2_visual_servo import (
    ImageVisualServo,
    VisualServoOutput,
    VisualServoRefusal,
    VisualTarget,
)


@dataclass(frozen=True, slots=True)
class VisualAlignmentStageLimits:
    """Immutable runner-owned values used by the restricted stage."""

    control_period_s: float
    required_pretransition_frames: int
    hard_duration_s: float
    post_credit_frame_timeout_s: float
    response_grace_s: float
    max_yaw_rate_rad_s: float
    max_command_rate_rad_s: float
    max_pitch_rad: float
    max_entry_attitude_delta_rad: float
    min_thrust: float
    max_thrust: float
    max_visual_controller_thrust: float


@dataclass(frozen=True, slots=True)
class VisualAlignmentStageRuntime:
    """Exact live-runtime operations injected by :class:`VQ2Runner`."""

    limits: VisualAlignmentStageLimits
    safety_abort_type: type[BaseException]
    cancelled_error_type: type[BaseException]
    monotonic: Callable[[], float]
    perf_counter_ns: Callable[[], int]
    sleep: Callable[[float], Awaitable[Any]]
    post_gate_observation_deadline: Callable[..., float]
    next_control_deadline: Callable[..., float]
    visual_alignment_yaw_rate: Callable[..., Tuple[float, float]]
    attitude_rate_command: Callable[..., AttitudeRateCommand]
    limit_command_rates: Callable[..., AttitudeRateCommand]
    validate_command: Callable[[AttitudeRateCommand], None]


class VisualAlignmentStageHost(Protocol):
    """Narrow view of live-runner authority used by this one stage."""

    _visual_tracking_enabled: bool
    _visual_transition: Any
    _gate0_transition_proof: Any
    _visual_alignment_summary: Optional[Dict[str, Any]]
    _last_flight_command_sent_s: Optional[float]
    _last_flight_command_started_ns: Optional[int]
    visual_gate_graph: Any
    visual_tracker: Any
    visual_config: Any
    adapter: Any
    estimate: Any
    recorder: Any

    async def _run_gate0(
        self,
        context: Any,
        *,
        capture_transition: bool = False,
    ) -> Dict[str, Any]: ...

    async def _wait_for_next_flight_command_slot(self) -> float: ...

    async def _send_flight_command(
        self,
        command: AttitudeRateCommand,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]: ...

    def _sample(self) -> None: ...

    def _watchdog(self, **kwargs: Any) -> None: ...

    def _assert_visual_alignment_race_boundary(self) -> Any: ...

    def _assert_visual_alignment_attitude(
        self,
        *,
        entry_roll_rad: float,
        entry_pitch_rad: float,
        phase: str,
    ) -> Dict[str, float]: ...

    def _require_visual_current_target(
        self,
        *,
        expected_gate_index: int,
        expected_track_id: str,
        now_s: Optional[float] = None,
    ) -> Tuple[VisualTrack, VisualTarget]: ...

    def _assert_visual_alignment_no_passage(
        self,
        track: VisualTrack,
        *,
        phase: str,
    ) -> Dict[str, Any]: ...

    def _visual_alignment_trend_summary(
        self,
        trend: Optional[VisualAlignmentTrend],
    ) -> Dict[str, Any]: ...

    def _outbound_receipt_primitive(self, receipt: Any) -> Dict[str, Any]: ...

    def _record_tick(
        self,
        stage: str,
        elapsed_s: float,
        command: Optional[AttitudeRateCommand],
    ) -> None: ...


async def run_visual_alignment_stage(
    host: VisualAlignmentStageHost,
    context: Any,
    *,
    runtime: VisualAlignmentStageRuntime,
) -> Dict[str, Any]:
    """Prove bounded Gate-1 image-error improvement without passage."""

    limits = runtime.limits
    abort_type = runtime.safety_abort_type
    if not host._visual_tracking_enabled:
        raise abort_type(
            "visual alignment tracker was not enabled before reset"
        )
    bound = host.visual_gate_graph.latest_snapshot
    if (
        bound is None
        or bound.current_track_id is None
        or bound.current_gate_index != 0
        or not bound.authority_usable
    ):
        raise abort_type(
            "visual alignment lacks a bound initial current gate"
        )
    initial_current_track_id = bound.current_track_id
    gate0 = await host._run_gate0(context, capture_transition=False)
    transition = host._visual_transition
    if (
        transition is None
        or transition.from_gate_index != 0
        or transition.to_gate_index != 1
        or transition.retired_track_id != initial_current_track_id
        or transition.promoted_track_id == initial_current_track_id
        or len(transition.pretransition_frame_tokens)
        < limits.required_pretransition_frames
        or transition.history_length_before_promotion
        != transition.history_length_after_promotion
    ):
        raise abort_type(
            "visual alignment lacks the proved pretracked 0->1 promotion"
        )
    promoted_track_id = transition.promoted_track_id
    proof = host._gate0_transition_proof
    race_credit_ns = transition.race_status.received_monotonic_ns
    if proof is None or race_credit_ns is None:
        raise abort_type(
            "visual alignment lacks exact Gate-0 transition timing"
        )

    summary: Dict[str, Any] = {
        "command_authority": "restricted_promoted_current_visual_track",
        "success": False,
        "outcome": "running",
        "reason": None,
        "authoritative_transition": [0, 1],
        "initial_current_track_id": initial_current_track_id,
        "promoted_current_track_id": promoted_track_id,
        "current_track_id": promoted_track_id,
        "next_track_ids": [],
        "ambiguity": False,
        "collision_outcome": "none",
        "abort_outcome": None,
        "cleanup_confirmed": False,
        "visual_navigation_command_count": 0,
        "post_credit_zero_command_count": 0,
        "fresh_control_frame_count": 0,
        "thrust_saturation_count": 0,
        "max_abs_yaw_excursion_rad": 0.0,
        "max_abs_measured_yaw_rate_rad_s": 0.0,
        "max_peak_body_rate_rad_s": 0.0,
        "min_command_yaw_rate_rad_s": None,
        "max_command_yaw_rate_rad_s": None,
        "min_command_thrust": None,
        "max_command_thrust": None,
        "latest_geometry": None,
        "gate0": gate0,
        **host._visual_alignment_trend_summary(None),
    }
    host._visual_alignment_summary = summary
    trend: Optional[VisualAlignmentTrend] = None
    latest_output: Optional[VisualServoOutput] = None
    latest_target: Optional[VisualTarget] = None
    latest_token: Any = None
    dispatch_attempt_s: Optional[float] = None

    def refresh_summary(
        *,
        outcome: str,
        reason: Optional[str],
        criteria_met: Optional[bool] = None,
    ) -> None:
        graph = host.visual_gate_graph.latest_snapshot
        summary.update(host._visual_alignment_trend_summary(trend))
        if criteria_met is not None:
            summary["alignment_criteria_met"] = bool(criteria_met)
        summary["next_track_ids"] = (
            [
                candidate.track_id
                for candidate in graph.next_candidates
            ]
            if graph is not None
            else []
        )
        summary["ambiguity"] = bool(
            graph is None
            or graph.next_selection_ambiguous
            or not graph.authority_usable
            or (
                graph.current_track is not None
                and graph.current_track.ambiguous
            )
        )
        summary["outcome"] = outcome
        summary["reason"] = reason
        summary["success"] = bool(
            outcome == "success"
            and summary.get("alignment_criteria_met")
        )
        host._visual_alignment_summary = dict(summary)

    async def reserve_terminal_cleanup_slot(
        *,
        uncertain_dispatch_s: Optional[float] = None,
    ) -> None:
        not_before = 0.0
        if host._last_flight_command_sent_s is not None:
            not_before = max(
                not_before,
                float(host._last_flight_command_sent_s)
                + limits.control_period_s,
            )
        if uncertain_dispatch_s is not None:
            not_before = max(
                not_before,
                float(uncertain_dispatch_s) + limits.control_period_s,
            )
        observed = runtime.monotonic()
        attempts = 0
        while observed < not_before:
            if attempts >= 8:
                raise abort_type(
                    "visual alignment terminal command slot was not reserved"
                )
            await runtime.sleep(not_before - observed)
            next_observed = runtime.monotonic()
            if next_observed < observed:
                raise abort_type(
                    "visual alignment terminal pacing clock regressed"
                )
            observed = next_observed
            attempts += 1

    try:
        post_credit_deadline_s = runtime.post_gate_observation_deadline(
            pass_confirmed_s=proof.pass_confirmed_monotonic_s,
            flight_started_s=proof.flight_started_monotonic_s,
            crossing_started_s=proof.crossing_started_monotonic_s,
            requested_duration_s=limits.post_credit_frame_timeout_s,
        )
        next_tick = max(
            proof.next_control_deadline_s,
            await host._wait_for_next_flight_command_slot(),
        )
        while True:
            now_s = runtime.monotonic()
            if now_s >= post_credit_deadline_s:
                raise abort_type(
                    "visual alignment promotion lacks one fresh "
                    "post-credit continuation frame"
                )
            previous_update = host.visual_tracker.latest_update
            host._sample()
            host._watchdog(
                require_target=False,
                allow_benign_pad_contact=False,
            )
            host._assert_visual_alignment_race_boundary()
            current = host.visual_tracker.track(promoted_track_id)
            current_publish_ns = current.history[-1].publication_monotonic_ns
            new_publication = bool(
                previous_update is None
                or host.visual_tracker.latest_update is None
                or host.visual_tracker.latest_update.token
                != previous_update.token
            )
            if (
                new_publication
                and (
                    not current.visible
                    or current.missed_frame_count != 0
                    or current.ambiguous
                )
            ):
                raise abort_type(
                    "visual alignment promoted identity was missed or "
                    "ambiguous on its first post-credit publication"
            )
            command = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
            dispatch_attempt_s = runtime.monotonic()
            try:
                await host._send_flight_command(command)
                dispatch_attempt_s = None
            except BaseException as exc:
                dispatch_attempt_s = runtime.monotonic()
                if isinstance(exc, runtime.cancelled_error_type):
                    raise
                raise abort_type(
                    "visual alignment post-credit zero dispatch "
                    "failed closed"
                ) from exc
            summary["post_credit_zero_command_count"] = int(
                summary["post_credit_zero_command_count"]
            ) + 1
            host._record_tick(
                "visual-align/post-credit-zero",
                now_s - proof.pass_confirmed_monotonic_s,
                command,
            )
            if (
                current_publish_ns is not None
                and int(current_publish_ns) > int(race_credit_ns)
            ):
                break
            next_tick = runtime.next_control_deadline(
                next_tick,
                runtime.monotonic(),
            )
            await runtime.sleep(
                max(
                    0.0,
                    min(next_tick, post_credit_deadline_s)
                    - runtime.monotonic(),
                )
            )

        entry_now_s = runtime.monotonic()
        entry_track, _entry_target = host._require_visual_current_target(
            expected_gate_index=1,
            expected_track_id=promoted_track_id,
            now_s=entry_now_s,
        )
        host._assert_visual_alignment_no_passage(
            entry_track,
            phase="entry",
        )
        if host.estimate is None:
            raise abort_type(
                "visual alignment lacks an entry attitude estimate"
            )
        entry_roll, entry_pitch, entry_yaw = (
            float(value)
            for value in host.estimate.orientation.to_euler()
        )
        entry_state = host._assert_visual_alignment_attitude(
            entry_roll_rad=entry_roll,
            entry_pitch_rad=entry_pitch,
            phase="entry",
        )
        summary["max_peak_body_rate_rad_s"] = float(
            entry_state["peak_body_rate_rad_s"]
        )
        if entry_pitch + limits.max_entry_attitude_delta_rad < 0.0:
            raise abort_type(
                "visual alignment entry pitch cannot retain a "
                "nonnegative braking target inside its envelope"
            )

        servo = ImageVisualServo(host.visual_config.servo)
        monitor = RestrictedAlignmentMonitor(
            track_id=promoted_track_id,
            required_improving_frames=(
                host.visual_config.lifecycle.required_improving_frames
            ),
        )
        segment_started_s = await host._wait_for_next_flight_command_slot()
        duration_s = min(
            float(
                host.visual_config.lifecycle.restricted_alignment_duration_s
            ),
            limits.hard_duration_s,
        )
        hard_deadline_s = segment_started_s + duration_s
        wire_anchor_ns = runtime.perf_counter_ns()
        wire_anchor_s = runtime.monotonic()
        last_wire_start_ns = wire_anchor_ns + math.floor(
            max(
                0.0,
                hard_deadline_s
                - wire_anchor_s
                - limits.control_period_s,
            )
            * 1_000_000_000
        )
        drain_receipts = getattr(
            host.adapter,
            "drain_outbound_receipts",
            None,
        )
        if not callable(drain_receipts):
            raise abort_type(
                "visual alignment requires exact outbound wire receipts"
            )
        prior_receipts = [
            host._outbound_receipt_primitive(value)
            for value in drain_receipts()
        ]
        host.recorder.emit(
            "visual_alignment_prior_outbound_receipts_drained",
            count=len(prior_receipts),
        )
        next_tick = segment_started_s

        async def finish_alignment_acceptance() -> Dict[str, Any]:
            nonlocal trend, latest_token
            await reserve_terminal_cleanup_slot()
            host._sample()
            host._watchdog(
                require_target=False,
                allow_benign_pad_contact=False,
                count_rate_sample=False,
            )
            host._assert_visual_alignment_race_boundary()
            terminal_now_s = runtime.monotonic()
            if terminal_now_s >= hard_deadline_s:
                raise abort_type(
                    "visual alignment hard window expired during "
                    "terminal acceptance"
                )
            terminal_state = host._assert_visual_alignment_attitude(
                entry_roll_rad=entry_roll,
                entry_pitch_rad=entry_pitch,
                phase="terminal acceptance",
            )
            summary["max_peak_body_rate_rad_s"] = max(
                float(summary["max_peak_body_rate_rad_s"]),
                float(terminal_state["peak_body_rate_rad_s"]),
            )
            _yaw_probe, terminal_excursion = (
                runtime.visual_alignment_yaw_rate(
                    requested_rate_rad_s=0.0,
                    measured_yaw_rad=terminal_state["yaw_rad"],
                    reference_yaw_rad=entry_yaw,
                    measured_yaw_rate_rad_s=(
                        terminal_state["yaw_rate_rad_s"]
                    ),
                    horizontal_error_norm=(
                        0.0
                        if latest_output is None
                        else latest_output.effective_horizontal_error
                    ),
                    horizontal_corridor_norm=(
                        host.visual_config.servo.horizontal_corridor
                    ),
                )
            )
            summary["max_abs_yaw_excursion_rad"] = max(
                float(summary["max_abs_yaw_excursion_rad"]),
                abs(terminal_excursion),
            )
            summary["max_abs_measured_yaw_rate_rad_s"] = max(
                float(summary["max_abs_measured_yaw_rate_rad_s"]),
                abs(float(terminal_state["yaw_rate_rad_s"])),
            )
            terminal_track, terminal_target = (
                host._require_visual_current_target(
                    expected_gate_index=1,
                    expected_track_id=promoted_track_id,
                    now_s=terminal_now_s,
                )
            )
            summary["latest_geometry"] = (
                host._assert_visual_alignment_no_passage(
                    terminal_track,
                    phase="terminal acceptance",
                )
            )
            if terminal_target.frame_token != latest_token:
                try:
                    terminal_output = servo.step(
                        terminal_target,
                        now_monotonic_s=terminal_now_s,
                        segment_elapsed_s=(
                            terminal_now_s - segment_started_s
                        ),
                        segment_yaw_excursion_rad=terminal_excursion,
                        requested_next_blend=0.0,
                        allow_advance=False,
                    )
                    trend = monitor.observe(
                        terminal_target,
                        response_evaluation_enabled=True,
                        corridor_frames=terminal_output.corridor_frames,
                    )
                except (
                    VisualServoRefusal,
                    VisualAlignmentRefusal,
                ) as exc:
                    raise abort_type(
                        "visual alignment terminal authority "
                        f"refused: {exc}"
                    ) from exc
                if (
                    terminal_output.advance_enabled
                    or terminal_output.next_gate_blend != 0.0
                    or terminal_output.target_roll_rad != 0.0
                    or terminal_output.target_pitch_rad < 0.0
                ):
                    raise abort_type(
                        "visual alignment terminal servo proposal "
                        "escaped its no-advance envelope"
                    )
                latest_token = terminal_target.frame_token
                summary["fresh_control_frame_count"] = int(
                    summary["fresh_control_frame_count"]
                ) + 1
                host.recorder.emit(
                    "visual_alignment_terminal_frame",
                    elapsed_s=terminal_now_s - segment_started_s,
                    target=asdict(terminal_target),
                    servo=asdict(terminal_output),
                    trend=asdict(trend),
                )
            if (
                trend is None
                or trend.abort_reason is not None
                or not trend.accepted
            ):
                raise abort_type(
                    "visual alignment improvement did not survive "
                    "terminal recheck"
                )
            refresh_summary(
                outcome="success",
                reason="restricted alignment criteria met",
                criteria_met=True,
            )
            host.recorder.emit(
                "visual_alignment_complete",
                **summary,
            )
            return dict(summary)

        while True:
            now_s = runtime.monotonic()
            if now_s >= hard_deadline_s:
                raise abort_type(
                    "visual alignment hard 0.90s window expired "
                    "without joint improvement"
                )
            host._sample()
            host._watchdog(
                require_target=False,
                allow_benign_pad_contact=False,
                enforce_benign_pad_budget=False,
                count_rate_sample=False,
            )
            host._assert_visual_alignment_race_boundary()
            state = host._assert_visual_alignment_attitude(
                entry_roll_rad=entry_roll,
                entry_pitch_rad=entry_pitch,
                phase="control",
            )
            summary["max_peak_body_rate_rad_s"] = max(
                float(summary["max_peak_body_rate_rad_s"]),
                float(state["peak_body_rate_rad_s"]),
            )
            track, target = host._require_visual_current_target(
                expected_gate_index=1,
                expected_track_id=promoted_track_id,
                now_s=now_s,
            )
            geometry = host._assert_visual_alignment_no_passage(
                track,
                phase="control",
            )
            summary["latest_geometry"] = geometry
            elapsed_s = now_s - segment_started_s
            yaw_probe, yaw_excursion = (
                runtime.visual_alignment_yaw_rate(
                    requested_rate_rad_s=0.0,
                    measured_yaw_rad=state["yaw_rad"],
                    reference_yaw_rad=entry_yaw,
                    measured_yaw_rate_rad_s=state["yaw_rate_rad_s"],
                    horizontal_error_norm=target.normalized_x,
                    horizontal_corridor_norm=(
                        host.visual_config.servo.horizontal_corridor
                    ),
                )
            )
            assert yaw_probe == 0.0
            summary["max_abs_yaw_excursion_rad"] = max(
                float(summary["max_abs_yaw_excursion_rad"]),
                abs(yaw_excursion),
            )
            summary["max_abs_measured_yaw_rate_rad_s"] = max(
                float(summary["max_abs_measured_yaw_rate_rad_s"]),
                abs(float(state["yaw_rate_rad_s"])),
            )

            if target.frame_token != latest_token:
                try:
                    latest_output = servo.step(
                        target,
                        now_monotonic_s=now_s,
                        segment_elapsed_s=elapsed_s,
                        segment_yaw_excursion_rad=yaw_excursion,
                        requested_next_blend=0.0,
                        allow_advance=False,
                    )
                    trend = monitor.observe(
                        target,
                        response_evaluation_enabled=(
                            elapsed_s >= limits.response_grace_s
                        ),
                        corridor_frames=latest_output.corridor_frames,
                    )
                except (
                    VisualServoRefusal,
                    VisualAlignmentRefusal,
                ) as exc:
                    raise abort_type(
                        f"visual alignment authority refused: {exc}"
                    ) from exc
                latest_target = target
                latest_token = target.frame_token
                summary["fresh_control_frame_count"] = int(
                    summary["fresh_control_frame_count"]
                ) + 1
                host.recorder.emit(
                    "visual_alignment_frame",
                    elapsed_s=elapsed_s,
                    target=asdict(target),
                    servo=asdict(latest_output),
                    trend=asdict(trend),
                )
                if trend.abort_reason is not None:
                    raise abort_type(
                        "visual alignment monitor aborted: "
                        f"{trend.abort_reason}"
                    )
                if trend.accepted:
                    return await finish_alignment_acceptance()
            if latest_output is None or latest_target is None:
                raise abort_type(
                    "visual alignment produced no exact fresh-frame command"
                )
            if (
                latest_output.advance_enabled
                or latest_output.next_gate_blend != 0.0
                or latest_output.target_roll_rad != 0.0
                or latest_output.target_pitch_rad < 0.0
                or abs(latest_output.yaw_rate_rad_s)
                > limits.max_yaw_rate_rad_s
                or not (
                    limits.min_thrust
                    <= latest_output.thrust
                    <= limits.max_visual_controller_thrust
                )
            ):
                raise abort_type(
                    "visual alignment servo proposal escaped its "
                    "no-advance envelope"
                )
            target_pitch_upper = min(
                limits.max_pitch_rad,
                entry_pitch + limits.max_entry_attitude_delta_rad,
            )
            target_pitch = max(
                0.0,
                min(
                    float(latest_output.target_pitch_rad),
                    target_pitch_upper,
                ),
            )
            command_thrust = min(
                limits.max_thrust,
                max(
                    limits.min_thrust,
                    float(latest_output.thrust),
                ),
            )
            if command_thrust != float(latest_output.thrust):
                summary["thrust_saturation_count"] = int(
                    summary["thrust_saturation_count"]
                ) + 1
            base_command = runtime.attitude_rate_command(
                host.estimate,
                target_roll_rad=0.0,
                target_pitch_rad=target_pitch,
                thrust=command_thrust,
            )
            limited = runtime.limit_command_rates(
                base_command,
                limits.max_command_rate_rad_s,
            )
            yaw_rate, yaw_excursion = (
                runtime.visual_alignment_yaw_rate(
                    requested_rate_rad_s=latest_output.yaw_rate_rad_s,
                    measured_yaw_rad=state["yaw_rad"],
                    reference_yaw_rad=entry_yaw,
                    measured_yaw_rate_rad_s=state["yaw_rate_rad_s"],
                    horizontal_error_norm=(
                        latest_output.effective_horizontal_error
                    ),
                    horizontal_corridor_norm=(
                        host.visual_config.servo.horizontal_corridor
                    ),
                )
            )
            command = AttitudeRateCommand(
                roll_rate=limited.roll_rate,
                pitch_rate=limited.pitch_rate,
                yaw_rate=yaw_rate,
                thrust=limited.thrust,
            )
            runtime.validate_command(command)
            if (
                abs(command.roll_rate) > limits.max_command_rate_rad_s
                or abs(command.pitch_rate) > limits.max_command_rate_rad_s
                or abs(command.yaw_rate) > limits.max_yaw_rate_rad_s
                or not (
                    limits.min_thrust
                    <= command.thrust
                    <= limits.max_thrust
                )
            ):
                raise abort_type(
                    "visual alignment command escaped its fixed envelope"
                )

            host._watchdog(
                require_target=False,
                allow_benign_pad_contact=False,
                enforce_benign_pad_budget=False,
                count_rate_sample=False,
            )
            send_now_s = runtime.monotonic()
            if hard_deadline_s - send_now_s <= limits.control_period_s:
                raise abort_type(
                    "visual alignment hard window expired before send"
                )
            host._assert_visual_alignment_race_boundary()
            send_state = host._assert_visual_alignment_attitude(
                entry_roll_rad=entry_roll,
                entry_pitch_rad=entry_pitch,
                phase="command send",
            )
            send_track, send_target = (
                host._require_visual_current_target(
                    expected_gate_index=1,
                    expected_track_id=promoted_track_id,
                    now_s=send_now_s,
                )
            )
            host._assert_visual_alignment_no_passage(
                send_track,
                phase="command send",
            )
            if send_target.frame_token != latest_target.frame_token:
                raise abort_type(
                    "visual alignment exact target changed before send"
                )
            yaw_rate, _send_excursion = (
                runtime.visual_alignment_yaw_rate(
                    requested_rate_rad_s=command.yaw_rate,
                    measured_yaw_rad=send_state["yaw_rad"],
                    reference_yaw_rad=entry_yaw,
                    measured_yaw_rate_rad_s=send_state["yaw_rate_rad_s"],
                    horizontal_error_norm=(
                        latest_output.effective_horizontal_error
                    ),
                    horizontal_corridor_norm=(
                        host.visual_config.servo.horizontal_corridor
                    ),
                )
            )
            command = replace(command, yaw_rate=yaw_rate)
            runtime.validate_command(command)
            wire_not_before_ns = (
                None
                if host._last_flight_command_started_ns is None
                else host._last_flight_command_started_ns
                + round(limits.control_period_s * 1_000_000_000)
            )
            dispatch_attempt_s = runtime.monotonic()
            try:
                await host._send_flight_command(
                    command,
                    require_wire_receipt=True,
                    wire_start_not_before_ns=wire_not_before_ns,
                    wire_start_deadline_ns=last_wire_start_ns,
                )
                dispatch_attempt_s = None
            except BaseException as exc:
                # The call may have reached the wire before failing.  Anchor
                # the uncertainty period at failure observation, not call
                # entry, so cleanup cannot issue a back-to-back zero command
                # after a blocking adapter failure.
                dispatch_attempt_s = runtime.monotonic()
                if isinstance(exc, runtime.cancelled_error_type):
                    raise
                raise abort_type(
                    "visual alignment command dispatch failed closed"
                ) from exc
            summary["visual_navigation_command_count"] = int(
                summary["visual_navigation_command_count"]
            ) + 1
            summary["min_command_yaw_rate_rad_s"] = (
                command.yaw_rate
                if summary["min_command_yaw_rate_rad_s"] is None
                else min(
                    float(summary["min_command_yaw_rate_rad_s"]),
                    command.yaw_rate,
                )
            )
            summary["max_command_yaw_rate_rad_s"] = (
                command.yaw_rate
                if summary["max_command_yaw_rate_rad_s"] is None
                else max(
                    float(summary["max_command_yaw_rate_rad_s"]),
                    command.yaw_rate,
                )
            )
            summary["min_command_thrust"] = (
                command.thrust
                if summary["min_command_thrust"] is None
                else min(
                    float(summary["min_command_thrust"]),
                    command.thrust,
                )
            )
            summary["max_command_thrust"] = (
                command.thrust
                if summary["max_command_thrust"] is None
                else max(
                    float(summary["max_command_thrust"]),
                    command.thrust,
                )
            )
            host._record_tick(
                "visual-align/restricted",
                send_now_s - segment_started_s,
                command,
            )
            refresh_summary(outcome="running", reason=None)

            next_tick = runtime.next_control_deadline(
                next_tick,
                runtime.monotonic(),
            )
            await runtime.sleep(
                max(
                    0.0,
                    min(next_tick, hard_deadline_s)
                    - runtime.monotonic(),
                )
            )
    except BaseException as exc:
        pacing_failure: Optional[BaseException] = None
        try:
            await reserve_terminal_cleanup_slot(
                uncertain_dispatch_s=dispatch_attempt_s,
            )
        except BaseException as reserve_exc:
            pacing_failure = reserve_exc
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "visual alignment cleanup-slot reservation also "
                    f"failed: {reserve_exc}"
                )
        terminal_reason = str(exc) or type(exc).__name__
        if pacing_failure is not None:
            terminal_reason = (
                f"{terminal_reason}; cleanup-slot reservation failed: "
                f"{pacing_failure}"
            )
        if "collision reported" in terminal_reason:
            summary["collision_outcome"] = "collision"
        summary["abort_outcome"] = terminal_reason
        refresh_summary(
            outcome=(
                "abort"
                if isinstance(exc, abort_type)
                else (
                    "interrupted"
                    if isinstance(exc, runtime.cancelled_error_type)
                    else "unexpected_error"
                )
            ),
            reason=terminal_reason,
            criteria_met=False,
        )
        try:
            host.recorder.emit(
                "visual_alignment_terminal",
                **summary,
            )
        except BaseException as recorder_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "visual alignment terminal evidence emission also "
                    f"failed: {recorder_exc}"
                )
        raise
