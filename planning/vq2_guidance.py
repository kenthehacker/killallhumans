"""Pure, authority-neutral mapless guidance for the build-3385 VQ2 stack.

This module deliberately stops at a local planning decision.  It cannot build a
``CommandProposalV1``, approve a command, send transport data, advance a gate,
or declare passage.  The safety caller supplies both the frozen
``GateAuthorityEpochV1``, local phase/race state, and phase-entry time.  Visual
state can only make the supplied objective eligible or ineligible.

The state transition is an explicit pure function: immutable memory in,
immutable memory and decision out.  The local types are planning-owned values,
not frozen wire schemas, despite consuming exact frozen ``/1`` inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from competition.vq2_contracts import (
    FrameEdge,
    FrameIdentityV1,
    GateAuthorityEpochV1,
    RelativeGateStateV1,
    RelativeStateHealth,
    TrackRole,
)


_HARD_TARGET_BEARING_NORM = (0.0, 0.0)
_HARD_UNCERTAINTY_SIGMA_MULTIPLIER = 3.0
_HARD_ALIGN_CORRIDOR_HALF_WIDTH_NORM = (0.95, 0.85)
_HARD_APPROACH_CORRIDOR_HALF_WIDTH_NORM = (0.18, 0.18)
_HARD_COMMIT_CORRIDOR_HALF_WIDTH_NORM = (0.08, 0.08)
_HARD_ALIGN_RATE_LIMIT_NORM_S = (2.0, 2.0)
_HARD_APPROACH_RATE_LIMIT_NORM_S = (0.35, 0.35)
_HARD_COMMIT_RATE_LIMIT_NORM_S = (0.18, 0.18)
_HARD_COMMIT_MIN_LOG_SCALE = -0.7
_HARD_COMMIT_MIN_EXPANSION_RATE_S = 0.05
_HARD_MAX_STATE_DECISION_AGE_NS = 100_000_000
_HARD_MAX_MEASUREMENT_AGE_NS = 150_000_000
_HARD_MAX_PREDICTION_LEAD_NS = 100_000_000
_HARD_MAX_MEASUREMENT_UNCERTAINTY_NS = 50_000_000
_NUMERICAL_MAX_UNCERTAINTY_SIGMA_MULTIPLIER = 1_000_000.0


class VQ2GuidancePhase(str, Enum):
    """Safety-selected phase; perception is never allowed to change it."""

    ACQUIRE = "acquire"
    ALIGN = "align"
    APPROACH = "approach"
    COMMIT = "commit"
    CONFIRMATION = "confirmation"
    POST_CREDIT_REACQUIRE = "post_credit_reacquire"


class VQ2GuidanceRaceState(str, Enum):
    """Caller-proved race state at the safety boundary."""

    NOT_UNDERWAY = "not_underway"
    UNDERWAY = "underway"
    FINISHED = "finished"
    ABORTED = "aborted"


class VQ2GuidanceObjectiveKind(str, Enum):
    """Controller-independent local objective classification."""

    HOLD = "hold"
    ACQUIRE_ACTIVE_GATE = "acquire_active_gate"
    RECENTER_ACTIVE_GATE = "recenter_active_gate"
    APPROACH_ACTIVE_GATE = "approach_active_gate"
    COMMIT_ACTIVE_GATE = "commit_active_gate"
    CONFIRM_GATE_CREDIT = "confirm_gate_credit"
    REACQUIRE_AFTER_CREDIT = "reacquire_after_credit"


class VQ2GuidanceWithholdingReason(str, Enum):
    """Closed-state reason paired with ``objective_permitted=False``."""

    SAFETY_SESSION_CHANGED = "safety_session_changed"
    SAFETY_AUTHORITY_REGRESSED = "safety_authority_regressed"
    SAFETY_AUTHORITY_DISCONTINUITY = "safety_authority_discontinuity"
    SAFETY_EVALUATION_TIME_REGRESSED = "safety_evaluation_time_regressed"
    SAFETY_PHASE_START_REJECTED = "safety_phase_start_rejected"
    SAFETY_PHASE_TRANSITION_REJECTED = "safety_phase_transition_rejected"
    SAFETY_RACE_STATE_TRANSITION_REJECTED = (
        "safety_race_state_transition_rejected"
    )
    RACE_NOT_UNDERWAY = "race_not_underway"
    RACE_TERMINAL = "race_terminal"
    PHASE_HOLDS_MOTION = "phase_holds_motion"
    AWAITING_GATE_CREDIT = "awaiting_gate_credit"
    POST_CREDIT_REACQUIRE_HOLD = "post_credit_reacquire_hold"
    ACTIVE_STATE_REQUIRED = "active_state_required"
    ACTIVE_ROLE_REQUIRED = "active_role_required"
    ACTIVE_AUTHORITY_MISMATCH = "active_authority_mismatch"
    ACTIVE_TRACK_CHANGED = "active_track_changed"
    RETIRED_ACTIVE_TRACK_REUSED = "retired_active_track_reused"
    ACTIVE_STATE_STALE = "active_state_stale"
    ACTIVE_TIMING_FUTURE = "active_timing_future"
    ACTIVE_TIMING_STALE = "active_timing_stale"
    ACTIVE_TIMING_UNCERTAIN = "active_timing_uncertain"
    ACTIVE_PREDICTION_HORIZON = "active_prediction_horizon"
    SHADOW_INPUT_INVALID = "shadow_input_invalid"
    ACTIVE_STATE_HEALTH = "active_state_health"
    ACTIVE_STATE_DROPOUT = "active_state_dropout"
    ACTIVE_INNOVATION_REJECTED = "active_innovation_rejected"
    OUTSIDE_UNCERTAINTY_CORRIDOR = "outside_uncertainty_corridor"
    BEARING_RATE_UNCERTAIN = "bearing_rate_uncertain"
    COMMIT_SCALE_UNCERTAIN = "commit_scale_uncertain"
    COMMIT_EXPANSION_UNCERTAIN = "commit_expansion_uncertain"
    COMMIT_REQUIRES_UNCLIPPED = "commit_requires_unclipped"


@dataclass(frozen=True, slots=True)
class VQ2SafetyGuidanceInput:
    """Planning view of caller-supplied safety authority and race state.

    Constructing this value does not authenticate its producer.  Integration
    must create it only behind the existing trusted safety-supervisor seam.
    ``phase_started_monotonic_ns`` is on ``evaluation_host_clock_id``; the pure
    transition validates its phase-lifecycle consistency so a bad update can
    fail closed while retaining prior memory.
    """

    authority: GateAuthorityEpochV1
    phase: VQ2GuidancePhase
    race_state: VQ2GuidanceRaceState
    evaluation_host_clock_id: str
    evaluation_monotonic_ns: int
    phase_started_monotonic_ns: int

    def __post_init__(self) -> None:
        if type(self.authority) is not GateAuthorityEpochV1:
            raise TypeError("authority must be an exact GateAuthorityEpochV1")
        if type(self.phase) is not VQ2GuidancePhase:
            raise TypeError("phase must be an exact VQ2GuidancePhase")
        if type(self.race_state) is not VQ2GuidanceRaceState:
            raise TypeError("race_state must be an exact VQ2GuidanceRaceState")
        if (
            type(self.evaluation_host_clock_id) is not str
            or not self.evaluation_host_clock_id
        ):
            raise TypeError("evaluation_host_clock_id must be a non-empty string")
        if self.evaluation_host_clock_id != self.authority.camera_host_clock_id:
            raise ValueError("evaluation host clock must match the authority camera clock")
        _nonnegative_exact_int(
            self.evaluation_monotonic_ns,
            "evaluation_monotonic_ns",
        )
        _nonnegative_exact_int(
            self.phase_started_monotonic_ns,
            "phase_started_monotonic_ns",
        )
        if self.phase_started_monotonic_ns > self.evaluation_monotonic_ns:
            raise ValueError("phase start cannot postdate evaluation")
        if (
            self.evaluation_monotonic_ns
            < self.authority.frame_publish_monotonic_ns_not_before
        ):
            raise ValueError("evaluation time cannot predate the authority cutover")


@dataclass(frozen=True, slots=True)
class VQ2GuidanceConfig:
    """Conservative image-space eligibility thresholds.

    These are offline planning defaults, not calibrated flight limits.  A live
    stage would require separate replay, timing, supervisor, and powered review.
    """

    target_bearing_norm: tuple[float, float] = _HARD_TARGET_BEARING_NORM
    uncertainty_sigma_multiplier: float = _HARD_UNCERTAINTY_SIGMA_MULTIPLIER
    align_corridor_half_width_norm: tuple[float, float] = (
        _HARD_ALIGN_CORRIDOR_HALF_WIDTH_NORM
    )
    approach_corridor_half_width_norm: tuple[float, float] = (
        _HARD_APPROACH_CORRIDOR_HALF_WIDTH_NORM
    )
    commit_corridor_half_width_norm: tuple[float, float] = (
        _HARD_COMMIT_CORRIDOR_HALF_WIDTH_NORM
    )
    align_rate_limit_norm_s: tuple[float, float] = _HARD_ALIGN_RATE_LIMIT_NORM_S
    approach_rate_limit_norm_s: tuple[float, float] = (
        _HARD_APPROACH_RATE_LIMIT_NORM_S
    )
    commit_rate_limit_norm_s: tuple[float, float] = _HARD_COMMIT_RATE_LIMIT_NORM_S
    commit_min_log_scale: float = _HARD_COMMIT_MIN_LOG_SCALE
    commit_min_expansion_rate_s: float = _HARD_COMMIT_MIN_EXPANSION_RATE_S
    max_state_decision_age_ns: int = _HARD_MAX_STATE_DECISION_AGE_NS
    max_measurement_age_ns: int = _HARD_MAX_MEASUREMENT_AGE_NS
    max_prediction_lead_ns: int = _HARD_MAX_PREDICTION_LEAD_NS
    max_measurement_uncertainty_ns: int = _HARD_MAX_MEASUREMENT_UNCERTAINTY_NS

    def __post_init__(self) -> None:
        target = _finite_pair(
            self.target_bearing_norm,
            "target_bearing_norm",
            minimum=-1.0,
            maximum=1.0,
        )
        if target != _HARD_TARGET_BEARING_NORM:
            raise ValueError("target_bearing_norm is fixed at (0.0, 0.0) for this MVP")
        sigma_multiplier = _positive_finite(
            self.uncertainty_sigma_multiplier,
            "uncertainty_sigma_multiplier",
        )
        if sigma_multiplier < _HARD_UNCERTAINTY_SIGMA_MULTIPLIER:
            raise ValueError("uncertainty_sigma_multiplier is tightening-only")
        if sigma_multiplier > _NUMERICAL_MAX_UNCERTAINTY_SIGMA_MULTIPLIER:
            raise ValueError(
                "uncertainty_sigma_multiplier exceeds the finite numerical ceiling"
            )
        corridor_pairs: dict[str, tuple[float, float]] = {}
        for name in (
            "align_corridor_half_width_norm",
            "approach_corridor_half_width_norm",
            "commit_corridor_half_width_norm",
        ):
            corridor_pairs[name] = _finite_pair(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=4.0,
                strictly_positive=True,
            )
        for name, hard_limit in (
            (
                "align_corridor_half_width_norm",
                _HARD_ALIGN_CORRIDOR_HALF_WIDTH_NORM,
            ),
            (
                "approach_corridor_half_width_norm",
                _HARD_APPROACH_CORRIDOR_HALF_WIDTH_NORM,
            ),
            (
                "commit_corridor_half_width_norm",
                _HARD_COMMIT_CORRIDOR_HALF_WIDTH_NORM,
            ),
        ):
            if any(
                value > maximum
                for value, maximum in zip(corridor_pairs[name], hard_limit)
            ):
                raise ValueError(f"{name} is tightening-only")
        if not _pair_ordered(
            corridor_pairs["commit_corridor_half_width_norm"],
            corridor_pairs["approach_corridor_half_width_norm"],
            corridor_pairs["align_corridor_half_width_norm"],
        ):
            raise ValueError("corridors must preserve commit <= approach <= align")
        rate_pairs: dict[str, tuple[float, float]] = {}
        for name in (
            "align_rate_limit_norm_s",
            "approach_rate_limit_norm_s",
            "commit_rate_limit_norm_s",
        ):
            rate_pairs[name] = _finite_pair(
                getattr(self, name),
                name,
                minimum=0.0,
                strictly_positive=True,
            )
        for name, hard_limit in (
            ("align_rate_limit_norm_s", _HARD_ALIGN_RATE_LIMIT_NORM_S),
            ("approach_rate_limit_norm_s", _HARD_APPROACH_RATE_LIMIT_NORM_S),
            ("commit_rate_limit_norm_s", _HARD_COMMIT_RATE_LIMIT_NORM_S),
        ):
            if any(
                value > maximum for value, maximum in zip(rate_pairs[name], hard_limit)
            ):
                raise ValueError(f"{name} is tightening-only")
        if not _pair_ordered(
            rate_pairs["commit_rate_limit_norm_s"],
            rate_pairs["approach_rate_limit_norm_s"],
            rate_pairs["align_rate_limit_norm_s"],
        ):
            raise ValueError("rate limits must preserve commit <= approach <= align")
        commit_min_log_scale = _finite(
            self.commit_min_log_scale,
            "commit_min_log_scale",
        )
        if commit_min_log_scale < _HARD_COMMIT_MIN_LOG_SCALE:
            raise ValueError("commit_min_log_scale is tightening-only")
        commit_min_expansion = _finite(
            self.commit_min_expansion_rate_s,
            "commit_min_expansion_rate_s",
            minimum=0.0,
        )
        if commit_min_expansion < _HARD_COMMIT_MIN_EXPANSION_RATE_S:
            raise ValueError("commit_min_expansion_rate_s is tightening-only")
        for name, hard_limit in (
            ("max_state_decision_age_ns", _HARD_MAX_STATE_DECISION_AGE_NS),
            ("max_measurement_age_ns", _HARD_MAX_MEASUREMENT_AGE_NS),
            ("max_prediction_lead_ns", _HARD_MAX_PREDICTION_LEAD_NS),
            (
                "max_measurement_uncertainty_ns",
                _HARD_MAX_MEASUREMENT_UNCERTAINTY_NS,
            ),
        ):
            value = _positive_exact_int(getattr(self, name), name)
            if value > hard_limit:
                raise ValueError(f"{name} is tightening-only")


@dataclass(frozen=True, slots=True)
class VQ2GuidanceSource:
    """Exact active-state correlation copied without reinterpretation."""

    host_clock_id: str
    decision_time_monotonic_ns: int
    prediction_time_monotonic_ns: int
    source_frame: FrameIdentityV1
    source_frame_publication_sequence: int
    source_frame_publish_monotonic_ns: int
    tracker_id: str
    track_role: TrackRole
    state_sequence: int
    measurement_update_sequence: int
    source_candidate_id: str

    def __post_init__(self) -> None:
        for name in ("host_clock_id", "tracker_id", "source_candidate_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise TypeError(f"{name} must be a non-empty exact string")
        for name in (
            "decision_time_monotonic_ns",
            "prediction_time_monotonic_ns",
            "source_frame_publication_sequence",
            "source_frame_publish_monotonic_ns",
            "state_sequence",
            "measurement_update_sequence",
        ):
            _nonnegative_exact_int(getattr(self, name), name)
        if self.prediction_time_monotonic_ns < self.decision_time_monotonic_ns:
            raise ValueError("source prediction time cannot predate decision time")
        if type(self.source_frame) is not FrameIdentityV1:
            raise TypeError("source_frame must be an exact FrameIdentityV1")
        if self.source_frame_publish_monotonic_ns > self.decision_time_monotonic_ns:
            raise ValueError("source frame publication cannot postdate decision time")
        if type(self.track_role) is not TrackRole:
            raise TypeError("track_role must be an exact TrackRole")
        if self.track_role is not TrackRole.ACTIVE:
            raise ValueError("guidance source correlation must be active")


_MeasurementUse = tuple[FrameIdentityV1, str, int]


@dataclass(frozen=True, slots=True)
class VQ2GuidanceMemory:
    """Minimal immutable history needed to reject stale or transferred state."""

    safety: VQ2SafetyGuidanceInput
    active_source: Optional[VQ2GuidanceSource]
    seen_active_measurements: tuple[_MeasurementUse, ...]
    retired_active_tracker_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.safety) is not VQ2SafetyGuidanceInput:
            raise TypeError("safety must be an exact VQ2SafetyGuidanceInput")
        if (
            self.safety.phase_started_monotonic_ns
            > self.safety.evaluation_monotonic_ns
        ):
            raise ValueError("accepted phase start cannot postdate evaluation")
        if self.active_source is not None and type(self.active_source) is not VQ2GuidanceSource:
            raise TypeError("active_source must be VQ2GuidanceSource or None")
        if type(self.seen_active_measurements) is not tuple:
            raise TypeError("seen_active_measurements must be an exact tuple")
        seen_keys: set[tuple[FrameIdentityV1, str]] = set()
        for item in self.seen_active_measurements:
            if type(item) is not tuple or len(item) != 3:
                raise TypeError("measurement history entries must be exact triples")
            frame, candidate_id, update_sequence = item
            if type(frame) is not FrameIdentityV1:
                raise TypeError("measurement history frame must be FrameIdentityV1")
            if type(candidate_id) is not str or not candidate_id:
                raise TypeError("measurement history candidate must be a string")
            _nonnegative_exact_int(update_sequence, "measurement update sequence")
            key = (frame, candidate_id)
            if key in seen_keys:
                raise ValueError("measurement history cannot repeat a source")
            seen_keys.add(key)
        if self.active_source is None:
            if self.seen_active_measurements:
                raise ValueError("measurement history requires an active source")
        else:
            if not self.seen_active_measurements:
                raise ValueError("active source requires measurement history")
            last_frame, last_candidate, last_update = self.seen_active_measurements[-1]
            if (
                last_frame != self.active_source.source_frame
                or last_candidate != self.active_source.source_candidate_id
                or last_update != self.active_source.measurement_update_sequence
            ):
                raise ValueError("active source must match the last measurement history entry")
        if type(self.retired_active_tracker_ids) is not tuple:
            raise TypeError("retired_active_tracker_ids must be an exact tuple")
        if any(type(item) is not str or not item for item in self.retired_active_tracker_ids):
            raise TypeError("retired tracker IDs must be non-empty strings")
        if len(set(self.retired_active_tracker_ids)) != len(
            self.retired_active_tracker_ids
        ):
            raise ValueError("retired tracker IDs must be unique")
        if (
            self.active_source is not None
            and self.active_source.tracker_id in self.retired_active_tracker_ids
        ):
            raise ValueError("the active tracker cannot also be retired")


@dataclass(frozen=True, slots=True)
class VQ2GuidanceDecision:
    """Local result echoing stable phase time, with no command authority."""

    authority: GateAuthorityEpochV1
    phase: VQ2GuidancePhase
    race_state: VQ2GuidanceRaceState
    evaluation_host_clock_id: str
    evaluation_monotonic_ns: int
    phase_started_monotonic_ns: int
    objective_kind: VQ2GuidanceObjectiveKind
    target_bearing_norm: tuple[float, float]
    objective_permitted: bool
    withholding_reason: Optional[VQ2GuidanceWithholdingReason]
    source: Optional[VQ2GuidanceSource]
    conservative_bearing_error_norm: Optional[tuple[float, float]]
    corridor_half_width_norm: tuple[float, float]
    corridor_margin_norm: Optional[tuple[float, float]]
    corridor_eligible: bool
    bearing_rate_eligible: bool
    scale_eligible: bool
    expansion_eligible: bool
    shadow_track_count: int

    def __post_init__(self) -> None:
        if type(self.authority) is not GateAuthorityEpochV1:
            raise TypeError("authority must be an exact GateAuthorityEpochV1")
        if type(self.phase) is not VQ2GuidancePhase:
            raise TypeError("phase must be an exact VQ2GuidancePhase")
        if type(self.race_state) is not VQ2GuidanceRaceState:
            raise TypeError("race_state must be an exact VQ2GuidanceRaceState")
        if (
            type(self.evaluation_host_clock_id) is not str
            or not self.evaluation_host_clock_id
        ):
            raise TypeError("evaluation_host_clock_id must be a non-empty string")
        if self.evaluation_host_clock_id != self.authority.camera_host_clock_id:
            raise ValueError("decision evaluation clock does not match authority")
        _nonnegative_exact_int(
            self.evaluation_monotonic_ns,
            "evaluation_monotonic_ns",
        )
        _nonnegative_exact_int(
            self.phase_started_monotonic_ns,
            "phase_started_monotonic_ns",
        )
        if self.phase_started_monotonic_ns > self.evaluation_monotonic_ns:
            raise ValueError("phase start cannot postdate decision evaluation")
        if type(self.objective_kind) is not VQ2GuidanceObjectiveKind:
            raise TypeError("objective_kind must be an exact VQ2GuidanceObjectiveKind")
        _finite_pair(
            self.target_bearing_norm,
            "target_bearing_norm",
            minimum=-1.0,
            maximum=1.0,
        )
        if type(self.objective_permitted) is not bool:
            raise TypeError("objective_permitted must be an exact bool")
        if self.objective_permitted != (self.withholding_reason is None):
            raise ValueError("objective permission and withholding reason must be paired")
        if self.withholding_reason is not None and type(
            self.withholding_reason
        ) is not VQ2GuidanceWithholdingReason:
            raise TypeError("withholding_reason has the wrong type")
        if self.source is not None and type(self.source) is not VQ2GuidanceSource:
            raise TypeError("source must be VQ2GuidanceSource or None")
        if self.source is not None:
            if self.source.host_clock_id != self.authority.camera_host_clock_id:
                raise ValueError("guidance source host clock does not match authority")
            if self.source.source_frame.stream_id != self.authority.camera_stream_id:
                raise ValueError("guidance source stream does not match authority")
            if self.source.source_frame.generation != self.authority.camera_generation:
                raise ValueError("guidance source generation does not match authority")
            if (
                self.source.source_frame_publication_sequence
                < self.authority.frame_publication_sequence_not_before
                or self.source.source_frame_publish_monotonic_ns
                < self.authority.frame_publish_monotonic_ns_not_before
            ):
                raise ValueError("guidance source predates the authority cutover")
        if self.conservative_bearing_error_norm is not None:
            _finite_pair(
                self.conservative_bearing_error_norm,
                "conservative_bearing_error_norm",
                minimum=0.0,
            )
        _finite_pair(
            self.corridor_half_width_norm,
            "corridor_half_width_norm",
            minimum=0.0,
            strictly_positive=True,
        )
        if self.corridor_margin_norm is not None:
            _finite_pair(self.corridor_margin_norm, "corridor_margin_norm")
        for name in (
            "corridor_eligible",
            "bearing_rate_eligible",
            "scale_eligible",
            "expansion_eligible",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        _nonnegative_exact_int(self.shadow_track_count, "shadow_track_count")
        actionable = {
            VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE,
            VQ2GuidanceObjectiveKind.APPROACH_ACTIVE_GATE,
            VQ2GuidanceObjectiveKind.COMMIT_ACTIVE_GATE,
        }
        if self.objective_permitted:
            if self.objective_kind not in actionable or self.source is None:
                raise ValueError("a permitted objective must be actionable and sourced")
            if not self.corridor_eligible or not self.bearing_rate_eligible:
                raise ValueError("a permitted objective must satisfy corridor and rate gates")
            if self.objective_kind is VQ2GuidanceObjectiveKind.COMMIT_ACTIVE_GATE and (
                not self.scale_eligible or not self.expansion_eligible
            ):
                raise ValueError("a permitted commit must satisfy scale and expansion gates")


@dataclass(frozen=True, slots=True)
class VQ2GuidanceTransition:
    memory: VQ2GuidanceMemory
    decision: VQ2GuidanceDecision

    def __post_init__(self) -> None:
        if type(self.memory) is not VQ2GuidanceMemory:
            raise TypeError("memory must be an exact VQ2GuidanceMemory")
        if type(self.decision) is not VQ2GuidanceDecision:
            raise TypeError("decision must be an exact VQ2GuidanceDecision")


def step_vq2_guidance(
    memory: Optional[VQ2GuidanceMemory],
    safety: VQ2SafetyGuidanceInput,
    *,
    active_state: Optional[RelativeGateStateV1],
    shadow_states: tuple[RelativeGateStateV1, ...] = (),
    config: Optional[VQ2GuidanceConfig] = None,
) -> VQ2GuidanceTransition:
    """Apply one deterministic, authority-gated mapless guidance transition.

    Semantic trust failures return a withheld decision.  Exact Python type
    violations raise immediately because they are programmer/interface errors.
    A rejected safety transition, including phase-start rewind or renewal,
    preserves prior memory; rejected visual input cannot poison accepted
    active-track history.
    """

    if memory is not None and type(memory) is not VQ2GuidanceMemory:
        raise TypeError("memory must be VQ2GuidanceMemory or None")
    if type(safety) is not VQ2SafetyGuidanceInput:
        raise TypeError("safety must be an exact VQ2SafetyGuidanceInput")
    if active_state is not None and type(active_state) is not RelativeGateStateV1:
        raise TypeError("active_state must be RelativeGateStateV1 or None")
    if type(shadow_states) is not tuple:
        raise TypeError("shadow_states must be an exact tuple")
    if any(type(state) is not RelativeGateStateV1 for state in shadow_states):
        raise TypeError("shadow_states must contain exact RelativeGateStateV1 values")
    if config is None:
        config = DEFAULT_VQ2_GUIDANCE_CONFIG
    elif type(config) is not VQ2GuidanceConfig:
        raise TypeError("config must be an exact VQ2GuidanceConfig")

    if memory is None:
        if (
            safety.phase is not VQ2GuidancePhase.ACQUIRE
            or safety.race_state is not VQ2GuidanceRaceState.NOT_UNDERWAY
        ):
            raise ValueError(
                "initial guidance input must be ACQUIRE with race NOT_UNDERWAY"
            )
        if safety.phase_started_monotonic_ns != safety.evaluation_monotonic_ns:
            raise ValueError(
                "initial guidance phase start must equal evaluation time"
            )
        base_memory = VQ2GuidanceMemory(
            safety=safety,
            active_source=None,
            seen_active_measurements=(),
            retired_active_tracker_ids=(),
        )
    else:
        transition_kind, rejection = _validate_safety_transition(
            memory.safety,
            safety,
        )
        if rejection is not None:
            return VQ2GuidanceTransition(
                memory=memory,
                decision=_withheld_decision(
                    memory.safety,
                    rejection,
                    config=config,
                    shadow_track_count=len(shadow_states),
                ),
            )
        base_memory = _memory_after_safety_transition(
            memory,
            safety,
            transition_kind=transition_kind,
        )

    shadow_rejection = _validate_shadow_isolation(
        shadow_states,
        safety,
        active_state=active_state,
        config=config,
    )
    if shadow_rejection is not None:
        return VQ2GuidanceTransition(
            memory=base_memory,
            decision=_withheld_decision(
                safety,
                shadow_rejection,
                config=config,
                shadow_track_count=len(shadow_states),
            ),
        )

    accepted_memory = base_memory
    source: Optional[VQ2GuidanceSource] = None
    active_rejection: Optional[VQ2GuidanceWithholdingReason] = None
    if active_state is not None:
        if active_state.track_role is not TrackRole.ACTIVE:
            active_rejection = VQ2GuidanceWithholdingReason.ACTIVE_ROLE_REQUIRED
        else:
            source = _source_from_state(active_state)
            active_rejection = _validate_active_state(
                base_memory,
                safety,
                active_state,
                source,
                config=config,
            )
            if active_rejection is None:
                accepted_memory = _memory_with_active_state(base_memory, source)
            else:
                source = None

    decision = _evaluate_decision(
        safety,
        active_state if active_rejection is None else None,
        source,
        active_rejection=active_rejection,
        shadow_track_count=len(shadow_states),
        config=config,
    )
    return VQ2GuidanceTransition(memory=accepted_memory, decision=decision)


def _validate_safety_transition(
    previous: VQ2SafetyGuidanceInput,
    current: VQ2SafetyGuidanceInput,
) -> tuple[str, Optional[VQ2GuidanceWithholdingReason]]:
    previous_authority = previous.authority
    current_authority = current.authority

    if current.evaluation_monotonic_ns < previous.evaluation_monotonic_ns:
        return (
            "invalid",
            VQ2GuidanceWithholdingReason.SAFETY_EVALUATION_TIME_REGRESSED,
        )
    if current_authority.session_id != previous_authority.session_id:
        return "invalid", VQ2GuidanceWithholdingReason.SAFETY_SESSION_CHANGED
    if (
        current_authority.camera_host_clock_id
        != previous_authority.camera_host_clock_id
        or current_authority.camera_stream_id != previous_authority.camera_stream_id
    ):
        return "invalid", VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_DISCONTINUITY
    if current_authority.reset_epoch < previous_authority.reset_epoch:
        return "invalid", VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_REGRESSED

    if current_authority.reset_epoch > previous_authority.reset_epoch:
        if (
            current_authority.reset_epoch != previous_authority.reset_epoch + 1
            or current_authority.gate_epoch != 0
            or current_authority.expected_gate_index != 0
            or current_authority.camera_generation
            <= previous_authority.camera_generation
            or current_authority.race_status_sequence
            <= previous_authority.race_status_sequence
            or current_authority.frame_publication_sequence_not_before
            <= previous_authority.frame_publication_sequence_not_before
            or current_authority.frame_publish_monotonic_ns_not_before
            <= previous_authority.frame_publish_monotonic_ns_not_before
            or current.evaluation_monotonic_ns
            <= previous.evaluation_monotonic_ns
        ):
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_DISCONTINUITY,
            )
        if current.phase is not VQ2GuidancePhase.ACQUIRE:
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED,
            )
        if current.race_state is not VQ2GuidanceRaceState.NOT_UNDERWAY:
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_RACE_STATE_TRANSITION_REJECTED,
            )
        if current.phase_started_monotonic_ns != current.evaluation_monotonic_ns:
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED,
            )
        return "reset", None

    if current_authority.camera_generation != previous_authority.camera_generation:
        return "invalid", VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_DISCONTINUITY
    if current_authority.gate_epoch < previous_authority.gate_epoch:
        return "invalid", VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_REGRESSED

    if current_authority.gate_epoch > previous_authority.gate_epoch:
        if (
            current_authority.gate_epoch != previous_authority.gate_epoch + 1
            or current_authority.expected_gate_index
            != previous_authority.expected_gate_index + 1
            or not _strict_forward_snapshot(previous_authority, current_authority)
            or current.evaluation_monotonic_ns
            <= previous.evaluation_monotonic_ns
        ):
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_DISCONTINUITY,
            )
        if (
            previous.phase is not VQ2GuidancePhase.CONFIRMATION
            or current.phase is not VQ2GuidancePhase.POST_CREDIT_REACQUIRE
        ):
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED,
            )
        if (
            previous.race_state is not VQ2GuidanceRaceState.UNDERWAY
            or current.race_state is not VQ2GuidanceRaceState.UNDERWAY
        ):
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_RACE_STATE_TRANSITION_REJECTED,
            )
        if current.phase_started_monotonic_ns != current.evaluation_monotonic_ns:
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED,
            )
        return "gate", None

    if (
        current_authority.expected_gate_index
        != previous_authority.expected_gate_index
    ):
        return "invalid", VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_DISCONTINUITY
    if not _nonregressing_snapshot(previous_authority, current_authority):
        return "invalid", VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_REGRESSED

    phase_changed = current.phase is not previous.phase
    race_changed = current.race_state is not previous.race_state
    if (phase_changed or race_changed) and not _strict_forward_snapshot(
        previous_authority,
        current_authority,
    ):
        reason = (
            VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED
            if phase_changed
            else VQ2GuidanceWithholdingReason.SAFETY_RACE_STATE_TRANSITION_REJECTED
        )
        return "invalid", reason
    if (
        (phase_changed or race_changed)
        and current.evaluation_monotonic_ns <= previous.evaluation_monotonic_ns
    ):
        reason = (
            VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED
            if phase_changed
            else VQ2GuidanceWithholdingReason.SAFETY_RACE_STATE_TRANSITION_REJECTED
        )
        return "invalid", reason
    if phase_changed and current.phase not in _NEXT_PHASES[previous.phase]:
        return (
            "invalid",
            VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED,
        )
    if race_changed and current.race_state not in _NEXT_RACE_STATES[
        previous.race_state
    ]:
        return (
            "invalid",
            VQ2GuidanceWithholdingReason.SAFETY_RACE_STATE_TRANSITION_REJECTED,
        )
    if phase_changed:
        if current.phase_started_monotonic_ns != current.evaluation_monotonic_ns:
            return (
                "invalid",
                VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED,
            )
    elif current.phase_started_monotonic_ns != previous.phase_started_monotonic_ns:
        return (
            "invalid",
            VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED,
        )
    return "same", None


def _memory_after_safety_transition(
    memory: VQ2GuidanceMemory,
    safety: VQ2SafetyGuidanceInput,
    *,
    transition_kind: str,
) -> VQ2GuidanceMemory:
    if transition_kind == "reset":
        return VQ2GuidanceMemory(
            safety=safety,
            active_source=None,
            seen_active_measurements=(),
            retired_active_tracker_ids=(),
        )
    if transition_kind == "gate":
        retired = memory.retired_active_tracker_ids
        if (
            memory.active_source is not None
            and memory.active_source.tracker_id not in retired
        ):
            retired = (*retired, memory.active_source.tracker_id)
        return VQ2GuidanceMemory(
            safety=safety,
            active_source=None,
            seen_active_measurements=(),
            retired_active_tracker_ids=retired,
        )
    return VQ2GuidanceMemory(
        safety=safety,
        active_source=memory.active_source,
        seen_active_measurements=memory.seen_active_measurements,
        retired_active_tracker_ids=memory.retired_active_tracker_ids,
    )


def _validate_shadow_isolation(
    shadow_states: tuple[RelativeGateStateV1, ...],
    safety: VQ2SafetyGuidanceInput,
    *,
    active_state: Optional[RelativeGateStateV1],
    config: VQ2GuidanceConfig,
) -> Optional[VQ2GuidanceWithholdingReason]:
    tracker_ids: set[str] = set()
    sources: set[tuple[FrameIdentityV1, str]] = set()
    if active_state is not None:
        tracker_ids.add(active_state.tracker_id)
        source_frame = active_state.timing.source_frame
        if source_frame is not None:
            sources.add((source_frame, active_state.source_candidate_id))
    for state in shadow_states:
        if (
            state.track_role is not TrackRole.SHADOW
            or state.authority != safety.authority
            or state.tracker_id in tracker_ids
            or state.timing.source_frame is None
            or _validate_state_timing(state, safety, config=config) is not None
        ):
            return VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
        source = (state.timing.source_frame, state.source_candidate_id)
        if source in sources:
            return VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
        tracker_ids.add(state.tracker_id)
        sources.add(source)
    return None


def _validate_active_state(
    memory: VQ2GuidanceMemory,
    safety: VQ2SafetyGuidanceInput,
    state: RelativeGateStateV1,
    source: VQ2GuidanceSource,
    *,
    config: VQ2GuidanceConfig,
) -> Optional[VQ2GuidanceWithholdingReason]:
    if state.track_role is not TrackRole.ACTIVE:
        return VQ2GuidanceWithholdingReason.ACTIVE_ROLE_REQUIRED
    if state.authority != safety.authority:
        return VQ2GuidanceWithholdingReason.ACTIVE_AUTHORITY_MISMATCH
    timing_rejection = _validate_state_timing(state, safety, config=config)
    if timing_rejection is not None:
        return timing_rejection
    if source.tracker_id in memory.retired_active_tracker_ids:
        return VQ2GuidanceWithholdingReason.RETIRED_ACTIVE_TRACK_REUSED
    previous = memory.active_source
    if previous is None:
        return None
    if source.tracker_id != previous.tracker_id:
        return VQ2GuidanceWithholdingReason.ACTIVE_TRACK_CHANGED
    if (
        source.state_sequence <= previous.state_sequence
        or source.decision_time_monotonic_ns < previous.decision_time_monotonic_ns
        or source.prediction_time_monotonic_ns < previous.prediction_time_monotonic_ns
        or source.source_frame_publication_sequence
        < previous.source_frame_publication_sequence
        or source.source_frame_publish_monotonic_ns
        < previous.source_frame_publish_monotonic_ns
    ):
        return VQ2GuidanceWithholdingReason.ACTIVE_STATE_STALE

    source_key = (source.source_frame, source.source_candidate_id)
    previous_key = (previous.source_frame, previous.source_candidate_id)
    history = {
        (frame, candidate): update
        for frame, candidate, update in memory.seen_active_measurements
    }
    if source_key == previous_key:
        if (
            source.measurement_update_sequence
            != previous.measurement_update_sequence
            or source.source_frame_publication_sequence
            != previous.source_frame_publication_sequence
            or source.source_frame_publish_monotonic_ns
            != previous.source_frame_publish_monotonic_ns
        ):
            return VQ2GuidanceWithholdingReason.ACTIVE_STATE_STALE
    else:
        if (
            source_key in history
            or source.measurement_update_sequence
            <= previous.measurement_update_sequence
            or source.source_frame_publication_sequence
            <= previous.source_frame_publication_sequence
            or source.source_frame_publish_monotonic_ns
            <= previous.source_frame_publish_monotonic_ns
        ):
            return VQ2GuidanceWithholdingReason.ACTIVE_STATE_STALE
    return None


def _validate_state_timing(
    state: RelativeGateStateV1,
    safety: VQ2SafetyGuidanceInput,
    *,
    config: VQ2GuidanceConfig,
) -> Optional[VQ2GuidanceWithholdingReason]:
    timing = state.timing
    evaluation_ns = safety.evaluation_monotonic_ns
    if (
        timing.host_clock_id != safety.evaluation_host_clock_id
        or timing.source_frame_publish_monotonic_ns is None
        or timing.source_frame_publish_monotonic_ns > evaluation_ns
        or timing.decision_time_monotonic_ns > evaluation_ns
        or timing.measurement_time_monotonic_ns > evaluation_ns
    ):
        return VQ2GuidanceWithholdingReason.ACTIVE_TIMING_FUTURE
    if (
        evaluation_ns - timing.decision_time_monotonic_ns
        > config.max_state_decision_age_ns
    ):
        return VQ2GuidanceWithholdingReason.ACTIVE_TIMING_STALE
    if timing.measurement_uncertainty_ns > config.max_measurement_uncertainty_ns:
        return VQ2GuidanceWithholdingReason.ACTIVE_TIMING_UNCERTAIN
    if (
        evaluation_ns
        - timing.measurement_time_monotonic_ns
        + timing.measurement_uncertainty_ns
        > config.max_measurement_age_ns
    ):
        return VQ2GuidanceWithholdingReason.ACTIVE_TIMING_STALE
    if (
        max(0, timing.prediction_time_monotonic_ns - evaluation_ns)
        + timing.delay_uncertainty_ns
        > config.max_prediction_lead_ns
    ):
        return VQ2GuidanceWithholdingReason.ACTIVE_PREDICTION_HORIZON
    return None


def _memory_with_active_state(
    memory: VQ2GuidanceMemory,
    source: VQ2GuidanceSource,
) -> VQ2GuidanceMemory:
    history = memory.seen_active_measurements
    source_key = (source.source_frame, source.source_candidate_id)
    if not history or (history[-1][0], history[-1][1]) != source_key:
        history = (
            *history,
            (
                source.source_frame,
                source.source_candidate_id,
                source.measurement_update_sequence,
            ),
        )
    return VQ2GuidanceMemory(
        safety=memory.safety,
        active_source=source,
        seen_active_measurements=history,
        retired_active_tracker_ids=memory.retired_active_tracker_ids,
    )


def _evaluate_decision(
    safety: VQ2SafetyGuidanceInput,
    state: Optional[RelativeGateStateV1],
    source: Optional[VQ2GuidanceSource],
    *,
    active_rejection: Optional[VQ2GuidanceWithholdingReason],
    shadow_track_count: int,
    config: VQ2GuidanceConfig,
) -> VQ2GuidanceDecision:
    objective_kind = _OBJECTIVE_BY_PHASE[safety.phase]
    corridor = _corridor_for_phase(safety.phase, config)
    rate_limit = _rate_limit_for_phase(safety.phase, config)

    error: Optional[tuple[float, float]] = None
    margin: Optional[tuple[float, float]] = None
    corridor_eligible = False
    rate_eligible = False
    scale_eligible = False
    expansion_eligible = False
    if state is not None:
        sigma = config.uncertainty_sigma_multiplier
        covariance = state.covariance.matrix
        bearing_sigma = (math.sqrt(covariance[0][0]), math.sqrt(covariance[1][1]))
        rate_sigma = (math.sqrt(covariance[3][3]), math.sqrt(covariance[4][4]))
        error = tuple(
            abs(state.bearing_norm[index] - config.target_bearing_norm[index])
            + sigma * bearing_sigma[index]
            for index in range(2)
        )
        margin = tuple(corridor[index] - error[index] for index in range(2))
        corridor_eligible = all(item >= 0.0 for item in margin)
        conservative_rate = tuple(
            abs(state.bearing_rate_norm_s[index]) + sigma * rate_sigma[index]
            for index in range(2)
        )
        rate_eligible = all(
            conservative_rate[index] <= rate_limit[index] for index in range(2)
        )
        scale_eligible = (
            state.log_scale - sigma * math.sqrt(covariance[2][2])
            >= config.commit_min_log_scale
        )
        expansion_eligible = (
            state.expansion_rate_s - sigma * math.sqrt(covariance[5][5])
            >= config.commit_min_expansion_rate_s
        )

    reason: Optional[VQ2GuidanceWithholdingReason]
    if safety.race_state is VQ2GuidanceRaceState.NOT_UNDERWAY:
        objective_kind = VQ2GuidanceObjectiveKind.HOLD
        reason = VQ2GuidanceWithholdingReason.RACE_NOT_UNDERWAY
    elif safety.race_state in {
        VQ2GuidanceRaceState.FINISHED,
        VQ2GuidanceRaceState.ABORTED,
    }:
        objective_kind = VQ2GuidanceObjectiveKind.HOLD
        reason = VQ2GuidanceWithholdingReason.RACE_TERMINAL
    elif safety.phase is VQ2GuidancePhase.ACQUIRE:
        reason = VQ2GuidanceWithholdingReason.PHASE_HOLDS_MOTION
    elif safety.phase is VQ2GuidancePhase.CONFIRMATION:
        reason = VQ2GuidanceWithholdingReason.AWAITING_GATE_CREDIT
    elif safety.phase is VQ2GuidancePhase.POST_CREDIT_REACQUIRE:
        reason = VQ2GuidanceWithholdingReason.POST_CREDIT_REACQUIRE_HOLD
    elif active_rejection is not None:
        reason = active_rejection
    elif state is None or source is None:
        reason = VQ2GuidanceWithholdingReason.ACTIVE_STATE_REQUIRED
    elif state.dropout_count != 0:
        reason = VQ2GuidanceWithholdingReason.ACTIVE_STATE_DROPOUT
    elif state.innovation_accepted is False:
        reason = VQ2GuidanceWithholdingReason.ACTIVE_INNOVATION_REJECTED
    elif (
        safety.phase is VQ2GuidancePhase.ALIGN
        and state.health
        not in {RelativeStateHealth.HEALTHY, RelativeStateHealth.DEGRADED}
    ) or (
        safety.phase in {VQ2GuidancePhase.APPROACH, VQ2GuidancePhase.COMMIT}
        and state.health is not RelativeStateHealth.HEALTHY
    ):
        reason = VQ2GuidanceWithholdingReason.ACTIVE_STATE_HEALTH
    elif not corridor_eligible:
        reason = VQ2GuidanceWithholdingReason.OUTSIDE_UNCERTAINTY_CORRIDOR
    elif not rate_eligible:
        reason = VQ2GuidanceWithholdingReason.BEARING_RATE_UNCERTAIN
    elif safety.phase is VQ2GuidancePhase.COMMIT and state.last_clipping != FrameEdge.NONE:
        reason = VQ2GuidanceWithholdingReason.COMMIT_REQUIRES_UNCLIPPED
    elif safety.phase is VQ2GuidancePhase.COMMIT and not scale_eligible:
        reason = VQ2GuidanceWithholdingReason.COMMIT_SCALE_UNCERTAIN
    elif safety.phase is VQ2GuidancePhase.COMMIT and not expansion_eligible:
        reason = VQ2GuidanceWithholdingReason.COMMIT_EXPANSION_UNCERTAIN
    else:
        reason = None

    return VQ2GuidanceDecision(
        authority=safety.authority,
        phase=safety.phase,
        race_state=safety.race_state,
        evaluation_host_clock_id=safety.evaluation_host_clock_id,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns,
        phase_started_monotonic_ns=safety.phase_started_monotonic_ns,
        objective_kind=objective_kind,
        target_bearing_norm=config.target_bearing_norm,
        objective_permitted=reason is None,
        withholding_reason=reason,
        source=source,
        conservative_bearing_error_norm=error,
        corridor_half_width_norm=corridor,
        corridor_margin_norm=margin,
        corridor_eligible=corridor_eligible,
        bearing_rate_eligible=rate_eligible,
        scale_eligible=scale_eligible,
        expansion_eligible=expansion_eligible,
        shadow_track_count=shadow_track_count,
    )


def _withheld_decision(
    safety: VQ2SafetyGuidanceInput,
    reason: VQ2GuidanceWithholdingReason,
    *,
    config: VQ2GuidanceConfig,
    shadow_track_count: int,
) -> VQ2GuidanceDecision:
    return VQ2GuidanceDecision(
        authority=safety.authority,
        phase=safety.phase,
        race_state=safety.race_state,
        evaluation_host_clock_id=safety.evaluation_host_clock_id,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns,
        phase_started_monotonic_ns=safety.phase_started_monotonic_ns,
        objective_kind=VQ2GuidanceObjectiveKind.HOLD,
        target_bearing_norm=config.target_bearing_norm,
        objective_permitted=False,
        withholding_reason=reason,
        source=None,
        conservative_bearing_error_norm=None,
        corridor_half_width_norm=_corridor_for_phase(safety.phase, config),
        corridor_margin_norm=None,
        corridor_eligible=False,
        bearing_rate_eligible=False,
        scale_eligible=False,
        expansion_eligible=False,
        shadow_track_count=shadow_track_count,
    )


def _source_from_state(state: RelativeGateStateV1) -> VQ2GuidanceSource:
    timing = state.timing
    if (
        timing.source_frame is None
        or timing.source_frame_publication_sequence is None
        or timing.source_frame_publish_monotonic_ns is None
    ):
        raise ValueError("relative state source correlation is incomplete")
    return VQ2GuidanceSource(
        host_clock_id=timing.host_clock_id,
        decision_time_monotonic_ns=timing.decision_time_monotonic_ns,
        prediction_time_monotonic_ns=timing.prediction_time_monotonic_ns,
        source_frame=timing.source_frame,
        source_frame_publication_sequence=timing.source_frame_publication_sequence,
        source_frame_publish_monotonic_ns=timing.source_frame_publish_monotonic_ns,
        tracker_id=state.tracker_id,
        track_role=state.track_role,
        state_sequence=state.state_sequence,
        measurement_update_sequence=state.measurement_update_sequence,
        source_candidate_id=state.source_candidate_id,
    )


def _nonregressing_snapshot(
    previous: GateAuthorityEpochV1,
    current: GateAuthorityEpochV1,
) -> bool:
    if (
        current.race_status_sequence < previous.race_status_sequence
        or current.race_status_boot_ms < previous.race_status_boot_ms
        or current.frame_publication_sequence_not_before
        < previous.frame_publication_sequence_not_before
        or current.frame_publish_monotonic_ns_not_before
        < previous.frame_publish_monotonic_ns_not_before
    ):
        return False
    return not (
        current.race_status_sequence == previous.race_status_sequence
        and current.race_status_boot_ms != previous.race_status_boot_ms
    )


def _strict_forward_snapshot(
    previous: GateAuthorityEpochV1,
    current: GateAuthorityEpochV1,
) -> bool:
    return bool(
        current.race_status_sequence > previous.race_status_sequence
        and current.race_status_boot_ms > previous.race_status_boot_ms
        and current.frame_publication_sequence_not_before
        > previous.frame_publication_sequence_not_before
        and current.frame_publish_monotonic_ns_not_before
        > previous.frame_publish_monotonic_ns_not_before
    )


def _corridor_for_phase(
    phase: VQ2GuidancePhase,
    config: VQ2GuidanceConfig,
) -> tuple[float, float]:
    if phase in {
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
    }:
        return config.align_corridor_half_width_norm
    if phase is VQ2GuidancePhase.APPROACH:
        return config.approach_corridor_half_width_norm
    return config.commit_corridor_half_width_norm


def _rate_limit_for_phase(
    phase: VQ2GuidancePhase,
    config: VQ2GuidanceConfig,
) -> tuple[float, float]:
    if phase in {
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
    }:
        return config.align_rate_limit_norm_s
    if phase is VQ2GuidancePhase.APPROACH:
        return config.approach_rate_limit_norm_s
    return config.commit_rate_limit_norm_s


def _finite(
    value: float,
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_pair(
    value: tuple[float, float],
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strictly_positive: bool = False,
) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{name} must be an exact 2-tuple")
    result = tuple(
        _finite(item, f"{name}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    )
    if strictly_positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _nonnegative_exact_int(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_exact_int(value: int, name: str) -> int:
    result = _nonnegative_exact_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _pair_ordered(
    tightest: tuple[float, float],
    middle: tuple[float, float],
    widest: tuple[float, float],
) -> bool:
    return all(
        tightest[index] <= middle[index] <= widest[index]
        for index in range(2)
    )


_NEXT_PHASES = {
    VQ2GuidancePhase.ACQUIRE: frozenset({VQ2GuidancePhase.ALIGN}),
    VQ2GuidancePhase.ALIGN: frozenset({VQ2GuidancePhase.APPROACH}),
    VQ2GuidancePhase.APPROACH: frozenset({VQ2GuidancePhase.COMMIT}),
    VQ2GuidancePhase.COMMIT: frozenset({VQ2GuidancePhase.CONFIRMATION}),
    VQ2GuidancePhase.CONFIRMATION: frozenset(),
    VQ2GuidancePhase.POST_CREDIT_REACQUIRE: frozenset(
        {VQ2GuidancePhase.ACQUIRE}
    ),
}

_NEXT_RACE_STATES = {
    VQ2GuidanceRaceState.NOT_UNDERWAY: frozenset(
        {VQ2GuidanceRaceState.UNDERWAY, VQ2GuidanceRaceState.ABORTED}
    ),
    VQ2GuidanceRaceState.UNDERWAY: frozenset(
        {VQ2GuidanceRaceState.FINISHED, VQ2GuidanceRaceState.ABORTED}
    ),
    VQ2GuidanceRaceState.FINISHED: frozenset(),
    VQ2GuidanceRaceState.ABORTED: frozenset(),
}

_OBJECTIVE_BY_PHASE = {
    VQ2GuidancePhase.ACQUIRE: VQ2GuidanceObjectiveKind.ACQUIRE_ACTIVE_GATE,
    VQ2GuidancePhase.ALIGN: VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE,
    VQ2GuidancePhase.APPROACH: VQ2GuidanceObjectiveKind.APPROACH_ACTIVE_GATE,
    VQ2GuidancePhase.COMMIT: VQ2GuidanceObjectiveKind.COMMIT_ACTIVE_GATE,
    VQ2GuidancePhase.CONFIRMATION: VQ2GuidanceObjectiveKind.CONFIRM_GATE_CREDIT,
    VQ2GuidancePhase.POST_CREDIT_REACQUIRE: (
        VQ2GuidanceObjectiveKind.REACQUIRE_AFTER_CREDIT
    ),
}


DEFAULT_VQ2_GUIDANCE_CONFIG = VQ2GuidanceConfig()


__all__ = [
    "DEFAULT_VQ2_GUIDANCE_CONFIG",
    "VQ2GuidanceConfig",
    "VQ2GuidanceDecision",
    "VQ2GuidanceMemory",
    "VQ2GuidanceObjectiveKind",
    "VQ2GuidancePhase",
    "VQ2GuidanceRaceState",
    "VQ2GuidanceSource",
    "VQ2GuidanceTransition",
    "VQ2GuidanceWithholdingReason",
    "VQ2SafetyGuidanceInput",
    "step_vq2_guidance",
]
