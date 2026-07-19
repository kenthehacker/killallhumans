"""Pure predictive control for the build-3385 VQ2 feature-state path.

This module consumes the frozen :class:`RelativeGateStateV1` contract and
returns the frozen :class:`CommandProposalV1` contract.  It deliberately has
no transport, reset, arm, approval, race-progression, or simulator imports.
The proposal is controller intent only; the external safety supervisor remains
the sole authority that may approve a command for transport.

``ControllerAttitudeInput`` is deliberately local and currently carries no
timestamp, clock identity, or source correlation.  The frozen proposal cannot
bind attitude provenance.  This candidate is therefore ineligible for shadow,
runtime, or powered wiring until a reviewed IMU timing/derotation seam exists.
Likewise, its body-rate clamp limits requested intent only; supervisor
watchdogs still own actual attitude/body-rate aborts.

Phase elapsed time is derived only from a host-monotonic phase start and the
proposal timestamp.  The tick carries the safety-expected phase start and a
minimum objective-evaluation watermark.  This pure module can compare those
local values but cannot authenticate their ownership, and the frozen proposal
cannot bind them; the integration adapter and supervisor must retain that
responsibility.

Gate 0 preserves the representable parts of the proved legacy controller:
normalized horizontal bearing to roll target, the launch/boost thrust schedule,
the 640x360 vertical pixel PD law, and the same quaternion attitude-to-rate PD.
This bounded candidate accepts only the exact centered image objective.
Upstream filtering must supply ``bearing_rate_norm_s`` corresponding to the
legacy filtered pixel rate for exact rate-term equivalence.  Legacy bbox-area
crossing/corridor logic is intentionally absent because it is neither present
nor safely reconstructable from ``RelativeGateStateV1``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from competition.vq2_contracts import (
    CommandProposalV1,
    GateAuthorityEpochV1,
    RelativeGateStateV1,
    RelativeStateHealth,
    SaturationDiagnosticsV1,
    TrackRole,
    UncertaintyDiagnosticsV1,
    validate_command_proposal_source,
)


_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


class VQ2ControlPhase(str, Enum):
    """The only offline controller modes implemented in this candidate."""

    GATE0_APPROACH = "gate0_approach"
    GATE1_RECENTER = "gate1_recenter"


class ControllerInputError(ValueError):
    """Raised when no contract-valid proposal can represent malformed input."""


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return 0.0 if value == 0.0 else value


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded token")
    return value


def _optional_reason(value: object, label: str) -> Optional[str]:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > 192:
        raise ValueError(f"{label} must be a nonempty string of at most 192 characters")
    return value


def _float_tuple(value: object, length: int, label: str) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    return tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class ControllerAttitudeInput:
    """Local read-only attitude input; not a wire or transport value.

    ``orientation_body_to_world_wxyz`` follows the existing VQ2 IMU estimate:
    a unit quaternion in ``(w, x, y, z)`` order.  Body rates are FRD rad/s.
    """

    orientation_body_to_world_wxyz: tuple[float, float, float, float]
    body_rates_rad_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        orientation = _float_tuple(
            self.orientation_body_to_world_wxyz,
            4,
            "orientation_body_to_world_wxyz",
        )
        norm = math.sqrt(sum(component * component for component in orientation))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("orientation quaternion must be unit length")
        rates = _float_tuple(self.body_rates_rad_s, 3, "body_rates_rad_s")
        object.__setattr__(self, "orientation_body_to_world_wxyz", orientation)
        object.__setattr__(self, "body_rates_rad_s", rates)


@dataclass(frozen=True, slots=True)
class ControllerTickInput:
    """Caller-owned tick identity and expected safety authority.

    Supplying this value grants no send or approval authority.  It only gives
    the pure controller the exact identifiers required by ``CommandProposalV1``.
    """

    proposal_id: int
    control_tick_id: int
    host_clock_id: str
    proposal_monotonic_ns: int
    control_tick_deadline_monotonic_ns: int
    minimum_state_decision_monotonic_ns: int
    minimum_state_sequence: int
    expected_phase_started_monotonic_ns: int
    minimum_phase_evaluation_monotonic_ns: int
    expected_authority: GateAuthorityEpochV1

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.proposal_id, "proposal_id")
        _exact_nonnegative_int(self.control_tick_id, "control_tick_id")
        _bounded_token(self.host_clock_id, "host_clock_id")
        proposal_time = _exact_nonnegative_int(
            self.proposal_monotonic_ns, "proposal_monotonic_ns"
        )
        deadline = _exact_nonnegative_int(
            self.control_tick_deadline_monotonic_ns,
            "control_tick_deadline_monotonic_ns",
        )
        minimum_state_decision = _exact_nonnegative_int(
            self.minimum_state_decision_monotonic_ns,
            "minimum_state_decision_monotonic_ns",
        )
        _exact_nonnegative_int(self.minimum_state_sequence, "minimum_state_sequence")
        expected_phase_start = _exact_nonnegative_int(
            self.expected_phase_started_monotonic_ns,
            "expected_phase_started_monotonic_ns",
        )
        minimum_phase_evaluation = _exact_nonnegative_int(
            self.minimum_phase_evaluation_monotonic_ns,
            "minimum_phase_evaluation_monotonic_ns",
        )
        if deadline < proposal_time:
            raise ControllerInputError("control tick deadline predates proposal time")
        if minimum_state_decision > proposal_time:
            raise ControllerInputError(
                "minimum state decision watermark postdates proposal time"
            )
        if expected_phase_start > proposal_time:
            raise ControllerInputError(
                "expected phase start postdates proposal time"
            )
        if minimum_phase_evaluation > proposal_time:
            raise ControllerInputError(
                "minimum phase evaluation watermark postdates proposal time"
            )
        if type(self.expected_authority) is not GateAuthorityEpochV1:
            raise TypeError("expected_authority must be exact GateAuthorityEpochV1")


@dataclass(frozen=True, slots=True)
class ControllerPhaseInput:
    """Explicit local phase/objective input, suitable for a guidance adapter."""

    mode: VQ2ControlPhase
    phase_host_clock_id: str
    phase_started_monotonic_ns: int
    evaluation_monotonic_ns: int
    initial_pitch_rad: float
    target_bearing_norm: tuple[float, float] = (0.0, 0.0)
    objective_permitted: bool = True
    withholding_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.mode) is not VQ2ControlPhase:
            raise TypeError("mode must be VQ2ControlPhase")
        _bounded_token(self.phase_host_clock_id, "phase_host_clock_id")
        phase_started = _exact_nonnegative_int(
            self.phase_started_monotonic_ns,
            "phase_started_monotonic_ns",
        )
        evaluation = _exact_nonnegative_int(
            self.evaluation_monotonic_ns,
            "evaluation_monotonic_ns",
        )
        if evaluation < phase_started:
            raise ControllerInputError(
                "phase evaluation predates phase start"
            )
        initial_pitch = _finite_float(self.initial_pitch_rad, "initial_pitch_rad")
        target = _float_tuple(self.target_bearing_norm, 2, "target_bearing_norm")
        if any(abs(component) > 4.0 for component in target):
            raise ValueError("target_bearing_norm must remain within contract bounds")
        if target != (0.0, 0.0):
            raise ValueError(
                "this bounded controller candidate requires an exact centered target"
            )
        if type(self.objective_permitted) is not bool:
            raise TypeError("objective_permitted must be an exact bool")
        reason = _optional_reason(self.withholding_reason, "withholding_reason")
        if self.objective_permitted == (reason is not None):
            raise ValueError(
                "withholding_reason is present exactly when the objective is withheld"
            )
        if self.mode is VQ2ControlPhase.GATE1_RECENTER and initial_pitch != 0.0:
            raise ValueError("Gate 1 recenter requires an exact-zero target pitch basis")
        object.__setattr__(self, "initial_pitch_rad", initial_pitch)
        object.__setattr__(self, "target_bearing_norm", target)
        object.__setattr__(self, "withholding_reason", reason)


@dataclass(frozen=True, slots=True)
class PredictiveControllerConfig:
    """Reviewed, immutable local tuning and fail-closed envelope."""

    gate0_roll_gain_rad_per_norm: float = 0.15
    gate0_max_roll_rad: float = 0.08
    gate0_pitch_blend_s: float = 0.8
    gate0_launch_end_s: float = 0.15
    gate0_boost_end_s: float = 0.45
    gate0_launch_thrust: float = 0.26
    gate0_boost_thrust: float = 0.32
    gate0_max_body_rate_rad_s: float = 0.25
    gate0_min_thrust: float = 0.21
    gate0_max_thrust: float = 0.32
    attitude_kp_roll: float = 1.0
    attitude_kp_pitch: float = 0.5
    attitude_kd_roll: float = 0.4
    attitude_kd_pitch: float = 0.2
    legacy_vertical_half_image_px: float = 180.0
    legacy_vertical_error_scale_px: float = 90.0
    legacy_vertical_rate_limit_px_s: float = 300.0
    vertical_neutral_thrust: float = 0.275
    vertical_position_gain: float = 0.040
    vertical_rate_damping_per_px_s: float = 0.00070
    gate1_roll_gain_rad_per_norm: float = 0.12
    gate1_roll_rate_gain_rad_s_per_norm_s: float = 0.025
    gate1_max_roll_rad: float = 0.05
    gate1_max_body_rate_rad_s: float = 0.12
    gate1_min_thrust: float = 0.21
    gate1_max_thrust: float = 0.30
    gate1_max_duration_s: float = 0.60
    gate1_corridor_x_norm: float = 0.10
    gate1_corridor_y_norm: float = 0.12
    gate1_corridor_rate_norm_s: float = 0.25
    max_abs_initial_pitch_rad: float = 0.6108652381980153
    max_abs_bearing_error_norm: float = 1.50
    max_abs_bearing_rate_norm_s: float = 4.0
    max_state_age_ns: int = 100_000_000
    max_measurement_age_ns: int = 150_000_000
    max_prediction_lead_ns: int = 100_000_000
    max_phase_objective_age_ns: int = 100_000_000
    max_measurement_uncertainty_ns: int = 50_000_000
    max_delay_uncertainty_ns: int = 50_000_000
    max_bearing_variance: float = 0.25
    max_log_scale_variance: float = 1.0
    max_bearing_rate_variance: float = 16.0
    max_expansion_rate_variance: float = 16.0

    def __post_init__(self) -> None:
        positive_fields = (
            "gate0_roll_gain_rad_per_norm",
            "gate0_max_roll_rad",
            "gate0_pitch_blend_s",
            "gate0_launch_end_s",
            "gate0_boost_end_s",
            "gate0_launch_thrust",
            "gate0_boost_thrust",
            "gate0_max_body_rate_rad_s",
            "gate0_min_thrust",
            "gate0_max_thrust",
            "attitude_kp_roll",
            "attitude_kp_pitch",
            "legacy_vertical_half_image_px",
            "legacy_vertical_error_scale_px",
            "legacy_vertical_rate_limit_px_s",
            "vertical_neutral_thrust",
            "vertical_position_gain",
            "vertical_rate_damping_per_px_s",
            "gate1_roll_gain_rad_per_norm",
            "gate1_roll_rate_gain_rad_s_per_norm_s",
            "gate1_max_roll_rad",
            "gate1_max_body_rate_rad_s",
            "gate1_min_thrust",
            "gate1_max_thrust",
            "gate1_max_duration_s",
            "gate1_corridor_x_norm",
            "gate1_corridor_y_norm",
            "gate1_corridor_rate_norm_s",
            "max_abs_initial_pitch_rad",
            "max_abs_bearing_error_norm",
            "max_abs_bearing_rate_norm_s",
            "max_bearing_variance",
            "max_log_scale_variance",
            "max_bearing_rate_variance",
            "max_expansion_rate_variance",
        )
        nonnegative_fields = ("attitude_kd_roll", "attitude_kd_pitch")
        for name in positive_fields:
            _positive_float(getattr(self, name), name)
        for name in nonnegative_fields:
            _nonnegative_float(getattr(self, name), name)
        for name in (
            "max_state_age_ns",
            "max_measurement_age_ns",
            "max_prediction_lead_ns",
            "max_phase_objective_age_ns",
            "max_measurement_uncertainty_ns",
            "max_delay_uncertainty_ns",
        ):
            if _exact_nonnegative_int(getattr(self, name), name) == 0:
                raise ValueError(f"{name} must be positive")
        frozen_gate0_legacy_values = {
            "gate0_roll_gain_rad_per_norm": 0.15,
            "gate0_max_roll_rad": 0.08,
            "gate0_pitch_blend_s": 0.8,
            "gate0_launch_end_s": 0.15,
            "gate0_boost_end_s": 0.45,
            "gate0_launch_thrust": 0.26,
            "gate0_boost_thrust": 0.32,
            "gate0_max_body_rate_rad_s": 0.25,
            "gate0_min_thrust": 0.21,
            "gate0_max_thrust": 0.32,
            "attitude_kp_roll": 1.0,
            "attitude_kp_pitch": 0.5,
            "attitude_kd_roll": 0.4,
            "attitude_kd_pitch": 0.2,
            "legacy_vertical_half_image_px": 180.0,
            "legacy_vertical_error_scale_px": 90.0,
            "legacy_vertical_rate_limit_px_s": 300.0,
            "vertical_neutral_thrust": 0.275,
            "vertical_position_gain": 0.040,
            "vertical_rate_damping_per_px_s": 0.00070,
        }
        for name, frozen_value in frozen_gate0_legacy_values.items():
            if getattr(self, name) != frozen_value:
                raise ValueError(
                    f"{name} is frozen to its reviewed Gate 0 legacy value"
                )
        frozen_gate1_corridor_values = {
            "gate1_corridor_x_norm": 0.10,
            "gate1_corridor_y_norm": 0.12,
            "gate1_corridor_rate_norm_s": 0.25,
        }
        for name, frozen_value in frozen_gate1_corridor_values.items():
            if getattr(self, name) != frozen_value:
                raise ValueError(
                    f"{name} is frozen to its reviewed Gate 1 corridor value"
                )
        hard_float_maxima = {
            "gate1_roll_gain_rad_per_norm": 0.12,
            "gate1_roll_rate_gain_rad_s_per_norm_s": 0.025,
            "gate1_max_roll_rad": 0.05,
            "gate1_max_body_rate_rad_s": 0.12,
            "gate1_max_thrust": 0.30,
            "gate1_max_duration_s": 0.60,
            "max_abs_initial_pitch_rad": 0.6108652381980153,
            "max_abs_bearing_error_norm": 1.50,
            "max_abs_bearing_rate_norm_s": 4.0,
            "max_bearing_variance": 0.25,
            "max_log_scale_variance": 1.0,
            "max_bearing_rate_variance": 16.0,
            "max_expansion_rate_variance": 16.0,
        }
        hard_integer_maxima = {
            "max_state_age_ns": 100_000_000,
            "max_measurement_age_ns": 150_000_000,
            "max_prediction_lead_ns": 100_000_000,
            "max_phase_objective_age_ns": 100_000_000,
            "max_measurement_uncertainty_ns": 50_000_000,
            "max_delay_uncertainty_ns": 50_000_000,
        }
        for name, hard_maximum in hard_float_maxima.items():
            if getattr(self, name) > hard_maximum:
                raise ValueError(
                    f"{name} cannot loosen its reviewed hard maximum"
                )
        for name, hard_maximum in hard_integer_maxima.items():
            if getattr(self, name) > hard_maximum:
                raise ValueError(
                    f"{name} cannot loosen its reviewed hard maximum"
                )
        for name, hard_minimum in (
            ("gate1_min_thrust", 0.21),
        ):
            if getattr(self, name) < hard_minimum:
                raise ValueError(
                    f"{name} cannot loosen its reviewed hard minimum"
                )
        if self.gate0_launch_end_s >= self.gate0_boost_end_s:
            raise ValueError("Gate 0 launch window must end before boost window")
        if not 0.0 <= self.gate0_min_thrust <= self.gate0_max_thrust <= 1.0:
            raise ValueError("Gate 0 thrust envelope is invalid")
        if not 0.0 <= self.gate1_min_thrust <= self.gate1_max_thrust <= 1.0:
            raise ValueError("Gate 1 thrust envelope is invalid")
        if not self.gate0_min_thrust <= self.vertical_neutral_thrust <= self.gate0_max_thrust:
            raise ValueError("vertical neutral thrust lies outside Gate 0 envelope")
        if not self.gate1_min_thrust <= self.vertical_neutral_thrust <= self.gate1_max_thrust:
            raise ValueError("vertical neutral thrust lies outside Gate 1 envelope")
        if not self.gate0_min_thrust <= self.gate0_launch_thrust <= self.gate0_max_thrust:
            raise ValueError("Gate 0 launch thrust lies outside its envelope")
        if not self.gate0_min_thrust <= self.gate0_boost_thrust <= self.gate0_max_thrust:
            raise ValueError("Gate 0 boost thrust lies outside its envelope")
        if self.gate1_max_body_rate_rad_s > self.gate0_max_body_rate_rad_s:
            raise ValueError("Gate 1 recenter body-rate envelope must not exceed Gate 0")
        if self.gate1_max_thrust > self.gate0_max_thrust:
            raise ValueError("Gate 1 recenter thrust envelope must not exceed Gate 0")


DEFAULT_PREDICTIVE_CONTROLLER_CONFIG = PredictiveControllerConfig()


def _clamp(value: float, minimum: float, maximum: float) -> tuple[float, bool]:
    limited = max(minimum, min(maximum, value))
    return limited, limited != value


def _yaw_from_quaternion(q: tuple[float, float, float, float]) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quaternion_from_euler(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _attitude_body_rates(
    attitude: ControllerAttitudeInput,
    *,
    target_roll_rad: float,
    target_pitch_rad: float,
    rate_limit_rad_s: float,
    config: PredictiveControllerConfig,
) -> tuple[tuple[float, float, float], tuple[bool, bool, bool]]:
    """Match the proved runner's roll/pitch quaternion-error PD, without yaw."""

    current = attitude.orientation_body_to_world_wxyz
    desired = _quaternion_from_euler(
        target_roll_rad,
        target_pitch_rad,
        _yaw_from_quaternion(current),
    )
    cw, cx, cy, cz = current[0], -current[1], -current[2], -current[3]
    dw, dx, dy, dz = desired
    error_w = cw * dw - cx * dx - cy * dy - cz * dz
    error_x = cw * dx + cx * dw + cy * dz - cz * dy
    error_y = cw * dy - cx * dz + cy * dw + cz * dx
    if error_w < 0.0:
        error_x, error_y = -error_x, -error_y
    raw_roll = (
        2.0 * config.attitude_kp_roll * error_x
        - config.attitude_kd_roll * attitude.body_rates_rad_s[0]
    )
    raw_pitch = (
        2.0 * config.attitude_kp_pitch * error_y
        - config.attitude_kd_pitch * attitude.body_rates_rad_s[1]
    )
    roll, roll_limited = _clamp(raw_roll, -rate_limit_rad_s, rate_limit_rad_s)
    pitch, pitch_limited = _clamp(raw_pitch, -rate_limit_rad_s, rate_limit_rad_s)
    return (roll, pitch, 0.0), (roll_limited, pitch_limited, False)


def _vertical_thrust(
    bearing_error_y_norm: float,
    bearing_rate_y_norm_s: float,
    *,
    minimum: float,
    maximum: float,
    config: PredictiveControllerConfig,
) -> tuple[float, bool]:
    """Normalized form of the legacy 640x360 vertical pixel PD law."""

    control_y_px = (
        config.legacy_vertical_half_image_px
        * (1.0 + bearing_error_y_norm)
    )
    control_y_rate_px_s = (
        config.legacy_vertical_half_image_px * bearing_rate_y_norm_s
    )
    position_term = config.vertical_position_gain * max(
        -1.0,
        min(
            1.0,
            (config.legacy_vertical_half_image_px - control_y_px)
            / config.legacy_vertical_error_scale_px,
        ),
    )
    rate_term = -config.vertical_rate_damping_per_px_s * max(
        -config.legacy_vertical_rate_limit_px_s,
        min(config.legacy_vertical_rate_limit_px_s, control_y_rate_px_s),
    )
    return _clamp(
        config.vertical_neutral_thrust + position_term + rate_term,
        minimum,
        maximum,
    )


def _failsafe_proposal(
    tick: ControllerTickInput,
    phase: ControllerPhaseInput,
    reason: str,
    *,
    uncertainty_limited: bool,
) -> CommandProposalV1:
    diagnostic_reason = reason if uncertainty_limited else None
    return CommandProposalV1(
        proposal_id=tick.proposal_id,
        control_tick_id=tick.control_tick_id,
        host_clock_id=tick.host_clock_id,
        proposal_monotonic_ns=tick.proposal_monotonic_ns,
        control_tick_deadline_monotonic_ns=(
            tick.control_tick_deadline_monotonic_ns
        ),
        source_state_decision_monotonic_ns=None,
        source_state_prediction_monotonic_ns=None,
        source_frame=None,
        source_frame_publication_sequence=None,
        source_frame_publish_monotonic_ns=None,
        source_tracker_id=None,
        source_track_role=None,
        source_state_sequence=None,
        source_measurement_update_sequence=None,
        source_candidate_id=None,
        authority=tick.expected_authority,
        requested_body_rates_rad_s=(0.0, 0.0, 0.0),
        requested_thrust=0.0,
        phase=phase.mode.value,
        reason=f"withheld:{reason}",
        saturation=SaturationDiagnosticsV1(
            body_rate_axes=(False, False, False),
            thrust=False,
        ),
        uncertainty=UncertaintyDiagnosticsV1(
            limited=uncertainty_limited,
            reason=diagnostic_reason,
        ),
    )


def _inside_gate1_corridor(
    state: RelativeGateStateV1,
    phase: ControllerPhaseInput,
    config: PredictiveControllerConfig,
) -> bool:
    bearing_error_x = state.bearing_norm[0] - phase.target_bearing_norm[0]
    bearing_error_y = state.bearing_norm[1] - phase.target_bearing_norm[1]
    return (
        abs(bearing_error_x) <= config.gate1_corridor_x_norm
        and abs(bearing_error_y) <= config.gate1_corridor_y_norm
        and abs(state.bearing_rate_norm_s[0])
        <= config.gate1_corridor_rate_norm_s
        and abs(state.bearing_rate_norm_s[1])
        <= config.gate1_corridor_rate_norm_s
    )


def _eligibility_failure(
    state: RelativeGateStateV1,
    tick: ControllerTickInput,
    phase: ControllerPhaseInput,
    config: PredictiveControllerConfig,
) -> tuple[Optional[str], bool]:
    expected_gate_index = (
        0 if phase.mode is VQ2ControlPhase.GATE0_APPROACH else 1
    )
    if tick.host_clock_id != tick.expected_authority.camera_host_clock_id:
        return "tick_host_clock_authority_mismatch", False
    if phase.phase_host_clock_id != tick.host_clock_id:
        return "phase_host_clock_mismatch", False
    if phase.phase_started_monotonic_ns > tick.proposal_monotonic_ns:
        return "phase_start_from_future", False
    if (
        phase.phase_started_monotonic_ns
        != tick.expected_phase_started_monotonic_ns
    ):
        return "phase_start_mismatch", False
    if phase.evaluation_monotonic_ns > tick.proposal_monotonic_ns:
        return "phase_evaluation_from_future", False
    if (
        phase.evaluation_monotonic_ns
        < tick.minimum_phase_evaluation_monotonic_ns
    ):
        return "phase_evaluation_regressed", False
    if (
        tick.proposal_monotonic_ns - phase.evaluation_monotonic_ns
        > config.max_phase_objective_age_ns
    ):
        return "phase_objective_stale", False
    if state.timing.host_clock_id != tick.host_clock_id:
        return "state_host_clock_mismatch", False
    if state.authority != tick.expected_authority:
        return "state_authority_mismatch", False
    if state.authority.expected_gate_index != expected_gate_index:
        return "phase_gate_authority_mismatch", False
    if state.state_sequence < tick.minimum_state_sequence:
        return "state_sequence_regressed", False
    if state.track_role is not TrackRole.ACTIVE:
        return "inactive_source_track", False
    if not phase.objective_permitted:
        return f"objective_withheld:{phase.withholding_reason}", False
    if state.innovation_accepted is False:
        return "state_innovation_rejected", True
    allowed_health = (
        {RelativeStateHealth.HEALTHY}
        if phase.mode is VQ2ControlPhase.GATE0_APPROACH
        else {RelativeStateHealth.HEALTHY, RelativeStateHealth.DEGRADED}
    )
    if state.health not in allowed_health:
        return f"state_health_{state.health.value}", True
    decision_time = state.timing.decision_time_monotonic_ns
    measurement_time = state.timing.measurement_time_monotonic_ns
    prediction_time = state.timing.prediction_time_monotonic_ns
    if decision_time > tick.proposal_monotonic_ns:
        return "state_decision_from_future", True
    if decision_time < tick.minimum_state_decision_monotonic_ns:
        return "state_decision_regressed", True
    if tick.proposal_monotonic_ns - decision_time > config.max_state_age_ns:
        return "state_decision_stale", True
    if measurement_time > tick.proposal_monotonic_ns:
        return "state_measurement_from_future", True
    if (
        tick.proposal_monotonic_ns
        - measurement_time
        + state.timing.measurement_uncertainty_ns
        > config.max_measurement_age_ns
    ):
        return "state_measurement_stale", True
    if state.timing.delay_uncertainty_ns > config.max_delay_uncertainty_ns:
        return "prediction_delay_uncertainty", True
    prediction_lead_ns = max(
        0, prediction_time - tick.proposal_monotonic_ns
    )
    if (
        prediction_lead_ns + state.timing.delay_uncertainty_ns
        > config.max_prediction_lead_ns
    ):
        return "state_prediction_too_far_ahead", True
    if (
        state.timing.measurement_uncertainty_ns
        > config.max_measurement_uncertainty_ns
    ):
        return "measurement_time_uncertainty", True
    bearing_error = (
        state.bearing_norm[0] - phase.target_bearing_norm[0],
        state.bearing_norm[1] - phase.target_bearing_norm[1],
    )
    if any(abs(component) > config.max_abs_bearing_error_norm for component in bearing_error):
        return "bearing_error_outside_control_envelope", True
    if any(
        abs(component) > config.max_abs_bearing_rate_norm_s
        for component in state.bearing_rate_norm_s
    ):
        return "bearing_rate_outside_control_envelope", True
    diagonal = tuple(state.covariance.matrix[index][index] for index in range(6))
    variance_limits = (
        config.max_bearing_variance,
        config.max_bearing_variance,
        config.max_log_scale_variance,
        config.max_bearing_rate_variance,
        config.max_bearing_rate_variance,
        config.max_expansion_rate_variance,
    )
    if any(value > limit for value, limit in zip(diagonal, variance_limits)):
        return "relative_state_covariance", True
    if abs(phase.initial_pitch_rad) > config.max_abs_initial_pitch_rad:
        return "initial_pitch_outside_control_envelope", True
    if (
        phase.mode is VQ2ControlPhase.GATE1_RECENTER
        and state.health is RelativeStateHealth.DEGRADED
        and not state.last_clipping
    ):
        if _inside_gate1_corridor(state, phase, config):
            return "gate1_recenter_corridor_unconfirmed_limited", True
        return "gate1_degraded_without_clipping", True
    return None, False


def propose_vq2_command(
    state: RelativeGateStateV1,
    *,
    attitude: ControllerAttitudeInput,
    tick: ControllerTickInput,
    phase: ControllerPhaseInput,
    config: PredictiveControllerConfig = DEFAULT_PREDICTIVE_CONTROLLER_CONFIG,
) -> CommandProposalV1:
    """Return one deterministic proposal, or an exact-zero fail-closed value.

    Malformed local values raise while valid-but-ineligible state, authority,
    objective, health, age, or uncertainty returns an exact-zero source-less
    proposal.  A nonzero proposal always cites and validates the complete exact
    source-state identity.
    """

    if type(state) is not RelativeGateStateV1:
        raise TypeError("state must be exact RelativeGateStateV1")
    if type(attitude) is not ControllerAttitudeInput:
        raise TypeError("attitude must be exact ControllerAttitudeInput")
    if type(tick) is not ControllerTickInput:
        raise TypeError("tick must be exact ControllerTickInput")
    if type(phase) is not ControllerPhaseInput:
        raise TypeError("phase must be exact ControllerPhaseInput")
    if type(config) is not PredictiveControllerConfig:
        raise TypeError("config must be exact PredictiveControllerConfig")

    failure, uncertainty_limited = _eligibility_failure(
        state, tick, phase, config
    )
    if failure is not None:
        return _failsafe_proposal(
            tick,
            phase,
            failure,
            uncertainty_limited=uncertainty_limited,
        )

    elapsed_s = (
        tick.proposal_monotonic_ns - phase.phase_started_monotonic_ns
    ) / 1_000_000_000
    if (
        phase.mode is VQ2ControlPhase.GATE1_RECENTER
        and elapsed_s >= config.gate1_max_duration_s
    ):
        return _failsafe_proposal(
            tick,
            phase,
            "gate1_recenter_time_limit",
            uncertainty_limited=False,
        )

    bearing_error_x = state.bearing_norm[0] - phase.target_bearing_norm[0]
    bearing_error_y = state.bearing_norm[1] - phase.target_bearing_norm[1]
    if phase.mode is VQ2ControlPhase.GATE1_RECENTER:
        if _inside_gate1_corridor(state, phase, config):
            corridor_limited = (
                state.health is RelativeStateHealth.DEGRADED
                or bool(state.last_clipping)
            )
            return _failsafe_proposal(
                tick,
                phase,
                (
                    "gate1_recenter_corridor_unconfirmed_limited"
                    if corridor_limited
                    else "gate1_recenter_corridor_reached"
                ),
                uncertainty_limited=corridor_limited,
            )
        target_roll_unclamped = (
            config.gate1_roll_gain_rad_per_norm * bearing_error_x
            + config.gate1_roll_rate_gain_rad_s_per_norm_s
            * state.bearing_rate_norm_s[0]
        )
        target_roll, _target_roll_limited = _clamp(
            target_roll_unclamped,
            -config.gate1_max_roll_rad,
            config.gate1_max_roll_rad,
        )
        target_pitch = 0.0
        rate_limit = config.gate1_max_body_rate_rad_s
        thrust, thrust_limited = _vertical_thrust(
            bearing_error_y,
            state.bearing_rate_norm_s[1],
            minimum=config.gate1_min_thrust,
            maximum=config.gate1_max_thrust,
            config=config,
        )
        reason = "bounded_gate1_recenter"
        uncertainty_limited = (
            state.health is RelativeStateHealth.DEGRADED
            or bool(state.last_clipping)
        )
        uncertainty_reason = (
            "bounded_gate1_recenter_degraded_or_clipped"
            if uncertainty_limited
            else None
        )
    else:
        target_roll, _target_roll_limited = _clamp(
            config.gate0_roll_gain_rad_per_norm * bearing_error_x,
            -config.gate0_max_roll_rad,
            config.gate0_max_roll_rad,
        )
        blend = min(1.0, elapsed_s / config.gate0_pitch_blend_s)
        target_pitch = (1.0 - blend) * phase.initial_pitch_rad
        rate_limit = config.gate0_max_body_rate_rad_s
        if elapsed_s < config.gate0_launch_end_s:
            thrust = config.gate0_launch_thrust
            thrust_limited = False
        elif elapsed_s < config.gate0_boost_end_s:
            thrust = config.gate0_boost_thrust
            thrust_limited = False
        else:
            thrust, thrust_limited = _vertical_thrust(
                bearing_error_y,
                state.bearing_rate_norm_s[1],
                minimum=config.gate0_min_thrust,
                maximum=config.gate0_max_thrust,
                config=config,
            )
        reason = "legacy_gate0_pixel_pd"
        uncertainty_limited = False
        uncertainty_reason = None

    rates, rate_saturation = _attitude_body_rates(
        attitude,
        target_roll_rad=target_roll,
        target_pitch_rad=target_pitch,
        rate_limit_rad_s=rate_limit,
        config=config,
    )
    proposal = CommandProposalV1(
        proposal_id=tick.proposal_id,
        control_tick_id=tick.control_tick_id,
        host_clock_id=tick.host_clock_id,
        proposal_monotonic_ns=tick.proposal_monotonic_ns,
        control_tick_deadline_monotonic_ns=(
            tick.control_tick_deadline_monotonic_ns
        ),
        source_state_decision_monotonic_ns=(
            state.timing.decision_time_monotonic_ns
        ),
        source_state_prediction_monotonic_ns=(
            state.timing.prediction_time_monotonic_ns
        ),
        source_frame=state.timing.source_frame,
        source_frame_publication_sequence=(
            state.timing.source_frame_publication_sequence
        ),
        source_frame_publish_monotonic_ns=(
            state.timing.source_frame_publish_monotonic_ns
        ),
        source_tracker_id=state.tracker_id,
        source_track_role=state.track_role,
        source_state_sequence=state.state_sequence,
        source_measurement_update_sequence=state.measurement_update_sequence,
        source_candidate_id=state.source_candidate_id,
        authority=state.authority,
        requested_body_rates_rad_s=rates,
        requested_thrust=thrust,
        phase=phase.mode.value,
        reason=reason,
        saturation=SaturationDiagnosticsV1(
            body_rate_axes=rate_saturation,
            thrust=thrust_limited,
        ),
        uncertainty=UncertaintyDiagnosticsV1(
            limited=uncertainty_limited,
            reason=uncertainty_reason,
        ),
    )
    validate_command_proposal_source(proposal, state)
    return proposal


__all__ = [
    "ControllerAttitudeInput",
    "ControllerInputError",
    "ControllerPhaseInput",
    "ControllerTickInput",
    "DEFAULT_PREDICTIVE_CONTROLLER_CONFIG",
    "PredictiveControllerConfig",
    "VQ2ControlPhase",
    "propose_vq2_command",
]
