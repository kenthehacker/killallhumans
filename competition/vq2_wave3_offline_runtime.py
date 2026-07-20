"""Deterministic generated/offline composition for the reviewed VQ2 stack.

This module joins already-decoded generated images, exact timestamped IMU
samples, the raw-camera relative estimator, the Wave 3 IMU adapter, and the
reviewed fixed-rate scheduler.  Its output stops at a quarantined
``CommandProposalV1``.  It owns no supervisor, approval, projection, send,
actuator, transport, reset, arm, cleanup, network, simulator, or powered
authority.

The camera-bearing correction remains standalone evidence.  The ordinary
relative estimator still consumes the unmodified camera observation.  Trace
events are assembled and resequenced by occurrence time so interleaved camera
and IMU facts are not falsely appended in observation order.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from competition.adapter import CameraFrame
from competition.vq2_contracts import (
    EventOutcome,
    FeatureCovarianceV1,
    FrameIdentityV1,
    FrameTimingV1,
    LatencyEventKind,
    LatencyEventV1,
    RelativeGateStateV1,
    validate_latency_event_sequence,
)
from competition.vq2_controller import ControllerTickInput
from competition.vq2_runtime import (
    ControlTickLeaseV1,
    FixedRateControlSchedulerV1,
    LatestFrameCursorV1,
    LatestFrameSelectionV1,
    MINIMUM_CONTROL_PERIOD_NS,
)
from competition.vq2_vision import VQ2VisionSnapshot
from competition.vq2_wave3_imu_adapter import (
    VQ2Wave3CoastLease,
    VQ2Wave3ImuAdapterMemory,
    VQ2Wave3ImuAdapterTransition,
    consume_vq2_wave3_coast_lease,
    step_vq2_wave3_imu_adapter,
)
from estimation.imu_attitude import ImuAttitudeConfig
from estimation.vq2_imu_derotation import (
    VQ2AttitudeDerotationInput,
    VQ2CameraToBodyCalibration,
    VQ2DerotationModel,
    VQ2ImuDerotationError,
    derotate_gate_observation,
)
from estimation.vq2_imu_provenance import (
    VQ2ImuProvenanceEstimator,
    VQ2ImuSource,
    VQ2TimedImuSample,
    VQ2TimestampedAttitude,
)
from estimation.vq2_relative_estimator import (
    RelativeEstimatorConfig,
    RelativeEstimatorError,
    RelativePredictionTarget,
    VQ2ImuCorrelatedEstimatorCoast,
    VQ2ImuCorrelatedEstimatorUpdate,
    VQ2RelativeGateEstimator,
)
from gate_detection.src.vq2_detector import VQ2GateDetector
from gate_detection.src.vq2_geometry import VQ2ApertureConfig
from gate_detection.src.vq2_observation_adapter import (
    gate_detection_with_aperture_to_observation_v1,
)
from planning.vq2_guidance import VQ2SafetyGuidanceInput


_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_CENTER_FEATURES = ("center_x_norm", "center_y_norm")

_CAMERA_STAGES = (
    (LatencyEventKind.CAMERA_FIRST_PACKET, "first_unique_packet_monotonic_ns"),
    (LatencyEventKind.CAMERA_FINAL_PACKET, "final_unique_packet_monotonic_ns"),
    (LatencyEventKind.FRAME_REASSEMBLED, "reassembly_complete_monotonic_ns"),
    (LatencyEventKind.DECODE_START, "decode_start_monotonic_ns"),
    (LatencyEventKind.DECODE_END, "decode_end_monotonic_ns"),
    (LatencyEventKind.FRAME_PUBLISHED, "publish_monotonic_ns"),
)

_OFFLINE_FORBIDDEN_TRACE_KINDS = frozenset(
    {
        LatencyEventKind.ACTUATOR_SAMPLE,
        LatencyEventKind.COMMAND_SEND_START,
        LatencyEventKind.COMMAND_SEND_END,
    }
)
_COMPLETED_TICK_KINDS = frozenset(
    {
        LatencyEventKind.CONTROL_TICK_DUE,
        LatencyEventKind.CONTROL_TICK_START,
        LatencyEventKind.PREDICTION_START,
        LatencyEventKind.PREDICTION_END,
        LatencyEventKind.ESTIMATOR_UPDATE_START,
        LatencyEventKind.ESTIMATOR_UPDATE_END,
        LatencyEventKind.CONTROLLER_START,
        LatencyEventKind.CONTROLLER_END,
        LatencyEventKind.CONTROL_TICK_END,
    }
)
_SKIPPED_TICK_KINDS = frozenset(
    {
        LatencyEventKind.CONTROL_TICK_DUE,
        LatencyEventKind.DEADLINE_MISSED,
        LatencyEventKind.CONTROL_TICK_SKIPPED,
    }
)
_OFFLINE_SKIP_REASONS = frozenset(
    {"tick_deadline_elapsed", "planned_work_exceeds_deadline"}
)
_COAST_LEASE_DISPOSITIONS = frozenset(
    {
        "coast_accepted",
        "coast_rejected",
        "distinct_frame_selected",
        "tick_skipped",
    }
)
_PERCEPTION_STAGE_KINDS = (
    LatencyEventKind.DETECTION_START,
    LatencyEventKind.DETECTION_END,
    LatencyEventKind.TRACKING_START,
    LatencyEventKind.TRACKING_END,
    LatencyEventKind.PREDICTION_START,
    LatencyEventKind.PREDICTION_END,
    LatencyEventKind.ESTIMATOR_UPDATE_START,
    LatencyEventKind.ESTIMATOR_UPDATE_END,
)
_PERCEPTION_FAILURE_STAGES = {
    "gate_detection_missing": (
        (LatencyEventKind.DETECTION_START, LatencyEventKind.DETECTION_END),
        LatencyEventKind.DETECTION_END,
        False,
    ),
    "gate_detection_ambiguous": (
        (LatencyEventKind.DETECTION_START, LatencyEventKind.DETECTION_END),
        LatencyEventKind.DETECTION_END,
        False,
    ),
    "aperture_geometry_unavailable": (
        (
            LatencyEventKind.DETECTION_START,
            LatencyEventKind.DETECTION_END,
            LatencyEventKind.TRACKING_START,
            LatencyEventKind.TRACKING_END,
        ),
        LatencyEventKind.TRACKING_END,
        False,
    ),
    "imu_correlation_unavailable": (
        (
            LatencyEventKind.DETECTION_START,
            LatencyEventKind.DETECTION_END,
            LatencyEventKind.TRACKING_START,
            LatencyEventKind.TRACKING_END,
            LatencyEventKind.PREDICTION_START,
            LatencyEventKind.PREDICTION_END,
        ),
        LatencyEventKind.PREDICTION_END,
        False,
    ),
    "relative_estimator_unavailable": (
        _PERCEPTION_STAGE_KINDS,
        LatencyEventKind.ESTIMATOR_UPDATE_END,
        True,
    ),
    "imu_derotation_unavailable": (
        _PERCEPTION_STAGE_KINDS,
        LatencyEventKind.ESTIMATOR_UPDATE_END,
        True,
    ),
}

# Deterministic ordering for simultaneous facts.  It preserves every lifecycle
# edge required by validate_latency_event_sequence without pretending that
# source clocks establish a finer causal order than the recorded host stamp.
_KIND_ORDER = {
    LatencyEventKind.CAMERA_FIRST_PACKET: 0,
    LatencyEventKind.CAMERA_FINAL_PACKET: 1,
    LatencyEventKind.FRAME_REASSEMBLED: 2,
    LatencyEventKind.DECODE_START: 3,
    LatencyEventKind.DECODE_END: 4,
    LatencyEventKind.FRAME_PUBLISHED: 5,
    LatencyEventKind.GYRO_SAMPLE: 6,
    LatencyEventKind.CONTROL_TICK_DUE: 10,
    LatencyEventKind.DEADLINE_MISSED: 11,
    LatencyEventKind.CONTROL_TICK_SKIPPED: 12,
    LatencyEventKind.CONTROL_TICK_START: 13,
    LatencyEventKind.DETECTION_START: 20,
    LatencyEventKind.DETECTION_END: 21,
    LatencyEventKind.TRACKING_START: 30,
    LatencyEventKind.TRACKING_END: 31,
    LatencyEventKind.PREDICTION_START: 40,
    LatencyEventKind.PREDICTION_END: 41,
    LatencyEventKind.ESTIMATOR_UPDATE_START: 50,
    LatencyEventKind.ESTIMATOR_UPDATE_END: 51,
    LatencyEventKind.FRAME_DROPPED: 55,
    LatencyEventKind.CONTROLLER_START: 60,
    LatencyEventKind.CONTROLLER_END: 61,
    LatencyEventKind.CONTROL_TICK_END: 70,
}


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _exact_nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be an exact finite float")
    if value <= 0.0:
        raise ValueError(f"{label} must be positive")
    return value


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded token")
    return value


@dataclass(frozen=True, slots=True)
class VQ2OfflinePerceptionTiming:
    """Caller-supplied occurrence times for one distinct-frame computation."""

    detection_start_monotonic_ns: int
    detection_end_monotonic_ns: int
    tracking_start_monotonic_ns: int
    tracking_end_monotonic_ns: int
    prediction_start_monotonic_ns: int
    prediction_end_monotonic_ns: int
    estimator_start_monotonic_ns: int
    estimator_end_monotonic_ns: int

    def __post_init__(self) -> None:
        values = tuple(
            _exact_nonnegative_int(getattr(self, name), name)
            for name in self.__dataclass_fields__
        )
        if any(later < earlier for earlier, later in zip(values, values[1:])):
            raise ValueError("perception timing stages must be monotonic")


@dataclass(frozen=True, slots=True)
class VQ2OfflineCoastTiming:
    """Occurrence times for one retained-frame coast attempt."""

    prediction_start_monotonic_ns: int
    prediction_end_monotonic_ns: int
    estimator_start_monotonic_ns: int
    estimator_end_monotonic_ns: int

    def __post_init__(self) -> None:
        values = tuple(
            _exact_nonnegative_int(getattr(self, name), name)
            for name in self.__dataclass_fields__
        )
        if any(later < earlier for earlier, later in zip(values, values[1:])):
            raise ValueError("coast timing stages must be monotonic")


@dataclass(frozen=True, slots=True)
class VQ2OfflineTickTiming:
    """One immutable generated timing plan for a scheduler poll."""

    tick_start_monotonic_ns: int
    controller_start_monotonic_ns: int
    controller_end_monotonic_ns: int
    tick_finish_monotonic_ns: int
    perception: Optional[VQ2OfflinePerceptionTiming]
    coast: Optional[VQ2OfflineCoastTiming] = None

    def __post_init__(self) -> None:
        start = _exact_nonnegative_int(
            self.tick_start_monotonic_ns, "tick_start_monotonic_ns"
        )
        controller_start = _exact_nonnegative_int(
            self.controller_start_monotonic_ns,
            "controller_start_monotonic_ns",
        )
        controller_end = _exact_nonnegative_int(
            self.controller_end_monotonic_ns,
            "controller_end_monotonic_ns",
        )
        finish = _exact_nonnegative_int(
            self.tick_finish_monotonic_ns, "tick_finish_monotonic_ns"
        )
        if type(self.perception) not in {
            VQ2OfflinePerceptionTiming,
            type(None),
        }:
            raise TypeError(
                "perception must be VQ2OfflinePerceptionTiming or None"
            )
        if type(self.coast) not in {VQ2OfflineCoastTiming, type(None)}:
            raise TypeError("coast must be VQ2OfflineCoastTiming or None")
        if self.perception is not None and self.coast is not None:
            raise ValueError("perception and coast timing are mutually exclusive")
        ordered = [start]
        if self.perception is not None:
            ordered.extend(
                getattr(self.perception, name)
                for name in self.perception.__dataclass_fields__
            )
        if self.coast is not None:
            ordered.extend(
                getattr(self.coast, name)
                for name in self.coast.__dataclass_fields__
            )
        ordered.extend((controller_start, controller_end, finish))
        if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
            raise ValueError("tick timing plan must be monotonic")


@dataclass(frozen=True, slots=True)
class VQ2OfflineTickInput:
    """Exact latest-value input for one offline scheduler poll."""

    snapshot: VQ2VisionSnapshot
    safety: VQ2SafetyGuidanceInput
    timing: VQ2OfflineTickTiming

    def __post_init__(self) -> None:
        if type(self.snapshot) is not VQ2VisionSnapshot:
            raise TypeError("snapshot must be exact VQ2VisionSnapshot")
        if type(self.safety) is not VQ2SafetyGuidanceInput:
            raise TypeError("safety must be exact VQ2SafetyGuidanceInput")
        if type(self.timing) is not VQ2OfflineTickTiming:
            raise TypeError("timing must be exact VQ2OfflineTickTiming")


@dataclass(frozen=True, slots=True)
class VQ2Wave3OfflineConfig:
    """Explicit configuration; no calibration or uncertainty is inferred."""

    host_clock_id: str
    camera_stream_id: str
    camera_generation: int
    imu_source: VQ2ImuSource
    camera_calibration: VQ2CameraToBodyCalibration
    derotation_model: VQ2DerotationModel
    capture_orientation_uncertainty_rad: float
    target_orientation_uncertainty_rad: float
    capture_host_time_uncertainty_ns: int
    target_host_time_uncertainty_ns: int
    fallback_center_covariance: FeatureCovarianceV1
    scheduler_start_monotonic_ns: int
    image_width: int = 640
    image_height: int = 360
    detector_min_area: int = 500
    detector_max_area: int = 500_000
    detector_max_aspect_ratio: float = 3.0
    detector_min_confidence: float = 0.10
    measurement_uncertainty_ns: int = 1_000_000
    control_period_ns: int = MINIMUM_CONTROL_PERIOD_NS
    first_control_tick_id: int = 0
    first_proposal_id: int = 0
    imu_history_limit: int = 256
    enable_single_tick_correlated_coast: bool = False
    tracker_id: str = "wave3b-active-gate"
    candidate_id_prefix: str = "wave3b"
    imu_attitude_config: ImuAttitudeConfig = ImuAttitudeConfig()
    aperture_config: VQ2ApertureConfig = VQ2ApertureConfig()
    relative_estimator_config: RelativeEstimatorConfig = RelativeEstimatorConfig()

    def __post_init__(self) -> None:
        _bounded_token(self.host_clock_id, "host_clock_id")
        _bounded_token(self.camera_stream_id, "camera_stream_id")
        _bounded_token(self.tracker_id, "tracker_id")
        _bounded_token(self.candidate_id_prefix, "candidate_id_prefix")
        if len(self.candidate_id_prefix) > 96:
            raise ValueError("candidate_id_prefix is too long for derived identities")
        _exact_nonnegative_int(self.camera_generation, "camera_generation")
        _exact_nonnegative_int(
            self.scheduler_start_monotonic_ns,
            "scheduler_start_monotonic_ns",
        )
        _positive_int(self.image_width, "image_width")
        _positive_int(self.image_height, "image_height")
        _positive_int(self.detector_min_area, "detector_min_area")
        _positive_int(self.detector_max_area, "detector_max_area")
        if self.detector_max_area < self.detector_min_area:
            raise ValueError("detector_max_area cannot be less than detector_min_area")
        _positive_float(
            self.detector_max_aspect_ratio, "detector_max_aspect_ratio"
        )
        _positive_float(self.detector_min_confidence, "detector_min_confidence")
        if self.detector_min_confidence > 1.0:
            raise ValueError("detector_min_confidence must be <= 1")
        _positive_int(self.measurement_uncertainty_ns, "measurement_uncertainty_ns")
        period = _positive_int(self.control_period_ns, "control_period_ns")
        if period < MINIMUM_CONTROL_PERIOD_NS:
            raise ValueError("control_period_ns cannot exceed the reviewed 50 Hz cap")
        if type(self.enable_single_tick_correlated_coast) is not bool:
            raise TypeError(
                "enable_single_tick_correlated_coast must be an exact bool"
            )
        if (
            self.enable_single_tick_correlated_coast
            and period != MINIMUM_CONTROL_PERIOD_NS
        ):
            raise ValueError(
                "single-tick correlated coast requires the reviewed 20 ms period"
            )
        _exact_nonnegative_int(self.first_control_tick_id, "first_control_tick_id")
        _exact_nonnegative_int(self.first_proposal_id, "first_proposal_id")
        if _positive_int(self.imu_history_limit, "imu_history_limit") < 2:
            raise ValueError("imu_history_limit must retain at least two attitudes")
        _positive_float(
            self.capture_orientation_uncertainty_rad,
            "capture_orientation_uncertainty_rad",
        )
        _positive_float(
            self.target_orientation_uncertainty_rad,
            "target_orientation_uncertainty_rad",
        )
        _positive_int(
            self.capture_host_time_uncertainty_ns,
            "capture_host_time_uncertainty_ns",
        )
        _positive_int(
            self.target_host_time_uncertainty_ns,
            "target_host_time_uncertainty_ns",
        )
        for value, expected, label in (
            (self.imu_source, VQ2ImuSource, "imu_source"),
            (
                self.camera_calibration,
                VQ2CameraToBodyCalibration,
                "camera_calibration",
            ),
            (self.derotation_model, VQ2DerotationModel, "derotation_model"),
            (
                self.fallback_center_covariance,
                FeatureCovarianceV1,
                "fallback_center_covariance",
            ),
            (self.imu_attitude_config, ImuAttitudeConfig, "imu_attitude_config"),
            (self.aperture_config, VQ2ApertureConfig, "aperture_config"),
            (
                self.relative_estimator_config,
                RelativeEstimatorConfig,
                "relative_estimator_config",
            ),
        ):
            if type(value) is not expected:
                raise TypeError(f"{label} must be exact {expected.__name__}")
        if self.imu_source.host_clock_id != self.host_clock_id:
            raise ValueError("IMU source and runtime host clocks must match")
        if self.fallback_center_covariance.feature_order != _CENTER_FEATURES:
            raise ValueError(
                "fallback_center_covariance must describe normalized center"
            )


@dataclass(frozen=True, slots=True)
class VQ2OfflineTickResult:
    """One completed/withheld tick and its immutable cumulative trace."""

    control_tick_id: int
    lease: Optional[ControlTickLeaseV1]
    selection: Optional[LatestFrameSelectionV1]
    transition: Optional[VQ2Wave3ImuAdapterTransition]
    perception_ran: bool
    skipped: bool
    reason: Optional[str]
    trace: tuple[LatencyEventV1, ...]
    coast_attempted: bool = False
    coast_timing: Optional[VQ2OfflineCoastTiming] = None
    consumed_coast_lease: Optional[VQ2Wave3CoastLease] = None
    consumed_coast_source_transition: Optional[
        VQ2Wave3ImuAdapterTransition
    ] = None
    coast_lease_disposition: Optional[str] = None

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.control_tick_id, "control_tick_id")
        if type(self.lease) not in {ControlTickLeaseV1, type(None)}:
            raise TypeError("lease must be ControlTickLeaseV1 or None")
        if self.lease is not None:
            lease_frame = self.lease.frame
            if lease_frame is not None:
                lease_frame = replace(lease_frame)
            replace(self.lease, frame=lease_frame)
        if type(self.selection) not in {LatestFrameSelectionV1, type(None)}:
            raise TypeError(
                "selection must be LatestFrameSelectionV1 or None"
            )
        if self.selection is not None:
            selection_timing = replace(
                self.selection.timing,
                identity=replace(self.selection.timing.identity),
            )
            replace(self.selection, timing=selection_timing)
        if type(self.transition) not in {
            VQ2Wave3ImuAdapterTransition,
            type(None),
        }:
            raise TypeError(
                "transition must be VQ2Wave3ImuAdapterTransition or None"
            )
        if self.transition is not None:
            self.transition.validate_integrity()
        if (
            type(self.perception_ran) is not bool
            or type(self.coast_attempted) is not bool
            or type(self.skipped) is not bool
        ):
            raise TypeError(
                "perception_ran, coast_attempted, and skipped must be exact bools"
            )
        if type(self.coast_timing) not in {VQ2OfflineCoastTiming, type(None)}:
            raise TypeError("coast_timing must be VQ2OfflineCoastTiming or None")
        if self.coast_timing is not None:
            replace(self.coast_timing)
        if type(self.consumed_coast_lease) not in {
            VQ2Wave3CoastLease,
            type(None),
        }:
            raise TypeError(
                "consumed_coast_lease must be VQ2Wave3CoastLease or None"
            )
        if self.consumed_coast_lease is not None:
            self.consumed_coast_lease.validate_integrity()
        if type(self.consumed_coast_source_transition) not in {
            VQ2Wave3ImuAdapterTransition,
            type(None),
        }:
            raise TypeError(
                "consumed_coast_source_transition must be "
                "VQ2Wave3ImuAdapterTransition or None"
            )
        if self.consumed_coast_source_transition is not None:
            self.consumed_coast_source_transition.validate_integrity()
        if (self.consumed_coast_lease is None) != (
            self.consumed_coast_source_transition is None
        ):
            raise ValueError(
                "consumed coast lease and source transition are all-or-none"
            )
        if self.coast_lease_disposition is not None and (
            type(self.coast_lease_disposition) is not str
            or self.coast_lease_disposition not in _COAST_LEASE_DISPOSITIONS
        ):
            raise ValueError("coast_lease_disposition is not a reviewed value")
        if (self.consumed_coast_lease is None) != (
            self.coast_lease_disposition is None
        ):
            raise ValueError("consumed coast lease and disposition are all-or-none")
        if (self.coast_timing is not None) != self.coast_attempted:
            raise ValueError("coast attempt and timing must be all-or-none")
        if self.perception_ran and self.coast_attempted:
            raise ValueError("perception and coast work are mutually exclusive")
        if self.reason is not None and (
            type(self.reason) is not str or not self.reason
        ):
            raise TypeError("reason must be a non-empty string or None")
        if type(self.trace) is not tuple:
            raise TypeError("trace must be an exact tuple")
        if any(type(event) is not LatencyEventV1 for event in self.trace):
            raise TypeError("trace must contain exact LatencyEventV1 values")
        reconstructed_trace = tuple(
            replace(
                event,
                frame=(None if event.frame is None else replace(event.frame)),
            )
            for event in self.trace
        )
        validate_latency_event_sequence(reconstructed_trace)
        if not self.trace:
            raise ValueError("offline result trace must be non-empty")
        if any(
            event.control_tick_id is not None
            and event.control_tick_id > self.control_tick_id
            for event in self.trace
        ):
            raise ValueError("offline result trace cannot contain a future tick")
        if len({event.host_clock_id for event in self.trace}) != 1:
            raise ValueError("offline result trace must use one host clock")
        if any(
            event.command_id is not None
            or event.kind in _OFFLINE_FORBIDDEN_TRACE_KINDS
            for event in self.trace
        ):
            raise ValueError(
                "offline result trace cannot carry send/actuator authority"
            )
        if any(event.queue_depth != 0 for event in self.trace):
            raise ValueError("offline result trace requires exact zero queue depth")
        gyro_events = tuple(
            event
            for event in self.trace
            if event.kind is LatencyEventKind.GYRO_SAMPLE
        )
        if any(
            event.frame is not None
            or event.control_tick_id is not None
            or event.outcome is not EventOutcome.OK
            for event in gyro_events
        ):
            raise ValueError("offline GYRO_SAMPLE must remain an occurrence-only fact")
        if self.skipped:
            if self.reason not in _OFFLINE_SKIP_REASONS:
                raise ValueError("skipped result has an unknown offline reason")
            if self.lease is not None or self.selection is not None:
                raise ValueError("skipped tick cannot carry a lease or selection")
            if (
                self.transition is not None
                or self.perception_ran
                or self.coast_attempted
            ):
                raise ValueError("skipped tick cannot claim pipeline work")
            if self.reason is None:
                raise ValueError("skipped tick requires a reason")
        else:
            if self.lease is None or self.transition is None:
                raise ValueError("completed tick requires a lease and transition")
            if self.lease.control_tick_id != self.control_tick_id:
                raise ValueError("result tick identity differs from its lease")
            if self.lease.frame is None:
                raise ValueError("offline runtime lease requires a latest frame")
            if (self.selection is not None) != self.perception_ran:
                raise ValueError(
                    "perception and distinct frame selection must be all-or-none"
                )
            if self.coast_attempted and self.selection is not None:
                raise ValueError("coast attempt cannot carry a distinct selection")
            proposal = self.transition.proposal
            if proposal.control_tick_id != self.control_tick_id:
                raise ValueError("proposal differs from the completed control tick")
            if self.selection is not None:
                if self.lease.frame != self.selection.timing.identity:
                    raise ValueError("lease frame differs from distinct selection")
            else:
                if self.transition.active_update is not None:
                    raise ValueError("repeated frame cannot carry a correlated update")
                if (
                    not self.coast_attempted
                    and (
                        proposal.source_frame is not None
                        or not proposal.is_exact_zero
                    )
                ):
                    raise ValueError(
                        "repeated frame must remain source-less exact zero"
                    )
                if not self.coast_attempted and (
                    self.transition.correlated_coast is not None
                    or self.transition.consumed_coast_lease is not None
                ):
                    raise ValueError(
                        "ordinary repeated frame cannot retain coast evidence"
                    )
                if (
                    self.coast_attempted
                    and self.consumed_coast_lease is None
                ):
                    raise ValueError("coast attempt requires its consumed lease")
            active_update = self.transition.active_update
            if active_update is not None:
                if self.selection is None or (
                    active_update.evidence.observation.frame_timing
                    != self.selection.timing
                ):
                    raise ValueError(
                        "correlated update differs from its distinct frame selection"
                    )
        consumed = self.consumed_coast_lease
        if consumed is not None:
            source_transition = self.consumed_coast_source_transition
            if source_transition is None:
                raise AssertionError("consumed lease lacks its source transition")
            if (
                source_transition.memory.coast_lease != consumed
                or source_transition.active_update != consumed.source_update
                or source_transition.correlated_coast is not None
                or source_transition.proposal != consumed.source_proposal
                or source_transition.memory.inner_memory.guidance_memory.safety
                != consumed.source_safety
            ):
                raise ValueError(
                    "consumed coast lease differs from its source transition"
                )
            if consumed.eligible_control_tick_id != self.control_tick_id:
                raise ValueError("consumed coast lease differs from result tick")
            if self.skipped:
                if self.coast_lease_disposition != "tick_skipped":
                    raise ValueError("skipped tick has the wrong coast disposition")
                current_due = self._one_event(
                    tuple(
                        event
                        for event in self.trace
                        if event.control_tick_id == self.control_tick_id
                    ),
                    LatencyEventKind.CONTROL_TICK_DUE,
                    "current consumed-lease due event",
                )
                if self.reason == "tick_deadline_elapsed":
                    if (
                        current_due.monotonic_ns
                        <= consumed.eligible_deadline_monotonic_ns
                    ):
                        raise ValueError(
                            "deadline-elapsed skip did not occur after lease expiry"
                        )
                elif self.reason == "planned_work_exceeds_deadline" and not (
                    consumed.eligible_due_monotonic_ns
                    <= current_due.monotonic_ns
                    <= consumed.eligible_deadline_monotonic_ns
                ):
                    raise ValueError(
                        "planned skip occurred outside the consumed lease window"
                    )
            elif self.coast_attempted:
                transition = self.transition
                if transition is None:
                    raise AssertionError("coast result lacks its transition")
                if transition.memory.coast_lease is not None:
                    raise ValueError("coast result retained its consumed lease")
                if (
                    transition.correlated_coast is not None
                    and transition.consumed_coast_lease != consumed
                ):
                    raise ValueError("coast transition changed its consumed lease")
                if (
                    transition.correlated_coast is not None
                    and transition.correlated_coast.prior_update
                    != consumed.source_update
                ):
                    raise ValueError("coast result changed its leased source update")
                coast_accepted = bool(
                    transition.correlated_coast is not None
                    and transition.outer_withholding_reason is None
                    and transition.accepted_attitude is not None
                    and transition.controller_attitude_provenance is not None
                )
                expected = "coast_accepted" if coast_accepted else "coast_rejected"
                if self.coast_lease_disposition != expected:
                    raise ValueError("coast result has the wrong lease disposition")
                lease = self.lease
                if lease is None or (
                    lease.due_monotonic_ns != consumed.eligible_due_monotonic_ns
                    or lease.deadline_monotonic_ns
                    != consumed.eligible_deadline_monotonic_ns
                    or lease.frame != consumed.source_proposal.source_frame
                ):
                    raise ValueError("coast result differs from eligible lease identity")
            elif self.selection is not None:
                if self.coast_lease_disposition != "distinct_frame_selected":
                    raise ValueError("distinct frame has the wrong coast disposition")
            else:
                raise ValueError("consumed lease lacks a consuming tick action")
        if self.transition is not None and (
            self.transition.consumed_coast_lease
            != self.consumed_coast_lease
        ):
            if self.coast_attempted:
                raise ValueError("coast transition changed its consumed lease")
            raise ValueError(
                "completed result changed its transition's consumed coast lease"
            )
        if consumed is not None:
            self._validate_consumed_lease_source_trace(
                consumed,
                self.consumed_coast_source_transition,
            )
        self._validate_trace_binding()

    @staticmethod
    def _one_event(
        events: tuple[LatencyEventV1, ...],
        kind: LatencyEventKind,
        label: str,
    ) -> LatencyEventV1:
        matching = tuple(event for event in events if event.kind is kind)
        if len(matching) != 1:
            raise ValueError(f"result trace requires exactly one {label}")
        return matching[0]

    def _validate_trace_binding(self) -> None:
        tick_events = tuple(
            event
            for event in self.trace
            if event.control_tick_id == self.control_tick_id
        )
        due = self._one_event(
            tick_events,
            LatencyEventKind.CONTROL_TICK_DUE,
            "current control-tick due event",
        )
        if due.frame is None:
            raise ValueError("offline result due event requires its latest frame")

        if self.skipped:
            if any(event.kind not in _SKIPPED_TICK_KINDS for event in tick_events):
                raise ValueError("skipped result trace carries completed tick work")
            skipped = self._one_event(
                tick_events,
                LatencyEventKind.CONTROL_TICK_SKIPPED,
                "current skipped-tick event",
            )
            if any(
                event.monotonic_ns > skipped.monotonic_ns
                for event in self.trace
            ):
                raise ValueError(
                    "skipped result trace contains facts after its terminal event"
                )
            if (
                due.outcome is not EventOutcome.OK
                or skipped.host_clock_id != due.host_clock_id
                or skipped.frame != due.frame
                or skipped.monotonic_ns != due.monotonic_ns
                or skipped.reason_code != self.reason
            ):
                raise ValueError("skipped result differs from its trace lifecycle")
            deadline_events = tuple(
                event
                for event in tick_events
                if event.kind is LatencyEventKind.DEADLINE_MISSED
            )
            expects_deadline_miss = self.reason == "tick_deadline_elapsed"
            if len(deadline_events) != int(expects_deadline_miss):
                raise ValueError("skipped result deadline evidence is inconsistent")
            if deadline_events:
                deadline = deadline_events[0]
                if (
                    deadline.host_clock_id != due.host_clock_id
                    or deadline.frame != due.frame
                    or deadline.monotonic_ns != due.monotonic_ns
                    or deadline.reason_code != self.reason
                ):
                    raise ValueError("deadline miss differs from its skipped tick")
            return

        lease = self.lease
        transition = self.transition
        if lease is None or transition is None:
            raise AssertionError("completed result structure was not prevalidated")
        if any(event.kind not in _COMPLETED_TICK_KINDS for event in tick_events):
            raise ValueError("completed result trace carries an invalid tick lifecycle")
        start = self._one_event(
            tick_events,
            LatencyEventKind.CONTROL_TICK_START,
            "current control-tick start event",
        )
        controller_start = self._one_event(
            tick_events,
            LatencyEventKind.CONTROLLER_START,
            "current controller start event",
        )
        controller_end = self._one_event(
            tick_events,
            LatencyEventKind.CONTROLLER_END,
            "current controller end event",
        )
        tick_end = self._one_event(
            tick_events,
            LatencyEventKind.CONTROL_TICK_END,
            "current control-tick end event",
        )
        if any(
            event.monotonic_ns > tick_end.monotonic_ns
            for event in self.trace
        ):
            raise ValueError(
                "completed result trace contains facts after its tick end"
            )
        core = (due, start, controller_start, controller_end, tick_end)
        proposal = transition.proposal
        if any(
            event.host_clock_id != proposal.host_clock_id
            or event.frame != lease.frame
            or event.outcome is not EventOutcome.OK
            for event in core
        ):
            raise ValueError("completed result differs from its trace correlations")
        if (
            due.monotonic_ns != lease.start_monotonic_ns
            or start.monotonic_ns != lease.start_monotonic_ns
        ):
            raise ValueError("completed result trace differs from its lease start")
        if (
            proposal.control_tick_deadline_monotonic_ns
            != lease.deadline_monotonic_ns
            or proposal.proposal_monotonic_ns != controller_end.monotonic_ns
        ):
            raise ValueError("completed proposal differs from its tick trace")
        if not (
            lease.start_monotonic_ns
            <= controller_start.monotonic_ns
            <= controller_end.monotonic_ns
            <= tick_end.monotonic_ns
            <= lease.deadline_monotonic_ns
        ):
            raise ValueError("completed result trace exceeds its lease window")

        selection = self.selection
        if selection is None:
            if self.coast_attempted:
                self._validate_coast_trace(lease.frame)
            else:
                if any(
                    event.kind
                    in {
                        LatencyEventKind.PREDICTION_START,
                        LatencyEventKind.PREDICTION_END,
                        LatencyEventKind.ESTIMATOR_UPDATE_START,
                        LatencyEventKind.ESTIMATOR_UPDATE_END,
                    }
                    for event in tick_events
                ):
                    raise ValueError(
                        "ordinary repeated trace cannot carry coast stages"
                    )
                self._validate_transition_reason(transition)
            return
        frame = selection.timing.identity
        for kind, field in _CAMERA_STAGES:
            matching = tuple(
                event
                for event in self.trace
                if event.kind is kind and event.frame == frame
            )
            if len(matching) != 1:
                raise ValueError(
                    f"distinct result trace requires exactly one {kind.value} event"
                )
            event = matching[0]
            if (
                event.host_clock_id != selection.timing.host_clock_id
                or event.control_tick_id is not None
                or event.monotonic_ns != getattr(selection.timing, field)
                or event.outcome is not EventOutcome.OK
            ):
                raise ValueError("camera trace differs from the distinct selection")
        active_update = transition.active_update
        if active_update is None:
            self._validate_failed_perception_trace(frame)
            return
        stages: dict[LatencyEventKind, LatencyEventV1] = {}
        for kind, expected_tick in (
            (LatencyEventKind.DETECTION_START, None),
            (LatencyEventKind.DETECTION_END, None),
            (LatencyEventKind.TRACKING_START, None),
            (LatencyEventKind.TRACKING_END, None),
            (LatencyEventKind.PREDICTION_START, self.control_tick_id),
            (LatencyEventKind.PREDICTION_END, self.control_tick_id),
            (LatencyEventKind.ESTIMATOR_UPDATE_START, None),
            (LatencyEventKind.ESTIMATOR_UPDATE_END, None),
        ):
            stages[kind] = self._require_frame_stage(
                kind,
                frame,
                control_tick_id=expected_tick,
            )
        if any(
            event.kind is LatencyEventKind.FRAME_DROPPED
            and event.frame == frame
            for event in self.trace
        ):
            raise ValueError("active correlated update cannot carry a dropped frame")
        self._validate_transition_reason(transition)
        evidence = active_update.evidence
        prediction_target = evidence.prediction_target
        if (
            prediction_target.decision_time_monotonic_ns
            != prediction_target.prediction_time_monotonic_ns
            or stages[LatencyEventKind.PREDICTION_END].monotonic_ns
            != prediction_target.prediction_time_monotonic_ns
        ):
            raise ValueError("prediction trace differs from its at-decision target")
        for attitude_input in (
            evidence.capture_attitude,
            evidence.target_attitude,
        ):
            attitude = attitude_input.attitude
            matching = tuple(
                event
                for event in self.trace
                if event.kind is LatencyEventKind.GYRO_SAMPLE
                and event.sensor_sample_id == attitude.sample_sequence
                and event.sensor_source_time_ns == attitude.source_time_us * 1_000
                and event.monotonic_ns == attitude.receive_monotonic_ns
            )
            if len(matching) != 1 or (
                matching[0].host_clock_id != attitude.source.host_clock_id
                or matching[0].frame is not None
                or matching[0].control_tick_id is not None
                or matching[0].outcome is not EventOutcome.OK
            ):
                raise ValueError("correlated attitude lacks its exact IMU trace fact")

    def _validate_consumed_lease_source_trace(
        self,
        consumed: VQ2Wave3CoastLease,
        source_transition: Optional[VQ2Wave3ImuAdapterTransition],
    ) -> None:
        """Bind every consumed lease to its independently retained source tick."""

        if source_transition is None:
            raise AssertionError("consumed lease trace lacks its source transition")
        frame = consumed.source_proposal.source_frame
        if frame is None:
            raise AssertionError("validated consumed lease lacks a source frame")
        source_tick_id = consumed.source_control_tick_id
        source_tick_events = tuple(
            event
            for event in self.trace
            if event.control_tick_id == source_tick_id
        )
        source_tick_kinds = (
            LatencyEventKind.CONTROL_TICK_DUE,
            LatencyEventKind.CONTROL_TICK_START,
            LatencyEventKind.PREDICTION_START,
            LatencyEventKind.PREDICTION_END,
            LatencyEventKind.CONTROLLER_START,
            LatencyEventKind.CONTROLLER_END,
            LatencyEventKind.CONTROL_TICK_END,
        )
        if (
            len(source_tick_events) != len(source_tick_kinds)
            or any(
                sum(event.kind is kind for event in source_tick_events) != 1
                for kind in source_tick_kinds
            )
        ):
            raise ValueError(
                "coast trace lacks the exact leased source-tick lifecycle"
            )
        source_stages = {
            kind: next(
                event for event in source_tick_events if event.kind is kind
            )
            for kind in source_tick_kinds
        }
        source_proposal = consumed.source_proposal
        source_due_ns = (
            consumed.source_control_tick_deadline_monotonic_ns
            - MINIMUM_CONTROL_PERIOD_NS
        )
        source_core = tuple(source_stages.values())
        if any(
            event.frame != frame
            or event.host_clock_id != source_proposal.host_clock_id
            or event.outcome is not EventOutcome.OK
            or event.reason_code is not None
            for event in source_core
        ):
            raise ValueError(
                "coast trace changed its leased source-tick correlations"
            )
        source_due = source_stages[LatencyEventKind.CONTROL_TICK_DUE]
        source_start = source_stages[LatencyEventKind.CONTROL_TICK_START]
        source_prediction_start = source_stages[
            LatencyEventKind.PREDICTION_START
        ]
        source_prediction_end = source_stages[LatencyEventKind.PREDICTION_END]
        source_controller_start = source_stages[
            LatencyEventKind.CONTROLLER_START
        ]
        source_controller_end = source_stages[LatencyEventKind.CONTROLLER_END]
        source_end = source_stages[LatencyEventKind.CONTROL_TICK_END]
        if (
            source_due.monotonic_ns != source_due_ns
            or source_start.monotonic_ns != source_due_ns
            or not (
                source_start.monotonic_ns
                <= source_prediction_start.monotonic_ns
                <= source_prediction_end.monotonic_ns
                <= source_controller_start.monotonic_ns
                <= source_controller_end.monotonic_ns
                <= source_end.monotonic_ns
                <= consumed.source_control_tick_deadline_monotonic_ns
            )
            or source_controller_end.monotonic_ns
            != source_proposal.proposal_monotonic_ns
        ):
            raise ValueError(
                "coast trace changed its leased source-tick timing"
            )
        source_prediction_target = (
            consumed.source_update.evidence.prediction_target
        )
        if (
            source_stages[LatencyEventKind.PREDICTION_END].monotonic_ns
            != source_prediction_target.prediction_time_monotonic_ns
        ):
            raise ValueError(
                "coast trace changed its leased source prediction target"
            )
        prior_estimator_stages = {
            kind: tuple(
                event
                for event in self.trace
                if event.kind is kind
                and event.frame == frame
                and event.control_tick_id is None
            )
            for kind in (
                LatencyEventKind.ESTIMATOR_UPDATE_START,
                LatencyEventKind.ESTIMATOR_UPDATE_END,
            )
        }
        if any(len(events) != 1 for events in prior_estimator_stages.values()) or any(
            event.outcome is not EventOutcome.OK or event.reason_code is not None
            for events in prior_estimator_stages.values()
            for event in events
        ):
            raise ValueError(
                "coast trace lacks the exact leased source estimator lifecycle"
            )
        prediction_ticks = {source_tick_id}
        estimator_ticks = {None}
        if self.coast_attempted:
            prediction_ticks.add(self.control_tick_id)
            estimator_ticks.add(self.control_tick_id)
        expected_work_correlations = {
            LatencyEventKind.PREDICTION_START: prediction_ticks,
            LatencyEventKind.PREDICTION_END: prediction_ticks,
            LatencyEventKind.ESTIMATOR_UPDATE_START: estimator_ticks,
            LatencyEventKind.ESTIMATOR_UPDATE_END: estimator_ticks,
        }
        for kind, expected_ticks in expected_work_correlations.items():
            matching = tuple(
                event
                for event in self.trace
                if event.kind is kind and event.frame == frame
            )
            if (
                len(matching) != len(expected_ticks)
                or {event.control_tick_id for event in matching}
                != expected_ticks
            ):
                raise ValueError(
                    "coast trace changed its source/current work lifecycles"
                )
        source_timing = (
            consumed.source_update.evidence.observation.frame_timing
        )
        for kind, field in _CAMERA_STAGES:
            matching = tuple(
                event
                for event in self.trace
                if event.kind is kind and event.frame == frame
            )
            if len(matching) != 1 or (
                matching[0].control_tick_id is not None
                or matching[0].monotonic_ns != getattr(source_timing, field)
                or matching[0].outcome is not EventOutcome.OK
            ):
                raise ValueError(
                    "coast trace changed its retained source-frame camera facts"
                )
        for kind in (
            LatencyEventKind.DETECTION_START,
            LatencyEventKind.DETECTION_END,
            LatencyEventKind.TRACKING_START,
            LatencyEventKind.TRACKING_END,
        ):
            matching = tuple(
                event
                for event in self.trace
                if event.kind is kind and event.frame == frame
            )
            if len(matching) != 1 or (
                matching[0].control_tick_id is not None
                or matching[0].outcome is not EventOutcome.OK
            ):
                raise ValueError(
                    "coast trace changed its retained source perception facts"
                )
        if any(
            event.kind is LatencyEventKind.FRAME_DROPPED
            and event.frame == frame
            for event in self.trace
        ):
            raise ValueError("coast trace relabeled its accepted source as dropped")
        for attitude_input in (
            consumed.source_update.evidence.capture_attitude,
            consumed.source_update.evidence.target_attitude,
        ):
            attitude = attitude_input.attitude
            matching = tuple(
                event
                for event in self.trace
                if event.kind is LatencyEventKind.GYRO_SAMPLE
                and event.sensor_sample_id == attitude.sample_sequence
                and event.sensor_source_time_ns == attitude.source_time_us * 1_000
                and event.monotonic_ns == attitude.receive_monotonic_ns
            )
            if len(matching) != 1 or (
                matching[0].host_clock_id != attitude.source.host_clock_id
                or matching[0].frame is not None
                or matching[0].control_tick_id is not None
                or matching[0].outcome is not EventOutcome.OK
            ):
                raise ValueError(
                    "consumed lease source attitude lacks its exact IMU trace fact"
                )

    def _validate_coast_trace(self, frame: FrameIdentityV1) -> None:
        timing = self.coast_timing
        transition = self.transition
        if timing is None or transition is None:
            raise AssertionError("coast trace lacks its validated result structure")
        consumed = self.consumed_coast_lease
        if consumed is None:
            raise AssertionError("coast trace lacks its consumed source lease")
        if frame != consumed.source_proposal.source_frame:
            raise ValueError("coast trace changed its consumed source frame")
        current = tuple(
            event
            for event in self.trace
            if event.control_tick_id == self.control_tick_id
        )
        if any(
            event.kind
            in {
                LatencyEventKind.CAMERA_FIRST_PACKET,
                LatencyEventKind.CAMERA_FINAL_PACKET,
                LatencyEventKind.FRAME_REASSEMBLED,
                LatencyEventKind.DECODE_START,
                LatencyEventKind.DECODE_END,
                LatencyEventKind.FRAME_PUBLISHED,
                LatencyEventKind.DETECTION_START,
                LatencyEventKind.DETECTION_END,
                LatencyEventKind.TRACKING_START,
                LatencyEventKind.TRACKING_END,
                LatencyEventKind.FRAME_DROPPED,
            }
            for event in current
        ):
            raise ValueError("coast trace re-emitted retained-frame perception facts")
        stages = {
            kind: self._one_event(current, kind, kind.value)
            for kind in (
                LatencyEventKind.PREDICTION_START,
                LatencyEventKind.PREDICTION_END,
                LatencyEventKind.ESTIMATOR_UPDATE_START,
                LatencyEventKind.ESTIMATOR_UPDATE_END,
            )
        }
        expected_times = {
            LatencyEventKind.PREDICTION_START: (
                timing.prediction_start_monotonic_ns
            ),
            LatencyEventKind.PREDICTION_END: timing.prediction_end_monotonic_ns,
            LatencyEventKind.ESTIMATOR_UPDATE_START: (
                timing.estimator_start_monotonic_ns
            ),
            LatencyEventKind.ESTIMATOR_UPDATE_END: (
                timing.estimator_end_monotonic_ns
            ),
        }
        if any(
            event.frame != frame
            or event.monotonic_ns != expected_times[kind]
            for kind, event in stages.items()
        ):
            raise ValueError("coast stages differ from their timing plan")
        coast = transition.correlated_coast
        if coast is None:
            if self.reason != "imu_correlated_coast_unavailable":
                raise ValueError("failed coast has an unknown reason")
            if any(
                stages[kind].outcome is not EventOutcome.OK
                or stages[kind].reason_code is not None
                for kind in (
                    LatencyEventKind.PREDICTION_START,
                    LatencyEventKind.PREDICTION_END,
                    LatencyEventKind.ESTIMATOR_UPDATE_START,
                )
            ):
                raise ValueError("failed coast has invalid preterminal stages")
            terminal = stages[LatencyEventKind.ESTIMATOR_UPDATE_END]
            if (
                terminal.outcome is not EventOutcome.ERROR
                or terminal.reason_code != self.reason
            ):
                raise ValueError("failed coast lacks its terminal error evidence")
            self._validate_transition_reason(transition)
            return

        if any(
            event.outcome is not EventOutcome.OK or event.reason_code is not None
            for event in stages.values()
        ):
            raise ValueError("completed coast requires exact-OK stage evidence")
        self._validate_transition_reason(transition)
        target = coast.evidence.prediction_target
        if (
            target.decision_time_monotonic_ns
            != target.prediction_time_monotonic_ns
            or target.prediction_time_monotonic_ns
            != stages[LatencyEventKind.PREDICTION_END].monotonic_ns
        ):
            raise ValueError("coast prediction trace differs from its target")
        for attitude_input in (
            coast.evidence.capture_attitude,
            coast.evidence.target_attitude,
        ):
            attitude = attitude_input.attitude
            matching = tuple(
                event
                for event in self.trace
                if event.kind is LatencyEventKind.GYRO_SAMPLE
                and event.sensor_sample_id == attitude.sample_sequence
                and event.sensor_source_time_ns == attitude.source_time_us * 1_000
                and event.monotonic_ns == attitude.receive_monotonic_ns
            )
            if len(matching) != 1:
                raise ValueError("coast attitude lacks its exact IMU trace fact")

    def _validate_transition_reason(
        self,
        transition: VQ2Wave3ImuAdapterTransition,
    ) -> None:
        expected = transition.outer_withholding_reason
        if expected is None and transition.proposal.is_exact_zero:
            expected = transition.proposal.reason
        if self.reason != expected:
            raise ValueError("completed result reason differs from its transition")

    def _validate_failed_perception_trace(self, frame: FrameIdentityV1) -> None:
        failure = _PERCEPTION_FAILURE_STAGES.get(self.reason)
        if failure is None:
            raise ValueError("failed perception result has an unknown reason")
        expected_kinds, terminal_kind, estimator_error = failure
        expected = frozenset(expected_kinds)
        actual = tuple(
            event
            for event in self.trace
            if event.frame == frame and event.kind in _PERCEPTION_STAGE_KINDS
        )
        if {event.kind for event in actual} != expected or len(actual) != len(expected):
            raise ValueError("failed perception trace has the wrong staged lifecycle")
        terminal: Optional[LatencyEventV1] = None
        for kind in expected_kinds:
            expected_tick = (
                self.control_tick_id
                if kind
                in {
                    LatencyEventKind.PREDICTION_START,
                    LatencyEventKind.PREDICTION_END,
                }
                else None
            )
            expected_outcome = (
                EventOutcome.ERROR
                if estimator_error
                and kind is LatencyEventKind.ESTIMATOR_UPDATE_END
                else EventOutcome.OK
            )
            expected_reason = (
                self.reason
                if expected_outcome is EventOutcome.ERROR
                else None
            )
            event = self._require_frame_stage(
                kind,
                frame,
                control_tick_id=expected_tick,
                outcome=expected_outcome,
                reason_code=expected_reason,
            )
            if kind is terminal_kind:
                terminal = event
        if terminal is None:
            raise AssertionError("failed perception terminal stage was not selected")
        dropped = tuple(
            event
            for event in self.trace
            if event.kind is LatencyEventKind.FRAME_DROPPED
            and event.frame == frame
        )
        if len(dropped) != 1:
            raise ValueError("failed perception trace requires one dropped-frame event")
        drop = dropped[0]
        if (
            drop.host_clock_id != terminal.host_clock_id
            or drop.control_tick_id is not None
            or drop.monotonic_ns != terminal.monotonic_ns
            or drop.outcome is not EventOutcome.DROPPED
            or drop.reason_code != self.reason
        ):
            raise ValueError("dropped-frame evidence differs from perception failure")

    def _require_frame_stage(
        self,
        kind: LatencyEventKind,
        frame: FrameIdentityV1,
        *,
        control_tick_id: Optional[int],
        outcome: EventOutcome = EventOutcome.OK,
        reason_code: Optional[str] = None,
    ) -> LatencyEventV1:
        matching = tuple(
            event
            for event in self.trace
            if event.kind is kind and event.frame == frame
        )
        if len(matching) != 1 or (
            matching[0].control_tick_id != control_tick_id
            or matching[0].outcome is not outcome
            or matching[0].reason_code != reason_code
        ):
            raise ValueError(
                f"distinct result trace requires one correlated {kind.value} event"
            )
        return matching[0]

    @property
    def proposal(self):
        """Return quarantined intent, or ``None`` for a skipped tick."""

        return None if self.transition is None else self.transition.proposal


class VQ2Wave3OfflineRuntime:
    """State-owning, I/O-free generated scheduler composition."""

    def __init__(self, config: VQ2Wave3OfflineConfig) -> None:
        if type(config) is not VQ2Wave3OfflineConfig:
            raise TypeError("config must be exact VQ2Wave3OfflineConfig")
        self.config = config
        self._cursor = LatestFrameCursorV1(
            expected_host_clock_id=config.host_clock_id,
            expected_stream_id=config.camera_stream_id,
        )
        self._scheduler = FixedRateControlSchedulerV1(
            start_monotonic_ns=config.scheduler_start_monotonic_ns,
            host_clock_id=config.host_clock_id,
            period_ns=config.control_period_ns,
            first_control_tick_id=config.first_control_tick_id,
        )
        self._imu = VQ2ImuProvenanceEstimator(
            config.imu_source,
            config=config.imu_attitude_config,
        )
        self._attitudes: tuple[VQ2TimestampedAttitude, ...] = ()
        self._estimator = VQ2RelativeGateEstimator(
            config.tracker_id,
            config=config.relative_estimator_config,
        )
        self._detector = VQ2GateDetector(
            image_width=config.image_width,
            image_height=config.image_height,
            min_area=config.detector_min_area,
            max_area=config.detector_max_area,
            max_aspect_ratio=config.detector_max_aspect_ratio,
            min_confidence=config.detector_min_confidence,
        )
        self._adapter_memory: Optional[VQ2Wave3ImuAdapterMemory] = None
        self._facts: tuple[LatencyEventV1, ...] = ()
        self._next_fact_sequence = 0
        self._next_proposal_id = config.first_proposal_id
        self._last_result: Optional[VQ2OfflineTickResult] = None

    @property
    def next_due_monotonic_ns(self) -> int:
        return self._scheduler.next_due_monotonic_ns

    @property
    def next_control_tick_id(self) -> int:
        return self._scheduler.next_control_tick_id

    @property
    def next_proposal_id(self) -> int:
        return self._next_proposal_id

    @property
    def adapter_memory(self) -> Optional[VQ2Wave3ImuAdapterMemory]:
        return self._adapter_memory

    @property
    def last_attitude(self) -> Optional[VQ2TimestampedAttitude]:
        return None if not self._attitudes else self._attitudes[-1]

    @property
    def attitude_history(self) -> tuple[VQ2TimestampedAttitude, ...]:
        return self._attitudes

    @property
    def last_result(self) -> Optional[VQ2OfflineTickResult]:
        return self._last_result

    @property
    def trace(self) -> tuple[LatencyEventV1, ...]:
        return self._assemble_trace((), ())

    @property
    def processed_frame_timing(self) -> Optional[FrameTimingV1]:
        return self._cursor.previous_timing

    def _source_transition_for_pending_lease(
        self,
        lease: Optional[VQ2Wave3CoastLease],
    ) -> Optional[VQ2Wave3ImuAdapterTransition]:
        if lease is None:
            return None
        prior = self._last_result
        if (
            prior is None
            or prior.transition is None
            or prior.transition.memory.coast_lease != lease
        ):
            raise AssertionError(
                "pending coast lease lacks its exact prior source transition"
            )
        return prior.transition

    def ingest_imu(
        self,
        sample: VQ2TimedImuSample,
    ) -> Optional[VQ2TimestampedAttitude]:
        """Transactionally ingest one occurrence; it grants no command causality."""

        if type(sample) is not VQ2TimedImuSample:
            raise TypeError("sample must be exact VQ2TimedImuSample")
        # The provenance wrapper clones its mutable estimator before committing.
        # Build the trace fact first so no post-update validation can fail.
        fact = self._event(
            LatencyEventKind.GYRO_SAMPLE,
            sample.receive_monotonic_ns,
            sensor_sample_id=sample.sample_sequence,
            sensor_source_time_ns=sample.source_time_us * 1_000,
        )
        self._assemble_trace((fact,), ())
        attitude = self._imu.update(sample)
        facts = (*self._facts, fact)
        attitudes = self._attitudes
        if attitude is not None:
            attitudes = (*attitudes, attitude)[-self.config.imu_history_limit :]
        self._facts = facts
        self._next_fact_sequence += 1
        self._attitudes = attitudes
        return attitude

    def step(
        self,
        tick_input: VQ2OfflineTickInput,
    ) -> Optional[VQ2OfflineTickResult]:
        """Poll one scheduler instant and return no authority beyond proposal."""

        if type(tick_input) is not VQ2OfflineTickInput:
            raise TypeError("tick_input must be exact VQ2OfflineTickInput")
        snapshot = tick_input.snapshot
        safety = tick_input.safety
        timing = tick_input.timing
        frame = self._validate_step_context(snapshot, safety, timing)
        now = timing.tick_start_monotonic_ns
        due = self._scheduler.next_due_monotonic_ns
        if now < due:
            return None

        tick_id = self._scheduler.next_control_tick_id
        deadline = due + self._scheduler.period_ns
        if now > deadline:
            return self._commit_skipped_tick(
                tick_id=tick_id,
                now_monotonic_ns=now,
                frame=frame,
                reason="tick_deadline_elapsed",
                deadline_missed=True,
            )
        if timing.tick_finish_monotonic_ns > deadline:
            return self._commit_skipped_tick(
                tick_id=tick_id,
                now_monotonic_ns=now,
                frame=frame,
                reason="planned_work_exceeds_deadline",
                deadline_missed=False,
            )

        candidate_cursor = copy.deepcopy(self._cursor)
        selection = candidate_cursor.select(snapshot)
        pending_coast_lease = (
            None
            if self._adapter_memory is None
            else self._adapter_memory.coast_lease
        )
        pending_coast_source_transition = (
            self._source_transition_for_pending_lease(pending_coast_lease)
        )
        coast_eligible = bool(
            self.config.enable_single_tick_correlated_coast
            and selection is None
            and pending_coast_lease is not None
            and tick_id == pending_coast_lease.eligible_control_tick_id
            and due == pending_coast_lease.eligible_due_monotonic_ns
            and deadline == pending_coast_lease.eligible_deadline_monotonic_ns
            and frame == pending_coast_lease.source_proposal.source_frame
        )
        if selection is not None:
            if timing.perception is None or timing.coast is not None:
                raise ValueError(
                    "distinct frame requires only perception timing"
                )
        elif timing.perception is not None or (
            (timing.coast is not None) != coast_eligible
        ):
            raise ValueError(
                "retained frame requires coast timing exactly when lease-eligible"
            )

        expected_tracker_id = self._tracker_id_for_safety(safety)
        tracker_scope_changed = self._estimator.tracker_id != expected_tracker_id
        candidate_estimator = (
            copy.deepcopy(self._estimator)
            if not tracker_scope_changed
            else VQ2RelativeGateEstimator(
                expected_tracker_id,
                config=self.config.relative_estimator_config,
            )
        )
        candidate_facts: list[LatencyEventV1] = []
        active_update: Optional[VQ2ImuCorrelatedEstimatorUpdate] = None
        correlated_coast: Optional[VQ2ImuCorrelatedEstimatorCoast] = None
        perception_reason: Optional[str] = None
        coast_reason: Optional[str] = None
        if selection is not None:
            (
                active_update,
                perception_reason,
                pipeline_facts,
            ) = self._process_distinct_frame(
                selection,
                safety,
                timing,
                candidate_estimator,
                tick_id,
            )
            candidate_facts.extend(pipeline_facts)
        elif coast_eligible:
            assert pending_coast_lease is not None
            (
                correlated_coast,
                coast_reason,
                coast_facts,
            ) = self._process_correlated_coast(
                pending_coast_lease,
                safety,
                timing,
                candidate_estimator,
                tick_id,
            )
            candidate_facts.extend(coast_facts)

        controller_tick = self._controller_tick(
            safety,
            timing,
            tick_id=tick_id,
            deadline_monotonic_ns=deadline,
            active_state=(
                active_update.state
                if active_update is not None
                else (
                    None if correlated_coast is None else correlated_coast.state
                )
            ),
        )
        candidate_facts.append(
            self._event(
                LatencyEventKind.CONTROLLER_START,
                timing.controller_start_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=len(candidate_facts),
            )
        )
        transition = step_vq2_wave3_imu_adapter(
            self._adapter_memory,
            safety,
            active_update=active_update,
            correlated_coast=correlated_coast,
            tick=controller_tick,
            enable_correlated_coast=(
                self.config.enable_single_tick_correlated_coast
                and (active_update is None or now == due)
            ),
        )
        if coast_eligible and correlated_coast is None:
            if coast_reason != "imu_correlated_coast_unavailable":
                raise AssertionError("failed coast lacks its stable reason")
            transition = replace(
                transition,
                outer_withholding_reason=coast_reason,
            )
        candidate_facts.append(
            self._event(
                LatencyEventKind.CONTROLLER_END,
                timing.controller_end_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=len(candidate_facts),
            )
        )

        preview_scheduler = self._scheduler_preview(
            tick_id=tick_id,
            due_monotonic_ns=now,
            finish_monotonic_ns=timing.tick_finish_monotonic_ns,
            frame=frame,
        )
        preview_trace = self._assemble_trace(
            tuple(candidate_facts), preview_scheduler
        )
        preview_lease = ControlTickLeaseV1(
            control_tick_id=tick_id,
            due_monotonic_ns=due,
            deadline_monotonic_ns=deadline,
            start_monotonic_ns=now,
            frame=frame,
        )
        reason = (
            perception_reason
            or coast_reason
            or transition.outer_withholding_reason
        )
        if reason is None and transition.proposal.is_exact_zero:
            reason = transition.proposal.reason
        result = VQ2OfflineTickResult(
            control_tick_id=tick_id,
            lease=preview_lease,
            selection=selection,
            transition=transition,
            perception_ran=selection is not None,
            skipped=False,
            reason=reason,
            trace=preview_trace,
            coast_attempted=coast_eligible,
            coast_timing=timing.coast if coast_eligible else None,
            consumed_coast_lease=(
                pending_coast_lease
                if selection is not None or coast_eligible
                else None
            ),
            consumed_coast_source_transition=(
                pending_coast_source_transition
                if selection is not None or coast_eligible
                else None
            ),
            coast_lease_disposition=(
                (
                    "distinct_frame_selected"
                    if selection is not None
                    else (
                        "coast_accepted"
                        if (
                            correlated_coast is not None
                            and transition.outer_withholding_reason is None
                            and transition.accepted_attitude is not None
                            and transition.controller_attitude_provenance is not None
                        )
                        else "coast_rejected"
                    )
                )
                if pending_coast_lease is not None
                and (selection is not None or coast_eligible)
                else None
            ),
        )

        lease = self._scheduler.begin_due(now, frame=frame, queue_depth=0)
        if lease is None:
            raise AssertionError("prevalidated on-time scheduler tick was skipped")
        if (
            lease.control_tick_id != tick_id
            or lease.due_monotonic_ns != due
            or lease.deadline_monotonic_ns != deadline
            or lease != preview_lease
        ):
            raise AssertionError("scheduler lease differs from prevalidated identity")
        self._scheduler.finish(
            lease,
            timing.tick_finish_monotonic_ns,
            queue_depth=0,
        )

        self._cursor = candidate_cursor
        coast_accepted = bool(
            coast_eligible
            and correlated_coast is not None
            and transition.outer_withholding_reason is None
            and transition.accepted_attitude is not None
            and transition.controller_attitude_provenance is not None
        )
        tracker_transition_accepted = bool(
            transition.memory.inner_memory.guidance_memory.safety == safety
        )
        distinct_estimator_commit_allowed = bool(
            selection is not None
            and (not tracker_scope_changed or tracker_transition_accepted)
        )
        if distinct_estimator_commit_allowed or coast_accepted:
            self._estimator = candidate_estimator
        self._adapter_memory = transition.memory
        self._facts = (*self._facts, *candidate_facts)
        self._next_fact_sequence += len(candidate_facts)
        self._next_proposal_id += 1
        actual_trace = self.trace
        if actual_trace != preview_trace:
            raise AssertionError("scheduler trace differs from prevalidated trace")
        self._last_result = result
        return result

    def _commit_skipped_tick(
        self,
        *,
        tick_id: int,
        now_monotonic_ns: int,
        frame: FrameIdentityV1,
        reason: str,
        deadline_missed: bool,
    ) -> VQ2OfflineTickResult:
        consumed_coast_lease = (
            None
            if self._adapter_memory is None
            else self._adapter_memory.coast_lease
        )
        consumed_coast_source_transition = (
            self._source_transition_for_pending_lease(consumed_coast_lease)
        )
        scheduler_facts = [
            self._event(
                LatencyEventKind.CONTROL_TICK_DUE,
                now_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=10_000,
            )
        ]
        if deadline_missed:
            scheduler_facts.append(
                self._event(
                    LatencyEventKind.DEADLINE_MISSED,
                    now_monotonic_ns,
                    frame=frame,
                    control_tick_id=tick_id,
                    outcome=EventOutcome.SKIPPED,
                    reason_code=reason,
                    sequence_offset=10_001,
                )
            )
        scheduler_facts.append(
            self._event(
                LatencyEventKind.CONTROL_TICK_SKIPPED,
                now_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                outcome=EventOutcome.SKIPPED,
                reason_code=reason,
                sequence_offset=10_002,
            )
        )
        preview_trace = self._assemble_trace((), tuple(scheduler_facts))
        result = VQ2OfflineTickResult(
            control_tick_id=tick_id,
            lease=None,
            selection=None,
            transition=None,
            perception_ran=False,
            skipped=True,
            reason=reason,
            trace=preview_trace,
            coast_attempted=False,
            coast_timing=None,
            consumed_coast_lease=consumed_coast_lease,
            consumed_coast_source_transition=(
                consumed_coast_source_transition
            ),
            coast_lease_disposition=(
                "tick_skipped" if consumed_coast_lease is not None else None
            ),
        )

        if deadline_missed:
            lease = self._scheduler.begin_due(
                now_monotonic_ns,
                frame=frame,
                queue_depth=0,
            )
            if lease is not None:
                raise AssertionError("prevalidated elapsed tick unexpectedly started")
        else:
            skipped = self._scheduler.skip_due(
                now_monotonic_ns,
                reason_code=reason,
                frame=frame,
                queue_depth=0,
            )
            if not skipped:
                raise AssertionError("prevalidated due tick was not skipped")
        if self.trace != preview_trace:
            raise AssertionError("scheduler skip trace differs from prevalidated trace")
        self._adapter_memory = consume_vq2_wave3_coast_lease(
            self._adapter_memory
        )
        self._last_result = result
        return result

    def _validate_step_context(
        self,
        snapshot: VQ2VisionSnapshot,
        safety: VQ2SafetyGuidanceInput,
        timing: VQ2OfflineTickTiming,
    ) -> FrameIdentityV1:
        frame_timing = snapshot.timing
        if type(frame_timing) is not FrameTimingV1:
            raise TypeError("offline runtime requires exact FrameTimingV1")
        frame = frame_timing.identity
        if any(
            type(value) is not int
            for value in (
                snapshot.frame_id,
                snapshot.generation,
                snapshot.sim_time_ns,
            )
        ):
            raise TypeError("snapshot identity fields must be exact integers")
        if (
            snapshot.frame_id != frame.frame_id
            or snapshot.generation != frame.generation
            or snapshot.sim_time_ns != frame_timing.camera_source_time_ns
        ):
            raise ValueError("snapshot identity differs from exact frame timing")
        if (
            snapshot.generation != self.config.camera_generation
            or frame.generation != self.config.camera_generation
        ):
            raise ValueError("snapshot changed configured camera generation")
        if frame.stream_id != self.config.camera_stream_id:
            raise ValueError("snapshot changed configured camera stream")
        if frame_timing.host_clock_id != self.config.host_clock_id:
            raise ValueError("snapshot changed configured host clock")
        if frame_timing.publish_monotonic_ns > timing.tick_start_monotonic_ns:
            raise ValueError("latest snapshot was published after the scheduler poll")
        camera_frame = snapshot.camera_frame
        if type(camera_frame) is not CameraFrame:
            raise TypeError("snapshot camera_frame must be exact CameraFrame")
        if any(
            type(value) is not int
            for value in (
                camera_frame.timestamp_us,
                camera_frame.width,
                camera_frame.height,
            )
        ):
            raise TypeError("camera frame metadata must be exact integers")
        if camera_frame.timestamp_us != snapshot.sim_time_ns // 1_000:
            raise ValueError("camera frame timestamp differs from snapshot sim time")
        if (
            camera_frame.width != self.config.image_width
            or camera_frame.height != self.config.image_height
        ):
            raise ValueError("snapshot dimensions differ from runtime configuration")
        image = camera_frame.image
        if type(image) is not np.ndarray:
            raise TypeError("snapshot image must be an exact numpy.ndarray")
        if image.shape != (
            self.config.image_height,
            self.config.image_width,
            3,
        ):
            raise ValueError("snapshot image has the wrong exact shape")
        if image.dtype != np.dtype(np.uint8):
            raise ValueError("offline snapshot image must use uint8 BGR storage")
        if not image.flags.c_contiguous:
            raise ValueError("offline snapshot image must be C-contiguous")
        if image.flags.writeable:
            raise ValueError("offline snapshot image must be read-only")
        # The legacy freshness clock is sampled independently by the receiver;
        # validate it, but never equate it to the exact /1 packet occurrence.
        if (
            type(snapshot.received_monotonic_s) is not float
            or not math.isfinite(snapshot.received_monotonic_s)
            or snapshot.received_monotonic_s < 0.0
        ):
            raise ValueError(
                "snapshot received_monotonic_s must be a finite nonnegative float"
            )

        authority = safety.authority
        source = self.config.imu_source
        if safety.evaluation_host_clock_id != self.config.host_clock_id:
            raise ValueError("safety changed configured host clock")
        if (
            authority.camera_host_clock_id != self.config.host_clock_id
            or authority.camera_stream_id != self.config.camera_stream_id
            or authority.camera_generation != self.config.camera_generation
        ):
            raise ValueError("safety authority changed configured camera identity")
        if (
            authority.session_id != source.session_id
            or authority.reset_epoch != source.reset_epoch
        ):
            raise ValueError("safety authority changed configured IMU epoch")
        if not (
            timing.tick_start_monotonic_ns
            <= safety.evaluation_monotonic_ns
            <= timing.controller_start_monotonic_ns
        ):
            raise ValueError("safety evaluation must occur inside the tick plan")
        return frame

    def _tracker_id_for_safety(
        self,
        safety: VQ2SafetyGuidanceInput,
    ) -> str:
        authority = safety.authority
        if authority.gate_epoch == 0 and authority.expected_gate_index == 0:
            return self.config.tracker_id
        suffix = (
            f"-gate-{authority.gate_epoch}-{authority.expected_gate_index}"
        )
        prefix_length = 128 - len(suffix)
        if prefix_length <= 0:
            raise ValueError("gate-scoped tracker suffix exceeds token bound")
        return f"{self.config.tracker_id[:prefix_length]}{suffix}"

    def _process_distinct_frame(
        self,
        selection: LatestFrameSelectionV1,
        safety: VQ2SafetyGuidanceInput,
        timing: VQ2OfflineTickTiming,
        estimator: VQ2RelativeGateEstimator,
        tick_id: int,
    ) -> tuple[
        Optional[VQ2ImuCorrelatedEstimatorUpdate],
        Optional[str],
        tuple[LatencyEventV1, ...],
    ]:
        perception = timing.perception
        if perception is None:
            raise AssertionError("distinct frame lacks prevalidated perception timing")
        if safety.evaluation_monotonic_ns != perception.prediction_end_monotonic_ns:
            raise ValueError(
                "distinct-frame safety evaluation must equal prediction end"
            )
        frame = selection.timing.identity
        facts: list[LatencyEventV1] = []
        for kind, field in _CAMERA_STAGES:
            facts.append(
                self._event(
                    kind,
                    getattr(selection.timing, field),
                    frame=frame,
                    sequence_offset=len(facts),
                )
            )
        facts.append(
            self._event(
                LatencyEventKind.DETECTION_START,
                perception.detection_start_monotonic_ns,
                frame=frame,
                sequence_offset=len(facts),
            )
        )
        detections = self._detector.detect(selection.snapshot.camera_frame.image)
        facts.append(
            self._event(
                LatencyEventKind.DETECTION_END,
                perception.detection_end_monotonic_ns,
                frame=frame,
                sequence_offset=len(facts),
            )
        )
        if len(detections) != 1:
            reason = (
                "gate_detection_missing"
                if not detections
                else "gate_detection_ambiguous"
            )
            facts.append(
                self._dropped_fact(
                    frame,
                    perception.detection_end_monotonic_ns,
                    reason,
                    sequence_offset=len(facts),
                )
            )
            return None, reason, tuple(facts)

        facts.append(
            self._event(
                LatencyEventKind.TRACKING_START,
                perception.tracking_start_monotonic_ns,
                frame=frame,
                sequence_offset=len(facts),
            )
        )
        observation = gate_detection_with_aperture_to_observation_v1(
            detections[0],
            selection.snapshot.camera_frame.image,
            frame_timing=selection.timing,
            authority=safety.authority,
            candidate_id=(
                f"{self.config.candidate_id_prefix}-{frame.generation}-{frame.frame_id}"
            ),
            measurement_uncertainty_ns=self.config.measurement_uncertainty_ns,
            fallback_center_covariance=self.config.fallback_center_covariance,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            geometry_config=self.config.aperture_config,
        )
        facts.append(
            self._event(
                LatencyEventKind.TRACKING_END,
                perception.tracking_end_monotonic_ns,
                frame=frame,
                sequence_offset=len(facts),
            )
        )
        if observation.fitted_inner_aperture_corners_norm is None:
            reason = "aperture_geometry_unavailable"
            facts.append(
                self._dropped_fact(
                    frame,
                    perception.tracking_end_monotonic_ns,
                    reason,
                    sequence_offset=len(facts),
                )
            )
            return None, reason, tuple(facts)

        facts.append(
            self._event(
                LatencyEventKind.PREDICTION_START,
                perception.prediction_start_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=len(facts),
            )
        )
        target = RelativePredictionTarget.at_decision(
            self.config.host_clock_id,
            safety.evaluation_monotonic_ns,
        )
        facts.append(
            self._event(
                LatencyEventKind.PREDICTION_END,
                perception.prediction_end_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=len(facts),
            )
        )
        selected_attitudes = self._select_attitudes(observation, target)
        if selected_attitudes is None:
            reason = "imu_correlation_unavailable"
            facts.append(
                self._dropped_fact(
                    frame,
                    perception.prediction_end_monotonic_ns,
                    reason,
                    sequence_offset=len(facts),
                )
            )
            return None, reason, tuple(facts)
        capture, target_attitude = selected_attitudes
        capture_input = VQ2AttitudeDerotationInput(
            capture,
            self.config.capture_orientation_uncertainty_rad,
            self.config.capture_host_time_uncertainty_ns,
        )
        target_input = VQ2AttitudeDerotationInput(
            target_attitude,
            self.config.target_orientation_uncertainty_rad,
            self.config.target_host_time_uncertainty_ns,
        )

        facts.append(
            self._event(
                LatencyEventKind.ESTIMATOR_UPDATE_START,
                perception.estimator_start_monotonic_ns,
                frame=frame,
                sequence_offset=len(facts),
            )
        )
        try:
            evidence = derotate_gate_observation(
                observation,
                target,
                capture_attitude=capture_input,
                target_attitude=target_input,
                calibration=self.config.camera_calibration,
                model=self.config.derotation_model,
            )
            update = estimator.update_with_imu_correlation(evidence)
        except (RelativeEstimatorError, VQ2ImuDerotationError) as exc:
            reason = (
                "relative_estimator_unavailable"
                if isinstance(exc, RelativeEstimatorError)
                else "imu_derotation_unavailable"
            )
            facts.append(
                self._event(
                    LatencyEventKind.ESTIMATOR_UPDATE_END,
                    perception.estimator_end_monotonic_ns,
                    frame=frame,
                    outcome=EventOutcome.ERROR,
                    reason_code=reason,
                    sequence_offset=len(facts),
                )
            )
            facts.append(
                self._dropped_fact(
                    frame,
                    perception.estimator_end_monotonic_ns,
                    reason,
                    sequence_offset=len(facts),
                )
            )
            return None, reason, tuple(facts)
        facts.append(
            self._event(
                LatencyEventKind.ESTIMATOR_UPDATE_END,
                perception.estimator_end_monotonic_ns,
                frame=frame,
                sequence_offset=len(facts),
            )
        )
        return update, None, tuple(facts)

    def _select_attitudes(
        self,
        observation,
        target: RelativePredictionTarget,
    ) -> Optional[tuple[VQ2TimestampedAttitude, VQ2TimestampedAttitude]]:
        available = tuple(
            attitude
            for attitude in self._attitudes
            if attitude.receive_monotonic_ns <= target.decision_time_monotonic_ns
        )
        if not available:
            return None
        measurement = observation.measurement_time_monotonic_ns
        capture = min(
            available,
            key=lambda attitude: (
                abs(attitude.receive_monotonic_ns - measurement),
                -attitude.receive_monotonic_ns,
                -attitude.sample_sequence,
            ),
        )
        if (
            abs(capture.receive_monotonic_ns - measurement)
            > self.config.derotation_model.max_capture_alignment_ns
        ):
            return None
        target_candidates = tuple(
            attitude
            for attitude in available
            if attitude.sample_sequence >= capture.sample_sequence
            and attitude.receive_monotonic_ns
            <= target.prediction_time_monotonic_ns
        )
        if not target_candidates:
            return None
        selected_target = max(
            target_candidates,
            key=lambda attitude: (
                attitude.receive_monotonic_ns,
                attitude.sample_sequence,
            ),
        )
        if (
            target.prediction_time_monotonic_ns
            - selected_target.receive_monotonic_ns
            > self.config.derotation_model.max_target_extrapolation_ns
        ):
            return None
        return capture, selected_target

    def _process_correlated_coast(
        self,
        lease: VQ2Wave3CoastLease,
        safety: VQ2SafetyGuidanceInput,
        timing: VQ2OfflineTickTiming,
        estimator: VQ2RelativeGateEstimator,
        tick_id: int,
    ) -> tuple[
        Optional[VQ2ImuCorrelatedEstimatorCoast],
        Optional[str],
        tuple[LatencyEventV1, ...],
    ]:
        coast_timing = timing.coast
        if coast_timing is None:
            raise AssertionError("eligible coast lacks prevalidated timing")
        if (
            safety.evaluation_monotonic_ns
            != coast_timing.prediction_end_monotonic_ns
        ):
            raise ValueError("coast safety evaluation must equal prediction end")
        frame = lease.source_proposal.source_frame
        if frame is None:
            raise AssertionError("validated coast lease lacks a source frame")
        facts = [
            self._event(
                LatencyEventKind.PREDICTION_START,
                coast_timing.prediction_start_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=0,
            ),
            self._event(
                LatencyEventKind.PREDICTION_END,
                coast_timing.prediction_end_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=1,
            ),
            self._event(
                LatencyEventKind.ESTIMATOR_UPDATE_START,
                coast_timing.estimator_start_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=2,
            ),
        ]
        reason = "imu_correlated_coast_unavailable"
        try:
            prior = lease.source_update
            if estimator.last_state != prior.state:
                raise RelativeEstimatorError(
                    "runtime estimator differs from coast lease source"
                )
            target = RelativePredictionTarget.at_decision(
                self.config.host_clock_id,
                safety.evaluation_monotonic_ns,
            )
            target_attitude = self._select_coast_target_attitude(prior, target)
            if target_attitude is None:
                raise VQ2ImuDerotationError(
                    "strictly newer coast target attitude is unavailable"
                )
            prior_target_input = prior.evidence.target_attitude
            target_input = VQ2AttitudeDerotationInput(
                target_attitude,
                prior_target_input.orientation_uncertainty_rad,
                prior_target_input.host_time_uncertainty_ns,
            )
            evidence = derotate_gate_observation(
                prior.evidence.observation,
                target,
                capture_attitude=prior.evidence.capture_attitude,
                target_attitude=target_input,
                calibration=prior.evidence.calibration,
                model=prior.evidence.model,
            )
            state = estimator.coast(target)
            coast = VQ2ImuCorrelatedEstimatorCoast(
                prior_update=prior,
                state=state,
                evidence=evidence,
            )
        except (
            RelativeEstimatorError,
            VQ2ImuDerotationError,
            TypeError,
            ValueError,
        ):
            facts.append(
                self._event(
                    LatencyEventKind.ESTIMATOR_UPDATE_END,
                    coast_timing.estimator_end_monotonic_ns,
                    frame=frame,
                    control_tick_id=tick_id,
                    outcome=EventOutcome.ERROR,
                    reason_code=reason,
                    sequence_offset=3,
                )
            )
            return None, reason, tuple(facts)
        facts.append(
            self._event(
                LatencyEventKind.ESTIMATOR_UPDATE_END,
                coast_timing.estimator_end_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=3,
            )
        )
        return coast, None, tuple(facts)

    def _select_coast_target_attitude(
        self,
        prior: VQ2ImuCorrelatedEstimatorUpdate,
        target: RelativePredictionTarget,
    ) -> Optional[VQ2TimestampedAttitude]:
        previous = prior.evidence.target_attitude.attitude
        candidates = tuple(
            attitude
            for attitude in self._attitudes
            if attitude.source == previous.source
            and attitude.sample_sequence > previous.sample_sequence
            and attitude.source_time_us > previous.source_time_us
            and attitude.receive_monotonic_ns > previous.receive_monotonic_ns
            and attitude.receive_monotonic_ns
            <= target.prediction_time_monotonic_ns
        )
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda attitude: (
                attitude.receive_monotonic_ns,
                attitude.sample_sequence,
            ),
        )
        if (
            target.prediction_time_monotonic_ns
            - selected.receive_monotonic_ns
            > prior.evidence.model.max_target_extrapolation_ns
        ):
            return None
        return selected

    def _controller_tick(
        self,
        safety: VQ2SafetyGuidanceInput,
        timing: VQ2OfflineTickTiming,
        *,
        tick_id: int,
        deadline_monotonic_ns: int,
        active_state: Optional[RelativeGateStateV1],
    ) -> ControllerTickInput:
        state = active_state
        return ControllerTickInput(
            proposal_id=self._next_proposal_id,
            control_tick_id=tick_id,
            host_clock_id=self.config.host_clock_id,
            proposal_monotonic_ns=timing.controller_end_monotonic_ns,
            control_tick_deadline_monotonic_ns=deadline_monotonic_ns,
            minimum_state_decision_monotonic_ns=(
                0 if state is None else state.timing.decision_time_monotonic_ns
            ),
            minimum_state_sequence=0 if state is None else state.state_sequence,
            expected_phase_started_monotonic_ns=(
                safety.phase_started_monotonic_ns
            ),
            minimum_phase_evaluation_monotonic_ns=(
                safety.evaluation_monotonic_ns
            ),
            expected_authority=safety.authority,
        )

    def _scheduler_preview(
        self,
        *,
        tick_id: int,
        due_monotonic_ns: int,
        finish_monotonic_ns: int,
        frame: FrameIdentityV1,
    ) -> tuple[LatencyEventV1, ...]:
        return (
            self._event(
                LatencyEventKind.CONTROL_TICK_DUE,
                due_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=10_000,
            ),
            self._event(
                LatencyEventKind.CONTROL_TICK_START,
                due_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=10_001,
            ),
            self._event(
                LatencyEventKind.CONTROL_TICK_END,
                finish_monotonic_ns,
                frame=frame,
                control_tick_id=tick_id,
                sequence_offset=10_002,
            ),
        )

    def _dropped_fact(
        self,
        frame: FrameIdentityV1,
        monotonic_ns: int,
        reason: str,
        *,
        sequence_offset: int,
    ) -> LatencyEventV1:
        return self._event(
            LatencyEventKind.FRAME_DROPPED,
            monotonic_ns,
            frame=frame,
            outcome=EventOutcome.DROPPED,
            reason_code=reason,
            sequence_offset=sequence_offset,
        )

    def _event(
        self,
        kind: LatencyEventKind,
        monotonic_ns: int,
        *,
        frame: Optional[FrameIdentityV1] = None,
        control_tick_id: Optional[int] = None,
        sensor_sample_id: Optional[int] = None,
        sensor_source_time_ns: Optional[int] = None,
        outcome: EventOutcome = EventOutcome.OK,
        reason_code: Optional[str] = None,
        sequence_offset: int = 0,
    ) -> LatencyEventV1:
        return LatencyEventV1(
            event_sequence=self._next_fact_sequence + sequence_offset,
            host_clock_id=self.config.host_clock_id,
            monotonic_ns=monotonic_ns,
            kind=kind,
            frame=frame,
            control_tick_id=control_tick_id,
            command_id=None,
            sensor_sample_id=sensor_sample_id,
            sensor_source_time_ns=sensor_source_time_ns,
            outcome=outcome,
            reason_code=reason_code,
            queue_depth=0,
        )

    def _assemble_trace(
        self,
        additional_facts: tuple[LatencyEventV1, ...],
        additional_scheduler: tuple[LatencyEventV1, ...],
    ) -> tuple[LatencyEventV1, ...]:
        scheduler_events = self._scheduler.trace.snapshot()
        values = (
            *self._facts,
            *additional_facts,
            *scheduler_events,
            *additional_scheduler,
        )
        indexed = tuple(enumerate(values))
        ordered = sorted(
            indexed,
            key=lambda item: (
                item[1].monotonic_ns,
                _KIND_ORDER.get(item[1].kind, 100),
                item[0],
            ),
        )
        trace = tuple(
            LatencyEventV1(
                event_sequence=index,
                host_clock_id=event.host_clock_id,
                monotonic_ns=event.monotonic_ns,
                kind=event.kind,
                frame=event.frame,
                control_tick_id=event.control_tick_id,
                command_id=event.command_id,
                sensor_sample_id=event.sensor_sample_id,
                sensor_source_time_ns=event.sensor_source_time_ns,
                outcome=event.outcome,
                reason_code=event.reason_code,
                queue_depth=event.queue_depth,
            )
            for index, (_original_index, event) in enumerate(ordered)
        )
        validate_latency_event_sequence(trace)
        return trace


__all__ = [
    "VQ2OfflineCoastTiming",
    "VQ2OfflinePerceptionTiming",
    "VQ2OfflineTickInput",
    "VQ2OfflineTickResult",
    "VQ2OfflineTickTiming",
    "VQ2Wave3OfflineConfig",
    "VQ2Wave3OfflineRuntime",
]
