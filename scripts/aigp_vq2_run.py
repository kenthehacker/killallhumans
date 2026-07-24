"""Safety-gated runner for AI Grand Prix VQ2 build 3385.

The current qualifier build exposes camera, ``HIGHRES_IMU``, race status,
actuator status, collision messages, and heartbeat, but no pose or usable gate
map.  This runner therefore performs only bounded training stages:

``preflight``
    Receive and validate every required stream.  Sends no arm or flight target.
``sign-id``
    Apply one short, isolated pair of below-hover yaw-rate pulses, then
    stop/reset before the empirically bounded pad-contact horizon.
``hover``
    Level and hold for 2.5 seconds, then stop/reset.
``gate0``
    Approach only the first visible gate and reset immediately when race status
    advances from gate 0 to gate 1.
``gate0-observe``
    Run the proved gate-0 stage, then hold zero thrust for at most 0.20 seconds
    while recording a three-frame observation of the next gate.
``visual-shadow``
    Keep the proved Gate-0 controller in sole command authority while the
    multi-target visual graph proves pre-credit Gate-1 tracking, promotion
    without history reset, and one fresh post-credit observation.
``visual-align``
    Reuse that proved promotion, then apply at most 0.90 seconds of
    no-advance image servo authority to prove uninterrupted horizontal and
    vertical Gate-1 alignment before cleanup.  Passage is forbidden.

Every powered stage proves both the race and IMU clocks rolled back after
``SIM_RESET``, calibrates a gyro-only flight estimator during the countdown,
waits past GO, confirms arming on a newer heartbeat, and runs a latched
watchdog at 50 Hz.  It never consumes the placeholder pose fields present in
``TelemetryState`` and never calls attitude-target mode.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import inspect
import json
import logging
import math
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from competition.adapter import AttitudeRateCommand, Quaternion
from competition.aigp_messages import RaceStatus
from competition.vq2_capture import MavlinkIngressV1, ReceivedIMUSampleV1
from competition.vq2_contracts import FrameEdge
from competition.vq2_passive_timing import CameraFrameTimingObservationV1
from competition.vq2_visual_tracker import (
    CameraFrameToken as VisualCameraFrameToken,
    MultiTargetTrackerConfig,
    MultiTargetVisualTracker,
    VisualDetectionFrame,
    VisualTrack,
    VisualTrackRole,
)
from estimation.imu_attitude import (
    AttitudeEstimate,
    ImuAttitudeConfig,
    ImuAttitudeEstimator,
)
from gate_detection.src.gate_detector import GateDetection
from gate_detection.src.vq2_detector import VQ2GateDetector
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateTransition,
    GateGraphError,
    GateGraphSnapshot,
    RollingVisualGateGraph,
)
from planning.vq2_visual_approach import (
    RollingVisualApproachServo,
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachPassageSafetyUnavailable,
    VisualApproachProposal,
    VisualApproachRefusal,
)
from planning.vq2_visual_alignment import VisualAlignmentTrend
from planning.vq2_visual_servo import (
    MAX_VISUAL_OBSERVATION_AGE_S,
    MAX_VISUAL_THRUST,
    MAX_VISUAL_YAW_RATE_RAD_S,
    VisualServoRefusal,
    VisualTarget,
)
from scripts.aigp_vq2_controller_config import (
    ControllerConfigError,
    VQ2ControllerConfig,
    default_controller_config,
    validate_controller_config,
)
from scripts.aigp_vq2_visual_config import (
    VisualConfigError,
    VisualNavigationConfig,
    default_visual_config,
    validate_visual_config,
)
from scripts.aigp_vq2_visual_alignment_stage import (
    VisualAlignmentStageLimits,
    VisualAlignmentStageRuntime,
    run_visual_alignment_stage,
)

if TYPE_CHECKING:
    from aigp_loop.replay import AsyncReplayRecorder
    from competition.aigp_mavlink import AIGPMavlinkAdapter
    from competition.vq2_vision import VQ2VisionThread


# Importing this module is part of the powered-child bootstrap path.  Keep the
# two modules that can construct live transports out of the import graph until
# immutable process/capability admission has succeeded.  Legacy callers load
# them on first use through ``_load_live_transport_dependencies``.
DEFAULT_MAVLINK_URL = "udpin:127.0.0.1:14550"
AIGPMavlinkAdapter: Any = None
VQ2VisionThread: Any = None
_attitude_error_body_rates: Any = None


def _load_live_transport_dependencies() -> Tuple[Any, Any, Any]:
    global AIGPMavlinkAdapter, VQ2VisionThread, _attitude_error_body_rates
    if AIGPMavlinkAdapter is None or _attitude_error_body_rates is None:
        from competition.aigp_mavlink import (
            AIGPMavlinkAdapter as adapter_type,
            _attitude_error_body_rates as attitude_error_body_rates,
        )

        if AIGPMavlinkAdapter is None:
            AIGPMavlinkAdapter = adapter_type
        if _attitude_error_body_rates is None:
            _attitude_error_body_rates = attitude_error_body_rates
    if VQ2VisionThread is None:
        from competition.vq2_vision import VQ2VisionThread as vision_type

        VQ2VisionThread = vision_type
    return AIGPMavlinkAdapter, VQ2VisionThread, _attitude_error_body_rates


logger = logging.getLogger("aigp.vq2")

CONTROL_HZ = 50.0
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ

SIGN_ID_RATE_RAD_S = 0.08
SIGN_ID_THRUST = 0.235
SIGN_ID_RESPONSE_SETTLE_S = 0.04
SIGN_ID_YAW_PULSE_DURATION_S = 0.21
SIGN_ID_YAW_NEUTRAL_DURATION_S = 0.24
SIGN_ID_YAW_REVERSAL_DURATION_S = 0.12
SIGN_ID_YAW_TERMINAL_DURATION_S = 0.04
SIGN_ID_HARD_EXPIRY_S = 0.95
SIGN_ID_MIN_RESPONSE_RAD_S = 0.006
SIGN_ID_MIN_YAW_GYRO_SAMPLES = 4
SIGN_ID_MIN_FRESH_IMAGE_FRAMES = 4
SIGN_ID_MIN_IMAGE_EFFECT_PX_S = 15.0
SIGN_ID_MAX_POLARITY_GAIN_RATIO = 2.0
SIGN_ID_MAX_ATTITUDE_EXCURSION_RAD = 0.05
SIGN_ID_MAX_MEASURED_YAW_RATE_RAD_S = 0.50

MAX_HEARTBEAT_AGE_S = 1.5
MAX_IMU_AGE_S = 0.050
MAX_RACE_AGE_S = 0.40
MAX_ACTUATOR_AGE_S = 0.10
MAX_VISION_AGE_S = 0.10
MAX_TARGET_LOSS_S = 0.25

CROSSING_TARGET_LOSS_S = 0.08
CROSSING_STATUS_TIMEOUT_S = 0.40
CROSSING_MIN_AREA_RATIO = 25.0
CROSSING_MIN_WIDTH_PX = 512
GATE0_FLIGHT_TIMEOUT_S = 5.0
GATE0_PITCH_BLEND_S = 0.8

# Retained only as an offline-scaffold identity. It is intentionally absent
# from every live CLI and powered dispatcher.
FULL_LAP_STAGE = "full-lap"
FULL_LAP_TIMEOUT_S = 45.0
FULL_LAP_MAX_GATE_INDEX = 15
FULL_LAP_INITIAL_GATE_MIN_AREA_PX = 6_000
FULL_LAP_INITIAL_GATE_MAX_AREA_PX = 8_000
COURSE_GATE_TIMEOUT_S = 8.0
COURSE_RECENTER_DURATION_S = 0.60
COURSE_RECENTER_MAX_NORMALIZED_X = 0.35
COURSE_APPROACH_PITCH_RAD = -0.20
COURSE_CROSSING_AREA_CAP_PX = int(0.70 * 640 * 360)
COURSE_RECENTER_MAX_RATE_RAD_S = 0.12
COURSE_ROLL_GAIN = 0.25
COURSE_RECENTER_ROLL_GAIN = 0.12
COURSE_RECENTER_ROLL_LIMIT_RAD = 0.05
COURSE_APPROACH_ROLL_LIMIT_RAD = 0.16
COURSE_TRANSITION_THRUST = 0.0
COURSE_RECENTER_THRUST = 0.275
COURSE_HIGH_GATE_Y_PX = 100.0
COURSE_GATE0_EXIT_PITCH_RAD = 0.0
COURSE_GATE0_BOOST_UNTIL_S = 0.80
COURSE_GATE0_MIN_THRUST = 0.21
COURSE_GATE0_EXPECTED_TARGET_LOSS_S = 2.50
COURSE_RACE_PACKET_PERIOD_S = 0.250
COURSE_RACE_PACKET_TARGET_LEAD_S = 0.060
COURSE_LINE_MIN_ROI_PIXELS = 128
COURSE_LINE_PRETURN_MIN_GATE_AREA_SCALE = 1.30
COURSE_LINE_PRETURN_MIN_SCORE = 0.04
COURSE_LINE_PRETURN_GAIN = 0.80
COURSE_LINE_PRETURN_LIMIT_RAD = 0.13
COURSE_LINE_EXIT_COUNTERROLL_ONSET_AREA_SCALE = 3.5
COURSE_LINE_EXIT_COUNTERROLL_RAD = 0.08
COURSE_LINE_PRETURN_TAPER_AREA_SCALE = 8.0
COURSE_LINE_PRETURN_REQUIRED_FRAMES = 3
COURSE_LINE_PRETURN_MAX_AGE_S = 0.25
COURSE_EDGE_CONTINUATION_MARGIN_PX = 2
COURSE_EDGE_CONTINUATION_MAX_ASPECT_RATIO = 2.60
COURSE_FRAGMENT_UNION_MAX_ASPECT_RATIO = 1.45
COURSE_FRAGMENT_UNION_RIGHT_EDGE_MAX_ASPECT_RATIO = 1.48
COURSE_FRAGMENT_UNION_MIN_IOU = 0.75
COURSE_UNTRACKED_CONTACT_MIN_AREA_PX = int(0.10 * 640 * 360)
COURSE_UNTRACKED_CONTACT_MIN_WIDTH_PX = 160
COURSE_UNTRACKED_CONTACT_MIN_HEIGHT_PX = 120

POST_GATE_OBSERVATION_TIMEOUT_S = 0.20
POST_GATE_REQUIRED_FRAMES = 3
POST_GATE_MAX_ATTITUDE_DELTA_RAD = math.radians(5.0)
POST_GATE_IMMEDIATE_MAX_BODY_RATE_RAD_S = 1.0
POST_GATE_SUSTAINED_MAX_BODY_RATE_RAD_S = 0.5

# User-authorized bounded live trial. Pixel-rate damping stays disabled until
# the authoritative replay/tracker-isolation prerequisite is accepted.
GATE1_RECENTER_STAGE = "gate1-recenter"
VISUAL_SHADOW_STAGE = "visual-shadow"
VISUAL_ALIGN_STAGE = "visual-align"
VISUAL_POWERED_STAGES = (VISUAL_SHADOW_STAGE, VISUAL_ALIGN_STAGE)
VISUAL_SHADOW_POST_CREDIT_TIMEOUT_S = 0.15
VISUAL_SHADOW_REQUIRED_PRETRANSITION_FRAMES = 3
VISUAL_ALIGN_HARD_DURATION_S = 0.90
VISUAL_ALIGN_POST_CREDIT_FRAME_TIMEOUT_S = 0.12
VISUAL_ALIGN_RESPONSE_GRACE_S = 0.12
VISUAL_ALIGN_MAX_YAW_RATE_RAD_S = 0.08
VISUAL_ALIGN_YAW_SOFT_STOP_RAD = 0.16
VISUAL_ALIGN_MAX_YAW_EXCURSION_RAD = 0.18
VISUAL_ALIGN_YAW_HOLD_HORIZON_S = 0.12
VISUAL_ALIGN_MAX_MEASURED_YAW_RATE_RAD_S = 0.35
VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S = 0.12
VISUAL_ALIGN_MAX_ABS_ROLL_RAD = 0.12
VISUAL_ALIGN_MIN_PITCH_RAD = -0.20
VISUAL_ALIGN_MAX_PITCH_RAD = 0.08
VISUAL_ALIGN_MAX_ENTRY_ATTITUDE_DELTA_RAD = 0.08
VISUAL_ALIGN_MAX_BODY_RATE_RAD_S = 0.50
VISUAL_ALIGN_MIN_THRUST = 0.21
VISUAL_ALIGN_MAX_THRUST = 0.30
# Real build-3385 replay proves the stable next gate is occluded by Gate 0 for
# 461.7 ms, then reappears with 0.034 normalized prediction residual and 0.549
# bbox IoU.  Preserve identity across that bounded aperture occlusion without
# extending the existing 12-publication retirement lease.
VISUAL_TRACKER_MAX_ASSOCIATION_GAP_NS = 500_000_000
VISUAL_TRACKER_CONFIG = MultiTargetTrackerConfig(
    max_association_gap_ns=VISUAL_TRACKER_MAX_ASSOCIATION_GAP_NS,
)
VISUAL_ALIGNMENT_STAGE_LIMITS = VisualAlignmentStageLimits(
    control_period_s=CONTROL_PERIOD_S,
    required_pretransition_frames=(
        VISUAL_SHADOW_REQUIRED_PRETRANSITION_FRAMES
    ),
    hard_duration_s=VISUAL_ALIGN_HARD_DURATION_S,
    post_credit_frame_timeout_s=(
        VISUAL_ALIGN_POST_CREDIT_FRAME_TIMEOUT_S
    ),
    response_grace_s=VISUAL_ALIGN_RESPONSE_GRACE_S,
    max_yaw_rate_rad_s=VISUAL_ALIGN_MAX_YAW_RATE_RAD_S,
    max_command_rate_rad_s=VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S,
    max_pitch_rad=VISUAL_ALIGN_MAX_PITCH_RAD,
    max_entry_attitude_delta_rad=(
        VISUAL_ALIGN_MAX_ENTRY_ATTITUDE_DELTA_RAD
    ),
    min_thrust=VISUAL_ALIGN_MIN_THRUST,
    max_thrust=VISUAL_ALIGN_MAX_THRUST,
    max_visual_controller_thrust=MAX_VISUAL_THRUST,
)
GATE1_RECENTER_DURATION_S = 0.60
GATE1_RECENTER_ROLL_GAIN = -0.24
GATE1_RECENTER_ROLL_RATE_GAIN = 0.0
GATE1_RECENTER_MAX_ROLL_RAD = 0.12
GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S = 0.12
GATE1_RECENTER_THRUST = 0.275
GATE1_RECENTER_TRANSITION_THRUST = GATE1_RECENTER_THRUST
GATE1_RECENTER_TARGET_PITCH_RAD = 0.10
GATE1_RECENTER_MIN_THRUST = 0.21
GATE1_RECENTER_MAX_THRUST = 0.30
GATE1_RECENTER_CORRIDOR_NORMALIZED_X = 0.35
GATE1_RECENTER_REQUIRED_CORRIDOR_FRAMES = 3
GATE1_RECENTER_DIVERGENCE_PX = 24.0
GATE1_RECENTER_MAX_ABS_X_RATE_NORM_S = 4.0
GATE1_RECENTER_MAX_ABS_ROLL_RAD = 0.15
GATE1_RECENTER_MIN_PITCH_RAD = -0.20
GATE1_RECENTER_MAX_PITCH_RAD = 0.10
GATE1_RECENTER_MAX_ATTITUDE_EXCURSION_RAD = 0.12
GATE1_RECENTER_MAX_MEASURED_BODY_RATE_RAD_S = 0.50
GATE1_CONTROLLER_MAX_YAW_EXCURSION_RAD = 0.05
GATE1_CONTROLLER_YAW_SOFT_STOP_RAD = 0.045
GATE1_RECENTER_NO_PASSAGE_MAX_AREA_PX = COURSE_UNTRACKED_CONTACT_MIN_AREA_PX
GATE1_RECENTER_NO_PASSAGE_MAX_WIDTH_PX = COURSE_UNTRACKED_CONTACT_MIN_WIDTH_PX

MAX_BENIGN_PAD_CONTACTS = 12
MAX_BENIGN_PAD_IMPULSE = 0.05

MAX_ROLL_RAD = math.radians(25.0)
MIN_PITCH_RAD = math.radians(-35.0)
MAX_PITCH_RAD = math.radians(10.0)
MAX_BODY_RATE_RAD_S = 2.0
IMMEDIATE_MAX_BODY_RATE_RAD_S = 3.0
MAX_COMMAND_RATE_RAD_S = 0.25
CALIBRATION_MAX_ATTITUDE_EXCURSION_RAD = 0.025

RESET_RACE_DROP_MS = 500
RESET_IMU_DROP_US = 100_000
RESET_PROOF_TIMEOUT_S = 2.8
RESET_MAX_ATTEMPTS = 4


def _replay_capture_dependencies():
    """Load optional evidence tooling only when private capture is requested."""

    from aigp_loop._util import environment_fingerprint, git_provenance
    from aigp_loop.replay import AsyncReplayRecorder, ReplayBundleWriter

    return (
        AsyncReplayRecorder,
        ReplayBundleWriter,
        environment_fingerprint,
        git_provenance,
    )


class SafetyAbort(RuntimeError):
    """A latched no-recovery flight watchdog failure."""


class CalibrationBootstrapError(RuntimeError):
    """The wrapper-owned powered-child admission could not be proved."""


class CalibrationEvidenceError(RuntimeError):
    """Required calibration evidence could not be validated or enqueued."""


class CalibrationLifecycleError(RuntimeError):
    """The admitted powered-child lifecycle could not complete exactly."""


CALIBRATION_STAGE = "calibration-excite"
LIVE_RUN_STAGES = (
    "preflight",
    "sign-id",
    "hover",
    "gate0",
    "gate0-observe",
    GATE1_RECENTER_STAGE,
    *VISUAL_POWERED_STAGES,
    CALIBRATION_STAGE,
)
CALIBRATION_CHILD_ROLE = "powered_child"
CALIBRATION_CAPABILITY_DOMAIN = "aigp-vq2-powered-child/1"
CALIBRATION_CAPABILITY_RELEASE_NS = 3_000_000_000
CALIBRATION_OWNED_HANDLE_CLOSE_NS = 2_000_000_000


@dataclass(frozen=True)
class CalibrationArguments:
    stage: str
    powered_attempt_envelope: str
    wrapper_process: str
    powered_process_authority: str
    attempt_capability_handle: str
    parent_liveness_handle: str
    record: str
    replay_bundle: str
    cleanup_certificate: str
    recording_approved: bool


def build_calibration_argument_parser() -> argparse.ArgumentParser:
    """Build the exact, wrapper-only powered-child parser.

    There is deliberately no address, waveform, duration, amplitude, thrust,
    geometry, or safety override on this surface.
    """

    parser = argparse.ArgumentParser(
        prog="python -m scripts.aigp_vq2_run",
        allow_abbrev=False,
    )
    parser.add_argument("--stage", required=True, choices=(CALIBRATION_STAGE,))
    parser.add_argument("--powered-attempt-envelope", required=True)
    parser.add_argument("--wrapper-process", required=True)
    parser.add_argument("--powered-process-authority", required=True)
    parser.add_argument("--attempt-capability-handle", required=True)
    parser.add_argument("--parent-liveness-handle", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--replay-bundle", required=True)
    parser.add_argument("--cleanup-certificate", required=True)
    parser.add_argument("--recording-approved", required=True, action="store_true")
    return parser


def parse_calibration_arguments(argv: Sequence[str]) -> CalibrationArguments:
    if type(argv) not in {list, tuple} or any(type(item) is not str for item in argv):
        raise TypeError("calibration argv must be an exact string list or tuple")
    namespace = build_calibration_argument_parser().parse_args(list(argv))
    return CalibrationArguments(
        stage=namespace.stage,
        powered_attempt_envelope=namespace.powered_attempt_envelope,
        wrapper_process=namespace.wrapper_process,
        powered_process_authority=namespace.powered_process_authority,
        attempt_capability_handle=namespace.attempt_capability_handle,
        parent_liveness_handle=namespace.parent_liveness_handle,
        record=namespace.record,
        replay_bundle=namespace.replay_bundle,
        cleanup_certificate=namespace.cleanup_certificate,
        recording_approved=namespace.recording_approved,
    )


class CalibrationProcessBoundary(Protocol):
    """Retained Windows process/handle proof supplied by the wrapper runtime."""

    def current_argv(self) -> Sequence[str]: ...

    def current_process_identity(self) -> Mapping[str, Any]: ...

    def retained_process_identity(self, handle: int) -> Mapping[str, Any]: ...

    def prove_inherited_handle_policy(
        self,
        *,
        capability_handle: int,
        parent_handle: int,
        process_authority: Mapping[str, Any],
    ) -> bool: ...

    def parent_signaled(self, handle: int) -> bool: ...

    def close_owned_handles(
        self,
        *,
        deadline_monotonic_ns: int,
        monotonic_ns: Callable[[], int],
    ) -> Any: ...


@dataclass
class CalibrationAdmissionServices:
    """Injected bootstrap services; none may construct or open a live transport."""

    process_boundary: CalibrationProcessBoundary
    capability_operations: Any
    monotonic_ns: Callable[[], int]
    contract: Any = None
    runtime: Any = None
    load_record: Optional[Callable[[str, Any], Mapping[str, Any]]] = None
    run_admitted: Optional[Callable[["CalibrationAdmission"], Any]] = None
    child_services: Optional["CalibrationChildServices"] = None
    owned_process_boundary: Optional[CalibrationProcessBoundary] = None
    close_unconsumed_capability: Optional[Callable[[], bool]] = None


@dataclass
class CalibrationAdmission:
    arguments: CalibrationArguments
    attempt: Dict[str, Any]
    live_freeze: Dict[str, Any]
    process_authority: Dict[str, Any]
    current_process: Dict[str, Any]
    wrapper_process: Dict[str, Any]
    process_argv: Tuple[str, ...]
    capability_handle: int
    parent_handle: int
    role_secret: bytearray = field(repr=False)
    admitted_monotonic_ns: int = 0
    total_deadline_monotonic_ns: int = 0
    prepower_deadline_monotonic_ns: int = 0
    powered_deadline_monotonic_ns: int = 0
    cleanup_deadline_monotonic_ns: int = 0
    replay_close_deadline_monotonic_ns: int = 0
    exit_deadline_monotonic_ns: int = 0
    attempt_envelope_sha256: str = ""
    process_authority_sha256: str = ""

    def erase_role_secret(self) -> None:
        for index in range(len(self.role_secret)):
            self.role_secret[index] = 0


def _powered_contract_modules(services: CalibrationAdmissionServices) -> Tuple[Any, Any]:
    contract = services.contract
    runtime = services.runtime
    if contract is None:
        from scripts import aigp_vq2_powered_attempt as contract_module

        contract = contract_module
    if runtime is None:
        from scripts import aigp_vq2_powered_runtime as runtime_module

        runtime = runtime_module
    return contract, runtime


def _stable_calibration_record(path: str, contract: Any, runtime: Any) -> Mapping[str, Any]:
    before = runtime.stable_file_identity(path)
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise CalibrationBootstrapError("immutable bootstrap record could not be read") from exc
    after = runtime.stable_file_identity(path)
    if before != after or hashlib.sha256(payload).hexdigest() != before.sha256:
        raise CalibrationBootstrapError("immutable bootstrap record changed while reading")
    try:
        return contract.parse_canonical_json_bytes(payload, file_form=True)
    except BaseException as exc:
        raise CalibrationBootstrapError("bootstrap record is not canonical JSON") from exc


def _load_calibration_record(
    path: str,
    services: CalibrationAdmissionServices,
    contract: Any,
    runtime: Any,
) -> Mapping[str, Any]:
    if services.load_record is not None:
        return services.load_record(path, contract)
    return _stable_calibration_record(path, contract, runtime)


def _calibration_require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise CalibrationBootstrapError(
            f"{label} does not match immutable wrapper authority"
        )


def admit_calibration_child(
    arguments: CalibrationArguments,
    services: CalibrationAdmissionServices,
) -> CalibrationAdmission:
    """Consume the exact child capability before any live import or contact."""

    if not isinstance(arguments, CalibrationArguments):
        raise TypeError("arguments must be CalibrationArguments")
    if not isinstance(services, CalibrationAdmissionServices):
        raise TypeError("services must be CalibrationAdmissionServices")
    contract, runtime = _powered_contract_modules(services)
    try:
        capability_handle = runtime.parse_decimal_handle(
            arguments.attempt_capability_handle
        )
        parent_handle = runtime.parse_decimal_handle(
            arguments.parent_liveness_handle
        )
        wrapper_token = runtime.parse_process_identity_token(
            arguments.wrapper_process
        )
    except BaseException as exc:
        raise CalibrationBootstrapError("child handle or wrapper token is invalid") from exc
    if capability_handle == parent_handle:
        raise CalibrationBootstrapError("capability and parent handles must be distinct")

    try:
        attempt_initial = contract.validate_attempt(
            _load_calibration_record(
                arguments.powered_attempt_envelope,
                services,
                contract,
                runtime,
            )
        )
        live_freeze = contract.validate_live_freeze(
            _load_calibration_record(
                attempt_initial["context"]["live_freeze"]["path"],
                services,
                contract,
                runtime,
            )
        )
        attempt = contract.validate_attempt(
            attempt_initial,
            live_freeze=live_freeze,
        )
        process_argv = tuple(services.process_boundary.current_argv())
        if not process_argv or any(type(item) is not str for item in process_argv):
            raise CalibrationBootstrapError("current process argv proof is invalid")
        authority = contract.validate_process_authority(
            _load_calibration_record(
                arguments.powered_process_authority,
                services,
                contract,
                runtime,
            ),
            attempt=attempt,
            argv=process_argv,
        )
    except CalibrationBootstrapError:
        raise
    except BaseException as exc:
        raise CalibrationBootstrapError(
            "attempt or powered-child authority validation failed"
        ) from exc

    if authority["role"] != CALIBRATION_CHILD_ROLE:
        raise CalibrationBootstrapError("process authority role is not powered child")
    context = attempt["context"]
    paths = context["paths"]
    exact_values = (
        (
            arguments.powered_attempt_envelope,
            paths["attempt_envelope"],
            "attempt-envelope path",
        ),
        (
            arguments.powered_process_authority,
            paths["child_authority"],
            "process-authority path",
        ),
        (arguments.record, paths["legacy_record"], "legacy-record path"),
        (arguments.replay_bundle, paths["replay_bundle"], "replay-bundle path"),
        (
            arguments.cleanup_certificate,
            paths["child_cleanup_certificate"],
            "cleanup-certificate path",
        ),
        (process_argv, tuple(context["child_argv"]), "powered-child argv"),
        (
            parent_handle,
            authority["parent_handle"]["value"],
            "parent-liveness handle",
        ),
    )
    for actual, expected, label in exact_values:
        _calibration_require_equal(actual, expected, label)
    if arguments.stage != CALIBRATION_STAGE or arguments.recording_approved is not True:
        raise CalibrationBootstrapError("powered child stage/recording approval changed")

    wrapper_expected = context["wrapper_process"]
    if type(wrapper_expected) is not dict:
        raise CalibrationBootstrapError("wrapper process authority is invalid")
    if (
        wrapper_token.pid != wrapper_expected["pid"]
        or wrapper_token.creation_filetime_100ns
        != wrapper_expected["creation_filetime_100ns"]
    ):
        raise CalibrationBootstrapError("wrapper identity token does not match attempt")
    try:
        current_process = runtime.validate_process_identity(
            services.process_boundary.current_process_identity()
        )
        retained_parent = runtime.validate_process_identity(
            services.process_boundary.retained_process_identity(parent_handle)
        )
    except BaseException as exc:
        raise CalibrationBootstrapError("retained process identity proof failed") from exc
    _calibration_require_equal(
        current_process,
        authority["process"],
        "current process identity",
    )
    _calibration_require_equal(
        retained_parent,
        wrapper_expected,
        "retained wrapper identity",
    )
    if services.process_boundary.prove_inherited_handle_policy(
        capability_handle=capability_handle,
        parent_handle=parent_handle,
        process_authority=authority,
    ) is not True:
        raise CalibrationBootstrapError("inherited handle policy is unproved")

    deadlines = authority["absolute_deadlines"]
    capability_deadline = min(
        deadlines["anchor"] + CALIBRATION_CAPABILITY_RELEASE_NS,
        deadlines["total"],
    )
    try:
        secret = runtime.read_bound_capability(
            capability_handle,
            parent_handle,
            domain=CALIBRATION_CAPABILITY_DOMAIN,
            context_sha256=attempt["context_sha256"],
            expected_capability_sha256=authority["capability_sha256"],
            deadline_monotonic_ns=capability_deadline,
            operations=services.capability_operations,
            monotonic_ns=services.monotonic_ns,
        )
    except BaseException as exc:
        raise CalibrationBootstrapError("powered-child capability admission failed") from exc
    admitted = runtime.read_qpc_ns(services.monotonic_ns)
    if admitted >= deadlines["total"]:
        erased = bytearray(secret)
        for index in range(len(erased)):
            erased[index] = 0
        raise CalibrationBootstrapError("powered-child total deadline expired at admission")
    return CalibrationAdmission(
        arguments=arguments,
        attempt=dict(attempt),
        live_freeze=dict(live_freeze),
        process_authority=dict(authority),
        current_process=dict(current_process),
        wrapper_process=dict(retained_parent),
        process_argv=process_argv,
        capability_handle=capability_handle,
        parent_handle=parent_handle,
        role_secret=bytearray(secret),
        admitted_monotonic_ns=admitted,
        total_deadline_monotonic_ns=deadlines["total"],
        prepower_deadline_monotonic_ns=deadlines["prepower"],
        powered_deadline_monotonic_ns=deadlines["powered"],
        cleanup_deadline_monotonic_ns=deadlines["cleanup"],
        replay_close_deadline_monotonic_ns=deadlines["replay_close"],
        exit_deadline_monotonic_ns=deadlines["exit"],
        attempt_envelope_sha256=contract.canonical_file_sha256(attempt),
        process_authority_sha256=contract.canonical_file_sha256(authority),
    )


CALIBRATION_COMMAND_FAILURE_CODES = frozenset(
    {
        "slot_missed",
        "deadline_expired",
        "stream_stale",
        "imu_not_advancing",
        "race_not_advancing",
        "estimator_unhealthy",
        "target_missing",
        "target_unstable",
        "target_out_of_corridor",
        "target_too_large",
        "attitude_excursion",
        "collision_observed",
        "gate_changed",
        "capture_failed",
        "parent_dead",
        "lease_invalid",
        "send_raised",
        "internal_error",
    }
)


class CalibrationCheckFailure(SafetyAbort):
    """A typed pre-send refusal with one frozen evidence reason code."""

    def __init__(self, reason_code: str, detail: str) -> None:
        if reason_code not in CALIBRATION_COMMAND_FAILURE_CODES:
            raise ValueError("unsupported calibration command failure code")
        if type(detail) is not str or not detail:
            raise ValueError("calibration failure detail must be nonempty")
        self.reason_code = reason_code
        self.detail = detail.encode("utf-8", "replace")[:512].decode(
            "utf-8", "ignore"
        )
        super().__init__(self.detail)


@dataclass(frozen=True)
class CalibrationSafetyFacts:
    """Primitive, same-occurrence facts consumed by one excitation check."""

    checked_monotonic_ns: int
    reset_epoch: Mapping[str, Any]
    frame: Mapping[str, Any]
    imu: Mapping[str, Any]
    race: Mapping[str, Any]
    heartbeat: Mapping[str, Any]
    actuator: Mapping[str, Any]
    imu_advance_monotonic_ns: int
    race_advance_monotonic_ns: int
    estimator_healthy: bool
    target_consecutive: int
    target_center_px: Tuple[float, float]
    target_bbox_px: Tuple[float, float, float, float]
    initial_target_bbox_area_px: float
    start_roll_rad: float
    start_pitch_rad: float
    current_roll_rad: float
    current_pitch_rad: float
    collision_count: int
    capture_healthy: bool
    parent_alive: bool
    lease_valid: bool


@dataclass(frozen=True)
class CalibrationSafetyAuthorization:
    reset_epoch: Dict[str, Any]
    source: Dict[str, Any]
    watchdogs: Dict[str, Any]


def _calibration_exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CalibrationCheckFailure("internal_error", f"{label} is invalid")
    return value


def _calibration_finite(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise CalibrationCheckFailure("internal_error", f"{label} is invalid")
    return float(value)


def _calibration_age_ns(now_ns: int, occurrence_ns: Any, label: str) -> int:
    occurrence = _calibration_exact_nonnegative_int(occurrence_ns, label)
    if occurrence > now_ns:
        raise CalibrationCheckFailure(
            "stream_stale", f"{label} follows the safety-check occurrence"
        )
    return now_ns - occurrence


def evaluate_calibration_safety(
    facts: CalibrationSafetyFacts,
    *,
    contract: Any = None,
) -> CalibrationSafetyAuthorization:
    """Validate the exact gate-0 corridor and source/watchdog lineage.

    This is intentionally independent of adapter mutable side state.  The
    caller supplies defensive-copy receiver envelopes from one drain pass.
    """

    if not isinstance(facts, CalibrationSafetyFacts):
        raise TypeError("facts must be CalibrationSafetyFacts")
    if contract is None:
        from scripts import aigp_vq2_powered_attempt as contract_module

        contract = contract_module
    now_ns = _calibration_exact_nonnegative_int(
        facts.checked_monotonic_ns, "checked_monotonic_ns"
    )
    try:
        reset_epoch = {
            "ingress_generation": facts.reset_epoch["ingress_generation"],
            "race_anchor_boot_ms": facts.reset_epoch["race_anchor_boot_ms"],
            "imu_anchor_usec": facts.reset_epoch["imu_anchor_usec"],
        }
        frame = dict(facts.frame)
        imu = contract.validate_received_imu(dict(facts.imu))
        race = contract.validate_received_race_status(dict(facts.race))
        heartbeat = contract.validate_received_heartbeat(dict(facts.heartbeat))
        actuator = contract.validate_received_actuator_output_status(
            dict(facts.actuator)
        )
    except CalibrationCheckFailure:
        raise
    except BaseException as exc:
        raise CalibrationCheckFailure(
            "stream_stale", "receiver or frame lineage is invalid"
        ) from exc

    generation = _calibration_exact_nonnegative_int(
        reset_epoch["ingress_generation"], "reset ingress generation"
    )
    race_anchor = _calibration_exact_nonnegative_int(
        reset_epoch["race_anchor_boot_ms"], "reset race anchor"
    )
    imu_anchor = _calibration_exact_nonnegative_int(
        reset_epoch["imu_anchor_usec"], "reset IMU anchor"
    )
    for label, observation in (
        ("IMU", imu),
        ("race", race),
        ("heartbeat", heartbeat),
        ("actuator", actuator),
    ):
        if observation["ingress"]["generation"] != generation:
            raise CalibrationCheckFailure(
                "stream_stale", f"{label} is outside the proved reset generation"
            )
    frame_keys = {
        "stream_id",
        "generation",
        "frame_id",
        "sim_time_ns",
        "timing",
        "width",
        "height",
    }
    if type(facts.frame) is not dict or set(frame) != frame_keys:
        raise CalibrationCheckFailure("capture_failed", "frame source shape is invalid")
    if frame["generation"] != generation:
        raise CalibrationCheckFailure(
            "capture_failed", "frame is outside the proved reset generation"
        )
    if frame["width"] != 640 or frame["height"] != 360:
        raise CalibrationCheckFailure(
            "capture_failed", "decoded dimensions are not stable 640x360"
        )
    timing = frame.get("timing")
    if type(timing) is not dict:
        raise CalibrationCheckFailure("capture_failed", "frame timing is absent")
    identity = timing.get("identity")
    if type(identity) is not dict or any(
        frame[name] != identity.get(name)
        for name in ("stream_id", "generation", "frame_id")
    ):
        raise CalibrationCheckFailure(
            "capture_failed", "frame identity does not match its timing record"
        )
    if frame["sim_time_ns"] != timing.get("camera_source_time_ns"):
        raise CalibrationCheckFailure(
            "capture_failed", "frame source token does not match timing"
        )
    if imu["imu"]["timestamp_us"] <= imu_anchor:
        raise CalibrationCheckFailure(
            "imu_not_advancing", "IMU did not advance beyond reset anchor"
        )
    if race["race_status"]["sim_boot_time_ms"] <= race_anchor:
        raise CalibrationCheckFailure(
            "race_not_advancing", "race clock did not advance beyond reset anchor"
        )

    heartbeat_age = _calibration_age_ns(
        now_ns,
        heartbeat["ingress"]["received_monotonic_ns"],
        "heartbeat receipt",
    )
    imu_age = _calibration_age_ns(
        now_ns, imu["ingress"]["received_monotonic_ns"], "IMU receipt"
    )
    race_age = _calibration_age_ns(
        now_ns, race["ingress"]["received_monotonic_ns"], "race receipt"
    )
    actuator_age = _calibration_age_ns(
        now_ns,
        actuator["ingress"]["received_monotonic_ns"],
        "actuator receipt",
    )
    vision_age = _calibration_age_ns(
        now_ns,
        timing.get("final_unique_packet_monotonic_ns"),
        "camera final-packet receipt",
    )
    imu_advance_age = _calibration_age_ns(
        now_ns, facts.imu_advance_monotonic_ns, "IMU advance"
    )
    race_advance_age = _calibration_age_ns(
        now_ns, facts.race_advance_monotonic_ns, "race advance"
    )
    age_limits = (
        (heartbeat_age, 1_500_000_000, "stream_stale", "heartbeat is stale"),
        (imu_age, 50_000_000, "stream_stale", "IMU receipt is stale"),
        (imu_advance_age, 50_000_000, "imu_not_advancing", "IMU is not advancing"),
        (race_age, 400_000_000, "stream_stale", "race status is stale"),
        (
            race_advance_age,
            400_000_000,
            "race_not_advancing",
            "race clock is not advancing",
        ),
        (actuator_age, 100_000_000, "stream_stale", "actuator status is stale"),
        (vision_age, 100_000_000, "stream_stale", "camera frame is stale"),
    )
    for age, maximum, reason, detail in age_limits:
        if age > maximum:
            raise CalibrationCheckFailure(reason, detail)
    if facts.capture_healthy is not True:
        raise CalibrationCheckFailure("capture_failed", "capture callback is unhealthy")
    if facts.parent_alive is not True:
        raise CalibrationCheckFailure("parent_dead", "wrapper parent is no longer live")
    if facts.lease_valid is not True:
        raise CalibrationCheckFailure("lease_invalid", "powered lease lineage is invalid")
    if facts.estimator_healthy is not True:
        raise CalibrationCheckFailure(
            "estimator_unhealthy", "attitude estimator is unhealthy"
        )
    if type(facts.target_consecutive) is not int or facts.target_consecutive < 3:
        raise CalibrationCheckFailure(
            "target_unstable", "gate target lacks three-frame confirmation"
        )
    if type(facts.collision_count) is not int or facts.collision_count < 0:
        raise CalibrationCheckFailure("internal_error", "collision count is invalid")
    if facts.collision_count:
        raise CalibrationCheckFailure(
            "collision_observed", "every calibration collision aborts"
        )
    if race["race_status"]["active_gate_index"] != 0:
        raise CalibrationCheckFailure("gate_changed", "active gate is not gate zero")

    if type(facts.target_center_px) is not tuple or len(facts.target_center_px) != 2:
        raise CalibrationCheckFailure("target_missing", "target center is unavailable")
    if type(facts.target_bbox_px) is not tuple or len(facts.target_bbox_px) != 4:
        raise CalibrationCheckFailure("target_missing", "target bbox is unavailable")
    center_x, center_y = (
        _calibration_finite(value, "target center")
        for value in facts.target_center_px
    )
    bbox = tuple(
        _calibration_finite(value, "target bbox") for value in facts.target_bbox_px
    )
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise CalibrationCheckFailure("target_missing", "target bbox is empty")
    initial_area = _calibration_finite(
        facts.initial_target_bbox_area_px, "initial target area"
    )
    if initial_area <= 0.0:
        raise CalibrationCheckFailure("target_missing", "initial target area is invalid")
    area = bbox[2] * bbox[3]
    if not 64.0 <= center_x <= 576.0 or not 36.0 <= center_y <= 324.0:
        raise CalibrationCheckFailure(
            "target_out_of_corridor", "target center left the closed safety corridor"
        )
    if bbox[2] > 160.0 or bbox[3] > 160.0 or area > 2.0 * initial_area:
        raise CalibrationCheckFailure(
            "target_too_large", "target bbox exceeded the calibration safety limit"
        )
    roll_excursion = abs(
        _calibration_finite(facts.current_roll_rad, "current roll")
        - _calibration_finite(facts.start_roll_rad, "start roll")
    )
    pitch_excursion = abs(
        _calibration_finite(facts.current_pitch_rad, "current pitch")
        - _calibration_finite(facts.start_pitch_rad, "start pitch")
    )
    if roll_excursion > 0.05 or pitch_excursion > 0.05:
        raise CalibrationCheckFailure(
            "attitude_excursion", "roll or pitch excursion exceeded 0.05 rad"
        )

    source = {
        "frame": frame,
        "imu": imu,
        "race": race,
        "heartbeat": heartbeat,
        "actuator": actuator,
    }
    watchdogs = {
        "checked_monotonic_ns": now_ns,
        "heartbeat_age_ns": heartbeat_age,
        "imu_age_ns": imu_age,
        "imu_advance_age_ns": imu_advance_age,
        "race_age_ns": race_age,
        "race_advance_age_ns": race_advance_age,
        "actuator_age_ns": actuator_age,
        "vision_age_ns": vision_age,
        "estimator_healthy": True,
        "target_consecutive": facts.target_consecutive,
        "target_center_px": [center_x, center_y],
        "target_bbox_px": list(bbox),
        "target_bbox_area_px": area,
        "initial_target_bbox_area_px": initial_area,
        "roll_excursion_rad": roll_excursion,
        "pitch_excursion_rad": pitch_excursion,
        "collision_count": 0,
        "gate_index": 0,
        "result": "pass",
        "failure_codes": [],
    }
    return CalibrationSafetyAuthorization(
        reset_epoch=reset_epoch,
        source=source,
        watchdogs=watchdogs,
    )


class CalibrationSnapshotCapture:
    """Thread-safe dimension admission that precedes every replay frame copy."""

    EXPECTED_WIDTH = 640
    EXPECTED_HEIGHT = 360

    def __init__(
        self,
        *,
        recorder: "JsonlRecorder",
        config_sha256: str,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        contract: Any = None,
    ) -> None:
        if not isinstance(recorder, JsonlRecorder) or recorder.replay is None:
            raise ValueError("calibration capture requires an active replay recorder")
        if (
            type(config_sha256) is not str
            or len(config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in config_sha256)
        ):
            raise ValueError("config_sha256 must be 64 lowercase hexadecimal characters")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        if contract is None:
            from scripts import aigp_vq2_powered_attempt as contract_module

            contract = contract_module
        self.recorder = recorder
        self.config_sha256 = config_sha256
        self.monotonic_ns = monotonic_ns
        self.contract = contract
        # Serialize the complete admission -> replay-forward sequence.  The
        # production receiver is single-threaded today, but callback ordering
        # must remain true if that implementation changes.
        self._lock = threading.RLock()
        self._dimensions: Optional[Tuple[int, int]] = None
        self._admission: Optional[Dict[str, Any]] = None
        self._failure: Optional[str] = None
        self._observed_frames = 0

    @property
    def admitted(self) -> bool:
        with self._lock:
            return self._admission is not None and self._failure is None

    @property
    def observed_frames(self) -> int:
        with self._lock:
            return self._observed_frames

    @property
    def failure(self) -> Optional[str]:
        with self._lock:
            return self._failure

    @property
    def admission(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self._admission is None else dict(self._admission)

    def _latch(self, reason: str) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = reason
        try:
            self.recorder.replay.fail(f"calibration decoded-frame capture failed: {reason}")
        except BaseException:
            pass

    def raise_if_failed(self) -> None:
        failure = self.failure
        if failure is not None:
            raise CalibrationCheckFailure("capture_failed", failure)

    def _snapshot_facts(self, snapshot: Any) -> Tuple[int, int, Dict[str, Any]]:
        image = getattr(getattr(snapshot, "camera_frame", None), "image", None)
        shape = getattr(image, "shape", None)
        if type(shape) is not tuple or len(shape) != 3 or shape[2] != 3:
            raise ValueError("decoded image must have exact HxWx3 shape")
        if str(getattr(image, "dtype", "")) != "uint8":
            raise ValueError("decoded image must have uint8 dtype")
        height = int(shape[0])
        width = int(shape[1])
        camera_frame = snapshot.camera_frame
        if (
            type(getattr(camera_frame, "width", None)) is not int
            or type(getattr(camera_frame, "height", None)) is not int
            or camera_frame.width != width
            or camera_frame.height != height
        ):
            raise ValueError("CameraFrame dimensions do not match decoded image")
        timing = getattr(snapshot, "timing", None)
        to_primitive = getattr(timing, "to_primitive", None)
        if not callable(to_primitive):
            raise ValueError("decoded snapshot lacks exact frame timing")
        timing_row = to_primitive()
        identity = timing_row.get("identity") if type(timing_row) is dict else None
        if type(identity) is not dict or (
            identity.get("generation") != getattr(snapshot, "generation", None)
            or identity.get("frame_id") != getattr(snapshot, "frame_id", None)
            or timing_row.get("camera_source_time_ns")
            != getattr(snapshot, "sim_time_ns", None)
        ):
            raise ValueError("snapshot identity does not match frame timing")
        return width, height, timing_row

    def __call__(self, snapshot: Any) -> bool:
        """Validate dimensions, then and only then forward to replay capture."""

        try:
            with self._lock:
                width, height, timing = self._snapshot_facts(snapshot)
                if self._failure is not None:
                    raise ValueError(self._failure)
                dimensions = (width, height)
                if self._dimensions is not None and dimensions != self._dimensions:
                    raise ValueError("decoded dimensions drifted within the session")
                if dimensions != (self.EXPECTED_WIDTH, self.EXPECTED_HEIGHT):
                    raise ValueError("decoded dimensions are not exact 640x360")
                first = self._dimensions is None
                if first:
                    self._dimensions = dimensions
                self._observed_frames += 1
                if first:
                    admitted_ns = self.monotonic_ns()
                    if type(admitted_ns) is not int or admitted_ns < 0:
                        raise ValueError("dimension admission clock is invalid")
                    row = {
                        "schema": "aigp-vq2-decoded-dimensions-admission/1",
                        "config_sha256": self.config_sha256,
                        "expected": {
                            "width": self.EXPECTED_WIDTH,
                            "height": self.EXPECTED_HEIGHT,
                        },
                        "observed": {"width": width, "height": height},
                        "first_frame_timing": timing,
                        "admitted_monotonic_ns": admitted_ns,
                        "status": "admitted",
                    }
                    checked = self.contract.validate_decoded_dimensions_admission(row)
                    if self.recorder.emit_powered(
                        "decoded_dimensions_admission", observation=checked
                    ) is not True:
                        raise CalibrationEvidenceError(
                            "decoded-dimensions admission enqueue failed"
                        )
                    self._admission = dict(checked)
                accepted = self.recorder.replay.capture_decoded_snapshot(snapshot)
                if accepted is not True:
                    raise CalibrationEvidenceError(
                        "decoded snapshot enqueue returned false"
                    )
            return True
        except BaseException as exc:
            self._latch(f"{type(exc).__name__}: {exc}")
            raise


def calibration_vision_options(
    capture: CalibrationSnapshotCapture,
    *,
    exclusive_socket_factory: Callable[[str, int], Any],
) -> Dict[str, Any]:
    """Return the non-overridable powered vision construction options."""

    if not isinstance(capture, CalibrationSnapshotCapture):
        raise TypeError("capture must be CalibrationSnapshotCapture")
    if not callable(exclusive_socket_factory):
        raise TypeError("exclusive_socket_factory must be callable")
    return {
        "on_snapshot": capture,
        "capture_snapshot_queue_enabled": True,
        "powered_exclusive": True,
        "exclusive_socket_factory": exclusive_socket_factory,
    }


@dataclass(frozen=True)
class CalibrationDispatchResult:
    audit_count_before: int
    audit_count_after: int
    receipt: Optional[Mapping[str, Any]]
    call_started_monotonic_ns: Optional[int]
    call_ended_monotonic_ns: Optional[int]
    error: Optional[BaseException] = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class CalibrationScheduleResult:
    anchor_monotonic_ns: int
    powered_expiry_monotonic_ns: int
    sent_ticks: Tuple[int, ...]
    skipped_before_generation: Tuple[int, ...]
    skipped_after_generation: Tuple[int, ...]
    completed: bool
    abort_reason_code: Optional[str]


@dataclass(frozen=True)
class CalibrationNonattitudeDispatch:
    category: str
    request_monotonic_ns: int
    receipt: Optional[Mapping[str, Any]]
    audit_count_before: int
    audit_count_after: int
    call_started_monotonic_ns: Optional[int]
    call_ended_monotonic_ns: Optional[int]
    boundary: Optional[Any] = field(default=None, compare=False, repr=False)
    error: Optional[BaseException] = field(default=None, compare=False, repr=False)


class CalibrationCommandEvidence:
    """Validator-backed rich command/tick evidence with one event sequence."""

    _EVENT_NAMES = {
        "generated": "calibration_command_generated",
        "sent": "calibration_command_sent",
        "not_sent": "calibration_command_not_sent",
        "tick": "calibration_tick_disposition",
        "phase": "calibration_phase_deadline",
    }

    def __init__(
        self,
        *,
        attempt: Mapping[str, Any],
        recorder: Any = None,
        contract: Any = None,
        initial_event_sequence: int = 0,
    ) -> None:
        if contract is None:
            from scripts import aigp_vq2_powered_attempt as contract_module

            contract = contract_module
        self.contract = contract
        self.attempt = contract.validate_attempt(dict(attempt))
        if type(initial_event_sequence) is not int or initial_event_sequence < 0:
            raise ValueError("initial_event_sequence must be nonnegative")
        self.recorder = recorder
        self._next_event_sequence = initial_event_sequence
        self.observations: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def next_event_sequence(self) -> int:
        return self._next_event_sequence

    def _sequence(self) -> int:
        value = self._next_event_sequence
        self._next_event_sequence += 1
        return value

    def _emit(self, event: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.recorder is not None:
            emit_powered = getattr(self.recorder, "emit_powered", None)
            if not callable(emit_powered):
                raise CalibrationEvidenceError(
                    "calibration recorder lacks strict powered-event support"
                )
            if emit_powered(event, observation=observation) is not True:
                raise CalibrationEvidenceError(
                    f"replay enqueue failed for {event}"
                )
        copied = json.loads(json.dumps(observation, allow_nan=False))
        self.observations.append((event, copied))
        return copied

    def record_phase_deadline(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        row = {
            "schema": "aigp-vq2-phase-deadline/1",
            "attempt_id": self.contract.ATTEMPT_ID,
            "producer_role": CALIBRATION_CHILD_ROLE,
            "phase": value["phase"],
            "event_sequence": self._sequence(),
            "started_monotonic_ns": value["started_monotonic_ns"],
            "duration_ns": value["duration_ns"],
            "parent_deadline_monotonic_ns": value[
                "parent_deadline_monotonic_ns"
            ],
            "deadline_monotonic_ns": value["deadline_monotonic_ns"],
        }
        checked = self.contract.validate_phase_deadline_event(row)
        return self._emit(self._EVENT_NAMES["phase"], checked)

    def _common(
        self,
        tick: Mapping[str, Any],
        authorization: CalibrationSafetyAuthorization,
    ) -> Dict[str, Any]:
        context = self.attempt["context"]
        absolute_tick = tick["absolute_tick"]
        return {
            "attempt_id": self.contract.ATTEMPT_ID,
            "session_id": self.contract.SESSION_ID,
            "candidate_commit": context["candidate_commit"],
            "attempt_context_sha256": self.attempt["context_sha256"],
            "host_clock_id": self.contract.HOST_CLOCK_ID,
            "reset_epoch": dict(authorization.reset_epoch),
            "plan": {
                "plan_id": self.contract.EXCITATION_PLAN_ID,
                "sha256": self.contract.EXCITATION_PLAN_SHA256,
            },
            "scope": "excitation",
            "command_id": f"excitation/{absolute_tick:03d}",
            "absolute_tick": absolute_tick,
            "segment_id": tick["segment_id"],
            "slot": {
                "release_monotonic_ns": tick["release_monotonic_ns"],
                "end_monotonic_ns": tick["end_monotonic_ns"],
                "powered_expiry_monotonic_ns": tick[
                    "powered_expiry_monotonic_ns"
                ],
            },
            "command": dict(tick["command"]),
            "source": dict(authorization.source),
            "watchdogs": dict(authorization.watchdogs),
        }

    def _cleanup_common(self, checked_monotonic_ns: int) -> Dict[str, Any]:
        if type(checked_monotonic_ns) is not int or checked_monotonic_ns < 0:
            raise ValueError("cleanup authorization time must be nonnegative")
        context = self.attempt["context"]
        return {
            "attempt_id": self.contract.ATTEMPT_ID,
            "session_id": self.contract.SESSION_ID,
            "candidate_commit": context["candidate_commit"],
            "attempt_context_sha256": self.attempt["context_sha256"],
            "host_clock_id": self.contract.HOST_CLOCK_ID,
            "reset_epoch": None,
            "plan": None,
            "scope": "cleanup_zero",
            "command_id": "cleanup/zero/0",
            "absolute_tick": None,
            "segment_id": None,
            "slot": None,
            "command": {
                "roll_rate_rad_s": 0.0,
                "pitch_rate_rad_s": 0.0,
                "yaw_rate_rad_s": 0.0,
                "thrust": 0.0,
            },
            "source": {
                "frame": None,
                "imu": None,
                "race": None,
                "heartbeat": None,
                "actuator": None,
            },
            "watchdogs": {
                "checked_monotonic_ns": checked_monotonic_ns,
                "heartbeat_age_ns": None,
                "imu_age_ns": None,
                "imu_advance_age_ns": None,
                "race_age_ns": None,
                "race_advance_age_ns": None,
                "actuator_age_ns": None,
                "vision_age_ns": None,
                "estimator_healthy": None,
                "target_consecutive": None,
                "target_center_px": None,
                "target_bbox_px": None,
                "target_bbox_area_px": None,
                "initial_target_bbox_area_px": None,
                "roll_excursion_rad": None,
                "pitch_excursion_rad": None,
                "collision_count": None,
                "gate_index": None,
                "result": "cleanup_authorized",
                "failure_codes": [],
            },
        }

    def record_generated(
        self,
        tick: Mapping[str, Any],
        authorization: CalibrationSafetyAuthorization,
        generated_monotonic_ns: int,
    ) -> Dict[str, Any]:
        row = {
            "schema": "aigp-vq2-calibration-command-generated/1",
            **self._common(tick, authorization),
            "event_sequence": self._sequence(),
            "generated_monotonic_ns": generated_monotonic_ns,
        }
        checked = self.contract.validate_calibration_command_generated(row)
        if self.recorder is not None:
            command = checked["command"]
            core = getattr(self.recorder, "record_command", None)
            if not callable(core) or core(
                "generated",
                AttitudeRateCommand(
                    command["roll_rate_rad_s"],
                    command["pitch_rate_rad_s"],
                    command["yaw_rate_rad_s"],
                    command["thrust"],
                ),
                monotonic_s=generated_monotonic_ns / 1_000_000_000.0,
                frame_token=(
                    checked["source"]["frame"]["generation"],
                    checked["source"]["frame"]["frame_id"],
                    checked["source"]["frame"]["sim_time_ns"],
                ),
            ) is not True:
                raise CalibrationEvidenceError("generated core command enqueue failed")
        return self._emit(self._EVENT_NAMES["generated"], checked)

    def record_cleanup_generated(
        self,
        *,
        checked_monotonic_ns: int,
        generated_monotonic_ns: int,
    ) -> Dict[str, Any]:
        """Record the sole exact-zero cleanup command before its API call."""

        row = {
            "schema": "aigp-vq2-calibration-command-generated/1",
            **self._cleanup_common(checked_monotonic_ns),
            "event_sequence": self._sequence(),
            "generated_monotonic_ns": generated_monotonic_ns,
        }
        checked = self.contract.validate_calibration_command_generated(row)
        if self.recorder is not None:
            core = getattr(self.recorder, "record_command", None)
            if not callable(core) or core(
                "generated",
                AttitudeRateCommand(0.0, 0.0, 0.0, 0.0),
                monotonic_s=generated_monotonic_ns / 1_000_000_000.0,
                frame_token=None,
            ) is not True:
                raise CalibrationEvidenceError(
                    "cleanup generated core command enqueue failed"
                )
        return self._emit(self._EVENT_NAMES["generated"], checked)

    def record_sent(
        self,
        generated: Mapping[str, Any],
        *,
        sent_monotonic_ns: int,
        dispatch: CalibrationDispatchResult,
    ) -> Dict[str, Any]:
        row = {
            key: value
            for key, value in generated.items()
            if key not in {"schema", "event_sequence", "generated_monotonic_ns"}
        }
        row.update(
            {
                "schema": "aigp-vq2-calibration-command-sent/1",
                "event_sequence": self._sequence(),
                "sent_monotonic_ns": sent_monotonic_ns,
                "generated_event_sequence": generated["event_sequence"],
                "generation_sha256": self.contract.canonical_object_sha256(
                    dict(generated)
                ),
                "transport": {
                    "receipt": dict(dispatch.receipt or {}),
                    "audit_count_before": dispatch.audit_count_before,
                    "audit_count_after": dispatch.audit_count_after,
                },
            }
        )
        checked = self.contract.validate_calibration_command_sent(
            row, generated=dict(generated)
        )
        if self.recorder is not None:
            command = checked["command"]
            source_frame = checked["source"]["frame"]
            core = getattr(self.recorder, "record_command", None)
            if not callable(core) or core(
                "sent",
                AttitudeRateCommand(
                    command["roll_rate_rad_s"],
                    command["pitch_rate_rad_s"],
                    command["yaw_rate_rad_s"],
                    command["thrust"],
                ),
                monotonic_s=sent_monotonic_ns / 1_000_000_000.0,
                frame_token=(
                    None
                    if source_frame is None
                    else (
                        source_frame["generation"],
                        source_frame["frame_id"],
                        source_frame["sim_time_ns"],
                    )
                ),
            ) is not True:
                raise CalibrationEvidenceError("sent core command enqueue failed")
        return self._emit(self._EVENT_NAMES["sent"], checked)

    def record_not_sent(
        self,
        generated: Mapping[str, Any],
        *,
        recorded_monotonic_ns: int,
        reason_code: str,
        detail: str,
        dispatch: Optional[CalibrationDispatchResult] = None,
    ) -> Dict[str, Any]:
        before = 0 if dispatch is None else dispatch.audit_count_before
        after = before if dispatch is None else dispatch.audit_count_after
        kind = "skipped_after_generation" if dispatch is None else "send_failed_or_uncertain"
        row = {
            key: value
            for key, value in generated.items()
            if key not in {"schema", "event_sequence", "generated_monotonic_ns"}
        }
        row.update(
            {
                "schema": "aigp-vq2-calibration-command-not-sent/1",
                "event_sequence": self._sequence(),
                "recorded_monotonic_ns": recorded_monotonic_ns,
                "generated_event_sequence": generated["event_sequence"],
                "generation_sha256": self.contract.canonical_object_sha256(
                    dict(generated)
                ),
                "outcome": {
                    "kind": kind,
                    "reason_code": reason_code,
                    "detail": detail,
                    "audit_count_before": before,
                    "audit_count_after": after,
                    "call_started_monotonic_ns": (
                        None if dispatch is None else dispatch.call_started_monotonic_ns
                    ),
                    "call_ended_monotonic_ns": (
                        None if dispatch is None else dispatch.call_ended_monotonic_ns
                    ),
                },
            }
        )
        checked = self.contract.validate_calibration_command_not_sent(
            row, generated=dict(generated)
        )
        return self._emit(self._EVENT_NAMES["not_sent"], checked)

    def record_tick_disposition(
        self,
        tick: Mapping[str, Any],
        *,
        recorded_monotonic_ns: int,
        disposition: str,
        generated_event_sequence: Optional[int],
        terminal_event_sequence: Optional[int],
        reason_code: Optional[str],
    ) -> Dict[str, Any]:
        row = {
            "schema": "aigp-vq2-calibration-tick-disposition/1",
            "attempt_id": self.contract.ATTEMPT_ID,
            "session_id": self.contract.SESSION_ID,
            "attempt_context_sha256": self.attempt["context_sha256"],
            "plan_id": self.contract.EXCITATION_PLAN_ID,
            "plan_sha256": self.contract.EXCITATION_PLAN_SHA256,
            "event_sequence": self._sequence(),
            "host_clock_id": self.contract.HOST_CLOCK_ID,
            "recorded_monotonic_ns": recorded_monotonic_ns,
            "absolute_tick": tick["absolute_tick"],
            "segment_id": tick["segment_id"],
            "slot": {
                "release_monotonic_ns": tick["release_monotonic_ns"],
                "end_monotonic_ns": tick["end_monotonic_ns"],
                "powered_expiry_monotonic_ns": tick[
                    "powered_expiry_monotonic_ns"
                ],
            },
            "disposition": disposition,
            "generated_event_sequence": generated_event_sequence,
            "terminal_event_sequence": terminal_event_sequence,
            "reason_code": reason_code,
        }
        checked = self.contract.validate_calibration_tick_disposition(row)
        return self._emit(self._EVENT_NAMES["tick"], checked)


class CalibrationLineageRecorder:
    """Sole strict recorder for receiver, reset, collision, and outbound facts."""

    _RECEIVED_EVENTS = {
        "aigp-vq2-received-heartbeat/1": (
            "received_heartbeat",
            "validate_received_heartbeat",
        ),
        "aigp-vq2-received-race-status/1": (
            "received_race_status",
            "validate_received_race_status",
        ),
        "aigp-vq2-received-actuator-output-status/1": (
            "received_actuator_output_status",
            "validate_received_actuator_output_status",
        ),
        "aigp-vq2-received-imu/1": ("received_imu", "validate_received_imu"),
    }

    def __init__(self, recorder: "JsonlRecorder", *, contract: Any = None) -> None:
        if not isinstance(recorder, JsonlRecorder) or recorder.replay is None:
            raise ValueError("calibration lineage requires an active replay recorder")
        if contract is None:
            from scripts import aigp_vq2_powered_attempt as contract_module

            contract = contract_module
        self.recorder = recorder
        self.contract = contract
        self.received: List[Dict[str, Any]] = []
        self.collisions: List[Dict[str, Any]] = []
        self.outbound_receipts: List[Dict[str, Any]] = []
        self.reset_boundaries: List[Dict[str, Any]] = []
        self._collision_generation: Optional[int] = None
        self._next_collision_sequence = 0

    @staticmethod
    def _primitive(value: Any, label: str) -> Dict[str, Any]:
        to_primitive = getattr(value, "to_primitive", None)
        if not callable(to_primitive):
            raise CalibrationEvidenceError(f"{label} lacks to_primitive()")
        primitive = to_primitive()
        if type(primitive) is not dict:
            raise CalibrationEvidenceError(f"{label} primitive is not an exact object")
        return primitive

    @staticmethod
    def _require_enqueue(result: Any, label: str) -> None:
        if result is not True:
            raise CalibrationEvidenceError(f"{label} replay enqueue failed")

    def record_received(
        self,
        observation: Any,
        *,
        estimator: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        primitive = self._primitive(observation, "received observation")
        schema = primitive.get("schema")
        route = self._RECEIVED_EVENTS.get(schema)
        if route is None:
            raise CalibrationEvidenceError("unsupported received-observation schema")
        event, validator_name = route
        validator = getattr(self.contract, validator_name)
        checked = validator(primitive)
        ingress = getattr(observation, "ingress", None)
        self._require_enqueue(
            self.recorder.record_mavlink_ingress(ingress),
            "legacy MAVLink ingress",
        )
        received_s = checked["ingress"]["received_monotonic_ns"] / 1_000_000_000.0
        if schema == "aigp-vq2-received-imu/1":
            # AsyncReplayRecorder emits the linked strict received_imu event in
            # the same operation as its core IMU row.
            self._require_enqueue(
                self.recorder.record_imu(
                    observation.imu,
                    None if estimator is None else dict(estimator),
                    received_s,
                    received_sample=observation,
                ),
                "received IMU",
            )
        else:
            if schema == "aigp-vq2-received-race-status/1":
                payload = checked["race_status"]
                race = RaceStatus(
                    sim_boot_time_ms=payload["sim_boot_time_ms"],
                    race_start_boot_time_ms=payload["race_start_boot_time_ms"],
                    race_finish_time_ns=payload["race_finish_time_ns"],
                    active_gate_index=payload["active_gate_index"],
                    last_gate_race_time=payload["last_gate_race_time"],
                )
                self._require_enqueue(
                    self.recorder.record_race(race, received_s),
                    "core race status",
                )
            self._require_enqueue(
                self.recorder.emit_powered(event, observation=checked),
                event,
            )
        self.received.append(dict(checked))
        return dict(checked)

    def record_collision(
        self,
        collision: Mapping[str, Any],
        *,
        reset_generation: int,
        observed_monotonic_ns: int,
        phase: str,
        disposition: str,
    ) -> Dict[str, Any]:
        if self._collision_generation != reset_generation:
            self._collision_generation = reset_generation
            self._next_collision_sequence = 0
        row = {
            "schema": "aigp-vq2-runner-collision-observation/1",
            "reset_generation": reset_generation,
            "observation_sequence": self._next_collision_sequence,
            "host_clock_id": self.contract.HOST_CLOCK_ID,
            "observed_monotonic_ns": observed_monotonic_ns,
            "phase": phase,
            "disposition": disposition,
            "boundary": "runner_drain_not_receiver_receipt",
            "collision": dict(collision),
        }
        checked = self.contract.validate_collision_observation(row)
        self._require_enqueue(
            self.recorder.emit_powered(
                "runner_collision_observation", observation=checked
            ),
            "runner collision observation",
        )
        self._next_collision_sequence += 1
        self.collisions.append(dict(checked))
        return dict(checked)

    def retain_reset_boundary_without_replay(
        self,
        boundary: Any,
        *,
        phase: str,
    ) -> Dict[str, Any]:
        """Retain cleanup evidence after replay persistence has failed.

        This path deliberately performs no recorder calls. It reconciles any
        rows that were committed before the failure, fills only the missing
        local certificate evidence, and leaves replay capture invalidated.
        """

        primitive = self._primitive(boundary, "calibration reset boundary")
        observations: List[Dict[str, Any]] = []
        received_by_token = {
            (
                item["ingress"]["stream_id"],
                item["ingress"]["generation"],
                item["ingress"]["sequence"],
            ): item
            for item in self.received
        }
        for observation in boundary.observations:
            raw = self._primitive(observation, "boundary observation")
            route = self._RECEIVED_EVENTS.get(raw.get("schema"))
            if route is None:
                raise CalibrationEvidenceError(
                    "unsupported received-observation schema"
                )
            checked = getattr(self.contract, route[1])(raw)
            ingress = checked["ingress"]
            token = (
                ingress["stream_id"],
                ingress["generation"],
                ingress["sequence"],
            )
            existing = received_by_token.get(token)
            if existing is not None and existing != checked:
                raise CalibrationEvidenceError(
                    "boundary ingress token conflicts with retained evidence"
                )
            if existing is None:
                existing = dict(checked)
                self.received.append(existing)
                received_by_token[token] = existing
            observations.append(dict(existing))

        old_generation = boundary.old_generation
        boundary_time = boundary.boundary_monotonic_ns
        available = [
            item
            for item in self.collisions
            if item["reset_generation"] == old_generation
            and item["observed_monotonic_ns"] == boundary_time
            and item["phase"] == phase
            and item["disposition"] == "reset_boundary_discard"
        ]
        used_tokens: set[tuple[int, int]] = set()
        collisions: List[Dict[str, Any]] = []
        if self._collision_generation != old_generation:
            self._collision_generation = old_generation
            prior_sequences = [
                item["observation_sequence"]
                for item in self.collisions
                if item["reset_generation"] == old_generation
            ]
            self._next_collision_sequence = (
                0 if not prior_sequences else max(prior_sequences) + 1
            )
        for collision in boundary.collisions:
            payload = collision.to_primitive()
            existing = next(
                (
                    item
                    for item in available
                    if item["collision"] == payload
                    and (
                        item["reset_generation"],
                        item["observation_sequence"],
                    )
                    not in used_tokens
                ),
                None,
            )
            if existing is None:
                row = {
                    "schema": "aigp-vq2-runner-collision-observation/1",
                    "reset_generation": old_generation,
                    "observation_sequence": self._next_collision_sequence,
                    "host_clock_id": self.contract.HOST_CLOCK_ID,
                    "observed_monotonic_ns": boundary_time,
                    "phase": phase,
                    "disposition": "reset_boundary_discard",
                    "boundary": "runner_drain_not_receiver_receipt",
                    "collision": payload,
                }
                existing = self.contract.validate_collision_observation(row)
                self._next_collision_sequence += 1
                self.collisions.append(dict(existing))
            token = (
                existing["reset_generation"],
                existing["observation_sequence"],
            )
            used_tokens.add(token)
            collisions.append(dict(existing))

        primitive["observations"] = observations
        primitive["collisions"] = collisions
        checked_boundary = self.contract.validate_reset_boundary(primitive)
        if checked_boundary not in self.reset_boundaries:
            self.reset_boundaries.append(dict(checked_boundary))
        self._collision_generation = boundary.new_generation
        self._next_collision_sequence = 0
        return dict(checked_boundary)

    def record_reset_boundary(
        self,
        boundary: Any,
        *,
        phase: str,
    ) -> Dict[str, Any]:
        primitive = self._primitive(boundary, "calibration reset boundary")
        observations = []
        for observation in boundary.observations:
            observations.append(self.record_received(observation))
        collisions = []
        for collision in boundary.collisions:
            collisions.append(
                self.record_collision(
                    collision.to_primitive(),
                    reset_generation=boundary.old_generation,
                    observed_monotonic_ns=boundary.boundary_monotonic_ns,
                    phase=phase,
                    disposition="reset_boundary_discard",
                )
            )
        primitive["observations"] = observations
        primitive["collisions"] = collisions
        checked = self.contract.validate_reset_boundary(primitive)
        self._require_enqueue(
            self.recorder.emit_powered(
                "calibration_reset_boundary", observation=checked
            ),
            "calibration reset boundary",
        )
        self.reset_boundaries.append(dict(checked))
        self._collision_generation = boundary.new_generation
        self._next_collision_sequence = 0
        return dict(checked)

    def record_outbound(self, receipt: Any) -> Dict[str, Any]:
        primitive = self._primitive(receipt, "outbound receipt")
        schema = primitive.get("schema")
        if schema == "aigp-vq2-attitude-target-outbound/1":
            checked = self.contract.validate_attitude_target_outbound(primitive)
            event = "attitude_target_outbound"
        elif schema == "aigp-vq2-nonattitude-outbound/1":
            checked = self.contract.validate_nonattitude_outbound(primitive)
            event = "nonattitude_outbound"
        else:
            raise CalibrationEvidenceError("unsupported outbound receipt schema")
        self._require_enqueue(
            self.recorder.emit_powered(event, observation=checked), event
        )
        self.outbound_receipts.append(dict(checked))
        return dict(checked)


class CalibrationAdapterDispatcher:
    """Bridge the pure scheduler to one powered adapter and its receipts."""

    def __init__(
        self,
        adapter: Any,
        lineage: CalibrationLineageRecorder,
        *,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        parent_alive: Callable[[], bool],
        lease_valid: Callable[[], bool],
    ) -> None:
        if not isinstance(lineage, CalibrationLineageRecorder):
            raise TypeError("lineage must be CalibrationLineageRecorder")
        for callback, label in (
            (monotonic_ns, "monotonic_ns"),
            (parent_alive, "parent_alive"),
            (lease_valid, "lease_valid"),
        ):
            if not callable(callback):
                raise TypeError(f"{label} must be callable")
        self.adapter = adapter
        self.lineage = lineage
        self.monotonic_ns = monotonic_ns
        self.parent_alive = parent_alive
        self.lease_valid = lease_valid

    def _now(self) -> int:
        value = self.monotonic_ns()
        if type(value) is not int or value < 0:
            raise CalibrationEvidenceError(
                "adapter dispatcher clock must return a nonnegative exact integer"
            )
        return value

    def _attitude_audit_count(self) -> int:
        return self._audit_count("attitude_target")

    def _audit_count(self, category: str) -> int:
        audit = self.adapter.outbound_audit()
        value = (
            audit.get(category)
            if isinstance(audit, Mapping)
            else getattr(audit, category, None)
        )
        if type(value) is not int or value < 0:
            raise CalibrationEvidenceError(
                f"adapter {category} audit is invalid"
            )
        return value

    def drain_outbound(self) -> List[Dict[str, Any]]:
        drain = getattr(self.adapter, "drain_outbound_receipts", None)
        if not callable(drain):
            raise CalibrationEvidenceError("adapter lacks outbound receipt drain")
        return [self.lineage.record_outbound(value) for value in drain()]

    async def dispatch(
        self,
        command: AttitudeRateCommand,
        deadline_monotonic_ns: int,
    ) -> CalibrationDispatchResult:
        if self.parent_alive() is not True:
            raise CalibrationCheckFailure("parent_dead", "wrapper parent is not live")
        if self.lease_valid() is not True:
            raise CalibrationCheckFailure("lease_invalid", "powered lease is invalid")
        if type(deadline_monotonic_ns) is not int or self._now() >= deadline_monotonic_ns:
            raise CalibrationCheckFailure(
                "deadline_expired", "adapter call deadline was reached"
            )
        self.drain_outbound()
        before = self._attitude_audit_count()
        call_started = self._now()
        error: Optional[BaseException] = None
        try:
            await self.adapter.send_attitude_rate(
                command,
                powered_deadline_monotonic_ns=deadline_monotonic_ns,
                powered_cleanup=False,
            )
        except BaseException as exc:
            error = exc
        call_ended = self._now()
        receipts = self.drain_outbound()
        after = self._attitude_audit_count()
        attitude_receipts = [
            value
            for value in receipts
            if value.get("schema") == "aigp-vq2-attitude-target-outbound/1"
        ]
        receipt = attitude_receipts[-1] if attitude_receipts else None
        if receipt is not None:
            call_started = receipt["call_start_monotonic_ns"]
            call_ended = receipt["call_end_monotonic_ns"]
            if receipt["outcome"] != "returned" and error is None:
                error = RuntimeError("adapter recorded a raised attitude call")
        if after != before + 1 and error is None:
            error = CalibrationEvidenceError(
                "attitude-target audit did not increment exactly once"
            )
        return CalibrationDispatchResult(
            audit_count_before=before,
            audit_count_after=after,
            receipt=receipt,
            call_started_monotonic_ns=call_started,
            call_ended_monotonic_ns=call_ended,
            error=error,
        )

    async def dispatch_cleanup_zero(
        self,
        deadline_monotonic_ns: int,
    ) -> CalibrationDispatchResult:
        """Call the one cleanup-authorized exact-zero rate API once."""

        if type(deadline_monotonic_ns) is not int or self._now() >= deadline_monotonic_ns:
            raise CalibrationCheckFailure(
                "deadline_expired", "cleanup-zero deadline was reached"
            )
        self.drain_outbound()
        before = self._attitude_audit_count()
        call_started = self._now()
        error: Optional[BaseException] = None
        try:
            await self.adapter.send_attitude_rate(
                AttitudeRateCommand(0.0, 0.0, 0.0, 0.0),
                powered_deadline_monotonic_ns=deadline_monotonic_ns,
                powered_cleanup=True,
            )
        except BaseException as exc:
            error = exc
        call_ended = self._now()
        receipts = self.drain_outbound()
        after = self._attitude_audit_count()
        candidates = [
            item
            for item in receipts
            if item.get("schema") == "aigp-vq2-attitude-target-outbound/1"
        ]
        receipt = candidates[-1] if candidates else None
        if receipt is not None:
            call_started = receipt["call_start_monotonic_ns"]
            call_ended = receipt["call_end_monotonic_ns"]
            if receipt["outcome"] != "returned" and error is None:
                error = RuntimeError("adapter recorded a raised cleanup-zero call")
        if after != before + 1 and error is None:
            error = CalibrationEvidenceError(
                "cleanup-zero attitude audit did not increment exactly once"
            )
        return CalibrationDispatchResult(
            audit_count_before=before,
            audit_count_after=after,
            receipt=receipt,
            call_started_monotonic_ns=call_started,
            call_ended_monotonic_ns=call_ended,
            error=error,
        )

    async def dispatch_nonattitude(
        self,
        category: str,
        deadline_monotonic_ns: int,
        *,
        cleanup: bool,
        persist_boundary: Optional[Callable[[Any], None]] = None,
        progress: Optional[Callable[[], None]] = None,
    ) -> CalibrationNonattitudeDispatch:
        """Dispatch one arm/disarm/reset and bind it to its exact receipt."""

        if category not in {"arm", "disarm", "sim_reset"}:
            raise ValueError("unsupported calibration nonattitude category")
        if type(cleanup) is not bool:
            raise TypeError("cleanup must be an exact boolean")
        if not cleanup:
            if self.parent_alive() is not True:
                raise CalibrationCheckFailure("parent_dead", "wrapper parent is not live")
            if self.lease_valid() is not True:
                raise CalibrationCheckFailure("lease_invalid", "powered lease is invalid")
        if type(deadline_monotonic_ns) is not int or self._now() >= deadline_monotonic_ns:
            raise CalibrationCheckFailure(
                "deadline_expired", f"{category} deadline was reached"
            )
        if category == "sim_reset" and not callable(persist_boundary):
            raise TypeError("sim_reset requires a boundary persistence callback")
        if category != "sim_reset" and persist_boundary is not None:
            raise TypeError("boundary persistence is valid only for sim_reset")
        if progress is not None and (
            not callable(progress) or category != "sim_reset" or not cleanup
        ):
            raise TypeError(
                "cooperative progress is valid only for cleanup sim_reset"
            )

        self.drain_outbound()
        before = self._audit_count(category)
        request_ns = self._now()
        call_started: Optional[int] = request_ns
        call_ended: Optional[int] = None
        boundary: Optional[Any] = None
        error: Optional[BaseException] = None
        try:
            if category == "sim_reset":
                reset_kwargs: Dict[str, Any] = {
                    "powered_deadline_monotonic_ns": deadline_monotonic_ns,
                    "powered_cleanup": cleanup,
                }
                if progress is not None:
                    parameters = inspect.signature(
                        self.adapter.reset_calibration_with_boundary
                    ).parameters
                    if "powered_progress" not in parameters:
                        raise CalibrationEvidenceError(
                            "cleanup reset lacks same-call parent supervision"
                        )
                    reset_kwargs["powered_progress"] = progress
                boundary = await self.adapter.reset_calibration_with_boundary(
                    persist_boundary,
                    **reset_kwargs,
                )
            else:
                method = getattr(self.adapter, category)
                await method(
                    powered_deadline_monotonic_ns=deadline_monotonic_ns,
                    powered_cleanup=cleanup,
                )
        except BaseException as exc:
            error = exc
        call_ended = self._now()
        receipts = self.drain_outbound()
        after = self._audit_count(category)
        candidates = [
            item
            for item in receipts
            if item.get("schema") == "aigp-vq2-nonattitude-outbound/1"
            and item.get("category") == category
        ]
        receipt = candidates[-1] if candidates else None
        if receipt is not None:
            call_started = receipt["call_start_monotonic_ns"]
            call_ended = receipt["call_end_monotonic_ns"]
            if receipt["outcome"] != "returned" and error is None:
                error = RuntimeError(
                    f"adapter recorded a raised {category} call"
                )
        if after != before + 1 and error is None:
            error = CalibrationEvidenceError(
                f"{category} audit did not increment exactly once"
            )
        return CalibrationNonattitudeDispatch(
            category=category,
            request_monotonic_ns=request_ns,
            receipt=receipt,
            audit_count_before=before,
            audit_count_after=after,
            call_started_monotonic_ns=call_started,
            call_ended_monotonic_ns=call_ended,
            boundary=boundary,
            error=error,
        )

    def begin_live_cleanup(self) -> None:
        """Permanently latch production and consume the one live cleanup epoch."""

        guards = getattr(self.adapter, "powered_outbound_guards", None)
        if guards is None:
            raise CalibrationEvidenceError("powered outbound guards are unavailable")
        parent_alive = self.parent_alive()
        lease_valid = self.lease_valid()
        source_promoted = getattr(self.adapter, "powered_source_promoted", None)
        if type(parent_alive) is not bool or type(lease_valid) is not bool:
            raise CalibrationEvidenceError("cleanup lineage callbacks are not exact booleans")
        if type(source_promoted) is not bool:
            raise CalibrationEvidenceError("source-promotion state is not an exact boolean")
        guards.enable_cleanup_live(
            parent_alive=parent_alive,
            lease_valid=lease_valid,
            source_promoted=source_promoted,
        )

    def begin_takeover_cleanup(self) -> None:
        """Enable the same single cleanup epoch after proved parent death."""

        guards = getattr(self.adapter, "powered_outbound_guards", None)
        if guards is None:
            raise CalibrationEvidenceError("powered outbound guards are unavailable")
        parent_alive = self.parent_alive()
        lease_valid = self.lease_valid()
        source_promoted = getattr(self.adapter, "powered_source_promoted", None)
        if type(parent_alive) is not bool or type(lease_valid) is not bool:
            raise CalibrationEvidenceError("takeover lineage callbacks are not exact booleans")
        if type(source_promoted) is not bool:
            raise CalibrationEvidenceError("source-promotion state is not an exact boolean")
        guards.note_parent_death()
        guards.enable_cleanup_takeover(
            parent_signaled=not parent_alive,
            abandoned_lease_owned=lease_valid,
            authority_valid=lease_valid,
            source_promoted=source_promoted,
        )

    async def disconnect(
        self,
        deadline_monotonic_ns: int,
        *,
        progress: Optional[Callable[[], None]] = None,
    ) -> None:
        if type(deadline_monotonic_ns) is not int:
            raise TypeError("disconnect deadline must be an exact integer")
        kwargs: Dict[str, Any] = {
            "deadline_monotonic_ns": deadline_monotonic_ns,
        }
        parameters = inspect.signature(self.adapter.disconnect).parameters
        if "powered_progress" in parameters:
            kwargs["powered_progress"] = progress
        await self.adapter.disconnect(**kwargs)


class CalibrationExcitationScheduler:
    """Exact 245-slot, no-catch-up calibration excitation scheduler."""

    def __init__(
        self,
        *,
        evidence: CalibrationCommandEvidence,
        safety_check: Callable[[int], CalibrationSafetyAuthorization],
        dispatch: Callable[
            [AttitudeRateCommand, int], Awaitable[CalibrationDispatchResult]
        ],
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        wait_until_ns: Optional[Callable[[int], Awaitable[None]]] = None,
        powered_parent_deadline_monotonic_ns: Optional[int] = None,
        contract: Any = None,
    ) -> None:
        if not isinstance(evidence, CalibrationCommandEvidence):
            raise TypeError("evidence must be CalibrationCommandEvidence")
        if not callable(safety_check) or not callable(dispatch):
            raise TypeError("scheduler safety_check and dispatch must be callable")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        if wait_until_ns is not None and not callable(wait_until_ns):
            raise TypeError("wait_until_ns must be callable or None")
        if contract is None:
            contract = evidence.contract
        self.contract = contract
        self.evidence = evidence
        self.safety_check = safety_check
        self.dispatch = dispatch
        self.monotonic_ns = monotonic_ns
        self.wait_until_ns = wait_until_ns or self._default_wait_until
        if powered_parent_deadline_monotonic_ns is not None and (
            type(powered_parent_deadline_monotonic_ns) is not int
            or powered_parent_deadline_monotonic_ns < 1
        ):
            raise ValueError("powered parent deadline must be a positive exact integer")
        self.powered_parent_deadline_monotonic_ns = (
            powered_parent_deadline_monotonic_ns
        )

    async def _default_wait_until(self, deadline_ns: int) -> None:
        now = self._now()
        if now < deadline_ns:
            await asyncio.sleep((deadline_ns - now) / 1_000_000_000.0)

    def _now(self) -> int:
        value = self.monotonic_ns()
        if type(value) is not int or value < 0:
            raise CalibrationEvidenceError(
                "scheduler monotonic clock must return a nonnegative exact integer"
            )
        return value

    def _tick(self, index: int, anchor_ns: int) -> Dict[str, Any]:
        return self.contract.excitation_tick(
            index, anchor_monotonic_ns=anchor_ns
        )

    def _skip_before(
        self,
        tick: Mapping[str, Any],
        reason_code: str,
        recorded_ns: int,
    ) -> None:
        self.evidence.record_tick_disposition(
            tick,
            recorded_monotonic_ns=recorded_ns,
            disposition="skipped_before_generation",
            generated_event_sequence=None,
            terminal_event_sequence=None,
            reason_code=reason_code,
        )

    def _skip_remaining(
        self,
        start: int,
        anchor_ns: int,
        reason_code: str,
        recorded_ns: int,
        skipped_before: List[int],
    ) -> None:
        for index in range(start, self.contract.frozen_excitation_plan()["tick_count"]):
            self._skip_before(self._tick(index, anchor_ns), reason_code, recorded_ns)
            skipped_before.append(index)

    async def run(self) -> CalibrationScheduleResult:
        plan = self.contract.frozen_excitation_plan()
        tick_count = plan["tick_count"]
        period_ns = plan["control_period_ns"]

        # No powered clock anchor exists until a complete initial gate passes.
        self.safety_check(0)
        anchor_ns = self._now()
        hard_expiry_ns = anchor_ns + plan["powered_hard_expiry_offset_ns"]
        authority_expiry_ns = (
            hard_expiry_ns
            if self.powered_parent_deadline_monotonic_ns is None
            else min(hard_expiry_ns, self.powered_parent_deadline_monotonic_ns)
        )
        sent: List[int] = []
        skipped_before: List[int] = []
        skipped_after: List[int] = []
        cursor = 0

        while cursor < tick_count:
            now_ns = self._now()
            if now_ns >= authority_expiry_ns:
                self._skip_remaining(
                    cursor,
                    anchor_ns,
                    "deadline_expired",
                    now_ns,
                    skipped_before,
                )
                return CalibrationScheduleResult(
                    anchor_ns,
                    hard_expiry_ns,
                    tuple(sent),
                    tuple(skipped_before),
                    tuple(skipped_after),
                    False,
                    "deadline_expired",
                )
            current = (now_ns - anchor_ns) // period_ns
            if current < 0:
                await self.wait_until_ns(anchor_ns)
                continue
            if current >= tick_count:
                self._skip_remaining(
                    cursor,
                    anchor_ns,
                    "slot_missed",
                    now_ns,
                    skipped_before,
                )
                break
            while cursor < current:
                self._skip_before(
                    self._tick(cursor, anchor_ns), "slot_missed", now_ns
                )
                skipped_before.append(cursor)
                cursor += 1
            if cursor >= tick_count:
                break
            tick = self._tick(cursor, anchor_ns)
            if now_ns < tick["release_monotonic_ns"]:
                await self.wait_until_ns(tick["release_monotonic_ns"])
                continue
            if now_ns >= tick["end_monotonic_ns"]:
                self._skip_before(tick, "slot_missed", now_ns)
                skipped_before.append(cursor)
                cursor += 1
                continue

            try:
                authorization = self.safety_check(cursor)
            except CalibrationCheckFailure as exc:
                failed_ns = self._now()
                self._skip_before(tick, exc.reason_code, failed_ns)
                skipped_before.append(cursor)
                self._skip_remaining(
                    cursor + 1,
                    anchor_ns,
                    exc.reason_code,
                    failed_ns,
                    skipped_before,
                )
                return CalibrationScheduleResult(
                    anchor_ns,
                    hard_expiry_ns,
                    tuple(sent),
                    tuple(skipped_before),
                    tuple(skipped_after),
                    False,
                    exc.reason_code,
                )
            checked_ns = self._now()
            if checked_ns >= authority_expiry_ns:
                self._skip_before(tick, "deadline_expired", checked_ns)
                skipped_before.append(cursor)
                self._skip_remaining(
                    cursor + 1,
                    anchor_ns,
                    "deadline_expired",
                    checked_ns,
                    skipped_before,
                )
                return CalibrationScheduleResult(
                    anchor_ns,
                    hard_expiry_ns,
                    tuple(sent),
                    tuple(skipped_before),
                    tuple(skipped_after),
                    False,
                    "deadline_expired",
                )
            recomputed = (checked_ns - anchor_ns) // period_ns
            if recomputed != cursor or checked_ns >= tick["end_monotonic_ns"]:
                self._skip_before(tick, "slot_missed", checked_ns)
                skipped_before.append(cursor)
                cursor += 1
                continue

            generated_ns = self._now()
            if generated_ns >= tick["end_monotonic_ns"]:
                self._skip_before(tick, "slot_missed", generated_ns)
                skipped_before.append(cursor)
                cursor += 1
                continue
            generated = self.evidence.record_generated(
                tick, authorization, generated_ns
            )

            # Re-run every safety predicate after evidence enqueue and directly
            # before dispatch.  The sent/not-sent event must still copy the
            # first generated observation byte-for-byte.
            try:
                self.safety_check(cursor)
                presend_ns = self._now()
                if presend_ns >= authority_expiry_ns:
                    raise CalibrationCheckFailure(
                        "deadline_expired", "powered deadline reached before send"
                    )
                if (
                    (presend_ns - anchor_ns) // period_ns != cursor
                    or presend_ns >= tick["end_monotonic_ns"]
                ):
                    raise CalibrationCheckFailure(
                        "slot_missed", "generated tick crossed its half-open slot"
                    )
            except CalibrationCheckFailure as exc:
                terminal_ns = self._now()
                terminal = self.evidence.record_not_sent(
                    generated,
                    recorded_monotonic_ns=terminal_ns,
                    reason_code=exc.reason_code,
                    detail=exc.detail,
                )
                self.evidence.record_tick_disposition(
                    tick,
                    recorded_monotonic_ns=self._now(),
                    disposition="skipped_after_generation",
                    generated_event_sequence=generated["event_sequence"],
                    terminal_event_sequence=terminal["event_sequence"],
                    reason_code=exc.reason_code,
                )
                skipped_after.append(cursor)
                if exc.reason_code != "slot_missed":
                    self._skip_remaining(
                        cursor + 1,
                        anchor_ns,
                        exc.reason_code,
                        terminal_ns,
                        skipped_before,
                    )
                    return CalibrationScheduleResult(
                        anchor_ns,
                        hard_expiry_ns,
                        tuple(sent),
                        tuple(skipped_before),
                        tuple(skipped_after),
                        False,
                        exc.reason_code,
                    )
                cursor += 1
                continue

            command_value = tick["command"]
            command = AttitudeRateCommand(
                command_value["roll_rate_rad_s"],
                command_value["pitch_rate_rad_s"],
                command_value["yaw_rate_rad_s"],
                command_value["thrust"],
            )
            validate_command(command)
            try:
                dispatch_result = await self.dispatch(
                    command,
                    min(
                        tick["end_monotonic_ns"],
                        authority_expiry_ns,
                    ),
                )
                if not isinstance(dispatch_result, CalibrationDispatchResult):
                    raise TypeError("dispatch returned an invalid result")
            except BaseException as exc:
                # A dispatcher that raises cannot prove whether its adapter call
                # began.  Preserve the conservative no-call uncertainty shape.
                dispatch_result = CalibrationDispatchResult(
                    audit_count_before=0,
                    audit_count_after=0,
                    receipt=None,
                    call_started_monotonic_ns=None,
                    call_ended_monotonic_ns=None,
                    error=exc,
                )
            receipt_returned = bool(
                dispatch_result.receipt is not None
                and dispatch_result.receipt.get("outcome") == "returned"
            )
            if dispatch_result.error is not None or not receipt_returned:
                terminal_ns = self._now()
                if isinstance(dispatch_result.error, CalibrationCheckFailure):
                    reason = dispatch_result.error.reason_code
                    if reason not in {
                        "deadline_expired",
                        "parent_dead",
                        "lease_invalid",
                        "internal_error",
                    }:
                        reason = "internal_error"
                else:
                    reason = (
                        "send_raised"
                        if dispatch_result.call_started_monotonic_ns is not None
                        else "internal_error"
                    )
                terminal = self.evidence.record_not_sent(
                    generated,
                    recorded_monotonic_ns=terminal_ns,
                    reason_code=reason,
                    detail="attitude-target dispatch failed or is uncertain",
                    dispatch=dispatch_result,
                )
                self.evidence.record_tick_disposition(
                    tick,
                    recorded_monotonic_ns=self._now(),
                    disposition="skipped_after_generation",
                    generated_event_sequence=generated["event_sequence"],
                    terminal_event_sequence=terminal["event_sequence"],
                    reason_code=reason,
                )
                skipped_after.append(cursor)
                self._skip_remaining(
                    cursor + 1,
                    anchor_ns,
                    reason,
                    terminal_ns,
                    skipped_before,
                )
                return CalibrationScheduleResult(
                    anchor_ns,
                    hard_expiry_ns,
                    tuple(sent),
                    tuple(skipped_before),
                    tuple(skipped_after),
                    False,
                    reason,
                )
            receipt_start = dispatch_result.receipt.get(
                "call_start_monotonic_ns"
            )
            receipt_started_in_slot = bool(
                type(receipt_start) is int
                and tick["release_monotonic_ns"] <= receipt_start
                < tick["end_monotonic_ns"]
                and receipt_start < authority_expiry_ns
            )
            sent_ns = self._now()
            terminal = self.evidence.record_sent(
                generated,
                sent_monotonic_ns=sent_ns,
                dispatch=dispatch_result,
            )
            self.evidence.record_tick_disposition(
                tick,
                recorded_monotonic_ns=self._now(),
                disposition="sent",
                generated_event_sequence=generated["event_sequence"],
                terminal_event_sequence=terminal["event_sequence"],
                reason_code=None,
            )
            sent.append(cursor)
            cursor += 1
            if not receipt_started_in_slot:
                # The returned receipt proves a call happened, so retain the
                # truthful sent pair.  Its out-of-slot start invalidates the
                # stage and no later tick may be dispatched.
                self._skip_remaining(
                    cursor,
                    anchor_ns,
                    "slot_missed",
                    sent_ns,
                    skipped_before,
                )
                return CalibrationScheduleResult(
                    anchor_ns,
                    hard_expiry_ns,
                    tuple(sent),
                    tuple(skipped_before),
                    tuple(skipped_after),
                    False,
                    "slot_missed",
                )

        return CalibrationScheduleResult(
            anchor_ns,
            hard_expiry_ns,
            tuple(sent),
            tuple(skipped_before),
            tuple(skipped_after),
            True,
            None,
        )


@dataclass(frozen=True)
class CalibrationLeaseProof:
    """One wrapper delegation or powered-child abandoned-lease proof."""

    owner_role: str
    generation: int
    record_sha256: str
    authority_valid: bool
    takeover_completed_monotonic_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.owner_role not in {"wrapper", "powered-child-parent-death"}:
            raise ValueError("calibration lease owner role is invalid")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("calibration lease generation must be nonnegative")
        if (
            type(self.record_sha256) is not str
            or len(self.record_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.record_sha256)
        ):
            raise ValueError("calibration lease record hash is invalid")
        if type(self.authority_valid) is not bool:
            raise TypeError("calibration lease authority must be an exact boolean")
        if self.takeover_completed_monotonic_ns is not None and (
            type(self.takeover_completed_monotonic_ns) is not int
            or self.takeover_completed_monotonic_ns < 0
        ):
            raise ValueError("calibration takeover completion is invalid")


class CalibrationLeaseBoundary(Protocol):
    def prove_live_delegation(
        self,
        *,
        attempt: Mapping[str, Any],
        process_authority: Mapping[str, Any],
    ) -> Any: ...

    def take_over_abandoned(
        self,
        *,
        role_secret: memoryview,
        attempt: Mapping[str, Any],
        process_authority: Mapping[str, Any],
        deadline_monotonic_ns: int,
    ) -> Any: ...

    def heartbeat_takeover(
        self,
        proof: Any,
        *,
        phase: str,
        deadline_monotonic_ns: int,
    ) -> Any: ...

    def release_takeover(
        self,
        proof: Any,
        *,
        deadline_monotonic_ns: int,
    ) -> bool: ...


class CalibrationCertificatePublisher(Protocol):
    def publish_create_new(
        self,
        path: str,
        value: Mapping[str, Any],
        *,
        deadline_monotonic_ns: int,
        progress_callback: Callable[[], None],
    ) -> str: ...


@dataclass(frozen=True)
class CalibrationClosedArtifacts:
    legacy_record: Mapping[str, Any]
    replay_bundle: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationChildRunOutput:
    certificate: Optional[Dict[str, Any]]
    certificate_sha256: Optional[str]
    process_result: Dict[str, Any]
    exit_code: int


@dataclass
class CalibrationChildServices:
    """Post-admission construction and proof seams for the powered child.

    Both factories are called only after capability/identity admission and a
    live lease proof.  The adapter factory owns the exclusive MAVLink bind;
    the vision factory receives the non-overridable exclusive camera options.
    """

    process_boundary: CalibrationProcessBoundary
    lease_boundary: CalibrationLeaseBoundary
    recorder_factory: Callable[[CalibrationAdmission], "JsonlRecorder"]
    adapter_factory: Callable[..., Any]
    vision_factory: Callable[..., Any]
    camera_socket_factory: Callable[[str, int], Any]
    publisher: CalibrationCertificatePublisher
    monotonic_ns: Callable[[], int]
    wait_until_ns: Optional[Callable[[int], Awaitable[None]]] = None
    detector_factory: Callable[[], Any] = VQ2GateDetector
    endpoint_evidence: Optional[
        Callable[[Any, Any, CalibrationAdmission], Mapping[str, Any]]
    ] = None
    transport_evidence: Optional[
        Callable[[Any, Any, Any], Mapping[str, Any]]
    ] = None
    artifact_closer: Optional[
        Callable[
            ["JsonlRecorder", CalibrationAdmission, Mapping[str, Any], int],
            Any,
        ]
    ] = None
    contract: Any = None
    runtime: Any = None


@dataclass
class _CalibrationObservationState:
    heartbeat: Optional[Dict[str, Any]] = None
    race: Optional[Dict[str, Any]] = None
    imu: Optional[Dict[str, Any]] = None
    actuator: Optional[Dict[str, Any]] = None
    heartbeat_object: Any = None
    race_object: Any = None
    imu_object: Any = None
    actuator_object: Any = None
    generation: Optional[int] = None
    sequence: Optional[int] = None
    imu_advance_monotonic_ns: int = 0
    race_advance_monotonic_ns: int = 0
    last_imu_timestamp_us: Optional[int] = None
    last_race_boot_ms: Optional[int] = None
    imu_regressed: bool = False
    race_regressed: bool = False

    def begin_generation(self, generation: int) -> None:
        if type(generation) is not int or generation < 0:
            raise CalibrationEvidenceError("reset generation is invalid")
        self.heartbeat = None
        self.race = None
        self.imu = None
        self.actuator = None
        self.heartbeat_object = None
        self.race_object = None
        self.imu_object = None
        self.actuator_object = None
        self.generation = generation
        self.sequence = None
        self.imu_advance_monotonic_ns = 0
        self.race_advance_monotonic_ns = 0
        self.last_imu_timestamp_us = None
        self.last_race_boot_ms = None
        self.imu_regressed = False
        self.race_regressed = False

    def accept(
        self,
        checked: Mapping[str, Any],
        original: Any,
        *,
        allow_clock_rollback: bool,
    ) -> str:
        ingress = checked["ingress"]
        generation = ingress["generation"]
        sequence = ingress["sequence"]
        if self.generation is None:
            self.generation = generation
        if generation != self.generation:
            raise CalibrationEvidenceError(
                "received envelope escaped the active reset generation"
            )
        if self.sequence is not None and sequence <= self.sequence:
            raise CalibrationEvidenceError(
                "received envelope sequence is not strictly increasing"
            )
        self.sequence = sequence
        schema = checked["schema"]
        if schema == "aigp-vq2-received-heartbeat/1":
            self.heartbeat = dict(checked)
            self.heartbeat_object = original
            return "heartbeat"
        if schema == "aigp-vq2-received-race-status/1":
            boot = checked["race_status"]["sim_boot_time_ms"]
            if self.last_race_boot_ms is None or boot > self.last_race_boot_ms:
                self.last_race_boot_ms = boot
                self.race_advance_monotonic_ns = ingress["received_monotonic_ns"]
            elif boot < self.last_race_boot_ms and not allow_clock_rollback:
                self.race_regressed = True
            self.race = dict(checked)
            self.race_object = original
            return "race"
        if schema == "aigp-vq2-received-imu/1":
            timestamp = checked["imu"]["timestamp_us"]
            if self.last_imu_timestamp_us is None or timestamp > self.last_imu_timestamp_us:
                self.last_imu_timestamp_us = timestamp
                self.imu_advance_monotonic_ns = ingress["received_monotonic_ns"]
            elif timestamp < self.last_imu_timestamp_us and not allow_clock_rollback:
                self.imu_regressed = True
            self.imu = dict(checked)
            self.imu_object = original
            return "imu"
        if schema == "aigp-vq2-received-actuator-output-status/1":
            self.actuator = dict(checked)
            self.actuator_object = original
            return "actuator"
        raise CalibrationEvidenceError("unsupported received observation")


class CalibrationChildLifecycle:
    """One admitted, bounded, fail-closed powered-child state machine."""

    _PHASE_DURATION_KEYS = {
        "connect": "child_connect",
        "preflight": "child_preflight",
        "reset_epoch": "child_reset_epoch",
        "normalize_disarmed": "child_normalize_disarmed",
        "countdown_go": "child_countdown_go",
        "arm": "child_arm",
        "powered_stage": "powered_stage",
        "cleanup": "child_cleanup",
        "replay_close": "child_replay_close",
        "finalize": "child_finalize",
        "parent_death_lease_takeover": "parent_death_lease_takeover",
    }
    _PHASE_PARENT = {
        "connect": "prepower",
        "preflight": "prepower",
        "reset_epoch": "prepower",
        "normalize_disarmed": "prepower",
        "countdown_go": "prepower",
        "arm": "prepower",
        "powered_stage": "powered",
        "cleanup": "cleanup",
        "parent_death_lease_takeover": "cleanup",
        "replay_close": "replay_close",
        "finalize": "exit",
    }
    _ZERO_COMMAND = {
        "roll_rate_rad_s": 0.0,
        "pitch_rate_rad_s": 0.0,
        "yaw_rate_rad_s": 0.0,
        "thrust": 0.0,
    }

    def __init__(
        self,
        admission: CalibrationAdmission,
        services: CalibrationChildServices,
    ) -> None:
        if not isinstance(admission, CalibrationAdmission):
            raise TypeError("admission must be CalibrationAdmission")
        if not isinstance(services, CalibrationChildServices):
            raise TypeError("services must be CalibrationChildServices")
        if not callable(services.monotonic_ns):
            raise TypeError("calibration lifecycle clock must be callable")
        self.admission = admission
        self.services = services
        if services.contract is None:
            from scripts import aigp_vq2_powered_attempt as contract_module

            self.contract = contract_module
        else:
            self.contract = services.contract
        if services.runtime is None:
            from scripts import aigp_vq2_powered_runtime as runtime_module

            self.runtime = runtime_module
        else:
            self.runtime = services.runtime
        self.durations = admission.attempt["context"]["deadline_durations_ns"]
        self.phase_deadlines: List[Dict[str, Any]] = []
        self.evidence = CalibrationCommandEvidence(
            attempt=admission.attempt,
            recorder=None,
            contract=self.contract,
        )
        self.recorder: Optional[JsonlRecorder] = None
        self.lineage: Optional[CalibrationLineageRecorder] = None
        self.capture: Optional[CalibrationSnapshotCapture] = None
        self.guards: Any = None
        self.adapter: Any = None
        self.vision: Any = None
        self._vision_start_attempted = False
        self.dispatcher: Optional[CalibrationAdapterDispatcher] = None
        self.detector: Any = None
        self.tracker = GateTargetTracker()
        self.estimator = self._new_estimator()
        self.estimate: Optional[AttitudeEstimate] = None
        self.state = _CalibrationObservationState()
        self.frame: Optional[Dict[str, Any]] = None
        self.target: Optional[GateTarget] = None
        self._last_frame_identity: Optional[Tuple[int, int]] = None
        self._countdown_observed = False
        self._reset_epoch: Optional[Dict[str, Any]] = None
        self._start_roll_rad: Optional[float] = None
        self._start_pitch_rad: Optional[float] = None
        self._initial_target_area: Optional[float] = None
        self._epoch_collision_count = 0
        self._collision_recorded_by_generation: Dict[int, int] = {}
        self._adapter_capture_failure: Optional[str] = None
        self._lease_proof: Optional[CalibrationLeaseProof] = None
        self._lease_boundary_proof: Any = None
        self._parent_mode = "live_delegation"
        self._parent_observed_ns = admission.admitted_monotonic_ns
        self._parent_death_observed = False
        self._takeover_completed_ns: Optional[int] = None
        self._takeover_record_sha256: Optional[str] = None
        self._takeover_last_heartbeat_ns: Optional[int] = None
        self._pending_takeover_phase: Optional[Dict[str, Any]] = None
        self._takeover_attempted = False
        self._takeover_release_attempted = False
        self._takeover_released = True
        self._cleanup_started = False
        self._cleanup_phase: Optional[Dict[str, Any]] = None
        self._certificate_published = False
        self._replay_closed = False
        self._stage_completed = False
        self._trigger = "stage_abort"
        self._reason_codes: set[str] = set()
        self._cleanup_failures: set[str] = set()
        self._collection_codes: set[str] = set()
        self._zero_command = self._zero_state(required=False)
        self._disarm = self._disarm_state("not_required")
        self._reset = self._reset_state("not_required")
        self._final_state = self._unobserved_final_state()
        self._certificate: Optional[Dict[str, Any]] = None
        self._certificate_sha256: Optional[str] = None
        self._certificate_reference_state = "absent"
        self._certificate_reference_sha256: Optional[str] = None
        self._artifacts = self._partial_artifacts("absent")

    @staticmethod
    def _new_estimator() -> ImuAttitudeEstimator:
        config = ImuAttitudeConfig(
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
        return ImuAttitudeEstimator(config)

    @classmethod
    def _zero_state(cls, *, required: bool) -> Dict[str, Any]:
        return {
            "state": "not_attempted" if required else "not_required",
            "required": required,
            "requested": dict(cls._ZERO_COMMAND) if required else None,
            "generated": None,
            "terminal": None,
            "outbound_receipt": None,
        }

    @staticmethod
    def _disarm_state(state: str) -> Dict[str, Any]:
        return {
            "state": state,
            "request_monotonic_ns": None,
            "receipt": None,
            "heartbeat_before": None,
            "heartbeat_after": None,
            "newer_confirmed": False,
        }

    @staticmethod
    def _reset_state(state: str) -> Dict[str, Any]:
        return {
            "state": state,
            "request_monotonic_ns": None,
            "receipt": None,
            "boundary": None,
            "baseline": None,
            "clean_epoch": None,
            "advancing_race": [],
            "advancing_imu": [],
            "rollback_and_advance_confirmed": False,
        }

    @staticmethod
    def _unobserved_final_state() -> Dict[str, Any]:
        return {
            "state": "unobserved",
            "heartbeat": None,
            "disarmed": None,
            "reset_epoch": None,
            "last_race": None,
            "last_imu": None,
        }

    def _partial_artifacts(self, state: str) -> Dict[str, Any]:
        return {
            "legacy_record": {
                "path": self.admission.arguments.record,
                "state": state,
                "sha256": None,
            },
            "replay_bundle": {
                "path": self.admission.arguments.replay_bundle,
                "state": state,
                "dataset_hash": None,
                "manifest_sha256": None,
                "records_sha256": None,
            },
        }

    def _now(self) -> int:
        value = self.services.monotonic_ns()
        if type(value) is not int or value < 0:
            raise CalibrationEvidenceError(
                "calibration lifecycle clock must return a nonnegative exact integer"
            )
        return value

    def _poll_interval_ns(self) -> int:
        value = getattr(self.runtime, "MAX_POLL_INTERVAL_NS", 50_000_000)
        if type(value) is not int or value < 1:
            raise CalibrationEvidenceError(
                "calibration poll interval must be a positive exact integer"
            )
        return min(value, 50_000_000)

    def _parent_deadline(self, phase: str) -> int:
        name = self._PHASE_PARENT[phase]
        return {
            "prepower": self.admission.prepower_deadline_monotonic_ns,
            "powered": self.admission.powered_deadline_monotonic_ns,
            "cleanup": self.admission.cleanup_deadline_monotonic_ns,
            "replay_close": self.admission.replay_close_deadline_monotonic_ns,
            "exit": self.admission.exit_deadline_monotonic_ns,
        }[name]

    def _phase(
        self,
        phase: str,
        *,
        emit: bool = True,
        parent_deadline_monotonic_ns: Optional[int] = None,
    ) -> Dict[str, Any]:
        duration_key = self._PHASE_DURATION_KEYS.get(phase)
        if duration_key is None or duration_key not in self.durations:
            raise CalibrationEvidenceError("calibration phase duration is unavailable")
        frozen = self.runtime.freeze_phase_deadline(
            phase,
            self.durations[duration_key],
            (
                self._parent_deadline(phase)
                if parent_deadline_monotonic_ns is None
                else parent_deadline_monotonic_ns
            ),
            monotonic_ns=self.services.monotonic_ns,
        )
        row = frozen.to_primitive()
        self.contract.validate_phase_deadline(row, expected_phase=phase)
        self.phase_deadlines.append(dict(row))
        if emit:
            self.evidence.record_phase_deadline(row)
        return dict(row)

    def _parent_alive(self) -> bool:
        signaled = self.services.process_boundary.parent_signaled(
            self.admission.parent_handle
        )
        if type(signaled) is not bool:
            raise CalibrationEvidenceError("parent liveness is not an exact boolean")
        return not signaled

    def _note_parent_death(self) -> None:
        """Latch one permanent wrapper-death invalidation occurrence."""

        first_observation = not self._parent_death_observed
        if first_observation:
            self._parent_death_observed = True
            self._parent_observed_ns = self._now()
        self._trigger = "parent_death"
        self._reason_codes.add("wrapper_death")
        if (
            first_observation
            and self.guards is not None
            and self._parent_mode == "live_delegation"
        ):
            self.guards.note_parent_death()

    def _service_takeover_heartbeat(self, deadline_ns: int) -> None:
        """Publish each due takeover heartbeat from the mutex-owning thread."""

        if self._parent_mode != "signaled_takeover" or self._takeover_released:
            return
        proof = self._lease_proof
        boundary_proof = self._lease_boundary_proof
        last = self._takeover_last_heartbeat_ns
        if proof is None or boundary_proof is None or last is None:
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError("takeover heartbeat state is unavailable")
        now = self._now()
        period = self.durations["lease_heartbeat_period"]
        maximum_gap = self.durations["lease_heartbeat_max_gap"]
        if now - last > maximum_gap:
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError("takeover heartbeat maximum gap was exceeded")
        if now < last + period:
            return
        deadline = min(deadline_ns, self.admission.exit_deadline_monotonic_ns)
        if now >= deadline:
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError("takeover heartbeat deadline expired")
        heartbeat = getattr(self.services.lease_boundary, "heartbeat_takeover", None)
        if not callable(heartbeat):
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError("takeover heartbeat boundary is unavailable")
        try:
            refreshed_boundary_proof = heartbeat(
                boundary_proof,
                phase="child_cleanup",
                deadline_monotonic_ns=deadline,
            )
            refreshed = self._coerce_lease_proof(refreshed_boundary_proof)
        except BaseException as exc:
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError("takeover heartbeat failed") from exc
        if (
            refreshed.owner_role != "powered-child-parent-death"
            or refreshed.authority_valid is not True
            or refreshed.takeover_completed_monotonic_ns
            != proof.takeover_completed_monotonic_ns
            or refreshed.generation <= proof.generation
        ):
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError("takeover heartbeat proof is invalid")
        self._lease_proof = refreshed
        self._lease_boundary_proof = refreshed_boundary_proof
        self._takeover_record_sha256 = refreshed.record_sha256
        completed = self._now()
        if completed >= deadline or completed - last > maximum_gap:
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError(
                "takeover heartbeat completed outside its bounded cadence"
            )
        self._takeover_last_heartbeat_ns = completed

    def _supervise_parent(self, deadline_ns: int, *, cleanup: bool) -> None:
        """Observe the wrapper at every bounded poll boundary."""

        if self._parent_mode == "signaled_takeover":
            self._service_takeover_heartbeat(deadline_ns)
            return
        if self._parent_alive():
            return
        self._note_parent_death()
        if not cleanup:
            raise CalibrationCheckFailure("parent_dead", "wrapper parent is not live")
        self._takeover_if_parent_dead(
            parent_deadline_ns=deadline_ns,
            enable_cleanup=bool(
                self.guards is not None
                and self.adapter is not None
                and getattr(self.adapter, "powered_source_promoted", None) is True
                and getattr(self.guards, "cleanup_state", None) != "closed"
            ),
            emit=self.recorder is not None and not self._replay_closed,
        )

    def _lease_valid(self) -> bool:
        proof = self._lease_proof
        if proof is None or proof.authority_valid is not True:
            return False
        if self._parent_mode == "live_delegation":
            return proof.owner_role == "wrapper"
        return proof.owner_role == "powered-child-parent-death"

    @staticmethod
    def _coerce_lease_proof(value: Any) -> CalibrationLeaseProof:
        if isinstance(value, CalibrationLeaseProof):
            return value
        try:
            return CalibrationLeaseProof(
                owner_role=value.owner_role,
                generation=value.generation,
                record_sha256=value.record_sha256,
                authority_valid=value.authority_valid,
                takeover_completed_monotonic_ns=value.takeover_completed_monotonic_ns,
            )
        except BaseException as exc:
            raise CalibrationEvidenceError("calibration lease proof is invalid") from exc

    async def _wait_until(self, deadline_ns: int) -> None:
        callback = self.services.wait_until_ns
        if callback is None:
            now = self._now()
            if now < deadline_ns:
                await asyncio.sleep((deadline_ns - now) / 1_000_000_000.0)
            return
        result = callback(deadline_ns)
        if inspect.isawaitable(result):
            await result
        elif result is not None:
            raise CalibrationEvidenceError("wait_until_ns returned an invalid value")

    async def _poll_pause(self, deadline_ns: int) -> None:
        self._supervise_parent(deadline_ns, cleanup=self._cleanup_started)
        now = self._now()
        if now >= deadline_ns:
            raise CalibrationCheckFailure("deadline_expired", "phase deadline expired")
        interval = self._poll_interval_ns()
        await self._wait_until(min(deadline_ns, now + interval))
        self._supervise_parent(deadline_ns, cleanup=self._cleanup_started)

    async def _run_supervised_callable(
        self,
        callback: Callable[[], Any],
        *,
        deadline_ns: int,
        cleanup: bool,
        replay_close: bool = False,
    ) -> Any:
        """Run one bounded blocking seam while the owner thread supervises death."""

        async def invoke() -> Any:
            result = (
                callback()
                if inspect.iscoroutinefunction(callback)
                else await asyncio.to_thread(callback)
            )
            if inspect.isawaitable(result):
                return await result
            return result

        task = asyncio.create_task(invoke())
        supervision_error: Optional[BaseException] = None
        interval_ns = self._poll_interval_ns()
        while not task.done():
            try:
                if replay_close:
                    if self._parent_mode == "live_delegation" and not self._parent_alive():
                        self._freeze_pending_takeover(
                            parent_deadline_ns=self.admission.exit_deadline_monotonic_ns,
                            emit=False,
                        )
                elif supervision_error is None:
                    self._supervise_parent(deadline_ns, cleanup=cleanup)
            except BaseException as exc:
                supervision_error = supervision_error or exc
            now = self._now()
            if now >= deadline_ns:
                supervision_error = supervision_error or CalibrationCheckFailure(
                    "deadline_expired", "supervised operation deadline expired"
                )
            remaining_ns = deadline_ns - now
            wait_ns = (
                interval_ns
                if remaining_ns <= 0
                else min(interval_ns, remaining_ns)
            )
            try:
                await asyncio.wait(
                    {task},
                    timeout=wait_ns / 1_000_000_000.0,
                )
            except asyncio.CancelledError as exc:
                supervision_error = supervision_error or exc
                current = asyncio.current_task()
                if current is not None and callable(getattr(current, "uncancel", None)):
                    current.uncancel()
        try:
            result = task.result()
        except BaseException as exc:
            if supervision_error is not None:
                exc.add_note(
                    "supervision also failed: "
                    f"{type(supervision_error).__name__}: {supervision_error}"
                )
            raise
        if supervision_error is not None:
            raise supervision_error
        if replay_close:
            if self._parent_mode == "live_delegation" and not self._parent_alive():
                self._freeze_pending_takeover(
                    parent_deadline_ns=self.admission.exit_deadline_monotonic_ns,
                    emit=False,
                )
        else:
            self._supervise_parent(deadline_ns, cleanup=cleanup)
        return result

    def _replay_estimator_fields(self) -> Optional[Dict[str, Any]]:
        estimate = self.estimate
        if estimate is None:
            return None
        return {
            "timestamp_us": int(estimate.timestamp_us),
            "rpy_rad": list(estimate.orientation.to_euler()),
            "orientation_wxyz": [
                estimate.orientation.w,
                estimate.orientation.x,
                estimate.orientation.y,
                estimate.orientation.z,
            ],
            "body_rates": list(estimate.body_rates),
            "gyro_bias": list(estimate.gyro_bias),
            "healthy": bool(estimate.healthy),
            "reason": estimate.reason,
            "propagated": bool(estimate.propagated),
        }

    def _drain_received(
        self,
        *,
        update_estimator: bool,
        allow_clock_rollback: bool,
    ) -> List[Dict[str, Any]]:
        if self.adapter is None or self.lineage is None:
            return []
        drain = getattr(self.adapter, "drain_received_observations", None)
        if not callable(drain):
            raise CalibrationEvidenceError(
                "powered adapter lacks the strict received-envelope drain"
            )
        recorded: List[Dict[str, Any]] = []
        for observation in drain():
            primitive = CalibrationLineageRecorder._primitive(
                observation, "received observation"
            )
            schema = primitive.get("schema")
            validator_name = {
                "aigp-vq2-received-heartbeat/1": "validate_received_heartbeat",
                "aigp-vq2-received-race-status/1": "validate_received_race_status",
                "aigp-vq2-received-actuator-output-status/1": (
                    "validate_received_actuator_output_status"
                ),
                "aigp-vq2-received-imu/1": "validate_received_imu",
            }.get(schema)
            if validator_name is None:
                raise CalibrationEvidenceError("unsupported received-envelope schema")
            checked = getattr(self.contract, validator_name)(primitive)
            if self.state.generation is not None and (
                checked["ingress"]["generation"] != self.state.generation
            ):
                raise CalibrationEvidenceError(
                    "received envelope changed generation outside reset boundary"
                )
            if schema == "aigp-vq2-received-imu/1" and update_estimator:
                estimate = self.estimator.update_imu(observation.imu)
                if estimate is not None:
                    self.estimate = estimate
            recorded_row = self.lineage.record_received(
                observation,
                estimator=(
                    self._replay_estimator_fields()
                    if schema == "aigp-vq2-received-imu/1" and update_estimator
                    else None
                ),
            )
            self.state.accept(
                recorded_row,
                observation,
                allow_clock_rollback=allow_clock_rollback,
            )
            recorded.append(recorded_row)
        return recorded

    def _drain_collisions(self, *, phase: str, abort: bool) -> List[Dict[str, Any]]:
        if self.adapter is None or self.lineage is None:
            return []
        drain = getattr(self.adapter, "drain_collisions", None)
        if not callable(drain):
            raise CalibrationEvidenceError("powered adapter lacks collision drain")
        generation = 0 if self.state.generation is None else self.state.generation
        rows = [
            self.lineage.record_collision(
                value,
                reset_generation=generation,
                observed_monotonic_ns=self._now(),
                phase=phase,
                disposition="cleanup_continue" if not abort else "abort",
            )
            for value in drain()
        ]
        if rows:
            self._collision_recorded_by_generation[generation] = (
                self._collision_recorded_by_generation.get(generation, 0)
                + len(rows)
            )
        if rows:
            self._epoch_collision_count += len(rows)
            self._collection_codes.add("collision_observed")
            if abort:
                raise CalibrationCheckFailure(
                    "collision_observed", "every post-boundary collision aborts"
                )
        return rows

    def _latch_adapter_capture_failure(self, detail: str, *, abort: bool) -> None:
        if self._adapter_capture_failure is None:
            self._adapter_capture_failure = detail
        self._reason_codes.add("capture_incomplete")
        if abort:
            raise CalibrationCheckFailure("capture_failed", detail)

    def _verify_adapter_queues(self, *, abort: bool) -> None:
        if self.adapter is None:
            return
        ingress_method = getattr(self.adapter, "ingress_stats", None)
        collision_method = getattr(self.adapter, "collision_stats", None)
        if not callable(ingress_method) or not callable(collision_method):
            raise CalibrationEvidenceError(
                "powered adapter lacks ingress/collision diagnostics"
            )
        ingress = ingress_method()
        collision = collision_method()
        required_ingress = (
            "generation",
            "next_sequence",
            "dropped",
            "imu_dropped",
            "other_dropped",
            "buffered_imu",
            "buffered_other",
        )
        required_collision = ("generation", "handled", "dropped", "buffered")
        if any(type(getattr(ingress, name, None)) is not int for name in required_ingress):
            raise CalibrationEvidenceError("powered ingress diagnostics are invalid")
        if any(
            type(getattr(collision, name, None)) is not int
            for name in required_collision
        ):
            raise CalibrationEvidenceError("powered collision diagnostics are invalid")
        generation = self.state.generation
        problems: List[str] = []
        if generation is not None and (
            ingress.generation != generation or collision.generation != generation
        ):
            problems.append("diagnostic generation mismatched receiver state")
        if ingress.dropped or ingress.imu_dropped or ingress.other_dropped:
            problems.append("receiver ingress dropped observations")
        if ingress.buffered_imu or ingress.buffered_other:
            problems.append("receiver ingress remained buffered after strict drain")
        if (
            generation is not None
            and self.state.sequence is not None
            and ingress.next_sequence != self.state.sequence + 1
        ):
            problems.append("receiver ingress sequence accounting mismatched")
        if collision.dropped:
            problems.append("collision receiver dropped observations")
        if collision.buffered:
            problems.append("collision observations remained buffered after strict drain")
        if generation is not None and collision.handled != (
            self._collision_recorded_by_generation.get(generation, 0)
        ):
            problems.append("collision handled/recorded accounting mismatched")
        if problems:
            self._latch_adapter_capture_failure("; ".join(problems), abort=abort)

    def _drain_frames(self) -> int:
        if self.vision is None:
            return 0
        pop = getattr(self.vision, "pop_capture_snapshot", None)
        if not callable(pop):
            raise CalibrationEvidenceError("powered vision lacks capture FIFO drain")
        count = 0
        while True:
            snapshot = pop()
            if snapshot is None:
                break
            identity = (int(snapshot.generation), int(snapshot.frame_id))
            if identity == self._last_frame_identity:
                raise CalibrationEvidenceError("capture FIFO repeated a frame identity")
            image = snapshot.camera_frame.image
            height, width = image.shape[:2]
            timing = snapshot.timing.to_primitive()
            self.frame = {
                "stream_id": timing["identity"]["stream_id"],
                "generation": int(snapshot.generation),
                "frame_id": int(snapshot.frame_id),
                "sim_time_ns": int(snapshot.sim_time_ns),
                "timing": timing,
                "width": int(width),
                "height": int(height),
            }
            if self.state.generation is not None and (
                self.frame["generation"] != self.state.generation
            ):
                raise CalibrationCheckFailure(
                    "capture_failed", "camera frame is outside reset generation"
                )
            detections = list(self.detector.detect(image))
            self.target = self.tracker.update(
                detections,
                frame_id=snapshot.frame_id,
                sim_time_ns=snapshot.sim_time_ns,
                received_monotonic_s=snapshot.received_monotonic_s,
            )
            self._last_frame_identity = identity
            count += 1
        return count

    def _drain_all(
        self,
        *,
        phase: str,
        update_estimator: bool,
        collision_abort: bool,
        allow_clock_rollback: bool = False,
        frames: bool = True,
    ) -> None:
        self._drain_received(
            update_estimator=update_estimator,
            allow_clock_rollback=allow_clock_rollback,
        )
        if self.dispatcher is not None:
            self.dispatcher.drain_outbound()
        self._drain_collisions(phase=phase, abort=collision_abort)
        if frames:
            self._drain_frames()
        self._verify_adapter_queues(abort=collision_abort)

    def _prepare_recorder(self) -> None:
        recorder = self.services.recorder_factory(self.admission)
        if not isinstance(recorder, JsonlRecorder) or recorder.replay is None:
            raise CalibrationEvidenceError(
                "powered recorder factory must return active JsonlRecorder"
            )
        if recorder.capture_fifo_enabled is not True:
            raise CalibrationEvidenceError(
                "powered recorder must enable the decoded-frame FIFO"
            )
        self.recorder = recorder
        self.evidence.recorder = recorder
        self.lineage = CalibrationLineageRecorder(
            recorder,
            contract=self.contract,
        )
        target_config = self.admission.attempt["context"]["target_config"]
        config_sha256 = target_config["sha256"]
        self.capture = CalibrationSnapshotCapture(
            recorder=recorder,
            config_sha256=config_sha256,
            monotonic_ns=self.services.monotonic_ns,
            contract=self.contract,
        )
        self.detector = self.services.detector_factory()
        if not callable(getattr(self.detector, "detect", None)):
            raise CalibrationEvidenceError("detector factory returned an invalid detector")

    def _prove_live_delegation(self) -> None:
        if self._parent_alive() is not True:
            raise CalibrationCheckFailure("parent_dead", "wrapper parent is already dead")
        boundary_proof = self.services.lease_boundary.prove_live_delegation(
            attempt=self.admission.attempt,
            process_authority=self.admission.process_authority,
        )
        proof = self._coerce_lease_proof(boundary_proof)
        if (
            proof.owner_role != "wrapper"
            or proof.authority_valid is not True
        ):
            raise CalibrationCheckFailure(
                "lease_invalid", "wrapper lease delegation is invalid"
            )
        self._lease_proof = proof
        self._lease_boundary_proof = boundary_proof

    def _construct_transports(self) -> None:
        if self.capture is None or self.lineage is None:
            raise CalibrationEvidenceError("capture/lineage was not prepared")
        transport = self.admission.live_freeze.get("transport")
        if type(transport) is not dict:
            raise CalibrationEvidenceError("live-freeze transport is unavailable")

        def frozen_bind(name: str) -> Dict[str, Any]:
            value = transport.get(name)
            if type(value) is not dict or set(value) != {
                "host",
                "port",
                "socket_policy",
            }:
                raise CalibrationEvidenceError(
                    f"live-freeze {name} must be an exact bind object"
                )
            if (
                type(value["host"]) is not str
                or not value["host"]
                or type(value["port"]) is not int
                or not 1 <= value["port"] <= 65_535
                or value["socket_policy"] != "ipv4-exclusive-address-use"
            ):
                raise CalibrationEvidenceError(f"live-freeze {name} is invalid")
            return dict(value)

        mavlink_bind = frozen_bind("mavlink_bind")
        camera_bind = frozen_bind("camera_bind")
        self.guards = self.runtime.PoweredOutboundGuards()
        self.adapter = self.services.adapter_factory(
            admission=self.admission,
            bind=mavlink_bind,
            outbound_guards=self.guards,
            role_valid=lambda: True,
            parent_alive=self._parent_alive,
            lease_valid=self._lease_valid,
        )
        # Establish the close path immediately after adapter ownership begins.
        # Any later validation, camera construction, or reset failure then
        # still reaches the adapter's bounded disconnect path during cleanup.
        self.dispatcher = CalibrationAdapterDispatcher(
            self.adapter,
            self.lineage,
            monotonic_ns=self.services.monotonic_ns,
            parent_alive=self._parent_alive,
            lease_valid=self._lease_valid,
        )
        if getattr(self.adapter, "powered_outbound_guards", None) is not self.guards:
            raise CalibrationEvidenceError(
                "powered adapter did not retain the injected outbound guards"
            )
        for name, expected in (
            ("enable_vision", False),
            ("telemetry_mode", "imu"),
            ("fetch_track_on_connect", False),
        ):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) != expected:
                raise CalibrationEvidenceError(
                    f"powered adapter {name} construction option changed"
                )
        options = calibration_vision_options(
            self.capture,
            exclusive_socket_factory=self.services.camera_socket_factory,
        )
        self.vision = self.services.vision_factory(
            admission=self.admission,
            bind=camera_bind,
            **options,
        )
        # Adapter connect starts MAVLink generation one.  Advance the
        # independently owned camera receiver once so every later reset keeps
        # their generation tokens equal without synthesizing a mapping.
        reset = getattr(self.vision, "reset", None)
        if not callable(reset):
            raise CalibrationEvidenceError("powered vision lacks reset()")
        reset()

    async def _start_vision(self, deadline_ns: int) -> None:
        if self.vision is None:
            raise CalibrationEvidenceError("powered vision was not constructed")
        if self._now() >= deadline_ns:
            raise CalibrationCheckFailure("deadline_expired", "vision start deadline expired")
        self._vision_start_attempted = True
        await self._run_supervised_callable(
            self.vision.start,
            deadline_ns=deadline_ns,
            cleanup=False,
        )
        if self._now() >= deadline_ns:
            raise CalibrationCheckFailure("deadline_expired", "vision start completed late")

    async def _stop_vision(self, deadline_ns: int, *, cleanup: bool = False) -> None:
        if self.vision is None:
            return
        now = self._now()
        if now >= deadline_ns:
            raise CalibrationCheckFailure("deadline_expired", "vision stop deadline expired")
        timeout_s = (deadline_ns - now) / 1_000_000_000.0
        parameters = inspect.signature(self.vision.stop).parameters
        callback = (
            partial(self.vision.stop, timeout_s=timeout_s)
            if "timeout_s" in parameters or any(
                value.kind == inspect.Parameter.VAR_KEYWORD
                for value in parameters.values()
            )
            else self.vision.stop
        )
        await self._run_supervised_callable(
            callback,
            deadline_ns=deadline_ns,
            cleanup=cleanup,
        )
        if self._now() >= deadline_ns:
            raise CalibrationCheckFailure("deadline_expired", "vision stop completed late")

    async def _connect(self, phase: Mapping[str, Any]) -> None:
        self._prove_live_delegation()
        self._construct_transports()
        await self._start_vision(phase["deadline_monotonic_ns"])
        await self.adapter.connect(
            DEFAULT_MAVLINK_URL,
            deadline_monotonic_ns=phase["deadline_monotonic_ns"],
        )
        self._drain_all(
            phase="connect",
            update_estimator=True,
            collision_abort=True,
        )
        if getattr(self.adapter, "powered_source_promoted", None) is not True:
            raise CalibrationCheckFailure(
                "stream_stale", "MAVLink source did not promote"
            )

    def _capture_healthy(self) -> bool:
        if self.capture is None or self.capture.failure is not None:
            return False
        if self._adapter_capture_failure is not None:
            return False
        if self.adapter is not None and (
            getattr(self.adapter, "powered_source_rejected", False) is True
        ):
            self._collection_codes.add("source_rejected")
            return False
        if self.vision is not None:
            diagnostics = getattr(self.vision, "source_diagnostics", None)
            if callable(diagnostics):
                source = diagnostics()
                if getattr(source, "source_rejected_latched", False) is True:
                    self._collection_codes.add("source_rejected")
                    return False
            stats_method = getattr(self.vision, "stats", None)
            if callable(stats_method):
                stats = stats_method()
                for name in (
                    "capture_snapshot_queue_dropped",
                    "receiver_dropped_partial_frames",
                    "snapshot_callback_errors",
                    "timing_overflow_latched",
                ):
                    value = getattr(stats, name, 0)
                    if value not in {0, False}:
                        return False
        return True

    def _required_sources_present(self) -> bool:
        return all(
            item is not None
            for item in (
                self.state.heartbeat,
                self.state.race,
                self.state.imu,
                self.state.actuator,
            )
        )

    async def _preflight(self, phase: Mapping[str, Any]) -> None:
        deadline = phase["deadline_monotonic_ns"]
        while self._now() < deadline:
            self._drain_all(
                phase="preflight",
                update_estimator=True,
                collision_abort=True,
            )
            if (
                self._required_sources_present()
                and self.estimate is not None
                and self.estimate.healthy
                and self.capture is not None
                and self.capture.admitted
                and self.target is not None
                and self._capture_healthy()
            ):
                await self._stop_vision(deadline)
                return
            await self._poll_pause(deadline)
        raise CalibrationCheckFailure("stream_stale", "powered preflight timed out")

    async def _wait_reset_baseline(
        self,
        deadline_ns: int,
        *,
        phase: str,
        cleanup: bool,
    ) -> Dict[str, Any]:
        last_race = (
            None
            if self.state.race is None
            else self.state.race["race_status"]["sim_boot_time_ms"]
        )
        last_imu = (
            None if self.state.imu is None else self.state.imu["imu"]["timestamp_us"]
        )
        race_advances = 0
        imu_advances = 0
        while self._now() < deadline_ns:
            rows = self._drain_received(
                update_estimator=False,
                allow_clock_rollback=False,
            )
            if self.dispatcher is not None:
                self.dispatcher.drain_outbound()
            self._drain_collisions(phase=phase, abort=not cleanup)
            self._verify_adapter_queues(abort=not cleanup)
            for row in rows:
                if row["schema"] == "aigp-vq2-received-race-status/1":
                    value = row["race_status"]["sim_boot_time_ms"]
                    if last_race is not None and value > last_race:
                        race_advances += 1
                    last_race = value
                elif row["schema"] == "aigp-vq2-received-imu/1":
                    value = row["imu"]["timestamp_us"]
                    if last_imu is not None and value > last_imu:
                        imu_advances += 1
                    last_imu = value
            if (
                race_advances >= 2
                and imu_advances >= 2
                and self.state.race is not None
                and self.state.imu is not None
                and self.state.race["race_status"]["sim_boot_time_ms"] >= 800
                and self.state.imu["imu"]["timestamp_us"] >= 200_000
            ):
                return {
                    "race": dict(self.state.race),
                    "imu": dict(self.state.imu),
                }
            await self._poll_pause(deadline_ns)
        raise CalibrationCheckFailure(
            "stream_stale", "fresh advancing reset baseline was not observed"
        )

    def _clear_for_reset_generation(self, generation: int) -> None:
        self.state.begin_generation(generation)
        self.estimator = self._new_estimator()
        self.estimate = None
        self.tracker.reset()
        self.target = None
        self.frame = None
        self._last_frame_identity = None
        self._epoch_collision_count = 0

    async def _observe_reset_epoch(
        self,
        *,
        baseline: Mapping[str, Any],
        boundary: Mapping[str, Any],
        deadline_ns: int,
        phase: str,
        cleanup: bool,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        pre_race = baseline["race"]["race_status"]["sim_boot_time_ms"]
        pre_imu = baseline["imu"]["imu"]["timestamp_us"]
        race_anchor: Optional[int] = None
        imu_anchor: Optional[int] = None
        race_last: Optional[int] = None
        imu_last: Optional[int] = None
        races: List[Dict[str, Any]] = []
        imus: List[Dict[str, Any]] = []
        while self._now() < deadline_ns:
            rows = self._drain_received(
                update_estimator=False,
                allow_clock_rollback=True,
            )
            if self.dispatcher is not None:
                self.dispatcher.drain_outbound()
            self._drain_collisions(phase=phase, abort=not cleanup)
            self._verify_adapter_queues(abort=not cleanup)
            for row in rows:
                if row["schema"] == "aigp-vq2-received-race-status/1":
                    value = row["race_status"]["sim_boot_time_ms"]
                    if race_anchor is None:
                        if clock_rolled_back(pre_race, value, RESET_RACE_DROP_MS):
                            race_anchor = value
                            race_last = value
                            self.state.last_race_boot_ms = value
                            self.state.race_advance_monotonic_ns = row["ingress"][
                                "received_monotonic_ns"
                            ]
                    elif value > race_last:
                        races.append(dict(row))
                        race_last = value
                        self.state.last_race_boot_ms = value
                        self.state.race_advance_monotonic_ns = row["ingress"][
                            "received_monotonic_ns"
                        ]
                    start = row["race_status"]["race_start_boot_time_ms"]
                    if start < 0 or value < start:
                        self._countdown_observed = True
                elif row["schema"] == "aigp-vq2-received-imu/1":
                    value = row["imu"]["timestamp_us"]
                    if imu_anchor is None:
                        if clock_rolled_back(pre_imu, value, RESET_IMU_DROP_US):
                            imu_anchor = value
                            imu_last = value
                            self.state.last_imu_timestamp_us = value
                            self.state.imu_advance_monotonic_ns = row["ingress"][
                                "received_monotonic_ns"
                            ]
                    elif value > imu_last:
                        imus.append(dict(row))
                        imu_last = value
                        self.state.last_imu_timestamp_us = value
                        self.state.imu_advance_monotonic_ns = row["ingress"][
                            "received_monotonic_ns"
                        ]
            if (
                race_anchor is not None
                and imu_anchor is not None
                and len(races) >= 2
                and len(imus) >= 2
            ):
                epoch = {
                    "ingress_generation": boundary["new_generation"],
                    "race_anchor_boot_ms": race_anchor,
                    "imu_anchor_usec": imu_anchor,
                }
                return epoch, races, imus
            await self._poll_pause(deadline_ns)
        raise CalibrationCheckFailure(
            "race_not_advancing", "reset rollback/advance proof timed out"
        )

    async def _execute_reset(
        self,
        *,
        deadline_ns: int,
        phase: str,
        cleanup: bool,
    ) -> Dict[str, Any]:
        if self.dispatcher is None or self.lineage is None:
            raise CalibrationEvidenceError("reset dispatcher is unavailable")
        baseline = await self._wait_reset_baseline(
            deadline_ns,
            phase=phase,
            cleanup=cleanup,
        )
        if cleanup:
            self._ensure_cleanup_send_authority(deadline_ns)
        captured: Dict[str, Any] = {}
        def persist(boundary: Any) -> None:
            try:
                checked = self.lineage.record_reset_boundary(boundary, phase=phase)
                captured["boundary"] = checked
                if checked["collisions"]:
                    generation = checked["old_generation"]
                    self._collision_recorded_by_generation[generation] = (
                        self._collision_recorded_by_generation.get(generation, 0)
                        + len(checked["collisions"])
                    )
                    self._epoch_collision_count += len(checked["collisions"])
                    self._collection_codes.add("collision_observed")
                    if not cleanup:
                        raise CalibrationCheckFailure(
                            "collision_observed",
                            "reset boundary contained a collision observation",
                        )
                ingress_stats = checked["ingress_stats"]
                collision_stats = checked["collision_stats"]
                boundary_problems: List[str] = []
                if (
                    ingress_stats["dropped"]
                    or ingress_stats["imu_dropped"]
                    or ingress_stats["other_dropped"]
                ):
                    boundary_problems.append(
                        "reset boundary reported dropped ingress observations"
                    )
                if (
                    ingress_stats["buffered_imu"]
                    + ingress_stats["buffered_other"]
                    != len(checked["observations"])
                ):
                    boundary_problems.append(
                        "reset boundary ingress buffering did not match preserved rows"
                    )
                if collision_stats["dropped"]:
                    boundary_problems.append(
                        "reset boundary reported dropped collision observations"
                    )
                if collision_stats["buffered"] != len(checked["collisions"]):
                    boundary_problems.append(
                        "reset boundary collision buffering did not match preserved rows"
                    )
                if collision_stats["handled"] != (
                    self._collision_recorded_by_generation.get(
                        checked["old_generation"], 0
                    )
                ):
                    boundary_problems.append(
                        "reset boundary collision handled/recorded accounting mismatched"
                    )
                if boundary_problems:
                    self._latch_adapter_capture_failure(
                        "; ".join(boundary_problems), abort=not cleanup
                    )
            except BaseException as exc:
                if not cleanup:
                    raise
                # Cleanup authority must not be suppressed by failed replay
                # enqueue. Preserve the exact batch for the certificate and
                # let the adapter continue its guarded reset send.
                checked = self.lineage.retain_reset_boundary_without_replay(
                    boundary,
                    phase=phase,
                )
                captured["boundary"] = checked
                if checked["collisions"]:
                    generation = checked["old_generation"]
                    self._collision_recorded_by_generation[generation] = (
                        self._collision_recorded_by_generation.get(generation, 0)
                        + len(checked["collisions"])
                    )
                    self._epoch_collision_count += len(checked["collisions"])
                    self._collection_codes.add("collision_observed")
                self._reason_codes.add("capture_incomplete")

        dispatch = await self.dispatcher.dispatch_nonattitude(
            "sim_reset",
            deadline_ns,
            cleanup=cleanup,
            persist_boundary=persist,
            progress=(
                (lambda: self._ensure_cleanup_send_authority(deadline_ns))
                if cleanup
                else None
            ),
        )
        boundary = captured.get("boundary")
        if cleanup:
            persistence_state = getattr(
                self.adapter,
                "calibration_reset_persistence_state",
                None,
            )
            if callable(persistence_state):
                state = persistence_state()
                if getattr(state, "failure_latched", None) is True:
                    self._reason_codes.add("capture_incomplete")
            if boundary is None and dispatch.boundary is not None:
                boundary = self.lineage.retain_reset_boundary_without_replay(
                    dispatch.boundary,
                    phase=phase,
                )
                captured["boundary"] = boundary
                if boundary["collisions"]:
                    generation = boundary["old_generation"]
                    self._collision_recorded_by_generation[generation] = (
                        self._collision_recorded_by_generation.get(generation, 0)
                        + len(boundary["collisions"])
                    )
                    self._epoch_collision_count += len(boundary["collisions"])
                    self._collection_codes.add("collision_observed")
                self._reason_codes.add("capture_incomplete")
        result = self._reset_state("request_failed")
        result.update(
            {
                "request_monotonic_ns": dispatch.request_monotonic_ns,
                "receipt": (
                    dict(dispatch.receipt)
                    if dispatch.receipt is not None
                    and dispatch.receipt.get("outcome") == "raised"
                    else None
                ),
                "boundary": boundary,
                "baseline": dict(baseline),
            }
        )
        if boundary is None:
            raise CalibrationEvidenceError("reset boundary was not preserved")
        self._clear_for_reset_generation(boundary["new_generation"])
        if dispatch.error is not None or dispatch.receipt is None:
            if (
                dispatch.receipt is not None
                and dispatch.receipt.get("outcome") == "returned"
            ):
                result["state"] = "unconfirmed"
                result["receipt"] = dict(dispatch.receipt)
            if cleanup:
                self._reset = result
            raise CalibrationLifecycleError("reset request failed or was uncertain")
        if dispatch.receipt.get("outcome") != "returned":
            if cleanup:
                self._reset = result
            raise CalibrationLifecycleError("reset request raised")
        result["state"] = "unconfirmed"
        result["receipt"] = dict(dispatch.receipt)
        try:
            epoch, races, imus = await self._observe_reset_epoch(
                baseline=baseline,
                boundary=boundary,
                deadline_ns=deadline_ns,
                phase=phase,
                cleanup=cleanup,
            )
        except BaseException:
            result["advancing_race"] = []
            result["advancing_imu"] = []
            if cleanup:
                self._reset = result
            raise
        result.update(
            {
                "state": "confirmed",
                "clean_epoch": epoch,
                "advancing_race": races,
                "advancing_imu": imus,
                "rollback_and_advance_confirmed": True,
            }
        )
        self._reset_epoch = dict(epoch)
        self.state.imu_regressed = False
        self.state.race_regressed = False
        if cleanup:
            self._reset = result
        return result

    async def _reset_epoch_phase(self, phase: Mapping[str, Any]) -> None:
        deadline = phase["deadline_monotonic_ns"]
        await self._stop_vision(deadline)
        await self._execute_reset(
            deadline_ns=deadline,
            phase="reset_epoch",
            cleanup=False,
        )
        self.vision.reset()
        await self._start_vision(deadline)

    async def _wait_for_heartbeat(
        self,
        deadline_ns: int,
        *,
        phase: str,
        update_estimator: bool,
        cleanup: bool,
    ) -> Dict[str, Any]:
        while self._now() < deadline_ns:
            self._drain_received(
                update_estimator=update_estimator,
                allow_clock_rollback=False,
            )
            if self.dispatcher is not None:
                self.dispatcher.drain_outbound()
            self._drain_collisions(phase=phase, abort=not cleanup)
            self._verify_adapter_queues(abort=not cleanup)
            heartbeat = self.state.heartbeat
            if heartbeat is not None and (
                self.state.generation is None
                or heartbeat["ingress"]["generation"] == self.state.generation
            ):
                return dict(heartbeat)
            await self._poll_pause(deadline_ns)
        raise CalibrationCheckFailure("stream_stale", "fresh heartbeat was not observed")

    async def _disarm_confirmed(
        self,
        deadline_ns: int,
        *,
        phase: str,
        cleanup: bool,
    ) -> Dict[str, Any]:
        if self.dispatcher is None:
            raise CalibrationEvidenceError("disarm dispatcher is unavailable")
        before = await self._wait_for_heartbeat(
            deadline_ns,
            phase=phase,
            update_estimator=not cleanup,
            cleanup=cleanup,
        )
        if cleanup:
            self._ensure_cleanup_send_authority(deadline_ns)
        dispatch_deadline = (
            min(
                deadline_ns,
                self._now() + self.durations["outbound_call"],
            )
            if cleanup
            else deadline_ns
        )
        dispatch = await self.dispatcher.dispatch_nonattitude(
            "disarm",
            dispatch_deadline,
            cleanup=cleanup,
        )
        if cleanup and self._resume_unsent_cleanup_after_takeover(
            dispatch,
            deadline_ns=deadline_ns,
        ):
            dispatch = await self.dispatcher.dispatch_nonattitude(
                "disarm",
                dispatch_deadline,
                cleanup=True,
            )
        result = {
            "state": "request_failed",
            "request_monotonic_ns": dispatch.request_monotonic_ns,
            "receipt": (
                None if dispatch.receipt is None else dict(dispatch.receipt)
            ),
            "heartbeat_before": before,
            "heartbeat_after": None,
            "newer_confirmed": False,
        }
        if dispatch.error is not None or dispatch.receipt is None:
            if result["receipt"] is not None and result["receipt"]["outcome"] == "returned":
                result["state"] = "unconfirmed"
            if cleanup:
                self._disarm = result
            raise CalibrationLifecycleError("disarm request failed or was uncertain")
        if dispatch.receipt["outcome"] != "returned":
            if cleanup:
                self._disarm = result
            raise CalibrationLifecycleError("disarm request raised")
        result["state"] = "unconfirmed"
        before_sequence = before["ingress"]["sequence"]
        last_after: Optional[Dict[str, Any]] = None
        while self._now() < deadline_ns:
            self._drain_received(
                update_estimator=not cleanup,
                allow_clock_rollback=False,
            )
            self.dispatcher.drain_outbound()
            self._drain_collisions(phase=phase, abort=not cleanup)
            self._verify_adapter_queues(abort=not cleanup)
            after = self.state.heartbeat
            if (
                after is not None
                and after["ingress"]["sequence"] > before_sequence
                and after["ingress"]["received_monotonic_ns"]
                > dispatch.request_monotonic_ns
            ):
                last_after = dict(after)
                if after["heartbeat"]["base_mode"] & 128 == 0:
                    result.update(
                        {
                            "state": "confirmed",
                            "heartbeat_after": last_after,
                            "newer_confirmed": True,
                        }
                    )
                    if cleanup:
                        self._disarm = result
                    return result
            await self._poll_pause(deadline_ns)
        result["heartbeat_after"] = last_after
        if cleanup:
            self._disarm = result
        raise CalibrationCheckFailure(
            "stream_stale", "disarm was not confirmed by a newer heartbeat"
        )

    async def _normalize_disarmed(self, phase: Mapping[str, Any]) -> None:
        result = await self._disarm_confirmed(
            phase["deadline_monotonic_ns"],
            phase="normalize_disarmed",
            cleanup=False,
        )
        if result["heartbeat_after"]["heartbeat"]["base_mode"] & 128:
            raise CalibrationLifecycleError("vehicle remained armed after normalization")

    def _current_attitude(self) -> Tuple[float, float]:
        if self.estimate is None or self.estimate.healthy is not True:
            raise CalibrationCheckFailure(
                "estimator_unhealthy", "attitude estimate is unavailable"
            )
        roll, pitch, _yaw = self.estimate.orientation.to_euler()
        return float(roll), float(pitch)

    def _safety_authorization(self, _tick: int) -> CalibrationSafetyAuthorization:
        self._drain_all(
            phase="powered_stage",
            update_estimator=True,
            collision_abort=True,
        )
        if self.capture is not None:
            self.capture.raise_if_failed()
        if self.state.imu_regressed or self.state.race_regressed:
            raise CalibrationCheckFailure(
                "stream_stale", "source clock regressed inside the accepted epoch"
            )
        if (
            self._reset_epoch is None
            or self.frame is None
            or self.state.imu is None
            or self.state.race is None
            or self.state.heartbeat is None
            or self.state.actuator is None
            or self.target is None
            or self._initial_target_area is None
            or self._start_roll_rad is None
            or self._start_pitch_rad is None
        ):
            raise CalibrationCheckFailure(
                "stream_stale", "complete powered safety sources are unavailable"
            )
        roll, pitch = self._current_attitude()
        facts = CalibrationSafetyFacts(
            checked_monotonic_ns=self._now(),
            reset_epoch=self._reset_epoch,
            frame=self.frame,
            imu=self.state.imu,
            race=self.state.race,
            heartbeat=self.state.heartbeat,
            actuator=self.state.actuator,
            imu_advance_monotonic_ns=self.state.imu_advance_monotonic_ns,
            race_advance_monotonic_ns=self.state.race_advance_monotonic_ns,
            estimator_healthy=(
                self.estimate is not None
                and self.estimate.healthy is True
                and self.estimator.is_ready
            ),
            target_consecutive=self.tracker.consecutive,
            target_center_px=(self.target.center_x, self.target.center_y),
            target_bbox_px=tuple(self.target.bbox),
            initial_target_bbox_area_px=self._initial_target_area,
            start_roll_rad=self._start_roll_rad,
            start_pitch_rad=self._start_pitch_rad,
            current_roll_rad=roll,
            current_pitch_rad=pitch,
            collision_count=self._epoch_collision_count,
            capture_healthy=self._capture_healthy(),
            parent_alive=self._parent_alive(),
            lease_valid=self._lease_valid(),
        )
        return evaluate_calibration_safety(facts, contract=self.contract)

    async def _countdown_go(self, phase: Mapping[str, Any]) -> None:
        deadline = phase["deadline_monotonic_ns"]
        go_observed_ns: Optional[int] = None
        while self._now() < deadline:
            self._drain_all(
                phase="countdown_go",
                update_estimator=True,
                collision_abort=True,
            )
            if self.capture is not None:
                self.capture.raise_if_failed()
            race = self.state.race
            heartbeat = self.state.heartbeat
            if heartbeat is not None and heartbeat["heartbeat"]["base_mode"] & 128:
                raise CalibrationCheckFailure(
                    "internal_error", "vehicle armed before the explicit arm phase"
                )
            if race is not None:
                status = race["race_status"]
                boot = status["sim_boot_time_ms"]
                start = status["race_start_boot_time_ms"]
                if start < 0 or boot < start:
                    self._countdown_observed = True
                if (
                    self._countdown_observed
                    and start >= 0
                    and boot >= start + 150
                ):
                    if go_observed_ns is None:
                        go_observed_ns = self._now()
                    if (
                        self.target is not None
                        and self.tracker.consecutive >= 3
                        and self.capture is not None
                        and self.capture.admitted
                        and self.estimate is not None
                        and self.estimator.is_ready
                        and self._capture_healthy()
                    ):
                        roll, pitch = self._current_attitude()
                        self._start_roll_rad = roll
                        self._start_pitch_rad = pitch
                        self._initial_target_area = float(self.target.bbox_area)
                        self._safety_authorization(0)
                        return
                    if self._now() - go_observed_ns > 1_000_000_000:
                        raise CalibrationCheckFailure(
                            "target_unstable", "GO passed without complete readiness"
                        )
            await self._poll_pause(deadline)
        raise CalibrationCheckFailure(
            "stream_stale", "fresh countdown and GO+150ms were not observed"
        )

    async def _arm_confirmed(self, phase: Mapping[str, Any]) -> None:
        if self.dispatcher is None:
            raise CalibrationEvidenceError("arm dispatcher is unavailable")
        deadline = phase["deadline_monotonic_ns"]
        self._safety_authorization(0)
        before = await self._wait_for_heartbeat(
            deadline,
            phase="arm",
            update_estimator=True,
            cleanup=False,
        )
        # The heartbeat wait drains fresh receiver state.  Re-authorize from
        # that same-occurrence state, then perform one final immediately
        # pre-dispatch check so no stale pre-wait decision can authorize arm.
        self._safety_authorization(0)
        if before["heartbeat"]["base_mode"] & 128:
            raise CalibrationLifecycleError("vehicle was armed before arm request")
        self._safety_authorization(0)
        dispatch = await self.dispatcher.dispatch_nonattitude(
            "arm", deadline, cleanup=False
        )
        if (
            dispatch.error is not None
            or dispatch.receipt is None
            or dispatch.receipt.get("outcome") != "returned"
        ):
            raise CalibrationLifecycleError("arm request failed or was uncertain")
        before_sequence = before["ingress"]["sequence"]
        while self._now() < deadline:
            self._drain_all(
                phase="arm",
                update_estimator=True,
                collision_abort=True,
            )
            after = self.state.heartbeat
            if (
                after is not None
                and after["ingress"]["sequence"] > before_sequence
                and after["ingress"]["received_monotonic_ns"]
                > dispatch.request_monotonic_ns
                and after["heartbeat"]["base_mode"] & 128
            ):
                self._safety_authorization(0)
                return
            await self._poll_pause(deadline)
        raise CalibrationCheckFailure(
            "stream_stale", "arm was not confirmed by a newer heartbeat"
        )

    async def _powered_stage(self, phase: Mapping[str, Any]) -> None:
        if self.dispatcher is None:
            raise CalibrationEvidenceError("powered dispatcher is unavailable")
        scheduler = CalibrationExcitationScheduler(
            evidence=self.evidence,
            safety_check=self._safety_authorization,
            dispatch=self.dispatcher.dispatch,
            monotonic_ns=self.services.monotonic_ns,
            wait_until_ns=self.services.wait_until_ns,
            powered_parent_deadline_monotonic_ns=phase["deadline_monotonic_ns"],
            contract=self.contract,
        )
        result = await scheduler.run()
        if not result.completed:
            raise CalibrationCheckFailure(
                result.abort_reason_code or "internal_error",
                "powered excitation did not complete every frozen slot",
            )

    def _takeover_if_parent_dead(
        self,
        *,
        parent_deadline_ns: int,
        enable_cleanup: bool,
        emit: bool,
        phase_override: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if self._parent_mode == "signaled_takeover":
            return True
        if self._parent_alive():
            return False
        self._note_parent_death()
        if self._takeover_attempted:
            self._cleanup_failures.update({"parent_dead", "lease_invalid"})
            raise CalibrationLifecycleError("parent-death takeover was exhausted")
        self._takeover_attempted = True
        phase = (
            dict(phase_override)
            if phase_override is not None
            else self._phase(
                "parent_death_lease_takeover",
                emit=emit,
                parent_deadline_monotonic_ns=parent_deadline_ns,
            )
        )
        self._pending_takeover_phase = None
        try:
            boundary_proof = self.services.lease_boundary.take_over_abandoned(
                role_secret=memoryview(self.admission.role_secret),
                attempt=self.admission.attempt,
                process_authority=self.admission.process_authority,
                deadline_monotonic_ns=phase["deadline_monotonic_ns"],
            )
            proof = self._coerce_lease_proof(boundary_proof)
        except BaseException as exc:
            self._cleanup_failures.update({"parent_dead", "lease_invalid"})
            raise CalibrationLifecycleError("abandoned live lease takeover failed") from exc
        if (
            proof.owner_role != "powered-child-parent-death"
            or proof.authority_valid is not True
            or proof.takeover_completed_monotonic_ns is None
            or proof.takeover_completed_monotonic_ns <= self._parent_observed_ns
            or proof.takeover_completed_monotonic_ns
            >= phase["deadline_monotonic_ns"]
        ):
            self._cleanup_failures.update({"parent_dead", "lease_invalid"})
            raise CalibrationLifecycleError("parent-death lease proof is invalid")
        self._lease_proof = proof
        self._lease_boundary_proof = boundary_proof
        self._parent_mode = "signaled_takeover"
        self._takeover_completed_ns = proof.takeover_completed_monotonic_ns
        self._takeover_record_sha256 = proof.record_sha256
        self._takeover_last_heartbeat_ns = proof.takeover_completed_monotonic_ns
        self._takeover_released = False
        if enable_cleanup:
            if self.guards is None or self.adapter is None:
                raise CalibrationEvidenceError("cleanup takeover transport is unavailable")
            self.guards.enable_cleanup_takeover(
                parent_signaled=True,
                abandoned_lease_owned=True,
                authority_valid=True,
                source_promoted=(
                    getattr(self.adapter, "powered_source_promoted", None) is True
                ),
            )
        return True

    def _freeze_pending_takeover(
        self,
        *,
        parent_deadline_ns: int,
        emit: bool,
    ) -> None:
        """Freeze the one-second takeover wall without acquiring during replay close."""

        if self._parent_mode == "signaled_takeover" or self._pending_takeover_phase is not None:
            return
        self._note_parent_death()
        self._pending_takeover_phase = self._phase(
            "parent_death_lease_takeover",
            emit=emit,
            parent_deadline_monotonic_ns=parent_deadline_ns,
        )

    def _post_cleanup_parent_check(
        self,
        *,
        parent_deadline_ns: int,
        emit: bool,
    ) -> bool:
        """Take over/release-capable authority after simulator sends are finished."""

        if self._parent_mode == "signaled_takeover":
            self._service_takeover_heartbeat(parent_deadline_ns)
            return True
        if self._parent_alive():
            return False
        self._note_parent_death()
        return self._takeover_if_parent_dead(
            parent_deadline_ns=parent_deadline_ns,
            enable_cleanup=False,
            emit=emit,
            phase_override=self._pending_takeover_phase,
        )

    def _enable_cleanup_authority(self, deadline_ns: int) -> None:
        if self.dispatcher is None or self.guards is None or self.adapter is None:
            raise CalibrationEvidenceError("cleanup transport authority is unavailable")
        if self._parent_mode == "signaled_takeover":
            self._service_takeover_heartbeat(deadline_ns)
            if self.guards.cleanup_state == "takeover_pending":
                self.guards.enable_cleanup_takeover(
                    parent_signaled=True,
                    abandoned_lease_owned=True,
                    authority_valid=True,
                    source_promoted=(
                        getattr(self.adapter, "powered_source_promoted", None) is True
                    ),
                )
            return
        if self._parent_alive():
            self._parent_observed_ns = self._now()
            boundary_proof = self.services.lease_boundary.prove_live_delegation(
                attempt=self.admission.attempt,
                process_authority=self.admission.process_authority,
            )
            proof = self._coerce_lease_proof(boundary_proof)
            if proof.owner_role != "wrapper" or proof.authority_valid is not True:
                self._cleanup_failures.add("lease_invalid")
                raise CalibrationLifecycleError("live cleanup lease is invalid")
            self._lease_proof = proof
            self._lease_boundary_proof = boundary_proof
            if not self._parent_alive():
                self._note_parent_death()
            else:
                try:
                    self.dispatcher.begin_live_cleanup()
                    return
                except BaseException:
                    if self._parent_alive():
                        raise
                    self._note_parent_death()
        self._takeover_if_parent_dead(
            parent_deadline_ns=deadline_ns,
            enable_cleanup=True,
            emit=True,
        )

    def _ensure_cleanup_send_authority(self, deadline_ns: int) -> None:
        self._supervise_parent(deadline_ns, cleanup=True)
        if not self._lease_valid():
            self._cleanup_failures.add("lease_invalid")
            raise CalibrationLifecycleError("cleanup lease authority is invalid")

    def _resume_unsent_cleanup_after_takeover(
        self,
        dispatch: Any,
        *,
        deadline_ns: int,
    ) -> bool:
        """Permit one retry only when the first local call provably never began."""

        if (
            self._parent_mode != "live_delegation"
            or getattr(dispatch, "receipt", None) is not None
            or getattr(dispatch, "audit_count_after", None)
            != getattr(dispatch, "audit_count_before", None)
            or self._parent_alive()
        ):
            return False
        self._note_parent_death()
        self._takeover_if_parent_dead(
            parent_deadline_ns=deadline_ns,
            enable_cleanup=True,
            emit=True,
        )
        self._ensure_cleanup_send_authority(deadline_ns)
        return True

    def _zero_required(self) -> bool:
        peer = None if self.adapter is None else getattr(self.adapter, "powered_peer", None)
        receipts = [] if self.lineage is None else self.lineage.outbound_receipts
        return bool(
            peer is not None
            or any(
                row.get("schema") == "aigp-vq2-attitude-target-outbound/1"
                or row.get("category") == "arm"
                for row in receipts
            )
        )

    async def _cleanup_zero(self, deadline_ns: int) -> None:
        required = self._zero_required()
        self._zero_command = self._zero_state(required=required)
        if not required:
            return
        if self.dispatcher is None:
            self._cleanup_failures.add("zero_failed")
            return
        self._ensure_cleanup_send_authority(deadline_ns)
        checked_ns = self._now()
        generated = self.evidence.record_cleanup_generated(
            checked_monotonic_ns=checked_ns,
            generated_monotonic_ns=self._now(),
        )
        self._zero_command["generated"] = generated
        try:
            self._ensure_cleanup_send_authority(deadline_ns)
            dispatch_deadline = min(
                deadline_ns,
                self._now() + self.durations["outbound_call"],
            )
            dispatch = await self.dispatcher.dispatch_cleanup_zero(
                dispatch_deadline
            )
            if self._resume_unsent_cleanup_after_takeover(
                dispatch,
                deadline_ns=deadline_ns,
            ):
                dispatch = await self.dispatcher.dispatch_cleanup_zero(
                    dispatch_deadline
                )
        except BaseException as exc:
            before = self.dispatcher._attitude_audit_count()
            dispatch = CalibrationDispatchResult(
                audit_count_before=before,
                audit_count_after=before,
                receipt=None,
                call_started_monotonic_ns=None,
                call_ended_monotonic_ns=None,
                error=exc,
            )
        if dispatch.receipt is not None and dispatch.receipt.get("outcome") == "returned":
            terminal = self.evidence.record_sent(
                generated,
                sent_monotonic_ns=self._now(),
                dispatch=dispatch,
            )
            self._zero_command.update(
                {
                    "state": "returned",
                    "terminal": terminal,
                    "outbound_receipt": dict(dispatch.receipt),
                }
            )
            if dispatch.error is not None:
                self._cleanup_failures.add("receipt_incomplete")
            return
        reason = "send_raised" if dispatch.call_started_monotonic_ns is not None else "internal_error"
        terminal = self.evidence.record_not_sent(
            generated,
            recorded_monotonic_ns=self._now(),
            reason_code=reason,
            detail="cleanup exact-zero dispatch failed or was uncertain",
            dispatch=dispatch,
        )
        self._zero_command.update(
            {
                "state": "failed",
                "terminal": terminal,
                "outbound_receipt": (
                    dict(dispatch.receipt)
                    if dispatch.receipt is not None
                    and dispatch.receipt.get("outcome") == "raised"
                    else None
                ),
            }
        )
        self._cleanup_failures.add("zero_failed")

    async def _wait_final_state(self, deadline_ns: int) -> None:
        while self._now() < deadline_ns:
            self._drain_received(
                update_estimator=False,
                allow_clock_rollback=False,
            )
            if self.dispatcher is not None:
                self.dispatcher.drain_outbound()
            self._drain_collisions(phase="cleanup", abort=False)
            self._verify_adapter_queues(abort=False)
            heartbeat = self.state.heartbeat
            race = self.state.race
            imu = self.state.imu
            if (
                self._reset_epoch is not None
                and heartbeat is not None
                and race is not None
                and imu is not None
                and heartbeat["ingress"]["generation"]
                == self._reset_epoch["ingress_generation"]
                and race["race_status"]["sim_boot_time_ms"]
                > self._reset_epoch["race_anchor_boot_ms"]
                and imu["imu"]["timestamp_us"]
                > self._reset_epoch["imu_anchor_usec"]
                and heartbeat["heartbeat"]["base_mode"] & 128 == 0
            ):
                self._final_state = {
                    "state": "confirmed",
                    "heartbeat": dict(heartbeat),
                    "disarmed": True,
                    "reset_epoch": dict(self._reset_epoch),
                    "last_race": dict(race),
                    "last_imu": dict(imu),
                }
                return
            await self._poll_pause(deadline_ns)
        if all(
            value is not None
            for value in (
                self.state.heartbeat,
                self.state.race,
                self.state.imu,
                self._reset_epoch,
            )
        ):
            self._final_state = {
                "state": "partial",
                "heartbeat": dict(self.state.heartbeat),
                "disarmed": bool(
                    self.state.heartbeat["heartbeat"]["base_mode"] & 128 == 0
                ),
                "reset_epoch": dict(self._reset_epoch),
                "last_race": dict(self.state.race),
                "last_imu": dict(self.state.imu),
            }
        raise CalibrationLifecycleError("final disarmed reset epoch was not proved")

    async def _close_transports(self, deadline_ns: int) -> None:
        try:
            await self._stop_vision(deadline_ns, cleanup=True)
        except BaseException:
            self._cleanup_failures.add("transport_unclosed")
        try:
            if self.guards is not None:
                self.guards.close_cleanup()
        except BaseException:
            self._cleanup_failures.add("transport_unclosed")
        try:
            if self.dispatcher is not None:
                await self.dispatcher.disconnect(
                    deadline_ns,
                    progress=lambda: self._supervise_parent(
                        deadline_ns,
                        cleanup=True,
                    ),
                )
        except BaseException:
            self._cleanup_failures.add("transport_unclosed")
        try:
            # Both producer threads are now stopped.  Drain every retained
            # application queue once more before taking the immutable
            # pre-close resource snapshot that is sealed into the manifest.
            self._drain_received(
                update_estimator=False,
                allow_clock_rollback=False,
            )
            if self.dispatcher is not None:
                self.dispatcher.drain_outbound()
            self._drain_collisions(phase="cleanup", abort=False)
            self._drain_frames()
            self._verify_adapter_queues(abort=False)
        except BaseException:
            self._cleanup_failures.update(
                {"receipt_incomplete", "transport_unclosed"}
            )

    async def _cleanup(self, phase: Mapping[str, Any]) -> None:
        self._cleanup_started = True
        self._cleanup_phase = dict(phase)
        deadline = phase["deadline_monotonic_ns"]
        try:
            if self.adapter is None or getattr(
                self.adapter, "powered_source_promoted", None
            ) is not True:
                self._zero_command = self._zero_state(required=self._zero_required())
                self._disarm = self._disarm_state("not_required")
                self._reset = self._reset_state("not_required")
                self._cleanup_failures.add("connect_failed")
                if not self._lease_valid():
                    self._cleanup_failures.update({"authority_invalid", "lease_invalid"})
                if not self._parent_alive():
                    try:
                        self._takeover_if_parent_dead(
                            parent_deadline_ns=deadline,
                            enable_cleanup=False,
                            emit=True,
                        )
                    except BaseException:
                        self._cleanup_failures.update({"parent_dead", "lease_invalid"})
                return
            self._enable_cleanup_authority(deadline)
            self._disarm = self._disarm_state("not_attempted")
            self._reset = self._reset_state("not_attempted")
            try:
                await self._cleanup_zero(deadline)
            except BaseException:
                self._cleanup_failures.add("zero_failed")
            try:
                self._ensure_cleanup_send_authority(deadline)
                await self._disarm_confirmed(
                    deadline,
                    phase="cleanup",
                    cleanup=True,
                )
            except BaseException:
                self._cleanup_failures.add("disarm_failed")
            try:
                # No cleanup operation consumes camera data. Stop its producer
                # and drain the old-generation FIFO before SIM_RESET advances
                # the independently owned MAVLink generation.
                await self._stop_vision(deadline, cleanup=True)
                self._drain_frames()
                self._ensure_cleanup_send_authority(deadline)
                await self._execute_reset(
                    deadline_ns=deadline,
                    phase="cleanup",
                    cleanup=True,
                )
            except BaseException:
                self._cleanup_failures.add("reset_failed")
            try:
                await self._wait_final_state(deadline)
            except BaseException:
                self._cleanup_failures.add("final_state_unproved")
            try:
                if self._parent_mode == "live_delegation" and not self._parent_alive():
                    self._takeover_if_parent_dead(
                        parent_deadline_ns=deadline,
                        enable_cleanup=True,
                        emit=True,
                    )
            except BaseException:
                self._cleanup_failures.update({"parent_dead", "lease_invalid"})
        finally:
            await self._close_transports(deadline)

    def _default_endpoint_evidence(self) -> Dict[str, Any]:
        mavlink = {
            "state": "not_opened",
            "bind": None,
            "frozen_peer": None,
            "rejected_source_count": 0,
        }
        if self.adapter is not None:
            state_method = getattr(self.adapter, "powered_transport_state", None)
            if not callable(state_method):
                raise CalibrationEvidenceError(
                    "powered adapter lacks public transport-state evidence"
                )
            state = state_method()
            if state is None:
                raise CalibrationEvidenceError(
                    "constructed powered adapter has no transport-state evidence"
                )
            bind_method = getattr(state, "bind_proof", None)
            if not callable(bind_method):
                raise CalibrationEvidenceError(
                    "powered transport state lacks public bind proof"
                )
            bind = bind_method()
            if type(bind) is not dict:
                raise CalibrationEvidenceError(
                    "powered transport bind proof must be an exact object"
                )
            bind = json.loads(json.dumps(bind, allow_nan=False))
            bind["role"] = "mavlink"
            bind["owner_process"] = self.admission.current_process
            peer = getattr(state, "frozen_peer", None)
            mavlink = {
                "state": getattr(state, "endpoint_state", None),
                "bind": bind,
                "frozen_peer": (
                    None
                    if peer is None
                    else {"host": peer[0], "port": peer[1]}
                ),
                "rejected_source_count": getattr(
                    state, "rejected_source_count", None
                ),
            }
        camera = {
            "state": "not_opened",
            "bind": None,
            "frozen_peer": None,
            "rejected_source_count": 0,
        }
        if self.vision is not None:
            diagnostics_method = getattr(self.vision, "source_diagnostics", None)
            if callable(diagnostics_method):
                diagnostics = diagnostics_method()
                state = getattr(diagnostics, "state", "not_opened")
                if state != "not_opened":
                    peer = getattr(diagnostics, "frozen_peer", None)
                    camera = {
                        "state": state,
                        "bind": {
                            "role": "camera",
                            "family": "AF_INET",
                            "requested": {
                                "host": diagnostics.requested_host,
                                "port": diagnostics.requested_port,
                            },
                            "actual": {
                                "host": diagnostics.actual_host,
                                "port": diagnostics.actual_port,
                            },
                            "socket_policy": diagnostics.socket_policy,
                            "owner_process": self.admission.current_process,
                        },
                        "frozen_peer": (
                            None
                            if peer is None
                            else {"host": peer[0], "port": peer[1]}
                        ),
                        "rejected_source_count": diagnostics.rejected_source_count,
                    }
        return {"mavlink": mavlink, "camera": camera}

    def _endpoints(self) -> Dict[str, Any]:
        callback = self.services.endpoint_evidence
        value = (
            self._default_endpoint_evidence()
            if callback is None
            else callback(self.adapter, self.vision, self.admission)
        )
        if type(value) is not dict:
            raise CalibrationEvidenceError("endpoint proof must be an exact object")
        result = json.loads(json.dumps(value, allow_nan=False))
        for endpoint in result.values():
            if isinstance(endpoint, dict) and endpoint.get("rejected_source_count", 0):
                self._collection_codes.add("source_rejected")
        camera = result.get("camera")
        if isinstance(camera, dict) and camera.get("state") in {
            "not_opened",
            "closed_without_peer",
        }:
            self._collection_codes.add("camera_missing")
        return result

    def _default_transport_evidence(self) -> Dict[str, Any]:
        guard_latched = bool(
            self.guards is not None
            and getattr(self.guards, "production_latched", False) is True
        )
        cleanup_closed = bool(
            self.guards is not None
            and getattr(self.guards, "cleanup_state", None) == "closed"
        )
        if self.vision is None:
            vision_closed = True
        else:
            diagnostics_method = getattr(self.vision, "source_diagnostics", None)
            if callable(diagnostics_method):
                diagnostics = diagnostics_method()
                endpoint_state = getattr(diagnostics, "state", None)
            else:
                endpoint_state = None
            vision_closed = bool(
                getattr(self.vision, "is_running", None) is False
                and endpoint_state
                in (
                    {"closed_with_peer", "closed_without_peer"}
                    if self._vision_start_attempted
                    else {"not_opened", "closed_with_peer", "closed_without_peer"}
                )
            )
        state = None
        if self.adapter is not None:
            state_method = getattr(self.adapter, "powered_transport_state", None)
            if not callable(state_method):
                raise CalibrationEvidenceError(
                    "powered adapter lacks public transport-state evidence"
                )
            state = state_method()
            if state is None:
                raise CalibrationEvidenceError(
                    "constructed powered adapter has no transport-state evidence"
                )
        socket_closed = bool(
            state is None or getattr(state, "endpoint_closed", None) is True
        )
        receiver_joined = bool(
            state is None or getattr(state, "receiver_joined", None) is True
        )
        announcer_joined = bool(
            state is None or getattr(state, "announcer_joined", None) is True
        )
        owned_handles_closed = bool(
            state is None or getattr(state, "owned_handles_closed", None) is True
        )
        return {
            "production_guard_latched": guard_latched,
            "cleanup_guard_closed": cleanup_closed,
            "vision_closed": vision_closed,
            "mavlink_socket_closed": socket_closed,
            "receiver_joined": receiver_joined,
            "announcer_joined": announcer_joined,
            "owned_handles_closed": bool(owned_handles_closed and vision_closed),
        }

    def _transport(self) -> Dict[str, Any]:
        callback = self.services.transport_evidence
        value = (
            self._default_transport_evidence()
            if callback is None
            else callback(self.adapter, self.vision, self.guards)
        )
        if type(value) is not dict:
            raise CalibrationEvidenceError("transport proof must be an exact object")
        expected = {
            "production_guard_latched",
            "cleanup_guard_closed",
            "vision_closed",
            "mavlink_socket_closed",
            "receiver_joined",
            "announcer_joined",
            "owned_handles_closed",
        }
        if set(value) != expected or any(type(item) is not bool for item in value.values()):
            raise CalibrationEvidenceError("transport proof shape is invalid")
        return dict(value)

    def _outbound_audit(self) -> Dict[str, int]:
        receipts = [] if self.lineage is None else self.lineage.outbound_receipts
        categories = (
            "timesync",
            "gcs_heartbeat",
            "sim_reset",
            "arm",
            "disarm",
            "attitude_target",
            "position_target",
            "other_command",
        )
        audit = {
            "timesync": 0,
            "gcs_heartbeat": 0,
            "sim_reset": 0,
            "arm": 0,
            "disarm": 0,
            "attitude_target": 0,
            "position_target": 0,
            "other_command": 0,
            "receipt_count": len(receipts),
            "receipt_returned": 0,
            "receipt_raised": 0,
            "receipt_dropped": 0,
            "receipt_buffered": 0,
        }
        prior = -1
        for receipt in receipts:
            sequence = receipt["outbound_sequence"]
            if sequence <= prior:
                self._cleanup_failures.add("receipt_incomplete")
            prior = sequence
            if receipt["schema"] == "aigp-vq2-attitude-target-outbound/1":
                audit["attitude_target"] += 1
            else:
                audit[receipt["category"]] += 1
            audit[f"receipt_{receipt['outcome']}"] += 1
        retained_category_counts = {
            name: audit[name]
            for name in categories
        }
        stats_method = (
            None if self.adapter is None else getattr(self.adapter, "outbound_receipt_stats", None)
        )
        reported_dropped = 0
        if callable(stats_method):
            stats = stats_method()
            dropped = getattr(stats, "dropped", 0)
            buffered = getattr(stats, "buffered", 0)
            if type(dropped) is not int or dropped < 0:
                raise CalibrationEvidenceError("outbound dropped count is invalid")
            if type(buffered) is not int or buffered < 0:
                raise CalibrationEvidenceError("outbound buffered count is invalid")
            reported_dropped = dropped
            if dropped > 0:
                self._cleanup_failures.add("receipt_incomplete")
                self._collection_codes.add("unexpected_outbound")
            if buffered > 0:
                audit["receipt_buffered"] = buffered
                self._cleanup_failures.add("receipt_incomplete")
        raw_method = None if self.adapter is None else getattr(self.adapter, "outbound_audit", None)
        if callable(raw_method):
            raw = raw_method()
            for name in categories:
                value = (
                    raw.get(name, 0)
                    if isinstance(raw, Mapping)
                    else getattr(raw, name, 0)
                )
                if type(value) is not int or value < 0:
                    raise CalibrationEvidenceError(
                        f"outbound attempted count {name} is invalid"
                    )
                if value < retained_category_counts[name]:
                    raise CalibrationEvidenceError(
                        f"outbound attempted count {name} is below retained receipts"
                    )
                audit[name] = value
        elif reported_dropped:
            # A receipt drop cannot truthfully be assigned to a category when
            # the adapter omitted its exact attempted-category audit.
            raise CalibrationEvidenceError(
                "dropped outbound receipts lack attempted-category evidence"
            )
        attempted = sum(audit[name] for name in categories)
        if attempted < audit["receipt_count"]:
            raise CalibrationEvidenceError(
                "outbound attempted count is below retained receipt count"
            )
        audit["receipt_dropped"] = attempted - audit["receipt_count"]
        if audit["receipt_dropped"]:
            self._cleanup_failures.add("receipt_incomplete")
            self._collection_codes.add("unexpected_outbound")
        if reported_dropped != audit["receipt_dropped"]:
            self._cleanup_failures.add("receipt_incomplete")
            self._collection_codes.add("unexpected_outbound")
        return self.contract.validate_outbound_audit(audit)

    def _cleanup_proved(
        self,
        endpoints: Mapping[str, Any],
        transport: Mapping[str, Any],
    ) -> bool:
        return bool(
            not self._cleanup_failures
            and self._lease_valid()
            and endpoints["mavlink"]["state"] == "closed_with_peer"
            and self._zero_command["state"] in {"not_required", "returned"}
            and self._disarm["state"] == "confirmed"
            and self._reset["state"] == "confirmed"
            and self._final_state["state"] == "confirmed"
            and all(transport.values())
        )

    def _build_cleanup_certificate(self) -> Tuple[Dict[str, Any], Dict[str, int]]:
        if self._cleanup_phase is None:
            raise CalibrationEvidenceError("cleanup phase was not frozen")
        endpoints = self._endpoints()
        transport = self._transport()
        audit = self._outbound_audit()
        if self.lineage is not None and self.lineage.collisions:
            self._collection_codes.add("collision_observed")
        if not all(transport.values()):
            self._cleanup_failures.add("transport_unclosed")
        proof = self._lease_proof
        if proof is None:
            proof = CalibrationLeaseProof(
                owner_role="wrapper",
                generation=0,
                record_sha256=self.admission.process_authority["lease_record_sha256"],
                authority_valid=False,
            )
        completed = self._now()
        deadline = self._cleanup_phase["deadline_monotonic_ns"]
        if completed >= deadline:
            self._cleanup_failures.add("deadline_expired")
            raise CalibrationEvidenceError("cleanup certificate deadline expired")
        proved = self._cleanup_proved(endpoints, transport)
        if not proved and not self._cleanup_failures:
            self._cleanup_failures.add("internal_error")
        trigger = self._trigger
        if trigger != "parent_death":
            trigger = "normal_completion" if self._stage_completed else "stage_abort"
        certificate = {
            "schema": "aigp-vq2-powered-cleanup-certificate/1",
            "task_id": self.contract.TASK_ID,
            "session_id": self.contract.SESSION_ID,
            "attempt_id": self.contract.ATTEMPT_ID,
            "producer_role": CALIBRATION_CHILD_ROLE,
            "cleanup_epoch": "child-cleanup-0",
            "authority": {
                "process_authority": {
                    "path": self.admission.arguments.powered_process_authority,
                    "sha256": self.admission.process_authority_sha256,
                },
                "attempt_context_sha256": self.admission.attempt["context_sha256"],
                "attempt_envelope_sha256": self.admission.attempt_envelope_sha256,
                "producer": self.admission.current_process,
            },
            "trigger": trigger,
            "started_monotonic_ns": self._cleanup_phase["started_monotonic_ns"],
            "deadline_monotonic_ns": deadline,
            "completed_monotonic_ns": completed,
            "parent_state": {
                "mode": self._parent_mode,
                "wrapper_process": self.admission.wrapper_process,
                "observed_monotonic_ns": self._parent_observed_ns,
                "takeover_completed_monotonic_ns": self._takeover_completed_ns,
                "takeover_lease_record_sha256": self._takeover_record_sha256,
            },
            "lease": {
                "owner_role": proof.owner_role,
                "generation": proof.generation,
                "record_sha256": proof.record_sha256,
                "authority_valid": proof.authority_valid,
            },
            "phase_deadlines": list(self.phase_deadlines),
            "endpoints": endpoints,
            "outbound_receipts": (
                [] if self.lineage is None else list(self.lineage.outbound_receipts)
            ),
            "zero_command": self._zero_command,
            "disarm": self._disarm,
            "reset": self._reset,
            "collisions": {
                "observations": (
                    [] if self.lineage is None else list(self.lineage.collisions)
                ),
                "invalidating_occurrence_count": (
                    0 if self.lineage is None else len(self.lineage.collisions)
                ),
            },
            "final_state": self._final_state,
            "transport": transport,
            "outcome": "proved" if proved else "failed",
            "failure_codes": [] if proved else sorted(self._cleanup_failures),
            "collection_invalidating_codes": sorted(self._collection_codes),
        }
        return self.contract.validate_cleanup_certificate(certificate), audit

    def _publish_certificate(
        self,
        certificate: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        if self._certificate_published:
            raise CalibrationEvidenceError("cleanup certificate publication repeated")
        self._certificate_published = True
        deadline = self._cleanup_phase["deadline_monotonic_ns"]
        def progress() -> None:
            self._post_cleanup_parent_check(
                parent_deadline_ns=deadline,
                emit=self.recorder is not None and not self._replay_closed,
            )

        progress()
        publish = self.services.publisher.publish_create_new
        parameters = inspect.signature(publish).parameters
        publish_kwargs: Dict[str, Any] = {
            "deadline_monotonic_ns": deadline,
        }
        if "progress" in parameters:
            publish_kwargs["progress"] = progress
        elif "progress_callback" in parameters:
            publish_kwargs["progress_callback"] = progress
        elif any(
            value.kind == inspect.Parameter.VAR_KEYWORD
            for value in parameters.values()
        ):
            publish_kwargs["progress"] = progress
        else:
            raise CalibrationEvidenceError(
                "cleanup certificate publisher lacks cooperative progress"
            )
        digest = publish(
            self.admission.arguments.cleanup_certificate,
            certificate,
            **publish_kwargs,
        )
        if self._now() >= deadline:
            raise CalibrationEvidenceError(
                "cleanup certificate publication completed too late"
            )
        expected = self.contract.canonical_file_sha256(certificate)
        if digest != expected:
            raise CalibrationEvidenceError("cleanup certificate readback hash mismatched")
        if certificate["parent_state"]["mode"] != self._parent_mode:
            raise CalibrationEvidenceError(
                "cleanup certificate parent state changed during publication"
            )
        self._certificate_reference_state = "published"
        self._certificate_reference_sha256 = digest
        return dict(certificate), digest

    def _preserve_failed_certificate_reference(self) -> None:
        """Classify one failed create-new target without overwriting forensic bytes."""

        target = Path(self.admission.arguments.cleanup_certificate)
        try:
            if not target.is_file():
                self._certificate_reference_state = "absent"
                self._certificate_reference_sha256 = None
                return
            self._certificate_reference_sha256 = self._file_sha256(target)
            self._certificate_reference_state = "invalid"
        except BaseException:
            self._certificate_reference_state = "absent"
            self._certificate_reference_sha256 = None
            self._reason_codes.add("capture_incomplete")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                payload = stream.read(1024 * 1024)
                if not payload:
                    break
                digest.update(payload)
        return digest.hexdigest()

    def _default_close_artifacts(
        self,
        outcome: Mapping[str, Any],
        deadline_ns: int,
    ) -> CalibrationClosedArtifacts:
        if self.recorder is None:
            partial = self._partial_artifacts("absent")
            return CalibrationClosedArtifacts(
                legacy_record=partial["legacy_record"],
                replay_bundle=partial["replay_bundle"],
            )
        now = self._now()
        if now >= deadline_ns:
            raise CalibrationEvidenceError("replay close deadline expired")
        stats = self.recorder.close(
            outcome=dict(outcome),
            timeout_s=(deadline_ns - now) / 1_000_000_000.0,
        )
        if self._now() >= deadline_ns:
            raise CalibrationEvidenceError("replay close completed too late")
        record_path = Path(self.admission.arguments.record)
        legacy = {
            "path": self.admission.arguments.record,
            "state": "partial",
            "sha256": None,
        }
        if record_path.is_file():
            legacy = {
                "path": self.admission.arguments.record,
                "state": "closed",
                "sha256": self._file_sha256(record_path),
            }
        replay = {
            "path": self.admission.arguments.replay_bundle,
            "state": "partial",
            "dataset_hash": None,
            "manifest_sha256": None,
            "records_sha256": None,
        }
        complete = getattr(stats, "complete", False) is True
        dataset_hash = getattr(stats, "dataset_hash", None)
        bundle_path = Path(self.admission.arguments.replay_bundle)
        manifest_path = bundle_path / "manifest.json"
        records_path = bundle_path / "records.jsonl"
        if (
            complete
            and type(dataset_hash) is str
            and manifest_path.is_file()
            and records_path.is_file()
        ):
            replay = {
                "path": self.admission.arguments.replay_bundle,
                "state": "closed",
                "dataset_hash": dataset_hash,
                "manifest_sha256": self._file_sha256(manifest_path),
                "records_sha256": self._file_sha256(records_path),
            }
        return CalibrationClosedArtifacts(
            legacy_record=legacy,
            replay_bundle=replay,
        )

    @staticmethod
    def _resource_counter_object(
        source: Any,
        names: Sequence[str],
        *,
        constructed: bool,
        boolean_names: Sequence[str] = (),
    ) -> Dict[str, Any]:
        if type(constructed) is not bool:
            raise CalibrationEvidenceError(
                "capture resource construction state is invalid"
            )
        booleans = frozenset(boolean_names)
        result: Dict[str, Any] = {"constructed": constructed}
        for name in names:
            value = 0 if not constructed else getattr(source, name, None)
            if name in booleans:
                if not constructed:
                    value = False
                if type(value) is not bool:
                    raise CalibrationEvidenceError(
                        f"capture resource boolean {name} is invalid"
                    )
            elif type(value) is not int or value < 0:
                raise CalibrationEvidenceError(
                    f"capture resource counter {name} is invalid"
                )
            result[name] = value
        return result

    def _powered_capture_resource_stats(self) -> Dict[str, Any]:
        """Take the sole raw, post-transport/pre-writer-close stats snapshot."""

        recorder_names = (
            "enqueued",
            "written",
            "dropped",
            "duplicate_frame_tokens",
            "writer_errors",
            "queue_high_watermark",
            "decoded_frames_enqueued",
            "decoded_frames_written",
            "decoded_frames_dropped",
            "complete",
        )
        vision_names = (
            "datagrams_received",
            "unique_datagrams",
            "duplicate_datagrams",
            "malformed_datagrams",
            "frames_reassembled",
            "frames_decoded",
            "decode_failures",
            "out_of_order_frame_drops",
            "reset_generation_drops",
            "processing_errors",
            "socket_errors",
            "snapshot_callback_errors",
            "resets",
            "remembered_chunk_keys",
            "timing_ledger_entries",
            "timing_ledger_high_watermark",
            "timing_ledger_capacity",
            "timing_overflow_latched",
            "receiver_buffered_partial_frames",
            "receiver_buffer_high_watermark",
            "receiver_buffer_capacity",
            "capture_snapshot_queue_entries",
            "capture_snapshot_queue_high_watermark",
            "capture_snapshot_queue_capacity",
            "capture_snapshot_queue_dropped",
            "capture_snapshot_queue_enabled",
            "receiver_dropped_partial_frames",
            "receiver_duplicate_chunks",
            "receiver_dropped_late_packets",
        )
        ingress_names = (
            "generation",
            "next_sequence",
            "highres_imu_received",
            "heartbeat_received",
            "race_status_received",
            "actuator_received",
            "dropped",
            "high_watermark",
            "imu_capacity",
            "other_capacity",
            "imu_dropped",
            "other_dropped",
            "imu_high_watermark",
            "other_high_watermark",
            "buffered_imu",
            "buffered_other",
        )
        collision_names = (
            "generation",
            "handled",
            "dropped",
            "high_watermark",
            "capacity",
            "buffered",
        )
        outbound_names = (
            "generation",
            "next_sequence",
            "returned",
            "raised",
            "dropped",
            "high_watermark",
            "capacity",
            "buffered",
        )

        replay = None if self.recorder is None else self.recorder.replay
        recorder_stats_method = None if replay is None else getattr(replay, "stats", None)
        if replay is not None and not callable(recorder_stats_method):
            raise CalibrationEvidenceError(
                "powered replay recorder lacks pre-close stats"
            )
        recorder_stats = None if replay is None else recorder_stats_method()

        vision_stats_method = None if self.vision is None else getattr(self.vision, "stats", None)
        if self.vision is not None and not callable(vision_stats_method):
            raise CalibrationEvidenceError("powered vision lacks final stats")
        vision_stats = None if self.vision is None else vision_stats_method()

        ingress_method = None if self.adapter is None else getattr(self.adapter, "ingress_stats", None)
        collision_method = None if self.adapter is None else getattr(self.adapter, "collision_stats", None)
        outbound_method = None if self.adapter is None else getattr(
            self.adapter, "outbound_receipt_stats", None
        )
        if self.adapter is not None and not all(
            callable(value)
            for value in (ingress_method, collision_method, outbound_method)
        ):
            raise CalibrationEvidenceError(
                "powered adapter lacks final capture resource stats"
            )
        ingress_stats = None if self.adapter is None else ingress_method()
        collision_stats = None if self.adapter is None else collision_method()
        outbound_stats = None if self.adapter is None else outbound_method()

        snapshot = {
            "constructed": self.capture is not None,
            "observed_frames": (
                0 if self.capture is None else self.capture.observed_frames
            ),
            "dimensions_admitted": bool(
                self.capture is not None and self.capture.admitted
            ),
            "failure_latched": bool(
                self.capture is not None and self.capture.failure is not None
            ),
        }
        if type(snapshot["observed_frames"]) is not int or snapshot[
            "observed_frames"
        ] < 0:
            raise CalibrationEvidenceError(
                "snapshot capture observed-frame count is invalid"
            )
        return {
            "schema": "aigp-vq2-powered-capture-resource-stats/1",
            "recorder": {
                **self._resource_counter_object(
                    recorder_stats,
                    recorder_names,
                    constructed=replay is not None,
                    boolean_names=("complete",),
                ),
                "failure_latched": bool(
                    recorder_stats is not None
                    and getattr(recorder_stats, "failure_reason", None) is not None
                ),
            },
            "vision": self._resource_counter_object(
                vision_stats,
                vision_names,
                constructed=self.vision is not None,
                boolean_names=(
                    "timing_overflow_latched",
                    "capture_snapshot_queue_enabled",
                ),
            ),
            "ingress": self._resource_counter_object(
                ingress_stats,
                ingress_names,
                constructed=self.adapter is not None,
            ),
            "collision": self._resource_counter_object(
                collision_stats,
                collision_names,
                constructed=self.adapter is not None,
            ),
            "outbound_receipts": self._resource_counter_object(
                outbound_stats,
                outbound_names,
                constructed=self.adapter is not None,
            ),
            "snapshot_capture": snapshot,
        }

    async def _close_artifacts(self, phase: Mapping[str, Any]) -> None:
        deadline = phase["deadline_monotonic_ns"]
        resource_stats = self._powered_capture_resource_stats()
        outcome = {
            "powered_stage_completed": self._stage_completed,
            "cleanup_certificate_outcome": (
                None if self._certificate is None else self._certificate["outcome"]
            ),
            "reason_codes": sorted(self._reason_codes),
            "vision_capture_stats": (
                {}
                if self.vision is None or not callable(getattr(self.vision, "stats", None))
                else {
                    name: getattr(self.vision.stats(), name)
                    for name in getattr(self.vision.stats(), "__dataclass_fields__", {})
                }
            ),
            "powered_capture_resource_stats": resource_stats,
        }
        closer = self.services.artifact_closer
        if closer is None:
            callback = partial(self._default_close_artifacts, outcome, deadline)
        else:
            callback = partial(
                closer,
                self.recorder,
                self.admission,
                outcome,
                deadline,
            )
        closed: Any = await self._run_supervised_callable(
            callback,
            deadline_ns=deadline,
            cleanup=False,
            replay_close=True,
        )
        if self._now() >= deadline:
            raise CalibrationEvidenceError("replay close completed too late")
        if isinstance(closed, CalibrationClosedArtifacts):
            artifacts = {
                "legacy_record": dict(closed.legacy_record),
                "replay_bundle": dict(closed.replay_bundle),
            }
        elif type(closed) is dict and set(closed) == {"legacy_record", "replay_bundle"}:
            artifacts = json.loads(json.dumps(closed, allow_nan=False))
        else:
            raise CalibrationEvidenceError("artifact closer returned an invalid proof")
        self._artifacts = artifacts
        self._replay_closed = True

    def _release_takeover(self) -> bool:
        if self._parent_mode != "signaled_takeover":
            return True
        if self._takeover_release_attempted:
            return self._takeover_released
        self._takeover_release_attempted = True
        self._takeover_released = False
        if self._lease_proof is None or self._lease_boundary_proof is None:
            return False
        now = self._now()
        deadline = min(
            self.admission.exit_deadline_monotonic_ns,
            now + self.durations["lease_release_and_verify"],
        )
        if now >= deadline:
            return False
        try:
            self._service_takeover_heartbeat(deadline)
            result = self.services.lease_boundary.release_takeover(
                self._lease_boundary_proof,
                deadline_monotonic_ns=deadline,
            )
        except BaseException:
            return False
        self._takeover_released = result is True and self._now() < deadline
        return self._takeover_released

    def _reason_for_exception(self, exc: BaseException) -> str:
        if isinstance(exc, asyncio.CancelledError):
            return "child_failed"
        if isinstance(exc, CalibrationCheckFailure):
            if exc.reason_code == "deadline_expired":
                return "deadline_expired"
            if exc.reason_code == "parent_dead":
                return "wrapper_death"
            if exc.reason_code == "capture_failed":
                return "capture_incomplete"
            if exc.reason_code in {"send_raised", "internal_error"}:
                return "command_reconciliation_failed"
            return "watchdog_failed"
        if isinstance(exc, CalibrationEvidenceError):
            return "capture_incomplete"
        return "internal_error"

    def _build_process_result(self, audit: Mapping[str, Any]) -> Dict[str, Any]:
        completed_monotonic_ns = self._now()
        if completed_monotonic_ns >= self.admission.exit_deadline_monotonic_ns:
            self._reason_codes.add("deadline_expired")
        if self._certificate_reference_state == "invalid":
            certificate_ref = {
                "path": self.admission.arguments.cleanup_certificate,
                "state": "invalid",
                "sha256": self._certificate_reference_sha256,
            }
            self._reason_codes.add("cleanup_unconfirmed")
        elif self._certificate is None or self._certificate_sha256 is None:
            certificate_ref = {
                "path": self.admission.arguments.cleanup_certificate,
                "state": "absent",
                "sha256": None,
            }
            self._reason_codes.add("cleanup_unconfirmed")
        else:
            certificate_ref = {
                "path": self.admission.arguments.cleanup_certificate,
                "state": "published",
                "sha256": self._certificate_sha256,
            }
            if self._certificate["outcome"] != "proved":
                self._reason_codes.add("cleanup_unconfirmed")
        if (
            self._artifacts["legacy_record"]["state"] != "closed"
            or self._artifacts["replay_bundle"]["state"] != "closed"
        ):
            self._reason_codes.add("capture_incomplete")
        if not self._stage_completed:
            self._reason_codes.add("child_failed")
        if not self._takeover_released:
            self._reason_codes.add("cleanup_unconfirmed")
        if "unexpected_outbound" in self._collection_codes:
            self._reason_codes.add("unexpected_outbound")
        if "collision_observed" in self._collection_codes:
            self._reason_codes.add("watchdog_failed")
        if "camera_missing" in self._collection_codes:
            self._reason_codes.add("capture_incomplete")
        reasons = sorted(self._reason_codes)
        result = {
            "schema": "aigp-vq2-powered-process-result/1",
            "task_id": self.contract.TASK_ID,
            "session_id": self.contract.SESSION_ID,
            "attempt_id": self.contract.ATTEMPT_ID,
            "producer_role": CALIBRATION_CHILD_ROLE,
            "process_authority_sha256": self.admission.process_authority_sha256,
            "started_monotonic_ns": self.admission.process_authority[
                "absolute_deadlines"
            ]["anchor"],
            "completed_monotonic_ns": completed_monotonic_ns,
            "outcome": "completed" if not reasons else "failed",
            "reason_codes": reasons,
            "phase_deadlines": list(self.phase_deadlines),
            "cleanup_certificate": certificate_ref,
            "outbound_audit": dict(audit),
            "artifacts": self._artifacts,
        }
        return self.contract.validate_process_result(
            result,
            cleanup_certificate=self._certificate,
        )

    async def run(self) -> CalibrationChildRunOutput:
        """Run every reached phase once and always enter cleanup after admission."""

        audit: Dict[str, Any] = {
            name: 0
            for name in (
                "timesync",
                "gcs_heartbeat",
                "sim_reset",
                "arm",
                "disarm",
                "attitude_target",
                "position_target",
                "other_command",
                "receipt_count",
                "receipt_returned",
                "receipt_raised",
                "receipt_dropped",
                "receipt_buffered",
            )
        }
        try:
            try:
                self._prepare_recorder()
                connect = self._phase("connect")
                await self._connect(connect)
                await self._preflight(self._phase("preflight"))
                await self._reset_epoch_phase(self._phase("reset_epoch"))
                await self._normalize_disarmed(self._phase("normalize_disarmed"))
                await self._countdown_go(self._phase("countdown_go"))
                await self._arm_confirmed(self._phase("arm"))
                await self._powered_stage(self._phase("powered_stage"))
                self._stage_completed = True
                self._trigger = "normal_completion"
            except BaseException as exc:
                self._reason_codes.add(self._reason_for_exception(exc))
                try:
                    if not self._parent_alive():
                        self._note_parent_death()
                except BaseException:
                    self._reason_codes.add("internal_error")
                if self.guards is not None:
                    try:
                        self.guards.latch_production("powered_stage_terminal")
                    except BaseException:
                        pass

            try:
                cleanup_phase = self._phase("cleanup")
                await self._cleanup(cleanup_phase)
            except BaseException as exc:
                self._reason_codes.add(self._reason_for_exception(exc))
                if not self._cleanup_failures:
                    self._cleanup_failures.add("internal_error")
                try:
                    await self._close_transports(
                        self.admission.cleanup_deadline_monotonic_ns
                    )
                except BaseException:
                    self._cleanup_failures.add("transport_unclosed")

            try:
                self._post_cleanup_parent_check(
                    parent_deadline_ns=self.admission.cleanup_deadline_monotonic_ns,
                    emit=self.recorder is not None and not self._replay_closed,
                )
                certificate, audit = self._build_cleanup_certificate()
                self._certificate, self._certificate_sha256 = self._publish_certificate(
                    certificate
                )
            except BaseException as exc:
                self._reason_codes.add(self._reason_for_exception(exc))
                self._reason_codes.add("cleanup_unconfirmed")
                self._preserve_failed_certificate_reference()
                try:
                    audit = self._outbound_audit()
                except BaseException:
                    self._reason_codes.add("command_reconciliation_failed")

            # A takeover observed during cleanup or through certificate
            # publication must be released before replay.close().  The lease
            # publisher owns release-intent publication, release, proof, and
            # final lease-index publication as one bounded operation.
            try:
                self._post_cleanup_parent_check(
                    parent_deadline_ns=self.admission.exit_deadline_monotonic_ns,
                    emit=self.recorder is not None and not self._replay_closed,
                )
            except BaseException:
                self._reason_codes.add("cleanup_unconfirmed")
            self._takeover_released = self._release_takeover()
            if not self._takeover_released:
                self._reason_codes.add("cleanup_unconfirmed")

            try:
                replay_phase = self._phase("replay_close")
                await self._close_artifacts(replay_phase)
            except BaseException as exc:
                self._reason_codes.add(self._reason_for_exception(exc))
                self._reason_codes.add("capture_incomplete")
                self._artifacts = self._partial_artifacts("partial")
                self._replay_closed = True

            # A parent that dies only after replay closure still requires a
            # bounded takeover/release, but cleanup commands are never replayed.
            try:
                took_over = self._post_cleanup_parent_check(
                    parent_deadline_ns=self.admission.exit_deadline_monotonic_ns,
                    emit=False,
                )
                if took_over:
                    self._takeover_released = self._release_takeover()
            except BaseException:
                self._reason_codes.add("cleanup_unconfirmed")
            if not self._takeover_released:
                self._reason_codes.add("cleanup_unconfirmed")
            try:
                self._phase("finalize", emit=False)
            except BaseException:
                self._reason_codes.add("deadline_expired")
            try:
                took_over = self._post_cleanup_parent_check(
                    parent_deadline_ns=self.admission.exit_deadline_monotonic_ns,
                    emit=False,
                )
                if took_over and not self._takeover_released:
                    self._takeover_released = self._release_takeover()
            except BaseException:
                self._reason_codes.add("cleanup_unconfirmed")
            process_result = self._build_process_result(audit)
            try:
                took_over = self._post_cleanup_parent_check(
                    parent_deadline_ns=self.admission.exit_deadline_monotonic_ns,
                    emit=False,
                )
                if took_over and not self._takeover_released:
                    self._takeover_released = self._release_takeover()
                    process_result = self._build_process_result(audit)
            except BaseException:
                self._reason_codes.add("cleanup_unconfirmed")
                process_result = self._build_process_result(audit)
            return CalibrationChildRunOutput(
                certificate=self._certificate,
                certificate_sha256=self._certificate_sha256,
                process_result=process_result,
                exit_code=0 if process_result["outcome"] == "completed" else 1,
            )
        finally:
            self.admission.erase_role_secret()


async def run_powered_calibration_child(
    admission: CalibrationAdmission,
    services: CalibrationChildServices,
) -> CalibrationChildRunOutput:
    return await CalibrationChildLifecycle(admission, services).run()


def next_control_deadline(
    previous_deadline_s: float,
    now_s: float,
    period_s: float = CONTROL_PERIOD_S,
) -> float:
    """Pace setpoints without replaying missed ticks after a loop stall.

    A stale absolute schedule can emit back-to-back commands while catching up.
    Keep the nominal grid only when it also leaves a complete period after the
    most recent send; otherwise drop the missed ticks.
    """

    values = (previous_deadline_s, now_s, period_s)
    if not all(math.isfinite(value) for value in values) or period_s <= 0.0:
        raise ValueError("control pacing inputs must be finite and period_s > 0")
    return max(previous_deadline_s + period_s, now_s + period_s)


@dataclass(frozen=True)
class GateTarget:
    frame_id: int
    sim_time_ns: int
    received_monotonic_s: float
    center_x: int
    center_y: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    composite: bool = False

    @property
    def bbox_area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    def age_s(self, now: Optional[float] = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.received_monotonic_s)


@dataclass(frozen=True)
class GateFragmentUnion:
    upper: GateDetection
    lower: GateDetection
    bbox: Tuple[int, int, int, int]
    center_x: int
    center_y: int
    confidence: float


@dataclass(frozen=True)
class ResetProof:
    attempt: int
    pre_race_boot_ms: int
    post_race_boot_ms: int
    pre_imu_us: int
    post_imu_us: int
    advancing_race_samples: int
    advancing_imu_samples: int
    countdown_observed: bool


@dataclass(frozen=True)
class StartContext:
    spawn_roll_rad: float
    spawn_pitch_rad: float
    initial_gate_x: int
    initial_gate_y: int
    initial_gate_area: int
    go_boot_ms: int


@dataclass(frozen=True)
class GateTransitionProof:
    """Internal authority and timing handoff from gate 0 to observation."""

    pre_gate_race_boot_ms: int
    post_gate_race_boot_ms: int
    flight_started_monotonic_s: float
    crossing_started_monotonic_s: Optional[float]
    pass_confirmed_monotonic_s: float
    next_control_deadline_s: float
    vision_generation: int
    vision_frame_id: int
    vision_sim_time_ns: int
    vision_received_monotonic_s: float
    pass_rpy_rad: Tuple[float, float, float]


@dataclass(frozen=True)
class CourseLineObservation:
    """Pixel-space direction of the cyan racing line across two image bands."""

    turn_score: float
    upper_center_x: float
    lower_center_x: float
    upper_pixel_count: int
    lower_pixel_count: int


@dataclass(frozen=True)
class StageResult:
    stage: str
    success: bool
    reason: str
    duration_s: float
    gate_index_before: Optional[int] = None
    gate_index_after: Optional[int] = None
    cleanup_confirmed: bool = False
    details: Optional[Dict[str, Any]] = None
    controller: Optional[Dict[str, Any]] = None


def controller_config_evidence(
    config: VQ2ControllerConfig | VisualNavigationConfig,
    *,
    candidate_commit: Optional[str],
) -> Dict[str, Any]:
    """Bind one normalized effective controller to its exact source commit."""

    if candidate_commit is not None and (
        type(candidate_commit) is not str
        or len(candidate_commit) != 40
        or any(character not in "0123456789abcdef" for character in candidate_commit)
    ):
        raise ValueError("candidate_commit must be 40 lowercase hexadecimal characters")
    effective = config.to_effective_mapping()
    return {
        "git_commit": candidate_commit,
        "config_schema": config.schema,
        "controller_family": config.controller_family,
        "config_sha256": config.effective_config_sha256,
        "effective_parameters": {
            key: value
            for key, value in effective.items()
            if key not in {"schema", "controller_family"}
        },
    }


def replay_controller_envelope(stage: str) -> Dict[str, Any]:
    """Return capture-level limits, with phase detail for visual alignment."""

    if type(stage) is not str or stage not in LIVE_RUN_STAGES:
        raise ValueError("replay controller envelope requires a live stage")
    if stage == VISUAL_ALIGN_STAGE:
        return {
            "control_hz": CONTROL_HZ,
            "max_roll_pitch_command_rate_rad_s": MAX_COMMAND_RATE_RAD_S,
            "yaw_rate_rad_s": VISUAL_ALIGN_MAX_YAW_RATE_RAD_S,
            "max_thrust": MAX_VISUAL_THRUST,
            "hard_stage_duration_s": None,
            "phase_envelopes": {
                "gate0_visual_handoff_bootstrap": {
                    "max_roll_pitch_command_rate_rad_s": (
                        MAX_COMMAND_RATE_RAD_S
                    ),
                    "yaw_rate_rad_s": VISUAL_ALIGN_MAX_YAW_RATE_RAD_S,
                    "max_thrust": MAX_VISUAL_THRUST,
                    "hard_duration_s": GATE0_FLIGHT_TIMEOUT_S,
                    "axis_authority": {
                        "yaw": "visual_next_track_blend",
                        "pitch": "bounded_visual_brake",
                        "roll_collective": "proved_gate0_bootstrap",
                    },
                },
                "crossing_confirmation": {
                    "exact_zero_rate_zero_thrust": True,
                    "max_wait_s": CROSSING_STATUS_TIMEOUT_S,
                },
                "post_credit_fresh_frame_wait": {
                    "exact_zero_rate_zero_thrust": True,
                    "max_wait_s": (
                        VISUAL_ALIGN_POST_CREDIT_FRAME_TIMEOUT_S
                    ),
                },
                "restricted_post_promotion_alignment": {
                    "max_roll_pitch_command_rate_rad_s": (
                        VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S
                    ),
                    "yaw_rate_rad_s": VISUAL_ALIGN_MAX_YAW_RATE_RAD_S,
                    "max_thrust": VISUAL_ALIGN_MAX_THRUST,
                    "hard_duration_s": VISUAL_ALIGN_HARD_DURATION_S,
                },
            },
        }
    return {
        "control_hz": CONTROL_HZ,
        "max_roll_pitch_command_rate_rad_s": MAX_COMMAND_RATE_RAD_S,
        "yaw_rate_rad_s": 0.0,
        "max_thrust": 0.35,
        "hard_stage_duration_s": None,
    }


def clock_rolled_back(pre_value: int, current_value: int, margin: int) -> bool:
    """Whether a simulator clock is authoritatively below its prior epoch."""

    return int(current_value) < int(pre_value) - int(margin)


def clock_within_epoch_envelope(
    anchor_value: int,
    current_value: int,
    elapsed_s: float,
    *,
    units_per_second: float,
    slack: int,
) -> bool:
    """Reject delayed packets whose clock is impossible in the proved epoch."""

    maximum = int(anchor_value) + int(max(0.0, elapsed_s) * units_per_second) + int(slack)
    return int(current_value) <= maximum


def cyan_course_line_observation(image: Any) -> Optional[CourseLineObservation]:
    """Measure a turn cue from the build-3385 cyan racing line."""

    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 3 or tuple(shape[:2]) != (360, 640):
        return None
    try:
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((85, 50, 40), dtype=np.uint8),
            np.array((105, 255, 255), dtype=np.uint8),
        )
        mask[:115, :] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        component_count, labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(mask, connectivity=8)
        )
        retained = np.zeros_like(mask)
        for label in range(1, component_count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= 8:
                retained[labels == label] = 255
        mask = retained
    except (ImportError, TypeError, ValueError):
        return None

    def band_center(y0: int, y1: int) -> Tuple[float, int]:
        _ys, xs = np.nonzero(mask[y0:y1, :])
        count = int(xs.size)
        return (float(xs.mean()) if count else math.nan, count)

    upper_x, upper_count = band_center(115, 145)
    lower_x, lower_count = band_center(145, 180)
    if (
        upper_count < COURSE_LINE_MIN_ROI_PIXELS
        or lower_count < COURSE_LINE_MIN_ROI_PIXELS
    ):
        return None
    score = max(-1.0, min(1.0, (upper_x - lower_x) / 320.0))
    return CourseLineObservation(
        turn_score=score,
        upper_center_x=upper_x,
        lower_center_x=lower_x,
        upper_pixel_count=upper_count,
        lower_pixel_count=lower_count,
    )


def course_line_preturn_roll(
    turn_score: float,
    *,
    gain: float = COURSE_LINE_PRETURN_GAIN,
    cap_rad: float = COURSE_LINE_PRETURN_LIMIT_RAD,
    min_abs_score: float = COURSE_LINE_PRETURN_MIN_SCORE,
) -> float:
    """Convert a proved cyan-line direction into a small Gate-0 bank bias."""

    values = (turn_score, gain, cap_rad, min_abs_score)
    if (
        any(type(value) not in {int, float} for value in values)
        or not all(math.isfinite(float(value)) for value in values)
        or not 0.0 <= float(gain) <= COURSE_LINE_PRETURN_GAIN
        or not 0.0 <= float(cap_rad) <= COURSE_LINE_PRETURN_LIMIT_RAD
        or not COURSE_LINE_PRETURN_MIN_SCORE
        <= float(min_abs_score)
        <= 0.25
    ):
        raise ValueError("course-line turn score must be finite and numeric")
    score = float(turn_score)
    if abs(score) < float(min_abs_score):
        return 0.0
    # This is a physical course-turn bias, not an image-centering correction.
    # A route bending toward larger image x requests the simulator's positive
    # body roll even though the camera rotation initially moves the gate right.
    return max(
        -float(cap_rad),
        min(float(cap_rad), float(gain) * score),
    )


def course_line_exit_counterroll(
    turn_score: float,
    *,
    cap_rad: float = COURSE_LINE_EXIT_COUNTERROLL_RAD,
    min_abs_score: float = COURSE_LINE_PRETURN_MIN_SCORE,
) -> float:
    """Counter-roll a proved course preturn before the Gate-0 crossing coast."""

    values = (turn_score, cap_rad, min_abs_score)
    if (
        any(type(value) not in {int, float} for value in values)
        or not all(math.isfinite(float(value)) for value in values)
        or not 0.0 <= float(cap_rad) <= COURSE_LINE_EXIT_COUNTERROLL_RAD
        or not COURSE_LINE_PRETURN_MIN_SCORE
        <= float(min_abs_score)
        <= 0.25
    ):
        raise ValueError("course-line exit-counterroll score must be finite and numeric")
    score = float(turn_score)
    if abs(score) < float(min_abs_score) or float(cap_rad) == 0.0:
        return 0.0
    return -math.copysign(float(cap_rad), score)


def gate0_centering_roll_target(
    normalized_x: float,
    *,
    gain: float = 0.15,
    cap_rad: float = 0.08,
) -> float:
    """Apply the previously live-proved Gate 0 image-centering law."""

    values = (normalized_x, gain, cap_rad)
    if (
        any(type(value) not in {int, float} for value in values)
        or not all(math.isfinite(float(value)) for value in values)
        or not 0.0 <= float(gain) <= 0.15
        or not 0.0 <= float(cap_rad) <= 0.08
    ):
        raise ValueError("gate-0 centering input must be finite and numeric")
    return max(
        -float(cap_rad),
        min(float(cap_rad), float(gain) * float(normalized_x)),
    )


def gate1_recenter_roll_target(
    normalized_x: float,
    normalized_x_rate_s: float,
    *,
    error_gain: float = GATE1_RECENTER_ROLL_GAIN,
    error_rate_gain: float = GATE1_RECENTER_ROLL_RATE_GAIN,
    cap_rad: float = GATE1_RECENTER_MAX_ROLL_RAD,
) -> float:
    """Return the live-corrected, bounded Gate 1 recenter roll target."""

    values = (
        normalized_x,
        normalized_x_rate_s,
        error_gain,
        error_rate_gain,
        cap_rad,
    )
    if (
        any(type(value) not in {int, float} for value in values)
        or not all(math.isfinite(float(value)) for value in values)
        or abs(float(normalized_x_rate_s))
        > GATE1_RECENTER_MAX_ABS_X_RATE_NORM_S
        or not -0.24 <= float(error_gain) <= 0.24
        or not -0.025 <= float(error_rate_gain) <= 0.025
        or not 0.0 <= float(cap_rad) <= GATE1_RECENTER_MAX_ROLL_RAD
    ):
        raise ValueError("gate-1 recenter horizontal inputs are outside bounds")
    target = (
        float(error_gain) * float(normalized_x)
        + float(error_rate_gain) * float(normalized_x_rate_s)
    )
    return max(
        -float(cap_rad),
        min(float(cap_rad), target),
    )


def course_line_turn_yaw_rate(
    turn_score: float,
    *,
    gain: float,
    cap_rad_s: float,
    min_abs_score: float,
) -> float:
    """Map a proved course turn to calibrated bounded local yaw."""

    values = (turn_score, gain, cap_rad_s, min_abs_score)
    if (
        any(type(value) not in {int, float} for value in values)
        or not all(math.isfinite(float(value)) for value in values)
        or not -0.80 <= float(gain) <= 0.0
        or not 0.0 <= float(cap_rad_s) <= SIGN_ID_RATE_RAD_S
        or not COURSE_LINE_PRETURN_MIN_SCORE
        <= float(min_abs_score)
        <= 0.25
        or (float(gain) == 0.0) != (float(cap_rad_s) == 0.0)
    ):
        raise ValueError("course-line yaw control is outside bounds")
    if abs(float(turn_score)) < float(min_abs_score):
        return 0.0
    return max(
        -float(cap_rad_s),
        min(float(cap_rad_s), float(gain) * float(turn_score)),
    )


def gate1_recenter_yaw_rate(
    normalized_x: float,
    *,
    error_gain: float,
    deadband_normalized_x: float,
    cap_rad_s: float,
) -> float:
    """Apply calibrated negative image-error feedback through bounded yaw."""

    values = (
        normalized_x,
        error_gain,
        deadband_normalized_x,
        cap_rad_s,
    )
    if (
        any(type(value) not in {int, float} for value in values)
        or not all(math.isfinite(float(value)) for value in values)
        or abs(float(normalized_x)) > 1.0
        or not -0.12 <= float(error_gain) <= 0.0
        or not 0.20 <= float(deadband_normalized_x) <= 0.35
        or not 0.0 <= float(cap_rad_s) <= SIGN_ID_RATE_RAD_S
        or (float(error_gain) == 0.0) != (float(cap_rad_s) == 0.0)
    ):
        raise ValueError("gate-1 recenter yaw control is outside bounds")
    if abs(float(normalized_x)) <= float(deadband_normalized_x):
        return 0.0
    return max(
        -float(cap_rad_s),
        min(float(cap_rad_s), float(error_gain) * float(normalized_x)),
    )


def gate1_recenter_absolute_error_slope_px_s(
    samples: Sequence[Tuple[float, float]],
) -> Optional[float]:
    """Least-squares slope of fresh-frame absolute horizontal error."""

    if not isinstance(samples, Sequence):
        raise TypeError("gate-1 error samples must be a sequence")
    if len(samples) < 2:
        return None
    normalized: List[Tuple[float, float]] = []
    for sample in samples:
        if (
            not isinstance(sample, Sequence)
            or len(sample) != 2
            or any(type(value) not in {int, float} for value in sample)
            or not all(math.isfinite(float(value)) for value in sample)
        ):
            raise ValueError("gate-1 error samples must contain finite pairs")
        normalized.append((float(sample[0]), float(sample[1])))
    if any(
        normalized[index][0] <= normalized[index - 1][0]
        for index in range(1, len(normalized))
    ):
        raise ValueError("gate-1 error sample times must increase strictly")
    mean_time = statistics.fmean(sample[0] for sample in normalized)
    mean_error = statistics.fmean(sample[1] for sample in normalized)
    denominator = sum(
        (sample[0] - mean_time) ** 2 for sample in normalized
    )
    if denominator <= 0.0:
        raise ValueError("gate-1 error sample times have no span")
    return sum(
        (sample[0] - mean_time) * (sample[1] - mean_error)
        for sample in normalized
    ) / denominator


def select_primary_gate(
    detections: Iterable[GateDetection],
) -> Optional[GateDetection]:
    """Select the largest plausible gate using only pixel geometry.

    Metric distance and detector-provided corners are intentionally ignored:
    their current scale is a placeholder.  Near-square filtering rejects the
    cyan racing line, starting lights, and thin fragments.
    """

    candidates: List[GateDetection] = []
    for detection in detections:
        _x, _y, width, height = detection.bbox
        if width < 20 or height < 20:
            continue
        short = min(width, height)
        long = max(width, height)
        if short <= 0 or long / short > 1.85:
            continue
        if not math.isfinite(detection.confidence) or detection.confidence < 0.10:
            continue
        candidates.append(detection)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.bbox[2] * item.bbox[3])


def _bbox_iou(
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    first_x, first_y, first_width, first_height = (int(value) for value in first)
    second_x, second_y, second_width, second_height = (
        int(value) for value in second
    )
    intersection_width = max(
        0,
        min(first_x + first_width, second_x + second_width)
        - max(first_x, second_x),
    )
    intersection_height = max(
        0,
        min(first_y + first_height, second_y + second_height)
        - max(first_y, second_y),
    )
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def select_tracked_fragment_union(
    detections: Iterable[GateDetection],
    *,
    prior_target: GateTarget,
    image_width: int,
    image_height: int,
) -> Optional[GateFragmentUnion]:
    """Fuse the proved complementary pieces of one top-clipped course gate.

    Build 3385 splits Gate 1 into an upper/right and lower/left contour while
    their union remains a stable near-square continuation of the prior gate.
    This selector is deliberately narrower than normal tracking and supplies
    guidance geometry only; a composite target can never arm a crossing.
    """

    if (
        type(image_width) is not int
        or type(image_height) is not int
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("fragment-union image dimensions must be positive ints")
    prior_x, prior_y, prior_width, prior_height = prior_target.bbox
    if (
        prior_x < 0
        or prior_y > COURSE_EDGE_CONTINUATION_MARGIN_PX
        or prior_width < 20
        or prior_height < 20
        or prior_x + prior_width > image_width
        or prior_y + prior_height > image_height
    ):
        return None

    valid: List[GateDetection] = []
    for detection in detections:
        x, y, width, height = (int(value) for value in detection.bbox)
        if (
            x < 0
            or y < 0
            or width < 20
            or height < 20
            or x + width > image_width
            or y + height > image_height
        ):
            continue
        short = min(width, height)
        long = max(width, height)
        if short <= 0 or long / short > COURSE_EDGE_CONTINUATION_MAX_ASPECT_RATIO:
            continue
        if not math.isfinite(detection.confidence) or detection.confidence < 0.10:
            continue
        valid.append(detection)

    candidates: List[
        Tuple[float, float, float, GateFragmentUnion]
    ] = []
    prior_area = max(1, prior_target.bbox_area)
    for upper in valid:
        upper_x, upper_y, upper_width, upper_height = (
            int(value) for value in upper.bbox
        )
        if upper_y > COURSE_EDGE_CONTINUATION_MARGIN_PX:
            continue
        for lower in valid:
            if lower is upper:
                continue
            lower_x, lower_y, lower_width, lower_height = (
                int(value) for value in lower.bbox
            )
            if (
                lower_y <= COURSE_EDGE_CONTINUATION_MARGIN_PX
                or int(upper.center_x) <= int(lower.center_x)
                or int(upper.center_y) >= int(lower.center_y)
            ):
                continue

            union_x = min(upper_x, lower_x)
            union_y = min(upper_y, lower_y)
            union_right = max(upper_x + upper_width, lower_x + lower_width)
            union_bottom = max(upper_y + upper_height, lower_y + lower_height)
            union_width = union_right - union_x
            union_height = union_bottom - union_y
            if union_width < 20 or union_height < 20:
                continue
            union_aspect = max(union_width, union_height) / min(
                union_width, union_height
            )
            union_aspect_limit = COURSE_FRAGMENT_UNION_MAX_ASPECT_RATIO
            if prior_target.composite and union_right == image_width:
                # A faster right turn made the same proved Gate-1 union two
                # frames wider while its upper fragment met the image edge.
                # Keep this bridge edge- and composite-specific; ordinary
                # unions retain the stricter near-square bound.
                union_aspect_limit = (
                    COURSE_FRAGMENT_UNION_RIGHT_EDGE_MAX_ASPECT_RATIO
                )
            if union_aspect > union_aspect_limit:
                continue

            horizontal_gap = max(
                0,
                max(upper_x, lower_x)
                - min(upper_x + upper_width, lower_x + lower_width),
            )
            vertical_gap = max(
                0,
                max(upper_y, lower_y)
                - min(upper_y + upper_height, lower_y + lower_height),
            )
            if (
                horizontal_gap > 0.45 * union_width
                or vertical_gap > 0.20 * union_height
            ):
                continue

            intersection_width = max(
                0,
                min(upper_x + upper_width, lower_x + lower_width)
                - max(upper_x, lower_x),
            )
            intersection_height = max(
                0,
                min(upper_y + upper_height, lower_y + lower_height)
                - max(upper_y, lower_y),
            )
            visible_support = (
                upper_width * upper_height
                + lower_width * lower_height
                - intersection_width * intersection_height
            )
            union_area = union_width * union_height
            support_ratio = visible_support / union_area
            if not 0.20 <= support_ratio <= 0.80:
                continue

            bbox = (union_x, union_y, union_width, union_height)
            overlap = _bbox_iou(bbox, prior_target.bbox)
            center_x = union_x + union_width // 2
            center_y = union_y + union_height // 2
            center_jump = math.hypot(
                center_x - prior_target.center_x,
                center_y - prior_target.center_y,
            )
            area_ratio = union_area / prior_area
            if (
                overlap < COURSE_FRAGMENT_UNION_MIN_IOU
                or center_jump > 32.0
                or not 0.70 <= area_ratio <= 1.35
            ):
                continue
            union = GateFragmentUnion(
                upper=upper,
                lower=lower,
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                confidence=min(float(upper.confidence), float(lower.confidence)),
            )
            candidates.append((-overlap, center_jump, -support_ratio, union))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[3]


def _vertical_frame_edge(
    bbox: Sequence[int],
    *,
    image_width: int,
) -> int:
    """Return the touched vertical edge without treating full-width blobs as gates."""

    x, _y, width, _height = (int(value) for value in bbox)
    touches_left = x <= COURSE_EDGE_CONTINUATION_MARGIN_PX
    touches_right = (
        x + width >= image_width - COURSE_EDGE_CONTINUATION_MARGIN_PX
    )
    if touches_left == touches_right:
        return 0
    return -1 if touches_left else 1


def select_tracked_edge_continuation(
    detections: Iterable[GateDetection],
    *,
    prior_target: GateTarget,
    image_width: int,
    image_height: int,
) -> Optional[GateDetection]:
    """Recover only the clipped continuation of a gate already at the right edge.

    Build 3385 can turn a valid near-square target into a thin vertical fragment
    for the last few frames before it leaves view.  The ordinary selector must
    keep rejecting such fragments globally.  This narrow selector is eligible
    only when both the proved prior target and candidate touch the right frame
    edge and satisfy the much tighter geometry proved by the live trace.
    """

    if (
        type(image_width) is not int
        or type(image_height) is not int
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("edge-continuation image dimensions must be positive ints")
    if prior_target.composite:
        return None
    prior_x, prior_y, prior_width, prior_height = prior_target.bbox
    if (
        prior_x < 0
        or prior_y < 0
        or prior_width < 20
        or prior_height < 20
        or prior_x + prior_width > image_width + COURSE_EDGE_CONTINUATION_MARGIN_PX
        or prior_y + prior_height
        > image_height + COURSE_EDGE_CONTINUATION_MARGIN_PX
    ):
        return None
    prior_edge = _vertical_frame_edge(prior_target.bbox, image_width=image_width)
    if (
        prior_edge != 1
        or prior_y <= COURSE_EDGE_CONTINUATION_MARGIN_PX
        or prior_y + prior_height
        >= image_height - COURSE_EDGE_CONTINUATION_MARGIN_PX
    ):
        return None

    candidates: List[Tuple[float, float, int, GateDetection]] = []
    prior_area = max(1, prior_target.bbox_area)
    for detection in detections:
        x, y, width, height = (int(value) for value in detection.bbox)
        if (
            x < 0
            or y < 0
            or width < 20
            or height < 20
            or x + width > image_width + COURSE_EDGE_CONTINUATION_MARGIN_PX
            or y + height > image_height + COURSE_EDGE_CONTINUATION_MARGIN_PX
        ):
            continue
        short = min(width, height)
        long = max(width, height)
        if short <= 0 or long / short > COURSE_EDGE_CONTINUATION_MAX_ASPECT_RATIO:
            continue
        if not math.isfinite(detection.confidence) or detection.confidence < 0.10:
            continue
        if _vertical_frame_edge(detection.bbox, image_width=image_width) != 1:
            continue
        if (
            y <= COURSE_EDGE_CONTINUATION_MARGIN_PX
            or y + height >= image_height - COURSE_EDGE_CONTINUATION_MARGIN_PX
        ):
            continue
        dx = int(detection.center_x) - prior_target.center_x
        dy = int(detection.center_y) - prior_target.center_y
        center_jump = math.hypot(dx, dy)
        area = int(width) * int(height)
        area_ratio = area / prior_area
        vertical_overlap = min(prior_y + prior_height, y + height) - max(
            prior_y, y
        )
        if (
            center_jump > 32.0
            or not -3 <= dx <= 20
            or not 0.20 <= area_ratio <= 1.10
            or vertical_overlap < 0.50 * min(prior_height, height)
        ):
            continue
        candidates.append(
            (center_jump, abs(math.log(area_ratio)), -area, detection)
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[3]


class GateTargetTracker:
    """Small temporal gate filter for the first bounded vision-only run."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.target: Optional[GateTarget] = None
        self.consecutive = 0
        self._last_frame_id: Optional[int] = None
        self.last_selected_detection: Optional[GateDetection] = None
        self.last_selected_detections: Tuple[GateDetection, ...] = ()
        self.last_selection_mode: Optional[str] = None

    def update(
        self,
        detections: Iterable[GateDetection],
        *,
        frame_id: int,
        sim_time_ns: int,
        received_monotonic_s: float,
        allow_tracked_edge_continuation: bool = False,
        allow_tracked_fragment_union: bool = False,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Optional[GateTarget]:
        if self._last_frame_id == int(frame_id):
            return self.target
        self._last_frame_id = int(frame_id)
        detection_list = list(detections)
        previous_target = self.target
        previous_streak = self.consecutive

        def candidate_for(detection: GateDetection) -> GateTarget:
            return GateTarget(
                frame_id=int(frame_id),
                sim_time_ns=int(sim_time_ns),
                received_monotonic_s=float(received_monotonic_s),
                center_x=int(detection.center_x),
                center_y=int(detection.center_y),
                bbox=tuple(int(value) for value in detection.bbox),
                confidence=float(detection.confidence),
            )

        def candidate_for_union(union: GateFragmentUnion) -> GateTarget:
            return GateTarget(
                frame_id=int(frame_id),
                sim_time_ns=int(sim_time_ns),
                received_monotonic_s=float(received_monotonic_s),
                center_x=union.center_x,
                center_y=union.center_y,
                bbox=union.bbox,
                confidence=union.confidence,
                composite=True,
            )

        def is_continuous(candidate: GateTarget) -> bool:
            if previous_target is None:
                return False
            dx = candidate.center_x - previous_target.center_x
            dy = candidate.center_y - previous_target.center_y
            center_jump = math.hypot(dx, dy)
            prior_area = max(1, previous_target.bbox_area)
            area_ratio = candidate.bbox_area / prior_area
            return center_jump <= 100.0 and 0.20 <= area_ratio <= 5.0

        selected: Optional[GateDetection] = None
        selected_detections: Tuple[GateDetection, ...] = ()
        candidate: Optional[GateTarget] = None
        selection_mode: Optional[str] = None
        continuous = False

        if (
            allow_tracked_fragment_union
            and previous_target is not None
            and previous_streak >= POST_GATE_REQUIRED_FRAMES
        ):
            if image_width is None or image_height is None:
                raise ValueError(
                    "image dimensions are required for tracked fragment union"
                )
            union = select_tracked_fragment_union(
                detection_list,
                prior_target=previous_target,
                image_width=image_width,
                image_height=image_height,
            )
            if union is not None:
                union_candidate = candidate_for_union(union)
                if is_continuous(union_candidate):
                    candidate = union_candidate
                    continuous = True
                    selected_detections = (union.upper, union.lower)
                    selection_mode = "tracked_fragment_union"

        if candidate is None:
            primary = select_primary_gate(detection_list)
            primary_candidate = candidate_for(primary) if primary is not None else None
            primary_continuous = (
                primary_candidate is not None and is_continuous(primary_candidate)
            )
            if primary_candidate is not None and (
                previous_target is None or primary_continuous
            ):
                selected = primary
                selected_detections = (primary,) if primary is not None else ()
                candidate = primary_candidate
                continuous = primary_continuous
                selection_mode = "primary"

        if (
            candidate is None
            and allow_tracked_edge_continuation
            and previous_target is not None
            and previous_streak >= POST_GATE_REQUIRED_FRAMES
        ):
            if image_width is None or image_height is None:
                raise ValueError(
                    "image dimensions are required for tracked edge continuation"
                )
            selected = select_tracked_edge_continuation(
                detection_list,
                prior_target=previous_target,
                image_width=image_width,
                image_height=image_height,
            )
            if selected is not None:
                candidate = candidate_for(selected)
                continuous = is_continuous(candidate)
                if continuous:
                    selected_detections = (selected,)
                    selection_mode = "tracked_edge_continuation"
                else:
                    selected = None
                    candidate = None
        self.last_selected_detection = selected
        self.last_selected_detections = selected_detections
        self.last_selection_mode = selection_mode
        if candidate is None:
            self.consecutive = 0
            return None
        self.consecutive = previous_streak + 1 if continuous else 1
        self.target = candidate
        return candidate

    def fresh(self, max_age_s: float, now: Optional[float] = None) -> bool:
        return self.target is not None and self.target.age_s(now) <= max_age_s


def gate_vertical_reference_px(
    initial_gate_y: float,
    spawn_pitch_rad: float,
    target_pitch_rad: float,
    *,
    focal_length_px: float = 320.0,
) -> float:
    """Expected gate row after changing pitch without translating."""

    delta = float(target_pitch_rad) - float(spawn_pitch_rad)
    delta = max(math.radians(-35.0), min(math.radians(35.0), delta))
    return float(initial_gate_y) + float(focal_length_px) * math.tan(delta)


def gate0_target_pitch_rad(
    spawn_pitch_rad: float,
    exit_pitch_rad: float,
    elapsed_s: float,
    *,
    blend_duration_s: float = GATE0_PITCH_BLEND_S,
) -> float:
    """Blend the Gate 0 pitch basis to a bounded full-lap exit pitch."""

    values = (spawn_pitch_rad, exit_pitch_rad, elapsed_s, blend_duration_s)
    if (
        any(type(value) not in {int, float} for value in values)
        or not all(math.isfinite(float(value)) for value in values)
        or not -0.10 <= float(exit_pitch_rad) <= 0.0
        or float(elapsed_s) < 0.0
        or not GATE0_PITCH_BLEND_S <= float(blend_duration_s) <= 1.0
    ):
        raise ValueError("gate-0 pitch schedule inputs are outside the envelope")
    blend = min(1.0, float(elapsed_s) / float(blend_duration_s))
    return (
        (1.0 - blend) * float(spawn_pitch_rad)
        + blend * float(exit_pitch_rad)
    )


def gate_control_center_y_px(
    target: GateTarget,
    image_height: int = 360,
    *,
    previous_center_y: Optional[float] = None,
) -> float:
    """Estimate outer-gate center when its lower edge clips the image.

    A head-on VQ2 gate is square.  Once the lower edge leaves the 360 px frame,
    the raw bbox center is biased sharply upward.  While left/right remain
    visible, width is the reliable side length, so infer the missing vertical
    half from it.  Otherwise retain the measured center and let freshness/
    continuity guards stop on ambiguous fragments.
    """

    x, y, width, height = target.bbox
    top_clipped = y <= 2
    bottom_clipped = y + height >= int(image_height) - 2
    horizontally_visible = x > 1 and x + width < 640 - 2
    if top_clipped and bottom_clipped:
        # No vertical edge remains from which to infer the square center.  A
        # fully clipped close-range frame must not create a fictitious jump;
        # hold the last bottom-only estimate for the few frames until pass.
        if previous_center_y is not None and math.isfinite(previous_center_y):
            return float(previous_center_y)
        return float(target.center_y)
    if bottom_clipped and horizontally_visible and width > height:
        return float(y) + 0.5 * float(width)
    return float(target.center_y)


def gate_vertical_thrust(control_y: float, control_y_rate: float) -> float:
    """Pixel-space PD for gate altitude, bounded to the validated envelope."""

    if not math.isfinite(control_y) or not math.isfinite(control_y_rate):
        raise SafetyAbort("non-finite gate vertical control input")
    proportional = 0.040 * max(-1.0, min(1.0, (180.0 - control_y) / 90.0))
    damping = -0.00070 * max(-300.0, min(300.0, control_y_rate))
    return max(0.21, min(0.32, 0.275 + proportional + damping))


def is_close_gate_crossing_candidate(
    target: GateTarget,
    *,
    initial_gate_area: int,
    control_y: float,
) -> bool:
    """Whether target loss may be the aperture expanding beyond the camera.

    This does not infer a pass. It only permits a bounded wait for the next
    authoritative race-status packet after a centered, monotonically expanded
    gate has clipped both vertical image edges.
    """

    if initial_gate_area <= 0 or not math.isfinite(control_y):
        return False
    _x, y, width, height = target.bbox
    return bool(
        target.bbox_area >= CROSSING_MIN_AREA_RATIO * initial_gate_area
        and width >= CROSSING_MIN_WIDTH_PX
        and y <= 2
        and y + height >= 358
        and abs(target.center_x - 320.0) <= 0.15 * width
        and abs(control_y - 180.0) <= 75.0
    )


def crossing_status_decision(
    *,
    baseline_race_boot_ms: int,
    current_race_boot_ms: int,
    active_gate_index: int,
    elapsed_s: float,
    timeout_s: float = CROSSING_STATUS_TIMEOUT_S,
) -> str:
    """Classify the bounded authoritative-status wait after visual commit."""

    if (
        not math.isfinite(elapsed_s)
        or elapsed_s < 0.0
        or not math.isfinite(timeout_s)
        or timeout_s <= 0.0
    ):
        raise ValueError("crossing status timing must be finite with timeout_s > 0")
    if active_gate_index not in (0, 1):
        return "invalid_gate_index"
    if current_race_boot_ms < baseline_race_boot_ms:
        return "race_clock_regressed"
    if current_race_boot_ms == baseline_race_boot_ms:
        return "status_timeout" if elapsed_s >= timeout_s else "waiting"
    if active_gate_index == 1:
        return "passed"
    if active_gate_index == 0:
        return "not_credited"
    raise AssertionError("unreachable gate crossing decision")


def full_lap_crossing_status_decision(
    *,
    baseline_race_boot_ms: int,
    current_race_boot_ms: int,
    expected_gate_index: int,
    active_gate_index: int,
    race_finished: bool,
    elapsed_s: float,
    timeout_s: float = CROSSING_STATUS_TIMEOUT_S,
) -> str:
    """Classify one mapless course crossing from authoritative race status."""

    if (
        not math.isfinite(elapsed_s)
        or elapsed_s < 0.0
        or not math.isfinite(timeout_s)
        or timeout_s <= 0.0
    ):
        raise ValueError("crossing status timing must be finite with timeout_s > 0")
    if expected_gate_index < 0 or active_gate_index < 0:
        return "invalid_gate_index"
    if current_race_boot_ms < baseline_race_boot_ms:
        return "race_clock_regressed"
    if current_race_boot_ms == baseline_race_boot_ms:
        return "status_timeout" if elapsed_s >= timeout_s else "waiting"
    if race_finished:
        return "finished"
    if active_gate_index == expected_gate_index + 1:
        return "passed"
    if active_gate_index == expected_gate_index:
        return "not_credited"
    return "invalid_gate_index"


def gate0_phase_alignment_delay_s(
    *,
    now_monotonic_ns: int,
    last_race_received_monotonic_ns: int,
    expected_target_loss_s: float = COURSE_GATE0_EXPECTED_TARGET_LOSS_S,
    packet_period_s: float = COURSE_RACE_PACKET_PERIOD_S,
    target_lead_s: float = COURSE_RACE_PACKET_TARGET_LEAD_S,
) -> float:
    """Delay launch so expected Gate 0 loss precedes a race packet."""

    if (
        type(now_monotonic_ns) is not int
        or type(last_race_received_monotonic_ns) is not int
        or now_monotonic_ns < 0
        or last_race_received_monotonic_ns < 0
        or type(expected_target_loss_s) not in {int, float}
        or type(packet_period_s) not in {int, float}
        or type(target_lead_s) not in {int, float}
        or not all(
            math.isfinite(float(value))
            for value in (expected_target_loss_s, packet_period_s, target_lead_s)
        )
        or float(expected_target_loss_s) <= 0.0
        or float(packet_period_s) <= 0.0
        or not 0.0 < float(target_lead_s) < float(packet_period_s)
    ):
        raise ValueError("gate-0 race-phase alignment inputs are invalid")
    period_ns = round(float(packet_period_s) * 1_000_000_000)
    target_loss_ns = round(float(expected_target_loss_s) * 1_000_000_000)
    lead_ns = round(float(target_lead_s) * 1_000_000_000)
    desired_start_phase_ns = (
        int(last_race_received_monotonic_ns) - lead_ns - target_loss_ns
    )
    delay_ns = (desired_start_phase_ns - int(now_monotonic_ns)) % period_ns
    return delay_ns / 1_000_000_000.0


def is_course_gate_crossing_candidate(
    target: GateTarget,
    *,
    acquisition_gate_area: int,
    control_y: float,
) -> bool:
    """Attainable close-crossing predicate for gates acquired at larger scale."""

    if (
        target.composite
        or acquisition_gate_area <= 0
        or not math.isfinite(control_y)
    ):
        return False
    _x, y, width, height = target.bbox
    minimum_area = min(
        CROSSING_MIN_AREA_RATIO * float(acquisition_gate_area),
        float(COURSE_CROSSING_AREA_CAP_PX),
    )
    return bool(
        target.bbox_area >= minimum_area
        and width >= CROSSING_MIN_WIDTH_PX
        and y <= 2
        and y + height >= 358
        and abs(target.center_x - 320.0) <= 0.15 * width
        and abs(control_y - 180.0) <= 75.0
    )


def full_lap_initial_gate_reference_is_valid(initial_gate_area: int) -> bool:
    """Whether Gate 0 has the proved build-3385 spawn-scale reference."""

    return bool(
        type(initial_gate_area) is int
        and FULL_LAP_INITIAL_GATE_MIN_AREA_PX
        <= initial_gate_area
        <= FULL_LAP_INITIAL_GATE_MAX_AREA_PX
    )


def select_untracked_contact_risk(
    detections: Iterable[GateDetection],
    *,
    accepted_target: Optional[GateTarget],
    image_width: int = 640,
    image_height: int = 360,
) -> Optional[GateDetection]:
    """Select large plausible gate geometry rejected by the live tracker.

    A live collision trace showed the accepted right-edge fragment going stale
    while a discontinuous, rapidly expanding view of the same gate remained in
    the raw detections.  Continuing flight from that stale fragment caused an
    impact.  This predicate never promotes the discontinuous geometry into
    control or crossing evidence; it only supports a fail-closed course abort.
    """

    if image_width != 640 or image_height != 360:
        raise ValueError("untracked-contact guard requires exact 640x360 vision")
    if accepted_target is not None:
        return None
    candidates: List[GateDetection] = []
    for detection in detections:
        x, y, width, height = (int(value) for value in detection.bbox)
        if (
            x < 0
            or y < 0
            or x + width > image_width
            or y + height > image_height
            or width * height < COURSE_UNTRACKED_CONTACT_MIN_AREA_PX
            or width < COURSE_UNTRACKED_CONTACT_MIN_WIDTH_PX
            or height < COURSE_UNTRACKED_CONTACT_MIN_HEIGHT_PX
            or select_primary_gate((detection,)) is not detection
        ):
            continue
        candidates.append(detection)
    return (
        max(candidates, key=lambda item: item.bbox[2] * item.bbox[3])
        if candidates
        else None
    )


def limit_command_rates(
    command: AttitudeRateCommand,
    max_rate_rad_s: float,
) -> AttitudeRateCommand:
    """Tighten a validated command without changing yaw or thrust."""

    if (
        not math.isfinite(max_rate_rad_s)
        or not 0.0 < max_rate_rad_s <= MAX_COMMAND_RATE_RAD_S
    ):
        raise ValueError("command-rate limit is outside the VQ2 envelope")
    limited = AttitudeRateCommand(
        roll_rate=max(-max_rate_rad_s, min(max_rate_rad_s, command.roll_rate)),
        pitch_rate=max(-max_rate_rad_s, min(max_rate_rad_s, command.pitch_rate)),
        yaw_rate=0.0,
        thrust=command.thrust,
    )
    validate_command(limited)
    return limited


def visual_alignment_yaw_rate(
    *,
    requested_rate_rad_s: float,
    measured_yaw_rad: float,
    reference_yaw_rad: float,
    measured_yaw_rate_rad_s: float,
    horizontal_error_norm: float,
    horizontal_corridor_norm: float,
) -> Tuple[float, float]:
    """Apply the immutable restricted-segment yaw envelope prospectively.

    The successful sign-ID calibration proves the command sign and the
    ``0.08 rad/s`` rate magnitude, but its ``0.05 rad`` experiment excursion
    was not a course-turn limit.  This stage owns a separately reviewed
    ``0.16 rad`` soft stop and ``0.18 rad`` hard stop.  Inward recovery is
    always retained; an outward command that has exhausted the soft envelope
    aborts while the target remains outside the horizontal corridor.
    """

    values = (
        requested_rate_rad_s,
        measured_yaw_rad,
        reference_yaw_rad,
        measured_yaw_rate_rad_s,
        horizontal_error_norm,
        horizontal_corridor_norm,
    )
    if not all(
        type(value) in {int, float} and math.isfinite(float(value))
        for value in values
    ):
        raise SafetyAbort("visual alignment yaw inputs are non-finite")
    if (
        float(horizontal_corridor_norm) <= 0.0
        or float(horizontal_corridor_norm) > 1.0
    ):
        raise SafetyAbort("visual alignment horizontal corridor is invalid")
    if abs(float(requested_rate_rad_s)) > (
        VISUAL_ALIGN_MAX_YAW_RATE_RAD_S + 1e-12
    ):
        raise SafetyAbort("visual alignment requested yaw rate exceeded its bound")
    if abs(float(measured_yaw_rate_rad_s)) > (
        VISUAL_ALIGN_MAX_MEASURED_YAW_RATE_RAD_S
    ):
        raise SafetyAbort(
            "visual alignment measured yaw rate exceeded its fixed bound"
        )

    excursion = math.atan2(
        math.sin(float(measured_yaw_rad) - float(reference_yaw_rad)),
        math.cos(float(measured_yaw_rad) - float(reference_yaw_rad)),
    )
    if abs(excursion) > VISUAL_ALIGN_MAX_YAW_EXCURSION_RAD:
        raise SafetyAbort(
            "visual alignment yaw excursion exceeded its fixed hard bound"
        )
    measured_rate = float(measured_yaw_rate_rad_s)
    projected_measured_excursion = (
        excursion
        + measured_rate * VISUAL_ALIGN_YAW_HOLD_HORIZON_S
    )
    if (
        excursion * measured_rate > 0.0
        and abs(projected_measured_excursion)
        > VISUAL_ALIGN_MAX_YAW_EXCURSION_RAD
    ):
        raise SafetyAbort(
            "visual alignment outward yaw momentum projects beyond its "
            "fixed hard bound"
        )

    requested = float(requested_rate_rad_s)
    outward = excursion * requested > 0.0
    if (
        outward
        and abs(excursion) >= VISUAL_ALIGN_YAW_SOFT_STOP_RAD
    ):
        if abs(float(horizontal_error_norm)) > float(
            horizontal_corridor_norm
        ):
            raise SafetyAbort(
                "visual alignment outward yaw authority exhausted outside "
                "the horizontal corridor"
            )
        return 0.0, excursion

    if outward:
        direction = math.copysign(1.0, excursion)
        soft_boundary = direction * VISUAL_ALIGN_YAW_SOFT_STOP_RAD
        remaining_rate = (
            soft_boundary - excursion
        ) / VISUAL_ALIGN_YAW_HOLD_HORIZON_S
        requested = direction * min(abs(requested), abs(remaining_rate))

    return requested, excursion


def course_recenter_rate_command(
    command: AttitudeRateCommand,
) -> AttitudeRateCommand:
    """Tighten an offline recenter proposal without amplifying its rates."""

    return limit_command_rates(command, COURSE_RECENTER_MAX_RATE_RAD_S)


def course_gate_roll_target(normalized_x: float, *, recenter: bool) -> float:
    """Choose phase-specific mapless gate roll within the watchdog margin."""

    if (
        type(normalized_x) not in {int, float}
        or not math.isfinite(float(normalized_x))
        or type(recenter) is not bool
    ):
        raise ValueError("course roll target inputs must be finite and typed")
    limit = (
        COURSE_RECENTER_ROLL_LIMIT_RAD
        if recenter
        else COURSE_APPROACH_ROLL_LIMIT_RAD
    )
    gain = COURSE_RECENTER_ROLL_GAIN if recenter else COURSE_ROLL_GAIN
    return max(
        -limit,
        min(limit, gain * float(normalized_x)),
    )


def course_gate_recenter_pitch_target(
    entry_pitch_rad: float,
    control_y: float,
) -> float:
    """Return the exact-zero pitch objective required by bounded recentering."""

    if (
        type(entry_pitch_rad) not in {int, float}
        or not math.isfinite(float(entry_pitch_rad))
        or type(control_y) not in {int, float}
        or not math.isfinite(float(control_y))
    ):
        raise ValueError("course recenter pitch inputs must be finite and numeric")
    return 0.0


def course_gate_recenter_required(
    elapsed_s: float,
    normalized_x: float,
    control_y: float,
) -> bool:
    """Retain recenter mode only inside its non-renewable hard window."""

    if (
        type(elapsed_s) not in {int, float}
        or not math.isfinite(float(elapsed_s))
        or float(elapsed_s) < 0.0
        or type(normalized_x) not in {int, float}
        or not math.isfinite(float(normalized_x))
        or type(control_y) not in {int, float}
        or not math.isfinite(float(control_y))
    ):
        raise ValueError("course recenter inputs must be finite and typed")
    return bool(float(elapsed_s) < COURSE_RECENTER_DURATION_S)


def post_gate_observation_deadline(
    *,
    pass_confirmed_s: float,
    flight_started_s: float,
    crossing_started_s: Optional[float],
    requested_duration_s: float = POST_GATE_OBSERVATION_TIMEOUT_S,
) -> float:
    """Fixed observation deadline nested inside every existing flight bound."""

    values = [
        float(pass_confirmed_s),
        float(flight_started_s),
        float(requested_duration_s),
    ]
    if crossing_started_s is not None:
        values.append(float(crossing_started_s))
    if (
        not all(math.isfinite(value) for value in values)
        or not 0.10
        <= float(requested_duration_s)
        <= POST_GATE_OBSERVATION_TIMEOUT_S
    ):
        raise ValueError("post-gate deadline inputs must be finite")
    candidates = [
        float(pass_confirmed_s) + float(requested_duration_s),
        float(flight_started_s) + GATE0_FLIGHT_TIMEOUT_S,
    ]
    if crossing_started_s is not None:
        candidates.append(float(crossing_started_s) + CROSSING_STATUS_TIMEOUT_S)
    return min(candidates)


def is_crossing_residue(
    target: GateTarget | GateDetection,
    *,
    image_width: int = 640,
    image_height: int = 360,
) -> bool:
    """Reject a large clipped remnant of gate 0 during gate-1 reacquisition.

    The predicate is deliberately scoped to the post-pass tracker.  Large
    clipped contours are useful evidence during the gate-0 approach, but they
    must not seed a fresh tracker after race status authoritatively advances.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x, y, width, height = (int(value) for value in target.bbox)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        return False

    width_fraction = width / image_width
    height_fraction = height / image_height
    area_fraction = width_fraction * height_fraction
    left = x < 3
    top = y < 3
    right = x + width > image_width - 3
    bottom = y + height > image_height - 3
    opposing_edges = (top and bottom) or (left and right)
    edge_count = sum((left, top, right, bottom))
    return bool(
        (opposing_edges and area_fraction >= 0.25)
        or (
            edge_count >= 1
            and area_fraction >= 0.60
            and width_fraction >= 0.70
            and height_fraction >= 0.70
        )
        or (width_fraction >= 0.90 and height_fraction >= 0.90)
    )


def _finite_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def gate_detection_summary(
    detection: GateDetection,
    *,
    detector_index: int,
    image_width: int = 640,
    image_height: int = 360,
    reject_crossing_residue: bool = False,
) -> Dict[str, Any]:
    """Return JSON-safe pixel diagnostics for one raw detector result."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x, y, width, height = (int(value) for value in detection.bbox)
    valid_bbox = bool(
        x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= image_width
        and y + height <= image_height
    )
    confidence = _finite_float(detection.confidence)
    raw_corners = getattr(detection, "corners", None)
    corners_px: Optional[List[List[float]]] = None
    try:
        candidate_corners = [
            [_finite_float(point[0]), _finite_float(point[1])]
            for point in raw_corners
        ]
        if (
            len(candidate_corners) == 4
            and all(value is not None for point in candidate_corners for value in point)
        ):
            corners_px = [
                [float(point[0]), float(point[1])] for point in candidate_corners
            ]
    except (TypeError, IndexError):
        corners_px = None
    axis_aspect = (
        max(width, height) / min(width, height)
        if width > 0 and height > 0
        else None
    )
    rejections: List[str] = []
    if not valid_bbox:
        rejections.append("invalid_bbox")
    if width < 20:
        rejections.append("min_width")
    if height < 20:
        rejections.append("min_height")
    if axis_aspect is None or axis_aspect > 1.85:
        rejections.append("axis_aspect_gt_1.85")
    if confidence is None:
        rejections.append("nonfinite_confidence")
    elif confidence < 0.10:
        rejections.append("confidence_below_0.10")
    residue = is_crossing_residue(
        detection,
        image_width=image_width,
        image_height=image_height,
    )
    base_selector_eligible = not rejections
    if residue and reject_crossing_residue:
        rejections.append("crossing_residue")
    return {
        "detector_index": int(detector_index),
        "center_px": [int(detection.center_x), int(detection.center_y)],
        "corners_px": corners_px,
        "bbox_xywh_px": [x, y, width, height],
        "reported_area_px": int(detection.area),
        "bbox_area_px": width * height,
        "bbox_area_fraction": (
            (width * height) / (image_width * image_height) if valid_bbox else None
        ),
        "axis_aspect_ratio": axis_aspect,
        "apparent_size_px": [
            _finite_float(detection.apparent_width_px),
            _finite_float(detection.apparent_height_px),
        ],
        "min_rect_aspect_ratio": _finite_float(detection.aspect_ratio),
        "rotation_deg": _finite_float(detection.rotation_deg),
        "rectangularity": _finite_float(detection.rectangularity),
        "confidence": confidence,
        "method": str(detection.detection_method),
        "edge_touch": {
            "left": valid_bbox and x < 3,
            "top": valid_bbox and y < 3,
            "right": valid_bbox and x + width > image_width - 3,
            "bottom": valid_bbox and y + height > image_height - 3,
        },
        "base_selector_eligible": base_selector_eligible,
        "post_pass_selector_eligible": base_selector_eligible and not residue,
        "selector_eligible": not rejections,
        "selector_rejections": rejections,
        "crossing_residue": residue,
    }


def attitude_rate_command(
    estimate: AttitudeEstimate,
    *,
    target_roll_rad: float,
    target_pitch_rad: float,
    thrust: float,
) -> AttitudeRateCommand:
    """Conservative roll/pitch attitude loop with yaw deliberately disabled."""

    if _attitude_error_body_rates is None:
        _load_live_transport_dependencies()

    desired = Quaternion.from_euler(
        float(target_roll_rad),
        float(target_pitch_rad),
        estimate.yaw,
    )
    roll_rate, pitch_rate, _yaw_rate = _attitude_error_body_rates(
        estimate.orientation,
        desired,
        omega=estimate.body_rates,
        kp=(1.0, 0.5, 0.0),
        kd=(0.4, 0.2, 0.0),
        max_rate=(MAX_COMMAND_RATE_RAD_S,) * 3,
    )
    command = AttitudeRateCommand(
        roll_rate=float(roll_rate),
        pitch_rate=float(pitch_rate),
        yaw_rate=0.0,
        thrust=float(thrust),
    )
    validate_command(command)
    return command


def validate_command(command: AttitudeRateCommand) -> None:
    values = (
        command.roll_rate,
        command.pitch_rate,
        command.yaw_rate,
        command.thrust,
    )
    if not all(math.isfinite(value) for value in values):
        raise SafetyAbort("non-finite command")
    if max(abs(command.roll_rate), abs(command.pitch_rate), abs(command.yaw_rate)) > (
        MAX_COMMAND_RATE_RAD_S + 1e-9
    ):
        raise SafetyAbort("commanded body rate exceeded conservative clamp")
    if not 0.0 <= command.thrust <= 0.35:
        raise SafetyAbort("commanded thrust exceeded conservative VQ2 envelope")


def is_benign_pad_contact(
    collision: Dict[str, Any],
    *,
    max_impulse: float = 0.01,
) -> bool:
    """Exact low-energy spawn-pad contact class observed during motor preload."""

    try:
        return (
            math.isfinite(float(max_impulse))
            and float(max_impulse) > 0.0
            and collision.get("id") == 1002
            and int(collision.get("threat_level", 99)) <= 1
            and abs(float(collision.get("impulse", math.inf)))
            <= float(max_impulse)
        )
    except (TypeError, ValueError, OverflowError):
        return False


class JsonlRecorder:
    def __init__(
        self,
        path: Optional[str],
        *,
        replay: Optional[AsyncReplayRecorder] = None,
        capture_fifo_enabled: bool = False,
        create_new: bool = False,
    ) -> None:
        if type(capture_fifo_enabled) is not bool:
            raise TypeError("capture_fifo_enabled must be an exact bool")
        if type(create_new) is not bool:
            raise TypeError("create_new must be an exact bool")
        if capture_fifo_enabled and replay is None:
            raise ValueError("capture FIFO requires a replay recorder")
        self.path = Path(path).resolve() if path else None
        self.replay = replay
        self.capture_fifo_enabled = capture_fifo_enabled
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if str(self.path).endswith(".gz"):
                self._handle = gzip.open(
                    self.path,
                    "xt" if create_new else "wt",
                    encoding="utf-8",
                )
            else:
                self._handle = self.path.open(
                    "x" if create_new else "w",
                    encoding="utf-8",
                )

    @property
    def capture_enabled(self) -> bool:
        return self.replay is not None

    def emit(self, event: str, **fields: Any) -> bool:
        if self._handle is not None:
            row = {"event": event, "wall_time_ns": time.time_ns(), **fields}
            self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        if self.replay is not None:
            self.replay.record_event(event, **fields)
        return True

    def emit_powered(self, event: str, *, observation: Mapping[str, Any]) -> bool:
        """Enqueue one exact nested powered observation and fail on backpressure."""

        if type(event) is not str or not event:
            raise TypeError("powered event name must be a nonempty exact string")
        if type(observation) is not dict:
            raise TypeError("powered observation must be an exact object")
        if self._handle is not None:
            row = {
                "event": event,
                "wall_time_ns": time.time_ns(),
                "observation": observation,
            }
            self._handle.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        if self.replay is None:
            return False
        try:
            accepted = self.replay.record_event(event, observation=observation)
        except BaseException as exc:
            try:
                self.replay.fail(
                    f"powered replay enqueue raised for {event}: "
                    f"{type(exc).__name__}"
                )
            finally:
                raise
        if accepted is not True:
            self.replay.fail(f"powered replay enqueue returned false for {event}")
            return False
        return True

    def record_imu(
        self,
        imu: Any,
        estimator: Optional[Dict[str, Any]],
        now_s: float,
        *,
        received_sample: Optional[Any] = None,
    ) -> bool:
        if self.replay is not None:
            return self.replay.record_imu(
                imu,
                estimator=estimator,
                received_monotonic_s=now_s,
                received_sample=received_sample,
            )
        return True

    def record_mavlink_ingress(self, ingress: Any) -> bool:
        if self.replay is not None:
            return self.replay.record_mavlink_ingress(ingress)
        return True

    def record_camera_timing(
        self, observation: CameraFrameTimingObservationV1
    ) -> None:
        if self.replay is not None:
            self.replay.record_event(
                "camera_frame_timing_observation",
                observation=observation.to_primitive(),
            )

    def record_race(self, race: Any, now_s: float) -> bool:
        if self.replay is not None:
            return self.replay.record_race(race, received_monotonic_s=now_s)
        return True

    def record_command(
        self,
        kind: str,
        command: AttitudeRateCommand,
        *,
        monotonic_s: float,
        frame_token: Optional[Tuple[int, int, int]],
    ) -> bool:
        if self.replay is not None:
            return self.replay.record_command(
                kind,
                command,
                monotonic_s=monotonic_s,
                frame_token=frame_token,
            )
        return True

    def capture_frame(self, image: Any, **fields: Any) -> None:
        if self.replay is not None:
            self.replay.capture_frame(image, **fields)

    def save_png(self, label: str, image: Any) -> Optional[str]:
        """Persist one deferred diagnostic image beside the JSONL capture."""

        if self.path is None:
            return None
        import cv2

        base = self.path
        if base.suffix == ".gz":
            base = base.with_suffix("")
        if base.suffix == ".jsonl":
            base = base.with_suffix("")
        safe_label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(label)
        ).strip("_") or "frame"
        output = base.parent / f"{base.name}_{safe_label}.png"
        if not cv2.imwrite(str(output), image):
            raise OSError(f"OpenCV could not write diagnostic image {output}")
        return str(output.resolve())

    def close(
        self,
        *,
        outcome: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Any:
        if timeout_s is not None and (
            type(timeout_s) not in {int, float}
            or not math.isfinite(float(timeout_s))
            or timeout_s <= 0
        ):
            raise ValueError("recorder close timeout must be finite and positive")
        handle_error: Optional[BaseException] = None
        handle_traceback = None
        if self._handle is not None:
            handle = self._handle
            self._handle = None
            try:
                handle.close()
            except BaseException as exc:
                handle_error = exc
                handle_traceback = exc.__traceback__
        replay_result = None
        replay_error: Optional[BaseException] = None
        if self.replay is not None:
            expected = None
            if outcome is not None:
                vision_stats = outcome.get("vision_capture_stats")
                if isinstance(vision_stats, dict):
                    expected = vision_stats.get("frames_decoded")
            if handle_error is not None:
                try:
                    self.replay.fail(
                        "legacy JSONL recorder close failed before replay seal: "
                        f"{type(handle_error).__name__}: {handle_error}"
                    )
                except BaseException as exc:
                    replay_error = exc
            try:
                kwargs = {
                    "outcome": outcome,
                    "expected_decoded_frames": expected,
                }
                if timeout_s is not None:
                    kwargs["timeout_s"] = float(timeout_s)
                replay_result = self.replay.close(**kwargs)
            except BaseException as exc:
                replay_error = replay_error or exc
        if handle_error is not None:
            if replay_error is not None:
                handle_error.add_note(
                    "Replay cleanup also failed: "
                    f"{type(replay_error).__name__}: {replay_error}"
                )
            raise handle_error.with_traceback(handle_traceback)
        if replay_error is not None:
            raise replay_error
        return replay_result


class VQ2Runner:
    def __init__(
        self,
        adapter: AIGPMavlinkAdapter,
        vision: VQ2VisionThread,
        *,
        recorder: Optional[JsonlRecorder] = None,
        controller_config: Optional[
            Mapping[str, Any] | VQ2ControllerConfig
        ] = None,
        controller_evidence: Optional[Mapping[str, Any]] = None,
        visual_config: Optional[
            Mapping[str, Any] | VisualNavigationConfig
        ] = None,
        visual_controller_evidence: Optional[Mapping[str, Any]] = None,
        visual_session_id: str = "direct-live-session",
    ) -> None:
        if adapter.enable_vision:
            raise ValueError("VQ2Runner requires adapter vision disabled")
        if adapter.telemetry_mode != "imu" or adapter.fetch_track_on_connect:
            raise ValueError("VQ2Runner requires track-free IMU telemetry mode")
        self.adapter = adapter
        self.vision = vision
        self.detector = VQ2GateDetector()
        self.tracker = GateTargetTracker()
        self.recorder = recorder or JsonlRecorder(None)
        try:
            self.controller_config = (
                default_controller_config()
                if controller_config is None
                else (
                    validate_controller_config(
                        controller_config.to_effective_mapping()
                    )
                    if isinstance(controller_config, VQ2ControllerConfig)
                    else validate_controller_config(controller_config)
                )
            )
        except ControllerConfigError as exc:
            raise ValueError(f"controller configuration refused: {exc}") from exc
        expected_controller = controller_config_evidence(
            self.controller_config,
            candidate_commit=(
                None
                if controller_evidence is None
                else controller_evidence.get("git_commit")
            ),
        )
        if controller_evidence is not None and dict(controller_evidence) != (
            expected_controller
        ):
            raise ValueError("controller evidence does not match effective config")
        self.controller_evidence = expected_controller
        try:
            self.visual_config = (
                default_visual_config()
                if visual_config is None
                else (
                    validate_visual_config(
                        visual_config.to_effective_mapping()
                    )
                    if isinstance(visual_config, VisualNavigationConfig)
                    else validate_visual_config(visual_config)
                )
            )
        except VisualConfigError as exc:
            raise ValueError(
                f"visual navigation configuration refused: {exc}"
            ) from exc
        expected_visual_controller = controller_config_evidence(
            self.visual_config,
            candidate_commit=(
                None
                if visual_controller_evidence is None
                else visual_controller_evidence.get("git_commit")
            ),
        )
        if (
            visual_controller_evidence is not None
            and dict(visual_controller_evidence) != expected_visual_controller
        ):
            raise ValueError(
                "visual controller evidence does not match effective config"
            )
        self.visual_controller_evidence = expected_visual_controller
        if (
            type(visual_session_id) is not str
            or not visual_session_id
            or len(visual_session_id) > 128
        ):
            raise ValueError("visual_session_id must be a bounded string")
        self._visual_session_id = visual_session_id

        config = ImuAttitudeConfig(
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
        if config.gravity_correction_kp != 0.0 or config.gyro_bias_ki != 0.0:
            raise AssertionError("powered VQ2 estimator must be pure gyro after bootstrap")
        self.estimator = ImuAttitudeEstimator(config)

        self.estimate: Optional[AttitudeEstimate] = None
        self._last_imu_us: Optional[int] = None
        self._last_imu_advance_s = 0.0
        self._last_race_boot_ms: Optional[int] = None
        self._last_race_advance_s = 0.0
        self._last_frame_identity: Optional[Tuple[int, int]] = None
        self._last_frame_sim_ns: Optional[int] = None
        self._imu_regressed = False
        self._race_regressed = False
        self._imu_forward_jump = False
        self._race_forward_jump = False
        self._epoch_imu_anchor_us: Optional[int] = None
        self._epoch_race_anchor_ms: Optional[int] = None
        self._epoch_anchor_monotonic_s = 0.0
        self._countdown_observed = False
        self._detection_error: Optional[str] = None
        self._estimator_unhealthy_latched = False
        self._estimator_failure_reason: Optional[str] = None
        self._benign_pad_contact_count = 0
        self._benign_pad_contact_impulse = 0.0
        self._high_rate_samples = 0
        self._abort_latched = False
        self._latest_raw_detections: List[GateDetection] = []
        self._latest_accepted_target: Optional[GateTarget] = None
        self._latest_detection_frame_id: Optional[int] = None
        self._latest_detection_frame_sim_ns: Optional[int] = None
        self._latest_detection_generation: Optional[int] = None
        self._latest_detection_received_s: Optional[float] = None
        self._latest_detection_image: Any = None
        self._post_gate_last_frame: Optional[Tuple[Tuple[int, int, int], Any]] = None
        self._vision_diagnostic_logging = False
        self._post_gate_reacquisition = False
        self._course_edge_continuation_gate_index: Optional[int] = None
        self._last_flight_command: Optional[AttitudeRateCommand] = None
        self._last_flight_command_started_ns: Optional[int] = None
        self._last_flight_command_sent_s: Optional[float] = None
        self._epoch_vision_started_s: Optional[float] = None
        self._epoch_vision_initial_frames: Optional[int] = None
        self._gate0_transition_proof: Optional[GateTransitionProof] = None
        self._gate0_early_turn_summary: Optional[Dict[str, Any]] = None
        self._gate1_recenter_summary: Optional[Dict[str, Any]] = None
        self._gate1_yaw_reference_rad: Optional[float] = None
        self._gate1_max_abs_yaw_excursion_rad = 0.0
        self._gate1_max_abs_measured_yaw_rate_rad_s = 0.0
        self._deferred_pngs: List[Tuple[str, Any]] = []
        self._visual_tracking_enabled = False
        self._visual_diagnostic_logging = False
        self._visual_reset_epoch = 0
        self.visual_tracker = MultiTargetVisualTracker(
            VISUAL_TRACKER_CONFIG
        )
        self.visual_gate_graph = RollingVisualGateGraph()
        self._visual_latest_tracker_update: Any = None
        self._visual_latest_graph_snapshot: Optional[GateGraphSnapshot] = None
        self._visual_transition: Optional[ConfirmedGateTransition] = None
        self._visual_shadow_summary: Optional[Dict[str, Any]] = None
        self._visual_alignment_summary: Optional[Dict[str, Any]] = None
        self._visual_gate0_blend_summary: Optional[Dict[str, Any]] = None
        self._visual_active_stage: Optional[str] = None

    def _gate1_yaw_envelope_state(self, *, phase: str) -> Tuple[float, bool]:
        """Enforce the code-owned calibrated yaw excursion envelope."""

        reference = self._gate1_yaw_reference_rad
        if reference is None:
            return 0.0, False
        if self.estimate is None:
            raise SafetyAbort(
                f"Gate-1 yaw attitude estimate unavailable during {phase}"
            )
        _roll, _pitch, yaw = self.estimate.orientation.to_euler()
        excursion = math.atan2(
            math.sin(float(yaw) - reference),
            math.cos(float(yaw) - reference),
        )
        self._gate1_max_abs_yaw_excursion_rad = max(
            self._gate1_max_abs_yaw_excursion_rad,
            abs(excursion),
        )
        self._gate1_max_abs_measured_yaw_rate_rad_s = max(
            self._gate1_max_abs_measured_yaw_rate_rad_s,
            abs(float(self.estimate.body_rates[2])),
        )
        if (
            abs(float(self.estimate.body_rates[2]))
            > GATE1_RECENTER_MAX_MEASURED_BODY_RATE_RAD_S
        ):
            raise SafetyAbort(
                "Gate-1 calibrated measured yaw rate exceeded its fixed bound "
                f"during {phase}"
            )
        if abs(excursion) > GATE1_CONTROLLER_MAX_YAW_EXCURSION_RAD:
            raise SafetyAbort(
                "Gate-1 calibrated yaw excursion exceeded its fixed bound "
                f"during {phase}"
            )
        return (
            excursion,
            abs(excursion) >= GATE1_CONTROLLER_YAW_SOFT_STOP_RAD,
        )

    def _replay_estimator_fields(self) -> Optional[Dict[str, Any]]:
        estimate = self.estimate
        if estimate is None:
            return None
        return {
            "timestamp_us": int(estimate.timestamp_us),
            "rpy_rad": list(estimate.orientation.to_euler()),
            "orientation_wxyz": [
                estimate.orientation.w,
                estimate.orientation.x,
                estimate.orientation.y,
                estimate.orientation.z,
            ],
            "body_rates": list(estimate.body_rates),
            "gyro_bias": list(estimate.gyro_bias),
            "healthy": bool(estimate.healthy),
            "reason": estimate.reason,
            "propagated": bool(estimate.propagated),
        }

    def _latest_frame_token(self) -> Optional[Tuple[int, int, int]]:
        if (
            self._latest_detection_generation is None
            or self._latest_detection_frame_id is None
            or self._latest_detection_frame_sim_ns is None
        ):
            return None
        return (
            self._latest_detection_generation,
            self._latest_detection_frame_id,
            self._latest_detection_frame_sim_ns,
        )

    @staticmethod
    def _visual_track_summary(track: VisualTrack) -> Dict[str, Any]:
        latest_token = track.latest_token
        first_token = track.first_token
        return {
            "track_id": track.track_id,
            "first_frame_token": (
                list(first_token.live_identity_tuple)
                if first_token.live_identity_tuple is not None
                else list(first_token.exact_tuple)
            ),
            "latest_frame_token": (
                list(latest_token.live_identity_tuple)
                if latest_token.live_identity_tuple is not None
                else list(latest_token.exact_tuple)
            ),
            "center_norm_image_down": list(track.center_norm),
            "bbox_norm_ltrb": list(track.bbox_norm),
            "apparent_scale": track.apparent_scale,
            "center_velocity_norm_s_image_down": list(
                track.center_velocity_norm_s
            ),
            "log_scale_rate_s": track.log_scale_rate_s,
            "confidence": track.confidence,
            "association_confidence": track.association_confidence,
            "consecutive_frame_count": track.consecutive_frame_count,
            "total_observation_count": track.total_observation_count,
            "missed_frame_count": track.missed_frame_count,
            "clipping_edges": int(track.clipping),
            "center_censored": track.center_censored,
            "role": track.role.value,
            "authoritative_gate_index": track.authoritative_gate_index,
            "ambiguous": track.ambiguous,
            "visible": track.visible,
        }

    @staticmethod
    def _visual_graph_summary(
        snapshot: Optional[GateGraphSnapshot],
    ) -> Optional[Dict[str, Any]]:
        if snapshot is None:
            return None
        return {
            "current_track_id": snapshot.current_track_id,
            "current_gate_index": snapshot.current_gate_index,
            "next_candidates": [
                {
                    "track_id": candidate.track_id,
                    "score": candidate.score,
                    "stable_frame_count": candidate.stable_frame_count,
                    "first_frame_token": (
                        list(candidate.first_token.live_identity_tuple)
                        if candidate.first_token.live_identity_tuple is not None
                        else list(candidate.first_token.exact_tuple)
                    ),
                    "latest_frame_token": (
                        list(candidate.latest_token.live_identity_tuple)
                        if candidate.latest_token.live_identity_tuple is not None
                        else list(candidate.latest_token.exact_tuple)
                    ),
                    "bearing_norm": candidate.bearing_norm,
                    "elevation_norm": candidate.elevation_norm,
                    "bearing_rate_norm_s": candidate.bearing_rate_norm_s,
                    "elevation_rate_norm_s": candidate.elevation_rate_norm_s,
                    "apparent_scale": candidate.apparent_scale,
                    "log_scale_rate_s": candidate.log_scale_rate_s,
                    "confidence": candidate.confidence,
                    "association_confidence": (
                        candidate.association_confidence
                    ),
                    "center_censored": candidate.center_censored,
                    "promotable": candidate.promotable,
                    "relationship": (
                        None
                        if candidate.relationship is None
                        else {
                            "basis": candidate.relationship.basis.value,
                            "current_anchor_frame_token": list(
                                candidate.relationship.current_anchor_token
                                .live_identity_tuple
                                or candidate.relationship.current_anchor_token
                                .exact_tuple
                            ),
                            "next_anchor_frame_token": list(
                                candidate.relationship.next_anchor_token
                                .live_identity_tuple
                                or candidate.relationship.next_anchor_token
                                .exact_tuple
                            ),
                            "anchor_publication_delta": (
                                candidate.relationship.anchor_publication_delta
                            ),
                            "anchor_time_gap_ns": (
                                candidate.relationship.anchor_time_gap_ns
                            ),
                            "observation_count": (
                                candidate.relationship.observation_count
                            ),
                            "simultaneous_observation_count": (
                                candidate.relationship
                                .simultaneous_observation_count
                            ),
                            "sequential_observation_count": (
                                candidate.relationship
                                .sequential_observation_count
                            ),
                            "observation_confidence": (
                                candidate.relationship.observation_confidence
                            ),
                            "fresh": candidate.relationship.fresh,
                            "contended": candidate.relationship.contended,
                            "relative_geometry_usable": (
                                candidate.relationship
                                .relative_geometry_usable
                            ),
                        }
                    ),
                }
                for candidate in snapshot.next_candidates
            ],
            "next_selection_ambiguous": snapshot.next_selection_ambiguous,
            "authority_usable": snapshot.authority_usable,
            "withholding_reason": snapshot.withholding_reason,
            "confirmed_transition_count": len(
                snapshot.confirmed_transitions
            ),
            "race_finished": snapshot.race_finished,
        }

    def _clear_epoch_state(self) -> None:
        self.estimator.reset()
        self.estimate = None
        self._last_imu_us = None
        self._last_imu_advance_s = 0.0
        self._last_race_boot_ms = None
        self._last_race_advance_s = 0.0
        self._last_frame_identity = None
        self._last_frame_sim_ns = None
        self._imu_regressed = False
        self._race_regressed = False
        self._imu_forward_jump = False
        self._race_forward_jump = False
        self._epoch_imu_anchor_us = None
        self._epoch_race_anchor_ms = None
        self._epoch_anchor_monotonic_s = 0.0
        self._countdown_observed = False
        self._detection_error = None
        self._estimator_unhealthy_latched = False
        self._estimator_failure_reason = None
        self._benign_pad_contact_count = 0
        self._benign_pad_contact_impulse = 0.0
        self._high_rate_samples = 0
        self._latest_raw_detections = []
        self._latest_accepted_target = None
        self._latest_detection_frame_id = None
        self._latest_detection_frame_sim_ns = None
        self._latest_detection_generation = None
        self._latest_detection_received_s = None
        self._latest_detection_image = None
        self._vision_diagnostic_logging = False
        self._post_gate_reacquisition = False
        self._course_edge_continuation_gate_index = None
        self._last_flight_command = None
        self._last_flight_command_started_ns = None
        self._last_flight_command_sent_s = None
        self._epoch_vision_started_s = None
        self._epoch_vision_initial_frames = None
        self._gate0_transition_proof = None
        self._gate1_recenter_summary = None
        self.tracker.reset()
        if self._visual_tracking_enabled:
            self.visual_tracker = MultiTargetVisualTracker(
                VISUAL_TRACKER_CONFIG
            )
            self.visual_gate_graph = RollingVisualGateGraph()
            self._visual_latest_tracker_update = None
            self._visual_latest_graph_snapshot = None
            self._visual_transition = None

    def _consume_imu_sample(
        self,
        imu: Any,
        received_sample: Optional[Any],
        now: float,
    ) -> None:
        stamp = int(imu.timestamp_us)
        if (
            self._epoch_imu_anchor_us is not None
            and not clock_within_epoch_envelope(
                self._epoch_imu_anchor_us,
                stamp,
                now - self._epoch_anchor_monotonic_s,
                units_per_second=1_000_000.0,
                slack=500_000,
            )
        ):
            self._imu_forward_jump = True
        elif self._last_imu_us is None or stamp > self._last_imu_us:
            estimator_was_ready = self.estimator.is_ready
            estimate = self.estimator.update_imu(imu)
            self._last_imu_us = stamp
            self._last_imu_advance_s = now
            if estimate is None and estimator_was_ready:
                # Transport freshness is not estimator health. Once the
                # estimator is ready, any rejected newer sample must latch
                # an abort instead of letting an old estimate look current.
                self._estimator_unhealthy_latched = True
                self._estimator_failure_reason = (
                    self.estimator.last_rejection_reason or "sample rejected"
                )
            elif estimate is not None:
                self.estimate = estimate
                if not estimate.healthy:
                    self._estimator_unhealthy_latched = True
                    self._estimator_failure_reason = (
                        estimate.reason or "unhealthy estimate"
                    )
        elif stamp < self._last_imu_us:
            self._imu_regressed = True
        record_imu = getattr(self.recorder, "record_imu", None)
        if callable(record_imu):
            record_imu(
                imu,
                self._replay_estimator_fields(),
                now,
                received_sample=received_sample,
            )

    def _sample(self) -> None:
        now = time.monotonic()
        telemetry = self.adapter.latest_telemetry
        drain_received_ingress = getattr(
            self.adapter, "drain_received_ingress", None
        )
        ordered_ingress = []
        untimed_imu = []
        if callable(drain_received_ingress):
            for item in drain_received_ingress():
                if type(item) is ReceivedIMUSampleV1:
                    ordered_ingress.append(
                        (item.ingress.sequence, "imu", (item.imu, item))
                    )
                elif type(item) is MavlinkIngressV1:
                    ordered_ingress.append((item.sequence, "arrival", item))
                else:
                    raise TypeError("exact receiver ingress item has invalid type")
        else:
            drain_received_imu = getattr(
                self.adapter, "drain_received_imu_samples", None
            )
            if callable(drain_received_imu):
                received_imu_samples = [
                    (received.imu, received) for received in drain_received_imu()
                ]
            else:
                drain_imu = getattr(self.adapter, "drain_imu_samples", None)
                if callable(drain_imu):
                    received_imu_samples = [
                        (imu, None) for imu in drain_imu()
                    ]
                else:
                    imu = telemetry.imu if telemetry is not None else None
                    received_imu_samples = (
                        [(imu, None)] if imu is not None else []
                    )
            drain_arrivals = getattr(
                self.adapter, "drain_mavlink_arrivals", None
            )
            arrivals = drain_arrivals() if callable(drain_arrivals) else []
            for arrival in arrivals:
                ordered_ingress.append(
                    (arrival.sequence, "arrival", arrival)
                )
            for imu, received_sample in received_imu_samples:
                if received_sample is None:
                    untimed_imu.append((imu, received_sample))
                else:
                    ordered_ingress.append(
                        (
                            received_sample.ingress.sequence,
                            "imu",
                            (imu, received_sample),
                        )
                    )
        record_arrival = getattr(self.recorder, "record_mavlink_ingress", None)
        ordered_ingress.sort(key=lambda item: item[0])
        for _sequence, kind, value in ordered_ingress:
            if kind == "arrival":
                if callable(record_arrival):
                    record_arrival(value)
            else:
                imu, received_sample = value
                self._consume_imu_sample(imu, received_sample, now)
        for imu, received_sample in untimed_imu:
            self._consume_imu_sample(imu, received_sample, now)

        race = self.adapter.race_status
        if race is not None:
            boot = int(race.sim_boot_time_ms)
            if (
                self._epoch_race_anchor_ms is not None
                and not clock_within_epoch_envelope(
                    self._epoch_race_anchor_ms,
                    boot,
                    now - self._epoch_anchor_monotonic_s,
                    units_per_second=1_000.0,
                    slack=700,
                )
            ):
                self._race_forward_jump = True
            elif self._last_race_boot_ms is None or boot > self._last_race_boot_ms:
                self._last_race_boot_ms = boot
                self._last_race_advance_s = now
                if race.race_start_boot_time_ms < 0 or boot < race.race_start_boot_time_ms:
                    self._countdown_observed = True
                record_race = getattr(self.recorder, "record_race", None)
                if callable(record_race):
                    record_race(race, now)
            elif boot < self._last_race_boot_ms:
                self._race_regressed = True

        capture_enabled = bool(
            getattr(self.recorder, "capture_enabled", False)
        )
        capture_fifo_enabled = (
            getattr(self.recorder, "capture_fifo_enabled", False) is True
        )
        pop_capture_snapshot = getattr(self.vision, "pop_capture_snapshot", None)
        if capture_fifo_enabled and callable(pop_capture_snapshot):
            snapshot = pop_capture_snapshot()
        else:
            snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        frame_identity = (
            None
            if snapshot is None
            else (int(snapshot.generation), int(snapshot.frame_id))
        )
        if snapshot is not None and frame_identity != self._last_frame_identity:
            consume_monotonic_ns = (
                time.perf_counter_ns() if capture_enabled else None
            )
            # The camera source timestamp is an opaque ordering token, not
            # frame identity.  Repeated control polls consume one publication
            # once, while a receiver generation restart can reuse frame IDs.
            self._last_frame_identity = frame_identity
            self._last_frame_sim_ns = snapshot.sim_time_ns
            self._latest_detection_frame_id = int(snapshot.frame_id)
            self._latest_detection_frame_sim_ns = int(snapshot.sim_time_ns)
            self._latest_detection_generation = int(snapshot.generation)
            self._latest_detection_received_s = float(snapshot.received_monotonic_s)
            work_started_ns = (
                time.perf_counter_ns() if capture_enabled else None
            )
            detector_started_ns = (
                time.perf_counter_ns() if capture_enabled else None
            )
            detector_latency_ms: Optional[float] = None
            try:
                image = snapshot.camera_frame.image
                self._latest_detection_image = image
                image_height, image_width = image.shape[:2]
                detections = list(self.detector.detect(image))
                detector_ended_ns = (
                    time.perf_counter_ns() if capture_enabled else None
                )
                if detector_started_ns is not None:
                    detector_latency_ms = (
                        detector_ended_ns - detector_started_ns
                    ) / 1_000_000.0
                self._latest_raw_detections = detections
                if self._visual_tracking_enabled:
                    visual_frame = VisualDetectionFrame.from_vision_snapshot(
                        snapshot,
                        detections,
                    )
                    visual_update = self.visual_tracker.update(visual_frame)
                    self._visual_latest_graph_snapshot = (
                        self.visual_gate_graph.observe(self.visual_tracker)
                    )
                    # Graph observation can authoritatively update tracker
                    # roles.  Retain the refreshed exact update rather than
                    # the pre-graph immutable view that may still say NEXT.
                    refreshed_update = self.visual_tracker.latest_update
                    if (
                        refreshed_update is None
                        or refreshed_update.token != visual_update.token
                    ):
                        raise SafetyAbort(
                            "visual graph role refresh lost exact frame identity"
                        )
                    self._visual_latest_tracker_update = refreshed_update
                    if self._visual_diagnostic_logging:
                        visual_race = self.adapter.race_status
                        self.recorder.emit(
                            "visual_gate_graph_frame",
                            phase=(
                                self._visual_active_stage
                                or "visual_navigation"
                            ),
                            frame_token=list(
                                visual_update.token.live_identity_tuple
                                or visual_update.token.exact_tuple
                            ),
                            camera_source_time_ns=int(snapshot.sim_time_ns),
                            final_unique_packet_monotonic_ns=(
                                visual_frame.final_unique_packet_monotonic_ns
                            ),
                            publish_monotonic_ns=(
                                visual_frame.publish_monotonic_ns
                            ),
                            received_monotonic_s=(
                                snapshot.received_monotonic_s
                            ),
                            race_boot_ms=(
                                int(visual_race.sim_boot_time_ms)
                                if visual_race is not None
                                else None
                            ),
                            gate_index=(
                                int(visual_race.active_gate_index)
                                if visual_race is not None
                                else None
                            ),
                            tracks=[
                                self._visual_track_summary(track)
                                for track in refreshed_update.tracks
                            ],
                            associations=[
                                {
                                    "track_id": item.track_id,
                                    "detection_source_index": (
                                        item.detection_source_index
                                    ),
                                    "cost": item.cost,
                                    "confidence": item.confidence,
                                    "bbox_iou": item.bbox_iou,
                                    "predicted_center_residual_norm": (
                                        item.predicted_center_residual_norm
                                    ),
                                    "log_width_change": item.log_width_change,
                                    "log_height_change": (
                                        item.log_height_change
                                    ),
                                    "log_area_residual": (
                                        item.log_area_residual
                                    ),
                                    "clipping_continuity": (
                                        item.clipping_continuity
                                    ),
                                    "temporal_consistency": (
                                        item.temporal_consistency
                                    ),
                                    "ambiguous": item.ambiguous,
                                }
                                for item in visual_update.associations
                            ],
                            graph=self._visual_graph_summary(
                                self._visual_latest_graph_snapshot
                            ),
                        )
                tracking_detections = detections
                if self._post_gate_reacquisition:
                    tracking_detections = [
                        detection
                        for detection in detections
                        if not is_crossing_residue(
                            detection,
                            image_width=image_width,
                            image_height=image_height,
                        )
                        and gate_detection_summary(
                            detection,
                            detector_index=0,
                            image_width=image_width,
                            image_height=image_height,
                            reject_crossing_residue=True,
                        )["selector_eligible"]
                    ]
                tracking_started_ns = (
                    time.perf_counter_ns() if capture_enabled else None
                )
                race = self.adapter.race_status
                course_edge_gate_index = self._course_edge_continuation_gate_index
                allow_course_edge_continuation = bool(
                    course_edge_gate_index is not None
                    and race is not None
                    and not race.race_finished
                    and int(race.active_gate_index) == course_edge_gate_index
                )
                accepted = self.tracker.update(
                    tracking_detections,
                    frame_id=snapshot.frame_id,
                    sim_time_ns=snapshot.sim_time_ns,
                    received_monotonic_s=snapshot.received_monotonic_s,
                    allow_tracked_edge_continuation=allow_course_edge_continuation,
                    allow_tracked_fragment_union=allow_course_edge_continuation,
                    image_width=image_width,
                    image_height=image_height,
                )
                tracking_ended_ns = (
                    time.perf_counter_ns() if capture_enabled else None
                )
                self._latest_accepted_target = accepted
                if self._post_gate_reacquisition:
                    self._post_gate_last_frame = (
                        (
                            int(snapshot.generation),
                            int(snapshot.frame_id),
                            int(snapshot.sim_time_ns),
                        ),
                        image,
                    )
                summaries = (
                    [
                        gate_detection_summary(
                            detection,
                            detector_index=index,
                            image_width=image_width,
                            image_height=image_height,
                            reject_crossing_residue=self._post_gate_reacquisition,
                        )
                        for index, detection in enumerate(detections)
                    ]
                    if self._vision_diagnostic_logging or capture_enabled
                    else []
                )
                if self._vision_diagnostic_logging:
                    selected = self.tracker.last_selected_detection
                    selected_index = next(
                        (
                            index
                            for index, detection in enumerate(detections)
                            if detection is selected
                        ),
                        None,
                    )
                    selected_indices = [
                        index
                        for index, detection in enumerate(detections)
                        if any(
                            detection is selected_detection
                            for selected_detection in (
                                self.tracker.last_selected_detections
                            )
                        )
                    ]
                    race = self.adapter.race_status
                    estimate = self.estimate
                    self.recorder.emit(
                        "vision_detection_frame",
                        phase=(
                            "gate1_reacquisition"
                            if self._post_gate_reacquisition
                            else "gate0_crossing"
                        ),
                        frame_id=snapshot.frame_id,
                        sim_time_ns=snapshot.sim_time_ns,
                        generation=snapshot.generation,
                        received_monotonic_s=snapshot.received_monotonic_s,
                        receive_age_s=snapshot.age_s(now),
                        image_size_px=[image_width, image_height],
                        race_boot_ms=(race.sim_boot_time_ms if race else None),
                        gate_index=(race.active_gate_index if race else None),
                        detections=summaries,
                        selected_detection_index=selected_index,
                        tracker_selected_detection_indices=selected_indices,
                        tracker_selection_mode=self.tracker.last_selection_mode,
                        tracker_streak=self.tracker.consecutive,
                        accepted_target=(asdict(accepted) if accepted else None),
                        tracker_target=(
                            asdict(self.tracker.target) if self.tracker.target else None
                        ),
                        rpy=(
                            list(estimate.orientation.to_euler()) if estimate else None
                        ),
                        body_rates=(list(estimate.body_rates) if estimate else None),
                        last_command=(
                            asdict(self._last_flight_command)
                            if self._last_flight_command
                            else None
                        ),
                    )
                if capture_enabled:
                    if snapshot.timing is None:
                        raise ValueError(
                            "capture-loaded frame lacks exact FrameTimingV1"
                        )
                    assert work_started_ns is not None
                    assert consume_monotonic_ns is not None
                    assert detector_started_ns is not None
                    assert detector_ended_ns is not None
                    assert tracking_started_ns is not None
                    assert tracking_ended_ns is not None
                    record_timing = getattr(
                        self.recorder, "record_camera_timing", None
                    )
                    if not callable(record_timing):
                        raise ValueError(
                            "capture recorder cannot preserve camera timing"
                        )
                capture_frame = getattr(self.recorder, "capture_frame", None)
                if capture_enabled and callable(capture_frame):
                    current_telemetry = self.adapter.latest_telemetry
                    current_imu = (
                        current_telemetry.imu
                        if current_telemetry is not None
                        else None
                    )
                    current_command = (
                        asdict(self._last_flight_command)
                        if self._last_flight_command is not None
                        else None
                    )
                    capture_frame(
                        image,
                        generation=int(snapshot.generation),
                        frame_id=int(snapshot.frame_id),
                        sim_time_ns=int(snapshot.sim_time_ns),
                        received_monotonic_s=float(snapshot.received_monotonic_s),
                        detector_latency_ms=detector_latency_ms,
                        detections=summaries,
                        tracker={
                            "consecutive": self.tracker.consecutive,
                            "target": asdict(self.tracker.target) if self.tracker.target else None,
                            **(
                                {
                                    "visual_gate_graph": (
                                        self._visual_graph_summary(
                                            self._visual_latest_graph_snapshot
                                        )
                                    ),
                                    "visual_tracks": (
                                        [
                                            self._visual_track_summary(track)
                                            for track in (
                                                self._visual_latest_tracker_update.tracks
                                            )
                                        ]
                                        if self._visual_latest_tracker_update
                                        is not None
                                        else []
                                    ),
                                }
                                if self._visual_tracking_enabled
                                else {}
                            ),
                        },
                        imu=current_imu,
                        estimator=self._replay_estimator_fields(),
                        race_status=self.adapter.race_status,
                        generated_command=current_command,
                        sent_command=current_command,
                        phase=(
                            "gate1_reacquisition"
                            if self._post_gate_reacquisition
                            else "gate0_or_preflight"
                        ),
                    )
                if capture_enabled:
                    # End-to-end passive frame work includes the synchronous
                    # replay snapshot/copy enqueue above.  The asynchronous
                    # writer remains separately diagnosed by capture stats.
                    observation = CameraFrameTimingObservationV1(
                        frame_timing=snapshot.timing,
                        consume_monotonic_ns=consume_monotonic_ns,
                        work_start_monotonic_ns=work_started_ns,
                        detection_start_monotonic_ns=detector_started_ns,
                        detection_end_monotonic_ns=detector_ended_ns,
                        tracking_start_monotonic_ns=tracking_started_ns,
                        tracking_end_monotonic_ns=tracking_ended_ns,
                        work_end_monotonic_ns=time.perf_counter_ns(),
                    )
                    record_timing(observation)
            except Exception as exc:  # OpenCV errors must fail closed in flight.
                if detector_started_ns is not None:
                    detector_latency_ms = (
                        time.perf_counter_ns() - detector_started_ns
                    ) / 1_000_000.0
                self._latest_raw_detections = []
                self._latest_accepted_target = None
                self._detection_error = f"{type(exc).__name__}: {exc}"
                self.recorder.emit(
                    "frame_processing_error",
                    generation=int(snapshot.generation),
                    frame_id=int(snapshot.frame_id),
                    sim_time_ns=int(snapshot.sim_time_ns),
                    reason=self._detection_error,
                )

    def _powered_vision_readiness(
        self,
        now: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Return post-reset frame-rate and exact-dimension readiness facts."""

        checked = time.monotonic() if now is None else float(now)
        failures: List[str] = []
        started = self._epoch_vision_started_s
        initial_frames = self._epoch_vision_initial_frames
        elapsed = None if started is None else max(0.0, checked - started)
        frames: Optional[int] = None
        fps: Optional[float] = None
        try:
            stats = self.vision.stats()
            observed_frames = getattr(stats, "frames_decoded", None)
            if type(observed_frames) is not int or observed_frames < 0:
                raise ValueError("decoded frame count is invalid")
            frames = observed_frames
        except Exception:
            failures.append("vision frame-rate statistics are unavailable")
        if started is None or initial_frames is None:
            failures.append("post-reset vision rate baseline is unavailable")
        elif elapsed is None or elapsed < 1.0:
            failures.append("post-reset vision rate has less than 1.0s evidence")
        elif frames is not None:
            decoded = frames - initial_frames
            if decoded < 0:
                failures.append("post-reset decoded frame count regressed")
            else:
                fps = decoded / max(elapsed, 1e-6)
                if fps < 20.0:
                    failures.append(f"post-reset vision rate is below 20fps ({fps:.1f})")

        image = self._latest_detection_image
        shape = getattr(image, "shape", None)
        dimensions = None
        if shape is not None and len(shape) >= 2:
            dimensions = [int(shape[1]), int(shape[0])]
        if dimensions != [640, 360]:
            failures.append("decoded dimensions are not stable 640x360")
        return failures, {
            "observation_s": elapsed,
            "frames_decoded": (
                None
                if frames is None or initial_frames is None
                else frames - initial_frames
            ),
            "fps": fps,
            "dimensions_px": dimensions,
        }

    def _stream_failures(
        self,
        *,
        require_estimator: bool,
        require_target: bool,
        require_armed: bool,
    ) -> List[str]:
        now = time.monotonic()
        failures: List[str] = []
        if self.adapter.heartbeat_age_s > MAX_HEARTBEAT_AGE_S:
            failures.append(f"heartbeat stale ({self.adapter.heartbeat_age_s:.3f}s)")
        if self.adapter.imu_age_s > MAX_IMU_AGE_S:
            failures.append(f"IMU receive stale ({self.adapter.imu_age_s:.3f}s)")
        if now - self._last_imu_advance_s > MAX_IMU_AGE_S:
            failures.append("IMU timestamp not advancing")
        if self.adapter.race_status_age_s > MAX_RACE_AGE_S:
            failures.append(f"race status stale ({self.adapter.race_status_age_s:.3f}s)")
        if now - self._last_race_advance_s > MAX_RACE_AGE_S:
            failures.append("race clock not advancing")
        if self.adapter.actuator_age_s > MAX_ACTUATOR_AGE_S:
            failures.append(f"actuator status stale ({self.adapter.actuator_age_s:.3f}s)")
        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if snapshot is None:
            failures.append("camera stale or unavailable")
        if self._detection_error is not None:
            failures.append(f"gate detector failed: {self._detection_error}")
        if self._imu_regressed:
            failures.append("IMU timestamp regressed")
        if self._race_regressed:
            failures.append("race clock regressed")
        if self._imu_forward_jump:
            failures.append("IMU timestamp jumped outside proved reset epoch")
        if self._race_forward_jump:
            failures.append("race clock jumped outside proved reset epoch")
        if self._estimator_unhealthy_latched:
            failures.append(
                "attitude estimator failure latched "
                f"({self._estimator_failure_reason or 'unknown reason'})"
            )
        if require_estimator:
            if not self.estimator.is_ready or self.estimate is None:
                failures.append("attitude estimator not ready")
            elif not self.estimate.healthy:
                failures.append(f"attitude estimator unhealthy ({self.estimate.reason})")
        if require_target:
            if not self.tracker.fresh(MAX_TARGET_LOSS_S, now):
                failures.append("primary gate target lost")
        if require_armed and not self.adapter.is_armed:
            failures.append("vehicle no longer armed")
        return failures

    def _watchdog(
        self,
        *,
        require_target: bool = True,
        allow_benign_pad_contact: bool = False,
        enforce_benign_pad_budget: bool = False,
        benign_pad_max_impulse: float = 0.01,
        count_rate_sample: bool = True,
    ) -> None:
        if self._abort_latched:
            raise SafetyAbort("abort already latched")
        failures = self._stream_failures(
            require_estimator=True,
            require_target=require_target,
            require_armed=True,
        )
        collisions = self.adapter.drain_collisions()
        if collisions:
            harmful = []
            for collision in collisions:
                benign_pad = allow_benign_pad_contact and is_benign_pad_contact(
                    collision,
                    max_impulse=benign_pad_max_impulse,
                )
                if benign_pad:
                    self._benign_pad_contact_count += 1
                    self._benign_pad_contact_impulse += abs(float(collision["impulse"]))
                    self.recorder.emit(
                        "benign_pad_contact",
                        collision=collision,
                        cumulative_count=self._benign_pad_contact_count,
                        cumulative_impulse=self._benign_pad_contact_impulse,
                    )
                else:
                    harmful.append(collision)
            if harmful:
                failures.append(f"collision reported: {harmful!r}")
            if enforce_benign_pad_budget and (
                self._benign_pad_contact_count > MAX_BENIGN_PAD_CONTACTS
                or self._benign_pad_contact_impulse > MAX_BENIGN_PAD_IMPULSE
            ):
                failures.append(
                    "repeated pad contacts exceeded launch budget "
                    f"(count={self._benign_pad_contact_count}, "
                    f"impulse={self._benign_pad_contact_impulse:.3f})"
                )
        if self.estimate is not None:
            roll, pitch, _yaw = self.estimate.orientation.to_euler()
            rates = self.estimate.body_rates
            if not all(math.isfinite(value) for value in (roll, pitch, *rates)):
                failures.append("non-finite estimated state")
            if abs(roll) > MAX_ROLL_RAD:
                failures.append(f"roll limit exceeded ({math.degrees(roll):.1f}deg)")
            if pitch < MIN_PITCH_RAD or pitch > MAX_PITCH_RAD:
                failures.append(f"pitch limit exceeded ({math.degrees(pitch):.1f}deg)")
            peak_rate = max(abs(value) for value in rates)
            if peak_rate > IMMEDIATE_MAX_BODY_RATE_RAD_S:
                failures.append(f"body rate immediate limit exceeded ({peak_rate:.2f}rad/s)")
            if count_rate_sample:
                self._high_rate_samples = self._high_rate_samples + 1 if (
                    peak_rate > MAX_BODY_RATE_RAD_S
                ) else 0
            if self._high_rate_samples >= 2:
                failures.append(f"body rate sustained limit exceeded ({peak_rate:.2f}rad/s)")
        if failures:
            self._abort_latched = True
            raise SafetyAbort("; ".join(failures))

    def _record_tick(
        self,
        stage: str,
        elapsed_s: float,
        command: Optional[AttitudeRateCommand],
    ) -> None:
        race = self.adapter.race_status
        telemetry = self.adapter.latest_telemetry
        imu = telemetry.imu if telemetry is not None else None
        target = self.tracker.target
        estimate = self.estimate
        self.recorder.emit(
            "tick",
            stage=stage,
            elapsed_s=elapsed_s,
            imu_us=self._last_imu_us,
            imu_accel=(list(imu.accel) if imu else None),
            race_boot_ms=(race.sim_boot_time_ms if race else None),
            race_start_ms=(race.race_start_boot_time_ms if race else None),
            gate_index=(race.active_gate_index if race else None),
            rpy=(list(estimate.orientation.to_euler()) if estimate else None),
            body_rates=(list(estimate.body_rates) if estimate else None),
            target=(asdict(target) if target else None),
            command=(asdict(command) if command else None),
        )

    @staticmethod
    def _outbound_receipt_primitive(receipt: Any) -> Dict[str, Any]:
        if isinstance(receipt, Mapping):
            return dict(receipt)
        convert = getattr(receipt, "to_primitive", None)
        if callable(convert):
            value = convert()
            if isinstance(value, Mapping):
                return dict(value)
        try:
            value = asdict(receipt)
        except (TypeError, ValueError) as exc:
            raise SafetyAbort("adapter returned an invalid outbound receipt") from exc
        if not isinstance(value, Mapping):
            raise SafetyAbort("adapter returned an invalid outbound receipt")
        return dict(value)

    async def _send_flight_command(
        self,
        command: AttitudeRateCommand,
        *,
        require_wire_receipt: bool = False,
        wire_start_not_before_ns: Optional[int] = None,
        wire_start_deadline_ns: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send one validated setpoint and optionally prove wire-call timing."""

        if type(require_wire_receipt) is not bool:
            raise TypeError("require_wire_receipt must be an exact bool")
        for label, value in (
            ("wire_start_not_before_ns", wire_start_not_before_ns),
            ("wire_start_deadline_ns", wire_start_deadline_ns),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{label} must be a non-negative exact integer")
        if (
            (wire_start_not_before_ns is not None or wire_start_deadline_ns is not None)
            and not require_wire_receipt
        ):
            raise ValueError("wire timing guards require an exact outbound receipt")
        drain_receipts = getattr(self.adapter, "drain_outbound_receipts", None)
        if require_wire_receipt:
            if not callable(drain_receipts):
                raise SafetyAbort("adapter lacks exact outbound receipt timing")
            stale = [
                self._outbound_receipt_primitive(value)
                for value in drain_receipts()
            ]
            if any(
                value.get("schema") == "aigp-vq2-attitude-target-outbound/1"
                for value in stale
            ):
                raise SafetyAbort("unexpected queued attitude-target receipt")

        generated_at = time.monotonic()
        frame_token = self._latest_frame_token()
        record_command = getattr(self.recorder, "record_command", None)
        if callable(record_command):
            record_command(
                "generated",
                command,
                monotonic_s=generated_at,
                frame_token=frame_token,
            )
        validate_command(command)
        send_options: Dict[str, Any] = {}
        if wire_start_not_before_ns is not None:
            send_options["call_start_not_before_monotonic_ns"] = (
                wire_start_not_before_ns
            )
        if wire_start_deadline_ns is not None:
            send_options["call_start_deadline_monotonic_ns"] = wire_start_deadline_ns
        await self.adapter.send_attitude_rate(command, **send_options)
        self._last_flight_command = command
        self._last_flight_command_sent_s = time.monotonic()
        wire_receipt: Optional[Dict[str, Any]] = None
        if require_wire_receipt:
            assert callable(drain_receipts)
            receipts = [
                self._outbound_receipt_primitive(value)
                for value in drain_receipts()
            ]
            attitude_receipts = [
                value
                for value in receipts
                if value.get("schema")
                == "aigp-vq2-attitude-target-outbound/1"
            ]
            if len(attitude_receipts) != 1:
                raise SafetyAbort(
                    "adapter did not return exactly one attitude-target receipt"
                )
            wire_receipt = attitude_receipts[0]
            call_start = wire_receipt.get("call_start_monotonic_ns")
            call_end = wire_receipt.get("call_end_monotonic_ns")
            if (
                wire_receipt.get("host_clock_id") != "host-perf-counter"
                or wire_receipt.get("api") != "send_attitude_rate"
                or wire_receipt.get("outcome") != "returned"
                or type(call_start) is not int
                or type(call_end) is not int
                or call_start < 0
                or call_end < call_start
            ):
                raise SafetyAbort("adapter outbound receipt is invalid")
            self._last_flight_command_started_ns = call_start
            self.recorder.emit("attitude_target_outbound", receipt=wire_receipt)
        if callable(record_command):
            record_command(
                "sent",
                command,
                monotonic_s=self._last_flight_command_sent_s,
                frame_token=frame_token,
            )
        return wire_receipt

    async def _wait_for_next_flight_command_slot(self) -> float:
        """Prove a full control period after the previous completed send."""

        now = time.monotonic()
        if (
            type(now) not in {int, float}
            or not math.isfinite(float(now))
            or float(now) < 0.0
        ):
            raise SafetyAbort("current flight-command timestamp is invalid")
        last_sent_s = self._last_flight_command_sent_s
        if last_sent_s is None:
            return float(now)
        if (
            type(last_sent_s) not in {int, float}
            or not math.isfinite(float(last_sent_s))
            or float(last_sent_s) < 0.0
            or float(last_sent_s) > float(now)
        ):
            raise SafetyAbort("last flight-command timestamp is invalid")
        not_before_s = float(last_sent_s) + CONTROL_PERIOD_S
        observed_s = float(now)
        wait_attempts = 0
        while observed_s < not_before_s:
            if wait_attempts >= 8:
                raise SafetyAbort("flight-command pacing wait returned early")
            await asyncio.sleep(not_before_s - observed_s)
            next_observed_s = time.monotonic()
            if (
                type(next_observed_s) not in {int, float}
                or not math.isfinite(float(next_observed_s))
                or float(next_observed_s) < observed_s
            ):
                raise SafetyAbort("flight-command pacing wait returned early")
            observed_s = float(next_observed_s)
            wait_attempts += 1
        ready_s = time.monotonic()
        if (
            type(ready_s) not in {int, float}
            or not math.isfinite(float(ready_s))
            or float(ready_s) < max(observed_s, not_before_s)
        ):
            raise SafetyAbort("flight-command pacing wait returned early")
        return float(ready_s)

    @staticmethod
    def _is_exact_zero_command(command: Optional[AttitudeRateCommand]) -> bool:
        return bool(
            command is not None
            and command.roll_rate == 0.0
            and command.pitch_rate == 0.0
            and command.yaw_rate == 0.0
            and command.thrust == 0.0
        )

    def _defer_snapshot(self, label: str) -> Optional[Dict[str, Any]]:
        """Copy a diagnostic frame in memory; encoding happens after cleanup."""

        if self.recorder.path is None:
            return None
        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if snapshot is None:
            self.recorder.emit("diagnostic_snapshot_unavailable", label=label)
            return None
        image = getattr(snapshot.camera_frame, "image", None)
        if image is None:
            self.recorder.emit("diagnostic_snapshot_unavailable", label=label)
            return None
        # VQ2 snapshots publish a new read-only ndarray per decoded frame.
        # Holding the reference is sufficient; copy/encode only after cleanup.
        self._deferred_pngs.append((str(label), image))
        metadata = {
            "label": str(label),
            "frame_id": int(snapshot.frame_id),
            "sim_time_ns": int(snapshot.sim_time_ns),
            "generation": int(snapshot.generation),
            "received_monotonic_s": float(snapshot.received_monotonic_s),
        }
        return metadata

    def _flush_deferred_snapshots(self) -> Tuple[List[str], List[str]]:
        paths: List[str] = []
        errors: List[str] = []
        pending = self._deferred_pngs
        self._deferred_pngs = []
        for label, image in pending:
            try:
                path = self.recorder.save_png(label, image)
                if path is not None:
                    paths.append(path)
                    self.recorder.emit(
                        "diagnostic_snapshot_saved",
                        label=label,
                        path=path,
                    )
            except Exception as exc:
                message = f"{label}: {type(exc).__name__}: {exc}"
                errors.append(message)
                logger.exception("Could not save deferred diagnostic snapshot %s", label)
                self.recorder.emit(
                    "diagnostic_snapshot_save_failed",
                    label=label,
                    reason=message,
                )
        return paths, errors

    async def preflight(
        self,
        timeout_s: float = 10.0,
        *,
        healthy_dwell_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Passively validate feeds, estimator bootstrap, detector, and rate."""

        if (
            type(healthy_dwell_s) not in {int, float}
            or not math.isfinite(healthy_dwell_s)
            or not 0.0 <= float(healthy_dwell_s) <= 8.0
        ):
            raise ValueError("healthy_dwell_s must be finite and in [0, 8]")
        dwell_s = float(healthy_dwell_s)

        if not self.vision.is_running:
            self.vision.start()
        self._clear_epoch_state()
        start = time.monotonic()
        initial_frames = self.vision.stats().frames_decoded
        last_log = start
        ready_since: Optional[float] = None
        while time.monotonic() - start < timeout_s:
            self._sample()
            elapsed = time.monotonic() - start
            stats = self.vision.stats()
            fps = (stats.frames_decoded - initial_frames) / max(elapsed, 1e-6)
            failures = self._stream_failures(
                require_estimator=True,
                require_target=True,
                require_armed=False,
            )
            ready = (
                elapsed >= 1.0
                and fps >= 20.0
                and self.tracker.consecutive >= 3
                and not failures
            )
            if ready:
                if ready_since is None:
                    ready_since = time.monotonic()
                healthy_elapsed = time.monotonic() - ready_since
            else:
                ready_since = None
                healthy_elapsed = 0.0
            if ready and healthy_elapsed >= dwell_s:
                assert self.estimate is not None and self.tracker.target is not None
                roll, pitch, yaw = self.estimate.orientation.to_euler()
                result = {
                    "vision_fps": fps,
                    "vision_frames": stats.frames_decoded - initial_frames,
                    "vision_duplicates": stats.duplicate_datagrams,
                    "imu_us": self._last_imu_us,
                    "attitude_rpy_rad": [roll, pitch, yaw],
                    "gyro_bias_rad_s": list(self.estimator.gyro_bias),
                    "gate_bbox": list(self.tracker.target.bbox),
                    "gate_center": [self.tracker.target.center_x, self.tracker.target.center_y],
                    "gate_confidence": self.tracker.target.confidence,
                    "race_gate_index": self.adapter.race_status.active_gate_index,
                    "healthy_dwell_s": healthy_elapsed,
                    "requested_healthy_dwell_s": dwell_s,
                    "observation_duration_s": elapsed,
                    # Build 3385 can boot Training with this bit already set,
                    # despite zero actuator demand.  Powered stages explicitly
                    # normalize to disarmed after their proved reset.
                    "sim_reports_armed": self.adapter.is_armed,
                }
                self.recorder.emit("preflight_pass", **result)
                logger.info(
                    "Preflight PASS: vision %.1f fps, IMU fresh, attitude "
                    "(roll=%.2fdeg pitch=%.2fdeg), gate bbox=%s",
                    fps,
                    math.degrees(roll),
                    math.degrees(pitch),
                    self.tracker.target.bbox,
                )
                return result
            if time.monotonic() - last_log >= 1.0:
                logger.info(
                    "Preflight: %.1fs, vision %.1ffps, calibration %.0f%%, "
                    "gate streak %d, waiting=%s",
                    elapsed,
                    fps,
                    100.0 * self.estimator.calibration_progress,
                    self.tracker.consecutive,
                    failures[:3],
                )
                last_log = time.monotonic()
            await asyncio.sleep(0.005)
        failures = self._stream_failures(
            require_estimator=True,
            require_target=True,
            require_armed=False,
        )
        raise SafetyAbort(f"preflight timed out: {failures}")

    async def _fresh_reset_baseline(self, timeout_s: float = 2.0) -> Tuple[int, int]:
        deadline = time.monotonic() + timeout_s
        last_race: Optional[int] = None
        last_imu: Optional[int] = None
        race_advances = 0
        imu_advances = 0
        current_pair: Optional[Tuple[int, int]] = None
        while time.monotonic() < deadline:
            telemetry = self.adapter.latest_telemetry
            imu = telemetry.imu if telemetry is not None else None
            race = self.adapter.race_status
            if (
                imu is not None
                and race is not None
                and self.adapter.imu_age_s <= MAX_IMU_AGE_S
                and self.adapter.race_status_age_s <= MAX_RACE_AGE_S
                and race.sim_boot_time_ms >= 800
                and imu.timestamp_us >= 200_000
            ):
                pair = (int(race.sim_boot_time_ms), int(imu.timestamp_us))
                if last_race is not None and pair[0] > last_race:
                    race_advances += 1
                if last_imu is not None and pair[1] > last_imu:
                    imu_advances += 1
                last_race, last_imu = pair
                current_pair = pair
                # These streams have very different cadences (about 4 Hz vs
                # 145 Hz).  Prove each independently; requiring simultaneous
                # advancement makes a healthy baseline impossible.
                if race_advances >= 2 and imu_advances >= 5:
                    return current_pair
            await asyncio.sleep(0.01)
        raise SafetyAbort("could not obtain fresh, advancing pre-reset race/IMU clocks")

    async def _observe_reset_proof(
        self,
        *,
        attempt: int,
        pre_race: int,
        pre_imu: int,
    ) -> Optional[ResetProof]:
        """Observe rollback after a reset that has already been sent."""

        deadline = time.monotonic() + RESET_PROOF_TIMEOUT_S
        race_samples: List[int] = []
        imu_samples: List[int] = []
        countdown_observed = False
        while time.monotonic() < deadline:
            telemetry = self.adapter.latest_telemetry
            imu = telemetry.imu if telemetry is not None else None
            race = self.adapter.race_status
            if race is not None:
                boot = int(race.sim_boot_time_ms)
                if clock_rolled_back(pre_race, boot, RESET_RACE_DROP_MS):
                    if not race_samples or boot > race_samples[-1]:
                        race_samples.append(boot)
                    if race.race_start_boot_time_ms < 0 or boot < race.race_start_boot_time_ms:
                        countdown_observed = True
            if imu is not None:
                stamp = int(imu.timestamp_us)
                if clock_rolled_back(pre_imu, stamp, RESET_IMU_DROP_US):
                    if not imu_samples or stamp > imu_samples[-1]:
                        imu_samples.append(stamp)
            if (
                len(race_samples) >= 2
                and len(imu_samples) >= 5
                and countdown_observed
            ):
                return ResetProof(
                    attempt=attempt,
                    pre_race_boot_ms=pre_race,
                    post_race_boot_ms=race_samples[-1],
                    pre_imu_us=pre_imu,
                    post_imu_us=imu_samples[-1],
                    advancing_race_samples=len(race_samples),
                    advancing_imu_samples=len(imu_samples),
                    countdown_observed=True,
                )
            await asyncio.sleep(0.005)
        return None

    def _accept_reset_proof(self, proof: ResetProof, *, restart_vision: bool) -> None:
        drain_imu = getattr(self.adapter, "drain_imu_samples", None)
        if callable(drain_imu):
            # Proof observation intentionally does not estimate attitude.  Drop
            # that accumulated batch and begin calibration on samples received
            # strictly after the accepted boundary.
            drain_imu()
        self._clear_epoch_state()
        if self._visual_tracking_enabled:
            self._visual_reset_epoch += 1
        self._epoch_race_anchor_ms = proof.post_race_boot_ms
        self._epoch_imu_anchor_us = proof.post_imu_us
        self._epoch_anchor_monotonic_s = time.monotonic()
        self._countdown_observed = proof.countdown_observed
        self.adapter.drain_collisions()
        if restart_vision:
            self.vision.reset()
            self.vision.start()
            self._epoch_vision_started_s = time.monotonic()
            frames_decoded = getattr(self.vision.stats(), "frames_decoded", None)
            if type(frames_decoded) is not int or frames_decoded < 0:
                raise SafetyAbort("vision did not expose a valid post-reset frame count")
            self._epoch_vision_initial_frames = frames_decoded
        self.recorder.emit("reset_proved", **asdict(proof))
        logger.info(
            "Reset epoch proved on attempt %d: race %d->%dms, IMU %d->%dus",
            proof.attempt,
            proof.pre_race_boot_ms,
            proof.post_race_boot_ms,
            proof.pre_imu_us,
            proof.post_imu_us,
        )

    async def establish_reset_epoch(self, *, restart_vision: bool) -> ResetProof:
        """Send reset and prove both authoritative simulator clocks rolled back."""

        self.vision.stop()
        self.tracker.reset()
        for attempt in range(1, RESET_MAX_ATTEMPTS + 1):
            pre_race, pre_imu = await self._fresh_reset_baseline()
            self.recorder.emit(
                "reset_sent",
                attempt=attempt,
                emergency=False,
                pre_race_boot_ms=pre_race,
                pre_gate_index=(
                    self.adapter.race_status.active_gate_index
                    if self.adapter.race_status is not None
                    else None
                ),
                pre_imu_us=pre_imu,
            )
            await self.adapter.reset()
            proof = await self._observe_reset_proof(
                attempt=attempt,
                pre_race=pre_race,
                pre_imu=pre_imu,
            )
            if proof is not None:
                self._accept_reset_proof(proof, restart_vision=restart_vision)
                return proof
            logger.warning("Reset attempt %d was not authoritatively proved; retrying", attempt)
        raise SafetyAbort("SIM_RESET was not proved after four bounded attempts")

    async def emergency_reset(self) -> Optional[ResetProof]:
        """Always send reset first; proof is best-effort and never a send gate."""

        self.vision.stop()
        self.tracker.reset()
        for attempt in range(1, RESET_MAX_ATTEMPTS + 1):
            telemetry = self.adapter.latest_telemetry
            imu = telemetry.imu if telemetry is not None else None
            race = self.adapter.race_status
            pre_race = int(race.sim_boot_time_ms) if race is not None else None
            pre_imu = int(imu.timestamp_us) if imu is not None else None
            self.recorder.emit(
                "reset_sent",
                attempt=attempt,
                emergency=True,
                pre_race_boot_ms=pre_race,
                pre_gate_index=(int(race.active_gate_index) if race is not None else None),
                pre_imu_us=pre_imu,
            )
            # This send is deliberately unconditional.  Stale/missing streams
            # may prevent proof, but can never prevent the emergency command.
            await self.adapter.reset()
            if pre_race is not None and pre_imu is not None:
                proof = await self._observe_reset_proof(
                    attempt=attempt,
                    pre_race=pre_race,
                    pre_imu=pre_imu,
                )
                if proof is not None:
                    self._accept_reset_proof(proof, restart_vision=False)
                    return proof
            else:
                await asyncio.sleep(0.5)
            logger.warning(
                "Emergency reset attempt %d was sent but not proved; retrying",
                attempt,
            )
        return None

    async def wait_for_go(self, timeout_s: float = 8.0) -> StartContext:
        deadline = time.monotonic() + timeout_s
        go_seen_at: Optional[float] = None
        while time.monotonic() < deadline:
            self._sample()
            race = self.adapter.race_status
            if self.adapter.is_armed:
                raise SafetyAbort("vehicle became armed before runner issued post-GO arm")
            collisions = self.adapter.drain_collisions()
            if collisions:
                raise SafetyAbort(f"collision during countdown: {collisions!r}")
            if (
                self._imu_regressed
                or self._race_regressed
                or self._imu_forward_jump
                or self._race_forward_jump
            ):
                raise SafetyAbort("clock left the proved reset epoch")
            if race is not None and race.active_gate_index != 0:
                raise SafetyAbort(f"fresh race did not start at gate 0 ({race.active_gate_index})")
            if (
                race is not None
                and self._countdown_observed
                and race.race_start_boot_time_ms >= 0
                and race.sim_boot_time_ms >= race.race_start_boot_time_ms + 150
            ):
                go_seen_at = go_seen_at or time.monotonic()
                failures = self._stream_failures(
                    require_estimator=True,
                    require_target=True,
                    require_armed=False,
                )
                vision_failures, vision_readiness = (
                    self._powered_vision_readiness(time.monotonic())
                )
                failures.extend(vision_failures)
                if self.tracker.consecutive < 3:
                    failures.append("gate target lacks three-frame confirmation")
                if not self._countdown_observed:
                    failures.append("fresh post-reset countdown was not observed")
                if not failures:
                    assert self.estimate is not None and self.tracker.target is not None
                    roll, pitch, _yaw = self.estimate.orientation.to_euler()
                    if abs(roll) > math.radians(5.0):
                        raise SafetyAbort("pad calibration roll is implausible")
                    if not MIN_PITCH_RAD <= pitch <= MAX_PITCH_RAD:
                        raise SafetyAbort("pad calibration pitch is outside safety envelope")
                    context = StartContext(
                        spawn_roll_rad=roll,
                        spawn_pitch_rad=pitch,
                        initial_gate_x=self.tracker.target.center_x,
                        initial_gate_y=self.tracker.target.center_y,
                        initial_gate_area=self.tracker.target.bbox_area,
                        go_boot_ms=int(race.sim_boot_time_ms),
                    )
                    self.recorder.emit(
                        "powered_vision_ready",
                        **vision_readiness,
                    )
                    self.recorder.emit("go_ready", **asdict(context))
                    return context
                if time.monotonic() - go_seen_at > 1.0:
                    raise SafetyAbort(f"GO passed without full readiness: {failures}")
            await asyncio.sleep(0.005)
        raise SafetyAbort("timed out waiting for fresh reset countdown and GO")

    async def arm_confirmed(self, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            token = self.adapter.heartbeat_sequence
            await self.adapter.arm()
            confirm_deadline = min(deadline, time.monotonic() + 0.45)
            while time.monotonic() < confirm_deadline:
                self._sample()
                if self.adapter.heartbeat_sequence > token and self.adapter.is_armed:
                    self.recorder.emit("arm_confirmed", heartbeat_sequence=self.adapter.heartbeat_sequence)
                    logger.info("Arming confirmed by a post-attempt heartbeat")
                    return
                failures = self._stream_failures(
                    require_estimator=True,
                    require_target=True,
                    require_armed=False,
                )
                if failures:
                    raise SafetyAbort(f"stream failure while confirming arm: {failures}")
                await asyncio.sleep(0.01)
        raise SafetyAbort("arming was not confirmed by a newer heartbeat")

    async def normalize_disarmed(self) -> None:
        """Force and confirm a ground-safe state before waiting for GO."""

        if not await self._disarm_confirmed():
            raise SafetyAbort("could not confirm disarmed state after reset")
        if self.adapter.is_armed:
            raise SafetyAbort("simulator still reports armed after disarm confirmation")
        self.recorder.emit("ground_disarmed")

    async def _disarm_confirmed(self, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            token = self.adapter.heartbeat_sequence
            try:
                await self.adapter.disarm()
            except Exception:
                logger.exception("Disarm send failed")
            confirm_deadline = min(deadline, time.monotonic() + 0.25)
            while time.monotonic() < confirm_deadline:
                if self.adapter.heartbeat_sequence > token and not self.adapter.is_armed:
                    self.recorder.emit(
                        "disarm_confirmed",
                        heartbeat_sequence=self.adapter.heartbeat_sequence,
                    )
                    return True
                await asyncio.sleep(0.01)
        return False

    async def safe_cleanup(self) -> bool:
        """Latch command production, cut thrust, confirm disarm, then reset."""

        self._abort_latched = True
        race_before_cleanup = self.adapter.race_status
        gate_index_before_cleanup = (
            int(race_before_cleanup.active_gate_index)
            if race_before_cleanup is not None
            else None
        )
        race_boot_before_cleanup = (
            int(race_before_cleanup.sim_boot_time_ms)
            if race_before_cleanup is not None
            else None
        )
        try:
            if self.adapter.is_armed:
                zero = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
                zero_is_recent = bool(
                    self._is_exact_zero_command(self._last_flight_command)
                    and self._last_flight_command_sent_s is not None
                    and time.monotonic() - self._last_flight_command_sent_s
                    < CONTROL_PERIOD_S
                )
                if zero_is_recent:
                    self.recorder.emit(
                        "zero_thrust_already_active",
                        gate_index=gate_index_before_cleanup,
                        race_boot_ms=race_boot_before_cleanup,
                    )
                else:
                    cleanup_send_started = time.monotonic()
                    frame_token = self._latest_frame_token()
                    record_command = getattr(self.recorder, "record_command", None)
                    if callable(record_command):
                        record_command(
                            "generated",
                            zero,
                            monotonic_s=cleanup_send_started,
                            frame_token=frame_token,
                        )
                    await self.adapter.send_attitude_rate(zero)
                    self._last_flight_command = zero
                    self._last_flight_command_sent_s = time.monotonic()
                    if callable(record_command):
                        record_command(
                            "sent",
                            zero,
                            monotonic_s=self._last_flight_command_sent_s,
                            frame_token=frame_token,
                        )
                    self.recorder.emit(
                        "zero_thrust_sent",
                        gate_index=gate_index_before_cleanup,
                        race_boot_ms=race_boot_before_cleanup,
                    )
        except Exception:
            logger.exception("Could not send the one-shot zero-thrust command")
        # Do not delay the unconditional reset fallback behind a long heartbeat
        # wait when the abort itself may be a transport failure.
        disarmed = await self._disarm_confirmed(timeout_s=0.6)
        if not disarmed:
            logger.error("Disarm was not confirmed before reset fallback")

        reset_proved = False
        try:
            reset_proved = await self.emergency_reset() is not None
        except Exception:
            logger.exception("Emergency SIM_RESET send/proof path failed")

        if not disarmed or self.adapter.is_armed:
            disarmed = await self._disarm_confirmed()
        confirmed = bool(disarmed and reset_proved and not self.adapter.is_armed)
        self.recorder.emit(
            "cleanup_complete",
            disarmed=disarmed,
            reset_proved=reset_proved,
            confirmed=confirmed,
            gate_index_before_cleanup=gate_index_before_cleanup,
            race_boot_before_cleanup=race_boot_before_cleanup,
        )
        if not confirmed:
            logger.critical("UNRESOLVED EMERGENCY: stop/reset state was not fully confirmed")
        return confirmed

    async def _run_sign_id(self) -> Dict[str, Any]:
        assert self.estimate is not None
        start_rpy = self.estimate.orientation.to_euler()
        max_excursion = 0.0
        max_abs_measured_yaw_rate = abs(float(self.estimate.body_rates[2]))
        segments = [
            (
                "neutral-pre-yaw",
                SIGN_ID_YAW_NEUTRAL_DURATION_S,
                (0.0, 0.0, 0.0),
            ),
            (
                "yaw-positive",
                SIGN_ID_YAW_PULSE_DURATION_S,
                (0.0, 0.0, SIGN_ID_RATE_RAD_S),
            ),
            (
                "neutral-reversal",
                SIGN_ID_YAW_REVERSAL_DURATION_S,
                (0.0, 0.0, 0.0),
            ),
            (
                "yaw-negative",
                SIGN_ID_YAW_PULSE_DURATION_S,
                (0.0, 0.0, -SIGN_ID_RATE_RAD_S),
            ),
            (
                "neutral-terminal",
                SIGN_ID_YAW_TERMINAL_DURATION_S,
                (0.0, 0.0, 0.0),
            ),
        ]
        responses: Dict[str, List[Tuple[int, float]]] = {
            "neutral-pre-yaw": [],
            "yaw-positive": [],
            "yaw-negative": [],
        }
        image_segment_names = (
            "neutral-pre-yaw",
            "yaw-positive",
            "yaw-negative",
        )
        image_samples: Dict[str, List[Tuple[float, float]]] = {
            name: [] for name in image_segment_names
        }
        wire_yaw_rates: Dict[str, List[float]] = {
            "yaw-positive": [],
            "yaw-negative": [],
        }

        def wrapped_yaw_excursion(yaw: float) -> float:
            delta = float(yaw) - float(start_rpy[2])
            return math.atan2(math.sin(delta), math.cos(delta))

        def image_slope(name: str) -> float:
            samples = image_samples[name]
            if len(samples) < SIGN_ID_MIN_FRESH_IMAGE_FRAMES:
                raise SafetyAbort(
                    f"sign-ID {name} lacks four fresh image frames"
                )
            if any(
                samples[index][0] <= samples[index - 1][0]
                for index in range(1, len(samples))
            ):
                raise SafetyAbort(
                    f"sign-ID {name} image times did not advance"
                )
            pairwise_slopes = [
                (later[1] - earlier[1]) / (later[0] - earlier[0])
                for index, earlier in enumerate(samples)
                for later in samples[index + 1 :]
            ]
            if not pairwise_slopes or not all(
                math.isfinite(value) for value in pairwise_slopes
            ):
                raise SafetyAbort(f"sign-ID {name} image slope is unavailable")
            return float(statistics.median(pairwise_slopes))

        def validate_wire_yaw(
            receipt: Optional[Mapping[str, Any]],
            command: AttitudeRateCommand,
        ) -> float:
            if not isinstance(receipt, Mapping):
                raise SafetyAbort("sign-ID lacks an outbound wire receipt")
            wire = receipt.get("wire")
            if not isinstance(wire, Mapping):
                raise SafetyAbort("sign-ID receipt lacks wire evidence")
            body_rates = wire.get("body_rates_rad_s")
            wire_thrust = wire.get("thrust")
            if (
                wire.get("type_mask") != 128
                or not isinstance(body_rates, Sequence)
                or len(body_rates) != 3
                or any(
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                    for value in body_rates
                )
                or tuple(float(value) for value in body_rates)
                != (
                    -float(command.roll_rate),
                    -float(command.pitch_rate),
                    -float(command.yaw_rate),
                )
                or type(wire_thrust) not in {int, float}
                or not math.isfinite(float(wire_thrust))
                or float(wire_thrust) != float(command.thrust)
            ):
                raise SafetyAbort("sign-ID yaw wire mapping is invalid")
            return float(body_rates[2])

        try:
            flight_start = await self._wait_for_next_flight_command_slot()
            next_tick = flight_start
            initial_target = self.tracker.target
            last_image_token = (
                None
                if (
                    initial_target is None
                    or type(self._latest_detection_generation) is not int
                )
                else (
                    int(self._latest_detection_generation),
                    int(initial_target.frame_id),
                    int(initial_target.sim_time_ns),
                )
            )
            active_segment: Optional[str] = None
            active_wire_release_s: Optional[float] = None
            for name, duration, rates in segments:
                segment_start = time.monotonic()
                while time.monotonic() - segment_start < duration:
                    elapsed = time.monotonic() - flight_start
                    if elapsed >= SIGN_ID_HARD_EXPIRY_S:
                        raise SafetyAbort("sign-ID hard expiry reached")
                    self._sample()
                    self._watchdog(allow_benign_pad_contact=True)
                    assert self.estimate is not None
                    race = self.adapter.race_status
                    if (
                        race is None
                        or int(race.active_gate_index) != 0
                    ):
                        raise SafetyAbort(
                            "sign-ID active gate changed from gate 0"
                        )
                    current_rpy = self.estimate.orientation.to_euler()
                    yaw_excursion = wrapped_yaw_excursion(current_rpy[2])
                    max_excursion = max(
                        max_excursion,
                        abs(current_rpy[0] - start_rpy[0]),
                        abs(current_rpy[1] - start_rpy[1]),
                        abs(yaw_excursion),
                    )
                    measured_yaw_rate = float(self.estimate.body_rates[2])
                    max_abs_measured_yaw_rate = max(
                        max_abs_measured_yaw_rate,
                        abs(measured_yaw_rate),
                    )
                    if max_excursion > SIGN_ID_MAX_ATTITUDE_EXCURSION_RAD:
                        raise SafetyAbort(
                            "sign-ID attitude excursion too large "
                            f"({max_excursion:.3f}rad)"
                        )
                    if (
                        abs(measured_yaw_rate)
                        > SIGN_ID_MAX_MEASURED_YAW_RATE_RAD_S
                    ):
                        raise SafetyAbort(
                            "sign-ID measured yaw rate exceeded "
                            f"{SIGN_ID_MAX_MEASURED_YAW_RATE_RAD_S:.2f}rad/s"
                        )

                    target = self.tracker.target
                    if (
                        target is None
                        or target.composite
                        or self.tracker.last_selection_mode != "primary"
                        or type(self._latest_detection_generation) is not int
                    ):
                        raise SafetyAbort(
                            "sign-ID yaw image target lost primary authority"
                        )
                    image_token = (
                        int(self._latest_detection_generation),
                        int(target.frame_id),
                        int(target.sim_time_ns),
                    )
                    if image_token != last_image_token:
                        if (
                            last_image_token is not None
                            and (
                                image_token[0] != last_image_token[0]
                                or image_token[1] <= last_image_token[1]
                                or image_token[2] <= last_image_token[2]
                            )
                        ):
                            raise SafetyAbort(
                                "sign-ID yaw image frame did not advance"
                            )
                        last_image_token = image_token
                        if (
                            active_segment in image_samples
                            and active_wire_release_s is not None
                            and float(target.received_monotonic_s)
                            >= active_wire_release_s
                            + SIGN_ID_RESPONSE_SETTLE_S
                        ):
                            assert active_segment is not None
                            image_samples[active_segment].append(
                                (
                                    float(target.received_monotonic_s),
                                    float(target.center_x),
                                )
                            )
                            self.recorder.emit(
                                "sign_id_yaw_image_frame",
                                segment=active_segment,
                                frame_id=target.frame_id,
                                sim_time_ns=target.sim_time_ns,
                                received_monotonic_s=(
                                    target.received_monotonic_s
                                ),
                                center_x=target.center_x,
                            )

                    if (
                        active_segment in responses
                        and active_wire_release_s is not None
                        and time.monotonic()
                        >= active_wire_release_s
                        + SIGN_ID_RESPONSE_SETTLE_S
                    ):
                        assert active_segment is not None
                        timestamp_us = int(self.estimate.timestamp_us)
                        if (
                            not responses[active_segment]
                            or timestamp_us
                            > responses[active_segment][-1][0]
                        ):
                            responses[active_segment].append(
                                (
                                    timestamp_us,
                                    float(self.estimate.body_rates[2]),
                                )
                            )

                    command = AttitudeRateCommand(
                        rates[0],
                        rates[1],
                        rates[2],
                        SIGN_ID_THRUST,
                    )
                    validate_command(command)
                    receipt = await self._send_flight_command(
                        command,
                        require_wire_receipt=True,
                    )
                    wire_yaw = validate_wire_yaw(receipt, command)
                    if active_segment != name:
                        assert receipt is not None
                        call_end_ns = receipt.get(
                            "call_end_monotonic_ns"
                        )
                        if type(call_end_ns) is not int or call_end_ns < 0:
                            raise SafetyAbort(
                                "sign-ID wire release timestamp is invalid"
                            )
                        active_segment = name
                        active_wire_release_s = (
                            call_end_ns / 1_000_000_000.0
                        )
                    if name in wire_yaw_rates:
                        wire_yaw_rates[name].append(wire_yaw)
                    elapsed = time.monotonic() - flight_start
                    self._record_tick(f"sign-id/{name}", elapsed, command)
                    next_tick = next_control_deadline(
                        next_tick,
                        time.monotonic(),
                    )
                    await asyncio.sleep(
                        max(0.0, next_tick - time.monotonic())
                    )

            sparse_responses = [
                name
                for name, values in responses.items()
                if len(values) < SIGN_ID_MIN_YAW_GYRO_SAMPLES
            ]
            if sparse_responses:
                raise SafetyAbort(
                    "sign-ID lacks required fresh body-rate samples for "
                    f"{sparse_responses}"
                )
            assert self.estimate is not None
            end_rpy = self.estimate.orientation.to_euler()
            final_yaw_excursion = wrapped_yaw_excursion(end_rpy[2])
            excursion = max(
                abs(end_rpy[0] - start_rpy[0]),
                abs(end_rpy[1] - start_rpy[1]),
                abs(final_yaw_excursion),
            )
            raw_means = {
                axis: (
                    statistics.fmean(
                        sample[1] for sample in values
                    )
                    if values
                    else 0.0
                )
                for axis, values in responses.items()
            }
            yaw_medians = {
                name: statistics.median(
                    sample[1] for sample in responses[name]
                )
                for name in (
                    "neutral-pre-yaw",
                    "yaw-positive",
                    "yaw-negative",
                )
            }
            baseline = {
                "yaw": yaw_medians["neutral-pre-yaw"],
            }
            means = {
                "yaw-positive": (
                    yaw_medians["yaw-positive"]
                    - yaw_medians["neutral-pre-yaw"]
                ),
                "yaw-negative": (
                    yaw_medians["yaw-negative"]
                    - yaw_medians["neutral-pre-yaw"]
                ),
            }
            if (
                abs(means["yaw-positive"])
                <= SIGN_ID_MIN_RESPONSE_RAD_S
                or abs(means["yaw-negative"])
                <= SIGN_ID_MIN_RESPONSE_RAD_S
                or means["yaw-positive"] * means["yaw-negative"] >= 0.0
            ):
                raise SafetyAbort(
                    "sign-ID yaw gyro response is inconclusive/wrong: "
                    f"{means}"
                )

            slopes = {
                name: image_slope(name)
                for name in image_segment_names
            }
            positive_effect = (
                slopes["yaw-positive"]
                - slopes["neutral-pre-yaw"]
            )
            negative_effect = (
                slopes["yaw-negative"]
                - slopes["neutral-pre-yaw"]
            )
            if (
                abs(positive_effect) < SIGN_ID_MIN_IMAGE_EFFECT_PX_S
                or abs(negative_effect) < SIGN_ID_MIN_IMAGE_EFFECT_PX_S
                or positive_effect * negative_effect >= 0.0
            ):
                raise SafetyAbort(
                    "sign-ID yaw image response is inconclusive/wrong: "
                    f"positive={positive_effect:.6f}px/s, "
                    f"negative={negative_effect:.6f}px/s"
                )

            gyro_gain = (
                means["yaw-positive"] - means["yaw-negative"]
            ) / (2.0 * SIGN_ID_RATE_RAD_S)
            positive_gyro_gain = (
                means["yaw-positive"] / SIGN_ID_RATE_RAD_S
            )
            negative_gyro_gain = (
                means["yaw-negative"] / -SIGN_ID_RATE_RAD_S
            )
            positive_image_gain = (
                positive_effect / SIGN_ID_RATE_RAD_S
            )
            negative_image_gain = (
                negative_effect / -SIGN_ID_RATE_RAD_S
            )
            image_gain = (
                positive_image_gain + negative_image_gain
            ) / 2.0
            gyro_gain_ratio = max(
                abs(positive_gyro_gain),
                abs(negative_gyro_gain),
            ) / min(
                abs(positive_gyro_gain),
                abs(negative_gyro_gain),
            )
            image_gain_ratio = max(
                abs(positive_image_gain),
                abs(negative_image_gain),
            ) / min(
                abs(positive_image_gain),
                abs(negative_image_gain),
            )
            if (
                gyro_gain_ratio > SIGN_ID_MAX_POLARITY_GAIN_RATIO
                or image_gain_ratio > SIGN_ID_MAX_POLARITY_GAIN_RATIO
            ):
                raise SafetyAbort(
                    "sign-ID yaw polarity gains are inconsistent: "
                    f"gyro_ratio={gyro_gain_ratio:.6f}, "
                    f"image_ratio={image_gain_ratio:.6f}"
                )
            controller_to_body_sign = (
                1 if gyro_gain > 0.0 else -1
            )
            controller_to_image_sign = (
                1 if image_gain > 0.0 else -1
            )
            pulse_summaries = {
                "positive": {
                    "command_yaw_rate_rad_s": SIGN_ID_RATE_RAD_S,
                    "wire_yaw_rate_rad_s": statistics.fmean(
                        wire_yaw_rates["yaw-positive"]
                    ),
                    "corrected_median_yaw_rate_rad_s": (
                        means["yaw-positive"]
                    ),
                    "gyro_rate_gain": positive_gyro_gain,
                    "fresh_image_frame_count": len(
                        image_samples["yaw-positive"]
                    ),
                    "image_start_x_px": image_samples[
                        "yaw-positive"
                    ][0][1],
                    "image_end_x_px": image_samples[
                        "yaw-positive"
                    ][-1][1],
                    "image_slope_px_s": slopes["yaw-positive"],
                    "neutral_image_slope_px_s": slopes[
                        "neutral-pre-yaw"
                    ],
                    "differential_image_effect_px_s": positive_effect,
                    "image_rate_gain_px_per_command_rad": (
                        positive_image_gain
                    ),
                },
                "negative": {
                    "command_yaw_rate_rad_s": -SIGN_ID_RATE_RAD_S,
                    "wire_yaw_rate_rad_s": statistics.fmean(
                        wire_yaw_rates["yaw-negative"]
                    ),
                    "corrected_median_yaw_rate_rad_s": (
                        means["yaw-negative"]
                    ),
                    "gyro_rate_gain": negative_gyro_gain,
                    "fresh_image_frame_count": len(
                        image_samples["yaw-negative"]
                    ),
                    "image_start_x_px": image_samples[
                        "yaw-negative"
                    ][0][1],
                    "image_end_x_px": image_samples[
                        "yaw-negative"
                    ][-1][1],
                    "image_slope_px_s": slopes["yaw-negative"],
                    "neutral_image_slope_px_s": slopes[
                        "neutral-pre-yaw"
                    ],
                    "differential_image_effect_px_s": negative_effect,
                    "image_rate_gain_px_per_command_rad": (
                        negative_image_gain
                    ),
                },
            }
            yaw_calibration = {
                "yaw_identified": True,
                "controller_to_body_sign": controller_to_body_sign,
                "controller_to_image_sign": controller_to_image_sign,
                "command_rate_abs_rad_s": SIGN_ID_RATE_RAD_S,
                "command_thrust": SIGN_ID_THRUST,
                "baseline_yaw_rate_rad_s": baseline["yaw"],
                "positive_local_yaw_baseline_rad_s": yaw_medians[
                    "neutral-pre-yaw"
                ],
                "negative_local_yaw_baseline_rad_s": yaw_medians[
                    "neutral-pre-yaw"
                ],
                "gyro_rate_gain": abs(gyro_gain),
                "signed_gyro_rate_gain": gyro_gain,
                "gyro_polarity_gain_ratio": gyro_gain_ratio,
                "image_rate_gain_px_per_command_rad": image_gain,
                "image_polarity_gain_ratio": image_gain_ratio,
                "neutral_image_slope_px_s": slopes[
                    "neutral-pre-yaw"
                ],
                "positive": pulse_summaries["positive"],
                "negative": pulse_summaries["negative"],
                "final_yaw_excursion_rad": final_yaw_excursion,
                "max_attitude_excursion_rad": max_excursion,
                "max_abs_measured_yaw_rate_rad_s": (
                    max_abs_measured_yaw_rate
                ),
            }
            self.recorder.emit(
                "sign_id_yaw_terminal",
                success=True,
                reason=None,
                **yaw_calibration,
            )
            return {
                "mean_responses_rad_s": means,
                "raw_mean_responses_rad_s": raw_means,
                "baseline_rates_rad_s": baseline,
                "final_attitude_excursion_rad": excursion,
                "max_attitude_excursion_rad": max_excursion,
                "yaw_calibration": yaw_calibration,
            }
        except BaseException as exc:
            try:
                self.recorder.emit(
                    "sign_id_yaw_terminal",
                    success=False,
                    reason=str(exc) or type(exc).__name__,
                    max_attitude_excursion_rad=max_excursion,
                    max_abs_measured_yaw_rate_rad_s=(
                        max_abs_measured_yaw_rate
                    ),
                )
            except BaseException as recorder_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "sign-ID yaw terminal evidence also failed: "
                        f"{recorder_exc}"
                    )
            raise

    def _check_calibration_envelope(
        self,
        context: StartContext,
    ) -> Tuple[float, float, float]:
        """Enforce the calibration-specific corridor on top of the watchdog."""

        race = self.adapter.race_status
        if race is None or int(race.active_gate_index) != 0:
            raise SafetyAbort("calibration active gate changed from gate 0")
        target = self.tracker.target
        if target is None or self.tracker.consecutive < 3:
            raise SafetyAbort("calibration target lacks three-frame confirmation")
        image_shape = getattr(self._latest_detection_image, "shape", None)
        if (
            image_shape is None
            or len(image_shape) < 2
            or int(image_shape[0]) != 360
            or int(image_shape[1]) != 640
        ):
            raise SafetyAbort("calibration decoded dimensions changed from 640x360")
        center_x = float(target.center_x)
        center_y = float(target.center_y)
        _x, _y, width, height = (float(value) for value in target.bbox)
        area = width * height
        if not 64.0 <= center_x <= 576.0 or not 36.0 <= center_y <= 324.0:
            raise SafetyAbort("calibration target left the closed safety corridor")
        if (
            width > 160.0
            or height > 160.0
            or area > 2.0 * float(context.initial_gate_area)
        ):
            raise SafetyAbort("calibration target exceeded the size safety limit")
        if self.estimate is None:
            raise SafetyAbort("calibration attitude estimate is unavailable")
        roll, pitch, _yaw = self.estimate.orientation.to_euler()
        roll_excursion = abs(float(roll) - float(context.spawn_roll_rad))
        pitch_excursion = abs(float(pitch) - float(context.spawn_pitch_rad))
        if (
            roll_excursion > CALIBRATION_MAX_ATTITUDE_EXCURSION_RAD
            or pitch_excursion > CALIBRATION_MAX_ATTITUDE_EXCURSION_RAD
        ):
            raise SafetyAbort(
                "calibration attitude excursion exceeded "
                f"{CALIBRATION_MAX_ATTITUDE_EXCURSION_RAD:.3f} rad"
            )
        return roll_excursion, pitch_excursion, area

    @staticmethod
    def _wait_for_calibration_release(deadline_ns: int) -> None:
        """Wait for a 50 Hz release without the coarse Windows asyncio timer.

        The default Windows event-loop timer advances in roughly 15.6 ms
        quanta on this host, so an ``asyncio.sleep`` for the remainder of a
        20 ms slot can wake a full slot late.  Sleep most of the interval with
        the high-resolution CPython waitable timer, then spin only the final
        2 ms.  ``time.sleep`` releases the GIL, so the MAVLink and vision
        receiver threads continue to run during the coarse portion.
        """

        if type(deadline_ns) is not int or deadline_ns < 0:
            raise ValueError("calibration release deadline must be nonnegative")
        spin_ns = 2_000_000
        while True:
            now_ns = time.perf_counter_ns()
            remaining_ns = deadline_ns - now_ns
            if remaining_ns <= 0:
                return
            if remaining_ns > spin_ns:
                time.sleep((remaining_ns - spin_ns) / 1_000_000_000.0)
                continue
            while time.perf_counter_ns() < deadline_ns:
                pass
            return

    async def _run_calibration_excite(
        self,
        context: StartContext,
    ) -> Dict[str, Any]:
        """Run the reviewed 4.9 s waveform without the task-local freeze ceremony.

        This is intentionally only an execution shortcut. It retains the same
        gate-zero corridor, attitude-excursion limit, zero yaw, exact 50 Hz
        half-open slots, per-tick watchdog, and fail-closed cleanup owned by
        ``run_powered_stage``. A missed slot aborts instead of being replayed.
        """

        from scripts import aigp_vq2_powered_attempt as calibration_contract

        plan = calibration_contract.fast_excitation_plan()
        period_ns = int(plan["control_period_ns"])
        if period_ns != int(CONTROL_PERIOD_S * 1_000_000_000):
            raise SafetyAbort("calibration plan is not exactly 50 Hz")
        if plan["stage"] != CALIBRATION_STAGE:
            raise SafetyAbort("calibration plan stage changed")
        plan_sha256 = calibration_contract.canonical_object_sha256(plan)
        flight_start_ns = time.perf_counter_ns()
        hard_deadline_ns = flight_start_ns + int(
            plan["powered_hard_expiry_offset_ns"]
        )
        sent = 0
        max_roll_excursion = 0.0
        max_pitch_excursion = 0.0
        max_target_area = float(context.initial_gate_area)
        self.recorder.emit(
            "calibration_plan_start",
            plan_id=plan["plan_id"],
            plan_sha256=plan_sha256,
            tick_count=plan["tick_count"],
            control_period_ns=period_ns,
        )
        for tick_index in range(int(plan["tick_count"])):
            nominal_release_ns = flight_start_ns + tick_index * period_ns
            slot_end_ns = nominal_release_ns + period_ns
            now_ns = time.perf_counter_ns()
            if now_ns < nominal_release_ns:
                self._wait_for_calibration_release(nominal_release_ns)
            now_ns = time.perf_counter_ns()
            if now_ns >= slot_end_ns or now_ns >= hard_deadline_ns:
                self.recorder.emit(
                    "calibration_slot_missed",
                    absolute_tick=tick_index,
                    observed_monotonic_ns=now_ns,
                )
                raise SafetyAbort(
                    f"calibration slot {tick_index} was missed; no catch-up send"
                )

            self._sample()
            # The frozen waveform uses the already-bounded sign-ID thrust and
            # can remain lightly loaded against the spawn pad.  Reuse the
            # sign-ID rule: only exact tiny id-1002/threat-1 contacts are
            # tolerated; every differently identified, larger, or higher-
            # threat collision remains terminal.
            self._watchdog(
                allow_benign_pad_contact=True,
                enforce_benign_pad_budget=False,
                benign_pad_max_impulse=0.02,
            )
            roll_excursion, pitch_excursion, target_area = (
                self._check_calibration_envelope(context)
            )
            max_roll_excursion = max(max_roll_excursion, roll_excursion)
            max_pitch_excursion = max(max_pitch_excursion, pitch_excursion)
            max_target_area = max(max_target_area, target_area)

            tick = calibration_contract.fast_excitation_tick(tick_index)
            command_value = tick["command"]
            command = AttitudeRateCommand(
                float(command_value["roll_rate_rad_s"]),
                float(command_value["pitch_rate_rad_s"]),
                float(command_value["yaw_rate_rad_s"]),
                float(command_value["thrust"]),
            )
            validate_command(command)

            # Keep fixed plan slots while spacing the adapter's exact wire-call
            # starts. If variable safety work finishes early, wait out only the
            # remaining pacing gap; the adapter rechecks both edges of the
            # window inside its send lock before any wire mutation.
            earliest_send_ns = nominal_release_ns
            if self._last_flight_command_started_ns is not None:
                earliest_send_ns = max(
                    earliest_send_ns,
                    self._last_flight_command_started_ns + period_ns,
                )
            checked_ns = time.perf_counter_ns()
            if checked_ns < earliest_send_ns:
                self._wait_for_calibration_release(earliest_send_ns)
                checked_ns = time.perf_counter_ns()
                if checked_ns >= slot_end_ns or checked_ns >= hard_deadline_ns:
                    self.recorder.emit(
                        "calibration_slot_missed",
                        absolute_tick=tick_index,
                        observed_monotonic_ns=checked_ns,
                    )
                    raise SafetyAbort(
                        f"calibration slot {tick_index} was missed while pacing"
                    )
            checked_ns = time.perf_counter_ns()
            if checked_ns >= slot_end_ns or checked_ns >= hard_deadline_ns:
                self.recorder.emit(
                    "calibration_slot_missed",
                    absolute_tick=tick_index,
                    observed_monotonic_ns=checked_ns,
                )
                raise SafetyAbort(
                    f"calibration slot {tick_index} expired before send"
                )
            previous_wire_start_ns = self._last_flight_command_started_ns
            receipt = await self._send_flight_command(
                command,
                require_wire_receipt=True,
                wire_start_not_before_ns=earliest_send_ns,
                wire_start_deadline_ns=min(slot_end_ns, hard_deadline_ns),
            )
            assert receipt is not None
            sent += 1
            call_start_ns = int(receipt["call_start_monotonic_ns"])
            call_end_ns = int(receipt["call_end_monotonic_ns"])
            self._record_tick(
                f"calibration-excite/{tick['segment_id']}",
                (call_start_ns - flight_start_ns) / 1_000_000_000.0,
                command,
            )
            if not (
                nominal_release_ns <= call_start_ns < slot_end_ns
                and call_start_ns < hard_deadline_ns
            ):
                self.recorder.emit(
                    "calibration_slot_missed",
                    absolute_tick=tick_index,
                    observed_monotonic_ns=call_start_ns,
                )
                raise SafetyAbort(
                    f"calibration slot {tick_index} wire dispatch started out of slot"
                )
            if (
                previous_wire_start_ns is not None
                and call_start_ns - previous_wire_start_ns < period_ns
            ):
                raise SafetyAbort("calibration wire dispatch exceeded 50 Hz")
            if call_end_ns >= hard_deadline_ns:
                raise SafetyAbort("calibration wire dispatch crossed the hard deadline")

        self.recorder.emit(
            "calibration_plan_complete",
            plan_id=plan["plan_id"],
            plan_sha256=plan_sha256,
            ticks_sent=sent,
        )
        return {
            "plan_id": plan["plan_id"],
            "plan_sha256": plan_sha256,
            "ticks_sent": sent,
            "ticks_expected": int(plan["tick_count"]),
            "max_roll_excursion_rad": max_roll_excursion,
            "max_pitch_excursion_rad": max_pitch_excursion,
            "max_target_area_px": max_target_area,
        }

    async def _run_hover(self, context: StartContext) -> Dict[str, Any]:
        assert self.estimate is not None
        flight_start = await self._wait_for_next_flight_command_slot()
        next_tick = flight_start
        max_abs_roll = 0.0
        max_abs_rate = 0.0
        while True:
            now = time.monotonic()
            elapsed = now - flight_start
            if elapsed >= 2.5:
                break
            self._sample()
            self._watchdog(
                allow_benign_pad_contact=elapsed < 0.35,
                enforce_benign_pad_budget=True,
            )
            assert self.estimate is not None
            blend = min(1.0, elapsed / 0.8)
            target_pitch = (1.0 - blend) * context.spawn_pitch_rad
            if elapsed < 0.15:
                thrust = 0.26
            elif elapsed < 1.0:
                # Spawn is pitched about 18deg nose-down.  The previous
                # 0.20->0.265 ramp never exceeded weight along world-up, so the
                # drone slid off the launch pad.  0.32 supplies positive climb
                # margin while the attitude loop levels it.
                thrust = 0.32
            else:
                thrust = min(0.29, 0.275 / max(0.95, math.cos(target_pitch)))
            command = attitude_rate_command(
                self.estimate,
                target_roll_rad=0.0,
                target_pitch_rad=target_pitch,
                thrust=thrust,
            )
            await self._send_flight_command(command)
            max_abs_roll = max(max_abs_roll, abs(self.estimate.roll))
            max_abs_rate = max(max_abs_rate, max(abs(v) for v in self.estimate.body_rates))
            self._record_tick("hover", elapsed, command)
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
        return {
            "max_abs_roll_deg": math.degrees(max_abs_roll),
            "max_abs_body_rate_rad_s": max_abs_rate,
            "final_rpy_rad": list(self.estimate.orientation.to_euler()),
        }

    def _visual_race_status_ref(self) -> AuthoritativeRaceStatusRef:
        received = getattr(
            self.adapter,
            "latest_received_race_status",
            None,
        )
        if received is None:
            raise SafetyAbort(
                "visual gate graph lacks exact received race-status provenance"
            )
        validate_integrity = getattr(received, "validate_integrity", None)
        if callable(validate_integrity):
            validate_integrity()
        ingress = received.ingress
        payload = received.race_status
        race = self.adapter.race_status
        if (
            race is None
            or int(payload.sim_boot_time_ms) != int(race.sim_boot_time_ms)
            or int(payload.active_gate_index) != int(race.active_gate_index)
            or int(payload.race_finish_time_ns)
            != int(race.race_finish_time_ns)
        ):
            raise SafetyAbort(
                "visual gate graph race payload is not the active exact ingress"
            )
        return AuthoritativeRaceStatusRef.live(
            session_id=self._visual_session_id,
            reset_epoch=self._visual_reset_epoch,
            race_generation=int(ingress.generation),
            race_status_sequence=int(ingress.sequence),
            race_status_boot_ms=int(payload.sim_boot_time_ms),
            active_gate_index=int(payload.active_gate_index),
            received_monotonic_ns=int(ingress.received_monotonic_ns),
            host_clock_id=str(ingress.host_clock_id),
            race_finished=bool(race.race_finished),
        )

    def _bind_initial_visual_gate(
        self,
        context: StartContext,
    ) -> GateGraphSnapshot:
        update = self._visual_latest_tracker_update
        if update is None:
            raise SafetyAbort("visual tracker has no fresh initial gate frames")
        expected_center = (
            2.0 * float(context.initial_gate_x) / 640.0 - 1.0,
            2.0 * float(context.initial_gate_y) / 360.0 - 1.0,
        )
        expected_area = float(context.initial_gate_area) / (640.0 * 360.0)
        candidates: List[VisualTrack] = []
        for track in update.visible_tracks:
            left, top, right, bottom = track.bbox_norm
            area = max(0.0, right - left) * max(0.0, bottom - top)
            area_ratio = area / max(expected_area, 1e-9)
            center_error = math.hypot(
                track.center_norm[0] - expected_center[0],
                track.center_norm[1] - expected_center[1],
            )
            if (
                not track.ambiguous
                and track.consecutive_frame_count
                >= VISUAL_SHADOW_REQUIRED_PRETRANSITION_FRAMES
                and track.confidence >= 0.20
                and track.association_confidence >= 0.10
                and center_error <= 0.22
                and 0.25 <= area_ratio <= 4.0
            ):
                candidates.append(track)
        if len(candidates) != 1:
            raise SafetyAbort(
                "initial visual current-gate association is "
                f"{'absent' if not candidates else 'ambiguous'} "
                f"(candidate_count={len(candidates)})"
            )
        race_ref = self._visual_race_status_ref()
        if race_ref.active_gate_index != 0 or race_ref.race_finished:
            raise SafetyAbort(
                "initial visual gate binding lacks authoritative gate 0"
            )
        snapshot = self.visual_gate_graph.bind_initial_current(
            self.visual_tracker,
            track_id=candidates[0].track_id,
            race_status=race_ref,
        )
        self._visual_latest_graph_snapshot = snapshot
        self.recorder.emit(
            "visual_initial_gate_bound",
            race_status=asdict(race_ref),
            current_track=self._visual_track_summary(candidates[0]),
            graph=self._visual_graph_summary(snapshot),
        )
        return snapshot

    def _visual_camera_token_at_race_credit(
        self,
        race_status: AuthoritativeRaceStatusRef,
    ) -> VisualCameraFrameToken:
        received_ns = race_status.received_monotonic_ns
        if received_ns is None:
            raise SafetyAbort("live race credit lacks host receive time")
        eligible: Dict[
            Tuple[str, int, int, int],
            Tuple[int, VisualCameraFrameToken],
        ] = {}
        for track in self.visual_tracker.tracks():
            for sample in track.history:
                token = sample.token
                identity = token.live_identity_tuple
                published = sample.publication_monotonic_ns
                if (
                    identity is not None
                    and published is not None
                    and int(published) <= int(received_ns)
                ):
                    eligible[identity] = (int(published), token)
        if not eligible:
            raise SafetyAbort(
                "no exact camera publication precedes authoritative race credit"
            )
        return max(eligible.values(), key=lambda item: item[0])[1]

    def _confirm_visual_transition(
        self,
        *,
        from_gate_index: int,
        to_gate_index: int,
    ) -> ConfirmedGateTransition:
        race_ref = self._visual_race_status_ref()
        if (
            race_ref.active_gate_index != to_gate_index
            or to_gate_index != from_gate_index + 1
            or race_ref.race_finished
        ):
            raise SafetyAbort(
                "visual graph transition lacks exact sequential race authority"
            )
        camera_token = self._visual_camera_token_at_race_credit(race_ref)
        try:
            transition = self.visual_gate_graph.confirm_transition(
                self.visual_tracker,
                race_status=race_ref,
                camera_token_at_credit=camera_token,
            )
        except GateGraphError as exc:
            raise SafetyAbort(
                f"visual gate promotion refused: {exc}"
            ) from exc
        if (
            transition.from_gate_index != from_gate_index
            or transition.to_gate_index != to_gate_index
            or len(transition.pretransition_frame_tokens)
            < VISUAL_SHADOW_REQUIRED_PRETRANSITION_FRAMES
            or transition.history_length_before_promotion
            != transition.history_length_after_promotion
        ):
            raise SafetyAbort(
                "visual gate promotion proof is incomplete or reset history"
            )
        self._visual_transition = transition
        self._visual_latest_graph_snapshot = (
            self.visual_gate_graph.latest_snapshot
        )
        self.recorder.emit(
            "visual_gate_transition_promoted",
            from_gate_index=transition.from_gate_index,
            to_gate_index=transition.to_gate_index,
            retired_track_id=transition.retired_track_id,
            promoted_track_id=transition.promoted_track_id,
            race_status=asdict(transition.race_status),
            camera_token_at_credit=list(
                transition.camera_token_at_credit.live_identity_tuple
                or transition.camera_token_at_credit.exact_tuple
            ),
            promoted_first_token=list(
                transition.promoted_first_token.live_identity_tuple
                or transition.promoted_first_token.exact_tuple
            ),
            pretransition_frame_tokens=[
                list(token.live_identity_tuple or token.exact_tuple)
                for token in transition.pretransition_frame_tokens
            ],
            history_length_before_promotion=(
                transition.history_length_before_promotion
            ),
            history_length_after_promotion=(
                transition.history_length_after_promotion
            ),
            graph=self._visual_graph_summary(
                self._visual_latest_graph_snapshot
            ),
        )
        return transition

    def _complete_gate0_pass(
        self,
        *,
        race: Any,
        pre_gate_race_boot_ms: int,
        flight_start_s: float,
        crossing_started_s: Optional[float],
        next_tick_s: float,
        max_gate_area: int,
        capture_transition: bool,
    ) -> Dict[str, Any]:
        """Build the sole authoritative gate-0 pass handoff."""

        post_gate_boot_ms = int(race.sim_boot_time_ms)
        if int(race.active_gate_index) != 1:
            raise SafetyAbort("gate-0 pass handoff did not contain gate index 1")
        if post_gate_boot_ms <= int(pre_gate_race_boot_ms):
            raise SafetyAbort(
                "gate-1 race status was not strictly newer than recorded gate 0"
            )
        if self.estimate is None:
            raise SafetyAbort("attitude estimate unavailable at gate-0 pass")
        snapshot = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if snapshot is None:
            raise SafetyAbort("camera unavailable at authoritative gate-0 pass")

        pass_confirmed_s = time.monotonic()
        paced_deadline = float(next_tick_s)
        if self._last_flight_command_sent_s is not None:
            paced_deadline = max(
                paced_deadline,
                self._last_flight_command_sent_s + CONTROL_PERIOD_S,
            )
        proof = GateTransitionProof(
            pre_gate_race_boot_ms=int(pre_gate_race_boot_ms),
            post_gate_race_boot_ms=post_gate_boot_ms,
            flight_started_monotonic_s=float(flight_start_s),
            crossing_started_monotonic_s=(
                float(crossing_started_s) if crossing_started_s is not None else None
            ),
            pass_confirmed_monotonic_s=pass_confirmed_s,
            next_control_deadline_s=paced_deadline,
            vision_generation=int(snapshot.generation),
            vision_frame_id=int(snapshot.frame_id),
            vision_sim_time_ns=int(snapshot.sim_time_ns),
            vision_received_monotonic_s=float(snapshot.received_monotonic_s),
            pass_rpy_rad=tuple(
                float(value) for value in self.estimate.orientation.to_euler()
            ),
        )
        self._gate0_transition_proof = proof
        visual_transition: Optional[ConfirmedGateTransition] = None
        if self._visual_tracking_enabled:
            visual_transition = self._confirm_visual_transition(
                from_gate_index=0,
                to_gate_index=1,
            )
        if capture_transition:
            self._defer_snapshot("gate1_race_credit")
        result = {
            "gate0_passed": True,
            "gate_transition_proved": True,
            "pre_gate_race_boot_ms": proof.pre_gate_race_boot_ms,
            "race_boot_ms": proof.post_gate_race_boot_ms,
            "last_gate_race_time": race.last_gate_race_time,
            "max_gate_area_px": int(max_gate_area),
            "crossing_confirmation_used": crossing_started_s is not None,
            "crossing_confirmation_elapsed_s": (
                pass_confirmed_s - crossing_started_s
                if crossing_started_s is not None
                else None
            ),
            "flight_elapsed_s": pass_confirmed_s - flight_start_s,
            "early_turn": (
                dict(self._gate0_early_turn_summary)
                if self._gate0_early_turn_summary is not None
                else {
                    "started": False,
                    "command_count": 0,
                    "max_abs_yaw_excursion_rad": 0.0,
                    "max_abs_measured_yaw_rate_rad_s": 0.0,
                }
            ),
            "visual_next_gate_blend": (
                None
                if self._visual_gate0_blend_summary is None
                else dict(self._visual_gate0_blend_summary)
            ),
            "visual_transition": (
                None
                if visual_transition is None
                else {
                    "retired_track_id": (
                        visual_transition.retired_track_id
                    ),
                    "promoted_track_id": (
                        visual_transition.promoted_track_id
                    ),
                    "pretransition_frame_count": len(
                        visual_transition.pretransition_frame_tokens
                    ),
                    "history_length_before_promotion": (
                        visual_transition.history_length_before_promotion
                    ),
                    "history_length_after_promotion": (
                        visual_transition.history_length_after_promotion
                    ),
                }
            ),
        }
        self.recorder.emit("gate0_pass_proved", **result)
        return result

    async def _run_gate0(
        self,
        context: StartContext,
        *,
        capture_transition: bool = False,
        exit_pitch_rad: float = 0.0,
        minimum_thrust: float = 0.21,
        boost_until_s: Optional[float] = None,
        observe_course_line: bool = False,
        course_line_preturn: bool = False,
        course_line_exit_counterroll_enabled: bool = False,
        crossing_hold_thrust: float = 0.0,
        visual_next_gate_blend: bool = False,
    ) -> Dict[str, Any]:
        controller = self.controller_config
        phase_timing = controller.phase_timing
        turn_cue = controller.turn_cue
        roll_control = controller.roll_control
        yaw_control = controller.yaw_control
        forward_braking = controller.forward_braking
        if type(visual_next_gate_blend) is not bool:
            raise ValueError("gate-0 visual next-gate blend flag must be bool")
        visual_lifecycle = self.visual_config.lifecycle
        if boost_until_s is None:
            boost_until_s = (
                visual_lifecycle.launch_boost_duration_s
                if visual_next_gate_blend
                else phase_timing.gate0_boost_until_s
            )
        visual_pitch_blend_s = (
            visual_lifecycle.launch_pitch_blend_s
            if visual_next_gate_blend
            else phase_timing.gate0_pitch_blend_s
        )
        visual_launch_boost_thrust = (
            float(visual_lifecycle.launch_boost_thrust)
            if visual_next_gate_blend
            else 0.32
        )
        gate0_target_pitch_rad(
            context.spawn_pitch_rad,
            exit_pitch_rad,
            0.0,
            blend_duration_s=visual_pitch_blend_s,
        )
        if (
            type(minimum_thrust) not in {int, float}
            or not math.isfinite(float(minimum_thrust))
            or not 0.21 <= float(minimum_thrust) <= 0.32
        ):
            raise ValueError("gate-0 minimum thrust is outside the validated envelope")
        if (
            type(boost_until_s) not in {int, float}
            or not math.isfinite(float(boost_until_s))
            or not 0.45 <= float(boost_until_s) <= 1.0
        ):
            raise ValueError("gate-0 boost duration is outside the validated envelope")
        if (
            visual_next_gate_blend
            and float(boost_until_s)
            != float(visual_lifecycle.launch_boost_duration_s)
        ):
            raise ValueError(
                "visual gate-0 launch duration must match hashed lifecycle "
                "configuration"
            )
        if type(observe_course_line) is not bool:
            raise ValueError("gate-0 course-line observation flag must be bool")
        if type(course_line_preturn) is not bool:
            raise ValueError("gate-0 course-line preturn flag must be bool")
        if type(course_line_exit_counterroll_enabled) is not bool:
            raise ValueError("gate-0 course-line exit-counterroll flag must be bool")
        if course_line_exit_counterroll_enabled and not course_line_preturn:
            raise ValueError("gate-0 course-line exit counter-roll requires preturn")
        if visual_next_gate_blend and (
            observe_course_line
            or course_line_preturn
            or course_line_exit_counterroll_enabled
        ):
            raise ValueError(
                "gate-0 visual next-gate blend is mutually exclusive with "
                "retired course-line authority"
            )
        if (
            type(crossing_hold_thrust) not in {int, float}
            or not math.isfinite(float(crossing_hold_thrust))
            or float(crossing_hold_thrust)
            not in {0.0, GATE1_RECENTER_TRANSITION_THRUST}
        ):
            raise ValueError(
                "gate-0 crossing hold thrust must be exact zero or the "
                "Gate-1 continuation value"
            )
        if (
            visual_next_gate_blend
            and float(crossing_hold_thrust) != 0.0
        ):
            raise ValueError(
                "gate-0 visual next-gate blend requires exact-zero "
                "crossing hold thrust"
            )

        flight_start = await self._wait_for_next_flight_command_slot()
        self._gate0_early_turn_summary = None
        self._visual_gate0_blend_summary = None
        next_tick = flight_start
        max_gate_area = context.initial_gate_area
        last_target_frame: Optional[int] = None
        last_control_y: Optional[float] = None
        last_target_time: Optional[float] = None
        control_y_rate = 0.0
        crossing_armed = False
        crossing_started_s: Optional[float] = None
        crossing_race_boot_ms: Optional[int] = None
        last_gate0_race_boot_ms: Optional[int] = int(context.go_boot_ms)
        filtered_course_turn = 0.0
        course_turn_streak = 0
        last_course_turn_s: Optional[float] = None
        proved_course_turn_score: Optional[float] = None
        course_line_exit_started = False
        early_turn_started_s: Optional[float] = None
        early_turn_command_count = 0
        visual_approach: Optional[RollingVisualApproachServo] = None
        latest_visual_proposal: Optional[VisualApproachProposal] = None
        last_visual_approach_token: Optional[VisualCameraFrameToken] = None
        visual_yaw_reference_rad: Optional[float] = None
        visual_blend_withdrawn = False
        if visual_next_gate_blend:
            bound = self.visual_gate_graph.latest_snapshot
            if (
                bound is None
                or bound.current_track_id is None
                or bound.current_gate_index != 0
                or not bound.authority_usable
            ):
                raise SafetyAbort(
                    "gate-0 visual blend lacks a bound authoritative current gate"
                )
            visual_approach = RollingVisualApproachServo(
                bound.current_track_id,
                0,
                self.visual_config.servo,
                next_gate_blend=(
                    self.visual_config.lifecycle.next_gate_blend_max
                ),
            )
            self._visual_gate0_blend_summary = {
                "enabled": True,
                "started": False,
                "current_track_id": bound.current_track_id,
                "blended_next_track_id": None,
                "observed_next_track_ids": [],
                "fresh_blend_frame_count": 0,
                "command_count": 0,
                "withdrawn_before_confirmation": False,
                "withdrawal_reason": None,
                "yaw_reference_rad": None,
                "max_abs_yaw_excursion_rad": 0.0,
                "max_abs_measured_yaw_rate_rad_s": 0.0,
                "latest_horizontal_error": None,
                "latest_vertical_error_image_down": None,
                "latest_scale_rate_s": None,
                "min_command_thrust": None,
                "max_command_thrust": None,
                "launch_collective_hold_s": float(boost_until_s),
                "launch_boost_thrust": visual_launch_boost_thrust,
                "command_axis_authority": {
                    "yaw": "visual_next_track_blend",
                    "pitch": "bounded_visual_brake",
                    "roll_collective": "proved_gate0_bootstrap",
                },
            }
        while True:
            now = time.monotonic()
            elapsed = now - flight_start
            if elapsed >= GATE0_FLIGHT_TIMEOUT_S:
                raise SafetyAbort("gate-0 wall-time limit reached")
            self._sample()
            race = self.adapter.race_status
            assert race is not None and self.estimate is not None
            if int(race.active_gate_index) == 0:
                last_gate0_race_boot_ms = int(race.sim_boot_time_ms)
            target = self.tracker.target
            assert target is not None
            max_gate_area = max(max_gate_area, target.bbox_area)
            if elapsed > 3.5 and max_gate_area < 1.25 * context.initial_gate_area:
                raise SafetyAbort("no visual approach progress toward gate 0")

            target_pitch = gate0_target_pitch_rad(
                context.spawn_pitch_rad,
                exit_pitch_rad,
                elapsed,
                blend_duration_s=visual_pitch_blend_s,
            )
            normalized_x = (target.center_x - 320.0) / 320.0
            target_roll = gate0_centering_roll_target(
                normalized_x,
                gain=roll_control.gate0_centering_gain,
                cap_rad=roll_control.gate0_target_cap_rad,
            )

            control_y = gate_control_center_y_px(
                target,
                previous_center_y=last_control_y,
            )
            if (
                not crossing_armed
                and target.age_s(now) <= CROSSING_TARGET_LOSS_S
                and self.tracker.consecutive >= 3
                and race.active_gate_index == 0
                and is_close_gate_crossing_candidate(
                    target,
                    initial_gate_area=context.initial_gate_area,
                    control_y=control_y,
                )
            ):
                crossing_armed = True
                if capture_transition:
                    self._vision_diagnostic_logging = True
                self.recorder.emit(
                    "crossing_candidate_armed",
                    elapsed_s=elapsed,
                    race_boot_ms=race.sim_boot_time_ms,
                    target=asdict(target),
                    control_y=control_y,
                )

            if visual_approach is not None:
                assert self._visual_gate0_blend_summary is not None
                if crossing_armed and not visual_blend_withdrawn:
                    visual_blend_withdrawn = True
                    latest_visual_proposal = None
                    self._visual_gate0_blend_summary.update(
                        {
                            "withdrawn_before_confirmation": True,
                            "withdrawal_reason": "crossing_candidate_armed",
                            "withdrawn_elapsed_s": elapsed,
                        }
                    )
                    self.recorder.emit(
                        "visual_next_gate_blend_withdrawn",
                        elapsed_s=elapsed,
                        reason="crossing_candidate_armed",
                        current_track_id=(
                            self._visual_gate0_blend_summary[
                                "current_track_id"
                            ]
                        ),
                        blended_next_track_id=(
                            self._visual_gate0_blend_summary[
                                "blended_next_track_id"
                            ]
                        ),
                    )
                elif not visual_blend_withdrawn:
                    graph = self.visual_gate_graph.latest_snapshot
                    if graph is None:
                        raise SafetyAbort(
                            "gate-0 visual blend lost its rolling graph"
                        )
                    if graph.latest_camera_token != last_visual_approach_token:
                        _roll, _pitch, measured_yaw = (
                            self.estimate.orientation.to_euler()
                        )
                        visual_yaw_excursion = (
                            0.0
                            if visual_yaw_reference_rad is None
                            else math.atan2(
                                math.sin(
                                    float(measured_yaw)
                                    - visual_yaw_reference_rad
                                ),
                                math.cos(
                                    float(measured_yaw)
                                    - visual_yaw_reference_rad
                                ),
                            )
                        )
                        try:
                            proposal = visual_approach.observe(
                                graph,
                                self.visual_tracker,
                                now_monotonic_s=(
                                    time.perf_counter_ns()
                                    / 1_000_000_000.0
                                ),
                                segment_elapsed_s=elapsed,
                                segment_yaw_excursion_rad=(
                                    visual_yaw_excursion
                                ),
                            )
                        except (
                            VisualApproachCurrentGeometryUnavailable,
                            VisualApproachPassageSafetyUnavailable,
                        ) as exc:
                            # The optional pre-pass blend owns no crossing
                            # authority.  Withdraw it permanently when current
                            # geometry becomes censored or a latched blend
                            # leaves its immutable passage corridor, then
                            # return to the proved Gate-0 bootstrap controller.
                            # All identity/provenance refusals remain fatal.
                            withdrawal_reason = (
                                "current_aperture_geometry_unavailable"
                                if isinstance(
                                    exc,
                                    VisualApproachCurrentGeometryUnavailable,
                                )
                                else "passage_safety_corridor_unavailable"
                            )
                            visual_blend_withdrawn = True
                            latest_visual_proposal = None
                            self._visual_gate0_blend_summary.update(
                                {
                                    "withdrawn_before_confirmation": True,
                                    "withdrawal_reason": withdrawal_reason,
                                    "withdrawn_elapsed_s": elapsed,
                                }
                            )
                            self.recorder.emit(
                                "visual_next_gate_blend_withdrawn",
                                elapsed_s=elapsed,
                                reason=withdrawal_reason,
                                refusal=str(exc),
                                current_track_id=(
                                    self._visual_gate0_blend_summary[
                                        "current_track_id"
                                    ]
                                ),
                                blended_next_track_id=(
                                    self._visual_gate0_blend_summary[
                                        "blended_next_track_id"
                                    ]
                                ),
                            )
                        except VisualApproachRefusal as exc:
                            raise SafetyAbort(
                                f"gate-0 visual blend refused: {exc}"
                            ) from exc
                        else:
                            last_visual_approach_token = (
                                graph.latest_camera_token
                            )
                            observed_ids = list(
                                self._visual_gate0_blend_summary[
                                    "observed_next_track_ids"
                                ]
                            )
                            for track_id in proposal.candidate_track_ids:
                                if track_id not in observed_ids:
                                    observed_ids.append(track_id)
                            self._visual_gate0_blend_summary[
                                "observed_next_track_ids"
                            ] = observed_ids
                            if proposal.servo_output.next_gate_blend > 0.0:
                                if visual_yaw_reference_rad is None:
                                    visual_yaw_reference_rad = float(
                                        measured_yaw
                                    )
                                    self._visual_gate0_blend_summary.update(
                                        {
                                            "started": True,
                                            "started_elapsed_s": elapsed,
                                            "yaw_reference_rad": (
                                                visual_yaw_reference_rad
                                            ),
                                        }
                                    )
                                latest_visual_proposal = proposal
                                self._visual_gate0_blend_summary.update(
                                    {
                                        "blended_next_track_id": (
                                            proposal.latched_next_track_id
                                        ),
                                        "fresh_blend_frame_count": (
                                            int(
                                                self._visual_gate0_blend_summary[
                                                    "fresh_blend_frame_count"
                                                ]
                                            )
                                            + 1
                                        ),
                                        "latest_horizontal_error": (
                                            proposal.servo_output
                                            .effective_horizontal_error
                                        ),
                                        "latest_vertical_error_image_down": (
                                            proposal.servo_output
                                            .effective_vertical_error_image_down
                                        ),
                                        "latest_scale_rate_s": (
                                            proposal.current_target
                                            .log_scale_rate_s
                                        ),
                                    }
                                )
                                self.recorder.emit(
                                    "visual_next_gate_blend_frame",
                                    elapsed_s=elapsed,
                                    current_target=asdict(
                                        proposal.current_target
                                    ),
                                    next_target=(
                                        None
                                        if proposal.next_target is None
                                        else asdict(proposal.next_target)
                                    ),
                                    servo=asdict(proposal.servo_output),
                                    candidate_track_ids=list(
                                        proposal.candidate_track_ids
                                    ),
                                    relationship_basis=(
                                        None
                                        if proposal.relationship_basis is None
                                        else proposal.relationship_basis.value
                                    ),
                                )
                            else:
                                latest_visual_proposal = None

            crossing_confirming = bool(
                crossing_started_s is not None
                or (
                    crossing_armed
                    and target.age_s(now) > CROSSING_TARGET_LOSS_S
                )
            )
            self._watchdog(
                require_target=not (
                    crossing_confirming or race.active_gate_index == 1
                ),
                allow_benign_pad_contact=elapsed < 0.35,
                enforce_benign_pad_budget=True,
            )
            if race.active_gate_index not in (0, 1):
                raise SafetyAbort(f"unexpected gate-index jump to {race.active_gate_index}")
            if self._gate1_yaw_reference_rad is not None:
                self._gate1_yaw_envelope_state(phase="Gate-0 approach")
            if visual_yaw_reference_rad is not None:
                # Once the optional blend establishes its segment reference,
                # its hard excursion/rate/momentum envelope remains latched
                # through current-only frames, geometry withdrawal, exact-zero
                # crossing confirmation, and the authoritative transition.
                # Proposal availability controls command authority, not safety
                # observation.
                _roll, _pitch, measured_visual_yaw = (
                    self.estimate.orientation.to_euler()
                )
                _zero_rate, latched_visual_yaw_excursion = (
                    visual_alignment_yaw_rate(
                        requested_rate_rad_s=0.0,
                        measured_yaw_rad=float(measured_visual_yaw),
                        reference_yaw_rad=visual_yaw_reference_rad,
                        measured_yaw_rate_rad_s=float(
                            self.estimate.body_rates[2]
                        ),
                        horizontal_error_norm=normalized_x,
                        horizontal_corridor_norm=(
                            self.visual_config.servo.horizontal_corridor
                        ),
                    )
                )
                assert self._visual_gate0_blend_summary is not None
                self._visual_gate0_blend_summary.update(
                    {
                        "max_abs_yaw_excursion_rad": max(
                            float(
                                self._visual_gate0_blend_summary[
                                    "max_abs_yaw_excursion_rad"
                                ]
                            ),
                            abs(latched_visual_yaw_excursion),
                        ),
                        "max_abs_measured_yaw_rate_rad_s": max(
                            float(
                                self._visual_gate0_blend_summary[
                                    "max_abs_measured_yaw_rate_rad_s"
                                ]
                            ),
                            abs(float(self.estimate.body_rates[2])),
                        ),
                    }
                )
            if not crossing_confirming and race.active_gate_index == 1:
                if last_gate0_race_boot_ms is None:
                    raise SafetyAbort("gate 1 appeared without a recorded gate-0 packet")
                return self._complete_gate0_pass(
                    race=race,
                    pre_gate_race_boot_ms=last_gate0_race_boot_ms,
                    flight_start_s=flight_start,
                    crossing_started_s=crossing_started_s,
                    next_tick_s=next_tick,
                    max_gate_area=max_gate_area,
                    capture_transition=capture_transition,
                )

            if crossing_confirming:
                if crossing_started_s is None:
                    crossing_started_s = now
                    if last_gate0_race_boot_ms is None:
                        raise SafetyAbort(
                            "crossing confirmation lacks a recorded gate-0 packet"
                        )
                    crossing_race_boot_ms = last_gate0_race_boot_ms
                    if capture_transition:
                        self._vision_diagnostic_logging = True
                        self._defer_snapshot("gate0_visual_loss")
                    self.recorder.emit(
                        "crossing_confirmation_started",
                        elapsed_s=elapsed,
                        baseline_race_boot_ms=crossing_race_boot_ms,
                        target_age_s=target.age_s(now),
                        hold_thrust=float(crossing_hold_thrust),
                    )
                assert crossing_race_boot_ms is not None
                decision = crossing_status_decision(
                    baseline_race_boot_ms=crossing_race_boot_ms,
                    current_race_boot_ms=int(race.sim_boot_time_ms),
                    active_gate_index=int(race.active_gate_index),
                    elapsed_s=now - crossing_started_s,
                )
                if decision != "waiting":
                    self.recorder.emit(
                        "crossing_status_decision",
                        decision=decision,
                        baseline_race_boot_ms=crossing_race_boot_ms,
                        current_race_boot_ms=race.sim_boot_time_ms,
                        gate_index=race.active_gate_index,
                    )
                    if decision == "passed":
                        return self._complete_gate0_pass(
                            race=race,
                            pre_gate_race_boot_ms=crossing_race_boot_ms,
                            flight_start_s=flight_start,
                            crossing_started_s=crossing_started_s,
                            next_tick_s=next_tick,
                            max_gate_area=max_gate_area,
                            capture_transition=capture_transition,
                        )
                    raise SafetyAbort(f"gate-0 crossing {decision.replace('_', ' ')}")
                command = AttitudeRateCommand(
                    0.0,
                    0.0,
                    0.0,
                    float(crossing_hold_thrust),
                )
                await self._send_flight_command(command)
                self._record_tick("gate0/confirm", elapsed, command)
                next_tick = next_control_deadline(next_tick, time.monotonic())
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            if target.frame_id != last_target_frame:
                if last_control_y is not None and last_target_time is not None:
                    dt_target = target.received_monotonic_s - last_target_time
                    if dt_target > 1e-3:
                        raw_rate = (control_y - last_control_y) / dt_target
                        raw_rate = max(-300.0, min(300.0, raw_rate))
                        control_y_rate = 0.65 * control_y_rate + 0.35 * raw_rate
                last_target_frame = target.frame_id
                last_control_y = control_y
                last_target_time = target.received_monotonic_s
                if (
                    (observe_course_line or course_line_preturn)
                    and target.bbox_area
                    >= turn_cue.min_gate_area_scale
                    * context.initial_gate_area
                ):
                    line = cyan_course_line_observation(
                        self._latest_detection_image,
                    )
                    if (
                        line is not None
                        and abs(line.turn_score) >= turn_cue.min_abs_score
                    ):
                        if (
                            course_turn_streak == 0
                            or filtered_course_turn * line.turn_score > 0.0
                        ):
                            filtered_course_turn = (
                                line.turn_score
                                if course_turn_streak == 0
                                else 0.65 * filtered_course_turn
                                + 0.35 * line.turn_score
                            )
                            course_turn_streak += 1
                        else:
                            filtered_course_turn = line.turn_score
                            course_turn_streak = 1
                        last_course_turn_s = elapsed
                    self.recorder.emit(
                        "course_line_observation",
                        frame_id=target.frame_id,
                        elapsed_s=elapsed,
                        gate_area_px=target.bbox_area,
                        observation=(asdict(line) if line is not None else None),
                        filtered_turn_score=filtered_course_turn,
                        consistent_frame_count=course_turn_streak,
                    )
            if (
                last_course_turn_s is None
                or elapsed - last_course_turn_s
                > COURSE_LINE_PRETURN_MAX_AGE_S
            ):
                course_turn_streak = 0
                filtered_course_turn = 0.0
            stable_course_line = (
                course_turn_streak >= COURSE_LINE_PRETURN_REQUIRED_FRAMES
            )
            if stable_course_line and early_turn_started_s is None:
                early_turn_started_s = now
                self._gate0_early_turn_summary = {
                    "started": True,
                    "started_elapsed_s": elapsed,
                    "start_gate_area_scale": (
                        float(target.bbox_area)
                        / float(context.initial_gate_area)
                    ),
                    "turn_score": filtered_course_turn,
                    "command_count": 0,
                    "max_abs_yaw_excursion_rad": 0.0,
                    "max_abs_measured_yaw_rate_rad_s": 0.0,
                }
            early_turn_active = bool(
                stable_course_line
                and early_turn_started_s is not None
                and now - early_turn_started_s
                < phase_timing.gate0_yaw_brake_duration_s
            )
            if (
                course_line_exit_counterroll_enabled
                and stable_course_line
                and target.bbox_area
                < turn_cue.exit_counterroll_onset_area_scale
                * context.initial_gate_area
            ):
                proved_course_turn_score = filtered_course_turn
            if (
                course_line_exit_counterroll_enabled
                and not course_line_exit_started
                and proved_course_turn_score is not None
                and stable_course_line
                and filtered_course_turn * proved_course_turn_score > 0.0
                and target.bbox_area
                >= turn_cue.exit_counterroll_onset_area_scale
                * context.initial_gate_area
            ):
                course_line_exit_started = True
            if course_line_exit_started:
                if (
                    proved_course_turn_score is not None
                    and proved_course_turn_score * normalized_x >= 0.0
                ):
                    target_roll = course_line_exit_counterroll(
                        proved_course_turn_score,
                        cap_rad=turn_cue.exit_counterroll_cap_rad,
                        min_abs_score=turn_cue.min_abs_score,
                    )
                    self.recorder.emit(
                        "course_line_exit_counterroll_applied",
                        frame_id=target.frame_id,
                        elapsed_s=elapsed,
                        gate_area_px=target.bbox_area,
                        proved_turn_score=proved_course_turn_score,
                        filtered_turn_score=filtered_course_turn,
                        consistent_frame_count=course_turn_streak,
                        normalized_x=normalized_x,
                        target_roll_rad=target_roll,
                    )
            elif (
                course_line_preturn
                and stable_course_line
                and target.bbox_area
                < turn_cue.preturn_taper_area_scale
                * context.initial_gate_area
            ):
                preturn_bias = course_line_preturn_roll(
                    filtered_course_turn,
                    gain=turn_cue.preturn_gain,
                    cap_rad=turn_cue.preturn_roll_cap_rad,
                    min_abs_score=turn_cue.min_abs_score,
                )
                target_roll = max(
                    -turn_cue.preturn_roll_cap_rad,
                    min(
                        turn_cue.preturn_roll_cap_rad,
                        target_roll + preturn_bias,
                    ),
                )
                self.recorder.emit(
                    "course_line_preturn_applied",
                    frame_id=target.frame_id,
                    elapsed_s=elapsed,
                    gate_area_px=target.bbox_area,
                    filtered_turn_score=filtered_course_turn,
                    consistent_frame_count=course_turn_streak,
                    roll_bias_rad=preturn_bias,
                    target_roll_rad=target_roll,
                )
            if early_turn_active:
                target_pitch = max(
                    target_pitch,
                    forward_braking.gate0_turn_pitch_rad,
                )
            if elapsed < 0.15:
                thrust = 0.26
            elif elapsed < float(boost_until_s):
                thrust = visual_launch_boost_thrust
            else:
                # Steer the camera ray through the opening center.  Image-rate
                # damping brakes climb before positional error grows near the
                # rapidly approaching gate.
                thrust = max(
                    float(minimum_thrust),
                    gate_vertical_thrust(control_y, control_y_rate),
                )
            if early_turn_active:
                thrust = min(
                    thrust,
                    forward_braking.gate0_turn_thrust_cap,
                )
            visual_blend_active = latest_visual_proposal is not None
            if visual_blend_active:
                assert latest_visual_proposal is not None
                visual_age_s = (
                    time.perf_counter_ns() / 1_000_000_000.0
                    - latest_visual_proposal.current_target.received_monotonic_s
                )
                if (
                    visual_age_s < -1e-6
                    or visual_age_s > MAX_VISUAL_OBSERVATION_AGE_S
                    or latest_visual_proposal.next_target is None
                    or latest_visual_proposal.servo_output.next_gate_blend
                    <= 0.0
                    or latest_visual_proposal.servo_output.advance_enabled
                ):
                    raise SafetyAbort(
                        "gate-0 visual blend proposal lost fresh no-advance "
                        "authority"
                    )
                # The stable next identity supplies bounded heading and
                # nonnegative braking-pitch authority.  Preserve the proved
                # current-aperture roll and vertical collective: exact live
                # evidence showed that replacing collective .319-.32 with
                # .244-.262 exhausted Gate-1's top-edge margin, while the
                # braking pitch reduced entry closure relative to the matched
                # bootstrap trace.
                target_pitch = max(
                    target_pitch,
                    min(
                        VISUAL_ALIGN_MAX_PITCH_RAD,
                        latest_visual_proposal.servo_output.target_pitch_rad,
                    ),
                )

            # At close range the uncorrected contour center becomes unsafe if
            # the lower gate edge is clipped.  Abort before impact when the
            # square-inferred center is plainly outside the opening corridor.
            if (
                target.bbox_area >= 8 * context.initial_gate_area
                and abs(control_y - 180.0) > 75.0
            ):
                raise SafetyAbort(
                    f"gate-0 close approach outside vertical corridor ({control_y:.1f}px)"
                )
            command = attitude_rate_command(
                self.estimate,
                target_roll_rad=target_roll,
                target_pitch_rad=target_pitch,
                thrust=thrust,
            )
            if visual_blend_active:
                command = AttitudeRateCommand(
                    roll_rate=command.roll_rate,
                    pitch_rate=max(
                        -VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S,
                        min(
                            VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S,
                            command.pitch_rate,
                        ),
                    ),
                    yaw_rate=command.yaw_rate,
                    thrust=command.thrust,
                )
            local_yaw_rate = 0.0
            yaw_excursion = 0.0
            yaw_soft_stopped = False
            if visual_blend_active:
                assert latest_visual_proposal is not None
                assert visual_yaw_reference_rad is not None
                _roll, _pitch, yaw = (
                    self.estimate.orientation.to_euler()
                )
                local_yaw_rate, yaw_excursion = (
                    visual_alignment_yaw_rate(
                        requested_rate_rad_s=(
                            latest_visual_proposal.servo_output
                            .yaw_rate_rad_s
                        ),
                        measured_yaw_rad=float(yaw),
                        reference_yaw_rad=visual_yaw_reference_rad,
                        measured_yaw_rate_rad_s=float(
                            self.estimate.body_rates[2]
                        ),
                        horizontal_error_norm=(
                            latest_visual_proposal.servo_output
                            .effective_horizontal_error
                        ),
                        horizontal_corridor_norm=(
                            self.visual_config.servo.horizontal_corridor
                        ),
                    )
                )
                yaw_soft_stopped = bool(
                    local_yaw_rate == 0.0
                    and abs(yaw_excursion)
                    >= VISUAL_ALIGN_YAW_SOFT_STOP_RAD
                )
            elif (
                early_turn_active
                and yaw_control.gate0_command_rate_cap_rad_s > 0.0
            ):
                if self._gate1_yaw_reference_rad is None:
                    _roll, _pitch, yaw = (
                        self.estimate.orientation.to_euler()
                    )
                    self._gate1_yaw_reference_rad = float(yaw)
                yaw_excursion, yaw_soft_stopped = (
                    self._gate1_yaw_envelope_state(
                        phase="Gate-0 early turn",
                    )
                )
                if not yaw_soft_stopped:
                    local_yaw_rate = course_line_turn_yaw_rate(
                        filtered_course_turn,
                        gain=yaw_control.gate0_turn_score_gain,
                        cap_rad_s=(
                            yaw_control.gate0_command_rate_cap_rad_s
                        ),
                        min_abs_score=turn_cue.min_abs_score,
                    )
            command = AttitudeRateCommand(
                roll_rate=command.roll_rate,
                pitch_rate=command.pitch_rate,
                yaw_rate=local_yaw_rate,
                thrust=command.thrust,
            )
            validate_command(command)
            if (
                abs(command.yaw_rate) > SIGN_ID_RATE_RAD_S
                or (
                    not (early_turn_active or visual_blend_active)
                    and command.yaw_rate != 0.0
                )
            ):
                raise SafetyAbort(
                    "Gate-0 bounded yaw escaped its fixed envelope"
                )
            if visual_blend_active:
                assert self._visual_gate0_blend_summary is not None
                command_count = (
                    int(
                        self._visual_gate0_blend_summary[
                            "command_count"
                        ]
                    )
                    + 1
                )
                self._visual_gate0_blend_summary.update(
                    {
                        "command_count": command_count,
                        "last_command_yaw_rate_rad_s": local_yaw_rate,
                        "last_target_pitch_rad": target_pitch,
                        "last_yaw_excursion_rad": yaw_excursion,
                        "yaw_soft_stopped": yaw_soft_stopped,
                        "max_abs_yaw_excursion_rad": max(
                            float(
                                self._visual_gate0_blend_summary[
                                    "max_abs_yaw_excursion_rad"
                                ]
                            ),
                            abs(yaw_excursion),
                        ),
                        "max_abs_measured_yaw_rate_rad_s": max(
                            float(
                                self._visual_gate0_blend_summary[
                                    "max_abs_measured_yaw_rate_rad_s"
                                ]
                            ),
                            abs(float(self.estimate.body_rates[2])),
                        ),
                        "min_command_thrust": (
                            command.thrust
                            if self._visual_gate0_blend_summary[
                                "min_command_thrust"
                            ]
                            is None
                            else min(
                                float(
                                    self._visual_gate0_blend_summary[
                                        "min_command_thrust"
                                    ]
                                ),
                                command.thrust,
                            )
                        ),
                        "max_command_thrust": (
                            command.thrust
                            if self._visual_gate0_blend_summary[
                                "max_command_thrust"
                            ]
                            is None
                            else max(
                                float(
                                    self._visual_gate0_blend_summary[
                                        "max_command_thrust"
                                    ]
                                ),
                                command.thrust,
                            )
                        ),
                    }
                )
                self.recorder.emit(
                    "visual_next_gate_blend_command",
                    elapsed_s=elapsed,
                    current_track_id=(
                        latest_visual_proposal.current_target.track_id
                    ),
                    next_track_id=(
                        latest_visual_proposal.next_target.track_id
                        if latest_visual_proposal.next_target is not None
                        else None
                    ),
                    effective_horizontal_error=(
                        latest_visual_proposal.servo_output
                        .effective_horizontal_error
                    ),
                    effective_vertical_error_image_down=(
                        latest_visual_proposal.servo_output
                        .effective_vertical_error_image_down
                    ),
                    target_pitch_rad=target_pitch,
                    thrust=command.thrust,
                    command_yaw_rate_rad_s=local_yaw_rate,
                    yaw_excursion_rad=yaw_excursion,
                    yaw_soft_stopped=yaw_soft_stopped,
                )
            if early_turn_active:
                early_turn_command_count += 1
                assert self._gate0_early_turn_summary is not None
                self._gate0_early_turn_summary.update(
                    {
                        "command_count": early_turn_command_count,
                        "last_command_yaw_rate_rad_s": local_yaw_rate,
                        "last_yaw_excursion_rad": yaw_excursion,
                        "yaw_soft_stopped": yaw_soft_stopped,
                        "max_abs_yaw_excursion_rad": (
                            self._gate1_max_abs_yaw_excursion_rad
                        ),
                        "max_abs_measured_yaw_rate_rad_s": (
                            self._gate1_max_abs_measured_yaw_rate_rad_s
                        ),
                    }
                )
                self.recorder.emit(
                    "gate0_early_turn_command",
                    elapsed_s=elapsed,
                    filtered_turn_score=filtered_course_turn,
                    target_roll_rad=target_roll,
                    target_pitch_rad=target_pitch,
                    thrust=thrust,
                    command_yaw_rate_rad_s=local_yaw_rate,
                    yaw_excursion_rad=yaw_excursion,
                    yaw_soft_stopped=yaw_soft_stopped,
                )
            await self._send_flight_command(command)
            self._record_tick("gate0", elapsed, command)
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    @staticmethod
    def _signed_error_trend(values: Sequence[float]) -> Dict[str, Any]:
        normalized = [float(value) for value in values]
        deltas = [
            normalized[index] - normalized[index - 1]
            for index in range(1, len(normalized))
        ]
        if not deltas:
            label = "insufficient"
        elif all(delta < 0.0 for delta in deltas):
            label = "negative_uninterrupted"
        elif all(delta > 0.0 for delta in deltas):
            label = "positive_uninterrupted"
        elif all(delta == 0.0 for delta in deltas):
            label = "flat"
        else:
            label = "mixed"
        return {
            "values": normalized,
            "deltas": deltas,
            "trend": label,
        }

    async def _run_visual_shadow(
        self,
        context: StartContext,
    ) -> Dict[str, Any]:
        """Use proved Gate-0 authority while the new graph remains commandless."""

        if not self._visual_tracking_enabled:
            raise SafetyAbort("visual shadow tracker was not enabled before reset")
        bound = self.visual_gate_graph.latest_snapshot
        if (
            bound is None
            or bound.current_track_id is None
            or bound.current_gate_index != 0
        ):
            raise SafetyAbort("visual shadow lacks a bound initial current gate")
        initial_current_track_id = bound.current_track_id
        gate0 = await self._run_gate0(context, capture_transition=False)
        transition = self._visual_transition
        if transition is None:
            raise SafetyAbort("visual shadow lacks a promoted 0->1 transition")
        if (
            transition.retired_track_id != initial_current_track_id
            or transition.promoted_track_id == initial_current_track_id
        ):
            raise SafetyAbort("visual shadow promotion changed identity incorrectly")

        proof = self._gate0_transition_proof
        if proof is None:
            raise SafetyAbort("visual shadow lacks Gate-0 transition timing proof")
        race_credit_ns = transition.race_status.received_monotonic_ns
        if race_credit_ns is None:
            raise SafetyAbort("visual shadow lacks exact race-credit receive time")
        deadline_s = post_gate_observation_deadline(
            pass_confirmed_s=proof.pass_confirmed_monotonic_s,
            flight_started_s=proof.flight_started_monotonic_s,
            crossing_started_s=proof.crossing_started_monotonic_s,
            requested_duration_s=VISUAL_SHADOW_POST_CREDIT_TIMEOUT_S,
        )
        next_tick = max(
            proof.next_control_deadline_s,
            await self._wait_for_next_flight_command_slot(),
        )
        post_credit_tokens: List[VisualCameraFrameToken] = []
        command_count = 0
        while time.monotonic() < deadline_s and not post_credit_tokens:
            self._sample()
            self._watchdog(
                require_target=False,
                allow_benign_pad_contact=False,
            )
            race = self.adapter.race_status
            if (
                race is None
                or int(race.active_gate_index) != 1
                or bool(race.race_finished)
            ):
                raise SafetyAbort(
                    "visual shadow lost authoritative Gate-1 boundary"
                )
            current = self.visual_tracker.track(
                transition.promoted_track_id
            )
            if current.ambiguous or current.role is not VisualTrackRole.CURRENT:
                raise SafetyAbort(
                    "visual shadow promoted track became ambiguous or lost role"
                )
            post_credit_tokens = [
                sample.token
                for sample in current.history
                if sample.publication_monotonic_ns is not None
                and int(sample.publication_monotonic_ns)
                > int(race_credit_ns)
            ]
            command = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
            await self._send_flight_command(command)
            self._record_tick("visual-shadow/post-credit", 0.0, command)
            command_count += 1
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(
                max(0.0, min(next_tick, deadline_s) - time.monotonic())
            )
        if not post_credit_tokens:
            raise SafetyAbort(
                "visual shadow promotion lacks a fresh post-credit frame"
            )

        promoted = self.visual_tracker.track(transition.promoted_track_id)
        token_set = set(transition.pretransition_frame_tokens)
        pretransition_samples = [
            sample for sample in promoted.history if sample.token in token_set
        ]
        if len(pretransition_samples) < (
            VISUAL_SHADOW_REQUIRED_PRETRANSITION_FRAMES
        ):
            raise SafetyAbort(
                "visual shadow retained fewer than three pre-transition frames"
            )
        horizontal_errors = [
            abs(sample.center_norm[0]) for sample in pretransition_samples
        ]
        vertical_errors = [
            abs(sample.center_norm[1]) for sample in pretransition_samples
        ]
        corridor_frames = sum(
            1
            for sample in pretransition_samples
            if abs(sample.center_norm[0])
            <= self.visual_config.servo.horizontal_corridor
            and abs(sample.center_norm[1])
            <= self.visual_config.servo.vertical_corridor
        )
        latest_graph = self.visual_gate_graph.latest_snapshot
        next_track_ids = (
            [candidate.track_id for candidate in latest_graph.next_candidates]
            if latest_graph is not None
            else []
        )
        summary = {
            "shadow_command_authority": "legacy_proved_gate0_only",
            "visual_navigation_command_count": 0,
            "post_credit_zero_command_count": command_count,
            "authoritative_transition": [0, 1],
            "initial_current_track_id": initial_current_track_id,
            "promoted_current_track_id": transition.promoted_track_id,
            "next_track_ids": next_track_ids,
            "pretransition_frame_count": len(pretransition_samples),
            "pretransition_frame_tokens": [
                list(sample.token.live_identity_tuple or sample.token.exact_tuple)
                for sample in pretransition_samples
            ],
            "post_credit_frame_tokens": [
                list(token.live_identity_tuple or token.exact_tuple)
                for token in post_credit_tokens
            ],
            "history_length_before_promotion": (
                transition.history_length_before_promotion
            ),
            "history_length_after_promotion": (
                transition.history_length_after_promotion
            ),
            "horizontal_abs_error_trend": self._signed_error_trend(
                horizontal_errors
            ),
            "vertical_abs_error_trend": self._signed_error_trend(
                vertical_errors
            ),
            "latest_scale": promoted.apparent_scale,
            "latest_log_scale_rate_s": promoted.log_scale_rate_s,
            "corridor_frames": corridor_frames,
            "association_confidence": promoted.association_confidence,
            "ambiguous": promoted.ambiguous,
            "gate0": gate0,
            "graph": self._visual_graph_summary(latest_graph),
        }
        self._visual_shadow_summary = summary
        self.recorder.emit("visual_shadow_complete", **summary)
        return summary

    def _require_visual_current_target(
        self,
        *,
        expected_gate_index: int,
        expected_track_id: str,
        now_s: Optional[float] = None,
    ) -> Tuple[VisualTrack, VisualTarget]:
        """Return only the exact graph-authorized current camera publication."""

        if (
            type(expected_gate_index) is not int
            or expected_gate_index < 0
            or type(expected_track_id) is not str
            or not expected_track_id
        ):
            raise SafetyAbort("visual current-target expectation is invalid")
        update = self.visual_tracker.latest_update
        graph = self.visual_gate_graph.latest_snapshot
        if update is None or graph is None:
            raise SafetyAbort("visual current-target authority is unavailable")
        if (
            graph.latest_camera_token != update.token
            or graph.tracker_frame_sequence != update.tracker_frame_sequence
            or graph.current_track_id != expected_track_id
            or graph.current_gate_index != expected_gate_index
            or graph.current_track is None
            or graph.current_track.track_id != expected_track_id
            or not graph.authority_usable
            or graph.next_selection_ambiguous
            or graph.race_finished
        ):
            raise SafetyAbort(
                "visual gate graph withheld exact current-target authority"
            )
        try:
            track = self.visual_tracker.track(expected_track_id)
        except KeyError as exc:
            raise SafetyAbort(
                "promoted visual current track disappeared"
            ) from exc
        if (
            track != graph.current_track
            or track.latest_token != update.token
            or track.role is not VisualTrackRole.CURRENT
            or track.authoritative_gate_index != expected_gate_index
            or not track.visible
            or track.missed_frame_count != 0
            or track.ambiguous
            or track.consecutive_frame_count < 1
        ):
            raise SafetyAbort(
                "promoted visual current track lost exact authority"
            )
        try:
            target = VisualTarget.from_visual_track(
                track,
                require_current_authority=True,
                expected_gate_index=expected_gate_index,
            )
        except VisualServoRefusal as exc:
            raise SafetyAbort(
                f"visual current-target adaptation refused: {exc}"
            ) from exc
        observed_s = (
            time.perf_counter_ns() / 1_000_000_000.0
            if now_s is None
            else float(now_s)
        )
        age_s = observed_s - float(target.received_monotonic_s)
        if (
            not math.isfinite(age_s)
            or age_s < -1e-6
            or age_s > MAX_VISUAL_OBSERVATION_AGE_S
        ):
            raise SafetyAbort("promoted visual current target is stale")
        return track, target

    def _assert_visual_alignment_race_boundary(self) -> Any:
        race = self.adapter.race_status
        if (
            race is None
            or bool(race.race_finished)
            or int(race.active_gate_index) != 1
        ):
            raise SafetyAbort(
                "visual alignment lost its no-passage Gate-1 boundary"
            )
        return race

    def _assert_visual_alignment_no_passage(
        self,
        track: VisualTrack,
        *,
        phase: str,
    ) -> Dict[str, Any]:
        """Abort before close or opposing-edge gate geometry can be crossed."""

        if type(track) is not VisualTrack:
            raise SafetyAbort("visual no-passage guard requires an exact track")
        left, top, right, bottom = (
            float(value) for value in track.bbox_norm
        )
        width_fraction = max(0.0, right - left)
        height_fraction = max(0.0, bottom - top)
        area_fraction = width_fraction * height_fraction
        width_px = width_fraction * 640.0
        height_px = height_fraction * 360.0
        opposing_edges = bool(
            (
                track.clipping & FrameEdge.LEFT
                and track.clipping & FrameEdge.RIGHT
            )
            or (
                track.clipping & FrameEdge.TOP
                and track.clipping & FrameEdge.BOTTOM
            )
        )
        if (
            area_fraction >= 0.10
            or width_fraction >= 0.25
            or height_fraction >= (1.0 / 3.0)
            or opposing_edges
        ):
            raise SafetyAbort(
                "visual alignment no-passage geometry bound reached "
                f"during {phase}"
            )
        raw_contact_risk = select_untracked_contact_risk(
            self._latest_raw_detections,
            accepted_target=None,
        )
        if raw_contact_risk is not None:
            raise SafetyAbort(
                "visual alignment raw no-passage geometry bound reached "
                f"during {phase}"
            )
        return {
            "area_fraction": area_fraction,
            "width_px": width_px,
            "height_px": height_px,
            "clipping_edges": int(track.clipping),
            "opposing_edges": opposing_edges,
        }

    def _assert_visual_alignment_attitude(
        self,
        *,
        entry_roll_rad: float,
        entry_pitch_rad: float,
        phase: str,
    ) -> Dict[str, float]:
        if self.estimate is None:
            raise SafetyAbort(
                f"visual alignment attitude is unavailable during {phase}"
            )
        roll, pitch, yaw = (
            float(value) for value in self.estimate.orientation.to_euler()
        )
        rates = tuple(float(value) for value in self.estimate.body_rates)
        peak_rate = max(abs(value) for value in rates)
        if (
            abs(roll) > VISUAL_ALIGN_MAX_ABS_ROLL_RAD
            or pitch < VISUAL_ALIGN_MIN_PITCH_RAD
            or pitch > VISUAL_ALIGN_MAX_PITCH_RAD
            or abs(roll - float(entry_roll_rad))
            > VISUAL_ALIGN_MAX_ENTRY_ATTITUDE_DELTA_RAD
            or abs(pitch - float(entry_pitch_rad))
            > VISUAL_ALIGN_MAX_ENTRY_ATTITUDE_DELTA_RAD
            or peak_rate > VISUAL_ALIGN_MAX_BODY_RATE_RAD_S
        ):
            raise SafetyAbort(
                "visual alignment attitude/rate envelope exceeded "
                f"during {phase}"
            )
        return {
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "peak_body_rate_rad_s": peak_rate,
            "yaw_rate_rad_s": rates[2],
        }

    @staticmethod
    def _visual_alignment_trend_summary(
        trend: Optional[VisualAlignmentTrend],
    ) -> Dict[str, Any]:
        if trend is None:
            return {
                "horizontal_abs_error_trend": {
                    "values": [],
                    "deltas": [],
                    "trend": "insufficient",
                },
                "vertical_abs_error_trend": {
                    "values": [],
                    "deltas": [],
                    "trend": "insufficient",
                },
                "log_scale_rate_trend": {
                    "values": [],
                    "trend": "insufficient",
                },
                "corridor_frames": 0,
                "eligible_joint_frame_count": 0,
                "improving_joint_frame_streak": 0,
                "alignment_criteria_met": False,
            }
        return {
            "horizontal_abs_error_trend": {
                "values": list(trend.horizontal_abs_errors),
                "deltas": list(trend.horizontal_deltas),
                "trend": trend.horizontal_trend,
            },
            "vertical_abs_error_trend": {
                "values": list(trend.vertical_abs_errors),
                "deltas": list(trend.vertical_deltas),
                "trend": trend.vertical_trend,
            },
            "log_scale_rate_trend": {
                "values": list(trend.log_scale_rates_s),
                "trend": trend.scale_rate_trend,
            },
            "corridor_frames": int(trend.corridor_frames),
            "eligible_joint_frame_count": int(
                trend.eligible_joint_frame_count
            ),
            "improving_joint_frame_streak": int(
                trend.improving_joint_frame_streak
            ),
            "alignment_criteria_met": bool(trend.accepted),
        }

    async def _run_visual_alignment(
        self,
        context: StartContext,
    ) -> Dict[str, Any]:
        """Delegate the bounded stage while retaining live runner authority."""

        if MAX_VISUAL_YAW_RATE_RAD_S != VISUAL_ALIGN_MAX_YAW_RATE_RAD_S:
            raise SafetyAbort(
                "visual servo yaw bound differs from the fixed live bound"
            )
        return await run_visual_alignment_stage(
            self,
            context,
            runtime=VisualAlignmentStageRuntime(
                limits=VISUAL_ALIGNMENT_STAGE_LIMITS,
                safety_abort_type=SafetyAbort,
                cancelled_error_type=asyncio.CancelledError,
                monotonic=time.monotonic,
                perf_counter_ns=time.perf_counter_ns,
                sleep=asyncio.sleep,
                post_gate_observation_deadline=(
                    post_gate_observation_deadline
                ),
                next_control_deadline=next_control_deadline,
                visual_alignment_yaw_rate=(
                    visual_alignment_yaw_rate
                ),
                attitude_rate_command=attitude_rate_command,
                limit_command_rates=limit_command_rates,
                validate_command=validate_command,
            ),
        )

    def _assert_gate1_no_passage_geometry(
        self,
        target: Optional[GateTarget],
        *,
        phase: str,
    ) -> None:
        if target is not None and (
            int(target.bbox_area) >= GATE1_RECENTER_NO_PASSAGE_MAX_AREA_PX
            or int(target.bbox[2])
            >= GATE1_RECENTER_NO_PASSAGE_MAX_WIDTH_PX
        ):
            raise SafetyAbort(
                "gate-1 recenter no-passage geometry bound reached "
                f"during {phase}"
            )
        raw_contact_risk = select_untracked_contact_risk(
            self._latest_raw_detections,
            accepted_target=None,
        )
        if raw_contact_risk is not None:
            raise SafetyAbort(
                "gate-1 recenter raw no-passage geometry bound reached "
                f"during {phase}"
            )

    async def _observe_gate1(
        self,
        gate0_details: Dict[str, Any],
        *,
        hold_thrust: float = 0.0,
    ) -> Dict[str, Any]:
        """Collect a bounded view after a proved gate-0 pass.

        Observation authority has exact-zero attitude rates. The powered
        recenter stage may preserve its exact transition thrust so the motors
        do not cut during handoff; no Gate 1 steering or passage is supplied.
        """

        if (
            type(hold_thrust) not in {int, float}
            or not math.isfinite(float(hold_thrust))
            or float(hold_thrust)
            not in {0.0, GATE1_RECENTER_TRANSITION_THRUST}
        ):
            raise ValueError(
                "post-gate observation thrust must be exact zero or the "
                "Gate-1 continuation value"
            )
        proof = self._gate0_transition_proof
        if proof is None or not gate0_details.get("gate_transition_proved"):
            raise SafetyAbort("gate-1 observation lacks an authoritative transition proof")
        if (
            int(gate0_details.get("pre_gate_race_boot_ms", -1))
            != proof.pre_gate_race_boot_ms
            or int(gate0_details.get("race_boot_ms", -1))
            != proof.post_gate_race_boot_ms
            or proof.post_gate_race_boot_ms <= proof.pre_gate_race_boot_ms
        ):
            raise SafetyAbort("gate-1 observation transition proof is inconsistent")

        race = self.adapter.race_status
        if (
            race is None
            or int(race.active_gate_index) != 1
            or int(race.sim_boot_time_ms) < proof.post_gate_race_boot_ms
        ):
            raise SafetyAbort("race status no longer matches the proved gate-1 transition")
        watermark = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if watermark is None:
            raise SafetyAbort("camera unavailable at gate-1 observation handoff")
        if int(watermark.generation) != proof.vision_generation:
            raise SafetyAbort("vision generation changed after gate-0 passage")
        if (
            int(watermark.frame_id) < proof.vision_frame_id
            or int(watermark.sim_time_ns) < proof.vision_sim_time_ns
            or float(watermark.received_monotonic_s)
            < proof.vision_received_monotonic_s
        ):
            raise SafetyAbort("camera snapshot regressed after gate-0 passage")

        # Deliberately skip any frame that existed before the tracker reset.
        # The vision receiver and its generation remain untouched.
        self._last_frame_identity = (
            int(watermark.generation),
            int(watermark.frame_id),
        )
        self._last_frame_sim_ns = int(watermark.sim_time_ns)
        self._latest_detection_frame_id = int(watermark.frame_id)
        self._latest_detection_frame_sim_ns = int(watermark.sim_time_ns)
        self._latest_detection_generation = int(watermark.generation)
        self._latest_detection_received_s = float(watermark.received_monotonic_s)
        self._latest_raw_detections = []
        self.tracker.reset()
        self._latest_accepted_target = None
        self._post_gate_reacquisition = True
        self._vision_diagnostic_logging = True

        hard_deadline = post_gate_observation_deadline(
            pass_confirmed_s=proof.pass_confirmed_monotonic_s,
            flight_started_s=proof.flight_started_monotonic_s,
            crossing_started_s=proof.crossing_started_monotonic_s,
            requested_duration_s=(
                self.controller_config.phase_timing
                .post_gate_observation_duration_s
            ),
        )
        observation_started_s = time.monotonic()
        if observation_started_s >= hard_deadline:
            self._post_gate_reacquisition = False
            raise SafetyAbort("gate-1 observation has no remaining safety budget")

        next_tick = max(
            proof.next_control_deadline_s,
            (
                self._last_flight_command_sent_s + CONTROL_PERIOD_S
                if self._last_flight_command_sent_s is not None
                else observation_started_s
            ),
        )
        transition_command = AttitudeRateCommand(
            0.0,
            0.0,
            0.0,
            float(hold_thrust),
        )
        last_processed_token = (
            int(watermark.generation),
            int(watermark.frame_id),
            int(watermark.sim_time_ns),
        )
        qualifying_frames: List[Dict[str, Any]] = []
        strict_high_rate_samples = 0
        self.recorder.emit(
            "post_gate_observation_started",
            pre_gate_race_boot_ms=proof.pre_gate_race_boot_ms,
            post_gate_race_boot_ms=proof.post_gate_race_boot_ms,
            hard_deadline_monotonic_s=hard_deadline,
            budget_s=hard_deadline - observation_started_s,
            hold_thrust=float(hold_thrust),
            watermark={
                "generation": watermark.generation,
                "frame_id": watermark.frame_id,
                "sim_time_ns": watermark.sim_time_ns,
                "received_monotonic_s": watermark.received_monotonic_s,
            },
        )

        try:
            initial_wait = min(next_tick, hard_deadline) - time.monotonic()
            if initial_wait > 0.0:
                await asyncio.sleep(initial_wait)
            while True:
                if time.monotonic() >= hard_deadline:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                self._sample()
                self._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=False,
                    enforce_benign_pad_budget=False,
                )
                now = time.monotonic()
                if now >= hard_deadline:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                race = self.adapter.race_status
                if race is None or int(race.active_gate_index) != 1:
                    gate_index = race.active_gate_index if race is not None else None
                    raise SafetyAbort(
                        f"gate index changed during gate-1 observation ({gate_index})"
                    )
                if int(race.sim_boot_time_ms) < proof.post_gate_race_boot_ms:
                    raise SafetyAbort("race clock regressed below the gate-1 proof")
                if float(hold_thrust) != 0.0:
                    self._assert_gate1_no_passage_geometry(
                        self._latest_accepted_target,
                        phase="powered observation",
                    )
                assert self.estimate is not None
                if self._gate1_yaw_reference_rad is not None:
                    self._gate1_yaw_envelope_state(
                        phase="Gate-1 observation",
                    )
                roll, pitch, _yaw = self.estimate.orientation.to_euler()
                if (
                    abs(roll - proof.pass_rpy_rad[0])
                    > POST_GATE_MAX_ATTITUDE_DELTA_RAD
                    or abs(pitch - proof.pass_rpy_rad[1])
                    > POST_GATE_MAX_ATTITUDE_DELTA_RAD
                ):
                    raise SafetyAbort(
                        "attitude changed over 5deg during zero-rate "
                        "transition observation"
                    )
                peak_rate = max(abs(value) for value in self.estimate.body_rates)
                if peak_rate > POST_GATE_IMMEDIATE_MAX_BODY_RATE_RAD_S:
                    raise SafetyAbort(
                        "body rate exceeded 1.0rad/s during gate-1 observation"
                    )
                strict_high_rate_samples = (
                    strict_high_rate_samples + 1
                    if peak_rate > POST_GATE_SUSTAINED_MAX_BODY_RATE_RAD_S
                    else 0
                )
                if strict_high_rate_samples >= 2:
                    raise SafetyAbort(
                        "body rate exceeded 0.5rad/s for two gate-1 observation samples"
                    )

                frame_token: Optional[Tuple[int, int, int]] = None
                if (
                    self._latest_detection_generation is not None
                    and self._latest_detection_frame_id is not None
                    and self._latest_detection_frame_sim_ns is not None
                ):
                    frame_token = (
                        self._latest_detection_generation,
                        self._latest_detection_frame_id,
                        self._latest_detection_frame_sim_ns,
                    )
                if frame_token is not None and frame_token != last_processed_token:
                    generation, frame_id, sim_time_ns = frame_token
                    received_s = self._latest_detection_received_s
                    if generation != int(watermark.generation):
                        raise SafetyAbort("vision generation changed during gate-1 observation")
                    if (
                        frame_id <= int(watermark.frame_id)
                        or sim_time_ns <= int(watermark.sim_time_ns)
                        or received_s is None
                        or received_s <= float(watermark.received_monotonic_s)
                    ):
                        raise SafetyAbort("post-pass camera frame did not advance strictly")
                    last_processed_token = frame_token
                    accepted = self._latest_accepted_target
                    if accepted is None or is_crossing_residue(accepted):
                        self.recorder.emit(
                            "post_gate_candidate_reset",
                            frame_id=frame_id,
                            sim_time_ns=sim_time_ns,
                            reason=(
                                "crossing_residue"
                                if accepted is not None
                                else "no_continuous_candidate"
                            ),
                        )
                        self.tracker.reset()
                        qualifying_frames = []
                    else:
                        record = {
                            "frame_id": accepted.frame_id,
                            "sim_time_ns": accepted.sim_time_ns,
                            "received_monotonic_s": accepted.received_monotonic_s,
                            "center_px": [accepted.center_x, accepted.center_y],
                            "bbox_xywh_px": list(accepted.bbox),
                            "confidence": accepted.confidence,
                            "tracker_streak": self.tracker.consecutive,
                            "rpy_rad": list(self.estimate.orientation.to_euler()),
                            "body_rates_rad_s": list(self.estimate.body_rates),
                        }
                        if self.tracker.consecutive == 1:
                            qualifying_frames = [record]
                        else:
                            qualifying_frames.append(record)
                            qualifying_frames = qualifying_frames[
                                -self.tracker.consecutive :
                            ]
                        self.recorder.emit("post_gate_candidate_frame", **record)
                        candidate_checked_s = time.monotonic()
                        if candidate_checked_s >= hard_deadline:
                            raise SafetyAbort(
                                "gate-1 observation timed out before three frames"
                            )
                        if (
                            self.tracker.consecutive >= POST_GATE_REQUIRED_FRAMES
                            and len(qualifying_frames) >= POST_GATE_REQUIRED_FRAMES
                            and accepted.age_s(candidate_checked_s) <= MAX_VISION_AGE_S
                        ):
                            # Recheck every generic guard immediately before
                            # accepting the observation result.
                            self._watchdog(
                                require_target=False,
                                allow_benign_pad_contact=False,
                                enforce_benign_pad_budget=False,
                            )
                            accepted_at_s = time.monotonic()
                            if accepted_at_s >= hard_deadline:
                                raise SafetyAbort(
                                    "gate-1 observation timed out before three frames"
                                )
                            final_race = self.adapter.race_status
                            if (
                                final_race is None
                                or int(final_race.active_gate_index) != 1
                                or int(final_race.sim_boot_time_ms)
                                < proof.post_gate_race_boot_ms
                            ):
                                raise SafetyAbort(
                                    "race status changed at gate-1 observation acceptance"
                                )
                            if float(hold_thrust) != 0.0:
                                self._assert_gate1_no_passage_geometry(
                                    accepted,
                                    phase="powered observation acceptance",
                                )
                            result = {
                                "gate1_observed": True,
                                "observation_elapsed_s": (
                                    accepted_at_s - observation_started_s
                                ),
                                "frame_count": POST_GATE_REQUIRED_FRAMES,
                                "frames": qualifying_frames[-POST_GATE_REQUIRED_FRAMES:],
                                "final_gate_bbox": list(accepted.bbox),
                                "final_gate_center": [
                                    accepted.center_x,
                                    accepted.center_y,
                                ],
                                "race_boot_ms": int(final_race.sim_boot_time_ms),
                                "gate_index": int(final_race.active_gate_index),
                                "hold_thrust": float(hold_thrust),
                            }
                            return result

                # Leave the final setpoint slot for cleanup, preventing a
                # zero-command burst at an odd nested deadline.
                if hard_deadline - time.monotonic() <= CONTROL_PERIOD_S:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                self._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=False,
                    enforce_benign_pad_budget=False,
                    count_rate_sample=False,
                )
                send_checked_s = time.monotonic()
                if hard_deadline - send_checked_s <= CONTROL_PERIOD_S:
                    raise SafetyAbort("gate-1 observation timed out before three frames")
                send_race = self.adapter.race_status
                if (
                    send_race is None
                    or int(send_race.active_gate_index) != 1
                    or int(send_race.sim_boot_time_ms)
                    < proof.post_gate_race_boot_ms
                ):
                    raise SafetyAbort(
                        "race status changed before gate-1 observation setpoint"
                    )
                if float(hold_thrust) != 0.0:
                    self._assert_gate1_no_passage_geometry(
                        self._latest_accepted_target,
                        phase="powered observation command send",
                    )

                await self._send_flight_command(transition_command)
                self._record_tick(
                    "gate0-observe/post-pass",
                    send_checked_s - observation_started_s,
                    transition_command,
                )
                next_tick = next_control_deadline(next_tick, time.monotonic())
                await asyncio.sleep(
                    max(0.0, min(next_tick, hard_deadline) - time.monotonic())
                )
        finally:
            self._post_gate_reacquisition = False

    async def _run_bounded_gate1_recenter(
        self,
        gate1_observation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Run the user-authorized bounded, no-passage Gate 1 diagnostic."""

        self._gate1_recenter_summary = None
        phase_timing = self.controller_config.phase_timing
        roll_control = self.controller_config.roll_control
        yaw_control = self.controller_config.yaw_control
        forward_braking = self.controller_config.forward_braking
        if not isinstance(gate1_observation, Mapping):
            raise SafetyAbort("gate-1 recenter lacks a valid observation")
        proof = self._gate0_transition_proof
        frames = gate1_observation.get("frames")

        if (
            proof is None
            or gate1_observation.get("gate1_observed") is not True
            or int(gate1_observation.get("gate_index", -1)) != 1
            or int(gate1_observation.get("frame_count", -1))
            != POST_GATE_REQUIRED_FRAMES
            or not isinstance(frames, Sequence)
            or len(frames) < POST_GATE_REQUIRED_FRAMES
        ):
            raise SafetyAbort("gate-1 recenter lacks the proved three-frame handoff")
        handoff_frames = list(frames[-POST_GATE_REQUIRED_FRAMES:])
        handoff_tokens: List[Tuple[int, int, float]] = []
        for frame in handoff_frames:
            if not isinstance(frame, Mapping):
                raise SafetyAbort(
                    "gate-1 recenter handoff contains an invalid frame"
                )
            frame_id = frame.get("frame_id")
            sim_time_ns = frame.get("sim_time_ns")
            received_s = frame.get("received_monotonic_s")
            if (
                type(frame_id) is not int
                or type(sim_time_ns) is not int
                or type(received_s) not in {int, float}
                or frame_id < 0
                or sim_time_ns < 0
                or not math.isfinite(float(received_s))
            ):
                raise SafetyAbort(
                    "gate-1 recenter handoff frame provenance is invalid"
                )
            handoff_tokens.append(
                (frame_id, sim_time_ns, float(received_s))
            )
        if any(
            handoff_tokens[index][0] <= handoff_tokens[index - 1][0]
            or handoff_tokens[index][1] <= handoff_tokens[index - 1][1]
            or handoff_tokens[index][2] <= handoff_tokens[index - 1][2]
            for index in range(1, len(handoff_tokens))
        ):
            raise SafetyAbort(
                "gate-1 recenter handoff frames did not advance strictly"
            )
        first_handoff_token = handoff_tokens[0]
        if (
            first_handoff_token[0] <= proof.vision_frame_id
            or first_handoff_token[1] <= proof.vision_sim_time_ns
            or first_handoff_token[2] <= proof.vision_received_monotonic_s
        ):
            raise SafetyAbort(
                "gate-1 recenter handoff did not begin after the proved transition"
            )
        race = self.adapter.race_status
        if (
            race is None
            or bool(race.race_finished)
            or int(race.active_gate_index) != 1
            or int(race.sim_boot_time_ms) < proof.post_gate_race_boot_ms
        ):
            raise SafetyAbort("gate-1 recenter race authority is invalid")
        entry_target = self.tracker.target
        if (
            entry_target is None
            or entry_target.composite
            or self._latest_accepted_target is not entry_target
            or self.tracker.last_selection_mode != "primary"
            or self.tracker.consecutive < POST_GATE_REQUIRED_FRAMES
        ):
            raise SafetyAbort("gate-1 recenter requires a fresh primary target")
        entry_error_px = float(entry_target.center_x) - 320.0
        entry_normalized_x = entry_error_px / 320.0
        self._gate1_recenter_summary = {
            "candidate_authority": "user_authorized_bounded_recenter_diagnostic",
            "success": False,
            "recenter_criteria_met": False,
            "outcome": "entry_validation",
            "reason": None,
            "entry_horizontal_error_px": entry_error_px,
            "entry_abs_horizontal_error_px": abs(entry_error_px),
            "final_horizontal_error_px": entry_error_px,
            "final_abs_horizontal_error_px": abs(entry_error_px),
            "entry_normalized_x": entry_normalized_x,
            "target_pitch_rad": forward_braking.gate1_target_pitch_rad,
            "max_target_area_px": int(entry_target.bbox_area),
            "max_target_width_px": int(entry_target.bbox[2]),
            "no_passage_max_area_px": (
                GATE1_RECENTER_NO_PASSAGE_MAX_AREA_PX
            ),
            "no_passage_max_width_px": (
                GATE1_RECENTER_NO_PASSAGE_MAX_WIDTH_PX
            ),
            "authoritative_max_gate_index": 1,
            "contact_safety_outcome": "clean_so_far",
            "cleanup_confirmed": False,
        }
        self._assert_gate1_no_passage_geometry(
            entry_target,
            phase="entry",
        )
        entry_generation = self._latest_detection_generation
        entry_frame_id = self._latest_detection_frame_id
        entry_sim_time_ns = self._latest_detection_frame_sim_ns
        entry_received_s = self._latest_detection_received_s
        if (
            entry_generation != proof.vision_generation
            or entry_frame_id != entry_target.frame_id
            or entry_sim_time_ns != entry_target.sim_time_ns
            or entry_received_s != entry_target.received_monotonic_s
        ):
            raise SafetyAbort("gate-1 recenter entry frame provenance is invalid")
        final_observation = frames[-1]
        if (
            not isinstance(final_observation, Mapping)
            or int(final_observation.get("frame_id", -1)) != entry_target.frame_id
            or int(final_observation.get("sim_time_ns", -1))
            != entry_target.sim_time_ns
            or float(final_observation.get("received_monotonic_s", math.nan))
            != entry_target.received_monotonic_s
            or list(final_observation.get("center_px", ()))
            != [entry_target.center_x, entry_target.center_y]
            or list(final_observation.get("bbox_xywh_px", ()))
            != list(entry_target.bbox)
        ):
            raise SafetyAbort("gate-1 recenter entry does not match observation")
        if entry_target.age_s() > MAX_VISION_AGE_S:
            raise SafetyAbort("gate-1 recenter entry target is stale")
        if self.estimate is None:
            raise SafetyAbort("gate-1 recenter attitude estimate is unavailable")

        entry_roll, entry_pitch, entry_yaw = (
            self.estimate.orientation.to_euler()
        )
        if (
            self._gate1_yaw_reference_rad is None
            and yaw_control.command_rate_cap_rad_s > 0.0
        ):
            self._gate1_yaw_reference_rad = float(entry_yaw)
        entry_yaw_excursion, entry_yaw_soft_stopped = (
            self._gate1_yaw_envelope_state(phase="Gate-1 recenter entry")
            if self._gate1_yaw_reference_rad is not None
            else (0.0, False)
        )
        if (
            abs(float(entry_roll)) > GATE1_RECENTER_MAX_ABS_ROLL_RAD
            or not (
                GATE1_RECENTER_MIN_PITCH_RAD
                <= float(entry_pitch)
                <= GATE1_RECENTER_MAX_PITCH_RAD
            )
        ):
            raise SafetyAbort(
                "gate-1 recenter entry attitude approached its bound"
            )
        entry_abs_error_px = abs(entry_error_px)
        fresh_error_samples: List[Tuple[float, float]] = [
            (float(entry_target.received_monotonic_s), entry_abs_error_px)
        ]
        min_roll = max_roll = float(entry_roll)
        min_pitch = max_pitch = float(entry_pitch)
        min_command_yaw_rate: Optional[float] = None
        max_command_yaw_rate: Optional[float] = None
        min_command_roll_rate: Optional[float] = None
        max_command_roll_rate: Optional[float] = None
        min_command_pitch_rate: Optional[float] = None
        max_command_pitch_rate: Optional[float] = None
        max_target_area = int(entry_target.bbox_area)
        max_target_width = int(entry_target.bbox[2])
        max_gate_index = 1
        fresh_control_frames = 0
        corridor_hold_frames = 0
        command_count = 0
        current_x_rate_norm_s = 0.0
        last_token = (
            int(entry_generation),
            int(entry_target.frame_id),
            int(entry_target.sim_time_ns),
        )
        last_center_x = float(entry_target.center_x)
        last_received_s = float(entry_target.received_monotonic_s)
        latest_target = entry_target
        recenter_started_s = await self._wait_for_next_flight_command_slot()
        fixed_hard_deadline_s = (
            recenter_started_s + GATE1_RECENTER_DURATION_S
        )
        hard_deadline_s = min(
            fixed_hard_deadline_s,
            recenter_started_s
            + phase_timing.gate1_recenter_duration_s,
        )
        wire_clock_anchor_ns = time.perf_counter_ns()
        wire_clock_anchor_s = time.monotonic()
        remaining_wire_budget_s = max(
            0.0,
            hard_deadline_s - wire_clock_anchor_s - CONTROL_PERIOD_S,
        )
        last_recenter_wire_start_ns = wire_clock_anchor_ns + math.floor(
            remaining_wire_budget_s * 1_000_000_000
        )
        next_tick = recenter_started_s
        last_dispatch_attempt_s: Optional[float] = None
        drain_outbound_receipts = getattr(
            self.adapter,
            "drain_outbound_receipts",
            None,
        )
        if not callable(drain_outbound_receipts):
            raise SafetyAbort(
                "gate-1 recenter requires exact outbound wire receipts"
            )
        prior_receipts = [
            self._outbound_receipt_primitive(value)
            for value in drain_outbound_receipts()
        ]

        summary: Dict[str, Any] = {
            "candidate_authority": "user_authorized_bounded_recenter_diagnostic",
            "success": False,
            "recenter_criteria_met": False,
            "outcome": "running",
            "reason": None,
            "duration_s": 0.0,
            "corridor_accepted_elapsed_s": None,
            "entry_horizontal_error_px": entry_error_px,
            "entry_abs_horizontal_error_px": entry_abs_error_px,
            "entry_normalized_x": entry_normalized_x,
            "entry_yaw_rad": float(entry_yaw),
            "entry_yaw_excursion_rad": entry_yaw_excursion,
            "entry_yaw_soft_stopped": entry_yaw_soft_stopped,
            "final_horizontal_error_px": entry_error_px,
            "final_abs_horizontal_error_px": entry_abs_error_px,
            "max_abs_yaw_excursion_rad": (
                self._gate1_max_abs_yaw_excursion_rad
            ),
            "max_abs_measured_yaw_rate_rad_s": (
                self._gate1_max_abs_measured_yaw_rate_rad_s
            ),
            "fresh_abs_horizontal_error_slope_px_s": None,
            "fresh_control_frame_count": 0,
            "corridor_hold_frame_count": 0,
            "min_roll_rad": min_roll,
            "max_roll_rad": max_roll,
            "min_pitch_rad": min_pitch,
            "max_pitch_rad": max_pitch,
            "min_command_roll_rate_rad_s": None,
            "max_command_roll_rate_rad_s": None,
            "min_command_pitch_rate_rad_s": None,
            "max_command_pitch_rate_rad_s": None,
            "min_command_yaw_rate_rad_s": None,
            "max_command_yaw_rate_rad_s": None,
            "command_count": 0,
            "target_pitch_rad": forward_braking.gate1_target_pitch_rad,
            "forward_thrust": forward_braking.gate1_forward_thrust,
            "requested_duration_s": (
                phase_timing.gate1_recenter_duration_s
            ),
            "fixed_hard_duration_s": GATE1_RECENTER_DURATION_S,
            "max_target_area_px": max_target_area,
            "max_target_width_px": max_target_width,
            "no_passage_max_area_px": (
                GATE1_RECENTER_NO_PASSAGE_MAX_AREA_PX
            ),
            "no_passage_max_width_px": (
                GATE1_RECENTER_NO_PASSAGE_MAX_WIDTH_PX
            ),
            "authoritative_max_gate_index": max_gate_index,
            "contact_safety_outcome": "clean_so_far",
            "cleanup_confirmed": False,
        }
        self._gate1_recenter_summary = summary

        def refresh_summary(
            *,
            outcome: str,
            reason: Optional[str],
            criteria_met: bool = False,
            corridor_accepted_elapsed_s: Optional[float] = None,
        ) -> None:
            nonlocal min_roll, max_roll, min_pitch, max_pitch
            nonlocal min_command_roll_rate, max_command_roll_rate
            nonlocal min_command_pitch_rate, max_command_pitch_rate
            nonlocal min_command_yaw_rate, max_command_yaw_rate
            nonlocal max_target_area, max_target_width, max_gate_index
            elapsed = max(0.0, time.monotonic() - recenter_started_s)
            slope = gate1_recenter_absolute_error_slope_px_s(
                fresh_error_samples
            )
            final_error_px = float(latest_target.center_x) - 320.0
            summary.update(
                {
                    "success": bool(
                        criteria_met and summary["cleanup_confirmed"]
                    ),
                    "recenter_criteria_met": bool(criteria_met),
                    "outcome": str(outcome),
                    "reason": reason,
                    "duration_s": elapsed,
                    "corridor_accepted_elapsed_s": (
                        corridor_accepted_elapsed_s
                    ),
                    "final_horizontal_error_px": final_error_px,
                    "final_abs_horizontal_error_px": abs(final_error_px),
                    "max_abs_yaw_excursion_rad": (
                        self._gate1_max_abs_yaw_excursion_rad
                    ),
                    "max_abs_measured_yaw_rate_rad_s": (
                        self._gate1_max_abs_measured_yaw_rate_rad_s
                    ),
                    "fresh_abs_horizontal_error_slope_px_s": slope,
                    "fresh_control_frame_count": fresh_control_frames,
                    "corridor_hold_frame_count": corridor_hold_frames,
                    "min_roll_rad": min_roll,
                    "max_roll_rad": max_roll,
                    "min_pitch_rad": min_pitch,
                    "max_pitch_rad": max_pitch,
                    "min_command_roll_rate_rad_s": min_command_roll_rate,
                    "max_command_roll_rate_rad_s": max_command_roll_rate,
                    "min_command_pitch_rate_rad_s": min_command_pitch_rate,
                    "max_command_pitch_rate_rad_s": max_command_pitch_rate,
                    "min_command_yaw_rate_rad_s": min_command_yaw_rate,
                    "max_command_yaw_rate_rad_s": max_command_yaw_rate,
                    "command_count": command_count,
                    "max_target_area_px": max_target_area,
                    "max_target_width_px": max_target_width,
                    "authoritative_max_gate_index": max_gate_index,
                    "contact_safety_outcome": (
                        "clean"
                        if reason is None
                        else (
                            (
                                "contact_abort"
                                if "collision" in reason or "contact" in reason
                                else "safety_abort"
                            )
                            if outcome == "abort"
                            else (
                                "interrupted"
                                if outcome == "interrupted"
                                else "infrastructure_error"
                            )
                        )
                    ),
                }
            )

        async def reserve_terminal_cleanup_slot(
            *,
            uncertain_dispatch_observed_s: Optional[float] = None,
        ) -> None:
            """Leave one command period after any completed or uncertain send."""

            if self._last_flight_command_sent_s is not None:
                await self._wait_for_next_flight_command_slot()
            if uncertain_dispatch_observed_s is None:
                return
            not_before_s = uncertain_dispatch_observed_s + CONTROL_PERIOD_S
            observed_s = time.monotonic()
            wait_attempts = 0
            while observed_s < not_before_s:
                if wait_attempts >= 8:
                    raise SafetyAbort(
                        "gate-1 recenter cleanup pacing wait returned early"
                    )
                await asyncio.sleep(not_before_s - observed_s)
                next_observed_s = time.monotonic()
                if (
                    type(next_observed_s) not in {int, float}
                    or not math.isfinite(float(next_observed_s))
                    or float(next_observed_s) < observed_s
                ):
                    raise SafetyAbort(
                        "gate-1 recenter cleanup pacing clock is invalid"
                    )
                observed_s = float(next_observed_s)
                wait_attempts += 1

        self.recorder.emit(
            "gate1_recenter_started",
            entry_target=asdict(entry_target),
            entry_error_px=entry_error_px,
            entry_abs_error_px=entry_abs_error_px,
            vision_generation=entry_generation,
            hard_deadline_monotonic_s=hard_deadline_s,
            last_recenter_wire_start_monotonic_ns=(
                last_recenter_wire_start_ns
            ),
            drained_prior_outbound_receipt_count=len(prior_receipts),
            control_law={
                "roll_gain": roll_control.gate1_error_gain,
                "roll_rate_gain": roll_control.gate1_error_rate_gain,
                "max_roll_rad": roll_control.gate1_target_cap_rad,
                "max_command_rate_rad_s": (
                    roll_control.command_rate_cap_rad_s
                ),
                "yaw_gain": yaw_control.gate1_error_gain,
                "yaw_deadband_normalized_x": (
                    yaw_control.gate1_deadband_normalized_x
                ),
                "max_yaw_command_rate_rad_s": (
                    yaw_control.command_rate_cap_rad_s
                ),
                "max_yaw_excursion_rad": (
                    GATE1_CONTROLLER_MAX_YAW_EXCURSION_RAD
                ),
                "target_pitch_rad": (
                    forward_braking.gate1_target_pitch_rad
                ),
                "thrust": forward_braking.gate1_forward_thrust,
                "requested_duration_s": (
                    phase_timing.gate1_recenter_duration_s
                ),
                "fixed_hard_duration_s": GATE1_RECENTER_DURATION_S,
            },
        )

        try:
            while True:
                now = time.monotonic()
                if now >= hard_deadline_s:
                    raise SafetyAbort(
                        "gate-1 recenter hard 0.60s window expired"
                    )
                self._sample()
                now = time.monotonic()
                if now >= hard_deadline_s:
                    raise SafetyAbort(
                        "gate-1 recenter hard 0.60s window expired"
                    )
                self._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=False,
                    enforce_benign_pad_budget=False,
                )
                race = self.adapter.race_status
                if race is None:
                    raise SafetyAbort("gate-1 recenter race status is unavailable")
                max_gate_index = max(max_gate_index, int(race.active_gate_index))
                if bool(race.race_finished) or int(race.active_gate_index) != 1:
                    raise SafetyAbort(
                        "gate index changed during gate-1 recenter "
                        f"({race.active_gate_index})"
                    )
                if int(race.sim_boot_time_ms) < proof.post_gate_race_boot_ms:
                    raise SafetyAbort(
                        "race clock regressed below the gate-1 transition proof"
                    )
                if self._latest_detection_generation != entry_generation:
                    raise SafetyAbort(
                        "vision generation changed during gate-1 recenter"
                    )
                accepted = self._latest_accepted_target
                if accepted is None:
                    contact_risk = select_untracked_contact_risk(
                        self._latest_raw_detections,
                        accepted_target=None,
                    )
                    if contact_risk is not None:
                        raise SafetyAbort(
                            "untracked contact risk during gate-1 recenter"
                        )
                    raise SafetyAbort(
                        "primary gate target lost during gate-1 recenter"
                    )
                if (
                    accepted.composite
                    or self.tracker.last_selection_mode != "primary"
                    or self.tracker.consecutive < POST_GATE_REQUIRED_FRAMES
                    or accepted.frame_id != self._latest_detection_frame_id
                    or accepted.sim_time_ns != self._latest_detection_frame_sim_ns
                    or accepted.received_monotonic_s
                    != self._latest_detection_received_s
                    or accepted.age_s(now) > MAX_VISION_AGE_S
                ):
                    raise SafetyAbort(
                        "gate-1 recenter target lost primary fresh-frame authority"
                    )
                self._assert_gate1_no_passage_geometry(
                    accepted,
                    phase="control",
                )
                if self.estimate is None:
                    raise SafetyAbort(
                        "gate-1 recenter attitude estimate is unavailable"
                    )
                roll, pitch, yaw = self.estimate.orientation.to_euler()
                yaw_excursion, yaw_soft_stopped = (
                    self._gate1_yaw_envelope_state(
                        phase="Gate-1 recenter",
                    )
                    if self._gate1_yaw_reference_rad is not None
                    else (0.0, False)
                )
                min_roll = min(min_roll, float(roll))
                max_roll = max(max_roll, float(roll))
                min_pitch = min(min_pitch, float(pitch))
                max_pitch = max(max_pitch, float(pitch))
                if (
                    abs(float(roll)) > GATE1_RECENTER_MAX_ABS_ROLL_RAD
                    or not (
                        GATE1_RECENTER_MIN_PITCH_RAD
                        <= float(pitch)
                        <= GATE1_RECENTER_MAX_PITCH_RAD
                    )
                    or abs(float(roll) - entry_roll)
                    > GATE1_RECENTER_MAX_ATTITUDE_EXCURSION_RAD
                    or abs(float(pitch) - entry_pitch)
                    > GATE1_RECENTER_MAX_ATTITUDE_EXCURSION_RAD
                ):
                    raise SafetyAbort(
                        "gate-1 recenter attitude excursion approached its bound"
                    )
                measured_peak_rate = max(
                    abs(float(value)) for value in self.estimate.body_rates
                )
                if (
                    measured_peak_rate
                    > GATE1_RECENTER_MAX_MEASURED_BODY_RATE_RAD_S
                ):
                    raise SafetyAbort(
                        "gate-1 recenter measured body rate approached its bound"
                    )

                token = (
                    int(entry_generation),
                    int(accepted.frame_id),
                    int(accepted.sim_time_ns),
                )
                latest_target = accepted
                if token != last_token:
                    if (
                        token[1] <= last_token[1]
                        or token[2] <= last_token[2]
                        or float(accepted.received_monotonic_s)
                        <= last_received_s
                    ):
                        raise SafetyAbort(
                            "gate-1 recenter frame did not advance strictly"
                        )
                    dt_target = (
                        float(accepted.received_monotonic_s) - last_received_s
                    )
                    if dt_target <= 1e-3:
                        raise SafetyAbort(
                            "gate-1 recenter frame interval is too small"
                        )
                    current_x_rate_norm_s = (
                        (float(accepted.center_x) - last_center_x)
                        / 320.0
                        / dt_target
                    )
                    try:
                        gate1_recenter_roll_target(
                            (float(accepted.center_x) - 320.0) / 320.0,
                            current_x_rate_norm_s,
                            error_gain=roll_control.gate1_error_gain,
                            error_rate_gain=(
                                roll_control.gate1_error_rate_gain
                            ),
                            cap_rad=roll_control.gate1_target_cap_rad,
                        )
                    except ValueError as exc:
                        raise SafetyAbort(
                            "gate-1 recenter horizontal rate is outside bounds"
                        ) from exc
                    fresh_control_frames += 1
                    current_error_px = float(accepted.center_x) - 320.0
                    current_abs_error_px = abs(current_error_px)
                    fresh_error_samples.append(
                        (
                            float(accepted.received_monotonic_s),
                            current_abs_error_px,
                        )
                    )
                    max_target_area = max(
                        max_target_area,
                        int(accepted.bbox_area),
                    )
                    max_target_width = max(
                        max_target_width,
                        int(accepted.bbox[2]),
                    )
                    if (
                        fresh_control_frames >= 3
                        and current_abs_error_px
                        > entry_abs_error_px + GATE1_RECENTER_DIVERGENCE_PX
                    ):
                        raise SafetyAbort(
                            "gate-1 recenter horizontal error diverged by more "
                            "than 24px"
                        )
                    if (
                        abs(current_error_px / 320.0)
                        <= GATE1_RECENTER_CORRIDOR_NORMALIZED_X
                    ):
                        corridor_hold_frames += 1
                    else:
                        corridor_hold_frames = 0
                    self.recorder.emit(
                        "gate1_recenter_fresh_frame",
                        fresh_control_frame=fresh_control_frames,
                        frame_id=accepted.frame_id,
                        sim_time_ns=accepted.sim_time_ns,
                        center_x=accepted.center_x,
                        horizontal_error_px=current_error_px,
                        normalized_x=current_error_px / 320.0,
                        normalized_x_rate_s=current_x_rate_norm_s,
                        corridor_hold_frames=corridor_hold_frames,
                        yaw_rad=float(yaw),
                        yaw_excursion_rad=yaw_excursion,
                        yaw_soft_stopped=yaw_soft_stopped,
                    )
                    last_token = token
                    last_center_x = float(accepted.center_x)
                    last_received_s = float(accepted.received_monotonic_s)

                    error_slope = gate1_recenter_absolute_error_slope_px_s(
                        fresh_error_samples
                    )
                    error_decreased = bool(
                        current_abs_error_px < entry_abs_error_px
                        and error_slope is not None
                        and error_slope < 0.0
                    )
                    if (
                        error_decreased
                        and corridor_hold_frames
                        >= GATE1_RECENTER_REQUIRED_CORRIDOR_FRAMES
                    ):
                        self._watchdog(
                            require_target=False,
                            allow_benign_pad_contact=False,
                            enforce_benign_pad_budget=False,
                            count_rate_sample=False,
                        )
                        accepted_s = time.monotonic()
                        final_race = self.adapter.race_status
                        if final_race is not None:
                            max_gate_index = max(
                                max_gate_index,
                                int(final_race.active_gate_index),
                            )
                        if accepted_s >= hard_deadline_s:
                            raise SafetyAbort(
                                "gate-1 recenter hard 0.60s window expired"
                            )
                        if (
                            final_race is None
                            or bool(final_race.race_finished)
                            or int(final_race.active_gate_index) != 1
                            or int(final_race.sim_boot_time_ms)
                            < proof.post_gate_race_boot_ms
                            or self._latest_detection_generation
                            != entry_generation
                        ):
                            raise SafetyAbort(
                                "gate-1 recenter authority changed at acceptance"
                            )
                        await self._wait_for_next_flight_command_slot()
                        self._sample()
                        self._watchdog(
                            require_target=False,
                            allow_benign_pad_contact=False,
                            enforce_benign_pad_budget=False,
                            count_rate_sample=False,
                        )
                        cleanup_ready_race = self.adapter.race_status
                        cleanup_ready_target = self._latest_accepted_target
                        cleanup_ready_s = time.monotonic()
                        cleanup_ready_token_is_valid = False
                        if cleanup_ready_target is not None:
                            cleanup_ready_token = (
                                int(entry_generation),
                                int(cleanup_ready_target.frame_id),
                                int(cleanup_ready_target.sim_time_ns),
                            )
                            cleanup_ready_token_is_valid = bool(
                                (
                                    cleanup_ready_token == last_token
                                    and float(
                                        cleanup_ready_target.received_monotonic_s
                                    )
                                    == last_received_s
                                )
                                or (
                                    cleanup_ready_token[1] > last_token[1]
                                    and cleanup_ready_token[2] > last_token[2]
                                    and float(
                                        cleanup_ready_target.received_monotonic_s
                                    )
                                    > last_received_s
                                )
                            )
                        if cleanup_ready_race is not None:
                            max_gate_index = max(
                                max_gate_index,
                                int(cleanup_ready_race.active_gate_index),
                            )
                        if (
                            cleanup_ready_race is None
                            or bool(cleanup_ready_race.race_finished)
                            or int(cleanup_ready_race.active_gate_index) != 1
                            or int(cleanup_ready_race.sim_boot_time_ms)
                            < proof.post_gate_race_boot_ms
                            or self._latest_detection_generation
                            != entry_generation
                            or cleanup_ready_target is None
                            or not cleanup_ready_token_is_valid
                            or cleanup_ready_target.composite
                            or self.tracker.target is not cleanup_ready_target
                            or self.tracker.last_selection_mode != "primary"
                            or self.tracker.consecutive
                            < POST_GATE_REQUIRED_FRAMES
                            or cleanup_ready_target.frame_id
                            != self._latest_detection_frame_id
                            or cleanup_ready_target.sim_time_ns
                            != self._latest_detection_frame_sim_ns
                            or cleanup_ready_target.received_monotonic_s
                            != self._latest_detection_received_s
                            or cleanup_ready_target.age_s(cleanup_ready_s)
                            > MAX_VISION_AGE_S
                        ):
                            raise SafetyAbort(
                                "gate-1 recenter authority changed before cleanup"
                            )
                        self._assert_gate1_no_passage_geometry(
                            cleanup_ready_target,
                            phase="cleanup readiness",
                        )
                        if self.estimate is None:
                            raise SafetyAbort(
                                "gate-1 recenter attitude estimate changed "
                                "before cleanup"
                            )
                        cleanup_roll, cleanup_pitch, _cleanup_yaw = (
                            self.estimate.orientation.to_euler()
                        )
                        if self._gate1_yaw_reference_rad is not None:
                            self._gate1_yaw_envelope_state(
                                phase="Gate-1 recenter cleanup readiness",
                            )
                        min_roll = min(min_roll, float(cleanup_roll))
                        max_roll = max(max_roll, float(cleanup_roll))
                        min_pitch = min(min_pitch, float(cleanup_pitch))
                        max_pitch = max(max_pitch, float(cleanup_pitch))
                        cleanup_peak_rate = max(
                            abs(float(value))
                            for value in self.estimate.body_rates
                        )
                        if (
                            abs(float(cleanup_roll))
                            > GATE1_RECENTER_MAX_ABS_ROLL_RAD
                            or not (
                                GATE1_RECENTER_MIN_PITCH_RAD
                                <= float(cleanup_pitch)
                                <= GATE1_RECENTER_MAX_PITCH_RAD
                            )
                            or abs(float(cleanup_roll) - entry_roll)
                            > GATE1_RECENTER_MAX_ATTITUDE_EXCURSION_RAD
                            or abs(float(cleanup_pitch) - entry_pitch)
                            > GATE1_RECENTER_MAX_ATTITUDE_EXCURSION_RAD
                            or cleanup_peak_rate
                            > GATE1_RECENTER_MAX_MEASURED_BODY_RATE_RAD_S
                        ):
                            raise SafetyAbort(
                                "gate-1 recenter state approached its bound "
                                "before cleanup"
                            )
                        assert cleanup_ready_target is not None
                        assert cleanup_ready_token_is_valid
                        cleanup_target_advanced = cleanup_ready_token != last_token
                        latest_target = cleanup_ready_target
                        max_target_area = max(
                            max_target_area,
                            int(cleanup_ready_target.bbox_area),
                        )
                        max_target_width = max(
                            max_target_width,
                            int(cleanup_ready_target.bbox[2]),
                        )
                        if cleanup_target_advanced:
                            cleanup_dt_s = (
                                float(
                                    cleanup_ready_target.received_monotonic_s
                                )
                                - last_received_s
                            )
                            cleanup_x_rate_norm_s = (
                                (
                                    float(cleanup_ready_target.center_x)
                                    - last_center_x
                                )
                                / 320.0
                                / cleanup_dt_s
                            )
                            try:
                                gate1_recenter_roll_target(
                                    (
                                        float(cleanup_ready_target.center_x)
                                        - 320.0
                                    )
                                    / 320.0,
                                    cleanup_x_rate_norm_s,
                                    error_gain=(
                                        roll_control.gate1_error_gain
                                    ),
                                    error_rate_gain=(
                                        roll_control.gate1_error_rate_gain
                                    ),
                                    cap_rad=(
                                        roll_control.gate1_target_cap_rad
                                    ),
                                )
                            except ValueError as rate_exc:
                                raise SafetyAbort(
                                    "gate-1 recenter final horizontal rate is "
                                    "outside bounds"
                                ) from rate_exc
                            fresh_error_samples.append(
                                (
                                    float(
                                        cleanup_ready_target.received_monotonic_s
                                    ),
                                    abs(
                                        float(cleanup_ready_target.center_x)
                                        - 320.0
                                    ),
                                )
                            )
                        cleanup_abs_error_px = abs(
                            float(cleanup_ready_target.center_x) - 320.0
                        )
                        cleanup_error_slope = (
                            gate1_recenter_absolute_error_slope_px_s(
                                fresh_error_samples
                            )
                        )
                        if (
                            cleanup_abs_error_px >= entry_abs_error_px
                            or cleanup_error_slope is None
                            or cleanup_error_slope >= 0.0
                            or abs(
                                (
                                    float(cleanup_ready_target.center_x)
                                    - 320.0
                                )
                                / 320.0
                            )
                            > GATE1_RECENTER_CORRIDOR_NORMALIZED_X
                        ):
                            raise SafetyAbort(
                                "gate-1 recenter criteria changed before cleanup"
                            )
                        refresh_summary(
                            outcome="corridor_hold",
                            reason=None,
                            criteria_met=True,
                            corridor_accepted_elapsed_s=(
                                accepted_s - recenter_started_s
                            ),
                        )
                        self.recorder.emit(
                            "gate1_recenter_terminal",
                            **summary,
                        )
                        return dict(summary)

                normalized_x = (
                    float(latest_target.center_x) - 320.0
                ) / 320.0
                try:
                    target_roll = gate1_recenter_roll_target(
                        normalized_x,
                        current_x_rate_norm_s,
                        error_gain=roll_control.gate1_error_gain,
                        error_rate_gain=(
                            roll_control.gate1_error_rate_gain
                        ),
                        cap_rad=roll_control.gate1_target_cap_rad,
                    )
                except ValueError as exc:
                    raise SafetyAbort(
                        "gate-1 recenter horizontal control is outside bounds"
                    ) from exc
                command = attitude_rate_command(
                    self.estimate,
                    target_roll_rad=target_roll,
                    target_pitch_rad=(
                        forward_braking.gate1_target_pitch_rad
                    ),
                    thrust=forward_braking.gate1_forward_thrust,
                )
                command = limit_command_rates(
                    command,
                    roll_control.command_rate_cap_rad_s,
                )
                pitch_rate = max(
                    -forward_braking.pitch_command_rate_cap_rad_s,
                    min(
                        forward_braking.pitch_command_rate_cap_rad_s,
                        command.pitch_rate,
                    ),
                )
                try:
                    yaw_rate = (
                        0.0
                        if yaw_soft_stopped
                        else gate1_recenter_yaw_rate(
                            normalized_x,
                            error_gain=yaw_control.gate1_error_gain,
                            deadband_normalized_x=(
                                yaw_control.gate1_deadband_normalized_x
                            ),
                            cap_rad_s=(
                                yaw_control.command_rate_cap_rad_s
                            ),
                        )
                    )
                except ValueError as exc:
                    raise SafetyAbort(
                        "gate-1 recenter yaw control is outside bounds"
                    ) from exc
                command = AttitudeRateCommand(
                    roll_rate=command.roll_rate,
                    pitch_rate=pitch_rate,
                    yaw_rate=yaw_rate,
                    thrust=command.thrust,
                )
                validate_command(command)
                if (
                    abs(command.yaw_rate) > SIGN_ID_RATE_RAD_S
                    or abs(command.roll_rate)
                    > GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S
                    or abs(command.pitch_rate)
                    > GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S
                    or not (
                        GATE1_RECENTER_MIN_THRUST
                        <= command.thrust
                        <= GATE1_RECENTER_MAX_THRUST
                    )
                ):
                    raise SafetyAbort(
                        "gate-1 recenter command escaped the bounded envelope"
                    )

                self._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=False,
                    enforce_benign_pad_budget=False,
                    count_rate_sample=False,
                )
                send_checked_s = time.monotonic()
                send_race = self.adapter.race_status
                if send_race is not None:
                    max_gate_index = max(
                        max_gate_index,
                        int(send_race.active_gate_index),
                    )
                if (
                    hard_deadline_s - send_checked_s
                    <= CONTROL_PERIOD_S
                ):
                    raise SafetyAbort(
                        "gate-1 recenter hard 0.60s window expired"
                    )
                if (
                    send_race is None
                    or bool(send_race.race_finished)
                    or int(send_race.active_gate_index) != 1
                    or int(send_race.sim_boot_time_ms)
                    < proof.post_gate_race_boot_ms
                    or self._latest_detection_generation != entry_generation
                    or self._latest_accepted_target is not latest_target
                    or self.tracker.target is not latest_target
                    or latest_target.composite
                    or self.tracker.last_selection_mode != "primary"
                    or self.tracker.consecutive < POST_GATE_REQUIRED_FRAMES
                    or latest_target.frame_id
                    != self._latest_detection_frame_id
                    or latest_target.sim_time_ns
                    != self._latest_detection_frame_sim_ns
                    or latest_target.received_monotonic_s
                    != self._latest_detection_received_s
                    or latest_target.age_s(send_checked_s) > MAX_VISION_AGE_S
                ):
                    raise SafetyAbort(
                        "gate-1 recenter authority changed before command send"
                    )
                self._assert_gate1_no_passage_geometry(
                    latest_target,
                    phase="command send",
                )
                wire_not_before_ns = (
                    None
                    if self._last_flight_command_started_ns is None
                    else self._last_flight_command_started_ns
                    + round(CONTROL_PERIOD_S * 1_000_000_000)
                )
                last_dispatch_attempt_s = time.monotonic()
                try:
                    await self._send_flight_command(
                        command,
                        require_wire_receipt=True,
                        wire_start_not_before_ns=wire_not_before_ns,
                        wire_start_deadline_ns=last_recenter_wire_start_ns,
                    )
                    last_dispatch_attempt_s = None
                except Exception as exc:
                    raise SafetyAbort(
                        "gate-1 recenter command dispatch failed closed"
                    ) from exc
                command_count += 1
                min_command_roll_rate = (
                    command.roll_rate
                    if min_command_roll_rate is None
                    else min(min_command_roll_rate, command.roll_rate)
                )
                max_command_roll_rate = (
                    command.roll_rate
                    if max_command_roll_rate is None
                    else max(max_command_roll_rate, command.roll_rate)
                )
                min_command_pitch_rate = (
                    command.pitch_rate
                    if min_command_pitch_rate is None
                    else min(min_command_pitch_rate, command.pitch_rate)
                )
                max_command_pitch_rate = (
                    command.pitch_rate
                    if max_command_pitch_rate is None
                    else max(max_command_pitch_rate, command.pitch_rate)
                )
                min_command_yaw_rate = (
                    command.yaw_rate
                    if min_command_yaw_rate is None
                    else min(min_command_yaw_rate, command.yaw_rate)
                )
                max_command_yaw_rate = (
                    command.yaw_rate
                    if max_command_yaw_rate is None
                    else max(max_command_yaw_rate, command.yaw_rate)
                )
                self._record_tick(
                    "gate1-recenter/recenter",
                    send_checked_s - recenter_started_s,
                    command,
                )
                refresh_summary(outcome="running", reason=None)
                next_tick = next_control_deadline(
                    next_tick,
                    time.monotonic(),
                )
                await asyncio.sleep(
                    max(
                        0.0,
                        min(next_tick, hard_deadline_s) - time.monotonic(),
                    )
                )
        except BaseException as exc:
            pacing_failure: Optional[BaseException] = None
            terminal_observed_s = time.monotonic()
            if (
                self._last_flight_command_sent_s is not None
                or last_dispatch_attempt_s is not None
            ):
                try:
                    await reserve_terminal_cleanup_slot(
                        uncertain_dispatch_observed_s=(
                            terminal_observed_s
                            if last_dispatch_attempt_s is not None
                            else None
                        )
                    )
                except BaseException as reserve_exc:
                    pacing_failure = reserve_exc
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            "gate-1 recenter terminal cleanup-slot reservation "
                            f"also failed: {reserve_exc}"
                        )
            terminal_reason = str(exc) or type(exc).__name__
            if pacing_failure is not None:
                terminal_reason = (
                    f"{terminal_reason}; cleanup-slot reservation failed: "
                    f"{pacing_failure}"
                )
            refresh_summary(
                outcome=(
                    "abort"
                    if isinstance(exc, SafetyAbort)
                    else (
                        "interrupted"
                        if isinstance(exc, asyncio.CancelledError)
                        else "unexpected_error"
                    )
                ),
                reason=terminal_reason,
                criteria_met=False,
            )
            try:
                self.recorder.emit("gate1_recenter_terminal", **summary)
            except BaseException as recorder_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "gate-1 recenter terminal evidence emission also "
                        f"failed: {recorder_exc}"
                    )
            raise

    async def _run_gate1_recenter_candidate(
        self,
        context: StartContext,
    ) -> Dict[str, Any]:
        """Compose proved Gate 0, zero-authority acquisition, and the candidate."""

        gate0 = await self._run_gate0(
            context,
            crossing_hold_thrust=GATE1_RECENTER_TRANSITION_THRUST,
            course_line_preturn=(
                self.controller_config.turn_cue.preturn_enabled
            ),
            course_line_exit_counterroll_enabled=(
                self.controller_config.turn_cue.exit_counterroll_enabled
            ),
        )
        observation = await self._observe_gate1(
            gate0,
            hold_thrust=GATE1_RECENTER_TRANSITION_THRUST,
        )
        recenter = await self._run_bounded_gate1_recenter(observation)
        return {
            "gate0": gate0,
            "gate1_observation": observation,
            "gate1_recenter": recenter,
        }

    @staticmethod
    def _official_lap_time_s(race: Any) -> float:
        start_ms = int(race.race_start_boot_time_ms)
        finish_ns = int(race.race_finish_time_ns)
        if start_ms < 0 or finish_ns < 0:
            raise SafetyAbort("authoritative lap timing is incomplete")
        elapsed_ns = finish_ns - start_ms * 1_000_000
        if elapsed_ns < 0:
            raise SafetyAbort("authoritative lap finish predates race start")
        return elapsed_ns / 1_000_000_000.0

    async def _acquire_course_gate(
        self,
        expected_gate_index: int,
        *,
        transition_race_boot_ms: int,
        lap_deadline_s: float,
    ) -> Dict[str, Any]:
        """Retire the crossed target and acquire one strictly newer gate."""

        if not 1 <= expected_gate_index <= FULL_LAP_MAX_GATE_INDEX:
            raise SafetyAbort(
                f"course gate index is outside the bounded campaign ({expected_gate_index})"
            )
        watermark = self.vision.snapshot(max_age_s=MAX_VISION_AGE_S)
        if watermark is None:
            raise SafetyAbort("camera unavailable at course-gate handoff")
        self._last_frame_identity = (
            int(watermark.generation),
            int(watermark.frame_id),
        )
        self._last_frame_sim_ns = int(watermark.sim_time_ns)
        self._latest_detection_frame_id = int(watermark.frame_id)
        self._latest_detection_frame_sim_ns = int(watermark.sim_time_ns)
        self._latest_detection_generation = int(watermark.generation)
        self._latest_detection_received_s = float(watermark.received_monotonic_s)
        self._latest_raw_detections = []
        self._latest_accepted_target = None
        self.tracker.reset()
        self._post_gate_reacquisition = True

        started_s = time.monotonic()
        hard_deadline_s = min(
            started_s + POST_GATE_OBSERVATION_TIMEOUT_S,
            float(lap_deadline_s),
        )
        if started_s >= hard_deadline_s:
            self._post_gate_reacquisition = False
            raise SafetyAbort("course-gate acquisition has no remaining safety budget")
        next_tick = max(
            started_s,
            (
                self._last_flight_command_sent_s + CONTROL_PERIOD_S
                if self._last_flight_command_sent_s is not None
                else started_s
            ),
        )
        last_token = (
            int(watermark.generation),
            int(watermark.frame_id),
            int(watermark.sim_time_ns),
        )
        frames: List[Dict[str, Any]] = []
        transition_command = AttitudeRateCommand(
            0.0,
            0.0,
            0.0,
            COURSE_TRANSITION_THRUST,
        )
        self.recorder.emit(
            "course_gate_acquisition_started",
            expected_gate_index=expected_gate_index,
            transition_race_boot_ms=int(transition_race_boot_ms),
            hard_deadline_monotonic_s=hard_deadline_s,
            watermark={
                "generation": watermark.generation,
                "frame_id": watermark.frame_id,
                "sim_time_ns": watermark.sim_time_ns,
                "received_monotonic_s": watermark.received_monotonic_s,
            },
        )
        try:
            initial_wait = min(next_tick, hard_deadline_s) - time.monotonic()
            if initial_wait > 0.0:
                await asyncio.sleep(initial_wait)
            while True:
                if time.monotonic() >= hard_deadline_s:
                    raise SafetyAbort(
                        f"gate {expected_gate_index} acquisition timed out"
                    )
                self._sample()
                self._watchdog(
                    require_target=False,
                    allow_benign_pad_contact=False,
                    enforce_benign_pad_budget=False,
                )
                now = time.monotonic()
                race = self.adapter.race_status
                if race is None:
                    raise SafetyAbort("race status unavailable during gate acquisition")
                if int(race.sim_boot_time_ms) < int(transition_race_boot_ms):
                    raise SafetyAbort("race clock regressed during gate acquisition")
                if (
                    race.race_finished
                    and int(race.sim_boot_time_ms) > int(transition_race_boot_ms)
                ):
                    return {
                        "race_finished": True,
                        "gate_index": int(race.active_gate_index),
                        "race_boot_ms": int(race.sim_boot_time_ms),
                        "race_finish_time_ns": int(race.race_finish_time_ns),
                        "official_lap_time_s": self._official_lap_time_s(race),
                        "acquisition_elapsed_s": now - started_s,
                        "frame_count": len(frames),
                    }
                if int(race.active_gate_index) != expected_gate_index:
                    raise SafetyAbort(
                        "gate index changed during acquisition "
                        f"({race.active_gate_index}, expected {expected_gate_index})"
                    )

                frame_token = self._latest_frame_token()
                if frame_token is not None and frame_token != last_token:
                    generation, frame_id, sim_time_ns = frame_token
                    received_s = self._latest_detection_received_s
                    if generation != int(watermark.generation):
                        raise SafetyAbort("vision generation changed during gate acquisition")
                    if (
                        frame_id <= int(watermark.frame_id)
                        or sim_time_ns <= int(watermark.sim_time_ns)
                        or received_s is None
                        or received_s <= float(watermark.received_monotonic_s)
                    ):
                        raise SafetyAbort("course camera frame did not advance strictly")
                    last_token = frame_token
                    accepted = self._latest_accepted_target
                    if accepted is None or is_crossing_residue(accepted):
                        self.tracker.reset()
                        frames = []
                    else:
                        record = {
                            "frame_id": accepted.frame_id,
                            "sim_time_ns": accepted.sim_time_ns,
                            "received_monotonic_s": accepted.received_monotonic_s,
                            "center_px": [accepted.center_x, accepted.center_y],
                            "bbox_xywh_px": list(accepted.bbox),
                            "confidence": accepted.confidence,
                            "tracker_streak": self.tracker.consecutive,
                        }
                        frames = (
                            [record]
                            if self.tracker.consecutive == 1
                            else (frames + [record])[-self.tracker.consecutive :]
                        )
                        self.recorder.emit(
                            "course_gate_candidate_frame",
                            expected_gate_index=expected_gate_index,
                            **record,
                        )
                        if (
                            self.tracker.consecutive >= POST_GATE_REQUIRED_FRAMES
                            and len(frames) >= POST_GATE_REQUIRED_FRAMES
                            and accepted.age_s(time.monotonic()) <= MAX_VISION_AGE_S
                        ):
                            final_race = self.adapter.race_status
                            if (
                                final_race is None
                                or int(final_race.active_gate_index)
                                != expected_gate_index
                                or int(final_race.sim_boot_time_ms)
                                < int(transition_race_boot_ms)
                            ):
                                raise SafetyAbort(
                                    "race status changed at gate acquisition acceptance"
                                )
                            result = {
                                "race_finished": False,
                                "gate_index": expected_gate_index,
                                "race_boot_ms": int(final_race.sim_boot_time_ms),
                                "acquisition_elapsed_s": time.monotonic() - started_s,
                                "frame_count": POST_GATE_REQUIRED_FRAMES,
                                "frames": frames[-POST_GATE_REQUIRED_FRAMES:],
                                "initial_gate_area_px": int(accepted.bbox_area),
                                "final_gate_bbox": list(accepted.bbox),
                                "final_gate_center": [
                                    accepted.center_x,
                                    accepted.center_y,
                                ],
                            }
                            self.recorder.emit("course_gate_acquired", **result)
                            return result

                if hard_deadline_s - time.monotonic() <= CONTROL_PERIOD_S:
                    raise SafetyAbort(
                        f"gate {expected_gate_index} acquisition timed out"
                    )
                await self._send_flight_command(transition_command)
                self._record_tick(
                    f"full-lap/gate{expected_gate_index}/acquire",
                    time.monotonic() - started_s,
                    transition_command,
                )
                next_tick = next_control_deadline(next_tick, time.monotonic())
                await asyncio.sleep(
                    max(0.0, min(next_tick, hard_deadline_s) - time.monotonic())
                )
        finally:
            self._post_gate_reacquisition = False

    def _complete_course_gate(
        self,
        *,
        race: Any,
        expected_gate_index: int,
        gate_started_s: float,
        lap_started_s: float,
        crossing_started_s: Optional[float],
        max_gate_area: int,
        acquisition_gate_area: int,
    ) -> Dict[str, Any]:
        finished = bool(race.race_finished)
        active_gate_index = int(race.active_gate_index)
        if not finished and active_gate_index != expected_gate_index + 1:
            raise SafetyAbort("course transition did not advance exactly one gate")
        now = time.monotonic()
        result: Dict[str, Any] = {
            "gate_index": expected_gate_index,
            "next_gate_index": None if finished else active_gate_index,
            "race_finished": finished,
            "race_boot_ms": int(race.sim_boot_time_ms),
            "race_finish_time_ns": int(race.race_finish_time_ns),
            "last_gate_race_time": int(race.last_gate_race_time),
            "gate_elapsed_s": now - gate_started_s,
            "lap_elapsed_s": now - lap_started_s,
            "crossing_confirmation_used": crossing_started_s is not None,
            "crossing_confirmation_elapsed_s": (
                now - crossing_started_s
                if crossing_started_s is not None
                else None
            ),
            "acquisition_gate_area_px": int(acquisition_gate_area),
            "max_gate_area_px": int(max_gate_area),
        }
        if finished:
            result["official_lap_time_s"] = self._official_lap_time_s(race)
        self.recorder.emit("course_gate_pass_proved", **result)
        return result

    async def _run_course_gate(
        self,
        expected_gate_index: int,
        *,
        acquisition: Dict[str, Any],
        lap_started_s: float,
        lap_deadline_s: float,
    ) -> Dict[str, Any]:
        """Visually center and pass one unknown-map course gate."""

        race = self.adapter.race_status
        if race is None or int(race.active_gate_index) != expected_gate_index:
            raise SafetyAbort(
                f"gate {expected_gate_index} approach lacks matching race status"
            )
        target = self.tracker.target
        if target is None or self.tracker.consecutive < POST_GATE_REQUIRED_FRAMES:
            raise SafetyAbort(
                f"gate {expected_gate_index} approach lacks a confirmed target"
            )
        assert self.estimate is not None
        _entry_roll, recenter_pitch_basis, _entry_yaw = (
            self.estimate.orientation.to_euler()
        )
        acquisition_gate_area = int(
            acquisition.get("initial_gate_area_px", target.bbox_area)
        )
        gate_started_s = await self._wait_for_next_flight_command_slot()
        gate_deadline_s = min(
            gate_started_s + COURSE_GATE_TIMEOUT_S,
            float(lap_deadline_s),
        )
        next_tick = max(
            gate_started_s,
            (
                self._last_flight_command_sent_s + CONTROL_PERIOD_S
                if self._last_flight_command_sent_s is not None
                else gate_started_s
            ),
        )
        max_gate_area = max(acquisition_gate_area, target.bbox_area)
        last_target_frame: Optional[int] = None
        last_course_line_frame: Optional[int] = None
        last_control_y: Optional[float] = None
        last_target_time: Optional[float] = None
        control_y_rate = 0.0
        crossing_armed = False
        crossing_started_s: Optional[float] = None
        crossing_race_boot_ms: Optional[int] = None
        last_current_gate_race_boot_ms = int(race.sim_boot_time_ms)
        course_vision_generation: Optional[int] = None
        while True:
            now = time.monotonic()
            if now >= gate_deadline_s:
                raise SafetyAbort(f"gate {expected_gate_index} wall-time limit reached")
            self._sample()
            latest_vision_generation = self._latest_detection_generation
            if (
                type(latest_vision_generation) is not int
                or latest_vision_generation < 0
            ):
                raise SafetyAbort("course vision generation is unavailable")
            if course_vision_generation is None:
                course_vision_generation = latest_vision_generation
            elif latest_vision_generation != course_vision_generation:
                raise SafetyAbort("vision generation changed during course gate")
            race = self.adapter.race_status
            assert race is not None and self.estimate is not None
            if (
                int(race.active_gate_index) == expected_gate_index
                and not race.race_finished
            ):
                last_current_gate_race_boot_ms = int(race.sim_boot_time_ms)
            target = self.tracker.target
            assert target is not None
            elapsed = now - gate_started_s
            normalized_x = (target.center_x - 320.0) / 320.0
            if (
                self._latest_detection_frame_id is not None
                and self._latest_detection_frame_id != last_course_line_frame
            ):
                last_course_line_frame = self._latest_detection_frame_id
                course_line = cyan_course_line_observation(
                    self._latest_detection_image,
                )
                self.recorder.emit(
                    "course_gate_line_observation",
                    expected_gate_index=expected_gate_index,
                    elapsed_s=elapsed,
                    frame_id=last_course_line_frame,
                    observation=(
                        asdict(course_line) if course_line is not None else None
                    ),
                )
            max_gate_area = max(max_gate_area, target.bbox_area)
            if elapsed > 5.0 and max_gate_area < 1.25 * acquisition_gate_area:
                raise SafetyAbort(
                    f"no visual approach progress toward gate {expected_gate_index}"
                )

            control_y = gate_control_center_y_px(
                target,
                previous_center_y=last_control_y,
            )
            if (
                not crossing_armed
                and target.age_s(now) <= CROSSING_TARGET_LOSS_S
                and self.tracker.consecutive >= POST_GATE_REQUIRED_FRAMES
                and int(race.active_gate_index) == expected_gate_index
                and not race.race_finished
                and is_course_gate_crossing_candidate(
                    target,
                    acquisition_gate_area=acquisition_gate_area,
                    control_y=control_y,
                )
            ):
                crossing_armed = True
                self._course_edge_continuation_gate_index = None
                self.recorder.emit(
                    "course_crossing_candidate_armed",
                    expected_gate_index=expected_gate_index,
                    elapsed_s=elapsed,
                    race_boot_ms=int(race.sim_boot_time_ms),
                    target=asdict(target),
                    control_y=control_y,
                )
            crossing_confirming = bool(
                crossing_started_s is not None
                or (
                    crossing_armed
                    and target.age_s(now) > CROSSING_TARGET_LOSS_S
                )
            )
            self._watchdog(
                require_target=not (
                    crossing_confirming
                    or race.race_finished
                    or int(race.active_gate_index) == expected_gate_index + 1
                ),
                allow_benign_pad_contact=False,
                enforce_benign_pad_budget=False,
            )
            active_gate_index = int(race.active_gate_index)
            if (
                not crossing_armed
                and not crossing_confirming
                and active_gate_index == expected_gate_index
                and not race.race_finished
                and self._latest_detection_frame_sim_ns is not None
                and self._latest_detection_received_s is not None
                and self._latest_detection_frame_sim_ns > target.sim_time_ns
                and self._latest_detection_received_s
                > target.received_monotonic_s
                and 0.0
                <= now - self._latest_detection_received_s
                <= MAX_VISION_AGE_S
            ):
                contact_risk = select_untracked_contact_risk(
                    self._latest_raw_detections,
                    accepted_target=self._latest_accepted_target,
                )
                if contact_risk is not None:
                    detector_index = next(
                        (
                            index
                            for index, detection in enumerate(
                                self._latest_raw_detections
                            )
                            if detection is contact_risk
                        ),
                        -1,
                    )
                    self.recorder.emit(
                        "course_untracked_contact_risk",
                        expected_gate_index=expected_gate_index,
                        elapsed_s=elapsed,
                        generation=self._latest_detection_generation,
                        frame_id=self._latest_detection_frame_id,
                        sim_time_ns=self._latest_detection_frame_sim_ns,
                        receive_age_s=(
                            now - self._latest_detection_received_s
                        ),
                        detection=gate_detection_summary(
                            contact_risk,
                            detector_index=detector_index,
                        ),
                    )
                    raise SafetyAbort(
                        f"large untracked gate geometry at gate "
                        f"{expected_gate_index}"
                    )
            if not crossing_confirming:
                if (
                    int(race.sim_boot_time_ms) > last_current_gate_race_boot_ms
                    and (
                        race.race_finished
                        or active_gate_index == expected_gate_index + 1
                    )
                ):
                    return self._complete_course_gate(
                        race=race,
                        expected_gate_index=expected_gate_index,
                        gate_started_s=gate_started_s,
                        lap_started_s=lap_started_s,
                        crossing_started_s=None,
                        max_gate_area=max_gate_area,
                        acquisition_gate_area=acquisition_gate_area,
                    )
                if active_gate_index != expected_gate_index:
                    raise SafetyAbort(
                        "unexpected course gate-index transition "
                        f"{expected_gate_index}->{active_gate_index}"
                    )

            if crossing_confirming:
                if crossing_started_s is None:
                    crossing_started_s = now
                    crossing_race_boot_ms = last_current_gate_race_boot_ms
                    self.recorder.emit(
                        "course_crossing_confirmation_started",
                        expected_gate_index=expected_gate_index,
                        elapsed_s=elapsed,
                        baseline_race_boot_ms=crossing_race_boot_ms,
                        target_age_s=target.age_s(now),
                    )
                assert crossing_race_boot_ms is not None
                decision = full_lap_crossing_status_decision(
                    baseline_race_boot_ms=crossing_race_boot_ms,
                    current_race_boot_ms=int(race.sim_boot_time_ms),
                    expected_gate_index=expected_gate_index,
                    active_gate_index=active_gate_index,
                    race_finished=bool(race.race_finished),
                    elapsed_s=now - crossing_started_s,
                )
                if decision != "waiting":
                    self.recorder.emit(
                        "course_crossing_status_decision",
                        expected_gate_index=expected_gate_index,
                        decision=decision,
                        baseline_race_boot_ms=crossing_race_boot_ms,
                        current_race_boot_ms=int(race.sim_boot_time_ms),
                        gate_index=active_gate_index,
                        race_finished=bool(race.race_finished),
                    )
                    if decision in {"passed", "finished"}:
                        return self._complete_course_gate(
                            race=race,
                            expected_gate_index=expected_gate_index,
                            gate_started_s=gate_started_s,
                            lap_started_s=lap_started_s,
                            crossing_started_s=crossing_started_s,
                            max_gate_area=max_gate_area,
                            acquisition_gate_area=acquisition_gate_area,
                        )
                    raise SafetyAbort(
                        f"gate {expected_gate_index} crossing "
                        f"{decision.replace('_', ' ')}"
                    )
                command = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
                await self._send_flight_command(command)
                self._record_tick(
                    f"full-lap/gate{expected_gate_index}/confirm",
                    elapsed,
                    command,
                )
                next_tick = next_control_deadline(next_tick, time.monotonic())
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            if target.frame_id != last_target_frame:
                if last_control_y is not None and last_target_time is not None:
                    dt_target = target.received_monotonic_s - last_target_time
                    if dt_target > 1e-3:
                        raw_rate = (control_y - last_control_y) / dt_target
                        raw_rate = max(-300.0, min(300.0, raw_rate))
                        control_y_rate = 0.65 * control_y_rate + 0.35 * raw_rate
                last_target_frame = target.frame_id
                last_control_y = control_y
                last_target_time = target.received_monotonic_s

            recenter_required = course_gate_recenter_required(
                elapsed,
                normalized_x,
                control_y,
            )
            if recenter_required:
                target_roll = course_gate_roll_target(normalized_x, recenter=True)
                target_pitch = course_gate_recenter_pitch_target(
                    recenter_pitch_basis,
                    control_y,
                )
                # Crossing confirmation necessarily commands zero thrust.  Use
                # the existing hard thrust ceiling briefly after each credit to
                # arrest the observed sink before accelerating at the next gate.
                thrust = (
                    COURSE_RECENTER_THRUST
                    if elapsed < COURSE_RECENTER_DURATION_S
                    or control_y < COURSE_HIGH_GATE_Y_PX
                    else gate_vertical_thrust(control_y, control_y_rate)
                )
                phase = "recenter"
            else:
                normalized_y = (control_y - 180.0) / 180.0
                image_error = max(abs(normalized_x), abs(normalized_y))
                speed_scale = max(0.15, min(1.0, (0.90 - image_error) / 0.70))
                target_roll = course_gate_roll_target(normalized_x, recenter=False)
                target_pitch = COURSE_APPROACH_PITCH_RAD * speed_scale
                thrust = gate_vertical_thrust(control_y, control_y_rate)
                if control_y < COURSE_HIGH_GATE_Y_PX:
                    # A high, top-clipped target coincided with residual sink in
                    # live traces.  Preserve recovery authority until it returns
                    # to a measurable vertical corridor.
                    thrust = COURSE_RECENTER_THRUST
                phase = "approach"

            if (
                target.bbox_area >= 8 * acquisition_gate_area
                and abs(control_y - 180.0) > 75.0
            ):
                raise SafetyAbort(
                    f"gate {expected_gate_index} close approach outside vertical "
                    f"corridor ({control_y:.1f}px)"
                )
            command = attitude_rate_command(
                self.estimate,
                target_roll_rad=target_roll,
                target_pitch_rad=target_pitch,
                thrust=thrust,
            )
            if phase == "recenter":
                command = course_recenter_rate_command(command)
            await self._send_flight_command(command)
            self._record_tick(
                f"full-lap/gate{expected_gate_index}/{phase}",
                elapsed,
                command,
            )
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def _align_gate0_race_phase(self) -> Dict[str, Any]:
        """Hold exact zero briefly so Gate 0 credit lands on the next packet."""

        received = getattr(self.adapter, "latest_received_race_status", None)
        if received is None:
            result = {"applied": False, "reason": "no exact race ingress"}
            self.recorder.emit("gate0_phase_alignment_skipped", **result)
            return result
        try:
            received.validate_integrity()
            race_received_ns = int(received.ingress.received_monotonic_ns)
            race_sequence = int(received.ingress.sequence)
            race_boot_ms = int(received.race_status.sim_boot_time_ms)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SafetyAbort("gate-0 phase alignment race ingress is invalid") from exc

        planned_delay_s = gate0_phase_alignment_delay_s(
            now_monotonic_ns=time.perf_counter_ns(),
            last_race_received_monotonic_ns=race_received_ns,
        )
        if planned_delay_s <= 0.001:
            result = {
                "applied": False,
                "reason": "already aligned",
                "planned_delay_s": planned_delay_s,
                "race_sequence": race_sequence,
                "race_boot_ms": race_boot_ms,
            }
            self.recorder.emit("gate0_phase_alignment_skipped", **result)
            return result

        started_s = time.monotonic()
        deadline_s = started_s + planned_delay_s
        next_tick = started_s
        zero = AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
        self.recorder.emit(
            "gate0_phase_alignment_started",
            planned_delay_s=planned_delay_s,
            race_sequence=race_sequence,
            race_boot_ms=race_boot_ms,
            expected_target_loss_s=COURSE_GATE0_EXPECTED_TARGET_LOSS_S,
            target_lead_s=COURSE_RACE_PACKET_TARGET_LEAD_S,
        )
        command_count = 0
        while time.monotonic() < deadline_s:
            self._sample()
            self._watchdog(
                require_target=True,
                allow_benign_pad_contact=True,
                enforce_benign_pad_budget=True,
            )
            race = self.adapter.race_status
            if race is None or int(race.active_gate_index) != 0 or race.race_finished:
                raise SafetyAbort("race state changed during gate-0 phase alignment")
            await self._send_flight_command(zero)
            self._record_tick(
                "full-lap/gate0/phase-align",
                time.monotonic() - started_s,
                zero,
            )
            command_count += 1
            next_tick = next_control_deadline(next_tick, time.monotonic())
            await asyncio.sleep(
                max(0.0, min(next_tick, deadline_s) - time.monotonic())
            )
        result = {
            "applied": True,
            "planned_delay_s": planned_delay_s,
            "actual_delay_s": time.monotonic() - started_s,
            "command_count": command_count,
            "race_sequence": race_sequence,
            "race_boot_ms": race_boot_ms,
        }
        self.recorder.emit("gate0_phase_alignment_complete", **result)
        return result

    async def _run_full_lap(self, context: StartContext) -> Dict[str, Any]:
        """Exercise retained full-lap orchestration as offline-only scaffolding.

        No live dispatcher admits this method. The unaccepted Gate 0 phase
        alignment, launch overrides, and cyan-line actuation are deliberately
        disabled even for direct test calls.
        """

        lap_started_s = time.monotonic()
        lap_deadline_s = lap_started_s + FULL_LAP_TIMEOUT_S
        phase_alignment = {
            "applied": False,
            "reason": "offline scaffold disables unaccepted race-phase actuation",
        }
        gate0 = await self._run_gate0(context)
        gate1_observation = await self._observe_gate1(gate0)
        final_bbox = gate1_observation["final_gate_bbox"]
        acquisition: Dict[str, Any] = {
            **gate1_observation,
            "initial_gate_area_px": int(final_bbox[2]) * int(final_bbox[3]),
            "race_finished": False,
        }
        acquisitions: List[Dict[str, Any]] = [dict(acquisition)]
        gates: List[Dict[str, Any]] = []
        expected_gate_index = 1
        while True:
            if time.monotonic() >= lap_deadline_s:
                raise SafetyAbort("full-lap wall-time limit reached")
            self._course_edge_continuation_gate_index = expected_gate_index
            try:
                gate_result = await self._run_course_gate(
                    expected_gate_index,
                    acquisition=acquisition,
                    lap_started_s=lap_started_s,
                    lap_deadline_s=lap_deadline_s,
                )
            finally:
                self._course_edge_continuation_gate_index = None
            gates.append(gate_result)
            if gate_result["race_finished"]:
                return {
                    "race_finished": True,
                    "official_lap_time_s": gate_result["official_lap_time_s"],
                    "race_finish_time_ns": gate_result["race_finish_time_ns"],
                    "highest_gate_index": expected_gate_index,
                    "proved_transition_count": 1 + len(gates),
                    "gate0_phase_alignment": phase_alignment,
                    "gate0": gate0,
                    "gate1_observation": gate1_observation,
                    "acquisitions": acquisitions,
                    "course_gates": gates,
                }
            expected_gate_index = int(gate_result["next_gate_index"])
            if expected_gate_index > FULL_LAP_MAX_GATE_INDEX:
                raise SafetyAbort("full-lap exceeded the bounded gate-index limit")
            acquisition = await self._acquire_course_gate(
                expected_gate_index,
                transition_race_boot_ms=int(gate_result["race_boot_ms"]),
                lap_deadline_s=lap_deadline_s,
            )
            acquisitions.append(dict(acquisition))
            if acquisition["race_finished"]:
                return {
                    "race_finished": True,
                    "official_lap_time_s": acquisition["official_lap_time_s"],
                    "race_finish_time_ns": acquisition["race_finish_time_ns"],
                    "highest_gate_index": int(acquisition["gate_index"]),
                    "proved_transition_count": 1 + len(gates),
                    "gate0_phase_alignment": phase_alignment,
                    "gate0": gate0,
                    "gate1_observation": gate1_observation,
                    "acquisitions": acquisitions,
                    "course_gates": gates,
                }

    async def run_powered_stage(
        self,
        stage: str,
        *,
        write_diagnostic_pngs: bool = True,
    ) -> StageResult:
        if type(write_diagnostic_pngs) is not bool:
            raise TypeError("write_diagnostic_pngs must be an exact bool")
        if stage not in {
            "sign-id",
            "hover",
            "gate0",
            "gate0-observe",
            GATE1_RECENTER_STAGE,
            *VISUAL_POWERED_STAGES,
            CALIBRATION_STAGE,
        }:
            raise ValueError(f"unsupported powered stage: {stage}")
        started = time.monotonic()
        reason = "unknown"
        success = False
        details: Dict[str, Any] = {}
        gate_before: Optional[int] = None
        gate_after: Optional[int] = None
        cleanup_confirmed = False
        try:
            self._deferred_pngs = []
            self._post_gate_last_frame = None
            self._abort_latched = False
            self._gate1_yaw_reference_rad = None
            self._gate1_max_abs_yaw_excursion_rad = 0.0
            self._gate1_max_abs_measured_yaw_rate_rad_s = 0.0
            self._visual_tracking_enabled = stage in VISUAL_POWERED_STAGES
            self._visual_diagnostic_logging = stage in VISUAL_POWERED_STAGES
            self._visual_active_stage = (
                stage if stage in VISUAL_POWERED_STAGES else None
            )
            self._visual_reset_epoch = 0
            self._visual_shadow_summary = None
            self._visual_alignment_summary = None
            self._visual_gate0_blend_summary = None
            await self.establish_reset_epoch(restart_vision=True)
            await self.normalize_disarmed()
            context = await self.wait_for_go()
            race = self.adapter.race_status
            gate_before = race.active_gate_index if race else None
            if stage in VISUAL_POWERED_STAGES:
                self._bind_initial_visual_gate(context)
            await self.arm_confirmed()
            if stage == "sign-id":
                details = await self._run_sign_id()
            elif stage == "hover":
                details = await self._run_hover(context)
            elif stage == CALIBRATION_STAGE:
                details = await self._run_calibration_excite(context)
            elif stage == "gate0":
                details = await self._run_gate0(context)
            elif stage == "gate0-observe":
                gate0_details = await self._run_gate0(
                    context,
                    capture_transition=write_diagnostic_pngs,
                )
                details = {"gate0": gate0_details}
                try:
                    details["gate1_observation"] = await self._observe_gate1(
                        gate0_details
                    )
                except SafetyAbort as exc:
                    details["gate1_observation"] = {
                        "gate1_observed": False,
                        "reason": str(exc),
                    }
                    raise
            elif stage == VISUAL_SHADOW_STAGE:
                details = {
                    "visual_shadow": await self._run_visual_shadow(context)
                }
            elif stage == VISUAL_ALIGN_STAGE:
                details = {
                    "visual_alignment": await self._run_visual_alignment(
                        context
                    )
                }
            elif stage == GATE1_RECENTER_STAGE:
                gate0_details = await self._run_gate0(
                    context,
                    capture_transition=write_diagnostic_pngs,
                    crossing_hold_thrust=(
                        GATE1_RECENTER_TRANSITION_THRUST
                    ),
                    course_line_preturn=(
                        self.controller_config.turn_cue.preturn_enabled
                    ),
                    course_line_exit_counterroll_enabled=(
                        self.controller_config.turn_cue
                        .exit_counterroll_enabled
                    ),
                )
                details = {"gate0": gate0_details}
                try:
                    observation = await self._observe_gate1(
                        gate0_details,
                        hold_thrust=GATE1_RECENTER_TRANSITION_THRUST,
                    )
                except SafetyAbort as exc:
                    details["gate1_observation"] = {
                        "gate1_observed": False,
                        "reason": str(exc),
                    }
                    raise
                details["gate1_observation"] = observation
                try:
                    recenter_result = (
                        await self._run_bounded_gate1_recenter(observation)
                    )
                    if not isinstance(recenter_result, Mapping):
                        raise SafetyAbort(
                            "gate-1 recenter returned an invalid result"
                        )
                    details["gate1_recenter"] = dict(recenter_result)
                    if (
                        recenter_result.get("recenter_criteria_met")
                        is not True
                    ):
                        if self._gate1_recenter_summary is None:
                            self._gate1_recenter_summary = dict(
                                recenter_result
                            )
                        raise SafetyAbort(
                            "gate-1 recenter returned without satisfying "
                            "its criteria"
                        )
                except BaseException as exc:
                    if self._gate1_recenter_summary is not None:
                        abort_summary = dict(self._gate1_recenter_summary)
                        if (
                            abort_summary.get("recenter_criteria_met")
                            is not True
                        ):
                            abort_summary["success"] = False
                            abort_summary["outcome"] = "abort"
                            abort_summary["reason"] = (
                                str(exc) or type(exc).__name__
                            )
                        self._gate1_recenter_summary = abort_summary
                        details["gate1_recenter"] = abort_summary
                    raise
            else:
                raise AssertionError("powered stage dispatch was not exhaustive")
            success = True
            reason = "stage completed"
        except (SafetyAbort, asyncio.CancelledError) as exc:
            if (
                stage == GATE1_RECENTER_STAGE
                and self._gate1_recenter_summary is not None
            ):
                details["gate1_recenter"] = dict(
                    self._gate1_recenter_summary
                )
            if (
                stage == VISUAL_SHADOW_STAGE
                and self._visual_shadow_summary is not None
            ):
                details["visual_shadow"] = dict(
                    self._visual_shadow_summary
                )
            if (
                stage == VISUAL_ALIGN_STAGE
                and self._visual_alignment_summary is not None
            ):
                details["visual_alignment"] = dict(
                    self._visual_alignment_summary
                )
            reason = str(exc) or type(exc).__name__
            logger.error("%s ABORT: %s", stage, reason)
            self.recorder.emit("stage_abort", stage=stage, reason=reason)
            if isinstance(exc, asyncio.CancelledError):
                raise
        except Exception as exc:
            if (
                stage == GATE1_RECENTER_STAGE
                and self._gate1_recenter_summary is not None
            ):
                details["gate1_recenter"] = dict(
                    self._gate1_recenter_summary
                )
            if (
                stage == VISUAL_SHADOW_STAGE
                and self._visual_shadow_summary is not None
            ):
                details["visual_shadow"] = dict(
                    self._visual_shadow_summary
                )
            if (
                stage == VISUAL_ALIGN_STAGE
                and self._visual_alignment_summary is not None
            ):
                details["visual_alignment"] = dict(
                    self._visual_alignment_summary
                )
            reason = f"unexpected {type(exc).__name__}: {exc}"
            logger.exception("%s failed unexpectedly", stage)
            self.recorder.emit("stage_abort", stage=stage, reason=reason)
        finally:
            cleanup_entry_race = self.adapter.race_status
            cleanup_entry_gate_index = (
                int(cleanup_entry_race.active_gate_index)
                if cleanup_entry_race is not None
                else None
            )
            cleanup_entry_race_finished = (
                bool(cleanup_entry_race.race_finished)
                if cleanup_entry_race is not None
                else None
            )
            if stage in VISUAL_POWERED_STAGES:
                details["authoritative_cleanup_entry"] = {
                    "gate_index": cleanup_entry_gate_index,
                    "race_finished": cleanup_entry_race_finished,
                    "transition": (
                        None
                        if self._visual_transition is None
                        else [
                            self._visual_transition.from_gate_index,
                            self._visual_transition.to_gate_index,
                        ]
                    ),
                }
                if success and (
                    cleanup_entry_gate_index != 1
                    or cleanup_entry_race_finished is not False
                    or self._visual_transition is None
                    or self._visual_transition.from_gate_index != 0
                    or self._visual_transition.to_gate_index != 1
                    or (
                        stage == VISUAL_ALIGN_STAGE
                        and (
                            self._visual_alignment_summary is None
                            or self._visual_alignment_summary.get(
                                "promoted_current_track_id"
                            )
                            != self._visual_transition.promoted_track_id
                            or self._visual_alignment_summary.get(
                                "alignment_criteria_met"
                            )
                            is not True
                        )
                    )
                ):
                    success = False
                    boundary_reason = (
                        f"{stage} cleanup boundary lacks proved 0->1 "
                        f"authority (gate_index={cleanup_entry_gate_index}, "
                        f"race_finished={cleanup_entry_race_finished})"
                    )
                    reason = (
                        boundary_reason
                        if reason in {"unknown", "stage completed"}
                        else f"{reason}; {boundary_reason}"
                    )
                    self.recorder.emit(
                        "stage_abort",
                        stage=stage,
                        reason=boundary_reason,
                    )
                    if (
                        stage == VISUAL_ALIGN_STAGE
                        and self._visual_alignment_summary is not None
                    ):
                        alignment_boundary = dict(
                            self._visual_alignment_summary
                        )
                        alignment_boundary["success"] = False
                        alignment_boundary["outcome"] = "abort"
                        alignment_boundary["reason"] = boundary_reason
                        alignment_boundary["abort_outcome"] = (
                            boundary_reason
                        )
                        self._visual_alignment_summary = (
                            alignment_boundary
                        )
            if (
                stage == GATE1_RECENTER_STAGE
                and success
                and (
                cleanup_entry_gate_index != 1
                or cleanup_entry_race_finished is not False
                )
            ):
                success = False
                boundary_reason = (
                    "gate-1 recenter cleanup boundary lost gate 1 authority "
                    f"(gate_index={cleanup_entry_gate_index}, "
                    f"race_finished={cleanup_entry_race_finished})"
                )
                reason = (
                    boundary_reason
                    if reason in {"unknown", "stage completed"}
                    else f"{reason}; {boundary_reason}"
                )
                if self._gate1_recenter_summary is not None:
                    boundary_summary = dict(self._gate1_recenter_summary)
                    prior_max_gate = boundary_summary.get(
                        "authoritative_max_gate_index"
                    )
                    if cleanup_entry_gate_index is not None:
                        boundary_summary["authoritative_max_gate_index"] = max(
                            int(prior_max_gate)
                            if type(prior_max_gate) is int
                            else cleanup_entry_gate_index,
                            cleanup_entry_gate_index,
                        )
                    boundary_summary["outcome"] = "abort"
                    boundary_summary["reason"] = boundary_reason
                    boundary_summary["contact_safety_outcome"] = (
                        "cleanup_boundary_authority_violation"
                    )
                    self._gate1_recenter_summary = boundary_summary
                self.recorder.emit(
                    "stage_abort",
                    stage=stage,
                    reason=boundary_reason,
                )
            recenter_summary_before_cleanup = (
                dict(self._gate1_recenter_summary)
                if (
                    stage == GATE1_RECENTER_STAGE
                    and self._gate1_recenter_summary is not None
                )
                else None
            )
            alignment_summary_before_cleanup = (
                dict(self._visual_alignment_summary)
                if (
                    stage == VISUAL_ALIGN_STAGE
                    and self._visual_alignment_summary is not None
                )
                else None
            )
            cleanup_confirmed = await self.safe_cleanup()
            if (
                stage == GATE1_RECENTER_STAGE
                and recenter_summary_before_cleanup is not None
            ):
                recenter_summary = recenter_summary_before_cleanup
                recenter_summary["cleanup_entry_gate_index"] = (
                    cleanup_entry_gate_index
                )
                recenter_summary["cleanup_entry_race_finished"] = (
                    cleanup_entry_race_finished
                )
                recenter_summary["cleanup_confirmed"] = bool(
                    cleanup_confirmed
                )
                recenter_summary["success"] = bool(
                    success
                    and cleanup_confirmed
                    and recenter_summary.get("recenter_criteria_met")
                )
                self._gate1_recenter_summary = recenter_summary
                details["gate1_recenter"] = recenter_summary
                self.recorder.emit(
                    "gate1_recenter_post_cleanup",
                    **recenter_summary,
                )
            if (
                stage == VISUAL_ALIGN_STAGE
                and alignment_summary_before_cleanup is not None
            ):
                alignment_summary = alignment_summary_before_cleanup
                alignment_summary["cleanup_entry_gate_index"] = (
                    cleanup_entry_gate_index
                )
                alignment_summary["cleanup_entry_race_finished"] = (
                    cleanup_entry_race_finished
                )
                alignment_summary["cleanup_confirmed"] = bool(
                    cleanup_confirmed
                )
                alignment_summary["success"] = bool(
                    success
                    and cleanup_confirmed
                    and alignment_summary.get(
                        "alignment_criteria_met"
                    )
                )
                if not cleanup_confirmed:
                    alignment_summary["outcome"] = "abort"
                    alignment_summary["reason"] = (
                        "visual alignment cleanup was unconfirmed"
                    )
                    alignment_summary["abort_outcome"] = (
                        "cleanup_unconfirmed"
                    )
                self._visual_alignment_summary = alignment_summary
                details["visual_alignment"] = alignment_summary
                self.recorder.emit(
                    "visual_alignment_post_cleanup",
                    **alignment_summary,
                )
            race = self.adapter.race_status
            gate_after = race.active_gate_index if race else None
            post_cleanup_diagnostic_errors: List[str] = []
            if (
                write_diagnostic_pngs
                and cleanup_confirmed
                and self._post_gate_last_frame is not None
            ):
                token, image = self._post_gate_last_frame
                observation = details.get("gate1_observation", {})
                if observation.get("gate1_observed"):
                    final_frame = observation.get("frames", [{}])[-1]
                    if (
                        token[1] == final_frame.get("frame_id")
                        and token[2] == final_frame.get("sim_time_ns")
                    ):
                        self._deferred_pngs.append(("gate1_acquired", image))
                        self.recorder.emit("next_gate_reacquired", **observation)
                    else:
                        post_cleanup_diagnostic_errors.append(
                            "acquired-frame PNG token did not match the accepted target"
                        )
                else:
                    self._deferred_pngs.append(
                        ("gate1_observation_terminal", image)
                    )
            if cleanup_confirmed and write_diagnostic_pngs:
                diagnostic_paths, diagnostic_errors = self._flush_deferred_snapshots()
                diagnostic_errors = (
                    post_cleanup_diagnostic_errors + diagnostic_errors
                )
            elif not cleanup_confirmed and write_diagnostic_pngs:
                self._deferred_pngs = []
                diagnostic_paths = []
                diagnostic_errors = [
                    "diagnostic images not encoded because cleanup was unconfirmed"
                ]
            else:
                self._deferred_pngs = []
                diagnostic_paths = []
                diagnostic_errors = []
            if diagnostic_paths:
                details["diagnostic_pngs"] = diagnostic_paths
            if diagnostic_errors:
                details["diagnostic_errors"] = diagnostic_errors
        return StageResult(
            stage=stage,
            success=success and cleanup_confirmed,
            reason=(reason if cleanup_confirmed else f"{reason}; cleanup unconfirmed"),
            duration_s=time.monotonic() - started,
            gate_index_before=gate_before,
            gate_index_after=gate_after,
            cleanup_confirmed=cleanup_confirmed,
            details=details,
            controller=dict(
                self.visual_controller_evidence
                if stage in VISUAL_POWERED_STAGES
                else self.controller_evidence
            ),
        )


def _consume_stopped_capture_tail(runner: VQ2Runner, vision: VQ2VisionThread) -> None:
    """Consume every pending publication after vision termination is proved."""

    if vision.is_running:
        raise RuntimeError("cannot consume capture tail while vision is running")
    queue_depth = getattr(vision, "capture_snapshot_queue_depth", None)
    if not callable(queue_depth):
        runner._sample()
        return
    while queue_depth() > 0:
        before = queue_depth()
        runner._sample()
        after = queue_depth()
        if after >= before:
            raise RuntimeError("stopped capture queue did not advance")


async def run_live(
    stage: str,
    address: str,
    record: Optional[str],
    *,
    replay_bundle: Optional[str] = None,
    recording_approved: bool = False,
    preflight_healthy_dwell_s: float = 0.0,
    preflight_timeout_s: float = 10.0,
    preflight_before_powered_stage: bool = True,
    write_diagnostic_pngs: bool = True,
    run_manifest_sha256: Optional[str] = None,
    controller_config: Optional[
        Mapping[str, Any] | VQ2ControllerConfig | VisualNavigationConfig
    ] = None,
    candidate_commit: Optional[str] = None,
    expected_controller_config_sha256: Optional[str] = None,
) -> StageResult:
    if type(stage) is not str or stage not in LIVE_RUN_STAGES:
        raise ValueError(f"unsupported live stage: {stage}")
    try:
        if stage in VISUAL_POWERED_STAGES:
            effective_visual_controller = (
                default_visual_config()
                if controller_config is None
                else (
                    validate_visual_config(
                        controller_config.to_effective_mapping()
                    )
                    if isinstance(
                        controller_config,
                        VisualNavigationConfig,
                    )
                    else validate_visual_config(controller_config)
                )
            )
            effective_controller = default_controller_config()
        else:
            effective_controller = (
                default_controller_config()
                if controller_config is None
                else (
                    validate_controller_config(
                        controller_config.to_effective_mapping()
                    )
                    if isinstance(controller_config, VQ2ControllerConfig)
                    else validate_controller_config(controller_config)
                )
            )
            effective_visual_controller = default_visual_config()
    except (ControllerConfigError, VisualConfigError) as exc:
        raise ValueError(f"controller configuration refused: {exc}") from exc
    legacy_controller = controller_config_evidence(
        effective_controller,
        candidate_commit=candidate_commit,
    )
    visual_controller = controller_config_evidence(
        effective_visual_controller,
        candidate_commit=candidate_commit,
    )
    controller = (
        visual_controller
        if stage in VISUAL_POWERED_STAGES
        else legacy_controller
    )
    if (
        stage not in {GATE1_RECENTER_STAGE, *VISUAL_POWERED_STAGES}
        and effective_controller.effective_config_sha256
        != default_controller_config().effective_config_sha256
    ):
        raise ValueError(
            "custom controller configurations are admitted only for "
            "gate1-recenter"
        )
    if expected_controller_config_sha256 is not None and (
        type(expected_controller_config_sha256) is not str
        or expected_controller_config_sha256
        != (
            effective_visual_controller.effective_config_sha256
            if stage in VISUAL_POWERED_STAGES
            else effective_controller.effective_config_sha256
        )
    ):
        raise ValueError(
            "expected controller config hash does not match effective config"
        )
    if type(recording_approved) is not bool:
        raise TypeError("recording_approved must be an exact bool")
    if type(preflight_before_powered_stage) is not bool:
        raise TypeError("preflight_before_powered_stage must be an exact bool")
    if type(write_diagnostic_pngs) is not bool:
        raise TypeError("write_diagnostic_pngs must be an exact bool")
    if run_manifest_sha256 is not None and (
        type(run_manifest_sha256) is not str
        or len(run_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in run_manifest_sha256)
    ):
        raise ValueError("run_manifest_sha256 must be 64 lowercase hexadecimal characters")
    if replay_bundle is not None and recording_approved is not True:
        raise PermissionError(
            "programmatic replay capture requires explicit recording_approved=True"
        )
    if (
        type(preflight_healthy_dwell_s) not in {int, float}
        or not math.isfinite(preflight_healthy_dwell_s)
        or not 0.0 <= float(preflight_healthy_dwell_s) <= 8.0
    ):
        raise ValueError(
            "preflight_healthy_dwell_s must be finite and in [0, 8]"
        )
    if stage != "preflight" and float(preflight_healthy_dwell_s) != 0.0:
        raise ValueError("preflight dwell is valid only for the preflight stage")
    if (
        type(preflight_timeout_s) not in {int, float}
        or not math.isfinite(preflight_timeout_s)
        or not 1.0 <= float(preflight_timeout_s) <= 10.0
    ):
        raise ValueError("preflight_timeout_s must be finite and in [1, 10]")
    if stage in VISUAL_POWERED_STAGES and (
        run_manifest_sha256 is None
        or candidate_commit is None
        or expected_controller_config_sha256 is None
        or replay_bundle is None
        or recording_approved is not True
        or preflight_before_powered_stage is not False
        or write_diagnostic_pngs is not False
    ):
        raise PermissionError(
            "visual-navigation powered stages require the clean, "
            "manifest-bound fast-cycle "
            "wrapper with private replay capture"
        )
    _load_live_transport_dependencies()
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
    )
    replay = None
    if replay_bundle is not None:
        (
            async_replay_recorder,
            replay_bundle_writer,
            capture_environment_fingerprint,
            capture_git_provenance,
        ) = _replay_capture_dependencies()
        repo_root = Path(__file__).resolve().parents[1]
        commit_hash, dirty_diff_hash, code_hash = capture_git_provenance(repo_root)
        replay_writer = replay_bundle_writer(
            replay_bundle,
            metadata={
                    "simulator_build": "3385",
                    "simulator_mode": "Training",
                    "simulator_mode_basis": "operator-attested-2026-07-20",
                    "stage": stage,
                    "preflight_healthy_dwell_s": float(
                        preflight_healthy_dwell_s
                    ),
                    "preflight_timeout_s": float(preflight_timeout_s),
                    "preflight_before_powered_stage": (
                        preflight_before_powered_stage
                    ),
                    "write_diagnostic_pngs": write_diagnostic_pngs,
                    "run_manifest_sha256": run_manifest_sha256,
                    "mavlink_address": address,
                    "capture_kind": "private-development-session",
                    "commit_hash": commit_hash,
                    "dirty_diff_hash": dirty_diff_hash,
                    "code_hash": code_hash,
                    "environment_fingerprint": capture_environment_fingerprint(),
                    "runner_evaluator_version": "vq2-runner-capture/1",
                    "timing_evidence_schemas": [
                        "aigp-vq2-mavlink-ingress/1",
                        "aigp-vq2-received-imu/1",
                        "aigp-vq2-camera-frame-timing-observation/1",
                    ],
                    # Frozen replay-evaluator RNG seed.  This is independent
                    # of simulator randomness and is bound into T1 identity.
                    "seed": 42,
                    "detector": {
                        "class": "VQ2GateDetector",
                        "image_size_px": [640, 360],
                        "min_area": 500,
                        "max_area": 500000,
                        "max_aspect_ratio": 3.0,
                        "min_confidence": 0.10,
                        "hsv_ranges": [
                            [[0, 50, 100], [12, 255, 255]],
                            [[150, 50, 100], [180, 255, 255]],
                        ],
                    },
                    "controller_envelope": replay_controller_envelope(
                        stage
                    ),
            },
            repo_root=repo_root,
        )
        try:
            replay = async_replay_recorder(replay_writer)
        except BaseException as exc:
            try:
                replay_writer.abort(
                    "async replay recorder construction failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            except BaseException as cleanup_exc:
                exc.add_note(
                    "Replay writer abort also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
    vision: Optional[VQ2VisionThread] = None
    recorder: Optional[JsonlRecorder] = None
    runner: Optional[VQ2Runner] = None
    result: Optional[StageResult] = None
    failure: Optional[str] = None
    capture_stats = None
    primary_exception: Optional[BaseException] = None
    primary_traceback = None
    cleanup_exceptions: List[BaseException] = []
    try:
        # The replay writer thread already exists at this point.  Every later
        # constructor is therefore inside the same cleanup ownership region.
        vision = VQ2VisionThread(
            on_snapshot=(
                replay.capture_decoded_snapshot if replay is not None else None
            ),
            capture_snapshot_queue_enabled=(
                replay is not None and stage == "preflight"
            ),
        )
        recorder = JsonlRecorder(
            record,
            replay=replay,
            capture_fifo_enabled=(replay is not None and stage == "preflight"),
        )
        if run_manifest_sha256 is not None:
            recorder.emit(
                "fast_cycle_binding",
                run_manifest_sha256=run_manifest_sha256,
                controller=controller,
            )
        else:
            recorder.emit(
                "controller_configuration",
                controller=controller,
            )
        runner = VQ2Runner(
            adapter,
            vision,
            recorder=recorder,
            controller_config=effective_controller,
            controller_evidence=legacy_controller,
            visual_config=effective_visual_controller,
            visual_controller_evidence=visual_controller,
            visual_session_id=(
                run_manifest_sha256
                or candidate_commit
                or "direct-live-session"
            ),
        )
        await adapter.connect(address)
        preflight = None
        if stage == "preflight" or preflight_before_powered_stage:
            preflight = await runner.preflight(
                timeout_s=float(preflight_timeout_s),
                healthy_dwell_s=float(preflight_healthy_dwell_s),
            )
        if stage == "preflight":
            assert preflight is not None
            result = StageResult(
                stage=stage,
                success=True,
                reason="passive preflight completed; no flight command sent",
                duration_s=0.0,
                gate_index_before=preflight.get("race_gate_index"),
                gate_index_after=preflight.get("race_gate_index"),
                cleanup_confirmed=True,
                details=preflight,
                controller=dict(controller),
            )
        else:
            result = await runner.run_powered_stage(
                stage,
                write_diagnostic_pngs=write_diagnostic_pngs,
            )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        primary_exception = exc
        primary_traceback = exc.__traceback__
    finally:
        if vision is not None:
            vision_stopped = False
            try:
                vision.stop()
                vision_stopped = True
            except BaseException as exc:
                cleanup_exceptions.append(exc)
                if replay is not None:
                    replay.fail(
                        f"vision termination not proved before replay seal: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if (
                vision_stopped
                and replay is not None
                and runner is not None
                and result is not None
                and stage == "preflight"
            ):
                try:
                    _consume_stopped_capture_tail(runner, vision)
                except BaseException as exc:
                    cleanup_exceptions.append(exc)
                    replay.fail(
                        "stopped-vision capture tail was not consumed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            try:
                vision_capture_stats = asdict(vision.stats())
            except BaseException as exc:
                cleanup_exceptions.append(exc)
                vision_capture_stats = {
                    "stats_error": f"{type(exc).__name__}: {exc}"
                }
                if replay is not None:
                    replay.fail(
                        f"vision stats unavailable: {type(exc).__name__}: {exc}"
                    )
        else:
            vision_capture_stats = {"unavailable": True}
            if replay is not None:
                replay.fail("vision construction failed before capture ownership")
        try:
            await adapter.disconnect()
        except BaseException as exc:
            cleanup_exceptions.append(exc)
        if recorder is not None:
            try:
                final_estimator = (
                    runner._replay_estimator_fields()
                    if runner is not None
                    else None
                )
                final_receive_s = time.monotonic()
                for value in adapter.drain_received_ingress():
                    if type(value) is ReceivedIMUSampleV1:
                        recorder.record_imu(
                            value.imu,
                            final_estimator,
                            final_receive_s,
                            received_sample=value,
                        )
                    elif type(value) is MavlinkIngressV1:
                        recorder.record_mavlink_ingress(value)
                    else:
                        raise TypeError(
                            "exact receiver ingress item has invalid type"
                        )
            except BaseException as exc:
                cleanup_exceptions.append(exc)
                if replay is not None:
                    replay.fail(
                        "final receiver ingress drain failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        try:
            ingress_stats = asdict(adapter.ingress_stats())
        except BaseException as exc:
            cleanup_exceptions.append(exc)
            ingress_stats = {"stats_error": f"{type(exc).__name__}: {exc}"}
        try:
            audit_value = adapter.outbound_audit()
            outbound_audit = asdict(audit_value)
            outbound_audit["disallowed_count"] = audit_value.disallowed_count
        except BaseException as exc:
            cleanup_exceptions.append(exc)
            outbound_audit = {"audit_error": f"{type(exc).__name__}: {exc}"}
        if result is not None:
            details = dict(result.details or {})
            details["mavlink_ingress_stats"] = ingress_stats
            details["mavlink_outbound_audit"] = outbound_audit
            result = replace(result, details=details)
            if (
                stage == "preflight"
                and outbound_audit.get("disallowed_count") != 0
            ):
                result = replace(
                    result,
                    success=False,
                    reason=(
                        f"{result.reason}; passive outbound audit was not zero"
                    ),
                )
        base_outcome = (
            asdict(result)
            if result is not None
            else {"success": False, "failure": failure or "runner did not return"}
        )
        base_outcome["vision_capture_stats"] = vision_capture_stats
        if cleanup_exceptions:
            base_outcome["transport_cleanup_errors"] = [
                f"{type(exc).__name__}: {exc}" for exc in cleanup_exceptions
            ]
        try:
            if recorder is not None:
                capture_stats = recorder.close(outcome=base_outcome)
            elif replay is not None:
                replay.fail("recorder construction failed before capture ownership")
                capture_stats = replay.close(outcome=base_outcome)
        except BaseException as exc:
            cleanup_exceptions.append(exc)
    if primary_exception is not None:
        raise primary_exception.with_traceback(primary_traceback)
    if cleanup_exceptions:
        raise cleanup_exceptions[0]
    assert result is not None
    return replay_capture_result(
        result,
        capture_requested=replay is not None,
        capture_stats=capture_stats,
    )


def replay_capture_result(
    result: StageResult,
    *,
    capture_requested: bool,
    capture_stats: Any,
) -> StageResult:
    """Fail closed when an explicitly requested replay is incomplete."""

    if capture_requested and (capture_stats is None or not capture_stats.complete):
        replay_details = (
            asdict(capture_stats)
            if capture_stats is not None
            else {"complete": False, "reason": "capture stats unavailable"}
        )
        details = dict(result.details or {})
        details["replay_capture"] = replay_details
        result = replace(
            result,
            success=False,
            reason=f"{result.reason}; replay capture incomplete",
            details=details,
        )
    elif capture_requested and capture_stats is not None:
        details = dict(result.details or {})
        details["replay_capture"] = asdict(capture_stats)
        result = replace(result, details=details)
    return result


def _default_record_path(stage: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return str(Path("captures") / f"vq2_{stage}_{stamp}.jsonl.gz")


def _default_replay_path(stage: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return str(Path("captures") / "replays" / f"vq2_{stage}_{stamp}.vq2replay")


def _calibration_cli_requested(argv: Sequence[str]) -> bool:
    return CALIBRATION_STAGE in argv or any(
        item
        in {
            "--powered-attempt-envelope",
            "--powered-process-authority",
            "--attempt-capability-handle",
            "--parent-liveness-handle",
            "--cleanup-certificate",
        }
        for item in argv
    )


def _write_calibration_stderr(stream: Any, message: bytes) -> None:
    try:
        stream.write(message)
    except TypeError:
        stream.write(message.decode("utf-8"))
    if hasattr(stream, "flush"):
        stream.flush()


class _OwnedCalibrationCapabilityOperations:
    """Track the inherited one-shot reader without ever closing it twice."""

    def __init__(self, operations: Any, capability_handle: int) -> None:
        if operations is None:
            raise TypeError("capability operations are required")
        if type(capability_handle) is not int or capability_handle < 1:
            raise ValueError("capability handle must be a positive exact integer")
        self._operations = operations
        self._capability_handle = capability_handle
        self._lock = threading.Lock()
        self._close_attempted = False
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._operations, name)

    def close_handle(self, handle: int) -> None:
        if type(handle) is not int or handle != self._capability_handle:
            raise CalibrationBootstrapError(
                "capability operations attempted to close an unexpected handle"
            )
        with self._lock:
            if self._close_attempted:
                raise CalibrationBootstrapError(
                    "capability reader close was attempted more than once"
                )
            self._close_attempted = True
            self._operations.close_handle(handle)
            self._closed = True

    def close_unconsumed(self) -> bool:
        with self._lock:
            if self._close_attempted:
                return self._closed
            self._close_attempted = True
            try:
                self._operations.close_handle(self._capability_handle)
            except BaseException:
                return False
            self._closed = True
            return True


def _close_default_calibration_process_boundary(
    process_boundary: Any,
    monotonic_ns: Callable[[], int],
) -> bool:
    """Close one default bootstrap boundary under one immutable local deadline."""

    try:
        started = monotonic_ns()
        if type(started) is not int or started < 0:
            return False
        deadline = started + CALIBRATION_OWNED_HANDLE_CLOSE_NS
        proof = process_boundary.close_owned_handles(
            deadline_monotonic_ns=deadline,
            monotonic_ns=monotonic_ns,
        )
    except BaseException:
        return False
    return getattr(proof, "proved", None) is True


def _create_default_calibration_delegated_lease(
    admission: CalibrationAdmission,
    process_boundary: CalibrationProcessBoundary,
    qpc_provider: Any,
) -> CalibrationLeaseBoundary:
    """Construct the production delegated lease only after capability admission."""

    from scripts import aigp_live_lease

    context = admission.attempt["context"]
    frequency = qpc_provider.query_performance_frequency_hz()
    if frequency != context["host"]["qpc_frequency_hz"]:
        raise CalibrationBootstrapError(
            "runtime QPC frequency changed from powered-child admission"
        )
    paths = context["paths"]
    store = aigp_live_lease.PoweredLeaseLedgerStore(
        paths["lease_directory"],
        paths["lease_final"],
        task_id=context["task_id"],
        session_id=context["session_id"],
        attempt_id=context["attempt_id"],
        attempt_envelope_sha256=admission.attempt_envelope_sha256,
        attempt_context_sha256=admission.attempt["context_sha256"],
        wrapper_process=admission.wrapper_process,
        qpc_frequency_hz=frequency,
        _clock_ns=qpc_provider.now_ns,
    )
    return aigp_live_lease.DelegatedPoweredLeaseBoundary(
        store,
        admission.arguments.powered_attempt_envelope,
        parent_signaled=process_boundary.parent_signaled,
        _clock_ns=qpc_provider.now_ns,
    )


def _create_default_calibration_recorder(
    admission: CalibrationAdmission,
    *,
    monotonic_ns: Callable[[], int],
) -> JsonlRecorder:
    """Create both create-new capture artifacts after child admission."""

    AsyncRecorder, ReplayWriter, _environment, _git = _replay_capture_dependencies()
    context = admission.attempt["context"]
    metadata = {
        "capture_kind": "powered_calibration",
        "producer_role": CALIBRATION_CHILD_ROLE,
        "task_id": context["task_id"],
        "session_id": context["session_id"],
        "attempt_id": context["attempt_id"],
        "candidate_commit": context["candidate_commit"],
        "attempt_context_sha256": admission.attempt["context_sha256"],
        "attempt_envelope_sha256": admission.attempt_envelope_sha256,
        "process_authority_sha256": admission.process_authority_sha256,
    }
    writer: Any = None
    replay: Any = None
    try:
        writer = ReplayWriter(
            admission.arguments.replay_bundle,
            session_id=context["session_id"],
            metadata=metadata,
            require_private=True,
        )
        replay = AsyncRecorder(writer)
        return JsonlRecorder(
            admission.arguments.record,
            replay=replay,
            capture_fifo_enabled=True,
            create_new=True,
        )
    except BaseException as exc:
        if replay is not None:
            try:
                replay.fail(
                    "powered calibration recorder construction failed: "
                    f"{type(exc).__name__}"
                )
                now = monotonic_ns()
                remaining = max(
                    1,
                    admission.replay_close_deadline_monotonic_ns - now,
                )
                replay.close(
                    outcome={"powered_calibration_recorder_constructed": False},
                    timeout_s=remaining / 1_000_000_000.0,
                )
            except BaseException as cleanup_exc:
                exc.add_note(
                    "Replay construction cleanup also failed: "
                    f"{type(cleanup_exc).__name__}"
                )
        elif writer is not None:
            try:
                writer.abort(
                    "powered calibration async recorder construction failed: "
                    f"{type(exc).__name__}"
                )
            except BaseException as cleanup_exc:
                exc.add_note(
                    "Replay writer abort also failed: "
                    f"{type(cleanup_exc).__name__}"
                )
        raise


def _create_default_calibration_adapter(
    *,
    admission: CalibrationAdmission,
    bind: Mapping[str, Any],
    outbound_guards: Any,
    role_valid: Callable[[], bool],
    parent_alive: Callable[[], bool],
    lease_valid: Callable[[], bool],
    runtime: Any,
    monotonic_ns: Callable[[], int],
) -> Any:
    """Bind and transfer the exclusive MAVLink endpoint after lease proof."""

    del admission
    from competition.aigp_mavlink import (
        AIGPMavlinkAdapter as PoweredAdapter,
        POWERED_RECEIVE_MODE_WORKER,
        PoweredMavlinkTransport,
    )

    endpoint = runtime.create_exclusive_udp_endpoint(bind["host"], bind["port"])
    transport: Any = None
    try:
        transport = PoweredMavlinkTransport.from_pymavlink(
            endpoint,
            outbound_guards=outbound_guards,
            role_valid=role_valid,
            parent_alive=parent_alive,
            lease_valid=lease_valid,
        )
        return PoweredAdapter(
            enable_vision=False,
            require_track=False,
            telemetry_mode="imu",
            fetch_track_on_connect=False,
            monotonic_ns=monotonic_ns,
            powered_transport=transport,
            powered_receive_mode=POWERED_RECEIVE_MODE_WORKER,
        )
    except BaseException:
        owner = transport if transport is not None else endpoint
        try:
            owner.close()
        except BaseException as close_exc:
            raise CalibrationLifecycleError(
                "partial powered MAVLink construction could not close its endpoint"
            ) from close_exc
        raise


def _create_default_calibration_vision(
    *,
    admission: CalibrationAdmission,
    bind: Mapping[str, Any],
    monotonic_ns: Callable[[], int],
    **options: Any,
) -> Any:
    """Construct the production receiver; its exclusive bind occurs at start()."""

    del admission
    from competition.vq2_vision import VQ2VisionThread as PoweredVision

    return PoweredVision(
        bind_host=bind["host"],
        port=bind["port"],
        monotonic_ns=monotonic_ns,
        **options,
    )


def _create_default_calibration_camera_socket(
    host: str,
    port: int,
    *,
    runtime: Any,
) -> Any:
    """Transfer one exclusively bound raw socket to VQ2VisionThread."""

    endpoint = runtime.create_exclusive_udp_endpoint(host, port)
    if type(endpoint) is not runtime.ExclusiveUdpEndpoint:
        try:
            endpoint.close()
        except BaseException as close_exc:
            raise CalibrationLifecycleError(
                "invalid camera endpoint could not close its socket"
            ) from close_exc
        raise CalibrationLifecycleError(
            "camera endpoint factory did not return exact ExclusiveUdpEndpoint"
        )
    try:
        return endpoint.transfer_socket()
    except BaseException:
        try:
            endpoint.close()
        except BaseException as close_exc:
            raise CalibrationLifecycleError(
                "partial powered camera construction could not close its endpoint"
            ) from close_exc
        raise


def build_default_calibration_services(
    arguments: CalibrationArguments,
    *,
    qpc_provider_factory: Optional[Callable[[], Any]] = None,
    process_boundary_factory: Optional[Callable[..., CalibrationProcessBoundary]] = None,
    capability_operations_factory: Optional[Callable[[], Any]] = None,
    delegated_lease_factory: Optional[
        Callable[[CalibrationAdmission, CalibrationProcessBoundary, Any], CalibrationLeaseBoundary]
    ] = None,
    recorder_builder: Optional[Callable[[CalibrationAdmission], JsonlRecorder]] = None,
    adapter_builder: Optional[Callable[..., Any]] = None,
    vision_builder: Optional[Callable[..., Any]] = None,
    camera_socket_builder: Optional[Callable[[str, int], Any]] = None,
    publisher_factory: Optional[Callable[..., CalibrationCertificatePublisher]] = None,
    child_runner: Optional[
        Callable[[CalibrationAdmission, CalibrationChildServices], Any]
    ] = None,
) -> CalibrationAdmissionServices:
    """Build inert admission services and defer every output/live effect."""

    if not isinstance(arguments, CalibrationArguments):
        raise TypeError("arguments must be CalibrationArguments")
    from scripts import aigp_vq2_powered_attempt as attempt_contract
    from scripts import aigp_vq2_powered_runtime as powered_runtime

    capability_handle = powered_runtime.parse_decimal_handle(
        arguments.attempt_capability_handle
    )
    parent_handle = powered_runtime.parse_decimal_handle(
        arguments.parent_liveness_handle
    )
    if capability_handle == parent_handle:
        raise CalibrationBootstrapError(
            "capability and parent handles must be distinct"
        )
    make_capability = (
        capability_operations_factory
        or powered_runtime.Win32CapabilityPipeOperations
    )
    make_qpc = qpc_provider_factory or powered_runtime.WindowsQpcProvider
    make_process = (
        process_boundary_factory
        or powered_runtime.RetainedChildBootstrapProcessBoundary
    )
    make_lease = (
        delegated_lease_factory
        or _create_default_calibration_delegated_lease
    )
    run_child = child_runner or run_powered_calibration_child
    capability_owner: Optional[_OwnedCalibrationCapabilityOperations] = None
    qpc: Any = None
    process_boundary: Optional[CalibrationProcessBoundary] = None
    try:
        capability_owner = _OwnedCalibrationCapabilityOperations(
            make_capability(), capability_handle
        )
        qpc = make_qpc()
        process_boundary = make_process(capability_handle, parent_handle)
        post_admission_lock = threading.Lock()
        post_admission_started = False

        def run_admitted(admission: CalibrationAdmission) -> Any:
            nonlocal post_admission_started
            with post_admission_lock:
                if post_admission_started:
                    raise CalibrationLifecycleError(
                        "post-admission calibration services are one-shot"
                    )
                post_admission_started = True
            lease = make_lease(admission, process_boundary, qpc)
            if publisher_factory is None:
                from scripts.aigp_vq2_powered_cleanup import (
                    CanonicalCreateNewPublisher,
                )

                publisher = CanonicalCreateNewPublisher(
                    contract=attempt_contract,
                    monotonic_ns=qpc.now_ns,
                )
            else:
                publisher = publisher_factory(
                    contract=attempt_contract,
                    monotonic_ns=qpc.now_ns,
                )
            child_services = CalibrationChildServices(
                process_boundary=process_boundary,
                lease_boundary=lease,
                recorder_factory=(
                    recorder_builder
                    if recorder_builder is not None
                    else partial(
                        _create_default_calibration_recorder,
                        monotonic_ns=qpc.now_ns,
                    )
                ),
                adapter_factory=(
                    adapter_builder
                    if adapter_builder is not None
                    else partial(
                        _create_default_calibration_adapter,
                        runtime=powered_runtime,
                        monotonic_ns=qpc.now_ns,
                    )
                ),
                vision_factory=(
                    vision_builder
                    if vision_builder is not None
                    else partial(
                        _create_default_calibration_vision,
                        monotonic_ns=qpc.now_ns,
                    )
                ),
                camera_socket_factory=(
                    camera_socket_builder
                    if camera_socket_builder is not None
                    else partial(
                        _create_default_calibration_camera_socket,
                        runtime=powered_runtime,
                    )
                ),
                publisher=publisher,
                monotonic_ns=qpc.now_ns,
                contract=attempt_contract,
                runtime=powered_runtime,
            )
            return run_child(admission, child_services)

        return CalibrationAdmissionServices(
            process_boundary=process_boundary,
            capability_operations=capability_owner,
            monotonic_ns=qpc.now_ns,
            contract=attempt_contract,
            runtime=powered_runtime,
            run_admitted=run_admitted,
            owned_process_boundary=process_boundary,
            close_unconsumed_capability=capability_owner.close_unconsumed,
        )
    except BaseException:
        if capability_owner is not None:
            capability_owner.close_unconsumed()
        if process_boundary is not None and qpc is not None:
            _close_default_calibration_process_boundary(
                process_boundary,
                qpc.now_ns,
            )
        raise


def _run_calibration_cli(
    argv: Sequence[str],
    *,
    services: Optional[CalibrationAdmissionServices],
    stdout: Any,
    stderr: Any,
) -> int:
    arguments = parse_calibration_arguments(argv)
    active_services = services
    if active_services is None:
        try:
            active_services = build_default_calibration_services(arguments)
        except BaseException:
            _write_calibration_stderr(
                stderr, b"powered calibration failed before admission\n"
            )
            return 2
    admission: Optional[CalibrationAdmission] = None
    try:
        admission = admit_calibration_child(arguments, active_services)
    except BaseException:
        close_failed = False
        if active_services.close_unconsumed_capability is not None:
            try:
                if active_services.close_unconsumed_capability() is not True:
                    close_failed = True
            except BaseException:
                close_failed = True
        if active_services.owned_process_boundary is not None:
            if (
                active_services.owned_process_boundary
                is not active_services.process_boundary
                or not _close_default_calibration_process_boundary(
                    active_services.owned_process_boundary,
                    active_services.monotonic_ns,
                )
            ):
                close_failed = True
        _write_calibration_stderr(
            stderr, b"powered calibration failed before admission\n"
        )
        if close_failed:
            _write_calibration_stderr(
                stderr,
                b"powered calibration bootstrap handle closure failed\n",
            )
            return 1
        return 2
    try:
        # This is the first point at which the child may import a module able
        # to construct the simulator transports.
        _load_live_transport_dependencies()
        if (
            active_services.run_admitted is None
            and active_services.child_services is None
        ):
            _write_calibration_stderr(
                stderr, b"powered calibration execution integration is unavailable\n"
            )
            return 2
        if (
            active_services.child_services is not None
            and active_services.child_services.process_boundary
            is not active_services.process_boundary
        ):
            raise CalibrationEvidenceError(
                "admission and child lifecycle must share one process boundary"
            )
        result = (
            active_services.run_admitted(admission)
            if active_services.run_admitted is not None
            else run_powered_calibration_child(
                admission, active_services.child_services
            )
        )
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        if isinstance(result, CalibrationChildRunOutput):
            if active_services.process_boundary.parent_signaled(
                admission.parent_handle
            ):
                raise CalibrationLifecycleError(
                    "wrapper parent died before process-result serialization"
                )
            contract, _runtime = _powered_contract_modules(active_services)
            payload = contract.canonical_json_file_bytes(result.process_result)
            if active_services.process_boundary.parent_signaled(
                admission.parent_handle
            ):
                raise CalibrationLifecycleError(
                    "wrapper parent died before process-result publication"
                )
            try:
                stdout.write(payload)
            except TypeError:
                stdout.write(payload.decode("utf-8"))
            if hasattr(stdout, "flush"):
                stdout.flush()
            if active_services.process_boundary.parent_signaled(
                admission.parent_handle
            ):
                raise CalibrationLifecycleError(
                    "wrapper parent died during process-result publication"
                )
            return result.exit_code
        if type(result) is not int or result not in {0, 1, 2}:
            raise CalibrationEvidenceError(
                "admitted runner returned an invalid exit code"
            )
        if active_services.process_boundary.parent_signaled(
            admission.parent_handle
        ):
            raise CalibrationLifecycleError(
                "wrapper parent died before powered-child exit"
            )
        return result
    except BaseException:
        _write_calibration_stderr(
            stderr, b"powered calibration failed after admission\n"
        )
        return 1
    finally:
        close_failed = False
        if active_services.close_unconsumed_capability is not None:
            try:
                if active_services.close_unconsumed_capability() is not True:
                    close_failed = True
            except BaseException:
                close_failed = True
        try:
            proof = active_services.process_boundary.close_owned_handles(
                deadline_monotonic_ns=admission.exit_deadline_monotonic_ns,
                monotonic_ns=active_services.monotonic_ns,
            )
            if getattr(proof, "proved", None) is not True:
                raise CalibrationLifecycleError(
                    "bootstrap owned-handle closure was not proved"
                )
        except BaseException:
            close_failed = True
        admission.erase_role_secret()
        if close_failed:
            _write_calibration_stderr(
                stderr,
                b"powered calibration bootstrap handle closure failed\n",
            )
            return 1


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    calibration_services: Optional[CalibrationAdmissionServices] = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    error_stream = sys.stderr.buffer if stderr is None else stderr
    output_stream = sys.stdout.buffer if stdout is None else stdout
    if _calibration_cli_requested(args):
        return _run_calibration_cli(
            args,
            services=calibration_services,
            stdout=output_stream,
            stderr=error_stream,
        )

    parser = argparse.ArgumentParser(
        description="Bounded AIGP VQ2 training runner",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "sign-id",
            "hover",
            "gate0",
            "gate0-observe",
            GATE1_RECENTER_STAGE,
        ),
        default="preflight",
    )
    parser.add_argument("--address", default=DEFAULT_MAVLINK_URL)
    parser.add_argument(
        "--record",
        nargs="?",
        const="auto",
        default=None,
        help="write JSONL capture; omit the value for an automatic gzip path",
    )
    parser.add_argument(
        "--replay-bundle",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "write a private decoded-frame replay bundle outside normal Git; "
            "requires --recording-approved"
        ),
    )
    parser.add_argument(
        "--recording-approved",
        action="store_true",
        help="attest that organizer approval/credentials permit this recording",
    )
    parser.add_argument(
        "--preflight-healthy-dwell-s",
        type=float,
        default=0.0,
        help="continue an already-healthy passive preflight for up to 8 seconds",
    )
    parser.add_argument(
        "--preflight-timeout-s",
        type=float,
        default=10.0,
        help="fail passive stream readiness after 1-10 seconds",
    )
    parser.add_argument("--verbose", action="store_true")
    parsed = parser.parse_args(args)
    if parsed.replay_bundle is not None and not parsed.recording_approved:
        parser.error("--replay-bundle requires explicit --recording-approved")
    if parsed.stage != "preflight" and parsed.preflight_healthy_dwell_s != 0.0:
        parser.error("--preflight-healthy-dwell-s requires --stage preflight")
    record = (
        _default_record_path(parsed.stage)
        if parsed.record == "auto"
        else parsed.record
    )
    replay_bundle = (
        _default_replay_path(parsed.stage)
        if parsed.replay_bundle == "auto"
        else parsed.replay_bundle
    )
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = asyncio.run(
        run_live(
            parsed.stage,
            parsed.address,
            record,
            replay_bundle=replay_bundle,
            recording_approved=parsed.recording_approved,
            preflight_healthy_dwell_s=parsed.preflight_healthy_dwell_s,
            preflight_timeout_s=parsed.preflight_timeout_s,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
