"""Import-inert foundation for the single VQ2 powered-calibration probe.

This module contains no concrete simulator, socket, process-spawn, mutex, or
live-port provider.  It admits the reviewed freeze through injected boundaries
and owns the fail-closed, single-attempt orchestration over those providers.
Importing this module has no external side effect.
"""

from __future__ import annotations

import argparse
import hashlib
import ntpath
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from scripts import aigp_vq2_powered_attempt as attempt_contract


PROBE_MODULE = "scripts.aigp_vq2_powered_calibration_probe"
IMPORT_AUDIT_MODULE = "scripts.aigp_vq2_powered_import_audit"
POWERED_IMPORT_SEED_MODULES = attempt_contract.IMPORT_INVENTORY_SEEDS

# Modules reached only through the child's deliberately lazy capture/transport
# loaders.  L0 imports these after the six frozen seed modules and inventories
# the resulting complete sys.modules graph.  Keep this literal ordinal-sorted.
POWERED_EAGER_IMPORT_MODULES = (
    "aigp_loop._util",
    "aigp_loop.replay",
    "competition.aigp_mavlink",
    "competition.vq2_vision",
)

if POWERED_EAGER_IMPORT_MODULES != tuple(
    sorted(set(POWERED_EAGER_IMPORT_MODULES), key=lambda item: item.encode("utf-8"))
):  # pragma: no cover - import-time code-owned invariant
    raise RuntimeError("POWERED_EAGER_IMPORT_MODULES must be unique and ordinal-sorted")

if POWERED_IMPORT_SEED_MODULES != tuple(
    sorted(set(POWERED_IMPORT_SEED_MODULES), key=lambda item: item.encode("utf-8"))
):  # pragma: no cover - import-time shared-contract invariant
    raise RuntimeError("POWERED_IMPORT_SEED_MODULES must be unique and ordinal-sorted")


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CAPABILITY_DOMAINS = MappingProxyType(
    {
        "lease_owner": "aigp-vq2-lease-owner/1",
        "child": "aigp-vq2-powered-child/1",
        "cleanup": "aigp-vq2-powered-cleanup/1",
    }
)
_ROLE_PHASE_DURATIONS = MappingProxyType(
    {
        "child_supervision": "child_total",
        "fallback_supervision": "fallback_total",
        "terminal_ready": "terminal_publish",
        "invalid_ready": "terminal_publish",
    }
)
_POSTRELEASE_PHASES = frozenset(
    {
        "bundle_verify",
        "capture_seal",
        "analysis",
        "split_publish",
        "terminal_ready",
        "poison_publish",
        "invalid_ready",
    }
)
_POWERED_RUNTIME_IMPORT_PROVIDERS = MappingProxyType(
    {
        "cv2.utils.fs": ("cv2", ("_native",), ("_native", "utils", "fs"), "venv"),
        "cv2.utils.logging": (
            "cv2",
            ("_native",),
            ("_native", "utils", "logging"),
            "venv",
        ),
        "cv2.utils.nested": (
            "cv2",
            ("_native",),
            ("_native", "utils", "nested"),
            "venv",
        ),
        "typing.io": ("typing", (), ("io",), "stdlib"),
        "typing.re": ("typing", (), ("re",), "stdlib"),
    }
)
if tuple(_POWERED_RUNTIME_IMPORT_PROVIDERS) != tuple(
    sorted(_POWERED_RUNTIME_IMPORT_PROVIDERS, key=lambda item: item.encode("utf-8"))
):  # pragma: no cover - import-time code-owned invariant
    raise RuntimeError("runtime import provider map must be ordinal-sorted")
if tuple(_POWERED_RUNTIME_IMPORT_PROVIDERS) != attempt_contract.RUNTIME_IMPORT_MODULES:
    raise RuntimeError("runtime import provider map drifted from the schema allowlist")


class PoweredCalibrationProbeError(RuntimeError):
    """Base class for fail-closed wrapper-foundation failures."""


class OfflineAdmissionError(PoweredCalibrationProbeError):
    """The reviewed freeze or current offline execution identity did not match."""


class AttemptGateError(PoweredCalibrationProbeError):
    """The single attempt is consumed, poisoned, ambiguous, or not create-new."""


class SecureBoundaryError(PoweredCalibrationProbeError):
    """An injected path/publication boundary failed a mandatory invariant."""


class PublicationError(PoweredCalibrationProbeError):
    """A create-new publication did not produce one proved complete file."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        state: str,
        observed_sha256: str | None = None,
    ) -> None:
        if observed_sha256 is not None and _SHA256_RE.fullmatch(observed_sha256) is None:
            raise ValueError("observed partial SHA-256 must be 64 lowercase hex")
        super().__init__(message)
        self.path = path
        self.state = state
        self.observed_sha256 = observed_sha256
        self.retry_allowed = False


class PartialPublicationError(PublicationError):
    """A target may contain forensic partial bytes and must never be retried."""


class BoundaryCreateNewError(RuntimeError):
    """Error raised by a secure boundary after an attempted create-new write."""

    def __init__(
        self,
        path: str,
        *,
        state: str,
        detail: str = "",
        observed_sha256: str | None = None,
    ) -> None:
        if state not in {"absent", "partial", "unknown"}:
            raise ValueError("boundary failure state must be absent, partial, or unknown")
        if observed_sha256 is not None and _SHA256_RE.fullmatch(observed_sha256) is None:
            raise ValueError("observed partial SHA-256 must be 64 lowercase hex")
        super().__init__(detail or f"create-new publication failed with state {state}")
        self.path = path
        self.state = state
        self.observed_sha256 = observed_sha256


class LiveIntegrationUnavailable(PoweredCalibrationProbeError):
    """The production runtime providers have not been wired into the wrapper."""


@dataclass(frozen=True)
class ProbeArguments:
    live_freeze: str
    live_freeze_sha256: str
    expected_commit: str


@dataclass(frozen=True)
class PathProof:
    """Handle-derived canonical path facts supplied by an injected boundary."""

    path: str
    final_path: str
    kind: str
    volume_id: str
    exists: bool = True
    non_reparse: bool = True
    ancestors_non_reparse: bool = True
    retained_handle: bool = True


@dataclass(frozen=True)
class FileIdentityProof:
    path: PathProof
    size_bytes: int
    sha256: str
    hash_kind: str = "file_bytes"
    stable_before_after: bool = True


@dataclass(frozen=True)
class StableJsonProof:
    identity: FileIdentityProof
    raw_bytes: bytes
    value: Any


@dataclass(frozen=True)
class GitWorktreeProof:
    worktree_path: str
    head_commit: str
    head_tree: str
    detached_head: bool
    tracked_clean: bool
    untracked_clean: bool
    ignored_clean: bool
    common_dir_outside_worktree: bool = True


@dataclass(frozen=True)
class ImportRevalidation:
    inventory: Mapping[str, Any]
    origins_reverified: bool
    user_site_on_sys_path: bool
    unexpected_candidate_or_venv_modules: tuple[str, ...] = ()
    unclassified_origins: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorAttemptObservation:
    attempt_id: str
    terminal_record_count: int
    valid_terminal_record_count: int


@dataclass(frozen=True)
class AttemptRootSnapshot:
    evidence_root: str
    live_poison_present: bool
    target_attempt_directory_present: bool
    target_attempt_envelope_present: bool
    prior_attempts: tuple[PriorAttemptObservation, ...] = ()
    unknown_attempt_entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class SecureDirectoryReceipt:
    path: str
    final_path: str
    parent_final_path: str
    volume_id: str
    parent_volume_id: str
    owner_id: str
    current_user_id: str
    created_new: bool
    owner_is_current_user: bool
    current_user_only_dacl: bool
    dacl_applied_at_create: bool
    non_reparse: bool
    ancestors_non_reparse: bool
    retained_handle: bool


@dataclass(frozen=True)
class CreateNewFileReceipt:
    path: str
    final_path: str
    parent_final_path: str
    volume_id: str
    parent_volume_id: str
    owner_id: str
    current_user_id: str
    size_bytes: int
    sha256: str
    completed_monotonic_ns: int
    created_new: bool
    regular_file: bool
    owner_is_current_user: bool
    current_user_only_dacl: bool
    dacl_applied_at_create: bool
    non_reparse: bool
    ancestors_non_reparse: bool
    flushed: bool
    readback_verified: bool


@dataclass(frozen=True)
class AttemptHandleSet:
    child_capability_read_handle: int
    child_parent_liveness_handle: int
    cleanup_capability_read_handle: int
    cleanup_parent_liveness_handle: int


@dataclass(repr=False)
class CapabilitySecrets:
    """Three mutable one-use values with deterministic in-place erasure."""

    _lease_owner: bytearray
    _child: bytearray
    _cleanup: bytearray
    _consumed: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        for name in ("_lease_owner", "_child", "_cleanup"):
            value = getattr(self, name)
            if not isinstance(value, (bytes, bytearray, memoryview)) or len(value) != 32:
                raise ValueError("each capability secret must be exactly 32 bytes")
            setattr(self, name, bytearray(value))

    def __repr__(self) -> str:
        return "CapabilitySecrets(<redacted>)"

    def _buffer(self, role: str) -> bytearray:
        if role == "lease_owner":
            return self._lease_owner
        if role == "child":
            return self._child
        if role == "cleanup":
            return self._cleanup
        raise KeyError(role)

    def _view_for_hash(self, role: str) -> memoryview:
        if role in self._consumed:
            raise PoweredCalibrationProbeError(
                f"{role} capability was already consumed"
            )
        return memoryview(self._buffer(role)).toreadonly()

    def consume_secret(self, role: str) -> bytearray:
        if role in self._consumed:
            raise PoweredCalibrationProbeError(
                f"{role} capability is one-use and was already consumed"
            )
        self._consumed.add(role)
        return self._buffer(role)

    def consume_frame(self, role: str) -> bytearray:
        if role not in {"child", "cleanup"}:
            raise KeyError("only child and cleanup capabilities have pipe frames")
        secret = self.consume_secret(role)
        frame = bytearray(36)
        frame[:4] = b"\x20\x00\x00\x00"
        frame[4:] = secret
        self.zeroize_role(role)
        return frame

    def zeroize_role(self, role: str) -> None:
        buffer = self._buffer(role)
        buffer[:] = b"\x00" * len(buffer)

    def zeroize_all(self) -> None:
        for role in ("lease_owner", "child", "cleanup"):
            self.zeroize_role(role)
            self._consumed.add(role)

    def is_zeroized(self, role: str) -> bool:
        return not any(self._buffer(role))


@dataclass(frozen=True)
class WrapperAbsoluteDeadlines:
    started_monotonic_ns: int
    live_contact_deadline_monotonic_ns: int
    total_deadline_monotonic_ns: int

    def as_record(self) -> dict[str, int]:
        return {
            "started_monotonic_ns": self.started_monotonic_ns,
            "live_contact_deadline_monotonic_ns": self.live_contact_deadline_monotonic_ns,
            "total_deadline_monotonic_ns": self.total_deadline_monotonic_ns,
        }


@dataclass(frozen=True)
class HeartbeatPump:
    """Provider-visible fixed cadence/deadline contract for a live phase."""

    phase: str
    deadline_monotonic_ns: int
    period_ns: int
    _emit: Callable[[], None] = field(repr=False)

    def __call__(self) -> None:
        self._emit()


@dataclass(frozen=True)
class AttemptMaterial:
    envelope: Mapping[str, Any]
    context_sha256: str
    child_argv: tuple[str, ...]
    cleanup_argv: tuple[str, ...]
    capabilities: CapabilitySecrets
    absolute_deadlines: WrapperAbsoluteDeadlines
    attempt_publish_deadline: Mapping[str, Any]


@dataclass(frozen=True)
class OfflineAdmission:
    arguments: ProbeArguments
    live_freeze: Mapping[str, Any]
    live_freeze_sha256: str
    implementation_inventory: Mapping[str, Any]
    environment_inventory: Mapping[str, Any]
    import_inventory: Mapping[str, Any]
    git: GitWorktreeProof


@dataclass(frozen=True)
class FoundationAdmission:
    wrapper_started_monotonic_ns: int
    qpc_frequency_hz: int
    offline: OfflineAdmission
    attempt_root: AttemptRootSnapshot


@dataclass
class PublicationLatch:
    attempted_paths: set[str] = field(default_factory=set)
    completed_paths: set[str] = field(default_factory=set)
    failed_paths: set[str] = field(default_factory=set)
    partial_paths: set[str] = field(default_factory=set)
    partial_sha256_by_path: dict[str, str] = field(default_factory=dict)
    poisoned: bool = False


@dataclass(frozen=True)
class FallbackFacts:
    child_created: bool
    child_tree_exit: str
    child_cleanup: str
    ports: str
    simulator_topology: str
    cleanup_capability: str
    fallback_already_attempted: bool
    wrapper_alive: bool


@dataclass(frozen=True)
class FallbackDecision:
    status: str
    spawn: bool
    reason: str
    retry_allowed: bool = False


@dataclass(frozen=True)
class TerminalDecision:
    terminal: str
    poison_required: bool
    reason_codes: tuple[str, ...]
    retry_allowed: bool = False


@dataclass(frozen=True)
class BlockedProcess:
    """A normally started, job-contained process still blocked on capability EOF."""

    handle: Any
    identity: Mapping[str, Any]
    authority: Mapping[str, Any]


@dataclass(frozen=True)
class ChildSupervisionOutcome:
    cleanup_proved: bool
    collection_valid: bool
    wrapper_death: bool = False
    reason_codes: tuple[str, ...] = ()
    artifact_state_patch: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FallbackSupervisionOutcome:
    cleanup_proved: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeaseReleaseOutcome:
    """Single-use release result, including kernel state when proof is incomplete."""

    kernel_released: bool
    released_monotonic_ns: int | None
    final_index: Mapping[str, Any] | None


@dataclass(frozen=True)
class SplitPublications:
    claim: Mapping[str, Any]
    registry: Mapping[str, Any]
    report: Mapping[str, Any]


@dataclass(frozen=True)
class LiveOrchestrationServices:
    host: "HostService"
    csprng: CSPRNGService
    launcher: LauncherService
    topology: TopologyService
    training: TrainingAttestationService
    process: ProcessService
    ports: PortService
    lease: LeaseService
    spawn: SpawnService
    supervision: SupervisionService
    postrelease: PostReleaseService


@dataclass(frozen=True)
class OrchestrationResult:
    status: str
    attempt_consumed: bool
    fallback_used: bool
    reason_codes: tuple[str, ...]
    terminal_receipt: CreateNewFileReceipt | None
    poison_receipt: CreateNewFileReceipt | None
    lifecycle_receipt: CreateNewFileReceipt | None
    ledger_events: tuple[Mapping[str, Any], ...]
    live_kernel_released: bool
    live_release_proved: bool
    no_live_after_release: bool


class OrchestrationPhaseError(PoweredCalibrationProbeError):
    """A service phase failed with one frozen invalidation reason."""

    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        wrapper_death: bool = False,
    ) -> None:
        if reason_code not in attempt_contract.INVALIDATION_REASON_CODES:
            raise ValueError(f"unknown powered invalidation reason {reason_code!r}")
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail
        self.wrapper_death = wrapper_death


class OfflineAdmissionService(Protocol):
    """Read-only host boundary used before any attempt or live contact."""

    def read_stable_json(self, path: str) -> StableJsonProof: ...

    def observe_file_identity(
        self, path: str, *, hash_kind: str
    ) -> FileIdentityProof: ...

    def current_working_directory(self) -> PathProof: ...

    def module_origin(self, module_name: str) -> PathProof: ...

    def git_worktree(self, path: str) -> GitWorktreeProof: ...

    def security_environment(self) -> Mapping[str, str | None]: ...

    def rederive_implementation_inventory(
        self, frozen_inventory: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def rederive_environment_inventory(
        self, frozen_inventory: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def rederive_import_inventory(
        self,
        frozen_inventory: Mapping[str, Any],
        eager_modules: Sequence[str],
    ) -> ImportRevalidation: ...


class QpcService(Protocol):
    def now_ns(self) -> int: ...

    def query_performance_frequency_hz(self) -> int: ...


class CSPRNGService(Protocol):
    def token_bytes(self, size: int) -> bytes: ...


class SecureCreateNewService(Protocol):
    """Handle-based Windows boundary; implementations must not repair ACLs later."""

    def inspect_attempt_root(self, paths: Mapping[str, str]) -> AttemptRootSnapshot: ...

    def open_private_directory(
        self, path: str, *, parent_path: str
    ) -> SecureDirectoryReceipt: ...

    def create_private_directory_create_new(
        self, path: str, *, parent_path: str
    ) -> SecureDirectoryReceipt: ...

    def create_new_file(
        self,
        path: str,
        payload: bytes,
        *,
        parent: SecureDirectoryReceipt,
        deadline_monotonic_ns: int,
    ) -> CreateNewFileReceipt: ...


class LauncherService(Protocol):
    def launch_and_wait(
        self,
        *,
        freeze: Mapping[str, Any],
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Any: ...


class TopologyService(Protocol):
    def prove_topology(
        self,
        *,
        launch_result: Any,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]: ...

    def prove_unchanged(
        self,
        *,
        launch_result: Any,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]: ...


class TrainingAttestationService(Protocol):
    def attest_training(
        self,
        *,
        topology_proof: Mapping[str, Any],
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]: ...


class ProcessService(Protocol):
    def current_wrapper_identity(self) -> Mapping[str, Any]: ...

    def retain_and_reprove(self, identity: Mapping[str, Any]) -> Any: ...

    def prove_prechild_identity(
        self,
        retained_wrapper: Any,
        *,
        topology_proof: Mapping[str, Any],
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]: ...

    def prove_child_tree_exit(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]: ...

    def prove_final_process_state(
        self,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]: ...

    def close_retained_wrapper(
        self, retained_wrapper: Any, *, deadline_monotonic_ns: int
    ) -> None: ...


class PortService(Protocol):
    def prove_prechild_free(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> Mapping[str, Any]: ...

    def prove_child_owners(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]: ...

    def prove_fallback_gate(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> Mapping[str, Any]: ...

    def prove_final_free(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> Mapping[str, Any]: ...


class LeaseService(Protocol):
    def acquire(
        self,
        *,
        owner_secret: bytes,
        qpc_frequency_hz: int,
        deadline_monotonic_ns: int,
    ) -> Any: ...

    def heartbeat(self, lease: Any, *, phase: str, deadline_monotonic_ns: int) -> None: ...

    def release_and_verify(
        self,
        lease: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> LeaseReleaseOutcome: ...


class SpawnService(Protocol):
    def allocate_attempt_handles(self, wrapper_process: Mapping[str, Any]) -> AttemptHandleSet: ...

    def spawn_powered_child_blocked(
        self,
        *,
        argv: Sequence[str],
        handles: AttemptHandleSet,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> BlockedProcess: ...

    def release_child_capability(
        self,
        child: Any,
        *,
        frame: bytearray,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None: ...

    def spawn_cleanup_fallback_blocked(
        self,
        *,
        argv: Sequence[str],
        handles: AttemptHandleSet,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> BlockedProcess: ...

    def release_cleanup_capability(
        self,
        child: Any,
        *,
        frame: bytearray,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None: ...

    def abort_blocked_process(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None: ...

    def close_attempt_handles(
        self, handles: AttemptHandleSet, *, deadline_monotonic_ns: int
    ) -> None: ...

    def close_process_handle(
        self, child: Any, *, deadline_monotonic_ns: int
    ) -> None: ...


class SupervisionService(Protocol):
    def supervise_powered_child(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> ChildSupervisionOutcome: ...

    def supervise_cleanup_fallback(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> FallbackSupervisionOutcome: ...


class HostService(Protocol):
    def utc_now(self) -> str: ...

    def host_boot_id_sha256(self) -> str: ...


class PostReleaseService(Protocol):
    def verify_bundle(
        self, *, phase_deadline: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def build_capture_seal(
        self, *, phase_deadline: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def analyze_capture(
        self, *, phase_deadline: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def publish_split(
        self, *, analysis: Any, phase_deadline: Mapping[str, Any]
    ) -> SplitPublications: ...

    def build_complete_terminal(self, *, context: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def build_live_poison(self, *, context: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def build_invalid_terminal(self, *, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


TRANCHE2_INTEGRATION_METHODS = MappingProxyType(
    {
        "clock": ("now_ns", "query_performance_frequency_hz"),
        "csprng": ("token_bytes",),
        "host": ("utc_now", "host_boot_id_sha256"),
        "launcher": ("launch_and_wait",),
        "topology": ("prove_topology", "prove_unchanged"),
        "training": ("attest_training",),
        "process": (
            "current_wrapper_identity",
            "retain_and_reprove",
            "prove_prechild_identity",
            "prove_child_tree_exit",
            "prove_final_process_state",
            "close_retained_wrapper",
        ),
        "ports": (
            "prove_prechild_free",
            "prove_child_owners",
            "prove_fallback_gate",
            "prove_final_free",
        ),
        "lease": ("acquire", "heartbeat", "release_and_verify"),
        "spawn": (
            "allocate_attempt_handles",
            "spawn_powered_child_blocked",
            "release_child_capability",
            "spawn_cleanup_fallback_blocked",
            "release_cleanup_capability",
            "abort_blocked_process",
            "close_attempt_handles",
            "close_process_handle",
        ),
        "supervision": ("supervise_powered_child", "supervise_cleanup_fallback"),
        "postrelease": (
            "verify_bundle",
            "build_capture_seal",
            "analyze_capture",
            "publish_split",
            "build_complete_terminal",
            "build_live_poison",
            "build_invalid_terminal",
        ),
    }
)


def validate_live_orchestration_services(
    value: LiveOrchestrationServices,
) -> LiveOrchestrationServices:
    """Reject an incompletely composed live boundary before attempt creation."""

    if not isinstance(value, LiveOrchestrationServices):
        raise LiveIntegrationUnavailable(
            "live orchestration services have the wrong aggregate type"
        )
    for role, method_names in TRANCHE2_INTEGRATION_METHODS.items():
        if role == "clock":
            continue
        provider = getattr(value, role, None)
        missing = [
            name
            for name in method_names
            if not callable(getattr(provider, name, None))
        ]
        if missing:
            raise LiveIntegrationUnavailable(
                f"live {role} provider lacks required methods {missing!r}"
            )
    return value


class OrchestrationRecordValidators:
    """Production validator adapter; tests may inject a recording equivalent."""

    def process_proof(self, value: Any) -> Mapping[str, Any]:
        return attempt_contract.validate_simulator_process_proof(value)

    def training_attestation(
        self, value: Any, *, process_proof: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return attempt_contract.validate_training_attestation(
            value, process_proof=process_proof
        )

    def process_authority(
        self,
        value: Any,
        *,
        attempt: Mapping[str, Any],
        argv: Sequence[str],
    ) -> Mapping[str, Any]:
        return attempt_contract.validate_process_authority(
            value, attempt=attempt, argv=argv
        )

    def lease_final(self, value: Any) -> Mapping[str, Any]:
        from scripts.aigp_live_lease import validate_powered_live_lease_index

        return validate_powered_live_lease_index(value)

    def bundle_verification(self, value: Any) -> Mapping[str, Any]:
        return attempt_contract.validate_bundle_verification(value)

    def capture_seal(self, value: Any) -> Mapping[str, Any]:
        return attempt_contract.validate_capture_seal(value)

    def analysis_report(self, value: Any) -> Mapping[str, Any]:
        return attempt_contract.validate_acquisition_report(value)

    def split_claim(self, value: Any) -> Mapping[str, Any]:
        return attempt_contract.validate_split_claim(value)

    def split_registry(
        self, value: Any, *, split_claim: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return attempt_contract.validate_split_registry(value, split_claim=split_claim)

    def complete_terminal(
        self, value: Any, *, lifecycle: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return attempt_contract.validate_attempt_complete(
            value, wrapper_lifecycle=lifecycle
        )

    def live_poison(self, value: Any) -> Mapping[str, Any]:
        return attempt_contract.validate_live_poison(value)

    def invalid_terminal(self, value: Any) -> Mapping[str, Any]:
        return attempt_contract.validate_attempt_invalid(value)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--live-freeze", required=True)
    parser.add_argument("--live-freeze-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> ProbeArguments:
    namespace = build_argument_parser().parse_args(argv)
    try:
        attempt_contract.validate_absolute_windows_path(
            namespace.live_freeze, path="$cli.live_freeze"
        )
    except attempt_contract.PoweredAttemptContractError as exc:
        raise OfflineAdmissionError(str(exc)) from exc
    if _SHA256_RE.fullmatch(namespace.live_freeze_sha256) is None:
        raise OfflineAdmissionError("--live-freeze-sha256 must be 64 lowercase hex")
    if _COMMIT_RE.fullmatch(namespace.expected_commit) is None:
        raise OfflineAdmissionError("--expected-commit must be 40 lowercase hex")
    return ProbeArguments(
        live_freeze=namespace.live_freeze,
        live_freeze_sha256=namespace.live_freeze_sha256,
        expected_commit=namespace.expected_commit,
    )


def _require_exact_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PoweredCalibrationProbeError(f"{name} must be an exact nonnegative integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    checked = _require_exact_nonnegative_int(value, name)
    if checked == 0:
        raise PoweredCalibrationProbeError(f"{name} must be positive")
    return checked


def _validate_path_proof(proof: PathProof, expected: str, *, kind: str) -> None:
    if proof.path != expected or proof.final_path != expected:
        raise OfflineAdmissionError(f"handle-final path mismatch for {expected!r}")
    if proof.kind != kind or not proof.exists:
        raise OfflineAdmissionError(f"expected existing {kind} at {expected!r}")
    if not proof.volume_id:
        raise OfflineAdmissionError(f"missing volume identity for {expected!r}")
    if not (proof.non_reparse and proof.ancestors_non_reparse and proof.retained_handle):
        raise OfflineAdmissionError(f"unproved/reparse path identity for {expected!r}")


def _validate_file_identity(
    proof: FileIdentityProof,
    expected_path: str,
    expected_sha256: str,
    *,
    hash_kind: str = "file_bytes",
) -> None:
    _validate_path_proof(proof.path, expected_path, kind="file")
    _require_exact_nonnegative_int(proof.size_bytes, "file size")
    if proof.hash_kind != hash_kind:
        raise OfflineAdmissionError(
            f"wrong identity hash semantics for {expected_path!r}: {proof.hash_kind!r}"
        )
    if proof.sha256 != expected_sha256 or not proof.stable_before_after:
        raise OfflineAdmissionError(f"file-byte identity drift for {expected_path!r}")


def _validate_stable_json(
    proof: StableJsonProof, expected_path: str, expected_sha256: str
) -> Mapping[str, Any]:
    _validate_file_identity(proof.identity, expected_path, expected_sha256)
    if not isinstance(proof.raw_bytes, bytes):
        raise OfflineAdmissionError(f"stable JSON bytes unavailable for {expected_path!r}")
    if len(proof.raw_bytes) != proof.identity.size_bytes:
        raise OfflineAdmissionError(f"stable JSON size mismatch for {expected_path!r}")
    if hashlib.sha256(proof.raw_bytes).hexdigest() != expected_sha256:
        raise OfflineAdmissionError(f"stable JSON hash mismatch for {expected_path!r}")
    try:
        canonical = attempt_contract.canonical_json_file_bytes(proof.value)
    except (TypeError, ValueError, attempt_contract.PoweredAttemptContractError) as exc:
        raise OfflineAdmissionError(f"invalid JSON value at {expected_path!r}: {exc}") from exc
    if proof.raw_bytes != canonical:
        raise OfflineAdmissionError(f"noncanonical JSON bytes at {expected_path!r}")
    if not isinstance(proof.value, Mapping):
        raise OfflineAdmissionError(f"JSON root must be an object at {expected_path!r}")
    return proof.value


def _semantic_subset(value: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    return {name: value[name] for name in names}


def _identity_refs(
    freeze: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any], str], ...]:
    refs: list[tuple[str, Mapping[str, Any], str]] = []
    for name in ("target_config", "capture_authorization", "excitation_plan"):
        refs.append(
            (
                f"inputs.{name}",
                freeze["inputs"][name],
                "canonical_object" if name == "excitation_plan" else "file_bytes",
            )
        )
    refs.append(
        (
            "candidate.implementation_inventory",
            freeze["candidate"]["implementation_inventory"],
            "file_bytes",
        )
    )
    for name in ("python", "powershell", "development_test_lock", "environment_inventory", "import_inventory"):
        refs.append((f"runtime.{name}", freeze["runtime"][name], "file_bytes"))
    for name in ("launcher_script", "launcher", "payload"):
        refs.append((f"simulator.{name}", freeze["simulator"][name], "file_bytes"))
    return tuple(refs)


def _require_no_analysis_ambiguities() -> None:
    # Lazy by design: importing the probe must not expand the inventory graph.
    from scripts import aigp_vq2_powered_calibration_analysis as analysis

    ambiguities = tuple(analysis.semantic_ambiguities())
    if ambiguities:
        raise OfflineAdmissionError(
            "offline analysis has unresolved semantic ambiguities: " + "; ".join(ambiguities)
        )


def _admit_offline_body(
    arguments: ProbeArguments,
    service: OfflineAdmissionService,
) -> OfflineAdmission:
    """Admit the reviewed freeze without consuming A01 or touching live state."""

    if not isinstance(arguments, ProbeArguments):
        raise TypeError("arguments must be ProbeArguments")
    freeze_document = service.read_stable_json(arguments.live_freeze)
    freeze_value = _validate_stable_json(
        freeze_document, arguments.live_freeze, arguments.live_freeze_sha256
    )
    try:
        freeze = attempt_contract.validate_live_freeze(freeze_value)
    except attempt_contract.PoweredAttemptContractError as exc:
        raise OfflineAdmissionError(str(exc)) from exc
    if freeze["paths"]["live_freeze"] != arguments.live_freeze:
        raise OfflineAdmissionError("CLI freeze path does not equal the frozen path")
    if freeze["candidate"]["commit"] != arguments.expected_commit:
        raise OfflineAdmissionError("expected commit does not equal the frozen candidate")

    invocation_validator = getattr(service, "validate_exact_invocation", None)
    if callable(invocation_validator):
        invocation_validator(freeze, arguments)

    worktree = freeze["candidate"]["live_worktree"]
    _validate_path_proof(service.current_working_directory(), worktree, kind="directory")
    expected_module = worktree + r"\scripts\aigp_vq2_powered_calibration_probe.py"
    _validate_path_proof(service.module_origin(PROBE_MODULE), expected_module, kind="file")

    git = service.git_worktree(worktree)
    if git.worktree_path != worktree:
        raise OfflineAdmissionError("Git worktree final path does not match the freeze")
    if git.head_commit != arguments.expected_commit or not git.detached_head:
        raise OfflineAdmissionError("Git HEAD is not the exact detached reviewed commit")
    if not (git.tracked_clean and git.untracked_clean and git.ignored_clean):
        raise OfflineAdmissionError("Git worktree is not clean including untracked/ignored files")
    if not git.common_dir_outside_worktree:
        raise OfflineAdmissionError("Git common directory placement is not proved")

    environment = dict(service.security_environment())
    required_security = freeze["execution"]["security_environment"]
    if environment.get("PYTHONNOUSERSITE") != required_security["PYTHONNOUSERSITE"]:
        raise OfflineAdmissionError("PYTHONNOUSERSITE does not match the freeze")
    if environment.get("PYTHONDONTWRITEBYTECODE") != required_security["PYTHONDONTWRITEBYTECODE"]:
        raise OfflineAdmissionError("PYTHONDONTWRITEBYTECODE does not match the freeze")
    for name in required_security["forbidden_defined"]:
        if environment.get(name) is not None:
            raise OfflineAdmissionError(f"forbidden environment variable is defined: {name}")

    implementation_ref = freeze["candidate"]["implementation_inventory"]
    environment_ref = freeze["runtime"]["environment_inventory"]
    import_ref = freeze["runtime"]["import_inventory"]
    implementation_value = _validate_stable_json(
        service.read_stable_json(implementation_ref["path"]),
        implementation_ref["path"],
        implementation_ref["sha256"],
    )
    environment_value = _validate_stable_json(
        service.read_stable_json(environment_ref["path"]),
        environment_ref["path"],
        environment_ref["sha256"],
    )
    import_value = _validate_stable_json(
        service.read_stable_json(import_ref["path"]),
        import_ref["path"],
        import_ref["sha256"],
    )
    try:
        implementation = attempt_contract.validate_implementation_inventory(
            implementation_value
        )
        frozen_environment = attempt_contract.validate_environment_inventory(
            environment_value
        )
        frozen_imports = attempt_contract.validate_import_inventory(import_value)
        attempt_contract.validate_live_freeze(
            freeze,
            implementation_inventory=implementation,
            environment_inventory=frozen_environment,
            import_inventory=frozen_imports,
        )
    except attempt_contract.PoweredAttemptContractError as exc:
        raise OfflineAdmissionError(str(exc)) from exc

    if git.head_tree != implementation["tree"] or git.head_commit != implementation["commit"]:
        raise OfflineAdmissionError("Git commit/tree does not match implementation inventory")

    implementation_now = attempt_contract.validate_implementation_inventory(
        service.rederive_implementation_inventory(implementation)
    )
    environment_now = attempt_contract.validate_environment_inventory(
        service.rederive_environment_inventory(frozen_environment)
    )
    import_audit = service.rederive_import_inventory(
        frozen_imports, POWERED_EAGER_IMPORT_MODULES
    )
    imports_now = attempt_contract.validate_import_inventory(import_audit.inventory)
    if _semantic_subset(implementation_now, ("commit", "tree", "entries")) != _semantic_subset(
        implementation, ("commit", "tree", "entries")
    ):
        raise OfflineAdmissionError("implementation inventory semantic payload drifted")
    if environment_now["variables"] != frozen_environment["variables"]:
        raise OfflineAdmissionError("environment inventory semantic payload drifted")
    if _semantic_subset(imports_now, ("python_sha256", "seeds", "entries")) != _semantic_subset(
        frozen_imports, ("python_sha256", "seeds", "entries")
    ):
        raise OfflineAdmissionError("import inventory semantic payload drifted")
    if (
        not import_audit.origins_reverified
        or import_audit.user_site_on_sys_path
        or import_audit.unexpected_candidate_or_venv_modules
        or import_audit.unclassified_origins
    ):
        raise OfflineAdmissionError("import origin/root revalidation is not exact")
    if imports_now["python_sha256"] != freeze["runtime"]["python"]["sha256"]:
        raise OfflineAdmissionError("import inventory Python hash does not match runtime")

    entries_by_path = {entry["path"]: entry for entry in implementation["entries"]}
    probe_entry = entries_by_path.get("scripts/aigp_vq2_powered_calibration_probe.py")
    if probe_entry is None:
        raise OfflineAdmissionError("implementation inventory omits the probe module")
    _validate_file_identity(
        service.observe_file_identity(expected_module, hash_kind="file_bytes"),
        expected_module,
        probe_entry["sha256"],
    )
    for _label, reference, hash_kind in _identity_refs(freeze):
        proof = service.observe_file_identity(
            reference["path"], hash_kind=hash_kind
        )
        _validate_file_identity(
            proof,
            reference["path"],
            reference["sha256"],
            hash_kind=hash_kind,
        )

    _require_no_analysis_ambiguities()
    return OfflineAdmission(
        arguments=arguments,
        live_freeze=freeze,
        live_freeze_sha256=arguments.live_freeze_sha256,
        implementation_inventory=implementation,
        environment_inventory=frozen_environment,
        import_inventory=frozen_imports,
        git=git,
    )


def admit_offline(
    arguments: ProbeArguments,
    service: OfflineAdmissionService,
    *,
    deadline_monotonic_ns: int | None = None,
    monotonic_ns: Callable[[], int] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> OfflineAdmission:
    """Run one semantic admission under an optional production QPC budget.

    Injected test services remain deliberately simple.  A production service
    advertises ``begin_bounded_admission``/``end_bounded_admission`` and must
    then receive an absolute deadline and the exact QPC callback.  This keeps
    the generic schema orchestration independent from Win32 while ensuring the
    real boundary cannot silently fall back to relative or wall-clock waits.
    """

    begin = getattr(service, "begin_bounded_admission", None)
    end = getattr(service, "end_bounded_admission", None)
    bounded = callable(begin) or callable(end)
    if bounded and not (callable(begin) and callable(end)):
        raise OfflineAdmissionError("offline admission budget boundary is incomplete")
    if bounded:
        if deadline_monotonic_ns is None or monotonic_ns is None:
            raise OfflineAdmissionError(
                "production offline admission requires an absolute QPC deadline"
            )
        begin(
            deadline_monotonic_ns=deadline_monotonic_ns,
            monotonic_ns=monotonic_ns,
            heartbeat=heartbeat,
        )
    succeeded = False
    try:
        result = _admit_offline_body(arguments, service)
        succeeded = True
        return result
    finally:
        if bounded:
            end(succeeded=succeeded)


def validate_attempt_gate(
    freeze: Mapping[str, Any], snapshot: AttemptRootSnapshot
) -> None:
    """Prove the one-attempt/root-poison gate without creating anything."""

    try:
        checked = attempt_contract.validate_live_freeze(freeze)
    except attempt_contract.PoweredAttemptContractError as exc:
        raise AttemptGateError(str(exc)) from exc
    if checked["session"]["attempt_limit"] != 1:
        raise AttemptGateError("the powered calibration attempt limit must be exactly one")
    if snapshot.evidence_root != checked["paths"]["evidence_root"]:
        raise AttemptGateError("attempt-root snapshot does not name the frozen root")
    if snapshot.live_poison_present:
        raise AttemptGateError("root poison exists and has no automatic clear")
    if snapshot.target_attempt_directory_present or snapshot.target_attempt_envelope_present:
        raise AttemptGateError("F00-A01 already exists; retry/replacement is forbidden")
    if snapshot.unknown_attempt_entries:
        raise AttemptGateError("unknown attempt-like root entries make consumption ambiguous")
    for prior in snapshot.prior_attempts:
        if (
            isinstance(prior.terminal_record_count, bool)
            or isinstance(prior.valid_terminal_record_count, bool)
            or prior.terminal_record_count != 1
            or prior.valid_terminal_record_count != 1
        ):
            raise AttemptGateError(
                f"prior attempt {prior.attempt_id!r} lacks exactly one valid sole terminal"
            )
    if snapshot.prior_attempts:
        raise AttemptGateError("the one-attempt limit was already consumed")


def derive_wrapper_absolute_deadlines(started_monotonic_ns: int) -> WrapperAbsoluteDeadlines:
    start = _require_exact_nonnegative_int(started_monotonic_ns, "wrapper start")
    durations = attempt_contract.DEADLINE_DURATIONS_NS
    result = WrapperAbsoluteDeadlines(
        started_monotonic_ns=start,
        live_contact_deadline_monotonic_ns=start
        + durations["wrapper_live_contact_absolute_offset"],
        total_deadline_monotonic_ns=start + durations["wrapper_total"],
    )
    # Exercise the exact attempt validator through a minimal internal call.
    attempt_contract._validate_wrapper_absolute_deadlines(
        result.as_record(), "$wrapper_absolute_deadlines"
    )
    return result


def derive_phase_deadline(
    phase: str, *, started_monotonic_ns: int, parent_deadline_monotonic_ns: int
) -> dict[str, Any]:
    if phase not in attempt_contract.WRAPPER_PHASES and phase not in {
        "offline_precheck",
        "invalid_terminal_publish",
    }:
        raise PoweredCalibrationProbeError(f"unknown wrapper phase {phase!r}")
    start = _require_exact_nonnegative_int(started_monotonic_ns, "phase start")
    parent = _require_positive_int(parent_deadline_monotonic_ns, "parent deadline")
    if start >= parent:
        raise PoweredCalibrationProbeError("phase cannot start at/after its parent deadline")
    duration_name = _ROLE_PHASE_DURATIONS.get(phase, phase)
    duration = attempt_contract.DEADLINE_DURATIONS_NS[duration_name]
    result = {
        "phase": phase,
        "started_monotonic_ns": start,
        "duration_ns": duration,
        "parent_deadline_monotonic_ns": parent,
        "deadline_monotonic_ns": min(start + duration, parent),
    }
    attempt_contract.validate_phase_deadline(result, expected_phase=phase)
    return result


def derive_terminal_parent_deadline(
    absolute: WrapperAbsoluteDeadlines, *, lease_release_monotonic_ns: int | None
) -> int:
    if lease_release_monotonic_ns is None:
        return absolute.total_deadline_monotonic_ns
    release = _require_exact_nonnegative_int(lease_release_monotonic_ns, "lease release")
    if release >= absolute.live_contact_deadline_monotonic_ns:
        raise PoweredCalibrationProbeError("lease release must precede live-contact cutoff")
    return min(
        absolute.total_deadline_monotonic_ns,
        release + attempt_contract.DEADLINE_DURATIONS_NS["postrelease_total"],
    )


def derive_child_argv(
    freeze: Mapping[str, Any],
    wrapper_process: Mapping[str, Any],
    handles: AttemptHandleSet,
) -> tuple[str, ...]:
    attempt_contract.validate_process_identity(wrapper_process)
    _validate_handle_set(handles)
    paths = freeze["paths"]
    python_path = freeze["runtime"]["python"]["path"]
    wrapper = f"{wrapper_process['pid']}:{wrapper_process['creation_filetime_100ns']}"
    return (
        python_path,
        "-E",
        "-s",
        "-B",
        "-m",
        "scripts.aigp_vq2_run",
        "--stage",
        "calibration-excite",
        "--powered-attempt-envelope",
        paths["attempt_envelope"],
        "--wrapper-process",
        wrapper,
        "--powered-process-authority",
        paths["child_authority"],
        "--attempt-capability-handle",
        str(handles.child_capability_read_handle),
        "--parent-liveness-handle",
        str(handles.child_parent_liveness_handle),
        "--record",
        paths["legacy_record"],
        "--replay-bundle",
        paths["replay_bundle"],
        "--cleanup-certificate",
        paths["child_cleanup_certificate"],
        "--recording-approved",
    )


def derive_cleanup_argv(
    freeze: Mapping[str, Any],
    wrapper_process: Mapping[str, Any],
    handles: AttemptHandleSet,
) -> tuple[str, ...]:
    attempt_contract.validate_process_identity(wrapper_process)
    _validate_handle_set(handles)
    paths = freeze["paths"]
    python_path = freeze["runtime"]["python"]["path"]
    wrapper = f"{wrapper_process['pid']}:{wrapper_process['creation_filetime_100ns']}"
    return (
        python_path,
        "-E",
        "-s",
        "-B",
        "-m",
        "scripts.aigp_vq2_powered_cleanup",
        "--powered-attempt-envelope",
        paths["attempt_envelope"],
        "--wrapper-process",
        wrapper,
        "--powered-process-authority",
        paths["cleanup_authority"],
        "--cleanup-capability-handle",
        str(handles.cleanup_capability_read_handle),
        "--parent-liveness-handle",
        str(handles.cleanup_parent_liveness_handle),
        "--cleanup-certificate",
        paths["fallback_cleanup_certificate"],
    )


def _validate_handle_set(handles: AttemptHandleSet) -> None:
    values = (
        handles.child_capability_read_handle,
        handles.child_parent_liveness_handle,
        handles.cleanup_capability_read_handle,
        handles.cleanup_parent_liveness_handle,
    )
    for index, value in enumerate(values):
        _require_positive_int(value, f"attempt handle {index}")
    if len(set(values)) != len(values):
        raise PoweredCalibrationProbeError("all inherited attempt handles must be distinct")


def generate_capability_secrets(
    random_bytes: Callable[[int], bytes],
) -> CapabilitySecrets:
    generated: list[bytes] = []
    for _ in range(3):
        value = random_bytes(32)
        if not isinstance(value, bytes) or len(value) != 32:
            raise PoweredCalibrationProbeError("CSPRNG must return exactly 32 bytes")
        generated.append(value)
    if len(set(generated)) != 3:
        raise PoweredCalibrationProbeError("the three CSPRNG capabilities must be independent")
    return CapabilitySecrets(generated[0], generated[1], generated[2])


def build_attempt_material(
    *,
    admission: OfflineAdmission,
    wrapper_process: Mapping[str, Any],
    host_boot_id_sha256: str,
    qpc_frequency_hz: int,
    handles: AttemptHandleSet,
    created_at_utc: str,
    wrapper_started_monotonic_ns: int,
    offline_precheck_completed_monotonic_ns: int,
    attempt_publish_started_monotonic_ns: int,
    random_bytes: Callable[[int], bytes],
) -> AttemptMaterial:
    freeze = admission.live_freeze
    attempt_contract.validate_process_identity(wrapper_process)
    if _SHA256_RE.fullmatch(host_boot_id_sha256) is None:
        raise PoweredCalibrationProbeError("host boot identity must be 64 lowercase hex")
    frequency = _require_positive_int(qpc_frequency_hz, "QPC frequency")
    absolute = derive_wrapper_absolute_deadlines(wrapper_started_monotonic_ns)
    offline = derive_phase_deadline(
        "offline_precheck",
        started_monotonic_ns=wrapper_started_monotonic_ns,
        parent_deadline_monotonic_ns=absolute.live_contact_deadline_monotonic_ns,
    )
    offline_completed = _require_exact_nonnegative_int(
        offline_precheck_completed_monotonic_ns, "offline precheck completion"
    )
    if not offline["started_monotonic_ns"] <= offline_completed < offline["deadline_monotonic_ns"]:
        raise PoweredCalibrationProbeError("offline precheck did not complete within its deadline")
    attempt_started = _require_exact_nonnegative_int(
        attempt_publish_started_monotonic_ns, "attempt publication start"
    )
    if attempt_started < offline_completed:
        raise PoweredCalibrationProbeError("attempt publication cannot precede offline admission")
    attempt_publish = derive_phase_deadline(
        "attempt_publish",
        started_monotonic_ns=attempt_started,
        parent_deadline_monotonic_ns=absolute.live_contact_deadline_monotonic_ns,
    )
    child_argv = derive_child_argv(freeze, wrapper_process, handles)
    cleanup_argv = derive_cleanup_argv(freeze, wrapper_process, handles)
    context = {
        "task_id": attempt_contract.TASK_ID,
        "session_id": attempt_contract.SESSION_ID,
        "attempt_id": attempt_contract.ATTEMPT_ID,
        "created_at_utc": created_at_utc,
        "host": {
            "host_clock_id": attempt_contract.HOST_CLOCK_ID,
            "host_boot_id_sha256": host_boot_id_sha256,
            "qpc_frequency_hz": frequency,
        },
        "live_freeze": {
            "path": freeze["paths"]["live_freeze"],
            "sha256": admission.live_freeze_sha256,
        },
        "candidate_commit": freeze["candidate"]["commit"],
        "target_config": {
            "path": freeze["inputs"]["target_config"]["path"],
            "sha256": freeze["inputs"]["target_config"]["sha256"],
        },
        "capture_authorization": {
            "path": freeze["inputs"]["capture_authorization"]["path"],
            "sha256": freeze["inputs"]["capture_authorization"]["sha256"],
        },
        "excitation_plan": {
            "path": freeze["inputs"]["excitation_plan"]["path"],
            # This is deliberately the canonical object SHA (no LF).
            "sha256": freeze["inputs"]["excitation_plan"]["sha256"],
            "plan_id": freeze["inputs"]["excitation_plan"]["plan_id"],
        },
        "wrapper_process": dict(wrapper_process),
        "paths": dict(freeze["paths"]),
        "child_argv": list(child_argv),
        "cleanup_argv": list(cleanup_argv),
        "deadline_durations_ns": dict(freeze["deadline_durations_ns"]),
        "wrapper_absolute_deadlines": absolute.as_record(),
        "prepublication_timing": {
            "wrapper_started_monotonic_ns": wrapper_started_monotonic_ns,
            "offline_precheck": {
                **offline,
                "completed_monotonic_ns": offline_completed,
                "outcome": "completed",
            },
            "attempt_publish": attempt_publish,
        },
    }
    context_sha256 = attempt_contract.canonical_object_sha256(context)
    capabilities = generate_capability_secrets(random_bytes)
    try:
        envelope = {
            "schema": "aigp-vq2-powered-calibration-attempt/1",
            "context": context,
            "context_sha256": context_sha256,
            "capabilities": {
                "algorithm": "sha256-domain-separated-context-v1",
                "lease_owner_sha256": attempt_contract.derive_capability_sha256(
                    _CAPABILITY_DOMAINS["lease_owner"],
                    context_sha256,
                    capabilities._view_for_hash("lease_owner"),
                ),
                "child_sha256": attempt_contract.derive_capability_sha256(
                    _CAPABILITY_DOMAINS["child"],
                    context_sha256,
                    capabilities._view_for_hash("child"),
                ),
                "cleanup_sha256": attempt_contract.derive_capability_sha256(
                    _CAPABILITY_DOMAINS["cleanup"],
                    context_sha256,
                    capabilities._view_for_hash("cleanup"),
                ),
            },
        }
        checked = attempt_contract.validate_attempt(envelope, live_freeze=freeze)
    except Exception:
        capabilities.zeroize_all()
        raise
    return AttemptMaterial(
        envelope=checked,
        context_sha256=context_sha256,
        child_argv=child_argv,
        cleanup_argv=cleanup_argv,
        capabilities=capabilities,
        absolute_deadlines=absolute,
        attempt_publish_deadline=attempt_publish,
    )


def _validate_secure_directory_receipt(
    receipt: SecureDirectoryReceipt,
    *,
    expected_path: str,
    expected_parent: str,
    require_created_new: bool = True,
) -> None:
    if (
        receipt.path != expected_path
        or receipt.final_path != expected_path
        or receipt.parent_final_path != expected_parent
    ):
        raise SecureBoundaryError("secure directory path/final-parent mismatch")
    if not receipt.volume_id or receipt.volume_id != receipt.parent_volume_id:
        raise SecureBoundaryError("secure directory crosses or lacks a proved volume")
    if receipt.owner_id != receipt.current_user_id:
        raise SecureBoundaryError("secure directory owner identity mismatch")
    if receipt.created_new is not require_created_new:
        raise SecureBoundaryError("secure directory creation/open state is wrong")
    if not all(
        (
            receipt.owner_is_current_user,
            receipt.current_user_only_dacl,
            receipt.dacl_applied_at_create,
            receipt.non_reparse,
            receipt.ancestors_non_reparse,
            receipt.retained_handle,
        )
    ):
        raise SecureBoundaryError("secure create-new directory invariants are incomplete")


def _validate_file_receipt(
    receipt: CreateNewFileReceipt,
    *,
    path: str,
    parent: SecureDirectoryReceipt,
    payload: bytes,
    deadline_monotonic_ns: int,
) -> None:
    if (
        receipt.path != path
        or receipt.final_path != path
        or receipt.parent_final_path != parent.final_path
    ):
        raise SecureBoundaryError("published file path/final-parent mismatch")
    if receipt.volume_id != parent.volume_id or receipt.parent_volume_id != parent.volume_id:
        raise SecureBoundaryError("published file volume identity mismatch")
    if receipt.owner_id != receipt.current_user_id or receipt.owner_id != parent.current_user_id:
        raise SecureBoundaryError("published file owner identity mismatch")
    expected_hash = hashlib.sha256(payload).hexdigest()
    if receipt.size_bytes != len(payload) or receipt.sha256 != expected_hash:
        raise SecureBoundaryError("published file readback bytes/hash mismatch")
    completed = _require_exact_nonnegative_int(
        receipt.completed_monotonic_ns, "publication completion"
    )
    if completed >= deadline_monotonic_ns:
        raise SecureBoundaryError("publication completed at/after its absolute deadline")
    if not all(
        (
            receipt.created_new,
            receipt.regular_file,
            receipt.owner_is_current_user,
            receipt.current_user_only_dacl,
            receipt.dacl_applied_at_create,
            receipt.non_reparse,
            receipt.ancestors_non_reparse,
            receipt.flushed,
            receipt.readback_verified,
        )
    ):
        raise SecureBoundaryError("create-new file invariants are incomplete")


class CreateNewJsonPublisher:
    """One-shot canonical JSON publisher with a shared no-retry poison latch."""

    def __init__(
        self,
        service: SecureCreateNewService,
        parent: SecureDirectoryReceipt,
        *,
        latch: PublicationLatch | None = None,
        recovery_lane: bool = False,
    ) -> None:
        self._service = service
        self.parent = parent
        self.latch = latch if latch is not None else PublicationLatch()
        self.recovery_lane = recovery_lane

    def publish(
        self,
        path: str,
        value: Any,
        *,
        deadline_monotonic_ns: int,
        validator: Callable[[Any], Any] | None = None,
    ) -> CreateNewFileReceipt:
        if self.latch.poisoned and not self.recovery_lane:
            raise PublicationError(
                "publication latch is poisoned; only a distinct recovery target is allowed",
                path=path,
                state="blocked",
            )
        if path in self.latch.attempted_paths:
            raise PublicationError(
                "create-new target was already attempted and cannot be retried",
                path=path,
                state="blocked",
            )
        parent_prefix = self.parent.path.rstrip("\\") + "\\"
        if not path.startswith(parent_prefix) or ntpath.dirname(path) != self.parent.path:
            raise PublicationError(
                "publication target is not an immediate child of the proved directory",
                path=path,
                state="absent",
            )
        checked = validator(value) if validator is not None else value
        payload = attempt_contract.canonical_json_file_bytes(checked)
        deadline = _require_positive_int(deadline_monotonic_ns, "publication deadline")
        self.latch.attempted_paths.add(path)
        try:
            receipt = self._service.create_new_file(
                path,
                payload,
                parent=self.parent,
                deadline_monotonic_ns=deadline,
            )
            _validate_file_receipt(
                receipt,
                path=path,
                parent=self.parent,
                payload=payload,
                deadline_monotonic_ns=deadline,
            )
        except BoundaryCreateNewError as exc:
            self.latch.poisoned = True
            self.latch.failed_paths.add(path)
            if exc.state in {"partial", "unknown"}:
                self.latch.partial_paths.add(path)
                if exc.observed_sha256 is not None:
                    self.latch.partial_sha256_by_path[path] = exc.observed_sha256
                raise PartialPublicationError(
                    "create-new publication may contain partial forensic bytes; no retry",
                    path=path,
                    state=exc.state,
                    observed_sha256=exc.observed_sha256,
                ) from exc
            raise PublicationError(
                "create-new publication failed; no retry",
                path=path,
                state=exc.state,
                observed_sha256=exc.observed_sha256,
            ) from exc
        except Exception as exc:
            self.latch.poisoned = True
            self.latch.failed_paths.add(path)
            self.latch.partial_paths.add(path)
            raise PartialPublicationError(
                "publication receipt/readback is unproved; preserve bytes and do not retry",
                path=path,
                state="unknown",
            ) from exc
        self.latch.completed_paths.add(path)
        return receipt


class AttemptWorkspace:
    """Consumed A01 directory plus publishers sharing one irreversible latch."""

    def __init__(
        self,
        service: SecureCreateNewService,
        freeze: Mapping[str, Any],
        root_directory: SecureDirectoryReceipt,
        directory: SecureDirectoryReceipt,
    ) -> None:
        self.service = service
        self.freeze = freeze
        self.root_directory = root_directory
        self.directory = directory
        self.latch = PublicationLatch()
        self.attempt_publisher = CreateNewJsonPublisher(
            service, directory, latch=self.latch
        )
        self._subdirectories: dict[str, SecureDirectoryReceipt] = {}

    @classmethod
    def consume(
        cls, service: SecureCreateNewService, freeze: Mapping[str, Any]
    ) -> "AttemptWorkspace":
        checked = attempt_contract.validate_live_freeze(freeze)
        snapshot = service.inspect_attempt_root(checked["paths"])
        validate_attempt_gate(checked, snapshot)
        path = checked["paths"]["attempt_dir"]
        parent = checked["paths"]["evidence_root"]
        try:
            root = service.open_private_directory(
                parent, parent_path=ntpath.dirname(parent)
            )
            _validate_secure_directory_receipt(
                root,
                expected_path=parent,
                expected_parent=ntpath.dirname(parent),
                require_created_new=False,
            )
            receipt = service.create_private_directory_create_new(
                path, parent_path=parent
            )
            _validate_secure_directory_receipt(
                receipt, expected_path=path, expected_parent=parent
            )
        except BoundaryCreateNewError as exc:
            raise AttemptGateError(
                "attempt-directory creation is absent/partial/unknown; A01 cannot be retried"
            ) from exc
        return cls(service, checked, root, receipt)

    def create_subdirectory(self, path_key: str) -> SecureDirectoryReceipt:
        if self.latch.poisoned:
            raise PublicationError(
                "attempt workspace is poisoned",
                path=self.freeze["paths"][path_key],
                state="blocked",
            )
        if path_key not in {"lease_directory", "wrapper_ledger_directory"}:
            raise SecureBoundaryError("unsupported protected attempt subdirectory")
        if path_key in self._subdirectories:
            raise SecureBoundaryError("protected subdirectory is create-new and single-use")
        path = self.freeze["paths"][path_key]
        try:
            receipt = self.service.create_private_directory_create_new(
                path, parent_path=self.directory.final_path
            )
            _validate_secure_directory_receipt(
                receipt,
                expected_path=path,
                expected_parent=self.directory.final_path,
            )
        except Exception as exc:
            self.latch.poisoned = True
            raise SecureBoundaryError(
                "protected subdirectory creation is unproved; attempt is poisoned"
            ) from exc
        self._subdirectories[path_key] = receipt
        return receipt

    def publish_attempt(
        self, material: AttemptMaterial
    ) -> CreateNewFileReceipt:
        return self.attempt_publisher.publish(
            self.freeze["paths"]["attempt_envelope"],
            material.envelope,
            deadline_monotonic_ns=material.attempt_publish_deadline[
                "deadline_monotonic_ns"
            ],
            validator=lambda value: attempt_contract.validate_attempt(
                value, live_freeze=self.freeze
            ),
        )

    def publisher_for(self, directory: SecureDirectoryReceipt) -> CreateNewJsonPublisher:
        return CreateNewJsonPublisher(
            self.service, directory, latch=self.latch
        )

    def recovery_publisher_for(
        self, directory: SecureDirectoryReceipt
    ) -> CreateNewJsonPublisher:
        """Return the poison/invalid lane; attempted paths still cannot repeat."""

        return CreateNewJsonPublisher(
            self.service,
            directory,
            latch=self.latch,
            recovery_lane=True,
        )

    def open_split_registry_directory(self) -> SecureDirectoryReceipt:
        path = ntpath.dirname(self.freeze["paths"]["split_registry"])
        receipt = self.service.open_private_directory(
            path, parent_path=self.root_directory.final_path
        )
        _validate_secure_directory_receipt(
            receipt,
            expected_path=path,
            expected_parent=self.root_directory.final_path,
            require_created_new=False,
        )
        return receipt


class WrapperTimeline:
    def __init__(self, absolute: WrapperAbsoluteDeadlines) -> None:
        self.absolute = absolute
        self.lease_release_monotonic_ns: int | None = None

    @property
    def terminal_parent_deadline(self) -> int:
        return derive_terminal_parent_deadline(
            self.absolute,
            lease_release_monotonic_ns=self.lease_release_monotonic_ns,
        )

    def note_lease_release(self, observed_monotonic_ns: int) -> None:
        if self.lease_release_monotonic_ns is not None:
            raise PoweredCalibrationProbeError("lease release deadline cannot be refreshed")
        self.lease_release_monotonic_ns = _require_exact_nonnegative_int(
            observed_monotonic_ns, "lease release"
        )
        derive_terminal_parent_deadline(
            self.absolute,
            lease_release_monotonic_ns=self.lease_release_monotonic_ns,
        )

    def parent_for(self, phase: str) -> int:
        if phase in _POSTRELEASE_PHASES:
            if phase in {"bundle_verify", "capture_seal", "analysis", "split_publish", "terminal_ready"} and self.lease_release_monotonic_ns is None:
                raise PoweredCalibrationProbeError(
                    f"{phase} requires proved lease release"
                )
            return self.terminal_parent_deadline
        return self.absolute.live_contact_deadline_monotonic_ns


class WrapperLedger:
    """Append-only canonical wrapper events and their immutable lifecycle index."""

    def __init__(
        self,
        *,
        publisher: CreateNewJsonPublisher,
        lifecycle_publisher: CreateNewJsonPublisher,
        ledger_directory: str,
        lifecycle_path: str,
        timeline: WrapperTimeline,
        clock: QpcService,
    ) -> None:
        self.publisher = publisher
        self.lifecycle_publisher = lifecycle_publisher
        self.ledger_directory = ledger_directory
        self.lifecycle_path = lifecycle_path
        self.timeline = timeline
        self.clock = clock
        self.events: list[dict[str, Any]] = []
        self.receipts: list[CreateNewFileReceipt] = []
        self.active_deadline: dict[str, Any] | None = None
        self._last_phase_rank = -1
        self._finalized = False
        self.lifecycle_value: Mapping[str, Any] | None = None

    def _event_path(self, sequence: int) -> str:
        return self.ledger_directory + f"\\event-{sequence:06d}.json"

    def _append(self, event: Mapping[str, Any]) -> CreateNewFileReceipt:
        if self._finalized:
            raise PublicationError(
                "wrapper lifecycle is already frozen",
                path=self._event_path(len(self.events)),
                state="blocked",
            )
        prior_hash = self.receipts[-1].sha256 if self.receipts else None
        checked = attempt_contract.validate_wrapper_event(
            event, prior_file_sha256=prior_hash
        )
        receipt = self.publisher.publish(
            self._event_path(len(self.events)),
            checked,
            deadline_monotonic_ns=checked["deadline_monotonic_ns"],
            validator=lambda value: attempt_contract.validate_wrapper_event(
                value, prior_file_sha256=prior_hash
            ),
        )
        self.events.append(checked)
        self.receipts.append(receipt)
        return receipt

    def record_attempt_publish_end(
        self,
        *,
        attempt_publish_deadline: Mapping[str, Any],
        observed_monotonic_ns: int,
    ) -> CreateNewFileReceipt:
        if self.events:
            raise PoweredCalibrationProbeError("attempt_publish end must be sequence zero")
        deadline = attempt_contract.validate_phase_deadline(
            attempt_publish_deadline, expected_phase="attempt_publish"
        )
        observed = _require_exact_nonnegative_int(
            observed_monotonic_ns, "attempt publication end"
        )
        event = {
            "schema": "aigp-vq2-powered-wrapper-event/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "event_sequence": 0,
            "predecessor_sha256": None,
            "event": "phase_end",
            "phase": "attempt_publish",
            "observed_monotonic_ns": observed,
            "duration_ns": deadline["duration_ns"],
            "parent_deadline_monotonic_ns": deadline[
                "parent_deadline_monotonic_ns"
            ],
            "deadline_monotonic_ns": deadline["deadline_monotonic_ns"],
            "outcome": "completed",
            "reason_code": None,
            "artifacts": [],
        }
        result = self._append(event)
        self._last_phase_rank = attempt_contract.WRAPPER_PHASES.index("attempt_publish")
        return result

    def start_phase(self, phase: str) -> Mapping[str, Any]:
        if not self.events or self.active_deadline is not None:
            raise PoweredCalibrationProbeError("ledger is not ready to start a phase")
        if phase not in attempt_contract.WRAPPER_PHASES or phase == "attempt_publish":
            raise PoweredCalibrationProbeError("invalid paired wrapper phase")
        rank = attempt_contract.WRAPPER_PHASES.index(phase)
        if rank <= self._last_phase_rank:
            raise PoweredCalibrationProbeError("wrapper phases cannot repeat or go backward")
        observed = _require_exact_nonnegative_int(self.clock.now_ns(), "QPC observation")
        deadline = derive_phase_deadline(
            phase,
            started_monotonic_ns=observed,
            parent_deadline_monotonic_ns=self.timeline.parent_for(phase),
        )
        event = {
            "schema": "aigp-vq2-powered-wrapper-event/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "event_sequence": len(self.events),
            "predecessor_sha256": self.receipts[-1].sha256,
            "event": "phase_start",
            "phase": deadline["phase"],
            "observed_monotonic_ns": observed,
            "duration_ns": deadline["duration_ns"],
            "parent_deadline_monotonic_ns": deadline[
                "parent_deadline_monotonic_ns"
            ],
            "deadline_monotonic_ns": deadline["deadline_monotonic_ns"],
            "outcome": None,
            "reason_code": None,
            "artifacts": [],
        }
        self._append(event)
        self.active_deadline = deadline
        self._last_phase_rank = rank
        return dict(deadline)

    def end_phase(
        self,
        *,
        outcome: str,
        reason_code: str | None = None,
        artifacts: Sequence[Mapping[str, Any]] = (),
    ) -> CreateNewFileReceipt:
        if self.active_deadline is None:
            raise PoweredCalibrationProbeError("no wrapper phase is active")
        observed = _require_exact_nonnegative_int(self.clock.now_ns(), "QPC observation")
        deadline = self.active_deadline
        ordered_artifacts = sorted(
            (dict(item) for item in artifacts), key=lambda item: item["name"].encode("utf-8")
        )
        event = {
            "schema": "aigp-vq2-powered-wrapper-event/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "event_sequence": len(self.events),
            "predecessor_sha256": self.receipts[-1].sha256,
            "event": "phase_end",
            "phase": deadline["phase"],
            "observed_monotonic_ns": observed,
            "duration_ns": deadline["duration_ns"],
            "parent_deadline_monotonic_ns": deadline[
                "parent_deadline_monotonic_ns"
            ],
            "deadline_monotonic_ns": deadline["deadline_monotonic_ns"],
            "outcome": outcome,
            "reason_code": reason_code,
            "artifacts": ordered_artifacts,
        }
        result = self._append(event)
        if (
            deadline["phase"] == "lease_release_and_verify"
            and outcome == "completed"
            and self.timeline.lease_release_monotonic_ns is None
        ):
            self.timeline.note_lease_release(observed)
        self.active_deadline = None
        return result

    def finalize_lifecycle(self) -> CreateNewFileReceipt:
        if self._finalized or not self.events or self.active_deadline is not None:
            raise PoweredCalibrationProbeError("wrapper lifecycle cannot be finalized now")
        records: list[dict[str, Any]] = []
        for event, receipt in zip(self.events, self.receipts, strict=True):
            records.append(
                {
                    "event_sequence": event["event_sequence"],
                    "path": receipt.path,
                    "sha256": receipt.sha256,
                    "event": event["event"],
                    "phase": event["phase"],
                    "observed_monotonic_ns": event["observed_monotonic_ns"],
                    "outcome": event["outcome"],
                    "reason_code": event["reason_code"],
                    "artifacts": event["artifacts"],
                }
            )
        lifecycle = {
            "schema": "aigp-vq2-powered-wrapper-lifecycle/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "records": records,
            "final_sequence": len(records) - 1,
            "final_record_sha256": records[-1]["sha256"],
            "live_contact_deadline_monotonic_ns": self.timeline.absolute.live_contact_deadline_monotonic_ns,
            "total_deadline_monotonic_ns": self.timeline.absolute.total_deadline_monotonic_ns,
        }
        checked = attempt_contract.validate_wrapper_lifecycle(
            lifecycle, ledger_events=self.events
        )
        receipt = self.lifecycle_publisher.publish(
            self.lifecycle_path,
            checked,
            deadline_monotonic_ns=self.timeline.terminal_parent_deadline,
            validator=lambda value: attempt_contract.validate_wrapper_lifecycle(
                value, ledger_events=self.events
            ),
        )
        self.lifecycle_value = checked
        self._finalized = True
        return receipt

    def enter_recovery_lane(
        self,
        *,
        event_publisher: CreateNewJsonPublisher,
        lifecycle_publisher: CreateNewJsonPublisher,
    ) -> None:
        if self._finalized:
            raise PoweredCalibrationProbeError("finalized lifecycle cannot enter recovery")
        self.publisher = event_publisher
        self.lifecycle_publisher = lifecycle_publisher


def artifact_ref(name: str, receipt: CreateNewFileReceipt) -> dict[str, Any]:
    value = {
        "name": name,
        "path": receipt.path,
        "size_bytes": receipt.size_bytes,
        "sha256": receipt.sha256,
    }
    return attempt_contract.validate_artifact_ref(value)


def decide_fallback(facts: FallbackFacts) -> FallbackDecision:
    """Pure, single-use wrapper-side cleanup-fallback eligibility decision."""

    allowed = {
        "child_tree_exit": {"not_created", "proved", "live", "unproved"},
        "child_cleanup": {"absent", "invalid", "valid"},
        "ports": {"not_opened", "free", "owned", "unproved"},
        "simulator_topology": {"not_launched", "unchanged", "changed", "unproved"},
        "cleanup_capability": {"available", "consumed", "mismatched", "unavailable"},
    }
    for name, values in allowed.items():
        if getattr(facts, name) not in values:
            raise PoweredCalibrationProbeError(f"invalid fallback fact {name}")
    if facts.fallback_already_attempted:
        return FallbackDecision("not_eligible", False, "fallback_already_attempted")
    if not facts.child_created:
        return FallbackDecision("not_eligible", False, "child_not_created")
    if facts.child_tree_exit != "proved":
        return FallbackDecision("not_eligible", False, "child_tree_exit_unproved")
    if facts.child_cleanup == "valid":
        return FallbackDecision("not_required", False, "child_cleanup_proved")
    if not facts.wrapper_alive:
        return FallbackDecision("not_eligible", False, "wrapper_not_alive")
    if facts.ports != "free":
        return FallbackDecision("not_eligible", False, "ports_not_free")
    if facts.simulator_topology != "unchanged":
        return FallbackDecision("not_eligible", False, "simulator_not_unchanged")
    if facts.cleanup_capability != "available":
        return FallbackDecision("not_eligible", False, "cleanup_capability_unavailable")
    return FallbackDecision("eligible_once", True, "child_cleanup_not_proved")


def derive_poison_required(
    *,
    cleanup_state: Mapping[str, Any],
    artifact_state: Mapping[str, Any],
    reason_codes: Sequence[str],
    attempt_envelope_state: str,
) -> bool:
    return attempt_contract.derive_poison_required(
        cleanup_state,
        artifact_state,
        reason_codes,
        attempt_envelope_state=attempt_envelope_state,
    )


def decide_terminal(
    *,
    completion_ready: bool,
    fallback_used: bool,
    cleanup_state: Mapping[str, Any],
    artifact_state: Mapping[str, Any],
    reason_codes: Sequence[str],
    attempt_envelope_state: str,
    publication_poisoned: bool = False,
) -> TerminalDecision:
    """Choose exactly complete or invalid; this function never grants a retry."""

    if not isinstance(completion_ready, bool) or not isinstance(fallback_used, bool):
        raise PoweredCalibrationProbeError("terminal booleans must be exact")
    # Validate both state records even on a caller-claimed success path.  The
    # real poison derivation is also the strict artifact-state validator.
    attempt_contract.validate_invalid_cleanup_state(cleanup_state)
    validation_reasons = tuple(reason_codes) or ("internal_error",)
    attempt_contract.derive_poison_required(
        cleanup_state,
        artifact_state,
        validation_reasons,
        attempt_envelope_state=attempt_envelope_state,
    )
    reasons = set(reason_codes)
    if publication_poisoned:
        reasons.add("terminal_write_failed")
    complete_cleanup = cleanup_state == {
        "child_exit": "proved",
        "fallback": "not_required",
        "ports": "free",
        "lease": "released",
        "processes": "exited",
        "transport": "closed",
        "scheduled_task": "absent",
        "simulator_topology": "unchanged",
        "simulator_responsive": "yes",
    }
    complete_artifacts = (
        artifact_state["legacy_record"] == "closed"
        and artifact_state["replay_bundle"] == "sealed"
        and all(
            artifact_state[name] == "valid"
            for name in (
                "bundle_verification",
                "capture_seal",
                "split_claim",
                "split_registry",
                "analysis_report",
                "wrapper_lifecycle",
            )
        )
        and artifact_state["attempt_complete"] == "absent"
    )
    exact_completion = (
        completion_ready
        and complete_cleanup
        and complete_artifacts
        and attempt_envelope_state == "valid"
        and not fallback_used
        and not reasons
        and not publication_poisoned
    )
    if exact_completion:
        return TerminalDecision("complete", False, ())
    if not reasons:
        reasons.add("internal_error")
    ordered = tuple(sorted(reasons, key=lambda item: item.encode("utf-8")))
    required = attempt_contract.derive_poison_required(
        cleanup_state,
        artifact_state,
        ordered,
        attempt_envelope_state=attempt_envelope_state,
    )
    if publication_poisoned:
        required = True
    return TerminalDecision("invalid", required, ordered)


def _initial_artifact_state() -> dict[str, Any]:
    return {
        "legacy_record": "absent",
        "legacy_record_sha256": None,
        "replay_bundle": "absent",
        "replay_dataset_hash": None,
        "replay_manifest_sha256": None,
        "replay_records_sha256": None,
        "bundle_verification": "absent",
        "bundle_verification_sha256": None,
        "capture_seal": "absent",
        "capture_seal_sha256": None,
        "split_claim": "absent",
        "split_claim_sha256": None,
        "split_registry": "absent",
        "split_registry_sha256": None,
        "analysis_report": "absent",
        "analysis_report_sha256": None,
        "wrapper_lifecycle": "absent",
        "wrapper_lifecycle_sha256": None,
        "attempt_complete": "absent",
        "attempt_complete_partial_sha256": None,
        "terminal_publication": "invalid_record",
        "forensic_bytes_preserved": True,
    }


def _initial_cleanup_state() -> dict[str, str]:
    return {
        "child_exit": "not_created",
        "fallback": "not_eligible",
        "ports": "not_opened",
        "lease": "not_acquired",
        "processes": "not_created",
        "transport": "not_opened",
        "scheduled_task": "not_created",
        "simulator_topology": "not_launched",
        "simulator_responsive": "not_launched",
    }


def _proved(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("proved") is not True:
        raise OrchestrationPhaseError("internal_error", f"{label} was not proved")
    return value


def _sanitize_detail(value: Any) -> str:
    text_value = str(value).replace("\x00", "?")
    encoded = text_value.encode("utf-8", errors="replace")[:4096]
    return encoded.decode("utf-8", errors="ignore")


class _SingleAttemptExecution:
    """One in-memory wrapper run; every live effect remains service-injected."""

    def __init__(
        self,
        *,
        arguments: ProbeArguments,
        foundation: FoundationAdmission,
        offline: OfflineAdmissionService,
        secure: SecureCreateNewService,
        clock: QpcService,
        services: LiveOrchestrationServices,
        validators: OrchestrationRecordValidators,
    ) -> None:
        self.arguments = arguments
        self.foundation = foundation
        self.offline = offline
        self.secure = secure
        self.clock = clock
        self.services = services
        self.validators = validators
        self.freeze = foundation.offline.live_freeze
        self.paths = self.freeze["paths"]
        self.workspace: AttemptWorkspace | None = None
        self.material: AttemptMaterial | None = None
        self.ledger: WrapperLedger | None = None
        self.ledger_directory: SecureDirectoryReceipt | None = None
        self.ledger_usable = False
        self.lifecycle_receipt: CreateNewFileReceipt | None = None
        self.attempt_receipt: CreateNewFileReceipt | None = None
        self.terminal_receipt: CreateNewFileReceipt | None = None
        self.poison_receipt: CreateNewFileReceipt | None = None
        self.lease: Any = None
        self.lease_acquired = False
        self.last_lease_heartbeat_monotonic_ns: int | None = None
        self.release_invoked = False
        self.kernel_released = False
        self.release_proved = False
        self.launcher_invoked = False
        self.launch_result: Any = None
        self.topology_proof: Mapping[str, Any] | None = None
        self.training_attestation: Mapping[str, Any] | None = None
        self.retained_wrapper: Any = None
        self.handles: AttemptHandleSet | None = None
        self.child: BlockedProcess | None = None
        self.fallback: BlockedProcess | None = None
        self.child_capability_released = False
        self.fallback_capability_released = False
        self.child_abort_completed = False
        self.fallback_abort_completed = False
        self.child_supervision_called = False
        self.fallback_supervision_called = False
        self.child_exit_attempted = False
        self.child_outcome: ChildSupervisionOutcome | None = None
        self.fallback_outcome: FallbackSupervisionOutcome | None = None
        self.child_exit_proved = False
        self.final_postcheck_proved = False
        self.fallback_gate_ports = "unproved"
        self.fallback_gate_topology = "unproved"
        self.wrapper_alive = True
        self.fallback_used = False
        self.no_live_after_release = False
        self._owned_close_attempted: set[str] = set()
        self.artifact_state = _initial_artifact_state()
        self.cleanup_state = _initial_cleanup_state()
        self.artifacts: dict[str, CreateNewFileReceipt] = {}
        self.reasons: set[str] = set()
        self.details: list[str] = []
        self.last_phase = "offline_precheck"

    def _now(self, label: str = "QPC observation") -> int:
        return _require_exact_nonnegative_int(self.clock.now_ns(), label)

    def _add_failure(self, reason: str, detail: Any, *, wrapper_death: bool = False) -> None:
        if reason not in attempt_contract.INVALIDATION_REASON_CODES:
            reason = "internal_error"
        self.reasons.add(reason)
        self.details.append(_sanitize_detail(detail))
        if wrapper_death or reason == "wrapper_death":
            self.wrapper_alive = False
            self.reasons.add("wrapper_death")

    def _mark_publication_failure(self, exc: PublicationError) -> None:
        path_to_state = {
            self.paths["bundle_verification"]: ("bundle_verification", "bundle_verification_sha256"),
            self.paths["capture_seal"]: ("capture_seal", "capture_seal_sha256"),
            self.paths["split_claim"]: ("split_claim", "split_claim_sha256"),
            self.paths["split_registry"]: ("split_registry", "split_registry_sha256"),
            self.paths["analysis_report"]: ("analysis_report", "analysis_report_sha256"),
            self.paths["wrapper_lifecycle"]: ("wrapper_lifecycle", "wrapper_lifecycle_sha256"),
        }
        pair = path_to_state.get(exc.path)
        if pair is not None:
            self.artifact_state[pair[0]] = "partial" if exc.state != "absent" else "absent"
            self.artifact_state[pair[1]] = None
        if exc.path == self.paths["attempt_complete"] and exc.state != "absent":
            self.artifact_state["attempt_complete"] = "partial"
            self.artifact_state["attempt_complete_partial_sha256"] = (
                exc.observed_sha256
            )

    def _accept_child_outcome(self, outcome: Any) -> ChildSupervisionOutcome:
        if not isinstance(outcome, ChildSupervisionOutcome):
            raise OrchestrationPhaseError(
                "child_failed", "child supervision result has the wrong type"
            )
        self.child_outcome = outcome
        for key, value in outcome.artifact_state_patch.items():
            if key not in self.artifact_state:
                raise OrchestrationPhaseError(
                    "artifact_mismatch", f"unknown child artifact-state key {key}"
                )
            self.artifact_state[key] = value
        for reason in outcome.reason_codes:
            self._add_failure(
                reason,
                f"powered-child supervision reported {reason}",
                wrapper_death=reason == "wrapper_death",
            )
        if outcome.wrapper_death:
            self._add_failure(
                "wrapper_death",
                "wrapper death was observed by child supervision",
                wrapper_death=True,
            )
        return outcome

    def _accept_fallback_outcome(self, outcome: Any) -> FallbackSupervisionOutcome:
        if not isinstance(outcome, FallbackSupervisionOutcome):
            raise OrchestrationPhaseError(
                "cleanup_unconfirmed", "fallback supervision result is invalid"
            )
        self.fallback_outcome = outcome
        for reason in outcome.reason_codes:
            self._add_failure(
                reason,
                f"fallback supervision reported {reason}",
                wrapper_death=reason == "wrapper_death",
            )
        return outcome

    def _heartbeat(self, phase: str, deadline: int) -> None:
        if not self.lease_acquired or self.release_proved or self.release_invoked:
            return
        observed = self._now(f"{phase} lease-heartbeat observation")
        cadence_failure: OrchestrationPhaseError | None = None
        if observed >= deadline:
            cadence_failure = OrchestrationPhaseError(
                "deadline_expired", f"{phase} heartbeat reached its phase deadline"
            )
        maximum_gap = _require_positive_int(
            self.freeze["deadline_durations_ns"]["lease_heartbeat_max_gap"],
            "lease heartbeat maximum gap",
        )
        if self.last_lease_heartbeat_monotonic_ns is not None:
            gap = observed - self.last_lease_heartbeat_monotonic_ns
            if gap < 0 or gap > maximum_gap:
                cadence_failure = OrchestrationPhaseError(
                    "lease_unverifiable",
                    f"{phase} lease heartbeat gap {gap} exceeds {maximum_gap}",
                )
        try:
            self.services.lease.heartbeat(
                self.lease, phase=phase, deadline_monotonic_ns=deadline
            )
        except BaseException as exc:
            raise OrchestrationPhaseError(
                (
                    "internal_error"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "lease_unverifiable"
                ),
                f"lease heartbeat failed in {phase} ({type(exc).__name__})",
            ) from exc
        self.last_lease_heartbeat_monotonic_ns = observed
        if cadence_failure is not None:
            raise cadence_failure

    def _phase(
        self,
        phase: str,
        operation: Callable[
            [Mapping[str, Any], HeartbeatPump],
            tuple[Any, Sequence[Mapping[str, Any]]],
        ],
        *,
        heartbeat_before: bool = True,
        heartbeat_after: bool = True,
    ) -> Any:
        if self.ledger is None or not self.ledger_usable:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"wrapper ledger unavailable before {phase}"
            )
        self.last_phase = phase
        try:
            deadline = self.ledger.start_phase(phase)
        except BaseException as exc:
            self.ledger_usable = False
            if isinstance(exc, PublicationError):
                self._mark_publication_failure(exc)
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            raise OrchestrationPhaseError(
                "internal_error" if interrupted else "artifact_mismatch",
                (
                    f"{phase} start was interrupted by {type(exc).__name__}"
                    if interrupted
                    else f"could not publish {phase} start: {exc}"
                ),
            ) from exc
        try:
            if heartbeat_before:
                self._heartbeat(phase, deadline["deadline_monotonic_ns"])
            heartbeat_period = _require_positive_int(
                self.freeze["deadline_durations_ns"]["lease_heartbeat_period"],
                "lease heartbeat period",
            )
            if heartbeat_period > self.freeze["deadline_durations_ns"][
                "lease_heartbeat_max_gap"
            ]:
                raise OrchestrationPhaseError(
                    "lease_unverifiable",
                    "heartbeat period exceeds the frozen maximum gap",
                )
            cooperative_heartbeat = HeartbeatPump(
                phase=phase,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                period_ns=heartbeat_period,
                _emit=lambda: self._heartbeat(
                    phase, deadline["deadline_monotonic_ns"]
                ),
            )
            result, artifacts = operation(deadline, cooperative_heartbeat)
            if heartbeat_after:
                self._heartbeat(phase, deadline["deadline_monotonic_ns"])
            self.ledger.end_phase(outcome="completed", artifacts=artifacts)
            return result
        except BaseException as exc:
            if isinstance(exc, OrchestrationPhaseError):
                failure = exc
            elif isinstance(exc, PublicationError):
                self._mark_publication_failure(exc)
                failure = OrchestrationPhaseError(
                    "artifact_mismatch", f"publication failed in {phase}: {exc}"
                )
            elif isinstance(exc, attempt_contract.PoweredAttemptContractError):
                failure = OrchestrationPhaseError(
                    "artifact_mismatch", f"record validation failed in {phase}: {exc}"
                )
            else:
                failure = OrchestrationPhaseError(
                    "internal_error",
                    f"unexpected {phase} failure ({type(exc).__name__})",
                )
            self._add_failure(
                failure.reason_code,
                failure.detail,
                wrapper_death=failure.wrapper_death,
            )
            try:
                if self.ledger.active_deadline is not None:
                    self.ledger.end_phase(
                        outcome="failed", reason_code=failure.reason_code
                    )
            except BaseException as ledger_exc:
                self.ledger_usable = False
                if isinstance(ledger_exc, PublicationError):
                    self._mark_publication_failure(ledger_exc)
                self._add_failure(
                    "artifact_mismatch",
                    "failed phase-end publication "
                    f"({type(ledger_exc).__name__})",
                )
            raise failure

    def _publish_attempt_json(
        self,
        *,
        name: str,
        path: str,
        value: Mapping[str, Any],
        deadline: int,
        validator: Callable[[Any], Any],
        recovery: bool = False,
    ) -> CreateNewFileReceipt:
        if self.workspace is None:
            raise OrchestrationPhaseError("internal_error", "attempt workspace is absent")
        publisher = (
            self.workspace.recovery_publisher_for(self.workspace.directory)
            if recovery
            else self.workspace.attempt_publisher
        )
        receipt = publisher.publish(
            path,
            value,
            deadline_monotonic_ns=deadline,
            validator=validator,
        )
        self.artifacts[name] = receipt
        return receipt

    def _revalidate_before_release(
        self,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None:
        refreshed = admit_offline(
            self.arguments,
            self.offline,
            deadline_monotonic_ns=deadline_monotonic_ns,
            monotonic_ns=self.clock.now_ns,
            heartbeat=heartbeat,
        )
        if (
            refreshed.live_freeze_sha256
            != self.foundation.offline.live_freeze_sha256
            or refreshed.git != self.foundation.offline.git
            or refreshed.implementation_inventory
            != self.foundation.offline.implementation_inventory
            or refreshed.environment_inventory
            != self.foundation.offline.environment_inventory
            or refreshed.import_inventory != self.foundation.offline.import_inventory
        ):
            raise OrchestrationPhaseError(
                "build_or_candidate_changed", "second offline admission drifted"
            )

    def _prepare_attempt(self) -> None:
        wrapper_process = self.services.process.current_wrapper_identity()
        attempt_contract.validate_process_identity(wrapper_process)
        self.retained_wrapper = self.services.process.retain_and_reprove(
            wrapper_process
        )
        self.handles = self.services.spawn.allocate_attempt_handles(wrapper_process)
        _validate_handle_set(self.handles)
        offline_completed = self._now("offline precheck completion")
        attempt_started = self._now("attempt publication start")
        self.material = build_attempt_material(
            admission=self.foundation.offline,
            wrapper_process=wrapper_process,
            host_boot_id_sha256=self.services.host.host_boot_id_sha256(),
            qpc_frequency_hz=self.foundation.qpc_frequency_hz,
            handles=self.handles,
            created_at_utc=self.services.host.utc_now(),
            wrapper_started_monotonic_ns=self.foundation.wrapper_started_monotonic_ns,
            offline_precheck_completed_monotonic_ns=offline_completed,
            attempt_publish_started_monotonic_ns=attempt_started,
            random_bytes=self.services.csprng.token_bytes,
        )
        self.workspace = AttemptWorkspace.consume(self.secure, self.freeze)
        self.ledger_directory = self.workspace.create_subdirectory(
            "wrapper_ledger_directory"
        )
        self.ledger = WrapperLedger(
            publisher=self.workspace.publisher_for(self.ledger_directory),
            lifecycle_publisher=self.workspace.attempt_publisher,
            ledger_directory=self.paths["wrapper_ledger_directory"],
            lifecycle_path=self.paths["wrapper_lifecycle"],
            timeline=WrapperTimeline(self.material.absolute_deadlines),
            clock=self.clock,
        )
        try:
            self.attempt_receipt = self.workspace.publish_attempt(self.material)
            self.ledger.record_attempt_publish_end(
                attempt_publish_deadline=self.material.attempt_publish_deadline,
                observed_monotonic_ns=self.attempt_receipt.completed_monotonic_ns,
            )
            self.ledger_usable = True
        except PublicationError as exc:
            self._mark_publication_failure(exc)
            self._add_failure("artifact_mismatch", f"attempt publication failed: {exc}")
            self.ledger_usable = False
            raise OrchestrationPhaseError(
                "artifact_mismatch", "attempt envelope publication failed"
            ) from exc

    def _run_live_sequence(self) -> None:
        assert self.material is not None and self.workspace is not None

        def acquire(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            owner_secret = self.material.capabilities.consume_secret("lease_owner")
            try:
                lease = self.services.lease.acquire(
                    owner_secret=owner_secret,
                    qpc_frequency_hz=self.foundation.qpc_frequency_hz,
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                )
            finally:
                self.material.capabilities.zeroize_role("lease_owner")
            self.lease = lease
            self.lease_acquired = True
            self.cleanup_state["lease"] = "retained"
            heartbeat()
            return lease, ()

        self._phase(
            "lease_acquire", acquire, heartbeat_before=False, heartbeat_after=False
        )

        def launch(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            # This latch distinguishes a lease-only failure from an ambiguous
            # launcher call that may already have contacted the simulator.
            self.launcher_invoked = True
            self.launch_result = self.services.launcher.launch_and_wait(
                freeze=self.freeze,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            self.cleanup_state.update(
                {
                    "scheduled_task": "unproved",
                    "simulator_topology": "unproved",
                    "simulator_responsive": "unproved",
                }
            )
            return self.launch_result, ()

        self._phase("launcher_return", launch)

        def topology_and_training(
            deadline: Mapping[str, Any],
            heartbeat: HeartbeatPump,
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            proof = self.services.topology.prove_topology(
                launch_result=self.launch_result,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            checked_proof = self.validators.process_proof(proof)
            if checked_proof.get("phase") != "prechild":
                raise OrchestrationPhaseError(
                    "topology_failed", "initial simulator proof is not prechild"
                )
            self._publish_attempt_json(
                name="process_prechild",
                path=self.paths["process_proof"],
                value=checked_proof,
                deadline=deadline["deadline_monotonic_ns"],
                validator=self.validators.process_proof,
            )
            heartbeat()
            attestation = self.services.training.attest_training(
                topology_proof=checked_proof,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            checked_attestation = self.validators.training_attestation(
                attestation, process_proof=checked_proof
            )
            self._publish_attempt_json(
                name="training_attestation",
                path=self.paths["training_attestation"],
                value=checked_attestation,
                deadline=deadline["deadline_monotonic_ns"],
                validator=lambda value: self.validators.training_attestation(
                    value, process_proof=checked_proof
                ),
            )
            self.topology_proof = checked_proof
            self.training_attestation = checked_attestation
            self.cleanup_state.update(
                {
                    "scheduled_task": "absent",
                    "simulator_topology": "unchanged",
                    "simulator_responsive": "yes",
                }
            )
            return checked_attestation, ()

        self._phase(
            "topology_and_training_attestation", topology_and_training
        )

        def prechild(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            assert self.topology_proof is not None
            _proved(
                self.services.process.prove_prechild_identity(
                    self.retained_wrapper,
                    topology_proof=self.topology_proof,
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                ),
                "prechild process identity",
            )
            heartbeat()
            _proved(
                self.services.ports.prove_prechild_free(
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                ),
                "prechild free ports",
            )
            self.cleanup_state["ports"] = "free"
            return True, ()

        self._phase("prechild_identity_and_ports", prechild)

        def spawn_child(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            assert self.handles is not None and self.training_attestation is not None
            child = self.services.spawn.spawn_powered_child_blocked(
                argv=self.material.child_argv,
                handles=self.handles,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            if not isinstance(child, BlockedProcess):
                raise OrchestrationPhaseError(
                    "child_spawn_failed", "spawn service did not return BlockedProcess"
                )
            self.child = child
            self.cleanup_state.update(
                {
                    "child_exit": "unproved",
                    "processes": "unproved",
                    "transport": "unproved",
                    "ports": "unproved",
                }
            )
            try:
                checked_authority = self.validators.process_authority(
                    child.authority,
                    attempt=self.material.envelope,
                    argv=self.material.child_argv,
                )
                self._publish_attempt_json(
                    name="child_authority",
                    path=self.paths["child_authority"],
                    value=checked_authority,
                    deadline=deadline["deadline_monotonic_ns"],
                    validator=lambda value: self.validators.process_authority(
                        value,
                        attempt=self.material.envelope,
                        argv=self.material.child_argv,
                    ),
                )
                self._revalidate_before_release(
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                )
                heartbeat()
                frame = self.material.capabilities.consume_frame("child")
                try:
                    self.services.spawn.release_child_capability(
                        child.handle,
                        frame=frame,
                        deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                        heartbeat=heartbeat,
                    )
                    self.child_capability_released = True
                finally:
                    frame[:] = b"\x00" * len(frame)
            except BaseException:
                try:
                    if not self.child_capability_released:
                        self.services.spawn.abort_blocked_process(
                            child.handle,
                            deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                            heartbeat=heartbeat,
                        )
                        self.child_abort_completed = True
                finally:
                    raise
            return child, ()

        try:
            self._phase("child_spawn", spawn_child)
        except OrchestrationPhaseError:
            # If release itself succeeded but its heartbeat/end evidence failed,
            # the child is live and safety supervision must continue in order.
            if not self.child_capability_released or not self.ledger_usable:
                raise

        def supervise_child(
            deadline: Mapping[str, Any],
            heartbeat: HeartbeatPump,
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            assert self.child is not None
            _proved(
                self.services.ports.prove_child_owners(
                    self.child.handle,
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                ),
                "child port ownership",
            )
            heartbeat()
            self.child_supervision_called = True
            outcome = self.services.supervision.supervise_powered_child(
                self.child.handle,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            outcome = self._accept_child_outcome(outcome)
            if outcome.wrapper_death:
                raise OrchestrationPhaseError(
                    "wrapper_death", "wrapper death was observed by child supervision",
                    wrapper_death=True,
                )
            if outcome.reason_codes or not outcome.collection_valid:
                reason = (
                    outcome.reason_codes[0]
                    if outcome.reason_codes
                    else "capture_incomplete"
                )
                raise OrchestrationPhaseError(
                    reason, "powered child marked collection invalid"
                )
            if not outcome.cleanup_proved:
                raise OrchestrationPhaseError(
                    "cleanup_unconfirmed", "child cleanup certificate is not proved"
                )
            return outcome, ()

        child_phase_failed = False
        try:
            self._phase("child_supervision", supervise_child)
        except OrchestrationPhaseError:
            child_phase_failed = True

        def child_exit(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            assert self.child is not None
            self.child_exit_attempted = True
            exit_proof = _proved(
                self.services.process.prove_child_tree_exit(
                    self.child.handle,
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                ),
                "child tree exit",
            )
            self.child_exit_proved = True
            self.cleanup_state["child_exit"] = "proved"
            if self.child_outcome is None or not self.child_outcome.cleanup_proved:
                port_gate = self.services.ports.prove_fallback_gate(
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                )
                heartbeat()
                topology_gate = self.services.topology.prove_unchanged(
                    launch_result=self.launch_result,
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                )
                self.fallback_gate_ports = (
                    "free" if port_gate.get("proved") is True else "unproved"
                )
                self.fallback_gate_topology = (
                    "unchanged"
                    if topology_gate.get("proved") is True
                    and topology_gate.get("topology", "unchanged") == "unchanged"
                    else "unproved"
                )
            return exit_proof, ()

        self._phase("child_exit_proof", child_exit)

        child_cleanup = (
            "valid"
            if self.child_outcome is not None and self.child_outcome.cleanup_proved
            else "invalid"
        )
        fallback_decision = decide_fallback(
            FallbackFacts(
                child_created=True,
                child_tree_exit="proved" if self.child_exit_proved else "unproved",
                child_cleanup=child_cleanup,
                ports=self.fallback_gate_ports,
                simulator_topology=self.fallback_gate_topology,
                cleanup_capability="available",
                fallback_already_attempted=False,
                wrapper_alive=self.wrapper_alive,
            )
        )
        if fallback_decision.status == "not_required":
            self.cleanup_state["fallback"] = "not_required"
        elif fallback_decision.spawn:
            self._run_fallback()
        else:
            self.cleanup_state["fallback"] = "not_eligible"
            self._add_failure(
                "cleanup_unconfirmed",
                f"fallback forbidden: {fallback_decision.reason}",
            )
        if child_phase_failed and not self.reasons:
            self._add_failure("child_failed", "child supervision failed")

    def _run_fallback(self) -> None:
        assert self.material is not None and self.handles is not None
        self.fallback_used = True
        self._add_failure(
            "cleanup_unconfirmed", "cleanup fallback use invalidates F00"
        )

        def spawn_fallback(
            deadline: Mapping[str, Any],
            heartbeat: HeartbeatPump,
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            fallback = self.services.spawn.spawn_cleanup_fallback_blocked(
                argv=self.material.cleanup_argv,
                handles=self.handles,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            if not isinstance(fallback, BlockedProcess):
                raise OrchestrationPhaseError(
                    "child_spawn_failed", "fallback spawn result has the wrong type"
                )
            self.fallback = fallback
            try:
                checked_authority = self.validators.process_authority(
                    fallback.authority,
                    attempt=self.material.envelope,
                    argv=self.material.cleanup_argv,
                )
                self._publish_attempt_json(
                    name="cleanup_authority",
                    path=self.paths["cleanup_authority"],
                    value=checked_authority,
                    deadline=deadline["deadline_monotonic_ns"],
                    validator=lambda value: self.validators.process_authority(
                        value,
                        attempt=self.material.envelope,
                        argv=self.material.cleanup_argv,
                    ),
                )
                self._revalidate_before_release(
                    deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                    heartbeat=heartbeat,
                )
                heartbeat()
                frame = self.material.capabilities.consume_frame("cleanup")
                try:
                    self.services.spawn.release_cleanup_capability(
                        fallback.handle,
                        frame=frame,
                        deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                        heartbeat=heartbeat,
                    )
                    self.fallback_capability_released = True
                finally:
                    frame[:] = b"\x00" * len(frame)
            except BaseException:
                try:
                    if not self.fallback_capability_released:
                        self.services.spawn.abort_blocked_process(
                            fallback.handle,
                            deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                            heartbeat=heartbeat,
                        )
                        self.fallback_abort_completed = True
                finally:
                    raise
            return fallback, ()

        try:
            self._phase("fallback_spawn", spawn_fallback)
        except OrchestrationPhaseError:
            if not self.fallback_capability_released or not self.ledger_usable:
                self.cleanup_state["fallback"] = "failed"
                return

        def supervise_fallback(
            deadline: Mapping[str, Any],
            heartbeat: HeartbeatPump,
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            assert self.fallback is not None
            self.fallback_supervision_called = True
            outcome = self.services.supervision.supervise_cleanup_fallback(
                self.fallback.handle,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            outcome = self._accept_fallback_outcome(outcome)
            if not outcome.cleanup_proved:
                reason = (
                    outcome.reason_codes[0]
                    if outcome.reason_codes
                    else "cleanup_unconfirmed"
                )
                raise OrchestrationPhaseError(
                    reason, "fallback cleanup did not prove safe state"
                )
            return outcome, ()

        try:
            self._phase("fallback_supervision", supervise_fallback)
            self.cleanup_state["fallback"] = "proved"
        except OrchestrationPhaseError:
            self.cleanup_state["fallback"] = (
                "proved"
                if self.fallback_outcome is not None
                and self.fallback_outcome.cleanup_proved
                else "failed"
            )

    def _record_safety_failure(
        self, label: str, exc: BaseException, *, reason: str
    ) -> None:
        if isinstance(exc, OrchestrationPhaseError):
            self._add_failure(
                exc.reason_code,
                f"{label}: {exc.detail}",
                wrapper_death=exc.wrapper_death,
            )
        else:
            self._add_failure(
                reason,
                f"{label} failed ({type(exc).__name__})",
            )

    def _unledgered_safety_window(
        self,
        phase: str,
        *,
        process: BlockedProcess | None = None,
    ) -> tuple[int, HeartbeatPump] | None:
        """Derive one fixed live-contact window without consulting the ledger."""

        assert self.material is not None
        try:
            started = self._now(f"unledgered {phase} start")
            derived = derive_phase_deadline(
                phase,
                started_monotonic_ns=started,
                parent_deadline_monotonic_ns=(
                    self.material.absolute_deadlines.live_contact_deadline_monotonic_ns
                ),
            )
            deadline = derived["deadline_monotonic_ns"]
            if process is not None:
                authority_deadlines = process.authority.get("absolute_deadlines")
                if isinstance(authority_deadlines, Mapping):
                    authority_exit = authority_deadlines.get("exit")
                    if (
                        isinstance(authority_exit, int)
                        and not isinstance(authority_exit, bool)
                    ):
                        deadline = min(deadline, authority_exit)
            if started >= deadline:
                raise OrchestrationPhaseError(
                    "deadline_expired",
                    f"unledgered {phase} has no remaining bounded window",
                )
            period = _require_positive_int(
                self.freeze["deadline_durations_ns"]["lease_heartbeat_period"],
                "lease heartbeat period",
            )
        except BaseException as exc:
            self._record_safety_failure(
                f"unledgered {phase} deadline derivation",
                exc,
                reason="deadline_expired",
            )
            return None

        def emit() -> None:
            # Lease evidence failure must invalidate the attempt, but it must
            # not suppress the already-authorized process-tree drain.
            try:
                self._heartbeat(phase, deadline)
            except BaseException as exc:
                self._record_safety_failure(
                    f"unledgered {phase} heartbeat",
                    exc,
                    reason="lease_unverifiable",
                )

        return deadline, HeartbeatPump(
            phase=phase,
            deadline_monotonic_ns=deadline,
            period_ns=period,
            _emit=emit,
        )

    def _confirm_process_tree_empty(
        self, name: str, process: BlockedProcess
    ) -> bool:
        close_name = f"{name}_process_handle"
        if close_name in self._owned_close_attempted:
            return True
        assert self.material is not None
        try:
            # The production boundary closes only when it already retains an
            # exact whole-job empty-tree proof.  Successful close is therefore
            # also the final pre-release confirmation that no descendant lives.
            self.services.spawn.close_process_handle(
                process.handle,
                deadline_monotonic_ns=(
                    self.material.absolute_deadlines.total_deadline_monotonic_ns
                ),
            )
        except BaseException as exc:
            self._record_safety_failure(
                f"{name} empty-tree confirmation", exc, reason="process_residue"
            )
            return False
        self._owned_close_attempted.add(close_name)
        return True

    def _unledgered_abort_blocked(
        self, name: str, process: BlockedProcess
    ) -> bool:
        completed_attribute = f"{name}_abort_completed"
        if getattr(self, completed_attribute):
            return True
        window = self._unledgered_safety_window("child_exit_proof")
        if window is None:
            return False
        deadline, heartbeat = window
        try:
            self.services.spawn.abort_blocked_process(
                process.handle,
                deadline_monotonic_ns=deadline,
                heartbeat=heartbeat,
            )
            setattr(self, completed_attribute, True)
            return True
        except BaseException as exc:
            self._record_safety_failure(
                f"{name} blocked-process abort", exc, reason="process_residue"
            )
            return False

    def _unledgered_supervise_child(self) -> None:
        assert self.child is not None
        window = self._unledgered_safety_window(
            "child_supervision", process=self.child
        )
        if window is None:
            return
        deadline, heartbeat = window
        try:
            _proved(
                self.services.ports.prove_child_owners(
                    self.child.handle,
                    deadline_monotonic_ns=deadline,
                    heartbeat=heartbeat,
                ),
                "child port ownership",
            )
        except BaseException as exc:
            self._record_safety_failure(
                "unledgered child-owner proof", exc, reason="port_in_use"
            )
        try:
            self.child_supervision_called = True
            outcome = self.services.supervision.supervise_powered_child(
                self.child.handle,
                deadline_monotonic_ns=deadline,
                heartbeat=heartbeat,
            )
            self._accept_child_outcome(outcome)
            if not outcome.collection_valid and not outcome.reason_codes:
                self._add_failure(
                    "capture_incomplete",
                    "unledgered child supervision marked collection invalid",
                )
            if not outcome.cleanup_proved and not outcome.reason_codes:
                self._add_failure(
                    "cleanup_unconfirmed",
                    "unledgered child supervision lacked cleanup proof",
                )
        except BaseException as exc:
            self._record_safety_failure(
                "unledgered child supervision", exc, reason="child_failed"
            )

    def _unledgered_prove_child_exit(self) -> None:
        assert self.child is not None
        window = self._unledgered_safety_window("child_exit_proof")
        if window is None:
            return
        deadline, heartbeat = window
        self.child_exit_attempted = True
        try:
            _proved(
                self.services.process.prove_child_tree_exit(
                    self.child.handle,
                    deadline_monotonic_ns=deadline,
                    heartbeat=heartbeat,
                ),
                "child tree exit",
            )
            self.child_exit_proved = True
            self.cleanup_state["child_exit"] = "proved"
        except BaseException as exc:
            self._record_safety_failure(
                "unledgered child tree drain", exc, reason="process_residue"
            )

    def _unledgered_prove_fallback_gate(self) -> None:
        if (
            self.fallback_gate_ports == "free"
            and self.fallback_gate_topology == "unchanged"
        ):
            return
        window = self._unledgered_safety_window("child_exit_proof")
        if window is None:
            return
        deadline, heartbeat = window
        try:
            ports = self.services.ports.prove_fallback_gate(
                deadline_monotonic_ns=deadline,
                heartbeat=heartbeat,
            )
            self.fallback_gate_ports = (
                "free"
                if isinstance(ports, Mapping) and ports.get("proved") is True
                else "unproved"
            )
        except BaseException as exc:
            self._record_safety_failure(
                "unledgered fallback port gate", exc, reason="port_in_use"
            )
        try:
            topology = self.services.topology.prove_unchanged(
                launch_result=self.launch_result,
                deadline_monotonic_ns=deadline,
                heartbeat=heartbeat,
            )
            self.fallback_gate_topology = (
                "unchanged"
                if isinstance(topology, Mapping)
                and topology.get("proved") is True
                and topology.get("topology", "unchanged") == "unchanged"
                else "unproved"
            )
        except BaseException as exc:
            self._record_safety_failure(
                "unledgered fallback topology gate",
                exc,
                reason="topology_failed",
            )

    def _unledgered_spawn_fallback(self) -> None:
        assert self.material is not None and self.handles is not None
        self.fallback_used = True
        self.cleanup_state["fallback"] = "failed"
        self._add_failure(
            "cleanup_unconfirmed", "cleanup fallback use invalidates F00"
        )
        window = self._unledgered_safety_window("fallback_spawn")
        if window is None:
            return
        deadline, heartbeat = window
        try:
            fallback = self.services.spawn.spawn_cleanup_fallback_blocked(
                argv=self.material.cleanup_argv,
                handles=self.handles,
                deadline_monotonic_ns=deadline,
                heartbeat=heartbeat,
            )
            if not isinstance(fallback, BlockedProcess):
                raise OrchestrationPhaseError(
                    "child_spawn_failed", "fallback spawn result has the wrong type"
                )
            self.fallback = fallback
            checked_authority = self.validators.process_authority(
                fallback.authority,
                attempt=self.material.envelope,
                argv=self.material.cleanup_argv,
            )
            self._publish_attempt_json(
                name="cleanup_authority",
                path=self.paths["cleanup_authority"],
                value=checked_authority,
                deadline=deadline,
                validator=lambda value: self.validators.process_authority(
                    value,
                    attempt=self.material.envelope,
                    argv=self.material.cleanup_argv,
                ),
            )
            self._revalidate_before_release(
                deadline_monotonic_ns=deadline,
                heartbeat=heartbeat,
            )
            heartbeat()
            frame = self.material.capabilities.consume_frame("cleanup")
            try:
                self.services.spawn.release_cleanup_capability(
                    fallback.handle,
                    frame=frame,
                    deadline_monotonic_ns=deadline,
                    heartbeat=heartbeat,
                )
                self.fallback_capability_released = True
            finally:
                frame[:] = b"\x00" * len(frame)
        except BaseException as exc:
            self._record_safety_failure(
                "unledgered fallback spawn", exc, reason="child_spawn_failed"
            )
            if self.fallback is not None and not self.fallback_capability_released:
                self._unledgered_abort_blocked("fallback", self.fallback)

    def _unledgered_supervise_fallback(self) -> None:
        assert self.fallback is not None
        window = self._unledgered_safety_window(
            "fallback_supervision", process=self.fallback
        )
        if window is None:
            return
        deadline, heartbeat = window
        try:
            self.fallback_supervision_called = True
            outcome = self.services.supervision.supervise_cleanup_fallback(
                self.fallback.handle,
                deadline_monotonic_ns=deadline,
                heartbeat=heartbeat,
            )
            outcome = self._accept_fallback_outcome(outcome)
            self.cleanup_state["fallback"] = (
                "proved" if outcome.cleanup_proved else "failed"
            )
            if not outcome.cleanup_proved and not outcome.reason_codes:
                self._add_failure(
                    "cleanup_unconfirmed",
                    "unledgered fallback cleanup did not prove safe state",
                )
        except BaseException as exc:
            self.cleanup_state["fallback"] = "failed"
            self._record_safety_failure(
                "unledgered fallback supervision",
                exc,
                reason="cleanup_unconfirmed",
            )

    def _run_process_safety_lane(self) -> bool:
        """Drain and confirm every spawned job before any lease release."""

        if self.child is not None:
            if self.child_capability_released:
                if not self.child_supervision_called:
                    self._unledgered_supervise_child()
                if not self.child_exit_attempted:
                    self._unledgered_prove_child_exit()
            else:
                self._unledgered_abort_blocked("child", self.child)

            child_empty = self._confirm_process_tree_empty("child", self.child)
            child_cleanup_proved = (
                self.child_outcome is not None
                and self.child_outcome.cleanup_proved
            )
            if self.child_capability_released and not child_cleanup_proved:
                if self.child_exit_proved and child_empty and self.wrapper_alive:
                    self._unledgered_prove_fallback_gate()
                    decision = decide_fallback(
                        FallbackFacts(
                            child_created=True,
                            child_tree_exit="proved",
                            child_cleanup="invalid",
                            ports=self.fallback_gate_ports,
                            simulator_topology=self.fallback_gate_topology,
                            cleanup_capability=(
                                "available"
                                if not self.fallback_used and self.fallback is None
                                else "consumed"
                            ),
                            fallback_already_attempted=self.fallback_used,
                            wrapper_alive=self.wrapper_alive,
                        )
                    )
                    if decision.spawn:
                        self._unledgered_spawn_fallback()
                    elif self.fallback is None:
                        self.cleanup_state["fallback"] = "not_eligible"
                        self._add_failure(
                            "cleanup_unconfirmed",
                            f"fallback forbidden: {decision.reason}",
                        )
                elif self.fallback is None:
                    self.cleanup_state["fallback"] = "not_eligible"
                    self._add_failure(
                        "cleanup_unconfirmed",
                        "fallback forbidden without natural child-tree exit proof",
                    )
        else:
            child_empty = True

        if self.fallback is not None:
            if self.fallback_capability_released:
                if not self.fallback_supervision_called:
                    self._unledgered_supervise_fallback()
                if self.fallback_outcome is None:
                    # A provider exception may precede its internal whole-job
                    # drain.  Abort is still bounded and the subsequent close
                    # remains the authoritative empty-tree confirmation.
                    self._unledgered_abort_blocked("fallback", self.fallback)
            else:
                self._unledgered_abort_blocked("fallback", self.fallback)
            fallback_empty = self._confirm_process_tree_empty(
                "fallback", self.fallback
            )
        else:
            fallback_empty = True

        if not child_empty or not fallback_empty:
            self.cleanup_state["processes"] = "residue"
            self._add_failure(
                "process_residue",
                "spawned process-tree emptiness was not proved before lease release",
            )
            return False
        return True

    def _postcheck_and_release(self) -> None:
        def postcheck(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            process = self.services.process.prove_final_process_state(
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            checked_process = self.validators.process_proof(process)
            if checked_process.get("phase") != "postchild":
                raise OrchestrationPhaseError(
                    "process_residue", "final simulator proof is not postchild"
                )
            heartbeat()
            ports = self.services.ports.prove_final_free(
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            heartbeat()
            topology = self.services.topology.prove_unchanged(
                launch_result=self.launch_result,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=heartbeat,
            )
            if (
                not isinstance(ports, Mapping)
                or ports.get("proved") is not True
                or ports.get("ports") != "free"
                or ports.get("transport") != "closed"
                or not isinstance(topology, Mapping)
                or topology.get("proved") is not True
                or topology.get("topology") != "unchanged"
                or topology.get("responsive") != "yes"
                or topology.get("scheduled_task") != "absent"
            ):
                raise OrchestrationPhaseError(
                    "cleanup_unconfirmed", "final process/port/topology proof failed"
                )
            self._publish_attempt_json(
                name="process_final",
                path=self.paths["process_final_proof"],
                value=checked_process,
                deadline=deadline["deadline_monotonic_ns"],
                validator=self.validators.process_proof,
            )
            self.cleanup_state.update(
                {
                    "processes": "exited",
                    "ports": "free",
                    "transport": "closed",
                    "simulator_topology": "unchanged",
                    "simulator_responsive": "yes",
                    "scheduled_task": "absent",
                }
            )
            self.final_postcheck_proved = True
            # The frozen ledger intentionally leaves this phase's artifact list
            # empty; the seal and terminal bind the independently published proof.
            return {
                "process": checked_process,
                "ports": ports,
                "topology": topology,
            }, ()

        # This gate is deliberately independent of wrapper-ledger usability.
        # A released capability creates process authority that must be drained
        # before the live lease can be relinquished, even on the recovery lane.
        if not self._run_process_safety_lane():
            return

        if not self.launcher_invoked:
            if self.child is not None or self.fallback is not None:
                self._add_failure(
                    "internal_error",
                    "spawned process exists although launcher was never invoked",
                )
                return
            # No launcher call means no simulator/process/transport authority
            # was created.  This exact local predicate permits best-effort
            # release of a successfully acquired lease without inventing a
            # postchild simulator proof.
            self.cleanup_state.update(
                {
                    "child_exit": "not_created",
                    "fallback": "not_eligible",
                    "ports": "not_opened",
                    "processes": "not_created",
                    "transport": "not_opened",
                    "scheduled_task": "not_created",
                    "simulator_topology": "not_launched",
                    "simulator_responsive": "not_launched",
                }
            )
            self.final_postcheck_proved = True

        if (
            self.launcher_invoked
            and self.ledger is not None
            and self.ledger_usable
        ):
            try:
                self._phase("postcheck_identity_process_ports", postcheck)
            except OrchestrationPhaseError:
                pass

        if self.launcher_invoked and not self.final_postcheck_proved:
            window = self._unledgered_safety_window(
                "postcheck_identity_process_ports"
            )
            if window is not None:
                deadline, heartbeat = window
                try:
                    postcheck(
                        {"deadline_monotonic_ns": deadline},
                        heartbeat,
                    )
                except BaseException as exc:
                    self._record_safety_failure(
                        "unledgered final process/port/topology gate",
                        exc,
                        reason="cleanup_unconfirmed",
                    )

        if not self.final_postcheck_proved:
            self.cleanup_state["processes"] = "unproved"
            self._add_failure(
                "cleanup_unconfirmed",
                "final process/port/topology state was not proved before lease release",
            )
            return

        if not self.lease_acquired:
            self.cleanup_state["lease"] = "not_acquired"
            return
        self._release_lease_once()

    def _release_lease_once(self) -> None:
        """Attempt the kernel transition once even when cadence evidence failed."""

        if not self.lease_acquired or self.release_invoked:
            return
        assert self.material is not None
        phase_started = False
        phase_reason: OrchestrationPhaseError | None = None
        deadline: Mapping[str, Any]
        if self.ledger is not None and self.ledger_usable:
            self.last_phase = "lease_release_and_verify"
            try:
                deadline = self.ledger.start_phase("lease_release_and_verify")
                phase_started = True
            except BaseException as exc:
                self.ledger_usable = False
                if isinstance(exc, PublicationError):
                    self._mark_publication_failure(exc)
                phase_reason = OrchestrationPhaseError(
                    (
                        "internal_error"
                        if isinstance(exc, (KeyboardInterrupt, SystemExit))
                        else "artifact_mismatch"
                    ),
                    "release phase-start publication failed "
                    f"({type(exc).__name__})",
                )
                self._add_failure(phase_reason.reason_code, phase_reason.detail)
        if not phase_started:
            try:
                started = self._now("unledgered release start")
                deadline = derive_phase_deadline(
                    "lease_release_and_verify",
                    started_monotonic_ns=started,
                    parent_deadline_monotonic_ns=(
                        self.material.absolute_deadlines.live_contact_deadline_monotonic_ns
                    ),
                )
            except BaseException as exc:
                self._record_safety_failure(
                    "unledgered lease-release deadline derivation",
                    exc,
                    reason="deadline_expired",
                )
                self.cleanup_state["lease"] = "retained"
                return

        try:
            self._heartbeat(
                "lease_release_and_verify", deadline["deadline_monotonic_ns"]
            )
        except OrchestrationPhaseError as exc:
            phase_reason = phase_reason or exc
            self._add_failure(exc.reason_code, exc.detail)

        def emit_release_heartbeat() -> None:
            nonlocal phase_reason
            try:
                self._heartbeat(
                    "lease_release_and_verify",
                    deadline["deadline_monotonic_ns"],
                )
            except OrchestrationPhaseError as exc:
                phase_reason = phase_reason or exc
                self._add_failure(exc.reason_code, exc.detail)

        release_heartbeat = HeartbeatPump(
            phase="lease_release_and_verify",
            deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
            period_ns=self.freeze["deadline_durations_ns"][
                "lease_heartbeat_period"
            ],
            _emit=emit_release_heartbeat,
        )

        # Invocation is single-use and may have released the kernel object even
        # when the boundary raises before returning its proof.  Latch offline
        # before crossing it, prohibit retries, and perform no later live call.
        self.release_invoked = True
        self.no_live_after_release = True
        lease_ref: Mapping[str, Any] | None = None
        try:
            outcome = self.services.lease.release_and_verify(
                self.lease,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                heartbeat=release_heartbeat,
            )
            if not isinstance(outcome, LeaseReleaseOutcome):
                raise OrchestrationPhaseError(
                    "lease_release_unconfirmed",
                    "release service returned an invalid outcome type",
                )
            if outcome.kernel_released:
                released = _require_exact_nonnegative_int(
                    outcome.released_monotonic_ns, "kernel lease release"
                )
                if released >= deadline["deadline_monotonic_ns"]:
                    raise OrchestrationPhaseError(
                        "lease_release_unconfirmed",
                        "kernel lease release reached its absolute deadline",
                    )
                self.kernel_released = True
                if self.ledger is not None:
                    self.ledger.timeline.note_lease_release(released)
                if outcome.final_index is None:
                    raise OrchestrationPhaseError(
                        "lease_release_unconfirmed",
                        "kernel release lacked a final index",
                    )
                checked = self.validators.lease_final(outcome.final_index)
                receipt = self._publish_attempt_json(
                    name="lease_final",
                    path=self.paths["lease_final"],
                    value=checked,
                    deadline=deadline["deadline_monotonic_ns"],
                    validator=self.validators.lease_final,
                    recovery=not self.ledger_usable,
                )
                self.release_proved = True
                self.cleanup_state["lease"] = "released"
                lease_ref = artifact_ref("lease_final", receipt)
            else:
                if outcome.released_monotonic_ns is not None or outcome.final_index is not None:
                    raise OrchestrationPhaseError(
                        "lease_release_unconfirmed",
                        "unreleased outcome carried contradictory release proof",
                    )
                raise OrchestrationPhaseError(
                    "lease_release_unconfirmed", "kernel release was not proved"
                )
        except BaseException as exc:
            if isinstance(exc, PublicationError):
                self._mark_publication_failure(exc)
            if isinstance(exc, OrchestrationPhaseError):
                failure = exc
            else:
                failure = OrchestrationPhaseError(
                    (
                        "internal_error"
                        if isinstance(exc, (KeyboardInterrupt, SystemExit))
                        else "lease_release_unconfirmed"
                    ),
                    f"release boundary failed ({type(exc).__name__})",
                )
            phase_reason = phase_reason or failure
            self._add_failure(failure.reason_code, failure.detail)
            if not self.release_proved:
                self.cleanup_state["lease"] = "unproved"

        if phase_started and self.ledger is not None and self.ledger_usable:
            try:
                if phase_reason is None and self.release_proved and lease_ref is not None:
                    self.ledger.end_phase(
                        outcome="completed", artifacts=(lease_ref,)
                    )
                else:
                    self.ledger.end_phase(
                        outcome="failed",
                        reason_code=(
                            phase_reason.reason_code
                            if phase_reason is not None
                            else "lease_release_unconfirmed"
                        ),
                    )
            except BaseException as exc:
                self.ledger_usable = False
                if isinstance(exc, PublicationError):
                    self._mark_publication_failure(exc)
                self._add_failure(
                    "artifact_mismatch", f"release phase-end publication failed: {exc}"
                )

    def _success_cleanup_ready(self) -> bool:
        return self.cleanup_state == {
            "child_exit": "proved",
            "fallback": "not_required",
            "ports": "free",
            "lease": "released",
            "processes": "exited",
            "transport": "closed",
            "scheduled_task": "absent",
            "simulator_topology": "unchanged",
            "simulator_responsive": "yes",
        }

    def _finalize_owned_resources(self) -> bool:
        """One-shot local-handle closure under the original total deadline."""

        if self.material is not None:
            self.material.capabilities.zeroize_all()
            deadline = self.material.absolute_deadlines.total_deadline_monotonic_ns
        else:
            deadline = (
                self.foundation.wrapper_started_monotonic_ns
                + self.freeze["deadline_durations_ns"]["wrapper_total"]
            )
        success = True

        def close_once(name: str, callback: Callable[[], None]) -> None:
            nonlocal success
            if name in self._owned_close_attempted:
                return
            self._owned_close_attempted.add(name)
            try:
                callback()
            except Exception as exc:
                success = False
                self._add_failure(
                    "internal_error", f"owned resource closure failed for {name}: {exc}"
                )

        if self.fallback is not None:
            close_once(
                "fallback_process_handle",
                lambda: self.services.spawn.close_process_handle(
                    self.fallback.handle, deadline_monotonic_ns=deadline
                ),
            )
        if self.child is not None:
            close_once(
                "child_process_handle",
                lambda: self.services.spawn.close_process_handle(
                    self.child.handle, deadline_monotonic_ns=deadline
                ),
            )
        if self.handles is not None:
            close_once(
                "attempt_handles",
                lambda: self.services.spawn.close_attempt_handles(
                    self.handles, deadline_monotonic_ns=deadline
                ),
            )
        if self.retained_wrapper is not None:
            close_once(
                "retained_wrapper",
                lambda: self.services.process.close_retained_wrapper(
                    self.retained_wrapper, deadline_monotonic_ns=deadline
                ),
            )
        return success

    def _run_postrelease_success(self) -> OrchestrationResult:
        assert self.workspace is not None and self.ledger is not None
        if not self.release_proved or not self.no_live_after_release:
            raise OrchestrationPhaseError(
                "lease_release_unconfirmed", "offline work requires proved release"
            )

        def bundle(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            value = self.services.postrelease.verify_bundle(
                phase_deadline=deadline
            )
            checked = self.validators.bundle_verification(value)
            receipt = self._publish_attempt_json(
                name="bundle_verification",
                path=self.paths["bundle_verification"],
                value=checked,
                deadline=deadline["deadline_monotonic_ns"],
                validator=self.validators.bundle_verification,
            )
            self.artifact_state.update(
                {
                    "bundle_verification": "valid",
                    "bundle_verification_sha256": receipt.sha256,
                }
            )
            return checked, (artifact_ref("bundle_verification", receipt),)

        bundle_value = self._phase("bundle_verify", bundle)

        def seal(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            value = self.services.postrelease.build_capture_seal(
                phase_deadline=deadline
            )
            checked = self.validators.capture_seal(value)
            receipt = self._publish_attempt_json(
                name="capture_seal",
                path=self.paths["capture_seal"],
                value=checked,
                deadline=deadline["deadline_monotonic_ns"],
                validator=self.validators.capture_seal,
            )
            self.artifact_state.update(
                {"capture_seal": "valid", "capture_seal_sha256": receipt.sha256}
            )
            return checked, (artifact_ref("capture_seal", receipt),)

        seal_value = self._phase("capture_seal", seal)

        def analyze(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            value = self.services.postrelease.analyze_capture(
                phase_deadline=deadline
            )
            return value, ()

        analysis_value = self._phase("analysis", analyze)

        def split(
            deadline: Mapping[str, Any], heartbeat: HeartbeatPump
        ) -> tuple[Any, Sequence[Mapping[str, Any]]]:
            publications = self.services.postrelease.publish_split(
                analysis=analysis_value,
                phase_deadline=deadline,
            )
            if not isinstance(publications, SplitPublications):
                raise OrchestrationPhaseError(
                    "artifact_mismatch", "split provider returned the wrong type"
                )
            claim = self.validators.split_claim(publications.claim)
            claim_receipt = self._publish_attempt_json(
                name="split_claim",
                path=self.paths["split_claim"],
                value=claim,
                deadline=deadline["deadline_monotonic_ns"],
                validator=self.validators.split_claim,
            )
            self.artifact_state.update(
                {
                    "split_claim": "valid",
                    "split_claim_sha256": claim_receipt.sha256,
                }
            )
            registry_directory = self.workspace.open_split_registry_directory()
            registry_publisher = CreateNewJsonPublisher(
                self.secure,
                registry_directory,
                latch=self.workspace.latch,
            )
            registry = self.validators.split_registry(
                publications.registry, split_claim=claim
            )
            registry_receipt = registry_publisher.publish(
                self.paths["split_registry"],
                registry,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                validator=lambda value: self.validators.split_registry(
                    value, split_claim=claim
                ),
            )
            self.artifacts["split_registry"] = registry_receipt
            self.artifact_state.update(
                {
                    "split_registry": "valid",
                    "split_registry_sha256": registry_receipt.sha256,
                }
            )
            report = self.validators.analysis_report(publications.report)
            report_receipt = self._publish_attempt_json(
                name="analysis_report",
                path=self.paths["analysis_report"],
                value=report,
                deadline=deadline["deadline_monotonic_ns"],
                validator=self.validators.analysis_report,
            )
            self.artifact_state.update(
                {
                    "analysis_report": "valid",
                    "analysis_report_sha256": report_receipt.sha256,
                }
            )
            refs = (
                artifact_ref("analysis_report", report_receipt),
                artifact_ref("split_claim", claim_receipt),
                artifact_ref("split_registry", registry_receipt),
            )
            return publications, refs

        split_values = self._phase("split_publish", split)

        terminal_deadline = self.ledger.start_phase("terminal_ready")
        self.ledger.end_phase(outcome="completed")
        try:
            self.lifecycle_receipt = self.ledger.finalize_lifecycle()
            assert self.ledger.lifecycle_value is not None
            self.artifact_state.update(
                {
                    "wrapper_lifecycle": "valid",
                    "wrapper_lifecycle_sha256": self.lifecycle_receipt.sha256,
                }
            )
            completed = self._now("complete terminal preparation")
            # `terminal_ready` is the append-only ledger phase.  The terminal
            # schema names the same frozen five-second publication window
            # `terminal_publish`; normalize only the copy passed to the
            # post-release builder and leave the ledger rows unchanged.
            terminal_publication_timing = {
                **terminal_deadline,
                "phase": "terminal_publish",
            }
            terminal_context = self._record_context(
                phase="terminal_ready",
                publication_timing={
                    **terminal_publication_timing,
                    "prepared_monotonic_ns": completed,
                },
                bundle=bundle_value,
                seal=seal_value,
                split=split_values,
                completed_monotonic_ns=completed,
                lifecycle=self.ledger.lifecycle_value,
            )
            complete = self.services.postrelease.build_complete_terminal(
                context=terminal_context
            )
            checked = self.validators.complete_terminal(
                complete, lifecycle=self.ledger.lifecycle_value
            )
            self.terminal_receipt = self._publish_attempt_json(
                name="attempt_complete",
                path=self.paths["attempt_complete"],
                value=checked,
                deadline=terminal_deadline["deadline_monotonic_ns"],
                validator=lambda value: self.validators.complete_terminal(
                    value, lifecycle=self.ledger.lifecycle_value or {}
                ),
            )
        except Exception as exc:
            if isinstance(exc, PublicationError):
                self._mark_publication_failure(exc)
            self._add_failure(
                "terminal_write_failed", f"complete terminal publication failed: {exc}"
            )
            return self._recover(after_lifecycle=True)
        return OrchestrationResult(
            status="complete",
            attempt_consumed=True,
            fallback_used=False,
            reason_codes=(),
            terminal_receipt=self.terminal_receipt,
            poison_receipt=None,
            lifecycle_receipt=self.lifecycle_receipt,
            ledger_events=tuple(self.ledger.events),
            live_kernel_released=self.kernel_released,
            live_release_proved=True,
            no_live_after_release=True,
        )

    def _record_context(self, *, phase: str, **extra: Any) -> dict[str, Any]:
        return {
            "phase": phase,
            "utc": self.services.host.utc_now(),
            "admission": self.foundation.offline,
            "material": self.material,
            "attempt_receipt": self.attempt_receipt,
            "artifacts": dict(self.artifacts),
            "artifact_state": copy_json(self.artifact_state),
            "cleanup_state": copy_json(self.cleanup_state),
            "reason_codes": tuple(
                sorted(self.reasons, key=lambda item: item.encode("utf-8"))
            ),
            "reason_detail": "; ".join(self.details)[:4096],
            "wrapper_alive": self.wrapper_alive,
            "child_process": self.child.identity if self.child is not None else None,
            "cleanup_process": self.fallback.identity if self.fallback is not None else None,
            "fallback_used": self.fallback_used,
            "lease_acquired": self.lease_acquired,
            "lease_release_proved": self.release_proved,
            **extra,
        }

    def _publish_poison_unledgered(
        self, deadline: Mapping[str, Any]
    ) -> CreateNewFileReceipt | None:
        assert self.workspace is not None
        prepared = self._now("poison preparation")
        context = self._record_context(
            phase=self.last_phase,
            publication_timing={
                **deadline,
                "prepared_monotonic_ns": prepared,
            },
            created_monotonic_ns=prepared,
        )
        try:
            poison = self.services.postrelease.build_live_poison(context=context)
            root_publisher = self.workspace.recovery_publisher_for(
                self.workspace.root_directory
            )
            receipt = root_publisher.publish(
                self.paths["live_poison"],
                poison,
                deadline_monotonic_ns=deadline["deadline_monotonic_ns"],
                validator=self.validators.live_poison,
            )
            self.artifacts["live_poison"] = receipt
            self.poison_receipt = receipt
            return receipt
        except Exception as exc:
            if isinstance(exc, PublicationError):
                self._mark_publication_failure(exc)
            self._add_failure(
                "terminal_write_failed", f"poison publication failed: {exc}"
            )
            return None

    def _recover(self, *, after_lifecycle: bool = False) -> OrchestrationResult:
        if self.workspace is None:
            raise AttemptGateError(
                "attempt directory was not proved; no automatic recovery publication is safe"
            )
        if not self.reasons:
            self._add_failure("internal_error", "invalid branch lacked a reason")
        if self.ledger is not None and self.ledger_usable and not after_lifecycle:
            assert self.ledger_directory is not None
            self.ledger.enter_recovery_lane(
                event_publisher=self.workspace.recovery_publisher_for(
                    self.ledger_directory
                ),
                lifecycle_publisher=self.workspace.recovery_publisher_for(
                    self.workspace.directory
                ),
            )
        prospective = copy_json(self.artifact_state)
        if self.ledger is not None and self.ledger_usable and not after_lifecycle:
            prospective["wrapper_lifecycle"] = "valid"
            prospective["wrapper_lifecycle_sha256"] = "0" * 64
        ordered_reasons = tuple(
            sorted(self.reasons, key=lambda item: item.encode("utf-8"))
        )
        envelope_state = "valid" if self.attempt_receipt is not None else "partial"
        try:
            poison_required = derive_poison_required(
                cleanup_state=self.cleanup_state,
                artifact_state=prospective,
                reason_codes=ordered_reasons,
                attempt_envelope_state=envelope_state,
            )
        except Exception:
            poison_required = True
            self.cleanup_state = {
                **self.cleanup_state,
                "ports": "unproved",
                "lease": "unproved" if self.lease_acquired else "not_acquired",
            }

        if poison_required:
            poison_deadline: Mapping[str, Any]
            if self.ledger is not None and self.ledger_usable and not after_lifecycle:
                try:
                    poison_deadline = self.ledger.start_phase("poison_publish")
                    receipt = self._publish_poison_unledgered(poison_deadline)
                    if receipt is not None:
                        self.ledger.end_phase(
                            outcome="completed",
                            artifacts=[artifact_ref("live_poison", receipt)],
                        )
                    else:
                        self.ledger.end_phase(
                            outcome="failed", reason_code="terminal_write_failed"
                        )
                except Exception as exc:
                    self.ledger_usable = False
                    self._add_failure(
                        "artifact_mismatch", f"poison ledger path failed: {exc}"
                    )
            else:
                start = self._now("unledgered poison start")
                poison_deadline = derive_phase_deadline(
                    "poison_publish",
                    started_monotonic_ns=start,
                    parent_deadline_monotonic_ns=(
                        self.ledger.timeline.terminal_parent_deadline
                        if self.ledger is not None
                        else self.material.absolute_deadlines.total_deadline_monotonic_ns
                    ),
                )
                self._publish_poison_unledgered(poison_deadline)

        if self.ledger is not None and self.ledger_usable and not after_lifecycle:
            try:
                self.ledger.start_phase("invalid_ready")
                self.ledger.end_phase(outcome="completed")
                self.lifecycle_receipt = self.ledger.finalize_lifecycle()
                self.artifact_state.update(
                    {
                        "wrapper_lifecycle": "valid",
                        "wrapper_lifecycle_sha256": self.lifecycle_receipt.sha256,
                    }
                )
            except Exception as exc:
                if isinstance(exc, PublicationError):
                    self._mark_publication_failure(exc)
                self.ledger_usable = False
                self._add_failure(
                    "artifact_mismatch", f"invalid lifecycle finalization failed: {exc}"
                )
                # A lifecycle failure can turn an otherwise safe invalid record
                # into a poison-required state.  Re-evaluate using the actual
                # post-failure artifact tuple, and publish only on the distinct
                # root recovery target if the earlier branch did not need it.
                if not poison_required:
                    try:
                        poison_required = derive_poison_required(
                            cleanup_state=self.cleanup_state,
                            artifact_state=self.artifact_state,
                            reason_codes=tuple(
                                sorted(
                                    self.reasons,
                                    key=lambda item: item.encode("utf-8"),
                                )
                            ),
                            attempt_envelope_state=envelope_state,
                        )
                    except Exception:
                        poison_required = True
                    if poison_required and self.poison_receipt is None:
                        start = self._now("late unledgered poison start")
                        poison_deadline = derive_phase_deadline(
                            "poison_publish",
                            started_monotonic_ns=start,
                            parent_deadline_monotonic_ns=(
                                self.ledger.timeline.terminal_parent_deadline
                                if self.ledger is not None
                                else self.material.absolute_deadlines.total_deadline_monotonic_ns
                            ),
                        )
                        self._publish_poison_unledgered(poison_deadline)

        invalid_start = self._now("invalid terminal start")
        terminal_parent = (
            self.ledger.timeline.terminal_parent_deadline
            if self.ledger is not None
            else self.material.absolute_deadlines.total_deadline_monotonic_ns
        )
        invalid_deadline = derive_phase_deadline(
            "invalid_terminal_publish",
            started_monotonic_ns=invalid_start,
            parent_deadline_monotonic_ns=terminal_parent,
        )
        invalidated = self._now("invalid terminal preparation")
        invalid_context = self._record_context(
            phase=self.last_phase,
            publication_timing={
                **invalid_deadline,
                "prepared_monotonic_ns": invalidated,
            },
            invalidated_monotonic_ns=invalidated,
            poison_required=poison_required,
            poison_receipt=self.poison_receipt,
            lifecycle_receipt=self.lifecycle_receipt,
        )
        try:
            invalid = self.services.postrelease.build_invalid_terminal(
                context=invalid_context
            )
            self.terminal_receipt = self._publish_attempt_json(
                name="attempt_invalid",
                path=self.paths["attempt_invalid"],
                value=invalid,
                deadline=invalid_deadline["deadline_monotonic_ns"],
                validator=self.validators.invalid_terminal,
                recovery=True,
            )
            status = "invalid"
        except Exception as exc:
            if isinstance(exc, PublicationError):
                self._mark_publication_failure(exc)
            self._add_failure(
                "terminal_write_failed", f"invalid terminal publication failed: {exc}"
            )
            status = "invalid_unproved"
            self.terminal_receipt = None
        return OrchestrationResult(
            status=status,
            attempt_consumed=True,
            fallback_used=self.fallback_used,
            reason_codes=tuple(
                sorted(self.reasons, key=lambda item: item.encode("utf-8"))
            ),
            terminal_receipt=self.terminal_receipt,
            poison_receipt=self.poison_receipt,
            lifecycle_receipt=self.lifecycle_receipt,
            ledger_events=tuple(self.ledger.events) if self.ledger is not None else (),
            live_kernel_released=self.kernel_released,
            live_release_proved=self.release_proved,
            no_live_after_release=self.no_live_after_release,
        )

    def execute(self) -> OrchestrationResult:
        try:
            try:
                self._prepare_attempt()
            except Exception as exc:
                if not isinstance(exc, OrchestrationPhaseError):
                    if isinstance(exc, PublicationError):
                        self._mark_publication_failure(exc)
                    self._add_failure(
                        "artifact_mismatch", f"attempt preparation failed: {exc}"
                    )
                self._finalize_owned_resources()
                if self.workspace is not None:
                    return self._recover()
                raise
            live_failed = False
            try:
                self._run_live_sequence()
            except OrchestrationPhaseError:
                live_failed = True
            self._postcheck_and_release()
            self._finalize_owned_resources()
            if not self.release_proved and self.lease_acquired:
                self._add_failure(
                    "lease_release_unconfirmed", "live lease release was not proved"
                )
            if self.fallback_used:
                live_failed = True
            if self.child_outcome is None or not self.child_outcome.collection_valid:
                live_failed = True
            if self.child_outcome is None or not self.child_outcome.cleanup_proved:
                if not (
                    self.fallback_outcome is not None
                    and self.fallback_outcome.cleanup_proved
                ):
                    live_failed = True
            if not self._success_cleanup_ready():
                live_failed = True
                if not self.reasons:
                    self._add_failure(
                        "cleanup_unconfirmed", "terminal cleanup tuple is not complete-safe"
                    )
            if live_failed or self.reasons:
                return self._recover()
            try:
                return self._run_postrelease_success()
            except OrchestrationPhaseError as exc:
                self._add_failure(
                    exc.reason_code, exc.detail, wrapper_death=exc.wrapper_death
                )
                return self._recover()
        finally:
            if self.material is not None:
                self.material.capabilities.zeroize_all()
            self._finalize_owned_resources()


def copy_json(value: Any) -> Any:
    """Small defensive JSON copy without introducing another import graph."""

    return attempt_contract.defensive_copy(value)


class _WindowsProductionOfflineAdmissionBase:
    """Concrete read-only L0 identity boundary.

    Imports used only by production admission remain lazy so importing this
    module stays inert and the frozen import audit controls the exact graph.
    The service never creates the attempt directory and never contacts a
    simulator, socket, mutex, or fixed port.
    """

    _MAX_JSON_BYTES = 64 * 1024 * 1024
    _GIT_TIMEOUT_SECONDS = 15.0

    @staticmethod
    def _modules() -> tuple[Any, Any, Any, Any]:
        import importlib
        import os
        import subprocess
        from pathlib import Path

        return importlib, os, subprocess, Path

    @classmethod
    def _require_windows(cls) -> tuple[Any, Any, Any, Any]:
        modules = cls._modules()
        if modules[1].name != "nt":
            raise OfflineAdmissionError("powered production admission requires Windows")
        return modules

    @classmethod
    def _run_git(
        cls,
        worktree: str,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> Any:
        _importlib, _os, subprocess, _Path = cls._require_windows()
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=worktree,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=cls._GIT_TIMEOUT_SECONDS,
                check=check,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OfflineAdmissionError(
                "bounded Git identity command failed"
            ) from exc

    @classmethod
    def _canonical_existing_path(
        cls, path: str, *, directory: bool
    ) -> tuple[str, Any]:
        _importlib, os, _subprocess, Path = cls._require_windows()
        if type(path) is not str or not ntpath.isabs(path):
            raise OfflineAdmissionError("identity path must be absolute")
        if ntpath.normpath(path) != path or os.path.abspath(path) != path:
            raise OfflineAdmissionError("identity path must be lexically canonical")
        target = Path(path)
        try:
            info = target.lstat()
        except OSError as exc:
            raise OfflineAdmissionError("identity path is missing") from exc
        if target.is_symlink() or bool(
            getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise OfflineAdmissionError("identity path is a reparse point")
        if directory != target.is_dir():
            raise OfflineAdmissionError("identity path has the wrong kind")
        if not directory and not target.is_file():
            raise OfflineAdmissionError("identity path is not a regular file")
        probe = Path(target.anchor)
        for component in target.parts[1:]:
            probe = probe / component
            try:
                ancestor = probe.lstat()
            except OSError as exc:
                raise OfflineAdmissionError("identity ancestry changed") from exc
            if probe.is_symlink() or bool(
                getattr(ancestor, "st_file_attributes", 0) & 0x400
            ):
                raise OfflineAdmissionError("identity ancestry traverses a reparse point")
        resolved = os.path.realpath(path)
        if os.path.normcase(resolved) != os.path.normcase(path):
            raise OfflineAdmissionError("identity path resolves through an alias")
        return path, info

    @classmethod
    def _path_proof(cls, path: str, *, directory: bool) -> PathProof:
        canonical, info = cls._canonical_existing_path(path, directory=directory)
        return PathProof(
            path=canonical,
            final_path=canonical,
            kind="directory" if directory else "file",
            volume_id=f"stdev-{int(info.st_dev):x}",
            exists=True,
            non_reparse=True,
            ancestors_non_reparse=True,
            retained_handle=True,
        )

    def read_stable_json(self, path: str) -> StableJsonProof:
        from scripts import aigp_vq2_powered_runtime as powered_runtime

        proof = self._path_proof(path, directory=False)
        try:
            identity_before = powered_runtime.stable_file_identity(
                path, max_bytes=self._MAX_JSON_BYTES
            )
            with open(path, "rb") as stream:
                raw = stream.read(self._MAX_JSON_BYTES + 1)
            identity_after = powered_runtime.stable_file_identity(
                path, max_bytes=self._MAX_JSON_BYTES
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise OfflineAdmissionError("stable JSON read failed") from exc
        if (
            len(raw) > self._MAX_JSON_BYTES
            or identity_before != identity_after
            or len(raw) != identity_before.size_bytes
            or hashlib.sha256(raw).hexdigest() != identity_before.sha256
        ):
            raise OfflineAdmissionError("stable JSON identity changed during read")
        try:
            value = attempt_contract.parse_canonical_json_bytes(raw, file_form=True)
        except attempt_contract.PoweredAttemptContractError as exc:
            raise OfflineAdmissionError("JSON file is not canonical") from exc
        return StableJsonProof(
            identity=FileIdentityProof(
                path=proof,
                size_bytes=identity_before.size_bytes,
                sha256=identity_before.sha256,
            ),
            raw_bytes=raw,
            value=value,
        )

    def observe_file_identity(
        self, path: str, *, hash_kind: str
    ) -> FileIdentityProof:
        from scripts import aigp_vq2_powered_runtime as powered_runtime

        proof = self._path_proof(path, directory=False)
        if hash_kind == "file_bytes":
            try:
                identity = powered_runtime.stable_file_identity(path)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise OfflineAdmissionError("stable file identity failed") from exc
            digest = identity.sha256
            size = identity.size_bytes
        elif hash_kind == "canonical_object":
            document = self.read_stable_json(path)
            digest = attempt_contract.canonical_object_sha256(document.value)
            size = document.identity.size_bytes
        else:
            raise OfflineAdmissionError("unsupported identity hash kind")
        return FileIdentityProof(
            path=proof,
            size_bytes=size,
            sha256=digest,
            hash_kind=hash_kind,
            stable_before_after=True,
        )

    def current_working_directory(self) -> PathProof:
        _importlib, os, _subprocess, _Path = self._require_windows()
        return self._path_proof(os.getcwd(), directory=True)

    def module_origin(self, module_name: str) -> PathProof:
        if type(module_name) is not str or not module_name:
            raise OfflineAdmissionError("module name must be nonempty")
        module = sys.modules.get(module_name)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if type(origin) is not str:
            raise OfflineAdmissionError("module has no file-backed origin")
        _importlib, os, _subprocess, _Path = self._require_windows()
        return self._path_proof(os.path.abspath(origin), directory=False)

    def git_worktree(self, path: str) -> GitWorktreeProof:
        _importlib, os, _subprocess, _Path = self._require_windows()
        self._canonical_existing_path(path, directory=True)
        head = self._run_git(path, ["rev-parse", "HEAD"]).stdout.decode(
            "ascii", errors="strict"
        ).strip()
        tree = self._run_git(path, ["rev-parse", "HEAD^{tree}"]).stdout.decode(
            "ascii", errors="strict"
        ).strip()
        symbolic = self._run_git(
            path, ["symbolic-ref", "-q", "HEAD"], check=False
        )
        if symbolic.returncode not in {0, 1}:
            raise OfflineAdmissionError("Git detached-HEAD query failed")
        status = self._run_git(
            path,
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored",
            ],
        ).stdout
        rows = [item for item in status.split(b"\x00") if item]
        tracked_clean = not any(not row.startswith((b"??", b"!!")) for row in rows)
        untracked_clean = not any(row.startswith(b"??") for row in rows)
        ignored_clean = not any(row.startswith(b"!!") for row in rows)
        common_raw = self._run_git(
            path, ["rev-parse", "--git-common-dir"]
        ).stdout.decode("utf-8", errors="strict").strip()
        common = (
            common_raw
            if ntpath.isabs(common_raw)
            else os.path.abspath(os.path.join(path, common_raw))
        )
        try:
            common_outside = os.path.commonpath((path, common)) != path
        except ValueError:
            common_outside = True
        return GitWorktreeProof(
            worktree_path=path,
            head_commit=head,
            head_tree=tree,
            detached_head=symbolic.returncode == 1,
            tracked_clean=tracked_clean,
            untracked_clean=untracked_clean,
            ignored_clean=ignored_clean,
            common_dir_outside_worktree=common_outside,
        )

    def security_environment(self) -> Mapping[str, str | None]:
        _importlib, os, _subprocess, _Path = self._require_windows()
        folded = {name.upper(): value for name, value in os.environ.items()}
        return {
            "PYTHONNOUSERSITE": folded.get("PYTHONNOUSERSITE"),
            "PYTHONDONTWRITEBYTECODE": folded.get("PYTHONDONTWRITEBYTECODE"),
            "PYTHONHOME": folded.get("PYTHONHOME"),
            "PYTHONPATH": folded.get("PYTHONPATH"),
            "PYTHONSTARTUP": folded.get("PYTHONSTARTUP"),
        }

    def rederive_implementation_inventory(
        self, frozen_inventory: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        checked = attempt_contract.validate_implementation_inventory(
            frozen_inventory
        )
        worktree = self.current_working_directory().path
        listing = self._run_git(
            worktree, ["ls-tree", "-r", "--full-tree", "-z", "HEAD"]
        ).stdout
        requested: list[tuple[str, str]] = []
        for raw in listing.split(b"\x00"):
            if not raw:
                continue
            try:
                metadata, raw_path = raw.split(b"\t", 1)
                mode, kind, object_id = metadata.decode("ascii").split(" ")
                path = raw_path.decode("utf-8", errors="strict")
            except (ValueError, UnicodeError) as exc:
                raise OfflineAdmissionError("Git tree entry is malformed") from exc
            if kind == "blob" and mode in {"100644", "100755"}:
                requested.append((path, object_id))
        requested.sort(key=lambda item: item[0].encode("utf-8"))
        if len({path for path, _object_id in requested}) != len(requested):
            raise OfflineAdmissionError("Git tree paths are not unique")
        batch_input = b"".join(
            object_id.encode("ascii") + b"\n" for _path, object_id in requested
        )
        batch = self._run_git(
            worktree, ["cat-file", "--batch"], input_bytes=batch_input
        ).stdout
        cursor = 0
        entries: list[dict[str, Any]] = []
        for path, expected_object in requested:
            newline = batch.find(b"\n", cursor)
            if newline < 0:
                raise OfflineAdmissionError("Git blob batch header is truncated")
            try:
                object_id, kind, size_text = batch[cursor:newline].decode(
                    "ascii", errors="strict"
                ).split(" ")
                size = int(size_text)
            except (ValueError, UnicodeError) as exc:
                raise OfflineAdmissionError("Git blob batch header is malformed") from exc
            cursor = newline + 1
            end = cursor + size
            if (
                object_id != expected_object
                or kind != "blob"
                or size < 0
                or end >= len(batch)
                or batch[end : end + 1] != b"\n"
            ):
                raise OfflineAdmissionError("Git blob batch identity mismatched")
            payload = batch[cursor:end]
            cursor = end + 1
            entries.append(
                {
                    "path": path,
                    "size_bytes": size,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        if cursor != len(batch):
            raise OfflineAdmissionError("Git blob batch has trailing output")
        head = self.git_worktree(worktree)
        result = {
            "schema": "aigp-vq2-powered-implementation-inventory/1",
            "commit": head.head_commit,
            "tree": head.head_tree,
            "entries": entries,
        }
        return attempt_contract.validate_implementation_inventory(result)

    def rederive_environment_inventory(
        self, frozen_inventory: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        checked = attempt_contract.validate_environment_inventory(frozen_inventory)
        _importlib, os, _subprocess, _Path = self._require_windows()
        seen: set[str] = set()
        variables: list[dict[str, Any]] = []
        for raw_name, value in os.environ.items():
            name = raw_name.upper()
            if name in seen:
                raise OfflineAdmissionError(
                    "environment names collide case-insensitively"
                )
            seen.add(name)
            variables.append(
                {
                    "name": name,
                    "defined": True,
                    "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                }
            )
        variables.sort(key=lambda item: item["name"].casefold().encode("utf-8"))
        return attempt_contract.validate_environment_inventory(
            {
                "schema": "aigp-vq2-powered-environment-inventory/1",
                "created_at_utc": checked["created_at_utc"],
                "variables": variables,
            }
        )

    def _git_environment(self) -> dict[str, str]:
        values = {
            name: value
            for name, value in self._native_environment().items()
            if not name.startswith("=") and not name.startswith("GIT_")
        }
        values.update(
            {
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "GCM_INTERACTIVE": "Never",
                "LC_ALL": "C",
            }
        )
        return values

    def _abort_process(self, process: Any) -> None:
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass
        try:
            process.communicate(timeout=0.05)
        except (OSError, self._subprocess.SubprocessError):
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def _run_process(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        input_bytes: bytes | None,
        environment: Mapping[str, str],
        stdout_limit: int,
    ) -> Any:
        if not argv or any(type(item) is not str or not item for item in argv):
            raise OfflineAdmissionError("bounded process argv is invalid")
        if input_bytes is not None and len(input_bytes) > self._MAX_GIT_INPUT_BYTES:
            raise OfflineAdmissionError("bounded process input exceeds its limit")
        self._checkpoint()
        try:
            process = self._subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(environment),
                stdin=(
                    self._subprocess.PIPE
                    if input_bytes is not None
                    else self._subprocess.DEVNULL
                ),
                stdout=self._subprocess.PIPE,
                stderr=self._subprocess.PIPE,
                shell=False,
            )
        except OSError as exc:
            raise OfflineAdmissionError("bounded identity process failed to start") from exc
        pending_input = input_bytes
        try:
            while True:
                now = self._checkpoint()
                assert self._deadline_monotonic_ns is not None
                remaining = self._deadline_monotonic_ns - now
                timeout = min(self._POLL_INTERVAL_NS, remaining) / 1_000_000_000.0
                try:
                    stdout, stderr = process.communicate(
                        input=pending_input, timeout=timeout
                    )
                    break
                except self._subprocess.TimeoutExpired as exc:
                    pending_input = None
                    partial_stdout = exc.output or b""
                    partial_stderr = exc.stderr or b""
                    if (
                        len(partial_stdout) > stdout_limit
                        or len(partial_stderr) > self._MAX_GIT_STDERR_BYTES
                    ):
                        raise OfflineAdmissionError(
                            "bounded identity process output exceeded its limit"
                        )
            if (
                len(stdout) > stdout_limit
                or len(stderr) > self._MAX_GIT_STDERR_BYTES
            ):
                raise OfflineAdmissionError(
                    "bounded identity process output exceeded its limit"
                )
            self._checkpoint()
            return self._subprocess.CompletedProcess(
                list(argv), process.returncode, stdout, stderr
            )
        except BaseException:
            self._abort_process(process)
            raise

    def _run_git(
        self,
        worktree: str,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        check: bool = True,
        stdout_limit: int | None = None,
    ) -> Any:
        argv = [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            *arguments,
        ]
        result = self._run_process(
            argv,
            cwd=worktree,
            input_bytes=input_bytes,
            environment=self._git_environment(),
            stdout_limit=(
                self._MAX_GIT_STDOUT_BYTES
                if stdout_limit is None
                else stdout_limit
            ),
        )
        if check and result.returncode != 0:
            raise OfflineAdmissionError("bounded Git identity command failed")
        return result

    @staticmethod
    def _decode_git_scalar(payload: bytes, label: str) -> str:
        try:
            value = payload.decode("utf-8", errors="strict").strip()
        except UnicodeError as exc:
            raise OfflineAdmissionError(f"{label} is not UTF-8") from exc
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise OfflineAdmissionError(f"{label} is not one exact scalar")
        return value

    def _normalize_git_path(self, value: str) -> str:
        normalized = ntpath.normpath(value.replace("/", "\\"))
        if not ntpath.isabs(normalized):
            normalized = self._os.path.abspath(normalized)
        return self._lexical(normalized)

    def git_worktree(self, path: str) -> GitWorktreeProof:
        self._path_proof(path, directory=True)
        top = self._normalize_git_path(
            self._decode_git_scalar(
                self._run_git(
                    path,
                    ["rev-parse", "--show-toplevel"],
                    stdout_limit=64 * 1024,
                ).stdout,
                "Git worktree root",
            )
        )
        self._path_proof(top, directory=True)
        if top != path:
            raise OfflineAdmissionError("Git commands resolved a different worktree root")

        head_before = self._decode_git_scalar(
            self._run_git(
                path, ["rev-parse", "--verify", "HEAD^{commit}"], stdout_limit=256
            ).stdout,
            "Git HEAD",
        )
        tree_before = self._decode_git_scalar(
            self._run_git(
                path, ["rev-parse", "--verify", "HEAD^{tree}"], stdout_limit=256
            ).stdout,
            "Git tree",
        )
        if (
            _COMMIT_RE.fullmatch(head_before) is None
            or _COMMIT_RE.fullmatch(tree_before) is None
        ):
            raise OfflineAdmissionError("Git HEAD/tree identity is noncanonical")
        symbolic = self._run_git(
            path, ["symbolic-ref", "-q", "HEAD"], check=False, stdout_limit=4096
        )
        if symbolic.returncode not in {0, 1}:
            raise OfflineAdmissionError("Git detached-HEAD query failed")
        status = self._run_git(
            path,
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored",
                "--no-ahead-behind",
            ],
            stdout_limit=16 * 1024 * 1024,
        ).stdout
        rows = [item for item in status.split(b"\x00") if item]
        tracked_clean = not any(
            not row.startswith((b"??", b"!!")) for row in rows
        )
        untracked_clean = not any(row.startswith(b"??") for row in rows)
        ignored_clean = not any(row.startswith(b"!!") for row in rows)

        common_raw = self._decode_git_scalar(
            self._run_git(
                path, ["rev-parse", "--git-common-dir"], stdout_limit=64 * 1024
            ).stdout,
            "Git common directory",
        )
        common = self._normalize_git_path(
            common_raw
            if ntpath.isabs(common_raw)
            else ntpath.join(path, common_raw)
        )
        self._path_proof(common, directory=True)
        try:
            shared = ntpath.commonpath((path, common))
            common_outside = ntpath.normcase(shared) != ntpath.normcase(path)
        except ValueError:
            common_outside = True

        head_after = self._decode_git_scalar(
            self._run_git(
                path, ["rev-parse", "--verify", "HEAD^{commit}"], stdout_limit=256
            ).stdout,
            "final Git HEAD",
        )
        tree_after = self._decode_git_scalar(
            self._run_git(
                path, ["rev-parse", "--verify", "HEAD^{tree}"], stdout_limit=256
            ).stdout,
            "final Git tree",
        )
        if head_after != head_before or tree_after != tree_before:
            raise OfflineAdmissionError("Git HEAD/tree changed during identity proof")
        return GitWorktreeProof(
            worktree_path=path,
            head_commit=head_before,
            head_tree=tree_before,
            detached_head=symbolic.returncode == 1,
            tracked_clean=tracked_clean,
            untracked_clean=untracked_clean,
            ignored_clean=ignored_clean,
            common_dir_outside_worktree=common_outside,
        )

    def rederive_implementation_inventory(
        self, frozen_inventory: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        attempt_contract.validate_implementation_inventory(frozen_inventory)
        worktree = self.current_working_directory().path
        before = self.git_worktree(worktree)
        listing = self._run_git(
            worktree,
            ["ls-tree", "-r", "--full-tree", "-z", before.head_tree],
            stdout_limit=16 * 1024 * 1024,
        ).stdout
        requested: list[tuple[str, str]] = []
        for raw in listing.split(b"\x00"):
            self._checkpoint()
            if not raw:
                continue
            try:
                metadata, raw_path = raw.split(b"\t", 1)
                mode, kind, object_id = metadata.decode("ascii").split(" ")
                repository_path = raw_path.decode("utf-8", errors="strict")
            except (ValueError, UnicodeError) as exc:
                raise OfflineAdmissionError("Git tree entry is malformed") from exc
            if kind == "blob" and mode in {"100644", "100755"}:
                if re.fullmatch(r"[0-9a-f]{40,64}", object_id) is None:
                    raise OfflineAdmissionError("Git blob object ID is noncanonical")
                requested.append((repository_path, object_id))
        requested.sort(key=lambda item: item[0].encode("utf-8"))
        if len({item[0] for item in requested}) != len(requested):
            raise OfflineAdmissionError("Git tree paths are not unique")
        batch_input = b"".join(
            object_id.encode("ascii") + b"\n" for _path, object_id in requested
        )
        batch = self._run_git(
            worktree,
            ["cat-file", "--batch"],
            input_bytes=batch_input,
            stdout_limit=self._MAX_GIT_STDOUT_BYTES,
        ).stdout
        cursor = 0
        entries: list[dict[str, Any]] = []
        for repository_path, expected_object in requested:
            self._checkpoint()
            newline = batch.find(b"\n", cursor)
            if newline < 0:
                raise OfflineAdmissionError("Git blob batch header is truncated")
            try:
                object_id, kind, size_text = batch[cursor:newline].decode(
                    "ascii", errors="strict"
                ).split(" ")
                size = int(size_text)
            except (ValueError, UnicodeError) as exc:
                raise OfflineAdmissionError("Git blob batch header is malformed") from exc
            cursor = newline + 1
            end = cursor + size
            if (
                object_id != expected_object
                or kind != "blob"
                or size < 0
                or end >= len(batch)
                or batch[end : end + 1] != b"\n"
            ):
                raise OfflineAdmissionError("Git blob batch identity mismatched")
            payload = batch[cursor:end]
            cursor = end + 1
            entries.append(
                {
                    "path": repository_path,
                    "size_bytes": size,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        if cursor != len(batch):
            raise OfflineAdmissionError("Git blob batch has trailing output")
        after = self.git_worktree(worktree)
        if after != before:
            raise OfflineAdmissionError("Git identity changed during blob inventory")
        return attempt_contract.validate_implementation_inventory(
            {
                "schema": "aigp-vq2-powered-implementation-inventory/1",
                "commit": before.head_commit,
                "tree": before.head_tree,
                "entries": entries,
            }
        )

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            common = ntpath.commonpath((path, root))
        except ValueError:
            return False
        return ntpath.normcase(common) == ntpath.normcase(root)

    @classmethod
    def _classify_root(
        cls,
        path: str,
        *,
        candidate_root: str,
        venv_root: str,
        stdlib_root: str,
    ) -> str | None:
        if cls._under(path, candidate_root):
            return "candidate"
        if cls._under(path, venv_root):
            return "venv"
        if cls._under(path, stdlib_root):
            return "stdlib"
        return None

    def _bounded_import(self, importlib: Any, name: str) -> Any:
        self._checkpoint()
        if name in sys.modules:
            module = importlib.import_module(name)
            self._checkpoint()
            return module
        completed = self._threading.Event()
        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                result["module"] = importlib.import_module(name)
            except BaseException as exc:  # pragma: no cover - returned to caller
                result["error"] = exc
            finally:
                completed.set()

        thread = self._threading.Thread(
            target=worker,
            name=f"aigp-offline-import-{name}",
            daemon=True,
        )
        thread.start()
        while not completed.wait(self._POLL_INTERVAL_NS / 1_000_000_000.0):
            self._checkpoint()
        self._checkpoint()
        if "error" in result:
            raise OfflineAdmissionError(f"bounded import failed for {name!r}") from result[
                "error"
            ]
        return result["module"]

    def _runtime_roots(self, candidate_root: str, sysconfig: Any) -> tuple[str, str]:
        candidate = self._lexical(candidate_root)
        executable = self._lexical(self._os.path.abspath(sys.executable))
        venv_root = self._lexical(ntpath.dirname(ntpath.dirname(executable)))
        stdlib_root = self._lexical(self._os.path.abspath(sys.base_prefix))
        if self._lexical(self._os.path.abspath(sys.prefix)) != venv_root:
            raise OfflineAdmissionError("running venv root does not match interpreter path")
        for root in (candidate, venv_root, stdlib_root):
            self._path_proof(root, directory=True)
        for left, right in (
            (candidate, venv_root),
            (candidate, stdlib_root),
            (venv_root, stdlib_root),
        ):
            if self._under(left, right) or self._under(right, left):
                raise OfflineAdmissionError("candidate, venv, and stdlib roots overlap")

        paths = sysconfig.get_paths()
        expected: list[str] = []
        for value in (
            candidate,
            ntpath.join(
                stdlib_root,
                f"python{sys.version_info.major}{sys.version_info.minor}.zip",
            ),
            ntpath.join(stdlib_root, "DLLs"),
            paths.get("stdlib"),
            stdlib_root,
            venv_root,
            paths.get("purelib"),
            paths.get("platlib"),
        ):
            if not value:
                continue
            checked = self._lexical(self._os.path.abspath(value))
            if checked not in expected:
                expected.append(checked)
        observed = list(sys.path)
        if any(type(item) is not str or not item for item in observed):
            raise OfflineAdmissionError("sys.path contains an empty/non-string alias")
        if observed != expected:
            raise OfflineAdmissionError("sys.path contains an alternate import root")
        for path in expected:
            if self._os.path.exists(path):
                self._path_proof(path, directory=self._os.path.isdir(path))
        return venv_root, stdlib_root

    def _namespace_roots(
        self,
        locations: Any,
        *,
        candidate_root: str,
        venv_root: str,
        stdlib_root: str,
    ) -> tuple[list[str], set[str | None]]:
        if locations is None:
            return [], {None}
        roots = sorted(
            {self._lexical(self._os.path.abspath(str(value))) for value in locations},
            key=lambda value: value.encode("utf-8"),
        )
        classes: set[str | None] = set()
        for root in roots:
            self._path_proof(root, directory=True)
            classes.add(
                self._classify_root(
                    root,
                    candidate_root=candidate_root,
                    venv_root=venv_root,
                    stdlib_root=stdlib_root,
                )
            )
        return roots, classes

    @staticmethod
    def _loader_lookup_name(module_name: str, spec: Any) -> str:
        spec_name = getattr(spec, "name", None)
        if module_name == "__main__":
            if spec_name != PROBE_MODULE:
                raise OfflineAdmissionError(
                    "import-audit __main__ is not the powered probe module"
                )
            return PROBE_MODULE
        if type(spec_name) is str and spec_name:
            return spec_name
        return module_name

    @staticmethod
    def _runtime_attribute(root: Any, path: Sequence[str], label: str) -> Any:
        current = root
        for component in path:
            try:
                current = getattr(current, component)
            except AttributeError as exc:
                raise OfflineAdmissionError(
                    f"runtime import provider lacks {label}"
                ) from exc
        return current

    def _runtime_import_provider_identity(
        self,
        module_name: str,
        module: Any,
        *,
        candidate_root: str,
        venv_root: str,
        stdlib_root: str,
    ) -> tuple[str, FileIdentityProof]:
        provider = _POWERED_RUNTIME_IMPORT_PROVIDERS.get(module_name)
        if provider is None:
            raise OfflineAdmissionError(
                f"spec-less runtime import is not allowlisted: {module_name!r}"
            )
        if (
            getattr(module, "__spec__", None) is not None
            or getattr(module, "__file__", None) is not None
            or getattr(module, "__path__", None) is not None
            or getattr(module, "__name__", None) != module_name
        ):
            raise OfflineAdmissionError(
                f"spec-less runtime import shape changed: {module_name!r}"
            )
        provider_module_name, provider_path, value_path, expected_root_class = provider
        provider_module = sys.modules.get(provider_module_name)
        if provider_module is None:
            raise OfflineAdmissionError(
                f"runtime import provider is absent: {provider_module_name!r}"
            )
        if self._runtime_attribute(
            provider_module, value_path, f"value path for {module_name!r}"
        ) is not module:
            raise OfflineAdmissionError(
                f"runtime import is not owned by its provider: {module_name!r}"
            )
        provider_value = self._runtime_attribute(
            provider_module, provider_path, f"file path for {module_name!r}"
        )
        provider_spec = getattr(provider_value, "__spec__", None)
        origin = getattr(provider_spec, "origin", None)
        if type(origin) is not str:
            raise OfflineAdmissionError(
                f"runtime import provider has no file origin: {module_name!r}"
            )
        actual = self._lexical(self._os.path.abspath(origin))
        if actual != origin or getattr(provider_value, "__file__", None) != actual:
            raise OfflineAdmissionError(
                f"runtime import provider origin is noncanonical: {module_name!r}"
            )
        root_class = self._classify_root(
            actual,
            candidate_root=candidate_root,
            venv_root=venv_root,
            stdlib_root=stdlib_root,
        )
        if root_class != expected_root_class:
            raise OfflineAdmissionError(
                f"runtime import provider root changed: {module_name!r}"
            )
        loader_filename = getattr(
            getattr(provider_spec, "loader", None), "get_filename", None
        )
        provider_spec_name = getattr(provider_spec, "name", None)
        if callable(loader_filename):
            try:
                loader_origin = self._lexical(
                    self._os.path.abspath(loader_filename(provider_spec_name))
                )
            except (ImportError, AttributeError, OSError, TypeError) as exc:
                raise OfflineAdmissionError(
                    f"runtime import provider loader is invalid: {module_name!r}"
                ) from exc
            if loader_origin != actual:
                raise OfflineAdmissionError(
                    f"runtime import provider loader drifted: {module_name!r}"
                )
        return actual, self.observe_file_identity(actual, hash_kind="file_bytes")

    def _initial_import_entry(
        self,
        module_name: str,
        module: Any,
        *,
        candidate_root: str,
        venv_root: str,
        stdlib_root: str,
    ) -> dict[str, Any]:
        self._checkpoint()
        spec = getattr(module, "__spec__", None)
        if spec is None:
            origin, identity = self._runtime_import_provider_identity(
                module_name,
                module,
                candidate_root=candidate_root,
                venv_root=venv_root,
                stdlib_root=stdlib_root,
            )
            return {
                "module": module_name,
                "origin": origin,
                "size_bytes": identity.size_bytes,
                "sha256": identity.sha256,
                "root_class": "runtime",
                "namespace_roots": [],
            }

        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            return {
                "module": module_name,
                "origin": None,
                "size_bytes": None,
                "sha256": None,
                "root_class": "builtin" if origin == "built-in" else "frozen",
                "namespace_roots": [],
            }

        locations = getattr(spec, "submodule_search_locations", None)
        if origin is None and locations is not None:
            roots, classes = self._namespace_roots(
                locations,
                candidate_root=candidate_root,
                venv_root=venv_root,
                stdlib_root=stdlib_root,
            )
            if not roots or len(classes) != 1 or None in classes:
                raise OfflineAdmissionError(
                    f"namespace import roots are mixed or unclassified: {module_name!r}"
                )
            return {
                "module": module_name,
                "origin": None,
                "size_bytes": None,
                "sha256": None,
                "root_class": "namespace",
                "namespace_roots": roots,
            }
        if type(origin) is not str:
            raise OfflineAdmissionError(
                f"import origin is not classifiable: {module_name!r}"
            )

        actual = self._lexical(self._os.path.abspath(origin))
        if actual != origin:
            raise OfflineAdmissionError(
                f"import origin is not canonical absolute: {module_name!r}"
            )
        root_class = self._classify_root(
            actual,
            candidate_root=candidate_root,
            venv_root=venv_root,
            stdlib_root=stdlib_root,
        )
        if root_class is None:
            raise OfflineAdmissionError(
                f"import origin is outside every frozen root: {module_name!r}"
            )
        module_file = getattr(module, "__file__", None)
        if type(module_file) is not str or self._os.path.abspath(module_file) != actual:
            raise OfflineAdmissionError(
                f"module file does not equal its import origin: {module_name!r}"
            )
        loader_filename = getattr(getattr(spec, "loader", None), "get_filename", None)
        if callable(loader_filename):
            lookup_name = self._loader_lookup_name(module_name, spec)
            try:
                loader_origin = self._lexical(
                    self._os.path.abspath(loader_filename(lookup_name))
                )
            except (ImportError, AttributeError, OSError) as exc:
                raise OfflineAdmissionError(
                    f"loader origin could not be verified: {module_name!r}"
                ) from exc
            if loader_origin != actual:
                raise OfflineAdmissionError(
                    f"loader origin does not equal module origin: {module_name!r}"
                )
        identity = self.observe_file_identity(actual, hash_kind="file_bytes")
        return {
            "module": module_name,
            "origin": actual,
            "size_bytes": identity.size_bytes,
            "sha256": identity.sha256,
            "root_class": root_class,
            "namespace_roots": [],
        }

    def derive_initial_import_inventory(
        self,
        seed_modules: Sequence[str],
        eager_modules: Sequence[str],
        *,
        audit_module: str,
    ) -> Mapping[str, Any]:
        """Derive the complete graph in one isolated, bounded L0 interpreter."""

        import importlib

        if tuple(seed_modules) != POWERED_IMPORT_SEED_MODULES:
            raise OfflineAdmissionError("powered import seed inventory changed")
        if tuple(eager_modules) != POWERED_EAGER_IMPORT_MODULES:
            raise OfflineAdmissionError("powered eager-import inventory changed")
        if audit_module != IMPORT_AUDIT_MODULE:
            raise OfflineAdmissionError("initial import audit module changed")
        for name in seed_modules:
            module = sys.modules.get(name)
            if module is None or getattr(getattr(module, "__spec__", None), "name", None) != name:
                raise OfflineAdmissionError(
                    f"powered import seed was not loaded in audit order: {name!r}"
                )
        for name in eager_modules:
            self._bounded_import(importlib, name)

        audit_main = sys.modules.get("__main__")
        audit_spec = getattr(audit_main, "__spec__", None)
        if getattr(audit_spec, "name", None) != audit_module:
            raise OfflineAdmissionError(
                "initial import inventory did not use the exact audit module"
            )
        production_main = sys.modules.get(PROBE_MODULE)
        if production_main is None:
            raise OfflineAdmissionError("powered probe seed module is absent")
        production_spec = getattr(production_main, "__spec__", None)
        if getattr(production_spec, "name", None) != PROBE_MODULE:
            raise OfflineAdmissionError("powered probe seed identity is invalid")
        # The live process executes PROBE_MODULE with -m.  Normalize the audit
        # interpreter's sole execution-module alias to that exact future
        # identity before taking the complete sys.modules snapshot.
        sys.modules["__main__"] = production_main

        sysconfig = self._bounded_import(importlib, "sysconfig")
        candidate_root = self.current_working_directory().path
        venv_root, stdlib_root = self._runtime_roots(candidate_root, sysconfig)
        module_snapshot = sorted(
            (
                (name, module)
                for name, module in tuple(sys.modules.items())
                if module is not None
            ),
            key=lambda item: item[0].encode("utf-8"),
        )
        module_names = [name for name, _module in module_snapshot]
        if len(module_names) != len(set(module_names)):
            raise OfflineAdmissionError("sys.modules contains duplicate names")
        entries = [
            self._initial_import_entry(
                name,
                module,
                candidate_root=candidate_root,
                venv_root=venv_root,
                stdlib_root=stdlib_root,
            )
            for name, module in module_snapshot
        ]
        final_snapshot = sorted(
            (
                (name, module)
                for name, module in tuple(sys.modules.items())
                if module is not None
            ),
            key=lambda item: item[0].encode("utf-8"),
        )
        if len(final_snapshot) != len(module_snapshot) or any(
            final_name != initial_name or final_module is not initial_module
            for (initial_name, initial_module), (final_name, final_module) in zip(
                module_snapshot, final_snapshot, strict=True
            )
        ):
            raise OfflineAdmissionError(
                "sys.modules changed while the initial inventory was derived"
            )
        python_identity = self.observe_file_identity(
            self._os.path.abspath(sys.executable), hash_kind="file_bytes"
        )
        return attempt_contract.validate_import_inventory(
            {
                "schema": "aigp-vq2-powered-import-inventory/1",
                "python_sha256": python_identity.sha256,
                "seeds": list(seed_modules),
                "entries": entries,
            }
        )

    def _rederive_import_inventory_hardened(
        self,
        frozen_inventory: Mapping[str, Any],
        eager_modules: Sequence[str],
    ) -> ImportRevalidation:
        frozen = attempt_contract.validate_import_inventory(frozen_inventory)
        import importlib

        if tuple(eager_modules) != POWERED_EAGER_IMPORT_MODULES:
            raise OfflineAdmissionError("powered eager-import inventory changed")
        for name in tuple(frozen["seeds"]) + tuple(eager_modules):
            self._bounded_import(importlib, name)
        sysconfig = self._bounded_import(importlib, "sysconfig")

        candidate_root = self.current_working_directory().path
        venv_root, stdlib_root = self._runtime_roots(candidate_root, sysconfig)
        frozen_names = {entry["module"] for entry in frozen["entries"]}
        refreshed: list[dict[str, Any]] = []
        origins_reverified = True
        for entry in frozen["entries"]:
            self._checkpoint()
            name = entry["module"]
            module = sys.modules.get(name)
            if module is None:
                origins_reverified = False
                refreshed.append(dict(entry))
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            root_class = entry["root_class"]
            candidate = dict(entry)
            if root_class in {"builtin", "frozen"}:
                expected_origin = "built-in" if root_class == "builtin" else "frozen"
                if origin != expected_origin:
                    origins_reverified = False
            elif root_class == "runtime":
                try:
                    runtime_origin, runtime_identity = (
                        self._runtime_import_provider_identity(
                            name,
                            module,
                            candidate_root=candidate_root,
                            venv_root=venv_root,
                            stdlib_root=stdlib_root,
                        )
                    )
                except OfflineAdmissionError:
                    origins_reverified = False
                else:
                    if runtime_origin != entry["origin"]:
                        origins_reverified = False
                    candidate["size_bytes"] = runtime_identity.size_bytes
                    candidate["sha256"] = runtime_identity.sha256
            elif root_class == "namespace":
                roots, classes = self._namespace_roots(
                    getattr(spec, "submodule_search_locations", None),
                    candidate_root=candidate_root,
                    venv_root=venv_root,
                    stdlib_root=stdlib_root,
                )
                if (
                    roots != entry["namespace_roots"]
                    or len(classes) != 1
                    or None in classes
                ):
                    origins_reverified = False
            else:
                if type(origin) is not str:
                    origins_reverified = False
                else:
                    actual = self._lexical(self._os.path.abspath(origin))
                    actual_class = self._classify_root(
                        actual,
                        candidate_root=candidate_root,
                        venv_root=venv_root,
                        stdlib_root=stdlib_root,
                    )
                    if actual != entry["origin"] or actual_class != root_class:
                        origins_reverified = False
                    module_file = getattr(module, "__file__", actual)
                    if (
                        type(module_file) is not str
                        or self._os.path.abspath(module_file) != actual
                    ):
                        origins_reverified = False
                    loader_filename = getattr(getattr(spec, "loader", None), "get_filename", None)
                    if callable(loader_filename):
                        try:
                            lookup_name = self._loader_lookup_name(name, spec)
                            if self._os.path.abspath(loader_filename(lookup_name)) != actual:
                                origins_reverified = False
                        except (ImportError, AttributeError, OSError):
                            origins_reverified = False
                    identity = self.observe_file_identity(
                        entry["origin"], hash_kind="file_bytes"
                    )
                    candidate["size_bytes"] = identity.size_bytes
                    candidate["sha256"] = identity.sha256
            refreshed.append(candidate)

        unexpected: list[str] = []
        unclassified: list[str] = []
        for name, module in tuple(sys.modules.items()):
            self._checkpoint()
            if module is None:
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            if origin in {"built-in", "frozen"}:
                continue
            locations = getattr(spec, "submodule_search_locations", None)
            if origin is None and locations is not None:
                roots, classes = self._namespace_roots(
                    locations,
                    candidate_root=candidate_root,
                    venv_root=venv_root,
                    stdlib_root=stdlib_root,
                )
                if None in classes or len(classes) != 1:
                    unclassified.extend(roots)
                if name not in frozen_names and classes & {"candidate", "venv"}:
                    unexpected.append(name)
                continue
            if origin is None:
                if name not in frozen_names:
                    unclassified.append(f"runtime:{name}")
                continue
            actual = self._lexical(self._os.path.abspath(str(origin)))
            root_class = self._classify_root(
                actual,
                candidate_root=candidate_root,
                venv_root=venv_root,
                stdlib_root=stdlib_root,
            )
            if root_class in {"candidate", "venv"} and name not in frozen_names:
                unexpected.append(name)
            elif root_class is None:
                unclassified.append(actual)

        inventory = attempt_contract.validate_import_inventory(
            {
                "schema": "aigp-vq2-powered-import-inventory/1",
                "python_sha256": self.observe_file_identity(
                    self._os.path.abspath(sys.executable), hash_kind="file_bytes"
                ).sha256,
                "seeds": list(frozen["seeds"]),
                "entries": refreshed,
            }
        )
        return ImportRevalidation(
            inventory=inventory,
            origins_reverified=origins_reverified,
            user_site_on_sys_path=False,
            unexpected_candidate_or_venv_modules=tuple(
                sorted(set(unexpected), key=lambda item: item.encode("utf-8"))
            ),
            unclassified_origins=tuple(
                sorted(set(unclassified), key=lambda item: item.encode("utf-8"))
            ),
        )

    @staticmethod
    def _under(path: str, root: str) -> bool:
        import os

        try:
            return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
        except ValueError:
            return False

    def rederive_import_inventory(
        self,
        frozen_inventory: Mapping[str, Any],
        eager_modules: Sequence[str],
    ) -> ImportRevalidation:
        frozen = attempt_contract.validate_import_inventory(frozen_inventory)
        import importlib
        import os
        import site
        import sysconfig

        if tuple(eager_modules) != POWERED_EAGER_IMPORT_MODULES:
            raise OfflineAdmissionError("powered eager-import inventory changed")
        for name in tuple(frozen["seeds"]) + tuple(eager_modules):
            importlib.import_module(name)

        candidate_root = self.current_working_directory().path
        venv_root = os.path.abspath(sys.prefix)
        stdlib_roots = tuple(
            dict.fromkeys(
                os.path.abspath(value)
                for value in (
                    sysconfig.get_paths().get("stdlib"),
                    sysconfig.get_paths().get("platstdlib"),
                )
                if value
            )
        )
        frozen_names = {entry["module"] for entry in frozen["entries"]}
        refreshed: list[dict[str, Any]] = []
        origins_reverified = True
        for entry in frozen["entries"]:
            name = entry["module"]
            module = sys.modules.get(name)
            if module is None:
                origins_reverified = False
                refreshed.append(dict(entry))
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            root_class = entry["root_class"]
            candidate = dict(entry)
            if root_class in {"builtin", "frozen"}:
                expected = "built-in" if root_class == "builtin" else "frozen"
                if origin != expected:
                    origins_reverified = False
            elif root_class == "namespace":
                locations = getattr(spec, "submodule_search_locations", None)
                roots = (
                    []
                    if locations is None
                    else sorted(
                        {os.path.abspath(str(value)) for value in locations},
                        key=lambda value: value.encode("utf-8"),
                    )
                )
                if roots != entry["namespace_roots"]:
                    origins_reverified = False
            else:
                if type(origin) is not str:
                    origins_reverified = False
                else:
                    actual = os.path.abspath(origin)
                    if os.path.normcase(actual) != os.path.normcase(entry["origin"]):
                        origins_reverified = False
                    try:
                        identity = self.observe_file_identity(
                            entry["origin"], hash_kind="file_bytes"
                        )
                    except (OfflineAdmissionError, OSError, RuntimeError):
                        origins_reverified = False
                    else:
                        candidate["size_bytes"] = identity.size_bytes
                        candidate["sha256"] = identity.sha256
            refreshed.append(candidate)

        unexpected: list[str] = []
        unclassified: list[str] = []
        for name, module in tuple(sys.modules.items()):
            if module is None:
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            if origin in {None, "built-in", "frozen"}:
                continue
            actual = os.path.abspath(str(origin))
            in_candidate = self._under(actual, candidate_root)
            in_venv = self._under(actual, venv_root)
            in_stdlib = any(self._under(actual, root) for root in stdlib_roots)
            if (in_candidate or in_venv) and name not in frozen_names:
                unexpected.append(name)
            elif not (in_candidate or in_venv or in_stdlib):
                unclassified.append(actual)

        user_site = site.getusersitepackages()
        user_roots = (
            [user_site] if type(user_site) is str else list(user_site or ())
        )
        user_site_on_path = any(
            any(self._under(entry, root) or self._under(root, entry) for root in user_roots)
            for entry in sys.path
            if type(entry) is str and entry
        )
        inventory = attempt_contract.validate_import_inventory(
            {
                "schema": "aigp-vq2-powered-import-inventory/1",
                "python_sha256": self.observe_file_identity(
                    os.path.abspath(sys.executable), hash_kind="file_bytes"
                ).sha256,
                "seeds": list(frozen["seeds"]),
                "entries": refreshed,
            }
        )
        return ImportRevalidation(
            inventory=inventory,
            origins_reverified=origins_reverified,
            user_site_on_sys_path=user_site_on_path,
            unexpected_candidate_or_venv_modules=tuple(
                sorted(set(unexpected), key=lambda item: item.encode("utf-8"))
            ),
            unclassified_origins=tuple(
                sorted(set(unclassified), key=lambda item: item.encode("utf-8"))
            ),
        )


class WindowsProductionOfflineAdmission(_WindowsProductionOfflineAdmissionBase):
    """Handle-derived, deadline-bounded production identity admission.

    Construction loads Win32 entry points but performs no filesystem, Git,
    process, simulator, socket, mutex, or private-root operation.  Every public
    observation must run inside ``admit_offline``'s absolute QPC scope.  Handles
    are retained across the first and pre-release admissions so a same-byte
    replacement cannot invalidate the reviewed identity between those gates.
    """

    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_READ_ATTRIBUTES = 0x00000080
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_EXISTING = 3
    _FILE_BEGIN = 0
    _DRIVE_FIXED = 3
    _POLL_INTERVAL_NS = 50_000_000
    _HEARTBEAT_PERIOD_NS = 1_000_000_000
    _MAX_JSON_BYTES = 64 * 1024 * 1024
    _MAX_IDENTITY_BYTES = 2 * 1024 * 1024 * 1024
    _MAX_GIT_INPUT_BYTES = 4 * 1024 * 1024
    _MAX_GIT_STDOUT_BYTES = 128 * 1024 * 1024
    _MAX_GIT_STDERR_BYTES = 1024 * 1024
    _SYSTEM_POWERSHELL_PATH = (
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    )
    _SYSTEM_POWERSHELL_LINK_COUNT = 2

    def __init__(self) -> None:
        import ctypes
        import os
        import subprocess
        import threading
        from ctypes import wintypes

        if os.name != "nt":
            raise OfflineAdmissionError("powered production admission requires Windows")
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._subprocess = subprocess
        self._threading = threading
        self._os = os
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)

        class FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", FILETIME),
                ("ftLastAccessTime", FILETIME),
                ("ftLastWriteTime", FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            ]

        self._BY_HANDLE_FILE_INFORMATION = BY_HANDLE_FILE_INFORMATION
        self._INVALID_HANDLE = ctypes.c_void_p(-1).value
        kernel = self._kernel
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(BY_HANDLE_FILE_INFORMATION),
        ]
        kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        kernel.GetDriveTypeW.restype = wintypes.UINT
        kernel.SetFilePointerEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.DWORD,
        ]
        kernel.SetFilePointerEx.restype = wintypes.BOOL
        kernel.ReadFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        kernel.ReadFile.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.GetEnvironmentStringsW.argtypes = []
        kernel.GetEnvironmentStringsW.restype = ctypes.c_void_p
        kernel.FreeEnvironmentStringsW.argtypes = [ctypes.c_void_p]
        kernel.FreeEnvironmentStringsW.restype = wintypes.BOOL

        self._deadline_monotonic_ns: int | None = None
        self._clock: Callable[[], int] | None = None
        self._heartbeat_callback: Callable[[], None] | None = None
        self._last_heartbeat_monotonic_ns: int | None = None
        self._retained: dict[str, tuple[int, Any, bool]] = {}
        self._closed = False

    def _winerror(self, label: str) -> OfflineAdmissionError:
        return OfflineAdmissionError(
            f"{label} failed with Win32 error "
            f"{int(self._ctypes.get_last_error())}"
        )

    def begin_bounded_admission(
        self,
        *,
        deadline_monotonic_ns: int,
        monotonic_ns: Callable[[], int],
        heartbeat: Callable[[], None] | None,
    ) -> None:
        if self._closed:
            raise OfflineAdmissionError("offline admission boundary is closed")
        if self._deadline_monotonic_ns is not None:
            raise OfflineAdmissionError("offline admission scopes cannot nest")
        deadline = _require_exact_nonnegative_int(
            deadline_monotonic_ns, "offline admission deadline"
        )
        if not callable(monotonic_ns):
            raise TypeError("offline admission monotonic clock must be callable")
        if heartbeat is not None and not callable(heartbeat):
            raise TypeError("offline admission heartbeat must be callable")
        self._deadline_monotonic_ns = deadline
        self._clock = monotonic_ns
        self._heartbeat_callback = heartbeat
        now = self._now()
        if now >= deadline:
            self._clear_budget()
            raise OfflineAdmissionError("offline admission deadline already expired")
        self._last_heartbeat_monotonic_ns = now

    def _clear_budget(self) -> None:
        self._deadline_monotonic_ns = None
        self._clock = None
        self._heartbeat_callback = None
        self._last_heartbeat_monotonic_ns = None

    def end_bounded_admission(self, *, succeeded: bool) -> None:
        failure: BaseException | None = None
        try:
            if succeeded:
                self._checkpoint()
        except BaseException as exc:
            failure = exc
        finally:
            self._clear_budget()
        if not succeeded or failure is not None:
            try:
                self.close()
            except BaseException as close_exc:
                if failure is None:
                    failure = close_exc
                else:
                    failure.add_note(
                        "retained offline handle close also failed: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )
        if failure is not None:
            raise failure

    def _now(self) -> int:
        if self._clock is None:
            raise OfflineAdmissionError("offline admission has no active QPC scope")
        return _require_exact_nonnegative_int(
            self._clock(), "offline admission QPC observation"
        )

    def _checkpoint(self) -> int:
        deadline = self._deadline_monotonic_ns
        if deadline is None:
            raise OfflineAdmissionError("offline operation escaped its QPC scope")
        now = self._now()
        if now >= deadline:
            raise OfflineAdmissionError("offline admission absolute deadline expired")
        callback = self._heartbeat_callback
        prior = self._last_heartbeat_monotonic_ns
        if (
            callback is not None
            and prior is not None
            and now - prior >= self._HEARTBEAT_PERIOD_NS
        ):
            callback()
            after = self._now()
            if after >= deadline:
                raise OfflineAdmissionError(
                    "offline admission expired during lease heartbeat"
                )
            self._last_heartbeat_monotonic_ns = after
            now = after
        return now

    def close(self) -> None:
        if self._closed:
            return
        failures: list[str] = []
        for path, (handle, _info, _directory) in reversed(
            tuple(self._retained.items())
        ):
            try:
                self._close_handle(handle)
            except BaseException as exc:
                failures.append(f"{path}:{type(exc).__name__}")
        self._retained.clear()
        self._closed = True
        if failures:
            raise OfflineAdmissionError(
                "offline retained-handle close failures: " + ",".join(failures)
            )

    @staticmethod
    def _lexical(path: str) -> str:
        try:
            return attempt_contract.validate_absolute_windows_path(
                path, path="$offline_path"
            )
        except attempt_contract.PoweredAttemptContractError as exc:
            raise OfflineAdmissionError(str(exc)) from exc

    def _close_handle(self, handle: int) -> None:
        if not self._kernel.CloseHandle(handle):
            raise self._winerror("CloseHandle(offline identity)")

    def _final_path(self, handle: int) -> str:
        required = int(self._kernel.GetFinalPathNameByHandleW(handle, None, 0, 0))
        if required <= 0:
            raise self._winerror("GetFinalPathNameByHandleW(size)")
        buffer = self._ctypes.create_unicode_buffer(required + 1)
        length = int(
            self._kernel.GetFinalPathNameByHandleW(
                handle, buffer, len(buffer), 0
            )
        )
        if length <= 0 or length >= len(buffer):
            raise self._winerror("GetFinalPathNameByHandleW")
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            raise OfflineAdmissionError("network identity paths are forbidden")
        if value.startswith("\\\\?\\"):
            value = value[4:]
        return self._lexical(value)

    @classmethod
    def _expected_file_link_count(cls, path: str) -> int:
        if path == cls._SYSTEM_POWERSHELL_PATH:
            return cls._SYSTEM_POWERSHELL_LINK_COUNT
        return 1

    def _file_info(self, handle: int, *, path: str, directory: bool) -> Any:
        info = self._BY_HANDLE_FILE_INFORMATION()
        if not self._kernel.GetFileInformationByHandle(
            handle, self._ctypes.byref(info)
        ):
            raise self._winerror("GetFileInformationByHandle(offline identity)")
        attributes = int(info.dwFileAttributes)
        if bool(attributes & self._FILE_ATTRIBUTE_DIRECTORY) is not directory:
            raise OfflineAdmissionError("offline identity has the wrong object kind")
        if attributes & self._FILE_ATTRIBUTE_REPARSE_POINT:
            raise OfflineAdmissionError("offline identity is a reparse point")
        if not directory:
            expected_links = self._expected_file_link_count(path)
            if int(info.nNumberOfLinks) != expected_links:
                if expected_links != 1:
                    raise OfflineAdmissionError(
                        "offline system PowerShell component-store identity changed"
                    )
                raise OfflineAdmissionError(
                    "offline file has an aliased hard-link identity"
                )
        return info

    @staticmethod
    def _identity_tuple(info: Any) -> tuple[int, ...]:
        return (
            int(info.dwVolumeSerialNumber),
            int(info.nFileIndexHigh),
            int(info.nFileIndexLow),
            int(info.ftCreationTime.dwHighDateTime),
            int(info.ftCreationTime.dwLowDateTime),
        )

    @classmethod
    def _file_state_tuple(cls, info: Any) -> tuple[int, ...]:
        return (
            *cls._identity_tuple(info),
            int(info.nNumberOfLinks),
            int(info.nFileSizeHigh),
            int(info.nFileSizeLow),
            int(info.ftLastWriteTime.dwHighDateTime),
            int(info.ftLastWriteTime.dwLowDateTime),
        )

    @staticmethod
    def _size(info: Any) -> int:
        return (int(info.nFileSizeHigh) << 32) | int(info.nFileSizeLow)

    @staticmethod
    def _volume_id(info: Any) -> str:
        return f"volume-{int(info.dwVolumeSerialNumber):08x}"

    def _retain_one(self, path: str, *, directory: bool) -> tuple[int, Any]:
        prior = self._retained.get(path)
        if prior is not None:
            handle, initial, initial_directory = prior
            if initial_directory is not directory:
                raise OfflineAdmissionError("retained path kind changed")
            current = self._file_info(handle, path=path, directory=directory)
            if (
                self._identity_tuple(current) != self._identity_tuple(initial)
                or (
                    not directory
                    and int(current.nNumberOfLinks)
                    != int(initial.nNumberOfLinks)
                )
                or self._final_path(handle) != path
            ):
                raise OfflineAdmissionError("retained path identity changed")
            return handle, current

        flags = self._FILE_FLAG_OPEN_REPARSE_POINT
        access = self._FILE_READ_ATTRIBUTES
        share = self._FILE_SHARE_READ | self._FILE_SHARE_WRITE
        if directory:
            flags |= self._FILE_FLAG_BACKUP_SEMANTICS
        else:
            access |= self._GENERIC_READ
            share = self._FILE_SHARE_READ
        handle = self._kernel.CreateFileW(
            path,
            access,
            share,
            None,
            self._OPEN_EXISTING,
            flags,
            None,
        )
        if handle == self._INVALID_HANDLE:
            raise self._winerror("CreateFileW(open offline identity)")
        value = int(handle)
        try:
            info = self._file_info(value, path=path, directory=directory)
            final = self._final_path(value)
            if final != path:
                raise OfflineAdmissionError(
                    f"offline handle final path alias: {final!r} != {path!r}"
                )
        except BaseException:
            self._close_handle(value)
            raise
        self._retained[path] = (value, info, directory)
        return value, info

    def _retain_path(self, path: str, *, directory: bool) -> tuple[int, Any]:
        checked = self._lexical(path)
        drive, tail = ntpath.splitdrive(checked)
        root = drive + "\\"
        if int(self._kernel.GetDriveTypeW(root)) != self._DRIVE_FIXED:
            raise OfflineAdmissionError("offline identity must reside on a fixed drive")
        parts = [part for part in tail.lstrip("\\").split("\\") if part]
        cursor = root
        self._retain_one(cursor, directory=True)
        directory_parts = parts if directory else parts[:-1]
        for part in directory_parts:
            cursor = ntpath.join(cursor, part)
            self._retain_one(cursor, directory=True)
        if directory:
            return self._retain_one(checked, directory=True)
        return self._retain_one(checked, directory=False)

    def _path_proof(self, path: str, *, directory: bool) -> PathProof:
        self._checkpoint()
        _handle, info = self._retain_path(path, directory=directory)
        self._checkpoint()
        return PathProof(
            path=path,
            final_path=path,
            kind="directory" if directory else "file",
            volume_id=self._volume_id(info),
            exists=True,
            non_reparse=True,
            ancestors_non_reparse=True,
            retained_handle=True,
        )

    def _read_retained_file(
        self,
        path: str,
        *,
        maximum_bytes: int,
        collect: bool,
    ) -> tuple[int, str, bytes | None]:
        handle, before = self._retain_path(path, directory=False)
        size = self._size(before)
        if size > maximum_bytes:
            raise OfflineAdmissionError("offline identity file exceeds its size bound")
        if not self._kernel.SetFilePointerEx(
            handle, 0, None, self._FILE_BEGIN
        ):
            raise self._winerror("SetFilePointerEx(offline identity)")
        digest = hashlib.sha256()
        collected = bytearray() if collect else None
        total = 0
        buffer = self._ctypes.create_string_buffer(64 * 1024)
        while total < size:
            self._checkpoint()
            requested = min(len(buffer), size - total)
            received = self._wintypes.DWORD(0)
            if not self._kernel.ReadFile(
                handle,
                buffer,
                requested,
                self._ctypes.byref(received),
                None,
            ):
                raise self._winerror("ReadFile(offline identity)")
            count = int(received.value)
            if count <= 0 or count > requested:
                raise OfflineAdmissionError("offline identity read was truncated")
            chunk = buffer.raw[:count]
            digest.update(chunk)
            if collected is not None:
                collected.extend(chunk)
            total += count
        after = self._file_info(handle, path=path, directory=False)
        if (
            total != size
            or self._file_state_tuple(after) != self._file_state_tuple(before)
            or self._final_path(handle) != path
        ):
            raise OfflineAdmissionError("offline file identity changed while hashing")
        self._checkpoint()
        return total, digest.hexdigest(), None if collected is None else bytes(collected)

    def read_stable_json(self, path: str) -> StableJsonProof:
        proof = self._path_proof(path, directory=False)
        size, digest, raw = self._read_retained_file(
            path, maximum_bytes=self._MAX_JSON_BYTES, collect=True
        )
        assert raw is not None
        try:
            value = attempt_contract.parse_canonical_json_bytes(raw, file_form=True)
        except attempt_contract.PoweredAttemptContractError as exc:
            raise OfflineAdmissionError("JSON file is not canonical") from exc
        return StableJsonProof(
            identity=FileIdentityProof(
                path=proof,
                size_bytes=size,
                sha256=digest,
            ),
            raw_bytes=raw,
            value=value,
        )

    def observe_file_identity(
        self, path: str, *, hash_kind: str
    ) -> FileIdentityProof:
        proof = self._path_proof(path, directory=False)
        if hash_kind == "file_bytes":
            size, digest, _raw = self._read_retained_file(
                path, maximum_bytes=self._MAX_IDENTITY_BYTES, collect=False
            )
        elif hash_kind == "canonical_object":
            document = self.read_stable_json(path)
            size = document.identity.size_bytes
            digest = attempt_contract.canonical_object_sha256(document.value)
        else:
            raise OfflineAdmissionError("unsupported identity hash kind")
        return FileIdentityProof(
            path=proof,
            size_bytes=size,
            sha256=digest,
            hash_kind=hash_kind,
            stable_before_after=True,
        )

    def current_working_directory(self) -> PathProof:
        return self._path_proof(self._os.getcwd(), directory=True)

    def module_origin(self, module_name: str) -> PathProof:
        if type(module_name) is not str or not module_name:
            raise OfflineAdmissionError("module name must be nonempty")
        module = sys.modules.get(module_name)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if type(origin) is not str:
            raise OfflineAdmissionError("module has no file-backed origin")
        if self._os.path.abspath(origin) != origin:
            raise OfflineAdmissionError("module origin is not canonical absolute")
        return self._path_proof(origin, directory=False)

    def validate_exact_invocation(
        self,
        freeze: Mapping[str, Any],
        arguments: ProbeArguments,
    ) -> None:
        self._checkpoint()
        expected_executable = freeze["runtime"]["python"]["path"]
        if self._os.path.abspath(sys.executable) != expected_executable:
            raise OfflineAdmissionError("running interpreter path does not match freeze")
        self._path_proof(expected_executable, directory=False)
        expected_tail = [
            "-E",
            "-s",
            "-B",
            "-m",
            PROBE_MODULE,
            "--live-freeze",
            arguments.live_freeze,
            "--live-freeze-sha256",
            arguments.live_freeze_sha256,
            "--expected-commit",
            arguments.expected_commit,
        ]
        observed = list(getattr(sys, "orig_argv", ()))
        self._validate_invocation_values(
            implementation=sys.implementation.name,
            version=tuple(sys.version_info[:3]),
            ignore_environment=sys.flags.ignore_environment,
            no_user_site=sys.flags.no_user_site,
            dont_write_bytecode=sys.flags.dont_write_bytecode,
            observed_argv=observed,
            expected_tail=expected_tail,
        )

    @staticmethod
    def _validate_invocation_values(
        *,
        implementation: str,
        version: tuple[int, ...],
        ignore_environment: int,
        no_user_site: int,
        dont_write_bytecode: int,
        observed_argv: Sequence[str],
        expected_tail: Sequence[str],
    ) -> None:
        if implementation != "cpython" or version != (3, 12, 2):
            raise OfflineAdmissionError("running interpreter build is not CPython 3.12.2")
        if (ignore_environment, no_user_site, dont_write_bytecode) != (1, 1, 1):
            raise OfflineAdmissionError("interpreter lacks exact -E -s -B isolation")
        if (
            len(observed_argv) != len(expected_tail) + 1
            or list(observed_argv[1:]) != list(expected_tail)
        ):
            raise OfflineAdmissionError("wrapper invocation is not the sole frozen argv")

    def _native_environment(self) -> dict[str, str]:
        self._checkpoint()
        block = self._kernel.GetEnvironmentStringsW()
        if not block:
            raise self._winerror("GetEnvironmentStringsW")
        values: dict[str, str] = {}
        step = self._ctypes.sizeof(self._ctypes.c_wchar)
        offset = 0
        try:
            while True:
                raw = self._ctypes.wstring_at(block + offset * step)
                if not raw:
                    break
                offset += len(raw) + 1
                separator = raw.find("=", 1 if raw.startswith("=") else 0)
                if separator <= 0:
                    raise OfflineAdmissionError("Windows environment entry is malformed")
                name = raw[:separator].upper()
                value = raw[separator + 1 :]
                if name in values:
                    raise OfflineAdmissionError(
                        "Windows environment names collide case-insensitively"
                    )
                values[name] = value
        finally:
            if not self._kernel.FreeEnvironmentStringsW(block):
                raise self._winerror("FreeEnvironmentStringsW")
        self._checkpoint()
        return values

    def security_environment(self) -> Mapping[str, str | None]:
        values = self._native_environment()
        return {
            "PYTHONNOUSERSITE": values.get("PYTHONNOUSERSITE"),
            "PYTHONDONTWRITEBYTECODE": values.get("PYTHONDONTWRITEBYTECODE"),
            "PYTHONHOME": values.get("PYTHONHOME"),
            "PYTHONPATH": values.get("PYTHONPATH"),
            "PYTHONSTARTUP": values.get("PYTHONSTARTUP"),
        }

    def rederive_environment_inventory(
        self, frozen_inventory: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        checked = attempt_contract.validate_environment_inventory(frozen_inventory)
        variables = [
            {
                "name": name,
                "defined": True,
                "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
            for name, value in self._native_environment().items()
        ]
        variables.sort(
            key=lambda item: (item["name"].casefold(), item["name"].encode("utf-8"))
        )
        return attempt_contract.validate_environment_inventory(
            {
                "schema": "aigp-vq2-powered-environment-inventory/1",
                "created_at_utc": checked["created_at_utc"],
                "variables": variables,
            }
        )

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            common = ntpath.commonpath((path, root))
        except ValueError:
            return False
        return ntpath.normcase(common) == ntpath.normcase(root)

    def rederive_import_inventory(
        self,
        frozen_inventory: Mapping[str, Any],
        eager_modules: Sequence[str],
    ) -> ImportRevalidation:
        return self._rederive_import_inventory_hardened(
            frozen_inventory, eager_modules
        )


class WindowsSecureCreateNew:
    """Handle-retained, protected-DACL Windows evidence publisher.

    Every directory and file created by this boundary receives a protected
    one-ACE DACL in the creating kernel call.  Publication failures preserve
    possible partial bytes and are reported through ``BoundaryCreateNewError``;
    this class never repairs or replaces an existing target.
    """

    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_EXISTING = 3
    _CREATE_NEW = 1
    _READ_CONTROL = 0x00020000
    _FILE_READ_ATTRIBUTES = 0x00000080
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _OWNER_SECURITY_INFORMATION = 0x00000001
    _DACL_SECURITY_INFORMATION = 0x00000004
    _SE_DACL_PROTECTED = 0x1000
    _ACL_SIZE_INFORMATION = 2
    _ACCESS_ALLOWED_ACE_TYPE = 0
    _FILE_ALL_ACCESS = 0x001F01FF
    _ERROR_ALREADY_EXISTS = 183
    _ERROR_FILE_EXISTS = 80
    _MAX_PUBLICATION_BYTES = 64 * 1024 * 1024

    def __init__(self, *, monotonic_ns: Callable[[], int] | None = None) -> None:
        import ctypes
        import os
        import time
        from ctypes import wintypes

        if os.name != "nt":
            raise LiveIntegrationUnavailable(
                "secure powered evidence publication requires Windows"
            )
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._clock = time.perf_counter_ns if monotonic_ns is None else monotonic_ns
        if not callable(self._clock):
            raise TypeError("secure publisher monotonic clock must be callable")
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._advapi = ctypes.WinDLL("advapi32", use_last_error=True)

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            ]

        class FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", FILETIME),
                ("ftLastAccessTime", FILETIME),
                ("ftLastWriteTime", FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            ]

        class ACL_SIZE_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("AceCount", wintypes.DWORD),
                ("AclBytesInUse", wintypes.DWORD),
                ("AclBytesFree", wintypes.DWORD),
            ]

        class ACE_HEADER(ctypes.Structure):
            _fields_ = [
                ("AceType", ctypes.c_ubyte),
                ("AceFlags", ctypes.c_ubyte),
                ("AceSize", wintypes.WORD),
            ]

        self._SECURITY_ATTRIBUTES = SECURITY_ATTRIBUTES
        self._BY_HANDLE_FILE_INFORMATION = BY_HANDLE_FILE_INFORMATION
        self._ACL_SIZE_INFORMATION_STRUCT = ACL_SIZE_INFORMATION
        self._ACE_HEADER = ACE_HEADER
        self._INVALID_HANDLE = ctypes.c_void_p(-1).value

        kernel = self._kernel
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, wintypes.LPVOID]
        kernel.CreateDirectoryW.restype = wintypes.BOOL
        kernel.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(BY_HANDLE_FILE_INFORMATION),
        ]
        kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        kernel.WriteFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        kernel.WriteFile.restype = wintypes.BOOL
        kernel.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel.FlushFileBuffers.restype = wintypes.BOOL
        kernel.SetFilePointerEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.DWORD,
        ]
        kernel.SetFilePointerEx.restype = wintypes.BOOL
        kernel.ReadFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        kernel.ReadFile.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.LocalFree.argtypes = [wintypes.HLOCAL]
        kernel.LocalFree.restype = wintypes.HLOCAL
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        advapi = self._advapi
        advapi.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        advapi.OpenProcessToken.restype = wintypes.BOOL
        advapi.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_uint,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi.GetTokenInformation.restype = wintypes.BOOL
        advapi.ConvertSidToStringSidW.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
        advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
        advapi.GetSecurityInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
        ]
        advapi.GetSecurityInfo.restype = wintypes.DWORD
        advapi.GetAclInformation.argtypes = [
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.c_int,
        ]
        advapi.GetAclInformation.restype = wintypes.BOOL
        advapi.GetAce.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
        ]
        advapi.GetAce.restype = wintypes.BOOL
        advapi.GetSecurityDescriptorControl.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL

        self.current_user_id = self._current_user_sid()
        self._directory_handles: dict[str, int] = {}
        self._directory_receipts: dict[str, SecureDirectoryReceipt] = {}
        self._closed = False

    def _error(self, label: str) -> SecureBoundaryError:
        return SecureBoundaryError(
            f"{label} failed with Win32 error {int(self._ctypes.get_last_error())}"
        )

    def _close_handle(self, handle: int) -> None:
        if not self._kernel.CloseHandle(handle):
            raise self._error("CloseHandle")

    def _sid_string(self, sid: Any) -> str:
        pointer = self._wintypes.LPWSTR()
        if not self._advapi.ConvertSidToStringSidW(
            sid, self._ctypes.byref(pointer)
        ):
            raise self._error("ConvertSidToStringSidW")
        try:
            value = pointer.value
            if type(value) is not str or not value.startswith("S-"):
                raise SecureBoundaryError("Windows SID string is invalid")
            return value
        finally:
            if self._kernel.LocalFree(pointer):
                raise SecureBoundaryError("Windows SID allocation could not be freed")

    def _current_user_sid(self) -> str:
        token = self._wintypes.HANDLE()
        if not self._advapi.OpenProcessToken(
            self._kernel.GetCurrentProcess(), 0x0008, self._ctypes.byref(token)
        ):
            raise self._error("OpenProcessToken")
        try:
            required = self._wintypes.DWORD(0)
            self._advapi.GetTokenInformation(
                token, 1, None, 0, self._ctypes.byref(required)
            )
            if required.value == 0:
                raise self._error("GetTokenInformation(size)")
            buffer = self._ctypes.create_string_buffer(required.value)
            if not self._advapi.GetTokenInformation(
                token,
                1,
                buffer,
                required,
                self._ctypes.byref(required),
            ):
                raise self._error("GetTokenInformation")
            sid_pointer = self._ctypes.c_void_p.from_buffer(buffer).value
            if not sid_pointer:
                raise SecureBoundaryError("current-user SID is unavailable")
            return self._sid_string(sid_pointer)
        finally:
            self._close_handle(int(token.value))

    def _security_attributes(self) -> tuple[Any, Any]:
        descriptor = self._wintypes.LPVOID()
        size = self._wintypes.DWORD(0)
        sddl = f"D:P(A;;FA;;;{self.current_user_id})"
        if not self._advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, self._ctypes.byref(descriptor), self._ctypes.byref(size)
        ):
            raise self._error("ConvertStringSecurityDescriptor")
        attributes = self._SECURITY_ATTRIBUTES()
        attributes.nLength = self._ctypes.sizeof(attributes)
        attributes.lpSecurityDescriptor = descriptor
        attributes.bInheritHandle = False
        return descriptor, attributes

    def _free_descriptor(self, descriptor: Any) -> None:
        if descriptor and self._kernel.LocalFree(descriptor):
            raise SecureBoundaryError("security descriptor could not be freed")

    @staticmethod
    def _lexical(path: str) -> str:
        import os

        if type(path) is not str or not ntpath.isabs(path):
            raise SecureBoundaryError("secure path must be absolute")
        if ntpath.normpath(path) != path or os.path.abspath(path) != path:
            raise SecureBoundaryError("secure path must be lexically canonical")
        return path

    def _open_handle(self, path: str, *, directory: bool) -> int:
        flags = self._FILE_FLAG_OPEN_REPARSE_POINT
        if directory:
            flags |= self._FILE_FLAG_BACKUP_SEMANTICS
        handle = self._kernel.CreateFileW(
            path,
            self._FILE_READ_ATTRIBUTES | self._READ_CONTROL,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            None,
            self._OPEN_EXISTING,
            flags,
            None,
        )
        if handle == self._INVALID_HANDLE:
            raise self._error("CreateFileW(open secure path)")
        value = int(handle)
        try:
            self._file_information(value, directory=directory)
            if self._normcase(self._final_path(value)) != self._normcase(path):
                raise SecureBoundaryError("secure handle final path mismatched")
        except BaseException:
            self._close_handle(value)
            raise
        return value

    @staticmethod
    def _normcase(path: str) -> str:
        import os

        return os.path.normcase(os.path.abspath(path))

    def _final_path(self, handle: int) -> str:
        required = int(self._kernel.GetFinalPathNameByHandleW(handle, None, 0, 0))
        if required <= 0:
            raise self._error("GetFinalPathNameByHandleW(size)")
        buffer = self._ctypes.create_unicode_buffer(required + 1)
        length = int(
            self._kernel.GetFinalPathNameByHandleW(
                handle, buffer, len(buffer), 0
            )
        )
        if length <= 0 or length >= len(buffer):
            raise self._error("GetFinalPathNameByHandleW")
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            raise SecureBoundaryError("network evidence paths are forbidden")
        if value.startswith("\\\\?\\"):
            value = value[4:]
        return self._lexical(value)

    def _file_information(self, handle: int, *, directory: bool) -> Any:
        info = self._BY_HANDLE_FILE_INFORMATION()
        if not self._kernel.GetFileInformationByHandle(
            handle, self._ctypes.byref(info)
        ):
            raise self._error("GetFileInformationByHandle")
        attributes = int(info.dwFileAttributes)
        if bool(attributes & self._FILE_ATTRIBUTE_DIRECTORY) is not directory:
            raise SecureBoundaryError("secure handle has the wrong object kind")
        if attributes & self._FILE_ATTRIBUTE_REPARSE_POINT:
            raise SecureBoundaryError("secure handle names a reparse point")
        if not directory and int(info.nNumberOfLinks) != 1:
            raise SecureBoundaryError("secure file has an aliased hard-link identity")
        return info

    @staticmethod
    def _identity(info: Any) -> tuple[int, int, int]:
        return (
            int(info.dwVolumeSerialNumber),
            int(info.nFileIndexHigh),
            int(info.nFileIndexLow),
        )

    @staticmethod
    def _volume_id(info: Any) -> str:
        return f"volume-{int(info.dwVolumeSerialNumber):08x}"

    def _verify_private_acl(self, handle: int) -> tuple[str, bool]:
        owner = self._wintypes.LPVOID()
        dacl = self._wintypes.LPVOID()
        descriptor = self._wintypes.LPVOID()
        result = self._advapi.GetSecurityInfo(
            handle,
            1,
            self._OWNER_SECURITY_INFORMATION | self._DACL_SECURITY_INFORMATION,
            self._ctypes.byref(owner),
            None,
            self._ctypes.byref(dacl),
            None,
            self._ctypes.byref(descriptor),
        )
        if result != 0 or not owner.value or not dacl.value or not descriptor.value:
            raise SecureBoundaryError("Windows security information query failed")
        try:
            owner_id = self._sid_string(owner)
            control = self._wintypes.WORD(0)
            revision = self._wintypes.DWORD(0)
            if not self._advapi.GetSecurityDescriptorControl(
                descriptor,
                self._ctypes.byref(control),
                self._ctypes.byref(revision),
            ):
                raise self._error("GetSecurityDescriptorControl")
            if not (int(control.value) & self._SE_DACL_PROTECTED):
                return owner_id, False
            info = self._ACL_SIZE_INFORMATION_STRUCT()
            if not self._advapi.GetAclInformation(
                dacl,
                self._ctypes.byref(info),
                self._ctypes.sizeof(info),
                self._ACL_SIZE_INFORMATION,
            ):
                raise self._error("GetAclInformation")
            if int(info.AceCount) != 1:
                return owner_id, False
            ace = self._wintypes.LPVOID()
            if not self._advapi.GetAce(dacl, 0, self._ctypes.byref(ace)):
                raise self._error("GetAce")
            header = self._ctypes.cast(
                ace, self._ctypes.POINTER(self._ACE_HEADER)
            ).contents
            if (
                int(header.AceType) != self._ACCESS_ALLOWED_ACE_TYPE
                or int(header.AceSize) < 12
            ):
                return owner_id, False
            mask = self._ctypes.c_uint32.from_address(int(ace.value) + 4).value
            ace_sid = self._sid_string(int(ace.value) + 8)
            exact = (
                owner_id == self.current_user_id
                and ace_sid == self.current_user_id
                and int(mask) == self._FILE_ALL_ACCESS
            )
            return owner_id, exact
        finally:
            self._free_descriptor(descriptor)

    def _prove_ancestry(self, path: str) -> None:
        from pathlib import Path

        lexical = Path(self._lexical(path))
        current = Path(lexical.anchor)
        for component in lexical.parts[1:]:
            current = current / component
            handle = self._open_handle(str(current), directory=True)
            self._close_handle(handle)

    def _directory_receipt(
        self,
        path: str,
        parent_path: str,
        *,
        handle: int,
        created_new: bool,
    ) -> SecureDirectoryReceipt:
        info = self._file_information(handle, directory=True)
        owner_id, private = self._verify_private_acl(handle)
        parent_handle = self._open_handle(parent_path, directory=True)
        try:
            parent_info = self._file_information(parent_handle, directory=True)
            parent_final = self._final_path(parent_handle)
        finally:
            self._close_handle(parent_handle)
        if self._volume_id(info) != self._volume_id(parent_info):
            raise SecureBoundaryError("secure directory crosses a volume")
        self._prove_ancestry(parent_path)
        receipt = SecureDirectoryReceipt(
            path=path,
            final_path=self._final_path(handle),
            parent_final_path=parent_final,
            volume_id=self._volume_id(info),
            parent_volume_id=self._volume_id(parent_info),
            owner_id=owner_id,
            current_user_id=self.current_user_id,
            created_new=created_new,
            owner_is_current_user=owner_id == self.current_user_id,
            current_user_only_dacl=private,
            dacl_applied_at_create=True,
            non_reparse=True,
            ancestors_non_reparse=True,
            retained_handle=True,
        )
        _validate_secure_directory_receipt(
            receipt,
            expected_path=path,
            expected_parent=parent_path,
            require_created_new=created_new,
        )
        return receipt

    def _retain_directory(
        self, path: str, parent_path: str, *, handle: int, created_new: bool
    ) -> SecureDirectoryReceipt:
        if path in self._directory_handles:
            self._close_handle(handle)
            existing = self._directory_receipts[path]
            if existing.parent_final_path != parent_path:
                raise SecureBoundaryError("secure directory parent changed")
            return existing
        try:
            receipt = self._directory_receipt(
                path,
                parent_path,
                handle=handle,
                created_new=created_new,
            )
        except BaseException:
            self._close_handle(handle)
            raise
        self._directory_handles[path] = handle
        self._directory_receipts[path] = receipt
        return receipt

    def open_private_directory(
        self, path: str, *, parent_path: str
    ) -> SecureDirectoryReceipt:
        if self._closed:
            raise SecureBoundaryError("secure publisher is closed")
        target = self._lexical(path)
        parent = self._lexical(parent_path)
        if ntpath.dirname(target) != parent:
            raise SecureBoundaryError("secure directory parent is not exact")
        if target in self._directory_receipts:
            handle = self._directory_handles[target]
            receipt = self._directory_receipt(
                target, parent, handle=handle, created_new=False
            )
            self._directory_receipts[target] = receipt
            return receipt
        try:
            handle = self._open_handle(target, directory=True)
        except SecureBoundaryError as exc:
            raise BoundaryCreateNewError(target, state="absent", detail=str(exc)) from exc
        return self._retain_directory(
            target, parent, handle=handle, created_new=False
        )

    def _create_directory_native(self, path: str) -> int:
        descriptor, attributes = self._security_attributes()
        try:
            if not self._kernel.CreateDirectoryW(
                path, self._ctypes.byref(attributes)
            ):
                error = int(self._ctypes.get_last_error())
                if error in {self._ERROR_ALREADY_EXISTS, self._ERROR_FILE_EXISTS}:
                    raise FileExistsError(path)
                raise self._error("CreateDirectoryW")
        finally:
            self._free_descriptor(descriptor)
        return self._open_handle(path, directory=True)

    def create_private_directory_create_new(
        self, path: str, *, parent_path: str
    ) -> SecureDirectoryReceipt:
        if self._closed:
            raise SecureBoundaryError("secure publisher is closed")
        target = self._lexical(path)
        parent = self._lexical(parent_path)
        if ntpath.dirname(target) != parent or parent not in self._directory_handles:
            raise SecureBoundaryError("secure create-new parent is not retained")
        parent_handle = self._directory_handles[parent]
        self._file_information(parent_handle, directory=True)
        if self._normcase(self._final_path(parent_handle)) != self._normcase(parent):
            raise SecureBoundaryError("secure create-new parent moved")
        created = False
        handle: int | None = None
        try:
            handle = self._create_directory_native(target)
            created = True
            return self._retain_directory(
                target, parent, handle=handle, created_new=True
            )
        except FileExistsError as exc:
            raise BoundaryCreateNewError(
                target, state="unknown", detail="create-new directory already exists"
            ) from exc
        except BoundaryCreateNewError:
            raise
        except BaseException as exc:
            raise BoundaryCreateNewError(
                target,
                state="partial" if created else "unknown",
                detail="secure create-new directory proof failed",
            ) from exc

    def inspect_attempt_root(self, paths: Mapping[str, str]) -> AttemptRootSnapshot:
        import os
        import re
        from pathlib import Path

        root = paths.get("evidence_root")
        if type(root) is not str:
            raise SecureBoundaryError("frozen evidence root is unavailable")
        self.open_private_directory(root, parent_path=ntpath.dirname(root))
        try:
            names = tuple(entry.name for entry in os.scandir(root))
        except OSError as exc:
            raise SecureBoundaryError("evidence root cannot be enumerated") from exc
        target_name = ntpath.basename(paths["attempt_dir"])
        prior: list[PriorAttemptObservation] = []
        unknown: list[str] = []
        exact_attempt = re.compile(r"^F[0-9]{2}-A[0-9]{2}$")
        attempt_like = re.compile(r"^F[^\\/]*-A[^\\/]*$", re.IGNORECASE)
        for name in sorted(names, key=lambda item: item.encode("utf-8")):
            if name == target_name:
                continue
            if exact_attempt.fullmatch(name):
                attempt_path = Path(root) / name
                terminal_count = 0
                valid_count = 0
                for terminal_name, validator in (
                    ("attempt-complete.json", attempt_contract.validate_attempt_complete),
                    ("attempt-invalid.json", attempt_contract.validate_attempt_invalid),
                ):
                    terminal = attempt_path / terminal_name
                    if terminal.exists():
                        terminal_count += 1
                        try:
                            document = self.read_stable_json(str(terminal))
                            validator(document.value)
                        except (OSError, RuntimeError, ValueError):
                            pass
                        else:
                            valid_count += 1
                prior.append(
                    PriorAttemptObservation(name, terminal_count, valid_count)
                )
            elif attempt_like.fullmatch(name):
                unknown.append(name)
        return AttemptRootSnapshot(
            evidence_root=root,
            live_poison_present=os.path.lexists(paths["live_poison"]),
            target_attempt_directory_present=os.path.lexists(paths["attempt_dir"]),
            target_attempt_envelope_present=os.path.lexists(
                paths["attempt_envelope"]
            ),
            prior_attempts=tuple(prior),
            unknown_attempt_entries=tuple(unknown),
        )

    def _read_handle_bytes(self, handle: int, size: int) -> bytes:
        if not self._kernel.SetFilePointerEx(handle, 0, None, 0):
            raise self._error("SetFilePointerEx")
        output = bytearray()
        while len(output) < size:
            count = min(1024 * 1024, size - len(output))
            buffer = self._ctypes.create_string_buffer(count)
            read = self._wintypes.DWORD(0)
            if not self._kernel.ReadFile(
                handle,
                buffer,
                count,
                self._ctypes.byref(read),
                None,
            ):
                raise self._error("ReadFile")
            if read.value == 0:
                break
            output.extend(buffer.raw[: int(read.value)])
        return bytes(output)

    def create_inheritable_output_file(
        self,
        path: str,
        *,
        parent: SecureDirectoryReceipt,
        deadline_monotonic_ns: int,
    ) -> int:
        """Create one protected child-output file and return its owned handle.

        The caller must pass this exact handle in an explicit CreateProcess
        handle list and close the parent copy immediately after spawn.  Unlike
        canonical wrapper publications, the child supplies the eventual bytes,
        so no completed-file receipt is claimed here.
        """

        target = self._lexical(path)
        if (
            self._closed
            or ntpath.dirname(target) != parent.final_path
            or parent.final_path not in self._directory_handles
            or type(deadline_monotonic_ns) is not int
            or self._clock() >= deadline_monotonic_ns
        ):
            raise SecureBoundaryError("child-output create-new admission failed")
        descriptor, attributes = self._security_attributes()
        attributes.bInheritHandle = True
        handle: int | None = None
        try:
            raw = self._kernel.CreateFileW(
                target,
                self._GENERIC_READ
                | self._GENERIC_WRITE
                | self._READ_CONTROL
                | self._FILE_READ_ATTRIBUTES,
                self._FILE_SHARE_READ,
                self._ctypes.byref(attributes),
                self._CREATE_NEW,
                self._FILE_FLAG_OPEN_REPARSE_POINT,
                None,
            )
            if raw == self._INVALID_HANDLE:
                error = int(self._ctypes.get_last_error())
                state = (
                    "unknown"
                    if error in {self._ERROR_ALREADY_EXISTS, self._ERROR_FILE_EXISTS}
                    else "absent"
                )
                raise BoundaryCreateNewError(
                    target,
                    state=state,
                    detail=f"child-output CreateFileW failed ({error})",
                )
            handle = int(raw)
        finally:
            self._free_descriptor(descriptor)
        try:
            info = self._file_information(handle, directory=False)
            owner, private = self._verify_private_acl(handle)
            if (
                owner != self.current_user_id
                or not private
                or self._normcase(self._final_path(handle))
                != self._normcase(target)
                or self._volume_id(info) != parent.volume_id
                or self._clock() >= deadline_monotonic_ns
            ):
                raise SecureBoundaryError("child-output creation proof failed")
            return handle
        except BaseException as exc:
            try:
                self._close_handle(handle)
            except BaseException as close_exc:
                exc.add_note(
                    "child-output handle close also failed: "
                    f"{type(close_exc).__name__}: {close_exc}"
                )
            raise BoundaryCreateNewError(
                target,
                state="partial",
                detail="child-output create-new proof failed",
                observed_sha256=hashlib.sha256(b"").hexdigest(),
            ) from exc

    def create_new_file(
        self,
        path: str,
        payload: bytes,
        *,
        parent: SecureDirectoryReceipt,
        deadline_monotonic_ns: int,
    ) -> CreateNewFileReceipt:
        if self._closed:
            raise SecureBoundaryError("secure publisher is closed")
        target = self._lexical(path)
        if (
            type(payload) is not bytes
            or len(payload) > self._MAX_PUBLICATION_BYTES
            or type(deadline_monotonic_ns) is not int
            or deadline_monotonic_ns <= 0
        ):
            raise SecureBoundaryError("create-new publication inputs are invalid")
        if (
            ntpath.dirname(target) != parent.final_path
            or parent.final_path not in self._directory_handles
        ):
            raise SecureBoundaryError("create-new publication parent is not retained")
        if self._clock() >= deadline_monotonic_ns:
            raise BoundaryCreateNewError(
                target, state="absent", detail="publication deadline expired"
            )
        parent_handle = self._directory_handles[parent.final_path]
        parent_info_before = self._file_information(parent_handle, directory=True)
        owner_id, private = self._verify_private_acl(parent_handle)
        if owner_id != self.current_user_id or not private:
            raise SecureBoundaryError("publication parent is not current-user-only")
        descriptor, attributes = self._security_attributes()
        handle: int | None = None
        created = False
        observed_hash: str | None = None
        try:
            try:
                raw_handle = self._kernel.CreateFileW(
                    target,
                    self._GENERIC_READ
                    | self._GENERIC_WRITE
                    | self._READ_CONTROL
                    | self._FILE_READ_ATTRIBUTES,
                    self._FILE_SHARE_READ,
                    self._ctypes.byref(attributes),
                    self._CREATE_NEW,
                    self._FILE_FLAG_OPEN_REPARSE_POINT,
                    None,
                )
                if raw_handle == self._INVALID_HANDLE:
                    error = int(self._ctypes.get_last_error())
                    if error in {
                        self._ERROR_ALREADY_EXISTS,
                        self._ERROR_FILE_EXISTS,
                    }:
                        raise FileExistsError(target)
                    raise self._error("CreateFileW(create-new publication)")
                handle = int(raw_handle)
                created = True
            finally:
                self._free_descriptor(descriptor)
        except FileExistsError as exc:
            raise BoundaryCreateNewError(
                target, state="unknown", detail="publication target already exists"
            ) from exc
        except BaseException as exc:
            if handle is not None:
                try:
                    self._close_handle(handle)
                except BaseException as close_exc:
                    exc.add_note(f"created publication handle close failed: {close_exc}")
            raise BoundaryCreateNewError(
                target,
                state="partial" if created else "absent",
                detail="create-new publication construction failed",
            ) from exc
        try:
            created_info = self._file_information(handle, directory=False)
            if self._normcase(self._final_path(handle)) != self._normcase(target):
                raise SecureBoundaryError("created publication final path mismatched")
            file_owner, file_private = self._verify_private_acl(handle)
            if file_owner != self.current_user_id or not file_private:
                raise SecureBoundaryError("created publication DACL is not exact")
            buffer = self._ctypes.create_string_buffer(payload, len(payload))
            written = self._wintypes.DWORD(0)
            if payload and (
                not self._kernel.WriteFile(
                    handle,
                    buffer,
                    len(payload),
                    self._ctypes.byref(written),
                    None,
                )
                or int(written.value) != len(payload)
            ):
                raise self._error("WriteFile(publication)")
            if not self._kernel.FlushFileBuffers(handle):
                raise self._error("FlushFileBuffers(publication)")
            observed = self._read_handle_bytes(handle, len(payload))
            observed_hash = hashlib.sha256(observed).hexdigest()
            if observed != payload:
                raise SecureBoundaryError("publication readback bytes mismatched")
            parent_info_after = self._file_information(parent_handle, directory=True)
            if self._identity(parent_info_before) != self._identity(parent_info_after):
                raise SecureBoundaryError("publication parent identity changed")
            reopened = self._open_handle(target, directory=False)
            try:
                reopened_info = self._file_information(reopened, directory=False)
                if self._identity(reopened_info) != self._identity(created_info):
                    raise SecureBoundaryError("publication path identity changed")
            finally:
                self._close_handle(reopened)
            completed = self._clock()
            if completed >= deadline_monotonic_ns:
                raise SecureBoundaryError("publication completed after its deadline")
            receipt = CreateNewFileReceipt(
                path=target,
                final_path=self._final_path(handle),
                parent_final_path=parent.final_path,
                volume_id=self._volume_id(created_info),
                parent_volume_id=self._volume_id(parent_info_after),
                owner_id=file_owner,
                current_user_id=self.current_user_id,
                size_bytes=len(observed),
                sha256=observed_hash,
                completed_monotonic_ns=completed,
                created_new=True,
                regular_file=True,
                owner_is_current_user=True,
                current_user_only_dacl=True,
                dacl_applied_at_create=True,
                non_reparse=True,
                ancestors_non_reparse=True,
                flushed=True,
                readback_verified=True,
            )
            _validate_file_receipt(
                receipt,
                path=target,
                parent=parent,
                payload=payload,
                deadline_monotonic_ns=deadline_monotonic_ns,
            )
            return receipt
        except BaseException as exc:
            if created and observed_hash is None and handle is not None:
                try:
                    info = self._file_information(handle, directory=False)
                    size = (int(info.nFileSizeHigh) << 32) | int(info.nFileSizeLow)
                    if size <= self._MAX_PUBLICATION_BYTES:
                        observed_hash = hashlib.sha256(
                            self._read_handle_bytes(handle, size)
                        ).hexdigest()
                except BaseException:
                    observed_hash = None
            raise BoundaryCreateNewError(
                target,
                state="partial" if created else "unknown",
                detail="create-new publication proof failed",
                observed_sha256=observed_hash,
            ) from exc
        finally:
            if handle is not None:
                pending = sys.exc_info()[1]
                try:
                    self._close_handle(handle)
                except BaseException as close_exc:
                    if pending is None:
                        raise
                    pending.add_note(
                        "publication handle close also failed: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )

    def close(self) -> None:
        if self._closed:
            return
        failures: list[str] = []
        for path, handle in reversed(tuple(self._directory_handles.items())):
            try:
                self._close_handle(handle)
            except BaseException as exc:
                failures.append(f"{path}:{type(exc).__name__}")
        self._directory_handles.clear()
        self._directory_receipts.clear()
        self._closed = True
        if failures:
            raise SecureBoundaryError(
                "secure retained-directory close failures: " + ",".join(failures)
            )


class WindowsProductionLiveBoundary:
    """Concrete Windows wrapper services for the sole frozen L1 attempt.

    Construction is local and inert with respect to FlightSim, ports, and the
    production mutex.  Those effects occur only through the ordered methods
    invoked after offline admission and attempt consumption.
    """

    _TASK_NAME = "AIGP-P2-F00-A01-Launch"
    _POLL_NS = 50_000_000
    _HEARTBEAT_EMIT_NS = 900_000_000
    _MAX_STDOUT_BYTES = 16 * 1024 * 1024
    _MAX_STDERR_BYTES = 1 * 1024 * 1024

    def __init__(
        self,
        *,
        freeze: Mapping[str, Any],
        secure: WindowsSecureCreateNew,
        clock: QpcService,
    ) -> None:
        from scripts import aigp_vq2_powered_runtime as powered_runtime

        self.freeze = attempt_contract.validate_live_freeze(freeze)
        self.secure = secure
        self.clock = clock
        self.runtime = powered_runtime
        self.process_operations = powered_runtime.Win32ProcessOperations()
        self.udp_operations = powered_runtime.Win32UdpOwnerTableOperations()
        self.wrapper_identity: dict[str, Any] | None = None
        self.wrapper_argv: tuple[str, ...] | None = None
        self.retained_wrapper: Any = None
        self.child_pipe: Any = None
        self.cleanup_pipe: Any = None
        self.child_parent: Any = None
        self.cleanup_parent: Any = None
        self.handle_set: AttemptHandleSet | None = None
        self.child: Any = None
        self.fallback: Any = None
        self.tree_proofs: dict[int, Any] = {}
        self.launch_result: dict[str, Any] | None = None
        self.simulator_handles: dict[str, Any] = {}
        self.prechild_proof: dict[str, Any] | None = None
        self.postchild_proof: dict[str, Any] | None = None
        self.training_attestation: dict[str, Any] | None = None
        self.attempt_envelope: dict[str, Any] | None = None
        self.attempt_envelope_sha256: str | None = None
        self.active_owner_observations: list[dict[str, Any]] = []
        self.final_ports_contract: dict[str, Any] | None = None
        self.lease_store: Any = None
        self.powered_lease: Any = None
        self.last_release_index: dict[str, Any] | None = None
        self._spawn_anchors: dict[int, int] = {}
        self._output_paths: dict[int, tuple[str, str]] = {}
        self.process_authorities: dict[int, dict[str, Any]] = {}
        self.process_results: dict[int, dict[str, Any]] = {}
        self.cleanup_certificates: dict[int, dict[str, Any]] = {}
        self.stable_file_proofs: dict[str, dict[str, Any]] = {}
        self._stable_file_handles: dict[str, int] = {}
        self._stable_file_payloads: dict[str, bytes] = {}
        self._closed_process_ids: set[int] = set()
        self._lease_release_attempted = False
        self._closed = False

    def _now(self) -> int:
        value = self.clock.now_ns()
        if type(value) is not int or value < 0:
            raise OrchestrationPhaseError("internal_error", "QPC value is invalid")
        return value

    @staticmethod
    def _utc_now_value() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def utc_now(self) -> str:
        return self._utc_now_value()

    def token_bytes(self, size: int) -> bytes:
        import secrets

        if type(size) is not int or size <= 0:
            raise ValueError("CSPRNG size must be a positive exact integer")
        return secrets.token_bytes(size)

    def host_boot_id_sha256(self) -> str:
        import ctypes
        import struct
        import winreg
        from ctypes import wintypes

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            ) as key:
                machine_guid, value_type = winreg.QueryValueEx(key, "MachineGuid")
        except OSError as exc:
            raise OrchestrationPhaseError(
                "internal_error", "host MachineGuid query failed"
            ) from exc
        if value_type not in {winreg.REG_SZ, winreg.REG_EXPAND_SZ} or type(
            machine_guid
        ) is not str:
            raise OrchestrationPhaseError(
                "internal_error", "host MachineGuid has an invalid type"
            )
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtQuerySystemInformation.argtypes = [
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
        ]
        ntdll.NtQuerySystemInformation.restype = ctypes.c_long
        buffer = ctypes.create_string_buffer(64)
        returned = wintypes.ULONG(0)
        status = int(
            ntdll.NtQuerySystemInformation(
                3, buffer, len(buffer), ctypes.byref(returned)
            )
        )
        if status < 0 or returned.value < 8:
            raise OrchestrationPhaseError(
                "internal_error", "host boot FILETIME query failed"
            )
        boot_filetime = struct.unpack_from("<Q", buffer.raw, 0)[0]
        if boot_filetime <= 0:
            raise OrchestrationPhaseError(
                "internal_error", "host boot FILETIME is invalid"
            )
        return hashlib.sha256(
            machine_guid.upper().encode("utf-8")
            + b"\x00"
            + struct.pack("<Q", boot_filetime)
        ).hexdigest()

    def _sleep_poll(self, deadline: int) -> None:
        import time

        now = self._now()
        if now >= deadline:
            return
        time.sleep(min(self._POLL_NS, deadline - now) / 1_000_000_000.0)

    def _require_before_deadline(self, deadline: int, label: str) -> int:
        value = _require_positive_int(deadline, f"{label} deadline")
        now = self._now()
        if now >= value:
            raise OrchestrationPhaseError(
                "deadline_expired", f"{label} reached its absolute deadline"
            )
        return now

    def _load_canonical_document(
        self,
        path: str,
        *,
        expected_sha256: str | None,
        validator: Callable[[Any], Mapping[str, Any]],
        label: str,
        maximum_bytes: int = 16 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Boundedly reject mutable/noncanonical authority-side JSON bytes."""

        import os
        from pathlib import Path

        source = Path(path)
        try:
            before = source.stat(follow_symlinks=False)
            if source.is_symlink() or not source.is_file() or before.st_size > maximum_bytes:
                raise OrchestrationPhaseError(
                    "artifact_mismatch", f"{label} is not one bounded regular file"
                )
            payload = source.read_bytes()
            after = source.stat(follow_symlinks=False)
        except OrchestrationPhaseError:
            raise
        except OSError as exc:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} could not be read stably"
            ) from exc
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_state != after_state or len(payload) != before.st_size:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} changed while it was read"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} byte hash changed"
            )
        try:
            checked = dict(validator(attempt_contract.strict_json_loads(payload)))
        except attempt_contract.PoweredAttemptContractError as exc:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} failed strict validation"
            ) from exc
        if payload != attempt_contract.canonical_json_file_bytes(checked):
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} is not one canonical JSON file"
            )
        if os.name == "nt" and not ntpath.isabs(path):
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} path is not absolute"
            )
        return checked

    def _load_attempt_envelope(self) -> dict[str, Any]:
        checked = self._load_canonical_document(
            self.freeze["paths"]["attempt_envelope"],
            expected_sha256=self.attempt_envelope_sha256,
            validator=lambda value: attempt_contract.validate_attempt(
                value, live_freeze=self.freeze
            ),
            label="attempt envelope",
        )
        digest = attempt_contract.canonical_file_sha256(checked)
        if self.attempt_envelope is not None and checked != self.attempt_envelope:
            raise OrchestrationPhaseError(
                "artifact_mismatch", "attempt envelope changed after lease acquisition"
            )
        self.attempt_envelope = checked
        self.attempt_envelope_sha256 = digest
        return checked

    @staticmethod
    def _secure_file_state(info: Any) -> tuple[int, ...]:
        return (
            int(info.dwFileAttributes),
            int(info.dwVolumeSerialNumber),
            int(info.nFileSizeHigh),
            int(info.nFileSizeLow),
            int(info.nNumberOfLinks),
            int(info.nFileIndexHigh),
            int(info.nFileIndexLow),
            int(info.ftCreationTime.dwHighDateTime),
            int(info.ftCreationTime.dwLowDateTime),
            int(info.ftLastWriteTime.dwHighDateTime),
            int(info.ftLastWriteTime.dwLowDateTime),
        )

    def _read_retained_complete_file(
        self,
        path: str,
        *,
        maximum_bytes: int,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
        label: str,
    ) -> tuple[bytes, Mapping[str, Any]]:
        """Retain one immutable file identity while parsing downstream evidence."""

        maximum = _require_positive_int(maximum_bytes, f"{label} maximum bytes")
        self._require_before_deadline(deadline_monotonic_ns, f"{label} read")
        retained = self._stable_file_handles.get(path)
        if retained is not None:
            info = self.secure._file_information(retained, directory=False)
            proof = self.stable_file_proofs[path]
            if (
                self.secure._normcase(self.secure._final_path(retained))
                != self.secure._normcase(proof["final_path"])
                or self._secure_file_state(info) != tuple(proof["file_state"])
                or int(info.nNumberOfLinks) != 1
                or proof.get("hardlink_count_one") is not True
            ):
                raise OrchestrationPhaseError(
                    "artifact_mismatch", f"retained {label} identity changed"
                )
            self._require_before_deadline(
                deadline_monotonic_ns, f"retained {label} readback"
            )
            return self._stable_file_payloads[path], dict(proof)

        handle: int | None = None
        last_heartbeat = self._now()
        while handle is None:
            now = self._require_before_deadline(
                deadline_monotonic_ns, f"{label} open"
            )
            raw = self.secure._kernel.CreateFileW(
                path,
                self.secure._GENERIC_READ
                | self.secure._FILE_READ_ATTRIBUTES
                | self.secure._READ_CONTROL,
                self.secure._FILE_SHARE_READ,
                None,
                self.secure._OPEN_EXISTING,
                self.secure._FILE_FLAG_OPEN_REPARSE_POINT,
                None,
            )
            if raw != self.secure._INVALID_HANDLE:
                handle = int(raw)
                break
            error = int(self.secure._ctypes.get_last_error())
            if error != 32:  # ERROR_SHARING_VIOLATION: an inherited writer remains.
                raise OrchestrationPhaseError(
                    "artifact_mismatch", f"{label} is absent or inaccessible ({error})"
                )
            if now - last_heartbeat >= heartbeat.period_ns:
                heartbeat()
                last_heartbeat = self._now()
            self._sleep_poll(deadline_monotonic_ns)

        try:
            before = self.secure._file_information(handle, directory=False)
            final_path = self.secure._final_path(handle)
            owner_id, private_acl = self.secure._verify_private_acl(handle)
            parent = self._attempt_directory_receipt()
            if (
                self.secure._normcase(final_path) != self.secure._normcase(path)
                or self.secure._volume_id(before) != parent.volume_id
                or owner_id != parent.current_user_id
                or private_acl is not True
                or int(before.nNumberOfLinks) != 1
            ):
                raise OrchestrationPhaseError(
                    "artifact_mismatch",
                    f"{label} final path/ACL/volume/link identity is not exact",
                )
            size = (int(before.nFileSizeHigh) << 32) | int(before.nFileSizeLow)
            if size > maximum:
                raise OrchestrationPhaseError(
                    "artifact_mismatch", f"{label} exceeds its hard byte ceiling"
                )
            first = self.secure._read_handle_bytes(handle, size)
            middle = self.secure._file_information(handle, directory=False)
            second = self.secure._read_handle_bytes(handle, size)
            after = self.secure._file_information(handle, directory=False)
            state = self._secure_file_state(before)
            if (
                len(first) != size
                or first != second
                or state != self._secure_file_state(middle)
                or state != self._secure_file_state(after)
                or self._now() >= deadline_monotonic_ns
            ):
                raise OrchestrationPhaseError(
                    "artifact_mismatch", f"{label} was not one stable complete file"
                )
            proof = {
                "path": path,
                "final_path": final_path,
                "volume_id": self.secure._volume_id(before),
                "file_id": [
                    int(before.dwVolumeSerialNumber),
                    int(before.nFileIndexHigh),
                    int(before.nFileIndexLow),
                ],
                "file_state": list(state),
                "size_bytes": size,
                "sha256": hashlib.sha256(first).hexdigest(),
                "regular_file": True,
                "non_reparse": True,
                "hardlink_count_one": int(before.nNumberOfLinks) == 1,
                "owner_is_current_user": True,
                "current_user_only_dacl": True,
                "stable_before_after": True,
                "readback_twice_equal": True,
                "retained_handle": True,
            }
            self._stable_file_handles[path] = handle
            self._stable_file_payloads[path] = first
            self.stable_file_proofs[path] = proof
            handle = None
            return first, dict(proof)
        except OrchestrationPhaseError:
            raise
        except BaseException as exc:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} retained-file proof failed"
            ) from exc
        finally:
            if handle is not None:
                self.secure._close_handle(handle)

    def _native_environment_for_spawn(self) -> dict[str, str]:
        """Read the native Windows block used for the exact child environment."""

        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetEnvironmentStringsW.argtypes = []
        kernel.GetEnvironmentStringsW.restype = ctypes.c_void_p
        kernel.FreeEnvironmentStringsW.argtypes = [ctypes.c_void_p]
        kernel.FreeEnvironmentStringsW.restype = ctypes.c_int
        block = kernel.GetEnvironmentStringsW()
        if not block:
            raise OrchestrationPhaseError(
                "build_or_candidate_changed", "native environment is unavailable"
            )
        values: dict[str, str] = {}
        step = ctypes.sizeof(ctypes.c_wchar)
        offset = 0
        try:
            while True:
                raw = ctypes.wstring_at(block + offset * step)
                if not raw:
                    break
                offset += len(raw) + 1
                separator = raw.find("=", 1 if raw.startswith("=") else 0)
                if separator <= 0:
                    raise OrchestrationPhaseError(
                        "build_or_candidate_changed",
                        "native environment contains a malformed entry",
                    )
                name = raw[:separator].upper()
                if "=" in name or name in values:
                    raise OrchestrationPhaseError(
                        "build_or_candidate_changed",
                        "native environment names are not spawn-safe and unique",
                    )
                values[name] = raw[separator + 1 :]
        finally:
            if not kernel.FreeEnvironmentStringsW(block):
                raise OrchestrationPhaseError(
                    "build_or_candidate_changed",
                    "native environment block could not be released",
                )

        reference = self.freeze["runtime"]["environment_inventory"]
        inventory = self._load_canonical_document(
            reference["path"],
            expected_sha256=reference["sha256"],
            validator=attempt_contract.validate_environment_inventory,
            label="environment inventory",
        )
        observed = [
            {
                "name": name,
                "defined": True,
                "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
            for name, value in values.items()
        ]
        observed.sort(
            key=lambda item: (item["name"].casefold(), item["name"].encode("utf-8"))
        )
        if observed != inventory["variables"]:
            raise OrchestrationPhaseError(
                "build_or_candidate_changed", "child environment drifted from the freeze"
            )
        return values

    @staticmethod
    def _spawn_environment_sha256(environment: Mapping[str, str]) -> str:
        """Derive the canonical inventory-variable digest for one spawn map."""

        if type(environment) is not dict:
            raise OrchestrationPhaseError(
                "build_or_candidate_changed",
                "native spawn environment is not an exact mapping",
            )
        variables: list[dict[str, Any]] = []
        for name, value in environment.items():
            if type(name) is not str or type(value) is not str:
                raise OrchestrationPhaseError(
                    "build_or_candidate_changed",
                    "native spawn environment contains a non-string entry",
                )
            variables.append(
                {
                    "name": name,
                    "defined": True,
                    "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                }
            )
        variables.sort(
            key=lambda item: (item["name"].casefold(), item["name"].encode("utf-8"))
        )
        semantic_inventory = {
            "schema": "aigp-vq2-powered-environment-inventory/1",
            # Provenance is excluded from the semantic digest.  This fixed valid
            # value lets the shared inventory validator check the derived rows.
            "created_at_utc": "1970-01-01T00:00:00.000000Z",
            "variables": variables,
        }
        try:
            return attempt_contract.environment_variables_sha256(semantic_inventory)
        except attempt_contract.PoweredAttemptContractError as exc:
            raise OrchestrationPhaseError(
                "build_or_candidate_changed",
                "native spawn environment is not canonical",
            ) from exc

    def _attempt_directory_receipt(self) -> SecureDirectoryReceipt:
        receipt = self.secure.open_private_directory(
            self.freeze["paths"]["attempt_dir"],
            parent_path=self.freeze["paths"]["evidence_root"],
        )
        _validate_secure_directory_receipt(
            receipt,
            expected_path=self.freeze["paths"]["attempt_dir"],
            expected_parent=self.freeze["paths"]["evidence_root"],
            require_created_new=False,
        )
        return receipt

    def _open_readonly_inheritable_nul(self) -> tuple[int, int]:
        import msvcrt
        import os

        descriptor = os.open(os.devnull, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            handle = int(msvcrt.get_osfhandle(descriptor))
            os.set_handle_inheritable(handle, True)
            if self.process_operations.handle_is_inheritable(handle) is not True:
                raise OrchestrationPhaseError(
                    "child_spawn_failed", "read-only NUL handle is not inheritable"
                )
            return descriptor, handle
        except BaseException:
            os.close(descriptor)
            raise

    def _wait_job_tree(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Any:
        last_heartbeat = [self._now()]

        def wait_ns(duration_ns: int) -> None:
            import time

            now = self._now()
            remaining = max(0, deadline_monotonic_ns - now)
            time.sleep(min(duration_ns, self._POLL_NS, remaining) / 1_000_000_000.0)
            now = self._now()
            if now - last_heartbeat[0] >= heartbeat.period_ns:
                heartbeat()
                last_heartbeat[0] = self._now()

        return self.runtime.wait_job_process_tree_exit(
            child,
            deadline_monotonic_ns=deadline_monotonic_ns,
            monotonic_ns=self.clock.now_ns,
            wait_ns=wait_ns,
            max_poll_interval_ns=self._POLL_NS,
        )

    def _wait_subprocess(
        self,
        process: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> int:
        last_heartbeat = self._now()
        while True:
            code = process.poll()
            if code is not None:
                return int(code)
            now = self._now()
            if now >= deadline_monotonic_ns:
                try:
                    process.kill()
                finally:
                    process.wait(timeout=1.0)
                raise OrchestrationPhaseError(
                    "deadline_expired", "bounded subprocess exceeded its deadline"
                )
            if now - last_heartbeat >= self._HEARTBEAT_EMIT_NS:
                heartbeat()
                last_heartbeat = self._now()
            self._sleep_poll(deadline_monotonic_ns)

    def _query_task_absent(
        self,
        phase: str,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> dict[str, Any]:
        import os
        import subprocess

        executable = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"),
            "System32",
            "schtasks.exe",
        )
        process = subprocess.Popen(
            [executable, "/Query", "/TN", self._TASK_NAME],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.freeze["execution"]["wrapper_cwd"],
            env=dict(os.environ),
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        code = self._wait_subprocess(
            process,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        observation = {
            "phase": phase,
            "observed_monotonic_ns": self._now(),
            "query_exit_code": code,
            "absent": code == 1,
        }
        if code != 1:
            raise OrchestrationPhaseError(
                "topology_failed", "frozen scheduled-task name is not absent"
            )
        return observation

    def _toolhelp_rows(self) -> tuple[tuple[int, int, str], ...]:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel.Process32FirstW.restype = wintypes.BOOL
        kernel.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel.Process32NextW.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
        if handle == ctypes.c_void_p(-1).value:
            raise OrchestrationPhaseError(
                "topology_failed", "process enumeration snapshot failed"
            )
        rows: list[tuple[int, int, str]] = []
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            if not kernel.Process32FirstW(handle, ctypes.byref(entry)):
                raise OrchestrationPhaseError(
                    "topology_failed", "process enumeration first row failed"
                )
            while True:
                rows.append(
                    (
                        int(entry.th32ProcessID),
                        int(entry.th32ParentProcessID),
                        str(entry.szExeFile),
                    )
                )
                entry.dwSize = ctypes.sizeof(entry)
                if not kernel.Process32NextW(handle, ctypes.byref(entry)):
                    error = int(ctypes.get_last_error())
                    if error == 18:
                        break
                    raise OrchestrationPhaseError(
                        "topology_failed", "process enumeration next row failed"
                    )
        finally:
            if not kernel.CloseHandle(handle):
                raise OrchestrationPhaseError(
                    "topology_failed", "process enumeration handle close failed"
                )
        return tuple(rows)

    def _enumerate_simulator(self) -> dict[str, Any]:
        import ntpath as windows_path

        expected = {
            "launcher": self.freeze["simulator"]["launcher"],
            "payload": self.freeze["simulator"]["payload"],
        }
        basenames = {
            windows_path.basename(value["path"]).casefold(): role
            for role, value in expected.items()
        }
        found: dict[str, list[dict[str, Any]]] = {"launcher": [], "payload": []}
        opened: list[Any] = []
        try:
            for pid, parent_pid, image_name in self._toolhelp_rows():
                role = basenames.get(image_name.casefold())
                if role is None or pid <= 0:
                    continue
                provisional = self.process_operations.open_process(
                    pid, inheritable=False, terminate_access=False
                )
                try:
                    argv = self.process_operations.query_process_argv(provisional)
                    identity = self.process_operations.query_process_identity(
                        provisional, argv
                    )
                    retained = self.runtime.RetainedProcessHandle(
                        provisional,
                        identity,
                        argv,
                        operations=self.process_operations,
                    )
                except BaseException:
                    self.process_operations.close_handle(provisional)
                    raise
                opened.append(retained)
                if (
                    windows_path.normcase(identity["image_path"])
                    != windows_path.normcase(expected[role]["path"])
                    or identity["image_sha256"] != expected[role]["sha256"]
                ):
                    raise OrchestrationPhaseError(
                        "topology_failed",
                        f"foreign process conflicts with frozen {role} image name",
                    )
                found[role].append(
                    {
                        "parent_pid": parent_pid,
                        "retained": retained,
                        "identity": identity,
                    }
                )
            if any(len(found[role]) > 1 for role in found):
                raise OrchestrationPhaseError(
                    "topology_failed", "simulator process multiplicity is not exact"
                )
            if bool(found["launcher"]) != bool(found["payload"]):
                raise OrchestrationPhaseError(
                    "topology_failed", "simulator topology is partially present"
                )
            if found["launcher"]:
                launcher = found["launcher"][0]
                payload = found["payload"][0]
                if (
                    payload["parent_pid"] != launcher["identity"]["pid"]
                    or payload["identity"]["windows_session_id"]
                    != launcher["identity"]["windows_session_id"]
                    or launcher["retained"].alive() is not True
                    or payload["retained"].alive() is not True
                ):
                    raise OrchestrationPhaseError(
                        "topology_failed", "simulator parent/session/liveness is invalid"
                    )
            return {
                "launcher": found["launcher"][0] if found["launcher"] else None,
                "payload": found["payload"][0] if found["payload"] else None,
            }
        except BaseException:
            for retained in reversed(opened):
                try:
                    retained.close()
                except BaseException:
                    pass
            raise

    @staticmethod
    def _close_simulator_snapshot(snapshot: Mapping[str, Any]) -> None:
        for role in ("payload", "launcher"):
            item = snapshot.get(role)
            if item is not None:
                item["retained"].close()

    def _window_proof(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import ctypes
        from ctypes import wintypes

        user = ctypes.WinDLL("user32", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        user.EnumWindows.restype = wintypes.BOOL
        user.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        user.GetWindowThreadProcessId.restype = wintypes.DWORD
        user.IsWindow.argtypes = [wintypes.HWND]
        user.IsWindow.restype = wintypes.BOOL
        user.IsWindowVisible.argtypes = [wintypes.HWND]
        user.IsWindowVisible.restype = wintypes.BOOL
        user.IsIconic.argtypes = [wintypes.HWND]
        user.IsIconic.restype = wintypes.BOOL
        user.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        user.GetWindow.restype = wintypes.HWND
        user.SendMessageTimeoutW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        user.SendMessageTimeoutW.restype = wintypes.LPARAM
        target_pid = payload["identity"]["pid"]
        candidates: list[int] = []

        def visit(hwnd: Any, _parameter: Any) -> bool:
            owner = wintypes.DWORD(0)
            user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if (
                int(owner.value) == target_pid
                and user.IsWindow(hwnd)
                and user.IsWindowVisible(hwnd)
                and not user.IsIconic(hwnd)
                and not user.GetWindow(hwnd, 4)
            ):
                candidates.append(int(hwnd))
            return True

        callback = callback_type(visit)
        if not user.EnumWindows(callback, 0):
            raise OrchestrationPhaseError(
                "topology_failed", "simulator window enumeration failed"
            )
        if len(candidates) != 1:
            raise OrchestrationPhaseError(
                "topology_failed", "simulator top-level window is not unique"
            )
        hwnd = candidates[0]
        result = ctypes.c_size_t(0)
        responsive = bool(
            user.SendMessageTimeoutW(
                hwnd, 0, 0, 0, 0x0002, 250, ctypes.byref(result)
            )
        )
        if not responsive:
            raise OrchestrationPhaseError(
                "topology_failed", "simulator window is not responsive"
            )
        return {
            "hwnd": hwnd,
            "owner_pid": target_pid,
            "visible": True,
            "unminimized": True,
            "responsive": True,
        }

    def launch_and_wait(
        self,
        *,
        freeze: Mapping[str, Any],
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Any:
        import subprocess

        if attempt_contract.validate_live_freeze(freeze) != self.freeze:
            raise OrchestrationPhaseError("launch_failed", "live freeze changed")
        environment = self._native_environment_for_spawn()
        if self._spawn_environment_sha256(environment) != self.freeze["execution"][
            "launcher_environment_sha256"
        ]:
            raise OrchestrationPhaseError(
                "build_or_candidate_changed",
                "launcher environment drifted from the freeze",
            )
        before_task = self._query_task_absent(
            "before_launch",
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        before_time = self._now()
        prelaunch = self._enumerate_simulator()
        preexisting = prelaunch["launcher"] is not None
        if preexisting:
            self._window_proof(prelaunch["payload"])
        argv = list(self.freeze["execution"]["launcher_argv"])
        process = subprocess.Popen(
            argv,
            cwd=self.freeze["execution"]["launcher_cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            exit_code = self._wait_subprocess(
                process,
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
            )
        except BaseException:
            self._close_simulator_snapshot(prelaunch)
            raise
        returned = self._now()
        if exit_code != 0 or returned >= deadline_monotonic_ns:
            self._close_simulator_snapshot(prelaunch)
            raise OrchestrationPhaseError(
                "launch_failed", f"frozen launcher exited {exit_code}"
            )
        after_task = self._query_task_absent(
            "after_launcher_return",
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        result = {
            "launch": {
                "disposition": (
                    "preexisting_exact_topology"
                    if preexisting
                    else "absent_before_launcher_current_after"
                ),
                "observed_before_launch_monotonic_ns": before_time,
                "launcher_return_monotonic_ns": returned,
                "launcher_exit_code": 0,
                "prelaunch_launcher_process": (
                    prelaunch["launcher"]["identity"] if preexisting else None
                ),
                "prelaunch_payload_process": (
                    prelaunch["payload"]["identity"] if preexisting else None
                ),
            },
            "task_observations": [before_task, after_task],
            "prelaunch": prelaunch,
        }
        self.launch_result = result
        return result

    def _stable_owner_snapshots(
        self,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
        required: int = 2,
    ) -> tuple[Any, ...]:
        stable: list[Any] = []
        last_heartbeat = self._now()
        while True:
            now = self._now()
            if now >= deadline_monotonic_ns:
                raise OrchestrationPhaseError(
                    "ports_busy", "UDP owner-table proof reached its deadline"
                )
            current = self.runtime.capture_udp_owner_snapshot(
                (14550, 5600),
                operations=self.udp_operations,
                monotonic_ns=self.clock.now_ns,
            )
            if stable and current.ownership_key() != stable[-1].ownership_key():
                stable.clear()
            stable.append(current)
            if len(stable) >= required:
                return tuple(stable[-required:])
            now = self._now()
            if now - last_heartbeat >= self._HEARTBEAT_EMIT_NS:
                heartbeat()
                last_heartbeat = self._now()
            self._sleep_poll(deadline_monotonic_ns)

    @staticmethod
    def _ports_empty(snapshot: Any) -> bool:
        return all(
            not snapshot.owner_pids(family, port)
            for family in (2, 23)
            for port in (14550, 5600)
        )

    def _exclusive_probes(self, deadline: int) -> list[dict[str, Any]]:
        return [
            self.runtime.probe_exclusive_udp_port(
                host,
                port,
                deadline_monotonic_ns=deadline,
                monotonic_ns=self.clock.now_ns,
            ).to_primitive()
            for host, port in (("127.0.0.1", 14550), ("0.0.0.0", 5600))
        ]

    def _free_port_contract(
        self,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
        postchild: bool,
    ) -> dict[str, Any]:
        first = self._stable_owner_snapshots(
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        if any(not self._ports_empty(item) for item in first):
            raise OrchestrationPhaseError(
                "ports_busy" if not postchild else "port_residue",
                "production receive ports are not free",
            )
        probes = self._exclusive_probes(deadline_monotonic_ns)
        observations = list(first)
        if postchild:
            heartbeat()
            third = self._stable_owner_snapshots(
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
            )[-1]
            if not self._ports_empty(third):
                raise OrchestrationPhaseError(
                    "port_residue", "receive port changed after exclusive probes"
                )
            observations.append(third)
        return {
            "owner_table_observations": [
                item.to_contract_observation() for item in observations
            ],
            "active_owner_observations": list(self.active_owner_observations),
            "exclusive_probes": probes,
            "status": "free",
        }

    def _adopt_simulator_snapshot(
        self, current: Mapping[str, Any], prelaunch: Mapping[str, Any]
    ) -> None:
        if current["launcher"] is None or current["payload"] is None:
            raise OrchestrationPhaseError(
                "topology_failed", "simulator topology is absent after launcher return"
            )
        if prelaunch["launcher"] is not None:
            for role in ("launcher", "payload"):
                if current[role]["identity"] != prelaunch[role]["identity"]:
                    self._close_simulator_snapshot(current)
                    raise OrchestrationPhaseError(
                        "topology_failed", "preexisting simulator identity changed"
                    )
            self._close_simulator_snapshot(current)
            self.simulator_handles = {
                role: prelaunch[role]["retained"] for role in ("launcher", "payload")
            }
        else:
            self.simulator_handles = {
                role: current[role]["retained"] for role in ("launcher", "payload")
            }

    def prove_topology(
        self,
        *,
        launch_result: Any,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]:
        if launch_result is not self.launch_result or self.launch_result is None:
            raise OrchestrationPhaseError("topology_failed", "launch result is not exact")
        current = self._enumerate_simulator()
        self._adopt_simulator_snapshot(current, launch_result["prelaunch"])
        launcher = self.simulator_handles["launcher"].reprove()
        payload = self.simulator_handles["payload"].reprove()
        if (
            self.wrapper_identity is None
            or payload["windows_session_id"]
            != self.wrapper_identity["windows_session_id"]
        ):
            raise OrchestrationPhaseError(
                "topology_failed", "wrapper and simulator are not in one session"
            )
        window = self._window_proof(
            {"identity": payload, "retained": self.simulator_handles["payload"]}
        )
        before_child = self._query_task_absent(
            "before_child",
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        ports = self._free_port_contract(
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
            postchild=False,
        )
        proof = {
            "schema": "aigp-vq2-simulator-process-proof/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "phase": "prechild",
            "observed_at_utc": self.utc_now(),
            "observed_monotonic_ns": self._now(),
            "host_clock_id": attempt_contract.HOST_CLOCK_ID,
            "wrapper_process": dict(self.wrapper_identity),
            "launch": dict(launch_result["launch"]),
            "launcher_process": launcher,
            "payload_process": payload,
            "window": window,
            "build": 3385,
            "topology": "one_launcher_parent_retained_one_payload_child",
            "scheduled_task": {
                "name": self._TASK_NAME,
                "observations": [
                    *launch_result["task_observations"],
                    before_child,
                ],
            },
            "ports": ports,
            "responsive": True,
        }
        self.prechild_proof = attempt_contract.validate_simulator_process_proof(
            proof
        )
        return dict(self.prechild_proof)

    def _read_training_response(
        self,
        *,
        challenge: str,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> str:
        """Read one attached-console response without an unbounded input call."""

        import ctypes
        import msvcrt
        import os
        from ctypes import wintypes

        if os.name != "nt" or not sys.stdin.isatty() or not sys.stdout.isatty():
            raise OrchestrationPhaseError(
                "training_unattested", "Training attestation requires an attached console"
            )
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleWindow.argtypes = []
        kernel.GetConsoleWindow.restype = wintypes.HWND
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetConsoleMode.restype = wintypes.BOOL
        stdin_handle = kernel.GetStdHandle(wintypes.DWORD(-10 & 0xFFFFFFFF))
        mode = wintypes.DWORD(0)
        if (
            not kernel.GetConsoleWindow()
            or not stdin_handle
            or stdin_handle == ctypes.c_void_p(-1).value
            or not kernel.GetConsoleMode(stdin_handle, ctypes.byref(mode))
        ):
            raise OrchestrationPhaseError(
                "training_unattested", "Training attestation console is not local/attached"
            )

        expected = f"TRAINING {challenge}"
        print(
            "Visually verify the proved FlightSim window is in Training mode, "
            f"then enter {expected}",
            flush=True,
        )
        response: list[str] = []
        last_heartbeat = self._now()
        while True:
            now = self._now()
            if now >= deadline_monotonic_ns:
                raise OrchestrationPhaseError(
                    "training_unattested", "Training attestation reached its deadline"
                )
            while msvcrt.kbhit():
                character = msvcrt.getwch()
                if character == "\r":
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(response)
                if character == "\x03":
                    raise OrchestrationPhaseError(
                        "training_unattested", "Training attestation was cancelled"
                    )
                if character == "\b":
                    if response:
                        response.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                if character in {"\x00", "\xe0"}:
                    # Consume the scan-code half of an extended key and reject it.
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    raise OrchestrationPhaseError(
                        "training_unattested", "Training attestation contains a special key"
                    )
                if not (" " <= character <= "~") or len(response) >= len(expected):
                    raise OrchestrationPhaseError(
                        "training_unattested", "Training attestation response is not exact"
                    )
                response.append(character)
                sys.stdout.write(character)
                sys.stdout.flush()
            now = self._now()
            if now - last_heartbeat >= heartbeat.period_ns:
                heartbeat()
                last_heartbeat = self._now()
            self._sleep_poll(deadline_monotonic_ns)

    def attest_training(
        self,
        *,
        topology_proof: Mapping[str, Any],
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]:
        import hmac
        import secrets

        checked_proof = attempt_contract.validate_simulator_process_proof(
            topology_proof
        )
        if (
            self.prechild_proof is None
            or checked_proof != self.prechild_proof
            or checked_proof["phase"] != "prechild"
        ):
            raise OrchestrationPhaseError(
                "training_unattested", "Training attestation did not bind the exact topology"
            )
        self._require_before_deadline(
            deadline_monotonic_ns, "Training attestation"
        )
        challenge = secrets.token_hex(16)
        if re.fullmatch(r"[0-9a-f]{32}", challenge) is None:
            raise OrchestrationPhaseError(
                "internal_error", "Training challenge generator returned invalid bytes"
            )
        response = self._read_training_response(
            challenge=challenge,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        expected = f"TRAINING {challenge}"
        if not hmac.compare_digest(response.encode("ascii", errors="strict"), expected.encode("ascii")):
            raise OrchestrationPhaseError(
                "training_unattested", "Training attestation response did not match exactly"
            )
        observed = self._require_before_deadline(
            deadline_monotonic_ns, "Training attestation completion"
        )
        value = attempt_contract.validate_training_attestation(
            {
                "schema": "aigp-vq2-training-mode-attestation/1",
                "task_id": attempt_contract.TASK_ID,
                "session_id": attempt_contract.SESSION_ID,
                "attempt_id": attempt_contract.ATTEMPT_ID,
                "attested_at_utc": self.utc_now(),
                "attested_monotonic_ns": observed,
                "host_clock_id": attempt_contract.HOST_CLOCK_ID,
                "mode": "Training",
                "method": "post_topology_visual_training_check_challenge",
                "challenge_sha256": hashlib.sha256(
                    challenge.encode("ascii")
                ).hexdigest(),
                "wrapper_process": dict(checked_proof["wrapper_process"]),
                "simulator_process_proof_sha256": attempt_contract.canonical_file_sha256(
                    checked_proof
                ),
            },
            process_proof=checked_proof,
        )
        self.training_attestation = value
        return dict(value)

    def current_wrapper_identity(self) -> Mapping[str, Any]:
        retained = self.runtime.retain_current_process(
            operations=self.process_operations
        )
        try:
            self.wrapper_argv = tuple(retained.expected_argv)
            self.wrapper_identity = retained.identity
            return dict(self.wrapper_identity)
        finally:
            retained.close()

    def retain_and_reprove(self, identity: Mapping[str, Any]) -> Any:
        if self.wrapper_argv is None or dict(identity) != self.wrapper_identity:
            raise OrchestrationPhaseError(
                "internal_error", "wrapper identity retention input changed"
            )
        retained = self.runtime.retain_process(
            identity["pid"],
            self.wrapper_argv,
            operations=self.process_operations,
        )
        if retained.reprove() != dict(identity):
            retained.close()
            raise OrchestrationPhaseError(
                "internal_error", "wrapper retained-handle identity mismatched"
            )
        self.retained_wrapper = retained
        return retained

    def allocate_attempt_handles(
        self, wrapper_process: Mapping[str, Any]
    ) -> AttemptHandleSet:
        checked_wrapper = attempt_contract.validate_process_identity(wrapper_process)
        if (
            self.handle_set is not None
            or self.wrapper_argv is None
            or self.wrapper_identity is None
            or checked_wrapper != self.wrapper_identity
            or self.retained_wrapper is None
            or self.retained_wrapper.alive() is not True
        ):
            raise OrchestrationPhaseError(
                "internal_error", "attempt handles lack exact retained wrapper authority"
            )
        child_pipe = cleanup_pipe = child_parent = cleanup_parent = None
        try:
            child_pipe = self.runtime.create_capability_pipe(
                operations=self.process_operations
            )
            cleanup_pipe = self.runtime.create_capability_pipe(
                operations=self.process_operations
            )
            child_parent = self.runtime.retain_process(
                checked_wrapper["pid"],
                self.wrapper_argv,
                inheritable=True,
                operations=self.process_operations,
            )
            cleanup_parent = self.runtime.retain_process(
                checked_wrapper["pid"],
                self.wrapper_argv,
                inheritable=True,
                operations=self.process_operations,
            )
            values = (
                child_pipe.read_handle,
                child_pipe.write_handle,
                cleanup_pipe.read_handle,
                cleanup_pipe.write_handle,
                child_parent.handle_value,
                cleanup_parent.handle_value,
            )
            if len(set(values)) != len(values):
                raise OrchestrationPhaseError(
                    "internal_error", "capability and liveness handles are not distinct"
                )
            handles = AttemptHandleSet(
                child_pipe.read_handle,
                child_parent.handle_value,
                cleanup_pipe.read_handle,
                cleanup_parent.handle_value,
            )
            _validate_handle_set(handles)
        except BaseException as exc:
            failures: list[str] = []
            for name, value, close in (
                ("cleanup_parent", cleanup_parent, lambda item: item.close()),
                ("child_parent", child_parent, lambda item: item.close()),
                ("cleanup_pipe", cleanup_pipe, lambda item: item.abort()),
                ("child_pipe", child_pipe, lambda item: item.abort()),
            ):
                if value is None:
                    continue
                try:
                    close(value)
                except BaseException as close_exc:
                    failures.append(f"{name}:{type(close_exc).__name__}")
            if failures:
                exc.add_note("attempt-handle cleanup failures: " + ",".join(failures))
            if isinstance(exc, OrchestrationPhaseError):
                raise
            raise OrchestrationPhaseError(
                "internal_error", "attempt handle allocation failed"
            ) from exc
        self.child_pipe = child_pipe
        self.cleanup_pipe = cleanup_pipe
        self.child_parent = child_parent
        self.cleanup_parent = cleanup_parent
        self.handle_set = handles
        return handles

    def acquire(
        self,
        *,
        owner_secret: bytes,
        qpc_frequency_hz: int,
        deadline_monotonic_ns: int,
    ) -> Any:
        import hmac
        from pathlib import Path
        from scripts import aigp_live_lease

        now = self._require_before_deadline(deadline_monotonic_ns, "lease acquisition")
        if self.powered_lease is not None or self.lease_store is not None:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "powered lease acquisition is single-use"
            )
        if not isinstance(owner_secret, (bytes, bytearray, memoryview)) or len(owner_secret) != 32:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "powered lease owner capability is malformed"
            )
        attempt = self._load_attempt_envelope()
        expected = attempt_contract.derive_capability_sha256(
            _CAPABILITY_DOMAINS["lease_owner"],
            attempt["context_sha256"],
            owner_secret,
        )
        if not hmac.compare_digest(
            expected, attempt["capabilities"]["lease_owner_sha256"]
        ):
            raise OrchestrationPhaseError(
                "lease_unverifiable", "powered lease owner capability did not match"
            )
        frequency = _require_positive_int(qpc_frequency_hz, "lease QPC frequency")
        if (
            self.wrapper_identity is None
            or attempt["context"]["wrapper_process"] != self.wrapper_identity
            or attempt["context"]["host"]["qpc_frequency_hz"] != frequency
        ):
            raise OrchestrationPhaseError(
                "lease_unverifiable", "lease owner/process/QPC context changed"
            )
        parent = self._attempt_directory_receipt()
        try:
            directory = self.secure.create_private_directory_create_new(
                self.freeze["paths"]["lease_directory"],
                parent_path=parent.final_path,
            )
            _validate_secure_directory_receipt(
                directory,
                expected_path=self.freeze["paths"]["lease_directory"],
                expected_parent=parent.final_path,
            )
            store = aigp_live_lease.PoweredLeaseLedgerStore(
                Path(directory.final_path),
                Path(self.freeze["paths"]["lease_final"]),
                task_id=attempt_contract.TASK_ID,
                session_id=attempt_contract.SESSION_ID,
                attempt_id=attempt_contract.ATTEMPT_ID,
                attempt_envelope_sha256=self.attempt_envelope_sha256,
                attempt_context_sha256=attempt["context_sha256"],
                wrapper_process=self.wrapper_identity,
                qpc_frequency_hz=frequency,
                publish_final_index=False,
                _clock_ns=self.clock.now_ns,
            )
            remaining_ms = max(
                0, min(5_000, (deadline_monotonic_ns - now) // 1_000_000)
            )
            lease = aigp_live_lease.PoweredLiveSimulatorLease(
                store,
                owner_role="wrapper",
                owner_token_sha256=expected,
                owner_process=self.wrapper_identity,
                initial_phase="lease_acquire",
                wait_timeout_ms=remaining_ms,
                _clock_ns=self.clock.now_ns,
            )
            self.lease_store = store
            self.powered_lease = lease
            lease.acquire()
            # Surface every successfully acquired lease to the orchestrator.
            # Its immediate cooperative heartbeat detects a late return and
            # routes the known-owned lease through the normal recorded release
            # recovery.  Raw release here would discard that evidence path.
            return lease
        except OrchestrationPhaseError:
            raise
        except BaseException as exc:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "powered lease acquisition failed"
            ) from exc

    def heartbeat(
        self, lease: Any, *, phase: str, deadline_monotonic_ns: int
    ) -> None:
        self._require_before_deadline(deadline_monotonic_ns, "lease heartbeat")
        if lease is not self.powered_lease or getattr(lease, "is_active", False) is not True:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "lease heartbeat lacks exact active ownership"
            )
        try:
            lease.heartbeat(phase=phase)
        except BaseException as exc:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "powered lease heartbeat publication failed"
            ) from exc
        if self._now() >= deadline_monotonic_ns:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "powered lease heartbeat completed too late"
            )

    def release_and_verify(
        self,
        lease: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> LeaseReleaseOutcome:
        from scripts import aigp_live_lease

        self._require_before_deadline(deadline_monotonic_ns, "lease release")
        if lease is not self.powered_lease or self.lease_store is None:
            raise OrchestrationPhaseError(
                "lease_release_unconfirmed", "lease release target is not exact"
            )
        if self._lease_release_attempted:
            raise OrchestrationPhaseError(
                "lease_release_unconfirmed", "lease release is single-use"
            )
        self._lease_release_attempted = True
        # A cadence callback failure must not prevent the one-shot kernel
        # release.  The orchestrator records that evidence failure separately.
        try:
            heartbeat()
        except BaseException:
            pass
        try:
            final_index, _digest = lease.release()
        except BaseException:
            records = self.lease_store.records
            if records and records[-1]["event"] == "released":
                released = records[-1]["observed_monotonic_ns"]
                return LeaseReleaseOutcome(True, released, None)
            return LeaseReleaseOutcome(False, None, None)
        records = self.lease_store.records
        if (
            not records
            or records[-1]["event"] != "released"
            or records[-1]["release_proved"] is not True
        ):
            return LeaseReleaseOutcome(False, None, None)
        checked = aigp_live_lease.validate_powered_live_lease_index(final_index)
        self.last_release_index = checked
        return LeaseReleaseOutcome(
            True,
            records[-1]["observed_monotonic_ns"],
            dict(checked),
        )

    def _authority_context(self) -> tuple[dict[str, Any], str, str]:
        if self.prechild_proof is None or self.training_attestation is None:
            raise OrchestrationPhaseError(
                "child_spawn_failed", "spawn authority lacks topology/Training evidence"
            )
        attempt = self._load_attempt_envelope()
        process_sha256 = attempt_contract.canonical_file_sha256(self.prechild_proof)
        training_sha256 = attempt_contract.canonical_file_sha256(
            self.training_attestation
        )
        process_value = self._load_canonical_document(
            self.freeze["paths"]["process_proof"],
            expected_sha256=process_sha256,
            validator=attempt_contract.validate_simulator_process_proof,
            label="prechild simulator process proof",
        )
        training_value = self._load_canonical_document(
            self.freeze["paths"]["training_attestation"],
            expected_sha256=training_sha256,
            validator=lambda value: attempt_contract.validate_training_attestation(
                value, process_proof=process_value
            ),
            label="Training attestation",
        )
        if process_value != self.prechild_proof or training_value != self.training_attestation:
            raise OrchestrationPhaseError(
                "build_or_candidate_changed", "spawn evidence changed after publication"
            )
        return attempt, process_sha256, training_sha256

    def _job_wait_callback(
        self,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Callable[[int], None]:
        last_heartbeat = [self._now()]

        def wait_ns(duration_ns: int) -> None:
            import time

            now = self._now()
            remaining = max(0, deadline_monotonic_ns - now)
            time.sleep(min(duration_ns, self._POLL_NS, remaining) / 1_000_000_000.0)
            now = self._now()
            if now - last_heartbeat[0] >= heartbeat.period_ns:
                heartbeat()
                last_heartbeat[0] = self._now()

        return wait_ns

    def _terminate_job_tree(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Any:
        return self.runtime.terminate_job_process_tree_residue(
            child,
            exit_code=1,
            deadline_monotonic_ns=deadline_monotonic_ns,
            monotonic_ns=self.clock.now_ns,
            wait_ns=self._job_wait_callback(
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
            ),
        )

    def _abort_spawned_child(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Any:
        try:
            child.capability_pipe.abort()
        except BaseException as exc:
            raise OrchestrationPhaseError(
                "process_residue", "blocked child capability pipe could not be aborted"
            ) from exc
        now = self._now()
        reserve = min(1_000_000_000, max(0, deadline_monotonic_ns - now))
        natural_deadline = max(now, deadline_monotonic_ns - reserve)
        proof = self._wait_job_tree(
            child,
            deadline_monotonic_ns=natural_deadline,
            heartbeat=heartbeat,
        )
        if proof.state == "residue":
            try:
                proof = self._terminate_job_tree(
                    child,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                    heartbeat=heartbeat,
                )
            except BaseException as exc:
                raise OrchestrationPhaseError(
                    "process_residue", "blocked child retained process-tree residue"
                ) from exc
        self.tree_proofs[id(child)] = proof
        return proof

    def _build_process_authority(
        self,
        *,
        role: str,
        argv: Sequence[str],
        child: Any,
        parent_process: Any,
        attempt: Mapping[str, Any],
        process_proof_sha256: str,
        training_attestation_sha256: str,
    ) -> dict[str, Any]:
        if self.lease_store is None or self.powered_lease is None:
            raise OrchestrationPhaseError(
                "child_spawn_failed", "spawn authority lacks a powered lease"
            )
        records = self.lease_store.records
        hashes = self.lease_store.record_hashes
        if not records or len(records) != len(hashes):
            raise OrchestrationPhaseError(
                "lease_unverifiable", "spawn authority lacks a complete lease row"
            )
        process_key = "child_process" if role == "powered_child" else "cleanup_process"
        expected_phase = "child_spawn" if role == "powered_child" else "fallback_spawn"
        current = records[-1]
        if (
            current[process_key] != child.identity
            or current["event"] != "phase"
            or current["phase"] != expected_phase
            or current["owner_role"] != "wrapper"
            or current["owner_process"] != self.wrapper_identity
            or current["wrapper_process"] != self.wrapper_identity
            or current["attempt_envelope_sha256"] != self.attempt_envelope_sha256
            or current["attempt_context_sha256"] != attempt["context_sha256"]
        ):
            raise OrchestrationPhaseError(
                "lease_unverifiable",
                "spawn identity/ownership is absent from the current lease row",
            )
        anchor = self._spawn_anchors.get(id(child))
        if anchor is None:
            raise OrchestrationPhaseError(
                "child_spawn_failed", "spawn authority lacks its creation anchor"
            )
        if role == "powered_child":
            deadlines = {
                "anchor": anchor,
                "total": anchor + 110_000_000_000,
                "prepower": anchor + 52_000_000_000,
                "powered": anchor + 57_000_000_000,
                "cleanup": anchor + 72_000_000_000,
                "replay_close": anchor + 107_000_000_000,
                "exit": anchor + 110_000_000_000,
            }
            capability_key = "child_sha256"
        else:
            deadlines = {
                "anchor": anchor,
                "total": anchor + 25_000_000_000,
                "exit": anchor + 25_000_000_000,
            }
            capability_key = "cleanup_sha256"
        value = {
            "schema": "aigp-vq2-powered-process-authority/1",
            "task_id": attempt_contract.TASK_ID,
            "session_id": attempt_contract.SESSION_ID,
            "attempt_id": attempt_contract.ATTEMPT_ID,
            "role": role,
            "created_at_utc": self.utc_now(),
            "created_monotonic_ns": self._now(),
            "attempt_envelope_sha256": self.attempt_envelope_sha256,
            "attempt_context_sha256": attempt["context_sha256"],
            "live_freeze_sha256": attempt["context"]["live_freeze"]["sha256"],
            "wrapper_process": dict(self.wrapper_identity),
            "process": dict(child.identity),
            "parent_handle": {
                "value": parent_process.handle_value,
                "process": dict(parent_process.identity),
                "access": "synchronize_query_limited_information",
                "inherited": True,
            },
            "capability_sha256": attempt["capabilities"][capability_key],
            "lease_record_sha256": hashes[-1],
            "training_attestation_sha256": training_attestation_sha256,
            "simulator_process_proof_sha256": process_proof_sha256,
            "argv_sha256": attempt_contract.canonical_object_sha256(list(argv)),
            "job": child.containment.to_primitive(),
            "absolute_deadlines": deadlines,
        }
        return attempt_contract.validate_process_authority(
            value, attempt=attempt, argv=argv
        )

    def _spawn_blocked_role(
        self,
        *,
        role: str,
        argv: Sequence[str],
        handles: AttemptHandleSet,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> BlockedProcess:
        import os

        self._require_before_deadline(deadline_monotonic_ns, f"{role} spawn")
        if handles is not self.handle_set:
            raise OrchestrationPhaseError(
                "child_spawn_failed", "spawn did not receive the exact attempt handles"
            )
        _validate_handle_set(handles)
        attempt, process_hash, training_hash = self._authority_context()
        expected_argv = (
            attempt["context"]["child_argv"]
            if role == "powered_child"
            else attempt["context"]["cleanup_argv"]
        )
        if list(argv) != expected_argv:
            raise OrchestrationPhaseError(
                "child_spawn_failed", "spawn argv changed from the attempt envelope"
            )
        if self.powered_lease is None or self.powered_lease.is_active is not True:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "spawn lacks active wrapper lease ownership"
            )
        if role == "powered_child":
            if self.child is not None:
                raise OrchestrationPhaseError(
                    "child_spawn_failed", "powered child spawn is single-use"
                )
            pipe = self.child_pipe
            parent_process = self.child_parent
            stdout_path = self.freeze["paths"]["runner_stdout"]
            stderr_path = self.freeze["paths"]["runner_stderr"]
            cwd = self.freeze["execution"]["child_cwd"]
            lease_phase = "child_spawn"
        else:
            if self.fallback is not None or self.child is None:
                raise OrchestrationPhaseError(
                    "child_spawn_failed", "cleanup fallback spawn ordering is invalid"
                )
            pipe = self.cleanup_pipe
            parent_process = self.cleanup_parent
            stdout_path = self.freeze["paths"]["cleanup_stdout"]
            stderr_path = self.freeze["paths"]["cleanup_stderr"]
            cwd = self.freeze["execution"]["cleanup_cwd"]
            lease_phase = "fallback_spawn"
        if pipe is None or parent_process is None or parent_process.alive() is not True:
            raise OrchestrationPhaseError(
                "child_spawn_failed", "spawn inheritance handles are unavailable"
            )

        environment = self._native_environment_for_spawn()
        directory = self._attempt_directory_receipt()
        stdout_handle: int | None = None
        stderr_handle: int | None = None
        null_descriptor: int | None = None
        null_handle: int | None = None
        spawned: Any = None
        close_failures: list[str] = []
        try:
            stdout_handle = self.secure.create_inheritable_output_file(
                stdout_path,
                parent=directory,
                deadline_monotonic_ns=deadline_monotonic_ns,
            )
            stderr_handle = self.secure.create_inheritable_output_file(
                stderr_path,
                parent=directory,
                deadline_monotonic_ns=deadline_monotonic_ns,
            )
            null_descriptor, null_handle = self._open_readonly_inheritable_nul()
            self._spawn_anchors[id(pipe)] = self._now()
            spawned = self.runtime.spawn_blocked_child(
                argv,
                cwd=cwd,
                environment=environment,
                capability_pipe=pipe,
                parent_process=parent_process,
                stdin_handle=null_handle,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                operations=self.process_operations,
            )
        except BaseException as exc:
            spawn_error = exc
        else:
            spawn_error = None
        finally:
            for name, value in (
                ("child_stdout", stdout_handle),
                ("child_stderr", stderr_handle),
            ):
                if value is not None:
                    try:
                        self.process_operations.close_handle(value)
                    except BaseException as close_exc:
                        close_failures.append(f"{name}:{type(close_exc).__name__}")
            if null_descriptor is not None:
                try:
                    os.close(null_descriptor)
                except BaseException as close_exc:
                    close_failures.append(f"child_stdin:{type(close_exc).__name__}")

        if spawned is None:
            detail = "blocked process creation failed"
            if close_failures:
                detail += "; parent standard-handle close failed"
            raise OrchestrationPhaseError("child_spawn_failed", detail) from spawn_error

        # The conservative QPC anchor is sampled immediately before CreateProcess.
        anchor = self._spawn_anchors.pop(id(pipe))
        self._spawn_anchors[id(spawned)] = anchor
        if role == "powered_child":
            self.child = spawned
        else:
            self.fallback = spawned
        self._output_paths[id(spawned)] = (stdout_path, stderr_path)
        if close_failures:
            try:
                self._abort_spawned_child(
                    spawned,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                    heartbeat=heartbeat,
                )
            finally:
                proof = self.tree_proofs.get(id(spawned))
                if proof is not None:
                    spawned.close_retained_handles(tree_exit_proof=proof)
                    self._closed_process_ids.add(id(spawned))
            raise OrchestrationPhaseError(
                "child_spawn_failed", "parent standard handles did not close after spawn"
            )

        try:
            if role == "powered_child":
                self.powered_lease.bind_child_process(spawned.identity)
            else:
                self.powered_lease.bind_cleanup_process(spawned.identity)
            # This immutable row must be persisted after the identity bind and
            # before the authority hashes it.
            self.powered_lease.publish_phase(lease_phase)
            authority = self._build_process_authority(
                role=role,
                argv=argv,
                child=spawned,
                parent_process=parent_process,
                attempt=attempt,
                process_proof_sha256=process_hash,
                training_attestation_sha256=training_hash,
            )
            self.process_authorities[id(spawned)] = authority
            self._require_before_deadline(deadline_monotonic_ns, f"{role} authority")
        except BaseException as exc:
            try:
                self._abort_spawned_child(
                    spawned,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                    heartbeat=heartbeat,
                )
            finally:
                proof = self.tree_proofs.get(id(spawned))
                if proof is not None:
                    spawned.close_retained_handles(tree_exit_proof=proof)
                    self._closed_process_ids.add(id(spawned))
            if isinstance(exc, OrchestrationPhaseError):
                raise
            raise OrchestrationPhaseError(
                "child_spawn_failed", "spawn authority construction failed"
            ) from exc
        return BlockedProcess(
            handle=spawned,
            identity=dict(spawned.identity),
            authority=authority,
        )

    def spawn_powered_child_blocked(
        self,
        *,
        argv: Sequence[str],
        handles: AttemptHandleSet,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> BlockedProcess:
        return self._spawn_blocked_role(
            role="powered_child",
            argv=argv,
            handles=handles,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )

    def spawn_cleanup_fallback_blocked(
        self,
        *,
        argv: Sequence[str],
        handles: AttemptHandleSet,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> BlockedProcess:
        return self._spawn_blocked_role(
            role="cleanup_fallback",
            argv=argv,
            handles=handles,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )

    def _release_role_capability(
        self,
        child: Any,
        *,
        role: str,
        frame: bytearray,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None:
        import hmac

        expected_child = self.child if role == "powered_child" else self.fallback
        if child is not expected_child:
            raise OrchestrationPhaseError(
                "child_spawn_failed", "capability release child is not exact"
            )
        if type(frame) is not bytearray or len(frame) != 36 or frame[:4] != b"\x20\x00\x00\x00":
            raise OrchestrationPhaseError(
                "child_spawn_failed", "capability release frame is malformed"
            )
        self._require_before_deadline(deadline_monotonic_ns, "capability release")
        attempt = self._load_attempt_envelope()
        capability_key = "child_sha256" if role == "powered_child" else "cleanup_sha256"
        domain_key = "child" if role == "powered_child" else "cleanup"
        observed = attempt_contract.derive_capability_sha256(
            _CAPABILITY_DOMAINS[domain_key],
            attempt["context_sha256"],
            memoryview(frame)[4:],
        )
        if not hmac.compare_digest(observed, attempt["capabilities"][capability_key]):
            raise OrchestrationPhaseError(
                "child_spawn_failed", "capability release secret did not match authority"
            )
        if self.powered_lease is None or self.powered_lease.is_active is not True:
            raise OrchestrationPhaseError(
                "lease_unverifiable", "capability release lacks active lease ownership"
            )
        heartbeat()
        self._require_before_deadline(deadline_monotonic_ns, "capability release")
        # This write/close is deliberately the final fallible operation here:
        # callers can then latch the child as released without a post-send gap.
        child.release_capability(
            memoryview(frame)[4:],
            deadline_monotonic_ns=deadline_monotonic_ns,
            monotonic_ns=self.clock.now_ns,
        )

    def release_child_capability(
        self,
        child: Any,
        *,
        frame: bytearray,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None:
        self._release_role_capability(
            child,
            role="powered_child",
            frame=frame,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )

    def release_cleanup_capability(
        self,
        child: Any,
        *,
        frame: bytearray,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None:
        self._release_role_capability(
            child,
            role="cleanup_fallback",
            frame=frame,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )

    def abort_blocked_process(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> None:
        if child is not self.child and child is not self.fallback:
            raise OrchestrationPhaseError(
                "process_residue", "blocked-process abort target is not exact"
            )
        self._abort_spawned_child(
            child,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )

    def close_attempt_handles(
        self, handles: AttemptHandleSet, *, deadline_monotonic_ns: int
    ) -> None:
        _require_positive_int(deadline_monotonic_ns, "attempt-handle close deadline")
        if handles is not self.handle_set:
            raise OrchestrationPhaseError(
                "internal_error", "attempt-handle close target is not exact"
            )
        failures: list[str] = []
        for name, value, close in (
            ("child_capability", self.child_pipe, lambda item: item.abort()),
            ("cleanup_capability", self.cleanup_pipe, lambda item: item.abort()),
            ("child_parent", self.child_parent, lambda item: item.close()),
            ("cleanup_parent", self.cleanup_parent, lambda item: item.close()),
        ):
            if value is None:
                continue
            try:
                close(value)
            except BaseException as exc:
                failures.append(f"{name}:{type(exc).__name__}")
        self.child_pipe = None
        self.cleanup_pipe = None
        self.child_parent = None
        self.cleanup_parent = None
        self.handle_set = None
        if failures:
            raise OrchestrationPhaseError(
                "internal_error", "attempt-handle close failures: " + ",".join(failures)
            )

    def close_process_handle(
        self, child: Any, *, deadline_monotonic_ns: int
    ) -> None:
        _require_positive_int(deadline_monotonic_ns, "process-handle close deadline")
        if child is not self.child and child is not self.fallback:
            raise OrchestrationPhaseError(
                "process_residue", "process-handle close target is not exact"
            )
        if id(child) in self._closed_process_ids:
            return
        proof = self.tree_proofs.get(id(child))
        if proof is None:
            raise OrchestrationPhaseError(
                "process_residue", "process handles lack complete tree-exit proof"
            )
        child.close_retained_handles(tree_exit_proof=proof)
        self._closed_process_ids.add(id(child))

    @staticmethod
    def _parse_canonical_payload(
        payload: bytes,
        *,
        validator: Callable[[Any], Mapping[str, Any]],
        label: str,
    ) -> dict[str, Any]:
        try:
            checked = dict(validator(attempt_contract.strict_json_loads(payload)))
        except BaseException as exc:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} failed strict schema validation"
            ) from exc
        if payload != attempt_contract.canonical_json_file_bytes(checked):
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{label} is not one canonical JSON file"
            )
        return checked

    @staticmethod
    def _validate_sanitized_stderr(payload: bytes, *, role: str) -> None:
        try:
            text_value = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{role} stderr is not strict UTF-8"
            ) from exc
        allowed = (
            {
                "powered calibration failed before admission\n",
                "powered calibration bootstrap handle closure failed\n",
                "powered calibration execution integration is unavailable\n",
                "powered calibration failed after admission\n",
            }
            if role == "powered_child"
            else {
                "powered cleanup failed before admission\n",
                "powered cleanup failed after admission\n",
                "powered cleanup failed\n",
            }
        )
        if not text_value:
            return
        lines = text_value.splitlines(keepends=True)
        if "".join(lines) != text_value or any(line not in allowed for line in lines):
            raise OrchestrationPhaseError(
                "artifact_mismatch", f"{role} stderr is not a sanitized diagnostic stream"
            )

    def _poll_root_exit(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> tuple[str, int | None]:
        last_heartbeat = self._now()
        while True:
            try:
                if child.process.signaled():
                    if child.process.reprove() != child.identity:
                        return "identity_changed", None
                    return "exited", child.process.exit_code()
                if (
                    self.retained_wrapper is not None
                    and self.retained_wrapper.alive() is not True
                ):
                    return "wrapper_death", None
            except BaseException:
                return "identity_changed", None
            now = self._now()
            if now >= deadline_monotonic_ns:
                return "timeout", None
            if now - last_heartbeat >= heartbeat.period_ns:
                heartbeat()
                last_heartbeat = self._now()
            self._sleep_poll(deadline_monotonic_ns)

    @staticmethod
    def _normal_phase_names(
        rows: Sequence[Mapping[str, Any]], *, role: str
    ) -> list[str]:
        return [
            row["phase"]
            for row in rows
            if row["phase"] != "parent_death_lease_takeover"
        ]

    @staticmethod
    def _phase_deadlines_bind_authority(
        rows: Sequence[Mapping[str, Any]],
        *,
        role: str,
        authority: Mapping[str, Any],
        completed_result: bool,
        certificate: bool,
    ) -> bool:
        limits = authority["absolute_deadlines"]
        if role == "powered_child":
            parent_by_phase = {
                "connect": "prepower",
                "preflight": "prepower",
                "reset_epoch": "prepower",
                "normalize_disarmed": "prepower",
                "countdown_go": "prepower",
                "arm": "prepower",
                "powered_stage": "powered",
                "cleanup": "cleanup",
                "replay_close": "replay_close",
                "finalize": "exit",
            }
            required = [
                "connect",
                "preflight",
                "reset_epoch",
                "normalize_disarmed",
                "countdown_go",
                "arm",
                "powered_stage",
                "cleanup",
            ]
            if not certificate:
                required += ["replay_close", "finalize"]
            takeover_parents = {
                limits["cleanup"],
                limits["replay_close"],
                limits["exit"],
            }
        else:
            parent_by_phase = {
                "connect": "total",
                "disarm": "total",
                "reset_and_epoch": "total",
                "finalize": "exit",
            }
            required = ["connect", "disarm", "reset_and_epoch", "finalize"]
            takeover_parents = {limits["total"], limits["exit"]}
        for row in rows:
            phase = row["phase"]
            if row["started_monotonic_ns"] < limits["anchor"]:
                return False
            if phase == "parent_death_lease_takeover":
                if row["parent_deadline_monotonic_ns"] not in takeover_parents:
                    return False
                continue
            limit_name = parent_by_phase.get(phase)
            if limit_name is None or row["parent_deadline_monotonic_ns"] != limits[limit_name]:
                return False
        names = WindowsProductionLiveBoundary._normal_phase_names(rows, role=role)
        if certificate:
            if role == "powered_child":
                # A stage failure may jump from the reached production prefix
                # directly to mandatory cleanup.  Proved cleanup is therefore
                # intentionally independent of whether every production phase
                # ran, but the normal phase sequence must still be exactly one
                # reached prefix followed by the single cleanup phase.
                production = required[:-1]
                if (
                    not names
                    or names[-1] != "cleanup"
                    or names[:-1] != production[: len(names) - 1]
                    or (completed_result and len(names) < 2)
                ):
                    return False
            elif completed_result and names != required:
                return False
            elif not names or names[-1] != "finalize":
                return False
        elif completed_result and names != required:
            return False
        return True

    def _lease_certificate_binding(
        self,
        certificate: Mapping[str, Any],
        *,
        role: str,
        child: Any,
    ) -> tuple[bool, bool]:
        """Return (exact binding, wrapper-death/takeover observed)."""

        if self.lease_store is None:
            return False, False
        lease = certificate["lease"]
        generation = lease["generation"]
        records = self.lease_store.records
        hashes = self.lease_store.record_hashes
        if generation >= len(records) or generation >= len(hashes):
            return False, certificate["parent_state"]["mode"] == "signaled_takeover"
        row = records[generation]
        process_key = "child_process" if role == "powered_child" else "cleanup_process"
        expected_owner_role = (
            "wrapper"
            if lease["owner_role"] == "wrapper"
            else (
                "powered-child-parent-death"
                if role == "powered_child"
                else "cleanup-fallback-parent-death"
            )
        )
        expected_owner_process = (
            self.wrapper_identity
            if expected_owner_role == "wrapper"
            else child.identity
        )
        wrapper_death = (
            certificate["parent_state"]["mode"] == "signaled_takeover"
            or lease["owner_role"] != "wrapper"
        )
        exact = (
            lease["record_sha256"] == hashes[generation]
            and lease["authority_valid"] is True
            and row[process_key] == child.identity
            and row["wrapper_process"] == self.wrapper_identity
            and lease["owner_role"] == expected_owner_role
            and row["owner_role"] == expected_owner_role
            and row["owner_process"] == expected_owner_process
            and certificate["parent_state"]["wrapper_process"]
            == self.wrapper_identity
        )
        if wrapper_death:
            exact = (
                exact
                and certificate["parent_state"]["mode"] == "signaled_takeover"
                and certificate["parent_state"]["takeover_lease_record_sha256"]
                == lease["record_sha256"]
            )
        if not wrapper_death:
            exact = (
                exact
                and lease["owner_role"] == "wrapper"
                and row["owner_role"] == "wrapper"
                and row["owner_process"] == self.wrapper_identity
            )
        return exact, wrapper_death

    @staticmethod
    def _audit_binds_certificate_receipts(
        audit: Mapping[str, Any], certificate: Mapping[str, Any]
    ) -> bool:
        receipts = certificate["outbound_receipts"]
        categories = {
            "timesync": 0,
            "gcs_heartbeat": 0,
            "sim_reset": 0,
            "arm": 0,
            "disarm": 0,
            "attitude_target": 0,
        }
        returned = 0
        raised = 0
        for receipt in receipts:
            if receipt["schema"] == "aigp-vq2-attitude-target-outbound/1":
                category = "attitude_target"
            else:
                category = receipt["category"]
            if category not in categories:
                return False
            categories[category] += 1
            if receipt["outcome"] == "returned":
                returned += 1
            elif receipt["outcome"] == "raised":
                raised += 1
            else:
                return False
        return (
            audit["receipt_count"] == len(receipts)
            and audit["receipt_returned"] == returned
            and audit["receipt_raised"] == raised
            and all(audit[name] == count for name, count in categories.items())
        )

    def _collect_supervision_evidence(
        self,
        child: Any,
        *,
        role: str,
        exit_code: int,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> tuple[bool, bool, bool, tuple[str, ...], Mapping[str, Any]]:
        reasons: set[str] = set()
        cleanup_proved = False
        collection_valid = False
        wrapper_death = False
        patch: dict[str, Any] = {}
        attempt = self._load_attempt_envelope()
        expected_authority = self.process_authorities.get(id(child))
        if expected_authority is None:
            reason = "child_failed" if role == "powered_child" else "cleanup_unconfirmed"
            return False, False, False, (reason,), patch
        authority_path = self.freeze["paths"][
            "child_authority" if role == "powered_child" else "cleanup_authority"
        ]
        argv = (
            attempt["context"]["child_argv"]
            if role == "powered_child"
            else attempt["context"]["cleanup_argv"]
        )
        try:
            authority_payload, authority_file = self._read_retained_complete_file(
                authority_path,
                maximum_bytes=2 * 1024 * 1024,
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
                label=f"{role} process authority",
            )
            authority = self._parse_canonical_payload(
                authority_payload,
                validator=lambda value: attempt_contract.validate_process_authority(
                    value, attempt=attempt, argv=argv
                ),
                label=f"{role} process authority",
            )
            if authority != expected_authority or authority["process"] != child.identity:
                raise OrchestrationPhaseError(
                    "artifact_mismatch", f"{role} process authority identity changed"
                )
        except BaseException:
            return False, False, False, ("artifact_mismatch",), patch

        stdout_path, stderr_path = self._output_paths[id(child)]
        try:
            stderr_payload, _stderr_file = self._read_retained_complete_file(
                stderr_path,
                maximum_bytes=self._MAX_STDERR_BYTES,
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
                label=f"{role} stderr",
            )
            self._validate_sanitized_stderr(stderr_payload, role=role)
        except BaseException:
            reasons.add("artifact_mismatch")

        certificate_path = self.freeze["paths"][
            "child_cleanup_certificate"
            if role == "powered_child"
            else "fallback_cleanup_certificate"
        ]
        certificate: dict[str, Any] | None = None
        certificate_file: Mapping[str, Any] | None = None
        try:
            certificate_payload, certificate_file = self._read_retained_complete_file(
                certificate_path,
                maximum_bytes=16 * 1024 * 1024,
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
                label=f"{role} cleanup certificate",
            )
            certificate = self._parse_canonical_payload(
                certificate_payload,
                validator=attempt_contract.validate_cleanup_certificate,
                label=f"{role} cleanup certificate",
            )
        except BaseException:
            reasons.add("cleanup_unconfirmed")

        result: dict[str, Any] | None = None
        try:
            stdout_payload, _stdout_file = self._read_retained_complete_file(
                stdout_path,
                maximum_bytes=self._MAX_STDOUT_BYTES,
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
                label=f"{role} stdout",
            )
            result = self._parse_canonical_payload(
                stdout_payload,
                validator=(
                    (lambda value: attempt_contract.validate_process_result(
                        value, cleanup_certificate=certificate
                    ))
                    if certificate is not None
                    else attempt_contract.validate_process_result
                ),
                label=f"{role} process result",
            )
        except BaseException:
            reasons.add("capture_incomplete" if role == "powered_child" else "cleanup_unconfirmed")

        if certificate is not None:
            lease_exact, takeover = self._lease_certificate_binding(
                certificate, role=role, child=child
            )
            wrapper_death = takeover
            authority_ref = certificate["authority"]
            endpoint_processes = [certificate["endpoints"]["mavlink"]]
            if certificate["endpoints"]["camera"] is not None:
                endpoint_processes.append(certificate["endpoints"]["camera"])
            endpoints_exact = all(
                endpoint["state"] == "not_opened"
                or endpoint["bind"]["owner_process"] == child.identity
                for endpoint in endpoint_processes
            )
            phases_exact = self._phase_deadlines_bind_authority(
                certificate["phase_deadlines"],
                role=role,
                authority=authority,
                completed_result=certificate["outcome"] == "proved",
                certificate=True,
            )
            if role == "powered_child":
                cleanup_rows = [
                    row
                    for row in certificate["phase_deadlines"]
                    if row["phase"] == "cleanup"
                ]
                certificate_window_exact = (
                    len(cleanup_rows) == 1
                    and certificate["started_monotonic_ns"]
                    == cleanup_rows[0]["started_monotonic_ns"]
                    and certificate["deadline_monotonic_ns"]
                    == cleanup_rows[0]["deadline_monotonic_ns"]
                )
            else:
                certificate_window_exact = (
                    certificate["started_monotonic_ns"]
                    >= authority["absolute_deadlines"]["anchor"]
                    and certificate["deadline_monotonic_ns"]
                    == authority["absolute_deadlines"]["total"]
                )
            trigger_exact = (
                certificate["trigger"] == "parent_death"
                if certificate["parent_state"]["mode"] == "signaled_takeover"
                else (
                    certificate["trigger"] == "wrapper_fallback"
                    if role == "cleanup_fallback"
                    else certificate["trigger"]
                    in {"normal_completion", "stage_abort"}
                )
            )
            certificate_exact = (
                certificate["producer_role"] == role
                and authority_ref["process_authority"]
                == {"path": authority_path, "sha256": authority_file["sha256"]}
                and authority_ref["attempt_context_sha256"]
                == attempt["context_sha256"]
                and authority_ref["attempt_envelope_sha256"]
                == self.attempt_envelope_sha256
                and authority_ref["producer"] == child.identity
                and certificate["started_monotonic_ns"]
                >= authority["absolute_deadlines"]["anchor"]
                and certificate["completed_monotonic_ns"]
                < authority["absolute_deadlines"]["exit"]
                and certificate_window_exact
                and trigger_exact
                and authority["absolute_deadlines"]["anchor"]
                <= certificate["parent_state"]["observed_monotonic_ns"]
                <= certificate["completed_monotonic_ns"]
                and lease_exact
                and endpoints_exact
                and phases_exact
            )
            if certificate["parent_state"]["mode"] == "signaled_takeover":
                certificate_exact = (
                    certificate_exact
                    and certificate["parent_state"][
                        "takeover_completed_monotonic_ns"
                    ]
                    <= certificate["completed_monotonic_ns"]
                )
            cleanup_proved = certificate_exact and certificate["outcome"] == "proved"
            if cleanup_proved:
                self.cleanup_certificates[id(child)] = certificate
            else:
                reasons.add("cleanup_unconfirmed")
            if certificate["collection_invalidating_codes"]:
                codes = set(certificate["collection_invalidating_codes"])
                if "unexpected_outbound" in codes:
                    reasons.add("unexpected_outbound")
                if "collision_observed" in codes:
                    reasons.add("watchdog_failed")
                if codes & {"camera_missing", "source_rejected"}:
                    reasons.add("capture_incomplete")
            if takeover:
                reasons.add("wrapper_death")

        if result is not None:
            self.process_results[id(child)] = result
            audit = result["outbound_audit"]
            result_exact = (
                result["producer_role"] == role
                and result["process_authority_sha256"] == authority_file["sha256"]
                and result["started_monotonic_ns"]
                == authority["absolute_deadlines"]["anchor"]
                and result["completed_monotonic_ns"]
                < authority["absolute_deadlines"]["exit"]
                and all(
                    row["started_monotonic_ns"]
                    <= result["completed_monotonic_ns"]
                    for row in result["phase_deadlines"]
                )
                and self._phase_deadlines_bind_authority(
                    result["phase_deadlines"],
                    role=role,
                    authority=authority,
                    completed_result=result["outcome"] == "completed",
                    certificate=False,
                )
                and (
                    (result["outcome"] == "completed" and exit_code == 0)
                    or (result["outcome"] == "failed" and exit_code == 1)
                )
            )
            if certificate is not None and certificate_file is not None:
                result_exact = (
                    result_exact
                    and result["cleanup_certificate"]["state"] == "published"
                    and result["cleanup_certificate"]["sha256"]
                    == certificate_file["sha256"]
                    and certificate["completed_monotonic_ns"]
                    <= result["completed_monotonic_ns"]
                    and self._audit_binds_certificate_receipts(audit, certificate)
                )
            if not result_exact:
                reasons.add("child_failed" if role == "powered_child" else "cleanup_unconfirmed")
            reasons.update(result["reason_codes"])
            if audit["position_target"] or audit["other_command"]:
                reasons.add("unexpected_outbound")
            if (
                audit["receipt_raised"]
                or audit["receipt_dropped"]
                or audit["receipt_buffered"]
            ):
                reasons.add("command_reconciliation_failed")
            if role == "powered_child":
                legacy = result["artifacts"]["legacy_record"]
                replay = result["artifacts"]["replay_bundle"]
                patch.update(
                    {
                        "legacy_record": legacy["state"],
                        "legacy_record_sha256": legacy["sha256"],
                        "replay_bundle": (
                            "sealed" if replay["state"] == "closed" else replay["state"]
                        ),
                        "replay_dataset_hash": replay["dataset_hash"],
                        "replay_manifest_sha256": replay["manifest_sha256"],
                        "replay_records_sha256": replay["records_sha256"],
                    }
                )
                artifacts_closed = (
                    legacy["state"] == "closed" and replay["state"] == "closed"
                )
            else:
                artifacts_closed = True
            collection_valid = (
                role == "powered_child"
                and result_exact
                and result["outcome"] == "completed"
                and cleanup_proved
                and certificate is not None
                and certificate["trigger"] == "normal_completion"
                and artifacts_closed
                and not wrapper_death
                and not reasons
            )
        ordered = tuple(sorted(reasons, key=lambda item: item.encode("utf-8")))
        return cleanup_proved, collection_valid, wrapper_death, ordered, patch

    def supervise_powered_child(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> ChildSupervisionOutcome:
        if child is not self.child:
            return ChildSupervisionOutcome(False, False, reason_codes=("child_failed",))
        state, exit_code = self._poll_root_exit(
            child,
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        if state == "wrapper_death":
            return ChildSupervisionOutcome(
                False,
                False,
                wrapper_death=True,
                reason_codes=("wrapper_death",),
            )
        if state == "timeout":
            return ChildSupervisionOutcome(
                False, False, reason_codes=("child_timeout",)
            )
        if state != "exited" or exit_code is None:
            return ChildSupervisionOutcome(
                False, False, reason_codes=("child_failed",)
            )
        try:
            cleanup, collection, wrapper_death, reasons, patch = (
                self._collect_supervision_evidence(
                    child,
                    role="powered_child",
                    exit_code=exit_code,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                    heartbeat=heartbeat,
                )
            )
        except BaseException:
            return ChildSupervisionOutcome(
                False, False, reason_codes=("capture_incomplete",)
            )
        return ChildSupervisionOutcome(
            cleanup,
            collection,
            wrapper_death=wrapper_death,
            reason_codes=reasons,
            artifact_state_patch=patch,
        )

    def supervise_cleanup_fallback(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> FallbackSupervisionOutcome:
        if child is not self.fallback:
            return FallbackSupervisionOutcome(False, ("cleanup_unconfirmed",))
        now = self._require_before_deadline(
            deadline_monotonic_ns, "fallback supervision"
        )
        reserve = min(1_000_000_000, max(0, deadline_monotonic_ns - now))
        natural_deadline = max(now, deadline_monotonic_ns - reserve)
        state, exit_code = self._poll_root_exit(
            child,
            deadline_monotonic_ns=natural_deadline,
            heartbeat=heartbeat,
        )
        if state != "exited" or exit_code is None:
            reasons = {
                "wrapper_death" if state == "wrapper_death" else "process_residue"
            }
            try:
                proof = self._wait_job_tree(
                    child,
                    deadline_monotonic_ns=natural_deadline,
                    heartbeat=heartbeat,
                )
                if proof.state != "exited":
                    proof = self._terminate_job_tree(
                        child,
                        deadline_monotonic_ns=deadline_monotonic_ns,
                        heartbeat=heartbeat,
                    )
                    reasons.add("process_residue")
                self.tree_proofs[id(child)] = proof
            except BaseException:
                reasons.add("process_residue")
                try:
                    forced = self._terminate_job_tree(
                        child,
                        deadline_monotonic_ns=deadline_monotonic_ns,
                        heartbeat=heartbeat,
                    )
                    self.tree_proofs[id(child)] = forced
                except BaseException:
                    pass
            return FallbackSupervisionOutcome(
                False,
                tuple(sorted(reasons, key=lambda item: item.encode("utf-8"))),
            )
        cleanup = False
        wrapper_death = False
        reasons: set[str] = set()
        try:
            cleanup, _collection, wrapper_death, evidence_reasons, _patch = (
                self._collect_supervision_evidence(
                    child,
                    role="cleanup_fallback",
                    exit_code=exit_code,
                    deadline_monotonic_ns=natural_deadline,
                    heartbeat=heartbeat,
                )
            )
            reasons.update(evidence_reasons)
        except BaseException:
            reasons.add("cleanup_unconfirmed")
        try:
            proof = self._wait_job_tree(
                child,
                deadline_monotonic_ns=natural_deadline,
                heartbeat=heartbeat,
            )
            if proof.state != "exited":
                terminated = self._terminate_job_tree(
                    child,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                    heartbeat=heartbeat,
                )
                self.tree_proofs[id(child)] = terminated
                return FallbackSupervisionOutcome(False, ("process_residue",))
            self.tree_proofs[id(child)] = proof
        except BaseException:
            reasons.add("process_residue")
            try:
                forced = self._terminate_job_tree(
                    child,
                    deadline_monotonic_ns=deadline_monotonic_ns,
                    heartbeat=heartbeat,
                )
                self.tree_proofs[id(child)] = forced
            except BaseException:
                pass
            return FallbackSupervisionOutcome(
                False,
                tuple(sorted(reasons, key=lambda item: item.encode("utf-8"))),
            )
        if wrapper_death:
            reasons.add("wrapper_death")
        return FallbackSupervisionOutcome(
            cleanup,
            tuple(sorted(reasons, key=lambda item: item.encode("utf-8"))),
        )

    def supervision_snapshot(self) -> Mapping[str, Any]:
        return {
            "stable_files": {
                path: dict(value)
                for path, value in sorted(
                    self.stable_file_proofs.items(), key=lambda item: item[0]
                )
            },
            "process_results": {
                str(key): attempt_contract.defensive_copy(value)
                for key, value in self.process_results.items()
            },
            "cleanup_certificates": {
                str(key): attempt_contract.defensive_copy(value)
                for key, value in self.cleanup_certificates.items()
            },
            "tree_exit": {
                str(key): value.to_primitive()
                for key, value in self.tree_proofs.items()
            },
        }

    def prove_prechild_identity(
        self,
        retained_wrapper: Any,
        *,
        topology_proof: Mapping[str, Any],
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]:
        if (
            retained_wrapper is not self.retained_wrapper
            or retained_wrapper.alive() is not True
            or self.prechild_proof is None
            or dict(topology_proof) != self.prechild_proof
            or any(handle.alive() is not True for handle in self.simulator_handles.values())
            or self._now() >= deadline_monotonic_ns
        ):
            raise OrchestrationPhaseError(
                "build_or_candidate_changed", "prechild process identity changed"
            )
        return {"proved": True, "wrapper_process": retained_wrapper.reprove()}

    def prove_child_tree_exit(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]:
        if child is not self.child:
            raise OrchestrationPhaseError(
                "process_residue", "child tree-exit target is not exact"
            )
        now = self._require_before_deadline(
            deadline_monotonic_ns, "child tree-exit proof"
        )
        reserve = min(1_000_000_000, max(0, deadline_monotonic_ns - now))
        natural_deadline = max(now, deadline_monotonic_ns - reserve)
        proof = self._wait_job_tree(
            child,
            deadline_monotonic_ns=natural_deadline,
            heartbeat=heartbeat,
        )
        if proof.state == "exited":
            self.tree_proofs[id(child)] = proof
            return {
                "proved": True,
                "tree_exit": proof.to_primitive(),
                "natural_exit_proved": True,
                "termination_is_cleanup_proof": False,
            }
        try:
            terminated = self._terminate_job_tree(
                child,
                deadline_monotonic_ns=deadline_monotonic_ns,
                heartbeat=heartbeat,
            )
        except BaseException as exc:
            raise OrchestrationPhaseError(
                "process_residue", "powered child process tree could not be emptied"
            ) from exc
        self.tree_proofs[id(child)] = terminated
        # The handles can now be closed, but forced termination is neither a
        # natural-exit occurrence nor proof of simulator cleanup.
        raise OrchestrationPhaseError(
            "process_residue",
            "powered child process-tree residue required forced termination",
        )

    def prove_prechild_free(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> Mapping[str, Any]:
        proof = self._free_port_contract(
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
            postchild=False,
        )
        return {"proved": True, "ports": proof}

    def prove_child_owners(
        self,
        child: Any,
        *,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]:
        if child is not self.child:
            raise OrchestrationPhaseError("ports_busy", "child handle is not exact")
        expected_pid = child.identity["pid"]
        last_heartbeat = self._now()
        matched: list[Any] = []
        while True:
            now = self._now()
            if now >= deadline_monotonic_ns:
                raise OrchestrationPhaseError(
                    "ports_busy", "child did not acquire both exact UDP ports"
                )
            snapshot = self.runtime.capture_udp_owner_snapshot(
                (14550, 5600),
                operations=self.udp_operations,
                monotonic_ns=self.clock.now_ns,
            )
            exact = (
                snapshot.owner_pids(2, 14550) == (expected_pid,)
                and snapshot.owner_pids(2, 5600) == (expected_pid,)
                and snapshot.owner_pids(23, 14550) == ()
                and snapshot.owner_pids(23, 5600) == ()
            )
            if exact:
                matched.append(snapshot)
                if len(matched) >= 2:
                    break
            else:
                matched.clear()
            if now - last_heartbeat >= self._HEARTBEAT_EMIT_NS:
                heartbeat()
                last_heartbeat = self._now()
            self._sleep_poll(deadline_monotonic_ns)
        self.active_owner_observations.extend(
            [
                {
                    "observed_monotonic_ns": matched[0].observed_monotonic_ns,
                    "port": 14550,
                    "role": "powered_child",
                    "pid": expected_pid,
                    "creation_filetime_100ns": child.identity[
                        "creation_filetime_100ns"
                    ],
                },
                {
                    "observed_monotonic_ns": matched[1].observed_monotonic_ns,
                    "port": 5600,
                    "role": "powered_child",
                    "pid": expected_pid,
                    "creation_filetime_100ns": child.identity[
                        "creation_filetime_100ns"
                    ],
                },
            ]
        )
        return {"proved": True, "snapshots": [item.to_primitive() for item in matched]}

    def prove_fallback_gate(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> Mapping[str, Any]:
        proof = self._free_port_contract(
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
            postchild=False,
        )
        return {"proved": True, "ports": proof}

    def _build_postchild_proof(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> dict[str, Any]:
        if self.prechild_proof is None or self.launch_result is None:
            raise OrchestrationPhaseError(
                "topology_failed", "prechild topology proof is unavailable"
            )
        launcher = self.simulator_handles["launcher"].reprove()
        payload = self.simulator_handles["payload"].reprove()
        window = self._window_proof(
            {"identity": payload, "retained": self.simulator_handles["payload"]}
        )
        if (
            launcher != self.prechild_proof["launcher_process"]
            or payload != self.prechild_proof["payload_process"]
            or window != self.prechild_proof["window"]
        ):
            raise OrchestrationPhaseError(
                "topology_failed", "simulator process/window topology changed"
            )
        task = self._query_task_absent(
            "after_child_or_fallback",
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )
        ports = self._free_port_contract(
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
            postchild=True,
        )
        proof = {
            **self.prechild_proof,
            "phase": "postchild",
            "observed_at_utc": self.utc_now(),
            "observed_monotonic_ns": self._now(),
            "scheduled_task": {
                "name": self._TASK_NAME,
                "observations": [
                    *self.prechild_proof["scheduled_task"]["observations"],
                    task,
                ],
            },
            "ports": ports,
        }
        self.final_ports_contract = ports
        self.postchild_proof = attempt_contract.validate_simulator_process_proof(
            proof
        )
        return dict(self.postchild_proof)

    def prove_final_process_state(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> Mapping[str, Any]:
        return self._build_postchild_proof(
            deadline_monotonic_ns=deadline_monotonic_ns,
            heartbeat=heartbeat,
        )

    def prove_final_free(
        self, *, deadline_monotonic_ns: int, heartbeat: HeartbeatPump
    ) -> Mapping[str, Any]:
        if self.final_ports_contract is None or self._now() >= deadline_monotonic_ns:
            raise OrchestrationPhaseError(
                "port_residue", "final free-port proof is unavailable"
            )
        return {"proved": True, "ports": "free", "transport": "closed"}

    def prove_unchanged(
        self,
        *,
        launch_result: Any,
        deadline_monotonic_ns: int,
        heartbeat: HeartbeatPump,
    ) -> Mapping[str, Any]:
        if (
            launch_result is not self.launch_result
            or self.postchild_proof is None
            or self._now() >= deadline_monotonic_ns
            or any(handle.alive() is not True for handle in self.simulator_handles.values())
        ):
            raise OrchestrationPhaseError(
                "topology_failed", "final simulator topology is not unchanged"
            )
        if self._window_proof(
            {
                "identity": self.postchild_proof["payload_process"],
                "retained": self.simulator_handles["payload"],
            }
        ) != self.postchild_proof["window"]:
            raise OrchestrationPhaseError(
                "topology_failed", "final simulator window changed"
            )
        return {
            "proved": True,
            "topology": "unchanged",
            "responsive": "yes",
            "scheduled_task": "absent",
        }

    def close_retained_wrapper(
        self, retained_wrapper: Any, *, deadline_monotonic_ns: int
    ) -> None:
        if retained_wrapper is not self.retained_wrapper:
            raise OrchestrationPhaseError(
                "internal_error", "retained wrapper close target changed"
            )
        started = self._now()
        retained_wrapper.close()
        self.retained_wrapper = None
        if started >= deadline_monotonic_ns or self._now() >= deadline_monotonic_ns:
            raise OrchestrationPhaseError(
                "deadline_expired", "retained wrapper handle closed too late"
            )

    def close(self) -> None:
        """Deterministically close every locally retained production handle.

        This boundary never converts closure into process-exit, cleanup, or
        lease-release proof.  A live lease or a spawned tree without exact exit
        evidence is reported as a close failure instead of being silently
        abandoned.
        """

        if self._closed:
            return
        failures: list[str] = []

        def close_one(label: str, operation: Callable[[], Any]) -> None:
            try:
                operation()
            except BaseException as exc:
                failures.append(f"{label}:{type(exc).__name__}")

        # Abort any still-unconsumed capability before waiting on a blocked
        # process.  Closing these parent-owned handles grants no live authority.
        for label, value, operation in (
            ("child_capability", self.child_pipe, lambda item: item.abort()),
            ("cleanup_capability", self.cleanup_pipe, lambda item: item.abort()),
            ("child_parent", self.child_parent, lambda item: item.close()),
            ("cleanup_parent", self.cleanup_parent, lambda item: item.close()),
        ):
            if value is not None:
                close_one(label, lambda item=value, close=operation: close(item))
        self.child_pipe = None
        self.cleanup_pipe = None
        self.child_parent = None
        self.cleanup_parent = None
        self.handle_set = None

        def original_total_deadline(child: Any) -> int:
            try:
                value = self.attempt_envelope["context"][
                    "wrapper_absolute_deadlines"
                ]["total_deadline_monotonic_ns"]
                return _require_positive_int(value, "boundary close total deadline")
            except BaseException:
                authority = self.process_authorities.get(id(child))
                if authority is not None:
                    try:
                        return _require_positive_int(
                            authority["absolute_deadlines"]["exit"],
                            "boundary close process deadline",
                        )
                    except BaseException:
                        pass
                failures.append("process_close:original_deadline_unavailable")
                try:
                    return max(1, self._now())
                except BaseException:
                    return 1

        def drain_and_close_process(role: str, child: Any) -> None:
            process_key = id(child)
            if process_key in self._closed_process_ids:
                return
            proof = self.tree_proofs.get(process_key)
            if proof is None or getattr(proof, "state", None) not in {
                "exited",
                "terminated_residue",
            }:
                failures.append(f"{role}_process:missing_tree_exit_proof")
                deadline = original_total_deadline(child)
                phase = (
                    "fallback_supervision"
                    if role == "fallback"
                    else "child_exit_proof"
                )
                heartbeat_failed = [False]

                def emit_close_heartbeat() -> None:
                    try:
                        active = (
                            self.powered_lease is not None
                            and self.powered_lease.is_active is True
                        )
                    except BaseException as exc:
                        active = False
                        if not heartbeat_failed[0]:
                            failures.append(
                                f"{role}_lease_state:{type(exc).__name__}"
                            )
                            heartbeat_failed[0] = True
                    if not active:
                        return
                    try:
                        self.heartbeat(
                            self.powered_lease,
                            phase=phase,
                            deadline_monotonic_ns=deadline,
                        )
                    except BaseException as exc:
                        if not heartbeat_failed[0]:
                            failures.append(
                                f"{role}_lease_heartbeat:{type(exc).__name__}"
                            )
                            heartbeat_failed[0] = True

                heartbeat = HeartbeatPump(
                    phase=phase,
                    deadline_monotonic_ns=deadline,
                    period_ns=self.freeze["deadline_durations_ns"][
                        "lease_heartbeat_period"
                    ],
                    _emit=emit_close_heartbeat,
                )
                try:
                    heartbeat()
                    now = self._now()
                    if now < deadline:
                        reserve = min(1_000_000_000, deadline - now)
                        natural_deadline = max(now, deadline - reserve)
                    else:
                        natural_deadline = deadline
                    proof = self._wait_job_tree(
                        child,
                        deadline_monotonic_ns=natural_deadline,
                        heartbeat=heartbeat,
                    )
                    if proof.state != "exited":
                        proof = self._terminate_job_tree(
                            child,
                            deadline_monotonic_ns=deadline,
                            heartbeat=heartbeat,
                        )
                        failures.append(
                            f"{role}_process:forced_termination_noncleanup"
                        )
                    self.tree_proofs[process_key] = proof
                except BaseException as exc:
                    failures.append(f"{role}_process_drain:{type(exc).__name__}")
                    return
            try:
                child.close_retained_handles(tree_exit_proof=proof)
            except BaseException as exc:
                failures.append(f"{role}_process:{type(exc).__name__}")
            else:
                self._closed_process_ids.add(process_key)

        for role, child in (("fallback", self.fallback), ("child", self.child)):
            if child is not None:
                drain_and_close_process(role, child)

        # Child-output and authority handles intentionally deny new writers
        # while supervision consumes the evidence.  Release them only after
        # every spawned tree has been proved empty or forcibly removed.
        if self.secure is not None:
            for path, handle in reversed(tuple(self._stable_file_handles.items())):
                close_one(
                    f"stable_file:{path}",
                    lambda value=handle: self.secure._close_handle(value),
                )
        self._stable_file_handles.clear()
        self._stable_file_payloads.clear()

        for role, retained in reversed(tuple(self.simulator_handles.items())):
            close_one(f"simulator_{role}", retained.close)
        self.simulator_handles.clear()
        if self.retained_wrapper is not None:
            close_one("retained_wrapper", self.retained_wrapper.close)
            self.retained_wrapper = None

        if self.powered_lease is not None:
            try:
                active = self.powered_lease.is_active
            except BaseException as exc:
                failures.append(f"powered_lease_state:{type(exc).__name__}")
            else:
                if active is not False:
                    failures.append("powered_lease:still_active")

        if self.secure is not None:
            close_one("secure_boundary", self.secure.close)
        self._closed = True
        if failures:
            raise SecureBoundaryError(
                "production boundary close failures: " + ",".join(failures)
            )


class ProbeOrchestrator:
    """Offline-first owner of the sole injected A01 orchestration."""

    def __init__(
        self,
        *,
        offline: OfflineAdmissionService,
        secure: SecureCreateNewService,
        clock: QpcService,
        live: LiveOrchestrationServices | None = None,
        validators: OrchestrationRecordValidators | None = None,
    ) -> None:
        self.offline = offline
        self.secure = secure
        self.clock = clock
        self.live = live
        self.validators = (
            validators if validators is not None else OrchestrationRecordValidators()
        )

    def admit(self, arguments: ProbeArguments) -> FoundationAdmission:
        wrapper_start = _require_exact_nonnegative_int(
            self.clock.now_ns(), "wrapper first QPC read"
        )
        qpc_frequency = _require_positive_int(
            self.clock.query_performance_frequency_hz(),
            "QueryPerformanceFrequency result",
        )
        admission = admit_offline(
            arguments,
            self.offline,
            deadline_monotonic_ns=(
                wrapper_start
                + attempt_contract.DEADLINE_DURATIONS_NS["offline_precheck"]
            ),
            monotonic_ns=self.clock.now_ns,
        )
        snapshot = self.secure.inspect_attempt_root(admission.live_freeze["paths"])
        validate_attempt_gate(admission.live_freeze, snapshot)
        return FoundationAdmission(
            wrapper_started_monotonic_ns=wrapper_start,
            qpc_frequency_hz=qpc_frequency,
            offline=admission,
            attempt_root=snapshot,
        )

    def run(self, arguments: ProbeArguments) -> OrchestrationResult:
        try:
            foundation = self.admit(arguments)
            return self.execute_admitted(arguments, foundation)
        finally:
            self._close_offline()

    def execute_admitted(
        self,
        arguments: ProbeArguments,
        foundation: FoundationAdmission,
        *,
        live: LiveOrchestrationServices | None = None,
    ) -> OrchestrationResult:
        """Consume one already-proved foundation without repeating admission.

        Production uses this seam to construct live-capable providers only
        after the passive identity/root gate succeeds.  The supplied
        foundation is bound back to the exact parsed arguments so it cannot be
        replayed for a different freeze or evidence root.
        """

        if not isinstance(arguments, ProbeArguments):
            raise TypeError("arguments must be exact ProbeArguments")
        if not isinstance(foundation, FoundationAdmission):
            raise TypeError("foundation must be exact FoundationAdmission")
        if foundation.offline.arguments != arguments:
            raise OfflineAdmissionError(
                "admitted foundation does not bind the exact probe arguments"
            )
        selected = self.live if live is None else live
        if selected is None:
            raise LiveIntegrationUnavailable(
                "production runtime providers are not wired; attempt remains unconsumed"
            )
        validate_live_orchestration_services(selected)
        return _SingleAttemptExecution(
                arguments=arguments,
                foundation=foundation,
                offline=self.offline,
                secure=self.secure,
                clock=self.clock,
                services=selected,
                validators=self.validators,
            ).execute()

    def _close_offline(self) -> None:
        close = getattr(self.offline, "close", None)
        if callable(close):
            pending = sys.exc_info()[1]
            try:
                close()
            except BaseException as close_exc:
                if pending is None:
                    raise
                pending.add_note(
                    "offline retained-handle close also failed: "
                    f"{type(close_exc).__name__}: {close_exc}"
                )


@dataclass(frozen=True)
class _ProductionFactories:
    """Factory-only test seam for offline-first production composition."""

    clock: Callable[[], Any]
    offline: Callable[[], Any]
    secure: Callable[[Any], Any]
    orchestrator: Callable[..., Any]
    boundary: Callable[..., Any]
    postrelease: Callable[..., Any]

    def __post_init__(self) -> None:
        for name in (
            "clock",
            "offline",
            "secure",
            "orchestrator",
            "boundary",
            "postrelease",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"production {name} factory must be callable")


def run_initial_import_inventory_audit(
    *,
    audit_module: str,
    seed_modules: Sequence[str],
) -> int:
    """Emit one non-live L0 import inventory to an inherited stdout pipe.

    This entry point constructs only the QPC and read-only offline identity
    boundaries.  It has no private-root, simulator, process-launch, mutex,
    socket, fixed-port, or publication provider.
    """

    if audit_module != IMPORT_AUDIT_MODULE:
        raise OfflineAdmissionError("initial import audit module changed")
    if tuple(seed_modules) != POWERED_IMPORT_SEED_MODULES:
        raise OfflineAdmissionError("powered import seed inventory changed")
    expected_tail = ["-E", "-s", "-B", "-m", audit_module]
    WindowsProductionOfflineAdmission._validate_invocation_values(
        implementation=sys.implementation.name,
        version=tuple(sys.version_info[:3]),
        ignore_environment=sys.flags.ignore_environment,
        no_user_site=sys.flags.no_user_site,
        dont_write_bytecode=sys.flags.dont_write_bytecode,
        observed_argv=list(getattr(sys, "orig_argv", ())),
        expected_tail=expected_tail,
    )
    if len(sys.argv) != 1:
        raise OfflineAdmissionError("initial import audit accepts no arguments")
    output = getattr(sys.stdout, "buffer", None)
    if output is None or sys.stdout.isatty():
        raise OfflineAdmissionError(
            "initial import audit stdout must be an inherited binary pipe"
        )

    from scripts import aigp_vq2_powered_runtime as powered_runtime

    clock = powered_runtime.WindowsQpcProvider()
    service = WindowsProductionOfflineAdmission()
    begun = False
    succeeded = False
    try:
        started = clock.now_ns()
        service.begin_bounded_admission(
            deadline_monotonic_ns=(
                started + attempt_contract.DEADLINE_DURATIONS_NS["offline_precheck"]
            ),
            monotonic_ns=clock.now_ns,
            heartbeat=None,
        )
        begun = True
        inventory = service.derive_initial_import_inventory(
            seed_modules,
            POWERED_EAGER_IMPORT_MODULES,
            audit_module=audit_module,
        )
        succeeded = True
    finally:
        try:
            if begun:
                service.end_bounded_admission(succeeded=succeeded)
        finally:
            service.close()
    payload = attempt_contract.canonical_json_file_bytes(inventory)
    if len(payload) > WindowsProductionOfflineAdmission._MAX_JSON_BYTES:
        raise OfflineAdmissionError("initial import inventory exceeds its size bound")
    output.write(payload)
    output.flush()
    return 0


def _new_production_clock() -> Any:
    from scripts import aigp_vq2_powered_runtime as powered_runtime

    try:
        return powered_runtime.WindowsQpcProvider()
    except Exception as exc:
        raise LiveIntegrationUnavailable(
            "production Windows QPC provider is unavailable"
        ) from exc


def _new_production_offline() -> WindowsProductionOfflineAdmission:
    return WindowsProductionOfflineAdmission()


def _new_production_secure(clock: Any) -> WindowsSecureCreateNew:
    return WindowsSecureCreateNew(monotonic_ns=clock.now_ns)


def _new_production_orchestrator(
    *, offline: Any, secure: Any, clock: Any
) -> ProbeOrchestrator:
    return ProbeOrchestrator(offline=offline, secure=secure, clock=clock)


def _new_production_boundary(
    *, freeze: Mapping[str, Any], secure: Any, clock: Any
) -> WindowsProductionLiveBoundary:
    return WindowsProductionLiveBoundary(
        freeze=freeze,
        secure=secure,
        clock=clock,
    )


def _build_production_postrelease_inputs(
    freeze: Mapping[str, Any], boundary: Any
) -> Any:
    from scripts.aigp_vq2_powered_calibration_analysis import PostReleaseInputs

    paths = freeze["paths"]
    return PostReleaseInputs(
        live_freeze_path=paths["live_freeze"],
        implementation_inventory_path=freeze["candidate"][
            "implementation_inventory"
        ]["path"],
        environment_inventory_path=freeze["runtime"]["environment_inventory"][
            "path"
        ],
        import_inventory_path=freeze["runtime"]["import_inventory"]["path"],
        paths=paths,
        supervision_snapshot=boundary.supervision_snapshot,
    )


def _new_production_postrelease(
    *, inputs: Any, clock: Any, boundary: Any
) -> Any:
    from scripts.aigp_vq2_powered_calibration_analysis import (
        ProductionPostReleaseService,
    )

    return ProductionPostReleaseService(
        inputs,
        now_ns=clock.now_ns,
        utc_now=boundary.utc_now,
        split_publications_factory=SplitPublications,
    )


def _default_production_factories() -> _ProductionFactories:
    return _ProductionFactories(
        clock=_new_production_clock,
        offline=_new_production_offline,
        secure=_new_production_secure,
        orchestrator=_new_production_orchestrator,
        boundary=_new_production_boundary,
        postrelease=_new_production_postrelease,
    )


def _execute_production(
    arguments: ProbeArguments,
    *,
    factories: _ProductionFactories | None = None,
) -> OrchestrationResult:
    """Admit offline, then construct the one shared live boundary and execute."""

    from contextlib import ExitStack

    selected = _default_production_factories() if factories is None else factories
    if not isinstance(selected, _ProductionFactories):
        raise TypeError("factories must be exact _ProductionFactories")

    try:
        with ExitStack() as resources:
            clock = selected.clock()

            offline = selected.offline()
            offline_close = getattr(offline, "close", None)
            if not callable(offline_close):
                raise LiveIntegrationUnavailable(
                    "production offline provider has no deterministic close"
                )
            resources.callback(offline_close)

            secure = selected.secure(clock)
            secure_close = getattr(secure, "close", None)
            if not callable(secure_close):
                raise LiveIntegrationUnavailable(
                    "production secure boundary has no deterministic close"
                )
            # Registered after offline, so the idempotent secure fallback runs
            # before retained offline identity handles are released.
            resources.callback(secure_close)

            orchestrator = selected.orchestrator(
                offline=offline,
                secure=secure,
                clock=clock,
            )
            foundation = orchestrator.admit(arguments)

            freeze = foundation.offline.live_freeze
            boundary = selected.boundary(
                freeze=freeze,
                secure=secure,
                clock=clock,
            )
            boundary_close = getattr(boundary, "close", None)
            if not callable(boundary_close):
                raise LiveIntegrationUnavailable(
                    "production live boundary has no deterministic close"
                )
            # LIFO close order is boundary -> secure fallback -> offline.
            resources.callback(boundary_close)

            postrelease_inputs = _build_production_postrelease_inputs(
                freeze, boundary
            )
            postrelease = selected.postrelease(
                inputs=postrelease_inputs,
                clock=clock,
                boundary=boundary,
            )
            services = LiveOrchestrationServices(
                host=boundary,
                csprng=boundary,
                launcher=boundary,
                topology=boundary,
                training=boundary,
                process=boundary,
                ports=boundary,
                lease=boundary,
                spawn=boundary,
                supervision=boundary,
                postrelease=postrelease,
            )
            validate_live_orchestration_services(services)
            return orchestrator.execute_admitted(
                arguments,
                foundation,
                live=services,
            )
    except (
        PoweredCalibrationProbeError,
        attempt_contract.PoweredAttemptContractError,
    ):
        raise
    except Exception as exc:
        raise LiveIntegrationUnavailable(
            "production runtime composition or deterministic cleanup failed"
        ) from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    orchestrator: ProbeOrchestrator | None = None,
    _production_factories: _ProductionFactories | None = None,
) -> int:
    try:
        arguments = parse_arguments(argv)
        if orchestrator is None:
            result = _execute_production(
                arguments,
                factories=_production_factories,
            )
        else:
            if _production_factories is not None:
                raise TypeError(
                    "production factories cannot accompany an injected orchestrator"
                )
            result = orchestrator.run(arguments)
        if result.status != "complete":
            return 2
    except (PoweredCalibrationProbeError, attempt_contract.PoweredAttemptContractError) as exc:
        print(f"powered calibration probe refused: {exc}", file=sys.stderr)
        return 2
    return 0


def _validate_production_main_module_identity() -> Any:
    module = sys.modules.get("__main__")
    spec = getattr(module, "__spec__", None)
    if (
        module is None
        or getattr(module, "__dict__", None) is not globals()
        or getattr(spec, "name", None) != PROBE_MODULE
        or type(getattr(spec, "origin", None)) is not str
    ):
        raise OfflineAdmissionError(
            "powered probe must execute as the exact -m production module"
        )
    loader_filename = getattr(getattr(spec, "loader", None), "get_filename", None)
    if not callable(loader_filename):
        raise OfflineAdmissionError("powered probe execution loader is not file-backed")
    try:
        loader_origin = ntpath.normpath(loader_filename(PROBE_MODULE))
    except (ImportError, AttributeError, OSError, TypeError) as exc:
        raise OfflineAdmissionError("powered probe execution loader is invalid") from exc
    if loader_origin != ntpath.normpath(spec.origin):
        raise OfflineAdmissionError("powered probe execution origin is inconsistent")
    return module


def _bind_production_main_module_alias() -> None:
    """Bind exact ``-m`` execution to the canonical reviewed module name."""

    module = _validate_production_main_module_identity()
    if sys.modules.get("__main__") is not module:
        raise OfflineAdmissionError("powered probe execution module identity changed")
    if PROBE_MODULE in sys.modules:
        raise OfflineAdmissionError(
            "powered probe canonical module alias was populated before binding"
        )
    sys.modules[PROBE_MODULE] = module
    if sys.modules.get(PROBE_MODULE) is not module:
        raise OfflineAdmissionError("powered probe canonical module alias changed")


__all__ = [
    "IMPORT_AUDIT_MODULE",
    "POWERED_EAGER_IMPORT_MODULES",
    "POWERED_IMPORT_SEED_MODULES",
    "TRANCHE2_INTEGRATION_METHODS",
    "AttemptGateError",
    "AttemptHandleSet",
    "AttemptMaterial",
    "AttemptRootSnapshot",
    "AttemptWorkspace",
    "BlockedProcess",
    "BoundaryCreateNewError",
    "CSPRNGService",
    "CapabilitySecrets",
    "ChildSupervisionOutcome",
    "CreateNewFileReceipt",
    "CreateNewJsonPublisher",
    "FallbackDecision",
    "FallbackFacts",
    "FallbackSupervisionOutcome",
    "FileIdentityProof",
    "FoundationAdmission",
    "GitWorktreeProof",
    "HostService",
    "HeartbeatPump",
    "ImportRevalidation",
    "LauncherService",
    "LeaseService",
    "LeaseReleaseOutcome",
    "LiveOrchestrationServices",
    "LiveIntegrationUnavailable",
    "OfflineAdmission",
    "OfflineAdmissionError",
    "OfflineAdmissionService",
    "OrchestrationPhaseError",
    "OrchestrationRecordValidators",
    "OrchestrationResult",
    "PartialPublicationError",
    "PathProof",
    "PriorAttemptObservation",
    "ProbeArguments",
    "ProbeOrchestrator",
    "PoweredCalibrationProbeError",
    "ProcessService",
    "PublicationError",
    "PublicationLatch",
    "QpcService",
    "PostReleaseService",
    "PortService",
    "SecureCreateNewService",
    "SecureBoundaryError",
    "SecureDirectoryReceipt",
    "StableJsonProof",
    "SpawnService",
    "SplitPublications",
    "SupervisionService",
    "TerminalDecision",
    "TopologyService",
    "TrainingAttestationService",
    "WrapperAbsoluteDeadlines",
    "WrapperLedger",
    "WrapperTimeline",
    "admit_offline",
    "artifact_ref",
    "build_argument_parser",
    "build_attempt_material",
    "decide_fallback",
    "decide_terminal",
    "derive_child_argv",
    "derive_cleanup_argv",
    "derive_phase_deadline",
    "derive_poison_required",
    "derive_terminal_parent_deadline",
    "derive_wrapper_absolute_deadlines",
    "generate_capability_secrets",
    "main",
    "parse_arguments",
    "run_initial_import_inventory_audit",
    "validate_attempt_gate",
    "validate_live_orchestration_services",
]


if __name__ == "__main__":  # pragma: no cover - production entry point
    _bind_production_main_module_alias()
    raise SystemExit(main())
